import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from upper_body_skeleton.motion_latent import stratified_episode_split
from upper_body_skeleton.pt_v2_evaluate import (
    EvaluationPrompt,
    _load_fixed_label_condition_builder,
    aggregate_episode_metrics,
    best_of_k_metrics,
    evaluate_reference_rows,
    free_length_best_of_k_metrics,
    select_evaluation_reference_rows,
    trajectory_kinematics,
    trajectory_pair_metrics,
)
from upper_body_skeleton.retarget_v2 import JOINT_ORDER


def _trajectory(frames=5, value=0.0):
    return np.full((frames, len(JOINT_ORDER)), value, dtype=np.float32)


def test_trajectory_kinematics_reports_joint_ranges_and_left_right_activity():
    trajectory = _trajectory(frames=4)
    trajectory[:, 3:9] = np.arange(4, dtype=np.float32)[:, None] * 0.5

    metrics = trajectory_kinematics(trajectory, fps=2.0)

    assert metrics["velocity_rms_rad_s"] == pytest.approx(np.sqrt(6.0 / 15.0))
    assert metrics["acceleration_rms_rad_s2"] == pytest.approx(0.0)
    assert metrics["per_joint_range_rad"]["joint_lShoulderPitch"] == pytest.approx(1.5)
    assert metrics["per_joint_range_rad"]["joint_rShoulderPitch"] == pytest.approx(0.0)
    activity = metrics["left_right_activity_rad_s"]
    assert activity["left"] == pytest.approx(1.0)
    assert activity["right"] == pytest.approx(0.0)
    assert activity["asymmetry"] == pytest.approx(1.0)


def test_pair_metrics_and_best_of_k_use_position_rmse_for_selection():
    reference = _trajectory(frames=5)
    far = _trajectory(frames=5, value=0.5)
    exact = reference.copy()

    result = best_of_k_metrics(
        [
            {"seed": 7, "trajectory": far},
            {"seed": 17, "trajectory": exact},
        ],
        reference,
        fps=30.0,
    )

    assert result["k"] == 2
    assert result["best_seed"] == 17
    assert result["position_rmse_best_of_k_rad"] == 0.0
    selected = result["selected_metrics"]
    assert selected["velocity_rmse_rad_s"] == 0.0
    assert selected["acceleration_rmse_rad_s2"] == 0.0
    assert selected["range_rmse_rad"] == 0.0
    assert set(selected["per_joint_range_rad"]) == set(JOINT_ORDER)


def test_free_length_metrics_report_native_duration_before_phase_alignment():
    reference = _trajectory(frames=5)
    generated = _trajectory(frames=8)
    result = free_length_best_of_k_metrics(
        [{"seed": 7, "trajectory": generated, "semantic_label_match": True}],
        reference,
        fps=30.0,
    )

    assert result["selected_generated_frame_count"] == 8
    assert result["selected_generated_sample_span_sec"] == pytest.approx(7 / 30)
    assert result["selected_sample_span_error_sec"] == pytest.approx(3 / 30)
    assert result["selected_frame_count_error"] == 3
    assert result["selected_metrics"]["position_rmse_rad"] == 0.0
    assert result["comparison_alignment"].startswith("phase_resample")


def test_pair_metrics_measure_velocity_and_acceleration_in_physical_units():
    reference = _trajectory(frames=4)
    generated = _trajectory(frames=4)
    generated[:, 0] = np.asarray([0.0, 0.5, 1.5, 3.0], dtype=np.float32)

    metrics = trajectory_pair_metrics(generated, reference, fps=2.0)

    expected_velocity = np.asarray([1.0, 2.0, 3.0])
    expected_acceleration = np.asarray([2.0, 2.0])
    assert metrics["velocity_rmse_rad_s"] == pytest.approx(
        np.sqrt(np.square(expected_velocity).sum() / (3 * len(JOINT_ORDER)))
    )
    assert metrics["acceleration_rmse_rad_s2"] == pytest.approx(
        np.sqrt(np.square(expected_acceleration).sum() / (2 * len(JOINT_ORDER)))
    )


def test_aggregate_reports_global_label_and_per_joint_summaries():
    reference = _trajectory(frames=4)
    rows = []
    for episode_index, emotion_id, value in ((1, "happy", 0.1), (2, "sad", 0.2)):
        rows.append(
            {
                "episode_index": episode_index,
                "behavior_id": "Behavior.GreetingOwner01",
                "emotion_id": emotion_id,
                "best_of_k": best_of_k_metrics(
                    [{"seed": 7, "trajectory": _trajectory(frames=4, value=value)}],
                    reference,
                    fps=30.0,
                ),
            }
        )

    aggregate = aggregate_episode_metrics(rows)

    assert aggregate["reference_count"] == 2
    assert aggregate["candidate_count"] == 2
    assert aggregate["position_rmse_best_of_k_rad"]["mean"] == pytest.approx(0.15)
    assert len(aggregate["by_label"]) == 2
    assert set(aggregate["per_joint_range_absolute_error_selected_rad"]) == set(JOINT_ORDER)
    json.dumps(aggregate)


def test_reference_selector_supports_emotion_only_filter_and_balanced_limit(tmp_path):
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "meta").mkdir(parents=True)
    behaviors = ("Behavior.GreetingOwner01", "Behavior.GreetingOwner04")
    rows = []
    for behavior_index, behavior_id in enumerate(behaviors):
        for local_index in range(10):
            episode_index = behavior_index * 10 + local_index
            rows.append(
                {
                    "episode_index": episode_index,
                    "sample_id": f"sample_{episode_index}",
                    "language_instruction": f"prompt {behavior_id}",
                    "behavior_id": behavior_id,
                    "emotion_id": "happy",
                }
            )
    pq.write_table(
        pa.Table.from_pylist(rows), dataset_dir / "meta" / "semantic_index.parquet"
    )
    episodes = [
        {
            "episode_index": row["episode_index"],
            "meta": {
                "behavior_id": row["behavior_id"],
                "emotion_id": row["emotion_id"],
            },
        }
        for row in rows
    ]
    train, validation, test = stratified_episode_split(episodes, seed=7)
    checkpoint_path = tmp_path / "motion.pt"
    torch.save(
        {
            "config": {"seed": 7},
            "split_episode_indices": {
                "train": [row["episode_index"] for row in train],
                "validation": [row["episode_index"] for row in validation],
                "test": [row["episode_index"] for row in test],
            },
        },
        checkpoint_path,
    )

    selected = select_evaluation_reference_rows(
        dataset_dir,
        checkpoint_path,
        split="test",
        emotion_ids=["happy"],
    )

    assert len(selected) == 2
    assert {row["behavior_id"] for row in selected} == set(behaviors)
    limited = select_evaluation_reference_rows(
        dataset_dir,
        checkpoint_path,
        split="test",
        emotion_ids=["happy"],
        max_references=1,
    )
    assert len(limited) == 1


def test_reference_evaluator_uses_fixed_seeds_and_caches_identical_label_prompts():
    class FakeGenerator:
        def __init__(self):
            self.calls = []

        def infer(self, text, **kwargs):
            self.calls.append((text, kwargs))
            value = 0.0 if kwargs["seed"] == 17 else 0.5
            frames = 7 if kwargs["frames"] is None else kwargs["frames"]
            return SimpleNamespace(
                behavior_id=kwargs["behavior_id"],
                emotion_id=kwargs["emotion_id"],
                trajectory=_trajectory(frames, value=value),
                predicted_duration_sec=(frames - 1) / 30.0,
            )

    rows = [
        {
            "episode_index": episode_index,
            "sample_id": f"sample_{episode_index}",
            "language_instruction": "same fixed-label prompt",
            "behavior_id": "Behavior.GreetingOwner01",
            "emotion_id": "happy",
        }
        for episode_index in (1, 2)
    ]
    references = {
        episode_index: {"actions": _trajectory(frames=5)} for episode_index in (1, 2)
    }
    generator = FakeGenerator()

    results = evaluate_reference_rows(
        generator,
        rows,
        references,
        fps=30.0,
        seeds=(7, 17),
        condition_builder=object(),
        prompt_provider=lambda row: EvaluationPrompt(
            text=row["language_instruction"], source="fixture"
        ),
    )

    assert len(generator.calls) == 4
    assert [call[1]["seed"] for call in generator.calls] == [7, 7, 17, 17]
    assert [call[1]["frames"] for call in generator.calls] == [None, 5, None, 5]
    assert [row["best_of_k"]["best_seed"] for row in results] == [17, 17]
    assert all(row["primary_non_oracle"] is row["best_of_k"] for row in results)
    assert all(
        row["primary_non_oracle"]["selected_generated_frame_count"] == 7
        for row in results
    )
    assert all(
        row["secondary_oracle_length"]["generation_length_policy"].startswith(
            "reference_frame_count"
        )
        for row in results
    )
    assert all(row["reference_sample_span_sec"] == pytest.approx(4 / 30) for row in results)
    assert all(row["prompt"]["source"] == "fixture" for row in results)


def test_lora_conditioned_evaluation_requires_recorded_lora_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="requires the Qwen Motion LoRA"):
        _load_fixed_label_condition_builder(
            tmp_path / "unused-semantic.pt",
            generator_checkpoint={
                "v2_contracts": {"text_motion_latent": {"contract_type": "fixture"}}
            },
            dataset_dir=tmp_path,
        )


@pytest.mark.parametrize("bad_fps", [0.0, -1.0, float("inf")])
def test_kinematics_rejects_invalid_fps(bad_fps):
    with pytest.raises(ValueError, match="fps"):
        trajectory_kinematics(_trajectory(), fps=bad_fps)
