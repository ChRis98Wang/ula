import json

import imageio.v2 as imageio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from upper_body_skeleton.mujoco_playback import compose_labeled_comparison_frame
from upper_body_skeleton.motion_latent import stratified_episode_split
from upper_body_skeleton.pt_dataset_mujoco_compare import (
    _prioritize_diverse_rows,
    inspect_comparison_video,
    select_dataset_reference_rows,
    trajectory_comparison_metrics,
    validate_dataset_contract,
)
from upper_body_skeleton.retarget_v2 import JOINT_ORDER


def test_compose_comparison_places_network_left_and_reference_right():
    network = np.zeros((32, 40, 3), dtype=np.uint8)
    reference = np.full((32, 40, 3), 255, dtype=np.uint8)

    combined = compose_labeled_comparison_frame(network, reference, title_height=24)

    assert combined.shape == (56, 80, 3)
    assert combined[24:, :40].mean() == 0
    assert combined[24:, 42:].mean() == 255


def test_trajectory_comparison_metrics_reports_exact_match():
    values = np.linspace(0.0, 0.2, 5 * len(JOINT_ORDER), dtype=np.float32).reshape(5, len(JOINT_ORDER))

    metrics = trajectory_comparison_metrics(values, values.copy(), fps=30.0)

    assert metrics["mae_rad"] == 0.0
    assert metrics["rmse_rad"] == 0.0
    assert set(metrics["per_joint_mae_rad"]) == set(JOINT_ORDER)


def test_diverse_selection_rotates_behaviors_and_emotions():
    rows = [
        {"episode_index": 1, "behavior_id": "Behavior.A", "emotion_id": "angry"},
        {"episode_index": 2, "behavior_id": "Behavior.A", "emotion_id": "happy"},
        {"episode_index": 3, "behavior_id": "Behavior.B", "emotion_id": "angry"},
        {"episode_index": 4, "behavior_id": "Behavior.B", "emotion_id": "happy"},
        {"episode_index": 5, "behavior_id": "Behavior.C", "emotion_id": "angry"},
        {"episode_index": 6, "behavior_id": "Behavior.C", "emotion_id": "happy"},
    ]

    selected = _prioritize_diverse_rows(rows)[:3]

    assert [row["behavior_id"] for row in selected] == ["Behavior.A", "Behavior.B", "Behavior.C"]
    assert [row["emotion_id"] for row in selected] == ["angry", "happy", "angry"]


def test_dataset_reference_selection_uses_checkpoint_split_and_exact_text(tmp_path):
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "meta").mkdir(parents=True)
    rows = [
        {
            "episode_index": index,
            "sample_id": f"sample_{index}",
            "language_instruction": f"exact text {index}",
            "behavior_id": (
                "Behavior.GreetingOwner01" if index < 10 else "Behavior.GreetingOwner04"
            ),
            "emotion_id": "happy",
        }
        for index in range(20)
    ]
    pq.write_table(pa.Table.from_pylist(rows), dataset_dir / "meta" / "semantic_index.parquet")
    episodes = [
        {
            "episode_index": row["episode_index"],
            "meta": {"behavior_id": row["behavior_id"], "emotion_id": row["emotion_id"]},
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

    selected = select_dataset_reference_rows(
        dataset_dir,
        checkpoint_path,
        motion_latent_split="test",
        behavior_id="Behavior.GreetingOwner01",
        emotion_id="happy",
    )

    expected_id = test[0]["episode_index"]
    assert selected == [next(row for row in rows if row["episode_index"] == expected_id)]

    selected_without_label_filter = select_dataset_reference_rows(
        dataset_dir,
        checkpoint_path,
        motion_latent_split="test",
        count=1,
    )
    assert selected_without_label_filter == selected

    selected_behavior_set = select_dataset_reference_rows(
        dataset_dir,
        checkpoint_path,
        motion_latent_split="test",
        behavior_id=["Behavior.GreetingOwner01", "Behavior.GreetingOwner04"],
        count=2,
    )
    assert {row["behavior_id"] for row in selected_behavior_set} == {
        "Behavior.GreetingOwner01",
        "Behavior.GreetingOwner04",
    }


def test_dataset_contract_rejects_wrong_joint_order(tmp_path):
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "meta").mkdir(parents=True)
    info = {
        "fps": 30,
        "features": {"observation.state": {"names": list(reversed(JOINT_ORDER))}},
    }
    (dataset_dir / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")

    with pytest.raises(ValueError, match="joint order"):
        validate_dataset_contract(dataset_dir)


def test_video_check_verifies_both_panes_are_nonblank(tmp_path):
    path = tmp_path / "comparison.mp4"
    rng = np.random.default_rng(7)
    with imageio.get_writer(path, fps=30, codec="libx264", macro_block_size=8) as writer:
        for _ in range(3):
            left = rng.integers(0, 128, size=(32, 32, 3), dtype=np.uint8)
            right = rng.integers(128, 256, size=(32, 32, 3), dtype=np.uint8)
            writer.append_data(compose_labeled_comparison_frame(left, right, title_height=24))

    result = inspect_comparison_video(path, expected_frames=3, title_height=24)

    assert result["nonblank"] is True
    assert result["decoded_frames"] == 3
    assert result["decoded_shape"] == [56, 64, 3]


def test_video_check_rejects_labels_over_blank_mujoco_panes(tmp_path):
    path = tmp_path / "blank_comparison.mp4"
    blank = np.zeros((32, 32, 3), dtype=np.uint8)
    with imageio.get_writer(path, fps=30, codec="libx264", macro_block_size=8) as writer:
        for _ in range(3):
            writer.append_data(compose_labeled_comparison_frame(blank, blank, title_height=24))

    with pytest.raises(ValueError, match="blank MuJoCo pane"):
        inspect_comparison_video(path, expected_frames=3, title_height=24)
