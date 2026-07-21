import json

import numpy as np
import pytest
import torch

from upper_body_skeleton.kimodo_semantics import KIMODO_BEHAVIOR_IDS, KIMODO_EMOTION_IDS
from upper_body_skeleton.motion_latent import (
    MotionMetricEncoder,
    compute_motion_normalization,
    stratified_episode_split,
)
from upper_body_skeleton.retarget_v2 import JOINT_LIMITS, JOINT_ORDER
from upper_body_skeleton.ula_training import KIMODO_CONDITION_DIM, KIMODO_V2_CONDITION_DIM, condition_vector
from upper_body_skeleton.ula_v2_conditioning import (
    build_active_window_contract,
    clean_joint_trajectory,
    extract_active_motion_window,
    extract_style_features,
    load_validated_episode_splits,
    prepare_v2_episode_splits,
    trim_episode,
)


def _episodes_and_checkpoint(tmp_path):
    semantic_episodes = [
        {
            "episode_index": index,
            "meta": {
                "behavior_id": "Behavior.GreetingOwner01",
                "emotion_id": "happy",
            },
        }
        for index in range(6)
    ]
    train_rows, validation_rows, test_rows = stratified_episode_split(semantic_episodes, seed=13)
    train_ids = {row["episode_index"] for row in train_rows}
    episodes = []
    phase = np.linspace(0.0, 2.0 * np.pi, 15, dtype=np.float32)
    for index, semantic in enumerate(semantic_episodes):
        actions = np.zeros((15, len(JOINT_ORDER)), dtype=np.float32)
        amplitude = 0.04 + 0.02 * index if index in train_ids else 1.2 + index
        actions[:, JOINT_ORDER.index("joint_lShoulderPitch")] = amplitude * np.sin(phase)
        actions[:, JOINT_ORDER.index("joint_lElbow")] = amplitude * 0.5 * np.cos(phase)
        actions[:, JOINT_ORDER.index("joint_rShoulderPitch")] = amplitude * 0.25 * np.cos(phase)
        meta = {
            **semantic["meta"],
            "language_instruction": "开心地挥手",
            "intent": "greeting",
            "observed_affect": "excited",
            "motion_style": "energetic",
            "semantic_gesture": "waving",
            "fps": 30.0,
        }
        episodes.append(
            {
                "episode_index": index,
                "actions": actions,
                "condition": condition_vector(meta),
                "meta": meta,
                "fps": 30.0,
            }
        )

    train_episodes = [episode for episode in episodes if episode["episode_index"] in train_ids]
    stats = compute_motion_normalization(train_episodes)
    torch.manual_seed(19)
    model = MotionMetricEncoder(action_dim=len(JOINT_ORDER), latent_dim=128, hidden_dim=16)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "normalization": stats,
        "behavior_ids": list(KIMODO_BEHAVIOR_IDS),
        "emotion_ids": list(KIMODO_EMOTION_IDS),
        "config": {
            "action_dim": len(JOINT_ORDER),
            "latent_dim": 128,
            "hidden_dim": 16,
            "seed": 13,
        },
        "split_episode_indices": {
            "train": [row["episode_index"] for row in train_rows],
            "validation": [row["episode_index"] for row in validation_rows],
            "test": [row["episode_index"] for row in test_rows],
        },
    }
    checkpoint_path = tmp_path / "motion_latent_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    return episodes, checkpoint, checkpoint_path


def test_clean_joint_trajectory_enforces_bounds_and_velocity_without_default_smoothing():
    actions = np.zeros((7, len(JOINT_ORDER)), dtype=np.float32)
    actions[1::2] = 100.0
    actions[2::2] = -100.0

    cleaned = clean_joint_trajectory(actions, fps=30.0, max_velocity_rad_s=3.0, smooth_window=1)

    assert cleaned.shape == actions.shape
    assert np.isfinite(cleaned).all()
    assert max(float(np.abs(np.diff(cleaned, axis=0)).max()) * 30.0, 0.0) <= 3.0 + 1e-5
    for index, joint in enumerate(JOINT_ORDER):
        lower, upper = JOINT_LIMITS[joint]
        assert cleaned[:, index].min() >= lower - 1e-6
        assert cleaned[:, index].max() <= upper + 1e-6


def test_active_window_trims_short_motion_and_preserves_static_context():
    fps = 30.0
    actions = np.zeros((180, len(JOINT_ORDER)), dtype=np.float32)
    active_values = 0.25 * np.sin(np.linspace(0.0, 2.0 * np.pi, 31, dtype=np.float32))
    actions[60:91, JOINT_ORDER.index("joint_lShoulderPitch")] = active_values
    contract = build_active_window_contract()

    window = extract_active_motion_window(actions, fps=fps, active_window_contract=contract)
    trimmed = trim_episode(
        {"episode_index": 1, "actions": actions, "fps": fps, "meta": {}},
        active_window_contract=contract,
    )

    assert window["near_static"] is False
    assert 0 < window["trim_start"] < window["active_start"]
    assert window["active_end"] < window["trim_end"] < len(actions)
    assert trimmed["original_frame_count"] == len(actions)
    assert trimmed["frame_count"] == window["trim_end"] - window["trim_start"]
    assert trimmed["duration_sec"] == pytest.approx(trimmed["frame_count"] / fps)
    assert trimmed["effective_duration_sec"] == pytest.approx(
        (window["active_end"] - window["active_start"]) / fps
    )


def test_active_window_keeps_full_span_motion():
    fps = 30.0
    actions = np.zeros((120, len(JOINT_ORDER)), dtype=np.float32)
    actions[:, JOINT_ORDER.index("joint_rShoulderPitch")] = 0.2 * np.sin(
        np.linspace(0.0, 8.0 * np.pi, len(actions), dtype=np.float32)
    )

    window = extract_active_motion_window(actions, fps=fps)

    assert window["near_static"] is False
    assert window["trim_start"] == 0
    assert window["trim_end"] == len(actions)


def test_active_window_does_not_trim_near_static_motion():
    fps = 30.0
    actions = np.zeros((90, len(JOINT_ORDER)), dtype=np.float32)
    actions[:, JOINT_ORDER.index("joint_lWristRoll")] = 1e-4 * np.sin(
        np.linspace(0.0, 2.0 * np.pi, len(actions), dtype=np.float32)
    )

    window = extract_active_motion_window(actions, fps=fps)

    assert window["near_static"] is True
    assert window["active_start"] == 0
    assert window["active_end"] == len(actions)
    assert window["trim_start"] == 0
    assert window["trim_end"] == len(actions)


def test_load_validated_episode_splits_rejects_overlap(tmp_path):
    episodes, checkpoint, checkpoint_path = _episodes_and_checkpoint(tmp_path)
    splits, contract = load_validated_episode_splits(checkpoint_path, episodes)
    ids = [{episode["episode_index"] for episode in splits[name]} for name in ("train", "validation", "test")]

    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])
    assert set.union(*ids) == {episode["episode_index"] for episode in episodes}
    assert contract["episode_count"] == len(episodes)
    assert len(contract["sha256"]) == 64

    checkpoint["split_episode_indices"]["validation"].append(
        checkpoint["split_episode_indices"]["train"][0]
    )
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match="overlap"):
        load_validated_episode_splits(checkpoint_path, episodes)


def test_prepare_v2_uses_train_only_style_stats_and_builds_finite_264d_conditions(tmp_path):
    episodes, checkpoint, checkpoint_path = _episodes_and_checkpoint(tmp_path)

    train, validation, test, contracts = prepare_v2_episode_splits(
        None,
        checkpoint_path,
        episodes=episodes,
        device="cpu",
    )

    train_ids = sorted(checkpoint["split_episode_indices"]["train"])
    assert contracts["style"]["fit_split"] == "train"
    assert contracts["style"]["fit_episode_indices"] == train_ids
    expected_features = np.stack(
        [extract_style_features(episode["actions"], fps=episode["fps"]) for episode in train]
    )
    assert np.allclose(contracts["style"]["mean"], expected_features.mean(axis=0), atol=1e-7)
    assert np.allclose(
        contracts["style"]["std"],
        np.maximum(expected_features.std(axis=0), contracts["style"]["eps"]),
        atol=1e-7,
    )
    assert max(np.max(np.abs(episode["actions"])) for episode in validation + test) > max(
        np.max(np.abs(episode["actions"])) for episode in train
    )

    prototype = np.asarray(contracts["motion_prototypes"]["groups"][0]["prototype"], dtype=np.float32)
    assert prototype.shape == (128,)
    assert np.isclose(np.linalg.norm(prototype), 1.0, atol=1e-5)
    for episode in train + validation + test:
        assert episode["condition"].shape == (KIMODO_V2_CONDITION_DIM,)
        assert episode["condition"].shape[0] == KIMODO_CONDITION_DIM + 128
        assert np.isfinite(episode["condition"]).all()
        assert np.array_equal(episode["condition"][-128:], prototype)
        assert np.array_equal(episode["condition"][KIMODO_CONDITION_DIM - 3 : KIMODO_CONDITION_DIM], episode["style_controls"])
    json.dumps(contracts, allow_nan=False)


def test_prepare_v2_is_deterministic(tmp_path):
    episodes, _, checkpoint_path = _episodes_and_checkpoint(tmp_path)

    first = prepare_v2_episode_splits(None, checkpoint_path, episodes=episodes, device="cpu")
    second = prepare_v2_episode_splits(None, checkpoint_path, episodes=list(reversed(episodes)), device="cpu")

    assert first[3]["sha256"] == second[3]["sha256"]
    for first_split, second_split in zip(first[:3], second[:3]):
        assert [episode["episode_index"] for episode in first_split] == [
            episode["episode_index"] for episode in second_split
        ]
        for first_episode, second_episode in zip(first_split, second_split):
            assert np.array_equal(first_episode["actions"], second_episode["actions"])
            assert np.array_equal(first_episode["condition"], second_episode["condition"])


def test_prepare_v2_preserves_variable_native_lengths_and_encodes_prototypes(tmp_path):
    episodes, _, checkpoint_path = _episodes_and_checkpoint(tmp_path)
    source_lengths = [45, 57, 68, 81, 94, 109]
    for episode, frame_count in zip(episodes, source_lengths):
        actions = np.zeros((frame_count, len(JOINT_ORDER)), dtype=np.float32)
        actions[:, JOINT_ORDER.index("joint_lShoulderPitch")] = 0.15 * np.sin(
            np.linspace(0.0, 6.0 * np.pi, frame_count, dtype=np.float32)
        )
        episode["actions"] = actions

    train, validation, test, contracts = prepare_v2_episode_splits(
        None,
        checkpoint_path,
        episodes=episodes,
        device="cpu",
    )

    prepared = train + validation + test
    assert sorted(episode["original_frame_count"] for episode in prepared) == source_lengths
    assert len({episode["frame_count"] for episode in prepared}) > 1
    assert all(episode["frame_count"] == episode["actions"].shape[0] for episode in prepared)
    assert contracts["duration"]["fixed_frame_count"] is None
    assert contracts["duration"]["trajectory_representation"] == "native_trimmed_frames"
    assert contracts["motion_prototypes"]["source_duration_policy"] == "native_trimmed_variable_length"
    assert contracts["active_window"]["near_static_policy"] == "preserve_full_trajectory"
