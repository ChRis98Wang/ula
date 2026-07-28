from __future__ import annotations

from copy import deepcopy
import csv
import hashlib

import numpy as np
import pytest

from upper_body_skeleton.retarget_v2_18d import JOINT_ORDER_18D
from upper_body_skeleton.robot_observable_motion_realizations import (
    build_conversational_realization_annotation,
)
from upper_body_skeleton.ula_training import (
    KIMODO_CONDITION_DIM,
    KIMODO_V2_CONDITION_DIM,
)
from upper_body_skeleton.ula_v2_conversational_realization_episode import (
    FORMAL_ELIGIBILITY_MODE,
    FORMAL_EPISODE_CONTRACT,
    PROMPT_TEXT_PROVENANCE,
    TRAINING_SEGMENT_REPRESENTATION,
    build_conversational_realization_v9_condition_vectors,
    load_conversational_realization_v9_episodes,
    validate_conversational_realization_v9_episode,
)


def _episode() -> dict:
    style = {
        "arm_amplitude": "moderate",
        "laterality": "both",
        "pace": "steady",
        "head_engagement": "engaged",
        "style_controls": {"amplitude": 0.25, "tempo": -0.1, "energy": 0.4},
    }
    realization = build_conversational_realization_annotation(
        {
            "dataset": "BEAT2",
            "annotation_kind": "official_gesture_semantic_event",
            "interaction_scope": "human_co_speech_interaction",
            "status": "passed",
            "quality_gate": {"passed": True},
        },
        style,
    )
    actions = np.arange(7 * 18, dtype=np.float32).reshape(7, 18) / 1000
    source_hash = "a" * 64
    realization_hash = "b" * 64
    trajectory_hash = "c" * 64
    admission = {
        "contract": FORMAL_EPISODE_CONTRACT,
        "trajectory_sha256": trajectory_hash,
        "source_record_sha256": source_hash,
        "realization_record_sha256": realization_hash,
        "motion_realization_ontology_sha256": realization[
            "motion_realization_ontology_sha256"
        ],
        "motion_realization_id": "conversational_gesturing",
        "training_channel_masks": {
            "motion": True,
            "motion_realization": True,
            "primary_intent": False,
            "emotion": False,
            "audio": False,
        },
    }
    prompt = realization["motion_realization_prompt"]["en"]
    return {
        "formal_episode_contract": FORMAL_EPISODE_CONTRACT,
        "eligibility_mode": FORMAL_ELIGIBILITY_MODE,
        "accepted_for_training": True,
        "native_variable_length": True,
        "clip_id": "ordinary_001",
        "dataset_source": "beat2_official_semantic_event_training_pool_v8_expanded",
        "source_clip_id": "speaker_take",
        "speaker_key": "speaker",
        "source_group_key": "BEAT2/speaker_take",
        "actions": actions,
        "action_dim_mask": np.ones(18, dtype=np.bool_),
        "condition": None,
        "fps": 30.0,
        "duration_sec": 6 / 30,
        "trajectory_sha256": trajectory_hash,
        "source_manifest_sha256": "d" * 64,
        "source_record_sha256": source_hash,
        "realization_manifest_sha256": "e" * 64,
        "realization_record_sha256": realization_hash,
        "retarget_qc_passed": True,
        "quality_gate": {
            "joint_limits_pass": True,
            "velocity_pass": True,
            "head_joint_limits_pass": True,
            "head_velocity_pass": True,
            "collision_pass": True,
            "passed": True,
        },
        "retarget_segment": {
            "source_start_frame": 10,
            "source_end_frame_exclusive": 17,
            "source_frame_count": 7,
            "output_frame_count": 7,
            "cropped": False,
            "fps": 30.0,
        },
        "training_segment": {
            "representation": TRAINING_SEGMENT_REPRESENTATION,
            "start_frame": 10,
            "end_frame_exclusive": 17,
            "frame_count": 7,
            "output_frame_count": 7,
            "fixed_window_sec": None,
            "cropped": False,
        },
        "motion_realization": realization,
        "motion_realization_supervision_mask": True,
        "motion_style": style,
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_text_provenance": PROMPT_TEXT_PROVENANCE,
        "observable_intent_id": None,
        "intent_supervision_mask": False,
        "intent_conditioning_mask": False,
        "emotion_id": None,
        "emotion_supervision_mask": False,
        "emotion_conditioning_mask": False,
        "behavior_supervision_mask": False,
        "audio_conditioning_enabled": False,
        "training_admission": admission,
    }


def test_conversational_episode_has_text_and_style_but_no_intent_or_emotion() -> None:
    episode = _episode()
    report = validate_conversational_realization_v9_episode(episode)
    assert report["motion_realization_id"] == "conversational_gesturing"
    latent = np.arange(1, 129, dtype=np.float32)
    conditions = build_conversational_realization_v9_condition_vectors(
        [episode], text_latents={episode["clip_id"]: latent}
    )
    assert conditions.shape == (1, KIMODO_V2_CONDITION_DIM)
    assert np.allclose(conditions[0, 133:136], [0.25, -0.1, 0.4])
    assert np.all(conditions[0, :133] == 0)
    assert np.isclose(np.linalg.norm(conditions[0, KIMODO_CONDITION_DIM:]), 1.0)


def test_conversational_episode_rejects_invented_primary_intent() -> None:
    episode = _episode()
    episode["observable_intent_id"] = "explain_present"
    with pytest.raises(ValueError, match="may not imply intent"):
        validate_conversational_realization_v9_episode(episode)


def test_conversational_episode_rejects_fixed_six_second_window() -> None:
    episode = _episode()
    episode["training_segment"]["fixed_window_sec"] = 6.0
    with pytest.raises(ValueError, match="fixed-window"):
        validate_conversational_realization_v9_episode(episode)


def test_conversational_manifest_loader_binds_native_18d_csv(tmp_path) -> None:
    episode = _episode()
    actions = episode.pop("actions")
    episode.pop("action_dim_mask")
    episode.pop("condition")
    csv_path = tmp_path / "ordinary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_sec", *JOINT_ORDER_18D])
        for index, row in enumerate(actions):
            writer.writerow([index / 30, *row.tolist()])
    trajectory_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    episode["trajectory_sha256"] = trajectory_hash
    episode["training_admission"]["trajectory_sha256"] = trajectory_hash
    episode["motion_18d"] = {
        "state": "passed",
        "safe_csv": str(csv_path),
        "safe_csv_sha256": trajectory_hash,
        "frames": 7,
        "csv_rows": 7,
    }
    manifest = tmp_path / "train_ready.jsonl"
    import json

    manifest.write_text(json.dumps(episode, sort_keys=True) + "\n", encoding="utf-8")
    loaded = load_conversational_realization_v9_episodes(manifest)
    assert loaded[0]["actions"].shape == (7, 18)
    assert loaded[0]["duration_sec"] == pytest.approx(6 / 30)

    changed = deepcopy(episode)
    changed["trajectory_sha256"] = "0" * 64
    manifest.write_text(json.dumps(changed, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="trajectory SHA256"):
        load_conversational_realization_v9_episodes(manifest)
