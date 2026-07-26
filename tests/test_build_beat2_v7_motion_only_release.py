import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.build_beat2_v7_motion_only_release import build_release
from upper_body_skeleton.retarget_v2_18d import CONTRACT_VERSION, JOINT_ORDER_18D
from upper_body_skeleton.ula_v2_18d_head import (
    FORMAL_SEMANTIC_SUPERVISION_MASKS,
    MOTION_ONLY_EPISODE_CONTRACT,
    load_18d_episodes,
    write_contract_csv,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, dict]:
    motion = tmp_path / "motion.csv"
    write_contract_csv(motion, np.zeros((3, 18), dtype=np.float32), fps=30.0)
    gates = {
        "axis_direction_pass": True,
        "collision_pass": True,
        "head_continuity_pass": True,
        "head_direction_pass": True,
        "head_joint_limits_pass": True,
        "head_velocity_pass": True,
        "joint_limits_pass": True,
        "passed": True,
        "target_fit_pass": True,
        "velocity_pass": True,
    }
    retarget_segment = {
        "representation": "native_variable_length_semantic_event_retimed_30hz_v1",
        "source_start_frame": 10,
        "source_end_frame_exclusive": 13,
        "source_frame_count": 3,
        "source_frame_coverage_sec": 0.1,
        "output_frame_count": 3,
        "output_sample_span_sec": 2 / 30,
        "output_frame_coverage_sec": 0.1,
        "fps": 30.0,
        "retimed": False,
        "cropped": False,
        "planner_duration_field": "output_sample_span_sec",
        "source_boundary_duration_field": "source_frame_coverage_sec",
        "legacy_quality_duration_sec_role": (
            "output_frame_coverage_compatibility_only_not_planner_target"
        ),
        "sha256": "a" * 64,
    }
    quality = tmp_path / "quality.json"
    quality.write_text(
        json.dumps(
            {
                "output_contract": CONTRACT_VERSION,
                "action_dim": 18,
                "frames": 3,
                "fps": 30.0,
                "joint_order": list(JOINT_ORDER_18D),
                "quality_gate": gates,
                "outputs": {"safe_csv": str(motion)},
                "retarget_segment": retarget_segment,
                "source_window_frames": 3,
            }
        ),
        encoding="utf-8",
    )
    record = {
        "clip_id": "sample_001",
        "task_id": "sample_001",
        "status": "passed",
        "frames": 3,
        "fps": 30.0,
        "quality_gate": gates,
        "quality_json": str(quality),
        "quality_json_sha256": _sha256(quality),
        "safe_csv": str(motion),
        "safe_csv_sha256": _sha256(motion),
        "retarget_segment": retarget_segment,
        "training_segment": {
            "representation": "native_variable_length_semantic_clip_v1",
            "start_frame": 10,
            "end_frame_exclusive": 13,
            "frame_count": 3,
            "fixed_window_sec": None,
        },
        "speaker_key": "speaker_01",
        "source_group_key": "source_01",
        "source_clip_id": "source_01",
        "canonical_prompt": {"en": "Metadata only, never used as supervision."},
        "prompt": "Metadata only, never used as supervision.",
        "annotation_kind": "official_gesture_semantic_event",
        "semantic_event": {"category": "metaphoric"},
        "canonical_action": "official_gesture_category:metaphoric",
        "canonical_action_role": "official_category_metadata_split_key_only",
        "semantic_mapping_status": "official_category_verified_metadata_only",
        "official_category_verified": True,
        "official_category_conditioning_enabled": False,
        "official_category_role": "verified_metadata_split_and_evaluation_only",
        "official_category_condition_channel": None,
        "official_category_loss": None,
        "robot_observable_motion_form": "candidate_unreviewed",
        "communicative_intent": "candidate_unreviewed",
        "canonical_prompt_role": "coarse_category_only",
        "semantic_supervision_masks": dict(FORMAL_SEMANTIC_SUPERVISION_MASKS),
        "behavior_id": "Behavior.InteractPresence",
        "behavior_review_status": "candidate_unreviewed",
        "behavior_supervision_mask": False,
        "emotion_id": "fear",
        "emotion_review_status": "official_protocol_confirmed",
        "emotion_supervision_mask": False,
        "source_emotion_label_verified": True,
        "official_emotion_conditioning_enabled": False,
        "emotion_supervision_role": "disabled_pending_robot_affect_review",
        "official_emotion_condition_channel": None,
        "official_emotion_loss": None,
        "affect_observable_review_status": "candidate_unreviewed",
        "affect_observable_supervision_mask": False,
    }
    manifest = tmp_path / "passed.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return manifest, record


def test_release_loads_as_formal_motion_only_episode(tmp_path):
    source, _ = _fixture(tmp_path)
    output = tmp_path / "release"
    report = build_release(
        passed_manifest=source,
        output_dir=output,
        verify_artifacts=True,
    )
    episodes = load_18d_episodes(manifest=output / "train_ready.jsonl")

    assert report["scale"]["train_ready_clips"] == 1
    assert len(episodes) == 1
    assert episodes[0]["accepted_for_training"] is True
    assert episodes[0]["formal_episode_contract"] == MOTION_ONLY_EPISODE_CONTRACT
    assert episodes[0]["behavior_supervision_mask"] is False
    assert episodes[0]["emotion_conditioning_mask"] is False
    assert all(value is False for value in episodes[0]["semantic_supervision_masks"].values())


def test_release_rejects_unmasked_text_supervision(tmp_path):
    source, record = _fixture(tmp_path)
    record["semantic_supervision_masks"]["prompt_text"] = True
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="semantic supervision masks"):
        build_release(passed_manifest=source, output_dir=tmp_path / "release")


def test_motion_only_manifest_requires_bound_release_report(tmp_path):
    source, _ = _fixture(tmp_path)
    output = tmp_path / "release"
    build_release(passed_manifest=source, output_dir=output)
    (output / "motion_only_release_report.json").unlink()

    with pytest.raises(ValueError, match="requires sibling"):
        load_18d_episodes(manifest=output / "train_ready.jsonl")
