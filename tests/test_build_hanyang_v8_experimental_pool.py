from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.build_hanyang_v8_experimental_pool import (
    POOL_ROW_KIND,
    build_experimental_pool,
)
from upper_body_skeleton.data_source_registry import (
    EMOTION_CRITIC_ROLE,
    HANYANG_EMOTIONAL_BODY_SOURCE_ID,
)
from upper_body_skeleton.hanyang_emotion_retarget import (
    DATASET_REVISION,
    SOURCE_FPS,
    SOURCE_FRAMES,
    json_hash,
    sha256_file,
    stable_json,
)
from upper_body_skeleton.hanyang_expanded_training import (
    HANYANG_ACTION_ORDER_18D,
    HANYANG_P7_ORDER,
    PERMANENTLY_UNOBSERVED_DOF_INDICES,
)


def _record_hash(value: dict) -> str:
    return json_hash(
        {key: item for key, item in value.items() if key != "record_sha256"}
    )


def _write_fixture(root: Path, *, confidence_missing_value: float = 0.0) -> Path:
    retarget_root = root / "retarget_v1"
    clip_dir = retarget_root / "clips" / "1_1_1_1"
    clip_dir.mkdir(parents=True)
    trajectory = clip_dir / "1_1_1_1_source_faithful_partial_18d.csv"
    values = np.zeros((SOURCE_FRAMES, len(HANYANG_ACTION_ORDER_18D)))
    np.savetxt(
        trajectory,
        values,
        delimiter=",",
        header=",".join(HANYANG_ACTION_ORDER_18D),
        comments="",
        fmt="%.8f",
    )
    confidence = np.ones_like(values, dtype=np.float32)
    confidence[:, PERMANENTLY_UNOBSERVED_DOF_INDICES] = confidence_missing_value
    confidence_path = clip_dir / "observation_confidence_18d.npy"
    np.save(confidence_path, confidence, allow_pickle=False)
    distribution = {emotion: 0.0 for emotion in HANYANG_P7_ORDER}
    distribution["happy"] = 1.0
    all_gates = {
        "collision_pass": True,
        "fixed_150_frames_30hz_pass": True,
        "head_tilt_proxy_pass": True,
        "joint_limits_pass": True,
        "limb_direction_pass": True,
        "passed": True,
        "per_joint_velocity_pass": True,
        "retime_factor_exactly_one_pass": True,
        "saturation_pass": True,
        "source_geometry_pass": True,
        "target_fit_pass": True,
    }
    quality = {
        "clip_id": "hanyang:1_1_1_1",
        "dataset_id": HANYANG_EMOTIONAL_BODY_SOURCE_ID,
        "participant_id": 1,
        "fixed_split_assignment": "train",
        "source_frames": SOURCE_FRAMES,
        "source_fps": SOURCE_FPS,
        "source_sha256": "1" * 64,
        "source_group_key": "hanyang:participant:01",
        "emotion_id": "happy",
        "action_dim": len(HANYANG_ACTION_ORDER_18D),
        "action_dim_mask": [
            index not in PERMANENTLY_UNOBSERVED_DOF_INDICES
            for index in range(len(HANYANG_ACTION_ORDER_18D))
        ],
        "quality_gate": all_gates,
        "smoothing": {
            "retime_factor": 1.0,
            "retimed": False,
            "smoothing_window": 5,
            "endpoint_policy": (
                "source_first_and_last_frames_preserved_no_terminal_hold"
            ),
        },
        "outputs": {
            "source_faithful_partial_18d_csv": str(trajectory),
            "source_faithful_partial_18d_csv_sha256": sha256_file(trajectory),
            # These exist in the quality audit only and must not enter the pool.
            "raw_csv": str(clip_dir / "1_1_1_1_raw_18d.csv"),
            "deployment_safe_partial_18d_csv": str(
                clip_dir / "1_1_1_1_deployment_safe_partial_18d.csv"
            ),
        },
        "per_frame_observation_confidence": {
            "path": str(confidence_path),
            "sha256": sha256_file(confidence_path),
            "shape": [SOURCE_FRAMES, len(HANYANG_ACTION_ORDER_18D)],
        },
        "trajectory": {
            "rms_jerk_rad_s3": 1.0,
            "max_acceleration_rad_s2": 1.0,
            "max_velocity_limit_ratio": 0.5,
        },
        "limb_target_error_p95_m": 0.001,
        "limb_direction_error_p95_deg": 1.0,
        "upper_body_collision_frame_rate": 0.0,
        "emotion_evaluation": {
            "soft_emotion_distribution": distribution,
            "rater_coverage_pass": True,
            "intended_majority_agrees": True,
            "intended_share": 1.0,
        },
    }
    quality["record_sha256"] = _record_hash(quality)
    quality_path = clip_dir / "quality.json"
    quality_path.write_text(
        json.dumps(quality, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    upstream = {
        "clip_id": quality["clip_id"],
        "dataset_id": HANYANG_EMOTIONAL_BODY_SOURCE_ID,
        "source_stem": "1_1_1_1",
        "participant_id": 1,
        "fixed_split_assignment": "train",
        "status": "passed",
        "quality_gate": all_gates,
        "quality_json": str(quality_path),
        "quality_json_sha256": sha256_file(quality_path),
        "quality_record_sha256": quality["record_sha256"],
        "kimodo_accessed_or_used": False,
    }
    upstream["record_sha256"] = _record_hash(upstream)
    passed = retarget_root / "passed_manifest.jsonl"
    passed.write_text(stable_json(upstream) + "\n", encoding="utf-8")
    return passed


def test_builds_source_faithful_hash_bound_unconditional_pool(
    tmp_path: Path,
) -> None:
    passed = _write_fixture(tmp_path)
    output = passed.parent / "experimental_pool_v8"
    receipt = build_experimental_pool(
        passed,
        output,
        expected_count=1,
        expected_upstream_sha256=sha256_file(passed),
    )
    rows = [
        json.loads(line)
        for line in (output / "manifest.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row["artifact_kind"] == POOL_ROW_KIND
    assert row["unconditional_motion_eligible"] is True
    assert row["local_emotion_condition_candidate"] is True
    assert row["emotion_condition_eligible"] is False
    assert row["emotion_condition_mask"] is False
    assert row["admission_tier"] == "unconditional_motion_only"
    assert row["group54_condition_mask"] is False
    assert row["style_condition_mask"] is False
    assert row["duration_condition_mask"] is False
    assert row["semantic_condition_mask"] is False
    assert row["soft_emotion_targets"]["p7"] == pytest.approx(
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    assert row["soft_emotion_targets"]["q2"] == pytest.approx([0.0, 1.0])
    manifest_text = (output / "manifest.jsonl").read_text()
    assert "raw_18d.csv" not in manifest_text
    assert "deployment_safe_partial_18d.csv" not in manifest_text
    review_text = (output / "human_review_bundle" / "manifest.jsonl").read_text()
    assert "raw_18d.csv" not in review_text
    assert "deployment_safe_partial_18d.csv" not in review_text
    review_row = json.loads(review_text)
    assert review_row["source_faithful_upstream_smoothing_preserved"] is True
    assert review_row["upstream_source_faithful_smoothing_window"] == 5
    assert review_row["endpoints_preserved"] is True
    assert review_row["additional_review_smoothing"] is False
    assert review_row["additional_review_smoothing_applied"] is False
    assert review_row["retiming_applied_for_review"] is False
    assert review_row["retimed"] is False
    assert review_row["review_status"] == "pending"
    assert receipt["pool_manifest_sha256"] == sha256_file(
        output / "manifest.jsonl"
    )
    assert receipt["formal"] is False
    assert (
        receipt["formal_rejection_reason"]
        == "insufficient_non_neutral_coverage"
    )
    assert receipt["human_review_required"] is True
    assert receipt["human_review_approved"] is False
    assert receipt["human_review_status"] == "HUMAN_REVIEW_BLOCKED"
    assert receipt["training_launch_allowed"] is False
    assert receipt["registry_snapshot_role"] == EMOTION_CRITIC_ROLE
    assert receipt["emotion_condition_eligible_count"] == 0
    assert receipt["record_sha256"] == _record_hash(receipt)


def test_rejects_nonzero_permanently_unobserved_confidence(
    tmp_path: Path,
) -> None:
    passed = _write_fixture(tmp_path, confidence_missing_value=0.25)
    with pytest.raises(ValueError, match="permanently unobserved"):
        build_experimental_pool(
            passed,
            passed.parent / "experimental_pool_v8",
            expected_count=1,
            expected_upstream_sha256=sha256_file(passed),
        )


def test_forbidden_dataset_path_fails_closed(tmp_path: Path) -> None:
    passed = _write_fixture(tmp_path)
    with pytest.raises(ValueError, match="permanently forbidden"):
        build_experimental_pool(
            passed,
            passed.parent / "kimodo_pool",
            expected_count=1,
            expected_upstream_sha256=sha256_file(passed),
        )
