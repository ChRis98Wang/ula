import csv
import hashlib
import json
import os
from pathlib import Path

import pytest

from tools.human_motion_review import build_ula0513_native_video_queue as BUILD
from upper_body_skeleton.retarget_v2_18d import JOINT_ORDER_18D


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(tmp_path: Path, *, physical: bool = True, frames: int = 11):
    trajectory = tmp_path / "Joy01" / "safe.csv"
    trajectory.parent.mkdir(parents=True, exist_ok=True)
    with trajectory.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(JOINT_ORDER_18D)
        writer.writerows([[0.0] * 18 for _ in range(frames)])
    record = {
        "schema_version": "1.0.0",
        "artifact_kind": BUILD.SOURCE_ARTIFACT_KIND,
        "dataset": "ULA0513_user_provided_robot_motion",
        "clip_id": "ula0513_native_v1__joy_01",
        "source_clip_id": "Robot_Model0530_V2_Joy01",
        "source_behavior_label": "Joy01",
        "source_behavior_label_role": "source_metadata_pending_review",
        "representation": BUILD.SOURCE_REPRESENTATION,
        "fps": 30.0,
        "training_segment": {
            "representation": BUILD.SOURCE_REPRESENTATION,
            "start_frame": 0,
            "end_frame_exclusive": frames,
            "frame_count": frames,
            "fixed_window_sec": None,
            "cropped": False,
            "resampled": False,
            "tiled": False,
            "duration_policy": "one_complete_source_authored_motion_asset",
        },
        "time_axes": {
            "output": {
                "sample_span_sec": (frames - 1) / 30.0,
                "frame_coverage_sec": frames / 30.0,
            }
        },
        "motion_18d": {
            "contract_version": BUILD.CONTRACT_VERSION,
            "joint_order": JOINT_ORDER_18D,
            "safe_csv_path": str(trajectory),
            "safe_csv_sha256": _sha(trajectory),
            "frame_count": frames,
            "native_head_3dof_present": True,
            "head_mapping_or_synthesis_used": False,
        },
        "physical_qc": {"passed": physical},
        "semantic_supervision_masks": dict(BUILD.EXPECTED_SEMANTIC_MASKS),
        "emotion_supervision_mask": False,
        "affect_observable_supervision_mask": False,
        "base_motion_eligible": False,
        "semantic_conditioning_eligible": False,
        "expressive_conditioning_eligible": False,
        "accepted_for_training": False,
    }
    record["record_sha256"] = BUILD.value_sha256(record)
    return record


def _write_manifest(path: Path, records) -> None:
    path.write_text("".join(BUILD.stable_json(row) + "\n" for row in records), encoding="utf-8")


def test_queue_preserves_full_native_length_and_hides_label_from_task_id(tmp_path):
    manifest = tmp_path / "source.jsonl"
    _write_manifest(manifest, [_record(tmp_path)])
    queue = tmp_path / "private" / "queue.jsonl"
    hidden = tmp_path / "hidden"
    summary = BUILD.build_queue(
        manifest,
        queue,
        hidden,
        secret_hex="11" * 32,
    )
    task = json.loads(queue.read_text(encoding="utf-8"))
    mapping = json.loads((hidden / "task_mapping.jsonl").read_text(encoding="utf-8"))
    assert summary["records"] == 1
    assert summary["fixed_window_sec"] is None
    assert task["training_segment"]["frame_count"] == 11
    assert task["training_segment"]["cropped"] is False
    assert task["retarget_segment"]["output_sample_span_sec"] == 10 / 30
    assert "joy" not in task["task_id"].lower()
    assert "joy" not in task["speaker_key"].lower()
    assert task["canonical_action"] is None
    assert task["canonical_prompt"] == BUILD.ANONYMOUS_PROMPT
    assert mapping["source_behavior_label"] == "Joy01"
    assert mapping["task_id"] == task["task_id"]
    assert (os.stat(hidden).st_mode & 0o777) == 0o700
    assert (os.stat(hidden / "task_mapping.jsonl").st_mode & 0o777) == 0o600


def test_queue_skips_physical_fail_but_never_silently_changes_a_pass(tmp_path):
    manifest = tmp_path / "source.jsonl"
    failed = _record(tmp_path, physical=False)
    _write_manifest(manifest, [failed])
    summary = BUILD.build_queue(
        manifest,
        tmp_path / "queue.jsonl",
        tmp_path / "hidden",
        secret_hex="22" * 32,
    )
    assert summary["records"] == 0
    assert summary["skipped_physical_fail"] == 1

    passed = _record(tmp_path)
    passed["training_segment"]["cropped"] = True
    passed["record_sha256"] = BUILD.value_sha256(
        {key: value for key, value in passed.items() if key != "record_sha256"}
    )
    _write_manifest(manifest, [passed])
    with pytest.raises(ValueError, match="native-length contract"):
        BUILD.build_queue(
            manifest,
            tmp_path / "queue2.jsonl",
            tmp_path / "hidden2",
            secret_hex="33" * 32,
        )


def test_queue_rejects_label_or_emotion_mask_admission(tmp_path):
    record = _record(tmp_path)
    record["emotion_supervision_mask"] = True
    record["record_sha256"] = BUILD.value_sha256(
        {key: value for key, value in record.items() if key != "record_sha256"}
    )
    manifest = tmp_path / "source.jsonl"
    _write_manifest(manifest, [record])
    with pytest.raises(ValueError, match="fail-closed"):
        BUILD.build_queue(
            manifest,
            tmp_path / "queue.jsonl",
            tmp_path / "hidden",
            secret_hex="44" * 32,
        )
