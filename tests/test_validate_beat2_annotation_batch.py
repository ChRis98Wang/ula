import csv
import hashlib
import json
from pathlib import Path

import pytest

from tools.human_motion_review.validate_beat2_annotation_batch import (
    JOINT_ORDER,
    REQUIRED_QUALITY_GATES,
    ROBOT_CONTRACT,
    main,
    validate_batch,
)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory_record(clip_id: str, start: int, *, eligible: bool = True) -> dict:
    return {
        "clip_id": clip_id,
        "fps": 30.0,
        "official_split": "train",
        "speaker_key": "12_zhao",
        "issues": [] if eligible else ["missing_textgrid_alignment"],
        "window": {
            "selection_status": (
                "selected_nonstatic_low_dynamic_with_aligned_speech"
                if eligible
                else "selected_nonstatic_low_dynamic"
            ),
            "start_frame": start,
            "end_frame_exclusive": start + 180,
        },
    }


def prewindowed_inventory_record(source_clip_id: str, start: int) -> dict:
    task_id = task(source_clip_id, start)
    return {
        "clip_id": task_id,
        "task_id": task_id,
        "source_clip_id": source_clip_id,
        "fps": 30.0,
        "official_split": "train",
        "speaker_key": "12_zhao",
        "issues": ["motion_audio_duration_mismatch_gt_0_3s"],
        "window": {
            "selection_status": "full_nonoverlap_boundary_validated",
            "start_frame": start,
            "end_frame_exclusive": start + 180,
        },
    }


def task(clip_id: str, start: int) -> str:
    return f"{clip_id}_f{start:06d}-{start + 180:06d}"


def state_record(clip_id: str, start: int, status: str) -> dict:
    return {
        "task_id": task(clip_id, start),
        "clip_id": clip_id,
        "status": status,
        "start_frame": start,
        "end_frame_exclusive": start + 180,
        "fps": 30.0,
        "official_split": "train",
        "speaker_key": "12_zhao",
        "accepted_for_training": False,
    }


def write_passed_evidence(root: Path, record: dict) -> None:
    task_id = record["task_id"]
    evidence = root / task_id
    evidence.mkdir(parents=True)
    safe = evidence / f"{task_id}_gmr_safe_18d.csv"
    with safe.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(JOINT_ORDER)
        writer.writerows([[0.0] * 18, [0.1] * 18])
    quality = evidence / "quality.json"
    quality.write_text(
        json.dumps(
            {
                "output_contract": ROBOT_CONTRACT,
                "action_dim": 18,
                "joint_order": JOINT_ORDER,
                "fps": 30.0,
                "frames": 2,
                "source_window_start_frame": record["start_frame"],
                "source_window_end_frame_exclusive": record[
                    "end_frame_exclusive"
                ],
                "quality_gate": {key: True for key in REQUIRED_QUALITY_GATES},
                "outputs": {"safe_csv": str(safe.resolve())},
            }
        ),
        encoding="utf-8",
    )
    record.update(
        {
            "retarget_contract": ROBOT_CONTRACT,
            "safe_csv": str(safe.resolve()),
            "safe_csv_sha256": digest(safe),
            "quality_json": str(quality.resolve()),
            "quality_json_sha256": digest(quality),
        }
    )


def fixture(tmp_path: Path) -> dict[str, Path]:
    inventory = tmp_path / "inventory.jsonl"
    write_jsonl(
        inventory,
        [
            inventory_record("pass_clip", 0),
            inventory_record("fail_clip", 30),
            inventory_record("pending_clip", 60),
            inventory_record("excluded_clip", 90, eligible=False),
        ],
    )
    passed_record = state_record("pass_clip", 0, "passed")
    write_passed_evidence(tmp_path / "evidence", passed_record)
    passed = tmp_path / "passed.jsonl"
    failed = tmp_path / "failed.jsonl"
    pending = tmp_path / "pending.jsonl"
    excluded = tmp_path / "excluded.jsonl"
    annotations = tmp_path / "annotations.jsonl"
    write_jsonl(passed, [passed_record])
    write_jsonl(failed, [state_record("fail_clip", 30, "quality_failed")])
    write_jsonl(pending, [state_record("pending_clip", 60, "pending")])
    excluded_record = state_record("excluded_clip", 90, "excluded")
    write_jsonl(excluded, [excluded_record])
    write_jsonl(
        annotations,
        [
            {
                "record_id": task("pass_clip", 0),
                "robot_contract": ROBOT_CONTRACT,
                "canonical_prompt": {"en": "Move one arm.", "zh": "移动一侧手臂。"},
                "observable_features": {
                    "arm": {"laterality": "left", "amplitude": "small"},
                    "head_motion": "subtle",
                    "torso_motion": "minimal",
                },
                "semantic_confidence": "medium",
                "review_flags": [],
                "prompt_provenance": "trajectory_only",
                "trajectory_path": passed_record["safe_csv"],
                "trajectory_sha256": passed_record["safe_csv_sha256"],
                "quality_json": passed_record["quality_json"],
                "quality_json_sha256": passed_record["quality_json_sha256"],
                "official_split": "train",
                "speaker_key": "12_zhao",
                "accepted_for_training": False,
                "manual_review_required": True,
            }
        ],
    )
    return {
        "inventory_path": inventory,
        "passed_path": passed,
        "failed_path": failed,
        "pending_path": pending,
        "excluded_path": excluded,
        "annotations_path": annotations,
    }


def validate(paths: dict[str, Path]) -> dict:
    return validate_batch(**paths, expected_eligible_count=3)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_valid_batch_is_closed_by_task_id_and_preserves_provenance(tmp_path):
    paths = fixture(tmp_path)
    summary = validate(paths)

    assert summary["valid"] is True
    assert summary["pipeline_ready"] is True
    assert summary["counts"] == {
        "inventory": 4,
        "eligible": 3,
        "inventory_excluded": 1,
        "passed": 1,
        "failed": 1,
        "pending": 1,
        "excluded": 1,
        "annotations": 1,
        "train_ready": 0,
    }
    assert all(value["matches"] for key, value in summary["closure"].items() if isinstance(value, dict))


def test_prewindowed_inventory_uses_declared_task_id_without_double_suffix(tmp_path):
    paths = fixture(tmp_path)
    record = prewindowed_inventory_record("pass_clip", 0)
    write_jsonl(paths["inventory_path"], [record])
    passed = read_jsonl(paths["passed_path"])[0]
    passed["clip_id"] = task("pass_clip", 0)
    write_jsonl(paths["passed_path"], [passed])
    write_jsonl(paths["failed_path"], [])
    write_jsonl(paths["pending_path"], [])
    write_jsonl(paths["excluded_path"], [])

    summary = validate_batch(**paths, expected_eligible_count=1)

    assert summary["valid"] is True
    assert summary["counts"]["eligible"] == 1
    assert summary["counts"]["passed"] == 1
    assert summary["legacy_base_clip_id_adaptations"]["passed"] == 0


def test_prewindowed_inventory_rejects_task_id_not_bound_to_source_window(tmp_path):
    paths = fixture(tmp_path)
    record = prewindowed_inventory_record("pass_clip", 0)
    record["task_id"] = "pass_clip_f000180-000360"
    record["clip_id"] = record["task_id"]
    write_jsonl(paths["inventory_path"], [record])

    with pytest.raises(ValueError, match="not bound to source_clip_id and window"):
        validate_batch(**paths, expected_eligible_count=1)


def test_state_overlap_and_missing_pending_fail_closure(tmp_path):
    paths = fixture(tmp_path)
    pending = read_jsonl(paths["pending_path"])[0]
    write_jsonl(paths["pending_path"], [])

    summary = validate(paths)

    assert summary["valid"] is False
    assert any("eligible inventory" in error for error in summary["errors"])

    write_jsonl(paths["failed_path"], [*read_jsonl(paths["failed_path"]), pending])
    write_jsonl(paths["pending_path"], [pending])
    summary = validate(paths)
    assert any("failed/pending overlap" in error for error in summary["errors"])


def test_annotation_for_nonpassed_task_is_rejected(tmp_path):
    paths = fixture(tmp_path)
    annotation = read_jsonl(paths["annotations_path"])[0]
    annotation["record_id"] = task("fail_clip", 30)
    failed_record = read_jsonl(paths["failed_path"])[0]
    annotation["trajectory_path"] = read_jsonl(paths["passed_path"])[0]["safe_csv"]
    annotation["trajectory_sha256"] = read_jsonl(paths["passed_path"])[0][
        "safe_csv_sha256"
    ]
    assert failed_record["status"] == "quality_failed"
    write_jsonl(paths["annotations_path"], [annotation])

    summary = validate(paths)

    assert any("only retarget-passed" in error for error in summary["errors"])
    assert any("annotations must cover exactly" in error for error in summary["errors"])


def test_annotation_admission_metadata_and_split_are_fail_closed(tmp_path):
    paths = fixture(tmp_path)
    annotation = read_jsonl(paths["annotations_path"])[0]
    annotation["accepted_for_training"] = True
    annotation["manual_review_required"] = False
    annotation.pop("speaker_key")
    write_jsonl(paths["annotations_path"], [annotation])

    summary = validate(paths)

    assert any("accepted_for_training must be false" in error for error in summary["errors"])
    assert any("manual_review_required" in error for error in summary["errors"])
    assert any("speaker_key was not preserved" in error for error in summary["errors"])


def test_evidence_hash_and_18d_contract_tampering_are_detected(tmp_path):
    paths = fixture(tmp_path)
    passed = read_jsonl(paths["passed_path"])[0]
    quality_path = Path(passed["quality_json"])
    quality = json.loads(quality_path.read_text())
    quality["action_dim"] = 15
    quality_path.write_text(json.dumps(quality), encoding="utf-8")

    summary = validate(paths)

    assert any("quality_json_sha256 mismatch" in error for error in summary["errors"])
    assert any("quality action_dim mismatch" in error for error in summary["errors"])


def test_legacy_base_clip_annotation_id_is_adapted_only_when_unambiguous(tmp_path):
    paths = fixture(tmp_path)
    annotation = read_jsonl(paths["annotations_path"])[0]
    annotation["record_id"] = "pass_clip"
    annotation["manual_human_review_required"] = annotation.pop(
        "manual_review_required"
    )
    write_jsonl(paths["annotations_path"], [annotation])

    summary = validate(paths)

    assert summary["valid"] is True
    assert summary["legacy_base_clip_id_adaptations"]["annotations"] == 1


def test_cli_writes_invalid_summary_and_returns_nonzero(tmp_path):
    paths = fixture(tmp_path)
    write_jsonl(paths["pending_path"], [])
    output = tmp_path / "summary.json"
    args = [
        "--inventory", str(paths["inventory_path"]),
        "--passed", str(paths["passed_path"]),
        "--failed", str(paths["failed_path"]),
        "--pending", str(paths["pending_path"]),
        "--excluded", str(paths["excluded_path"]),
        "--annotations", str(paths["annotations_path"]),
        "--expected-eligible-count", "3",
        "--output", str(output),
    ]

    assert main(args) == 1
    assert json.loads(output.read_text())["pipeline_ready"] is False


def test_valid_cli_writes_deterministic_speaker_review_queue(tmp_path):
    paths = fixture(tmp_path)
    output = tmp_path / "summary.json"
    queue = tmp_path / "review_queue.jsonl"
    args = [
        "--inventory", str(paths["inventory_path"]),
        "--passed", str(paths["passed_path"]),
        "--failed", str(paths["failed_path"]),
        "--pending", str(paths["pending_path"]),
        "--excluded", str(paths["excluded_path"]),
        "--annotations", str(paths["annotations_path"]),
        "--expected-eligible-count", "3",
        "--output", str(output),
        "--review-queue-output", str(queue),
    ]

    assert main(args) == 0
    summary = json.loads(output.read_text())
    queued = read_jsonl(queue)
    assert summary["counts"]["train_ready"] == 0
    assert summary["distributions"]["eligible_by_speaker_key"] == {"12_zhao": 3}
    assert summary["review_queue"]["sha256"] == digest(queue)
    assert queued[0]["manual_review_required"] is True
    assert queued[0]["accepted_for_training"] is False
    assert queued[0]["observable_features"]["arm"]["laterality"] == "left"
    assert queued[0]["semantic_confidence"] == "medium"
    assert queued[0]["speech_context_included"] is False
    assert not any(
        "speech" in key.lower()
        for key in queued[0]
        if key != "speech_context_included"
    )
