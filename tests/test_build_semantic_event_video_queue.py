import hashlib
import json
from pathlib import Path

import pytest

from tools.human_motion_review import build_semantic_event_video_queue as queue


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(tmp_path: Path) -> dict:
    trajectory = tmp_path / "safe.csv"
    trajectory.write_text("joint\n0.0\n", encoding="utf-8")
    quality = tmp_path / "quality.json"
    quality.write_text(
        '{"quality_gate":{"passed":true,"axis_direction_pass":true}}\n',
        encoding="utf-8",
    )
    return {
        "task_id": "event_a",
        "source_clip_id": "source_a",
        "speaker_key": "1_wayne",
        "official_split": "train",
        "fixed_split_assignment": "validation",
        "fps": 30,
        "status": "passed",
        "quality_gate": {"passed": True, "axis_direction_pass": True},
        "accepted_for_training": False,
        "official_category_verified": True,
        "official_category_role": "verified_metadata_split_and_evaluation_only",
        "official_category_condition_channel": None,
        "official_category_loss": None,
        "official_category_conditioning_enabled": False,
        "robot_observable_motion_form": "candidate_unreviewed",
        "communicative_intent": "candidate_unreviewed",
        "canonical_prompt_role": "coarse_category_only",
        "canonical_action": "official_gesture_category:deictic",
        "canonical_action_role": "official_category_metadata_split_key_only",
        "semantic_mapping_status": "official_category_verified_metadata_only",
        "semantic_supervision_masks": dict(queue.EXPECTED_MASKS),
        "semantic_event": {"category": "deictic", "intensity": "low"},
        "emotion_id": "happy",
        "emotion_supervision_mask": False,
        "source_emotion_label_verified": True,
        "emotion_supervision_role": "disabled_pending_robot_affect_review",
        "official_emotion_conditioning_enabled": False,
        "official_emotion_condition_channel": None,
        "official_emotion_loss": None,
        "affect_observable_review_status": "candidate_unreviewed",
        "affect_observable_supervision_mask": False,
        "canonical_prompt": {"en": "Perform a low-intensity deictic gesture."},
        "safe_csv": str(trajectory),
        "safe_csv_sha256": _sha(trajectory),
        "quality_json": str(quality),
        "quality_json_sha256": _sha(quality),
        "retarget_segment": {"cropped": False, "fps": 30},
        "upstream_inventory_record_sha256": "a" * 64,
        "selected_record_sha256": "b" * 64,
        "retarget_input_manifest_sha256": "c" * 64,
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")


def test_build_queue_preserves_fail_closed_semantic_audit(tmp_path):
    manifest = tmp_path / "passed.jsonl"
    _write_jsonl(manifest, [_record(tmp_path)])
    output = tmp_path / "review_queue.jsonl"

    summary = queue.build_queue([manifest], output)
    record = json.loads(output.read_text(encoding="utf-8"))

    assert summary["records"] == 1
    assert summary["accepted_for_training"] == 0
    assert record["status"] == "passed"
    assert record["fps"] == 30
    assert record["quality_gate"]["passed"] is True
    assert record["safe_csv"] == record["trajectory_path"]
    assert record["safe_csv_sha256"] == record["trajectory_sha256"]
    assert record["canonical_action"] == "official_gesture_category:deictic"
    assert record["canonical_prompt_role"] == "coarse_category_only"
    assert record["semantic_supervision_masks"] == queue.EXPECTED_MASKS
    assert record["communicative_intent"] == "candidate_unreviewed"
    assert record["affect_observable_review_status"] == "candidate_unreviewed"
    assert record["affect_observable_supervision_mask"] is False
    assert record["semantic_action_completeness_review_required"] is True
    assert record["affect_observable_review_required"] is True
    assert record["manual_review_required"] is True
    assert record["accepted_for_training"] is False
    assert "具体意图待视频复核" in record["canonical_prompt"]["zh"]


def test_queue_rejects_unreviewed_prompt_supervision(tmp_path):
    record = _record(tmp_path)
    record["semantic_supervision_masks"]["prompt_text"] = True

    with pytest.raises(ValueError, match="supervision masks"):
        queue.queue_record(record)


def test_queue_rejects_tampered_evidence(tmp_path):
    record = _record(tmp_path)
    Path(record["safe_csv"]).write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="evidence mismatch"):
        queue.queue_record(record)
