import hashlib
import json
from pathlib import Path

import pytest

from tools.human_motion_review.build_interact_review_evidence_migration_v2 import (
    ARC_PROTOCOL,
    PUBLIC_KIND,
    build_migration,
)


def _json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    old = tmp_path / "old/public"
    new = tmp_path / "new/public"
    video = new / "videos/dyad.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    video_hash = _sha(video)
    arc_row = {"sample_id": "dyad-1", "video_path": str(video), "video_sha256": video_hash}
    affect_row = {"sample_id": "dyad-1", "video_path": str(video), "video_sha256": video_hash}
    for root in (old, new):
        _jsonl(root / "arc_action_review_queue.jsonl", [arc_row])
        _jsonl(root / "affect_review_queue.jsonl", [affect_row])
    _jsonl(old / "axis_review_queue.jsonl", [{"sample_id": "old-axis"}])
    _jsonl(new / "axis_review_queue.jsonl", [{"sample_id": "new-axis"}])

    def summary(root: Path, axis_state: str):
        return {
            "artifact_kind": PUBLIC_KIND,
            "accepted_for_training": False,
            "fixed_duration_window_used": False,
            "duration_policy": "natural_no_fixed_window",
            "dyadic_arc_action_records": 1,
            "dyadic_affect_records": 1,
            "dyad_run_state_sha256": "d" * 64,
            "axis_run_state_sha256": axis_state,
            "axis_queue": "/original/public/axis_review_queue.jsonl",
            "axis_queue_sha256": _sha(root / "axis_review_queue.jsonl"),
            "arc_action_queue": "/original/public/arc_action_review_queue.jsonl",
            "arc_action_queue_sha256": _sha(root / "arc_action_review_queue.jsonl"),
            "affect_queue": "/original/public/affect_review_queue.jsonl",
            "affect_queue_sha256": _sha(root / "affect_review_queue.jsonl"),
        }

    old_summary_path = old / "summary.json"
    new_summary_path = new / "summary.json"
    _json(old_summary_path, summary(old, "a" * 64))
    _json(new_summary_path, summary(new, "b" * 64))
    review = {
        **arc_row,
        "protocol_version": ARC_PROTOCOL,
        "accepted_for_training": False,
        "fixed_duration_window_used": False,
        "native_duration_preserved": True,
        "temporal_unit": "complete_natural_interaction_arc",
    }
    review_path = tmp_path / "review.jsonl"
    _jsonl(review_path, [review])
    review_summary_path = tmp_path / "review.summary.json"
    _json(
        review_summary_path,
        {
            "artifact_kind": "interact_dyadic_arc_action_blind_review_submission_v2",
            "accepted_for_training": False,
            "submission_jsonl_sha256": _sha(review_path),
            "input_integrity": {
                "public_summary_sha256": _sha(old_summary_path),
                "arc_action_queue_sha256": _sha(old / "arc_action_review_queue.jsonl"),
            },
        },
    )
    return old_summary_path, new_summary_path, review_path, review_summary_path


def test_axis_only_rebuild_carries_arc_review_with_explicit_evidence(tmp_path):
    old, new, review, review_summary = _fixture(tmp_path)
    result = build_migration(
        old_public_summary=old,
        new_public_summary=new,
        arc_review_submission=review,
        arc_review_summary=review_summary,
        output=tmp_path / "migration.json",
    )
    assert result["review_axes_allowed_to_carry_forward"] == ["arc_action"]
    assert result["review_axes_invalidated_and_require_fresh_blind_review"] == ["axis"]
    assert result["unchanged_dyadic_evidence"]["arc_action_queue_byte_identical"] is True
    assert result["accepted_for_training"] is False


def test_migration_rejects_changed_dyadic_queue(tmp_path):
    old, new, review, review_summary = _fixture(tmp_path)
    new_value = json.loads(new.read_text())
    new_value["arc_action_queue_sha256"] = "0" * 64
    _json(new, new_value)
    with pytest.raises(ValueError, match="Dyadic evidence changed"):
        build_migration(
            old_public_summary=old,
            new_public_summary=new,
            arc_review_submission=review,
            arc_review_summary=review_summary,
            output=tmp_path / "migration.json",
        )
