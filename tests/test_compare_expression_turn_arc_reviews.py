import hashlib
import json
from pathlib import Path

from tools.human_motion_review.compare_expression_turn_arc_reviews import compare_reviews


def _stable(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(_stable(row) + "\n" for row in rows), encoding="utf-8")


def _sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _review(queue, *, reviewer, role, complete=True, action=True):
    row = {
        **queue,
        "queue_sha256": None,
        "arc_protocol_version": "robot_expression_arc_blind_video_v1",
        "action_protocol_version": "robot_action_semantics_blind_video_v1",
        "arc_review_id": f"{reviewer}-arc",
        "action_review_id": f"{reviewer}-action",
        "arc_reviewer_id": reviewer,
        "action_reviewer_id": reviewer,
        "onset_status": "complete" if complete else "boundary_truncated",
        "apex_status": "complete",
        "offset_status": "complete" if complete else "boundary_truncated",
        "observable_description": "A visible gesture arc.",
        "training_admission": False,
    }
    if role == "r1":
        row["action_result"] = "observable_match" if action else "not_observable"
    else:
        row["action_result"] = "pass" if action else "fail"
        row["action_observability"] = "observable" if action else "not_observable"
    return row


def test_dual_review_requires_both_reviewers_to_confirm_complete_arc(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    queue_row = {
        "sample_id": "expr_001",
        "video_path": str(video),
        "video_sha256": _sha(video),
        "context_level": 0,
        "audio_available": False,
        "label_metadata_exposed": False,
    }
    queue = tmp_path / "queue.jsonl"
    _jsonl(queue, [queue_row])
    public = tmp_path / "summary.json"
    public.write_text(
        json.dumps(
            {
                "artifact_kind": "expression_turn_v8_separate_blind_review_bundle",
                "source_identity_official_action_text_and_emotion_exposed": False,
                "accepted_for_training": False,
                "arc_action_queue": str(queue),
                "arc_action_queue_sha256": _sha(queue),
            }
        ),
        encoding="utf-8",
    )
    r1 = _review(queue_row, reviewer="r1", role="r1", complete=True)
    r2 = _review(queue_row, reviewer="r2", role="r2", complete=False)
    r1["queue_sha256"] = _sha(queue)
    r2["queue_sha256"] = _sha(queue)
    r1_path = tmp_path / "r1.jsonl"
    r2_path = tmp_path / "r2.jsonl"
    _jsonl(r1_path, [r1])
    _jsonl(r2_path, [r2])
    summary = compare_reviews(
        public_summary=public,
        review_r1=r1_path,
        review_r2=r2_path,
        output_root=tmp_path / "out",
    )
    assert summary["qualification_status_distribution"] == {
        "natural_context_expansion_required": 1
    }
    assert summary["review_disagreement_count"] == 1
    assert summary["accepted_for_training_count"] == 0
