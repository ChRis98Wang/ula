import json
from pathlib import Path

from tools.human_motion_review import audit_interact_full_dyad_review_progress_v3 as audit
from tools.human_motion_review import finalize_interact_full_dyad_review_shard_v3 as finalize


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(finalize.stable_json(row) + "\n" for row in rows), encoding="utf-8")


def _fixture(tmp_path: Path, *, include_submission: bool = True) -> tuple[Path, Path]:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"hash-bound-video")
    queue_row = {
        "sample_id": "sample-1",
        "video_path": str(video),
        "video_sha256": finalize.sha256_file(video),
        "accepted_for_training": False,
        "fixed_duration_window_used": False,
    }
    shard_queue = tmp_path / "source/shard_000/review_queue.jsonl"
    _write_jsonl(shard_queue, [queue_row])
    shard_summary = {
        "artifact_kind": finalize.SHARD_KIND,
        "queue_kind": "arc_action",
        "shard_index": 0,
        "shard_count": 1,
        "records": 1,
        "review_queue": str(shard_queue),
        "review_queue_sha256": finalize.sha256_file(shard_queue),
        "accepted_for_training": False,
        "fixed_duration_window_used": False,
    }
    shard_summary_path = tmp_path / "source/shard_000/summary.json"
    _write_json(shard_summary_path, shard_summary)
    root = {
        "artifact_kind": audit.SHARDS_KIND,
        "queue_kind": "arc_action",
        "shard_count": 1,
        "records": 1,
        "shards": [
            {
                "shard_index": 0,
                "records": 1,
                "summary": str(shard_summary_path),
                "summary_sha256": finalize.sha256_file(shard_summary_path),
                "review_queue_sha256": finalize.sha256_file(shard_queue),
            }
        ],
        "source_public_summary_sha256": "a" * 64,
        "source_queue_sha256": "b" * 64,
        "coverage_complete_without_overlap": True,
        "accepted_for_training": False,
        "fixed_duration_window_used": False,
    }
    root_path = tmp_path / "source/summary.json"
    _write_json(root_path, root)
    submissions = tmp_path / "submissions"
    if include_submission:
        provenance = {
            "shard_summary_sha256": finalize.sha256_file(shard_summary_path),
            "shard_queue_sha256": finalize.sha256_file(shard_queue),
            "source_public_summary_sha256": "a" * 64,
            "source_queue_sha256": "b" * 64,
            "queue_record_hash_method": "sha256_stable_json_utf8",
            "queue_record_sha256": finalize.value_sha256(queue_row),
        }
        reviewed = dict(queue_row)
        reviewed.update(
            {
                "artifact_kind": finalize.SUBMISSION_KIND,
                "reviewer_id": "reviewer-1",
                "decode_complete": True,
                "blind_review_provenance": provenance,
            }
        )
        submission_path = submissions / "shard_000/submission.jsonl"
        _write_jsonl(submission_path, [reviewed])
        _write_json(
            submissions / "shard_000/summary.json",
            {
                "artifact_kind": finalize.SUMMARY_KIND,
                "queue_kind": "arc_action",
                "shard_index": 0,
                "shard_count": 1,
                "records": 1,
                "reviewer_id": "reviewer-1",
                "submission": str(submission_path),
                "submission_sha256": finalize.sha256_file(submission_path),
                "shard_summary_sha256": finalize.sha256_file(shard_summary_path),
                "shard_queue_sha256": finalize.sha256_file(shard_queue),
                "source_public_summary_sha256": "a" * 64,
                "source_queue_sha256": "b" * 64,
                "coverage_complete_without_overlap": True,
                "all_videos_hash_verified_and_probed_to_eof": True,
                "accepted_for_training": False,
                "fixed_duration_window_used": False,
            },
        )
    return root_path, submissions


def test_complete_partial_audit_is_merge_ready(tmp_path):
    root, submissions = _fixture(tmp_path)
    result = audit.audit(shards_summary_path=root, submissions_root=submissions)
    assert result["merge_ready"] is True
    assert result["valid_reviewed_records"] == 1
    assert result["accepted_for_training"] is False


def test_missing_shard_is_reported_without_admission(tmp_path):
    root, submissions = _fixture(tmp_path, include_submission=False)
    result = audit.audit(shards_summary_path=root, submissions_root=submissions)
    assert result["merge_ready"] is False
    assert result["missing_shard_indexes"] == [0]
    assert result["valid_reviewed_records"] == 0


def test_tampered_submission_is_invalid(tmp_path):
    root, submissions = _fixture(tmp_path)
    submission = submissions / "shard_000/submission.jsonl"
    submission.write_text(submission.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    result = audit.audit(shards_summary_path=root, submissions_root=submissions)
    assert result["merge_ready"] is False
    assert result["invalid_shards"][0]["shard_index"] == 0
