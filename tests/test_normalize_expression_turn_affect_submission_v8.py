from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.human_motion_review import (
    merge_expression_turn_expansion_qualification_v8 as MERGE,
)
from tools.human_motion_review import (
    normalize_expression_turn_affect_submission_v8 as NORMALIZE,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(NORMALIZE.stable_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _fixture(tmp_path: Path) -> dict[str, Path]:
    public = tmp_path / "public"
    videos = public / "videos"
    videos.mkdir(parents=True)
    queue_rows = []
    submission_rows = []
    specs = [("sample_a", "observable", "neutral", 0.93), ("sample_b", "ambiguous", None, None)]
    for index, (sample_id, result, predicted, confidence) in enumerate(specs):
        video = videos / f"{sample_id}.mp4"
        video.write_bytes(f"video-{sample_id}".encode("ascii"))
        common = {
            "sample_id": sample_id,
            "video_path": str(video),
            "video_sha256": NORMALIZE.sha256_file(video),
            "context_level": 2,
            "fps": 30.0,
            "frame_count": 100 + index,
            "allowed_classes": list(NORMALIZE.ALLOWED_CLASSES),
        }
        queue_rows.append(common)
        submission_rows.append(
            {
                **common,
                "schema_version": NORMALIZE.SCHEMA_VERSION,
                "affect_protocol_version": NORMALIZE.PROTOCOL,
                "affect_review_id": "non-unique-source-review-id",
                "affect_reviewer_id": "blind-affect-r1",
                "result": result,
                "predicted_class": predicted,
                "confidence": confidence,
                "audio_available": False,
                "label_metadata_exposed": False,
                "full_video_reviewed": True,
                "decode_started_at_frame": 0,
                "decode_reached_eof": True,
                "decoded_frame_count": common["frame_count"],
                "frame_count_verified": True,
                "native_duration_preserved": True,
                "fixed_duration_window_used": False,
                "training_admission": False,
                "video_sha256_verified": True,
            }
        )
    queue = public / "affect_review_queue.jsonl"
    _write_jsonl(queue, queue_rows)
    public_summary = public / "summary.json"
    _write_json(
        public_summary,
        {
            "artifact_kind": NORMALIZE.PUBLIC_KIND,
            "accepted_for_training": False,
            "all_samples_native_variable_length": True,
            "fixed_duration_window_used": False,
            "affect_queue": str(queue),
            "affect_queue_sha256": NORMALIZE.sha256_file(queue),
        },
    )
    for row in submission_rows:
        row["source_affect_queue_sha256"] = NORMALIZE.sha256_file(queue)
    submission = tmp_path / "source.jsonl"
    _write_jsonl(submission, submission_rows)
    source_summary = tmp_path / "source.summary.json"
    _write_json(
        source_summary,
        {
            "source_public_summary_sha256": NORMALIZE.sha256_file(public_summary),
            "source_affect_queue_sha256": NORMALIZE.sha256_file(queue),
            "records_expected": 2,
            "records_reviewed": 2,
            "coverage": {"fraction": 1},
            "strict_blind_attestation": {
                "fixed_seconds_or_fixed_duration_window_used": False,
                "all_training_admission_false": True,
            },
        },
    )
    return {
        "public_summary": public_summary,
        "queue": queue,
        "submission": submission,
        "source_summary": source_summary,
    }


def test_normalization_changes_schema_only_and_passes_qualification_validator(tmp_path):
    fixture = _fixture(tmp_path)
    output = tmp_path / "normalized.jsonl"
    summary = tmp_path / "normalized.summary.json"
    result = NORMALIZE.normalize_submission(
        public_summary_path=fixture["public_summary"],
        source_submission_path=fixture["submission"],
        source_summary_path=fixture["source_summary"],
        output_submission_path=output,
        output_summary_path=summary,
    )
    source_rows = {row["sample_id"]: row for row in _read_jsonl(fixture["submission"])}
    normalized_rows = {row["sample_id"]: row for row in _read_jsonl(output)}

    assert result["result_distribution"] == {
        "observable": 1,
        "ambiguous": 1,
        "not_observable": 0,
    }
    assert len({row["affect_review_id"] for row in normalized_rows.values()}) == 2
    for sample_id, row in normalized_rows.items():
        source = source_rows[sample_id]
        for field in ("result", "predicted_class", "confidence"):
            assert row[field] == source[field]
        assert row["normalization"]["policy"].endswith("no_judgment_change_v1")

    queue = MERGE.index_bound(
        MERGE.read_bound_jsonl(fixture["queue"]), "sample_id", context="test queue"
    )
    arc_reviews = {
        sample_id: (
            {
                "arc_reviewer_id": "independent-arc-r1",
                "action_reviewer_id": "independent-arc-r1",
            },
            "0" * 64,
        )
        for sample_id in queue
    }
    verified = MERGE._verify_affect_reviews(
        submission_path=output,
        summary_path=summary,
        queue_path=fixture["queue"],
        queue=queue,
        arc_reviews=arc_reviews,
    )
    assert set(verified) == set(queue)


def test_normalization_rejects_pseudo_label_on_ambiguous_result(tmp_path):
    fixture = _fixture(tmp_path)
    rows = _read_jsonl(fixture["submission"])
    rows[1]["predicted_class"] = "happy"
    rows[1]["confidence"] = 0.9
    _write_jsonl(fixture["submission"], rows)

    with pytest.raises(ValueError, match="pseudo-label"):
        NORMALIZE.normalize_submission(
            public_summary_path=fixture["public_summary"],
            source_submission_path=fixture["submission"],
            source_summary_path=fixture["source_summary"],
            output_submission_path=tmp_path / "normalized.jsonl",
            output_summary_path=tmp_path / "normalized.summary.json",
        )
