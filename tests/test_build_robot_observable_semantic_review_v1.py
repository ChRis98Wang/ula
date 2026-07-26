from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools.human_motion_review import build_robot_observable_semantic_review_v1 as SEM


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(SEM.stable_json(row) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _source_fixture(tmp_path: Path) -> dict[str, Path]:
    public = tmp_path / "source_public"
    videos = public / "videos"
    videos.mkdir(parents=True)
    queue_rows = []
    candidate_rows = []
    for index in range(2):
        sample_id = f"expr_test_{index}"
        video = videos / f"{sample_id}.mp4"
        video.write_bytes(f"anonymous-video-{index}".encode("ascii"))
        video_sha = SEM.sha256_file(video)
        queue_rows.append(
            {
                "sample_id": sample_id,
                "video_path": str(video),
                "video_sha256": video_sha,
                "context_level": index + 1,
                "frame_count": 90 + index * 75,
                "fps": 30.0,
            }
        )
        candidate_rows.append(
            {
                "schema_version": SEM.SCHEMA_VERSION,
                "artifact_kind": SEM.TRAIN_CANDIDATE_KIND,
                "sample_id": sample_id,
                "video_sha256": video_sha,
                "context_level": index + 1,
                "frame_count": 90 + index * 75,
                "fps": 30.0,
                "trajectory_sha256": str(index + 1) * 64,
                "arc_action_complete": True,
                "action_observability": "observable",
                "semantic_supervision_mask": False,
                "native_variable_length": True,
                "fixed_duration_window_used": False,
                "license_training_admission": False,
                "accepted_for_training": False,
            }
        )
    queue = public / "arc_action_review_queue.jsonl"
    _write_jsonl(queue, queue_rows)
    public_summary = public / "summary.json"
    _write_json(
        public_summary,
        {
            "artifact_kind": SEM.SOURCE_PUBLIC_KIND,
            "arc_action_queue": str(queue),
            "arc_action_queue_sha256": SEM.sha256_file(queue),
            "all_samples_native_variable_length": True,
            "fixed_duration_window_used": False,
            "accepted_for_training": False,
        },
    )
    private = tmp_path / "qualification"
    candidates = private / "train_candidate.jsonl"
    _write_jsonl(candidates, candidate_rows)
    qualification_summary = private / "summary.json"
    _write_json(
        qualification_summary,
        {
            "artifact_kind": SEM.QUALIFICATION_KIND,
            "validation_passed": True,
            "native_variable_length": True,
            "fixed_duration_window_used": False,
            "accepted_for_training": False,
            "inputs": {
                "public_summary": {
                    "path": str(public_summary),
                    "sha256": SEM.sha256_file(public_summary),
                },
                "arc_action_queue": {
                    "path": str(queue),
                    "sha256": SEM.sha256_file(queue),
                },
            },
            "outputs": {
                "train_candidate": {
                    "path": str(candidates),
                    "sha256": SEM.sha256_file(candidates),
                    "records": len(candidate_rows),
                }
            },
        },
    )
    return {
        "public_summary": public_summary,
        "qualification_summary": qualification_summary,
    }


def _build_author(tmp_path: Path) -> dict[str, Path]:
    source = _source_fixture(tmp_path)
    public = tmp_path / "author_public"
    hidden = tmp_path / "author_hidden"
    SEM.build_author_bundle(
        public_summary=source["public_summary"],
        qualification_summary=source["qualification_summary"],
        public_root=public,
        hidden_root=hidden,
    )
    return source | {"author_public": public, "author_hidden": hidden}


def _author_submission(tmp_path: Path, author_public: Path, *, bad_text: str | None = None):
    queue_path = author_public / "review_queue.jsonl"
    queue_rows = _read_jsonl(queue_path)
    bound_rows = SEM.read_bound_jsonl(queue_path)
    line_hash = {row["sample_id"]: digest for row, digest in bound_rows}
    rows = []
    for index, queue in enumerate(queue_rows):
        text = (
            bad_text
            if bad_text is not None and index == 0
            else (
                "The robot raises one forearm, points to the side, and returns both arms to rest."
                if index == 0
                else "The robot opens both arms, pauses in a wide pose, and lowers them to rest."
            )
        )
        rows.append(
            {
                **{
                    key: queue[key]
                    for key in (
                        "sample_id",
                        "video_path",
                        "video_sha256",
                        "context_level",
                        "frame_count",
                        "fps",
                    )
                },
                "protocol_version": SEM.AUTHOR_PROTOCOL,
                "review_id": f"author-review-{index}",
                "reviewer_id": "independent-author-r1",
                "candidate_text": text,
                "candidate_text_sha256": SEM.text_sha256(text),
                "candidate_text_provenance": SEM.TEXT_PROVENANCE,
                "observable_description": "Visible ordered arm motion followed by a settled ending.",
                "full_decode_to_eof": True,
                "decoded_frame_count": queue["frame_count"],
                "native_duration_preserved": True,
                "fixed_duration_window_used": False,
                "audio_available": False,
                "label_metadata_exposed": False,
                "emotion_inference_performed": False,
                "training_admission": False,
                "blind_review_provenance": {
                    "public_summary_sha256": SEM.sha256_file(
                        author_public / "summary.json"
                    ),
                    "public_queue_sha256": SEM.sha256_file(queue_path),
                    "public_queue_record_sha256": line_hash[queue["sample_id"]]
                },
            }
        )
    submission = tmp_path / "author_submission.jsonl"
    _write_jsonl(submission, rows)
    summary = tmp_path / "author_submission.summary.json"
    public_summary = author_public / "summary.json"
    _write_json(
        summary,
        {
            "artifact_kind": SEM.AUTHOR_SUMMARY_KIND,
            "records": len(rows),
            "coverage_complete": True,
            "training_admission": False,
            "submission_path": str(submission),
            "submission_sha256": SEM.sha256_file(submission),
            "public_summary_path": str(public_summary),
            "public_summary_sha256": SEM.sha256_file(public_summary),
            "public_queue_path": str(queue_path),
            "public_queue_sha256": SEM.sha256_file(queue_path),
        },
    )
    return submission, summary


def _build_matcher(tmp_path: Path) -> dict[str, Path]:
    fixture = _build_author(tmp_path)
    submission, submission_summary = _author_submission(
        tmp_path, fixture["author_public"]
    )
    public = tmp_path / "matcher_public"
    hidden = tmp_path / "matcher_hidden"
    SEM.build_matcher_bundle(
        author_public_summary=fixture["author_public"] / "summary.json",
        author_submission=submission,
        author_submission_summary=submission_summary,
        public_root=public,
        hidden_root=hidden,
    )
    return fixture | {
        "author_submission": submission,
        "author_submission_summary": submission_summary,
        "matcher_public": public,
        "matcher_hidden": hidden,
    }


def _matcher_submission(
    tmp_path: Path, matcher_public: Path, *, reviewer_id: str = "independent-matcher-r2"
):
    queue_path = matcher_public / "review_queue.jsonl"
    queue_rows = _read_jsonl(queue_path)
    line_hash = {
        row["sample_id"]: digest for row, digest in SEM.read_bound_jsonl(queue_path)
    }
    rows = []
    for index, queue in enumerate(queue_rows):
        result = "observable_match" if index == 0 else "mismatch"
        rows.append(
            {
                **{
                    key: queue[key]
                    for key in (
                        "sample_id",
                        "video_path",
                        "video_sha256",
                        "context_level",
                        "frame_count",
                        "fps",
                        "candidate_text",
                        "candidate_text_sha256",
                        "candidate_text_provenance",
                    )
                },
                "protocol_version": SEM.MATCHER_PROTOCOL,
                "review_id": f"matcher-review-{index}",
                "reviewer_id": reviewer_id,
                "match_result": result,
                "match_confidence": 0.94 if index == 0 else 0.91,
                "match_evidence": "The ordered visible motion matches." if index == 0 else "The visible motion does not match.",
                "full_decode_to_eof": True,
                "decoded_frame_count": queue["frame_count"],
                "native_duration_preserved": True,
                "fixed_duration_window_used": False,
                "audio_available": False,
                "label_metadata_exposed": False,
                "emotion_inference_performed": False,
                "training_admission": False,
                "blind_review_provenance": {
                    "public_summary_sha256": SEM.sha256_file(
                        matcher_public / "summary.json"
                    ),
                    "public_queue_sha256": SEM.sha256_file(queue_path),
                    "public_queue_record_sha256": line_hash[queue["sample_id"]]
                },
            }
        )
    submission = tmp_path / "matcher_submission.jsonl"
    _write_jsonl(submission, rows)
    summary = tmp_path / "matcher_submission.summary.json"
    public_summary = matcher_public / "summary.json"
    _write_json(
        summary,
        {
            "artifact_kind": SEM.MATCHER_SUMMARY_KIND,
            "records": len(rows),
            "coverage_complete": True,
            "training_admission": False,
            "submission_path": str(submission),
            "submission_sha256": SEM.sha256_file(submission),
            "public_summary_path": str(public_summary),
            "public_summary_sha256": SEM.sha256_file(public_summary),
            "public_queue_path": str(queue_path),
            "public_queue_sha256": SEM.sha256_file(queue_path),
        },
    )
    return submission, summary


def test_author_bundle_is_anonymous_native_length_and_exact_whitelist(tmp_path):
    fixture = _build_author(tmp_path)
    public = fixture["author_public"]
    rows = _read_jsonl(public / "review_queue.jsonl")

    assert len(rows) == 2
    assert {row["frame_count"] for row in rows} == {90, 165}
    assert all(row["fixed_duration_window_used"] is False for row in rows)
    assert all(row["candidate_text"] is None for row in rows)
    assert not any("trajectory" in SEM.stable_json(row) for row in rows)
    assert {path.name for path in public.iterdir()} == {
        "review_queue.jsonl",
        "summary.json",
        "videos",
    }
    assert all(path.stat().st_nlink == 1 for path in public.rglob("*") if path.is_file())
    assert not (os.stat(public / "summary.json").st_mode & 0o222)
    assert oct(os.stat(fixture["author_hidden"]).st_mode & 0o777) == "0o700"


def test_matcher_and_merge_enable_only_independent_observable_matches(tmp_path):
    fixture = _build_matcher(tmp_path)
    submission, submission_summary = _matcher_submission(
        tmp_path, fixture["matcher_public"]
    )
    result = SEM.merge_matcher_reviews(
        matcher_public_summary=fixture["matcher_public"] / "summary.json",
        matcher_hidden_summary=fixture["matcher_hidden"] / "summary.json",
        matcher_submission=submission,
        matcher_submission_summary=submission_summary,
        output_root=tmp_path / "merged",
    )
    rows = _read_jsonl(tmp_path / "merged" / "semantic_qualification.jsonl")

    assert result["semantic_supervision_enabled"] == 1
    assert rows[0]["semantic_supervision_mask"] is True
    assert rows[0]["candidate_text"].startswith("The robot ")
    assert rows[1]["semantic_supervision_mask"] is False
    assert rows[1]["candidate_text"] is None
    assert all(row["accepted_for_training"] is False for row in rows)
    assert all(row["fixed_duration_window_used"] is False for row in rows)


def test_affect_or_mental_state_in_semantic_text_is_rejected(tmp_path):
    fixture = _build_author(tmp_path)
    submission, summary = _author_submission(
        tmp_path,
        fixture["author_public"],
        bad_text="The robot happily raises one hand because it feels excited and wants attention.",
    )

    with pytest.raises(ValueError, match="affect, mental-state"):
        SEM.build_matcher_bundle(
            author_public_summary=fixture["author_public"] / "summary.json",
            author_submission=submission,
            author_submission_summary=summary,
            public_root=tmp_path / "bad_matcher_public",
            hidden_root=tmp_path / "bad_matcher_hidden",
        )


def test_matcher_must_be_independent_from_author(tmp_path):
    fixture = _build_matcher(tmp_path)
    submission, summary = _matcher_submission(
        tmp_path,
        fixture["matcher_public"],
        reviewer_id="independent-author-r1",
    )

    with pytest.raises(ValueError, match="violates the blind contract"):
        SEM.merge_matcher_reviews(
            matcher_public_summary=fixture["matcher_public"] / "summary.json",
            matcher_hidden_summary=fixture["matcher_hidden"] / "summary.json",
            matcher_submission=submission,
            matcher_submission_summary=summary,
            output_root=tmp_path / "rejected_merge",
        )
    assert not (tmp_path / "rejected_merge").exists()


def test_fixed_duration_source_is_rejected(tmp_path):
    source = _source_fixture(tmp_path)
    summary = json.loads(source["qualification_summary"].read_text(encoding="utf-8"))
    summary["fixed_duration_window_used"] = True
    _write_json(source["qualification_summary"], summary)

    with pytest.raises(ValueError, match="native-length"):
        SEM.build_author_bundle(
            public_summary=source["public_summary"],
            qualification_summary=source["qualification_summary"],
            public_root=tmp_path / "public",
            hidden_root=tmp_path / "hidden",
        )
