#!/usr/bin/env python3
"""Build and merge label-blind robot-observable semantic review bundles.

The three stages deliberately separate authorship from matching:

* ``author`` exposes only complete anonymous robot videos.
* ``matcher`` exposes the same videos plus the independently authored text.
* ``merge`` enables semantic supervision only after an independent match.

All outputs remain unaccepted for training.  License admission and the final
formal manifest are separate gates.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any, Iterable


SCHEMA_VERSION = "1.0.0"
QUALIFICATION_KIND = "expression_turn_v8_expansion_qualification_summary_v1"
TRAIN_CANDIDATE_KIND = "expression_turn_v8_expansion_train_candidate_v1"
SOURCE_PUBLIC_KIND = "expression_turn_v8_expansion_separate_blind_review_bundle_v1"

AUTHOR_BUNDLE_KIND = "robot_observable_semantic_author_blind_bundle_v1"
AUTHOR_QUEUE_KIND = "robot_observable_semantic_author_queue_v1"
AUTHOR_PROTOCOL = "robot_observable_semantic_text_author_blind_video_v1"
AUTHOR_SUMMARY_KIND = "robot_observable_semantic_text_author_submission_summary_v1"

MATCHER_BUNDLE_KIND = "robot_observable_semantic_matcher_blind_bundle_v1"
MATCHER_QUEUE_KIND = "robot_observable_semantic_matcher_queue_v1"
MATCHER_PROTOCOL = "robot_observable_semantic_text_matcher_blind_video_v1"
MATCHER_SUMMARY_KIND = "robot_observable_semantic_text_matcher_submission_summary_v1"

HIDDEN_AUTHOR_KIND = "robot_observable_semantic_author_hidden_mapping_v1"
HIDDEN_MATCHER_KIND = "robot_observable_semantic_matcher_hidden_mapping_v1"
MERGED_KIND = "robot_observable_semantic_qualification_v1"
MERGED_SUMMARY_KIND = "robot_observable_semantic_qualification_summary_v1"

TEXT_PROVENANCE = "independently_authored_robot_observable_text_v1"
DURATION_POLICY = "complete_natural_expression_arc_native_variable_length_no_fixed_window"
MATCH_RESULTS = {"observable_match", "mismatch", "ambiguous", "not_observable"}
MIN_MATCH_CONFIDENCE = 0.7

_SAMPLE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_FORBIDDEN_TEXT = re.compile(
    r"\b(?:angry|anger|fear|fearful|happy|happily|sad|sadly|surprise|surprised|"
    r"neutral|emotion|emotional|feels?|feeling|thinks?|thinking|wants?|intends?|"
    r"says?|speaks?|spoken|talks?|transcript)\b",
    flags=re.IGNORECASE,
)


def stable_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Unreadable JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_bound_jsonl(path: Path) -> list[tuple[dict[str, Any], str]]:
    rows: list[tuple[dict[str, Any], str]] = []
    try:
        handle = path.open("rb")
    except OSError as error:
        raise ValueError(f"Unreadable JSONL: {path}") from error
    with handle:
        for line_number, raw in enumerate(handle, 1):
            payload = raw[:-1] if raw.endswith(b"\n") else raw
            if payload.endswith(b"\r"):
                raise ValueError(f"CRLF JSONL is not accepted: {path}:{line_number}")
            if not payload.strip():
                continue
            try:
                value = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid JSONL record: {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object: {path}:{line_number}")
            rows.append((value, hashlib.sha256(payload).hexdigest()))
    return rows


def index_rows(
    rows: Iterable[tuple[dict[str, Any], str]], *, context: str
) -> dict[str, tuple[dict[str, Any], str]]:
    result: dict[str, tuple[dict[str, Any], str]] = {}
    for row, line_sha in rows:
        sample_id = row.get("sample_id")
        if (
            not isinstance(sample_id, str)
            or not _SAMPLE_ID.fullmatch(sample_id)
            or sample_id in result
        ):
            raise ValueError(f"{context} contains an invalid or duplicate sample_id")
        result[sample_id] = (row, line_sha)
    return result


def atomic_json(path: Path, value: object, *, mode: int = 0o644) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def atomic_jsonl(
    path: Path, rows: Iterable[dict[str, Any]], *, mode: int = 0o644
) -> None:
    payload = "".join(stable_json(row) + "\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def _declared_path(owner: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing path binding: {field}")
    path = Path(value)
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve()


def _bound_file(
    owner: Path,
    binding: object,
    *,
    context: str,
    expected: Path | None = None,
) -> Path:
    if not isinstance(binding, dict):
        raise ValueError(f"{context} is not a file binding")
    path = _declared_path(owner, binding.get("path"), field=f"{context}.path")
    if expected is not None and path != expected.resolve():
        raise ValueError(f"{context} path differs from the supplied input")
    if not path.is_file() or binding.get("sha256") != sha256_file(path):
        raise ValueError(f"{context} SHA256 binding mismatch")
    return path


def _fresh_roots(public_root: Path, hidden_root: Path) -> tuple[Path, Path]:
    public_root = public_root.resolve()
    hidden_root = hidden_root.resolve()
    for path in (public_root, hidden_root):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite review bundle: {path}")
    public_root.mkdir(parents=True)
    hidden_root.mkdir(parents=True)
    os.chmod(hidden_root, 0o700)
    return public_root, hidden_root


def _copy_video(source: Path, destination: Path, expected_sha: str) -> None:
    if not source.is_file() or sha256_file(source) != expected_sha:
        raise ValueError(f"anonymous source video hash mismatch: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if destination.stat().st_nlink != 1 or sha256_file(destination) != expected_sha:
        raise ValueError(f"public video copy is not independent and hash-bound: {destination}")
    os.chmod(destination, 0o444)


def _harden_public(public_root: Path, expected_videos: set[str]) -> None:
    expected_root = {"review_queue.jsonl", "summary.json", "videos"}
    actual_root = {path.name for path in public_root.iterdir()}
    if actual_root != expected_root:
        raise ValueError(f"public bundle whitelist mismatch: {sorted(actual_root)}")
    videos = public_root / "videos"
    actual_videos = {path.name for path in videos.iterdir() if path.is_file()}
    if actual_videos != expected_videos or any(not path.is_file() for path in videos.iterdir()):
        raise ValueError("public video whitelist mismatch")
    for path in public_root.rglob("*"):
        if path.is_file():
            if path.stat().st_nlink != 1:
                raise ValueError(f"public file is hard-linked: {path}")
            os.chmod(path, 0o444)
        elif path.is_dir():
            os.chmod(path, 0o555)
    os.chmod(public_root, 0o555)


def _validate_author_text(text: object, text_hash: object) -> str:
    if not isinstance(text, str):
        raise ValueError("candidate_text must be a string")
    candidate = text.strip()
    if candidate != text or "\n" in candidate or not 20 <= len(candidate) <= 300:
        raise ValueError("candidate_text must be one concise 20-300 character line")
    if not candidate.startswith("The robot "):
        raise ValueError("candidate_text must describe visible robot motion starting with 'The robot '")
    if _FORBIDDEN_TEXT.search(candidate):
        raise ValueError("candidate_text contains affect, mental-state, speech, or transcript inference")
    if text_hash != text_sha256(candidate):
        raise ValueError("candidate_text SHA256 mismatch")
    return candidate


def _validate_source_candidates(
    public_summary_path: Path, qualification_summary_path: Path
) -> tuple[
    dict[str, tuple[dict[str, Any], str]],
    dict[str, tuple[dict[str, Any], str]],
    Path,
]:
    public_summary_path = public_summary_path.resolve()
    qualification_summary_path = qualification_summary_path.resolve()
    public = read_json(public_summary_path)
    qualification = read_json(qualification_summary_path)
    if (
        public.get("artifact_kind") != SOURCE_PUBLIC_KIND
        or public.get("accepted_for_training") is not False
        or public.get("fixed_duration_window_used") is not False
        or public.get("all_samples_native_variable_length") is not True
        or qualification.get("artifact_kind") != QUALIFICATION_KIND
        or qualification.get("accepted_for_training") is not False
        or qualification.get("validation_passed") is not True
        or qualification.get("fixed_duration_window_used") is not False
        or qualification.get("native_variable_length") is not True
    ):
        raise ValueError("source qualification is not a fail-closed native-length artifact")
    source_binding = (qualification.get("inputs") or {}).get("public_summary")
    _bound_file(
        qualification_summary_path,
        source_binding,
        context="qualification.inputs.public_summary",
        expected=public_summary_path,
    )
    queue_path = _declared_path(
        public_summary_path,
        public.get("arc_action_queue"),
        field="arc_action_queue",
    )
    if (
        queue_path.parent != public_summary_path.parent
        or not queue_path.is_file()
        or public.get("arc_action_queue_sha256") != sha256_file(queue_path)
    ):
        raise ValueError("source public arc queue binding mismatch")
    queue_binding = (qualification.get("inputs") or {}).get("arc_action_queue")
    _bound_file(
        qualification_summary_path,
        queue_binding,
        context="qualification.inputs.arc_action_queue",
        expected=queue_path,
    )
    candidate_path = _bound_file(
        qualification_summary_path,
        (qualification.get("outputs") or {}).get("train_candidate"),
        context="qualification.outputs.train_candidate",
    )
    queued = index_rows(read_bound_jsonl(queue_path), context="source public queue")
    candidates = index_rows(read_bound_jsonl(candidate_path), context="train candidates")
    declared_count = (qualification.get("outputs") or {}).get("train_candidate", {}).get(
        "records"
    )
    if declared_count != len(candidates) or not candidates:
        raise ValueError("train candidate count is empty or differs from its binding")
    if not set(candidates).issubset(queued):
        raise ValueError("train candidates are not a subset of the anonymous public queue")
    for sample_id, (candidate, _candidate_sha) in candidates.items():
        queue = queued[sample_id][0]
        if (
            candidate.get("artifact_kind") != TRAIN_CANDIDATE_KIND
            or candidate.get("accepted_for_training") is not False
            or candidate.get("license_training_admission") is not False
            or candidate.get("arc_action_complete") is not True
            or candidate.get("action_observability") != "observable"
            or candidate.get("semantic_supervision_mask") is not False
            or candidate.get("native_variable_length") is not True
            or candidate.get("fixed_duration_window_used") is not False
        ):
            raise ValueError(f"{sample_id}: source row is not a complete fail-closed candidate")
        for field in ("video_sha256", "context_level", "frame_count", "fps"):
            if candidate.get(field) != queue.get(field):
                raise ValueError(f"{sample_id}: qualification/public {field} mismatch")
        video = Path(str(queue.get("video_path") or "")).resolve()
        if video.parent != public_summary_path.parent / "videos":
            raise ValueError(f"{sample_id}: source video escapes the anonymous public bundle")
        if not video.is_file() or queue.get("video_sha256") != sha256_file(video):
            raise ValueError(f"{sample_id}: source video SHA256 mismatch")
    return queued, candidates, candidate_path


def build_author_bundle(
    *,
    public_summary: Path,
    qualification_summary: Path,
    public_root: Path,
    hidden_root: Path,
) -> dict[str, Any]:
    queued, candidates, candidate_path = _validate_source_candidates(
        public_summary, qualification_summary
    )
    public_root, hidden_root = _fresh_roots(public_root, hidden_root)
    queue_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    expected_videos: set[str] = set()
    for sample_id in sorted(candidates):
        source, source_line_sha = queued[sample_id]
        candidate, candidate_line_sha = candidates[sample_id]
        filename = f"{sample_id}.mp4"
        destination = public_root / "videos" / filename
        _copy_video(
            Path(str(source["video_path"])).resolve(),
            destination,
            str(source["video_sha256"]),
        )
        expected_videos.add(filename)
        row = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": AUTHOR_QUEUE_KIND,
            "protocol_version": AUTHOR_PROTOCOL,
            "sample_id": sample_id,
            "video_path": str(destination),
            "video_sha256": source["video_sha256"],
            "context_level": source["context_level"],
            "frame_count": source["frame_count"],
            "fps": source["fps"],
            "temporal_unit": "complete_natural_expression_arc",
            "duration_policy": DURATION_POLICY,
            "native_duration_preserved": True,
            "fixed_duration_window_used": False,
            "audio_available": False,
            "source_identity_official_text_and_affect_exposed": False,
            "candidate_text": None,
            "candidate_text_sha256": None,
            "candidate_text_provenance": TEXT_PROVENANCE,
            "observable_description": None,
            "training_admission": False,
        }
        queue_rows.append(row)
        hidden_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_kind": HIDDEN_AUTHOR_KIND,
                "sample_id": sample_id,
                "public_queue_record_sha256": text_sha256(stable_json(row)),
                "source_public_queue_record_sha256": source_line_sha,
                "source_train_candidate_record_sha256": candidate_line_sha,
                "source_trajectory_sha256": candidate.get("trajectory_sha256"),
                "video_sha256": source["video_sha256"],
                "accepted_for_training": False,
            }
        )
    queue_path = public_root / "review_queue.jsonl"
    atomic_jsonl(queue_path, queue_rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": AUTHOR_BUNDLE_KIND,
        "records": len(queue_rows),
        "review_queue": str(queue_path),
        "review_queue_sha256": sha256_file(queue_path),
        "source_public_summary": str(public_summary.resolve()),
        "source_public_summary_sha256": sha256_file(public_summary.resolve()),
        "source_qualification_summary": str(qualification_summary.resolve()),
        "source_qualification_summary_sha256": sha256_file(
            qualification_summary.resolve()
        ),
        "source_train_candidate_manifest_sha256": sha256_file(candidate_path),
        "duration_policy": DURATION_POLICY,
        "all_samples_native_variable_length": True,
        "fixed_duration_window_used": False,
        "source_identity_official_text_and_affect_exposed": False,
        "accepted_for_training": False,
    }
    summary_path = public_root / "summary.json"
    atomic_json(summary_path, summary)
    mapping_path = hidden_root / "sample_mapping.jsonl"
    atomic_jsonl(mapping_path, hidden_rows, mode=0o600)
    hidden_summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": HIDDEN_AUTHOR_KIND,
        "records": len(hidden_rows),
        "public_summary": str(summary_path),
        "public_summary_sha256": sha256_file(summary_path),
        "sample_mapping": str(mapping_path),
        "sample_mapping_sha256": sha256_file(mapping_path),
        "accepted_for_training": False,
    }
    atomic_json(hidden_root / "summary.json", hidden_summary, mode=0o600)
    _harden_public(public_root, expected_videos)
    return summary


def _validate_author_submission(
    *,
    public_summary_path: Path,
    submission_path: Path,
    submission_summary_path: Path,
) -> tuple[
    dict[str, tuple[dict[str, Any], str]],
    dict[str, tuple[dict[str, Any], str]],
    Path,
]:
    public_summary_path = public_summary_path.resolve()
    submission_path = submission_path.resolve()
    submission_summary_path = submission_summary_path.resolve()
    public = read_json(public_summary_path)
    if (
        public.get("artifact_kind") != AUTHOR_BUNDLE_KIND
        or public.get("accepted_for_training") is not False
        or public.get("fixed_duration_window_used") is not False
        or public.get("all_samples_native_variable_length") is not True
    ):
        raise ValueError("author public bundle violates the native blind contract")
    queue_path = _declared_path(
        public_summary_path, public.get("review_queue"), field="review_queue"
    )
    if not queue_path.is_file() or public.get("review_queue_sha256") != sha256_file(queue_path):
        raise ValueError("author public queue SHA256 binding mismatch")
    queue = index_rows(read_bound_jsonl(queue_path), context="author queue")
    submissions = index_rows(read_bound_jsonl(submission_path), context="author submission")
    summary = read_json(submission_summary_path)
    if (
        summary.get("artifact_kind") != AUTHOR_SUMMARY_KIND
        or summary.get("records") != len(submissions)
        or summary.get("coverage_complete") is not True
        or summary.get("training_admission") is not False
        or _declared_path(
            submission_summary_path,
            summary.get("submission_path"),
            field="submission_path",
        )
        != submission_path
        or summary.get("submission_sha256") != sha256_file(submission_path)
        or _declared_path(
            submission_summary_path,
            summary.get("public_summary_path"),
            field="public_summary_path",
        )
        != public_summary_path
        or summary.get("public_summary_sha256") != sha256_file(public_summary_path)
        or _declared_path(
            submission_summary_path,
            summary.get("public_queue_path"),
            field="public_queue_path",
        )
        != queue_path
        or summary.get("public_queue_sha256") != sha256_file(queue_path)
    ):
        raise ValueError("author submission summary binding mismatch")
    if set(submissions) != set(queue):
        raise ValueError("author submission does not exactly cover its public queue")
    reviewer_ids: set[str] = set()
    review_ids: set[str] = set()
    for sample_id, (row, _line_sha) in submissions.items():
        source, source_line_sha = queue[sample_id]
        for field in (
            "sample_id",
            "video_path",
            "video_sha256",
            "context_level",
            "frame_count",
            "fps",
        ):
            if row.get(field) != source.get(field):
                raise ValueError(f"{sample_id}: author {field} binding mismatch")
        reviewer = row.get("reviewer_id")
        review_id = row.get("review_id")
        if not isinstance(reviewer, str) or not reviewer or not isinstance(review_id, str) or not review_id:
            raise ValueError(f"{sample_id}: author reviewer/review identity is missing")
        if review_id in review_ids:
            raise ValueError(f"{sample_id}: duplicate author review_id")
        review_ids.add(review_id)
        reviewer_ids.add(reviewer)
        provenance = row.get("blind_review_provenance")
        video = Path(str(source.get("video_path") or "")).resolve()
        if (
            row.get("protocol_version") != AUTHOR_PROTOCOL
            or row.get("candidate_text_provenance") != TEXT_PROVENANCE
            or row.get("full_decode_to_eof") is not True
            or row.get("decoded_frame_count") != source.get("frame_count")
            or row.get("native_duration_preserved") is not True
            or row.get("fixed_duration_window_used") is not False
            or row.get("audio_available") is not False
            or row.get("label_metadata_exposed") is not False
            or row.get("emotion_inference_performed") is not False
            or row.get("training_admission") is not False
            or not isinstance(provenance, dict)
            or provenance.get("public_summary_sha256") != sha256_file(public_summary_path)
            or provenance.get("public_queue_sha256") != sha256_file(queue_path)
            or provenance.get("public_queue_record_sha256") != source_line_sha
            or not video.is_file()
            or sha256_file(video) != source.get("video_sha256")
        ):
            raise ValueError(f"{sample_id}: author submission violates the blind contract")
        _validate_author_text(row.get("candidate_text"), row.get("candidate_text_sha256"))
        description = row.get("observable_description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"{sample_id}: author observable_description is required")
    if not reviewer_ids:
        raise ValueError("author submission has no reviewer identity")
    return queue, submissions, queue_path


def build_matcher_bundle(
    *,
    author_public_summary: Path,
    author_submission: Path,
    author_submission_summary: Path,
    public_root: Path,
    hidden_root: Path,
) -> dict[str, Any]:
    source_queue, authors, source_queue_path = _validate_author_submission(
        public_summary_path=author_public_summary,
        submission_path=author_submission,
        submission_summary_path=author_submission_summary,
    )
    public_root, hidden_root = _fresh_roots(public_root, hidden_root)
    queue_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    expected_videos: set[str] = set()
    for sample_id in sorted(authors):
        source, _source_line_sha = source_queue[sample_id]
        author, author_line_sha = authors[sample_id]
        filename = f"{sample_id}.mp4"
        destination = public_root / "videos" / filename
        _copy_video(
            Path(str(source["video_path"])).resolve(),
            destination,
            str(source["video_sha256"]),
        )
        expected_videos.add(filename)
        row = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": MATCHER_QUEUE_KIND,
            "protocol_version": MATCHER_PROTOCOL,
            "sample_id": sample_id,
            "video_path": str(destination),
            "video_sha256": source["video_sha256"],
            "context_level": source["context_level"],
            "frame_count": source["frame_count"],
            "fps": source["fps"],
            "temporal_unit": "complete_natural_expression_arc",
            "duration_policy": DURATION_POLICY,
            "native_duration_preserved": True,
            "fixed_duration_window_used": False,
            "audio_available": False,
            "source_identity_official_text_and_affect_exposed": False,
            "candidate_text": author["candidate_text"],
            "candidate_text_sha256": author["candidate_text_sha256"],
            "candidate_text_provenance": TEXT_PROVENANCE,
            "match_result": None,
            "match_confidence": None,
            "match_evidence": None,
            "training_admission": False,
        }
        queue_rows.append(row)
        hidden_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_kind": HIDDEN_MATCHER_KIND,
                "sample_id": sample_id,
                "public_queue_record_sha256": text_sha256(stable_json(row)),
                "author_submission_record_sha256": author_line_sha,
                "author_review_id": author["review_id"],
                "author_reviewer_id": author["reviewer_id"],
                "candidate_text_sha256": author["candidate_text_sha256"],
                "video_sha256": source["video_sha256"],
                "accepted_for_training": False,
            }
        )
    queue_path = public_root / "review_queue.jsonl"
    atomic_jsonl(queue_path, queue_rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": MATCHER_BUNDLE_KIND,
        "records": len(queue_rows),
        "review_queue": str(queue_path),
        "review_queue_sha256": sha256_file(queue_path),
        "source_author_public_summary_sha256": sha256_file(
            author_public_summary.resolve()
        ),
        "source_author_submission_sha256": sha256_file(author_submission.resolve()),
        "source_author_submission_summary_sha256": sha256_file(
            author_submission_summary.resolve()
        ),
        "source_author_queue_sha256": sha256_file(source_queue_path),
        "duration_policy": DURATION_POLICY,
        "all_samples_native_variable_length": True,
        "fixed_duration_window_used": False,
        "source_identity_official_text_and_affect_exposed": False,
        "accepted_for_training": False,
    }
    summary_path = public_root / "summary.json"
    atomic_json(summary_path, summary)
    mapping_path = hidden_root / "sample_mapping.jsonl"
    atomic_jsonl(mapping_path, hidden_rows, mode=0o600)
    hidden_summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": HIDDEN_MATCHER_KIND,
        "records": len(hidden_rows),
        "public_summary": str(summary_path),
        "public_summary_sha256": sha256_file(summary_path),
        "sample_mapping": str(mapping_path),
        "sample_mapping_sha256": sha256_file(mapping_path),
        "accepted_for_training": False,
    }
    atomic_json(hidden_root / "summary.json", hidden_summary, mode=0o600)
    _harden_public(public_root, expected_videos)
    return summary


def merge_matcher_reviews(
    *,
    matcher_public_summary: Path,
    matcher_hidden_summary: Path,
    matcher_submission: Path,
    matcher_submission_summary: Path,
    output_root: Path,
) -> dict[str, Any]:
    public_path = matcher_public_summary.resolve()
    hidden_path = matcher_hidden_summary.resolve()
    submission_path = matcher_submission.resolve()
    submission_summary_path = matcher_submission_summary.resolve()
    public = read_json(public_path)
    hidden = read_json(hidden_path)
    if (
        public.get("artifact_kind") != MATCHER_BUNDLE_KIND
        or public.get("accepted_for_training") is not False
        or public.get("fixed_duration_window_used") is not False
        or public.get("all_samples_native_variable_length") is not True
        or hidden.get("artifact_kind") != HIDDEN_MATCHER_KIND
        or hidden.get("accepted_for_training") is not False
        or _declared_path(hidden_path, hidden.get("public_summary"), field="public_summary")
        != public_path
        or hidden.get("public_summary_sha256") != sha256_file(public_path)
    ):
        raise ValueError("matcher bundle summary violates the fail-closed native contract")
    queue_path = _declared_path(public_path, public.get("review_queue"), field="review_queue")
    if not queue_path.is_file() or public.get("review_queue_sha256") != sha256_file(queue_path):
        raise ValueError("matcher public queue SHA256 binding mismatch")
    mapping_path = _declared_path(
        hidden_path, hidden.get("sample_mapping"), field="sample_mapping"
    )
    if not mapping_path.is_file() or hidden.get("sample_mapping_sha256") != sha256_file(
        mapping_path
    ):
        raise ValueError("matcher hidden mapping SHA256 binding mismatch")
    queue = index_rows(read_bound_jsonl(queue_path), context="matcher queue")
    mappings = index_rows(read_bound_jsonl(mapping_path), context="matcher hidden mapping")
    submissions = index_rows(read_bound_jsonl(submission_path), context="matcher submission")
    summary = read_json(submission_summary_path)
    if (
        set(queue) != set(mappings)
        or set(queue) != set(submissions)
        or public.get("records") != len(queue)
        or hidden.get("records") != len(mappings)
        or summary.get("artifact_kind") != MATCHER_SUMMARY_KIND
        or summary.get("records") != len(submissions)
        or summary.get("coverage_complete") is not True
        or summary.get("training_admission") is not False
        or _declared_path(
            submission_summary_path,
            summary.get("submission_path"),
            field="submission_path",
        )
        != submission_path
        or summary.get("submission_sha256") != sha256_file(submission_path)
        or _declared_path(
            submission_summary_path,
            summary.get("public_summary_path"),
            field="public_summary_path",
        )
        != public_path
        or summary.get("public_summary_sha256") != sha256_file(public_path)
        or _declared_path(
            submission_summary_path,
            summary.get("public_queue_path"),
            field="public_queue_path",
        )
        != queue_path
        or summary.get("public_queue_sha256") != sha256_file(queue_path)
    ):
        raise ValueError("matcher coverage or submission summary binding mismatch")

    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite semantic merge: {output_root}")
    merged_rows: list[dict[str, Any]] = []
    result_counts: Counter[str] = Counter()
    enabled_count = 0
    review_ids: set[str] = set()
    for sample_id in sorted(queue):
        source, source_line_sha = queue[sample_id]
        mapping, _mapping_line_sha = mappings[sample_id]
        row, row_line_sha = submissions[sample_id]
        for field in (
            "sample_id",
            "video_path",
            "video_sha256",
            "context_level",
            "frame_count",
            "fps",
            "candidate_text",
            "candidate_text_sha256",
            "candidate_text_provenance",
        ):
            if row.get(field) != source.get(field):
                raise ValueError(f"{sample_id}: matcher {field} binding mismatch")
        reviewer = row.get("reviewer_id")
        review_id = row.get("review_id")
        result = row.get("match_result")
        confidence = row.get("match_confidence")
        provenance = row.get("blind_review_provenance")
        video = Path(str(source.get("video_path") or "")).resolve()
        if (
            mapping.get("artifact_kind") != HIDDEN_MATCHER_KIND
            or mapping.get("public_queue_record_sha256") != source_line_sha
            or mapping.get("candidate_text_sha256")
            != source.get("candidate_text_sha256")
            or mapping.get("video_sha256") != source.get("video_sha256")
            or not isinstance(reviewer, str)
            or not reviewer
            or reviewer == mapping.get("author_reviewer_id")
            or not isinstance(review_id, str)
            or not review_id
            or review_id == mapping.get("author_review_id")
            or review_id in review_ids
            or result not in MATCH_RESULTS
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
            or row.get("protocol_version") != MATCHER_PROTOCOL
            or row.get("full_decode_to_eof") is not True
            or row.get("decoded_frame_count") != source.get("frame_count")
            or row.get("native_duration_preserved") is not True
            or row.get("fixed_duration_window_used") is not False
            or row.get("audio_available") is not False
            or row.get("label_metadata_exposed") is not False
            or row.get("emotion_inference_performed") is not False
            or row.get("training_admission") is not False
            or not isinstance(provenance, dict)
            or provenance.get("public_summary_sha256") != sha256_file(public_path)
            or provenance.get("public_queue_sha256") != sha256_file(queue_path)
            or provenance.get("public_queue_record_sha256") != source_line_sha
            or not video.is_file()
            or sha256_file(video) != source.get("video_sha256")
        ):
            raise ValueError(f"{sample_id}: matcher submission violates the blind contract")
        review_ids.add(review_id)
        evidence = row.get("match_evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError(f"{sample_id}: matcher evidence is required")
        if result == "observable_match" and float(confidence) < MIN_MATCH_CONFIDENCE:
            raise ValueError(f"{sample_id}: observable match confidence is below threshold")
        candidate = _validate_author_text(
            source.get("candidate_text"), source.get("candidate_text_sha256")
        )
        enabled = result == "observable_match"
        result_counts[result] += 1
        enabled_count += int(enabled)
        merged_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_kind": MERGED_KIND,
                "sample_id": sample_id,
                "context_level": source["context_level"],
                "frame_count": source["frame_count"],
                "fps": source["fps"],
                "video_sha256": source["video_sha256"],
                "candidate_text": candidate if enabled else None,
                "candidate_text_sha256": source["candidate_text_sha256"] if enabled else None,
                "candidate_text_provenance": TEXT_PROVENANCE if enabled else None,
                "semantic_match_result": result,
                "semantic_match_confidence": float(confidence),
                "semantic_supervision_mask": enabled,
                "native_variable_length": True,
                "fixed_duration_window_used": False,
                "author_review_id": mapping["author_review_id"],
                "author_submission_record_sha256": mapping[
                    "author_submission_record_sha256"
                ],
                "matcher_review_id": review_id,
                "matcher_submission_record_sha256": row_line_sha,
                "license_training_admission": False,
                "accepted_for_training": False,
                "training_blockers": [
                    "license_training_admission_unconfirmed",
                    "formal_train_manifest_not_built",
                ],
            }
        )
    output_root.mkdir(parents=True)
    os.chmod(output_root, 0o700)
    output_path = output_root / "semantic_qualification.jsonl"
    atomic_jsonl(output_path, merged_rows, mode=0o600)
    merged_summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": MERGED_SUMMARY_KIND,
        "records": len(merged_rows),
        "semantic_supervision_enabled": enabled_count,
        "result_distribution": {
            key: result_counts.get(key, 0) for key in sorted(MATCH_RESULTS)
        },
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "matcher_public_summary_sha256": sha256_file(public_path),
        "matcher_hidden_summary_sha256": sha256_file(hidden_path),
        "matcher_submission_sha256": sha256_file(submission_path),
        "matcher_submission_summary_sha256": sha256_file(submission_summary_path),
        "duration_policy": DURATION_POLICY,
        "native_variable_length": True,
        "fixed_duration_window_used": False,
        "license_training_admission": False,
        "accepted_for_training": False,
    }
    atomic_json(output_root / "summary.json", merged_summary, mode=0o600)
    return merged_summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    author = subparsers.add_parser("author")
    author.add_argument("--public-summary", type=Path, required=True)
    author.add_argument("--qualification-summary", type=Path, required=True)
    author.add_argument("--public-root", type=Path, required=True)
    author.add_argument("--hidden-root", type=Path, required=True)

    matcher = subparsers.add_parser("matcher")
    matcher.add_argument("--author-public-summary", type=Path, required=True)
    matcher.add_argument("--author-submission", type=Path, required=True)
    matcher.add_argument("--author-submission-summary", type=Path, required=True)
    matcher.add_argument("--public-root", type=Path, required=True)
    matcher.add_argument("--hidden-root", type=Path, required=True)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--matcher-public-summary", type=Path, required=True)
    merge.add_argument("--matcher-hidden-summary", type=Path, required=True)
    merge.add_argument("--matcher-submission", type=Path, required=True)
    merge.add_argument("--matcher-submission-summary", type=Path, required=True)
    merge.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "author":
        result = build_author_bundle(
            public_summary=args.public_summary,
            qualification_summary=args.qualification_summary,
            public_root=args.public_root,
            hidden_root=args.hidden_root,
        )
    elif args.command == "matcher":
        result = build_matcher_bundle(
            author_public_summary=args.author_public_summary,
            author_submission=args.author_submission,
            author_submission_summary=args.author_submission_summary,
            public_root=args.public_root,
            hidden_root=args.hidden_root,
        )
    else:
        result = merge_matcher_reviews(
            matcher_public_summary=args.matcher_public_summary,
            matcher_hidden_summary=args.matcher_hidden_summary,
            matcher_submission=args.matcher_submission,
            matcher_submission_summary=args.matcher_submission_summary,
            output_root=args.output_root,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
