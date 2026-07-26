#!/usr/bin/env python3
"""Normalize a blind affect submission without changing its judgments."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "1.0.0"
PUBLIC_KIND = "expression_turn_v8_expansion_separate_blind_review_bundle_v1"
PROTOCOL = "robot_affect_blind_video_v1"
SUMMARY_KIND = "robot_affect_blind_review_submission_summary_v1"
ALLOWED_CLASSES = ("angry", "fear", "happy", "neutral", "sad", "surprise")
RESULTS = ("observable", "ambiguous", "not_observable")


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


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Unreadable JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
                raise ValueError(f"Invalid JSONL: {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object: {path}:{line_number}")
            rows.append(value)
    return rows


def index_rows(rows: Iterable[dict[str, Any]], *, context: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in result:
            raise ValueError(f"{context} contains an invalid or duplicate sample_id")
        result[sample_id] = row
    return result


def _declared_path(owner: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing path: {field}")
    path = Path(value)
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        "".join(stable_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def normalize_submission(
    *,
    public_summary_path: Path,
    source_submission_path: Path,
    source_summary_path: Path,
    output_submission_path: Path,
    output_summary_path: Path,
) -> dict[str, Any]:
    supplied = [
        public_summary_path.resolve(),
        source_submission_path.resolve(),
        source_summary_path.resolve(),
    ]
    if any(not path.is_file() for path in supplied):
        raise ValueError("normalization input is missing")
    public_summary_path, source_submission_path, source_summary_path = supplied
    output_submission_path = output_submission_path.resolve()
    output_summary_path = output_summary_path.resolve()
    if output_submission_path.exists() or output_summary_path.exists():
        raise FileExistsError("refusing to overwrite normalized affect artifacts")

    public = read_json(public_summary_path)
    if (
        public.get("artifact_kind") != PUBLIC_KIND
        or public.get("accepted_for_training") is not False
        or public.get("all_samples_native_variable_length") is not True
        or public.get("fixed_duration_window_used") is not False
    ):
        raise ValueError("public bundle violates the native variable-length contract")
    queue_path = _declared_path(
        public_summary_path, public.get("affect_queue"), field="affect_queue"
    )
    queue_sha = sha256_file(queue_path) if queue_path.is_file() else None
    if (
        queue_path.parent != public_summary_path.parent
        or queue_sha != public.get("affect_queue_sha256")
    ):
        raise ValueError("public affect queue binding mismatch")
    queue = index_rows(read_jsonl(queue_path), context="public affect queue")
    source = index_rows(read_jsonl(source_submission_path), context="source submission")
    source_summary = read_json(source_summary_path)
    if (
        source_summary.get("source_public_summary_sha256")
        != sha256_file(public_summary_path)
        or source_summary.get("source_affect_queue_sha256") != queue_sha
        or source_summary.get("records_expected") != len(queue)
        or source_summary.get("records_reviewed") != len(source)
        or source_summary.get("coverage", {}).get("fraction") != 1
        or source_summary.get("strict_blind_attestation", {}).get(
            "fixed_seconds_or_fixed_duration_window_used"
        )
        is not False
        or source_summary.get("strict_blind_attestation", {}).get(
            "all_training_admission_false"
        )
        is not True
    ):
        raise ValueError("source summary coverage, hash, or blind attestation mismatch")
    if set(source) != set(queue):
        raise ValueError("source affect coverage differs from public queue")

    normalized: list[dict[str, Any]] = []
    result_counts: Counter[str] = Counter()
    for sample_id in sorted(queue):
        queued = queue[sample_id]
        row = source[sample_id]
        for field in (
            "sample_id",
            "video_sha256",
            "context_level",
            "fps",
            "frame_count",
            "allowed_classes",
        ):
            if row.get(field) != queued.get(field):
                raise ValueError(f"{sample_id}: source/public {field} mismatch")
        result = row.get("result")
        predicted = row.get("predicted_class")
        confidence = row.get("confidence")
        if result == "observable":
            if (
                predicted not in ALLOWED_CLASSES
                or isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(float(confidence))
                or not 0.0 <= float(confidence) <= 1.0
            ):
                raise ValueError(f"{sample_id}: invalid observable judgment")
        elif result in {"ambiguous", "not_observable"}:
            if predicted is not None or confidence is not None:
                raise ValueError(f"{sample_id}: non-observable judgment carries a pseudo-label")
        else:
            raise ValueError(f"{sample_id}: invalid affect result")
        reviewer = row.get("affect_reviewer_id")
        if (
            row.get("affect_protocol_version") != PROTOCOL
            or not isinstance(reviewer, str)
            or not reviewer
            or row.get("source_affect_queue_sha256") != queue_sha
            or row.get("full_video_reviewed") is not True
            or row.get("decode_started_at_frame") != 0
            or row.get("decode_reached_eof") is not True
            or row.get("decoded_frame_count") != queued.get("frame_count")
            or row.get("frame_count_verified") is not True
            or row.get("video_sha256_verified") is not True
            or row.get("native_duration_preserved") is not True
            or row.get("fixed_duration_window_used") is not False
            or row.get("training_admission") is not False
            or row.get("label_metadata_exposed") is not False
            or row.get("audio_available") is not False
        ):
            raise ValueError(f"{sample_id}: source evidence binding is incomplete")
        video = Path(str(queued.get("video_path") or "")).resolve()
        if not video.is_file() or sha256_file(video) != queued.get("video_sha256"):
            raise ValueError(f"{sample_id}: public video SHA256 changed")
        source_row_sha = hashlib.sha256(stable_json(row).encode("utf-8")).hexdigest()
        output = dict(row)
        output.update(
            {
                "affect_review_id": f"{reviewer}__{sample_id}",
                "public_queue_sha256": queue_sha,
                "normalization": {
                    "policy": "mechanical_schema_normalization_no_judgment_change_v1",
                    "source_affect_review_id": row.get("affect_review_id"),
                    "source_record_sha256": source_row_sha,
                    "source_submission_sha256": sha256_file(source_submission_path),
                    "source_summary_sha256": sha256_file(source_summary_path),
                },
            }
        )
        normalized.append(output)
        result_counts[str(result)] += 1

    atomic_jsonl(output_submission_path, normalized)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": SUMMARY_KIND,
        "submission_path": str(output_submission_path),
        "submission_sha256": sha256_file(output_submission_path),
        "public_queue_path": str(queue_path),
        "public_queue_sha256": queue_sha,
        "coverage": {
            "expected_records": len(queue),
            "reviewed_records": len(normalized),
            "complete": True,
        },
        "integrity": {
            "all_full_video_reviewed": True,
            "all_native_variable_length_reviewed": True,
            "fixed_duration_window_used": False,
            "judgments_changed_by_normalization": False,
        },
        "result_distribution": {
            key: result_counts.get(key, 0) for key in RESULTS
        },
        "source_submission_path": str(source_submission_path),
        "source_submission_sha256": sha256_file(source_submission_path),
        "source_summary_path": str(source_summary_path),
        "source_summary_sha256": sha256_file(source_summary_path),
        "training_admission": False,
    }
    atomic_json(output_summary_path, summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-summary", type=Path, required=True)
    parser.add_argument("--source-submission", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--output-submission", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = normalize_submission(
        public_summary_path=args.public_summary,
        source_submission_path=args.source_submission,
        source_summary_path=args.source_summary,
        output_submission_path=args.output_submission,
        output_summary_path=args.output_summary,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
