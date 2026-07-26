#!/usr/bin/env python3
"""Normalize InterAct arc-review bindings without changing judgments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


LEGACY_QUEUE_HASH_FIELD = "arc_action_review_queue_sha256"
QUEUE_HASH_FIELD = "arc_action_queue_sha256"
SUMMARY_KIND = "interact_dyadic_arc_action_blind_review_submission_v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl_with_hashes(path: Path) -> list[tuple[dict[str, Any], str]]:
    rows: list[tuple[dict[str, Any], str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.rstrip("\r\n")
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append((value, hashlib.sha256(line.encode("utf-8")).hexdigest()))
    return rows


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    )
    _atomic_write(path, payload)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _atomic_write(path: Path, payload: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def normalize_provenance(row: dict[str, Any], queue_sha256: str) -> dict[str, Any]:
    provenance = row.get("blind_review_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Arc review lacks blind_review_provenance")
    declared = {
        value
        for field in (QUEUE_HASH_FIELD, LEGACY_QUEUE_HASH_FIELD)
        if isinstance((value := provenance.get(field)), str)
    }
    if declared != {queue_sha256}:
        raise ValueError("Arc review queue hash binding mismatch")
    normalized = dict(row)
    normalized_provenance = dict(provenance)
    normalized_provenance[QUEUE_HASH_FIELD] = queue_sha256
    normalized["blind_review_provenance"] = normalized_provenance
    completeness = normalized.get("expression_completeness_result")
    if completeness == "incomplete":
        if not isinstance(normalized.get("expansion_request"), dict) or not any(
            normalized.get(f"{phase}_status") == "incomplete"
            for phase in ("onset", "apex", "offset")
        ):
            raise ValueError("Incomplete arc review lacks expansion evidence")
        normalized["expression_completeness_result"] = "incomplete_requires_expansion"
    elif completeness not in {"complete", "incomplete_requires_expansion"}:
        raise ValueError("Unknown arc completeness result")
    return normalized


def normalize(
    *,
    public_summary_path: Path,
    source_submission_path: Path,
    source_summary_path: Path,
    output_submission_path: Path,
    output_summary_path: Path,
) -> dict[str, Any]:
    paths = [
        public_summary_path,
        source_submission_path,
        source_summary_path,
        output_submission_path,
        output_summary_path,
    ]
    public_summary_path, source_submission_path, source_summary_path, output_submission_path, output_summary_path = (
        path.resolve() for path in paths
    )
    if output_submission_path.exists() or output_summary_path.exists():
        raise FileExistsError("Normalization outputs already exist")

    public = read_json(public_summary_path)
    queue_value = public.get("arc_action_queue") or public.get("arc_action_review_queue")
    if not isinstance(queue_value, str):
        raise ValueError("Public summary lacks an arc-action queue path")
    queue_path = Path(queue_value).resolve()
    queue_sha256 = sha256_file(queue_path)
    public_queue_hashes = {
        value
        for field in ("arc_action_queue_sha256", "arc_action_review_queue_sha256")
        if isinstance((value := public.get(field)), str)
    }
    if public_queue_hashes != {queue_sha256}:
        raise ValueError("Public arc queue hash binding mismatch")
    public_sha256 = sha256_file(public_summary_path)

    source_summary = read_json(source_summary_path)
    source_submission_sha256 = sha256_file(source_submission_path)
    source_bindings = source_summary.get("input_bindings") or {}
    source_public_hashes = {
        value
        for value in (
            source_summary.get("public_summary_sha256"),
            source_bindings.get("public_summary_sha256"),
        )
        if isinstance(value, str)
    }
    source_queue_hashes = {
        value
        for value in (
            source_summary.get("arc_action_review_queue_sha256"),
            source_summary.get("arc_action_queue_sha256"),
            source_bindings.get("arc_action_queue_sha256"),
        )
        if isinstance(value, str)
    }
    if (
        source_summary.get("artifact_kind") != SUMMARY_KIND
        or source_summary.get("record_count") is None
        or source_summary.get("output_jsonl_sha256") != source_submission_sha256
        or source_public_hashes != {public_sha256}
        or source_queue_hashes != {queue_sha256}
        or source_summary.get("accepted_for_training") is not False
    ):
        raise ValueError("Source arc-review summary binding mismatch")

    queue_rows = read_jsonl_with_hashes(queue_path)
    source_rows = read_jsonl_with_hashes(source_submission_path)
    if len(source_rows) != len(queue_rows) or len(source_rows) != source_summary["record_count"]:
        raise ValueError("Arc-review coverage mismatch")
    queue_by_id = {row["sample_id"]: (row, line_sha) for row, line_sha in queue_rows}
    if len(queue_by_id) != len(queue_rows):
        raise ValueError("Public arc queue contains duplicate sample IDs")

    normalized_rows: list[dict[str, Any]] = []
    for row, _source_line_sha in source_rows:
        sample_id = row.get("sample_id")
        if sample_id not in queue_by_id:
            raise ValueError(f"Unexpected arc review sample: {sample_id}")
        queued, queue_line_sha = queue_by_id[sample_id]
        provenance = row.get("blind_review_provenance") or {}
        declared_record_hash = provenance.get("queue_record_sha256")
        declared_hash_method = provenance.get("queue_record_hash_method")
        if (
            row.get("video_path") != queued.get("video_path")
            or row.get("video_sha256") != queued.get("video_sha256")
            or provenance.get("public_summary_sha256") != public_sha256
            or declared_record_hash not in {None, queue_line_sha}
            or declared_hash_method not in {None, "sha256_utf8_line_without_lf"}
        ):
            raise ValueError(f"Arc review provenance mismatch: {sample_id}")
        video = Path(row["video_path"]).resolve()
        if not video.is_file() or sha256_file(video) != row["video_sha256"]:
            raise ValueError(f"Arc review video binding mismatch: {sample_id}")
        normalized = normalize_provenance(row, queue_sha256)
        normalized["blind_review_provenance"]["queue_record_hash_method"] = (
            "sha256_utf8_line_without_lf"
        )
        normalized["blind_review_provenance"]["queue_record_sha256"] = queue_line_sha
        normalized_rows.append(normalized)

    atomic_write_jsonl(output_submission_path, normalized_rows)
    output_submission_sha256 = sha256_file(output_submission_path)
    normalized_summary = dict(source_summary)
    normalized_summary.update(
        {
            "input_bindings": {
                "arc_action_queue_sha256": queue_sha256,
                "public_summary_sha256": public_sha256,
            },
            "normalization": {
                "judgments_changed": False,
                "normalized_fields": [
                    "blind_review_provenance.arc_action_queue_sha256",
                    "blind_review_provenance.queue_record_hash_method",
                    "blind_review_provenance.queue_record_sha256",
                    "expression_completeness_result_enum",
                ],
                "source_submission_sha256": source_submission_sha256,
                "source_summary_sha256": sha256_file(source_summary_path),
            },
            "output_jsonl": str(output_submission_path),
            "output_jsonl_sha256": output_submission_sha256,
        }
    )
    atomic_write_json(output_summary_path, normalized_summary)
    return normalized_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-summary", type=Path, required=True)
    parser.add_argument("--source-submission", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--output-submission", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = normalize(
        public_summary_path=args.public_summary,
        source_submission_path=args.source_submission,
        source_summary_path=args.source_summary,
        output_submission_path=args.output_submission,
        output_summary_path=args.output_summary,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
