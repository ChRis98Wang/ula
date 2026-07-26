#!/usr/bin/env python3
"""Merge a complete set of validated InterAct v3 blind-review shards."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from tools.human_motion_review.finalize_interact_full_dyad_review_shard_v3 import (
    SHARD_KIND,
    SUBMISSION_KIND,
    SUMMARY_KIND as SHARD_SUBMISSION_SUMMARY_KIND,
    read_json,
    read_jsonl,
    sha256_file,
    stable_json,
    value_sha256,
)


SHARDS_KIND = "anonymous_blind_review_queue_shards_summary_v1"
MERGE_KIND = "interact_full_dyad_blind_review_shards_merge_v3"


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _path(owner: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing declared path: {field}")
    path = Path(value)
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve()


def _index(rows: list[dict[str, Any]], *, context: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in result:
            raise ValueError(f"{context} contains an invalid or duplicate sample ID")
        result[sample_id] = row
    return result


def merge(
    *,
    shards_summary_path: Path,
    submissions_root: Path,
    output_submission_path: Path,
    output_summary_path: Path,
) -> dict[str, Any]:
    shards_summary_path = shards_summary_path.resolve()
    submissions_root = submissions_root.resolve()
    output_submission_path = output_submission_path.resolve()
    output_summary_path = output_summary_path.resolve()
    if output_submission_path.exists() or output_summary_path.exists():
        raise FileExistsError("Merged review outputs already exist")

    root = read_json(shards_summary_path)
    queue_kind = root.get("queue_kind")
    shard_entries = root.get("shards")
    if (
        root.get("artifact_kind") != SHARDS_KIND
        or queue_kind not in {"arc_action", "affect"}
        or root.get("accepted_for_training") is not False
        or root.get("fixed_duration_window_used") is not False
        or root.get("coverage_complete_without_overlap") is not True
        or not isinstance(shard_entries, list)
        or len(shard_entries) != root.get("shard_count")
    ):
        raise ValueError("Shard-set summary violates its fail-closed coverage contract")

    all_rows: dict[str, dict[str, Any]] = {}
    reviewers: Counter[str] = Counter()
    reviewed_shards: list[dict[str, Any]] = []
    for expected_index, shard_entry in enumerate(shard_entries):
        if not isinstance(shard_entry, dict) or shard_entry.get("shard_index") != expected_index:
            raise ValueError("Shard-set indexes are not complete and ordered")
        source_summary_path = _path(
            shards_summary_path, shard_entry.get("summary"), field="shard summary"
        )
        if (
            not source_summary_path.is_file()
            or sha256_file(source_summary_path) != shard_entry.get("summary_sha256")
        ):
            raise ValueError(f"Source shard summary SHA mismatch: {expected_index}")
        source_summary = read_json(source_summary_path)
        source_queue_path = _path(
            source_summary_path, source_summary.get("review_queue"), field="shard queue"
        )
        if (
            source_summary.get("artifact_kind") != SHARD_KIND
            or source_summary.get("queue_kind") != queue_kind
            or source_summary.get("shard_index") != expected_index
            or source_summary.get("review_queue_sha256") != shard_entry.get(
                "review_queue_sha256"
            )
            or sha256_file(source_queue_path) != source_summary.get("review_queue_sha256")
        ):
            raise ValueError(f"Source shard binding mismatch: {expected_index}")
        queued = _index(read_jsonl(source_queue_path), context=f"shard {expected_index} queue")

        submission_summary_path = (
            submissions_root / f"shard_{expected_index:03d}" / "summary.json"
        ).resolve()
        if not submission_summary_path.is_file():
            raise ValueError(f"Missing reviewed shard: {expected_index}")
        submission_summary = read_json(submission_summary_path)
        submission_path = _path(
            submission_summary_path,
            submission_summary.get("submission"),
            field="shard submission",
        )
        if (
            submission_summary.get("artifact_kind") != SHARD_SUBMISSION_SUMMARY_KIND
            or submission_summary.get("queue_kind") != queue_kind
            or submission_summary.get("shard_index") != expected_index
            or submission_summary.get("shard_count") != root.get("shard_count")
            or submission_summary.get("records") != len(queued)
            or submission_summary.get("shard_summary_sha256")
            != sha256_file(source_summary_path)
            or submission_summary.get("shard_queue_sha256")
            != source_summary.get("review_queue_sha256")
            or submission_summary.get("source_public_summary_sha256")
            != root.get("source_public_summary_sha256")
            or submission_summary.get("source_queue_sha256")
            != root.get("source_queue_sha256")
            or submission_summary.get("coverage_complete_without_overlap") is not True
            or submission_summary.get("all_videos_hash_verified_and_probed_to_eof") is not True
            or submission_summary.get("accepted_for_training") is not False
            or submission_summary.get("fixed_duration_window_used") is not False
            or not submission_path.is_file()
            or submission_path.parent != submission_summary_path.parent
            or sha256_file(submission_path) != submission_summary.get("submission_sha256")
        ):
            raise ValueError(f"Reviewed shard summary binding mismatch: {expected_index}")
        reviewed = _index(
            read_jsonl(submission_path), context=f"shard {expected_index} submission"
        )
        if set(reviewed) != set(queued):
            raise ValueError(f"Reviewed shard coverage mismatch: {expected_index}")
        reviewer_id = submission_summary.get("reviewer_id")
        if not isinstance(reviewer_id, str) or not reviewer_id:
            raise ValueError(f"Reviewed shard lacks a reviewer: {expected_index}")
        reviewers[reviewer_id] += len(reviewed)

        for sample_id, row in reviewed.items():
            source = queued[sample_id]
            if sample_id in all_rows:
                raise ValueError(f"Cross-shard duplicate sample ID: {sample_id}")
            if (
                row.get("artifact_kind") != SUBMISSION_KIND
                or row.get("reviewer_id") != reviewer_id
                or row.get("video_path") != source.get("video_path")
                or row.get("video_sha256") != source.get("video_sha256")
                or row.get("decode_complete") is not True
                or row.get("accepted_for_training") is not False
                or row.get("fixed_duration_window_used") is not False
            ):
                raise ValueError(f"Reviewed row violates its source binding: {sample_id}")
            provenance = row.get("blind_review_provenance")
            if not isinstance(provenance, dict) or (
                provenance.get("shard_summary_sha256") != sha256_file(source_summary_path)
                or provenance.get("shard_queue_sha256")
                != source_summary.get("review_queue_sha256")
                or provenance.get("source_public_summary_sha256")
                != root.get("source_public_summary_sha256")
                or provenance.get("source_queue_sha256") != root.get("source_queue_sha256")
                or provenance.get("queue_record_hash_method")
                != "sha256_stable_json_utf8"
                or provenance.get("queue_record_sha256") != value_sha256(source)
            ):
                raise ValueError(f"Reviewed row provenance mismatch: {sample_id}")
            all_rows[sample_id] = row
        reviewed_shards.append(
            {
                "shard_index": expected_index,
                "reviewer_id": reviewer_id,
                "records": len(reviewed),
                "summary": str(submission_summary_path),
                "summary_sha256": sha256_file(submission_summary_path),
                "submission_sha256": sha256_file(submission_path),
            }
        )

    if len(all_rows) != root.get("records"):
        raise ValueError("Merged review count differs from the full source queue")
    ordered = [all_rows[sample_id] for sample_id in sorted(all_rows)]
    atomic_write(
        output_submission_path,
        "".join(stable_json(row) + "\n" for row in ordered),
    )
    summary = {
        "artifact_kind": MERGE_KIND,
        "schema_version": "3.0.0",
        "queue_kind": queue_kind,
        "records": len(ordered),
        "shard_count": len(reviewed_shards),
        "submission": str(output_submission_path),
        "submission_sha256": sha256_file(output_submission_path),
        "source_shards_summary": str(shards_summary_path),
        "source_shards_summary_sha256": sha256_file(shards_summary_path),
        "source_public_summary_sha256": root["source_public_summary_sha256"],
        "source_queue_sha256": root["source_queue_sha256"],
        "reviewer_record_distribution": dict(sorted(reviewers.items())),
        "reviewed_shards": reviewed_shards,
        "coverage_complete_without_overlap": True,
        "all_shards_fail_closed_and_hash_bound": True,
        "fixed_duration_window_used": False,
        "accepted_for_training": False,
    }
    atomic_write(output_summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-summary", type=Path, required=True)
    parser.add_argument("--submissions-root", type=Path, required=True)
    parser.add_argument("--output-submission", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = merge(
        shards_summary_path=args.shards_summary,
        submissions_root=args.submissions_root,
        output_submission_path=args.output_submission,
        output_summary_path=args.output_summary,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
