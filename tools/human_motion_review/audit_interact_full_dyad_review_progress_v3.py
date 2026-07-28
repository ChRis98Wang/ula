#!/usr/bin/env python3
"""Audit partial InterAct v3 blind-review shard coverage without admitting data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from tools.human_motion_review.finalize_interact_full_dyad_review_shard_v3 import (
    SHARD_KIND,
    SUBMISSION_KIND,
    SUMMARY_KIND,
    read_json,
    read_jsonl,
    sha256_file,
    value_sha256,
)
from tools.human_motion_review.merge_interact_full_dyad_review_shards_v3 import (
    SHARDS_KIND,
)


REPORT_KIND = "interact_full_dyad_blind_review_progress_audit_v3"


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


def _audit_reviewed_shard(
    *,
    root: dict[str, Any],
    root_summary_path: Path,
    shard_entry: dict[str, Any],
    submission_summary_path: Path,
) -> dict[str, Any]:
    index = shard_entry["shard_index"]
    source_summary_path = _path(
        root_summary_path, shard_entry.get("summary"), field="source shard summary"
    )
    if (
        not source_summary_path.is_file()
        or sha256_file(source_summary_path) != shard_entry.get("summary_sha256")
    ):
        raise ValueError("source shard summary SHA mismatch")
    source = read_json(source_summary_path)
    source_queue_path = _path(
        source_summary_path, source.get("review_queue"), field="source shard queue"
    )
    if (
        source.get("artifact_kind") != SHARD_KIND
        or source.get("queue_kind") != root.get("queue_kind")
        or source.get("shard_index") != index
        or source.get("accepted_for_training") is not False
        or source.get("fixed_duration_window_used") is not False
        or source.get("review_queue_sha256")
        != shard_entry.get("review_queue_sha256")
        or not source_queue_path.is_file()
        or sha256_file(source_queue_path) != source.get("review_queue_sha256")
    ):
        raise ValueError("source shard binding mismatch")
    queued = _index(read_jsonl(source_queue_path), context="source shard queue")
    if len(queued) != shard_entry.get("records"):
        raise ValueError("source shard record count mismatch")

    summary = read_json(submission_summary_path)
    submission_path = _path(
        submission_summary_path, summary.get("submission"), field="submission"
    )
    if (
        summary.get("artifact_kind") != SUMMARY_KIND
        or summary.get("queue_kind") != root.get("queue_kind")
        or summary.get("shard_index") != index
        or summary.get("shard_count") != root.get("shard_count")
        or summary.get("records") != len(queued)
        or summary.get("shard_summary_sha256") != sha256_file(source_summary_path)
        or summary.get("shard_queue_sha256") != source.get("review_queue_sha256")
        or summary.get("source_public_summary_sha256")
        != root.get("source_public_summary_sha256")
        or summary.get("source_queue_sha256") != root.get("source_queue_sha256")
        or summary.get("coverage_complete_without_overlap") is not True
        or summary.get("all_videos_hash_verified_and_probed_to_eof") is not True
        or summary.get("accepted_for_training") is not False
        or summary.get("fixed_duration_window_used") is not False
        or not submission_path.is_file()
        or submission_path.parent != submission_summary_path.parent
        or sha256_file(submission_path) != summary.get("submission_sha256")
    ):
        raise ValueError("reviewed shard summary binding mismatch")
    reviewed = _index(read_jsonl(submission_path), context="reviewed shard submission")
    if set(reviewed) != set(queued):
        raise ValueError("reviewed shard coverage mismatch")
    reviewer_id = summary.get("reviewer_id")
    if not isinstance(reviewer_id, str) or not reviewer_id:
        raise ValueError("reviewer ID is missing")
    for sample_id, row in reviewed.items():
        source_row = queued[sample_id]
        provenance = row.get("blind_review_provenance")
        if (
            row.get("artifact_kind") != SUBMISSION_KIND
            or row.get("reviewer_id") != reviewer_id
            or row.get("video_path") != source_row.get("video_path")
            or row.get("video_sha256") != source_row.get("video_sha256")
            or row.get("decode_complete") is not True
            or row.get("accepted_for_training") is not False
            or row.get("fixed_duration_window_used") is not False
            or not isinstance(provenance, dict)
            or provenance.get("shard_summary_sha256") != sha256_file(source_summary_path)
            or provenance.get("shard_queue_sha256")
            != source.get("review_queue_sha256")
            or provenance.get("source_public_summary_sha256")
            != root.get("source_public_summary_sha256")
            or provenance.get("source_queue_sha256") != root.get("source_queue_sha256")
            or provenance.get("queue_record_hash_method") != "sha256_stable_json_utf8"
            or provenance.get("queue_record_sha256") != value_sha256(source_row)
        ):
            raise ValueError(f"reviewed row binding mismatch: {sample_id}")
    return {
        "shard_index": index,
        "records": len(reviewed),
        "reviewer_id": reviewer_id,
        "summary": str(submission_summary_path),
        "summary_sha256": sha256_file(submission_summary_path),
        "submission_sha256": sha256_file(submission_path),
    }


def audit(*, shards_summary_path: Path, submissions_root: Path) -> dict[str, Any]:
    shards_summary_path = shards_summary_path.resolve()
    submissions_root = submissions_root.resolve()
    root = read_json(shards_summary_path)
    entries = root.get("shards")
    if (
        root.get("artifact_kind") != SHARDS_KIND
        or root.get("queue_kind") not in {"arc_action", "affect"}
        or root.get("accepted_for_training") is not False
        or root.get("fixed_duration_window_used") is not False
        or root.get("coverage_complete_without_overlap") is not True
        or not isinstance(entries, list)
        or len(entries) != root.get("shard_count")
    ):
        raise ValueError("shard-set summary violates its fail-closed coverage contract")

    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    missing: list[int] = []
    for expected_index, entry in enumerate(entries):
        if not isinstance(entry, dict) or entry.get("shard_index") != expected_index:
            raise ValueError("source shard indexes are not complete and ordered")
        summary_path = submissions_root / f"shard_{expected_index:03d}" / "summary.json"
        if not summary_path.is_file():
            missing.append(expected_index)
            continue
        try:
            valid.append(
                _audit_reviewed_shard(
                    root=root,
                    root_summary_path=shards_summary_path,
                    shard_entry=entry,
                    submission_summary_path=summary_path.resolve(),
                )
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            invalid.append({"shard_index": expected_index, "error": str(error)})

    reviewed_records = sum(item["records"] for item in valid)
    expected_records = int(root["records"])
    return {
        "artifact_kind": REPORT_KIND,
        "schema_version": "3.0.0",
        "queue_kind": root["queue_kind"],
        "source_shards_summary": str(shards_summary_path),
        "source_shards_summary_sha256": sha256_file(shards_summary_path),
        "submissions_root": str(submissions_root),
        "expected_shards": root["shard_count"],
        "valid_reviewed_shards": len(valid),
        "valid_reviewed_shard_indexes": [item["shard_index"] for item in valid],
        "missing_shard_indexes": missing,
        "invalid_shards": invalid,
        "expected_records": expected_records,
        "valid_reviewed_records": reviewed_records,
        "coverage_fraction": reviewed_records / expected_records,
        "merge_ready": not missing and not invalid and reviewed_records == expected_records,
        "accepted_for_training": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-summary", type=Path, required=True)
    parser.add_argument("--submissions-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            audit(
                shards_summary_path=args.shards_summary,
                submissions_root=args.submissions_root,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
