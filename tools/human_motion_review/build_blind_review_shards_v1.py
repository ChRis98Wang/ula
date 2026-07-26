#!/usr/bin/env python3
"""Split an anonymous blind-review queue into balanced hash-bound shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


QUEUE_FIELDS = {
    "arc_action": ("arc_action_queue", "arc_action_queue_sha256"),
    "affect": ("affect_queue", "affect_queue_sha256"),
}


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _atomic_write(path: Path, payload: str) -> None:
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


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    )
    _atomic_write(path, payload)


def assign_shards(
    rows: list[dict[str, Any]], *, queue_sha256: str, shard_count: int
) -> list[list[dict[str, Any]]]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    sample_ids = [row.get("sample_id") for row in rows]
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids):
        raise ValueError("Every review row must have a non-empty sample_id")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Review queue contains duplicate sample IDs")
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{queue_sha256}\0{row['sample_id']}".encode("utf-8")
        ).hexdigest(),
    )
    shards = [[] for _ in range(shard_count)]
    for index, row in enumerate(ordered):
        shards[index % shard_count].append(row)
    return shards


def build_shards(
    *,
    public_summary_path: Path,
    queue_kind: str,
    shard_count: int,
    output_root: Path,
) -> dict[str, Any]:
    public_summary_path = public_summary_path.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    if queue_kind not in QUEUE_FIELDS:
        raise ValueError(f"Unsupported queue kind: {queue_kind}")

    public = read_json(public_summary_path)
    if public.get("accepted_for_training") is not False:
        raise ValueError("Source public bundle unexpectedly admits training")
    if public.get("fixed_duration_window_used") is not False:
        raise ValueError("Source public bundle uses fixed-duration windows")
    queue_field, queue_hash_field = QUEUE_FIELDS[queue_kind]
    queue_path = Path(public[queue_field]).resolve()
    queue_sha256 = sha256_file(queue_path)
    if public.get(queue_hash_field) != queue_sha256:
        raise ValueError("Source review queue SHA mismatch")
    rows = read_jsonl(queue_path)
    shards = assign_shards(rows, queue_sha256=queue_sha256, shard_count=shard_count)

    public_root = public_summary_path.parent.resolve()
    for row in rows:
        video = Path(str(row.get("video_path") or "")).resolve()
        if not video.is_file() or video.parent != public_root / "videos":
            raise ValueError(f"Review video escapes the anonymous public root: {video}")
        video_sha256 = row.get("video_sha256")
        if not isinstance(video_sha256, str) or len(video_sha256) != 64:
            raise ValueError(f"Review video lacks a SHA binding: {row.get('sample_id')}")
        if any(row.get(field) is not False for field in ("accepted_for_training",)):
            raise ValueError("Review row unexpectedly admits training")

    output_root.mkdir(parents=True)
    shard_summaries: list[dict[str, Any]] = []
    for index, shard_rows in enumerate(shards):
        shard_root = output_root / f"shard_{index:03d}"
        queue_output = shard_root / "review_queue.jsonl"
        atomic_jsonl(queue_output, shard_rows)
        summary = {
            "artifact_kind": "anonymous_blind_review_queue_shard_v1",
            "schema_version": "1.0.0",
            "queue_kind": queue_kind,
            "shard_index": index,
            "shard_count": shard_count,
            "records": len(shard_rows),
            "review_queue": str(queue_output),
            "review_queue_sha256": sha256_file(queue_output),
            "source_public_summary": str(public_summary_path),
            "source_public_summary_sha256": sha256_file(public_summary_path),
            "source_queue": str(queue_path),
            "source_queue_sha256": queue_sha256,
            "video_policy": "reference_hash_bound_anonymous_source_public_video",
            "fixed_duration_window_used": False,
            "accepted_for_training": False,
        }
        atomic_json(shard_root / "summary.json", summary)
        shard_summaries.append(
            {
                "shard_index": index,
                "records": len(shard_rows),
                "summary": str(shard_root / "summary.json"),
                "summary_sha256": sha256_file(shard_root / "summary.json"),
                "review_queue_sha256": summary["review_queue_sha256"],
            }
        )

    root_summary = {
        "artifact_kind": "anonymous_blind_review_queue_shards_summary_v1",
        "schema_version": "1.0.0",
        "queue_kind": queue_kind,
        "records": len(rows),
        "shard_count": shard_count,
        "minimum_shard_records": min((len(shard) for shard in shards), default=0),
        "maximum_shard_records": max((len(shard) for shard in shards), default=0),
        "coverage_complete_without_overlap": True,
        "source_public_summary": str(public_summary_path),
        "source_public_summary_sha256": sha256_file(public_summary_path),
        "source_queue": str(queue_path),
        "source_queue_sha256": queue_sha256,
        "shards": shard_summaries,
        "fixed_duration_window_used": False,
        "accepted_for_training": False,
    }
    atomic_json(output_root / "summary.json", root_summary)
    return root_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-summary", type=Path, required=True)
    parser.add_argument("--queue-kind", choices=sorted(QUEUE_FIELDS), required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_shards(
        public_summary_path=args.public_summary,
        queue_kind=args.queue_kind,
        shard_count=args.shard_count,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
