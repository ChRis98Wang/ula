#!/usr/bin/env python3
"""Atomically terminalize one verified staging result from a legacy BEAT2 batch."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


REQUIRED_QUALITY_GATES = (
    "joint_limits_pass",
    "velocity_pass",
    "target_fit_pass",
    "collision_pass",
    "axis_direction_pass",
    "head_joint_limits_pass",
    "head_velocity_pass",
    "head_direction_pass",
    "head_continuity_pass",
    "passed",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_from_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            records.append(value)
    return records


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    os.replace(temporary, path)


def require_hash(path: Path, expected: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"SHA256 mismatch for {path}: expected {expected}, got {actual}")
    return actual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--expected-status-sha256", required=True)
    parser.add_argument("--expected-quality-sha256", required=True)
    parser.add_argument("--expected-raw-sha256", required=True)
    parser.add_argument("--expected-safe-sha256", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-existing-results", type=int, required=True)
    parser.add_argument("--expected-final-passed", type=int, required=True)
    parser.add_argument("--expected-final-failed", type=int, required=True)
    parser.add_argument("--commit", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    inventory = args.inventory.resolve()
    status_path = output_root / "status.json"
    require_hash(status_path, args.expected_status_sha256)
    status = read_json(status_path)
    if status.get("run_state") != "interrupted_resumable":
        raise RuntimeError("legacy status must be interrupted_resumable")
    if status.get("run_id") != args.run_id:
        raise RuntimeError("run_id does not match legacy status")
    if status.get("inventory_sha256") != sha256(inventory):
        raise RuntimeError("inventory hash does not match legacy status")
    if int(status.get("pending_count", -1)) != 1:
        raise RuntimeError("legacy run must contain exactly one pending result")
    if int(status.get("saved_result_count", -1)) != args.expected_existing_results:
        raise RuntimeError("unexpected existing result count")

    pending = read_jsonl(output_root / "pending_manifest.jsonl")
    if len(pending) != 1 or pending[0].get("task_id") != args.task_id:
        raise RuntimeError("pending manifest does not contain only the requested task")
    task = dict(pending[0])
    task.pop("status", None)
    task.pop("pending_reason", None)

    inventory_records = read_jsonl(inventory)
    inventory_ids = {
        str(record.get("task_id") or record.get("clip_id")) for record in inventory_records
    }
    if args.task_id not in inventory_ids:
        raise RuntimeError("task is not present in the bound inventory")

    stage = output_root / "staging" / args.run_id / args.task_id
    destination = output_root / "passed" / args.task_id
    failed_destination = output_root / "failed" / args.task_id
    result_path = output_root / "state" / "results" / f"{args.task_id}.json"
    if not stage.is_dir():
        raise RuntimeError(f"missing staging directory: {stage}")
    if destination.exists() or failed_destination.exists() or result_path.exists():
        raise RuntimeError("terminal destination or result already exists")

    quality_path = stage / "quality.json"
    raw_files = sorted(stage.glob("*_gmr_raw_18d.csv"))
    safe_files = sorted(stage.glob("*_gmr_safe_18d.csv"))
    if len(raw_files) != 1 or len(safe_files) != 1:
        raise RuntimeError("staging must contain exactly one raw and one safe 18D CSV")
    raw_path, safe_path = raw_files[0], safe_files[0]
    preflight_hashes = {
        "quality": require_hash(quality_path, args.expected_quality_sha256),
        "raw_csv": require_hash(raw_path, args.expected_raw_sha256),
        "safe_csv": require_hash(safe_path, args.expected_safe_sha256),
    }
    quality = read_json(quality_path)
    gate = quality.get("quality_gate") or {}
    failed_gates = [name for name in REQUIRED_QUALITY_GATES if gate.get(name) is not True]
    if failed_gates:
        raise RuntimeError(f"strict quality gates failed: {failed_gates}")
    required_values = {
        "source_sha256": args.expected_source_sha256,
        "source_window_start_frame": int(task["start_frame"]),
        "source_window_end_frame_exclusive": int(task["end_frame_exclusive"]),
        "output_contract": "ula_v2_18d_head_v1",
        "action_dim": 18,
    }
    for key, expected in required_values.items():
        if quality.get(key) != expected:
            raise RuntimeError(
                f"quality field {key!r} mismatch: expected {expected!r}, got {quality.get(key)!r}"
            )
    if len(quality.get("joint_order") or []) != 18:
        raise RuntimeError("quality report does not contain an 18D joint order")
    safe_line_count = sum(1 for _ in safe_path.open(encoding="utf-8"))
    if safe_line_count != int(quality["frames"]) + 1:
        raise RuntimeError("safe CSV row count does not match quality frames")

    log_path = output_root / "logs" / args.task_id / f"{args.run_id}.log"
    if not log_path.is_file():
        raise RuntimeError(f"missing task log: {log_path}")
    audit_path = output_root / f"legacy_finalization_{args.task_id}.json"
    audit = {
        "schema_version": "legacy_beat2_terminal_recovery_v1",
        "task_id": args.task_id,
        "run_id": args.run_id,
        "inventory": str(inventory),
        "inventory_sha256": sha256(inventory),
        "preflight_status_sha256": args.expected_status_sha256,
        "preflight_hashes": preflight_hashes,
        "source_sha256": args.expected_source_sha256,
        "quality_gate": dict(gate),
        "source_stage": str(stage),
        "destination": str(destination),
        "recovery_reason": "legacy_per_future_manifest_rewrite_backlog_after_complete_worker_output",
        "state": "validated_dry_run" if not args.commit else "prepared",
        "validated_at": utc_now(),
    }
    if not args.commit:
        print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    atomic_json(audit_path, audit)
    quality["outputs"] = {
        "raw_csv": str((destination / raw_path.name).resolve()),
        "safe_csv": str((destination / safe_path.name).resolve()),
    }
    atomic_json(quality_path, quality)
    os.replace(stage, destination)
    published_quality = destination / "quality.json"
    published_safe = destination / safe_path.name
    finished_at = utc_from_mtime(log_path)
    started_at = utc_from_mtime(log_path.parent)
    result = {
        **task,
        "run_id": args.run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_sec": max(
            0.0,
            (log_path.stat().st_mtime - log_path.parent.stat().st_mtime),
        ),
        "log_path": str(log_path.resolve()),
        "source_sha256": args.expected_source_sha256,
        "returncode": 0,
        "status": "passed",
        "output_dir": str(destination.resolve()),
        "quality_json": str(published_quality.resolve()),
        "quality_json_sha256": sha256(published_quality),
        "safe_csv": str(published_safe.resolve()),
        "safe_csv_sha256": sha256(published_safe),
        "quality_gate": dict(gate),
        "frames": int(quality["frames"]),
        "duration_sec": float(quality["duration_sec"]),
        "legacy_recovery": {
            "audit_path": str(audit_path.resolve()),
            "preflight_quality_sha256": args.expected_quality_sha256,
            "preflight_raw_sha256": args.expected_raw_sha256,
            "preflight_safe_sha256": args.expected_safe_sha256,
            "terminalized_at": utc_now(),
        },
    }
    atomic_json(result_path, result)

    state_records = [
        read_json(path)
        for path in sorted((output_root / "state" / "results").glob("*.json"))
    ]
    if len(state_records) != args.expected_existing_results + 1:
        raise RuntimeError("terminal result count did not increase by exactly one")
    state_ids = {str(record.get("task_id")) for record in state_records}
    if state_ids != inventory_ids:
        missing = sorted(inventory_ids - state_ids)[:10]
        extra = sorted(state_ids - inventory_ids)[:10]
        raise RuntimeError(f"terminal result coverage mismatch: missing={missing}, extra={extra}")
    counts = Counter(record.get("status") for record in state_records)
    if counts["passed"] != args.expected_final_passed:
        raise RuntimeError("final passed count mismatch")
    non_passed = len(state_records) - counts["passed"]
    if non_passed != args.expected_final_failed:
        raise RuntimeError("final non-passed count mismatch")

    ordered = sorted(state_records, key=lambda record: record["task_id"])
    atomic_jsonl(
        output_root / "passed_manifest.jsonl",
        [record for record in ordered if record.get("status") == "passed"],
    )
    atomic_jsonl(
        output_root / "failed_manifest.jsonl",
        [record for record in ordered if record.get("status") != "passed"],
    )
    atomic_jsonl(output_root / "pending_manifest.jsonl", [])

    audit.update(
        {
            "state": "committed",
            "committed_at": utc_now(),
            "published_quality_sha256": result["quality_json_sha256"],
            "published_safe_sha256": result["safe_csv_sha256"],
            "result_path": str(result_path.resolve()),
            "result_sha256": sha256(result_path),
            "final_counts": {
                "passed": counts["passed"],
                "non_passed": non_passed,
                "total": len(state_records),
            },
        }
    )
    atomic_json(audit_path, audit)
    status.update(
        {
            "run_state": "finished",
            "updated_at": utc_now(),
            "finished_at": utc_now(),
            "saved_result_count": len(state_records),
            "pending_count": 0,
            "coverage_complete": True,
            "counts": dict(sorted(counts.items())),
            "legacy_finalization_audit": str(audit_path.resolve()),
            "legacy_finalization_audit_sha256": sha256(audit_path),
        }
    )
    atomic_json(status_path, status)
    print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
