#!/usr/bin/env python3
"""Batch-retarget a manifest of Motion-X++ clips to the ULA V2 contract."""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from retarget_motionx322_v2 import (
    AXIS_POLICY,
    DEFAULT_MODEL,
    ULA_V2_15D_CONTRACT,
    ULA_V2_18D_CONTRACT,
)


DEFAULT_ROOT = Path("/home/gez/nas/cloud/gez/human_motion")
DEFAULT_MANIFEST = DEFAULT_ROOT / "catalog/expression_candidates_haa500.csv"
RETARGET_SCRIPT = Path(__file__).with_name("retarget_motionx322_v2.py")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--rejected-root", type=Path)
    parser.add_argument("--smplx-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--output-contract",
        choices=(ULA_V2_15D_CONTRACT, ULA_V2_18D_CONTRACT),
        default=ULA_V2_15D_CONTRACT,
    )
    parser.add_argument("--skip-model-sha-check", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--summary-every",
        type=int,
        default=10,
        help="Atomically checkpoint the full batch summary every N completed clips",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-velocity", type=float, default=3.0)
    parser.add_argument("--smoothing-window", type=int, default=7)
    parser.add_argument("--posture-cost", type=float, default=0.02)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_manifest(path, root):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    tasks = []
    seen = set()
    for row in rows:
        clip_id = row.get("clip_id", "").strip()
        motion_relpath = row.get("motion_relpath", "").strip()
        if not clip_id or not motion_relpath:
            raise ValueError("Manifest rows require clip_id and motion_relpath")
        if clip_id in seen:
            raise ValueError(f"Duplicate clip_id in manifest: {clip_id}")
        seen.add(clip_id)
        source = (root / motion_relpath).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        tasks.append(
            {
                "clip_id": clip_id,
                "action": row.get("action", ""),
                "tier": row.get("tier", ""),
                "source": source,
            }
        )
    return tasks


def quality_is_current(output_dir, clip_id, output_contract):
    quality_path = output_dir / "quality.json"
    dimension_label = "18d" if output_contract == ULA_V2_18D_CONTRACT else "15d"
    safe_csv = output_dir / f"{clip_id}_gmr_safe_{dimension_label}.csv"
    if not quality_path.is_file() or not safe_csv.is_file():
        return False
    try:
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        quality.get("axis_policy") == AXIS_POLICY
        and quality.get("output_contract", ULA_V2_15D_CONTRACT) == output_contract
        and quality.get("quality_gate", {}).get("passed", False)
        and quality.get("quality_gate", {}).get("axis_direction_pass", False)
    )


def publish_directory(stage_dir, destination, archive_root):
    if destination.exists():
        archive_root.mkdir(parents=True, exist_ok=True)
        archived = archive_root / destination.name
        suffix = 1
        while archived.exists():
            archived = archive_root / f"{destination.name}_{suffix}"
            suffix += 1
        os.replace(destination, archived)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(stage_dir, destination)


def run_task(task, args, batch_id, staging_root, log_root):
    clip_id = task["clip_id"]
    final_dir = args.output_root / clip_id
    if not args.force and quality_is_current(final_dir, clip_id, args.output_contract):
        quality = json.loads((final_dir / "quality.json").read_text(encoding="utf-8"))
        return {
            **task,
            "source": str(task["source"]),
            "status": "skipped_current_pass",
            "output_dir": str(final_dir),
            "elapsed_sec": 0.0,
            "quality_gate": quality.get("quality_gate", {}),
            "target_error_p95_m": quality.get("limb_target_error_p95_m"),
            "collision_frame_fraction": quality.get("upper_body_collision_frame_rate"),
        }

    stage_dir = staging_root / clip_id
    if stage_dir.exists():
        stale = staging_root / f"{clip_id}.stale_{int(time.time())}"
        os.replace(stage_dir, stale)
    stage_dir.mkdir(parents=True)
    log_path = log_root / f"{clip_id}.log"
    command = [
        sys.executable,
        str(RETARGET_SCRIPT),
        "--motionx",
        str(task["source"]),
        "--output-dir",
        str(stage_dir),
        "--output-contract",
        args.output_contract,
        "--smplx-model",
        str(args.smplx_model),
        "--fps",
        str(args.fps),
        "--max-velocity",
        str(args.max_velocity),
        "--smoothing-window",
        str(args.smoothing_window),
        "--posture-cost",
        str(args.posture_cost),
    ]
    if args.skip_model_sha_check:
        command.append("--skip-model-sha-check")
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        return {
            **task,
            "source": str(task["source"]),
            "status": "process_failed",
            "returncode": completed.returncode,
            "stage_dir": str(stage_dir),
            "log": str(log_path),
            "elapsed_sec": elapsed,
        }

    quality_path = stage_dir / "quality.json"
    try:
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {
            **task,
            "source": str(task["source"]),
            "status": "invalid_quality_report",
            "error": str(error),
            "stage_dir": str(stage_dir),
            "log": str(log_path),
            "elapsed_sec": elapsed,
        }

    gate = quality.get("quality_gate", {})
    passed = bool(
        quality.get("axis_policy") == AXIS_POLICY
        and gate.get("passed", False)
        and gate.get("axis_direction_pass", False)
    )
    if passed:
        archive_root = args.rejected_root / "superseded" / batch_id
        publish_directory(stage_dir, final_dir, archive_root)
        status = "passed"
        output_dir = final_dir
    else:
        rejected_dir = args.rejected_root / batch_id / clip_id
        publish_directory(stage_dir, rejected_dir, args.rejected_root / "superseded" / batch_id)
        status = "quality_failed"
        output_dir = rejected_dir
    return {
        **task,
        "source": str(task["source"]),
        "status": status,
        "output_dir": str(output_dir),
        "log": str(log_path),
        "elapsed_sec": elapsed,
        "quality_gate": gate,
        "target_error_p95_m": quality.get("limb_target_error_p95_m"),
        "collision_frame_fraction": quality.get("upper_body_collision_frame_rate"),
    }


def summary_payload(args, batch_id, tasks, results, started, finished=False):
    counts = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    return {
        "schema_version": 1,
        "batch_id": batch_id,
        "axis_policy": AXIS_POLICY,
        "output_contract": args.output_contract,
        "smplx_model": str(args.smplx_model.resolve()),
        "smplx_model_sha_check": "skipped" if args.skip_model_sha_check else "strict",
        "manifest": str(args.manifest.resolve()),
        "output_root": str(args.output_root.resolve()),
        "rejected_root": str(args.rejected_root.resolve()),
        "workers": args.workers,
        "summary_every": args.summary_every,
        "total_tasks": len(tasks),
        "completed_tasks": len(results),
        "finished": finished,
        "counts": dict(sorted(counts.items())),
        "started_at": started,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "results": sorted(results, key=lambda item: item["clip_id"]),
    }


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.summary_every < 1:
        raise ValueError("summary-every must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    args.root = args.root.resolve()
    args.manifest = args.manifest.resolve()
    args.smplx_model = args.smplx_model.resolve()
    if args.output_root is None:
        relative = (
            "processed/ula_v2_18d_head/v1"
            if args.output_contract == ULA_V2_18D_CONTRACT
            else "processed/ula_v2_15d/v1"
        )
        args.output_root = args.root / relative
    if args.rejected_root is None:
        relative = (
            "processed/rejected/v1/motionx_fixed_v2_18d_head_qc_failed"
            if args.output_contract == ULA_V2_18D_CONTRACT
            else "processed/rejected/v1/motionx_fixed_v2_qc_failed"
        )
        args.rejected_root = args.root / relative
    args.output_root = args.output_root.resolve()
    args.rejected_root = args.rejected_root.resolve()
    tasks = load_manifest(args.manifest, args.root)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    started = datetime.now(timezone.utc).isoformat()
    summary_name = (
        "_batch_18d_head_v1_summary.json"
        if args.output_contract == ULA_V2_18D_CONTRACT
        else "_batch_fixed_v2_summary.json"
    )
    summary_path = args.output_root / summary_name

    if args.dry_run:
        payload = summary_payload(args, batch_id, tasks, [], started, finished=False)
        payload["dry_run"] = True
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    staging_name = (
        ".motionx_18d_head_v1_staging"
        if args.output_contract == ULA_V2_18D_CONTRACT
        else ".motionx_fixed_v2_staging"
    )
    staging_root = args.output_root.parent / staging_name / batch_id
    log_root = args.output_root.parent / "batch_logs" / batch_id
    staging_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    results = []
    atomic_json(summary_path, summary_payload(args, batch_id, tasks, results, started))

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_task, task, args, batch_id, staging_root, log_root): task
            for task in tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
            except Exception as error:  # preserve progress when one worker fails unexpectedly
                result = {
                    **task,
                    "source": str(task["source"]),
                    "status": "worker_failed",
                    "error": repr(error),
                }
            results.append(result)
            if len(results) % args.summary_every == 0 or len(results) == len(tasks):
                atomic_json(
                    summary_path,
                    summary_payload(args, batch_id, tasks, results, started),
                )
            print(
                f"[{len(results):03d}/{len(tasks):03d}] "
                f"{result['status']}: {result['clip_id']}",
                flush=True,
            )

    payload = summary_payload(args, batch_id, tasks, results, started, finished=True)
    atomic_json(summary_path, payload)
    try:
        staging_root.rmdir()
    except OSError:
        # Keep failed workers' partial outputs for diagnosis and manual recovery.
        pass
    print(json.dumps(payload["counts"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
