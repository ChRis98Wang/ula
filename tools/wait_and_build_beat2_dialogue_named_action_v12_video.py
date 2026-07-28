#!/usr/bin/env python3
"""Wait for V12 training and GPU capacity, then build the honest 60s video."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.experimental import (  # noqa: E402
    build_beat2_dialogue_named_action_v12_video as builder,
)


STATE_KIND = "beat2_dialogue_named_action_v12_video_waiter_state_v1"
RECEIPT_KIND = "beat2_dialogue_named_action_v12_video_waiter_receipt_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False
    ) as stream:
        stream.write(payload)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def gpu_snapshot(gate: Mapping[str, Any]) -> dict[str, Any]:
    index = int(gate["index"])
    completed = subprocess.run(
        ["nvidia-smi", f"--id={index}", "--query-gpu=memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    try:
        free = float(completed.stdout.strip().splitlines()[0])
    except (IndexError, ValueError):
        free = float("-inf")
    threshold = float(gate["minimum_free_memory_mib"])
    return {
        "ready": completed.returncode == 0 and math.isfinite(free) and free >= threshold,
        "known": completed.returncode == 0,
        "gpu_index": index,
        "free_memory_mib": free,
        "minimum_free_memory_mib": threshold,
        "policy": "wait_for_capacity_never_stop_or_kill_processes",
    }


def latest_progress(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    last = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                last = json.loads(line)
    if not isinstance(last, dict):
        return None
    return {key: last.get(key) for key in ("step", "validation", "train")}


def build_command(config_path: Path) -> list[str]:
    return [
        str(Path(sys.executable).resolve()), "-u", str(Path(builder.__file__).resolve()),
        "--config", str(config_path.resolve()), "--overwrite",
    ]


def wait_and_build(
    config: Mapping[str, Any], *, config_path: Path, poll_seconds: float
) -> dict[str, Any]:
    output_dir = Path(config["_output_dir"])
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        summary = builder._read_json(summary_path)
        if summary.get("status") == "complete":
            return {"status": "already_complete", "summary": str(summary_path)}
    state_path = output_dir / "waiter_state.json"
    receipt_path = output_dir / "waiter_receipt.json"
    consecutive_gpu_passes = 0
    while True:
        completion = builder.training_completion_state(config)
        gpu = None
        if completion.get("ready") is True:
            gpu = gpu_snapshot(config["gpu_wait_gate"])
            consecutive_gpu_passes = consecutive_gpu_passes + 1 if gpu["ready"] else 0
            status = (
                "ready_to_build"
                if consecutive_gpu_passes >= int(config["gpu_wait_gate"]["required_consecutive_passes"])
                else "waiting_for_gpu_capacity"
            )
        else:
            consecutive_gpu_passes = 0
            status = str(completion["status"])
        state = {
            "schema_version": 1,
            "artifact_kind": STATE_KIND,
            "status": status,
            "updated_utc": utc_now(),
            "config": str(config_path),
            "config_sha256": config["_config_sha256"],
            "training_completion": completion,
            "latest_progress": latest_progress(Path(config["_training_progress"])),
            "gpu": gpu,
            "consecutive_gpu_passes": consecutive_gpu_passes,
            "never_stops_or_kills_processes": True,
        }
        _atomic_json(state_path, state)
        if status == "ready_to_build":
            break
        time.sleep(poll_seconds)
    command = build_command(config_path)
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise builder.V12VideoError(
            "video builder failed: " + completed.stderr.strip()[-4000:]
        )
    summary = builder._read_json(summary_path)
    if summary.get("status") != "complete":
        raise builder.V12VideoError("video builder returned without a complete summary")
    receipt = {
        "schema_version": 1,
        "artifact_kind": RECEIPT_KIND,
        "status": "complete",
        "completed_utc": utc_now(),
        "config": str(config_path),
        "builder_command": command,
        "builder_stdout_tail": completed.stdout.strip()[-4000:],
        "summary": str(summary_path),
        "video": summary["renders"]["final"],
        "never_stopped_or_killed_processes": True,
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=builder.DEFAULT_CONFIG)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config_path = args.config.resolve()
        config = builder.read_config(config_path)
        plan = builder.prepare_plan(config)
        if args.dry_run:
            result = plan | {
                "training_completion": builder.training_completion_state(config),
                "builder_command": build_command(config_path),
                "builder_launched": False,
            }
        else:
            lock_path = Path(config["_output_dir"]) / "waiter.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("a+", encoding="utf-8") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise builder.V12VideoError("another V12 video waiter holds the lock") from error
                result = wait_and_build(
                    config, config_path=config_path, poll_seconds=float(args.poll_seconds)
                )
    except (OSError, RuntimeError, ValueError, builder.V12VideoError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
