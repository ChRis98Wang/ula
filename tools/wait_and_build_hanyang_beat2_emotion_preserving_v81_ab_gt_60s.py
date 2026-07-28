#!/usr/bin/env python3
"""CPU waiter that launches the formal V8.1 60-second video after retry4.

Waiting and artifact admission do not initialize CUDA.  The builder child is
launched only after the retry4 supervisor has written its final receipt,
exited successfully, both formal 60k best checkpoints pass deep validation,
and GPU 0 is free for two consecutive observations.  The waiter never stops
or kills another process.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.experimental import (  # noqa: E402
    build_hanyang_beat2_emotion_preserving_v81_ab_gt_60s as builder,
)


STATE_ARTIFACT_KIND = (
    "hanyang_beat2_emotion_preserving_v81_ab_gt_60s_waiter_state_v1"
)
RECEIPT_ARTIFACT_KIND = (
    "hanyang_beat2_emotion_preserving_v81_ab_gt_60s_waiter_receipt_v1"
)


class WaiterError(RuntimeError):
    """Terminal fail-closed waiter error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(value))
    payload.pop("sha256", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        stream.write(payload)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def gpu_snapshot(
    gate: Mapping[str, Any],
    *,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = (
        subprocess.run
    ),
) -> dict[str, Any]:
    index = int(gate["index"])
    memory = command_runner(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    processes = command_runner(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if memory.returncode != 0 or processes.returncode != 0:
        return {
            "ready": False,
            "known": False,
            "gpu_index": index,
            "policy": "wait_never_kill",
        }
    try:
        free = float(memory.stdout.strip().splitlines()[0])
    except (IndexError, ValueError):
        free = float("-inf")
    process_rows = [
        line.strip()
        for line in processes.stdout.splitlines()
        if line.strip() and "no running processes" not in line.casefold()
    ]
    threshold = float(gate["minimum_free_memory_mib"])
    return {
        "ready": (
            math.isfinite(free)
            and free >= threshold
            and not process_rows
        ),
        "known": True,
        "gpu_index": index,
        "free_memory_mib": free,
        "minimum_free_memory_mib": threshold,
        "compute_processes": process_rows,
        "policy": "wait_never_kill",
    }


def build_command(
    config_path: Path,
    *,
    overwrite: bool,
    python_executable: str = sys.executable,
) -> list[str]:
    command = [
        str(Path(python_executable).resolve()),
        "-u",
        str(Path(builder.__file__).resolve()),
        "--config",
        str(config_path.resolve()),
    ]
    if overwrite:
        command.append("--overwrite")
    return command


def wait_and_build(
    validated: Mapping[str, Any],
    *,
    config_path: Path,
    poll_seconds: float,
    timeout_seconds: float,
    overwrite: bool,
    completion_reader: Callable[
        [Mapping[str, Any]], Mapping[str, Any]
    ] = builder.formal_completion_state,
    gpu_reader: Callable[
        [Mapping[str, Any]], Mapping[str, Any]
    ] = gpu_snapshot,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = (
        subprocess.run
    ),
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    if (
        not math.isfinite(poll_seconds)
        or poll_seconds <= 0
        or not math.isfinite(timeout_seconds)
        or timeout_seconds < 0
    ):
        raise WaiterError("poll must be positive and timeout non-negative")
    output_dir = Path(validated["_output_dir"])
    state_path = output_dir / "waiter_state_v1.json"
    receipt_path = output_dir / "waiter_receipt_v1.json"
    completed_summary = output_dir / builder.SUMMARY_FILENAME
    if completed_summary.is_file():
        completed = builder.validate_completed_output(validated)
        return {
            "status": "already_complete",
            "output": completed,
            "builder_launched": False,
        }
    started = clock()
    consecutive_gpu_passes = 0
    last_status = None
    observations: list[dict[str, Any]] = []
    # The output directory is dedicated to this one immutable comparison and
    # a valid completed summary was already handled above.  Always permit the
    # child to replace partial MP4/CSV/NPZ files left by a crash before the
    # atomic summary commit; otherwise Restart=on-failure could deadlock on a
    # final_video-without-summary residue.
    command = build_command(config_path, overwrite=True)
    while True:
        if timeout_seconds > 0 and clock() - started >= timeout_seconds:
            raise WaiterError("timed out before safe video launch")
        completion = dict(completion_reader(validated))
        if completion.get("ready") is not True:
            status = str(completion.get("status") or "waiting_formal")
            consecutive_gpu_passes = 0
            observation = {
                "utc": utc_now(),
                "status": status,
                "current_stage": completion.get("current_stage"),
            }
        else:
            gpu = dict(gpu_reader(validated["gpu_wait_gate"]))
            if gpu.get("ready") is True:
                consecutive_gpu_passes += 1
            else:
                consecutive_gpu_passes = 0
            status = (
                "ready_to_launch"
                if consecutive_gpu_passes
                >= int(
                    validated["gpu_wait_gate"][
                        "required_consecutive_passes"
                    ]
                )
                else "waiting_for_gpu_release"
            )
            observation = {
                "utc": utc_now(),
                "status": status,
                "formal_completion": {
                    "status": completion.get("status"),
                    "supervisor_receipt": completion.get(
                        "supervisor_receipt"
                    ),
                },
                "gpu": gpu,
                "consecutive_gpu_passes": consecutive_gpu_passes,
            }
        if status != last_status:
            observations.append(observation)
            state = {
                "schema_version": 1,
                "artifact_kind": STATE_ARTIFACT_KIND,
                "status": status,
                "updated_utc": utc_now(),
                "config": str(config_path.resolve()),
                "config_sha256": validated["_config_sha256"],
                "builder_command": command,
                "builder_launched": False,
                "observations": observations,
                "never_kills_or_stops_processes": True,
            }
            state["sha256"] = canonical_sha256(state)
            _atomic_json(state_path, state)
            last_status = status
        if status == "ready_to_launch":
            break
        sleeper(poll_seconds)
    result = command_runner(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise WaiterError(
            "formal video builder failed: " + result.stderr.strip()[-2000:]
        )
    completed = builder.validate_completed_output(validated)
    receipt = {
        "schema_version": 1,
        "artifact_kind": RECEIPT_ARTIFACT_KIND,
        "status": "complete",
        "completed_utc": utc_now(),
        "config": str(config_path.resolve()),
        "config_sha256": validated["_config_sha256"],
        "builder_command": command,
        "builder_stdout": result.stdout.strip()[-4000:],
        "gpu_gate": {
            "required_consecutive_passes": validated["gpu_wait_gate"][
                "required_consecutive_passes"
            ],
            "last_observation": observations[-1],
            "policy": "wait_never_kill",
        },
        "output": completed,
        "never_killed_or_stopped_processes": True,
    }
    receipt["sha256"] = canonical_sha256(receipt)
    _atomic_json(receipt_path, receipt)
    return {
        "status": "complete",
        "receipt": str(receipt_path),
        "receipt_file_sha256": builder.abc_video.sha256_file(receipt_path),
        "output": completed,
        "builder_launched": True,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=builder.DEFAULT_CONFIG)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=0.0,
        help="0 waits indefinitely",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate, prepare pending plan, and print command only",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.expanduser().resolve()
    try:
        validated = builder.read_config(config_path)
        plan = builder.prepare_plan(validated)
        completed_summary = (
            Path(validated["_output_dir"]) / builder.SUMMARY_FILENAME
        )
        command = build_command(
            config_path,
            overwrite=bool(args.overwrite or not completed_summary.is_file()),
        )
        if args.dry_run:
            result = {
                "status": plan["status"],
                "builder_command_after_formal_completion": command,
                "gpu_accessed": False,
                "builder_launched": False,
                "safe_to_queue": True,
            }
        else:
            lock_path = (
                Path(validated["_output_dir"]) / "waiter_v1.lock"
            )
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("a+", encoding="utf-8") as lock:
                try:
                    fcntl.flock(
                        lock, fcntl.LOCK_EX | fcntl.LOCK_NB
                    )
                except BlockingIOError:
                    raise WaiterError(
                        "another V8.1 video waiter holds the lock"
                    )
                result = wait_and_build(
                    validated,
                    config_path=config_path,
                    poll_seconds=float(args.poll_seconds),
                    timeout_seconds=float(args.timeout_seconds),
                    overwrite=bool(args.overwrite),
                )
    except (
        WaiterError,
        builder.V81ComparisonError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
