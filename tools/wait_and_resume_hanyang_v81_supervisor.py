#!/usr/bin/env python3
"""Resume the V8.1 supervisor only after its current systemd unit disappears.

This guard deliberately preserves the supervisor's fail-closed behavior.  It
only hands over a state that is still marked ``running`` and whose process lock
is no longer held.  Completed and terminally failed states are never restarted.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence


ACTIVE_STATES = {"active", "activating", "reloading"}
TERMINAL_STATES = {"complete", "failed"}


def _unit_is_active(unit: str) -> bool:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", unit],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() in ACTIVE_STATES


def _read_status(state_path: Path) -> str | None:
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    status = payload.get("status")
    return status if isinstance(status, str) else None


def _lock_is_available(lock_path: Path) -> bool:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True


def wait_and_resume(
    *,
    watched_unit: str,
    state_path: Path,
    poll_seconds: float,
    resume_command: Sequence[str],
) -> int:
    lock_path = Path(str(state_path) + ".lock")
    inactive_observations = 0
    while True:
        if _unit_is_active(watched_unit):
            inactive_observations = 0
            time.sleep(poll_seconds)
            continue

        inactive_observations += 1
        status = _read_status(state_path)
        if status in TERMINAL_STATES:
            print(
                f"guard exiting: supervisor state is terminal ({status})",
                flush=True,
            )
            return 0

        # Require two observations and a released supervisor lock.  This avoids
        # a duplicate launch during a transient unit-state transition.
        if inactive_observations < 2 or not _lock_is_available(lock_path):
            time.sleep(poll_seconds)
            continue
        if status != "running":
            print(
                f"guard waiting: supervisor state is unreadable or unexpected ({status!r})",
                flush=True,
            )
            time.sleep(poll_seconds)
            continue

        print(
            "guard resuming interrupted V8.1 supervisor: "
            + " ".join(resume_command),
            flush=True,
        )
        os.execv(resume_command[0], list(resume_command))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watched-unit", required=True)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("resume_command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if args.resume_command[:1] == ["--"]:
        args.resume_command = args.resume_command[1:]
    if not args.resume_command:
        parser.error("a resume command is required after --")
    executable = Path(args.resume_command[0])
    if not executable.is_absolute() or not executable.is_file():
        parser.error("resume command must start with an absolute executable")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return wait_and_resume(
        watched_unit=args.watched_unit,
        state_path=args.state.resolve(),
        poll_seconds=args.poll_seconds,
        resume_command=args.resume_command,
    )


if __name__ == "__main__":
    sys.exit(main())
