#!/usr/bin/env python3
"""Supervise the clean BEAT2 v7 run and exact-resume only after a failure.

The primary trainer was launched as a transient user service without a restart
policy.  This helper leaves that process untouched while it is active.  If the
service becomes inactive before the trainer writes its terminal summary, the
helper resumes from the trainer-authored ``last.pt`` using the same validated
formal configuration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.train_ula_v2_18d_formal_from_scratch import (
    read_formal_config,
    train_formal,
)


DEFAULT_SERVICE = "ula-v7-clean-adaln-main.service"


def service_state(service: str) -> str:
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            service,
            "--property=ActiveState",
            "--value",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()


def resume_config(config_path: Path, last_checkpoint: Path) -> dict:
    if not last_checkpoint.is_file():
        raise FileNotFoundError(
            f"clean foundation resume checkpoint is missing: {last_checkpoint}"
        )
    config = read_formal_config(config_path)
    config["training"] = dict(config["training"])
    config["training"]["resume_from"] = str(last_checkpoint.resolve())
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--service", default=DEFAULT_SERVICE)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.poll_seconds < 1.0 or args.poll_seconds > 300.0:
        raise ValueError("poll-seconds must be between 1 and 300")
    config_path = args.config.expanduser().resolve()
    base_config = read_formal_config(config_path)
    run_root = Path(str(base_config["output_dir"])).resolve()
    training_dir = run_root / "training"
    summary_path = training_dir / "training_summary.json"
    last_checkpoint = training_dir / "last.pt"

    while not summary_path.is_file():
        state = service_state(args.service)
        if state in {"active", "activating", "reloading"}:
            print(
                json.dumps(
                    {
                        "status": "primary_training_active",
                        "service": args.service,
                        "active_state": state,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            time.sleep(args.poll_seconds)
            continue
        print(
            json.dumps(
                {
                    "status": "primary_training_inactive_exact_resume_starting",
                    "service": args.service,
                    "active_state": state,
                    "last_checkpoint": str(last_checkpoint),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        result = train_formal(resume_config(config_path, last_checkpoint))
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        break

    if not summary_path.is_file():
        raise RuntimeError("clean foundation exited without a terminal training summary")
    print(
        json.dumps(
            {
                "status": "clean_foundation_complete",
                "training_summary": str(summary_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
