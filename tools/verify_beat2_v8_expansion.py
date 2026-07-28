#!/usr/bin/env python3
"""Verify a BEAT2 v8 expansion and its immutable v7/test-set bindings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_beat2_v8_expansion import (
    DEFAULT_LOCK,
    DEFAULT_LOCKED_TRAIN_READY,
    DEFAULT_OUTPUT_DIR,
    verify_expansion,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--locked-provenance", type=Path, default=DEFAULT_LOCK)
    parser.add_argument(
        "--locked-train-ready", type=Path, default=DEFAULT_LOCKED_TRAIN_READY
    )
    parser.add_argument("--verify-artifacts", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = verify_expansion(
        args.output_dir,
        locked_provenance=args.locked_provenance,
        locked_train_ready=args.locked_train_ready,
        verify_artifacts=args.verify_artifacts,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
