#!/usr/bin/env python3
"""Inspect XEM or build its isolated participant-disjoint research baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from upper_body_skeleton.xem_emotion_research import (  # noqa: E402
    inspect_xem_archive,
    load_config,
    run_xem_research_pipeline,
    validate_official_metadata,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="verify ZIP/MAT structure and print a receipt without writing outputs",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.inspect_only:
        config = load_config(args.config)
        metadata = validate_official_metadata(
            config["metadata_path"],
            expected_sha256=config["expected_metadata_sha256"],
        )
        result = inspect_xem_archive(
            config["archive_path"],
            mat_member=config["mat_member"],
            mat_variable=config["mat_variable"],
        )
        result["official_metadata"] = metadata
        result["research_scope"] = config["research_scope"]
        result["writes_performed"] = False
    else:
        result = run_xem_research_pipeline(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
