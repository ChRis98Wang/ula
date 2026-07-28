#!/usr/bin/env python3
"""Build the non-candidate matched-step V8.1 control/treatment audit video."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# MuJoCo chooses its GL backend during import, before render_motion is called.
os.environ.setdefault("MUJOCO_GL", "egl")

from tools.experimental import (  # noqa: E402
    build_hanyang_beat2_emotion_preserving_v81_ab_gt_60s as base,
)


DEFAULT_AUDIT_ROOT = (
    PROJECT_ROOT
    / "training/runs"
    / "hanyang_beat2_emotion_preserving_v81_matched_step1500_replay"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "training/runs"
    / "hanyang_beat2_emotion_preserving_v81_matched_step1500_gt_60s"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=base.DEFAULT_CONFIG)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--step", type=int, default=1500)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.prepare_only and args.validate_only:
        parser.error("--prepare-only and --validate-only are mutually exclusive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validated = base.read_config(args.config)
        if args.validate_only:
            output = base.validate_completed_matched_step_audit(
                validated,
                audit_root=args.audit_root,
                output_dir=args.output_dir,
                step=args.step,
            )
            output["status"] = "complete_validated"
        elif args.prepare_only:
            completion = base.matched_step_audit_completion(
                validated, audit_root=args.audit_root, step=args.step
            )
            prepared = deepcopy(dict(validated))
            prepared["_output_dir"] = str(args.output_dir.expanduser().resolve())
            plan = base.prepare_plan(
                prepared,
                completion=completion,
                evidence_mode="matched_step_audit",
            )
            output = {
                "status": plan["status"],
                "plan": str(
                    args.output_dir.expanduser().resolve() / base.PLAN_FILENAME
                ),
                "completion_sha256": completion["sha256"],
                "paired_metrics": completion["paired_metrics"],
                "gpu_accessed": False,
                "generation_or_render_executed": False,
            }
        else:
            summary = base.build_matched_step_audit_comparison(
                validated,
                audit_root=args.audit_root,
                output_dir=args.output_dir,
                step=args.step,
                overwrite=bool(args.overwrite),
            )
            output = {
                "status": summary["status"],
                "summary": str(
                    args.output_dir.expanduser().resolve()
                    / base.MATCHED_AUDIT_SUMMARY_FILENAME
                ),
                "video": summary["artifacts"]["final_video"]["path"],
                "video_sha256": summary["artifacts"]["final_video"]["sha256"],
                "duration_sec": summary["artifacts"]["final_video"][
                    "duration_sec"
                ],
                "paired_metrics": summary["paired_audit_metrics"],
                "formal_candidate": False,
            }
    except (
        base.V81ComparisonError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
