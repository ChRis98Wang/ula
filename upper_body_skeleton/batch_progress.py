#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def count_outputs(output_root):
    root = Path(output_root)
    return {
        "joint_csv": len(list(root.rglob("*.v2_upper_body_joints.csv"))),
        "skeleton_json": len(list(root.rglob("*.keypoint_upper_body_skeleton_smoothed.json"))),
        "monitor_json": len(list(root.rglob("*.v2_monitor_report.json"))),
        "retarget_report": len(list(root.rglob("*.v2_retarget_report.json"))),
    }


def manifest_summary(manifest_path):
    path = Path(manifest_path)
    if not path.exists():
        return {"exists": False}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    statuses = {}
    for row in rows:
        status = row.get("status", "")
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "exists": True,
        "rows": len(rows),
        "statuses": statuses,
        "errors": [row for row in rows if row.get("status", "").startswith("error")][:20],
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize batch retarget progress")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "output_root": args.output_root,
                "outputs": count_outputs(args.output_root),
                "manifest": manifest_summary(args.manifest),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
