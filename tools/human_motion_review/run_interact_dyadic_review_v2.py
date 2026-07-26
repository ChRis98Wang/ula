#!/usr/bin/env python3
"""Render the four native-duration InterAct dyads with the native BVH parser."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.gmr_v2.interact_bvh_adapter import INTERACT_NATIVE_AXIS_POLICY
from tools.human_motion_review.render_beat2_annotation_review import validate_video


DEFAULT_RECEIPT = Path(
    "/home/gez/nas/cloud/gez/human_motion/catalog/"
    "interact_axis_smoke_four_performance_v1_receipt.json"
)
DEFAULT_OUTPUT = Path(
    "/home/gez/nas/cloud/gez/human_motion/review/"
    "interact_blind_expression_v2/staging"
)
RENDERER = PROJECT_ROOT / "tools/human_motion_review/render_interact_dyadic_review.py"
STATE_NAME = "interact_native_bvh_dyads_v2.run_state.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def validate_result(summary_path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("axis_policy") != INTERACT_NATIVE_AXIS_POLICY:
        raise ValueError("Dyadic render uses the wrong InterAct axis policy")
    if summary.get("source_interval") != expected["source_interval"]:
        raise ValueError("Dyadic render changed the cataloged natural boundary")
    if summary.get("actor_a_bvh_sha256") != expected["actor_a_bvh_sha256"]:
        raise ValueError("Dyadic actor A source SHA mismatch")
    if summary.get("actor_b_bvh_sha256") != expected["actor_b_bvh_sha256"]:
        raise ValueError("Dyadic actor B source SHA mismatch")
    video = Path(summary["output_mp4"])
    check = validate_video(
        video,
        expected_frames=int(summary["frames"]),
        expected_width=1280,
        expected_height=720,
        expected_fps=30.0,
    )
    return {
        "turn_id": expected["turn_id"],
        "status": "rendered_pending_blind_review",
        "source_interval": expected["source_interval"],
        "native_duration_preserved": True,
        "summary_json": str(summary_path.resolve()),
        "summary_json_sha256": sha256_file(summary_path),
        "video": str(video.resolve()),
        "video_sha256": sha256_file(video),
        "video_validation": check,
        "accepted_for_training": False,
    }


def render_one(group: dict[str, Any], output_root: Path, resume: bool) -> dict[str, Any]:
    turn_id = group["turn_id"]
    output_mp4 = output_root / f"{turn_id}_dyad.mp4"
    summary_json = output_root / f"{turn_id}_dyad_summary.json"
    if resume and output_mp4.is_file() and summary_json.is_file():
        return validate_result(summary_json, group)
    subprocess.run(
        [
            sys.executable,
            str(RENDERER),
            "--actor-a-bvh",
            group["actor_a_bvh"],
            "--actor-b-bvh",
            group["actor_b_bvh"],
            "--start-frame",
            str(group["source_interval"][0]),
            "--end-frame",
            str(group["source_interval"][1]),
            "--output-mp4",
            str(output_mp4),
            "--summary-json",
            str(summary_json),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return validate_result(summary_json, group)


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    receipt_path = args.receipt.resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    by_turn: dict[str, list[dict[str, Any]]] = {}
    for row in receipt.get("selected") or []:
        by_turn.setdefault(row["turn_id"], []).append(row)
    if len(by_turn) != 4 or any(len(rows) != 2 for rows in by_turn.values()):
        raise ValueError("InterAct native dyad runner requires four two-actor turns")
    groups = []
    for turn_id, rows in sorted(by_turn.items()):
        rows.sort(key=lambda row: row["target_actor_id"])
        left, right = rows
        left_interval = left["source_interval"]
        right_interval = right["source_interval"]
        interval = [left_interval["start_frame"], left_interval["end_frame_exclusive"]]
        if interval != [right_interval["start_frame"], right_interval["end_frame_exclusive"]]:
            raise ValueError(f"InterAct partners have different boundaries: {turn_id}")
        groups.append(
            {
                "turn_id": turn_id,
                "source_interval": interval,
                "actor_a_bvh": left["source_bvh"],
                "actor_a_bvh_sha256": left["source_bvh_sha256"],
                "actor_b_bvh": right["source_bvh"],
                "actor_b_bvh_sha256": right["source_bvh_sha256"],
            }
        )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / STATE_NAME
    state: dict[str, Any] = {
        "schema_version": "2.0.0",
        "artifact_kind": "interact_native_bvh_dyadic_review_v2_run_state",
        "status": "running",
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "axis_policy": INTERACT_NATIVE_AXIS_POLICY,
        "duration_policy": "cataloged_complete_natural_interaction_boundary_no_fixed_window",
        "renderer_sha256": sha256_file(RENDERER),
        "results": {},
        "accepted_for_training": False,
    }
    atomic_json(state_path, state)
    lock = threading.Lock()
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(render_one, group, output_root, args.resume): group
            for group in groups
        }
        for future in as_completed(futures):
            group = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                failures += 1
                result = {
                    "turn_id": group["turn_id"],
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "accepted_for_training": False,
                }
            with lock:
                state["results"][group["turn_id"]] = result
                atomic_json(state_path, state)
            print(
                f"[{len(state['results']):02d}/{len(groups):02d}] "
                f"{group['turn_id']}: {result['status']}",
                flush=True,
            )
    state["status"] = "failed" if failures else "complete_pending_blind_review"
    state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    state["failure_count"] = failures
    atomic_json(state_path, state)
    print(json.dumps(state, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
