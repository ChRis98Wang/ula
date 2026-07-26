#!/usr/bin/env python3
"""Run the versioned native-BVH InterAct axis smoke end to end.

Every selected episode keeps its cataloged natural boundaries.  Automated
retarget checks and rendered evidence are produced, but this runner never
admits an episode to training; an independent blind review remains required.
"""

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


DEFAULT_INPUT = Path(
    "/home/gez/nas/cloud/gez/human_motion/catalog/"
    "interact_axis_smoke_four_performance_v1_receipt.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/gez/nas/cloud/gez/human_motion/processed/"
    "interact_18d_axis_smoke_v16_native_bvh_camera_corrected_review_v2"
)
DEFAULT_REVIEW_ROOT = Path(
    "/home/gez/nas/cloud/gez/human_motion/review/"
    "interact_18d_axis_smoke_v16_native_bvh_camera_corrected_review_v2"
)
RETARGET_SCRIPT = PROJECT_ROOT / "tools/gmr_v2/retarget_interact_bvh_v2.py"
REVIEW_SCRIPT = PROJECT_ROOT / "tools/human_motion_review/render_interact_axis_review.py"
ROBOT_RENDER_MODULE = "upper_body_skeleton.mujoco_playback"
STATE_NAME = "interact_native_bvh_axis_smoke_v2.run_state.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-receipt", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--only-task-id", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reuse-existing-retarget",
        action="store_true",
        help="Reuse hash-validated quality/CSV artifacts in output-root and rerender evidence",
    )
    parser.add_argument(
        "--reuse-source-run-state",
        type=Path,
        help="Completed run state that proves the reused quality/CSV artifact hashes",
    )
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def output_paths(task_id: str, output_root: Path, review_root: Path) -> dict[str, Path]:
    return {
        "quality": output_root / f"{task_id}_quality.json",
        "safe_csv": output_root / f"{task_id}_safe_18d.csv",
        "robot_mp4": review_root / f"{task_id}_robot.mp4",
        "robot_summary": review_root / f"{task_id}_robot_summary.json",
        "comparison_mp4": review_root / f"{task_id}_source_vs_robot.mp4",
        "axis_summary": review_root / f"{task_id}_axis_review.json",
    }


def validate_completed(row: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    quality = load_json(paths["quality"])
    if quality.get("episode_task_id") != row["episode_task_id"]:
        raise ValueError("Completed InterAct task ID mismatch")
    if quality.get("episode_task_record_sha256") != row["episode_task_record_sha256"]:
        raise ValueError("Completed InterAct task record SHA mismatch")
    if quality.get("source_sha256") != row["source_bvh_sha256"]:
        raise ValueError("Completed InterAct source SHA mismatch")
    if quality.get("axis_policy") != INTERACT_NATIVE_AXIS_POLICY:
        raise ValueError("Completed InterAct task uses the wrong axis policy")
    if quality.get("legacy_gmr_euler_component_reorder_used") is not False:
        raise ValueError("Completed InterAct task used the legacy Euler reorder")
    if quality.get("accepted_for_training") is not False:
        raise ValueError("Axis smoke must remain blocked from training")
    axis_summary = load_json(paths["axis_summary"])
    if (
        axis_summary.get("public_frame_labels_anonymous") is not True
        or axis_summary.get("identity_or_partner_metadata_drawn") is not False
    ):
        raise ValueError("Completed InterAct review video exposes identity metadata")
    if (
        axis_summary.get("robot_front_camera_screen_right_axis") != "+Y"
        or axis_summary.get("source_projection")
        != "episode_aligned_dual_view_front_plus_y_z_and_side_plus_x_z"
        or axis_summary.get("robot_side_label_positions")
        != {"screen_left": "ROBOT LEFT", "screen_right": "ROBOT RIGHT"}
    ):
        raise ValueError("Completed InterAct review uses the mirrored front-camera contract")
    frames = int(quality["frames"])
    robot_check = validate_video(
        paths["robot_mp4"],
        expected_frames=frames,
        expected_width=960,
        expected_height=720,
        expected_fps=30.0,
    )
    comparison_check = validate_video(
        paths["comparison_mp4"],
        expected_frames=frames,
        expected_width=1600,
        expected_height=720,
        expected_fps=30.0,
    )
    return {
        "episode_task_id": row["episode_task_id"],
        "status": "rendered_pending_blind_review",
        "source_interval": row["source_interval"],
        "native_duration_preserved": True,
        "automated_quality_passed": quality["quality_gate"]["passed"] is True,
        "failed_automated_gates": sorted(
            key
            for key, value in quality["quality_gate"].items()
            if key != "passed" and value is not True
        ),
        "collision_frame_rate": quality["upper_body_collision_frame_rate"],
        "limb_direction_cosine_all_p01": quality["limb_direction_cosine_all_p01"],
        "limb_target_error_p95_m": quality["limb_target_error_p95_m"],
        "artifacts": {
            key: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for key, path in paths.items()
        },
        "video_validation": {"robot": robot_check, "comparison": comparison_check},
        "accepted_for_training": False,
    }


def run_command(command: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def validate_existing_retarget(
    row: dict[str, Any], paths: dict[str, Path]
) -> None:
    for key in ("quality", "safe_csv"):
        if not paths[key].is_file():
            raise FileNotFoundError(paths[key])
    quality = load_json(paths["quality"])
    if quality.get("episode_task_id") != row["episode_task_id"]:
        raise ValueError("Reusable InterAct task ID mismatch")
    if quality.get("episode_task_record_sha256") != row["episode_task_record_sha256"]:
        raise ValueError("Reusable InterAct task record SHA mismatch")
    if quality.get("source_sha256") != row["source_bvh_sha256"]:
        raise ValueError("Reusable InterAct source SHA mismatch")
    if quality.get("axis_policy") != INTERACT_NATIVE_AXIS_POLICY:
        raise ValueError("Reusable InterAct output uses the wrong axis policy")
    if quality.get("accepted_for_training") is not False:
        raise ValueError("Reusable InterAct output unexpectedly admits training")


def validate_reuse_source_state(
    source_state_path: Path,
    *,
    receipt_hash: str,
    rows: list[dict[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    source_state_path = source_state_path.resolve()
    state = load_json(source_state_path)
    if state.get("artifact_kind") != "interact_native_bvh_axis_smoke_v2_run_state":
        raise ValueError("Reusable source state artifact kind is invalid")
    if state.get("status") != "complete_pending_blind_review":
        raise ValueError("Reusable source state is not complete")
    if state.get("failure_count") != 0:
        raise ValueError("Reusable source state contains failures")
    if state.get("input_receipt_sha256") != receipt_hash:
        raise ValueError("Reusable source state receipt binding differs")
    results = state.get("results") or {}
    for row in rows:
        task_id = row["episode_task_id"]
        result = results.get(task_id)
        if result is None or result.get("status") != "rendered_pending_blind_review":
            raise ValueError(f"Reusable source state lacks completed task: {task_id}")
        expected = output_paths(task_id, output_root, Path("/unused"))
        for key, artifact_key in (("quality", "quality"), ("safe_csv", "safe_csv")):
            artifact = (result.get("artifacts") or {}).get(artifact_key) or {}
            path = Path(str(artifact.get("path") or "")).resolve()
            if path != expected[key].resolve() or not path.is_file():
                raise ValueError(f"Reusable artifact path mismatch: {task_id}/{key}")
            if sha256_file(path) != artifact.get("sha256"):
                raise ValueError(f"Reusable artifact SHA mismatch: {task_id}/{key}")
    return {
        "path": str(source_state_path),
        "sha256": sha256_file(source_state_path),
        "physical_retarget_code_sha256": (state.get("code_sha256") or {}).get(
            "retarget"
        ),
        "task_count": len(rows),
        "all_quality_and_safe_csv_hashes_verified": True,
    }


def process_row(
    row: dict[str, Any],
    output_root: Path,
    review_root: Path,
    resume: bool,
    reuse_existing_retarget: bool,
) -> dict[str, Any]:
    task_id = row["episode_task_id"]
    paths = output_paths(task_id, output_root, review_root)
    if resume and all(path.is_file() for path in paths.values()):
        return validate_completed(row, paths)

    interval = row["source_interval"]
    if reuse_existing_retarget:
        validate_existing_retarget(row, paths)
    else:
        run_command(
            [
                sys.executable,
                str(RETARGET_SCRIPT),
                "--bvh",
                row["source_bvh"],
                "--start-frame",
                str(interval["start_frame"]),
                "--end-frame",
                str(interval["end_frame_exclusive"]),
                "--output-dir",
                str(output_root),
                "--episode-task-id",
                task_id,
                "--episode-task-record-sha256",
                row["episode_task_record_sha256"],
                "--expected-source-sha256",
                row["source_bvh_sha256"],
                "--partner-actor-id",
                row["partner_actor_id"],
            ]
        )
    render_env = os.environ.copy()
    render_env.update({"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"})
    run_command(
        [
            sys.executable,
            "-m",
            ROBOT_RENDER_MODULE,
            "--joint-csv",
            str(paths["safe_csv"]),
            "--output-mp4",
            str(paths["robot_mp4"]),
            "--summary-json",
            str(paths["robot_summary"]),
            "--fps",
            "30",
            "--width",
            "960",
            "--height",
            "720",
            "--camera-margin",
            "1.12",
            "--camera-lookat-z-offset",
            "-0.06",
        ],
        env=render_env,
    )
    run_command(
        [
            sys.executable,
            str(REVIEW_SCRIPT),
            "--source-bvh",
            row["source_bvh"],
            "--source-start-frame",
            str(interval["start_frame"]),
            "--source-end-frame",
            str(interval["end_frame_exclusive"]),
            "--robot-mp4",
            str(paths["robot_mp4"]),
            "--quality-json",
            str(paths["quality"]),
            "--output-mp4",
            str(paths["comparison_mp4"]),
            "--summary-json",
            str(paths["axis_summary"]),
            "--actor-id",
            row["target_actor_id"],
            "--partner-actor-id",
            row["partner_actor_id"],
        ]
    )
    return validate_completed(row, paths)


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    receipt_path = args.input_receipt.resolve()
    receipt = load_json(receipt_path)
    rows = list(receipt.get("selected") or [])
    if args.only_task_id:
        requested = set(args.only_task_id)
        rows = [row for row in rows if row["episode_task_id"] in requested]
        missing = requested.difference(row["episode_task_id"] for row in rows)
        if missing:
            raise ValueError(f"Unknown InterAct task IDs: {sorted(missing)}")
    if not rows:
        raise ValueError("InterAct v2 smoke selection is empty")

    output_root = args.output_root.resolve()
    review_root = args.review_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    review_root.mkdir(parents=True, exist_ok=True)
    state_path = review_root / STATE_NAME
    if args.reuse_existing_retarget and args.reuse_source_run_state is None:
        raise ValueError("Reusing retarget artifacts requires --reuse-source-run-state")
    if not args.reuse_existing_retarget and args.reuse_source_run_state is not None:
        raise ValueError("--reuse-source-run-state requires --reuse-existing-retarget")
    reuse_binding = (
        validate_reuse_source_state(
            args.reuse_source_run_state,
            receipt_hash=sha256_file(receipt_path),
            rows=rows,
            output_root=output_root,
        )
        if args.reuse_existing_retarget
        else None
    )
    lock = threading.Lock()
    state: dict[str, Any] = {
        "schema_version": "2.0.0",
        "artifact_kind": "interact_native_bvh_axis_smoke_v2_run_state",
        "status": "running",
        "input_receipt": str(receipt_path),
        "input_receipt_sha256": sha256_file(receipt_path),
        "axis_policy": INTERACT_NATIVE_AXIS_POLICY,
        "duration_policy": (
            "cataloged_complete_natural_interaction_boundary;"
            "no_fixed_target_minimum_or_maximum_duration"
        ),
        "public_review_video_identity_metadata_exposed": False,
        "front_camera_projection_corrected": True,
        "robot_front_camera_screen_right_axis": "+Y",
        "reused_existing_retarget_artifacts": args.reuse_existing_retarget,
        "reused_retarget_source_run_state": reuse_binding,
        "task_count": len(rows),
        "workers": args.workers,
        "code_sha256": {
            "retarget": sha256_file(RETARGET_SCRIPT),
            "review": sha256_file(REVIEW_SCRIPT),
            "runner": sha256_file(Path(__file__)),
        },
        "results": {},
        "accepted_for_training": False,
    }
    atomic_json(state_path, state)

    def persist(task_id: str, result: dict[str, Any]) -> None:
        with lock:
            state["results"][task_id] = result
            atomic_json(state_path, state)

    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_row,
                row,
                output_root,
                review_root,
                args.resume,
                args.reuse_existing_retarget,
            ): row
            for row in rows
        }
        for future in as_completed(futures):
            row = futures[future]
            task_id = row["episode_task_id"]
            try:
                result = future.result()
            except Exception as exc:
                failures += 1
                result = {
                    "episode_task_id": task_id,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "accepted_for_training": False,
                }
            persist(task_id, result)
            print(f"[{len(state['results']):04d}/{len(rows):04d}] {task_id}: {result['status']}", flush=True)

    state["status"] = "failed" if failures else "complete_pending_blind_review"
    state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    state["failure_count"] = failures
    state["rendered_count"] = sum(
        result.get("status") == "rendered_pending_blind_review"
        for result in state["results"].values()
    )
    atomic_json(state_path, state)
    print(json.dumps(state, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
