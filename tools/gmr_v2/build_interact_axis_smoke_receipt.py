#!/usr/bin/env python3
"""Build a fail-closed InterAct axis/fit/video smoke receipt."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

try:
    from tools.human_motion_collection.build_interact_dyadic_turns import (
        NATURAL_DURATION_POLICY,
        duration_constraint_key_paths,
        record_sha256,
    )
    from tools.human_motion_review.render_beat2_annotation_review import validate_video
except ModuleNotFoundError:  # pragma: no cover - direct invocation by path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.human_motion_collection.build_interact_dyadic_turns import (
        NATURAL_DURATION_POLICY,
        duration_constraint_key_paths,
        record_sha256,
    )
    from tools.human_motion_review.render_beat2_annotation_review import validate_video


DEFAULT_SPEC = Path(__file__).resolve().parents[2] / "configs/interact_axis_smoke_four_performance_v1.json"
DEFAULT_OUTPUT = Path(
    "/home/gez/nas/cloud/gez/human_motion/catalog/"
    "interact_axis_smoke_four_performance_v1_receipt.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> str:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_tasks(path: Path) -> dict[str, dict[str, Any]]:
    tasks = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            task = json.loads(line)
            task_id = str(task["episode_task_id"])
            if task_id in tasks:
                raise ValueError(f"Duplicate task {task_id} at line {line_number}")
            claimed = task.get("episode_task_record_sha256")
            unhashed = {key: value for key, value in task.items() if key != "episode_task_record_sha256"}
            if claimed != record_sha256(unhashed):
                raise ValueError(f"Task record SHA mismatch: {task_id}")
            tasks[task_id] = task
    return tasks


def failed_gate_names(quality: dict[str, Any]) -> list[str]:
    return sorted(
        key
        for key, value in (quality.get("quality_gate") or {}).items()
        if key != "passed" and value is not True
    )


def attempt_row(path: Path) -> dict[str, Any]:
    quality = load_json(path)
    return {
        "attempt_generation": path.parent.name,
        "episode_task_id": quality["episode_task_id"],
        "quality_json": str(path.resolve()),
        "quality_json_sha256": sha256_file(path),
        "axis_policy": quality.get("axis_policy"),
        "elbow_branch_policy": quality.get("elbow_branch_policy"),
        "source_interval": quality.get("source_interval"),
        "automated_quality_passed": quality.get("quality_gate", {}).get("passed") is True,
        "failed_gates": failed_gate_names(quality),
        "limb_target_error_p95_m": quality.get("limb_target_error_p95_m"),
        "upper_body_collision_frame_rate": quality.get("upper_body_collision_frame_rate"),
        "limb_direction_cosine_all_p01": quality.get("limb_direction_cosine_all_p01"),
        "head_gates": {
            key: quality.get(key)
            for key in (
                "head_joint_limits_pass",
                "head_velocity_pass",
                "head_direction_pass",
                "head_continuity_pass",
            )
        },
        "accepted_for_training": False,
    }


def selected_row(
    quality_path: Path,
    task: dict[str, Any],
    review_dir: Path,
) -> dict[str, Any]:
    quality = load_json(quality_path)
    task_id = task["episode_task_id"]
    if quality.get("episode_task_id") != task_id:
        raise ValueError(f"Quality/task ID mismatch: {quality_path}")
    if quality.get("episode_task_record_sha256") != task["episode_task_record_sha256"]:
        raise ValueError(f"Quality/task record SHA mismatch: {task_id}")
    if quality.get("source_sha256") != task["retarget_task"]["source_bvh_sha256"]:
        raise ValueError(f"Source SHA mismatch: {task_id}")
    if quality.get("quality_gate", {}).get("passed") is not True or failed_gate_names(quality):
        raise ValueError(f"Selected task does not pass every automated gate: {task_id}")
    if quality.get("accepted_for_training") is not False:
        raise ValueError(f"Selected smoke must remain blocked from training: {task_id}")

    robot_summary_path = review_dir / f"{task_id}_robot_summary.json"
    axis_summary_path = review_dir / f"{task_id}_axis_review.json"
    robot_summary = load_json(robot_summary_path)
    axis_summary = load_json(axis_summary_path)
    frames = int(robot_summary["frames"])
    robot_mp4 = review_dir / f"{task_id}_robot.mp4"
    comparison_mp4 = review_dir / f"{task_id}_source_vs_robot.mp4"
    robot_check = validate_video(
        robot_mp4,
        expected_frames=frames,
        expected_width=960,
        expected_height=720,
        expected_fps=30.0,
    )
    comparison_check = validate_video(
        comparison_mp4,
        expected_frames=frames,
        expected_width=1600,
        expected_height=720,
        expected_fps=30.0,
    )
    if axis_summary.get("axis_visual_qc_status") != "pending_blind_human_review":
        raise ValueError(f"Unexpected visual status: {task_id}")

    raw_csv = Path(quality["outputs"]["raw_csv"])
    safe_csv = Path(quality["outputs"]["safe_csv"])
    return {
        "episode_task_id": task_id,
        "episode_task_record_sha256": task["episode_task_record_sha256"],
        "performance_id": task["performance_id"],
        "turn_id": task["turn_id"],
        "target_actor_id": task["target_actor_lineage"]["actor_id"],
        "partner_actor_id": task["interaction_partner_lineage"]["actor_id"],
        "source_bvh": quality["source_bvh"],
        "source_bvh_sha256": quality["source_sha256"],
        "source_interval": quality["source_interval"],
        "source_sample_span_sec_role": "diagnostic_only_never_a_cut_or_admission_gate",
        "output_frames": int(quality["frames"]),
        "output_sample_span_sec": quality["output_sample_span_sec"],
        "output_sample_span_sec_role": "diagnostic_only_velocity_safe_retime_result",
        "retimed": bool(quality["retimed"]),
        "automated_quality_gate": quality["quality_gate"],
        "limb_target_error_p95_m": quality["limb_target_error_p95_m"],
        "upper_body_collision_frame_rate": quality["upper_body_collision_frame_rate"],
        "limb_direction_cosine_all_p01": quality["limb_direction_cosine_all_p01"],
        "positive_elbow_branch_values": quality["positive_elbow_branch_values"],
        "negative_elbow_branch_values": quality["negative_elbow_branch_values"],
        "episode_frame_alignment": quality["episode_frame_alignment"],
        "head_gates": {
            key: quality[key]
            for key in (
                "head_joint_limits_pass",
                "head_velocity_pass",
                "head_direction_pass",
                "head_continuity_pass",
            )
        },
        "artifacts": {
            "raw_csv": str(raw_csv.resolve()),
            "raw_csv_sha256": sha256_file(raw_csv),
            "safe_csv": str(safe_csv.resolve()),
            "safe_csv_sha256": sha256_file(safe_csv),
            "quality_json": str(quality_path.resolve()),
            "quality_json_sha256": sha256_file(quality_path),
            "robot_mp4": str(robot_mp4.resolve()),
            "robot_mp4_sha256": sha256_file(robot_mp4),
            "robot_summary_json": str(robot_summary_path.resolve()),
            "robot_summary_json_sha256": sha256_file(robot_summary_path),
            "source_vs_robot_mp4": str(comparison_mp4.resolve()),
            "source_vs_robot_mp4_sha256": sha256_file(comparison_mp4),
            "axis_review_json": str(axis_summary_path.resolve()),
            "axis_review_json_sha256": sha256_file(axis_summary_path),
        },
        "video_validation": {
            "robot": robot_check,
            "source_vs_robot": comparison_check,
        },
        "axis_visual_qc_passed": False,
        "semantic_review_passed": False,
        "emotion_review_passed": False,
        "license_training_use_confirmed": False,
        "accepted_for_training": False,
    }


def main() -> None:
    args = parse_args()
    spec_path = args.spec.resolve()
    spec = load_json(spec_path)
    if spec.get("duration_policy") != NATURAL_DURATION_POLICY:
        raise ValueError("Receipt spec does not use the natural expression duration policy")
    tasks_path = Path(spec["catalog_task_manifest"]).resolve()
    summary_path = Path(spec["catalog_summary"]).resolve()
    tasks = load_tasks(tasks_path)
    review_dir = Path(spec["review_dir"]).resolve()

    selected = []
    for value in spec["selected_quality_reports"]:
        quality_path = Path(value).resolve()
        quality = load_json(quality_path)
        task_id = quality["episode_task_id"]
        if task_id not in tasks:
            raise ValueError(f"Selected task is absent from catalog: {task_id}")
        selected.append(selected_row(quality_path, tasks[task_id], review_dir))

    if len(selected) != 8 or len({row["episode_task_id"] for row in selected}) != 8:
        raise ValueError("Axis smoke requires exactly eight distinct selected actor tasks")
    by_performance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_performance[row["performance_id"]].append(row)
    if len(by_performance) != 4 or any(len(rows) != 2 for rows in by_performance.values()):
        raise ValueError("Axis smoke requires exactly four complete two-actor performances")
    for performance_id, rows in by_performance.items():
        if len({tuple(row["source_interval"].get(key) for key in ("start_frame", "end_frame_exclusive")) for row in rows}) != 1:
            raise ValueError(f"Partner source intervals differ: {performance_id}")
        if {row["target_actor_id"] for row in rows} != {row["partner_actor_id"] for row in rows}:
            raise ValueError(f"Partner identity mismatch: {performance_id}")

    attempts = []
    for directory in map(Path, spec["attempt_dirs"]):
        attempts.extend(attempt_row(path) for path in sorted(directory.glob("*_quality.json")))
    if not attempts:
        raise ValueError("Receipt has no pass/fail attempt history")

    source_spans = [row["source_interval"]["sample_span_sec"] for row in selected]
    report = {
        "schema_version": "1.0.0",
        "artifact_kind": "interact_axis_smoke_four_performance_receipt",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "build_spec": str(spec_path),
        "build_spec_sha256": sha256_file(spec_path),
        "catalog_task_manifest": str(tasks_path),
        "catalog_task_manifest_sha256": sha256_file(tasks_path),
        "catalog_summary": str(summary_path),
        "catalog_summary_sha256": sha256_file(summary_path),
        "duration_policy": NATURAL_DURATION_POLICY,
        "duration_audit": {
            "elapsed_time_is_diagnostic_only": True,
            "fixed_minimum_maximum_or_target_duration_used": False,
            "selected_source_sample_span_sec_diagnostic": {
                "minimum": min(source_spans),
                "maximum": max(source_spans),
                "distinct_values": sorted(set(source_spans)),
            },
        },
        "selected": selected,
        "selected_summary": {
            "performance_count": len(by_performance),
            "actor_task_count": len(selected),
            "all_automated_quality_gates_passed": all(
                row["automated_quality_gate"]["passed"] is True for row in selected
            ),
            "all_videos_fully_decodable_nonblank_silent_h264_yuv420p": all(
                check["passed"] is True
                and check["fully_decodable"] is True
                and check["nonblank"] is True
                and check["audio_streams"] == 0
                and check["codec"] == "h264"
                and check["pixel_format"] == "yuv420p"
                for row in selected
                for check in row["video_validation"].values()
            ),
        },
        "attempt_history": attempts,
        "attempt_summary": {
            "count": len(attempts),
            "automated_passed": sum(row["automated_quality_passed"] for row in attempts),
            "automated_failed": sum(not row["automated_quality_passed"] for row in attempts),
            "generation_distribution": dict(
                sorted(Counter(row["attempt_generation"] for row in attempts).items())
            ),
        },
        "isolated_performances": spec["isolated_performances"],
        "admission_gate": {
            "automated_eight_actor_four_performance_qc_passed": True,
            "video_technical_qc_passed": True,
            "axis_visual_blind_review_passed": False,
            "expression_completeness_blind_review_passed": False,
            "semantic_review_passed": False,
            "emotion_review_passed": False,
            "license_training_use_confirmed": False,
            "passed": False,
        },
        "pilot_scope": "smoke_only_not_representative_of_dataset_pass_rate",
        "accepted_for_retarget_batch": False,
        "accepted_for_training": False,
    }
    forbidden = duration_constraint_key_paths(report)
    if forbidden:
        raise ValueError("Forbidden duration constraint field(s): " + ", ".join(forbidden))
    receipt_sha256 = atomic_json(args.output.resolve(), report)
    print(
        json.dumps(
            {
                "receipt": str(args.output.resolve()),
                "receipt_sha256": receipt_sha256,
                **report["selected_summary"],
                **report["attempt_summary"],
                "accepted_for_training": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
