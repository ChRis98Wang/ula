#!/usr/bin/env python3
"""Build a native-length 18D catalog from the user-provided 0513 robot CSVs.

Only ``motion_viewer`` trajectories are used. Control and light tables are
duplicate/non-motion representations and are deliberately excluded. Each CSV
remains one complete source-authored motion asset: this builder never crops,
tiles, resamples, or manufactures fixed-duration episodes.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np

from upper_body_skeleton.retarget_v2_18d import (
    CONTRACT_VERSION,
    JOINT_LIMITS_18D,
    JOINT_ORDER_18D,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = PROJECT_ROOT / "0513csv.zip"
DEFAULT_SOURCE_ROOT = Path(
    "/home/gez/nas/cloud/gez/human_motion/raw/ULA0513_native_v1/motion_viewer"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/gez/nas/cloud/gez/human_motion/catalog/"
    "ula0513_native_expression_turn_v1"
)
DEFAULT_PROCESSED_ROOT = Path(
    "/home/gez/nas/cloud/gez/human_motion/processed/ULA0513_native_18d_v1"
)
EXPECTED_ARCHIVE_SHA256 = (
    "0517f05099e95df2a1b360e1c897d11d31ca96b6c8bc22c7c96f4f42e1d442c3"
)
EXPECTED_SOURCE_COUNT = 32
SCHEMA_VERSION = "1.0.0"
ARTIFACT_KIND = "ula0513_native_expression_turn_candidate"
REPRESENTATION = "native_variable_length_robot_expression_turn_v1"
OUTPUT_STEM = "ula0513_native_expression_turn_v1"
FPS = 30.0
FRAME_TIME_TOLERANCE_SEC = 2.0e-5
MAX_JOINT_SPEED_RAD_S = 12.0
MAX_SAFE_PROJECTION_RAD = 0.01
FILENAME = re.compile(r"Robot_Model0530_V2_(?P<label>[A-Za-z0-9]+)\.csv\Z")
SEMANTIC_MASKS = {
    "communicative_intent": False,
    "prompt_text": False,
    "robot_observable_motion_form": False,
    "source_behavior_label": False,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument(
        "--expected-archive-sha256", default=EXPECTED_ARCHIVE_SHA256
    )
    parser.add_argument("--expected-source-count", type=int, default=EXPECTED_SOURCE_COUNT)
    return parser.parse_args(argv)


def stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_sha256(value: dict[str, Any]) -> str:
    return sha256_bytes(stable_json(value).encode("utf-8"))


def atomic_text(path: Path, payload: str) -> str:
    data = payload.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    return sha256_bytes(data)


def atomic_json(path: Path, value: object) -> str:
    return atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    return atomic_text(path, "".join(stable_json(row) + "\n" for row in rows))


def _behavior_slug(label: str) -> str:
    value = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", label)
    value = re.sub(r"(?<=[A-Za-z])(?=[0-9])", "_", value)
    return value.lower()


def _read_native_csv(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        required = ["time_from_start", *JOINT_ORDER_18D]
        missing = [name for name in required if name not in fields]
        if missing:
            raise ValueError(f"missing required columns: {missing}")
        rows = list(reader)
    if len(rows) < 3:
        raise ValueError("native expression turn needs at least three frames")
    try:
        times = np.asarray(
            [float(row["time_from_start"]) for row in rows], dtype=np.float64
        )
        actions = np.asarray(
            [[float(row[name]) for name in JOINT_ORDER_18D] for row in rows],
            dtype=np.float64,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("CSV contains a non-numeric time or 18D joint value") from error
    if not np.isfinite(times).all() or not np.isfinite(actions).all():
        raise ValueError("CSV contains non-finite time or 18D joint values")
    return times, actions, fields


def _write_safe_csv(path: Path, actions: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(JOINT_ORDER_18D)
        writer.writerows(
            [f"{float(value):.10f}" for value in row] for row in actions
        )
    os.replace(temporary, path)
    return sha256_file(path)


def _timing_metrics(times: np.ndarray) -> dict[str, Any]:
    deltas = np.diff(times)
    expected = 1.0 / FPS
    strictly_increasing = bool(np.all(deltas > 0.0))
    max_error = float(np.max(np.abs(deltas - expected)))
    return {
        "starts_at_zero": bool(abs(float(times[0])) <= FRAME_TIME_TOLERANCE_SEC),
        "strictly_increasing": strictly_increasing,
        "frame_time_median_sec": float(np.median(deltas)),
        "frame_time_max_abs_error_sec": max_error,
        "native_30hz": bool(
            strictly_increasing and max_error <= FRAME_TIME_TOLERANCE_SEC
        ),
    }


def _physical_metrics(
    actions: np.ndarray, times: np.ndarray
) -> tuple[dict[str, Any], np.ndarray]:
    lower = np.asarray([JOINT_LIMITS_18D[name][0] for name in JOINT_ORDER_18D])
    upper = np.asarray([JOINT_LIMITS_18D[name][1] for name in JOINT_ORDER_18D])
    below = np.maximum(lower[None, :] - actions, 0.0)
    above = np.maximum(actions - upper[None, :], 0.0)
    violation = np.maximum(below, above)
    deltas = np.diff(times)
    speed = np.abs(np.diff(actions, axis=0) / deltas[:, None])
    per_joint_speed = np.max(speed, axis=0)
    per_joint_violation = np.max(violation, axis=0)
    safe_actions = np.clip(actions, lower[None, :], upper[None, :])
    max_projection = float(np.max(np.abs(safe_actions - actions)))
    return {
        "joint_limits_pass": bool(np.max(violation) <= 1.0e-5),
        "safe_projection_pass": bool(max_projection <= MAX_SAFE_PROJECTION_RAD),
        "joint_speed_pass": bool(np.max(speed) <= MAX_JOINT_SPEED_RAD_S),
        "max_joint_limit_violation_rad": float(np.max(violation)),
        "max_safe_projection_rad": max_projection,
        "safe_projection_threshold_rad": MAX_SAFE_PROJECTION_RAD,
        "max_joint_speed_rad_s": float(np.max(speed)),
        "per_joint_max_limit_violation_rad": {
            name: float(per_joint_violation[index])
            for index, name in enumerate(JOINT_ORDER_18D)
        },
        "per_joint_max_speed_rad_s": {
            name: float(per_joint_speed[index])
            for index, name in enumerate(JOINT_ORDER_18D)
        },
    }, safe_actions


def build_record(
    source_path: Path,
    *,
    source_root: Path,
    processed_root: Path,
    archive_path: Path,
    archive_sha256: str,
) -> dict[str, Any]:
    match = FILENAME.fullmatch(source_path.name)
    if match is None:
        raise ValueError(f"unexpected motion_viewer filename: {source_path.name}")
    behavior_label = match.group("label")
    behavior_slug = _behavior_slug(behavior_label)
    clip_id = f"ula0513_native_v1__{behavior_slug}"
    times, actions, source_fields = _read_native_csv(source_path)
    timing = _timing_metrics(times)
    physical, safe_actions = _physical_metrics(actions, times)
    frame_count = int(actions.shape[0])
    physical_qc_passed = bool(
        timing["starts_at_zero"]
        and timing["native_30hz"]
        and physical["safe_projection_pass"]
        and physical["joint_speed_pass"]
    )
    safe_csv = processed_root / clip_id / "safe.csv"
    safe_sha256 = _write_safe_csv(safe_csv, safe_actions)
    source_sha256 = sha256_file(source_path)
    sample_span = (frame_count - 1) / FPS
    frame_coverage = frame_count / FPS
    record = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "dataset": "ULA0513_user_provided_robot_motion",
        "dataset_revision": archive_sha256,
        "clip_id": clip_id,
        "task_id": clip_id,
        "source_clip_id": source_path.stem,
        "source_group_key": clip_id,
        "speaker_key": clip_id,
        "fixed_split_assignment": "train",
        "eval_eligible": False,
        "split_policy": "user_robot_assets_train_only_not_generalization_evidence",
        "source": {
            "archive_path": str(archive_path),
            "archive_sha256": archive_sha256,
            "archive_member": f"0513csv/motion_viewer/{source_path.name}",
            "csv_path": str(source_path),
            "csv_relpath": source_path.relative_to(source_root).as_posix(),
            "csv_sha256": source_sha256,
            "source_columns": source_fields,
            "selection": "motion_viewer_only_control_and_light_duplicates_excluded",
            "user_provided_training_authorization": True,
        },
        "representation": REPRESENTATION,
        "fps": FPS,
        "training_segment": {
            "representation": REPRESENTATION,
            "start_frame": 0,
            "end_frame_exclusive": frame_count,
            "frame_count": frame_count,
            "fixed_window_sec": None,
            "cropped": False,
            "resampled": False,
            "tiled": False,
            "duration_policy": "one_complete_source_authored_motion_asset",
        },
        "time_axes": {
            "source": {
                "first_sample_sec": float(times[0]),
                "last_sample_sec": float(times[-1]),
                "sample_span_sec": sample_span,
                "frame_coverage_sec": frame_coverage,
            },
            "output": {
                "first_sample_sec": 0.0,
                "last_sample_sec": sample_span,
                "sample_span_sec": sample_span,
                "frame_coverage_sec": frame_coverage,
            },
            "planner_duration_field": "time_axes.output.sample_span_sec",
            "frame_coverage_is_not_planner_target": True,
        },
        "motion_18d": {
            "contract_version": CONTRACT_VERSION,
            "joint_order": list(JOINT_ORDER_18D),
            "safe_csv_path": str(safe_csv),
            "safe_csv_sha256": safe_sha256,
            "frame_count": frame_count,
            "native_head_3dof_present": True,
            "head_mapping_or_synthesis_used": False,
            "safety_projection_applied": bool(
                physical["max_safe_projection_rad"] > 0.0
            ),
            "safety_projection_policy": (
                "clip_only_within_0.01rad_to_conservative_network_joint_limits;_"
                "larger_hardware_range_motion_remains_quarantined"
            ),
        },
        "physical_qc": {
            "passed": physical_qc_passed,
            "timing": timing,
            **physical,
            "visual_mujoco_qc_status": "pending_full_length_blind_video_review",
        },
        "expression_turn": {
            "complete_motion_arc_verified": False,
            "review_status": "pending_full_length_blind_video_review",
            "duration_gate_used": False,
        },
        "source_behavior_label": behavior_label,
        "source_behavior_label_role": (
            "source_authored_asset_name_metadata_pending_robot_observability_review"
        ),
        "canonical_action": None,
        "canonical_prompt": None,
        "semantic_supervision_masks": dict(SEMANTIC_MASKS),
        "emotion_supervision_mask": False,
        "affect_observable_supervision_mask": False,
        "audio_enabled": False,
        "base_motion_eligible": False,
        "semantic_conditioning_eligible": False,
        "expressive_conditioning_eligible": False,
        "accepted_for_training": False,
        "training_admission_status": (
            "pending_complete_arc_action_semantic_and_independent_affect_review"
        ),
    }
    record["record_sha256"] = record_sha256(record)
    return record


def build_catalog(
    archive_path: Path,
    source_root: Path,
    output_dir: Path,
    processed_root: Path,
    *,
    expected_archive_sha256: str,
    expected_source_count: int,
) -> dict[str, Any]:
    archive_path = archive_path.resolve()
    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    processed_root = processed_root.resolve()
    archive_sha256 = sha256_file(archive_path)
    if archive_sha256 != expected_archive_sha256:
        raise ValueError(
            f"0513 archive SHA256 mismatch: {archive_sha256} != {expected_archive_sha256}"
        )
    source_paths = sorted(source_root.glob("*.csv"))
    if len(source_paths) != int(expected_source_count):
        raise ValueError(
            f"expected {expected_source_count} motion_viewer CSVs, found {len(source_paths)}"
        )
    records = [
        build_record(
            source_path,
            source_root=source_root,
            processed_root=processed_root,
            archive_path=archive_path,
            archive_sha256=archive_sha256,
        )
        for source_path in source_paths
    ]
    clip_ids = [str(record["clip_id"]) for record in records]
    if len(clip_ids) != len(set(clip_ids)):
        raise ValueError("0513 behavior-derived clip IDs are not unique")
    manifest_path = output_dir / f"{OUTPUT_STEM}.jsonl"
    manifest_sha256 = atomic_jsonl(manifest_path, records)
    frame_counts = [int(record["training_segment"]["frame_count"]) for record in records]
    physical_counts = Counter(
        "passed" if record["physical_qc"]["passed"] else "failed"
        for record in records
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "ula0513_native_expression_turn_catalog",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive": str(archive_path),
        "archive_sha256": archive_sha256,
        "source_root": str(source_root),
        "processed_root": str(processed_root),
        "record_count": len(records),
        "distinct_behavior_count": len({record["source_behavior_label"] for record in records}),
        "frame_count": sum(frame_counts),
        "frame_count_min": min(frame_counts),
        "frame_count_max": max(frame_counts),
        "distinct_frame_count_count": len(set(frame_counts)),
        "frame_coverage_hours_at_30hz": sum(frame_counts) / FPS / 3600.0,
        "sample_span_hours_at_30hz": (
            sum(max(0, frame_count - 1) for frame_count in frame_counts)
            / FPS
            / 3600.0
        ),
        "physical_qc_counts": dict(sorted(physical_counts.items())),
        "complete_arc_verified_count": 0,
        "semantic_conditioning_eligible_count": 0,
        "expressive_conditioning_eligible_count": 0,
        "accepted_for_training_count": 0,
        "duration_policy": (
            "one_source_authored_motion_asset_per_episode_no_crop_tile_or_resample"
        ),
        "fixed_window_sec": None,
        "legacy_fixed_150_frame_export_used": False,
        "control_or_light_duplicate_tables_used": False,
        "output": {
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha256,
        },
    }
    summary_path = output_dir / f"{OUTPUT_STEM}.summary.json"
    atomic_json(summary_path, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_catalog(
        args.archive,
        args.source_root,
        args.output_dir,
        args.processed_root,
        expected_archive_sha256=str(args.expected_archive_sha256),
        expected_source_count=int(args.expected_source_count),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
