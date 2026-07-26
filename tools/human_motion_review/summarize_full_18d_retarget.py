#!/usr/bin/env python3
"""Summarize and strictly validate a full ULA V2 18D retarget batch."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .adjudicate_training_dataset import (
        EXPECTED_18D_JOINT_ORDER,
        REQUIRED_18D_GATES,
        sha256_file,
    )
except ImportError:
    from adjudicate_training_dataset import (  # type: ignore
        EXPECTED_18D_JOINT_ORDER,
        REQUIRED_18D_GATES,
        sha256_file,
    )


REPORT_SCHEMA_VERSION = "1.1.0"
HEAD_JOINTS = ["head_roll_joint", "head_pitch_joint", "head_yaw_joint"]
PASSED_MANIFEST = "full_retarget_passed.jsonl"
REJECTED_MANIFEST = "full_retarget_rejected.jsonl"
REPORT_NAME = "full_retarget_scale_report.json"
DEFAULT_SMPLX_MODEL = Path(
    "/home/gez/shuaiwang/.cache/ula_smplx/SMPLX_NEUTRAL_2020.npz"
)
EXPECTED_SMPLX_SHA256 = (
    "bdf06146e27d92022fe5dadad3b9203373f6879eca8e4d8235359ee3ec6a5a74"
)
REQUIRED_INVENTORY_COLUMNS = {
    "clip_id",
    "action",
    "canonical_prompt_en",
    "robot_contract",
    "motion_relpath",
    "frame_count",
    "nominal_fps",
    "motion_sha256",
    "review_state",
    "manual_review_required",
    "accepted_for_training",
}
BATCH_ID_PATTERN = re.compile(r"^\d{8}_\d{6}$")


def _parse_int(value: str, field: str, clip_id: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{clip_id}: invalid {field} {value!r}") from error
    if parsed <= 0:
        raise ValueError(f"{clip_id}: {field} must be positive")
    return parsed


def _parse_float(value: str, field: str, clip_id: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{clip_id}: invalid {field} {value!r}") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{clip_id}: {field} must be finite and positive")
    return parsed


def _parse_bool(value: str, field: str, clip_id: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{clip_id}: invalid {field} {value!r}")


def load_inventory(
    path: Path,
    *,
    expected_count: int | None,
    expected_contract: str,
) -> list[dict]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            missing_columns = REQUIRED_INVENTORY_COLUMNS - columns
            if missing_columns:
                raise ValueError(f"inventory is missing columns: {sorted(missing_columns)}")
            raw_rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError(f"cannot read inventory {path}: {error}") from error

    rows = []
    seen = set()
    for raw in raw_rows:
        clip_id = (raw.get("clip_id") or "").strip()
        action = (raw.get("action") or "").strip()
        motion_relpath = (raw.get("motion_relpath") or "").strip()
        if not clip_id or not action or not motion_relpath:
            raise ValueError("inventory rows require clip_id, action, and motion_relpath")
        if clip_id in seen:
            raise ValueError(f"duplicate inventory clip_id: {clip_id}")
        seen.add(clip_id)
        if raw.get("robot_contract") != expected_contract:
            raise ValueError(
                f"{clip_id}: inventory contract {raw.get('robot_contract')!r} is not {expected_contract!r}"
            )
        if Path(motion_relpath).stem != clip_id:
            raise ValueError(f"{clip_id}: motion_relpath does not identify the same clip")
        action_associated = clip_id == action or clip_id.startswith(f"{action}_")
        rows.append(
            {
                **raw,
                "clip_id": clip_id,
                "action": action,
                "motion_relpath": motion_relpath,
                "frame_count": _parse_int(raw.get("frame_count"), "frame_count", clip_id),
                "nominal_fps": _parse_float(raw.get("nominal_fps"), "nominal_fps", clip_id),
                "manual_review_required": _parse_bool(
                    raw.get("manual_review_required"), "manual_review_required", clip_id
                ),
                "accepted_for_training": _parse_bool(
                    raw.get("accepted_for_training"), "accepted_for_training", clip_id
                ),
                "action_associated": action_associated,
            }
        )
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} inventory rows, found {len(rows)}")
    return sorted(rows, key=lambda item: item["clip_id"])


def _batch_id_for_path(path: Path) -> str | None:
    for parent in path.parents:
        if BATCH_ID_PATTERN.fullmatch(parent.name):
            return parent.name
    return None


def discover_quality_candidates(
    passed_root: Path,
    rejected_root: Path,
    inventory_ids: set[str],
    *,
    expected_contract: str = "ula_v2_18d_head_v1",
) -> tuple[dict[str, list[dict]], dict]:
    by_clip = defaultdict(list)
    orphan_paths = []
    invalid_json_paths = []
    wrong_contract_paths = []
    seen_paths = set()
    sources = [
        ("passed_root", sorted(passed_root.glob("*/quality.json"))),
        ("rejected_root", sorted(rejected_root.rglob("quality.json"))),
    ]
    for partition, paths in sources:
        for path in paths:
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            clip_id = path.parent.name
            if clip_id not in inventory_ids:
                orphan_paths.append(str(resolved))
                continue
            try:
                quality = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                quality = None
                invalid_json_paths.append(str(resolved))
            if isinstance(quality, dict) and quality.get("output_contract") != expected_contract:
                wrong_contract_paths.append(str(resolved))
            by_clip[clip_id].append(
                {
                    "path": resolved,
                    "sample_dir": resolved.parent,
                    "partition": partition,
                    "batch_id": _batch_id_for_path(resolved),
                    "mtime_ns": resolved.stat().st_mtime_ns,
                    "quality": quality,
                }
            )
    diagnostics = {
        "discovered_quality_files": len(seen_paths),
        "orphan_quality_paths": sorted(orphan_paths),
        "invalid_json_paths": sorted(invalid_json_paths),
        "wrong_contract_paths": sorted(wrong_contract_paths),
    }
    return by_clip, diagnostics


def select_latest_matching_candidate(
    candidates: list[dict],
    *,
    expected_contract: str,
) -> tuple[dict | None, dict]:
    matching = [
        candidate
        for candidate in candidates
        if isinstance(candidate["quality"], dict)
        and candidate["quality"].get("output_contract") == expected_contract
    ]
    if not matching:
        return None, {
            "candidate_count": len(candidates),
            "matching_candidate_count": 0,
            "superseded_quality_paths": [],
        }
    newest_mtime = max(candidate["mtime_ns"] for candidate in matching)
    newest = [candidate for candidate in matching if candidate["mtime_ns"] == newest_mtime]
    if len(newest) != 1:
        paths = sorted(str(candidate["path"]) for candidate in newest)
        raise ValueError(f"ambiguous latest quality evidence with identical mtime: {paths}")
    selected = newest[0]
    superseded = sorted(str(candidate["path"]) for candidate in matching if candidate is not selected)
    return selected, {
        "candidate_count": len(candidates),
        "matching_candidate_count": len(matching),
        "selected_mtime_ns": selected["mtime_ns"],
        "superseded_quality_paths": superseded,
    }


def _inspect_safe_csv(path: Path, *, expected_rows: int | None, fps: float | None) -> dict:
    result = {
        "path": str(path),
        "sha256": None,
        "bytes": None,
        "rows": None,
        "columns": None,
        "reasons": [],
        "head_metrics": None,
    }
    if not path.is_file():
        result["reasons"] = ["safe_csv_missing"]
        return result
    result["sha256"] = sha256_file(path)
    result["bytes"] = path.stat().st_size
    reasons = []
    row_count = 0
    row_width_mismatch = False
    non_numeric = False
    non_finite = False
    head_min = [math.inf, math.inf, math.inf]
    head_max = [-math.inf, -math.inf, -math.inf]
    previous_head = None
    max_head_velocity = 0.0
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            result["columns"] = len(header) if header is not None else 0
            if header != EXPECTED_18D_JOINT_ORDER:
                reasons.append("safe_csv_header_mismatch")
            for row in reader:
                row_count += 1
                if len(row) != len(EXPECTED_18D_JOINT_ORDER):
                    row_width_mismatch = True
                    continue
                numeric_row = []
                row_valid = True
                for value in row:
                    try:
                        numeric = float(value)
                    except ValueError:
                        non_numeric = True
                        row_valid = False
                        break
                    if not math.isfinite(numeric):
                        non_finite = True
                        row_valid = False
                        break
                    numeric_row.append(numeric)
                if not row_valid:
                    continue
                head = numeric_row[-3:]
                for index, value in enumerate(head):
                    head_min[index] = min(head_min[index], value)
                    head_max[index] = max(head_max[index], value)
                if previous_head is not None and fps is not None:
                    max_head_velocity = max(
                        max_head_velocity,
                        *(abs(value - previous) * fps for value, previous in zip(head, previous_head)),
                    )
                previous_head = head
    except (OSError, UnicodeError, csv.Error):
        result["reasons"] = ["safe_csv_unreadable"]
        return result
    result["rows"] = row_count
    if row_width_mismatch:
        reasons.append("safe_csv_row_width_mismatch")
    if non_numeric:
        reasons.append("safe_csv_non_numeric")
    if non_finite:
        reasons.append("safe_csv_non_finite")
    if expected_rows is not None and row_count != expected_rows:
        reasons.append("safe_csv_row_count_mismatch")
    if previous_head is None:
        reasons.append("safe_csv_no_valid_motion_rows")
    else:
        result["head_metrics"] = {
            joint: {
                "min_rad": head_min[index],
                "max_rad": head_max[index],
                "range_rad": head_max[index] - head_min[index],
                "range_deg": math.degrees(head_max[index] - head_min[index]),
                "max_abs_rad": max(abs(head_min[index]), abs(head_max[index])),
            }
            for index, joint in enumerate(HEAD_JOINTS)
        }
        result["head_metrics"]["max_component_velocity_rad_s"] = max_head_velocity
    result["reasons"] = sorted(set(reasons))
    return result


def _quality_number(quality: dict, key: str, reasons: list[str], *, positive=False):
    value = quality.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        reasons.append(f"quality_field_invalid:{key}")
        return None
    if positive and value <= 0:
        reasons.append(f"quality_field_invalid:{key}")
        return None
    return value


def _compare_head_metrics(quality: dict, csv_metrics: dict | None, reasons: list[str]) -> None:
    if quality.get("head_joint_order") != HEAD_JOINTS:
        reasons.append("head_joint_order_mismatch")
    if csv_metrics is None:
        return
    quality_ranges = quality.get("joint_ranges") or {}
    for joint in HEAD_JOINTS:
        quality_range = quality_ranges.get(joint)
        if not isinstance(quality_range, dict):
            reasons.append(f"head_range_missing:{joint}")
            continue
        for key in ("min_rad", "max_rad"):
            value = quality_range.get(key)
            expected = csv_metrics[joint][key]
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                reasons.append(f"head_range_invalid:{joint}:{key}")
            elif not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-6):
                reasons.append(f"head_range_csv_mismatch:{joint}:{key}")
    quality_velocity = quality.get("head_safe_max_velocity_rad_s")
    csv_velocity = csv_metrics["max_component_velocity_rad_s"]
    if not isinstance(quality_velocity, (int, float)) or not math.isfinite(quality_velocity):
        reasons.append("head_velocity_metric_invalid")
    elif not math.isclose(quality_velocity, csv_velocity, rel_tol=0.0, abs_tol=1e-5):
        reasons.append("head_velocity_csv_mismatch")


def inspect_selected_candidate(
    inventory: dict,
    candidate: dict,
    selection: dict,
    *,
    dataset_root: Path,
    expected_contract: str,
) -> dict:
    clip_id = inventory["clip_id"]
    quality = candidate["quality"]
    reasons = []
    frames = _quality_number(quality, "frames", reasons, positive=True)
    fps = _quality_number(quality, "fps", reasons, positive=True)
    source_frames = _quality_number(quality, "source_frames", reasons, positive=True)
    source_fps = _quality_number(quality, "source_fps", reasons, positive=True)

    if quality.get("output_contract") != expected_contract:
        reasons.append("output_contract_mismatch")
    if quality.get("action_dim") != 18:
        reasons.append("action_dim_mismatch")
    if quality.get("joint_order") != EXPECTED_18D_JOINT_ORDER:
        reasons.append("joint_order_mismatch")
    if not inventory["action_associated"]:
        reasons.append("inventory_action_clip_mismatch")

    expected_source = (dataset_root / inventory["motion_relpath"]).resolve()
    reported_source = quality.get("source_motionx")
    if not reported_source or Path(reported_source).resolve() != expected_source:
        reasons.append("source_path_mismatch")
    if not expected_source.is_file():
        reasons.append("source_file_missing")
    if quality.get("source_sha256") != inventory.get("motion_sha256"):
        reasons.append("source_sha256_mismatch")
    if source_frames != inventory["frame_count"]:
        reasons.append("source_frames_mismatch")
    if source_fps is not None and not math.isclose(
        source_fps, inventory["nominal_fps"], rel_tol=0.0, abs_tol=1e-9
    ):
        reasons.append("source_fps_mismatch")

    gates = quality.get("quality_gate")
    if not isinstance(gates, dict):
        reasons.append("quality_gate_missing")
        gates = {}
    for gate in REQUIRED_18D_GATES:
        if gate not in gates:
            reasons.append(f"gate_missing:{gate}")
        elif gates[gate] is not True:
            reasons.append(f"gate_failed:{gate}")
    for gate, value in gates.items():
        if isinstance(value, bool) and value is False:
            reasons.append(f"gate_failed:{gate}")

    expected_rows = int(frames) if isinstance(frames, (int, float)) else None
    safe_csv_path = candidate["sample_dir"] / f"{clip_id}_gmr_safe_18d.csv"
    csv_report = _inspect_safe_csv(safe_csv_path, expected_rows=expected_rows, fps=fps)
    reasons.extend(csv_report["reasons"])
    _compare_head_metrics(quality, csv_report["head_metrics"], reasons)

    duration = frames / fps if frames is not None and fps is not None else None
    reported_duration = quality.get("duration_sec")
    if duration is not None and (
        not isinstance(reported_duration, (int, float))
        or not math.isclose(reported_duration, duration, rel_tol=0.0, abs_tol=1e-9)
    ):
        reasons.append("output_duration_mismatch")
    if candidate["partition"] == "passed_root" and gates.get("passed") is not True:
        reasons.append("passed_partition_contains_failed_gate")
    if candidate["partition"] == "rejected_root" and not reasons:
        reasons.append("rejected_partition_contains_passing_quality")
    reasons = sorted(set(reasons))
    retarget_status = "retarget_qc_passed" if not reasons else "retarget_qc_rejected"

    return {
        "schema_version": 1,
        "clip_id": clip_id,
        "action": inventory["action"],
        "canonical_prompt_en": inventory.get("canonical_prompt_en"),
        "retarget_status": retarget_status,
        "training_admission": {
            "accepted_for_training": False,
            "state": "pending_semantic_review",
            "inventory_review_state": inventory.get("review_state"),
            "manual_review_required": inventory["manual_review_required"],
            "rule": "retarget QC never grants semantic training admission",
        },
        "source": {
            "path": str(expected_source),
            "relpath": inventory["motion_relpath"],
            "sha256": inventory.get("motion_sha256"),
            "frames": inventory["frame_count"],
            "fps": inventory["nominal_fps"],
            "duration_sec": inventory["frame_count"] / inventory["nominal_fps"],
        },
        "output_18d": {
            "quality_json": str(candidate["path"]),
            "quality_sha256": sha256_file(candidate["path"]),
            "safe_csv": csv_report,
            "frames": frames,
            "fps": fps,
            "duration_sec": duration,
            "retime_factor": quality.get("retime_factor"),
            "output_contract": quality.get("output_contract"),
            "action_dim": quality.get("action_dim"),
            "head_metrics": csv_report["head_metrics"],
        },
        "quality": {
            "passed": retarget_status == "retarget_qc_passed",
            "reasons": reasons,
            "quality_gate": quality.get("quality_gate"),
            "partition": candidate["partition"],
            "batch_id": candidate["batch_id"],
            "selection": selection,
        },
    }


def _percentile(sorted_values: list[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def distribution(values) -> dict:
    sorted_values = sorted(
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    )
    if not sorted_values:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(sorted_values),
        "min": sorted_values[0],
        "p50": _percentile(sorted_values, 0.50),
        "p90": _percentile(sorted_values, 0.90),
        "p95": _percentile(sorted_values, 0.95),
        "p99": _percentile(sorted_values, 0.99),
        "max": sorted_values[-1],
        "mean": sum(sorted_values) / len(sorted_values),
    }


def _head_distributions(records: list[dict]) -> dict:
    metrics = {}
    for joint in HEAD_JOINTS:
        for key in ("max_abs_rad", "range_rad", "range_deg"):
            name = f"{joint}.{key}"
            metrics[name] = distribution(
                record["output_18d"]["head_metrics"][joint][key]
                for record in records
                if record["output_18d"]["head_metrics"] is not None
            )
    metrics["max_component_velocity_rad_s"] = distribution(
        record["output_18d"]["head_metrics"]["max_component_velocity_rad_s"]
        for record in records
        if record["output_18d"]["head_metrics"] is not None
    )
    return metrics


def _jsonl_bytes(records: list[dict]) -> bytes:
    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    ]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _model_provenance(path: Path | None, expected_sha256: str | None) -> dict:
    expected = expected_sha256.lower() if expected_sha256 else None
    base = {
        "path": str(path) if path else None,
        "expected_sha256": expected,
        "verification_scope": "verified_once_out_of_band_for_batch",
        "quality_reports_per_clip_sha256_verified": False,
        "quality_report_hash_policy": (
            "batch subprocess used skip; per-clip quality model hash may be null"
        ),
    }
    if path is None or not path.is_file():
        return {
            **base,
            "present": False,
            "sha256": None,
            "sha256_matches_expected": False,
            "size_bytes": None,
            "mtime_ns": None,
        }
    stat = path.stat()
    actual = sha256_file(path)
    return {
        **base,
        "present": True,
        "sha256": actual,
        "sha256_matches_expected": expected is not None and actual == expected,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _load_batch_summary(
    path: Path | None,
    expected_contract: str,
    *,
    inventory_by_id: dict[str, dict],
    dataset_root: Path,
    selected_output_dirs_by_id: dict[str, Path],
) -> dict:
    if path is None or not path.is_file():
        return {"present": False, "path": str(path) if path else None}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid batch summary {path}: {error}") from error
    if payload.get("output_contract") != expected_contract:
        raise ValueError("batch summary output_contract mismatch")
    results = payload.get("results")
    if not isinstance(results, list):
        results = []
    result_ids = []
    unknown_result_ids = []
    action_mismatch_ids = []
    source_mismatch_ids = []
    output_dir_mismatch_ids = []
    for result in results:
        if not isinstance(result, dict):
            continue
        clip_id = result.get("clip_id")
        result_ids.append(clip_id)
        if not isinstance(clip_id, str):
            unknown_result_ids.append(repr(clip_id))
            continue
        inventory = inventory_by_id.get(clip_id)
        if inventory is None:
            unknown_result_ids.append(clip_id)
            continue
        if result.get("action") != inventory["action"]:
            action_mismatch_ids.append(clip_id)
        expected_source = (dataset_root / inventory["motion_relpath"]).resolve()
        if not result.get("source") or Path(result["source"]).resolve() != expected_source:
            source_mismatch_ids.append(clip_id)
        selected_output_dir = selected_output_dirs_by_id.get(clip_id)
        if (
            selected_output_dir is None
            or not result.get("output_dir")
            or Path(result["output_dir"]).resolve() != selected_output_dir
        ):
            output_dir_mismatch_ids.append(clip_id)
    valid_result_ids = [clip_id for clip_id in result_ids if isinstance(clip_id, str)]
    return {
        "present": True,
        "path": str(path),
        "sha256": sha256_file(path),
        "batch_id": payload.get("batch_id"),
        "total_tasks": payload.get("total_tasks"),
        "completed_tasks": payload.get("completed_tasks"),
        "finished": payload.get("finished") is True,
        "counts": payload.get("counts") or {},
        "manifest": payload.get("manifest"),
        "output_root": payload.get("output_root"),
        "rejected_root": payload.get("rejected_root"),
        "smplx_model": payload.get("smplx_model"),
        "smplx_model_sha_check": payload.get("smplx_model_sha_check"),
        "result_count": len(results),
        "result_clip_ids_unique": len(valid_result_ids) == len(set(valid_result_ids)) == len(results),
        "result_inventory_coverage": set(valid_result_ids) == set(inventory_by_id),
        "unknown_result_ids": sorted(unknown_result_ids),
        "action_mismatch_ids": sorted(action_mismatch_ids),
        "source_mismatch_ids": sorted(source_mismatch_ids),
        "output_dir_mismatch_ids": sorted(output_dir_mismatch_ids),
    }


def _scale(records: list[dict]) -> dict:
    source_duration = sum(record["source"]["duration_sec"] for record in records)
    output_duration = sum(record["output_18d"]["duration_sec"] or 0.0 for record in records)
    return {
        "clips": len(records),
        "source_frames": sum(record["source"]["frames"] for record in records),
        "source_duration_sec": source_duration,
        "source_duration_hours": source_duration / 3600.0,
        "output_frames": sum(record["output_18d"]["frames"] or 0 for record in records),
        "output_duration_sec": output_duration,
        "output_duration_hours": output_duration / 3600.0,
        "safe_csv_bytes": sum(
            record["output_18d"]["safe_csv"]["bytes"] or 0 for record in records
        ),
    }


def _action_coverage(inventory: list[dict], records: list[dict], missing_ids: set[str]) -> dict:
    by_action = defaultdict(lambda: Counter(inventory=0, passed=0, rejected=0, missing=0))
    action_for_clip = {}
    for row in inventory:
        by_action[row["action"]]["inventory"] += 1
        action_for_clip[row["clip_id"]] = row["action"]
        if row["clip_id"] in missing_ids:
            by_action[row["action"]]["missing"] += 1
    for record in records:
        key = "passed" if record["retarget_status"] == "retarget_qc_passed" else "rejected"
        by_action[record["action"]][key] += 1
    details = {action: dict(counts) for action, counts in sorted(by_action.items())}
    actions_with_pass = {record["action"] for record in records if record["retarget_status"] == "retarget_qc_passed"}
    all_actions = set(by_action)
    return {
        "inventory_action_count": len(all_actions),
        "actions_with_passed_clip_count": len(actions_with_pass),
        "actions_without_passed_clip_count": len(all_actions - actions_with_pass),
        "actions_without_passed_clip": sorted(all_actions - actions_with_pass),
        "per_action": details,
    }


def summarize_full_retarget(
    inventory_path: Path,
    passed_root: Path,
    rejected_root: Path,
    output_dir: Path,
    *,
    dataset_root: Path | None = None,
    batch_summary_path: Path | None = None,
    smplx_model_path: Path | None = None,
    expected_smplx_sha256: str | None = EXPECTED_SMPLX_SHA256,
    expected_count: int | None = 6944,
    expected_contract: str = "ula_v2_18d_head_v1",
) -> dict:
    inventory_path = inventory_path.resolve()
    passed_root = passed_root.resolve()
    rejected_root = rejected_root.resolve()
    output_dir = output_dir.resolve()
    dataset_root = (dataset_root or inventory_path.parent.parent).resolve()
    if smplx_model_path is not None:
        smplx_model_path = smplx_model_path.resolve()
    if batch_summary_path is None:
        default_summary = passed_root / "_batch_18d_head_v1_summary.json"
        batch_summary_path = default_summary if default_summary.is_file() else None
    elif batch_summary_path is not None:
        batch_summary_path = batch_summary_path.resolve()

    inventory = load_inventory(
        inventory_path,
        expected_count=expected_count,
        expected_contract=expected_contract,
    )
    inventory_ids = {row["clip_id"] for row in inventory}
    candidates, discovery = discover_quality_candidates(
        passed_root,
        rejected_root,
        inventory_ids,
        expected_contract=expected_contract,
    )
    selected_records = []
    missing_ids = []
    for row in inventory:
        selected, selection = select_latest_matching_candidate(
            candidates.get(row["clip_id"], []),
            expected_contract=expected_contract,
        )
        if selected is None:
            missing_ids.append(row["clip_id"])
            continue
        selected_records.append(
            inspect_selected_candidate(
                row,
                selected,
                selection,
                dataset_root=dataset_root,
                expected_contract=expected_contract,
            )
        )
    selected_records.sort(key=lambda item: item["clip_id"])
    passed = [
        record for record in selected_records if record["retarget_status"] == "retarget_qc_passed"
    ]
    rejected = [
        record
        for record in selected_records
        if record["retarget_status"] == "retarget_qc_rejected"
    ]
    missing_ids = sorted(missing_ids)
    if any(record["training_admission"]["accepted_for_training"] for record in selected_records):
        raise AssertionError("retarget QA attempted to grant training admission")
    if len(passed) + len(rejected) + len(missing_ids) != len(inventory):
        raise AssertionError("passed + rejected + missing does not equal inventory size")

    output_dir.mkdir(parents=True, exist_ok=True)
    passed_path = output_dir / PASSED_MANIFEST
    rejected_path = output_dir / REJECTED_MANIFEST
    _atomic_write(passed_path, _jsonl_bytes(passed))
    _atomic_write(rejected_path, _jsonl_bytes(rejected))

    reason_counts = Counter(
        reason for record in rejected for reason in record["quality"]["reasons"]
    )
    gate_failure_counts = Counter(
        reason.removeprefix("gate_failed:")
        for reason in reason_counts.elements()
        if reason.startswith("gate_failed:")
    )
    inventory_by_id = {row["clip_id"]: row for row in inventory}
    selected_output_dirs_by_id = {
        record["clip_id"]: Path(record["output_18d"]["quality_json"]).parent.resolve()
        for record in selected_records
    }
    batch_summary = _load_batch_summary(
        batch_summary_path,
        expected_contract,
        inventory_by_id=inventory_by_id,
        dataset_root=dataset_root,
        selected_output_dirs_by_id=selected_output_dirs_by_id,
    )
    batch_finished = batch_summary["present"] and batch_summary.get("finished") is True
    batch_count_matches = batch_summary["present"] and (
        batch_summary.get("total_tasks") == len(inventory)
        and batch_summary.get("completed_tasks") == len(inventory)
    )
    batch_manifest_matches = batch_summary["present"] and bool(
        batch_summary.get("manifest")
        and Path(batch_summary["manifest"]).resolve() == inventory_path
    )
    batch_root_matches = batch_summary["present"] and bool(
        batch_summary.get("output_root")
        and Path(batch_summary["output_root"]).resolve() == passed_root
        and batch_summary.get("rejected_root")
        and Path(batch_summary["rejected_root"]).resolve() == rejected_root
    )
    batch_counts = batch_summary.get("counts") or {}
    allowed_batch_statuses = {"passed", "quality_failed", "skipped_current_pass"}
    unexpected_batch_statuses = sorted(set(batch_counts) - allowed_batch_statuses)
    batch_counts_are_nonnegative_integers = batch_summary["present"] and all(
        type(value) is int and value >= 0 for value in batch_counts.values()
    )
    batch_status_count_matches = (
        batch_counts_are_nonnegative_integers
        and sum(batch_counts.values()) == len(inventory)
    )
    batch_has_no_process_errors = not unexpected_batch_statuses
    batch_result_output_dirs_match = batch_summary["present"] and not batch_summary.get(
        "output_dir_mismatch_ids"
    )
    batch_result_associations_pass = batch_summary["present"] and all(
        (
            batch_summary.get("result_count") == len(inventory),
            batch_summary.get("result_clip_ids_unique") is True,
            batch_summary.get("result_inventory_coverage") is True,
            not batch_summary.get("unknown_result_ids"),
            not batch_summary.get("action_mismatch_ids"),
            not batch_summary.get("source_mismatch_ids"),
            batch_result_output_dirs_match,
        )
    )
    batch_qc_counts_match_manifests = batch_counts_are_nonnegative_integers and all(
        (
            batch_counts.get("passed", 0)
            + batch_counts.get("skipped_current_pass", 0)
            == len(passed),
            batch_counts.get("quality_failed", 0) == len(rejected),
        )
    )
    model_provenance = _model_provenance(smplx_model_path, expected_smplx_sha256)
    batch_model_path_matches = batch_summary["present"] and bool(
        smplx_model_path
        and batch_summary.get("smplx_model")
        and Path(batch_summary["smplx_model"]).resolve() == smplx_model_path
    )
    batch_model_hash_policy_is_explicit = (
        batch_summary["present"]
        and batch_summary.get("smplx_model_sha_check") == "skipped"
    )
    model_provenance_pass = all(
        (
            model_provenance["present"],
            model_provenance["sha256_matches_expected"],
            batch_model_path_matches,
            batch_model_hash_policy_is_explicit,
        )
    )
    completeness_passed = (
        len(inventory) == expected_count if expected_count is not None else True
    ) and all(
        (
            not missing_ids,
            batch_finished,
            batch_count_matches,
            batch_manifest_matches,
            batch_root_matches,
            batch_status_count_matches,
            batch_counts_are_nonnegative_integers,
            batch_has_no_process_errors,
            batch_result_associations_pass,
            batch_qc_counts_match_manifests,
            model_provenance_pass,
        )
    )
    inventory_source_duration = sum(
        row["frame_count"] / row["nominal_fps"] for row in inventory
    )
    selected_partition_counts = Counter(
        record["quality"]["partition"] for record in selected_records
    )
    selected_batch_counts = Counter(
        record["quality"]["batch_id"] or "unversioned_passed_root"
        for record in selected_records
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "policy": {
            "output_contract": expected_contract,
            "training_admission": "deny_until_independent_semantic_review",
            "retarget_qc_grants_training_admission": False,
            "quality_selection": "latest_mtime_among_matching_contract_candidates",
        },
        "inputs": {
            "inventory": {
                "path": str(inventory_path),
                "sha256": sha256_file(inventory_path),
                "records": len(inventory),
            },
            "dataset_root": str(dataset_root),
            "passed_root": str(passed_root),
            "rejected_root": str(rejected_root),
            "batch_summary": batch_summary,
            "smplx_model_provenance": model_provenance,
        },
        "counts": {
            "inventory": len(inventory),
            "passed": len(passed),
            "rejected": len(rejected),
            "missing": len(missing_ids),
        },
        "rates": {
            "physical_pass_fraction": len(passed) / len(inventory) if inventory else 0.0,
            "physical_reject_fraction": len(rejected) / len(inventory) if inventory else 0.0,
            "missing_fraction": len(missing_ids) / len(inventory) if inventory else 0.0,
        },
        "completeness": {
            "passed": completeness_passed,
            "equation": "passed + rejected + missing == inventory",
            "equation_holds": len(passed) + len(rejected) + len(missing_ids) == len(inventory),
            "batch_finished": batch_finished,
            "batch_count_matches_inventory": batch_count_matches,
            "batch_manifest_matches_inventory": batch_manifest_matches,
            "batch_roots_match_inputs": batch_root_matches,
            "batch_status_count_matches_inventory": batch_status_count_matches,
            "batch_counts_are_nonnegative_integers": batch_counts_are_nonnegative_integers,
            "batch_has_no_process_errors": batch_has_no_process_errors,
            "batch_result_associations_pass": batch_result_associations_pass,
            "batch_result_output_dirs_match_selected_quality": (
                batch_result_output_dirs_match
            ),
            "batch_qc_counts_match_manifests": batch_qc_counts_match_manifests,
            "model_provenance_pass": model_provenance_pass,
            "batch_model_path_matches": batch_model_path_matches,
            "batch_model_hash_policy_is_explicit": batch_model_hash_policy_is_explicit,
            "unexpected_batch_statuses": unexpected_batch_statuses,
            "missing_clip_ids": missing_ids,
        },
        "scale": {
            "inventory": {
                "clips": len(inventory),
                "source_frames": sum(row["frame_count"] for row in inventory),
                "source_duration_sec": sum(
                    row["frame_count"] / row["nominal_fps"] for row in inventory
                ),
                "source_duration_hours": inventory_source_duration / 3600.0,
            },
            "all_selected": _scale(selected_records),
            "passed": _scale(passed),
            "rejected": _scale(rejected),
        },
        "action_coverage": _action_coverage(inventory, selected_records, set(missing_ids)),
        "failure_reason_counts": dict(sorted(reason_counts.items())),
        "failed_gate_counts": dict(sorted(gate_failure_counts.items())),
        "head_distributions": {
            "all_selected": _head_distributions(selected_records),
            "passed": _head_distributions(passed),
            "rejected": _head_distributions(rejected),
        },
        "evidence_discovery": {
            **discovery,
            "clips_with_multiple_matching_candidates": sum(
                1
                for records in candidates.values()
                if sum(
                    isinstance(candidate["quality"], dict)
                    and candidate["quality"].get("output_contract") == expected_contract
                    for candidate in records
                )
                > 1
            ),
            "superseded_matching_quality_count": sum(
                len(record["quality"]["selection"]["superseded_quality_paths"])
                for record in selected_records
            ),
            "selected_partition_counts": dict(sorted(selected_partition_counts.items())),
            "selected_batch_counts": dict(sorted(selected_batch_counts.items())),
        },
        "association_checks": {
            "inventory_action_clip_mismatch_count": sum(
                not row["action_associated"] for row in inventory
            ),
            "inventory_accepted_for_training_count": sum(
                row["accepted_for_training"] for row in inventory
            ),
            "source_association_failure_count": sum(
                reason_counts[reason]
                for reason in (
                    "source_path_mismatch",
                    "source_sha256_mismatch",
                    "source_frames_mismatch",
                    "source_fps_mismatch",
                )
            ),
            "batch_result_action_mismatch_count": len(
                batch_summary.get("action_mismatch_ids") or []
            ),
            "batch_result_source_mismatch_count": len(
                batch_summary.get("source_mismatch_ids") or []
            ),
            "batch_result_output_dir_mismatch_count": len(
                batch_summary.get("output_dir_mismatch_ids") or []
            ),
            "batch_model_path_matches": batch_model_path_matches,
            "batch_model_hash_policy_is_explicit": batch_model_hash_policy_is_explicit,
        },
        "outputs": {
            "passed_manifest": {
                "path": str(passed_path),
                "records": len(passed),
                "sha256": sha256_file(passed_path),
            },
            "rejected_manifest": {
                "path": str(rejected_path),
                "records": len(rejected),
                "sha256": sha256_file(rejected_path),
            },
        },
        "invariants": {
            "inventory_clip_ids_unique": len(inventory_ids) == len(inventory),
            "every_selected_record_has_18d_contract": all(
                record["output_18d"]["output_contract"] == expected_contract
                for record in selected_records
            ),
            "passed_manifest_all_qc_passed": all(
                record["quality"]["passed"] is True for record in passed
            ),
            "no_manifest_grants_training_admission": all(
                record["training_admission"]["accepted_for_training"] is False
                for record in selected_records
            ),
        },
    }
    report_path = output_dir / REPORT_NAME
    _atomic_write(
        report_path,
        (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--passed-root", type=Path, required=True)
    parser.add_argument("--rejected-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--batch-summary", type=Path)
    parser.add_argument("--smplx-model", type=Path, default=DEFAULT_SMPLX_MODEL)
    parser.add_argument(
        "--expected-smplx-sha256",
        default=EXPECTED_SMPLX_SHA256,
    )
    parser.add_argument("--expected-count", type=int, default=6944)
    parser.add_argument("--expected-contract", default="ula_v2_18d_head_v1")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    report = summarize_full_retarget(
        args.inventory,
        args.passed_root,
        args.rejected_root,
        args.output_dir,
        dataset_root=args.dataset_root,
        batch_summary_path=args.batch_summary,
        smplx_model_path=args.smplx_model,
        expected_smplx_sha256=args.expected_smplx_sha256,
        expected_count=None if args.expected_count == 0 else args.expected_count,
        expected_contract=args.expected_contract,
    )
    print(
        json.dumps(
            {
                "counts": report["counts"],
                "complete": report["completeness"]["passed"],
                "output_dir": str(args.output_dir.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.require_complete and not report["completeness"]["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
