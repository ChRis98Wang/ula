#!/usr/bin/env python3
"""Fail-closed output contract for v8 expression-turn 18D retargeting."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tools.gmr_v2.safety_monotonic_retime_v1 import (
    ALGORITHM_CONTRACT,
    ALGORITHM_CONTRACT_SHA256,
    ALGORITHM_NAME,
    MAX_SLOWDOWN_RATIO,
    implementation_sha256,
    trajectory_sha256,
)
from upper_body_skeleton.retarget_v2_18d import CONTRACT_VERSION, JOINT_ORDER_18D

from .expression_turn_contract import (
    ExpressionTurnContractError,
    validate_expression_turn_candidate,
)


QUALITY_ARTIFACT_KIND = "beat2_expression_turn_18d_v8_quality"
RETARGET_SEGMENT_REPRESENTATION = (
    "native_variable_length_expression_turn_safety_retimed_30hz_v2"
)
TRAINING_ADMISSION_STATUS = "pending_independent_arc_action_affect_review"
REQUIRED_18D_GATES = {
    "joint_limits_pass",
    "velocity_pass",
    "target_fit_pass",
    "collision_pass",
    "axis_direction_pass",
    "head_joint_limits_pass",
    "head_velocity_pass",
    "head_direction_pass",
    "head_continuity_pass",
    "passed",
}
FORBIDDEN_LEGACY_KEYS = {
    "semantic_event",
    "official_semantic_event",
    "official_gesture_semantic_spans",
    "semantic_label_status",
    "semantic_gesture",
    "prompt",
    "prompt_schema",
    "prompt_source",
    "prompt_sha256",
    "prompt_contract",
}
FORBIDDEN_LEGACY_VALUE_TOKENS = {
    "native_variable_length_semantic_event",
    "official_semantic_event_variable_length",
    "full_nonoverlap_6s",
    "six_second_window",
}


def _require_mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExpressionTurnContractError(f"{path} must be an object")
    return value


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _walk(value: object, path: str = "quality"):
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            yield str(key), child, child_path
            yield from _walk(child, child_path)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}[{index}]")


def _validate_no_legacy_semantic_contract(quality: Mapping[str, Any]) -> None:
    for key, value, path in _walk(quality):
        if key in FORBIDDEN_LEGACY_KEYS:
            raise ExpressionTurnContractError(
                f"{path} leaks the legacy semantic-event retarget contract"
            )
        if isinstance(value, str) and any(
            token in value for token in FORBIDDEN_LEGACY_VALUE_TOKENS
        ):
            raise ExpressionTurnContractError(
                f"{path} contains a legacy semantic-event representation"
            )


def _stable_contract_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _validate_retarget_segment(
    value: object,
    *,
    start_frame: int,
    end_frame: int,
    frames: int,
    fps: float,
) -> None:
    segment = _require_mapping(value, "quality.retarget_segment")
    recorded_hash = segment.get("sha256")
    payload = {key: item for key, item in segment.items() if key != "sha256"}
    if not _is_sha256(recorded_hash) or _stable_contract_sha256(payload) != recorded_hash:
        raise ExpressionTurnContractError("quality.retarget_segment SHA256 is invalid")
    source_frames = end_frame - start_frame
    expected = {
        "representation": RETARGET_SEGMENT_REPRESENTATION,
        "source_start_frame": start_frame,
        "source_end_frame_exclusive": end_frame,
        "source_frame_count": source_frames,
        "output_frame_count": frames,
        "fps": fps,
        "retimed": frames != source_frames,
        "cropped": False,
        "duration_policy": "natural_rest_to_natural_rest_no_fixed_or_max_duration",
        "retime_policy": ALGORITHM_NAME,
        "max_slowdown_ratio": MAX_SLOWDOWN_RATIO,
        "fixed_target_duration_sec": None,
        "planner_duration_field": "output_sample_span_sec",
        "source_boundary_duration_field": "source_frame_coverage_sec",
    }
    for field, expected_value in expected.items():
        if segment.get(field) != expected_value:
            raise ExpressionTurnContractError(
                f"quality.retarget_segment.{field} does not match the expression turn"
            )
    durations = {
        "source_frame_coverage_sec": source_frames / fps,
        "output_sample_span_sec": max(0, frames - 1) / fps,
        "output_frame_coverage_sec": frames / fps,
    }
    for field, expected_value in durations.items():
        value = segment.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isclose(
                float(value), float(expected_value), rel_tol=0.0, abs_tol=1e-9
            )
        ):
            raise ExpressionTurnContractError(
                f"quality.retarget_segment.{field} has invalid duration semantics"
            )
    if segment.get("legacy_quality_duration_sec_role") != (
        "output_frame_coverage_compatibility_only_not_planner_target"
    ):
        raise ExpressionTurnContractError(
            "quality.retarget_segment legacy duration role is invalid"
        )


def _validate_safe_csv(
    path: Path,
    *,
    expected_sha256: str,
    expected_frames: int,
) -> np.ndarray:
    if not path.is_file():
        raise ExpressionTurnContractError(f"safe 18D CSV does not exist: {path}")
    if not _is_sha256(expected_sha256) or _sha256_file(path) != expected_sha256:
        raise ExpressionTurnContractError("safe 18D CSV SHA256 does not match")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(JOINT_ORDER_18D):
            raise ExpressionTurnContractError("safe CSV joint order is not exact 18D")
        values_by_row = []
        for row_count, row in enumerate(reader, 1):
            values = []
            for joint in JOINT_ORDER_18D:
                try:
                    value = float(row[joint])
                except (KeyError, TypeError, ValueError) as error:
                    raise ExpressionTurnContractError(
                        f"safe CSV contains a non-numeric {joint} value"
                    ) from error
                if not math.isfinite(value):
                    raise ExpressionTurnContractError(
                        f"safe CSV contains a non-finite {joint} value"
                    )
                values.append(value)
            values_by_row.append(values)
    row_count = len(values_by_row)
    if row_count != expected_frames:
        raise ExpressionTurnContractError(
            "safe CSV row count does not match retarget output frames"
        )
    return np.asarray(values_by_row, dtype=np.float64)


def _finite_number(value: object) -> bool:
    return bool(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _validate_safety_retime(
    value: object,
    *,
    source_frames: int,
    output_frames: int,
    fps: float,
    safe_csv: np.ndarray,
) -> None:
    audit = _require_mapping(value, "quality.safety_monotonic_retime")
    exact = {
        "artifact_kind": "ula_18d_safety_monotonic_retime_v1",
        "algorithm": ALGORITHM_NAME,
        "algorithm_contract": ALGORITHM_CONTRACT,
        "algorithm_contract_sha256": ALGORITHM_CONTRACT_SHA256,
        "algorithm_implementation_sha256": implementation_sha256(),
        "fps": fps,
        "max_slowdown_ratio": MAX_SLOWDOWN_RATIO,
        "source_frame_count": source_frames,
        "output_frame_count": output_frames,
        "minimum_output_frame_count": output_frames,
        "time_map_strictly_increasing": True,
        "first_frame_preserved": True,
        "last_frame_preserved": True,
        "post_velocity_pass": True,
        "slowdown_ratio_pass": True,
        "cropped": False,
        "tiled": False,
        "target_duration_sec": None,
        "blind_review_must_use_retimed_output": True,
    }
    for field, expected in exact.items():
        if audit.get(field) != expected:
            raise ExpressionTurnContractError(
                f"quality.safety_monotonic_retime.{field} is invalid"
            )
    for field in (
        "raw_input_trajectory_sha256",
        "filtered_input_trajectory_sha256",
        "output_trajectory_sha256",
    ):
        if not _is_sha256(audit.get(field)):
            raise ExpressionTurnContractError(
                f"quality.safety_monotonic_retime.{field} is not a SHA256"
            )
    if trajectory_sha256(safe_csv) != audit.get("output_trajectory_sha256"):
        raise ExpressionTurnContractError(
            "quality safety-retime output trajectory SHA256 does not match safe CSV"
        )

    expected_ratio = output_frames / source_frames
    source_span = (source_frames - 1) / fps
    output_span = (output_frames - 1) / fps
    expected_sample_ratio = output_span / source_span
    numeric_expected = {
        "retime_ratio": expected_ratio,
        "source_sample_span_sec": source_span,
        "output_sample_span_sec": output_span,
        "sample_span_slowdown_ratio": expected_sample_ratio,
    }
    for field, expected in numeric_expected.items():
        actual = audit.get(field)
        if not _finite_number(actual) or not math.isclose(
            float(actual), float(expected), rel_tol=0.0, abs_tol=1e-9
        ):
            raise ExpressionTurnContractError(
                f"quality.safety_monotonic_retime.{field} is inconsistent"
            )
    if output_frames < source_frames or expected_ratio > MAX_SLOWDOWN_RATIO + 1e-12:
        raise ExpressionTurnContractError(
            "quality safety-retime slowdown ratio is outside the fail-closed bound"
        )

    required_span = audit.get("required_continuous_sample_span_sec")
    if (
        not _finite_number(required_span)
        or float(required_span) < source_span - 1e-9
        or float(required_span) > output_span + 1e-9
        or output_span - float(required_span) >= 1.0 / fps + 1e-9
    ):
        raise ExpressionTurnContractError(
            "quality safety-retime is not the minimum uniform-grid slowdown"
        )
    expected_minimum_frames = int(math.ceil(float(required_span) * fps - 1e-12)) + 1
    if expected_minimum_frames != output_frames:
        raise ExpressionTurnContractError(
            "quality safety-retime output frame count is not minimal"
        )

    time_map = audit.get("input_frame_output_times_sec")
    if (
        not isinstance(time_map, list)
        or len(time_map) != source_frames
        or any(not _finite_number(item) for item in time_map)
        or not math.isclose(float(time_map[0]), 0.0, abs_tol=1e-12)
        or not math.isclose(float(time_map[-1]), output_span, abs_tol=1e-9)
        or any(float(right) <= float(left) for left, right in zip(time_map, time_map[1:]))
    ):
        raise ExpressionTurnContractError(
            "quality safety-retime time map is not full and strictly monotonic"
        )

    for field, csv_endpoint in (
        ("retime_input_first_frame", safe_csv[0]),
        ("retime_input_last_frame", safe_csv[-1]),
    ):
        endpoint = audit.get(field)
        if (
            not isinstance(endpoint, list)
            or len(endpoint) != len(JOINT_ORDER_18D)
            or any(not _finite_number(item) for item in endpoint)
            or not np.allclose(
                np.asarray(endpoint, dtype=np.float64),
                csv_endpoint,
                rtol=0.0,
                atol=5e-9,
            )
        ):
            raise ExpressionTurnContractError(
                f"quality safety-retime did not preserve {field}"
            )

    velocity_fields = (
        "raw_max_velocity_rad_s_by_joint",
        "pre_retime_max_velocity_rad_s_by_joint",
        "post_retime_max_velocity_rad_s_by_joint",
        "post_retime_max_acceleration_rad_s2_by_joint",
    )
    velocity_maps = {}
    for field in velocity_fields:
        values = audit.get(field)
        if (
            not isinstance(values, Mapping)
            or set(values) != set(JOINT_ORDER_18D)
            or any(not _finite_number(item) or float(item) < 0 for item in values.values())
        ):
            raise ExpressionTurnContractError(
                f"quality.safety_monotonic_retime.{field} is invalid"
            )
        velocity_maps[field] = values
    max_velocity = audit.get("max_velocity_rad_s")
    if not _finite_number(max_velocity) or float(max_velocity) <= 0:
        raise ExpressionTurnContractError(
            "quality safety-retime velocity limit is invalid"
        )
    if any(
        float(value) > float(max_velocity) + 1e-6
        for value in velocity_maps["post_retime_max_velocity_rad_s_by_joint"].values()
    ):
        raise ExpressionTurnContractError(
            "quality safety-retime output exceeds its velocity limit"
        )
    triggering = audit.get("triggering_joints")
    expected_triggering = [
        joint
        for joint in JOINT_ORDER_18D
        if float(velocity_maps["pre_retime_max_velocity_rad_s_by_joint"][joint])
        > float(max_velocity) + 1e-9
    ]
    if triggering != expected_triggering:
        raise ExpressionTurnContractError(
            "quality safety-retime triggering joint list is inconsistent"
        )
    trigger_count = audit.get("triggering_segment_count")
    if isinstance(trigger_count, bool) or not isinstance(trigger_count, int) or trigger_count < 0:
        raise ExpressionTurnContractError(
            "quality safety-retime triggering segment count is invalid"
        )


def validate_safety_monotonic_retime(
    value: object,
    *,
    source_frames: int,
    output_frames: int,
    fps: float,
    safe_csv_path: str | Path,
    safe_csv_sha256: str,
) -> None:
    """Validate a final CSV against the complete monotonic-retime audit."""

    safe_csv = _validate_safe_csv(
        Path(safe_csv_path).resolve(),
        expected_sha256=safe_csv_sha256,
        expected_frames=output_frames,
    )
    _validate_safety_retime(
        value,
        source_frames=source_frames,
        output_frames=output_frames,
        fps=fps,
        safe_csv=safe_csv,
    )


def validate_expression_turn_retarget_output(
    quality: Mapping[str, Any],
    *,
    input_record: Mapping[str, Any],
    catalog_binding: Mapping[str, Any],
    safe_csv_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate one 18D output while keeping all training admission closed."""

    input_report = validate_expression_turn_candidate(
        input_record, catalog_binding=catalog_binding
    )
    quality = _require_mapping(quality, "quality")
    _validate_no_legacy_semantic_contract(quality)
    if quality.get("artifact_kind") != QUALITY_ARTIFACT_KIND:
        raise ExpressionTurnContractError("quality artifact_kind is invalid")
    if quality.get("output_contract") != CONTRACT_VERSION:
        raise ExpressionTurnContractError("quality output_contract is not ULA v2 18D")
    if quality.get("action_dim") != 18 or quality.get("joint_order") != list(
        JOINT_ORDER_18D
    ):
        raise ExpressionTurnContractError("quality joint contract is not exact 18D")

    fps = float(input_record["fps"])
    if quality.get("fps") != input_record.get("fps"):
        raise ExpressionTurnContractError("quality fps does not match input candidate")
    input_segment = input_record["training_segment"]
    start_frame = int(input_segment["start_frame"])
    end_frame = int(input_segment["end_frame_exclusive"])
    source_frames = end_frame - start_frame
    if quality.get("training_segment") != input_segment:
        raise ExpressionTurnContractError(
            "quality training_segment changed the natural expression-turn boundary"
        )
    for field in ("context_plan", "time_axes", "expression_turn"):
        if quality.get(field) != input_record.get(field):
            raise ExpressionTurnContractError(
                f"quality {field} does not preserve the input contract"
            )
    if (
        quality.get("source_window_start_frame") != start_frame
        or quality.get("source_window_end_frame_exclusive") != end_frame
        or quality.get("source_window_frames") != source_frames
    ):
        raise ExpressionTurnContractError(
            "quality source window does not match the natural training segment"
        )

    lineage = input_report["lineage"]
    for field, expected in lineage.items():
        if quality.get(field) != expected:
            raise ExpressionTurnContractError(f"quality lineage mismatch: {field}")
    preserved_hashes = (
        "expression_turn_contract_sha256",
        "source_inventory_manifest_sha256",
        "split_assignment_manifest_sha256",
        "motion_sha256",
        "upstream_event_record_sha256",
    )
    for field in preserved_hashes:
        if quality.get(field) != input_record.get(field):
            raise ExpressionTurnContractError(
                f"quality did not preserve input lineage field {field}"
            )
    if quality.get("source_sha256") != input_record.get("motion_sha256"):
        raise ExpressionTurnContractError("quality source SHA256 is not the input motion")

    frames = quality.get("frames")
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 3:
        raise ExpressionTurnContractError("quality output frame count is invalid")
    if frames < source_frames:
        raise ExpressionTurnContractError(
            "30 Hz expression turn was shortened or lost natural boundary coverage"
        )
    _validate_retarget_segment(
        quality.get("retarget_segment"),
        start_frame=start_frame,
        end_frame=end_frame,
        frames=frames,
        fps=fps,
    )

    gates = _require_mapping(quality.get("quality_gate"), "quality.quality_gate")
    if not REQUIRED_18D_GATES.issubset(gates):
        missing = sorted(REQUIRED_18D_GATES.difference(gates))
        raise ExpressionTurnContractError(f"quality is missing 18D gates: {missing}")
    if any(not isinstance(value, bool) for value in gates.values()):
        raise ExpressionTurnContractError("all reported quality gates must be booleans")
    if any(value is not True for value in gates.values()):
        raise ExpressionTurnContractError("all reported 18D quality gates must pass")

    if quality.get("semantic_supervision_masks") != input_record.get(
        "semantic_supervision_masks"
    ) or any(quality["semantic_supervision_masks"].values()):
        raise ExpressionTurnContractError("retarget output semantic masks are not zero")
    false_fields = (
        "emotion_supervision_mask",
        "official_emotion_conditioning_enabled",
        "affect_observable_supervision_mask",
        "official_category_conditioning_enabled",
        "accepted_for_training",
    )
    if any(quality.get(field) is not False for field in false_fields):
        raise ExpressionTurnContractError(
            "retarget output enabled semantic, emotion, or training admission"
        )
    if quality.get("canonical_prompt") is not None or quality.get("canonical_action") is not None:
        raise ExpressionTurnContractError(
            "retarget output leaked pre-review action conditioning"
        )
    if quality.get("training_admission_status") != TRAINING_ADMISSION_STATUS:
        raise ExpressionTurnContractError("retarget output admission status is invalid")

    outputs = _require_mapping(quality.get("outputs"), "quality.outputs")
    declared_safe_path = Path(str(outputs.get("safe_csv") or "")).resolve()
    resolved_safe_path = (
        Path(safe_csv_path).resolve() if safe_csv_path is not None else declared_safe_path
    )
    if declared_safe_path != resolved_safe_path:
        raise ExpressionTurnContractError("quality safe CSV path binding does not match")
    safe_csv_sha256 = quality.get("safe_csv_sha256")
    safe_csv = _validate_safe_csv(
        resolved_safe_path,
        expected_sha256=str(safe_csv_sha256 or ""),
        expected_frames=frames,
    )
    _validate_safety_retime(
        quality.get("safety_monotonic_retime"),
        source_frames=source_frames,
        output_frames=frames,
        fps=fps,
        safe_csv=safe_csv,
    )

    return {
        "clip_id": input_report["clip_id"],
        "retarget_output_valid": True,
        "physical_qc_eligible": True,
        "base_motion_eligible": False,
        "base_motion_status": "pending_independent_blind_arc_review",
        "semantic_conditioning_eligible": False,
        "expressive_conditioning_eligible": False,
        "emotion_supervision_mask": False,
        "accepted_for_training": False,
        "frames": frames,
        "lineage": dict(lineage),
    }


__all__ = [
    "QUALITY_ARTIFACT_KIND",
    "REQUIRED_18D_GATES",
    "RETARGET_SEGMENT_REPRESENTATION",
    "TRAINING_ADMISSION_STATUS",
    "validate_expression_turn_retarget_output",
    "validate_safety_monotonic_retime",
]
