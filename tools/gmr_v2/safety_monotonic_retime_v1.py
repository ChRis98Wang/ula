#!/usr/bin/env python3
"""Minimal 30 Hz monotonic slowdown for joint-velocity safety."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.signal import savgol_filter


ALGORITHM_NAME = "ula_18d_minimum_velocity_safety_monotonic_retime_v1"
MAX_SLOWDOWN_RATIO = 1.25
ALGORITHM_CONTRACT = {
    "algorithm": ALGORITHM_NAME,
    "endpoint_policy": "joint_limit_clipped_filtered_endpoints_preserved_exactly",
    "fixed_or_target_duration_allowed": False,
    "input_frame_mapping": "strictly_increasing_all_input_frames_retained",
    "interpolation": "piecewise_linear_per_joint",
    "output_grid": "uniform_30hz_including_both_endpoints",
    "output_interval_count": "ceil(sum(required_segment_duration_sec)*fps)",
    "required_segment_duration_sec": (
        "max(1/fps,max_abs_filtered_joint_delta/max_velocity_rad_s)"
    ),
    "slack_policy": "sub_frame_rounding_slack_distributed_proportionally",
    "slowdown_only": True,
    "source_crop_allowed": False,
    "source_tile_allowed": False,
    "version": 1,
}
ALGORITHM_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        ALGORITHM_CONTRACT,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def trajectory_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(b"ula-trajectory-float64-le-v1\0")
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _per_joint_max_abs_derivative(value: np.ndarray, fps: float) -> np.ndarray:
    if len(value) < 2:
        return np.zeros(value.shape[1], dtype=np.float64)
    return np.max(np.abs(np.diff(value, axis=0)) * float(fps), axis=0)


def minimum_velocity_safety_retime(
    raw: np.ndarray,
    *,
    fps: float,
    max_velocity_rad_s: float,
    smoothing_window: int,
    joint_order: Sequence[str],
    joint_limits: Mapping[str, tuple[float, float]],
    elbow_enforcer: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return a full-arc slowdown on the minimum feasible uniform 30 Hz grid."""

    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] < 3 or raw.shape[1] != len(joint_order):
        raise ValueError("raw trajectory must be [frames>=3, len(joint_order)]")
    if not np.isfinite(raw).all():
        raise ValueError("raw trajectory contains non-finite values")
    if not math.isfinite(float(fps)) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if not math.isfinite(float(max_velocity_rad_s)) or max_velocity_rad_s <= 0:
        raise ValueError("max_velocity_rad_s must be finite and positive")

    lower = np.asarray([joint_limits[name][0] for name in joint_order], dtype=np.float64)
    upper = np.asarray([joint_limits[name][1] for name in joint_order], dtype=np.float64)
    clipped = np.clip(raw, lower, upper)
    window = min(
        int(smoothing_window),
        len(clipped) if len(clipped) % 2 else len(clipped) - 1,
    )
    if window >= 5:
        if window % 2 == 0:
            window -= 1
        filtered = savgol_filter(
            clipped,
            window_length=window,
            polyorder=2,
            axis=0,
            mode="interp",
        )
    else:
        filtered = clipped.copy()
    filtered = np.clip(filtered, lower, upper)
    # Savitzky-Golay extrapolation may move the boundaries. The expression
    # arc contract requires both safe source endpoints to survive unchanged.
    filtered[0] = clipped[0]
    filtered[-1] = clipped[-1]
    if elbow_enforcer is not None:
        filtered = np.asarray(elbow_enforcer(filtered), dtype=np.float64)
    filtered = np.round(np.clip(filtered, lower, upper), 8)

    nominal_dt = 1.0 / float(fps)
    absolute_delta = np.abs(np.diff(filtered, axis=0))
    segment_peak_delta = np.max(absolute_delta, axis=1)
    required_segment_duration = np.maximum(
        nominal_dt, segment_peak_delta / float(max_velocity_rad_s)
    )
    required_duration = float(required_segment_duration.sum())
    output_interval_count = max(
        len(filtered) - 1,
        int(math.ceil(required_duration * float(fps) - 1e-12)),
    )
    output_duration = output_interval_count / float(fps)
    scaled_segment_duration = required_segment_duration * (
        output_duration / required_duration
    )
    input_frame_output_times = np.r_[0.0, np.cumsum(scaled_segment_duration)]
    input_frame_output_times[-1] = output_duration
    output_times = np.arange(output_interval_count + 1, dtype=np.float64) / float(fps)
    safe = np.column_stack(
        [
            np.interp(
                output_times,
                input_frame_output_times,
                filtered[:, joint_index],
            )
            for joint_index in range(filtered.shape[1])
        ]
    )
    safe[0] = filtered[0]
    safe[-1] = filtered[-1]
    if elbow_enforcer is not None:
        safe = np.asarray(elbow_enforcer(safe), dtype=np.float64)
    safe = np.round(np.clip(safe, lower, upper), 8)

    pre_velocity = _per_joint_max_abs_derivative(filtered, fps)
    raw_velocity = _per_joint_max_abs_derivative(raw, fps)
    post_velocity = _per_joint_max_abs_derivative(safe, fps)
    triggering_indices = np.flatnonzero(
        pre_velocity > float(max_velocity_rad_s) + 1e-9
    )
    if len(safe) >= 3:
        post_acceleration = np.max(
            np.abs(np.diff(safe, n=2, axis=0)) * float(fps) ** 2,
            axis=0,
        )
    else:
        post_acceleration = np.zeros(safe.shape[1], dtype=np.float64)
    source_frames = int(len(filtered))
    output_frames = int(len(safe))
    audit = {
        "artifact_kind": "ula_18d_safety_monotonic_retime_v1",
        "algorithm": ALGORITHM_NAME,
        "algorithm_contract": dict(ALGORITHM_CONTRACT),
        "algorithm_contract_sha256": ALGORITHM_CONTRACT_SHA256,
        "algorithm_implementation_sha256": implementation_sha256(),
        "raw_input_trajectory_sha256": trajectory_sha256(raw),
        "filtered_input_trajectory_sha256": trajectory_sha256(filtered),
        "output_trajectory_sha256": trajectory_sha256(safe),
        "fps": float(fps),
        "max_velocity_rad_s": float(max_velocity_rad_s),
        "max_slowdown_ratio": MAX_SLOWDOWN_RATIO,
        "source_frame_count": source_frames,
        "output_frame_count": output_frames,
        "retime_ratio": float(output_frames / source_frames),
        "source_sample_span_sec": float((source_frames - 1) / fps),
        "required_continuous_sample_span_sec": required_duration,
        "output_sample_span_sec": float((output_frames - 1) / fps),
        "sample_span_slowdown_ratio": float(
            (output_frames - 1) / (source_frames - 1)
        ),
        "minimum_output_frame_count": output_interval_count + 1,
        "input_frame_output_times_sec": input_frame_output_times.tolist(),
        "time_map_strictly_increasing": bool(
            np.all(np.diff(input_frame_output_times) > 0)
        ),
        "first_frame_preserved": bool(np.array_equal(safe[0], filtered[0])),
        "last_frame_preserved": bool(np.array_equal(safe[-1], filtered[-1])),
        "retime_input_first_frame": filtered[0].tolist(),
        "retime_input_last_frame": filtered[-1].tolist(),
        "triggering_joints": [joint_order[index] for index in triggering_indices],
        "triggering_segment_count": int(
            np.count_nonzero(segment_peak_delta > max_velocity_rad_s / fps + 1e-12)
        ),
        "raw_max_velocity_rad_s_by_joint": {
            name: float(raw_velocity[index]) for index, name in enumerate(joint_order)
        },
        "pre_retime_max_velocity_rad_s_by_joint": {
            name: float(pre_velocity[index]) for index, name in enumerate(joint_order)
        },
        "post_retime_max_velocity_rad_s_by_joint": {
            name: float(post_velocity[index]) for index, name in enumerate(joint_order)
        },
        "post_retime_max_acceleration_rad_s2_by_joint": {
            name: float(post_acceleration[index])
            for index, name in enumerate(joint_order)
        },
        "post_velocity_pass": bool(
            np.max(post_velocity, initial=0.0)
            <= float(max_velocity_rad_s) + 1e-6
        ),
        "slowdown_ratio_pass": bool(
            output_frames / source_frames <= MAX_SLOWDOWN_RATIO + 1e-12
        ),
        "cropped": False,
        "tiled": False,
        "target_duration_sec": None,
        "blind_review_must_use_retimed_output": True,
    }
    return safe, input_frame_output_times, output_times, audit


__all__ = [
    "ALGORITHM_CONTRACT",
    "ALGORITHM_CONTRACT_SHA256",
    "ALGORITHM_NAME",
    "MAX_SLOWDOWN_RATIO",
    "implementation_sha256",
    "minimum_velocity_safety_retime",
    "trajectory_sha256",
]
