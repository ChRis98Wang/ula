#!/usr/bin/env python3
"""Draft conservative bilingual motion labels from ULA V2 18D trajectories.

Only robot joint trajectories are used to choose prompt text. Speech transcripts
are copied into a separate context field and are never treated as action labels.
Every generated label remains pending human review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "deliverables/interactive_human_motion_v1/samples/beat2_conversation/sample_manifest.jsonl"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "deliverables/interactive_human_motion_v1/annotations/ula_v2_18d_drafts_v1"
)
SCHEMA_VERSION = "1.1.0"
ALGORITHM_VERSION = "ula_v2_18d_observable_rules_v1.2.0"
PROMPT_TEMPLATE_VERSION = "ula_v2_18d_bilingual_prompt_v1.2.1"
ROBOT_CONTRACT = "ula_v2_18d_head_v1"
PROMPT_PROVENANCE = "trajectory_only_ula_v2_18d_features_no_speech_semantics"
SPEECH_CONTEXT_ROLE = "speech_context_only_not_action_label"

JOINT_ORDER = [
    "joint_pelvisYaw",
    "joint_pelvisPitch",
    "joint_pelvisRoll",
    "joint_lShoulderPitch",
    "joint_lShoulderRoll",
    "joint_lShoulderYaw",
    "joint_lElbow",
    "joint_lWristRoll",
    "joint_lWristPitch",
    "joint_rShoulderPitch",
    "joint_rShoulderRoll",
    "joint_rShoulderYaw",
    "joint_rElbow",
    "joint_rWristRoll",
    "joint_rWristPitch",
    "head_roll_joint",
    "head_pitch_joint",
    "head_yaw_joint",
]

GROUPS = {
    "torso": slice(0, 3),
    "left_arm": slice(3, 9),
    "right_arm": slice(9, 15),
    "head": slice(15, 18),
}

# These versioned rules describe kinematics, never communicative intent/emotion.
THRESHOLDS = {
    "minimum_frames": 12,
    "minimum_fps": 1.0,
    "static_arm_speed_rad_s": 0.035,
    "active_side_fraction_of_dominant": 0.55,
    "subtle_arm_excursion_deg": 8.0,
    "small_arm_excursion_deg": 24.0,
    "large_arm_excursion_deg": 55.0,
    "motion_activity_speed_rad_s": 0.12,
    "intermittent_activity_fraction": 0.38,
    "continuous_activity_fraction": 0.72,
    "coordinated_speed_correlation": 0.55,
    "coordinated_energy_ratio_min": 0.55,
    "period_min_sec": 0.35,
    "period_max_sec": 2.0,
    "periodic_autocorrelation_min": 0.48,
    "head_active_excursion_deg": 5.0,
    "head_clear_excursion_deg": 18.0,
    "torso_active_excursion_deg": 4.0,
    "torso_clear_excursion_deg": 15.0,
    "axis_dominance_min_ratio": 1.25,
    "overall_active_joint_excursion_deg": 5.0,
    "overall_active_joint_speed_rad_s": 0.03,
    "overall_pace_slow_max_hz": 0.30,
    "overall_pace_quick_min_hz": 0.42,
    "arm_energy_balanced_min_ratio": 0.82,
    "head_repeat_min_excursion_deg": 12.0,
    "head_repeat_min_sweeps": 4,
    "repeat_band_low_percentile": 20.0,
    "repeat_band_high_percentile": 80.0,
    "repeat_min_transition_sec": 0.25,
    "torso_variation_low_max_speed_rad_s": 0.20,
    "torso_variation_high_min_speed_rad_s": 0.30,
}

HEAD_AXES = ("roll", "pitch", "yaw")
TORSO_AXES = ("yaw", "pitch", "roll")

UNSAFE_ID_PATTERN = re.compile(r"[^A-Za-z0-9_.:-]")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=None, help="Fallback FPS")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            records.append(value)
    return records


def _nested(record: dict[str, Any], *keys: str) -> Any:
    value: Any = record
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def task_id(record: dict[str, Any]) -> str:
    value = (
        record.get("task_id")
        or record.get("sample_id")
        or record.get("record_id")
        or record.get("clip_id")
    )
    if not isinstance(value, str) or not value or UNSAFE_ID_PATTERN.search(value):
        raise ValueError(
            "record requires a filesystem-safe task_id/sample_id/record_id/clip_id"
        )
    return value


def source_clip_id(record: dict[str, Any]) -> str | None:
    for value in (record.get("source_clip_id"), record.get("clip_id")):
        if isinstance(value, str) and value:
            return value
    source_relpath = _nested(record, "source_motion", "relpath")
    if isinstance(source_relpath, str) and source_relpath:
        return Path(source_relpath).stem
    return None


def trajectory_reference(record: dict[str, Any]) -> str:
    candidates = (
        _nested(record, "retarget", "safe_csv"),
        record.get("trajectory_path"),
        _nested(record, "trajectory", "path"),
        record.get("safe_csv"),
        record.get("safe_csv_path"),
        record.get("motion_path"),
    )
    for value in candidates:
        if isinstance(value, str) and value:
            return value
    raise ValueError("record does not contain an 18D safe CSV path")


def resolve_trajectory_path(record: dict[str, Any], manifest: Path) -> Path:
    value = Path(trajectory_reference(record))
    if value.is_absolute():
        return value
    return (manifest.parent / value).resolve()


def resolve_manifest_path(value: Any, manifest: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"passed record is missing {field}")
    path = Path(value)
    return path if path.is_absolute() else (manifest.parent / path).resolve()


def record_fps(record: dict[str, Any], fallback: float | None) -> float:
    candidates = (
        record.get("fps"),
        _nested(record, "source_motion", "fps"),
        _nested(record, "trajectory", "fps"),
        fallback,
    )
    for value in candidates:
        if value is not None:
            fps = float(value)
            if np.isfinite(fps) and fps >= THRESHOLDS["minimum_fps"]:
                return fps
    raise ValueError("record is missing a valid FPS")


def speech_context(record: dict[str, Any]) -> str | None:
    for key in ("window_transcript_context", "source_speech_context", "speech_context"):
        value = record.get(key)
        if isinstance(value, str):
            return value
    return None


def source_warnings(record: dict[str, Any]) -> list[str]:
    value = record.get("source_warnings")
    if value is None:
        value = record.get("inventory_issues")
    if value is None:
        return []
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError("source_warnings must be a list of strings")
    return value


def validate_passed_record(
    record: dict[str, Any], manifest: Path, trajectory: Path
) -> dict[str, bool]:
    if record.get("status") != "passed":
        raise ValueError("status must be exactly 'passed'")
    gate = record.get("quality_gate")
    if not isinstance(gate, dict) or gate.get("passed") is not True:
        raise ValueError("quality_gate.passed must be true")
    strict_gates = {key: value for key, value in gate.items() if key != "passed"}
    if not strict_gates or not all(value is True for value in strict_gates.values()):
        raise ValueError("every strict quality gate must be boolean true")

    expected_csv_hash = record.get("safe_csv_sha256")
    if not isinstance(expected_csv_hash, str) or expected_csv_hash != sha256(trajectory):
        raise ValueError("safe_csv_sha256 is missing or does not match")
    quality_path = resolve_manifest_path(
        record.get("quality_json"), manifest, "quality_json"
    )
    if not quality_path.is_file():
        raise ValueError(f"quality_json does not exist: {quality_path}")
    expected_quality_hash = record.get("quality_json_sha256")
    if (
        not isinstance(expected_quality_hash, str)
        or expected_quality_hash != sha256(quality_path)
    ):
        raise ValueError("quality_json_sha256 is missing or does not match")
    try:
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid quality_json: {error}") from error
    if not isinstance(quality, dict):
        raise ValueError("quality_json must contain an object")
    if quality.get("quality_gate") != gate:
        raise ValueError("manifest quality_gate does not match quality_json")
    if quality.get("output_contract") != ROBOT_CONTRACT:
        raise ValueError("quality_json output_contract mismatch")
    if quality.get("action_dim") != len(JOINT_ORDER):
        raise ValueError("quality_json action_dim mismatch")
    if quality.get("joint_order") != JOINT_ORDER:
        raise ValueError("quality_json joint_order mismatch")
    return gate


def load_trajectory(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"empty trajectory CSV: {path}") from error
        if header != JOINT_ORDER:
            raise ValueError(f"unexpected 18D joint order: {header}")
        rows = list(reader)
    if len(rows) < int(THRESHOLDS["minimum_frames"]):
        raise ValueError(f"trajectory has fewer than {THRESHOLDS['minimum_frames']} frames")
    try:
        values = np.asarray(rows, dtype=np.float64)
    except ValueError as error:
        raise ValueError(f"trajectory contains non-numeric values: {path}") from error
    if values.shape != (len(rows), len(JOINT_ORDER)):
        raise ValueError(f"expected (T,18), found {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("trajectory contains NaN or infinite values")
    return values


def _smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) < window:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same")


def _edge_smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Smooth a 1D signal without introducing zero-valued edge motion."""
    if len(values) < window:
        return values.copy()
    before = window // 2
    after = window - 1 - before
    padded = np.pad(values, (before, after), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def _group_features(values: np.ndarray, fps: float, group: slice) -> dict[str, float]:
    q05, q95 = np.percentile(values[:, group], [5.0, 95.0], axis=0)
    excursion_rad = float(np.sqrt(np.mean(np.square(q95 - q05))))
    velocity = np.diff(values[:, group], axis=0) * fps
    speed = np.sqrt(np.mean(np.square(velocity), axis=1))
    return {
        "excursion_deg": float(np.degrees(excursion_rad)),
        "mean_speed_rad_s": float(np.mean(speed)),
        "p95_speed_rad_s": float(np.percentile(speed, 95.0)),
    }


def _group_speed(values: np.ndarray, fps: float, group: slice) -> np.ndarray:
    velocity = np.diff(values[:, group], axis=0) * fps
    return np.sqrt(np.mean(np.square(velocity), axis=1))


def _periodicity(speed: np.ndarray, fps: float) -> tuple[bool, float | None, float]:
    signal = _smooth(speed) - float(np.mean(speed))
    denominator = float(np.dot(signal, signal))
    min_lag = max(1, int(round(THRESHOLDS["period_min_sec"] * fps)))
    max_lag = min(len(signal) - 2, int(round(THRESHOLDS["period_max_sec"] * fps)))
    if denominator < 1e-10 or max_lag < min_lag:
        return False, None, 0.0
    correlations = np.asarray(
        [
            float(np.dot(signal[:-lag], signal[lag:]) / denominator)
            for lag in range(min_lag, max_lag + 1)
        ]
    )
    best_offset = int(np.argmax(correlations))
    peak = float(correlations[best_offset])
    lag = min_lag + best_offset
    enough_cycles = len(signal) / lag >= 2.25
    periodic = peak >= THRESHOLDS["periodic_autocorrelation_min"] and enough_cycles
    return periodic, round(lag / fps, 4) if periodic else None, peak


def _motion_level(excursion_deg: float, group: str) -> str:
    active = THRESHOLDS[f"{group}_active_excursion_deg"]
    clear = THRESHOLDS[f"{group}_clear_excursion_deg"]
    if excursion_deg < active:
        return "minimal"
    if excursion_deg < clear:
        return "subtle"
    return "clear"


def _repeat_sweep_count(values: np.ndarray, fps: float) -> int:
    """Count full-band alternating sweeps with hysteresis and time separation."""
    signal = _edge_smooth(values)
    low, high = np.percentile(
        signal,
        [
            THRESHOLDS["repeat_band_low_percentile"],
            THRESHOLDS["repeat_band_high_percentile"],
        ],
    )
    if float(high - low) < 1e-10:
        return 0
    minimum_separation = max(
        1, int(round(THRESHOLDS["repeat_min_transition_sec"] * fps))
    )
    previous_state: int | None = None
    previous_transition = -minimum_separation
    sweeps = 0
    for frame, value in enumerate(signal):
        state = -1 if value <= low else 1 if value >= high else 0
        if state == 0:
            continue
        if previous_state is None:
            previous_state = state
            previous_transition = frame
        elif (
            state != previous_state
            and frame - previous_transition >= minimum_separation
        ):
            sweeps += 1
            previous_state = state
            previous_transition = frame
    return sweeps


def _axis_motion_features(
    values: np.ndarray,
    fps: float,
    group: slice,
    axis_names: tuple[str, ...],
    active_excursion_deg: float,
) -> dict[str, Any]:
    group_values = values[:, group]
    q05, q95 = np.percentile(group_values, [5.0, 95.0], axis=0)
    excursion_deg = np.degrees(q95 - q05)
    velocity = np.diff(group_values, axis=0) * fps
    mean_abs_speed = np.mean(np.abs(velocity), axis=0)
    sweep_counts = [
        _repeat_sweep_count(group_values[:, index], fps)
        for index in range(group_values.shape[1])
    ]

    order = np.argsort(excursion_deg)
    dominant_index = int(order[-1])
    second_excursion = (
        float(excursion_deg[order[-2]]) if len(order) > 1 else 0.0
    )
    dominant_excursion = float(excursion_deg[dominant_index])
    dominance_ratio = dominant_excursion / max(second_excursion, 1e-12)
    if dominant_excursion < active_excursion_deg:
        dominant_axis = "none"
    elif dominance_ratio >= THRESHOLDS["axis_dominance_min_ratio"]:
        dominant_axis = axis_names[dominant_index]
    else:
        dominant_axis = "mixed"

    return {
        "dominant_axis": dominant_axis,
        "dominant_axis_excursion_deg": dominant_excursion,
        "dominance_ratio": float(dominance_ratio),
        "per_axis": {
            axis: {
                "excursion_deg": float(excursion_deg[index]),
                "mean_abs_speed_rad_s": float(mean_abs_speed[index]),
                "full_band_sweep_count": int(sweep_counts[index]),
            }
            for index, axis in enumerate(axis_names)
        },
    }


def _head_repeated_pattern(axis_motion: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    for axis in HEAD_AXES:
        metrics = axis_motion["per_axis"][axis]
        if (
            metrics["excursion_deg"]
            >= THRESHOLDS["head_repeat_min_excursion_deg"]
            and metrics["full_band_sweep_count"]
            >= THRESHOLDS["head_repeat_min_sweeps"]
        ):
            candidates.append(axis)
    if not candidates:
        return {"axis": "none", "pattern": "none", "sweep_count": 0}

    dominant_axis = axis_motion["dominant_axis"]
    selected = (
        dominant_axis
        if dominant_axis in candidates
        else max(
            candidates,
            key=lambda axis: (
                axis_motion["per_axis"][axis]["excursion_deg"],
                axis_motion["per_axis"][axis]["full_band_sweep_count"],
            ),
        )
    )
    patterns = {
        "yaw": "repeated_yaw_turns",
        "pitch": "repeated_pitch_nods",
        "roll": "repeated_roll_tilts",
    }
    return {
        "axis": selected,
        "pattern": patterns[selected],
        "sweep_count": axis_motion["per_axis"][selected]["full_band_sweep_count"],
    }


def _overall_motion_features(values: np.ndarray, fps: float) -> dict[str, Any]:
    q05, q95 = np.percentile(values, [5.0, 95.0], axis=0)
    robust_excursion = q95 - q05
    velocity = np.diff(values, axis=0) * fps
    mean_abs_speed = np.mean(np.abs(velocity), axis=0)
    active = np.logical_and(
        robust_excursion
        >= np.radians(THRESHOLDS["overall_active_joint_excursion_deg"]),
        mean_abs_speed >= THRESHOLDS["overall_active_joint_speed_rad_s"],
    )
    if np.any(active):
        # For a sinusoid this amplitude-normalized value approximates cycles/sec.
        normalized_rates = mean_abs_speed[active] / (
            2.0 * robust_excursion[active]
        )
        normalized_change_rate = float(np.median(normalized_rates))
        if normalized_change_rate < THRESHOLDS["overall_pace_slow_max_hz"]:
            pace = "slow"
        elif normalized_change_rate >= THRESHOLDS["overall_pace_quick_min_hz"]:
            pace = "quick"
        else:
            pace = "steady"
    else:
        normalized_change_rate = 0.0
        pace = "minimal"
    return {
        "pace": pace,
        "normalized_change_rate_hz": normalized_change_rate,
        "active_joint_count": int(np.count_nonzero(active)),
        "rate_measure": (
            "median_active_joint_mean_abs_velocity_over_twice_robust_excursion"
        ),
    }


def _torso_variation_intensity(motion_level: str, mean_speed: float) -> str:
    if motion_level == "minimal":
        return "minimal"
    if mean_speed < THRESHOLDS["torso_variation_low_max_speed_rad_s"]:
        return "low"
    if mean_speed >= THRESHOLDS["torso_variation_high_min_speed_rad_s"]:
        return "high"
    return "medium"


def extract_features(values: np.ndarray, fps: float) -> dict[str, Any]:
    groups = {
        name: _group_features(values, fps, group) for name, group in GROUPS.items()
    }
    left_speed = _group_speed(values, fps, GROUPS["left_arm"])
    right_speed = _group_speed(values, fps, GROUPS["right_arm"])
    combined_speed = np.sqrt((np.square(left_speed) + np.square(right_speed)) * 0.5)
    left_energy = groups["left_arm"]["mean_speed_rad_s"]
    right_energy = groups["right_arm"]["mean_speed_rad_s"]
    dominant_energy = max(left_energy, right_energy)
    side_threshold = max(
        THRESHOLDS["static_arm_speed_rad_s"],
        dominant_energy * THRESHOLDS["active_side_fraction_of_dominant"],
    )
    active_sides = [
        side
        for side, energy in (("left", left_energy), ("right", right_energy))
        if energy >= side_threshold
    ]
    laterality = (
        "none"
        if not active_sides
        else "both"
        if len(active_sides) == 2
        else active_sides[0]
    )

    if np.std(left_speed) > 1e-8 and np.std(right_speed) > 1e-8:
        correlation = float(np.corrcoef(left_speed, right_speed)[0, 1])
    else:
        correlation = 0.0
    energy_ratio = min(left_energy, right_energy) / max(dominant_energy, 1e-12)
    if laterality == "none":
        energy_dominance = "none"
    elif laterality == "both" and energy_ratio >= THRESHOLDS[
        "arm_energy_balanced_min_ratio"
    ]:
        energy_dominance = "balanced"
    else:
        energy_dominance = "left" if left_energy > right_energy else "right"
    bilateral_coordination = bool(
        laterality == "both"
        and correlation >= THRESHOLDS["coordinated_speed_correlation"]
        and energy_ratio >= THRESHOLDS["coordinated_energy_ratio_min"]
    )
    activity_fraction = float(
        np.mean(combined_speed >= THRESHOLDS["motion_activity_speed_rad_s"])
    )
    if activity_fraction < THRESHOLDS["intermittent_activity_fraction"]:
        continuity = "intermittent"
    elif activity_fraction >= THRESHOLDS["continuous_activity_fraction"]:
        continuity = "continuous"
    else:
        continuity = "mixed"
    periodic, period_sec, periodicity_score = _periodicity(combined_speed, fps)
    arm_excursion = max(
        groups["left_arm"]["excursion_deg"], groups["right_arm"]["excursion_deg"]
    )
    if arm_excursion < THRESHOLDS["subtle_arm_excursion_deg"]:
        amplitude = "very_small"
    elif arm_excursion < THRESHOLDS["small_arm_excursion_deg"]:
        amplitude = "small"
    elif arm_excursion < THRESHOLDS["large_arm_excursion_deg"]:
        amplitude = "moderate"
    else:
        amplitude = "large"

    head_motion = _motion_level(groups["head"]["excursion_deg"], "head")
    torso_motion = _motion_level(groups["torso"]["excursion_deg"], "torso")
    head_axis_motion = _axis_motion_features(
        values,
        fps,
        GROUPS["head"],
        HEAD_AXES,
        THRESHOLDS["head_active_excursion_deg"],
    )
    torso_axis_motion = _axis_motion_features(
        values,
        fps,
        GROUPS["torso"],
        TORSO_AXES,
        THRESHOLDS["torso_active_excursion_deg"],
    )

    return {
        "fps": fps,
        "frames": len(values),
        "duration_sec": round(max(0, len(values) - 1) / fps, 6),
        "sample_span_sec": round(max(0, len(values) - 1) / fps, 6),
        "frame_coverage_sec": round(len(values) / fps, 6),
        "duration_time_axis": "sample_span=(frame_count-1)/fps",
        "groups": groups,
        "arm": {
            "amplitude": amplitude,
            "laterality": laterality,
            "left_to_right_energy_ratio": float(left_energy / max(right_energy, 1e-12)),
            "energy_dominance": energy_dominance,
            "bilateral_speed_correlation": correlation,
            "bilateral_energy_balance": energy_ratio,
            "bilateral_temporally_coordinated": bilateral_coordination,
            "activity_fraction": activity_fraction,
            "continuity": continuity,
            "regularly_repeated": periodic,
            "estimated_period_sec": period_sec,
            "periodicity_score": periodicity_score,
        },
        "overall_motion": _overall_motion_features(values, fps),
        "head_motion": head_motion,
        "head": {
            "motion_level": head_motion,
            "axis_motion": head_axis_motion,
            "repeated_pattern": _head_repeated_pattern(head_axis_motion),
        },
        "torso_motion": torso_motion,
        "torso": {
            "motion_level": torso_motion,
            "axis_motion": torso_axis_motion,
            "variation_intensity": _torso_variation_intensity(
                torso_motion, groups["torso"]["mean_speed_rad_s"]
            ),
        },
    }


AMPLITUDE_TEXT = {
    "very_small": ("very small", "很小幅度的"),
    "small": ("small", "小幅度的"),
    "moderate": ("moderate", "中等幅度的"),
    "large": ("broad", "较大范围的"),
}

PACE_TEXT = {
    "slow": (
        "keep an unhurried overall movement pace",
        "保持较慢的整体动作节奏",
    ),
    "steady": (
        "keep a steady overall movement pace",
        "保持平稳的整体动作节奏",
    ),
    "quick": ("use a quick overall movement pace", "采用较快的整体动作节奏"),
}

HEAD_REPEAT_TEXT = {
    "repeated_yaw_turns": (
        "repeated yaw-axis head turns",
        "反复的偏航轴转头运动",
    ),
    "repeated_pitch_nods": (
        "repeated pitch-axis nodding motions",
        "反复的俯仰轴点头运动",
    ),
    "repeated_roll_tilts": (
        "repeated roll-axis side tilts",
        "反复的滚转轴侧倾运动",
    ),
}

VARIATION_TEXT = {
    "low": ("low", "较低"),
    "medium": ("medium", "中等"),
    "high": ("high", "较高"),
}


def _axis_phrases(axis: str, subject: str) -> tuple[str, str]:
    subject_zh = {"head": "头部", "torso": "躯干"}[subject]
    if axis == "mixed":
        return (f"across multiple {subject} axes", f"跨多个{subject_zh}关节轴")
    if axis == "none":
        return ("", "")
    axis_zh = {"yaw": "偏航", "pitch": "俯仰", "roll": "滚转"}[axis]
    return (f"mainly around the {axis} axis", f"以{axis_zh}轴为主")


def prompt_from_features(features: dict[str, Any]) -> dict[str, str]:
    arm = features["arm"]
    amplitude_en, amplitude_zh = AMPLITUDE_TEXT[arm["amplitude"]]
    laterality = arm["laterality"]
    if laterality == "none":
        en = "Keep both arms nearly still"
        zh = "双臂基本保持不动"
    elif laterality == "left":
        en = f"Make {amplitude_en} observable movements with the left arm"
        zh = f"用左臂做{amplitude_zh}可观察运动"
    elif laterality == "right":
        en = f"Make {amplitude_en} observable movements with the right arm"
        zh = f"用右臂做{amplitude_zh}可观察运动"
    elif arm["bilateral_temporally_coordinated"]:
        en = f"Move both arms together with {amplitude_en} observable movements"
        zh = f"双臂同步做{amplitude_zh}可观察运动"
    else:
        en = f"Use both arms for {amplitude_en} observable movements"
        zh = f"用双臂做{amplitude_zh}可观察运动"

    modifiers_en: list[str] = []
    modifiers_zh: list[str] = []
    energy_dominance = arm["energy_dominance"]
    if laterality == "both":
        if energy_dominance == "balanced":
            modifiers_en.append("keep motion energy similar across the two arms")
            modifiers_zh.append("保持双臂运动能量接近")
        elif energy_dominance in {"left", "right"}:
            side_zh = "左臂" if energy_dominance == "left" else "右臂"
            modifiers_en.append(
                f"place more motion energy in the {energy_dominance} arm"
            )
            modifiers_zh.append(f"让{side_zh}承担更多运动能量")

    if laterality != "none" and arm["regularly_repeated"]:
        modifiers_en.append("repeat the arm motion at a regular pace")
        modifiers_zh.append("有规律地重复手臂运动")
    elif laterality != "none" and arm["continuity"] == "continuous":
        modifiers_en.append("keep the arm motion continuous")
        modifiers_zh.append("保持手臂运动连续")
    elif laterality != "none" and arm["continuity"] == "intermittent":
        modifiers_en.append("leave pauses between short arm movements")
        modifiers_zh.append("在短促的手臂运动之间留出停顿")

    pace = features["overall_motion"]["pace"]
    if pace in PACE_TEXT:
        pace_en, pace_zh = PACE_TEXT[pace]
        modifiers_en.append(pace_en)
        modifiers_zh.append(pace_zh)

    head_level = features["head_motion"]
    if head_level in {"subtle", "clear"}:
        level_en = "subtle" if head_level == "subtle" else "clear"
        level_zh = "轻微" if head_level == "subtle" else "明显"
        axis_en, axis_zh = _axis_phrases(
            features["head"]["axis_motion"]["dominant_axis"], "head"
        )
        head_en = f"add {level_en} head motion"
        head_zh = f"加入{level_zh}的头部运动"
        if axis_en:
            head_en += f" {axis_en}"
            head_zh += f"，{axis_zh}"
        repeat_pattern = features["head"]["repeated_pattern"]["pattern"]
        if repeat_pattern in HEAD_REPEAT_TEXT:
            repeat_en, repeat_zh = HEAD_REPEAT_TEXT[repeat_pattern]
            head_en += f", including {repeat_en}"
            head_zh += f"，其中包含{repeat_zh}"
        modifiers_en.append(head_en)
        modifiers_zh.append(head_zh)

    torso_level = features["torso_motion"]
    if torso_level in {"subtle", "clear"}:
        level_en = "subtle" if torso_level == "subtle" else "clear"
        level_zh = "轻微" if torso_level == "subtle" else "明显"
        axis_en, axis_zh = _axis_phrases(
            features["torso"]["axis_motion"]["dominant_axis"], "torso"
        )
        variation = features["torso"]["variation_intensity"]
        variation_en, variation_zh = VARIATION_TEXT[variation]
        torso_en = f"add {level_en} torso motion"
        torso_zh = f"加入{level_zh}的躯干运动"
        if axis_en:
            torso_en += f" {axis_en}"
            torso_zh += f"，{axis_zh}"
        torso_en += f" with {variation_en} variation intensity"
        torso_zh += f"，变化强度{variation_zh}"
        modifiers_en.append(torso_en)
        modifiers_zh.append(torso_zh)

    if modifiers_en:
        en += ", and " + "; ".join(modifiers_en)
        zh += "，并" + "；".join(modifiers_zh)
    return {"en": en + ".", "zh": zh + "。"}


def _fingerprint(path: Path, fps: float, source: dict[str, Any]) -> str:
    payload = {
        "algorithm_version": ALGORITHM_VERSION,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "robot_contract": ROBOT_CONTRACT,
        "joint_order": JOINT_ORDER,
        "fps": fps,
        "safe_csv_sha256": sha256(path),
        "speech_context": speech_context(source),
        "source_clip_id": source_clip_id(source),
        "official_split": source.get("official_split"),
        "speaker_key": source.get("speaker_key"),
        "source_warnings": source_warnings(source),
        "thresholds": THRESHOLDS,
    }
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def _existing_by_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    return {record["task_id"]: record for record in read_jsonl(path)}


def _prune_stale_text(directory: Path, expected_names: set[str]) -> None:
    if not directory.is_dir():
        return
    for path in directory.glob("*.txt"):
        if path.name not in expected_names:
            path.unlink()


def build_labels(
    input_manifest: Path,
    output_dir: Path,
    *,
    fallback_fps: float | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    records = read_jsonl(input_manifest)
    existing = _existing_by_id(output_dir / "draft_prompts.jsonl") if resume else {}
    drafts: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    candidate_ids: list[str | None] = []
    for source in records:
        try:
            candidate_ids.append(task_id(source))
        except ValueError:
            candidate_ids.append(None)
    duplicate_ids = {
        value
        for value, count in Counter(candidate_ids).items()
        if value is not None and count > 1
    }
    reused = 0

    for source_index, source in enumerate(records):
        try:
            item_id = task_id(source)
            if item_id in duplicate_ids:
                raise ValueError(f"duplicate task_id: {item_id}")
            trajectory = resolve_trajectory_path(source, input_manifest)
            validate_passed_record(source, input_manifest, trajectory)
            fps = record_fps(source, fallback_fps)
            fingerprint = _fingerprint(trajectory, fps, source)
            previous = existing.get(item_id)
            if previous and previous.get("input_fingerprint") == fingerprint:
                draft = dict(previous)
                draft["source_record_index"] = source_index
                draft["source_manifest"] = str(input_manifest.resolve())
                reused += 1
            else:
                values = load_trajectory(trajectory)
                features = extract_features(values, fps)
                prompt = prompt_from_features(features)
                near_static = bool(
                    features["arm"]["laterality"] == "none"
                    and features["head_motion"] == "minimal"
                    and features["torso_motion"] == "minimal"
                )
                draft = {
                    "schema_version": SCHEMA_VERSION,
                    "algorithm_version": ALGORITHM_VERSION,
                    "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                    "task_id": item_id,
                    "source_clip_id": source_clip_id(source),
                    "source_record_index": source_index,
                    "source_manifest": str(input_manifest.resolve()),
                    "robot_contract": ROBOT_CONTRACT,
                    "fps": fps,
                    "trajectory_path": str(trajectory),
                    "trajectory_sha256": sha256(trajectory),
                    "quality_json": str(
                        resolve_manifest_path(
                            source.get("quality_json"),
                            input_manifest,
                            "quality_json",
                        )
                    ),
                    "quality_json_sha256": source["quality_json_sha256"],
                    "retarget_quality_gate": source["quality_gate"],
                    "input_fingerprint": fingerprint,
                    "prompt_provenance": PROMPT_PROVENANCE,
                    "labeling_thresholds": THRESHOLDS,
                    "canonical_action": "robot_observable_upper_body_motion",
                    "canonical_prompt": prompt,
                    "observable_features": features,
                    "semantic_confidence": "low" if near_static else "medium",
                    "review_flags": (
                        ["near_static_observable_state"] if near_static else []
                    ),
                    "speech_context": speech_context(source),
                    "speech_context_role": SPEECH_CONTEXT_ROLE,
                    "source_speech_context": speech_context(source),
                    "source_speech_context_role": SPEECH_CONTEXT_ROLE,
                    "official_split": source.get("official_split"),
                    "speaker_key": source.get("speaker_key"),
                    "source_warnings": source_warnings(source),
                    "accepted_for_training": False,
                    "decision": "needs_human_review",
                    "manual_review_required": True,
                    "manual_human_review_required": True,
                    "review_state": (
                        "algorithmic_motion_description_draft_pending_human_review"
                    ),
                }
                segment = source.get("retarget_segment")
                if isinstance(segment, dict):
                    draft["retarget_segment"] = dict(segment)
                    start = segment.get("source_start_frame")
                    end = segment.get("source_end_frame_exclusive")
                    if isinstance(start, int) and not isinstance(start, bool):
                        draft["source_window_start_frame"] = start
                    if isinstance(end, int) and not isinstance(end, bool):
                        draft["source_window_end_frame_exclusive"] = end
            drafts.append(draft)
        except (FileNotFoundError, OSError, ValueError, KeyError, TypeError) as error:
            fallback_id = (
                source.get("task_id")
                or source.get("sample_id")
                or source.get("record_id")
                or source.get("clip_id")
            )
            rejected.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "algorithm_version": ALGORITHM_VERSION,
                    "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                    "task_id": fallback_id,
                    "source_clip_id": source_clip_id(source),
                    "source_record_index": source_index,
                    "accepted_for_training": False,
                    "decision": "rejected",
                    "manual_review_required": False,
                    "manual_human_review_required": False,
                    "review_state": "rejected_by_automatic_validation",
                    "rejection_reason": str(error),
                }
            )

    drafts.sort(key=lambda item: item["task_id"])
    rejected.sort(
        key=lambda item: (str(item.get("task_id")), item["source_record_index"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_text_names = {f"{draft['task_id']}.txt" for draft in drafts}
    _prune_stale_text(output_dir / "text/en", expected_text_names)
    _prune_stale_text(output_dir / "text/zh", expected_text_names)
    for draft in drafts:
        item_id = draft["task_id"]
        atomic_write_text(
            output_dir / "text/en" / f"{item_id}.txt",
            draft["canonical_prompt"]["en"] + "\n",
        )
        atomic_write_text(
            output_dir / "text/zh" / f"{item_id}.txt",
            draft["canonical_prompt"]["zh"] + "\n",
        )

    draft_payload = "".join(stable_json(item) + "\n" for item in drafts)
    rejected_payload = "".join(stable_json(item) + "\n" for item in rejected)
    atomic_write_text(output_dir / "draft_prompts.jsonl", draft_payload)
    atomic_write_text(output_dir / "needs_human_review.jsonl", draft_payload)
    atomic_write_text(output_dir / "rejected.jsonl", rejected_payload)
    feature_distributions = {
        "overall_pace": dict(
            sorted(
                Counter(
                    item["observable_features"]["overall_motion"]["pace"]
                    for item in drafts
                ).items()
            )
        ),
        "arm_laterality": dict(
            sorted(
                Counter(
                    item["observable_features"]["arm"]["laterality"]
                    for item in drafts
                ).items()
            )
        ),
        "arm_energy_dominance": dict(
            sorted(
                Counter(
                    item["observable_features"]["arm"]["energy_dominance"]
                    for item in drafts
                ).items()
            )
        ),
        "head_dominant_axis": dict(
            sorted(
                Counter(
                    item["observable_features"]["head"]["axis_motion"][
                        "dominant_axis"
                    ]
                    for item in drafts
                ).items()
            )
        ),
        "head_repeated_pattern": dict(
            sorted(
                Counter(
                    item["observable_features"]["head"]["repeated_pattern"][
                        "pattern"
                    ]
                    for item in drafts
                ).items()
            )
        ),
        "torso_dominant_axis": dict(
            sorted(
                Counter(
                    item["observable_features"]["torso"]["axis_motion"][
                        "dominant_axis"
                    ]
                    for item in drafts
                ).items()
            )
        ),
        "torso_variation_intensity": dict(
            sorted(
                Counter(
                    item["observable_features"]["torso"][
                        "variation_intensity"
                    ]
                    for item in drafts
                ).items()
            )
        ),
    }
    unique_english_prompts = len(
        {item["canonical_prompt"]["en"] for item in drafts}
    )
    unique_chinese_prompts = len(
        {item["canonical_prompt"]["zh"] for item in drafts}
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "robot_contract": ROBOT_CONTRACT,
        "input_manifest": str(input_manifest.resolve()),
        "input_records": len(records),
        "draft_records": len(drafts),
        "needs_human_review_records": len(drafts),
        "rejected_records": len(rejected),
        "accepted_for_training_records": 0,
        "resume_reused_records": reused,
        "status_counts": {
            "draft_pending_human_review": len(drafts),
            "rejected": len(rejected),
            "accepted_for_training": 0,
        },
        "rejection_reason_counts": dict(
            sorted(Counter(item["rejection_reason"] for item in rejected).items())
        ),
        "label_diversity": {
            "unique_english_prompts": unique_english_prompts,
            "unique_chinese_prompts": unique_chinese_prompts,
            "duplicate_english_prompt_records": len(drafts)
            - unique_english_prompts,
            "duplicate_chinese_prompt_records": len(drafts)
            - unique_chinese_prompts,
        },
        "observable_feature_counts": feature_distributions,
        "prompt_provenance": PROMPT_PROVENANCE,
        "speech_context_role": SPEECH_CONTEXT_ROLE,
        "thresholds": THRESHOLDS,
        "outputs": {
            "draft_prompts": str((output_dir / "draft_prompts.jsonl").resolve()),
            "needs_human_review": str(
                (output_dir / "needs_human_review.jsonl").resolve()
            ),
            "rejected": str((output_dir / "rejected.jsonl").resolve()),
            "english_text_dir": str((output_dir / "text/en").resolve()),
            "chinese_text_dir": str((output_dir / "text/zh").resolve()),
        },
    }
    atomic_write_text(
        output_dir / "summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_labels(
        args.input_manifest,
        args.output_dir,
        fallback_fps=args.fps,
        resume=args.resume,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
