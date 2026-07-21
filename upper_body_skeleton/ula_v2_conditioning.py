#!/usr/bin/env python3
"""Leakage-safe conditioning and trajectory preprocessing for ULA MMDiT V2."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from upper_body_skeleton.kimodo_semantics import KIMODO_BEHAVIOR_IDS, KIMODO_EMOTION_IDS
from upper_body_skeleton.motion_latent import (
    encode_motion_episodes,
    load_motion_latent_episodes,
    load_motion_metric_checkpoint,
    stratified_episode_split,
)
from upper_body_skeleton.retarget_v2 import JOINT_LIMITS, JOINT_ORDER


V2_PREPROCESS_CONTRACT_VERSION = 1
V2_ACTIVE_WINDOW_CONTRACT_VERSION = 2
V2_DURATION_CONTRACT_VERSION = 1
V2_STYLE_BANK_CONTRACT_VERSION = 1
V2_STYLE_CONTRACT_VERSION = 1
V2_PROTOTYPE_CONTRACT_VERSION = 1
V2_CONDITION_CONTRACT_VERSION = 1
STYLE_FEATURE_NAMES = ("signed_arm_balance", "log_arm_amplitude", "log_arm_speed")
_SPLIT_NAMES = ("train", "validation", "test")
_LEFT_ARM_INDICES = np.asarray(
    [index for index, name in enumerate(JOINT_ORDER) if name.startswith("joint_l")], dtype=np.int64
)
_RIGHT_ARM_INDICES = np.asarray(
    [index for index, name in enumerate(JOINT_ORDER) if name.startswith("joint_r")], dtype=np.int64
)
_ARM_INDICES = np.concatenate([_LEFT_ARM_INDICES, _RIGHT_ARM_INDICES])


def _condition_dimensions():
    # Imported lazily because ula_training imports this module for V2 training.
    from upper_body_skeleton.ula_training import (
        KIMODO_CONDITION_DIM,
        KIMODO_MOTION_LATENT_DIM,
        KIMODO_V2_CONDITION_DIM,
    )

    return int(KIMODO_CONDITION_DIM), int(KIMODO_MOTION_LATENT_DIM), int(KIMODO_V2_CONDITION_DIM)


def _canonical_json(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _contract_with_hash(payload):
    result = dict(payload)
    result["sha256"] = hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()
    return result


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_id(episode):
    try:
        value = episode["episode_index"]
    except (KeyError, TypeError) as exc:
        raise ValueError("each episode must contain episode_index") from exc
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("episode_index must be an integer, not bool")
    integer = int(value)
    if integer != value:
        raise ValueError(f"episode_index must be an integer: {value!r}")
    return integer


def _semantic_key(episode):
    meta = episode.get("meta") or {}
    behavior_id = str(meta.get("behavior_id") or "")
    emotion_id = str(meta.get("emotion_id") or "")
    if behavior_id not in KIMODO_BEHAVIOR_IDS:
        raise ValueError(f"unknown Kimodo behavior_id for episode {_episode_id(episode)}: {behavior_id!r}")
    if emotion_id not in KIMODO_EMOTION_IDS:
        raise ValueError(f"unknown Kimodo emotion_id for episode {_episode_id(episode)}: {emotion_id!r}")
    return behavior_id, emotion_id


def _normalized_split_ids(raw_manifest):
    if not isinstance(raw_manifest, dict) or set(raw_manifest) != set(_SPLIT_NAMES):
        raise ValueError("motion checkpoint must contain exactly train/validation/test episode partitions")
    normalized = {}
    all_ids = []
    for split_name in _SPLIT_NAMES:
        values = raw_manifest[split_name]
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"motion checkpoint split {split_name!r} must be a non-empty sequence")
        ids = []
        for value in values:
            if isinstance(value, (bool, np.bool_)):
                raise ValueError("motion checkpoint episode IDs must be integers, not bool")
            integer = int(value)
            if integer != value:
                raise ValueError(f"motion checkpoint episode ID must be an integer: {value!r}")
            ids.append(integer)
        if len(ids) != len(set(ids)):
            raise ValueError(f"motion checkpoint split {split_name!r} contains duplicate episode IDs")
        normalized[split_name] = sorted(ids)
        all_ids.extend(ids)
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("motion checkpoint train/validation/test episode partitions overlap")
    return normalized


def load_validated_episode_splits(split_checkpoint, episodes):
    """Load a Motion Metric split and prove that it exactly partitions ``episodes``."""
    split_checkpoint = Path(split_checkpoint)
    checkpoint = torch.load(split_checkpoint, map_location="cpu", weights_only=True)
    split_ids = _normalized_split_ids(checkpoint.get("split_episode_indices"))

    episodes = list(episodes)
    if not episodes:
        raise ValueError("cannot validate a split against an empty episode collection")
    episode_ids = [_episode_id(episode) for episode in episodes]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("dataset episodes contain duplicate episode_index values")
    partition_ids = {value for values in split_ids.values() for value in values}
    if partition_ids != set(episode_ids):
        missing = sorted(set(episode_ids) - partition_ids)
        unexpected = sorted(partition_ids - set(episode_ids))
        raise ValueError(
            "motion checkpoint partitions do not cover the dataset exactly "
            f"(missing={missing[:8]}, unexpected={unexpected[:8]})"
        )

    if list(checkpoint.get("behavior_ids") or []) != list(KIMODO_BEHAVIOR_IDS):
        raise ValueError("motion checkpoint behavior_ids do not match the Kimodo schema")
    if list(checkpoint.get("emotion_ids") or []) != list(KIMODO_EMOTION_IDS):
        raise ValueError("motion checkpoint emotion_ids do not match the Kimodo schema")
    config = checkpoint.get("config") or {}
    if int(config.get("action_dim", -1)) != len(JOINT_ORDER):
        raise ValueError("motion checkpoint action_dim does not match the upper-body joint schema")
    if "seed" not in config:
        raise ValueError("motion checkpoint does not record its stratified split seed")

    semantic_episodes = []
    episode_by_id = {}
    for episode in episodes:
        episode_index = _episode_id(episode)
        behavior_id, emotion_id = _semantic_key(episode)
        semantic_episodes.append(
            {
                "episode_index": episode_index,
                "meta": {"behavior_id": behavior_id, "emotion_id": emotion_id},
            }
        )
        episode_by_id[episode_index] = episode
    expected = stratified_episode_split(semantic_episodes, seed=int(config["seed"]))
    for split_name, expected_episodes in zip(_SPLIT_NAMES, expected):
        expected_ids = sorted(_episode_id(episode) for episode in expected_episodes)
        if split_ids[split_name] != expected_ids:
            raise ValueError(
                f"motion checkpoint {split_name} partition does not match dataset labels and recorded seed"
            )

    split_contract = _contract_with_hash(
        {
            "contract_type": "ula_v2_episode_split",
            "contract_version": 1,
            "source_checkpoint_sha256": _sha256_file(split_checkpoint),
            "seed": int(config["seed"]),
            "episode_count": len(episodes),
            "episode_indices": split_ids,
        }
    )
    return (
        {name: [episode_by_id[index] for index in split_ids[name]] for name in _SPLIT_NAMES},
        split_contract,
    )


def build_preprocess_contract(*, max_velocity_rad_s=3.0, smooth_window=1, default_fps=30.0):
    max_velocity_rad_s = float(max_velocity_rad_s)
    default_fps = float(default_fps)
    smooth_window = int(smooth_window)
    if not np.isfinite(max_velocity_rad_s) or max_velocity_rad_s <= 0:
        raise ValueError("max_velocity_rad_s must be finite and positive")
    if not np.isfinite(default_fps) or default_fps <= 0:
        raise ValueError("default_fps must be finite and positive")
    if smooth_window <= 0 or smooth_window % 2 == 0:
        raise ValueError("smooth_window must be a positive odd integer")
    return _contract_with_hash(
        {
            "contract_type": "ula_v2_trajectory_preprocess",
            "contract_version": V2_PREPROCESS_CONTRACT_VERSION,
            "joint_order": list(JOINT_ORDER),
            "joint_bounds_rad": {name: [float(value) for value in JOINT_LIMITS[name]] for name in JOINT_ORDER},
            "max_velocity_rad_s": max_velocity_rad_s,
            "smooth_window": smooth_window,
            "default_fps": default_fps,
            "operation_order": ["finite_check", "joint_clamp", "velocity_limit", "moving_average", "joint_clamp", "velocity_limit"],
        }
    )


def _limit_velocity(values, max_step):
    limited = np.array(values, dtype=np.float32, copy=True)
    for frame in range(1, limited.shape[0]):
        limited[frame] = np.clip(limited[frame], limited[frame - 1] - max_step, limited[frame - 1] + max_step)
    return limited


def _moving_average(values, window):
    if window == 1:
        return values
    radius = window // 2
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    cumulative = np.concatenate(
        [np.zeros((1, values.shape[1]), dtype=np.float64), np.cumsum(padded, axis=0, dtype=np.float64)], axis=0
    )
    return ((cumulative[window:] - cumulative[:-window]) / float(window)).astype(np.float32)


def clean_joint_trajectory(actions, *, fps=30.0, preprocess_contract=None, max_velocity_rad_s=3.0, smooth_window=1):
    """Clamp one trajectory to versioned position/velocity limits without importing MuJoCo."""
    if preprocess_contract is None:
        preprocess_contract = build_preprocess_contract(
            max_velocity_rad_s=max_velocity_rad_s,
            smooth_window=smooth_window,
            default_fps=fps,
        )
    if preprocess_contract.get("contract_version") != V2_PREPROCESS_CONTRACT_VERSION:
        raise ValueError("unsupported V2 preprocess contract version")
    if list(preprocess_contract.get("joint_order") or []) != JOINT_ORDER:
        raise ValueError("preprocess contract joint order does not match Kimodo")
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("trajectory fps must be finite and positive")
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(JOINT_ORDER) or values.shape[0] < 1:
        raise ValueError(f"actions must have shape [frames, {len(JOINT_ORDER)}]")
    if not np.isfinite(values).all():
        raise ValueError("actions contain non-finite joint values")

    lower = np.asarray([JOINT_LIMITS[name][0] for name in JOINT_ORDER], dtype=np.float32)
    upper = np.asarray([JOINT_LIMITS[name][1] for name in JOINT_ORDER], dtype=np.float32)
    max_step = float(preprocess_contract["max_velocity_rad_s"]) / fps
    cleaned = np.clip(values, lower, upper)
    cleaned = _limit_velocity(cleaned, max_step)
    cleaned = _moving_average(cleaned, int(preprocess_contract["smooth_window"]))
    cleaned = np.clip(cleaned, lower, upper)
    cleaned = _limit_velocity(cleaned, max_step)
    return np.ascontiguousarray(cleaned, dtype=np.float32)


def build_active_window_contract(
    *,
    min_speed_rad_s=0.08,
    noise_mad_scale=4.0,
    minimum_dynamic_cap_rad_s=0.35,
    activity_p95_ratio=0.50,
    smoothing_sec=0.10,
    min_active_sec=0.20,
    max_inactive_gap_sec=0.15,
    static_context_sec=0.25,
    margin_sec=0.10,
    min_trim_sec=0.25,
):
    values = {
        "min_speed_rad_s": float(min_speed_rad_s),
        "noise_mad_scale": float(noise_mad_scale),
        "minimum_dynamic_cap_rad_s": float(minimum_dynamic_cap_rad_s),
        "activity_p95_ratio": float(activity_p95_ratio),
        "smoothing_sec": float(smoothing_sec),
        "min_active_sec": float(min_active_sec),
        "max_inactive_gap_sec": float(max_inactive_gap_sec),
        "static_context_sec": float(static_context_sec),
        "margin_sec": float(margin_sec),
        "min_trim_sec": float(min_trim_sec),
    }
    if any(not np.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("active-window parameters must be finite and non-negative")
    if values["min_speed_rad_s"] <= 0 or values["minimum_dynamic_cap_rad_s"] <= 0:
        raise ValueError("active-window speed thresholds must be positive")
    if values["noise_mad_scale"] <= 0 or values["min_active_sec"] <= 0:
        raise ValueError("noise_mad_scale and min_active_sec must be positive")
    if values["minimum_dynamic_cap_rad_s"] < values["min_speed_rad_s"]:
        raise ValueError("minimum dynamic cap cannot be lower than min_speed_rad_s")
    if not 0 < values["activity_p95_ratio"] <= 1:
        raise ValueError("activity_p95_ratio must be in (0, 1]")
    return _contract_with_hash(
        {
            "contract_type": "ula_v2_active_motion_window",
            "contract_version": V2_ACTIVE_WINDOW_CONTRACT_VERSION,
            "speed_source": "cleaned_joint_positions",
            "joint_aggregation": "rms_of_top_3_absolute_joint_velocities",
            "adaptive_noise_estimator": "lower_half_median_absolute_deviation",
            "activity_hysteresis": "none; static context and margin preserve slow onset and return",
            "boundary_semantics": "zero_based_start_inclusive_end_exclusive",
            "near_static_policy": "preserve_full_trajectory",
            "full_span_policy": "preserve_full_trajectory",
            **values,
        }
    )


def _odd_frame_window(seconds, fps):
    frames = max(1, int(round(float(seconds) * float(fps))))
    return frames if frames % 2 else frames + 1


def _run_ranges(mask, value):
    ranges = []
    start = None
    for index, current in enumerate(mask):
        if bool(current) == bool(value) and start is None:
            start = index
        elif bool(current) != bool(value) and start is not None:
            ranges.append((start, index))
            start = None
    if start is not None:
        ranges.append((start, len(mask)))
    return ranges


def _fill_short_inactive_gaps(mask, max_gap_frames):
    result = np.asarray(mask, dtype=bool).copy()
    for start, end in _run_ranges(result, False):
        if start > 0 and end < len(result) and end - start <= max_gap_frames:
            result[start:end] = True
    return result


def _remove_short_active_runs(mask, min_active_frames):
    result = np.asarray(mask, dtype=bool).copy()
    for start, end in _run_ranges(result, True):
        if end - start < min_active_frames:
            result[start:end] = False
    return result


def extract_active_motion_window(actions, *, fps=30.0, active_window_contract=None):
    """Detect a robust active-motion span on an already cleaned trajectory."""
    if active_window_contract is None:
        active_window_contract = build_active_window_contract()
    if active_window_contract.get("contract_version") != V2_ACTIVE_WINDOW_CONTRACT_VERSION:
        raise ValueError("unsupported V2 active-window contract version")
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(JOINT_ORDER) or values.shape[0] < 1:
        raise ValueError(f"actions must have shape [frames, {len(JOINT_ORDER)}]")
    if not np.isfinite(values).all():
        raise ValueError("actions contain non-finite joint values")
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("trajectory fps must be finite and positive")
    frame_count = int(values.shape[0])
    if frame_count < 2:
        return {
            "active_start": 0,
            "active_end": frame_count,
            "trim_start": 0,
            "trim_end": frame_count,
            "near_static": True,
            "activity_threshold_rad_s": float(active_window_contract["min_speed_rad_s"]),
            "peak_activity_rad_s": 0.0,
        }

    absolute_velocity = np.abs(np.diff(values, axis=0)) * fps
    top_count = min(3, absolute_velocity.shape[1])
    top_velocity = np.partition(absolute_velocity, -top_count, axis=1)[:, -top_count:]
    transition_score = np.sqrt(np.mean(np.square(top_velocity), axis=1, dtype=np.float64))
    frame_score = np.empty(frame_count, dtype=np.float64)
    frame_score[0] = transition_score[0]
    frame_score[-1] = transition_score[-1]
    if frame_count > 2:
        frame_score[1:-1] = np.maximum(transition_score[:-1], transition_score[1:])
    smooth_window = _odd_frame_window(active_window_contract["smoothing_sec"], fps)
    smoothed_score = _moving_average(frame_score[:, None].astype(np.float32), smooth_window)[:, 0]

    lower_half = smoothed_score[smoothed_score <= np.median(smoothed_score)]
    noise_center = float(np.median(lower_half))
    noise_mad = float(np.median(np.abs(lower_half - noise_center)))
    adaptive_threshold = noise_center + float(active_window_contract["noise_mad_scale"]) * 1.4826 * noise_mad
    dynamic_cap = max(
        float(active_window_contract["minimum_dynamic_cap_rad_s"]),
        float(active_window_contract["activity_p95_ratio"]) * float(np.percentile(smoothed_score, 95)),
    )
    threshold = max(
        float(active_window_contract["min_speed_rad_s"]),
        min(adaptive_threshold, dynamic_cap),
    )
    active = smoothed_score >= threshold
    active = _fill_short_inactive_gaps(
        active, max(0, int(round(float(active_window_contract["max_inactive_gap_sec"]) * fps)))
    )
    min_active_frames = max(1, int(round(float(active_window_contract["min_active_sec"]) * fps)))
    active = _remove_short_active_runs(active, min_active_frames)
    active_indices = np.flatnonzero(active)
    peak_activity = float(np.max(smoothed_score))
    if active_indices.size == 0:
        return {
            "active_start": 0,
            "active_end": frame_count,
            "trim_start": 0,
            "trim_end": frame_count,
            "near_static": True,
            "activity_threshold_rad_s": threshold,
            "peak_activity_rad_s": peak_activity,
        }

    active_start = int(active_indices[0])
    active_end = int(active_indices[-1]) + 1
    boundary_padding = int(
        round(
            (float(active_window_contract["static_context_sec"]) + float(active_window_contract["margin_sec"]))
            * fps
        )
    )
    trim_start = max(0, active_start - boundary_padding)
    trim_end = min(frame_count, active_end + boundary_padding)
    removed_frames = trim_start + (frame_count - trim_end)
    min_trim_frames = int(round(float(active_window_contract["min_trim_sec"]) * fps))
    if removed_frames < min_trim_frames:
        trim_start, trim_end = 0, frame_count
    return {
        "active_start": active_start,
        "active_end": active_end,
        "trim_start": trim_start,
        "trim_end": trim_end,
        "near_static": False,
        "activity_threshold_rad_s": threshold,
        "peak_activity_rad_s": peak_activity,
    }


def trim_episode(episode, *, active_window_contract=None):
    """Trim one cleaned episode while retaining source-frame timing metadata."""
    item = dict(episode)
    item["meta"] = dict(episode.get("meta") or {})
    fps = float(episode.get("fps") or item["meta"].get("fps") or 30.0)
    values = np.asarray(episode["actions"], dtype=np.float32)
    window = extract_active_motion_window(
        values,
        fps=fps,
        active_window_contract=active_window_contract,
    )
    trim_start = int(window["trim_start"])
    trim_end = int(window["trim_end"])
    item["actions"] = np.ascontiguousarray(values[trim_start:trim_end], dtype=np.float32)
    item["fps"] = fps
    item["original_frame_count"] = int(values.shape[0])
    item["active_start"] = int(window["active_start"])
    item["active_end"] = int(window["active_end"])
    item["trim_start"] = trim_start
    item["trim_end"] = trim_end
    item["near_static"] = bool(window["near_static"])
    item["activity_threshold_rad_s"] = float(window["activity_threshold_rad_s"])
    item["peak_activity_rad_s"] = float(window["peak_activity_rad_s"])
    item["frame_count"] = int(item["actions"].shape[0])
    item["effective_duration_sec"] = float((item["active_end"] - item["active_start"]) / fps)
    item["duration_sec"] = float(item["frame_count"] / fps)
    item["duration_censored_start"] = bool(item["active_start"] == 0)
    item["duration_censored_end"] = bool(item["active_end"] == item["original_frame_count"])
    item["duration_supervision_valid"] = bool(
        not item["near_static"]
        and not item["duration_censored_start"]
        and not item["duration_censored_end"]
    )
    return item


def build_duration_contract(train_episodes, active_window_contract):
    train_episodes = list(train_episodes)
    if not train_episodes:
        raise ValueError("cannot build a V2 duration contract without train episodes")
    frame_counts = np.asarray([int(episode["actions"].shape[0]) for episode in train_episodes], dtype=np.int64)
    original_frame_counts = np.asarray(
        [int(episode["original_frame_count"]) for episode in train_episodes], dtype=np.int64
    )
    durations = np.asarray([float(episode["duration_sec"]) for episode in train_episodes], dtype=np.float64)
    valid_duration_episodes = [episode for episode in train_episodes if episode["duration_supervision_valid"]]
    valid_durations = np.asarray(
        [float(episode["duration_sec"]) for episode in valid_duration_episodes], dtype=np.float64
    )
    if np.any(frame_counts < 1) or not np.isfinite(durations).all() or np.any(durations <= 0):
        raise ValueError("V2 train episodes contain invalid durations")
    return _contract_with_hash(
        {
            "contract_type": "ula_v2_variable_duration",
            "contract_version": V2_DURATION_CONTRACT_VERSION,
            "fixed_frame_count": None,
            "trajectory_representation": "native_trimmed_frames",
            "batching_policy": "phase_normalized_multi_resolution_resampling",
            "duration_formula": "frame_count/fps",
            "duration_supervision_policy": "exclude_near_static_and_window_boundary_censored_episodes",
            "duration_supervision_episode_count": len(valid_duration_episodes),
            "duration_supervision_sec": None
            if valid_durations.size == 0
            else {
                "min": float(valid_durations.min()),
                "median": float(np.median(valid_durations)),
                "max": float(valid_durations.max()),
            },
            "active_window_contract_sha256": active_window_contract["sha256"],
            "statistics_source": "train",
            "train_episode_count": len(train_episodes),
            "train_frame_count": {
                "min": int(frame_counts.min()),
                "median": float(np.median(frame_counts)),
                "max": int(frame_counts.max()),
            },
            "train_original_frame_count": {
                "min": int(original_frame_counts.min()),
                "median": float(np.median(original_frame_counts)),
                "max": int(original_frame_counts.max()),
            },
            "train_duration_sec": {
                "min": float(durations.min()),
                "median": float(np.median(durations)),
                "max": float(durations.max()),
            },
        }
    )


def extract_style_features(actions, *, fps=30.0, eps=1e-8):
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(JOINT_ORDER) or values.shape[0] < 1:
        raise ValueError(f"actions must have shape [frames, {len(JOINT_ORDER)}]")
    if not np.isfinite(values).all():
        raise ValueError("actions contain non-finite joint values")
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("trajectory fps must be finite and positive")

    centered = values - values.mean(axis=0, keepdims=True)
    joint_amplitude = np.sqrt(np.mean(np.square(centered), axis=0, dtype=np.float64))
    left_activity = float(np.sqrt(np.mean(np.square(joint_amplitude[_LEFT_ARM_INDICES]), dtype=np.float64)))
    right_activity = float(np.sqrt(np.mean(np.square(joint_amplitude[_RIGHT_ARM_INDICES]), dtype=np.float64)))
    signed_balance = (right_activity - left_activity) / (right_activity + left_activity + float(eps))
    arm_amplitude = float(np.sqrt(np.mean(np.square(centered[:, _ARM_INDICES]), dtype=np.float64)))
    if values.shape[0] > 1:
        arm_velocity = np.diff(values[:, _ARM_INDICES], axis=0) * fps
        arm_speed = float(np.sqrt(np.mean(np.square(arm_velocity), dtype=np.float64)))
    else:
        arm_speed = 0.0
    features = np.asarray([signed_balance, np.log1p(arm_amplitude), np.log1p(arm_speed)], dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError("computed non-finite V2 style features")
    return features


def fit_style_contract(train_episodes, *, clip=5.0, eps=1e-4):
    train_episodes = list(train_episodes)
    if not train_episodes:
        raise ValueError("cannot fit V2 style statistics without train episodes")
    clip = float(clip)
    eps = float(eps)
    if not np.isfinite(clip) or clip <= 0 or not np.isfinite(eps) or eps <= 0:
        raise ValueError("style clip and eps must be finite and positive")
    features = np.stack(
        [extract_style_features(episode["actions"], fps=float(episode.get("fps") or 30.0)) for episode in train_episodes]
    ).astype(np.float32)
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(features.std(axis=0, dtype=np.float64).astype(np.float32), eps)
    return _contract_with_hash(
        {
            "contract_type": "ula_v2_style_normalization",
            "contract_version": V2_STYLE_CONTRACT_VERSION,
            "feature_names": list(STYLE_FEATURE_NAMES),
            "feature_definition": {
                "signed_arm_balance": "(right_rms_amplitude-left_rms_amplitude)/(right+left+eps)",
                "log_arm_amplitude": "log1p(rms temporal arm deviation in radians)",
                "log_arm_speed": "log1p(rms arm velocity in radians/second)",
            },
            "mean": mean.tolist(),
            "std": std.tolist(),
            "eps": eps,
            "clip": clip,
            "fit_split": "train",
            "fit_episode_count": len(train_episodes),
            "fit_episode_indices": sorted(_episode_id(episode) for episode in train_episodes),
        }
    )


def normalize_style_features(features, style_contract):
    if style_contract.get("contract_version") != V2_STYLE_CONTRACT_VERSION:
        raise ValueError("unsupported V2 style contract version")
    if list(style_contract.get("feature_names") or []) != list(STYLE_FEATURE_NAMES):
        raise ValueError("V2 style feature order does not match")
    values = np.asarray(features, dtype=np.float32)
    mean = np.asarray(style_contract["mean"], dtype=np.float32)
    std = np.asarray(style_contract["std"], dtype=np.float32)
    if values.shape != (len(STYLE_FEATURE_NAMES),) or mean.shape != values.shape or std.shape != values.shape:
        raise ValueError("V2 style features and statistics must have shape [3]")
    if not np.isfinite(values).all() or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("V2 style features and statistics must be finite with positive std")
    return np.clip((values - mean) / std, -float(style_contract["clip"]), float(style_contract["clip"])).astype(
        np.float32
    )


def build_style_bank_contract(train_episodes):
    groups = defaultdict(list)
    for episode in train_episodes:
        behavior_id, emotion_id = _semantic_key(episode)
        controls = np.asarray(episode["style_controls"], dtype=np.float32)
        if controls.shape != (len(STYLE_FEATURE_NAMES),) or not np.isfinite(controls).all():
            raise ValueError("train episode has invalid V2 style controls")
        groups[(behavior_id, emotion_id)].append(
            {
                "episode_index": _episode_id(episode),
                "controls": controls.tolist(),
            }
        )
    rows = []
    for behavior_id, emotion_id in sorted(groups):
        values = sorted(groups[(behavior_id, emotion_id)], key=lambda row: row["episode_index"])
        rows.append(
            {
                "behavior_id": behavior_id,
                "emotion_id": emotion_id,
                "styles": values,
            }
        )
    return _contract_with_hash(
        {
            "contract_type": "ula_v2_train_style_bank",
            "contract_version": V2_STYLE_BANK_CONTRACT_VERSION,
            "feature_names": list(STYLE_FEATURE_NAMES),
            "source_split": "train",
            "default_policy": "seeded_empirical_sample",
            "groups": rows,
        }
    )


def style_controls_for_semantic(style_bank, behavior_id, emotion_id, *, seed=None, mean=False):
    if style_bank.get("contract_version") != V2_STYLE_BANK_CONTRACT_VERSION:
        raise ValueError("unsupported V2 style bank contract version")
    matches = [
        row
        for row in style_bank.get("groups") or []
        if row.get("behavior_id") == behavior_id and row.get("emotion_id") == emotion_id
    ]
    if len(matches) != 1:
        raise ValueError(f"V2 style bank has no unique group for {behavior_id}/{emotion_id}")
    styles = matches[0].get("styles") or []
    values = np.asarray([row["controls"] for row in styles], dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(STYLE_FEATURE_NAMES) or not np.isfinite(values).all():
        raise ValueError("V2 style bank group is invalid")
    if mean:
        return values.mean(axis=0, dtype=np.float64).astype(np.float32)
    index = 0 if seed is None else int(seed) % values.shape[0]
    return values[index].copy()


def build_motion_prototype_contract(train_episodes, motion_checkpoint, *, device="cpu", batch_size=128):
    train_episodes = list(train_episodes)
    if not train_episodes:
        raise ValueError("cannot build motion prototypes without train episodes")
    model, stats, checkpoint = load_motion_metric_checkpoint(motion_checkpoint, device=device)
    _, expected_latent_dim, _ = _condition_dimensions()
    config = checkpoint.get("config") or {}
    if int(config.get("action_dim", -1)) != len(JOINT_ORDER):
        raise ValueError("Motion Metric action dimension does not match Kimodo")
    if int(config.get("latent_dim", -1)) != expected_latent_dim:
        raise ValueError(f"Motion Metric latent dimension must be {expected_latent_dim}")
    episodes_by_length = defaultdict(list)
    for row, episode in enumerate(train_episodes):
        actions = np.asarray(episode["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != len(JOINT_ORDER) or actions.shape[0] < 1:
            raise ValueError(f"episode {_episode_id(episode)} has invalid actions for Motion Metric encoding")
        episodes_by_length[int(actions.shape[0])].append((row, episode))
    embeddings = [None] * len(train_episodes)
    for frame_count in sorted(episodes_by_length):
        rows = episodes_by_length[frame_count]
        encoded = encode_motion_episodes(
            model,
            [episode for _, episode in rows],
            stats,
            batch_size=int(batch_size),
            device=device,
        )
        for encoded_row, (source_row, _) in enumerate(rows):
            embeddings[source_row] = np.asarray(encoded.embeddings[encoded_row], dtype=np.float32)
    groups = defaultdict(list)
    for row, episode in enumerate(train_episodes):
        groups[_semantic_key(episode)].append(embeddings[row])

    group_rows = []
    for behavior_id, emotion_id in sorted(groups):
        values = np.stack(groups[(behavior_id, emotion_id)]).astype(np.float32)
        prototype = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        norm = float(np.linalg.norm(prototype))
        if not np.isfinite(norm) or norm <= 1e-8:
            raise ValueError(f"degenerate Motion Metric prototype for {behavior_id}/{emotion_id}")
        prototype /= norm
        group_rows.append(
            {
                "behavior_id": behavior_id,
                "emotion_id": emotion_id,
                "train_episode_count": int(values.shape[0]),
                "prototype": prototype.tolist(),
            }
        )
    return _contract_with_hash(
        {
            "contract_type": "ula_v2_motion_prototype_bank",
            "contract_version": V2_PROTOTYPE_CONTRACT_VERSION,
            "source_checkpoint_sha256": _sha256_file(motion_checkpoint),
            "source_split": "train",
            "latent_dim": expected_latent_dim,
            "source_duration_policy": "native_trimmed_variable_length",
            "normalization_source": "motion_metric_checkpoint_train_split",
            "l2_normalized": True,
            "groups": group_rows,
        }
    )


def motion_prototype_for_semantic(prototype_contract, behavior_id, emotion_id):
    if prototype_contract.get("contract_version") != V2_PROTOTYPE_CONTRACT_VERSION:
        raise ValueError("unsupported V2 motion prototype contract version")
    matches = [
        row
        for row in prototype_contract.get("groups") or []
        if row.get("behavior_id") == behavior_id and row.get("emotion_id") == emotion_id
    ]
    if len(matches) != 1:
        raise ValueError(f"V2 prototype bank has no unique group for {behavior_id}/{emotion_id}")
    prototype = np.asarray(matches[0]["prototype"], dtype=np.float32)
    expected_dim = int(prototype_contract["latent_dim"])
    if prototype.shape != (expected_dim,) or not np.isfinite(prototype).all():
        raise ValueError("V2 motion prototype has invalid shape or values")
    norm = float(np.linalg.norm(prototype))
    if not np.isclose(norm, 1.0, atol=1e-5):
        raise ValueError("V2 motion prototype is not L2-normalized")
    return prototype.copy()


def assemble_v2_condition(base_condition, *, behavior_id, emotion_id, prototype_contract, style_controls=None):
    base_dim, latent_dim, v2_dim = _condition_dimensions()
    base = np.asarray(base_condition, dtype=np.float32).copy()
    if base.shape != (base_dim,) or not np.isfinite(base).all():
        raise ValueError(f"base Kimodo condition must be a finite vector with shape [{base_dim}]")
    if style_controls is None:
        controls = np.zeros(len(STYLE_FEATURE_NAMES), dtype=np.float32)
    else:
        controls = np.asarray(style_controls, dtype=np.float32)
    if controls.shape != (len(STYLE_FEATURE_NAMES),) or not np.isfinite(controls).all():
        raise ValueError("V2 style controls must be a finite vector with shape [3]")
    prototype = motion_prototype_for_semantic(prototype_contract, behavior_id, emotion_id)
    if prototype.shape != (latent_dim,):
        raise ValueError(f"V2 motion prototype must have shape [{latent_dim}]")
    base[-len(STYLE_FEATURE_NAMES) :] = controls
    condition = np.concatenate([base, prototype]).astype(np.float32)
    if condition.shape != (v2_dim,) or not np.isfinite(condition).all():
        raise ValueError(f"assembled V2 condition must be a finite vector with shape [{v2_dim}]")
    return condition


def _load_dataset_episodes(dataset_dir):
    dataset_dir = Path(dataset_dir)
    semantic_path = dataset_dir / "meta" / "semantic_index.parquet"
    if not semantic_path.is_file():
        raise FileNotFoundError(f"Kimodo semantic index not found: {semantic_path}")
    semantic_rows = pq.read_table(semantic_path).to_pylist()
    meta_by_id = {int(row["episode_index"]): row for row in semantic_rows}
    if len(meta_by_id) != len(semantic_rows):
        raise ValueError("Kimodo semantic index contains duplicate episode indices")
    episodes = load_motion_latent_episodes(dataset_dir)
    result = []
    for episode in episodes:
        episode_index = _episode_id(episode)
        if episode_index not in meta_by_id:
            raise ValueError(f"episode {episode_index} has no semantic metadata")
        item = dict(episode)
        item["meta"] = dict(meta_by_id[episode_index])
        item["fps"] = float(item["meta"].get("fps") or 30.0)
        result.append(item)
    return result


def _base_condition(episode, expected_dim):
    condition = episode.get("condition")
    if condition is None:
        # Lazy import avoids a module cycle when ula_training integrates this module.
        from upper_body_skeleton.ula_training import condition_vector

        condition = condition_vector(episode.get("meta") or {})
    condition = np.asarray(condition, dtype=np.float32)
    if condition.shape != (expected_dim,) or not np.isfinite(condition).all():
        raise ValueError(f"episode {_episode_id(episode)} has an invalid base Kimodo condition")
    return condition.copy()


def prepare_v2_episode_splits(
    dataset_dir,
    split_checkpoint,
    *,
    device="cpu",
    episodes=None,
    max_velocity_rad_s=3.0,
    smooth_window=1,
    default_fps=30.0,
    style_clip=5.0,
    active_window_contract=None,
):
    """Build leakage-safe V2 train/validation/test episodes and serializable contracts."""
    source_episodes = _load_dataset_episodes(dataset_dir) if episodes is None else list(episodes)
    split_episodes, split_contract = load_validated_episode_splits(split_checkpoint, source_episodes)
    preprocess_contract = build_preprocess_contract(
        max_velocity_rad_s=max_velocity_rad_s,
        smooth_window=smooth_window,
        default_fps=default_fps,
    )
    if active_window_contract is None:
        active_window_contract = build_active_window_contract()

    cleaned_splits = {}
    for split_name in _SPLIT_NAMES:
        cleaned_splits[split_name] = []
        for episode in split_episodes[split_name]:
            item = dict(episode)
            item["meta"] = dict(episode.get("meta") or {})
            fps = float(episode.get("fps") or item["meta"].get("fps") or default_fps)
            item["fps"] = fps
            item["actions"] = clean_joint_trajectory(
                episode["actions"], fps=fps, preprocess_contract=preprocess_contract
            )
            cleaned_splits[split_name].append(
                trim_episode(item, active_window_contract=active_window_contract)
            )

    duration_contract = build_duration_contract(cleaned_splits["train"], active_window_contract)
    style_contract = fit_style_contract(cleaned_splits["train"], clip=style_clip)
    prototype_contract = build_motion_prototype_contract(
        cleaned_splits["train"], split_checkpoint, device=device
    )
    base_dim, latent_dim, v2_dim = _condition_dimensions()
    for split_name in _SPLIT_NAMES:
        for episode in cleaned_splits[split_name]:
            style_features = extract_style_features(episode["actions"], fps=episode["fps"])
            style_controls = normalize_style_features(style_features, style_contract)
            behavior_id, emotion_id = _semantic_key(episode)
            episode["style_features"] = style_features
            episode["style_controls"] = style_controls
            episode["condition"] = assemble_v2_condition(
                _base_condition(episode, base_dim),
                behavior_id=behavior_id,
                emotion_id=emotion_id,
                prototype_contract=prototype_contract,
                style_controls=style_controls,
            )
    style_bank_contract = build_style_bank_contract(cleaned_splits["train"])

    condition_contract = _contract_with_hash(
        {
            "contract_type": "ula_v2_condition",
            "contract_version": V2_CONDITION_CONTRACT_VERSION,
            "condition_dim": v2_dim,
            "base_condition_dim": base_dim,
            "motion_latent_dim": latent_dim,
            "layout": [
                {"name": "kimodo_base_with_style_controls", "start": 0, "end": base_dim},
                {"name": "train_group_motion_prototype", "start": base_dim, "end": v2_dim},
            ],
            "style_control_indices": [base_dim - 3, base_dim - 2, base_dim - 1],
            "default_inference_style_controls": [0.0, 0.0, 0.0],
            "split_contract_sha256": split_contract["sha256"],
            "preprocess_contract_sha256": preprocess_contract["sha256"],
            "active_window_contract_sha256": active_window_contract["sha256"],
            "duration_contract_sha256": duration_contract["sha256"],
            "style_contract_sha256": style_contract["sha256"],
            "style_bank_contract_sha256": style_bank_contract["sha256"],
            "prototype_contract_sha256": prototype_contract["sha256"],
        }
    )
    contracts = {
        "contract_version": V2_CONDITION_CONTRACT_VERSION,
        "split": split_contract,
        "preprocess": preprocess_contract,
        "active_window": active_window_contract,
        "duration": duration_contract,
        "style": style_contract,
        "style_bank": style_bank_contract,
        "motion_prototypes": prototype_contract,
        "condition": condition_contract,
    }
    contracts["sha256"] = hashlib.sha256(_canonical_json(contracts).encode("ascii")).hexdigest()
    return (
        cleaned_splits["train"],
        cleaned_splits["validation"],
        cleaned_splits["test"],
        contracts,
    )


__all__ = [
    "STYLE_FEATURE_NAMES",
    "assemble_v2_condition",
    "build_active_window_contract",
    "build_duration_contract",
    "build_style_bank_contract",
    "build_motion_prototype_contract",
    "build_preprocess_contract",
    "clean_joint_trajectory",
    "extract_style_features",
    "extract_active_motion_window",
    "fit_style_contract",
    "load_validated_episode_splits",
    "motion_prototype_for_semantic",
    "normalize_style_features",
    "prepare_v2_episode_splits",
    "style_controls_for_semantic",
    "trim_episode",
]
