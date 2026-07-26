#!/usr/bin/env python3
"""Reproducible held-out evaluation for V2 joint-motion generators."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pyarrow.parquet as pq
import torch

from upper_body_skeleton.pt_dataset_mujoco_compare import (
    DEFAULT_DATASET_DIR,
    DEFAULT_MOTION_SPLIT_CHECKPOINT,
    load_reference_trajectories,
    select_dataset_reference_rows,
    validate_dataset_contract,
)
from upper_body_skeleton.pt_mujoco_infer import (
    DEFAULT_SEMANTIC_ADAPTER_CHECKPOINT,
    EXPERIMENTAL_KIMODO_CHECKPOINT,
    PtMotionGenerator,
    load_motion_latent_lora_condition_builder,
    validate_generator_condition_source,
)
from upper_body_skeleton.retarget_v2 import JOINT_ORDER
from upper_body_skeleton.semantic_adapter import (
    BEHAVIOR_TO_INDEX,
    EMOTION_TO_INDEX,
    SemanticPrediction,
    validate_condition_bank,
    validate_semantic_adapter_checkpoint,
    validate_semantic_labels,
)
from upper_body_skeleton.ula_training import (
    frame_count_to_coverage,
    frame_count_to_sample_span,
)
from upper_body_skeleton.ula_training_v2 import resample_motion_phase


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_JSON = (
    REPO_ROOT
    / "training"
    / "runs"
    / "kimodo_mmdit_lite_qwen_compatible_5k_math_sdp"
    / "v2_evaluation"
    / "metrics.json"
)
DEFAULT_SEEDS = (7, 17, 29, 43)
SPLIT_NAMES = ("train", "validation", "test")
LEFT_ARM_SLICE = slice(3, 9)
RIGHT_ARM_SLICE = slice(9, 15)


@dataclass(frozen=True)
class EvaluationPrompt:
    """Prompt attached to a reference; alternative providers can supply paraphrases."""

    text: str
    source: str
    pair_id: str | None = None


class FixedLabelConditionBuilder:
    """Read the canonical condition bank without loading Qwen for known-label evaluation."""

    def __init__(self, condition_bank):
        self.condition_bank = validate_condition_bank(condition_bank)
        self.last_prediction = None

    def __call__(self, text, *, behavior_id=None, emotion_id=None, condition_dim=136, **_kwargs):
        if behavior_id is None or emotion_id is None:
            raise ValueError("fixed-label evaluation requires behavior_id and emotion_id")
        validate_semantic_labels(behavior_id, emotion_id)
        expected_dim = int(self.condition_bank["condition_dim"])
        if int(condition_dim) != expected_dim:
            raise ValueError(
                "condition bank has dimension "
                f"{expected_dim}, generator expects {int(condition_dim)}"
            )
        self.last_prediction = SemanticPrediction(
            text=str(text).strip(),
            behavior_id=behavior_id,
            emotion_id=emotion_id,
            behavior_confidence=1.0,
            emotion_confidence=1.0,
        )
        vector = self.condition_bank["vectors"][
            BEHAVIOR_TO_INDEX[behavior_id], EMOTION_TO_INDEX[emotion_id]
        ]
        return vector.detach().cpu().numpy().astype(np.float32, copy=True)


def _atomic_json_write(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_trajectory(value, *, name):
    trajectory = np.asarray(value, dtype=np.float32)
    if trajectory.ndim != 2 or trajectory.shape[1] != len(JOINT_ORDER):
        raise ValueError(f"{name} must have shape [frames, {len(JOINT_ORDER)}]")
    if trajectory.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one frame")
    if not np.isfinite(trajectory).all():
        raise ValueError(f"{name} contains non-finite values")
    return trajectory


def _validated_fps(fps):
    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    return fps


def _rms(value):
    value = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.square(value).mean())) if value.size else 0.0


def trajectory_kinematics(trajectory, *, fps):
    """Return physical-scale kinematics and activity for one trajectory."""

    trajectory = _validated_trajectory(trajectory, name="trajectory")
    fps = _validated_fps(fps)
    velocity = np.diff(trajectory, axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps
    joint_range = np.ptp(trajectory, axis=0)
    left_activity = _rms(velocity[:, LEFT_ARM_SLICE])
    right_activity = _rms(velocity[:, RIGHT_ARM_SLICE])
    activity_total = left_activity + right_activity
    return {
        "velocity_rms_rad_s": _rms(velocity),
        "acceleration_rms_rad_s2": _rms(acceleration),
        "per_joint_range_rad": {
            joint: float(joint_range[index]) for index, joint in enumerate(JOINT_ORDER)
        },
        "left_right_activity_rad_s": {
            "left": left_activity,
            "right": right_activity,
            "asymmetry": (
                float(abs(left_activity - right_activity) / activity_total)
                if activity_total > 1e-12
                else 0.0
            ),
        },
    }


def trajectory_pair_metrics(generated, reference, *, fps):
    """Compare one generated trajectory with one time-aligned reference."""

    generated = _validated_trajectory(generated, name="generated trajectory")
    reference = _validated_trajectory(reference, name="reference trajectory")
    if generated.shape != reference.shape:
        raise ValueError("generated and reference trajectories must have identical shapes")
    fps = _validated_fps(fps)

    position_error = generated - reference
    generated_velocity = np.diff(generated, axis=0) * fps
    reference_velocity = np.diff(reference, axis=0) * fps
    generated_acceleration = np.diff(generated_velocity, axis=0) * fps
    reference_acceleration = np.diff(reference_velocity, axis=0) * fps
    generated_profile = trajectory_kinematics(generated, fps=fps)
    reference_profile = trajectory_kinematics(reference, fps=fps)

    generated_range = generated_profile["per_joint_range_rad"]
    reference_range = reference_profile["per_joint_range_rad"]
    range_error = np.asarray(
        [generated_range[joint] - reference_range[joint] for joint in JOINT_ORDER],
        dtype=np.float64,
    )
    generated_activity = generated_profile["left_right_activity_rad_s"]
    reference_activity = reference_profile["left_right_activity_rad_s"]
    return {
        "position_rmse_rad": _rms(position_error),
        "velocity_rmse_rad_s": _rms(generated_velocity - reference_velocity),
        "acceleration_rmse_rad_s2": _rms(
            generated_acceleration - reference_acceleration
        ),
        "per_joint_position_rmse_rad": {
            joint: _rms(position_error[:, index])
            for index, joint in enumerate(JOINT_ORDER)
        },
        "range_rmse_rad": _rms(range_error),
        "per_joint_range_rad": {
            joint: {
                "generated": float(generated_range[joint]),
                "reference": float(reference_range[joint]),
                "absolute_error": float(abs(generated_range[joint] - reference_range[joint])),
            }
            for joint in JOINT_ORDER
        },
        "left_right_activity_rad_s": {
            "generated": generated_activity,
            "reference": reference_activity,
            "absolute_error": {
                name: float(abs(generated_activity[name] - reference_activity[name]))
                for name in ("left", "right", "asymmetry")
            },
        },
        "generated_kinematics": generated_profile,
        "reference_kinematics": reference_profile,
    }


def best_of_k_metrics(candidates, reference, *, fps):
    """Select one candidate by position RMSE and retain deterministic seed scores."""

    candidates = list(candidates)
    if not candidates:
        raise ValueError("best-of-K evaluation requires at least one candidate")
    scored = []
    for candidate_index, candidate in enumerate(candidates):
        if "seed" not in candidate or "trajectory" not in candidate:
            raise ValueError("each candidate must contain seed and trajectory")
        metrics = trajectory_pair_metrics(candidate["trajectory"], reference, fps=fps)
        scored.append(
            {
                "candidate_index": int(candidate_index),
                "seed": int(candidate["seed"]),
                "metrics": metrics,
            }
        )
    best = min(
        scored,
        key=lambda row: (row["metrics"]["position_rmse_rad"], row["candidate_index"]),
    )
    return {
        "k": len(scored),
        "selection_metric": "position_rmse_rad",
        "position_rmse_best_of_k_rad": float(best["metrics"]["position_rmse_rad"]),
        "best_candidate_index": int(best["candidate_index"]),
        "best_seed": int(best["seed"]),
        "selected_metrics": best["metrics"],
        "candidate_scores": [
            {
                "candidate_index": int(row["candidate_index"]),
                "seed": int(row["seed"]),
                "position_rmse_rad": float(row["metrics"]["position_rmse_rad"]),
                "velocity_rmse_rad_s": float(row["metrics"]["velocity_rmse_rad_s"]),
                "acceleration_rmse_rad_s2": float(
                    row["metrics"]["acceleration_rmse_rad_s2"]
                ),
                "range_rmse_rad": float(row["metrics"]["range_rmse_rad"]),
            }
            for row in scored
        ],
    }


def free_length_best_of_k_metrics(candidates, reference, *, fps):
    """Score native-length candidates without using reference length at generation."""
    candidates = list(candidates)
    if not candidates:
        raise ValueError("free-length best-of-K evaluation requires candidates")
    reference = _validated_trajectory(reference, name="reference trajectory")
    if reference.shape[0] < 2:
        raise ValueError("free-length evaluation requires at least two reference frames")
    fps = _validated_fps(fps)
    reference_frames = int(reference.shape[0])
    reference_span = frame_count_to_sample_span(reference_frames, fps)
    scored = []
    for candidate_index, candidate in enumerate(candidates):
        generated = _validated_trajectory(
            candidate.get("trajectory"), name="free-length generated trajectory"
        )
        if generated.shape[0] < 2:
            raise ValueError("free-length generation must contain at least two frames")
        generated_frames = int(generated.shape[0])
        generated_span = frame_count_to_sample_span(generated_frames, fps)
        aligned = resample_motion_phase(generated, reference_frames)
        metrics = trajectory_pair_metrics(aligned, reference, fps=fps)
        scored.append(
            {
                "candidate_index": int(candidate_index),
                "seed": int(candidate["seed"]),
                "metrics": metrics,
                "generated_frame_count": generated_frames,
                "generated_sample_span_sec": generated_span,
                "generated_frame_coverage_sec": frame_count_to_coverage(
                    generated_frames, fps
                ),
                "sample_span_error_sec": generated_span - reference_span,
                "sample_span_absolute_error_sec": abs(
                    generated_span - reference_span
                ),
                "frame_count_error": generated_frames - reference_frames,
                "frame_count_absolute_error": abs(
                    generated_frames - reference_frames
                ),
                "semantic_label_match": bool(
                    candidate.get("semantic_label_match", True)
                ),
                "predicted_duration_sec": candidate.get(
                    "predicted_duration_sec"
                ),
            }
        )
    best = min(
        scored,
        key=lambda row: (
            row["metrics"]["position_rmse_rad"],
            row["sample_span_absolute_error_sec"],
            row["candidate_index"],
        ),
    )
    return {
        "k": len(scored),
        "generation_length_policy": "duration_head_native_length_no_reference_frames",
        "comparison_alignment": "phase_resample_generated_to_reference_for_action_metrics_only",
        "selection_metric": (
            "phase_aligned_position_rmse_rad_then_sample_span_absolute_error_sec"
        ),
        "position_rmse_best_of_k_rad": float(
            best["metrics"]["position_rmse_rad"]
        ),
        "best_candidate_index": int(best["candidate_index"]),
        "best_seed": int(best["seed"]),
        "selected_metrics": best["metrics"],
        "selected_generated_frame_count": int(best["generated_frame_count"]),
        "selected_generated_sample_span_sec": float(
            best["generated_sample_span_sec"]
        ),
        "selected_sample_span_error_sec": float(best["sample_span_error_sec"]),
        "selected_sample_span_absolute_error_sec": float(
            best["sample_span_absolute_error_sec"]
        ),
        "selected_frame_count_error": int(best["frame_count_error"]),
        "selected_frame_count_absolute_error": int(
            best["frame_count_absolute_error"]
        ),
        "selected_semantic_label_match": bool(best["semantic_label_match"]),
        "candidate_scores": [
            {
                "candidate_index": int(row["candidate_index"]),
                "seed": int(row["seed"]),
                "generated_frame_count": int(row["generated_frame_count"]),
                "generated_sample_span_sec": float(
                    row["generated_sample_span_sec"]
                ),
                "generated_frame_coverage_sec": float(
                    row["generated_frame_coverage_sec"]
                ),
                "sample_span_error_sec": float(row["sample_span_error_sec"]),
                "sample_span_absolute_error_sec": float(
                    row["sample_span_absolute_error_sec"]
                ),
                "frame_count_error": int(row["frame_count_error"]),
                "frame_count_absolute_error": int(
                    row["frame_count_absolute_error"]
                ),
                "semantic_label_match": bool(row["semantic_label_match"]),
                "predicted_duration_sec": row["predicted_duration_sec"],
                "phase_aligned_position_rmse_rad": float(
                    row["metrics"]["position_rmse_rad"]
                ),
                "phase_aligned_velocity_rmse_rad_s": float(
                    row["metrics"]["velocity_rmse_rad_s"]
                ),
                "phase_aligned_acceleration_rmse_rad_s2": float(
                    row["metrics"]["acceleration_rmse_rad_s2"]
                ),
                "range_rmse_rad": float(row["metrics"]["range_rmse_rad"]),
            }
            for row in scored
        ],
    }


def _scalar_summary(values):
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot summarize an empty metric")
    if not np.isfinite(values).all():
        raise ValueError("cannot summarize non-finite metrics")
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def aggregate_episode_metrics(results, *, include_groups=True):
    results = list(results)
    if not results:
        raise ValueError("cannot aggregate an empty evaluation")

    selected = [row["best_of_k"]["selected_metrics"] for row in results]
    aggregate = {
        "reference_count": len(results),
        "candidate_count": int(sum(row["best_of_k"]["k"] for row in results)),
        "position_rmse_best_of_k_rad": _scalar_summary(
            row["best_of_k"]["position_rmse_best_of_k_rad"] for row in results
        ),
        "velocity_rmse_selected_rad_s": _scalar_summary(
            row["velocity_rmse_rad_s"] for row in selected
        ),
        "acceleration_rmse_selected_rad_s2": _scalar_summary(
            row["acceleration_rmse_rad_s2"] for row in selected
        ),
        "range_rmse_selected_rad": _scalar_summary(row["range_rmse_rad"] for row in selected),
        "left_right_activity_absolute_error_selected": {
            name: _scalar_summary(
                row["left_right_activity_rad_s"]["absolute_error"][name]
                for row in selected
            )
            for name in ("left", "right", "asymmetry")
        },
        "per_joint_range_absolute_error_selected_rad": {
            joint: _scalar_summary(
                row["per_joint_range_rad"][joint]["absolute_error"] for row in selected
            )
            for joint in JOINT_ORDER
        },
        "best_seed_counts": {
            str(seed): int(count)
            for seed, count in sorted(
                Counter(int(row["best_of_k"]["best_seed"]) for row in results).items()
            )
        },
    }
    if all(
        "selected_sample_span_absolute_error_sec" in row["best_of_k"]
        for row in results
    ):
        aggregate.update(
            {
                "primary_length_policy": (
                    "duration_head_native_length_no_reference_frames"
                ),
                "sample_span_mae_selected_sec": _scalar_summary(
                    row["best_of_k"]["selected_sample_span_absolute_error_sec"]
                    for row in results
                ),
                "frame_count_absolute_error_selected": _scalar_summary(
                    row["best_of_k"]["selected_frame_count_absolute_error"]
                    for row in results
                ),
                "semantic_label_match_rate_selected": float(
                    np.mean(
                        [
                            bool(row["best_of_k"]["selected_semantic_label_match"])
                            for row in results
                        ]
                    )
                ),
            }
        )
    if include_groups:
        grouped = defaultdict(list)
        for row in results:
            grouped[(row["behavior_id"], row["emotion_id"])].append(row)
        aggregate["by_label"] = {
            f"{behavior_id}|{emotion_id}": {
                "behavior_id": behavior_id,
                "emotion_id": emotion_id,
                "metrics": aggregate_episode_metrics(group, include_groups=False),
            }
            for (behavior_id, emotion_id), group in sorted(grouped.items())
        }
    return aggregate


def _normalize_filter(values, *, field):
    if values is None:
        return None
    if isinstance(values, str):
        values = [values]
    normalized = []
    for value in values:
        value = str(value).strip()
        if not value:
            raise ValueError(f"{field} cannot be empty")
        if value not in normalized:
            normalized.append(value)
    return normalized or None


def _balanced_reference_order(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(str(row["behavior_id"]), str(row["emotion_id"]))].append(row)
    for group in groups.values():
        group.sort(key=lambda row: int(row["episode_index"]))
    ordered = []
    depth = 0
    while True:
        added = False
        for key in sorted(groups):
            if depth < len(groups[key]):
                ordered.append(groups[key][depth])
                added = True
        if not added:
            return ordered
        depth += 1


def select_evaluation_reference_rows(
    dataset_dir,
    split_checkpoint_path,
    *,
    split="test",
    behavior_ids=None,
    emotion_ids=None,
    max_references=None,
):
    """Select all matching references through the comparison script's split validator."""

    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}")
    behavior_ids = _normalize_filter(behavior_ids, field="behavior_id")
    emotion_ids = _normalize_filter(emotion_ids, field="emotion_id")
    max_references = None if max_references is None else int(max_references)
    if max_references is not None and max_references <= 0:
        raise ValueError("max_references must be positive when provided")

    semantic_path = Path(dataset_dir) / "meta" / "semantic_index.parquet"
    rows = pq.read_table(
        semantic_path,
        columns=[
            "episode_index",
            "sample_id",
            "language_instruction",
            "behavior_id",
            "emotion_id",
        ],
    ).to_pylist()
    dataset_behaviors = {str(row["behavior_id"]) for row in rows}
    dataset_emotions = {str(row["emotion_id"]) for row in rows}
    unknown_behaviors = set(behavior_ids or ()) - dataset_behaviors
    unknown_emotions = set(emotion_ids or ()) - dataset_emotions
    if unknown_behaviors:
        raise ValueError(f"unknown behavior_id filters: {sorted(unknown_behaviors)}")
    if unknown_emotions:
        raise ValueError(f"unknown emotion_id filters: {sorted(unknown_emotions)}")

    checkpoint = torch.load(split_checkpoint_path, map_location="cpu", weights_only=True)
    manifest = checkpoint.get("split_episode_indices")
    if not isinstance(manifest, dict) or split not in manifest:
        raise ValueError("motion split checkpoint has no requested episode partition")
    allowed_ids = {int(value) for value in manifest[split]}
    behavior_filter = None if behavior_ids is None else set(behavior_ids)
    emotion_filter = None if emotion_ids is None else set(emotion_ids)
    matching = [
        row
        for row in rows
        if int(row["episode_index"]) in allowed_ids
        and (behavior_filter is None or str(row["behavior_id"]) in behavior_filter)
        and (emotion_filter is None or str(row["emotion_id"]) in emotion_filter)
    ]
    if not matching:
        raise ValueError("no held-out references match the requested filters")

    # select_dataset_reference_rows performs the canonical split/dataset contract checks.
    validated = []
    if emotion_ids is None:
        validated = select_dataset_reference_rows(
            dataset_dir,
            split_checkpoint_path,
            motion_latent_split=split,
            behavior_id=behavior_ids,
            count=len(matching),
        )
    else:
        requested_behaviors = behavior_ids or sorted(dataset_behaviors)
        for emotion_id in emotion_ids:
            count = sum(str(row["emotion_id"]) == emotion_id for row in matching)
            if count:
                validated.extend(
                    select_dataset_reference_rows(
                        dataset_dir,
                        split_checkpoint_path,
                        motion_latent_split=split,
                        behavior_id=requested_behaviors,
                        emotion_id=emotion_id,
                        count=count,
                    )
                )
    validated_ids = {int(row["episode_index"]) for row in validated}
    matching_ids = {int(row["episode_index"]) for row in matching}
    if validated_ids != matching_ids:
        raise ValueError("validated reference selection does not match the requested filters")
    ordered = _balanced_reference_order(validated)
    return ordered if max_references is None else ordered[:max_references]


def dataset_prompt(row):
    text = str(row.get("language_instruction", "")).strip()
    if not text:
        raise ValueError(f"episode {row.get('episode_index')} has no language_instruction")
    return EvaluationPrompt(text=text, source="dataset.language_instruction")


def evaluate_reference_rows(
    generator,
    selected_rows,
    references: Mapping[int, Mapping[str, object]],
    *,
    fps,
    seeds=DEFAULT_SEEDS,
    condition_builder,
    sampling_steps=32,
    max_velocity_rad_s=3.0,
    smooth_window=None,
    prompt_provider: Callable[[Mapping[str, object]], EvaluationPrompt] = dataset_prompt,
):
    """Evaluate injected references; prompt_provider is the paraphrase extension point."""

    fps = _validated_fps(fps)
    seeds = [int(seed) for seed in seeds]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be a non-empty collection of unique integers")
    sampling_steps = int(sampling_steps)
    if sampling_steps <= 0:
        raise ValueError("sampling_steps must be positive")

    generated_cache = {}
    results = []
    for row in selected_rows:
        episode_index = int(row["episode_index"])
        if episode_index not in references:
            raise ValueError(f"reference trajectory for episode {episode_index} is missing")
        reference = _validated_trajectory(
            references[episode_index]["actions"], name=f"reference episode {episode_index}"
        )
        reference_style_controls = references[episode_index].get("style_controls")
        behavior_id = str(row["behavior_id"])
        emotion_id = str(row["emotion_id"])
        validate_semantic_labels(behavior_id, emotion_id)
        prompt = prompt_provider(row)
        if not isinstance(prompt, EvaluationPrompt) or not prompt.text.strip():
            raise ValueError("prompt_provider must return a non-empty EvaluationPrompt")

        primary_candidates = []
        oracle_candidates = []
        for seed in seeds:
            primary_cache_key = (
                "primary_non_oracle",
                prompt.text,
                behavior_id,
                emotion_id,
                float(fps),
                int(sampling_steps),
                int(seed),
                float(max_velocity_rad_s),
                None if smooth_window is None else int(smooth_window),
                "default_style",
            )
            primary_motion = generated_cache.get(primary_cache_key)
            if primary_motion is None:
                primary_motion = generator.infer(
                    prompt.text,
                    behavior_id=behavior_id,
                    emotion_id=emotion_id,
                    frames=None,
                    fps=fps,
                    sampling_steps=sampling_steps,
                    seed=seed,
                    max_velocity_rad_s=max_velocity_rad_s,
                    smooth_window=smooth_window,
                    condition_builder=condition_builder,
                    style_controls=None,
                )
                generated_cache[primary_cache_key] = primary_motion
            if (
                primary_motion.behavior_id != behavior_id
                or primary_motion.emotion_id != emotion_id
            ):
                raise ValueError(
                    f"generator resolved {primary_motion.behavior_id}/"
                    f"{primary_motion.emotion_id}, "
                    f"expected {behavior_id}/{emotion_id}"
                )
            primary_candidates.append(
                {
                    "seed": seed,
                    "trajectory": primary_motion.trajectory,
                    "predicted_duration_sec": getattr(
                        primary_motion, "predicted_duration_sec", None
                    ),
                    "semantic_label_match": True,
                }
            )

            oracle_cache_key = (
                "secondary_oracle_length",
                prompt.text,
                behavior_id,
                emotion_id,
                int(reference.shape[0]),
                float(fps),
                int(sampling_steps),
                int(seed),
                float(max_velocity_rad_s),
                None if smooth_window is None else int(smooth_window),
                None
                if reference_style_controls is None
                else tuple(float(value) for value in reference_style_controls),
            )
            oracle_motion = generated_cache.get(oracle_cache_key)
            if oracle_motion is None:
                oracle_motion = generator.infer(
                    prompt.text,
                    behavior_id=behavior_id,
                    emotion_id=emotion_id,
                    frames=reference.shape[0],
                    fps=fps,
                    sampling_steps=sampling_steps,
                    seed=seed,
                    max_velocity_rad_s=max_velocity_rad_s,
                    smooth_window=smooth_window,
                    condition_builder=condition_builder,
                    style_controls=reference_style_controls,
                )
                generated_cache[oracle_cache_key] = oracle_motion
            if (
                oracle_motion.behavior_id != behavior_id
                or oracle_motion.emotion_id != emotion_id
            ):
                raise ValueError(
                    f"oracle diagnostic resolved {oracle_motion.behavior_id}/"
                    f"{oracle_motion.emotion_id}, expected {behavior_id}/{emotion_id}"
                )
            oracle_candidates.append(
                {
                    "seed": seed,
                    "trajectory": oracle_motion.trajectory,
                    "predicted_duration_sec": getattr(
                        oracle_motion, "predicted_duration_sec", None
                    ),
                }
            )

        primary = free_length_best_of_k_metrics(
            primary_candidates, reference, fps=fps
        )
        secondary = best_of_k_metrics(oracle_candidates, reference, fps=fps)
        secondary["generation_length_policy"] = (
            "reference_frame_count_for_secondary_diagnostic_only"
        )
        reference_frame_count = int(reference.shape[0])
        reference_sample_span = frame_count_to_sample_span(
            reference_frame_count, fps
        )

        results.append(
            {
                "episode_index": episode_index,
                "sample_id": row.get("sample_id"),
                "behavior_id": behavior_id,
                "emotion_id": emotion_id,
                "frames": reference_frame_count,
                "reference_frame_count": reference_frame_count,
                "prompt": {
                    "text": prompt.text,
                    "source": prompt.source,
                    "pair_id": prompt.pair_id,
                },
                "primary_non_oracle": primary,
                "secondary_oracle_length": secondary,
                "best_of_k": primary,
                "reference_duration_sec": reference_sample_span,
                "reference_sample_span_sec": reference_sample_span,
                "reference_frame_coverage_sec": frame_count_to_coverage(
                    reference_frame_count, fps
                ),
                "predicted_duration_sec": [
                    candidate["predicted_duration_sec"]
                    for candidate in primary_candidates
                ],
            }
        )
    return results


def _generator_training_coverage(checkpoint, *, dataset_episode_count, reference_ids):
    config = checkpoint.get("config") or {}
    declared_ids = None
    declared_source = None
    for source, value in (
        ("checkpoint.training_episode_indices", checkpoint.get("training_episode_indices")),
        (
            "checkpoint.split_episode_indices.train",
            (checkpoint.get("split_episode_indices") or {}).get("train"),
        ),
        ("config.training_episode_indices", config.get("training_episode_indices")),
    ):
        if value is not None:
            declared_ids = {int(item) for item in value}
            declared_source = source
            break

    episodes_loaded = config.get("episodes_loaded")
    if declared_ids is not None:
        overlap = sorted(declared_ids & set(reference_ids))
        held_out = not overlap
    elif episodes_loaded is not None and int(episodes_loaded) == int(dataset_episode_count):
        overlap = None
        held_out = False
    else:
        overlap = None
        held_out = None
    return {
        "dataset_episode_count": int(dataset_episode_count),
        "generator_episodes_loaded": (
            None if episodes_loaded is None else int(episodes_loaded)
        ),
        "declared_training_ids_source": declared_source,
        "declared_training_episode_count": (
            None if declared_ids is None else len(declared_ids)
        ),
        "reference_overlap_count": None if overlap is None else len(overlap),
        "reference_overlap_episode_indices": overlap,
        "references_are_generator_held_out": held_out,
    }


def _load_fixed_label_condition_builder(
    semantic_adapter_checkpoint,
    *,
    generator_checkpoint,
    dataset_dir,
    motion_latent_lora_checkpoint=None,
    motion_latent_device="auto",
    motion_latent_local_files_only=True,
):
    text_motion_contract = (generator_checkpoint.get("v2_contracts") or {}).get(
        "text_motion_latent"
    )
    if text_motion_contract is not None:
        if motion_latent_lora_checkpoint is None:
            raise ValueError(
                "this V2 evaluation requires the Qwen Motion LoRA checkpoint recorded by the generator"
            )
        builder, lora_checkpoint, condition_source, checkpoint_hash = (
            load_motion_latent_lora_condition_builder(
                generator_checkpoint,
                motion_latent_lora_checkpoint,
                dataset_dir=dataset_dir,
                device=motion_latent_device,
                local_files_only=motion_latent_local_files_only,
            )
        )
        builder.motion_latent_lora_provenance = {
            "checkpoint": str(Path(motion_latent_lora_checkpoint).resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "qwen": lora_checkpoint["qwen"],
            "best_step": int(lora_checkpoint["best_step"]),
        }
        return builder, condition_source
    if motion_latent_lora_checkpoint is not None:
        raise ValueError("generator was not trained with Qwen LoRA text-motion latents")
    semantic_checkpoint = torch.load(
        semantic_adapter_checkpoint, map_location="cpu", weights_only=True
    )
    validate_semantic_adapter_checkpoint(
        semantic_checkpoint, path=semantic_adapter_checkpoint
    )
    condition_bank = semantic_checkpoint.get("condition_bank")
    if condition_bank is None:
        raise ValueError("semantic adapter checkpoint has no canonical condition bank")
    condition_source = validate_generator_condition_source(
        generator_checkpoint, condition_bank
    )
    dataset_semantic_path = Path(dataset_dir) / "meta" / "semantic_index.parquet"
    dataset_hash = _file_sha256(dataset_semantic_path)
    if condition_source["semantic_index_sha256"] != dataset_hash:
        raise ValueError("evaluation dataset does not match the generator condition source")
    return FixedLabelConditionBuilder(condition_bank), condition_source


def run_v2_evaluation(
    *,
    dataset_dir=DEFAULT_DATASET_DIR,
    split_checkpoint_path=DEFAULT_MOTION_SPLIT_CHECKPOINT,
    generator_checkpoint=EXPERIMENTAL_KIMODO_CHECKPOINT,
    semantic_adapter_checkpoint=DEFAULT_SEMANTIC_ADAPTER_CHECKPOINT,
    motion_latent_lora_checkpoint=None,
    output_json=DEFAULT_OUTPUT_JSON,
    split="test",
    behavior_ids=None,
    emotion_ids=None,
    max_references=None,
    seeds=DEFAULT_SEEDS,
    device="auto",
    motion_latent_device=None,
    motion_latent_local_files_only=True,
    sampling_steps=32,
    max_velocity_rad_s=3.0,
    smooth_window=None,
):
    dataset_dir = Path(dataset_dir)
    behavior_ids = _normalize_filter(behavior_ids, field="behavior_id")
    emotion_ids = _normalize_filter(emotion_ids, field="emotion_id")
    seeds = [int(seed) for seed in seeds]
    max_references = None if max_references is None else int(max_references)
    dataset_contract = validate_dataset_contract(dataset_dir)
    selected_rows = select_evaluation_reference_rows(
        dataset_dir,
        split_checkpoint_path,
        split=split,
        behavior_ids=behavior_ids,
        emotion_ids=emotion_ids,
        max_references=max_references,
    )
    generator = PtMotionGenerator.from_checkpoint(generator_checkpoint, device=device)
    references, dataset_action_stats = load_reference_trajectories(
        dataset_dir,
        selected_rows,
        generator_checkpoint=generator.checkpoint,
    )
    condition_builder, condition_source = _load_fixed_label_condition_builder(
        semantic_adapter_checkpoint,
        generator_checkpoint=generator.checkpoint,
        dataset_dir=dataset_dir,
        motion_latent_lora_checkpoint=motion_latent_lora_checkpoint,
        motion_latent_device=motion_latent_device or device,
        motion_latent_local_files_only=motion_latent_local_files_only,
    )
    motion_latent_lora_provenance = getattr(
        condition_builder, "motion_latent_lora_provenance", None
    )
    results = evaluate_reference_rows(
        generator,
        selected_rows,
        references,
        fps=dataset_contract["fps"],
        seeds=seeds,
        condition_builder=condition_builder,
        sampling_steps=sampling_steps,
        max_velocity_rad_s=max_velocity_rad_s,
        smooth_window=smooth_window,
    )
    reference_ids = [int(row["episode_index"]) for row in selected_rows]
    coverage = _generator_training_coverage(
        generator.checkpoint,
        dataset_episode_count=int(dataset_action_stats["dataset_episode_count"]),
        reference_ids=reference_ids,
    )
    payload = {
        "schema_version": 1,
        "protocol": {
            "name": "v2_fixed_label_heldout_free_length_primary_v2",
            "trajectory_variant": "postprocessed",
            "primary_generation_length": (
                "duration_head_native_length_no_reference_frames"
            ),
            "primary_action_comparison": (
                "phase_aligned_position_rmse_reference_length_not_generation_input"
            ),
            "primary_length_metrics": [
                "sample_span_mae_selected_sec",
                "frame_count_absolute_error_selected",
            ],
            "secondary_diagnostic": "oracle_reference_frame_count_best_of_k",
            "selection_metric": (
                "validation_only_phase_aligned_position_rmse_then_duration_error"
            ),
            "duration_time_axis": "sample_span=(frame_count-1)/fps",
            "prompt_provider": "dataset.language_instruction",
            "condition_source": "canonical_behavior_emotion_bank",
            "primary_style_policy": "default_style_no_reference_trajectory_style",
            "split_policy": {
                "validation": "model_selection_and_threshold_definition_allowed",
                "test": "sealed_final_report_once_no_model_selection_no_threshold_definition",
                "requested_split_model_selection_eligible": split == "validation",
                "requested_split_threshold_definition_eligible": split
                == "validation",
            },
            "text_motion_conditioning": (
                "qwen_motion_lora_128d"
                if motion_latent_lora_provenance is not None
                else "none"
            ),
            "extension_points": [
                "prompt_provider_for_paraphrase",
                "reference_selector_for_pair_holdout",
            ],
        },
        "request": {
            "split": split,
            "behavior_ids": behavior_ids,
            "emotion_ids": emotion_ids,
            "max_references": max_references,
            "seeds": seeds,
            "sampling_steps": int(sampling_steps),
            "max_velocity_rad_s": float(max_velocity_rad_s),
            "smooth_window": None if smooth_window is None else int(smooth_window),
        },
        "provenance": {
            "dataset_dir": str(dataset_dir.resolve()),
            "dataset_contract": dataset_contract,
            "generator_checkpoint": str(Path(generator_checkpoint).resolve()),
            "generator_checkpoint_sha256": _file_sha256(generator_checkpoint),
            "generator_architecture": generator.info.architecture,
            "generator_configured_steps": generator.info.configured_steps,
            "split_checkpoint": str(Path(split_checkpoint_path).resolve()),
            "split_checkpoint_sha256": _file_sha256(split_checkpoint_path),
            "semantic_adapter_checkpoint": None
            if motion_latent_lora_provenance is not None
            else str(Path(semantic_adapter_checkpoint).resolve()),
            "semantic_adapter_checkpoint_sha256": None
            if motion_latent_lora_provenance is not None
            else _file_sha256(semantic_adapter_checkpoint),
            "motion_latent_lora": motion_latent_lora_provenance,
            "condition_source": condition_source,
            "generator_training_coverage": coverage,
        },
        "aggregate": aggregate_episode_metrics(results),
        "episodes": results,
    }
    _atomic_json_write(payload, output_json)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a PT V2 generator against Motion Metric held-out LeRobot trajectories"
        )
    )
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument(
        "--split-checkpoint", default=str(DEFAULT_MOTION_SPLIT_CHECKPOINT)
    )
    parser.add_argument(
        "--generator-checkpoint", default=str(EXPERIMENTAL_KIMODO_CHECKPOINT)
    )
    parser.add_argument(
        "--semantic-adapter-checkpoint",
        default=str(DEFAULT_SEMANTIC_ADAPTER_CHECKPOINT),
    )
    parser.add_argument("--motion-latent-lora-checkpoint")
    parser.add_argument("--motion-latent-device")
    parser.add_argument("--allow-motion-latent-download", action="store_true")
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--split", choices=SPLIT_NAMES, default="test")
    parser.add_argument("--behavior-id", action="append", dest="behavior_ids")
    parser.add_argument("--emotion-id", action="append", dest="emotion_ids")
    parser.add_argument("--max-references", type=int)
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        dest="seeds",
        help=f"Fixed inference seed; repeat for best-of-K (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sampling-steps", type=int, default=32)
    parser.add_argument("--max-velocity-rad-s", type=float, default=3.0)
    parser.add_argument(
        "--smooth-window",
        type=int,
        help="Defaults to the generator checkpoint preprocessing contract",
    )
    args = parser.parse_args(argv)
    seeds = DEFAULT_SEEDS if args.seeds is None else tuple(args.seeds)
    result = run_v2_evaluation(
        dataset_dir=args.dataset_dir,
        split_checkpoint_path=args.split_checkpoint,
        generator_checkpoint=args.generator_checkpoint,
        semantic_adapter_checkpoint=args.semantic_adapter_checkpoint,
        motion_latent_lora_checkpoint=args.motion_latent_lora_checkpoint,
        output_json=args.output_json,
        split=args.split,
        behavior_ids=args.behavior_ids,
        emotion_ids=args.emotion_ids,
        max_references=args.max_references,
        seeds=seeds,
        device=args.device,
        motion_latent_device=args.motion_latent_device,
        motion_latent_local_files_only=not args.allow_motion_latent_download,
        sampling_steps=args.sampling_steps,
        max_velocity_rad_s=args.max_velocity_rad_s,
        smooth_window=args.smooth_window,
    )
    print(
        json.dumps(
            {
                "output_json": str(Path(args.output_json).resolve()),
                "references": result["aggregate"]["reference_count"],
                "candidates": result["aggregate"]["candidate_count"],
                "position_rmse_best_of_k_mean_rad": result["aggregate"][
                    "position_rmse_best_of_k_rad"
                ]["mean"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
