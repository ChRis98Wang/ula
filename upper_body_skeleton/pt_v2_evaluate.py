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

        candidates = []
        for seed in seeds:
            cache_key = (
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
            motion = generated_cache.get(cache_key)
            if motion is None:
                motion = generator.infer(
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
                generated_cache[cache_key] = motion
            if motion.behavior_id != behavior_id or motion.emotion_id != emotion_id:
                raise ValueError(
                    f"generator resolved {motion.behavior_id}/{motion.emotion_id}, "
                    f"expected {behavior_id}/{emotion_id}"
                )
            candidates.append(
                {
                    "seed": seed,
                    "trajectory": motion.trajectory,
                    "predicted_duration_sec": getattr(motion, "predicted_duration_sec", None),
                }
            )

        results.append(
            {
                "episode_index": episode_index,
                "sample_id": row.get("sample_id"),
                "behavior_id": behavior_id,
                "emotion_id": emotion_id,
                "frames": int(reference.shape[0]),
                "prompt": {
                    "text": prompt.text,
                    "source": prompt.source,
                    "pair_id": prompt.pair_id,
                },
                "best_of_k": best_of_k_metrics(candidates, reference, fps=fps),
                "reference_duration_sec": float(reference.shape[0] / fps),
                "predicted_duration_sec": [
                    candidate["predicted_duration_sec"] for candidate in candidates
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
    semantic_adapter_checkpoint, *, generator_checkpoint, dataset_dir
):
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
    output_json=DEFAULT_OUTPUT_JSON,
    split="test",
    behavior_ids=None,
    emotion_ids=None,
    max_references=None,
    seeds=DEFAULT_SEEDS,
    device="auto",
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
            "name": "v2_fixed_label_heldout_best_of_k",
            "trajectory_variant": "postprocessed",
            "selection_metric": "position_rmse_rad",
            "prompt_provider": "dataset.language_instruction",
            "condition_source": "canonical_behavior_emotion_bank",
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
            "semantic_adapter_checkpoint": str(
                Path(semantic_adapter_checkpoint).resolve()
            ),
            "semantic_adapter_checkpoint_sha256": _file_sha256(
                semantic_adapter_checkpoint
            ),
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
        output_json=args.output_json,
        split=args.split,
        behavior_ids=args.behavior_ids,
        emotion_ids=args.emotion_ids,
        max_references=args.max_references,
        seeds=seeds,
        device=args.device,
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
