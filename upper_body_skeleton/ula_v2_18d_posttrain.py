#!/usr/bin/env python3
"""Auditable interaction-domain post-training for an existing 18D ULA V2 model.

The trainer is deliberately separate from the 18D migration/head-adapter path.  It
fine-tunes the complete network at a low learning rate, mixes BEAT 18D examples
with optional Kimodo 15D replay, and preserves missing replay head dimensions as
unobserved values through an explicit per-dimension loss mask.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
import gc
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from upper_body_skeleton.retarget_v2 import JOINT_ORDER
from upper_body_skeleton.retarget_v2_18d import CONTRACT_VERSION, JOINT_ORDER_18D
from upper_body_skeleton.ula_training import (
    KIMODO_V2_CONDITION_DIM,
    TRANSITION_IDS,
    choose_device,
    planner_duration_loss,
    planner_loss,
    planner_transition_loss,
)
from upper_body_skeleton.ula_v2_18d_head import (
    ACTION_DIM,
    LEGACY_ACTION_DIM,
    MOTION_ONLY_EPISODE_CONTRACT,
    MOTION_ONLY_RANDOM_INIT_MODE,
    attach_condition_cache,
    configure_head_adapter_policy,
    frozen_weight_max_error,
    load_18d_episodes,
    load_contract_checkpoint,
    restore_frozen_weights,
    sha256_file,
    validate_condition_cache_for_generator,
    validate_qwen_checkpoint_for_generator,
)


POSTTRAIN_SCHEMA_VERSION = 1
POSTTRAIN_ARTIFACT_KIND = "ula_mmdit_v2_18d_interaction_posttrain"
FULL_RANDOM_INITIALIZATION_MODES = frozenset(
    {
        "full_generator_random_qwen_lora_frozen_v1",
        MOTION_ONLY_RANDOM_INIT_MODE,
    }
)
MAX_FULL_FINETUNE_LR = 1e-4
MAX_HEAD_PROJECTION_LR = 1e-3
SPLIT_NAMES = ("train", "validation", "test")
KNOWN_OUTPUT_NAMES = {
    "last.pt",
    "last.pt.tmp",
    "progress.jsonl",
    "split_manifest.json",
    "training_summary.json",
    "ula_fm_checkpoint.pt",
    "ula_fm_checkpoint.pt.tmp",
}
DEFAULT_CONFIG = {
    "steps": 10_000,
    "batch_size": 16,
    "validation_batch_size": 16,
    "phase_frame_choices": [64, 96, 128],
    "lr": 1e-5,
    "minimum_lr_ratio": 0.1,
    "warmup_steps": 250,
    "weight_decay": 1e-4,
    "adam_eps": 1e-6,
    "max_grad_norm": 1.0,
    "ema_decay": 0.9995,
    "validation_interval": 250,
    "checkpoint_interval": 250,
    "log_interval": 25,
    "replay_evaluation_count": 128,
    "maximum_replay_regression_fraction": 0.03,
    "maximum_replay_regression_absolute": 0.02,
    "early_stopping_patience": 8,
    "early_stopping_min_delta": 1e-4,
    "split_fractions": {"train": 0.7, "validation": 0.15, "test": 0.15},
    "seed": 7,
    "device": "auto",
    "resume_from": None,
    "overwrite": False,
    "allow_unsafe_training_data": False,
    "loss": {
        "flow": 1.0,
        "position": 0.25,
        "body": 0.1,
        "velocity": 0.01,
        "acceleration": 0.0005,
    },
}
DEFAULT_NATIVE_BATCHING = {
    "homogeneous_bucket_batches": True,
    "max_motion_tokens_per_microbatch": 4096,
    "max_attention_elements_per_microbatch": 8_000_000,
    "gradient_accumulation_mode": "dynamic_episode_weighted",
    "oversize_sequence_policy": "single_full_episode_or_fail",
}
OPTIONAL_LOSS_NAMES = frozenset(
    {
        "jerk",
        "head_flow",
        "head_position",
        "head_velocity",
        "head_acceleration",
        "head_jerk",
        "planner",
        "planner_duration",
        "planner_transition",
    }
)
TRAINING_POLICIES = frozenset({"full_network", "head_projection_only"})
SAMPLER_MODES = frozenset({"domain_speaker", "source_speaker_activity"})


def _atomic_torch_save(payload, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_json_save(payload, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_hash(payload: Mapping | Sequence) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_hash(value) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _episode_fps_and_duration(episode: Mapping) -> tuple[float, float]:
    fps = float(episode.get("fps") or 30.0)
    frame_count = int(np.asarray(episode["actions"]).shape[0])
    if not math.isfinite(fps) or fps <= 0 or frame_count < 2:
        raise ValueError(f"{_episode_id(episode)}: invalid fps/effective duration")
    sample_span = float((frame_count - 1) / fps)
    declared = episode.get("duration_sec")
    if declared is not None:
        try:
            declared = float(declared)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{_episode_id(episode)}: duration_sec must be the output sample span"
            ) from exc
        if not math.isfinite(declared) or not math.isclose(
            declared, sample_span, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError(
                f"{_episode_id(episode)}: duration_sec must equal "
                "(output_frame_count-1)/fps; frame coverage N/fps cannot supervise "
                "the planner"
            )
    return fps, sample_span


def _replay_content_record(episode: Mapping) -> dict:
    fps, duration = _episode_fps_and_duration(episode)
    return {
        "episode_id": _episode_id(episode),
        "actions_sha256": _array_hash(episode["actions"]),
        "condition_sha256": _array_hash(episode["condition"]),
        "fps": fps,
        "effective_duration_sec": duration,
    }


def _episode_id(episode: Mapping) -> str:
    value = None
    for field in ("clip_id", "episode_id", "episode_index"):
        candidate = episode.get(field)
        if candidate is not None and str(candidate).strip():
            value = candidate
            break
    if value is None or not str(value).strip():
        raise ValueError("episode is missing clip_id/episode_id/episode_index")
    return str(value).strip()


def resolve_posttrain_config(config: Mapping | None = None) -> dict:
    supplied = dict(config or {})
    scope = supplied.get("training_scope")
    if scope is not None:
        batching = supplied.get("batching") or {}
        if scope == "formal_variable_length_semantic_units":
            if (
                supplied.get("formal_training_enabled") is not True
                or supplied.get("temporal_unit_policy")
                != "full_semantic_unit_variable_length_30hz"
                or not isinstance(batching, Mapping)
                or batching.get("mode") != "native_variable_length"
            ):
                raise ValueError(
                    "formal variable-length training requires native_variable_length "
                    "batching and full-semantic-unit 30 Hz inputs"
                )
            if str(supplied.get("training_policy") or "full_network") != "full_network":
                raise ValueError("formal training requires the full_network policy")
        elif scope == "head_mechanism_experiment_only":
            if (
                supplied.get("formal_training_enabled") is not False
                or supplied.get("temporal_unit_policy")
                != "fixed_window_experimental"
            ):
                raise ValueError(
                    "head-mechanism experiments must remain fixed-window and non-formal"
                )
        else:
            raise ValueError(f"unsupported training_scope: {scope!r}")
    resolved = deepcopy(DEFAULT_CONFIG)
    resolved.update({key: value for key, value in supplied.items() if key not in {"loss", "split_fractions"}})
    resolved["loss"] = dict(DEFAULT_CONFIG["loss"] | dict(supplied.get("loss") or {}))
    resolved["split_fractions"] = dict(
        DEFAULT_CONFIG["split_fractions"] | dict(supplied.get("split_fractions") or {})
    )
    for name in ("steps", "batch_size", "validation_batch_size"):
        resolved[name] = int(resolved[name])
        if resolved[name] <= 0:
            raise ValueError(f"{name} must be positive")
    for name in (
        "warmup_steps",
        "validation_interval",
        "checkpoint_interval",
        "log_interval",
        "early_stopping_patience",
        "replay_evaluation_count",
    ):
        resolved[name] = int(resolved[name])
        if resolved[name] <= 0:
            raise ValueError(f"{name} must be positive")
    for name in (
        "lr",
        "minimum_lr_ratio",
        "weight_decay",
        "adam_eps",
        "max_grad_norm",
        "ema_decay",
        "early_stopping_min_delta",
        "maximum_replay_regression_fraction",
        "maximum_replay_regression_absolute",
    ):
        resolved[name] = float(resolved[name])
        if not math.isfinite(resolved[name]):
            raise ValueError(f"{name} must be finite")
    requested_policy = str(resolved.get("training_policy") or "full_network")
    maximum_lr = (
        MAX_HEAD_PROJECTION_LR
        if requested_policy == "head_projection_only"
        else MAX_FULL_FINETUNE_LR
    )
    if not 0 < resolved["lr"] <= maximum_lr:
        raise ValueError(
            f"{requested_policy} training lr must be in (0, {maximum_lr}]"
        )
    if not 0 < resolved["minimum_lr_ratio"] <= 1:
        raise ValueError("minimum_lr_ratio must be in (0, 1]")
    if (
        resolved["weight_decay"] < 0
        or resolved["early_stopping_min_delta"] < 0
        or resolved["maximum_replay_regression_fraction"] < 0
        or resolved["maximum_replay_regression_absolute"] < 0
    ):
        raise ValueError("weight decay, stopping delta, and replay tolerances must be non-negative")
    if resolved["adam_eps"] <= 0 or resolved["max_grad_norm"] <= 0:
        raise ValueError("adam_eps and max_grad_norm must be positive")
    if not 0 < resolved["ema_decay"] < 1:
        raise ValueError("ema_decay must be between zero and one")
    frames = sorted({int(value) for value in resolved["phase_frame_choices"]})
    if not frames or frames[0] < 3:
        raise ValueError("phase_frame_choices must contain frame counts of at least 3")
    resolved["phase_frame_choices"] = frames
    unknown_losses = set(resolved["loss"]) - set(DEFAULT_CONFIG["loss"]) - OPTIONAL_LOSS_NAMES
    if unknown_losses:
        raise ValueError(f"unsupported post-training losses: {sorted(unknown_losses)}")
    for name, value in resolved["loss"].items():
        resolved["loss"][name] = float(value)
        if not math.isfinite(resolved["loss"][name]) or resolved["loss"][name] < 0:
            raise ValueError(f"loss.{name} must be finite and non-negative")
    if not any(resolved["loss"].values()):
        raise ValueError("at least one post-training loss weight must be positive")
    if set(resolved["split_fractions"]) != set(SPLIT_NAMES):
        raise ValueError(f"split_fractions must define exactly {SPLIT_NAMES}")
    split_total = 0.0
    for name in SPLIT_NAMES:
        value = float(resolved["split_fractions"][name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"split_fractions.{name} must be finite and positive")
        resolved["split_fractions"][name] = value
        split_total += value
    if not math.isclose(split_total, 1.0, abs_tol=1e-8):
        raise ValueError("split_fractions must sum to one")
    resolved["allow_unsafe_training_data"] = bool(resolved["allow_unsafe_training_data"])
    resolved["overwrite"] = bool(resolved["overwrite"])
    if "require_disjoint_replay_evaluation" in supplied:
        resolved["require_disjoint_replay_evaluation"] = bool(
            resolved["require_disjoint_replay_evaluation"]
        )
    if resolved.get("resume_from") in (None, ""):
        resolved["resume_from"] = None
    else:
        resolved["resume_from"] = str(resolved["resume_from"])
    if "training_policy" in supplied:
        policy = str(resolved["training_policy"])
        if policy not in TRAINING_POLICIES:
            raise ValueError(f"training_policy must be one of {sorted(TRAINING_POLICIES)}")
        resolved["training_policy"] = policy
    if "sampler" in supplied:
        sampler = resolved["sampler"]
        if not isinstance(sampler, Mapping):
            raise ValueError("sampler must be a mapping")
        sampler = dict(sampler)
        mode = str(sampler.get("mode") or "domain_speaker")
        if mode not in SAMPLER_MODES:
            raise ValueError(f"sampler.mode must be one of {sorted(SAMPLER_MODES)}")
        sampler["mode"] = mode
        if mode == "source_speaker_activity":
            edges = [float(value) for value in sampler.get("activity_bin_edges_rad_s", [])]
            if len(edges) != 3 or any(not math.isfinite(value) or value < 0 for value in edges):
                raise ValueError(
                    "source_speaker_activity requires three finite non-negative "
                    "activity_bin_edges_rad_s values"
                )
            if edges != sorted(set(edges)):
                raise ValueError("activity_bin_edges_rad_s must be strictly increasing")
            sampler["activity_bin_edges_rad_s"] = edges
        resolved["sampler"] = sampler
    if "batching" in supplied:
        batching = resolved["batching"]
        if not isinstance(batching, Mapping):
            raise ValueError("batching must be a mapping")
        batching = dict(batching)
        mode = str(batching.get("mode") or "fixed_resample")
        if mode not in {"fixed_resample", "native_variable_length"}:
            raise ValueError("batching.mode must be fixed_resample or native_variable_length")
        batching["mode"] = mode
        if mode == "native_variable_length":
            buckets = sorted(
                {int(value) for value in batching.get("length_buckets") or ()}
            )
            if not buckets or buckets[0] < 3:
                raise ValueError(
                    "native_variable_length batching needs length_buckets >= 3"
                )
            batching["length_buckets"] = buckets
            for field, default in DEFAULT_NATIVE_BATCHING.items():
                batching.setdefault(field, default)
            if batching["homogeneous_bucket_batches"] is not True:
                raise ValueError(
                    "native_variable_length training requires homogeneous bucket batches"
                )
            for field in (
                "max_motion_tokens_per_microbatch",
                "max_attention_elements_per_microbatch",
            ):
                value = batching[field]
                if isinstance(value, bool) or int(value) != value or int(value) <= 0:
                    raise ValueError(f"batching.{field} must be a positive integer")
                batching[field] = int(value)
            target = batching.get("target_effective_batch_size", resolved["batch_size"])
            if isinstance(target, bool) or int(target) != target or int(target) <= 0:
                raise ValueError(
                    "batching.target_effective_batch_size must be a positive integer"
                )
            batching["target_effective_batch_size"] = int(target)
            if batching["gradient_accumulation_mode"] != "dynamic_episode_weighted":
                raise ValueError(
                    "native batching requires dynamic_episode_weighted accumulation"
                )
            if batching["oversize_sequence_policy"] != "single_full_episode_or_fail":
                raise ValueError(
                    "native batching must preserve a full episode or fail before training"
                )
        resolved["batching"] = batching
    return resolved


def _nested_value(record: Mapping, *paths: tuple[str, ...]):
    for path in paths:
        value = record
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                value = None
                break
            value = value[key]
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def episode_group_keys(episode: Mapping) -> tuple[str, str]:
    speaker = _nested_value(
        episode,
        ("speaker_key",),
        ("speaker_id",),
        ("meta", "speaker_key"),
        ("source", "speaker_key"),
    )
    source_group = _nested_value(
        episode,
        ("source_group_key",),
        ("source_group_id",),
        ("source_clip_id",),
        ("split", "source_group_key"),
        ("source", "source_group_key"),
        ("source", "source_clip_id"),
    )
    clip_id = _episode_id(episode)
    if speaker is None:
        raise ValueError(f"BEAT episode {clip_id} is missing speaker_key")
    if source_group is None:
        raise ValueError(f"BEAT episode {clip_id} is missing source_group_key/source_clip_id")
    return speaker, source_group


def canonicalize_beat_episodes(episodes: Sequence[Mapping]) -> list[dict]:
    canonical = []
    seen = set()
    for episode in episodes:
        clip_id = _episode_id(episode)
        if clip_id in seen:
            raise ValueError(f"duplicate BEAT clip id: {clip_id}")
        actions = np.asarray(episode.get("actions"), dtype=np.float32)
        condition = np.asarray(episode.get("condition"), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] < 3 or actions.shape[1] != ACTION_DIM:
            raise ValueError(f"BEAT episode {clip_id} must contain [frames, 18] actions")
        if condition.shape != (KIMODO_V2_CONDITION_DIM,):
            raise ValueError(f"BEAT episode {clip_id} must contain a 264D attached condition")
        if not np.isfinite(actions).all() or not np.isfinite(condition).all():
            raise ValueError(f"BEAT episode {clip_id} contains non-finite values")
        fps, duration = _episode_fps_and_duration(episode)
        speaker, source_group = episode_group_keys(episode)
        emotion_mask = episode.get("emotion_supervision_mask")
        if not isinstance(emotion_mask, (bool, np.bool_)):
            raise ValueError(f"BEAT episode {clip_id} needs a boolean emotion_supervision_mask")
        emotion_id = episode.get("emotion_id")
        source_emotion_verified = episode.get("source_emotion_label_verified")
        if source_emotion_verified is None:
            if not bool(emotion_mask) and emotion_id not in (None, ""):
                raise ValueError(
                    f"unresolved BEAT emotion {clip_id} must not carry emotion_id"
                )
        else:
            if not isinstance(source_emotion_verified, (bool, np.bool_)):
                raise ValueError(
                    f"BEAT episode {clip_id} needs a boolean source emotion label gate"
                )
            affect_mask = episode.get("affect_observable_supervision_mask")
            conditioning_mask = episode.get("emotion_conditioning_mask")
            if not isinstance(affect_mask, (bool, np.bool_)) or not isinstance(
                conditioning_mask, (bool, np.bool_)
            ):
                raise ValueError(
                    f"BEAT episode {clip_id} needs explicit affect/conditioning masks"
                )
            if bool(source_emotion_verified) and emotion_id in (None, ""):
                raise ValueError(
                    f"verified source emotion label {clip_id} is missing emotion_id"
                )
            if not bool(source_emotion_verified) and emotion_id not in (None, ""):
                raise ValueError(
                    f"unverified source emotion label {clip_id} must not carry emotion_id"
                )
            expected_conditioning = bool(
                source_emotion_verified and affect_mask and emotion_mask
            )
            if bool(conditioning_mask) is not expected_conditioning:
                raise ValueError(
                    f"BEAT emotion conditioning mask {clip_id} violates the dual gate"
                )
        item = dict(episode)
        expression_turn_v8 = episode.get("formal_episode_contract") == (
            "beat2_expression_turn_v8_train_episode_v1"
        )
        item.update(
            {
                "clip_id": clip_id,
                "actions": np.ascontiguousarray(actions),
                "condition": np.ascontiguousarray(condition),
                "action_dim_mask": np.ones(ACTION_DIM, dtype=np.bool_),
                "domain": (
                    str(episode.get("dataset_source") or "expression_turn_v8")
                    if expression_turn_v8
                    else "beat2"
                ),
                "speaker_key": speaker,
                "source_group_key": source_group,
                "emotion_supervision_mask": bool(emotion_mask),
                "fps": fps,
                "duration_sec": duration,
                "frame_coverage_sec": float(actions.shape[0] / fps),
            }
        )
        canonical.append(item)
        seen.add(clip_id)
    if not canonical:
        raise ValueError("at least one BEAT 18D episode is required")
    return canonical


def canonicalize_kimodo_replay(episodes: Sequence[Mapping]) -> list[dict]:
    """Pad Kimodo 15D inputs while keeping head dimensions explicitly unobserved."""
    canonical = []
    seen = set()
    for index, episode in enumerate(episodes):
        episode_id = _episode_id(episode)
        clip_id = f"kimodo:{episode_id}"
        if clip_id in seen:
            raise ValueError(f"duplicate Kimodo replay id: {clip_id}")
        actions_15d = np.asarray(episode.get("actions"), dtype=np.float32)
        condition = np.asarray(episode.get("condition"), dtype=np.float32)
        if actions_15d.ndim != 2 or actions_15d.shape[0] < 3 or actions_15d.shape[1] != LEGACY_ACTION_DIM:
            raise ValueError(f"Kimodo replay {episode_id} must contain [frames, 15] actions")
        if condition.shape != (KIMODO_V2_CONDITION_DIM,):
            raise ValueError(f"Kimodo replay {episode_id} must contain a 264D condition")
        if not np.isfinite(actions_15d).all() or not np.isfinite(condition).all():
            raise ValueError(f"Kimodo replay {episode_id} contains non-finite values")
        fps, duration = _episode_fps_and_duration(episode)
        actions = np.zeros((actions_15d.shape[0], ACTION_DIM), dtype=np.float32)
        actions[:, :LEGACY_ACTION_DIM] = actions_15d
        dim_mask = np.zeros(ACTION_DIM, dtype=np.bool_)
        dim_mask[:LEGACY_ACTION_DIM] = True
        meta = episode.get("meta") or {}
        speaker = str(
            episode.get("speaker_key")
            or (meta.get("speaker_key") if isinstance(meta, Mapping) else None)
            or "kimodo_replay"
        )
        item = dict(episode)
        item.update(
            {
                "clip_id": clip_id,
                "actions": actions,
                "condition": np.ascontiguousarray(condition),
                "action_dim_mask": dim_mask,
                "domain": "kimodo",
                "speaker_key": speaker,
                "source_group_key": f"kimodo:{episode_id}",
                "emotion_supervision_mask": True,
                "head_target_policy": "unobserved_loss_mask_zero",
                "replay_source_validated": bool(episode.get("replay_source_validated", False)),
                "replay_original_index": index,
                "fps": fps,
                "duration_sec": duration,
                "frame_coverage_sec": float(actions_15d.shape[0] / fps),
            }
        )
        canonical.append(item)
        seen.add(clip_id)
    return canonical


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, value):
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, first, second):
        first_root, second_root = self.find(first), self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def validate_strict_group_splits(splits: Mapping[str, Sequence[Mapping]]) -> dict:
    if set(splits) != set(SPLIT_NAMES):
        raise ValueError(f"splits must define exactly {SPLIT_NAMES}")
    speaker_splits = defaultdict(set)
    source_splits = defaultdict(set)
    clip_ids = set()
    for split_name in SPLIT_NAMES:
        if not splits[split_name]:
            raise ValueError(f"strict split {split_name} is empty")
        for episode in splits[split_name]:
            clip_id = _episode_id(episode)
            if clip_id in clip_ids:
                raise ValueError(f"episode {clip_id} occurs in multiple splits")
            clip_ids.add(clip_id)
            speaker, source_group = episode_group_keys(episode)
            speaker_splits[speaker].add(split_name)
            source_splits[source_group].add(split_name)
    leaked_speakers = {key: sorted(value) for key, value in speaker_splits.items() if len(value) != 1}
    leaked_sources = {key: sorted(value) for key, value in source_splits.items() if len(value) != 1}
    if leaked_speakers or leaked_sources:
        raise ValueError(
            "speaker/source-group leakage across splits: "
            f"speakers={leaked_speakers}, source_groups={leaked_sources}"
        )
    return {
        "speaker_to_split": {key: next(iter(value)) for key, value in sorted(speaker_splits.items())},
        "source_group_to_split": {key: next(iter(value)) for key, value in sorted(source_splits.items())},
        "episode_count": len(clip_ids),
    }


def strict_group_split(
    episodes: Sequence[Mapping],
    *,
    seed=7,
    fractions: Mapping[str, float] | None = None,
) -> tuple[dict[str, list[dict]], dict]:
    """Split connected speaker/source-group components without leakage."""
    fractions = dict(fractions or DEFAULT_CONFIG["split_fractions"])
    if set(fractions) != set(SPLIT_NAMES) or not math.isclose(
        sum(float(fractions[name]) for name in SPLIT_NAMES), 1.0, abs_tol=1e-8
    ):
        raise ValueError(f"fractions must define {SPLIT_NAMES} and sum to one")
    canonical = canonicalize_beat_episodes(episodes)
    fixed_assignments = [episode.get("fixed_split_assignment") for episode in canonical]
    if any(value is not None for value in fixed_assignments):
        if not all(value in SPLIT_NAMES for value in fixed_assignments):
            raise ValueError(
                "fixed_split_assignment must be present and valid on every episode"
            )
        splits = {name: [] for name in SPLIT_NAMES}
        for episode, split_name in zip(canonical, fixed_assignments):
            splits[split_name].append(episode)
        for name in SPLIT_NAMES:
            splits[name] = sorted(splits[name], key=_episode_id)
        validation = validate_strict_group_splits(splits)
        records = [
            {
                "clip_id": _episode_id(episode),
                "speaker_key": episode["speaker_key"],
                "source_group_key": episode["source_group_key"],
                "split": split_name,
            }
            for split_name in SPLIT_NAMES
            for episode in splits[split_name]
        ]
        contract = {
            "contract_type": "speaker_source_group_strict_split",
            "contract_version": 1,
            "assignment_policy": "fixed_pre_quarantine_assignment",
            "seed": int(seed),
            "fractions": {name: float(fractions[name]) for name in SPLIT_NAMES},
            "counts": {name: len(splits[name]) for name in SPLIT_NAMES},
            "speaker_to_split": validation["speaker_to_split"],
            "source_group_to_split": validation["source_group_to_split"],
            "episodes": sorted(records, key=lambda row: row["clip_id"]),
        }
        contract["sha256"] = _json_hash(contract)
        return splits, contract
    union_find = _UnionFind()
    keys = []
    for episode in canonical:
        speaker, source_group = episode_group_keys(episode)
        speaker_node = f"speaker:{speaker}"
        source_node = f"source:{source_group}"
        union_find.union(speaker_node, source_node)
        keys.append((speaker_node, source_node))
    components = defaultdict(list)
    for episode, (speaker_node, _) in zip(canonical, keys):
        components[union_find.find(speaker_node)].append(episode)
    if len(components) < len(SPLIT_NAMES):
        raise ValueError(
            "strict speaker/source-group train/validation/test requires at least three "
            f"disconnected groups, found {len(components)}"
        )
    ordered = sorted(
        components.values(),
        key=lambda rows: (
            -len(rows),
            hashlib.sha256(
                f"{int(seed)}:{','.join(sorted(_episode_id(row) for row in rows))}".encode()
            ).hexdigest(),
        ),
    )
    total = len(canonical)
    targets = {name: float(fractions[name]) * total for name in SPLIT_NAMES}
    assigned = {name: 0 for name in SPLIT_NAMES}
    splits = {name: [] for name in SPLIT_NAMES}
    for component in ordered:
        split_name = min(
            SPLIT_NAMES,
            key=lambda name: (
                assigned[name] / max(targets[name], 1e-12),
                SPLIT_NAMES.index(name),
            ),
        )
        splits[split_name].extend(component)
        assigned[split_name] += len(component)
    for name in SPLIT_NAMES:
        splits[name] = sorted(splits[name], key=_episode_id)
    validation = validate_strict_group_splits(splits)
    records = [
        {
            "clip_id": _episode_id(episode),
            "speaker_key": episode["speaker_key"],
            "source_group_key": episode["source_group_key"],
            "split": split_name,
        }
        for split_name in SPLIT_NAMES
        for episode in splits[split_name]
    ]
    contract = {
        "contract_type": "speaker_source_group_strict_split",
        "contract_version": 1,
        "seed": int(seed),
        "fractions": {name: float(fractions[name]) for name in SPLIT_NAMES},
        "counts": {name: len(splits[name]) for name in SPLIT_NAMES},
        "speaker_to_split": validation["speaker_to_split"],
        "source_group_to_split": validation["source_group_to_split"],
        "episodes": sorted(records, key=lambda row: row["clip_id"]),
    }
    contract["sha256"] = _json_hash(contract)
    return splits, contract


class DomainSpeakerBalancedSampler:
    """Round-robin domains and speakers, then sample within a speaker bucket."""

    def __init__(self, episodes: Sequence[Mapping], *, seed=7):
        if not episodes:
            raise ValueError("balanced sampler requires episodes")
        self.episodes = list(episodes)
        self.buckets = defaultdict(lambda: defaultdict(list))
        for index, episode in enumerate(self.episodes):
            domain = str(episode.get("domain") or "unknown")
            speaker = str(episode.get("speaker_key") or "unknown")
            self.buckets[domain][speaker].append(index)
        self.domains = sorted(self.buckets)
        self.speakers = {domain: sorted(self.buckets[domain]) for domain in self.domains}
        self.domain_cursor = 0
        self.speaker_cursor = {domain: 0 for domain in self.domains}
        self.rng = random.Random(int(seed))

    def sample(self, batch_size: int) -> list[dict]:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        selected = []
        for _ in range(int(batch_size)):
            domain = self.domains[self.domain_cursor % len(self.domains)]
            self.domain_cursor += 1
            speaker_list = self.speakers[domain]
            speaker = speaker_list[self.speaker_cursor[domain] % len(speaker_list)]
            self.speaker_cursor[domain] += 1
            selected.append(self.episodes[self.rng.choice(self.buckets[domain][speaker])])
        return selected

    def state_dict(self) -> dict:
        return {
            "domain_cursor": int(self.domain_cursor),
            "speaker_cursor": dict(self.speaker_cursor),
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state: Mapping) -> None:
        self.domain_cursor = int(state["domain_cursor"])
        stored = dict(state["speaker_cursor"])
        if set(stored) != set(self.speaker_cursor):
            raise ValueError("sampler domain set changed during resume")
        self.speaker_cursor = {name: int(value) for name, value in stored.items()}
        self.rng.setstate(state["rng_state"])


def _episode_dataset_source(episode: Mapping) -> str:
    return str(
        episode.get("dataset_source")
        or episode.get("source_dataset")
        or episode.get("dataset_id")
        or episode.get("domain")
        or "unknown"
    )


def head_activity_rms_velocity(episode: Mapping) -> float | None:
    mask = np.asarray(episode.get("action_dim_mask"), dtype=np.bool_)
    if mask.shape != (ACTION_DIM,) or not mask[LEGACY_ACTION_DIM:].all():
        return None
    actions = np.asarray(episode.get("actions"), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] < 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"episode {_episode_id(episode)} has invalid actions for activity")
    _, duration = _episode_fps_and_duration(episode)
    dt = max(1e-4, duration / float(actions.shape[0] - 1))
    velocity = np.diff(actions[:, LEGACY_ACTION_DIM:], axis=0) / dt
    return float(np.sqrt(np.mean(np.square(velocity), dtype=np.float64)))


class SourceSpeakerActivityBalancedSampler:
    """Balance domain, dataset source, speaker, head activity, and source clip."""

    STATE_VERSION = 1

    def __init__(
        self,
        episodes: Sequence[Mapping],
        *,
        activity_bin_edges_rad_s: Sequence[float],
        seed=7,
    ):
        if not episodes:
            raise ValueError("balanced sampler requires episodes")
        edges = tuple(float(value) for value in activity_bin_edges_rad_s)
        if (
            len(edges) != 3
            or any(not math.isfinite(value) or value < 0 for value in edges)
            or list(edges) != sorted(set(edges))
        ):
            raise ValueError("activity bin edges must contain three increasing values")
        self.episodes = list(episodes)
        self.activity_bin_edges_rad_s = edges
        self.buckets = defaultdict(list)
        for index, episode in enumerate(self.episodes):
            domain = str(episode.get("domain") or "unknown")
            dataset_source = _episode_dataset_source(episode)
            speaker = str(episode.get("speaker_key") or "unknown")
            source_group = str(episode.get("source_group_key") or _episode_id(episode))
            activity = head_activity_rms_velocity(episode)
            if activity is None:
                activity_bin = "head_unobserved"
            else:
                activity_bin = f"activity_{sum(activity >= edge for edge in edges)}"
            self.buckets[
                (domain, dataset_source, speaker, activity_bin, source_group)
            ].append(index)
        for key in self.buckets:
            self.buckets[key].sort(key=lambda index: _episode_id(self.episodes[index]))

        self.domains = sorted({key[0] for key in self.buckets})
        self.datasets = {
            domain: sorted({key[1] for key in self.buckets if key[0] == domain})
            for domain in self.domains
        }
        self.speakers = {
            (domain, dataset): sorted(
                {
                    key[2]
                    for key in self.buckets
                    if key[:2] == (domain, dataset)
                }
            )
            for domain in self.domains
            for dataset in self.datasets[domain]
        }
        self.activities = {
            (domain, dataset, speaker): sorted(
                {
                    key[3]
                    for key in self.buckets
                    if key[:3] == (domain, dataset, speaker)
                }
            )
            for (domain, dataset), speakers in self.speakers.items()
            for speaker in speakers
        }
        self.source_groups = {
            (domain, dataset, speaker, activity): sorted(
                {
                    key[4]
                    for key in self.buckets
                    if key[:4] == (domain, dataset, speaker, activity)
                }
            )
            for (domain, dataset, speaker), activities in self.activities.items()
            for activity in activities
        }
        self.domain_cursor = 0
        self.dataset_cursor = {domain: 0 for domain in self.domains}
        self.speaker_cursor = {key: 0 for key in self.speakers}
        self.activity_cursor = {key: 0 for key in self.activities}
        self.source_cursor = {key: 0 for key in self.source_groups}
        self.rng = random.Random(int(seed))
        membership = [
            {
                "bucket": list(key),
                "clip_ids": sorted(_episode_id(self.episodes[index]) for index in indices),
            }
            for key, indices in sorted(self.buckets.items())
        ]
        self.structure_sha256 = _json_hash(
            {"activity_bin_edges_rad_s": list(edges), "membership": membership}
        )

    @staticmethod
    def _next(options, cursors, key):
        cursor = cursors[key]
        value = options[cursor % len(options)]
        cursors[key] = cursor + 1
        return value

    def sample(self, batch_size: int) -> list[dict]:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        selected = []
        for _ in range(int(batch_size)):
            domain = self.domains[self.domain_cursor % len(self.domains)]
            self.domain_cursor += 1
            dataset = self._next(self.datasets[domain], self.dataset_cursor, domain)
            dataset_key = (domain, dataset)
            speaker = self._next(
                self.speakers[dataset_key], self.speaker_cursor, dataset_key
            )
            speaker_key = (domain, dataset, speaker)
            activity = self._next(
                self.activities[speaker_key], self.activity_cursor, speaker_key
            )
            activity_key = (domain, dataset, speaker, activity)
            source_group = self._next(
                self.source_groups[activity_key], self.source_cursor, activity_key
            )
            bucket = self.buckets[activity_key + (source_group,)]
            selected.append(self.episodes[self.rng.choice(bucket)])
        return selected

    def state_dict(self) -> dict:
        return {
            "state_version": self.STATE_VERSION,
            "structure_sha256": self.structure_sha256,
            "domain_cursor": int(self.domain_cursor),
            "dataset_cursor": dict(self.dataset_cursor),
            "speaker_cursor": dict(self.speaker_cursor),
            "activity_cursor": dict(self.activity_cursor),
            "source_cursor": dict(self.source_cursor),
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state: Mapping) -> None:
        if state.get("state_version") != self.STATE_VERSION:
            raise ValueError("sampler state version changed during resume")
        if state.get("structure_sha256") != self.structure_sha256:
            raise ValueError("sampler episode/source/activity structure changed during resume")
        for name, current in (
            ("dataset_cursor", self.dataset_cursor),
            ("speaker_cursor", self.speaker_cursor),
            ("activity_cursor", self.activity_cursor),
            ("source_cursor", self.source_cursor),
        ):
            stored = dict(state[name])
            if set(stored) != set(current):
                raise ValueError(f"sampler {name} keys changed during resume")
            current.update({key: int(value) for key, value in stored.items()})
        self.domain_cursor = int(state["domain_cursor"])
        self.rng.setstate(state["rng_state"])


def native_length_bucket(frame_count: int, buckets: Sequence[int]) -> int:
    """Map a full native sequence to the same bucket used by collation."""

    frame_count = int(frame_count)
    if frame_count < 3:
        raise ValueError("native sequence must contain at least three frames")
    candidates = sorted({int(value) for value in buckets if int(value) >= 3})
    if not candidates:
        raise ValueError("native length buckets must contain a value of at least three")
    selected = next((value for value in candidates if value >= frame_count), None)
    return selected if selected is not None else int(math.ceil(frame_count / 32.0) * 32)


def native_length_microbatch_capacity(
    bucket_frames: int,
    *,
    semantic_tokens: int,
    max_batch_size: int,
    max_motion_tokens: int,
    max_attention_elements: int,
) -> dict:
    """Return a fail-closed microbatch capacity without shortening a sequence."""

    values = {
        "bucket_frames": bucket_frames,
        "semantic_tokens": semantic_tokens,
        "max_batch_size": max_batch_size,
        "max_motion_tokens": max_motion_tokens,
        "max_attention_elements": max_attention_elements,
    }
    if any(isinstance(value, bool) or int(value) != value or int(value) <= 0 for value in values.values()):
        raise ValueError("native microbatch limits must be positive integers")
    bucket_frames = int(bucket_frames)
    semantic_tokens = int(semantic_tokens)
    sequence_tokens = bucket_frames + semantic_tokens
    by_motion = int(max_motion_tokens) // bucket_frames
    per_episode_attention = sequence_tokens * sequence_tokens
    by_attention = int(max_attention_elements) // per_episode_attention
    capacity = min(int(max_batch_size), by_motion, by_attention)
    if capacity < 1:
        raise ValueError(
            "full native sequence exceeds the single-episode token/attention budget: "
            f"bucket={bucket_frames}, sequence_tokens={sequence_tokens}, "
            f"motion_budget={int(max_motion_tokens)}, "
            f"attention_budget={int(max_attention_elements)}"
        )
    return {
        "bucket_frames": bucket_frames,
        "sequence_tokens": sequence_tokens,
        "per_episode_attention_elements": per_episode_attention,
        "capacity": capacity,
        "capacity_by_motion_tokens": by_motion,
        "capacity_by_attention_elements": by_attention,
        "max_motion_tokens": int(max_motion_tokens),
        "max_attention_elements": int(max_attention_elements),
    }


class NativeLengthBucketSampler:
    """Sample one length bucket per microbatch with exact resumable state."""

    STATE_VERSION = 1

    def __init__(
        self,
        episodes: Sequence[Mapping],
        *,
        buckets: Sequence[int],
        sampler_config: Mapping | None,
        seed: int,
    ):
        if not episodes:
            raise ValueError("native bucket sampler requires episodes")
        self.buckets = tuple(sorted({int(value) for value in buckets if int(value) >= 3}))
        if not self.buckets:
            raise ValueError("native bucket sampler requires length buckets")
        self.sampler_config = dict(sampler_config or {})
        self.seed = int(seed)
        grouped = defaultdict(list)
        for episode in episodes:
            frames = int(np.asarray(episode.get("actions")).shape[0])
            grouped[native_length_bucket(frames, self.buckets)].append(episode)
        self.bucket_episodes = {
            bucket: sorted(rows, key=_episode_id) for bucket, rows in sorted(grouped.items())
        }
        self.samplers = {
            bucket: self._make_balanced_sampler(
                rows,
                seed=self.seed
                + int.from_bytes(
                    hashlib.sha256(f"bucket:{bucket}".encode("ascii")).digest()[:4],
                    byteorder="big",
                ),
            )
            for bucket, rows in self.bucket_episodes.items()
        }
        membership = {
            str(bucket): [_episode_id(episode) for episode in rows]
            for bucket, rows in self.bucket_episodes.items()
        }
        self.structure_sha256 = _json_hash(
            {
                "buckets": list(self.buckets),
                "membership": membership,
                "sampler_config": self.sampler_config,
            }
        )
        self.schedule_rng = random.Random(self.seed + 7919)
        self.bucket_schedule = [
            bucket
            for bucket, rows in self.bucket_episodes.items()
            for _ in range(len(rows))
        ]
        self.schedule_rng.shuffle(self.bucket_schedule)
        self.schedule_cursor = 0

    def _make_balanced_sampler(self, episodes, *, seed):
        if self.sampler_config.get("mode") in (None, "domain_speaker"):
            return DomainSpeakerBalancedSampler(episodes, seed=seed)
        return SourceSpeakerActivityBalancedSampler(
            episodes,
            activity_bin_edges_rad_s=self.sampler_config[
                "activity_bin_edges_rad_s"
            ],
            seed=seed,
        )

    def _next_bucket(self) -> int:
        if self.schedule_cursor >= len(self.bucket_schedule):
            self.schedule_rng.shuffle(self.bucket_schedule)
            self.schedule_cursor = 0
        bucket = int(self.bucket_schedule[self.schedule_cursor])
        self.schedule_cursor += 1
        return bucket

    def validate_budgets(
        self, *, semantic_tokens: int, max_batch_size: int, batching: Mapping
    ) -> dict:
        plans = {
            bucket: native_length_microbatch_capacity(
                bucket,
                semantic_tokens=semantic_tokens,
                max_batch_size=max_batch_size,
                max_motion_tokens=batching["max_motion_tokens_per_microbatch"],
                max_attention_elements=batching[
                    "max_attention_elements_per_microbatch"
                ],
            )
            for bucket in self.bucket_episodes
        }
        return {
            "bucket_episode_counts": {
                str(bucket): len(rows) for bucket, rows in self.bucket_episodes.items()
            },
            "bucket_plans": {str(bucket): plan for bucket, plan in plans.items()},
            "maximum_native_frame_count": max(
                int(np.asarray(episode["actions"]).shape[0])
                for rows in self.bucket_episodes.values()
                for episode in rows
            ),
            "maximum_bucket_frames": max(self.bucket_episodes),
            "no_cropping": True,
        }

    def sample_microbatch(
        self,
        *,
        remaining_effective_batch: int,
        semantic_tokens: int,
        max_batch_size: int,
        batching: Mapping,
    ) -> tuple[list[dict], dict]:
        if int(remaining_effective_batch) <= 0:
            raise ValueError("remaining effective batch must be positive")
        bucket = self._next_bucket()
        plan = native_length_microbatch_capacity(
            bucket,
            semantic_tokens=semantic_tokens,
            max_batch_size=max_batch_size,
            max_motion_tokens=batching["max_motion_tokens_per_microbatch"],
            max_attention_elements=batching[
                "max_attention_elements_per_microbatch"
            ],
        )
        microbatch_size = min(int(remaining_effective_batch), plan["capacity"])
        selected = self.samplers[bucket].sample(microbatch_size)
        observed_buckets = {
            native_length_bucket(
                int(np.asarray(episode["actions"]).shape[0]), self.buckets
            )
            for episode in selected
        }
        if observed_buckets != {bucket}:
            raise RuntimeError("native bucket sampler produced a mixed-length microbatch")
        return selected, plan | {
            "microbatch_size": microbatch_size,
            "motion_tokens": microbatch_size * bucket,
            "attention_elements": (
                microbatch_size * plan["per_episode_attention_elements"]
            ),
        }

    def state_dict(self) -> dict:
        return {
            "state_version": self.STATE_VERSION,
            "structure_sha256": self.structure_sha256,
            "bucket_schedule": list(self.bucket_schedule),
            "schedule_cursor": int(self.schedule_cursor),
            "schedule_rng_state": self.schedule_rng.getstate(),
            "bucket_sampler_states": {
                bucket: sampler.state_dict() for bucket, sampler in self.samplers.items()
            },
        }

    def load_state_dict(self, state: Mapping) -> None:
        if state.get("state_version") != self.STATE_VERSION:
            raise ValueError("native bucket sampler state version changed during resume")
        if state.get("structure_sha256") != self.structure_sha256:
            raise ValueError("native bucket sampler membership changed during resume")
        schedule = [int(value) for value in state.get("bucket_schedule") or ()]
        if sorted(schedule) != sorted(self.bucket_schedule):
            raise ValueError("native bucket schedule changed during resume")
        cursor = int(state.get("schedule_cursor", -1))
        if not 0 <= cursor <= len(schedule):
            raise ValueError("native bucket schedule cursor is invalid")
        stored_sampler_states = dict(state.get("bucket_sampler_states") or {})
        if set(stored_sampler_states) != set(self.samplers):
            raise ValueError("native bucket sampler set changed during resume")
        self.bucket_schedule = schedule
        self.schedule_cursor = cursor
        self.schedule_rng.setstate(state["schedule_rng_state"])
        for bucket, sampler in self.samplers.items():
            sampler.load_state_dict(stored_sampler_states[bucket])


class ModelEMA:
    def __init__(self, model, decay: float):
        self.decay = float(decay)
        self.shadow = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model) -> None:
        for name, value in model.state_dict().items():
            current = value.detach()
            if torch.is_floating_point(current):
                self.shadow[name].mul_(self.decay).add_(
                    current, alpha=1.0 - self.decay
                )
            else:
                self.shadow[name].copy_(current)

    @contextmanager
    def apply(self, model):
        original = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        model.load_state_dict(self.shadow, strict=True)
        try:
            yield model
        finally:
            model.load_state_dict(original, strict=True)


def _sampler_for_config(episodes: Sequence[Mapping], config: Mapping, *, seed: int):
    sampler_config = config.get("sampler")
    batching = config.get("batching")
    if (
        isinstance(batching, Mapping)
        and batching.get("mode") == "native_variable_length"
        and batching.get("homogeneous_bucket_batches") is True
    ):
        return NativeLengthBucketSampler(
            episodes,
            buckets=batching["length_buckets"],
            sampler_config=sampler_config,
            seed=seed,
        )
    if not isinstance(sampler_config, Mapping) or sampler_config.get("mode") in (
        None,
        "domain_speaker",
    ):
        return DomainSpeakerBalancedSampler(episodes, seed=seed)
    return SourceSpeakerActivityBalancedSampler(
        episodes,
        activity_bin_edges_rad_s=sampler_config["activity_bin_edges_rad_s"],
        seed=seed,
    )


@torch.no_grad()
def _restore_frozen_ema_state(ema: ModelEMA, policy) -> None:
    for name, frozen in policy.frozen_state.items():
        destination = ema.shadow[name]
        source = frozen.to(device=destination.device, dtype=destination.dtype)
        if name == "input.weight":
            destination[:, :LEGACY_ACTION_DIM].copy_(
                source[:, :LEGACY_ACTION_DIM]
            )
        elif name in ("output.weight", "output.bias"):
            destination[:LEGACY_ACTION_DIM].copy_(source[:LEGACY_ACTION_DIM])
        else:
            destination.copy_(source)


def resample_trajectory(actions, frame_count: int) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float32)
    frame_count = int(frame_count)
    if values.ndim != 2 or values.shape[1] != ACTION_DIM or values.shape[0] < 3:
        raise ValueError("actions must have shape [frames>=3, 18]")
    if frame_count < 3:
        raise ValueError("frame_count must be at least 3")
    if values.shape[0] == frame_count:
        return np.ascontiguousarray(values)
    source = np.linspace(0.0, 1.0, values.shape[0], dtype=np.float64)
    target = np.linspace(0.0, 1.0, frame_count, dtype=np.float64)
    output = np.empty((frame_count, ACTION_DIM), dtype=np.float32)
    for dimension in range(ACTION_DIM):
        output[:, dimension] = np.interp(target, source, values[:, dimension])
    return output


def _normalize_masked_actions(actions, dim_mask, action_stats):
    mean = np.asarray(action_stats["mean"], dtype=np.float32)
    std = np.asarray(action_stats["std"], dtype=np.float32)
    if mean.shape != (ACTION_DIM,) or std.shape != (ACTION_DIM,) or np.any(std <= 0):
        raise ValueError("18D checkpoint action stats are invalid")
    normalized = (actions - mean[None, :]) / std[None, :]
    # Zero is a missing-data sentinel only after normalization.  This avoids
    # treating a physical head angle of zero as replay ground truth.
    return normalized * dim_mask[None, :]


def batch_tensors(
    episodes: Sequence[Mapping], *, frame_count: int, action_stats: Mapping, device
):
    actions, conditions, masks, durations = [], [], [], []
    for episode in episodes:
        values = resample_trajectory(episode["actions"], frame_count)
        mask = np.asarray(episode.get("action_dim_mask"), dtype=np.bool_)
        if mask.shape != (ACTION_DIM,) or not mask[:LEGACY_ACTION_DIM].all():
            raise ValueError(f"episode {_episode_id(episode)} has an invalid 18D loss mask")
        actions.append(_normalize_masked_actions(values, mask, action_stats))
        conditions.append(np.asarray(episode["condition"], dtype=np.float32))
        masks.append(mask)
        _, duration = _episode_fps_and_duration(episode)
        durations.append(duration)
    return (
        torch.as_tensor(np.stack(actions), dtype=torch.float32, device=device),
        torch.as_tensor(np.stack(conditions), dtype=torch.float32, device=device),
        torch.as_tensor(np.stack(masks), dtype=torch.bool, device=device),
        torch.as_tensor(durations, dtype=torch.float32, device=device),
    )


def native_variable_length_batch_tensors(
    episodes: Sequence[Mapping], *, buckets: Sequence[int], action_stats: Mapping, device
):
    """Collate native 30 Hz clips without resampling or learning from padding."""
    from upper_body_skeleton.ula_v2_18d_random_init import (
        collate_variable_length_18d,
    )

    batch = collate_variable_length_18d(episodes, buckets=buckets)
    actions = batch["actions"].to(device=device)
    frame_valid = batch["frame_valid_mask"].to(device=device)
    dim_mask = batch["action_dim_mask"].to(device=device)
    if torch.any(dim_mask[:, :LEGACY_ACTION_DIM] == 0):
        raise ValueError("native variable-length batch has an invalid 15D body mask")
    mean = torch.as_tensor(action_stats["mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(action_stats["std"], dtype=torch.float32, device=device)
    if mean.shape != (ACTION_DIM,) or std.shape != (ACTION_DIM,) or torch.any(std <= 0):
        raise ValueError("18D checkpoint action stats are invalid")
    observed = frame_valid[:, :, None] & dim_mask[:, None, :]
    actions = ((actions - mean[None, None, :]) / std[None, None, :]) * observed
    conditions = torch.as_tensor(
        np.stack([episode["condition"] for episode in episodes]),
        dtype=torch.float32,
        device=device,
    )
    return (
        actions,
        conditions,
        dim_mask,
        batch["durations_sec"].to(device=device),
        frame_valid,
    )


def _batch_tensors_for_config(
    episodes: Sequence[Mapping],
    *,
    frame_count: int,
    action_stats: Mapping,
    device,
    batching: Mapping | None,
):
    batching = dict(batching or {})
    if batching.get("mode") == "native_variable_length":
        return native_variable_length_batch_tensors(
            episodes,
            buckets=batching["length_buckets"],
            action_stats=action_stats,
            device=device,
        )
    actions, conditions, masks, durations = batch_tensors(
        episodes,
        frame_count=frame_count,
        action_stats=action_stats,
        device=device,
    )
    return actions, conditions, masks, durations, None


def _transition_targets_for_episodes(
    episodes: Sequence[Mapping], *, device
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = []
    masks = []
    valid_ids = set(TRANSITION_IDS.values())
    for episode in episodes:
        supervised = episode.get("transition_supervision_mask") is True
        value = episode.get("transition_id")
        if not supervised:
            targets.append(TRANSITION_IDS["continue"])
            masks.append(False)
            continue
        if value is None:
            raise ValueError(
                f"{_episode_id(episode)}: verified transition target is missing"
            )
        if isinstance(value, str):
            if value not in TRANSITION_IDS:
                raise ValueError(
                    f"{_episode_id(episode)}: unknown planner transition {value!r}"
                )
            value = TRANSITION_IDS[value]
        if isinstance(value, bool) or int(value) not in valid_ids:
            raise ValueError(
                f"{_episode_id(episode)}: invalid planner transition target {value!r}"
            )
        targets.append(int(value))
        masks.append(True)
    return (
        torch.tensor(targets, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.bool, device=device),
    )


def transition_supervision_contract(episodes: Sequence[Mapping]) -> dict:
    """Validate that transition labels come from real adjacent sequence context."""
    supervised = []
    for episode in episodes:
        clip_id = _episode_id(episode)
        masked = episode.get("transition_supervision_mask") is True
        value = episode.get("transition_id")
        if not masked:
            if value is not None:
                raise ValueError(
                    f"{clip_id}: transition_id cannot exist without transition_supervision_mask=true"
                )
            continue
        if episode.get("transition_label_source") != "verified_adjacent_sequence":
            raise ValueError(
                f"{clip_id}: supervised transition lacks verified adjacent-sequence provenance"
            )
        sequence_id = str(episode.get("transition_sequence_id") or "").strip()
        sequence_index = episode.get("transition_sequence_index")
        if not sequence_id or isinstance(sequence_index, bool) or not isinstance(
            sequence_index, int
        ) or sequence_index < 0:
            raise ValueError(f"{clip_id}: transition sequence identity/index is invalid")
        if isinstance(value, str):
            if value not in TRANSITION_IDS:
                raise ValueError(f"{clip_id}: unknown transition label {value!r}")
            label = value
        elif not isinstance(value, bool) and int(value) in set(TRANSITION_IDS.values()):
            label = next(name for name, index in TRANSITION_IDS.items() if index == int(value))
        else:
            raise ValueError(f"{clip_id}: invalid transition label {value!r}")
        if label != "end" and not str(episode.get("transition_next_clip_id") or "").strip():
            raise ValueError(f"{clip_id}: non-end transition lacks its verified next clip")
        supervised.append(label)
    if supervised and set(supervised) == {"end"}:
        raise ValueError(
            "all verified transition labels are end; refusing single-class transition training"
        )
    counts = {name: supervised.count(name) for name in TRANSITION_IDS}
    enabled = bool(supervised)
    return {
        "duration_head_supervision": "native_output_sample_span_(N-1)/fps",
        "duration_head_trainable": True,
        "transition_supervised_episode_count": len(supervised),
        "transition_class_counts": counts,
        "transition_head_trainable": enabled,
        "transition_head_status": (
            "trainable_verified_adjacent_sequence_labels"
            if enabled
            else "untrained_no_verified_adjacent_sequence_labels"
        ),
        "transition_inference_enabled": enabled,
        "missing_transition_policy": "mask_false_no_default_end_label",
    }


def _default_style_evaluation_episodes(
    episodes: Sequence[Mapping],
) -> list[dict]:
    from upper_body_skeleton.ula_v2_18d_random_init import (
        default_style_evaluation_conditions,
    )

    result = []
    for episode in episodes:
        item = dict(episode)
        item["condition"] = default_style_evaluation_conditions(
            episode["condition"]
        )
        item["evaluation_conditioning_policy"] = (
            "motion_only_zero_text_semantics_default_style_non_oracle"
            if episode.get("formal_episode_contract")
            == MOTION_ONLY_EPISODE_CONTRACT
            else "text_explicit_semantics_default_style_non_oracle"
        )
        result.append(item)
    return result


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = torch.broadcast_to(mask, values.shape).to(values.dtype)
    denominator = expanded.sum()
    if float(denominator.detach().cpu()) <= 0:
        return values.new_zeros(())
    return (values * expanded).sum() / denominator


def _derivatives(
    actions: torch.Tensor,
    durations: torch.Tensor,
    frame_counts: torch.Tensor | None = None,
):
    frame_count = actions.shape[1]
    if frame_counts is None:
        intervals = float(max(1, frame_count - 1))
    else:
        intervals = (frame_counts.to(durations.dtype) - 1.0).clamp_min(1.0)
    dt = (durations / intervals).clamp_min(1e-4)[:, None, None]
    velocity = (actions[:, 1:] - actions[:, :-1]) / dt
    acceleration = (
        (velocity[:, 1:] - velocity[:, :-1]) / dt
        if frame_count > 2
        else velocity[:, :0]
    )
    jerk = (
        (acceleration[:, 1:] - acceleration[:, :-1]) / dt
        if frame_count > 3
        else acceleration[:, :0]
    )
    return velocity, acceleration, jerk


def masked_18d_objective(
    model,
    actions: torch.Tensor,
    conditions: torch.Tensor,
    dim_mask: torch.Tensor,
    durations: torch.Tensor,
    *,
    loss_weights: Mapping[str, float] | None = None,
    teacher_model=None,
    generator=None,
    frame_valid_mask: torch.Tensor | None = None,
    transition_targets: torch.Tensor | None = None,
    transition_supervision_mask: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    flow_times: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Flow and physical derivative losses with joint and optional frame masks.

    ``noise`` and ``flow_times`` may be supplied together for paired
    counterfactual objectives.  In that case two condition variants can be
    compared at the exact same flow state instead of accidentally comparing
    different random draws.
    """
    if actions.ndim != 3 or actions.shape[-1] != ACTION_DIM:
        raise ValueError("actions must have shape [batch, frames, 18]")
    if conditions.shape != (actions.shape[0], KIMODO_V2_CONDITION_DIM):
        raise ValueError("conditions must have shape [batch, 264]")
    if dim_mask.shape != (actions.shape[0], ACTION_DIM) or dim_mask.dtype != torch.bool:
        raise ValueError("dim_mask must be boolean [batch, 18]")
    if durations.shape != (actions.shape[0],) or torch.any(durations <= 0):
        raise ValueError("durations must be positive [batch]")
    weights = dict(DEFAULT_CONFIG["loss"] | dict(loss_weights or {}))
    if frame_valid_mask is None:
        observed = dim_mask[:, None, :]
        frame_counts = None
        velocity_mask = acceleration_mask = jerk_mask = observed
    else:
        if (
            frame_valid_mask.shape != actions.shape[:2]
            or frame_valid_mask.dtype != torch.bool
        ):
            raise ValueError("frame_valid_mask must be boolean [batch, frames]")
        frame_counts = frame_valid_mask.sum(dim=1)
        if torch.any(frame_counts < 3):
            raise ValueError("every variable-length episode needs at least three frames")
        observed = frame_valid_mask[:, :, None] & dim_mask[:, None, :]
        velocity_valid = frame_valid_mask[:, 1:] & frame_valid_mask[:, :-1]
        acceleration_valid = (
            frame_valid_mask[:, 2:]
            & frame_valid_mask[:, 1:-1]
            & frame_valid_mask[:, :-2]
        )
        jerk_valid = (
            frame_valid_mask[:, 3:]
            & frame_valid_mask[:, 2:-1]
            & frame_valid_mask[:, 1:-2]
            & frame_valid_mask[:, :-3]
        )
        velocity_mask = velocity_valid[:, :, None] & dim_mask[:, None, :]
        acceleration_mask = acceleration_valid[:, :, None] & dim_mask[:, None, :]
        jerk_mask = jerk_valid[:, :, None] & dim_mask[:, None, :]
    if noise is not None:
        if (
            noise.shape != actions.shape
            or noise.dtype != actions.dtype
            or noise.device != actions.device
            or not torch.isfinite(noise).all()
        ):
            raise ValueError(
                "explicit noise must be finite and match actions shape/dtype/device"
            )
        noise = noise.clone()
    elif frame_valid_mask is None:
        if generator is None:
            noise = torch.randn_like(actions)
        else:
            noise = torch.randn(
                actions.shape,
                dtype=actions.dtype,
                device=actions.device,
                generator=generator,
            )
    else:
        # Draw only native valid samples, so changing a padding bucket cannot
        # alter this episode or shift RNG draws for the next batch row.
        noise = torch.zeros_like(actions)
        for row, count in enumerate(frame_counts.tolist()):
            noise[row, :count] = torch.randn(
                (count, actions.shape[-1]),
                dtype=actions.dtype,
                device=actions.device,
                generator=generator,
            )
    if flow_times is not None:
        if (
            flow_times.shape != (actions.shape[0],)
            or flow_times.dtype != actions.dtype
            or flow_times.device != actions.device
            or not torch.isfinite(flow_times).all()
            or torch.any(flow_times < 0.0)
            or torch.any(flow_times > 1.0)
        ):
            raise ValueError(
                "explicit flow_times must be finite [batch] values in [0, 1] "
                "matching actions dtype/device"
            )
        t = flow_times
    else:
        t = torch.rand(
            actions.shape[0],
            dtype=actions.dtype,
            device=actions.device,
            generator=generator,
        )
    noise = noise * observed
    x_t = ((1.0 - t[:, None, None]) * noise + t[:, None, None] * actions) * observed
    target = (actions - noise) * observed
    if frame_valid_mask is None:
        predicted = model(x_t, t, conditions)
    else:
        from upper_body_skeleton.ula_v2_18d_random_init import (
            forward_with_frame_mask,
        )

        predicted = forward_with_frame_mask(
            model, x_t, t, conditions, frame_valid_mask
        )
    reconstructed = x_t + (1.0 - t[:, None, None]) * predicted
    flow = _masked_mean((predicted - target).square(), observed)
    position = _masked_mean(
        F.smooth_l1_loss(reconstructed, actions, reduction="none"), observed
    )
    reconstructed_velocity, reconstructed_acceleration, reconstructed_jerk = _derivatives(
        reconstructed, durations, frame_counts
    )
    target_velocity, target_acceleration, target_jerk = _derivatives(
        actions, durations, frame_counts
    )
    velocity = _masked_mean(
        F.smooth_l1_loss(
            reconstructed_velocity, target_velocity, reduction="none"
        ),
        velocity_mask,
    )
    acceleration = _masked_mean(
        F.smooth_l1_loss(
            reconstructed_acceleration, target_acceleration, reduction="none"
        ),
        acceleration_mask,
    )
    if teacher_model is None:
        body = predicted.new_zeros(())
    else:
        with torch.no_grad():
            if frame_valid_mask is None:
                teacher = teacher_model(x_t, t, conditions)
            else:
                teacher = forward_with_frame_mask(
                    teacher_model, x_t, t, conditions, frame_valid_mask
                )
        body_mask = observed[..., :LEGACY_ACTION_DIM]
        body = _masked_mean(
            (predicted[..., :LEGACY_ACTION_DIM] - teacher[..., :LEGACY_ACTION_DIM]).square(),
            body_mask,
        )
    losses = {
        "flow": flow,
        "position": position,
        "body": body,
        "velocity": velocity,
        "acceleration": acceleration,
    }
    requested_optional = OPTIONAL_LOSS_NAMES.intersection(loss_weights or {})
    if "planner" in requested_optional:
        if transition_targets is not None and transition_targets.shape != (actions.shape[0],):
            raise ValueError("planner transition targets must match the batch")
        losses["planner"] = planner_loss(
            model,
            conditions,
            durations,
            transition_targets,
            transition_supervision_mask=transition_supervision_mask,
        )
    if "planner_duration" in requested_optional:
        losses["planner_duration"] = planner_duration_loss(
            model, conditions, durations
        )
    if (
        "planner_transition" in requested_optional
        and float(weights.get("planner_transition", 0.0)) > 0.0
    ):
        if transition_targets is None or transition_targets.shape != (actions.shape[0],):
            raise ValueError("planner_transition requires one target per episode")
        losses["planner_transition"] = planner_transition_loss(
            model,
            conditions,
            transition_targets,
            transition_supervision_mask=transition_supervision_mask,
        )
    if "jerk" in requested_optional:
        losses["jerk"] = _masked_mean(
            F.smooth_l1_loss(reconstructed_jerk, target_jerk, reduction="none"),
            jerk_mask,
        )
    if requested_optional.intersection(
        {
            "head_flow",
            "head_position",
            "head_velocity",
            "head_acceleration",
            "head_jerk",
        }
    ):
        head_masks = {
            "head_flow": observed[..., LEGACY_ACTION_DIM:],
            "head_position": observed[..., LEGACY_ACTION_DIM:],
            "head_velocity": velocity_mask[..., LEGACY_ACTION_DIM:],
            "head_acceleration": acceleration_mask[..., LEGACY_ACTION_DIM:],
            "head_jerk": jerk_mask[..., LEGACY_ACTION_DIM:],
        }
        head_slices = {
            "head_flow": (predicted[..., LEGACY_ACTION_DIM:] - target[..., LEGACY_ACTION_DIM:]).square(),
            "head_position": F.smooth_l1_loss(
                reconstructed[..., LEGACY_ACTION_DIM:],
                actions[..., LEGACY_ACTION_DIM:],
                reduction="none",
            ),
            "head_velocity": F.smooth_l1_loss(
                reconstructed_velocity[..., LEGACY_ACTION_DIM:],
                target_velocity[..., LEGACY_ACTION_DIM:],
                reduction="none",
            ),
            "head_acceleration": F.smooth_l1_loss(
                reconstructed_acceleration[..., LEGACY_ACTION_DIM:],
                target_acceleration[..., LEGACY_ACTION_DIM:],
                reduction="none",
            ),
            "head_jerk": F.smooth_l1_loss(
                reconstructed_jerk[..., LEGACY_ACTION_DIM:],
                target_jerk[..., LEGACY_ACTION_DIM:],
                reduction="none",
            ),
        }
        for name in (
            "head_flow",
            "head_position",
            "head_velocity",
            "head_acceleration",
            "head_jerk",
        ):
            if name in requested_optional:
                losses[name] = _masked_mean(head_slices[name], head_masks[name])
    losses["total"] = sum(
        float(weights.get(name, 0.0)) * value for name, value in losses.items()
    )
    return losses


def evaluate_posttrain(
    model,
    episodes: Sequence[Mapping],
    *,
    action_stats,
    frame_count,
    batch_size,
    device,
    loss_weights,
    teacher_model=None,
    seed=0,
    batching: Mapping | None = None,
    conditioning_policy: str = "attached_conditions",
) -> dict:
    if not episodes:
        raise ValueError("evaluation requires episodes")
    declared_conditioning = {
        str(episode.get("evaluation_conditioning_policy"))
        for episode in episodes
        if episode.get("evaluation_conditioning_policy")
    }
    if conditioning_policy == "attached_conditions" and len(declared_conditioning) == 1:
        conditioning_policy = next(iter(declared_conditioning))
    was_training = model.training
    model.eval()
    totals = defaultdict(float)
    domain_totals = defaultdict(lambda: defaultdict(float))
    domain_counts = defaultdict(int)
    count = 0
    evaluated_frame_counts = []
    transition_required = float(loss_weights.get("planner_transition", 0.0)) > 0.0
    try:
        with torch.no_grad():
            # Model-selection metrics are accumulated per semantic unit.  A
            # clip-id-derived seed makes noise/timestep draws independent of
            # evaluation batch size, input order, and neighboring clip lengths.
            for episode in sorted(episodes, key=_episode_id):
                rows = [episode]
                actions, conditions, masks, durations, frame_valid = _batch_tensors_for_config(
                    rows,
                    frame_count=frame_count,
                    action_stats=action_stats,
                    device=device,
                    batching=batching,
                )
                episode_seed = int.from_bytes(
                    hashlib.sha256(
                        f"{int(seed)}:{_episode_id(episode)}".encode("utf-8")
                    ).digest()[:8],
                    byteorder="big",
                ) % (2**63 - 1)
                generator = torch.Generator(device=torch.device(device).type).manual_seed(
                    episode_seed
                )
                transition_targets = transition_mask = None
                if transition_required:
                    transition_targets, transition_mask = _transition_targets_for_episodes(
                        rows, device=device
                    )
                losses = masked_18d_objective(
                    model,
                    actions,
                    conditions,
                    masks,
                    durations,
                    loss_weights=loss_weights,
                    teacher_model=teacher_model,
                    generator=generator,
                    frame_valid_mask=frame_valid,
                    transition_targets=transition_targets,
                    transition_supervision_mask=transition_mask,
                )
                evaluated_frame_counts.extend(
                    frame_valid.sum(dim=1).detach().cpu().tolist()
                    if frame_valid is not None
                    else [int(actions.shape[1])] * len(rows)
                )
                values = {name: float(value.detach().cpu()) for name, value in losses.items()}
                for name, value in values.items():
                    totals[name] += value
                count += 1
                domain = str(episode["domain"])
                for name, value in values.items():
                    domain_totals[domain][name] += value
                domain_counts[domain] += 1
    finally:
        model.train(was_training)
    result = {name: value / count for name, value in totals.items()}
    result["by_domain"] = {
        domain: {
            name: value / domain_counts[domain]
            for name, value in sorted(domain_totals[domain].items())
        }
        for domain in sorted(domain_totals)
    }
    batching_mode = str((batching or {}).get("mode") or "fixed_resample")
    result["batching"] = {
        "mode": batching_mode,
        "requested_batch_size": int(batch_size),
        "metric_accumulation": "per_episode_stable_seed_equal_episode_weight",
        "episode_count": int(count),
        "frame_count_min": int(min(evaluated_frame_counts)),
        "frame_count_max": int(max(evaluated_frame_counts)),
        "unique_frame_counts": sorted({int(value) for value in evaluated_frame_counts}),
        "padding_ignored_by_attention_and_loss": (
            batching_mode == "native_variable_length"
        ),
    }
    result["conditioning_policy"] = str(conditioning_policy)
    result["model_selection_metric"] = (
        "mean_per_episode_total_weighted_objective_stable_clip_seed"
    )
    return result


def deterministic_replay_evaluation_subset(
    episodes: Sequence[Mapping], *, count: int, seed: int
) -> list[dict]:
    """Choose a stable replay probe without depending on input ordering."""

    count = int(count)
    if count <= 0:
        raise ValueError("replay evaluation count must be positive")
    ordered = sorted(
        (dict(episode) for episode in episodes),
        key=lambda episode: hashlib.sha256(
            f"{int(seed)}:{_episode_id(episode)}".encode("utf-8")
        ).hexdigest(),
    )
    return ordered[: min(count, len(ordered))]


def replay_regression_guard(
    current: Mapping | None,
    baseline: Mapping | None,
    *,
    maximum_fraction: float,
    maximum_absolute: float,
) -> dict:
    if current is None or baseline is None:
        return {
            "applicable": False,
            "passed": True,
            "baseline_total": None,
            "current_total": None,
            "delta": None,
            "allowed_delta": None,
        }
    baseline_total = float(baseline["total"])
    current_total = float(current["total"])
    allowed_delta = max(
        float(maximum_absolute), abs(baseline_total) * float(maximum_fraction)
    )
    delta = current_total - baseline_total
    return {
        "applicable": True,
        "passed": bool(delta <= allowed_delta),
        "baseline_total": baseline_total,
        "current_total": current_total,
        "delta": delta,
        "allowed_delta": allowed_delta,
        "maximum_fraction": float(maximum_fraction),
        "maximum_absolute": float(maximum_absolute),
    }


def posttrain_release_decision(data_provenance: Mapping, replay_guard: Mapping) -> dict:
    """Fail closed unless the complete formal training contract is present."""

    input_eligible = bool(data_provenance.get("input_formal_release_eligible"))
    applicable = replay_guard.get("applicable") is True
    passed = replay_guard.get("passed") is True
    experimental_scope = (
        data_provenance.get("training_scope") == "head_mechanism_experiment_only"
        or data_provenance.get("temporal_unit_policy")
        == "fixed_window_experimental"
        or data_provenance.get("formal_training_enabled") is False
    )
    formal_scope = bool(
        data_provenance.get("training_scope")
        == "formal_variable_length_semantic_units"
        and data_provenance.get("formal_training_enabled") is True
        and data_provenance.get("temporal_unit_policy")
        == "full_semantic_unit_variable_length_30hz"
        and data_provenance.get("batching_mode") == "native_variable_length"
        and data_provenance.get("semantic_boundary_contract_validated") is True
        and data_provenance.get("training_policy") == "full_network"
    )
    random_from_scratch = bool(
        data_provenance.get("generator_initialization_mode")
        in FULL_RANDOM_INITIALIZATION_MODES
        and data_provenance.get("forgetting_guard_applicable") is False
    )
    replay_guard_required = not random_from_scratch
    guard_ok = bool(
        (not replay_guard_required) or (applicable and passed)
    )
    eligible = bool(
        input_eligible
        and not data_provenance.get("unsafe_training_data")
        and formal_scope
        and guard_ok
        and not experimental_scope
    )
    if data_provenance.get("unsafe_training_data"):
        status = "experimental_unreviewed_unsafe"
    elif experimental_scope:
        status = "experimental_fixed_window_head_mechanism_only"
    elif not formal_scope:
        status = "blocked_missing_formal_variable_length_training_contract"
    elif replay_guard_required and not applicable:
        status = "blocked_missing_kimodo_replay_regression_evaluation"
    elif replay_guard_required and not passed:
        status = "blocked_kimodo_replay_regression"
    elif not input_eligible:
        status = "blocked_ineligible_formal_training_input"
    else:
        status = "adjudicated_posttrain_candidate"
    return {
        "formal_release_eligible": eligible,
        "artifact_status": status,
        "input_formal_release_eligible": input_eligible,
        "formal_training_contract_complete": formal_scope,
        "replay_regression_guard_required": replay_guard_required,
        "forgetting_guard_applicable": replay_guard_required,
        "replay_regression_guard": dict(replay_guard),
        "experimental_scope_blocked_formal_release": experimental_scope,
    }


def _read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on {path}:{line_number}") from exc
    return records


def load_attached_beat_episodes(
    manifest: str | Path,
    condition_cache: str | Path,
    *,
    allow_unreviewed=False,
    allow_unsafe_condition_cache=False,
    dataset_source: str | None = None,
    speaker_namespace: str | None = None,
    source_group_namespace: str | None = None,
) -> list[dict]:
    """Load BEAT trajectories, attach cached conditions, and restore group keys."""
    manifest = Path(manifest)
    raw_by_clip = {}
    raw_records = _read_jsonl(manifest)
    for record in raw_records:
        clip_id = str(record.get("clip_id") or record.get("sample_id") or "").strip()
        if clip_id:
            raw_by_clip[clip_id] = record
    from upper_body_skeleton.ula_v2_expression_turn_episode import (
        attach_expression_turn_v8_condition_cache,
        is_expression_turn_v8_episode,
        load_expression_turn_v8_episodes,
        validate_expression_turn_v8_episode,
    )
    from upper_body_skeleton.ula_v2_conversational_realization_episode import (
        attach_conversational_realization_v9_condition_cache,
        is_conversational_realization_v9_episode,
        load_conversational_realization_v9_episodes,
        validate_conversational_realization_v9_episode,
    )

    v8_flags = [is_expression_turn_v8_episode(record) for record in raw_records]
    conversational_flags = [
        is_conversational_realization_v9_episode(record) for record in raw_records
    ]
    if sum((any(v8_flags), any(conversational_flags))) > 1 or (
        any(v8_flags) and not all(v8_flags)
    ) or (any(conversational_flags) and not all(conversational_flags)):
        raise ValueError(f"manifest mixes incompatible episode contracts: {manifest}")
    expression_turn_v8 = bool(v8_flags and all(v8_flags))
    conversational_realization_v9 = bool(
        conversational_flags and all(conversational_flags)
    )
    if expression_turn_v8:
        if allow_unreviewed or allow_unsafe_condition_cache:
            raise ValueError("expression-turn v8 has no unsafe or unreviewed loading mode")
        loaded = load_expression_turn_v8_episodes(manifest)
        attached = attach_expression_turn_v8_condition_cache(loaded, condition_cache)
    elif conversational_realization_v9:
        if allow_unreviewed or allow_unsafe_condition_cache:
            raise ValueError("conversational realization v9 has no unsafe loading mode")
        loaded = load_conversational_realization_v9_episodes(manifest)
        attached = attach_conversational_realization_v9_condition_cache(
            loaded, condition_cache
        )
    else:
        loaded = load_18d_episodes(manifest=manifest, allow_unreviewed=allow_unreviewed)
        attached = attach_condition_cache(
            loaded,
            condition_cache,
            allow_unsafe_metadata=allow_unsafe_condition_cache,
        )
    enriched = []
    for episode in attached:
        raw = raw_by_clip.get(episode["clip_id"], {})
        item = dict(episode)
        for field in ("speaker_key", "source_group_key", "source_group_id", "source_clip_id"):
            if item.get(field) in (None, "") and raw.get(field) not in (None, ""):
                item[field] = raw[field]
        if item.get("source_group_key") in (None, ""):
            item["source_group_key"] = _nested_value(
                raw,
                ("split", "source_group_key"),
                ("source", "source_group_key"),
                ("source", "source_clip_id"),
            )
        if item.get("speaker_key") in (None, ""):
            item["speaker_key"] = _nested_value(
                raw, ("meta", "speaker_key"), ("source", "speaker_key")
            )
        if dataset_source:
            item["dataset_source"] = str(dataset_source)
        if speaker_namespace:
            item["speaker_key"] = f"{speaker_namespace}:{item['speaker_key']}"
        if source_group_namespace:
            item["source_group_key"] = (
                f"{source_group_namespace}:{item['source_group_key']}"
            )
        if expression_turn_v8:
            validate_expression_turn_v8_episode(item, require_attached_condition=True)
        elif conversational_realization_v9:
            validate_conversational_realization_v9_episode(
                item, require_attached_condition=True
            )
        enriched.append(item)
    return canonicalize_beat_episodes(enriched)


def load_kimodo_replay_splits(
    dataset_dir: str | Path,
    split_checkpoint: str | Path,
    qwen_checkpoint: str | Path,
    *,
    device="cpu",
    local_files_only=True,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Load disjoint Kimodo optimization, validation-guard, and final-test splits."""
    from upper_body_skeleton.cross_modal_latent import load_qwen_motion_text_encoder
    from upper_body_skeleton.ula_v2_conditioning import prepare_v2_episode_splits

    encoder, qwen_payload = load_qwen_motion_text_encoder(
        qwen_checkpoint, device=device, local_files_only=local_files_only
    )
    source = {
        "checkpoint_sha256": sha256_file(qwen_checkpoint),
        "artifact_kind": qwen_payload.get("artifact_kind"),
        "global_step": int(qwen_payload.get("global_step", 0)),
        "best_step": int(qwen_payload.get("best_step", 0)),
        "model_name": str((qwen_payload.get("qwen") or {}).get("model_name")),
        "revision": str((qwen_payload.get("qwen") or {}).get("revision")),
        "latent_dim": int((qwen_payload.get("config") or {}).get("latent_dim", 128)),
    }
    train, validation, test, contracts = prepare_v2_episode_splits(
        dataset_dir,
        split_checkpoint,
        device=device,
        text_motion_encoder=lambda texts: encoder.encode(texts, batch_size=16),
        text_motion_source=source,
    )
    for split_name, rows in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        for episode in rows:
            episode["replay_source_validated"] = True
            episode["replay_source_split"] = split_name
    del encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    provenance = {
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "split_checkpoint": str(Path(split_checkpoint).resolve()),
        "split_checkpoint_sha256": sha256_file(split_checkpoint),
        "qwen_checkpoint": str(Path(qwen_checkpoint).resolve()),
        "qwen_checkpoint_sha256": sha256_file(qwen_checkpoint),
        "v2_contract_sha256": contracts["sha256"],
        "train_episode_count": len(train),
        "validation_episode_count": len(validation),
        "test_episode_count": len(test),
        "validation_guard_episode_count": len(validation),
        "final_test_episode_count": len(test),
        "evaluation_policy": (
            "original_validation_for_guard_original_test_after_model_selection"
        ),
        "train_content_sha256": _json_hash(
            [
                _replay_content_record(episode)
                for episode in sorted(train, key=_episode_id)
            ]
        ),
        "validation_content_sha256": _json_hash(
            [
                _replay_content_record(episode)
                for episode in sorted(validation, key=_episode_id)
            ]
        ),
        "test_content_sha256": _json_hash(
            [
                _replay_content_record(episode)
                for episode in sorted(test, key=_episode_id)
            ]
        ),
    }
    return train, validation, test, provenance


def load_kimodo_replay_episodes(
    dataset_dir: str | Path,
    split_checkpoint: str | Path,
    qwen_checkpoint: str | Path,
    *,
    device="cpu",
    local_files_only=True,
) -> tuple[list[dict], dict]:
    """Backward-compatible loader returning the validated Kimodo train split."""
    train, _, _, provenance = load_kimodo_replay_splits(
        dataset_dir,
        split_checkpoint,
        qwen_checkpoint,
        device=device,
        local_files_only=local_files_only,
    )
    return train, provenance


def build_data_provenance(
    beat_splits: Mapping[str, Sequence[Mapping]],
    replay: Sequence[Mapping],
    split_contract: Mapping,
    *,
    replay_evaluation: Sequence[Mapping] = (),
    replay_test: Sequence[Mapping] = (),
    replay_provenance: Mapping | None = None,
    source_checkpoint: Mapping | None = None,
) -> tuple[dict, dict]:
    all_beat = [episode for name in SPLIT_NAMES for episode in beat_splits[name]]
    cache_records = [
        episode.get("condition_cache_provenance")
        for episode in all_beat
        if episode.get("condition_cache_provenance")
    ]
    cache_consistent = not cache_records or all(
        record == cache_records[0] for record in cache_records[1:]
    )
    cache_provenance = cache_records[0] if cache_records and cache_consistent else {}
    all_adjudicated = all(
        bool(episode.get("accepted_for_training"))
        and episode.get("eligibility_mode")
        == (
            "expression_turn_v8_adjudicated_train_ready"
            if episode.get("formal_episode_contract")
            == "beat2_expression_turn_v8_train_episode_v1"
            else "adjudicated_train_ready"
        )
        for episode in all_beat
    )
    unsafe_reasons = []
    source_checkpoint = dict(source_checkpoint or {})
    source_data = source_checkpoint.get("data_provenance") or {}
    source_unsafe = bool(
        source_checkpoint.get("unsafe_training_data")
        or (isinstance(source_data, Mapping) and source_data.get("unsafe_training_data"))
    )
    if source_unsafe:
        unsafe_reasons.append("unsafe_initial_checkpoint")
    if not all_adjudicated:
        unsafe_reasons.append("beat_motion_not_all_adjudicated_train_ready")
    if not cache_records:
        unsafe_reasons.append("unversioned_direct_conditions")
    elif not cache_consistent:
        unsafe_reasons.append("mixed_condition_cache_provenance")
    elif cache_provenance.get("unsafe_condition_cache") is True:
        unsafe_reasons.append("unsafe_condition_cache")
    if replay and not all(bool(item.get("replay_source_validated")) for item in replay):
        unsafe_reasons.append("unvalidated_kimodo_replay_source")
    if replay_evaluation and not all(
        bool(item.get("replay_source_validated")) for item in replay_evaluation
    ):
        unsafe_reasons.append("unvalidated_kimodo_replay_evaluation_source")
    if replay_test and not all(
        bool(item.get("replay_source_validated")) for item in replay_test
    ):
        unsafe_reasons.append("unvalidated_kimodo_replay_test_source")
    records = []
    for split_name in SPLIT_NAMES:
        for episode in beat_splits[split_name]:
            fps, duration = _episode_fps_and_duration(episode)
            record = {
                    "clip_id": _episode_id(episode),
                    "domain": str(episode.get("domain") or "beat2"),
                    "split": split_name,
                    "speaker_key": episode["speaker_key"],
                    "source_group_key": episode["source_group_key"],
                    "actions_sha256": episode.get("trajectory_sha256")
                    or _array_hash(episode["actions"]),
                    "condition_sha256": _array_hash(episode["condition"]),
                    "source_manifest_sha256": episode.get("source_manifest_sha256"),
                    "source_record_sha256": episode.get("source_record_sha256"),
                    "source_clip_id": episode.get("source_clip_id"),
                    "source_sha256": episode.get("source_sha256"),
                    "fps": fps,
                    "effective_duration_sec": duration,
                    "training_segment": deepcopy(episode.get("training_segment")),
                    "retarget_segment": deepcopy(episode.get("retarget_segment")),
                    "quality_source_window_frames": episode.get(
                        "quality_source_window_frames"
                    ),
                    "quality_output_frame_count": episode.get(
                        "quality_output_frame_count"
                    ),
                    "retarget_source_lineage": deepcopy(
                        episode.get("retarget_source_lineage") or {}
                    ),
                    "formal_source_metadata": deepcopy(
                        episode.get("formal_source_metadata") or {}
                    ),
                    "review_state": episode.get("review_state", "unspecified"),
                    "emotion_id": episode.get("emotion_id"),
                    "emotion_supervision_mask": bool(
                        episode.get("emotion_supervision_mask")
                    ),
                    "transition_id": episode.get("transition_id"),
                    "transition_supervision_mask": bool(
                        episode.get("transition_supervision_mask")
                    ),
                    "transition_label_source": episode.get(
                        "transition_label_source"
                    ),
                    "adjudicated_train_ready": bool(
                        episode.get("accepted_for_training")
                        and episode.get("eligibility_mode")
                        == (
                            "expression_turn_v8_adjudicated_train_ready"
                            if episode.get("formal_episode_contract")
                            == "beat2_expression_turn_v8_train_episode_v1"
                            else "adjudicated_train_ready"
                        )
                    ),
                    "action_dim_mask": [True] * ACTION_DIM,
                    "temporal_quarantine_challenge": bool(
                        episode.get("temporal_quarantine_challenge", False)
                    ),
                }
            if episode.get("dataset_source"):
                record["dataset_source"] = str(episode["dataset_source"])
            records.append(record)
    for episode in replay:
        records.append(
            {
                "clip_id": _episode_id(episode),
                "domain": "kimodo",
                "split": "train_replay_only",
                "speaker_key": episode["speaker_key"],
                "source_group_key": episode["source_group_key"],
                "actions_sha256": _array_hash(episode["actions"][:, :LEGACY_ACTION_DIM]),
                "condition_sha256": _array_hash(episode["condition"]),
                "emotion_supervision_mask": True,
                "action_dim_mask": [True] * LEGACY_ACTION_DIM
                + [False] * (ACTION_DIM - LEGACY_ACTION_DIM),
                "head_target_policy": "unobserved_loss_mask_zero",
            }
        )
    for episode in replay_evaluation:
        records.append(
            {
                "clip_id": _episode_id(episode),
                "domain": "kimodo",
                "split": "replay_evaluation_only",
                "source_split": episode.get("replay_source_split"),
                "speaker_key": episode["speaker_key"],
                "source_group_key": episode["source_group_key"],
                "actions_sha256": _array_hash(
                    episode["actions"][:, :LEGACY_ACTION_DIM]
                ),
                "condition_sha256": _array_hash(episode["condition"]),
                "emotion_supervision_mask": True,
                "action_dim_mask": [True] * LEGACY_ACTION_DIM
                + [False] * (ACTION_DIM - LEGACY_ACTION_DIM),
                "head_target_policy": "unobserved_loss_mask_zero",
                "optimization_eligible": False,
            }
        )
    for episode in replay_test:
        records.append(
            {
                "clip_id": _episode_id(episode),
                "domain": "kimodo",
                "split": "replay_test_only",
                "source_split": episode.get("replay_source_split"),
                "speaker_key": episode["speaker_key"],
                "source_group_key": episode["source_group_key"],
                "actions_sha256": _array_hash(
                    episode["actions"][:, :LEGACY_ACTION_DIM]
                ),
                "condition_sha256": _array_hash(episode["condition"]),
                "emotion_supervision_mask": True,
                "action_dim_mask": [True] * LEGACY_ACTION_DIM
                + [False] * (ACTION_DIM - LEGACY_ACTION_DIM),
                "head_target_policy": "unobserved_loss_mask_zero",
                "optimization_eligible": False,
                "model_selection_eligible": False,
            }
        )
    contract = {
        "contract_type": "ula_v2_18d_interaction_posttrain_data",
        "contract_version": 1,
        "split_contract_sha256": split_contract["sha256"],
        "episode_count": len(records),
        "records": sorted(records, key=lambda row: row["clip_id"]),
    }
    contract["sha256"] = _json_hash(contract)
    provenance = {
        "all_beat_adjudicated_train_ready": all_adjudicated,
        "unsafe_training_data": bool(unsafe_reasons),
        "unsafe_reasons": sorted(set(unsafe_reasons)),
        "release_status": (
            "experimental_unreviewed_unsafe"
            if unsafe_reasons
            else "adjudicated_posttrain_pending_replay_guard"
        ),
        "input_formal_release_eligible": not unsafe_reasons,
        "formal_release_eligible": False,
        "beat_counts": {name: len(beat_splits[name]) for name in SPLIT_NAMES},
        "beat_clean_evaluation_counts": {
            name: sum(
                not bool(episode.get("temporal_quarantine_challenge", False))
                for episode in beat_splits[name]
            )
            for name in ("validation", "test")
        },
        "beat_temporal_challenge_counts": {
            name: sum(
                bool(episode.get("temporal_quarantine_challenge", False))
                for episode in beat_splits[name]
            )
            for name in ("validation", "test")
        },
        "kimodo_replay_count": len(replay),
        "kimodo_replay_evaluation_count": len(replay_evaluation),
        "kimodo_replay_test_count": len(replay_test),
        "kimodo_replay_evaluation_disjoint": not bool(
            {_episode_id(item) for item in replay}
            & {_episode_id(item) for item in replay_evaluation}
        ),
        "kimodo_replay_test_pairwise_disjoint": not bool(
            ({_episode_id(item) for item in replay} | {
                _episode_id(item) for item in replay_evaluation
            })
            & {_episode_id(item) for item in replay_test}
        ),
        "emotion_supervised_beat_count": sum(
            episode.get("emotion_supervision_mask") is True for episode in all_beat
        ),
        "emotion_unresolved_beat_count": sum(
            episode.get("emotion_supervision_mask") is False for episode in all_beat
        ),
        "condition_cache": cache_provenance,
        "source_manifests": [
            {"path": path, "sha256": digest}
            for path, digest in sorted(
                {
                    (
                        episode.get("source_manifest"),
                        episode.get("source_manifest_sha256"),
                    )
                    for episode in all_beat
                    if episode.get("source_manifest")
                }
            )
        ],
        "replay": dict(replay_provenance or {}),
        "source_checkpoint_unsafe": source_unsafe,
        "formal_source_record_hash_count": sum(
            bool(episode.get("source_record_sha256")) for episode in all_beat
        ),
        "data_contract_sha256": contract["sha256"],
    }
    return provenance, contract


def _lr_scale(step, *, total_steps, warmup_steps, minimum_ratio):
    if warmup_steps and step <= warmup_steps:
        return max(1e-8, float(step) / float(warmup_steps))
    decay_steps = max(1, int(total_steps) - int(warmup_steps))
    progress = min(
        1.0,
        max(0.0, (float(step) - float(warmup_steps)) / float(decay_steps)),
    )
    return float(minimum_ratio) + (1.0 - float(minimum_ratio)) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def _cpu_state_dict(state):
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def _checkpoint_payload(
    initial_checkpoint: Mapping,
    model_state,
    *,
    raw_model_state,
    best_model_state,
    optimizer,
    ema,
    sampler,
    frame_rng,
    config,
    initial_checkpoint_path,
    initial_checkpoint_sha256,
    data_provenance,
    data_contract,
    split_contract,
    step,
    best_step,
    best_validation_loss,
    validation_metrics,
    replay_guard,
    current_replay_guard=None,
    best_replay_guard=None,
    stale_validations,
    include_training_state,
):
    payload = deepcopy(dict(initial_checkpoint))
    payload.pop("training_state", None)
    release = posttrain_release_decision(data_provenance, replay_guard)
    training_policy = str(config.get("training_policy") or "full_network")
    payload.update(
        {
            "schema_version": max(
                int(initial_checkpoint.get("schema_version", 1)), POSTTRAIN_SCHEMA_VERSION
            ),
            "model_state_dict": _cpu_state_dict(model_state),
            "global_step": int(initial_checkpoint.get("global_step", 0)) + int(step),
            "posttrain_step": int(step),
            "best_step": int(best_step),
            "best_validation_loss": float(best_validation_loss),
            "validation_metrics": dict(validation_metrics),
            "posttrain_artifact_kind": POSTTRAIN_ARTIFACT_KIND,
            "artifact_status": release["artifact_status"],
            "unsafe_training_data": data_provenance["unsafe_training_data"],
            "formal_release_eligible": release["formal_release_eligible"],
            "training_scope": data_provenance.get("training_scope"),
            "formal_training_enabled": data_provenance.get(
                "formal_training_enabled"
            ),
            "temporal_unit_policy": data_provenance.get("temporal_unit_policy"),
            "data_provenance": dict(data_provenance),
            "posttrain_data_contract": dict(data_contract),
            "posttrain_split_contract": dict(split_contract),
            "planner_supervision_contract": deepcopy(
                data_provenance.get("planner_supervision") or {}
            ),
            "posttrain_config": dict(config),
            "posttrain_source": {
                "checkpoint": str(Path(initial_checkpoint_path).resolve()),
                "checkpoint_sha256": initial_checkpoint_sha256,
                "source_global_step": int(initial_checkpoint.get("global_step", 0)),
            },
            "training_contract": {
                "mode": (
                    "low_lr_full_network_interaction_domain_posttrain"
                    if training_policy == "full_network"
                    else "head_projection_only_pretrain"
                ),
                "training_policy": training_policy,
                "all_model_parameters_trainable": training_policy == "full_network",
                "only_new_projection_slices_trainable": (
                    training_policy == "head_projection_only"
                ),
                "max_allowed_lr": (
                    MAX_FULL_FINETUNE_LR
                    if training_policy == "full_network"
                    else MAX_HEAD_PROJECTION_LR
                ),
                "ema_decay": float(config["ema_decay"]),
                "early_stopping_patience_validations": int(
                    config["early_stopping_patience"]
                ),
                "kimodo_15d_head_policy": "unobserved_per_dimension_loss_mask_zero",
                "kimodo_replay_evaluation_policy": (
                    "original_validation_guard_test_after_model_selection"
                    if data_provenance.get("kimodo_replay_evaluation_count", 0) > 0
                    else "legacy_training_replay_subset_or_missing"
                ),
                "emotion_policy": (
                    "source_label_provenance_only_until_anonymous_robot_affect_"
                    "blind_match_dual_gate"
                ),
                "planner_supervision": deepcopy(
                    data_provenance.get("planner_supervision") or {}
                ),
                "replay_regression_guard": dict(replay_guard),
                "formal_release_decision": release,
            },
        }
    )
    payload["config"] = dict(initial_checkpoint.get("config") or {}) | {
        "checkpoint_step": int(initial_checkpoint.get("global_step", 0)) + int(step),
        "checkpoint_loss": float(validation_metrics.get("total", best_validation_loss)),
    }
    if include_training_state:
        payload["training_state"] = {
            "raw_model_state_dict": _cpu_state_dict(raw_model_state),
            "ema_state_dict": _cpu_state_dict(ema.shadow),
            "best_model_state_dict": _cpu_state_dict(best_model_state),
            "optimizer_state_dict": optimizer.state_dict(),
            "sampler_state_dict": sampler.state_dict(),
            "frame_rng_state": frame_rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
            "stale_validations": int(stale_validations),
            "current_replay_guard": dict(current_replay_guard or replay_guard),
            "best_replay_guard": dict(best_replay_guard or replay_guard),
        }
    return payload


def _load_resume(
    path,
    *,
    model,
    ema,
    optimizer,
    sampler,
    frame_rng,
    config,
    initial_checkpoint_sha256,
    data_contract,
    device,
):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("posttrain_artifact_kind") != POSTTRAIN_ARTIFACT_KIND:
        raise ValueError("resume checkpoint is not an 18D interaction post-training artifact")
    if (
        (checkpoint.get("posttrain_source") or {}).get("checkpoint_sha256")
        != initial_checkpoint_sha256
    ):
        raise ValueError("resume source checkpoint changed")
    if (
        (checkpoint.get("posttrain_data_contract") or {}).get("sha256")
        != data_contract["sha256"]
    ):
        raise ValueError("resume data/split contract changed")
    previous_config = checkpoint.get("posttrain_config") or {}
    mutable_resume_fields = {"steps", "resume_from", "overwrite"}
    compared_fields = (set(previous_config) | set(config)) - mutable_resume_fields
    for field in sorted(compared_fields):
        if previous_config.get(field) != config.get(field):
            raise ValueError(f"resume config mismatch for {field}")
    state = checkpoint.get("training_state")
    if not isinstance(state, Mapping):
        raise ValueError("resume checkpoint does not contain exact training state")
    model.load_state_dict(state["raw_model_state_dict"], strict=True)
    current = model.state_dict()
    ema.shadow = {
        name: value.detach().to(current[name].device).clone()
        for name, value in state["ema_state_dict"].items()
    }
    optimizer.load_state_dict(state["optimizer_state_dict"])
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if torch.is_tensor(value):
                optimizer_state[key] = value.to(device)
    sampler.load_state_dict(state["sampler_state_dict"])
    frame_rng.bit_generator.state = state["frame_rng_state"]
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state.get("cuda_rng_state_all"):
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    current_replay_guard = state.get("current_replay_guard")
    best_replay_guard = state.get("best_replay_guard")
    if not isinstance(current_replay_guard, Mapping) or not isinstance(
        best_replay_guard, Mapping
    ):
        raise ValueError("resume checkpoint is missing exact replay guard state")
    return checkpoint, {
        name: value.detach().clone()
        for name, value in state["best_model_state_dict"].items()
    }, int(state.get("stale_validations", 0)), dict(current_replay_guard), dict(
        best_replay_guard
    )


def train_18d_posttrain(
    *,
    initial_checkpoint_path: str | Path,
    beat_episodes: Sequence[Mapping],
    output_dir: str | Path,
    kimodo_replay_episodes: Sequence[Mapping] = (),
    kimodo_replay_probe_episodes: Sequence[Mapping] = (),
    kimodo_replay_test_episodes: Sequence[Mapping] = (),
    replay_provenance: Mapping | None = None,
    config: Mapping | None = None,
) -> dict:
    config = resolve_posttrain_config(config)
    output_dir = Path(output_dir)
    resume_from = config.get("resume_from")
    if output_dir.exists() and any(output_dir.iterdir()) and not resume_from:
        if not config["overwrite"]:
            raise FileExistsError(f"post-training output directory is not empty: {output_dir}")
        for name in KNOWN_OUTPUT_NAMES:
            path = output_dir / name
            if path.is_file():
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(choose_device(str(config["device"])))
    model, initial_checkpoint = load_contract_checkpoint(
        initial_checkpoint_path, expected_action_dim=ACTION_DIM, device=device
    )
    initial_checkpoint_sha256 = sha256_file(initial_checkpoint_path)
    random_initialization = initial_checkpoint.get("random_initialization")
    full_random_initialization = bool(
        isinstance(random_initialization, Mapping)
        and random_initialization.get("mode")
        in FULL_RANDOM_INITIALIZATION_MODES
        and (initial_checkpoint.get("action_contract") or {}).get("migration")
        == "none_full_18d_random_initialization"
    )
    if full_random_initialization:
        teacher_model = None
        if float(config["loss"].get("body", 0.0)) != 0.0:
            raise ValueError(
                "full-random training requires loss.body=0 because no pretrained "
                "generator teacher exists"
            )
        if (
            config.get("training_scope")
            == "formal_variable_length_semantic_units"
            and float(
                config["loss"].get(
                    "planner_duration", config["loss"].get("planner", 0.0)
                )
            )
            <= 0.0
        ):
            raise ValueError(
                "formal full-random training requires a positive planner_duration loss"
            )
    else:
        teacher_model, _ = load_contract_checkpoint(
            initial_checkpoint_path, expected_action_dim=ACTION_DIM, device=device
        )
        teacher_model.requires_grad_(False).eval()
    training_policy = str(config.get("training_policy") or "full_network")
    model.requires_grad_(True).train()
    head_policy = None
    if training_policy == "head_projection_only":
        head_policy = configure_head_adapter_policy(model)
        # Disable dropout so body-teacher differences measure head cross-talk.
        model.eval()
    action_stats = initial_checkpoint["action_stats"]

    if config.get("training_scope") == "formal_variable_length_semantic_units":
        from upper_body_skeleton.ula_v2_18d_random_init import (
            validate_formal_variable_length_episode,
        )

        for episode in beat_episodes:
            validate_formal_variable_length_episode(
                episode, require_attached_condition=True
            )
    planner_supervision = transition_supervision_contract(beat_episodes)
    if (
        float(config["loss"].get("planner_transition", 0.0)) > 0.0
        and not planner_supervision["transition_head_trainable"]
    ):
        raise ValueError(
            "planner_transition loss is positive but no verified adjacent-sequence labels exist"
        )
    beat = canonicalize_beat_episodes(beat_episodes)
    replay = canonicalize_kimodo_replay(kimodo_replay_episodes)
    replay_probe = canonicalize_kimodo_replay(kimodo_replay_probe_episodes)
    replay_test = canonicalize_kimodo_replay(kimodo_replay_test_episodes)
    replay_ids = {_episode_id(item) for item in replay}
    probe_ids = {_episode_id(item) for item in replay_probe}
    test_ids = {_episode_id(item) for item in replay_test}
    replay_overlap = (replay_ids & probe_ids) | (replay_ids & test_ids) | (
        probe_ids & test_ids
    )
    if replay_overlap:
        raise ValueError(
            "Kimodo replay optimization/evaluation leakage: "
            f"{sorted(replay_overlap)[:5]}"
        )
    if config.get("require_disjoint_replay_evaluation") and (
        not replay or not replay_probe or not replay_test
    ):
        raise ValueError(
            "require_disjoint_replay_evaluation needs non-empty Kimodo train and "
            "disjoint validation/test replay episodes"
        )
    if full_random_initialization:
        from upper_body_skeleton.ula_v2_18d_random_init import (
            validate_random_checkpoint_split,
        )

        beat_splits, split_contract = validate_random_checkpoint_split(
            initial_checkpoint,
            beat,
            requested_fractions=config["split_fractions"],
        )
    else:
        beat_splits, split_contract = strict_group_split(
            beat,
            seed=seed,
            fractions=config["split_fractions"],
        )
    train_episodes = beat_splits["train"] + replay
    validation_episodes = [
        episode
        for episode in beat_splits["validation"]
        if not episode.get("temporal_quarantine_challenge", False)
    ]
    validation_challenge_episodes = [
        episode
        for episode in beat_splits["validation"]
        if episode.get("temporal_quarantine_challenge", False)
    ]
    test_episodes = [
        episode
        for episode in beat_splits["test"]
        if not episode.get("temporal_quarantine_challenge", False)
    ]
    test_challenge_episodes = [
        episode
        for episode in beat_splits["test"]
        if episode.get("temporal_quarantine_challenge", False)
    ]
    if not validation_episodes or not test_episodes:
        raise ValueError(
            "temporal quarantine left no clean primary validation/test episodes"
        )
    replay_evaluation_source = replay_probe or replay
    replay_evaluation_episodes = deterministic_replay_evaluation_subset(
        replay_evaluation_source,
        count=config["replay_evaluation_count"],
        seed=seed + 65537,
    ) if replay else []
    replay_test_episodes = (
        deterministic_replay_evaluation_subset(
            replay_test,
            count=config["replay_evaluation_count"],
            seed=seed + 131071,
        )
        if replay_test
        else []
    )
    data_provenance, data_contract = build_data_provenance(
        beat_splits,
        replay,
        split_contract,
        replay_evaluation=replay_probe,
        replay_test=replay_test,
        replay_provenance=replay_provenance,
        source_checkpoint=initial_checkpoint,
    )
    formal_episode_contract_validated = bool(
        config.get("training_scope") == "formal_variable_length_semantic_units"
        and config.get("formal_training_enabled") is True
        and config.get("temporal_unit_policy")
        == "full_semantic_unit_variable_length_30hz"
        and (config.get("batching") or {}).get("mode")
        == "native_variable_length"
    )
    execution_contract = {
        "contract_type": "ula_v2_18d_training_execution_scope",
        "contract_version": 1,
        "training_scope": config.get("training_scope"),
        "formal_training_enabled": config.get("formal_training_enabled") is True,
        "temporal_unit_policy": config.get("temporal_unit_policy"),
        "batching_mode": (config.get("batching") or {}).get("mode", "fixed_resample"),
        "native_batching_memory_policy": (
            {
                "homogeneous_bucket_batches": config["batching"][
                    "homogeneous_bucket_batches"
                ],
                "max_motion_tokens_per_microbatch": config["batching"][
                    "max_motion_tokens_per_microbatch"
                ],
                "max_attention_elements_per_microbatch": config["batching"][
                    "max_attention_elements_per_microbatch"
                ],
                "target_effective_batch_size": config["batching"][
                    "target_effective_batch_size"
                ],
                "gradient_accumulation_mode": config["batching"][
                    "gradient_accumulation_mode"
                ],
                "oversize_sequence_policy": config["batching"][
                    "oversize_sequence_policy"
                ],
                "cropping_allowed": False,
            }
            if (config.get("batching") or {}).get("mode")
            == "native_variable_length"
            else None
        ),
        "duration_target_time_axis": "retarget_output_sample_span",
        "duration_target_formula": "(output_frame_count-1)/fps",
        "padding_duration_policy": "padding_never_changes_duration_or_loss_mask",
        "source_boundary_time_axis": "original_semantic_source_interval",
        "semantic_boundary_contract_validated": formal_episode_contract_validated,
        "training_policy": training_policy,
        "generator_initialization_mode": (
            random_initialization.get("mode")
            if full_random_initialization
            else "pretrained_generator_warm_start"
        ),
        "forgetting_guard_applicable": not full_random_initialization,
        "split_policy": (
            "reuse_random_initialization_split"
            if full_random_initialization
            else "strict_group_split_for_warm_start"
        ),
        "action_style_statistics_policy": (
            "reuse_random_initialization_train_only_statistics_no_refit"
            if full_random_initialization
            else "reuse_source_checkpoint_statistics"
        ),
        "primary_evaluation_conditioning": (
            "motion_only_zero_text_semantics_default_style_non_oracle"
            if initial_checkpoint.get("formal_episode_contract")
            == MOTION_ONLY_EPISODE_CONTRACT
            else "text_explicit_semantics_default_style_non_oracle"
            if full_random_initialization
            else "attached_conditions"
        ),
        "planner_supervision": planner_supervision,
    }
    execution_contract["sha256"] = _json_hash(execution_contract)
    data_contract = {
        key: value for key, value in data_contract.items() if key != "sha256"
    }
    data_contract["training_execution_contract"] = execution_contract
    data_contract["sha256"] = _json_hash(data_contract)
    data_provenance.update(execution_contract)
    data_provenance["training_execution_contract_sha256"] = execution_contract[
        "sha256"
    ]
    data_provenance["data_contract_sha256"] = data_contract["sha256"]
    if not formal_episode_contract_validated:
        data_provenance["input_formal_release_eligible"] = False
        data_provenance["formal_release_eligible"] = False
        data_provenance["release_status"] = (
            "experimental_fixed_window_head_mechanism_only"
        )
    if data_provenance["unsafe_training_data"] and not config[
        "allow_unsafe_training_data"
    ]:
        raise ValueError(
            "strict post-training refuses unsafe/unadjudicated inputs: "
            + ", ".join(data_provenance["unsafe_reasons"])
        )
    cache = data_provenance.get("condition_cache") or {}
    if cache:
        validate_condition_cache_for_generator(
            initial_checkpoint,
            cache,
            generator_checkpoint_path=initial_checkpoint_path,
            allow_unsafe=config["allow_unsafe_training_data"],
        )

    if full_random_initialization:
        validation_episodes = _default_style_evaluation_episodes(
            validation_episodes
        )
        validation_challenge_episodes = _default_style_evaluation_episodes(
            validation_challenge_episodes
        )
        test_episodes = _default_style_evaluation_episodes(test_episodes)
        test_challenge_episodes = _default_style_evaluation_episodes(
            test_challenge_episodes
        )
        replay_evaluation_episodes = _default_style_evaluation_episodes(
            replay_evaluation_episodes
        )
        replay_test_episodes = _default_style_evaluation_episodes(
            replay_test_episodes
        )

    _atomic_json_save(split_contract, output_dir / "split_manifest.json")
    optimized_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        optimized_parameters,
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
        eps=float(config["adam_eps"]),
    )
    ema = ModelEMA(model, config["ema_decay"])
    sampler = _sampler_for_config(train_episodes, config, seed=seed + 17)
    frame_rng = np.random.default_rng(seed + 104729)
    evaluation_frame_count = max(config["phase_frame_choices"])
    batching_config = config.get("batching")
    native_batching_audit = None
    if isinstance(sampler, NativeLengthBucketSampler):
        native_batching_audit = sampler.validate_budgets(
            semantic_tokens=int(model.semantic_tokens),
            max_batch_size=int(config["batch_size"]),
            batching=batching_config,
        )
    initial_validation_metrics = evaluate_posttrain(
        model,
        validation_episodes,
        action_stats=action_stats,
        frame_count=evaluation_frame_count,
        batch_size=config["validation_batch_size"],
        device=device,
        loss_weights=config["loss"],
        teacher_model=teacher_model,
        seed=seed + 1_000_003,
        batching=batching_config,
    )
    # Test and test-challenge are sealed until the validation-selected model freezes.
    initial_test_metrics = None
    initial_validation_challenge_metrics = (
        evaluate_posttrain(
            model,
            validation_challenge_episodes,
            action_stats=action_stats,
            frame_count=evaluation_frame_count,
            batch_size=config["validation_batch_size"],
            device=device,
            loss_weights=config["loss"],
            teacher_model=teacher_model,
            seed=seed + 4_000_003,
            batching=batching_config,
        )
        if validation_challenge_episodes
        else None
    )
    initial_test_challenge_metrics = None
    initial_replay_metrics = (
        evaluate_posttrain(
            model,
            replay_evaluation_episodes,
            action_stats=action_stats,
            frame_count=evaluation_frame_count,
            batch_size=config["validation_batch_size"],
            device=device,
            loss_weights=config["loss"],
            teacher_model=teacher_model,
            seed=seed + 3_000_003,
            batching=batching_config,
        )
        if replay_evaluation_episodes
        else None
    )
    global_step = 0
    best_step = 0
    # A random, untrained generator is a baseline diagnostic, never a selectable model.
    best_validation_loss = (
        float("inf")
        if full_random_initialization
        else float(initial_validation_metrics["total"])
    )
    best_model_state = _cpu_state_dict(ema.shadow)
    stale_validations = 0
    validation_metrics = dict(initial_validation_metrics)
    validation_challenge_metrics = initial_validation_challenge_metrics
    replay_validation_metrics = initial_replay_metrics
    initial_replay_guard = replay_regression_guard(
        replay_validation_metrics,
        initial_replay_metrics,
        maximum_fraction=config["maximum_replay_regression_fraction"],
        maximum_absolute=config["maximum_replay_regression_absolute"],
    )
    replay_guard = dict(initial_replay_guard)
    best_replay_guard = dict(initial_replay_guard)
    if resume_from:
        (
            checkpoint,
            best_model_state,
            stale_validations,
            replay_guard,
            best_replay_guard,
        ) = _load_resume(
            resume_from,
            model=model,
            ema=ema,
            optimizer=optimizer,
            sampler=sampler,
            frame_rng=frame_rng,
            config=config,
            initial_checkpoint_sha256=initial_checkpoint_sha256,
            data_contract=data_contract,
            device=device,
        )
        global_step = int(checkpoint["posttrain_step"])
        best_step = int(checkpoint["best_step"])
        best_validation_loss = float(checkpoint["best_validation_loss"])
        validation_metrics = dict(checkpoint.get("validation_metrics") or {})
        if head_policy is not None and frozen_weight_max_error(model, head_policy) != 0.0:
            raise ValueError("resume checkpoint changed head-pretrain frozen parameters")
    else:
        (output_dir / "progress.jsonl").write_text("", encoding="utf-8")
        baseline_event = {
            "step": 0,
            "event": "pretrain_baseline",
            "validation": initial_validation_metrics,
            "validation_temporal_challenge": initial_validation_challenge_metrics,
            "replay_validation": initial_replay_metrics,
            "replay_regression_guard": replay_guard,
        }
        line = json.dumps(baseline_event, sort_keys=True)
        print(line, flush=True)
        with (output_dir / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
    if global_step > int(config["steps"]):
        raise ValueError("target steps must not be less than resumed posttrain step")

    resumed_terminal_early_stop = bool(
        resume_from
        and stale_validations >= int(config["early_stopping_patience"])
    )

    print(
        json.dumps(
            {
                "device": str(device),
                "start_step": global_step,
                "target_steps": config["steps"],
                "trainable_parameters": sum(
                    parameter.numel() for parameter in optimized_parameters
                ),
                "training_policy": training_policy,
                "split_counts": {
                    "beat_train": len(beat_splits["train"]),
                    "beat_validation": len(validation_episodes),
                    "beat_validation_temporal_challenge": len(
                        validation_challenge_episodes
                    ),
                    "beat_test": len(test_episodes),
                    "beat_test_temporal_challenge": len(test_challenge_episodes),
                    "kimodo_replay": len(replay),
                    "kimodo_replay_probe_source": len(replay_probe),
                    "kimodo_replay_test_source": len(replay_test),
                },
                "unsafe_training_data": data_provenance["unsafe_training_data"],
                "release_status": data_provenance["release_status"],
                "data_contract_sha256": data_contract["sha256"],
                "native_batching_audit": native_batching_audit,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    progress_path = output_dir / "progress.jsonl"
    stopped_early = resumed_terminal_early_stop
    last_train = {}
    last_grad_norm = 0.0
    frozen_weight_error = 0.0
    loop_target_step = global_step if resumed_terminal_early_stop else int(config["steps"])
    for step in range(global_step + 1, loop_target_step + 1):
        scale = _lr_scale(
            step,
            total_steps=config["steps"],
            warmup_steps=config["warmup_steps"],
            minimum_ratio=config["minimum_lr_ratio"],
        )
        current_lr = float(config["lr"]) * scale
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.zero_grad(set_to_none=True)
        selected = []
        native_frame_counts = None
        microbatch_plans = []
        if isinstance(sampler, NativeLengthBucketSampler):
            target_effective_batch = int(
                batching_config["target_effective_batch_size"]
            )
            remaining = target_effective_batch
            accumulated_losses = defaultdict(float)
            native_frame_counts = []
            while remaining > 0:
                microbatch, plan = sampler.sample_microbatch(
                    remaining_effective_batch=remaining,
                    semantic_tokens=int(model.semantic_tokens),
                    max_batch_size=int(config["batch_size"]),
                    batching=batching_config,
                )
                selected.extend(microbatch)
                requested_frame_count = int(
                    frame_rng.choice(config["phase_frame_choices"])
                )
                (
                    actions,
                    conditions,
                    masks,
                    durations,
                    frame_valid,
                ) = _batch_tensors_for_config(
                    microbatch,
                    frame_count=requested_frame_count,
                    action_stats=action_stats,
                    device=device,
                    batching=batching_config,
                )
                if int(actions.shape[1]) != int(plan["bucket_frames"]):
                    raise RuntimeError(
                        "native collation bucket differs from the preflight memory plan"
                    )
                transition_targets = transition_mask = None
                if float(config["loss"].get("planner_transition", 0.0)) > 0.0:
                    transition_targets, transition_mask = _transition_targets_for_episodes(
                        microbatch, device=device
                    )
                losses = masked_18d_objective(
                    model,
                    actions,
                    conditions,
                    masks,
                    durations,
                    loss_weights=config["loss"],
                    teacher_model=teacher_model,
                    frame_valid_mask=frame_valid,
                    transition_targets=transition_targets,
                    transition_supervision_mask=transition_mask,
                )
                if not torch.isfinite(losses["total"]):
                    raise FloatingPointError(
                        f"non-finite post-training loss at step {step}"
                    )
                sample_weight = len(microbatch) / float(target_effective_batch)
                (losses["total"] * sample_weight).backward()
                for name, value in losses.items():
                    accumulated_losses[name] += (
                        float(value.detach().cpu()) * sample_weight
                    )
                native_frame_counts.extend(
                    frame_valid.sum(dim=1).detach().cpu().tolist()
                )
                microbatch_plans.append(dict(plan))
                remaining -= len(microbatch)
            frame_count = max(plan["bucket_frames"] for plan in microbatch_plans)
            last_train = dict(accumulated_losses)
        else:
            selected = sampler.sample(config["batch_size"])
            requested_frame_count = int(frame_rng.choice(config["phase_frame_choices"]))
            (
                actions,
                conditions,
                masks,
                durations,
                frame_valid,
            ) = _batch_tensors_for_config(
                selected,
                frame_count=requested_frame_count,
                action_stats=action_stats,
                device=device,
                batching=batching_config,
            )
            frame_count = int(actions.shape[1])
            transition_targets = transition_mask = None
            if float(config["loss"].get("planner_transition", 0.0)) > 0.0:
                transition_targets, transition_mask = _transition_targets_for_episodes(
                    selected, device=device
                )
            losses = masked_18d_objective(
                model,
                actions,
                conditions,
                masks,
                durations,
                loss_weights=config["loss"],
                teacher_model=teacher_model,
                frame_valid_mask=frame_valid,
                transition_targets=transition_targets,
                transition_supervision_mask=transition_mask,
            )
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(f"non-finite post-training loss at step {step}")
            losses["total"].backward()
            last_train = {
                name: float(value.detach().cpu()) for name, value in losses.items()
            }
        last_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(optimized_parameters, config["max_grad_norm"])
        )
        optimizer.step()
        if head_policy is not None:
            restore_frozen_weights(model, head_policy)
            frozen_weight_error = frozen_weight_max_error(model, head_policy)
            if frozen_weight_error != 0.0:
                raise RuntimeError(
                    f"head pretrain changed frozen parameters at step {step}: "
                    f"{frozen_weight_error}"
                )
        ema.update(model)
        if head_policy is not None:
            _restore_frozen_ema_state(ema, head_policy)
        global_step = step

        should_validate = (
            step == 1
            or step % int(config["validation_interval"]) == 0
            or step == int(config["steps"])
        )
        should_checkpoint = (
            step == 1
            or step % int(config["checkpoint_interval"]) == 0
            or step == int(config["steps"])
        )
        is_best = False
        if should_validate:
            with ema.apply(model):
                validation_metrics = evaluate_posttrain(
                    model,
                    validation_episodes,
                    action_stats=action_stats,
                    frame_count=evaluation_frame_count,
                    batch_size=config["validation_batch_size"],
                    device=device,
                    loss_weights=config["loss"],
                    teacher_model=teacher_model,
                    seed=seed + 1_000_003,
                    batching=batching_config,
                )
                validation_challenge_metrics = (
                    evaluate_posttrain(
                        model,
                        validation_challenge_episodes,
                        action_stats=action_stats,
                        frame_count=evaluation_frame_count,
                        batch_size=config["validation_batch_size"],
                        device=device,
                        loss_weights=config["loss"],
                        teacher_model=teacher_model,
                        seed=seed + 4_000_003,
                        batching=batching_config,
                    )
                    if validation_challenge_episodes
                    else None
                )
                replay_validation_metrics = (
                    evaluate_posttrain(
                        model,
                        replay_evaluation_episodes,
                        action_stats=action_stats,
                        frame_count=evaluation_frame_count,
                        batch_size=config["validation_batch_size"],
                        device=device,
                        loss_weights=config["loss"],
                        teacher_model=teacher_model,
                        seed=seed + 3_000_003,
                        batching=batching_config,
                    )
                    if replay_evaluation_episodes
                    else None
                )
            replay_guard = replay_regression_guard(
                replay_validation_metrics,
                initial_replay_metrics,
                maximum_fraction=config["maximum_replay_regression_fraction"],
                maximum_absolute=config["maximum_replay_regression_absolute"],
            )
            if replay_guard["passed"] and validation_metrics["total"] < (
                best_validation_loss - float(config["early_stopping_min_delta"])
            ):
                best_validation_loss = float(validation_metrics["total"])
                best_step = step
                best_model_state = _cpu_state_dict(ema.shadow)
                best_replay_guard = dict(replay_guard)
                stale_validations = 0
                is_best = True
            else:
                stale_validations += 1

        event = {
            "step": step,
            "lr": current_lr,
            "frames": frame_count,
            "native_frame_counts": native_frame_counts,
            "microbatch_count": len(microbatch_plans) if microbatch_plans else 1,
            "microbatch_plans": microbatch_plans or None,
            "grad_norm": last_grad_norm,
            "batch_domain_counts": {
                domain: sum(item["domain"] == domain for item in selected)
                for domain in sorted({item["domain"] for item in selected})
            },
            "train": last_train,
        }
        if head_policy is not None:
            event["frozen_weight_max_abs_error"] = frozen_weight_error
        if should_validate:
            event.update(
                {
                    "validation": validation_metrics,
                    "validation_temporal_challenge": validation_challenge_metrics,
                    "replay_validation": replay_validation_metrics,
                    "replay_regression_guard": replay_guard,
                    "is_best": is_best,
                    "stale_validations": stale_validations,
                }
            )
        if step == 1 or step % int(config["log_interval"]) == 0 or should_validate:
            line = json.dumps(event, sort_keys=True)
            print(line, flush=True)
            with progress_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")

        if is_best:
            best_payload = _checkpoint_payload(
                initial_checkpoint,
                best_model_state,
                raw_model_state=model.state_dict(),
                best_model_state=best_model_state,
                optimizer=optimizer,
                ema=ema,
                sampler=sampler,
                frame_rng=frame_rng,
                config=config,
                initial_checkpoint_path=initial_checkpoint_path,
                initial_checkpoint_sha256=initial_checkpoint_sha256,
                data_provenance=data_provenance,
                data_contract=data_contract,
                split_contract=split_contract,
                step=step,
                best_step=best_step,
                best_validation_loss=best_validation_loss,
                validation_metrics=validation_metrics,
                replay_guard=best_replay_guard,
                stale_validations=stale_validations,
                include_training_state=False,
            )
            _atomic_torch_save(best_payload, output_dir / "ula_fm_checkpoint.pt")
        stop_now = should_validate and stale_validations >= int(
            config["early_stopping_patience"]
        )
        if should_checkpoint or stop_now:
            last_payload = _checkpoint_payload(
                initial_checkpoint,
                ema.shadow,
                raw_model_state=model.state_dict(),
                best_model_state=best_model_state,
                optimizer=optimizer,
                ema=ema,
                sampler=sampler,
                frame_rng=frame_rng,
                config=config,
                initial_checkpoint_path=initial_checkpoint_path,
                initial_checkpoint_sha256=initial_checkpoint_sha256,
                data_provenance=data_provenance,
                data_contract=data_contract,
                split_contract=split_contract,
                step=step,
                best_step=best_step,
                best_validation_loss=best_validation_loss,
                validation_metrics=validation_metrics,
                replay_guard=replay_guard,
                current_replay_guard=replay_guard,
                best_replay_guard=best_replay_guard,
                stale_validations=stale_validations,
                include_training_state=True,
            )
            _atomic_torch_save(last_payload, output_dir / "last.pt")
        if stop_now:
            stopped_early = True
            break

    best_path = output_dir / "ula_fm_checkpoint.pt"
    if not best_path.is_file():
        best_payload = _checkpoint_payload(
            initial_checkpoint,
            best_model_state,
            raw_model_state=model.state_dict(),
            best_model_state=best_model_state,
            optimizer=optimizer,
            ema=ema,
            sampler=sampler,
            frame_rng=frame_rng,
            config=config,
            initial_checkpoint_path=initial_checkpoint_path,
            initial_checkpoint_sha256=initial_checkpoint_sha256,
            data_provenance=data_provenance,
            data_contract=data_contract,
            split_contract=split_contract,
            step=best_step,
            best_step=best_step,
            best_validation_loss=best_validation_loss,
            validation_metrics=validation_metrics,
            replay_guard=best_replay_guard,
            stale_validations=stale_validations,
            include_training_state=False,
        )
        _atomic_torch_save(best_payload, best_path)
    model.load_state_dict(best_model_state, strict=True)
    if head_policy is not None:
        frozen_weight_error = frozen_weight_max_error(model, head_policy)
        if frozen_weight_error != 0.0:
            raise RuntimeError(
                "selected head-pretrain checkpoint changed frozen parameters: "
                f"{frozen_weight_error}"
            )
    final_validation_metrics = evaluate_posttrain(
        model,
        validation_episodes,
        action_stats=action_stats,
        frame_count=evaluation_frame_count,
        batch_size=config["validation_batch_size"],
        device=device,
        loss_weights=config["loss"],
        teacher_model=teacher_model,
        seed=seed + 1_000_003,
        batching=batching_config,
    )
    test_metrics = evaluate_posttrain(
        model,
        test_episodes,
        action_stats=action_stats,
        frame_count=evaluation_frame_count,
        batch_size=config["validation_batch_size"],
        device=device,
        loss_weights=config["loss"],
        teacher_model=teacher_model,
        seed=seed + 2_000_003,
        batching=batching_config,
    )
    # The disjoint Kimodo test is also evaluated only once on the frozen model.
    initial_replay_test_metrics = None
    final_replay_test_metrics = (
        evaluate_posttrain(
            model,
            replay_test_episodes,
            action_stats=action_stats,
            frame_count=evaluation_frame_count,
            batch_size=config["validation_batch_size"],
            device=device,
            loss_weights=config["loss"],
            teacher_model=teacher_model,
            seed=seed + 6_000_003,
            batching=batching_config,
        )
        if replay_test_episodes
        else None
    )
    final_validation_challenge_metrics = (
        evaluate_posttrain(
            model,
            validation_challenge_episodes,
            action_stats=action_stats,
            frame_count=evaluation_frame_count,
            batch_size=config["validation_batch_size"],
            device=device,
            loss_weights=config["loss"],
            teacher_model=teacher_model,
            seed=seed + 4_000_003,
            batching=batching_config,
        )
        if validation_challenge_episodes
        else None
    )
    final_test_challenge_metrics = (
        evaluate_posttrain(
            model,
            test_challenge_episodes,
            action_stats=action_stats,
            frame_count=evaluation_frame_count,
            batch_size=config["validation_batch_size"],
            device=device,
            loss_weights=config["loss"],
            teacher_model=teacher_model,
            seed=seed + 5_000_003,
            batching=batching_config,
        )
        if test_challenge_episodes
        else None
    )
    final_replay_metrics = (
        evaluate_posttrain(
            model,
            replay_evaluation_episodes,
            action_stats=action_stats,
            frame_count=evaluation_frame_count,
            batch_size=config["validation_batch_size"],
            device=device,
            loss_weights=config["loss"],
            teacher_model=teacher_model,
            seed=seed + 3_000_003,
            batching=batching_config,
        )
        if replay_evaluation_episodes
        else None
    )
    final_replay_guard = replay_regression_guard(
        final_replay_metrics,
        initial_replay_metrics,
        maximum_fraction=config["maximum_replay_regression_fraction"],
        maximum_absolute=config["maximum_replay_regression_absolute"],
    )
    release = posttrain_release_decision(data_provenance, final_replay_guard)
    best_payload = torch.load(best_path, map_location="cpu", weights_only=True)
    final_data_provenance = dict(best_payload.get("data_provenance") or data_provenance)
    final_data_provenance.update(
        {
            "release_status": release["artifact_status"],
            "formal_release_eligible": release["formal_release_eligible"],
        }
    )
    best_payload["best_validation_loss"] = float(final_validation_metrics["total"])
    best_payload["validation_metrics"] = dict(final_validation_metrics)
    best_payload["artifact_status"] = release["artifact_status"]
    best_payload["formal_release_eligible"] = release["formal_release_eligible"]
    best_payload["data_provenance"] = final_data_provenance
    best_payload["config"] = dict(best_payload.get("config") or {}) | {
        "checkpoint_loss": float(final_validation_metrics["total"]),
    }
    best_payload["training_contract"] = dict(
        best_payload.get("training_contract") or {}
    ) | {
        "replay_regression_guard": dict(final_replay_guard),
        "formal_release_decision": release,
        "training_scope": final_data_provenance.get("training_scope"),
        "formal_training_enabled": final_data_provenance.get(
            "formal_training_enabled"
        ),
        "temporal_unit_policy": final_data_provenance.get(
            "temporal_unit_policy"
        ),
        "kimodo_final_test_report": {
            "evaluation_count": len(replay_test_episodes),
            "initial": initial_replay_test_metrics,
            "final": final_replay_test_metrics,
            "policy": "evaluated_only_after_model_selection",
        },
    }
    _atomic_torch_save(best_payload, best_path)
    summary = {
        "output_dir": str(output_dir.resolve()),
        "checkpoint": str(best_path.resolve()),
        "last_checkpoint": str((output_dir / "last.pt").resolve()),
        "completed_steps": int(global_step),
        "target_steps": int(config["steps"]),
        "stopped_early": stopped_early,
        "best_step": int(best_step),
        "best_validation_loss": float(final_validation_metrics["total"]),
        "initial_validation": initial_validation_metrics,
        "final_validation": final_validation_metrics,
        "initial_test": initial_test_metrics,
        "test": test_metrics,
        "initial_validation_temporal_challenge": initial_validation_challenge_metrics,
        "final_validation_temporal_challenge": final_validation_challenge_metrics,
        "initial_test_temporal_challenge": initial_test_challenge_metrics,
        "test_temporal_challenge": final_test_challenge_metrics,
        "model_selection_evaluation_policy": (
            "clean_validation_only_temporal_failures_reported_as_challenge"
        ),
        "initial_replay_validation": initial_replay_metrics,
        "final_replay_validation": final_replay_metrics,
        "initial_replay_test": initial_replay_test_metrics,
        "final_replay_test": final_replay_test_metrics,
        "final_replay_regression_guard": final_replay_guard,
        "replay_evaluation_count": len(replay_evaluation_episodes),
        "replay_evaluation_source": (
            "disjoint_original_validation"
            if replay_probe
            else "legacy_training_replay_subset"
        ),
        "replay_test_count": len(replay_test_episodes),
        "replay_test_policy": (
            "original_test_evaluated_only_after_model_selection"
            if replay_test
            else "missing"
        ),
        "last_train": last_train,
        "data_provenance": final_data_provenance,
        "data_contract_sha256": data_contract["sha256"],
        "split_contract_sha256": split_contract["sha256"],
        "artifact_status": release["artifact_status"],
        "formal_release_eligible": release["formal_release_eligible"],
        "formal_release_decision": release,
        "training_scope": final_data_provenance.get("training_scope"),
        "formal_training_enabled": final_data_provenance.get(
            "formal_training_enabled"
        ),
        "temporal_unit_policy": final_data_provenance.get(
            "temporal_unit_policy"
        ),
        "training_policy": training_policy,
        "native_batching_audit": native_batching_audit,
        "frozen_weight_max_abs_error": (
            frozen_weight_error if head_policy is not None else None
        ),
    }
    _atomic_json_save(summary, output_dir / "training_summary.json")
    print(json.dumps(summary, sort_keys=True), flush=True)
    if head_policy is not None:
        head_policy.remove()
    return summary


__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_NATIVE_BATCHING",
    "DomainSpeakerBalancedSampler",
    "MAX_FULL_FINETUNE_LR",
    "MAX_HEAD_PROJECTION_LR",
    "ModelEMA",
    "NativeLengthBucketSampler",
    "POSTTRAIN_ARTIFACT_KIND",
    "SourceSpeakerActivityBalancedSampler",
    "batch_tensors",
    "build_data_provenance",
    "canonicalize_beat_episodes",
    "canonicalize_kimodo_replay",
    "deterministic_replay_evaluation_subset",
    "episode_group_keys",
    "head_activity_rms_velocity",
    "evaluate_posttrain",
    "load_attached_beat_episodes",
    "load_kimodo_replay_episodes",
    "load_kimodo_replay_splits",
    "native_variable_length_batch_tensors",
    "native_length_bucket",
    "native_length_microbatch_capacity",
    "masked_18d_objective",
    "posttrain_release_decision",
    "replay_regression_guard",
    "resolve_posttrain_config",
    "strict_group_split",
    "train_18d_posttrain",
    "validate_strict_group_splits",
]
