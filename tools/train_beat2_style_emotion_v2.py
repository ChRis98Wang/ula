#!/usr/bin/env python3
"""BEAT2-only text style/emotion post-training with hard anti-collapse gates.

This is intentionally independent from the legacy metadata A/B post-trainer.
It starts from the completed clean V7 AdaLN foundation, consumes only the
frozen-Qwen BEAT2 cache, and never places a per-trajectory oracle style value
in the generator condition.  The three style channels are predicted from the
128D text latent by :class:`QwenStyleHead`; the trajectory controls remain
target-only auxiliary supervision.

The official BEAT2 filename emotion/category/intensity metadata is useful for
experimental grouping, but it is not human-verified robot affect
observability.  Consequently every emitted checkpoint is experimental-only
and is never marked as formally release eligible.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import fnmatch
import hashlib
import io
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.train_beat2_experimental_metadata_posttrain_ab import (  # noqa: E402
    EXPECTED_SPLIT_COUNTS,
    _foundation_receipt,
    _load_128d_cache,
    _load_clean_episodes,
    _resolve_device,
    _seed_everything,
    _stable_subset,
    _state_dict_sha256,
    _strip_namespace,
    _validate_style_cache_lineage,
)
from tools.train_beat2_qwen_motion_alignment import (  # noqa: E402
    NO_EXTERNAL_DATA_POLICY,
    SEMANTIC_SCOPE_ACKNOWLEDGEMENT,
)
from upper_body_skeleton.beat2_condition_control import (  # noqa: E402
    CONDITION_DIM,
    STYLE_CONTROL_SLICE,
    TEXT_LATENT_DIM,
    TEXT_LATENT_SLICE,
    QwenStyleHead,
    aligned_vs_rolled_shuffled_hinge_loss,
    assemble_text_style_conditions,
    masked_per_example_flow_mse,
    masked_style_control_smooth_l1,
    sample_condition_keep_mask,
)
from upper_body_skeleton.beat2_semantic_perceptual import (  # noqa: E402
    Beat2SemanticPerceptualLoss,
)
from upper_body_skeleton.ula_training import (  # noqa: E402
    ULA_MMDIT_V3_ADALN_ARCHITECTURE,
)
from upper_body_skeleton.ula_v2_18d_head import (  # noqa: E402
    ACTION_DIM,
    load_contract_checkpoint,
    sha256_file,
    validate_motion_only_checkpoint_isolation,
)
from upper_body_skeleton.ula_v2_18d_posttrain import (  # noqa: E402
    ModelEMA,
    _batch_tensors_for_config,
    masked_18d_objective,
    native_length_bucket,
    native_length_microbatch_capacity,
    strict_group_split,
)
from upper_body_skeleton.ula_v2_18d_random_init import (  # noqa: E402
    forward_with_frame_mask,
)


SCHEMA_VERSION = 2
CONFIG_ARTIFACT_KIND = "beat2_text_style_emotion_posttrain_config_v2"
BRIDGE_ARTIFACT_KIND = "beat2_text_style_emotion_no_oracle_bridge_v2"
STATE_ARTIFACT_KIND = "beat2_text_style_emotion_posttrain_state_v2"
CHECKPOINT_ARTIFACT_KIND = "beat2_text_style_emotion_generator_v2"
SUMMARY_ARTIFACT_KIND = "beat2_text_style_emotion_training_summary_v2"
CONDITION_POLICY = (
    "zero_base_0_133_qwen_predicted_style_133_136_"
    "frozen_qwen_text_136_264_no_trajectory_oracle_v2"
)
TRAINING_POLICY = (
    "adaln_condition_path_plus_qwen_style_head_per_episode_rank_"
    "balanced_group_gate_global54_semantic_perceptual_v6"
)
SAMPLING_POLICY = "tempered_joint_text_group_native_length_v1"
NEGATIVE_POLICY = (
    "deterministic_cross_semantic_group_shared_flow_state_"
    "per_episode_hinge_v2"
)
TEXT_DROPOUT_POLICY = "deterministic_full_text_and_predicted_style_dropout_v1"
FORBIDDEN_EXTERNAL_TOKEN = "kimodo"

MODEL_TRAINABLE_PATTERNS = (
    "motion_latent_condition.*",
    "condition_pool.*",
    "blocks.*.modulation.*",
    "output_modulation.*",
    "plan.*",
    "duration_head.*",
)
EXPLICIT_FROZEN_PATTERNS = (
    "input.*",
    "time.*",
    "time_mlp.*",
    "frame.*",
    "blocks.*.attn.*",
    "blocks.*.ffn.*",
    "output.*",
    "style_condition.*",
    "legacy_condition.*",
    "behavior_condition.*",
    "emotion_condition.*",
    "family_condition.*",
    "transition_head.*",
)


DEFAULT_CONFIG = {
    "schema_version": SCHEMA_VERSION,
    "artifact_kind": CONFIG_ARTIFACT_KIND,
    "semantic_scope_acknowledgement": SEMANTIC_SCOPE_ACKNOWLEDGEMENT,
    "data_policy": NO_EXTERNAL_DATA_POLICY,
    "condition_policy": CONDITION_POLICY,
    "training_policy": TRAINING_POLICY,
    "manifest_path": (
        "/home/gez/nas/cloud/gez/human_motion/processed/"
        "beat2_semantic_event_training_pool_18d_v7_full/"
        "adjudication_min30f/train_ready.jsonl"
    ),
    "expected_manifest_sha256": (
        "2b3692c4f0a9ea8e10f3bde74fa178800556ed9bb79d0918a770b963a7f7c7fd"
    ),
    "foundation_checkpoint": (
        "/home/gez/shuaiwang/ula-motion-generate/training/runs/"
        "beat2_18d_from_scratch_formal_v7_clean_adaln/training/"
        "ula_fm_checkpoint.pt"
    ),
    "foundation_training_summary": (
        "/home/gez/shuaiwang/ula-motion-generate/training/runs/"
        "beat2_18d_from_scratch_formal_v7_clean_adaln/training/"
        "training_summary.json"
    ),
    "style_condition_cache": (
        "/home/gez/shuaiwang/ula-motion-generate/training/runs/"
        "beat2_18d_from_scratch_formal_v7_clean_adaln/conditioning/"
        "conditions.npz"
    ),
    "frozen_condition_cache": (
        "/home/gez/shuaiwang/ula-motion-generate/training/runs/"
        "beat2_qwen_motion_alignment_ab_v1/"
        "conditions_128d_frozen_base.npz"
    ),
    "output_dir": (
        "/home/gez/shuaiwang/ula-motion-generate/training/runs/"
        "beat2_text_style_emotion_v2_global54_semantic_v6"
    ),
    "dataset_source": "beat2_official_semantic_event_training_pool_v7",
    "speaker_namespace": "beat2",
    "source_group_namespace": "beat2-official-semantic-event",
    "seed": 20260727,
    "device": "cuda",
    "semantic_perceptual": {
        "enabled": True,
        "descriptor_cache": (
            "/home/gez/shuaiwang/ula-motion-generate/training/runs/"
            "beat2_qwen_motion_alignment_ab_v1/"
            "beat2_motion_descriptors_v1.npz"
        ),
        "expected_descriptor_cache_sha256": (
            "e793eb5f60d8d5a61c22a431c8aa15925d4592924d91f76c85b4134b1d0e3f52"
        ),
        "motion_encoder_checkpoint": (
            "/home/gez/shuaiwang/ula-motion-generate/training/runs/"
            "beat2_qwen_motion_alignment_ab_v1/motion_encoder_best.pt"
        ),
        "expected_motion_encoder_sha256": (
            "89b45b4f5d40af443e105d7c24678dc0720ec041f5254b3041631bd322d85ef0"
        ),
        "cosine_weight": 0.25,
        "contrastive_weight": 0.0,
        "global_contrastive_weight": 1.0,
        "temperature": 0.07,
        "outer_weight": 0.02,
        "warmup_steps": 500,
        "use_global_train_prototype_bank": True,
        "use_in_batch_contrastive": False,
        "prototype_aggregation": "require_identical",
        "prototype_consistency_tolerance": 1e-6,
        "apply_only_to_condition_kept": True,
        "reconstruction_policy": (
            "shared_aligned_forward_x_t_plus_one_minus_t_times_velocity_v1"
        ),
    },
    "style_head": {
        "text_latent_dim": 128,
        "hidden_dim": 128,
        "style_dim": 3,
        "zero_initialize_output": True,
    },
    "training": {
        "steps": 80_000,
        "batch_size": 16,
        "lr": 2e-5,
        "style_head_lr": 1e-4,
        "minimum_lr_ratio": 0.1,
        "warmup_steps": 1_000,
        "weight_decay": 1e-4,
        "adam_eps": 1e-6,
        "max_grad_norm": 1.0,
        "ema_decay": 0.9995,
        "validation_interval": 1_000,
        "checkpoint_interval": 500,
        "log_interval": 25,
        "evaluation_episode_count": 54,
        "condition_dropout_probability": 0.10,
        "style_smooth_l1_weight": 0.25,
        "condition_ranking_weight": 50.0,
        "condition_ranking_margin": 0.005,
        "condition_response_floor": 0.04,
        "condition_response_floor_weight": 10.0,
        "loss": {
            "flow": 1.0,
            "position": 1.0,
            "body": 0.0,
            "velocity": 0.02,
            "acceleration": 0.0015,
            "jerk": 0.00004,
            "head_flow": 1.0,
            "head_position": 0.5,
            "head_velocity": 0.01,
            "head_acceleration": 0.0008,
            "head_jerk": 0.00004,
            "planner_duration": 0.1,
            "planner_transition": 0.0,
        },
        "batching": {
            "mode": "native_variable_length",
            "length_buckets": [48, 64, 96, 128, 192, 256, 384, 512],
            "homogeneous_bucket_batches": True,
            "max_motion_tokens_per_microbatch": 4_096,
            "max_attention_elements_per_microbatch": 8_000_000,
            "target_effective_batch_size": 16,
            "gradient_accumulation_mode": "dynamic_episode_weighted",
            "oversize_sequence_policy": "single_full_episode_or_fail",
        },
        "sampler": {
            "mode": SAMPLING_POLICY,
            "reference_group_size": 37.5,
            "maximum_group_multiplier": 4.0,
            "low_confidence_group_count": 10,
        },
    },
    "anti_collapse_gates": {
        "minimum_aligned_vs_zero_prediction_rms": 0.003,
        "minimum_response_retention_ratio": 0.75,
        "minimum_aligned_vs_cross_group_prediction_rms": 0.002,
        "minimum_cross_group_minus_aligned_flow_loss": 0.0001,
        "minimum_correct_flow_win_rate": 0.55,
        "maximum_duration_mae_sec": 1.0,
        "maximum_style_smooth_l1": 1.5,
    },
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _deep_merge(defaults: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    unknown = sorted(set(override) - set(defaults))
    if unknown:
        raise ValueError(f"unknown text style/emotion config keys: {unknown}")
    result = deepcopy(dict(defaults))
    for key, value in override.items():
        if isinstance(result.get(key), Mapping):
            if not isinstance(value, Mapping):
                raise ValueError(f"config section {key!r} must be a mapping")
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _iter_string_values(value: Any, path: str = "config"):
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield from _iter_string_values(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for index, child in enumerate(value):
            yield from _iter_string_values(child, f"{path}[{index}]")


def _reject_forbidden_external_tokens(value: Mapping[str, Any]) -> None:
    for field, text in _iter_string_values(value):
        if FORBIDDEN_EXTERNAL_TOKEN in text.lower():
            raise ValueError(
                f"{field} contains a forbidden external-data token"
            )


def _resolved_safe_path(value: str | Path, *, field: str) -> str:
    path = Path(value).expanduser().resolve()
    if FORBIDDEN_EXTERNAL_TOKEN in str(path).lower():
        raise ValueError(f"{field} contains a forbidden external-data token")
    return str(path)


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def _finite_nonnegative(value: Any, *, field: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


def _semantic_weight_at_step(
    step: int, *, target_weight: float, warmup_steps: int
) -> float:
    """Linearly introduce the frozen semantic critic without a loss shock."""

    if isinstance(step, bool) or int(step) != step or int(step) < 0:
        raise ValueError("semantic loss step must be a non-negative integer")
    target_weight = _finite_nonnegative(
        target_weight, field="semantic_perceptual.outer_weight"
    )
    warmup_steps = _positive_int(
        warmup_steps, field="semantic_perceptual.warmup_steps"
    )
    return float(target_weight) * min(1.0, float(step) / float(warmup_steps))


def validate_config(config: Mapping[str, Any]) -> dict:
    """Resolve and strictly validate the independent V2 training contract."""
    values = _deep_merge(DEFAULT_CONFIG, config)
    _reject_forbidden_external_tokens(values)
    exact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": CONFIG_ARTIFACT_KIND,
        "semantic_scope_acknowledgement": SEMANTIC_SCOPE_ACKNOWLEDGEMENT,
        "data_policy": NO_EXTERNAL_DATA_POLICY,
        "condition_policy": CONDITION_POLICY,
        "training_policy": TRAINING_POLICY,
    }
    for field, expected in exact.items():
        if values.get(field) != expected:
            raise ValueError(f"{field} must be {expected!r}")
    manifest_hash = str(values["expected_manifest_sha256"])
    if (
        len(manifest_hash) != 64
        or any(character not in "0123456789abcdef" for character in manifest_hash)
    ):
        raise ValueError("expected_manifest_sha256 must be a lowercase SHA256")
    for field in (
        "manifest_path",
        "foundation_checkpoint",
        "foundation_training_summary",
        "style_condition_cache",
        "frozen_condition_cache",
        "output_dir",
    ):
        values[field] = _resolved_safe_path(values[field], field=field)
    for field in ("dataset_source", "speaker_namespace", "source_group_namespace"):
        if not str(values.get(field) or "").strip():
            raise ValueError(f"{field} is required")
        values[field] = str(values[field])
    values["seed"] = int(values["seed"])
    values["device"] = str(values["device"])

    semantic = values["semantic_perceptual"]
    fixed_semantic = {
        "enabled": True,
        "apply_only_to_condition_kept": True,
        "use_global_train_prototype_bank": True,
        "use_in_batch_contrastive": False,
        "prototype_aggregation": "require_identical",
        "reconstruction_policy": (
            "shared_aligned_forward_x_t_plus_one_minus_t_times_velocity_v1"
        ),
    }
    for field, expected in fixed_semantic.items():
        if semantic.get(field) != expected:
            raise ValueError(
                f"semantic_perceptual.{field} must be {expected!r}"
            )
    for field in ("descriptor_cache", "motion_encoder_checkpoint"):
        semantic[field] = _resolved_safe_path(
            semantic[field], field=f"semantic_perceptual.{field}"
        )
    for field in (
        "expected_descriptor_cache_sha256",
        "expected_motion_encoder_sha256",
    ):
        digest = str(semantic[field])
        if (
            len(digest) != 64
            or any(
                character not in "0123456789abcdef" for character in digest
            )
        ):
            raise ValueError(
                f"semantic_perceptual.{field} must be a lowercase SHA256"
            )
        semantic[field] = digest
    for field in (
        "cosine_weight",
        "contrastive_weight",
        "global_contrastive_weight",
        "temperature",
        "outer_weight",
        "prototype_consistency_tolerance",
    ):
        semantic[field] = _finite_nonnegative(
            semantic[field], field=f"semantic_perceptual.{field}"
        )
    semantic["warmup_steps"] = _positive_int(
        semantic["warmup_steps"],
        field="semantic_perceptual.warmup_steps",
    )
    if (
        semantic["cosine_weight"] <= 0
        or semantic["contrastive_weight"] != 0
        or semantic["global_contrastive_weight"] <= 0
        or semantic["temperature"] <= 0
        or semantic["outer_weight"] <= 0
    ):
        raise ValueError(
            "global semantic perceptual constants are invalid"
        )

    head = values["style_head"]
    expected_head = {
        "text_latent_dim": TEXT_LATENT_DIM,
        "style_dim": STYLE_CONTROL_SLICE.stop - STYLE_CONTROL_SLICE.start,
        "zero_initialize_output": True,
    }
    for field, expected in expected_head.items():
        if head.get(field) != expected:
            raise ValueError(f"style_head.{field} must be {expected!r}")
    head["hidden_dim"] = _positive_int(
        head["hidden_dim"], field="style_head.hidden_dim"
    )

    training = values["training"]
    for field in (
        "steps",
        "batch_size",
        "warmup_steps",
        "validation_interval",
        "checkpoint_interval",
        "log_interval",
        "evaluation_episode_count",
    ):
        training[field] = _positive_int(
            training[field], field=f"training.{field}"
        )
    if training["warmup_steps"] > training["steps"]:
        raise ValueError("training.warmup_steps cannot exceed training.steps")
    if semantic["warmup_steps"] > training["steps"]:
        raise ValueError(
            "semantic_perceptual.warmup_steps cannot exceed training.steps"
        )
    for field in (
        "lr",
        "style_head_lr",
        "minimum_lr_ratio",
        "weight_decay",
        "adam_eps",
        "max_grad_norm",
        "ema_decay",
        "condition_dropout_probability",
        "style_smooth_l1_weight",
        "condition_ranking_weight",
        "condition_ranking_margin",
        "condition_response_floor",
        "condition_response_floor_weight",
    ):
        training[field] = _finite_nonnegative(
            training[field], field=f"training.{field}"
        )
    if not 0 < training["lr"] <= 1e-3:
        raise ValueError("training.lr must be in (0, 1e-3]")
    if not 0 < training["style_head_lr"] <= 1e-2:
        raise ValueError("training.style_head_lr must be in (0, 1e-2]")
    if not 0 < training["minimum_lr_ratio"] <= 1:
        raise ValueError("training.minimum_lr_ratio must be in (0, 1]")
    if not 0 < training["ema_decay"] < 1:
        raise ValueError("training.ema_decay must be in (0, 1)")
    if not 0 <= training["condition_dropout_probability"] < 1:
        raise ValueError(
            "training.condition_dropout_probability must be in [0, 1)"
        )
    if (
        training["style_smooth_l1_weight"] <= 0
        or training["condition_ranking_weight"] <= 0
        or training["condition_ranking_margin"] <= 0
        or training["condition_response_floor"] <= 0
        or training["condition_response_floor_weight"] <= 0
    ):
        raise ValueError(
            "style, shared-state ranking, and ranking margin must be positive"
        )
    expected_losses = set(DEFAULT_CONFIG["training"]["loss"])
    if set(training["loss"]) != expected_losses:
        raise ValueError("training.loss fields changed")
    for field in sorted(training["loss"]):
        training["loss"][field] = _finite_nonnegative(
            training["loss"][field], field=f"training.loss.{field}"
        )
    if (
        training["loss"]["body"] != 0
        or training["loss"]["planner_transition"] != 0
        or training["loss"]["planner_duration"] <= 0
    ):
        raise ValueError(
            "body/transition must remain zero and planner_duration must be positive"
        )

    batching = training["batching"]
    fixed_batching = {
        "mode": "native_variable_length",
        "homogeneous_bucket_batches": True,
        "gradient_accumulation_mode": "dynamic_episode_weighted",
        "oversize_sequence_policy": "single_full_episode_or_fail",
    }
    for field, expected in fixed_batching.items():
        if batching.get(field) != expected:
            raise ValueError(f"training.batching.{field} must be {expected!r}")
    batching["length_buckets"] = sorted(
        {
            _positive_int(value, field="training.batching.length_buckets")
            for value in batching["length_buckets"]
        }
    )
    for field in (
        "max_motion_tokens_per_microbatch",
        "max_attention_elements_per_microbatch",
        "target_effective_batch_size",
    ):
        batching[field] = _positive_int(
            batching[field], field=f"training.batching.{field}"
        )
    sampler = training["sampler"]
    if sampler.get("mode") != SAMPLING_POLICY:
        raise ValueError(f"training.sampler.mode must be {SAMPLING_POLICY!r}")
    sampler["reference_group_size"] = _finite_nonnegative(
        sampler["reference_group_size"],
        field="training.sampler.reference_group_size",
    )
    sampler["maximum_group_multiplier"] = _finite_nonnegative(
        sampler["maximum_group_multiplier"],
        field="training.sampler.maximum_group_multiplier",
    )
    sampler["low_confidence_group_count"] = _positive_int(
        sampler["low_confidence_group_count"],
        field="training.sampler.low_confidence_group_count",
    )
    if (
        sampler["reference_group_size"] <= 0
        or sampler["maximum_group_multiplier"] < 1
    ):
        raise ValueError("tempered group sampler constants are invalid")

    gates = values["anti_collapse_gates"]
    if set(gates) != set(DEFAULT_CONFIG["anti_collapse_gates"]):
        raise ValueError("anti_collapse_gates fields changed")
    for field in gates:
        gates[field] = _finite_nonnegative(
            gates[field], field=f"anti_collapse_gates.{field}"
        )
    if gates["minimum_response_retention_ratio"] <= 0:
        raise ValueError("minimum response retention must be positive")
    return values


def read_config(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("text style/emotion config must be a JSON object")
    return validate_config(value)


def effective_config(
    config: Mapping[str, Any],
    *,
    smoke_test: bool,
    device: str | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    values = deepcopy(dict(config))
    if device is not None:
        values["device"] = str(device)
    if output_dir is not None:
        values["output_dir"] = str(output_dir)
    elif smoke_test:
        values["output_dir"] = str(Path(values["output_dir"] + "_smoke"))
    if smoke_test:
        training = values["training"]
        training.update(
            {
                "steps": 2,
                "batch_size": 4,
                "warmup_steps": 1,
                "validation_interval": 1,
                "checkpoint_interval": 1,
                "log_interval": 1,
                "evaluation_episode_count": 8,
            }
        )
        training["batching"]["target_effective_batch_size"] = 4
        values["semantic_perceptual"]["warmup_steps"] = 1
    return validate_config(values)


def _semantic_perceptual_receipt(config: Mapping[str, Any]) -> dict:
    semantic = config["semantic_perceptual"]
    artifacts = {
        "descriptor_cache": (
            Path(semantic["descriptor_cache"]),
            str(semantic["expected_descriptor_cache_sha256"]),
        ),
        "motion_encoder_checkpoint": (
            Path(semantic["motion_encoder_checkpoint"]),
            str(semantic["expected_motion_encoder_sha256"]),
        ),
    }
    receipt = {
        "enabled": True,
        "data_policy": NO_EXTERNAL_DATA_POLICY,
        "no_external_data": True,
        "no_kimodo": True,
        "apply_only_to_condition_kept": True,
        "reconstruction_policy": semantic["reconstruction_policy"],
        "cosine_weight": float(semantic["cosine_weight"]),
        "contrastive_weight": float(semantic["contrastive_weight"]),
        "global_contrastive_weight": float(
            semantic["global_contrastive_weight"]
        ),
        "temperature": float(semantic["temperature"]),
        "outer_weight": float(semantic["outer_weight"]),
        "warmup_steps": int(semantic["warmup_steps"]),
        "use_global_train_prototype_bank": True,
        "use_in_batch_contrastive": False,
        "prototype_fit_split": "train",
        "prototype_aggregation": semantic["prototype_aggregation"],
        "prototype_consistency_tolerance": float(
            semantic["prototype_consistency_tolerance"]
        ),
        "validation_or_test_rows_used_for_prototypes": 0,
    }
    for field, (path, expected_sha256) in artifacts.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"semantic perceptual {field} does not exist: {path}"
            )
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"semantic perceptual {field} SHA256 changed: "
                f"{actual_sha256} != {expected_sha256}"
            )
        receipt[field] = {
            "path": str(path.resolve()),
            "sha256": actual_sha256,
        }
    qwen_cache = Path(config["frozen_condition_cache"])
    if not qwen_cache.is_file():
        raise FileNotFoundError(
            f"global semantic Qwen cache does not exist: {qwen_cache}"
        )
    receipt["global_prototype_source_qwen_cache"] = {
        "path": str(qwen_cache.resolve()),
        "sha256": sha256_file(qwen_cache),
    }
    return receipt


def _build_semantic_perceptual(
    config: Mapping[str, Any],
    *,
    action_stats: Mapping[str, Any],
    device: torch.device,
) -> tuple[Beat2SemanticPerceptualLoss, dict]:
    receipt = _semantic_perceptual_receipt(config)
    semantic = config["semantic_perceptual"]
    module = Beat2SemanticPerceptualLoss.from_artifacts(
        descriptor_cache_path=receipt["descriptor_cache"]["path"],
        motion_encoder_checkpoint_path=receipt[
            "motion_encoder_checkpoint"
        ]["path"],
        action_stats=action_stats,
        qwen_condition_cache_path=receipt[
            "global_prototype_source_qwen_cache"
        ]["path"],
        prototype_aggregation=semantic["prototype_aggregation"],
        prototype_consistency_tolerance=float(
            semantic["prototype_consistency_tolerance"]
        ),
        cosine_weight=float(semantic["cosine_weight"]),
        contrastive_weight=float(semantic["contrastive_weight"]),
        global_contrastive_weight=float(
            semantic["global_contrastive_weight"]
        ),
        temperature=float(semantic["temperature"]),
        validate_inputs=False,
        device=device,
    )
    module.train()
    if any(parameter.requires_grad for parameter in module.parameters()):
        raise RuntimeError("semantic perceptual critic must stay fully frozen")
    return module, receipt


def _atomic_json_save(value: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(target)


def _atomic_torch_save(value: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(target)


def _atomic_npz_save(path: str | Path, **arrays) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_bytes(buffer.getvalue())
    temporary.replace(target)


def _append_jsonl(value: Mapping[str, Any], path: str | Path) -> None:
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
            + "\n"
        )


def tempered_group_weights(
    group_counts: Mapping[int, int],
    *,
    reference_group_size: float = 37.5,
    maximum_multiplier: float = 4.0,
) -> dict[int, float]:
    """Return mean-one per-example multipliers for joint text groups."""
    if not group_counts or any(int(count) <= 0 for count in group_counts.values()):
        raise ValueError("group counts must be positive")
    raw = {
        int(group): min(
            float(maximum_multiplier),
            math.sqrt(float(reference_group_size) / float(count)),
        )
        for group, count in group_counts.items()
    }
    example_weighted_mean = sum(
        raw[int(group)] * int(count) for group, count in group_counts.items()
    ) / float(sum(int(count) for count in group_counts.values()))
    return {
        group: float(weight / example_weighted_mean)
        for group, weight in raw.items()
    }


def _identity_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    payload = {
        name: arrays[name].astype(str).tolist()
        for name in (
            "clip_ids",
            "prompts",
            "fixed_split_assignments",
            "speaker_keys",
            "semantic_group_indices",
            "trajectory_sha256",
        )
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _prepare_rows(
    config: Mapping[str, Any], *, smoke_test: bool
) -> tuple[dict[str, list[dict]], dict, dict, dict]:
    """Load strict sources and discard every oracle generator style condition."""
    foundation_checkpoint, foundation_receipt = _foundation_receipt(
        config, smoke_test=smoke_test
    )
    style_payload, style_metadata = _validate_style_cache_lineage(
        config, foundation_checkpoint, foundation_receipt
    )
    qwen, qwen_metadata = _load_128d_cache(
        config["frozen_condition_cache"],
        variant="frozen_base",
        config=config,
    )
    clean_episodes = _load_clean_episodes(config)
    episode_by_clip = {str(row["clip_id"]): row for row in clean_episodes}
    style_index = {
        str(clip_id): index
        for index, clip_id in enumerate(style_payload["clip_ids"].astype(str))
    }
    qwen_clip_ids = qwen["clip_ids"].astype(str).tolist()
    if (
        len(set(qwen_clip_ids)) != len(qwen_clip_ids)
        or set(qwen_clip_ids) != set(episode_by_clip)
        or set(style_index) != set(episode_by_clip)
    ):
        raise ValueError(
            "manifest, strict style cache, and frozen-Qwen cache clip sets differ"
        )
    style_records = {
        str(row["clip_id"]): row
        for row in style_metadata.get("episodes") or ()
    }
    foundation_splits = {
        str(row["clip_id"]): str(row["split"])
        for row in (
            (foundation_checkpoint.get("v2_contracts") or {})
            .get("split", {})
            .get("episodes", ())
        )
    }
    if set(style_records) != set(episode_by_clip) or set(foundation_splits) != set(
        episode_by_clip
    ):
        raise ValueError("foundation/style split lineage is incomplete")

    rows = []
    generator_conditions = []
    style_targets = []
    style_features = []
    for qwen_index, clip_id in enumerate(qwen_clip_ids):
        episode = episode_by_clip[clip_id]
        style_row = style_index[clip_id]
        style_record = style_records[clip_id]
        split_name = str(qwen["fixed_split_assignments"][qwen_index])
        prompt = str(qwen["prompts"][qwen_index])
        trajectory = str(qwen["trajectory_sha256"][qwen_index])
        speaker = str(qwen["speaker_keys"][qwen_index])
        if (
            prompt != str(style_payload["prompts"][style_row])
            or split_name != str(episode.get("fixed_split_assignment"))
            or split_name != str(style_record.get("fixed_split_assignment"))
            or split_name != foundation_splits[clip_id]
            or trajectory != str(episode.get("trajectory_sha256"))
            or trajectory != str(style_record.get("trajectory_sha256"))
            or speaker
            != _strip_namespace(episode["speaker_key"], config["speaker_namespace"])
        ):
            raise ValueError(f"{clip_id}: source identity/split lineage changed")
        latent = np.asarray(qwen["conditions"][qwen_index], dtype=np.float32)
        target_style = np.asarray(
            style_payload["style_controls"][style_row], dtype=np.float32
        )
        feature_style = np.asarray(
            style_payload["style_features"][style_row], dtype=np.float32
        )
        condition = np.zeros(CONDITION_DIM, dtype=np.float32)
        condition[TEXT_LATENT_SLICE] = latent
        if (
            np.any(condition[: TEXT_LATENT_SLICE.start] != 0.0)
            or np.any(condition[STYLE_CONTROL_SLICE] != 0.0)
        ):
            raise RuntimeError("generator preparation leaked an oracle style value")
        item = dict(episode)
        item["condition"] = condition
        item.pop("condition_cache_provenance", None)
        item["experimental_semantic_group_index"] = int(
            qwen["semantic_group_indices"][qwen_index]
        )
        item["qwen_text_latent_128d"] = latent.copy()
        item["continuous_style_training_target"] = target_style.copy()
        item["continuous_style_feature_target"] = feature_style.copy()
        item["experimental_condition_contract"] = {
            "artifact_kind": BRIDGE_ARTIFACT_KIND,
            "condition_policy": CONDITION_POLICY,
            "oracle_style_in_generator_condition": False,
            "continuous_style_target_role": (
                "target_only_arm_balance_amplitude_speed_not_emotion_truth"
            ),
            "formal_release_eligible": False,
        }
        rows.append(item)
        generator_conditions.append(condition)
        style_targets.append(target_style)
        style_features.append(feature_style)

    splits, split_contract = strict_group_split(
        rows,
        seed=int(config["seed"]),
        fractions={"train": 0.7, "validation": 0.15, "test": 0.15},
    )
    split_counts = {name: len(values) for name, values in splits.items()}
    if not smoke_test and split_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"fixed BEAT2 split counts changed: {split_counts}")
    train_group_counts = Counter(
        int(row["experimental_semantic_group_index"]) for row in splits["train"]
    )
    sampler_config = config["training"]["sampler"]
    group_weights = tempered_group_weights(
        train_group_counts,
        reference_group_size=float(sampler_config["reference_group_size"]),
        maximum_multiplier=float(sampler_config["maximum_group_multiplier"]),
    )
    low_threshold = int(sampler_config["low_confidence_group_count"])
    for split_rows in splits.values():
        for row in split_rows:
            group = int(row["experimental_semantic_group_index"])
            row["tempered_group_sampling_weight"] = float(
                group_weights.get(group, 0.0)
            )
            row["text_group_train_count"] = int(train_group_counts.get(group, 0))
            row["text_group_low_confidence"] = bool(
                train_group_counts.get(group, 0) < low_threshold
            )

    arrays = {
        "clip_ids": np.asarray(qwen_clip_ids),
        "prompts": qwen["prompts"].astype(str),
        "generator_input_conditions": np.stack(generator_conditions).astype(
            np.float32
        ),
        "qwen_text_latents": np.asarray(qwen["conditions"], dtype=np.float32),
        "continuous_style_targets": np.stack(style_targets).astype(np.float32),
        "continuous_style_features": np.stack(style_features).astype(np.float32),
        "fixed_split_assignments": qwen[
            "fixed_split_assignments"
        ].astype(str),
        "speaker_keys": qwen["speaker_keys"].astype(str),
        "semantic_group_indices": np.asarray(
            qwen["semantic_group_indices"], dtype=np.int64
        ),
        "trajectory_sha256": qwen["trajectory_sha256"].astype(str),
    }
    data_receipt = {
        "manifest": {
            "path": str(Path(config["manifest_path"]).resolve()),
            "sha256": config["expected_manifest_sha256"],
        },
        "foundation": foundation_receipt,
        "style_cache": {
            "path": style_metadata["path"],
            "sha256": style_metadata["cache_sha256"],
            "use": "continuous_target_only_never_generator_input",
        },
        "frozen_qwen_cache": {
            "path": qwen_metadata["path"],
            "sha256": qwen_metadata["cache_sha256"],
            "variant": "frozen_base",
        },
        "identity_arrays_sha256": _identity_sha256(arrays),
        "split_contract_sha256": split_contract["sha256"],
        "split_counts": split_counts,
        "train_group_counts": {
            str(group): int(count)
            for group, count in sorted(train_group_counts.items())
        },
        "normalized_tempered_group_weights": {
            str(group): float(weight)
            for group, weight in sorted(group_weights.items())
        },
        "low_confidence_group_indices": sorted(
            int(group)
            for group, count in train_group_counts.items()
            if count < low_threshold
        ),
        "semantic_claim": (
            "official_metadata_experimental_control_not_formal_robot_affect_truth"
        ),
        "continuous_style_target": (
            "standardized_arm_balance_log_amplitude_log_speed_target_only"
        ),
        "no_external_data": True,
        "no_kimodo": True,
    }
    return splits, split_contract, arrays, {
        "foundation_checkpoint": foundation_checkpoint,
        "data_receipt": data_receipt,
    }


def _bridge_paths(output_dir: str | Path) -> tuple[Path, Path]:
    cache = Path(output_dir) / "prepared" / "text_style_no_oracle_bridge_v2.npz"
    return cache, cache.with_suffix(cache.suffix + ".json")


def _write_or_validate_bridge(
    output_dir: str | Path,
    arrays: Mapping[str, np.ndarray],
    data_receipt: Mapping[str, Any],
) -> dict:
    cache_path, metadata_path = _bridge_paths(output_dir)
    conditions = np.asarray(
        arrays["generator_input_conditions"], dtype=np.float32
    )
    targets = np.asarray(arrays["continuous_style_targets"], dtype=np.float32)
    if (
        conditions.ndim != 2
        or conditions.shape[1] != CONDITION_DIM
        or targets.shape != (len(conditions), 3)
        or not np.isfinite(conditions).all()
        or not np.isfinite(targets).all()
        or np.any(conditions[:, : TEXT_LATENT_SLICE.start] != 0.0)
        or np.any(conditions[:, STYLE_CONTROL_SLICE] != 0.0)
    ):
        raise ValueError(
            "prepared bridge generator input contains an oracle style or invalid layout"
        )
    if cache_path.is_file() or metadata_path.is_file():
        if not cache_path.is_file() or not metadata_path.is_file():
            raise ValueError("prepared V2 bridge is incomplete")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("artifact_kind") != BRIDGE_ARTIFACT_KIND
            or metadata.get("condition_policy") != CONDITION_POLICY
            or metadata.get("cache_sha256") != sha256_file(cache_path)
            or metadata.get("source_identity_sha256")
            != data_receipt["identity_arrays_sha256"]
        ):
            raise ValueError("prepared bridge is not an isolated V2 artifact")
        with np.load(cache_path, allow_pickle=False) as payload:
            conditions = np.asarray(
                payload["generator_input_conditions"], dtype=np.float32
            )
            targets = np.asarray(
                payload["continuous_style_targets"], dtype=np.float32
            )
        if (
            conditions.shape[1:] != (CONDITION_DIM,)
            or targets.shape != (len(conditions), 3)
            or np.any(conditions[:, : TEXT_LATENT_SLICE.start] != 0.0)
            or np.any(conditions[:, STYLE_CONTROL_SLICE] != 0.0)
        ):
            raise ValueError("prepared bridge leaked target style into conditions")
        return metadata
    _atomic_npz_save(cache_path, **arrays)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": BRIDGE_ARTIFACT_KIND,
        "experimental_only": True,
        "formal_release_eligible": False,
        "condition_policy": CONDITION_POLICY,
        "cache_sha256": sha256_file(cache_path),
        "source_identity_sha256": data_receipt["identity_arrays_sha256"],
        "source_manifest_sha256": data_receipt["manifest"]["sha256"],
        "foundation_checkpoint_sha256": data_receipt["foundation"]["sha256"],
        "frozen_qwen_cache_sha256": data_receipt["frozen_qwen_cache"]["sha256"],
        "style_cache_sha256": data_receipt["style_cache"]["sha256"],
        "count": int(len(arrays["clip_ids"])),
        "layout": {
            "exact_zero_base": [0, TEXT_LATENT_SLICE.start],
            "predicted_style_runtime_only": [
                STYLE_CONTROL_SLICE.start,
                STYLE_CONTROL_SLICE.stop,
            ],
            "frozen_qwen_text_latent": [
                TEXT_LATENT_SLICE.start,
                TEXT_LATENT_SLICE.stop,
            ],
            "continuous_style_targets": (
                "separate_target_only_arm_balance_amplitude_speed"
            ),
        },
        "oracle_style_in_generator_condition": False,
        "semantic_claim": data_receipt["semantic_claim"],
        "no_external_data": True,
        "no_kimodo": True,
    }
    _atomic_json_save(metadata, metadata_path)
    return metadata


class TemperedGroupNativeBucketSampler:
    """Exact-resumable native-length sampler with tempered joint-group weights."""

    STATE_VERSION = 1

    def __init__(
        self,
        episodes: Sequence[Mapping[str, Any]],
        *,
        buckets: Sequence[int],
        seed: int,
    ):
        if not episodes:
            raise ValueError("tempered native sampler requires episodes")
        self.buckets = tuple(sorted({int(value) for value in buckets}))
        self.seed = int(seed)
        grouped: dict[int, list[dict]] = defaultdict(list)
        for episode in episodes:
            frames = int(np.asarray(episode["actions"]).shape[0])
            grouped[native_length_bucket(frames, self.buckets)].append(
                dict(episode)
            )
        self.bucket_episodes = {
            bucket: sorted(rows, key=lambda row: str(row["clip_id"]))
            for bucket, rows in sorted(grouped.items())
        }
        self.bucket_weights = {
            bucket: [
                float(row["tempered_group_sampling_weight"]) for row in rows
            ]
            for bucket, rows in self.bucket_episodes.items()
        }
        if any(
            not weights or any(weight <= 0 or not math.isfinite(weight) for weight in weights)
            for weights in self.bucket_weights.values()
        ):
            raise ValueError("tempered native sampler weights are invalid")
        self.schedule_rng = random.Random(self.seed + 7919)
        self.choice_rngs = {
            bucket: random.Random(
                self.seed
                + int.from_bytes(
                    hashlib.sha256(f"group-bucket:{bucket}".encode("ascii")).digest()[
                        :4
                    ],
                    byteorder="big",
                )
            )
            for bucket in self.bucket_episodes
        }
        self.bucket_schedule = [
            bucket
            for bucket, rows in self.bucket_episodes.items()
            for _ in range(len(rows))
        ]
        self.schedule_rng.shuffle(self.bucket_schedule)
        self.schedule_cursor = 0
        structure = {
            "buckets": list(self.buckets),
            "membership": {
                str(bucket): [str(row["clip_id"]) for row in rows]
                for bucket, rows in self.bucket_episodes.items()
            },
            "weights": {
                str(bucket): self.bucket_weights[bucket]
                for bucket in self.bucket_episodes
            },
            "policy": SAMPLING_POLICY,
        }
        self.structure_sha256 = hashlib.sha256(
            _canonical_json(structure)
        ).hexdigest()

    def _next_bucket(self) -> int:
        if self.schedule_cursor >= len(self.bucket_schedule):
            self.schedule_rng.shuffle(self.bucket_schedule)
            self.schedule_cursor = 0
        bucket = int(self.bucket_schedule[self.schedule_cursor])
        self.schedule_cursor += 1
        return bucket

    def sample_microbatch(
        self,
        *,
        remaining_effective_batch: int,
        semantic_tokens: int,
        max_batch_size: int,
        batching: Mapping[str, Any],
    ) -> tuple[list[dict], dict]:
        bucket = self._next_bucket()
        plan = native_length_microbatch_capacity(
            bucket,
            semantic_tokens=int(semantic_tokens),
            max_batch_size=int(max_batch_size),
            max_motion_tokens=int(
                batching["max_motion_tokens_per_microbatch"]
            ),
            max_attention_elements=int(
                batching["max_attention_elements_per_microbatch"]
            ),
        )
        count = min(int(remaining_effective_batch), int(plan["capacity"]))
        if count <= 0:
            raise ValueError("remaining effective batch must be positive")
        rows = self.choice_rngs[bucket].choices(
            self.bucket_episodes[bucket],
            weights=self.bucket_weights[bucket],
            k=count,
        )
        return [dict(row) for row in rows], plan | {
            "microbatch_size": count,
            "motion_tokens": count * bucket,
            "attention_elements": count
            * int(plan["per_episode_attention_elements"]),
            "sampling_policy": SAMPLING_POLICY,
        }

    def state_dict(self) -> dict:
        return {
            "state_version": self.STATE_VERSION,
            "structure_sha256": self.structure_sha256,
            "bucket_schedule": list(self.bucket_schedule),
            "schedule_cursor": int(self.schedule_cursor),
            "schedule_rng_state": self.schedule_rng.getstate(),
            "choice_rng_states": {
                bucket: rng.getstate() for bucket, rng in self.choice_rngs.items()
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (
            state.get("state_version") != self.STATE_VERSION
            or state.get("structure_sha256") != self.structure_sha256
        ):
            raise ValueError("tempered native sampler resume structure changed")
        schedule = [int(value) for value in state.get("bucket_schedule") or ()]
        if sorted(schedule) != sorted(self.bucket_schedule):
            raise ValueError("tempered native sampler schedule changed")
        cursor = int(state.get("schedule_cursor", -1))
        if not 0 <= cursor <= len(schedule):
            raise ValueError("tempered native sampler cursor is invalid")
        choice_states = dict(state.get("choice_rng_states") or {})
        if set(choice_states) != set(self.choice_rngs):
            raise ValueError("tempered native sampler bucket RNG set changed")
        self.bucket_schedule = schedule
        self.schedule_cursor = cursor
        self.schedule_rng.setstate(state["schedule_rng_state"])
        for bucket, rng in self.choice_rngs.items():
            rng.setstate(choice_states[bucket])


class CrossGroupNegativePool:
    """Deterministically select one different joint text group per row."""

    def __init__(self, episodes: Sequence[Mapping[str, Any]], *, seed: int):
        self.rows = sorted(
            (dict(row) for row in episodes), key=lambda row: str(row["clip_id"])
        )
        self.seed = int(seed)
        self.groups = {
            int(row["experimental_semantic_group_index"]) for row in self.rows
        }
        if len(self.groups) < 2:
            raise ValueError("cross-group ranking requires at least two text groups")

    def select(
        self,
        batch: Sequence[Mapping[str, Any]],
        *,
        step: int,
        microbatch_index: int,
    ) -> list[dict]:
        selected = []
        for row_index, row in enumerate(batch):
            group = int(row["experimental_semantic_group_index"])
            digest = hashlib.sha256(
                (
                    f"{self.seed}:{int(step)}:{int(microbatch_index)}:"
                    f"{row_index}:{row['clip_id']}"
                ).encode("utf-8")
            ).digest()
            start = int.from_bytes(digest[:8], byteorder="big") % len(self.rows)
            candidate = None
            for offset in range(len(self.rows)):
                value = self.rows[(start + offset) % len(self.rows)]
                if int(value["experimental_semantic_group_index"]) != group:
                    candidate = value
                    break
            if candidate is None:
                raise RuntimeError("could not construct a cross-group negative")
            selected.append(dict(candidate))
        return selected


def _matches_trainable_policy(name: str) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in MODEL_TRAINABLE_PATTERNS)


def configure_condition_optimizer(
    model: torch.nn.Module,
    style_head: QwenStyleHead,
    config: Mapping[str, Any],
) -> tuple[torch.optim.Optimizer, dict]:
    """Freeze the motion prior and expose only the declared condition path."""
    model.requires_grad_(False)
    trainable_names = []
    module_roots = set()
    for name, parameter in model.named_parameters():
        if _matches_trainable_policy(name):
            parameter.requires_grad_(True)
            trainable_names.append(name)
            module_roots.add(name.split(".", 1)[0])
    required_roots = {
        "motion_latent_condition",
        "condition_pool",
        "blocks",
        "output_modulation",
        "plan",
        "duration_head",
    }
    if module_roots != required_roots:
        raise ValueError(
            f"AdaLN condition-path allowlist is incomplete: {sorted(module_roots)}"
        )
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not _matches_trainable_policy(name)
    ]
    if unexpected:
        raise RuntimeError(f"unexpected trainable foundation parameters: {unexpected}")
    explicitly_frozen = {
        name
        for name, parameter in model.named_parameters()
        if any(
            fnmatch.fnmatchcase(name, pattern)
            for pattern in EXPLICIT_FROZEN_PATTERNS
        )
        and not parameter.requires_grad
    }
    for required in ("input.weight", "output.weight", "blocks.0.attn.in_proj_weight"):
        if required not in explicitly_frozen:
            raise ValueError(f"required frozen motion-prior tensor missing: {required}")
    style_head.requires_grad_(True)

    training = config["training"]
    model_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    head_parameters = list(style_head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model_parameters,
                "lr": float(training["lr"]),
                "weight_decay": float(training["weight_decay"]),
                "role": "adaln_condition_path",
            },
            {
                "params": head_parameters,
                "lr": float(training["style_head_lr"]),
                "weight_decay": float(training["weight_decay"]),
                "role": "qwen_style_head",
            },
        ],
        eps=float(training["adam_eps"]),
    )
    receipt = {
        "policy": TRAINING_POLICY,
        "full_network": False,
        "model_trainable_patterns": list(MODEL_TRAINABLE_PATTERNS),
        "model_trainable_tensor_names": sorted(trainable_names),
        "model_trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model_parameters)
        ),
        "style_head_trainable_parameter_count": int(
            sum(parameter.numel() for parameter in head_parameters)
        ),
        "attention_ffn_input_output_time_frozen": True,
        "explicit_frozen_parameter_count": int(
            sum(
                parameter.numel()
                for name, parameter in model.named_parameters()
                if name in explicitly_frozen
            )
        ),
    }
    return optimizer, receipt


def _restore_frozen_ema(
    ema: ModelEMA,
    foundation_state: Mapping[str, torch.Tensor],
    *,
    trainable_names: set[str],
) -> None:
    with torch.no_grad():
        for name, source in foundation_state.items():
            if name not in trainable_names:
                destination = ema.shadow[name]
                destination.copy_(
                    source.to(
                        device=destination.device, dtype=destination.dtype
                    )
                )


def _frozen_parameter_audit(
    state: Mapping[str, torch.Tensor],
    foundation_state: Mapping[str, torch.Tensor],
    *,
    trainable_names: set[str],
) -> dict:
    changed = []
    maximum_error = 0.0
    for name, original in foundation_state.items():
        if name in trainable_names:
            continue
        current = state[name].detach().cpu()
        original = original.detach().cpu()
        if not torch.equal(current, original):
            changed.append(name)
            maximum_error = max(
                maximum_error,
                float((current - original).abs().max()),
            )
    return {
        "passed": not changed,
        "changed_frozen_tensor_names": changed,
        "maximum_abs_error": maximum_error,
    }


def _lr_scale(
    step: int, *, total_steps: int, warmup_steps: int, minimum_ratio: float
) -> float:
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


def _manual_seed_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed) % (2**63 - 1))
    return generator


def _explicit_shared_flow_state(
    actions: torch.Tensor,
    frame_valid: torch.Tensor | None,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if frame_valid is None:
        noise = torch.randn(
            actions.shape,
            dtype=actions.dtype,
            device=actions.device,
            generator=generator,
        )
    else:
        noise = torch.zeros_like(actions)
        for row, count in enumerate(frame_valid.sum(dim=1).tolist()):
            noise[row, :count] = torch.randn(
                (count, actions.shape[-1]),
                dtype=actions.dtype,
                device=actions.device,
                generator=generator,
            )
    flow_times = torch.rand(
        actions.shape[0],
        dtype=actions.dtype,
        device=actions.device,
        generator=generator,
    )
    return noise, flow_times


def _style_smooth_l1(
    predicted: torch.Tensor,
    targets: torch.Tensor,
    supervision_mask: torch.Tensor,
) -> torch.Tensor:
    if supervision_mask.dtype != torch.bool or supervision_mask.shape != (
        predicted.shape[0],
    ):
        raise ValueError("style supervision mask must be boolean [batch]")
    return masked_style_control_smooth_l1(
        predicted,
        targets,
        supervision_mask=supervision_mask,
    )


def _condition_pair_losses(
    model: torch.nn.Module,
    actions: torch.Tensor,
    noise: torch.Tensor,
    flow_times: torch.Tensor,
    aligned_conditions: torch.Tensor,
    negative_conditions: torch.Tensor,
    dim_masks: torch.Tensor,
    frame_valid: torch.Tensor,
    conditioning_mask: torch.Tensor,
    *,
    response_floor: float,
    ranking_margin: float,
) -> dict[str, torch.Tensor]:
    """Compare every conditioned row at one exact shared flow state.

    Ranking is reduced per episode rather than after averaging the batch.  This
    prevents a few large correct-vs-negative gaps from hiding that most rows
    still prefer the wrong text condition.
    """
    observed = frame_valid[:, :, None] & dim_masks[:, None, :]
    masked_noise = noise * observed
    x_t = (
        (1.0 - flow_times[:, None, None]) * masked_noise
        + flow_times[:, None, None] * actions
    ) * observed
    aligned_prediction = forward_with_frame_mask(
        model, x_t, flow_times, aligned_conditions, frame_valid
    )
    negative_prediction = forward_with_frame_mask(
        model, x_t, flow_times, negative_conditions, frame_valid
    )
    zero_prediction = forward_with_frame_mask(
        model,
        x_t,
        flow_times,
        torch.zeros_like(aligned_conditions),
        frame_valid,
    )
    target = (actions - masked_noise) * observed
    ranking = aligned_vs_rolled_shuffled_hinge_loss(
        aligned_prediction,
        negative_prediction,
        target,
        observed,
        margin=float(ranking_margin),
        reduction="none",
    )
    row_weights = conditioning_mask.to(actions.dtype)
    denominator = row_weights.sum().clamp_min(1.0)
    ranking_loss = (
        ranking["hinge_per_example"] * row_weights
    ).sum() / denominator
    ranking_satisfied = (
        (
            ranking["rolled_shuffled_flow_mse_per_example"]
            - ranking["aligned_flow_mse_per_example"]
        )
        >= float(ranking_margin)
    ).to(actions.dtype)
    ranking_satisfied = (ranking_satisfied * row_weights).sum() / denominator
    conditioned_observed = observed & conditioning_mask[:, None, None]
    response_squared = (aligned_prediction - zero_prediction).square()
    response_count = conditioned_observed.sum().to(response_squared.dtype)
    response_rms = (
        (
            response_squared
            * conditioned_observed.to(response_squared.dtype)
        ).sum()
        / response_count.clamp_min(1.0)
        + 1e-12
    ).sqrt()
    reconstructed_aligned = (
        x_t
        + (1.0 - flow_times[:, None, None]) * aligned_prediction
    ) * observed
    return {
        "ranking_loss": ranking_loss,
        "ranking_gap": (
            (
                ranking["rolled_shuffled_flow_mse_per_example"]
                - ranking["aligned_flow_mse_per_example"]
            )
            * row_weights
        ).sum()
        / denominator,
        "ranking_satisfied_fraction": ranking_satisfied,
        "aligned_flow_mse": (
            ranking["aligned_flow_mse_per_example"] * row_weights
        ).sum()
        / denominator,
        "negative_flow_mse": (
            ranking["rolled_shuffled_flow_mse_per_example"] * row_weights
        ).sum()
        / denominator,
        "response_floor_loss": F.relu(
            float(response_floor) - response_rms
        ),
        "response_rms": response_rms,
        "reconstructed_aligned_normalized": reconstructed_aligned,
    }


def _batch_conditions(
    style_head: QwenStyleHead,
    rows: Sequence[Mapping[str, Any]],
    base_conditions: torch.Tensor,
    keep_mask: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        torch.any(base_conditions[:, : TEXT_LATENT_SLICE.start] != 0.0)
        or torch.any(base_conditions[:, STYLE_CONTROL_SLICE] != 0.0)
    ):
        raise RuntimeError("a target/oracle style reached the runtime collator")
    latents = base_conditions[:, TEXT_LATENT_SLICE]
    targets = torch.as_tensor(
        np.stack([row["continuous_style_training_target"] for row in rows]),
        dtype=torch.float32,
        device=device,
    )
    predicted = style_head(latents, keep_mask)
    conditions = assemble_text_style_conditions(
        latents, predicted, keep_mask
    )
    if (
        torch.any(conditions[:, : STYLE_CONTROL_SLICE.start] != 0.0)
        or torch.any(conditions[~keep_mask] != 0.0)
    ):
        raise RuntimeError("runtime text/style condition isolation failed")
    return conditions, predicted, targets


def _negative_conditions(
    style_head: QwenStyleHead,
    negative_rows: Sequence[Mapping[str, Any]],
    keep_mask: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    latents = torch.as_tensor(
        np.stack([row["qwen_text_latent_128d"] for row in negative_rows]),
        dtype=torch.float32,
        device=device,
    )
    predicted = style_head(latents, keep_mask)
    return assemble_text_style_conditions(latents, predicted, keep_mask)


def _semantic_perceptual_batch(
    semantic_perceptual: Beat2SemanticPerceptualLoss,
    reconstructed_normalized: torch.Tensor,
    base_conditions: torch.Tensor,
    frame_valid: torch.Tensor,
    durations: torch.Tensor,
    rows: Sequence[Mapping[str, Any]],
    keep_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Apply the frozen critic only where a text condition was retained."""

    if keep_mask.dtype != torch.bool or keep_mask.shape != (
        reconstructed_normalized.shape[0],
    ):
        raise ValueError("semantic perceptual keep_mask must be boolean [batch]")
    if len(rows) != reconstructed_normalized.shape[0]:
        raise ValueError("semantic perceptual rows do not match the batch")
    if not bool(torch.any(keep_mask).item()):
        zero = reconstructed_normalized.sum() * 0.0
        return {
            "total": zero,
            "cosine": zero.detach(),
            "contrastive": zero.detach(),
            "global_contrastive": zero.detach(),
            "cross_group_cosine_gap": zero.detach(),
            "hard_cross_group_margin": zero.detach(),
            "hard_cross_group_margin_positive_fraction": zero.detach(),
            "motion_to_text_group_recall_at_1": zero.detach(),
            "text_to_motion_group_recall_at_1": zero.detach(),
            "motion_encoder_group_accuracy": zero.detach(),
            "global_positive_cosine": zero.detach(),
            "global_hard_negative_cosine": zero.detach(),
            "global_hard_cross_group_margin": zero.detach(),
            "global_hard_cross_group_margin_positive_fraction": zero.detach(),
            "global_motion_to_prototype_recall_at_1": zero.detach(),
            "global_mean_positive_rank": zero.detach(),
            "global_prototype_count": zero.detach(),
        }
    groups = torch.as_tensor(
        [
            int(row["experimental_semantic_group_index"])
            for row in rows
        ],
        dtype=torch.long,
        device=reconstructed_normalized.device,
    )
    return semantic_perceptual(
        reconstructed_normalized[keep_mask],
        base_conditions[keep_mask, TEXT_LATENT_SLICE],
        frame_mask=frame_valid[keep_mask],
        durations_sec=durations[keep_mask],
        group_ids=groups[keep_mask],
    )


def _rms_difference(
    left: torch.Tensor, right: torch.Tensor, observed: torch.Tensor
) -> float:
    values = (left - right).square()
    return float(values[observed].mean().sqrt().detach().cpu())


@torch.no_grad()
def condition_diagnostics(
    model: torch.nn.Module,
    style_head: QwenStyleHead,
    semantic_perceptual: Beat2SemanticPerceptualLoss,
    episodes: Sequence[Mapping[str, Any]],
    negative_pool: CrossGroupNegativePool,
    *,
    action_stats: Mapping[str, Any],
    batching: Mapping[str, Any],
    device: torch.device,
    seed: int,
) -> dict:
    if not episodes:
        raise ValueError("condition diagnostics require episodes")
    was_model_training = model.training
    was_head_training = style_head.training
    model.eval()
    style_head.eval()
    try:
        bucket_frames = max(
            native_length_bucket(
                int(np.asarray(row["actions"]).shape[0]),
                batching["length_buckets"],
            )
            for row in episodes
        )
        actions, base_conditions, dim_masks, durations, frame_valid = (
            _batch_tensors_for_config(
                episodes,
                frame_count=bucket_frames,
                action_stats=action_stats,
                device=device,
                batching=batching,
            )
        )
        keep_mask = torch.ones(
            len(episodes), dtype=torch.bool, device=device
        )
        aligned, predicted_style, style_targets = _batch_conditions(
            style_head,
            episodes,
            base_conditions,
            keep_mask,
            device=device,
        )
        negative_rows = negative_pool.select(
            episodes, step=int(seed), microbatch_index=0
        )
        negative = _negative_conditions(
            style_head, negative_rows, keep_mask, device=device
        )
        zero = torch.zeros_like(aligned)
        generator = _manual_seed_generator(device, seed)
        noise, flow_times = _explicit_shared_flow_state(
            actions, frame_valid, generator=generator
        )
        diagnostic_loss_weights = {
            "flow": 1.0,
            "position": 0.0,
            "body": 0.0,
            "velocity": 0.0,
            "acceleration": 0.0,
        }
        aligned_losses = masked_18d_objective(
            model,
            actions,
            aligned,
            dim_masks,
            durations,
            loss_weights=diagnostic_loss_weights,
            frame_valid_mask=frame_valid,
            noise=noise,
            flow_times=flow_times,
        )
        negative_losses = masked_18d_objective(
            model,
            actions,
            negative,
            dim_masks,
            durations,
            loss_weights=diagnostic_loss_weights,
            frame_valid_mask=frame_valid,
            noise=noise,
            flow_times=flow_times,
        )
        observed = dim_masks[:, None, :]
        if frame_valid is not None:
            observed = observed & frame_valid[:, :, None]
        masked_noise = noise * observed
        x_t = (
            (1.0 - flow_times[:, None, None]) * masked_noise
            + flow_times[:, None, None] * actions
        ) * observed
        aligned_prediction = forward_with_frame_mask(
            model, x_t, flow_times, aligned, frame_valid
        )
        negative_prediction = forward_with_frame_mask(
            model, x_t, flow_times, negative, frame_valid
        )
        zero_prediction = forward_with_frame_mask(
            model, x_t, flow_times, zero, frame_valid
        )
        planned = model.plan_condition(aligned)["duration_sec"]
        style_loss = masked_style_control_smooth_l1(
            predicted_style, style_targets
        )
        target = (actions - noise) * observed
        aligned_per_example = masked_per_example_flow_mse(
            aligned_prediction, target, observed
        )
        negative_per_example = masked_per_example_flow_mse(
            negative_prediction, target, observed
        )
        correct_wins = (
            aligned_per_example + 0.005 < negative_per_example
        ).float()
        reconstructed_aligned = (
            x_t
            + (1.0 - flow_times[:, None, None]) * aligned_prediction
        ) * observed
        semantic = _semantic_perceptual_batch(
            semantic_perceptual,
            reconstructed_aligned,
            base_conditions,
            frame_valid,
            durations,
            episodes,
            keep_mask,
        )
        return {
            "episode_count": len(episodes),
            "aligned_flow_loss": float(
                aligned_losses["flow"].detach().cpu()
            ),
            "cross_group_flow_loss": float(
                negative_losses["flow"].detach().cpu()
            ),
            "cross_group_minus_aligned_flow_loss": float(
                (negative_losses["flow"] - aligned_losses["flow"])
                .detach()
                .cpu()
            ),
            "correct_flow_win_rate": float(
                correct_wins.mean().detach().cpu()
            ),
            "aligned_vs_zero_prediction_rms": _rms_difference(
                aligned_prediction, zero_prediction, observed
            ),
            "aligned_vs_cross_group_prediction_rms": _rms_difference(
                aligned_prediction, negative_prediction, observed
            ),
            "duration_mae_sec": float(
                (planned - durations).abs().mean().detach().cpu()
            ),
            "style_smooth_l1": float(style_loss.detach().cpu()),
            "style_prediction_rms": float(
                predicted_style.square().mean().sqrt().detach().cpu()
            ),
            "semantic_perceptual_total": float(
                semantic["total"].detach().cpu()
            ),
            "semantic_cosine_loss": float(
                semantic["cosine"].detach().cpu()
            ),
            "semantic_contrastive_loss": float(
                semantic["contrastive"].detach().cpu()
            ),
            "semantic_global_contrastive_loss": float(
                semantic["global_contrastive"].detach().cpu()
            ),
            "semantic_cross_group_cosine_gap": float(
                semantic["cross_group_cosine_gap"].detach().cpu()
            ),
            "semantic_hard_cross_group_margin": float(
                semantic["hard_cross_group_margin"].detach().cpu()
            ),
            "semantic_hard_margin_positive_fraction": float(
                semantic[
                    "hard_cross_group_margin_positive_fraction"
                ].detach().cpu()
            ),
            "semantic_motion_to_text_group_recall_at_1": float(
                semantic["motion_to_text_group_recall_at_1"]
                .detach()
                .cpu()
            ),
            "semantic_text_to_motion_group_recall_at_1": float(
                semantic["text_to_motion_group_recall_at_1"]
                .detach()
                .cpu()
            ),
            "semantic_motion_encoder_group_accuracy": float(
                semantic["motion_encoder_group_accuracy"]
                .detach()
                .cpu()
            ),
            "semantic_global_positive_cosine": float(
                semantic["global_positive_cosine"].detach().cpu()
            ),
            "semantic_global_hard_negative_cosine": float(
                semantic["global_hard_negative_cosine"].detach().cpu()
            ),
            "semantic_global_hard_cross_group_margin": float(
                semantic["global_hard_cross_group_margin"]
                .detach()
                .cpu()
            ),
            "semantic_global_hard_margin_positive_fraction": float(
                semantic[
                    "global_hard_cross_group_margin_positive_fraction"
                ].detach().cpu()
            ),
            "semantic_global_motion_to_prototype_recall_at_1": float(
                semantic["global_motion_to_prototype_recall_at_1"]
                .detach()
                .cpu()
            ),
            "semantic_global_mean_positive_rank": float(
                semantic["global_mean_positive_rank"].detach().cpu()
            ),
            "semantic_global_prototype_count": int(
                semantic["global_prototype_count"].detach().cpu()
            ),
            "shared_noise_and_flow_times": True,
            "negative_policy": NEGATIVE_POLICY,
        }
    finally:
        if was_model_training:
            model.train()
        if was_head_training:
            style_head.train()


def anti_collapse_decision(
    initial: Mapping[str, Any],
    final: Mapping[str, Any],
    gates: Mapping[str, float],
    *,
    enforce: bool,
) -> dict:
    initial_response = float(initial["aligned_vs_zero_prediction_rms"])
    final_response = float(final["aligned_vs_zero_prediction_rms"])
    retention = final_response / max(initial_response, 1e-12)
    checks = {
        "aligned_vs_zero_absolute": final_response
        >= float(gates["minimum_aligned_vs_zero_prediction_rms"]),
        "aligned_vs_zero_retention": retention
        >= float(gates["minimum_response_retention_ratio"]),
        "aligned_vs_cross_group_absolute": float(
            final["aligned_vs_cross_group_prediction_rms"]
        )
        >= float(gates["minimum_aligned_vs_cross_group_prediction_rms"]),
        "correct_beats_cross_group": float(
            final["cross_group_minus_aligned_flow_loss"]
        )
        >= float(gates["minimum_cross_group_minus_aligned_flow_loss"]),
        "correct_flow_win_rate": float(final["correct_flow_win_rate"])
        >= float(gates["minimum_correct_flow_win_rate"]),
        "duration_mae": float(final["duration_mae_sec"])
        <= float(gates["maximum_duration_mae_sec"]),
        "style_auxiliary_loss": float(final["style_smooth_l1"])
        <= float(gates["maximum_style_smooth_l1"]),
    }
    finite = all(
        math.isfinite(float(value))
        for value in (
            final_response,
            retention,
            final["aligned_vs_cross_group_prediction_rms"],
            final["cross_group_minus_aligned_flow_loss"],
            final["correct_flow_win_rate"],
            final["duration_mae_sec"],
            final["style_smooth_l1"],
            final["semantic_perceptual_total"],
            final["semantic_cross_group_cosine_gap"],
            final["semantic_hard_cross_group_margin"],
            final["semantic_hard_margin_positive_fraction"],
            final["semantic_motion_to_text_group_recall_at_1"],
            final["semantic_text_to_motion_group_recall_at_1"],
            final["semantic_global_contrastive_loss"],
            final["semantic_global_hard_cross_group_margin"],
            final[
                "semantic_global_hard_margin_positive_fraction"
            ],
            final[
                "semantic_global_motion_to_prototype_recall_at_1"
            ],
            final["semantic_global_mean_positive_rank"],
        )
    )
    checks["all_metrics_finite"] = finite
    passed = all(checks.values())
    return {
        "passed": bool(passed),
        "enforced": bool(enforce),
        "checks": checks,
        "response_retention_ratio": float(retention),
        "thresholds": deepcopy(dict(gates)),
        "failure_reasons": [
            name for name, value in checks.items() if not value
        ],
    }


def _paths(config: Mapping[str, Any]) -> dict[str, Path]:
    root = Path(config["output_dir"])
    return {
        "root": root,
        "state": root / "last_state_v2.pt",
        "checkpoint": root / "generator_text_style_emotion_v2.pt",
        "summary": root / "training_summary_v2.json",
        "progress": root / "progress_v2.jsonl",
    }


def _config_contract(config: Mapping[str, Any], *, smoke_test: bool) -> dict:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": CONFIG_ARTIFACT_KIND,
        "effective_config": deepcopy(dict(config)),
        "smoke_test": bool(smoke_test),
        "condition_policy": CONDITION_POLICY,
        "training_policy": TRAINING_POLICY,
    }
    payload["sha256"] = _mapping_sha256(payload)
    return payload


def _input_contract(
    config_contract: Mapping[str, Any],
    data_receipt: Mapping[str, Any],
    bridge_metadata: Mapping[str, Any],
) -> dict:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "beat2_text_style_emotion_input_contract_v2",
        "config_contract_sha256": config_contract["sha256"],
        "manifest_sha256": data_receipt["manifest"]["sha256"],
        "foundation_checkpoint_sha256": data_receipt["foundation"]["sha256"],
        "style_cache_sha256": data_receipt["style_cache"]["sha256"],
        "frozen_qwen_cache_sha256": data_receipt["frozen_qwen_cache"]["sha256"],
        "semantic_descriptor_cache_sha256": data_receipt[
            "semantic_perceptual"
        ]["descriptor_cache"]["sha256"],
        "semantic_motion_encoder_sha256": data_receipt[
            "semantic_perceptual"
        ]["motion_encoder_checkpoint"]["sha256"],
        "identity_arrays_sha256": data_receipt["identity_arrays_sha256"],
        "split_contract_sha256": data_receipt["split_contract_sha256"],
        "bridge_cache_sha256": bridge_metadata["cache_sha256"],
    }
    payload["sha256"] = _mapping_sha256(payload)
    return payload


def _save_state(
    path: Path,
    *,
    step: int,
    target_steps: int,
    input_contract: Mapping[str, Any],
    model: torch.nn.Module,
    style_head: QwenStyleHead,
    model_ema: ModelEMA,
    style_head_ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    sampler: TemperedGroupNativeBucketSampler,
    initial_diagnostics: Mapping[str, Any],
) -> None:
    numpy_state = np.random.get_state()
    _atomic_torch_save(
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": STATE_ARTIFACT_KIND,
            "experimental_only": True,
            "formal_release_eligible": False,
            "condition_policy": CONDITION_POLICY,
            "input_contract_sha256": input_contract["sha256"],
            "step": int(step),
            "target_steps": int(target_steps),
            "model_state_dict": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "style_head_state_dict": {
                name: value.detach().cpu()
                for name, value in style_head.state_dict().items()
            },
            "model_ema_state_dict": {
                name: value.detach().cpu()
                for name, value in model_ema.shadow.items()
            },
            "style_head_ema_state_dict": {
                name: value.detach().cpu()
                for name, value in style_head_ema.shadow.items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "sampler_state_dict": sampler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": {
                "bit_generator": str(numpy_state[0]),
                "keys": torch.from_numpy(
                    np.asarray(numpy_state[1], dtype=np.uint32).copy()
                ),
                "position": int(numpy_state[2]),
                "has_gaussian": int(numpy_state[3]),
                "cached_gaussian": float(numpy_state[4]),
            },
            "initial_diagnostics": deepcopy(dict(initial_diagnostics)),
        },
        path,
    )


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer, *, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                state[name] = value.to(device)


def _load_state(
    path: Path,
    *,
    input_contract: Mapping[str, Any],
    target_steps: int,
    model: torch.nn.Module,
    style_head: QwenStyleHead,
    model_ema: ModelEMA,
    style_head_ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    sampler: TemperedGroupNativeBucketSampler,
    device: torch.device,
) -> tuple[int, dict]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("artifact_kind") != STATE_ARTIFACT_KIND:
        raise ValueError(
            "resume checkpoint is not an isolated text style/emotion V2 state"
        )
    if (
        state.get("schema_version") != SCHEMA_VERSION
        or state.get("condition_policy") != CONDITION_POLICY
        or state.get("input_contract_sha256") != input_contract["sha256"]
        or int(state.get("target_steps", -1)) != int(target_steps)
    ):
        raise ValueError("V2 exact-resume contract changed")
    step = int(state.get("step", -1))
    if not 1 <= step <= int(target_steps):
        raise ValueError("V2 resume step is invalid")
    model.load_state_dict(state["model_state_dict"], strict=True)
    style_head.load_state_dict(state["style_head_state_dict"], strict=True)
    model_ema.shadow = {
        name: value.to(device)
        for name, value in state["model_ema_state_dict"].items()
    }
    style_head_ema.shadow = {
        name: value.to(device)
        for name, value in state["style_head_ema_state_dict"].items()
    }
    optimizer.load_state_dict(state["optimizer_state_dict"])
    _move_optimizer_state(optimizer, device=device)
    sampler.load_state_dict(state["sampler_state_dict"])
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state.get("cuda_rng_state_all"):
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    random.setstate(state["python_rng_state"])
    numpy_state = state["numpy_rng_state"]
    if not isinstance(numpy_state, Mapping):
        raise ValueError("V2 resume NumPy RNG state uses an unsafe legacy layout")
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            numpy_state["keys"].detach().cpu().numpy().astype(
                np.uint32, copy=True
            ),
            int(numpy_state["position"]),
            int(numpy_state["has_gaussian"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    return step, dict(state["initial_diagnostics"])


def _model_and_head(
    config: Mapping[str, Any],
    foundation_checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, QwenStyleHead, dict]:
    model, checkpoint = load_contract_checkpoint(
        foundation_checkpoint_path,
        expected_action_dim=ACTION_DIM,
        device=device,
    )
    validate_motion_only_checkpoint_isolation(checkpoint)
    if checkpoint.get("architecture") != ULA_MMDIT_V3_ADALN_ARCHITECTURE:
        raise ValueError("V2 text training requires the clean V3 AdaLN foundation")
    style_head = QwenStyleHead(**config["style_head"]).to(device)
    return model, style_head, checkpoint


def train(
    config: Mapping[str, Any],
    *,
    smoke_test: bool,
    overwrite: bool,
    resume: bool,
) -> dict:
    """Run or exactly resume the independent BEAT2-only V2 post-training."""
    paths = _paths(config)
    if paths["checkpoint"].is_file() and paths["summary"].is_file() and not overwrite:
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        checkpoint = torch.load(
            paths["checkpoint"], map_location="cpu", weights_only=True
        )
        if (
            summary.get("artifact_kind") != SUMMARY_ARTIFACT_KIND
            or checkpoint.get("artifact_kind") != CHECKPOINT_ARTIFACT_KIND
            or summary.get("checkpoint_sha256")
            != sha256_file(paths["checkpoint"])
        ):
            raise ValueError("existing output is not a valid completed V2 run")
        return summary
    if overwrite:
        for name in ("state", "checkpoint", "summary", "progress"):
            if paths[name].is_file():
                paths[name].unlink()
    paths["root"].mkdir(parents=True, exist_ok=True)
    if paths["progress"].is_file() and not paths["state"].is_file():
        raise ValueError("V2 progress exists without an exact-resume state")

    splits, split_contract, bridge_arrays, preparation = _prepare_rows(
        config, smoke_test=smoke_test
    )
    semantic_receipt = _semantic_perceptual_receipt(config)
    if (
        semantic_receipt["global_prototype_source_qwen_cache"]["sha256"]
        != preparation["data_receipt"]["frozen_qwen_cache"]["sha256"]
    ):
        raise ValueError(
            "global prototype source differs from the frozen Qwen cache"
        )
    preparation["data_receipt"]["semantic_perceptual"] = deepcopy(
        semantic_receipt
    )
    bridge_metadata = _write_or_validate_bridge(
        config["output_dir"],
        bridge_arrays,
        preparation["data_receipt"],
    )
    config_contract = _config_contract(config, smoke_test=smoke_test)
    input_contract = _input_contract(
        config_contract, preparation["data_receipt"], bridge_metadata
    )
    training = config["training"]
    validation_subset = _stable_subset(
        splits["validation"],
        count=int(training["evaluation_episode_count"]),
        seed=int(config["seed"]),
        label="v2_validation",
    )
    diagnostic_candidates = _stable_subset(
        splits["validation"],
        count=len(splits["validation"]),
        seed=int(config["seed"]),
        label="v2_balanced_group_diagnostic",
    )
    diagnostic_groups = {}
    for row in diagnostic_candidates:
        diagnostic_groups.setdefault(
            int(row["experimental_semantic_group_index"]), row
        )
    diagnostic_subset = list(diagnostic_groups.values())
    if len(diagnostic_subset) < 2:
        diagnostic_subset = validation_subset
    diagnostic_subset = diagnostic_subset[
        : min(
            int(training["evaluation_episode_count"]),
            len(diagnostic_subset),
        )
    ]

    _seed_everything(int(config["seed"]))
    device = _resolve_device(config["device"])
    model, style_head, foundation_checkpoint = _model_and_head(
        config,
        preparation["data_receipt"]["foundation"]["path"],
        device=device,
    )
    semantic_perceptual, loaded_semantic_receipt = _build_semantic_perceptual(
        config,
        action_stats=foundation_checkpoint["action_stats"],
        device=device,
    )
    if loaded_semantic_receipt != semantic_receipt:
        raise RuntimeError("semantic perceptual artifact receipt changed")
    foundation_state = {
        name: value.detach().cpu().clone()
        for name, value in foundation_checkpoint["model_state_dict"].items()
    }
    if _state_dict_sha256(model.state_dict()) != _state_dict_sha256(
        foundation_state
    ):
        raise ValueError("V2 model did not start at the exact clean foundation")
    optimizer, optimizer_receipt = configure_condition_optimizer(
        model, style_head, config
    )
    trainable_names = set(optimizer_receipt["model_trainable_tensor_names"])
    optimized_parameters = [
        parameter
        for parameter in list(model.parameters()) + list(style_head.parameters())
        if parameter.requires_grad
    ]
    model_ema = ModelEMA(model, float(training["ema_decay"]))
    style_head_ema = ModelEMA(style_head, float(training["ema_decay"]))
    sampler = TemperedGroupNativeBucketSampler(
        splits["train"],
        buckets=training["batching"]["length_buckets"],
        seed=int(config["seed"]) + 17,
    )
    negative_pool = CrossGroupNegativePool(
        splits["train"], seed=int(config["seed"]) + 37
    )

    global_step = 0
    initial_diagnostics = None
    if paths["state"].is_file():
        if not resume:
            raise FileExistsError(
                "V2 state exists; pass --resume or --overwrite explicitly"
            )
        global_step, initial_diagnostics = _load_state(
            paths["state"],
            input_contract=input_contract,
            target_steps=int(training["steps"]),
            model=model,
            style_head=style_head,
            model_ema=model_ema,
            style_head_ema=style_head_ema,
            optimizer=optimizer,
            sampler=sampler,
            device=device,
        )
        _restore_frozen_ema(
            model_ema,
            foundation_state,
            trainable_names=trainable_names,
        )
    else:
        paths["progress"].write_text("", encoding="utf-8")
        initial_diagnostics = condition_diagnostics(
            model,
            style_head,
            semantic_perceptual,
            diagnostic_subset,
            negative_pool,
            action_stats=foundation_checkpoint["action_stats"],
            batching=training["batching"],
            device=device,
            seed=int(config["seed"]) + 1_000_003,
        )
        _append_jsonl(
            {
                "event": "foundation_step0",
                "step": 0,
                "exact_clean_foundation": True,
                "diagnostics": initial_diagnostics,
                "condition_policy": CONDITION_POLICY,
            },
            paths["progress"],
        )

    target_steps = int(training["steps"])
    if global_step > target_steps:
        raise ValueError("V2 resume step exceeds target")
    started = time.monotonic()
    last_train: dict[str, float] = {}
    last_diagnostics = dict(initial_diagnostics)
    last_grad_norm = 0.0
    model.train()
    style_head.train()
    for step in range(global_step + 1, target_steps + 1):
        scale = _lr_scale(
            step,
            total_steps=target_steps,
            warmup_steps=int(training["warmup_steps"]),
            minimum_ratio=float(training["minimum_lr_ratio"]),
        )
        optimizer.param_groups[0]["lr"] = float(training["lr"]) * scale
        optimizer.param_groups[1]["lr"] = float(training["style_head_lr"]) * scale
        semantic_weight = _semantic_weight_at_step(
            step,
            target_weight=float(
                config["semantic_perceptual"]["outer_weight"]
            ),
            warmup_steps=int(
                config["semantic_perceptual"]["warmup_steps"]
            ),
        )
        optimizer.zero_grad(set_to_none=True)
        remaining = int(training["batching"]["target_effective_batch_size"])
        accumulated = defaultdict(float)
        sampled_clip_ids = []
        negative_group_pairs = []
        microbatch_plans = []
        microbatch_index = 0
        while remaining > 0:
            batch, plan = sampler.sample_microbatch(
                remaining_effective_batch=remaining,
                semantic_tokens=int(model.semantic_tokens),
                max_batch_size=int(training["batch_size"]),
                batching=training["batching"],
            )
            actions, base_conditions, dim_masks, durations, frame_valid = (
                _batch_tensors_for_config(
                    batch,
                    frame_count=int(plan["bucket_frames"]),
                    action_stats=foundation_checkpoint["action_stats"],
                    device=device,
                    batching=training["batching"],
                )
            )
            dropout_generator = _manual_seed_generator(
                device,
                int(config["seed"])
                + step * 1_000_003
                + microbatch_index * 1009
                + 11,
            )
            keep_mask = sample_condition_keep_mask(
                len(batch),
                float(training["condition_dropout_probability"]),
                device=device,
                generator=dropout_generator,
            )
            aligned, predicted_style, style_targets = _batch_conditions(
                style_head,
                batch,
                base_conditions,
                keep_mask,
                device=device,
            )
            negative_rows = negative_pool.select(
                batch, step=step, microbatch_index=microbatch_index
            )
            negative = _negative_conditions(
                style_head, negative_rows, keep_mask, device=device
            )
            flow_generator = _manual_seed_generator(
                device,
                int(config["seed"])
                + step * 2_000_003
                + microbatch_index * 2017
                + 23,
            )
            noise, flow_times = _explicit_shared_flow_state(
                actions, frame_valid, generator=flow_generator
            )
            aligned_losses = masked_18d_objective(
                model,
                actions,
                aligned,
                dim_masks,
                durations,
                loss_weights=training["loss"],
                frame_valid_mask=frame_valid,
                noise=noise,
                flow_times=flow_times,
            )
            condition_pair = _condition_pair_losses(
                model,
                actions,
                noise,
                flow_times,
                aligned,
                negative,
                dim_masks,
                frame_valid,
                keep_mask,
                response_floor=float(
                    training["condition_response_floor"]
                ),
                ranking_margin=float(
                    training["condition_ranking_margin"]
                ),
            )
            style_loss = _style_smooth_l1(
                predicted_style, style_targets, keep_mask
            )
            ranking_loss = condition_pair["ranking_loss"]
            response_floor_loss = condition_pair["response_floor_loss"]
            response_rms = condition_pair["response_rms"]
            semantic = _semantic_perceptual_batch(
                semantic_perceptual,
                condition_pair["reconstructed_aligned_normalized"],
                base_conditions,
                frame_valid,
                durations,
                batch,
                keep_mask,
            )
            total = (
                aligned_losses["total"]
                + float(training["style_smooth_l1_weight"]) * style_loss
                + float(training["condition_ranking_weight"]) * ranking_loss
                + float(training["condition_response_floor_weight"])
                * response_floor_loss
                + semantic_weight * semantic["total"]
            )
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite V2 loss at step {step}")
            sample_weight = len(batch) / float(
                training["batching"]["target_effective_batch_size"]
            )
            (total * sample_weight).backward()
            accumulated["objective_total"] += (
                float(total.detach().cpu()) * sample_weight
            )
            for name, value in aligned_losses.items():
                accumulated[name] += (
                    float(value.detach().cpu()) * sample_weight
                )
            accumulated["style_smooth_l1"] += (
                float(style_loss.detach().cpu()) * sample_weight
            )
            accumulated["condition_ranking"] += (
                float(ranking_loss.detach().cpu()) * sample_weight
            )
            accumulated["condition_response_floor"] += (
                float(response_floor_loss.detach().cpu()) * sample_weight
            )
            accumulated["condition_response_rms"] += (
                float(response_rms.detach().cpu()) * sample_weight
            )
            accumulated["cross_group_minus_aligned_flow"] += (
                float(condition_pair["ranking_gap"].detach().cpu())
                * sample_weight
            )
            accumulated["condition_ranking_satisfied_fraction"] += (
                float(
                    condition_pair["ranking_satisfied_fraction"]
                    .detach()
                    .cpu()
                )
                * sample_weight
            )
            accumulated["text_keep_fraction"] += (
                float(keep_mask.float().mean().detach().cpu()) * sample_weight
            )
            accumulated["semantic_weight"] += semantic_weight * sample_weight
            for metric_name in (
                "total",
                "cosine",
                "contrastive",
                "global_contrastive",
                "cross_group_cosine_gap",
                "hard_cross_group_margin",
                "hard_cross_group_margin_positive_fraction",
                "motion_to_text_group_recall_at_1",
                "text_to_motion_group_recall_at_1",
                "motion_encoder_group_accuracy",
                "global_positive_cosine",
                "global_hard_negative_cosine",
                "global_hard_cross_group_margin",
                "global_hard_cross_group_margin_positive_fraction",
                "global_motion_to_prototype_recall_at_1",
                "global_mean_positive_rank",
                "global_prototype_count",
            ):
                accumulated[f"semantic_{metric_name}"] += (
                    float(semantic[metric_name].detach().cpu())
                    * sample_weight
                )
            sampled_clip_ids.extend(str(row["clip_id"]) for row in batch)
            negative_group_pairs.extend(
                [
                    [
                        int(row["experimental_semantic_group_index"]),
                        int(negative_row["experimental_semantic_group_index"]),
                    ]
                    for row, negative_row in zip(batch, negative_rows)
                ]
            )
            microbatch_plans.append(dict(plan))
            remaining -= len(batch)
            microbatch_index += 1
        last_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                optimized_parameters, float(training["max_grad_norm"])
            )
        )
        optimizer.step()
        model_ema.update(model)
        style_head_ema.update(style_head)
        _restore_frozen_ema(
            model_ema,
            foundation_state,
            trainable_names=trainable_names,
        )
        global_step = step
        last_train = dict(accumulated)
        should_validate = (
            step == 1
            or step % int(training["validation_interval"]) == 0
            or step == target_steps
        )
        should_checkpoint = (
            step == 1
            or step % int(training["checkpoint_interval"]) == 0
            or step == target_steps
        )
        event = {
            "event": "train_step",
            "step": step,
            "model_lr": optimizer.param_groups[0]["lr"],
            "style_head_lr": optimizer.param_groups[1]["lr"],
            "grad_norm": last_grad_norm,
            "train": last_train,
            "sampled_clip_ids_sha256": hashlib.sha256(
                _canonical_json(sampled_clip_ids)
            ).hexdigest(),
            "cross_group_pairs_sha256": hashlib.sha256(
                _canonical_json(negative_group_pairs)
            ).hexdigest(),
            "microbatch_plans": microbatch_plans,
        }
        if should_validate:
            with model_ema.apply(model), style_head_ema.apply(style_head):
                last_diagnostics = condition_diagnostics(
                    model,
                    style_head,
                    semantic_perceptual,
                    diagnostic_subset,
                    negative_pool,
                    action_stats=foundation_checkpoint["action_stats"],
                    batching=training["batching"],
                    device=device,
                    seed=int(config["seed"]) + 1_000_003,
                )
            event["diagnostics"] = last_diagnostics
        if (
            step == 1
            or step % int(training["log_interval"]) == 0
            or should_validate
        ):
            print(json.dumps(event, sort_keys=True), flush=True)
            _append_jsonl(event, paths["progress"])
        if should_checkpoint:
            _save_state(
                paths["state"],
                step=step,
                target_steps=target_steps,
                input_contract=input_contract,
                model=model,
                style_head=style_head,
                model_ema=model_ema,
                style_head_ema=style_head_ema,
                optimizer=optimizer,
                sampler=sampler,
                initial_diagnostics=initial_diagnostics,
            )

    final_model_state = {
        name: value.detach().cpu().clone()
        for name, value in model_ema.shadow.items()
    }
    final_style_head_state = {
        name: value.detach().cpu().clone()
        for name, value in style_head_ema.shadow.items()
    }
    model.load_state_dict(final_model_state, strict=True)
    style_head.load_state_dict(final_style_head_state, strict=True)
    final_diagnostics = condition_diagnostics(
        model,
        style_head,
        semantic_perceptual,
        diagnostic_subset,
        negative_pool,
        action_stats=foundation_checkpoint["action_stats"],
        batching=training["batching"],
        device=device,
        seed=int(config["seed"]) + 1_000_003,
    )
    gate = anti_collapse_decision(
        initial_diagnostics,
        final_diagnostics,
        config["anti_collapse_gates"],
        enforce=not smoke_test,
    )
    frozen_audit = _frozen_parameter_audit(
        final_model_state,
        foundation_state,
        trainable_names=trainable_names,
    )
    if not frozen_audit["passed"]:
        raise RuntimeError("V2 training changed the frozen motion prior")

    checkpoint_payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": CHECKPOINT_ARTIFACT_KIND,
        "experimental_only": True,
        "formal_release_eligible": False,
        "semantic_scope": SEMANTIC_SCOPE_ACKNOWLEDGEMENT,
        "semantic_supervision_status": (
            "official_metadata_experimental_not_formal_robot_affect_truth"
        ),
        "condition_policy": CONDITION_POLICY,
        "architecture": foundation_checkpoint["architecture"],
        "action_dim": ACTION_DIM,
        "condition_dim": CONDITION_DIM,
        "joint_order": deepcopy(foundation_checkpoint["joint_order"]),
        "config": deepcopy(foundation_checkpoint.get("config") or {}),
        "model_state_dict": final_model_state,
        "qwen_style_head_config": style_head.architecture_config(),
        "qwen_style_head_state_dict": final_style_head_state,
        "action_stats": deepcopy(foundation_checkpoint["action_stats"]),
        "global_step": int(foundation_checkpoint.get("global_step", 0))
        + target_steps,
        "foundation_receipt": deepcopy(
            preparation["data_receipt"]["foundation"]
        ),
        "data_receipt": deepcopy(preparation["data_receipt"]),
        "bridge_receipt": deepcopy(bridge_metadata),
        "input_contract": deepcopy(input_contract),
        "training_contract": {
            **optimizer_receipt,
            "steps": target_steps,
            "seed": int(config["seed"]),
            "loss": deepcopy(training["loss"]),
            "style_smooth_l1_weight": float(
                training["style_smooth_l1_weight"]
            ),
            "condition_ranking_weight": float(
                training["condition_ranking_weight"]
            ),
            "condition_ranking_margin": float(
                training["condition_ranking_margin"]
            ),
            "condition_response_floor": float(
                training["condition_response_floor"]
            ),
            "condition_response_floor_weight": float(
                training["condition_response_floor_weight"]
            ),
            "condition_dropout_probability": float(
                training["condition_dropout_probability"]
            ),
            "text_dropout_policy": TEXT_DROPOUT_POLICY,
            "negative_policy": NEGATIVE_POLICY,
            "shared_noise_and_flow_times": True,
            "semantic_perceptual": deepcopy(semantic_receipt),
            "batching": deepcopy(training["batching"]),
            "sampler": deepcopy(training["sampler"]),
            "external_motion_checkpoint_count": 0,
        },
        "split_contract": {
            "sha256": split_contract["sha256"],
            "counts": {name: len(rows) for name, rows in splits.items()},
        },
        "metrics": {
            "initial_condition_diagnostics": initial_diagnostics,
            "final_condition_diagnostics": final_diagnostics,
        },
        "anti_collapse_gate": gate,
        "frozen_parameter_audit": frozen_audit,
        "no_external_data": True,
        "no_kimodo": True,
    }
    if gate["passed"] or smoke_test:
        _atomic_torch_save(checkpoint_payload, paths["checkpoint"])
        checkpoint_sha256 = sha256_file(paths["checkpoint"])
    else:
        checkpoint_sha256 = None
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": SUMMARY_ARTIFACT_KIND,
        "experimental_only": True,
        "formal_release_eligible": False,
        "semantic_scope": SEMANTIC_SCOPE_ACKNOWLEDGEMENT,
        "condition_policy": CONDITION_POLICY,
        "completed_steps": target_steps,
        "target_steps": target_steps,
        "checkpoint": (
            str(paths["checkpoint"].resolve()) if checkpoint_sha256 else None
        ),
        "checkpoint_sha256": checkpoint_sha256,
        "resumable_state": str(paths["state"].resolve()),
        "input_contract_sha256": input_contract["sha256"],
        "initial_condition_diagnostics": initial_diagnostics,
        "final_condition_diagnostics": final_diagnostics,
        "anti_collapse_gate": gate,
        "frozen_parameter_audit": frozen_audit,
        "semantic_perceptual": deepcopy(semantic_receipt),
        "last_train": last_train,
        "last_grad_norm": last_grad_norm,
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "status": (
            "smoke_complete"
            if smoke_test
            else (
                "experimental_candidate"
                if gate["passed"]
                else "rejected_condition_collapse_or_quality_gate"
            )
        ),
        "no_external_data": True,
        "no_kimodo": True,
    }
    _atomic_json_save(summary, paths["summary"])
    if not smoke_test and not gate["passed"]:
        raise RuntimeError(
            "V2 training completed but failed hard anti-collapse gates: "
            + ", ".join(gate["failure_reasons"])
        )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            PROJECT_ROOT / "configs" / "beat2_text_style_emotion_v2.json"
        ),
    )
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = effective_config(
        read_config(args.config),
        smoke_test=bool(args.smoke_test),
        device=args.device,
        output_dir=args.output_dir,
    )
    if args.prepare_only:
        _, _, arrays, preparation = _prepare_rows(
            config, smoke_test=bool(args.smoke_test)
        )
        metadata = _write_or_validate_bridge(
            config["output_dir"], arrays, preparation["data_receipt"]
        )
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return 0
    result = train(
        config,
        smoke_test=bool(args.smoke_test),
        overwrite=bool(args.overwrite),
        resume=bool(args.resume),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
