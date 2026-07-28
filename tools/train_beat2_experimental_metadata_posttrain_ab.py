#!/usr/bin/env python3
"""Paired BEAT2 experimental-metadata generator post-training.

This entry point deliberately does not extend or relax the formal motion-only
trainer.  It verifies a completed clean BEAT2 foundation and its immutable
style-only cache, combines that style cache with either of two paired 128D text
latents, and emits explicitly experimental generator artifacts.

The two branches share one preparation contract.  Their only permitted input
difference is the frozen-Qwen versus BEAT2-LoRA 128D condition-cache hash.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.train_beat2_qwen_motion_alignment import (  # noqa: E402
    EXPECTED_QWEN_MODEL,
    NO_EXTERNAL_DATA_POLICY,
    SEMANTIC_SCOPE_ACKNOWLEDGEMENT,
    validate_clean_generator_foundation,
)
from upper_body_skeleton.ula_training import (  # noqa: E402
    KIMODO_CONDITION_DIM,
    KIMODO_MOTION_LATENT_DIM,
    KIMODO_V2_CONDITION_DIM,
    ULA_MMDIT_V3_ADALN_ARCHITECTURE,
)
from upper_body_skeleton.ula_v2_18d_head import (  # noqa: E402
    ACTION_DIM,
    MOTION_ONLY_CONDITION_CACHE_ARTIFACT_KIND,
    MOTION_ONLY_EPISODE_CONTRACT,
    STYLE_CONTROL_SLICE,
    load_condition_cache,
    load_contract_checkpoint,
    sha256_file,
    validate_condition_cache_for_generator,
    validate_motion_only_checkpoint_isolation,
)
from upper_body_skeleton.ula_v2_18d_posttrain import (  # noqa: E402
    ModelEMA,
    NativeLengthBucketSampler,
    _batch_tensors_for_config,
    _sampler_for_config,
    evaluate_posttrain,
    load_attached_beat_episodes,
    masked_18d_objective,
    strict_group_split,
)
from upper_body_skeleton.ula_v2_18d_random_init import (  # noqa: E402
    forward_with_frame_mask,
)


SCHEMA_VERSION = 1
CONFIG_ARTIFACT_KIND = "beat2_experimental_metadata_posttrain_ab_config_v1"
PAIR_ARTIFACT_KIND = "beat2_experimental_metadata_posttrain_pair_v1"
BRIDGE_CACHE_ARTIFACT_KIND = (
    "beat2_experimental_metadata_264d_condition_cache_v1"
)
CHECKPOINT_ARTIFACT_KIND = (
    "beat2_experimental_metadata_conditioned_posttrain_v1"
)
SUMMARY_ARTIFACT_KIND = (
    "beat2_experimental_metadata_conditioned_posttrain_training_summary_v1"
)
STATE_ARTIFACT_KIND = (
    "beat2_experimental_metadata_conditioned_posttrain_state_v1"
)
COMPARISON_ARTIFACT_KIND = (
    "beat2_experimental_metadata_conditioned_posttrain_comparison_v1"
)
EXPERIMENTAL_CONDITION_POLICY = (
    "beat2_metadata_text_latent_136_264_plus_trajectory_style_133_136_v1"
)
TRAINING_POLICY = "zero_latent_preserving_condition_path_only_v1"
EXPECTED_QWEN_REVISION = (
    "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
)
EXPECTED_SPLIT_COUNTS = {
    "train": 7_522,
    "validation": 1_629,
    "test": 2_988,
}
VARIANTS = ("frozen_base", "lora_finetuned")
IDENTITY_ARRAY_NAMES = (
    "clip_ids",
    "task_ids",
    "prompts",
    "fixed_split_assignments",
    "speaker_keys",
    "semantic_group_indices",
    "trajectory_sha256",
)
FORBIDDEN_PATH_TOKEN = "kimodo"
MOTION_LATENT_WEIGHT_NAME = "motion_latent_condition.0.weight"
PLAN_WEIGHT_NAME = "plan.0.weight"


DEFAULT_CONFIG = {
    "schema_version": SCHEMA_VERSION,
    "artifact_kind": CONFIG_ARTIFACT_KIND,
    "semantic_scope_acknowledgement": SEMANTIC_SCOPE_ACKNOWLEDGEMENT,
    "data_policy": NO_EXTERNAL_DATA_POLICY,
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
    "lora_condition_cache": (
        "/home/gez/shuaiwang/ula-motion-generate/training/runs/"
        "beat2_qwen_motion_alignment_ab_v1/"
        "conditions_128d_lora_finetuned.npz"
    ),
    "output_dir": (
        "/home/gez/shuaiwang/ula-motion-generate/training/runs/"
        "beat2_experimental_metadata_posttrain_ab_v1"
    ),
    "dataset_source": "beat2_official_semantic_event_training_pool_v7",
    "speaker_namespace": "beat2",
    "source_group_namespace": "beat2-official-semantic-event",
    "seed": 20260726,
    "device": "cuda",
    "training_policy": TRAINING_POLICY,
    "training": {
        "steps": 50_000,
        "batch_size": 16,
        "lr": 5e-5,
        "minimum_lr_ratio": 0.1,
        "warmup_steps": 1_000,
        "weight_decay": 1e-4,
        "adam_eps": 1e-6,
        "max_grad_norm": 1.0,
        "ema_decay": 0.9995,
        "validation_interval": 5_000,
        "checkpoint_interval": 1_000,
        "log_interval": 50,
        "evaluation_episode_count": 128,
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
            "mode": "source_speaker_activity",
            "activity_bin_edges_rad_s": [0.03, 0.10, 0.25],
        },
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


def _self_hashed(value: Mapping[str, Any]) -> dict:
    result = deepcopy(dict(value))
    result.pop("sha256", None)
    result["sha256"] = _mapping_sha256(result)
    return result


def _validate_self_hash(value: Mapping[str, Any], *, context: str) -> None:
    expected = _self_hashed(value)["sha256"]
    if value.get("sha256") != expected:
        raise ValueError(f"{context} self hash is invalid")


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
            json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n"
        )


def _deep_merge(defaults: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    unknown = sorted(set(override) - set(defaults))
    if unknown:
        raise ValueError(f"unknown experimental posttrain config keys: {unknown}")
    result = deepcopy(dict(defaults))
    for key, value in override.items():
        if isinstance(result.get(key), Mapping):
            if not isinstance(value, Mapping):
                raise ValueError(f"config section {key!r} must be a mapping")
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _reject_forbidden_path(value: str | Path, *, field: str) -> Path:
    path = Path(value).expanduser().resolve()
    if FORBIDDEN_PATH_TOKEN in str(path).lower():
        raise ValueError(f"{field} contains a forbidden external-data token")
    return path


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def _finite_nonnegative(value: Any, *, field: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


def validate_config(config: Mapping[str, Any]) -> dict:
    values = _deep_merge(DEFAULT_CONFIG, config)
    exact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": CONFIG_ARTIFACT_KIND,
        "semantic_scope_acknowledgement": SEMANTIC_SCOPE_ACKNOWLEDGEMENT,
        "data_policy": NO_EXTERNAL_DATA_POLICY,
        "training_policy": TRAINING_POLICY,
    }
    for field, expected in exact.items():
        if values.get(field) != expected:
            raise ValueError(f"{field} must be {expected!r}")
    expected_manifest = str(values["expected_manifest_sha256"])
    if (
        len(expected_manifest) != 64
        or any(character not in "0123456789abcdef" for character in expected_manifest)
    ):
        raise ValueError("expected_manifest_sha256 must be a lowercase SHA256")
    for field in (
        "manifest_path",
        "foundation_checkpoint",
        "foundation_training_summary",
        "style_condition_cache",
        "frozen_condition_cache",
        "lora_condition_cache",
        "output_dir",
    ):
        values[field] = str(_reject_forbidden_path(values[field], field=field))
    for field in ("dataset_source", "speaker_namespace", "source_group_namespace"):
        if not str(values.get(field) or "").strip():
            raise ValueError(f"{field} is required")
        values[field] = str(values[field])
    values["seed"] = int(values["seed"])
    values["device"] = str(values["device"])

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
        training[field] = _positive_int(training[field], field=f"training.{field}")
    if training["warmup_steps"] > training["steps"]:
        raise ValueError("training.warmup_steps cannot exceed training.steps")
    for field in (
        "lr",
        "minimum_lr_ratio",
        "weight_decay",
        "adam_eps",
        "max_grad_norm",
        "ema_decay",
    ):
        training[field] = _finite_nonnegative(
            training[field], field=f"training.{field}"
        )
    if not 0 < training["lr"] <= 1e-3:
        raise ValueError("training.lr must be in (0, 1e-3]")
    if not 0 < training["minimum_lr_ratio"] <= 1:
        raise ValueError("training.minimum_lr_ratio must be in (0, 1]")
    if not 0 < training["ema_decay"] < 1:
        raise ValueError("training.ema_decay must be in (0, 1)")
    expected_loss_names = set(DEFAULT_CONFIG["training"]["loss"])
    if set(training["loss"]) != expected_loss_names:
        raise ValueError("training.loss fields changed")
    for field in sorted(training["loss"]):
        training["loss"][field] = _finite_nonnegative(
            training["loss"][field], field=f"training.loss.{field}"
        )
    if training["loss"]["body"] != 0 or training["loss"]["planner_transition"] != 0:
        raise ValueError(
            "experimental condition-path posttrain requires body and "
            "planner_transition losses to remain zero"
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
        {_positive_int(value, field="training.batching.length_buckets") for value in batching["length_buckets"]}
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
    if sampler.get("mode") != "source_speaker_activity":
        raise ValueError(
            "training.sampler.mode must be 'source_speaker_activity'"
        )
    edges = [float(value) for value in sampler["activity_bin_edges_rad_s"]]
    if (
        len(edges) != 3
        or edges != sorted(set(edges))
        or any(not math.isfinite(value) or value < 0 for value in edges)
    ):
        raise ValueError("training.sampler.activity_bin_edges_rad_s is invalid")
    sampler["activity_bin_edges_rad_s"] = edges
    return values


def read_config(path: str | Path) -> dict:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("experimental posttrain config must be a JSON object")
    return validate_config(value)


def effective_config(
    config: Mapping[str, Any],
    *,
    smoke_test: bool,
    device: str | None = None,
    output_dir: str | Path | None = None,
    foundation_checkpoint: str | Path | None = None,
    foundation_training_summary: str | Path | None = None,
    style_condition_cache: str | Path | None = None,
    frozen_condition_cache: str | Path | None = None,
    lora_condition_cache: str | Path | None = None,
) -> dict:
    values = deepcopy(dict(config))
    overrides = {
        "device": device,
        "output_dir": output_dir,
        "foundation_checkpoint": foundation_checkpoint,
        "foundation_training_summary": foundation_training_summary,
        "style_condition_cache": style_condition_cache,
        "frozen_condition_cache": frozen_condition_cache,
        "lora_condition_cache": lora_condition_cache,
    }
    for field, value in overrides.items():
        if value is not None:
            values[field] = str(value)
    if smoke_test:
        training = values["training"]
        training.update(
            {
                "steps": 1,
                "batch_size": 2,
                "warmup_steps": 1,
                "validation_interval": 1,
                "checkpoint_interval": 1,
                "log_interval": 1,
                "evaluation_episode_count": 2,
            }
        )
        training["batching"]["target_effective_batch_size"] = 2
    return validate_config(values)


def _load_json_mapping(path: str | Path, *, context: str) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _foundation_receipt(
    config: Mapping[str, Any], *, smoke_test: bool
) -> tuple[dict, dict]:
    path = Path(config["foundation_checkpoint"])
    receipt = validate_clean_generator_foundation(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    validate_motion_only_checkpoint_isolation(checkpoint)
    if checkpoint.get("architecture") != ULA_MMDIT_V3_ADALN_ARCHITECTURE:
        raise ValueError(
            "experimental metadata bridge requires the clean V3 AdaLN foundation"
        )
    contracts = checkpoint["v2_contracts"]
    split_contract = contracts["split"]
    style_contract = contracts["style"]
    receipt.update(
        {
            "split_contract_sha256": split_contract["sha256"],
            "style_contract_sha256": style_contract["sha256"],
            "posttrain_step": int(checkpoint.get("posttrain_step", 0)),
            "formal_release_eligible": bool(
                checkpoint.get("formal_release_eligible", False)
            ),
            "random_initialization": deepcopy(
                checkpoint["random_initialization"]
            ),
            "sources": deepcopy(checkpoint["sources"]),
        }
    )
    summary_path = Path(config["foundation_training_summary"])
    if smoke_test and not summary_path.is_file():
        receipt["completion_validation"] = "smoke_random_foundation_allowed"
        receipt["training_summary"] = None
        return checkpoint, receipt
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"clean foundation completion summary is missing: {summary_path}"
        )
    summary = _load_json_mapping(summary_path, context="foundation training summary")
    summary_checkpoint = Path(str(summary.get("checkpoint") or "")).resolve()
    completed_steps = int(summary.get("completed_steps", -1))
    target_steps = int(summary.get("target_steps", -1))
    stopped_early = summary.get("stopped_early")
    completed = completed_steps == target_steps or (
        stopped_early is True and 0 < completed_steps <= target_steps
    )
    if (
        summary_checkpoint != path.resolve()
        or not completed
        or summary.get("formal_release_eligible") is not True
        or summary.get("artifact_status") != "adjudicated_posttrain_candidate"
        or checkpoint.get("formal_release_eligible") is not True
        or checkpoint.get("artifact_status") != "adjudicated_posttrain_candidate"
        or int(checkpoint.get("best_step", -1)) != int(summary.get("best_step", -2))
        or (checkpoint.get("posttrain_data_contract") or {}).get("sha256")
        != summary.get("data_contract_sha256")
        or split_contract.get("sha256") != summary.get("split_contract_sha256")
    ):
        raise ValueError(
            "foundation is not a completed, formally eligible clean BEAT2 run"
        )
    receipt["completion_validation"] = "completed_training_summary_verified"
    receipt["training_summary"] = {
        "path": str(summary_path.resolve()),
        "sha256": sha256_file(summary_path),
        "completed_steps": completed_steps,
        "target_steps": target_steps,
        "stopped_early": bool(stopped_early),
        "best_step": int(summary["best_step"]),
        "data_contract_sha256": str(summary["data_contract_sha256"]),
    }
    return checkpoint, receipt


def _validate_style_cache_lineage(
    config: Mapping[str, Any],
    foundation_checkpoint: Mapping[str, Any],
    foundation_receipt: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict]:
    cache_path = Path(config["style_condition_cache"])
    clip_ids, prompts, conditions, metadata = load_condition_cache(cache_path)
    if metadata.get("artifact_kind") != MOTION_ONLY_CONDITION_CACHE_ARTIFACT_KIND:
        raise ValueError("style source is not the strict motion-only cache")
    if metadata.get("formal_episode_contract") != MOTION_ONLY_EPISODE_CONTRACT:
        raise ValueError("style cache formal episode contract changed")
    if metadata.get("cache_sha256") != sha256_file(cache_path):
        raise ValueError("style cache hash changed")
    source_checkpoint_path = Path(metadata["generator_checkpoint"])
    source_checkpoint = torch.load(
        source_checkpoint_path, map_location="cpu", weights_only=True
    )
    validate_condition_cache_for_generator(
        source_checkpoint,
        metadata,
        generator_checkpoint_path=source_checkpoint_path,
    )
    if (
        metadata.get("split_contract_sha256")
        != foundation_receipt["split_contract_sha256"]
        or metadata.get("style_contract_sha256")
        != foundation_receipt["style_contract_sha256"]
    ):
        raise ValueError("style cache/foundation split or style contract changed")
    direct = (
        metadata.get("generator_checkpoint_sha256")
        == foundation_receipt["sha256"]
    )
    if not direct:
        posttrain_source = foundation_checkpoint.get("posttrain_source") or {}
        foundation_cache = (
            foundation_checkpoint.get("data_provenance") or {}
        ).get("condition_cache") or {}
        if (
            posttrain_source.get("checkpoint_sha256")
            != metadata.get("generator_checkpoint_sha256")
            or foundation_cache.get("cache_sha256")
            != metadata.get("cache_sha256")
            or foundation_cache.get("split_contract_sha256")
            != metadata.get("split_contract_sha256")
            or foundation_cache.get("style_contract_sha256")
            != metadata.get("style_contract_sha256")
        ):
            raise ValueError(
                "trained foundation does not prove lineage through the supplied "
                "strict style-only cache"
            )
    conditions = np.asarray(conditions, dtype=np.float32)
    if (
        conditions.shape != (len(clip_ids), KIMODO_V2_CONDITION_DIM)
        or not np.isfinite(conditions).all()
        or np.any(conditions[:, : STYLE_CONTROL_SLICE.start] != 0.0)
        or np.any(conditions[:, STYLE_CONTROL_SLICE.stop :] != 0.0)
    ):
        raise ValueError("strict style cache condition layout changed")
    with np.load(cache_path, allow_pickle=False) as raw_cache:
        style_features = np.asarray(
            raw_cache["style_features"], dtype=np.float32
        ).copy()
        style_controls = np.asarray(
            raw_cache["style_controls"], dtype=np.float32
        ).copy()
    if (
        style_features.shape != (len(clip_ids), 3)
        or style_controls.shape != (len(clip_ids), 3)
        or not np.isfinite(style_features).all()
        or not np.isfinite(style_controls).all()
        or not np.array_equal(
            conditions[:, STYLE_CONTROL_SLICE], style_controls
        )
    ):
        raise ValueError("strict style cache feature/control arrays changed")
    payload = {
        "clip_ids": np.asarray(clip_ids).astype(str),
        "prompts": np.asarray(prompts).astype(str),
        "conditions": conditions,
        "style_features": style_features,
        "style_controls": style_controls,
    }
    metadata = {
        **metadata,
        "path": str(cache_path.resolve()),
        "lineage_mode": (
            "direct_clean_random_foundation"
            if direct
            else "completed_clean_posttrain_from_bound_random_foundation"
        ),
    }
    return payload, metadata


def _validate_qwen_adapter_checkpoint(
    metadata: Mapping[str, Any],
    *,
    variant: str,
    config: Mapping[str, Any],
) -> dict:
    adapter_path = Path(str(metadata.get("adapter_checkpoint") or "")).resolve()
    if not adapter_path.is_file():
        raise FileNotFoundError(f"{variant} adapter checkpoint is missing: {adapter_path}")
    if metadata.get("adapter_checkpoint_sha256") != sha256_file(adapter_path):
        raise ValueError(f"{variant} adapter checkpoint hash changed")
    payload = torch.load(adapter_path, map_location="cpu", weights_only=True)
    expected_kind = (
        "beat2_qwen_frozen_base_alignment_v1"
        if variant == "frozen_base"
        else "beat2_qwen_lora_alignment_v1"
    )
    expected_qwen_policy = (
        "official_base_frozen"
        if variant == "frozen_base"
        else "official_base_plus_beat2_only_lora"
    )
    sources = payload.get("sources") or {}
    qwen = payload.get("qwen") or {}
    if (
        payload.get("artifact_kind") != expected_kind
        or payload.get("variant") != variant
        or payload.get("no_kimodo") is not True
        or payload.get("data_policy") != NO_EXTERNAL_DATA_POLICY
        or payload.get("semantic_scope") != SEMANTIC_SCOPE_ACKNOWLEDGEMENT
        or payload.get("qwen_policy") != expected_qwen_policy
        or sources.get("manifest_sha256")
        != config["expected_manifest_sha256"]
        or sources.get("input_motion_or_text_checkpoint_count") != 0
        or qwen.get("model_name") != EXPECTED_QWEN_MODEL
        or qwen.get("revision") != EXPECTED_QWEN_REVISION
        or qwen.get("input_checkpoint_kind") != "official_base_only"
    ):
        raise ValueError(f"{variant} adapter checkpoint provenance changed")
    if variant == "frozen_base" and payload.get("qwen_lora_state_dict") is not None:
        raise ValueError("frozen-base adapter unexpectedly contains LoRA weights")
    if variant == "lora_finetuned" and not isinstance(
        payload.get("qwen_lora_state_dict"), Mapping
    ):
        raise ValueError("LoRA adapter checkpoint lacks its LoRA state")
    return {
        "path": str(adapter_path),
        "sha256": metadata["adapter_checkpoint_sha256"],
        "artifact_kind": expected_kind,
        "qwen_policy": expected_qwen_policy,
        "motion_encoder_checkpoint_sha256": sources[
            "motion_encoder_checkpoint_sha256"
        ],
    }


def _load_128d_cache(
    path: str | Path,
    *,
    variant: str,
    config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict]:
    cache_path = Path(path).resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(f"{variant} 128D cache is missing: {cache_path}")
    _reject_forbidden_path(cache_path, field=f"{variant}_condition_cache")
    metadata_path = cache_path.with_suffix(cache_path.suffix + ".json")
    metadata = _load_json_mapping(
        metadata_path, context=f"{variant} 128D cache metadata"
    )
    exact = {
        "schema_version": 1,
        "artifact_kind": "beat2_qwen_motion_latent_condition_cache_v1",
        "variant": variant,
        "data_policy": NO_EXTERNAL_DATA_POLICY,
        "no_kimodo": True,
        "condition_dim": KIMODO_MOTION_LATENT_DIM,
        "motion_latent_dim": KIMODO_MOTION_LATENT_DIM,
        "base_condition_dim": 0,
        "source_manifest_sha256": config["expected_manifest_sha256"],
        "speaker_split_contract": "fixed_17_train_4_validation_4_test",
        "semantic_scope": SEMANTIC_SCOPE_ACKNOWLEDGEMENT,
    }
    if any(metadata.get(field) != expected for field, expected in exact.items()):
        raise ValueError(f"{variant} 128D cache metadata contract changed")
    if metadata.get("cache_sha256") != sha256_file(cache_path):
        raise ValueError(f"{variant} 128D cache hash changed")
    qwen = metadata.get("qwen") or {}
    if (
        qwen.get("model_name") != EXPECTED_QWEN_MODEL
        or qwen.get("revision") != EXPECTED_QWEN_REVISION
        or qwen.get("input_checkpoint_kind") != "official_base_only"
        or qwen.get("source") != "official_huggingface_base"
    ):
        raise ValueError(f"{variant} cache Qwen base provenance changed")
    adapter_receipt = _validate_qwen_adapter_checkpoint(
        metadata, variant=variant, config=config
    )
    with np.load(cache_path, allow_pickle=False) as source:
        missing = sorted(
            (
                set(IDENTITY_ARRAY_NAMES)
                | {"conditions", "motion_latents"}
            )
            - set(source.files)
        )
        if missing:
            raise ValueError(f"{variant} 128D cache is missing arrays: {missing}")
        arrays = {name: source[name].copy() for name in source.files}
    count = len(arrays["clip_ids"])
    conditions = np.asarray(arrays["conditions"], dtype=np.float32)
    motion_latents = np.asarray(arrays["motion_latents"], dtype=np.float32)
    if (
        int(metadata.get("count", -1)) != count
        or conditions.shape != (count, KIMODO_MOTION_LATENT_DIM)
        or motion_latents.shape != conditions.shape
        or not np.isfinite(conditions).all()
        or not np.array_equal(conditions, motion_latents)
        or len(set(arrays["clip_ids"].astype(str).tolist())) != count
    ):
        raise ValueError(f"{variant} 128D cache payload is invalid")
    norms = np.linalg.norm(conditions.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4):
        raise ValueError(f"{variant} 128D conditions are not unit normalized")
    clip_order_sha256 = hashlib.sha256(
        _canonical_json(arrays["clip_ids"].astype(str).tolist())
    ).hexdigest()
    trajectory_order_sha256 = hashlib.sha256(
        _canonical_json(arrays["trajectory_sha256"].astype(str).tolist())
    ).hexdigest()
    if (
        metadata.get("clip_order_sha256") != clip_order_sha256
        or metadata.get("trajectory_order_sha256")
        != trajectory_order_sha256
    ):
        raise ValueError(f"{variant} 128D cache ordering hash changed")
    arrays["conditions"] = conditions
    arrays["motion_latents"] = motion_latents
    metadata = {
        **metadata,
        "path": str(cache_path),
        "metadata_path": str(metadata_path.resolve()),
        "adapter_receipt": adapter_receipt,
    }
    return arrays, metadata


def _validate_paired_128d_caches(
    frozen: Mapping[str, np.ndarray],
    lora: Mapping[str, np.ndarray],
) -> str:
    for name in IDENTITY_ARRAY_NAMES:
        if not np.array_equal(frozen[name], lora[name]):
            raise ValueError(f"B/C 128D cache identity mismatch for {name}")
    identity = {
        name: (
            frozen[name].astype(str).tolist()
            if frozen[name].dtype.kind in {"U", "S", "O"}
            else frozen[name].tolist()
        )
        for name in IDENTITY_ARRAY_NAMES
    }
    return hashlib.sha256(_canonical_json(identity)).hexdigest()


def _load_clean_episodes(config: Mapping[str, Any]) -> list[dict]:
    manifest_path = Path(config["manifest_path"])
    if not manifest_path.is_file():
        raise FileNotFoundError(f"BEAT2 manifest is missing: {manifest_path}")
    if sha256_file(manifest_path) != config["expected_manifest_sha256"]:
        raise ValueError("BEAT2 manifest hash changed")
    episodes = load_attached_beat_episodes(
        manifest_path,
        config["style_condition_cache"],
        allow_unreviewed=False,
        allow_unsafe_condition_cache=False,
        dataset_source=config["dataset_source"],
        speaker_namespace=config["speaker_namespace"],
        source_group_namespace=config["source_group_namespace"],
    )
    if not episodes:
        raise ValueError("BEAT2 manifest contains no accepted clean episodes")
    return episodes


def _strip_namespace(value: Any, namespace: str) -> str:
    text = str(value)
    prefix = f"{namespace}:"
    return text[len(prefix) :] if text.startswith(prefix) else text


def _align_bridge_inputs(
    *,
    episodes: Sequence[Mapping[str, Any]],
    style_payload: Mapping[str, np.ndarray],
    style_metadata: Mapping[str, Any],
    frozen: Mapping[str, np.ndarray],
    lora: Mapping[str, np.ndarray],
    foundation_checkpoint: Mapping[str, Any],
    smoke_test: bool,
    speaker_namespace: str,
) -> tuple[list[dict], dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    identity_sha256 = _validate_paired_128d_caches(frozen, lora)
    qwen_clip_ids = frozen["clip_ids"].astype(str).tolist()
    qwen_clip_set = set(qwen_clip_ids)
    if len(qwen_clip_set) != len(qwen_clip_ids):
        raise ValueError("paired Qwen cache contains duplicate clip ids")

    episode_by_clip = {str(row["clip_id"]): row for row in episodes}
    style_index = {
        clip_id: index
        for index, clip_id in enumerate(style_payload["clip_ids"].astype(str))
    }
    if set(episode_by_clip) != set(style_index):
        raise ValueError("strict style cache and BEAT2 manifest clip sets differ")
    if smoke_test:
        if not qwen_clip_set or not qwen_clip_set.issubset(episode_by_clip):
            raise ValueError("smoke Qwen cache is not a BEAT2 manifest subset")
    elif qwen_clip_set != set(episode_by_clip):
        raise ValueError(
            "production Qwen cache must cover every clean foundation episode"
        )

    style_record_by_clip = {
        str(record["clip_id"]): record
        for record in style_metadata.get("episodes") or ()
    }
    if set(style_record_by_clip) != set(style_index):
        raise ValueError("style cache episode metadata clip set changed")
    foundation_split_records = (
        (foundation_checkpoint.get("v2_contracts") or {}).get("split") or {}
    ).get("episodes") or ()
    foundation_split_by_clip = {
        str(record["clip_id"]): str(record["split"])
        for record in foundation_split_records
    }
    if set(foundation_split_by_clip) != set(episode_by_clip):
        raise ValueError("foundation fixed-split contract clip set changed")

    selected = []
    style_conditions = []
    style_features = []
    style_controls = []
    for index, clip_id in enumerate(qwen_clip_ids):
        episode = episode_by_clip[clip_id]
        style_row = style_index[clip_id]
        style_record = style_record_by_clip[clip_id]
        qwen_prompt = str(frozen["prompts"][index])
        qwen_split = str(frozen["fixed_split_assignments"][index])
        qwen_trajectory = str(frozen["trajectory_sha256"][index])
        qwen_speaker = str(frozen["speaker_keys"][index])
        episode_speaker = _strip_namespace(
            episode["speaker_key"], speaker_namespace
        )
        if (
            qwen_prompt != str(style_payload["prompts"][style_row])
            or qwen_split != str(episode.get("fixed_split_assignment"))
            or qwen_split != str(style_record.get("fixed_split_assignment"))
            or qwen_split != foundation_split_by_clip[clip_id]
            or qwen_trajectory != str(episode.get("trajectory_sha256"))
            or qwen_trajectory != str(style_record.get("trajectory_sha256"))
            or qwen_speaker != episode_speaker
        ):
            raise ValueError(
                f"{clip_id}: Qwen/style/trajectory/fixed-split identity changed"
            )
        selected.append(dict(episode))
        style_conditions.append(style_payload["conditions"][style_row])
        style_features.append(style_payload["style_features"][style_row])
        style_controls.append(style_payload["style_controls"][style_row])

    selected_splits = {
        name: sum(
            str(row.get("fixed_split_assignment")) == name for row in selected
        )
        for name in EXPECTED_SPLIT_COUNTS
    }
    if any(count == 0 for count in selected_splits.values()):
        raise ValueError("bridge cache must retain train, validation, and test rows")
    if not smoke_test and selected_splits != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            f"production bridge fixed-split counts changed: {selected_splits}"
        )
    common = {
        "clip_ids": np.asarray(qwen_clip_ids),
        "prompts": frozen["prompts"].astype(str),
        "fixed_split_assignments": frozen[
            "fixed_split_assignments"
        ].astype(str),
        "speaker_keys": frozen["speaker_keys"].astype(str),
        "semantic_group_indices": np.asarray(
            frozen["semantic_group_indices"], dtype=np.int64
        ),
        "trajectory_sha256": frozen["trajectory_sha256"].astype(str),
        "style_conditions": np.stack(style_conditions).astype(np.float32),
        "style_features": np.stack(style_features).astype(np.float32),
        "style_controls": np.stack(style_controls).astype(np.float32),
    }
    return selected, dict(frozen), dict(lora), {
        "identity_arrays_sha256": identity_sha256,
        "selected_split_counts": selected_splits,
        "common": common,
    }


def _bridge_cache_paths(output_dir: str | Path, variant: str) -> tuple[Path, Path]:
    cache = (
        Path(output_dir)
        / "prepared"
        / f"conditions_264d_{variant}.experimental.npz"
    )
    return cache, cache.with_suffix(cache.suffix + ".json")


def _write_bridge_cache(
    *,
    variant: str,
    source_128d: Mapping[str, np.ndarray],
    source_metadata: Mapping[str, Any],
    common: Mapping[str, np.ndarray],
    identity_arrays_sha256: str,
    foundation_receipt: Mapping[str, Any],
    style_metadata: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict:
    conditions = np.asarray(common["style_conditions"], dtype=np.float32).copy()
    latents = np.asarray(source_128d["conditions"], dtype=np.float32)
    conditions[:, KIMODO_CONDITION_DIM:] = latents
    if (
        conditions.shape
        != (len(common["clip_ids"]), KIMODO_V2_CONDITION_DIM)
        or np.any(conditions[:, : STYLE_CONTROL_SLICE.start] != 0.0)
        or not np.array_equal(
            conditions[:, STYLE_CONTROL_SLICE], common["style_controls"]
        )
        or not np.array_equal(
            conditions[:, KIMODO_CONDITION_DIM:], latents
        )
        or not np.isfinite(conditions).all()
    ):
        raise ValueError(f"{variant} experimental 264D bridge layout is invalid")
    cache_path, metadata_path = _bridge_cache_paths(
        config["output_dir"], variant
    )
    _atomic_npz_save(
        cache_path,
        clip_ids=np.asarray(common["clip_ids"]),
        prompts=np.asarray(common["prompts"]),
        conditions=conditions,
        latent_128d=latents,
        style_features=np.asarray(common["style_features"], dtype=np.float32),
        style_controls=np.asarray(common["style_controls"], dtype=np.float32),
        fixed_split_assignments=np.asarray(
            common["fixed_split_assignments"]
        ),
        speaker_keys=np.asarray(common["speaker_keys"]),
        semantic_group_indices=np.asarray(
            common["semantic_group_indices"], dtype=np.int64
        ),
        trajectory_sha256=np.asarray(common["trajectory_sha256"]),
    )
    records = []
    for index, clip_id in enumerate(common["clip_ids"].astype(str)):
        records.append(
            {
                "clip_id": clip_id,
                "prompt_sha256": hashlib.sha256(
                    str(common["prompts"][index]).encode("utf-8")
                ).hexdigest(),
                "fixed_split_assignment": str(
                    common["fixed_split_assignments"][index]
                ),
                "trajectory_sha256": str(
                    common["trajectory_sha256"][index]
                ),
                "style_features": common["style_features"][index].tolist(),
                "style_controls": common["style_controls"][index].tolist(),
                "latent_sha256": hashlib.sha256(
                    np.ascontiguousarray(latents[index]).tobytes()
                ).hexdigest(),
            }
        )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": BRIDGE_CACHE_ARTIFACT_KIND,
        "experimental_only": True,
        "formal_release_eligible": False,
        "semantic_scope": SEMANTIC_SCOPE_ACKNOWLEDGEMENT,
        "semantic_supervision_status": (
            "official_metadata_alignment_only_not_formal_robot_semantics"
        ),
        "variant": variant,
        "data_policy": NO_EXTERNAL_DATA_POLICY,
        "no_kimodo": True,
        "condition_policy": EXPERIMENTAL_CONDITION_POLICY,
        "condition_dim": KIMODO_V2_CONDITION_DIM,
        "base_condition_dim": KIMODO_CONDITION_DIM,
        "motion_latent_dim": KIMODO_MOTION_LATENT_DIM,
        "layout": {
            "exact_zero": [0, STYLE_CONTROL_SLICE.start],
            "trajectory_style": [
                STYLE_CONTROL_SLICE.start,
                STYLE_CONTROL_SLICE.stop,
            ],
            "experimental_metadata_text_latent": [
                KIMODO_CONDITION_DIM,
                KIMODO_V2_CONDITION_DIM,
            ],
        },
        "count": len(conditions),
        "cache_sha256": sha256_file(cache_path),
        "identity_arrays_sha256": identity_arrays_sha256,
        "source_manifest_sha256": config["expected_manifest_sha256"],
        "foundation_checkpoint": foundation_receipt["path"],
        "foundation_checkpoint_sha256": foundation_receipt["sha256"],
        "foundation_data_isolation_contract_sha256": foundation_receipt[
            "data_isolation_contract_sha256"
        ],
        "foundation_split_contract_sha256": foundation_receipt[
            "split_contract_sha256"
        ],
        "foundation_style_contract_sha256": foundation_receipt[
            "style_contract_sha256"
        ],
        "style_cache": style_metadata["path"],
        "style_cache_sha256": style_metadata["cache_sha256"],
        "source_128d_cache": source_metadata["path"],
        "source_128d_cache_sha256": source_metadata["cache_sha256"],
        "adapter_checkpoint": source_metadata["adapter_receipt"]["path"],
        "adapter_checkpoint_sha256": source_metadata["adapter_receipt"]["sha256"],
        "qwen": deepcopy(source_metadata["qwen"]),
        "episodes": records,
    }
    _atomic_json_save(metadata, metadata_path)
    return {
        "path": str(cache_path.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "cache_sha256": metadata["cache_sha256"],
        "variant": variant,
        "artifact_kind": BRIDGE_CACHE_ARTIFACT_KIND,
        "source_128d_cache_sha256": source_metadata["cache_sha256"],
        "style_cache_sha256": style_metadata["cache_sha256"],
        "identity_arrays_sha256": identity_arrays_sha256,
        "count": len(conditions),
    }


def _training_config_for_core(config: Mapping[str, Any]) -> dict:
    training = config["training"]
    return {
        "batch_size": int(training["batch_size"]),
        "batching": deepcopy(training["batching"]),
        "sampler": deepcopy(training["sampler"]),
    }


def _sampler_schedule_probe(
    train_episodes: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    microbatch_count: int = 16,
) -> dict:
    core = _training_config_for_core(config)
    sampler = _sampler_for_config(
        train_episodes, core, seed=int(config["seed"]) + 17
    )
    records = []
    if not isinstance(sampler, NativeLengthBucketSampler):
        raise ValueError("experimental posttrain requires native bucket sampling")
    for _ in range(int(microbatch_count)):
        batch, plan = sampler.sample_microbatch(
            remaining_effective_batch=int(
                config["training"]["batching"]["target_effective_batch_size"]
            ),
            semantic_tokens=7,
            max_batch_size=int(config["training"]["batch_size"]),
            batching=config["training"]["batching"],
        )
        records.append(
            {
                "clip_ids": [str(row["clip_id"]) for row in batch],
                "bucket_frames": int(plan["bucket_frames"]),
                "microbatch_size": int(plan["microbatch_size"]),
            }
        )
    return {
        "probe_microbatch_count": len(records),
        "sha256": hashlib.sha256(_canonical_json(records)).hexdigest(),
        "records": records,
    }


def _prepared_pair_path(config: Mapping[str, Any]) -> Path:
    return Path(config["output_dir"]) / "prepared" / "pair_contract.json"


def _config_contract(config: Mapping[str, Any], *, smoke_test: bool) -> dict:
    selected = deepcopy(dict(config))
    selected["smoke_test"] = bool(smoke_test)
    return _self_hashed(
        {
            "contract_type": "beat2_experimental_metadata_posttrain_config",
            "contract_version": 1,
            "resolved": selected,
        }
    )


def _bridge_cache_receipt_from_disk(path: str | Path) -> dict:
    cache_path = Path(path)
    metadata_path = cache_path.with_suffix(cache_path.suffix + ".json")
    metadata = _load_json_mapping(
        metadata_path, context="experimental bridge-cache metadata"
    )
    if (
        metadata.get("artifact_kind") != BRIDGE_CACHE_ARTIFACT_KIND
        or metadata.get("experimental_only") is not True
        or metadata.get("formal_release_eligible") is not False
        or metadata.get("semantic_scope") != SEMANTIC_SCOPE_ACKNOWLEDGEMENT
        or metadata.get("cache_sha256") != sha256_file(cache_path)
    ):
        raise ValueError("experimental bridge-cache metadata is invalid")
    return metadata


def validate_pair_contract(
    config: Mapping[str, Any], *, smoke_test: bool
) -> dict:
    pair_path = _prepared_pair_path(config)
    if not pair_path.is_file():
        raise FileNotFoundError(f"prepared pair contract is missing: {pair_path}")
    pair = _load_json_mapping(pair_path, context="prepared pair contract")
    _validate_self_hash(pair, context="prepared pair contract")
    if (
        pair.get("artifact_kind") != PAIR_ARTIFACT_KIND
        or pair.get("experimental_only") is not True
        or pair.get("formal_release_eligible") is not False
        or pair.get("semantic_scope") != SEMANTIC_SCOPE_ACKNOWLEDGEMENT
        or pair.get("smoke_test") is not bool(smoke_test)
        or pair.get("config_contract_sha256")
        != _config_contract(config, smoke_test=smoke_test)["sha256"]
    ):
        raise ValueError("prepared pair contract/config changed")
    paths_to_hash = [
        (pair["foundation"]["path"], pair["foundation"]["sha256"]),
        (pair["style_cache"]["path"], pair["style_cache"]["sha256"]),
        (pair["manifest"]["path"], pair["manifest"]["sha256"]),
    ]
    for variant in VARIANTS:
        paths_to_hash.extend(
            [
                (
                    pair["branches"][variant]["source_128d_cache"]["path"],
                    pair["branches"][variant]["source_128d_cache"]["sha256"],
                ),
                (
                    pair["branches"][variant]["bridge_cache"]["path"],
                    pair["branches"][variant]["bridge_cache"]["sha256"],
                ),
            ]
        )
    for path, expected in paths_to_hash:
        if not Path(path).is_file() or sha256_file(path) != expected:
            raise ValueError(f"prepared pair input changed: {path}")
    for variant in VARIANTS:
        metadata = _bridge_cache_receipt_from_disk(
            pair["branches"][variant]["bridge_cache"]["path"]
        )
        if (
            metadata.get("variant") != variant
            or metadata.get("foundation_checkpoint_sha256")
            != pair["foundation"]["sha256"]
            or metadata.get("style_cache_sha256")
            != pair["style_cache"]["sha256"]
            or metadata.get("identity_arrays_sha256")
            != pair["identity_arrays_sha256"]
        ):
            raise ValueError(f"{variant} prepared bridge-cache receipt changed")
    return pair


def prepare_pair(
    config: Mapping[str, Any],
    *,
    smoke_test: bool,
    overwrite: bool,
) -> dict:
    pair_path = _prepared_pair_path(config)
    if pair_path.is_file() and not overwrite:
        return validate_pair_contract(config, smoke_test=smoke_test)
    output_dir = Path(config["output_dir"])
    branch_outputs = [
        output_dir / variant / "generator_experimental.pt"
        for variant in VARIANTS
    ]
    if overwrite and any(path.exists() for path in branch_outputs):
        raise FileExistsError(
            "refusing to overwrite preparation after branch checkpoints exist"
        )

    foundation_checkpoint, foundation_receipt = _foundation_receipt(
        config, smoke_test=smoke_test
    )
    style_payload, style_metadata = _validate_style_cache_lineage(
        config, foundation_checkpoint, foundation_receipt
    )
    episodes = _load_clean_episodes(config)
    frozen, frozen_metadata = _load_128d_cache(
        config["frozen_condition_cache"],
        variant="frozen_base",
        config=config,
    )
    lora, lora_metadata = _load_128d_cache(
        config["lora_condition_cache"],
        variant="lora_finetuned",
        config=config,
    )
    selected, frozen, lora, alignment = _align_bridge_inputs(
        episodes=episodes,
        style_payload=style_payload,
        style_metadata=style_metadata,
        frozen=frozen,
        lora=lora,
        foundation_checkpoint=foundation_checkpoint,
        smoke_test=smoke_test,
        speaker_namespace=config["speaker_namespace"],
    )
    splits, selected_split_contract = strict_group_split(
        selected,
        seed=int(config["seed"]),
        fractions={"train": 0.7, "validation": 0.15, "test": 0.15},
    )
    if selected_split_contract.get("assignment_policy") != (
        "fixed_pre_quarantine_assignment"
    ):
        raise ValueError("bridge preparation did not retain fixed split assignments")
    sampler_probe = _sampler_schedule_probe(splits["train"], config)

    common = alignment["common"]
    frozen_bridge = _write_bridge_cache(
        variant="frozen_base",
        source_128d=frozen,
        source_metadata=frozen_metadata,
        common=common,
        identity_arrays_sha256=alignment["identity_arrays_sha256"],
        foundation_receipt=foundation_receipt,
        style_metadata=style_metadata,
        config=config,
    )
    lora_bridge = _write_bridge_cache(
        variant="lora_finetuned",
        source_128d=lora,
        source_metadata=lora_metadata,
        common=common,
        identity_arrays_sha256=alignment["identity_arrays_sha256"],
        foundation_receipt=foundation_receipt,
        style_metadata=style_metadata,
        config=config,
    )
    config_contract = _config_contract(config, smoke_test=smoke_test)
    pair = _self_hashed(
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": PAIR_ARTIFACT_KIND,
            "experimental_only": True,
            "formal_release_eligible": False,
            "semantic_scope": SEMANTIC_SCOPE_ACKNOWLEDGEMENT,
            "semantic_supervision_status": (
                "official_metadata_alignment_only_not_formal_robot_semantics"
            ),
            "data_policy": NO_EXTERNAL_DATA_POLICY,
            "no_kimodo": True,
            "smoke_test": bool(smoke_test),
            "config_contract_sha256": config_contract["sha256"],
            "foundation": {
                "path": foundation_receipt["path"],
                "sha256": foundation_receipt["sha256"],
                "architecture": foundation_receipt["architecture"],
                "formal_episode_contract": foundation_receipt[
                    "formal_episode_contract"
                ],
                "data_isolation_contract_sha256": foundation_receipt[
                    "data_isolation_contract_sha256"
                ],
                "split_contract_sha256": foundation_receipt[
                    "split_contract_sha256"
                ],
                "style_contract_sha256": foundation_receipt[
                    "style_contract_sha256"
                ],
                "completion_validation": foundation_receipt[
                    "completion_validation"
                ],
                "training_summary": foundation_receipt["training_summary"],
            },
            "manifest": {
                "path": str(Path(config["manifest_path"]).resolve()),
                "sha256": config["expected_manifest_sha256"],
            },
            "style_cache": {
                "path": style_metadata["path"],
                "sha256": style_metadata["cache_sha256"],
                "artifact_kind": style_metadata["artifact_kind"],
                "lineage_mode": style_metadata["lineage_mode"],
            },
            "identity_arrays_sha256": alignment["identity_arrays_sha256"],
            "selected_episode_count": len(selected),
            "selected_split_counts": alignment["selected_split_counts"],
            "selected_split_contract_sha256": selected_split_contract["sha256"],
            "sampler_initial_schedule_probe": sampler_probe,
            "optimization_contract": {
                "training_policy": TRAINING_POLICY,
                "seed": int(config["seed"]),
                "steps": int(config["training"]["steps"]),
                "optimizer": {
                    "type": "AdamW",
                    "lr": float(config["training"]["lr"]),
                    "minimum_lr_ratio": float(
                        config["training"]["minimum_lr_ratio"]
                    ),
                    "warmup_steps": int(config["training"]["warmup_steps"]),
                    "weight_decay": float(
                        config["training"]["weight_decay"]
                    ),
                    "adam_eps": float(config["training"]["adam_eps"]),
                    "max_grad_norm": float(
                        config["training"]["max_grad_norm"]
                    ),
                    "plan_nonlatent_weight_decay": 0.0,
                },
                "ema_decay": float(config["training"]["ema_decay"]),
                "loss": deepcopy(config["training"]["loss"]),
                "batching": deepcopy(config["training"]["batching"]),
                "sampler": deepcopy(config["training"]["sampler"]),
                "early_stopping": "disabled_fixed_equal_step_budget",
            },
            "branches": {
                "frozen_base": {
                    "source_128d_cache": {
                        "path": frozen_metadata["path"],
                        "sha256": frozen_metadata["cache_sha256"],
                        "adapter_checkpoint_sha256": frozen_metadata[
                            "adapter_checkpoint_sha256"
                        ],
                    },
                    "bridge_cache": {
                        "path": frozen_bridge["path"],
                        "sha256": frozen_bridge["cache_sha256"],
                    },
                },
                "lora_finetuned": {
                    "source_128d_cache": {
                        "path": lora_metadata["path"],
                        "sha256": lora_metadata["cache_sha256"],
                        "adapter_checkpoint_sha256": lora_metadata[
                            "adapter_checkpoint_sha256"
                        ],
                    },
                    "bridge_cache": {
                        "path": lora_bridge["path"],
                        "sha256": lora_bridge["cache_sha256"],
                    },
                },
            },
            "only_allowed_branch_input_difference": (
                "source 128D condition cache and its Qwen adapter provenance"
            ),
        }
    )
    _atomic_json_save(config_contract, output_dir / "resolved_config_contract.json")
    _atomic_json_save(pair, pair_path)
    return validate_pair_contract(config, smoke_test=smoke_test)


def _state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(_canonical_json(list(value.shape)))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _numeric_delta(left: Any, right: Any) -> Any:
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        return float(right) - float(left)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return {
            key: _numeric_delta(left[key], right[key])
            for key in sorted(set(left) & set(right))
            if isinstance(left[key], (Mapping, int, float))
            and isinstance(right[key], (Mapping, int, float))
        }
    return None


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _load_bridge_cache(
    pair: Mapping[str, Any], *, variant: str
) -> tuple[dict[str, np.ndarray], dict]:
    receipt = pair["branches"][variant]["bridge_cache"]
    cache_path = Path(receipt["path"])
    metadata = _bridge_cache_receipt_from_disk(cache_path)
    if (
        metadata.get("variant") != variant
        or metadata.get("cache_sha256") != receipt["sha256"]
        or metadata.get("foundation_checkpoint_sha256")
        != pair["foundation"]["sha256"]
        or metadata.get("source_manifest_sha256") != pair["manifest"]["sha256"]
        or metadata.get("identity_arrays_sha256")
        != pair["identity_arrays_sha256"]
    ):
        raise ValueError(f"{variant} bridge cache is outside the paired contract")
    with np.load(cache_path, allow_pickle=False) as payload:
        required = {
            "clip_ids",
            "prompts",
            "conditions",
            "latent_128d",
            "style_features",
            "style_controls",
            "fixed_split_assignments",
            "speaker_keys",
            "semantic_group_indices",
            "trajectory_sha256",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"{variant} bridge cache is missing arrays: {missing}")
        arrays = {name: payload[name].copy() for name in payload.files}
    conditions = np.asarray(arrays["conditions"], dtype=np.float32)
    latents = np.asarray(arrays["latent_128d"], dtype=np.float32)
    controls = np.asarray(arrays["style_controls"], dtype=np.float32)
    count = len(arrays["clip_ids"])
    if (
        conditions.shape != (count, KIMODO_V2_CONDITION_DIM)
        or latents.shape != (count, KIMODO_MOTION_LATENT_DIM)
        or controls.shape != (count, 3)
        or np.any(conditions[:, : STYLE_CONTROL_SLICE.start] != 0.0)
        or not np.array_equal(conditions[:, STYLE_CONTROL_SLICE], controls)
        or not np.array_equal(
            conditions[:, KIMODO_CONDITION_DIM:], latents
        )
        or not np.isfinite(conditions).all()
    ):
        raise ValueError(f"{variant} bridge cache condition layout changed")
    arrays["conditions"] = conditions
    arrays["latent_128d"] = latents
    return arrays, metadata


def _load_branch_episodes(
    config: Mapping[str, Any],
    pair: Mapping[str, Any],
    *,
    variant: str,
) -> tuple[dict[str, list[dict]], dict, dict]:
    arrays, metadata = _load_bridge_cache(pair, variant=variant)
    clean_episodes = _load_clean_episodes(config)
    clean_by_clip = {str(row["clip_id"]): row for row in clean_episodes}
    selected = []
    for index, raw_clip_id in enumerate(arrays["clip_ids"]):
        clip_id = str(raw_clip_id)
        if clip_id not in clean_by_clip:
            raise ValueError(f"{variant} bridge references unknown clip {clip_id}")
        clean = clean_by_clip[clip_id]
        if (
            str(clean.get("fixed_split_assignment"))
            != str(arrays["fixed_split_assignments"][index])
            or str(clean.get("trajectory_sha256"))
            != str(arrays["trajectory_sha256"][index])
        ):
            raise ValueError(f"{clip_id}: bridge split/trajectory changed at training")
        item = dict(clean)
        item["condition"] = np.asarray(
            arrays["conditions"][index], dtype=np.float32
        ).copy()
        item["experimental_semantic_group_index"] = int(
            arrays["semantic_group_indices"][index]
        )
        item["experimental_condition_contract"] = {
            "artifact_kind": BRIDGE_CACHE_ARTIFACT_KIND,
            "variant": variant,
            "cache_sha256": metadata["cache_sha256"],
            "semantic_scope": SEMANTIC_SCOPE_ACKNOWLEDGEMENT,
            "formal_release_eligible": False,
        }
        selected.append(item)
    splits, split_contract = strict_group_split(
        selected,
        seed=int(config["seed"]),
        fractions={"train": 0.7, "validation": 0.15, "test": 0.15},
    )
    if (
        split_contract.get("sha256")
        != pair["selected_split_contract_sha256"]
        or {name: len(rows) for name, rows in splits.items()}
        != pair["selected_split_counts"]
    ):
        raise ValueError(f"{variant} branch split contract changed")
    return splits, split_contract, metadata


def _stable_subset(
    episodes: Sequence[Mapping[str, Any]],
    *,
    count: int,
    seed: int,
    label: str,
) -> list[dict]:
    ordered = sorted(
        (dict(row) for row in episodes),
        key=lambda row: hashlib.sha256(
            f"{int(seed)}:{label}:{row['clip_id']}".encode("utf-8")
        ).hexdigest(),
    )
    return ordered[: min(int(count), len(ordered))]


def _diverse_diagnostic_subset(
    episodes: Sequence[Mapping[str, Any]], *, count: int, seed: int
) -> list[dict]:
    ordered = _stable_subset(
        episodes, count=len(episodes), seed=seed, label="condition_diagnostic"
    )
    selected = []
    seen_groups = set()
    for row in ordered:
        group = row.get("experimental_semantic_group_index")
        if group in seen_groups:
            continue
        selected.append(row)
        seen_groups.add(group)
        if len(selected) >= int(count):
            return selected
    selected_ids = {str(row["clip_id"]) for row in selected}
    for row in ordered:
        if str(row["clip_id"]) not in selected_ids:
            selected.append(row)
            selected_ids.add(str(row["clip_id"]))
        if len(selected) >= int(count):
            break
    return selected


def _configure_condition_path_optimizer(
    model: torch.nn.Module,
    config: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[torch.optim.Optimizer, torch.Tensor, dict]:
    model.requires_grad_(False)
    named = dict(model.named_parameters())
    if MOTION_LATENT_WEIGHT_NAME not in named or PLAN_WEIGHT_NAME not in named:
        raise ValueError("foundation architecture lacks the controlled latent weights")
    latent_weight = named[MOTION_LATENT_WEIGHT_NAME]
    plan_weight = named[PLAN_WEIGHT_NAME]
    if tuple(latent_weight.shape)[1:] != (KIMODO_MOTION_LATENT_DIM,):
        raise ValueError("motion latent first-layer weight shape changed")
    if tuple(plan_weight.shape)[1:] != (KIMODO_V2_CONDITION_DIM,):
        raise ValueError("planner first-layer weight shape changed")
    latent_weight.requires_grad_(True)
    plan_weight.requires_grad_(True)
    plan_gradient_mask = torch.zeros_like(plan_weight, device=device)
    plan_gradient_mask[:, KIMODO_CONDITION_DIM:] = 1.0
    plan_weight.register_hook(lambda gradient: gradient * plan_gradient_mask)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [latent_weight],
                "weight_decay": float(training["weight_decay"]),
                "role": "motion_latent_first_layer_weight",
            },
            {
                "params": [plan_weight],
                # Weight decay would silently alter masked non-latent columns.
                "weight_decay": 0.0,
                "role": "planner_latent_columns_only",
            },
        ],
        lr=float(training["lr"]),
        eps=float(training["adam_eps"]),
    )
    receipt = {
        "policy": TRAINING_POLICY,
        "full_network": False,
        "trainable_tensor_names": [
            MOTION_LATENT_WEIGHT_NAME,
            PLAN_WEIGHT_NAME,
        ],
        "effective_trainable_parameter_names": [
            MOTION_LATENT_WEIGHT_NAME,
            f"{PLAN_WEIGHT_NAME}[:,{KIMODO_CONDITION_DIM}:{KIMODO_V2_CONDITION_DIM}]",
        ],
        "optimizer_parameter_count": int(
            latent_weight.numel() + plan_weight.numel()
        ),
        "effective_trainable_parameter_count": int(
            latent_weight.numel()
            + plan_weight.shape[0] * KIMODO_MOTION_LATENT_DIM
        ),
        "plan_gradient_mask_sha256": hashlib.sha256(
            plan_gradient_mask.detach().cpu().numpy().tobytes()
        ).hexdigest(),
        "zero_latent_preservation_mechanism": (
            "only latent-input weights update; biases/downstream weights frozen; "
            "planner nonlatent columns gradient-masked with zero weight decay"
        ),
    }
    return optimizer, plan_gradient_mask, receipt


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer, *, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                state[name] = value.to(device)


@torch.no_grad()
def _restore_ema_preserved_state(
    ema: ModelEMA, foundation_state: Mapping[str, torch.Tensor]
) -> None:
    for name, source in foundation_state.items():
        destination = ema.shadow[name]
        source_value = source.to(
            device=destination.device, dtype=destination.dtype
        )
        if name == MOTION_LATENT_WEIGHT_NAME:
            continue
        if name == PLAN_WEIGHT_NAME:
            destination[:, :KIMODO_CONDITION_DIM].copy_(
                source_value[:, :KIMODO_CONDITION_DIM]
            )
        else:
            destination.copy_(source_value)


def _lr_scale(
    step: int, *, total_steps: int, warmup_steps: int, minimum_ratio: float
) -> float:
    if warmup_steps and step <= warmup_steps:
        return max(1e-8, float(step) / float(warmup_steps))
    decay_steps = max(1, int(total_steps) - int(warmup_steps))
    progress = min(
        1.0,
        max(
            0.0,
            (float(step) - float(warmup_steps)) / float(decay_steps),
        ),
    )
    return float(minimum_ratio) + (1.0 - float(minimum_ratio)) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def _zero_latent_probe(
    model: torch.nn.Module,
    episodes: Sequence[Mapping[str, Any]],
    *,
    action_stats: Mapping[str, Any],
    device: torch.device,
    batching: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [dict(row) for row in episodes[: min(2, len(episodes))]]
    if not rows:
        raise ValueError("zero-latent probe requires episodes")
    actions, conditions, _, _, frame_valid = _batch_tensors_for_config(
        rows,
        frame_count=max(batching["length_buckets"]),
        action_stats=action_stats,
        device=device,
        batching=batching,
    )
    conditions[:, KIMODO_CONDITION_DIM:] = 0.0
    t = torch.linspace(
        0.25, 0.75, actions.shape[0], dtype=actions.dtype, device=device
    )
    x_t = actions * 0.375
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            predicted = forward_with_frame_mask(
                model, x_t, t, conditions, frame_valid
            )
    finally:
        model.train(was_training)
    return predicted.detach().cpu(), frame_valid.detach().cpu()


def _condition_response_diagnostics(
    model: torch.nn.Module,
    episodes: Sequence[Mapping[str, Any]],
    *,
    action_stats: Mapping[str, Any],
    device: torch.device,
    batching: Mapping[str, Any],
) -> dict:
    if len(episodes) < 2:
        raise ValueError("condition diagnostics require at least two episodes")
    latents = [
        np.asarray(row["condition"], dtype=np.float32)[
            KIMODO_CONDITION_DIM:
        ].copy()
        for row in episodes
    ]
    aligned_zero_squared = 0.0
    aligned_shuffled_squared = 0.0
    observed_values = 0
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for index, episode in enumerate(episodes):
                actions, conditions, _, _, frame_valid = _batch_tensors_for_config(
                    [episode],
                    frame_count=max(batching["length_buckets"]),
                    action_stats=action_stats,
                    device=device,
                    batching=batching,
                )
                zero = conditions.clone()
                zero[:, KIMODO_CONDITION_DIM:] = 0.0
                shuffled = conditions.clone()
                shuffled[:, KIMODO_CONDITION_DIM:] = torch.as_tensor(
                    latents[(index + 1) % len(latents)],
                    dtype=conditions.dtype,
                    device=device,
                )
                t = torch.full(
                    (1,), 0.5, dtype=actions.dtype, device=device
                )
                x_t = actions * 0.375
                aligned_prediction = forward_with_frame_mask(
                    model, x_t, t, conditions, frame_valid
                )
                zero_prediction = forward_with_frame_mask(
                    model, x_t, t, zero, frame_valid
                )
                shuffled_prediction = forward_with_frame_mask(
                    model, x_t, t, shuffled, frame_valid
                )
                mask = frame_valid[:, :, None].expand_as(aligned_prediction)
                aligned_zero_squared += float(
                    (
                        (aligned_prediction - zero_prediction)
                        .square()
                        .masked_select(mask)
                        .sum()
                    ).cpu()
                )
                aligned_shuffled_squared += float(
                    (
                        (aligned_prediction - shuffled_prediction)
                        .square()
                        .masked_select(mask)
                        .sum()
                    ).cpu()
                )
                observed_values += int(mask.sum().cpu())
    finally:
        model.train(was_training)
    return {
        "episode_count": len(episodes),
        "aligned_vs_zero_prediction_rms": math.sqrt(
            aligned_zero_squared / max(1, observed_values)
        ),
        "aligned_vs_shuffled_prediction_rms": math.sqrt(
            aligned_shuffled_squared / max(1, observed_values)
        ),
        "diagnostic_scope": (
            "fixed-x_t condition-response mechanism diagnostic; not a formal "
            "semantic-quality metric"
        ),
    }


def _preservation_audit(
    *,
    final_state: Mapping[str, torch.Tensor],
    foundation_state: Mapping[str, torch.Tensor],
    baseline_zero_output: torch.Tensor,
    final_zero_output: torch.Tensor,
    frame_valid: torch.Tensor,
) -> dict:
    if set(final_state) != set(foundation_state):
        raise ValueError("experimental model state keys differ from foundation")
    frozen_max = 0.0
    plan_nonlatent_max = 0.0
    for name in sorted(final_state):
        current = final_state[name].detach().cpu()
        source = foundation_state[name].detach().cpu()
        if name == MOTION_LATENT_WEIGHT_NAME:
            continue
        if name == PLAN_WEIGHT_NAME:
            nonlatent = (
                current[:, :KIMODO_CONDITION_DIM]
                - source[:, :KIMODO_CONDITION_DIM]
            ).abs()
            plan_nonlatent_max = max(
                plan_nonlatent_max,
                float(nonlatent.max()) if nonlatent.numel() else 0.0,
            )
            continue
        if torch.is_floating_point(current):
            error = float((current - source).abs().max())
        else:
            error = 0.0 if torch.equal(current, source) else float("inf")
        frozen_max = max(frozen_max, error)
    mask = frame_valid[:, :, None].expand_as(final_zero_output)
    zero_error = float(
        (final_zero_output - baseline_zero_output)
        .abs()
        .masked_select(mask)
        .max()
    )
    passed = (
        frozen_max == 0.0
        and plan_nonlatent_max == 0.0
        and zero_error == 0.0
    )
    return {
        "zero_latent_exact_equivalence_passed": passed,
        "zero_latent_max_abs_error": zero_error,
        "frozen_parameter_max_abs_error": frozen_max,
        "nonlatent_plan_columns_max_abs_error": plan_nonlatent_max,
        "foundation_file_unchanged": True,
    }


def _branch_paths(config: Mapping[str, Any], variant: str) -> dict[str, Path]:
    root = Path(config["output_dir"]) / variant
    return {
        "root": root,
        "checkpoint": root / "generator_experimental.pt",
        "summary": root / "training_summary.json",
        "state": root / "last_state.pt",
        "progress": root / "progress.jsonl",
    }


def _validate_experimental_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    pair: Mapping[str, Any],
    variant: str,
) -> None:
    cache_receipt = payload.get("condition_cache_receipt") or {}
    foundation_receipt = payload.get("foundation_receipt") or {}
    pair_receipt = payload.get("pair_contract") or {}
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("artifact_kind") != CHECKPOINT_ARTIFACT_KIND
        or payload.get("experimental_only") is not True
        or payload.get("formal_release_eligible") is not False
        or payload.get("semantic_scope") != SEMANTIC_SCOPE_ACKNOWLEDGEMENT
        or payload.get("variant") != variant
        or payload.get("architecture") != ULA_MMDIT_V3_ADALN_ARCHITECTURE
        or int(payload.get("action_dim", -1)) != ACTION_DIM
        or int(payload.get("condition_dim", -1)) != KIMODO_V2_CONDITION_DIM
        or foundation_receipt.get("checkpoint_sha256")
        != pair["foundation"]["sha256"]
        or cache_receipt.get("variant") != variant
        or cache_receipt.get("cache_sha256")
        != pair["branches"][variant]["bridge_cache"]["sha256"]
        or cache_receipt.get("identity_arrays_sha256")
        != pair["identity_arrays_sha256"]
        or cache_receipt.get("no_kimodo") is not True
        or cache_receipt.get("semantic_scope")
        != SEMANTIC_SCOPE_ACKNOWLEDGEMENT
        or pair_receipt.get("sha256") != pair["sha256"]
        or (payload.get("preservation") or {}).get(
            "zero_latent_exact_equivalence_passed"
        )
        is not True
        or not isinstance(payload.get("model_state_dict"), Mapping)
    ):
        raise ValueError(f"{variant} experimental checkpoint contract changed")


def _save_resume_state(
    path: Path,
    *,
    variant: str,
    pair: Mapping[str, Any],
    config: Mapping[str, Any],
    step: int,
    model: torch.nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    sampler: NativeLengthBucketSampler,
    initial_validation: Mapping[str, Any],
    initial_condition_response: Mapping[str, Any],
) -> None:
    _atomic_torch_save(
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": STATE_ARTIFACT_KIND,
            "experimental_only": True,
            "formal_release_eligible": False,
            "variant": variant,
            "pair_contract_sha256": pair["sha256"],
            "config_contract_sha256": pair["config_contract_sha256"],
            "foundation_checkpoint_sha256": pair["foundation"]["sha256"],
            "condition_cache_sha256": pair["branches"][variant][
                "bridge_cache"
            ]["sha256"],
            "step": int(step),
            "target_steps": int(config["training"]["steps"]),
            "model_state_dict": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "ema_state_dict": {
                name: value.detach().cpu()
                for name, value in ema.shadow.items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "sampler_state_dict": sampler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
            "python_rng_state": random.getstate(),
            "initial_validation": deepcopy(dict(initial_validation)),
            "initial_condition_response": deepcopy(
                dict(initial_condition_response)
            ),
        },
        path,
    )


def _load_resume_state(
    path: Path,
    *,
    variant: str,
    pair: Mapping[str, Any],
    config: Mapping[str, Any],
    model: torch.nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    sampler: NativeLengthBucketSampler,
    device: torch.device,
) -> tuple[int, dict, dict]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (
        state.get("artifact_kind") != STATE_ARTIFACT_KIND
        or state.get("variant") != variant
        or state.get("pair_contract_sha256") != pair["sha256"]
        or state.get("config_contract_sha256")
        != pair["config_contract_sha256"]
        or state.get("foundation_checkpoint_sha256")
        != pair["foundation"]["sha256"]
        or state.get("condition_cache_sha256")
        != pair["branches"][variant]["bridge_cache"]["sha256"]
        or int(state.get("target_steps", -1))
        != int(config["training"]["steps"])
    ):
        raise ValueError(f"{variant} resume state contract changed")
    step = int(state.get("step", -1))
    if not 1 <= step <= int(config["training"]["steps"]):
        raise ValueError(f"{variant} resume step is invalid")
    model.load_state_dict(state["model_state_dict"], strict=True)
    ema.shadow = {
        name: value.to(device)
        for name, value in state["ema_state_dict"].items()
    }
    optimizer.load_state_dict(state["optimizer_state_dict"])
    _move_optimizer_state(optimizer, device=device)
    sampler.load_state_dict(state["sampler_state_dict"])
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state.get("cuda_rng_state_all"):
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    random.setstate(state["python_rng_state"])
    return (
        step,
        dict(state["initial_validation"]),
        dict(state["initial_condition_response"]),
    )


def _evaluate(
    model: torch.nn.Module,
    episodes: Sequence[Mapping[str, Any]],
    *,
    foundation_checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    conditioning_policy: str,
) -> dict:
    return evaluate_posttrain(
        model,
        episodes,
        action_stats=foundation_checkpoint["action_stats"],
        frame_count=max(config["training"]["batching"]["length_buckets"]),
        batch_size=int(config["training"]["batch_size"]),
        device=device,
        loss_weights=config["training"]["loss"],
        teacher_model=None,
        seed=int(seed),
        batching=config["training"]["batching"],
        conditioning_policy=conditioning_policy,
    )


def _parameter_delta_receipt(
    final_state: Mapping[str, torch.Tensor],
    foundation_state: Mapping[str, torch.Tensor],
) -> dict:
    latent_delta = (
        final_state[MOTION_LATENT_WEIGHT_NAME].detach().cpu()
        - foundation_state[MOTION_LATENT_WEIGHT_NAME].detach().cpu()
    )
    planner_delta = (
        final_state[PLAN_WEIGHT_NAME]
        .detach()
        .cpu()[:, KIMODO_CONDITION_DIM:]
        - foundation_state[PLAN_WEIGHT_NAME]
        .detach()
        .cpu()[:, KIMODO_CONDITION_DIM:]
    )
    return {
        "motion_latent_first_layer_delta_rms": float(
            latent_delta.square().mean().sqrt()
        ),
        "planner_latent_columns_delta_rms": float(
            planner_delta.square().mean().sqrt()
        ),
        "motion_latent_first_layer_changed": bool(
            torch.any(latent_delta != 0)
        ),
        "planner_latent_columns_changed": bool(
            torch.any(planner_delta != 0)
        ),
    }


def train_variant(
    config: Mapping[str, Any],
    *,
    variant: str,
    smoke_test: bool,
    overwrite: bool,
    resume: bool,
) -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"unknown experimental posttrain variant: {variant}")
    pair = validate_pair_contract(config, smoke_test=smoke_test)
    paths = _branch_paths(config, variant)
    if paths["checkpoint"].is_file() and paths["summary"].is_file() and not overwrite:
        payload = torch.load(
            paths["checkpoint"], map_location="cpu", weights_only=True
        )
        _validate_experimental_checkpoint_payload(
            payload, pair=pair, variant=variant
        )
        summary = _load_json_mapping(
            paths["summary"], context=f"{variant} training summary"
        )
        if (
            summary.get("checkpoint_sha256")
            != sha256_file(paths["checkpoint"])
            or summary.get("pair_contract_sha256") != pair["sha256"]
            or summary.get("completed_steps") != config["training"]["steps"]
        ):
            raise ValueError(f"{variant} existing summary/checkpoint changed")
        return summary
    if overwrite:
        for name in ("checkpoint", "summary", "state", "progress"):
            if paths[name].is_file():
                paths[name].unlink()
    paths["root"].mkdir(parents=True, exist_ok=True)
    if (
        paths["progress"].is_file()
        and paths["progress"].stat().st_size > 0
        and not paths["state"].is_file()
    ):
        raise ValueError(
            f"{variant} has progress but no exact resume state; use --overwrite "
            "to restart explicitly"
        )

    splits, split_contract, bridge_metadata = _load_branch_episodes(
        config, pair, variant=variant
    )
    validation_subset = _stable_subset(
        splits["validation"],
        count=int(config["training"]["evaluation_episode_count"]),
        seed=int(config["seed"]),
        label="validation",
    )
    test_subset = _stable_subset(
        splits["test"],
        count=int(config["training"]["evaluation_episode_count"]),
        seed=int(config["seed"]),
        label="test",
    )
    diagnostic_subset = _diverse_diagnostic_subset(
        splits["validation"],
        count=max(
            2, min(8, int(config["training"]["evaluation_episode_count"]))
        ),
        seed=int(config["seed"]),
    )
    evaluation_ids = {
        "validation": [str(row["clip_id"]) for row in validation_subset],
        "test": [str(row["clip_id"]) for row in test_subset],
    }
    evaluation_ids_sha256 = hashlib.sha256(
        _canonical_json(evaluation_ids)
    ).hexdigest()

    _seed_everything(int(config["seed"]))
    device = _resolve_device(config["device"])
    model, foundation_checkpoint = load_contract_checkpoint(
        pair["foundation"]["path"],
        expected_action_dim=ACTION_DIM,
        device=device,
    )
    if sha256_file(pair["foundation"]["path"]) != pair["foundation"]["sha256"]:
        raise ValueError("foundation changed while starting a branch")
    validate_motion_only_checkpoint_isolation(foundation_checkpoint)
    foundation_state = foundation_checkpoint["model_state_dict"]
    foundation_state_sha256 = _state_dict_sha256(foundation_state)
    if _state_dict_sha256(model.state_dict()) != foundation_state_sha256:
        raise ValueError("branch did not start from the exact foundation state")
    model.train()
    optimizer, _, trainable_receipt = _configure_condition_path_optimizer(
        model, config, device=device
    )
    optimized_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    ema = ModelEMA(model, float(config["training"]["ema_decay"]))
    core_config = _training_config_for_core(config)
    sampler = _sampler_for_config(
        splits["train"], core_config, seed=int(config["seed"]) + 17
    )
    if not isinstance(sampler, NativeLengthBucketSampler):
        raise ValueError("branch sampler is not native variable-length")
    observed_sampler_probe = _sampler_schedule_probe(splits["train"], config)
    if (
        observed_sampler_probe["sha256"]
        != pair["sampler_initial_schedule_probe"]["sha256"]
    ):
        raise ValueError("branch sampler initial schedule differs from its pair")
    baseline_zero_output, zero_frame_valid = _zero_latent_probe(
        model,
        validation_subset,
        action_stats=foundation_checkpoint["action_stats"],
        device=device,
        batching=config["training"]["batching"],
    )

    global_step = 0
    initial_validation = None
    initial_condition_response = None
    if resume and paths["state"].is_file():
        (
            global_step,
            initial_validation,
            initial_condition_response,
        ) = _load_resume_state(
            paths["state"],
            variant=variant,
            pair=pair,
            config=config,
            model=model,
            ema=ema,
            optimizer=optimizer,
            sampler=sampler,
            device=device,
        )
        _restore_ema_preserved_state(ema, foundation_state)
    elif paths["state"].is_file():
        raise FileExistsError(
            f"{variant} resume state exists; pass --resume or --overwrite"
        )
    else:
        paths["progress"].write_text("", encoding="utf-8")
        initial_validation = _evaluate(
            model,
            validation_subset,
            foundation_checkpoint=foundation_checkpoint,
            config=config,
            device=device,
            seed=int(config["seed"]) + 1_000_003,
            conditioning_policy=(
                "experimental_metadata_128d_plus_oracle_trajectory_style"
            ),
        )
        initial_condition_response = _condition_response_diagnostics(
            model,
            diagnostic_subset,
            action_stats=foundation_checkpoint["action_stats"],
            device=device,
            batching=config["training"]["batching"],
        )
        _append_jsonl(
            {
                "event": "foundation_step0",
                "step": 0,
                "foundation_model_state_sha256": foundation_state_sha256,
                "exact_foundation_state": True,
                "validation": initial_validation,
                "condition_response": initial_condition_response,
            },
            paths["progress"],
        )

    training = config["training"]
    target_steps = int(training["steps"])
    if global_step > target_steps:
        raise ValueError(f"{variant} resume step exceeds target")
    started = time.monotonic()
    last_train = {}
    last_validation = dict(initial_validation)
    last_grad_norm = 0.0
    for step in range(global_step + 1, target_steps + 1):
        scale = _lr_scale(
            step,
            total_steps=target_steps,
            warmup_steps=int(training["warmup_steps"]),
            minimum_ratio=float(training["minimum_lr_ratio"]),
        )
        current_lr = float(training["lr"]) * scale
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.zero_grad(set_to_none=True)
        remaining = int(
            training["batching"]["target_effective_batch_size"]
        )
        accumulated = defaultdict(float)
        sampled_clip_ids = []
        plans = []
        while remaining > 0:
            microbatch, plan = sampler.sample_microbatch(
                remaining_effective_batch=remaining,
                semantic_tokens=int(model.semantic_tokens),
                max_batch_size=int(training["batch_size"]),
                batching=training["batching"],
            )
            (
                actions,
                conditions,
                dim_masks,
                durations,
                frame_valid,
            ) = _batch_tensors_for_config(
                microbatch,
                frame_count=int(plan["bucket_frames"]),
                action_stats=foundation_checkpoint["action_stats"],
                device=device,
                batching=training["batching"],
            )
            losses = masked_18d_objective(
                model,
                actions,
                conditions,
                dim_masks,
                durations,
                loss_weights=training["loss"],
                teacher_model=None,
                frame_valid_mask=frame_valid,
            )
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(
                    f"{variant} non-finite loss at step {step}"
                )
            sample_weight = len(microbatch) / float(
                training["batching"]["target_effective_batch_size"]
            )
            (losses["total"] * sample_weight).backward()
            for name, value in losses.items():
                accumulated[name] += float(value.detach().cpu()) * sample_weight
            sampled_clip_ids.extend(str(row["clip_id"]) for row in microbatch)
            plans.append(dict(plan))
            remaining -= len(microbatch)
        last_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                optimized_parameters, float(training["max_grad_norm"])
            )
        )
        optimizer.step()
        ema.update(model)
        _restore_ema_preserved_state(ema, foundation_state)
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
            "lr": current_lr,
            "grad_norm": last_grad_norm,
            "train": last_train,
            "sampled_clip_ids_sha256": hashlib.sha256(
                _canonical_json(sampled_clip_ids)
            ).hexdigest(),
            "microbatch_plans": plans,
        }
        if should_validate:
            with ema.apply(model):
                last_validation = _evaluate(
                    model,
                    validation_subset,
                    foundation_checkpoint=foundation_checkpoint,
                    config=config,
                    device=device,
                    seed=int(config["seed"]) + 1_000_003,
                    conditioning_policy=(
                        "experimental_metadata_128d_plus_oracle_trajectory_style"
                    ),
                )
            event["validation"] = last_validation
        if (
            step == 1
            or step % int(training["log_interval"]) == 0
            or should_validate
        ):
            print(json.dumps(event, sort_keys=True), flush=True)
            _append_jsonl(event, paths["progress"])
        if should_checkpoint:
            _save_resume_state(
                paths["state"],
                variant=variant,
                pair=pair,
                config=config,
                step=step,
                model=model,
                ema=ema,
                optimizer=optimizer,
                sampler=sampler,
                initial_validation=initial_validation,
                initial_condition_response=initial_condition_response,
            )

    final_state = {
        name: value.detach().cpu().clone()
        for name, value in ema.shadow.items()
    }
    model.load_state_dict(final_state, strict=True)
    final_validation = _evaluate(
        model,
        validation_subset,
        foundation_checkpoint=foundation_checkpoint,
        config=config,
        device=device,
        seed=int(config["seed"]) + 1_000_003,
        conditioning_policy=(
            "experimental_metadata_128d_plus_oracle_trajectory_style"
        ),
    )
    test_metrics = _evaluate(
        model,
        test_subset,
        foundation_checkpoint=foundation_checkpoint,
        config=config,
        device=device,
        seed=int(config["seed"]) + 2_000_003,
        conditioning_policy=(
            "experimental_metadata_128d_plus_oracle_trajectory_style"
        ),
    )
    final_condition_response = _condition_response_diagnostics(
        model,
        diagnostic_subset,
        action_stats=foundation_checkpoint["action_stats"],
        device=device,
        batching=training["batching"],
    )
    final_zero_output, final_zero_mask = _zero_latent_probe(
        model,
        validation_subset,
        action_stats=foundation_checkpoint["action_stats"],
        device=device,
        batching=training["batching"],
    )
    if not torch.equal(zero_frame_valid, final_zero_mask):
        raise RuntimeError("zero-latent probe mask changed")
    preservation = _preservation_audit(
        final_state=final_state,
        foundation_state=foundation_state,
        baseline_zero_output=baseline_zero_output,
        final_zero_output=final_zero_output,
        frame_valid=zero_frame_valid,
    )
    if not preservation["zero_latent_exact_equivalence_passed"]:
        raise RuntimeError(f"{variant} violated zero-latent foundation preservation")
    if final_condition_response["aligned_vs_zero_prediction_rms"] <= 0:
        raise RuntimeError(f"{variant} has no nonzero latent-condition response")
    if sha256_file(pair["foundation"]["path"]) != pair["foundation"]["sha256"]:
        preservation["foundation_file_unchanged"] = False
        raise RuntimeError("foundation file changed during experimental posttrain")
    parameter_delta = _parameter_delta_receipt(final_state, foundation_state)
    foundation_receipt = {
        "path": pair["foundation"]["path"],
        "checkpoint_sha256": pair["foundation"]["sha256"],
        "architecture": pair["foundation"]["architecture"],
        "formal_episode_contract": pair["foundation"][
            "formal_episode_contract"
        ],
        "data_isolation_contract_sha256": pair["foundation"][
            "data_isolation_contract_sha256"
        ],
        "split_contract_sha256": pair["foundation"][
            "split_contract_sha256"
        ],
        "style_contract_sha256": pair["foundation"][
            "style_contract_sha256"
        ],
        "random_initialization": deepcopy(
            foundation_checkpoint["random_initialization"]
        ),
        "sources": deepcopy(foundation_checkpoint["sources"]),
    }
    condition_receipt = {
        "path": pair["branches"][variant]["bridge_cache"]["path"],
        "cache_sha256": pair["branches"][variant]["bridge_cache"]["sha256"],
        "artifact_kind": BRIDGE_CACHE_ARTIFACT_KIND,
        "variant": variant,
        "source_128d_cache_sha256": pair["branches"][variant][
            "source_128d_cache"
        ]["sha256"],
        "style_cache_sha256": pair["style_cache"]["sha256"],
        "source_manifest_sha256": pair["manifest"]["sha256"],
        "identity_arrays_sha256": pair["identity_arrays_sha256"],
        "no_kimodo": True,
        "semantic_scope": SEMANTIC_SCOPE_ACKNOWLEDGEMENT,
    }
    checkpoint_payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": CHECKPOINT_ARTIFACT_KIND,
        "experimental_only": True,
        "formal_release_eligible": False,
        "semantic_scope": SEMANTIC_SCOPE_ACKNOWLEDGEMENT,
        "semantic_supervision_status": (
            "official_metadata_alignment_only_not_formal_robot_semantics"
        ),
        "variant": variant,
        "architecture": foundation_checkpoint["architecture"],
        "action_dim": ACTION_DIM,
        "condition_dim": KIMODO_V2_CONDITION_DIM,
        "joint_order": deepcopy(foundation_checkpoint["joint_order"]),
        "config": {
            "hidden_dim": int(
                (foundation_checkpoint.get("config") or {}).get(
                    "hidden_dim", 384
                )
            ),
            "layers": int(
                (foundation_checkpoint.get("config") or {}).get("layers", 6)
            ),
            "semantic_tokens": int(
                (foundation_checkpoint.get("config") or {}).get(
                    "semantic_tokens", 7
                )
            ),
        },
        "model_state_dict": final_state,
        "action_stats": deepcopy(foundation_checkpoint["action_stats"]),
        "global_step": int(foundation_checkpoint.get("global_step", 0))
        + target_steps,
        "foundation_receipt": foundation_receipt,
        "condition_cache_receipt": condition_receipt,
        "pair_contract": {
            "path": str(_prepared_pair_path(config).resolve()),
            "sha256": pair["sha256"],
        },
        "training_contract": {
            **trainable_receipt,
            "seed": int(config["seed"]),
            "steps": target_steps,
            "optimizer": deepcopy(pair["optimization_contract"]["optimizer"]),
            "loss": deepcopy(training["loss"]),
            "batching": deepcopy(training["batching"]),
            "sampler": deepcopy(training["sampler"]),
            "sampler_initial_state_sha256": pair[
                "sampler_initial_schedule_probe"
            ]["sha256"],
            "fixed_equal_step_budget": True,
            "early_stopping": "disabled",
            "replay_episode_count": 0,
            "external_motion_checkpoint_count": 0,
        },
        "split_contract": {
            "sha256": split_contract["sha256"],
            "counts": {name: len(rows) for name, rows in splits.items()},
            "evaluation_ids_sha256": evaluation_ids_sha256,
        },
        "metrics": {
            "initial_validation": initial_validation,
            "final_validation": final_validation,
            "test": test_metrics,
            "initial_condition_response": initial_condition_response,
            "final_condition_response": final_condition_response,
        },
        "preservation": preservation,
        "parameter_delta": parameter_delta,
    }
    _validate_experimental_checkpoint_payload(
        checkpoint_payload, pair=pair, variant=variant
    )
    _atomic_torch_save(checkpoint_payload, paths["checkpoint"])
    checkpoint_sha256 = sha256_file(paths["checkpoint"])
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": SUMMARY_ARTIFACT_KIND,
        "experimental_only": True,
        "formal_release_eligible": False,
        "semantic_scope": SEMANTIC_SCOPE_ACKNOWLEDGEMENT,
        "variant": variant,
        "checkpoint": str(paths["checkpoint"].resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "completed_steps": target_steps,
        "target_steps": target_steps,
        "stopped_early": False,
        "resumable_state": str(paths["state"].resolve()),
        "pair_contract_sha256": pair["sha256"],
        "foundation_checkpoint_sha256": pair["foundation"]["sha256"],
        "condition_cache_sha256": condition_receipt["cache_sha256"],
        "source_128d_cache_sha256": condition_receipt[
            "source_128d_cache_sha256"
        ],
        "evaluation_ids_sha256": evaluation_ids_sha256,
        "initial_validation": initial_validation,
        "final_validation": final_validation,
        "test": test_metrics,
        "metrics_delta": _numeric_delta(initial_validation, final_validation),
        "initial_condition_response": initial_condition_response,
        "final_condition_response": final_condition_response,
        "preservation": preservation,
        "parameter_delta": parameter_delta,
        "training_policy": trainable_receipt,
        "last_train": last_train,
        "last_grad_norm": last_grad_norm,
        "elapsed_seconds_this_invocation": time.monotonic() - started,
    }
    _atomic_json_save(summary, paths["summary"])
    return summary


def _load_experimental_model(
    config: Mapping[str, Any],
    pair: Mapping[str, Any],
    *,
    variant: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict, dict]:
    paths = _branch_paths(config, variant)
    if not paths["checkpoint"].is_file() or not paths["summary"].is_file():
        raise FileNotFoundError(f"{variant} experimental branch is incomplete")
    payload = torch.load(
        paths["checkpoint"], map_location="cpu", weights_only=True
    )
    _validate_experimental_checkpoint_payload(
        payload, pair=pair, variant=variant
    )
    summary = _load_json_mapping(
        paths["summary"], context=f"{variant} training summary"
    )
    if (
        summary.get("checkpoint_sha256") != sha256_file(paths["checkpoint"])
        or summary.get("variant") != variant
        or summary.get("pair_contract_sha256") != pair["sha256"]
        or summary.get("completed_steps")
        != pair["optimization_contract"]["steps"]
        or summary.get("stopped_early") is not False
    ):
        raise ValueError(f"{variant} summary/checkpoint receipt changed")
    model, _ = load_contract_checkpoint(
        pair["foundation"]["path"],
        expected_action_dim=ACTION_DIM,
        device=device,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload, summary


def _cross_swap_diagnostics(
    frozen_model: torch.nn.Module,
    lora_model: torch.nn.Module,
    frozen_rows: Sequence[Mapping[str, Any]],
    lora_rows: Sequence[Mapping[str, Any]],
    *,
    action_stats: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
) -> dict:
    if [row["clip_id"] for row in frozen_rows] != [
        row["clip_id"] for row in lora_rows
    ]:
        raise ValueError("cross-swap rows differ in clip order")
    totals = defaultdict(float)
    observed_values = 0
    with torch.no_grad():
        for frozen_row, lora_row in zip(
            frozen_rows, lora_rows, strict=True
        ):
            actions, frozen_condition, _, _, frame_valid = (
                _batch_tensors_for_config(
                    [frozen_row],
                    frame_count=max(
                        config["training"]["batching"]["length_buckets"]
                    ),
                    action_stats=action_stats,
                    device=device,
                    batching=config["training"]["batching"],
                )
            )
            _, lora_condition, _, _, _ = _batch_tensors_for_config(
                [lora_row],
                frame_count=max(
                    config["training"]["batching"]["length_buckets"]
                ),
                action_stats=action_stats,
                device=device,
                batching=config["training"]["batching"],
            )
            if not torch.equal(
                frozen_condition[:, :KIMODO_CONDITION_DIM],
                lora_condition[:, :KIMODO_CONDITION_DIM],
            ):
                raise ValueError("cross-swap trajectory style/base condition changed")
            t = torch.full((1,), 0.5, dtype=actions.dtype, device=device)
            x_t = actions * 0.375
            frozen_native = forward_with_frame_mask(
                frozen_model, x_t, t, frozen_condition, frame_valid
            )
            frozen_cross = forward_with_frame_mask(
                frozen_model, x_t, t, lora_condition, frame_valid
            )
            lora_native = forward_with_frame_mask(
                lora_model, x_t, t, lora_condition, frame_valid
            )
            lora_cross = forward_with_frame_mask(
                lora_model, x_t, t, frozen_condition, frame_valid
            )
            mask = frame_valid[:, :, None].expand_as(frozen_native)
            pairs = {
                "frozen_model_native_vs_lora_cache": (
                    frozen_native,
                    frozen_cross,
                ),
                "lora_model_native_vs_frozen_cache": (
                    lora_native,
                    lora_cross,
                ),
                "native_branch_output_difference": (
                    frozen_native,
                    lora_native,
                ),
                "crossed_branch_output_difference": (
                    frozen_cross,
                    lora_cross,
                ),
            }
            for name, (left, right) in pairs.items():
                totals[name] += float(
                    (left - right).square().masked_select(mask).sum().cpu()
                )
            observed_values += int(mask.sum().cpu())
    result = {
        f"{name}_rms": math.sqrt(value / max(1, observed_values))
        for name, value in totals.items()
    }
    result.update(
        {
            "episode_count": len(frozen_rows),
            "diagnostic_scope": (
                "fixed-x_t cache cross-swap mechanism diagnostic; not formal "
                "semantic-quality evidence"
            ),
        }
    )
    return result


def compare_variants(
    config: Mapping[str, Any], *, smoke_test: bool
) -> dict:
    pair = validate_pair_contract(config, smoke_test=smoke_test)
    device = _resolve_device(config["device"])
    frozen_model, frozen_payload, frozen_summary = _load_experimental_model(
        config, pair, variant="frozen_base", device=device
    )
    lora_model, lora_payload, lora_summary = _load_experimental_model(
        config, pair, variant="lora_finetuned", device=device
    )
    controlled_fields = (
        "policy",
        "full_network",
        "seed",
        "steps",
        "optimizer",
        "loss",
        "batching",
        "sampler",
        "sampler_initial_state_sha256",
        "fixed_equal_step_budget",
        "early_stopping",
        "replay_episode_count",
        "external_motion_checkpoint_count",
        "effective_trainable_parameter_names",
        "effective_trainable_parameter_count",
    )
    frozen_training = frozen_payload["training_contract"]
    lora_training = lora_payload["training_contract"]
    if {
        field: frozen_training.get(field) for field in controlled_fields
    } != {field: lora_training.get(field) for field in controlled_fields}:
        raise ValueError("B/C generator optimization contracts differ")
    if (
        frozen_payload["foundation_receipt"]["checkpoint_sha256"]
        != lora_payload["foundation_receipt"]["checkpoint_sha256"]
        or frozen_payload["split_contract"]
        != lora_payload["split_contract"]
        or frozen_summary["evaluation_ids_sha256"]
        != lora_summary["evaluation_ids_sha256"]
        or frozen_payload["condition_cache_receipt"][
            "identity_arrays_sha256"
        ]
        != lora_payload["condition_cache_receipt"][
            "identity_arrays_sha256"
        ]
        or frozen_payload["condition_cache_receipt"]["style_cache_sha256"]
        != lora_payload["condition_cache_receipt"]["style_cache_sha256"]
    ):
        raise ValueError("B/C generator controlled inputs differ")

    frozen_splits, _, _ = _load_branch_episodes(
        config, pair, variant="frozen_base"
    )
    lora_splits, _, _ = _load_branch_episodes(
        config, pair, variant="lora_finetuned"
    )
    diagnostic_count = max(
        2, min(8, int(config["training"]["evaluation_episode_count"]))
    )
    frozen_test = _diverse_diagnostic_subset(
        frozen_splits["test"],
        count=diagnostic_count,
        seed=int(config["seed"]) + 99,
    )
    lora_by_clip = {
        str(row["clip_id"]): row for row in lora_splits["test"]
    }
    lora_test = [lora_by_clip[str(row["clip_id"])] for row in frozen_test]
    foundation_checkpoint = torch.load(
        pair["foundation"]["path"], map_location="cpu", weights_only=True
    )
    cross_swap = _cross_swap_diagnostics(
        frozen_model,
        lora_model,
        frozen_test,
        lora_test,
        action_stats=foundation_checkpoint["action_stats"],
        config=config,
        device=device,
    )
    comparison = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": COMPARISON_ARTIFACT_KIND,
        "experimental_only": True,
        "formal_release_eligible": False,
        "semantic_scope": SEMANTIC_SCOPE_ACKNOWLEDGEMENT,
        "semantic_supervision_status": (
            "official_metadata_alignment_only_not_formal_robot_semantics"
        ),
        "question": (
            "Does the BEAT2-only Qwen LoRA cache improve the same clean "
            "generator foundation under an otherwise identical posttrain?"
        ),
        "pair_contract": {
            "path": str(_prepared_pair_path(config).resolve()),
            "sha256": pair["sha256"],
        },
        "controlled_variables": {
            "foundation_checkpoint_sha256": pair["foundation"]["sha256"],
            "style_cache_sha256": pair["style_cache"]["sha256"],
            "identity_arrays_sha256": pair["identity_arrays_sha256"],
            "fixed_split_contract_sha256": pair[
                "selected_split_contract_sha256"
            ],
            "optimization_contract": deepcopy(
                pair["optimization_contract"]
            ),
            "trainable_parameter_strategy": TRAINING_POLICY,
            "full_network": False,
            "zero_latent_exact_A_preservation_required": True,
        },
        "independent_variable": {
            "frozen_base": {
                "source_128d_cache_sha256": pair["branches"][
                    "frozen_base"
                ]["source_128d_cache"]["sha256"],
                "bridge_cache_sha256": pair["branches"]["frozen_base"][
                    "bridge_cache"
                ]["sha256"],
            },
            "lora_finetuned": {
                "source_128d_cache_sha256": pair["branches"][
                    "lora_finetuned"
                ]["source_128d_cache"]["sha256"],
                "bridge_cache_sha256": pair["branches"][
                    "lora_finetuned"
                ]["bridge_cache"]["sha256"],
            },
        },
        "frozen_base": frozen_summary,
        "lora_finetuned": lora_summary,
        "delta_lora_minus_frozen": {
            "initial_validation": _numeric_delta(
                frozen_summary["initial_validation"],
                lora_summary["initial_validation"],
            ),
            "final_validation": _numeric_delta(
                frozen_summary["final_validation"],
                lora_summary["final_validation"],
            ),
            "test": _numeric_delta(
                frozen_summary["test"], lora_summary["test"]
            ),
            "condition_response": _numeric_delta(
                frozen_summary["final_condition_response"],
                lora_summary["final_condition_response"],
            ),
        },
        "cross_swap_diagnostic": cross_swap,
        "interpretation_limits": [
            "The text conditions are derived from 54 canonical metadata prompts.",
            "The source release masks formal prompt supervision.",
            "Trajectory style is oracle-derived and identical across branches.",
            "This comparison is experimental and cannot be promoted to a formal semantic release.",
        ],
    }
    output_path = Path(config["output_dir"]) / "comparison.json"
    _atomic_json_save(comparison, output_path)
    return comparison


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("prepare", "frozen", "lora", "compare", "all"),
        default="prepare",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="exact-resume an existing per-branch last_state.pt (default: true)",
    )
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--foundation-checkpoint", type=Path)
    parser.add_argument("--foundation-training-summary", type=Path)
    parser.add_argument("--style-condition-cache", type=Path)
    parser.add_argument("--frozen-condition-cache", type=Path)
    parser.add_argument("--lora-condition-cache", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = effective_config(
        read_config(args.config),
        smoke_test=args.smoke_test,
        device=args.device,
        output_dir=args.output_dir,
        foundation_checkpoint=args.foundation_checkpoint,
        foundation_training_summary=args.foundation_training_summary,
        style_condition_cache=args.style_condition_cache,
        frozen_condition_cache=args.frozen_condition_cache,
        lora_condition_cache=args.lora_condition_cache,
    )
    result = None
    if args.stage in {"prepare", "all"}:
        result = prepare_pair(
            config,
            smoke_test=args.smoke_test,
            overwrite=args.overwrite,
        )
    if args.stage in {"frozen", "all"}:
        result = train_variant(
            config,
            variant="frozen_base",
            smoke_test=args.smoke_test,
            overwrite=args.overwrite,
            resume=args.resume,
        )
    if args.stage in {"lora", "all"}:
        result = train_variant(
            config,
            variant="lora_finetuned",
            smoke_test=args.smoke_test,
            overwrite=args.overwrite,
            resume=args.resume,
        )
    if args.stage in {"compare", "all"}:
        result = compare_variants(config, smoke_test=args.smoke_test)
    print(
        json.dumps(
            {
                "stage": args.stage,
                "smoke_test": bool(args.smoke_test),
                "artifact_kind": (result or {}).get("artifact_kind"),
                "output_dir": config["output_dir"],
                "formal_release_eligible": (result or {}).get(
                    "formal_release_eligible"
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
