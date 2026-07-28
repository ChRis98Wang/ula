#!/usr/bin/env python3
"""Build strict BEAT2-only A/B/C evaluation videos for the clean AdaLN model.

This is an experimental evaluation tool.  It intentionally does not import,
modify, or relax the formal long-video publication contract.

The three arms are:

* A: clean trajectory-style-only condition.
* B: the same style condition plus the frozen-Qwen 128D text-to-motion latent.
* C: the same style condition plus the BEAT2-only Qwen-LoRA 128D latent.

Every arm uses the same held-out BEAT2 clip, native frame count, sampling seed,
sampling steps, playback transform, and renderer settings.  Missing or
unverifiable inputs are fatal.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import imageio_ffmpeg
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from upper_body_skeleton.retarget_v2_18d import JOINT_ORDER_18D
from upper_body_skeleton.ula_training import (
    KIMODO_MOTION_LATENT_DIM,
    KIMODO_V2_CONDITION_DIM,
    ULA_MMDIT_V3_ADALN_ARCHITECTURE,
    create_ula_model,
    sample_trajectory,
    write_generated_csv,
)


SCHEMA_VERSION = 1
ARTIFACT_KIND = "beat2_clean_adaln_abc_experimental_video_v1"
CONFIG_ARTIFACT_KIND = "beat2_clean_adaln_abc_video_config_v1"
CHECKPOINT_ARTIFACT_KIND = "ula_mmdit_v2_generator"
ACTION_DIM = 18
CONDITION_DIM = 264
STYLE_SLICE = slice(133, 136)
LATENT_SLICE = slice(136, 264)
EXPECTED_HIDDEN_DIM = 384
EXPECTED_LAYERS = 6
EXPECTED_SEMANTIC_TOKENS = 7
EXPECTED_INITIALIZATION_MODE = "full_generator_random_no_qwen_no_kimodo_v1"
EXPECTED_QWEN_DISABLED_POLICY = "disabled_not_configured_not_loaded_v1"
EXPECTED_KIMODO_POLICY = (
    "forbidden_dataset_checkpoint_replay_and_condition_channels_v1"
)
EXPECTED_STYLE_POLICY = (
    "trajectory_style_indices_133_136_only_all_other_dimensions_exact_zero_v1"
)
EXPECTED_QWEN_CACHE_KIND = "beat2_qwen_motion_latent_condition_cache_v1"
EXPECTED_EXPERIMENTAL_CHECKPOINT_KIND = (
    "beat2_experimental_metadata_conditioned_posttrain_v1"
)
EXPECTED_EXPERIMENTAL_SUMMARY_KIND = (
    "beat2_experimental_metadata_conditioned_posttrain_training_summary_v1"
)
EXPECTED_EXPERIMENTAL_264D_CACHE_KIND = (
    "beat2_experimental_metadata_264d_condition_cache_v1"
)
EXPECTED_EXPERIMENTAL_TRAINING_POLICY = (
    "zero_latent_preserving_condition_path_only_v1"
)
MOTION_LATENT_WEIGHT_NAME = "motion_latent_condition.0.weight"
PLAN_WEIGHT_NAME = "plan.0.weight"
EXPECTED_TRAINABLE_TENSOR_NAMES = [
    MOTION_LATENT_WEIGHT_NAME,
    PLAN_WEIGHT_NAME,
]
EXPECTED_EFFECTIVE_TRAINABLE_PARAMETER_NAMES = [
    MOTION_LATENT_WEIGHT_NAME,
    f"{PLAN_WEIGHT_NAME}[:,136:264]",
]
EXPECTED_QWEN_SCOPE = (
    "experimental_official_metadata_alignment_only_not_formal_generator_supervision"
)
HEAD_INDICES = tuple(range(15, 18))
SHA256_RE_LENGTH = 64


class EvaluationContractError(RuntimeError):
    """Raised when the experimental evaluation cannot prove its inputs."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def value_sha256(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def self_hashed_mapping_sha256(value: Mapping[str, Any]) -> str:
    """Return the producer-compatible hash for a self-hashed JSON object."""
    payload = dict(value)
    payload.pop("sha256", None)
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def validate_self_hashed_json(
    path: str | Path, *, expected_sha256: str, field: str
) -> str:
    """Validate the embedded canonical hash, not formatting-dependent file bytes."""
    expected_sha256 = _require_sha(expected_sha256, f"{field}.expected_sha256")
    payload = load_json(path)
    embedded_sha256 = _require_sha(payload.get("sha256"), f"{field}.sha256")
    calculated_sha256 = self_hashed_mapping_sha256(payload)
    if (
        embedded_sha256 != calculated_sha256
        or embedded_sha256 != expected_sha256
    ):
        raise EvaluationContractError(f"{field} SHA mismatch")
    return embedded_sha256


def validate_experimental_training_contract(
    training_contract: Any,
    model_state: Any,
    *,
    branch: str,
) -> dict[str, Any]:
    """Validate the producer's condition-path-only optimization receipt."""
    if not isinstance(training_contract, Mapping):
        raise EvaluationContractError(
            f"branch {branch} training_contract is missing"
        )
    if not isinstance(model_state, Mapping):
        raise EvaluationContractError(
            f"branch {branch} model_state_dict is missing"
        )
    if (
        training_contract.get("policy")
        != EXPECTED_EXPERIMENTAL_TRAINING_POLICY
        or training_contract.get("full_network") is not False
        or training_contract.get("trainable_tensor_names")
        != EXPECTED_TRAINABLE_TENSOR_NAMES
        or training_contract.get("effective_trainable_parameter_names")
        != EXPECTED_EFFECTIVE_TRAINABLE_PARAMETER_NAMES
    ):
        raise EvaluationContractError(
            f"branch {branch} has no auditable trainable condition path"
        )
    latent_weight = model_state.get(MOTION_LATENT_WEIGHT_NAME)
    plan_weight = model_state.get(PLAN_WEIGHT_NAME)
    if (
        not isinstance(latent_weight, torch.Tensor)
        or not isinstance(plan_weight, torch.Tensor)
        or latent_weight.ndim != 2
        or plan_weight.ndim != 2
        or latent_weight.shape[1] != KIMODO_MOTION_LATENT_DIM
        or plan_weight.shape[1] != CONDITION_DIM
    ):
        raise EvaluationContractError(
            f"branch {branch} trainable condition tensors are invalid"
        )
    optimizer_parameter_count = latent_weight.numel() + plan_weight.numel()
    effective_parameter_count = (
        latent_weight.numel()
        + plan_weight.shape[0] * KIMODO_MOTION_LATENT_DIM
    )
    plan_gradient_mask = torch.zeros_like(plan_weight, device="cpu")
    plan_gradient_mask[:, LATENT_SLICE] = 1.0
    plan_gradient_mask_sha256 = hashlib.sha256(
        plan_gradient_mask.numpy().tobytes()
    ).hexdigest()
    if (
        training_contract.get("optimizer_parameter_count")
        != optimizer_parameter_count
        or training_contract.get("effective_trainable_parameter_count")
        != effective_parameter_count
        or training_contract.get("plan_gradient_mask_sha256")
        != plan_gradient_mask_sha256
    ):
        raise EvaluationContractError(
            f"branch {branch} trainable condition-path receipt is invalid"
        )
    return {
        "policy": EXPECTED_EXPERIMENTAL_TRAINING_POLICY,
        "optimizer_parameter_count": optimizer_parameter_count,
        "effective_trainable_parameter_count": effective_parameter_count,
        "plan_gradient_mask_sha256": plan_gradient_mask_sha256,
    }


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationContractError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise EvaluationContractError(f"JSON payload must be an object: {path}")
    return value


def _resolve_path(value: Any, *, config_dir: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationContractError(f"{field} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def _require_file(path: Path, field: str) -> Path:
    if not path.is_file():
        raise EvaluationContractError(f"{field} does not exist: {path}")
    return path


def _require_sha(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != SHA256_RE_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EvaluationContractError(f"{field} must be a lowercase SHA256")
    return value


def _checkpoint_sources_are_beat2_only(checkpoint: Mapping[str, Any]) -> None:
    sources = checkpoint.get("sources")
    manifests = sources.get("motion_manifests") if isinstance(sources, Mapping) else None
    if not isinstance(manifests, list) or not manifests:
        raise EvaluationContractError("checkpoint has no hash-bound motion manifests")
    for index, source in enumerate(manifests):
        if not isinstance(source, Mapping):
            raise EvaluationContractError(f"checkpoint motion source {index} is invalid")
        dataset_source = str(source.get("dataset_source", "")).lower()
        if "beat2" not in dataset_source:
            raise EvaluationContractError(
                f"checkpoint motion source {index} is not BEAT2-only"
            )
        if source.get("manifest_fixed_split") is not True:
            raise EvaluationContractError(
                f"checkpoint motion source {index} is not a fixed split"
            )
        _require_sha(
            source.get("manifest_sha256"),
            f"checkpoint.sources.motion_manifests[{index}].manifest_sha256",
        )


def _validate_action_stats(value: Any) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping):
        raise EvaluationContractError("checkpoint action_stats are missing")
    result: dict[str, np.ndarray] = {}
    for name in ("mean", "std"):
        raw = value.get(name)
        if isinstance(raw, torch.Tensor):
            array = raw.detach().cpu().numpy()
        else:
            array = np.asarray(raw, dtype=np.float32)
        if array.shape != (ACTION_DIM,) or not np.isfinite(array).all():
            raise EvaluationContractError(
                f"checkpoint action_stats.{name} must be finite [18]"
            )
        result[name] = array.astype(np.float32, copy=False)
    if np.any(result["std"] <= 0.0):
        raise EvaluationContractError("checkpoint action_stats.std must be positive")
    return result


def validate_checkpoint(
    path: str | Path,
    *,
    expected_manifest_sha256: str,
    branch: str,
    device: str = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load one exact 384x6 AdaLN checkpoint and validate clean lineage."""
    path = _require_file(Path(path), f"branch {branch} checkpoint")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise EvaluationContractError(
            f"cannot load branch {branch} checkpoint {path}: {error}"
        ) from error
    if not isinstance(checkpoint, Mapping):
        raise EvaluationContractError(f"branch {branch} checkpoint is not an object")
    exact = {
        "artifact_kind": CHECKPOINT_ARTIFACT_KIND,
        "architecture": ULA_MMDIT_V3_ADALN_ARCHITECTURE,
        "action_dim": ACTION_DIM,
        "condition_dim": CONDITION_DIM,
        "joint_order": list(JOINT_ORDER_18D),
    }
    mismatches = [
        key for key, expected in exact.items() if checkpoint.get(key) != expected
    ]
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        mismatches.append("config")
        config = {}
    expected_config = {
        "hidden_dim": EXPECTED_HIDDEN_DIM,
        "layers": EXPECTED_LAYERS,
        "semantic_tokens": EXPECTED_SEMANTIC_TOKENS,
        "initialization_mode": EXPECTED_INITIALIZATION_MODE,
    }
    mismatches.extend(
        f"config.{key}"
        for key, expected in expected_config.items()
        if config.get(key) != expected
    )
    if mismatches:
        raise EvaluationContractError(
            f"branch {branch} is not the exact full AdaLN 384x6 contract: "
            f"{sorted(set(mismatches))}"
        )

    random_initialization = checkpoint.get("random_initialization")
    if not isinstance(random_initialization, Mapping):
        raise EvaluationContractError(
            f"branch {branch} random initialization provenance is missing"
        )
    required_init = {
        "mode": EXPECTED_INITIALIZATION_MODE,
        "qwen_policy": EXPECTED_QWEN_DISABLED_POLICY,
        "kimodo_policy": EXPECTED_KIMODO_POLICY,
        "generator_checkpoint_inputs": [],
    }
    bad_init = [
        key
        for key, expected in required_init.items()
        if random_initialization.get(key) != expected
    ]
    if bad_init:
        raise EvaluationContractError(
            f"branch {branch} violates clean no-Kimodo initialization: {bad_init}"
        )
    initialization_state_sha256 = _require_sha(
        random_initialization.get("generator_state_sha256"),
        f"branch {branch} random_initialization.generator_state_sha256",
    )
    if checkpoint.get("unsafe_training_data") not in (None, False, [], {}):
        raise EvaluationContractError(f"branch {branch} used unsafe training data")
    _checkpoint_sources_are_beat2_only(checkpoint)
    source_hashes = {
        source["manifest_sha256"]
        for source in checkpoint["sources"]["motion_manifests"]
    }
    if source_hashes != {expected_manifest_sha256}:
        raise EvaluationContractError(
            f"branch {branch} does not bind only the requested BEAT2 manifest"
        )
    completed_step = checkpoint.get("global_step", checkpoint.get("posttrain_step"))
    if not isinstance(completed_step, int) or completed_step <= 0:
        raise EvaluationContractError(
            f"branch {branch} is not a trained checkpoint (global_step <= 0)"
        )
    action_stats = _validate_action_stats(checkpoint.get("action_stats"))
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise EvaluationContractError(
            f"branch {branch} checkpoint model_state_dict is missing"
        )
    model = create_ula_model(
        ULA_MMDIT_V3_ADALN_ARCHITECTURE,
        action_dim=ACTION_DIM,
        condition_dim=CONDITION_DIM,
        hidden_dim=EXPECTED_HIDDEN_DIM,
        layers=EXPECTED_LAYERS,
        semantic_tokens=EXPECTED_SEMANTIC_TOKENS,
    )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise EvaluationContractError(
            f"branch {branch} model state does not match full AdaLN 384x6: {error}"
        ) from error
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise EvaluationContractError(
            f"branch {branch} checkpoint contains non-finite parameters"
        )
    model.action_stats = {
        name: torch.as_tensor(value, dtype=torch.float32, device=device)
        for name, value in action_stats.items()
    }
    model.to(device)
    model.eval()
    metadata = {
        "path": str(path),
        "sha256": sha256_file(path),
        "global_step": completed_step,
        "best_step": checkpoint.get("best_step"),
        "initialization_state_sha256": initialization_state_sha256,
        "manifest_sha256": expected_manifest_sha256,
        "architecture": ULA_MMDIT_V3_ADALN_ARCHITECTURE,
        "hidden_dim": EXPECTED_HIDDEN_DIM,
        "layers": EXPECTED_LAYERS,
        "semantic_tokens": EXPECTED_SEMANTIC_TOKENS,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "no_kimodo": True,
    }
    return model, metadata


def validate_completion_summary(
    path: str | Path,
    *,
    checkpoint_path: Path,
    branch: str,
    variant: str | None = None,
    checkpoint_sha256: str | None = None,
    foundation_checkpoint_sha256: str | None = None,
    condition_cache_sha256: str | None = None,
    pair_contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Require a trainer-written terminal summary before consuming a checkpoint."""
    path = _require_file(Path(path), f"branch {branch} completion summary")
    summary = load_json(path)
    completed = summary.get("completed_steps")
    target = summary.get("target_steps")
    stopped_early = summary.get("stopped_early")
    if not isinstance(completed, int) or completed <= 0:
        raise EvaluationContractError(
            f"branch {branch} completion summary has no positive completed_steps"
        )
    if not isinstance(target, int) or target <= 0:
        raise EvaluationContractError(
            f"branch {branch} completion summary has no positive target_steps"
        )
    if completed < target and stopped_early is not True:
        raise EvaluationContractError(
            f"branch {branch} is incomplete: {completed}/{target}"
        )
    raw_checkpoint = summary.get("checkpoint")
    if not isinstance(raw_checkpoint, str) or not raw_checkpoint:
        raise EvaluationContractError(
            f"branch {branch} completion summary does not name its checkpoint"
        )
    summary_checkpoint = Path(raw_checkpoint).expanduser()
    if not summary_checkpoint.is_absolute():
        summary_checkpoint = path.parent / summary_checkpoint
    if summary_checkpoint.resolve() != checkpoint_path.resolve():
        raise EvaluationContractError(
            f"branch {branch} completion summary names a different checkpoint"
        )
    if variant is not None:
        exact_experimental = {
            "artifact_kind": EXPECTED_EXPERIMENTAL_SUMMARY_KIND,
            "experimental_only": True,
            "formal_release_eligible": False,
            "variant": variant,
            "checkpoint_sha256": checkpoint_sha256,
            "foundation_checkpoint_sha256": foundation_checkpoint_sha256,
            "condition_cache_sha256": condition_cache_sha256,
            "pair_contract_sha256": pair_contract_sha256,
            "stopped_early": False,
        }
        bad = [
            key
            for key, expected in exact_experimental.items()
            if summary.get(key) != expected
        ]
        if bad:
            raise EvaluationContractError(
                f"branch {branch} experimental completion receipt mismatch: {bad}"
            )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "completed_steps": completed,
        "target_steps": target,
        "stopped_early": stopped_early is True,
        "artifact_kind": summary.get("artifact_kind"),
    }


def _validate_clean_foundation_receipt(
    receipt: Any,
    *,
    foundation_checkpoint_sha256: str,
    manifest_sha256: str,
    branch: str,
) -> str:
    if not isinstance(receipt, Mapping):
        raise EvaluationContractError(
            f"branch {branch} foundation_receipt is missing"
        )
    if (
        receipt.get("checkpoint_sha256") != foundation_checkpoint_sha256
        or receipt.get("architecture") != ULA_MMDIT_V3_ADALN_ARCHITECTURE
    ):
        raise EvaluationContractError(
            f"branch {branch} does not bind the exact A foundation"
        )
    for field in (
        "data_isolation_contract_sha256",
        "split_contract_sha256",
        "style_contract_sha256",
    ):
        _require_sha(receipt.get(field), f"branch {branch} foundation_receipt.{field}")
    random_initialization = receipt.get("random_initialization")
    required_init = {
        "mode": EXPECTED_INITIALIZATION_MODE,
        "qwen_policy": EXPECTED_QWEN_DISABLED_POLICY,
        "kimodo_policy": EXPECTED_KIMODO_POLICY,
        "generator_checkpoint_inputs": [],
    }
    if not isinstance(random_initialization, Mapping):
        raise EvaluationContractError(
            f"branch {branch} foundation random initialization is missing"
        )
    bad = [
        key
        for key, expected in required_init.items()
        if random_initialization.get(key) != expected
    ]
    if bad:
        raise EvaluationContractError(
            f"branch {branch} foundation is not clean/no-Kimodo: {bad}"
        )
    initialization_sha256 = _require_sha(
        random_initialization.get("generator_state_sha256"),
        f"branch {branch} foundation generator_state_sha256",
    )
    _checkpoint_sources_are_beat2_only(receipt)
    source_hashes = {
        source["manifest_sha256"]
        for source in receipt["sources"]["motion_manifests"]
    }
    if source_hashes != {manifest_sha256}:
        raise EvaluationContractError(
            f"branch {branch} foundation receipt binds a different dataset"
        )
    return initialization_sha256


def load_experimental_264d_cache(
    receipt: Any,
    *,
    variant: str,
    manifest_records: Mapping[str, Mapping[str, Any]],
    manifest_sha256: str,
    source_128d_cache_sha256: str,
    style_cache_sha256: str,
    expected_conditions: Mapping[str, np.ndarray],
) -> tuple[str, dict[str, np.ndarray]]:
    """Validate the actual 264D cache consumed during B/C generator training."""
    if not isinstance(receipt, Mapping):
        raise EvaluationContractError(
            f"experimental {variant} condition_cache_receipt is missing"
        )
    exact = {
        "artifact_kind": EXPECTED_EXPERIMENTAL_264D_CACHE_KIND,
        "variant": variant,
        "source_128d_cache_sha256": source_128d_cache_sha256,
        "style_cache_sha256": style_cache_sha256,
        "source_manifest_sha256": manifest_sha256,
        "no_kimodo": True,
        "semantic_scope": EXPECTED_QWEN_SCOPE,
    }
    bad = [key for key, expected in exact.items() if receipt.get(key) != expected]
    if bad:
        raise EvaluationContractError(
            f"experimental {variant} condition cache receipt mismatch: {bad}"
        )
    cache_sha256 = _require_sha(
        receipt.get("cache_sha256"),
        f"experimental {variant} condition_cache_receipt.cache_sha256",
    )
    raw_path = receipt.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise EvaluationContractError(
            f"experimental {variant} condition cache path is missing"
        )
    path = _require_file(Path(raw_path).expanduser().resolve(), f"{variant} 264D cache")
    if sha256_file(path) != cache_sha256:
        raise EvaluationContractError(
            f"experimental {variant} 264D cache SHA mismatch"
        )
    try:
        with np.load(path, allow_pickle=False) as payload:
            clip_ids = payload["clip_ids"].astype(str)
            conditions = np.asarray(payload["conditions"], dtype=np.float32)
    except (OSError, ValueError, KeyError) as error:
        raise EvaluationContractError(
            f"cannot load experimental {variant} 264D cache: {error}"
        ) from error
    if conditions.shape != (len(clip_ids), CONDITION_DIM):
        raise EvaluationContractError(
            f"experimental {variant} cache is not [episodes,264]"
        )
    if set(clip_ids.tolist()) != set(manifest_records):
        raise EvaluationContractError(
            f"experimental {variant} cache clip set differs from the manifest"
        )
    if len(set(clip_ids.tolist())) != len(clip_ids):
        raise EvaluationContractError(
            f"experimental {variant} cache has duplicate clip IDs"
        )
    result = {
        clip_id: conditions[index] for index, clip_id in enumerate(clip_ids)
    }
    for clip_id, expected in expected_conditions.items():
        actual = result.get(clip_id)
        if actual is None or not np.array_equal(actual, expected):
            raise EvaluationContractError(
                f"experimental {variant} cache condition mismatch for {clip_id}"
            )
        if np.any(actual[:133]):
            raise EvaluationContractError(
                f"experimental {variant} cache polluted indices 0:133"
            )
    return cache_sha256, result


def validate_experimental_checkpoint(
    path: str | Path,
    *,
    branch: str,
    variant: str,
    expected_manifest_sha256: str,
    foundation_checkpoint_sha256: str,
    source_128d_cache_sha256: str,
    style_cache_sha256: str,
    manifest_records: Mapping[str, Mapping[str, Any]],
    expected_conditions: Mapping[str, np.ndarray],
    device: str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load B/C through their independent experimental-only contract."""
    path = _require_file(Path(path), f"branch {branch} experimental checkpoint")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise EvaluationContractError(
            f"cannot load branch {branch} experimental checkpoint: {error}"
        ) from error
    if not isinstance(checkpoint, Mapping):
        raise EvaluationContractError(
            f"branch {branch} experimental checkpoint is not an object"
        )
    exact = {
        "schema_version": 1,
        "artifact_kind": EXPECTED_EXPERIMENTAL_CHECKPOINT_KIND,
        "experimental_only": True,
        "formal_release_eligible": False,
        "semantic_scope": EXPECTED_QWEN_SCOPE,
        "variant": variant,
        "architecture": ULA_MMDIT_V3_ADALN_ARCHITECTURE,
        "action_dim": ACTION_DIM,
        "condition_dim": CONDITION_DIM,
        "joint_order": list(JOINT_ORDER_18D),
    }
    bad = [key for key, expected in exact.items() if checkpoint.get(key) != expected]
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        bad.append("config")
        config = {}
    expected_config = {
        "hidden_dim": EXPECTED_HIDDEN_DIM,
        "layers": EXPECTED_LAYERS,
        "semantic_tokens": EXPECTED_SEMANTIC_TOKENS,
    }
    bad.extend(
        f"config.{key}"
        for key, expected in expected_config.items()
        if config.get(key) != expected
    )
    if bad:
        raise EvaluationContractError(
            f"branch {branch} is not the exact experimental AdaLN contract: {bad}"
        )
    initialization_sha256 = _validate_clean_foundation_receipt(
        checkpoint.get("foundation_receipt"),
        foundation_checkpoint_sha256=foundation_checkpoint_sha256,
        manifest_sha256=expected_manifest_sha256,
        branch=branch,
    )
    condition_cache_sha256, _ = load_experimental_264d_cache(
        checkpoint.get("condition_cache_receipt"),
        variant=variant,
        manifest_records=manifest_records,
        manifest_sha256=expected_manifest_sha256,
        source_128d_cache_sha256=source_128d_cache_sha256,
        style_cache_sha256=style_cache_sha256,
        expected_conditions=expected_conditions,
    )
    pair_contract = checkpoint.get("pair_contract")
    if not isinstance(pair_contract, Mapping):
        raise EvaluationContractError(f"branch {branch} pair_contract is missing")
    pair_sha256 = _require_sha(
        pair_contract.get("sha256"), f"branch {branch} pair_contract.sha256"
    )
    pair_path_raw = pair_contract.get("path")
    if not isinstance(pair_path_raw, str) or not pair_path_raw:
        raise EvaluationContractError(
            f"branch {branch} pair_contract.path is missing"
        )
    pair_path = _require_file(
        Path(pair_path_raw).expanduser().resolve(), f"branch {branch} pair contract"
    )
    validate_self_hashed_json(
        pair_path,
        expected_sha256=pair_sha256,
        field=f"branch {branch} pair contract",
    )
    preservation = checkpoint.get("preservation")
    if (
        not isinstance(preservation, Mapping)
        or preservation.get("zero_latent_exact_equivalence_passed") is not True
        or preservation.get("zero_latent_max_abs_error") != 0.0
        or preservation.get("frozen_parameter_max_abs_error") != 0.0
        or preservation.get("nonlatent_plan_columns_max_abs_error") != 0.0
    ):
        raise EvaluationContractError(
            f"branch {branch} did not prove exact A/non-latent preservation"
        )
    completed_step = checkpoint.get("global_step")
    if not isinstance(completed_step, int) or completed_step <= 0:
        raise EvaluationContractError(f"branch {branch} has no completed training step")
    action_stats = _validate_action_stats(checkpoint.get("action_stats"))
    state = checkpoint.get("model_state_dict")
    trainable_receipt = validate_experimental_training_contract(
        checkpoint.get("training_contract"),
        state,
        branch=branch,
    )
    model = create_ula_model(
        ULA_MMDIT_V3_ADALN_ARCHITECTURE,
        action_dim=ACTION_DIM,
        condition_dim=CONDITION_DIM,
        hidden_dim=EXPECTED_HIDDEN_DIM,
        layers=EXPECTED_LAYERS,
        semantic_tokens=EXPECTED_SEMANTIC_TOKENS,
    )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise EvaluationContractError(
            f"branch {branch} experimental model state is invalid: {error}"
        ) from error
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise EvaluationContractError(
            f"branch {branch} contains non-finite model parameters"
        )
    model.action_stats = {
        name: torch.as_tensor(value, dtype=torch.float32, device=device)
        for name, value in action_stats.items()
    }
    model.to(device)
    model.eval()
    metadata = {
        "path": str(path),
        "sha256": sha256_file(path),
        "artifact_kind": EXPECTED_EXPERIMENTAL_CHECKPOINT_KIND,
        "experimental_only": True,
        "formal_release_eligible": False,
        "variant": variant,
        "global_step": completed_step,
        "initialization_state_sha256": initialization_sha256,
        "foundation_checkpoint_sha256": foundation_checkpoint_sha256,
        "condition_cache_sha256": condition_cache_sha256,
        "pair_contract_sha256": pair_sha256,
        "manifest_sha256": expected_manifest_sha256,
        "architecture": ULA_MMDIT_V3_ADALN_ARCHITECTURE,
        "hidden_dim": EXPECTED_HIDDEN_DIM,
        "layers": EXPECTED_LAYERS,
        "semantic_tokens": EXPECTED_SEMANTIC_TOKENS,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_contract": trainable_receipt,
        "no_kimodo": True,
    }
    return model, metadata


def validate_zero_latent_equivalence(
    foundation: torch.nn.Module,
    experimental: torch.nn.Module,
    *,
    style_conditions: Sequence[np.ndarray],
    device: str,
) -> None:
    """Independently confirm B/C collapse exactly to A when latent is zero."""
    generator = torch.Generator(device=device)
    generator.manual_seed(20260727)
    x = torch.randn((1, 11, ACTION_DIM), generator=generator, device=device)
    t = torch.tensor([0.371], dtype=torch.float32, device=device)
    with torch.no_grad():
        for index, condition in enumerate(style_conditions):
            tensor = torch.as_tensor(condition, dtype=torch.float32, device=device)[
                None, :
            ]
            foundation_output = foundation(x, t, tensor)
            experimental_output = experimental(x, t, tensor)
            if not torch.equal(foundation_output, experimental_output):
                maximum_error = float(
                    torch.max(torch.abs(foundation_output - experimental_output)).cpu()
                )
                raise EvaluationContractError(
                    "experimental zero-latent path is not exactly A-equivalent "
                    f"for style probe {index}; max_abs_error={maximum_error}"
                )


def load_manifest(
    path: str | Path, *, expected_sha256: str
) -> tuple[dict[str, dict[str, Any]], str]:
    path = _require_file(Path(path), "BEAT2 manifest")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != _require_sha(expected_sha256, "source_manifest_sha256"):
        raise EvaluationContractError(
            f"BEAT2 manifest hash mismatch: {actual_sha256}"
        )
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise EvaluationContractError(
                    f"invalid manifest JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise EvaluationContractError(
                    f"manifest record {line_number} is not an object"
                )
            clip_id = record.get("clip_id")
            if not isinstance(clip_id, str) or not clip_id:
                raise EvaluationContractError(
                    f"manifest record {line_number} has no clip_id"
                )
            if clip_id in records:
                raise EvaluationContractError(f"duplicate manifest clip_id: {clip_id}")
            if record.get("dataset") != "BEAT2":
                raise EvaluationContractError(
                    f"manifest record {clip_id} is not exact dataset BEAT2"
                )
            records[clip_id] = record
    if not records:
        raise EvaluationContractError("BEAT2 manifest is empty")
    return records, actual_sha256


def _load_npz_sidecar(path: Path) -> dict[str, Any]:
    sidecar = Path(str(path) + ".json")
    metadata = load_json(_require_file(sidecar, f"sidecar for {path.name}"))
    actual_sha256 = sha256_file(path)
    if metadata.get("cache_sha256") != actual_sha256:
        raise EvaluationContractError(f"cache SHA mismatch for {path}")
    return metadata


def load_style_cache(
    path: str | Path, *, manifest_records: Mapping[str, Mapping[str, Any]]
) -> dict[str, np.ndarray]:
    path = _require_file(Path(path), "clean style cache")
    metadata = _load_npz_sidecar(path)
    required = {
        "artifact_kind": "ula_v2_18d_motion_only_style_condition_cache",
        "condition_dim": CONDITION_DIM,
        "condition_policy": EXPECTED_STYLE_POLICY,
        "kimodo_policy": EXPECTED_KIMODO_POLICY,
        "qwen_policy": EXPECTED_QWEN_DISABLED_POLICY,
        "condition_exact_zero_ranges": [[0, 133], [136, 264]],
        "condition_nonzero_indices": [133, 134, 135],
    }
    bad = [key for key, expected in required.items() if metadata.get(key) != expected]
    if bad:
        raise EvaluationContractError(
            f"clean style cache violates style-only/no-Kimodo contract: {bad}"
        )
    try:
        with np.load(path, allow_pickle=False) as payload:
            clip_ids = payload["clip_ids"].astype(str)
            conditions = np.asarray(payload["conditions"], dtype=np.float32)
    except (OSError, ValueError, KeyError) as error:
        raise EvaluationContractError(f"cannot load clean style cache: {error}") from error
    if conditions.shape != (len(clip_ids), CONDITION_DIM):
        raise EvaluationContractError("clean style cache has invalid condition shape")
    if len(set(clip_ids.tolist())) != len(clip_ids):
        raise EvaluationContractError("clean style cache contains duplicate clip IDs")
    if set(clip_ids.tolist()) != set(manifest_records):
        raise EvaluationContractError(
            "clean style cache clip set does not exactly match the BEAT2 manifest"
        )
    if not np.isfinite(conditions).all():
        raise EvaluationContractError("clean style cache contains non-finite values")
    if np.any(conditions[:, :133]) or np.any(conditions[:, 136:]):
        raise EvaluationContractError(
            "clean style cache has nonzero values outside indices 133:136"
        )
    return {clip_id: conditions[index] for index, clip_id in enumerate(clip_ids)}


def load_qwen_cache(
    path: str | Path,
    *,
    variant: str,
    manifest_records: Mapping[str, Mapping[str, Any]],
    manifest_sha256: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = _require_file(Path(path), f"Qwen {variant} cache")
    metadata = _load_npz_sidecar(path)
    required = {
        "artifact_kind": EXPECTED_QWEN_CACHE_KIND,
        "condition_dim": KIMODO_MOTION_LATENT_DIM,
        "motion_latent_dim": KIMODO_MOTION_LATENT_DIM,
        "data_policy": "beat2_only_no_external_motion_dataset_v1",
        "no_kimodo": True,
        "semantic_scope": EXPECTED_QWEN_SCOPE,
        "source_manifest_sha256": manifest_sha256,
        "variant": variant,
    }
    bad = [key for key, expected in required.items() if metadata.get(key) != expected]
    if bad:
        raise EvaluationContractError(
            f"Qwen {variant} cache violates BEAT2-only provenance: {bad}"
        )
    qwen = metadata.get("qwen")
    if (
        not isinstance(qwen, Mapping)
        or qwen.get("source") != "official_huggingface_base"
        or qwen.get("input_checkpoint_kind") != "official_base_only"
    ):
        raise EvaluationContractError(
            f"Qwen {variant} cache is not based on the pinned official model"
        )
    if variant == "lora_finetuned":
        lora = qwen.get("lora")
        if not isinstance(lora, Mapping) or lora.get("training_data") != "BEAT2_only":
            raise EvaluationContractError("Qwen LoRA cache is not BEAT2-only")
    try:
        with np.load(path, allow_pickle=False) as payload:
            clip_ids = payload["clip_ids"].astype(str)
            conditions = np.asarray(payload["conditions"], dtype=np.float32)
            splits = payload["fixed_split_assignments"].astype(str)
            trajectory_sha256 = payload["trajectory_sha256"].astype(str)
    except (OSError, ValueError, KeyError) as error:
        raise EvaluationContractError(
            f"cannot load Qwen {variant} cache: {error}"
        ) from error
    if conditions.shape != (len(clip_ids), KIMODO_MOTION_LATENT_DIM):
        raise EvaluationContractError(f"Qwen {variant} cache has invalid shape")
    if len(set(clip_ids.tolist())) != len(clip_ids):
        raise EvaluationContractError(f"Qwen {variant} cache has duplicate clip IDs")
    if set(clip_ids.tolist()) != set(manifest_records):
        raise EvaluationContractError(
            f"Qwen {variant} cache clip set does not match the BEAT2 manifest"
        )
    if not np.isfinite(conditions).all():
        raise EvaluationContractError(
            f"Qwen {variant} cache contains non-finite conditions"
        )
    result: dict[str, np.ndarray] = {}
    for index, clip_id in enumerate(clip_ids):
        record = manifest_records[clip_id]
        if splits[index] != record.get("fixed_split_assignment"):
            raise EvaluationContractError(
                f"Qwen {variant} split mismatch for {clip_id}"
            )
        expected_trajectory_sha256 = record.get("motion_18d", {}).get(
            "safe_csv_sha256"
        )
        if trajectory_sha256[index] != expected_trajectory_sha256:
            raise EvaluationContractError(
                f"Qwen {variant} trajectory hash mismatch for {clip_id}"
            )
        result[clip_id] = conditions[index]
    return result, metadata


def compose_condition(
    style_condition: np.ndarray, latent: np.ndarray | None
) -> np.ndarray:
    """Create the exact clean 264D A/B/C condition."""
    style_condition = np.asarray(style_condition, dtype=np.float32)
    if style_condition.shape != (CONDITION_DIM,):
        raise EvaluationContractError("style condition must have shape [264]")
    if np.any(style_condition[:133]) or np.any(style_condition[136:]):
        raise EvaluationContractError(
            "style source contains nonzero values outside indices 133:136"
        )
    condition = np.zeros(CONDITION_DIM, dtype=np.float32)
    condition[STYLE_SLICE] = style_condition[STYLE_SLICE]
    if latent is not None:
        latent = np.asarray(latent, dtype=np.float32)
        if latent.shape != (KIMODO_MOTION_LATENT_DIM,):
            raise EvaluationContractError("Qwen latent must have shape [128]")
        if not np.isfinite(latent).all():
            raise EvaluationContractError("Qwen latent contains non-finite values")
        condition[LATENT_SLICE] = latent
    if np.any(condition[:133]):
        raise EvaluationContractError("condition indices 0:133 must be exact zero")
    return condition


def _distribution_metrics(values: np.ndarray) -> dict[str, float]:
    flat = np.abs(np.asarray(values, dtype=np.float64)).reshape(-1)
    if not flat.size:
        return {"rms": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "rms": float(np.sqrt(np.mean(np.square(flat)))),
        "p95": float(np.percentile(flat, 95)),
        "max": float(np.max(flat)),
    }


def trajectory_metrics(trajectory: np.ndarray, *, fps: float) -> dict[str, Any]:
    """Report velocity/acceleration/jerk and expressiveness proxies."""
    values = np.asarray(trajectory, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != ACTION_DIM
        or not np.isfinite(values).all()
    ):
        raise EvaluationContractError("trajectory must be finite [frames,18]")
    velocity = np.diff(values, axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps
    jerk = np.diff(acceleration, axis=0) * fps
    centered = values - np.mean(values, axis=0, keepdims=True)
    joint_ranges = np.ptp(values, axis=0) if len(values) else np.zeros(ACTION_DIM)
    head = values[:, HEAD_INDICES]
    head_velocity = np.diff(head, axis=0) * fps
    return {
        "frames": int(len(values)),
        "sample_span_sec": float(max(0, len(values) - 1) / fps),
        "velocity_rad_s": _distribution_metrics(velocity),
        "acceleration_rad_s2": _distribution_metrics(acceleration),
        "jerk_rad_s3": _distribution_metrics(jerk),
        "amplitude": {
            "rms_about_joint_mean_rad": float(
                np.sqrt(np.mean(np.square(centered))) if centered.size else 0.0
            ),
            "joint_range_rms_rad": float(np.sqrt(np.mean(np.square(joint_ranges)))),
            "joint_range_p95_rad": float(np.percentile(joint_ranges, 95)),
            "joint_range_max_rad": float(np.max(joint_ranges)),
        },
        "head_activity": {
            "joint_indices": list(HEAD_INDICES),
            "joint_names": list(JOINT_ORDER_18D[15:18]),
            "angle_rad": _distribution_metrics(
                head - np.mean(head, axis=0, keepdims=True)
            ),
            "velocity_rad_s": _distribution_metrics(head_velocity),
            "joint_range_rad": _distribution_metrics(
                np.ptp(head, axis=0) if len(head) else np.zeros(3)
            ),
        },
    }


def trajectory_delta_metrics(
    actual: np.ndarray, counterfactual: np.ndarray
) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64)
    counterfactual = np.asarray(counterfactual, dtype=np.float64)
    if actual.shape != counterfactual.shape:
        raise EvaluationContractError("condition sensitivity trajectories differ in shape")
    return _distribution_metrics(actual - counterfactual)


def postprocess_trajectory(
    trajectory: np.ndarray, *, fps: float, max_velocity_rad_s: float, smooth_window: int
) -> np.ndarray:
    from upper_body_skeleton.long_emotion_infer import (
        postprocess_trajectory as playback_postprocess,
    )

    return playback_postprocess(
        trajectory,
        fps=fps,
        max_velocity_rad_s=max_velocity_rad_s,
        smooth_window=smooth_window,
    )


def _render_single(
    trajectory: np.ndarray,
    *,
    csv_path: Path,
    mp4_path: Path,
    fps: float,
    width: int,
    height: int,
    simplified: bool,
) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    from upper_body_skeleton.mujoco_playback import render_motion

    write_generated_csv(csv_path, trajectory, fps=fps)
    return render_motion(
        csv_path,
        mp4_path,
        fps=fps,
        width=width,
        height=height,
        simplified=simplified,
    )


def build_side_by_side(
    inputs: Sequence[str | Path],
    output: str | Path,
    *,
    labels: Sequence[str] = ("A STYLE-ONLY", "B FROZEN QWEN", "C BEAT2 QWEN LORA"),
    pane_width: int,
) -> dict[str, Any]:
    """Use ffmpeg to label and hstack the three equal-size branch videos."""
    if len(inputs) != 3 or len(labels) != 3:
        raise EvaluationContractError("side-by-side requires exactly three inputs")
    inputs = [Path(path) for path in inputs]
    for index, path in enumerate(inputs):
        _require_file(path, f"side-by-side input {index}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pane_width = int(pane_width)
    if pane_width <= 0:
        raise EvaluationContractError("pane_width must be positive")
    # The compact imageio-ffmpeg build can omit drawtext.  Pillow creates the
    # deterministic title strip; ffmpeg still performs all video stacking and
    # final H.264 encoding.
    from PIL import Image, ImageDraw, ImageFont

    title_height = 40
    banner = Image.new("RGB", (pane_width * 3, title_height), (17, 19, 24))
    draw = ImageDraw.Draw(banner)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    try:
        font = ImageFont.truetype(str(font_path), 20)
    except OSError:
        font = ImageFont.load_default()
    for index, label in enumerate(labels):
        bounds = draw.textbbox((0, 0), label, font=font)
        text_width = bounds[2] - bounds[0]
        x = index * pane_width + max(4, (pane_width - text_width) // 2)
        draw.text((x, 8), label, fill=(242, 244, 248), font=font)
        if index:
            draw.line(
                (index * pane_width, 0, index * pane_width, title_height),
                fill=(210, 214, 220),
                width=2,
            )
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.mp4")
    banner_path = output.with_name(f".{output.name}.{os.getpid()}.titles.png")
    banner.save(banner_path)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y"]
    for path in inputs:
        command.extend(["-i", str(path)])
    command.extend(["-loop", "1", "-framerate", "30", "-i", str(banner_path)])
    command.extend(
        [
            "-filter_complex",
            "[0:v][1:v][2:v]hstack=inputs=3[body];"
            "[3:v][body]vstack=inputs=2:shortest=1[outv]",
            "-map",
            "[outv]",
            "-an",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary),
        ]
    )
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise EvaluationContractError(
                "ffmpeg side-by-side failed: " + completed.stderr[-2000:]
            )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
        banner_path.unlink(missing_ok=True)
    return {
        "output_mp4": str(output),
        "sha256": sha256_file(output),
        "inputs": [str(path) for path in inputs],
        "labels": list(labels),
        "ffmpeg": imageio_ffmpeg.get_ffmpeg_exe(),
    }


def _validate_cases(
    value: Any, *, manifest_records: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise EvaluationContractError("held_out_cases must be a non-empty list")
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, case in enumerate(value):
        if not isinstance(case, Mapping):
            raise EvaluationContractError(f"held_out_cases[{index}] is invalid")
        clip_id = case.get("clip_id")
        seed = case.get("seed")
        if not isinstance(clip_id, str) or clip_id not in manifest_records:
            raise EvaluationContractError(
                f"held_out_cases[{index}].clip_id is not in the manifest"
            )
        if clip_id in seen:
            raise EvaluationContractError(f"duplicate held-out clip: {clip_id}")
        seen.add(clip_id)
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise EvaluationContractError(
                f"held_out_cases[{index}].seed must be a nonnegative integer"
            )
        record = manifest_records[clip_id]
        if record.get("fixed_split_assignment") != "test":
            raise EvaluationContractError(f"held-out clip is not fixed test: {clip_id}")
        if record.get("accepted_for_training") is not True:
            raise EvaluationContractError(f"held-out clip is not QC accepted: {clip_id}")
        motion = record.get("motion_18d")
        if (
            not isinstance(motion, Mapping)
            or motion.get("action_dim") != ACTION_DIM
            or motion.get("output_contract") != "ula_v2_18d_head_v1"
            or motion.get("quality_gate", {}).get("passed") is not True
        ):
            raise EvaluationContractError(
                f"held-out clip lacks passed 18D physical QC: {clip_id}"
            )
        frames = record.get("frames")
        fps = record.get("fps")
        if not isinstance(frames, int) or frames < 4:
            raise EvaluationContractError(f"held-out clip has too few frames: {clip_id}")
        if not isinstance(fps, (int, float)) or not math.isclose(
            float(fps), 30.0, abs_tol=1e-9
        ):
            raise EvaluationContractError(f"held-out clip is not 30 Hz: {clip_id}")
        cases.append(
            {
                "clip_id": clip_id,
                "seed": seed,
                "frames": frames,
                "fps": float(fps),
                "prompt": record.get("prompt"),
                "pilot_dynamic_band": record.get("pilot_dynamic_band"),
                "trajectory_sha256": motion.get("safe_csv_sha256"),
            }
        )
    return cases


def _condition_counterfactuals(condition: np.ndarray) -> dict[str, np.ndarray]:
    zero = np.zeros_like(condition)
    style_ablated = condition.copy()
    style_ablated[STYLE_SLICE] = 0.0
    latent_ablated = condition.copy()
    latent_ablated[LATENT_SLICE] = 0.0
    result = {"zero_condition": zero, "style_ablated": style_ablated}
    if np.any(condition[LATENT_SLICE]):
        result["latent_ablated"] = latent_ablated
    return result


def _sample(
    model: torch.nn.Module,
    condition: np.ndarray,
    *,
    frames: int,
    steps: int,
    seed: int,
    device: str,
) -> np.ndarray:
    return sample_trajectory(
        model,
        condition=condition,
        frames=frames,
        action_dim=ACTION_DIM,
        steps=steps,
        device=device,
        seed=seed,
        action_stats=model.action_stats,
    )


def validate_inputs(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = load_json(config_path)
    if config.get("artifact_kind") != CONFIG_ARTIFACT_KIND:
        raise EvaluationContractError(
            f"config artifact_kind must be {CONFIG_ARTIFACT_KIND}"
        )
    config_dir = config_path.parent
    manifest_path = _resolve_path(
        config.get("source_manifest"),
        config_dir=config_dir,
        field="source_manifest",
    )
    manifest_records, manifest_sha256 = load_manifest(
        manifest_path,
        expected_sha256=config.get("source_manifest_sha256"),
    )
    cases = _validate_cases(
        config.get("held_out_cases"), manifest_records=manifest_records
    )
    paths = {
        "manifest": manifest_path,
        "style_cache": _resolve_path(
            config.get("style_condition_cache"),
            config_dir=config_dir,
            field="style_condition_cache",
        ),
        "frozen_cache": _resolve_path(
            config.get("frozen_qwen_condition_cache"),
            config_dir=config_dir,
            field="frozen_qwen_condition_cache",
        ),
        "lora_cache": _resolve_path(
            config.get("lora_qwen_condition_cache"),
            config_dir=config_dir,
            field="lora_qwen_condition_cache",
        ),
        "output_dir": _resolve_path(
            config.get("output_dir"),
            config_dir=config_dir,
            field="output_dir",
        ),
    }
    branches = config.get("branches")
    if not isinstance(branches, Mapping) or set(branches) != {"A", "B", "C"}:
        raise EvaluationContractError("branches must contain exactly A, B, and C")
    checkpoints: dict[str, Path] = {}
    completion_summaries: dict[str, Path] = {}
    for branch in ("A", "B", "C"):
        record = branches[branch]
        if not isinstance(record, Mapping):
            raise EvaluationContractError(f"branch {branch} config is invalid")
        checkpoints[branch] = _resolve_path(
            record.get("checkpoint"),
            config_dir=config_dir,
            field=f"branches.{branch}.checkpoint",
        )
        completion_summaries[branch] = _resolve_path(
            record.get("completion_summary"),
            config_dir=config_dir,
            field=f"branches.{branch}.completion_summary",
        )
    sampling = config.get("sampling")
    playback = config.get("playback")
    render = config.get("render")
    if not isinstance(sampling, Mapping):
        raise EvaluationContractError("sampling config is missing")
    if not isinstance(playback, Mapping):
        raise EvaluationContractError("playback config is missing")
    if not isinstance(render, Mapping):
        raise EvaluationContractError("render config is missing")
    steps = sampling.get("steps")
    if not isinstance(steps, int) or steps <= 0:
        raise EvaluationContractError("sampling.steps must be positive")
    return {
        "config": config,
        "config_path": config_path,
        "config_sha256": sha256_file(config_path),
        "manifest_records": manifest_records,
        "manifest_sha256": manifest_sha256,
        "paths": paths,
        "checkpoints": checkpoints,
        "completion_summaries": completion_summaries,
        "cases": cases,
        "sampling": dict(sampling),
        "playback": dict(playback),
        "render": dict(render),
    }


def build_evaluation(config_path: str | Path, *, validate_only: bool = False) -> dict:
    plan = validate_inputs(config_path)
    records = plan["manifest_records"]
    manifest_sha256 = plan["manifest_sha256"]
    style = load_style_cache(plan["paths"]["style_cache"], manifest_records=records)
    frozen, frozen_metadata = load_qwen_cache(
        plan["paths"]["frozen_cache"],
        variant="frozen_base",
        manifest_records=records,
        manifest_sha256=manifest_sha256,
    )
    lora, lora_metadata = load_qwen_cache(
        plan["paths"]["lora_cache"],
        variant="lora_finetuned",
        manifest_records=records,
        manifest_sha256=manifest_sha256,
    )
    output_dir = plan["paths"]["output_dir"]
    validation_summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "status": "validated_inputs_not_generated",
        "created_at": utc_now(),
        "config": str(plan["config_path"]),
        "config_sha256": plan["config_sha256"],
        "source_manifest": str(plan["paths"]["manifest"]),
        "source_manifest_sha256": manifest_sha256,
        "held_out_cases": plan["cases"],
        "condition_caches": {
            "style": {
                "path": str(plan["paths"]["style_cache"]),
                "sha256": sha256_file(plan["paths"]["style_cache"]),
            },
            "frozen_qwen": {
                "path": str(plan["paths"]["frozen_cache"]),
                "sha256": sha256_file(plan["paths"]["frozen_cache"]),
                "variant": frozen_metadata["variant"],
            },
            "lora_qwen": {
                "path": str(plan["paths"]["lora_cache"]),
                "sha256": sha256_file(plan["paths"]["lora_cache"]),
                "variant": lora_metadata["variant"],
            },
        },
        "contracts": {
            "dataset": "BEAT2-only",
            "kimodo": "forbidden/not used",
            "architecture": "ula_mmdit_v3_adaln full 384x6",
            "condition_A": "zero[0:133] + style[133:136] + zero[136:264]",
            "condition_BC": "zero[0:133] + style[133:136] + Qwen latent[136:264]",
            "fairness": "same held-out clips/native lengths/seeds/steps/playback/render",
            "semantic_scope": EXPECTED_QWEN_SCOPE,
            "expression_scope": (
                "18D upper-body plus 3-DoF head orientation; no facial blendshape "
                "channels are present or evaluated"
            ),
        },
        "output_dir": str(output_dir),
    }
    if validate_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(output_dir / "validation.json", validation_summary)
        return validation_summary

    device = str(plan["sampling"].get("device", "cuda"))
    models: dict[str, torch.nn.Module] = {}
    checkpoint_metadata: dict[str, dict[str, Any]] = {}
    foundation_model, foundation_metadata = validate_checkpoint(
        plan["checkpoints"]["A"],
        expected_manifest_sha256=manifest_sha256,
        branch="A",
        device=device,
    )
    foundation_metadata["completion_summary"] = validate_completion_summary(
        plan["completion_summaries"]["A"],
        checkpoint_path=plan["checkpoints"]["A"],
        branch="A",
    )
    models["A"] = foundation_model
    checkpoint_metadata["A"] = foundation_metadata
    style_cache_sha256 = sha256_file(plan["paths"]["style_cache"])
    source_128d_hashes = {
        "B": sha256_file(plan["paths"]["frozen_cache"]),
        "C": sha256_file(plan["paths"]["lora_cache"]),
    }
    variants = {"B": "frozen_base", "C": "lora_finetuned"}
    latent_maps = {"B": frozen, "C": lora}
    for branch in ("B", "C"):
        expected_conditions = {
            clip_id: compose_condition(style[clip_id], latent_maps[branch][clip_id])
            for clip_id in records
        }
        model, metadata = validate_experimental_checkpoint(
            plan["checkpoints"][branch],
            branch=branch,
            variant=variants[branch],
            expected_manifest_sha256=manifest_sha256,
            foundation_checkpoint_sha256=foundation_metadata["sha256"],
            source_128d_cache_sha256=source_128d_hashes[branch],
            style_cache_sha256=style_cache_sha256,
            manifest_records=records,
            expected_conditions=expected_conditions,
            device=device,
        )
        metadata["completion_summary"] = validate_completion_summary(
            plan["completion_summaries"][branch],
            checkpoint_path=plan["checkpoints"][branch],
            branch=branch,
            variant=variants[branch],
            checkpoint_sha256=metadata["sha256"],
            foundation_checkpoint_sha256=foundation_metadata["sha256"],
            condition_cache_sha256=metadata["condition_cache_sha256"],
            pair_contract_sha256=metadata["pair_contract_sha256"],
        )
        validate_zero_latent_equivalence(
            foundation_model,
            model,
            style_conditions=[
                compose_condition(style[case["clip_id"]], None)
                for case in plan["cases"]
            ],
            device=device,
        )
        models[branch] = model
        checkpoint_metadata[branch] = metadata
    initialization_hashes = {
        metadata["initialization_state_sha256"]
        for metadata in checkpoint_metadata.values()
    }
    if len(initialization_hashes) != 1:
        raise EvaluationContractError(
            "A/B/C checkpoints do not share the same random foundation initialization"
        )
    checkpoint_hashes = {
        metadata["sha256"] for metadata in checkpoint_metadata.values()
    }
    if len(checkpoint_hashes) != 3:
        raise EvaluationContractError(
            "A/B/C checkpoints must be three distinct trained artifacts"
        )

    sampling_steps = int(plan["sampling"]["steps"])
    max_velocity = float(plan["playback"].get("max_velocity_rad_s", 3.0))
    smooth_window = int(plan["playback"].get("smooth_window", 5))
    width = int(plan["render"].get("width", 640))
    height = int(plan["render"].get("height", 720))
    simplified = bool(plan["render"].get("simplified", False))
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise EvaluationContractError("render width/height must be positive even integers")

    output_dir.mkdir(parents=True, exist_ok=True)
    case_summaries = []
    for case_index, case in enumerate(plan["cases"]):
        clip_id = case["clip_id"]
        conditions = {
            "A": compose_condition(style[clip_id], None),
            "B": compose_condition(style[clip_id], frozen[clip_id]),
            "C": compose_condition(style[clip_id], lora[clip_id]),
        }
        case_dir = output_dir / f"case_{case_index + 1:02d}"
        branch_summaries: dict[str, dict[str, Any]] = {}
        branch_videos = []
        for branch in ("A", "B", "C"):
            condition = conditions[branch]
            raw = _sample(
                models[branch],
                condition,
                frames=case["frames"],
                steps=sampling_steps,
                seed=case["seed"],
                device=device,
            )
            processed = postprocess_trajectory(
                raw,
                fps=case["fps"],
                max_velocity_rad_s=max_velocity,
                smooth_window=smooth_window,
            )
            sensitivities: dict[str, dict[str, Any]] = {}
            for name, counterfactual_condition in _condition_counterfactuals(
                condition
            ).items():
                counterfactual_raw = _sample(
                    models[branch],
                    counterfactual_condition,
                    frames=case["frames"],
                    steps=sampling_steps,
                    seed=case["seed"],
                    device=device,
                )
                counterfactual_processed = postprocess_trajectory(
                    counterfactual_raw,
                    fps=case["fps"],
                    max_velocity_rad_s=max_velocity,
                    smooth_window=smooth_window,
                )
                sensitivities[name] = {
                    "raw_trajectory_delta_rad": trajectory_delta_metrics(
                        raw, counterfactual_raw
                    ),
                    "playback_trajectory_delta_rad": trajectory_delta_metrics(
                        processed, counterfactual_processed
                    ),
                }
            branch_dir = case_dir / branch
            csv_path = branch_dir / "playback.csv"
            mp4_path = branch_dir / "preview.mp4"
            render_summary = _render_single(
                processed,
                csv_path=csv_path,
                mp4_path=mp4_path,
                fps=case["fps"],
                width=width,
                height=height,
                simplified=simplified,
            )
            npz_path = branch_dir / "trajectories.npz"
            np.savez_compressed(
                npz_path,
                raw=raw.astype(np.float32),
                playback=processed.astype(np.float32),
                condition=condition,
                clip_id=np.asarray(clip_id),
                seed=np.asarray(case["seed"], dtype=np.int64),
                fps=np.asarray(case["fps"], dtype=np.float32),
                joint_order=np.asarray(JOINT_ORDER_18D),
            )
            branch_summary = {
                "branch": branch,
                "checkpoint": checkpoint_metadata[branch],
                "condition": {
                    "sha256": value_sha256(condition.tolist()),
                    "exact_zero_0_133": bool(not np.any(condition[:133])),
                    "style_133_136": condition[STYLE_SLICE].tolist(),
                    "latent_136_264": (
                        "exact_zero"
                        if not np.any(condition[LATENT_SLICE])
                        else {
                            "l2": float(np.linalg.norm(condition[LATENT_SLICE])),
                            "sha256": value_sha256(
                                condition[LATENT_SLICE].tolist()
                            ),
                        }
                    ),
                },
                "raw_metrics": trajectory_metrics(raw, fps=case["fps"]),
                "playback_metrics": trajectory_metrics(
                    processed, fps=case["fps"]
                ),
                "condition_sensitivity": sensitivities,
                "artifacts": {
                    "csv": str(csv_path),
                    "csv_sha256": sha256_file(csv_path),
                    "npz": str(npz_path),
                    "npz_sha256": sha256_file(npz_path),
                    "mp4": str(mp4_path),
                    "mp4_sha256": sha256_file(mp4_path),
                },
                "render": render_summary,
            }
            atomic_json(branch_dir / "metrics.json", branch_summary)
            branch_summaries[branch] = branch_summary
            branch_videos.append(mp4_path)
        comparison = build_side_by_side(
            branch_videos,
            case_dir / "ABC_side_by_side.mp4",
            pane_width=width,
        )
        case_summary = {
            **case,
            "branches": branch_summaries,
            "side_by_side": comparison,
        }
        atomic_json(case_dir / "summary.json", case_summary)
        case_summaries.append(case_summary)

    summary = {
        **validation_summary,
        "status": "complete",
        "completed_at": utc_now(),
        "checkpoints": checkpoint_metadata,
        "sampling": {
            "steps": sampling_steps,
            "device": device,
            "same_seed_per_case_across_branches": True,
        },
        "playback": {
            "max_velocity_rad_s": max_velocity,
            "smooth_window": smooth_window,
        },
        "render": {
            "backend": "MuJoCo EGL headless",
            "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl"),
            "width_per_branch": width,
            "height_per_branch": height,
            "side_by_side_encoder": (
                "Pillow title strip + ffmpeg hstack/vstack/libx264"
            ),
        },
        "cases": case_summaries,
    }
    atomic_json(output_dir / "summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate manifest and condition caches without loading checkpoints.",
    )
    parser.add_argument(
        "--wait-for-completion",
        action="store_true",
        help=(
            "Wait for all three trainer completion summaries and checkpoints, then "
            "run once. Intended for a persistent systemd service."
        ),
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument(
        "--timeout-hours",
        type=float,
        default=18.0,
        help="Fail closed if terminal artifacts do not arrive in this time.",
    )
    return parser.parse_args(argv)


def wait_for_completion_inputs(
    config_path: str | Path, *, poll_seconds: float, timeout_hours: float
) -> None:
    """Wait only for terminal trainer artifacts; never consume live checkpoints."""
    if not math.isfinite(poll_seconds) or poll_seconds < 1.0:
        raise EvaluationContractError("poll-seconds must be finite and at least 1")
    if not math.isfinite(timeout_hours) or timeout_hours <= 0.0:
        raise EvaluationContractError("timeout-hours must be finite and positive")
    config_path = Path(config_path).resolve()
    config = load_json(config_path)
    branches = config.get("branches")
    if not isinstance(branches, Mapping) or set(branches) != {"A", "B", "C"}:
        raise EvaluationContractError("branches must contain exactly A, B, and C")
    required: list[tuple[str, Path]] = []
    for branch in ("A", "B", "C"):
        record = branches[branch]
        if not isinstance(record, Mapping):
            raise EvaluationContractError(f"branch {branch} config is invalid")
        required.extend(
            [
                (
                    f"{branch} checkpoint",
                    _resolve_path(
                        record.get("checkpoint"),
                        config_dir=config_path.parent,
                        field=f"branches.{branch}.checkpoint",
                    ),
                ),
                (
                    f"{branch} completion summary",
                    _resolve_path(
                        record.get("completion_summary"),
                        config_dir=config_path.parent,
                        field=f"branches.{branch}.completion_summary",
                    ),
                ),
            ]
        )
    deadline = time.monotonic() + timeout_hours * 3600.0
    while True:
        missing = [(name, path) for name, path in required if not path.is_file()]
        if not missing:
            return
        if time.monotonic() >= deadline:
            details = ", ".join(f"{name}: {path}" for name, path in missing)
            raise EvaluationContractError(
                f"timed out waiting for terminal A/B/C artifacts: {details}"
            )
        print(
            json.dumps(
                {
                    "status": "waiting_for_terminal_training_artifacts",
                    "missing": [name for name, _ in missing],
                    "checked_at": utc_now(),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        time.sleep(poll_seconds)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.wait_for_completion:
            wait_for_completion_inputs(
                args.config,
                poll_seconds=args.poll_seconds,
                timeout_hours=args.timeout_hours,
            )
        summary = build_evaluation(args.config, validate_only=args.validate_only)
    except EvaluationContractError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": summary["status"],
                "artifact_kind": summary["artifact_kind"],
                "output_dir": summary["output_dir"],
                "cases": len(summary["held_out_cases"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
