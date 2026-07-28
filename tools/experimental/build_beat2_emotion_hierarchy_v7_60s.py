#!/usr/bin/env python3
"""Build the gated 60-second BEAT2 V7 emotion-hierarchy comparison.

This entry point deliberately reuses the already-audited raw playback,
classifier-free guidance, MuJoCo rendering, and text-panel implementation from
``build_beat2_style_emotion_v2_60s``.  It replaces only the artifact contracts
needed for V7 and adds a full 54-example/six-prototype admission gate.

The checked-in configuration exposes a fair frozen-Qwen versus future
LoRA-Qwen interface.  Only the frozen branch is currently eligible.  The LoRA
branch cannot be selected until a real completed generator checkpoint and
training summary are entered in the configuration and pass the same V7 gates.
No placeholder checkpoint is accepted.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.experimental import build_beat2_clean_abc_video as abc_video  # noqa: E402
from tools.experimental import build_beat2_style_emotion_v2_60s as base  # noqa: E402
from tools.train_beat2_emotion_hierarchy_v7 import (  # noqa: E402
    CHECKPOINT_ARTIFACT_KIND,
    SUMMARY_ARTIFACT_KIND as TRAINING_SUMMARY_ARTIFACT_KIND,
    TRAINING_POLICY,
)
from upper_body_skeleton.beat2_condition_control import (  # noqa: E402
    TEXT_LATENT_DIM,
)
from upper_body_skeleton.beat2_emotion_hierarchy import (  # noqa: E402
    INTENDED_WEAK_LABEL_ROLE,
)


SCHEMA_VERSION = 7
CONFIG_ARTIFACT_KIND = "beat2_emotion_hierarchy_qualitative_60s_config_v7"
SUMMARY_ARTIFACT_KIND = "beat2_emotion_hierarchy_qualitative_60s_v7"
DATA_POLICY = "beat2_only_no_external_motion_dataset_v1"
COMPARISON_INTERFACE_KIND = "beat2_qwen_generator_ab_interface_v7"
SHARED_NOISE_POLICY = (
    "same_pair_seed_same_initial_noise_prefix_across_qwen_generator_variants_v1"
)
PLAYBACK_CONTRACT = {
    "planner_native_duration": True,
    "endpoint_hold_only_if_needed": True,
    "smoothing": False,
    "time_warp": False,
    "network_frame_crop": False,
    "boundary_blend": False,
    "last_frame_blend": False,
}
FULL_DIAGNOSTIC_METRICS = {
    "loss": "semantic_global_contrastive_loss",
    "rank": "semantic_global_mean_positive_rank",
    "margin": "semantic_global_hard_cross_group_margin",
}
EXPECTED_EMOTION_PROMPT_TOKENS = {
    "happy affect",
    "angry affect",
    "surprised affect",
    "sad affect",
    "fearful affect",
}
OUTPUT_FILENAMES = {
    "summary": "summary_v7.json",
    "trajectory": "trajectories_v7.npz",
    "neutral_csv": "neutral_endpoint_hold_only_v7.csv",
    "neutral_video": "neutral_endpoint_hold_only_v7.mp4",
    "emotion_csv": "emotion_endpoint_hold_only_v7.csv",
    "emotion_video": "emotion_endpoint_hold_only_v7.mp4",
    "ass": "prompt_timeline_v7.ass",
    "final_video": "beat2_emotion_hierarchy_v7_60s.mp4",
}


class EmotionHierarchyVideoError(base.StyleEmotionVideoError):
    """Raised when a V7 artifact is incomplete or fails admission."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EmotionHierarchyVideoError(f"{field} must be a mapping")
    return value


def _finite_metric(
    diagnostics: Mapping[str, Any], *, name: str, phase: str
) -> float:
    try:
        value = float(diagnostics[name])
    except (KeyError, TypeError, ValueError) as error:
        raise EmotionHierarchyVideoError(
            f"{phase} full six-emotion diagnostic is missing {name}"
        ) from error
    if not math.isfinite(value):
        raise EmotionHierarchyVideoError(
            f"{phase} full six-emotion diagnostic {name} is non-finite"
        )
    return value


def six_emotion_full_diagnostic_decision(
    initial: Mapping[str, Any],
    final: Mapping[str, Any],
    gate_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare only fixed full-diagnostic loss/rank/margin, never batch R@1."""

    required_gate_fields = {
        "required_episode_count",
        "required_prototype_count",
        "non_regression_tolerance",
        "minimum_loss_decrease",
        "minimum_rank_decrease",
        "minimum_margin_increase",
        "minimum_clear_improvement_count",
    }
    if set(gate_config) != required_gate_fields:
        raise EmotionHierarchyVideoError(
            "six_emotion_full_diagnostic_gate fields changed"
        )
    required_episodes = int(gate_config["required_episode_count"])
    required_prototypes = int(gate_config["required_prototype_count"])
    tolerance = float(gate_config["non_regression_tolerance"])
    clear_thresholds = {
        "loss": float(gate_config["minimum_loss_decrease"]),
        "rank": float(gate_config["minimum_rank_decrease"]),
        "margin": float(gate_config["minimum_margin_increase"]),
    }
    required_clear = int(gate_config["minimum_clear_improvement_count"])
    if (
        required_episodes != 54
        or required_prototypes != 6
        or not math.isfinite(tolerance)
        or tolerance < 0
        or any(
            not math.isfinite(value) or value <= 0
            for value in clear_thresholds.values()
        )
        or not 1 <= required_clear <= 3
    ):
        raise EmotionHierarchyVideoError(
            "six-emotion full-diagnostic thresholds are invalid"
        )

    shape_checks = {}
    for phase, diagnostics in (("step0", initial), ("final", final)):
        shape_checks[f"{phase}_has_54_fixed_examples"] = (
            int(diagnostics.get("episode_count", -1)) == required_episodes
        )
        shape_checks[f"{phase}_has_six_global_emotion_prototypes"] = (
            int(
                diagnostics.get("semantic_global_prototype_count", -1)
            )
            == required_prototypes
        )

    initial_values = {
        role: _finite_metric(initial, name=name, phase="step0")
        for role, name in FULL_DIAGNOSTIC_METRICS.items()
    }
    final_values = {
        role: _finite_metric(final, name=name, phase="final")
        for role, name in FULL_DIAGNOSTIC_METRICS.items()
    }
    improvements = {
        "loss": initial_values["loss"] - final_values["loss"],
        "rank": initial_values["rank"] - final_values["rank"],
        "margin": final_values["margin"] - initial_values["margin"],
    }
    non_regression = {
        role: improvement >= -tolerance
        for role, improvement in improvements.items()
    }
    clear = {
        role: improvement >= clear_thresholds[role]
        for role, improvement in improvements.items()
    }
    clear_count = sum(bool(value) for value in clear.values())
    checks = {
        **shape_checks,
        "loss_not_worse_than_step0": non_regression["loss"],
        "rank_not_worse_than_step0": non_regression["rank"],
        "margin_not_worse_than_step0": non_regression["margin"],
        "minimum_clear_improvement_count": clear_count >= required_clear,
    }
    passed = all(checks.values())
    return {
        "passed": bool(passed),
        "diagnostic_scope": (
            "fixed_54_examples_full_six_emotion_global_prototype_bank"
        ),
        "batch_recall_at_1_used": False,
        "metrics_used": deepcopy(FULL_DIAGNOSTIC_METRICS),
        "initial": initial_values,
        "final": final_values,
        "improvements": improvements,
        "clear_improvements": clear,
        "clear_improvement_count": clear_count,
        "checks": checks,
        "thresholds": deepcopy(dict(gate_config)),
        "failure_reasons": [
            name for name, value in checks.items() if not value
        ],
    }


def shared_variant_noise(
    *, seed: int, frames: int, variant_names: Sequence[str]
) -> dict[str, np.ndarray]:
    """Return bit-identical copies for a future frozen/LoRA A/B comparison."""

    names = tuple(str(value) for value in variant_names)
    if len(names) < 2 or len(set(names)) != len(names):
        raise EmotionHierarchyVideoError(
            "variant noise requires at least two unique variant names"
        )
    noise = base.shared_initial_noise(seed=int(seed), frames=int(frames))
    return {name: noise.copy() for name in names}


def _validate_frozen_audit(value: object, *, field: str) -> dict[str, Any]:
    audit = _require_mapping(value, field=field)
    try:
        maximum_error = float(audit.get("maximum_abs_error", math.inf))
    except (TypeError, ValueError) as error:
        raise EmotionHierarchyVideoError(
            f"{field} maximum_abs_error is invalid"
        ) from error
    changed = audit.get("changed_frozen_tensor_names")
    if (
        audit.get("passed") is not True
        or changed != []
        or maximum_error != 0.0
    ):
        raise EmotionHierarchyVideoError(
            f"{field} did not pass the exact frozen-parameter audit"
        )
    return dict(audit)


def _validate_hierarchy_receipt(
    semantic: object, *, expected_qwen_cache_sha256: str
) -> None:
    receipt = _require_mapping(
        semantic, field="training_contract.semantic_perceptual"
    )
    hierarchy = _require_mapping(
        receipt.get("hierarchy"),
        field="training_contract.semantic_perceptual.hierarchy",
    )
    primary_counts = hierarchy.get("primary_prototype_counts")
    prototype_metadata = hierarchy.get("prototype_metadata")
    supervision = receipt.get("emotion_supervision_ingress")
    if (
        receipt.get("enabled") is not True
        or receipt.get("no_external_data") is not True
        or receipt.get("no_kimodo") is not True
        or receipt.get("label_role") != INTENDED_WEAK_LABEL_ROLE
        or receipt.get("human_perceived_emotion_truth") is not False
        or receipt.get("formal_emotion_supervision_enabled") is not False
        or int(receipt.get("human_confirmed_observable_rows", -1)) != 0
        or float(receipt.get("weak_label_weight", -1.0)) != 0.1
        or receipt.get("schedule_mode")
        != "simultaneous_hierarchy_no_stage_schedule_v1"
        or receipt.get("prototype_fit_split") != "train"
        or int(
            receipt.get(
                "validation_or_test_rows_used_for_prototypes", -1
            )
        )
        != 0
        or not isinstance(primary_counts, Mapping)
        or dict(primary_counts) != {"binary": 2, "emotion": 6}
        or int(hierarchy.get("auxiliary_group_prototype_count", -1)) != 54
        or not isinstance(prototype_metadata, Mapping)
        or int(prototype_metadata.get("emotion_count", -1)) != 6
        or int(prototype_metadata.get("group_count", -1)) != 54
        or prototype_metadata.get("fit_split") != "train"
        or int(
            prototype_metadata.get(
                "validation_or_test_rows_used", -1
            )
        )
        != 0
        or not isinstance(supervision, Mapping)
        or int(supervision.get("human_confirmed_observable_rows", -1))
        != 0
        or (receipt.get("global_prototype_source_qwen_cache") or {}).get(
            "sha256"
        )
        != expected_qwen_cache_sha256
    ):
        raise EmotionHierarchyVideoError(
            "checkpoint six-emotion hierarchy receipt is incomplete"
        )


def _load_prompt_latents_for_variant(
    cache_path: Path,
    *,
    prompts: Sequence[str],
    manifest_records: Mapping[str, Mapping[str, Any]],
    manifest_sha256: str,
    variant: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cache_latents, metadata = abc_video.load_qwen_cache(
        cache_path,
        variant=variant,
        manifest_records=manifest_records,
        manifest_sha256=manifest_sha256,
    )
    try:
        with np.load(cache_path, allow_pickle=False) as payload:
            clip_ids = payload["clip_ids"].astype(str)
            cache_prompts = payload["prompts"].astype(str)
            splits = payload["fixed_split_assignments"].astype(str)
    except (OSError, ValueError, KeyError) as error:
        raise EmotionHierarchyVideoError(
            f"cannot read {variant} prompt identities: {error}"
        ) from error
    if (
        clip_ids.shape != cache_prompts.shape
        or clip_ids.shape != splits.shape
        or len(set(clip_ids.tolist())) != len(clip_ids)
    ):
        raise EmotionHierarchyVideoError(
            f"{variant} prompt identity arrays are invalid"
        )
    result: dict[str, np.ndarray] = {}
    prompt_receipts: list[dict[str, Any]] = []
    for prompt in sorted(set(str(value) for value in prompts)):
        indices = np.flatnonzero(cache_prompts == prompt)
        test_indices = [
            int(index) for index in indices if splits[int(index)] == "test"
        ]
        if not test_indices:
            raise EmotionHierarchyVideoError(
                f"prompt is not a held-out canonical BEAT2 prompt: {prompt}"
            )
        representative = test_indices[0]
        latent = np.asarray(
            cache_latents[str(clip_ids[representative])],
            dtype=np.float32,
        )
        if latent.shape != (TEXT_LATENT_DIM,) or not np.isfinite(
            latent
        ).all():
            raise EmotionHierarchyVideoError(
                f"{variant} canonical Qwen latent is invalid"
            )
        if any(
            not np.array_equal(
                np.asarray(
                    cache_latents[str(clip_ids[int(index)])],
                    dtype=np.float32,
                ),
                latent,
            )
            for index in indices
        ):
            raise EmotionHierarchyVideoError(
                f"same canonical prompt has multiple {variant} latents: "
                f"{prompt}"
            )
        result[prompt] = latent.copy()
        prompt_receipts.append(
            {
                "prompt": prompt,
                "cache_row_count": int(len(indices)),
                "held_out_test_row_count": len(test_indices),
                "representative_test_clip_id": str(
                    clip_ids[representative]
                ),
                "latent_sha256": hashlib.sha256(
                    latent.tobytes()
                ).hexdigest(),
                "latent_l2": float(np.linalg.norm(latent)),
                "all_prompt_rows_exactly_equal": True,
            }
        )
    return result, {
        "path": str(cache_path),
        "sha256": abc_video.sha256_file(cache_path),
        "artifact_kind": metadata["artifact_kind"],
        "variant": variant,
        "data_policy": metadata["data_policy"],
        "source_manifest_sha256": metadata["source_manifest_sha256"],
        "prompt_receipts": prompt_receipts,
        "offline_cache_lookup_only": True,
        "network_or_model_download": False,
    }


def _comparison_interface(config: Mapping[str, Any]) -> dict[str, Any]:
    interface = _require_mapping(
        config.get("qwen_generator_comparison"),
        field="qwen_generator_comparison",
    )
    required = {
        "ab_contract_receipt",
        "ab_contract_receipt_sha256",
        "artifact_kind",
        "active_variant",
        "shared_noise_policy",
        "same_comparison_pair_seeds",
        "variants",
    }
    if set(interface) != required:
        raise EmotionHierarchyVideoError(
            "qwen_generator_comparison fields changed"
        )
    variants = _require_mapping(
        interface.get("variants"),
        field="qwen_generator_comparison.variants",
    )
    if (
        interface.get("artifact_kind") != COMPARISON_INTERFACE_KIND
        or interface.get("shared_noise_policy") != SHARED_NOISE_POLICY
        or interface.get("same_comparison_pair_seeds") is not True
        or set(variants) != {"frozen_base", "lora_finetuned"}
        or not _is_sha256(interface.get("ab_contract_receipt_sha256"))
    ):
        raise EmotionHierarchyVideoError(
            "frozen/LoRA same-noise comparison interface is invalid"
        )
    receipt_path = base._require_file(
        base._resolve_path(
            interface.get("ab_contract_receipt"),
            config_dir=PROJECT_ROOT,
            field="qwen_generator_comparison.ab_contract_receipt",
        ),
        "qwen_generator_comparison.ab_contract_receipt",
    )
    if (
        abc_video.sha256_file(receipt_path)
        != interface["ab_contract_receipt_sha256"]
    ):
        raise EmotionHierarchyVideoError(
            "frozen/LoRA A/B contract receipt changed"
        )
    ab_receipt = abc_video.load_json(receipt_path)
    claims = ab_receipt.get("claims")
    receipt_arms = ab_receipt.get("arms")
    if (
        ab_receipt.get("schema_version") != SCHEMA_VERSION
        or ab_receipt.get("artifact_kind")
        != "beat2_emotion_hierarchy_qwen_generator_ab_contract_receipt_v7"
        or ab_receipt.get("status")
        != "contract_validated_training_not_started"
        or ab_receipt.get("data_policy") != DATA_POLICY
        or ab_receipt.get("no_external_data") is not True
        or ab_receipt.get("no_kimodo") is not True
        or not isinstance(claims, Mapping)
        or claims.get("generator_training_completed") is not False
        or claims.get("motion_quality_success_claimed") is not False
        or claims.get("robot_emotion_success_claimed") is not False
        or not isinstance(receipt_arms, Mapping)
        or set(receipt_arms) != {"frozen_base", "lora_finetuned"}
        or (ab_receipt.get("shared_contract") or {}).get(
            "same_noise_and_flow_time_implementation"
        )
        is not True
    ):
        raise EmotionHierarchyVideoError(
            "frozen/LoRA A/B contract receipt is invalid"
        )
    active_name = str(interface.get("active_variant", ""))
    if active_name not in variants:
        raise EmotionHierarchyVideoError(
            "active Qwen/generator variant is unknown"
        )
    normalized = {}
    for name, raw in variants.items():
        variant = _require_mapping(
            raw, field=f"qwen_generator_comparison.variants.{name}"
        )
        expected_fields = {
            "condition_cache",
            "condition_cache_sha256",
            "condition_variant",
            "generation_enabled",
            "generator_checkpoint",
            "generator_training_summary",
            "expected_training_policy",
        }
        if set(variant) != expected_fields:
            raise EmotionHierarchyVideoError(
                f"{name} comparison variant fields changed"
            )
        expected_condition_variant = (
            "frozen_base" if name == "frozen_base" else "lora_finetuned"
        )
        if (
            variant.get("condition_variant")
            != expected_condition_variant
            or not _is_sha256(variant.get("condition_cache_sha256"))
            or not isinstance(variant.get("generation_enabled"), bool)
        ):
            raise EmotionHierarchyVideoError(
                f"{name} comparison variant contract is invalid"
            )
        cache_path = base._resolve_path(
            variant.get("condition_cache"),
            config_dir=PROJECT_ROOT,
            field=f"qwen_generator_comparison.variants.{name}.condition_cache",
        )
        if not cache_path.is_file() or (
            abc_video.sha256_file(cache_path)
            != variant["condition_cache_sha256"]
        ):
            raise EmotionHierarchyVideoError(
                f"{name} condition cache is missing or changed"
            )
        enabled = bool(variant["generation_enabled"])
        checkpoint = variant.get("generator_checkpoint")
        training_summary = variant.get("generator_training_summary")
        expected_policy = variant.get("expected_training_policy")
        if enabled:
            if (
                not isinstance(checkpoint, str)
                or not checkpoint
                or not isinstance(training_summary, str)
                or not training_summary
                or not isinstance(expected_policy, str)
                or not expected_policy
            ):
                raise EmotionHierarchyVideoError(
                    f"{name} is marked available without real artifacts"
                )
            checkpoint_path = base._require_file(
                base._resolve_path(
                    checkpoint,
                    config_dir=PROJECT_ROOT,
                    field=f"{name}.generator_checkpoint",
                ),
                f"{name}.generator_checkpoint",
            )
            summary_path = base._require_file(
                base._resolve_path(
                    training_summary,
                    config_dir=PROJECT_ROOT,
                    field=f"{name}.generator_training_summary",
                ),
                f"{name}.generator_training_summary",
            )
        else:
            if (
                checkpoint is not None
                or training_summary is not None
                or expected_policy is not None
            ):
                raise EmotionHierarchyVideoError(
                    f"{name} unavailable branch must not contain fake "
                    "generator artifacts"
                )
            checkpoint_path = None
            summary_path = None
        normalized[name] = {
            **dict(variant),
            "condition_cache": str(cache_path),
            "generator_checkpoint": (
                str(checkpoint_path) if checkpoint_path is not None else None
            ),
            "generator_training_summary": (
                str(summary_path) if summary_path is not None else None
            ),
        }
        receipt_arm = receipt_arms[name]
        if (
            not isinstance(receipt_arm, Mapping)
            or str(Path(str(receipt_arm.get("condition_cache"))).resolve())
            != str(cache_path)
            or receipt_arm.get("condition_cache_sha256")
            != variant["condition_cache_sha256"]
        ):
            raise EmotionHierarchyVideoError(
                f"{name} differs from the sealed A/B contract receipt"
            )
    active = normalized[active_name]
    if active["generation_enabled"] is not True:
        raise EmotionHierarchyVideoError(
            f"{active_name} is not enabled for gated generation"
        )
    return {
        **dict(interface),
        "ab_contract_receipt": str(receipt_path),
        "variants": normalized,
        "active_variant": active_name,
        "active": active,
    }


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable V7 video contract without loading the model."""

    if (
        config.get("schema_version") != SCHEMA_VERSION
        or config.get("artifact_kind") != CONFIG_ARTIFACT_KIND
        or config.get("data_policy") != DATA_POLICY
        or config.get("no_external_data") is not True
        or config.get("no_kimodo") is not True
        or config.get("playback_contract") != PLAYBACK_CONTRACT
        or int(config.get("minimum_training_steps", 0)) < 80_000
        or not math.isclose(
            float(config.get("target_duration_sec", 0.0)),
            60.0,
            abs_tol=1e-9,
        )
    ):
        raise EmotionHierarchyVideoError(
            "V7 qualitative video config contract is invalid"
        )
    interface = _comparison_interface(config)
    active = interface["active"]
    config_dir = PROJECT_ROOT
    top_paths = {
        "qwen_condition_cache": base._resolve_path(
            config.get("qwen_condition_cache"),
            config_dir=config_dir,
            field="qwen_condition_cache",
        ),
        "generator_checkpoint": base._resolve_path(
            config.get("generator_checkpoint"),
            config_dir=config_dir,
            field="generator_checkpoint",
        ),
        "generator_training_summary": base._resolve_path(
            config.get("generator_training_summary"),
            config_dir=config_dir,
            field="generator_training_summary",
        ),
    }
    if (
        str(top_paths["qwen_condition_cache"])
        != active["condition_cache"]
        or str(top_paths["generator_checkpoint"])
        != active["generator_checkpoint"]
        or str(top_paths["generator_training_summary"])
        != active["generator_training_summary"]
    ):
        raise EmotionHierarchyVideoError(
            "top-level artifacts do not match the active comparison variant"
        )
    gate = _require_mapping(
        config.get("six_emotion_full_diagnostic_gate"),
        field="six_emotion_full_diagnostic_gate",
    )
    # Validate the threshold schema using an intentionally improving fixture.
    synthetic_initial = {
        "episode_count": 54,
        "semantic_global_prototype_count": 6,
        "semantic_global_contrastive_loss": 1.0,
        "semantic_global_mean_positive_rank": 3.0,
        "semantic_global_hard_cross_group_margin": -0.5,
    }
    synthetic_final = {
        **synthetic_initial,
        "semantic_global_contrastive_loss": 0.0,
        "semantic_global_mean_positive_rank": 1.0,
        "semantic_global_hard_cross_group_margin": 0.5,
    }
    six_emotion_full_diagnostic_decision(
        synthetic_initial, synthetic_final, gate
    )
    pairs = config.get("comparison_pairs")
    if not isinstance(pairs, list):
        raise EmotionHierarchyVideoError(
            "comparison_pairs must be a list"
        )
    emotion_text = {
        token
        for token in EXPECTED_EMOTION_PROMPT_TOKENS
        if any(
            token in str(pair.get("emotion_prompt", "")).lower()
            for pair in pairs
            if isinstance(pair, Mapping)
        )
    }
    if emotion_text != EXPECTED_EMOTION_PROMPT_TOKENS:
        raise EmotionHierarchyVideoError(
            "60-second comparison must cover all five non-neutral emotions"
        )
    return {
        **deepcopy(dict(config)),
        "_validated_comparison_interface": interface,
    }


def _strict_v7_checkpoint_admission(
    checkpoint_path: Path,
    training_summary_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_qwen_cache_sha256: str,
    minimum_training_steps: int,
    expected_training_policy: str,
    gate_config: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise EmotionHierarchyVideoError(
            f"cannot load V7 checkpoint for admission: {error}"
        ) from error
    if not isinstance(checkpoint, Mapping):
        raise EmotionHierarchyVideoError("V7 checkpoint is not a mapping")
    summary = abc_video.load_json(training_summary_path)
    training_contract = _require_mapping(
        checkpoint.get("training_contract"), field="training_contract"
    )
    data_receipt = _require_mapping(
        checkpoint.get("data_receipt"), field="data_receipt"
    )
    metrics = _require_mapping(checkpoint.get("metrics"), field="metrics")
    initial = _require_mapping(
        metrics.get("initial_condition_diagnostics"),
        field="metrics.initial_condition_diagnostics",
    )
    final = _require_mapping(
        metrics.get("final_condition_diagnostics"),
        field="metrics.final_condition_diagnostics",
    )
    gate = _require_mapping(
        checkpoint.get("anti_collapse_gate"),
        field="anti_collapse_gate",
    )
    checkpoint_audit = _validate_frozen_audit(
        checkpoint.get("frozen_parameter_audit"),
        field="checkpoint.frozen_parameter_audit",
    )
    summary_audit = _validate_frozen_audit(
        summary.get("frozen_parameter_audit"),
        field="summary.frozen_parameter_audit",
    )
    checkpoint_sha256 = abc_video.sha256_file(checkpoint_path)
    try:
        summary_checkpoint_path = Path(
            str(summary.get("checkpoint"))
        ).resolve()
    except (TypeError, ValueError):
        summary_checkpoint_path = Path()
    if (
        checkpoint.get("schema_version") != SCHEMA_VERSION
        or checkpoint.get("artifact_kind") != CHECKPOINT_ARTIFACT_KIND
        or checkpoint.get("experimental_only") is not True
        or checkpoint.get("formal_release_eligible") is not False
        or checkpoint.get("no_external_data") is not True
        or checkpoint.get("no_kimodo") is not True
        or data_receipt.get("no_external_data") is not True
        or data_receipt.get("no_kimodo") is not True
        or (data_receipt.get("manifest") or {}).get("sha256")
        != expected_manifest_sha256
        or (data_receipt.get("frozen_qwen_cache") or {}).get("sha256")
        != expected_qwen_cache_sha256
        or training_contract.get("policy") != expected_training_policy
        or int(training_contract.get("steps", 0))
        < max(80_000, int(minimum_training_steps))
        or int(
            training_contract.get("external_motion_checkpoint_count", -1)
        )
        != 0
        or gate.get("passed") is not True
        or summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("artifact_kind")
        != TRAINING_SUMMARY_ARTIFACT_KIND
        or summary.get("status") != "experimental_candidate"
        or summary.get("checkpoint_sha256") != checkpoint_sha256
        or summary_checkpoint_path != checkpoint_path.resolve()
        or int(summary.get("completed_steps", -1))
        != int(summary.get("target_steps", -2))
        or int(summary.get("completed_steps", 0))
        < max(80_000, int(minimum_training_steps))
        or summary.get("anti_collapse_gate") != gate
        or summary.get("no_external_data") is not True
        or summary.get("no_kimodo") is not True
        or checkpoint_audit != summary_audit
        or summary.get("initial_condition_diagnostics") != initial
        or summary.get("final_condition_diagnostics") != final
    ):
        raise EmotionHierarchyVideoError(
            "V7 checkpoint/summary is incomplete, rejected, or not "
            "BEAT2-only"
        )
    _validate_hierarchy_receipt(
        training_contract.get("semantic_perceptual"),
        expected_qwen_cache_sha256=expected_qwen_cache_sha256,
    )
    hierarchy_gate = six_emotion_full_diagnostic_decision(
        initial, final, gate_config
    )
    if hierarchy_gate["passed"] is not True:
        raise EmotionHierarchyVideoError(
            "V7 full six-emotion diagnostic did not improve: "
            + ", ".join(hierarchy_gate["failure_reasons"])
        )
    return {
        "frozen_parameter_audit": checkpoint_audit,
        "existing_anti_collapse_gate": deepcopy(dict(gate)),
        "six_emotion_full_diagnostic_gate": hierarchy_gate,
        "training_policy": expected_training_policy,
    }


@contextmanager
def _patched_v7_video_engine(
    config: Mapping[str, Any],
    comparison: Mapping[str, Any],
):
    active = comparison["active"]
    active_variant = str(active["condition_variant"])
    expected_policy = str(active["expected_training_policy"])
    gate_config = config["six_emotion_full_diagnostic_gate"]
    original_loader = base._load_v2_checkpoint
    original_prompt_loader = base._load_prompt_latents
    original_ass_builder = base.build_ass_document

    def v7_loader(
        checkpoint_path: Path,
        training_summary_path: Path,
        *,
        expected_manifest_sha256: str,
        expected_qwen_cache_sha256: str,
        minimum_training_steps: int,
        device: torch.device,
    ):
        if expected_qwen_cache_sha256 != active["condition_cache_sha256"]:
            raise EmotionHierarchyVideoError(
                "active Qwen cache SHA changed before checkpoint admission"
            )
        admission = _strict_v7_checkpoint_admission(
            checkpoint_path,
            training_summary_path,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_qwen_cache_sha256=expected_qwen_cache_sha256,
            minimum_training_steps=minimum_training_steps,
            expected_training_policy=expected_policy,
            gate_config=gate_config,
        )
        model, style_head, receipt = original_loader(
            checkpoint_path,
            training_summary_path,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_qwen_cache_sha256=expected_qwen_cache_sha256,
            minimum_training_steps=minimum_training_steps,
            device=device,
        )
        receipt.update(admission)
        receipt["qwen_condition_variant"] = active_variant
        return model, style_head, receipt

    def v7_prompt_loader(
        cache_path: Path,
        *,
        prompts: Sequence[str],
        manifest_records: Mapping[str, Mapping[str, Any]],
        manifest_sha256: str,
    ):
        return _load_prompt_latents_for_variant(
            cache_path,
            prompts=prompts,
            manifest_records=manifest_records,
            manifest_sha256=manifest_sha256,
            variant=active_variant,
        )

    def v7_ass_builder(*args, **kwargs):
        document = original_ass_builder(*args, **kwargs)
        document = document.replace(
            "BEAT2 V2 · TEXT → QWEN 128D → PREDICTED STYLE → AdaLN",
            (
                "BEAT2 V7 · SIX-EMOTION HIERARCHY · "
                f"QWEN {active_variant.upper()} → AdaLN"
            ),
        )
        document = document.replace(
            "CANONICAL 54-GROUP BEAT2 METADATA · OPEN TEXT UNVALIDATED",
            (
                "FIXED 54-EXAMPLE / 6-PROTOTYPE GATE PASSED · "
                "INTENDED WEAK LABELS"
            ),
        )
        return document

    replacements = {
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "CONFIG_ARTIFACT_KIND": CONFIG_ARTIFACT_KIND,
        "SUMMARY_ARTIFACT_KIND": SUMMARY_ARTIFACT_KIND,
        "CHECKPOINT_ARTIFACT_KIND": CHECKPOINT_ARTIFACT_KIND,
        "CHECKPOINT_SUMMARY_ARTIFACT_KIND": (
            TRAINING_SUMMARY_ARTIFACT_KIND
        ),
        "EXPECTED_TRAINING_POLICY": expected_policy,
        "OUTPUT_SUMMARY_FILENAME": OUTPUT_FILENAMES["summary"],
        "OUTPUT_TRAJECTORY_FILENAME": OUTPUT_FILENAMES["trajectory"],
        "OUTPUT_NEUTRAL_CSV_FILENAME": OUTPUT_FILENAMES["neutral_csv"],
        "OUTPUT_NEUTRAL_VIDEO_FILENAME": OUTPUT_FILENAMES["neutral_video"],
        "OUTPUT_EMOTION_CSV_FILENAME": OUTPUT_FILENAMES["emotion_csv"],
        "OUTPUT_EMOTION_VIDEO_FILENAME": OUTPUT_FILENAMES["emotion_video"],
        "OUTPUT_ASS_FILENAME": OUTPUT_FILENAMES["ass"],
        "OUTPUT_FINAL_VIDEO_FILENAME": OUTPUT_FILENAMES["final_video"],
        "_load_v2_checkpoint": v7_loader,
        "_load_prompt_latents": v7_prompt_loader,
        "build_ass_document": v7_ass_builder,
    }
    originals = {name: getattr(base, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(base, name, value)


def build_video(
    config_path: str | Path, *, overwrite: bool = False
) -> dict[str, Any]:
    """Build one admitted branch; both branches use identical configured seeds."""

    config_path = Path(config_path).resolve()
    raw_config = abc_video.load_json(config_path)
    validated = validate_config(raw_config)
    comparison = validated.pop("_validated_comparison_interface")
    with _patched_v7_video_engine(validated, comparison):
        summary = base.build_video(config_path, overwrite=overwrite)
    summary["qwen_generator_comparison"] = {
        "artifact_kind": comparison["artifact_kind"],
        "active_variant": comparison["active_variant"],
        "shared_noise_policy": comparison["shared_noise_policy"],
        "same_comparison_pair_seeds": True,
        "future_lora_generation_enabled": bool(
            comparison["variants"]["lora_finetuned"][
                "generation_enabled"
            ]
        ),
        "future_lora_checkpoint_fabricated": False,
    }
    summary["playback_contract"] = deepcopy(PLAYBACK_CONTRACT)
    summary_path = (
        Path(validated["output_dir"]).resolve()
        / OUTPUT_FILENAMES["summary"]
    )
    abc_video.atomic_json(summary_path, summary)
    return summary


def wait_for_completion_and_build(
    config_path: str | Path,
    *,
    overwrite: bool = False,
    poll_seconds: float = 30.0,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Poll the V7 summary; generate only after a real admitted completion."""

    config_path = Path(config_path).resolve()
    config = abc_video.load_json(config_path)
    if (
        config.get("schema_version") != SCHEMA_VERSION
        or config.get("artifact_kind") != CONFIG_ARTIFACT_KIND
        or config.get("data_policy") != DATA_POLICY
        or config.get("no_external_data") is not True
        or config.get("no_kimodo") is not True
    ):
        raise EmotionHierarchyVideoError(
            "cannot wait on an invalid V7 video config"
        )
    if (
        not math.isfinite(float(poll_seconds))
        or float(poll_seconds) <= 0
        or (
            timeout_seconds is not None
            and (
                not math.isfinite(float(timeout_seconds))
                or float(timeout_seconds) <= 0
            )
        )
    ):
        raise EmotionHierarchyVideoError(
            "poll and optional timeout seconds must be positive"
        )
    summary_path = base._resolve_path(
        config.get("generator_training_summary"),
        config_dir=config_path.parent,
        field="generator_training_summary",
    )
    checkpoint_path = base._resolve_path(
        config.get("generator_checkpoint"),
        config_dir=config_path.parent,
        field="generator_checkpoint",
    )
    started = time.monotonic()
    while True:
        if summary_path.is_file():
            training_summary = abc_video.load_json(summary_path)
            status = training_summary.get("status")
            if status in {
                "rejected_condition_collapse_or_quality_gate",
                "smoke_complete",
            }:
                raise EmotionHierarchyVideoError(
                    f"V7 training ended without an admissible artifact: "
                    f"{status}"
                )
            if status == "experimental_candidate":
                if not checkpoint_path.is_file():
                    raise EmotionHierarchyVideoError(
                        "V7 summary claims a candidate but checkpoint is missing"
                    )
                return build_video(config_path, overwrite=overwrite)
        if (
            timeout_seconds is not None
            and time.monotonic() - started >= float(timeout_seconds)
        ):
            raise EmotionHierarchyVideoError(
                "timed out before an admissible V7 checkpoint was completed"
            )
        time.sleep(float(poll_seconds))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "beat2_emotion_hierarchy_v7_60s.json"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--wait-for-completion", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.wait_for_completion:
            summary = wait_for_completion_and_build(
                args.config,
                overwrite=args.overwrite,
                poll_seconds=args.poll_seconds,
                timeout_seconds=args.timeout_seconds,
            )
        else:
            summary = build_video(args.config, overwrite=args.overwrite)
    except (
        EmotionHierarchyVideoError,
        base.StyleEmotionVideoError,
        abc_video.EvaluationContractError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": summary["status"],
                "artifact_kind": summary["artifact_kind"],
                "active_variant": summary["qwen_generator_comparison"][
                    "active_variant"
                ],
                "video": summary["artifacts"]["final_video"]["path"],
                "duration_sec": summary["artifacts"]["final_video"][
                    "duration_sec"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
