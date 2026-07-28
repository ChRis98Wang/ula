"""Versioned 18D head-adapter contract for the ULA MMDiT V2 generator.

The 18D contract is append-only: the original 15 robot joints retain their
order and weights, while head roll, pitch, and yaw occupy the last three
channels.  The default adaptation policy updates only the newly added slices
of the input and output projections.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from upper_body_skeleton.data_source_registry import (
    DATA_SOURCE_REGISTRY_HASH_FIELD,
    GENERATOR_FOUNDATION_ROLE,
    assert_no_forbidden_data_lineage,
    validate_contract_source_binding,
    validate_data_source_registry_contract,
)
from upper_body_skeleton.kimodo_semantics import (
    KIMODO_BEHAVIOR_FAMILIES,
    KIMODO_BEHAVIOR_IDS,
    KIMODO_EMOTION_IDS,
    kimodo_behavior_family,
)
from upper_body_skeleton.retarget_v2 import JOINT_ORDER
from upper_body_skeleton.retarget_v2_18d import (
    CONTRACT_VERSION,
    JOINT_LIMITS_18D,
    JOINT_ORDER_18D,
    joint_order_for_action_dim,
)
from upper_body_skeleton.ula_training import (
    AFFECT_IDS,
    GESTURE_IDS,
    INTENT_IDS,
    KIMODO_CONDITION_DIM,
    KIMODO_V2_CONDITION_DIM,
    LEGACY_CONDITION_DIM,
    STYLE_IDS,
    ULA_MMDIT_V2_ARCHITECTURE,
    ULA_MMDIT_V3_ADALN_ARCHITECTURE,
    build_condition_from_text,
    create_ula_model,
    sample_span_to_frame_count,
)
from upper_body_skeleton.ula_v2_conditioning import (
    extract_style_features,
    normalize_style_features,
)


LEGACY_ACTION_DIM = len(JOINT_ORDER)
ACTION_DIM = len(JOINT_ORDER_18D)
HEAD_SLICE = slice(LEGACY_ACTION_DIM, ACTION_DIM)
CHECKPOINT_SCHEMA_VERSION = 2
ARTIFACT_KIND = "ula_mmdit_v2_generator"
SUPPORTED_GENERATOR_ARCHITECTURES = frozenset(
    {ULA_MMDIT_V2_ARCHITECTURE, ULA_MMDIT_V3_ADALN_ARCHITECTURE}
)
ADAPTER_POLICY = "new_input_columns_and_output_rows_only_v1"
LEGACY_FORWARD_ATOL = 1e-5
BODY_DISTILLATION_WEIGHT = 1.0
BODY_COMPATIBILITY_THRESHOLDS = {
    "nonzero_head_forward_mean_abs_normalized": 0.08,
    "nonzero_head_forward_p95_abs_normalized": 0.25,
    "sampling_body_mean_abs_rad": 0.02,
    "sampling_body_p95_abs_rad": 0.05,
    "sampling_body_max_abs_rad": 0.15,
}
DEFAULT_BASE_CHECKPOINT = Path(
    "training/runs/kimodo_ula_v2_lora_50k/ula_fm_checkpoint.pt"
)
DEFAULT_QWEN_CHECKPOINT = Path(
    "training/runs/kimodo_qwen_motion_latent_lora_v2/best.pt"
)
CONDITION_CACHE_SCHEMA_VERSION = 3
MOTION_ONLY_CONDITION_CACHE_SCHEMA_VERSION = 1
MOTION_ONLY_CONDITION_CACHE_ARTIFACT_KIND = (
    "ula_v2_18d_motion_only_style_condition_cache"
)
MOTION_ONLY_NO_QWEN_POLICY = "disabled_not_configured_not_loaded_v1"
MOTION_ONLY_NO_KIMODO_POLICY = (
    "forbidden_dataset_checkpoint_replay_and_condition_channels_v1"
)
MOTION_ONLY_RANDOM_INIT_MODE = "full_generator_random_no_qwen_no_kimodo_v1"
MOTION_ONLY_STYLE_ONLY_CONDITION_POLICY = (
    "trajectory_style_indices_133_136_only_all_other_dimensions_exact_zero_v1"
)
MOTION_ONLY_DATA_ISOLATION_CONTRACT_TYPE = (
    "ula_v2_18d_beat2_only_no_kimodo_no_qwen"
)
FORMAL_ADJUDICATION_SCHEMA_VERSION = "1.2.0"
FORMAL_RELEASE_REPORT_FILENAME = "dataset_scale_report.json"
MOTION_ONLY_EPISODE_CONTRACT = "ula_v2_18d_motion_only_physical_qc_v1"
MOTION_ONLY_RELEASE_REPORT_FILENAME = "motion_only_release_report.json"
FORMAL_VARIABLE_SEGMENT_REPRESENTATION = "native_variable_length_semantic_clip_v1"
FORMAL_RETARGET_SEGMENT_REPRESENTATION = (
    "native_variable_length_semantic_event_retimed_30hz_v1"
)
FORMAL_SELECTED_LINEAGE_FIELDS = (
    "upstream_inventory_record_sha256",
    "selected_record_sha256",
    "inventory_manifest_sha256",
    "pilot_selector_contract_sha256",
    "pilot_speaker_group_sha256",
    "pilot_source_group_sha256",
    "prompt_sha256",
)
FORMAL_REQUIRED_18D_GATES = frozenset(
    {
        "joint_limits_pass",
        "velocity_pass",
        "target_fit_pass",
        "collision_pass",
        "axis_direction_pass",
        "head_joint_limits_pass",
        "head_velocity_pass",
        "head_direction_pass",
        "head_continuity_pass",
        "passed",
    }
)
FORMAL_REQUIRED_REVIEW_GATES = frozenset(
    {
        "action_recognizable",
        "text_consistent",
        "observable_in_18d",
        "context_available",
        "physical_qc",
        "subject_action_split_safe",
        "affect_observable_in_18d",
    }
)
FORMAL_MOTION_ADMISSION_REVIEW_GATES = FORMAL_REQUIRED_REVIEW_GATES - {
    "affect_observable_in_18d"
}
FORMAL_REQUIRED_RELEASE_INVARIANTS = frozenset(
    {
        "every_semantic_record_adjudicated_once",
        "train_ready_requires_independent_acceptance_and_18d_qc",
        "unknown_subject_never_eval",
        "failed_18d_qc_always_rejected",
        "semantic_critical_always_rejected",
        "independent_review_rejected_always_rejected",
        "emotion_train_ready_requires_motion_and_explicit_supervision",
        "unverified_affect_never_emotion_conditioned",
        "official_category_metadata_only_never_conditioned",
        "unresolved_emotion_never_train_ready",
        "behavior_one_hot_requires_explicit_human_confirmation",
        "project_weak_behavior_never_one_hot_supervised",
    }
)
MOTION_ONLY_REQUIRED_RELEASE_INVARIANTS = frozenset(
    {
        "every_record_passes_18d_physical_qc",
        "all_semantic_supervision_masked",
        "all_affect_supervision_masked",
        "all_prompt_latents_masked_zero",
        "all_sequences_native_variable_length",
        "no_fixed_duration_training_unit",
        "no_semantic_review_claimed",
    }
)
POSTTRAIN_LINEAGE_ARTIFACT_KIND = "ula_mmdit_v2_18d_interaction_posttrain"
POSTTRAIN_LINEAGE_MODE = "low_lr_full_network_interaction_domain_posttrain"
POSTTRAIN_LINEAGE_MARKERS = frozenset(
    {
        "posttrain_artifact_kind",
        "posttrain_source",
        "posttrain_step",
        "posttrain_data_contract",
        "posttrain_split_contract",
        "posttrain_config",
    }
)
KIMODO_BEHAVIOR_SLICE = slice(
    LEGACY_CONDITION_DIM,
    LEGACY_CONDITION_DIM + len(KIMODO_BEHAVIOR_IDS),
)
KIMODO_EMOTION_SLICE = slice(
    KIMODO_BEHAVIOR_SLICE.stop,
    KIMODO_BEHAVIOR_SLICE.stop + len(KIMODO_EMOTION_IDS),
)
KIMODO_BEHAVIOR_FAMILY_SLICE = slice(
    KIMODO_EMOTION_SLICE.stop,
    KIMODO_EMOTION_SLICE.stop + len(KIMODO_BEHAVIOR_FAMILIES),
)
STYLE_CONTROL_SLICE = slice(KIMODO_CONDITION_DIM - 3, KIMODO_CONDITION_DIM)
LEGACY_INTENT_SLICE = slice(0, len(INTENT_IDS))
LEGACY_AFFECT_SLICE = slice(LEGACY_INTENT_SLICE.stop, LEGACY_INTENT_SLICE.stop + len(AFFECT_IDS))
LEGACY_STYLE_SLICE = slice(LEGACY_AFFECT_SLICE.stop, LEGACY_AFFECT_SLICE.stop + len(STYLE_IDS))
LEGACY_GESTURE_SLICE = slice(
    LEGACY_STYLE_SLICE.stop,
    LEGACY_STYLE_SLICE.stop + len(GESTURE_IDS),
)
LEGACY_SEMANTIC_DEFAULTS = {
    "intent": "explaining",
    "observed_affect": "low_confidence_unknown",
    "motion_style": "restrained",
    "semantic_gesture": "upper_body_gesture",
}
FORMAL_SEMANTIC_SUPERVISION_MASKS = {
    "official_category": False,
    "robot_observable_motion_form": False,
    "communicative_intent": False,
    "prompt_text": False,
    "legacy_gesture": False,
}
OFFICIAL_CATEGORY_CONDITIONING_ROLE = (
    "verified_metadata_split_and_evaluation_only"
)
OFFICIAL_CATEGORY_CONDITIONING_ENABLED = False
OFFICIAL_CATEGORY_CONDITION_CHANNEL = None
OFFICIAL_CATEGORY_LOSS = None
OFFICIAL_EMOTION_DISABLED_ROLE = "disabled_pending_robot_affect_review"
OFFICIAL_EMOTION_ENABLED_ROLE = (
    "enabled_verified_robot_affect_observable_in_18d"
)
OFFICIAL_EMOTION_CONDITION_CHANNEL = (
    "kimodo_emotion_one_hot_and_legacy_affect"
)
BLIND_AFFECT_PROTOCOL_VERSION = "robot_affect_blind_video_v1"
BLIND_AFFECT_REQUIRED_BLINDED_FIELDS = frozenset(
    {
        "audio",
        "canonical_prompt",
        "official_emotion_label",
        "official_gesture_category",
        "source_text",
    }
)
MIN_BLIND_AFFECT_CONFIDENCE = 0.7
AFFECT_OBSERVABLE_REVIEW_STATUSES = frozenset(
    {
        "candidate_unreviewed",
        "verified",
        "not_verified",
        "rejected",
        "legacy_not_applicable",
    }
)
MOTION_STYLE_TO_LEGACY = {
    "energetic": "energetic",
    "sharp": "energetic",
    "relaxed": "relaxed",
    "restrained": "restrained",
    "slow_safe": "restrained",
}
KIMODO_EMOTION_TO_LEGACY_AFFECT = {
    "neutral": "neutral",
    "sad": "sad_like",
    "happy": "friendly",
    "angry": "angry_like",
    "surprise": "excited",
    "fear": "nervous",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(payload, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json_save(payload, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _checkpoint_model_shape(checkpoint: Mapping) -> dict:
    config = checkpoint.get("config") or {}
    return {
        "hidden_dim": int(config.get("hidden_dim", 384)),
        "layers": int(config.get("layers", 6)),
        "semantic_tokens": int(config.get("semantic_tokens", 7)),
    }


def validate_motion_only_style_condition(
    condition,
    *,
    context: str = "motion-only condition",
) -> np.ndarray:
    """Require an exact-zero 264D condition outside trajectory style controls."""
    value = np.asarray(condition, dtype=np.float32)
    if value.shape[-1:] != (KIMODO_V2_CONDITION_DIM,):
        raise ValueError(
            f"{context} must end in {KIMODO_V2_CONDITION_DIM} dimensions"
        )
    if not np.isfinite(value).all():
        raise ValueError(f"{context} contains non-finite values")
    if np.any(value[..., : STYLE_CONTROL_SLICE.start] != 0.0) or np.any(
        value[..., STYLE_CONTROL_SLICE.stop :] != 0.0
    ):
        raise ValueError(
            f"{context} must be exactly zero outside trajectory style indices "
            f"{STYLE_CONTROL_SLICE.start}:{STYLE_CONTROL_SLICE.stop}"
        )
    return value


def validate_motion_only_checkpoint_isolation(checkpoint: Mapping) -> dict:
    """Fail closed unless a motion-only checkpoint proves BEAT2-only isolation."""
    assert_no_forbidden_data_lineage(
        checkpoint, context="motion_only_checkpoint"
    )
    if checkpoint.get("formal_episode_contract") != MOTION_ONLY_EPISODE_CONTRACT:
        raise ValueError("checkpoint is not a motion-only formal generator")
    contracts = checkpoint.get("v2_contracts")
    if not isinstance(contracts, Mapping):
        raise ValueError("motion-only checkpoint lacks its aggregate contracts")
    isolation = contracts.get("data_isolation")
    expected = {
        "contract_type": MOTION_ONLY_DATA_ISOLATION_CONTRACT_TYPE,
        "contract_version": 1,
        "formal_episode_contract": MOTION_ONLY_EPISODE_CONTRACT,
        "dataset_family_whitelist": ["BEAT2"],
        "motion_source_policy": "hash_bound_beat2_manifests_only",
        "manifest_split_policy": "fixed_manifest_assignment_required",
        "qwen_policy": MOTION_ONLY_NO_QWEN_POLICY,
        "kimodo_policy": MOTION_ONLY_NO_KIMODO_POLICY,
        "generator_checkpoint_inputs": [],
        "condition_policy": MOTION_ONLY_STYLE_ONLY_CONDITION_POLICY,
        "condition_nonzero_indices": list(
            range(STYLE_CONTROL_SLICE.start, STYLE_CONTROL_SLICE.stop)
        ),
        "condition_exact_zero_ranges": [
            [0, STYLE_CONTROL_SLICE.start],
            [STYLE_CONTROL_SLICE.stop, KIMODO_V2_CONDITION_DIM],
        ],
    }
    if not isinstance(isolation, Mapping):
        raise ValueError("motion-only checkpoint lacks its data-isolation contract")
    if {key: isolation.get(key) for key in expected} != expected:
        raise ValueError("motion-only checkpoint data-isolation policy changed")
    if isolation.get("sha256") != _contract_sha256(isolation):
        raise ValueError("motion-only checkpoint data-isolation hash is invalid")

    split_contract = contracts.get("split")
    if not isinstance(split_contract, Mapping) or split_contract.get(
        "assignment_policy"
    ) != "fixed_pre_quarantine_assignment":
        raise ValueError("motion-only checkpoint must use fixed manifest split assignments")

    sources = checkpoint.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("motion-only checkpoint source contract is missing")
    forbidden_qwen_fields = {
        "qwen_checkpoint",
        "qwen_checkpoint_sha256",
        "qwen_model_name",
        "qwen_revision",
    }
    if (
        forbidden_qwen_fields.intersection(sources)
        or forbidden_qwen_fields.intersection(checkpoint)
        or forbidden_qwen_fields.intersection(checkpoint.get("config") or {})
    ):
        raise ValueError("motion-only checkpoint must not bind a Qwen checkpoint")
    text_contract = contracts.get("text_motion_latent")
    if (
        not isinstance(text_contract, Mapping)
        or text_contract.get("contract_type")
        != "ula_v2_reserved_zero_text_motion_latent"
        or text_contract.get("encoder_training_policy") != MOTION_ONLY_NO_QWEN_POLICY
        or "source" in text_contract
    ):
        raise ValueError("motion-only checkpoint text reserve is not Qwen-free")
    condition_contract = contracts.get("condition")
    if (
        not isinstance(condition_contract, Mapping)
        or condition_contract.get("motion_only_condition_policy")
        != MOTION_ONLY_STYLE_ONLY_CONDITION_POLICY
        or condition_contract.get("exact_zero_ranges")
        != expected["condition_exact_zero_ranges"]
        or condition_contract.get("data_isolation_contract_sha256")
        != isolation.get("sha256")
    ):
        raise ValueError("motion-only checkpoint condition isolation changed")
    source_records = sources.get("motion_manifests")
    if not isinstance(source_records, list) or not source_records:
        raise ValueError("motion-only checkpoint requires hash-bound BEAT2 manifests")
    for index, record in enumerate(source_records):
        gate = record.get("license_gate") if isinstance(record, Mapping) else None
        dataset_source = (
            str(record.get("dataset_source") or "").strip().lower()
            if isinstance(record, Mapping)
            else ""
        )
        if (
            not isinstance(record, Mapping)
            or not dataset_source.startswith(("beat2_", "beat2-"))
            or not isinstance(gate, Mapping)
            or str(gate.get("dataset_family") or "").strip().upper() != "BEAT2"
            or not _is_sha256(record.get("manifest_sha256"))
            or record.get("manifest_fixed_split") is not True
            or not _is_sha256(record.get("fixed_split_assignment_sha256"))
        ):
            raise ValueError(
                f"motion-only checkpoint source {index} is not a fixed-split "
                "hash-bound BEAT2 manifest"
            )
    registry_contract = sources.get("data_source_registry")
    if registry_contract is not None:
        dataset_sources = [record["dataset_source"] for record in source_records]
        registry_contract = validate_data_source_registry_contract(
            registry_contract,
            expected_role=GENERATOR_FOUNDATION_ROLE,
            expected_dataset_sources=dataset_sources,
        )
        if contracts.get("data_source_registry") != registry_contract:
            raise ValueError(
                "motion-only checkpoint registry copies are inconsistent"
            )
        for name, contract in (
            ("split", split_contract),
            ("action statistics", contracts.get("action_statistics")),
            ("style normalization", contracts.get("style")),
            ("duration", contracts.get("duration")),
            ("data isolation", isolation),
        ):
            validate_contract_source_binding(
                contract,
                registry_contract,
                context=f"motion_only_checkpoint.{name}",
            )

    random_initialization = checkpoint.get("random_initialization")
    if not isinstance(random_initialization, Mapping) or (
        random_initialization.get("mode") != MOTION_ONLY_RANDOM_INIT_MODE
        or random_initialization.get("qwen_policy") != MOTION_ONLY_NO_QWEN_POLICY
        or random_initialization.get("kimodo_policy") != MOTION_ONLY_NO_KIMODO_POLICY
        or random_initialization.get("generator_checkpoint_inputs") != []
    ):
        raise ValueError("motion-only random initialization isolation policy changed")
    return dict(isolation)


def validate_checkpoint_contract(checkpoint: Mapping, *, expected_action_dim=None) -> list[str]:
    """Validate either the legacy 15D or append-only 18D checkpoint contract."""
    assert_no_forbidden_data_lineage(
        checkpoint, context="generator_checkpoint"
    )
    if checkpoint.get("artifact_kind") != ARTIFACT_KIND:
        raise ValueError(f"unexpected checkpoint artifact_kind: {checkpoint.get('artifact_kind')!r}")
    if checkpoint.get("architecture") not in SUPPORTED_GENERATOR_ARCHITECTURES:
        raise ValueError(f"unexpected architecture: {checkpoint.get('architecture')!r}")
    if int(checkpoint.get("condition_dim", -1)) != KIMODO_V2_CONDITION_DIM:
        raise ValueError(
            f"ULA MMDiT V2 checkpoint must use condition_dim={KIMODO_V2_CONDITION_DIM}"
        )
    action_dim = int(checkpoint.get("action_dim", -1))
    if expected_action_dim is not None and action_dim != int(expected_action_dim):
        raise ValueError(f"checkpoint action_dim={action_dim}, expected {int(expected_action_dim)}")
    expected_order = joint_order_for_action_dim(action_dim)
    if list(checkpoint.get("joint_order") or []) != expected_order:
        raise ValueError(f"checkpoint joint_order does not match its {action_dim}D contract")
    if action_dim == ACTION_DIM:
        action_contract = checkpoint.get("action_contract") or {}
        if action_contract.get("version") != CONTRACT_VERSION:
            raise ValueError(
                f"18D checkpoint requires action contract {CONTRACT_VERSION!r}"
            )
        if int(action_contract.get("legacy_prefix_dim", -1)) != LEGACY_ACTION_DIM:
            raise ValueError("18D checkpoint does not declare the preserved 15D prefix")

    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint is missing model_state_dict")
    shape = _checkpoint_model_shape(checkpoint)
    required_shapes = {
        "input.weight": (shape["hidden_dim"], action_dim),
        "output.weight": (action_dim, shape["hidden_dim"]),
        "output.bias": (action_dim,),
    }
    for name, expected_shape in required_shapes.items():
        value = state.get(name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape:
            raise ValueError(f"checkpoint {name} must have shape {expected_shape}")
    stats = checkpoint.get("action_stats")
    if not isinstance(stats, Mapping):
        raise ValueError("checkpoint is missing action_stats")
    for name in ("mean", "std"):
        value = torch.as_tensor(stats.get(name))
        if tuple(value.shape) != (action_dim,) or not torch.isfinite(value).all():
            raise ValueError(f"checkpoint action_stats.{name} must be finite [{action_dim}]")
    if torch.any(torch.as_tensor(stats["std"]) <= 0):
        raise ValueError("checkpoint action_stats.std must be positive")
    if checkpoint.get("formal_episode_contract") == MOTION_ONLY_EPISODE_CONTRACT:
        validate_motion_only_checkpoint_isolation(checkpoint)
    return expected_order


def instantiate_checkpoint_model(checkpoint: Mapping, *, device="cpu"):
    validate_checkpoint_contract(checkpoint)
    shape = _checkpoint_model_shape(checkpoint)
    model = create_ula_model(
        checkpoint["architecture"],
        action_dim=int(checkpoint["action_dim"]),
        condition_dim=int(checkpoint["condition_dim"]),
        **shape,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.action_stats = {
        name: torch.as_tensor(value, dtype=torch.float32).clone()
        for name, value in checkpoint["action_stats"].items()
    }
    model.planner_supervision_contract = deepcopy(
        checkpoint.get("planner_supervision_contract")
        or (checkpoint.get("training_contract") or {}).get("planner_supervision")
        or {}
    )
    return model.to(device)


def load_contract_checkpoint(path: str | Path, *, expected_action_dim=None, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    validate_checkpoint_contract(checkpoint, expected_action_dim=expected_action_dim)
    return instantiate_checkpoint_model(checkpoint, device=device), checkpoint


def compute_18d_action_stats(
    trajectories: Sequence[np.ndarray], base_action_stats: Mapping
) -> dict[str, torch.Tensor]:
    if not trajectories:
        raise ValueError("at least one 18D trajectory is required for action statistics")
    for trajectory in trajectories:
        if trajectory.ndim != 2 or trajectory.shape[1] != ACTION_DIM:
            raise ValueError(f"trajectory must have shape [frames, {ACTION_DIM}]")
        if not np.isfinite(trajectory).all():
            raise ValueError("trajectory contains non-finite values")
    head = np.concatenate([value[:, HEAD_SLICE] for value in trajectories], axis=0)
    base_mean = torch.as_tensor(base_action_stats["mean"], dtype=torch.float32)
    base_std = torch.as_tensor(base_action_stats["std"], dtype=torch.float32)
    if tuple(base_mean.shape) != (LEGACY_ACTION_DIM,) or tuple(base_std.shape) != (
        LEGACY_ACTION_DIM,
    ):
        raise ValueError("base action statistics must contain exactly 15 channels")
    head_mean = torch.from_numpy(head.mean(axis=0).astype(np.float32))
    head_std = torch.from_numpy(head.std(axis=0).astype(np.float32)).clamp_min(1e-3)
    return {
        "mean": torch.cat([base_mean, head_mean]),
        "std": torch.cat([base_std, head_std]),
    }


def _expanded_state_dict(base_checkpoint: Mapping) -> dict[str, torch.Tensor]:
    shape = _checkpoint_model_shape(base_checkpoint)
    expanded = create_ula_model(
        ULA_MMDIT_V2_ARCHITECTURE,
        action_dim=ACTION_DIM,
        condition_dim=KIMODO_V2_CONDITION_DIM,
        **shape,
    )
    destination = expanded.state_dict()
    source = base_checkpoint["model_state_dict"]
    for name, target in destination.items():
        if name == "input.weight":
            target.zero_()
            target[:, :LEGACY_ACTION_DIM].copy_(source[name])
        elif name == "output.weight":
            target.zero_()
            target[:LEGACY_ACTION_DIM].copy_(source[name])
        elif name == "output.bias":
            target.zero_()
            target[:LEGACY_ACTION_DIM].copy_(source[name])
        else:
            if name not in source or tuple(source[name].shape) != tuple(target.shape):
                raise ValueError(f"base checkpoint cannot migrate parameter {name}")
            target.copy_(source[name])
    return destination


def verify_migrated_prefix(base_checkpoint: Mapping, migrated_checkpoint: Mapping) -> float:
    base = base_checkpoint["model_state_dict"]
    migrated = migrated_checkpoint["model_state_dict"]
    maximum = 0.0
    for name, value in base.items():
        if name == "input.weight":
            candidate = migrated[name][:, :LEGACY_ACTION_DIM]
        elif name in ("output.weight", "output.bias"):
            candidate = migrated[name][:LEGACY_ACTION_DIM]
        else:
            candidate = migrated[name]
        maximum = max(maximum, float((candidate.cpu() - value.cpu()).abs().max()))
    for name in ("mean", "std"):
        maximum = max(
            maximum,
            float(
                (
                    torch.as_tensor(migrated_checkpoint["action_stats"][name])[:LEGACY_ACTION_DIM]
                    - torch.as_tensor(base_checkpoint["action_stats"][name])
                )
                .abs()
                .max()
            ),
        )
    return maximum


def legacy_forward_max_error(
    base_model,
    expanded_model,
    *,
    seed=17,
    batch_size=2,
    frames=11,
    device="cpu",
) -> float:
    generator = torch.Generator(device=device).manual_seed(int(seed))
    x_15d = torch.randn(
        batch_size, frames, LEGACY_ACTION_DIM, generator=generator, device=device
    )
    x_18d = F.pad(x_15d, (0, ACTION_DIM - LEGACY_ACTION_DIM))
    condition = torch.randn(
        batch_size, KIMODO_V2_CONDITION_DIM, generator=generator, device=device
    )
    t = torch.rand(batch_size, generator=generator, device=device)
    base_training, expanded_training = base_model.training, expanded_model.training
    base_model.eval()
    expanded_model.eval()
    with torch.no_grad():
        expected = base_model(x_15d, t, condition)
        actual = expanded_model(x_18d, t, condition)[..., :LEGACY_ACTION_DIM]
    base_model.train(base_training)
    expanded_model.train(expanded_training)
    return float((expected - actual).abs().max().cpu())


def migrate_15d_checkpoint(
    base_checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    action_stats: Mapping,
) -> tuple[dict, dict]:
    base_checkpoint_path = Path(base_checkpoint_path)
    base = torch.load(base_checkpoint_path, map_location="cpu", weights_only=True)
    validate_checkpoint_contract(base, expected_action_dim=LEGACY_ACTION_DIM)
    migrated = deepcopy(base)
    migrated_config = dict(base.get("config") or {})
    migrated_config.update(
        {
            "action_dim": ACTION_DIM,
            "action_contract_version": CONTRACT_VERSION,
            "adapter_policy": ADAPTER_POLICY,
        }
    )
    migrated.update(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "artifact_kind": ARTIFACT_KIND,
            "model_state_dict": _expanded_state_dict(base),
            "joint_order": list(JOINT_ORDER_18D),
            "action_dim": ACTION_DIM,
            "action_stats": {
                name: torch.as_tensor(value, dtype=torch.float32).cpu()
                for name, value in action_stats.items()
            },
            "config": migrated_config,
            "action_contract": {
                "version": CONTRACT_VERSION,
                "joint_order": list(JOINT_ORDER_18D),
                "legacy_prefix_dim": LEGACY_ACTION_DIM,
                "legacy_joint_order": list(JOINT_ORDER),
                "migration": "append_only_zero_initialized_projection_slices",
            },
            "head_adapter": {
                "policy": ADAPTER_POLICY,
                "trainable_slices": {
                    "input.weight": ["all", [LEGACY_ACTION_DIM, ACTION_DIM]],
                    "output.weight": [[LEGACY_ACTION_DIM, ACTION_DIM], "all"],
                    "output.bias": [[LEGACY_ACTION_DIM, ACTION_DIM]],
                },
                "legacy_zero_padded_input_compatible": True,
            },
            "migration_source": {
                "path": str(base_checkpoint_path.resolve()),
                "sha256": sha256_file(base_checkpoint_path),
                "source_action_dim": LEGACY_ACTION_DIM,
                "source_global_step": int(base.get("global_step", 0)),
            },
        }
    )
    migrated.pop("training_state", None)
    validate_checkpoint_contract(migrated, expected_action_dim=ACTION_DIM)
    prefix_error = verify_migrated_prefix(base, migrated)
    if prefix_error != 0.0:
        raise RuntimeError(f"15D prefix changed during migration: max error {prefix_error}")
    base_model = instantiate_checkpoint_model(base)
    migrated_model = instantiate_checkpoint_model(migrated)
    forward_error = legacy_forward_max_error(base_model, migrated_model)
    if forward_error > LEGACY_FORWARD_ATOL:
        raise RuntimeError(
            "zero-padded legacy forward exceeds the float32 compatibility tolerance: "
            f"{forward_error} > {LEGACY_FORWARD_ATOL}"
        )
    migrated["migration_verification"] = {
        "weight_prefix_max_abs_error": prefix_error,
        "legacy_forward_max_abs_error": forward_error,
        "legacy_forward_atol": LEGACY_FORWARD_ATOL,
    }
    _atomic_torch_save(migrated, output_path)
    return migrated, migrated["migration_verification"]


def _resolve_prompt(record: Mapping) -> str:
    prompt = record.get("canonical_prompt") or record.get("prompt") or record.get("text")
    if isinstance(prompt, Mapping):
        prompt = prompt.get("en") or prompt.get("zh")
    prompt = str(prompt or "").strip()
    if not prompt:
        raise ValueError(f"missing canonical prompt for {record.get('clip_id')!r}")
    return prompt


def _semantic_mappings(record: Mapping):
    """Yield explicit semantic fields without consulting natural-language text."""
    yield "record", record
    for name in ("labels", "meta_semantics", "meta"):
        value = record.get(name)
        if isinstance(value, Mapping):
            yield name, value


def _explicit_semantic_value(record: Mapping, field: str):
    values = []
    for source, mapping in _semantic_mappings(record):
        if field not in mapping:
            continue
        value = mapping[field]
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        if isinstance(value, str):
            value = value.strip()
        values.append((source, value))
    if not values:
        return None
    expected = values[0][1]
    if any(value != expected for _, value in values[1:]):
        details = ", ".join(f"{source}={value!r}" for source, value in values)
        raise ValueError(f"conflicting explicit {field}: {details}")
    return expected


def _resolve_structured_semantics(record: Mapping) -> dict:
    """Resolve the fail-closed Kimodo condition labels for one episode."""
    clip_id = str(record.get("clip_id") or record.get("sample_id") or "<unknown>")
    behavior_id = _explicit_semantic_value(record, "behavior_id")
    if behavior_id not in KIMODO_BEHAVIOR_IDS:
        if behavior_id is None:
            raise ValueError(f"missing explicit behavior_id for {clip_id}")
        raise ValueError(f"unknown Kimodo behavior_id for {clip_id}: {behavior_id!r}")

    behavior_review_status = _explicit_semantic_value(record, "behavior_review_status")
    if behavior_review_status is not None and not isinstance(behavior_review_status, str):
        raise ValueError(f"behavior_review_status for {clip_id} must be a string")
    behavior_supervision_mask = _explicit_semantic_value(
        record, "behavior_supervision_mask"
    )
    if behavior_supervision_mask is not None and not isinstance(
        behavior_supervision_mask, bool
    ):
        raise ValueError(f"behavior_supervision_mask for {clip_id} must be boolean")
    if behavior_review_status is None:
        if behavior_supervision_mask is False:
            raise ValueError(
                f"unsupervised behavior for {clip_id} requires an explicit review status"
            )
        behavior_review_status = "legacy_resolved"
        behavior_supervision_mask = True
    elif behavior_review_status in {"human_confirmed", "legacy_resolved"}:
        if behavior_supervision_mask is not True:
            raise ValueError(
                f"{behavior_review_status} behavior for {clip_id} requires "
                "behavior_supervision_mask=true"
            )
    elif behavior_review_status in {"candidate_unreviewed", "rejected"}:
        if behavior_supervision_mask is None:
            behavior_supervision_mask = False
        elif behavior_supervision_mask is not False:
            raise ValueError(
                f"{behavior_review_status} behavior for {clip_id} requires "
                "behavior_supervision_mask=false"
            )
    else:
        raise ValueError(
            f"unknown behavior_review_status for {clip_id}: {behavior_review_status!r}"
        )

    emotion_id = _explicit_semantic_value(record, "emotion_id")
    if emotion_id is not None and emotion_id not in KIMODO_EMOTION_IDS:
        raise ValueError(f"unknown Kimodo emotion_id for {clip_id}: {emotion_id!r}")

    review_status = _explicit_semantic_value(record, "emotion_review_status")
    if review_status is not None and not isinstance(review_status, str):
        raise ValueError(f"emotion_review_status for {clip_id} must be a string")
    supervision_mask = _explicit_semantic_value(record, "emotion_supervision_mask")
    if supervision_mask is not None and not isinstance(supervision_mask, bool):
        raise ValueError(f"emotion_supervision_mask for {clip_id} must be boolean")

    official_emotion = bool(
        record.get("annotation_kind") == "official_gesture_semantic_event"
        and review_status == "official_protocol_confirmed"
    )
    official_emotion_enabled = _explicit_semantic_value(
        record, "official_emotion_conditioning_enabled"
    )
    source_emotion_label_verified = _explicit_semantic_value(
        record, "source_emotion_label_verified"
    )
    if source_emotion_label_verified is not None and not isinstance(
        source_emotion_label_verified, bool
    ):
        raise ValueError(
            f"source_emotion_label_verified for {clip_id} must be boolean"
        )
    if review_status == "unresolved":
        if emotion_id is not None:
            raise ValueError(
                f"unresolved emotion for {clip_id} must not carry an emotion_id"
            )
        if supervision_mask is not False:
            raise ValueError(
                f"unresolved emotion for {clip_id} requires emotion_supervision_mask=false"
            )
    elif official_emotion:
        if emotion_id is None:
            raise ValueError(f"missing official source emotion label for {clip_id}")
        if source_emotion_label_verified is not True:
            raise ValueError(f"official source emotion label is not verified for {clip_id}")
        if not isinstance(official_emotion_enabled, bool):
            raise ValueError(
                f"official_emotion_conditioning_enabled for {clip_id} must be boolean"
            )
        if supervision_mask is not official_emotion_enabled:
            raise ValueError(
                f"emotion_supervision_mask for {clip_id} must match official emotion "
                "conditioning enablement"
            )
        expected_role = (
            OFFICIAL_EMOTION_ENABLED_ROLE
            if official_emotion_enabled
            else OFFICIAL_EMOTION_DISABLED_ROLE
        )
        if record.get("emotion_supervision_role") != expected_role:
            raise ValueError(f"official emotion supervision role is invalid for {clip_id}")
        expected_channel = (
            OFFICIAL_EMOTION_CONDITION_CHANNEL if official_emotion_enabled else None
        )
        if (
            "official_emotion_condition_channel" not in record
            or record.get("official_emotion_condition_channel") != expected_channel
        ):
            raise ValueError(f"official emotion condition channel is invalid for {clip_id}")
        if (
            "official_emotion_loss" not in record
            or record.get("official_emotion_loss") is not None
        ):
            raise ValueError(f"official emotion loss must be explicit null for {clip_id}")
    elif source_emotion_label_verified is True:
        if emotion_id is None or review_status not in {
            "human_confirmed",
            "legacy_resolved",
        }:
            raise ValueError(
                f"verified source emotion label for {clip_id} lacks a resolved label"
            )
        if not isinstance(supervision_mask, bool):
            raise ValueError(
                f"adjudicated emotion supervision mask for {clip_id} must be boolean"
            )
    else:
        if emotion_id is None:
            raise ValueError(
                f"missing explicit emotion_id for {clip_id}; use "
                "emotion_review_status='unresolved' with emotion_supervision_mask=false"
            )
        if supervision_mask is False:
            raise ValueError(
                f"resolved emotion for {clip_id} cannot disable emotion supervision"
            )
        supervision_mask = True
        review_status = review_status or "legacy_resolved"

    affect_review_status = _explicit_semantic_value(
        record, "affect_observable_review_status"
    )
    affect_supervision_mask = _explicit_semantic_value(
        record, "affect_observable_supervision_mask"
    )
    emotion_conditioning_mask = _explicit_semantic_value(
        record, "emotion_conditioning_mask"
    )
    formal_official = record.get("annotation_kind") == "official_gesture_semantic_event"
    has_affect_contract = bool(
        formal_official
        or affect_review_status is not None
        or affect_supervision_mask is not None
        or emotion_conditioning_mask is not None
    )
    if has_affect_contract:
        if affect_review_status not in AFFECT_OBSERVABLE_REVIEW_STATUSES:
            raise ValueError(
                f"affect_observable_review_status for {clip_id} must be one of "
                f"{sorted(AFFECT_OBSERVABLE_REVIEW_STATUSES)}"
            )
        if not isinstance(affect_supervision_mask, bool):
            raise ValueError(
                f"affect_observable_supervision_mask for {clip_id} must be boolean"
            )
        if affect_review_status == "legacy_not_applicable":
            if formal_official or affect_supervision_mask is not bool(supervision_mask):
                raise ValueError(
                    f"legacy affect observability state is invalid for {clip_id}"
                )
        elif (affect_review_status == "verified") is not affect_supervision_mask:
            raise ValueError(
                f"affect observability status/mask disagree for {clip_id}"
            )
        expected_conditioning_mask = bool(
            supervision_mask is True
            and affect_supervision_mask is True
            and (not official_emotion or official_emotion_enabled is True)
        )
        if not isinstance(emotion_conditioning_mask, bool):
            raise ValueError(f"emotion_conditioning_mask for {clip_id} must be boolean")
        if emotion_conditioning_mask is not expected_conditioning_mask:
            raise ValueError(
                f"emotion_conditioning_mask for {clip_id} must require both source-label "
                "and robot-affect observability supervision"
            )
    else:
        affect_review_status = "legacy_not_applicable"
        affect_supervision_mask = bool(supervision_mask)
        emotion_conditioning_mask = bool(supervision_mask)

    return {
        "behavior_id": behavior_id,
        "behavior_review_status": behavior_review_status,
        "behavior_supervision_mask": behavior_supervision_mask,
        "emotion_id": emotion_id,
        "emotion_review_status": review_status,
        "emotion_supervision_mask": supervision_mask,
        "source_emotion_label_verified": (
            source_emotion_label_verified
            if source_emotion_label_verified is not None
            else review_status != "unresolved"
        ),
        "official_emotion_conditioning_enabled": (
            official_emotion_enabled if official_emotion else bool(supervision_mask)
        ),
        "emotion_supervision_role": (
            record.get("emotion_supervision_role")
            if official_emotion
            else "legacy_conditioning"
        ),
        "official_emotion_condition_channel": (
            record.get("official_emotion_condition_channel")
            if official_emotion
            else None
        ),
        "official_emotion_loss": (
            record.get("official_emotion_loss") if official_emotion else None
        ),
        "affect_observable_review_status": affect_review_status,
        "affect_observable_supervision_mask": affect_supervision_mask,
        "emotion_conditioning_mask": emotion_conditioning_mask,
    }


def _resolve_legacy_semantics(record: Mapping) -> dict:
    resolved = {}
    vocabularies = {
        "intent": INTENT_IDS,
        "semantic_gesture": GESTURE_IDS,
    }
    for field, vocabulary in vocabularies.items():
        value = _explicit_semantic_value(record, field)
        if value is None:
            value = LEGACY_SEMANTIC_DEFAULTS[field]
        if value not in vocabulary:
            raise ValueError(f"unknown structured {field}: {value!r}")
        resolved[field] = value

    affect_values = []
    for source, mapping in _semantic_mappings(record):
        if "observed_affect" not in mapping:
            continue
        value = mapping["observed_affect"]
        if isinstance(value, Mapping):
            value = value.get("label")
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        value = KIMODO_EMOTION_TO_LEGACY_AFFECT.get(str(value).strip(), str(value).strip())
        affect_values.append((source, value))
    if affect_values and any(value != affect_values[0][1] for _, value in affect_values[1:]):
        details = ", ".join(f"{source}={value!r}" for source, value in affect_values)
        raise ValueError(f"conflicting explicit observed_affect: {details}")
    observed_affect = (
        affect_values[0][1]
        if affect_values
        else LEGACY_SEMANTIC_DEFAULTS["observed_affect"]
    )
    emotion_conditioning_mask = _explicit_semantic_value(
        record, "emotion_conditioning_mask"
    )
    affect_review_status = _explicit_semantic_value(
        record, "affect_observable_review_status"
    )
    if emotion_conditioning_mask is True and affect_review_status == "verified":
        emotion_id = _explicit_semantic_value(record, "emotion_id")
        if emotion_id not in KIMODO_EMOTION_IDS:
            raise ValueError(
                "conditioned emotion requires a verified robot-affect label in the "
                "six-class ontology"
            )
        observed_affect = KIMODO_EMOTION_TO_LEGACY_AFFECT[emotion_id]
    if observed_affect not in AFFECT_IDS:
        raise ValueError(f"unknown structured observed_affect: {observed_affect!r}")
    resolved["observed_affect"] = observed_affect

    normalized_motion_style = _explicit_semantic_value(record, "motion_style")
    raw_motion_style = _explicit_semantic_value(record, "source_motion_style")
    raw_motion_style = raw_motion_style or normalized_motion_style
    if raw_motion_style is None:
        raw_motion_style = LEGACY_SEMANTIC_DEFAULTS["motion_style"]
    motion_style = MOTION_STYLE_TO_LEGACY.get(raw_motion_style)
    if motion_style not in STYLE_IDS:
        raise ValueError(f"unknown structured motion_style: {raw_motion_style!r}")
    if normalized_motion_style is not None:
        declared_style = MOTION_STYLE_TO_LEGACY.get(normalized_motion_style)
        if declared_style != motion_style:
            raise ValueError("source_motion_style conflicts with normalized motion_style")
    resolved["motion_style"] = motion_style
    resolved["source_motion_style"] = raw_motion_style

    for field in (
        "arousal",
        "valence",
        "arousal_token",
        "valence_token",
        "motion_energy",
    ):
        value = _explicit_semantic_value(record, field)
        if value is not None:
            if isinstance(value, bool):
                raise ValueError(f"structured {field} must be numeric, not boolean")
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"structured {field} must be finite")
        resolved[field] = value
    return resolved


def _contract_sha256(contract: Mapping) -> str:
    payload = {key: value for key, value in contract.items() if key != "sha256"}
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _generator_style_contract(generator_checkpoint: Mapping) -> dict:
    contracts = generator_checkpoint.get("v2_contracts") or {}
    style_contract = contracts.get("style") or {}
    if not isinstance(style_contract, Mapping) or not style_contract:
        raise ValueError("target generator checkpoint lacks v2_contracts.style")
    recorded_sha256 = style_contract.get("sha256")
    if not isinstance(recorded_sha256, str) or _contract_sha256(style_contract) != recorded_sha256:
        raise ValueError("target generator style contract hash is invalid")
    normalize_style_features(np.zeros(3, dtype=np.float32), style_contract)
    condition_contract = contracts.get("condition") or {}
    if condition_contract.get("style_contract_sha256") != recorded_sha256:
        raise ValueError("target generator condition contract is not bound to its style contract")
    if list(condition_contract.get("style_control_indices") or []) != list(
        range(STYLE_CONTROL_SLICE.start, STYLE_CONTROL_SLICE.stop)
    ):
        raise ValueError("target generator style control indices do not match the 136D base")
    return dict(style_contract)


def _load_target_generator_contract(path: str | Path) -> tuple[dict, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    validate_checkpoint_contract(checkpoint)
    return checkpoint, _generator_style_contract(checkpoint)


def _structured_condition_base(
    prompt: str,
    semantics: Mapping,
    legacy_semantics: Mapping,
    style_controls: np.ndarray,
    semantic_supervision: Mapping | None = None,
) -> np.ndarray:
    emotion_id = semantics["emotion_id"]
    supervised = semantics["emotion_conditioning_mask"] is True
    # The existing builder requires a member of the six-class vocabulary.  For
    # unresolved samples the temporary class is removed immediately, leaving a
    # dimension-preserving all-zero Kimodo emotion block rather than neutral.
    builder_emotion_id = emotion_id if supervised else KIMODO_EMOTION_IDS[0]
    condition = build_condition_from_text(
        prompt,
        intent=legacy_semantics["intent"],
        affect=legacy_semantics["observed_affect"],
        style=legacy_semantics["motion_style"],
        gesture=legacy_semantics["semantic_gesture"],
        arousal=legacy_semantics["arousal"],
        valence=legacy_semantics["valence"],
        arousal_token=legacy_semantics["arousal_token"],
        valence_token=legacy_semantics["valence_token"],
        motion_energy=(
            0.05
            if legacy_semantics["motion_energy"] is None
            else legacy_semantics["motion_energy"]
        ),
        behavior_id=semantics["behavior_id"],
        emotion_id=builder_emotion_id,
        condition_dim=KIMODO_CONDITION_DIM,
    ).astype(np.float32, copy=True)
    if semantics["behavior_supervision_mask"] is False:
        condition[KIMODO_BEHAVIOR_SLICE] = 0.0
        condition[KIMODO_BEHAVIOR_FAMILY_SLICE] = 0.0
    if not supervised:
        condition[KIMODO_EMOTION_SLICE] = 0.0
        condition[LEGACY_AFFECT_SLICE] = 0.0
    semantic_supervision = dict(semantic_supervision or {})
    if semantic_supervision.get("communicative_intent", True) is False:
        condition[LEGACY_INTENT_SLICE] = 0.0
    if semantic_supervision.get("legacy_gesture", True) is False:
        condition[LEGACY_GESTURE_SLICE] = 0.0
    controls = np.asarray(style_controls, dtype=np.float32)
    if controls.shape != (3,) or not np.isfinite(controls).all():
        raise ValueError("trajectory-derived style controls must be finite with shape [3]")
    condition[STYLE_CONTROL_SLICE] = controls
    return condition


def semantic_supervision_policy(record: Mapping) -> dict:
    """Resolve and validate which semantic channels may supervise an episode."""
    if record.get("annotation_kind") != "official_gesture_semantic_event":
        return {
            "official_category": None,
            "official_category_verified": False,
            "official_category_conditioning_enabled": False,
            "official_category_role": "not_applicable",
            "official_category_condition_channel": None,
            "official_category_loss": None,
            "robot_observable_motion_form": None,
            "communicative_intent": None,
            "canonical_prompt_role": "legacy_full_prompt",
            "semantic_supervision_masks": {
                "official_category": False,
                "robot_observable_motion_form": False,
                "communicative_intent": True,
                "prompt_text": True,
                "legacy_gesture": True,
            },
        }
    semantic_event = record.get("semantic_event")
    category = (
        semantic_event.get("category")
        if isinstance(semantic_event, Mapping)
        else None
    )
    errors = []
    if category not in {"deictic", "iconic", "metaphoric"}:
        errors.append("semantic_event.category is not an official category")
    if record.get("official_category_verified") is not True:
        errors.append("official_category_verified must be true")
    if record.get("official_category_conditioning_enabled") is not False:
        errors.append("official_category_conditioning_enabled must be false")
    if (
        record.get("official_category_role")
        != OFFICIAL_CATEGORY_CONDITIONING_ROLE
    ):
        errors.append(
            "official_category_role must be metadata/split/evaluation only"
        )
    if (
        "official_category_condition_channel" not in record
        or record.get("official_category_condition_channel") is not None
    ):
        errors.append("official_category_condition_channel must be explicit null")
    if "official_category_loss" not in record or record.get("official_category_loss") is not None:
        errors.append("official_category_loss must be explicit null")
    if record.get("source_emotion_label_verified") is not True:
        errors.append("source_emotion_label_verified must be true")
    official_emotion_enabled = record.get("official_emotion_conditioning_enabled")
    if not isinstance(official_emotion_enabled, bool):
        errors.append("official_emotion_conditioning_enabled must be boolean")
    if record.get("emotion_supervision_mask") is not official_emotion_enabled:
        errors.append(
            "emotion_supervision_mask must match official emotion conditioning enablement"
        )
    expected_emotion_role = (
        OFFICIAL_EMOTION_ENABLED_ROLE
        if official_emotion_enabled is True
        else OFFICIAL_EMOTION_DISABLED_ROLE
    )
    if record.get("emotion_supervision_role") != expected_emotion_role:
        errors.append("emotion_supervision_role does not match official emotion state")
    expected_emotion_channel = (
        OFFICIAL_EMOTION_CONDITION_CHANNEL
        if official_emotion_enabled is True
        else None
    )
    if (
        "official_emotion_condition_channel" not in record
        or record.get("official_emotion_condition_channel")
        != expected_emotion_channel
    ):
        errors.append("official_emotion_condition_channel is invalid")
    if "official_emotion_loss" not in record or record.get("official_emotion_loss") is not None:
        errors.append("official_emotion_loss must be explicit null")
    if record.get("robot_observable_motion_form") != "candidate_unreviewed":
        errors.append("robot_observable_motion_form must remain candidate_unreviewed")
    if record.get("communicative_intent") != "candidate_unreviewed":
        errors.append("communicative_intent must remain candidate_unreviewed")
    if record.get("canonical_prompt_role") != "coarse_category_only":
        errors.append("canonical_prompt_role must be coarse_category_only")
    masks = record.get("semantic_supervision_masks")
    if masks != FORMAL_SEMANTIC_SUPERVISION_MASKS:
        errors.append("semantic_supervision_masks do not match the coarse-category policy")
    canonical_action = record.get("canonical_action")
    if canonical_action != f"official_gesture_category:{category}":
        errors.append("canonical_action must be the coarse official-category split key")
    if record.get("canonical_action_role") != "official_category_metadata_split_key_only":
        errors.append("canonical_action_role must prevent communicative-intent use")
    if (
        record.get("semantic_mapping_status")
        != "official_category_verified_metadata_only"
    ):
        errors.append("semantic_mapping_status must declare the coarse official mapping")
    affect_status = record.get(
        "affect_observable_review_status", "candidate_unreviewed"
    )
    affect_mask = record.get("affect_observable_supervision_mask", False)
    if affect_status not in AFFECT_OBSERVABLE_REVIEW_STATUSES:
        errors.append("affect_observable_review_status is invalid or missing")
    if not isinstance(affect_mask, bool):
        errors.append("affect_observable_supervision_mask must be boolean")
    elif (affect_status == "verified") is not affect_mask:
        errors.append("affect observability status/mask disagree")
    expected_emotion_conditioning = bool(
        record.get("emotion_supervision_mask") is True
        and official_emotion_enabled is True
        and affect_mask is True
    )
    if (
        "emotion_conditioning_mask" in record
        and record.get("emotion_conditioning_mask") is not expected_emotion_conditioning
    ):
        errors.append(
            "emotion_conditioning_mask must require source emotion and affect observability"
        )
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "official_category": category,
        "official_category_verified": True,
        "official_category_conditioning_enabled": False,
        "official_category_role": OFFICIAL_CATEGORY_CONDITIONING_ROLE,
        "official_category_condition_channel": None,
        "official_category_loss": None,
        "source_emotion_label_verified": True,
        "official_emotion_conditioning_enabled": official_emotion_enabled,
        "emotion_supervision_role": expected_emotion_role,
        "official_emotion_condition_channel": expected_emotion_channel,
        "official_emotion_loss": None,
        "robot_observable_motion_form": "candidate_unreviewed",
        "communicative_intent": "candidate_unreviewed",
        "canonical_prompt_role": "coarse_category_only",
        "canonical_action_role": "official_category_metadata_split_key_only",
        "semantic_mapping_status": "official_category_verified_metadata_only",
        "semantic_supervision_masks": dict(FORMAL_SEMANTIC_SUPERVISION_MASKS),
        "affect_observable_review_status": affect_status,
        "affect_observable_supervision_mask": affect_mask,
        "emotion_conditioning_mask": expected_emotion_conditioning,
    }


def _resolve_motion_path(record: Mapping, *, manifest_path: Path) -> Path | None:
    candidates = [
        record.get("trajectory_path"),
        record.get("retarget_csv_path"),
        record.get("csv_path"),
        (record.get("action") or {}).get("retarget_csv_path")
        if isinstance(record.get("action"), Mapping)
        else None,
        (record.get("source") or {}).get("retarget_csv_path")
        if isinstance(record.get("source"), Mapping)
        else None,
        (record.get("outputs") or {}).get("safe_csv")
        if isinstance(record.get("outputs"), Mapping)
        else None,
        (record.get("motion_18d") or {}).get("safe_csv")
        if isinstance(record.get("motion_18d"), Mapping)
        else None,
    ]
    for value in candidates:
        if value:
            path = Path(value)
            return path if path.is_absolute() else manifest_path.parent / path
    return None


def _read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on {path}:{line_number}") from exc
    return records


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _content_contract_valid(contract) -> bool:
    if not isinstance(contract, Mapping):
        return False
    payload = {key: value for key, value in contract.items() if key != "sha256"}
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return contract.get("sha256") == digest


def _project_weak_behavior_record(record: Mapping) -> bool:
    contract = record.get("behavior_mapping_contract")
    return bool(
        record.get("behavior_id") == "Behavior.InteractPresence"
        and record.get("behavior_review_status") == "candidate_unreviewed"
        and record.get("behavior_supervision_mask") is False
        and record.get("behavior_source") == "project_dataset_scope_weak_mapping_v1"
        and isinstance(contract, Mapping)
        and contract.get("source") == "project_dataset_scope_weak_mapping_v1"
        and contract.get("behavior_id") == "Behavior.InteractPresence"
        and contract.get("supervision") == "weak_candidate_masked"
        and _content_contract_valid(contract)
    )


def _official_source_emotion_label_verified_record(record: Mapping) -> bool:
    contract = record.get("emotion_protocol_contract")
    return bool(
        record.get("emotion_id") in KIMODO_EMOTION_IDS
        and record.get("emotion_review_status") == "official_protocol_confirmed"
        and record.get("source_emotion_label_verified") is True
        and record.get("emotion_source") == "official_beat2_filename_protocol"
        and isinstance(contract, Mapping)
        and contract.get("source") == "official_beat2_filename_protocol"
        and contract.get("emotion_id") == record.get("emotion_id")
        and contract.get("source_sha256") == record.get("source_sha256")
        and _content_contract_valid(contract)
    )


def _formal_blind_affect_errors(record: Mapping, review: Mapping) -> list[str]:
    errors = []
    proof = review.get("blind_affect_review")
    if not isinstance(proof, Mapping):
        return ["independent_review blind affect evidence is missing"]
    if proof.get("protocol_version") != BLIND_AFFECT_PROTOCOL_VERSION:
        errors.append("blind affect protocol version is invalid")
    if not str(proof.get("review_id") or "").strip():
        errors.append("blind affect review_id is missing")
    if not str(proof.get("anonymous_video_id") or "").strip():
        errors.append("blind affect anonymous video id is missing")
    if not _is_sha256(proof.get("video_sha256")):
        errors.append("blind affect video SHA256 is invalid")
    if proof.get("target_emotion_exposed") is not False:
        errors.append("blind affect review exposed the target emotion")
    if proof.get("audio_available") is not False:
        errors.append("blind affect review did not use silent video")
    blinded_to = proof.get("blinded_to")
    if not isinstance(blinded_to, list) or set(blinded_to) != set(
        BLIND_AFFECT_REQUIRED_BLINDED_FIELDS
    ):
        errors.append("blind affect review did not hide all target fields")
    reviewer = proof.get("reviewer")
    if not isinstance(reviewer, Mapping) or (
        reviewer.get("kind") != "agent"
        or reviewer.get("independent_of_annotation_logic") is not True
        or not str(reviewer.get("reviewer_id") or "").strip()
    ):
        errors.append("blind affect reviewer provenance is invalid")
    observed = proof.get("observed_affect")
    if not isinstance(observed, Mapping) or (
        observed.get("status") != "label"
        or observed.get("emotion_id") != record.get("emotion_id")
    ):
        errors.append("blind affect label does not match the official source label")
    else:
        confidence = observed.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or float(confidence) < MIN_BLIND_AFFECT_CONFIDENCE
            or float(confidence) > 1.0
        ):
            errors.append("blind affect confidence is below the formal threshold")
    return errors


def _declared_artifact_path(value, *, relative_to: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    return (path if path.is_absolute() else relative_to / path).resolve()


def _require_fields(errors: list[str], name: str, value, expected: Mapping) -> None:
    if not isinstance(value, Mapping):
        errors.append(f"{name} must be an object")
        return
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            errors.append(f"{name}.{field} must equal {expected_value!r}")


def _official_semantic_retarget_errors(
    record: Mapping, motion: Mapping, quality_payload: Mapping
) -> list[str]:
    if record.get("annotation_kind") != "official_gesture_semantic_event":
        return []
    errors = []
    try:
        supervision = semantic_supervision_policy(record)
    except ValueError as exc:
        errors.append(f"semantic supervision contract invalid: {exc}")
        supervision = {}
    for field in (
        "canonical_action",
        "canonical_action_role",
        "semantic_mapping_status",
        "official_category_verified",
        "official_category_conditioning_enabled",
        "official_category_role",
        "official_category_condition_channel",
        "official_category_loss",
        "robot_observable_motion_form",
        "communicative_intent",
        "canonical_prompt_role",
        "semantic_supervision_masks",
        "semantic_event",
        "emotion_id",
        "emotion_review_status",
        "emotion_source",
        "emotion_protocol_contract",
        "source_emotion_label_verified",
    ):
        if quality_payload.get(field) != record.get(field):
            errors.append(f"quality JSON {field} mismatch")
    expected_pending_emotion = {
        "emotion_supervision_mask": False,
        "official_emotion_conditioning_enabled": False,
        "emotion_supervision_role": OFFICIAL_EMOTION_DISABLED_ROLE,
        "official_emotion_condition_channel": None,
        "official_emotion_loss": None,
        "affect_observable_review_status": "candidate_unreviewed",
        "affect_observable_supervision_mask": False,
    }
    for field, expected_value in expected_pending_emotion.items():
        if field not in quality_payload or quality_payload.get(field) != expected_value:
            errors.append(f"quality JSON pending {field} mismatch")
    if supervision and supervision.get("official_category") != (
        record.get("semantic_event") or {}
    ).get("category"):
        errors.append("official category supervision does not match semantic_event")
    source_segment = record.get("training_segment")
    if not isinstance(source_segment, Mapping):
        return ["official semantic event training_segment is missing"]
    if source_segment.get("representation") != FORMAL_VARIABLE_SEGMENT_REPRESENTATION:
        errors.append("official semantic event training_segment representation is invalid")
    if source_segment.get("fixed_window_sec") is not None:
        errors.append("official semantic event cannot use a fixed window")
    source_start = source_segment.get("start_frame")
    source_end = source_segment.get("end_frame_exclusive")
    source_frames = source_segment.get("frame_count")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (source_start, source_end, source_frames)
    ) or not (
        isinstance(source_start, int)
        and isinstance(source_end, int)
        and isinstance(source_frames, int)
        and source_start >= 0
        and source_end > source_start
        and source_frames == source_end - source_start
    ):
        return errors + ["official semantic event source interval is invalid"]
    output_frames = motion.get("frames")
    fps = motion.get("fps")
    if (
        isinstance(output_frames, bool)
        or not isinstance(output_frames, int)
        or output_frames < 2
        or isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(float(fps))
        or float(fps) <= 0.0
    ):
        return errors + ["motion_18d output interval is invalid"]
    retarget = motion.get("retarget_segment")
    if not isinstance(retarget, Mapping):
        return errors + ["motion_18d.retarget_segment is missing"]
    if quality_payload.get("retarget_segment") != retarget:
        errors.append("motion_18d.retarget_segment does not match quality JSON")
    payload = {key: value for key, value in retarget.items() if key != "sha256"}
    if not _content_contract_valid(retarget):
        errors.append("motion_18d.retarget_segment hash is invalid")
    expected = {
        "representation": FORMAL_RETARGET_SEGMENT_REPRESENTATION,
        "source_start_frame": source_start,
        "source_end_frame_exclusive": source_end,
        "source_frame_count": source_frames,
        "output_frame_count": output_frames,
        "fps": fps,
        "retimed": output_frames != source_frames,
        "cropped": False,
        "planner_duration_field": "output_sample_span_sec",
        "source_boundary_duration_field": "source_frame_coverage_sec",
        "legacy_quality_duration_sec_role": (
            "output_frame_coverage_compatibility_only_not_planner_target"
        ),
    }
    for field, expected_value in expected.items():
        if payload.get(field) != expected_value:
            errors.append(f"motion_18d.retarget_segment.{field} is inconsistent")
    if motion.get("source_window_frames") != source_frames:
        errors.append("motion_18d.source_window_frames does not match training_segment")
    if quality_payload.get("source_window_frames") != source_frames:
        errors.append("quality source_window_frames does not match training_segment")
    if quality_payload.get("frames") != output_frames:
        errors.append("quality output frames do not match motion_18d")
    lineage = motion.get("upstream_lineage")
    if not isinstance(lineage, Mapping):
        errors.append("motion_18d.upstream_lineage is missing")
    else:
        for field in FORMAL_SELECTED_LINEAGE_FIELDS:
            value = record.get(field)
            if not _is_sha256(value):
                errors.append(f"{field} must be an explicit SHA256")
            if lineage.get(field) != value:
                errors.append(f"motion_18d.upstream_lineage.{field} mismatch")
            if quality_payload.get(field) != value:
                errors.append(f"quality JSON {field} mismatch")
    if record.get("inventory_record_sha256") not in (
        None,
        record.get("upstream_inventory_record_sha256"),
    ):
        errors.append("inventory_record_sha256 has ambiguous overwritten lineage")
    if isinstance(fps, (int, float)) and not isinstance(fps, bool) and fps > 0:
        expected_durations = {
            "source_frame_coverage_sec": source_frames / float(fps),
            "output_sample_span_sec": max(0, output_frames - 1) / float(fps),
            "output_frame_coverage_sec": output_frames / float(fps),
        }
        for field, expected_value in expected_durations.items():
            value = retarget.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not math.isclose(
                    float(value), expected_value, rel_tol=0.0, abs_tol=1e-12
                )
            ):
                errors.append(f"motion_18d.retarget_segment.{field} is inconsistent")
    return errors


def _formal_train_ready_record_errors(record: Mapping, *, manifest_path: Path) -> list[str]:
    errors: list[str] = []
    if record.get("annotation_kind") == "official_gesture_semantic_event":
        try:
            semantic_supervision_policy(record)
        except ValueError as exc:
            errors.append(f"semantic supervision contract invalid: {exc}")
    if record.get("adjudication_schema_version") != FORMAL_ADJUDICATION_SCHEMA_VERSION:
        errors.append(
            "adjudication_schema_version must equal "
            f"{FORMAL_ADJUDICATION_SCHEMA_VERSION!r}"
        )

    adjudication = record.get("adjudication")
    _require_fields(
        errors,
        "adjudication",
        adjudication,
        {
            "status": "train_ready",
            "semantic_critical": False,
            "semantic_pending_reasons": [],
            "rejection_causes": [],
            "reasons": [],
        },
    )

    review = record.get("independent_review")
    _require_fields(
        errors,
        "independent_review",
        review,
        {
            "present": True,
            "status": "agent_reviewed",
            "training_acceptance": True,
        },
    )
    if isinstance(review, Mapping):
        if not isinstance(review.get("review_id"), str) or not review["review_id"].strip():
            errors.append("independent_review.review_id must be non-empty")
        review_gates = review.get("gates")
        if not isinstance(review_gates, Mapping) or set(review_gates) != set(
            FORMAL_REQUIRED_REVIEW_GATES
        ):
            errors.append("independent_review.gates must exactly match the 18D review contract")
        elif any(
            review_gates[gate] is not True
            for gate in FORMAL_MOTION_ADMISSION_REVIEW_GATES
        ):
            errors.append("every motion-admission independent_review gate must pass")

    motion = record.get("motion_18d")
    _require_fields(
        errors,
        "motion_18d",
        motion,
        {
            "state": "passed",
            "partition": "accepted",
            "reasons": [],
            "output_contract": CONTRACT_VERSION,
            "action_dim": ACTION_DIM,
        },
    )
    if isinstance(motion, Mapping):
        quality_payload = None
        frames = motion.get("frames")
        rows = motion.get("csv_rows")
        if isinstance(frames, bool) or not isinstance(frames, int) or frames < 2:
            errors.append("motion_18d.frames must be an integer >= 2")
        if rows != frames:
            errors.append("motion_18d.csv_rows must equal motion_18d.frames")
        fps = motion.get("fps")
        if (
            isinstance(fps, bool)
            or not isinstance(fps, (int, float))
            or not np.isfinite(float(fps))
            or not math.isclose(float(fps), 30.0, abs_tol=1e-9)
        ):
            errors.append("motion_18d.fps must be exactly 30 Hz")
        quality_gates = motion.get("quality_gate")
        if not isinstance(quality_gates, Mapping) or not FORMAL_REQUIRED_18D_GATES.issubset(
            quality_gates
        ):
            errors.append("motion_18d.quality_gate is incomplete")
        elif any(quality_gates[gate] is not True for gate in FORMAL_REQUIRED_18D_GATES):
            errors.append("every required motion_18d quality gate must pass")
        elif any(value is False for value in quality_gates.values()):
            errors.append("motion_18d.quality_gate contains a failed gate")

        quality_path = _declared_artifact_path(
            motion.get("quality_json"), relative_to=manifest_path.parent
        )
        quality_hash = motion.get("quality_sha256")
        if quality_path is None or not quality_path.is_file():
            errors.append("motion_18d.quality_json must reference an existing file")
        elif not _is_sha256(quality_hash) or sha256_file(quality_path) != quality_hash:
            errors.append("motion_18d quality JSON hash mismatch")
        else:
            try:
                quality_payload = json.loads(quality_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                quality_payload = None
            if not isinstance(quality_payload, Mapping):
                errors.append("motion_18d quality JSON must contain an object")
            else:
                for field in ("output_contract", "action_dim", "frames", "fps", "quality_gate"):
                    if quality_payload.get(field) != motion.get(field):
                        errors.append(f"motion_18d.{field} does not match quality JSON")
                if quality_payload.get("joint_order") != list(JOINT_ORDER_18D):
                    errors.append("motion_18d quality JSON joint order does not match 18D contract")
                quality_outputs = quality_payload.get("outputs")
                quality_safe_csv = (
                    quality_outputs.get("safe_csv")
                    if isinstance(quality_outputs, Mapping)
                    else None
                )
                declared_safe_csv = _declared_artifact_path(
                    motion.get("safe_csv"), relative_to=manifest_path.parent
                )
                output_safe_csv = _declared_artifact_path(
                    quality_safe_csv, relative_to=quality_path.parent
                )
                if declared_safe_csv is None or output_safe_csv != declared_safe_csv:
                    errors.append("motion_18d.safe_csv does not match quality JSON output")
                errors.extend(
                    _official_semantic_retarget_errors(record, motion, quality_payload)
                )

        if _declared_artifact_path(
            motion.get("safe_csv"), relative_to=manifest_path.parent
        ) is None:
            errors.append("motion_18d.safe_csv must be declared")
        if not _is_sha256(motion.get("safe_csv_sha256")):
            errors.append("motion_18d.safe_csv_sha256 must be a SHA256 digest")

    eligibility = record.get("training_eligibility")
    weak_behavior = _project_weak_behavior_record(record)
    supervised_behavior = bool(
        record.get("behavior_id") in KIMODO_BEHAVIOR_IDS
        and record.get("behavior_review_status") == "human_confirmed"
        and record.get("behavior_supervision_mask") is True
        and isinstance(record.get("human_review"), Mapping)
        and record["human_review"].get("reviewer_kind") == "human"
    )
    if not isinstance(eligibility, Mapping):
        errors.append("training_eligibility must be an object")
    else:
        _require_fields(
            errors,
            "training_eligibility.motion_style",
            eligibility.get("motion_style"),
            {
                "eligible": True,
                "status": "train_ready",
                "requires": ["passed_18d_qc", "independent_training_acceptance"],
            },
        )
        expected_behavior = (
            {
                "eligible": False,
                "status": "masked_project_weak_candidate",
                "one_hot_supervision_mask": False,
                "requires": [
                    "project_weak_mapping_provenance",
                    "behavior_condition_channels_zero",
                ],
            }
            if weak_behavior
            else {
                "eligible": True,
                "status": "train_ready",
                "one_hot_supervision_mask": True,
                "requires": ["human_confirmed_behavior", "motion_style_train_ready"],
            }
        )
        _require_fields(
            errors,
            "training_eligibility.behavior",
            eligibility.get("behavior"),
            expected_behavior,
        )
        is_official_event = (
            record.get("annotation_kind") == "official_gesture_semantic_event"
        )
        if is_official_event:
            official_emotion = _official_source_emotion_label_verified_record(record)
            affect_ready = bool(
                record.get("affect_observable_review_status") == "verified"
                and record.get("affect_observable_supervision_mask") is True
            )
            emotion_ready = bool(
                official_emotion
                and affect_ready
                and record.get("emotion_supervision_mask") is True
                and record.get("official_emotion_conditioning_enabled") is True
                and record.get("emotion_conditioning_mask") is True
            )
            if emotion_ready and isinstance(review, Mapping):
                if (review.get("gates") or {}).get("affect_observable_in_18d") is not True:
                    errors.append(
                        "emotion conditioning requires affect_observable_in_18d=true"
                    )
                errors.extend(_formal_blind_affect_errors(record, review))
            expected_emotion = {
                "eligible": emotion_ready,
                "status": (
                    "train_ready"
                    if emotion_ready
                    else "masked_pending_verified_blind_robot_affect_review"
                ),
                "loss_mask": emotion_ready,
                "source_label_verified": official_emotion,
                "affect_observable_verified": emotion_ready,
                "conditioning_mask": emotion_ready,
                "requires": [
                    "motion_style_train_ready",
                    "verified_official_emotion_protocol",
                    "anonymous_silent_video_sha256_bound_blind_affect_review",
                    "target_emotion_not_exposed",
                    "observed_affect_matches_official_label",
                    "affect_observable_in_18d_gate_true",
                ],
            }
            _require_fields(
                errors,
                "training_eligibility.emotion",
                eligibility.get("emotion"),
                expected_emotion,
            )
            if not official_emotion:
                errors.append("formal official event requires verified source emotion")
            expected_official_state = {
                "emotion_supervision_mask": emotion_ready,
                "official_emotion_conditioning_enabled": emotion_ready,
                "emotion_supervision_role": (
                    OFFICIAL_EMOTION_ENABLED_ROLE
                    if emotion_ready
                    else OFFICIAL_EMOTION_DISABLED_ROLE
                ),
                "official_emotion_condition_channel": (
                    OFFICIAL_EMOTION_CONDITION_CHANNEL
                    if emotion_ready
                    else None
                ),
                "official_emotion_loss": None,
                "affect_observable_review_status": (
                    "verified" if emotion_ready else "not_verified"
                ),
                "affect_observable_supervision_mask": emotion_ready,
                "emotion_conditioning_mask": emotion_ready,
            }
            for field, expected_value in expected_official_state.items():
                if field not in record or record.get(field) != expected_value:
                    errors.append(
                        f"formal official emotion field {field} does not match blind-review state"
                    )
        else:
            legacy_source_ready = bool(
                record.get("emotion_id") in KIMODO_EMOTION_IDS
                and record.get("emotion_review_status") == "human_confirmed"
                and record.get("network_semantic_supervision_ready") is True
                and record.get("source_emotion_label_verified") is True
            )
            legacy_affect_ready = bool(
                record.get("affect_observable_review_status") == "verified"
                and record.get("affect_observable_supervision_mask") is True
            )
            legacy_emotion_ready = bool(
                legacy_source_ready
                and legacy_affect_ready
                and record.get("emotion_supervision_mask") is True
                and record.get("emotion_conditioning_mask") is True
            )
            if legacy_emotion_ready and isinstance(review, Mapping):
                if (review.get("gates") or {}).get("affect_observable_in_18d") is not True:
                    errors.append(
                        "emotion conditioning requires affect_observable_in_18d=true"
                    )
                errors.extend(_formal_blind_affect_errors(record, review))
            expected_legacy_emotion = {
                "eligible": legacy_emotion_ready,
                "status": (
                    "train_ready"
                    if legacy_emotion_ready
                    else (
                        "masked_pending_verified_blind_robot_affect_review"
                        if legacy_source_ready
                        else "blocked_unresolved_emotion"
                    )
                ),
                "loss_mask": legacy_emotion_ready,
                "source_label_verified": legacy_source_ready,
                "affect_observable_verified": legacy_emotion_ready,
                "conditioning_mask": legacy_emotion_ready,
                "requires": [
                    "motion_style_train_ready",
                    "human_confirmed_emotion",
                    "verified_blind_robot_affect_review",
                ],
            }
            _require_fields(
                errors,
                "training_eligibility.emotion",
                eligibility.get("emotion"),
                expected_legacy_emotion,
            )
            expected_common_state = {
                "source_emotion_label_verified": legacy_source_ready,
                "emotion_supervision_mask": legacy_emotion_ready,
                "affect_observable_review_status": (
                    "verified" if legacy_emotion_ready else "not_verified"
                ),
                "affect_observable_supervision_mask": legacy_emotion_ready,
                "emotion_conditioning_mask": legacy_emotion_ready,
            }
            for field, expected_value in expected_common_state.items():
                if field not in record or record.get(field) != expected_value:
                    errors.append(
                        f"formal emotion field {field} does not match blind-review state"
                    )

    if not (supervised_behavior or weak_behavior):
        errors.append(
            "formal behavior must be explicitly human supervised or project-weak masked"
        )

    split = record.get("split")
    _require_fields(
        errors,
        "split",
        split,
        {"assignment": "train", "eval_eligible": False},
    )
    if isinstance(split, Mapping):
        if not isinstance(split.get("source_group_key"), str) or not split[
            "source_group_key"
        ].strip():
            errors.append("split.source_group_key must be non-empty")
        if split.get("action_key") != record.get("canonical_action"):
            errors.append("split.action_key must match canonical_action")
        expected_subject_policy = (
            "train_only_unknown" if split.get("subject_key") is None else "subject_disjoint"
        )
        if split.get("subject_policy") != expected_subject_policy:
            errors.append("split.subject_policy does not match subject_key")
    if not isinstance(record.get("canonical_action"), str) or not record[
        "canonical_action"
    ].strip():
        errors.append("canonical_action must be non-empty")
    return errors


def _validate_formal_release_report(
    manifest_path: Path, *, manifest_sha256: str, record_count: int
) -> None:
    report_path = manifest_path.parent / FORMAL_RELEASE_REPORT_FILENAME
    if not report_path.is_file():
        raise ValueError(
            f"formal train-ready manifest requires sibling {FORMAL_RELEASE_REPORT_FILENAME}"
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid formal train-ready release report: {error}") from error
    errors = []
    if not isinstance(report, Mapping):
        errors.append("report must be an object")
    else:
        if report.get("schema_version") != 1:
            errors.append("schema_version must equal 1")
        if report.get("adjudication_schema_version") != FORMAL_ADJUDICATION_SCHEMA_VERSION:
            errors.append("adjudication_schema_version mismatch")
        output = (report.get("outputs") or {}).get("train_ready")
        if not isinstance(output, Mapping):
            errors.append("outputs.train_ready must be an object")
        else:
            output_path = _declared_artifact_path(
                output.get("path"), relative_to=report_path.parent
            )
            if output_path != manifest_path.resolve():
                errors.append("outputs.train_ready.path does not bind this manifest")
            if output.get("records") != record_count:
                errors.append("outputs.train_ready.records does not match manifest")
            if output.get("sha256") != manifest_sha256:
                errors.append("outputs.train_ready.sha256 does not match manifest")
        if (report.get("counts") or {}).get("train_ready") != record_count:
            errors.append("counts.train_ready does not match manifest")
        if (report.get("scale") or {}).get("train_ready_clips") != record_count:
            errors.append("scale.train_ready_clips does not match manifest")
        invariants = report.get("invariants")
        if not isinstance(invariants, Mapping) or not FORMAL_REQUIRED_RELEASE_INVARIANTS.issubset(
            invariants
        ):
            errors.append("release invariants are incomplete")
        elif any(invariants[name] is not True for name in FORMAL_REQUIRED_RELEASE_INVARIANTS):
            errors.append("every required release invariant must pass")
    if errors:
        raise ValueError("invalid formal train-ready release report: " + "; ".join(errors))


def _motion_only_train_ready_record_errors(
    record: Mapping, *, manifest_path: Path
) -> list[str]:
    """Validate physical-QC-only admission without implying semantic review."""

    errors: list[str] = []
    if record.get("formal_episode_contract") != MOTION_ONLY_EPISODE_CONTRACT:
        errors.append("formal_episode_contract is not the motion-only contract")
    if record.get("accepted_for_training") is not True:
        errors.append("motion-only train-ready record must be explicitly accepted")
    _require_fields(
        errors,
        "adjudication",
        record.get("adjudication"),
        {"status": "motion_only_train_ready", "reasons": []},
    )
    admission = record.get("motion_only_admission")
    _require_fields(
        errors,
        "motion_only_admission",
        admission,
        {
            "physical_qc_only": True,
            "semantic_review_required": False,
            "independent_semantic_review_claimed": False,
            "text_conditioning_enabled": False,
            "emotion_conditioning_enabled": False,
            "audio_conditioning_enabled": False,
            "native_variable_length": True,
            "fixed_duration_training_unit": False,
        },
    )
    if not isinstance(admission, Mapping) or not _is_sha256(
        admission.get("source_record_sha256")
    ):
        errors.append("motion_only_admission.source_record_sha256 is invalid")

    if record.get("semantic_supervision_masks") != FORMAL_SEMANTIC_SUPERVISION_MASKS:
        errors.append("all semantic supervision must be masked for motion-only pretraining")
    for field in (
        "behavior_supervision_mask",
        "emotion_supervision_mask",
        "affect_observable_supervision_mask",
        "emotion_conditioning_mask",
        "official_category_conditioning_enabled",
        "official_emotion_conditioning_enabled",
    ):
        if record.get(field) is not False:
            errors.append(f"{field} must be false for motion-only pretraining")
    review = record.get("independent_review")
    if isinstance(review, Mapping) and (
        review.get("training_acceptance") is True
        or review.get("status") in {"agent_reviewed", "human_confirmed"}
    ):
        errors.append("motion-only admission cannot claim semantic review acceptance")

    motion = record.get("motion_18d")
    _require_fields(
        errors,
        "motion_18d",
        motion,
        {
            "state": "passed",
            "partition": "accepted_motion_only",
            "reasons": [],
            "output_contract": CONTRACT_VERSION,
            "action_dim": ACTION_DIM,
        },
    )
    if isinstance(motion, Mapping):
        frames = motion.get("frames")
        if isinstance(frames, bool) or not isinstance(frames, int) or frames < 2:
            errors.append("motion_18d.frames must be an integer >= 2")
        if motion.get("csv_rows") != frames:
            errors.append("motion_18d.csv_rows must equal motion_18d.frames")
        fps = motion.get("fps")
        if (
            isinstance(fps, bool)
            or not isinstance(fps, (int, float))
            or not math.isclose(float(fps), 30.0, abs_tol=1e-9)
        ):
            errors.append("motion_18d.fps must be exactly 30 Hz")
        quality_gates = motion.get("quality_gate")
        if not isinstance(quality_gates, Mapping) or not FORMAL_REQUIRED_18D_GATES.issubset(
            quality_gates
        ):
            errors.append("motion_18d.quality_gate is incomplete")
        elif any(value is not True for value in quality_gates.values()):
            errors.append("every declared motion_18d quality gate must pass")

        quality_path = _declared_artifact_path(
            motion.get("quality_json"), relative_to=manifest_path.parent
        )
        quality_hash = motion.get("quality_sha256")
        if quality_path is None or not quality_path.is_file():
            errors.append("motion_18d.quality_json must reference an existing file")
        elif not _is_sha256(quality_hash) or sha256_file(quality_path) != quality_hash:
            errors.append("motion_18d quality JSON hash mismatch")
        else:
            try:
                quality = json.loads(quality_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                quality = None
            if not isinstance(quality, Mapping):
                errors.append("motion_18d quality JSON must contain an object")
            else:
                for field in ("output_contract", "action_dim", "frames", "fps", "quality_gate"):
                    if quality.get(field) != motion.get(field):
                        errors.append(f"motion_18d.{field} does not match quality JSON")
                if quality.get("joint_order") != list(JOINT_ORDER_18D):
                    errors.append("motion_18d quality JSON joint order does not match 18D contract")
                outputs = quality.get("outputs")
                quality_csv = outputs.get("safe_csv") if isinstance(outputs, Mapping) else None
                declared_csv = _declared_artifact_path(
                    motion.get("safe_csv"), relative_to=manifest_path.parent
                )
                resolved_quality_csv = _declared_artifact_path(
                    quality_csv, relative_to=quality_path.parent
                )
                if declared_csv is None or resolved_quality_csv != declared_csv:
                    errors.append("motion_18d.safe_csv does not match quality JSON output")
        if not _is_sha256(motion.get("safe_csv_sha256")):
            errors.append("motion_18d.safe_csv_sha256 must be a SHA256 digest")

    segment = record.get("training_segment")
    if not isinstance(segment, Mapping):
        errors.append("training_segment must be present")
    else:
        if segment.get("representation") != FORMAL_VARIABLE_SEGMENT_REPRESENTATION:
            errors.append("training_segment must preserve the native variable-length contract")
        if segment.get("fixed_window_sec") is not None:
            errors.append("motion-only pretraining cannot use a fixed duration window")
    retarget = (motion or {}).get("retarget_segment") if isinstance(motion, Mapping) else None
    if not isinstance(retarget, Mapping):
        errors.append("motion_18d.retarget_segment must be present")
    elif retarget.get("cropped") is not False:
        errors.append("motion-only trajectory cannot be cropped by the retarget stage")
    return errors


def _validate_motion_only_release_report(
    manifest_path: Path, *, manifest_sha256: str, record_count: int
) -> None:
    report_path = manifest_path.parent / MOTION_ONLY_RELEASE_REPORT_FILENAME
    if not report_path.is_file():
        raise ValueError(
            f"motion-only manifest requires sibling {MOTION_ONLY_RELEASE_REPORT_FILENAME}"
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid motion-only release report: {error}") from error
    errors: list[str] = []
    if not isinstance(report, Mapping):
        errors.append("report must be an object")
    else:
        if report.get("schema_version") != 1:
            errors.append("schema_version must equal 1")
        if report.get("formal_episode_contract") != MOTION_ONLY_EPISODE_CONTRACT:
            errors.append("formal_episode_contract mismatch")
        output = (report.get("outputs") or {}).get("train_ready")
        if not isinstance(output, Mapping):
            errors.append("outputs.train_ready must be an object")
        else:
            output_path = _declared_artifact_path(
                output.get("path"), relative_to=report_path.parent
            )
            if output_path != manifest_path.resolve():
                errors.append("outputs.train_ready.path does not bind this manifest")
            if output.get("records") != record_count:
                errors.append("outputs.train_ready.records does not match manifest")
            if output.get("sha256") != manifest_sha256:
                errors.append("outputs.train_ready.sha256 does not match manifest")
        invariants = report.get("invariants")
        if not isinstance(invariants, Mapping) or not MOTION_ONLY_REQUIRED_RELEASE_INVARIANTS.issubset(
            invariants
        ):
            errors.append("motion-only release invariants are incomplete")
        elif any(
            invariants[name] is not True
            for name in MOTION_ONLY_REQUIRED_RELEASE_INVARIANTS
        ):
            errors.append("every motion-only release invariant must pass")
    if errors:
        raise ValueError("invalid motion-only release report: " + "; ".join(errors))


def _formal_motion_file_errors(
    record: Mapping,
    *,
    manifest_path: Path,
    motion_path: Path,
    values: np.ndarray,
) -> list[str]:
    motion = record.get("motion_18d") or {}
    errors = []
    declared_path = _declared_artifact_path(
        motion.get("safe_csv"), relative_to=manifest_path.parent
    )
    if declared_path != motion_path.resolve():
        errors.append("motion_18d.safe_csv does not match loaded trajectory")
    actual_sha256 = sha256_file(motion_path)
    if motion.get("safe_csv_sha256") != actual_sha256:
        errors.append("18D trajectory hash mismatch")
    if motion.get("frames") != int(values.shape[0]):
        errors.append("motion_18d.frames does not match trajectory row count")
    if motion.get("csv_rows") != int(values.shape[0]):
        errors.append("motion_18d.csv_rows does not match trajectory row count")
    return errors


def read_joint_csv(path: str | Path, *, expected_joint_order=JOINT_ORDER_18D) -> np.ndarray:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        if fields == ["time_sec", *expected_joint_order]:
            fields = fields[1:]
        elif fields != list(expected_joint_order):
            raise ValueError(
                f"{path} joint columns must exactly match the {len(expected_joint_order)}D contract"
            )
        rows = [
            [float(row[name]) for name in expected_joint_order]
            for row in reader
        ]
    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != len(expected_joint_order):
        raise ValueError(f"{path} does not contain a usable trajectory")
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite values")
    return values


def load_18d_episodes(
    *,
    manifest: str | Path | None = None,
    motion_root: str | Path | None = None,
    semantics: str | Path | None = None,
    allow_unreviewed=False,
) -> list[dict]:
    """Load strict 18D CSV episodes from a merged manifest or a motion-root/semantics pair."""
    if manifest is None and (motion_root is None or semantics is None):
        raise ValueError("provide --manifest or both --motion-root and --semantics")
    for field, value in (
        ("manifest", manifest),
        ("motion_root", motion_root),
        ("semantics", semantics),
    ):
        if value is not None:
            assert_no_forbidden_data_lineage(
                {field: str(value)}, context="18d_episode_loader"
            )
    if manifest is not None:
        manifest_path = Path(manifest)
        source_records = _read_jsonl(manifest_path)
        source_mode = "adjudicated_manifest"
    else:
        semantics_path = Path(semantics)
        semantic_by_clip = {
            str(record["clip_id"]): record for record in _read_jsonl(semantics_path)
        }
        source_records = []
        for path in sorted(Path(motion_root).glob("**/*_gmr_safe_18d.csv")):
            clip_id = path.name.removesuffix("_gmr_safe_18d.csv")
            if clip_id not in semantic_by_clip:
                continue
            record = deepcopy(semantic_by_clip[clip_id])
            record["trajectory_path"] = str(path.resolve())
            source_records.append(record)
        manifest_path = semantics_path
        source_mode = "motion_root_semantics_join"

    manifest_sha256 = sha256_file(manifest_path)

    episodes = []
    seen = set()
    release_report_validated = False
    for record in source_records:
        assert_no_forbidden_data_lineage(
            record, context="18d_episode_manifest_record"
        )
        clip_id = str(record.get("clip_id") or record.get("sample_id") or "").strip()
        if not clip_id or clip_id in seen:
            raise ValueError(f"missing or duplicate clip_id: {clip_id!r}")
        semantics = _resolve_structured_semantics(record)
        legacy_semantics = _resolve_legacy_semantics(record)
        review = record.get("review_status") or {}
        adjudication = record.get("adjudication") or {}
        independent_review = record.get("independent_review") or {}
        motion_18d = record.get("motion_18d") or {}
        standard_train_ready = (
            isinstance(adjudication, Mapping)
            and adjudication.get("status") == "train_ready"
        )
        motion_only_train_ready = bool(
            record.get("formal_episode_contract") == MOTION_ONLY_EPISODE_CONTRACT
            and isinstance(adjudication, Mapping)
            and adjudication.get("status") == "motion_only_train_ready"
        )
        claims_train_ready = standard_train_ready or motion_only_train_ready
        # Using the unsafe flag is an explicit experimental admission path.  It
        # never upgrades a record to formal status, even when the record carries
        # fields that claim it was adjudicated.
        accepted = False
        if claims_train_ready and not allow_unreviewed:
            errors = (
                _motion_only_train_ready_record_errors(
                    record, manifest_path=manifest_path
                )
                if motion_only_train_ready
                else _formal_train_ready_record_errors(
                    record, manifest_path=manifest_path
                )
            )
            if errors:
                raise ValueError(
                    f"{clip_id}: invalid formal train-ready record: " + "; ".join(errors)
                )
            if not release_report_validated:
                validator = (
                    _validate_motion_only_release_report
                    if motion_only_train_ready
                    else _validate_formal_release_report
                )
                validator(
                    manifest_path,
                    manifest_sha256=manifest_sha256,
                    record_count=len(source_records),
                )
                release_report_validated = True
            accepted = True
        if not accepted and not allow_unreviewed:
            continue
        motion_path = _resolve_motion_path(record, manifest_path=manifest_path)
        if motion_path is None or not motion_path.is_file():
            raise FileNotFoundError(f"18D trajectory for {clip_id} is missing: {motion_path}")
        values = read_joint_csv(motion_path)
        trajectory_sha256 = sha256_file(motion_path)
        if accepted:
            motion_errors = _formal_motion_file_errors(
                record,
                manifest_path=manifest_path,
                motion_path=motion_path,
                values=values,
            )
            if motion_errors:
                raise ValueError(
                    f"{clip_id}: invalid formal 18D motion evidence: "
                    + "; ".join(motion_errors)
                )
        fps = float(
            motion_18d.get("fps")
            if accepted
            else record.get("fps") or (record.get("time_window") or {}).get("fps") or 30.0
        )
        prompt = _resolve_prompt(record)
        split = record.get("split") or {}
        source_metadata_fields = (
            "official_split",
            "emotion_label_source",
            "source_emotion_label",
            "semantic_label_status",
            "official_gesture_semantic_spans",
            "audio_policy",
            "annotation_relpath",
            "textgrid_relpath",
            "motion_relpath",
            "window_transcript_context",
            "window_transcript_role",
            "inventory_manifest_sha256",
            "inventory_record_sha256",
            "upstream_inventory_record_sha256",
            "selected_record_sha256",
            "pilot_selector_contract_sha256",
            "pilot_speaker_group_sha256",
            "pilot_source_group_sha256",
            "prompt_schema",
            "prompt_contract",
            "fixed_split_assignment",
            "canonical_action",
            "canonical_action_role",
            "semantic_mapping_status",
            "official_category_verified",
            "official_category_conditioning_enabled",
            "official_category_role",
            "official_category_condition_channel",
            "official_category_loss",
            "robot_observable_motion_form",
            "communicative_intent",
            "canonical_prompt_role",
            "semantic_supervision_masks",
            "source_emotion_label_verified",
            "official_emotion_conditioning_enabled",
            "emotion_supervision_role",
            "official_emotion_condition_channel",
            "official_emotion_loss",
            "affect_observable_review_status",
            "affect_observable_supervision_mask",
            "emotion_conditioning_mask",
        )
        formal_source_metadata = {
            field: deepcopy(record[field])
            for field in source_metadata_fields
            if field in record
        }
        source_record_sha256 = hashlib.sha256(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        episodes.append(
            {
                "clip_id": clip_id,
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                **semantics,
                **legacy_semantics,
                "actions": values,
                "fps": fps,
                "duration_sec": float((values.shape[0] - 1) / fps),
                "trajectory_path": str(motion_path.resolve()),
                "trajectory_sha256": trajectory_sha256,
                "accepted_for_training": accepted,
                "review_state": str(
                    adjudication.get("status") or review.get("state") or "unspecified"
                ),
                "eligibility_mode": (
                    "adjudicated_train_ready" if accepted else "unsafe_allow_unreviewed"
                ),
                "source_mode": source_mode,
                "source_manifest": str(manifest_path.resolve()),
                "source_manifest_sha256": manifest_sha256,
                "formal_episode_contract": record.get("formal_episode_contract"),
                "motion_only_admission": deepcopy(
                    record.get("motion_only_admission")
                ),
                "source_clip_id": record.get("source_clip_id"),
                "source_sha256": record.get("source_sha256"),
                "training_segment": deepcopy(record.get("training_segment")),
                "retarget_segment": deepcopy(motion_18d.get("retarget_segment")),
                "quality_source_window_frames": motion_18d.get(
                    "source_window_frames"
                ),
                "quality_output_frame_count": motion_18d.get("frames"),
                "retarget_source_lineage": deepcopy(
                    motion_18d.get("upstream_lineage")
                ),
                "window": deepcopy(record.get("window")),
                "selection_status": record.get("selection_status"),
                "annotation_kind": record.get("annotation_kind"),
                "semantic_event": deepcopy(record.get("semantic_event")),
                "canonical_action": record.get("canonical_action"),
                "canonical_action_role": record.get("canonical_action_role"),
                "semantic_mapping_status": record.get("semantic_mapping_status"),
                "official_category_verified": record.get(
                    "official_category_verified"
                ),
                "official_category_conditioning_enabled": record.get(
                    "official_category_conditioning_enabled"
                ),
                "official_category_role": record.get("official_category_role"),
                "official_category_condition_channel": record.get(
                    "official_category_condition_channel"
                ),
                "official_category_loss": record.get("official_category_loss"),
                "robot_observable_motion_form": record.get(
                    "robot_observable_motion_form"
                ),
                "communicative_intent": record.get("communicative_intent"),
                "canonical_prompt_role": record.get("canonical_prompt_role"),
                "semantic_supervision_masks": deepcopy(
                    record.get("semantic_supervision_masks")
                ),
                "interaction_scope": record.get("interaction_scope"),
                "behavior_source": record.get("behavior_source"),
                "behavior_mapping_contract": deepcopy(
                    record.get("behavior_mapping_contract")
                ),
                "human_review": deepcopy(record.get("human_review")),
                "emotion_source": record.get("emotion_source"),
                "emotion_protocol_contract": deepcopy(
                    record.get("emotion_protocol_contract")
                ),
                "independent_review": deepcopy(record.get("independent_review")),
                "retarget_qc_passed": bool(
                    accepted
                    and motion_18d.get("state") == "passed"
                    and isinstance(motion_18d.get("quality_gate"), Mapping)
                    and motion_18d["quality_gate"]
                    and all(
                        value is True
                        for value in motion_18d["quality_gate"].values()
                    )
                ),
                "formal_source_metadata": formal_source_metadata,
                "source_record_sha256": source_record_sha256,
                "fixed_split_assignment": (
                    record.get("fixed_split_assignment")
                    or split.get("assignment")
                ),
                "split_assignment": split.get("assignment"),
                "eval_eligible": bool(split.get("eval_eligible", False)),
            }
        )
        seen.add(clip_id)
    if not episodes:
        suffix = " (use --allow-unreviewed only for an explicit smoke test)" if not allow_unreviewed else ""
        raise ValueError("no eligible 18D episodes were found" + suffix)
    return episodes


def build_condition_cache(
    episodes: Sequence[Mapping],
    qwen_checkpoint: str | Path,
    output_path: str | Path,
    *,
    base_checkpoint: str | Path,
    device="auto",
    local_files_only=True,
    batch_size=16,
) -> dict:
    output_path = Path(output_path)
    if output_path.suffix != ".npz":
        raise ValueError("condition cache output must use the .npz suffix")
    if not episodes:
        raise ValueError("at least one structured 18D episode is required")
    prompts = [_resolve_prompt(item) for item in episodes]
    clip_ids = [str(item.get("clip_id") or item.get("sample_id") or "").strip() for item in episodes]
    if any(not clip_id for clip_id in clip_ids) or len(set(clip_ids)) != len(clip_ids):
        raise ValueError("condition cache episodes require unique non-empty clip ids")
    semantics = [_resolve_structured_semantics(item) for item in episodes]
    legacy_semantics = [_resolve_legacy_semantics(item) for item in episodes]
    semantic_supervision = [semantic_supervision_policy(item) for item in episodes]
    generator_checkpoint, style_contract = _load_target_generator_contract(base_checkpoint)
    validate_qwen_checkpoint_for_generator(generator_checkpoint, qwen_checkpoint)
    style_features = []
    style_controls = []
    for episode in episodes:
        actions = np.asarray(episode.get("actions"), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or actions.shape[0] < 1:
            raise ValueError(
                f"condition cache requires 18D actions for {episode.get('clip_id')!r}"
            )
        features = extract_style_features(
            actions[:, :LEGACY_ACTION_DIM],
            fps=float(episode.get("fps") or 30.0),
        )
        style_features.append(features)
        style_controls.append(normalize_style_features(features, style_contract))
    try:
        from upper_body_skeleton.cross_modal_latent import load_qwen_motion_text_encoder
    except ImportError as exc:
        raise RuntimeError(
            "Qwen condition caching requires requirements-semantic-adapter.txt"
        ) from exc
    encoder, qwen_payload = load_qwen_motion_text_encoder(
        qwen_checkpoint, device=device, local_files_only=local_files_only
    )
    latent_dim = KIMODO_V2_CONDITION_DIM - KIMODO_CONDITION_DIM
    latents = np.zeros((len(episodes), latent_dim), dtype=np.float32)
    prompt_supervised_indices = [
        index
        for index, policy in enumerate(semantic_supervision)
        if policy["semantic_supervision_masks"]["prompt_text"] is True
    ]
    if prompt_supervised_indices:
        encoded = np.asarray(
            encoder.encode(
                [prompts[index] for index in prompt_supervised_indices],
                batch_size=int(batch_size),
            ),
            dtype=np.float32,
        )
        if encoded.shape != (len(prompt_supervised_indices), latent_dim):
            raise ValueError("Qwen encoder returned an invalid motion-latent shape")
        latents[prompt_supervised_indices] = encoded
    bases = np.stack(
        [
            _structured_condition_base(
                prompt,
                labels,
                legacy,
                controls,
                semantic_supervision=policy["semantic_supervision_masks"],
            )
            for prompt, labels, legacy, controls, policy in zip(
                prompts,
                semantics,
                legacy_semantics,
                style_controls,
                semantic_supervision,
            )
        ]
    ).astype(np.float32)
    conditions = np.concatenate([bases, latents.astype(np.float32)], axis=-1)
    if conditions.shape != (len(episodes), KIMODO_V2_CONDITION_DIM):
        raise RuntimeError(f"condition cache has unexpected shape {conditions.shape}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        clip_ids=np.asarray(clip_ids),
        prompts=np.asarray(prompts),
        conditions=conditions,
        behavior_ids=np.asarray([item["behavior_id"] for item in semantics]),
        behavior_review_statuses=np.asarray(
            [item["behavior_review_status"] for item in semantics]
        ),
        behavior_supervision_mask=np.asarray(
            [item["behavior_supervision_mask"] for item in semantics], dtype=np.bool_
        ),
        emotion_ids=np.asarray([item["emotion_id"] or "" for item in semantics]),
        emotion_supervision_mask=np.asarray(
            [item["emotion_supervision_mask"] for item in semantics], dtype=np.bool_
        ),
        affect_observable_review_statuses=np.asarray(
            [item["affect_observable_review_status"] for item in semantics]
        ),
        affect_observable_supervision_mask=np.asarray(
            [
                item["affect_observable_supervision_mask"]
                for item in semantics
            ],
            dtype=np.bool_,
        ),
        emotion_conditioning_mask=np.asarray(
            [item["emotion_conditioning_mask"] for item in semantics],
            dtype=np.bool_,
        ),
        canonical_prompt_roles=np.asarray(
            [item["canonical_prompt_role"] for item in semantic_supervision]
        ),
        official_category_supervision_mask=np.asarray(
            [
                item["semantic_supervision_masks"]["official_category"]
                for item in semantic_supervision
            ],
            dtype=np.bool_,
        ),
        robot_observable_motion_form_supervision_mask=np.asarray(
            [
                item["semantic_supervision_masks"]["robot_observable_motion_form"]
                for item in semantic_supervision
            ],
            dtype=np.bool_,
        ),
        communicative_intent_supervision_mask=np.asarray(
            [
                item["semantic_supervision_masks"]["communicative_intent"]
                for item in semantic_supervision
            ],
            dtype=np.bool_,
        ),
        prompt_text_supervision_mask=np.asarray(
            [
                item["semantic_supervision_masks"]["prompt_text"]
                for item in semantic_supervision
            ],
            dtype=np.bool_,
        ),
        legacy_gesture_supervision_mask=np.asarray(
            [
                item["semantic_supervision_masks"]["legacy_gesture"]
                for item in semantic_supervision
            ],
            dtype=np.bool_,
        ),
        style_features=np.stack(style_features).astype(np.float32),
        style_controls=np.stack(style_controls).astype(np.float32),
    )
    metadata = {
        "schema_version": CONDITION_CACHE_SCHEMA_VERSION,
        "artifact_kind": "ula_v2_qwen_motion_condition_cache",
        "condition_dim": KIMODO_V2_CONDITION_DIM,
        "base_condition_dim": KIMODO_CONDITION_DIM,
        "motion_latent_dim": KIMODO_V2_CONDITION_DIM - KIMODO_CONDITION_DIM,
        "count": len(episodes),
        "qwen_checkpoint": str(Path(qwen_checkpoint).resolve()),
        "qwen_checkpoint_sha256": sha256_file(qwen_checkpoint),
        "qwen_model_name": (qwen_payload.get("qwen") or {}).get("model_name"),
        "qwen_revision": (qwen_payload.get("qwen") or {}).get("revision"),
        "generator_checkpoint": str(Path(base_checkpoint).resolve()),
        "generator_checkpoint_sha256": sha256_file(base_checkpoint),
        "style_contract_sha256": style_contract["sha256"],
        "cache_sha256": sha256_file(output_path),
        "semantic_condition_contract": {
            "version": 3,
            "behavior_ids": list(KIMODO_BEHAVIOR_IDS),
            "emotion_ids": list(KIMODO_EMOTION_IDS),
            "emotion_one_hot_slice": [
                KIMODO_EMOTION_SLICE.start,
                KIMODO_EMOTION_SLICE.stop,
            ],
            "behavior_one_hot_slice": [
                KIMODO_BEHAVIOR_SLICE.start,
                KIMODO_BEHAVIOR_SLICE.stop,
            ],
            "behavior_family_slice": [
                KIMODO_BEHAVIOR_FAMILY_SLICE.start,
                KIMODO_BEHAVIOR_FAMILY_SLICE.stop,
            ],
            "unsupervised_behavior_encoding": (
                "zero_kimodo_behavior_and_family_one_hot"
            ),
            "unresolved_emotion_encoding": "zero_kimodo_emotion_one_hot",
            "emotion_conditioning_requires": [
                "verified_source_emotion_label",
                "verified_robot_affect_observable_in_18d",
            ],
            "masked_affect_encoding": (
                "zero_kimodo_emotion_one_hot_and_legacy_affect_slice"
            ),
            "legacy_semantics_source": "explicit_fields_no_prompt_inference",
            "missing_legacy_defaults": dict(LEGACY_SEMANTIC_DEFAULTS),
            "motion_style_mapping": dict(MOTION_STYLE_TO_LEGACY),
            "kimodo_emotion_to_legacy_affect": dict(
                KIMODO_EMOTION_TO_LEGACY_AFFECT
            ),
            "style_controls_source": "18d_actions_15d_prefix",
            "style_control_slice": [STYLE_CONTROL_SLICE.start, STYLE_CONTROL_SLICE.stop],
            "formal_semantic_supervision_masks": dict(
                FORMAL_SEMANTIC_SUPERVISION_MASKS
            ),
            "official_category_conditioning_enabled": False,
            "official_category_role": OFFICIAL_CATEGORY_CONDITIONING_ROLE,
            "official_category_condition_channel": None,
            "official_category_loss": None,
            "official_category_conditioned_count": 0,
            "formal_optimization_targets": [
                "motion_flow_18d",
                "head_3dof",
                "trajectory_style",
                "verified_observable_emotion_only",
                "native_duration",
            ],
            "forbidden_category_routes": [
                "qwen_motion_latent",
                "legacy_gesture",
                "legacy_intent",
                "kimodo_behavior",
                "kimodo_emotion",
            ],
            "masked_prompt_text_encoding": "zero_qwen_motion_latent_128d",
            "masked_communicative_intent_encoding": "zero_legacy_intent_slice",
            "masked_legacy_gesture_encoding": "zero_legacy_gesture_slice",
        },
        "emotion_supervised_count": sum(
            item["emotion_supervision_mask"] is True for item in semantics
        ),
        "emotion_unresolved_count": sum(
            item["emotion_supervision_mask"] is False for item in semantics
        ),
        "affect_observable_verified_count": sum(
            item["affect_observable_supervision_mask"] is True
            for item in semantics
        ),
        "emotion_conditioned_count": sum(
            item["emotion_conditioning_mask"] is True for item in semantics
        ),
        "official_category_conditioned_count": 0,
        "behavior_supervised_count": sum(
            item["behavior_supervision_mask"] is True for item in semantics
        ),
        "behavior_unsupervised_count": sum(
            item["behavior_supervision_mask"] is False for item in semantics
        ),
        "episodes": [
            {
                "clip_id": clip_id,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "annotation_kind": episode.get("annotation_kind"),
                "semantic_event": deepcopy(episode.get("semantic_event")),
                "canonical_action": episode.get("canonical_action"),
                "emotion_source": episode.get("emotion_source"),
                "emotion_protocol_contract": deepcopy(
                    episode.get("emotion_protocol_contract")
                ),
                "source_sha256": episode.get("source_sha256"),
                "behavior_source": episode.get("behavior_source"),
                "behavior_mapping_contract": deepcopy(
                    episode.get("behavior_mapping_contract")
                ),
                **labels,
                **legacy,
                **policy,
                "style_features": features.tolist(),
                "style_controls": controls.tolist(),
            }
            for clip_id, prompt, episode, labels, legacy, policy, features, controls in zip(
                clip_ids,
                prompts,
                episodes,
                semantics,
                legacy_semantics,
                semantic_supervision,
                style_features,
                style_controls,
            )
        ],
    }
    _atomic_json_save(metadata, output_path.with_suffix(output_path.suffix + ".json"))
    return metadata


def build_motion_only_condition_cache(
    episodes: Sequence[Mapping],
    output_path: str | Path,
    *,
    base_checkpoint: str | Path,
) -> dict:
    """Build a BEAT2 motion-only cache without importing or loading Qwen."""
    output_path = Path(output_path)
    if output_path.suffix != ".npz":
        raise ValueError("condition cache output must use the .npz suffix")
    if not episodes:
        raise ValueError("at least one motion-only 18D episode is required")
    clip_ids = [
        str(item.get("clip_id") or item.get("sample_id") or "").strip()
        for item in episodes
    ]
    if any(not clip_id for clip_id in clip_ids) or len(set(clip_ids)) != len(
        clip_ids
    ):
        raise ValueError("motion-only cache requires unique non-empty clip ids")
    prompts = [_resolve_prompt(item) for item in episodes]
    for clip_id, episode in zip(clip_ids, episodes, strict=True):
        if episode.get("formal_episode_contract") != MOTION_ONLY_EPISODE_CONTRACT:
            raise ValueError(f"{clip_id}: cache input is not motion-only")
        if episode.get("behavior_supervision_mask") is not False:
            raise ValueError(f"{clip_id}: behavior supervision must be disabled")
        for field in (
            "emotion_supervision_mask",
            "affect_observable_supervision_mask",
            "emotion_conditioning_mask",
            "official_category_conditioning_enabled",
            "official_emotion_conditioning_enabled",
        ):
            if episode.get(field) is not False:
                raise ValueError(f"{clip_id}: {field} must be false")
        if episode.get("semantic_supervision_masks") != (
            FORMAL_SEMANTIC_SUPERVISION_MASKS
        ):
            raise ValueError(f"{clip_id}: every semantic channel must be masked")
        if episode.get("fixed_split_assignment") not in {
            "train",
            "validation",
            "test",
        }:
            raise ValueError(f"{clip_id}: fixed manifest split is missing")

    generator_checkpoint, style_contract = _load_target_generator_contract(
        base_checkpoint
    )
    validate_motion_only_checkpoint_isolation(generator_checkpoint)
    split_contract = (generator_checkpoint.get("v2_contracts") or {}).get("split")
    data_source_registry = (
        (generator_checkpoint.get("sources") or {}).get(
            "data_source_registry"
        )
    )
    if data_source_registry is not None:
        data_source_registry = validate_data_source_registry_contract(
            data_source_registry,
            expected_role=GENERATOR_FOUNDATION_ROLE,
            expected_dataset_sources=[
                episode.get("dataset_source") for episode in episodes
            ],
        )
    split_assignment = {
        str(record.get("clip_id")): record.get("split")
        for record in (split_contract or {}).get("episodes") or ()
    }
    if split_assignment != {
        clip_id: episode.get("fixed_split_assignment")
        for clip_id, episode in zip(clip_ids, episodes, strict=True)
    }:
        raise ValueError(
            "motion-only cache episodes do not match the checkpoint fixed split"
        )

    style_features = []
    style_controls = []
    for clip_id, episode in zip(clip_ids, episodes, strict=True):
        actions = np.asarray(episode.get("actions"), dtype=np.float32)
        if (
            actions.ndim != 2
            or actions.shape[1] != ACTION_DIM
            or actions.shape[0] < 2
            or not np.isfinite(actions).all()
        ):
            raise ValueError(f"{clip_id}: motion-only cache requires finite 18D actions")
        features = extract_style_features(
            actions[:, :LEGACY_ACTION_DIM],
            fps=float(episode.get("fps") or 30.0),
        )
        style_features.append(features)
        style_controls.append(normalize_style_features(features, style_contract))
    style_features_array = np.stack(style_features).astype(np.float32)
    style_controls_array = np.stack(style_controls).astype(np.float32)
    conditions = np.zeros(
        (len(episodes), KIMODO_V2_CONDITION_DIM), dtype=np.float32
    )
    conditions[:, STYLE_CONTROL_SLICE] = style_controls_array
    validate_motion_only_style_condition(
        conditions, context="motion-only condition cache"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        clip_ids=np.asarray(clip_ids),
        prompts=np.asarray(prompts),
        conditions=conditions,
        style_features=style_features_array,
        style_controls=style_controls_array,
    )
    metadata = {
        "schema_version": MOTION_ONLY_CONDITION_CACHE_SCHEMA_VERSION,
        "artifact_kind": MOTION_ONLY_CONDITION_CACHE_ARTIFACT_KIND,
        "formal_episode_contract": MOTION_ONLY_EPISODE_CONTRACT,
        "condition_dim": KIMODO_V2_CONDITION_DIM,
        "count": len(episodes),
        "generator_checkpoint": str(Path(base_checkpoint).resolve()),
        "generator_checkpoint_sha256": sha256_file(base_checkpoint),
        "style_contract_sha256": style_contract["sha256"],
        "split_contract_sha256": split_contract["sha256"],
        **(
            {
                "data_source_registry": data_source_registry,
                DATA_SOURCE_REGISTRY_HASH_FIELD: data_source_registry["sha256"],
            }
            if data_source_registry is not None
            else {}
        ),
        "cache_sha256": sha256_file(output_path),
        "qwen_policy": MOTION_ONLY_NO_QWEN_POLICY,
        "kimodo_policy": MOTION_ONLY_NO_KIMODO_POLICY,
        "condition_policy": MOTION_ONLY_STYLE_ONLY_CONDITION_POLICY,
        "condition_nonzero_indices": list(
            range(STYLE_CONTROL_SLICE.start, STYLE_CONTROL_SLICE.stop)
        ),
        "condition_exact_zero_ranges": [
            [0, STYLE_CONTROL_SLICE.start],
            [STYLE_CONTROL_SLICE.stop, KIMODO_V2_CONDITION_DIM],
        ],
        "episodes": [
            {
                "clip_id": clip_id,
                "dataset_source": episode.get("dataset_source"),
                "source_manifest_sha256": episode.get(
                    "source_manifest_sha256"
                ),
                "source_record_sha256": episode.get("source_record_sha256"),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "fixed_split_assignment": episode["fixed_split_assignment"],
                "trajectory_sha256": episode.get("trajectory_sha256"),
                "style_features": features.tolist(),
                "style_controls": controls.tolist(),
            }
            for clip_id, prompt, episode, features, controls in zip(
                clip_ids,
                prompts,
                episodes,
                style_features_array,
                style_controls_array,
                strict=True,
            )
        ],
    }
    _atomic_json_save(metadata, output_path.with_suffix(output_path.suffix + ".json"))
    return metadata


def _validate_motion_only_cache_metadata(
    cache_metadata: Mapping,
    *,
    metadata_path: Path,
    actual_cache_sha256: str,
    clip_ids: Sequence[str],
    prompts: Sequence[str],
    conditions: np.ndarray,
    style_features,
    style_controls,
    structured_payload_present: bool,
    semantic_payload_present: bool,
    affect_payload_present: bool,
) -> dict:
    assert_no_forbidden_data_lineage(
        cache_metadata, context="motion_only_condition_cache"
    )
    exact = {
        "schema_version": MOTION_ONLY_CONDITION_CACHE_SCHEMA_VERSION,
        "artifact_kind": MOTION_ONLY_CONDITION_CACHE_ARTIFACT_KIND,
        "formal_episode_contract": MOTION_ONLY_EPISODE_CONTRACT,
        "condition_dim": KIMODO_V2_CONDITION_DIM,
        "count": len(clip_ids),
        "qwen_policy": MOTION_ONLY_NO_QWEN_POLICY,
        "kimodo_policy": MOTION_ONLY_NO_KIMODO_POLICY,
        "condition_policy": MOTION_ONLY_STYLE_ONLY_CONDITION_POLICY,
        "condition_nonzero_indices": list(
            range(STYLE_CONTROL_SLICE.start, STYLE_CONTROL_SLICE.stop)
        ),
        "condition_exact_zero_ranges": [
            [0, STYLE_CONTROL_SLICE.start],
            [STYLE_CONTROL_SLICE.stop, KIMODO_V2_CONDITION_DIM],
        ],
    }
    if any(cache_metadata.get(field) != expected for field, expected in exact.items()):
        raise ValueError("motion-only condition cache isolation contract changed")
    forbidden_qwen_fields = {
        "qwen_checkpoint",
        "qwen_checkpoint_sha256",
        "qwen_model_name",
        "qwen_revision",
    }
    if forbidden_qwen_fields.intersection(cache_metadata):
        raise ValueError("motion-only condition cache must not bind Qwen")
    if structured_payload_present or semantic_payload_present or affect_payload_present:
        raise ValueError(
            "motion-only condition cache must not carry semantic condition payloads"
        )
    if cache_metadata.get("cache_sha256") != actual_cache_sha256:
        raise ValueError("motion-only condition cache hash does not match metadata")
    validate_motion_only_style_condition(
        conditions, context="motion-only condition cache"
    )
    if (
        not isinstance(style_features, np.ndarray)
        or not isinstance(style_controls, np.ndarray)
        or style_features.shape != (len(clip_ids), 3)
        or style_controls.shape != (len(clip_ids), 3)
        or not np.isfinite(style_features).all()
        or not np.isfinite(style_controls).all()
    ):
        raise ValueError("motion-only cache has invalid trajectory style arrays")

    generator_value = cache_metadata.get("generator_checkpoint")
    if not isinstance(generator_value, str) or not generator_value.strip():
        raise ValueError("motion-only cache lacks its target random checkpoint")
    generator_path = Path(generator_value)
    if not generator_path.is_absolute():
        generator_path = metadata_path.parent / generator_path
    if not generator_path.is_file():
        raise FileNotFoundError(
            f"motion-only cache generator checkpoint is missing: {generator_path}"
        )
    generator_sha256 = sha256_file(generator_path)
    if cache_metadata.get("generator_checkpoint_sha256") != generator_sha256:
        raise ValueError("motion-only cache generator checkpoint hash changed")
    generator_checkpoint, style_contract = _load_target_generator_contract(
        generator_path
    )
    validate_motion_only_checkpoint_isolation(generator_checkpoint)
    split_contract = (generator_checkpoint.get("v2_contracts") or {}).get("split")
    generator_registry = (
        (generator_checkpoint.get("sources") or {}).get(
            "data_source_registry"
        )
    )
    cache_registry = cache_metadata.get("data_source_registry")
    if generator_registry is not None:
        generator_registry = validate_data_source_registry_contract(
            generator_registry,
            expected_role=GENERATOR_FOUNDATION_ROLE,
        )
        cache_registry = validate_data_source_registry_contract(
            cache_registry,
            expected_role=GENERATOR_FOUNDATION_ROLE,
        )
        if (
            cache_registry != generator_registry
            or cache_metadata.get(DATA_SOURCE_REGISTRY_HASH_FIELD)
            != generator_registry["sha256"]
        ):
            raise ValueError(
                "motion-only cache data-source registry binding changed"
            )
    elif cache_registry is not None:
        raise ValueError(
            "legacy generator cannot acquire an unbound cache source registry"
        )
    if (
        cache_metadata.get("style_contract_sha256") != style_contract["sha256"]
        or cache_metadata.get("split_contract_sha256")
        != (split_contract or {}).get("sha256")
    ):
        raise ValueError("motion-only cache style/split binding changed")

    episode_metadata = cache_metadata.get("episodes")
    if not isinstance(episode_metadata, list) or len(episode_metadata) != len(
        clip_ids
    ):
        raise ValueError("motion-only cache episode metadata count is invalid")
    split_assignment = {
        str(record.get("clip_id")): record.get("split")
        for record in (split_contract or {}).get("episodes") or ()
    }
    split_dataset_source = {
        str(record.get("clip_id")): record.get("dataset_source")
        for record in (split_contract or {}).get("episodes") or ()
    }
    for index, (clip_id, prompt) in enumerate(
        zip(clip_ids, prompts, strict=True)
    ):
        record = episode_metadata[index]
        if (
            not isinstance(record, Mapping)
            or record.get("clip_id") != clip_id
            or record.get("prompt_sha256")
            != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            or record.get("fixed_split_assignment") != split_assignment.get(clip_id)
            or not _is_sha256(record.get("trajectory_sha256"))
        ):
            raise ValueError("motion-only cache episode lineage changed")
        if generator_registry is not None and (
            record.get("dataset_source") != split_dataset_source.get(clip_id)
            or not _is_sha256(record.get("source_manifest_sha256"))
            or not _is_sha256(record.get("source_record_sha256"))
        ):
            raise ValueError(
                "motion-only cache episode data-source lineage changed"
            )
        recorded_features = np.asarray(record.get("style_features"), dtype=np.float32)
        recorded_controls = np.asarray(record.get("style_controls"), dtype=np.float32)
        if (
            not np.array_equal(recorded_features, style_features[index])
            or not np.array_equal(recorded_controls, style_controls[index])
            or not np.array_equal(
                normalize_style_features(recorded_features, style_contract),
                recorded_controls,
            )
            or not np.array_equal(
                conditions[index, STYLE_CONTROL_SLICE], recorded_controls
            )
        ):
            raise ValueError("motion-only cache trajectory style binding changed")
    return {
        **dict(cache_metadata),
        "generator_checkpoint": str(generator_path.resolve()),
        "unsafe_condition_cache": False,
    }


def load_condition_cache(
    cache_path: str | Path, *, allow_unsafe_metadata=False
) -> tuple[list[str], list[str], np.ndarray, dict]:
    cache_path = Path(cache_path)
    assert_no_forbidden_data_lineage(
        {"cache_path": str(cache_path)}, context="condition_cache"
    )
    with np.load(cache_path, allow_pickle=False) as payload:
        clip_ids = payload["clip_ids"].astype(str).tolist()
        prompts = payload["prompts"].astype(str).tolist()
        conditions = payload["conditions"].astype(np.float32)
        structured_names = {
            "behavior_ids",
            "behavior_review_statuses",
            "behavior_supervision_mask",
            "emotion_ids",
            "emotion_supervision_mask",
        }
        style_names = {"style_features", "style_controls"}
        semantic_mask_names = {
            "canonical_prompt_roles",
            "official_category_supervision_mask",
            "robot_observable_motion_form_supervision_mask",
            "communicative_intent_supervision_mask",
            "prompt_text_supervision_mask",
            "legacy_gesture_supervision_mask",
        }
        affect_names = {
            "affect_observable_review_statuses",
            "affect_observable_supervision_mask",
            "emotion_conditioning_mask",
        }
        present_structured_names = structured_names.intersection(payload.files)
        if present_structured_names and present_structured_names != structured_names:
            raise ValueError("condition cache has an incomplete structured semantic payload")
        if present_structured_names:
            behavior_ids = payload["behavior_ids"].astype(str).tolist()
            behavior_review_statuses = payload["behavior_review_statuses"].astype(str).tolist()
            raw_behavior_supervision_mask = payload["behavior_supervision_mask"]
            if raw_behavior_supervision_mask.dtype != np.dtype(np.bool_):
                raise ValueError("condition cache behavior supervision mask must be boolean")
            behavior_supervision_mask = raw_behavior_supervision_mask.tolist()
            emotion_ids = payload["emotion_ids"].astype(str).tolist()
            raw_supervision_mask = payload["emotion_supervision_mask"]
            if raw_supervision_mask.dtype != np.dtype(np.bool_):
                raise ValueError("condition cache emotion supervision mask must be boolean")
            emotion_supervision_mask = raw_supervision_mask.tolist()
        else:
            behavior_ids = behavior_review_statuses = behavior_supervision_mask = None
            emotion_ids = emotion_supervision_mask = None
        present_style_names = style_names.intersection(payload.files)
        if present_style_names and present_style_names != style_names:
            raise ValueError("condition cache has an incomplete trajectory style payload")
        if present_style_names:
            style_features = payload["style_features"].astype(np.float32)
            style_controls = payload["style_controls"].astype(np.float32)
        else:
            style_features = style_controls = None
        present_semantic_mask_names = semantic_mask_names.intersection(payload.files)
        if present_semantic_mask_names and present_semantic_mask_names != semantic_mask_names:
            raise ValueError("condition cache has an incomplete semantic supervision payload")
        if present_semantic_mask_names:
            canonical_prompt_roles = payload["canonical_prompt_roles"].astype(str).tolist()
            semantic_mask_arrays = {}
            for name in sorted(semantic_mask_names - {"canonical_prompt_roles"}):
                values = payload[name]
                if values.dtype != np.dtype(np.bool_):
                    raise ValueError(f"condition cache {name} must be boolean")
                semantic_mask_arrays[name] = values.tolist()
        else:
            canonical_prompt_roles = None
            semantic_mask_arrays = None
        present_affect_names = affect_names.intersection(payload.files)
        if present_affect_names and present_affect_names != affect_names:
            raise ValueError("condition cache has an incomplete affect supervision payload")
        if present_affect_names:
            affect_observable_review_statuses = payload[
                "affect_observable_review_statuses"
            ].astype(str).tolist()
            raw_affect_mask = payload["affect_observable_supervision_mask"]
            raw_emotion_conditioning_mask = payload["emotion_conditioning_mask"]
            if raw_affect_mask.dtype != np.dtype(np.bool_):
                raise ValueError(
                    "condition cache affect observability supervision mask must be boolean"
                )
            if raw_emotion_conditioning_mask.dtype != np.dtype(np.bool_):
                raise ValueError(
                    "condition cache emotion conditioning mask must be boolean"
                )
            affect_observable_supervision_mask = raw_affect_mask.tolist()
            emotion_conditioning_mask = raw_emotion_conditioning_mask.tolist()
        else:
            affect_observable_review_statuses = None
            affect_observable_supervision_mask = None
            emotion_conditioning_mask = None
    if len(prompts) != len(clip_ids):
        raise ValueError("condition cache clip_ids/prompts length mismatch")
    if conditions.shape != (len(clip_ids), KIMODO_V2_CONDITION_DIM):
        raise ValueError(f"invalid condition cache shape: {conditions.shape}")
    if not np.isfinite(conditions).all():
        raise ValueError("condition cache contains non-finite values")
    if len(set(clip_ids)) != len(clip_ids):
        raise ValueError("condition cache contains duplicate clip ids")
    actual_cache_sha256 = sha256_file(cache_path)
    metadata_path = cache_path.with_suffix(cache_path.suffix + ".json")
    if metadata_path.is_file():
        cache_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert_no_forbidden_data_lineage(
            cache_metadata, context="condition_cache.metadata"
        )
        if (
            cache_metadata.get("artifact_kind")
            == MOTION_ONLY_CONDITION_CACHE_ARTIFACT_KIND
        ):
            cache_metadata = _validate_motion_only_cache_metadata(
                cache_metadata,
                metadata_path=metadata_path,
                actual_cache_sha256=actual_cache_sha256,
                clip_ids=clip_ids,
                prompts=prompts,
                conditions=conditions,
                style_features=style_features,
                style_controls=style_controls,
                structured_payload_present=behavior_ids is not None,
                semantic_payload_present=canonical_prompt_roles is not None,
                affect_payload_present=(
                    affect_observable_review_statuses is not None
                ),
            )
            cache_metadata = {
                **cache_metadata,
                "path": str(cache_path.resolve()),
                "metadata_path": str(metadata_path.resolve()),
            }
            return clip_ids, prompts, conditions, cache_metadata
        metadata_schema = cache_metadata.get("schema_version")
        if metadata_schema not in (1, 2, CONDITION_CACHE_SCHEMA_VERSION):
            raise ValueError("condition cache metadata has an unsupported schema version")
        if cache_metadata.get("artifact_kind") != "ula_v2_qwen_motion_condition_cache":
            raise ValueError("condition cache metadata has the wrong artifact kind")
        if cache_metadata.get("cache_sha256") != actual_cache_sha256:
            raise ValueError("condition cache hash does not match its metadata")
        if int(cache_metadata.get("condition_dim", -1)) != KIMODO_V2_CONDITION_DIM:
            raise ValueError("condition cache metadata has the wrong condition dimension")
        if int(cache_metadata.get("base_condition_dim", -1)) != KIMODO_CONDITION_DIM:
            raise ValueError("condition cache metadata has the wrong base condition dimension")
        if int(cache_metadata.get("motion_latent_dim", -1)) != (
            KIMODO_V2_CONDITION_DIM - KIMODO_CONDITION_DIM
        ):
            raise ValueError("condition cache metadata has the wrong motion latent dimension")
        if int(cache_metadata.get("count", -1)) != len(clip_ids):
            raise ValueError("condition cache metadata count does not match its payload")
        if metadata_schema in (2, CONDITION_CACHE_SCHEMA_VERSION):
            if behavior_ids is None:
                raise ValueError("structured condition cache metadata requires semantic arrays")
            if metadata_schema == CONDITION_CACHE_SCHEMA_VERSION and (
                canonical_prompt_roles is None
                or semantic_mask_arrays is None
                or affect_observable_review_statuses is None
            ):
                raise ValueError(
                    "schema-3 condition cache requires semantic supervision arrays"
                )
            if metadata_schema == CONDITION_CACHE_SCHEMA_VERSION and not all(
                len(values) == len(clip_ids)
                for values in [
                    canonical_prompt_roles,
                    *semantic_mask_arrays.values(),
                    affect_observable_review_statuses,
                    affect_observable_supervision_mask,
                    emotion_conditioning_mask,
                ]
            ):
                raise ValueError(
                    "condition cache semantic supervision array length mismatch"
                )
            if not all(
                len(values) == len(clip_ids)
                for values in (
                    behavior_ids,
                    behavior_review_statuses,
                    behavior_supervision_mask,
                    emotion_ids,
                    emotion_supervision_mask,
                )
            ):
                raise ValueError("condition cache semantic array length mismatch")
            if (
                style_features.shape != (len(clip_ids), 3)
                or style_controls.shape != (len(clip_ids), 3)
                or not np.isfinite(style_features).all()
                or not np.isfinite(style_controls).all()
            ):
                raise ValueError("condition cache has invalid trajectory-derived style arrays")
            generator_path_value = cache_metadata.get("generator_checkpoint")
            if not isinstance(generator_path_value, str) or not generator_path_value.strip():
                raise ValueError("condition cache metadata is missing its generator checkpoint")
            generator_path = Path(generator_path_value)
            if not generator_path.is_absolute():
                generator_path = metadata_path.parent / generator_path
            if not generator_path.is_file():
                raise FileNotFoundError(
                    f"condition cache generator checkpoint is missing: {generator_path}"
                )
            generator_sha256 = sha256_file(generator_path)
            if cache_metadata.get("generator_checkpoint_sha256") != generator_sha256:
                raise ValueError("condition cache generator checkpoint hash does not match")
            _, style_contract = _load_target_generator_contract(generator_path)
            if cache_metadata.get("style_contract_sha256") != style_contract["sha256"]:
                raise ValueError("condition cache style contract hash does not match")
            semantic_contract = cache_metadata.get("semantic_condition_contract") or {}
            expected_contract = {
                "version": (
                    3 if metadata_schema == CONDITION_CACHE_SCHEMA_VERSION else 1
                ),
                "behavior_ids": list(KIMODO_BEHAVIOR_IDS),
                "emotion_ids": list(KIMODO_EMOTION_IDS),
                "emotion_one_hot_slice": [
                    KIMODO_EMOTION_SLICE.start,
                    KIMODO_EMOTION_SLICE.stop,
                ],
                "behavior_one_hot_slice": [
                    KIMODO_BEHAVIOR_SLICE.start,
                    KIMODO_BEHAVIOR_SLICE.stop,
                ],
                "behavior_family_slice": [
                    KIMODO_BEHAVIOR_FAMILY_SLICE.start,
                    KIMODO_BEHAVIOR_FAMILY_SLICE.stop,
                ],
                "unsupervised_behavior_encoding": (
                    "zero_kimodo_behavior_and_family_one_hot"
                ),
                "unresolved_emotion_encoding": "zero_kimodo_emotion_one_hot",
                "legacy_semantics_source": "explicit_fields_no_prompt_inference",
                "missing_legacy_defaults": dict(LEGACY_SEMANTIC_DEFAULTS),
                "motion_style_mapping": dict(MOTION_STYLE_TO_LEGACY),
                "kimodo_emotion_to_legacy_affect": dict(
                    KIMODO_EMOTION_TO_LEGACY_AFFECT
                ),
                "style_controls_source": "18d_actions_15d_prefix",
                "style_control_slice": [
                    STYLE_CONTROL_SLICE.start,
                    STYLE_CONTROL_SLICE.stop,
                ],
            }
            if metadata_schema == CONDITION_CACHE_SCHEMA_VERSION:
                expected_contract.update(
                    {
                        "emotion_conditioning_requires": [
                            "verified_source_emotion_label",
                            "verified_robot_affect_observable_in_18d",
                        ],
                        "masked_affect_encoding": (
                            "zero_kimodo_emotion_one_hot_and_legacy_affect_slice"
                        ),
                        "formal_semantic_supervision_masks": dict(
                            FORMAL_SEMANTIC_SUPERVISION_MASKS
                        ),
                        "official_category_conditioning_enabled": False,
                        "official_category_role": (
                            OFFICIAL_CATEGORY_CONDITIONING_ROLE
                        ),
                        "official_category_condition_channel": None,
                        "official_category_loss": None,
                        "official_category_conditioned_count": 0,
                        "formal_optimization_targets": [
                            "motion_flow_18d",
                            "head_3dof",
                            "trajectory_style",
                            "verified_observable_emotion_only",
                            "native_duration",
                        ],
                        "forbidden_category_routes": [
                            "qwen_motion_latent",
                            "legacy_gesture",
                            "legacy_intent",
                            "kimodo_behavior",
                            "kimodo_emotion",
                        ],
                        "masked_prompt_text_encoding": (
                            "zero_qwen_motion_latent_128d"
                        ),
                        "masked_communicative_intent_encoding": (
                            "zero_legacy_intent_slice"
                        ),
                        "masked_legacy_gesture_encoding": (
                            "zero_legacy_gesture_slice"
                        ),
                    }
                )
            if semantic_contract != expected_contract:
                raise ValueError("condition cache semantic contract does not match the network")
            episode_metadata = cache_metadata.get("episodes")
            if not isinstance(episode_metadata, list) or len(episode_metadata) != len(clip_ids):
                raise ValueError("condition cache episode metadata count is invalid")
            for index, (clip_id, prompt) in enumerate(zip(clip_ids, prompts)):
                episode_record = episode_metadata[index]
                if not isinstance(episode_record, Mapping):
                    raise ValueError("condition cache episode metadata must contain mappings")
                if episode_record.get("clip_id") != clip_id or episode_record.get(
                    "prompt_sha256"
                ) != hashlib.sha256(prompt.encode("utf-8")).hexdigest():
                    raise ValueError("condition cache episode metadata does not match its payload")
                labels = _resolve_structured_semantics(episode_record)
                legacy = _resolve_legacy_semantics(episode_record)
                policy = (
                    semantic_supervision_policy(episode_record)
                    if metadata_schema == CONDITION_CACHE_SCHEMA_VERSION
                    else None
                )
                if policy is not None:
                    masks = policy["semantic_supervision_masks"]
                    if canonical_prompt_roles[index] != policy["canonical_prompt_role"]:
                        raise ValueError(
                            "condition cache canonical prompt role does not match metadata"
                        )
                    mask_array_names = {
                        "official_category": "official_category_supervision_mask",
                        "robot_observable_motion_form": (
                            "robot_observable_motion_form_supervision_mask"
                        ),
                        "communicative_intent": (
                            "communicative_intent_supervision_mask"
                        ),
                        "prompt_text": "prompt_text_supervision_mask",
                        "legacy_gesture": "legacy_gesture_supervision_mask",
                    }
                    for mask_name, array_name in mask_array_names.items():
                        if semantic_mask_arrays[array_name][index] is not masks[mask_name]:
                            raise ValueError(
                                "condition cache semantic supervision mask does not "
                                "match episode metadata"
                            )
                    if (
                        affect_observable_review_statuses[index]
                        != labels["affect_observable_review_status"]
                        or affect_observable_supervision_mask[index]
                        is not labels["affect_observable_supervision_mask"]
                        or emotion_conditioning_mask[index]
                        is not labels["emotion_conditioning_mask"]
                    ):
                        raise ValueError(
                            "condition cache affect supervision payload does not match metadata"
                        )
                payload_emotion = emotion_ids[index] or None
                if (
                    labels["behavior_id"] != behavior_ids[index]
                    or labels["behavior_review_status"]
                    != behavior_review_statuses[index]
                    or labels["behavior_supervision_mask"]
                    != behavior_supervision_mask[index]
                    or labels["emotion_id"] != payload_emotion
                    or labels["emotion_supervision_mask"]
                    != emotion_supervision_mask[index]
                ):
                    raise ValueError("condition cache semantic metadata does not match its payload")
                expected_behavior = np.zeros(len(KIMODO_BEHAVIOR_IDS), dtype=np.float32)
                if labels["behavior_supervision_mask"]:
                    expected_behavior[
                        KIMODO_BEHAVIOR_IDS.index(labels["behavior_id"])
                    ] = 1.0
                if not np.array_equal(
                    conditions[index, KIMODO_BEHAVIOR_SLICE], expected_behavior
                ):
                    raise ValueError("condition cache behavior vector does not match its label")
                expected_family = np.zeros(
                    len(KIMODO_BEHAVIOR_FAMILIES), dtype=np.float32
                )
                if labels["behavior_supervision_mask"]:
                    family = kimodo_behavior_family(labels["behavior_id"])
                    expected_family[KIMODO_BEHAVIOR_FAMILIES.index(family)] = 1.0
                if not np.array_equal(
                    conditions[index, KIMODO_BEHAVIOR_FAMILY_SLICE], expected_family
                ):
                    raise ValueError(
                        "condition cache behavior family vector does not match its label"
                    )
                expected_emotion = np.zeros(len(KIMODO_EMOTION_IDS), dtype=np.float32)
                if labels["emotion_conditioning_mask"]:
                    expected_emotion[KIMODO_EMOTION_IDS.index(labels["emotion_id"])] = 1.0
                if not np.array_equal(
                    conditions[index, KIMODO_EMOTION_SLICE], expected_emotion
                ):
                    raise ValueError("condition cache emotion vector does not match its label")
                legacy_blocks = (
                    (LEGACY_INTENT_SLICE, INTENT_IDS, legacy["intent"]),
                    (LEGACY_AFFECT_SLICE, AFFECT_IDS, legacy["observed_affect"]),
                    (LEGACY_STYLE_SLICE, STYLE_IDS, legacy["motion_style"]),
                    (LEGACY_GESTURE_SLICE, GESTURE_IDS, legacy["semantic_gesture"]),
                )
                for vector_slice, vocabulary, label in legacy_blocks:
                    expected = np.zeros(len(vocabulary), dtype=np.float32)
                    masked = bool(
                        policy is not None
                        and (
                            (
                                vector_slice == LEGACY_INTENT_SLICE
                                and not policy["semantic_supervision_masks"][
                                    "communicative_intent"
                                ]
                            )
                            or (
                                vector_slice == LEGACY_AFFECT_SLICE
                                and not labels["emotion_conditioning_mask"]
                            )
                            or (
                                vector_slice == LEGACY_GESTURE_SLICE
                                and not policy["semantic_supervision_masks"][
                                    "legacy_gesture"
                                ]
                            )
                        )
                    )
                    if not masked:
                        expected[vocabulary[label]] = 1.0
                    if not np.array_equal(conditions[index, vector_slice], expected):
                        raise ValueError(
                            "condition cache legacy semantic vector does not match metadata"
                        )
                if (
                    policy is not None
                    and not policy["semantic_supervision_masks"]["prompt_text"]
                    and np.any(conditions[index, KIMODO_CONDITION_DIM:] != 0.0)
                ):
                    raise ValueError(
                        "condition cache masked prompt has a non-zero Qwen motion latent"
                    )
                recorded_features = np.asarray(
                    episode_record.get("style_features"), dtype=np.float32
                )
                recorded_controls = np.asarray(
                    episode_record.get("style_controls"), dtype=np.float32
                )
                if not np.array_equal(recorded_features, style_features[index]) or not np.array_equal(
                    recorded_controls, style_controls[index]
                ):
                    raise ValueError("condition cache style metadata does not match its payload")
                normalized_controls = normalize_style_features(
                    style_features[index], style_contract
                )
                if not np.array_equal(normalized_controls, style_controls[index]):
                    raise ValueError("condition cache style controls use the wrong normalization")
                if not np.array_equal(
                    conditions[index, STYLE_CONTROL_SLICE], style_controls[index]
                ):
                    raise ValueError("condition cache base condition has stale style controls")
            supervised_count = sum(bool(value) for value in emotion_supervision_mask)
            if int(cache_metadata.get("emotion_supervised_count", -1)) != supervised_count:
                raise ValueError("condition cache supervised emotion count is invalid")
            if int(cache_metadata.get("emotion_unresolved_count", -1)) != (
                len(clip_ids) - supervised_count
            ):
                raise ValueError("condition cache unresolved emotion count is invalid")
            affect_verified_count = sum(
                bool(value) for value in affect_observable_supervision_mask
            )
            if int(cache_metadata.get("affect_observable_verified_count", -1)) != (
                affect_verified_count
            ):
                raise ValueError(
                    "condition cache affect-observable verified count is invalid"
                )
            conditioned_count = sum(bool(value) for value in emotion_conditioning_mask)
            if int(cache_metadata.get("emotion_conditioned_count", -1)) != conditioned_count:
                raise ValueError("condition cache emotion-conditioned count is invalid")
            if int(cache_metadata.get("official_category_conditioned_count", -1)) != 0:
                raise ValueError("official category must not be conditioned in schema-3")
            behavior_supervised_count = sum(
                bool(value) for value in behavior_supervision_mask
            )
            if int(cache_metadata.get("behavior_supervised_count", -1)) != (
                behavior_supervised_count
            ):
                raise ValueError("condition cache supervised behavior count is invalid")
            if int(cache_metadata.get("behavior_unsupervised_count", -1)) != (
                len(clip_ids) - behavior_supervised_count
            ):
                raise ValueError("condition cache unsupervised behavior count is invalid")
        else:
            if behavior_ids is not None:
                raise ValueError("legacy condition cache metadata cannot bind semantic arrays")
            expected_episodes = [
                {
                    "clip_id": clip_id,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                }
                for clip_id, prompt in zip(clip_ids, prompts)
            ]
            if cache_metadata.get("episodes") != expected_episodes:
                raise ValueError("condition cache episode metadata does not match its payload")

        qwen_path_value = cache_metadata.get("qwen_checkpoint")
        if not isinstance(qwen_path_value, str) or not qwen_path_value.strip():
            raise ValueError("condition cache metadata is missing its Qwen checkpoint path")
        qwen_path = Path(qwen_path_value)
        if not qwen_path.is_absolute():
            qwen_path = metadata_path.parent / qwen_path
        if not qwen_path.is_file():
            raise FileNotFoundError(f"condition cache Qwen checkpoint is missing: {qwen_path}")
        qwen_sha256 = sha256_file(qwen_path)
        if cache_metadata.get("qwen_checkpoint_sha256") != qwen_sha256:
            raise ValueError("condition cache Qwen checkpoint hash does not match its metadata")
        qwen_payload = torch.load(qwen_path, map_location="cpu", weights_only=True)
        recorded_qwen = qwen_payload.get("qwen") or {}
        if cache_metadata.get("qwen_model_name") != recorded_qwen.get("model_name"):
            raise ValueError("condition cache Qwen model name does not match its checkpoint")
        if cache_metadata.get("qwen_revision") != recorded_qwen.get("revision"):
            raise ValueError("condition cache Qwen revision does not match its checkpoint")
        cache_metadata = {
            **cache_metadata,
            "qwen_checkpoint": str(qwen_path.resolve()),
            **(
                {"generator_checkpoint": str(generator_path.resolve())}
                if metadata_schema in (2, CONDITION_CACHE_SCHEMA_VERSION)
                else {}
            ),
            "unsafe_condition_cache": False,
        }
    else:
        if not allow_unsafe_metadata:
            raise ValueError(
                "condition cache metadata is required; pass the explicit unsafe flag only "
                "for a test or smoke run"
            )
        cache_metadata = {
            "artifact_kind": "unversioned_test_condition_cache",
            "cache_sha256": actual_cache_sha256,
            "condition_dim": KIMODO_V2_CONDITION_DIM,
            "count": len(clip_ids),
            "unsafe_condition_cache": True,
        }
    cache_metadata = {
        **cache_metadata,
        "path": str(cache_path.resolve()),
        "metadata_path": str(metadata_path.resolve()) if metadata_path.is_file() else None,
    }
    return clip_ids, prompts, conditions, cache_metadata


def attach_condition_cache(
    episodes: Sequence[dict],
    cache_path: str | Path,
    *,
    allow_unsafe_metadata=False,
) -> list[dict]:
    clip_ids, prompts, conditions, cache_metadata = load_condition_cache(
        cache_path, allow_unsafe_metadata=allow_unsafe_metadata
    )
    by_clip = {
        clip_id: (prompt, condition)
        for clip_id, prompt, condition in zip(clip_ids, prompts, conditions)
    }
    semantic_by_clip = {
        record["clip_id"]: record
        for record in cache_metadata.get("episodes", [])
        if isinstance(record, Mapping) and "behavior_id" in record
    }
    motion_only_cache = (
        cache_metadata.get("artifact_kind")
        == MOTION_ONLY_CONDITION_CACHE_ARTIFACT_KIND
    )
    motion_only_by_clip = (
        {
            str(record.get("clip_id")): record
            for record in cache_metadata.get("episodes", [])
            if isinstance(record, Mapping)
        }
        if motion_only_cache
        else {}
    )
    style_contract = None
    if semantic_by_clip or motion_only_cache:
        _, style_contract = _load_target_generator_contract(
            cache_metadata["generator_checkpoint"]
        )
    attached = []
    for episode in episodes:
        clip_id = episode["clip_id"]
        if clip_id not in by_clip:
            raise ValueError(f"condition cache is missing {clip_id}")
        cached_prompt, condition = by_clip[clip_id]
        if cached_prompt != _resolve_prompt(episode):
            raise ValueError(f"condition cache prompt changed for {clip_id}")
        item = dict(episode)
        if motion_only_cache:
            if (
                episode.get("formal_episode_contract")
                != MOTION_ONLY_EPISODE_CONTRACT
                or clip_id not in motion_only_by_clip
            ):
                raise ValueError(
                    f"motion-only cache cannot attach to non-motion episode {clip_id}"
                )
            validate_motion_only_style_condition(
                condition, context=f"{clip_id}: motion-only cached condition"
            )
            cached_record = motion_only_by_clip[clip_id]
            if (
                episode.get("fixed_split_assignment")
                != cached_record.get("fixed_split_assignment")
                or episode.get("trajectory_sha256")
                != cached_record.get("trajectory_sha256")
            ):
                raise ValueError(
                    f"motion-only cache fixed split/trajectory changed for {clip_id}"
                )
            actions = np.asarray(episode.get("actions"), dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
                raise ValueError(
                    f"motion-only cache requires unchanged 18D actions for {clip_id}"
                )
            features = extract_style_features(
                actions[:, :LEGACY_ACTION_DIM],
                fps=float(episode.get("fps") or 30.0),
            )
            controls = normalize_style_features(features, style_contract)
            if (
                not np.array_equal(
                    features,
                    np.asarray(cached_record["style_features"], dtype=np.float32),
                )
                or not np.array_equal(
                    controls,
                    np.asarray(cached_record["style_controls"], dtype=np.float32),
                )
                or not np.array_equal(
                    condition[STYLE_CONTROL_SLICE], controls
                )
            ):
                raise ValueError(
                    f"motion-only cache trajectory style changed for {clip_id}"
                )
            item["style_features"] = features
            item["style_controls"] = controls
        if semantic_by_clip:
            if clip_id not in semantic_by_clip:
                raise ValueError(f"condition cache is missing semantic metadata for {clip_id}")
            episode_semantics = _resolve_structured_semantics(episode)
            cached_record = semantic_by_clip[clip_id]
            cached_semantics = _resolve_structured_semantics(cached_record)
            if episode_semantics != cached_semantics:
                raise ValueError(f"condition cache semantic labels changed for {clip_id}")
            episode_legacy = _resolve_legacy_semantics(episode)
            cached_legacy = _resolve_legacy_semantics(cached_record)
            if episode_legacy != cached_legacy:
                raise ValueError(f"condition cache legacy semantic labels changed for {clip_id}")
            if cache_metadata.get("schema_version") == CONDITION_CACHE_SCHEMA_VERSION:
                episode_policy = semantic_supervision_policy(episode)
                cached_policy = semantic_supervision_policy(cached_record)
                if episode_policy != cached_policy:
                    raise ValueError(
                        f"condition cache semantic supervision policy changed for {clip_id}"
                    )
                masks = episode_policy["semantic_supervision_masks"]
                if not masks["prompt_text"] and np.any(
                    condition[KIMODO_CONDITION_DIM:] != 0.0
                ):
                    raise ValueError(
                        f"condition cache masked prompt latent is non-zero for {clip_id}"
                    )
                if not masks["communicative_intent"] and np.any(
                    condition[LEGACY_INTENT_SLICE] != 0.0
                ):
                    raise ValueError(
                        f"condition cache masked intent channel is non-zero for {clip_id}"
                    )
                if not masks["legacy_gesture"] and np.any(
                    condition[LEGACY_GESTURE_SLICE] != 0.0
                ):
                    raise ValueError(
                        f"condition cache masked gesture channel is non-zero for {clip_id}"
                    )
                if not episode_semantics["emotion_conditioning_mask"] and (
                    np.any(condition[KIMODO_EMOTION_SLICE] != 0.0)
                    or np.any(condition[LEGACY_AFFECT_SLICE] != 0.0)
                ):
                    raise ValueError(
                        f"condition cache unverified robot affect channels are non-zero for {clip_id}"
                    )
                item.update(cached_policy)
            actions = np.asarray(episode.get("actions"), dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
                raise ValueError(f"condition cache requires unchanged 18D actions for {clip_id}")
            features = extract_style_features(
                actions[:, :LEGACY_ACTION_DIM],
                fps=float(episode.get("fps") or 30.0),
            )
            controls = normalize_style_features(features, style_contract)
            if not np.array_equal(
                features, np.asarray(cached_record["style_features"], dtype=np.float32)
            ) or not np.array_equal(
                controls, np.asarray(cached_record["style_controls"], dtype=np.float32)
            ):
                raise ValueError(f"condition cache trajectory-derived style changed for {clip_id}")
            item.update(cached_semantics)
            item.update(cached_legacy)
            item["style_features"] = features
            item["style_controls"] = controls
        item["condition"] = condition.copy()
        item["condition_cache_provenance"] = dict(cache_metadata)
        attached.append(item)
    return attached


def training_data_provenance(episodes: Sequence[Mapping]) -> dict:
    assert_no_forbidden_data_lineage(
        episodes, context="training_data_provenance.episodes"
    )
    cache_records = [item.get("condition_cache_provenance") or {} for item in episodes]
    canonical_cache = cache_records[0] if cache_records else {}
    if any(record != canonical_cache for record in cache_records):
        raise ValueError("episodes do not share one immutable condition cache")
    assert_no_forbidden_data_lineage(
        canonical_cache, context="training_data_provenance.condition_cache"
    )
    cache_path = canonical_cache.get("path")
    if cache_path and sha256_file(cache_path) != canonical_cache.get("cache_sha256"):
        raise ValueError("condition cache changed after episode loading")
    records = []
    for item in sorted(episodes, key=lambda row: row["clip_id"]):
        trajectory_path = Path(item["trajectory_path"]) if item.get("trajectory_path") else None
        trajectory_hash = item.get("trajectory_sha256")
        if trajectory_path is not None:
            actual_hash = sha256_file(trajectory_path)
            if trajectory_hash is not None and trajectory_hash != actual_hash:
                raise ValueError(f"trajectory changed after loading: {item['clip_id']}")
            trajectory_hash = actual_hash
        records.append(
            {
                "clip_id": item["clip_id"],
                "dataset_source": item.get("dataset_source"),
                "source_clip_id": item.get("source_clip_id"),
                "source_manifest_sha256": item.get(
                    "source_manifest_sha256"
                ),
                "source_record_sha256": item.get("source_record_sha256"),
                "review_state": item.get("review_state", "unspecified"),
                "eligibility_mode": item.get("eligibility_mode", "unversioned_direct_input"),
                "trajectory_path": None if trajectory_path is None else str(trajectory_path.resolve()),
                "trajectory_sha256": trajectory_hash,
                "frames": int(item["actions"].shape[0]),
                "fps": float(item.get("fps") or 30.0),
                "duration_sec": float(item.get("duration_sec") or 0.0),
                "prompt": item["prompt"],
                "prompt_sha256": item.get("prompt_sha256")
                or hashlib.sha256(str(item["prompt"]).encode("utf-8")).hexdigest(),
                "behavior_id": item.get("behavior_id"),
                "behavior_review_status": item.get("behavior_review_status"),
                "behavior_supervision_mask": item.get("behavior_supervision_mask"),
                "emotion_id": item.get("emotion_id"),
                "emotion_review_status": item.get("emotion_review_status"),
                "emotion_supervision_mask": item.get("emotion_supervision_mask"),
                "source_emotion_label_verified": item.get(
                    "source_emotion_label_verified"
                ),
                "affect_observable_review_status": item.get(
                    "affect_observable_review_status"
                ),
                "affect_observable_supervision_mask": item.get(
                    "affect_observable_supervision_mask"
                ),
                "emotion_conditioning_mask": item.get(
                    "emotion_conditioning_mask"
                ),
                "annotation_kind": item.get("annotation_kind"),
                "semantic_event": deepcopy(item.get("semantic_event")),
                "canonical_action": item.get("canonical_action"),
                "canonical_action_role": item.get("canonical_action_role"),
                "semantic_mapping_status": item.get("semantic_mapping_status"),
                "official_category_verified": item.get(
                    "official_category_verified"
                ),
                "official_category_conditioning_enabled": item.get(
                    "official_category_conditioning_enabled"
                ),
                "official_category_role": item.get("official_category_role"),
                "official_category_condition_channel": item.get(
                    "official_category_condition_channel"
                ),
                "official_category_loss": item.get("official_category_loss"),
                "semantic_supervision_masks": deepcopy(
                    item.get("semantic_supervision_masks")
                ),
                "intent": item.get("intent"),
                "observed_affect": item.get("observed_affect"),
                "motion_style": item.get("motion_style"),
                "source_motion_style": item.get("source_motion_style"),
                "semantic_gesture": item.get("semantic_gesture"),
                "style_features": (
                    None
                    if item.get("style_features") is None
                    else np.asarray(item["style_features"], dtype=np.float32).tolist()
                ),
                "style_controls": (
                    None
                    if item.get("style_controls") is None
                    else np.asarray(item["style_controls"], dtype=np.float32).tolist()
                ),
                "split_assignment": item.get("split_assignment"),
                "eval_eligible": bool(item.get("eval_eligible", False)),
            }
        )
    manifests = sorted(
        {
            (item.get("source_manifest"), item.get("source_manifest_sha256"))
            for item in episodes
            if item.get("source_manifest")
        }
    )
    for path, digest in manifests:
        if sha256_file(path) != digest:
            raise ValueError(f"source manifest changed after episode loading: {path}")
    unsafe_episodes = any(
        row["eligibility_mode"] != "adjudicated_train_ready" for row in records
    )
    unsafe_cache = bool(canonical_cache.get("unsafe_condition_cache", False))
    data_source_registry = canonical_cache.get("data_source_registry")
    if data_source_registry is not None:
        data_source_registry = validate_data_source_registry_contract(
            data_source_registry,
            expected_dataset_sources=[
                row["dataset_source"]
                for row in records
                if row["dataset_source"] is not None
            ],
        )
    return {
        "contract_version": CONTRACT_VERSION,
        "episode_count": len(records),
        "total_frames": sum(row["frames"] for row in records),
        "total_duration_sec": sum(row["duration_sec"] for row in records),
        "emotion_supervised_episode_count": sum(
            row["emotion_supervision_mask"] is True for row in records
        ),
        "emotion_unresolved_episode_count": sum(
            row["emotion_supervision_mask"] is False for row in records
        ),
        "source_emotion_label_verified_episode_count": sum(
            row["source_emotion_label_verified"] is True for row in records
        ),
        "affect_observable_verified_episode_count": sum(
            row["affect_observable_supervision_mask"] is True for row in records
        ),
        "emotion_conditioned_episode_count": sum(
            row["emotion_conditioning_mask"] is True for row in records
        ),
        "official_category_conditioned_episode_count": sum(
            row["official_category_conditioning_enabled"] is True for row in records
        ),
        "behavior_supervised_episode_count": sum(
            row["behavior_supervision_mask"] is True for row in records
        ),
        "behavior_unsupervised_episode_count": sum(
            row["behavior_supervision_mask"] is False for row in records
        ),
        "all_adjudicated_train_ready": not unsafe_episodes,
        "unsafe_condition_cache": unsafe_cache,
        "unsafe_training_data": unsafe_episodes or unsafe_cache,
        **(
            {
                "data_source_registry": data_source_registry,
                DATA_SOURCE_REGISTRY_HASH_FIELD: data_source_registry["sha256"],
            }
            if data_source_registry is not None
            else {}
        ),
        "source_manifests": [
            {"path": path, "sha256": digest} for path, digest in manifests
        ],
        "condition_cache": canonical_cache,
        "episodes": records,
    }


def _posttrain_contract_sha256(contract: Mapping) -> str:
    payload = {key: value for key, value in contract.items() if key != "sha256"}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _posttrain_lineage_error(message: str) -> ValueError:
    return ValueError(f"invalid posttrain condition-cache lineage: {message}")


def _validate_posttrain_condition_cache_lineage(
    generator_checkpoint: Mapping,
    cache_provenance: Mapping,
    *,
    generator_checkpoint_path: str | Path | None,
    cache_source_sha256: str,
) -> dict:
    if generator_checkpoint_path is None:
        raise _posttrain_lineage_error("current checkpoint path is required")
    current_path = Path(generator_checkpoint_path).resolve()
    if not current_path.is_file():
        raise _posttrain_lineage_error("current checkpoint file is missing")
    try:
        current_on_disk = torch.load(current_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise _posttrain_lineage_error(f"current checkpoint cannot be loaded: {error}") from error
    if not isinstance(current_on_disk, Mapping):
        raise _posttrain_lineage_error("current checkpoint payload must be an object")

    for field in POSTTRAIN_LINEAGE_MARKERS:
        if current_on_disk.get(field) != generator_checkpoint.get(field):
            raise _posttrain_lineage_error(
                f"in-memory checkpoint field {field!r} does not match its file"
            )
    if current_on_disk.get("global_step") != generator_checkpoint.get("global_step"):
        raise _posttrain_lineage_error("in-memory global_step does not match its file")
    try:
        validate_checkpoint_contract(current_on_disk, expected_action_dim=ACTION_DIM)
    except (TypeError, ValueError) as error:
        raise _posttrain_lineage_error(f"current checkpoint contract is invalid: {error}") from error

    if current_on_disk.get("posttrain_artifact_kind") != POSTTRAIN_LINEAGE_ARTIFACT_KIND:
        raise _posttrain_lineage_error("posttrain_artifact_kind is missing or unsupported")
    source = current_on_disk.get("posttrain_source")
    if not isinstance(source, Mapping):
        raise _posttrain_lineage_error("posttrain_source must be an object")
    source_path_value = source.get("checkpoint")
    source_sha256 = source.get("checkpoint_sha256")
    source_global_step = source.get("source_global_step")
    if not isinstance(source_path_value, str) or not source_path_value.strip():
        raise _posttrain_lineage_error("posttrain_source.checkpoint is missing")
    if not _is_sha256(source_sha256):
        raise _posttrain_lineage_error("posttrain_source.checkpoint_sha256 is invalid")
    if (
        isinstance(source_global_step, bool)
        or not isinstance(source_global_step, int)
        or source_global_step < 0
    ):
        raise _posttrain_lineage_error("posttrain_source.source_global_step is invalid")
    if source_sha256 != cache_source_sha256:
        raise _posttrain_lineage_error(
            "immediate source hash does not match the cache-bound generator"
        )

    cache_source_path_value = cache_provenance.get("generator_checkpoint")
    if not isinstance(cache_source_path_value, str) or not cache_source_path_value.strip():
        raise _posttrain_lineage_error("condition cache lacks its source generator path")
    source_path = Path(source_path_value).resolve()
    cache_source_path = Path(cache_source_path_value).resolve()
    if source_path != cache_source_path:
        raise _posttrain_lineage_error(
            "immediate source path does not match the cache-bound generator"
        )
    if not source_path.is_file():
        raise _posttrain_lineage_error("cache-bound source checkpoint file is missing")
    actual_source_sha256 = sha256_file(source_path)
    if actual_source_sha256 != source_sha256:
        raise _posttrain_lineage_error("cache-bound source checkpoint hash changed on disk")
    try:
        source_checkpoint = torch.load(source_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise _posttrain_lineage_error(f"source checkpoint cannot be loaded: {error}") from error
    if not isinstance(source_checkpoint, Mapping):
        raise _posttrain_lineage_error("source checkpoint payload must be an object")
    try:
        validate_checkpoint_contract(source_checkpoint, expected_action_dim=ACTION_DIM)
    except (TypeError, ValueError) as error:
        raise _posttrain_lineage_error(f"source checkpoint contract is invalid: {error}") from error
    if source_checkpoint.get("global_step") != source_global_step:
        raise _posttrain_lineage_error("source_global_step does not match the source checkpoint")

    preserved_fields = (
        "artifact_kind",
        "architecture",
        "condition_dim",
        "action_dim",
        "joint_order",
        "action_contract",
        "migration_source",
        "v2_contracts",
    )
    for field in preserved_fields:
        if current_on_disk.get(field) != source_checkpoint.get(field):
            raise _posttrain_lineage_error(f"post-training changed preserved field {field!r}")
    for field in ("mean", "std"):
        current_value = torch.as_tensor((current_on_disk.get("action_stats") or {}).get(field))
        source_value = torch.as_tensor((source_checkpoint.get("action_stats") or {}).get(field))
        if not torch.equal(current_value, source_value):
            raise _posttrain_lineage_error(f"post-training changed action_stats.{field}")

    posttrain_step = current_on_disk.get("posttrain_step")
    current_global_step = current_on_disk.get("global_step")
    if isinstance(posttrain_step, bool) or not isinstance(posttrain_step, int) or posttrain_step < 0:
        raise _posttrain_lineage_error("posttrain_step is invalid")
    if (
        isinstance(current_global_step, bool)
        or not isinstance(current_global_step, int)
        or current_global_step != source_global_step + posttrain_step
    ):
        raise _posttrain_lineage_error(
            "global_step does not equal source_global_step plus posttrain_step"
        )

    data_contract = current_on_disk.get("posttrain_data_contract")
    split_contract = current_on_disk.get("posttrain_split_contract")
    expected_contracts = (
        (
            "posttrain_data_contract",
            data_contract,
            "ula_v2_18d_interaction_posttrain_data",
        ),
        (
            "posttrain_split_contract",
            split_contract,
            "speaker_source_group_strict_split",
        ),
    )
    for name, contract, contract_type in expected_contracts:
        if not isinstance(contract, Mapping):
            raise _posttrain_lineage_error(f"{name} must be an object")
        if contract.get("contract_type") != contract_type or contract.get("contract_version") != 1:
            raise _posttrain_lineage_error(f"{name} type/version is invalid")
        recorded_sha256 = contract.get("sha256")
        if not _is_sha256(recorded_sha256) or _posttrain_contract_sha256(contract) != recorded_sha256:
            raise _posttrain_lineage_error(f"{name} hash is invalid")
    if data_contract.get("split_contract_sha256") != split_contract.get("sha256"):
        raise _posttrain_lineage_error("posttrain data contract is not bound to its split contract")

    posttrain_config = current_on_disk.get("posttrain_config")
    if not isinstance(posttrain_config, Mapping) or not posttrain_config:
        raise _posttrain_lineage_error("posttrain_config is missing")
    training_contract = current_on_disk.get("training_contract")
    if not isinstance(training_contract, Mapping):
        raise _posttrain_lineage_error("training_contract is missing")
    if (
        training_contract.get("mode") != POSTTRAIN_LINEAGE_MODE
        or training_contract.get("all_model_parameters_trainable") is not True
    ):
        raise _posttrain_lineage_error("training_contract does not describe full post-training")

    data_provenance = current_on_disk.get("data_provenance")
    if not isinstance(data_provenance, Mapping):
        raise _posttrain_lineage_error("data_provenance is missing")
    if data_provenance.get("data_contract_sha256") != data_contract.get("sha256"):
        raise _posttrain_lineage_error("data provenance is not bound to its data contract")
    lineage_cache = data_provenance.get("condition_cache")
    if not isinstance(lineage_cache, Mapping):
        raise _posttrain_lineage_error("data provenance lacks condition-cache lineage")
    for field in (
        "cache_sha256",
        "generator_checkpoint_sha256",
        "qwen_checkpoint_sha256",
        "qwen_model_name",
        "qwen_revision",
        "style_contract_sha256",
    ):
        if lineage_cache.get(field) != cache_provenance.get(field):
            raise _posttrain_lineage_error(
                f"data provenance condition cache changed field {field!r}"
            )
    return {
        "mode": "verified_immediate_posttrain_source",
        "current_checkpoint": str(current_path),
        "current_checkpoint_sha256": sha256_file(current_path),
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": source_sha256,
        "posttrain_step": posttrain_step,
    }


def validate_condition_cache_for_generator(
    generator_checkpoint: Mapping,
    cache_provenance: Mapping,
    *,
    generator_checkpoint_path: str | Path | None = None,
    allow_unsafe=False,
) -> dict:
    """Bind a condition cache to the text and style contracts of a generator."""
    assert_no_forbidden_data_lineage(
        generator_checkpoint, context="condition_cache.generator_checkpoint"
    )
    assert_no_forbidden_data_lineage(
        cache_provenance, context="condition_cache.provenance"
    )
    if cache_provenance.get("unsafe_condition_cache") is True:
        if allow_unsafe:
            return {"validated": False, "unsafe_condition_cache": True}
        raise ValueError("strict generator use refuses an unsafe condition cache")
    cache_artifact_kind = cache_provenance.get("artifact_kind")
    if cache_artifact_kind == MOTION_ONLY_CONDITION_CACHE_ARTIFACT_KIND:
        validate_motion_only_checkpoint_isolation(generator_checkpoint)
        expected = {
            "schema_version": MOTION_ONLY_CONDITION_CACHE_SCHEMA_VERSION,
            "formal_episode_contract": MOTION_ONLY_EPISODE_CONTRACT,
            "qwen_policy": MOTION_ONLY_NO_QWEN_POLICY,
            "kimodo_policy": MOTION_ONLY_NO_KIMODO_POLICY,
            "condition_policy": MOTION_ONLY_STYLE_ONLY_CONDITION_POLICY,
            "condition_dim": KIMODO_V2_CONDITION_DIM,
        }
        if any(
            cache_provenance.get(field) != value
            for field, value in expected.items()
        ):
            raise ValueError("motion-only cache/generator isolation contract changed")
        if {
            "qwen_checkpoint",
            "qwen_checkpoint_sha256",
            "qwen_model_name",
            "qwen_revision",
        }.intersection(cache_provenance):
            raise ValueError("motion-only cache must not contain Qwen lineage")
        style_contract = _generator_style_contract(generator_checkpoint)
        split_contract = (generator_checkpoint.get("v2_contracts") or {}).get(
            "split"
        ) or {}
        generator_registry = (
            (generator_checkpoint.get("sources") or {}).get(
                "data_source_registry"
            )
        )
        cache_registry = cache_provenance.get("data_source_registry")
        if generator_registry is not None:
            generator_registry = validate_data_source_registry_contract(
                generator_registry,
                expected_role=GENERATOR_FOUNDATION_ROLE,
            )
            cache_registry = validate_data_source_registry_contract(
                cache_registry,
                expected_role=GENERATOR_FOUNDATION_ROLE,
            )
            if (
                cache_registry != generator_registry
                or cache_provenance.get(DATA_SOURCE_REGISTRY_HASH_FIELD)
                != generator_registry["sha256"]
            ):
                raise ValueError(
                    "motion-only cache/generator data-source registry changed"
                )
        elif cache_registry is not None:
            raise ValueError(
                "legacy generator cannot acquire an unbound cache source registry"
            )
        if (
            cache_provenance.get("style_contract_sha256")
            != style_contract["sha256"]
            or cache_provenance.get("split_contract_sha256")
            != split_contract.get("sha256")
        ):
            raise ValueError("motion-only cache style/fixed-split contract changed")
        source_checkpoint_sha256 = cache_provenance.get(
            "generator_checkpoint_sha256"
        )
        if not _is_sha256(source_checkpoint_sha256):
            raise ValueError("motion-only cache lacks its target checkpoint hash")
        if generator_checkpoint_path is None:
            raise ValueError(
                "strict motion-only cache validation requires a generator checkpoint path"
            )
        if sha256_file(generator_checkpoint_path) != source_checkpoint_sha256:
            raise ValueError("motion-only cache targets a different generator checkpoint")
        return {
            "validated": True,
            "unsafe_condition_cache": False,
            "qwen_policy": MOTION_ONLY_NO_QWEN_POLICY,
            "kimodo_policy": MOTION_ONLY_NO_KIMODO_POLICY,
            "style_contract_sha256": style_contract["sha256"],
            "split_contract_sha256": split_contract["sha256"],
            **(
                {DATA_SOURCE_REGISTRY_HASH_FIELD: generator_registry["sha256"]}
                if generator_registry is not None
                else {}
            ),
            "generator_checkpoint_sha256": source_checkpoint_sha256,
            "generator_checkpoint_compatibility": "direct_checkpoint",
        }
    expression_turn_v8_cache = (
        cache_artifact_kind == "ula_v2_expression_turn_v8_condition_cache"
    )
    if cache_artifact_kind not in {
        "ula_v2_qwen_motion_condition_cache",
        "ula_v2_expression_turn_v8_condition_cache",
    }:
        raise ValueError("generator requires a versioned Qwen motion condition cache")
    if expression_turn_v8_cache:
        from upper_body_skeleton.ula_v2_expression_turn_episode import (
            CONDITION_CACHE_SCHEMA_VERSION as EXPRESSION_CACHE_SCHEMA_VERSION,
            FORMAL_EPISODE_CONTRACT as EXPRESSION_FORMAL_EPISODE_CONTRACT,
        )

        if (
            cache_provenance.get("schema_version") != EXPRESSION_CACHE_SCHEMA_VERSION
            or cache_provenance.get("formal_episode_contract")
            != EXPRESSION_FORMAL_EPISODE_CONTRACT
            or generator_checkpoint.get("formal_episode_contract")
            != EXPRESSION_FORMAL_EPISODE_CONTRACT
        ):
            raise ValueError("expression-turn cache/generator formal contract changed")
        expected_contract_hashes = {
            "text_motion_contract_sha256": (
                (generator_checkpoint.get("v2_contracts") or {}).get(
                    "text_motion_latent"
                )
                or {}
            ).get("sha256"),
            "semantic_supervision_contract_sha256": (
                generator_checkpoint.get("semantic_supervision_contract") or {}
            ).get("sha256"),
        }
        for field, expected_hash in expected_contract_hashes.items():
            if not expected_hash or cache_provenance.get(field) != expected_hash:
                raise ValueError(f"expression-turn condition cache {field} changed")
    text_contract = (generator_checkpoint.get("v2_contracts") or {}).get(
        "text_motion_latent"
    ) or {}
    source = text_contract.get("source") or {}
    expected = {
        "qwen_checkpoint_sha256": source.get("checkpoint_sha256"),
        "qwen_model_name": source.get("model_name"),
        "qwen_revision": source.get("revision"),
    }
    if not all(isinstance(value, str) and value for value in expected.values()):
        raise ValueError("generator checkpoint lacks a complete Qwen text-motion latent contract")
    for field, value in expected.items():
        if cache_provenance.get(field) != value:
            raise ValueError(f"condition cache {field} does not match the generator contract")
    style_validation = {}
    if (
        cache_provenance.get("schema_version") == CONDITION_CACHE_SCHEMA_VERSION
        or expression_turn_v8_cache
    ):
        style_contract = _generator_style_contract(generator_checkpoint)
        style_sha256 = style_contract["sha256"]
        if cache_provenance.get("style_contract_sha256") != style_sha256:
            raise ValueError("condition cache style contract does not match the generator")
        source_checkpoint_sha256 = cache_provenance.get("generator_checkpoint_sha256")
        if not isinstance(source_checkpoint_sha256, str) or not source_checkpoint_sha256:
            raise ValueError("condition cache lacks its target generator checkpoint hash")
        lineage_validation = {}
        if generator_checkpoint_path is not None:
            current_sha256 = sha256_file(generator_checkpoint_path)
            has_posttrain_markers = any(
                field in generator_checkpoint for field in POSTTRAIN_LINEAGE_MARKERS
            )
            if source_checkpoint_sha256 == current_sha256:
                compatibility_mode = "direct_checkpoint"
            elif has_posttrain_markers:
                lineage_validation = _validate_posttrain_condition_cache_lineage(
                    generator_checkpoint,
                    cache_provenance,
                    generator_checkpoint_path=generator_checkpoint_path,
                    cache_source_sha256=source_checkpoint_sha256,
                )
                compatibility_mode = lineage_validation["mode"]
            else:
                migration_sha256 = (generator_checkpoint.get("migration_source") or {}).get(
                    "sha256"
                )
                if source_checkpoint_sha256 != migration_sha256:
                    raise ValueError("condition cache targets a different generator checkpoint")
                compatibility_mode = "migration_source"
        else:
            raise ValueError(
                "strict versioned condition-cache validation requires a generator checkpoint path"
            )
        style_validation = {
            "style_contract_sha256": style_sha256,
            "generator_checkpoint_sha256": source_checkpoint_sha256,
            "generator_checkpoint_compatibility": compatibility_mode,
            **({"posttrain_lineage": lineage_validation} if lineage_validation else {}),
        }
    return {
        "validated": True,
        "unsafe_condition_cache": False,
        **expected,
        **style_validation,
    }


def validate_qwen_checkpoint_for_generator(
    generator_checkpoint: Mapping, qwen_checkpoint: str | Path
) -> dict:
    qwen_checkpoint = Path(qwen_checkpoint)
    payload = torch.load(qwen_checkpoint, map_location="cpu", weights_only=True)
    qwen = payload.get("qwen") or {}
    provenance = {
        "artifact_kind": "ula_v2_qwen_motion_condition_cache",
        "unsafe_condition_cache": False,
        "qwen_checkpoint_sha256": sha256_file(qwen_checkpoint),
        "qwen_model_name": qwen.get("model_name"),
        "qwen_revision": qwen.get("revision"),
    }
    return validate_condition_cache_for_generator(generator_checkpoint, provenance)


def resample_trajectory(actions: np.ndarray, frame_count: int) -> np.ndarray:
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"actions must have shape [frames, {ACTION_DIM}]")
    frame_count = int(frame_count)
    if frame_count < 2:
        raise ValueError("frame_count must be at least 2")
    if actions.shape[0] == frame_count:
        return actions.astype(np.float32, copy=True)
    source = np.linspace(0.0, 1.0, actions.shape[0])
    target = np.linspace(0.0, 1.0, frame_count)
    return np.stack(
        [np.interp(target, source, actions[:, index]) for index in range(ACTION_DIM)],
        axis=-1,
    ).astype(np.float32)


@dataclass
class HeadAdapterPolicy:
    handles: list
    frozen_state: dict[str, torch.Tensor]

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def configure_head_adapter_policy(model) -> HeadAdapterPolicy:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    handles = []
    for name in ("input.weight", "output.weight", "output.bias"):
        parameter = dict(model.named_parameters())[name]
        parameter.requires_grad_(True)
        mask = torch.zeros_like(parameter)
        if name == "input.weight":
            mask[:, HEAD_SLICE] = 1
        else:
            mask[HEAD_SLICE] = 1
        handles.append(parameter.register_hook(lambda gradient, mask=mask: gradient * mask))
    return HeadAdapterPolicy(
        handles=handles,
        frozen_state={name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
    )


def restore_frozen_weights(model, policy: HeadAdapterPolicy) -> None:
    with torch.no_grad():
        state = model.state_dict()
        for name, frozen in policy.frozen_state.items():
            source = frozen.to(device=state[name].device, dtype=state[name].dtype)
            if name == "input.weight":
                state[name][:, :LEGACY_ACTION_DIM].copy_(source[:, :LEGACY_ACTION_DIM])
            elif name in ("output.weight", "output.bias"):
                state[name][:LEGACY_ACTION_DIM].copy_(source[:LEGACY_ACTION_DIM])
            else:
                state[name].copy_(source)


def frozen_weight_max_error(model, policy: HeadAdapterPolicy) -> float:
    maximum = 0.0
    state = model.state_dict()
    for name, frozen in policy.frozen_state.items():
        current = state[name].detach().cpu()
        if name == "input.weight":
            current, frozen = current[:, :LEGACY_ACTION_DIM], frozen[:, :LEGACY_ACTION_DIM]
        elif name in ("output.weight", "output.bias"):
            current, frozen = current[:LEGACY_ACTION_DIM], frozen[:LEGACY_ACTION_DIM]
        maximum = max(maximum, float((current - frozen).abs().max()))
    return maximum


def trainable_parameter_count(model) -> int:
    # Only the gradient-mask slices count; the containing tensors are larger.
    hidden_dim = int(model.hidden_dim)
    return hidden_dim * 3 + 3 * hidden_dim + 3


def _normalize(actions: torch.Tensor, stats: Mapping) -> torch.Tensor:
    mean = torch.as_tensor(stats["mean"], dtype=actions.dtype, device=actions.device)
    std = torch.as_tensor(stats["std"], dtype=actions.dtype, device=actions.device)
    return (actions - mean) / std


def _head_objective(
    model,
    actions,
    conditions,
    *,
    base_model=None,
    body_distillation_weight=BODY_DISTILLATION_WEIGHT,
    generator=None,
) -> dict[str, torch.Tensor]:
    if generator is None:
        noise = torch.randn_like(actions)
        t = torch.rand(actions.shape[0], device=actions.device)
    else:
        noise = torch.randn(actions.shape, device=actions.device, generator=generator)
        t = torch.rand(actions.shape[0], device=actions.device, generator=generator)
    x_t = (1.0 - t[:, None, None]) * noise + t[:, None, None] * actions
    target = actions - noise
    predicted = model(x_t, t, conditions)
    reconstructed = x_t + (1.0 - t[:, None, None]) * predicted
    predicted_head = predicted[..., HEAD_SLICE]
    target_head = target[..., HEAD_SLICE]
    reconstructed_head = reconstructed[..., HEAD_SLICE]
    actions_head = actions[..., HEAD_SLICE]
    velocity_predicted = reconstructed_head[:, 1:] - reconstructed_head[:, :-1]
    velocity_target = actions_head[:, 1:] - actions_head[:, :-1]
    losses = {
        "head_flow": F.mse_loss(predicted_head, target_head),
        "head_position": F.smooth_l1_loss(reconstructed_head, actions_head),
        "head_velocity": F.smooth_l1_loss(velocity_predicted, velocity_target),
    }
    if base_model is not None:
        with torch.no_grad():
            teacher_body = base_model(x_t[..., :LEGACY_ACTION_DIM], t, conditions)
        losses["body_distillation"] = F.mse_loss(
            predicted[..., :LEGACY_ACTION_DIM], teacher_body
        )
    else:
        losses["body_distillation"] = predicted.new_zeros(())
    losses["total"] = (
        losses["head_flow"]
        + 0.25 * losses["head_position"]
        + 0.05 * losses["head_velocity"]
        + float(body_distillation_weight) * losses["body_distillation"]
    )
    return losses


def _split_episodes(
    episodes: Sequence[dict], seed: int
) -> tuple[list[dict], list[dict], str]:
    evaluation = [item for item in episodes if item.get("eval_eligible") is True]
    training = [item for item in episodes if item.get("split_assignment") == "train"]
    if evaluation:
        if not training:
            training = [item for item in episodes if item not in evaluation]
        if not training:
            raise ValueError("manifest provides evaluation clips but no training clips")
        return training, evaluation, "manifest_eval_eligible"
    if episodes and all(item.get("split_assignment") == "train" for item in episodes):
        # These clips are explicitly train-only.  Reusing them for an optimization
        # monitor is honest; presenting a synthetic holdout as generalization is not.
        return list(episodes), list(episodes), "train_reuse_no_eval_eligible_clips"
    ordered = sorted(
        episodes,
        key=lambda item: hashlib.sha256(f"{seed}:{item['clip_id']}".encode()).hexdigest(),
    )
    if len(ordered) == 1:
        return ordered, ordered, "single_clip_optimization_monitor"
    validation_count = max(1, int(round(len(ordered) * 0.2)))
    validation_count = min(validation_count, len(ordered) - 1)
    return ordered[validation_count:], ordered[:validation_count], "deterministic_clip_holdout"


def _sample_batch(episodes, *, batch_size, frames, device, rng):
    selected = [episodes[rng.randrange(len(episodes))] for _ in range(int(batch_size))]
    actions = np.stack([resample_trajectory(item["actions"], frames) for item in selected])
    conditions = np.stack([item["condition"] for item in selected])
    return (
        torch.as_tensor(actions, dtype=torch.float32, device=device),
        torch.as_tensor(conditions, dtype=torch.float32, device=device),
    )


def evaluate_head_adapter(
    model,
    episodes,
    *,
    action_stats,
    frames,
    device,
    base_model=None,
    body_distillation_weight=BODY_DISTILLATION_WEIGHT,
    seed=0,
) -> dict:
    generator = torch.Generator(device=device).manual_seed(int(seed))
    actions = torch.as_tensor(
        np.stack([resample_trajectory(item["actions"], frames) for item in episodes]),
        dtype=torch.float32,
        device=device,
    )
    conditions = torch.as_tensor(
        np.stack([item["condition"] for item in episodes]),
        dtype=torch.float32,
        device=device,
    )
    actions = _normalize(actions, action_stats)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        losses = _head_objective(
            model,
            actions,
            conditions,
            base_model=base_model,
            body_distillation_weight=body_distillation_weight,
            generator=generator,
        )
    model.train(was_training)
    return {name: float(value.cpu()) for name, value in losses.items()}


def nonzero_head_forward_drift_metrics(
    base_model,
    expanded_model,
    *,
    seed=41,
    batch_size=4,
    frames=24,
    device="cpu",
) -> dict:
    """Measure body-output cross-talk with deterministic non-zero head inputs."""
    generator = torch.Generator(device=device).manual_seed(int(seed))
    body = torch.randn(
        int(batch_size), int(frames), LEGACY_ACTION_DIM, generator=generator, device=device
    )
    head = torch.randn(
        int(batch_size), int(frames), ACTION_DIM - LEGACY_ACTION_DIM,
        generator=generator,
        device=device,
    )
    condition = torch.randn(
        int(batch_size), KIMODO_V2_CONDITION_DIM, generator=generator, device=device
    )
    t = torch.rand(int(batch_size), generator=generator, device=device)
    base_training, expanded_training = base_model.training, expanded_model.training
    base_model.eval()
    expanded_model.eval()
    with torch.no_grad():
        teacher = base_model(body, t, condition)
        student = expanded_model(torch.cat([body, head], dim=-1), t, condition)[
            ..., :LEGACY_ACTION_DIM
        ]
    base_model.train(base_training)
    expanded_model.train(expanded_training)
    absolute = (student - teacher).abs().detach().float().cpu().reshape(-1)
    return {
        "seed": int(seed),
        "batch_size": int(batch_size),
        "frames": int(frames),
        "mean_abs_normalized": float(absolute.mean()),
        "p95_abs_normalized": float(torch.quantile(absolute, 0.95)),
        "max_abs_normalized": float(absolute.max()),
    }


def body_sampling_drift_metrics(
    base_model,
    expanded_model,
    conditions,
    *,
    action_stats,
    frames=90,
    steps=24,
    seeds=(53, 97),
    device="cpu",
) -> dict:
    """Compare complete coupled 15D/18D flows from identical body noise."""
    if int(frames) < 2 or int(steps) < 1:
        raise ValueError("body drift sampling requires at least two frames and one step")
    condition = torch.as_tensor(np.stack(conditions), dtype=torch.float32, device=device)
    if condition.ndim != 2 or condition.shape[1] != KIMODO_V2_CONDITION_DIM:
        raise ValueError("body drift conditions must be [batch, 264]")
    lower, upper = _joint_bounds(action_stats, device=device)
    base_lower, base_upper = lower[:LEGACY_ACTION_DIM], upper[:LEGACY_ACTION_DIM]
    mean = torch.as_tensor(action_stats["mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(action_stats["std"], dtype=torch.float32, device=device)
    velocity_absolute = []
    physical_absolute = []
    base_training, expanded_training = base_model.training, expanded_model.training
    base_model.eval()
    expanded_model.eval()
    with torch.no_grad():
        for seed in seeds:
            generator = torch.Generator(device=device).manual_seed(int(seed))
            body = torch.randn(
                condition.shape[0], int(frames), LEGACY_ACTION_DIM,
                generator=generator,
                device=device,
            )
            head = torch.randn(
                condition.shape[0], int(frames), ACTION_DIM - LEGACY_ACTION_DIM,
                generator=generator,
                device=device,
            )
            body = torch.maximum(torch.minimum(body, base_upper), base_lower)
            head = torch.maximum(
                torch.minimum(head, upper[HEAD_SLICE]), lower[HEAD_SLICE]
            )
            base_x = body.clone()
            expanded_x = torch.cat([body, head], dim=-1)
            for index in range(int(steps)):
                t = torch.full(
                    (condition.shape[0],),
                    index / float(steps),
                    dtype=torch.float32,
                    device=device,
                )
                base_velocity = base_model(base_x, t, condition)
                expanded_velocity = expanded_model(expanded_x, t, condition)
                velocity_absolute.append(
                    (expanded_velocity[..., :LEGACY_ACTION_DIM] - base_velocity)
                    .abs()
                    .detach()
                    .float()
                    .cpu()
                    .reshape(-1)
                )
                base_x = torch.maximum(
                    torch.minimum(base_x + base_velocity / float(steps), base_upper),
                    base_lower,
                )
                expanded_x = torch.maximum(
                    torch.minimum(expanded_x + expanded_velocity / float(steps), upper),
                    lower,
                )
            base_physical = base_x * std[:LEGACY_ACTION_DIM] + mean[:LEGACY_ACTION_DIM]
            expanded_physical = (
                expanded_x[..., :LEGACY_ACTION_DIM] * std[:LEGACY_ACTION_DIM]
                + mean[:LEGACY_ACTION_DIM]
            )
            physical_absolute.append(
                (expanded_physical - base_physical).abs().detach().float().cpu().reshape(-1)
            )
    base_model.train(base_training)
    expanded_model.train(expanded_training)
    velocity_absolute = torch.cat(velocity_absolute)
    physical_absolute = torch.cat(physical_absolute)
    return {
        "seeds": [int(value) for value in seeds],
        "condition_count": int(condition.shape[0]),
        "frames": int(frames),
        "sampling_steps": int(steps),
        "body_velocity_mean_abs_normalized": float(velocity_absolute.mean()),
        "body_velocity_p95_abs_normalized": float(torch.quantile(velocity_absolute, 0.95)),
        "body_velocity_max_abs_normalized": float(velocity_absolute.max()),
        "body_mean_abs_rad": float(physical_absolute.mean()),
        "body_p95_abs_rad": float(torch.quantile(physical_absolute, 0.95)),
        "body_max_abs_rad": float(physical_absolute.max()),
    }


def assess_body_compatibility(nonzero_forward: Mapping, sampling: Mapping) -> dict:
    observed = {
        "nonzero_head_forward_mean_abs_normalized": nonzero_forward[
            "mean_abs_normalized"
        ],
        "nonzero_head_forward_p95_abs_normalized": nonzero_forward[
            "p95_abs_normalized"
        ],
        "sampling_body_mean_abs_rad": sampling["body_mean_abs_rad"],
        "sampling_body_p95_abs_rad": sampling["body_p95_abs_rad"],
        "sampling_body_max_abs_rad": sampling["body_max_abs_rad"],
    }
    checks = {
        name: float(observed[name]) <= float(limit)
        for name, limit in BODY_COMPATIBILITY_THRESHOLDS.items()
    }
    return {
        "passed": all(checks.values()),
        "thresholds": dict(BODY_COMPATIBILITY_THRESHOLDS),
        "observed": observed,
        "checks": checks,
        "nonzero_head_forward": dict(nonzero_forward),
        "complete_sampling": dict(sampling),
    }


def train_head_adapter(
    *,
    base_checkpoint_path: str | Path,
    episodes: Sequence[dict],
    output_dir: str | Path,
    steps=20,
    batch_size=2,
    frames=48,
    lr=1e-3,
    body_distillation_weight=BODY_DISTILLATION_WEIGHT,
    device="auto",
    seed=7,
    log_interval=5,
    require_train_ready=False,
    allow_unsafe_condition_cache=False,
) -> dict:
    if not episodes or any("condition" not in item for item in episodes):
        raise ValueError("episodes with cached 264D conditions are required")
    body_distillation_weight = float(body_distillation_weight)
    if not np.isfinite(body_distillation_weight) or body_distillation_weight <= 0:
        raise ValueError("body_distillation_weight must be finite and positive")
    data_provenance = training_data_provenance(episodes)
    if require_train_ready and not data_provenance["all_adjudicated_train_ready"]:
        raise ValueError("strict training requires adjudicated_train_ready episodes only")
    resolved_device = torch.device(
        device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    base_model, base_checkpoint = load_contract_checkpoint(
        base_checkpoint_path, expected_action_dim=LEGACY_ACTION_DIM, device=resolved_device
    )
    base_model.requires_grad_(False).eval()
    validate_condition_cache_for_generator(
        base_checkpoint,
        data_provenance.get("condition_cache") or {},
        generator_checkpoint_path=base_checkpoint_path,
        allow_unsafe=allow_unsafe_condition_cache,
    )
    action_stats = compute_18d_action_stats(
        [item["actions"] for item in episodes], base_checkpoint["action_stats"]
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    migrated_path = output_dir / "migrated_initial.pt"
    migrated, migration_verification = migrate_15d_checkpoint(
        base_checkpoint_path, migrated_path, action_stats=action_stats
    )
    migrated["data_provenance"] = data_provenance
    _atomic_torch_save(migrated, migrated_path)
    model = instantiate_checkpoint_model(migrated, device=resolved_device)
    policy = configure_head_adapter_policy(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(lr),
        weight_decay=0.0,
    )
    train_episodes, validation_episodes, validation_mode = _split_episodes(episodes, seed)
    initial_validation = evaluate_head_adapter(
        model,
        validation_episodes,
        action_stats=action_stats,
        frames=frames,
        device=resolved_device,
        base_model=base_model,
        body_distillation_weight=body_distillation_weight,
        seed=seed + 1000,
    )
    initial_train = evaluate_head_adapter(
        model,
        train_episodes,
        action_stats=action_stats,
        frames=frames,
        device=resolved_device,
        base_model=base_model,
        body_distillation_weight=body_distillation_weight,
        seed=seed + 2000,
    )
    progress_path = output_dir / "progress.jsonl"
    progress_path.write_text("", encoding="utf-8")
    rng = random.Random(seed + 1)
    best_loss = float("inf")
    best_step = 0
    last_losses = {}
    best_path = output_dir / "ula_fm_checkpoint.pt"
    last_path = output_dir / "last.pt"
    # The backbone is frozen and the teacher runs in eval mode. Keeping the
    # student in eval mode makes distillation measure head-input cross-talk,
    # rather than unrelated Transformer dropout noise. Gradients still flow
    # through the three trainable projection slices.
    model.eval()
    for step in range(1, int(steps) + 1):
        actions, conditions = _sample_batch(
            train_episodes,
            batch_size=batch_size,
            frames=frames,
            device=resolved_device,
            rng=rng,
        )
        actions = _normalize(actions, action_stats)
        optimizer.zero_grad(set_to_none=True)
        losses = _head_objective(
            model,
            actions,
            conditions,
            base_model=base_model,
            body_distillation_weight=body_distillation_weight,
        )
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(f"non-finite head adapter loss at step {step}")
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
        )
        optimizer.step()
        restore_frozen_weights(model, policy)
        frozen_error = frozen_weight_max_error(model, policy)
        if frozen_error != 0.0:
            raise RuntimeError(f"frozen 15D weights changed at step {step}: {frozen_error}")
        last_losses = {name: float(value.detach().cpu()) for name, value in losses.items()}
        if step == 1 or step % int(log_interval) == 0 or step == int(steps):
            validation = evaluate_head_adapter(
                model,
                validation_episodes,
                action_stats=action_stats,
                frames=frames,
                device=resolved_device,
                base_model=base_model,
                body_distillation_weight=body_distillation_weight,
                seed=seed + 1000,
            )
            event = {
                "step": step,
                "train": last_losses,
                "validation": validation,
                "frozen_prefix_max_abs_error": frozen_error,
            }
            with progress_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, sort_keys=True) + "\n")
            if validation["total"] < best_loss:
                best_loss = validation["total"]
                best_step = step
                payload = deepcopy(migrated)
                payload.update(
                    {
                        "model_state_dict": {
                            name: value.detach().cpu().clone()
                            for name, value in model.state_dict().items()
                        },
                        "global_step": int(base_checkpoint.get("global_step", 0)) + step,
                        "adapter_step": step,
                        "best_step": step,
                        "best_validation_loss": best_loss,
                        "validation_metrics": validation,
                        "training_contract": {
                            "adapter_policy": ADAPTER_POLICY,
                            "base_global_step": int(base_checkpoint.get("global_step", 0)),
                            "head_adapter_steps": int(steps),
                            "trainable_parameter_count": trainable_parameter_count(model),
                            "only_new_projection_slices_trainable": True,
                            "student_forward_mode": "eval_dropout_disabled",
                            "weight_decay": 0.0,
                            "body_distillation_weight": body_distillation_weight,
                            "validation_mode": validation_mode,
                        },
                    }
                )
                _atomic_torch_save(payload, best_path)

    final_adapter_train = evaluate_head_adapter(
        model,
        train_episodes,
        action_stats=action_stats,
        frames=frames,
        device=resolved_device,
        base_model=base_model,
        body_distillation_weight=body_distillation_weight,
        seed=seed + 2000,
    )
    final_adapter_validation = evaluate_head_adapter(
        model,
        validation_episodes,
        action_stats=action_stats,
        frames=frames,
        device=resolved_device,
        base_model=base_model,
        body_distillation_weight=body_distillation_weight,
        seed=seed + 1000,
    )
    last_payload = deepcopy(migrated)
    last_payload.update(
        {
            "model_state_dict": {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            },
            "global_step": int(base_checkpoint.get("global_step", 0)) + int(steps),
            "adapter_step": int(steps),
            "best_step": int(best_step),
            "best_validation_loss": float(best_loss),
            "validation_metrics": final_adapter_validation,
            "training_contract": {
                "adapter_policy": ADAPTER_POLICY,
                "base_global_step": int(base_checkpoint.get("global_step", 0)),
                "head_adapter_steps": int(steps),
                "trainable_parameter_count": trainable_parameter_count(model),
                "only_new_projection_slices_trainable": True,
                "student_forward_mode": "eval_dropout_disabled",
                "weight_decay": 0.0,
                "body_distillation_weight": body_distillation_weight,
                "validation_mode": validation_mode,
            },
        }
    )
    _atomic_torch_save(last_payload, last_path)

    best_model, best_checkpoint = load_contract_checkpoint(
        best_path, expected_action_dim=ACTION_DIM, device=resolved_device
    )
    final_validation = evaluate_head_adapter(
        best_model,
        validation_episodes,
        action_stats=action_stats,
        frames=frames,
        device=resolved_device,
        base_model=base_model,
        body_distillation_weight=body_distillation_weight,
        seed=seed + 1000,
    )
    legacy_error = legacy_forward_max_error(
        base_model, best_model, seed=seed + 2000, device=resolved_device
    )
    prefix_error = verify_migrated_prefix(base_checkpoint, best_checkpoint)
    if prefix_error != 0.0:
        raise RuntimeError(f"trained checkpoint changed frozen 15D weights: {prefix_error}")
    if legacy_error > LEGACY_FORWARD_ATOL:
        raise RuntimeError(
            f"trained checkpoint legacy forward error {legacy_error} exceeds {LEGACY_FORWARD_ATOL}"
        )
    nonzero_forward = nonzero_head_forward_drift_metrics(
        base_model,
        best_model,
        seed=seed + 3000,
        device=resolved_device,
    )
    sampling_drift = body_sampling_drift_metrics(
        base_model,
        best_model,
        [item["condition"] for item in validation_episodes[:4]],
        action_stats=action_stats,
        frames=max(90, int(frames)),
        steps=24,
        seeds=(seed + 4000, seed + 5000),
        device=resolved_device,
    )
    body_compatibility = assess_body_compatibility(nonzero_forward, sampling_drift)
    if not body_compatibility["passed"]:
        failed = [name for name, passed in body_compatibility["checks"].items() if not passed]
        raise RuntimeError(f"trained checkpoint failed body compatibility gates: {failed}")
    best_checkpoint["body_compatibility"] = body_compatibility
    _atomic_torch_save(best_checkpoint, best_path)
    summary = {
        "artifact_kind": "ula_v2_18d_head_adapter_training_summary",
        "contract_version": CONTRACT_VERSION,
        "device": str(resolved_device),
        "episodes": len(episodes),
        "split": {"train": len(train_episodes), "validation": len(validation_episodes)},
        "validation_mode": validation_mode,
        "generalization_evaluation_available": validation_mode == "manifest_eval_eligible",
        "steps": int(steps),
        "best_step": int(best_step),
        "initial_validation": initial_validation,
        "initial_train": initial_train,
        "final_validation": final_validation,
        "final_adapter_validation": final_adapter_validation,
        "final_adapter_train": final_adapter_train,
        "last_train": last_losses,
        "trainable_parameter_count": trainable_parameter_count(model),
        "body_distillation_weight": body_distillation_weight,
        "student_forward_mode": "eval_dropout_disabled",
        "body_compatibility": body_compatibility,
        "total_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "frozen_weight_prefix_max_abs_error": prefix_error,
        "legacy_zero_padded_forward_max_abs_error": legacy_error,
        "legacy_forward_atol": LEGACY_FORWARD_ATOL,
        "migration_verification": migration_verification,
        "checkpoint": str(best_path.resolve()),
        "last_checkpoint": str(last_path.resolve()),
        "migrated_initial_checkpoint": str(migrated_path.resolve()),
        "source_checkpoint_sha256": sha256_file(base_checkpoint_path),
        "data_provenance": data_provenance,
    }
    _atomic_json_save(summary, output_dir / "training_summary.json")
    policy.remove()
    return summary


def benchmark_contract_inference(
    model,
    condition,
    *,
    frames=90,
    steps=24,
    warmup=2,
    repeats=5,
    seed=7,
    device="cpu",
) -> tuple[dict, np.ndarray]:
    """Measure full flow-trajectory wall time, separate from 30 Hz playback."""
    if int(warmup) < 0 or int(repeats) < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")

    def synchronize():
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(torch.device(device))

    trajectory = None
    for index in range(int(warmup)):
        trajectory = sample_contract_trajectory(
            model,
            condition,
            frames=frames,
            steps=steps,
            seed=seed + index,
            device=device,
        )
    durations = []
    for index in range(int(repeats)):
        synchronize()
        started = time.perf_counter()
        trajectory = sample_contract_trajectory(
            model,
            condition,
            frames=frames,
            steps=steps,
            seed=seed + 1000 + index,
            device=device,
        )
        synchronize()
        durations.append(time.perf_counter() - started)
    median_seconds = float(np.median(durations))
    mean_seconds = float(np.mean(durations))
    playback_seconds = float(frames) / 30.0
    device_value = torch.device(device)
    gpu_name = (
        torch.cuda.get_device_name(device_value)
        if device_value.type == "cuda"
        else None
    )
    metrics = {
        "artifact_kind": "ula_v2_full_trajectory_inference_benchmark",
        "action_dim": int(model.action_dim),
        "device": str(device_value),
        "gpu_name": gpu_name,
        "frames": int(frames),
        "sampling_steps": int(steps),
        "warmup_runs": int(warmup),
        "measured_runs": int(repeats),
        "wall_seconds_each": durations,
        "wall_seconds_median": median_seconds,
        "wall_seconds_mean": mean_seconds,
        "output_frames_per_wall_second_median": float(frames) / median_seconds,
        "model_forward_calls_per_second_median": float(steps) / median_seconds,
        "equivalent_wall_ms_per_output_frame": median_seconds * 1000.0 / float(frames),
        "trajectory_duration_at_30hz_seconds": playback_seconds,
        "realtime_factor_at_30hz": playback_seconds / median_seconds,
        "playback_control_hz": 30.0,
        "note": (
            "Output frames/s is batched full-trajectory throughput, not a causal control-loop rate; "
            "30 Hz is the downstream playback contract."
        ),
    }
    return metrics, trajectory


def benchmark_text_to_trajectory_inference(
    model,
    prompt,
    qwen_checkpoint,
    *,
    frames=90,
    steps=24,
    warmup=2,
    repeats=5,
    seed=7,
    device="cuda",
    local_files_only=True,
) -> tuple[dict, np.ndarray]:
    """Benchmark warm text encoding plus full flow generation with models resident."""
    if int(warmup) < 0 or int(repeats) < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    from upper_body_skeleton.cross_modal_latent import load_qwen_motion_text_encoder

    device_value = torch.device(device)
    text_encoder, qwen_payload = load_qwen_motion_text_encoder(
        qwen_checkpoint,
        device=device_value,
        local_files_only=local_files_only,
    )

    def synchronize():
        if device_value.type == "cuda":
            torch.cuda.synchronize(device_value)

    def encode_condition():
        base = build_condition_from_text(
            prompt,
            condition_dim=KIMODO_CONDITION_DIM,
        ).astype(np.float32)
        latent = text_encoder.encode([prompt], batch_size=1)[0].astype(np.float32)
        condition = np.concatenate([base, latent])
        if condition.shape != (KIMODO_V2_CONDITION_DIM,):
            raise RuntimeError(f"warm text condition has shape {condition.shape}")
        return condition

    trajectory = None
    for index in range(int(warmup)):
        condition = encode_condition()
        trajectory = sample_contract_trajectory(
            model,
            condition,
            frames=frames,
            steps=steps,
            seed=seed + index,
            device=device_value,
        )
    if device_value.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device_value)
    text_durations = []
    generator_durations = []
    total_durations = []
    for index in range(int(repeats)):
        synchronize()
        total_started = time.perf_counter()
        text_started = total_started
        condition = encode_condition()
        synchronize()
        text_finished = time.perf_counter()
        trajectory = sample_contract_trajectory(
            model,
            condition,
            frames=frames,
            steps=steps,
            seed=seed + 1000 + index,
            device=device_value,
        )
        synchronize()
        finished = time.perf_counter()
        text_durations.append(text_finished - text_started)
        generator_durations.append(finished - text_finished)
        total_durations.append(finished - total_started)

    text_median = float(np.median(text_durations))
    generator_median = float(np.median(generator_durations))
    total_median = float(np.median(total_durations))
    gpu_name = (
        torch.cuda.get_device_name(device_value) if device_value.type == "cuda" else None
    )
    metrics = {
        "artifact_kind": "ula_v2_warm_text_to_trajectory_inference_benchmark",
        "action_dim": int(model.action_dim),
        "device": str(device_value),
        "gpu_name": gpu_name,
        "models_co_resident": True,
        "cold_model_load_included": False,
        "frames": int(frames),
        "sampling_steps": int(steps),
        "warmup_runs": int(warmup),
        "measured_runs": int(repeats),
        "text_encode_wall_seconds_each": text_durations,
        "generator_wall_seconds_each": generator_durations,
        "total_wall_seconds_each": total_durations,
        "text_encode_wall_seconds_median": text_median,
        "generator_wall_seconds_median": generator_median,
        "total_wall_seconds_median": total_median,
        "text_encode_wall_ms_median": text_median * 1000.0,
        "generator_wall_ms_median": generator_median * 1000.0,
        "total_wall_ms_median": total_median * 1000.0,
        "end_to_end_trajectories_per_second": 1.0 / total_median,
        "end_to_end_output_frames_per_wall_second": float(frames) / total_median,
        "trajectory_duration_at_30hz_seconds": float(frames) / 30.0,
        "realtime_factor_at_30hz": (float(frames) / 30.0) / total_median,
        "playback_control_hz": 30.0,
        "qwen_checkpoint": str(Path(qwen_checkpoint).resolve()),
        "qwen_checkpoint_sha256": sha256_file(qwen_checkpoint),
        "qwen_model_name": (qwen_payload.get("qwen") or {}).get("model_name"),
        "qwen_revision": (qwen_payload.get("qwen") or {}).get("revision"),
        "cuda_peak_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device_value))
            if device_value.type == "cuda"
            else None
        ),
        "cuda_peak_memory_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device_value))
            if device_value.type == "cuda"
            else None
        ),
        "note": (
            "Warm Qwen text encoding and full-trajectory flow generation are timed with both "
            "models already loaded; 30 Hz remains the separate downstream playback contract."
        ),
    }
    return metrics, trajectory


def _joint_bounds(stats: Mapping, *, device):
    lower = torch.tensor(
        [JOINT_LIMITS_18D[name][0] for name in JOINT_ORDER_18D],
        dtype=torch.float32,
        device=device,
    )
    upper = torch.tensor(
        [JOINT_LIMITS_18D[name][1] for name in JOINT_ORDER_18D],
        dtype=torch.float32,
        device=device,
    )
    mean = torch.as_tensor(stats["mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(stats["std"], dtype=torch.float32, device=device)
    return (lower - mean) / std, (upper - mean) / std


def predict_contract_duration_sec(
    model,
    checkpoint: Mapping,
    condition,
    *,
    device="cpu",
) -> float:
    """Predict native expression-arc length and clamp only to train support."""
    duration = (checkpoint.get("v2_contracts") or {}).get("duration")
    if not isinstance(duration, Mapping):
        raise ValueError(
            "checkpoint has no learned variable-duration contract; pass frames "
            "explicitly only for a diagnostic override"
        )
    if duration.get("sha256") != _contract_sha256(duration):
        raise ValueError("checkpoint variable-duration contract hash is invalid")
    if duration.get("fixed_frame_count") is not None or duration.get(
        "fixed_duration_sec"
    ) is not None:
        raise ValueError("fixed-frame or fixed-duration inference is forbidden")
    if "fixed" in str(duration.get("trajectory_representation") or "").lower():
        raise ValueError("fixed-length trajectory representation is forbidden")
    bounds = duration.get("duration_supervision_sec") or duration.get(
        "train_duration_sec"
    )
    if not isinstance(bounds, Mapping):
        raise ValueError("variable-duration contract has no train-derived bounds")
    lower = float(bounds.get("min"))
    upper = float(bounds.get("max"))
    if not math.isfinite(lower) or not math.isfinite(upper) or lower <= 0.0 or upper < lower:
        raise ValueError("variable-duration contract bounds are invalid")
    tensor = torch.as_tensor(condition, dtype=torch.float32, device=device)
    if tensor.shape != (KIMODO_V2_CONDITION_DIM,):
        raise ValueError(f"condition must be [{KIMODO_V2_CONDITION_DIM}]")
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            predicted = float(
                model.plan_condition(tensor[None])["duration_sec"][0].detach().cpu()
            )
    finally:
        model.train(was_training)
    if not math.isfinite(predicted):
        raise FloatingPointError("duration head returned a non-finite value")
    return max(lower, min(upper, predicted))


def predict_contract_frame_count(
    model,
    checkpoint: Mapping,
    condition,
    *,
    fps=30.0,
    device="cpu",
) -> tuple[int, float]:
    predicted = predict_contract_duration_sec(
        model, checkpoint, condition, device=device
    )
    return sample_span_to_frame_count(predicted, fps), predicted


def sample_contract_trajectory(
    model,
    condition,
    *,
    frames,
    steps=24,
    seed=7,
    device="cpu",
) -> np.ndarray:
    action_dim = int(model.action_dim)
    joint_order = joint_order_for_action_dim(action_dim)
    frames = int(frames)
    steps = int(steps)
    if frames < 2 or steps < 1:
        raise ValueError("sampling requires at least two frames and one flow step")
    condition = torch.as_tensor(condition, dtype=torch.float32, device=device)
    if condition.shape != (KIMODO_V2_CONDITION_DIM,):
        raise ValueError(f"condition must be [{KIMODO_V2_CONDITION_DIM}]")
    stats = model.action_stats
    if action_dim == ACTION_DIM:
        lower, upper = _joint_bounds(stats, device=device)
    else:
        lower18, upper18 = _joint_bounds(
            {
                "mean": torch.cat([torch.as_tensor(stats["mean"]), torch.zeros(3)]),
                "std": torch.cat([torch.as_tensor(stats["std"]), torch.ones(3)]),
            },
            device=device,
        )
        lower, upper = lower18[:action_dim], upper18[:action_dim]
    generator = torch.Generator(device=device).manual_seed(int(seed))
    x = torch.randn((1, frames, action_dim), generator=generator, device=device)
    x = torch.maximum(torch.minimum(x, upper), lower)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for index in range(steps):
            t = torch.tensor([index / float(steps)], dtype=torch.float32, device=device)
            velocity = model(x, t, condition[None])
            x = torch.maximum(torch.minimum(x + velocity / float(steps), upper), lower)
    model.train(was_training)
    mean = torch.as_tensor(stats["mean"], device=device)
    std = torch.as_tensor(stats["std"], device=device)
    result = (x[0] * std + mean).cpu().numpy().astype(np.float32)
    if result.shape != (frames, len(joint_order)) or not np.isfinite(result).all():
        raise RuntimeError("generated trajectory violates checkpoint contract")
    return result


def write_contract_csv(path: str | Path, trajectory: np.ndarray, *, fps=30.0) -> None:
    if trajectory.ndim != 2:
        raise ValueError("trajectory must be [frames, actions]")
    if not np.isfinite(trajectory).all():
        raise ValueError("trajectory contains non-finite values")
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    joint_order = joint_order_for_action_dim(trajectory.shape[1])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_sec", *joint_order])
        for index, values in enumerate(trajectory):
            writer.writerow([f"{index / float(fps):.6f}", *[f"{float(v):.8f}" for v in values]])


def write_contract_npz(
    path: str | Path,
    trajectory: np.ndarray,
    *,
    fps=30.0,
    prompt="",
    checkpoint_path: str | Path | None = None,
) -> None:
    if trajectory.ndim != 2:
        raise ValueError("trajectory must be [frames, actions]")
    joint_order = joint_order_for_action_dim(trajectory.shape[1])
    if not np.isfinite(trajectory).all():
        raise ValueError("trajectory contains non-finite values")
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        trajectory=trajectory.astype(np.float32),
        joint_order=np.asarray(joint_order),
        action_dim=np.asarray(len(joint_order), dtype=np.int64),
        action_contract=np.asarray(
            CONTRACT_VERSION if len(joint_order) == ACTION_DIM else "legacy_ula_15d"
        ),
        fps=np.asarray(fps, dtype=np.float32),
        prompt=np.asarray(str(prompt)),
        checkpoint_sha256=np.asarray(sha256_file(checkpoint_path) if checkpoint_path else ""),
    )


__all__ = [
    "ACTION_DIM",
    "ADAPTER_POLICY",
    "ARTIFACT_KIND",
    "BODY_COMPATIBILITY_THRESHOLDS",
    "BODY_DISTILLATION_WEIGHT",
    "CONDITION_CACHE_SCHEMA_VERSION",
    "DEFAULT_BASE_CHECKPOINT",
    "DEFAULT_QWEN_CHECKPOINT",
    "FORMAL_ADJUDICATION_SCHEMA_VERSION",
    "FORMAL_MOTION_ADMISSION_REVIEW_GATES",
    "FORMAL_REQUIRED_18D_GATES",
    "FORMAL_REQUIRED_RELEASE_INVARIANTS",
    "FORMAL_REQUIRED_REVIEW_GATES",
    "FORMAL_SEMANTIC_SUPERVISION_MASKS",
    "HEAD_SLICE",
    "KIMODO_BEHAVIOR_FAMILY_SLICE",
    "KIMODO_BEHAVIOR_SLICE",
    "KIMODO_EMOTION_SLICE",
    "LEGACY_AFFECT_SLICE",
    "LEGACY_ACTION_DIM",
    "LEGACY_GESTURE_SLICE",
    "LEGACY_INTENT_SLICE",
    "MOTION_ONLY_EPISODE_CONTRACT",
    "MOTION_ONLY_CONDITION_CACHE_ARTIFACT_KIND",
    "MOTION_ONLY_CONDITION_CACHE_SCHEMA_VERSION",
    "MOTION_ONLY_DATA_ISOLATION_CONTRACT_TYPE",
    "MOTION_ONLY_NO_KIMODO_POLICY",
    "MOTION_ONLY_NO_QWEN_POLICY",
    "MOTION_ONLY_RANDOM_INIT_MODE",
    "MOTION_ONLY_RELEASE_REPORT_FILENAME",
    "MOTION_ONLY_REQUIRED_RELEASE_INVARIANTS",
    "MOTION_ONLY_STYLE_ONLY_CONDITION_POLICY",
    "STYLE_CONTROL_SLICE",
    "attach_condition_cache",
    "assess_body_compatibility",
    "benchmark_contract_inference",
    "benchmark_text_to_trajectory_inference",
    "body_sampling_drift_metrics",
    "build_condition_cache",
    "build_motion_only_condition_cache",
    "compute_18d_action_stats",
    "configure_head_adapter_policy",
    "evaluate_head_adapter",
    "instantiate_checkpoint_model",
    "legacy_forward_max_error",
    "load_18d_episodes",
    "load_condition_cache",
    "load_contract_checkpoint",
    "migrate_15d_checkpoint",
    "nonzero_head_forward_drift_metrics",
    "predict_contract_duration_sec",
    "predict_contract_frame_count",
    "read_joint_csv",
    "sample_contract_trajectory",
    "semantic_supervision_policy",
    "train_head_adapter",
    "training_data_provenance",
    "validate_condition_cache_for_generator",
    "validate_checkpoint_contract",
    "validate_motion_only_checkpoint_isolation",
    "validate_motion_only_style_condition",
    "verify_migrated_prefix",
    "validate_qwen_checkpoint_for_generator",
    "write_contract_csv",
    "write_contract_npz",
]
