#!/usr/bin/env python3
"""Initialize or train the formal native-length 18D ULA V2 baseline.

This entry point has no warm-start or unsafe mode.  It accepts only the complete
set of adjudicated manifests bound into a full-random checkpoint, and consumes a
condition cache cryptographically bound to that checkpoint.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.init_ula_v2_18d_random import (
    _atomic_json,
    _atomic_torch_save,
    load_formal_sources,
)
from tools.train_ula_v2_18d_posttrain import resolve_bound_motion_sources
from upper_body_skeleton.ula_v2_18d_head import (
    MOTION_ONLY_EPISODE_CONTRACT,
    build_condition_cache,
    sha256_file,
    validate_checkpoint_contract,
    validate_qwen_checkpoint_for_generator,
)
from upper_body_skeleton.ula_v2_18d_posttrain import (
    load_attached_beat_episodes,
    train_18d_posttrain,
)
from upper_body_skeleton.ula_v2_18d_random_init import (
    DEFAULT_LENGTH_BUCKETS,
    DEFAULT_SPLIT_FRACTIONS,
    RANDOM_INIT_MODE,
    build_random_18d_checkpoint,
)
from upper_body_skeleton.ula_v2_expression_turn_episode import (
    FORMAL_EPISODE_CONTRACT as EXPRESSION_TURN_V8_EPISODE_CONTRACT,
    build_expression_turn_v8_condition_cache,
    is_expression_turn_v8_episode,
    validate_expression_turn_v8_episode,
)


SCHEMA_VERSION = 1
ARTIFACT_KIND = "ula_v2_18d_formal_from_scratch_entry_v1"
FORMAL_SCOPE = "formal_variable_length_semantic_units"
FORMAL_TEMPORAL_POLICY = "full_semantic_unit_variable_length_30hz"
FORMAL_QWEN_POLICY = "frozen_bound_checkpoint_prompt_latent_masked_zero_v1"
EXPRESSION_TURN_V8_QWEN_POLICY = (
    "frozen_bound_checkpoint_blind_qualification_masked_prompt_latent_v1"
)
MOTION_ONLY_CONDITIONING_SCOPE = "motion_head_style_duration_only_v1"
MOTION_ONLY_OPTIMIZATION_TARGETS = [
    "motion_flow_18d",
    "head_3dof",
    "trajectory_style",
    "native_duration",
]
MOTION_ONLY_CONDITIONING_POLICY = {
    "scope": MOTION_ONLY_CONDITIONING_SCOPE,
    "qwen_prompt_latent": "masked_zero",
    "official_category": "metadata_split_evaluation_only",
    "communicative_intent": "masked_zero",
    "legacy_gesture": "masked_zero",
    "kimodo_behavior": "masked_zero",
    "kimodo_emotion": "masked_zero",
    "legacy_affect": "masked_zero",
    "trajectory_style": "enabled",
    "native_duration": "enabled",
}
EXPRESSION_TURN_V8_CONDITIONING_SCOPE = (
    "blind_reviewed_expression_turn_three_tier_conditioning_v1"
)
EXPRESSION_TURN_V8_OPTIMIZATION_TARGETS = [
    "motion_flow_18d",
    "head_3dof",
    "trajectory_style",
    "qualification_masked_observable_text",
    "qualification_masked_blind_affect",
    "native_duration",
]
EXPRESSION_TURN_V8_CONDITIONING_POLICY = {
    "scope": EXPRESSION_TURN_V8_CONDITIONING_SCOPE,
    "qwen_prompt_latent": "enabled_only_for_semantic_conditioning_qualification",
    "official_category": "forbidden",
    "communicative_intent": (
        "enabled_only_for_independent_blind_dyadic_qualification"
    ),
    "legacy_gesture": "masked_zero",
    "kimodo_behavior": "masked_zero",
    "kimodo_emotion": "enabled_only_for_blind_expressive_qualification",
    "legacy_affect": "masked_zero",
    "trajectory_style": "enabled",
    "native_duration": "enabled_full_expression_arc",
}
SOURCE_PROVENANCE_LOCK_KIND = "beat2_semantic_event_pilot_v7_provenance_lock"
MOTION_ONLY_PROVENANCE_LOCK_KIND = (
    "ula_v2_18d_motion_only_pretrain_provenance_lock_v1"
)
EXPRESSION_TURN_V8_PROVENANCE_LOCK_KIND = (
    "ula_v2_expression_turn_v8_multisource_provenance_lock_v1"
)
ACQUISITION_RECEIPT_KIND = "beat2_motion_only_acquisition"
USER_CONFIRMATION_RECEIPT_KIND = (
    "ula_noncommercial_research_user_confirmation_v1"
)
NONCOMMERCIAL_CONFIRMATION_TEXT = (
    "确认本次为非商业研究并接受上述数据条款"
)
NONCOMMERCIAL_CONFIRMATION_POLICY = (
    "noncommercial_research_user_confirmation_required_v1"
)
USER_OWNED_AUTHORIZATION_POLICY = "user_owned_explicit_authorization_v1"
NONCOMMERCIAL_RESEARCH_SCOPE = "non-commercial_research_only"
FORBIDDEN_CONFIG_KEYS = {
    "allow_unreviewed",
    "allow_unsafe_condition_cache",
    "allow_unsafe_training_data",
    "base_checkpoint",
    "base_15d_checkpoint",
    "generator_checkpoint",
    "initial_checkpoint",
    "kimodo_dataset_dir",
    "kimodo_split_checkpoint",
    "resume_from",
}
FORBIDDEN_FORMAL_TEMPORAL_KEYS = {
    "clip_frames",
    "crop_frames",
    "fixed_duration_sec",
    "fixed_frame_count",
    "fixed_window_sec",
    "frame_count",
    "frames",
    "max_duration_sec",
    "min_duration_sec",
    "phase_frame_choices",
    "phase_frames",
    "target_duration_sec",
    "window_frames",
}


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("formal from-scratch config must be a JSON object")
    return value


def _require_exact(mapping: dict, field: str, expected) -> None:
    if field in mapping and mapping[field] != expected:
        raise ValueError(f"formal config requires {field}={expected!r}")
    mapping[field] = expected


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_mapping_sha256(value: Mapping) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def resolve_formal_config(
    value: Mapping,
    *,
    manifests: Sequence[str | Path] = (),
    qwen_checkpoint: str | Path | None = None,
    condition_cache: str | Path | None = None,
) -> dict:
    """Resolve the immutable formal policy while allowing explicit path overrides."""
    config = deepcopy(dict(value))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("artifact_kind") != ARTIFACT_KIND:
        raise ValueError(f"artifact_kind must be {ARTIFACT_KIND!r}")
    forbidden = sorted(FORBIDDEN_CONFIG_KEYS.intersection(config))
    if forbidden:
        raise ValueError(
            "formal from-scratch entry refuses warm-start, replay, resume, and unsafe keys: "
            f"{forbidden}"
        )
    _require_exact(config, "initialization_mode", RANDOM_INIT_MODE)
    _require_exact(config, "audio_policy", "disabled_not_loaded")
    formal_episode_contract = config.get("formal_episode_contract")
    expression_turn_v8 = formal_episode_contract == EXPRESSION_TURN_V8_EPISODE_CONTRACT
    if formal_episode_contract not in (
        None,
        MOTION_ONLY_EPISODE_CONTRACT,
        EXPRESSION_TURN_V8_EPISODE_CONTRACT,
    ):
        raise ValueError(f"unsupported formal_episode_contract: {formal_episode_contract!r}")
    expected_qwen_policy = (
        EXPRESSION_TURN_V8_QWEN_POLICY if expression_turn_v8 else FORMAL_QWEN_POLICY
    )
    expected_conditioning_policy = (
        EXPRESSION_TURN_V8_CONDITIONING_POLICY
        if expression_turn_v8
        else MOTION_ONLY_CONDITIONING_POLICY
    )
    expected_optimization_targets = (
        EXPRESSION_TURN_V8_OPTIMIZATION_TARGETS
        if expression_turn_v8
        else MOTION_ONLY_OPTIMIZATION_TARGETS
    )
    _require_exact(config, "qwen_policy", expected_qwen_policy)
    conditioning_policy = deepcopy(dict(config.get("conditioning_policy") or {}))
    for field, expected in expected_conditioning_policy.items():
        _require_exact(conditioning_policy, field, expected)
    config["conditioning_policy"] = conditioning_policy
    if not str(config.get("output_dir") or "").strip():
        raise ValueError("output_dir is required")
    if not str(config.get("source_provenance_lock") or "").strip():
        raise ValueError("source_provenance_lock is required")
    if not _is_sha256(config.get("source_provenance_lock_sha256")):
        raise ValueError("source_provenance_lock_sha256 must be a lowercase SHA256")

    sources = config.get("motion_sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("motion_sources must be a non-empty list")
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise ValueError(f"motion_sources[{index}] must be an object")
        for field in (
            "dataset_source",
            "manifest",
            "source_inventory",
            "source_inventory_sha256",
            "provenance_lock_artifact_key",
            "speaker_namespace",
            "source_group_namespace",
        ):
            if not str(source.get(field) or "").strip():
                raise ValueError(f"motion_sources[{index}].{field} is required")
        if not _is_sha256(source["source_inventory_sha256"]):
            raise ValueError(
                f"motion_sources[{index}].source_inventory_sha256 must be a lowercase SHA256"
            )
        if not isinstance(source.get("license_gate"), Mapping):
            raise ValueError(f"motion_sources[{index}].license_gate is required")
    if manifests:
        requested = [str(Path(path).resolve()) for path in manifests]
        if len(requested) != len(sources):
            raise ValueError(
                "repeated --manifest values must provide the complete configured source set"
            )
        for source, manifest in zip(sources, requested, strict=True):
            source["manifest"] = manifest

    if qwen_checkpoint is not None:
        config["qwen_checkpoint"] = str(Path(qwen_checkpoint).resolve())
    if condition_cache is not None:
        config["condition_cache"] = str(Path(condition_cache).resolve())

    training = deepcopy(dict(config.get("training") or {}))
    fixed_temporal_controls = sorted(
        FORBIDDEN_FORMAL_TEMPORAL_KEYS.intersection(training)
    )
    if fixed_temporal_controls:
        raise ValueError(
            "formal native-length training refuses fixed/cropped temporal controls: "
            f"{fixed_temporal_controls}"
        )
    _require_exact(training, "training_scope", FORMAL_SCOPE)
    _require_exact(training, "formal_training_enabled", True)
    _require_exact(training, "temporal_unit_policy", FORMAL_TEMPORAL_POLICY)
    _require_exact(training, "training_policy", "full_network")
    _require_exact(training, "allow_unsafe_training_data", False)
    _require_exact(
        training,
        "optimization_targets",
        expected_optimization_targets,
    )
    batching = deepcopy(dict(training.get("batching") or {}))
    _require_exact(batching, "mode", "native_variable_length")
    buckets = batching.get("length_buckets", list(DEFAULT_LENGTH_BUCKETS))
    buckets = sorted({int(value) for value in buckets})
    if not buckets or buckets[0] < 3:
        raise ValueError("formal native batching requires length_buckets >= 3")
    batching["length_buckets"] = buckets
    _require_exact(batching, "homogeneous_bucket_batches", True)
    _require_exact(
        batching,
        "gradient_accumulation_mode",
        "dynamic_episode_weighted",
    )
    _require_exact(
        batching,
        "oversize_sequence_policy",
        "single_full_episode_or_fail",
    )
    for field, default in (
        ("max_motion_tokens_per_microbatch", 4096),
        ("max_attention_elements_per_microbatch", 8_000_000),
    ):
        value = batching.get(field, default)
        if isinstance(value, bool) or int(value) != value or int(value) <= 0:
            raise ValueError(f"formal batching.{field} must be a positive integer")
        batching[field] = int(value)
    target_effective_batch = batching.get(
        "target_effective_batch_size", training.get("batch_size")
    )
    if (
        isinstance(target_effective_batch, bool)
        or int(target_effective_batch) != target_effective_batch
        or int(target_effective_batch) <= 0
    ):
        raise ValueError(
            "formal batching.target_effective_batch_size must be a positive integer"
        )
    batching["target_effective_batch_size"] = int(target_effective_batch)
    training["batching"] = batching
    loss = deepcopy(dict(training.get("loss") or {}))
    if float(loss.get("body", 0.0)) != 0.0:
        raise ValueError("full-random formal training requires loss.body=0")
    if float(loss.get("planner_duration", 0.0)) <= 0.0:
        raise ValueError(
            "full-random formal training requires a positive planner_duration loss"
        )
    if float(loss.get("planner_transition", 0.0)) != 0.0:
        raise ValueError(
            "semantic-event-only formal data cannot train planner_transition"
        )
    loss["body"] = 0.0
    loss["planner_transition"] = 0.0
    training["loss"] = loss
    config["training"] = training
    return config


def read_formal_config(
    path: str | Path,
    *,
    manifests: Sequence[str | Path] = (),
    qwen_checkpoint: str | Path | None = None,
    condition_cache: str | Path | None = None,
) -> dict:
    return resolve_formal_config(
        _read_json(Path(path)),
        manifests=manifests,
        qwen_checkpoint=qwen_checkpoint,
        condition_cache=condition_cache,
    )


def formal_paths(config: Mapping) -> dict[str, Path]:
    root = Path(str(config["output_dir"])).resolve()
    initialization = root / "initialization"
    configured_cache = str(config.get("condition_cache") or "").strip()
    return {
        "root": root,
        "checkpoint": initialization / "random_init.pt",
        "split": initialization / "split_manifest.json",
        "report": initialization / "initialization_report.json",
        "condition_cache": (
            Path(configured_cache).resolve()
            if configured_cache
            else root / "conditioning" / "conditions.npz"
        ),
        "training": root / "training",
    }


def _required_file(config: Mapping, field: str) -> Path:
    value = str(config.get(field) or "").strip()
    if not value:
        raise ValueError(f"{field} must be supplied explicitly for this stage")
    path = Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{field} is missing: {path}")
    return path


def _validate_bound_qwen(config: Mapping) -> Path:
    path = _required_file(config, "qwen_checkpoint")
    expected = str(config.get("qwen_checkpoint_sha256") or "").strip()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("qwen_checkpoint_sha256 must be an explicit lowercase SHA256")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"Qwen checkpoint hash mismatch: expected {expected}, observed {actual}"
        )
    return path


def _source_license_readiness(config: Mapping) -> tuple[list[dict], list[str]]:
    """Audit per-source use authority independently from provenance integrity."""

    audits: list[dict] = []
    blockers: list[str] = []
    for index, source in enumerate(config.get("motion_sources") or []):
        prefix = f"motion_sources[{index}]"
        if not isinstance(source, Mapping):
            blockers.append(f"{prefix}.license_source_not_object")
            audits.append(
                {
                    "index": index,
                    "dataset_source": "",
                    "status": "blocked_invalid_source",
                }
            )
            continue
        source_name = str(source.get("dataset_source") or "").strip()
        gate = source.get("license_gate")
        if not isinstance(gate, Mapping):
            blockers.append(f"{prefix}.license_gate_missing")
            audits.append(
                {
                    "index": index,
                    "dataset_source": source_name,
                    "status": "blocked_missing_license_gate",
                }
            )
            continue
        policy = str(gate.get("policy") or "").strip()
        audit = {
            "index": index,
            "dataset_source": source_name,
            "policy": policy,
            "dataset_family": gate.get("dataset_family"),
            "terms_status": gate.get("terms_status"),
            "allowed_scope": gate.get("allowed_scope"),
            "metadata_statement": gate.get("metadata_statement"),
            "official_project_statement": gate.get(
                "official_project_statement"
            ),
            "ready": False,
        }
        if policy == NONCOMMERCIAL_CONFIRMATION_POLICY:
            if gate.get("allowed_scope") != NONCOMMERCIAL_RESEARCH_SCOPE:
                blockers.append(f"{prefix}.noncommercial_scope_invalid")
            family = str(gate.get("dataset_family") or "").strip().lower()
            if family == "beat2":
                if gate.get("terms_status") != "conflicting_upstream_statements":
                    blockers.append(f"{prefix}.beat2_license_conflict_not_recorded")
                if gate.get("metadata_statement") != "apache-2.0":
                    blockers.append(f"{prefix}.beat2_hf_metadata_statement_missing")
                if gate.get("official_project_statement") != "Non-commercial":
                    blockers.append(f"{prefix}.beat_official_terms_statement_missing")
                audit["terms_conflict"] = True
            confirmation = gate.get("user_confirmation")
            confirmed = bool(
                isinstance(confirmation, Mapping)
                and confirmation.get("confirmed") is True
                and confirmation.get("acknowledges_upstream_terms") is True
                and confirmation.get("authorized_scope")
                == NONCOMMERCIAL_RESEARCH_SCOPE
                and str(confirmation.get("confirmed_by") or "").strip()
                and str(confirmation.get("confirmed_at") or "").strip()
            )
            audit["user_confirmation_present"] = confirmed
            if not confirmed:
                blockers.append(
                    f"{prefix}.noncommercial_research_use_not_explicitly_confirmed"
                )
            audit["ready"] = confirmed and not any(
                blocker.startswith(f"{prefix}.") for blocker in blockers
            )
        elif policy == USER_OWNED_AUTHORIZATION_POLICY:
            authorization = gate.get("user_authorization")
            authorized = bool(
                isinstance(authorization, Mapping)
                and authorization.get("confirmed") is True
                and authorization.get("ownership_asserted") is True
                and str(gate.get("allowed_scope") or "").strip()
                and authorization.get("authorized_scope")
                == gate.get("allowed_scope")
                and str(authorization.get("confirmed_by") or "").strip()
                and str(authorization.get("confirmed_at") or "").strip()
            )
            audit["user_authorization_present"] = authorized
            if not authorized:
                blockers.append(f"{prefix}.user_owned_authorization_not_explicit")
            audit["ready"] = authorized
        else:
            blockers.append(f"{prefix}.license_policy_unsupported:{policy or '<empty>'}")
        audits.append(audit)
    return audits, blockers


def _license_bound_source_provenance(
    config: Mapping, provenance: Sequence[Mapping], license_audit: Sequence[Mapping]
) -> list[dict]:
    if not (
        len(config.get("motion_sources") or ())
        == len(provenance)
        == len(license_audit)
    ):
        raise ValueError("license/source provenance cardinality mismatch")
    result = []
    for source, record, audit in zip(
        config["motion_sources"], provenance, license_audit, strict=True
    ):
        gate = source["license_gate"]
        if audit.get("ready") is not True:
            raise ValueError("cannot bind a blocked source license gate")
        result.append(
            dict(record)
            | {
                "license_gate": deepcopy(dict(gate)),
                "license_gate_sha256": _canonical_mapping_sha256(gate),
                "license_audit": deepcopy(dict(audit)),
            }
        )
    return result


def _validate_checkpoint_license_binding(
    checkpoint: Mapping, config: Mapping
) -> None:
    records = (checkpoint.get("sources") or {}).get("motion_manifests") or ()
    sources = config.get("motion_sources") or ()
    if len(records) != len(sources):
        raise ValueError("checkpoint/source license binding count changed")
    for index, (record, source) in enumerate(zip(records, sources, strict=True)):
        gate = source.get("license_gate") if isinstance(source, Mapping) else None
        if not isinstance(record, Mapping) or not isinstance(gate, Mapping):
            raise ValueError(f"motion_sources[{index}] license binding is missing")
        if record.get("dataset_source") != source.get("dataset_source") or record.get(
            "license_gate_sha256"
        ) != _canonical_mapping_sha256(gate):
            raise ValueError(
                f"motion_sources[{index}] license gate differs from initialization"
            )


def _source_provenance_blockers(config: Mapping) -> list[str]:
    """Verify the pinned acquisition and source inventory before formal use."""

    blockers: list[str] = []
    expression_turn_v8 = (
        config.get("formal_episode_contract") == EXPRESSION_TURN_V8_EPISODE_CONTRACT
    )
    motion_only = (
        config.get("formal_episode_contract") == MOTION_ONLY_EPISODE_CONTRACT
    )
    lock_value = str(config.get("source_provenance_lock") or "").strip()
    if not lock_value:
        return ["source_provenance_lock_not_explicit"]
    lock_path = Path(lock_value).resolve()
    if not lock_path.is_file():
        return [f"source_provenance_lock_missing:{lock_path}"]
    expected_lock_hash = config.get("source_provenance_lock_sha256")
    if not _is_sha256(expected_lock_hash):
        blockers.append("source_provenance_lock_sha256_not_explicit")
    elif sha256_file(lock_path) != expected_lock_hash:
        blockers.append("source_provenance_lock_sha256_mismatch")
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        blockers.append("source_provenance_lock_invalid_json")
        return blockers
    if not isinstance(lock, Mapping):
        blockers.append("source_provenance_lock_not_object")
        return blockers
    expected_lock_kind = (
        EXPRESSION_TURN_V8_PROVENANCE_LOCK_KIND
        if expression_turn_v8
        else (
            MOTION_ONLY_PROVENANCE_LOCK_KIND
            if motion_only
            else SOURCE_PROVENANCE_LOCK_KIND
        )
    )
    if lock.get("artifact_kind") != expected_lock_kind:
        blockers.append("source_provenance_lock_kind_mismatch")
    if expression_turn_v8:
        if lock.get("formal_episode_contract") != EXPRESSION_TURN_V8_EPISODE_CONTRACT:
            blockers.append("source_provenance_lock_episode_contract_mismatch")
        if lock.get("duration_policy") != (
            "complete_expression_arc_variable_length_no_fixed_duration_v1"
        ):
            blockers.append("source_provenance_lock_duration_policy_invalid")
        if lock.get("source_count") != len(config.get("motion_sources") or ()):
            blockers.append("source_provenance_lock_source_count_mismatch")
        if lock.get("scale_gate_passed") is not True:
            blockers.append("source_provenance_lock_dataset_scale_gate_failed")
        if not isinstance(lock.get("dataset_scale"), Mapping):
            blockers.append("source_provenance_lock_dataset_scale_missing")
        if not isinstance(lock.get("minimum_training_scale"), Mapping):
            blockers.append("source_provenance_lock_minimum_training_scale_missing")
    elif motion_only:
        if lock.get("formal_episode_contract") != MOTION_ONLY_EPISODE_CONTRACT:
            blockers.append("source_provenance_lock_episode_contract_mismatch")
        if lock.get("duration_policy") != (
            "native_variable_length_physical_qc_no_fixed_duration_v1"
        ):
            blockers.append("source_provenance_lock_duration_policy_invalid")
        scale = lock.get("dataset_scale")
        if not isinstance(scale, Mapping) or scale.get("episode_count") != 12345:
            blockers.append("source_provenance_lock_dataset_scale_missing")
    if lock.get("accepted_for_training") is not True:
        blockers.append("source_provenance_lock_training_not_accepted")
    if lock.get("formal_release_allowed") is not True:
        blockers.append("source_provenance_lock_formal_release_not_allowed")
    license_gate = lock.get("license_gate")
    if not isinstance(license_gate, Mapping):
        blockers.append("source_provenance_lock_license_gate_missing")
    elif expression_turn_v8:
        if license_gate.get("authority_policy") != (
            "separate_per_source_license_gates_v1"
        ):
            blockers.append("source_provenance_lock_per_source_license_policy_missing")
        if license_gate.get("formal_release_blocked") is not False:
            blockers.append("source_provenance_lock_data_review_pending")
    else:
        if license_gate.get("training_authorized_by_this_lock") is not True:
            blockers.append("source_provenance_lock_training_not_authorized")
        if license_gate.get("formal_release_blocked") is not False:
            blockers.append("source_provenance_lock_license_review_pending")

    locked_artifacts = lock.get("locked_artifacts")
    if not isinstance(locked_artifacts, Mapping):
        blockers.append("source_provenance_lock_artifacts_missing")
        return blockers
    if expression_turn_v8 or motion_only:
        for key, artifact in locked_artifacts.items():
            if not isinstance(artifact, Mapping):
                blockers.append(f"source_provenance_lock_artifact_invalid:{key}")
                continue
            artifact_path_value = str(artifact.get("path") or "").strip()
            artifact_hash = artifact.get("sha256")
            if not artifact_path_value:
                blockers.append(f"source_provenance_lock_artifact_path_missing:{key}")
                continue
            artifact_path = Path(artifact_path_value).resolve()
            if not artifact_path.is_file():
                blockers.append(
                    f"source_provenance_lock_artifact_missing:{key}:{artifact_path}"
                )
            elif not _is_sha256(artifact_hash) or sha256_file(artifact_path) != artifact_hash:
                blockers.append(f"source_provenance_lock_artifact_sha256_mismatch:{key}")
    if not expression_turn_v8:
        acquisition = locked_artifacts.get("acquisition_receipt")
        if not isinstance(acquisition, Mapping):
            blockers.append("source_provenance_lock_acquisition_receipt_missing")
        else:
            receipt_path_value = str(acquisition.get("path") or "").strip()
            receipt_hash = acquisition.get("sha256")
            if not receipt_path_value:
                blockers.append("acquisition_receipt_path_missing")
            else:
                receipt_path = Path(receipt_path_value).resolve()
                if not receipt_path.is_file():
                    blockers.append(f"acquisition_receipt_missing:{receipt_path}")
                elif not _is_sha256(receipt_hash) or sha256_file(receipt_path) != receipt_hash:
                    blockers.append("acquisition_receipt_sha256_mismatch")
                else:
                    try:
                        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                    except (OSError, UnicodeError, json.JSONDecodeError):
                        blockers.append("acquisition_receipt_invalid_json")
                    else:
                        if not isinstance(receipt, Mapping):
                            blockers.append("acquisition_receipt_not_object")
                        else:
                            verification = receipt.get("verification")
                            required_verification = (
                                "all_selected_files_present",
                                "all_selected_sha256_available",
                                "all_selected_sha256_match",
                                "all_selected_sizes_match",
                            )
                            if receipt.get("artifact_kind") != ACQUISITION_RECEIPT_KIND:
                                blockers.append("acquisition_receipt_kind_mismatch")
                            if receipt.get("audio_policy") != "excluded_not_downloaded":
                                blockers.append("acquisition_receipt_audio_policy_invalid")
                            if not isinstance(verification, Mapping) or any(
                                verification.get(field) is not True
                                for field in required_verification
                            ):
                                blockers.append("acquisition_receipt_integrity_not_verified")
                            if (
                                not isinstance(verification, Mapping)
                                or verification.get("forbidden_audio_selected") is not False
                            ):
                                blockers.append("acquisition_receipt_forbidden_audio_selected")

    if motion_only:
        required_artifacts = {
            "acquisition_receipt",
            "training_pool_low_medium",
            "physical_qc_passed_manifest",
            "train_ready_manifest",
            "motion_only_release_report",
            "user_confirmation_receipt",
        }
        blockers.extend(
            f"source_provenance_lock_required_artifact_missing:{key}"
            for key in sorted(required_artifacts.difference(locked_artifacts))
        )
        confirmation_artifact = locked_artifacts.get("user_confirmation_receipt")
        if isinstance(confirmation_artifact, Mapping):
            confirmation_path_value = str(
                confirmation_artifact.get("path") or ""
            ).strip()
            confirmation_path = (
                Path(confirmation_path_value).resolve()
                if confirmation_path_value
                else None
            )
            try:
                confirmation = (
                    json.loads(confirmation_path.read_text(encoding="utf-8"))
                    if confirmation_path is not None
                    else None
                )
            except (OSError, UnicodeError, json.JSONDecodeError):
                confirmation = None
            if not isinstance(confirmation, Mapping):
                blockers.append("user_confirmation_receipt_invalid")
            else:
                exact = {
                    "artifact_kind": USER_CONFIRMATION_RECEIPT_KIND,
                    "confirmed": True,
                    "confirmation_text": NONCOMMERCIAL_CONFIRMATION_TEXT,
                    "authorized_scope": NONCOMMERCIAL_RESEARCH_SCOPE,
                    "acknowledges_upstream_terms": True,
                }
                if any(confirmation.get(key) != value for key, value in exact.items()):
                    blockers.append("user_confirmation_receipt_content_invalid")
                if not str(confirmation.get("confirmed_by") or "").strip() or not str(
                    confirmation.get("confirmed_at") or ""
                ).strip():
                    blockers.append("user_confirmation_receipt_identity_missing")
                for index, source in enumerate(config.get("motion_sources") or ()):
                    gate = source.get("license_gate") if isinstance(source, Mapping) else None
                    user_confirmation = (
                        gate.get("user_confirmation")
                        if isinstance(gate, Mapping)
                        else None
                    )
                    if (
                        isinstance(gate, Mapping)
                        and gate.get("policy") == NONCOMMERCIAL_CONFIRMATION_POLICY
                        and (
                            not isinstance(user_confirmation, Mapping)
                            or user_confirmation.get("confirmed_by")
                            != confirmation.get("confirmed_by")
                            or user_confirmation.get("confirmed_at")
                            != confirmation.get("confirmed_at")
                            or user_confirmation.get("authorized_scope")
                            != confirmation.get("authorized_scope")
                        )
                    ):
                        blockers.append(
                            f"motion_sources[{index}].confirmation_receipt_mismatch"
                        )

    for index, source in enumerate(config.get("motion_sources") or []):
        if not isinstance(source, Mapping):
            continue
        key = str(source.get("provenance_lock_artifact_key") or "").strip()
        locked = locked_artifacts.get(key) if key else None
        if not isinstance(locked, Mapping):
            blockers.append(f"motion_sources[{index}].provenance_artifact_missing:{key}")
            continue
        inventory_value = str(source.get("source_inventory") or "").strip()
        inventory_path = Path(inventory_value).resolve() if inventory_value else None
        locked_path_value = str(locked.get("path") or "").strip()
        locked_path = Path(locked_path_value).resolve() if locked_path_value else None
        expected_inventory_hash = source.get("source_inventory_sha256")
        if inventory_path is None or locked_path != inventory_path:
            blockers.append(f"motion_sources[{index}].source_inventory_path_mismatch")
        if (
            not _is_sha256(expected_inventory_hash)
            or locked.get("sha256") != expected_inventory_hash
        ):
            blockers.append(f"motion_sources[{index}].source_inventory_hash_not_locked")
        if inventory_path is None or not inventory_path.is_file():
            blockers.append(f"motion_sources[{index}].source_inventory_missing:{inventory_path}")
        elif (
            _is_sha256(expected_inventory_hash)
            and sha256_file(inventory_path) != expected_inventory_hash
        ):
            blockers.append(f"motion_sources[{index}].source_inventory_sha256_mismatch")
    return blockers


def audit_formal_readiness(config: Mapping, *, stage: str) -> dict:
    """Return a read-only readiness report; missing blind review remains blocking."""
    paths = formal_paths(config)
    blockers = _source_provenance_blockers(config)
    license_audit, license_blockers = _source_license_readiness(config)
    blockers.extend(license_blockers)
    for index, source in enumerate(config["motion_sources"]):
        manifest = Path(str(source["manifest"])).resolve()
        if not manifest.is_file():
            blockers.append(f"motion_sources[{index}].manifest_missing:{manifest}")
    qwen = str(config.get("qwen_checkpoint") or "").strip()
    if not qwen:
        blockers.append("qwen_checkpoint_not_explicit")
    elif not Path(qwen).resolve().is_file():
        blockers.append(f"qwen_checkpoint_missing:{Path(qwen).resolve()}")
    else:
        expected_qwen_hash = str(config.get("qwen_checkpoint_sha256") or "").strip()
        if len(expected_qwen_hash) != 64:
            blockers.append("qwen_checkpoint_sha256_not_explicit")
        elif sha256_file(Path(qwen).resolve()) != expected_qwen_hash:
            blockers.append("qwen_checkpoint_sha256_mismatch")
    if stage in {"cache", "train"}:
        if not paths["checkpoint"].is_file():
            blockers.append(f"random_init_checkpoint_missing:{paths['checkpoint']}")
    if stage in {"cache", "train"}:
        cache = str(config.get("condition_cache") or "").strip()
        if not cache:
            blockers.append("condition_cache_not_explicit")
        elif stage == "train" and not Path(cache).resolve().is_file():
            blockers.append(f"condition_cache_missing:{Path(cache).resolve()}")
    expression_turn_v8 = (
        config.get("formal_episode_contract") == EXPRESSION_TURN_V8_EPISODE_CONTRACT
    )
    return {
        "artifact_kind": ARTIFACT_KIND,
        "stage": stage,
        "ready": not blockers,
        "blockers": blockers,
        "initialization_mode": RANDOM_INIT_MODE,
        "training_scope": FORMAL_SCOPE,
        "temporal_unit_policy": FORMAL_TEMPORAL_POLICY,
        "batching_mode": "native_variable_length",
        "unsafe_training_data": False,
        "allow_unreviewed": False,
        "formal_episode_contract": config.get("formal_episode_contract"),
        "qwen_policy": (
            EXPRESSION_TURN_V8_QWEN_POLICY if expression_turn_v8 else FORMAL_QWEN_POLICY
        ),
        "conditioning_scope": (
            EXPRESSION_TURN_V8_CONDITIONING_SCOPE
            if expression_turn_v8
            else MOTION_ONLY_CONDITIONING_SCOPE
        ),
        "optimization_targets": list(
            EXPRESSION_TURN_V8_OPTIMIZATION_TARGETS
            if expression_turn_v8
            else MOTION_ONLY_OPTIMIZATION_TARGETS
        ),
        "license_audit": license_audit,
        "source_provenance_lock": str(
            Path(str(config.get("source_provenance_lock") or "")).resolve()
        ),
        "paths": {key: str(value) for key, value in paths.items()},
    }


def initialize_formal(config: Mapping) -> dict:
    readiness = audit_formal_readiness(config, stage="initialize")
    if not readiness["ready"]:
        raise ValueError("formal initialization is blocked: " + ", ".join(readiness["blockers"]))
    qwen_checkpoint = _validate_bound_qwen(config)
    paths = formal_paths(config)
    existing = [
        str(paths[name])
        for name in ("checkpoint", "split", "report")
        if paths[name].exists()
    ]
    if existing:
        raise FileExistsError(f"refusing to overwrite formal initialization outputs: {existing}")

    episodes, provenance = load_formal_sources(config)
    _validate_formal_episode_mode(config, episodes)
    provenance = _license_bound_source_provenance(
        config, provenance, readiness["license_audit"]
    )
    model = dict(config.get("model") or {})
    checkpoint, split, report = build_random_18d_checkpoint(
        episodes,
        qwen_checkpoint=qwen_checkpoint,
        source_provenance=provenance,
        seed=int(config.get("seed", 7)),
        fractions=dict(config.get("split_fractions") or DEFAULT_SPLIT_FRACTIONS),
        hidden_dim=int(model.get("hidden_dim", 384)),
        layers=int(model.get("layers", 6)),
        semantic_tokens=int(model.get("semantic_tokens", 7)),
        style_clip=float(config.get("style_clip", 5.0)),
        length_buckets=tuple(config["training"]["batching"]["length_buckets"]),
    )
    _atomic_torch_save(checkpoint, paths["checkpoint"])
    _atomic_json(split, paths["split"])
    report.update(
        {
            "formal_entry_artifact_kind": ARTIFACT_KIND,
            "checkpoint": str(paths["checkpoint"]),
            "checkpoint_sha256": sha256_file(paths["checkpoint"]),
            "split_manifest": str(paths["split"]),
            "sources": provenance,
            "license_audit": readiness["license_audit"],
            "training_command_executed": False,
        }
    )
    _atomic_json(report, paths["report"])
    return report


def _validate_motion_only_conditioning(episodes: Sequence[Mapping]) -> None:
    for row in episodes:
        clip_id = str(row.get("clip_id") or "<unknown>")
        if row.get("behavior_supervision_mask") is not False:
            raise ValueError(
                f"{clip_id}: motion-only experiment requires behavior conditioning masked"
            )
        if row.get("emotion_conditioning_mask") is not False:
            raise ValueError(
                f"{clip_id}: motion-only experiment requires emotion conditioning masked"
            )
        masks = row.get("semantic_supervision_masks")
        if masks is not None and (
            not isinstance(masks, Mapping)
            or any(
                masks.get(field) is not False
                for field in (
                    "official_category",
                    "robot_observable_motion_form",
                    "communicative_intent",
                    "prompt_text",
                    "legacy_gesture",
                )
            )
        ):
            raise ValueError(
                f"{clip_id}: motion-only experiment requires every semantic channel masked"
            )


def _validate_formal_episode_mode(
    config: Mapping,
    episodes: Sequence[Mapping],
    *,
    require_attached_condition: bool = False,
) -> None:
    expression_turn_v8 = (
        config.get("formal_episode_contract") == EXPRESSION_TURN_V8_EPISODE_CONTRACT
    )
    motion_only_physical_qc = (
        config.get("formal_episode_contract") == MOTION_ONLY_EPISODE_CONTRACT
    )
    flags = [is_expression_turn_v8_episode(episode) for episode in episodes]
    if expression_turn_v8:
        if not flags or not all(flags):
            raise ValueError("v8 formal entry requires only expression-turn v8 episodes")
        for episode in episodes:
            validate_expression_turn_v8_episode(
                episode,
                require_attached_condition=require_attached_condition,
            )
    else:
        if any(flags):
            raise ValueError(
                "expression-turn v8 episodes require the explicit formal_episode_contract"
            )
        contracts = {row.get("formal_episode_contract") for row in episodes}
        if motion_only_physical_qc:
            if contracts != {MOTION_ONLY_EPISODE_CONTRACT}:
                raise ValueError(
                    "motion-only formal entry requires only physical-QC motion-only episodes"
                )
        elif MOTION_ONLY_EPISODE_CONTRACT in contracts:
            raise ValueError(
                "physical-QC motion-only episodes require the explicit formal_episode_contract"
            )
        _validate_motion_only_conditioning(episodes)


def build_formal_condition_cache(config: Mapping) -> dict:
    readiness = audit_formal_readiness(config, stage="cache")
    if not readiness["ready"]:
        raise ValueError(
            "formal condition caching is blocked: " + ", ".join(readiness["blockers"])
        )
    qwen_checkpoint = _validate_bound_qwen(config)
    paths = formal_paths(config)
    output = paths["condition_cache"]
    metadata = output.with_suffix(output.suffix + ".json")
    if output.exists() or metadata.exists():
        raise FileExistsError(
            f"refusing to overwrite formal condition cache outputs: {[str(output), str(metadata)]}"
        )
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=True)
    validate_checkpoint_contract(checkpoint, expected_action_dim=18)
    _validate_checkpoint_license_binding(checkpoint, config)
    validate_qwen_checkpoint_for_generator(checkpoint, qwen_checkpoint)
    requested = [
        Path(str(source["manifest"])).resolve() for source in config["motion_sources"]
    ]
    resolve_bound_motion_sources(checkpoint, requested_manifests=requested)
    episodes, _ = load_formal_sources(config)
    _validate_formal_episode_mode(config, episodes)
    from upper_body_skeleton.ula_v2_18d_random_init import (
        validate_random_checkpoint_split,
    )

    validate_random_checkpoint_split(
        checkpoint,
        episodes,
        requested_fractions=dict(config.get("split_fractions") or DEFAULT_SPLIT_FRACTIONS),
    )
    if config.get("formal_episode_contract") == EXPRESSION_TURN_V8_EPISODE_CONTRACT:
        result = build_expression_turn_v8_condition_cache(
            episodes,
            qwen_checkpoint,
            output,
            base_checkpoint=paths["checkpoint"],
            device=str(config.get("condition_cache_device") or "auto"),
            local_files_only=True,
            batch_size=int(config.get("condition_cache_batch_size") or 16),
        )
    else:
        result = build_condition_cache(
            episodes,
            qwen_checkpoint,
            output,
            base_checkpoint=paths["checkpoint"],
            device=str(config.get("condition_cache_device") or "auto"),
            local_files_only=True,
            batch_size=int(config.get("condition_cache_batch_size") or 16),
        )
    return dict(result) | {
        "condition_cache": str(output),
        "condition_cache_metadata": str(metadata),
        "training_command_executed": False,
    }


def _load_attached_sources(config: Mapping, checkpoint: Mapping) -> list[dict]:
    requested = [Path(str(source["manifest"])).resolve() for source in config["motion_sources"]]
    bindings = resolve_bound_motion_sources(checkpoint, requested_manifests=requested)
    cache = _required_file(config, "condition_cache")
    episodes = []
    seen = set()
    for binding in bindings:
        loaded = load_attached_beat_episodes(
            binding["manifest"],
            cache,
            allow_unreviewed=False,
            allow_unsafe_condition_cache=False,
            dataset_source=binding["dataset_source"],
            speaker_namespace=binding["speaker_namespace"],
            source_group_namespace=binding["source_group_namespace"],
        )
        _validate_formal_episode_mode(
            config,
            loaded,
            require_attached_condition=True,
        )
        duplicates = seen.intersection(str(row["clip_id"]) for row in loaded)
        if duplicates:
            raise ValueError(f"formal manifests contain duplicate clip IDs: {sorted(duplicates)[:5]}")
        seen.update(str(row["clip_id"]) for row in loaded)
        episodes.extend(loaded)
    if not episodes:
        raise ValueError("formal manifests contain no adjudicated train-ready episodes")
    return episodes


def train_formal(config: Mapping) -> dict:
    readiness = audit_formal_readiness(config, stage="train")
    if not readiness["ready"]:
        raise ValueError("formal training is blocked: " + ", ".join(readiness["blockers"]))
    paths = formal_paths(config)
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=True)
    validate_checkpoint_contract(checkpoint, expected_action_dim=18)
    if (checkpoint.get("random_initialization") or {}).get("mode") != RANDOM_INIT_MODE:
        raise ValueError("formal entry accepts only its full-random 18D initialization")
    _validate_checkpoint_license_binding(checkpoint, config)
    qwen_checkpoint = _validate_bound_qwen(config)
    validate_qwen_checkpoint_for_generator(checkpoint, qwen_checkpoint)
    episodes = _load_attached_sources(config, checkpoint)
    training_config = dict(config["training"])
    training_config["split_fractions"] = dict(config["split_fractions"])
    return train_18d_posttrain(
        initial_checkpoint_path=paths["checkpoint"],
        beat_episodes=episodes,
        output_dir=paths["training"],
        kimodo_replay_episodes=(),
        replay_provenance={},
        config=training_config,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("audit", "initialize", "cache", "train"), default="audit"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        default=[],
        help="Repeat to override the complete configured manifest set in source order.",
    )
    parser.add_argument("--qwen-checkpoint", type=Path)
    parser.add_argument("--condition-cache", type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = read_formal_config(
        args.config,
        manifests=args.manifest,
        qwen_checkpoint=args.qwen_checkpoint,
        condition_cache=args.condition_cache,
    )
    if args.stage == "audit":
        result = audit_formal_readiness(config, stage="train")
    elif args.stage == "initialize":
        result = initialize_formal(config)
    elif args.stage == "cache":
        result = build_formal_condition_cache(config)
    else:
        result = train_formal(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
