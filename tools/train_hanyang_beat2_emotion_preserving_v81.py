#!/usr/bin/env python3
"""Fail-closed BEAT2 V7 + strict-Hanyang emotion-preserving V8.1 entry.

V8.1 is deliberately not a continuation of the old motion-only V8 loop.
BEAT2 keeps the complete V7 condition and loss path.  Hanyang contributes only
confidence-masked partial-motion loss and receives a reserved domain marker,
never the all-zero BEAT2 CFG-null condition.  A frozen copy of the exact V7
foundation anchors the BEAT2 conditioned response.

Persistent execution is blocked until a hash-bound approval receipt exists.
This module contains the validation and loss-path contracts used by the future
paired-arm runner; importing it and running ``--preflight`` cannot start a
training process.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from datetime import datetime
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import train_beat2_emotion_hierarchy_v7 as v7  # noqa: E402
from tools import train_beat2_style_emotion_v2 as v7_engine  # noqa: E402
from tools import (  # noqa: E402
    select_beat2_emotion_hierarchy_v7_qwen_winner as winner_selector,
)
from upper_body_skeleton.hanyang_emotion_retarget import (  # noqa: E402
    sha256_file,
)
from upper_body_skeleton.hanyang_expanded_generator import (  # noqa: E402
    collate_confidence_weighted_18d,
    confidence_weighted_18d_objective,
    load_hanyang_partial_motion_episodes,
)
from upper_body_skeleton.ula_v2_18d_head import (  # noqa: E402
    ACTION_DIM,
)
from upper_body_skeleton.retarget_v2_18d import (  # noqa: E402
    JOINT_ORDER_18D,
)
from upper_body_skeleton.ula_training import (  # noqa: E402
    ULA_MMDIT_V3_ADALN_ARCHITECTURE,
    create_ula_model,
)
from upper_body_skeleton.ula_v2_18d_posttrain import (  # noqa: E402
    ModelEMA,
    NativeLengthBucketSampler,
)
from upper_body_skeleton.ula_v2_18d_random_init import (  # noqa: E402
    forward_with_frame_mask,
)


SCHEMA_VERSION = "8.1"
CONFIG_ARTIFACT_KIND = (
    "hanyang_beat2_emotion_preserving_generator_config_v8_1"
)
PLAN_ARTIFACT_KIND = (
    "hanyang_beat2_emotion_preserving_paired_plan_v8_1"
)
APPROVAL_ARTIFACT_KIND = (
    "hanyang_beat2_emotion_preserving_approval_v8_1"
)
CHECKPOINT_ARTIFACT_KIND = (
    "hanyang_beat2_emotion_preserving_generator_v8_1"
)
STATE_ARTIFACT_KIND = (
    "hanyang_beat2_emotion_preserving_exact_resume_state_v8_1"
)
SUMMARY_ARTIFACT_KIND = (
    "hanyang_beat2_emotion_preserving_summary_v8_1"
)
FOUNDATION_SHA256 = (
    "b18f7cf1050144b9870d85047d39c50e123d495e5f0be65079bc3d0b50db2b3b"
)
FOUNDATION_RELATIVE_PATH = (
    "training/runs/beat2_emotion_hierarchy_v7/"
    "generator_emotion_hierarchy_v7.pt"
)
V7_CONDITION_POLICY = (
    "zero_base_0_133_qwen_predicted_style_133_136_"
    "frozen_qwen_text_136_264_no_trajectory_oracle_v2"
)
V81_TRAINING_POLICY = (
    "beat2_v7_complete_emotion_hierarchy_plus_hanyang_source344_minus_"
    "unapproved_boundary21_safe323_domain_isolated_masked_partial_motion_"
    "response_anchor_v8_1"
)
WINNER_ARM_LAUNCH_POLICY = (
    "same_winner_independent_initialization_supervisor_sequential_execution_"
    "no_cross_arm_warm_start_v1"
)
V7_DIAGNOSTIC_SEED_OFFSET = 1_000_003
V7_DIAGNOSTIC_SEED_POLICY = "exact_v7_root_seed_plus_1000003_v1"
HANYANG_SOURCE_POOL_COUNT = 344
HANYANG_BOUNDARY_COUNT = 21
HANYANG_TRAINING_ELIGIBLE_COUNT = 323
HANYANG_SOURCE_SPLIT_COUNTS = {
    "train": 291,
    "validation": 10,
    "test": 43,
}
HANYANG_BOUNDARY_SPLIT_COUNTS = {
    "train": 10,
    "validation": 5,
    "test": 6,
}
HANYANG_TRAINING_ELIGIBLE_SPLIT_COUNTS = {
    "train": 281,
    "validation": 5,
    "test": 37,
}
HANYANG_BOUNDARY_REVIEW_MANIFEST_SHA256 = (
    "aafee677a3394102dbea5d35fb3f6c8b6c86ac5dd4f6ac7faccac75b9bce7c3a"
)
HANYANG_SAFE_CLIP_IDS_SHA256 = (
    "b4ed7a2235c06d0712eb6905c6c0b87a346a47a5238c976b0e121eb5c0354474"
)
HANYANG_EXCLUDED_CLIP_IDS_SHA256 = (
    "43fa4b98941912548b84b11aefdf7602a53cfa3b1c45eb07df8c5fd2e6bc830c"
)
V7_RUNTIME_MODEL_CONFIG = {
    "architecture": ULA_MMDIT_V3_ADALN_ARCHITECTURE,
    "hidden_dim": 384,
    "layers": 6,
    "semantic_tokens": 7,
    "seed": 7,
    "action_dim": ACTION_DIM,
    "initialization_mode": "full_generator_random_no_qwen_no_kimodo_v1",
    "fixed_window_training": False,
    "formal_episode_contract": "ula_v2_18d_motion_only_physical_qc_v1",
    "checkpoint_step": 267000,
}
HANYANG_DOMAIN_MARKER_INDEX = 0
HANYANG_DOMAIN_MARKER_VALUE = 1.0
BEAT2_STYLE_SLICE = slice(133, 136)
BEAT2_QWEN_SLICE = slice(136, 264)
BEAT2_RESERVED_BASE_SLICE = slice(0, 133)
HUMAN_APPROVAL_BLOCKED = "HUMAN_APPROVAL_BLOCKED_V8_1"
REQUIRED_BEAT2_TERMS = frozenset(
    {
        "flow",
        "condition_ranking",
        "condition_response_floor",
        "hierarchy_binary_loss",
        "hierarchy_emotion_loss",
        "hierarchy_group_auxiliary_loss",
        "emotion_response_anchor",
    }
)
HANYANG_MOTION_PARAMETER_PREFIXES = (
    "input.",
    "time_mlp.",
)
HANYANG_MOTION_PARAMETER_FRAGMENTS = (
    ".attn.",
    ".ffn.",
)
HANYANG_MOTION_PARAMETER_EXACT = frozenset(
    {"output.weight", "output.bias"}
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _resolve_file(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _resolve_output(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path")
    return Path(value).expanduser().resolve()


def _verify_file(
    values: dict[str, Any],
    *,
    path_field: str,
    sha_field: str,
) -> None:
    path = _resolve_file(values.get(path_field), field=path_field)
    expected = values.get(sha_field)
    if not _is_sha256(expected) or sha256_file(path) != expected:
        raise ValueError(f"{path_field} SHA256 mismatch")
    values[path_field] = str(path)


def _read_jsonl_mappings(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}:{row_number}: row must be an object")
        rows.append(dict(value))
    return rows


def _validate_hanyang_boundary_exclusion(
    values: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the bound boundary manifest and derive the exact safe ID set."""

    expected_review_sha = values.get(
        "expected_hanyang_boundary_review_manifest_sha256"
    )
    expected_safe_sha = values.get("expected_hanyang_safe_clip_ids_sha256")
    expected_excluded_sha = values.get(
        "expected_hanyang_excluded_clip_ids_sha256"
    )
    if (
        expected_review_sha != HANYANG_BOUNDARY_REVIEW_MANIFEST_SHA256
        or expected_safe_sha != HANYANG_SAFE_CLIP_IDS_SHA256
        or expected_excluded_sha != HANYANG_EXCLUDED_CLIP_IDS_SHA256
    ):
        raise ValueError("Hanyang boundary exclusion hashes changed")

    source_manifest = _resolve_file(
        values.get("hanyang_strict_manifest"),
        field="hanyang_strict_manifest",
    )
    pool_receipt_path = _resolve_file(
        values.get("hanyang_pool_receipt"),
        field="hanyang_pool_receipt",
    )
    pool_receipt = json.loads(
        pool_receipt_path.read_text(encoding="utf-8")
    )
    if not isinstance(pool_receipt, Mapping):
        raise ValueError("Hanyang pool receipt must be an object")
    receipt_record = {
        key: value
        for key, value in pool_receipt.items()
        if key != "record_sha256"
    }
    review_bundle = pool_receipt.get("human_review_bundle")
    if (
        pool_receipt.get("record_sha256")
        != canonical_sha256(receipt_record)
        or pool_receipt.get("pool_row_count") != HANYANG_SOURCE_POOL_COUNT
        or pool_receipt.get("human_review_approved") is not False
        or pool_receipt.get("training_launch_allowed") is not False
        or not isinstance(review_bundle, Mapping)
        or review_bundle.get("sample_count") != HANYANG_BOUNDARY_COUNT
        or review_bundle.get("human_review_evidence_complete") is not False
        or review_bundle.get("review_status") != "HUMAN_REVIEW_BLOCKED"
        or review_bundle.get("manifest_sha256") != expected_review_sha
    ):
        raise ValueError("Hanyang pool/review boundary contract changed")
    review_manifest = _resolve_file(
        review_bundle.get("manifest"),
        field="hanyang_pool_receipt.human_review_bundle.manifest",
    )
    expected_review_manifest = (
        pool_receipt_path.parent
        / "human_review_bundle"
        / "manifest.jsonl"
    ).resolve()
    if (
        review_manifest != expected_review_manifest
        or sha256_file(review_manifest) != expected_review_sha
    ):
        raise ValueError("Hanyang boundary review manifest binding changed")

    source_rows = _read_jsonl_mappings(source_manifest)
    review_rows = _read_jsonl_mappings(review_manifest)
    if (
        len(source_rows) != HANYANG_SOURCE_POOL_COUNT
        or len(review_rows) != HANYANG_BOUNDARY_COUNT
    ):
        raise ValueError("Hanyang source/boundary row count changed")
    source_by_id: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        clip_id = str(row.get("clip_id") or "")
        if not clip_id.startswith("hanyang:") or clip_id in source_by_id:
            raise ValueError("Hanyang source clip IDs are invalid or duplicated")
        source_by_id[clip_id] = row
    if dict(
        Counter(
            str(row.get("fixed_split_assignment"))
            for row in source_rows
        )
    ) != HANYANG_SOURCE_SPLIT_COUNTS:
        raise ValueError("Hanyang source split counts changed")

    excluded_by_id: dict[str, dict[str, Any]] = {}
    for row in review_rows:
        clip_id = str(row.get("clip_id") or "")
        source = source_by_id.get(clip_id)
        record = {
            key: value
            for key, value in row.items()
            if key != "record_sha256"
        }
        if (
            source is None
            or clip_id in excluded_by_id
            or row.get("record_sha256") != canonical_sha256(record)
            or row.get("artifact_kind")
            != "hanyang_v8_source_faithful_human_review_row_v1"
            or row.get("human_review_approved") is not False
            or row.get("training_launch_allowed") is not False
            or row.get("review_status") != "pending"
            or row.get("training_lane")
            != "motion_only_pending_human_approval"
            or row.get("kimodo_accessed_or_used") is not False
            or row.get("fixed_split_assignment")
            != source.get("fixed_split_assignment")
        ):
            raise ValueError(
                f"{clip_id or '<missing>'}: unsafe Hanyang boundary row"
            )
        excluded_by_id[clip_id] = row

    excluded_clip_ids = sorted(excluded_by_id)
    safe_clip_ids = sorted(set(source_by_id) - set(excluded_by_id))
    excluded_split_counts = dict(
        Counter(
            str(row["fixed_split_assignment"])
            for row in excluded_by_id.values()
        )
    )
    safe_split_counts = dict(
        Counter(
            str(source_by_id[clip_id]["fixed_split_assignment"])
            for clip_id in safe_clip_ids
        )
    )
    if (
        len(excluded_clip_ids) != HANYANG_BOUNDARY_COUNT
        or len(safe_clip_ids) != HANYANG_TRAINING_ELIGIBLE_COUNT
        or excluded_split_counts != HANYANG_BOUNDARY_SPLIT_COUNTS
        or safe_split_counts != HANYANG_TRAINING_ELIGIBLE_SPLIT_COUNTS
        or canonical_sha256(excluded_clip_ids) != expected_excluded_sha
        or canonical_sha256(safe_clip_ids) != expected_safe_sha
    ):
        raise ValueError("Hanyang safe/excluded ID contract changed")
    return {
        "review_manifest": str(review_manifest),
        "review_manifest_sha256": expected_review_sha,
        "source_clip_ids": sorted(source_by_id),
        "excluded_clip_ids": excluded_clip_ids,
        "safe_clip_ids": safe_clip_ids,
        "source_split_counts": dict(HANYANG_SOURCE_SPLIT_COUNTS),
        "excluded_split_counts": dict(HANYANG_BOUNDARY_SPLIT_COUNTS),
        "safe_split_counts": dict(
            HANYANG_TRAINING_ELIGIBLE_SPLIT_COUNTS
        ),
        "excluded_clip_ids_sha256": expected_excluded_sha,
        "safe_clip_ids_sha256": expected_safe_sha,
    }


def _hanyang_artifact_audit(
    values: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the compact immutable Hanyang audit embedded in every artifact."""

    return {
        "strict_hanyang_count": HANYANG_SOURCE_POOL_COUNT,
        "hanyang_strict_count": HANYANG_SOURCE_POOL_COUNT,
        "hanyang_source_pool_count": HANYANG_SOURCE_POOL_COUNT,
        "hanyang_source_manifest_sha256": values[
            "expected_hanyang_strict_manifest_sha256"
        ],
        "hanyang_pool_receipt_sha256": values[
            "expected_hanyang_pool_receipt_sha256"
        ],
        "hanyang_source_split_counts": dict(HANYANG_SOURCE_SPLIT_COUNTS),
        "hanyang_boundary_candidate_count": HANYANG_BOUNDARY_COUNT,
        "hanyang_boundary_manifest_sha256": (
            HANYANG_BOUNDARY_REVIEW_MANIFEST_SHA256
        ),
        "hanyang_boundary_split_counts": dict(
            HANYANG_BOUNDARY_SPLIT_COUNTS
        ),
        "hanyang_boundary_excluded_count": HANYANG_BOUNDARY_COUNT,
        "hanyang_excluded_clip_ids_sha256": (
            HANYANG_EXCLUDED_CLIP_IDS_SHA256
        ),
        "hanyang_boundary_admitted_count": 0,
        "boundary_hanyang_admitted_count": 0,
        "hanyang_training_eligible_count": (
            HANYANG_TRAINING_ELIGIBLE_COUNT
        ),
        "hanyang_training_eligible_split_counts": dict(
            HANYANG_TRAINING_ELIGIBLE_SPLIT_COUNTS
        ),
        "hanyang_safe_clip_ids_sha256": HANYANG_SAFE_CLIP_IDS_SHA256,
        "hanyang_condition_labels_masked": True,
        "kimodo_admitted_count": 0,
    }


def _reject_source_bearing_kimodo(value: object, *, field: str = "root") -> None:
    """Reject Kimodo in source-bearing values while allowing the deny-policy text."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in {"deny_policy", "kimodo_admitted_count"}:
                continue
            _reject_source_bearing_kimodo(
                child, field=f"{field}.{key_text}"
            )
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, child in enumerate(value):
            _reject_source_bearing_kimodo(
                child, field=f"{field}[{index}]"
            )
    elif isinstance(value, str) and "kimodo" in value.casefold():
        raise ValueError(f"{field} contains forbidden Kimodo lineage")


def _validate_v7_inheritance(
    values: Mapping[str, Any],
) -> dict[str, Any]:
    config_path = _resolve_file(
        values.get("v7_reference_config"), field="v7_reference_config"
    )
    expected = values.get("expected_v7_reference_config_sha256")
    if not _is_sha256(expected) or sha256_file(config_path) != expected:
        raise ValueError("V7 reference config SHA256 mismatch")
    config = v7.read_config(config_path)
    if (
        config["condition_policy"] != V7_CONDITION_POLICY
        or config["training_policy"] != v7.TRAINING_POLICY
        or config["emotion_hierarchy"]["enabled"] is not True
        or float(config["emotion_hierarchy"]["binary_weight"]) <= 0
        or float(config["emotion_hierarchy"]["emotion_weight"]) <= 0
        or float(config["emotion_hierarchy"]["group_auxiliary_weight"]) <= 0
        or config["semantic_perceptual"]["enabled"] is not True
        or float(config["training"]["condition_ranking_weight"]) <= 0
        or float(config["training"]["condition_response_floor_weight"]) <= 0
        or float(config["training"]["condition_response_floor"]) <= 0
    ):
        raise ValueError("V8.1 must inherit the complete V7 emotion path")
    return config


def _validate_approval(
    gate: Mapping[str, Any],
    *,
    config_without_approval_sha256: str,
    selected_foundation_sha256: str | None,
    hanyang_audit: Mapping[str, Any],
) -> dict[str, Any]:
    if set(gate) != {
        "required",
        "status",
        "approval_receipt",
        "expected_approval_receipt_sha256",
    }:
        raise ValueError("approval_gate fields changed")
    if gate.get("required") is not True:
        raise ValueError("V8.1 approval must remain required")
    status = gate.get("status")
    if status not in {"blocked", "approved"}:
        raise ValueError("approval_gate.status must be blocked or approved")
    if status == "blocked":
        if (
            gate.get("approval_receipt") is not None
            or gate.get("expected_approval_receipt_sha256") is not None
        ):
            raise ValueError("blocked gate cannot carry an approval receipt")
        return dict(gate)
    receipt_path = _resolve_file(
        gate.get("approval_receipt"), field="approval_gate.approval_receipt"
    )
    expected = gate.get("expected_approval_receipt_sha256")
    if not _is_sha256(expected) or sha256_file(receipt_path) != expected:
        raise ValueError("approval receipt SHA256 mismatch")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    record = {key: value for key, value in receipt.items() if key != "sha256"}
    try:
        approved_time = datetime.fromisoformat(
            str(receipt.get("approved_utc") or "").replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError("approval receipt UTC is invalid") from exc
    if (
        receipt.get("artifact_kind") != APPROVAL_ARTIFACT_KIND
        or receipt.get("decision") != "approved"
        or receipt.get("training_launch_allowed") is not True
        or receipt.get("config_without_approval_sha256")
        != config_without_approval_sha256
        or receipt.get("foundation_checkpoint_sha256")
        != selected_foundation_sha256
        or receipt.get("strict_hanyang_count") != 344
        or receipt.get("boundary_hanyang_admitted_count") != 0
        or receipt.get("kimodo_admitted_count") != 0
        or any(
            receipt.get(field) != expected
            for field, expected in hanyang_audit.items()
        )
        or not str(receipt.get("approved_by") or "").strip()
        or approved_time.tzinfo is None
        or receipt.get("sha256") != canonical_sha256(record)
    ):
        raise ValueError("approval receipt content changed")
    result = dict(gate)
    result["approval_receipt"] = str(receipt_path)
    return result


def config_without_approval_sha256(
    validated_config: Mapping[str, Any],
) -> str:
    """Return the exact approval binding used by the validator."""

    unsigned = deepcopy(dict(validated_config))
    unsigned["approval_gate"] = {
        "required": True,
        "status": "blocked",
        "approval_receipt": None,
        "expected_approval_receipt_sha256": None,
    }
    return canonical_sha256(unsigned)


def _validate_qwen_ab_selection(
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        "required",
        "status",
        "selection_receipt",
        "expected_selection_receipt_file_sha256",
    }
    derived_fields = {
        "winner_selected",
        "selected_qwen_variant",
        "selected_foundation_checkpoint",
        "selected_foundation_sha256",
        "selected_condition_cache",
        "selected_condition_cache_sha256",
        "selection_receipt_canonical_sha256",
        "invariant_contract_sha256",
    }
    if not fields.issubset(selection) or (
        set(selection) - fields - derived_fields
    ):
        raise ValueError("qwen_ab_selection_gate fields changed")
    values = {field: selection[field] for field in fields}
    if values["required"] is not True:
        raise ValueError("Qwen winner receipt must remain required")
    pending = (
        values["status"] == "blocked_no_selected_receipt"
        and values["selection_receipt"] is None
        and values["expected_selection_receipt_file_sha256"] is None
    )
    if pending:
        return {
            **values,
            "winner_selected": False,
            "selected_qwen_variant": None,
            "selected_foundation_checkpoint": None,
            "selected_foundation_sha256": None,
            "selected_condition_cache": None,
            "selected_condition_cache_sha256": None,
            "selection_receipt_canonical_sha256": None,
            "invariant_contract_sha256": None,
        }
    if (
        values["status"] != "selected_receipt_bound"
        or not _is_sha256(
            values["expected_selection_receipt_file_sha256"]
        )
    ):
        raise ValueError("Qwen A/B selected receipt is incomplete")
    receipt_path = _resolve_file(
        values["selection_receipt"],
        field="qwen_ab_selection_gate.selection_receipt",
    )
    if (
        sha256_file(receipt_path)
        != values["expected_selection_receipt_file_sha256"]
    ):
        raise ValueError("Qwen winner receipt file SHA256 mismatch")
    receipt = winner_selector.validate_selection_receipt(
        json.loads(receipt_path.read_text(encoding="utf-8")),
        require_selected=True,
    )
    if (
        receipt.get("artifact_kind")
        != winner_selector.RECEIPT_ARTIFACT_KIND
        or receipt.get("eligible_for_v8_1_binding") is not True
        or receipt.get("winner_selected") is not True
        or receipt.get("no_kimodo") is not True
        or receipt.get("no_hanyang") is not True
        or receipt.get("no_external_data") is not True
        or not _is_sha256(receipt.get("invariant_contract_sha256"))
    ):
        raise ValueError("Qwen winner receipt cannot bind V8.1")
    arms = receipt.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != {
        "frozen_base",
        "lora_finetuned",
    }:
        raise ValueError("winner receipt A/B arms changed")
    for variant, arm in arms.items():
        training = (arm.get("training_invariants") or {})
        if (
            arm.get("variant") != variant
            or arm.get("completed") is not True
            or (arm.get("anti_collapse_gate") or {}).get("passed") is not True
            or arm.get("no_kimodo") is not True
            or arm.get("no_hanyang") is not True
            or arm.get("no_external_data") is not True
            or (arm.get("summary") or {}).get("completed_steps") != 80000
            or training.get("steps") != 80000
            or training.get("shared_noise_and_flow_times") is not True
        ):
            raise ValueError("winner receipt arm is incomplete or contaminated")
        for artifact_name in ("summary", "checkpoint", "condition_cache"):
            artifact = arm.get(artifact_name)
            if not isinstance(artifact, Mapping):
                raise ValueError(
                    f"winner receipt {variant} {artifact_name} is missing"
                )
            artifact_path = _resolve_file(
                artifact.get("path"),
                field=f"winner_receipt.arms.{variant}.{artifact_name}.path",
            )
            artifact_sha = artifact.get(
                "file_sha256" if artifact_name == "summary" else "sha256"
            )
            if (
                not _is_sha256(artifact_sha)
                or sha256_file(artifact_path) != artifact_sha
            ):
                raise ValueError(
                    f"winner receipt {variant} {artifact_name} SHA256 changed"
                )
    frozen = arms["frozen_base"]
    lora = arms["lora_finetuned"]
    if (
        frozen["training_invariants"] != lora["training_invariants"]
        or frozen.get("foundation_origin_sha256")
        != lora.get("foundation_origin_sha256")
        or frozen.get("split_contract") != lora.get("split_contract")
        or (frozen.get("checkpoint") or {}).get("global_step")
        != (lora.get("checkpoint") or {}).get("global_step")
    ):
        raise ValueError("winner receipt A/B invariant values differ")
    shared = receipt.get("shared_invariant_contract")
    if (
        not isinstance(shared, Mapping)
        or not isinstance(shared.get("values"), Mapping)
        or shared.get("sha256")
        != winner_selector.canonical_sha256(shared["values"])
        or receipt["invariant_contract_sha256"] != shared["sha256"]
        or any(
            shared.get(field) is not True
            for field in (
                "same_foundation",
                "same_seed",
                "same_split",
                "same_steps",
                "same_noise_and_flow_time_implementation",
                "same_sampler",
                "same_loss_and_batching",
            )
        )
    ):
        raise ValueError("winner shared invariant contract changed")
    shared_values = shared["values"]
    for name, value in frozen["training_invariants"].items():
        if shared_values.get(name) != value:
            raise ValueError("winner shared training invariants changed")
    if (
        shared_values.get("foundation_origin_sha256")
        != frozen.get("foundation_origin_sha256")
        or shared_values.get("checkpoint_global_step")
        != (frozen.get("checkpoint") or {}).get("global_step")
        or shared_values.get("checkpoint_split_contract")
        != frozen.get("split_contract")
    ):
        raise ValueError("winner shared foundation/split binding changed")
    selected_variant = receipt["selected_variant"]
    if selected_variant not in {"frozen_base", "lora_finetuned"}:
        raise ValueError("winner receipt selected variant changed")
    checkpoint_record = receipt["selected_checkpoint"]
    cache_record = receipt["selected_condition_cache"]
    selected_arm = arms[selected_variant]
    if (
        checkpoint_record != selected_arm.get("checkpoint")
        or cache_record != selected_arm.get("condition_cache")
        or (receipt.get("comparison") or {}).get("winner")
        != selected_variant
    ):
        raise ValueError("selected winner artifacts do not match selected arm")
    selected_path = _resolve_file(
        checkpoint_record.get("path"),
        field="winner_receipt.selected_checkpoint.path",
    )
    selected_sha = checkpoint_record.get("sha256")
    selected_cache = _resolve_file(
        cache_record.get("path"),
        field="winner_receipt.selected_condition_cache.path",
    )
    selected_cache_sha = cache_record.get("sha256")
    if (
        not _is_sha256(selected_sha)
        or sha256_file(selected_path) != selected_sha
        or not _is_sha256(selected_cache_sha)
        or sha256_file(selected_cache) != selected_cache_sha
    ):
        raise ValueError("selected winner artifact SHA256 mismatch")
    if selected_variant == "frozen_base":
        expected_path = (PROJECT_ROOT / FOUNDATION_RELATIVE_PATH).resolve()
        if selected_path != expected_path or selected_sha != FOUNDATION_SHA256:
            raise ValueError("frozen winner is not completed frozen-A")
    checkpoint = torch.load(
        selected_path, map_location="cpu", weights_only=True
    )
    if (
        checkpoint.get("artifact_kind") != v7.CHECKPOINT_ARTIFACT_KIND
        or checkpoint.get("condition_policy") != V7_CONDITION_POLICY
        or checkpoint.get("architecture") != "ula_mmdit_v3_adaln"
        or checkpoint.get("no_kimodo") is not True
        or checkpoint.get("no_external_data") is not True
        or "qwen_style_head_state_dict" not in checkpoint
    ):
        raise ValueError("selected winner checkpoint contract changed")
    return {
        **values,
        "selection_receipt": str(receipt_path),
        "winner_selected": True,
        "selected_qwen_variant": selected_variant,
        "selected_foundation_checkpoint": str(selected_path),
        "selected_foundation_sha256": selected_sha,
        "selected_condition_cache": str(selected_cache),
        "selected_condition_cache_sha256": selected_cache_sha,
        "selection_receipt_canonical_sha256": receipt["sha256"],
        "invariant_contract_sha256": receipt[
            "invariant_contract_sha256"
        ],
    }


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate every V8.1 invariant before any data or CUDA initialization."""

    if not isinstance(config, Mapping):
        raise ValueError("V8.1 config must be a mapping")
    values = deepcopy(dict(config))
    exact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": CONFIG_ARTIFACT_KIND,
        "training_policy": V81_TRAINING_POLICY,
        "beat2_condition_policy": V7_CONDITION_POLICY,
        "hanyang_condition_policy": (
            "reserved_domain_marker_dim0_no_cfg_dropout_no_semantic_target_v1"
        ),
        "deny_policy": (
            "kimodo_permanent_hard_deny_raw_cache_normalizer_split_checkpoint_v1"
        ),
        "foundation_role": (
            "completed_frozen_a_candidate_only_not_selected_winner"
        ),
        "formal_release_eligible": False,
    }
    for field, expected in exact.items():
        if values.get(field) != expected:
            raise ValueError(f"{field} must be {expected!r}")

    _verify_file(
        values,
        path_field="foundation_checkpoint",
        sha_field="expected_foundation_checkpoint_sha256",
    )
    foundation = Path(values["foundation_checkpoint"])
    expected_foundation = (PROJECT_ROOT / FOUNDATION_RELATIVE_PATH).resolve()
    if (
        foundation != expected_foundation
        or values["expected_foundation_checkpoint_sha256"]
        != FOUNDATION_SHA256
    ):
        raise ValueError("V8.1 foundation is not the exact completed V7")
    checkpoint = torch.load(
        foundation, map_location="cpu", weights_only=True
    )
    if (
        checkpoint.get("artifact_kind") != v7.CHECKPOINT_ARTIFACT_KIND
        or checkpoint.get("condition_policy") != V7_CONDITION_POLICY
        or checkpoint.get("architecture") != "ula_mmdit_v3_adaln"
        or int(checkpoint.get("condition_dim", -1)) != 264
        or "qwen_style_head_state_dict" not in checkpoint
    ):
        raise ValueError("completed V7 checkpoint contract changed")

    v7_config = _validate_v7_inheritance(values)
    values["v7_reference_config"] = str(
        _resolve_file(
            values["v7_reference_config"], field="v7_reference_config"
        )
    )
    for path_field, sha_field in (
        ("hanyang_strict_manifest", "expected_hanyang_strict_manifest_sha256"),
        ("hanyang_pool_receipt", "expected_hanyang_pool_receipt_sha256"),
        ("hanyang_rejected_manifest", "expected_hanyang_rejected_manifest_sha256"),
    ):
        _verify_file(values, path_field=path_field, sha_field=sha_field)
    _validate_hanyang_boundary_exclusion(values)

    data = values.get("data_contract")
    expected_data_contract = {
        "beat2_source",
        "hanyang_source",
        "hanyang_strict_count",
        "hanyang_source_manifest_sha256",
        "hanyang_pool_receipt_sha256",
        "hanyang_source_split_counts",
        "hanyang_boundary_candidate_count",
        "hanyang_boundary_manifest_sha256",
        "hanyang_boundary_split_counts",
        "hanyang_boundary_excluded_count",
        "hanyang_boundary_admitted_count",
        "hanyang_excluded_clip_ids_sha256",
        "hanyang_training_eligible_count",
        "hanyang_training_eligible_split_counts",
        "hanyang_safe_clip_ids_sha256",
        "kimodo_admitted_count",
        "hanyang_supervision",
    }
    if not isinstance(data, Mapping) or set(data) != expected_data_contract:
        raise ValueError("data_contract fields changed")
    if (
        data.get("beat2_source")
        != "v7_reference_config_exact_train_split"
        or data.get("hanyang_source")
        != "strict_qc_source344_minus_unapproved_boundary21_v1"
        or data.get("hanyang_strict_count") != HANYANG_SOURCE_POOL_COUNT
        or data.get("hanyang_source_manifest_sha256")
        != values["expected_hanyang_strict_manifest_sha256"]
        or data.get("hanyang_pool_receipt_sha256")
        != values["expected_hanyang_pool_receipt_sha256"]
        or data.get("hanyang_source_split_counts")
        != HANYANG_SOURCE_SPLIT_COUNTS
        or data.get("hanyang_boundary_candidate_count")
        != HANYANG_BOUNDARY_COUNT
        or data.get("hanyang_boundary_manifest_sha256")
        != HANYANG_BOUNDARY_REVIEW_MANIFEST_SHA256
        or data.get("hanyang_boundary_split_counts")
        != HANYANG_BOUNDARY_SPLIT_COUNTS
        or data.get("hanyang_boundary_excluded_count")
        != HANYANG_BOUNDARY_COUNT
        or data.get("hanyang_boundary_admitted_count") != 0
        or data.get("hanyang_excluded_clip_ids_sha256")
        != HANYANG_EXCLUDED_CLIP_IDS_SHA256
        or data.get("hanyang_training_eligible_count")
        != HANYANG_TRAINING_ELIGIBLE_COUNT
        or data.get("hanyang_training_eligible_split_counts")
        != HANYANG_TRAINING_ELIGIBLE_SPLIT_COUNTS
        or data.get("hanyang_safe_clip_ids_sha256")
        != HANYANG_SAFE_CLIP_IDS_SHA256
        or data.get("kimodo_admitted_count") != 0
        or data.get("hanyang_supervision")
        != "confidence_masked_partial_motion_only"
    ):
        raise ValueError(
            "source344/excluded21/eligible323/Kimodo data contract changed"
        )

    diagnostics = values.get("diagnostics")
    if diagnostics != {
        "policy": (
            "fixed_v7_54group_validation_same_negative_noise_ema_v1"
        ),
        "validation_group_count": 54,
        "run_at_step_one": True,
        "interval_steps": 1000,
        "run_at_every_checkpoint": True,
        "minimum_response_retention": 0.9,
        "minimum_cross_response_retention": 0.9,
        "minimum_flow_gap_retention": 0.9,
        "minimum_q2_recall_retention": 0.9,
        "minimum_q6_recall_retention": 0.9,
        "minimum_global54_recall_retention": 0.9,
        "hanyang_validation_episode_count": 5,
        "best_policy": (
            "highest_min_emotion_retention_then_lowest_beat2_and_hanyang_loss"
        ),
    }:
        raise ValueError("emotion diagnostic/candidate gate changed")

    isolation = values.get("domain_isolation")
    if isolation != {
        "hanyang_domain_marker_index": HANYANG_DOMAIN_MARKER_INDEX,
        "hanyang_domain_marker_value": HANYANG_DOMAIN_MARKER_VALUE,
        "beat2_reserved_base_slice": [0, 133],
        "beat2_style_slice": [133, 136],
        "beat2_qwen_slice": [136, 264],
        "hanyang_cfg_dropout_allowed": False,
        "hanyang_semantic_loss_allowed": False,
        "hanyang_style_loss_allowed": False,
        "hanyang_emotion_loss_allowed": False,
        "beat2_trainable_parameter_policy": (
            "exact_v7_adaln_condition_path_plus_style_head_v1"
        ),
        "hanyang_trainable_parameter_policy": (
            "motion_backbone_only_no_condition_or_modulation_v1"
        ),
        "gradient_overlap_allowed": False,
    }:
        raise ValueError("domain isolation contract changed")

    training = values.get("training")
    required_training = {
        "steps",
        "effective_batch_size",
        "beat2_fraction",
        "hanyang_fraction",
        "hanyang_period_steps",
        "hanyang_active_steps_per_period",
        "hanyang_examples_per_active_step",
        "condition_ranking_weight",
        "condition_response_floor",
        "condition_response_floor_weight",
        "semantic_perceptual_outer_weight",
        "emotion_response_anchor_weight",
        "seed",
        "noise_seed",
        "split_contract",
        "hanyang_motion_lr",
        "checkpoint_interval",
        "log_interval",
        "ema_decay",
        "max_grad_norm",
    }
    if not isinstance(training, Mapping) or set(training) != required_training:
        raise ValueError("training fields changed")
    integer_fields = (
        "steps",
        "effective_batch_size",
        "hanyang_period_steps",
        "hanyang_active_steps_per_period",
        "hanyang_examples_per_active_step",
        "seed",
        "noise_seed",
        "checkpoint_interval",
        "log_interval",
    )
    if any(type(training[name]) is not int for name in integer_fields):
        raise ValueError("training integer fields must be integers")
    if (
        training["steps"] != 60000
        or training["effective_batch_size"] != 16
        or training["hanyang_period_steps"] != 5
        or training["hanyang_active_steps_per_period"] != 4
        or training["hanyang_examples_per_active_step"] != 1
        or training["split_contract"] != "v7_beat2_plus_fixed_hanyang_participant_v1"
        or training["checkpoint_interval"] <= 0
        or training["log_interval"] <= 0
        or training["checkpoint_interval"] > training["steps"]
        or training["log_interval"] > training["steps"]
    ):
        raise ValueError("V8.1 schedule contract changed")
    exposure = (
        training["hanyang_active_steps_per_period"]
        * training["hanyang_examples_per_active_step"]
        / (
            training["hanyang_period_steps"]
            * training["effective_batch_size"]
        )
    )
    if (
        not math.isclose(float(training["beat2_fraction"]), 0.95)
        or not math.isclose(float(training["hanyang_fraction"]), 0.05)
        or not math.isclose(exposure, 0.05)
    ):
        raise ValueError("V8.1 exposure must be exactly BEAT2 95/Hanyang 5")
    inherited = v7_config["training"]
    scalar_training_fields = (
        "beat2_fraction",
        "hanyang_fraction",
        "condition_ranking_weight",
        "condition_response_floor",
        "condition_response_floor_weight",
        "semantic_perceptual_outer_weight",
        "emotion_response_anchor_weight",
        "hanyang_motion_lr",
        "ema_decay",
        "max_grad_norm",
    )
    if any(
        isinstance(training[name], bool)
        or not isinstance(training[name], (int, float))
        or not math.isfinite(float(training[name]))
        for name in scalar_training_fields
    ):
        raise ValueError("training scalar fields must be finite")
    for new_name, inherited_name in (
        ("condition_ranking_weight", "condition_ranking_weight"),
        ("condition_response_floor", "condition_response_floor"),
        ("condition_response_floor_weight", "condition_response_floor_weight"),
    ):
        if float(training[new_name]) != float(inherited[inherited_name]):
            raise ValueError(f"{new_name} must equal V7")
    if (
        float(training["semantic_perceptual_outer_weight"])
        != float(v7_config["semantic_perceptual"]["outer_weight"])
        or float(training["emotion_response_anchor_weight"]) <= 0
        or not 0 < float(training["hanyang_motion_lr"]) <= float(
            inherited["lr"]
        )
        or float(training["ema_decay"]) != float(inherited["ema_decay"])
        or float(training["max_grad_norm"]) != float(
            inherited["max_grad_norm"]
        )
    ):
        raise ValueError("emotion preservation weights are invalid")

    selection_raw = values.get("qwen_ab_selection_gate")
    if not isinstance(selection_raw, Mapping):
        raise ValueError("qwen_ab_selection_gate must be a mapping")
    selection = _validate_qwen_ab_selection(selection_raw)
    values["qwen_ab_selection_gate"] = selection

    arms = values.get("winner_overlay_arms")
    if not isinstance(arms, Mapping) or set(arms) != {
        "launch_policy",
        "shared_selected_foundation_sha256",
        "shared_selected_qwen_variant",
        "shared_selected_condition_cache_sha256",
        "shared_seed",
        "shared_noise_seed",
        "shared_split_contract",
        "winner_control_0pct_hanyang",
        "winner_isolated_5pct_hanyang",
    }:
        raise ValueError("winner_overlay_arms fields changed")
    if (
        arms.get("launch_policy")
        != WINNER_ARM_LAUNCH_POLICY
        or arms.get("shared_seed") != training["seed"]
        or arms.get("shared_noise_seed") != training["noise_seed"]
        or arms.get("shared_split_contract") != training["split_contract"]
    ):
        raise ValueError("blocked winner-overlay invariants changed")
    selected_triplet = (
        selection["selected_foundation_sha256"],
        selection["selected_qwen_variant"],
        selection["selected_condition_cache_sha256"],
    )
    arm_triplet = (
        arms.get("shared_selected_foundation_sha256"),
        arms.get("shared_selected_qwen_variant"),
        arms.get("shared_selected_condition_cache_sha256"),
    )
    if arm_triplet != selected_triplet:
        raise ValueError("winner-overlay arms are not bound to one winner")
    outputs: set[Path] = set()
    for arm_name, expected_fractions in (
        ("winner_control_0pct_hanyang", (0.95, 0.0, 0.05)),
        ("winner_isolated_5pct_hanyang", (0.95, 0.05, 0.0)),
    ):
        arm = arms.get(arm_name)
        if not isinstance(arm, Mapping) or set(arm) != {
            "beat2_fraction",
            "hanyang_fraction",
            "matched_noop_fraction",
            "output_dir",
        }:
            raise ValueError(f"{arm_name} fields changed")
        actual_fractions = (
            float(arm.get("beat2_fraction", -1)),
            float(arm.get("hanyang_fraction", -1)),
            float(arm.get("matched_noop_fraction", -1)),
        )
        if actual_fractions != expected_fractions:
            raise ValueError(f"{arm_name} domain fractions changed")
        arm_copy = dict(arm)
        output = _resolve_output(
            arm["output_dir"],
            field=f"winner_overlay_arms.{arm_name}.output_dir",
        )
        if output in outputs:
            raise ValueError("winner-overlay arms must have distinct outputs")
        outputs.add(output)
        arm_copy["output_dir"] = str(output)
        arms = dict(arms)
        arms[arm_name] = arm_copy
    values["winner_overlay_arms"] = arms

    _reject_source_bearing_kimodo(values)
    gate = values.get("approval_gate")
    if not isinstance(gate, Mapping):
        raise ValueError("approval_gate must be a mapping")
    values["approval_gate"] = _validate_approval(
        gate,
        config_without_approval_sha256=config_without_approval_sha256(
            values
        ),
        selected_foundation_sha256=selection[
            "selected_foundation_sha256"
        ],
        hanyang_audit=_hanyang_artifact_audit(values),
    )
    return values


def read_config(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_config(value)


def hanyang_domain_conditions(
    batch_size: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the non-null Hanyang-only domain condition."""

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    conditions = torch.zeros(batch_size, 264, dtype=dtype, device=device)
    conditions[:, HANYANG_DOMAIN_MARKER_INDEX] = HANYANG_DOMAIN_MARKER_VALUE
    if (
        torch.count_nonzero(conditions[:, BEAT2_STYLE_SLICE]).item() != 0
        or torch.count_nonzero(conditions[:, BEAT2_QWEN_SLICE]).item() != 0
        or torch.count_nonzero(conditions).item() != batch_size
    ):
        raise RuntimeError("Hanyang domain condition isolation failed")
    return conditions


def assert_nonzero_beat2_conditions(conditions: torch.Tensor) -> None:
    """Reject null, Hanyang-domain, or legacy-style BEAT2 conditions."""

    if conditions.ndim != 2 or conditions.shape[1] != 264:
        raise ValueError("BEAT2 conditions must have shape [batch,264]")
    if torch.count_nonzero(conditions[:, BEAT2_RESERVED_BASE_SLICE]).item():
        raise ValueError("BEAT2 reserved base condition block must be zero")
    if bool(torch.any(torch.linalg.vector_norm(
        conditions[:, BEAT2_QWEN_SLICE], dim=1
    ) <= 0).item()):
        raise ValueError("every BEAT2 row requires a nonzero frozen-Qwen latent")


def condition_response_anchor_loss(
    student_aligned: torch.Tensor,
    student_zero: torch.Tensor,
    teacher_aligned: torch.Tensor,
    teacher_zero: torch.Tensor,
    observed: torch.Tensor,
) -> torch.Tensor:
    """Anchor the V7 condition response, not merely its unconditional output."""

    if not (
        student_aligned.shape
        == student_zero.shape
        == teacher_aligned.shape
        == teacher_zero.shape
        == observed.shape
    ):
        raise ValueError("emotion response anchor tensors must share one shape")
    if observed.dtype != torch.bool or not bool(torch.any(observed).item()):
        raise ValueError("emotion response anchor requires observed values")
    student_response = student_aligned - student_zero
    teacher_response = (
        teacher_aligned.detach() - teacher_zero.detach()
    )
    return (
        (student_response - teacher_response).square()[observed].mean()
    )


def hanyang_motion_parameter_name(name: str) -> bool:
    """Allow Hanyang gradients only into condition-independent motion tensors."""

    return (
        name in HANYANG_MOTION_PARAMETER_EXACT
        or name.startswith(HANYANG_MOTION_PARAMETER_PREFIXES)
        or (
            name.startswith("blocks.")
            and any(
                fragment in name
                for fragment in HANYANG_MOTION_PARAMETER_FRAGMENTS
            )
        )
    )


def parameter_gradient_routes(
    model: nn.Module,
) -> dict[str, tuple[str, ...]]:
    """Return disjoint BEAT2-condition and Hanyang-motion parameter routes."""

    beat2 = []
    hanyang = []
    overlap = []
    for name, _parameter in model.named_parameters():
        beat2_allowed = v7_engine._matches_trainable_policy(name)
        hanyang_allowed = hanyang_motion_parameter_name(name)
        if beat2_allowed:
            beat2.append(name)
        if hanyang_allowed:
            hanyang.append(name)
        if beat2_allowed and hanyang_allowed:
            overlap.append(name)
    if overlap:
        raise RuntimeError(
            f"BEAT2/Hanyang parameter routes overlap: {overlap}"
        )
    required_beat2_roots = {
        "motion_latent_condition",
        "condition_pool",
        "blocks",
        "output_modulation",
        "plan",
        "duration_head",
    }
    if {name.split(".", 1)[0] for name in beat2} != required_beat2_roots:
        raise RuntimeError("BEAT2 route no longer matches the complete V7 path")
    if not hanyang or not any(name.startswith("blocks.") for name in hanyang):
        raise RuntimeError("Hanyang motion-only parameter route is incomplete")
    return {
        "beat2_condition": tuple(beat2),
        "hanyang_motion": tuple(hanyang),
    }


def route_disjoint_gradients(
    *,
    model: nn.Module,
    style_head: nn.Module,
    beat2_loss: torch.Tensor,
    hanyang_loss: torch.Tensor | None,
) -> dict[str, tuple[str, ...]]:
    """Backpropagate each domain only through its explicit parameter allowlist.

    ``torch.autograd.grad`` is used instead of a shared ``loss.backward()`` so
    Hanyang can never update AdaLN modulation, Qwen/style projections, plan
    heads, or the style head.  BEAT2 remains the sole owner of those tensors.
    """

    routes = parameter_gradient_routes(model)
    named = dict(model.named_parameters())
    beat2_named = [
        (name, named[name]) for name in routes["beat2_condition"]
    ]
    beat2_named.extend(
        (f"style_head.{name}", parameter)
        for name, parameter in style_head.named_parameters()
    )
    beat2_parameters = [parameter for _name, parameter in beat2_named]
    hanyang_parameters = [
        named[name] for name in routes["hanyang_motion"]
    ]
    for parameter in list(model.parameters()) + list(style_head.parameters()):
        parameter.grad = None
    beat2_gradients = torch.autograd.grad(
        beat2_loss,
        beat2_parameters,
        retain_graph=hanyang_loss is not None,
        allow_unused=False,
    )
    for parameter, gradient in zip(beat2_parameters, beat2_gradients):
        parameter.grad = gradient
    if hanyang_loss is not None:
        hanyang_gradients = torch.autograd.grad(
            hanyang_loss,
            hanyang_parameters,
            allow_unused=False,
        )
        for parameter, gradient in zip(
            hanyang_parameters, hanyang_gradients
        ):
            if parameter.grad is not None:
                raise RuntimeError("domain gradient firewall overlap")
            parameter.grad = gradient
    return {
        **routes,
        "beat2_style_head": tuple(
            name for name, _parameter in beat2_named
            if name.startswith("style_head.")
        ),
    }


def hanyang_partial_motion_objective(
    model: torch.nn.Module,
    actions: torch.Tensor,
    observation_confidence: torch.Tensor,
    durations_sec: torch.Tensor,
    frame_valid_mask: torch.Tensor,
    *,
    loss_weights: Mapping[str, float],
    noise: torch.Tensor | None = None,
    flow_times: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Run Hanyang only through its domain-isolated masked-motion lane."""

    conditions = hanyang_domain_conditions(
        actions.shape[0], device=actions.device, dtype=actions.dtype
    )
    return confidence_weighted_18d_objective(
        model,
        actions,
        conditions,
        observation_confidence,
        durations_sec,
        frame_valid_mask,
        loss_weights=loss_weights,
        noise=noise,
        flow_times=flow_times,
        generator=generator,
        require_hanyang_partial_weights=True,
    )


def audit_beat2_loss_terms(terms: Mapping[str, torch.Tensor]) -> None:
    missing = sorted(REQUIRED_BEAT2_TERMS - set(terms))
    if missing:
        raise RuntimeError(
            f"BEAT2 emotion-preserving objective is incomplete: {missing}"
        )
    for name in REQUIRED_BEAT2_TERMS:
        value = terms[name]
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise RuntimeError(f"BEAT2 term {name} must be a scalar tensor")
        if not bool(torch.isfinite(value).item()):
            raise FloatingPointError(f"BEAT2 term {name} is non-finite")


def build_lineage_contract(
    config: Mapping[str, Any],
    *,
    arm: str,
) -> dict[str, Any]:
    values = validate_config(config)
    if arm not in {
        "winner_control_0pct_hanyang",
        "winner_isolated_5pct_hanyang",
    }:
        raise ValueError("unknown winner-overlay arm")
    selected = values["winner_overlay_arms"][arm]
    selection = values["qwen_ab_selection_gate"]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": PLAN_ARTIFACT_KIND,
        "arm": arm,
        "frozen_a_candidate_checkpoint": values["foundation_checkpoint"],
        "frozen_a_candidate_checkpoint_sha256": FOUNDATION_SHA256,
        "selected_foundation_checkpoint": selection[
            "selected_foundation_checkpoint"
        ],
        "selected_foundation_checkpoint_sha256": selection[
            "selected_foundation_sha256"
        ],
        "v7_reference_config_sha256": values[
            "expected_v7_reference_config_sha256"
        ],
        "beat2_condition_policy": V7_CONDITION_POLICY,
        "training_policy": V81_TRAINING_POLICY,
        "qwen_ab_selection_status": selection["status"],
        "qwen_variant": selection["selected_qwen_variant"],
        "qwen_condition_cache_sha256": selection[
            "selected_condition_cache_sha256"
        ],
        "hanyang_strict_manifest_sha256": values[
            "expected_hanyang_strict_manifest_sha256"
        ],
        "beat2_fraction": selected["beat2_fraction"],
        "hanyang_fraction": selected["hanyang_fraction"],
        "matched_noop_fraction": selected["matched_noop_fraction"],
        "maximum_hanyang_per_effective_batch": (
            1 if selected["hanyang_fraction"] else 0
        ),
        "seed": values["training"]["seed"],
        "noise_seed": values["training"]["noise_seed"],
        "split_contract": values["training"]["split_contract"],
        "same_selected_winner_required": True,
        "parallel_pair_required": True,
        "sequential_warm_start_allowed": False,
        "approval_status": values["approval_gate"]["status"],
        "formal_release_eligible": False,
    }
    payload.update(_hanyang_artifact_audit(values))
    payload["sha256"] = canonical_sha256(payload)
    return payload


def require_launch_approval(config: Mapping[str, Any]) -> dict[str, Any]:
    values = validate_config(config)
    if values["qwen_ab_selection_gate"]["winner_selected"] is not True:
        raise RuntimeError(
            f"{HUMAN_APPROVAL_BLOCKED}: Qwen A/B winner is not selected"
        )
    if values["approval_gate"]["status"] != "approved":
        raise RuntimeError(
            f"{HUMAN_APPROVAL_BLOCKED}: persistent V8.1 training requires "
            "a hash-bound approval receipt"
        )
    return values


def _atomic_torch(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(value: Mapping[str, Any], path: Path) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                value, ensure_ascii=False, sort_keys=True, allow_nan=False
            )
            + "\n"
        )


def _v7_diagnostic_seed(v7_config: Mapping[str, Any]) -> int:
    """Return the exact fixed diagnostic seed used by the V7 runner."""

    seed = v7_config.get("seed")
    if type(seed) is not int:
        raise ValueError("V7 root seed must be an integer")
    return int(seed) + V7_DIAGNOSTIC_SEED_OFFSET


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))


def _manual_generator(
    device: torch.device, seed: int
) -> torch.Generator:
    generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed) % (2**63 - 1))
    return generator


def paired_slot_quotas(step: int) -> tuple[int, int]:
    """Return the matched BEAT2 core and treatment-only Hanyang slot."""

    if type(step) is not int or step <= 0:
        raise ValueError("step must be a positive integer")
    hanyang_slot = 1 if (step - 1) % 5 < 4 else 0
    return 16 - hanyang_slot, hanyang_slot


def expected_exposure(
    completed_steps: int, *, arm: str
) -> dict[str, int]:
    """Return exact slot counts for any prefix, including short smoke runs."""

    if type(completed_steps) is not int or completed_steps < 0:
        raise ValueError("completed_steps must be a nonnegative integer")
    if arm not in {
        "winner_control_0pct_hanyang",
        "winner_isolated_5pct_hanyang",
    }:
        raise ValueError("unknown V8.1 arm")
    complete_periods, remainder = divmod(completed_steps, 5)
    intervention_slots = complete_periods * 4 + min(remainder, 4)
    beat2 = completed_steps * 16 - intervention_slots
    treatment = arm == "winner_isolated_5pct_hanyang"
    return {
        "beat2": beat2,
        "hanyang": intervention_slots if treatment else 0,
        "matched_noop": 0 if treatment else intervention_slots,
    }


def _runner_paths(
    config: Mapping[str, Any], *, arm: str
) -> dict[str, Path]:
    root = Path(config["winner_overlay_arms"][arm]["output_dir"])
    return {
        "root": root,
        "state": root / "last_state_v8_1.pt",
        "checkpoint": root / "last_generator_v8_1.pt",
        "best": root / "best_admissible_generator_v8_1.pt",
        "summary": root / "training_summary_v8_1.json",
        "progress": root / "progress_v8_1.jsonl",
    }


def _selected_v7_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    selected = config["qwen_ab_selection_gate"]
    if selected["winner_selected"] is not True:
        raise RuntimeError(f"{HUMAN_APPROVAL_BLOCKED}: winner missing")
    values = v7.read_config(config["v7_reference_config"])
    cache_path = str(selected["selected_condition_cache"])
    if selected["selected_qwen_variant"] == "frozen_base":
        if (
            Path(cache_path).resolve()
            != Path(values["frozen_condition_cache"]).resolve()
        ):
            raise ValueError("frozen winner cache differs from V7 frozen-A")
    else:
        values.update(
            {
                "qwen_condition_variant": "lora_finetuned",
                "expected_qwen_condition_cache_sha256": selected[
                    "selected_condition_cache_sha256"
                ],
                "frozen_condition_cache": cache_path,
                "output_dir": str(
                    Path(config["winner_overlay_arms"][
                        "winner_isolated_5pct_hanyang"
                    ]["output_dir"]).parent
                    / "selected_lora_v7_reference"
                ),
            }
        )
    return v7.validate_config(values)


def _runner_input_contract(
    config: Mapping[str, Any],
    *,
    arm: str,
    target_steps: int,
    v7_config: Mapping[str, Any],
) -> dict[str, Any]:
    selection = config["qwen_ab_selection_gate"]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "hanyang_beat2_v8_1_runner_input_contract",
        "arm": arm,
        "target_steps": int(target_steps),
        "selected_qwen_variant": selection["selected_qwen_variant"],
        "selected_foundation_sha256": selection[
            "selected_foundation_sha256"
        ],
        "selected_condition_cache_sha256": selection[
            "selected_condition_cache_sha256"
        ],
        "v7_reference_config_sha256": config[
            "expected_v7_reference_config_sha256"
        ],
        "v7_effective_config_sha256": canonical_sha256(v7_config),
        "hanyang_strict_manifest_sha256": config[
            "expected_hanyang_strict_manifest_sha256"
        ],
        "hanyang_pool_receipt_sha256": config[
            "expected_hanyang_pool_receipt_sha256"
        ],
        "hanyang_rejected_manifest_sha256": config[
            "expected_hanyang_rejected_manifest_sha256"
        ],
        "seed": config["training"]["seed"],
        "noise_seed": config["training"]["noise_seed"],
        "diagnostic_seed_policy": V7_DIAGNOSTIC_SEED_POLICY,
        "diagnostic_seed": _v7_diagnostic_seed(v7_config),
        "paired_beat2_slot_policy": (
            "same_step_same_sampler_state_batch_dropout_noise_time_v1"
        ),
    }
    payload.update(_hanyang_artifact_audit(config))
    payload["sha256"] = canonical_sha256(payload)
    return payload


def _beat2_microbatch_objective(
    *,
    model: nn.Module,
    teacher: nn.Module,
    style_head: nn.Module,
    semantic_perceptual: nn.Module,
    negative_pool: Any,
    rows: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    action_stats: Mapping[str, Any],
    v7_config: Mapping[str, Any],
    step: int,
    microbatch_index: int,
    device: torch.device,
    noise_seed: int,
    anchor_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
    training = v7_config["training"]
    actions, base_conditions, dim_masks, durations, frame_valid = (
        v7_engine._batch_tensors_for_config(
            rows,
            frame_count=int(plan["bucket_frames"]),
            action_stats=action_stats,
            device=device,
            batching=training["batching"],
        )
    )
    dropout_generator = _manual_generator(
        device,
        int(noise_seed) + step * 1_000_003 + microbatch_index * 1009 + 11,
    )
    keep_mask = v7_engine.sample_condition_keep_mask(
        len(rows),
        float(training["condition_dropout_probability"]),
        device=device,
        generator=dropout_generator,
    )
    aligned, predicted_style, style_targets = v7_engine._batch_conditions(
        style_head,
        rows,
        base_conditions,
        keep_mask,
        device=device,
    )
    assert_nonzero_beat2_conditions(
        aligned[keep_mask] if bool(torch.any(keep_mask).item()) else base_conditions
    )
    negative_rows = negative_pool.select(
        rows, step=step, microbatch_index=microbatch_index
    )
    negative = v7_engine._negative_conditions(
        style_head, negative_rows, keep_mask, device=device
    )
    flow_generator = _manual_generator(
        device,
        int(noise_seed) + step * 2_000_003 + microbatch_index * 2017 + 23,
    )
    noise, flow_times = v7_engine._explicit_shared_flow_state(
        actions, frame_valid, generator=flow_generator
    )
    aligned_losses = v7_engine.masked_18d_objective(
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
    pair = v7_engine._condition_pair_losses(
        model,
        actions,
        noise,
        flow_times,
        aligned,
        negative,
        dim_masks,
        frame_valid,
        keep_mask,
        response_floor=float(training["condition_response_floor"]),
        ranking_margin=float(training["condition_ranking_margin"]),
    )
    style_loss = v7_engine._style_smooth_l1(
        predicted_style, style_targets, keep_mask
    )
    semantic = v7_engine._semantic_perceptual_batch(
        semantic_perceptual,
        pair["reconstructed_aligned_normalized"],
        base_conditions,
        frame_valid,
        durations,
        rows,
        keep_mask,
    )
    observed = frame_valid[:, :, None] & dim_masks[:, None, :]
    masked_noise = noise * observed
    x_t = (
        (1.0 - flow_times[:, None, None]) * masked_noise
        + flow_times[:, None, None] * actions
    ) * observed
    student_aligned = forward_with_frame_mask(
        model, x_t, flow_times, aligned, frame_valid
    )
    student_zero = forward_with_frame_mask(
        model, x_t, flow_times, torch.zeros_like(aligned), frame_valid
    )
    with torch.no_grad():
        teacher_aligned = forward_with_frame_mask(
            teacher, x_t, flow_times, aligned.detach(), frame_valid
        )
        teacher_zero = forward_with_frame_mask(
            teacher,
            x_t,
            flow_times,
            torch.zeros_like(aligned),
            frame_valid,
        )
    anchor = condition_response_anchor_loss(
        student_aligned,
        student_zero,
        teacher_aligned,
        teacher_zero,
        observed,
    )
    semantic_weight = v7_engine._semantic_weight_at_step(
        step,
        target_weight=float(
            v7_config["semantic_perceptual"]["outer_weight"]
        ),
        warmup_steps=int(
            v7_config["semantic_perceptual"]["warmup_steps"]
        ),
    )
    total = (
        aligned_losses["total"]
        + float(training["style_smooth_l1_weight"]) * style_loss
        + float(training["condition_ranking_weight"])
        * pair["ranking_loss"]
        + float(training["condition_response_floor_weight"])
        * pair["response_floor_loss"]
        + semantic_weight * semantic["total"]
        + float(anchor_weight) * anchor
    )
    semantic_zero = semantic["total"] * 0.0
    terms = {
        "flow": aligned_losses["flow"],
        "condition_ranking": pair["ranking_loss"],
        "condition_response_floor": pair["response_floor_loss"],
        "hierarchy_binary_loss": semantic.get(
            "hierarchy_binary_loss", semantic_zero
        ),
        "hierarchy_emotion_loss": semantic.get(
            "hierarchy_emotion_loss", semantic_zero
        ),
        "hierarchy_group_auxiliary_loss": semantic.get(
            "hierarchy_group_auxiliary_loss", semantic_zero
        ),
        "emotion_response_anchor": anchor,
    }
    audit_beat2_loss_terms(terms)
    metrics = {
        **terms,
        "total": total,
        "position": aligned_losses["position"],
        "velocity": aligned_losses["velocity"],
        "style_smooth_l1": style_loss,
        "condition_response_rms": pair["response_rms"],
        "semantic_total": semantic["total"],
    }
    receipt = {
        "clip_ids": [str(row["clip_id"]) for row in rows],
        "clip_ids_sha256": canonical_sha256(
            [str(row["clip_id"]) for row in rows]
        ),
        "negative_clip_ids_sha256": canonical_sha256(
            [str(row["clip_id"]) for row in negative_rows]
        ),
        "keep_mask": keep_mask.detach().cpu().tolist(),
        "noise_sha256": hashlib.sha256(
            noise.detach().cpu().numpy().tobytes()
        ).hexdigest(),
        "flow_times_sha256": hashlib.sha256(
            flow_times.detach().cpu().numpy().tobytes()
        ).hexdigest(),
    }
    return total, metrics, receipt


def _gradient_norm(parameters: Sequence[nn.Parameter]) -> float:
    squares = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt().cpu())


def _fixed_v7_diagnostic_subset(
    episodes: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    count: int,
) -> list[dict[str, Any]]:
    candidates = v7_engine._stable_subset(
        episodes,
        count=len(episodes),
        seed=int(seed),
        label="v2_balanced_group_diagnostic",
    )
    groups: dict[int, dict[str, Any]] = {}
    for row in candidates:
        groups.setdefault(
            int(row["experimental_semantic_group_index"]), dict(row)
        )
    selected = list(groups.values())[: int(count)]
    if len(selected) != int(count):
        raise ValueError("fixed V7 diagnostics do not cover all 54 groups")
    return selected


def _captured_hierarchy_metrics(
    captured: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    mapping = {
        "q2_loss": "hierarchy_binary_loss",
        "q2_recall_at_1": "hierarchy_binary_recall_at_1",
        "q2_hard_margin": "hierarchy_binary_hard_margin",
        "q6_loss": "hierarchy_emotion_loss",
        "q6_recall_at_1": "hierarchy_emotion_recall_at_1",
        "q6_hard_margin": "hierarchy_emotion_hard_margin",
        "global54_loss": "hierarchy_group_auxiliary_loss",
        "global54_recall_at_1": (
            "hierarchy_group_auxiliary_recall_at_1"
        ),
        "global54_hard_margin": (
            "hierarchy_group_auxiliary_hard_margin"
        ),
    }
    result = {}
    for output_name, source_name in mapping.items():
        value = captured.get(source_name)
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise RuntimeError(
                f"V7 hierarchy diagnostic missing {source_name}"
            )
        result[output_name] = float(value.detach().cpu())
    return result


@torch.no_grad()
def _ema_validation_diagnostics(
    *,
    model: nn.Module,
    style_head: nn.Module,
    semantic_perceptual: nn.Module,
    beat2_rows: Sequence[Mapping[str, Any]],
    hanyang_rows: Sequence[Mapping[str, Any]],
    negative_pool: Any,
    action_stats: Mapping[str, Any],
    v7_config: Mapping[str, Any],
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    captured: dict[str, torch.Tensor] = {}

    def capture_hierarchy(
        _module: nn.Module, _inputs: tuple[Any, ...], output: Mapping[str, Any]
    ) -> None:
        captured.update(
            {
                key: value
                for key, value in output.items()
                if key.startswith("hierarchy_")
                and isinstance(value, torch.Tensor)
            }
        )

    handle = semantic_perceptual.register_forward_hook(capture_hierarchy)
    try:
        beat2 = v7_engine.condition_diagnostics(
            model,
            style_head,
            semantic_perceptual,
            beat2_rows,
            negative_pool,
            action_stats=action_stats,
            batching=v7_config["training"]["batching"],
            device=device,
            seed=int(seed),
        )
    finally:
        handle.remove()
    beat2.update(_captured_hierarchy_metrics(captured))
    was_training = model.training
    model.eval()
    hanyang_batch = collate_confidence_weighted_18d(
        hanyang_rows,
        buckets=v7_config["training"]["batching"]["length_buckets"],
        action_stats=action_stats,
        device=device,
    )
    hanyang_loss_weights = {
        name: float(v7_config["training"]["loss"][name])
        for name in ("flow", "position", "velocity", "acceleration", "jerk")
    }
    hanyang_losses = hanyang_partial_motion_objective(
        model,
        hanyang_batch["actions"],
        hanyang_batch["observation_confidence"],
        hanyang_batch["durations_sec"],
        hanyang_batch["frame_valid_mask"],
        loss_weights=hanyang_loss_weights,
        generator=_manual_generator(device, int(seed) + 700_001),
    )
    if was_training:
        model.train()
    hanyang = {
        name: float(value.detach().cpu())
        for name, value in hanyang_losses.items()
    }
    if not all(
        math.isfinite(value)
        for value in list(beat2.values())
        if isinstance(value, (int, float))
    ) or not all(math.isfinite(value) for value in hanyang.values()):
        raise FloatingPointError("non-finite held-out V8.1 diagnostics")
    return {
        "beat2": beat2,
        "hanyang": hanyang,
        "beat2_clip_ids_sha256": canonical_sha256(
            [str(row["clip_id"]) for row in beat2_rows]
        ),
        "hanyang_clip_ids_sha256": canonical_sha256(
            [str(row["clip_id"]) for row in hanyang_rows]
        ),
        "seed": int(seed),
        "ema": True,
    }


def _retention_ratio(current: float, baseline: float) -> float:
    current_value = float(current)
    baseline_value = float(baseline)
    if not math.isfinite(current_value) or not math.isfinite(baseline_value):
        return float("-inf")
    if baseline_value <= 1e-12:
        return 1.0 if current_value >= baseline_value else 0.0
    return current_value / baseline_value


def _diagnostic_candidate_gate(
    *,
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
    v7_config: Mapping[str, Any],
    diagnostics_config: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_beat2 = baseline["beat2"]
    current_beat2 = current["beat2"]
    absolute = v7_engine.anti_collapse_decision(
        baseline_beat2,
        current_beat2,
        v7_config["anti_collapse_gates"],
        enforce=True,
    )
    retentions = {
        "aligned_vs_zero": _retention_ratio(
            current_beat2["aligned_vs_zero_prediction_rms"],
            baseline_beat2["aligned_vs_zero_prediction_rms"],
        ),
        "aligned_vs_cross_group": _retention_ratio(
            current_beat2["aligned_vs_cross_group_prediction_rms"],
            baseline_beat2["aligned_vs_cross_group_prediction_rms"],
        ),
        "flow_gap": _retention_ratio(
            current_beat2["cross_group_minus_aligned_flow_loss"],
            baseline_beat2["cross_group_minus_aligned_flow_loss"],
        ),
        "q2_recall": _retention_ratio(
            current_beat2["q2_recall_at_1"],
            baseline_beat2["q2_recall_at_1"],
        ),
        "q6_recall": _retention_ratio(
            current_beat2["q6_recall_at_1"],
            baseline_beat2["q6_recall_at_1"],
        ),
        "global54_recall": _retention_ratio(
            current_beat2["global54_recall_at_1"],
            baseline_beat2["global54_recall_at_1"],
        ),
    }
    thresholds = {
        "aligned_vs_zero": float(
            diagnostics_config["minimum_response_retention"]
        ),
        "aligned_vs_cross_group": float(
            diagnostics_config["minimum_cross_response_retention"]
        ),
        "flow_gap": float(
            diagnostics_config["minimum_flow_gap_retention"]
        ),
        "q2_recall": float(
            diagnostics_config["minimum_q2_recall_retention"]
        ),
        "q6_recall": float(
            diagnostics_config["minimum_q6_recall_retention"]
        ),
        "global54_recall": float(
            diagnostics_config["minimum_global54_recall_retention"]
        ),
    }
    retention_checks = {
        name: value >= thresholds[name]
        for name, value in retentions.items()
    }
    finite_hanyang = math.isfinite(float(current["hanyang"]["total"]))
    admissible = bool(
        absolute["passed"]
        and all(retention_checks.values())
        and finite_hanyang
    )
    minimum_retention = min(retentions.values())
    beat2_loss_ratio = _retention_ratio(
        baseline_beat2["aligned_flow_loss"],
        max(float(current_beat2["aligned_flow_loss"]), 1e-12),
    )
    hanyang_loss_ratio = _retention_ratio(
        baseline["hanyang"]["total"],
        max(float(current["hanyang"]["total"]), 1e-12),
    )
    score = (
        minimum_retention
        + 0.01 * beat2_loss_ratio
        + 0.01 * hanyang_loss_ratio
    )
    return {
        "admissible": admissible,
        "absolute_v7_gate": absolute,
        "retentions": retentions,
        "retention_thresholds": thresholds,
        "retention_checks": retention_checks,
        "hanyang_validation_finite": finite_hanyang,
        "minimum_emotion_retention": minimum_retention,
        "selection_score": float(score),
        "failure_reasons": [
            *(
                []
                if absolute["passed"]
                else [
                    f"absolute:{reason}"
                    for reason in absolute["failure_reasons"]
                ]
            ),
            *[
                f"retention:{name}"
                for name, passed in retention_checks.items()
                if not passed
            ],
            *([] if finite_hanyang else ["hanyang_validation_nonfinite"]),
        ],
    }


def _numpy_rng_payload() -> dict[str, Any]:
    state = np.random.get_state()
    return {
        "bit_generator": str(state[0]),
        "keys": torch.from_numpy(
            np.asarray(state[1], dtype=np.uint32).copy()
        ),
        "position": int(state[2]),
        "has_gaussian": int(state[3]),
        "cached_gaussian": float(state[4]),
    }


def _save_runner_state(
    path: Path,
    *,
    step: int,
    target_steps: int,
    input_contract_sha256: str,
    model: nn.Module,
    style_head: nn.Module,
    model_ema: ModelEMA,
    style_head_ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    beat2_sampler: Any,
    hanyang_sampler: Any,
    exposure: Mapping[str, int],
    last_event: Mapping[str, Any],
    baseline_diagnostics: Mapping[str, Any],
    last_diagnostics: Mapping[str, Any] | None,
    best_admissible: Mapping[str, Any] | None,
) -> None:
    _atomic_torch(
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": STATE_ARTIFACT_KIND,
            "step": int(step),
            "target_steps": int(target_steps),
            "input_contract_sha256": input_contract_sha256,
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
            "beat2_sampler_state_dict": beat2_sampler.state_dict(),
            "hanyang_sampler_state_dict": hanyang_sampler.state_dict(),
            "exposure": dict(exposure),
            "last_event": deepcopy(dict(last_event)),
            "baseline_diagnostics": deepcopy(dict(baseline_diagnostics)),
            "last_diagnostics": deepcopy(
                dict(last_diagnostics) if last_diagnostics is not None else None
            ),
            "best_admissible": deepcopy(
                dict(best_admissible) if best_admissible is not None else None
            ),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else []
            ),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": _numpy_rng_payload(),
        },
        path,
    )


def _load_runner_state(
    path: Path,
    *,
    target_steps: int,
    input_contract_sha256: str,
    model: nn.Module,
    style_head: nn.Module,
    model_ema: ModelEMA,
    style_head_ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    beat2_sampler: Any,
    hanyang_sampler: Any,
    device: torch.device,
) -> tuple[
    int,
    dict[str, int],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (
        state.get("schema_version") != SCHEMA_VERSION
        or state.get("artifact_kind") != STATE_ARTIFACT_KIND
        or state.get("input_contract_sha256") != input_contract_sha256
        or int(state.get("target_steps", -1)) != int(target_steps)
    ):
        raise ValueError("V8.1 exact-resume contract changed")
    step = int(state.get("step", -1))
    if not 1 <= step <= target_steps:
        raise ValueError("V8.1 resume step is invalid")
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
    for optimizer_state in optimizer.state.values():
        for name, value in optimizer_state.items():
            if isinstance(value, torch.Tensor):
                optimizer_state[name] = value.to(device)
    beat2_sampler.load_state_dict(state["beat2_sampler_state_dict"])
    hanyang_sampler.load_state_dict(state["hanyang_sampler_state_dict"])
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state["cuda_rng_state_all"]:
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    random.setstate(state["python_rng_state"])
    numpy_state = state["numpy_rng_state"]
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            numpy_state["keys"].numpy().astype(np.uint32, copy=True),
            int(numpy_state["position"]),
            int(numpy_state["has_gaussian"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    return (
        step,
        {key: int(value) for key, value in state["exposure"].items()},
        dict(state["last_event"]),
        dict(state["baseline_diagnostics"]),
        (
            dict(state["last_diagnostics"])
            if state["last_diagnostics"] is not None
            else None
        ),
        (
            dict(state["best_admissible"])
            if state["best_admissible"] is not None
            else None
        ),
    )


def load_v7_checkpoint_for_v81(
    path: str | Path,
    *,
    expected_sha256: str,
    device: torch.device | str,
) -> tuple[nn.Module, nn.Module, dict[str, Any]]:
    """Strictly instantiate a completed V7 generator and its Qwen style head.

    V7 is a post-training artifact and intentionally does not satisfy the
    generic random-initialization checkpoint contract.  This loader validates
    the narrower V7 contract without weakening that generic validator.
    """

    target = Path(path).expanduser().resolve()
    if (
        not target.is_file()
        or not _is_sha256(expected_sha256)
        or sha256_file(target) != expected_sha256
    ):
        raise ValueError("selected V7 checkpoint path or SHA256 changed")
    checkpoint = torch.load(target, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("selected V7 checkpoint must be a mapping")
    model_config = checkpoint.get("config")
    action_stats = checkpoint.get("action_stats")
    model_state = checkpoint.get("model_state_dict")
    style_config = checkpoint.get("qwen_style_head_config")
    style_state = checkpoint.get("qwen_style_head_state_dict")
    anti_collapse = checkpoint.get("anti_collapse_gate")
    frozen_audit = checkpoint.get("frozen_parameter_audit")
    if (
        checkpoint.get("schema_version") != v7.SCHEMA_VERSION
        or checkpoint.get("artifact_kind") != v7.CHECKPOINT_ARTIFACT_KIND
        or checkpoint.get("condition_policy") != V7_CONDITION_POLICY
        or checkpoint.get("architecture")
        != ULA_MMDIT_V3_ADALN_ARCHITECTURE
        or int(checkpoint.get("action_dim", -1)) != ACTION_DIM
        or int(checkpoint.get("condition_dim", -1)) != 264
        or list(checkpoint.get("joint_order") or [])
        != list(JOINT_ORDER_18D)
        or checkpoint.get("formal_release_eligible") is not False
        or checkpoint.get("experimental_only") is not True
        or checkpoint.get("no_external_data") is not True
        or checkpoint.get("no_kimodo") is not True
        or not isinstance(anti_collapse, Mapping)
        or anti_collapse.get("passed") is not True
        or anti_collapse.get("enforced") is not True
        or not isinstance(frozen_audit, Mapping)
        or frozen_audit.get("passed") is not True
        or frozen_audit.get("changed_frozen_tensor_names") != []
        or float(frozen_audit.get("maximum_abs_error", math.inf)) != 0.0
        or not isinstance(model_config, Mapping)
        or set(model_config)
        != {*V7_RUNTIME_MODEL_CONFIG, "checkpoint_loss"}
        or any(
            model_config.get(name) != expected
            for name, expected in V7_RUNTIME_MODEL_CONFIG.items()
        )
        or not math.isfinite(float(model_config.get("checkpoint_loss", math.nan)))
        or not isinstance(model_state, Mapping)
        or not isinstance(style_config, Mapping)
        or not isinstance(style_state, Mapping)
    ):
        raise ValueError("selected V7 runtime checkpoint contract changed")
    if not isinstance(action_stats, Mapping) or set(action_stats) != {
        "mean",
        "std",
    }:
        raise ValueError("selected V7 action statistics changed")
    normalized_stats: dict[str, torch.Tensor] = {}
    for name in ("mean", "std"):
        value = torch.as_tensor(action_stats[name], dtype=torch.float32)
        if tuple(value.shape) != (ACTION_DIM,) or not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError(f"selected V7 action_stats.{name} is invalid")
        normalized_stats[name] = value.detach().cpu().clone()
    if bool(torch.any(normalized_stats["std"] <= 0)):
        raise ValueError("selected V7 action_stats.std must be positive")
    for context, state in (
        ("model", model_state),
        ("Qwen style head", style_state),
    ):
        if any(
            not isinstance(value, torch.Tensor)
            or not bool(torch.isfinite(value).all())
            for value in state.values()
        ):
            raise ValueError(f"selected V7 {context} state is non-finite")

    resolved_device = torch.device(device)
    model = create_ula_model(
        ULA_MMDIT_V3_ADALN_ARCHITECTURE,
        action_dim=ACTION_DIM,
        condition_dim=264,
        hidden_dim=int(model_config["hidden_dim"]),
        layers=int(model_config["layers"]),
        semantic_tokens=int(model_config["semantic_tokens"]),
    )
    try:
        model.load_state_dict(model_state, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "selected V7 generator state_dict failed strict loading"
        ) from exc
    model.action_stats = normalized_stats
    model.planner_supervision_contract = deepcopy(
        checkpoint.get("planner_supervision_contract")
        or (checkpoint.get("training_contract") or {}).get(
            "planner_supervision"
        )
        or {}
    )
    try:
        style_head = v7_engine.QwenStyleHead.from_config(style_config)
        style_head.load_state_dict(style_state, strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            "selected V7 Qwen style-head contract failed strict loading"
        ) from exc
    if style_head.architecture_config() != dict(style_config):
        raise ValueError("selected V7 Qwen style-head config did not round-trip")
    return model.to(resolved_device), style_head.to(resolved_device), dict(
        checkpoint
    )


def _prepare_runner(
    config: Mapping[str, Any],
    *,
    arm: str,
    device: torch.device,
    preloaded_hanyang: tuple[
        list[dict[str, Any]], dict[str, Any]
    ] | None = None,
) -> dict[str, Any]:
    """Load the selected winner and both strict domains after all gates."""

    v7_config = _selected_v7_config(config)
    with v7._patched_v7_engine(v7_config):
        beat2_splits, beat2_split_contract, _bridge, preparation = (
            v7_engine._prepare_rows(v7_config, smoke_test=False)
        )
        selected_path = config["qwen_ab_selection_gate"][
            "selected_foundation_checkpoint"
        ]
        model, style_head, selected_checkpoint = load_v7_checkpoint_for_v81(
            selected_path,
            expected_sha256=config["qwen_ab_selection_gate"][
                "selected_foundation_sha256"
            ],
            device=device,
        )
        semantic_perceptual, semantic_receipt = (
            v7_engine._build_semantic_perceptual(
                v7_config,
                action_stats=selected_checkpoint["action_stats"],
                device=device,
            )
        )
    if preloaded_hanyang is None:
        hanyang, hanyang_receipt = _pre_device_hanyang_gate(config)
    else:
        hanyang, hanyang_receipt = preloaded_hanyang
    hanyang_splits = {
        name: [
            row
            for row in hanyang
            if row["fixed_split_assignment"] == name
        ]
        for name in ("train", "validation", "test")
    }
    if {
        name: len(rows) for name, rows in hanyang_splits.items()
    } != HANYANG_TRAINING_ELIGIBLE_SPLIT_COUNTS:
        raise ValueError("filtered Hanyang participant split changed")
    teacher = deepcopy(model).to(device)
    teacher.requires_grad_(False)
    teacher.eval()
    routes = parameter_gradient_routes(model)
    named = dict(model.named_parameters())
    model.requires_grad_(False)
    for name in routes["beat2_condition"] + routes["hanyang_motion"]:
        named[name].requires_grad_(True)
    style_head.requires_grad_(True)
    training = config["training"]
    v7_training = v7_config["training"]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [named[name] for name in routes["beat2_condition"]],
                "lr": float(v7_training["lr"]),
                "weight_decay": float(v7_training["weight_decay"]),
                "role": "beat2_v7_condition_path",
            },
            {
                "params": [named[name] for name in routes["hanyang_motion"]],
                "lr": float(training["hanyang_motion_lr"]),
                "weight_decay": float(v7_training["weight_decay"]),
                "role": "hanyang_motion_backbone",
            },
            {
                "params": list(style_head.parameters()),
                "lr": float(v7_training["style_head_lr"]),
                "weight_decay": float(v7_training["weight_decay"]),
                "role": "beat2_qwen_style_head",
            },
        ],
        eps=float(v7_training["adam_eps"]),
    )
    beat2_sampler = v7.SourceGroupFirstNativeBucketSampler(
        beat2_splits["train"],
        buckets=v7_training["batching"]["length_buckets"],
        seed=int(training["seed"]) + 17,
    )
    hanyang_sampler = NativeLengthBucketSampler(
        hanyang_splits["train"],
        buckets=v7_training["batching"]["length_buckets"],
        sampler_config={"mode": "domain_speaker"},
        seed=int(training["seed"]) + 37,
    )
    negative_pool = v7_engine.CrossGroupNegativePool(
        beat2_splits["train"], seed=int(training["seed"]) + 37
    )
    diagnostic_rows = _fixed_v7_diagnostic_subset(
        beat2_splits["validation"],
        seed=int(training["seed"]),
        count=int(config["diagnostics"]["validation_group_count"]),
    )
    hanyang_diagnostic_rows = sorted(
        hanyang_splits["validation"], key=lambda row: str(row["clip_id"])
    )[: int(config["diagnostics"]["hanyang_validation_episode_count"])]
    if len(hanyang_diagnostic_rows) != int(
        config["diagnostics"]["hanyang_validation_episode_count"]
    ) or {
        str(row["clip_id"]) for row in hanyang_diagnostic_rows
    } != {
        str(row["clip_id"]) for row in hanyang_splits["validation"]
    }:
        raise ValueError(
            "Hanyang diagnostics must be the filtered validation-only five"
        )
    return {
        "v7_config": v7_config,
        "beat2_splits": beat2_splits,
        "beat2_split_contract": beat2_split_contract,
        "beat2_preparation": preparation,
        "hanyang_splits": hanyang_splits,
        "hanyang_receipt": hanyang_receipt,
        "selected_checkpoint": selected_checkpoint,
        "model": model,
        "teacher": teacher,
        "style_head": style_head,
        "semantic_perceptual": semantic_perceptual,
        "semantic_receipt": semantic_receipt,
        "optimizer": optimizer,
        "routes": routes,
        "beat2_sampler": beat2_sampler,
        "hanyang_sampler": hanyang_sampler,
        "negative_pool": negative_pool,
        "diagnostic_rows": diagnostic_rows,
        "hanyang_diagnostic_rows": hanyang_diagnostic_rows,
    }


def _pre_device_hanyang_gate(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load source344, then exclude the exact unapproved boundary21 IDs."""

    boundary = _validate_hanyang_boundary_exclusion(config)
    hanyang, receipt = load_hanyang_partial_motion_episodes(
        config["hanyang_strict_manifest"],
        expected_manifest_sha256=config[
            "expected_hanyang_strict_manifest_sha256"
        ],
        pool_receipt_path=config["hanyang_pool_receipt"],
        expected_pool_receipt_sha256=config[
            "expected_hanyang_pool_receipt_sha256"
        ],
        expected_upstream_passed_manifest_sha256=config[
            "expected_hanyang_upstream_passed_manifest_sha256"
        ],
    )
    if len(hanyang) != HANYANG_SOURCE_POOL_COUNT:
        raise ValueError("strict Hanyang admission is not exactly 344")
    if any(
        row.get("emotion_conditioning_mask") is not False
        or row.get("semantic_supervision_mask") is not False
        or row.get("style_supervision_mask") is not False
        for row in hanyang
    ):
        raise ValueError("Hanyang semantic/emotion condition lane reopened")
    source_ids = {str(row["clip_id"]) for row in hanyang}
    safe_ids = set(boundary["safe_clip_ids"])
    excluded_ids = set(boundary["excluded_clip_ids"])
    if (
        source_ids != safe_ids | excluded_ids
        or safe_ids & excluded_ids
        or len(source_ids) != HANYANG_SOURCE_POOL_COUNT
    ):
        raise ValueError("loaded Hanyang IDs differ from boundary contract")
    eligible = [
        row for row in hanyang if str(row["clip_id"]) in safe_ids
    ]
    eligible_split_counts = dict(
        Counter(
            str(row["fixed_split_assignment"]) for row in eligible
        )
    )
    if (
        len(eligible) != HANYANG_TRAINING_ELIGIBLE_COUNT
        or eligible_split_counts
        != HANYANG_TRAINING_ELIGIBLE_SPLIT_COUNTS
        or any(str(row["clip_id"]) in excluded_ids for row in eligible)
        or canonical_sha256(
            sorted(str(row["clip_id"]) for row in eligible)
        )
        != HANYANG_SAFE_CLIP_IDS_SHA256
    ):
        raise ValueError("Hanyang boundary IDs were not fully excluded")
    filtered_receipt = deepcopy(dict(receipt))
    source_receipt_sha256 = filtered_receipt.pop("sha256", None)
    filtered_receipt.update(
        {
            "source_loader_receipt_sha256": source_receipt_sha256,
            "source_episode_count": HANYANG_SOURCE_POOL_COUNT,
            "source_split_counts": dict(HANYANG_SOURCE_SPLIT_COUNTS),
            "episode_count": HANYANG_TRAINING_ELIGIBLE_COUNT,
            "split_counts": dict(
                HANYANG_TRAINING_ELIGIBLE_SPLIT_COUNTS
            ),
            "boundary_review_manifest": boundary["review_manifest"],
            **_hanyang_artifact_audit(config),
        }
    )
    filtered_receipt["sha256"] = canonical_sha256(filtered_receipt)
    return eligible, filtered_receipt


def _runner_checkpoint_payload(
    *,
    config: Mapping[str, Any],
    arm: str,
    step: int,
    target_steps: int,
    input_contract: Mapping[str, Any],
    model_ema: ModelEMA,
    style_head_ema: ModelEMA,
    selected_checkpoint: Mapping[str, Any],
    exposure: Mapping[str, int],
    last_event: Mapping[str, Any],
    smoke_test: bool,
    checkpoint_role: str,
    diagnostic_gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    arm_contract = config["winner_overlay_arms"][arm]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": CHECKPOINT_ARTIFACT_KIND,
        "arm": arm,
        "global_step": int(selected_checkpoint.get("global_step", 0))
        + int(step),
        "v8_1_step": int(step),
        "target_steps": int(target_steps),
        "architecture": selected_checkpoint["architecture"],
        "action_dim": selected_checkpoint["action_dim"],
        "condition_dim": selected_checkpoint["condition_dim"],
        "joint_order": deepcopy(selected_checkpoint["joint_order"]),
        "action_stats": deepcopy(selected_checkpoint["action_stats"]),
        "model_state_dict": {
            name: value.detach().cpu()
            for name, value in model_ema.shadow.items()
        },
        "qwen_style_head_config": deepcopy(
            selected_checkpoint["qwen_style_head_config"]
        ),
        "qwen_style_head_state_dict": {
            name: value.detach().cpu()
            for name, value in style_head_ema.shadow.items()
        },
        "condition_policy": V7_CONDITION_POLICY,
        "training_policy": V81_TRAINING_POLICY,
        "selected_qwen_variant": config["qwen_ab_selection_gate"][
            "selected_qwen_variant"
        ],
        "selected_foundation_sha256": config["qwen_ab_selection_gate"][
            "selected_foundation_sha256"
        ],
        "input_contract": deepcopy(dict(input_contract)),
        "exposure": dict(exposure),
        "declared_slot_fractions": {
            "beat2": float(arm_contract["beat2_fraction"]),
            "hanyang": float(arm_contract["hanyang_fraction"]),
            "matched_noop": float(
                arm_contract["matched_noop_fraction"]
            ),
        },
        "last_event": deepcopy(dict(last_event)),
        "checkpoint_role": checkpoint_role,
        "candidate_eligible": bool(
            diagnostic_gate is not None
            and diagnostic_gate.get("admissible") is True
            and checkpoint_role == "best_admissible"
        ),
        "diagnostic_gate": deepcopy(
            dict(diagnostic_gate) if diagnostic_gate is not None else None
        ),
        "formal_release_eligible": False,
        "smoke_test": bool(smoke_test),
    }
    payload.update(_hanyang_artifact_audit(config))
    return payload


def run_arm(
    config: Mapping[str, Any],
    *,
    arm: str,
    resume: bool = False,
    overwrite: bool = False,
    smoke_test: bool = False,
    smoke_output_dir: str | Path | None = None,
    audit_stop_step: int | None = None,
    audit_output_dir: str | Path | None = None,
    device_override: str | None = None,
) -> dict[str, Any]:
    """Run one independently submitted arm from the exact selected winner."""

    values = require_launch_approval(config)
    if arm not in {
        "winner_control_0pct_hanyang",
        "winner_isolated_5pct_hanyang",
    }:
        raise ValueError("unknown V8.1 arm")
    if smoke_test and smoke_output_dir is None:
        raise ValueError("smoke_test requires an explicit temporary output")
    audit_replay = audit_stop_step is not None or audit_output_dir is not None
    if (audit_stop_step is None) != (audit_output_dir is None):
        raise ValueError(
            "audit replay requires both audit_stop_step and audit_output_dir"
        )
    if smoke_test and audit_replay:
        raise ValueError("smoke_test and audit replay are mutually exclusive")
    if smoke_test:
        values = deepcopy(values)
        values["winner_overlay_arms"][arm]["output_dir"] = str(
            Path(smoke_output_dir).resolve()
        )
    elif audit_replay:
        values = deepcopy(values)
        values["winner_overlay_arms"][arm]["output_dir"] = str(
            Path(audit_output_dir).resolve()
        )
    paths = _runner_paths(values, arm=arm)
    target_steps = 1 if smoke_test else int(values["training"]["steps"])
    run_end_step = (
        int(audit_stop_step) if audit_stop_step is not None else target_steps
    )
    if not 1 <= run_end_step <= target_steps:
        raise ValueError("audit_stop_step must be within the formal schedule")
    checkpoint_aligned = (
        run_end_step % int(values["training"]["checkpoint_interval"]) == 0
    )
    diagnostic_aligned = (
        run_end_step % int(values["diagnostics"]["interval_steps"]) == 0
        or (
            bool(values["diagnostics"]["run_at_every_checkpoint"])
            and checkpoint_aligned
        )
    )
    if audit_replay and not (checkpoint_aligned and diagnostic_aligned):
        raise ValueError(
            "audit_stop_step must be both a checkpoint and diagnostic step"
        )
    v7_config = _selected_v7_config(values)
    input_contract = _runner_input_contract(
        values, arm=arm, target_steps=target_steps, v7_config=v7_config
    )
    preloaded_hanyang = _pre_device_hanyang_gate(values)
    if overwrite:
        for path in paths.values():
            if path != paths["root"] and path.is_file():
                path.unlink()
    if paths["state"].is_file() and not resume and not overwrite:
        raise FileExistsError("V8.1 state exists; use --resume or --overwrite")
    paths["root"].mkdir(parents=True, exist_ok=True)
    device_name = device_override or "cuda"
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    _seed_everything(int(values["training"]["seed"]))
    runtime = _prepare_runner(
        values,
        arm=arm,
        device=device,
        preloaded_hanyang=preloaded_hanyang,
    )
    model = runtime["model"]
    style_head = runtime["style_head"]
    diagnostic_seed = _v7_diagnostic_seed(runtime["v7_config"])
    if diagnostic_seed != input_contract["diagnostic_seed"]:
        raise RuntimeError(
            "runtime V7 diagnostic seed differs from the audited input contract"
        )
    optimizer = runtime["optimizer"]
    model_ema = ModelEMA(model, float(values["training"]["ema_decay"]))
    style_head_ema = ModelEMA(
        style_head, float(values["training"]["ema_decay"])
    )
    beat2_sampler = runtime["beat2_sampler"]
    hanyang_sampler = runtime["hanyang_sampler"]
    exposure = {"beat2": 0, "hanyang": 0, "matched_noop": 0}
    step = 0
    last_event: dict[str, Any] = {}
    baseline_diagnostics: dict[str, Any] | None = None
    last_diagnostics: dict[str, Any] | None = None
    best_admissible: dict[str, Any] | None = None
    if paths["state"].is_file():
        if not resume:
            raise FileExistsError("resume state exists")
        (
            step,
            exposure,
            last_event,
            baseline_diagnostics,
            last_diagnostics,
            best_admissible,
        ) = _load_runner_state(
            paths["state"],
            target_steps=target_steps,
            input_contract_sha256=input_contract["sha256"],
            model=model,
            style_head=style_head,
            model_ema=model_ema,
            style_head_ema=style_head_ema,
            optimizer=optimizer,
            beat2_sampler=beat2_sampler,
            hanyang_sampler=hanyang_sampler,
            device=device,
        )
        if step > run_end_step:
            raise RuntimeError("audit replay state is beyond audit_stop_step")
    else:
        baseline_diagnostics = _ema_validation_diagnostics(
            model=model,
            style_head=style_head,
            semantic_perceptual=runtime["semantic_perceptual"],
            beat2_rows=runtime["diagnostic_rows"],
            hanyang_rows=runtime["hanyang_diagnostic_rows"],
            negative_pool=runtime["negative_pool"],
            action_stats=runtime["selected_checkpoint"]["action_stats"],
            v7_config=runtime["v7_config"],
            device=device,
            seed=diagnostic_seed,
        )
        paths["progress"].write_text("", encoding="utf-8")
        _append_jsonl(
            {
                "event": "initialized",
                "arm": arm,
                "step": 0,
                "input_contract_sha256": input_contract["sha256"],
                "selected_foundation_sha256": values[
                    "qwen_ab_selection_gate"
                ]["selected_foundation_sha256"],
                "smoke_test": bool(smoke_test),
                "audit_replay": bool(audit_replay),
                "audit_stop_step": (
                    int(run_end_step) if audit_replay else None
                ),
                "baseline_diagnostics": baseline_diagnostics,
            },
            paths["progress"],
        )
    if baseline_diagnostics is None:
        raise RuntimeError("V8.1 baseline diagnostics are missing")
    routes = runtime["routes"]
    named = dict(model.named_parameters())
    beat_parameters = [named[name] for name in routes["beat2_condition"]]
    hanyang_parameters = [
        named[name] for name in routes["hanyang_motion"]
    ]
    style_parameters = list(style_head.parameters())
    all_parameters = beat_parameters + hanyang_parameters + style_parameters
    v7_training = runtime["v7_config"]["training"]
    hanyang_loss_weights = {
        name: float(v7_training["loss"][name])
        for name in ("flow", "position", "velocity", "acceleration", "jerk")
    }
    started = time.monotonic()
    for current_step in range(step + 1, run_end_step + 1):
        beat_quota, hanyang_slot = paired_slot_quotas(current_step)
        lr_scale = v7_engine._lr_scale(
            current_step,
            total_steps=target_steps,
            warmup_steps=min(
                int(v7_training["warmup_steps"]), target_steps
            ),
            minimum_ratio=float(v7_training["minimum_lr_ratio"]),
        )
        optimizer.param_groups[0]["lr"] = (
            float(v7_training["lr"]) * lr_scale
        )
        optimizer.param_groups[1]["lr"] = (
            float(values["training"]["hanyang_motion_lr"]) * lr_scale
        )
        optimizer.param_groups[2]["lr"] = (
            float(v7_training["style_head_lr"]) * lr_scale
        )
        optimizer.zero_grad(set_to_none=True)
        remaining = beat_quota
        beat_total = None
        motion_guard_total = None
        beat_metrics: defaultdict[str, float] = defaultdict(float)
        beat_receipts = []
        microbatch_index = 0
        while remaining > 0:
            rows, plan = beat2_sampler.sample_microbatch(
                remaining_effective_batch=remaining,
                semantic_tokens=int(model.semantic_tokens),
                max_batch_size=int(v7_training["batch_size"]),
                batching=v7_training["batching"],
            )
            total, metrics, receipt = _beat2_microbatch_objective(
                model=model,
                teacher=runtime["teacher"],
                style_head=style_head,
                semantic_perceptual=runtime["semantic_perceptual"],
                negative_pool=runtime["negative_pool"],
                rows=rows,
                plan=plan,
                action_stats=runtime["selected_checkpoint"]["action_stats"],
                v7_config=runtime["v7_config"],
                step=current_step,
                microbatch_index=microbatch_index,
                device=device,
                noise_seed=int(values["training"]["noise_seed"]),
                anchor_weight=float(
                    values["training"]["emotion_response_anchor_weight"]
                ),
            )
            weight = len(rows) / 16.0
            weighted = total * weight
            beat_total = weighted if beat_total is None else beat_total + weighted
            guard = (
                metrics["emotion_response_anchor"]
                * weight
                * float(
                    values["training"]["emotion_response_anchor_weight"]
                )
            )
            motion_guard_total = (
                guard
                if motion_guard_total is None
                else motion_guard_total + guard
            )
            for name, metric in metrics.items():
                beat_metrics[name] += float(metric.detach().cpu()) * weight
            beat_receipts.append(receipt)
            exposure["beat2"] += len(rows)
            remaining -= len(rows)
            microbatch_index += 1
        if beat_total is None:
            raise RuntimeError("V8.1 step has no BEAT2 core batch")
        hanyang_total = None
        hanyang_metrics: dict[str, float] = {}
        hanyang_clip_ids: list[str] = []
        treatment = arm == "winner_isolated_5pct_hanyang"
        if hanyang_slot and treatment:
            rows, _plan = hanyang_sampler.sample_microbatch(
                remaining_effective_batch=1,
                semantic_tokens=int(model.semantic_tokens),
                max_batch_size=1,
                batching=v7_training["batching"],
            )
            if len(rows) != 1:
                raise RuntimeError("Hanyang slot exceeded one episode")
            batch = collate_confidence_weighted_18d(
                rows,
                buckets=v7_training["batching"]["length_buckets"],
                action_stats=runtime["selected_checkpoint"]["action_stats"],
                device=device,
            )
            losses = hanyang_partial_motion_objective(
                model,
                batch["actions"],
                batch["observation_confidence"],
                batch["durations_sec"],
                batch["frame_valid_mask"],
                loss_weights=hanyang_loss_weights,
                generator=_manual_generator(
                    device,
                    int(values["training"]["noise_seed"])
                    + current_step * 3_000_017
                    + 41,
                ),
            )
            hanyang_total = losses["total"] / 16.0
            hanyang_metrics = {
                name: float(value.detach().cpu()) / 16.0
                for name, value in losses.items()
            }
            hanyang_clip_ids = [str(rows[0]["clip_id"])]
            exposure["hanyang"] += 1
        elif hanyang_slot:
            exposure["matched_noop"] += 1
        expected_prefix_exposure = expected_exposure(
            current_step, arm=arm
        )
        if exposure != expected_prefix_exposure:
            raise RuntimeError(
                "V8.1 exposure diverged from the matched slot schedule"
            )
        guarded_motion_loss = motion_guard_total
        if hanyang_total is not None:
            guarded_motion_loss = guarded_motion_loss + hanyang_total
        route_disjoint_gradients(
            model=model,
            style_head=style_head,
            beat2_loss=beat_total,
            hanyang_loss=guarded_motion_loss,
        )
        gradient_norms = {
            "beat2_condition": _gradient_norm(beat_parameters),
            "beat2_style_head": _gradient_norm(style_parameters),
            "hanyang_motion": _gradient_norm(hanyang_parameters),
        }
        unclipped = float(
            torch.nn.utils.clip_grad_norm_(
                all_parameters, float(values["training"]["max_grad_norm"])
            )
        )
        if not math.isfinite(unclipped):
            raise FloatingPointError("non-finite V8.1 gradient norm")
        optimizer.step()
        model_ema.update(model)
        style_head_ema.update(style_head)
        should_diagnose = (
            current_step == 1
            or current_step
            % int(values["diagnostics"]["interval_steps"])
            == 0
            or (
                values["diagnostics"]["run_at_every_checkpoint"]
                and current_step
                % int(values["training"]["checkpoint_interval"])
                == 0
            )
            or current_step == run_end_step
        )
        diagnostic_record = None
        if should_diagnose:
            with model_ema.apply(model), style_head_ema.apply(style_head):
                current_diagnostics = _ema_validation_diagnostics(
                    model=model,
                    style_head=style_head,
                    semantic_perceptual=runtime["semantic_perceptual"],
                    beat2_rows=runtime["diagnostic_rows"],
                    hanyang_rows=runtime["hanyang_diagnostic_rows"],
                    negative_pool=runtime["negative_pool"],
                    action_stats=runtime["selected_checkpoint"][
                        "action_stats"
                    ],
                    v7_config=runtime["v7_config"],
                    device=device,
                    seed=diagnostic_seed,
                )
            diagnostic_gate = _diagnostic_candidate_gate(
                baseline=baseline_diagnostics,
                current=current_diagnostics,
                v7_config=runtime["v7_config"],
                diagnostics_config=values["diagnostics"],
            )
            diagnostic_record = {
                "step": current_step,
                "diagnostics": current_diagnostics,
                "gate": diagnostic_gate,
            }
            last_diagnostics = diagnostic_record
        last_event = {
            "event": "train_step",
            "arm": arm,
            "step": current_step,
            "beat2_quota": beat_quota,
            "hanyang_slot": hanyang_slot,
            "hanyang_examples": len(hanyang_clip_ids),
            "beat2": dict(beat_metrics),
            "hanyang": hanyang_metrics,
            "gradient_norms_before_clip": gradient_norms,
            "global_gradient_norm_before_clip": unclipped,
            "clip_max_norm": float(values["training"]["max_grad_norm"]),
            "learning_rates": {
                str(group["role"]): float(group["lr"])
                for group in optimizer.param_groups
            },
            "condition_response_anchor": beat_metrics[
                "emotion_response_anchor"
            ],
            "beat2_batch_receipts": beat_receipts,
            "hanyang_clip_ids": hanyang_clip_ids,
            "exposure": dict(exposure),
            "expected_exposure": expected_prefix_exposure,
            "elapsed_sec": time.monotonic() - started,
        }
        if diagnostic_record is not None:
            last_event["held_out_diagnostics"] = diagnostic_record
            if (
                not smoke_test
                and not audit_replay
                and diagnostic_record["gate"]["admissible"]
                and (
                best_admissible is None
                or float(diagnostic_record["gate"]["selection_score"])
                > float(best_admissible["gate"]["selection_score"])
                )
            ):
                best_admissible = deepcopy(diagnostic_record)
                best_admissible["checkpoint_pending_write"] = True
        should_save = (
            current_step == 1
            or current_step
            % int(values["training"]["checkpoint_interval"])
            == 0
            or current_step == run_end_step
        )
        should_log = (
            current_step == 1
            or current_step % int(values["training"]["log_interval"]) == 0
            or current_step == run_end_step
        )
        if should_log:
            _append_jsonl(last_event, paths["progress"])
        if should_save:
            _save_runner_state(
                paths["state"],
                step=current_step,
                target_steps=target_steps,
                input_contract_sha256=input_contract["sha256"],
                model=model,
                style_head=style_head,
                model_ema=model_ema,
                style_head_ema=style_head_ema,
                optimizer=optimizer,
                beat2_sampler=beat2_sampler,
                hanyang_sampler=hanyang_sampler,
                exposure=exposure,
                last_event=last_event,
                baseline_diagnostics=baseline_diagnostics,
                last_diagnostics=last_diagnostics,
                best_admissible=best_admissible,
            )
            payload = _runner_checkpoint_payload(
                config=values,
                arm=arm,
                step=current_step,
                target_steps=target_steps,
                input_contract=input_contract,
                model_ema=model_ema,
                style_head_ema=style_head_ema,
                selected_checkpoint=runtime["selected_checkpoint"],
                exposure=exposure,
                last_event=last_event,
                smoke_test=smoke_test,
                checkpoint_role="last_audit_only",
                diagnostic_gate=(
                    last_diagnostics["gate"]
                    if last_diagnostics is not None
                    else None
                ),
            )
            if audit_replay:
                payload.update(
                    {
                        "audit_replay": True,
                        "audit_stop_step": int(run_end_step),
                        "audit_purpose": (
                            "matched_step_deterministic_replay_not_formal_candidate"
                        ),
                        "candidate_eligible": False,
                    }
                )
            _atomic_torch(payload, paths["checkpoint"])
        if (
            best_admissible is not None
            and best_admissible.pop("checkpoint_pending_write", False)
        ):
            best_payload = _runner_checkpoint_payload(
                config=values,
                arm=arm,
                step=current_step,
                target_steps=target_steps,
                input_contract=input_contract,
                model_ema=model_ema,
                style_head_ema=style_head_ema,
                selected_checkpoint=runtime["selected_checkpoint"],
                exposure=exposure,
                last_event=last_event,
                smoke_test=smoke_test,
                checkpoint_role="best_admissible",
                diagnostic_gate=best_admissible["gate"],
            )
            _atomic_torch(best_payload, paths["best"])
            best_admissible["checkpoint"] = str(paths["best"])
            best_admissible["checkpoint_sha256"] = sha256_file(
                paths["best"]
            )
            _save_runner_state(
                paths["state"],
                step=current_step,
                target_steps=target_steps,
                input_contract_sha256=input_contract["sha256"],
                model=model,
                style_head=style_head,
                model_ema=model_ema,
                style_head_ema=style_head_ema,
                optimizer=optimizer,
                beat2_sampler=beat2_sampler,
                hanyang_sampler=hanyang_sampler,
                exposure=exposure,
                last_event=last_event,
                baseline_diagnostics=baseline_diagnostics,
                last_diagnostics=last_diagnostics,
                best_admissible=best_admissible,
            )
    final_expected_exposure = expected_exposure(run_end_step, arm=arm)
    if exposure != final_expected_exposure:
        raise RuntimeError("final V8.1 exposure contract mismatch")
    total_slots = run_end_step * 16
    actual_slot_fractions = {
        name: count / float(total_slots)
        for name, count in exposure.items()
    }
    arm_contract = values["winner_overlay_arms"][arm]
    declared_fraction_assertion_applicable = run_end_step % 5 == 0
    declared_fraction_assertion_passed = all(
        math.isclose(
            actual_slot_fractions[name],
            float(arm_contract[f"{name}_fraction"]),
            abs_tol=1e-12,
        )
        for name in ("beat2", "hanyang", "matched_noop")
    )
    if (
        declared_fraction_assertion_applicable
        and not declared_fraction_assertion_passed
    ):
        raise RuntimeError("long-run V8.1 slot fractions changed")
    checkpoint_sha256 = sha256_file(paths["checkpoint"])
    last_gate = (
        last_diagnostics["gate"] if last_diagnostics is not None else None
    )
    last_status = (
        "admissible"
        if last_gate is not None and last_gate["admissible"]
        else "rejected"
    )
    candidate_available = best_admissible is not None and paths[
        "best"
    ].is_file()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": SUMMARY_ARTIFACT_KIND,
        "arm": arm,
        "completed_steps": run_end_step,
        "last_checkpoint": str(paths["checkpoint"]),
        "last_checkpoint_sha256": checkpoint_sha256,
        "last_checkpoint_status": last_status,
        "last_diagnostics": last_diagnostics,
        "baseline_diagnostics": baseline_diagnostics,
        "best_admissible": best_admissible,
        "candidate_available": candidate_available,
        "run_status": (
            "technical_smoke_completed_not_candidate"
            if smoke_test
            else (
                "matched_step_audit_replay_completed_not_candidate"
                if audit_replay
                else (
                    "candidate_available"
                    if candidate_available
                    else "rejected_no_admissible_checkpoint"
                )
            )
        ),
        "state": str(paths["state"]),
        "state_sha256": sha256_file(paths["state"]),
        "input_contract": input_contract,
        "exposure": exposure,
        "expected_exposure": final_expected_exposure,
        "declared_slot_fractions": {
            "beat2": float(arm_contract["beat2_fraction"]),
            "hanyang": float(arm_contract["hanyang_fraction"]),
            "matched_noop": float(
                arm_contract["matched_noop_fraction"]
            ),
        },
        "actual_slot_fractions": actual_slot_fractions,
        "declared_fraction_assertion_applicable": (
            declared_fraction_assertion_applicable
        ),
        "declared_fraction_assertion_passed": (
            declared_fraction_assertion_passed
            if declared_fraction_assertion_applicable
            else None
        ),
        "prefix_schedule_assertion_passed": True,
        "last_event": last_event,
        "smoke_test": bool(smoke_test),
        "formal_release_eligible": False,
    }
    if audit_replay:
        summary.update(
            {
                "target_steps": int(target_steps),
                "audit_replay": True,
                "audit_stop_step": int(run_end_step),
                "audit_purpose": (
                    "matched_step_deterministic_replay_not_formal_candidate"
                ),
                "candidate_available": False,
            }
        )
    summary.update(_hanyang_artifact_audit(values))
    _atomic_json(summary, paths["summary"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preflight the fail-closed emotion-preserving V8.1 pair"
    )
    parser.add_argument(
        "--config",
        default=str(
            PROJECT_ROOT
            / "configs"
            / "hanyang_beat2_emotion_preserving_v81.json"
        ),
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="validate and print paired lineage; never starts training",
    )
    parser.add_argument(
        "--arm",
        choices=(
            "winner_control_0pct_hanyang",
            "winner_isolated_5pct_hanyang",
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-output-dir")
    parser.add_argument("--audit-stop-step", type=int)
    parser.add_argument("--audit-output-dir")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    config = read_config(args.config)
    plans = {
        arm: build_lineage_contract(config, arm=arm)
        for arm in (
            "winner_control_0pct_hanyang",
            "winner_isolated_5pct_hanyang",
        )
    }
    print(json.dumps(plans, ensure_ascii=False, indent=2, sort_keys=True))
    if args.preflight:
        return
    if args.arm is None:
        parser.error("--arm is required unless --preflight is used")
    summary = run_arm(
        config,
        arm=args.arm,
        resume=bool(args.resume),
        overwrite=bool(args.overwrite),
        smoke_test=bool(args.smoke_test),
        smoke_output_dir=args.smoke_output_dir,
        audit_stop_step=args.audit_stop_step,
        audit_output_dir=args.audit_output_dir,
        device_override=args.device,
    )
    print(
        json.dumps(
            summary, ensure_ascii=False, indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
