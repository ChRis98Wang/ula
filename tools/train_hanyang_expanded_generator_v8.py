#!/usr/bin/env python3
"""Isolated BEAT2 + partial-Hanyang expanded-generator experiment V8.

The persistent run is fail-closed behind explicit human review.  ``--smoke-test``
is the only mode allowed while the review gate is closed; it executes two
technical optimizer steps and cannot produce a release-eligible checkpoint.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from upper_body_skeleton.data_source_registry import (  # noqa: E402
    KIMODO_PERMANENT_DENY_POLICY,
    assert_no_forbidden_data_lineage,
)
from upper_body_skeleton.hanyang_emotion_retarget import (  # noqa: E402
    sha256_file,
)
from upper_body_skeleton.hanyang_expanded_generator import (  # noqa: E402
    DATA_CONTRACT_ARTIFACT_KIND,
    EXPERIMENT_ARTIFACT_KIND,
    HANYANG_DOMAIN_MAX_FRACTION,
    SCHEMA_VERSION,
    action_normalizer_contract,
    build_experimental_admission_contract,
    canonical_sha256,
    collate_confidence_weighted_18d,
    confidence_weighted_18d_objective,
    load_beat2_frozen_qwen_motion_episodes,
    load_hanyang_partial_motion_episodes,
    split_expanded_episodes,
)
from upper_body_skeleton.ula_training import (  # noqa: E402
    KIMODO_V2_CONDITION_DIM,
)
from upper_body_skeleton.ula_v2_18d_head import (  # noqa: E402
    ACTION_DIM,
    load_contract_checkpoint,
)
from upper_body_skeleton.ula_v2_18d_posttrain import (  # noqa: E402
    ModelEMA,
    NativeLengthBucketSampler,
)
from upper_body_skeleton.ula_v2_18d_random_init import (  # noqa: E402
    ULA_MMDIT_V3_ADALN_ARCHITECTURE,
)


CONFIG_ARTIFACT_KIND = "hanyang_beat2_expanded_generator_config_v8"
STATE_ARTIFACT_KIND = "hanyang_beat2_expanded_generator_state_v8"
CHECKPOINT_ARTIFACT_KIND = "hanyang_beat2_expanded_generator_checkpoint_v8"
SUMMARY_ARTIFACT_KIND = "hanyang_beat2_expanded_generator_summary_v8"
HUMAN_REVIEW_BLOCKED = "HUMAN_REVIEW_BLOCKED"
KNOWN_OUTPUTS = (
    "last_state_v8.pt",
    "latest_generator_v8.pt",
    "generator_expanded_v8.pt",
    "training_summary_v8.json",
    "progress_v8.jsonl",
)
PREPARED_OUTPUTS = (
    "admission_contract_v8.json",
    "data_contract_v8.json",
    "split_contract_v8.json",
    "normalizer_contract_v8.json",
)


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


def _atomic_torch(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _append_jsonl(value: Mapping[str, Any], path: Path) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _resolved_file(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value).expanduser().resolve()
    assert_no_forbidden_data_lineage(
        {f"{field}_path": str(path)}, context="v8_config"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def _resolved_output(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("output_dir must be a non-empty path")
    path = Path(value).expanduser().resolve()
    assert_no_forbidden_data_lineage(
        {"output_dir": str(path)}, context="v8_config"
    )
    return str(path)


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    values = deepcopy(dict(config))
    exact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": CONFIG_ARTIFACT_KIND,
        "experiment_artifact_kind": EXPERIMENT_ARTIFACT_KIND,
        "experimental_only": True,
        "formal_release_eligible": False,
        "data_policy": (
            "beat2_plus_hanyang_partial_motion_experiment_no_external_"
            "foundation_v1"
        ),
        "condition_policy": (
            "beat2_frozen_qwen_128d_hanyang_all_zero_unconditional_v1"
        ),
        "deny_policy": KIMODO_PERMANENT_DENY_POLICY,
    }
    for field, expected in exact.items():
        if values.get(field) != expected:
            raise ValueError(f"{field} must be {expected!r}")

    input_fields = (
        "foundation_checkpoint",
        "beat2_manifest",
        "foundation_condition_cache",
        "frozen_qwen_cache",
        "hanyang_passed_manifest",
        "hanyang_batch_receipt",
        "hanyang_pool_manifest",
        "hanyang_pool_receipt",
    )
    for field in input_fields:
        values[field] = _resolved_file(values.get(field), field=field)
        expected = values.get(f"expected_{field}_sha256")
        if not _is_sha256(expected):
            raise ValueError(f"expected_{field}_sha256 must be a SHA256")
        if sha256_file(values[field]) != expected:
            raise ValueError(f"{field} SHA256 mismatch")
    values["output_dir"] = _resolved_output(values.get("output_dir"))
    values["smoke_output_dir"] = _resolved_output(
        values.get("smoke_output_dir")
    )
    if Path(values["output_dir"]) == Path(values["smoke_output_dir"]):
        raise ValueError("smoke and persistent output directories must differ")

    review = values.get("human_review")
    if not isinstance(review, Mapping) or set(review) != {
        "required",
        "approved",
        "approval_receipt",
        "expected_approval_receipt_sha256",
        "sample_bundle",
    }:
        raise ValueError("human_review fields changed")
    if review.get("required") is not True:
        raise ValueError("human review must remain required")
    approved = review.get("approved")
    if type(approved) is not bool:
        raise ValueError("human_review.approved must be boolean")
    sample_bundle = review.get("sample_bundle")
    if not isinstance(sample_bundle, Mapping) or set(sample_bundle) != {
        "review_manifest",
        "expected_review_manifest_sha256",
        "review_video_reel",
        "expected_review_video_reel_sha256",
        "review_queue",
        "expected_review_queue_sha256",
        "review_bundle_receipt",
        "expected_review_bundle_receipt_sha256",
        "cross_split_selection_clip_ids_sha256",
        "visual_evidence_clip_ids_sha256",
    }:
        raise ValueError("human-review sample bundle fields changed")
    review_manifest = _resolved_file(
        sample_bundle.get("review_manifest"),
        field="human_review.sample_bundle.review_manifest",
    )
    if (
        not _is_sha256(sample_bundle.get("expected_review_manifest_sha256"))
        or sha256_file(review_manifest)
        != sample_bundle["expected_review_manifest_sha256"]
        or not _is_sha256(
            sample_bundle.get("cross_split_selection_clip_ids_sha256")
        )
    ):
        raise ValueError("human-review manifest binding is invalid")
    review_rows = []
    with Path(review_manifest).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                not isinstance(row, Mapping)
                or row.get("training_launch_allowed") is not False
                or row.get("human_review_approved") is not False
                or row.get("review_status") != "pending"
            ):
                raise ValueError(
                    f"blocked review row changed at line {line_number}"
                )
            review_rows.append(dict(row))
    review_clip_ids = sorted(
        str(row.get("clip_id") or "") for row in review_rows
    )
    review_clip_ids_sha256 = canonical_sha256(review_clip_ids)
    if (
        len(review_rows) != 21
        or "" in review_clip_ids
        or len(set(review_clip_ids)) != len(review_clip_ids)
        or review_clip_ids_sha256
        != sample_bundle["cross_split_selection_clip_ids_sha256"]
    ):
        raise ValueError("human-review clip gate is incomplete or changed")
    sample_bundle = dict(sample_bundle)
    sample_bundle["review_manifest"] = review_manifest
    evidence_fields = (
        "review_video_reel",
        "expected_review_video_reel_sha256",
        "review_queue",
        "expected_review_queue_sha256",
        "review_bundle_receipt",
        "expected_review_bundle_receipt_sha256",
        "visual_evidence_clip_ids_sha256",
    )
    evidence_ready = all(
        sample_bundle.get(field) is not None for field in evidence_fields
    )
    if approved and not evidence_ready:
        raise ValueError("approved config requires the complete visual evidence")
    if evidence_ready:
        bundle_receipt_path = _resolved_file(
            sample_bundle.get("review_bundle_receipt"),
            field="human_review.sample_bundle.review_bundle_receipt",
        )
        if (
            not _is_sha256(
                sample_bundle.get(
                    "expected_review_bundle_receipt_sha256"
                )
            )
            or sha256_file(bundle_receipt_path)
            != sample_bundle[
                "expected_review_bundle_receipt_sha256"
            ]
        ):
            raise ValueError("human-review bundle receipt binding is invalid")
        review_video = _resolved_file(
            sample_bundle.get("review_video_reel"),
            field="human_review.sample_bundle.review_video_reel",
        )
        if (
            not _is_sha256(
                sample_bundle.get("expected_review_video_reel_sha256")
            )
            or sha256_file(review_video)
            != sample_bundle["expected_review_video_reel_sha256"]
        ):
            raise ValueError("human-review video binding is invalid")
        sample_bundle["review_video_reel"] = review_video
        review_queue = _resolved_file(
            sample_bundle.get("review_queue"),
            field="human_review.sample_bundle.review_queue",
        )
        if (
            not _is_sha256(
                sample_bundle.get("expected_review_queue_sha256")
            )
            or sha256_file(review_queue)
            != sample_bundle["expected_review_queue_sha256"]
        ):
            raise ValueError("human-review queue binding is invalid")
        sample_bundle["review_queue"] = review_queue
        bundle_receipt = json.loads(
            Path(bundle_receipt_path).read_text(encoding="utf-8")
        )
        if not isinstance(bundle_receipt, Mapping):
            raise ValueError("human-review bundle receipt must be an object")
        bundle_record_hash = canonical_sha256(
            {
                key: value
                for key, value in bundle_receipt.items()
                if key != "record_sha256"
            }
        )
        if (
            bundle_receipt.get("artifact_kind")
            != "hanyang_training_sample_human_review_bundle_v1"
            or bundle_receipt.get("record_sha256")
            != bundle_record_hash
            or bundle_receipt.get("cross_split_review_manifest_sha256")
            != sample_bundle["expected_review_manifest_sha256"]
            or bundle_receipt.get("reel_sha256")
            != sample_bundle["expected_review_video_reel_sha256"]
            or bundle_receipt.get("review_queue_sha256")
            != sample_bundle["expected_review_queue_sha256"]
            or bundle_receipt.get("sample_count") != 14
            or bundle_receipt.get("partial_motion_only_sample_count") != 14
            or bundle_receipt.get(
                "emotion_condition_eligible_sample_count"
            )
            != 0
            or bundle_receipt.get("human_review_approved") is not False
            or bundle_receipt.get("training_launch_allowed") is not False
            or bundle_receipt.get(
                "render_pass_grants_training_admission"
            )
            is not False
        ):
            raise ValueError("human-review bundle receipt contract changed")
        queue_rows = []
        with Path(review_queue).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise ValueError(
                        f"invalid review queue row {line_number}"
                    )
                row_hash = canonical_sha256(
                    {
                        key: value
                        for key, value in row.items()
                        if key != "record_sha256"
                    }
                )
                assert_no_forbidden_data_lineage(
                    row, context=f"human_review_queue[{line_number}]"
                )
                if (
                    row.get("artifact_kind")
                    != "hanyang_training_sample_review_item_v1"
                    or row.get("record_sha256") != row_hash
                    or row.get("training_lane") != "partial_motion_only"
                    or row.get("emotion_condition_eligible") is not False
                    or row.get("accepted_for_training") is not False
                    or row.get(
                        "render_pass_grants_training_admission"
                    )
                    is not False
                    or sha256_file(row["source_video"])
                    != row["source_video_sha256"]
                    or sha256_file(row["robot_video"])
                    != row["robot_video_sha256"]
                    or sha256_file(row["review_segment"])
                    != row["review_segment_sha256"]
                    or sha256_file(row["source_csv"])
                    != row["source_csv_sha256"]
                    or sha256_file(
                        row["source_faithful_partial_18d_csv"]
                    )
                    != row[
                        "source_faithful_partial_18d_csv_sha256"
                    ]
                    or sha256_file(row["quality_json"])
                    != row["quality_json_sha256"]
                ):
                    raise ValueError(
                        f"review queue lineage changed at row {line_number}"
                    )
                queue_rows.append(dict(row))
        visual_clip_ids = sorted(
            str(row["clip_id"]) for row in queue_rows
        )
        if (
            len(queue_rows) != 14
            or canonical_sha256(visual_clip_ids)
            != sample_bundle["visual_evidence_clip_ids_sha256"]
        ):
            raise ValueError("visual-evidence clip set changed")
        sample_bundle["review_bundle_receipt"] = bundle_receipt_path
    elif any(
        sample_bundle.get(field) is not None for field in evidence_fields
    ):
        raise ValueError(
            "blocked config cannot predeclare an unverified review video"
        )
    review = dict(review)
    review["sample_bundle"] = sample_bundle
    values["human_review"] = review
    if approved:
        approval_receipt = _resolved_file(
            review.get("approval_receipt"),
            field="human_review.approval_receipt",
        )
        expected_approval = review.get(
            "expected_approval_receipt_sha256"
        )
        if (
            not _is_sha256(expected_approval)
            or sha256_file(approval_receipt) != expected_approval
        ):
            raise ValueError("human-review approval receipt is invalid")
        approval = json.loads(
            Path(approval_receipt).read_text(encoding="utf-8")
        )
        if not isinstance(approval, Mapping):
            raise ValueError("human-review approval receipt must be an object")
        reviewed_by = str(approval.get("reviewed_by") or "").strip()
        reviewed_utc = str(approval.get("reviewed_utc") or "").strip()
        try:
            reviewed_time = datetime.fromisoformat(
                reviewed_utc.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError("human-review approval UTC is invalid") from exc
        approval_record_hash = canonical_sha256(
            {
                key: value
                for key, value in approval.items()
                if key != "record_sha256"
            }
        )
        if (
            approval.get("artifact_kind")
            != "hanyang_training_sample_approval_gate_v1"
            or approval.get("record_sha256") != approval_record_hash
            or approval.get("decision") != "approved"
            or approval.get("human_review_approved") is not True
            or approval.get("training_launch_allowed") is not True
            or not reviewed_by
            or reviewed_time.tzinfo is None
            or not str(approval.get("decision_notes") or "").strip()
            or approval.get("cross_split_review_manifest_sha256")
            != sample_bundle["expected_review_manifest_sha256"]
            or approval.get("review_reel_sha256")
            != sample_bundle["expected_review_video_reel_sha256"]
            or approval.get("review_queue_sha256")
            != sample_bundle["expected_review_queue_sha256"]
            or approval.get("review_bundle_sha256")
            != sample_bundle[
                "expected_review_bundle_receipt_sha256"
            ]
            or sorted(
                str(value) for value in approval.get("proposed_clip_ids") or ()
            )
            != visual_clip_ids
            or sorted(
                str(value) for value in approval.get("accepted_clip_ids") or ()
            )
            != visual_clip_ids
            or list(approval.get("rejected_clip_ids") or ()) != []
            or approval.get(
                "dataset_partial_motion_lane_approved"
            )
            is not True
            or approval.get("emotion_conditioning_approved") is not False
        ):
            raise ValueError("human-review approval content is not fully bound")
        review = dict(review)
        review["approval_receipt"] = approval_receipt
        values["human_review"] = review
    elif (
        review.get("approval_receipt") is not None
        or review.get("expected_approval_receipt_sha256") is not None
    ):
        raise ValueError("unapproved config cannot carry an approval receipt")

    training = values.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("training must be a mapping")
    required_training = {
        "steps",
        "batch_size",
        "lr",
        "minimum_lr_ratio",
        "warmup_steps",
        "weight_decay",
        "adam_eps",
        "max_grad_norm",
        "ema_decay",
        "validation_interval",
        "checkpoint_interval",
        "export_interval",
        "log_interval",
        "evaluation_episode_count",
        "condition_dropout_probability",
        "hanyang_domain_fraction",
        "hanyang_schedule_period_steps",
        "hanyang_active_steps_per_period",
        "hanyang_examples_per_active_step",
        "loss",
        "batching",
        "sampler",
    }
    if set(training) != required_training:
        raise ValueError("training fields changed")
    integers = (
        "steps",
        "batch_size",
        "warmup_steps",
        "validation_interval",
        "checkpoint_interval",
        "export_interval",
        "log_interval",
        "evaluation_episode_count",
        "hanyang_schedule_period_steps",
        "hanyang_active_steps_per_period",
        "hanyang_examples_per_active_step",
    )
    if any(
        type(training[name]) is not int or training[name] <= 0
        for name in integers
    ):
        raise ValueError("training integer fields must be positive")
    if int(training["batch_size"]) != 16:
        raise ValueError("v8 safety contract requires effective batch size 16")
    period = int(training["hanyang_schedule_period_steps"])
    active = int(training["hanyang_active_steps_per_period"])
    per_active = int(training["hanyang_examples_per_active_step"])
    batch_size = int(training["batch_size"])
    if int(training["steps"]) % period != 0:
        raise ValueError("training steps must contain complete Hanyang periods")
    if int(training["warmup_steps"]) >= int(training["steps"]):
        raise ValueError("warmup_steps must be smaller than training steps")
    for interval in (
        "validation_interval",
        "checkpoint_interval",
        "export_interval",
        "log_interval",
    ):
        if int(training[interval]) > int(training["steps"]):
            raise ValueError(f"{interval} exceeds training steps")
    scalar_bounds = {
        "lr": (0.0, 2e-5, False),
        "minimum_lr_ratio": (0.0, 1.0, False),
        "weight_decay": (0.0, 0.01, True),
        "adam_eps": (0.0, 1e-3, False),
        "max_grad_norm": (0.0, 10.0, False),
        "ema_decay": (0.9, 1.0, False),
        "condition_dropout_probability": (0.0, 0.2, True),
    }
    for name, (lower, upper, lower_inclusive) in scalar_bounds.items():
        value = training[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite scalar")
        numeric = float(value)
        lower_ok = numeric >= lower if lower_inclusive else numeric > lower
        if not math.isfinite(numeric) or not lower_ok or numeric > upper:
            raise ValueError(f"{name} is outside the safe V8 range")
    if float(training["ema_decay"]) >= 1.0:
        raise ValueError("ema_decay must be smaller than one")
    if per_active > 1:
        raise ValueError("at most one Hanyang episode is allowed per batch of 16")
    exposure = active * per_active / float(period * batch_size)
    configured_fraction = float(training["hanyang_domain_fraction"])
    if (
        active > period
        or not math.isclose(exposure, configured_fraction, abs_tol=1e-12)
        or configured_fraction > HANYANG_DOMAIN_MAX_FRACTION
    ):
        raise ValueError("Hanyang schedule does not match the safe exposure cap")
    if not math.isclose(configured_fraction, 0.05, abs_tol=1e-12):
        raise ValueError("first V8 A/B must use the configured 5% exposure")
    if not isinstance(training["loss"], Mapping) or set(training["loss"]) != {
        "flow",
        "position",
        "velocity",
        "acceleration",
        "jerk",
    }:
        raise ValueError("V8 loss contract changed")
    loss_values = []
    for name, value in training["loss"].items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"loss.{name} must be a finite scalar")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError("V8 loss weights must be finite and nonnegative")
        loss_values.append(numeric)
    if sum(loss_values) <= 0.0:
        raise ValueError("V8 loss weights cannot all be zero")
    batching = training["batching"]
    if (
        not isinstance(batching, Mapping)
        or set(batching)
        != {
            "mode",
            "length_buckets",
            "homogeneous_bucket_batches",
            "max_motion_tokens_per_microbatch",
            "max_attention_elements_per_microbatch",
            "oversize_sequence_policy",
        }
        or batching.get("mode") != "native_variable_length"
        or batching.get("homogeneous_bucket_batches") is not True
        or batching.get("oversize_sequence_policy")
        != "single_full_episode_or_fail"
    ):
        raise ValueError("native batching contract changed")
    buckets = batching["length_buckets"]
    if (
        not isinstance(buckets, list)
        or not buckets
        or any(type(value) is not int or value < 4 for value in buckets)
        or buckets != sorted(set(buckets))
    ):
        raise ValueError("length_buckets must be unique increasing integers")
    for field in (
        "max_motion_tokens_per_microbatch",
        "max_attention_elements_per_microbatch",
    ):
        if type(batching[field]) is not int or batching[field] <= 0:
            raise ValueError(f"{field} must be a positive integer")
    sampler = training["sampler"]
    if sampler != {"mode": "domain_speaker"}:
        raise ValueError("V8 sampler must be domain_speaker")
    if type(values.get("seed")) is not int:
        raise ValueError("seed must be an integer")
    if values.get("device") not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    assert_no_forbidden_data_lineage(values, context="v8_config")
    return values


def _paths(config: Mapping[str, Any], *, smoke_test: bool) -> dict[str, Path]:
    root = Path(
        config["smoke_output_dir"] if smoke_test else config["output_dir"]
    )
    return {
        "root": root,
        "prepared": root / "prepared",
        "state": root / "last_state_v8.pt",
        "latest": root / "latest_generator_v8.pt",
        "checkpoint": root / "generator_expanded_v8.pt",
        "summary": root / "training_summary_v8.json",
        "progress": root / "progress_v8.jsonl",
    }


def _write_contract(value: Mapping[str, Any], path: Path) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value:
            raise ValueError(f"prepared contract changed: {path}")
    else:
        _atomic_json(value, path)


def _prepare(
    config: Mapping[str, Any],
    *,
    paths: Mapping[str, Path],
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    beat2, beat2_receipt = load_beat2_frozen_qwen_motion_episodes(
        config["beat2_manifest"],
        config["foundation_condition_cache"],
        config["frozen_qwen_cache"],
        expected_manifest_sha256=config[
            "expected_beat2_manifest_sha256"
        ],
        expected_foundation_condition_cache_sha256=config[
            "expected_foundation_condition_cache_sha256"
        ],
        expected_frozen_qwen_cache_sha256=config[
            "expected_frozen_qwen_cache_sha256"
        ],
    )
    hanyang, hanyang_receipt = load_hanyang_partial_motion_episodes(
        config["hanyang_pool_manifest"],
        expected_manifest_sha256=config[
            "expected_hanyang_pool_manifest_sha256"
        ],
        pool_receipt_path=config["hanyang_pool_receipt"],
        expected_pool_receipt_sha256=config[
            "expected_hanyang_pool_receipt_sha256"
        ],
        expected_upstream_passed_manifest_sha256=config[
            "expected_hanyang_passed_manifest_sha256"
        ],
    )
    splits, split_contract = split_expanded_episodes(beat2, hanyang)
    admission = build_experimental_admission_contract(
        hanyang_loader_receipt=hanyang_receipt,
        beat2_loader_receipt=beat2_receipt,
        hanyang_batch_receipt_path=config["hanyang_batch_receipt"],
        expected_hanyang_batch_receipt_sha256=config[
            "expected_hanyang_batch_receipt_sha256"
        ],
        hanyang_domain_fraction=float(
            config["training"]["hanyang_domain_fraction"]
        ),
        human_review=config["human_review"],
    )
    admission = dict(admission)
    admission.update(
        {
            "human_review_required": True,
            "human_review_approved": bool(
                config["human_review"]["approved"]
            ),
            "training_gate": (
                "APPROVED"
                if config["human_review"]["approved"]
                else HUMAN_REVIEW_BLOCKED
            ),
            "sample_bundle": config["human_review"]["sample_bundle"],
            "maximum_hanyang_examples_per_effective_batch_16": 1,
            "target_episode_exposure_fraction": float(
                config["training"]["hanyang_domain_fraction"]
            ),
            "natural_train_pool_fraction": (
                len(splits["train"]["hanyang"])
                / float(
                    len(splits["train"]["hanyang"])
                    + len(splits["train"]["beat2"])
                )
            ),
        }
    )
    admission["sha256"] = canonical_sha256(
        {key: value for key, value in admission.items() if key != "sha256"}
    )
    data_contract = {
        "artifact_kind": DATA_CONTRACT_ARTIFACT_KIND,
        "schema_version": SCHEMA_VERSION,
        "experimental_only": True,
        "formal_release_eligible": False,
        "beat2_loader_receipt": beat2_receipt,
        "hanyang_loader_receipt": hanyang_receipt,
        "split_contract_sha256": split_contract["sha256"],
        "admission_contract_sha256": admission["sha256"],
        "source_whitelist_exact": [
            "beat2_official_semantic_event_training_pool_v7",
            "hanyang_duksung_emotional_body_motion_v1",
        ],
        "hanyang_loss_policy": (
            "per_frame_per_dimension_confidence_with_adjacent_min_derivatives"
        ),
        "hanyang_condition_policy": "all_zero_unconditional",
        "beat2_condition_policy": "frozen_qwen_128d_no_lora",
        "hanyang_emotion_style_semantic_duration_losses": "all_disabled",
        "no_seven_class_claim": True,
        "insufficient_heldout_emotion_coverage": True,
        "deny_policy": KIMODO_PERMANENT_DENY_POLICY,
    }
    assert_no_forbidden_data_lineage(data_contract, context="v8_data")
    data_contract["sha256"] = canonical_sha256(data_contract)
    paths["prepared"].mkdir(parents=True, exist_ok=True)
    _write_contract(
        admission, paths["prepared"] / PREPARED_OUTPUTS[0]
    )
    _write_contract(
        data_contract, paths["prepared"] / PREPARED_OUTPUTS[1]
    )
    _write_contract(
        split_contract, paths["prepared"] / PREPARED_OUTPUTS[2]
    )
    return splits, {
        "admission": admission,
        "data": data_contract,
        "split": split_contract,
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _manual_generator(
    device: torch.device, seed: int
) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _lr_scale(
    step: int, *, total_steps: int, warmup_steps: int, minimum_ratio: float
) -> float:
    if step <= warmup_steps:
        return max(minimum_ratio, step / float(max(1, warmup_steps)))
    progress = (step - warmup_steps) / float(
        max(1, total_steps - warmup_steps)
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def _hanyang_quota(step: int, training: Mapping[str, Any]) -> int:
    period = int(training["hanyang_schedule_period_steps"])
    active = int(training["hanyang_active_steps_per_period"])
    cursor = (int(step) - 1) % period
    return (
        int(training["hanyang_examples_per_active_step"])
        if cursor < active
        else 0
    )


def _batch_objective(
    model: torch.nn.Module,
    episodes: Sequence[Mapping[str, Any]],
    *,
    source: str,
    action_stats: Mapping[str, Any],
    training: Mapping[str, Any],
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor]:
    batch = collate_confidence_weighted_18d(
        episodes,
        buckets=training["batching"]["length_buckets"],
        action_stats=action_stats,
        device=device,
    )
    conditions = batch["conditions"]
    if source == "beat2":
        generator = _manual_generator(device, seed + 17)
        keep = (
            torch.rand(
                len(episodes), device=device, generator=generator
            )
            >= float(training["condition_dropout_probability"])
        )
        conditions = conditions * keep[:, None]
    elif source == "hanyang":
        if torch.count_nonzero(conditions).item() != 0:
            raise ValueError("Hanyang generator condition must remain all zero")
    else:
        raise ValueError(f"unknown V8 source: {source}")
    flow_generator = _manual_generator(device, seed + 31)
    return confidence_weighted_18d_objective(
        model,
        batch["actions"],
        conditions,
        batch["observation_confidence"],
        batch["durations_sec"],
        batch["frame_valid_mask"],
        loss_weights=training["loss"],
        generator=flow_generator,
        require_hanyang_partial_weights=(source == "hanyang"),
    )


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    splits: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    *,
    action_stats: Mapping[str, Any],
    training: Mapping[str, Any],
    device: torch.device,
    seed: int,
) -> dict[str, dict[str, float]]:
    model.eval()
    output: dict[str, dict[str, float]] = {}
    count = int(training["evaluation_episode_count"])
    for offset, source in enumerate(("beat2", "hanyang")):
        episodes = sorted(
            splits["validation"][source], key=lambda row: str(row["clip_id"])
        )[:count]
        losses = _batch_objective(
            model,
            episodes,
            source=source,
            action_stats=action_stats,
            training=training,
            device=device,
            seed=seed + offset * 1009,
        )
        output[source] = {
            name: float(value.detach().cpu())
            for name, value in losses.items()
        }
    model.train()
    return output


def _checkpoint_payload(
    *,
    model_state: Mapping[str, torch.Tensor],
    foundation: Mapping[str, Any],
    contracts: Mapping[str, Any],
    input_contract: Mapping[str, Any],
    config: Mapping[str, Any],
    step: int,
    target_steps: int,
    exposure: Mapping[str, int],
    smoke_test: bool,
) -> dict[str, Any]:
    total_exposure = int(exposure["beat2"]) + int(exposure["hanyang"])
    lineage = {
        "input_contract_sha256": input_contract["sha256"],
        "config_sha256": input_contract["config_sha256"],
        "approval_receipt_sha256": (
            config["human_review"]["expected_approval_receipt_sha256"]
            if config["human_review"]["approved"]
            else None
        ),
        "sources": {
            "beat2": {
                "dataset_source": (
                    "beat2_official_semantic_event_training_pool_v7"
                ),
                "manifest_sha256": config[
                    "expected_beat2_manifest_sha256"
                ],
                "split_counts": deepcopy(
                    contracts["split"]["source_counts"]["beat2"]
                ),
            },
            "hanyang": {
                "dataset_source": (
                    "hanyang_duksung_emotional_body_motion_v1"
                ),
                "upstream_passed_manifest_sha256": config[
                    "expected_hanyang_passed_manifest_sha256"
                ],
                "experimental_pool_manifest_sha256": config[
                    "expected_hanyang_pool_manifest_sha256"
                ],
                "experimental_pool_receipt_sha256": config[
                    "expected_hanyang_pool_receipt_sha256"
                ],
                "split_counts": deepcopy(
                    contracts["split"]["source_counts"]["hanyang"]
                ),
                "role": (
                    "secondary_partial_motion_domain_experiment_not_foundation"
                ),
            },
        },
        "condition_caches": {
            "foundation_condition_cache_sha256": config[
                "expected_foundation_condition_cache_sha256"
            ],
            "frozen_qwen_cache_sha256": config[
                "expected_frozen_qwen_cache_sha256"
            ],
            "qwen_variant": "frozen_base_no_lora",
        },
        "admission_contract": deepcopy(contracts["admission"]),
        "data_contract": deepcopy(contracts["data"]),
        "split_contract": deepcopy(contracts["split"]),
        "normalizer_contract": deepcopy(contracts["normalizer"]),
        "human_review_sample_bundle": deepcopy(
            config["human_review"]["sample_bundle"]
        ),
    }
    assert_no_forbidden_data_lineage(
        lineage, context="v8_checkpoint_lineage"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": CHECKPOINT_ARTIFACT_KIND,
        "experiment_artifact_kind": EXPERIMENT_ARTIFACT_KIND,
        "experimental_only": True,
        "formal_release_eligible": False,
        "smoke_test": bool(smoke_test),
        "human_review_required": True,
        "human_review_approved": bool(config["human_review"]["approved"]),
        "training_gate": (
            "TECHNICAL_SMOKE_ONLY"
            if smoke_test
            else "APPROVED"
        ),
        "no_seven_class_claim": True,
        "insufficient_heldout_emotion_coverage": True,
        "architecture": foundation["architecture"],
        "action_dim": ACTION_DIM,
        "condition_dim": KIMODO_V2_CONDITION_DIM,
        "joint_order": deepcopy(foundation["joint_order"]),
        "config": deepcopy(foundation.get("config") or {}),
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model_state.items()
        },
        "action_stats": deepcopy(foundation["action_stats"]),
        "global_step": int(foundation.get("global_step", 0)) + int(step),
        "posttrain_step": int(step),
        "posttrain_target_steps": int(target_steps),
        "foundation_checkpoint_sha256": config[
            "expected_foundation_checkpoint_sha256"
        ],
        "fresh_optimizer_from_clean_foundation": True,
        "optimizer_state_imported_from_v7": False,
        "contracts": {
            name: value["sha256"] for name, value in contracts.items()
        },
        "input_contract_sha256": input_contract["sha256"],
        "config_sha256": input_contract["config_sha256"],
        "approval_receipt_sha256": lineage[
            "approval_receipt_sha256"
        ],
        "sources": deepcopy(lineage["sources"]),
        "condition_caches": deepcopy(lineage["condition_caches"]),
        "lineage": lineage,
        "exposure": {
            "beat2_episode_count": int(exposure["beat2"]),
            "hanyang_episode_count": int(exposure["hanyang"]),
            "actual_hanyang_episode_fraction": (
                int(exposure["hanyang"]) / float(max(1, total_exposure))
            ),
            "maximum_hanyang_examples_per_effective_batch_16": 1,
        },
        "condition_policy": (
            "beat2_frozen_qwen_128d_hanyang_all_zero_unconditional_v1"
        ),
        "deny_policy": KIMODO_PERMANENT_DENY_POLICY,
    }


def _save_state(
    path: Path,
    *,
    model: torch.nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    beat2_sampler: NativeLengthBucketSampler,
    hanyang_sampler: NativeLengthBucketSampler,
    input_contract_sha256: str,
    step: int,
    target_steps: int,
    exposure: Mapping[str, int],
) -> None:
    _atomic_torch(
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": STATE_ARTIFACT_KIND,
            "input_contract_sha256": input_contract_sha256,
            "step": int(step),
            "target_steps": int(target_steps),
            "model_state_dict": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "ema_state_dict": {
                name: value.detach().cpu()
                for name, value in ema.shadow.items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "beat2_sampler_state_dict": beat2_sampler.state_dict(),
            "hanyang_sampler_state_dict": hanyang_sampler.state_dict(),
            "exposure": dict(exposure),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else []
            ),
            "python_rng_state": random.getstate(),
        },
        path,
    )


def _load_state(
    path: Path,
    *,
    model: torch.nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    beat2_sampler: NativeLengthBucketSampler,
    hanyang_sampler: NativeLengthBucketSampler,
    input_contract_sha256: str,
    target_steps: int,
    device: torch.device,
) -> tuple[int, dict[str, int]]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (
        state.get("artifact_kind") != STATE_ARTIFACT_KIND
        or state.get("schema_version") != SCHEMA_VERSION
        or state.get("input_contract_sha256") != input_contract_sha256
        or int(state.get("target_steps", -1)) != int(target_steps)
    ):
        raise ValueError("V8 exact-resume state contract changed")
    model.load_state_dict(state["model_state_dict"], strict=True)
    ema.shadow = {
        name: value.to(device)
        for name, value in state["ema_state_dict"].items()
    }
    optimizer.load_state_dict(state["optimizer_state_dict"])
    for optimizer_state in optimizer.state.values():
        for name, value in optimizer_state.items():
            if isinstance(value, torch.Tensor):
                optimizer_state[name] = value.to(device)
    beat2_sampler.load_state_dict(state["beat2_sampler_state_dict"])
    hanyang_sampler.load_state_dict(state["hanyang_sampler_state_dict"])
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state.get("cuda_rng_state_all"):
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    random.setstate(state["python_rng_state"])
    exposure = {
        "beat2": int(state["exposure"]["beat2"]),
        "hanyang": int(state["exposure"]["hanyang"]),
    }
    return int(state["step"]), exposure


def train(
    config: Mapping[str, Any],
    *,
    smoke_test: bool,
    overwrite: bool,
    resume: bool,
) -> dict[str, Any]:
    config = validate_config(config)
    if not smoke_test and not config["human_review"]["approved"]:
        raise RuntimeError(
            f"{HUMAN_REVIEW_BLOCKED}: inspect the sample bundle and provide "
            "an explicit approval receipt before persistent V8 training"
        )
    paths = _paths(config, smoke_test=smoke_test)
    if overwrite:
        for name in KNOWN_OUTPUTS:
            target = paths["root"] / name
            if target.is_file():
                target.unlink()
        for name in PREPARED_OUTPUTS:
            target = paths["prepared"] / name
            if target.is_file():
                target.unlink()
    paths["root"].mkdir(parents=True, exist_ok=True)
    if paths["state"].is_file() and not resume and not overwrite:
        raise FileExistsError("V8 state exists; use --resume or --overwrite")

    splits, contracts = _prepare(config, paths=paths)
    foundation_path = config["foundation_checkpoint"]
    foundation_hash = sha256_file(foundation_path)
    model, foundation = load_contract_checkpoint(
        foundation_path, expected_action_dim=ACTION_DIM, device="cpu"
    )
    if (
        foundation_hash != config["expected_foundation_checkpoint_sha256"]
        or foundation.get("architecture")
        != ULA_MMDIT_V3_ADALN_ARCHITECTURE
        or int(foundation.get("condition_dim", -1))
        != KIMODO_V2_CONDITION_DIM
    ):
        raise ValueError("V8 requires the exact clean V3-AdaLN foundation")
    assert_no_forbidden_data_lineage(
        {
            "foundation_checkpoint_path": foundation_path,
            "foundation_sources": foundation.get("sources") or (),
        },
        context="v8_foundation",
    )
    normalizer = action_normalizer_contract(
        foundation["action_stats"],
        foundation_checkpoint=foundation_path,
        foundation_checkpoint_sha256=foundation_hash,
    )
    _write_contract(
        normalizer, paths["prepared"] / PREPARED_OUTPUTS[3]
    )
    contracts["normalizer"] = normalizer
    input_contract = {
        "artifact_kind": "hanyang_beat2_expanded_input_contract_v8",
        "schema_version": SCHEMA_VERSION,
        "config_sha256": canonical_sha256(config),
        "contracts": {
            name: value["sha256"] for name, value in contracts.items()
        },
        "foundation_checkpoint_sha256": foundation_hash,
        "human_review_approved": bool(
            config["human_review"]["approved"]
        ),
        "smoke_test": bool(smoke_test),
    }
    input_contract["sha256"] = canonical_sha256(input_contract)

    device_name = str(config["device"])
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    _seed_everything(int(config["seed"]))
    model = model.to(device)
    model.requires_grad_(True)
    training = config["training"]
    target_steps = 2 if smoke_test else int(training["steps"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["lr"]),
        weight_decay=float(training["weight_decay"]),
        eps=float(training["adam_eps"]),
    )
    ema = ModelEMA(model, float(training["ema_decay"]))
    beat2_sampler = NativeLengthBucketSampler(
        splits["train"]["beat2"],
        buckets=training["batching"]["length_buckets"],
        sampler_config=training["sampler"],
        seed=int(config["seed"]) + 17,
    )
    hanyang_sampler = NativeLengthBucketSampler(
        splits["train"]["hanyang"],
        buckets=training["batching"]["length_buckets"],
        sampler_config=training["sampler"],
        seed=int(config["seed"]) + 37,
    )
    step = 0
    exposure = {"beat2": 0, "hanyang": 0}
    if paths["state"].is_file():
        if not resume:
            raise FileExistsError("V8 state exists but --resume was not supplied")
        step, exposure = _load_state(
            paths["state"],
            model=model,
            ema=ema,
            optimizer=optimizer,
            beat2_sampler=beat2_sampler,
            hanyang_sampler=hanyang_sampler,
            input_contract_sha256=input_contract["sha256"],
            target_steps=target_steps,
            device=device,
        )
    else:
        paths["progress"].write_text("", encoding="utf-8")
        _append_jsonl(
            {
                "event": "initialized",
                "step": 0,
                "smoke_test": bool(smoke_test),
                "training_gate": (
                    "TECHNICAL_SMOKE_ONLY"
                    if smoke_test
                    else "APPROVED"
                ),
                "fresh_optimizer_from_clean_foundation": True,
                "input_contract_sha256": input_contract["sha256"],
            },
            paths["progress"],
        )

    optimized_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    model.train()
    started = time.monotonic()
    last_losses: dict[str, float] = {}
    last_validation: dict[str, dict[str, float]] = {}
    last_grad_norm = 0.0
    for current_step in range(step + 1, target_steps + 1):
        scale = _lr_scale(
            current_step,
            total_steps=target_steps,
            warmup_steps=min(int(training["warmup_steps"]), target_steps),
            minimum_ratio=float(training["minimum_lr_ratio"]),
        )
        optimizer.param_groups[0]["lr"] = float(training["lr"]) * scale
        optimizer.zero_grad(set_to_none=True)
        hanyang_quota = _hanyang_quota(current_step, training)
        beat2_quota = int(training["batch_size"]) - hanyang_quota
        accumulated: defaultdict[str, float] = defaultdict(float)
        sampled: dict[str, list[str]] = {"beat2": [], "hanyang": []}
        microbatch_index = 0
        for source, quota, sampler in (
            ("hanyang", hanyang_quota, hanyang_sampler),
            ("beat2", beat2_quota, beat2_sampler),
        ):
            remaining = quota
            while remaining > 0:
                episodes, _plan = sampler.sample_microbatch(
                    remaining_effective_batch=remaining,
                    semantic_tokens=int(model.semantic_tokens),
                    max_batch_size=int(training["batch_size"]),
                    batching=training["batching"],
                )
                losses = _batch_objective(
                    model,
                    episodes,
                    source=source,
                    action_stats=foundation["action_stats"],
                    training=training,
                    device=device,
                    seed=(
                        int(config["seed"])
                        + current_step * 1_000_003
                        + microbatch_index * 1009
                    ),
                )
                sample_fraction = len(episodes) / float(
                    training["batch_size"]
                )
                (losses["total"] * sample_fraction).backward()
                for name, value in losses.items():
                    accumulated[f"{source}_{name}"] += (
                        float(value.detach().cpu()) * sample_fraction
                    )
                sampled[source].extend(
                    str(episode["clip_id"]) for episode in episodes
                )
                exposure[source] += len(episodes)
                remaining -= len(episodes)
                microbatch_index += 1
        if len(sampled["hanyang"]) > 1:
            raise RuntimeError("Hanyang per-effective-batch cap was exceeded")
        last_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                optimized_parameters, float(training["max_grad_norm"])
            )
        )
        if not math.isfinite(last_grad_norm):
            raise FloatingPointError("non-finite V8 gradient norm")
        optimizer.step()
        ema.update(model)
        step = current_step
        last_losses = dict(accumulated)
        should_validate = (
            current_step == 1
            or current_step % int(training["validation_interval"]) == 0
            or current_step == target_steps
        )
        should_checkpoint = (
            current_step == 1
            or current_step % int(training["checkpoint_interval"]) == 0
            or current_step == target_steps
        )
        should_export = (
            current_step % int(training["export_interval"]) == 0
            or current_step == target_steps
        )
        if should_validate:
            with ema.apply(model):
                last_validation = _evaluate(
                    model,
                    splits,
                    action_stats=foundation["action_stats"],
                    training=training,
                    device=device,
                    seed=int(config["seed"]) + 9_000_001,
                )
        total_exposure = exposure["beat2"] + exposure["hanyang"]
        event = {
            "event": "train_step",
            "step": current_step,
            "smoke_test": bool(smoke_test),
            "lr": optimizer.param_groups[0]["lr"],
            "grad_norm": last_grad_norm,
            "losses": last_losses,
            "exposure": dict(exposure),
            "actual_hanyang_episode_fraction": (
                exposure["hanyang"] / float(max(1, total_exposure))
            ),
            "hanyang_examples_this_effective_batch_16": len(
                sampled["hanyang"]
            ),
            "sampled_clip_ids_sha256": {
                source: hashlib.sha256(
                    json.dumps(values, sort_keys=True).encode("utf-8")
                ).hexdigest()
                for source, values in sampled.items()
            },
        }
        if should_validate:
            event["validation"] = last_validation
        if (
            current_step == 1
            or current_step % int(training["log_interval"]) == 0
            or should_validate
        ):
            print(json.dumps(event, sort_keys=True), flush=True)
            _append_jsonl(event, paths["progress"])
        if should_checkpoint:
            _save_state(
                paths["state"],
                model=model,
                ema=ema,
                optimizer=optimizer,
                beat2_sampler=beat2_sampler,
                hanyang_sampler=hanyang_sampler,
                input_contract_sha256=input_contract["sha256"],
                step=current_step,
                target_steps=target_steps,
                exposure=exposure,
            )
        if should_export:
            latest = _checkpoint_payload(
                model_state=ema.shadow,
                foundation=foundation,
                contracts=contracts,
                input_contract=input_contract,
                config=config,
                step=current_step,
                target_steps=target_steps,
                exposure=exposure,
                smoke_test=smoke_test,
            )
            _atomic_torch(latest, paths["latest"])

    final_payload = _checkpoint_payload(
        model_state=ema.shadow,
        foundation=foundation,
        contracts=contracts,
        input_contract=input_contract,
        config=config,
        step=step,
        target_steps=target_steps,
        exposure=exposure,
        smoke_test=smoke_test,
    )
    _atomic_torch(final_payload, paths["checkpoint"])
    total_exposure = exposure["beat2"] + exposure["hanyang"]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": SUMMARY_ARTIFACT_KIND,
        "experiment_artifact_kind": EXPERIMENT_ARTIFACT_KIND,
        "experimental_only": True,
        "formal_release_eligible": False,
        "smoke_test": bool(smoke_test),
        "human_review_required": True,
        "human_review_approved": bool(config["human_review"]["approved"]),
        "training_gate": (
            "TECHNICAL_SMOKE_ONLY" if smoke_test else "APPROVED"
        ),
        "steps": step,
        "target_steps": target_steps,
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_sha256": sha256_file(paths["checkpoint"]),
        "foundation_checkpoint_sha256": foundation_hash,
        "fresh_optimizer_from_clean_foundation": True,
        "elapsed_sec": time.monotonic() - started,
        "last_grad_norm": last_grad_norm,
        "last_losses": last_losses,
        "last_validation": last_validation,
        "exposure": {
            "beat2_episode_count": exposure["beat2"],
            "hanyang_episode_count": exposure["hanyang"],
            "actual_hanyang_episode_fraction": (
                exposure["hanyang"] / float(max(1, total_exposure))
            ),
            "configured_target_fraction": float(
                training["hanyang_domain_fraction"]
            ),
            "maximum_hanyang_examples_per_effective_batch_16": 1,
        },
        "contracts": {
            name: value["sha256"] for name, value in contracts.items()
        },
        "input_contract_sha256": input_contract["sha256"],
        "config_sha256": input_contract["config_sha256"],
        "approval_receipt_sha256": (
            config["human_review"]["expected_approval_receipt_sha256"]
            if config["human_review"]["approved"]
            else None
        ),
        "sources": deepcopy(final_payload["sources"]),
        "condition_caches": deepcopy(final_payload["condition_caches"]),
        "no_seven_class_claim": True,
        "insufficient_heldout_emotion_coverage": True,
    }
    _atomic_json(summary, paths["summary"])
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    summary = train(
        config,
        smoke_test=bool(args.smoke_test),
        overwrite=bool(args.overwrite),
        resume=bool(args.resume),
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
