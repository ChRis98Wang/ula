#!/usr/bin/env python3
"""Fail-closed CPU-only winner selection for the V7 Qwen generator A/B.

The selector consumes the sealed pre-training A/B contract plus both completed
generator summaries and checkpoints.  It never treats the Qwen alignment
experiment itself as motion evidence, never selects an incomplete arm, and
defaults to the frozen-Qwen A arm unless LoRA-Qwen B shows a material
multi-metric condition *and* emotion improvement without flow or duration
regression.

The output is a self-hashed receipt.  ``status=pending`` is the only permitted
result while B is incomplete.  A V8.1 config may bind only a
``status=selected`` receipt after separately pinning the receipt file SHA256.
No CUDA operation is used by this module.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence


# A standalone selector process must not see accelerators.  Do not mutate this
# variable when imported by the V8.1 trainer: that process needs CUDA after its
# CPU-only receipt validation finishes.  Every load in this module is still
# explicitly mapped to CPU.
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
CONFIG_ARTIFACT_KIND = (
    "beat2_emotion_hierarchy_qwen_ab_winner_selection_config_v1"
)
RECEIPT_ARTIFACT_KIND = (
    "beat2_emotion_hierarchy_qwen_ab_winner_selection_receipt_v1"
)
SEALED_AB_ARTIFACT_KIND = (
    "beat2_emotion_hierarchy_qwen_generator_ab_contract_receipt_v7"
)
SUMMARY_ARTIFACT_KIND = "beat2_emotion_hierarchy_training_summary_v7"
CHECKPOINT_ARTIFACT_KIND = "beat2_emotion_hierarchy_generator_v7"
CONDITION_POLICY = (
    "zero_base_0_133_qwen_predicted_style_133_136_"
    "frozen_qwen_text_136_264_no_trajectory_oracle_v2"
)
DATA_POLICY = "beat2_only_no_external_motion_dataset_v1"
FROZEN_VARIANT = "frozen_base"
LORA_VARIANT = "lora_finetuned"
PENDING_STATUS = "pending"
SELECTED_STATUS = "selected"
EXPECTED_STEPS = 80_000
ARM_FILENAMES = {
    "summary": "training_summary_v7.json",
    "checkpoint": "generator_emotion_hierarchy_v7.pt",
}


DECISION_POLICY: dict[str, Any] = {
    "policy": (
        "material_multimetric_condition_and_emotion_gain_"
        "with_flow_duration_non_regression_v1"
    ),
    "near_tie_winner": FROZEN_VARIANT,
    "single_metric_can_select_lora": False,
    "condition_metrics": [
        {
            "name": "aligned_vs_zero_prediction_rms",
            "direction": "higher",
            "scale_floor": 0.02,
        },
        {
            "name": "aligned_vs_cross_group_prediction_rms",
            "direction": "higher",
            "scale_floor": 0.02,
        },
        {
            "name": "cross_group_minus_aligned_flow_loss",
            "direction": "higher",
            "scale_floor": 0.002,
        },
        {
            "name": "correct_flow_win_rate",
            "direction": "higher",
            "scale_floor": 0.25,
        },
    ],
    "emotion_metrics": [
        {
            "name": "semantic_cross_group_cosine_gap",
            "direction": "higher",
            "scale_floor": 0.05,
        },
        {
            "name": "semantic_global_hard_cross_group_margin",
            "direction": "higher",
            "scale_floor": 0.10,
        },
        {
            "name": (
                "semantic_global_motion_to_prototype_recall_at_1"
            ),
            "direction": "higher",
            "scale_floor": 0.25,
        },
        {
            "name": "semantic_global_positive_cosine",
            "direction": "higher",
            "scale_floor": 0.25,
        },
    ],
    "minimum_composite_normalized_gain": 0.05,
    "minimum_individual_normalized_gain": 0.03,
    "minimum_material_metrics_per_family": 2,
    "maximum_individual_normalized_regression": 0.05,
    "motion_non_regression_guards": [
        {
            "name": "aligned_flow_loss",
            "direction": "lower",
            "maximum_relative_regression": 0.005,
            "maximum_absolute_regression": 0.005,
        },
        {
            "name": "duration_mae_sec",
            "direction": "lower",
            "maximum_relative_regression": 0.02,
            "maximum_absolute_regression": 0.02,
        },
    ],
    "requires_both_arms_anti_collapse_passed": True,
    "requires_exact_shared_training_contract": True,
    "requires_no_external_data": True,
    "requires_no_kimodo": True,
    "requires_no_hanyang": True,
}


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _resolve_path(
    value: object,
    *,
    field: str,
    must_exist: bool,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if must_exist and not path.is_file():
        raise FileNotFoundError(path)
    _reject_forbidden_path(path, field=field)
    return path


def _reject_forbidden_path(path: Path, *, field: str) -> None:
    folded = str(path).casefold()
    if any(token in folded for token in ("kimodo", "komodo", "hanyang")):
        raise ValueError(f"{field} contains forbidden external lineage")


def _reject_hanyang_strings(value: object, *, field: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_hanyang_strings(child, field=f"{field}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, child in enumerate(value):
            _reject_hanyang_strings(child, field=f"{field}[{index}]")
    elif isinstance(value, str) and "hanyang" in value.casefold():
        raise ValueError(f"{field} contains forbidden Hanyang lineage")


def _reject_forbidden_source_strings(
    value: object,
    *,
    key: str = "root",
) -> None:
    """Reject Kimodo/Komodo in source-bearing fields, not deny-policy text."""

    if isinstance(value, Mapping):
        for child_key, child in value.items():
            _reject_forbidden_source_strings(
                child, key=str(child_key)
            )
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for child in value:
            _reject_forbidden_source_strings(child, key=key)
        return
    if not isinstance(value, str):
        return
    folded = value.casefold()
    if not any(token in folded for token in ("kimodo", "komodo")):
        return
    source_key = key.casefold()
    source_bearing = any(
        token in source_key
        for token in (
            "path",
            "source",
            "dataset",
            "manifest",
            "checkpoint",
            "cache",
            "output",
            "input",
        )
    )
    looks_like_path = "/" in value or "\\" in value
    if source_bearing or looks_like_path:
        raise ValueError(f"{key} contains forbidden Kimodo lineage")


def _read_json(path: Path, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{context} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _require_finite(value: object, *, field: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _require_finite(child, field=f"{field}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, child in enumerate(value):
            _require_finite(child, field=f"{field}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError(f"{field} must be finite")


def _validate_internal_sha(
    value: Mapping[str, Any], *, context: str
) -> None:
    claimed = value.get("sha256")
    record = {key: child for key, child in value.items() if key != "sha256"}
    if not _is_sha256(claimed) or claimed != canonical_sha256(record):
        raise ValueError(f"{context} canonical SHA256 mismatch")


def _validate_decision_policy(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("decision_policy must be a mapping")
    if dict(value) != DECISION_POLICY:
        raise ValueError("decision_policy differs from the reviewed policy")
    return deepcopy(DECISION_POLICY)


def read_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    values = _read_json(config_path, context="winner selection config")
    expected_fields = {
        "schema_version",
        "artifact_kind",
        "sealed_ab_receipt",
        "expected_sealed_ab_receipt_sha256",
        "arms",
        "decision_policy",
        "output_receipt",
    }
    if set(values) != expected_fields:
        raise ValueError("winner selection config fields changed")
    if (
        values.get("schema_version") != SCHEMA_VERSION
        or values.get("artifact_kind") != CONFIG_ARTIFACT_KIND
        or not _is_sha256(
            values.get("expected_sealed_ab_receipt_sha256")
        )
    ):
        raise ValueError("winner selection config contract changed")
    arms = values.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != {
        FROZEN_VARIANT,
        LORA_VARIANT,
    }:
        raise ValueError("winner selection arms changed")
    expected_arm_fields = {
        "summary",
        "checkpoint",
        "expected_summary_sha256",
        "expected_checkpoint_sha256",
    }
    resolved_arms: dict[str, dict[str, Any]] = {}
    for variant in (FROZEN_VARIANT, LORA_VARIANT):
        arm = arms[variant]
        if not isinstance(arm, Mapping) or set(arm) != expected_arm_fields:
            raise ValueError(f"{variant} selection fields changed")
        expected_summary = arm.get("expected_summary_sha256")
        expected_checkpoint = arm.get("expected_checkpoint_sha256")
        if variant == FROZEN_VARIANT:
            if not _is_sha256(expected_summary) or not _is_sha256(
                expected_checkpoint
            ):
                raise ValueError("frozen A artifacts must be hash-pinned")
        elif expected_summary is not None or expected_checkpoint is not None:
            raise ValueError(
                "LoRA B hashes are sealed by its completed summary, not "
                "predeclared before training"
            )
        resolved_arms[variant] = {
            **dict(arm),
            "summary": str(
                _resolve_path(
                    arm["summary"],
                    field=f"arms.{variant}.summary",
                    must_exist=False,
                )
            ),
            "checkpoint": str(
                _resolve_path(
                    arm["checkpoint"],
                    field=f"arms.{variant}.checkpoint",
                    must_exist=False,
                )
            ),
        }
    values["sealed_ab_receipt"] = str(
        _resolve_path(
            values["sealed_ab_receipt"],
            field="sealed_ab_receipt",
            must_exist=True,
        )
    )
    values["arms"] = resolved_arms
    values["decision_policy"] = _validate_decision_policy(
        values["decision_policy"]
    )
    values["output_receipt"] = str(
        _resolve_path(
            values["output_receipt"],
            field="output_receipt",
            must_exist=False,
        )
    )
    values["config_path"] = str(config_path)
    values["config_sha256"] = sha256_file(config_path)
    return values


def _validate_file_receipt(
    value: Mapping[str, Any],
    *,
    context: str,
) -> tuple[Path, str]:
    path = _resolve_path(
        value.get("path"), field=f"{context}.path", must_exist=True
    )
    expected = value.get("sha256")
    actual = sha256_file(path)
    if not _is_sha256(expected) or actual != expected:
        raise ValueError(f"{context} SHA256 mismatch")
    return path, actual


def validate_sealed_ab_receipt(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> dict[str, Any]:
    receipt_path = _resolve_path(
        str(path), field="sealed_ab_receipt", must_exist=True
    )
    actual_file_sha256 = sha256_file(receipt_path)
    if (
        not _is_sha256(expected_file_sha256)
        or actual_file_sha256 != expected_file_sha256
    ):
        raise ValueError("sealed A/B receipt file SHA256 mismatch")
    receipt = _read_json(receipt_path, context="sealed A/B receipt")
    _validate_internal_sha(receipt, context="sealed A/B receipt")
    if (
        receipt.get("schema_version") != 7
        or receipt.get("artifact_kind") != SEALED_AB_ARTIFACT_KIND
        or receipt.get("status")
        != "contract_validated_training_not_started"
        or receipt.get("smoke_test") is not False
        or receipt.get("data_policy") != DATA_POLICY
        or receipt.get("no_external_data") is not True
        or receipt.get("no_kimodo") is not True
    ):
        raise ValueError("sealed A/B receipt contract changed")
    shared = receipt.get("shared_contract")
    arms = receipt.get("arms")
    if (
        not isinstance(shared, Mapping)
        or not isinstance(arms, Mapping)
        or set(arms) != {FROZEN_VARIANT, LORA_VARIANT}
        or int(shared.get("steps", -1)) != EXPECTED_STEPS
        or int(shared.get("seed", -1)) < 0
        or shared.get("same_noise_and_flow_time_implementation") is not True
        or not _is_sha256(receipt.get("invariant_config_sha256"))
        or not _is_sha256(
            receipt.get("paired_cache_identity_sha256")
        )
    ):
        raise ValueError("sealed A/B shared contract changed")
    claims = receipt.get("claims")
    if (
        not isinstance(claims, Mapping)
        or claims.get("contract_parity_only") is not True
        or claims.get("generator_training_completed") is not False
        or claims.get("requires_generator_motion_evaluation_after_training")
        is not True
    ):
        raise ValueError("sealed A/B claims changed")
    foundation_path = _resolve_path(
        shared.get("foundation_checkpoint"),
        field="shared_contract.foundation_checkpoint",
        must_exist=True,
    )
    if sha256_file(foundation_path) != shared.get(
        "foundation_checkpoint_sha256"
    ):
        raise ValueError("sealed foundation checkpoint SHA256 mismatch")
    style_path = _resolve_path(
        shared.get("style_condition_cache"),
        field="shared_contract.style_condition_cache",
        must_exist=True,
    )
    if sha256_file(style_path) != shared.get(
        "style_condition_cache_sha256"
    ):
        raise ValueError("sealed style cache SHA256 mismatch")
    for variant, arm in arms.items():
        if not isinstance(arm, Mapping):
            raise ValueError(f"{variant} sealed arm must be a mapping")
        cache_path = _resolve_path(
            arm.get("condition_cache"),
            field=f"arms.{variant}.condition_cache",
            must_exist=True,
        )
        if sha256_file(cache_path) != arm.get("condition_cache_sha256"):
            raise ValueError(f"{variant} condition cache SHA256 mismatch")
        adapter = arm.get("adapter_receipt")
        if not isinstance(adapter, Mapping):
            raise ValueError(f"{variant} adapter receipt missing")
        _validate_file_receipt(
            adapter, context=f"arms.{variant}.adapter_receipt"
        )
        output_dir = _resolve_path(
            str(Path(str(arm.get("output_dir"))) / ".not_a_file"),
            field=f"arms.{variant}.output_dir",
            must_exist=False,
        ).parent
        arm["condition_cache"] = str(cache_path)
        arm["output_dir"] = str(output_dir)
    _reject_hanyang_strings(receipt)
    _reject_forbidden_source_strings(receipt)
    result = deepcopy(receipt)
    result["path"] = str(receipt_path)
    result["file_sha256"] = actual_file_sha256
    result["arms"] = deepcopy(dict(arms))
    return result


def _valid_gate(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    checks = value.get("checks")
    return (
        value.get("passed") is True
        and value.get("enforced") is True
        and isinstance(checks, Mapping)
        and bool(checks)
        and all(check is True for check in checks.values())
        and value.get("failure_reasons") == []
    )


def _expected_arm_paths(
    sealed: Mapping[str, Any],
    variant: str,
) -> tuple[Path, Path]:
    output = Path(sealed["arms"][variant]["output_dir"]).resolve()
    return (
        output / ARM_FILENAMES["summary"],
        output / ARM_FILENAMES["checkpoint"],
    )


def _arm_availability(
    summary_path: Path, checkpoint_path: Path
) -> dict[str, Any]:
    return {
        "summary_exists": summary_path.is_file(),
        "checkpoint_exists": checkpoint_path.is_file(),
    }


def _validate_arm(
    *,
    variant: str,
    summary_path: Path,
    checkpoint_path: Path,
    sealed: Mapping[str, Any],
    expected_summary_sha256: str | None,
    expected_checkpoint_sha256: str | None,
) -> dict[str, Any]:
    if variant not in {FROZEN_VARIANT, LORA_VARIANT}:
        raise ValueError("unknown Qwen A/B variant")
    expected_summary_path, expected_checkpoint_path = (
        _expected_arm_paths(sealed, variant)
    )
    if (
        summary_path.resolve() != expected_summary_path
        or checkpoint_path.resolve() != expected_checkpoint_path
    ):
        raise ValueError(f"{variant} artifacts are outside sealed output")
    if not summary_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"{variant} completed artifacts are missing")
    summary_file_sha = sha256_file(summary_path)
    checkpoint_file_sha = sha256_file(checkpoint_path)
    if expected_summary_sha256 is not None and (
        not _is_sha256(expected_summary_sha256)
        or summary_file_sha != expected_summary_sha256
    ):
        raise ValueError(f"{variant} summary SHA256 mismatch")
    if expected_checkpoint_sha256 is not None and (
        not _is_sha256(expected_checkpoint_sha256)
        or checkpoint_file_sha != expected_checkpoint_sha256
    ):
        raise ValueError(f"{variant} checkpoint SHA256 mismatch")
    summary = _read_json(summary_path, context=f"{variant} summary")
    if (
        summary.get("schema_version") != 7
        or summary.get("artifact_kind") != SUMMARY_ARTIFACT_KIND
        or summary.get("status") != "experimental_candidate"
        or int(summary.get("completed_steps", -1)) != EXPECTED_STEPS
        or int(summary.get("target_steps", -1)) != EXPECTED_STEPS
        or summary.get("formal_release_eligible") is not False
        or summary.get("no_external_data") is not True
        or summary.get("no_kimodo") is not True
        or summary.get("checkpoint_sha256") != checkpoint_file_sha
        or Path(str(summary.get("checkpoint"))).resolve()
        != checkpoint_path.resolve()
        or not _valid_gate(summary.get("anti_collapse_gate"))
    ):
        raise ValueError(f"{variant} summary is not a completed candidate")
    frozen_audit = summary.get("frozen_parameter_audit")
    if (
        not isinstance(frozen_audit, Mapping)
        or frozen_audit.get("passed") is not True
        or frozen_audit.get("changed_frozen_tensor_names") != []
        or float(frozen_audit.get("maximum_abs_error", math.inf)) != 0.0
    ):
        raise ValueError(f"{variant} frozen-parameter audit failed")
    initial = summary.get("initial_condition_diagnostics")
    final = summary.get("final_condition_diagnostics")
    if not isinstance(initial, Mapping) or not isinstance(final, Mapping):
        raise ValueError(f"{variant} condition diagnostics are missing")
    _require_finite(initial, field=f"{variant}.initial_diagnostics")
    _require_finite(final, field=f"{variant}.final_diagnostics")

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"{variant} checkpoint must be a mapping")
    if (
        checkpoint.get("schema_version") != 7
        or checkpoint.get("artifact_kind") != CHECKPOINT_ARTIFACT_KIND
        or checkpoint.get("condition_policy") != CONDITION_POLICY
        or checkpoint.get("architecture") != "ula_mmdit_v3_adaln"
        or int(checkpoint.get("action_dim", -1)) != 18
        or int(checkpoint.get("condition_dim", -1)) != 264
        or checkpoint.get("formal_release_eligible") is not False
        or checkpoint.get("no_external_data") is not True
        or checkpoint.get("no_kimodo") is not True
        or not _valid_gate(checkpoint.get("anti_collapse_gate"))
        or checkpoint.get("anti_collapse_gate")
        != summary.get("anti_collapse_gate")
        or (checkpoint.get("metrics") or {}).get(
            "initial_condition_diagnostics"
        )
        != initial
        or (checkpoint.get("metrics") or {}).get(
            "final_condition_diagnostics"
        )
        != final
    ):
        raise ValueError(f"{variant} checkpoint contract changed")
    training = checkpoint.get("training_contract")
    data = checkpoint.get("data_receipt")
    foundation = checkpoint.get("foundation_receipt")
    split = checkpoint.get("split_contract")
    input_contract = checkpoint.get("input_contract")
    shared = sealed["shared_contract"]
    if not all(
        isinstance(value, Mapping)
        for value in (training, data, foundation, split, input_contract)
    ):
        raise ValueError(f"{variant} checkpoint receipts are incomplete")
    if (
        int(training.get("steps", -1)) != int(shared["steps"])
        or int(training.get("seed", -1)) != int(shared["seed"])
        or training.get("shared_noise_and_flow_times") is not True
        or training.get("negative_policy") != shared["negative_policy"]
        or training.get("text_dropout_policy")
        != shared["text_dropout_policy"]
        or int(training.get("external_motion_checkpoint_count", -1)) != 0
        or canonical_sha256(training.get("loss"))
        != shared["loss_sha256"]
        or canonical_sha256(training.get("batching"))
        != shared["batching_sha256"]
        or canonical_sha256(training.get("sampler"))
        != shared["sampler_sha256"]
        or foundation.get("sha256")
        != shared["foundation_checkpoint_sha256"]
        or (data.get("foundation") or {}).get("sha256")
        != shared["foundation_checkpoint_sha256"]
        or (data.get("manifest") or {}).get("sha256")
        != shared["manifest_sha256"]
        or (data.get("style_cache") or {}).get("sha256")
        != shared["style_condition_cache_sha256"]
        or data.get("no_external_data") is not True
        or data.get("no_kimodo") is not True
        or split.get("counts") != shared["fixed_split_counts"]
        or input_contract.get("manifest_sha256")
        != shared["manifest_sha256"]
        or input_contract.get("foundation_checkpoint_sha256")
        != shared["foundation_checkpoint_sha256"]
        or input_contract.get("style_cache_sha256")
        != shared["style_condition_cache_sha256"]
        or summary.get("input_contract_sha256")
        != input_contract.get("sha256")
    ):
        raise ValueError(f"{variant} differs from the sealed A/B contract")
    expected_cache = sealed["arms"][variant]
    qwen_cache = data.get("frozen_qwen_cache")
    if (
        not isinstance(qwen_cache, Mapping)
        or qwen_cache.get("variant") != variant
        or qwen_cache.get("sha256")
        != expected_cache["condition_cache_sha256"]
        or Path(str(qwen_cache.get("path"))).resolve()
        != Path(expected_cache["condition_cache"]).resolve()
        or input_contract.get("frozen_qwen_cache_sha256")
        != expected_cache["condition_cache_sha256"]
    ):
        raise ValueError(f"{variant} checkpoint uses the wrong Qwen cache")
    posttrain_step = int(foundation.get("posttrain_step", -1))
    if (
        posttrain_step < 0
        or int(checkpoint.get("global_step", -1))
        != posttrain_step + EXPECTED_STEPS
        or foundation.get("dataset_family_whitelist") != ["BEAT2"]
        or foundation.get("generator_checkpoint_inputs") != []
    ):
        raise ValueError(f"{variant} foundation origin changed")
    _reject_hanyang_strings(summary)
    _reject_hanyang_strings(
        {
            "foundation": foundation,
            "data_receipt": data,
            "input_contract": input_contract,
            "training_contract": training,
        }
    )
    _reject_forbidden_source_strings(summary)
    _reject_forbidden_source_strings(
        {
            "foundation": foundation,
            "data_receipt": data,
            "input_contract": input_contract,
            "training_contract": training,
        }
    )
    return {
        "variant": variant,
        "completed": True,
        "summary": {
            "path": str(summary_path.resolve()),
            "file_sha256": summary_file_sha,
            "status": summary["status"],
            "completed_steps": int(summary["completed_steps"]),
        },
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": checkpoint_file_sha,
            "global_step": int(checkpoint["global_step"]),
        },
        "condition_cache": {
            "path": str(Path(expected_cache["condition_cache"]).resolve()),
            "sha256": expected_cache["condition_cache_sha256"],
        },
        "foundation_origin_sha256": foundation["sha256"],
        "split_contract": deepcopy(dict(split)),
        "training_invariants": {
            "seed": int(training["seed"]),
            "steps": int(training["steps"]),
            "loss_sha256": canonical_sha256(training["loss"]),
            "batching_sha256": canonical_sha256(training["batching"]),
            "sampler_sha256": canonical_sha256(training["sampler"]),
            "negative_policy": training["negative_policy"],
            "text_dropout_policy": training["text_dropout_policy"],
            "shared_noise_and_flow_times": True,
            "manifest_sha256": (data["manifest"])["sha256"],
            "style_cache_sha256": (data["style_cache"])["sha256"],
            "identity_arrays_sha256": data["identity_arrays_sha256"],
            "data_split_contract_sha256": data["split_contract_sha256"],
        },
        "initial_condition_diagnostics": deepcopy(dict(initial)),
        "final_condition_diagnostics": deepcopy(dict(final)),
        "anti_collapse_gate": deepcopy(
            dict(summary["anti_collapse_gate"])
        ),
        "no_external_data": True,
        "no_kimodo": True,
        "no_hanyang": True,
    }


def _shared_pair_contract(
    frozen: Mapping[str, Any],
    lora: Mapping[str, Any],
    sealed: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        frozen["training_invariants"] != lora["training_invariants"]
        or frozen["foundation_origin_sha256"]
        != lora["foundation_origin_sha256"]
        or frozen["split_contract"] != lora["split_contract"]
        or frozen["checkpoint"]["global_step"]
        != lora["checkpoint"]["global_step"]
    ):
        raise ValueError(
            "completed Qwen A/B differs in foundation, seed, split, steps, "
            "noise, sampler, loss, or batching"
        )
    invariants = {
        **deepcopy(dict(frozen["training_invariants"])),
        "foundation_origin_sha256": frozen[
            "foundation_origin_sha256"
        ],
        "checkpoint_global_step": frozen["checkpoint"]["global_step"],
        "checkpoint_split_contract": deepcopy(
            dict(frozen["split_contract"])
        ),
        "sealed_invariant_config_sha256": sealed[
            "invariant_config_sha256"
        ],
        "sealed_paired_cache_identity_sha256": sealed[
            "paired_cache_identity_sha256"
        ],
    }
    return {
        "values": invariants,
        "sha256": canonical_sha256(invariants),
        "same_foundation": True,
        "same_seed": True,
        "same_split": True,
        "same_steps": True,
        "same_noise_and_flow_time_implementation": True,
        "same_sampler": True,
        "same_loss_and_batching": True,
    }


def _metric_comparison(
    frozen_metrics: Mapping[str, Any],
    lora_metrics: Mapping[str, Any],
    specifications: Sequence[Mapping[str, Any]],
    *,
    individual_gain_threshold: float,
) -> tuple[list[dict[str, Any]], float, int, bool]:
    rows: list[dict[str, Any]] = []
    normalized_gains: list[float] = []
    material_count = 0
    regression_limit = float(
        DECISION_POLICY["maximum_individual_normalized_regression"]
    )
    no_material_regression = True
    for specification in specifications:
        name = str(specification["name"])
        frozen_value = float(frozen_metrics[name])
        lora_value = float(lora_metrics[name])
        direction = str(specification["direction"])
        scale = max(
            abs(frozen_value), float(specification["scale_floor"])
        )
        signed_delta = (
            lora_value - frozen_value
            if direction == "higher"
            else frozen_value - lora_value
        )
        normalized_gain = signed_delta / scale
        material = normalized_gain >= individual_gain_threshold
        if material:
            material_count += 1
        if normalized_gain < -regression_limit:
            no_material_regression = False
        normalized_gains.append(normalized_gain)
        rows.append(
            {
                "name": name,
                "direction": direction,
                "frozen_a": frozen_value,
                "lora_b": lora_value,
                "absolute_lora_minus_frozen": (
                    lora_value - frozen_value
                ),
                "normalized_gain_for_lora": normalized_gain,
                "material_improvement": material,
            }
        )
    composite = sum(normalized_gains) / len(normalized_gains)
    return rows, composite, material_count, no_material_regression


def decide_winner(
    frozen_metrics: Mapping[str, Any],
    lora_metrics: Mapping[str, Any],
    *,
    policy: Mapping[str, Any] = DECISION_POLICY,
) -> dict[str, Any]:
    policy = _validate_decision_policy(policy)
    _require_finite(frozen_metrics, field="frozen_metrics")
    _require_finite(lora_metrics, field="lora_metrics")
    individual_threshold = float(
        policy["minimum_individual_normalized_gain"]
    )
    condition = _metric_comparison(
        frozen_metrics,
        lora_metrics,
        policy["condition_metrics"],
        individual_gain_threshold=individual_threshold,
    )
    emotion = _metric_comparison(
        frozen_metrics,
        lora_metrics,
        policy["emotion_metrics"],
        individual_gain_threshold=individual_threshold,
    )
    guard_rows: list[dict[str, Any]] = []
    motion_guard_passed = True
    for guard in policy["motion_non_regression_guards"]:
        name = str(guard["name"])
        frozen_value = float(frozen_metrics[name])
        lora_value = float(lora_metrics[name])
        if guard["direction"] != "lower":
            raise ValueError("motion guard direction changed")
        regression = lora_value - frozen_value
        relative_regression = regression / max(abs(frozen_value), 1e-12)
        passed = (
            regression <= float(guard["maximum_absolute_regression"])
            and relative_regression
            <= float(guard["maximum_relative_regression"])
        )
        motion_guard_passed = motion_guard_passed and passed
        guard_rows.append(
            {
                "name": name,
                "frozen_a": frozen_value,
                "lora_b": lora_value,
                "absolute_regression": regression,
                "relative_regression": relative_regression,
                "passed": passed,
            }
        )
    minimum_composite = float(
        policy["minimum_composite_normalized_gain"]
    )
    minimum_metrics = int(
        policy["minimum_material_metrics_per_family"]
    )
    lora_material_gain = (
        condition[1] >= minimum_composite
        and emotion[1] >= minimum_composite
        and condition[2] >= minimum_metrics
        and emotion[2] >= minimum_metrics
        and condition[3]
        and emotion[3]
        and motion_guard_passed
    )
    winner = LORA_VARIANT if lora_material_gain else FROZEN_VARIANT
    reasons: list[str] = []
    if condition[1] < minimum_composite:
        reasons.append("condition_composite_gain_below_threshold")
    if emotion[1] < minimum_composite:
        reasons.append("emotion_composite_gain_below_threshold")
    if condition[2] < minimum_metrics:
        reasons.append("too_few_condition_metrics_materially_improved")
    if emotion[2] < minimum_metrics:
        reasons.append("too_few_emotion_metrics_materially_improved")
    if not condition[3]:
        reasons.append("condition_metric_material_regression")
    if not emotion[3]:
        reasons.append("emotion_metric_material_regression")
    if not motion_guard_passed:
        reasons.append("flow_or_duration_regression")
    if not reasons:
        reasons.append("lora_material_multimetric_gain_without_regression")
    elif winner == FROZEN_VARIANT:
        reasons.append("near_tie_or_risk_defaults_to_frozen")
    return {
        "winner": winner,
        "lora_material_gain": lora_material_gain,
        "condition": {
            "metrics": condition[0],
            "composite_normalized_gain": condition[1],
            "material_metric_count": condition[2],
            "no_material_regression": condition[3],
        },
        "emotion": {
            "metrics": emotion[0],
            "composite_normalized_gain": emotion[1],
            "material_metric_count": emotion[2],
            "no_material_regression": emotion[3],
        },
        "motion_non_regression": {
            "metrics": guard_rows,
            "passed": motion_guard_passed,
        },
        "reasons": reasons,
    }


def _base_receipt(
    *,
    config: Mapping[str, Any],
    sealed: Mapping[str, Any],
    frozen: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": RECEIPT_ARTIFACT_KIND,
        "cpu_only": True,
        "cuda_visible_devices": "",
        "selection_config": {
            "path": config["config_path"],
            "file_sha256": config["config_sha256"],
        },
        "sealed_ab_receipt": {
            "path": sealed["path"],
            "file_sha256": sealed["file_sha256"],
            "canonical_sha256": sealed["sha256"],
        },
        "decision_policy": deepcopy(DECISION_POLICY),
        "arms": {
            FROZEN_VARIANT: deepcopy(dict(frozen)),
        },
        "data_policy": DATA_POLICY,
        "no_external_data": True,
        "no_kimodo": True,
        "no_hanyang": True,
        "formal_release_eligible": False,
        "human_perceived_emotion_truth": False,
    }


def build_selection_receipt(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a pending or selected receipt without initializing CUDA."""

    if config.get("decision_policy") != DECISION_POLICY:
        raise ValueError("validated winner selection config is required")
    sealed = validate_sealed_ab_receipt(
        config["sealed_ab_receipt"],
        expected_file_sha256=config[
            "expected_sealed_ab_receipt_sha256"
        ],
    )
    arm_config = config["arms"]
    frozen_summary = Path(arm_config[FROZEN_VARIANT]["summary"])
    frozen_checkpoint = Path(arm_config[FROZEN_VARIANT]["checkpoint"])
    frozen = _validate_arm(
        variant=FROZEN_VARIANT,
        summary_path=frozen_summary,
        checkpoint_path=frozen_checkpoint,
        sealed=sealed,
        expected_summary_sha256=arm_config[FROZEN_VARIANT][
            "expected_summary_sha256"
        ],
        expected_checkpoint_sha256=arm_config[FROZEN_VARIANT][
            "expected_checkpoint_sha256"
        ],
    )
    receipt = _base_receipt(
        config=config, sealed=sealed, frozen=frozen
    )
    lora_summary = Path(arm_config[LORA_VARIANT]["summary"])
    lora_checkpoint = Path(arm_config[LORA_VARIANT]["checkpoint"])
    availability = _arm_availability(lora_summary, lora_checkpoint)
    if not all(availability.values()):
        receipt.update(
            {
                "status": PENDING_STATUS,
                "winner_selected": False,
                "eligible_for_v8_1_binding": False,
                "pending_reasons": [
                    "lora_b_summary_missing"
                    if not availability["summary_exists"]
                    else "lora_b_checkpoint_missing"
                ],
                "arms": {
                    **receipt["arms"],
                    LORA_VARIANT: {
                        "variant": LORA_VARIANT,
                        "completed": False,
                        **availability,
                        "summary": str(lora_summary.resolve()),
                        "checkpoint": str(lora_checkpoint.resolve()),
                    },
                },
                "invariant_contract_sha256": None,
                "comparison": None,
                "selected_variant": None,
                "selected_checkpoint": None,
                "selected_condition_cache": None,
            }
        )
        receipt["sha256"] = canonical_sha256(receipt)
        return receipt
    lora = _validate_arm(
        variant=LORA_VARIANT,
        summary_path=lora_summary,
        checkpoint_path=lora_checkpoint,
        sealed=sealed,
        expected_summary_sha256=None,
        expected_checkpoint_sha256=None,
    )
    shared = _shared_pair_contract(frozen, lora, sealed)
    comparison = decide_winner(
        frozen["final_condition_diagnostics"],
        lora["final_condition_diagnostics"],
    )
    winner = comparison["winner"]
    selected = frozen if winner == FROZEN_VARIANT else lora
    receipt.update(
        {
            "status": SELECTED_STATUS,
            "winner_selected": True,
            "eligible_for_v8_1_binding": True,
            "pending_reasons": [],
            "arms": {
                FROZEN_VARIANT: deepcopy(dict(frozen)),
                LORA_VARIANT: deepcopy(dict(lora)),
            },
            "shared_invariant_contract": shared,
            "invariant_contract_sha256": shared["sha256"],
            "comparison": comparison,
            "selected_variant": winner,
            "selected_checkpoint": deepcopy(
                dict(selected["checkpoint"])
            ),
            "selected_condition_cache": deepcopy(
                dict(selected["condition_cache"])
            ),
        }
    )
    receipt["sha256"] = canonical_sha256(receipt)
    return receipt


def validate_selection_receipt(
    receipt: Mapping[str, Any],
    *,
    require_selected: bool,
) -> dict[str, Any]:
    if not isinstance(receipt, Mapping):
        raise ValueError("selection receipt must be a mapping")
    _validate_internal_sha(receipt, context="selection receipt")
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("artifact_kind") != RECEIPT_ARTIFACT_KIND
        or receipt.get("cpu_only") is not True
        or receipt.get("data_policy") != DATA_POLICY
        or receipt.get("no_external_data") is not True
        or receipt.get("no_kimodo") is not True
        or receipt.get("no_hanyang") is not True
        or receipt.get("formal_release_eligible") is not False
        or receipt.get("human_perceived_emotion_truth") is not False
        or receipt.get("decision_policy") != DECISION_POLICY
    ):
        raise ValueError("selection receipt contract changed")
    status = receipt.get("status")
    if status not in {PENDING_STATUS, SELECTED_STATUS}:
        raise ValueError("selection receipt status is invalid")
    if require_selected and status != SELECTED_STATUS:
        raise ValueError("Qwen A/B winner selection is still pending")
    if status == PENDING_STATUS:
        if (
            receipt.get("winner_selected") is not False
            or receipt.get("eligible_for_v8_1_binding") is not False
            or any(
                receipt.get(field) is not None
                for field in (
                    "invariant_contract_sha256",
                    "comparison",
                    "selected_variant",
                    "selected_checkpoint",
                    "selected_condition_cache",
                )
            )
        ):
            raise ValueError("pending receipt carries a selected winner")
    else:
        if (
            receipt.get("winner_selected") is not True
            or receipt.get("eligible_for_v8_1_binding") is not True
            or receipt.get("selected_variant")
            not in {FROZEN_VARIANT, LORA_VARIANT}
            or not _is_sha256(
                receipt.get("invariant_contract_sha256")
            )
            or not isinstance(receipt.get("comparison"), Mapping)
            or not isinstance(
                receipt.get("selected_checkpoint"), Mapping
            )
            or not isinstance(
                receipt.get("selected_condition_cache"), Mapping
            )
        ):
            raise ValueError("selected receipt is incomplete")
    return deepcopy(dict(receipt))


def write_receipt(
    receipt: Mapping[str, Any], output_path: str | Path
) -> Path:
    path = Path(output_path).expanduser().resolve()
    _reject_forbidden_path(path, field="output_receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        dict(receipt),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        stream.write(payload)
        temporary = Path(stream.name)
    os.replace(temporary, path)
    return path


def _raise_if_lora_terminal_without_checkpoint(
    config: Mapping[str, Any],
) -> None:
    """Fail immediately when B has a terminal summary but no checkpoint."""

    arm = config["arms"][LORA_VARIANT]
    summary_path = Path(arm["summary"])
    checkpoint_path = Path(arm["checkpoint"])
    if not summary_path.is_file() or checkpoint_path.is_file():
        return
    summary = _read_json(summary_path, context="LoRA B summary")
    completed = int(summary.get("completed_steps", -1))
    status = summary.get("status")
    terminal = (
        completed >= EXPECTED_STEPS
        or status
        in {
            "experimental_candidate",
            "rejected_condition_collapse_or_quality_gate",
        }
    )
    if terminal:
        raise ValueError(
            "LoRA B has a terminal summary without a selectable checkpoint"
        )


def wait_for_completion(
    config: Mapping[str, Any],
    *,
    poll_seconds: float,
    timeout_seconds: float,
    clock=time.monotonic,
    sleeper=time.sleep,
) -> tuple[dict[str, Any], bool]:
    """Silently wait for B, returning ``(receipt, timed_out)``.

    All static A/B inputs are validated once before waiting.  While B is
    absent, the loop performs only filesystem/terminal-summary checks.  As
    soon as both B artifacts exist, the complete hash, lineage, metric, and
    anti-collapse validation is rerun.  Validation errors are intentionally
    not caught.
    """

    poll_seconds = float(poll_seconds)
    timeout_seconds = float(timeout_seconds)
    if not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError("poll_seconds must be finite and positive")
    if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
        raise ValueError("timeout_seconds must be finite and non-negative")
    receipt = build_selection_receipt(config)
    if receipt["status"] == SELECTED_STATUS:
        return receipt, False
    _raise_if_lora_terminal_without_checkpoint(config)
    started = float(clock())
    lora = config["arms"][LORA_VARIANT]
    summary_path = Path(lora["summary"])
    checkpoint_path = Path(lora["checkpoint"])
    while True:
        if timeout_seconds > 0:
            elapsed = float(clock()) - started
            remaining = timeout_seconds - elapsed
            if remaining <= 0:
                return receipt, True
            wait_seconds = min(poll_seconds, remaining)
        else:
            wait_seconds = poll_seconds
        sleeper(wait_seconds)
        _raise_if_lora_terminal_without_checkpoint(config)
        if not (
            summary_path.is_file() and checkpoint_path.is_file()
        ):
            continue
        receipt = build_selection_receipt(config)
        if receipt["status"] == SELECTED_STATUS:
            return receipt, False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(
            PROJECT_ROOT
            / "configs"
            / "beat2_emotion_hierarchy_v7_qwen_winner_selection.json"
        ),
    )
    parser.add_argument(
        "--require-selected",
        action="store_true",
        help="exit non-zero instead of accepting a pending receipt",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate inputs and print the result without writing",
    )
    parser.add_argument(
        "--wait-for-completion",
        action="store_true",
        help=(
            "silently wait in-process while B is incomplete; validation "
            "errors still fail immediately"
        ),
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=30.0,
        help="poll interval for --wait-for-completion (default: 30)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=0.0,
        help=(
            "wait timeout; 0 means wait indefinitely (default: 0)"
        ),
    )
    args = parser.parse_args()
    if (
        not math.isfinite(args.poll_seconds)
        or args.poll_seconds <= 0
        or not math.isfinite(args.timeout_seconds)
        or args.timeout_seconds < 0
    ):
        parser.error(
            "--poll-seconds must be positive and --timeout-seconds "
            "must be non-negative finite values"
        )
    config = read_config(args.config)
    if args.wait_for_completion:
        receipt, timed_out = wait_for_completion(
            config,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_seconds,
        )
    else:
        receipt = build_selection_receipt(config)
        timed_out = False
    validate_selection_receipt(
        receipt,
        require_selected=bool(args.require_selected and not timed_out),
    )
    if args.validate_only:
        output_path = None
        output_file_sha = None
    else:
        path = write_receipt(receipt, config["output_receipt"])
        output_path = str(path)
        output_file_sha = sha256_file(path)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "winner_selected": receipt["winner_selected"],
                "selected_variant": receipt["selected_variant"],
                "eligible_for_v8_1_binding": receipt[
                    "eligible_for_v8_1_binding"
                ],
                "receipt": output_path,
                "receipt_file_sha256": output_file_sha,
                "receipt_canonical_sha256": receipt["sha256"],
                "timed_out": timed_out,
                "cpu_only": True,
                "no_kimodo": True,
                "no_hanyang": True,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    if timed_out:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
