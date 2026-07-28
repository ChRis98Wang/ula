#!/usr/bin/env python3
"""Independent BEAT2 intended-emotion hierarchy V7 trainer.

The numerical training engine is reused from
``train_beat2_style_emotion_v2.py``.  This adapter replaces only its sampler,
frozen semantic critic, and versioned artifact paths while it runs.  It does
not modify or overwrite the V6 config or outputs.

All emotion targets remain official BEAT2 intended-performance weak metadata.
The adapter binds the shared emotion-supervision ingress, requires its current
human-confirmed count to be zero, and never marks a checkpoint as formal or as
perceived robot-affect truth.

Binary and six-emotion losses run simultaneously in this version.  V7 does
not claim to implement a stage-by-stage curriculum schedule.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import train_beat2_style_emotion_v2 as base  # noqa: E402
from upper_body_skeleton.beat2_emotion_hierarchy import (  # noqa: E402
    INTENDED_WEAK_LABEL_ROLE,
    SAMPLING_POLICY,
    Beat2EmotionHierarchyLoss,
    Beat2EmotionHierarchyPerceptualLoss,
    SourceGroupFirstNativeBucketSampler,
    load_beat2_emotion_hierarchy_prototypes,
    reject_forbidden_external_tokens,
    validate_emotion_supervision_binding,
)
from upper_body_skeleton.beat2_semantic_perceptual import (  # noqa: E402
    Beat2SemanticPerceptualLoss,
)


SCHEMA_VERSION = 7
CONFIG_ARTIFACT_KIND = "beat2_emotion_hierarchy_posttrain_config_v7"
BRIDGE_ARTIFACT_KIND = "beat2_emotion_hierarchy_no_oracle_bridge_v7"
STATE_ARTIFACT_KIND = "beat2_emotion_hierarchy_posttrain_state_v7"
CHECKPOINT_ARTIFACT_KIND = "beat2_emotion_hierarchy_generator_v7"
SUMMARY_ARTIFACT_KIND = "beat2_emotion_hierarchy_training_summary_v7"
INPUT_CONTRACT_ARTIFACT_KIND = "beat2_emotion_hierarchy_input_contract_v7"
TRAINING_POLICY = (
    "adaln_qwen_style_source_group_first_binary_six_emotion_"
    "global_prototypes_global54_auxiliary_intended_weak_v7"
)
LABEL_CONTRACT = {
    "label_role": INTENDED_WEAK_LABEL_ROLE,
    "human_perceived_emotion_truth": False,
    "formal_emotion_supervision_enabled": False,
    "human_confirmed_observable_rows_expected": 0,
    "intended_metadata_weak_weight": 0.1,
}
SELECTION_ORDER = "emotion_then_speaker_then_source_group_then_event"
QWEN_CONDITION_VARIANTS = frozenset(
    {"frozen_base", "lora_finetuned"}
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _resolved_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path")
    path = str(Path(value).expanduser().resolve())
    reject_forbidden_external_tokens({field: path})
    return path


def _hierarchy_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("emotion_hierarchy")
    if not isinstance(value, Mapping):
        raise ValueError("emotion_hierarchy must be a mapping")
    return value


def _supervision_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("emotion_supervision_ingress")
    if not isinstance(value, Mapping):
        raise ValueError("emotion_supervision_ingress must be a mapping")
    return value


def _selected_qwen_variant(config: Mapping[str, Any]) -> str:
    """Return the explicit V7 cache variant, preserving the legacy frozen arm."""

    variant = str(config.get("qwen_condition_variant", "frozen_base"))
    if variant not in QWEN_CONDITION_VARIANTS:
        raise ValueError(
            "qwen_condition_variant must be frozen_base or lora_finetuned"
        )
    return variant


def _validate_v7_fields(config: Mapping[str, Any]) -> None:
    reject_forbidden_external_tokens(config)
    exact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": CONFIG_ARTIFACT_KIND,
        "training_policy": TRAINING_POLICY,
        "intended_emotion_label_contract": LABEL_CONTRACT,
    }
    for field, expected in exact.items():
        if config.get(field) != expected:
            raise ValueError(f"{field} must be {expected!r}")

    variant = _selected_qwen_variant(config)
    expected_cache_sha256 = config.get(
        "expected_qwen_condition_cache_sha256"
    )
    if expected_cache_sha256 is not None and not _is_sha256(
        expected_cache_sha256
    ):
        raise ValueError(
            "expected_qwen_condition_cache_sha256 must be a lowercase SHA256"
        )
    if (
        variant == "lora_finetuned"
        and not _is_sha256(expected_cache_sha256)
    ):
        raise ValueError(
            "the LoRA V7 arm requires expected_qwen_condition_cache_sha256"
        )

    hierarchy = _hierarchy_config(config)
    expected_hierarchy = {
        "enabled",
        "binary_weight",
        "emotion_weight",
        "group_auxiliary_weight",
        "temperature",
        "prototype_aggregation",
        "prototype_consistency_tolerance",
        "schedule_mode",
        "weak_label_weight",
        "label_role",
        "human_perceived_emotion_truth",
        "global_prototype_fit_split",
        "validation_or_test_rows_used_for_prototypes",
    }
    if set(hierarchy) != expected_hierarchy:
        raise ValueError("emotion_hierarchy fields changed")
    if (
        hierarchy.get("enabled") is not True
        or hierarchy.get("label_role") != INTENDED_WEAK_LABEL_ROLE
        or hierarchy.get("human_perceived_emotion_truth") is not False
        or hierarchy.get("global_prototype_fit_split") != "train"
        or hierarchy.get("validation_or_test_rows_used_for_prototypes") != 0
        or hierarchy.get("prototype_aggregation") != "require_identical"
        or hierarchy.get("schedule_mode")
        != "simultaneous_hierarchy_no_stage_schedule_v1"
    ):
        raise ValueError("emotion hierarchy weak-label contract changed")
    binary_weight = float(hierarchy["binary_weight"])
    emotion_weight = float(hierarchy["emotion_weight"])
    group_weight = float(hierarchy["group_auxiliary_weight"])
    weak_weight = float(hierarchy["weak_label_weight"])
    temperature = float(hierarchy["temperature"])
    tolerance = float(hierarchy["prototype_consistency_tolerance"])
    if (
        binary_weight <= 0
        or emotion_weight <= 0
        or not 0 < group_weight <= 0.25
        or weak_weight != 0.1
        or temperature <= 0
        or tolerance < 0
    ):
        raise ValueError("emotion hierarchy weights are invalid")

    supervision = _supervision_config(config)
    expected_supervision = {
        "manifest_path",
        "expected_manifest_sha256",
        "audit_path",
        "expected_audit_sha256",
        "required_training_tier",
        "expected_weak_weight",
        "expected_human_confirmed_observable_rows",
    }
    if set(supervision) != expected_supervision:
        raise ValueError("emotion_supervision_ingress fields changed")
    if (
        supervision.get("required_training_tier")
        != "intended_metadata_weak"
        or float(supervision.get("expected_weak_weight", -1)) != 0.1
        or supervision.get("expected_human_confirmed_observable_rows") != 0
        or not _is_sha256(supervision.get("expected_manifest_sha256"))
        or not _is_sha256(supervision.get("expected_audit_sha256"))
    ):
        raise ValueError("emotion supervision ingress contract changed")

    semantic = config.get("semantic_perceptual") or {}
    if (
        float(semantic.get("global_contrastive_weight", -1))
        != group_weight
        or float(semantic.get("temperature", -1)) != temperature
    ):
        raise ValueError(
            "semantic 54-group auxiliary does not match hierarchy config"
        )
    outer_weight = float(semantic.get("outer_weight", -1))
    cosine_weight = float(semantic.get("cosine_weight", -1))
    effective = {
        "aligned_qwen_group_cosine": outer_weight * cosine_weight,
        "binary_intended_emotion": (
            outer_weight * weak_weight * binary_weight
        ),
        "six_intended_emotion": (
            outer_weight * weak_weight * emotion_weight
        ),
        "global54_auxiliary": (
            outer_weight * weak_weight * group_weight
        ),
    }
    primary_floor = min(
        effective["binary_intended_emotion"],
        effective["six_intended_emotion"],
    )
    auxiliary_ceiling = max(
        effective["aligned_qwen_group_cosine"],
        effective["global54_auxiliary"],
    )
    if (
        outer_weight <= 0
        or cosine_weight < 0
        or primary_floor < 10.0 * auxiliary_ceiling
    ):
        raise ValueError(
            "binary/six-emotion effective coefficients must each be at "
            "least ten times every prompt/group auxiliary"
        )
    sampler = (config.get("training") or {}).get("sampler") or {}
    if (
        sampler.get("mode") != SAMPLING_POLICY
        or sampler.get("selection_order") != SELECTION_ORDER
        or sampler.get("emotion_policy") != "uniform_six"
        or sampler.get("event_row_weighting") != "none"
        or sampler.get("exact_resume") is not True
    ):
        raise ValueError("source-group-first sampler contract changed")


def _resolve_v7_paths(config: Mapping[str, Any]) -> dict[str, Any]:
    values = deepcopy(dict(config))
    supervision = dict(_supervision_config(values))
    supervision["manifest_path"] = _resolved_path(
        supervision["manifest_path"],
        field="emotion_supervision_ingress.manifest_path",
    )
    supervision["audit_path"] = _resolved_path(
        supervision["audit_path"],
        field="emotion_supervision_ingress.audit_path",
    )
    values["emotion_supervision_ingress"] = supervision
    return values


def _receipt_contract_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(base._canonical_json(value)).hexdigest()


@contextmanager
def _patched_v7_engine(config: Mapping[str, Any]):
    """Temporarily specialize the shared V6 numerical engine for V7."""

    names = (
        "DEFAULT_CONFIG",
        "SCHEMA_VERSION",
        "CONFIG_ARTIFACT_KIND",
        "BRIDGE_ARTIFACT_KIND",
        "STATE_ARTIFACT_KIND",
        "CHECKPOINT_ARTIFACT_KIND",
        "SUMMARY_ARTIFACT_KIND",
        "TRAINING_POLICY",
        "SAMPLING_POLICY",
        "TemperedGroupNativeBucketSampler",
        "_semantic_perceptual_receipt",
        "_build_semantic_perceptual",
        "_load_128d_cache",
        "_prepare_rows",
        "_paths",
        "_bridge_paths",
        "_input_contract",
    )
    originals = {name: getattr(base, name) for name in names}
    original_receipt = originals["_semantic_perceptual_receipt"]
    original_load_128d_cache = originals["_load_128d_cache"]
    original_prepare_rows = originals["_prepare_rows"]
    receipt_cache: dict[str, Any] = {}
    bank_cache: dict[str, Any] = {}
    selected_variant = _selected_qwen_variant(config)
    explicit_variant = "qwen_condition_variant" in config
    selected_cache_metadata: dict[str, Any] = {}

    def v7_load_128d_cache(
        path: str | Path,
        *,
        variant: str,
        config: Mapping[str, Any],
    ):
        if variant != "frozen_base":
            raise RuntimeError(
                "the shared engine changed its legacy cache request contract"
            )
        arrays, metadata = original_load_128d_cache(
            path,
            variant=selected_variant,
            config=config,
        )
        expected_sha256 = config.get(
            "expected_qwen_condition_cache_sha256"
        )
        if (
            expected_sha256 is not None
            and metadata["cache_sha256"] != expected_sha256
        ):
            raise ValueError(
                f"{selected_variant} Qwen condition cache SHA256 changed"
            )
        selected_cache_metadata.clear()
        selected_cache_metadata.update(deepcopy(metadata))
        return arrays, metadata

    def v7_prepare_rows(
        runtime_config: Mapping[str, Any], *, smoke_test: bool
    ):
        result = original_prepare_rows(
            runtime_config, smoke_test=smoke_test
        )
        if explicit_variant:
            data_receipt = result[3]["data_receipt"]
            qwen_receipt = data_receipt["frozen_qwen_cache"]
            qwen_receipt.update(
                {
                    "variant": selected_variant,
                    "metadata_path": selected_cache_metadata[
                        "metadata_path"
                    ],
                    "adapter_receipt": deepcopy(
                        selected_cache_metadata["adapter_receipt"]
                    ),
                    "prompt_set_sha256": selected_cache_metadata[
                        "prompt_set_sha256"
                    ],
                    "clip_order_sha256": selected_cache_metadata[
                        "clip_order_sha256"
                    ],
                    "trajectory_order_sha256": (
                        selected_cache_metadata[
                            "trajectory_order_sha256"
                        ]
                    ),
                    "legacy_receipt_field_name": "frozen_qwen_cache",
                }
            )
        return result

    def hierarchy_bank():
        if "bank" not in bank_cache:
            hierarchy = _hierarchy_config(config)
            bank_cache["bank"] = load_beat2_emotion_hierarchy_prototypes(
                config["manifest_path"],
                config["frozen_condition_cache"],
                expected_manifest_sha256=config[
                    "expected_manifest_sha256"
                ],
                expected_qwen_variant=selected_variant,
                prototype_aggregation=hierarchy[
                    "prototype_aggregation"
                ],
                prototype_consistency_tolerance=float(
                    hierarchy["prototype_consistency_tolerance"]
                ),
            )
        return bank_cache["bank"]

    def supervision_receipt():
        if "supervision" not in receipt_cache:
            supervision = _supervision_config(config)
            receipt_cache["supervision"] = (
                validate_emotion_supervision_binding(
                    supervision["manifest_path"],
                    supervision["audit_path"],
                    config["manifest_path"],
                    expected_supervision_manifest_sha256=supervision[
                        "expected_manifest_sha256"
                    ],
                    expected_supervision_audit_sha256=supervision[
                        "expected_audit_sha256"
                    ],
                    expected_source_manifest_sha256=config[
                        "expected_manifest_sha256"
                    ],
                    expected_weak_weight=float(
                        supervision["expected_weak_weight"]
                    ),
                )
            )
        return receipt_cache["supervision"]

    def v7_semantic_receipt(
        runtime_config: Mapping[str, Any],
    ) -> dict[str, Any]:
        if "semantic" not in receipt_cache:
            receipt = original_receipt(runtime_config)
            bank = hierarchy_bank()
            hierarchy = _hierarchy_config(runtime_config)
            prototype_metadata = deepcopy(bank.metadata)
            prototype_metadata["contract_sha256"] = (
                _receipt_contract_sha256(prototype_metadata)
            )
            receipt.update(
                {
                    "label_role": INTENDED_WEAK_LABEL_ROLE,
                    "human_perceived_emotion_truth": False,
                    "formal_emotion_supervision_enabled": False,
                    "human_confirmed_observable_rows": 0,
                    "weak_label_weight": float(
                        hierarchy["weak_label_weight"]
                    ),
                    "schedule_mode": hierarchy["schedule_mode"],
                    "effective_coefficients_after_outer_weight": {
                        "aligned_qwen_group_cosine": float(
                            runtime_config["semantic_perceptual"][
                                "outer_weight"
                            ]
                        )
                        * float(
                            runtime_config["semantic_perceptual"][
                                "cosine_weight"
                            ]
                        ),
                        "binary_intended_emotion": float(
                            runtime_config["semantic_perceptual"][
                                "outer_weight"
                            ]
                        )
                        * float(hierarchy["weak_label_weight"])
                        * float(hierarchy["binary_weight"]),
                        "six_intended_emotion": float(
                            runtime_config["semantic_perceptual"][
                                "outer_weight"
                            ]
                        )
                        * float(hierarchy["weak_label_weight"])
                        * float(hierarchy["emotion_weight"]),
                        "global54_auxiliary": float(
                            runtime_config["semantic_perceptual"][
                                "outer_weight"
                            ]
                        )
                        * float(hierarchy["weak_label_weight"])
                        * float(hierarchy["group_auxiliary_weight"]),
                    },
                    "hierarchy": {
                        "binary_weight": float(
                            hierarchy["binary_weight"]
                        ),
                        "emotion_weight": float(
                            hierarchy["emotion_weight"]
                        ),
                        "group_auxiliary_weight": float(
                            hierarchy["group_auxiliary_weight"]
                        ),
                        "temperature": float(hierarchy["temperature"]),
                        "primary_prototype_counts": {
                            "binary": 2,
                            "emotion": 6,
                        },
                        "auxiliary_group_prototype_count": 54,
                        "prototype_metadata": prototype_metadata,
                    },
                    "emotion_supervision_ingress": deepcopy(
                        supervision_receipt()
                    ),
                }
            )
            if explicit_variant:
                receipt["qwen_condition_variant"] = selected_variant
            receipt_cache["semantic"] = receipt
        return deepcopy(receipt_cache["semantic"])

    def v7_build_semantic(
        runtime_config: Mapping[str, Any],
        *,
        action_stats: Mapping[str, Any],
        device: torch.device,
    ):
        receipt = v7_semantic_receipt(runtime_config)
        semantic = runtime_config["semantic_perceptual"]
        hierarchy_config = _hierarchy_config(runtime_config)
        semantic_base = Beat2SemanticPerceptualLoss.from_artifacts(
            descriptor_cache_path=receipt["descriptor_cache"]["path"],
            motion_encoder_checkpoint_path=receipt[
                "motion_encoder_checkpoint"
            ]["path"],
            action_stats=action_stats,
            qwen_condition_cache_path=None,
            cosine_weight=float(semantic["cosine_weight"]),
            contrastive_weight=0.0,
            global_contrastive_weight=0.0,
            temperature=float(hierarchy_config["temperature"]),
            validate_inputs=False,
            device=device,
        )
        hierarchy_loss = Beat2EmotionHierarchyLoss(
            hierarchy_bank(),
            binary_weight=float(hierarchy_config["binary_weight"]),
            emotion_weight=float(hierarchy_config["emotion_weight"]),
            group_auxiliary_weight=float(
                hierarchy_config["group_auxiliary_weight"]
            ),
            weak_label_weight=float(
                hierarchy_config["weak_label_weight"]
            ),
            temperature=float(hierarchy_config["temperature"]),
            validate_inputs=False,
        ).to(device)
        module = Beat2EmotionHierarchyPerceptualLoss(
            semantic_base,
            hierarchy_loss,
            artifact_metadata={
                "data_policy": runtime_config["data_policy"],
                "no_kimodo": True,
                "label_role": INTENDED_WEAK_LABEL_ROLE,
                "formal_release_eligible": False,
                "emotion_supervision_ingress": deepcopy(
                    supervision_receipt()
                ),
                "hierarchy_prototypes": deepcopy(
                    hierarchy_bank().metadata
                ),
            },
        ).to(device)
        module.train()
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise RuntimeError("V7 hierarchy critic must remain frozen")
        return module, receipt

    def v7_paths(runtime_config: Mapping[str, Any]) -> dict[str, Path]:
        root = Path(runtime_config["output_dir"])
        return {
            "root": root,
            "state": root / "last_state_v7.pt",
            "checkpoint": root / "generator_emotion_hierarchy_v7.pt",
            "summary": root / "training_summary_v7.json",
            "progress": root / "progress_v7.jsonl",
        }

    def v7_bridge_paths(output_dir: str | Path) -> tuple[Path, Path]:
        cache = (
            Path(output_dir)
            / "prepared"
            / "emotion_hierarchy_no_oracle_bridge_v7.npz"
        )
        return cache, cache.with_suffix(cache.suffix + ".json")

    def v7_input_contract(
        config_contract: Mapping[str, Any],
        data_receipt: Mapping[str, Any],
        bridge_metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        semantic = data_receipt["semantic_perceptual"]
        supervision = semantic["emotion_supervision_ingress"]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": INPUT_CONTRACT_ARTIFACT_KIND,
            "config_contract_sha256": config_contract["sha256"],
            "manifest_sha256": data_receipt["manifest"]["sha256"],
            "foundation_checkpoint_sha256": data_receipt["foundation"][
                "sha256"
            ],
            "style_cache_sha256": data_receipt["style_cache"]["sha256"],
            "frozen_qwen_cache_sha256": data_receipt[
                "frozen_qwen_cache"
            ]["sha256"],
            "semantic_descriptor_cache_sha256": semantic[
                "descriptor_cache"
            ]["sha256"],
            "semantic_motion_encoder_sha256": semantic[
                "motion_encoder_checkpoint"
            ]["sha256"],
            "emotion_supervision_manifest_sha256": supervision[
                "supervision_manifest"
            ]["sha256"],
            "emotion_supervision_audit_sha256": supervision[
                "supervision_audit"
            ]["sha256"],
            "hierarchy_prototype_contract_sha256": semantic["hierarchy"][
                "prototype_metadata"
            ]["contract_sha256"],
            "identity_arrays_sha256": data_receipt[
                "identity_arrays_sha256"
            ],
            "split_contract_sha256": data_receipt["split_contract_sha256"],
            "bridge_cache_sha256": bridge_metadata["cache_sha256"],
            "label_role": INTENDED_WEAK_LABEL_ROLE,
            "human_perceived_emotion_truth": False,
        }
        if explicit_variant:
            qwen_receipt = data_receipt["frozen_qwen_cache"]
            payload.update(
                {
                    "qwen_condition_variant": selected_variant,
                    "qwen_condition_cache_sha256": qwen_receipt[
                        "sha256"
                    ],
                    "qwen_adapter_checkpoint_sha256": qwen_receipt[
                        "adapter_receipt"
                    ]["sha256"],
                }
            )
        payload["sha256"] = base._mapping_sha256(payload)
        return payload

    replacements = {
        "DEFAULT_CONFIG": deepcopy(dict(config)),
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "CONFIG_ARTIFACT_KIND": CONFIG_ARTIFACT_KIND,
        "BRIDGE_ARTIFACT_KIND": BRIDGE_ARTIFACT_KIND,
        "STATE_ARTIFACT_KIND": STATE_ARTIFACT_KIND,
        "CHECKPOINT_ARTIFACT_KIND": CHECKPOINT_ARTIFACT_KIND,
        "SUMMARY_ARTIFACT_KIND": SUMMARY_ARTIFACT_KIND,
        "TRAINING_POLICY": TRAINING_POLICY,
        "SAMPLING_POLICY": SAMPLING_POLICY,
        "TemperedGroupNativeBucketSampler": (
            SourceGroupFirstNativeBucketSampler
        ),
        "_semantic_perceptual_receipt": v7_semantic_receipt,
        "_build_semantic_perceptual": v7_build_semantic,
        "_load_128d_cache": v7_load_128d_cache,
        "_prepare_rows": v7_prepare_rows,
        "_paths": v7_paths,
        "_bridge_paths": v7_bridge_paths,
        "_input_contract": v7_input_contract,
    }
    try:
        for name, value in replacements.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(base, name, value)


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate V7 plus every reused numerical-engine contract."""

    if not isinstance(config, Mapping):
        raise ValueError("V7 config must be a mapping")
    values = _resolve_v7_paths(config)
    _validate_v7_fields(values)
    with _patched_v7_engine(values):
        values = base.validate_config(values)
    _validate_v7_fields(values)
    return values


def read_config(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_config(value)


def effective_config(
    config: Mapping[str, Any], *, smoke_test: bool
) -> dict[str, Any]:
    values = validate_config(config)
    with _patched_v7_engine(values):
        values = base.effective_config(values, smoke_test=smoke_test)
    return validate_config(values)


def train(
    config: Mapping[str, Any],
    *,
    smoke_test: bool,
    overwrite: bool,
    resume: bool,
) -> dict[str, Any]:
    """Run the isolated V7 experiment through the reused numerical engine."""

    values = validate_config(config)
    with _patched_v7_engine(values):
        return base.train(
            values,
            smoke_test=smoke_test,
            overwrite=overwrite,
            resume=resume,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(
            PROJECT_ROOT / "configs" / "beat2_emotion_hierarchy_v7.json"
        ),
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = effective_config(
        read_config(args.config), smoke_test=bool(args.smoke_test)
    )
    summary = train(
        config,
        smoke_test=bool(args.smoke_test),
        overwrite=bool(args.overwrite),
        resume=bool(args.resume),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
