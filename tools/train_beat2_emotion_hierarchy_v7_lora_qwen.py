#!/usr/bin/env python3
"""Strict LoRA-Qwen arm for the BEAT2 emotion-hierarchy V7 generator A/B.

The checked-in JSON is an overlay on the frozen-Qwen V7 config rather than a
second copy of the full training contract.  This makes it impossible to change
the foundation, split, sampler, seed, steps, loss, batching, or noise policy in
the LoRA arm without failing the paired-config audit.

The LoRA text-alignment metrics are provenance only.  They are never treated as
evidence that the generated robot motion has better emotion or expression.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import train_beat2_emotion_hierarchy_v7 as v7  # noqa: E402
from tools.train_beat2_experimental_metadata_posttrain_ab import (  # noqa: E402
    _load_128d_cache,
    _validate_paired_128d_caches,
)
from upper_body_skeleton.beat2_emotion_hierarchy import (  # noqa: E402
    reject_forbidden_external_tokens,
)
from upper_body_skeleton.ula_v2_18d_head import sha256_file  # noqa: E402


OVERLAY_ARTIFACT_KIND = "beat2_emotion_hierarchy_lora_qwen_overlay_v7"
AB_RECEIPT_ARTIFACT_KIND = (
    "beat2_emotion_hierarchy_qwen_generator_ab_contract_receipt_v7"
)
AB_RECEIPT_FILENAME = "qwen_generator_ab_receipt_v7.json"
LORA_VARIANT = "lora_finetuned"
FROZEN_VARIANT = "frozen_base"
OVERLAY_FIELDS = {
    "schema_version",
    "artifact_kind",
    "base_frozen_config",
    "expected_base_frozen_config_sha256",
    "qwen_condition_variant",
    "qwen_condition_cache",
    "expected_qwen_condition_cache_sha256",
    "output_dir",
}
ALLOWED_AB_CONFIG_DIFFERENCES = frozenset(
    {
        "frozen_condition_cache",
        "qwen_condition_variant",
        "expected_qwen_condition_cache_sha256",
        "output_dir",
    }
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _resolve_project_path(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    reject_forbidden_external_tokens({field: str(path)})
    return path


def _read_json_mapping(path: Path, *, context: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return value


def read_lora_overlay(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Resolve the locked overlay into paired frozen and LoRA V7 configs."""

    overlay_path = Path(path).expanduser().resolve()
    overlay = _read_json_mapping(
        overlay_path, context="V7 LoRA-Qwen overlay"
    )
    reject_forbidden_external_tokens(overlay)
    if set(overlay) != OVERLAY_FIELDS:
        raise ValueError("V7 LoRA-Qwen overlay fields changed")
    if (
        overlay.get("schema_version") != v7.SCHEMA_VERSION
        or overlay.get("artifact_kind") != OVERLAY_ARTIFACT_KIND
        or overlay.get("qwen_condition_variant") != LORA_VARIANT
        or not _is_sha256(
            overlay.get("expected_base_frozen_config_sha256")
        )
        or not _is_sha256(
            overlay.get("expected_qwen_condition_cache_sha256")
        )
    ):
        raise ValueError("V7 LoRA-Qwen overlay contract changed")

    frozen_path = _resolve_project_path(
        overlay["base_frozen_config"], field="base_frozen_config"
    )
    if not frozen_path.is_file():
        raise FileNotFoundError(
            f"frozen V7 config does not exist: {frozen_path}"
        )
    if (
        sha256_file(frozen_path)
        != overlay["expected_base_frozen_config_sha256"]
    ):
        raise ValueError("frozen V7 base config SHA256 changed")
    frozen = v7.read_config(frozen_path)
    if "qwen_condition_variant" in frozen:
        raise ValueError(
            "the frozen V7 base config must retain its legacy frozen arm"
        )

    cache_path = _resolve_project_path(
        overlay["qwen_condition_cache"], field="qwen_condition_cache"
    )
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"LoRA-Qwen condition cache does not exist: {cache_path}"
        )
    if (
        sha256_file(cache_path)
        != overlay["expected_qwen_condition_cache_sha256"]
    ):
        raise ValueError("LoRA-Qwen condition cache SHA256 changed")
    output_dir = _resolve_project_path(
        overlay["output_dir"], field="output_dir"
    )
    if output_dir == Path(frozen["output_dir"]).resolve():
        raise ValueError("LoRA arm must not overwrite the frozen V7 output")

    lora = deepcopy(frozen)
    lora.update(
        {
            "qwen_condition_variant": LORA_VARIANT,
            "expected_qwen_condition_cache_sha256": overlay[
                "expected_qwen_condition_cache_sha256"
            ],
            "frozen_condition_cache": str(cache_path),
            "output_dir": str(output_dir),
        }
    )
    lora = v7.validate_config(lora)
    return frozen, lora, {
        **overlay,
        "path": str(overlay_path),
        "sha256": sha256_file(overlay_path),
        "base_frozen_config": str(frozen_path),
        "qwen_condition_cache": str(cache_path),
        "output_dir": str(output_dir),
    }


def _invariant_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(config))
    for field in ALLOWED_AB_CONFIG_DIFFERENCES:
        value.pop(field, None)
    return value


def build_ab_receipt(
    frozen: Mapping[str, Any],
    lora: Mapping[str, Any],
    *,
    overlay_receipt: Mapping[str, Any],
    smoke_test: bool = False,
) -> dict[str, Any]:
    """Validate and describe the exact generator A/B before any training."""

    frozen = v7.validate_config(frozen)
    lora = v7.validate_config(lora)
    if v7._selected_qwen_variant(frozen) != FROZEN_VARIANT:
        raise ValueError("A arm must select the frozen-base Qwen cache")
    if v7._selected_qwen_variant(lora) != LORA_VARIANT:
        raise ValueError("B arm must select the BEAT2-LoRA Qwen cache")

    frozen_invariant = _invariant_config(frozen)
    lora_invariant = _invariant_config(lora)
    if frozen_invariant != lora_invariant:
        raise ValueError(
            "V7 Qwen A/B differs outside cache, variant, and output_dir"
        )
    if Path(frozen["output_dir"]).resolve() == Path(
        lora["output_dir"]
    ).resolve():
        raise ValueError("V7 Qwen A/B output directories must be distinct")

    frozen_arrays, frozen_metadata = _load_128d_cache(
        frozen["frozen_condition_cache"],
        variant=FROZEN_VARIANT,
        config=frozen,
    )
    lora_arrays, lora_metadata = _load_128d_cache(
        lora["frozen_condition_cache"],
        variant=LORA_VARIANT,
        config=lora,
    )
    if (
        lora_metadata["cache_sha256"]
        != lora["expected_qwen_condition_cache_sha256"]
    ):
        raise ValueError("LoRA cache receipt differs from the locked config")
    if frozen_metadata["cache_sha256"] == lora_metadata["cache_sha256"]:
        raise ValueError("frozen and LoRA Qwen caches must be distinct")
    paired_identity_sha256 = _validate_paired_128d_caches(
        frozen_arrays, lora_arrays
    )
    split_names = frozen_arrays["fixed_split_assignments"].astype(str)
    split_counts = {
        name: int(count)
        for name, count in sorted(Counter(split_names.tolist()).items())
    }
    split_assignments_sha256 = v7.base._mapping_sha256(
        {"fixed_split_assignments": split_names.tolist()}
    )

    shared_metadata_fields = (
        "source_manifest_sha256",
        "csv_set_sha256",
        "speaker_split_contract",
        "prompt_set_sha256",
        "clip_order_sha256",
        "trajectory_order_sha256",
        "count",
        "condition_dim",
        "motion_latent_dim",
        "generator_contract",
    )
    changed_metadata = [
        field
        for field in shared_metadata_fields
        if frozen_metadata.get(field) != lora_metadata.get(field)
    ]
    if changed_metadata:
        raise ValueError(
            "paired Qwen cache lineage differs for "
            + ", ".join(changed_metadata)
        )
    for field in (
        "model_name",
        "revision",
        "source",
        "input_checkpoint_kind",
    ):
        if (frozen_metadata.get("qwen") or {}).get(field) != (
            lora_metadata.get("qwen") or {}
        ).get(field):
            raise ValueError(
                f"paired Qwen base provenance differs for {field}"
            )

    invariant_sha256 = v7.base._mapping_sha256(frozen_invariant)
    payload = {
        "schema_version": v7.SCHEMA_VERSION,
        "artifact_kind": AB_RECEIPT_ARTIFACT_KIND,
        "status": "contract_validated_training_not_started",
        "smoke_test": bool(smoke_test),
        "allowed_config_differences": sorted(
            ALLOWED_AB_CONFIG_DIFFERENCES
        ),
        "invariant_config_sha256": invariant_sha256,
        "paired_cache_identity_sha256": paired_identity_sha256,
        "shared_contract": {
            "manifest_sha256": frozen["expected_manifest_sha256"],
            "foundation_checkpoint": frozen["foundation_checkpoint"],
            "foundation_checkpoint_sha256": sha256_file(
                frozen["foundation_checkpoint"]
            ),
            "foundation_training_summary_sha256": sha256_file(
                frozen["foundation_training_summary"]
            ),
            "style_condition_cache": frozen["style_condition_cache"],
            "style_condition_cache_sha256": sha256_file(
                frozen["style_condition_cache"]
            ),
            "fixed_split_counts": split_counts,
            "fixed_split_assignments_sha256": (
                split_assignments_sha256
            ),
            "seed": int(frozen["seed"]),
            "steps": int(frozen["training"]["steps"]),
            "training_sha256": v7.base._mapping_sha256(
                frozen["training"]
            ),
            "loss_sha256": v7.base._mapping_sha256(
                frozen["training"]["loss"]
            ),
            "batching_sha256": v7.base._mapping_sha256(
                frozen["training"]["batching"]
            ),
            "sampler_sha256": v7.base._mapping_sha256(
                frozen["training"]["sampler"]
            ),
            "sampler_policy": v7.SAMPLING_POLICY,
            "negative_policy": v7.base.NEGATIVE_POLICY,
            "text_dropout_policy": v7.base.TEXT_DROPOUT_POLICY,
            "same_noise_and_flow_time_implementation": True,
        },
        "arms": {
            FROZEN_VARIANT: {
                "condition_cache": frozen_metadata["path"],
                "condition_cache_sha256": frozen_metadata[
                    "cache_sha256"
                ],
                "adapter_receipt": deepcopy(
                    frozen_metadata["adapter_receipt"]
                ),
                "output_dir": frozen["output_dir"],
            },
            LORA_VARIANT: {
                "condition_cache": lora_metadata["path"],
                "condition_cache_sha256": lora_metadata["cache_sha256"],
                "adapter_receipt": deepcopy(
                    lora_metadata["adapter_receipt"]
                ),
                "output_dir": lora["output_dir"],
            },
        },
        "overlay": deepcopy(dict(overlay_receipt)),
        "claims": {
            "contract_parity_only": True,
            "generator_training_completed": False,
            "motion_quality_success_claimed": False,
            "robot_emotion_success_claimed": False,
            "lora_text_alignment_metric_is_motion_success": False,
            "requires_generator_motion_evaluation_after_training": True,
        },
        "data_policy": frozen["data_policy"],
        "no_external_data": True,
        "no_kimodo": True,
    }
    payload["sha256"] = v7.base._mapping_sha256(payload)
    return payload


def write_ab_receipt(
    receipt: Mapping[str, Any], output_dir: str | Path
) -> Path:
    path = Path(output_dir) / AB_RECEIPT_FILENAME
    v7.base._atomic_json_save(dict(receipt), path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(
            PROJECT_ROOT
            / "configs"
            / "beat2_emotion_hierarchy_v7_lora_qwen.json"
        ),
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    frozen, lora, overlay = read_lora_overlay(args.config)
    if args.smoke_test:
        frozen = v7.effective_config(frozen, smoke_test=True)
        lora = v7.effective_config(lora, smoke_test=True)
    receipt = build_ab_receipt(
        frozen,
        lora,
        overlay_receipt=overlay,
        smoke_test=bool(args.smoke_test),
    )
    receipt_path = write_ab_receipt(receipt, lora["output_dir"])
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "receipt": str(receipt_path.resolve()),
                    "receipt_sha256": receipt["sha256"],
                    "no_kimodo": True,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return

    summary = v7.train(
        lora,
        smoke_test=bool(args.smoke_test),
        overwrite=bool(args.overwrite),
        resume=bool(args.resume),
    )
    print(
        json.dumps(
            {
                "ab_receipt": str(receipt_path.resolve()),
                "training_summary": summary,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
