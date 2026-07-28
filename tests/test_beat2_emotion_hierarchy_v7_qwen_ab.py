from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools import train_beat2_style_emotion_v2 as base_trainer
from tools.train_beat2_emotion_hierarchy_v7 import (
    _patched_v7_engine,
)
from tools.train_beat2_emotion_hierarchy_v7_lora_qwen import (
    AB_RECEIPT_ARTIFACT_KIND,
    FROZEN_VARIANT,
    LORA_VARIANT,
    build_ab_receipt,
    read_lora_overlay,
)
from upper_body_skeleton.beat2_emotion_hierarchy import (
    load_beat2_emotion_hierarchy_prototypes,
)


PROJECT_ROOT = Path(__file__).parents[1]
OVERLAY_PATH = (
    PROJECT_ROOT
    / "configs"
    / "beat2_emotion_hierarchy_v7_lora_qwen.json"
)


def _paired_configs():
    return read_lora_overlay(OVERLAY_PATH)


def test_checked_in_lora_overlay_builds_strict_generator_ab_receipt():
    frozen, lora, overlay = _paired_configs()
    receipt = build_ab_receipt(
        frozen, lora, overlay_receipt=overlay
    )
    assert receipt["artifact_kind"] == AB_RECEIPT_ARTIFACT_KIND
    assert receipt["status"] == (
        "contract_validated_training_not_started"
    )
    assert set(receipt["arms"]) == {FROZEN_VARIANT, LORA_VARIANT}
    assert receipt["shared_contract"]["seed"] == 20260727
    assert receipt["shared_contract"]["steps"] == 80_000
    assert receipt["shared_contract"]["same_noise_and_flow_time_implementation"]
    assert receipt["no_external_data"] is True
    assert receipt["no_kimodo"] is True
    assert receipt["claims"] == {
        "contract_parity_only": True,
        "generator_training_completed": False,
        "motion_quality_success_claimed": False,
        "robot_emotion_success_claimed": False,
        "lora_text_alignment_metric_is_motion_success": False,
        "requires_generator_motion_evaluation_after_training": True,
    }
    assert (
        receipt["arms"][FROZEN_VARIANT]["condition_cache_sha256"]
        == "a38f43335dfdcff606df06cea25f4e7cb3be43f3bbd01d26f66dfa49a4b6d272"
    )
    assert (
        receipt["arms"][LORA_VARIANT]["condition_cache_sha256"]
        == "38ef15d8baa6f70971882adf4b9eee18449c27ac23457aad6fc229d8f88a1753"
    )
    assert (
        receipt["arms"][FROZEN_VARIANT]["adapter_receipt"][
            "qwen_policy"
        ]
        == "official_base_frozen"
    )
    assert (
        receipt["arms"][LORA_VARIANT]["adapter_receipt"][
            "qwen_policy"
        ]
        == "official_base_plus_beat2_only_lora"
    )


def test_ab_rejects_any_training_difference_outside_qwen_selection():
    frozen, lora, overlay = _paired_configs()
    changed = deepcopy(lora)
    changed["training"]["loss"]["jerk"] *= 2.0
    with pytest.raises(ValueError, match="differs outside"):
        build_ab_receipt(
            frozen, changed, overlay_receipt=overlay
        )


def test_lora_variant_is_used_by_patched_v7_cache_loader():
    _, lora, _ = _paired_configs()
    with _patched_v7_engine(lora):
        _, metadata = base_trainer._load_128d_cache(
            lora["frozen_condition_cache"],
            variant="frozen_base",
            config=lora,
        )
    assert metadata["variant"] == LORA_VARIANT
    assert (
        metadata["adapter_receipt"]["qwen_policy"]
        == "official_base_plus_beat2_only_lora"
    )


def test_lora_hierarchy_prototypes_are_built_from_train_rows_only():
    _, lora, _ = _paired_configs()
    bank = load_beat2_emotion_hierarchy_prototypes(
        lora["manifest_path"],
        lora["frozen_condition_cache"],
        expected_manifest_sha256=lora["expected_manifest_sha256"],
        expected_qwen_variant=LORA_VARIANT,
        prototype_aggregation=lora["emotion_hierarchy"][
            "prototype_aggregation"
        ],
        prototype_consistency_tolerance=lora["emotion_hierarchy"][
            "prototype_consistency_tolerance"
        ],
    )
    metadata = bank.metadata["group_prototype_metadata"]
    assert metadata["qwen_variant"] == LORA_VARIANT
    assert metadata["train_row_count"] == 7_522
    assert metadata["validation_or_test_row_count_used"] == 0
    assert metadata["prototype_count"] == 54
    assert bank.metadata["no_kimodo"] is True


def test_lora_overlay_rejects_cache_with_the_frozen_variant(
    tmp_path: Path,
):
    overlay = json.loads(OVERLAY_PATH.read_text(encoding="utf-8"))
    frozen_cache = (
        PROJECT_ROOT
        / "training"
        / "runs"
        / "beat2_qwen_motion_alignment_ab_v1"
        / "conditions_128d_frozen_base.npz"
    )
    overlay["qwen_condition_cache"] = str(frozen_cache)
    overlay["expected_qwen_condition_cache_sha256"] = (
        "a38f43335dfdcff606df06cea25f4e7cb3be43f3bbd01d26f66dfa49a4b6d272"
    )
    path = tmp_path / "wrong_variant.json"
    path.write_text(json.dumps(overlay), encoding="utf-8")
    frozen, lora, receipt = read_lora_overlay(path)
    with pytest.raises(
        ValueError, match="lora_finetuned 128D cache metadata contract"
    ):
        build_ab_receipt(
            frozen, lora, overlay_receipt=receipt
        )


def test_lora_overlay_hard_rejects_kimodo_anywhere(tmp_path: Path):
    overlay = json.loads(OVERLAY_PATH.read_text(encoding="utf-8"))
    overlay["output_dir"] = str(tmp_path / "kimodo-forbidden")
    path = tmp_path / "forbidden.json"
    path.write_text(json.dumps(overlay), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden external-data token"):
        read_lora_overlay(path)
