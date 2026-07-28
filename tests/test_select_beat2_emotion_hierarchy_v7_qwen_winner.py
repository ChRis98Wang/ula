from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from tools import (
    select_beat2_emotion_hierarchy_v7_qwen_winner as selector,
)


def test_library_import_does_not_hide_gpu_from_training_process():
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "consumer-visible-sentinel"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; "
                "import tools.select_beat2_emotion_hierarchy_v7_qwen_winner; "
                "print(os.environ['CUDA_VISIBLE_DEVICES'])"
            ),
        ],
        cwd=selector.PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "consumer-visible-sentinel"


def _gate() -> dict:
    return {
        "passed": True,
        "enforced": True,
        "checks": {
            "aligned_vs_zero_absolute": True,
            "aligned_vs_zero_retention": True,
            "aligned_vs_cross_group_absolute": True,
            "correct_beats_cross_group": True,
            "correct_flow_win_rate": True,
            "duration_mae": True,
            "style_auxiliary_loss": True,
            "all_metrics_finite": True,
        },
        "response_retention_ratio": 1.0,
        "thresholds": {
            "minimum_aligned_vs_zero_prediction_rms": 0.003,
            "minimum_response_retention_ratio": 0.75,
            "minimum_aligned_vs_cross_group_prediction_rms": 0.002,
            "minimum_cross_group_minus_aligned_flow_loss": 0.0001,
            "minimum_correct_flow_win_rate": 0.55,
            "maximum_duration_mae_sec": 1.0,
            "maximum_style_smooth_l1": 1.5,
        },
        "failure_reasons": [],
    }


def _metrics() -> dict[str, float]:
    return {
        "aligned_flow_loss": 0.40,
        "duration_mae_sec": 0.50,
        "aligned_vs_zero_prediction_rms": 0.060,
        "aligned_vs_cross_group_prediction_rms": 0.080,
        "cross_group_minus_aligned_flow_loss": 0.007,
        "correct_flow_win_rate": 0.60,
        "semantic_cross_group_cosine_gap": 0.15,
        "semantic_global_hard_cross_group_margin": 0.04,
        "semantic_global_motion_to_prototype_recall_at_1": 0.50,
        "semantic_global_positive_cosine": 0.50,
        "style_smooth_l1": 0.40,
        "semantic_perceptual_total": 0.30,
    }


def _materially_better_metrics() -> dict[str, float]:
    values = _metrics()
    for name in (
        "aligned_vs_zero_prediction_rms",
        "aligned_vs_cross_group_prediction_rms",
        "cross_group_minus_aligned_flow_loss",
        "correct_flow_win_rate",
        "semantic_cross_group_cosine_gap",
        "semantic_global_hard_cross_group_margin",
        "semantic_global_motion_to_prototype_recall_at_1",
        "semantic_global_positive_cosine",
    ):
        values[name] *= 1.10
    values["aligned_flow_loss"] *= 0.99
    values["duration_mae_sec"] *= 0.99
    return values


def _write_bytes(path: Path, value: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return selector.sha256_file(path)


def _sealed_receipt(tmp_path: Path) -> tuple[Path, dict, dict]:
    foundation = tmp_path / "foundation" / "ula_fm_checkpoint.pt"
    style = tmp_path / "foundation" / "conditions.npz"
    foundation_sha = _write_bytes(foundation, b"foundation")
    style_sha = _write_bytes(style, b"style")
    arms = {}
    arm_paths = {}
    for variant in (selector.FROZEN_VARIANT, selector.LORA_VARIANT):
        output = tmp_path / variant
        cache = tmp_path / "qwen" / f"{variant}.npz"
        adapter = tmp_path / "qwen" / f"{variant}.pt"
        cache_sha = _write_bytes(cache, variant.encode("utf-8"))
        adapter_sha = _write_bytes(
            adapter, f"adapter:{variant}".encode("utf-8")
        )
        arms[variant] = {
            "condition_cache": str(cache.resolve()),
            "condition_cache_sha256": cache_sha,
            "adapter_receipt": {
                "path": str(adapter.resolve()),
                "sha256": adapter_sha,
                "qwen_policy": (
                    "official_base_frozen"
                    if variant == selector.FROZEN_VARIANT
                    else "official_base_plus_beat2_only_lora"
                ),
            },
            "output_dir": str(output.resolve()),
        }
        arm_paths[variant] = {
            "output": output,
            "cache": cache,
            "cache_sha": cache_sha,
        }
    loss = {"flow": 1.0, "position": 1.0}
    batching = {
        "mode": "native_variable_length",
        "target_effective_batch_size": 16,
    }
    sampler = {
        "mode": "source_group_first_uniform_emotion_native_length_v1",
        "exact_resume": True,
    }
    shared = {
        "manifest_sha256": "1" * 64,
        "foundation_checkpoint": str(foundation.resolve()),
        "foundation_checkpoint_sha256": foundation_sha,
        "foundation_training_summary_sha256": "2" * 64,
        "style_condition_cache": str(style.resolve()),
        "style_condition_cache_sha256": style_sha,
        "fixed_split_counts": {
            "train": 8,
            "validation": 2,
            "test": 2,
        },
        "fixed_split_assignments_sha256": "3" * 64,
        "seed": 20260727,
        "steps": selector.EXPECTED_STEPS,
        "training_sha256": "4" * 64,
        "loss_sha256": selector.canonical_sha256(loss),
        "batching_sha256": selector.canonical_sha256(batching),
        "sampler_sha256": selector.canonical_sha256(sampler),
        "sampler_policy": (
            "source_group_first_uniform_emotion_native_length_v1"
        ),
        "negative_policy": (
            "deterministic_cross_semantic_group_shared_flow_state_"
            "per_episode_hinge_v2"
        ),
        "text_dropout_policy": (
            "deterministic_full_text_and_predicted_style_dropout_v1"
        ),
        "same_noise_and_flow_time_implementation": True,
    }
    receipt = {
        "schema_version": 7,
        "artifact_kind": selector.SEALED_AB_ARTIFACT_KIND,
        "status": "contract_validated_training_not_started",
        "smoke_test": False,
        "allowed_config_differences": [
            "expected_qwen_condition_cache_sha256",
            "frozen_condition_cache",
            "output_dir",
            "qwen_condition_variant",
        ],
        "invariant_config_sha256": "5" * 64,
        "paired_cache_identity_sha256": "6" * 64,
        "shared_contract": shared,
        "arms": arms,
        "claims": {
            "contract_parity_only": True,
            "generator_training_completed": False,
            "motion_quality_success_claimed": False,
            "robot_emotion_success_claimed": False,
            "lora_text_alignment_metric_is_motion_success": False,
            "requires_generator_motion_evaluation_after_training": True,
        },
        "data_policy": selector.DATA_POLICY,
        "no_external_data": True,
        "no_kimodo": True,
    }
    receipt["sha256"] = selector.canonical_sha256(receipt)
    path = tmp_path / "sealed" / "ab_receipt.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path, receipt, arm_paths


def _write_arm(
    *,
    variant: str,
    arm_paths: dict,
    sealed: dict,
    metrics: dict[str, float],
    seed: int = 20260727,
    injected_source_path: str | None = None,
) -> tuple[Path, Path]:
    output = arm_paths[variant]["output"]
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / selector.ARM_FILENAMES["checkpoint"]
    summary_path = output / selector.ARM_FILENAMES["summary"]
    shared = sealed["shared_contract"]
    loss = {"flow": 1.0, "position": 1.0}
    batching = {
        "mode": "native_variable_length",
        "target_effective_batch_size": 16,
    }
    sampler = {
        "mode": "source_group_first_uniform_emotion_native_length_v1",
        "exact_resume": True,
    }
    input_contract = {
        "sha256": (
            "7" * 64 if variant == selector.FROZEN_VARIANT else "8" * 64
        ),
        "manifest_sha256": shared["manifest_sha256"],
        "foundation_checkpoint_sha256": shared[
            "foundation_checkpoint_sha256"
        ],
        "style_cache_sha256": shared["style_condition_cache_sha256"],
        "frozen_qwen_cache_sha256": arm_paths[variant]["cache_sha"],
    }
    data_receipt = {
        "foundation": {
            "sha256": shared["foundation_checkpoint_sha256"],
        },
        "manifest": {
            "path": injected_source_path or "/data/BEAT2/train.jsonl",
            "sha256": shared["manifest_sha256"],
        },
        "style_cache": {
            "path": shared["style_condition_cache"],
            "sha256": shared["style_condition_cache_sha256"],
        },
        "frozen_qwen_cache": {
            "path": str(arm_paths[variant]["cache"].resolve()),
            "sha256": arm_paths[variant]["cache_sha"],
            "variant": variant,
        },
        "identity_arrays_sha256": "9" * 64,
        "split_contract_sha256": "a" * 64,
        "no_external_data": True,
        "no_kimodo": True,
    }
    training = {
        "steps": selector.EXPECTED_STEPS,
        "seed": seed,
        "loss": loss,
        "batching": batching,
        "sampler": sampler,
        "negative_policy": shared["negative_policy"],
        "text_dropout_policy": shared["text_dropout_policy"],
        "shared_noise_and_flow_times": True,
        "external_motion_checkpoint_count": 0,
    }
    initial = deepcopy(metrics)
    gate = _gate()
    checkpoint = {
        "schema_version": 7,
        "artifact_kind": selector.CHECKPOINT_ARTIFACT_KIND,
        "condition_policy": selector.CONDITION_POLICY,
        "architecture": "ula_mmdit_v3_adaln",
        "action_dim": 18,
        "condition_dim": 264,
        "global_step": 347000,
        "formal_release_eligible": False,
        "no_external_data": True,
        "no_kimodo": True,
        "foundation_receipt": {
            "sha256": shared["foundation_checkpoint_sha256"],
            "posttrain_step": 267000,
            "dataset_family_whitelist": ["BEAT2"],
            "generator_checkpoint_inputs": [],
        },
        "data_receipt": data_receipt,
        "split_contract": {
            "sha256": "b" * 64,
            "counts": shared["fixed_split_counts"],
        },
        "input_contract": input_contract,
        "training_contract": training,
        "metrics": {
            "initial_condition_diagnostics": initial,
            "final_condition_diagnostics": metrics,
        },
        "anti_collapse_gate": gate,
    }
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha = selector.sha256_file(checkpoint_path)
    summary = {
        "schema_version": 7,
        "artifact_kind": selector.SUMMARY_ARTIFACT_KIND,
        "status": "experimental_candidate",
        "completed_steps": selector.EXPECTED_STEPS,
        "target_steps": selector.EXPECTED_STEPS,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "input_contract_sha256": input_contract["sha256"],
        "initial_condition_diagnostics": initial,
        "final_condition_diagnostics": metrics,
        "anti_collapse_gate": gate,
        "frozen_parameter_audit": {
            "passed": True,
            "changed_frozen_tensor_names": [],
            "maximum_abs_error": 0.0,
        },
        "formal_release_eligible": False,
        "no_external_data": True,
        "no_kimodo": True,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_path, checkpoint_path


def _write_config(
    tmp_path: Path,
    *,
    sealed_path: Path,
    arm_paths: dict,
    frozen_summary: Path,
    frozen_checkpoint: Path,
) -> Path:
    config = {
        "schema_version": selector.SCHEMA_VERSION,
        "artifact_kind": selector.CONFIG_ARTIFACT_KIND,
        "sealed_ab_receipt": str(sealed_path.resolve()),
        "expected_sealed_ab_receipt_sha256": selector.sha256_file(
            sealed_path
        ),
        "arms": {
            selector.FROZEN_VARIANT: {
                "summary": str(frozen_summary.resolve()),
                "checkpoint": str(frozen_checkpoint.resolve()),
                "expected_summary_sha256": selector.sha256_file(
                    frozen_summary
                ),
                "expected_checkpoint_sha256": selector.sha256_file(
                    frozen_checkpoint
                ),
            },
            selector.LORA_VARIANT: {
                "summary": str(
                    (
                        arm_paths[selector.LORA_VARIANT]["output"]
                        / selector.ARM_FILENAMES["summary"]
                    ).resolve()
                ),
                "checkpoint": str(
                    (
                        arm_paths[selector.LORA_VARIANT]["output"]
                        / selector.ARM_FILENAMES["checkpoint"]
                    ).resolve()
                ),
                "expected_summary_sha256": None,
                "expected_checkpoint_sha256": None,
            },
        },
        "decision_policy": deepcopy(selector.DECISION_POLICY),
        "output_receipt": str(
            (tmp_path / "selection" / "receipt.json").resolve()
        ),
    }
    path = tmp_path / "selection_config.json"
    path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _pending_fixture(tmp_path: Path):
    sealed_path, sealed, arm_paths = _sealed_receipt(tmp_path)
    frozen_summary, frozen_checkpoint = _write_arm(
        variant=selector.FROZEN_VARIANT,
        arm_paths=arm_paths,
        sealed=sealed,
        metrics=_metrics(),
    )
    config_path = _write_config(
        tmp_path,
        sealed_path=sealed_path,
        arm_paths=arm_paths,
        frozen_summary=frozen_summary,
        frozen_checkpoint=frozen_checkpoint,
    )
    return (
        selector.read_config(config_path),
        sealed,
        arm_paths,
    )


def test_incomplete_b_produces_self_hashed_pending_receipt(tmp_path: Path):
    config, _, _ = _pending_fixture(tmp_path)
    receipt = selector.build_selection_receipt(config)
    assert receipt["status"] == selector.PENDING_STATUS
    assert receipt["winner_selected"] is False
    assert receipt["eligible_for_v8_1_binding"] is False
    assert receipt["selected_variant"] is None
    assert receipt["invariant_contract_sha256"] is None
    assert receipt["arms"][selector.FROZEN_VARIANT]["completed"] is True
    assert receipt["arms"][selector.LORA_VARIANT]["completed"] is False
    selector.validate_selection_receipt(
        receipt, require_selected=False
    )
    with pytest.raises(ValueError, match="still pending"):
        selector.validate_selection_receipt(
            receipt, require_selected=True
        )


def test_native_wait_silently_transitions_from_pending_to_selected(
    tmp_path: Path,
):
    config, sealed, arm_paths = _pending_fixture(tmp_path)
    now = [0.0]
    calls = []

    def sleeper(seconds: float) -> None:
        calls.append(seconds)
        now[0] += seconds
        _write_arm(
            variant=selector.LORA_VARIANT,
            arm_paths=arm_paths,
            sealed=sealed,
            metrics=_materially_better_metrics(),
        )

    receipt, timed_out = selector.wait_for_completion(
        config,
        poll_seconds=5.0,
        timeout_seconds=30.0,
        clock=lambda: now[0],
        sleeper=sleeper,
    )
    assert calls == [5.0]
    assert timed_out is False
    assert receipt["status"] == selector.SELECTED_STATUS
    assert receipt["selected_variant"] == selector.LORA_VARIANT


def test_native_wait_timeout_returns_pending_without_selection(
    tmp_path: Path,
):
    config, _, _ = _pending_fixture(tmp_path)
    now = [0.0]

    def sleeper(seconds: float) -> None:
        now[0] += seconds

    receipt, timed_out = selector.wait_for_completion(
        config,
        poll_seconds=4.0,
        timeout_seconds=10.0,
        clock=lambda: now[0],
        sleeper=sleeper,
    )
    assert timed_out is True
    assert now[0] == 10.0
    assert receipt["status"] == selector.PENDING_STATUS
    assert receipt["winner_selected"] is False


def test_native_wait_fails_immediately_on_terminal_b_without_checkpoint(
    tmp_path: Path,
):
    config, _, arm_paths = _pending_fixture(tmp_path)
    summary_path = (
        arm_paths[selector.LORA_VARIANT]["output"]
        / selector.ARM_FILENAMES["summary"]
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "status": "rejected_condition_collapse_or_quality_gate",
                "completed_steps": selector.EXPECTED_STEPS,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="terminal summary"):
        selector.wait_for_completion(
            config,
            poll_seconds=5.0,
            timeout_seconds=0.0,
            sleeper=lambda _: pytest.fail("must fail before sleeping"),
        )


def test_material_multimetric_gain_selects_lora(tmp_path: Path):
    config, sealed, arm_paths = _pending_fixture(tmp_path)
    _write_arm(
        variant=selector.LORA_VARIANT,
        arm_paths=arm_paths,
        sealed=sealed,
        metrics=_materially_better_metrics(),
    )
    receipt = selector.build_selection_receipt(config)
    assert receipt["status"] == selector.SELECTED_STATUS
    assert receipt["selected_variant"] == selector.LORA_VARIANT
    assert receipt["comparison"]["lora_material_gain"] is True
    assert receipt["comparison"]["condition"]["material_metric_count"] >= 2
    assert receipt["comparison"]["emotion"]["material_metric_count"] >= 2
    assert receipt["comparison"]["motion_non_regression"]["passed"] is True
    assert selector._is_sha256(receipt["invariant_contract_sha256"])
    selector.validate_selection_receipt(receipt, require_selected=True)


def test_completed_near_tie_receipt_selects_pinned_frozen_artifacts(
    tmp_path: Path,
):
    config, sealed, arm_paths = _pending_fixture(tmp_path)
    _write_arm(
        variant=selector.LORA_VARIANT,
        arm_paths=arm_paths,
        sealed=sealed,
        metrics=_metrics(),
    )
    receipt = selector.build_selection_receipt(config)
    frozen = receipt["arms"][selector.FROZEN_VARIANT]
    assert receipt["status"] == selector.SELECTED_STATUS
    assert receipt["selected_variant"] == selector.FROZEN_VARIANT
    assert receipt["comparison"]["lora_material_gain"] is False
    assert receipt["selected_checkpoint"] == frozen["checkpoint"]
    assert receipt["selected_condition_cache"] == frozen["condition_cache"]
    assert receipt["eligible_for_v8_1_binding"] is True
    selector.validate_selection_receipt(receipt, require_selected=True)


def test_near_tie_defaults_to_frozen():
    frozen = _metrics()
    lora = {
        name: (
            value * 1.01
            if name not in {"aligned_flow_loss", "duration_mae_sec"}
            else value
        )
        for name, value in frozen.items()
    }
    decision = selector.decide_winner(frozen, lora)
    assert decision["winner"] == selector.FROZEN_VARIANT
    assert decision["lora_material_gain"] is False
    assert "near_tie_or_risk_defaults_to_frozen" in decision["reasons"]


def test_one_large_indicator_cannot_select_lora():
    frozen = _metrics()
    lora = deepcopy(frozen)
    lora["aligned_vs_zero_prediction_rms"] *= 3.0
    decision = selector.decide_winner(frozen, lora)
    assert decision["winner"] == selector.FROZEN_VARIANT
    assert decision["condition"]["material_metric_count"] == 1
    assert decision["emotion"]["material_metric_count"] == 0


def test_flow_regression_blocks_otherwise_material_lora():
    frozen = _metrics()
    lora = _materially_better_metrics()
    lora["aligned_flow_loss"] = frozen["aligned_flow_loss"] * 1.02
    decision = selector.decide_winner(frozen, lora)
    assert decision["winner"] == selector.FROZEN_VARIANT
    assert decision["motion_non_regression"]["passed"] is False
    assert "flow_or_duration_regression" in decision["reasons"]


def test_pair_contract_mismatch_fails_closed(tmp_path: Path):
    config, sealed, arm_paths = _pending_fixture(tmp_path)
    _write_arm(
        variant=selector.LORA_VARIANT,
        arm_paths=arm_paths,
        sealed=sealed,
        metrics=_materially_better_metrics(),
        seed=20260728,
    )
    with pytest.raises(ValueError, match="sealed A/B contract"):
        selector.build_selection_receipt(config)


@pytest.mark.parametrize(
    "forbidden_path",
    (
        "/external/" + "han" + "yang/train.jsonl",
        "/external/" + "ki" + "modo/train.jsonl",
    ),
)
def test_external_dataset_source_token_fails_closed(
    tmp_path: Path,
    forbidden_path: str,
):
    config, sealed, arm_paths = _pending_fixture(tmp_path)
    _write_arm(
        variant=selector.LORA_VARIANT,
        arm_paths=arm_paths,
        sealed=sealed,
        metrics=_materially_better_metrics(),
        injected_source_path=forbidden_path,
    )
    with pytest.raises(ValueError, match="forbidden"):
        selector.build_selection_receipt(config)


def test_sealed_receipt_file_tamper_fails_closed(tmp_path: Path):
    config, _, _ = _pending_fixture(tmp_path)
    receipt_path = Path(config["sealed_ab_receipt"])
    receipt_path.write_text(
        receipt_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="file SHA256 mismatch"):
        selector.build_selection_receipt(config)
