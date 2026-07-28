from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.experimental import (
    build_beat2_emotion_hierarchy_v7_60s as video,
)


PROJECT_ROOT = Path(__file__).parents[1]


def _gate() -> dict:
    return {
        "required_episode_count": 54,
        "required_prototype_count": 6,
        "non_regression_tolerance": 1e-6,
        "minimum_loss_decrease": 0.02,
        "minimum_rank_decrease": 0.1,
        "minimum_margin_increase": 0.02,
        "minimum_clear_improvement_count": 2,
    }


def _diagnostics(
    *,
    loss: float = 0.6,
    rank: float = 3.5,
    margin: float = -0.3,
    batch_recall: float = 0.0,
) -> dict:
    return {
        "episode_count": 54,
        "semantic_global_prototype_count": 6,
        "semantic_global_contrastive_loss": loss,
        "semantic_global_mean_positive_rank": rank,
        "semantic_global_hard_cross_group_margin": margin,
        "semantic_global_motion_to_prototype_recall_at_1": batch_recall,
    }


def test_checked_in_config_is_v7_raw_playback_and_does_not_fake_lora():
    config = json.loads(
        (
            PROJECT_ROOT
            / "configs"
            / "beat2_emotion_hierarchy_v7_60s.json"
        ).read_text(encoding="utf-8")
    )
    interface = config["qwen_generator_comparison"]
    lora = interface["variants"]["lora_finetuned"]

    assert config["schema_version"] == 7
    assert config["minimum_training_steps"] == 80_000
    assert config["target_duration_sec"] == 60.0
    assert config["playback_contract"] == video.PLAYBACK_CONTRACT
    assert interface["same_comparison_pair_seeds"] is True
    assert interface["shared_noise_policy"] == video.SHARED_NOISE_POLICY
    assert interface["active_variant"] == "frozen_base"
    assert lora["condition_cache_sha256"] == (
        "38ef15d8baa6f70971882adf4b9eee18449c27ac23457aad6fc229d8f88a1753"
    )
    assert lora["generation_enabled"] is False
    assert lora["generator_checkpoint"] is None
    assert lora["generator_training_summary"] is None
    assert lora["expected_training_policy"] is None


def test_full_six_emotion_gate_uses_loss_rank_margin_not_batch_recall():
    initial = _diagnostics(batch_recall=1.0)
    final = _diagnostics(
        loss=0.55,
        rank=3.35,
        margin=-0.27,
        batch_recall=0.0,
    )

    decision = video.six_emotion_full_diagnostic_decision(
        initial, final, _gate()
    )

    assert decision["passed"] is True
    assert decision["batch_recall_at_1_used"] is False
    assert decision["clear_improvement_count"] == 3
    assert "recall" not in " ".join(decision["metrics_used"].values())


@pytest.mark.parametrize(
    ("loss", "rank", "margin", "failed_check"),
    [
        (0.6001, 3.3, -0.25, "loss_not_worse_than_step0"),
        (0.5, 3.5001, -0.25, "rank_not_worse_than_step0"),
        (0.5, 3.3, -0.3001, "margin_not_worse_than_step0"),
    ],
)
def test_full_six_emotion_gate_rejects_any_step0_regression(
    loss, rank, margin, failed_check
):
    decision = video.six_emotion_full_diagnostic_decision(
        _diagnostics(),
        _diagnostics(loss=loss, rank=rank, margin=margin),
        _gate(),
    )

    assert decision["passed"] is False
    assert decision["checks"][failed_check] is False


def test_full_gate_requires_multiple_clear_improvements():
    decision = video.six_emotion_full_diagnostic_decision(
        _diagnostics(),
        _diagnostics(loss=0.55, rank=3.5, margin=-0.3),
        _gate(),
    )

    assert decision["clear_improvement_count"] == 1
    assert decision["checks"]["minimum_clear_improvement_count"] is False
    assert decision["passed"] is False


def test_full_gate_rejects_partial_or_batch_sized_diagnostic():
    final = _diagnostics(loss=0.5, rank=3.0, margin=-0.2)
    final["episode_count"] = 16

    decision = video.six_emotion_full_diagnostic_decision(
        _diagnostics(), final, _gate()
    )

    assert decision["passed"] is False
    assert decision["checks"]["final_has_54_fixed_examples"] is False


def test_frozen_and_lora_interface_gets_bit_identical_independent_noise():
    result = video.shared_variant_noise(
        seed=99,
        frames=30,
        variant_names=("frozen_base", "lora_finetuned"),
    )

    assert np.array_equal(result["frozen_base"], result["lora_finetuned"])
    result["frozen_base"][0, 0] += 1.0
    assert not np.array_equal(
        result["frozen_base"], result["lora_finetuned"]
    )


def test_exact_frozen_parameter_audit_is_required():
    assert video._validate_frozen_audit(
        {
            "passed": True,
            "changed_frozen_tensor_names": [],
            "maximum_abs_error": 0.0,
        },
        field="audit",
    )["passed"]

    with pytest.raises(
        video.EmotionHierarchyVideoError,
        match="exact frozen-parameter audit",
    ):
        video._validate_frozen_audit(
            {
                "passed": True,
                "changed_frozen_tensor_names": ["input.weight"],
                "maximum_abs_error": 1e-8,
            },
            field="audit",
        )


def test_waiter_exits_on_rejected_training_without_calling_video(
    tmp_path, monkeypatch
):
    summary = tmp_path / "training_summary_v7.json"
    summary.write_text(
        json.dumps(
            {
                "status": (
                    "rejected_condition_collapse_or_quality_gate"
                )
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 7,
                "artifact_kind": video.CONFIG_ARTIFACT_KIND,
                "data_policy": video.DATA_POLICY,
                "no_external_data": True,
                "no_kimodo": True,
                "generator_training_summary": str(summary),
                "generator_checkpoint": str(tmp_path / "missing.pt"),
            }
        ),
        encoding="utf-8",
    )
    called = False

    def forbidden_build(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("video generation must not start")

    monkeypatch.setattr(video, "build_video", forbidden_build)
    with pytest.raises(
        video.EmotionHierarchyVideoError,
        match="without an admissible artifact",
    ):
        video.wait_for_completion_and_build(config, poll_seconds=0.01)
    assert called is False


def test_v7_output_names_are_independent_from_v2():
    assert video.OUTPUT_FILENAMES["summary"] == "summary_v7.json"
    assert video.OUTPUT_FILENAMES["trajectory"] == "trajectories_v7.npz"
    assert (
        video.OUTPUT_FILENAMES["final_video"]
        == "beat2_emotion_hierarchy_v7_60s.mp4"
    )
