from __future__ import annotations

from copy import deepcopy
import fnmatch
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.train_beat2_style_emotion_v2 import (
    BRIDGE_ARTIFACT_KIND,
    CHECKPOINT_ARTIFACT_KIND,
    CONDITION_POLICY,
    DEFAULT_CONFIG,
    MODEL_TRAINABLE_PATTERNS,
    STATE_ARTIFACT_KIND,
    CrossGroupNegativePool,
    TemperedGroupNativeBucketSampler,
    _batch_conditions,
    _condition_pair_losses,
    _load_state,
    _semantic_weight_at_step,
    _write_or_validate_bridge,
    anti_collapse_decision,
    configure_condition_optimizer,
    read_config,
    tempered_group_weights,
    validate_config,
)
from upper_body_skeleton.beat2_condition_control import QwenStyleHead
from upper_body_skeleton.ula_training import UlaMMDiTV3AdaLNModel


def _episode(clip_id: str, group: int, frames: int, weight: float = 1.0):
    return {
        "clip_id": clip_id,
        "actions": np.zeros((frames, 18), dtype=np.float32),
        "condition": np.zeros(264, dtype=np.float32),
        "experimental_semantic_group_index": group,
        "tempered_group_sampling_weight": weight,
        "qwen_text_latent_128d": np.full(128, group + 1, dtype=np.float32),
        "continuous_style_training_target": np.array(
            [group, group + 0.5, group + 1.0], dtype=np.float32
        ),
    }


def test_checked_in_config_is_strong_no_oracle_v2_contract():
    config = read_config(
        Path(__file__).parents[1]
        / "configs"
        / "beat2_text_style_emotion_v2.json"
    )
    assert config["condition_policy"] == CONDITION_POLICY
    assert config["training"]["condition_ranking_weight"] >= 25.0
    assert config["training"]["condition_response_floor_weight"] >= 5.0
    assert config["training"]["condition_response_floor"] >= 0.02
    assert config["training"]["loss"]["planner_duration"] > 0
    assert config["anti_collapse_gates"]["minimum_correct_flow_win_rate"] > 0.5
    assert config["semantic_perceptual"]["enabled"] is True
    assert config["semantic_perceptual"]["outer_weight"] == pytest.approx(0.02)
    assert config["semantic_perceptual"]["warmup_steps"] == 500
    assert config["semantic_perceptual"]["apply_only_to_condition_kept"] is True
    assert (
        config["semantic_perceptual"]["use_global_train_prototype_bank"]
        is True
    )
    assert config["semantic_perceptual"]["use_in_batch_contrastive"] is False
    assert config["semantic_perceptual"]["contrastive_weight"] == 0.0
    assert config["semantic_perceptual"]["global_contrastive_weight"] > 0.0
    assert (
        config["semantic_perceptual"]["prototype_aggregation"]
        == "require_identical"
    )
    assert "global54" in config["training_policy"]
    assert "semantic_perceptual" in config["training_policy"]


def test_semantic_perceptual_weight_has_exact_linear_warmup():
    values = [
        _semantic_weight_at_step(
            step, target_weight=0.02, warmup_steps=500
        )
        for step in (0, 1, 250, 500, 900)
    ]
    assert values == pytest.approx([0.0, 0.00004, 0.01, 0.02, 0.02])


def test_config_rejects_forbidden_external_data_token_anywhere(tmp_path):
    config = deepcopy(DEFAULT_CONFIG)
    config["output_dir"] = str(tmp_path / "forbidden-kimodo-run")
    with pytest.raises(ValueError, match="forbidden external-data token"):
        validate_config(config)


def test_tempered_group_formula_is_exact_up_to_mean_one_normalization():
    counts = {0: 1, 1: 25, 2: 100}
    weights = tempered_group_weights(
        counts, reference_group_size=37.5, maximum_multiplier=4.0
    )
    raw = {
        group: min(4.0, (37.5 / count) ** 0.5)
        for group, count in counts.items()
    }
    assert weights[0] / weights[1] == pytest.approx(raw[0] / raw[1])
    assert weights[1] / weights[2] == pytest.approx(raw[1] / raw[2])
    example_mean = sum(weights[g] * counts[g] for g in counts) / sum(
        counts.values()
    )
    assert example_mean == pytest.approx(1.0)


def test_native_tempered_sampler_is_exact_resumable_and_bucket_homogeneous():
    episodes = [
        _episode("a", 0, 40, 2.0),
        _episode("b", 1, 45, 0.5),
        _episode("c", 0, 70, 2.0),
        _episode("d", 1, 80, 0.5),
    ]
    kwargs = {"buckets": [48, 96], "seed": 12}
    sampler = TemperedGroupNativeBucketSampler(episodes, **kwargs)
    batching = {
        "max_motion_tokens_per_microbatch": 4096,
        "max_attention_elements_per_microbatch": 8_000_000,
    }
    first, first_plan = sampler.sample_microbatch(
        remaining_effective_batch=3,
        semantic_tokens=7,
        max_batch_size=3,
        batching=batching,
    )
    assert {
        48 if row["actions"].shape[0] <= 48 else 96 for row in first
    } == {first_plan["bucket_frames"]}
    state = sampler.state_dict()
    expected, expected_plan = sampler.sample_microbatch(
        remaining_effective_batch=3,
        semantic_tokens=7,
        max_batch_size=3,
        batching=batching,
    )
    resumed = TemperedGroupNativeBucketSampler(episodes, **kwargs)
    resumed.load_state_dict(state)
    observed, observed_plan = resumed.sample_microbatch(
        remaining_effective_batch=3,
        semantic_tokens=7,
        max_batch_size=3,
        batching=batching,
    )
    assert [row["clip_id"] for row in observed] == [
        row["clip_id"] for row in expected
    ]
    assert observed_plan == expected_plan


def test_cross_group_negative_pool_is_deterministic_and_never_same_group():
    episodes = [_episode("a", 0, 48), _episode("b", 1, 48), _episode("c", 2, 48)]
    pool = CrossGroupNegativePool(episodes, seed=9)
    first = pool.select(episodes, step=7, microbatch_index=2)
    second = pool.select(episodes, step=7, microbatch_index=2)
    assert [row["clip_id"] for row in first] == [
        row["clip_id"] for row in second
    ]
    assert all(
        source["experimental_semantic_group_index"]
        != negative["experimental_semantic_group_index"]
        for source, negative in zip(episodes, first)
    )


def test_runtime_batch_condition_never_accepts_oracle_style():
    head = QwenStyleHead(hidden_dim=8)
    rows = [_episode("a", 0, 48), _episode("b", 1, 48)]
    base = torch.zeros(2, 264)
    base[:, 136:] = torch.randn(2, 128)
    keep = torch.tensor([True, False])
    conditions, _, targets = _batch_conditions(
        head, rows, base, keep, device=torch.device("cpu")
    )
    assert torch.count_nonzero(conditions[:, :133]) == 0
    assert torch.count_nonzero(conditions[1]) == 0
    assert targets[0, 0] == 0
    leaked = base.clone()
    leaked[:, 133:136] = targets
    with pytest.raises(RuntimeError, match="oracle style"):
        _batch_conditions(
            head, rows, leaked, keep, device=torch.device("cpu")
        )


def test_optimizer_exposes_only_declared_adaln_condition_path():
    model = UlaMMDiTV3AdaLNModel(
        action_dim=18, hidden_dim=32, layers=1, semantic_tokens=7
    )
    head = QwenStyleHead(hidden_dim=8)
    config = validate_config(DEFAULT_CONFIG)
    _, receipt = configure_condition_optimizer(model, head, config)
    names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert names == set(receipt["model_trainable_tensor_names"])
    assert all(
        any(
            fnmatch.fnmatchcase(name, pattern)
            for pattern in MODEL_TRAINABLE_PATTERNS
        )
        for name in names
    )
    for forbidden in (
        "input.weight",
        "time_mlp.0.weight",
        "blocks.0.attn.in_proj_weight",
        "blocks.0.ffn.0.weight",
        "output.weight",
        "style_condition.0.weight",
    ):
        assert not dict(model.named_parameters())[forbidden].requires_grad


def test_bridge_writer_rejects_oracle_style_in_generator_input(tmp_path):
    conditions = np.zeros((2, 264), dtype=np.float32)
    conditions[:, 136:] = 1.0
    arrays = {
        "clip_ids": np.array(["a", "b"]),
        "prompts": np.array(["one", "two"]),
        "generator_input_conditions": conditions,
        "qwen_text_latents": np.ones((2, 128), dtype=np.float32),
        "continuous_style_targets": np.ones((2, 3), dtype=np.float32),
        "continuous_style_features": np.ones((2, 3), dtype=np.float32),
        "fixed_split_assignments": np.array(["train", "validation"]),
        "speaker_keys": np.array(["s1", "s2"]),
        "semantic_group_indices": np.array([0, 1], dtype=np.int64),
        "trajectory_sha256": np.array(["0" * 64, "1" * 64]),
    }
    receipt = {
        "identity_arrays_sha256": "2" * 64,
        "manifest": {"sha256": "3" * 64},
        "foundation": {"sha256": "4" * 64},
        "frozen_qwen_cache": {"sha256": "5" * 64},
        "style_cache": {"sha256": "6" * 64},
        "semantic_claim": "experimental",
    }
    metadata = _write_or_validate_bridge(tmp_path, arrays, receipt)
    assert metadata["artifact_kind"] == BRIDGE_ARTIFACT_KIND
    leaked = dict(arrays)
    leaked["generator_input_conditions"] = conditions.copy()
    leaked["generator_input_conditions"][:, 133] = 1.0
    with pytest.raises(ValueError, match="oracle style"):
        _write_or_validate_bridge(tmp_path / "leaked", leaked, receipt)


def test_hard_gate_requires_response_retention_gap_and_win_rate():
    initial = {
        "aligned_vs_zero_prediction_rms": 0.02,
    }
    final = {
        "aligned_vs_zero_prediction_rms": 0.015,
        "aligned_vs_cross_group_prediction_rms": 0.01,
        "cross_group_minus_aligned_flow_loss": 0.001,
        "correct_flow_win_rate": 0.7,
        "duration_mae_sec": 0.5,
        "style_smooth_l1": 0.5,
        "semantic_perceptual_total": 1.0,
        "semantic_cross_group_cosine_gap": 0.1,
        "semantic_hard_cross_group_margin": -0.2,
        "semantic_hard_margin_positive_fraction": 0.2,
        "semantic_motion_to_text_group_recall_at_1": 0.1,
        "semantic_text_to_motion_group_recall_at_1": 0.1,
        "semantic_global_contrastive_loss": 3.0,
        "semantic_global_hard_cross_group_margin": -0.2,
        "semantic_global_hard_margin_positive_fraction": 0.2,
        "semantic_global_motion_to_prototype_recall_at_1": 0.1,
        "semantic_global_mean_positive_rank": 20.0,
    }
    gates = deepcopy(DEFAULT_CONFIG["anti_collapse_gates"])
    passed = anti_collapse_decision(initial, final, gates, enforce=True)
    assert passed["passed"]
    collapsed = dict(final)
    collapsed["correct_flow_win_rate"] = 0.4
    failed = anti_collapse_decision(initial, collapsed, gates, enforce=True)
    assert not failed["passed"]
    assert "correct_flow_win_rate" in failed["failure_reasons"]


def test_condition_ranking_is_per_episode_not_a_batch_mean_loophole():
    class ScalarConditionModel(torch.nn.Module):
        def forward_masked(
            self, x_t, _flow_times, conditions, _frame_valid
        ):
            return conditions[:, :1, None].expand_as(x_t)

    actions = torch.zeros(2, 3, 18)
    noise = torch.zeros_like(actions)
    flow_times = torch.tensor([0.25, 0.75])
    aligned = torch.zeros(2, 264)
    aligned[:, 0] = torch.tensor([0.0, 1.0])
    negative = torch.zeros_like(aligned)
    negative[:, 0] = torch.tensor([1.0, 0.9])
    dim_masks = torch.ones(2, 18, dtype=torch.bool)
    frame_valid = torch.ones(2, 3, dtype=torch.bool)
    result = _condition_pair_losses(
        ScalarConditionModel(),
        actions,
        noise,
        flow_times,
        aligned,
        negative,
        dim_masks,
        frame_valid,
        torch.ones(2, dtype=torch.bool),
        response_floor=0.02,
        ranking_margin=0.005,
    )

    # Row zero wins by a large margin while row one loses.  A batch-mean
    # hinge would incorrectly be zero; the per-row hinge must remain active.
    assert result["ranking_gap"] > 0.0
    assert result["ranking_satisfied_fraction"] == torch.tensor(0.5)
    assert result["ranking_loss"] > 0.0
    assert result["reconstructed_aligned_normalized"].shape == actions.shape
    assert torch.count_nonzero(
        result["reconstructed_aligned_normalized"][0]
    ) == 0
    assert torch.allclose(
        result["reconstructed_aligned_normalized"][1],
        torch.full_like(actions[1], 0.25),
    )


def test_resume_loader_rejects_old_artifact_kind_before_mutating(tmp_path):
    path = tmp_path / "old_state.pt"
    torch.save(
        {"artifact_kind": "beat2_experimental_metadata_conditioned_posttrain_state_v1"},
        path,
    )
    with pytest.raises(ValueError, match="isolated text style/emotion V2"):
        _load_state(
            path,
            input_contract={"sha256": "0" * 64},
            target_steps=2,
            model=None,
            style_head=None,
            model_ema=None,
            style_head_ema=None,
            optimizer=None,
            sampler=None,
            device=torch.device("cpu"),
        )


def test_artifact_kinds_are_independent_from_v1():
    assert STATE_ARTIFACT_KIND.endswith("_v2")
    assert CHECKPOINT_ARTIFACT_KIND.endswith("_v2")
