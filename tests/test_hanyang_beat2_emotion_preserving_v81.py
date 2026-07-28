from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from tools import train_beat2_style_emotion_v2 as v7_engine
from tools import train_hanyang_beat2_emotion_preserving_v81 as trainer
from upper_body_skeleton.ula_v2_18d_head import load_contract_checkpoint
from tools import promote_hanyang_beat2_emotion_preserving_v81 as promoter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "hanyang_beat2_emotion_preserving_v81.json"
)


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


class _ToyMaskedFlow(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.linspace(0.1, 0.3, 18))
        self.condition_gain = nn.Parameter(torch.tensor(0.2))

    def forward_masked(
        self,
        x_t: torch.Tensor,
        _times: torch.Tensor,
        conditions: torch.Tensor,
        frame_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        condition = conditions[:, 0, None, None]
        return (
            x_t * self.scale[None, None, :]
            + condition * self.condition_gain
        ) * frame_valid_mask[:, :, None]


class _ConditionSensitiveFlow(nn.Module):
    def forward_masked(
        self,
        x_t: torch.Tensor,
        _times: torch.Tensor,
        conditions: torch.Tensor,
        frame_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        signal = conditions[:, 136:154].reshape(
            conditions.shape[0], 1, 18
        )
        return (0.1 * x_t + signal) * frame_valid_mask[:, :, None]


class _RouteBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Linear(2, 2)
        self.ffn = nn.Linear(2, 2)
        self.modulation = nn.Sequential(nn.Identity(), nn.Linear(2, 2))


class _RouteModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Linear(2, 2)
        self.motion_latent_condition = nn.Linear(2, 2)
        self.condition_pool = nn.Linear(2, 2)
        self.time_mlp = nn.Linear(2, 2)
        self.blocks = nn.ModuleList([_RouteBlock()])
        self.output_modulation = nn.Linear(2, 2)
        self.plan = nn.Linear(2, 2)
        self.duration_head = nn.Linear(2, 1)
        self.output = nn.Linear(2, 2)
        self.semantic_tokens = 1


class _FakeSampler:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.cursor = 0

    def sample_microbatch(self, *, remaining_effective_batch, **_kwargs):
        count = int(remaining_effective_batch)
        rows = [
            {"clip_id": f"{self.prefix}:{self.cursor + index}"}
            for index in range(count)
        ]
        self.cursor += count
        return rows, {"bucket_frames": 2}

    def state_dict(self):
        return {"cursor": self.cursor, "prefix": self.prefix}

    def load_state_dict(self, state):
        if state["prefix"] != self.prefix:
            raise ValueError("sampler prefix changed")
        self.cursor = int(state["cursor"])


def test_checked_in_config_locks_exact_completed_v7_and_is_blocked():
    config = trainer.validate_config(_config())
    assert config["training_policy"] == (
        "beat2_v7_complete_emotion_hierarchy_plus_hanyang_source344_minus_"
        "unapproved_boundary21_safe323_domain_isolated_masked_partial_motion_"
        "response_anchor_v8_1"
    )
    assert (
        config["expected_foundation_checkpoint_sha256"]
        == trainer.FOUNDATION_SHA256
    )
    assert config["beat2_condition_policy"] == trainer.V7_CONDITION_POLICY
    assert config["approval_gate"] == {
        "required": True,
        "status": "blocked",
        "approval_receipt": None,
        "expected_approval_receipt_sha256": None,
    }
    with pytest.raises(RuntimeError, match=trainer.HUMAN_APPROVAL_BLOCKED):
        trainer.require_launch_approval(config)


def test_beat2_condition_is_nonzero_qwen_and_hanyang_is_not_cfg_null():
    beat2 = torch.zeros(3, 264)
    beat2[:, 133:136] = 0.25
    beat2[:, 136:264] = torch.linspace(-1, 1, 128)
    trainer.assert_nonzero_beat2_conditions(beat2)
    with pytest.raises(ValueError, match="nonzero frozen-Qwen"):
        trainer.assert_nonzero_beat2_conditions(torch.zeros_like(beat2))

    hanyang = trainer.hanyang_domain_conditions(3)
    assert torch.all(hanyang[:, 0] == 1)
    assert torch.count_nonzero(hanyang[:, 1:]).item() == 0
    assert not torch.equal(hanyang, torch.zeros_like(hanyang))
    assert torch.count_nonzero(hanyang[:, 133:264]).item() == 0


def test_hanyang_missing_dof_has_zero_gradient_and_no_condition_loss():
    model = _ToyMaskedFlow()
    actions = torch.linspace(-0.8, 0.9, 2 * 6 * 18).reshape(2, 6, 18)
    confidence = torch.ones_like(actions)
    missing = [7, 8, 13, 14, 17]
    confidence[..., missing] = 0
    valid = torch.tensor(
        [
            [True, True, True, True, True, False],
            [True, True, True, True, True, True],
        ]
    )
    confidence *= valid[:, :, None]
    losses = trainer.hanyang_partial_motion_objective(
        model,
        actions,
        confidence,
        torch.tensor([4 / 30, 5 / 30]),
        valid,
        loss_weights={
            "flow": 1.0,
            "position": 1.0,
            "velocity": 0.02,
            "acceleration": 0.0015,
            "jerk": 0.00004,
        },
        noise=torch.linspace(0.9, -0.7, 2 * 6 * 18).reshape(2, 6, 18),
        flow_times=torch.tensor([0.25, 0.75]),
    )
    assert set(losses) == {
        "total",
        "flow",
        "position",
        "velocity",
        "acceleration",
        "jerk",
        "observed_weight_mean",
    }
    losses["total"].backward()
    assert torch.count_nonzero(model.scale.grad[missing]).item() == 0
    assert torch.count_nonzero(model.scale.grad[[0, 3, 6, 12, 16]]).item() > 0


def test_beat2_shuffle_zero_response_and_hierarchy_terms_are_mandatory():
    model = _ConditionSensitiveFlow()
    actions = torch.zeros(2, 4, 18)
    noise = torch.zeros_like(actions)
    times = torch.tensor([0.4, 0.6])
    aligned = torch.zeros(2, 264)
    aligned[0, 136:154] = 0.5
    aligned[1, 136:154] = -0.25
    negative = aligned.roll(1, dims=0)
    masks = torch.ones(2, 18, dtype=torch.bool)
    valid = torch.ones(2, 4, dtype=torch.bool)
    pair = v7_engine._condition_pair_losses(
        model,
        actions,
        noise,
        times,
        aligned,
        negative,
        masks,
        valid,
        torch.ones(2, dtype=torch.bool),
        response_floor=0.04,
        ranking_margin=0.005,
    )
    assert pair["response_rms"].item() > 0
    assert not torch.equal(aligned, negative)
    terms = {
        "flow": torch.tensor(1.0),
        "condition_ranking": pair["ranking_loss"],
        "condition_response_floor": pair["response_floor_loss"],
        "hierarchy_binary_loss": torch.tensor(0.5),
        "hierarchy_emotion_loss": torch.tensor(0.75),
        "hierarchy_group_auxiliary_loss": torch.tensor(0.1),
        "emotion_response_anchor": torch.tensor(0.0),
    }
    trainer.audit_beat2_loss_terms(terms)
    missing = dict(terms)
    missing.pop("hierarchy_emotion_loss")
    with pytest.raises(RuntimeError, match="incomplete"):
        trainer.audit_beat2_loss_terms(missing)


def test_condition_response_anchor_compares_condition_delta():
    teacher_zero = torch.zeros(1, 2, 3)
    teacher_aligned = torch.ones(1, 2, 3)
    observed = torch.ones_like(teacher_zero, dtype=torch.bool)
    equal = trainer.condition_response_anchor_loss(
        teacher_aligned.clone(),
        teacher_zero.clone(),
        teacher_aligned,
        teacher_zero,
        observed,
    )
    collapsed = trainer.condition_response_anchor_loss(
        teacher_zero.clone(),
        teacher_zero.clone(),
        teacher_aligned,
        teacher_zero,
        observed,
    )
    assert equal.item() == 0
    assert collapsed.item() == 1


def test_domain_gradient_firewall_is_disjoint_and_enforced():
    model = _RouteModel()
    style_head = nn.Linear(2, 2)
    routes = trainer.parameter_gradient_routes(model)
    assert set(routes["beat2_condition"]).isdisjoint(
        routes["hanyang_motion"]
    )
    assert any(
        name.endswith("modulation.1.weight")
        for name in routes["beat2_condition"]
    )
    assert all(
        "modulation" not in name
        and "condition" not in name
        and not name.startswith("plan.")
        for name in routes["hanyang_motion"]
    )
    named = dict(model.named_parameters())
    beat2_loss = sum(
        named[name].square().sum()
        for name in routes["beat2_condition"]
    ) + sum(parameter.square().sum() for parameter in style_head.parameters())
    hanyang_loss = sum(
        named[name].square().sum()
        for name in routes["hanyang_motion"]
    )
    receipt = trainer.route_disjoint_gradients(
        model=model,
        style_head=style_head,
        beat2_loss=beat2_loss,
        hanyang_loss=hanyang_loss,
    )
    assert receipt["beat2_style_head"]
    for name, parameter in model.named_parameters():
        if name in routes["beat2_condition"] or name in routes["hanyang_motion"]:
            assert parameter.grad is not None
        else:
            assert parameter.grad is None


def test_exact_five_percent_exposure_and_max_one_hanyang():
    training = trainer.validate_config(_config())["training"]
    quotas = [
        1 if step % training["hanyang_period_steps"]
        < training["hanyang_active_steps_per_period"] else 0
        for step in range(training["hanyang_period_steps"])
    ]
    assert quotas == [1, 1, 1, 1, 0]
    assert max(quotas) == 1
    assert sum(quotas) / (
        len(quotas) * training["effective_batch_size"]
    ) == 0.05
    assert [trainer.paired_slot_quotas(step) for step in range(1, 6)] == [
        (15, 1),
        (15, 1),
        (15, 1),
        (15, 1),
        (16, 0),
    ]
    assert trainer.expected_exposure(
        60000, arm="winner_control_0pct_hanyang"
    ) == {
        "beat2": 912000,
        "hanyang": 0,
        "matched_noop": 48000,
    }
    assert trainer.expected_exposure(
        60000, arm="winner_isolated_5pct_hanyang"
    ) == {
        "beat2": 912000,
        "hanyang": 48000,
        "matched_noop": 0,
    }


def test_source344_excludes_boundary21_to_safe323_in_lineage():
    config = trainer.validate_config(_config())
    strict_rows = [
        json.loads(line)
        for line in Path(config["hanyang_strict_manifest"]).read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    assert len(strict_rows) == 344
    assert all(row["fixed_split_assignment"] in {
        "train", "validation", "test"
    } for row in strict_rows)
    boundary = trainer._validate_hanyang_boundary_exclusion(config)
    assert len(boundary["source_clip_ids"]) == 344
    assert len(boundary["excluded_clip_ids"]) == 21
    assert len(boundary["safe_clip_ids"]) == 323
    assert boundary["review_manifest_sha256"] == (
        trainer.HANYANG_BOUNDARY_REVIEW_MANIFEST_SHA256
    )
    assert boundary["excluded_clip_ids_sha256"] == (
        trainer.HANYANG_EXCLUDED_CLIP_IDS_SHA256
    )
    assert boundary["safe_clip_ids_sha256"] == (
        trainer.HANYANG_SAFE_CLIP_IDS_SHA256
    )
    for arm in (
        "winner_control_0pct_hanyang",
        "winner_isolated_5pct_hanyang",
    ):
        lineage = trainer.build_lineage_contract(config, arm=arm)
        assert lineage["hanyang_strict_count"] == 344
        assert lineage["hanyang_source_pool_count"] == 344
        assert lineage["hanyang_boundary_candidate_count"] == 21
        assert lineage["hanyang_boundary_excluded_count"] == 21
        assert lineage["hanyang_boundary_admitted_count"] == 0
        assert lineage["hanyang_training_eligible_count"] == 323
        assert lineage["hanyang_training_eligible_split_counts"] == {
            "train": 281,
            "validation": 5,
            "test": 37,
        }
        assert lineage["hanyang_safe_clip_ids_sha256"] == (
            trainer.HANYANG_SAFE_CLIP_IDS_SHA256
        )
        assert lineage["hanyang_excluded_clip_ids_sha256"] == (
            trainer.HANYANG_EXCLUDED_CLIP_IDS_SHA256
        )
        assert lineage["kimodo_admitted_count"] == 0
        assert lineage["maximum_hanyang_per_effective_batch"] in {0, 1}
        assert lineage["beat2_fraction"] == 0.95
        assert (
            lineage["hanyang_fraction"]
            + lineage["matched_noop_fraction"]
            == 0.05
        )


def test_pre_device_gate_physically_excludes_boundary_from_every_split():
    config = trainer.validate_config(_config())
    boundary = trainer._validate_hanyang_boundary_exclusion(config)
    episodes, receipt = trainer._pre_device_hanyang_gate(config)
    episode_ids = {str(row["clip_id"]) for row in episodes}
    assert len(episodes) == 323
    assert episode_ids == set(boundary["safe_clip_ids"])
    assert episode_ids.isdisjoint(boundary["excluded_clip_ids"])
    assert {
        split: sum(
            row["fixed_split_assignment"] == split for row in episodes
        )
        for split in ("train", "validation", "test")
    } == {"train": 281, "validation": 5, "test": 37}
    assert receipt["source_episode_count"] == 344
    assert receipt["hanyang_boundary_excluded_count"] == 21
    assert receipt["episode_count"] == 323
    assert receipt["hanyang_condition_labels_masked"] is True
    assert receipt["kimodo_admitted_count"] == 0


def test_winner_overlay_waits_for_ab_then_shares_seed_noise_and_split():
    config = trainer.validate_config(_config())
    selection = config["qwen_ab_selection_gate"]
    assert selection["status"] == "blocked_no_selected_receipt"
    assert selection["selection_receipt"] is None
    assert selection["winner_selected"] is False
    assert selection["selected_foundation_sha256"] is None
    pair = config["winner_overlay_arms"]
    assert pair["shared_selected_foundation_sha256"] is None
    assert pair["shared_selected_qwen_variant"] is None
    assert pair["shared_seed"] == config["training"]["seed"]
    assert pair["shared_noise_seed"] == config["training"]["noise_seed"]
    assert pair["shared_split_contract"] == config["training"]["split_contract"]
    assert pair["launch_policy"] == trainer.WINNER_ARM_LAUNCH_POLICY
    assert (
        pair["winner_control_0pct_hanyang"]["output_dir"]
        != pair["winner_isolated_5pct_hanyang"]["output_dir"]
    )
    assert pair["winner_control_0pct_hanyang"]["beat2_fraction"] == 0.95
    assert (
        pair["winner_control_0pct_hanyang"]["matched_noop_fraction"]
        == 0.05
    )
    assert (
        pair["winner_isolated_5pct_hanyang"]["matched_noop_fraction"]
        == 0.0
    )


def test_diagnostics_reuse_exact_v7_seed_in_audited_runner_contract():
    config = trainer.validate_config(_config())
    v7_config = trainer.v7.read_config(config["v7_reference_config"])
    expected = int(v7_config["seed"]) + 1_000_003
    contract = trainer._runner_input_contract(
        config,
        arm="winner_control_0pct_hanyang",
        target_steps=1,
        v7_config=v7_config,
    )
    assert trainer._v7_diagnostic_seed(v7_config) == expected
    assert contract["diagnostic_seed_policy"] == (
        "exact_v7_root_seed_plus_1000003_v1"
    )
    assert contract["diagnostic_seed"] == expected
    assert contract["diagnostic_seed"] != (
        int(config["training"]["noise_seed"]) + 8_000_003
    )
    assert contract["hanyang_source_pool_count"] == 344
    assert contract["hanyang_boundary_excluded_count"] == 21
    assert contract["hanyang_training_eligible_count"] == 323
    assert contract["hanyang_safe_clip_ids_sha256"] == (
        trainer.HANYANG_SAFE_CLIP_IDS_SHA256
    )
    assert contract["hanyang_excluded_clip_ids_sha256"] == (
        trainer.HANYANG_EXCLUDED_CLIP_IDS_SHA256
    )


@pytest.mark.parametrize(
    "mutation",
    (
        ("expected_foundation_checkpoint_sha256", "0" * 64),
        ("data_contract.hanyang_strict_count", 365),
        ("data_contract.hanyang_boundary_excluded_count", 20),
        ("data_contract.hanyang_boundary_admitted_count", 21),
        ("data_contract.hanyang_training_eligible_count", 344),
        ("expected_hanyang_safe_clip_ids_sha256", "0" * 64),
        ("data_contract.kimodo_admitted_count", 1),
        ("training.hanyang_fraction", 0.06),
        ("domain_isolation.hanyang_cfg_dropout_allowed", True),
        ("domain_isolation.gradient_overlap_allowed", True),
        ("training.condition_ranking_weight", 0.0),
        ("training.condition_response_floor_weight", 0.0),
        ("training.emotion_response_anchor_weight", 0.0),
        ("training.emotion_response_anchor_weight", float("nan")),
        (
            "winner_overlay_arms.winner_control_0pct_hanyang.beat2_fraction",
            1.0,
        ),
        (
            "winner_overlay_arms.launch_policy",
            "sequential_frozen_then_lora",
        ),
        ("qwen_ab_selection_gate.status", "selected_receipt_bound"),
    ),
)
def test_config_fails_closed_on_emotion_data_or_pairing_regression(mutation):
    field, value = mutation
    config = deepcopy(_config())
    target = config
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    with pytest.raises((ValueError, FileNotFoundError)):
        trainer.validate_config(config)


def _fake_v7_config() -> dict:
    return {
        "seed": 20260727,
        "training": {
            "batch_size": 16,
            "lr": 2e-5,
            "style_head_lr": 1e-4,
            "weight_decay": 1e-4,
            "adam_eps": 1e-6,
            "warmup_steps": 1,
            "minimum_lr_ratio": 0.1,
            "batching": {
                "target_effective_batch_size": 16,
                "length_buckets": [2],
            },
            "loss": {
                "flow": 1.0,
                "position": 1.0,
                "velocity": 0.02,
                "acceleration": 0.0015,
                "jerk": 0.00004,
            },
        }
    }


def _approved_fake_config(tmp_path: Path) -> dict:
    config = _config()
    selected = config["qwen_ab_selection_gate"]
    selected.update(
        {
            "status": "selected_receipt_bound",
            "winner_selected": True,
            "selected_qwen_variant": "frozen_base",
            "selected_foundation_checkpoint": config[
                "foundation_checkpoint"
            ],
            "selected_foundation_sha256": trainer.FOUNDATION_SHA256,
            "selected_condition_cache": (
                "/home/gez/shuaiwang/ula-motion-generate/training/runs/"
                "beat2_qwen_motion_alignment_ab_v1/"
                "conditions_128d_frozen_base.npz"
            ),
            "selected_condition_cache_sha256": (
                "a38f43335dfdcff606df06cea25f4e7cb3be43f3bbd01d26f66dfa49a4b6d272"
            ),
        }
    )
    config["approval_gate"]["status"] = "approved"
    for arm in config["winner_overlay_arms"]:
        if arm.startswith("winner_"):
            config["winner_overlay_arms"][arm]["output_dir"] = str(
                tmp_path / arm
            )
    return config


def _fake_runtime(config: dict) -> dict:
    model = _RouteModel()
    teacher = deepcopy(model)
    teacher.requires_grad_(False)
    style_head = nn.Linear(2, 2)
    routes = trainer.parameter_gradient_routes(model)
    named = dict(model.named_parameters())
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [named[name] for name in routes["beat2_condition"]],
                "lr": 2e-5,
                "role": "beat2_v7_condition_path",
            },
            {
                "params": [named[name] for name in routes["hanyang_motion"]],
                "lr": 2e-6,
                "role": "hanyang_motion_backbone",
            },
            {
                "params": list(style_head.parameters()),
                "lr": 1e-4,
                "role": "beat2_qwen_style_head",
            },
        ]
    )
    return {
        "v7_config": _fake_v7_config(),
        "selected_checkpoint": {
            "global_step": 10,
            "architecture": "toy",
            "action_dim": 18,
            "condition_dim": 264,
            "joint_order": [f"j{index}" for index in range(18)],
            "action_stats": {
                "mean": torch.zeros(18),
                "std": torch.ones(18),
            },
            "qwen_style_head_config": {},
        },
        "model": model,
        "teacher": teacher,
        "style_head": style_head,
        "semantic_perceptual": nn.Identity(),
        "optimizer": optimizer,
        "routes": routes,
        "beat2_sampler": _FakeSampler("beat2"),
        "hanyang_sampler": _FakeSampler("hanyang"),
        "negative_pool": object(),
        "diagnostic_rows": [{"clip_id": f"beat2_diag:{index}"} for index in range(54)],
        "hanyang_diagnostic_rows": [
            {"clip_id": f"hanyang_diag:{index}"} for index in range(5)
        ],
    }


def test_cpu_step1_runner_has_matched_beat_receipt_and_exact_resume(
    tmp_path, monkeypatch
):
    config = _approved_fake_config(tmp_path)
    fake_v7 = _fake_v7_config()
    monkeypatch.setattr(
        trainer, "require_launch_approval", lambda value: deepcopy(value)
    )
    monkeypatch.setattr(
        trainer, "_selected_v7_config", lambda _value: fake_v7
    )
    monkeypatch.setattr(
        trainer,
        "_prepare_runner",
        lambda value, *, arm, device, preloaded_hanyang: _fake_runtime(value),
    )
    monkeypatch.setattr(
        trainer,
        "_pre_device_hanyang_gate",
        lambda _value: ([], {}),
    )
    fake_diagnostics = {
        "beat2": {
            "aligned_vs_zero_prediction_rms": 1.0,
            "aligned_vs_cross_group_prediction_rms": 1.0,
            "cross_group_minus_aligned_flow_loss": 1.0,
            "correct_flow_win_rate": 1.0,
            "duration_mae_sec": 0.0,
            "style_smooth_l1": 0.0,
            "aligned_flow_loss": 1.0,
            "semantic_perceptual_total": 0.0,
            "semantic_cross_group_cosine_gap": 1.0,
            "semantic_hard_cross_group_margin": 1.0,
            "semantic_hard_margin_positive_fraction": 1.0,
            "semantic_motion_to_text_group_recall_at_1": 1.0,
            "semantic_text_to_motion_group_recall_at_1": 1.0,
            "semantic_global_contrastive_loss": 0.0,
            "semantic_global_hard_cross_group_margin": 1.0,
            "semantic_global_hard_margin_positive_fraction": 1.0,
            "semantic_global_motion_to_prototype_recall_at_1": 1.0,
            "semantic_global_mean_positive_rank": 1.0,
            "q2_recall_at_1": 1.0,
            "q6_recall_at_1": 1.0,
            "global54_recall_at_1": 1.0,
        },
        "hanyang": {"total": 1.0},
    }
    observed_diagnostic_seeds = []

    def fake_validation_diagnostics(**kwargs):
        observed_diagnostic_seeds.append(kwargs["seed"])
        return deepcopy(fake_diagnostics)

    monkeypatch.setattr(
        trainer,
        "_ema_validation_diagnostics",
        fake_validation_diagnostics,
    )
    monkeypatch.setattr(
        trainer,
        "_diagnostic_candidate_gate",
        lambda **_kwargs: {
            "admissible": True,
            "selection_score": 1.0,
            "minimum_emotion_retention": 1.0,
            "failure_reasons": [],
        },
    )

    def fake_beat_objective(**kwargs):
        model = kwargs["model"]
        style_head = kwargs["style_head"]
        routes = trainer.parameter_gradient_routes(model)
        named = dict(model.named_parameters())
        total = sum(
            named[name].square().sum()
            for name in routes["beat2_condition"]
        ) + sum(
            parameter.square().sum()
            for parameter in style_head.parameters()
        )
        scalar = total * 0 + 0.1
        motion_anchor = sum(
            named[name].square().sum()
            for name in routes["hanyang_motion"]
        ) * 0 + 0.1
        metrics = {
            "total": total,
            "flow": scalar,
            "condition_ranking": scalar,
            "condition_response_floor": scalar,
            "hierarchy_binary_loss": scalar,
            "hierarchy_emotion_loss": scalar,
            "hierarchy_group_auxiliary_loss": scalar,
            "emotion_response_anchor": motion_anchor,
            "position": scalar,
            "velocity": scalar,
            "style_smooth_l1": scalar,
            "condition_response_rms": scalar,
            "semantic_total": scalar,
        }
        return total, metrics, {
            "clip_ids_sha256": trainer.canonical_sha256(
                [row["clip_id"] for row in kwargs["rows"]]
            ),
            "noise_sha256": trainer.canonical_sha256(
                [kwargs["step"], kwargs["microbatch_index"], "noise"]
            ),
            "flow_times_sha256": trainer.canonical_sha256(
                [kwargs["step"], kwargs["microbatch_index"], "times"]
            ),
        }

    monkeypatch.setattr(
        trainer, "_beat2_microbatch_objective", fake_beat_objective
    )
    monkeypatch.setattr(
        trainer,
        "collate_confidence_weighted_18d",
        lambda *_args, **_kwargs: {
            "actions": torch.zeros(1, 2, 18),
            "observation_confidence": torch.ones(1, 2, 18),
            "durations_sec": torch.ones(1),
            "frame_valid_mask": torch.ones(1, 2, dtype=torch.bool),
        },
    )

    def fake_hanyang(model, *_args, **_kwargs):
        named = dict(model.named_parameters())
        total = sum(
            parameter.square().sum()
            for name, parameter in named.items()
            if trainer.hanyang_motion_parameter_name(name)
        )
        return {"total": total, "flow": total}

    monkeypatch.setattr(
        trainer, "hanyang_partial_motion_objective", fake_hanyang
    )
    control = trainer.run_arm(
        config,
        arm="winner_control_0pct_hanyang",
        smoke_test=True,
        smoke_output_dir=tmp_path / "control_smoke",
        device_override="cpu",
    )
    treatment = trainer.run_arm(
        config,
        arm="winner_isolated_5pct_hanyang",
        smoke_test=True,
        smoke_output_dir=tmp_path / "treatment_smoke",
        device_override="cpu",
    )
    assert observed_diagnostic_seeds
    assert set(observed_diagnostic_seeds) == {
        int(fake_v7["seed"]) + 1_000_003
    }
    assert (
        control["last_event"]["beat2_batch_receipts"]
        == treatment["last_event"]["beat2_batch_receipts"]
    )
    assert control["exposure"] == {
        "beat2": 15,
        "hanyang": 0,
        "matched_noop": 1,
    }
    assert treatment["exposure"] == {
        "beat2": 15,
        "hanyang": 1,
        "matched_noop": 0,
    }
    assert control["declared_slot_fractions"] == {
        "beat2": 0.95,
        "hanyang": 0.0,
        "matched_noop": 0.05,
    }
    assert control["actual_slot_fractions"] == {
        "beat2": 15 / 16,
        "hanyang": 0.0,
        "matched_noop": 1 / 16,
    }
    assert control["declared_fraction_assertion_applicable"] is False
    assert control["prefix_schedule_assertion_passed"] is True
    assert control["candidate_available"] is False
    assert control["run_status"] == "technical_smoke_completed_not_candidate"
    assert control["best_admissible"] is None
    assert treatment["last_event"]["gradient_norms_before_clip"][
        "hanyang_motion"
    ] > 0
    for artifact in (control, treatment):
        assert artifact["hanyang_source_pool_count"] == 344
        assert artifact["hanyang_boundary_excluded_count"] == 21
        assert artifact["hanyang_training_eligible_count"] == 323
        assert artifact["hanyang_safe_clip_ids_sha256"] == (
            trainer.HANYANG_SAFE_CLIP_IDS_SHA256
        )
        assert artifact["hanyang_excluded_clip_ids_sha256"] == (
            trainer.HANYANG_EXCLUDED_CLIP_IDS_SHA256
        )
    treatment_checkpoint = torch.load(
        treatment["last_checkpoint"],
        map_location="cpu",
        weights_only=True,
    )
    assert treatment_checkpoint["hanyang_source_pool_count"] == 344
    assert treatment_checkpoint["hanyang_boundary_excluded_count"] == 21
    assert treatment_checkpoint["hanyang_training_eligible_count"] == 323
    resumed = trainer.run_arm(
        config,
        arm="winner_isolated_5pct_hanyang",
        resume=True,
        smoke_test=True,
        smoke_output_dir=tmp_path / "treatment_smoke",
        device_override="cpu",
    )
    assert (
        resumed["last_checkpoint_sha256"]
        == treatment["last_checkpoint_sha256"]
    )

    exact = _approved_fake_config(tmp_path / "exact")
    exact["training"]["steps"] = 2
    exact["training"]["checkpoint_interval"] = 1
    uninterrupted = trainer.run_arm(
        exact,
        arm="winner_isolated_5pct_hanyang",
        device_override="cpu",
    )
    assert uninterrupted["candidate_available"] is True
    assert Path(
        uninterrupted["best_admissible"]["checkpoint"]
    ).is_file()
    interrupted = _approved_fake_config(tmp_path / "interrupted")
    interrupted["training"]["steps"] = 2
    interrupted["training"]["checkpoint_interval"] = 1
    interrupt_once = {"enabled": True}

    def interrupting_objective(**kwargs):
        if kwargs["step"] == 2 and interrupt_once["enabled"]:
            raise RuntimeError("synthetic interruption")
        return fake_beat_objective(**kwargs)

    monkeypatch.setattr(
        trainer, "_beat2_microbatch_objective", interrupting_objective
    )
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        trainer.run_arm(
            interrupted,
            arm="winner_isolated_5pct_hanyang",
            device_override="cpu",
        )
    interrupt_once["enabled"] = False
    resumed_exact = trainer.run_arm(
        interrupted,
        arm="winner_isolated_5pct_hanyang",
        resume=True,
        device_override="cpu",
    )
    left = torch.load(
        uninterrupted["last_checkpoint"],
        map_location="cpu",
        weights_only=True,
    )
    right = torch.load(
        resumed_exact["last_checkpoint"],
        map_location="cpu",
        weights_only=True,
    )
    assert left["exposure"] == right["exposure"]
    assert all(
        torch.equal(left["model_state_dict"][name], value)
        for name, value in right["model_state_dict"].items()
    )
    assert all(
        torch.equal(left["qwen_style_head_state_dict"][name], value)
        for name, value in right["qwen_style_head_state_dict"].items()
    )


def test_unselected_winner_fails_before_prepare_or_cuda(monkeypatch):
    monkeypatch.setattr(
        trainer,
        "_prepare_runner",
        lambda *_args, **_kwargs: pytest.fail("reached data preparation"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: pytest.fail("reached CUDA inspection"),
    )
    with pytest.raises(RuntimeError, match="winner is not selected"):
        trainer.run_arm(
            _config(), arm="winner_isolated_5pct_hanyang"
        )


def test_real_frozen_a_uses_strict_v7_loader_without_relaxing_generic():
    checkpoint_path = (
        PROJECT_ROOT / trainer.FOUNDATION_RELATIVE_PATH
    ).resolve()
    with pytest.raises(ValueError, match="artifact_kind"):
        load_contract_checkpoint(
            checkpoint_path,
            expected_action_dim=18,
            device="cpu",
        )
    model, style_head, checkpoint = trainer.load_v7_checkpoint_for_v81(
        checkpoint_path,
        expected_sha256=trainer.FOUNDATION_SHA256,
        device="cpu",
    )
    assert checkpoint["artifact_kind"] == trainer.v7.CHECKPOINT_ARTIFACT_KIND
    assert model.input.weight.shape == (384, 18)
    assert model.condition_dim == 264
    assert style_head.architecture_config() == checkpoint[
        "qwen_style_head_config"
    ]
    assert all(
        torch.equal(model.state_dict()[name].cpu(), value)
        for name, value in checkpoint["model_state_dict"].items()
    )
    assert all(
        torch.equal(style_head.state_dict()[name].cpu(), value)
        for name, value in checkpoint[
            "qwen_style_head_state_dict"
        ].items()
    )


def _write_synthetic_pending_receipt(tmp_path: Path) -> Path:
    current_path = (
        PROJECT_ROOT
        / "training/runs/beat2_emotion_hierarchy_v7_qwen_ab_selection/"
        "qwen_ab_winner_selection_receipt_v1.json"
    )
    receipt = json.loads(current_path.read_text(encoding="utf-8"))
    receipt.update(
        {
            "status": "pending",
            "winner_selected": False,
            "eligible_for_v8_1_binding": False,
            "pending_reasons": ["lora_b_incomplete"],
            "arms": {"frozen_base": receipt["arms"]["frozen_base"]},
            "invariant_contract_sha256": None,
            "comparison": None,
            "selected_variant": None,
            "selected_checkpoint": None,
            "selected_condition_cache": None,
        }
    )
    receipt.pop("shared_invariant_contract", None)
    receipt["sha256"] = trainer.winner_selector.canonical_sha256(
        {key: value for key, value in receipt.items() if key != "sha256"}
    )
    path = tmp_path / "pending_winner_receipt.json"
    path.write_text(
        json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def test_pending_winner_receipt_cannot_bind_v81(tmp_path):
    config = _config()
    receipt = _write_synthetic_pending_receipt(tmp_path)
    config["qwen_ab_selection_gate"] = {
        "required": True,
        "status": "selected_receipt_bound",
        "selection_receipt": str(receipt),
        "expected_selection_receipt_file_sha256": trainer.sha256_file(
            receipt
        ),
    }
    with pytest.raises(ValueError, match="still pending"):
        trainer.validate_config(config)


def _write_synthetic_selected_receipt(
    tmp_path: Path,
    *,
    selected_cache_from_lora: bool = False,
    tamper_shared_seed: bool = False,
) -> Path:
    pending_path = (
        PROJECT_ROOT
        / "training/runs/beat2_emotion_hierarchy_v7_qwen_ab_selection/"
        "qwen_ab_winner_selection_receipt_v1.json"
    )
    receipt = json.loads(pending_path.read_text(encoding="utf-8"))
    frozen = deepcopy(receipt["arms"]["frozen_base"])
    lora = deepcopy(frozen)
    lora["variant"] = "lora_finetuned"
    lora_cache = (
        PROJECT_ROOT
        / "training/runs/beat2_qwen_motion_alignment_ab_v1/"
        "conditions_128d_lora_finetuned.npz"
    )
    lora["condition_cache"] = {
        "path": str(lora_cache),
        "sha256": trainer.sha256_file(lora_cache),
    }
    shared_values = {
        **deepcopy(frozen["training_invariants"]),
        "foundation_origin_sha256": frozen["foundation_origin_sha256"],
        "checkpoint_global_step": frozen["checkpoint"]["global_step"],
        "checkpoint_split_contract": deepcopy(frozen["split_contract"]),
        "sealed_invariant_config_sha256": "1" * 64,
        "sealed_paired_cache_identity_sha256": "2" * 64,
    }
    if tamper_shared_seed:
        shared_values["seed"] += 1
    shared = {
        "values": shared_values,
        "sha256": trainer.winner_selector.canonical_sha256(shared_values),
        "same_foundation": True,
        "same_seed": True,
        "same_split": True,
        "same_steps": True,
        "same_noise_and_flow_time_implementation": True,
        "same_sampler": True,
        "same_loss_and_batching": True,
    }
    receipt.update(
        {
            "status": "selected",
            "winner_selected": True,
            "eligible_for_v8_1_binding": True,
            "pending_reasons": [],
            "arms": {
                "frozen_base": frozen,
                "lora_finetuned": lora,
            },
            "shared_invariant_contract": shared,
            "invariant_contract_sha256": shared["sha256"],
            "comparison": {"winner": "frozen_base"},
            "selected_variant": "frozen_base",
            "selected_checkpoint": deepcopy(frozen["checkpoint"]),
            "selected_condition_cache": deepcopy(
                lora["condition_cache"]
                if selected_cache_from_lora
                else frozen["condition_cache"]
            ),
        }
    )
    receipt["sha256"] = trainer.winner_selector.canonical_sha256(
        {key: value for key, value in receipt.items() if key != "sha256"}
    )
    path = tmp_path / (
        "tampered_winner_receipt.json"
        if selected_cache_from_lora or tamper_shared_seed
        else "selected_winner_receipt.json"
    )
    path.write_text(
        json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _bind_selected_receipt(config: dict, receipt: Path) -> dict:
    config = deepcopy(config)
    config["qwen_ab_selection_gate"] = {
        "required": True,
        "status": "selected_receipt_bound",
        "selection_receipt": str(receipt),
        "expected_selection_receipt_file_sha256": trainer.sha256_file(
            receipt
        ),
    }
    config["winner_overlay_arms"].update(
        {
            "shared_selected_foundation_sha256": trainer.FOUNDATION_SHA256,
            "shared_selected_qwen_variant": "frozen_base",
            "shared_selected_condition_cache_sha256": (
                "a38f43335dfdcff606df06cea25f4e7cb3be43f3bbd01d26f66dfa49a4b6d272"
            ),
        }
    )
    return config


def test_selected_receipt_cross_binds_variant_checkpoint_cache(tmp_path):
    receipt = _write_synthetic_selected_receipt(tmp_path)
    validated = trainer.validate_config(
        _bind_selected_receipt(_config(), receipt)
    )
    assert validated["qwen_ab_selection_gate"]["winner_selected"] is True
    assert (
        validated["qwen_ab_selection_gate"]["selected_qwen_variant"]
        == "frozen_base"
    )


def test_selected_receipt_rejects_artifact_swap_and_fake_invariant(tmp_path):
    swapped = _write_synthetic_selected_receipt(
        tmp_path, selected_cache_from_lora=True
    )
    with pytest.raises(ValueError, match="do not match selected arm"):
        trainer.validate_config(
            _bind_selected_receipt(_config(), swapped)
        )
    tampered = _write_synthetic_selected_receipt(
        tmp_path, tamper_shared_seed=True
    )
    with pytest.raises(ValueError, match="shared training invariants"):
        trainer.validate_config(
            _bind_selected_receipt(_config(), tampered)
        )


def test_promote_atomically_binds_selected_winner_and_explicit_approval(
    tmp_path,
):
    selected = _write_synthetic_selected_receipt(tmp_path)
    output_config = tmp_path / "derived" / "selected_v81.json"
    output_approval = tmp_path / "derived" / "approval_v81.json"
    result = promoter.promote(
        base_config_path=CONFIG_PATH,
        selected_winner_receipt_path=selected,
        output_config_path=output_config,
        output_approval_receipt_path=output_approval,
        approved_by="unit-test-reviewer",
        approved_utc="2026-07-27T23:30:00Z",
        decision_notes="Explicit unit-test approval for the derived config.",
    )
    assert result["training_started"] is False
    assert result["formal_systemd_launch_allowed_by_this_command"] is False
    assert result["selected_qwen_variant"] == "frozen_base"
    assert output_config.is_file()
    assert output_approval.is_file()
    validated = trainer.read_config(output_config)
    assert validated["qwen_ab_selection_gate"]["winner_selected"] is True
    assert validated["approval_gate"]["status"] == "approved"
    approval = json.loads(output_approval.read_text(encoding="utf-8"))
    assert approval["approved_by"] == "unit-test-reviewer"
    assert approval["approved_utc"] == "2026-07-27T23:30:00Z"
    assert approval["training_launch_allowed"] is True
    assert approval["hanyang_source_pool_count"] == 344
    assert approval["hanyang_boundary_excluded_count"] == 21
    assert approval["hanyang_training_eligible_count"] == 323
    assert approval["hanyang_boundary_manifest_sha256"] == (
        trainer.HANYANG_BOUNDARY_REVIEW_MANIFEST_SHA256
    )
    assert approval["hanyang_safe_clip_ids_sha256"] == (
        trainer.HANYANG_SAFE_CLIP_IDS_SHA256
    )
    assert approval["hanyang_excluded_clip_ids_sha256"] == (
        trainer.HANYANG_EXCLUDED_CLIP_IDS_SHA256
    )
    assert set(result["cpu_smoke_commands"]) == {
        "winner_control_0pct_hanyang",
        "winner_isolated_5pct_hanyang",
    }
    assert not (tmp_path / "derived/winner_control_0pct_hanyang_smoke").exists()
    assert not (
        tmp_path / "derived/winner_isolated_5pct_hanyang_smoke"
    ).exists()


def test_promote_rejects_pending_or_implicit_approval_without_outputs(
    tmp_path,
):
    pending = _write_synthetic_pending_receipt(tmp_path)
    output_config = tmp_path / "selected_v81.json"
    output_approval = tmp_path / "approval_v81.json"
    with pytest.raises(ValueError, match="explicit UTC offset"):
        promoter.promote(
            base_config_path=CONFIG_PATH,
            selected_winner_receipt_path=pending,
            output_config_path=output_config,
            output_approval_receipt_path=output_approval,
            approved_by="reviewer",
            approved_utc="2026-07-27T23:30:00",
            decision_notes="Approval must not infer a timezone.",
        )
    assert not output_config.exists()
    assert not output_approval.exists()
    with pytest.raises(ValueError, match="still pending"):
        promoter.promote(
            base_config_path=CONFIG_PATH,
            selected_winner_receipt_path=pending,
            output_config_path=output_config,
            output_approval_receipt_path=output_approval,
            approved_by="reviewer",
            approved_utc="2026-07-27T23:30:00Z",
            decision_notes="Pending receipt must remain blocked.",
        )
    assert not output_config.exists()
    assert not output_approval.exists()


def test_held_out_emotion_retention_is_a_hard_candidate_gate():
    baseline_beat2 = {
        "aligned_vs_zero_prediction_rms": 1.0,
        "aligned_vs_cross_group_prediction_rms": 1.0,
        "cross_group_minus_aligned_flow_loss": 1.0,
        "correct_flow_win_rate": 1.0,
        "duration_mae_sec": 0.0,
        "style_smooth_l1": 0.0,
        "aligned_flow_loss": 1.0,
        "semantic_perceptual_total": 0.0,
        "semantic_cross_group_cosine_gap": 1.0,
        "semantic_hard_cross_group_margin": 1.0,
        "semantic_hard_margin_positive_fraction": 1.0,
        "semantic_motion_to_text_group_recall_at_1": 1.0,
        "semantic_text_to_motion_group_recall_at_1": 1.0,
        "semantic_global_contrastive_loss": 0.0,
        "semantic_global_hard_cross_group_margin": 1.0,
        "semantic_global_hard_margin_positive_fraction": 1.0,
        "semantic_global_motion_to_prototype_recall_at_1": 1.0,
        "semantic_global_mean_positive_rank": 1.0,
        "q2_recall_at_1": 1.0,
        "q6_recall_at_1": 1.0,
        "global54_recall_at_1": 1.0,
    }
    baseline = {
        "beat2": baseline_beat2,
        "hanyang": {"total": 1.0},
    }
    degraded = deepcopy(baseline)
    degraded["beat2"]["q6_recall_at_1"] = 0.89
    v7_config = trainer.v7.read_config(
        _config()["v7_reference_config"]
    )
    result = trainer._diagnostic_candidate_gate(
        baseline=baseline,
        current=degraded,
        v7_config=v7_config,
        diagnostics_config=_config()["diagnostics"],
    )
    assert result["admissible"] is False
    assert result["retention_checks"]["q6_recall"] is False
    assert "retention:q6_recall" in result["failure_reasons"]

    retained = deepcopy(baseline)
    retained["beat2"]["q6_recall_at_1"] = 0.95
    result = trainer._diagnostic_candidate_gate(
        baseline=baseline,
        current=retained,
        v7_config=v7_config,
        diagnostics_config=_config()["diagnostics"],
    )
    assert result["admissible"] is True
