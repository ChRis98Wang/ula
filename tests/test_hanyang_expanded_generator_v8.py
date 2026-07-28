from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools import train_hanyang_expanded_generator_v8 as trainer
from upper_body_skeleton.hanyang_emotion_retarget import ACTION_DIM_MASK_18D
from upper_body_skeleton.hanyang_expanded_generator import (
    collate_confidence_weighted_18d,
    confidence_weighted_18d_objective,
    load_hanyang_partial_motion_episodes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HANYANG_POOL_ROOT = Path(
    "/home/gez/nas/cloud/gez/human_motion/external_emotion_research/"
    "hanyang_emotional_body_motion_zenodo_10052504_v1/retarget_v1/"
    "experimental_pool_v8"
)
UPSTREAM_PASSED_SHA256 = (
    "be9dcf53f0aa2acc2695475e8625d1f05550a07fb8403e46d0e0c8e3f633daab"
)


class _ToyMaskedFlow(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.full((18,), 0.25))

    def forward_masked(
        self,
        x_t: torch.Tensor,
        _t: torch.Tensor,
        _condition: torch.Tensor,
        frame_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        return x_t * self.scale * frame_valid_mask[:, :, None]


def _partial_batch() -> dict[str, torch.Tensor]:
    actions = torch.linspace(
        -0.8, 0.9, 2 * 6 * 18, dtype=torch.float32
    ).reshape(2, 6, 18)
    confidence = torch.linspace(
        0.2, 1.0, 2 * 6 * 18, dtype=torch.float32
    ).reshape(2, 6, 18)
    confidence[..., [7, 8, 13, 14, 17]] = 0.0
    valid = torch.tensor(
        [
            [True, True, True, True, True, False],
            [True, True, True, True, True, True],
        ]
    )
    confidence = confidence * valid[:, :, None]
    return {
        "actions": actions,
        "conditions": torch.zeros(2, 264),
        "confidence": confidence,
        "durations": torch.tensor([4.0 / 30.0, 5.0 / 30.0]),
        "valid": valid,
        "noise": torch.linspace(
            0.7, -0.6, 2 * 6 * 18, dtype=torch.float32
        ).reshape(2, 6, 18),
        "times": torch.tensor([0.25, 0.75]),
    }


def _loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return confidence_weighted_18d_objective(
        model,
        batch["actions"],
        batch["conditions"],
        batch["confidence"],
        batch["durations"],
        batch["valid"],
        loss_weights={
            "flow": 1.0,
            "position": 1.0,
            "velocity": 0.02,
            "acceleration": 0.0015,
            "jerk": 0.00004,
        },
        noise=batch["noise"],
        flow_times=batch["times"],
        require_hanyang_partial_weights=True,
    )


def test_partial_missing_dimensions_and_padding_have_zero_gradient_and_effect():
    model = _ToyMaskedFlow()
    batch = _partial_batch()
    first = _loss(model, batch)
    first["total"].backward()
    assert torch.count_nonzero(model.scale.grad[[7, 8, 13, 14, 17]]) == 0
    assert torch.count_nonzero(model.scale.grad[[0, 3, 6, 12, 16]]) > 0

    changed = {key: value.clone() for key, value in batch.items()}
    changed["actions"][..., [7, 8, 13, 14, 17]] = 1e6
    changed["noise"][..., [7, 8, 13, 14, 17]] = -1e6
    changed["actions"][0, 5] = 2e6
    changed["noise"][0, 5] = -2e6
    second = _loss(model, changed)
    assert torch.equal(first["total"].detach(), second["total"].detach())


@pytest.mark.parametrize(
    "weights",
    (
        {
            "flow": -1.0,
            "position": 0.0,
            "velocity": 0.0,
            "acceleration": 0.0,
            "jerk": 0.0,
        },
        {
            "flow": float("nan"),
            "position": 0.0,
            "velocity": 0.0,
            "acceleration": 0.0,
            "jerk": 0.0,
        },
        {
            "flow": 0.0,
            "position": 0.0,
            "velocity": 0.0,
            "acceleration": 0.0,
            "jerk": 0.0,
        },
    ),
)
def test_objective_rejects_invalid_loss_weights(weights):
    batch = _partial_batch()
    with pytest.raises(ValueError, match="loss weights"):
        confidence_weighted_18d_objective(
            _ToyMaskedFlow(),
            batch["actions"],
            batch["conditions"],
            batch["confidence"],
            batch["durations"],
            batch["valid"],
            loss_weights=weights,
            noise=batch["noise"],
            flow_times=batch["times"],
            require_hanyang_partial_weights=True,
        )


def test_objective_rejects_non_prefix_frame_mask():
    batch = _partial_batch()
    batch["valid"][0] = torch.tensor(
        [True, True, False, True, True, False]
    )
    batch["confidence"][0] *= batch["valid"][0, :, None]
    with pytest.raises(ValueError, match="contiguous"):
        _loss(_ToyMaskedFlow(), batch)


def test_collator_uses_confidence_presence_not_fractional_amplitude():
    actions = np.linspace(-1.0, 1.0, 6 * 18, dtype=np.float32).reshape(
        6, 18
    )
    confidence = np.ones_like(actions)
    confidence[..., [7, 8, 13, 14, 17]] = 0.0
    confidence[:, 5] = 0.25
    episode = {
        "clip_id": "hanyang:test",
        "actions": actions,
        "observation_confidence": confidence,
        "condition": np.zeros(264, dtype=np.float32),
        "action_dim_mask": np.asarray(ACTION_DIM_MASK_18D),
        "fps": 30.0,
        "duration_sec": 5.0 / 30.0,
    }
    stats = {
        "mean": torch.zeros(18),
        "std": torch.ones(18),
    }
    batch = collate_confidence_weighted_18d(
        [episode], buckets=[8], action_stats=stats, device="cpu"
    )
    assert torch.equal(batch["actions"][0, :6, 5], torch.from_numpy(actions[:, 5]))
    assert torch.equal(
        batch["observation_confidence"][0, :6, 5],
        torch.full((6,), 0.25),
    )
    assert torch.count_nonzero(
        batch["actions"][0, :, [7, 8, 13, 14, 17]]
    ) == 0


@pytest.mark.skipif(
    not (HANYANG_POOL_ROOT / "manifest.jsonl").is_file(),
    reason="completed Hanyang pool is unavailable",
)
def test_completed_hanyang_pool_loads_all_344_as_unconditional_motion_only():
    manifest = HANYANG_POOL_ROOT / "manifest.jsonl"
    receipt = HANYANG_POOL_ROOT / "admission_receipt.json"
    episodes, loader_receipt = load_hanyang_partial_motion_episodes(
        manifest,
        expected_manifest_sha256=trainer.sha256_file(manifest),
        pool_receipt_path=receipt,
        expected_pool_receipt_sha256=trainer.sha256_file(receipt),
        expected_upstream_passed_manifest_sha256=UPSTREAM_PASSED_SHA256,
    )
    assert len(episodes) == 344
    assert loader_receipt["split_counts"] == {
        "train": 291,
        "validation": 10,
        "test": 43,
    }
    assert all(not np.any(episode["condition"]) for episode in episodes)
    assert all(
        episode["emotion_conditioning_mask"] is False
        and episode["style_supervision_mask"] is False
        and episode["semantic_supervision_mask"] is False
        and episode["duration_supervision_mask"] is False
        for episode in episodes
    )


def test_persistent_training_is_blocked_before_data_prepare(monkeypatch):
    config_path = PROJECT_ROOT / "configs/hanyang_beat2_expanded_generator_v8.json"
    if not config_path.is_file():
        pytest.skip("V8 config not generated yet")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["human_review"]["approved"] = False
    config["human_review"]["approval_receipt"] = None
    config["human_review"]["expected_approval_receipt_sha256"] = None

    def forbidden_prepare(*_args, **_kwargs):
        raise AssertionError("blocked run reached data preparation")

    monkeypatch.setattr(trainer, "_prepare", forbidden_prepare)
    with pytest.raises(RuntimeError, match=trainer.HUMAN_REVIEW_BLOCKED):
        trainer.train(
            deepcopy(config),
            smoke_test=False,
            overwrite=False,
            resume=False,
        )


def test_current_config_remains_blocked_pending_emotion_condition_revision():
    config_path = PROJECT_ROOT / "configs/hanyang_beat2_expanded_generator_v8.json"
    if not config_path.is_file():
        pytest.skip("V8 config not generated yet")
    validated = trainer.validate_config(
        json.loads(config_path.read_text(encoding="utf-8"))
    )
    review = validated["human_review"]
    assert review["required"] is True
    assert review["approved"] is False
    assert review["approval_receipt"] is None
    assert review["expected_approval_receipt_sha256"] is None


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("steps", 60000.9),
        ("steps", 60001),
        ("lr", float("nan")),
        ("weight_decay", -1.0),
        ("ema_decay", 2.0),
    ),
)
def test_config_rejects_invalid_training_scalars(field, value):
    config = json.loads(
        (
            PROJECT_ROOT
            / "configs/hanyang_beat2_expanded_generator_v8.json"
        ).read_text(encoding="utf-8")
    )
    config["training"][field] = value
    with pytest.raises(ValueError):
        trainer.validate_config(config)


def test_config_rejects_nan_or_all_zero_loss_weights():
    config = json.loads(
        (
            PROJECT_ROOT
            / "configs/hanyang_beat2_expanded_generator_v8.json"
        ).read_text(encoding="utf-8")
    )
    config["training"]["loss"]["jerk"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        trainer.validate_config(config)
    for name in config["training"]["loss"]:
        config["training"]["loss"][name] = 0.0
    with pytest.raises(ValueError, match="all be zero"):
        trainer.validate_config(config)


def test_checkpoint_payload_carries_durable_exact_source_and_cache_lineage():
    config = json.loads(
        (
            PROJECT_ROOT
            / "configs/hanyang_beat2_expanded_generator_v8.json"
        ).read_text(encoding="utf-8")
    )
    split = {
        "sha256": "1" * 64,
        "source_counts": {
            "beat2": {"train": 7522, "validation": 1629, "test": 2988},
            "hanyang": {"train": 291, "validation": 10, "test": 43},
        },
    }
    contracts = {
        "admission": {"sha256": "2" * 64, "formal_release_eligible": False},
        "data": {
            "sha256": "3" * 64,
            "source_whitelist_exact": [
                "beat2_official_semantic_event_training_pool_v7",
                "hanyang_duksung_emotional_body_motion_v1",
            ],
        },
        "split": split,
        "normalizer": {
            "sha256": "4" * 64,
            "hanyang_refit_performed": False,
        },
    }
    input_contract = {
        "sha256": "5" * 64,
        "config_sha256": "6" * 64,
    }
    payload = trainer._checkpoint_payload(
        model_state={"weight": torch.zeros(1)},
        foundation={
            "architecture": "ula_mmdit_v3_adaln",
            "joint_order": [f"joint_{index}" for index in range(18)],
            "config": {},
            "action_stats": {
                "mean": torch.zeros(18),
                "std": torch.ones(18),
            },
            "global_step": 267000,
        },
        contracts=contracts,
        input_contract=input_contract,
        config=config,
        step=2,
        target_steps=2,
        exposure={"beat2": 30, "hanyang": 2},
        smoke_test=True,
    )
    assert payload["input_contract_sha256"] == "5" * 64
    assert payload["config_sha256"] == "6" * 64
    assert payload["approval_receipt_sha256"] is None
    assert set(payload["sources"]) == {"beat2", "hanyang"}
    assert (
        payload["sources"]["hanyang"]["experimental_pool_manifest_sha256"]
        == config["expected_hanyang_pool_manifest_sha256"]
    )
    assert (
        payload["condition_caches"]["frozen_qwen_cache_sha256"]
        == config["expected_frozen_qwen_cache_sha256"]
    )
    assert payload["lineage"]["admission_contract"]["sha256"] == "2" * 64
