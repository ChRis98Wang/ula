from collections import defaultdict
import copy
import hashlib
import json
import random

import numpy as np
import pytest
import torch

from tools.train_beat2_experimental_metadata_posttrain_ab import (
    BRIDGE_CACHE_ARTIFACT_KIND,
    DEFAULT_CONFIG,
    KIMODO_CONDITION_DIM,
    KIMODO_V2_CONDITION_DIM,
    MOTION_LATENT_WEIGHT_NAME,
    PLAN_WEIGHT_NAME,
    _batch_tensors_for_config,
    _configure_condition_path_optimizer,
    _load_resume_state,
    _lr_scale,
    _preservation_audit,
    _restore_ema_preserved_state,
    _save_resume_state,
    _seed_everything,
    _state_dict_sha256,
    _training_config_for_core,
    _validate_paired_128d_caches,
    _write_bridge_cache,
    validate_config,
)
from upper_body_skeleton.ula_training import UlaMMDiTV3AdaLNModel
from upper_body_skeleton.ula_v2_18d_head import validate_checkpoint_contract
from upper_body_skeleton.ula_v2_18d_posttrain import (
    ModelEMA,
    NativeLengthBucketSampler,
    _sampler_for_config,
    masked_18d_objective,
)


def test_config_fails_closed_on_scope_policy_and_external_path():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["training_policy"] = "full_network"
    with pytest.raises(ValueError, match="training_policy"):
        validate_config(config)

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["manifest_path"] = "/datasets/kimodo_forbidden/train.jsonl"
    with pytest.raises(ValueError, match="forbidden external-data token"):
        validate_config(config)

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["training"]["loss"]["body"] = 0.1
    with pytest.raises(ValueError, match="body"):
        validate_config(config)


def test_paired_128d_identity_must_match_except_conditions():
    common = {
        "clip_ids": np.asarray(["a", "b"]),
        "task_ids": np.asarray(["ta", "tb"]),
        "prompts": np.asarray(["pa", "pb"]),
        "fixed_split_assignments": np.asarray(["train", "test"]),
        "speaker_keys": np.asarray(["s1", "s2"]),
        "semantic_group_indices": np.asarray([0, 1], dtype=np.int64),
        "trajectory_sha256": np.asarray(["1" * 64, "2" * 64]),
        "conditions": np.eye(2, 128, dtype=np.float32),
    }
    frozen = {name: value.copy() for name, value in common.items()}
    lora = {name: value.copy() for name, value in common.items()}
    lora["conditions"] *= -1
    assert len(_validate_paired_128d_caches(frozen, lora)) == 64

    lora["trajectory_sha256"][1] = "3" * 64
    with pytest.raises(ValueError, match="trajectory_sha256"):
        _validate_paired_128d_caches(frozen, lora)


def test_bridge_cache_preserves_zero_and_style_slices(tmp_path):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["output_dir"] = str(tmp_path)
    count = 2
    style_conditions = np.zeros(
        (count, KIMODO_V2_CONDITION_DIM), dtype=np.float32
    )
    style_controls = np.asarray(
        [[-0.5, 0.0, 0.5], [0.25, -0.25, 0.75]], dtype=np.float32
    )
    style_conditions[:, 133:136] = style_controls
    latent = np.zeros((count, 128), dtype=np.float32)
    latent[0, 0] = 1.0
    latent[1, 1] = 1.0
    common = {
        "clip_ids": np.asarray(["a", "b"]),
        "prompts": np.asarray(["pa", "pb"]),
        "fixed_split_assignments": np.asarray(["train", "test"]),
        "speaker_keys": np.asarray(["s1", "s2"]),
        "semantic_group_indices": np.asarray([0, 1], dtype=np.int64),
        "trajectory_sha256": np.asarray(["1" * 64, "2" * 64]),
        "style_conditions": style_conditions,
        "style_features": np.zeros((count, 3), dtype=np.float32),
        "style_controls": style_controls,
    }
    source = {"conditions": latent}
    source_metadata = {
        "path": str(tmp_path / "source.npz"),
        "cache_sha256": "a" * 64,
        "adapter_receipt": {
            "path": str(tmp_path / "adapter.pt"),
            "sha256": "b" * 64,
        },
        "qwen": {"model_name": "official-test"},
    }
    foundation = {
        "path": str(tmp_path / "foundation.pt"),
        "sha256": "c" * 64,
        "data_isolation_contract_sha256": "d" * 64,
        "split_contract_sha256": "e" * 64,
        "style_contract_sha256": "f" * 64,
    }
    style_metadata = {
        "path": str(tmp_path / "style.npz"),
        "cache_sha256": "0" * 64,
    }
    receipt = _write_bridge_cache(
        variant="frozen_base",
        source_128d=source,
        source_metadata=source_metadata,
        common=common,
        identity_arrays_sha256="9" * 64,
        foundation_receipt=foundation,
        style_metadata=style_metadata,
        config=config,
    )
    assert receipt["artifact_kind"] == BRIDGE_CACHE_ARTIFACT_KIND
    with np.load(receipt["path"], allow_pickle=False) as payload:
        conditions = payload["conditions"]
        assert np.count_nonzero(conditions[:, :133]) == 0
        assert np.array_equal(conditions[:, 133:136], style_controls)
        assert np.array_equal(conditions[:, 136:264], latent)


def test_condition_path_optimizer_preserves_zero_latent_behavior_exactly():
    torch.manual_seed(4)
    model = UlaMMDiTV3AdaLNModel(
        action_dim=18,
        condition_dim=264,
        hidden_dim=32,
        layers=1,
        semantic_tokens=7,
    )
    config = validate_config(copy.deepcopy(DEFAULT_CONFIG))
    foundation_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    optimizer, _, receipt = _configure_condition_path_optimizer(
        model, config, device=torch.device("cpu")
    )
    assert receipt["full_network"] is False
    assert receipt["effective_trainable_parameter_count"] == (
        model.motion_latent_condition[0].weight.numel()
        + model.plan[0].weight.shape[0] * 128
    )
    ema = ModelEMA(model, 0.9)
    x = torch.randn(2, 6, 18)
    t = torch.tensor([0.25, 0.75])
    condition = torch.zeros(2, 264)
    condition[:, 133:136] = torch.randn(2, 3)
    condition[:, 136:] = torch.randn(2, 128)
    zero_condition = condition.clone()
    zero_condition[:, 136:] = 0
    with torch.no_grad():
        baseline = model(x, t, zero_condition).clone()
    loss = model(x, t, condition).square().mean()
    loss = loss + model.plan_condition(condition)["duration_sec"].mean()
    loss.backward()
    optimizer.step()
    ema.update(model)
    _restore_ema_preserved_state(ema, foundation_state)
    final_state = {
        name: value.detach().clone() for name, value in ema.shadow.items()
    }
    model.load_state_dict(final_state, strict=True)
    with torch.no_grad():
        final = model(x, t, zero_condition)
    audit = _preservation_audit(
        final_state=final_state,
        foundation_state=foundation_state,
        baseline_zero_output=baseline,
        final_zero_output=final,
        frame_valid=torch.ones(2, 6, dtype=torch.bool),
    )
    assert audit["zero_latent_exact_equivalence_passed"] is True
    assert audit["zero_latent_max_abs_error"] == 0.0
    assert audit["frozen_parameter_max_abs_error"] == 0.0
    assert audit["nonlatent_plan_columns_max_abs_error"] == 0.0
    assert not torch.equal(
        final_state[MOTION_LATENT_WEIGHT_NAME],
        foundation_state[MOTION_LATENT_WEIGHT_NAME],
    )
    assert not torch.equal(
        final_state[PLAN_WEIGHT_NAME][:, KIMODO_CONDITION_DIM:],
        foundation_state[PLAN_WEIGHT_NAME][:, KIMODO_CONDITION_DIM:],
    )


def test_experimental_artifact_cannot_masquerade_as_formal_checkpoint():
    with pytest.raises(ValueError, match="artifact_kind"):
        validate_checkpoint_contract(
            {
                "artifact_kind": (
                    "beat2_experimental_metadata_conditioned_posttrain_v1"
                )
            }
        )


def test_condition_path_exact_resume_matches_uninterrupted_training(tmp_path):
    """Exercise the real branch optimizer/sampler/loss and resume payload."""

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["device"] = "cpu"
    config["training"].update(
        {
            "steps": 3,
            "batch_size": 2,
            "lr": 2e-5,
            "minimum_lr_ratio": 0.1,
            "warmup_steps": 1,
            "weight_decay": 1e-4,
            "adam_eps": 1e-6,
            "max_grad_norm": 1.0,
            "ema_decay": 0.9,
        }
    )
    config["training"]["batching"].update(
        {
            "length_buckets": [8],
            "max_motion_tokens_per_microbatch": 16,
            "max_attention_elements_per_microbatch": 450,
            "target_effective_batch_size": 2,
        }
    )
    config = validate_config(config)
    pair = {
        "sha256": "1" * 64,
        "config_contract_sha256": "2" * 64,
        "foundation": {"sha256": "3" * 64},
        "branches": {
            "frozen_base": {"bridge_cache": {"sha256": "4" * 64}}
        },
    }
    action_stats = {
        "mean": torch.zeros(18),
        "std": torch.ones(18),
    }
    episodes = []
    phase = np.linspace(0.0, 2.0 * np.pi, 8, dtype=np.float32)[:, None]
    for index in range(4):
        actions = np.sin(
            phase + 0.13 * index
        ) * np.linspace(0.01, 0.08, 18, dtype=np.float32)[None, :]
        condition = np.zeros(KIMODO_V2_CONDITION_DIM, dtype=np.float32)
        condition[133:136] = np.asarray(
            [0.1 * index, -0.05 * index, 0.025 * index], dtype=np.float32
        )
        condition[136:] = np.linspace(
            -0.2 + 0.01 * index,
            0.2 + 0.01 * index,
            128,
            dtype=np.float32,
        )
        episodes.append(
            {
                "clip_id": f"synthetic_{index}",
                "actions": actions.astype(np.float32),
                "condition": condition,
                "action_dim_mask": np.ones(18, dtype=np.bool_),
                "fps": 30.0,
                "duration_sec": 7 / 30.0,
                "domain": "beat2",
                "dataset_source": "beat2_synthetic_resume_fixture",
                "speaker_key": f"speaker_{index % 2}",
                "source_group_key": f"source_{index}",
            }
        )

    _seed_everything(719)
    foundation = UlaMMDiTV3AdaLNModel(
        action_dim=18,
        condition_dim=KIMODO_V2_CONDITION_DIM,
        hidden_dim=32,
        layers=1,
        semantic_tokens=7,
    )
    foundation_state = {
        name: value.detach().cpu().clone()
        for name, value in foundation.state_dict().items()
    }

    def new_training_state():
        model = UlaMMDiTV3AdaLNModel(
            action_dim=18,
            condition_dim=KIMODO_V2_CONDITION_DIM,
            hidden_dim=32,
            layers=1,
            semantic_tokens=7,
        )
        model.load_state_dict(foundation_state, strict=True)
        model.train()
        optimizer, _, _ = _configure_condition_path_optimizer(
            model, config, device=torch.device("cpu")
        )
        ema = ModelEMA(model, float(config["training"]["ema_decay"]))
        sampler = _sampler_for_config(
            episodes,
            _training_config_for_core(config),
            seed=int(config["seed"]) + 17,
        )
        assert isinstance(sampler, NativeLengthBucketSampler)
        return model, optimizer, ema, sampler

    def train_step(model, optimizer, ema, sampler, step):
        training = config["training"]
        scale = _lr_scale(
            step,
            total_steps=int(training["steps"]),
            warmup_steps=int(training["warmup_steps"]),
            minimum_ratio=float(training["minimum_lr_ratio"]),
        )
        current_lr = float(training["lr"]) * scale
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.zero_grad(set_to_none=True)
        remaining = int(
            training["batching"]["target_effective_batch_size"]
        )
        accumulated = defaultdict(float)
        sampled_clip_ids = []
        plans = []
        while remaining > 0:
            microbatch, plan = sampler.sample_microbatch(
                remaining_effective_batch=remaining,
                semantic_tokens=int(model.semantic_tokens),
                max_batch_size=int(training["batch_size"]),
                batching=training["batching"],
            )
            actions, conditions, masks, durations, frame_valid = (
                _batch_tensors_for_config(
                    microbatch,
                    frame_count=int(plan["bucket_frames"]),
                    action_stats=action_stats,
                    device=torch.device("cpu"),
                    batching=training["batching"],
                )
            )
            losses = masked_18d_objective(
                model,
                actions,
                conditions,
                masks,
                durations,
                loss_weights=training["loss"],
                teacher_model=None,
                frame_valid_mask=frame_valid,
            )
            sample_weight = len(microbatch) / float(
                training["batching"]["target_effective_batch_size"]
            )
            (losses["total"] * sample_weight).backward()
            for name, value in losses.items():
                accumulated[name] += (
                    float(value.detach().cpu()) * sample_weight
                )
            sampled_clip_ids.extend(row["clip_id"] for row in microbatch)
            plans.append(dict(plan))
            remaining -= len(microbatch)
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                float(training["max_grad_norm"]),
            )
        )
        optimizer.step()
        ema.update(model)
        _restore_ema_preserved_state(ema, foundation_state)
        return {
            "step": step,
            "lr": current_lr,
            "grad_norm": grad_norm,
            "train": dict(accumulated),
            "sampled_clip_ids_sha256": hashlib.sha256(
                json.dumps(
                    sampled_clip_ids,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("ascii")
            ).hexdigest(),
            "microbatch_plans": plans,
        }

    def final_receipt(model, optimizer, ema, sampler, events):
        trainable_state = {
            MOTION_LATENT_WEIGHT_NAME: model.state_dict()[
                MOTION_LATENT_WEIGHT_NAME
            ],
            PLAN_WEIGHT_NAME: model.state_dict()[PLAN_WEIGHT_NAME][
                :, KIMODO_CONDITION_DIM:
            ],
        }
        return {
            "raw_model_sha256": _state_dict_sha256(model.state_dict()),
            "ema_sha256": _state_dict_sha256(ema.shadow),
            "trainable_state_sha256": _state_dict_sha256(trainable_state),
            "optimizer_state": optimizer.state_dict(),
            "sampler_state": sampler.state_dict(),
            "torch_rng_state": torch.get_rng_state().clone(),
            "python_rng_state": random.getstate(),
            "events": events,
        }

    _seed_everything(int(config["seed"]))
    continuous_model, continuous_optimizer, continuous_ema, continuous_sampler = (
        new_training_state()
    )
    continuous_events = [
        train_step(
            continuous_model,
            continuous_optimizer,
            continuous_ema,
            continuous_sampler,
            step,
        )
        for step in range(1, 4)
    ]
    continuous = final_receipt(
        continuous_model,
        continuous_optimizer,
        continuous_ema,
        continuous_sampler,
        continuous_events,
    )

    _seed_everything(int(config["seed"]))
    resumed_model, resumed_optimizer, resumed_ema, resumed_sampler = (
        new_training_state()
    )
    resumed_events = [
        train_step(
            resumed_model,
            resumed_optimizer,
            resumed_ema,
            resumed_sampler,
            1,
        )
    ]
    state_path = tmp_path / "isolated_exact_resume" / "last_state.pt"
    _save_resume_state(
        state_path,
        variant="frozen_base",
        pair=pair,
        config=config,
        step=1,
        model=resumed_model,
        ema=resumed_ema,
        optimizer=resumed_optimizer,
        sampler=resumed_sampler,
        initial_validation={"total": 1.25},
        initial_condition_response={"aligned_vs_zero_prediction_rms": 0.5},
    )

    # Simulate a fresh process whose construction and unrelated work consumed RNG.
    _seed_everything(999_999)
    resumed_model, resumed_optimizer, resumed_ema, resumed_sampler = (
        new_training_state()
    )
    random.random()
    torch.rand(11)
    resumed_step, initial_validation, initial_condition_response = (
        _load_resume_state(
            state_path,
            variant="frozen_base",
            pair=pair,
            config=config,
            model=resumed_model,
            ema=resumed_ema,
            optimizer=resumed_optimizer,
            sampler=resumed_sampler,
            device=torch.device("cpu"),
        )
    )
    assert resumed_step == 1
    assert initial_validation == {"total": 1.25}
    assert initial_condition_response == {
        "aligned_vs_zero_prediction_rms": 0.5
    }
    for step in range(resumed_step + 1, 4):
        resumed_events.append(
            train_step(
                resumed_model,
                resumed_optimizer,
                resumed_ema,
                resumed_sampler,
                step,
            )
        )
    resumed = final_receipt(
        resumed_model,
        resumed_optimizer,
        resumed_ema,
        resumed_sampler,
        resumed_events,
    )

    def assert_nested_exact(left, right):
        if isinstance(left, torch.Tensor):
            assert isinstance(right, torch.Tensor)
            assert torch.equal(left, right)
        elif isinstance(left, dict):
            assert isinstance(right, dict)
            assert left.keys() == right.keys()
            for key in left:
                assert_nested_exact(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            assert isinstance(right, type(left))
            assert len(left) == len(right)
            for left_item, right_item in zip(left, right, strict=True):
                assert_nested_exact(left_item, right_item)
        else:
            assert left == right

    assert continuous["raw_model_sha256"] == resumed["raw_model_sha256"]
    assert continuous["ema_sha256"] == resumed["ema_sha256"]
    assert (
        continuous["trainable_state_sha256"]
        == resumed["trainable_state_sha256"]
    )
    assert_nested_exact(
        continuous["optimizer_state"], resumed["optimizer_state"]
    )
    assert_nested_exact(continuous["sampler_state"], resumed["sampler_state"])
    assert torch.equal(
        continuous["torch_rng_state"], resumed["torch_rng_state"]
    )
    assert continuous["python_rng_state"] == resumed["python_rng_state"]
    assert continuous["events"] == resumed["events"]
