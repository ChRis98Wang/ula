import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tools import train_ula_v2_18d_staged as staged
from upper_body_skeleton.retarget_v2_18d import CONTRACT_VERSION, JOINT_ORDER_18D
from upper_body_skeleton.ula_training import (
    KIMODO_V2_CONDITION_DIM,
    ULA_MMDIT_V2_ARCHITECTURE,
    create_ula_model,
)
from upper_body_skeleton.ula_v2_18d_head import ARTIFACT_KIND
from upper_body_skeleton.ula_v2_18d_posttrain import (
    ACTION_DIM,
    DEFAULT_CONFIG,
    LEGACY_ACTION_DIM,
    SourceSpeakerActivityBalancedSampler,
    masked_18d_objective,
    resolve_posttrain_config,
    train_18d_posttrain,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = nn.Linear(ACTION_DIM, ACTION_DIM)

    def forward(self, x_t, _t, _condition):
        return self.output(x_t)


def _legacy_masked_mean(values, mask):
    expanded = torch.broadcast_to(mask, values.shape).to(values.dtype)
    return (values * expanded).sum() / expanded.sum()


def _legacy_objective(model, actions, conditions, mask, durations, *, seed):
    generator = torch.Generator().manual_seed(seed)
    observed = mask[:, None, :]
    noise = torch.randn(actions.shape, generator=generator)
    t = torch.rand(actions.shape[0], generator=generator)
    noise = noise * observed
    x_t = ((1.0 - t[:, None, None]) * noise + t[:, None, None] * actions) * observed
    target = (actions - noise) * observed
    predicted = model(x_t, t, conditions)
    reconstructed = x_t + (1.0 - t[:, None, None]) * predicted
    frame_count = actions.shape[1]
    dt = (durations / float(frame_count - 1)).clamp_min(1e-4)[:, None, None]
    reconstructed_velocity = (reconstructed[:, 1:] - reconstructed[:, :-1]) / dt
    target_velocity = (actions[:, 1:] - actions[:, :-1]) / dt
    reconstructed_acceleration = (
        reconstructed_velocity[:, 1:] - reconstructed_velocity[:, :-1]
    ) / dt
    target_acceleration = (target_velocity[:, 1:] - target_velocity[:, :-1]) / dt
    losses = {
        "flow": _legacy_masked_mean((predicted - target).square(), observed),
        "position": _legacy_masked_mean(
            F.smooth_l1_loss(reconstructed, actions, reduction="none"), observed
        ),
        "body": predicted.new_zeros(()),
        "velocity": _legacy_masked_mean(
            F.smooth_l1_loss(
                reconstructed_velocity, target_velocity, reduction="none"
            ),
            observed,
        ),
        "acceleration": _legacy_masked_mean(
            F.smooth_l1_loss(
                reconstructed_acceleration, target_acceleration, reduction="none"
            ),
            observed,
        ),
    }
    losses["total"] = sum(
        DEFAULT_CONFIG["loss"][name] * value for name, value in losses.items()
    )
    return losses


def test_old_config_keeps_legacy_loss_keys_and_bitwise_objective():
    resolved = resolve_posttrain_config({})
    assert "training_policy" not in resolved
    assert "sampler" not in resolved
    assert resolved["loss"] == DEFAULT_CONFIG["loss"]

    torch.manual_seed(31)
    model = TinyModel()
    actions = torch.randn(2, 6, ACTION_DIM)
    conditions = torch.randn(2, KIMODO_V2_CONDITION_DIM)
    mask = torch.ones(2, ACTION_DIM, dtype=torch.bool)
    durations = torch.tensor([0.5, 0.8])
    expected = _legacy_objective(
        model, actions, conditions, mask, durations, seed=47
    )
    actual = masked_18d_objective(
        model,
        actions,
        conditions,
        mask,
        durations,
        generator=torch.Generator().manual_seed(47),
    )
    assert actual.keys() == expected.keys()
    for name in expected:
        assert torch.equal(actual[name], expected[name]), name


def test_head_derivative_losses_use_duration_and_mask_unobserved_replay_head():
    torch.manual_seed(5)
    model = TinyModel()
    actions = torch.randn(1, 6, ACTION_DIM)
    conditions = torch.zeros(1, KIMODO_V2_CONDITION_DIM)
    observed = torch.ones(1, ACTION_DIM, dtype=torch.bool)
    weights = {
        "flow": 0.0,
        "position": 0.0,
        "body": 0.0,
        "velocity": 0.0,
        "acceleration": 0.0,
        "head_velocity": 1.0,
        "head_acceleration": 1.0,
        "head_jerk": 1.0,
    }
    slow = masked_18d_objective(
        model,
        actions,
        conditions,
        observed,
        torch.tensor([2.0]),
        loss_weights=weights,
        generator=torch.Generator().manual_seed(9),
    )
    fast = masked_18d_objective(
        model,
        actions,
        conditions,
        observed,
        torch.tensor([1.0]),
        loss_weights=weights,
        generator=torch.Generator().manual_seed(9),
    )
    assert fast["head_velocity"] > slow["head_velocity"]
    assert fast["head_acceleration"] > slow["head_acceleration"]
    assert fast["head_jerk"] > slow["head_jerk"]

    replay_mask = observed.clone()
    replay_mask[:, LEGACY_ACTION_DIM:] = False
    changed = actions.clone()
    changed[..., LEGACY_ACTION_DIM:] = 10000.0
    first = masked_18d_objective(
        model,
        actions,
        conditions,
        replay_mask,
        torch.tensor([1.0]),
        loss_weights=weights,
        generator=torch.Generator().manual_seed(13),
    )
    second = masked_18d_objective(
        model,
        changed,
        conditions,
        replay_mask,
        torch.tensor([1.0]),
        loss_weights=weights,
        generator=torch.Generator().manual_seed(13),
    )
    for name in ("head_velocity", "head_acceleration", "head_jerk", "total"):
        assert torch.equal(first[name], second[name])
        assert first[name].item() == 0.0

    short = masked_18d_objective(
        model,
        actions[:, :3],
        conditions,
        observed,
        torch.tensor([1.0]),
        loss_weights=weights,
        generator=torch.Generator().manual_seed(17),
    )
    assert short["head_jerk"].item() == 0.0
    assert torch.isfinite(short["total"])


def _activity_episode(dataset, speaker, activity, source_group, clip_id):
    frames = 8
    duration = (frames - 1) / 30.0
    actions = np.zeros((frames, ACTION_DIM), dtype=np.float32)
    actions[:, -1] = np.linspace(
        0.0, activity * math.sqrt(3.0) * duration, frames, dtype=np.float32
    )
    return {
        "clip_id": clip_id,
        "actions": actions,
        "action_dim_mask": np.ones(ACTION_DIM, dtype=np.bool_),
        "condition": np.zeros(KIMODO_V2_CONDITION_DIM, dtype=np.float32),
        "duration_sec": duration,
        "fps": 30.0,
        "domain": "beat2",
        "dataset_source": dataset,
        "speaker_key": speaker,
        "source_group_key": source_group,
    }


def test_source_activity_sampler_balances_sources_and_resumes_exactly():
    episodes = []
    for dataset in ("beat2", "supplemental_interaction"):
        for activity_index, activity in enumerate((0.05, 0.14, 0.18, 0.30)):
            for window in range(2):
                episodes.append(
                    _activity_episode(
                        dataset,
                        f"{dataset}_speaker",
                        activity,
                        f"{dataset}_source_{activity_index}",
                        f"{dataset}_{activity_index}_{window}",
                    )
                )
    sampler = SourceSpeakerActivityBalancedSampler(
        episodes, activity_bin_edges_rad_s=[0.12, 0.16, 0.21], seed=23
    )
    batch = sampler.sample(16)
    assert sum(row["dataset_source"] == "beat2" for row in batch) == 8
    assert sum(row["dataset_source"] == "supplemental_interaction" for row in batch) == 8
    assert len({row["source_group_key"] for row in batch}) == 8

    state = sampler.state_dict()
    expected = [row["clip_id"] for row in sampler.sample(19)]
    restored = SourceSpeakerActivityBalancedSampler(
        list(reversed(episodes)),
        activity_bin_edges_rad_s=[0.12, 0.16, 0.21],
        seed=999,
    )
    restored.load_state_dict(state)
    assert [row["clip_id"] for row in restored.sample(19)] == expected

    incompatible = SourceSpeakerActivityBalancedSampler(
        episodes[:-1], activity_bin_edges_rad_s=[0.12, 0.16, 0.21], seed=23
    )
    with pytest.raises(ValueError, match="structure changed"):
        incompatible.load_state_dict(state)


def _checkpoint(path):
    torch.manual_seed(13)
    model = create_ula_model(
        ULA_MMDIT_V2_ARCHITECTURE,
        action_dim=ACTION_DIM,
        condition_dim=KIMODO_V2_CONDITION_DIM,
        hidden_dim=16,
        layers=1,
        semantic_tokens=6,
    )
    payload = {
        "schema_version": 2,
        "artifact_kind": ARTIFACT_KIND,
        "architecture": ULA_MMDIT_V2_ARCHITECTURE,
        "condition_dim": KIMODO_V2_CONDITION_DIM,
        "action_dim": ACTION_DIM,
        "joint_order": list(JOINT_ORDER_18D),
        "model_state_dict": model.state_dict(),
        "action_stats": {
            "mean": torch.zeros(ACTION_DIM),
            "std": torch.ones(ACTION_DIM),
        },
        "action_contract": {
            "version": CONTRACT_VERSION,
            "joint_order": list(JOINT_ORDER_18D),
            "legacy_prefix_dim": LEGACY_ACTION_DIM,
        },
        "config": {"hidden_dim": 16, "layers": 1, "semantic_tokens": 6},
        "global_step": 100,
    }
    torch.save(payload, path)
    return payload


def _head_train_episodes():
    episodes = []
    phase = np.linspace(0.0, 2.0 * np.pi, 7, dtype=np.float32)[:, None]
    for speaker_index in range(5):
        for source_index in range(2):
            actions = np.sin(phase + 0.2 * source_index) * np.linspace(
                0.01, 0.08, ACTION_DIM, dtype=np.float32
            )[None, :]
            episodes.append(
                {
                    "clip_id": f"speaker_{speaker_index}_{source_index}",
                    "actions": actions,
                    "condition": np.full(
                        KIMODO_V2_CONDITION_DIM,
                        0.01 * (speaker_index + 1),
                        dtype=np.float32,
                    ),
                    "fps": 30.0,
                    "duration_sec": 6 / 30.0,
                    "speaker_key": f"speaker_{speaker_index}",
                    "source_group_key": f"speaker_{speaker_index}:source_{source_index}",
                    "dataset_source": "beat2",
                    "emotion_id": None,
                    "emotion_supervision_mask": False,
                    "accepted_for_training": False,
                    "eligibility_mode": "unsafe_allow_unreviewed",
                }
            )
    return episodes


def test_head_projection_training_and_resume_keep_frozen_weights_bitwise(tmp_path):
    initial_path = tmp_path / "initial.pt"
    initial = _checkpoint(initial_path)
    output = tmp_path / "head"
    config = {
        "training_policy": "head_projection_only",
        "steps": 2,
        "batch_size": 4,
        "validation_batch_size": 2,
        "phase_frame_choices": [4],
        "lr": 3e-4,
        "minimum_lr_ratio": 0.5,
        "warmup_steps": 1,
        "weight_decay": 0.01,
        "validation_interval": 1,
        "checkpoint_interval": 1,
        "log_interval": 1,
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 0.0,
        "ema_decay": 0.9,
        "device": "cpu",
        "seed": 29,
        "allow_unsafe_training_data": True,
        "sampler": {
            "mode": "source_speaker_activity",
            "activity_bin_edges_rad_s": [0.12, 0.16, 0.21],
        },
        "loss": {
            "flow": 0.0,
            "position": 0.0,
            "body": 1.0,
            "velocity": 0.0,
            "acceleration": 0.0,
            "head_flow": 1.0,
            "head_position": 0.25,
            "head_velocity": 0.001,
            "head_acceleration": 0.00001,
            "head_jerk": 0.000001,
        },
    }
    summary = train_18d_posttrain(
        initial_checkpoint_path=initial_path,
        beat_episodes=_head_train_episodes(),
        output_dir=output,
        config=config,
    )
    assert summary["training_policy"] == "head_projection_only"
    assert summary["frozen_weight_max_abs_error"] == 0.0
    last = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
    trained = last["training_state"]["raw_model_state_dict"]
    for name, before in initial["model_state_dict"].items():
        after = trained[name]
        if name == "input.weight":
            assert torch.equal(after[:, :LEGACY_ACTION_DIM], before[:, :LEGACY_ACTION_DIM])
        elif name in ("output.weight", "output.bias"):
            assert torch.equal(after[:LEGACY_ACTION_DIM], before[:LEGACY_ACTION_DIM])
        else:
            assert torch.equal(after, before), name
    assert not torch.equal(
        trained["output.weight"][LEGACY_ACTION_DIM:],
        initial["model_state_dict"]["output.weight"][LEGACY_ACTION_DIM:],
    )

    resumed = train_18d_posttrain(
        initial_checkpoint_path=initial_path,
        beat_episodes=_head_train_episodes(),
        output_dir=output,
        config=config | {"steps": 3, "resume_from": str(output / "last.pt")},
    )
    assert resumed["completed_steps"] == 3
    assert resumed["frozen_weight_max_abs_error"] == 0.0


def test_staged_config_requires_audio_disabled_and_matching_split(tmp_path):
    source = json.loads(
        Path("configs/beat2_18d_from_15d_staged_v1.json").read_text(encoding="utf-8")
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(source), encoding="utf-8")
    loaded = staged.read_config(config_path)
    assert loaded["audio_policy"] == "disabled_not_loaded"
    assert loaded["training_scope"] == "head_mechanism_experiment_only"
    assert loaded["formal_training_enabled"] is False
    assert loaded["head_pretrain"]["training_policy"] == "head_projection_only"
    assert loaded["joint_train"]["training_policy"] == "full_network"
    assert loaded["joint_train"]["require_disjoint_replay_evaluation"] is True

    source["audio_policy"] = "enabled"
    config_path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="audio_policy=disabled_not_loaded"):
        staged.read_config(config_path)

    source = json.loads(
        Path("configs/beat2_18d_from_15d_staged_v1.json").read_text(encoding="utf-8")
    )
    source.update(
        training_scope="formal_variable_length_semantic_units",
        formal_training_enabled=True,
    )
    source["motion_sources"][0]["temporal_unit"] = "full_semantic_unit"
    config_path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="formal variable-length training is not implemented"):
        staged.read_config(config_path)


def test_temporal_quarantine_excludes_only_failed_train_and_keeps_eval_challenge():
    config = json.loads(
        Path("configs/beat2_18d_from_15d_staged_v1.json").read_text(encoding="utf-8")
    )
    episodes = _head_train_episodes()
    for episode in episodes:
        episode["actions"] = np.zeros((9, ACTION_DIM), dtype=np.float32)
        episode["duration_sec"] = 8 / 30.0
        episode["trajectory_sha256"] = episode["clip_id"]
    pre_splits, _ = staged.initial_strict_split(config, episodes)
    train_id = pre_splits["train"][0]["clip_id"]
    validation_id = pre_splits["validation"][0]["clip_id"]
    by_id = {episode["clip_id"]: episode for episode in episodes}
    for clip_id in (train_id, validation_id):
        by_id[clip_id]["actions"][4, -1] = 1.0

    filtered, summary, exclusions, challenges, pre_contract = staged.temporal_quarantine(
        config, episodes
    )
    filtered_ids = {episode["clip_id"] for episode in filtered}
    assert train_id not in filtered_ids
    assert validation_id in filtered_ids
    assert {record["clip_id"] for record in exclusions} == {train_id}
    assert validation_id in {record["clip_id"] for record in challenges}
    assert summary["counts"]["train"]["excluded_from_training"] == 1
    assert summary["counts"]["validation"]["retained_as_challenge"] == 1
    assert summary["pre_quarantine_split_contract_sha256"] == pre_contract["sha256"]
    assert summary["evaluated_records_sha256"]
    assert summary["input_episode_content_sha256"]
    assert summary["filtered_episode_content_sha256"]
    train_exclusion = exclusions[0]
    assert train_exclusion["metrics"]["peak_abs_acceleration_by_axis_rad_s2"][:2] == [
        0.0,
        0.0,
    ]
    assert train_exclusion["metrics"]["peak_abs_acceleration_max_axis_rad_s2"] > 20.0
    assert (
        train_exclusion["metrics"]["moving_average_residual_rms_max_axis_degrees"]
        > 0.3
    )

    post_splits, post_contract = staged.initial_strict_split(config, filtered)
    assert train_id not in {
        episode["clip_id"] for episode in post_splits["train"]
    }
    assert validation_id in {
        episode["clip_id"] for episode in post_splits["validation"]
    }
    assert post_contract["assignment_policy"] == "fixed_pre_quarantine_assignment"
    assert post_contract["speaker_to_split"] == pre_contract["speaker_to_split"]
    assert all("fixed_split_assignment" in episode for episode in filtered)
    filtered_by_id = {episode["clip_id"]: episode for episode in filtered}
    assert filtered_by_id[validation_id]["temporal_quarantine_challenge"] is True
    assert filtered_by_id[validation_id]["temporal_quarantine_failed_gates"]

    changed = [dict(episode) for episode in filtered]
    changed[0] = dict(changed[0], actions=changed[0]["actions"].copy())
    changed[0]["actions"][0, 0] += 0.001
    assert staged.episodes_content_contract(changed)["sha256"] != (
        staged.episodes_content_contract(filtered)["sha256"]
    )

    wrong_fps = [dict(episode) for episode in episodes]
    wrong_fps[0]["fps"] = 25.0
    wrong_fps[0]["duration_sec"] = 8 / 25.0
    with pytest.raises(ValueError, match="requires 30 Hz input"):
        staged.temporal_quarantine(config, wrong_fps)


def test_completed_training_without_matching_receipt_is_never_reauthorized(tmp_path):
    config = json.loads(
        Path("configs/beat2_18d_from_15d_staged_v1.json").read_text(encoding="utf-8")
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    config["motion_sources"] = [
        {
            "dataset_source": "fixture",
            "temporal_unit": "fixed_window_experimental",
            "manifest": str(manifest),
        }
    ]
    config["output_root"] = str(tmp_path / "root")
    paths = staged.stage_paths(config)
    paths["temporal_summary"].parent.mkdir(parents=True, exist_ok=True)
    paths["temporal_summary"].write_text("{}\n", encoding="utf-8")
    initial = tmp_path / "initial.pt"
    torch.save({"fixture": True}, initial)
    condition_cache = tmp_path / "conditions.npz"
    condition_cache.write_bytes(b"cache")
    condition_cache.with_name("conditions.npz.json").write_text(
        "{}\n", encoding="utf-8"
    )
    output = tmp_path / "completed"
    output.mkdir()
    training = dict(config["head_pretrain"])
    training["steps"] = 1
    expected = resolve_posttrain_config(
        training
        | {
            "allow_unsafe_training_data": True,
            "training_scope": "head_mechanism_experiment_only",
            "formal_training_enabled": False,
            "temporal_unit_policy": "fixed_window_experimental",
        }
    )
    torch.save(
        {"posttrain_config": expected}, output / "ula_fm_checkpoint.pt"
    )
    (output / "training_summary.json").write_text(
        json.dumps(
            {"target_steps": 1, "completed_steps": 1, "stopped_early": False}
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="lacks its matching receipt"):
        staged.run_training(
            config,
            paths,
            _head_train_episodes(),
            stage="head-pretrain",
            initial_checkpoint=initial,
            condition_cache=condition_cache,
            output_dir=output,
            training_config=training,
        )


def test_pipeline_status_is_derived_from_all_valid_receipts(tmp_path):
    config = json.loads(
        Path("configs/beat2_18d_from_15d_staged_v1.json").read_text(encoding="utf-8")
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    config["motion_sources"] = [
        {
            "dataset_source": "fixture",
            "temporal_unit": "fixed_window_experimental",
            "manifest": str(manifest),
        }
    ]
    config["output_root"] = str(tmp_path / "root")
    paths = staged.stage_paths(config)
    paths["temporal_summary"].parent.mkdir(parents=True, exist_ok=True)
    paths["temporal_summary"].write_text("{}\n", encoding="utf-8")
    migrate_output = tmp_path / "migrate.bin"
    migrate_output.write_bytes(b"migrate")
    joint_output = paths["joint_run"] / "ula_fm_checkpoint.pt"
    joint_output.parent.mkdir(parents=True, exist_ok=True)
    joint_output.write_bytes(b"joint")
    staged._write_receipt(paths, "migrate", {"stage": "migrate"}, [migrate_output])
    staged._write_receipt(paths, "joint", {"stage": "joint"}, [joint_output])

    staged._status(config, paths, completed=["migrate"])

    status = json.loads(paths["status"].read_text(encoding="utf-8"))
    assert status["completed_stages"] == ["migrate", "joint"]
    assert status["final_checkpoint"] == str(joint_output.resolve())
