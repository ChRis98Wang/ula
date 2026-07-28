import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from upper_body_skeleton.retarget_v2_18d import CONTRACT_VERSION, JOINT_ORDER_18D
from upper_body_skeleton.ula_training import (
    KIMODO_V2_CONDITION_DIM,
    ULA_MMDIT_V2_ARCHITECTURE,
    create_ula_model,
)
from upper_body_skeleton.ula_v2_18d_head import (
    ARTIFACT_KIND,
    MOTION_ONLY_RANDOM_INIT_MODE,
    load_contract_checkpoint,
)
from upper_body_skeleton.ula_v2_18d_posttrain import (
    ACTION_DIM,
    LEGACY_ACTION_DIM,
    DomainSpeakerBalancedSampler,
    NativeLengthBucketSampler,
    batch_tensors,
    canonicalize_beat_episodes,
    canonicalize_kimodo_replay,
    deterministic_replay_evaluation_subset,
    evaluate_posttrain,
    masked_18d_objective,
    native_length_bucket,
    native_length_microbatch_capacity,
    posttrain_release_decision,
    replay_regression_guard,
    resolve_posttrain_config,
    strict_group_split,
    train_18d_posttrain,
    validate_strict_group_splits,
)


def _make_checkpoint(path):
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
            "mean": torch.linspace(-0.05, 0.05, ACTION_DIM),
            "std": torch.linspace(0.2, 0.4, ACTION_DIM),
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


def _trajectory(dimensions, *, phase_offset=0.0, frames=7):
    phase = np.linspace(0.0, 2.0 * np.pi, frames, dtype=np.float32)[:, None]
    scale = np.linspace(0.01, 0.08, dimensions, dtype=np.float32)[None, :]
    return np.sin(phase + phase_offset) * scale


def _beat_episodes(*, accepted=False):
    episodes = []
    for speaker_index in range(5):
        for source_index in range(2):
            clip_id = f"speaker_{speaker_index}_source_{source_index}"
            episodes.append(
                {
                    "clip_id": clip_id,
                    "prompt": f"Observable interaction {clip_id}",
                    "actions": _trajectory(
                        ACTION_DIM,
                        phase_offset=0.1 * (speaker_index + source_index),
                    ),
                    "condition": np.full(
                        KIMODO_V2_CONDITION_DIM,
                        0.01 * speaker_index,
                        dtype=np.float32,
                    ),
                    "fps": 30.0,
                    "duration_sec": 6 / 30.0,
                    "speaker_key": f"speaker_{speaker_index}",
                    "source_group_key": f"speaker_{speaker_index}:source_{source_index}",
                    "emotion_id": None,
                    "emotion_review_status": "unresolved",
                    "emotion_supervision_mask": False,
                    "accepted_for_training": bool(accepted),
                    "eligibility_mode": (
                        "adjudicated_train_ready"
                        if accepted
                        else "unsafe_allow_unreviewed"
                    ),
                }
            )
    return episodes


def _replay_episodes():
    return [
        {
            "episode_index": index,
            "actions": _trajectory(LEGACY_ACTION_DIM, phase_offset=0.2 * index),
            "condition": np.full(
                KIMODO_V2_CONDITION_DIM, 0.03 * (index + 1), dtype=np.float32
            ),
            "fps": 30.0,
            "replay_source_validated": True,
        }
        for index in range(2)
    ]


def test_strict_split_keeps_speaker_and_source_groups_disjoint():
    splits, contract = strict_group_split(_beat_episodes(), seed=19)
    validation = validate_strict_group_splits(splits)

    assert contract["counts"] == {name: len(splits[name]) for name in splits}
    assert all(splits[name] for name in ("train", "validation", "test"))
    assert len(validation["speaker_to_split"]) == 5
    for speaker in {row["speaker_key"] for rows in splits.values() for row in rows}:
        occurrences = {
            split_name
            for split_name, rows in splits.items()
            if any(row["speaker_key"] == speaker for row in rows)
        }
        assert len(occurrences) == 1

    leaked = {name: list(rows) for name, rows in splits.items()}
    leaked["validation"].append(dict(leaked["train"][0], clip_id="forced_leak"))
    with pytest.raises(ValueError, match="leakage"):
        validate_strict_group_splits(leaked)


def test_kimodo_replay_head_is_unobserved_and_cannot_change_masked_loss_or_gradient():
    replay = canonicalize_kimodo_replay(_replay_episodes()[:1])[0]
    assert replay["actions"].shape[1] == ACTION_DIM
    assert replay["action_dim_mask"].tolist() == [True] * 15 + [False] * 3
    assert np.count_nonzero(replay["actions"][:, 15:]) == 0
    assert replay["head_target_policy"] == "unobserved_loss_mask_zero"

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.output = nn.Linear(ACTION_DIM, ACTION_DIM)

        def forward(self, x_t, _t, _condition):
            return self.output(x_t)

    torch.manual_seed(5)
    model = TinyModel()
    actions = torch.randn(1, 6, ACTION_DIM)
    changed_head = actions.clone()
    changed_head[..., 15:] = 10_000.0
    condition = torch.zeros(1, KIMODO_V2_CONDITION_DIM)
    mask = torch.tensor([[True] * 15 + [False] * 3], dtype=torch.bool)
    duration = torch.ones(1)
    first = masked_18d_objective(
        model,
        actions,
        condition,
        mask,
        duration,
        generator=torch.Generator().manual_seed(11),
    )
    second = masked_18d_objective(
        model,
        changed_head,
        condition,
        mask,
        duration,
        generator=torch.Generator().manual_seed(11),
    )
    for name in first:
        assert torch.equal(first[name], second[name])
    first["total"].backward()
    assert torch.count_nonzero(model.output.weight.grad[15:]) == 0
    assert torch.count_nonzero(model.output.bias.grad[15:]) == 0


def test_masked_objective_accepts_exact_shared_flow_state_for_condition_pairs():
    class ConditionSensitiveModel(nn.Module):
        def forward(self, x_t, _t, condition):
            return x_t + condition[:, :1, None]

        def forward_masked(self, x_t, t, condition, _frame_valid_mask):
            return self.forward(x_t, t, condition)

    model = ConditionSensitiveModel()
    actions = torch.linspace(-0.4, 0.6, 2 * 6 * ACTION_DIM).reshape(
        2, 6, ACTION_DIM
    )
    conditions = torch.zeros(2, KIMODO_V2_CONDITION_DIM)
    conditions[:, 0] = torch.tensor([0.25, -0.5])
    shuffled = conditions.flip(0)
    dim_mask = torch.ones(2, ACTION_DIM, dtype=torch.bool)
    frame_valid = torch.tensor(
        [
            [True, True, True, True, False, False],
            [True, True, True, True, True, True],
        ]
    )
    durations = torch.tensor([0.1, 5.0 / 30.0])
    noise = torch.linspace(0.8, -0.7, actions.numel()).reshape_as(actions)
    noise[0, 4:] = 10_000.0
    flow_times = torch.tensor([0.2, 0.75])

    first = masked_18d_objective(
        model,
        actions,
        conditions,
        dim_mask,
        durations,
        frame_valid_mask=frame_valid,
        generator=torch.Generator().manual_seed(1),
        noise=noise,
        flow_times=flow_times,
    )
    repeated = masked_18d_objective(
        model,
        actions,
        conditions,
        dim_mask,
        durations,
        frame_valid_mask=frame_valid,
        generator=torch.Generator().manual_seed(999),
        noise=noise,
        flow_times=flow_times,
    )
    counterfactual = masked_18d_objective(
        model,
        actions,
        shuffled,
        dim_mask,
        durations,
        frame_valid_mask=frame_valid,
        noise=noise,
        flow_times=flow_times,
    )

    assert all(torch.equal(first[name], repeated[name]) for name in first)
    assert not torch.equal(first["flow"], counterfactual["flow"])

    with pytest.raises(ValueError, match="explicit noise"):
        masked_18d_objective(
            model,
            actions,
            conditions,
            dim_mask,
            durations,
            noise=noise[:, :-1],
            flow_times=flow_times,
        )
    with pytest.raises(ValueError, match="explicit flow_times"):
        masked_18d_objective(
            model,
            actions,
            conditions,
            dim_mask,
            durations,
            noise=noise,
            flow_times=torch.tensor([0.2, 1.1]),
        )


def test_balanced_sampler_round_robins_domains_and_speakers_and_restores_state():
    beat = canonicalize_beat_episodes(_beat_episodes())
    replay = canonicalize_kimodo_replay(_replay_episodes())
    sampler = DomainSpeakerBalancedSampler(beat + replay, seed=3)
    batch = sampler.sample(20)
    assert sum(row["domain"] == "beat2" for row in batch) == 10
    assert sum(row["domain"] == "kimodo" for row in batch) == 10
    beat_speakers = [row["speaker_key"] for row in batch if row["domain"] == "beat2"]
    assert {speaker: beat_speakers.count(speaker) for speaker in set(beat_speakers)} == {
        f"speaker_{index}": 2 for index in range(5)
    }

    state = sampler.state_dict()
    expected = [row["clip_id"] for row in sampler.sample(7)]
    restored = DomainSpeakerBalancedSampler(beat + replay, seed=999)
    restored.load_state_dict(state)
    assert [row["clip_id"] for row in restored.sample(7)] == expected


def test_native_bucket_sampler_is_homogeneous_budgeted_and_exactly_resumable():
    rows = _beat_episodes()[:5]
    for row, frames in zip(rows, (5, 7, 17, 33, 70), strict=True):
        row["actions"] = _trajectory(ACTION_DIM, frames=frames)
        row["duration_sec"] = (frames - 1) / 30.0
    episodes = canonicalize_beat_episodes(rows)
    sampler = NativeLengthBucketSampler(
        episodes,
        buckets=(8, 32, 64),
        sampler_config={"mode": "domain_speaker"},
        seed=29,
    )
    batching = {
        "max_motion_tokens_per_microbatch": 128,
        "max_attention_elements_per_microbatch": 20_000,
    }
    audit = sampler.validate_budgets(
        semantic_tokens=7, max_batch_size=8, batching=batching
    )
    assert audit["maximum_native_frame_count"] == 70
    assert audit["maximum_bucket_frames"] == 96
    assert audit["no_cropping"] is True

    observed = []
    for _ in range(6):
        microbatch, plan = sampler.sample_microbatch(
            remaining_effective_batch=8,
            semantic_tokens=7,
            max_batch_size=8,
            batching=batching,
        )
        buckets = {
            native_length_bucket(len(row["actions"]), (8, 32, 64))
            for row in microbatch
        }
        assert buckets == {plan["bucket_frames"]}
        assert plan["microbatch_size"] <= plan["capacity"]
        assert plan["motion_tokens"] <= batching[
            "max_motion_tokens_per_microbatch"
        ]
        assert plan["attention_elements"] <= batching[
            "max_attention_elements_per_microbatch"
        ]
        observed.append((plan["bucket_frames"], [row["clip_id"] for row in microbatch]))

    state = sampler.state_dict()
    expected, expected_plan = sampler.sample_microbatch(
        remaining_effective_batch=8,
        semantic_tokens=7,
        max_batch_size=8,
        batching=batching,
    )
    restored = NativeLengthBucketSampler(
        episodes,
        buckets=(8, 32, 64),
        sampler_config={"mode": "domain_speaker"},
        seed=29,
    )
    restored.load_state_dict(state)
    actual, actual_plan = restored.sample_microbatch(
        remaining_effective_batch=8,
        semantic_tokens=7,
        max_batch_size=8,
        batching=batching,
    )
    assert actual_plan == expected_plan
    assert [row["clip_id"] for row in actual] == [row["clip_id"] for row in expected]


def test_native_microbatch_budget_fails_before_cropping_a_long_episode():
    with pytest.raises(ValueError, match="full native sequence exceeds"):
        native_length_microbatch_capacity(
            2592,
            semantic_tokens=7,
            max_batch_size=16,
            max_motion_tokens=4096,
            max_attention_elements=6_000_000,
        )
    plan = native_length_microbatch_capacity(
        2592,
        semantic_tokens=7,
        max_batch_size=16,
        max_motion_tokens=4096,
        max_attention_elements=8_000_000,
    )
    assert plan["capacity"] == 1
    assert plan["sequence_tokens"] == 2599


def test_replay_probe_and_regression_guard_are_deterministic_and_fail_closed():
    replay = canonicalize_kimodo_replay(_replay_episodes())
    first = deterministic_replay_evaluation_subset(replay, count=1, seed=17)
    second = deterministic_replay_evaluation_subset(list(reversed(replay)), count=1, seed=17)
    assert [row["clip_id"] for row in first] == [row["clip_id"] for row in second]

    passed = replay_regression_guard(
        {"total": 1.02},
        {"total": 1.0},
        maximum_fraction=0.03,
        maximum_absolute=0.01,
    )
    failed = replay_regression_guard(
        {"total": 1.04},
        {"total": 1.0},
        maximum_fraction=0.03,
        maximum_absolute=0.01,
    )
    assert passed["passed"] is True
    assert failed["passed"] is False
    safe_input = {
        "input_formal_release_eligible": True,
        "unsafe_training_data": False,
        "training_scope": "formal_variable_length_semantic_units",
        "formal_training_enabled": True,
        "temporal_unit_policy": "full_semantic_unit_variable_length_30hz",
        "batching_mode": "native_variable_length",
        "semantic_boundary_contract_validated": True,
        "training_policy": "full_network",
        "generator_initialization_mode": "pretrained_generator_warm_start",
        "forgetting_guard_applicable": True,
    }
    assert posttrain_release_decision(safe_input, passed)[
        "formal_release_eligible"
    ] is True
    missing = replay_regression_guard(
        None,
        None,
        maximum_fraction=0.03,
        maximum_absolute=0.01,
    )
    decision = posttrain_release_decision(safe_input, missing)
    assert decision["formal_release_eligible"] is False
    assert decision["artifact_status"].startswith("blocked_missing")

    experimental = posttrain_release_decision(
        safe_input
        | {
            "training_scope": "head_mechanism_experiment_only",
            "formal_training_enabled": False,
            "temporal_unit_policy": "fixed_window_experimental",
        },
        passed,
    )
    assert experimental["formal_release_eligible"] is False
    assert experimental["artifact_status"] == (
        "experimental_fixed_window_head_mechanism_only"
    )

    from_scratch = safe_input | {
        "generator_initialization_mode": (
            "full_generator_random_qwen_lora_frozen_v1"
        ),
        "forgetting_guard_applicable": False,
    }
    no_guard = replay_regression_guard(
        None,
        None,
        maximum_fraction=0.03,
        maximum_absolute=0.01,
    )
    random_decision = posttrain_release_decision(from_scratch, no_guard)
    assert random_decision["formal_release_eligible"] is True
    assert random_decision["replay_regression_guard_required"] is False

    clean_from_scratch = from_scratch | {
        "generator_initialization_mode": MOTION_ONLY_RANDOM_INIT_MODE
    }
    clean_decision = posttrain_release_decision(clean_from_scratch, no_guard)
    assert clean_decision["formal_release_eligible"] is True
    assert clean_decision["replay_regression_guard_required"] is False


def test_duration_fallback_uses_sample_intervals_and_formal_scope_fails_closed():
    episode = canonicalize_beat_episodes(_beat_episodes()[:1])[0]
    episode.pop("duration_sec")
    _, _, _, durations = batch_tensors(
        [episode],
        frame_count=4,
        action_stats={
            "mean": np.zeros(ACTION_DIM, dtype=np.float32),
            "std": np.ones(ACTION_DIM, dtype=np.float32),
        },
        device="cpu",
    )
    assert durations.item() == pytest.approx((7 - 1) / 30.0)

    with pytest.raises(ValueError, match="requires native_variable_length"):
        resolve_posttrain_config(
            {
                "training_scope": "formal_variable_length_semantic_units",
                "formal_training_enabled": True,
                "temporal_unit_policy": "full_semantic_unit_variable_length_30hz",
            }
        )
    formal = resolve_posttrain_config(
        {
            "training_scope": "formal_variable_length_semantic_units",
            "formal_training_enabled": True,
            "temporal_unit_policy": "full_semantic_unit_variable_length_30hz",
            "batching": {
                "mode": "native_variable_length",
                "length_buckets": [8, 16],
            },
        }
    )
    assert formal["batching"]["mode"] == "native_variable_length"
    assert formal["batching"]["homogeneous_bucket_batches"] is True
    assert formal["batching"]["target_effective_batch_size"] == 16
    assert formal["batching"]["gradient_accumulation_mode"] == (
        "dynamic_episode_weighted"
    )


def test_frame_coverage_duration_cannot_enter_posttrain_planner():
    episode = _beat_episodes()[:1][0]
    episode["duration_sec"] = len(episode["actions"]) / episode["fps"]
    with pytest.raises(ValueError, match="frame coverage N/fps cannot supervise"):
        canonicalize_beat_episodes([episode])


def test_native_evaluation_metric_is_invariant_to_batch_size_and_input_order():
    frame_counts = (5, 17, 8)
    episodes = []
    for index, frames in enumerate(frame_counts):
        episode = _beat_episodes()[index]
        episode["actions"] = _trajectory(
            ACTION_DIM, phase_offset=0.2 * index, frames=frames
        )
        episode["duration_sec"] = (frames - 1) / 30.0
        episodes.append(episode)
    episodes = canonicalize_beat_episodes(episodes)
    torch.manual_seed(101)
    model = create_ula_model(
        ULA_MMDIT_V2_ARCHITECTURE,
        action_dim=ACTION_DIM,
        condition_dim=KIMODO_V2_CONDITION_DIM,
        hidden_dim=16,
        layers=1,
        semantic_tokens=6,
    )
    kwargs = {
        "action_stats": {
            "mean": torch.zeros(ACTION_DIM),
            "std": torch.ones(ACTION_DIM),
        },
        "frame_count": 8,
        "device": "cpu",
        "loss_weights": {
            "flow": 1.0,
            "position": 1.0,
            "body": 0.0,
            "velocity": 0.1,
            "acceleration": 0.01,
        },
        "seed": 919,
        "batching": {"mode": "native_variable_length", "length_buckets": [8, 32]},
    }

    first = evaluate_posttrain(model, episodes, batch_size=1, **kwargs)
    second = evaluate_posttrain(model, episodes, batch_size=2, **kwargs)
    third = evaluate_posttrain(model, list(reversed(episodes)), batch_size=3, **kwargs)

    for name in ("flow", "position", "body", "velocity", "acceleration", "total"):
        assert first[name] == second[name] == third[name]
    assert first["by_domain"] == second["by_domain"] == third["by_domain"]
    assert first["batching"]["metric_accumulation"] == (
        "per_episode_stable_seed_equal_episode_weight"
    )


def test_native_training_accumulates_episode_weighted_microbatches_without_cropping(
    tmp_path,
):
    initial = tmp_path / "initial_18d.pt"
    _make_checkpoint(initial)
    output = tmp_path / "native_posttrain"
    summary = train_18d_posttrain(
        initial_checkpoint_path=initial,
        beat_episodes=_beat_episodes(accepted=False),
        output_dir=output,
        config={
            "steps": 1,
            "batch_size": 4,
            "validation_batch_size": 2,
            "phase_frame_choices": [4],
            "lr": 1e-5,
            "warmup_steps": 1,
            "validation_interval": 1,
            "checkpoint_interval": 1,
            "log_interval": 1,
            "early_stopping_patience": 10,
            "ema_decay": 0.9,
            "device": "cpu",
            "seed": 43,
            "allow_unsafe_training_data": True,
            "batching": {
                "mode": "native_variable_length",
                "length_buckets": [8],
                "max_motion_tokens_per_microbatch": 8,
                "max_attention_elements_per_microbatch": 196,
                "target_effective_batch_size": 4,
            },
        },
    )

    events = [
        json.loads(line)
        for line in (output / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    step_events = [event for event in events if event.get("step") == 1]
    assert len(step_events) == 1
    event = step_events[0]
    assert event["microbatch_count"] == 4
    assert event["native_frame_counts"] == [7, 7, 7, 7]
    assert event["frames"] == 8
    assert event["batch_domain_counts"] == {"beat2": 4}
    assert all(
        plan["microbatch_size"] == 1
        and plan["bucket_frames"] == 8
        and plan["motion_tokens"] == 8
        and plan["attention_elements"] == 196
        for plan in event["microbatch_plans"]
    )
    assert summary["native_batching_audit"]["no_cropping"] is True
    assert summary["native_batching_audit"]["maximum_native_frame_count"] == 7


def test_native_training_resume_matches_uninterrupted_model_and_bucket_schedule(
    tmp_path,
):
    initial = tmp_path / "initial_18d.pt"
    _make_checkpoint(initial)
    config = {
        "steps": 2,
        "batch_size": 4,
        "validation_batch_size": 2,
        "phase_frame_choices": [4],
        "lr": 1e-5,
        "warmup_steps": 1,
        "validation_interval": 1,
        "checkpoint_interval": 1,
        "log_interval": 1,
        "early_stopping_patience": 10,
        "ema_decay": 0.9,
        "device": "cpu",
        "seed": 47,
        "allow_unsafe_training_data": True,
        "batching": {
            "mode": "native_variable_length",
            "length_buckets": [8],
            "max_motion_tokens_per_microbatch": 8,
            "max_attention_elements_per_microbatch": 196,
            "target_effective_batch_size": 4,
        },
    }
    episodes = _beat_episodes(accepted=False)
    uninterrupted_dir = tmp_path / "uninterrupted"
    resumed_dir = tmp_path / "resumed"
    train_18d_posttrain(
        initial_checkpoint_path=initial,
        beat_episodes=episodes,
        output_dir=uninterrupted_dir,
        config=config,
    )
    train_18d_posttrain(
        initial_checkpoint_path=initial,
        beat_episodes=episodes,
        output_dir=resumed_dir,
        config=config | {"steps": 1},
    )
    train_18d_posttrain(
        initial_checkpoint_path=initial,
        beat_episodes=episodes,
        output_dir=resumed_dir,
        config=config | {"resume_from": str(resumed_dir / "last.pt")},
    )

    uninterrupted = torch.load(
        uninterrupted_dir / "last.pt", map_location="cpu", weights_only=True
    )
    resumed = torch.load(
        resumed_dir / "last.pt", map_location="cpu", weights_only=True
    )
    assert uninterrupted["posttrain_step"] == resumed["posttrain_step"] == 2
    first_state = uninterrupted["training_state"]
    second_state = resumed["training_state"]
    for field in ("raw_model_state_dict", "ema_state_dict"):
        assert first_state[field].keys() == second_state[field].keys()
        for name in first_state[field]:
            assert torch.equal(first_state[field][name], second_state[field][name]), name
    assert first_state["sampler_state_dict"] == second_state["sampler_state_dict"]

    def step_two_event(path):
        return next(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if json.loads(line).get("step") == 2
        )

    first_event = step_two_event(uninterrupted_dir / "progress.jsonl")
    second_event = step_two_event(resumed_dir / "progress.jsonl")
    for field in (
        "microbatch_count",
        "microbatch_plans",
        "native_frame_counts",
        "batch_domain_counts",
        "train",
    ):
        assert first_event[field] == second_event[field]


def test_disjoint_replay_validation_and_final_test_and_clean_challenge_selection(
    tmp_path,
):
    initial = tmp_path / "initial_18d.pt"
    _make_checkpoint(initial)
    beat = _beat_episodes(accepted=False)
    split, _ = strict_group_split(beat, seed=37)
    validation_challenge = split["validation"][0]["clip_id"]
    test_challenge = split["test"][0]["clip_id"]
    for episode in beat:
        episode["temporal_quarantine_challenge"] = episode["clip_id"] in {
            validation_challenge,
            test_challenge,
        }
    train_replay = _replay_episodes()
    validation_replay = [
        dict(row, episode_index=row["episode_index"] + 10, replay_source_split="validation")
        for row in _replay_episodes()
    ]
    test_replay = [
        dict(row, episode_index=row["episode_index"] + 20, replay_source_split="test")
        for row in _replay_episodes()
    ]
    config = {
        "steps": 1,
        "batch_size": 4,
        "validation_batch_size": 2,
        "phase_frame_choices": [4],
        "lr": 1e-5,
        "warmup_steps": 1,
        "validation_interval": 1,
        "checkpoint_interval": 1,
        "log_interval": 1,
        "early_stopping_patience": 10,
        "ema_decay": 0.9,
        "device": "cpu",
        "seed": 37,
        "allow_unsafe_training_data": True,
        "require_disjoint_replay_evaluation": True,
    }
    summary = train_18d_posttrain(
        initial_checkpoint_path=initial,
        beat_episodes=beat,
        kimodo_replay_episodes=train_replay,
        kimodo_replay_probe_episodes=validation_replay,
        kimodo_replay_test_episodes=test_replay,
        output_dir=tmp_path / "disjoint",
        config=config,
    )
    assert summary["replay_evaluation_source"] == "disjoint_original_validation"
    assert summary["replay_test_policy"] == (
        "original_test_evaluated_only_after_model_selection"
    )
    assert summary["initial_replay_test"] is None
    assert summary["final_replay_test"] is not None
    assert summary["initial_validation_temporal_challenge"] is not None
    assert summary["final_validation_temporal_challenge"] is not None
    assert summary["model_selection_evaluation_policy"].startswith(
        "clean_validation_only"
    )
    checkpoint = torch.load(summary["checkpoint"], map_location="cpu", weights_only=True)
    role_by_id = {
        row["clip_id"]: row["split"]
        for row in checkpoint["posttrain_data_contract"]["records"]
        if row["domain"] == "kimodo"
    }
    assert {role_by_id[f"kimodo:{index}"] for index in (0, 1)} == {
        "train_replay_only"
    }
    assert {role_by_id[f"kimodo:{index}"] for index in (10, 11)} == {
        "replay_evaluation_only"
    }
    assert {role_by_id[f"kimodo:{index}"] for index in (20, 21)} == {
        "replay_test_only"
    }

    with pytest.raises(ValueError, match="optimization/evaluation leakage"):
        train_18d_posttrain(
            initial_checkpoint_path=initial,
            beat_episodes=beat,
            kimodo_replay_episodes=train_replay,
            kimodo_replay_probe_episodes=train_replay,
            kimodo_replay_test_episodes=test_replay,
            output_dir=tmp_path / "leak",
            config=config,
        )


def test_cpu_posttrain_marks_unreviewed_checkpoint_unsafe_and_resumes(tmp_path):
    initial = tmp_path / "initial_18d.pt"
    _make_checkpoint(initial)
    output = tmp_path / "posttrain"
    base_config = {
        "steps": 2,
        "batch_size": 4,
        "validation_batch_size": 2,
        "phase_frame_choices": [4],
        "lr": 1e-5,
        "minimum_lr_ratio": 0.5,
        "warmup_steps": 1,
        "validation_interval": 1,
        "checkpoint_interval": 1,
        "log_interval": 1,
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 0.0,
        "ema_decay": 0.9,
        "device": "cpu",
        "seed": 23,
        "allow_unsafe_training_data": True,
    }
    summary = train_18d_posttrain(
        initial_checkpoint_path=initial,
        beat_episodes=_beat_episodes(accepted=False),
        kimodo_replay_episodes=_replay_episodes(),
        output_dir=output,
        config=base_config,
    )

    assert summary["completed_steps"] == 2
    assert summary["artifact_status"] == "experimental_unreviewed_unsafe"
    assert summary["formal_release_eligible"] is False
    assert summary["initial_validation"]["total"] >= summary["final_validation"]["total"]
    assert summary["initial_test"] is None
    assert summary["initial_replay_validation"]["total"] >= 0.0
    assert summary["final_replay_validation"]["total"] >= 0.0
    assert summary["final_replay_regression_guard"]["passed"] is True
    assert summary["replay_evaluation_count"] == len(_replay_episodes())
    assert summary["data_provenance"]["emotion_unresolved_beat_count"] == 10
    assert "beat_motion_not_all_adjudicated_train_ready" in summary["data_provenance"][
        "unsafe_reasons"
    ]
    with pytest.raises(ValueError, match="permanently forbidden dataset 'kimodo'"):
        load_contract_checkpoint(
            summary["checkpoint"], expected_action_dim=ACTION_DIM
        )
    checkpoint = torch.load(
        summary["checkpoint"], map_location="cpu", weights_only=True
    )
    assert checkpoint["action_dim"] == ACTION_DIM
    assert checkpoint["posttrain_artifact_kind"].endswith("interaction_posttrain")
    assert checkpoint["unsafe_training_data"] is True
    assert checkpoint["formal_release_eligible"] is False
    assert checkpoint["artifact_status"] == summary["artifact_status"]
    assert checkpoint["validation_metrics"] == summary["final_validation"]
    assert checkpoint["best_validation_loss"] == summary["best_validation_loss"]
    assert checkpoint["config"]["checkpoint_loss"] == summary["final_validation"]["total"]
    assert checkpoint["data_provenance"]["release_status"] == summary[
        "artifact_status"
    ]
    assert checkpoint["data_provenance"]["formal_release_eligible"] is False
    assert summary["data_provenance"]["release_status"] == summary["artifact_status"]
    assert checkpoint["training_contract"]["replay_regression_guard"] == summary[
        "final_replay_regression_guard"
    ]
    assert checkpoint["training_contract"]["all_model_parameters_trainable"] is True
    assert checkpoint["planner_supervision_contract"] == checkpoint[
        "training_contract"
    ]["planner_supervision"]
    assert checkpoint["planner_supervision_contract"][
        "duration_head_supervision"
    ] == "native_output_sample_span_(N-1)/fps"
    assert checkpoint["training_contract"]["emotion_policy"].endswith(
        "blind_match_dual_gate"
    )
    replay_records = [
        row
        for row in checkpoint["posttrain_data_contract"]["records"]
        if row["domain"] == "kimodo"
    ]
    assert replay_records
    assert all(row["action_dim_mask"][-3:] == [False, False, False] for row in replay_records)
    assert Path(summary["last_checkpoint"]).is_file()
    assert json.loads((output / "split_manifest.json").read_text())["sha256"]

    with pytest.raises(ValueError, match="resume config mismatch for warmup_steps"):
        train_18d_posttrain(
            initial_checkpoint_path=initial,
            beat_episodes=_beat_episodes(accepted=False),
            kimodo_replay_episodes=_replay_episodes(),
            output_dir=output,
            config=base_config
            | {
                "steps": 3,
                "warmup_steps": 2,
                "resume_from": str(output / "last.pt"),
            },
        )

    resumed = train_18d_posttrain(
        initial_checkpoint_path=initial,
        beat_episodes=_beat_episodes(accepted=False),
        kimodo_replay_episodes=_replay_episodes(),
        output_dir=output,
        config=base_config
        | {"steps": 3, "resume_from": str(output / "last.pt")},
    )
    assert resumed["completed_steps"] == 3
    resumed_last = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
    assert resumed_last["posttrain_step"] == 3
    assert "optimizer_state_dict" in resumed_last["training_state"]
    assert "current_replay_guard" in resumed_last["training_state"]
    assert "best_replay_guard" in resumed_last["training_state"]

    # Simulate a crash after the final last.pt write but before finalization.
    (output / "training_summary.json").unlink()
    (output / "ula_fm_checkpoint.pt").unlink()
    finalized = train_18d_posttrain(
        initial_checkpoint_path=initial,
        beat_episodes=_beat_episodes(accepted=False),
        kimodo_replay_episodes=_replay_episodes(),
        output_dir=output,
        config=base_config
        | {"steps": 3, "resume_from": str(output / "last.pt")},
    )
    assert finalized["completed_steps"] == 3

    # A crash after an early-stop checkpoint must not run one more update.
    early_stop_last = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
    early_stop_last["training_state"]["stale_validations"] = base_config[
        "early_stopping_patience"
    ]
    torch.save(early_stop_last, output / "last.pt")
    (output / "training_summary.json").unlink()
    (output / "ula_fm_checkpoint.pt").unlink()
    stopped = train_18d_posttrain(
        initial_checkpoint_path=initial,
        beat_episodes=_beat_episodes(accepted=False),
        kimodo_replay_episodes=_replay_episodes(),
        output_dir=output,
        config=base_config
        | {"steps": 4, "resume_from": str(output / "last.pt")},
    )
    assert stopped["stopped_early"] is True
    assert stopped["completed_steps"] == 3


def test_strict_mode_refuses_unreviewed_or_unversioned_inputs(tmp_path):
    initial = tmp_path / "initial_18d.pt"
    _make_checkpoint(initial)
    with pytest.raises(ValueError, match="refuses unsafe/unadjudicated"):
        train_18d_posttrain(
            initial_checkpoint_path=initial,
            beat_episodes=_beat_episodes(accepted=False),
            output_dir=tmp_path / "strict_run",
            config={
                "steps": 1,
                "batch_size": 2,
                "phase_frame_choices": [4],
                "validation_interval": 1,
                "checkpoint_interval": 1,
                "log_interval": 1,
                "early_stopping_patience": 1,
                "device": "cpu",
            },
        )
