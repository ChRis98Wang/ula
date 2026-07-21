import numpy as np
import pytest
import torch

from upper_body_skeleton.kimodo_semantics import KIMODO_BEHAVIOR_IDS, KIMODO_EMOTION_IDS
from upper_body_skeleton.retarget_v2 import JOINT_ORDER
from upper_body_skeleton.ula_training import (
    KIMODO_V2_CONDITION_DIM,
    ULA_MMDIT_V2_ARCHITECTURE,
    UlaMMDiTV2Model,
    create_ula_model,
)
from upper_body_skeleton.ula_training_v2 import (
    ModelEMA,
    SemanticModeSampler,
    flow_matching_v2_objective,
    resample_motion_phase,
    resolve_v2_config,
)


def test_mmdit_v2_uses_structured_condition_tokens_and_variable_frames():
    model = create_ula_model(
        ULA_MMDIT_V2_ARCHITECTURE,
        action_dim=len(JOINT_ORDER),
        condition_dim=KIMODO_V2_CONDITION_DIM,
        hidden_dim=64,
        layers=1,
        semantic_tokens=7,
    )
    condition = torch.randn(2, KIMODO_V2_CONDITION_DIM)

    short = model(torch.randn(2, 37, len(JOINT_ORDER)), torch.tensor([0.2, 0.8]), condition)
    long = model(torch.randn(2, 83, len(JOINT_ORDER)), torch.tensor([0.2, 0.8]), condition)

    assert isinstance(model, UlaMMDiTV2Model)
    assert short.shape == (2, 37, len(JOINT_ORDER))
    assert long.shape == (2, 83, len(JOINT_ORDER))
    assert model.last_joint_sequence_shape == (2, 90, 64)


def test_resample_motion_phase_preserves_endpoints_and_is_not_fixed_to_150():
    actions = np.linspace(0.0, 1.0, 21, dtype=np.float32)[:, None] * np.ones((1, len(JOINT_ORDER)), dtype=np.float32)

    resampled = resample_motion_phase(actions, 47)

    assert resampled.shape == (47, len(JOINT_ORDER))
    assert np.allclose(resampled[0], actions[0])
    assert np.allclose(resampled[-1], actions[-1])


def test_v2_objective_has_finite_structural_and_duration_gradients():
    model = UlaMMDiTV2Model(
        action_dim=len(JOINT_ORDER),
        condition_dim=KIMODO_V2_CONDITION_DIM,
        hidden_dim=64,
        layers=1,
        semantic_tokens=6,
    )
    actions = torch.randn(2, 24, len(JOINT_ORDER)) * 0.1
    condition = torch.randn(2, KIMODO_V2_CONDITION_DIM)
    durations = torch.tensor([1.2, 3.4])
    stats = {"mean": torch.zeros(len(JOINT_ORDER)), "std": torch.ones(len(JOINT_ORDER))}
    weights = {
        "flow": 1.0,
        "position": 0.1,
        "velocity": 0.01,
        "acceleration": 0.001,
        "descriptor": 0.01,
        "motion_latent": 0.0,
        "duration": 0.1,
    }

    losses = flow_matching_v2_objective(
        model,
        actions,
        condition,
        durations,
        loss_weights=weights,
        action_stats=stats,
    )
    losses["total"].backward()

    assert set(losses) == set(weights) | {"total"}
    assert all(torch.isfinite(value) for value in losses.values())
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_semantic_mode_sampler_covers_every_group_and_restores_state():
    episodes = []
    for behavior_id in KIMODO_BEHAVIOR_IDS:
        for emotion_id in KIMODO_EMOTION_IDS:
            episodes.append(
                {
                    "meta": {"behavior_id": behavior_id, "emotion_id": emotion_id},
                    "style_controls": np.zeros(3, dtype=np.float32),
                }
            )
    sampler = SemanticModeSampler(episodes, seed=3)
    state = sampler.state_dict()
    first = sampler.sample(len(episodes))
    sampler.load_state_dict(state)
    replay = sampler.sample(len(episodes))

    first_keys = [(row["meta"]["behavior_id"], row["meta"]["emotion_id"]) for row in first]
    replay_keys = [(row["meta"]["behavior_id"], row["meta"]["emotion_id"]) for row in replay]
    assert len(set(first_keys)) == len(episodes)
    assert replay_keys == first_keys


def test_ema_temporarily_applies_shadow_and_restores_model():
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = ModelEMA(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(model)

    with ema.apply(model):
        assert torch.allclose(model.weight, torch.full_like(model.weight, 2.0))
    assert torch.allclose(model.weight, torch.full_like(model.weight, 3.0))


def test_v2_config_requires_explicit_data_contract_paths():
    with pytest.raises(ValueError, match="requires"):
        resolve_v2_config({"output_dir": "/tmp/run"})

    config = resolve_v2_config(
        {
            "dataset_dir": "/tmp/data",
            "split_checkpoint": "/tmp/split.pt",
            "output_dir": "/tmp/run",
            "phase_frame_choices": [32, 64],
        }
    )
    assert config["architecture"] == ULA_MMDIT_V2_ARCHITECTURE
    assert config["phase_frame_choices"] == [32, 64]
