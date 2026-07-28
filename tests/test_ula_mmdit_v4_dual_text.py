from __future__ import annotations

import pytest
import torch

from upper_body_skeleton.ula_training import (
    KIMODO_V2_CONDITION_DIM,
    ULA_MMDIT_V4_DUAL_TEXT_ADALN_ARCHITECTURE,
    create_ula_model,
)
from upper_body_skeleton.ula_v2_18d_posttrain import masked_18d_objective


def _model():
    return create_ula_model(
        ULA_MMDIT_V4_DUAL_TEXT_ADALN_ARCHITECTURE,
        action_dim=18,
        condition_dim=KIMODO_V2_CONDITION_DIM,
        hidden_dim=32,
        layers=1,
        semantic_tokens=7,
    )


def test_v4_has_separate_action_directive_and_dialogue_parameters() -> None:
    model = _model()
    names = dict(model.named_parameters())
    assert "action_directive_condition.0.weight" in names
    assert "dialogue_condition.0.weight" in names
    assert names["action_directive_condition.0.weight"].data_ptr() != names[
        "dialogue_condition.0.weight"
    ].data_ptr()
    assert not any(name.startswith("motion_latent_condition") for name in names)


def test_v4_role_channels_change_different_semantic_tokens() -> None:
    model = _model().eval()
    baseline = torch.zeros(1, KIMODO_V2_CONDITION_DIM)
    directive = baseline.clone()
    dialogue = baseline.clone()
    directive[:, 136:200] = 1.0
    dialogue[:, 200:264] = 1.0
    with torch.no_grad():
        baseline_tokens = model.semantic_condition_tokens(baseline)
        directive_tokens = model.semantic_condition_tokens(directive)
        dialogue_tokens = model.semantic_condition_tokens(dialogue)
    directive_delta = (directive_tokens - baseline_tokens).abs().sum(dim=-1)
    dialogue_delta = (dialogue_tokens - baseline_tokens).abs().sum(dim=-1)
    assert torch.nonzero(directive_delta[0], as_tuple=False).flatten().tolist() == [5]
    assert torch.nonzero(dialogue_delta[0], as_tuple=False).flatten().tolist() == [6]


def test_v4_requires_exact_role_token_layout() -> None:
    with pytest.raises(ValueError, match="exactly seven"):
        create_ula_model(
            ULA_MMDIT_V4_DUAL_TEXT_ADALN_ARCHITECTURE,
            action_dim=18,
            condition_dim=KIMODO_V2_CONDITION_DIM,
            hidden_dim=32,
            layers=1,
            semantic_tokens=8,
        )


def test_counterfactual_action_loss_backpropagates_into_both_text_roles() -> None:
    model = _model().train()
    actions = torch.randn(2, 8, 18)
    conditions = torch.zeros(2, KIMODO_V2_CONDITION_DIM)
    conditions[:, 136:200] = torch.nn.functional.normalize(torch.randn(2, 64), dim=-1)
    conditions[:, 200:264] = torch.nn.functional.normalize(torch.randn(2, 64), dim=-1)
    negatives = conditions[:, None, :].repeat(1, 2, 1)
    negatives[:, 0, 200:264] = torch.nn.functional.normalize(
        torch.randn(2, 64), dim=-1
    )
    negatives[:, 1, 136:200] = torch.nn.functional.normalize(
        torch.randn(2, 64), dim=-1
    )
    losses = masked_18d_objective(
        model,
        actions,
        conditions,
        torch.ones(2, 18, dtype=torch.bool),
        torch.ones(2),
        loss_weights={"flow": 1.0, "condition_contrastive": 0.5},
        counterfactual_conditions=negatives,
        noise=torch.randn_like(actions),
        flow_times=torch.tensor([0.25, 0.75]),
    )
    assert losses["condition_contrastive"].ndim == 0
    assert losses["dialogue_contrastive"].ndim == 0
    assert losses["action_directive_contrastive"].ndim == 0
    assert torch.allclose(
        losses["condition_contrastive"],
        (losses["dialogue_contrastive"] + losses["action_directive_contrastive"])
        / 2.0,
    )
    losses["total"].backward()
    assert model.action_directive_condition[0].weight.grad is not None
    assert model.dialogue_condition[0].weight.grad is not None
