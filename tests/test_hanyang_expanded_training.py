from __future__ import annotations

from pathlib import Path

import pytest
import torch

from upper_body_skeleton.hanyang_expanded_training import (
    HANYANG_P7_ORDER,
    HANYANG_Q2_ORDER,
    HANYANG_Q6_ORDER,
    PERMANENTLY_UNOBSERVED_DOF_INDICES,
    derivative_observation_weights,
    hanyang_p7_to_hierarchy_targets,
    hanyang_training_admission,
    observation_weight_sha256,
    reject_kimodo_strings,
    validate_observation_weight,
    weighted_masked_mean,
)


def _valid_weight(
    *,
    batch: int = 2,
    frames: int = 6,
) -> torch.Tensor:
    weight = torch.linspace(
        0.15,
        1.0,
        batch * frames * 18,
        dtype=torch.float32,
    ).reshape(batch, frames, 18)
    weight[..., list(PERMANENTLY_UNOBSERVED_DOF_INDICES)] = 0.0
    return weight


def test_validate_observation_weight_is_strict_and_padding_is_zero() -> None:
    weight = _valid_weight()
    valid = torch.tensor(
        [
            [True, True, True, False, False, False],
            [True, True, True, True, True, False],
        ],
        dtype=torch.bool,
    )
    weight = weight * valid.unsqueeze(-1)
    assert validate_observation_weight(
        weight, frame_valid_mask=valid
    ) is weight

    for index in PERMANENTLY_UNOBSERVED_DOF_INDICES:
        changed = weight.clone()
        changed[0, 0, index] = 1e-8
        with pytest.raises(ValueError, match="exactly zero"):
            validate_observation_weight(changed)

    out_of_range = weight.clone()
    out_of_range[0, 0, 0] = 1.001
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        validate_observation_weight(out_of_range)

    nan_weight = weight.clone()
    nan_weight[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        validate_observation_weight(nan_weight)

    padded_nonzero = weight.clone()
    padded_nonzero[0, 5, 0] = 0.25
    with pytest.raises(ValueError, match="padded"):
        validate_observation_weight(
            padded_nonzero, frame_valid_mask=valid
        )


def test_derivative_weights_are_adjacent_two_three_four_frame_minima() -> None:
    weight = torch.ones(1, 5, 18, dtype=torch.float32)
    weight[..., list(PERMANENTLY_UNOBSERVED_DOF_INDICES)] = 0.0
    weight[0, :, 0] = torch.tensor([0.9, 0.4, 0.8, 0.2, 0.7])

    derivative = derivative_observation_weights(weight)

    assert derivative.velocity.shape == (1, 4, 18)
    assert derivative.acceleration.shape == (1, 3, 18)
    assert derivative.jerk.shape == (1, 2, 18)
    assert torch.equal(
        derivative.velocity[0, :, 0],
        torch.tensor([0.4, 0.4, 0.2, 0.2]),
    )
    assert torch.equal(
        derivative.acceleration[0, :, 0],
        torch.tensor([0.4, 0.2, 0.2]),
    )
    assert torch.equal(
        derivative.jerk[0, :, 0],
        torch.tensor([0.2, 0.2]),
    )
    assert torch.count_nonzero(
        derivative.jerk[
            ..., list(PERMANENTLY_UNOBSERVED_DOF_INDICES)
        ]
    ) == 0


def test_derivative_weights_return_empty_time_axes_when_unavailable() -> None:
    weight = _valid_weight(batch=1, frames=1)
    derivative = derivative_observation_weights(weight)
    assert derivative.velocity.shape == (1, 0, 18)
    assert derivative.acceleration.shape == (1, 0, 18)
    assert derivative.jerk.shape == (1, 0, 18)


def test_weighted_masked_mean_ignores_zero_weight_values_and_gradients() -> None:
    values = torch.tensor(
        [1.0, 2.0, 3.0, 4.0],
        dtype=torch.float32,
        requires_grad=True,
    )
    weight = torch.tensor([1.0, 0.0, 0.5, 0.0], dtype=torch.float32)
    loss = weighted_masked_mean(values, weight)
    changed = values.detach().clone()
    changed[1] = 1e20
    changed[3] = torch.nan
    changed_loss = weighted_masked_mean(changed, weight)

    assert torch.equal(loss.detach(), changed_loss)
    assert loss.item() == pytest.approx((1.0 + 0.5 * 3.0) / 1.5)
    loss.backward()
    assert torch.equal(
        values.grad,
        torch.tensor([2.0 / 3.0, 0.0, 1.0 / 3.0, 0.0]),
    )


def test_weighted_masked_mean_all_zero_is_differentiable_zero() -> None:
    values = torch.tensor(
        [torch.nan, 99.0], dtype=torch.float32, requires_grad=True
    )
    loss = weighted_masked_mean(values, torch.zeros_like(values))
    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(values.grad, torch.zeros_like(values))


def test_observation_weight_hash_binds_values_shape_and_dtype() -> None:
    weight = _valid_weight()
    assert observation_weight_sha256(weight) == observation_weight_sha256(
        weight.clone()
    )
    changed = weight.clone()
    changed[0, 0, 0] += 0.01
    assert observation_weight_sha256(changed) != observation_weight_sha256(
        weight
    )
    assert observation_weight_sha256(weight.double()) != (
        observation_weight_sha256(weight)
    )


def test_p7_maps_to_v7_q2_q6_and_disgust_discount() -> None:
    assert HANYANG_P7_ORDER == (
        "happy",
        "sad",
        "surprise",
        "angry",
        "disgust",
        "fear",
        "neutral",
    )
    assert HANYANG_Q2_ORDER == ("neutral", "non_neutral")
    assert HANYANG_Q6_ORDER == (
        "neutral",
        "sad",
        "happy",
        "angry",
        "surprise",
        "fear",
    )
    p7 = torch.tensor(
        [[0.20, 0.10, 0.15, 0.05, 0.25, 0.05, 0.20]],
        dtype=torch.float32,
    )
    targets = hanyang_p7_to_hierarchy_targets(p7)

    assert torch.allclose(targets.q2, torch.tensor([[0.2, 0.8]]))
    assert torch.allclose(
        targets.q6,
        torch.tensor([[0.20, 0.10, 0.20, 0.05, 0.15, 0.05]]) / 0.75,
    )
    assert targets.disgust_mass.item() == pytest.approx(0.25)
    assert targets.six_class_supervision_weight.item() == pytest.approx(0.75)
    assert targets.q6.sum().item() == pytest.approx(1.0)


def test_pure_disgust_has_q6_zero_weight_and_is_not_hard_mapped() -> None:
    p7 = torch.zeros(1, 7, dtype=torch.float32)
    p7[0, HANYANG_P7_ORDER.index("disgust")] = 1.0
    targets = hanyang_p7_to_hierarchy_targets(p7)

    assert torch.equal(targets.q2, torch.tensor([[0.0, 1.0]]))
    assert torch.count_nonzero(targets.q6) == 0
    assert targets.disgust_mass.item() == 1.0
    assert targets.six_class_supervision_weight.item() == 0.0


@pytest.mark.parametrize(
    "p7",
    (
        torch.ones(7, dtype=torch.float32),
        torch.tensor(
            [1.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=torch.float32,
        ),
        torch.tensor(
            [torch.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            dtype=torch.float32,
        ),
    ),
)
def test_p7_validation_fails_closed(p7: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        hanyang_p7_to_hierarchy_targets(p7)


def test_p7_rejects_probability_above_one_with_tolerated_sum_error() -> None:
    p7 = torch.zeros(7, dtype=torch.float32)
    p7[HANYANG_P7_ORDER.index("neutral")] = 1.0000005
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        hanyang_p7_to_hierarchy_targets(p7)


@pytest.mark.parametrize(
    (
        "qc_pass",
        "coverage",
        "agreement",
        "share",
        "motion_eligible",
        "emotion_eligible",
    ),
    (
        (True, True, True, 0.70, True, True),
        (True, True, True, 0.699999, True, False),
        (True, False, True, 0.90, True, False),
        (True, True, False, 0.90, True, False),
        (False, True, True, 0.90, False, False),
    ),
)
def test_admission_has_one_motion_lane_and_fail_closed_conditioning(
    qc_pass: bool,
    coverage: bool,
    agreement: bool,
    share: float,
    motion_eligible: bool,
    emotion_eligible: bool,
) -> None:
    admission = hanyang_training_admission(
        qc_pass=qc_pass,
        rater_coverage_pass=coverage,
        intended_majority_agrees=agreement,
        intended_share=share,
        lineage={"dataset_source": "hanyang_duksung_v1"},
    )
    assert admission["only_unconditional_motion"] is bool(
        motion_eligible and not emotion_eligible
    )
    assert admission["unconditional_motion_eligible"] is motion_eligible
    assert admission["emotion_condition_eligible"] is emotion_eligible
    assert admission["group54_condition_eligible"] is False
    assert admission["style_condition_eligible"] is False
    assert admission["duration_condition_eligible"] is False
    assert admission["semantic_condition_eligible"] is False


@pytest.mark.parametrize(
    "payload",
    (
        "KIMODO",
        "/datasets/Ki-mo-do/train.npz",
        {"source": ["safe", {"cache_path": Path("/cache/k_i_m_o_d_o")}]},
        {"kimodo_manifest": "hidden-in-key"},
    ),
)
def test_kimodo_strings_are_rejected_recursively(payload: object) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        reject_kimodo_strings(payload)
    with pytest.raises(ValueError, match="forbidden"):
        hanyang_training_admission(
            qc_pass=True,
            rater_coverage_pass=True,
            intended_majority_agrees=True,
            intended_share=1.0,
            lineage=payload,
        )
