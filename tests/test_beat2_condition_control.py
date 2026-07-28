import copy

import numpy as np
import pytest
import torch

from upper_body_skeleton.beat2_condition_control import (
    CONDITION_DIM,
    QwenStyleHead,
    STYLE_CONTROL_SLICE,
    TEXT_LATENT_SLICE,
    aligned_vs_rolled_shuffled_hinge_loss,
    apply_condition_keep_mask,
    assemble_text_style_conditions,
    build_train_group_mean_style_baseline,
    masked_per_example_flow_mse,
    masked_style_control_mse,
    masked_style_control_smooth_l1,
    rolled_shuffled_conditions,
    sample_condition_keep_mask,
)


def _baseline_inputs():
    source = np.zeros((7, CONDITION_DIM), dtype=np.float32)
    source[:, 136] = np.arange(7, dtype=np.float32)
    targets = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0],
            [10.0, 20.0, 30.0],
            [1000.0, 1000.0, 1000.0],
            [-1000.0, -1000.0, -1000.0],
            [2000.0, 2000.0, 2000.0],
            [-2000.0, -2000.0, -2000.0],
        ],
        dtype=np.float32,
    )
    groups = np.asarray(["a", "a", "b", "a", "b", "unseen", "unseen"])
    splits = np.asarray(
        ["train", "train", "train", "validation", "test", "validation", "test"]
    )
    return source, targets, groups, splits


def test_train_group_baseline_is_cross_fit_and_eval_targets_are_unused():
    source, targets, groups, splits = _baseline_inputs()
    transformed, receipt = build_train_group_mean_style_baseline(
        source, targets, groups, splits
    )

    assert np.array_equal(
        transformed[:, : STYLE_CONTROL_SLICE.start],
        source[:, : STYLE_CONTROL_SLICE.start],
    )
    assert np.array_equal(
        transformed[:, STYLE_CONTROL_SLICE.stop :],
        source[:, STYLE_CONTROL_SLICE.stop :],
    )
    expected_global = np.asarray([14.0 / 3.0, 26.0 / 3.0, 38.0 / 3.0])
    assert np.array_equal(transformed[0, STYLE_CONTROL_SLICE], targets[1])
    assert np.array_equal(transformed[1, STYLE_CONTROL_SLICE], targets[0])
    assert np.allclose(
        transformed[2, STYLE_CONTROL_SLICE],
        np.asarray([2.0, 3.0, 4.0]),
    )
    assert np.allclose(
        transformed[3, STYLE_CONTROL_SLICE],
        np.asarray([2.0, 3.0, 4.0]),
    )
    assert np.array_equal(
        transformed[4, STYLE_CONTROL_SLICE],
        targets[2],
    )
    assert np.allclose(
        transformed[5:, STYLE_CONTROL_SLICE],
        expected_global[None, :],
    )
    assert receipt["generator_input_policy"] is False
    assert receipt["target_row_excluded_from_own_training_assignment"] is True
    assert receipt["evaluation_target_style_controls_used"] is False
    assert receipt["source_rows_used_by_split"] == {
        "train": 3,
        "validation": 0,
        "test": 0,
    }

    mutated_eval = targets.copy()
    mutated_eval[3:] += 99999.0
    transformed_eval_mutation, _ = build_train_group_mean_style_baseline(
        source, mutated_eval, groups, splits
    )
    assert np.array_equal(transformed_eval_mutation, transformed)

    for train_row in range(3):
        mutated_train = targets.copy()
        mutated_train[train_row] += np.asarray([123.0, 456.0, 789.0])
        changed, _ = build_train_group_mean_style_baseline(
            source, mutated_train, groups, splits
        )
        assert np.array_equal(
            changed[train_row, STYLE_CONTROL_SLICE],
            transformed[train_row, STYLE_CONTROL_SLICE],
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("nonzero_source_style", "style slice"),
        ("unknown_split", "unknown assignments"),
        ("nonfinite_source", "finite"),
        ("nonfinite_targets", "finite"),
        ("single_train", "at least two training"),
    ],
)
def test_train_group_baseline_fails_closed(mutation, match):
    source, targets, groups, splits = _baseline_inputs()
    if mutation == "nonzero_source_style":
        source[0, STYLE_CONTROL_SLICE.start] = 1.0
    elif mutation == "unknown_split":
        splits[0] = "dev"
    elif mutation == "nonfinite_source":
        source[0, 0] = np.nan
    elif mutation == "nonfinite_targets":
        targets[0, 0] = np.inf
    elif mutation == "single_train":
        splits[1:3] = "test"
    with pytest.raises(ValueError, match=match):
        build_train_group_mean_style_baseline(
            source, targets, groups, splits
        )


def test_qwen_style_head_mask_is_exact_and_state_contract_round_trips():
    torch.manual_seed(8)
    head = QwenStyleHead(hidden_dim=11)
    with torch.no_grad():
        head.output_projection.weight.fill_(0.25)
        head.output_projection.bias.fill_(0.5)
    latents = torch.randn(3, 128, requires_grad=True)
    keep = torch.tensor([True, False, True])
    output = head(latents, keep)
    assert output.shape == (3, 3)
    assert torch.count_nonzero(output[1]).item() == 0
    assert torch.count_nonzero(output[[0, 2]]).item() > 0
    output.sum().backward()
    assert torch.count_nonzero(latents.grad[1]).item() == 0
    assert torch.count_nonzero(latents.grad[[0, 2]]).item() > 0

    config = head.architecture_config()
    restored = QwenStyleHead.from_config(copy.deepcopy(config))
    restored.load_state_dict(head.state_dict(), strict=True)
    with torch.no_grad():
        assert torch.equal(restored(latents.detach(), keep), output.detach())


def test_qwen_style_head_zero_initializes_only_its_output_projection():
    torch.manual_seed(3)
    head = QwenStyleHead(hidden_dim=9)
    assert torch.count_nonzero(head.output_projection.weight).item() == 0
    assert torch.count_nonzero(head.output_projection.bias).item() == 0
    assert torch.count_nonzero(head.input_projection.weight).item() > 0
    result = head(torch.randn(2, 128), torch.ones(2, dtype=torch.bool))
    assert torch.count_nonzero(result).item() == 0


def test_condition_assembly_has_no_oracle_input_and_dropped_rows_are_zero():
    latents = torch.arange(3 * 128, dtype=torch.float32).reshape(3, 128)
    predicted = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    keep = torch.tensor([True, False, True])
    conditions = assemble_text_style_conditions(latents, predicted, keep)
    assert conditions.shape == (3, CONDITION_DIM)
    assert torch.count_nonzero(
        conditions[:, : STYLE_CONTROL_SLICE.start]
    ).item() == 0
    assert torch.equal(conditions[0, STYLE_CONTROL_SLICE], predicted[0])
    assert torch.equal(conditions[2, STYLE_CONTROL_SLICE], predicted[2])
    assert torch.equal(conditions[0, TEXT_LATENT_SLICE], latents[0])
    assert torch.equal(conditions[2, TEXT_LATENT_SLICE], latents[2])
    assert torch.count_nonzero(conditions[1]).item() == 0


def test_condition_dropout_sampling_is_seeded_and_mask_application_is_exact():
    first_generator = torch.Generator().manual_seed(42)
    second_generator = torch.Generator().manual_seed(42)
    first = sample_condition_keep_mask(
        100, 0.3, device="cpu", generator=first_generator
    )
    second = sample_condition_keep_mask(
        100, 0.3, device="cpu", generator=second_generator
    )
    assert torch.equal(first, second)
    assert first.any() and not first.all()
    assert sample_condition_keep_mask(4, 0.0, device="cpu").all()
    assert not sample_condition_keep_mask(4, 1.0, device="cpu").any()

    conditions = torch.ones(3, 264)
    masked = apply_condition_keep_mask(
        conditions, torch.tensor([True, False, True])
    )
    assert torch.equal(masked[0], conditions[0])
    assert torch.count_nonzero(masked[1]).item() == 0
    assert torch.equal(masked[2], conditions[2])
    assert torch.equal(conditions, torch.ones_like(conditions))


def test_style_losses_respect_explicit_supervision_mask():
    predicted = torch.tensor(
        [[0.0, 0.0, 0.0], [100.0, 100.0, 100.0], [2.0, 2.0, 2.0]]
    )
    target = torch.tensor(
        [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    mask = torch.tensor([True, False, True])
    assert torch.equal(
        masked_style_control_mse(predicted, target, mask),
        torch.tensor(2.5),
    )
    assert torch.equal(
        masked_style_control_smooth_l1(
            predicted, target, mask, beta=1.0
        ),
        torch.tensor(1.0),
    )
    empty = torch.zeros(3, dtype=torch.bool)
    assert torch.equal(
        masked_style_control_mse(predicted, target, empty),
        torch.tensor(0.0),
    )


def test_rolled_shuffled_conditions_is_deterministic_and_never_identity():
    conditions = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    expected = torch.stack((conditions[2], conditions[0], conditions[1]))
    assert torch.equal(
        rolled_shuffled_conditions(conditions, shift=1), expected
    )
    assert torch.equal(
        rolled_shuffled_conditions(conditions, shift=4), expected
    )
    with pytest.raises(ValueError, match="itself"):
        rolled_shuffled_conditions(conditions, shift=3)
    with pytest.raises(ValueError, match="at least two"):
        rolled_shuffled_conditions(conditions[:1])


def test_masked_per_example_flow_mse_supports_frame_and_element_masks():
    predictions = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 4.0], [6.0, 8.0]],
        ]
    )
    targets = torch.zeros_like(predictions)
    frame_mask = torch.tensor([[True, False], [True, True]])
    result = masked_per_example_flow_mse(
        predictions, targets, frame_mask
    )
    assert torch.allclose(
        result,
        torch.tensor([(1.0 + 4.0) / 2.0, (4.0 + 16.0 + 36.0 + 64.0) / 4.0]),
    )
    element_mask = torch.tensor(
        [
            [[True, False], [False, False]],
            [[False, True], [True, False]],
        ]
    )
    assert torch.allclose(
        masked_per_example_flow_mse(predictions, targets, element_mask),
        torch.tensor([1.0, (16.0 + 36.0) / 2.0]),
    )


def test_aligned_vs_rolled_shuffled_hinge_reports_gap_and_backpropagates():
    targets = torch.zeros(2, 2, 1)
    aligned = torch.zeros_like(targets, requires_grad=True)
    shuffled = torch.tensor(
        [[[1.0], [1.0]], [[0.1], [0.1]]], requires_grad=True
    )
    observed = torch.ones(2, 2, dtype=torch.bool)
    result = aligned_vs_rolled_shuffled_hinge_loss(
        aligned,
        shuffled,
        targets,
        observed,
        margin=0.25,
    )
    assert torch.allclose(
        result["aligned_flow_mse_per_example"], torch.zeros(2)
    )
    assert torch.allclose(
        result["rolled_shuffled_flow_mse_per_example"],
        torch.tensor([1.0, 0.01]),
    )
    assert torch.allclose(
        result["hinge_per_example"], torch.tensor([0.0, 0.24])
    )
    assert torch.equal(
        result["ranking_satisfied_fraction"], torch.tensor(0.5)
    )
    result["loss"].backward()
    assert shuffled.grad is not None
    assert torch.count_nonzero(shuffled.grad[0]).item() == 0
    assert torch.count_nonzero(shuffled.grad[1]).item() > 0
