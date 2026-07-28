"""Leakage-resistant text, style, and condition-control training utilities.

The primary generator path is deliberately simple:

* a frozen 128D text latent is mapped to three style controls by
  :class:`QwenStyleHead`;
* the predicted style controls and the text latent are assembled into a fresh
  264D condition vector; and
* an explicit boolean mask makes unconditional rows exactly zero.

Per-trajectory style controls are supervision targets only.  They are never
accepted by :func:`assemble_text_style_conditions`.

The train-group mean helper in this module is an audited diagnostic baseline,
not a generator-input policy.  It uses only the fixed training split and
excludes every training target row from its own assigned baseline.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from numbers import Integral
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


CONDITION_DIM = 264
STYLE_CONTROL_DIM = 3
TEXT_LATENT_DIM = 128
STYLE_CONTROL_SLICE = slice(133, 136)
TEXT_LATENT_SLICE = slice(136, 264)
FIXED_SPLIT_NAMES = ("train", "validation", "test")
STYLE_HEAD_SCHEMA_VERSION = 1
TRAIN_GROUP_BASELINE_SCHEMA_VERSION = 1


def _require_floating_tensor(
    value: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> None:
    if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
        raise TypeError(f"{name} must be a floating-point torch tensor")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")


def _require_boolean_batch_mask(
    value: torch.Tensor,
    *,
    batch_size: int,
    name: str,
    device: torch.device,
) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.bool
        or value.shape != (batch_size,)
    ):
        raise ValueError(f"{name} must be boolean [{batch_size}]")
    if value.device != device:
        raise ValueError(f"{name} must be on the same device as its inputs")


class QwenStyleHead(nn.Module):
    """Predict three expected motion-style controls from a 128D text latent.

    ``conditioning_mask`` is required rather than inferred.  False rows use
    ``torch.where`` to produce exact zeros even when the network has biases.
    The output projection starts at zero by default so adding the head to an
    existing motion prior begins at its neutral-style operating point.
    """

    def __init__(
        self,
        text_latent_dim: int = TEXT_LATENT_DIM,
        hidden_dim: int = 128,
        style_dim: int = STYLE_CONTROL_DIM,
        *,
        zero_initialize_output: bool = True,
    ) -> None:
        super().__init__()
        if int(text_latent_dim) <= 0:
            raise ValueError("text_latent_dim must be positive")
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if int(style_dim) != STYLE_CONTROL_DIM:
            raise ValueError(f"style_dim must be {STYLE_CONTROL_DIM}")
        self.text_latent_dim = int(text_latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.style_dim = int(style_dim)
        self.zero_initialize_output = bool(zero_initialize_output)
        self.input_projection = nn.Linear(self.text_latent_dim, self.hidden_dim)
        self.activation = nn.SiLU()
        self.output_projection = nn.Linear(self.hidden_dim, self.style_dim)
        if self.zero_initialize_output:
            nn.init.zeros_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        text_latents: torch.Tensor,
        conditioning_mask: torch.Tensor,
    ) -> torch.Tensor:
        _require_floating_tensor(text_latents, name="text_latents")
        if (
            text_latents.ndim != 2
            or text_latents.shape[1] != self.text_latent_dim
        ):
            raise ValueError(
                "text_latents must have shape "
                f"[batch, {self.text_latent_dim}]"
            )
        _require_boolean_batch_mask(
            conditioning_mask,
            batch_size=text_latents.shape[0],
            name="conditioning_mask",
            device=text_latents.device,
        )
        predicted = self.output_projection(
            self.activation(self.input_projection(text_latents))
        )
        return torch.where(
            conditioning_mask[:, None],
            predicted,
            torch.zeros_like(predicted),
        )

    def architecture_config(self) -> dict[str, Any]:
        """Return the JSON-safe constructor contract stored with checkpoints."""
        return {
            "schema_version": STYLE_HEAD_SCHEMA_VERSION,
            "class_name": type(self).__name__,
            "text_latent_dim": self.text_latent_dim,
            "hidden_dim": self.hidden_dim,
            "style_dim": self.style_dim,
            "zero_initialize_output": self.zero_initialize_output,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "QwenStyleHead":
        """Recreate a head before loading its standard PyTorch state dict."""
        if not isinstance(config, Mapping):
            raise TypeError("style-head config must be a mapping")
        expected_keys = {
            "schema_version",
            "class_name",
            "text_latent_dim",
            "hidden_dim",
            "style_dim",
            "zero_initialize_output",
        }
        if set(config) != expected_keys:
            raise ValueError("style-head config fields changed")
        if (
            config.get("schema_version") != STYLE_HEAD_SCHEMA_VERSION
            or config.get("class_name") != cls.__name__
        ):
            raise ValueError("style-head config identity changed")
        return cls(
            text_latent_dim=int(config["text_latent_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            style_dim=int(config["style_dim"]),
            zero_initialize_output=bool(config["zero_initialize_output"]),
        )


def sample_condition_keep_mask(
    batch_size: int,
    drop_probability: float,
    *,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a deterministic-with-generator classifier-free keep mask."""
    batch_size = int(batch_size)
    probability = float(drop_probability)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("drop_probability must be finite and in [0, 1]")
    device = torch.device(device)
    if probability == 0.0:
        return torch.ones(batch_size, dtype=torch.bool, device=device)
    if probability == 1.0:
        return torch.zeros(batch_size, dtype=torch.bool, device=device)
    return (
        torch.rand(
            batch_size,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        >= probability
    )


def apply_condition_keep_mask(
    conditions: torch.Tensor,
    conditioning_mask: torch.Tensor,
) -> torch.Tensor:
    """Return a copy whose unconditional rows are exactly all-zero."""
    _require_floating_tensor(conditions, name="conditions")
    if conditions.ndim != 2:
        raise ValueError("conditions must have shape [batch, dimensions]")
    _require_boolean_batch_mask(
        conditioning_mask,
        batch_size=conditions.shape[0],
        name="conditioning_mask",
        device=conditions.device,
    )
    return torch.where(
        conditioning_mask[:, None],
        conditions,
        torch.zeros_like(conditions),
    )


def assemble_text_style_conditions(
    text_latents: torch.Tensor,
    predicted_style_controls: torch.Tensor,
    conditioning_mask: torch.Tensor,
) -> torch.Tensor:
    """Build fresh 264D generator conditions without accepting oracle styles."""
    _require_floating_tensor(text_latents, name="text_latents")
    if text_latents.ndim != 2 or text_latents.shape[1] != TEXT_LATENT_DIM:
        raise ValueError(
            f"text_latents must have shape [batch, {TEXT_LATENT_DIM}]"
        )
    _require_floating_tensor(
        predicted_style_controls,
        name="predicted_style_controls",
        shape=(text_latents.shape[0], STYLE_CONTROL_DIM),
    )
    if (
        predicted_style_controls.device != text_latents.device
        or predicted_style_controls.dtype != text_latents.dtype
    ):
        raise ValueError(
            "predicted_style_controls must share text_latents dtype and device"
        )
    _require_boolean_batch_mask(
        conditioning_mask,
        batch_size=text_latents.shape[0],
        name="conditioning_mask",
        device=text_latents.device,
    )
    prefix = text_latents.new_zeros(
        (text_latents.shape[0], STYLE_CONTROL_SLICE.start)
    )
    conditions = torch.cat(
        (prefix, predicted_style_controls, text_latents),
        dim=1,
    )
    if conditions.shape != (text_latents.shape[0], CONDITION_DIM):
        raise RuntimeError("internal 264D condition layout changed")
    return apply_condition_keep_mask(conditions, conditioning_mask)


def _validate_style_loss_inputs(
    predicted_style_controls: torch.Tensor,
    target_style_controls: torch.Tensor,
    supervision_mask: torch.Tensor | None,
) -> torch.Tensor:
    _require_floating_tensor(
        predicted_style_controls, name="predicted_style_controls"
    )
    if (
        predicted_style_controls.ndim != 2
        or predicted_style_controls.shape[1] != STYLE_CONTROL_DIM
    ):
        raise ValueError(
            "predicted_style_controls must have shape "
            f"[batch, {STYLE_CONTROL_DIM}]"
        )
    _require_floating_tensor(
        target_style_controls,
        name="target_style_controls",
        shape=tuple(predicted_style_controls.shape),
    )
    if (
        target_style_controls.device != predicted_style_controls.device
        or target_style_controls.dtype != predicted_style_controls.dtype
    ):
        raise ValueError(
            "target_style_controls must share predicted values' dtype and device"
        )
    if supervision_mask is None:
        mask = torch.ones(
            predicted_style_controls.shape[0],
            dtype=torch.bool,
            device=predicted_style_controls.device,
        )
    else:
        _require_boolean_batch_mask(
            supervision_mask,
            batch_size=predicted_style_controls.shape[0],
            name="supervision_mask",
            device=predicted_style_controls.device,
        )
        mask = supervision_mask
    return mask


def masked_style_control_mse(
    predicted_style_controls: torch.Tensor,
    target_style_controls: torch.Tensor,
    supervision_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean style regression error over explicitly supervised rows."""
    mask = _validate_style_loss_inputs(
        predicted_style_controls,
        target_style_controls,
        supervision_mask,
    )
    per_example = (
        predicted_style_controls - target_style_controls
    ).square().mean(dim=1)
    weights = mask.to(per_example.dtype)
    return (per_example * weights).sum() / weights.sum().clamp_min(1.0)


def masked_style_control_smooth_l1(
    predicted_style_controls: torch.Tensor,
    target_style_controls: torch.Tensor,
    supervision_mask: torch.Tensor | None = None,
    *,
    beta: float = 1.0,
) -> torch.Tensor:
    """Smooth-L1 style regression over explicitly supervised rows."""
    beta = float(beta)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and positive")
    mask = _validate_style_loss_inputs(
        predicted_style_controls,
        target_style_controls,
        supervision_mask,
    )
    per_example = F.smooth_l1_loss(
        predicted_style_controls,
        target_style_controls,
        reduction="none",
        beta=beta,
    ).mean(dim=1)
    weights = mask.to(per_example.dtype)
    return (per_example * weights).sum() / weights.sum().clamp_min(1.0)


def rolled_shuffled_conditions(
    conditions: torch.Tensor,
    *,
    shift: int = 1,
) -> torch.Tensor:
    """Deterministically roll a batch to create the ranking negative."""
    _require_floating_tensor(conditions, name="conditions")
    if conditions.ndim < 2:
        raise ValueError("conditions must have a batch and feature dimension")
    batch_size = int(conditions.shape[0])
    if batch_size < 2:
        raise ValueError("rolled condition negatives need at least two rows")
    if isinstance(shift, bool) or not isinstance(shift, Integral):
        raise TypeError("shift must be an integer")
    normalized_shift = int(shift) % batch_size
    if normalized_shift == 0:
        raise ValueError("shift must not map each row to itself")
    return torch.roll(conditions, shifts=normalized_shift, dims=0)


def _broadcast_observed_mask(
    observed_mask: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(observed_mask, torch.Tensor) or observed_mask.dtype != torch.bool:
        raise TypeError("observed_mask must be a boolean torch tensor")
    if observed_mask.device != values.device:
        raise ValueError("observed_mask must be on the same device as predictions")
    mask = observed_mask
    if mask.ndim == 1 and mask.shape == values.shape[:1]:
        mask = mask.reshape((values.shape[0],) + (1,) * (values.ndim - 1))
    elif (
        mask.ndim == values.ndim - 1
        and tuple(mask.shape) == tuple(values.shape[:-1])
    ):
        mask = mask.unsqueeze(-1)
    try:
        return torch.broadcast_to(mask, values.shape)
    except RuntimeError as exc:
        raise ValueError(
            "observed_mask is not broadcastable to prediction shape "
            f"{tuple(values.shape)}"
        ) from exc


def masked_per_example_flow_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    observed_mask: torch.Tensor,
) -> torch.Tensor:
    """Return one observed-element flow MSE for every batch row."""
    _require_floating_tensor(predictions, name="predictions")
    _require_floating_tensor(
        targets, name="targets", shape=tuple(predictions.shape)
    )
    if predictions.ndim < 2:
        raise ValueError("predictions must have a batch and feature dimension")
    if targets.device != predictions.device or targets.dtype != predictions.dtype:
        raise ValueError("targets must share predictions' dtype and device")
    mask = _broadcast_observed_mask(observed_mask, predictions)
    flattened_mask = mask.reshape(mask.shape[0], -1)
    counts = flattened_mask.sum(dim=1)
    squared = (predictions - targets).square()
    numerator = (squared * mask.to(squared.dtype)).reshape(
        squared.shape[0], -1
    ).sum(dim=1)
    return numerator / counts.to(numerator.dtype).clamp_min(1.0)


def aligned_vs_rolled_shuffled_hinge_loss(
    aligned_predictions: torch.Tensor,
    rolled_shuffled_predictions: torch.Tensor,
    targets: torch.Tensor,
    observed_mask: torch.Tensor,
    *,
    margin: float = 0.05,
    reduction: str = "mean",
) -> dict[str, torch.Tensor]:
    """Rank aligned text below a rolled-text negative by a flow-MSE margin."""
    margin = float(margin)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("margin must be finite and non-negative")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("reduction must be one of: mean, sum, none")
    aligned = masked_per_example_flow_mse(
        aligned_predictions, targets, observed_mask
    )
    shuffled = masked_per_example_flow_mse(
        rolled_shuffled_predictions, targets, observed_mask
    )
    hinge = F.relu(aligned - shuffled + margin)
    if reduction == "mean":
        loss = hinge.mean()
    elif reduction == "sum":
        loss = hinge.sum()
    else:
        loss = hinge
    gap = shuffled - aligned
    return {
        "loss": loss,
        "aligned_flow_mse": aligned.mean(),
        "rolled_shuffled_flow_mse": shuffled.mean(),
        "ranking_gap": gap.mean(),
        "ranking_satisfied_fraction": (gap >= margin).to(
            aligned.dtype
        ).mean(),
        "aligned_flow_mse_per_example": aligned,
        "rolled_shuffled_flow_mse_per_example": shuffled,
        "hinge_per_example": hinge,
    }


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _semantic_group_key(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("semantic group ids cannot be boolean")
    if isinstance(value, Integral):
        return f"integer:{int(value)}"
    if isinstance(value, (str, np.str_)):
        text = str(value)
        if not text:
            raise ValueError("semantic group ids cannot be empty")
        return "string:" + json.dumps(text, ensure_ascii=False)
    raise TypeError("semantic group ids must be strings or integers")


def _exclusive_means(
    values: np.ndarray,
    indices: Sequence[int],
) -> dict[int, np.ndarray]:
    ordered = np.asarray(list(indices), dtype=np.int64)
    if ordered.size < 2:
        raise ValueError("exclusive means require at least two source rows")
    selected = np.asarray(values[ordered], dtype=np.float64)
    prefix = np.zeros((selected.shape[0] + 1, selected.shape[1]), dtype=np.float64)
    suffix = np.zeros_like(prefix)
    prefix[1:] = np.cumsum(selected, axis=0, dtype=np.float64)
    suffix[:-1] = np.cumsum(selected[::-1], axis=0, dtype=np.float64)[::-1]
    sums_without_self = prefix[:-1] + suffix[1:]
    means = sums_without_self / float(selected.shape[0] - 1)
    return {
        int(row): means[position].astype(np.float32)
        for position, row in enumerate(ordered.tolist())
    }


def build_train_group_mean_style_baseline(
    source_conditions: np.ndarray,
    target_style_controls: np.ndarray,
    semantic_group_ids: Sequence[Any],
    fixed_split_assignments: Sequence[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build an audited train-only, leave-one-out diagnostic baseline.

    This function is intentionally not the generator-input policy.  The source
    style slice must be exactly zero.  Evaluation targets are validated but
    never read into any statistic.  A training row receives the mean of the
    other rows in its semantic group; a singleton group falls back to the
    global training mean with that row excluded.  Non-training rows receive the
    full training-group mean, or the full global training mean for unseen
    groups.
    """
    source = np.asarray(source_conditions)
    targets = np.asarray(target_style_controls)
    groups = np.asarray(semantic_group_ids, dtype=object)
    splits = np.asarray(fixed_split_assignments)
    if source.ndim != 2 or source.shape[1] != CONDITION_DIM:
        raise ValueError(
            f"source_conditions must have shape [rows, {CONDITION_DIM}]"
        )
    row_count = int(source.shape[0])
    if targets.shape != (row_count, STYLE_CONTROL_DIM):
        raise ValueError(
            f"target_style_controls must have shape [rows, {STYLE_CONTROL_DIM}]"
        )
    if groups.shape != (row_count,):
        raise ValueError("semantic_group_ids must have shape [rows]")
    if splits.shape != (row_count,):
        raise ValueError("fixed_split_assignments must have shape [rows]")
    try:
        source = np.asarray(source, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TypeError("conditions and style targets must be numeric") from exc
    if not np.isfinite(source).all():
        raise ValueError("source_conditions must contain only finite values")
    if not np.isfinite(targets).all():
        raise ValueError(
            "target_style_controls must contain only finite values"
        )
    if np.any(source[:, STYLE_CONTROL_SLICE] != 0.0):
        raise ValueError("source condition style slice must be exactly zero")

    group_keys = np.asarray(
        [_semantic_group_key(value) for value in groups.tolist()],
        dtype=object,
    )
    split_names = np.asarray([str(value) for value in splits.tolist()])
    unknown_splits = sorted(set(split_names) - set(FIXED_SPLIT_NAMES))
    if unknown_splits:
        raise ValueError(
            f"fixed split contains unknown assignments: {unknown_splits}"
        )
    train_indices = np.flatnonzero(split_names == "train").astype(np.int64)
    if train_indices.size < 2:
        raise ValueError(
            "strict target-row exclusion requires at least two training rows"
        )

    train_members: defaultdict[str, list[int]] = defaultdict(list)
    for row in train_indices.tolist():
        train_members[str(group_keys[row])].append(int(row))
    global_exclusive = _exclusive_means(targets, train_indices.tolist())
    group_exclusive: dict[int, np.ndarray] = {}
    group_means: dict[str, np.ndarray] = {}
    for key, members in train_members.items():
        group_means[key] = np.asarray(
            targets[np.asarray(members, dtype=np.int64)].mean(
                axis=0, dtype=np.float64
            ),
            dtype=np.float32,
        )
        if len(members) > 1:
            group_exclusive.update(_exclusive_means(targets, members))
    global_mean = np.asarray(
        targets[train_indices].mean(axis=0, dtype=np.float64),
        dtype=np.float32,
    )

    assigned = np.empty((row_count, STYLE_CONTROL_DIM), dtype=np.float32)
    assignment_modes: Counter[str] = Counter()
    for row in range(row_count):
        key = str(group_keys[row])
        if split_names[row] == "train":
            if row in group_exclusive:
                assigned[row] = group_exclusive[row]
                assignment_modes["train_leave_one_out_group"] += 1
            else:
                assigned[row] = global_exclusive[row]
                assignment_modes["train_leave_one_out_global_fallback"] += 1
        elif key in group_means:
            assigned[row] = group_means[key]
            assignment_modes["evaluation_train_group_mean"] += 1
        else:
            assigned[row] = global_mean
            assignment_modes["evaluation_global_train_fallback"] += 1

    transformed = np.ascontiguousarray(source.copy())
    transformed[:, STYLE_CONTROL_SLICE] = assigned
    non_style_indices = np.r_[
        0 : STYLE_CONTROL_SLICE.start,
        STYLE_CONTROL_SLICE.stop : CONDITION_DIM,
    ]
    if not np.array_equal(
        transformed[:, non_style_indices],
        source[:, non_style_indices],
    ):
        raise RuntimeError("non-style condition columns changed")
    train_group_records = [
        {
            "semantic_group_key": key,
            "training_row_count": len(train_members[key]),
            "style_controls": group_means[key].tolist(),
        }
        for key in sorted(train_members)
    ]
    split_counts = Counter(split_names.tolist())
    receipt = {
        "schema_version": TRAIN_GROUP_BASELINE_SCHEMA_VERSION,
        "artifact_kind": "beat2_train_group_style_diagnostic_baseline",
        "generator_input_policy": False,
        "baseline_only_not_generator_input": True,
        "row_count": row_count,
        "condition_dim": CONDITION_DIM,
        "style_control_slice": [
            STYLE_CONTROL_SLICE.start,
            STYLE_CONTROL_SLICE.stop,
        ],
        "fixed_split_names": list(FIXED_SPLIT_NAMES),
        "fixed_split_counts": {
            name: int(split_counts.get(name, 0)) for name in FIXED_SPLIT_NAMES
        },
        "fit_split": "train",
        "fit_row_count": int(train_indices.size),
        "fit_semantic_group_count": len(train_members),
        "source_rows_used_by_split": {
            name: int(train_indices.size) if name == "train" else 0
            for name in FIXED_SPLIT_NAMES
        },
        "target_row_excluded_from_own_training_assignment": True,
        "evaluation_target_style_controls_used": False,
        "source_style_slice_zero_validated": True,
        "non_style_columns_preserved_exactly": True,
        "assignment_mode_counts": {
            key: int(value) for key, value in sorted(assignment_modes.items())
        },
        "global_train_style_controls": global_mean.tolist(),
        "train_group_means": train_group_records,
        "train_source_style_controls_sha256": _array_sha256(
            targets[train_indices]
        ),
        "source_conditions_sha256": _array_sha256(source),
        "assigned_style_controls_sha256": _array_sha256(assigned),
        "transformed_conditions_sha256": _array_sha256(transformed),
    }
    return transformed, receipt


__all__ = [
    "CONDITION_DIM",
    "FIXED_SPLIT_NAMES",
    "QwenStyleHead",
    "STYLE_CONTROL_DIM",
    "STYLE_CONTROL_SLICE",
    "TEXT_LATENT_DIM",
    "TEXT_LATENT_SLICE",
    "aligned_vs_rolled_shuffled_hinge_loss",
    "apply_condition_keep_mask",
    "assemble_text_style_conditions",
    "build_train_group_mean_style_baseline",
    "masked_per_example_flow_mse",
    "masked_style_control_mse",
    "masked_style_control_smooth_l1",
    "rolled_shuffled_conditions",
    "sample_condition_keep_mask",
]
