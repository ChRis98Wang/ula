"""Pure training contracts for Hanyang partial-18D supervision.

Hanyang retargeting observes joint dimensions with different confidence and
does not observe four wrist rotations or head yaw at all.  Confidence is a
loss weight, not an action-amplitude multiplier: callers must not multiply
actions, targets, flow states, or noise by these floating-point weights.
Boolean ``weight > 0`` presence masks may be used to keep wholly missing
dimensions out of model inputs, while the floating-point values belong only
in weighted loss numerator/denominator calculations.

This module is deliberately independent of the existing BEAT2 trainers.  It
uses only PyTorch and the Python standard library so an expanded-data trainer
can opt into these contracts without changing the legacy BEAT2 numerical path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
import re
from numbers import Real
import struct
from typing import Any, NamedTuple

import torch


HANYANG_OBSERVATION_WEIGHT_CONTRACT = (
    "hanyang_partial_18d_observation_weight_v1"
)
HANYANG_ADMISSION_CONTRACT = "hanyang_expanded_training_admission_v1"
ACTION_DIM_18D = 18

HANYANG_ACTION_ORDER_18D = (
    "joint_pelvisYaw",
    "joint_pelvisPitch",
    "joint_pelvisRoll",
    "joint_lShoulderPitch",
    "joint_lShoulderRoll",
    "joint_lShoulderYaw",
    "joint_lElbow",
    "joint_lWristRoll",
    "joint_lWristPitch",
    "joint_rShoulderPitch",
    "joint_rShoulderRoll",
    "joint_rShoulderYaw",
    "joint_rElbow",
    "joint_rWristRoll",
    "joint_rWristPitch",
    "head_roll_joint",
    "head_pitch_joint",
    "head_yaw_joint",
)
PERMANENTLY_UNOBSERVED_DOF_NAMES = (
    "joint_lWristRoll",
    "joint_lWristPitch",
    "joint_rWristRoll",
    "joint_rWristPitch",
    "head_yaw_joint",
)
PERMANENTLY_UNOBSERVED_DOF_INDICES = tuple(
    HANYANG_ACTION_ORDER_18D.index(name)
    for name in PERMANENTLY_UNOBSERVED_DOF_NAMES
)

HANYANG_P7_ORDER = (
    "happy",
    "sad",
    "surprise",
    "angry",
    "disgust",
    "fear",
    "neutral",
)
HANYANG_Q2_ORDER = ("neutral", "non_neutral")
HANYANG_Q6_ORDER = (
    "neutral",
    "sad",
    "happy",
    "angry",
    "surprise",
    "fear",
)
EMOTION_CONDITION_MIN_INTENDED_SHARE = 0.70
PERMANENTLY_DISABLED_CONDITION_LANES = (
    "group54",
    "style",
    "duration",
    "semantic",
)

_FORBIDDEN_DATASET_TOKEN = "kimodo"
_P7_INDEX = {name: index for index, name in enumerate(HANYANG_P7_ORDER)}


class DerivativeObservationWeights(NamedTuple):
    """Aligned observation weights for first, second, and third differences."""

    velocity: torch.Tensor
    acceleration: torch.Tensor
    jerk: torch.Tensor


class HanyangEmotionTargets(NamedTuple):
    """V7-compatible hierarchical targets derived from Hanyang rater P7."""

    q2: torch.Tensor
    q6: torch.Tensor
    disgust_mass: torch.Tensor
    six_class_supervision_weight: torch.Tensor


def _require_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be bool")
    return value


def _normalized_dataset_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def reject_kimodo_strings(
    payload: object,
    *,
    context: str = "hanyang_expanded_training",
) -> None:
    """Reject every string reference to the permanently forbidden dataset.

    Mapping keys, values, nested sequences, and ``os.PathLike`` objects are
    inspected.  Separators and capitalization cannot hide the token.
    """

    def walk(value: object, field: str) -> None:
        if isinstance(value, str):
            if _FORBIDDEN_DATASET_TOKEN in _normalized_dataset_token(value):
                raise ValueError(
                    f"{field} contains a permanently forbidden dataset token"
                )
            return
        if isinstance(value, os.PathLike):
            walk(os.fspath(value), field)
            return
        if isinstance(value, Mapping):
            for key, child in value.items():
                walk(key, f"{field}.<key>")
                walk(child, f"{field}.{key}")
            return
        if isinstance(value, Sequence) and not isinstance(
            value, (bytes, bytearray)
        ):
            for index, child in enumerate(value):
                walk(child, f"{field}[{index}]")

    walk(payload, context)


def validate_observation_weight(
    observation_weight: torch.Tensor,
    *,
    frame_valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Validate and return a Hanyang observation weight ``[B,T,18]``.

    Values must be finite floating-point numbers in ``[0, 1]``.  The four
    wrist-orientation dimensions and head yaw are required to be exactly zero.
    When supplied, ``frame_valid_mask`` must be boolean ``[B,T]`` and every
    padded frame must have zero weight in every dimension.
    """

    if not isinstance(observation_weight, torch.Tensor):
        raise TypeError("observation_weight must be a torch.Tensor")
    if not observation_weight.is_floating_point():
        raise TypeError("observation_weight must be floating point")
    if observation_weight.ndim != 3:
        raise ValueError("observation_weight must have shape [B,T,18]")
    batch, frames, action_dim = observation_weight.shape
    if batch <= 0 or frames <= 0 or action_dim != ACTION_DIM_18D:
        raise ValueError("observation_weight must have shape [B,T,18]")
    if not bool(torch.isfinite(observation_weight).all().item()):
        raise ValueError("observation_weight must contain only finite values")
    if bool(
        ((observation_weight < 0) | (observation_weight > 1)).any().item()
    ):
        raise ValueError("observation_weight values must lie in [0,1]")

    missing_index = torch.tensor(
        PERMANENTLY_UNOBSERVED_DOF_INDICES,
        dtype=torch.long,
        device=observation_weight.device,
    )
    permanently_missing = observation_weight.index_select(-1, missing_index)
    if int(torch.count_nonzero(permanently_missing).item()) != 0:
        names = ", ".join(PERMANENTLY_UNOBSERVED_DOF_NAMES)
        raise ValueError(
            "permanently unobserved Hanyang DOFs must be exactly zero: "
            f"{names}"
        )

    if frame_valid_mask is not None:
        if not isinstance(frame_valid_mask, torch.Tensor):
            raise TypeError("frame_valid_mask must be a torch.Tensor")
        if frame_valid_mask.dtype is not torch.bool:
            raise TypeError("frame_valid_mask must have dtype torch.bool")
        if frame_valid_mask.device != observation_weight.device:
            raise ValueError(
                "frame_valid_mask and observation_weight must share a device"
            )
        if frame_valid_mask.shape != (batch, frames):
            raise ValueError("frame_valid_mask must have shape [B,T]")
        padded_weight = observation_weight.masked_select(
            ~frame_valid_mask.unsqueeze(-1).expand_as(observation_weight)
        )
        if int(torch.count_nonzero(padded_weight).item()) != 0:
            raise ValueError(
                "padded frames must have exactly zero observation weight"
            )
    return observation_weight


def observation_weight_sha256(observation_weight: torch.Tensor) -> str:
    """Return a deterministic hash of a validated observation-weight tensor."""

    weight = validate_observation_weight(observation_weight)
    canonical = weight.detach().cpu().contiguous()
    header = {
        "contract": HANYANG_OBSERVATION_WEIGHT_CONTRACT,
        "dtype": str(canonical.dtype),
        "shape": list(canonical.shape),
    }
    digest = hashlib.sha256(
        json.dumps(
            header,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    flat = canonical.reshape(-1)
    format_code = {
        torch.float16: "e",
        torch.float32: "f",
        torch.float64: "d",
    }.get(canonical.dtype)
    if format_code is None:
        # bfloat16 has no struct code.  Hash its exact 16-bit representation.
        if canonical.dtype is not torch.bfloat16:
            raise TypeError(
                f"unsupported observation_weight dtype: {canonical.dtype}"
            )
        bit_pattern = canonical.view(torch.int16).reshape(-1)
        for chunk in bit_pattern.split(4096):
            values = chunk.tolist()
            digest.update(struct.pack(f"<{len(values)}h", *values))
        return digest.hexdigest()
    for chunk in flat.split(4096):
        values = chunk.tolist()
        digest.update(struct.pack(f"<{len(values)}{format_code}", *values))
    return digest.hexdigest()


def _adjacent_minimum(
    observation_weight: torch.Tensor,
    *,
    frame_count: int,
) -> torch.Tensor:
    frames = observation_weight.shape[1]
    if frames < frame_count:
        return observation_weight[:, :0, :]
    aligned = tuple(
        observation_weight[
            :, offset : frames - (frame_count - 1 - offset), :
        ]
        for offset in range(frame_count)
    )
    return torch.stack(aligned, dim=0).amin(dim=0)


def derivative_observation_weights(
    observation_weight: torch.Tensor,
) -> DerivativeObservationWeights:
    """Derive velocity/acceleration/jerk weights by adjacent-frame minima.

    Velocity uses the minimum over each adjacent pair, acceleration over each
    adjacent triple, and jerk over each adjacent group of four frames.  Output
    shapes are respectively ``[B,T-1,18]``, ``[B,T-2,18]``, and
    ``[B,T-3,18]`` (clamped to zero temporal length when unavailable).
    """

    weight = validate_observation_weight(observation_weight)
    return DerivativeObservationWeights(
        velocity=_adjacent_minimum(weight, frame_count=2),
        acceleration=_adjacent_minimum(weight, frame_count=3),
        jerk=_adjacent_minimum(weight, frame_count=4),
    )


def weighted_masked_mean(
    values: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Return ``sum(values * weight) / sum(weight)`` with a hard zero mask.

    Shapes must match exactly.  Values at zero-weight positions may be
    arbitrary (including non-finite): they cannot alter the result and receive
    exactly zero gradient.  An entirely zero-weight input returns a
    differentiable scalar zero rather than inventing supervision.
    """

    if not isinstance(values, torch.Tensor) or not isinstance(
        weight, torch.Tensor
    ):
        raise TypeError("values and weight must be torch.Tensor instances")
    if not values.is_floating_point() or not weight.is_floating_point():
        raise TypeError("values and weight must be floating point")
    if values.shape != weight.shape:
        raise ValueError("values and weight must have exactly matching shapes")
    if values.device != weight.device:
        raise ValueError("values and weight must share a device")
    if not bool(torch.isfinite(weight).all().item()):
        raise ValueError("weight must contain only finite values")
    if bool(((weight < 0) | (weight > 1)).any().item()):
        raise ValueError("weight values must lie in [0,1]")

    active = weight > 0
    active_values = values.masked_select(active)
    if not bool(torch.isfinite(active_values).all().item()):
        raise ValueError("positive-weight values must be finite")
    safe_values = torch.where(active, values, torch.zeros_like(values))
    numerator = (safe_values * weight).sum()
    denominator = weight.sum()
    tiny = torch.finfo(numerator.dtype).tiny
    return numerator / denominator.clamp_min(tiny)


def hanyang_p7_to_hierarchy_targets(
    p7: torch.Tensor,
    *,
    probability_atol: float = 1e-6,
) -> HanyangEmotionTargets:
    """Convert Hanyang P7 rater probabilities to V7 Q2/Q6 targets.

    Input order is ``[happy,sad,surprise,angry,disgust,fear,neutral]``.
    Q2 order is ``[neutral,non_neutral]``.  Q6 order is
    ``[neutral,sad,happy,angry,surprise,fear]`` and is normalized after
    removing disgust.  Its supervision weight is the retained probability
    mass ``1 - disgust_mass``.  A pure-disgust row therefore has zero Q6 and
    zero Q6 supervision weight; it is never hard-mapped to another class.
    """

    if not isinstance(p7, torch.Tensor):
        raise TypeError("p7 must be a torch.Tensor")
    if not p7.is_floating_point():
        raise TypeError("p7 must be floating point")
    if p7.ndim < 1 or p7.shape[-1] != len(HANYANG_P7_ORDER):
        raise ValueError("p7 must have shape [...,7]")
    if p7.numel() == 0:
        raise ValueError("p7 must contain at least one distribution")
    if not bool(torch.isfinite(p7).all().item()):
        raise ValueError("p7 must contain only finite probabilities")
    if bool(((p7 < 0) | (p7 > 1)).any().item()):
        raise ValueError("p7 probabilities must lie in [0,1]")
    tolerance = float(probability_atol)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("probability_atol must be finite and non-negative")
    probability_sum = p7.sum(dim=-1)
    if not bool(
        torch.allclose(
            probability_sum,
            torch.ones_like(probability_sum),
            rtol=0.0,
            atol=tolerance,
        )
    ):
        raise ValueError("p7 probabilities must sum to one")

    neutral = p7[..., _P7_INDEX["neutral"]]
    q2 = torch.stack((neutral, 1.0 - neutral), dim=-1)

    retained = torch.stack(
        tuple(p7[..., _P7_INDEX[name]] for name in HANYANG_Q6_ORDER),
        dim=-1,
    )
    disgust_mass = p7[..., _P7_INDEX["disgust"]]
    six_class_supervision_weight = retained.sum(dim=-1)
    denominator = six_class_supervision_weight.unsqueeze(-1)
    q6 = torch.where(
        denominator > 0,
        retained / denominator.clamp_min(torch.finfo(p7.dtype).tiny),
        torch.zeros_like(retained),
    )
    return HanyangEmotionTargets(
        q2=q2,
        q6=q6,
        disgust_mass=disgust_mass,
        six_class_supervision_weight=six_class_supervision_weight,
    )


def hanyang_training_admission(
    *,
    qc_pass: bool,
    rater_coverage_pass: bool,
    intended_majority_agrees: bool,
    intended_share: float,
    lineage: object = (),
) -> dict[str, Any]:
    """Build the fail-closed Hanyang expanded-training admission flags.

    Every QC-passing row may enter the unconditional motion lane.  Emotion
    conditioning additionally requires adequate rater coverage, agreement of
    the intended label with the rater majority, and intended share >= 0.70.
    Hanyang is permanently ineligible for the 54-group, style, duration, and
    semantic conditioning lanes.
    """

    reject_kimodo_strings(lineage, context="hanyang_training_admission.lineage")
    qc = _require_bool(qc_pass, name="qc_pass")
    coverage = _require_bool(
        rater_coverage_pass, name="rater_coverage_pass"
    )
    agreement = _require_bool(
        intended_majority_agrees, name="intended_majority_agrees"
    )
    if isinstance(intended_share, bool) or not isinstance(
        intended_share, Real
    ):
        raise TypeError("intended_share must be a real number")
    share = float(intended_share)
    if not math.isfinite(share) or not 0.0 <= share <= 1.0:
        raise ValueError("intended_share must lie in [0,1]")
    emotion_eligible = bool(
        qc
        and coverage
        and agreement
        and share >= EMOTION_CONDITION_MIN_INTENDED_SHARE
    )
    return {
        "contract": HANYANG_ADMISSION_CONTRACT,
        "qc_pass": qc,
        "only_unconditional_motion": bool(qc and not emotion_eligible),
        "unconditional_motion_eligible": qc,
        "emotion_condition_eligible": emotion_eligible,
        "emotion_condition_min_intended_share": (
            EMOTION_CONDITION_MIN_INTENDED_SHARE
        ),
        "group54_condition_eligible": False,
        "style_condition_eligible": False,
        "duration_condition_eligible": False,
        "semantic_condition_eligible": False,
    }


__all__ = [
    "ACTION_DIM_18D",
    "DerivativeObservationWeights",
    "EMOTION_CONDITION_MIN_INTENDED_SHARE",
    "HANYANG_ACTION_ORDER_18D",
    "HANYANG_ADMISSION_CONTRACT",
    "HANYANG_OBSERVATION_WEIGHT_CONTRACT",
    "HANYANG_P7_ORDER",
    "HANYANG_Q2_ORDER",
    "HANYANG_Q6_ORDER",
    "HanyangEmotionTargets",
    "PERMANENTLY_DISABLED_CONDITION_LANES",
    "PERMANENTLY_UNOBSERVED_DOF_INDICES",
    "PERMANENTLY_UNOBSERVED_DOF_NAMES",
    "derivative_observation_weights",
    "hanyang_p7_to_hierarchy_targets",
    "hanyang_training_admission",
    "observation_weight_sha256",
    "reject_kimodo_strings",
    "validate_observation_weight",
    "weighted_masked_mean",
]
