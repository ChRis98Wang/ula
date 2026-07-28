"""Differentiable BEAT2-only semantic/perceptual loss.

This module is deliberately independent of the generator trainer.  It mirrors
the 742D descriptor used by ``train_beat2_qwen_motion_alignment.py``, applies
the train-only normalization stored with that experiment, and passes the
descriptor through its frozen MotionEncoder.  The resulting 128D unit vector
can be compared directly with the aligned frozen-Qwen condition cache.

The input to :class:`Beat2SemanticPerceptualLoss` is the generator's normalized
18D output.  Padding is removed before any temporal statistic is computed, and
``durations_sec`` is interpreted as the physical sample span ``(N - 1) / fps``.
No external motion data or Kimodo artifact is accepted by the artifact loader.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


BEAT2_DATA_POLICY = "beat2_only_no_external_motion_dataset_v1"
MOTION_ENCODER_ARTIFACT_KIND = "beat2_random_init_motion_encoder_v1"
DESCRIPTOR_CACHE_ARTIFACT_KIND = "beat2_18d_motion_descriptor_cache_v1"
QWEN_CONDITION_ARTIFACT_KIND = "beat2_qwen_motion_latent_condition_cache_v1"
PROTOTYPE_BANK_ARTIFACT_KIND = "beat2_train_qwen_group_prototype_bank_v1"
ACTION_DIM = 18
LATENT_DIM = 128
DEFAULT_FPS = 30.0
DEFAULT_PHASE_SAMPLES = 24
DESCRIPTOR_CLIP = 20.0
RMS_SQUARED_FLOOR = 1e-12

BEAT2_JOINT_NAMES = (
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


def descriptor_dim(phase_samples: int = DEFAULT_PHASE_SAMPLES) -> int:
    """Return the width of the complete BEAT2 motion descriptor."""

    phase_samples = int(phase_samples)
    if phase_samples < 4:
        raise ValueError("phase_samples must be at least four")
    # phase + eight position + six velocity + three acceleration statistics,
    # all per joint, followed by four trajectory-global values.
    return ACTION_DIM * (phase_samples + 8 + 6 + 3) + 4


def _prefix_frame_mask(
    actions: torch.Tensor,
    frame_mask: torch.Tensor | None,
    *,
    validate: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, frame_count, _ = actions.shape
    if frame_mask is None:
        mask = torch.ones(
            (batch_size, frame_count), dtype=torch.bool, device=actions.device
        )
    else:
        mask = torch.as_tensor(frame_mask, device=actions.device)
        if mask.shape != (batch_size, frame_count):
            raise ValueError(
                "frame_mask must have shape [batch, frames], got "
                f"{tuple(mask.shape)}"
            )
        if mask.dtype != torch.bool:
            raise ValueError("frame_mask must be boolean")
    lengths = mask.sum(dim=1, dtype=torch.int64)
    if validate:
        if bool(torch.any(lengths < 3).item()):
            raise ValueError("every BEAT2 trajectory must contain at least three frames")
        expected = (
            torch.arange(frame_count, device=actions.device)[None, :]
            < lengths[:, None]
        )
        if not torch.equal(mask, expected):
            raise ValueError(
                "frame_mask must be a contiguous valid prefix followed by padding"
            )
    return mask, lengths


def _duration_tensor(
    lengths: torch.Tensor,
    durations_sec: torch.Tensor | None,
    *,
    fps: float,
    dtype: torch.dtype,
    device: torch.device,
    validate: bool,
) -> torch.Tensor:
    if not math.isfinite(float(fps)) or float(fps) <= 0:
        raise ValueError("fps must be finite and positive")
    if durations_sec is None:
        durations = (lengths.to(dtype=dtype) - 1.0) / float(fps)
    else:
        durations = torch.as_tensor(durations_sec, dtype=dtype, device=device)
        if durations.shape != lengths.shape:
            raise ValueError(
                f"durations_sec must have shape {tuple(lengths.shape)}, got "
                f"{tuple(durations.shape)}"
            )
    if validate and bool(
        torch.any(~torch.isfinite(durations) | (durations <= 0)).item()
    ):
        raise ValueError("durations_sec must contain finite positive sample spans")
    return durations


def _single_motion_descriptor(
    actions: torch.Tensor,
    *,
    duration_sec: torch.Tensor,
    phase_samples: int,
) -> torch.Tensor:
    """Build one full descriptor while preserving gradients to ``actions``."""

    frame_count = int(actions.shape[0])
    phase = F.interpolate(
        actions.transpose(0, 1).unsqueeze(0),
        size=int(phase_samples),
        mode="linear",
        align_corners=True,
    ).squeeze(0).transpose(0, 1)

    dt = duration_sec / float(frame_count - 1)
    difference = torch.diff(actions, dim=0)
    velocity = difference / dt
    acceleration = torch.diff(velocity, dim=0) / dt

    minimum = actions.amin(dim=0)
    maximum = actions.amax(dim=0)
    position_features = torch.stack(
        (
            actions.mean(dim=0),
            actions.std(dim=0, correction=0),
            minimum,
            maximum,
            actions[0],
            actions[-1],
            maximum - minimum,
            actions[-1] - actions[0],
        ),
        dim=0,
    )

    abs_velocity = velocity.abs()
    velocity_features = torch.stack(
        (
            velocity.mean(dim=0),
            velocity.std(dim=0, correction=0),
            velocity.square().mean(dim=0).clamp_min(RMS_SQUARED_FLOOR).sqrt(),
            abs_velocity.mean(dim=0),
            torch.quantile(
                abs_velocity, 0.95, dim=0, interpolation="linear"
            ),
            abs_velocity.amax(dim=0),
        ),
        dim=0,
    )

    abs_acceleration = acceleration.abs()
    acceleration_features = torch.stack(
        (
            abs_acceleration.mean(dim=0),
            acceleration.square().mean(dim=0)
            .clamp_min(RMS_SQUARED_FLOOR)
            .sqrt(),
            torch.quantile(
                abs_acceleration, 0.95, dim=0, interpolation="linear"
            ),
        ),
        dim=0,
    )

    global_features = torch.stack(
        (
            torch.log1p(actions.new_tensor(float(frame_count))),
            duration_sec,
            difference.abs().sum(dim=0).mean(),
            velocity.square().mean().clamp_min(RMS_SQUARED_FLOOR).sqrt(),
        )
    )
    return torch.cat(
        (
            phase.reshape(-1),
            position_features.reshape(-1),
            velocity_features.reshape(-1),
            acceleration_features.reshape(-1),
            global_features,
        )
    )


def beat2_motion_descriptor_tensor(
    actions: torch.Tensor,
    *,
    frame_mask: torch.Tensor | None = None,
    durations_sec: torch.Tensor | None = None,
    phase_samples: int = DEFAULT_PHASE_SAMPLES,
    fps: float = DEFAULT_FPS,
    validate: bool = True,
) -> torch.Tensor:
    """Compute the complete differentiable BEAT2 descriptor.

    Args:
        actions: Physical joint angles with shape ``[B, T, 18]``.
        frame_mask: Boolean valid-prefix mask with shape ``[B, T]``.  Values in
            the padded suffix cannot affect the result or its gradient.
        durations_sec: Physical sample spans with shape ``[B]``.  When omitted,
            each span is derived as ``(valid_frames - 1) / fps``.
        phase_samples: Phase-resampled trajectory length.  The stored encoder
            currently uses 24.
        fps: Fallback rate used only when ``durations_sec`` is omitted.
        validate: Fail closed on non-prefix masks and non-finite inputs.  Known
            safe collators may set this false to avoid device synchronization.

    Returns:
        A float32 tensor with shape ``[B, descriptor_dim(phase_samples)]``.
    """

    if not isinstance(actions, torch.Tensor):
        raise TypeError("actions must be a torch.Tensor")
    if actions.ndim != 3 or actions.shape[-1] != ACTION_DIM:
        raise ValueError(
            f"actions must have shape [batch, frames, {ACTION_DIM}], got "
            f"{tuple(actions.shape)}"
        )
    if not actions.is_floating_point():
        raise ValueError("actions must use a floating-point dtype")
    phase_samples = int(phase_samples)
    expected_dim = descriptor_dim(phase_samples)

    # Descriptor normalization and the frozen encoder were fitted in float32.
    # This cast remains differentiable under mixed-precision generator training.
    values = actions.float()
    _, lengths = _prefix_frame_mask(values, frame_mask, validate=validate)
    durations = _duration_tensor(
        lengths,
        durations_sec,
        fps=float(fps),
        dtype=values.dtype,
        device=values.device,
        validate=validate,
    )
    if validate and bool(torch.any(~torch.isfinite(values)).item()):
        raise ValueError("actions contain non-finite values")

    # One host transfer is substantially cheaper than one GPU synchronization
    # per variable-length item.
    length_values = lengths.detach().cpu().tolist()
    rows = []
    for batch_index, length in enumerate(length_values):
        rows.append(
            _single_motion_descriptor(
                values[batch_index, : int(length)],
                duration_sec=durations[batch_index],
                phase_samples=phase_samples,
            )
        )
    result = torch.stack(rows, dim=0)
    if result.shape[1] != expected_dim:
        raise RuntimeError(
            f"internal descriptor width mismatch: {result.shape[1]} vs {expected_dim}"
        )
    if validate and bool(torch.any(~torch.isfinite(result)).item()):
        raise ValueError("motion descriptor contains non-finite values")
    return result


def standardize_beat2_descriptors(
    descriptors: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    *,
    clip: float = DESCRIPTOR_CLIP,
) -> torch.Tensor:
    """Apply the stored train-only descriptor normalization."""

    if descriptors.ndim != 2:
        raise ValueError("descriptors must have shape [batch, descriptor_dim]")
    mean = torch.as_tensor(mean, dtype=descriptors.dtype, device=descriptors.device)
    scale = torch.as_tensor(
        scale, dtype=descriptors.dtype, device=descriptors.device
    )
    if mean.shape != descriptors.shape[1:] or scale.shape != descriptors.shape[1:]:
        raise ValueError("descriptor normalization width does not match descriptors")
    if bool(torch.any(~torch.isfinite(mean)).item()) or bool(
        torch.any(~torch.isfinite(scale) | (scale <= 0)).item()
    ):
        raise ValueError("descriptor normalization is invalid")
    standardized = (descriptors - mean) / scale
    if clip is not None:
        clip = float(clip)
        if not math.isfinite(clip) or clip <= 0:
            raise ValueError("descriptor clip must be finite and positive")
        standardized = standardized.clamp(-clip, clip)
    return standardized


class Beat2MotionEncoder(nn.Module):
    """Checkpoint-compatible implementation of the frozen BEAT2 encoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        label_sizes: Mapping[str, int],
        *,
        dropout: float,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.input_dim),
        )
        self.category_head = nn.Linear(
            self.latent_dim, int(label_sizes["category"])
        )
        self.intensity_head = nn.Linear(
            self.latent_dim, int(label_sizes["intensity"])
        )
        self.emotion_head = nn.Linear(
            self.latent_dim, int(label_sizes["emotion"])
        )
        self.group_head = nn.Linear(self.latent_dim, int(label_sizes["group"]))

    def forward(self, descriptors: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.encoder(descriptors)
        return {
            "raw": raw,
            "embedding": F.normalize(raw, dim=-1),
            "reconstruction": self.decoder(raw),
            "category_logits": self.category_head(raw),
            "intensity_logits": self.intensity_head(raw),
            "emotion_logits": self.emotion_head(raw),
            "group_logits": self.group_head(raw),
        }


def _load_descriptor_artifact(
    path: str | Path,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {"descriptor_mean", "descriptor_scale", "metadata_json"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(
                f"descriptor cache is missing fields: {sorted(missing)}"
            )
        mean = np.asarray(archive["descriptor_mean"], dtype=np.float32)
        scale = np.asarray(archive["descriptor_scale"], dtype=np.float32)
        metadata = json.loads(str(archive["metadata_json"].item()))
    expected = {
        "artifact_kind": DESCRIPTOR_CACHE_ARTIFACT_KIND,
        "data_policy": BEAT2_DATA_POLICY,
        "no_kimodo": True,
        "normalization_fit_split": "train",
        "joint_names": list(BEAT2_JOINT_NAMES),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"descriptor cache metadata mismatch for {key}")
    phase_samples = int(metadata.get("phase_samples", -1))
    expected_width = descriptor_dim(phase_samples)
    if int(metadata.get("descriptor_dim", -1)) != expected_width:
        raise ValueError("descriptor cache declares the wrong descriptor width")
    if mean.shape != (expected_width,) or scale.shape != (expected_width,):
        raise ValueError("descriptor cache normalization has the wrong shape")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("descriptor cache normalization is non-finite or non-positive")
    return torch.from_numpy(mean), torch.from_numpy(scale), metadata


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_train_qwen_group_prototypes(
    condition_cache_path: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
    expected_csv_set_sha256: str | None = None,
    expected_group_count: int | None = None,
    expected_variant: str = "frozen_base",
    aggregation: str = "require_identical",
    consistency_tolerance: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Build one frozen Qwen prototype per semantic group from train rows only.

    ``require_identical`` is the fail-closed default: all train rows assigned to
    one group must carry the same aligned Qwen latent within
    ``consistency_tolerance``.  ``normalized_train_mean`` is an explicit
    alternative when a future cache intentionally contains prompt variants.
    Neither mode reads validation or test rows.
    """

    path = Path(condition_cache_path)
    expected_variant = str(expected_variant)
    if expected_variant not in {"frozen_base", "lora_finetuned"}:
        raise ValueError("unsupported Qwen condition cache variant")
    sidecar_path = path.with_suffix(path.suffix + ".json")
    if not sidecar_path.is_file():
        raise ValueError("Qwen condition cache metadata sidecar is missing")
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    expected = {
        "artifact_kind": QWEN_CONDITION_ARTIFACT_KIND,
        "data_policy": BEAT2_DATA_POLICY,
        "no_kimodo": True,
        "variant": expected_variant,
        "condition_dim": LATENT_DIM,
        "motion_latent_dim": LATENT_DIM,
        "condition_normalization": "unit_l2_per_canonical_prompt",
    }
    if expected_manifest_sha256 is not None:
        expected["source_manifest_sha256"] = str(expected_manifest_sha256)
    if expected_csv_set_sha256 is not None:
        expected["csv_set_sha256"] = str(expected_csv_set_sha256)
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Qwen condition cache metadata mismatch for {key}")
    cache_sha256 = _sha256_file(path)
    if metadata.get("cache_sha256") != cache_sha256:
        raise ValueError("Qwen condition cache sha256 does not match its sidecar")

    if aggregation not in {"require_identical", "normalized_train_mean"}:
        raise ValueError(
            "prototype aggregation must be require_identical or "
            "normalized_train_mean"
        )
    consistency_tolerance = float(consistency_tolerance)
    if not math.isfinite(consistency_tolerance) or consistency_tolerance < 0:
        raise ValueError("consistency_tolerance must be finite and non-negative")

    with np.load(path, allow_pickle=False) as archive:
        required = {
            "conditions",
            "motion_latents",
            "semantic_group_indices",
            "fixed_split_assignments",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(
                f"Qwen condition cache is missing fields: {sorted(missing)}"
            )
        conditions = np.asarray(archive["conditions"], dtype=np.float32)
        motion_latents = np.asarray(
            archive["motion_latents"], dtype=np.float32
        )
        stored_groups = archive["semantic_group_indices"]
        split_names = np.asarray(archive["fixed_split_assignments"]).astype(str)
    if conditions.ndim != 2 or conditions.shape[1] != LATENT_DIM:
        raise ValueError("Qwen condition cache must contain [N, 128] conditions")
    if motion_latents.shape != conditions.shape or not np.array_equal(
        motion_latents, conditions
    ):
        raise ValueError(
            "Qwen condition cache motion_latents must equal direct conditions"
        )
    if stored_groups.dtype.kind not in "iu" or stored_groups.shape != (
        conditions.shape[0],
    ):
        raise ValueError("Qwen condition cache group indices are invalid")
    if split_names.shape != (conditions.shape[0],):
        raise ValueError("Qwen condition cache split assignments are invalid")
    if not np.isfinite(conditions).all():
        raise ValueError("Qwen condition cache contains non-finite values")

    train_mask = split_names == "train"
    if not train_mask.any():
        raise ValueError("Qwen condition cache has no train rows")
    train_conditions = conditions[train_mask]
    train_groups = np.asarray(stored_groups[train_mask], dtype=np.int64)
    group_ids = sorted(set(int(value) for value in train_groups.tolist()))
    if expected_group_count is None:
        expected_group_count = int(metadata.get("unique_condition_count", -1))
    expected_group_count = int(expected_group_count)
    if group_ids != list(range(expected_group_count)):
        raise ValueError(
            "train Qwen condition cache does not cover every expected group"
        )

    prototypes = []
    group_counts = {}
    maximum_deviation = 0.0
    for group_id in group_ids:
        values = train_conditions[train_groups == group_id]
        group_counts[str(group_id)] = int(values.shape[0])
        deviation = float(np.max(np.abs(values - values[0])))
        maximum_deviation = max(maximum_deviation, deviation)
        if aggregation == "require_identical":
            if deviation > consistency_tolerance:
                raise ValueError(
                    f"train Qwen group {group_id} is not latent-consistent: "
                    f"{deviation} > {consistency_tolerance}"
                )
            prototype = values[0].astype(np.float64)
        else:
            prototype = values.mean(axis=0, dtype=np.float64)
        norm = float(np.linalg.norm(prototype))
        if not math.isfinite(norm) or norm <= 1e-8:
            raise ValueError(f"train Qwen group {group_id} has a zero prototype")
        prototypes.append((prototype / norm).astype(np.float32))
    prototype_array = np.stack(prototypes)
    prototype_metadata = {
        "artifact_kind": PROTOTYPE_BANK_ARTIFACT_KIND,
        "data_policy": BEAT2_DATA_POLICY,
        "no_kimodo": True,
        "fit_split": "train",
        "source_condition_cache": str(path),
        "source_condition_cache_sha256": cache_sha256,
        "source_manifest_sha256": metadata["source_manifest_sha256"],
        "csv_set_sha256": metadata["csv_set_sha256"],
        "qwen_variant": metadata["variant"],
        "aggregation": aggregation,
        "consistency_tolerance": consistency_tolerance,
        "maximum_within_group_linf_deviation": maximum_deviation,
        "prototype_count": len(group_ids),
        "prototype_dim": LATENT_DIM,
        "group_ids": group_ids,
        "train_row_count": int(train_mask.sum()),
        "train_group_counts": group_counts,
        "validation_or_test_row_count_used": 0,
    }
    return (
        torch.from_numpy(prototype_array),
        torch.as_tensor(group_ids, dtype=torch.long),
        prototype_metadata,
    )


def load_frozen_beat2_motion_encoder(
    checkpoint_path: str | Path,
    *,
    descriptor_metadata: Mapping[str, Any],
    descriptor_width: int,
) -> tuple[Beat2MotionEncoder, dict[str, Any]]:
    """Load and contract-check the existing BEAT2-only MotionEncoder."""

    checkpoint = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=True
    )
    expected = {
        "artifact_kind": MOTION_ENCODER_ARTIFACT_KIND,
        "data_policy": BEAT2_DATA_POLICY,
        "no_kimodo": True,
        "initialization": "random_seeded_no_input_checkpoint",
        "input_checkpoint_count": 0,
        "source_manifest_sha256": descriptor_metadata.get("manifest_sha256"),
        "csv_set_sha256": descriptor_metadata.get("csv_set_sha256"),
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"motion encoder checkpoint mismatch for {key}")

    model_config = checkpoint.get("model_config") or {}
    if int(model_config.get("input_dim", -1)) != int(descriptor_width):
        raise ValueError("motion encoder input dimension does not match descriptor")
    if int(model_config.get("latent_dim", -1)) != LATENT_DIM:
        raise ValueError(f"motion encoder latent dimension must be {LATENT_DIM}")
    label_contract = checkpoint.get("label_contract") or {}
    label_sizes = {
        "category": len(label_contract.get("categories", ())),
        "intensity": len(label_contract.get("intensities", ())),
        "emotion": len(label_contract.get("emotions", ())),
        "group": len(label_contract.get("groups", ())),
    }
    if any(size <= 0 for size in label_sizes.values()):
        raise ValueError("motion encoder label contract is incomplete")
    model = Beat2MotionEncoder(
        model_config["input_dim"],
        model_config["hidden_dim"],
        model_config["latent_dim"],
        label_sizes,
        dropout=model_config["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False)
    model.eval()
    return model, checkpoint


def group_aware_contrastive_loss(
    motion_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    *,
    group_ids: torch.Tensor | None = None,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric multi-positive InfoNCE without same-group false negatives."""

    if motion_embeddings.ndim != 2 or motion_embeddings.shape != text_embeddings.shape:
        raise ValueError(
            "motion_embeddings and text_embeddings must have equal [batch, dim] shape"
        )
    if motion_embeddings.shape[0] == 0:
        raise ValueError("contrastive batch cannot be empty")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")

    batch_size = motion_embeddings.shape[0]
    if group_ids is None:
        positives = torch.eye(
            batch_size, dtype=torch.bool, device=motion_embeddings.device
        )
    else:
        groups = torch.as_tensor(group_ids, device=motion_embeddings.device)
        if groups.shape != (batch_size,):
            raise ValueError("group_ids must have shape [batch]")
        positives = groups[:, None] == groups[None, :]

    logits = motion_embeddings @ text_embeddings.transpose(0, 1)
    logits = logits / temperature
    negative_infinity = torch.finfo(logits.dtype).min
    positive_logits = logits.masked_fill(~positives, negative_infinity)
    motion_to_text = (
        torch.logsumexp(logits, dim=1)
        - torch.logsumexp(positive_logits, dim=1)
    ).mean()
    text_to_motion = (
        torch.logsumexp(logits, dim=0)
        - torch.logsumexp(positive_logits, dim=0)
    ).mean()
    return 0.5 * (motion_to_text + text_to_motion)


def motion_to_global_prototype_info_nce(
    motion_embeddings: torch.Tensor,
    motion_group_ids: torch.Tensor,
    prototype_embeddings: torch.Tensor,
    prototype_group_ids: torch.Tensor,
    *,
    temperature: float = 0.07,
    validate: bool = True,
) -> dict[str, torch.Tensor]:
    """Contrast every motion against the fixed global semantic prototype bank.

    The positive mask is based only on explicit semantic group equality.  Thus
    prototypes for groups absent from the current minibatch remain negatives
    and can never be promoted to positives by minibatch composition.
    """

    if motion_embeddings.ndim != 2 or motion_embeddings.shape[1] != LATENT_DIM:
        raise ValueError("motion_embeddings must have shape [batch, 128]")
    if (
        prototype_embeddings.ndim != 2
        or prototype_embeddings.shape[1] != LATENT_DIM
    ):
        raise ValueError("prototype_embeddings must have shape [groups, 128]")
    if motion_embeddings.shape[0] == 0 or prototype_embeddings.shape[0] < 2:
        raise ValueError("global prototype InfoNCE needs motions and at least two groups")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")

    device = motion_embeddings.device
    groups = torch.as_tensor(
        motion_group_ids, dtype=torch.long, device=device
    )
    prototype_groups = torch.as_tensor(
        prototype_group_ids, dtype=torch.long, device=device
    )
    if groups.shape != (motion_embeddings.shape[0],):
        raise ValueError("motion_group_ids must have shape [batch]")
    if prototype_groups.shape != (prototype_embeddings.shape[0],):
        raise ValueError("prototype_group_ids must have shape [groups]")
    if validate and torch.unique(
        prototype_groups
    ).numel() != prototype_groups.numel():
        raise ValueError("global prototype bank must contain one row per group")

    motion = F.normalize(motion_embeddings.float(), dim=-1)
    prototypes = F.normalize(
        prototype_embeddings.detach().to(device=device, dtype=torch.float32),
        dim=-1,
    )
    positive_mask = groups[:, None] == prototype_groups[None, :]
    if validate and not bool(torch.all(positive_mask.any(dim=1)).item()):
        missing = groups[~positive_mask.any(dim=1)].detach().cpu().unique().tolist()
        raise ValueError(
            f"global prototype bank has no positive for motion groups {missing}"
        )
    negative_mask = ~positive_mask
    if validate and not bool(torch.all(negative_mask.any(dim=1)).item()):
        raise ValueError("global prototype bank provides no cross-group negatives")

    cosine = motion @ prototypes.transpose(0, 1)
    logits = cosine / temperature
    negative_infinity = torch.finfo(logits.dtype).min
    positive_logits = logits.masked_fill(~positive_mask, negative_infinity)
    loss = (
        torch.logsumexp(logits, dim=1)
        - torch.logsumexp(positive_logits, dim=1)
    ).mean()

    with torch.no_grad():
        positive_cosine = cosine.masked_fill(
            ~positive_mask, -torch.inf
        ).amax(dim=1)
        hard_negative_cosine = cosine.masked_fill(
            ~negative_mask, -torch.inf
        ).amax(dim=1)
        hard_margin = positive_cosine - hard_negative_cosine
        retrieved_groups = prototype_groups[cosine.argmax(dim=1)]
        recall_at_1 = (retrieved_groups == groups).float().mean()
        positive_rank = 1 + (
            negative_mask
            & (cosine >= positive_cosine[:, None])
        ).sum(dim=1)
    return {
        "loss": loss,
        "positive_cosine": positive_cosine.mean(),
        "hard_negative_cosine": hard_negative_cosine.mean(),
        "hard_margin": hard_margin.mean(),
        "hard_margin_positive_fraction": (hard_margin > 0).float().mean(),
        "recall_at_1": recall_at_1,
        "mean_positive_rank": positive_rank.float().mean(),
        "prototype_count": cosine.new_tensor(
            prototype_embeddings.shape[0], dtype=torch.int64
        ),
    }


def _cross_group_metrics(
    motion_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    group_ids: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    similarity = motion_embeddings.detach() @ text_embeddings.detach().transpose(0, 1)
    batch_size = similarity.shape[0]
    if group_ids is None:
        groups = torch.arange(batch_size, device=similarity.device)
    else:
        groups = torch.as_tensor(group_ids, device=similarity.device)
        if groups.shape != (batch_size,):
            raise ValueError("group_ids must have shape [batch]")
    same_group = groups[:, None] == groups[None, :]
    cross_group = ~same_group
    aligned = similarity.diagonal()
    same_values = similarity[same_group]
    cross_values = similarity[cross_group]
    zero = similarity.new_zeros(())
    if cross_values.numel():
        cross_mean = cross_values.mean()
        hardest = similarity.masked_fill(~cross_group, -torch.inf).amax(dim=1)
        margin = aligned - hardest
        m2t_group = groups[similarity.argmax(dim=1)]
        t2m_group = groups[similarity.argmax(dim=0)]
        m2t_recall = (m2t_group == groups).float().mean()
        t2m_recall = (t2m_group == groups).float().mean()
        hard_margin_mean = margin.mean()
        hard_margin_positive_fraction = (margin > 0).float().mean()
    else:
        cross_mean = zero
        m2t_recall = zero
        t2m_recall = zero
        hard_margin_mean = zero
        hard_margin_positive_fraction = zero
    return {
        "aligned_cosine": aligned.mean(),
        "same_group_cosine": same_values.mean(),
        "cross_group_cosine": cross_mean,
        "cross_group_cosine_gap": same_values.mean() - cross_mean,
        "hard_cross_group_margin": hard_margin_mean,
        "hard_cross_group_margin_positive_fraction": hard_margin_positive_fraction,
        "motion_to_text_group_recall_at_1": m2t_recall,
        "text_to_motion_group_recall_at_1": t2m_recall,
        "cross_group_pair_count": similarity.new_tensor(
            cross_values.numel(), dtype=torch.int64
        ),
    }


class Beat2SemanticPerceptualLoss(nn.Module):
    """Frozen descriptor encoder plus Qwen-aligned semantic losses."""

    def __init__(
        self,
        motion_encoder: Beat2MotionEncoder,
        *,
        descriptor_mean: torch.Tensor,
        descriptor_scale: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        global_prototype_embeddings: torch.Tensor | None = None,
        global_prototype_group_ids: torch.Tensor | None = None,
        phase_samples: int = DEFAULT_PHASE_SAMPLES,
        fps: float = DEFAULT_FPS,
        descriptor_clip: float = DESCRIPTOR_CLIP,
        cosine_weight: float = 1.0,
        contrastive_weight: float = 1.0,
        global_contrastive_weight: float = 0.0,
        temperature: float = 0.07,
        validate_inputs: bool = True,
        artifact_metadata: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        self.motion_encoder = motion_encoder.requires_grad_(False).eval()
        self.phase_samples = int(phase_samples)
        self.fps = float(fps)
        self.descriptor_clip = float(descriptor_clip)
        self.cosine_weight = float(cosine_weight)
        self.contrastive_weight = float(contrastive_weight)
        self.global_contrastive_weight = float(global_contrastive_weight)
        self.temperature = float(temperature)
        self.validate_inputs = bool(validate_inputs)
        self.artifact_metadata = dict(artifact_metadata or {})

        expected_width = descriptor_dim(self.phase_samples)
        descriptor_mean = torch.as_tensor(descriptor_mean, dtype=torch.float32)
        descriptor_scale = torch.as_tensor(descriptor_scale, dtype=torch.float32)
        action_mean = torch.as_tensor(action_mean, dtype=torch.float32)
        action_std = torch.as_tensor(action_std, dtype=torch.float32)
        if descriptor_mean.shape != (expected_width,) or descriptor_scale.shape != (
            expected_width,
        ):
            raise ValueError("descriptor normalization width is invalid")
        if bool(torch.any(~torch.isfinite(descriptor_mean)).item()) or bool(
            torch.any(
                ~torch.isfinite(descriptor_scale) | (descriptor_scale <= 0)
            ).item()
        ):
            raise ValueError("descriptor normalization is invalid")
        if action_mean.shape != (ACTION_DIM,) or action_std.shape != (ACTION_DIM,):
            raise ValueError("generator action normalization must contain 18 values")
        if bool(torch.any(~torch.isfinite(action_mean)).item()) or bool(
            torch.any(~torch.isfinite(action_std) | (action_std <= 0)).item()
        ):
            raise ValueError("generator action normalization is invalid")
        if self.motion_encoder.input_dim != expected_width:
            raise ValueError("motion encoder and descriptor widths differ")
        if self.motion_encoder.latent_dim != LATENT_DIM:
            raise ValueError(f"motion encoder latent dimension must be {LATENT_DIM}")
        for name, value in (
            ("cosine_weight", self.cosine_weight),
            ("contrastive_weight", self.contrastive_weight),
            ("global_contrastive_weight", self.global_contrastive_weight),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            self.cosine_weight == 0
            and self.contrastive_weight == 0
            and self.global_contrastive_weight == 0
        ):
            raise ValueError("at least one semantic loss weight must be positive")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(self.descriptor_clip) or self.descriptor_clip <= 0:
            raise ValueError("descriptor_clip must be finite and positive")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")

        if (global_prototype_embeddings is None) != (
            global_prototype_group_ids is None
        ):
            raise ValueError(
                "global prototype embeddings and group ids must be provided together"
            )
        if global_prototype_embeddings is None:
            prototype_embeddings = torch.empty(
                (0, LATENT_DIM), dtype=torch.float32
            )
            prototype_group_ids = torch.empty((0,), dtype=torch.long)
        else:
            prototype_embeddings = torch.as_tensor(
                global_prototype_embeddings, dtype=torch.float32
            )
            prototype_group_ids = torch.as_tensor(
                global_prototype_group_ids, dtype=torch.long
            )
            if (
                prototype_embeddings.ndim != 2
                or prototype_embeddings.shape[1] != LATENT_DIM
                or prototype_group_ids.shape != (
                    prototype_embeddings.shape[0],
                )
            ):
                raise ValueError("global prototype bank shapes are invalid")
            if prototype_embeddings.shape[0] < 2:
                raise ValueError("global prototype bank needs at least two groups")
            if torch.unique(prototype_group_ids).numel() != len(
                prototype_group_ids
            ):
                raise ValueError(
                    "global prototype bank must contain one row per group"
                )
            if bool(
                torch.any(~torch.isfinite(prototype_embeddings)).item()
            ) or bool(
                torch.any(prototype_embeddings.norm(dim=-1) <= 1e-8).item()
            ):
                raise ValueError("global prototype bank is non-finite or zero")
            prototype_embeddings = F.normalize(
                prototype_embeddings, dim=-1
            )
        if self.global_contrastive_weight > 0 and not len(
            prototype_group_ids
        ):
            raise ValueError(
                "global_contrastive_weight requires a global prototype bank"
            )

        self.register_buffer("descriptor_mean", descriptor_mean)
        self.register_buffer("descriptor_scale", descriptor_scale)
        self.register_buffer("action_mean", action_mean)
        self.register_buffer("action_std", action_std)
        self.register_buffer(
            "global_prototype_embeddings", prototype_embeddings
        )
        self.register_buffer(
            "global_prototype_group_ids", prototype_group_ids
        )

    @classmethod
    def from_artifacts(
        cls,
        *,
        descriptor_cache_path: str | Path,
        motion_encoder_checkpoint_path: str | Path,
        action_stats: Mapping[str, Any],
        qwen_condition_cache_path: str | Path | None = None,
        prototype_aggregation: str = "require_identical",
        prototype_consistency_tolerance: float = 1e-6,
        cosine_weight: float = 1.0,
        contrastive_weight: float = 1.0,
        global_contrastive_weight: float = 0.0,
        temperature: float = 0.07,
        descriptor_clip: float = DESCRIPTOR_CLIP,
        validate_inputs: bool = True,
        device: str | torch.device | None = None,
    ) -> "Beat2SemanticPerceptualLoss":
        """Construct the loss from the existing BEAT2 alignment artifacts."""

        mean, scale, descriptor_metadata = _load_descriptor_artifact(
            descriptor_cache_path
        )
        model, checkpoint = load_frozen_beat2_motion_encoder(
            motion_encoder_checkpoint_path,
            descriptor_metadata=descriptor_metadata,
            descriptor_width=mean.numel(),
        )
        if not {"mean", "std"}.issubset(action_stats):
            raise ValueError("action_stats must contain mean and std")
        prototype_embeddings = None
        prototype_group_ids = None
        prototype_metadata = None
        if qwen_condition_cache_path is not None:
            (
                prototype_embeddings,
                prototype_group_ids,
                prototype_metadata,
            ) = load_train_qwen_group_prototypes(
                qwen_condition_cache_path,
                expected_manifest_sha256=descriptor_metadata[
                    "manifest_sha256"
                ],
                expected_csv_set_sha256=descriptor_metadata[
                    "csv_set_sha256"
                ],
                expected_group_count=model.group_head.out_features,
                aggregation=prototype_aggregation,
                consistency_tolerance=prototype_consistency_tolerance,
            )
        module = cls(
            model,
            descriptor_mean=mean,
            descriptor_scale=scale,
            action_mean=action_stats["mean"],
            action_std=action_stats["std"],
            global_prototype_embeddings=prototype_embeddings,
            global_prototype_group_ids=prototype_group_ids,
            phase_samples=int(descriptor_metadata["phase_samples"]),
            fps=float(descriptor_metadata["fps"]),
            descriptor_clip=descriptor_clip,
            cosine_weight=cosine_weight,
            contrastive_weight=contrastive_weight,
            global_contrastive_weight=global_contrastive_weight,
            temperature=temperature,
            validate_inputs=validate_inputs,
            artifact_metadata={
                "data_policy": BEAT2_DATA_POLICY,
                "no_kimodo": True,
                "manifest_sha256": descriptor_metadata["manifest_sha256"],
                "csv_set_sha256": descriptor_metadata["csv_set_sha256"],
                "descriptor_cache_path": str(Path(descriptor_cache_path)),
                "motion_encoder_checkpoint_path": str(
                    Path(motion_encoder_checkpoint_path)
                ),
                "motion_encoder_step": int(checkpoint["step"]),
                "global_prototype_bank": prototype_metadata,
            },
        )
        if device is not None:
            module = module.to(device)
        return module

    def train(self, mode: bool = True) -> "Beat2SemanticPerceptualLoss":
        # A containing trainer may call train() recursively.  The checkpoint's
        # Dropout must remain disabled while gradients still flow to the input.
        super().train(mode)
        self.motion_encoder.eval()
        return self

    def describe(
        self,
        normalized_actions: torch.Tensor,
        *,
        frame_mask: torch.Tensor | None = None,
        durations_sec: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw and train-standardized descriptors."""

        physical = (
            normalized_actions.float() * self.action_std[None, None, :]
            + self.action_mean[None, None, :]
        )
        raw = beat2_motion_descriptor_tensor(
            physical,
            frame_mask=frame_mask,
            durations_sec=durations_sec,
            phase_samples=self.phase_samples,
            fps=self.fps,
            validate=self.validate_inputs,
        )
        # These artifact buffers were validated once during construction.
        # Avoid a device synchronization in every training step.
        standardized = (
            (raw - self.descriptor_mean) / self.descriptor_scale
        ).clamp(-self.descriptor_clip, self.descriptor_clip)
        return raw, standardized

    def forward(
        self,
        normalized_actions: torch.Tensor,
        aligned_qwen_latents: torch.Tensor,
        *,
        frame_mask: torch.Tensor | None = None,
        durations_sec: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute semantic loss and detached retrieval diagnostics.

        ``aligned_qwen_latents`` must be the direct 128D ``conditions`` field
        exported by ``conditions_128d_frozen_base.npz``.  It is detached here;
        gradients intentionally flow only through the generated motion.
        """

        if aligned_qwen_latents.ndim != 2 or aligned_qwen_latents.shape != (
            normalized_actions.shape[0],
            LATENT_DIM,
        ):
            raise ValueError(
                "aligned_qwen_latents must have shape [batch, 128]"
            )
        targets = aligned_qwen_latents.detach().to(
            device=normalized_actions.device, dtype=torch.float32
        )
        if self.validate_inputs and bool(torch.any(~torch.isfinite(targets)).item()):
            raise ValueError("aligned_qwen_latents contain non-finite values")
        if self.validate_inputs and bool(
            torch.any(targets.norm(dim=-1) <= 1e-8).item()
        ):
            raise ValueError("aligned_qwen_latents contain a zero vector")
        targets = F.normalize(targets, dim=-1)

        raw_descriptors, descriptors = self.describe(
            normalized_actions,
            frame_mask=frame_mask,
            durations_sec=durations_sec,
        )
        encoder_output = self.motion_encoder(descriptors)
        motion_embeddings = encoder_output["embedding"]
        aligned_cosine = (motion_embeddings * targets).sum(dim=-1)
        cosine = (1.0 - aligned_cosine).mean()
        contrastive = (
            group_aware_contrastive_loss(
                motion_embeddings,
                targets,
                group_ids=group_ids,
                temperature=self.temperature,
            )
            if self.contrastive_weight > 0
            else motion_embeddings.sum() * 0.0
        )
        global_result = None
        if self.global_prototype_embeddings.shape[0]:
            if group_ids is not None:
                global_result = motion_to_global_prototype_info_nce(
                    motion_embeddings,
                    group_ids,
                    self.global_prototype_embeddings,
                    self.global_prototype_group_ids,
                    temperature=self.temperature,
                    validate=self.validate_inputs,
                )
            elif self.global_contrastive_weight > 0:
                raise ValueError(
                    "group_ids are required for global prototype InfoNCE"
                )
        global_contrastive = (
            global_result["loss"]
            if global_result is not None
            else motion_embeddings.sum() * 0.0
        )
        total = (
            self.cosine_weight * cosine
            + self.contrastive_weight * contrastive
            + self.global_contrastive_weight * global_contrastive
        )
        metrics = _cross_group_metrics(
            motion_embeddings, targets, group_ids
        )
        result = {
            "total": total,
            "cosine": cosine,
            "contrastive": contrastive,
            "global_contrastive": global_contrastive,
            "motion_embeddings": motion_embeddings,
            "raw_descriptors": raw_descriptors,
            "standardized_descriptors": descriptors,
        }
        result.update(
            {name: value.detach() for name, value in metrics.items()}
        )
        if global_result is not None:
            result.update(
                {
                    "global_positive_cosine": global_result[
                        "positive_cosine"
                    ].detach(),
                    "global_hard_negative_cosine": global_result[
                        "hard_negative_cosine"
                    ].detach(),
                    "global_hard_cross_group_margin": global_result[
                        "hard_margin"
                    ].detach(),
                    "global_hard_cross_group_margin_positive_fraction": (
                        global_result[
                            "hard_margin_positive_fraction"
                        ].detach()
                    ),
                    "global_motion_to_prototype_recall_at_1": global_result[
                        "recall_at_1"
                    ].detach(),
                    "global_mean_positive_rank": global_result[
                        "mean_positive_rank"
                    ].detach(),
                    "global_prototype_count": global_result[
                        "prototype_count"
                    ].detach(),
                }
            )
        if group_ids is not None:
            groups = torch.as_tensor(
                group_ids, dtype=torch.long, device=motion_embeddings.device
            )
            if self.validate_inputs and bool(
                torch.any(
                    (groups < 0)
                    | (groups >= encoder_output["group_logits"].shape[-1])
                ).item()
            ):
                raise ValueError("group_ids are outside the MotionEncoder label contract")
            result["motion_encoder_group_accuracy"] = (
                encoder_output["group_logits"].detach().argmax(dim=-1) == groups
            ).float().mean()
        return result


__all__ = [
    "ACTION_DIM",
    "BEAT2_DATA_POLICY",
    "BEAT2_JOINT_NAMES",
    "Beat2MotionEncoder",
    "Beat2SemanticPerceptualLoss",
    "DEFAULT_FPS",
    "DEFAULT_PHASE_SAMPLES",
    "LATENT_DIM",
    "beat2_motion_descriptor_tensor",
    "descriptor_dim",
    "group_aware_contrastive_loss",
    "load_train_qwen_group_prototypes",
    "load_frozen_beat2_motion_encoder",
    "motion_to_global_prototype_info_nce",
    "standardize_beat2_descriptors",
]
