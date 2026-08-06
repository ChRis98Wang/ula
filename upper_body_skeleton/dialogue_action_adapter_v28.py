"""Variable-length formal-GT trajectory adapter for dialogue gestures."""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import nn


ARTIFACT_KIND = "dialogue_action_formal_gt_adapter_v28"
ARCHITECTURE = "intent_one_hot_realization_embedding_to_masked_fourier_trajectory"
DEFAULT_CONFIG = {
    "intent_dim": 27,
    "action_dim": 18,
    "realization_count": 24,
    "realization_embedding_dim": 64,
    "condition_dim": 384,
    "hidden_dim": 768,
    "time_frequencies": 47,
}


class DialogueActionFormalAdapterV28(nn.Module):
    """Decode reviewed intent/realization pairs over per-sample normalized phase."""

    def __init__(self, config: Mapping | None = None) -> None:
        super().__init__()
        resolved = dict(DEFAULT_CONFIG | dict(config or {}))
        self.config = resolved
        intent_dim = int(resolved["intent_dim"])
        action_dim = int(resolved["action_dim"])
        realization_count = int(resolved["realization_count"])
        realization_dim = int(resolved["realization_embedding_dim"])
        condition_dim = int(resolved["condition_dim"])
        hidden_dim = int(resolved["hidden_dim"])
        frequencies = int(resolved["time_frequencies"])
        if (
            intent_dim <= 0
            or action_dim != 18
            or realization_count <= 0
            or min(realization_dim, condition_dim, hidden_dim, frequencies) <= 0
        ):
            raise ValueError("invalid V28 formal-adapter dimensions")
        self.intent_dim = intent_dim
        self.action_dim = action_dim
        self.realization_count = realization_count
        self.intent_condition = nn.Sequential(
            nn.Linear(intent_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.realization_embedding = nn.Embedding(realization_count, realization_dim)
        self.basis_dim = 2 + 2 * frequencies
        self.coefficient_decoder = nn.Sequential(
            nn.Linear(condition_dim + realization_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim * self.basis_dim),
        )
        self.register_buffer(
            "time_frequency_values",
            torch.arange(1, frequencies + 1, dtype=torch.float32),
            persistent=True,
        )

    def forward_padded(
        self,
        intent_one_hot: torch.Tensor,
        realization_ids: torch.Tensor,
        frame_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if intent_one_hot.ndim != 2 or intent_one_hot.shape[1] != self.intent_dim:
            raise ValueError("V28 intent input must be [batch, intent_dim]")
        batch = intent_one_hot.shape[0]
        if realization_ids.ndim != 1 or realization_ids.shape[0] != batch:
            raise ValueError("V28 realization IDs must be [batch]")
        if frame_lengths.ndim != 1 or frame_lengths.shape[0] != batch:
            raise ValueError("V28 frame lengths must be [batch]")
        if realization_ids.dtype not in {torch.int32, torch.int64}:
            raise ValueError("V28 realization IDs must be integer tensors")
        if frame_lengths.dtype not in {torch.int32, torch.int64}:
            raise ValueError("V28 frame lengths must be integer tensors")
        if bool(
            torch.any(realization_ids < 0)
            or torch.any(realization_ids >= self.realization_count)
        ):
            raise ValueError("V28 realization ID is out of range")
        if bool(torch.any(frame_lengths < 2)):
            raise ValueError("V28 trajectories require at least two frames")

        maximum_frames = int(torch.max(frame_lengths).detach().cpu())
        frame_index = torch.arange(
            maximum_frames, device=intent_one_hot.device, dtype=intent_one_hot.dtype
        )[None, :]
        valid_mask = frame_index < frame_lengths[:, None]
        phase = frame_index / (frame_lengths[:, None].to(intent_one_hot.dtype) - 1.0)
        phase = phase.clamp(0.0, 1.0)
        angular = (
            2.0
            * math.pi
            * phase[:, :, None]
            * self.time_frequency_values[None, None].to(
                device=phase.device, dtype=phase.dtype
            )
        )
        basis = torch.cat(
            [
                torch.ones_like(phase[:, :, None]),
                phase[:, :, None],
                torch.sin(angular),
                torch.cos(angular),
            ],
            dim=-1,
        )
        intent = self.intent_condition(intent_one_hot)
        realization = self.realization_embedding(realization_ids)
        coefficients = self.coefficient_decoder(
            torch.cat([intent, realization], dim=-1)
        ).reshape(batch, self.action_dim, self.basis_dim)
        output = torch.einsum("btk,bak->bta", basis, coefficients)
        return output, valid_mask

    def forward(
        self,
        intent_one_hot: torch.Tensor,
        realization_ids: torch.Tensor,
        *,
        frames: int,
    ) -> torch.Tensor:
        lengths = torch.full(
            (intent_one_hot.shape[0],),
            int(frames),
            dtype=torch.long,
            device=intent_one_hot.device,
        )
        output, _ = self.forward_padded(intent_one_hot, realization_ids, lengths)
        return output


__all__ = [
    "ARCHITECTURE",
    "ARTIFACT_KIND",
    "DEFAULT_CONFIG",
    "DialogueActionFormalAdapterV28",
]
