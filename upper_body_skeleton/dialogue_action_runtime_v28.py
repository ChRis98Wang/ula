"""Dataset-free runtime for the reviewed V28 formal action adapter."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from upper_body_skeleton.dialogue_action_adapter_v28 import (
    ARCHITECTURE,
    ARTIFACT_KIND,
    DialogueActionFormalAdapterV28,
)
from upper_body_skeleton.retarget_v2_18d import JOINT_LIMITS_18D, JOINT_ORDER_18D
from upper_body_skeleton.ula_v2_18d_head import (
    ACTION_DIM,
    load_contract_checkpoint,
)


def _validated_payload(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    payload = checkpoint.get("dialogue_action_formal_adapter_v28")
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint does not contain a V28 formal action adapter")
    config = payload.get("config")
    state = payload.get("model_state_dict")
    supported = payload.get("supported_intent_ids")
    slots = payload.get("supported_intent_slots")
    registry = payload.get("realization_registry")
    if (
        payload.get("artifact_kind") != ARTIFACT_KIND
        or payload.get("architecture") != ARCHITECTURE
        or not isinstance(config, Mapping)
        or int(config.get("action_dim", -1)) != ACTION_DIM
        or not isinstance(state, Mapping)
        or not isinstance(supported, list)
        or not supported
        or len(set(map(str, supported))) != len(supported)
        or not isinstance(slots, Mapping)
        or set(map(str, slots)) != set(map(str, supported))
        or not isinstance(registry, list)
        or not registry
        or payload.get("runtime_dataset_or_csv_access") is not False
        or payload.get("formal_release_eligible") is not False
        or checkpoint.get("formal_release_eligible") is not False
    ):
        raise ValueError("invalid V28 formal action adapter contract")

    intent_dim = int(config.get("intent_dim", -1))
    realization_count = int(config.get("realization_count", -1))
    expected_ids = list(range(realization_count))
    observed_ids = []
    seen_per_action: dict[str, list[int]] = defaultdict(list)
    for record in registry:
        if not isinstance(record, Mapping):
            raise ValueError("invalid V28 realization registry record")
        action_id = str(record.get("action_id") or "")
        slot = int(record.get("intent_slot", -1))
        realization_id = int(record.get("global_realization_id", -1))
        if (
            action_id not in supported
            or slot != int(slots.get(action_id, -1))
            or slot < 0
            or slot >= intent_dim
            or realization_id < 0
            or not str(record.get("trajectory_sha256") or "")
        ):
            raise ValueError("invalid V28 realization registry binding")
        observed_ids.append(realization_id)
        seen_per_action[action_id].append(realization_id)
    if sorted(observed_ids) != expected_ids or set(seen_per_action) != set(supported):
        raise ValueError("V28 realization registry is incomplete")
    return dict(payload)


class DialogueActionV28Runtime:
    """Generate supported reviewed actions without reading training trajectories."""

    def __init__(self, checkpoint_path: str | Path, *, device: str = "cpu") -> None:
        self.checkpoint_path = Path(checkpoint_path).resolve()
        _, checkpoint = load_contract_checkpoint(
            self.checkpoint_path,
            expected_action_dim=ACTION_DIM,
            device="cpu",
        )
        payload = _validated_payload(checkpoint)
        self.device = torch.device(device)
        self.adapter = DialogueActionFormalAdapterV28(payload["config"]).to(self.device)
        self.adapter.load_state_dict(payload["model_state_dict"], strict=True)
        self.adapter.requires_grad_(False).eval()
        self.intent_dim = int(payload["config"]["intent_dim"])
        self.intent_slots = {
            str(key): int(value)
            for key, value in payload["supported_intent_slots"].items()
        }
        grouped: dict[str, list[int]] = defaultdict(list)
        for record in payload["realization_registry"]:
            grouped[str(record["action_id"])].append(
                int(record["global_realization_id"])
            )
        self.realization_ids = {
            action_id: tuple(sorted(values))
            for action_id, values in grouped.items()
        }
        self.supported_intent_ids = tuple(sorted(self.realization_ids))
        self.lower = np.asarray(
            [JOINT_LIMITS_18D[name][0] for name in JOINT_ORDER_18D],
            dtype=np.float32,
        )
        self.upper = np.asarray(
            [JOINT_LIMITS_18D[name][1] for name in JOINT_ORDER_18D],
            dtype=np.float32,
        )

    def generate(
        self,
        action_id: str,
        *,
        frames: int,
        seed: int = 0,
    ) -> np.ndarray:
        action_id = str(action_id)
        frames = int(frames)
        if action_id not in self.realization_ids:
            raise ValueError(
                f"V28 pilot does not support {action_id!r}; "
                f"supported={self.supported_intent_ids}"
            )
        if frames < 2:
            raise ValueError("V28 generation requires at least two frames")
        choices = self.realization_ids[action_id]
        realization_id = choices[int(seed) % len(choices)]
        intent = torch.zeros(1, self.intent_dim, dtype=torch.float32, device=self.device)
        intent[0, self.intent_slots[action_id]] = 1.0
        with torch.no_grad():
            output = self.adapter(
                intent,
                torch.tensor([realization_id], dtype=torch.long, device=self.device),
                frames=frames,
            )[0].detach().cpu().numpy().astype(np.float32)
        output = np.clip(output, self.lower, self.upper)
        if output.shape != (frames, ACTION_DIM) or not np.isfinite(output).all():
            raise RuntimeError("V28 adapter returned invalid motion")
        return output


__all__ = ["DialogueActionV28Runtime"]
