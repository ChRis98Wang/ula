"""V4 checkpoint migration and dual-text cache for dialogue-conditioned actions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from upper_body_skeleton.ula_training import (
    KIMODO_CONDITION_DIM,
    KIMODO_V2_CONDITION_DIM,
    ULA_MMDIT_V3_ADALN_ARCHITECTURE,
    ULA_MMDIT_V4_DUAL_TEXT_ADALN_ARCHITECTURE,
    create_ula_model,
)
from upper_body_skeleton.ula_v2_18d_head import (
    ACTION_DIM,
    STYLE_CONTROL_SLICE,
    validate_checkpoint_contract,
)
from upper_body_skeleton.ula_v2_conversational_realization_episode import (
    _Beat2QwenMotionTextEncoder,
)
from upper_body_skeleton.ula_v2_dialogue_action_episode import (
    CONDITION_CACHE_ARTIFACT_KIND,
    CONDITION_CACHE_SCHEMA_VERSION,
    DIALOGUE_LATENT_SLICE,
    DIRECTIVE_LATENT_SLICE,
    FORMAL_EPISODE_CONTRACT,
    array_sha256,
    canonical_sha256,
    sha256_file,
    style_controls,
    validate_dialogue_action_v11_episode,
)


MIGRATION_MODE = "pretrained_18d_motion_foundation_dual_text_roles_v1"
DUAL_TEXT_CONTRACT = "action_directive_64d_plus_dialogue_64d_no_mapping_head_v1"
PROJECTION_POLICY = "fixed_seeded_orthonormal_128d_to_64d_no_trainable_head_v1"


def _atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _model_shape(checkpoint: Mapping[str, Any]) -> dict[str, int]:
    config = checkpoint.get("config") or {}
    return {
        "hidden_dim": int(config.get("hidden_dim") or checkpoint.get("hidden_dim") or 384),
        "layers": int(config.get("layers") or checkpoint.get("layers") or 6),
        "semantic_tokens": int(
            config.get("semantic_tokens") or checkpoint.get("semantic_tokens") or 7
        ),
    }


def migrate_v3_foundation_to_v4_dual_text(
    source_checkpoint: str | Path,
    qwen_checkpoint: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    source_checkpoint = Path(source_checkpoint).resolve()
    qwen_checkpoint = Path(qwen_checkpoint).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite migrated checkpoint: {output_path}")
    source = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
    validate_checkpoint_contract(source, expected_action_dim=ACTION_DIM)
    if source.get("architecture") != ULA_MMDIT_V3_ADALN_ARCHITECTURE:
        raise ValueError("dual-text migration requires the V3 AdaLN 18D foundation")
    qwen = torch.load(qwen_checkpoint, map_location="cpu", weights_only=True)
    if (
        qwen.get("artifact_kind") != "beat2_qwen_lora_alignment_v1"
        or qwen.get("variant") != "lora_finetuned"
        or qwen.get("no_kimodo") is not True
    ):
        raise ValueError("dual-text migration requires the BEAT2-only Qwen LoRA")

    shape = _model_shape(source)
    torch.manual_seed(1107)
    model = create_ula_model(
        ULA_MMDIT_V4_DUAL_TEXT_ADALN_ARCHITECTURE,
        action_dim=ACTION_DIM,
        condition_dim=KIMODO_V2_CONDITION_DIM,
        **shape,
    )
    destination = model.state_dict()
    source_state = source["model_state_dict"]
    role_prefixes = ("action_directive_condition.", "dialogue_condition.")
    copied = []
    for name, value in destination.items():
        if name.startswith(role_prefixes):
            continue
        source_value = source_state.get(name)
        if source_value is None or tuple(source_value.shape) != tuple(value.shape):
            raise ValueError(f"foundation cannot migrate shared V4 parameter {name}")
        value.copy_(source_value)
        copied.append(name)

    old_first_weight = source_state["motion_latent_condition.0.weight"]
    old_first_bias = source_state["motion_latent_condition.0.bias"]
    old_second_weight = source_state["motion_latent_condition.2.weight"]
    old_second_bias = source_state["motion_latent_condition.2.bias"]
    hidden = shape["hidden_dim"]
    if old_first_weight.shape != (hidden * 2, 128):
        raise ValueError("foundation text projection does not have two 128D latent tokens")
    for role, row_slice, column_slice in (
        ("action_directive_condition", slice(0, hidden), slice(0, 64)),
        ("dialogue_condition", slice(hidden, hidden * 2), slice(64, 128)),
    ):
        destination[f"{role}.0.weight"].copy_(
            old_first_weight[row_slice, column_slice]
        )
        destination[f"{role}.0.bias"].copy_(old_first_bias[row_slice])
        destination[f"{role}.2.weight"].copy_(
            old_second_weight[row_slice, row_slice]
        )
        destination[f"{role}.2.bias"].copy_(old_second_bias[row_slice])

    qwen_receipt = qwen.get("qwen") or {}
    contract = {
        "contract_type": "ula_v4_dual_text_conditioning",
        "contract_version": 1,
        "policy": DUAL_TEXT_CONTRACT,
        "action_directive_slice": [DIRECTIVE_LATENT_SLICE.start, DIRECTIVE_LATENT_SLICE.stop],
        "dialogue_slice": [DIALOGUE_LATENT_SLICE.start, DIALOGUE_LATENT_SLICE.stop],
        "projection_policy": PROJECTION_POLICY,
        "mapping_head": "absent",
        "qwen_checkpoint": str(qwen_checkpoint),
        "qwen_checkpoint_sha256": sha256_file(qwen_checkpoint),
        "qwen_model_name": qwen_receipt.get("model_name"),
        "qwen_revision": qwen_receipt.get("revision"),
        "source_foundation_checkpoint": str(source_checkpoint),
        "source_foundation_checkpoint_sha256": sha256_file(source_checkpoint),
    }
    contract["sha256"] = canonical_sha256(contract)
    payload = deepcopy(source)
    payload.update(
        {
            "architecture": ULA_MMDIT_V4_DUAL_TEXT_ADALN_ARCHITECTURE,
            "model_state_dict": destination,
            "formal_episode_contract": FORMAL_EPISODE_CONTRACT,
            "random_initialization": {
                "mode": MIGRATION_MODE,
                "source_foundation_checkpoint_sha256": sha256_file(source_checkpoint),
                "shared_parameter_count": len(copied),
                "dual_text_projection_initialization": (
                    "deterministic_disjoint_blocks_from_unused_v3_text_projection_v1"
                ),
            },
            "migration_source": {
                "path": str(source_checkpoint),
                "sha256": sha256_file(source_checkpoint),
                "architecture": source["architecture"],
                "global_step": int(source.get("global_step") or 0),
            },
            "dual_text_conditioning_contract": contract,
            "config": dict(source.get("config") or {})
            | {
                "architecture": ULA_MMDIT_V4_DUAL_TEXT_ADALN_ARCHITECTURE,
                **shape,
            },
        }
    )
    _atomic_torch(output_path, payload)
    reloaded = torch.load(output_path, map_location="cpu", weights_only=True)
    validate_checkpoint_contract(reloaded, expected_action_dim=ACTION_DIM)
    return {
        "artifact_kind": "ula_v4_dual_text_migration_receipt_v1",
        "checkpoint": str(output_path),
        "checkpoint_sha256": sha256_file(output_path),
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
        "dual_text_conditioning_contract": contract,
    }


def _orthonormal_projection(seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    matrix = rng.standard_normal((128, 64)).astype(np.float64)
    q, _ = np.linalg.qr(matrix)
    return np.ascontiguousarray(q[:, :64].astype(np.float32))


def _project_role_embeddings(values: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    projected = np.asarray(values, dtype=np.float32) @ matrix
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-8):
        raise ValueError("projected role embedding has zero or invalid norm")
    return np.ascontiguousarray(projected / norms)


def _encode_unique(
    encoder: _Beat2QwenMotionTextEncoder,
    texts: Sequence[str],
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    unique = sorted(set(str(text).strip() for text in texts))
    if not unique or any(not text for text in unique):
        raise ValueError("dual-text cache contains an empty text")
    encoded = np.asarray(encoder.encode(unique, batch_size=int(batch_size)), dtype=np.float32)
    if encoded.shape != (len(unique), 128):
        raise ValueError(f"Qwen encoder returned {encoded.shape}, expected {(len(unique), 128)}")
    return {text: encoded[index] for index, text in enumerate(unique)}


def _read_pairs(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            clip_id = str(value.get("anchor_clip_id") or "")
            if not clip_id or clip_id in result:
                raise ValueError(f"duplicate counterfactual anchor: {clip_id!r}")
            result[clip_id] = value
    return result


def build_dual_text_condition_cache(
    episodes: Sequence[Mapping[str, Any]],
    counterfactual_manifest: str | Path,
    qwen_checkpoint: str | Path,
    generator_checkpoint: str | Path,
    output_path: str | Path,
    *,
    device: str = "auto",
    batch_size: int = 16,
) -> dict[str, Any]:
    output_path = Path(output_path).resolve()
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    if output_path.exists() or metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite dual-text cache: {output_path}")
    counterfactual_manifest = Path(counterfactual_manifest).resolve()
    qwen_checkpoint = Path(qwen_checkpoint).resolve()
    generator_checkpoint = Path(generator_checkpoint).resolve()
    checkpoint = torch.load(generator_checkpoint, map_location="cpu", weights_only=True)
    validate_checkpoint_contract(checkpoint, expected_action_dim=ACTION_DIM)
    contract = checkpoint.get("dual_text_conditioning_contract") or {}
    if (
        checkpoint.get("formal_episode_contract") != FORMAL_EPISODE_CONTRACT
        or contract.get("policy") != DUAL_TEXT_CONTRACT
        or contract.get("qwen_checkpoint_sha256") != sha256_file(qwen_checkpoint)
    ):
        raise ValueError("generator is not bound to the requested dual-text Qwen contract")
    episodes = list(episodes)
    for episode in episodes:
        validate_dialogue_action_v11_episode(episode)
    by_id = {str(episode["clip_id"]): episode for episode in episodes}
    pairs = _read_pairs(counterfactual_manifest)
    if set(by_id) != set(pairs):
        raise ValueError("counterfactual membership differs from action episodes")

    qwen_payload = torch.load(qwen_checkpoint, map_location="cpu", weights_only=True)
    encoder = _Beat2QwenMotionTextEncoder(
        qwen_payload, device=device, local_files_only=True
    )
    directive_raw = _encode_unique(
        encoder,
        [str(row["action_directive_text"]) for row in episodes],
        batch_size=batch_size,
    )
    dialogue_raw = _encode_unique(
        encoder,
        [str(row["dialogue_text"]) for row in episodes],
        batch_size=batch_size,
    )
    directive_matrix = _orthonormal_projection(1111)
    dialogue_matrix = _orthonormal_projection(2222)
    directive_texts = sorted(directive_raw)
    dialogue_texts = sorted(dialogue_raw)
    directive_projected = _project_role_embeddings(
        np.stack([directive_raw[text] for text in directive_texts]), directive_matrix
    )
    dialogue_projected = _project_role_embeddings(
        np.stack([dialogue_raw[text] for text in dialogue_texts]), dialogue_matrix
    )
    directive_latent = {
        text: directive_projected[index] for index, text in enumerate(directive_texts)
    }
    dialogue_latent = {
        text: dialogue_projected[index] for index, text in enumerate(dialogue_texts)
    }

    conditions = np.zeros((len(episodes), KIMODO_V2_CONDITION_DIM), dtype=np.float32)
    negatives = np.zeros((len(episodes), 2, KIMODO_V2_CONDITION_DIM), dtype=np.float32)
    rows = []
    for index, episode in enumerate(episodes):
        clip_id = str(episode["clip_id"])
        directive = str(episode["action_directive_text"])
        dialogue = str(episode["dialogue_text"])
        conditions[index, STYLE_CONTROL_SLICE] = style_controls(episode, clip_id=clip_id)
        conditions[index, DIRECTIVE_LATENT_SLICE] = directive_latent[directive]
        conditions[index, DIALOGUE_LATENT_SLICE] = dialogue_latent[dialogue]
        negatives[index] = conditions[index]
        pair = pairs[clip_id]
        dialogue_source = by_id[pair["dialogue_shuffled"]["source_clip_id"]]
        directive_source = by_id[
            pair["action_directive_shuffled"]["source_clip_id"]
        ]
        negatives[index, 0, DIALOGUE_LATENT_SLICE] = dialogue_latent[
            str(dialogue_source["dialogue_text"])
        ]
        negatives[index, 1, DIRECTIVE_LATENT_SLICE] = directive_latent[
            str(directive_source["action_directive_text"])
        ]
        attached = dict(episode)
        attached["condition"] = conditions[index]
        attached["counterfactual_conditions"] = negatives[index]
        validate_dialogue_action_v11_episode(
            attached, require_attached_condition=True
        )
        rows.append(
            {
                "clip_id": clip_id,
                "trajectory_sha256": episode["trajectory_sha256"],
                "action_directive_text_sha256": episode[
                    "action_directive_text_sha256"
                ],
                "dialogue_text_sha256": episode["dialogue_text_sha256"],
                "counterfactual_record_sha256": pair["record_sha256"],
                "condition_sha256": array_sha256(conditions[index]),
                "counterfactual_conditions_sha256": array_sha256(negatives[index]),
            }
        )

    clip_ids = [str(episode["clip_id"]) for episode in episodes]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            clip_ids=np.asarray(clip_ids),
            conditions=conditions,
            counterfactual_conditions=negatives,
        )
    os.replace(temporary, output_path)
    qwen_receipt = qwen_payload.get("qwen") or {}
    metadata = {
        "schema_version": CONDITION_CACHE_SCHEMA_VERSION,
        "artifact_kind": CONDITION_CACHE_ARTIFACT_KIND,
        "formal_episode_contract": FORMAL_EPISODE_CONTRACT,
        "condition_dim": KIMODO_V2_CONDITION_DIM,
        "count": len(episodes),
        "unsafe_condition_cache": False,
        "cache_sha256": sha256_file(output_path),
        "generator_checkpoint": str(generator_checkpoint),
        "generator_checkpoint_sha256": sha256_file(generator_checkpoint),
        "qwen_checkpoint": str(qwen_checkpoint),
        "qwen_checkpoint_sha256": sha256_file(qwen_checkpoint),
        "qwen_model_name": qwen_receipt.get("model_name"),
        "qwen_revision": qwen_receipt.get("revision"),
        "counterfactual_manifest": str(counterfactual_manifest),
        "counterfactual_manifest_sha256": sha256_file(counterfactual_manifest),
        "dual_text_conditioning_contract_sha256": contract["sha256"],
        "projection_policy": PROJECTION_POLICY,
        "action_directive_projection_sha256": array_sha256(directive_matrix),
        "dialogue_projection_sha256": array_sha256(dialogue_matrix),
        "mapping_head": "absent",
        "role_slices": {
            "action_directive": [DIRECTIVE_LATENT_SLICE.start, DIRECTIVE_LATENT_SLICE.stop],
            "dialogue": [DIALOGUE_LATENT_SLICE.start, DIALOGUE_LATENT_SLICE.stop],
        },
        "episodes": rows,
    }
    _atomic_json(metadata_path, metadata)
    del encoder
    return metadata


def attach_dual_text_condition_cache(
    episodes: Sequence[Mapping[str, Any]], cache_path: str | Path
) -> list[dict[str, Any]]:
    cache_path = Path(cache_path).resolve()
    metadata_path = cache_path.with_suffix(cache_path.suffix + ".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        not isinstance(metadata, dict)
        or metadata.get("artifact_kind") != CONDITION_CACHE_ARTIFACT_KIND
        or metadata.get("formal_episode_contract") != FORMAL_EPISODE_CONTRACT
        or metadata.get("unsafe_condition_cache") is not False
        or metadata.get("cache_sha256") != sha256_file(cache_path)
    ):
        raise ValueError("dual-text condition cache metadata is invalid")
    with np.load(cache_path, allow_pickle=False) as payload:
        if set(payload.files) != {
            "clip_ids",
            "conditions",
            "counterfactual_conditions",
        }:
            raise ValueError("dual-text condition cache arrays are incomplete")
        clip_ids = payload["clip_ids"].astype(str).tolist()
        conditions = payload["conditions"].astype(np.float32)
        negatives = payload["counterfactual_conditions"].astype(np.float32)
    episodes = list(episodes)
    if (
        clip_ids != [str(row["clip_id"]) for row in episodes]
        or conditions.shape != (len(episodes), KIMODO_V2_CONDITION_DIM)
        or negatives.shape != (len(episodes), 2, KIMODO_V2_CONDITION_DIM)
    ):
        raise ValueError("dual-text condition cache shape or order changed")
    metadata_rows = metadata.get("episodes")
    if not isinstance(metadata_rows, list) or len(metadata_rows) != len(episodes):
        raise ValueError("dual-text condition cache episode bindings are missing")
    provenance = dict(metadata) | {
        "path": str(cache_path),
        "metadata_path": str(metadata_path),
    }
    attached = []
    for index, episode in enumerate(episodes):
        expected = {
            "clip_id": str(episode["clip_id"]),
            "trajectory_sha256": episode["trajectory_sha256"],
            "action_directive_text_sha256": episode["action_directive_text_sha256"],
            "dialogue_text_sha256": episode["dialogue_text_sha256"],
            "counterfactual_record_sha256": episode["dialogue_action_alignment"][
                "hard_negative_record_sha256"
            ],
            "condition_sha256": array_sha256(conditions[index]),
            "counterfactual_conditions_sha256": array_sha256(negatives[index]),
        }
        if metadata_rows[index] != expected:
            raise ValueError(f"{episode['clip_id']}: dual-text cache binding changed")
        item = dict(episode)
        item["condition"] = np.ascontiguousarray(conditions[index])
        item["counterfactual_conditions"] = np.ascontiguousarray(negatives[index])
        item["condition_cache_provenance"] = provenance
        validate_dialogue_action_v11_episode(item, require_attached_condition=True)
        attached.append(item)
    return attached


__all__ = [
    "DUAL_TEXT_CONTRACT",
    "MIGRATION_MODE",
    "PROJECTION_POLICY",
    "attach_dual_text_condition_cache",
    "build_dual_text_condition_cache",
    "migrate_v3_foundation_to_v4_dual_text",
]
