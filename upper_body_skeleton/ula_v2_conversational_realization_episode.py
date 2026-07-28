"""Formal variable-length episodes for ordinary conversational gesturing.

This contract trains how a robot moves while speaking without inventing a
communicative intent or emotion.  Its text is derived from an official BEAT2
co-speech event and observable trajectory style; transcripts and filenames are
never semantic supervision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from upper_body_skeleton.robot_observable_motion_realizations import (
    PROVENANCE as REALIZATION_PROMPT_PROVENANCE,
    REALIZATION_ID,
    validate_conversational_realization_annotation,
)
from upper_body_skeleton.ula_training import (
    KIMODO_CONDITION_DIM,
    KIMODO_V2_CONDITION_DIM,
)
from upper_body_skeleton.ula_v2_18d_head import (
    ACTION_DIM,
    STYLE_CONTROL_SLICE,
)


FORMAL_EPISODE_CONTRACT = "ula_v2_18d_conversational_realization_v9_episode_v1"
FORMAL_ELIGIBILITY_MODE = "conversational_realization_v9_train_ready"
TRAINING_SEGMENT_REPRESENTATION = (
    "native_variable_length_conversational_gesturing_30hz_v1"
)
CONDITION_CACHE_ARTIFACT_KIND = "ula_v2_conversational_realization_v9_condition_cache"
CONDITION_CACHE_SCHEMA_VERSION = 1
PROMPT_TEXT_PROVENANCE = REALIZATION_PROMPT_PROVENANCE

_REQUIRED_QUALITY_GATES = {
    "joint_limits_pass",
    "velocity_pass",
    "head_joint_limits_pass",
    "head_velocity_pass",
    "collision_pass",
    "passed",
}
_ZERO_SUPERVISION_FIELDS = {
    "intent_supervision_mask": False,
    "intent_conditioning_mask": False,
    "emotion_supervision_mask": False,
    "emotion_conditioning_mask": False,
    "behavior_supervision_mask": False,
    "audio_conditioning_enabled": False,
}


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            records.append(value)
    if not records:
        raise ValueError(f"conversational realization manifest is empty: {path}")
    return records


def _resolve_path(value: object, *, manifest: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    return (path if path.is_absolute() else manifest.parent / path).resolve()


def is_conversational_realization_v9_episode(episode: Mapping[str, Any]) -> bool:
    return episode.get("formal_episode_contract") == FORMAL_EPISODE_CONTRACT


def _style_controls(episode: Mapping[str, Any], *, clip_id: str) -> np.ndarray:
    style = _require_mapping(episode.get("motion_style"), f"{clip_id}.motion_style")
    controls = _require_mapping(style.get("style_controls"), f"{clip_id}.style_controls")
    if set(controls) != {"amplitude", "tempo", "energy"}:
        raise ValueError(f"{clip_id}: style controls must be amplitude/tempo/energy")
    vector = np.asarray(
        [controls["amplitude"], controls["tempo"], controls["energy"]],
        dtype=np.float32,
    )
    if vector.shape != (3,) or not np.isfinite(vector).all() or np.any(np.abs(vector) > 1.0):
        raise ValueError(f"{clip_id}: style controls must be finite values in [-1, 1]")
    return vector


def _validate_condition(
    episode: Mapping[str, Any], *, clip_id: str, require_attached_condition: bool
) -> None:
    condition = episode.get("condition")
    if condition is None:
        if require_attached_condition:
            raise ValueError(f"{clip_id}: attached conversational condition is required")
        return
    vector = np.asarray(condition, dtype=np.float32)
    if vector.shape != (KIMODO_V2_CONDITION_DIM,) or not np.isfinite(vector).all():
        raise ValueError(
            f"{clip_id}: condition must be finite {KIMODO_V2_CONDITION_DIM}D"
        )
    allowed = np.zeros(KIMODO_V2_CONDITION_DIM, dtype=np.bool_)
    allowed[STYLE_CONTROL_SLICE] = True
    allowed[KIMODO_CONDITION_DIM:] = True
    if np.any(vector[~allowed] != 0.0):
        raise ValueError(f"{clip_id}: intent, emotion, behavior and legacy channels must be zero")
    if not np.array_equal(vector[STYLE_CONTROL_SLICE], _style_controls(episode, clip_id=clip_id)):
        raise ValueError(f"{clip_id}: attached trajectory-style controls changed")
    latent = vector[KIMODO_CONDITION_DIM:]
    if not math.isclose(float(np.linalg.norm(latent)), 1.0, rel_tol=0.0, abs_tol=1e-4):
        raise ValueError(f"{clip_id}: conversational Qwen latent must be L2-normalized")


def validate_conversational_realization_v9_episode(
    episode: Mapping[str, Any], *, require_attached_condition: bool = False
) -> dict[str, Any]:
    episode = _require_mapping(episode, "episode")
    clip_id = str(episode.get("clip_id") or "").strip()
    if not clip_id:
        raise ValueError("conversational realization episode is missing clip_id")
    if not is_conversational_realization_v9_episode(episode):
        raise ValueError(f"{clip_id}: formal episode contract is not conversational v9")
    if episode.get("accepted_for_training") is not True or episode.get(
        "eligibility_mode"
    ) != FORMAL_ELIGIBILITY_MODE:
        raise ValueError(f"{clip_id}: conversational realization is not train-ready")
    if episode.get("native_variable_length") is not True:
        raise ValueError(f"{clip_id}: native_variable_length must be true")

    realization = _require_mapping(episode.get("motion_realization"), "motion_realization")
    validate_conversational_realization_annotation(realization)
    prompt = str(episode.get("prompt") or "").strip()
    expected_prompt = str((realization.get("motion_realization_prompt") or {}).get("en") or "")
    if prompt != expected_prompt or not prompt:
        raise ValueError(f"{clip_id}: prompt must equal the verified realization prompt")
    if episode.get("prompt_sha256") != hashlib.sha256(prompt.encode("utf-8")).hexdigest():
        raise ValueError(f"{clip_id}: prompt SHA256 changed")
    if episode.get("prompt_text_provenance") != PROMPT_TEXT_PROVENANCE:
        raise ValueError(f"{clip_id}: prompt provenance is invalid")
    if episode.get("observable_intent_id") is not None or episode.get("emotion_id") is not None:
        raise ValueError(f"{clip_id}: ordinary speaking may not imply intent or emotion")
    for field, expected in _ZERO_SUPERVISION_FIELDS.items():
        if episode.get(field) is not expected:
            raise ValueError(f"{clip_id}: {field} must remain {expected}")
    if episode.get("motion_realization_supervision_mask") is not True:
        raise ValueError(f"{clip_id}: motion realization supervision must be enabled")

    actions = np.asarray(episode.get("actions"), dtype=np.float32)
    if (
        actions.ndim != 2
        or actions.shape[0] < 3
        or actions.shape[1] != ACTION_DIM
        or not np.isfinite(actions).all()
    ):
        raise ValueError(f"{clip_id}: actions must be finite [frames, {ACTION_DIM}]")
    action_dim_mask = np.asarray(
        episode.get("action_dim_mask", np.ones(ACTION_DIM, dtype=np.bool_)),
        dtype=np.bool_,
    )
    if action_dim_mask.shape != (ACTION_DIM,) or not action_dim_mask.all():
        raise ValueError(f"{clip_id}: all 18 robot joints must be observed")
    fps = episode.get("fps")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isclose(float(fps), 30.0, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise ValueError(f"{clip_id}: conversational realization must be exactly 30 Hz")

    segment = _require_mapping(episode.get("training_segment"), "training_segment")
    if segment.get("representation") != TRAINING_SEGMENT_REPRESENTATION:
        raise ValueError(f"{clip_id}: training segment representation is invalid")
    if segment.get("fixed_window_sec") is not None or segment.get("cropped") is not False:
        raise ValueError(f"{clip_id}: fixed-window or cropped training is forbidden")
    start, end, source_frames = (
        segment.get("start_frame"),
        segment.get("end_frame_exclusive"),
        segment.get("frame_count"),
    )
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (start, end, source_frames)
        )
        or start < 0
        or end <= start
        or source_frames != end - start
        or segment.get("output_frame_count") != int(actions.shape[0])
    ):
        raise ValueError(f"{clip_id}: native variable-length interval is inconsistent")
    retarget = _require_mapping(episode.get("retarget_segment"), "retarget_segment")
    if (
        retarget.get("source_start_frame") != start
        or retarget.get("source_end_frame_exclusive") != end
        or retarget.get("source_frame_count") != source_frames
        or retarget.get("output_frame_count") != int(actions.shape[0])
        or retarget.get("cropped") is not False
        or retarget.get("fps") != 30.0
    ):
        raise ValueError(f"{clip_id}: retarget source/output time axes are inconsistent")
    expected_duration = (actions.shape[0] - 1) / float(fps)
    if not math.isclose(
        float(episode.get("duration_sec", -1.0)), expected_duration, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError(f"{clip_id}: duration must use the (N-1)/fps sample-span axis")

    quality = _require_mapping(episode.get("quality_gate"), "quality_gate")
    if not _REQUIRED_QUALITY_GATES.issubset(quality) or any(
        quality.get(gate) is not True for gate in _REQUIRED_QUALITY_GATES
    ):
        raise ValueError(f"{clip_id}: required robot physical quality gates must pass")
    if episode.get("retarget_qc_passed") is not True:
        raise ValueError(f"{clip_id}: retarget QC must be passed")
    for field in (
        "trajectory_sha256",
        "source_manifest_sha256",
        "source_record_sha256",
        "realization_manifest_sha256",
        "realization_record_sha256",
    ):
        if not _is_sha256(episode.get(field)):
            raise ValueError(f"{clip_id}: {field} must be a SHA256")
    for field in ("source_clip_id", "speaker_key", "source_group_key", "dataset_source"):
        if not str(episode.get(field) or "").strip():
            raise ValueError(f"{clip_id}: {field} is required")

    admission = _require_mapping(episode.get("training_admission"), "training_admission")
    expected_admission = {
        "contract": FORMAL_EPISODE_CONTRACT,
        "trajectory_sha256": episode["trajectory_sha256"],
        "source_record_sha256": episode["source_record_sha256"],
        "realization_record_sha256": episode["realization_record_sha256"],
        "motion_realization_ontology_sha256": realization[
            "motion_realization_ontology_sha256"
        ],
        "motion_realization_id": REALIZATION_ID,
        "training_channel_masks": {
            "motion": True,
            "motion_realization": True,
            "primary_intent": False,
            "emotion": False,
            "audio": False,
        },
    }
    if admission != expected_admission:
        raise ValueError(f"{clip_id}: training admission hash binding is invalid")
    _style_controls(episode, clip_id=clip_id)
    _validate_condition(
        episode, clip_id=clip_id, require_attached_condition=require_attached_condition
    )
    return {
        "clip_id": clip_id,
        "formal_episode_contract": FORMAL_EPISODE_CONTRACT,
        "frame_count": int(actions.shape[0]),
        "motion_realization_id": REALIZATION_ID,
    }


def load_conversational_realization_v9_episodes(
    manifest: str | Path,
) -> list[dict[str, Any]]:
    from upper_body_skeleton.ula_v2_18d_head import read_joint_csv

    manifest = Path(manifest).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"conversational realization manifest is missing: {manifest}")
    manifest_sha256 = _sha256_file(manifest)
    records = _read_jsonl(manifest)
    if not all(is_conversational_realization_v9_episode(record) for record in records):
        raise ValueError("conversational loader refuses mixed or unmarked contracts")
    episodes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        clip_id = str(record.get("clip_id") or "").strip()
        if not clip_id or clip_id in seen:
            raise ValueError(f"missing or duplicate conversational clip_id: {clip_id!r}")
        motion = _require_mapping(record.get("motion_18d"), f"{clip_id}.motion_18d")
        motion_path = _resolve_path(
            motion.get("safe_csv") or record.get("trajectory_path"), manifest=manifest
        )
        if motion_path is None or not motion_path.is_file():
            raise FileNotFoundError(f"{clip_id}: 18D trajectory is missing: {motion_path}")
        values = read_joint_csv(motion_path)
        trajectory_sha256 = _sha256_file(motion_path)
        if trajectory_sha256 != motion.get("safe_csv_sha256") or trajectory_sha256 != record.get(
            "trajectory_sha256"
        ):
            raise ValueError(f"{clip_id}: trajectory SHA256 changed")
        if int(motion.get("frames", -1)) != values.shape[0] or int(
            motion.get("csv_rows", -1)
        ) != values.shape[0]:
            raise ValueError(f"{clip_id}: 18D row count changed")
        episode = deepcopy(record)
        episode.update(
            {
                "actions": np.ascontiguousarray(values),
                "action_dim_mask": np.ones(ACTION_DIM, dtype=np.bool_),
                "condition": None,
                "fps": 30.0,
                "duration_sec": float((values.shape[0] - 1) / 30.0),
                "trajectory_path": str(motion_path),
                "trajectory_sha256": trajectory_sha256,
                "loaded_manifest": str(manifest),
                "loaded_manifest_sha256": manifest_sha256,
                "retarget_qc_passed": motion.get("state") == "passed",
            }
        )
        validate_conversational_realization_v9_episode(episode)
        episodes.append(episode)
        seen.add(clip_id)
    return episodes


def build_conversational_realization_v9_condition_vectors(
    episodes: Sequence[Mapping[str, Any]], *, text_latents: Mapping[str, np.ndarray]
) -> np.ndarray:
    if not episodes:
        raise ValueError("conversational realization condition cache requires episodes")
    conditions = np.zeros((len(episodes), KIMODO_V2_CONDITION_DIM), dtype=np.float32)
    expected_ids: set[str] = set()
    for row, episode in enumerate(episodes):
        report = validate_conversational_realization_v9_episode(episode)
        clip_id = report["clip_id"]
        expected_ids.add(clip_id)
        conditions[row, STYLE_CONTROL_SLICE] = _style_controls(episode, clip_id=clip_id)
        latent = np.asarray(text_latents.get(clip_id), dtype=np.float32)
        expected_shape = (KIMODO_V2_CONDITION_DIM - KIMODO_CONDITION_DIM,)
        if latent.shape != expected_shape or not np.isfinite(latent).all():
            raise ValueError(f"{clip_id}: Qwen latent must be finite {expected_shape}")
        norm = float(np.linalg.norm(latent))
        if not math.isfinite(norm) or norm <= 1e-8:
            raise ValueError(f"{clip_id}: Qwen latent has zero or invalid norm")
        conditions[row, KIMODO_CONDITION_DIM:] = latent / norm
        attached = dict(episode)
        attached["condition"] = conditions[row]
        validate_conversational_realization_v9_episode(
            attached, require_attached_condition=True
        )
    if set(text_latents) != expected_ids:
        raise ValueError("conversational Qwen latent membership differs from episodes")
    return conditions


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class _Beat2QwenMotionTextEncoder:
    """Read the existing BEAT2-only LoRA/text head without training new weights."""

    def __init__(self, checkpoint: Mapping[str, Any], *, device: str, local_files_only: bool):
        import torch
        from peft import set_peft_model_state_dict
        from tools import train_beat2_qwen_motion_alignment as beat2_qwen

        if (
            checkpoint.get("artifact_kind") != "beat2_qwen_lora_alignment_v1"
            or checkpoint.get("variant") != "lora_finetuned"
            or checkpoint.get("no_kimodo") is not True
            or checkpoint.get("data_policy")
            != "beat2_only_no_external_motion_dataset_v1"
        ):
            raise ValueError("conversational v9 requires the BEAT2-only no-Kimodo Qwen LoRA")
        resolved_device = torch.device(
            device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        qwen_receipt = _require_mapping(checkpoint.get("qwen"), "qwen")
        alignment = _require_mapping(checkpoint.get("alignment_config"), "alignment_config")
        config = {
            "model_name": qwen_receipt["model_name"],
            "revision": qwen_receipt["revision"],
            "local_files_only": bool(local_files_only),
            "attention_backend": "eager",
            "qwen_component_dim": int(checkpoint["qwen_component_dim"]),
            "alignment": dict(alignment),
        }
        base_qwen, tokenizer, metadata = beat2_qwen._load_official_qwen_base(
            config, device=resolved_device
        )
        qwen, _ = beat2_qwen._apply_beat2_lora(base_qwen, config, metadata)
        result = set_peft_model_state_dict(qwen, checkpoint["qwen_lora_state_dict"])
        if getattr(result, "unexpected_keys", None):
            raise ValueError(f"unexpected BEAT2 Qwen LoRA keys: {result.unexpected_keys}")
        state = _require_mapping(checkpoint.get("text_head_state_dict"), "text_head_state_dict")
        label_sizes = {
            name: int(state[f"{name}_head.weight"].shape[0])
            for name in ("category", "intensity", "emotion")
        }
        head = beat2_qwen.TextAlignmentHead(
            int(checkpoint["qwen_component_dim"]),
            int(alignment["hidden_dim"]),
            int(checkpoint["latent_dim"]),
            label_sizes,
            dropout=float(alignment["dropout"]),
        ).to(resolved_device)
        head.load_state_dict(state, strict=True)
        self.qwen = qwen.requires_grad_(False).eval()
        self.head = head.requires_grad_(False).eval()
        self.tokenizer = tokenizer
        self.device = resolved_device
        self.instruction = str(checkpoint["instruction"])
        self.component_dim = int(checkpoint["qwen_component_dim"])
        self.max_length = 96
        self._beat2_qwen = beat2_qwen

    def encode(self, texts: Sequence[str], *, batch_size: int = 16) -> np.ndarray:
        import torch

        prompts = [str(text).strip() for text in texts]
        if not prompts or any(not prompt for prompt in prompts):
            raise ValueError("BEAT2 Qwen realization prompts must be non-empty")
        encoded: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(prompts), int(batch_size)):
                tokens = self._beat2_qwen.tokenize_prompts(
                    self.tokenizer,
                    prompts[start : start + int(batch_size)],
                    instruction=self.instruction,
                    max_length=self.max_length,
                    device=self.device,
                )
                components = self._beat2_qwen._qwen_components(
                    self.qwen, tokens, component_dim=self.component_dim
                )
                encoded.append(
                    self.head(components)["embedding"].float().cpu().numpy()
                )
        return np.concatenate(encoded, axis=0).astype(np.float32)


def build_conversational_realization_v9_condition_cache(
    episodes: Sequence[Mapping[str, Any]],
    qwen_checkpoint: str | Path,
    output_path: str | Path,
    *,
    base_checkpoint: str | Path,
    device: str = "auto",
    local_files_only: bool = True,
    batch_size: int = 16,
) -> dict[str, Any]:
    import torch

    from upper_body_skeleton.ula_v2_18d_head import (
        sha256_file,
        validate_checkpoint_contract,
        validate_qwen_checkpoint_for_generator,
    )

    output_path = Path(output_path).resolve()
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    if output_path.suffix != ".npz":
        raise ValueError("conversational condition cache must use the .npz suffix")
    if output_path.exists() or metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite condition cache: {output_path}")
    base_checkpoint = Path(base_checkpoint).resolve()
    qwen_checkpoint = Path(qwen_checkpoint).resolve()
    checkpoint = torch.load(base_checkpoint, map_location="cpu", weights_only=True)
    validate_checkpoint_contract(checkpoint, expected_action_dim=ACTION_DIM)
    if checkpoint.get("formal_episode_contract") != FORMAL_EPISODE_CONTRACT:
        raise ValueError("condition cache target is not a conversational v9 checkpoint")
    validate_qwen_checkpoint_for_generator(checkpoint, qwen_checkpoint)
    qwen_payload = torch.load(qwen_checkpoint, map_location="cpu", weights_only=True)
    if qwen_payload.get("artifact_kind") == "beat2_qwen_lora_alignment_v1":
        encoder = _Beat2QwenMotionTextEncoder(
            qwen_payload, device=device, local_files_only=local_files_only
        )
    else:
        from upper_body_skeleton.cross_modal_latent import load_qwen_motion_text_encoder

        encoder, qwen_payload = load_qwen_motion_text_encoder(
            qwen_checkpoint, device=device, local_files_only=local_files_only
        )
    encoded = np.asarray(
        encoder.encode([str(episode["prompt"]) for episode in episodes], batch_size=int(batch_size)),
        dtype=np.float32,
    )
    expected_shape = (len(episodes), KIMODO_V2_CONDITION_DIM - KIMODO_CONDITION_DIM)
    if encoded.shape != expected_shape:
        raise ValueError(f"Qwen encoder returned {encoded.shape}, expected {expected_shape}")
    text_latents = {
        str(episode["clip_id"]): encoded[index]
        for index, episode in enumerate(episodes)
    }
    conditions = build_conversational_realization_v9_condition_vectors(
        episodes, text_latents=text_latents
    )
    clip_ids = [str(episode["clip_id"]) for episode in episodes]
    prompts = [str(episode["prompt"]) for episode in episodes]
    style_controls = np.stack(
        [_style_controls(episode, clip_id=str(episode["clip_id"])) for episode in episodes]
    ).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            clip_ids=np.asarray(clip_ids),
            prompts=np.asarray(prompts),
            conditions=conditions,
            style_controls=style_controls,
        )
    os.replace(temporary, output_path)
    qwen = qwen_payload.get("qwen") or {}
    metadata = {
        "schema_version": CONDITION_CACHE_SCHEMA_VERSION,
        "artifact_kind": CONDITION_CACHE_ARTIFACT_KIND,
        "formal_episode_contract": FORMAL_EPISODE_CONTRACT,
        "condition_dim": KIMODO_V2_CONDITION_DIM,
        "count": len(episodes),
        "unsafe_condition_cache": False,
        "cache_sha256": sha256_file(output_path),
        "qwen_checkpoint": str(qwen_checkpoint),
        "qwen_checkpoint_sha256": sha256_file(qwen_checkpoint),
        "qwen_model_name": qwen.get("model_name"),
        "qwen_revision": qwen.get("revision"),
        "generator_checkpoint": str(base_checkpoint),
        "generator_checkpoint_sha256": sha256_file(base_checkpoint),
        "motion_realization_ontology_sha256": episodes[0]["motion_realization"][
            "motion_realization_ontology_sha256"
        ],
        "episodes": [
            {
                "clip_id": str(episode["clip_id"]),
                "prompt_sha256": episode["prompt_sha256"],
                "trajectory_sha256": episode["trajectory_sha256"],
                "realization_record_sha256": episode["realization_record_sha256"],
                "condition_sha256": _array_sha256(conditions[index]),
                "style_controls": style_controls[index].tolist(),
            }
            for index, episode in enumerate(episodes)
        ],
    }
    _atomic_json(metadata_path, metadata)
    del encoder
    return metadata


def attach_conversational_realization_v9_condition_cache(
    episodes: Sequence[Mapping[str, Any]], cache_path: str | Path
) -> list[dict[str, Any]]:
    from upper_body_skeleton.ula_v2_18d_head import sha256_file

    cache_path = Path(cache_path).resolve()
    metadata_path = cache_path.with_suffix(cache_path.suffix + ".json")
    if not cache_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"conversational condition cache is missing: {cache_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        not isinstance(metadata, dict)
        or metadata.get("artifact_kind") != CONDITION_CACHE_ARTIFACT_KIND
        or metadata.get("formal_episode_contract") != FORMAL_EPISODE_CONTRACT
        or metadata.get("unsafe_condition_cache") is not False
        or metadata.get("cache_sha256") != sha256_file(cache_path)
    ):
        raise ValueError("conversational condition cache metadata is invalid")
    with np.load(cache_path, allow_pickle=False) as payload:
        if set(payload.files) != {"clip_ids", "prompts", "conditions", "style_controls"}:
            raise ValueError("conversational condition cache arrays are incomplete")
        clip_ids = payload["clip_ids"].astype(str).tolist()
        prompts = payload["prompts"].astype(str).tolist()
        conditions = payload["conditions"].astype(np.float32)
        style_controls = payload["style_controls"].astype(np.float32)
    count = len(episodes)
    if (
        metadata.get("count") != count
        or clip_ids != [str(episode["clip_id"]) for episode in episodes]
        or conditions.shape != (count, KIMODO_V2_CONDITION_DIM)
        or style_controls.shape != (count, 3)
    ):
        raise ValueError("conversational condition cache shape or order changed")
    rows = metadata.get("episodes")
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError("conversational cache episode bindings are missing")
    provenance = dict(metadata) | {
        "path": str(cache_path),
        "metadata_path": str(metadata_path),
    }
    attached: list[dict[str, Any]] = []
    for index, episode in enumerate(episodes):
        validate_conversational_realization_v9_episode(episode)
        clip_id = str(episode["clip_id"])
        if prompts[index] != episode["prompt"] or not np.array_equal(
            style_controls[index], _style_controls(episode, clip_id=clip_id)
        ):
            raise ValueError(f"{clip_id}: cached prompt or style changed")
        expected_row = {
            "clip_id": clip_id,
            "prompt_sha256": episode["prompt_sha256"],
            "trajectory_sha256": episode["trajectory_sha256"],
            "realization_record_sha256": episode["realization_record_sha256"],
            "condition_sha256": _array_sha256(conditions[index]),
            "style_controls": style_controls[index].tolist(),
        }
        if rows[index] != expected_row:
            raise ValueError(f"{clip_id}: cached hash binding changed")
        item = dict(episode)
        item["condition"] = np.ascontiguousarray(conditions[index])
        item["condition_cache_provenance"] = provenance
        validate_conversational_realization_v9_episode(
            item, require_attached_condition=True
        )
        attached.append(item)
    return attached


__all__ = [
    "CONDITION_CACHE_ARTIFACT_KIND",
    "CONDITION_CACHE_SCHEMA_VERSION",
    "FORMAL_ELIGIBILITY_MODE",
    "FORMAL_EPISODE_CONTRACT",
    "PROMPT_TEXT_PROVENANCE",
    "TRAINING_SEGMENT_REPRESENTATION",
    "attach_conversational_realization_v9_condition_cache",
    "build_conversational_realization_v9_condition_cache",
    "build_conversational_realization_v9_condition_vectors",
    "is_conversational_realization_v9_episode",
    "load_conversational_realization_v9_episodes",
    "validate_conversational_realization_v9_episode",
]
