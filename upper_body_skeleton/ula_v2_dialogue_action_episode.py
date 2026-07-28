"""Formal BEAT2 episodes for directive-controlled 18D actions with dialogue."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from upper_body_skeleton.ula_training import (
    KIMODO_CONDITION_DIM,
    KIMODO_V2_CONDITION_DIM,
)
from upper_body_skeleton.ula_v2_18d_head import ACTION_DIM, STYLE_CONTROL_SLICE
from upper_body_skeleton.beat2_observable_action_summary import (
    SUMMARY_SOURCE as ACTION_SUMMARY_SOURCE,
    validate_action_summary,
)


FORMAL_EPISODE_CONTRACT = "ula_v2_18d_dialogue_action_v11_episode_v1"
FORMAL_ELIGIBILITY_MODE = "dialogue_action_v11_train_ready"
ARTIFACT_KIND = "ula_v2_dialogue_action_v11_train_episode"
TRAINING_SEGMENT_REPRESENTATION = "native_variable_length_dialogue_action_18d_30hz_v1"
ACTION_ROLE = "robot_brain_complete_upper_body_action_directive_primary_control"
DIALOGUE_ROLE = "moving_speaker_utterance_auxiliary_action_context"
ACTION_SUPERVISION_SCOPE = "complete_observed_robot_upper_body_18d_motion_sequence"
CONDITION_CACHE_ARTIFACT_KIND = "ula_v2_dialogue_action_v11_dual_text_condition_cache"
CONDITION_CACHE_SCHEMA_VERSION = 1
DIRECTIVE_LATENT_SLICE = slice(KIMODO_CONDITION_DIM, KIMODO_CONDITION_DIM + 64)
DIALOGUE_LATENT_SLICE = slice(KIMODO_CONDITION_DIM + 64, KIMODO_V2_CONDITION_DIM)


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def action_directive_from_motion_style(episode: Mapping[str, Any]) -> str:
    style = _mapping(episode.get("motion_style"), "motion_style")
    amplitude = str(style.get("arm_amplitude") or "moderate").replace("_", " ")
    laterality = str(style.get("laterality") or "both").replace("_", " ")
    pace = str(style.get("pace") or "steady").replace("_", " ")
    head = str(style.get("head_engagement") or "natural").replace("_", " ")
    return (
        "While speaking, perform a continuous 18-DoF upper-body action: "
        f"move {laterality} arms with {amplitude} range at a {pace} pace, "
        f"with {head} head motion."
    )


def style_controls(episode: Mapping[str, Any], *, clip_id: str) -> np.ndarray:
    style = _mapping(episode.get("motion_style"), f"{clip_id}.motion_style")
    controls = _mapping(style.get("style_controls"), f"{clip_id}.style_controls")
    vector = np.asarray(
        [controls.get("amplitude"), controls.get("tempo"), controls.get("energy")],
        dtype=np.float32,
    )
    if vector.shape != (3,) or not np.isfinite(vector).all() or np.any(np.abs(vector) > 1.0):
        raise ValueError(f"{clip_id}: style controls must be finite values in [-1, 1]")
    return vector


def is_dialogue_action_v11_episode(episode: Mapping[str, Any]) -> bool:
    return episode.get("formal_episode_contract") == FORMAL_EPISODE_CONTRACT


def validate_dialogue_action_v11_episode(
    episode: Mapping[str, Any], *, require_attached_condition: bool = False
) -> dict[str, Any]:
    episode = _mapping(episode, "episode")
    clip_id = str(episode.get("clip_id") or "").strip()
    if not clip_id:
        raise ValueError("dialogue/action episode is missing clip_id")
    if (
        episode.get("artifact_kind") != ARTIFACT_KIND
        or episode.get("formal_episode_contract") != FORMAL_EPISODE_CONTRACT
        or episode.get("eligibility_mode") != FORMAL_ELIGIBILITY_MODE
        or episode.get("accepted_for_training") is not True
    ):
        raise ValueError(f"{clip_id}: dialogue/action release contract changed")
    if episode.get("native_variable_length") is not True:
        raise ValueError(f"{clip_id}: native_variable_length must be true")

    directive = str(episode.get("action_directive_text") or "").strip()
    dialogue = str(episode.get("dialogue_text") or "").strip()
    if not directive or not dialogue:
        raise ValueError(f"{clip_id}: action directive and dialogue must be non-empty")
    if "gesture" in directive.casefold():
        raise ValueError(f"{clip_id}: active action directive may not narrow motion to gesture")
    if episode.get("prompt") != directive:
        raise ValueError(f"{clip_id}: compatibility prompt must equal action directive")
    if episode.get("action_directive_text_sha256") != text_sha256(directive):
        raise ValueError(f"{clip_id}: action directive hash changed")
    if episode.get("dialogue_text_sha256") != text_sha256(dialogue):
        raise ValueError(f"{clip_id}: dialogue hash changed")
    if (
        episode.get("action_supervision_scope") != ACTION_SUPERVISION_SCOPE
        or episode.get("complete_18d_action_supervision_mask") is not True
        or episode.get("self_speech_action_context_mask") is not True
        or episode.get("partner_response_supervision_mask") is not False
    ):
        raise ValueError(f"{clip_id}: complete-action supervision masks changed")
    if episode.get("source_transcript_used_as_action_or_emotion_label") is not False:
        raise ValueError(f"{clip_id}: transcript may not become an action/emotion label")
    action_contract = _mapping(episode.get("action_directive_contract"), "action_directive_contract")
    contract_source = action_contract.get("source")
    if contract_source not in {
        "verified_18d_trajectory_style_action_directive_v1",
        ACTION_SUMMARY_SOURCE,
    } or action_contract != {
        "role": ACTION_ROLE,
        "primary_control": True,
        "derived_from_dialogue_text": False,
        "supervision_scope": ACTION_SUPERVISION_SCOPE,
        "source": contract_source,
    }:
        raise ValueError(f"{clip_id}: action directive role changed")
    action_summary = episode.get("action_summary")
    if contract_source == ACTION_SUMMARY_SOURCE:
        if not isinstance(action_summary, Mapping):
            raise ValueError(f"{clip_id}: named action directive requires action_summary")
        validate_action_summary(action_summary)
        if directive != action_summary.get("prompt_en"):
            raise ValueError(f"{clip_id}: action directive differs from action summary")
    elif action_summary is not None:
        raise ValueError(f"{clip_id}: legacy style directive may not carry action_summary")
    dialogue_contract = _mapping(episode.get("dialogue_contract"), "dialogue_contract")
    if (
        dialogue_contract.get("role") != DIALOGUE_ROLE
        or dialogue_contract.get("auxiliary_context") is not True
        or dialogue_contract.get("partner_response_supervision") is not False
        or dialogue_contract.get("action_label_supervision") is not False
    ):
        raise ValueError(f"{clip_id}: dialogue role changed")
    alignment = _mapping(episode.get("dialogue_action_alignment"), "dialogue_action_alignment")
    if (
        alignment.get("positive_pair") is not True
        or alignment.get("hard_negative_required") is not True
        or not str(alignment.get("hard_negative_record_sha256") or "")
    ):
        raise ValueError(f"{clip_id}: dialogue/action alignment changed")

    segment = _mapping(episode.get("training_segment"), "training_segment")
    if (
        segment.get("representation") != TRAINING_SEGMENT_REPRESENTATION
        or segment.get("fixed_window_sec") is not None
        or segment.get("cropped") is not False
    ):
        raise ValueError(f"{clip_id}: fixed-window or cropped action training is forbidden")
    frames = int(episode.get("frames", -1))
    output_frames = int(segment.get("output_frame_count", -1))
    if frames < 3 or output_frames != frames:
        raise ValueError(f"{clip_id}: output action frame count changed")
    if not math.isclose(float(episode.get("fps", 0.0)), 30.0, abs_tol=1e-9):
        raise ValueError(f"{clip_id}: action data must be exactly 30 Hz")

    condition = episode.get("condition")
    negatives = episode.get("counterfactual_conditions")
    if condition is None:
        if require_attached_condition:
            raise ValueError(f"{clip_id}: attached dual-text condition is required")
    else:
        vector = np.asarray(condition, dtype=np.float32)
        negative_vectors = np.asarray(negatives, dtype=np.float32)
        if vector.shape != (KIMODO_V2_CONDITION_DIM,) or not np.isfinite(vector).all():
            raise ValueError(f"{clip_id}: condition must be finite 264D")
        if negative_vectors.shape != (2, KIMODO_V2_CONDITION_DIM) or not np.isfinite(
            negative_vectors
        ).all():
            raise ValueError(f"{clip_id}: two finite 264D counterfactuals are required")
        allowed = np.zeros(KIMODO_V2_CONDITION_DIM, dtype=np.bool_)
        allowed[STYLE_CONTROL_SLICE] = True
        allowed[DIRECTIVE_LATENT_SLICE] = True
        allowed[DIALOGUE_LATENT_SLICE] = True
        if np.any(vector[~allowed] != 0.0) or np.any(negative_vectors[:, ~allowed] != 0.0):
            raise ValueError(f"{clip_id}: unused condition channels must remain zero")
        if not np.array_equal(vector[STYLE_CONTROL_SLICE], style_controls(episode, clip_id=clip_id)):
            raise ValueError(f"{clip_id}: trajectory style controls changed")
        if not np.all(negative_vectors[:, STYLE_CONTROL_SLICE] == vector[STYLE_CONTROL_SLICE]):
            raise ValueError(f"{clip_id}: counterfactuals must preserve trajectory style")
        for latent_slice in (DIRECTIVE_LATENT_SLICE, DIALOGUE_LATENT_SLICE):
            if not math.isclose(
                float(np.linalg.norm(vector[latent_slice])), 1.0, abs_tol=1e-4
            ):
                raise ValueError(f"{clip_id}: role text latent must be L2-normalized")
        if np.array_equal(vector, negative_vectors[0]) or np.array_equal(vector, negative_vectors[1]):
            raise ValueError(f"{clip_id}: counterfactual condition equals the positive")
    return {"clip_id": clip_id, "frames": frames}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    return rows


def load_dialogue_action_v11_records(manifest: str | Path) -> list[dict[str, Any]]:
    manifest = Path(manifest).resolve()
    rows = _read_jsonl(manifest)
    for row in rows:
        validate_dialogue_action_v11_episode(row)
    return rows


def _prefer_local_rclone_cache(path: Path) -> Path:
    nas_root = Path("/home/gez/nas")
    cache_root = Path(
        "/home/gez/shuaiwang/.cache/rclone/vfs/ula_nas/xdream"
    )
    try:
        relative = path.relative_to(nas_root)
    except ValueError:
        return path.resolve()
    candidate = cache_root / relative
    if candidate.is_file():
        return candidate.resolve()
    return path.resolve()


def load_dialogue_action_v11_episodes(manifest: str | Path) -> list[dict[str, Any]]:
    from upper_body_skeleton.ula_v2_18d_head import read_joint_csv

    manifest = Path(manifest).resolve()
    manifest_hash = sha256_file(manifest)
    episodes = []
    seen = set()
    for record in load_dialogue_action_v11_records(manifest):
        validate_dialogue_action_v11_episode(record)
        clip_id = str(record["clip_id"])
        if clip_id in seen:
            raise ValueError(f"duplicate dialogue/action clip_id: {clip_id}")
        motion_path = _prefer_local_rclone_cache(Path(str(record["trajectory_path"])))
        values = read_joint_csv(motion_path)
        if values.shape != (int(record["frames"]), ACTION_DIM):
            raise ValueError(f"{clip_id}: 18D trajectory shape changed")
        if sha256_file(motion_path) != record["trajectory_sha256"]:
            raise ValueError(f"{clip_id}: 18D trajectory hash changed")
        episode = deepcopy(record)
        episode.update(
            {
                "actions": np.ascontiguousarray(values),
                "action_dim_mask": np.ones(ACTION_DIM, dtype=np.bool_),
                "condition": None,
                "counterfactual_conditions": None,
                "duration_sec": float((values.shape[0] - 1) / 30.0),
                "loaded_manifest": str(manifest),
                "loaded_manifest_sha256": manifest_hash,
            }
        )
        validate_dialogue_action_v11_episode(episode)
        episodes.append(episode)
        seen.add(clip_id)
    return episodes


__all__ = [
    "ACTION_ROLE",
    "ACTION_SUPERVISION_SCOPE",
    "ARTIFACT_KIND",
    "CONDITION_CACHE_ARTIFACT_KIND",
    "CONDITION_CACHE_SCHEMA_VERSION",
    "DIALOGUE_LATENT_SLICE",
    "DIALOGUE_ROLE",
    "DIRECTIVE_LATENT_SLICE",
    "FORMAL_ELIGIBILITY_MODE",
    "FORMAL_EPISODE_CONTRACT",
    "TRAINING_SEGMENT_REPRESENTATION",
    "action_directive_from_motion_style",
    "array_sha256",
    "canonical_sha256",
    "is_dialogue_action_v11_episode",
    "load_dialogue_action_v11_episodes",
    "load_dialogue_action_v11_records",
    "sha256_file",
    "style_controls",
    "text_sha256",
    "validate_dialogue_action_v11_episode",
]
