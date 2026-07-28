"""Dataset contract for directive-controlled motion with dialogue context.

The directive is the primary robot control.  BEAT2 speech is an auxiliary
co-speech context channel and is never promoted to an action, intent, emotion,
or listener-response label.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from typing import Any


FORMAL_EPISODE_CONTRACT = "ula_v2_18d_dialogue_directive_v10_episode_v1"
FORMAL_ELIGIBILITY_MODE = "dialogue_directive_v10_train_ready"
ARTIFACT_KIND = "ula_v2_dialogue_directive_v10_train_episode"
TRAINING_SEGMENT_REPRESENTATION = (
    "native_variable_length_dialogue_directive_18d_30hz_v1"
)
DIRECTIVE_ROLE = "robot_brain_motion_directive_primary_control"
DIALOGUE_ROLE = (
    "moving_speaker_self_utterance_auxiliary_not_partner_response_label"
)
DIALOGUE_ALIGNMENT_POLICY = (
    "single_ascii_space_join_of_overlapping_words_tier_tokens_v1"
)


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


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def validate_dialogue_directive_v10_episode(
    episode: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one dataset record without loading its 18D trajectory."""

    episode = _mapping(episode, "episode")
    clip_id = str(episode.get("clip_id") or "").strip()
    if not clip_id:
        raise ValueError("dialogue/directive episode is missing clip_id")
    if episode.get("artifact_kind") != ARTIFACT_KIND:
        raise ValueError(f"{clip_id}: invalid dialogue/directive artifact kind")
    if episode.get("formal_episode_contract") != FORMAL_EPISODE_CONTRACT:
        raise ValueError(f"{clip_id}: invalid dialogue/directive episode contract")
    if (
        episode.get("accepted_for_training") is not True
        or episode.get("eligibility_mode") != FORMAL_ELIGIBILITY_MODE
    ):
        raise ValueError(f"{clip_id}: dialogue/directive episode is not train-ready")
    if episode.get("native_variable_length") is not True:
        raise ValueError(f"{clip_id}: native_variable_length must be true")

    directive = str(episode.get("directive_text") or "").strip()
    dialogue = str(episode.get("dialogue_text") or "").strip()
    if not directive or not dialogue:
        raise ValueError(f"{clip_id}: directive_text and dialogue_text must be non-empty")
    if episode.get("directive_text_sha256") != text_sha256(directive):
        raise ValueError(f"{clip_id}: directive text SHA256 changed")
    if episode.get("dialogue_text_sha256") != text_sha256(dialogue):
        raise ValueError(f"{clip_id}: dialogue text SHA256 changed")
    if episode.get("prompt") != directive:
        raise ValueError(f"{clip_id}: compatibility prompt must equal directive_text")
    if episode.get("directive_conditioning_mask") is not True:
        raise ValueError(f"{clip_id}: directive conditioning must be enabled")
    if episode.get("dialogue_conditioning_mask") is not True:
        raise ValueError(f"{clip_id}: dialogue conditioning must be enabled")
    if episode.get("source_transcript_used_as_action_or_emotion_label") is not False:
        raise ValueError(f"{clip_id}: transcript may not become an action/emotion label")
    if episode.get("partner_response_supervision_mask") is not False:
        raise ValueError(f"{clip_id}: BEAT2 does not supervise partner responses")
    if episode.get("self_speech_gesture_supervision_mask") is not True:
        raise ValueError(f"{clip_id}: co-speech gesture supervision must be enabled")

    alignment = _mapping(
        episode.get("dialogue_motion_alignment"),
        f"{clip_id}.dialogue_motion_alignment",
    )
    expected_alignment = {
        "positive_pair": True,
        "evidence": "exact_words_tier_overlap_with_native_motion_interval",
        "supervision": "weak_temporal_co_speech_alignment",
        "specific_action_label": False,
        "hard_negative_required": True,
    }
    if {key: alignment.get(key) for key in expected_alignment} != expected_alignment:
        raise ValueError(f"{clip_id}: dialogue-motion alignment contract changed")
    if not _is_sha256(alignment.get("hard_negative_record_sha256")):
        raise ValueError(f"{clip_id}: dialogue hard-negative binding is invalid")

    directive_contract = _mapping(
        episode.get("directive_contract"), f"{clip_id}.directive_contract"
    )
    expected_directive = {
        "role": DIRECTIVE_ROLE,
        "source": "verified_conversational_motion_realization_prompt_v9",
        "primary_control": True,
        "derived_from_dialogue_text": False,
        "specific_intent_supervision": False,
    }
    if {key: directive_contract.get(key) for key in expected_directive} != expected_directive:
        raise ValueError(f"{clip_id}: directive contract changed")

    dialogue_contract = _mapping(
        episode.get("dialogue_contract"), f"{clip_id}.dialogue_contract"
    )
    expected_dialogue = {
        "role": DIALOGUE_ROLE,
        "source": "beat2_words_textgrid_native_interval_overlap_v1",
        "alignment_policy": DIALOGUE_ALIGNMENT_POLICY,
        "auxiliary_context": True,
        "action_label_supervision": False,
        "emotion_label_supervision": False,
        "partner_response_supervision": False,
    }
    if {key: dialogue_contract.get(key) for key in expected_dialogue} != expected_dialogue:
        raise ValueError(f"{clip_id}: dialogue contract changed")
    token_count = dialogue_contract.get("token_count")
    if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 1:
        raise ValueError(f"{clip_id}: dialogue token count must be positive")
    if not _is_sha256(dialogue_contract.get("token_sequence_sha256")):
        raise ValueError(f"{clip_id}: dialogue token sequence SHA256 is invalid")
    if not _is_sha256(dialogue_contract.get("textgrid_sha256")):
        raise ValueError(f"{clip_id}: TextGrid SHA256 is invalid")
    if not str(dialogue_contract.get("textgrid_relpath") or "").strip():
        raise ValueError(f"{clip_id}: TextGrid relative path is missing")

    segment = _mapping(episode.get("training_segment"), f"{clip_id}.training_segment")
    if segment.get("representation") != TRAINING_SEGMENT_REPRESENTATION:
        raise ValueError(f"{clip_id}: training segment representation changed")
    if segment.get("fixed_window_sec") is not None or segment.get("cropped") is not False:
        raise ValueError(f"{clip_id}: fixed-window or cropped training is forbidden")
    start = segment.get("start_frame")
    end = segment.get("end_frame_exclusive")
    frames = segment.get("frame_count")
    if (
        any(isinstance(value, bool) or not isinstance(value, int) for value in (start, end, frames))
        or start < 0
        or end <= start
        or frames != end - start
    ):
        raise ValueError(f"{clip_id}: native frame interval is invalid")
    fps = episode.get("fps")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isclose(float(fps), 30.0, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise ValueError(f"{clip_id}: dialogue/directive data must be exactly 30 Hz")
    if (
        not math.isclose(
            float(dialogue_contract.get("interval_start_sec", -1.0)),
            start / 30.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        or not math.isclose(
            float(dialogue_contract.get("interval_end_sec", -1.0)),
            end / 30.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        raise ValueError(f"{clip_id}: dialogue and motion intervals are not aligned")

    for field in (
        "trajectory_sha256",
        "source_manifest_sha256",
        "source_record_sha256",
        "base_v9_manifest_sha256",
        "base_v9_record_sha256",
    ):
        if not _is_sha256(episode.get(field)):
            raise ValueError(f"{clip_id}: {field} must be a SHA256")

    admission = _mapping(episode.get("training_admission"), f"{clip_id}.training_admission")
    expected_admission = {
        "contract": FORMAL_EPISODE_CONTRACT,
        "trajectory_sha256": episode["trajectory_sha256"],
        "source_record_sha256": episode["source_record_sha256"],
        "base_v9_record_sha256": episode["base_v9_record_sha256"],
        "directive_text_sha256": episode["directive_text_sha256"],
        "dialogue_text_sha256": episode["dialogue_text_sha256"],
        "textgrid_sha256": dialogue_contract["textgrid_sha256"],
        "hard_negative_record_sha256": alignment["hard_negative_record_sha256"],
        "training_channel_masks": {
            "motion_18d": True,
            "directive_text": True,
            "dialogue_text": True,
            "dialogue_motion_alignment": True,
            "trajectory_style": True,
            "primary_intent": False,
            "emotion": False,
            "partner_response": False,
            "audio": False,
        },
    }
    if admission != expected_admission:
        raise ValueError(f"{clip_id}: training admission binding changed")
    return {
        "clip_id": clip_id,
        "frame_count": int(episode.get("frames") or 0),
        "directive_text_sha256": episode["directive_text_sha256"],
        "dialogue_text_sha256": episode["dialogue_text_sha256"],
    }


__all__ = [
    "ARTIFACT_KIND",
    "DIALOGUE_ALIGNMENT_POLICY",
    "DIALOGUE_ROLE",
    "DIRECTIVE_ROLE",
    "FORMAL_ELIGIBILITY_MODE",
    "FORMAL_EPISODE_CONTRACT",
    "TRAINING_SEGMENT_REPRESENTATION",
    "canonical_sha256",
    "text_sha256",
    "validate_dialogue_directive_v10_episode",
]
