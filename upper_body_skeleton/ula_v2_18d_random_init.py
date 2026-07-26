"""Strict full-random initialization contract for the 18D ULA MMDiT V2.

This module deliberately does not accept a generator checkpoint.  The only
learned artifact it consumes is the frozen Qwen LoRA text encoder.  Dataset
statistics and style normalization are fitted after a speaker/source-group
split, using the optimization split only.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from upper_body_skeleton.kimodo_semantics import (
    KIMODO_BEHAVIOR_IDS,
    KIMODO_BEHAVIOR_FAMILIES,
    KIMODO_EMOTION_IDS,
)
from upper_body_skeleton.retarget_v2 import JOINT_ORDER
from upper_body_skeleton.retarget_v2_18d import (
    CONTRACT_VERSION,
    JOINT_ORDER_18D,
)
from upper_body_skeleton.ula_training import (
    KIMODO_CONDITION_DIM,
    KIMODO_MOTION_LATENT_DIM,
    KIMODO_V2_CONDITION_DIM,
    TRANSITION_IDS,
    LEGACY_CONDITION_DIM,
    ULA_MMDIT_V2_ARCHITECTURE,
    create_ula_model,
)
from upper_body_skeleton.ula_v2_18d_head import (
    ACTION_DIM,
    ARTIFACT_KIND,
    CHECKPOINT_SCHEMA_VERSION,
    CONDITION_CACHE_SCHEMA_VERSION,
    FORMAL_SELECTED_LINEAGE_FIELDS,
    FORMAL_SEMANTIC_SUPERVISION_MASKS,
    LEGACY_ACTION_DIM,
    LEGACY_AFFECT_SLICE,
    LEGACY_GESTURE_SLICE,
    LEGACY_INTENT_SLICE,
    MOTION_ONLY_EPISODE_CONTRACT,
    OFFICIAL_CATEGORY_CONDITIONING_ROLE,
    OFFICIAL_EMOTION_CONDITION_CHANNEL,
    OFFICIAL_EMOTION_DISABLED_ROLE,
    OFFICIAL_EMOTION_ENABLED_ROLE,
    STYLE_CONTROL_SLICE,
    _formal_blind_affect_errors,
    semantic_supervision_policy,
    sha256_file,
    validate_checkpoint_contract,
)
from upper_body_skeleton.ula_v2_18d_posttrain import strict_group_split
from upper_body_skeleton.ula_v2_conditioning import (
    STYLE_FEATURE_NAMES,
    extract_style_features,
)


RANDOM_INIT_CONTRACT_VERSION = 1
RANDOM_INIT_MODE = "full_generator_random_qwen_lora_frozen_v1"
VARIABLE_SEGMENT_REPRESENTATION = "native_variable_length_semantic_clip_v1"
RETARGET_SEGMENT_REPRESENTATION = (
    "native_variable_length_semantic_event_retimed_30hz_v1"
)
FORMAL_SEMANTIC_EVENT_SELECTION_STATUS = (
    "official_semantic_event_variable_length_boundary_validated"
)
PROJECT_BEHAVIOR_MAPPING_SOURCE = "project_dataset_scope_weak_mapping_v1"
DEFAULT_LENGTH_BUCKETS = (48, 64, 96, 128, 192, 256, 384, 512)
DEFAULT_SPLIT_FRACTIONS = {"train": 0.8, "validation": 0.1, "test": 0.1}
LEGACY_FORMAL_EPISODE_CONTRACT = "official_semantic_event_train_episode_v1"


def _canonical_json(value) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _contract(payload: Mapping) -> dict:
    result = dict(payload)
    result["sha256"] = hashlib.sha256(
        _canonical_json(payload).encode("ascii")
    ).hexdigest()
    return result


def _episode_id(episode: Mapping) -> str:
    clip_id = str(episode.get("clip_id") or "").strip()
    if not clip_id:
        raise ValueError("random initialization episode is missing clip_id")
    return clip_id


def _formal_episode_contract(episodes: Sequence[Mapping]) -> str:
    """Return one explicit dataset contract and reject mixed supervision regimes."""
    from upper_body_skeleton.ula_v2_expression_turn_episode import (
        FORMAL_EPISODE_CONTRACT,
        is_expression_turn_v8_episode,
    )

    regimes = []
    for episode in episodes:
        if is_expression_turn_v8_episode(episode):
            regimes.append(FORMAL_EPISODE_CONTRACT)
        elif episode.get("formal_episode_contract") == MOTION_ONLY_EPISODE_CONTRACT:
            regimes.append(MOTION_ONLY_EPISODE_CONTRACT)
        else:
            regimes.append(LEGACY_FORMAL_EPISODE_CONTRACT)
    unique = set(regimes)
    if len(unique) != 1:
        raise ValueError(
            "random initialization cannot mix expression-turn v8, motion-only, "
            "and legacy semantic-event episode contracts"
        )
    return regimes[0]


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _validate_embedded_contract(contract: Mapping, *, name: str) -> None:
    if not isinstance(contract, Mapping) or not _is_sha256(contract.get("sha256")):
        raise ValueError(f"random checkpoint {name} contract is missing its SHA256")
    payload = {key: value for key, value in contract.items() if key != "sha256"}
    if _contract(payload)["sha256"] != contract["sha256"]:
        raise ValueError(f"random checkpoint {name} contract hash mismatch")


def _validate_posttrain_json_contract(contract: Mapping, *, name: str) -> None:
    if not isinstance(contract, Mapping) or not _is_sha256(contract.get("sha256")):
        raise ValueError(f"random checkpoint {name} contract is missing its SHA256")
    payload = {key: value for key, value in contract.items() if key != "sha256"}
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if digest != contract["sha256"]:
        raise ValueError(f"random checkpoint {name} contract hash mismatch")


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _validate_retarget_segment(
    episode: Mapping,
    *,
    clip_id: str,
    source_start: int,
    source_end: int,
    source_frame_count: int,
    output_frame_count: int,
    fps: float,
) -> None:
    segment = episode.get("retarget_segment")
    if not isinstance(segment, Mapping):
        raise ValueError(f"{clip_id}: retarget_segment contract is missing")
    payload = {key: value for key, value in segment.items() if key != "sha256"}
    if not _is_sha256(segment.get("sha256")) or _contract(payload)["sha256"] != segment.get(
        "sha256"
    ):
        raise ValueError(f"{clip_id}: retarget_segment contract hash is invalid")
    expected_exact = {
        "representation": RETARGET_SEGMENT_REPRESENTATION,
        "source_start_frame": source_start,
        "source_end_frame_exclusive": source_end,
        "source_frame_count": source_frame_count,
        "output_frame_count": output_frame_count,
        "fps": fps,
        "retimed": output_frame_count != source_frame_count,
        "cropped": False,
        "planner_duration_field": "output_sample_span_sec",
        "source_boundary_duration_field": "source_frame_coverage_sec",
        "legacy_quality_duration_sec_role": (
            "output_frame_coverage_compatibility_only_not_planner_target"
        ),
    }
    for field, expected in expected_exact.items():
        if segment.get(field) != expected:
            raise ValueError(
                f"{clip_id}: retarget_segment.{field} does not match source/output evidence"
            )
    expected_durations = {
        "source_frame_coverage_sec": source_frame_count / fps,
        "output_sample_span_sec": max(0, output_frame_count - 1) / fps,
        "output_frame_coverage_sec": output_frame_count / fps,
    }
    for field, expected in expected_durations.items():
        value = segment.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError(
                f"{clip_id}: retarget_segment.{field} uses the wrong time axis"
            )
    if episode.get("quality_source_window_frames") != source_frame_count:
        raise ValueError(f"{clip_id}: quality source-window length is not bound")
    if episode.get("quality_output_frame_count") != output_frame_count:
        raise ValueError(f"{clip_id}: quality output length is not bound")


def _validate_motion_only_variable_length_episode(
    episode: Mapping, *, require_attached_condition: bool
) -> None:
    clip_id = _episode_id(episode)
    if episode.get("accepted_for_training") is not True or episode.get(
        "eligibility_mode"
    ) != "adjudicated_train_ready":
        raise ValueError(f"{clip_id}: motion-only initialization requires train-ready data")
    admission = episode.get("motion_only_admission")
    expected_admission = {
        "physical_qc_only": True,
        "semantic_review_required": False,
        "independent_semantic_review_claimed": False,
        "text_conditioning_enabled": False,
        "emotion_conditioning_enabled": False,
        "audio_conditioning_enabled": False,
        "native_variable_length": True,
        "fixed_duration_training_unit": False,
    }
    if not isinstance(admission, Mapping) or any(
        admission.get(field) != expected
        for field, expected in expected_admission.items()
    ):
        raise ValueError(f"{clip_id}: motion-only admission contract is invalid")
    if episode.get("semantic_supervision_masks") != FORMAL_SEMANTIC_SUPERVISION_MASKS:
        raise ValueError(f"{clip_id}: motion-only semantic channels must all be masked")
    for field in (
        "behavior_supervision_mask",
        "emotion_supervision_mask",
        "affect_observable_supervision_mask",
        "emotion_conditioning_mask",
        "official_category_conditioning_enabled",
        "official_emotion_conditioning_enabled",
    ):
        if episode.get(field) is not False:
            raise ValueError(f"{clip_id}: motion-only field {field} must be false")

    actions = np.asarray(episode.get("actions"), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] < 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"{clip_id}: actions must have shape [frames, {ACTION_DIM}]")
    if not np.isfinite(actions).all():
        raise ValueError(f"{clip_id}: actions contain non-finite values")
    fps = float(episode.get("fps") or 0.0)
    if not math.isfinite(fps) or not math.isclose(fps, 30.0, abs_tol=1e-9):
        raise ValueError(f"{clip_id}: motion-only training data must be exactly 30 Hz")
    segment = episode.get("training_segment")
    if not isinstance(segment, Mapping):
        raise ValueError(f"{clip_id}: training_segment contract is missing")
    if segment.get("representation") != VARIABLE_SEGMENT_REPRESENTATION:
        raise ValueError(f"{clip_id}: motion-only training requires native variable lengths")
    if segment.get("fixed_window_sec") is not None:
        raise ValueError(f"{clip_id}: fixed-window motion-only samples are forbidden")
    start = segment.get("start_frame")
    end = segment.get("end_frame_exclusive")
    source_frames = segment.get("frame_count")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (start, end, source_frames)
    ) or start < 0 or end <= start or source_frames != end - start:
        raise ValueError(f"{clip_id}: native source interval is inconsistent")
    _validate_retarget_segment(
        episode,
        clip_id=clip_id,
        source_start=start,
        source_end=end,
        source_frame_count=source_frames,
        output_frame_count=int(actions.shape[0]),
        fps=fps,
    )
    if episode.get("retarget_qc_passed") is not True:
        raise ValueError(f"{clip_id}: motion-only training requires passed physical QC")
    for field in ("speaker_key", "source_group_key", "dataset_source", "source_clip_id"):
        if not str(episode.get(field) or "").strip():
            raise ValueError(f"{clip_id}: {field} is required for strict splitting")
    for field in (
        "source_sha256",
        "trajectory_sha256",
        "source_manifest_sha256",
        "source_record_sha256",
    ):
        if not _is_sha256(episode.get(field)):
            raise ValueError(f"{clip_id}: {field} must bind formal source provenance")

    if require_attached_condition or episode.get("condition") is not None:
        condition = np.asarray(episode.get("condition"), dtype=np.float32)
        if condition.shape != (KIMODO_V2_CONDITION_DIM,):
            raise ValueError(f"{clip_id}: attached motion-only condition is missing")
        behavior_start = LEGACY_CONDITION_DIM
        behavior_end = behavior_start + len(KIMODO_BEHAVIOR_IDS)
        emotion_start = behavior_end
        emotion_end = emotion_start + len(KIMODO_EMOTION_IDS)
        family_start = emotion_end
        family_end = family_start + len(KIMODO_BEHAVIOR_FAMILIES)
        masked_slices = (
            LEGACY_INTENT_SLICE,
            LEGACY_AFFECT_SLICE,
            LEGACY_GESTURE_SLICE,
            slice(behavior_start, behavior_end),
            slice(emotion_start, emotion_end),
            slice(family_start, family_end),
            slice(KIMODO_CONDITION_DIM, KIMODO_V2_CONDITION_DIM),
        )
        if any(np.any(condition[index] != 0.0) for index in masked_slices):
            raise ValueError(
                f"{clip_id}: motion-only text, semantic, and emotion conditions must be zero"
            )


def validate_formal_variable_length_episode(
    episode: Mapping, *, require_attached_condition: bool = False
) -> None:
    """Reject fixed-window, unresolved, or unadjudicated formal examples."""
    from upper_body_skeleton.ula_v2_expression_turn_episode import (
        EXPRESSION_TURN_CONTRACT_VERSION,
        is_expression_turn_v8_episode,
        validate_expression_turn_v8_episode,
    )

    if is_expression_turn_v8_episode(episode):
        validate_expression_turn_v8_episode(
            episode, require_attached_condition=require_attached_condition
        )
        return
    if episode.get("formal_episode_contract") == MOTION_ONLY_EPISODE_CONTRACT:
        _validate_motion_only_variable_length_episode(
            episode, require_attached_condition=require_attached_condition
        )
        return
    if episode.get("expression_turn_contract_version") == (
        EXPRESSION_TURN_CONTRACT_VERSION
    ):
        raise ValueError(
            f"{_episode_id(episode)}: v8 data requires an explicit formal episode contract"
        )
    clip_id = _episode_id(episode)
    if episode.get("accepted_for_training") is not True or episode.get(
        "eligibility_mode"
    ) != "adjudicated_train_ready":
        raise ValueError(f"{clip_id}: random baseline requires adjudicated train-ready data")
    if episode.get("source_emotion_label_verified") is not True:
        raise ValueError(f"{clip_id}: official source emotion label must be verified")
    if not isinstance(episode.get("emotion_supervision_mask"), bool):
        raise ValueError(f"{clip_id}: emotion supervision mask must be explicit")
    if episode.get("emotion_id") not in KIMODO_EMOTION_IDS:
        raise ValueError(f"{clip_id}: emotion_id is missing or outside the network ontology")
    if not str(episode.get("prompt") or "").strip():
        raise ValueError(f"{clip_id}: canonical motion prompt is empty")

    emotion_status = episode.get("emotion_review_status")
    if emotion_status == "human_confirmed":
        emotion_review = episode.get("human_review")
        if (
            not isinstance(emotion_review, Mapping)
            or emotion_review.get("reviewer_kind") != "human"
            or emotion_review.get("decision") != "confirmed"
            or not str(emotion_review.get("reviewer_id") or "").strip()
            or not str(emotion_review.get("reviewed_at") or "").strip()
        ):
            raise ValueError(f"{clip_id}: human-confirmed emotion provenance is missing")
    elif emotion_status == "official_protocol_confirmed":
        emotion_contract = episode.get("emotion_protocol_contract")
        emotion_payload = (
            {
                key: value
                for key, value in emotion_contract.items()
                if key != "sha256"
            }
            if isinstance(emotion_contract, Mapping)
            else {}
        )
        if (
            episode.get("emotion_source") != "official_beat2_filename_protocol"
            or not isinstance(emotion_contract, Mapping)
            or emotion_contract.get("source")
            != "official_beat2_filename_protocol"
            or emotion_contract.get("emotion_id") != episode.get("emotion_id")
            or emotion_contract.get("source_sha256")
            != episode.get("source_sha256")
            or _contract(emotion_payload)["sha256"]
            != emotion_contract.get("sha256")
        ):
            raise ValueError(f"{clip_id}: official emotion protocol contract is invalid")
    else:
        raise ValueError(
            f"{clip_id}: formal emotion cannot use legacy or unresolved supervision"
        )

    affect_verified = bool(
        episode.get("affect_observable_review_status") == "verified"
        and episode.get("affect_observable_supervision_mask") is True
    )
    emotion_conditioned = bool(
        episode.get("emotion_supervision_mask") is True
        and episode.get("official_emotion_conditioning_enabled") is True
        and episode.get("emotion_conditioning_mask") is True
        and affect_verified
    )
    expected_emotion_state = {
        "emotion_supervision_mask": emotion_conditioned,
        "official_emotion_conditioning_enabled": emotion_conditioned,
        "emotion_supervision_role": (
            OFFICIAL_EMOTION_ENABLED_ROLE
            if emotion_conditioned
            else OFFICIAL_EMOTION_DISABLED_ROLE
        ),
        "official_emotion_condition_channel": (
            OFFICIAL_EMOTION_CONDITION_CHANNEL if emotion_conditioned else None
        ),
        "official_emotion_loss": None,
        "affect_observable_review_status": (
            "verified" if emotion_conditioned else "not_verified"
        ),
        "affect_observable_supervision_mask": emotion_conditioned,
        "emotion_conditioning_mask": emotion_conditioned,
    }
    for field, expected in expected_emotion_state.items():
        if field not in episode or episode.get(field) != expected:
            raise ValueError(
                f"{clip_id}: formal emotion field {field} does not match blind-review state"
            )
    if emotion_conditioned:
        review = episode.get("independent_review")
        if not isinstance(review, Mapping):
            raise ValueError(f"{clip_id}: blind affect review evidence is missing")
        blind_errors = _formal_blind_affect_errors(episode, review)
        if blind_errors:
            raise ValueError(f"{clip_id}: " + "; ".join(blind_errors))

    if require_attached_condition or episode.get("condition") is not None:
        condition = np.asarray(episode.get("condition"), dtype=np.float32)
        if condition.shape != (KIMODO_V2_CONDITION_DIM,):
            raise ValueError(f"{clip_id}: attached Qwen condition is missing")
        emotion_start = LEGACY_CONDITION_DIM + len(KIMODO_BEHAVIOR_IDS)
        emotion_end = emotion_start + len(KIMODO_EMOTION_IDS)
        expected_emotion = np.zeros(len(KIMODO_EMOTION_IDS), dtype=np.float32)
        if emotion_conditioned:
            expected_emotion[KIMODO_EMOTION_IDS.index(episode["emotion_id"])] = 1.0
        if not np.array_equal(condition[emotion_start:emotion_end], expected_emotion):
            raise ValueError(
                f"{clip_id}: attached emotion condition does not match observable-affect policy"
            )
        if not emotion_conditioned and np.any(condition[LEGACY_AFFECT_SLICE] != 0.0):
            raise ValueError(f"{clip_id}: unverified legacy affect channel must remain zero")

    semantic_event = episode.get("semantic_event")
    if not isinstance(semantic_event, Mapping):
        raise ValueError(f"{clip_id}: official semantic event evidence is missing")
    for field in ("category", "intensity"):
        if not str(semantic_event.get(field) or "").strip():
            raise ValueError(f"{clip_id}: official semantic event {field} is missing")
    if "lexical_anchor" not in semantic_event or not isinstance(
        semantic_event.get("lexical_anchor"), str
    ):
        raise ValueError(f"{clip_id}: official semantic event lexical_anchor is not preserved")
    expected_gesture = (
        "pointing" if semantic_event.get("category") == "deictic" else "upper_body_gesture"
    )
    if semantic_event.get("category") not in {"deictic", "iconic", "metaphoric"}:
        raise ValueError(f"{clip_id}: unsupported official semantic-event category")
    if episode.get("semantic_gesture") != expected_gesture:
        raise ValueError(f"{clip_id}: legacy semantic gesture mapping is inconsistent")
    try:
        semantic_policy = semantic_supervision_policy(episode)
    except ValueError as exc:
        raise ValueError(f"{clip_id}: semantic supervision contract is invalid: {exc}") from exc
    if semantic_policy["semantic_supervision_masks"] != (
        FORMAL_SEMANTIC_SUPERVISION_MASKS
    ):
        raise ValueError(f"{clip_id}: formal semantic supervision masks changed")
    if require_attached_condition or episode.get("condition") is not None:
        cache = episode.get("condition_cache_provenance") or {}
        if cache.get("schema_version") != CONDITION_CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"{clip_id}: formal training requires a schema-3 masked condition cache"
            )
        condition = np.asarray(episode.get("condition"), dtype=np.float32)
        if np.any(condition[KIMODO_CONDITION_DIM:] != 0.0):
            raise ValueError(f"{clip_id}: masked prompt Qwen latent must remain zero")
        if np.any(condition[LEGACY_INTENT_SLICE] != 0.0):
            raise ValueError(f"{clip_id}: unverified communicative intent must remain zero")
        if np.any(condition[LEGACY_GESTURE_SLICE] != 0.0):
            raise ValueError(f"{clip_id}: masked legacy gesture must remain zero")

    behavior_status = episode.get("behavior_review_status")
    behavior_supervised = episode.get("behavior_supervision_mask") is True
    if behavior_supervised:
        if episode.get("behavior_id") not in KIMODO_BEHAVIOR_IDS:
            raise ValueError(
                f"{clip_id}: supervised behavior_id is outside the network ontology"
            )
        if behavior_status != "human_confirmed":
            raise ValueError(
                f"{clip_id}: behavior supervision lacks an accepted explicit review status"
            )
        review = episode.get("human_review")
        if (
            not isinstance(review, Mapping)
            or review.get("reviewer_kind") != "human"
            or review.get("decision") not in {"confirmed", "behavior_confirmed"}
            or not str(review.get("reviewer_id") or "").strip()
            or not str(review.get("reviewed_at") or "").strip()
        ):
            raise ValueError(f"{clip_id}: supervised behavior lacks independent human provenance")
    else:
        if (
            behavior_status != "candidate_unreviewed"
            or episode.get("behavior_id") != "Behavior.InteractPresence"
            or episode.get("behavior_source") != PROJECT_BEHAVIOR_MAPPING_SOURCE
        ):
            raise ValueError(
                f"{clip_id}: unsupervised behavior must use the explicit project weak mapping"
            )
        mapping = episode.get("behavior_mapping_contract")
        if not isinstance(mapping, Mapping):
            raise ValueError(f"{clip_id}: project behavior mapping contract is missing")
        if mapping.get("source") != PROJECT_BEHAVIOR_MAPPING_SOURCE:
            raise ValueError(f"{clip_id}: behavior mapping contract source mismatch")
        mapping_payload = {key: value for key, value in mapping.items() if key != "sha256"}
        if (
            not str(mapping.get("revision") or "").strip()
            or mapping.get("behavior_id") != "Behavior.InteractPresence"
            or mapping.get("supervision") != "weak_candidate_masked"
            or _contract(mapping_payload)["sha256"] != mapping.get("sha256")
        ):
            raise ValueError(
                f"{clip_id}: project behavior mapping revision/content hash is invalid"
            )
        if require_attached_condition or episode.get("condition") is not None:
            condition = np.asarray(episode.get("condition"), dtype=np.float32)
            behavior_start = LEGACY_CONDITION_DIM
            behavior_end = behavior_start + len(KIMODO_BEHAVIOR_IDS)
            family_start = behavior_end + len(KIMODO_EMOTION_IDS)
            family_end = family_start + len(KIMODO_BEHAVIOR_FAMILIES)
            if condition.shape != (KIMODO_V2_CONDITION_DIM,):
                raise ValueError(f"{clip_id}: attached Qwen condition is missing")
            if np.any(condition[behavior_start:behavior_end] != 0.0) or np.any(
                condition[family_start:family_end] != 0.0
            ):
                raise ValueError(
                    f"{clip_id}: weak behavior/family condition channels must remain zero"
                )

    segment = episode.get("training_segment")
    if not isinstance(segment, Mapping):
        raise ValueError(f"{clip_id}: training_segment contract is missing")
    if segment.get("representation") != VARIABLE_SEGMENT_REPRESENTATION:
        raise ValueError(f"{clip_id}: formal training requires native variable-length clips")
    if segment.get("fixed_window_sec") is not None:
        raise ValueError(f"{clip_id}: fixed-window samples are smoke-only")
    boundary_source = segment.get("boundary_source")
    if (
        not isinstance(boundary_source, Mapping)
        or not boundary_source
        or not isinstance(boundary_source.get("mode"), str)
        or not boundary_source["mode"].strip()
    ):
        raise ValueError(
            f"{clip_id}: structured official boundary evidence is missing"
        )
    selection_status = segment.get("selection_status") or episode.get(
        "selection_status"
    )
    if selection_status is None and isinstance(episode.get("window"), Mapping):
        selection_status = episode["window"].get("selection_status")
    if selection_status != FORMAL_SEMANTIC_EVENT_SELECTION_STATUS:
        raise ValueError(f"{clip_id}: official semantic-event selection status is missing")

    actions = np.asarray(episode.get("actions"), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] < 3 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"{clip_id}: actions must have shape [frames, {ACTION_DIM}]")
    if not np.isfinite(actions).all():
        raise ValueError(f"{clip_id}: actions contain non-finite values")
    fps = float(episode.get("fps") or 0.0)
    if not math.isfinite(fps) or not math.isclose(fps, 30.0, abs_tol=1e-9):
        raise ValueError(f"{clip_id}: formal semantic events must be exactly 30 Hz")
    start = segment.get("start_frame")
    end = segment.get("end_frame_exclusive")
    frame_count = segment.get("frame_count")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (start, end, frame_count)):
        raise ValueError(f"{clip_id}: semantic source interval must use integer frames")
    if start < 0 or end <= start or frame_count != end - start:
        raise ValueError(f"{clip_id}: semantic source interval is inconsistent")
    output_frame_count = int(actions.shape[0])
    _validate_retarget_segment(
        episode,
        clip_id=clip_id,
        source_start=start,
        source_end=end,
        source_frame_count=frame_count,
        output_frame_count=output_frame_count,
        fps=fps,
    )
    if episode.get("retarget_qc_passed") is not True:
        raise ValueError(f"{clip_id}: formal semantic event requires passed retarget QC")
    if episode.get("annotation_kind") != "official_gesture_semantic_event":
        raise ValueError(f"{clip_id}: official gesture annotation provenance is missing")
    source_metadata = episode.get("formal_source_metadata")
    retarget_lineage = episode.get("retarget_source_lineage")
    if not isinstance(source_metadata, Mapping) or not isinstance(
        retarget_lineage, Mapping
    ):
        raise ValueError(f"{clip_id}: explicit two-level source lineage is missing")
    for field in FORMAL_SELECTED_LINEAGE_FIELDS:
        if not _is_sha256(source_metadata.get(field)) or (
            retarget_lineage.get(field) != source_metadata.get(field)
        ):
            raise ValueError(f"{clip_id}: {field} lineage is missing or inconsistent")
    if not str(episode.get("source_clip_id") or "").strip():
        raise ValueError(f"{clip_id}: source_clip_id is required")
    for field in (
        "source_sha256",
        "trajectory_sha256",
        "source_manifest_sha256",
        "source_record_sha256",
    ):
        if not _is_sha256(episode.get(field)):
            raise ValueError(f"{clip_id}: {field} must bind formal source provenance")
    for field in ("speaker_key", "source_group_key", "dataset_source"):
        if not str(episode.get(field) or "").strip():
            raise ValueError(f"{clip_id}: {field} is required for strict splitting")


def validate_random_checkpoint_split(
    checkpoint: Mapping,
    episodes: Sequence[Mapping],
    *,
    requested_fractions: Mapping[str, float],
) -> tuple[dict[str, list[dict]], dict]:
    """Restore the immutable initialization split and train-only statistic fit set."""
    random_initialization = checkpoint.get("random_initialization")
    if not isinstance(random_initialization, Mapping) or random_initialization.get(
        "mode"
    ) != RANDOM_INIT_MODE:
        raise ValueError("checkpoint is not the supported full-random generator baseline")
    if (checkpoint.get("action_contract") or {}).get("migration") != (
        "none_full_18d_random_initialization"
    ):
        raise ValueError("full-random checkpoint unexpectedly has generator migration lineage")
    contracts = checkpoint.get("v2_contracts")
    if not isinstance(contracts, Mapping):
        raise ValueError("full-random checkpoint is missing V2 contracts")
    _validate_embedded_contract(contracts, name="V2 aggregate")
    formal_episode_contract = _formal_episode_contract(episodes)
    if checkpoint.get("formal_episode_contract") != formal_episode_contract or (
        contracts.get("formal_episode_contract") != formal_episode_contract
    ):
        raise ValueError(
            "formal episode contract differs from random initialization"
        )
    split_contract = contracts.get("split")
    action_contract = contracts.get("action_statistics")
    style_contract = contracts.get("style")
    duration_contract = contracts.get("duration")
    if (contracts.get("batching") or {}).get(
        "formal_episode_contract"
    ) != formal_episode_contract:
        raise ValueError("random checkpoint batching episode contract changed")
    _validate_posttrain_json_contract(split_contract, name="split")
    for name, contract in (
        ("action statistics", action_contract),
        ("style", style_contract),
        ("duration", duration_contract),
    ):
        _validate_embedded_contract(contract, name=name)

    stored_fractions = split_contract.get("fractions")
    if not isinstance(stored_fractions, Mapping) or set(stored_fractions) != {
        "train",
        "validation",
        "test",
    }:
        raise ValueError("random checkpoint split fractions are invalid")
    for name in ("train", "validation", "test"):
        if not math.isclose(
            float(requested_fractions[name]),
            float(stored_fractions[name]),
            abs_tol=1e-12,
        ):
            raise ValueError("post-training split fractions differ from random initialization")

    split_indices = checkpoint.get("split_episode_indices")
    if not isinstance(split_indices, Mapping) or set(split_indices) != {
        "train",
        "validation",
        "test",
    }:
        raise ValueError("random checkpoint split episode indices are missing")
    stored_ids = {
        name: [str(value) for value in split_indices[name]]
        for name in ("train", "validation", "test")
    }
    flattened = [clip_id for name in stored_ids for clip_id in stored_ids[name]]
    if len(flattened) != len(set(flattened)):
        raise ValueError("random checkpoint split contains duplicate clip IDs")
    by_id = {_episode_id(episode): dict(episode) for episode in episodes}
    if len(by_id) != len(episodes) or set(by_id) != set(flattened):
        raise ValueError("formal training clip IDs differ from random initialization")

    contract_records = {
        str(record.get("clip_id")): dict(record)
        for record in split_contract.get("episodes") or ()
    }
    contract_assignment = {
        clip_id: str(record.get("split"))
        for clip_id, record in contract_records.items()
    }
    expected_assignment = {
        clip_id: name for name in stored_ids for clip_id in stored_ids[name]
    }
    if set(duration_contract.get("fit_clip_ids") or ()) != set(stored_ids["train"]):
        raise ValueError("random checkpoint duration fit set changed")
    if duration_contract.get("fixed_frame_count") is not None or duration_contract.get(
        "fixed_duration_sec"
    ) is not None:
        raise ValueError("random checkpoint duration contract became fixed-length")
    if contract_assignment != expected_assignment:
        raise ValueError("random checkpoint split indices disagree with its split contract")
    for clip_id, episode in by_id.items():
        record = contract_records[clip_id]
        if (
            str(episode.get("speaker_key")) != str(record.get("speaker_key"))
            or str(episode.get("source_group_key"))
            != str(record.get("source_group_key"))
        ):
            raise ValueError(
                f"{clip_id}: speaker/source group differs from random initialization"
            )
    splits = {
        name: [by_id[clip_id] for clip_id in stored_ids[name]]
        for name in ("train", "validation", "test")
    }
    from upper_body_skeleton.ula_v2_18d_posttrain import (
        validate_strict_group_splits,
    )

    validate_strict_group_splits(splits)
    train_ids = set(stored_ids["train"])
    if set(checkpoint.get("training_episode_indices") or ()) != train_ids:
        raise ValueError("random checkpoint training episode indices changed")
    if set(action_contract.get("fit_clip_ids") or ()) != train_ids:
        raise ValueError("action statistics were not fitted on the immutable train split")
    if set(style_contract.get("fit_clip_ids") or ()) != train_ids:
        raise ValueError("style statistics were not fitted on the immutable train split")
    return splits, dict(split_contract)


def compute_masked_train_action_stats(
    episodes: Sequence[Mapping], *, eps: float = 1e-4
) -> tuple[dict[str, torch.Tensor], dict]:
    """Fit per-joint population statistics from optimization-visible values only."""
    if not episodes:
        raise ValueError("cannot fit action statistics without train episodes")
    eps = float(eps)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("action-stat epsilon must be finite and positive")
    count = np.zeros(ACTION_DIM, dtype=np.int64)
    total = np.zeros(ACTION_DIM, dtype=np.float64)
    total_square = np.zeros(ACTION_DIM, dtype=np.float64)
    clip_ids = []
    total_frames = 0
    for episode in episodes:
        clip_id = _episode_id(episode)
        actions = np.asarray(episode.get("actions"), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or actions.shape[0] < 1:
            raise ValueError(f"{clip_id}: action-stat input must be [frames, {ACTION_DIM}]")
        if not np.isfinite(actions).all():
            raise ValueError(f"{clip_id}: action-stat input contains non-finite values")
        mask = np.asarray(
            episode.get("action_dim_mask", np.ones(ACTION_DIM, dtype=np.bool_)),
            dtype=np.bool_,
        )
        if mask.shape != (ACTION_DIM,) or not mask.any():
            raise ValueError(f"{clip_id}: action_dim_mask must observe at least one joint")
        values = actions.astype(np.float64, copy=False)
        observed = np.flatnonzero(mask)
        count[observed] += values.shape[0]
        total[observed] += values[:, observed].sum(axis=0, dtype=np.float64)
        total_square[observed] += np.square(values[:, observed]).sum(
            axis=0, dtype=np.float64
        )
        total_frames += int(values.shape[0])
        clip_ids.append(clip_id)
    missing = [JOINT_ORDER_18D[index] for index in np.flatnonzero(count == 0)]
    if missing:
        raise ValueError(f"train split has no observations for joints: {missing}")
    mean = total / count
    variance = np.maximum(total_square / count - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(variance), eps)
    stats = {
        "mean": torch.from_numpy(mean.astype(np.float32)),
        "std": torch.from_numpy(std.astype(np.float32)),
    }
    provenance = _contract(
        {
            "contract_type": "ula_v2_18d_train_only_action_statistics",
            "contract_version": 1,
            "fit_split": "train",
            "fit_clip_count": len(clip_ids),
            "fit_frame_count": total_frames,
            "fit_clip_ids": sorted(clip_ids),
            "observed_frame_count_by_joint": {
                name: int(count[index]) for index, name in enumerate(JOINT_ORDER_18D)
            },
            "population_std": True,
            "eps": eps,
        }
    )
    return stats, provenance


def fit_train_style_contract(
    episodes: Sequence[Mapping], *, clip: float = 5.0, eps: float = 1e-4
) -> dict:
    if not episodes:
        raise ValueError("cannot fit style statistics without train episodes")
    features = []
    clip_ids = []
    for episode in episodes:
        mask = np.asarray(
            episode.get("action_dim_mask", np.ones(ACTION_DIM, dtype=np.bool_)),
            dtype=np.bool_,
        )
        if mask.shape != (ACTION_DIM,) or not mask[:LEGACY_ACTION_DIM].all():
            raise ValueError("style statistics require every 15D body channel")
        actions = np.asarray(episode["actions"], dtype=np.float32)
        features.append(
            extract_style_features(
                actions[:, :LEGACY_ACTION_DIM], fps=float(episode.get("fps") or 30.0)
            )
        )
        clip_ids.append(_episode_id(episode))
    values = np.stack(features).astype(np.float32)
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(values.std(axis=0, dtype=np.float64).astype(np.float32), eps)
    return _contract(
        {
            "contract_type": "ula_v2_style_normalization",
            "contract_version": 1,
            "feature_names": list(STYLE_FEATURE_NAMES),
            "feature_definition": {
                "signed_arm_balance": (
                    "(right_rms_amplitude-left_rms_amplitude)/(right+left+eps)"
                ),
                "log_arm_amplitude": "log1p(rms temporal arm deviation in radians)",
                "log_arm_speed": "log1p(rms arm velocity in radians/second)",
            },
            "mean": mean.tolist(),
            "std": std.tolist(),
            "eps": float(eps),
            "clip": float(clip),
            "fit_split": "train",
            "fit_episode_count": len(clip_ids),
            "fit_clip_ids": sorted(clip_ids),
        }
    )


def qwen_lora_source_contract(qwen_checkpoint: str | Path) -> dict:
    qwen_checkpoint = Path(qwen_checkpoint)
    payload = torch.load(qwen_checkpoint, map_location="cpu", weights_only=True)
    if payload.get("artifact_kind") != "qwen_motion_cross_modal_alignment":
        raise ValueError("Qwen checkpoint is not a motion-alignment LoRA artifact")
    if not isinstance(payload.get("qwen_lora_state_dict"), Mapping) or not payload[
        "qwen_lora_state_dict"
    ]:
        raise ValueError("Qwen checkpoint has no trained LoRA state")
    qwen = payload.get("qwen") or {}
    config = payload.get("config") or {}
    latent_dim = int(config.get("latent_dim", -1))
    if latent_dim != KIMODO_MOTION_LATENT_DIM:
        raise ValueError(
            f"Qwen motion latent must be {KIMODO_MOTION_LATENT_DIM}D, got {latent_dim}"
        )
    model_name = str(qwen.get("model_name") or "").strip()
    revision = str(qwen.get("revision") or "").strip()
    if not model_name or not revision:
        raise ValueError("Qwen checkpoint lacks a pinned model name/revision")
    return {
        "checkpoint_sha256": sha256_file(qwen_checkpoint),
        "artifact_kind": payload["artifact_kind"],
        "global_step": int(payload.get("global_step", 0)),
        "best_step": int(payload.get("best_step", 0)),
        "model_name": model_name,
        "revision": revision,
        "latent_dim": latent_dim,
        "lora_policy": "reuse_frozen_existing_lora_no_generator_weight_reuse",
    }


def build_length_bucket_contract(
    train_episodes: Sequence[Mapping],
    *,
    buckets: Sequence[int] = DEFAULT_LENGTH_BUCKETS,
) -> dict:
    formal_episode_contract = _formal_episode_contract(train_episodes)
    if formal_episode_contract in {
        LEGACY_FORMAL_EPISODE_CONTRACT,
        MOTION_ONLY_EPISODE_CONTRACT,
    }:
        source_representation = VARIABLE_SEGMENT_REPRESENTATION
    else:
        from upper_body_skeleton.ula_v2_expression_turn_episode import (
            EXPRESSION_TURN_REPRESENTATION,
        )

        source_representation = EXPRESSION_TURN_REPRESENTATION
    frame_counts = [int(np.asarray(episode["actions"]).shape[0]) for episode in train_episodes]
    if not frame_counts:
        raise ValueError("cannot build length buckets without train episodes")
    normalized = sorted({int(value) for value in buckets if int(value) >= 3})
    if not normalized:
        raise ValueError("length buckets must contain a frame count of at least three")
    maximum = max(frame_counts)
    if normalized[-1] < maximum:
        normalized.append(int(math.ceil(maximum / 32.0) * 32))
    if len(set(frame_counts)) < 2:
        raise ValueError("formal random training cannot be built from one fixed frame length")
    assignment = {
        _episode_id(episode): next(
            value
            for value in normalized
            if value >= int(np.asarray(episode["actions"]).shape[0])
        )
        for episode in train_episodes
    }
    return _contract(
        {
            "contract_type": "ula_v2_native_variable_length_batching",
            "contract_version": 1,
            "formal_episode_contract": formal_episode_contract,
            "source_representation": source_representation,
            "fixed_window_training": False,
            "length_buckets_frames": normalized,
            "collation": "right_pad_to_bucket_with_boolean_frame_valid_mask",
            "attention_mask": "semantic_tokens_valid_plus_motion_frame_valid_mask",
            "loss_mask": "frame_valid_mask_and_per_joint_observation_mask",
            "duration_target": "native_sample_intervals_divided_by_fps",
            "train_frame_count_min": min(frame_counts),
            "train_frame_count_max": maximum,
            "train_assignment": assignment,
        }
    )


def build_native_duration_contract(train_episodes: Sequence[Mapping]) -> dict:
    """Bind the duration head to complete, native-length training episodes."""
    if not train_episodes:
        raise ValueError("cannot build a duration contract without train episodes")
    formal_episode_contract = _formal_episode_contract(train_episodes)
    frame_counts: list[int] = []
    durations: list[float] = []
    fit_clip_ids: list[str] = []
    for episode in train_episodes:
        clip_id = _episode_id(episode)
        if formal_episode_contract not in {
            LEGACY_FORMAL_EPISODE_CONTRACT,
            MOTION_ONLY_EPISODE_CONTRACT,
        }:
            arc_review = (
                (episode.get("expression_turn_review_record") or {}).get(
                    "motion_arc_review"
                )
                or {}
            )
            incomplete = [
                phase
                for phase in ("onset", "apex", "offset")
                if (arc_review.get(phase) or {}).get("status") != "complete"
            ]
            if incomplete:
                raise ValueError(
                    f"{clip_id}: duration supervision requires a complete "
                    f"onset/apex/offset arc; incomplete={incomplete}"
                )
        actions = np.asarray(episode["actions"])
        fps = float(episode.get("fps") or 30.0)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or actions.shape[0] < 2:
            raise ValueError(f"{clip_id}: duration supervision requires at least two 18D frames")
        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"{clip_id}: duration supervision fps must be positive")
        frame_count = int(actions.shape[0])
        duration = float((frame_count - 1) / fps)
        frame_counts.append(frame_count)
        durations.append(duration)
        fit_clip_ids.append(clip_id)

    duration_values = np.asarray(durations, dtype=np.float64)
    duration_statistics = {
        "min": float(duration_values.min()),
        "median": float(np.median(duration_values)),
        "max": float(duration_values.max()),
    }
    motion_only = formal_episode_contract == MOTION_ONLY_EPISODE_CONTRACT
    return _contract(
        {
            "contract_type": "ula_v2_native_complete_expression_duration",
            "contract_version": 1,
            "formal_episode_contract": formal_episode_contract,
            "fixed_frame_count": None,
            "fixed_duration_sec": None,
            "trajectory_representation": (
                "native_variable_length_physical_qc_motion"
                if motion_only
                else "native_variable_length_complete_expression_arc"
            ),
            "semantic_unit": (
                "native_physical_qc_motion_segment"
                if motion_only
                else "natural_onset_apex_offset_expression_arc"
            ),
            "duration_formula": "(frame_count-1)/fps",
            "duration_supervision_policy": (
                "all_train_episodes_native_output_sample_span_no_crop_no_fixed_target"
                if motion_only
                else "all_train_episodes_complete_arc_output_sample_span_no_crop_no_fixed_target"
            ),
            "duration_supervision_episode_count": len(train_episodes),
            "duration_supervision_sec": duration_statistics,
            "train_episode_count": len(train_episodes),
            "train_frame_count": {
                "min": min(frame_counts),
                "median": float(np.median(np.asarray(frame_counts, dtype=np.float64))),
                "max": max(frame_counts),
            },
            "train_duration_sec": duration_statistics,
            "statistics_source": "train_only",
            "fit_clip_ids": sorted(fit_clip_ids),
            "cropping_allowed": False,
            "padding_affects_duration": False,
            "inference_length_policy": (
                "learned_duration_head_clamped_to_observed_train_support"
            ),
        }
    )


def collate_variable_length_18d(
    episodes: Sequence[Mapping], *, buckets: Sequence[int] = DEFAULT_LENGTH_BUCKETS
) -> dict[str, torch.Tensor]:
    """Pad native clips to a length bucket without converting them to 6s windows."""
    if not episodes:
        raise ValueError("cannot collate an empty batch")
    frame_counts = [int(np.asarray(episode["actions"]).shape[0]) for episode in episodes]
    candidates = sorted({int(value) for value in buckets if int(value) >= 3})
    required = max(frame_counts)
    bucket = next((value for value in candidates if value >= required), None)
    if bucket is None:
        bucket = int(math.ceil(required / 32.0) * 32)
    actions = torch.zeros((len(episodes), bucket, ACTION_DIM), dtype=torch.float32)
    frame_valid = torch.zeros((len(episodes), bucket), dtype=torch.bool)
    joint_valid = torch.zeros((len(episodes), ACTION_DIM), dtype=torch.bool)
    durations = torch.empty(len(episodes), dtype=torch.float32)
    for row, episode in enumerate(episodes):
        values = torch.as_tensor(np.asarray(episode["actions"]), dtype=torch.float32)
        if values.ndim != 2 or values.shape[1] != ACTION_DIM:
            raise ValueError(f"{_episode_id(episode)}: invalid 18D actions")
        count = int(values.shape[0])
        actions[row, :count] = values
        frame_valid[row, :count] = True
        mask = np.asarray(
            episode.get("action_dim_mask", np.ones(ACTION_DIM, dtype=np.bool_)),
            dtype=np.bool_,
        )
        if mask.shape != (ACTION_DIM,):
            raise ValueError(f"{_episode_id(episode)}: invalid action_dim_mask")
        joint_valid[row] = torch.from_numpy(mask)
        durations[row] = (count - 1) / float(episode.get("fps") or 30.0)
    return {
        "actions": actions,
        "frame_valid_mask": frame_valid,
        "action_dim_mask": joint_valid,
        "durations_sec": durations,
        "frame_counts": torch.tensor(frame_counts, dtype=torch.int64),
    }


def default_style_evaluation_conditions(conditions) -> np.ndarray:
    """Remove reference-trajectory style controls from primary evaluation."""
    values = np.asarray(conditions, dtype=np.float32)
    if values.ndim not in (1, 2) or values.shape[-1] != KIMODO_V2_CONDITION_DIM:
        raise ValueError(
            f"conditions must end in {KIMODO_V2_CONDITION_DIM} dimensions"
        )
    if not np.isfinite(values).all():
        raise ValueError("conditions contain non-finite values")
    result = values.copy()
    result[..., STYLE_CONTROL_SLICE] = 0.0
    return result


def forward_with_frame_mask(model, x_t, t, condition, frame_valid_mask):
    """ULA MMDiT V2 forward pass with key padding for native-length batches."""
    if frame_valid_mask.shape != x_t.shape[:2] or frame_valid_mask.dtype != torch.bool:
        raise ValueError("frame_valid_mask must be bool [batch, frames]")
    motion = model.input(x_t)
    motion = motion + model.time(t)[:, None, :]
    frame_counts = frame_valid_mask.sum(dim=1)
    frame_indices = torch.arange(x_t.shape[1], device=x_t.device)[None, :]
    positions = frame_indices.to(x_t.dtype) / (frame_counts - 1).clamp_min(1)[:, None]
    positions = positions.masked_fill(~frame_valid_mask, 0.0)
    if not hasattr(model.frame, "embed_positions"):
        raise TypeError("native variable-length attention requires position-aware frame embedding")
    motion = motion + model.frame.embed_positions(positions).to(motion.dtype)
    semantic = model.semantic_condition_tokens(condition)
    hidden = torch.cat([semantic, motion], dim=1)
    semantic_valid = torch.ones(
        (x_t.shape[0], model.semantic_tokens), dtype=torch.bool, device=x_t.device
    )
    valid = torch.cat([semantic_valid, frame_valid_mask], dim=1)
    hidden = model.blocks(hidden, src_key_padding_mask=~valid)
    return model.output(model.output_norm(hidden[:, model.semantic_tokens :, :]))


def _explicit_random_initialize(model: nn.Module, seed: int) -> None:
    """Initialize every learned generator tensor without cloned-layer symmetry."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        for name, parameter in model.named_parameters():
            with torch.no_grad():
                if name.endswith("weight") and parameter.ndim >= 2:
                    nn.init.xavier_uniform_(parameter)
                elif name.endswith("weight"):
                    parameter.fill_(1.0)
                else:
                    parameter.zero_()


def _split_for_initialization(
    episodes: Sequence[Mapping], *, seed: int, fractions: Mapping[str, float]
) -> tuple[dict[str, list[dict]], dict]:
    provisional = []
    for episode in episodes:
        validate_formal_variable_length_episode(episode)
        item = dict(episode)
        item["condition"] = np.zeros(KIMODO_V2_CONDITION_DIM, dtype=np.float32)
        provisional.append(item)
    return strict_group_split(provisional, seed=seed, fractions=fractions)


def build_random_18d_checkpoint(
    episodes: Sequence[Mapping],
    *,
    qwen_checkpoint: str | Path,
    source_provenance: Sequence[Mapping],
    seed: int = 7,
    fractions: Mapping[str, float] = DEFAULT_SPLIT_FRACTIONS,
    hidden_dim: int = 384,
    layers: int = 6,
    semantic_tokens: int = 7,
    style_clip: float = 5.0,
    length_buckets: Sequence[int] = DEFAULT_LENGTH_BUCKETS,
) -> tuple[dict, dict, dict]:
    """Return a contract-valid random checkpoint, split contract, and audit report."""
    episodes = list(episodes)
    if not episodes:
        raise ValueError("at least one formal variable-length episode is required")
    if len({_episode_id(episode) for episode in episodes}) != len(episodes):
        raise ValueError("clip_id must be globally unique across motion sources")
    formal_episode_contract = _formal_episode_contract(episodes)
    motion_only = formal_episode_contract == MOTION_ONLY_EPISODE_CONTRACT
    expression_turn_v8 = formal_episode_contract not in {
        LEGACY_FORMAL_EPISODE_CONTRACT,
        MOTION_ONLY_EPISODE_CONTRACT,
    }
    qwen_source = qwen_lora_source_contract(qwen_checkpoint)
    splits, split_contract = _split_for_initialization(
        episodes, seed=int(seed), fractions=fractions
    )
    train = splits["train"]
    action_stats, action_stats_contract = compute_masked_train_action_stats(train)
    style_contract = fit_train_style_contract(train, clip=style_clip)
    batching_contract = build_length_bucket_contract(train, buckets=length_buckets)
    duration_contract = build_native_duration_contract(train)

    semantic_episodes = (
        [
            episode
            for episode in episodes
            if (episode.get("training_channel_masks") or {}).get(
                "semantic_conditioning"
            )
            is True
        ]
        if expression_turn_v8
        else ([] if motion_only else episodes)
    )
    prompt_profile_counts: dict[str, int] = {}
    communicative_intent_count = 0
    text_provenance_by_profile: dict[str, str] = {}
    if expression_turn_v8:
        from upper_body_skeleton.ula_v2_expression_turn_episode import (
            DYADIC_INTERACTION_PROMPT_PROFILE,
            DYADIC_PROMPT_TEXT_PROVENANCE,
            MOTION_FORM_PROMPT_PROFILE,
            PROMPT_TEXT_PROVENANCE,
        )

        for episode in semantic_episodes:
            profile = str(episode["prompt_semantics_profile"])
            prompt_profile_counts[profile] = prompt_profile_counts.get(profile, 0) + 1
            communicative_intent_count += int(
                (episode.get("semantic_supervision_masks") or {}).get(
                    "communicative_intent"
                )
                is True
            )
        text_provenance_by_profile = {
            MOTION_FORM_PROMPT_PROFILE: PROMPT_TEXT_PROVENANCE,
            DYADIC_INTERACTION_PROMPT_PROFILE: DYADIC_PROMPT_TEXT_PROVENANCE,
        }
    text_contract_payload = {
        "contract_type": "ula_v2_qwen_lora_text_motion_latent",
        "contract_version": 3 if motion_only else (2 if expression_turn_v8 else 1),
        "latent_dim": KIMODO_MOTION_LATENT_DIM,
        "l2_normalized": True,
        "text_field": (
            None
            if motion_only
            else ("prompt" if expression_turn_v8 else "canonical_prompt")
        ),
        "episode_count": len(episodes),
        "conditioned_episode_count": len(semantic_episodes),
        "masked_episode_count": len(episodes) - len(semantic_episodes),
        "unique_text_count": len(
            {
                str(episode.get("prompt") or "").strip()
                for episode in semantic_episodes
                if str(episode.get("prompt") or "").strip()
            }
        ),
        "source": qwen_source,
        "encoder_training_policy": "frozen_existing_lora",
    }
    if expression_turn_v8:
        text_contract_payload.update(
            {
                "formal_episode_contract": formal_episode_contract,
                "qualification_required": "semantic_conditioning",
                "text_provenance_by_prompt_semantics_profile": (
                    text_provenance_by_profile
                ),
                "prompt_semantics_profile_counts": dict(
                    sorted(prompt_profile_counts.items())
                ),
                "communicative_intent_conditioned_count": (
                    communicative_intent_count
                ),
                "unqualified_text_policy": "prompt_absent_qwen_latent_zero",
            }
        )
    elif motion_only:
        text_contract_payload.update(
            {
                "formal_episode_contract": formal_episode_contract,
                "qualification_required": None,
                "conditioning_policy": "all_text_latents_exact_zero",
                "metadata_text_use": "provenance_only_not_a_training_condition",
                "unqualified_text_policy": "qwen_latent_zero",
            }
        )
    text_contract = _contract(text_contract_payload)
    evaluation_contract = _contract(
        {
            "contract_type": "ula_v2_random_baseline_non_oracle_evaluation",
            "contract_version": 3 if motion_only else (2 if expression_turn_v8 else 1),
            "formal_episode_contract": formal_episode_contract,
            "primary_conditioning": (
                "qualification_masked_blind_text_semantics_plus_blind_affect_default_style"
                if expression_turn_v8
                else (
                    "motion_only_zero_text_emotion_audio_default_style"
                    if motion_only
                    else "text_plus_explicit_semantics_default_style"
                )
            ),
            "default_style_controls": [0.0, 0.0, 0.0],
            "style_control_slice": [STYLE_CONTROL_SLICE.start, STYLE_CONTROL_SLICE.stop],
            "forbidden_primary_inputs": [
                "reference_trajectory_derived_style_controls",
                "reference_motion_latent",
                "validation_or_test_action_statistics",
            ]
            + (
                [
                    "text_conditioning",
                    "semantic_conditioning",
                    "emotion_conditioning",
                    "audio_conditioning",
                ]
                if motion_only
                else []
            ),
            "interaction_model_selection_split": "validation",
            "interaction_final_report_split": "test_once_after_model_selection",
            "kimodo_policy": (
                "original_validation_for_model_selection_and_disjoint_original_test_once"
            ),
            "forgetting_guard": "not_applicable_no_pretrained_generator_baseline",
            "pretrained_comparison_policy": (
                "same_data_same_split_same_qwen_same_sampling_seeds_separate_initialization"
            ),
        }
    )
    if expression_turn_v8:
        tier_counts = {
            tier: sum(
                episode.get("training_qualification_tier") == tier
                for episode in episodes
            )
            for tier in (
                "base_motion",
                "semantic_conditioning",
                "expressive_conditioning",
            )
        }
        expressive_count = sum(
            (episode.get("training_channel_masks") or {}).get(
                "expressive_conditioning"
            )
            is True
            for episode in episodes
        )
        semantic_supervision_contract = _contract(
            {
                "contract_type": (
                    "ula_v2_18d_expression_turn_v8_semantic_supervision"
                ),
                "contract_version": 2,
                "formal_episode_contract": formal_episode_contract,
                "qualification_tier_counts": tier_counts,
                "base_motion_eligible_count": len(episodes),
                "semantic_conditioned_count": len(semantic_episodes),
                "expressive_conditioned_count": expressive_count,
                "channel_masks_by_highest_tier": {
                    "base_motion": {
                        "motion": True,
                        "semantic_conditioning": False,
                        "expressive_conditioning": False,
                    },
                    "semantic_conditioning": {
                        "motion": True,
                        "semantic_conditioning": True,
                        "expressive_conditioning": False,
                    },
                    "expressive_conditioning": {
                        "motion": True,
                        "semantic_conditioning": True,
                        "expressive_conditioning": True,
                    },
                },
                "semantic_text_sources_by_prompt_profile": (
                    text_provenance_by_profile
                ),
                "prompt_semantics_profile_counts": dict(
                    sorted(prompt_profile_counts.items())
                ),
                "communicative_intent_conditioned_count": (
                    communicative_intent_count
                ),
                "communicative_intent_source": (
                    "independent_blind_dyadic_interaction_prompt_only"
                ),
                "expressive_label_source": (
                    "independent_blind_affect_consensus_or_adjudication_v1"
                ),
                "official_category_conditioning_enabled": False,
                "official_emotion_conditioning_enabled": False,
                "legacy_semantic_event_forbidden": True,
                "legacy_affect_encoding": "always_zero",
                "unqualified_channel_policy": "exact_zero_mask",
                "formal_optimization_targets": [
                    "motion_flow_18d",
                    "head_3dof",
                    "trajectory_style",
                    "qualification_masked_observable_text",
                    "qualification_masked_blind_affect",
                    "native_duration",
                ],
            }
        )
    elif motion_only:
        semantic_supervision_contract = _contract(
            {
                "contract_type": "ula_v2_18d_motion_only_supervision",
                "contract_version": 1,
                "formal_episode_contract": formal_episode_contract,
                "motion_conditioned_count": len(episodes),
                "semantic_conditioned_count": 0,
                "expressive_conditioned_count": 0,
                "audio_conditioned_count": 0,
                "semantic_supervision_masks": dict(
                    FORMAL_SEMANTIC_SUPERVISION_MASKS
                ),
                "behavior_conditioning_enabled": False,
                "emotion_conditioning_enabled": False,
                "audio_conditioning_enabled": False,
                "metadata_text_use": "provenance_only_not_a_training_condition",
                "unqualified_channel_policy": "exact_zero_mask",
                "condition_cache_schema_version": CONDITION_CACHE_SCHEMA_VERSION,
                "formal_optimization_targets": [
                    "motion_flow_18d",
                    "head_3dof",
                    "trajectory_style",
                    "native_duration",
                ],
            }
        )
    else:
        semantic_supervision_contract = _contract(
            {
                "contract_type": "ula_v2_18d_formal_semantic_supervision",
                "contract_version": 1,
                "semantic_supervision_masks": dict(
                    FORMAL_SEMANTIC_SUPERVISION_MASKS
                ),
                "official_category_conditioning_enabled": False,
                "official_category_role": OFFICIAL_CATEGORY_CONDITIONING_ROLE,
                "official_category_condition_channel": None,
                "official_category_loss": None,
                "official_category_conditioned_count": 0,
                "official_category_use": "metadata_split_and_evaluation_only",
                "source_emotion_label_use": (
                    "provenance_only_until_blind_robot_affect_review"
                ),
                "emotion_conditioning_requires": [
                    "verified_official_source_emotion_label",
                    "anonymous_silent_video_sha256_bound_blind_affect_review",
                    "target_emotion_not_exposed",
                    "observed_affect_matches_official_label",
                    "affect_observable_in_18d_gate_true",
                ],
                "unverified_emotion_encoding": (
                    "zero_kimodo_emotion_one_hot_and_legacy_affect_slice"
                ),
                "condition_cache_schema_version": CONDITION_CACHE_SCHEMA_VERSION,
                "formal_optimization_targets": [
                    "motion_flow_18d",
                    "head_3dof",
                    "trajectory_style",
                    "verified_observable_emotion_only",
                    "native_duration",
                ],
                "forbidden_category_routes": [
                    "qwen_motion_latent",
                    "legacy_gesture",
                    "legacy_intent",
                    "kimodo_behavior",
                    "kimodo_emotion",
                ],
            }
        )
    condition_contract = _contract(
        {
            "contract_type": "ula_v2_condition",
            "contract_version": 4 if motion_only else (3 if expression_turn_v8 else 2),
            "formal_episode_contract": formal_episode_contract,
            "condition_dim": KIMODO_V2_CONDITION_DIM,
            "base_condition_dim": KIMODO_CONDITION_DIM,
            "motion_latent_dim": KIMODO_MOTION_LATENT_DIM,
            "layout": [
                {"name": "explicit_semantics_with_style", "start": 0, "end": 136},
                {"name": "qwen_lora_text_motion_latent", "start": 136, "end": 264},
            ],
            "style_control_indices": list(
                range(STYLE_CONTROL_SLICE.start, STYLE_CONTROL_SLICE.stop)
            ),
            "default_inference_style_controls": [0.0, 0.0, 0.0],
            "split_contract_sha256": split_contract["sha256"],
            "style_contract_sha256": style_contract["sha256"],
            "duration_contract_sha256": duration_contract["sha256"],
            "text_motion_contract_sha256": text_contract["sha256"],
            "evaluation_contract_sha256": evaluation_contract["sha256"],
            "semantic_supervision_contract_sha256": (
                semantic_supervision_contract["sha256"]
            ),
        }
    )
    contracts = _contract(
        {
            "contract_version": 4 if motion_only else (3 if expression_turn_v8 else 2),
            "formal_episode_contract": formal_episode_contract,
            "split": split_contract,
            "action_statistics": action_stats_contract,
            "style": style_contract,
            "batching": batching_contract,
            "duration": duration_contract,
            "text_motion_latent": text_contract,
            "condition": condition_contract,
            "evaluation": evaluation_contract,
            "semantic_supervision": semantic_supervision_contract,
        }
    )

    # Model construction itself consumes RNG through PyTorch's default reset
    # methods, so isolate both construction and our explicit independent reset.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = create_ula_model(
            ULA_MMDIT_V2_ARCHITECTURE,
            action_dim=ACTION_DIM,
            condition_dim=KIMODO_V2_CONDITION_DIM,
            hidden_dim=int(hidden_dim),
            layers=int(layers),
            semantic_tokens=int(semantic_tokens),
        )
        _explicit_random_initialize(model, int(seed))
    model.action_stats = action_stats
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    state_sha256 = _state_sha256(state)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    source_records = [dict(record) for record in source_provenance]
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "architecture": ULA_MMDIT_V2_ARCHITECTURE,
        "model_state_dict": state,
        "joint_order": list(JOINT_ORDER_18D),
        "condition_dim": KIMODO_V2_CONDITION_DIM,
        "base_condition_dim": KIMODO_CONDITION_DIM,
        "action_dim": ACTION_DIM,
        "action_stats": {
            name: value.detach().cpu().clone() for name, value in action_stats.items()
        },
        "action_contract": {
            "version": CONTRACT_VERSION,
            "joint_order": list(JOINT_ORDER_18D),
            "legacy_prefix_dim": LEGACY_ACTION_DIM,
            "legacy_joint_order": list(JOINT_ORDER),
            "migration": "none_full_18d_random_initialization",
        },
        "formal_episode_contract": formal_episode_contract,
        "v2_contracts": contracts,
        "semantic_supervision_contract": semantic_supervision_contract,
        "split_episode_indices": {
            name: [_episode_id(episode) for episode in splits[name]]
            for name in ("train", "validation", "test")
        },
        "training_episode_indices": [
            _episode_id(episode) for episode in splits["train"]
        ],
        "sources": {
            "motion_manifests": source_records,
            "qwen_checkpoint_sha256": qwen_source["checkpoint_sha256"],
        },
        "random_initialization": {
            "contract_version": RANDOM_INIT_CONTRACT_VERSION,
            "mode": RANDOM_INIT_MODE,
            "seed": int(seed),
            "generator_checkpoint_inputs": [],
            "generator_parameter_count": int(parameter_count),
            "generator_state_sha256": state_sha256,
            "matrix_initialization": "xavier_uniform_independent_per_parameter",
            "bias_initialization": "zeros",
            "normalization_initialization": "unit_scale_zero_bias",
            "qwen_policy": "existing_lora_frozen_not_part_of_generator_state",
        },
        "planner_supervision_contract": {
            "duration_head_supervision": "native_output_sample_span_(N-1)/fps",
            "duration_head_trainable": True,
            "transition_supervised_episode_count": 0,
            "transition_class_counts": {
                name: 0 for name in TRANSITION_IDS
            },
            "transition_head_trainable": False,
            "transition_head_status": (
                "untrained_no_verified_adjacent_sequence_labels"
            ),
            "transition_inference_enabled": False,
            "missing_transition_policy": "mask_false_no_default_end_label",
        },
        "config": {
            "architecture": ULA_MMDIT_V2_ARCHITECTURE,
            "hidden_dim": int(hidden_dim),
            "layers": int(layers),
            "semantic_tokens": int(semantic_tokens),
            "seed": int(seed),
            "action_dim": ACTION_DIM,
            "initialization_mode": RANDOM_INIT_MODE,
            "fixed_window_training": False,
            "formal_episode_contract": formal_episode_contract,
        },
        "global_step": 0,
        "best_step": 0,
        "artifact_status": "untrained_random_initialization_waiting_for_formal_training",
    }
    validate_checkpoint_contract(checkpoint, expected_action_dim=ACTION_DIM)
    report = {
        "artifact_status": checkpoint["artifact_status"],
        "generator_parameter_count": int(parameter_count),
        "generator_state_sha256": state_sha256,
        "initialization_seed": int(seed),
        "qwen_checkpoint_sha256": qwen_source["checkpoint_sha256"],
        "generator_checkpoint_inputs": [],
        "split_counts": {name: len(splits[name]) for name in splits},
        "formal_episode_contract": formal_episode_contract,
        "split_contract_sha256": split_contract["sha256"],
        "semantic_supervision_contract_sha256": (
            semantic_supervision_contract["sha256"]
        ),
        "official_category_conditioned_count": 0,
        "action_statistics_fit_split": "train",
        "action_statistics_fit_clip_count": len(train),
        "action_statistics_contract_sha256": action_stats_contract["sha256"],
        "style_statistics_fit_split": "train",
        "style_contract_sha256": style_contract["sha256"],
            "variable_length_contract_sha256": batching_contract["sha256"],
        "duration_contract_sha256": duration_contract["sha256"],
        "formal_training_started": False,
        "forgetting_guard_applicable": False,
        "kimodo_holdout_required": True,
        "primary_evaluation": evaluation_contract["primary_conditioning"],
    }
    if expression_turn_v8:
        report["qualification_tier_counts"] = dict(
            semantic_supervision_contract["qualification_tier_counts"]
        )
        report["semantic_conditioned_count"] = len(semantic_episodes)
        report["expressive_conditioned_count"] = int(
            semantic_supervision_contract["expressive_conditioned_count"]
        )
    return checkpoint, split_contract, report


__all__ = [
    "DEFAULT_LENGTH_BUCKETS",
    "FORMAL_SEMANTIC_EVENT_SELECTION_STATUS",
    "PROJECT_BEHAVIOR_MAPPING_SOURCE",
    "RANDOM_INIT_MODE",
    "VARIABLE_SEGMENT_REPRESENTATION",
    "build_length_bucket_contract",
    "build_native_duration_contract",
    "build_random_18d_checkpoint",
    "collate_variable_length_18d",
    "compute_masked_train_action_stats",
    "default_style_evaluation_conditions",
    "fit_train_style_contract",
    "forward_with_frame_mask",
    "qwen_lora_source_contract",
    "validate_formal_variable_length_episode",
    "validate_random_checkpoint_split",
]
