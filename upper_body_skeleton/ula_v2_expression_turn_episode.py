"""Formal training contract for reviewed BEAT2 expression turns.

The legacy formal loader treats an official semantic-event label as training
evidence.  Expression-turn v8 deliberately does not.  This module recomputes
the three review qualifications from the anonymous review record and opens
conditioning channels only when the corresponding qualification is present.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tools.human_motion_review.expression_turn_contract import (
    ACTION_TEXT_PROVENANCE,
    CONTRACT_VERSION as EXPRESSION_TURN_CONTRACT_VERSION,
    REPRESENTATION as EXPRESSION_TURN_REPRESENTATION,
    evaluate_expression_turn,
)
from tools.human_motion_review.expression_turn_retarget_contract import (
    REQUIRED_18D_GATES,
    RETARGET_SEGMENT_REPRESENTATION,
)
from upper_body_skeleton.kimodo_semantics import KIMODO_EMOTION_IDS
from upper_body_skeleton.ula_training import (
    KIMODO_CONDITION_DIM,
    KIMODO_V2_CONDITION_DIM,
)
from upper_body_skeleton.ula_v2_18d_head import (
    ACTION_DIM,
    KIMODO_BEHAVIOR_FAMILY_SLICE,
    KIMODO_BEHAVIOR_SLICE,
    KIMODO_EMOTION_SLICE,
    LEGACY_AFFECT_SLICE,
    STYLE_CONTROL_SLICE,
)


FORMAL_EPISODE_CONTRACT = "beat2_expression_turn_v8_train_episode_v1"
FORMAL_ELIGIBILITY_MODE = "expression_turn_v8_adjudicated_train_ready"
PROMPT_TEXT_PROVENANCE = "independent_blind_action_observable_description_v1"
DYADIC_PROMPT_TEXT_PROVENANCE = "independent_blind_dyadic_interaction_prompt_v1"
MOTION_FORM_PROMPT_PROFILE = "robot_observable_motion_form_v1"
DYADIC_INTERACTION_PROMPT_PROFILE = "blind_dyadic_communicative_intent_v1"
CONDITION_CACHE_ARTIFACT_KIND = "ula_v2_expression_turn_v8_condition_cache"
CONDITION_CACHE_SCHEMA_VERSION = 1
HUMAN_RETARGET_PHYSICAL_PROFILE = "human_retarget_18d_safety_v2"
NATIVE_ROBOT_PHYSICAL_PROFILE = "native_robot_18d_full_asset_v1"
NATIVE_ROBOT_RETARGET_SEGMENT_REPRESENTATION = (
    "native_variable_length_robot_expression_turn_30hz_v1"
)
NATIVE_ROBOT_REQUIRED_18D_GATES = {
    "joint_limits_pass",
    "velocity_pass",
    "timing_pass",
    "safe_projection_pass",
    "head_joint_limits_pass",
    "head_velocity_pass",
    "collision_pass",
    "video_decode_pass",
    "video_frame_count_pass",
    "video_nonblank_pass",
    "passed",
}
QUALIFICATION_TIERS = (
    "base_motion",
    "semantic_conditioning",
    "expressive_conditioning",
)
CHANNEL_MASK_KEYS = (
    "motion",
    "semantic_conditioning",
    "expressive_conditioning",
)
SEMANTIC_MASK_KEYS = {
    "official_category",
    "robot_observable_motion_form",
    "communicative_intent",
    "prompt_text",
    "legacy_gesture",
}
FORBIDDEN_LEGACY_KEYS = {
    "official_semantic_event",
    "official_gesture_semantic_spans",
    "semantic_event",
    "semantic_gesture",
}
LINEAGE_SHA256_FIELDS = (
    "expression_turn_contract_sha256",
    "source_inventory_manifest_sha256",
    "split_assignment_manifest_sha256",
    "inventory_record_sha256",
    "upstream_inventory_record_sha256",
    "selected_record_sha256",
    "retarget_input_manifest_sha256",
    "retarget_quality_record_sha256",
    "trajectory_sha256",
    "source_sha256",
)


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


def _ascii_contract_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _require_mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _walk_keys(value: object):
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            yield from _walk_keys(child)


def is_expression_turn_v8_episode(episode: Mapping[str, Any]) -> bool:
    """Return true only for the explicit v8 formal episode marker."""

    return episode.get("formal_episode_contract") == FORMAL_EPISODE_CONTRACT


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            records.append(value)
    if not records:
        raise ValueError(f"expression-turn v8 manifest is empty: {path}")
    return records


def _resolve_manifest_path(value: object, *, manifest: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if not path.is_absolute():
        path = manifest.parent / path
    return path.resolve()


def load_expression_turn_v8_episodes(manifest: str | Path) -> list[dict[str, Any]]:
    """Load only hash-bound, reviewed v8 episodes without legacy admission logic."""

    from upper_body_skeleton.ula_v2_18d_head import read_joint_csv

    manifest = Path(manifest).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"expression-turn v8 manifest is missing: {manifest}")
    manifest_sha256 = _sha256_file(manifest)
    records = _read_jsonl(manifest)
    if not all(is_expression_turn_v8_episode(record) for record in records):
        raise ValueError(
            "expression-turn v8 loader refuses mixed or unmarked episode contracts"
        )

    episodes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        clip_id = str(record.get("clip_id") or "").strip()
        if not clip_id or clip_id in seen:
            raise ValueError(f"missing or duplicate expression-turn clip_id: {clip_id!r}")
        motion = _require_mapping(record.get("motion_18d"), f"{clip_id}.motion_18d")
        motion_path = _resolve_manifest_path(
            motion.get("safe_csv") or record.get("trajectory_path"), manifest=manifest
        )
        if motion_path is None or not motion_path.is_file():
            raise FileNotFoundError(f"{clip_id}: reviewed 18D trajectory is missing: {motion_path}")
        values = read_joint_csv(motion_path)
        trajectory_sha256 = _sha256_file(motion_path)
        if motion.get("safe_csv_sha256") != trajectory_sha256:
            raise ValueError(f"{clip_id}: reviewed 18D trajectory SHA256 changed")
        for field in ("frames", "csv_rows"):
            if motion.get(field) != int(values.shape[0]):
                raise ValueError(f"{clip_id}: motion_18d.{field} does not match the CSV")
        fps = motion.get("fps")
        if (
            isinstance(fps, bool)
            or not isinstance(fps, (int, float))
            or not math.isclose(float(fps), 30.0, rel_tol=0.0, abs_tol=1e-9)
        ):
            raise ValueError(f"{clip_id}: motion_18d.fps must be exactly 30 Hz")
        if record.get("trajectory_sha256") != trajectory_sha256:
            raise ValueError(f"{clip_id}: top-level trajectory SHA256 is not bound to the CSV")

        quality_gate = motion.get("quality_gate")
        if record.get("quality_gate") != quality_gate:
            raise ValueError(f"{clip_id}: top-level and motion_18d quality gates differ")
        retarget_segment = motion.get("retarget_segment")
        if record.get("retarget_segment") != retarget_segment:
            raise ValueError(f"{clip_id}: top-level and motion_18d retarget segments differ")
        physical_profile = motion.get("physical_evidence_profile")
        if record.get("physical_evidence_profile") != physical_profile:
            raise ValueError(
                f"{clip_id}: top-level and motion_18d physical evidence profiles differ"
            )

        episode = deepcopy(record)
        episode.update(
            {
                "actions": np.ascontiguousarray(values),
                "action_dim_mask": np.ones(ACTION_DIM, dtype=np.bool_),
                "condition": None,
                "fps": float(fps),
                "duration_sec": float((values.shape[0] - 1) / float(fps)),
                "trajectory_path": str(motion_path),
                "trajectory_sha256": trajectory_sha256,
                "source_manifest": str(manifest),
                "source_manifest_sha256": manifest_sha256,
                "source_record_sha256": _canonical_sha256(record),
                "retarget_qc_passed": motion.get("state") == "passed",
                "quality_source_window_frames": motion.get("source_window_frames"),
                "quality_output_frame_count": motion.get("frames"),
            }
        )
        validate_expression_turn_v8_episode(episode, require_attached_condition=False)
        episodes.append(episode)
        seen.add(clip_id)
    return episodes


def _validate_retarget_segment(
    episode: Mapping[str, Any],
    *,
    clip_id: str,
    source_start: int,
    source_end: int,
    output_frames: int,
    fps: float,
) -> None:
    segment = _require_mapping(episode.get("retarget_segment"), "retarget_segment")
    recorded_hash = segment.get("sha256")
    payload = {key: value for key, value in segment.items() if key != "sha256"}
    if not _is_sha256(recorded_hash) or _ascii_contract_sha256(payload) != recorded_hash:
        raise ValueError(f"{clip_id}: retarget_segment SHA256 is invalid")
    source_frames = source_end - source_start
    physical_profile = episode.get(
        "physical_evidence_profile", HUMAN_RETARGET_PHYSICAL_PROFILE
    )
    if physical_profile == HUMAN_RETARGET_PHYSICAL_PROFILE:
        representation = RETARGET_SEGMENT_REPRESENTATION
    elif physical_profile == NATIVE_ROBOT_PHYSICAL_PROFILE:
        representation = NATIVE_ROBOT_RETARGET_SEGMENT_REPRESENTATION
        if output_frames != source_frames:
            raise ValueError(f"{clip_id}: native robot motion may not be implicitly retimed")
    else:
        raise ValueError(f"{clip_id}: unsupported physical evidence profile")
    expected = {
        "representation": representation,
        "source_start_frame": source_start,
        "source_end_frame_exclusive": source_end,
        "source_frame_count": source_frames,
        "output_frame_count": output_frames,
        "fps": fps,
        "retimed": output_frames != source_frames,
        "cropped": False,
        "planner_duration_field": "output_sample_span_sec",
        "source_boundary_duration_field": "source_frame_coverage_sec",
        "legacy_quality_duration_sec_role": (
            "output_frame_coverage_compatibility_only_not_planner_target"
        ),
    }
    for field, expected_value in expected.items():
        if segment.get(field) != expected_value:
            raise ValueError(f"{clip_id}: retarget_segment.{field} is inconsistent")
    expected_durations = {
        "source_frame_coverage_sec": source_frames / fps,
        "output_sample_span_sec": max(0, output_frames - 1) / fps,
        "output_frame_coverage_sec": output_frames / fps,
    }
    for field, expected_value in expected_durations.items():
        value = segment.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not math.isclose(
                float(value), expected_value, rel_tol=0.0, abs_tol=1e-9
            )
        ):
            raise ValueError(f"{clip_id}: retarget_segment.{field} has the wrong time axis")


def _expected_semantic_masks(
    semantic_eligible: bool, *, communicative_intent_eligible: bool = False
) -> dict[str, bool]:
    return {
        "official_category": False,
        "robot_observable_motion_form": semantic_eligible,
        "communicative_intent": communicative_intent_eligible,
        "prompt_text": semantic_eligible,
        "legacy_gesture": False,
    }


def _validate_condition(
    episode: Mapping[str, Any],
    *,
    clip_id: str,
    semantic_eligible: bool,
    expressive_eligible: bool,
    blind_affect_class: str | None,
    require_attached_condition: bool,
) -> None:
    if episode.get("condition") is None:
        if require_attached_condition:
            raise ValueError(f"{clip_id}: attached v8 condition is required")
        return
    condition = np.asarray(episode.get("condition"), dtype=np.float32)
    if condition.shape != (KIMODO_V2_CONDITION_DIM,) or not np.isfinite(condition).all():
        raise ValueError(f"{clip_id}: condition must be a finite {KIMODO_V2_CONDITION_DIM}D vector")

    allowed = np.zeros(KIMODO_V2_CONDITION_DIM, dtype=np.bool_)
    allowed[STYLE_CONTROL_SLICE] = True
    if semantic_eligible:
        allowed[KIMODO_CONDITION_DIM:] = True
    if expressive_eligible:
        allowed[KIMODO_EMOTION_SLICE] = True
    if np.any(condition[~allowed] != 0.0):
        raise ValueError(f"{clip_id}: an unqualified condition channel is non-zero")
    if np.any(condition[KIMODO_BEHAVIOR_SLICE] != 0.0) or np.any(
        condition[KIMODO_BEHAVIOR_FAMILY_SLICE] != 0.0
    ):
        raise ValueError(f"{clip_id}: v8 has no independently reviewed behavior label")
    if np.any(condition[LEGACY_AFFECT_SLICE] != 0.0):
        raise ValueError(f"{clip_id}: v8 never maps blind affect into the legacy affect channel")

    qwen_latent = condition[KIMODO_CONDITION_DIM:]
    if semantic_eligible:
        norm = float(np.linalg.norm(qwen_latent))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-4):
            raise ValueError(f"{clip_id}: qualified prompt Qwen latent must be L2-normalized")
    elif np.any(qwen_latent != 0.0):
        raise ValueError(f"{clip_id}: base-only motion must mask the Qwen latent")

    expected_emotion = np.zeros(len(KIMODO_EMOTION_IDS), dtype=np.float32)
    if expressive_eligible:
        if blind_affect_class not in KIMODO_EMOTION_IDS:
            raise ValueError(f"{clip_id}: blind affect class is outside the network ontology")
        expected_emotion[KIMODO_EMOTION_IDS.index(str(blind_affect_class))] = 1.0
    if not np.array_equal(condition[KIMODO_EMOTION_SLICE], expected_emotion):
        raise ValueError(f"{clip_id}: blind affect condition does not match its qualification")


def validate_expression_turn_v8_episode(
    episode: Mapping[str, Any], *, require_attached_condition: bool = False
) -> dict[str, Any]:
    """Validate one reviewed, train-ready v8 episode and return its tier masks."""

    episode = _require_mapping(episode, "episode")
    clip_id = str(episode.get("clip_id") or "").strip()
    if not clip_id:
        raise ValueError("expression-turn v8 episode is missing clip_id")
    if not is_expression_turn_v8_episode(episode):
        raise ValueError(f"{clip_id}: formal_episode_contract is not expression-turn v8")
    if episode.get("expression_turn_contract_version") != EXPRESSION_TURN_CONTRACT_VERSION:
        raise ValueError(f"{clip_id}: expression-turn contract version is invalid")
    if any(key in FORBIDDEN_LEGACY_KEYS for key in _walk_keys(episode)):
        raise ValueError(f"{clip_id}: v8 episode contains a legacy semantic-event field")
    if episode.get("accepted_for_training") is not True or episode.get(
        "eligibility_mode"
    ) != FORMAL_ELIGIBILITY_MODE:
        raise ValueError(f"{clip_id}: v8 episode is not adjudicated train-ready")

    review_record = _require_mapping(
        episode.get("expression_turn_review_record"),
        "expression_turn_review_record",
    )
    if review_record.get("clip_id") != clip_id:
        raise ValueError(f"{clip_id}: review record clip binding is inconsistent")
    review_sha256 = _canonical_sha256(review_record)
    if episode.get("expression_turn_review_record_sha256") != review_sha256:
        raise ValueError(f"{clip_id}: review record SHA256 is invalid")
    qualification_report = evaluate_expression_turn(review_record)
    if episode.get("qualification_report") != qualification_report:
        raise ValueError(f"{clip_id}: qualification report was not recomputed from blind evidence")
    report_sha256 = _canonical_sha256(qualification_report)
    if episode.get("qualification_report_sha256") != report_sha256:
        raise ValueError(f"{clip_id}: qualification report SHA256 is invalid")
    if episode.get("qualifications") != qualification_report["qualifications"]:
        raise ValueError(f"{clip_id}: three-tier qualification fields are inconsistent")

    base_eligible = qualification_report["qualifications"]["base_motion"]["eligible"]
    semantic_eligible = qualification_report["qualifications"]["semantic_conditioning"][
        "eligible"
    ]
    expressive_eligible = qualification_report["qualifications"][
        "expressive_conditioning"
    ]["eligible"]
    if base_eligible is not True:
        raise ValueError(f"{clip_id}: base motion qualification is required for training")
    expected_channels = {
        "motion": True,
        "semantic_conditioning": bool(semantic_eligible),
        "expressive_conditioning": bool(expressive_eligible),
    }
    if episode.get("training_channel_masks") != expected_channels:
        raise ValueError(f"{clip_id}: training channel masks do not match review tiers")
    if episode.get("training_qualification_tier") != qualification_report.get(
        "highest_qualification"
    ):
        raise ValueError(f"{clip_id}: admitted qualification tier is inconsistent")
    prompt_semantics_profile = episode.get(
        "prompt_semantics_profile", MOTION_FORM_PROMPT_PROFILE
    )
    if prompt_semantics_profile not in {
        MOTION_FORM_PROMPT_PROFILE,
        DYADIC_INTERACTION_PROMPT_PROFILE,
    }:
        raise ValueError(f"{clip_id}: unsupported prompt semantics profile")
    communicative_intent_eligible = bool(
        semantic_eligible
        and prompt_semantics_profile == DYADIC_INTERACTION_PROMPT_PROFILE
    )
    if episode.get("semantic_supervision_masks") != _expected_semantic_masks(
        bool(semantic_eligible),
        communicative_intent_eligible=communicative_intent_eligible,
    ):
        raise ValueError(f"{clip_id}: semantic supervision masks do not match action review")

    action_review = review_record.get("action_semantic_review") or {}
    if communicative_intent_eligible:
        motion_description = str(
            action_review.get("robot_observable_motion_description") or ""
        ).strip()
        intent_description = str(
            action_review.get("communicative_intent_description") or ""
        ).strip()
        expected_prompt = str(action_review.get("robot_prompt") or "").strip()
        if (
            action_review.get("communicative_intent_result") != "observable"
            or not motion_description
            or not intent_description
            or not expected_prompt
            or action_review.get("candidate_text") != expected_prompt
            or action_review.get("robot_prompt_sha256")
            != hashlib.sha256(expected_prompt.encode("utf-8")).hexdigest()
        ):
            raise ValueError(
                f"{clip_id}: dyadic prompt lacks independently observable motion/intent evidence"
            )
    else:
        expected_prompt = str(action_review.get("observable_description") or "").strip()
    prompt = str(episode.get("prompt") or "").strip()
    if semantic_eligible:
        if not expected_prompt or prompt != expected_prompt:
            raise ValueError(f"{clip_id}: prompt is not the blind observable action description")
        if action_review.get("candidate_text_provenance") != ACTION_TEXT_PROVENANCE:
            raise ValueError(f"{clip_id}: action text provenance is not independent")
        expected_prompt_provenance = (
            DYADIC_PROMPT_TEXT_PROVENANCE
            if communicative_intent_eligible
            else PROMPT_TEXT_PROVENANCE
        )
        if episode.get("prompt_text_provenance") != expected_prompt_provenance:
            raise ValueError(f"{clip_id}: formal prompt provenance is invalid")
        if episode.get("prompt_review_id") != action_review.get("review_id"):
            raise ValueError(f"{clip_id}: prompt is not bound to the action review")
        if episode.get("prompt_sha256") != hashlib.sha256(prompt.encode("utf-8")).hexdigest():
            raise ValueError(f"{clip_id}: prompt SHA256 is invalid")
    elif any(
        episode.get(field) not in (None, "")
        for field in ("prompt", "prompt_sha256", "prompt_text_provenance", "prompt_review_id")
    ):
        raise ValueError(f"{clip_id}: base-only motion may not carry a prompt condition")

    blind_affect_class = qualification_report.get("blind_affect_class")
    expected_emotion_mask = bool(expressive_eligible)
    for field in (
        "emotion_supervision_mask",
        "emotion_conditioning_mask",
        "affect_observable_supervision_mask",
    ):
        if episode.get(field) is not expected_emotion_mask:
            raise ValueError(f"{clip_id}: {field} does not match expressive qualification")
    if episode.get("emotion_id") != (
        blind_affect_class if expressive_eligible else None
    ):
        raise ValueError(f"{clip_id}: emotion_id is not the blind affect result")
    if episode.get("emotion_source") != (
        "independent_blind_affect_consensus_or_adjudication_v1"
        if expressive_eligible
        else None
    ):
        raise ValueError(f"{clip_id}: emotion source is not independent blind evidence")
    if episode.get("official_emotion_conditioning_enabled") is not False:
        raise ValueError(f"{clip_id}: official emotion conditioning is forbidden")
    if episode.get("official_category_conditioning_enabled") is not False:
        raise ValueError(f"{clip_id}: official category conditioning is forbidden")
    if episode.get("behavior_supervision_mask") is not False or episode.get(
        "behavior_id"
    ) is not None:
        raise ValueError(f"{clip_id}: v8 has no independently reviewed behavior label")

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
        raise ValueError(f"{clip_id}: expression-turn v8 requires observed body and head joints")
    fps = episode.get("fps")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isclose(float(fps), 30.0, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise ValueError(f"{clip_id}: expression-turn v8 must be exactly 30 Hz")
    fps = float(fps)

    training_segment = _require_mapping(episode.get("training_segment"), "training_segment")
    if training_segment.get("representation") != EXPRESSION_TURN_REPRESENTATION:
        raise ValueError(f"{clip_id}: training segment is not expression-turn v8")
    if training_segment.get("fixed_window_sec") is not None or training_segment.get(
        "cropped"
    ) is not False:
        raise ValueError(f"{clip_id}: v8 formal training may not crop to a fixed duration")
    start = training_segment.get("start_frame")
    end = training_segment.get("end_frame_exclusive")
    source_frames = training_segment.get("frame_count")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (start, end, source_frames)
    ) or start < 0 or end <= start or source_frames != end - start:
        raise ValueError(f"{clip_id}: training source interval is inconsistent")
    selected_level = int(qualification_report["selected_context_level"])
    selected_context = review_record["context_plan"]["levels"][selected_level]
    if (
        selected_context.get("start_frame") != start
        or selected_context.get("end_frame_exclusive") != end
    ):
        raise ValueError(f"{clip_id}: training interval differs from reviewed anonymous video")

    _validate_retarget_segment(
        episode,
        clip_id=clip_id,
        source_start=start,
        source_end=end,
        output_frames=int(actions.shape[0]),
        fps=fps,
    )
    physical_profile = episode.get(
        "physical_evidence_profile", HUMAN_RETARGET_PHYSICAL_PROFILE
    )
    required_gates = (
        REQUIRED_18D_GATES
        if physical_profile == HUMAN_RETARGET_PHYSICAL_PROFILE
        else NATIVE_ROBOT_REQUIRED_18D_GATES
        if physical_profile == NATIVE_ROBOT_PHYSICAL_PROFILE
        else set()
    )
    quality_gate = _require_mapping(episode.get("quality_gate"), "quality_gate")
    if (
        not required_gates
        or not required_gates.issubset(quality_gate)
        or any(quality_gate.get(gate) is not True for gate in required_gates)
        or any(isinstance(value, bool) and value is False for value in quality_gate.values())
    ):
        raise ValueError(f"{clip_id}: all required 18D physical quality gates must pass")
    if episode.get("retarget_qc_passed") is not True:
        raise ValueError(f"{clip_id}: retarget QC is not passed")
    if episode.get("quality_source_window_frames") != source_frames or episode.get(
        "quality_output_frame_count"
    ) != int(actions.shape[0]):
        raise ValueError(f"{clip_id}: source/output frame lineage is inconsistent")

    for field in LINEAGE_SHA256_FIELDS:
        if not _is_sha256(episode.get(field)):
            raise ValueError(f"{clip_id}: {field} must be a SHA256")
    for field in ("source_clip_id", "speaker_key", "source_group_key", "dataset_source"):
        if not str(episode.get(field) or "").strip():
            raise ValueError(f"{clip_id}: {field} is required for provenance and splitting")

    admission = _require_mapping(episode.get("training_admission"), "training_admission")
    expected_admission = {
        "contract": FORMAL_EPISODE_CONTRACT,
        "expression_turn_review_record_sha256": review_sha256,
        "qualification_report_sha256": report_sha256,
        "retarget_quality_record_sha256": episode["retarget_quality_record_sha256"],
        "training_qualification_tier": qualification_report["highest_qualification"],
        "training_channel_masks": expected_channels,
    }
    if admission != expected_admission:
        raise ValueError(f"{clip_id}: training admission is not bound to review and retarget evidence")

    _validate_condition(
        episode,
        clip_id=clip_id,
        semantic_eligible=bool(semantic_eligible),
        expressive_eligible=bool(expressive_eligible),
        blind_affect_class=blind_affect_class,
        require_attached_condition=require_attached_condition,
    )
    return {
        "clip_id": clip_id,
        "formal_episode_contract": FORMAL_EPISODE_CONTRACT,
        "highest_qualification": qualification_report["highest_qualification"],
        "training_channel_masks": expected_channels,
        "prompt_semantics_profile": prompt_semantics_profile,
        "frame_count": int(actions.shape[0]),
        "source_frame_count": int(source_frames),
    }


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_expression_turn_v8_condition_vectors(
    episodes: Sequence[Mapping[str, Any]],
    *,
    text_latents: Mapping[str, np.ndarray],
    style_contract: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build qualification-masked conditions without consulting source labels."""

    from upper_body_skeleton.ula_v2_conditioning import (
        extract_style_features,
        normalize_style_features,
    )

    if not episodes:
        raise ValueError("expression-turn v8 condition cache requires episodes")
    conditions = np.zeros((len(episodes), KIMODO_V2_CONDITION_DIM), dtype=np.float32)
    style_features: list[np.ndarray] = []
    style_controls: list[np.ndarray] = []
    expected_text_ids: set[str] = set()
    for row, episode in enumerate(episodes):
        report = validate_expression_turn_v8_episode(episode)
        clip_id = report["clip_id"]
        actions = np.asarray(episode["actions"], dtype=np.float32)
        features = extract_style_features(actions[:, :15], fps=float(episode["fps"]))
        controls = normalize_style_features(features, style_contract)
        conditions[row, STYLE_CONTROL_SLICE] = controls
        style_features.append(np.asarray(features, dtype=np.float32))
        style_controls.append(np.asarray(controls, dtype=np.float32))

        channels = report["training_channel_masks"]
        if channels["semantic_conditioning"]:
            expected_text_ids.add(clip_id)
            latent = np.asarray(text_latents.get(clip_id), dtype=np.float32)
            expected_shape = (KIMODO_V2_CONDITION_DIM - KIMODO_CONDITION_DIM,)
            if latent.shape != expected_shape or not np.isfinite(latent).all():
                raise ValueError(f"{clip_id}: Qwen latent must be finite {expected_shape}")
            norm = float(np.linalg.norm(latent))
            if not math.isfinite(norm) or norm <= 1e-8:
                raise ValueError(f"{clip_id}: Qwen latent has zero or invalid norm")
            conditions[row, KIMODO_CONDITION_DIM:] = latent / norm
        if channels["expressive_conditioning"]:
            emotion_id = str(episode["emotion_id"])
            conditions[row, KIMODO_EMOTION_SLICE.start + KIMODO_EMOTION_IDS.index(emotion_id)] = 1.0

        attached = dict(episode)
        attached["condition"] = conditions[row]
        validate_expression_turn_v8_episode(attached, require_attached_condition=True)

    supplied_text_ids = set(text_latents)
    if supplied_text_ids != expected_text_ids:
        missing = sorted(expected_text_ids - supplied_text_ids)
        extra = sorted(supplied_text_ids - expected_text_ids)
        raise ValueError(
            f"expression-turn Qwen latent membership differs: missing={missing[:5]}, extra={extra[:5]}"
        )
    return (
        conditions,
        np.stack(style_features).astype(np.float32),
        np.stack(style_controls).astype(np.float32),
    )


def build_expression_turn_v8_condition_cache(
    episodes: Sequence[Mapping[str, Any]],
    qwen_checkpoint: str | Path,
    output_path: str | Path,
    *,
    base_checkpoint: str | Path,
    device: str = "auto",
    local_files_only: bool = True,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Encode only blind-qualified text/affect and bind it to the random checkpoint."""

    import torch

    from upper_body_skeleton.cross_modal_latent import load_qwen_motion_text_encoder
    from upper_body_skeleton.ula_v2_18d_head import (
        sha256_file,
        validate_checkpoint_contract,
        validate_qwen_checkpoint_for_generator,
    )

    output_path = Path(output_path).resolve()
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    if output_path.suffix != ".npz":
        raise ValueError("expression-turn condition cache must use the .npz suffix")
    if output_path.exists() or metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite expression-turn condition cache: {output_path}")
    base_checkpoint = Path(base_checkpoint).resolve()
    qwen_checkpoint = Path(qwen_checkpoint).resolve()
    checkpoint = torch.load(base_checkpoint, map_location="cpu", weights_only=True)
    validate_checkpoint_contract(checkpoint, expected_action_dim=ACTION_DIM)
    if checkpoint.get("formal_episode_contract") != FORMAL_EPISODE_CONTRACT:
        raise ValueError("condition cache target is not an expression-turn v8 random checkpoint")
    validate_qwen_checkpoint_for_generator(checkpoint, qwen_checkpoint)
    style_contract = (checkpoint.get("v2_contracts") or {}).get("style") or {}
    if not _is_sha256(style_contract.get("sha256")):
        raise ValueError("expression-turn checkpoint has no bound style contract")

    semantic = [
        episode
        for episode in episodes
        if (episode.get("training_channel_masks") or {}).get("semantic_conditioning") is True
    ]
    text_latents: dict[str, np.ndarray] = {}
    encoder = None
    qwen_payload: Mapping[str, Any]
    if semantic:
        encoder, qwen_payload = load_qwen_motion_text_encoder(
            qwen_checkpoint, device=device, local_files_only=local_files_only
        )
        encoded = np.asarray(
            encoder.encode(
                [str(episode["prompt"]) for episode in semantic],
                batch_size=int(batch_size),
            ),
            dtype=np.float32,
        )
        expected_shape = (
            len(semantic),
            KIMODO_V2_CONDITION_DIM - KIMODO_CONDITION_DIM,
        )
        if encoded.shape != expected_shape:
            raise ValueError(f"Qwen encoder returned {encoded.shape}, expected {expected_shape}")
        text_latents = {
            str(episode["clip_id"]): encoded[index]
            for index, episode in enumerate(semantic)
        }
    else:
        qwen_payload = torch.load(qwen_checkpoint, map_location="cpu", weights_only=True)
    conditions, style_features, style_controls = build_expression_turn_v8_condition_vectors(
        episodes,
        text_latents=text_latents,
        style_contract=style_contract,
    )
    clip_ids = [str(episode["clip_id"]) for episode in episodes]
    prompts = [str(episode.get("prompt") or "") for episode in episodes]
    semantic_masks = np.asarray(
        [episode["training_channel_masks"]["semantic_conditioning"] for episode in episodes],
        dtype=np.bool_,
    )
    expressive_masks = np.asarray(
        [episode["training_channel_masks"]["expressive_conditioning"] for episode in episodes],
        dtype=np.bool_,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            clip_ids=np.asarray(clip_ids),
            prompts=np.asarray(prompts),
            conditions=conditions,
            semantic_conditioning_mask=semantic_masks,
            expressive_conditioning_mask=expressive_masks,
            style_features=style_features,
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
        "style_contract_sha256": style_contract["sha256"],
        "text_motion_contract_sha256": (
            (checkpoint.get("v2_contracts") or {}).get("text_motion_latent") or {}
        ).get("sha256"),
        "semantic_supervision_contract_sha256": (
            checkpoint.get("semantic_supervision_contract") or {}
        ).get("sha256"),
        "semantic_conditioned_count": int(semantic_masks.sum()),
        "expressive_conditioned_count": int(expressive_masks.sum()),
        "episodes": [
            {
                "clip_id": str(episode["clip_id"]),
                "prompt_sha256": episode.get("prompt_sha256"),
                "prompt_semantics_profile": episode.get(
                    "prompt_semantics_profile", MOTION_FORM_PROMPT_PROFILE
                ),
                "trajectory_sha256": episode["trajectory_sha256"],
                "expression_turn_review_record_sha256": episode[
                    "expression_turn_review_record_sha256"
                ],
                "qualification_report_sha256": episode["qualification_report_sha256"],
                "training_channel_masks": deepcopy(episode["training_channel_masks"]),
                "condition_sha256": _array_sha256(conditions[index]),
                "style_features": style_features[index].tolist(),
                "style_controls": style_controls[index].tolist(),
            }
            for index, episode in enumerate(episodes)
        ],
    }
    _atomic_json(metadata_path, metadata)
    if encoder is not None:
        del encoder
    return metadata


def attach_expression_turn_v8_condition_cache(
    episodes: Sequence[Mapping[str, Any]], cache_path: str | Path
) -> list[dict[str, Any]]:
    """Attach a checkpoint-bound v8 cache and revalidate every qualification mask."""

    import torch

    from upper_body_skeleton.ula_v2_18d_head import (
        sha256_file,
        validate_checkpoint_contract,
        validate_condition_cache_for_generator,
    )
    from upper_body_skeleton.ula_v2_conditioning import (
        extract_style_features,
        normalize_style_features,
    )

    cache_path = Path(cache_path).resolve()
    metadata_path = cache_path.with_suffix(cache_path.suffix + ".json")
    if not cache_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"expression-turn condition cache or metadata is missing: {cache_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        not isinstance(metadata, dict)
        or metadata.get("artifact_kind") != CONDITION_CACHE_ARTIFACT_KIND
        or metadata.get("schema_version") != CONDITION_CACHE_SCHEMA_VERSION
        or metadata.get("formal_episode_contract") != FORMAL_EPISODE_CONTRACT
        or metadata.get("unsafe_condition_cache") is not False
    ):
        raise ValueError("expression-turn condition cache metadata contract is invalid")
    if metadata.get("cache_sha256") != sha256_file(cache_path):
        raise ValueError("expression-turn condition cache SHA256 changed")
    with np.load(cache_path, allow_pickle=False) as payload:
        required = {
            "clip_ids",
            "prompts",
            "conditions",
            "semantic_conditioning_mask",
            "expressive_conditioning_mask",
            "style_features",
            "style_controls",
        }
        if set(payload.files) != required:
            raise ValueError("expression-turn condition cache arrays are incomplete or unexpected")
        clip_ids = payload["clip_ids"].astype(str).tolist()
        prompts = payload["prompts"].astype(str).tolist()
        conditions = payload["conditions"].astype(np.float32)
        semantic_masks = payload["semantic_conditioning_mask"]
        expressive_masks = payload["expressive_conditioning_mask"]
        style_features = payload["style_features"].astype(np.float32)
        style_controls = payload["style_controls"].astype(np.float32)
    count = len(episodes)
    if (
        int(metadata.get("count", -1)) != count
        or clip_ids != [str(episode["clip_id"]) for episode in episodes]
        or conditions.shape != (count, KIMODO_V2_CONDITION_DIM)
        or style_features.shape != (count, 3)
        or style_controls.shape != (count, 3)
        or semantic_masks.dtype != np.dtype(np.bool_)
        or expressive_masks.dtype != np.dtype(np.bool_)
    ):
        raise ValueError("expression-turn condition cache shape or episode order changed")
    generator_path = Path(str(metadata.get("generator_checkpoint") or "")).resolve()
    qwen_path = Path(str(metadata.get("qwen_checkpoint") or "")).resolve()
    if not generator_path.is_file() or not qwen_path.is_file():
        raise FileNotFoundError("expression-turn cache source checkpoint is missing")
    if metadata.get("generator_checkpoint_sha256") != sha256_file(generator_path):
        raise ValueError("expression-turn cache generator checkpoint changed")
    if metadata.get("qwen_checkpoint_sha256") != sha256_file(qwen_path):
        raise ValueError("expression-turn cache Qwen checkpoint changed")
    checkpoint = torch.load(generator_path, map_location="cpu", weights_only=True)
    validate_checkpoint_contract(checkpoint, expected_action_dim=ACTION_DIM)
    validate_condition_cache_for_generator(
        checkpoint,
        metadata,
        generator_checkpoint_path=generator_path,
    )
    style_contract = (checkpoint.get("v2_contracts") or {}).get("style") or {}
    metadata_rows = metadata.get("episodes")
    if not isinstance(metadata_rows, list) or len(metadata_rows) != count:
        raise ValueError("expression-turn cache episode bindings are missing")

    provenance = dict(metadata) | {
        "path": str(cache_path),
        "metadata_path": str(metadata_path),
    }
    attached: list[dict[str, Any]] = []
    for index, episode in enumerate(episodes):
        validate_expression_turn_v8_episode(episode)
        clip_id = str(episode["clip_id"])
        expected_prompt = str(episode.get("prompt") or "")
        if prompts[index] != expected_prompt:
            raise ValueError(f"{clip_id}: expression-turn cached prompt changed")
        expected_masks = episode["training_channel_masks"]
        if bool(semantic_masks[index]) is not expected_masks["semantic_conditioning"] or bool(
            expressive_masks[index]
        ) is not expected_masks["expressive_conditioning"]:
            raise ValueError(f"{clip_id}: expression-turn cache qualification masks changed")
        features = extract_style_features(
            np.asarray(episode["actions"], dtype=np.float32)[:, :15],
            fps=float(episode["fps"]),
        )
        controls = normalize_style_features(features, style_contract)
        if not np.array_equal(features, style_features[index]) or not np.array_equal(
            controls, style_controls[index]
        ):
            raise ValueError(f"{clip_id}: expression-turn cached trajectory style changed")
        row = metadata_rows[index]
        expected_binding = {
            "clip_id": clip_id,
            "prompt_sha256": episode.get("prompt_sha256"),
            "prompt_semantics_profile": episode.get(
                "prompt_semantics_profile", MOTION_FORM_PROMPT_PROFILE
            ),
            "trajectory_sha256": episode["trajectory_sha256"],
            "expression_turn_review_record_sha256": episode[
                "expression_turn_review_record_sha256"
            ],
            "qualification_report_sha256": episode["qualification_report_sha256"],
            "training_channel_masks": deepcopy(expected_masks),
            "condition_sha256": _array_sha256(conditions[index]),
            "style_features": style_features[index].tolist(),
            "style_controls": style_controls[index].tolist(),
        }
        if row != expected_binding:
            raise ValueError(f"{clip_id}: expression-turn cache review/trajectory binding changed")
        item = dict(episode)
        item["condition"] = np.ascontiguousarray(conditions[index])
        item["condition_cache_provenance"] = provenance
        validate_expression_turn_v8_episode(item, require_attached_condition=True)
        attached.append(item)
    return attached


__all__ = [
    "CHANNEL_MASK_KEYS",
    "CONDITION_CACHE_ARTIFACT_KIND",
    "CONDITION_CACHE_SCHEMA_VERSION",
    "EXPRESSION_TURN_CONTRACT_VERSION",
    "EXPRESSION_TURN_REPRESENTATION",
    "DYADIC_INTERACTION_PROMPT_PROFILE",
    "DYADIC_PROMPT_TEXT_PROVENANCE",
    "FORMAL_ELIGIBILITY_MODE",
    "FORMAL_EPISODE_CONTRACT",
    "HUMAN_RETARGET_PHYSICAL_PROFILE",
    "NATIVE_ROBOT_PHYSICAL_PROFILE",
    "NATIVE_ROBOT_REQUIRED_18D_GATES",
    "NATIVE_ROBOT_RETARGET_SEGMENT_REPRESENTATION",
    "MOTION_FORM_PROMPT_PROFILE",
    "PROMPT_TEXT_PROVENANCE",
    "QUALIFICATION_TIERS",
    "attach_expression_turn_v8_condition_cache",
    "build_expression_turn_v8_condition_cache",
    "build_expression_turn_v8_condition_vectors",
    "is_expression_turn_v8_episode",
    "load_expression_turn_v8_episodes",
    "validate_expression_turn_v8_episode",
]
