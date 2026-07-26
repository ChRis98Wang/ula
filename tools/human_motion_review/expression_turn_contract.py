#!/usr/bin/env python3
"""Fail-closed acceptance contract for variable-length expression turns.

The v8 contract deliberately separates three questions:

* Does the robot motion contain a complete onset, apex, and offset?
* Does an independent reviewer recognize the proposed action semantics?
* Is affect observable in the silent robot rendering?

The answers form three explicit qualification tiers. Physical quality and a
complete arc admit reusable base motion. Independent action matching admits
semantic conditioning. Blind affect consensus (or independent blind
adjudication) additionally admits expressive conditioning. This module never
reads or compares an official emotion label.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


CONTRACT_VERSION = "beat2_expression_turn_v8"
ARTIFACT_KIND = "ula_v2_18d_expression_turn_acceptance"
CANDIDATE_ARTIFACT_KIND = "beat2_expression_turn_v8_candidate"
REPRESENTATION = "native_variable_length_expression_turn_v1"
CONTEXT_POLICY = "evidence_anchored_progressive_expansion_no_fixed_duration_v1"
ARC_PROTOCOL = "robot_expression_arc_blind_video_v1"
ACTION_PROTOCOL = "robot_action_semantics_blind_video_v1"
AFFECT_PROTOCOL = "robot_affect_blind_video_v1"
AFFECT_ADJUDICATION_PROTOCOL = "robot_affect_blind_consensus_adjudication_v1"
ACTION_TEXT_PROVENANCE = "independently_authored_robot_observable_text_v1"
MIN_AFFECT_CONFIDENCE = 0.7

PHASE_STATUSES = {"complete", "incomplete", "ambiguous", "not_observable", "pending"}
ACTION_RESULTS = {
    "observable_match",
    "mismatch",
    "ambiguous",
    "not_observable",
    "pending",
}
AFFECT_RESULTS = {"observable", "ambiguous", "not_observable", "pending"}
AFFECT_CLASSES = {"neutral", "sad", "happy", "angry", "surprise", "fear"}
SEMANTIC_SUPERVISION_MASK_KEYS = {
    "official_category",
    "robot_observable_motion_form",
    "communicative_intent",
    "prompt_text",
    "legacy_gesture",
}
SELECTION_STATUS_BY_KIND = {
    "stress100": "selected_stress_pending_retarget_qc",
    "representative100": "selected_representative_pending_retarget_qc",
}
SELECTION_LINEAGE_FIELDS = {
    "expression_turn_selection_kind",
    "expression_turn_selection_rank",
    "expression_turn_selection_status",
    "expression_turn_selection_record_sha256",
}
LEGACY_PILOT_LINEAGE_FIELDS = {
    "expression_turn_pilot_rank",
    "expression_turn_pilot_selection_status",
    "expression_turn_pilot_record_sha256",
}

COMPLETE_PHASE_BASES = {
    "onset": {"natural_rest_or_low_motion", "coherent_motion_entry"},
    "apex": {"distinct_motion_or_pose_peak"},
    "offset": {"natural_settle", "coherent_turn_handoff"},
}

# A duration may be reported as a diagnostic, but it may not select or reject a
# turn. In particular, no six-second crop or arbitrary minimum is admitted.
HARD_DURATION_KEYS = {
    "fixed_window_sec",
    "fixed_frame_count",
    "min_duration_sec",
    "minimum_duration_sec",
    "max_duration_sec",
    "maximum_duration_sec",
    "target_duration_sec",
    "min_frame_count",
    "minimum_frame_count",
    "max_frame_count",
    "maximum_frame_count",
}

FORBIDDEN_BLIND_KEYS = {
    "canonical_action",
    "canonical_prompt",
    "category",
    "emotion_id",
    "emotion_label",
    "event_label",
    "official_emotion",
    "official_gesture_category",
    "prompt",
    "source",
    "source_clip_id",
    "source_label",
    "source_text",
    "speaker_id",
    "speaker_key",
    "transcript",
}

CONTRACT_DEFINITION = {
    "contract_version": CONTRACT_VERSION,
    "representation": REPRESENTATION,
    "duration_policy": "diagnostic_only_no_minimum_maximum_or_fixed_duration_gate",
    "context_policy": CONTEXT_POLICY,
    "base_motion_qualification": [
        "18d_physical_qc_pass",
        "natural_onset_apex_offset_complete",
        "same_source_neighbor_safe_interval",
    ],
    "semantic_conditioning_qualification": [
        "base_motion_qualified",
        "independent_blind_action_semantics_observable_match",
    ],
    "expressive_conditioning_qualification": [
        "semantic_conditioning_qualified",
        "at_least_two_independent_blind_affect_reviews_reach_observable_consensus",
        "or_independent_blind_adjudication_resolves_multiple_reviews",
    ],
    "context_expansion": (
        "use_the_next_predeclared_natural_boundary_interval_when_motion_arc_or_"
        "action_evidence_is_incomplete"
    ),
    "reject_when": [
        "18d_physical_qc_failed",
        "motion_arc_incomplete_after_context_exhaustion",
        "action_semantics_unobservable_blocks_semantic_not_base_motion",
        "action_semantics_mismatch_blocks_semantic_not_base_motion",
        "neighbor_or_source_boundary_would_be_crossed",
        "blind_review_contains_label_or_identity_leakage",
        "fixed_or_hard_duration_rule_present",
    ],
    "affect_policy": (
        "separate_silent_blind_review_never_gates_base_motion;_consensus_or_"
        "independent_adjudication_is_required_for_expressive_qualification"
    ),
    "official_emotion_used": False,
}


class ExpressionTurnContractError(ValueError):
    """Raised when an expression-turn record violates the v8 contract."""


def contract_definition() -> dict[str, Any]:
    """Return a mutable copy of the public v8 contract declaration."""

    return copy.deepcopy(CONTRACT_DEFINITION)


def _require_mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExpressionTurnContractError(f"{path} must be an object")
    return value


def _require_nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExpressionTurnContractError(f"{path} must be a non-empty string")
    return value.strip()


def _require_frame(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExpressionTurnContractError(f"{path} must be an integer frame index")
    return value


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _validate_interval(value: object, path: str) -> tuple[int, int]:
    interval = _require_mapping(value, path)
    start = _require_frame(interval.get("start_frame"), f"{path}.start_frame")
    end = _require_frame(
        interval.get("end_frame_exclusive"), f"{path}.end_frame_exclusive"
    )
    if start < 0 or end <= start:
        raise ExpressionTurnContractError(f"{path} must be a non-empty half-open interval")
    return start, end


def _contains(outer: tuple[int, int], inner: tuple[int, int]) -> bool:
    return outer[0] <= inner[0] and inner[1] <= outer[1]


def _walk_items(value: object, path: str = "record"):
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            yield str(key), child, child_path
            yield from _walk_items(child, child_path)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            yield from _walk_items(child, f"{path}[{index}]")


def _validate_no_hard_duration_rule(record: Mapping[str, Any]) -> None:
    for key, value, path in _walk_items(record):
        if key.lower() in HARD_DURATION_KEYS and value is not None:
            raise ExpressionTurnContractError(
                f"{path} is forbidden: duration is diagnostic, not an acceptance gate"
            )


def _forbidden_blind_key(key: str) -> bool:
    lowered = key.lower()
    return bool(
        lowered in FORBIDDEN_BLIND_KEYS
        or lowered.startswith("official_")
        or lowered.startswith("source_")
        or "transcript" in lowered
    )


def _validate_blind_payload(payload: Mapping[str, Any], path: str) -> None:
    for key, _value, child_path in _walk_items(payload, path):
        if _forbidden_blind_key(key):
            raise ExpressionTurnContractError(
                f"{child_path} leaks source, semantic, or affect target metadata"
            )


def _validate_review_envelope(
    value: object,
    path: str,
    *,
    protocol: str,
    selected_level: int,
) -> tuple[Mapping[str, Any], str, str, str]:
    review = _require_mapping(value, path)
    _validate_blind_payload(review, path)
    if review.get("protocol_version") != protocol:
        raise ExpressionTurnContractError(f"{path}.protocol_version is invalid")
    review_id = _require_nonempty_string(review.get("review_id"), f"{path}.review_id")
    reviewer_id = _require_nonempty_string(
        review.get("reviewer_id"), f"{path}.reviewer_id"
    )
    video_sha256 = review.get("anonymous_video_sha256")
    if not _is_sha256(video_sha256):
        raise ExpressionTurnContractError(
            f"{path}.anonymous_video_sha256 must be a SHA256"
        )
    if review.get("context_level") != selected_level:
        raise ExpressionTurnContractError(f"{path} is not bound to selected context level")
    if review.get("audio_available") is not False:
        raise ExpressionTurnContractError(f"{path} must use silent evidence")
    if review.get("label_metadata_exposed") is not False:
        raise ExpressionTurnContractError(f"{path} exposed target metadata")
    return review, review_id, reviewer_id, str(video_sha256)


def _validate_context_plan(
    record: Mapping[str, Any],
) -> tuple[int, Mapping[str, Any], Mapping[str, Any] | None]:
    core = _validate_interval(record.get("core_interval"), "record.core_interval")
    plan = _require_mapping(record.get("context_plan"), "record.context_plan")
    if plan.get("policy") != CONTEXT_POLICY:
        raise ExpressionTurnContractError("record.context_plan.policy is invalid")
    if plan.get("same_source_only") is not True:
        raise ExpressionTurnContractError("context expansion must remain in one source")
    if plan.get("neighbor_crossing_allowed") is not False:
        raise ExpressionTurnContractError("context expansion may not cross neighbor turns")

    source = _validate_interval(plan.get("source_interval"), "record.context_plan.source_interval")
    admissible = _validate_interval(
        plan.get("admissible_interval"), "record.context_plan.admissible_interval"
    )
    if not _contains(source, admissible):
        raise ExpressionTurnContractError("admissible context lies outside its source")
    if not _contains(admissible, core):
        raise ExpressionTurnContractError("core interval lies outside admissible context")

    levels_value = plan.get("levels")
    if not isinstance(levels_value, list) or not levels_value:
        raise ExpressionTurnContractError("record.context_plan.levels must be non-empty")

    levels: list[Mapping[str, Any]] = []
    previous_interval: tuple[int, int] | None = None
    for index, raw_level in enumerate(levels_value):
        level = _require_mapping(raw_level, f"record.context_plan.levels[{index}]")
        if level.get("level") != index:
            raise ExpressionTurnContractError("context levels must be contiguous from zero")
        interval = _validate_interval(level, f"record.context_plan.levels[{index}]")
        if not _contains(admissible, interval) or not _contains(interval, core):
            raise ExpressionTurnContractError(
                f"context level {index} must contain the core and stay inside barriers"
            )
        _require_nonempty_string(
            level.get("left_boundary_basis"),
            f"record.context_plan.levels[{index}].left_boundary_basis",
        )
        _require_nonempty_string(
            level.get("right_boundary_basis"),
            f"record.context_plan.levels[{index}].right_boundary_basis",
        )
        if previous_interval is not None:
            if not _contains(interval, previous_interval) or interval == previous_interval:
                raise ExpressionTurnContractError(
                    "each context level must strictly and monotonically expand"
                )
            if level.get("parent_level") != index - 1:
                raise ExpressionTurnContractError("expanded context has invalid parent level")
            _require_nonempty_string(
                level.get("expansion_reason"),
                f"record.context_plan.levels[{index}].expansion_reason",
            )
        previous_interval = interval
        levels.append(level)

    selected_level = plan.get("selected_level")
    if isinstance(selected_level, bool) or not isinstance(selected_level, int):
        raise ExpressionTurnContractError("selected context level must be an integer")
    if not 0 <= selected_level < len(levels):
        raise ExpressionTurnContractError("selected context level is not in the plan")
    next_level = levels[selected_level + 1] if selected_level + 1 < len(levels) else None
    return selected_level, levels[selected_level], next_level


def _validate_arc_review(
    value: object,
    *,
    selected_level: int,
    selected_interval: tuple[int, int],
) -> tuple[dict[str, str], str, str, str]:
    review, review_id, reviewer_id, video_sha256 = _validate_review_envelope(
        value,
        "record.motion_arc_review",
        protocol=ARC_PROTOCOL,
        selected_level=selected_level,
    )
    statuses: dict[str, str] = {}
    complete_evidence: dict[str, int] = {}
    for phase in ("onset", "apex", "offset"):
        phase_review = _require_mapping(
            review.get(phase), f"record.motion_arc_review.{phase}"
        )
        status = phase_review.get("status")
        if status not in PHASE_STATUSES:
            raise ExpressionTurnContractError(
                f"record.motion_arc_review.{phase}.status is invalid"
            )
        statuses[phase] = str(status)
        evidence_frame = phase_review.get("evidence_frame")
        if evidence_frame is not None:
            frame = _require_frame(
                evidence_frame, f"record.motion_arc_review.{phase}.evidence_frame"
            )
            if not selected_interval[0] <= frame < selected_interval[1]:
                raise ExpressionTurnContractError(
                    f"record.motion_arc_review.{phase} evidence is outside the interval"
                )
        if status == "complete":
            if evidence_frame is None:
                raise ExpressionTurnContractError(
                    f"complete {phase} review requires an evidence frame"
                )
            basis = phase_review.get("basis")
            if basis not in COMPLETE_PHASE_BASES[phase]:
                raise ExpressionTurnContractError(
                    f"complete {phase} review lacks natural-boundary evidence"
                )
            complete_evidence[phase] = int(evidence_frame)

    if set(complete_evidence) == {"onset", "apex", "offset"}:
        if not (
            complete_evidence["onset"]
            < complete_evidence["apex"]
            < complete_evidence["offset"]
        ):
            raise ExpressionTurnContractError(
                "complete onset, apex, and offset evidence must be temporally ordered"
            )
    return statuses, review_id, reviewer_id, video_sha256


def _validate_action_review(
    value: object,
    *,
    selected_level: int,
) -> tuple[str, str | None, str | None, str | None]:
    if value is None:
        return "not_reviewed", None, None, None
    review, review_id, reviewer_id, video_sha256 = _validate_review_envelope(
        value,
        "record.action_semantic_review",
        protocol=ACTION_PROTOCOL,
        selected_level=selected_level,
    )
    result = review.get("result")
    if result not in ACTION_RESULTS:
        raise ExpressionTurnContractError("action semantic review result is invalid")
    if result != "pending":
        if review.get("candidate_text_provenance") != ACTION_TEXT_PROVENANCE:
            raise ExpressionTurnContractError(
                "action candidate text is not independently authored for robot observability"
            )
        if not _is_sha256(review.get("candidate_text_sha256")):
            raise ExpressionTurnContractError("action candidate text SHA256 is invalid")
    if result == "observable_match":
        _require_nonempty_string(
            review.get("observable_description"),
            "record.action_semantic_review.observable_description",
        )
    return str(result), review_id, reviewer_id, video_sha256


def _validate_affect_outcome(review: Mapping[str, Any], path: str) -> dict[str, Any]:
    result = review.get("result")
    if result not in AFFECT_RESULTS:
        raise ExpressionTurnContractError(f"{path}.result is invalid")
    predicted_class = review.get("predicted_class")
    confidence = review.get("confidence")
    if result == "observable":
        if predicted_class not in AFFECT_CLASSES:
            raise ExpressionTurnContractError(
                f"{path}.predicted_class is outside the affect ontology"
            )
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise ExpressionTurnContractError(f"{path}.confidence is invalid")
    elif predicted_class is not None:
        raise ExpressionTurnContractError(
            f"{path}: ambiguous, unobservable, or pending affect may not carry a class"
        )
    return {
        "result": str(result),
        "predicted_class": str(predicted_class) if predicted_class is not None else None,
        "confidence": float(confidence) if result == "observable" else None,
    }


def _validate_affect_review(
    value: object,
    *,
    selected_level: int,
    path: str,
) -> dict[str, Any]:
    review, review_id, reviewer_id, video_sha256 = _validate_review_envelope(
        value,
        path,
        protocol=AFFECT_PROTOCOL,
        selected_level=selected_level,
    )
    return {
        **_validate_affect_outcome(review, path),
        "review_id": review_id,
        "reviewer_id": reviewer_id,
        "video_sha256": video_sha256,
    }


def _validate_affect_adjudication(
    value: object,
    *,
    selected_level: int,
    reviews: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if value is None:
        return None
    if len(reviews) < 2:
        raise ExpressionTurnContractError(
            "affect adjudication requires at least two independent input reviews"
        )
    path = "record.affect_adjudication"
    review, review_id, reviewer_id, video_sha256 = _validate_review_envelope(
        value,
        path,
        protocol=AFFECT_ADJUDICATION_PROTOCOL,
        selected_level=selected_level,
    )
    input_review_ids = review.get("input_review_ids")
    expected_ids = {item["review_id"] for item in reviews}
    if (
        not isinstance(input_review_ids, list)
        or len(input_review_ids) != len(set(input_review_ids))
        or set(input_review_ids) != expected_ids
    ):
        raise ExpressionTurnContractError(
            "affect adjudication is not bound to the complete blind review set"
        )
    return {
        **_validate_affect_outcome(review, path),
        "review_id": review_id,
        "reviewer_id": reviewer_id,
        "video_sha256": video_sha256,
    }


def _validate_affect_evidence(
    record: Mapping[str, Any],
    *,
    selected_level: int,
    evidence_video_sha256: str,
    action_review_id: str | None,
    action_reviewer_id: str | None,
) -> dict[str, Any]:
    singular = record.get("affect_review")
    plural = record.get("affect_reviews")
    if singular is not None and plural is not None:
        raise ExpressionTurnContractError(
            "use affect_review or affect_reviews, not both"
        )
    if plural is not None and not isinstance(plural, list):
        raise ExpressionTurnContractError("record.affect_reviews must be a list")
    raw_reviews = list(plural) if plural is not None else ([] if singular is None else [singular])
    reviews = [
        _validate_affect_review(
            value,
            selected_level=selected_level,
            path=(
                f"record.affect_reviews[{index}]"
                if plural is not None
                else "record.affect_review"
            ),
        )
        for index, value in enumerate(raw_reviews)
    ]

    review_ids = [str(item["review_id"]) for item in reviews]
    reviewer_ids = [str(item["reviewer_id"]) for item in reviews]
    if len(review_ids) != len(set(review_ids)):
        raise ExpressionTurnContractError("affect blind review IDs must be unique")
    if len(reviewer_ids) != len(set(reviewer_ids)):
        raise ExpressionTurnContractError("affect blind reviewers must be independent")
    if action_review_id is not None and action_review_id in review_ids:
        raise ExpressionTurnContractError(
            "action semantics and affect require independent blind reviews"
        )
    if action_reviewer_id is not None and action_reviewer_id in reviewer_ids:
        raise ExpressionTurnContractError(
            "action semantics and affect require independent blind reviews"
        )
    if any(item["video_sha256"] != evidence_video_sha256 for item in reviews):
        raise ExpressionTurnContractError(
            "affect reviews are not bound to the selected anonymous video"
        )

    adjudication = _validate_affect_adjudication(
        record.get("affect_adjudication"),
        selected_level=selected_level,
        reviews=reviews,
    )
    if adjudication is not None:
        if adjudication["reviewer_id"] in set(reviewer_ids) | {
            action_reviewer_id
        } or adjudication["review_id"] in set(review_ids) | {action_review_id}:
            raise ExpressionTurnContractError(
                "affect adjudicator must be independent of action and affect reviewers"
            )
        if adjudication["video_sha256"] != evidence_video_sha256:
            raise ExpressionTurnContractError(
                "affect adjudication is not bound to the selected anonymous video"
            )

    qualified = False
    status = "not_reviewed"
    blind_class = None
    basis = None
    if adjudication is not None and (
        adjudication["result"] == "observable"
        and float(adjudication["confidence"]) >= MIN_AFFECT_CONFIDENCE
    ):
        qualified = True
        status = "observable_adjudicated"
        blind_class = adjudication["predicted_class"]
        basis = "independent_blind_adjudication"
    elif len(reviews) >= 2 and all(
        item["result"] == "observable"
        and float(item["confidence"]) >= MIN_AFFECT_CONFIDENCE
        for item in reviews
    ) and len({item["predicted_class"] for item in reviews}) == 1:
        qualified = True
        status = "observable_consensus"
        blind_class = reviews[0]["predicted_class"]
        basis = "independent_blind_review_consensus"
    elif reviews:
        results = {item["result"] for item in reviews}
        if results == {"not_observable"}:
            status = "not_observable"
        elif results == {"pending"}:
            status = "pending"
        elif results == {"ambiguous"}:
            status = "ambiguous"
        else:
            status = "conflicting_or_insufficient_consensus"

    observable_count = sum(item["result"] == "observable" for item in reviews)
    return {
        "qualified": qualified,
        "status": status,
        "basis": basis,
        "blind_affect_class": blind_class,
        "review_count": len(reviews),
        "observable_review_count": observable_count,
        "candidate_for_adjudication": bool(observable_count and not qualified),
    }


def _stable_record_sha256(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def validate_expression_turn_candidate(
    record: Mapping[str, Any],
    *,
    catalog_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a producer candidate without granting any training tier.

    Catalog construction can prove interval integrity and fail-closed masks. It
    cannot prove robot physical quality, visible action semantics, or affect.
    Those qualifications require the reviewed record accepted by
    :func:`evaluate_expression_turn`.
    """

    record = _require_mapping(record, "record")
    if record.get("artifact_kind") != CANDIDATE_ARTIFACT_KIND:
        raise ExpressionTurnContractError("candidate artifact_kind is invalid")
    clip_id = _require_nonempty_string(record.get("clip_id"), "record.clip_id")
    if record.get("representation") != REPRESENTATION:
        raise ExpressionTurnContractError("candidate representation is invalid")
    _validate_no_hard_duration_rule(record)

    selected_level, selected, _next = _validate_context_plan(record)
    selected_interval = _validate_interval(selected, "selected_context_level")
    segment = _require_mapping(record.get("training_segment"), "record.training_segment")
    segment_interval = _validate_interval(segment, "record.training_segment")
    if segment_interval != selected_interval:
        raise ExpressionTurnContractError(
            "candidate training segment is not the selected natural-boundary level"
        )
    frame_count = segment_interval[1] - segment_interval[0]
    if segment.get("frame_count") != frame_count:
        raise ExpressionTurnContractError("candidate training frame_count is inconsistent")
    if segment.get("representation") != REPRESENTATION:
        raise ExpressionTurnContractError("candidate segment representation is invalid")
    if segment.get("cropped") is not False or segment.get("fixed_window_sec") is not None:
        raise ExpressionTurnContractError("candidate must not be duration-cropped")
    if segment.get("duration_policy") != (
        "natural_rest_to_natural_rest_no_fixed_or_max_duration"
    ):
        raise ExpressionTurnContractError("candidate duration policy is invalid")

    fps = record.get("fps")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(float(fps))
        or float(fps) <= 0
    ):
        raise ExpressionTurnContractError("candidate fps is invalid")
    time_axes = _require_mapping(record.get("time_axes"), "record.time_axes")
    source_axis = _require_mapping(time_axes.get("source"), "record.time_axes.source")
    turn_axis = _require_mapping(time_axes.get("turn"), "record.time_axes.turn")
    if (
        source_axis.get("start_frame") != segment_interval[0]
        or source_axis.get("end_frame_exclusive") != segment_interval[1]
        or turn_axis.get("start_frame") != 0
        or turn_axis.get("end_frame_exclusive") != frame_count
        or source_axis.get("frame_count") != frame_count
        or turn_axis.get("frame_count") != frame_count
    ):
        raise ExpressionTurnContractError("candidate source/turn time axes are inconsistent")
    expected_sample_span = round(max(0, frame_count - 1) / float(fps), 6)
    for path, axis in (
        ("record.time_axes.source", source_axis),
        ("record.time_axes.turn", turn_axis),
    ):
        sample_span = axis.get("sample_span_sec")
        if (
            isinstance(sample_span, bool)
            or not isinstance(sample_span, (int, float))
            or not math.isclose(
                float(sample_span), expected_sample_span, rel_tol=0.0, abs_tol=1e-6
            )
        ):
            raise ExpressionTurnContractError(f"{path}.sample_span_sec is inconsistent")

    turn = _require_mapping(record.get("expression_turn"), "record.expression_turn")
    if turn.get("complete_motion_arc_verified") is not False:
        raise ExpressionTurnContractError(
            "producer candidate may not pre-verify motion arc completeness"
        )
    if "pending_blind_video_review" not in str(
        turn.get("automated_motion_arc_candidate_status") or ""
    ):
        raise ExpressionTurnContractError("candidate must remain pending blind arc review")
    peak = _require_mapping(turn.get("peak"), "record.expression_turn.peak")
    peak_frame = _require_frame(
        peak.get("source_frame"), "record.expression_turn.peak.source_frame"
    )
    if not segment_interval[0] < peak_frame < segment_interval[1] - 1:
        raise ExpressionTurnContractError("candidate apex is not inside onset and offset")
    included = turn.get("included_event_spans")
    if not isinstance(included, list) or not included:
        raise ExpressionTurnContractError("candidate must bind included event spans")
    if turn.get("included_event_count") != len(included):
        raise ExpressionTurnContractError("candidate included event count is inconsistent")
    for index, raw_span in enumerate(included):
        span = _require_mapping(raw_span, f"record.expression_turn.included_event_spans[{index}]")
        source_span = _require_mapping(
            span.get("source_time_axis"),
            f"record.expression_turn.included_event_spans[{index}].source_time_axis",
        )
        turn_span = _require_mapping(
            span.get("turn_time_axis"),
            f"record.expression_turn.included_event_spans[{index}].turn_time_axis",
        )
        if (
            turn_span.get("start_frame")
            != source_span.get("start_frame_floor") - segment_interval[0]
            or turn_span.get("end_frame_exclusive")
            != source_span.get("end_frame_exclusive_ceil") - segment_interval[0]
        ):
            raise ExpressionTurnContractError(
                "candidate included event source/turn axes are inconsistent"
            )

    masks = _require_mapping(
        record.get("semantic_supervision_masks"),
        "record.semantic_supervision_masks",
    )
    if set(masks) != SEMANTIC_SUPERVISION_MASK_KEYS or any(
        value is not False for value in masks.values()
    ):
        raise ExpressionTurnContractError(
            "producer candidate semantic supervision must be fully masked"
        )
    fail_closed_fields = (
        "emotion_supervision_mask",
        "official_emotion_conditioning_enabled",
        "affect_observable_supervision_mask",
        "accepted_for_training",
    )
    if any(record.get(field) is not False for field in fail_closed_fields):
        raise ExpressionTurnContractError(
            "producer candidate emotion and admission fields must be fail-closed"
        )
    if record.get("official_category_conditioning_enabled") is not False:
        raise ExpressionTurnContractError(
            "producer candidate may not use official category conditioning"
        )
    if record.get("canonical_prompt") is not None or record.get("canonical_action") is not None:
        raise ExpressionTurnContractError(
            "producer candidate may not pre-assign action text before blind review"
        )

    candidate_record_hash = record.get("expression_turn_record_sha256")
    if not _is_sha256(candidate_record_hash):
        raise ExpressionTurnContractError("candidate record SHA256 is invalid")
    candidate_payload = {
        key: value
        for key, value in record.items()
        if key
        not in (
            {"expression_turn_record_sha256"}
            | SELECTION_LINEAGE_FIELDS
            | LEGACY_PILOT_LINEAGE_FIELDS
        )
    }
    if _stable_record_sha256(candidate_payload) != candidate_record_hash:
        raise ExpressionTurnContractError("candidate record SHA256 does not match")
    if not _is_sha256(record.get("expression_turn_contract_sha256")):
        raise ExpressionTurnContractError("candidate contract SHA256 is invalid")

    present_selection_fields = SELECTION_LINEAGE_FIELDS.intersection(record)
    present_pilot_fields = LEGACY_PILOT_LINEAGE_FIELDS.intersection(record)
    if present_selection_fields and present_selection_fields != SELECTION_LINEAGE_FIELDS:
        raise ExpressionTurnContractError(
            "expression-turn selection lineage fields are incomplete"
        )
    if present_pilot_fields and present_pilot_fields != LEGACY_PILOT_LINEAGE_FIELDS:
        raise ExpressionTurnContractError("pilot candidate lineage fields are incomplete")
    if present_selection_fields and present_pilot_fields:
        raise ExpressionTurnContractError(
            "new selection lineage may not be mixed with legacy pilot lineage"
        )

    selection_kind = None
    selection_record_hash = None
    if present_selection_fields:
        selection_kind = record.get("expression_turn_selection_kind")
        if selection_kind not in SELECTION_STATUS_BY_KIND:
            raise ExpressionTurnContractError("expression-turn selection kind is invalid")
        rank = record.get("expression_turn_selection_rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ExpressionTurnContractError("expression-turn selection rank is invalid")
        if record.get("expression_turn_selection_status") != SELECTION_STATUS_BY_KIND[
            str(selection_kind)
        ]:
            raise ExpressionTurnContractError("expression-turn selection status is invalid")
        selection_record_hash = record.get(
            "expression_turn_selection_record_sha256"
        )
        if not _is_sha256(selection_record_hash):
            raise ExpressionTurnContractError(
                "expression-turn selection record SHA256 is invalid"
            )
        selection_payload = {
            key: value
            for key, value in record.items()
            if key != "expression_turn_selection_record_sha256"
        }
        if _stable_record_sha256(selection_payload) != selection_record_hash:
            raise ExpressionTurnContractError(
                "expression-turn selection record SHA256 does not match"
            )

    pilot_record_hash = None
    if present_pilot_fields:
        rank = record.get("expression_turn_pilot_rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ExpressionTurnContractError("pilot rank is invalid")
        if record.get("expression_turn_pilot_selection_status") != (
            "selected_stratified_pending_retarget_qc"
        ):
            raise ExpressionTurnContractError("pilot selection status is invalid")
        pilot_record_hash = record.get("expression_turn_pilot_record_sha256")
        if not _is_sha256(pilot_record_hash):
            raise ExpressionTurnContractError("pilot record SHA256 is invalid")
        pilot_payload = {
            key: value
            for key, value in record.items()
            if key != "expression_turn_pilot_record_sha256"
        }
        if _stable_record_sha256(pilot_payload) != pilot_record_hash:
            raise ExpressionTurnContractError("pilot record SHA256 does not match")

    manifest_hash = None
    if catalog_binding is not None:
        binding = _require_mapping(catalog_binding, "catalog_binding")
        required_binding = {
            "retarget_input_manifest_sha256",
            "expression_turn_contract_sha256",
            "source_inventory_manifest_sha256",
            "split_assignment_manifest_sha256",
        }
        if not required_binding.issubset(binding):
            missing = sorted(required_binding.difference(binding))
            raise ExpressionTurnContractError(
                f"catalog binding is missing required fields: {missing}"
            )
        for field in required_binding:
            if not _is_sha256(binding.get(field)):
                raise ExpressionTurnContractError(
                    f"catalog_binding.{field} is not a SHA256"
                )
        for field in required_binding - {"retarget_input_manifest_sha256"}:
            if record.get(field) != binding.get(field):
                raise ExpressionTurnContractError(
                    f"candidate {field} does not match catalog binding"
                )
        bound_selection_kind = binding.get("selection_kind")
        if bound_selection_kind is not None:
            if bound_selection_kind not in SELECTION_STATUS_BY_KIND:
                raise ExpressionTurnContractError(
                    "catalog binding selection_kind is invalid"
                )
            if present_pilot_fields:
                raise ExpressionTurnContractError(
                    "formal catalog binding rejects legacy pilot lineage"
                )
            if selection_kind != bound_selection_kind:
                raise ExpressionTurnContractError(
                    "candidate selection kind does not match catalog binding"
                )
        if (
            binding.get("require_selection_record") is True
            and selection_record_hash is None
        ):
            raise ExpressionTurnContractError(
                "catalog binding requires a unified selection record"
            )
        if binding.get("require_pilot_record") is True and pilot_record_hash is None:
            raise ExpressionTurnContractError(
                "catalog binding requires a selected pilot record"
            )
        manifest_hash = str(binding["retarget_input_manifest_sha256"])

    selected_record_hash = _stable_record_sha256(record)
    inventory_record_hash = (
        selection_record_hash or pilot_record_hash or str(candidate_record_hash)
    )

    return {
        "contract_version": CONTRACT_VERSION,
        "clip_id": clip_id,
        "boundary_candidate_valid": True,
        "selected_context_level": selected_level,
        "selection_kind": selection_kind or ("legacy_pilot" if pilot_record_hash else None),
        "lineage": {
            "inventory_record_sha256": inventory_record_hash,
            "upstream_inventory_record_sha256": str(candidate_record_hash),
            "selected_record_sha256": selected_record_hash,
            "retarget_input_manifest_sha256": manifest_hash,
        },
        "qualifications": {
            "base_motion": {
                "eligible": False,
                "status": "pending_18d_retarget_and_blind_arc_review",
            },
            "semantic_conditioning": {
                "eligible": False,
                "status": "pending_independent_action_semantic_review",
            },
            "expressive_conditioning": {
                "eligible": False,
                "status": "pending_independent_affect_consensus_or_adjudication",
            },
        },
        "official_emotion_used": False,
        "emotion_supervision_mask": False,
        "accepted_for_training": False,
    }


def _next_context_result(
    *,
    next_level: Mapping[str, Any] | None,
    exhausted_reason: str,
) -> tuple[str, list[str], int | None, dict[str, int] | None]:
    if next_level is None:
        return "reject", [exhausted_reason], None, None
    return (
        "expand_context_for_base_motion",
        ["more_same_source_natural_boundary_context_available"],
        int(next_level["level"]),
        {
            "start_frame": int(next_level["start_frame"]),
            "end_frame_exclusive": int(next_level["end_frame_exclusive"]),
        },
    )


def evaluate_expression_turn(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and evaluate one v8 expression-turn review record.

    Structural, privacy, or provenance violations raise
    :class:`ExpressionTurnContractError`. The result reports independent
    ``base_motion``, ``semantic_conditioning``, and ``expressive_conditioning``
    qualifications so missing semantics or affect never discards usable motion.
    """

    record = _require_mapping(record, "record")
    if record.get("artifact_kind") != ARTIFACT_KIND:
        raise ExpressionTurnContractError("record.artifact_kind is invalid")
    if record.get("contract_version") != CONTRACT_VERSION:
        raise ExpressionTurnContractError("record.contract_version is invalid")
    clip_id = _require_nonempty_string(record.get("clip_id"), "record.clip_id")
    _validate_no_hard_duration_rule(record)

    if record.get("emotion_conditioning_enabled") not in (None, False):
        raise ExpressionTurnContractError("v8 may not enable emotion conditioning")
    if record.get("emotion_supervision_mask") not in (None, False):
        raise ExpressionTurnContractError("v8 may not enable emotion supervision")
    if record.get("official_category_conditioning_enabled") not in (None, False):
        raise ExpressionTurnContractError("official category may not condition v8")

    selected_level, selected, next_level = _validate_context_plan(record)
    selected_interval = _validate_interval(selected, "selected_context_level")
    phase_statuses, _arc_id, _arc_reviewer, arc_video = _validate_arc_review(
        record.get("motion_arc_review"),
        selected_level=selected_level,
        selected_interval=selected_interval,
    )
    action_result, action_id, action_reviewer, action_video = _validate_action_review(
        record.get("action_semantic_review"), selected_level=selected_level
    )
    if action_video is not None and action_video != arc_video:
        raise ExpressionTurnContractError(
            "arc and action reviews are not bound to the same anonymous video"
        )
    affect = _validate_affect_evidence(
        record,
        selected_level=selected_level,
        evidence_video_sha256=arc_video,
        action_review_id=action_id,
        action_reviewer_id=action_reviewer,
    )

    physical_qc = _require_mapping(record.get("physical_qc"), "record.physical_qc")
    if physical_qc.get("passed") is not True:
        decision = "reject"
        reasons = ["18d_physical_qc_failed"]
        base_motion_eligible = False
        base_status = "ineligible_physical_qc"
        next_context_level = None
        next_interval = None
    elif "pending" in phase_statuses.values():
        decision = "needs_human"
        reasons = ["motion_arc_review_pending"]
        base_motion_eligible = False
        base_status = "pending_arc_review"
        next_context_level = None
        next_interval = None
    elif set(phase_statuses.values()) != {"complete"}:
        decision, reasons, next_context_level, next_interval = _next_context_result(
            next_level=next_level,
            exhausted_reason="motion_arc_incomplete_after_context_exhaustion",
        )
        base_motion_eligible = False
        base_status = (
            "expand_context_for_complete_arc"
            if next_level is not None
            else "ineligible_incomplete_arc"
        )
    else:
        base_motion_eligible = True
        base_status = "qualified"
        reasons = []
        if action_result in {"ambiguous", "not_observable"} and next_level is not None:
            decision = "retain_base_motion_expand_context_for_semantics"
            next_context_level = int(next_level["level"])
            next_interval = {
                "start_frame": int(next_level["start_frame"]),
                "end_frame_exclusive": int(next_level["end_frame_exclusive"]),
            }
        else:
            decision = "retain_base_motion"
            next_context_level = None
            next_interval = None

    semantic_eligible = base_motion_eligible and action_result == "observable_match"
    if not base_motion_eligible:
        semantic_status = "blocked_by_base_motion"
    elif semantic_eligible:
        semantic_status = "qualified"
    elif action_result == "mismatch":
        semantic_status = "ineligible_action_mismatch"
    elif action_result in {"not_reviewed", "pending"}:
        semantic_status = "pending_action_review"
    elif next_level is not None:
        semantic_status = "expand_context_available"
    else:
        semantic_status = "ineligible_action_unobservable_after_context_exhaustion"

    expressive_eligible = semantic_eligible and bool(affect["qualified"])
    if not semantic_eligible:
        expressive_status = "blocked_by_semantic_conditioning"
    elif expressive_eligible:
        expressive_status = f"qualified_{affect['basis']}"
    else:
        expressive_status = f"ineligible_affect_{affect['status']}"

    if expressive_eligible:
        highest_qualification = "expressive_conditioning"
    elif semantic_eligible:
        highest_qualification = "semantic_conditioning"
    elif base_motion_eligible:
        highest_qualification = "base_motion"
    else:
        highest_qualification = "none"

    return {
        "contract_version": CONTRACT_VERSION,
        "clip_id": clip_id,
        "decision": decision,
        "reasons": reasons,
        "selected_context_level": selected_level,
        "next_context_level": next_context_level,
        "next_interval": next_interval,
        "highest_qualification": highest_qualification,
        "qualifications": {
            "base_motion": {
                "eligible": base_motion_eligible,
                "status": base_status,
            },
            "semantic_conditioning": {
                "eligible": semantic_eligible,
                "status": semantic_status,
            },
            "expressive_conditioning": {
                "eligible": expressive_eligible,
                "status": expressive_status,
            },
        },
        "motion_training_eligible": base_motion_eligible,
        "base_motion_eligible": base_motion_eligible,
        "semantic_conditioning_eligible": semantic_eligible,
        "expressive_conditioning_eligible": expressive_eligible,
        "action_semantics_verified": semantic_eligible,
        "affect_review_status": affect["status"],
        "affect_review_count": affect["review_count"],
        "affect_observable_review_count": affect["observable_review_count"],
        "blind_affect_class": affect["blind_affect_class"],
        "affect_candidate_for_separate_adjudication": affect[
            "candidate_for_adjudication"
        ],
        "emotion_supervision_candidate": expressive_eligible,
        "emotion_conditioning_enabled": False,
        "emotion_supervision_mask": False,
        "official_emotion_used": False,
        "duration_gate_used": False,
    }


__all__ = [
    "ACTION_PROTOCOL",
    "AFFECT_ADJUDICATION_PROTOCOL",
    "AFFECT_PROTOCOL",
    "ARC_PROTOCOL",
    "ARTIFACT_KIND",
    "CANDIDATE_ARTIFACT_KIND",
    "CONTRACT_VERSION",
    "CONTEXT_POLICY",
    "ExpressionTurnContractError",
    "contract_definition",
    "evaluate_expression_turn",
    "validate_expression_turn_candidate",
]
