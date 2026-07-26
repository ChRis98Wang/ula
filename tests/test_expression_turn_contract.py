import copy
import hashlib
import json

import pytest

from tools.human_motion_review.expression_turn_contract import (
    ACTION_PROTOCOL,
    AFFECT_ADJUDICATION_PROTOCOL,
    AFFECT_PROTOCOL,
    ARC_PROTOCOL,
    ARTIFACT_KIND,
    CANDIDATE_ARTIFACT_KIND,
    CONTRACT_VERSION,
    CONTEXT_POLICY,
    ExpressionTurnContractError,
    contract_definition,
    evaluate_expression_turn,
    validate_expression_turn_candidate,
)


VIDEO_SHA256 = "a" * 64
TEXT_SHA256 = "b" * 64


def _review_envelope(protocol, review_id, reviewer_id):
    return {
        "protocol_version": protocol,
        "review_id": review_id,
        "reviewer_id": reviewer_id,
        "anonymous_video_sha256": VIDEO_SHA256,
        "context_level": 0,
        "audio_available": False,
        "label_metadata_exposed": False,
    }


def _record(*, with_expansion=True):
    levels = [
        {
            "level": 0,
            "start_frame": 100,
            "end_frame_exclusive": 103,
            "left_boundary_basis": "natural_low_motion_basin",
            "right_boundary_basis": "natural_low_motion_basin",
        }
    ]
    if with_expansion:
        levels.append(
            {
                "level": 1,
                "parent_level": 0,
                "start_frame": 95,
                "end_frame_exclusive": 110,
                "left_boundary_basis": "preceding_natural_low_motion_basin",
                "right_boundary_basis": "following_natural_low_motion_basin",
                "expansion_reason": "resolve_arc_or_semantic_ambiguity",
            }
        )
    arc = _review_envelope(ARC_PROTOCOL, "arc-review", "arc-reviewer")
    arc.update(
        {
            "onset": {
                "status": "complete",
                "evidence_frame": 100,
                "basis": "natural_rest_or_low_motion",
            },
            "apex": {
                "status": "complete",
                "evidence_frame": 101,
                "basis": "distinct_motion_or_pose_peak",
            },
            "offset": {
                "status": "complete",
                "evidence_frame": 102,
                "basis": "natural_settle",
            },
        }
    )
    action = _review_envelope(ACTION_PROTOCOL, "action-review", "action-reviewer")
    action.update(
        {
            "result": "observable_match",
            "observable_description": "Raises one forearm, holds, then returns.",
            "candidate_text_sha256": TEXT_SHA256,
            "candidate_text_provenance": (
                "independently_authored_robot_observable_text_v1"
            ),
        }
    )
    affect = _review_envelope(AFFECT_PROTOCOL, "affect-review", "affect-reviewer")
    affect.update(
        {
            "result": "not_observable",
            "predicted_class": None,
            "confidence": 0.94,
        }
    )
    return {
        "artifact_kind": ARTIFACT_KIND,
        "contract_version": CONTRACT_VERSION,
        "clip_id": "short-complete-turn",
        "fixed_window_sec": None,
        "duration_sec": 0.1,
        "core_interval": {"start_frame": 100, "end_frame_exclusive": 103},
        "context_plan": {
            "policy": CONTEXT_POLICY,
            "same_source_only": True,
            "neighbor_crossing_allowed": False,
            "source_interval": {"start_frame": 0, "end_frame_exclusive": 1000},
            "admissible_interval": {
                "start_frame": 90,
                "end_frame_exclusive": 200,
            },
            "selected_level": 0,
            "levels": levels,
        },
        "physical_qc": {"passed": True},
        "motion_arc_review": arc,
        "action_semantic_review": action,
        "affect_review": affect,
        "official_category_conditioning_enabled": False,
        "emotion_conditioning_enabled": False,
        "emotion_supervision_mask": False,
    }


def _candidate_record():
    review_record = _record()
    candidate = {
        "artifact_kind": CANDIDATE_ARTIFACT_KIND,
        "clip_id": "producer-candidate",
        "representation": "native_variable_length_expression_turn_v1",
        "fps": 30.0,
        "core_interval": copy.deepcopy(review_record["core_interval"]),
        "context_plan": copy.deepcopy(review_record["context_plan"]),
        "training_segment": {
            "representation": "native_variable_length_expression_turn_v1",
            "start_frame": 100,
            "end_frame_exclusive": 103,
            "frame_count": 3,
            "fixed_window_sec": None,
            "cropped": False,
            "duration_policy": "natural_rest_to_natural_rest_no_fixed_or_max_duration",
        },
        "time_axes": {
            "source": {
                "start_frame": 100,
                "end_frame_exclusive": 103,
                "frame_count": 3,
                "sample_span_sec": 0.066667,
            },
            "turn": {
                "start_frame": 0,
                "end_frame_exclusive": 3,
                "frame_count": 3,
                "sample_span_sec": 0.066667,
            },
        },
        "expression_turn": {
            "complete_motion_arc_verified": False,
            "automated_motion_arc_candidate_status": (
                "natural_boundary_and_apex_proxy_pending_blind_video_review"
            ),
            "peak": {"source_frame": 101},
            "included_event_count": 1,
            "included_event_spans": [
                {
                    "source_time_axis": {
                        "start_frame_floor": 100,
                        "end_frame_exclusive_ceil": 103,
                    },
                    "turn_time_axis": {
                        "start_frame": 0,
                        "end_frame_exclusive": 3,
                    },
                }
            ],
        },
        "semantic_supervision_masks": {
            "official_category": False,
            "robot_observable_motion_form": False,
            "communicative_intent": False,
            "prompt_text": False,
            "legacy_gesture": False,
        },
        "emotion_supervision_mask": False,
        "official_emotion_conditioning_enabled": False,
        "affect_observable_supervision_mask": False,
        "official_category_conditioning_enabled": False,
        "canonical_prompt": None,
        "canonical_action": None,
        "accepted_for_training": False,
        "expression_turn_contract_sha256": "c" * 64,
        "source_inventory_manifest_sha256": "d" * 64,
        "split_assignment_manifest_sha256": "e" * 64,
    }
    candidate["expression_turn_record_sha256"] = hashlib.sha256(
        json.dumps(
            candidate,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return candidate


def _selection_candidate_record(kind="stress100"):
    candidate = _candidate_record()
    statuses = {
        "stress100": "selected_stress_pending_retarget_qc",
        "representative100": "selected_representative_pending_retarget_qc",
    }
    candidate.update(
        {
            "expression_turn_selection_kind": kind,
            "expression_turn_selection_rank": 1,
            "expression_turn_selection_status": statuses[kind],
        }
    )
    candidate["expression_turn_selection_record_sha256"] = hashlib.sha256(
        json.dumps(
            candidate,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return candidate


def test_contract_declares_no_hard_duration_or_official_emotion_gate():
    contract = contract_definition()
    assert contract["duration_policy"] == (
        "diagnostic_only_no_minimum_maximum_or_fixed_duration_gate"
    )
    assert contract["official_emotion_used"] is False


def test_three_frame_complete_turn_is_accepted_without_a_minimum_duration_gate():
    result = evaluate_expression_turn(_record())
    assert result["decision"] == "retain_base_motion"
    assert result["motion_training_eligible"] is True
    assert result["base_motion_eligible"] is True
    assert result["semantic_conditioning_eligible"] is True
    assert result["expressive_conditioning_eligible"] is False
    assert result["duration_gate_used"] is False


def test_long_turn_does_not_pass_when_onset_is_incomplete_and_expands_context():
    record = _record()
    record["core_interval"] = {"start_frame": 100, "end_frame_exclusive": 400}
    record["context_plan"]["admissible_interval"]["end_frame_exclusive"] = 500
    record["context_plan"]["levels"][0]["end_frame_exclusive"] = 400
    record["context_plan"]["levels"][1]["end_frame_exclusive"] = 410
    record["motion_arc_review"]["onset"] = {"status": "incomplete"}
    record["motion_arc_review"]["apex"]["evidence_frame"] = 250
    record["motion_arc_review"]["offset"]["evidence_frame"] = 399
    record["duration_sec"] = 10.0

    result = evaluate_expression_turn(record)
    assert result["decision"] == "expand_context_for_base_motion"
    assert result["next_context_level"] == 1
    assert result["next_interval"] == {
        "start_frame": 95,
        "end_frame_exclusive": 410,
    }


def test_incomplete_arc_is_rejected_only_after_safe_context_is_exhausted():
    record = _record(with_expansion=False)
    record["motion_arc_review"]["offset"] = {"status": "not_observable"}
    result = evaluate_expression_turn(record)
    assert result["decision"] == "reject"
    assert result["reasons"] == [
        "motion_arc_incomplete_after_context_exhaustion"
    ]


def test_fixed_six_second_crop_is_forbidden_but_duration_diagnostic_is_allowed():
    fixed = _record()
    fixed["fixed_window_sec"] = 6.0
    with pytest.raises(ExpressionTurnContractError, match="duration is diagnostic"):
        evaluate_expression_turn(fixed)

    diagnostic = _record()
    diagnostic["duration_sec"] = 6.0
    assert evaluate_expression_turn(diagnostic)["decision"] == "retain_base_motion"


def test_minimum_duration_rule_is_forbidden_even_when_nested():
    record = _record()
    record["context_plan"]["minimum_duration_sec"] = 1.0
    with pytest.raises(ExpressionTurnContractError, match="duration is diagnostic"):
        evaluate_expression_turn(record)


def test_progressive_context_must_be_nested_and_neighbor_safe():
    record = _record()
    record["context_plan"]["levels"][1]["start_frame"] = 100
    record["context_plan"]["levels"][1]["end_frame_exclusive"] = 103
    with pytest.raises(ExpressionTurnContractError, match="strictly and monotonically"):
        evaluate_expression_turn(record)

    record = _record()
    record["context_plan"]["neighbor_crossing_allowed"] = True
    with pytest.raises(ExpressionTurnContractError, match="may not cross neighbor"):
        evaluate_expression_turn(record)


def test_action_and_affect_reviews_must_be_independent():
    record = _record()
    record["affect_review"]["reviewer_id"] = "action-reviewer"
    with pytest.raises(ExpressionTurnContractError, match="independent blind reviews"):
        evaluate_expression_turn(record)


def test_unobservable_affect_does_not_reject_motion_and_stays_masked():
    result = evaluate_expression_turn(_record())
    assert result["decision"] == "retain_base_motion"
    assert result["affect_review_status"] == "not_observable"
    assert result["base_motion_eligible"] is True
    assert result["semantic_conditioning_eligible"] is True
    assert result["expressive_conditioning_eligible"] is False
    assert result["emotion_conditioning_enabled"] is False
    assert result["emotion_supervision_mask"] is False


def test_observable_affect_is_only_a_separate_adjudication_candidate():
    record = _record()
    record["affect_review"].update(
        {"result": "observable", "predicted_class": "happy", "confidence": 0.91}
    )
    record["official_metadata"] = {"emotion": "angry"}
    first = evaluate_expression_turn(record)
    record["official_metadata"]["emotion"] = "happy"
    second = evaluate_expression_turn(record)
    assert first == second
    assert first["affect_candidate_for_separate_adjudication"] is True
    assert first["expressive_conditioning_eligible"] is False
    assert first["official_emotion_used"] is False
    assert first["emotion_supervision_mask"] is False


def test_official_label_leakage_inside_blind_review_is_rejected():
    record = _record()
    record["affect_review"]["official_emotion"] = "happy"
    with pytest.raises(ExpressionTurnContractError, match="leaks source"):
        evaluate_expression_turn(record)

    record = _record()
    record["action_semantic_review"]["source_clip_id"] = "speaker_clip"
    with pytest.raises(ExpressionTurnContractError, match="leaks source"):
        evaluate_expression_turn(record)


def test_action_mismatch_blocks_semantics_but_preserves_base_motion():
    mismatch = _record()
    mismatch["action_semantic_review"]["result"] = "mismatch"
    result = evaluate_expression_turn(mismatch)
    assert result["decision"] == "retain_base_motion"
    assert result["base_motion_eligible"] is True
    assert result["semantic_conditioning_eligible"] is False
    assert result["expressive_conditioning_eligible"] is False
    assert result["qualifications"]["semantic_conditioning"]["status"] == (
        "ineligible_action_mismatch"
    )


def test_physical_qc_failure_rejects_all_three_qualification_tiers():
    failed_qc = _record()
    failed_qc["physical_qc"]["passed"] = False
    result = evaluate_expression_turn(failed_qc)
    assert result["reasons"] == ["18d_physical_qc_failed"]
    assert result["base_motion_eligible"] is False
    assert result["semantic_conditioning_eligible"] is False
    assert result["expressive_conditioning_eligible"] is False


def test_action_ambiguity_can_expand_but_never_discards_complete_base_motion():
    record = _record()
    record["action_semantic_review"]["result"] = "ambiguous"
    result = evaluate_expression_turn(record)
    assert result["decision"] == "retain_base_motion_expand_context_for_semantics"
    assert result["base_motion_eligible"] is True
    assert result["semantic_conditioning_eligible"] is False

    exhausted = _record(with_expansion=False)
    exhausted["action_semantic_review"]["result"] = "ambiguous"
    result = evaluate_expression_turn(exhausted)
    assert result["decision"] == "retain_base_motion"
    assert result["base_motion_eligible"] is True
    assert result["semantic_conditioning_eligible"] is False
    assert result["qualifications"]["semantic_conditioning"]["status"] == (
        "ineligible_action_unobservable_after_context_exhaustion"
    )


def test_complete_arc_evidence_must_be_inside_and_temporally_ordered():
    record = _record()
    record["motion_arc_review"]["apex"]["evidence_frame"] = 100
    with pytest.raises(ExpressionTurnContractError, match="temporally ordered"):
        evaluate_expression_turn(record)

    record = _record()
    record["motion_arc_review"]["offset"]["evidence_frame"] = 103
    with pytest.raises(ExpressionTurnContractError, match="outside the interval"):
        evaluate_expression_turn(record)


def test_affect_review_can_be_absent_without_blocking_motion_training():
    record = _record()
    record["affect_review"] = None
    result = evaluate_expression_turn(record)
    assert result["decision"] == "retain_base_motion"
    assert result["affect_review_status"] == "not_reviewed"
    assert result["emotion_supervision_mask"] is False


def test_two_independent_observable_affect_reviews_enable_expressive_tier_only():
    record = _record()
    first = record.pop("affect_review")
    first.update(
        {"result": "observable", "predicted_class": "happy", "confidence": 0.91}
    )
    second = copy.deepcopy(first)
    second["review_id"] = "affect-review-2"
    second["reviewer_id"] = "affect-reviewer-2"
    second["confidence"] = 0.86
    record["affect_reviews"] = [first, second]

    result = evaluate_expression_turn(record)
    assert result["highest_qualification"] == "expressive_conditioning"
    assert result["base_motion_eligible"] is True
    assert result["semantic_conditioning_eligible"] is True
    assert result["expressive_conditioning_eligible"] is True
    assert result["affect_review_status"] == "observable_consensus"
    assert result["blind_affect_class"] == "happy"
    assert result["emotion_supervision_candidate"] is True
    assert result["emotion_supervision_mask"] is False


def test_conflicting_affect_reviews_need_an_independent_blind_adjudication():
    record = _record()
    first = record.pop("affect_review")
    first.update(
        {"result": "observable", "predicted_class": "happy", "confidence": 0.91}
    )
    second = copy.deepcopy(first)
    second.update(
        {
            "review_id": "affect-review-2",
            "reviewer_id": "affect-reviewer-2",
            "predicted_class": "neutral",
        }
    )
    record["affect_reviews"] = [first, second]

    unresolved = evaluate_expression_turn(record)
    assert unresolved["expressive_conditioning_eligible"] is False
    assert unresolved["affect_review_status"] == (
        "conflicting_or_insufficient_consensus"
    )
    assert unresolved["affect_candidate_for_separate_adjudication"] is True

    adjudication = _review_envelope(
        AFFECT_ADJUDICATION_PROTOCOL,
        "affect-adjudication",
        "independent-affect-adjudicator",
    )
    adjudication.update(
        {
            "input_review_ids": ["affect-review", "affect-review-2"],
            "result": "observable",
            "predicted_class": "happy",
            "confidence": 0.88,
        }
    )
    record["affect_adjudication"] = adjudication
    resolved = evaluate_expression_turn(record)
    assert resolved["expressive_conditioning_eligible"] is True
    assert resolved["affect_review_status"] == "observable_adjudicated"


def test_affect_consensus_requires_independent_reviewers_and_confidence():
    record = _record()
    first = record.pop("affect_review")
    first.update(
        {"result": "observable", "predicted_class": "happy", "confidence": 0.91}
    )
    duplicate = copy.deepcopy(first)
    duplicate["review_id"] = "affect-review-2"
    record["affect_reviews"] = [first, duplicate]
    with pytest.raises(ExpressionTurnContractError, match="reviewers must be independent"):
        evaluate_expression_turn(record)

    duplicate["reviewer_id"] = "affect-reviewer-2"
    duplicate["confidence"] = 0.69
    result = evaluate_expression_turn(record)
    assert result["expressive_conditioning_eligible"] is False
    assert result["affect_candidate_for_separate_adjudication"] is True


def test_missing_action_review_preserves_base_but_blocks_conditioning():
    record = _record()
    record["action_semantic_review"] = None
    result = evaluate_expression_turn(record)
    assert result["decision"] == "retain_base_motion"
    assert result["base_motion_eligible"] is True
    assert result["semantic_conditioning_eligible"] is False
    assert result["expressive_conditioning_eligible"] is False


def test_producer_candidate_contract_keeps_all_training_tiers_pending():
    result = validate_expression_turn_candidate(_candidate_record())
    assert result["boundary_candidate_valid"] is True
    assert result["accepted_for_training"] is False
    assert result["official_emotion_used"] is False
    assert all(
        tier["eligible"] is False for tier in result["qualifications"].values()
    )
    assert result["qualifications"]["base_motion"]["status"] == (
        "pending_18d_retarget_and_blind_arc_review"
    )


def test_producer_candidate_cannot_preverify_arc_or_enable_a_mask():
    candidate = _candidate_record()
    candidate["expression_turn"]["complete_motion_arc_verified"] = True
    with pytest.raises(ExpressionTurnContractError, match="may not pre-verify"):
        validate_expression_turn_candidate(candidate)

    candidate = _candidate_record()
    candidate["semantic_supervision_masks"]["prompt_text"] = True
    with pytest.raises(ExpressionTurnContractError, match="fully masked"):
        validate_expression_turn_candidate(candidate)


def test_producer_candidate_must_bind_selected_context_and_self_hash():
    candidate = _candidate_record()
    candidate["training_segment"]["start_frame"] = 99
    with pytest.raises(ExpressionTurnContractError, match="selected natural-boundary"):
        validate_expression_turn_candidate(candidate)

    candidate = _candidate_record()
    candidate["expression_turn_record_sha256"] = "0" * 64
    with pytest.raises(ExpressionTurnContractError, match="does not match"):
        validate_expression_turn_candidate(candidate)


@pytest.mark.parametrize("selection_kind", ["stress100", "representative100"])
def test_selected_candidate_returns_unambiguous_retarget_lineage(selection_kind):
    candidate = _selection_candidate_record(selection_kind)
    binding = {
        "retarget_input_manifest_sha256": "f" * 64,
        "expression_turn_contract_sha256": "c" * 64,
        "source_inventory_manifest_sha256": "d" * 64,
        "split_assignment_manifest_sha256": "e" * 64,
        "selection_kind": selection_kind,
        "require_selection_record": True,
    }
    result = validate_expression_turn_candidate(
        candidate, catalog_binding=binding
    )
    lineage = result["lineage"]
    assert lineage["inventory_record_sha256"] == candidate[
        "expression_turn_selection_record_sha256"
    ]
    assert lineage["upstream_inventory_record_sha256"] == candidate[
        "expression_turn_record_sha256"
    ]
    assert lineage["selected_record_sha256"] == hashlib.sha256(
        json.dumps(
            candidate,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert lineage["retarget_input_manifest_sha256"] == "f" * 64


def test_selected_candidate_rejects_hash_or_catalog_binding_mismatch():
    candidate = _selection_candidate_record()
    candidate["expression_turn_selection_rank"] = 2
    with pytest.raises(ExpressionTurnContractError, match="selection record SHA256"):
        validate_expression_turn_candidate(candidate)

    candidate = _selection_candidate_record()
    binding = {
        "retarget_input_manifest_sha256": "f" * 64,
        "expression_turn_contract_sha256": "0" * 64,
        "source_inventory_manifest_sha256": "d" * 64,
        "split_assignment_manifest_sha256": "e" * 64,
        "selection_kind": "stress100",
        "require_selection_record": True,
    }
    with pytest.raises(ExpressionTurnContractError, match="catalog binding"):
        validate_expression_turn_candidate(candidate, catalog_binding=binding)


def test_formal_binding_rejects_wrong_selection_kind_and_legacy_pilot():
    candidate = _selection_candidate_record("representative100")
    binding = {
        "retarget_input_manifest_sha256": "f" * 64,
        "expression_turn_contract_sha256": "c" * 64,
        "source_inventory_manifest_sha256": "d" * 64,
        "split_assignment_manifest_sha256": "e" * 64,
        "selection_kind": "stress100",
        "require_selection_record": True,
    }
    with pytest.raises(ExpressionTurnContractError, match="selection kind"):
        validate_expression_turn_candidate(candidate, catalog_binding=binding)

    legacy = _candidate_record()
    legacy.update(
        {
            "expression_turn_pilot_rank": 1,
            "expression_turn_pilot_selection_status": (
                "selected_stratified_pending_retarget_qc"
            ),
        }
    )
    legacy["expression_turn_pilot_record_sha256"] = hashlib.sha256(
        json.dumps(
            legacy,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ExpressionTurnContractError, match="legacy pilot"):
        validate_expression_turn_candidate(legacy, catalog_binding=binding)


def test_input_is_not_mutated():
    record = _record()
    original = copy.deepcopy(record)
    evaluate_expression_turn(record)
    assert record == original
