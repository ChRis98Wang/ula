from __future__ import annotations

from copy import deepcopy

import pytest

from upper_body_skeleton.beat2_observable_action_summary import (
    load_action_summary_ontology,
    summarize_observable_action,
    validate_action_summary,
)


def _style(**overrides):
    motion_style = {
        "arm_amplitude": "medium",
        "laterality": "both",
        "head_engagement": "natural",
        "torso_engagement": "natural",
    }
    motion_style.update(overrides)
    return {"motion_style": motion_style, "candidate_routes": []}


def test_ontology_expands_to_29_named_actions() -> None:
    ontology = load_action_summary_ontology()
    ids = [row["id"] for row in ontology["labels"]]
    assert len(ids) == 29
    assert len(ids) == len(set(ids))
    assert {"head_nod", "single_arm_wave", "deictic_one_arm", "iconic_two_arm"} <= set(ids)


def test_official_category_and_motion_form_both_affect_summary() -> None:
    record = _style(laterality="left")
    deictic = summarize_observable_action(record, official_category="deictic")
    iconic = summarize_observable_action(record, official_category="iconic")
    assert deictic["action_id"] == "deictic_one_arm"
    assert iconic["action_id"] == "iconic_one_arm"
    assert deictic["dialogue_used_to_assign_action"] is False
    assert deictic["pragmatic_intent_claimed"] is False


def test_tier_a_head_action_overrides_coarse_official_category() -> None:
    record = _style()
    record["candidate_routes"] = [
        {
            "candidate_intent_id": "agree_nod",
            "routing_evidence": ["tier_a=true", "head_periodic=true"],
        }
    ]
    summary = summarize_observable_action(record, official_category="metaphoric")
    assert summary["action_id"] == "head_nod"
    assert summary["confidence"] == "high"


def test_wave_is_physical_motion_not_a_greeting_claim() -> None:
    record = _style(laterality="right")
    record["candidate_routes"] = [
        {
            "candidate_intent_id": "wave_to_person",
            "routing_evidence": ["single_arm_periodic=true"],
        }
    ]
    summary = summarize_observable_action(record, official_category="deictic")
    assert summary["action_id"] == "single_arm_wave"
    assert summary["pragmatic_intent_claimed"] is False
    assert "greet" not in summary["prompt_en"].casefold()


def test_validator_rejects_invented_dialogue_evidence() -> None:
    summary = summarize_observable_action(
        _style(arm_amplitude="small"), official_category="iconic"
    )
    invalid = deepcopy(summary)
    invalid["dialogue_used_to_assign_action"] = True
    with pytest.raises(ValueError, match="dialogue_used_to_assign_action"):
        validate_action_summary(invalid)
