from __future__ import annotations

import pytest

from upper_body_skeleton.robot_observable_motion_realizations import (
    build_conversational_realization_annotation,
    load_motion_realization_ontology,
    validate_conversational_realization_annotation,
)


def _source() -> dict:
    return {
        "dataset": "BEAT2",
        "annotation_kind": "official_gesture_semantic_event",
        "interaction_scope": "human_co_speech_interaction",
        "status": "passed",
        "quality_gate": {"passed": True},
        "prompt": "wave hello from a filename",
    }


def _style() -> dict:
    return {
        "arm_amplitude": "moderate",
        "laterality": "right",
        "pace": "steady",
        "head_engagement": "engaged",
    }


def test_realization_ontology_is_independent_of_primary_intent() -> None:
    ontology = load_motion_realization_ontology()
    label = ontology["labels"][0]
    assert label["id"] == "conversational_gesturing"
    assert label["does_not_imply_primary_intent"] is True


def test_builds_trainable_ordinary_speaking_realization_without_intent() -> None:
    annotation = build_conversational_realization_annotation(_source(), _style())
    validate_conversational_realization_annotation(annotation)
    assert annotation["motion_realization_supervision_mask"] is True
    assert annotation["source_transcript_semantics_used"] is False
    assert annotation["does_not_imply_primary_intent"] is True
    assert "observable_intent_id" not in annotation
    assert "right arm" in annotation["motion_realization_prompt"]["en"]


def test_rejects_non_co_speech_source() -> None:
    source = _source()
    source["interaction_scope"] = "motion_only"
    with pytest.raises(ValueError, match="co-speech"):
        build_conversational_realization_annotation(source, _style())


def test_prompt_hash_cannot_be_silently_changed() -> None:
    annotation = build_conversational_realization_annotation(_source(), _style())
    annotation["motion_realization_prompt"]["en"] = "Wave hello."
    with pytest.raises(ValueError, match="prompt SHA256"):
        validate_conversational_realization_annotation(annotation)
