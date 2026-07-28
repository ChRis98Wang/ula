#!/usr/bin/env python3
"""Versioned, fail-closed semantics for observable 18D robot interactions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


DEFAULT_ONTOLOGY_PATH = (
    Path(__file__).resolve().parents[1]
    / "training"
    / "contracts"
    / "robot_observable_interaction_intents_v1.json"
)
TRAIN_READY_REVIEW_STATUSES = frozenset(
    {"independent_blind_consensus", "independent_blind_adjudication"}
)
UNSUPERVISED_REVIEW_STATUSES = frozenset(
    {"pending_review", "pending_adjudication", "rejected"}
)
EVIDENCE_MODES = frozenset(
    {
        "visual_primary",
        "visual_plus_dyadic_context",
        "visual_plus_target_context",
        "visual_plus_speech_context",
    }
)


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def ontology_sha256(path: str | Path = DEFAULT_ONTOLOGY_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_observable_intent_ontology(
    path: str | Path = DEFAULT_ONTOLOGY_PATH,
) -> dict[str, Any]:
    ontology = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_observable_intent_ontology(ontology)
    return ontology


def validate_observable_intent_ontology(ontology: Mapping[str, Any]) -> None:
    if ontology.get("schema_version") != "1.0.0":
        raise ValueError("observable intent ontology schema_version must be 1.0.0")
    _require_nonempty_string(ontology.get("ontology_id"), "ontology_id")
    slot_count = ontology.get("condition_slot_count")
    intents = ontology.get("intents")
    if isinstance(slot_count, bool) or not isinstance(slot_count, int) or slot_count < 1:
        raise ValueError("condition_slot_count must be a positive integer")
    if not isinstance(intents, list) or len(intents) != slot_count:
        raise ValueError("intents must exactly fill condition_slot_count")

    ids: list[str] = []
    slots: list[int] = []
    for index, intent in enumerate(intents):
        if not isinstance(intent, Mapping):
            raise ValueError(f"intents[{index}] must be an object")
        intent_id = _require_nonempty_string(intent.get("id"), f"intents[{index}].id")
        if intent_id in {"greeting", "farewell", "greeting_wave", "farewell_wave"}:
            raise ValueError(
                f"{intent_id} is contextual, not a distinct observable motion intent"
            )
        slot = intent.get("slot")
        if isinstance(slot, bool) or not isinstance(slot, int):
            raise ValueError(f"intents[{index}].slot must be an integer")
        evidence_mode = intent.get("evidence_mode")
        if evidence_mode not in EVIDENCE_MODES:
            raise ValueError(f"unknown evidence_mode for {intent_id}: {evidence_mode!r}")
        for field in (
            "name_en",
            "name_zh",
            "family",
            "canonical_prompt_en",
            "canonical_prompt_zh",
            "visual_signature",
        ):
            _require_nonempty_string(intent.get(field), f"{intent_id}.{field}")
        hard_negatives = intent.get("hard_negatives")
        if not isinstance(hard_negatives, list) or not hard_negatives:
            raise ValueError(f"{intent_id}.hard_negatives must be non-empty")
        if intent_id in hard_negatives:
            raise ValueError(f"{intent_id} cannot be its own hard negative")
        ids.append(intent_id)
        slots.append(slot)

    if len(set(ids)) != len(ids):
        raise ValueError("observable intent ids must be unique")
    if slots != list(range(slot_count)):
        raise ValueError("observable intent slots must be contiguous and ordered")
    known_ids = set(ids)
    for intent in intents:
        unknown = set(intent["hard_negatives"]) - known_ids
        if unknown:
            raise ValueError(
                f"{intent['id']} has unknown hard negatives: {sorted(unknown)}"
            )

    review_contract = ontology.get("review_contract")
    if not isinstance(review_contract, Mapping):
        raise ValueError("review_contract must be an object")
    if review_contract.get("filename_or_transcript_auto_admission") is not False:
        raise ValueError("filename/transcript auto-admission must remain disabled")
    if review_contract.get("source_label_visible") is not False:
        raise ValueError("source labels must remain hidden during primary intent review")
    if review_contract.get("audio_visible_for_primary_intent") is not False:
        raise ValueError("audio must remain hidden during primary intent review")
    if review_contract.get("hard_negative_check_required") is not True:
        raise ValueError("hard-negative review must be required")

    primary_ids = set(ids)
    pragmatic_ids = {
        _require_nonempty_string(item.get("id"), "pragmatic_roles[].id")
        for item in ontology.get("pragmatic_roles", [])
        if isinstance(item, Mapping)
    }
    if primary_ids & pragmatic_ids:
        raise ValueError("pragmatic roles may not duplicate observable intent ids")
    equivalence_rules = ontology.get("visual_equivalence_rules")
    if not isinstance(equivalence_rules, list) or not equivalence_rules:
        raise ValueError("visual_equivalence_rules must be non-empty")
    for rule in equivalence_rules:
        if rule.get("observable_intent_id") not in primary_ids:
            raise ValueError("visual equivalence rule references an unknown intent")
        unknown_roles = set(rule.get("context_roles") or []) - pragmatic_ids
        if unknown_roles:
            raise ValueError(
                f"visual equivalence rule references unknown roles: {sorted(unknown_roles)}"
            )


def observable_intent_ids(
    ontology: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    ontology = ontology or load_observable_intent_ontology()
    return tuple(intent["id"] for intent in ontology["intents"])


def intent_definition(
    intent_id: str, ontology: Mapping[str, Any] | None = None
) -> Mapping[str, Any]:
    ontology = ontology or load_observable_intent_ontology()
    for intent in ontology["intents"]:
        if intent["id"] == intent_id:
            return intent
    raise ValueError(f"unknown observable_intent_id: {intent_id!r}")


def build_observable_intent_one_hot(
    intent_id: str, ontology: Mapping[str, Any] | None = None
) -> np.ndarray:
    ontology = ontology or load_observable_intent_ontology()
    intent = intent_definition(intent_id, ontology)
    result = np.zeros(int(ontology["condition_slot_count"]), dtype=np.float32)
    result[int(intent["slot"])] = 1.0
    return result


def validate_observable_intent_annotation(
    record: Mapping[str, Any],
    ontology: Mapping[str, Any] | None = None,
    *,
    expected_ontology_sha256: str | None = None,
) -> None:
    """Validate an admitted or masked v9 intent label without text inference."""

    ontology = ontology or load_observable_intent_ontology()
    known_ids = set(observable_intent_ids(ontology))
    ontology_id = ontology["ontology_id"]
    if record.get("intent_ontology_id") != ontology_id:
        raise ValueError("intent_ontology_id does not match the loaded ontology")
    if expected_ontology_sha256 is not None and record.get(
        "intent_ontology_sha256"
    ) != expected_ontology_sha256:
        raise ValueError("intent_ontology_sha256 does not match the ontology artifact")

    intent_id = record.get("observable_intent_id")
    review_status = record.get("intent_review_status")
    supervision_mask = record.get("intent_supervision_mask")
    conditioning_mask = record.get("intent_conditioning_mask")
    if not isinstance(supervision_mask, bool) or not isinstance(conditioning_mask, bool):
        raise ValueError("intent masks must be explicit booleans")
    if supervision_mask != conditioning_mask:
        raise ValueError("intent supervision and conditioning masks must agree")

    if supervision_mask:
        if intent_id not in known_ids:
            raise ValueError("supervised intent requires a known observable_intent_id")
        if review_status not in TRAIN_READY_REVIEW_STATUSES:
            raise ValueError("supervised intent requires independent blind review")
        evidence = record.get("intent_review_evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError("supervised intent requires intent_review_evidence")
        reviewers = evidence.get("reviewer_ids")
        minimum_reviewers = int(
            ontology["review_contract"]["minimum_independent_reviewers"]
        )
        if (
            not isinstance(reviewers, list)
            or len(reviewers) < minimum_reviewers
            or len(set(reviewers)) != len(reviewers)
            or any(not isinstance(value, str) or not value.strip() for value in reviewers)
        ):
            raise ValueError("intent evidence lacks independent reviewer ids")
        if evidence.get("label_metadata_exposed") is not False:
            raise ValueError("source label metadata may not be exposed in blind review")
        if evidence.get("audio_available") is not False:
            raise ValueError("audio may not be used for the primary observable intent")
        if evidence.get("hard_negative_checked") is not True:
            raise ValueError("hard-negative comparison is required")
        if not _is_sha256(evidence.get("video_sha256")):
            raise ValueError("intent evidence requires a valid video_sha256")
        confidence = evidence.get("minimum_confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or float(confidence)
            < float(ontology["review_contract"]["minimum_confidence"])
        ):
            raise ValueError("intent evidence confidence is below the contract minimum")
        prompt = _require_nonempty_string(record.get("intent_prompt"), "intent_prompt")
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if record.get("intent_prompt_sha256") != prompt_sha:
            raise ValueError("intent_prompt_sha256 is invalid")
    else:
        if intent_id is not None:
            raise ValueError("masked intent must not carry observable_intent_id")
        if review_status not in UNSUPERVISED_REVIEW_STATUSES:
            raise ValueError("masked intent has an unknown review status")

    pragmatic_role = record.get("pragmatic_role")
    pragmatic_mask = record.get("pragmatic_role_supervision_mask")
    if not isinstance(pragmatic_mask, bool):
        raise ValueError("pragmatic_role_supervision_mask must be boolean")
    if pragmatic_mask:
        role_ids = {item["id"] for item in ontology["pragmatic_roles"]}
        if pragmatic_role not in role_ids:
            raise ValueError("unknown supervised pragmatic_role")
        allowed = set(intent_definition(str(intent_id), ontology).get("allowed_pragmatic_roles", []))
        if pragmatic_role not in allowed:
            raise ValueError("pragmatic_role is not allowed for this observable intent")
        context_evidence = record.get("pragmatic_role_context_evidence")
        if not isinstance(context_evidence, Mapping) or not context_evidence.get(
            "human_verified"
        ):
            raise ValueError("pragmatic role requires independently verified context evidence")
    elif pragmatic_role is not None:
        raise ValueError("masked pragmatic role must be null")


__all__ = [
    "DEFAULT_ONTOLOGY_PATH",
    "EVIDENCE_MODES",
    "TRAIN_READY_REVIEW_STATUSES",
    "UNSUPERVISED_REVIEW_STATUSES",
    "build_observable_intent_one_hot",
    "intent_definition",
    "load_observable_intent_ontology",
    "observable_intent_ids",
    "ontology_sha256",
    "validate_observable_intent_annotation",
    "validate_observable_intent_ontology",
]
