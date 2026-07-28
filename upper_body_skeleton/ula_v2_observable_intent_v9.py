#!/usr/bin/env python3
"""Network condition adapter for the versioned observable-intent v9 contract."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from upper_body_skeleton.kimodo_semantics import (
    KIMODO_BEHAVIOR_FAMILIES,
    KIMODO_BEHAVIOR_IDS,
    KIMODO_EMOTION_IDS,
)
from upper_body_skeleton.robot_observable_intents import (
    build_observable_intent_one_hot,
    load_observable_intent_ontology,
    ontology_sha256,
    validate_observable_intent_annotation,
)


NETWORK_CONDITION_CONTRACT = "ula_v2_18d_observable_intent_v9"
LEGACY_INTENT_DIM = 6
LEGACY_AFFECT_DIM = 8
LEGACY_STYLE_DIM = 3
LEGACY_GESTURE_DIM = 6
LEGACY_SCALAR_DIM = 5
LEGACY_TEXT_EMBED_DIM = 64
LEGACY_CONDITION_DIM = (
    LEGACY_INTENT_DIM
    + LEGACY_AFFECT_DIM
    + LEGACY_STYLE_DIM
    + LEGACY_GESTURE_DIM
    + LEGACY_SCALAR_DIM
    + LEGACY_TEXT_EMBED_DIM
)
KIMODO_STYLE_CONTROL_DIM = 3
MOTION_TEXT_LATENT_DIM = 128
KIMODO_V2_CONDITION_DIM = (
    LEGACY_CONDITION_DIM
    + len(KIMODO_BEHAVIOR_IDS)
    + len(KIMODO_EMOTION_IDS)
    + len(KIMODO_BEHAVIOR_FAMILIES)
    + KIMODO_STYLE_CONTROL_DIM
    + MOTION_TEXT_LATENT_DIM
)
OBSERVABLE_INTENT_SLICE_V9 = slice(
    LEGACY_CONDITION_DIM, LEGACY_CONDITION_DIM + len(KIMODO_BEHAVIOR_IDS)
)
KIMODO_EMOTION_SLICE = slice(
    OBSERVABLE_INTENT_SLICE_V9.stop,
    OBSERVABLE_INTENT_SLICE_V9.stop + len(KIMODO_EMOTION_IDS),
)
LEGACY_KIMODO_FAMILY_SLICE = slice(
    KIMODO_EMOTION_SLICE.stop,
    KIMODO_EMOTION_SLICE.stop + len(KIMODO_BEHAVIOR_FAMILIES),
)
LEGACY_INTENT_SLICE = slice(0, LEGACY_INTENT_DIM)
LEGACY_GESTURE_SLICE = slice(
    LEGACY_INTENT_DIM + LEGACY_AFFECT_DIM + LEGACY_STYLE_DIM,
    LEGACY_INTENT_DIM + LEGACY_AFFECT_DIM + LEGACY_STYLE_DIM + LEGACY_GESTURE_DIM,
)


def _validate_overlay_contract(overlay: Mapping[str, Any]) -> None:
    if overlay.get("network_condition_contract") != NETWORK_CONDITION_CONTRACT:
        raise ValueError("v9 overlay is missing its network_condition_contract")
    if overlay.get("primary_semantic_channel") != "observable_intent_one_hot":
        raise ValueError("v9 observable intent must be the primary semantic channel")
    if overlay.get("text_channel_role") != "auxiliary_semantic_prompt":
        raise ValueError("v9 text must be declared as an auxiliary semantic channel")
    if overlay.get("legacy_behavior_conditioning_enabled") is not False:
        raise ValueError("legacy Kimodo behavior conditioning must be disabled in v9")
    if overlay.get("legacy_intent_conditioning_enabled") is not False:
        raise ValueError("legacy coarse intent conditioning must be disabled in v9")


def apply_observable_intent_overlay_v9(
    base_condition: np.ndarray,
    overlay: Mapping[str, Any],
) -> np.ndarray:
    """Install a reviewed v9 intent without mutating historical v8 conditions."""

    _validate_overlay_contract(overlay)
    ontology = load_observable_intent_ontology()
    digest = ontology_sha256()
    validate_observable_intent_annotation(
        overlay, ontology, expected_ontology_sha256=digest
    )

    condition = np.asarray(base_condition, dtype=np.float32)
    if condition.ndim < 1 or condition.shape[-1] != KIMODO_V2_CONDITION_DIM:
        raise ValueError(
            f"v9 base condition must end in {KIMODO_V2_CONDITION_DIM} dimensions"
        )
    result = condition.copy()
    result[..., LEGACY_INTENT_SLICE] = 0.0
    result[..., LEGACY_GESTURE_SLICE] = 0.0
    result[..., OBSERVABLE_INTENT_SLICE_V9] = 0.0
    result[..., LEGACY_KIMODO_FAMILY_SLICE] = 0.0
    if overlay["intent_conditioning_mask"] is True:
        expected = build_observable_intent_one_hot(
            overlay["observable_intent_id"], ontology
        )
        stored = overlay.get("intent_one_hot")
        if stored is not None and not np.array_equal(
            np.asarray(stored, dtype=np.float32), expected
        ):
            raise ValueError("stored intent_one_hot does not match ontology slot")
        result[..., OBSERVABLE_INTENT_SLICE_V9] = expected
    return result


def validate_v9_checkpoint_metadata(metadata: Mapping[str, Any]) -> None:
    if metadata.get("network_condition_contract") != NETWORK_CONDITION_CONTRACT:
        raise ValueError("checkpoint does not declare the v9 condition contract")
    if metadata.get("intent_ontology_id") != load_observable_intent_ontology()[
        "ontology_id"
    ]:
        raise ValueError("checkpoint intent ontology id is incompatible")
    if metadata.get("intent_ontology_sha256") != ontology_sha256():
        raise ValueError("checkpoint intent ontology artifact is incompatible")
    if metadata.get("condition_dim") != KIMODO_V2_CONDITION_DIM:
        raise ValueError("checkpoint condition_dim is incompatible")


__all__ = [
    "KIMODO_V2_CONDITION_DIM",
    "LEGACY_KIMODO_FAMILY_SLICE",
    "NETWORK_CONDITION_CONTRACT",
    "OBSERVABLE_INTENT_SLICE_V9",
    "apply_observable_intent_overlay_v9",
    "validate_v9_checkpoint_metadata",
]
