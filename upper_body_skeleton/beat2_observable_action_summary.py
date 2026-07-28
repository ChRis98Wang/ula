"""Concise, trajectory-grounded action summaries for BEAT2 18D motion."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_ONTOLOGY_PATH = (
    Path(__file__).resolve().parents[1]
    / "training/contracts/beat2_observable_action_summaries_v1.json"
)
ONTOLOGY_ID = "beat2_observable_action_summaries_v1"
SUMMARY_SOURCE = "verified_18d_trajectory_observable_action_summary_v2"


def ontology_sha256(path: str | Path = DEFAULT_ONTOLOGY_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_action_summary_ontology(
    path: str | Path = DEFAULT_ONTOLOGY_PATH,
) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != "1.1.0" or value.get("ontology_id") != ONTOLOGY_ID:
        raise ValueError("unexpected BEAT2 action-summary ontology")
    labels = value.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError("BEAT2 action-summary ontology has no labels")
    ids = [str(row.get("id") or "") for row in labels]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("BEAT2 action-summary label IDs must be unique and non-empty")
    for row in labels:
        if not str(row.get("prompt_en") or "").strip() or not str(
            row.get("prompt_zh") or ""
        ).strip():
            raise ValueError(f"{row.get('id')}: bilingual prompts are required")
    policy = value.get("label_policy") or {}
    required_false = (
        "dialogue_used_to_assign_action",
        "emotion_used_to_assign_action",
        "pragmatic_intent_claimed",
    )
    if any(policy.get(field) is not False for field in required_false):
        raise ValueError("action summaries may not claim dialogue, emotion, or intent evidence")
    return value


def _route_by_id(style_record: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    routes = style_record.get("candidate_routes") or []
    if not isinstance(routes, list):
        raise ValueError("candidate_routes must be a list")
    return {
        str(route["candidate_intent_id"]): route
        for route in routes
        if isinstance(route, Mapping) and route.get("candidate_intent_id")
    }


def _tier_a(route: Mapping[str, Any] | None) -> bool:
    return bool(route) and "tier_a=true" in set(route.get("routing_evidence") or [])


def summarize_observable_action(
    style_record: Mapping[str, Any],
    *,
    official_category: str,
    ontology_path: str | Path = DEFAULT_ONTOLOGY_PATH,
) -> dict[str, Any]:
    """Map verified trajectory evidence to one concise physical action label."""

    ontology = load_action_summary_ontology(ontology_path)
    labels = {row["id"]: row for row in ontology["labels"]}
    routes = _route_by_id(style_record)
    style = style_record.get("motion_style")
    if not isinstance(style, Mapping):
        raise ValueError("motion_style is required for an action summary")
    official_category = str(official_category).strip()
    if official_category not in {"deictic", "iconic", "metaphoric"}:
        raise ValueError(f"unsupported official BEAT2 category: {official_category!r}")

    evidence = []
    confidence = "medium"
    if _tier_a(routes.get("agree_nod")):
        action_id = "head_nod"
        evidence = list(routes["agree_nod"]["routing_evidence"])
        confidence = "high"
    elif _tier_a(routes.get("disagree_head_shake")):
        action_id = "head_shake"
        evidence = list(routes["disagree_head_shake"]["routing_evidence"])
        confidence = "high"
    elif "search_scan" in routes:
        action_id = "head_scan"
        evidence = list(routes["search_scan"].get("routing_evidence") or [])
    elif "bow" in routes:
        action_id = "upper_body_bow"
        evidence = list(routes["bow"].get("routing_evidence") or [])
    elif "withdraw_turn_away" in routes:
        action_id = "turn_away"
        evidence = list(routes["withdraw_turn_away"].get("routing_evidence") or [])
    elif "wave_to_person" in routes:
        action_id = "single_arm_wave"
        evidence = list(routes["wave_to_person"].get("routing_evidence") or [])
    elif "raise_hand_get_attention" in routes:
        action_id = "raise_one_arm"
        evidence = list(routes["raise_hand_get_attention"].get("routing_evidence") or [])
    elif "applaud" in routes:
        action_id = "bilateral_repeated_motion"
        evidence = list(routes["applaud"].get("routing_evidence") or [])
    else:
        amplitude = str(style.get("arm_amplitude") or "")
        laterality = str(style.get("laterality") or "")
        head = str(style.get("head_engagement") or "")
        torso = str(style.get("torso_engagement") or "")
        prefix = official_category
        if amplitude in {"very_small", "small"} and head == "expressive":
            action_id = f"{prefix}_head_led"
            evidence = [f"head_engagement={head}", f"arm_amplitude={amplitude}"]
        elif amplitude in {"very_small", "small"} and torso == "expressive":
            action_id = f"{prefix}_torso_led"
            evidence = [f"torso_engagement={torso}", f"arm_amplitude={amplitude}"]
        elif amplitude in {"very_small", "small"}:
            action_id = f"{prefix}_subtle"
            evidence = [f"arm_amplitude={amplitude}"]
        elif laterality in {"left", "right"}:
            action_id = f"{prefix}_one_arm"
            evidence = [f"laterality={laterality}"]
        elif amplitude == "large" or "celebrate" in routes:
            action_id = f"{prefix}_broad_two_arm"
            evidence = ["laterality=both", "arm_amplitude=large"]
        elif "shrug_uncertain" in routes:
            action_id = f"{prefix}_compact_two_arm"
            evidence = list(routes["shrug_uncertain"].get("routing_evidence") or [])
        else:
            action_id = f"{prefix}_two_arm"
            evidence = ["laterality=both"]

    label = labels[action_id]
    return {
        "ontology_id": ontology["ontology_id"],
        "ontology_sha256": ontology_sha256(ontology_path),
        "action_id": action_id,
        "prompt_en": label["prompt_en"],
        "prompt_zh": label["prompt_zh"],
        "confidence": confidence,
        "evidence_mode": "verified_18d_trajectory_rule",
        "trajectory_evidence": [f"official_category={official_category}", *evidence],
        "official_category": official_category,
        "official_category_verified": True,
        "dialogue_used_to_assign_action": False,
        "emotion_used_to_assign_action": False,
        "pragmatic_intent_claimed": False,
        "interaction_intent_supervision_mask": False,
        "style_values_stored_separately": True,
    }


def validate_action_summary(
    summary: Mapping[str, Any],
    *,
    ontology_path: str | Path = DEFAULT_ONTOLOGY_PATH,
) -> None:
    ontology = load_action_summary_ontology(ontology_path)
    labels = {row["id"]: row for row in ontology["labels"]}
    action_id = summary.get("action_id")
    if action_id not in labels:
        raise ValueError(f"unknown action summary: {action_id!r}")
    if (
        summary.get("ontology_id") != ontology["ontology_id"]
        or summary.get("ontology_sha256") != ontology_sha256(ontology_path)
        or summary.get("prompt_en") != labels[action_id]["prompt_en"]
        or summary.get("prompt_zh") != labels[action_id]["prompt_zh"]
    ):
        raise ValueError(f"{action_id}: action-summary ontology binding changed")
    if summary.get("confidence") not in {"high", "medium"}:
        raise ValueError(f"{action_id}: invalid action-summary confidence")
    if (
        summary.get("official_category")
        not in set(ontology["label_policy"]["official_categories"])
        or summary.get("official_category_verified") is not True
    ):
        raise ValueError(f"{action_id}: verified official category is required")
    for field in (
        "dialogue_used_to_assign_action",
        "emotion_used_to_assign_action",
        "pragmatic_intent_claimed",
        "interaction_intent_supervision_mask",
    ):
        if summary.get(field) is not False:
            raise ValueError(f"{action_id}: {field} must remain false")
    if summary.get("style_values_stored_separately") is not True:
        raise ValueError(f"{action_id}: style controls must remain separate")


__all__ = [
    "DEFAULT_ONTOLOGY_PATH",
    "ONTOLOGY_ID",
    "SUMMARY_SOURCE",
    "load_action_summary_ontology",
    "ontology_sha256",
    "summarize_observable_action",
    "validate_action_summary",
]
