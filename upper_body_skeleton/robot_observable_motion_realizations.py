#!/usr/bin/env python3
"""Fail-closed secondary labels for ordinary co-speech motion realization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_REALIZATION_ONTOLOGY_PATH = (
    Path(__file__).resolve().parents[1]
    / "training/contracts/robot_observable_motion_realizations_v1.json"
)
REALIZATION_ID = "conversational_gesturing"
PROVENANCE = "official_beat2_co_speech_event_plus_verified_18d_style_v1"


def realization_ontology_sha256(
    path: str | Path = DEFAULT_REALIZATION_ONTOLOGY_PATH,
) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_motion_realization_ontology(
    path: str | Path = DEFAULT_REALIZATION_ONTOLOGY_PATH,
) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_motion_realization_ontology(value)
    return value


def validate_motion_realization_ontology(ontology: Mapping[str, Any]) -> None:
    if ontology.get("schema_version") != "1.0.0":
        raise ValueError("motion-realization ontology schema_version must be 1.0.0")
    if ontology.get("ontology_id") != "robot_observable_motion_realizations_v1":
        raise ValueError("unexpected motion-realization ontology_id")
    labels = ontology.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError("motion-realization labels must be non-empty")
    ids: set[str] = set()
    for label in labels:
        if not isinstance(label, Mapping):
            raise ValueError("motion-realization labels must be objects")
        label_id = label.get("id")
        if not isinstance(label_id, str) or not label_id:
            raise ValueError("motion-realization label id must be non-empty")
        if label_id in ids:
            raise ValueError("motion-realization label ids must be unique")
        ids.add(label_id)
        if label.get("does_not_imply_primary_intent") is not True:
            raise ValueError(f"{label_id} must not imply a primary intent")
        for field in ("name_en", "name_zh", "canonical_prompt_en", "canonical_prompt_zh"):
            if not isinstance(label.get(field), str) or not label[field].strip():
                raise ValueError(f"{label_id}.{field} must be non-empty")
    admission = ontology.get("admission_contract")
    if not isinstance(admission, Mapping):
        raise ValueError("motion-realization admission_contract must be an object")
    for field in ("source_transcript_semantics_used", "source_filename_semantics_used"):
        if admission.get(field) is not False:
            raise ValueError(f"{field} must remain false")
    if admission.get("primary_intent_supervision_independent") is not True:
        raise ValueError("primary intent supervision must remain independent")


def _style_phrase(style: Mapping[str, Any]) -> tuple[str, str]:
    amplitude_en = {
        "very_small": "very small",
        "small": "small",
        "moderate": "moderate",
        "large": "broad",
    }[str(style["arm_amplitude"])]
    amplitude_zh = {
        "very_small": "很小幅度",
        "small": "小幅度",
        "moderate": "中等幅度",
        "large": "较大幅度",
    }[str(style["arm_amplitude"])]
    pace_en = {
        "minimal": "minimal",
        "slow": "unhurried",
        "steady": "steady",
        "quick": "quick",
    }[str(style["pace"])]
    pace_zh = {
        "minimal": "极少",
        "slow": "舒缓",
        "steady": "平稳",
        "quick": "较快",
    }[str(style["pace"])]
    side_en = {
        "none": "with little arm movement",
        "left": "led by the left arm",
        "right": "led by the right arm",
        "both": "using both arms",
    }[str(style["laterality"])]
    side_zh = {
        "none": "手臂动作很少",
        "left": "以左臂为主",
        "right": "以右臂为主",
        "both": "使用双臂",
    }[str(style["laterality"])]
    head_en = {
        "quiet": "with a quiet head",
        "engaged": "with engaged head movement",
        "expressive": "with expressive head movement",
    }[str(style["head_engagement"])]
    head_zh = {
        "quiet": "头部保持安静",
        "engaged": "配合自然头部动作",
        "expressive": "配合明显头部动作",
    }[str(style["head_engagement"])]
    return (
        f"{amplitude_en}, {pace_en}-paced gestures {side_en}, {head_en}",
        f"{side_zh}做{amplitude_zh}、{pace_zh}节奏的手势，{head_zh}",
    )


def build_conversational_realization_annotation(
    source: Mapping[str, Any],
    style: Mapping[str, Any],
    *,
    ontology_path: str | Path = DEFAULT_REALIZATION_ONTOLOGY_PATH,
) -> dict[str, Any]:
    ontology = load_motion_realization_ontology(ontology_path)
    admission = ontology["admission_contract"]
    if source.get("dataset") != admission["supported_dataset"]:
        raise ValueError("conversational realization requires the supported dataset")
    if source.get("annotation_kind") != admission["required_annotation_kind"]:
        raise ValueError("conversational realization requires an official gesture event")
    if source.get("interaction_scope") != admission["required_interaction_scope"]:
        raise ValueError("conversational realization requires co-speech interaction scope")
    if source.get("status") != "passed" or source.get("quality_gate", {}).get("passed") is not True:
        raise ValueError("conversational realization requires passed robot physical QC")
    english_style, chinese_style = _style_phrase(style)
    prompt_en = f"Speak naturally with {english_style}."
    prompt_zh = f"自然说话，同时{chinese_style}。"
    return {
        "motion_realization_ontology_id": ontology["ontology_id"],
        "motion_realization_ontology_sha256": realization_ontology_sha256(ontology_path),
        "motion_realization_id": REALIZATION_ID,
        "motion_realization_supervision_mask": True,
        "motion_realization_conditioning_mask": True,
        "motion_realization_review_status": "verified_official_co_speech_event_and_robot_trajectory",
        "motion_realization_prompt": {"en": prompt_en, "zh": prompt_zh},
        "motion_realization_prompt_sha256": {
            "en": hashlib.sha256(prompt_en.encode("utf-8")).hexdigest(),
            "zh": hashlib.sha256(prompt_zh.encode("utf-8")).hexdigest(),
        },
        "motion_realization_prompt_provenance": PROVENANCE,
        "source_transcript_semantics_used": False,
        "source_filename_semantics_used": False,
        "audio_used": False,
        "does_not_imply_primary_intent": True,
        "does_not_imply_emotion": True,
    }


def validate_conversational_realization_annotation(
    annotation: Mapping[str, Any],
    *,
    ontology_path: str | Path = DEFAULT_REALIZATION_ONTOLOGY_PATH,
) -> None:
    ontology = load_motion_realization_ontology(ontology_path)
    if annotation.get("motion_realization_ontology_id") != ontology["ontology_id"]:
        raise ValueError("motion-realization ontology id mismatch")
    if annotation.get("motion_realization_ontology_sha256") != realization_ontology_sha256(ontology_path):
        raise ValueError("motion-realization ontology SHA256 mismatch")
    if annotation.get("motion_realization_id") != REALIZATION_ID:
        raise ValueError("unknown conversational motion-realization id")
    for field in ("motion_realization_supervision_mask", "motion_realization_conditioning_mask"):
        if annotation.get(field) is not True:
            raise ValueError(f"{field} must be true")
    if annotation.get("source_transcript_semantics_used") is not False:
        raise ValueError("transcript semantics may not establish realization labels")
    if annotation.get("source_filename_semantics_used") is not False:
        raise ValueError("filename semantics may not establish realization labels")
    if annotation.get("does_not_imply_primary_intent") is not True:
        raise ValueError("motion realization must remain independent of primary intent")
    prompts = annotation.get("motion_realization_prompt")
    hashes = annotation.get("motion_realization_prompt_sha256")
    if not isinstance(prompts, Mapping) or not isinstance(hashes, Mapping):
        raise ValueError("motion-realization bilingual prompt evidence is missing")
    for language in ("en", "zh"):
        prompt = prompts.get(language)
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"motion-realization {language} prompt is missing")
        if hashes.get(language) != hashlib.sha256(prompt.encode("utf-8")).hexdigest():
            raise ValueError(f"motion-realization {language} prompt SHA256 mismatch")


__all__ = [
    "DEFAULT_REALIZATION_ONTOLOGY_PATH",
    "PROVENANCE",
    "REALIZATION_ID",
    "build_conversational_realization_annotation",
    "load_motion_realization_ontology",
    "realization_ontology_sha256",
    "validate_conversational_realization_annotation",
    "validate_motion_realization_ontology",
]
