#!/usr/bin/env python3
"""Build deterministic, robot-observable semantics for HAA500 expression clips."""

import argparse
import csv
import hashlib
import io
import json
import os
import re
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path


DEFAULT_ROOT = Path("/home/gez/nas/cloud/gez/human_motion")
DATASET_REVISION = "c74fa62247289ed31e407b6133d954d3c171db43"
BUILDER_VERSION = "1.0.0"
OUTPUT_STEM = "expression_semantics_haa500_v1"
ACTION_PATTERN = re.compile(r"(?:_\d+)?_clip\d+$")

TORSO = ["joint_pelvisYaw", "joint_pelvisPitch", "joint_pelvisRoll"]
LEFT_ARM_CORE = [
    "joint_lShoulderPitch",
    "joint_lShoulderRoll",
    "joint_lShoulderYaw",
    "joint_lElbow",
]
RIGHT_ARM_CORE = [
    "joint_rShoulderPitch",
    "joint_rShoulderRoll",
    "joint_rShoulderYaw",
    "joint_rElbow",
]
LEFT_WRIST = ["joint_lWristRoll", "joint_lWristPitch"]
RIGHT_WRIST = ["joint_rWristRoll", "joint_rWristPitch"]
LEFT_ARM = LEFT_ARM_CORE + LEFT_WRIST
RIGHT_ARM = RIGHT_ARM_CORE + RIGHT_WRIST
UNMAPPED_CHANNELS = ["head", "face", "fingers"]


def dofs(
    required_groups,
    required_joint_names=(),
    alternative_joint_name_sets=(),
    optional_joint_names=(),
    unavailable_channels=UNMAPPED_CHANNELS,
):
    alternatives = [list(group) for group in alternative_joint_name_sets]
    required = list(required_joint_names)
    minimum = len(required)
    if alternatives:
        minimum += min(len(group) for group in alternatives)
    return {
        "joint_space": "ula_v2_15d",
        "required_groups": list(required_groups),
        "required_joint_names": required,
        "alternative_joint_name_sets": alternatives,
        "optional_joint_names": list(optional_joint_names),
        "minimum_observable_dof_count": minimum,
        "unavailable_channels": list(unavailable_channels),
    }


ACTION_SPECS = {
    "applauding": {
        "canonical_prompt": {
            "en": "Clap both hands together repeatedly in front of the torso.",
            "zh": "在躯干前方反复将双手合拢鼓掌。",
        },
        "observable_attributes": {
            "primary_effectors": ["left_arm", "right_arm"],
            "temporal_pattern": ["repeated", "rhythmic", "bilaterally_coordinated"],
            "spatial_pattern": ["both_arms_converge_and_separate_in_front_of_torso"],
            "unavailable_cues": ["clap_sound", "palm_contact", "finger_shape"],
        },
        "required_dofs": dofs(
            ["both_arms"],
            LEFT_ARM_CORE + RIGHT_ARM_CORE,
            optional_joint_names=LEFT_WRIST + RIGHT_WRIST + TORSO,
        ),
        "context_dependency": {
            "level": "none",
            "type": "none",
            "details": "The repeated bilateral arm motion is readable without a scene object.",
            "standalone_readability": "high",
        },
        "text_pattern": r"\b(?:applaud\w*|clap\w*)\b",
    },
    "arm_wave": {
        "canonical_prompt": {
            "en": "Extend the arms and perform a fluid side-to-side arm-wave motion.",
            "zh": "伸展手臂，做流畅的左右手臂波浪动作。",
        },
        "observable_attributes": {
            "primary_effectors": ["left_arm", "right_arm", "wrists"],
            "temporal_pattern": ["fluid", "sequential", "wave_propagation"],
            "spatial_pattern": ["arms_extended_laterally", "alternating_joint_elevation"],
            "unavailable_cues": ["finger_shape", "greeting_intent"],
        },
        "required_dofs": dofs(
            ["both_arms", "both_wrists"], LEFT_ARM + RIGHT_ARM, optional_joint_names=TORSO
        ),
        "context_dependency": {
            "level": "none",
            "type": "none",
            "details": "This label denotes a dance-like arm wave, not a greeting wave.",
            "standalone_readability": "medium",
        },
        "text_pattern": r"\b(?:arm[- ]?wav\w*|wav(?:e|es|ed|ing)|wave[- ]?like|wavelike)\b",
    },
    "blowing_kisses": {
        "canonical_prompt": {
            "en": "Move one or both hands from near the mouth outward in a repeated sending gesture.",
            "zh": "将一只或双手从嘴边反复向外送出。",
        },
        "observable_attributes": {
            "primary_effectors": ["one_or_both_arms", "wrists"],
            "temporal_pattern": ["hand_to_mouth_then_outward", "may_repeat"],
            "spatial_pattern": ["near_face_to_forward_space"],
            "unavailable_cues": ["lip_pucker", "facial_expression", "finger_shape"],
        },
        "required_dofs": dofs(
            ["at_least_one_arm"],
            alternative_joint_name_sets=[LEFT_ARM, RIGHT_ARM],
            optional_joint_names=TORSO,
        ),
        "context_dependency": {
            "level": "required",
            "type": "face_and_hand_detail",
            "details": "The 15DoF robot omits the lips, face, and fingers that disambiguate a kiss.",
            "standalone_readability": "low",
        },
        "text_pattern": r"\b(?:kiss\w*|blow\w*\s+(?:a\s+)?kiss\w*)\b",
    },
    "bowing_fullbody": {
        "canonical_prompt": {
            "en": "From upright, bend the torso forward deeply while lowering the arms, then return upright.",
            "zh": "从直立姿态开始，躯干深度前屈并放低手臂，然后恢复直立。",
        },
        "observable_attributes": {
            "primary_effectors": ["torso", "left_arm", "right_arm"],
            "temporal_pattern": ["upright_to_deep_flexion_to_upright"],
            "spatial_pattern": ["deep_forward_torso_flexion", "arms_lower_with_torso"],
            "unavailable_cues": ["kneeling", "leg_flexion", "forehead_ground_contact"],
        },
        "required_dofs": dofs(
            ["torso"],
            TORSO,
            optional_joint_names=LEFT_ARM + RIGHT_ARM,
            unavailable_channels=UNMAPPED_CHANNELS + ["lower_body"],
        ),
        "context_dependency": {
            "level": "partial",
            "type": "missing_lower_body",
            "details": "The fixed-base 15DoF contract preserves torso and arm motion but not kneeling.",
            "standalone_readability": "medium",
        },
        "text_pattern": r"\bbow(?:ing|s|ed)?\b",
    },
    "bowing_waist": {
        "canonical_prompt": {
            "en": "Bend the torso forward at the waist with a straight back, then return upright.",
            "zh": "保持背部平直，从腰部前屈鞠躬，然后恢复直立。",
        },
        "observable_attributes": {
            "primary_effectors": ["torso"],
            "temporal_pattern": ["upright_to_forward_flexion_to_upright"],
            "spatial_pattern": ["controlled_forward_torso_pitch"],
            "unavailable_cues": ["head_gaze"],
        },
        "required_dofs": dofs(
            ["torso"], TORSO, optional_joint_names=LEFT_ARM + RIGHT_ARM
        ),
        "context_dependency": {
            "level": "none",
            "type": "none",
            "details": "Torso flexion is directly observable in the 15DoF contract.",
            "standalone_readability": "high",
        },
        "text_pattern": r"\bbow(?:ing|s|ed)?\b",
    },
    "hailing_taxi": {
        "canonical_prompt": {
            "en": "Raise one arm high and make a repeated attention-getting wave.",
            "zh": "高举一只手臂，反复挥动以引起注意。",
        },
        "observable_attributes": {
            "primary_effectors": ["one_arm", "wrist"],
            "temporal_pattern": ["arm_raise", "attention_wave", "may_repeat"],
            "spatial_pattern": ["hand_above_or_outside_shoulder", "gesture_toward_scene"],
            "unavailable_cues": ["taxi", "driver", "eye_contact", "open_hand_shape"],
        },
        "required_dofs": dofs(
            ["at_least_one_arm"],
            alternative_joint_name_sets=[LEFT_ARM, RIGHT_ARM],
            optional_joint_names=TORSO,
        ),
        "context_dependency": {
            "level": "partial",
            "type": "scene_object",
            "details": "The robot shows an attention gesture; taxi-specific meaning needs scene context.",
            "standalone_readability": "high",
        },
        "text_pattern": r"\b(?:hail\w*|taxi|cab|attention[- ]getting)\b",
    },
    "hugging_human": {
        "canonical_prompt": {
            "en": "Open both arms, move them forward and inward as if embracing a person, then release.",
            "zh": "张开双臂，向前并向内环抱一个人，然后松开。",
        },
        "observable_attributes": {
            "primary_effectors": ["left_arm", "right_arm", "torso"],
            "temporal_pattern": ["arms_open", "arms_close_inward", "optional_release"],
            "spatial_pattern": ["bilateral_enclosure_in_front_of_torso"],
            "unavailable_cues": ["second_person", "body_contact", "facial_expression"],
        },
        "required_dofs": dofs(
            ["both_arms"],
            LEFT_ARM_CORE + RIGHT_ARM_CORE,
            optional_joint_names=LEFT_WRIST + RIGHT_WRIST + TORSO,
        ),
        "context_dependency": {
            "level": "required",
            "type": "missing_second_person",
            "details": "The single-person trajectory omits the interaction partner and contact.",
            "standalone_readability": "medium",
        },
        "text_pattern": r"\b(?:hug\w*|embrac\w*)\b",
    },
    "salute": {
        "canonical_prompt": {
            "en": "Raise the right hand toward the forehead, pause briefly, then lower it.",
            "zh": "将右手抬至额前，短暂停留后放下。",
        },
        "observable_attributes": {
            "primary_effectors": ["right_arm", "right_wrist"],
            "temporal_pattern": ["raise", "brief_hold", "lower"],
            "spatial_pattern": ["right_hand_approaches_forehead"],
            "unavailable_cues": ["joined_fingers", "eye_gaze"],
        },
        "required_dofs": dofs(
            ["right_arm", "right_wrist"], RIGHT_ARM, optional_joint_names=TORSO
        ),
        "context_dependency": {
            "level": "none",
            "type": "none",
            "details": "The right-arm trajectory remains recognizable without finger articulation.",
            "standalone_readability": "high",
        },
        "text_pattern": r"\b(?:salut\w*|forehead|brow)\b",
    },
}

FLAG_ORDER = [
    "empty",
    "cross_action_duplicate",
    "same_action_duplicate",
    "refusal",
    "vague_or_non_observable",
    "canonical_action_text_conflict",
    "output_wrapper",
    "manifest_action_mismatch",
    "manifest_source_text_mismatch",
]
CRITICAL_FLAGS = {
    "empty",
    "cross_action_duplicate",
    "refusal",
    "vague_or_non_observable",
    "canonical_action_text_conflict",
    "manifest_action_mismatch",
    "manifest_source_text_mismatch",
}
REFUSAL_PATTERN = re.compile(
    r"(?:^|\b)(?:sorry|i can(?:not|'t) (?:assist|provide|describe)|"
    r"i (?:am |'m )?unable to (?:assist|provide|describe)|"
    r"cannot provide the requested|cannot assist)",
    re.IGNORECASE,
)
VAGUE_PATTERN = re.compile(
    r"(?:images?|frames?) (?:appear|are|seem(?: to be)?) (?:identical|repetitive)|"
    r"no (?:visible|observable|significant|discernible) "
    r"(?:movement|motion|action|difference|change)|"
    r"do not show any (?:movement|motion|action|change)|"
    r"no clear temporal sequence|motion (?:is )?minimal to non-existent|"
    r"cannot (?:give|provide) (?:a )?detailed description|no content visible",
    re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def action_from_stem(stem):
    return ACTION_PATTERN.sub("", stem)


def normalize_text(value):
    return " ".join((value or "").strip().split())


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_text(value):
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path, root):
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def load_text_index(label_root):
    groups = defaultdict(list)
    label_count = 0
    empty_count = 0
    for path in sorted(label_root.glob("*.txt")):
        text = normalize_text(path.read_text(encoding="utf-8", errors="replace"))
        action = action_from_stem(path.stem)
        groups[text].append({"clip_id": path.stem, "action": action})
        label_count += 1
        empty_count += not text
    cross_action_groups = sum(
        len({item["action"] for item in items}) > 1
        for text, items in groups.items()
        if text
    )
    return groups, {
        "label_count": label_count,
        "unique_normalized_text_count": len(groups) - ("" in groups),
        "empty_text_count": empty_count,
        "duplicate_text_group_count": sum(
            len(items) > 1 for text, items in groups.items() if text
        ),
        "cross_action_duplicate_group_count": cross_action_groups,
    }


def source_text_assessment(
    canonical_action,
    source_text,
    group,
    manifest_action,
    manifest_source_text,
):
    normalized = normalize_text(source_text)
    actions = sorted({item["action"] for item in group}) if normalized else []
    flags = set()
    if not normalized:
        flags.add("empty")
    if len(group) > 1:
        flags.add(
            "cross_action_duplicate" if len(actions) > 1 else "same_action_duplicate"
        )
    if REFUSAL_PATTERN.search(normalized):
        flags.add("refusal")
    if VAGUE_PATTERN.search(normalized):
        flags.add("vague_or_non_observable")
    if normalized and not re.search(
        ACTION_SPECS[canonical_action]["text_pattern"], normalized, re.IGNORECASE
    ):
        flags.add("canonical_action_text_conflict")
    if re.match(r"^\s*Output\s*:", normalized, re.IGNORECASE):
        flags.add("output_wrapper")
    if manifest_action and manifest_action != canonical_action:
        flags.add("manifest_action_mismatch")
    if manifest_source_text and normalize_text(manifest_source_text) != normalized:
        flags.add("manifest_source_text_mismatch")
    ordered_flags = [flag for flag in FLAG_ORDER if flag in flags]

    if "empty" in flags or "refusal" in flags:
        source_confidence = 0.0
    elif "canonical_action_text_conflict" in flags:
        source_confidence = 0.1
    elif "cross_action_duplicate" in flags:
        source_confidence = 0.2
    elif "vague_or_non_observable" in flags:
        source_confidence = 0.3
    elif "same_action_duplicate" in flags:
        source_confidence = 0.7
    else:
        source_confidence = 0.9
    critical = bool(flags & CRITICAL_FLAGS)
    return {
        "normalized_text_sha256": sha256_text(normalized),
        "flags": ordered_flags,
        "duplicate_group_size": len(group),
        "duplicate_action_count": len(actions),
        "duplicate_actions": actions,
        "conditioning_eligible": False,
        "recommended_use": "reject" if critical else "auxiliary_after_manual_review",
    }, source_confidence, critical


def build_record(root, manifest_path, row, text_groups):
    clip_id = row.get("clip_id", "").strip()
    if not clip_id:
        raise ValueError("Every manifest row requires clip_id")
    canonical_action = action_from_stem(clip_id)
    if canonical_action not in ACTION_SPECS:
        raise ValueError(f"Unsupported filename action for semantics v1: {clip_id}")
    motion_relpath = row.get("motion_relpath", "").strip()
    label_relpath = row.get("label_relpath", "").strip()
    if not motion_relpath or not label_relpath:
        raise ValueError(f"Missing source path for {clip_id}")
    motion_path = root / motion_relpath
    label_path = root / label_relpath
    if not motion_path.is_file():
        raise FileNotFoundError(motion_path)
    source_text = (
        label_path.read_text(encoding="utf-8", errors="replace").strip()
        if label_path.is_file()
        else ""
    )
    normalized = normalize_text(source_text)
    group = text_groups.get(normalized, []) if normalized else []
    text_quality, source_confidence, critical = source_text_assessment(
        canonical_action,
        source_text,
        group,
        row.get("action", "").strip(),
        row.get("semantic_label", ""),
    )
    action_confidence = 0.99
    if "manifest_action_mismatch" in text_quality["flags"]:
        action_confidence = 0.7
    prompt_confidence = 0.95
    overall_confidence = round(
        0.65 * action_confidence + 0.20 * prompt_confidence + 0.15 * source_confidence,
        3,
    )
    confidence_rationale = [
        "canonical_action is parsed from the source filename",
        "canonical_prompt is curated for observable ULA V2 motion",
        "GPT-4V source text is auxiliary and never overrides the filename action",
    ]
    if text_quality["flags"]:
        confidence_rationale.append(
            "source text flags: " + ", ".join(text_quality["flags"])
        )
    spec = ACTION_SPECS[canonical_action]
    return {
        "schema_version": "1.0.0",
        "clip_id": clip_id,
        "canonical_action": canonical_action,
        "canonical_prompt": deepcopy(spec["canonical_prompt"]),
        "observable_attributes": deepcopy(spec["observable_attributes"]),
        "required_dofs": deepcopy(spec["required_dofs"]),
        "context_dependency": deepcopy(spec["context_dependency"]),
        "source_text": source_text,
        "source_text_quality": text_quality,
        "source": {
            "dataset": "Motion-X++/HAA500",
            "dataset_revision": DATASET_REVISION,
            "manifest_relpath": display_path(manifest_path, root),
            "motion_relpath": motion_relpath,
            "motion_sha256": sha256_file(motion_path),
            "source_text_relpath": label_relpath,
            "source_text_sha256": sha256_file(label_path) if label_path.is_file() else None,
        },
        "provenance": {
            "primary_label_source": "filename_action",
            "filename_action": canonical_action,
            "manifest_action": row.get("action", "").strip(),
            "auxiliary_text_source": "Motion-X++ semantic_label generated by GPT-4V",
            "auxiliary_text_role": "auxiliary_only",
            "builder": "tools/human_motion_collection/build_expression_semantics_v1.py",
            "builder_version": BUILDER_VERSION,
        },
        "confidence": {
            "canonical_action": action_confidence,
            "canonical_prompt": prompt_confidence,
            "source_text_alignment": source_confidence,
            "overall": overall_confidence,
            "rationale": confidence_rationale,
        },
        "review_status": {
            "state": (
                "machine_flagged_pending_review"
                if critical
                else "machine_labeled_pending_review"
            ),
            "manual_review_required": True,
            "accepted_for_training": False,
        },
    }


def semantics_schema():
    string_array = {
        "type": "array",
        "items": {"type": "string", "minLength": 1},
        "uniqueItems": True,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:ula:motion-xplusplus:expression-semantics-haa500:v1",
        "title": "ULA robot-observable HAA500 expression semantics v1",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "clip_id",
            "canonical_action",
            "canonical_prompt",
            "observable_attributes",
            "required_dofs",
            "context_dependency",
            "source_text",
            "source_text_quality",
            "source",
            "provenance",
            "confidence",
            "review_status",
        ],
        "properties": {
            "schema_version": {"const": "1.0.0"},
            "clip_id": {"type": "string", "minLength": 1},
            "canonical_action": {"enum": sorted(ACTION_SPECS)},
            "canonical_prompt": {
                "type": "object",
                "additionalProperties": False,
                "required": ["en", "zh"],
                "properties": {
                    "en": {"type": "string", "minLength": 1},
                    "zh": {"type": "string", "minLength": 1},
                },
            },
            "observable_attributes": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "primary_effectors",
                    "temporal_pattern",
                    "spatial_pattern",
                    "unavailable_cues",
                ],
                "properties": {
                    "primary_effectors": string_array,
                    "temporal_pattern": string_array,
                    "spatial_pattern": string_array,
                    "unavailable_cues": string_array,
                },
            },
            "required_dofs": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "joint_space",
                    "required_groups",
                    "required_joint_names",
                    "alternative_joint_name_sets",
                    "optional_joint_names",
                    "minimum_observable_dof_count",
                    "unavailable_channels",
                ],
                "properties": {
                    "joint_space": {"const": "ula_v2_15d"},
                    "required_groups": string_array,
                    "required_joint_names": string_array,
                    "alternative_joint_name_sets": {
                        "type": "array",
                        "items": string_array,
                    },
                    "optional_joint_names": string_array,
                    "minimum_observable_dof_count": {"type": "integer", "minimum": 1},
                    "unavailable_channels": string_array,
                },
            },
            "context_dependency": {
                "type": "object",
                "additionalProperties": False,
                "required": ["level", "type", "details", "standalone_readability"],
                "properties": {
                    "level": {"enum": ["none", "partial", "required"]},
                    "type": {
                        "enum": [
                            "none",
                            "face_and_hand_detail",
                            "missing_lower_body",
                            "scene_object",
                            "missing_second_person",
                        ]
                    },
                    "details": {"type": "string", "minLength": 1},
                    "standalone_readability": {"enum": ["high", "medium", "low"]},
                },
            },
            "source_text": {"type": "string"},
            "source_text_quality": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "normalized_text_sha256",
                    "flags",
                    "duplicate_group_size",
                    "duplicate_action_count",
                    "duplicate_actions",
                    "conditioning_eligible",
                    "recommended_use",
                ],
                "properties": {
                    "normalized_text_sha256": {
                        "type": "string",
                        "pattern": "^[0-9a-f]{64}$",
                    },
                    "flags": {
                        "type": "array",
                        "items": {"enum": FLAG_ORDER},
                        "uniqueItems": True,
                    },
                    "duplicate_group_size": {"type": "integer", "minimum": 0},
                    "duplicate_action_count": {"type": "integer", "minimum": 0},
                    "duplicate_actions": string_array,
                    "conditioning_eligible": {"const": False},
                    "recommended_use": {
                        "enum": ["reject", "auxiliary_after_manual_review"]
                    },
                },
            },
            "source": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "dataset",
                    "dataset_revision",
                    "manifest_relpath",
                    "motion_relpath",
                    "motion_sha256",
                    "source_text_relpath",
                    "source_text_sha256",
                ],
                "properties": {
                    "dataset": {"const": "Motion-X++/HAA500"},
                    "dataset_revision": {"const": DATASET_REVISION},
                    "manifest_relpath": {"type": "string", "minLength": 1},
                    "motion_relpath": {"type": "string", "minLength": 1},
                    "motion_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                    "source_text_relpath": {"type": "string", "minLength": 1},
                    "source_text_sha256": {
                        "anyOf": [
                            {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                            {"type": "null"},
                        ]
                    },
                },
            },
            "provenance": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "primary_label_source",
                    "filename_action",
                    "manifest_action",
                    "auxiliary_text_source",
                    "auxiliary_text_role",
                    "builder",
                    "builder_version",
                ],
                "properties": {
                    "primary_label_source": {"const": "filename_action"},
                    "filename_action": {"enum": sorted(ACTION_SPECS)},
                    "manifest_action": {"type": "string"},
                    "auxiliary_text_source": {"type": "string", "minLength": 1},
                    "auxiliary_text_role": {"const": "auxiliary_only"},
                    "builder": {"type": "string", "minLength": 1},
                    "builder_version": {"const": BUILDER_VERSION},
                },
            },
            "confidence": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "canonical_action",
                    "canonical_prompt",
                    "source_text_alignment",
                    "overall",
                    "rationale",
                ],
                "properties": {
                    "canonical_action": {"type": "number", "minimum": 0, "maximum": 1},
                    "canonical_prompt": {"type": "number", "minimum": 0, "maximum": 1},
                    "source_text_alignment": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "overall": {"type": "number", "minimum": 0, "maximum": 1},
                    "rationale": string_array,
                },
            },
            "review_status": {
                "type": "object",
                "additionalProperties": False,
                "required": ["state", "manual_review_required", "accepted_for_training"],
                "properties": {
                    "state": {
                        "enum": [
                            "machine_labeled_pending_review",
                            "machine_flagged_pending_review",
                        ]
                    },
                    "manual_review_required": {"const": True},
                    "accepted_for_training": {"const": False},
                },
            },
        },
    }


CSV_FIELDS = [
    "clip_id",
    "canonical_action",
    "canonical_prompt_en",
    "canonical_prompt_zh",
    "primary_effectors",
    "temporal_pattern",
    "spatial_pattern",
    "unavailable_cues",
    "required_groups",
    "required_joint_names",
    "alternative_joint_name_sets",
    "optional_joint_names",
    "minimum_observable_dof_count",
    "unavailable_channels",
    "context_level",
    "context_type",
    "standalone_readability",
    "context_details",
    "source_text",
    "source_text_flags",
    "duplicate_group_size",
    "duplicate_action_count",
    "duplicate_actions",
    "conditioning_eligible",
    "recommended_use",
    "motion_relpath",
    "motion_sha256",
    "source_text_relpath",
    "source_text_sha256",
    "primary_label_source",
    "auxiliary_text_role",
    "canonical_action_confidence",
    "canonical_prompt_confidence",
    "source_text_alignment_confidence",
    "overall_confidence",
    "review_state",
    "manual_review_required",
    "accepted_for_training",
]


def flatten_record(record):
    observable = record["observable_attributes"]
    required = record["required_dofs"]
    context = record["context_dependency"]
    quality = record["source_text_quality"]
    source = record["source"]
    provenance = record["provenance"]
    confidence = record["confidence"]
    review = record["review_status"]
    return {
        "clip_id": record["clip_id"],
        "canonical_action": record["canonical_action"],
        "canonical_prompt_en": record["canonical_prompt"]["en"],
        "canonical_prompt_zh": record["canonical_prompt"]["zh"],
        "primary_effectors": "|".join(observable["primary_effectors"]),
        "temporal_pattern": "|".join(observable["temporal_pattern"]),
        "spatial_pattern": "|".join(observable["spatial_pattern"]),
        "unavailable_cues": "|".join(observable["unavailable_cues"]),
        "required_groups": "|".join(required["required_groups"]),
        "required_joint_names": "|".join(required["required_joint_names"]),
        "alternative_joint_name_sets": ";".join(
            "|".join(group) for group in required["alternative_joint_name_sets"]
        ),
        "optional_joint_names": "|".join(required["optional_joint_names"]),
        "minimum_observable_dof_count": required["minimum_observable_dof_count"],
        "unavailable_channels": "|".join(required["unavailable_channels"]),
        "context_level": context["level"],
        "context_type": context["type"],
        "standalone_readability": context["standalone_readability"],
        "context_details": context["details"],
        "source_text": record["source_text"],
        "source_text_flags": "|".join(quality["flags"]),
        "duplicate_group_size": quality["duplicate_group_size"],
        "duplicate_action_count": quality["duplicate_action_count"],
        "duplicate_actions": "|".join(quality["duplicate_actions"]),
        "conditioning_eligible": quality["conditioning_eligible"],
        "recommended_use": quality["recommended_use"],
        "motion_relpath": source["motion_relpath"],
        "motion_sha256": source["motion_sha256"],
        "source_text_relpath": source["source_text_relpath"],
        "source_text_sha256": source["source_text_sha256"] or "",
        "primary_label_source": provenance["primary_label_source"],
        "auxiliary_text_role": provenance["auxiliary_text_role"],
        "canonical_action_confidence": confidence["canonical_action"],
        "canonical_prompt_confidence": confidence["canonical_prompt"],
        "source_text_alignment_confidence": confidence["source_text_alignment"],
        "overall_confidence": confidence["overall"],
        "review_state": review["state"],
        "manual_review_required": review["manual_review_required"],
        "accepted_for_training": review["accepted_for_training"],
    }


def atomic_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def build_and_write(root, manifest=None, output_dir=None):
    root = Path(root).resolve()
    manifest = (
        Path(manifest).resolve()
        if manifest
        else root / "catalog/expression_candidates_haa500.csv"
    )
    output_dir = Path(output_dir).resolve() if output_dir else root / "catalog"
    label_root = root / "raw/Motion-Xplusplus/extracted/haa500/semantic_label"
    text_groups, text_index_summary = load_text_index(label_root)
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    records = [build_record(root, manifest, row, text_groups) for row in rows]
    records.sort(key=lambda item: item["clip_id"])
    clip_ids = [record["clip_id"] for record in records]
    if len(clip_ids) != len(set(clip_ids)):
        raise ValueError("Duplicate clip_id in expression candidate manifest")

    schema = semantics_schema()
    schema_bytes = (
        json.dumps(schema, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    jsonl_bytes = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for record in records
    ).encode("utf-8")
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(flatten_record(record) for record in records)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    schema_path = output_dir / f"{OUTPUT_STEM}.schema.json"
    jsonl_path = output_dir / f"{OUTPUT_STEM}.jsonl"
    csv_path = output_dir / f"{OUTPUT_STEM}.csv"
    summary_path = output_dir / f"{OUTPUT_STEM}.summary.json"
    flag_counts = Counter(
        flag for record in records for flag in record["source_text_quality"]["flags"]
    )
    confidence_values = [record["confidence"]["overall"] for record in records]
    summary = {
        "schema_version": "1.0.0",
        "dataset": "Motion-X++/HAA500",
        "dataset_revision": DATASET_REVISION,
        "record_count": len(records),
        "primary_label_policy": "filename_action",
        "auxiliary_text_policy": "GPT-4V source text never overrides canonical_action",
        "conditioning_policy": "deny_until_manual_review",
        "deterministic_build": True,
        "counts_by_action": dict(sorted(Counter(clip["canonical_action"] for clip in records).items())),
        "counts_by_source_text_flag": dict(sorted(flag_counts.items())),
        "counts_by_review_state": dict(
            sorted(Counter(clip["review_status"]["state"] for clip in records).items())
        ),
        "conditioning_eligible_count": sum(
            clip["source_text_quality"]["conditioning_eligible"] for clip in records
        ),
        "accepted_for_training_count": sum(
            clip["review_status"]["accepted_for_training"] for clip in records
        ),
        "overall_confidence": {
            "minimum": min(confidence_values) if confidence_values else None,
            "mean": (
                round(sum(confidence_values) / len(confidence_values), 6)
                if confidence_values
                else None
            ),
            "maximum": max(confidence_values) if confidence_values else None,
        },
        "duplicate_analysis_scope": {
            "label_root": display_path(label_root, root),
            **text_index_summary,
        },
        "outputs": {
            "schema": display_path(schema_path, root),
            "jsonl": display_path(jsonl_path, root),
            "csv": display_path(csv_path, root),
            "summary": display_path(summary_path, root),
        },
        "output_sha256": {
            "schema": sha256_bytes(schema_bytes),
            "jsonl": sha256_bytes(jsonl_bytes),
            "csv": sha256_bytes(csv_bytes),
        },
    }
    summary_bytes = (
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    atomic_write(schema_path, schema_bytes)
    atomic_write(jsonl_path, jsonl_bytes)
    atomic_write(csv_path, csv_bytes)
    atomic_write(summary_path, summary_bytes)
    return summary, records


def main():
    args = parse_args()
    summary, _ = build_and_write(args.root, args.manifest, args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
