#!/usr/bin/env python3
"""Build fail-closed BEAT2 semantics for the Kimodo conditioning contract.

BEAT2 conversation clips do not carry trustworthy discrete emotion labels.  This
tool therefore creates only conservative behavior candidates and trajectory-
derived motion styles automatically.  Emotion supervision is enabled only after
an explicit human review passes the bilingual prompt checks in this module.

The input is the JSONL produced by ``label_ula_v2_18d_motion.py``.  An optional
human-review JSONL can confirm or reject individual records.  Speech transcripts
are deliberately neither read nor copied into the outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "beat2_kimodo_semantics_v1.1.0"
BEHAVIOR_ONTOLOGY_VERSION = "kimodo_behavior_27_v1"
EMOTION_ONTOLOGY_VERSION = "kimodo_emotion_6_v1"
AUTOMATIC_BEHAVIOR_SOURCE = "trajectory_only_conservative_conversation_candidate"
UNRESOLVED_EMOTION_SOURCE = "unresolved_beat2_has_no_emotion_ground_truth"

KIMODO_BEHAVIOR_IDS = (
    "Behavior.IdleLowPower",
    "Behavior.IdleQuiet",
    "Behavior.IdleAttentive",
    "Behavior.InteractPresence",
    "Behavior.GreetingOwner01",
    "Behavior.GreetingOwner02",
    "Behavior.GreetingOwner03",
    "Behavior.GreetingOwner04",
    "Behavior.GreetingGuest",
    "Behavior.Farewell",
    "Behavior.Joy",
    "Behavior.Aversion",
    "Behavior.CuriousLook",
    "Behavior.Comfort",
    "Behavior.Alert",
    "Behavior.SeekAttention",
    "Behavior.ActiveListening",
    "Behavior.Disappointment",
    "Behavior.Withdrawal",
    "Behavior.Hesitate",
    "Behavior.Search",
    "Behavior.Error",
    "Behavior.Dance.Base",
    "Behavior.Dance.Sway",
    "Behavior.Dance.Accent",
    "Behavior.Dance.Stop",
    "Behavior.FingerHeart",
)
KIMODO_EMOTION_IDS = ("neutral", "sad", "happy", "angry", "surprise", "fear")
MOTION_STYLE_IDS = ("slow_safe", "energetic", "sharp", "relaxed", "restrained")

AUTO_BEHAVIOR_IDS = {
    "Behavior.IdleQuiet",
    "Behavior.IdleAttentive",
    "Behavior.InteractPresence",
    "Behavior.ActiveListening",
}
EMOTION_REVIEW_STATUSES = {"unresolved", "human_confirmed", "rejected"}
BEHAVIOR_REVIEW_STATUSES = {"candidate_unreviewed", "human_confirmed", "rejected"}

# A confirmed prompt must describe robot-visible motion, not only an abstract
# communicative intent such as "greet" or "be happy".
EN_ACTION_PATTERN = re.compile(
    r"\b(?:arm|arms|hand|hands|head|torso|shoulder|shoulders|elbow|elbows|"
    r"wrist|wrists|nod|nodding|turn|turning|tilt|tilting|raise|raising|lower|"
    r"lowering|open|opening|close|closing|sway|swaying|wave|waving|reach|"
    r"reaching|bow|bowing|pause|pausing)\b",
    re.IGNORECASE,
)
ZH_ACTION_PATTERN = re.compile(
    r"(?:手臂|双臂|左臂|右臂|手部|双手|左手|右手|头部|转头|点头|侧倾|躯干|"
    r"肩部|肩膀|手肘|肘部|手腕|抬起|抬高|放下|降低|张开|收拢|闭合|摇摆|"
    r"挥动|挥手|伸出|前伸|鞠躬|停顿|停留)"
)
GENERIC_EN_PROMPTS = {
    "perform an action",
    "make a gesture",
    "move naturally",
    "express the emotion",
}
GENERIC_ZH_PROMPTS = {"做一个动作", "做一个手势", "自然地运动", "表达情绪"}
WINDOW_ID_PATTERN = re.compile(
    r"^(?P<source_clip_id>.+)_f(?P<start>\d+)-(?P<end>\d+)$"
)
GENERIC_CANONICAL_ACTIONS = {
    "action",
    "gesture",
    "motion",
    "robot_observable_upper_body_motion",
    "upper_body_motion",
    *KIMODO_EMOTION_IDS,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--human-reviews",
        type=Path,
        help="Optional JSONL of explicit human confirmations or rejections.",
    )
    return parser.parse_args(argv)


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    _atomic_write(path, "".join(_stable_json(record) + "\n" for record in records))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read JSONL {path}: {error}") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"record at {path}:{line_number} must be an object")
        clip_id = _clip_id(record)
        if clip_id in seen:
            raise ValueError(f"duplicate clip_id {clip_id!r} in {path}")
        seen.add(clip_id)
        records.append(record)
    return records


def _clip_id(record: dict[str, Any]) -> str:
    # Window/task IDs take precedence over a base source clip ID so multiple
    # windows from the same source cannot collapse onto one semantic record.
    value = record.get("task_id") or record.get("record_id") or record.get("clip_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("record requires clip_id, task_id, or record_id")
    return value.strip()


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return " ".join(value.strip().split())


def _features(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("observable_features")
    if not isinstance(value, dict):
        raise ValueError(f"{_clip_id(record)}: observable_features must be an object")
    return value


def _feature_string(mapping: Any, key: str, default: str) -> str:
    if isinstance(mapping, dict) and isinstance(mapping.get(key), str):
        return mapping[key]
    return default


def conservative_behavior_candidate(features: dict[str, Any]) -> str:
    """Choose only a low-semantic conversation-state candidate.

    The rules intentionally never emit greeting, comfort, warning, emotion, or
    dance behaviors.  Those meanings are not identifiable from joint motion
    alone and require human review with the original visual context.
    """

    arm = features.get("arm") if isinstance(features.get("arm"), dict) else {}
    laterality = _feature_string(arm, "laterality", "none")
    amplitude = _feature_string(arm, "amplitude", "very_small")
    head_level = _feature_string(features, "head_motion", "minimal")
    torso_level = _feature_string(features, "torso_motion", "minimal")
    repeated = (
        features.get("head", {}).get("repeated_pattern", {}).get("pattern", "none")
        if isinstance(features.get("head"), dict)
        else "none"
    )

    if laterality != "none" and amplitude not in {"very_small", "small"}:
        return "Behavior.InteractPresence"
    if repeated == "repeated_pitch_nods":
        return "Behavior.ActiveListening"
    if head_level in {"subtle", "clear"} or torso_level == "clear":
        return "Behavior.IdleAttentive"
    if laterality != "none":
        return "Behavior.ActiveListening"
    return "Behavior.IdleQuiet"


def motion_style_from_features(features: dict[str, Any]) -> tuple[str, dict[str, str]]:
    """Map robot kinematics to a style label without estimating emotion."""

    arm = features.get("arm") if isinstance(features.get("arm"), dict) else {}
    overall = (
        features.get("overall_motion")
        if isinstance(features.get("overall_motion"), dict)
        else {}
    )
    amplitude = _feature_string(arm, "amplitude", "very_small")
    continuity = _feature_string(arm, "continuity", "intermittent")
    pace = _feature_string(overall, "pace", "minimal")

    if pace == "quick" and continuity == "intermittent":
        label = "sharp"
    elif pace == "quick" or amplitude == "large":
        label = "energetic"
    elif pace == "slow" and amplitude not in {"large"}:
        label = "slow_safe"
    elif amplitude in {"very_small", "small"} or pace == "minimal":
        label = "restrained"
    else:
        label = "relaxed"
    return label, {
        "amplitude": amplitude,
        "continuity": continuity,
        "pace": pace,
        "head_motion": _feature_string(features, "head_motion", "minimal"),
        "torso_motion": _feature_string(features, "torso_motion", "minimal"),
        "source": "trajectory_only_18d_observable_features",
    }


def _controlled_feature(
    mapping: Any,
    key: str,
    allowed: set[str],
    default: str,
) -> str:
    value = _feature_string(mapping, key, default)
    return value if value in allowed else default


def _axis_for_action(features: dict[str, Any], group: str) -> str:
    group_features = features.get(group)
    if not isinstance(group_features, dict):
        return "multi_axis"
    axis_motion = group_features.get("axis_motion")
    if not isinstance(axis_motion, dict):
        return "multi_axis"
    axis = axis_motion.get("dominant_axis")
    return axis if axis in {"roll", "pitch", "yaw"} else "multi_axis"


def canonical_action_from_features(features: dict[str, Any]) -> str:
    """Build a concrete, emotion-free key from robot-observable kinematics."""

    arm = features.get("arm") if isinstance(features.get("arm"), dict) else {}
    laterality = _controlled_feature(
        arm, "laterality", {"none", "left", "right", "both"}, "none"
    )
    amplitude = _controlled_feature(
        arm,
        "amplitude",
        {"very_small", "small", "medium", "moderate", "large"},
        "very_small",
    )
    continuity = _controlled_feature(
        arm, "continuity", {"continuous", "intermittent"}, "intermittent"
    )
    parts: list[str] = []
    if laterality != "none":
        side = "bilateral" if laterality == "both" else laterality
        amplitude_key = "medium" if amplitude == "moderate" else amplitude
        parts.append(f"{side}_{amplitude_key}_{continuity}_arm_motion")

    head = features.get("head") if isinstance(features.get("head"), dict) else {}
    repeated = head.get("repeated_pattern")
    repeated_pattern = repeated.get("pattern") if isinstance(repeated, dict) else None
    repeat_names = {
        "repeated_pitch_nods": "repeated_head_pitch_nods",
        "repeated_yaw_turns": "repeated_head_yaw_turns",
        "repeated_roll_tilts": "repeated_head_roll_tilts",
    }
    head_level = _controlled_feature(
        features, "head_motion", {"minimal", "subtle", "clear"}, "minimal"
    )
    if repeated_pattern in repeat_names:
        parts.append(repeat_names[repeated_pattern])
    elif head_level != "minimal":
        parts.append(f"{head_level}_{_axis_for_action(features, 'head')}_head_motion")

    torso_level = _controlled_feature(
        features, "torso_motion", {"minimal", "subtle", "clear"}, "minimal"
    )
    if torso_level != "minimal":
        parts.append(f"{torso_level}_{_axis_for_action(features, 'torso')}_torso_motion")
    return "_with_".join(parts) if parts else "near_static_upper_body_pose"


def _canonical_action(
    annotation: dict[str, Any], features: dict[str, Any]
) -> tuple[str, str]:
    value = annotation.get("canonical_action")
    if isinstance(value, str) and value.strip():
        normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
        emotion_tokens = set(normalized.split("_")) & set(KIMODO_EMOTION_IDS)
        if normalized not in GENERIC_CANONICAL_ACTIONS and not emotion_tokens:
            return value.strip(), "preserved_input_canonical_action"
    return canonical_action_from_features(features), "trajectory_only_18d_observable_features"


def _window_provenance(
    annotation: dict[str, Any], clip_id: str, source_clip_id: str | None
) -> dict[str, int]:
    match = WINDOW_ID_PATTERN.fullmatch(clip_id)
    parsed_source = match.group("source_clip_id") if match else None
    parsed_start = int(match.group("start")) if match else None
    parsed_end = int(match.group("end")) if match else None

    explicit_start = annotation.get(
        "source_window_start_frame", annotation.get("start_frame")
    )
    explicit_end = annotation.get(
        "source_window_end_frame_exclusive", annotation.get("end_frame_exclusive")
    )
    if (explicit_start is None) != (explicit_end is None):
        raise ValueError(f"{clip_id}: source window provenance must include start and end")
    if explicit_start is not None:
        if (
            isinstance(explicit_start, bool)
            or not isinstance(explicit_start, int)
            or isinstance(explicit_end, bool)
            or not isinstance(explicit_end, int)
        ):
            raise ValueError(f"{clip_id}: source window provenance must use integer frames")
        if match and (explicit_start != parsed_start or explicit_end != parsed_end):
            raise ValueError(f"{clip_id}: explicit source window conflicts with task_id")
        start, end = explicit_start, explicit_end
    elif match:
        start, end = parsed_start, parsed_end
    else:
        return {}

    if start is None or end is None or start < 0 or end <= start:
        raise ValueError(f"{clip_id}: source window must be a non-empty half-open interval")
    source_matches = (
        parsed_source == source_clip_id
        or bool(
            parsed_source
            and source_clip_id
            and re.search(
                rf"(?:^|__){re.escape(source_clip_id)}_sem\d+$",
                parsed_source,
            )
        )
    )
    if parsed_source and source_clip_id and not source_matches:
        raise ValueError(f"{clip_id}: task_id source conflicts with source_clip_id")
    return {
        "source_window_start_frame": start,
        "source_window_end_frame_exclusive": end,
    }


def _source_group_key(record: dict[str, Any], speaker_key: str) -> str:
    source_clip = record.get("source_clip_id")
    if not isinstance(source_clip, str) or not source_clip.strip():
        source_clip = _clip_id(record).split("_f", 1)[0]
    return f"beat2/{speaker_key}/{source_clip.strip()}"


def _prompt(record: dict[str, Any]) -> dict[str, str]:
    value = record.get("canonical_prompt")
    if not isinstance(value, dict):
        raise ValueError(f"{_clip_id(record)}: canonical_prompt must be an object")
    return {
        "en": _nonempty_string(value.get("en"), "canonical_prompt.en"),
        "zh": _nonempty_string(value.get("zh"), "canonical_prompt.zh"),
    }


def build_automatic_record(annotation: dict[str, Any]) -> dict[str, Any]:
    clip_id = _clip_id(annotation)
    speaker_key = _nonempty_string(annotation.get("speaker_key"), "speaker_key")
    features = _features(annotation)
    behavior_id = conservative_behavior_candidate(features)
    style, style_evidence = motion_style_from_features(features)
    canonical_action, canonical_action_source = _canonical_action(annotation, features)
    raw_source_clip_id = annotation.get("source_clip_id")
    source_clip_id = (
        raw_source_clip_id.strip()
        if isinstance(raw_source_clip_id, str) and raw_source_clip_id.strip()
        else None
    )
    window_provenance = _window_provenance(annotation, clip_id, source_clip_id)
    if behavior_id not in AUTO_BEHAVIOR_IDS:
        raise AssertionError(f"automatic rule emitted unsafe behavior {behavior_id}")

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "clip_id": clip_id,
        "canonical_action": canonical_action,
        "canonical_action_source": canonical_action_source,
        "behavior_id": behavior_id,
        "behavior_review_status": "candidate_unreviewed",
        "behavior_source": AUTOMATIC_BEHAVIOR_SOURCE,
        "behavior_supervision_mask": False,
        "behavior_training_eligibility": "blocked_unreviewed_behavior",
        "emotion_id": None,
        "emotion_review_status": "unresolved",
        "emotion_source": UNRESOLVED_EMOTION_SOURCE,
        "emotion_confidence": 0.0,
        "emotion_supervision_mask": False,
        "canonical_prompt": _prompt(annotation),
        "observed_affect": {
            "label": None,
            "source": UNRESOLVED_EMOTION_SOURCE,
            "confidence": 0.0,
        },
        "motion_style": style,
        "motion_style_evidence": style_evidence,
        "speaker_key": speaker_key,
        "source_group_key": _source_group_key(annotation, speaker_key),
        "ontology": {
            "behavior": BEHAVIOR_ONTOLOGY_VERSION,
            "emotion": EMOTION_ONTOLOGY_VERSION,
        },
        "review_required": True,
        "network_semantic_supervision_ready": False,
        "motion_style_training_eligibility": "pending_adjudication",
        "emotion_training_eligibility": "blocked_unresolved_emotion",
        **window_provenance,
    }
    for key in (
        "source_clip_id",
        "official_split",
        "trajectory_path",
        "trajectory_sha256",
        "quality_json",
        "quality_json_sha256",
        "robot_contract",
        "source_beat2_npz",
        "source_sha256",
    ):
        if annotation.get(key) is not None:
            record[key] = annotation[key]
    return record


def _normalized_prompt_for_generic_check(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", " ", text.lower()).strip()


def validate_specific_bilingual_action(prompt: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(prompt, dict):
        return ["canonical_prompt must be an object"]
    en = prompt.get("en")
    zh = prompt.get("zh")
    if not isinstance(en, str) or not en.strip():
        errors.append("canonical_prompt.en must be non-empty")
    elif (
        _normalized_prompt_for_generic_check(en) in GENERIC_EN_PROMPTS
        or not EN_ACTION_PATTERN.search(en)
    ):
        errors.append("canonical_prompt.en must name a concrete robot-visible action")
    if not isinstance(zh, str) or not zh.strip():
        errors.append("canonical_prompt.zh must be non-empty")
    elif (
        _normalized_prompt_for_generic_check(zh) in GENERIC_ZH_PROMPTS
        or not ZH_ACTION_PATTERN.search(zh)
    ):
        errors.append("canonical_prompt.zh must name a concrete robot-visible action")
    return errors


def _contains_exact_emotion_word(text: str, emotion_id: str) -> bool:
    pattern = rf"(?<![A-Za-z]){re.escape(emotion_id)}(?![A-Za-z])"
    return re.search(pattern, text, re.IGNORECASE) is not None


def apply_human_review(record: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    decision = review.get("decision")
    if decision not in {"confirmed", "behavior_confirmed", "rejected"}:
        raise ValueError(
            f"{record['clip_id']}: review decision must be confirmed, "
            "behavior_confirmed, or rejected"
        )
    if review.get("reviewer_kind") != "human":
        raise ValueError(f"{record['clip_id']}: semantic confirmation requires reviewer_kind=human")
    reviewer_id = _nonempty_string(review.get("reviewer_id"), "reviewer_id")
    reviewed = dict(record)
    reviewed["human_review"] = {
        "reviewer_id": reviewer_id,
        "reviewer_kind": "human",
        "reviewed_at": _nonempty_string(review.get("reviewed_at"), "reviewed_at"),
        "notes": str(review.get("notes") or ""),
        "decision": decision,
    }
    if decision == "rejected":
        reviewed.update(
            {
                "behavior_review_status": "rejected",
                "behavior_source": "human_review_rejected",
                "behavior_supervision_mask": False,
                "behavior_training_eligibility": "blocked_rejected_behavior",
                "emotion_id": None,
                "emotion_review_status": "rejected",
                "emotion_source": "human_review_rejected",
                "emotion_confidence": 0.0,
                "emotion_supervision_mask": False,
                "observed_affect": {
                    "label": None,
                    "source": "human_review_rejected",
                    "confidence": 0.0,
                },
                "review_required": False,
                "network_semantic_supervision_ready": False,
                "emotion_training_eligibility": "blocked_rejected_emotion",
            }
        )
        return reviewed

    behavior_id = review.get("behavior_id")
    if behavior_id not in KIMODO_BEHAVIOR_IDS:
        raise ValueError(f"{record['clip_id']}: unknown behavior_id {behavior_id!r}")
    prompt = review.get("canonical_prompt")
    prompt_errors = validate_specific_bilingual_action(prompt)
    if prompt_errors:
        raise ValueError(f"{record['clip_id']}: " + "; ".join(prompt_errors))
    reviewed.update(
        {
            "behavior_id": behavior_id,
            "behavior_review_status": "human_confirmed",
            "behavior_source": "human_visual_review",
            "behavior_supervision_mask": True,
            "behavior_training_eligibility": "pending_adjudication",
            "canonical_prompt": {
                "en": _nonempty_string(prompt["en"], "canonical_prompt.en"),
                "zh": _nonempty_string(prompt["zh"], "canonical_prompt.zh"),
            },
        }
    )
    if decision == "behavior_confirmed":
        reviewed.update(
            {
                "review_required": True,
                "network_semantic_supervision_ready": False,
                "emotion_training_eligibility": "blocked_unresolved_emotion",
            }
        )
        return reviewed

    emotion_id = review.get("emotion_id")
    if emotion_id not in KIMODO_EMOTION_IDS:
        raise ValueError(f"{record['clip_id']}: unknown emotion_id {emotion_id!r}")
    confidence = review.get("emotion_confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError(f"{record['clip_id']}: emotion_confidence must be numeric")
    confidence = float(confidence)
    if not 0.0 < confidence <= 1.0:
        raise ValueError(f"{record['clip_id']}: emotion_confidence must be in (0, 1]")
    en = str(prompt["en"])
    if not _contains_exact_emotion_word(en, emotion_id):
        raise ValueError(
            f"{record['clip_id']}: canonical_prompt.en must contain exact "
            f"emotion word {emotion_id!r}"
        )

    reviewed.update(
        {
            "emotion_id": emotion_id,
            "emotion_review_status": "human_confirmed",
            "emotion_source": "human_visual_review",
            "emotion_confidence": confidence,
            "emotion_supervision_mask": True,
            "observed_affect": {
                "label": emotion_id,
                "source": "human_visual_review",
                "confidence": confidence,
            },
            "review_required": False,
            "network_semantic_supervision_ready": True,
            "emotion_training_eligibility": "pending_adjudication",
        }
    )
    return reviewed


def validate_network_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    clip_id = record.get("clip_id", "<missing>")
    for field in ("clip_id", "speaker_key", "source_group_key"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            errors.append(f"{field} must be a non-empty string")
    canonical_action = record.get("canonical_action")
    if not isinstance(canonical_action, str) or not canonical_action.strip():
        errors.append("canonical_action must be a non-empty concrete action")
    else:
        normalized_action = re.sub(
            r"[^a-z0-9]+", "_", canonical_action.strip().lower()
        ).strip("_")
        if normalized_action in GENERIC_CANONICAL_ACTIONS:
            errors.append("canonical_action must not be a generic motion or emotion label")
        if set(normalized_action.split("_")) & set(KIMODO_EMOTION_IDS):
            errors.append("canonical_action must not encode an emotion label")
    if record.get("behavior_id") not in KIMODO_BEHAVIOR_IDS:
        errors.append("behavior_id is not in the 27-class Kimodo ontology")
    behavior_status = record.get("behavior_review_status")
    behavior_mask = record.get("behavior_supervision_mask")
    behavior_eligibility = record.get("behavior_training_eligibility")
    if behavior_status not in BEHAVIOR_REVIEW_STATUSES:
        errors.append("behavior_review_status is invalid")
    if behavior_status == "human_confirmed":
        if behavior_mask is not True:
            errors.append("human-confirmed behavior must enable behavior_supervision_mask")
        if behavior_eligibility != "pending_adjudication":
            errors.append("human-confirmed behavior eligibility must remain pending")
    else:
        if behavior_mask is not False:
            errors.append("unconfirmed/rejected behavior_supervision_mask must be false")
        expected_behavior_eligibility = (
            "blocked_unreviewed_behavior"
            if behavior_status == "candidate_unreviewed"
            else "blocked_rejected_behavior"
        )
        if behavior_eligibility != expected_behavior_eligibility:
            errors.append("unconfirmed/rejected behavior eligibility must remain blocked")
    if record.get("motion_style") not in MOTION_STYLE_IDS:
        errors.append("motion_style is not trajectory-derived recognized style")
    if record.get("motion_style_training_eligibility") != "pending_adjudication":
        errors.append("motion/style eligibility must remain pending adjudication")
    errors.extend(validate_specific_bilingual_action(record.get("canonical_prompt")))

    status = record.get("emotion_review_status")
    emotion_id = record.get("emotion_id")
    source = record.get("emotion_source")
    confidence = record.get("emotion_confidence")
    mask = record.get("emotion_supervision_mask")
    affect = record.get("observed_affect")
    if status not in EMOTION_REVIEW_STATUSES:
        errors.append("emotion_review_status is invalid")
    if status == "human_confirmed":
        if emotion_id not in KIMODO_EMOTION_IDS:
            errors.append("human-confirmed emotion_id is invalid")
        if source != "human_visual_review":
            errors.append("human-confirmed emotion_source must be human_visual_review")
        if mask is not True:
            errors.append("human-confirmed emotion must enable emotion_supervision_mask")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0.0 < float(confidence) <= 1.0
        ):
            errors.append("human-confirmed emotion_confidence must be in (0, 1]")
        prompt = record.get("canonical_prompt")
        en = prompt.get("en", "") if isinstance(prompt, dict) else ""
        if emotion_id in KIMODO_EMOTION_IDS and not _contains_exact_emotion_word(en, emotion_id):
            errors.append("canonical_prompt.en lacks the exact confirmed emotion word")
        if record.get("behavior_review_status") != "human_confirmed":
            errors.append("emotion-confirmed record also requires human-confirmed behavior")
        if record.get("network_semantic_supervision_ready") is not True:
            errors.append("confirmed record must be marked network_semantic_supervision_ready")
        if record.get("emotion_training_eligibility") != "pending_adjudication":
            errors.append("confirmed emotion eligibility must remain pending adjudication")
        if not isinstance(affect, dict) or affect.get("label") != emotion_id:
            errors.append("observed_affect must match the confirmed emotion_id")
    else:
        if emotion_id is not None:
            errors.append("unconfirmed/rejected emotion_id must be null")
        if mask is not False:
            errors.append("unconfirmed/rejected emotion_supervision_mask must be false")
        if confidence != 0.0:
            errors.append("unconfirmed/rejected emotion_confidence must be 0.0")
        if record.get("network_semantic_supervision_ready") is not False:
            errors.append("unconfirmed/rejected record cannot be supervision-ready")
        if not isinstance(affect, dict) or affect.get("label") is not None:
            errors.append("unconfirmed/rejected observed_affect.label must be null")
        if status == "unresolved" and source != UNRESOLVED_EMOTION_SOURCE:
            errors.append("unresolved emotion_source must document absent BEAT2 ground truth")
        expected_eligibility = (
            "blocked_unresolved_emotion"
            if status == "unresolved"
            else "blocked_rejected_emotion"
        )
        if record.get("emotion_training_eligibility") != expected_eligibility:
            errors.append("unconfirmed/rejected emotion eligibility must remain blocked")
        if (
            status == "unresolved"
            and behavior_status != "human_confirmed"
            and record.get("behavior_id") not in AUTO_BEHAVIOR_IDS
        ):
            errors.append(
                "automatic unresolved behavior must be a conservative conversation candidate"
            )
    return [f"{clip_id}: {message}" for message in errors]


def review_queue_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        **record,
        "review_contract": {
            "allowed_behavior_ids": list(KIMODO_BEHAVIOR_IDS),
            "allowed_emotion_ids": list(KIMODO_EMOTION_IDS),
            "behavior_policy": (
                "The trajectory rule emits only an unreviewed candidate. A human must confirm "
                "the exact behavior_id before it can be used as supervised one-hot conditioning. "
                "Use decision=behavior_confirmed when affect remains unresolved."
            ),
            "emotion_policy": (
                "Use visible performed affect only. Do not infer emotion from transcript, "
                "audio filename, speaker, or session identifier. Reject or leave unresolved "
                "when the six-class affect is not visually defensible."
            ),
            "prompt_policy": (
                "Both languages must name concrete robot-visible motion. The English prompt "
                "must contain the exact confirmed emotion_id word."
            ),
            "review_template": {
                "clip_id": record["clip_id"],
                "decision": "confirmed_behavior_confirmed_or_rejected",
                "reviewer_id": "",
                "reviewer_kind": "human",
                "reviewed_at": "",
                "behavior_id": record["behavior_id"],
                "emotion_id": None,
                "emotion_confidence": None,
                "canonical_prompt": record["canonical_prompt"],
                "notes": "",
            },
        },
    }


def build_semantics(
    annotations: list[dict[str, Any]],
    reviews: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    review_index = {_clip_id(review): review for review in (reviews or [])}
    annotation_ids = {_clip_id(annotation) for annotation in annotations}
    unknown_reviews = sorted(set(review_index) - annotation_ids)
    if unknown_reviews:
        raise ValueError(f"human reviews reference unknown clip_ids: {unknown_reviews[:5]}")

    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for annotation in sorted(annotations, key=_clip_id):
        automatic = build_automatic_record(annotation)
        record = (
            apply_human_review(automatic, review_index[automatic["clip_id"]])
            if automatic["clip_id"] in review_index
            else automatic
        )
        errors.extend(validate_network_record(record))
        records.append(record)
    if errors:
        raise ValueError("semantic contract validation failed:\n" + "\n".join(errors))

    queue = [
        review_queue_record(record)
        for record in records
        if record["emotion_review_status"] == "unresolved"
    ]
    status_counts = Counter(record["emotion_review_status"] for record in records)
    behavior_counts = Counter(record["behavior_id"] for record in records)
    style_counts = Counter(record["motion_style"] for record in records)
    canonical_actions = {record["canonical_action"] for record in records}
    emotion_eligibility_counts = Counter(
        record["emotion_training_eligibility"] for record in records
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "input_records": len(annotations),
        "output_records": len(records),
        "human_reviews_applied": len(review_index),
        "emotion_review_status_counts": dict(sorted(status_counts.items())),
        "emotion_supervised_records": sum(
            record["emotion_supervision_mask"] is True for record in records
        ),
        "behavior_supervised_records": sum(
            record["behavior_supervision_mask"] is True for record in records
        ),
        "human_review_queue_records": len(queue),
        "behavior_counts": dict(sorted(behavior_counts.items())),
        "canonical_action_unique": len(canonical_actions),
        "motion_style_counts": dict(sorted(style_counts.items())),
        "motion_style_pending_adjudication_records": sum(
            record["motion_style_training_eligibility"] == "pending_adjudication"
            for record in records
        ),
        "emotion_training_eligibility_counts": dict(
            sorted(emotion_eligibility_counts.items())
        ),
        "validation_errors": 0,
        "transcript_or_audio_metadata_used_for_labels": False,
    }
    return records, queue, summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    annotations = read_jsonl(args.input_annotations)
    reviews = read_jsonl(args.human_reviews) if args.human_reviews else None
    records, queue, summary = build_semantics(annotations, reviews)
    output_dir = args.output_dir.resolve()
    write_jsonl(output_dir / "network_semantics.jsonl", records)
    write_jsonl(output_dir / "human_review_queue.jsonl", queue)
    write_jsonl(
        output_dir / "emotion_supervised.jsonl",
        (record for record in records if record["emotion_supervision_mask"] is True),
    )
    write_jsonl(
        output_dir / "rejected.jsonl",
        (
            record
            for record in records
            if record["emotion_review_status"] == "rejected"
        ),
    )
    _atomic_write(
        output_dir / "summary.json",
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
