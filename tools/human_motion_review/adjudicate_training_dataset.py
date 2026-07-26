#!/usr/bin/env python3
"""Deterministically adjudicate semantic, 18D QC, and independent review data."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


ADJUDICATION_SCHEMA_VERSION = "1.2.0"
EXPECTED_18D_JOINT_ORDER = [
    "joint_pelvisYaw",
    "joint_pelvisPitch",
    "joint_pelvisRoll",
    "joint_lShoulderPitch",
    "joint_lShoulderRoll",
    "joint_lShoulderYaw",
    "joint_lElbow",
    "joint_lWristRoll",
    "joint_lWristPitch",
    "joint_rShoulderPitch",
    "joint_rShoulderRoll",
    "joint_rShoulderYaw",
    "joint_rElbow",
    "joint_rWristRoll",
    "joint_rWristPitch",
    "head_roll_joint",
    "head_pitch_joint",
    "head_yaw_joint",
]
REQUIRED_18D_GATES = [
    "joint_limits_pass",
    "velocity_pass",
    "target_fit_pass",
    "collision_pass",
    "axis_direction_pass",
    "head_joint_limits_pass",
    "head_velocity_pass",
    "head_direction_pass",
    "head_continuity_pass",
    "passed",
]
REQUIRED_REVIEW_GATES = {
    "action_recognizable",
    "text_consistent",
    "observable_in_18d",
    "context_available",
    "physical_qc",
    "subject_action_split_safe",
    "affect_observable_in_18d",
}
MOTION_ADMISSION_REVIEW_GATES = REQUIRED_REVIEW_GATES - {
    "affect_observable_in_18d"
}
ALLOWED_REVIEW_STATUSES = {"agent_reviewed", "needs_human", "rejected"}
OUTPUT_FILENAMES = {
    "train_ready": "train_ready.jsonl",
    "needs_human": "needs_human.jsonl",
    "rejected": "rejected.jsonl",
}
BEAT_WINDOW_ID_PATTERN = re.compile(
    r"^(?P<source_clip_id>.+)_f(?P<start>\d+)-(?P<end>\d+)$"
)
BEAT_WINDOW_CONVENTION = "zero_based_half_open_[start,end)"
PROJECT_BEHAVIOR_MAPPING_SOURCE = "project_dataset_scope_weak_mapping_v1"
OFFICIAL_EMOTION_SOURCE = "official_beat2_filename_protocol"
OFFICIAL_EMOTION_STATUS = "official_protocol_confirmed"
OFFICIAL_EMOTION_DISABLED_ROLE = "disabled_pending_robot_affect_review"
OFFICIAL_EMOTION_ENABLED_ROLE = "enabled_verified_robot_affect_observable_in_18d"
OFFICIAL_EMOTION_CONDITION_CHANNEL = (
    "kimodo_emotion_one_hot_and_legacy_affect"
)
BLIND_AFFECT_PROTOCOL_VERSION = "robot_affect_blind_video_v1"
BLIND_AFFECT_REQUIRED_BLINDED_FIELDS = {
    "audio",
    "canonical_prompt",
    "official_emotion_label",
    "official_gesture_category",
    "source_text",
}
ALLOWED_EMOTION_IDS = {"neutral", "sad", "happy", "angry", "surprise", "fear"}
MIN_BLIND_AFFECT_CONFIDENCE = 0.7
FORMAL_SEMANTIC_SUPERVISION_MASKS = {
    "official_category": False,
    "robot_observable_motion_form": False,
    "communicative_intent": False,
    "prompt_text": False,
    "legacy_gesture": False,
}
OFFICIAL_CATEGORY_ROLE = "verified_metadata_split_and_evaluation_only"
VARIABLE_SEGMENT_REPRESENTATION = "native_variable_length_semantic_clip_v1"
RETARGET_SEGMENT_REPRESENTATION = (
    "native_variable_length_semantic_event_retimed_30hz_v1"
)
OFFICIAL_SELECTED_LINEAGE_FIELDS = (
    "upstream_inventory_record_sha256",
    "selected_record_sha256",
    "inventory_manifest_sha256",
    "pilot_selector_contract_sha256",
    "pilot_speaker_group_sha256",
    "pilot_source_group_sha256",
    "prompt_sha256",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_contract_valid(contract: object) -> bool:
    if not isinstance(contract, dict):
        return False
    recorded = contract.get("sha256")
    payload = {key: value for key, value in contract.items() if key != "sha256"}
    actual = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return recorded == actual


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _validate_blind_affect_review(sample: dict) -> None:
    """Validate evidence that was collected without exposing the target label."""

    clip_id = sample.get("sample_id")
    proof = sample.get("blind_affect_review")
    if not isinstance(proof, dict):
        raise ValueError(f"{clip_id}: affect-observable gate requires blind review evidence")
    if proof.get("protocol_version") != BLIND_AFFECT_PROTOCOL_VERSION:
        raise ValueError(f"{clip_id}: blind affect protocol version is invalid")
    if not str(proof.get("review_id") or "").strip():
        raise ValueError(f"{clip_id}: blind affect review_id is missing")
    if not str(proof.get("anonymous_video_id") or "").strip():
        raise ValueError(f"{clip_id}: anonymous blind-review video id is missing")
    if not _is_sha256(proof.get("video_sha256")):
        raise ValueError(f"{clip_id}: blind affect video SHA256 is invalid")
    if proof.get("target_emotion_exposed") is not False:
        raise ValueError(f"{clip_id}: blind affect review exposed the target emotion")
    if proof.get("audio_available") is not False:
        raise ValueError(f"{clip_id}: blind affect review must use silent video")
    blinded_to = proof.get("blinded_to")
    if not isinstance(blinded_to, list) or set(blinded_to) != (
        BLIND_AFFECT_REQUIRED_BLINDED_FIELDS
    ):
        raise ValueError(f"{clip_id}: blind affect review did not hide all target fields")
    reviewer = proof.get("reviewer")
    if not isinstance(reviewer, dict) or (
        reviewer.get("kind") != "agent"
        or reviewer.get("independent_of_annotation_logic") is not True
        or not str(reviewer.get("reviewer_id") or "").strip()
    ):
        raise ValueError(f"{clip_id}: blind affect reviewer provenance is invalid")
    observed = proof.get("observed_affect")
    if not isinstance(observed, dict) or observed.get("status") != "label":
        raise ValueError(f"{clip_id}: blind affect review must provide an observed label")
    if observed.get("emotion_id") not in ALLOWED_EMOTION_IDS:
        raise ValueError(f"{clip_id}: blind affect label is outside the network ontology")
    confidence = observed.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
        or float(confidence) < MIN_BLIND_AFFECT_CONFIDENCE
    ):
        raise ValueError(f"{clip_id}: blind affect confidence is below the formal threshold")
    evidence = sample.get("evidence") or {}
    preview = evidence.get("preview") if isinstance(evidence, dict) else None
    if not isinstance(preview, dict) or preview.get("sha256") != proof.get("video_sha256"):
        raise ValueError(f"{clip_id}: blind affect proof is not bound to the reviewed video")


def _project_weak_behavior(record: dict) -> bool:
    contract = record.get("behavior_mapping_contract")
    return bool(
        record.get("behavior_id") == "Behavior.InteractPresence"
        and record.get("behavior_review_status") == "candidate_unreviewed"
        and record.get("behavior_supervision_mask") is False
        and record.get("behavior_source") == PROJECT_BEHAVIOR_MAPPING_SOURCE
        and isinstance(contract, dict)
        and contract.get("source") == PROJECT_BEHAVIOR_MAPPING_SOURCE
        and contract.get("behavior_id") == "Behavior.InteractPresence"
        and contract.get("supervision") == "weak_candidate_masked"
        and _content_contract_valid(contract)
    )


def _official_source_emotion_label_verified(record: dict) -> bool:
    contract = record.get("emotion_protocol_contract")
    return bool(
        record.get("emotion_id")
        and record.get("emotion_review_status") == OFFICIAL_EMOTION_STATUS
        and record.get("source_emotion_label_verified") is True
        and record.get("emotion_source") == OFFICIAL_EMOTION_SOURCE
        and isinstance(contract, dict)
        and contract.get("source") == OFFICIAL_EMOTION_SOURCE
        and contract.get("emotion_id") == record.get("emotion_id")
        and contract.get("source_sha256") == record.get("source_sha256")
        and _content_contract_valid(contract)
    )


def _blind_affect_matches_source(record: dict, review: dict | None) -> bool:
    if not review or review.get("status") != "agent_reviewed":
        return False
    gates = review.get("gates") or {}
    if gates.get("affect_observable_in_18d") is not True:
        return False
    proof = review.get("blind_affect_review") or {}
    observed = proof.get("observed_affect") or {}
    return bool(
        proof.get("protocol_version") == BLIND_AFFECT_PROTOCOL_VERSION
        and proof.get("target_emotion_exposed") is False
        and proof.get("audio_available") is False
        and observed.get("status") == "label"
        and observed.get("emotion_id") == record.get("emotion_id")
        and float(observed.get("confidence", -1.0)) >= MIN_BLIND_AFFECT_CONFIDENCE
    )


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load JSON {path}: {error}") from error


def load_semantics(path: Path, expected_count: int | None = None) -> list[dict]:
    records = []
    seen = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot load semantics JSONL {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid semantics JSON at {path}:{line_number}: {error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"semantics record at {path}:{line_number} must be an object")
        clip_id = record.get("clip_id")
        if not isinstance(clip_id, str) or not clip_id:
            raise ValueError(f"semantics record at {path}:{line_number} has no clip_id")
        if clip_id in seen:
            raise ValueError(f"duplicate semantics clip_id: {clip_id}")
        seen.add(clip_id)
        records.append(record)
    if expected_count is not None and len(records) != expected_count:
        raise ValueError(f"expected {expected_count} semantic records, found {len(records)}")
    return sorted(records, key=lambda item: item["clip_id"])


def load_independent_reviews(path: Path) -> tuple[str, dict[str, dict]]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError("review bundle must be an object")
    reviewer = payload.get("reviewer") or {}
    if reviewer.get("kind") != "agent" or reviewer.get("independent_of_annotation_logic") is not True:
        raise ValueError("review bundle must come from an independent agent")
    review_id = payload.get("review_id")
    if not isinstance(review_id, str) or not review_id:
        raise ValueError("review bundle must have a review_id")
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError("review bundle samples must be a list")

    by_clip = {}
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("every review sample must be an object")
        clip_id = sample.get("sample_id")
        if not isinstance(clip_id, str) or not clip_id:
            raise ValueError("every review sample must have a sample_id")
        if clip_id in by_clip:
            raise ValueError(f"duplicate review sample_id: {clip_id}")
        status = sample.get("status")
        if status not in ALLOWED_REVIEW_STATUSES:
            raise ValueError(f"{clip_id}: unsupported independent review status {status!r}")
        accepted = sample.get("training_acceptance")
        if not isinstance(accepted, bool):
            raise ValueError(f"{clip_id}: training_acceptance must be boolean")
        gates = sample.get("gates")
        if not isinstance(gates, dict) or set(gates) != REQUIRED_REVIEW_GATES:
            raise ValueError(f"{clip_id}: independent review gates are incomplete")
        if not all(isinstance(value, bool) for value in gates.values()):
            raise ValueError(f"{clip_id}: every independent review gate must be boolean")
        if accepted and (
            status != "agent_reviewed"
            or not all(gates[gate] for gate in MOTION_ADMISSION_REVIEW_GATES)
        ):
            raise ValueError(
                f"{clip_id}: accepted review does not pass all motion-admission gates"
            )
        if gates["affect_observable_in_18d"] is True:
            _validate_blind_affect_review(sample)
        if status in {"needs_human", "rejected"} and accepted:
            raise ValueError(f"{clip_id}: unresolved or rejected review cannot be accepted")
        split = sample.get("split") or {}
        if accepted and split.get("assignment") != "train":
            raise ValueError(f"{clip_id}: accepted review must be assigned to train")
        if split.get("subject_key") is None:
            if accepted and split.get("subject_policy") != "train_only_unknown":
                raise ValueError(f"{clip_id}: unknown subject must use train_only_unknown")
            if split.get("eval_eligible") is not False:
                raise ValueError(f"{clip_id}: unknown subject cannot be evaluation eligible")
        by_clip[clip_id] = sample
    return review_id, by_clip


def semantic_critical_reasons(record: dict, critical_flags: set[str]) -> list[str]:
    reasons = []
    if not record.get("canonical_action"):
        reasons.append("semantic_missing_canonical_action")
    prompt = record.get("canonical_prompt") or {}
    if not isinstance(prompt, dict) or not prompt.get("en"):
        reasons.append("semantic_missing_canonical_prompt_en")

    semantic_qc = record.get("semantic_qc") or {}
    if record.get("semantic_critical") is True or semantic_qc.get("critical") is True:
        reasons.append("semantic_explicit_critical")
    if semantic_qc.get("severity") == "critical":
        reasons.append("semantic_severity_critical")

    source_quality = record.get("source_text_quality") or {}
    flags = source_quality.get("flags") or []
    if source_quality.get("recommended_use") == "reject":
        reasons.append("semantic_recommended_use_reject")
        reasons.extend(f"semantic_flag:{flag}" for flag in flags)
    reasons.extend(f"semantic_critical_flag:{flag}" for flag in flags if flag in critical_flags)
    return sorted(set(reasons))


def semantic_pending_reasons(record: dict) -> list[str]:
    """Return resolvable supervision blockers that must not be rejected."""

    if record.get("behavior_id") is None:
        return []
    if _project_weak_behavior(record):
        return []
    if (
        record.get("behavior_review_status") != "human_confirmed"
        or record.get("behavior_supervision_mask") is not True
    ):
        return ["semantic_behavior_confirmation_missing"]
    return []


def _inspect_safe_csv(
    path: Path,
    *,
    expected_header: list[str],
    expected_rows: int | None,
) -> tuple[list[str], int | None]:
    if not path.is_file():
        return ["qc_safe_csv_missing"], None
    reasons = []
    row_count = 0
    row_width_invalid = False
    non_numeric = False
    non_finite = False
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header != expected_header:
                reasons.append("qc_safe_csv_header_mismatch")
            for row in reader:
                row_count += 1
                if len(row) != len(expected_header):
                    row_width_invalid = True
                    continue
                for value in row:
                    try:
                        numeric = float(value)
                    except ValueError:
                        non_numeric = True
                        continue
                    if not math.isfinite(numeric):
                        non_finite = True
    except (OSError, UnicodeError, csv.Error):
        return ["qc_safe_csv_unreadable"], None
    if row_width_invalid:
        reasons.append("qc_safe_csv_row_width_mismatch")
    if non_numeric:
        reasons.append("qc_safe_csv_non_numeric")
    if non_finite:
        reasons.append("qc_safe_csv_non_finite")
    if expected_rows is not None and row_count != expected_rows:
        reasons.append("qc_safe_csv_row_count_mismatch")
    return sorted(set(reasons)), row_count


def _declared_path_mismatch(value, actual: Path, *, relative_to: Path | None = None) -> bool:
    if not isinstance(value, str) or not value.strip():
        return True
    declared = Path(value)
    if not declared.is_absolute():
        if relative_to is None:
            return declared.name != actual.name
        declared = relative_to / declared
    return declared.resolve() != actual.resolve()


def _semantic_source_clip_id(semantic: dict, clip_id: str) -> tuple[str | None, list[str]]:
    reasons = []
    value = semantic.get("source_clip_id")
    source_clip = value.strip() if isinstance(value, str) and value.strip() else None
    match = BEAT_WINDOW_ID_PATTERN.fullmatch(clip_id)
    parsed_source = match.group("source_clip_id") if match else None
    if source_clip is not None and parsed_source is not None and source_clip != parsed_source:
        reasons.append("qc_semantic_source_clip_mismatch")
    return source_clip or parsed_source, reasons


def _semantic_window(
    semantic: dict, clip_id: str
) -> tuple[int | None, int | None, list[str]]:
    reasons = []
    match = BEAT_WINDOW_ID_PATTERN.fullmatch(clip_id)
    parsed_start = int(match.group("start")) if match else None
    parsed_end = int(match.group("end")) if match else None
    start = semantic.get("source_window_start_frame")
    end = semantic.get("source_window_end_frame_exclusive")
    if start is None and end is None:
        start, end = parsed_start, parsed_end
    elif (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
    ):
        reasons.append("qc_semantic_window_provenance_invalid")
        return None, None, reasons
    elif match and (start != parsed_start or end != parsed_end):
        reasons.append("qc_semantic_window_provenance_mismatch")
    if start is None or end is None:
        reasons.append("qc_semantic_window_provenance_missing")
    elif start < 0 or end <= start:
        reasons.append("qc_semantic_window_provenance_invalid")
    return start, end, reasons


def _inspect_source_provenance(
    quality: dict, semantic: dict, clip_id: str
) -> tuple[list[str], dict]:
    reasons = []
    source_motionx = quality.get("source_motionx")
    source_beat2 = quality.get("source_beat2_npz")
    has_motionx = isinstance(source_motionx, str) and bool(source_motionx.strip())
    has_beat2 = isinstance(source_beat2, str) and bool(source_beat2.strip())
    if has_motionx == has_beat2:
        reasons.append(
            "qc_source_contract_ambiguous" if has_motionx else "qc_source_clip_mismatch"
        )
        return reasons, {
            "kind": None,
            "path": None,
            "source_clip_id": None,
            "window_start_frame": None,
            "window_end_frame_exclusive": None,
        }

    if has_motionx:
        declared_source = semantic.get("source_clip_id")
        expected_source = (
            declared_source.strip()
            if isinstance(declared_source, str) and declared_source.strip()
            else clip_id
        )
        if Path(source_motionx).stem != expected_source:
            reasons.append("qc_source_clip_mismatch")
        return reasons, {
            "kind": "motionx",
            "path": source_motionx,
            "source_clip_id": Path(source_motionx).stem,
            "window_start_frame": None,
            "window_end_frame_exclusive": None,
        }

    expected_source, source_reasons = _semantic_source_clip_id(semantic, clip_id)
    reasons.extend(source_reasons)
    if expected_source is None or Path(source_beat2).stem != expected_source:
        reasons.append("qc_source_clip_mismatch")
    if Path(source_beat2).suffix.lower() != ".npz":
        reasons.append("qc_source_beat2_format_mismatch")
    declared_source = semantic.get("source_beat2_npz")
    if declared_source is not None and _declared_path_mismatch(
        declared_source, Path(source_beat2)
    ):
        reasons.append("qc_source_beat2_path_mismatch")
    declared_source_hash = semantic.get("source_sha256")
    if declared_source_hash is not None and declared_source_hash != quality.get("source_sha256"):
        reasons.append("qc_source_sha256_mismatch")

    start, end, window_reasons = _semantic_window(semantic, clip_id)
    reasons.extend(window_reasons)
    actual_start = quality.get("source_window_start_frame")
    actual_end = quality.get("source_window_end_frame_exclusive")
    if isinstance(actual_start, bool) or not isinstance(actual_start, int):
        reasons.append("qc_source_window_start_invalid")
    elif start is not None and actual_start != start:
        reasons.append("qc_source_window_start_mismatch")
    if isinstance(actual_end, bool) or not isinstance(actual_end, int):
        reasons.append("qc_source_window_end_invalid")
    elif end is not None and actual_end != end:
        reasons.append("qc_source_window_end_mismatch")
    if quality.get("source_window_convention") != BEAT_WINDOW_CONVENTION:
        reasons.append("qc_source_window_convention_mismatch")
    if start is not None and end is not None:
        expected_source_frames = end - start
        source_window_frames = quality.get("source_window_frames")
        source_frames = quality.get("source_frames")
        if (
            isinstance(source_window_frames, bool)
            or not isinstance(source_window_frames, int)
            or source_window_frames != expected_source_frames
        ):
            reasons.append("qc_source_window_frames_mismatch")
        if (
            isinstance(source_frames, bool)
            or not isinstance(source_frames, int)
            or source_frames != expected_source_frames
        ):
            reasons.append("qc_source_frames_mismatch")
    return reasons, {
        "kind": "beat2",
        "path": source_beat2,
        "source_clip_id": Path(source_beat2).stem,
        "window_start_frame": actual_start,
        "window_end_frame_exclusive": actual_end,
    }


def _inspect_official_semantic_retarget(quality: dict, semantic: dict) -> list[str]:
    if semantic.get("annotation_kind") != "official_gesture_semantic_event":
        return []
    reasons = []
    event = semantic.get("semantic_event")
    category = event.get("category") if isinstance(event, dict) else None
    expected_semantic_fields = {
        "canonical_action": f"official_gesture_category:{category}",
        "canonical_action_role": "official_category_metadata_split_key_only",
        "semantic_mapping_status": "official_category_verified_metadata_only",
        "official_category_verified": True,
        "official_category_conditioning_enabled": False,
        "official_category_role": OFFICIAL_CATEGORY_ROLE,
        "official_category_condition_channel": None,
        "official_category_loss": None,
        "robot_observable_motion_form": "candidate_unreviewed",
        "communicative_intent": "candidate_unreviewed",
        "canonical_prompt_role": "coarse_category_only",
        "semantic_supervision_masks": FORMAL_SEMANTIC_SUPERVISION_MASKS,
        "source_emotion_label_verified": True,
        "emotion_supervision_mask": False,
        "official_emotion_conditioning_enabled": False,
        "emotion_supervision_role": OFFICIAL_EMOTION_DISABLED_ROLE,
        "official_emotion_condition_channel": None,
        "official_emotion_loss": None,
        "affect_observable_review_status": "candidate_unreviewed",
        "affect_observable_supervision_mask": False,
    }
    if category not in {"deictic", "iconic", "metaphoric"}:
        reasons.append("qc_semantic_official_category_invalid")
    for field, expected_value in expected_semantic_fields.items():
        if field not in semantic or semantic.get(field) != expected_value:
            reasons.append(f"qc_semantic_{field}_mismatch")
        if field not in quality or quality.get(field) != expected_value:
            reasons.append(f"qc_retarget_semantic_{field}_mismatch")
    for field in (
        "semantic_event",
        "emotion_id",
        "emotion_review_status",
        "emotion_source",
        "emotion_protocol_contract",
    ):
        if quality.get(field) != semantic.get(field):
            reasons.append(f"qc_retarget_semantic_{field}_mismatch")
    source_segment = semantic.get("training_segment")
    if not isinstance(source_segment, dict):
        return ["qc_retarget_segment_source_contract_missing"]
    if source_segment.get("representation") != VARIABLE_SEGMENT_REPRESENTATION:
        reasons.append("qc_retarget_segment_source_representation_mismatch")
    if source_segment.get("fixed_window_sec") is not None:
        reasons.append("qc_retarget_segment_fixed_window_forbidden")
    source_start = source_segment.get("start_frame")
    source_end = source_segment.get("end_frame_exclusive")
    source_frames = source_segment.get("frame_count")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (source_start, source_end, source_frames)
    ) or not (
        isinstance(source_start, int)
        and isinstance(source_end, int)
        and isinstance(source_frames, int)
        and source_start >= 0
        and source_end > source_start
        and source_frames == source_end - source_start
    ):
        return reasons + ["qc_retarget_segment_source_interval_invalid"]
    output_frames = quality.get("frames")
    fps = quality.get("fps")
    if (
        isinstance(output_frames, bool)
        or not isinstance(output_frames, int)
        or output_frames < 2
        or isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(float(fps))
        or float(fps) <= 0.0
    ):
        return reasons + ["qc_retarget_segment_output_interval_invalid"]
    retarget = quality.get("retarget_segment")
    if not isinstance(retarget, dict):
        return reasons + ["qc_retarget_segment_missing"]
    if not _content_contract_valid(retarget):
        reasons.append("qc_retarget_segment_hash_mismatch")
    expected = {
        "representation": RETARGET_SEGMENT_REPRESENTATION,
        "source_start_frame": source_start,
        "source_end_frame_exclusive": source_end,
        "source_frame_count": source_frames,
        "output_frame_count": output_frames,
        "fps": float(fps),
        "retimed": output_frames != source_frames,
        "cropped": False,
        "planner_duration_field": "output_sample_span_sec",
        "source_boundary_duration_field": "source_frame_coverage_sec",
        "legacy_quality_duration_sec_role": (
            "output_frame_coverage_compatibility_only_not_planner_target"
        ),
    }
    for field, expected_value in expected.items():
        if retarget.get(field) != expected_value:
            reasons.append(f"qc_retarget_segment_{field}_mismatch")
    expected_durations = {
        "source_frame_coverage_sec": source_frames / float(fps),
        "output_sample_span_sec": (output_frames - 1) / float(fps),
        "output_frame_coverage_sec": output_frames / float(fps),
    }
    for field, expected_value in expected_durations.items():
        value = retarget.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not math.isclose(
                float(value), expected_value, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            reasons.append(f"qc_retarget_segment_{field}_mismatch")
    if quality.get("source_window_frames") != source_frames:
        reasons.append("qc_retarget_segment_source_window_frames_mismatch")
    return reasons


def _inspect_official_selected_lineage(
    quality: dict, semantic: dict
) -> tuple[list[str], dict]:
    if semantic.get("annotation_kind") != "official_gesture_semantic_event":
        return [], {}
    reasons = []
    lineage = {}
    for field in OFFICIAL_SELECTED_LINEAGE_FIELDS:
        semantic_value = semantic.get(field)
        quality_value = quality.get(field)
        lineage[field] = quality_value
        if not (
            isinstance(semantic_value, str)
            and len(semantic_value) == 64
            and all(character in "0123456789abcdef" for character in semantic_value)
        ):
            reasons.append(f"qc_selected_lineage_{field}_missing")
        if quality_value != semantic_value:
            reasons.append(f"qc_selected_lineage_{field}_mismatch")
    if semantic.get("inventory_record_sha256") not in (
        None,
        semantic.get("upstream_inventory_record_sha256"),
    ):
        reasons.append("qc_selected_lineage_ambiguous_inventory_record_sha256")
    return reasons, lineage


def index_quality_evidence(
    quality_root: Path,
    rejected_quality_root: Path | None = None,
    clip_ids: set[str] | None = None,
) -> dict[str, dict]:
    roots = [("accepted", quality_root)]
    if rejected_quality_root is not None:
        roots.append(("rejected", rejected_quality_root))
    index = {}
    seen_paths = set()
    for partition, root in roots:
        if not root.is_dir():
            continue
        for quality_path in sorted(root.rglob("quality.json")):
            resolved = quality_path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            clip_id = quality_path.parent.name
            if clip_ids is not None and clip_id not in clip_ids:
                continue
            if clip_id in index:
                previous = index[clip_id]["quality_path"]
                raise ValueError(
                    f"duplicate 18D quality evidence for {clip_id}: {previous} and {quality_path}"
                )
            index[clip_id] = {
                "sample_dir": quality_path.parent.resolve(),
                "quality_path": resolved,
                "partition": partition,
                "root": root.resolve(),
            }
    return index


def inspect_18d_quality(
    quality_root: Path,
    clip_id: str,
    *,
    evidence: dict | None,
    expected_contract: str,
    expected_action_dim: int,
    expected_fps: float,
    semantic: dict | None = None,
) -> dict:
    sample_dir = evidence["sample_dir"] if evidence is not None else quality_root / clip_id
    quality_path = evidence["quality_path"] if evidence is not None else sample_dir / "quality.json"
    partition = evidence["partition"] if evidence is not None else None
    safe_csv = sample_dir / f"{clip_id}_gmr_safe_18d.csv"
    raw_csv = sample_dir / f"{clip_id}_gmr_raw_18d.csv"
    base = {
        "state": "missing",
        "partition": partition,
        "quality_json": str(quality_path),
        "safe_csv": str(safe_csv),
        "raw_csv": str(raw_csv) if raw_csv.is_file() else None,
        "quality_sha256": None,
        "safe_csv_sha256": None,
        "raw_csv_sha256": sha256_file(raw_csv) if raw_csv.is_file() else None,
        "reasons": ["qc_18d_missing"],
        "output_contract": None,
        "action_dim": None,
        "frames": None,
        "fps": None,
        "sample_span_sec": None,
        "frame_coverage_sec": None,
        "duration_time_axis": "sample_span=(frame_count-1)/fps",
        "quality_gate": None,
        "csv_rows": None,
        "source_provenance": None,
        "source_window_start_frame": None,
        "source_window_end_frame_exclusive": None,
        "source_window_frames": None,
        "source_frames": None,
        "retarget_segment": None,
        "upstream_lineage": None,
    }
    if not quality_path.is_file():
        return base

    base["quality_sha256"] = sha256_file(quality_path)
    try:
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        base.update(state="failed", reasons=["qc_quality_json_invalid"])
        return base
    if not isinstance(quality, dict):
        base.update(state="failed", reasons=["qc_quality_json_not_object"])
        return base

    reasons = []
    output_contract = quality.get("output_contract")
    action_dim = quality.get("action_dim")
    frames = quality.get("frames")
    fps = quality.get("fps")
    joint_order = quality.get("joint_order")
    gates = quality.get("quality_gate")
    semantic = semantic or {"clip_id": clip_id}

    if output_contract != expected_contract:
        reasons.append("qc_output_contract_mismatch")
    if action_dim != expected_action_dim:
        reasons.append("qc_action_dim_mismatch")
    if joint_order != EXPECTED_18D_JOINT_ORDER:
        reasons.append("qc_joint_order_mismatch")
    if not isinstance(frames, int) or isinstance(frames, bool) or frames <= 0:
        reasons.append("qc_frames_invalid")
        expected_rows = None
    else:
        expected_rows = frames
    if not isinstance(fps, (int, float)) or isinstance(fps, bool) or fps <= 0:
        reasons.append("qc_fps_invalid")
    elif not math.isclose(float(fps), expected_fps, rel_tol=0.0, abs_tol=1e-9):
        reasons.append("qc_fps_mismatch")
    source_reasons, source_provenance = _inspect_source_provenance(
        quality, semantic, clip_id
    )
    reasons.extend(source_reasons)
    reasons.extend(_inspect_official_semantic_retarget(quality, semantic))
    lineage_reasons, upstream_lineage = _inspect_official_selected_lineage(
        quality, semantic
    )
    reasons.extend(lineage_reasons)

    if semantic.get("quality_json") is not None and _declared_path_mismatch(
        semantic.get("quality_json"), quality_path
    ):
        reasons.append("qc_quality_json_path_mismatch")
    if (
        semantic.get("quality_json_sha256") is not None
        and semantic.get("quality_json_sha256") != base["quality_sha256"]
    ):
        reasons.append("qc_quality_json_sha256_mismatch")

    if not isinstance(gates, dict):
        reasons.append("qc_quality_gate_missing")
        gates = {}
    for gate in REQUIRED_18D_GATES:
        if gate not in gates:
            reasons.append(f"qc_gate_missing:{gate}")
        elif gates[gate] is not True:
            reasons.append(f"qc_gate_failed:{gate}")
    for gate, value in gates.items():
        if isinstance(value, bool) and value is False:
            reasons.append(f"qc_gate_failed:{gate}")

    csv_reasons, csv_rows = _inspect_safe_csv(
        safe_csv,
        expected_header=EXPECTED_18D_JOINT_ORDER,
        expected_rows=expected_rows,
    )
    reasons.extend(csv_reasons)
    if semantic.get("trajectory_path") is not None and _declared_path_mismatch(
        semantic.get("trajectory_path"), safe_csv
    ):
        reasons.append("qc_safe_csv_path_mismatch")
    safe_csv_hash = sha256_file(safe_csv) if safe_csv.is_file() else None
    if (
        semantic.get("trajectory_sha256") is not None
        and semantic.get("trajectory_sha256") != safe_csv_hash
    ):
        reasons.append("qc_safe_csv_sha256_mismatch")
    outputs = quality.get("outputs")
    output_safe_csv = outputs.get("safe_csv") if isinstance(outputs, dict) else None
    if _declared_path_mismatch(
        output_safe_csv, safe_csv, relative_to=quality_path.parent
    ):
        reasons.append("qc_outputs_safe_csv_mismatch")
    if partition == "rejected" and not reasons:
        reasons.append("qc_rejected_partition_conflict")
    base.update(
        state="failed" if reasons else "passed",
        reasons=sorted(set(reasons)),
        output_contract=output_contract,
        action_dim=action_dim,
        frames=frames,
        fps=fps,
        sample_span_sec=(
            float(max(0, frames - 1) / float(fps))
            if isinstance(frames, int)
            and not isinstance(frames, bool)
            and frames > 0
            and isinstance(fps, (int, float))
            and not isinstance(fps, bool)
            and float(fps) > 0.0
            else None
        ),
        frame_coverage_sec=(
            float(frames / float(fps))
            if isinstance(frames, int)
            and not isinstance(frames, bool)
            and frames > 0
            and isinstance(fps, (int, float))
            and not isinstance(fps, bool)
            and float(fps) > 0.0
            else None
        ),
        quality_gate=quality.get("quality_gate"),
        csv_rows=csv_rows,
        safe_csv_sha256=safe_csv_hash,
        source_provenance=source_provenance,
        source_window_start_frame=quality.get("source_window_start_frame"),
        source_window_end_frame_exclusive=quality.get(
            "source_window_end_frame_exclusive"
        ),
        source_window_frames=quality.get("source_window_frames"),
        source_frames=quality.get("source_frames"),
        retarget_segment=copy.deepcopy(quality.get("retarget_segment")),
        upstream_lineage=upstream_lineage or None,
    )
    return base


def _review_summary(review_id: str, review: dict | None) -> dict:
    if review is None:
        return {
            "review_id": review_id,
            "present": False,
            "status": None,
            "training_acceptance": False,
            "gates": None,
            "conflict_ids": [],
            "blind_affect_review": None,
        }
    return {
        "review_id": review_id,
        "present": True,
        "status": review["status"],
        "training_acceptance": review["training_acceptance"],
        "gates": copy.deepcopy(review["gates"]),
        "conflict_ids": sorted(review.get("conflict_ids") or []),
        "blind_affect_review": copy.deepcopy(review.get("blind_affect_review")),
    }


def _review_reasons(review: dict | None) -> list[str]:
    if review is None:
        return ["independent_review_missing"]
    reasons = []
    status = review["status"]
    if status == "rejected":
        reasons.append("independent_review_rejected")
    elif status == "needs_human":
        reasons.append("independent_review_needs_human")
    elif not review["training_acceptance"]:
        reasons.append("independent_review_not_accepted")
    for gate in MOTION_ADMISSION_REVIEW_GATES:
        passed = review["gates"][gate]
        if passed is False:
            reasons.append(f"independent_review_gate_failed:{gate}")
    return sorted(set(reasons))


def _build_split(record: dict, review: dict | None, status: str) -> dict:
    review_split = (review or {}).get("split") or {}
    source = record.get("source") or {}
    subject_key = (
        record.get("speaker_key")
        or source.get("subject_key")
        or review_split.get("subject_key")
    )
    source_group = (
        record.get("source_group_key")
        or record.get("source_clip_id")
        or source.get("source_group_key")
        or source.get("motion_relpath")
        or review_split.get("source_group_key")
        or record["clip_id"]
    )
    unknown_subject = subject_key is None
    return {
        "subject_key": subject_key,
        "subject_policy": "train_only_unknown" if unknown_subject else "subject_disjoint",
        "action_key": record.get("canonical_action"),
        "source_group_key": source_group,
        "assignment": "train" if status == "train_ready" else None,
        "eval_eligible": False,
        "restriction": "unknown_subject_train_only" if unknown_subject else "subject_disjoint_required",
    }


def _next_action(status: str, quality_state: str, review: dict | None) -> str:
    if status == "train_ready":
        return "use_for_training_only"
    if status == "rejected":
        return "exclude_from_training"
    if quality_state == "missing" and review is None:
        return "produce_18d_qc_then_independent_review"
    if quality_state == "missing":
        return "produce_18d_qc"
    return "complete_or_resolve_independent_review"


def _training_eligibility(
    semantic: dict, status: str, review: dict | None
) -> dict:
    motion_style_eligible = status == "train_ready"
    if motion_style_eligible:
        motion_style_status = "train_ready"
    elif status == "rejected":
        motion_style_status = "rejected"
    else:
        motion_style_status = "pending_adjudication"

    official_emotion = _official_source_emotion_label_verified(semantic)
    emotion_semantic_ready = bool(
        official_emotion
        or (
            semantic.get("emotion_supervision_mask") is True
            and semantic.get("emotion_id")
            and semantic.get("emotion_review_status") == "human_confirmed"
            and semantic.get("network_semantic_supervision_ready") is True
        )
    )
    affect_observable_verified = _blind_affect_matches_source(semantic, review)
    emotion_eligible = bool(
        motion_style_eligible
        and emotion_semantic_ready
        and affect_observable_verified
    )
    weak_behavior = _project_weak_behavior(semantic)
    behavior_declared = semantic.get("behavior_id") is not None
    behavior_semantic_ready = bool(
        behavior_declared
        and semantic.get("behavior_review_status") == "human_confirmed"
        and semantic.get("behavior_supervision_mask") is True
    )
    behavior_eligible = motion_style_eligible and behavior_semantic_ready
    if not behavior_declared:
        behavior_status = "not_applicable"
    elif behavior_eligible:
        behavior_status = "train_ready"
    elif weak_behavior:
        behavior_status = "masked_project_weak_candidate"
    elif behavior_declared and not behavior_semantic_ready:
        behavior_status = "blocked_unconfirmed_behavior"
    elif status == "rejected":
        behavior_status = "rejected"
    else:
        behavior_status = "pending_adjudication"
    if emotion_eligible:
        emotion_status = "train_ready"
    elif semantic.get("emotion_review_status") == "unresolved":
        emotion_status = "blocked_unresolved_emotion"
    elif semantic.get("emotion_review_status") == "rejected":
        emotion_status = "blocked_rejected_emotion"
    elif not emotion_semantic_ready:
        emotion_status = "blocked_missing_confirmed_emotion"
    elif not affect_observable_verified:
        emotion_status = "masked_pending_verified_blind_robot_affect_review"
    elif status == "rejected":
        emotion_status = "rejected"
    else:
        emotion_status = "pending_adjudication"
    return {
        "behavior": {
            "eligible": behavior_eligible,
            "status": behavior_status,
            "one_hot_supervision_mask": behavior_eligible,
            "requires": (
                ["project_weak_mapping_provenance", "behavior_condition_channels_zero"]
                if weak_behavior
                else ["human_confirmed_behavior", "motion_style_train_ready"]
            ),
        },
        "motion_style": {
            "eligible": motion_style_eligible,
            "status": motion_style_status,
            "requires": ["passed_18d_qc", "independent_training_acceptance"],
        },
        "emotion": {
            "eligible": emotion_eligible,
            "status": emotion_status,
            "loss_mask": emotion_eligible,
            "source_label_verified": emotion_semantic_ready,
            "affect_observable_verified": affect_observable_verified,
            "conditioning_mask": emotion_eligible,
            "requires": (
                [
                    "motion_style_train_ready",
                    "verified_official_emotion_protocol",
                    "anonymous_silent_video_sha256_bound_blind_affect_review",
                    "target_emotion_not_exposed",
                    "observed_affect_matches_official_label",
                    "affect_observable_in_18d_gate_true",
                ]
                if official_emotion
                else [
                    "motion_style_train_ready",
                    "human_confirmed_emotion",
                    "verified_blind_robot_affect_review",
                ]
            ),
        },
    }


def adjudicate_record(
    semantic: dict,
    quality: dict,
    review_id: str,
    review: dict | None,
    *,
    critical_flags: set[str],
) -> dict:
    semantic_reasons = semantic_critical_reasons(semantic, critical_flags)
    semantic_pending = semantic_pending_reasons(semantic)
    review_reasons = _review_reasons(review)
    all_reasons = sorted(
        set(semantic_reasons + semantic_pending + quality["reasons"] + review_reasons)
    )

    rejection_causes = list(semantic_reasons)
    if quality["state"] == "failed":
        rejection_causes.extend(quality["reasons"])
    if review is not None and review["status"] == "rejected":
        rejection_causes.extend(review_reasons)
    rejection_causes = sorted(set(rejection_causes))

    review_accepted = bool(review and review["training_acceptance"])
    if rejection_causes:
        status = "rejected"
    elif review_accepted and quality["state"] == "passed" and not semantic_pending:
        status = "train_ready"
    else:
        status = "needs_human"

    merged = copy.deepcopy(semantic)
    training_eligibility = _training_eligibility(semantic, status, review)
    next_action = _next_action(status, quality["state"], review)
    if semantic_pending and status != "rejected":
        next_action = "obtain_explicit_human_behavior_confirmation"
    if status == "train_ready" and not training_eligibility["emotion"]["eligible"]:
        next_action = "use_for_motion_style_training_only_keep_emotion_loss_masked"
    merged["adjudication_schema_version"] = ADJUDICATION_SCHEMA_VERSION
    merged["adjudication"] = {
        "status": status,
        "reasons": all_reasons,
        "rejection_causes": rejection_causes,
        "semantic_critical": bool(semantic_reasons),
        "semantic_pending_reasons": semantic_pending,
        "next_action": next_action,
        "rule": (
            "motion/style train_ready requires independent training_acceptance=true and "
            "passed 18D QC plus explicit human behavior confirmation when behavior_id is "
            "declared; emotion additionally requires confirmed emotion supervision"
        ),
    }
    merged["training_eligibility"] = training_eligibility
    emotion_enabled = bool(training_eligibility["emotion"]["eligible"])
    source_emotion_verified = bool(
        training_eligibility["emotion"]["source_label_verified"]
    )
    merged.update(
        {
            "source_emotion_label_verified": source_emotion_verified,
            "emotion_supervision_mask": emotion_enabled,
            "affect_observable_review_status": (
                "verified" if emotion_enabled else "not_verified"
            ),
            "affect_observable_supervision_mask": emotion_enabled,
            "emotion_conditioning_mask": emotion_enabled,
        }
    )
    if semantic.get("annotation_kind") == "official_gesture_semantic_event":
        affect_verified = bool(emotion_enabled)
        merged.update(
            {
                "official_emotion_conditioning_enabled": emotion_enabled,
                "emotion_supervision_role": (
                    OFFICIAL_EMOTION_ENABLED_ROLE
                    if emotion_enabled
                    else OFFICIAL_EMOTION_DISABLED_ROLE
                ),
                "official_emotion_condition_channel": (
                    OFFICIAL_EMOTION_CONDITION_CHANNEL
                    if emotion_enabled
                    else None
                ),
                "official_emotion_loss": None,
            }
        )
    merged["motion_18d"] = copy.deepcopy(quality)
    merged["independent_review"] = _review_summary(review_id, review)
    merged["split"] = _build_split(semantic, review, status)
    return merged


def _validate_adjudicated(records: list[dict]) -> None:
    seen = set()
    for record in records:
        clip_id = record["clip_id"]
        if clip_id in seen:
            raise AssertionError(f"duplicate adjudicated clip_id: {clip_id}")
        seen.add(clip_id)
        status = record["adjudication"]["status"]
        review = record["independent_review"]
        quality = record["motion_18d"]
        split = record["split"]
        eligibility = record["training_eligibility"]
        if status == "train_ready":
            if not (
                review["present"]
                and review["status"] == "agent_reviewed"
                and review["training_acceptance"] is True
                and quality["state"] == "passed"
                and record["adjudication"]["semantic_critical"] is False
                and split["assignment"] == "train"
            ):
                raise AssertionError(f"{clip_id}: invalid train_ready admission")
            if eligibility["motion_style"]["eligible"] is not True:
                raise AssertionError(f"{clip_id}: train_ready motion/style was not eligible")
            supervised_behavior = bool(
                record.get("behavior_review_status") == "human_confirmed"
                and record.get("behavior_supervision_mask") is True
                and eligibility["behavior"]["eligible"] is True
                and eligibility["behavior"]["one_hot_supervision_mask"] is True
            )
            masked_weak_behavior = bool(
                _project_weak_behavior(record)
                and eligibility["behavior"]["eligible"] is False
                and eligibility["behavior"]["one_hot_supervision_mask"] is False
                and eligibility["behavior"]["status"]
                == "masked_project_weak_candidate"
            )
            if record.get("behavior_id") is not None and not (
                supervised_behavior or masked_weak_behavior
            ):
                raise AssertionError(f"{clip_id}: unconfirmed behavior entered train_ready")
        if status == "rejected" and not record["adjudication"]["rejection_causes"]:
            raise AssertionError(f"{clip_id}: rejected record has no rejection cause")
        if quality["state"] == "failed" and status != "rejected":
            raise AssertionError(f"{clip_id}: failed 18D QC was not rejected")
        if record["adjudication"]["semantic_critical"] and status != "rejected":
            raise AssertionError(f"{clip_id}: semantic-critical record was not rejected")
        if review["status"] == "rejected" and status != "rejected":
            raise AssertionError(f"{clip_id}: independently rejected record was not rejected")
        if split["subject_key"] is None and (
            split["assignment"] not in {None, "train"} or split["eval_eligible"] is not False
        ):
            raise AssertionError(f"{clip_id}: unknown subject escaped train-only policy")
        if eligibility["emotion"]["eligible"] and not (
            eligibility["motion_style"]["eligible"]
            and record.get("emotion_supervision_mask") is True
            and eligibility["emotion"]["loss_mask"] is True
            and eligibility["emotion"]["source_label_verified"] is True
            and eligibility["emotion"]["affect_observable_verified"] is True
            and eligibility["emotion"]["conditioning_mask"] is True
            and record.get("affect_observable_review_status") == "verified"
            and record.get("affect_observable_supervision_mask") is True
            and record.get("emotion_conditioning_mask") is True
        ):
            raise AssertionError(f"{clip_id}: invalid emotion training eligibility")
        if (
            record.get("annotation_kind") == "official_gesture_semantic_event"
            and not eligibility["emotion"]["eligible"]
            and any(
                (
                    record.get("emotion_supervision_mask") is True,
                    record.get("official_emotion_conditioning_enabled") is True,
                    record.get("affect_observable_supervision_mask") is True,
                    record.get("emotion_conditioning_mask") is True,
                )
            )
        ):
            raise AssertionError(
                f"{clip_id}: unverified official robot affect escaped a conditioning mask"
            )
        if record.get("emotion_review_status") == "unresolved" and (
            eligibility["emotion"]["eligible"]
            or eligibility["emotion"]["loss_mask"]
        ):
            raise AssertionError(f"{clip_id}: unresolved emotion escaped the loss mask")
        if eligibility["behavior"]["eligible"] and not (
            eligibility["motion_style"]["eligible"]
            and eligibility["behavior"]["one_hot_supervision_mask"]
        ):
            raise AssertionError(f"{clip_id}: invalid behavior training eligibility")


def _jsonl_bytes(records: list[dict]) -> bytes:
    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    ]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _counter_dict(values) -> dict:
    return dict(sorted(Counter(values).items()))


def _build_report(
    records: list[dict],
    *,
    semantics_path: Path,
    review_path: Path,
    quality_root: Path,
    rejected_quality_root: Path | None,
    quality_ids: set[str],
    output_dir: Path,
    expected_count: int | None,
    review_ids: set[str],
    output_metadata: dict,
) -> dict:
    by_action = defaultdict(Counter)
    reason_counts = Counter()
    for record in records:
        status = record["adjudication"]["status"]
        by_action[record.get("canonical_action") or "<missing>"][status] += 1
        reason_counts.update(record["adjudication"]["reasons"])
    semantic_ids = {record["clip_id"] for record in records}
    status_observed = Counter(record["adjudication"]["status"] for record in records)
    status_counts = {status: status_observed.get(status, 0) for status in OUTPUT_FILENAMES}
    quality_observed = Counter(record["motion_18d"]["state"] for record in records)
    quality_counts = {
        state: quality_observed.get(state, 0) for state in ("passed", "failed", "missing")
    }
    review_observed = Counter(
        record["independent_review"]["status"] or "missing" for record in records
    )
    review_counts = {
        state: review_observed.get(state, 0)
        for state in ("agent_reviewed", "needs_human", "rejected", "missing")
    }
    partition_counts = _counter_dict(
        record["motion_18d"]["partition"] or "missing" for record in records
    )
    train_ready = [record for record in records if record["adjudication"]["status"] == "train_ready"]
    motion_style_ready = [
        record
        for record in records
        if record["training_eligibility"]["motion_style"]["eligible"]
    ]
    emotion_ready = [
        record
        for record in records
        if record["training_eligibility"]["emotion"]["eligible"]
    ]
    behavior_ready = [
        record
        for record in records
        if record["training_eligibility"]["behavior"]["eligible"]
    ]
    passed_qc = [record for record in records if record["motion_18d"]["state"] == "passed"]
    train_ready_frames = sum(record["motion_18d"]["frames"] or 0 for record in train_ready)
    passed_frames = sum(record["motion_18d"]["frames"] or 0 for record in passed_qc)
    train_ready_duration = sum(
        max(0, (record["motion_18d"]["frames"] or 0) - 1)
        / record["motion_18d"]["fps"]
        for record in train_ready
        if record["motion_18d"]["fps"]
    )
    passed_duration = sum(
        max(0, (record["motion_18d"]["frames"] or 0) - 1)
        / record["motion_18d"]["fps"]
        for record in passed_qc
        if record["motion_18d"]["fps"]
    )
    train_ready_frame_coverage = sum(
        (record["motion_18d"]["frames"] or 0) / record["motion_18d"]["fps"]
        for record in train_ready
        if record["motion_18d"]["fps"]
    )
    passed_frame_coverage = sum(
        (record["motion_18d"]["frames"] or 0) / record["motion_18d"]["fps"]
        for record in passed_qc
        if record["motion_18d"]["fps"]
    )
    return {
        "schema_version": 1,
        "adjudication_schema_version": ADJUDICATION_SCHEMA_VERSION,
        "inputs": {
            "semantics": {
                "path": str(semantics_path),
                "sha256": sha256_file(semantics_path),
                "expected_records": expected_count,
                "records": len(records),
            },
            "independent_review": {
                "path": str(review_path),
                "sha256": sha256_file(review_path),
                "matched_records": sum(
                    record["independent_review"]["present"] for record in records
                ),
            },
            "quality_roots": {
                "accepted": str(quality_root),
                "rejected": str(rejected_quality_root) if rejected_quality_root else None,
            },
        },
        "counts": status_counts,
        "quality_states": quality_counts,
        "quality_partitions": partition_counts,
        "review_states": review_counts,
        "semantic_critical_count": sum(
            record["adjudication"]["semantic_critical"] for record in records
        ),
        "semantic_critical_policy": {
            "explicit_semantic_critical": True,
            "semantic_qc_severity_critical": True,
            "source_text_recommended_use_reject": True,
        },
        "reason_counts": dict(sorted(reason_counts.items())),
        "action_status_counts": {
            action: dict(sorted(counts.items())) for action, counts in sorted(by_action.items())
        },
        "coverage": {
            "quality_available_fraction": (
                quality_counts["passed"] + quality_counts["failed"]
            )
            / len(records)
            if records
            else 0.0,
            "quality_pass_fraction": quality_counts["passed"] / len(records) if records else 0.0,
            "review_fraction": sum(
                record["independent_review"]["present"] for record in records
            )
            / len(records)
            if records
            else 0.0,
            "train_ready_fraction": len(train_ready) / len(records) if records else 0.0,
            "quality_complete": quality_counts.get("missing", 0) == 0,
        },
        "scale": {
            "qc_passed_clips": len(passed_qc),
            "qc_passed_frames": passed_frames,
            "qc_passed_duration_sec": passed_duration,
            "qc_passed_duration_hours": passed_duration / 3600.0,
            "qc_passed_frame_coverage_sec": passed_frame_coverage,
            "train_ready_clips": len(train_ready),
            "train_ready_frames": train_ready_frames,
            "train_ready_duration_sec": train_ready_duration,
            "train_ready_duration_hours": train_ready_duration / 3600.0,
            "train_ready_frame_coverage_sec": train_ready_frame_coverage,
            "duration_time_axis": "sample_span=(frame_count-1)/fps",
            "frame_coverage_time_axis": "frame_coverage=frame_count/fps",
            "planner_duration_source": "sample_span_only",
            "train_ready_safe_csv_bytes": sum(
                Path(record["motion_18d"]["safe_csv"]).stat().st_size for record in train_ready
            ),
            "motion_style_train_ready_clips": len(motion_style_ready),
            "behavior_train_ready_clips": len(behavior_ready),
            "emotion_train_ready_clips": len(emotion_ready),
            "source_emotion_label_verified_clips": sum(
                record.get("source_emotion_label_verified") is True
                for record in records
            ),
            "blind_affect_observable_verified_clips": sum(
                record.get("affect_observable_supervision_mask") is True
                for record in records
            ),
            "official_category_conditioned_clips": 0,
        },
        "split_policy": {
            "unknown_subject": "train_only",
            "train_ready_unknown_subject_count": sum(
                record["split"]["subject_key"] is None for record in train_ready
            ),
            "evaluation_eligible_count": sum(record["split"]["eval_eligible"] for record in records),
        },
        "orphans": {
            "quality_scan_scope": "semantic_clip_ids_only",
            "quality_sample_ids": sorted(quality_ids - semantic_ids),
            "review_sample_ids": sorted(review_ids - semantic_ids),
        },
        "outputs": output_metadata,
        "invariants": {
            "every_semantic_record_adjudicated_once": len(records) == len(semantic_ids),
            "train_ready_requires_independent_acceptance_and_18d_qc": all(
                record["independent_review"]["training_acceptance"] is True
                and record["motion_18d"]["state"] == "passed"
                for record in train_ready
            ),
            "unknown_subject_never_eval": all(
                record["split"]["subject_key"] is not None
                or record["split"]["eval_eligible"] is False
                for record in records
            ),
            "failed_18d_qc_always_rejected": all(
                record["motion_18d"]["state"] != "failed"
                or record["adjudication"]["status"] == "rejected"
                for record in records
            ),
            "semantic_critical_always_rejected": all(
                not record["adjudication"]["semantic_critical"]
                or record["adjudication"]["status"] == "rejected"
                for record in records
            ),
            "independent_review_rejected_always_rejected": all(
                record["independent_review"]["status"] != "rejected"
                or record["adjudication"]["status"] == "rejected"
                for record in records
            ),
            "emotion_train_ready_requires_motion_and_explicit_supervision": all(
                record["training_eligibility"]["motion_style"]["eligible"]
                and record.get("emotion_supervision_mask") is True
                and record.get("source_emotion_label_verified") is True
                and record.get("affect_observable_supervision_mask") is True
                and record.get("emotion_conditioning_mask") is True
                for record in emotion_ready
            ),
            "unverified_affect_never_emotion_conditioned": all(
                record.get("affect_observable_supervision_mask") is True
                or (
                    record.get("emotion_supervision_mask") is not True
                    and record.get("official_emotion_conditioning_enabled") is not True
                    and record.get("emotion_conditioning_mask") is not True
                )
                for record in records
                if record.get("annotation_kind") == "official_gesture_semantic_event"
            ),
            "official_category_metadata_only_never_conditioned": all(
                record.get("official_category_conditioning_enabled") is False
                and record.get("official_category_condition_channel") is None
                and record.get("official_category_loss") is None
                for record in records
                if record.get("annotation_kind") == "official_gesture_semantic_event"
            ),
            "unresolved_emotion_never_train_ready": all(
                record.get("emotion_review_status") != "unresolved"
                or not record["training_eligibility"]["emotion"]["eligible"]
                for record in records
            ),
            "behavior_one_hot_requires_explicit_human_confirmation": all(
                record.get("behavior_review_status") == "human_confirmed"
                and record.get("behavior_supervision_mask") is True
                and record["training_eligibility"]["motion_style"]["eligible"]
                for record in behavior_ready
            ),
            "project_weak_behavior_never_one_hot_supervised": all(
                not _project_weak_behavior(record)
                or (
                    record["training_eligibility"]["behavior"]["eligible"] is False
                    and record["training_eligibility"]["behavior"][
                        "one_hot_supervision_mask"
                    ]
                    is False
                )
                for record in records
            ),
        },
        "output_dir": str(output_dir),
    }


def adjudicate_dataset(
    semantics_path: Path,
    quality_root: Path,
    review_path: Path,
    output_dir: Path,
    *,
    rejected_quality_root: Path | None = None,
    expected_count: int | None = None,
    expected_contract: str = "ula_v2_18d_head_v1",
    expected_action_dim: int = 18,
    expected_fps: float = 30.0,
    critical_flags: set[str] | None = None,
) -> dict:
    semantics_path = semantics_path.resolve()
    quality_root = quality_root.resolve()
    rejected_quality_root = rejected_quality_root.resolve() if rejected_quality_root else None
    review_path = review_path.resolve()
    output_dir = output_dir.resolve()
    semantics = load_semantics(semantics_path, expected_count)
    review_id, reviews = load_independent_reviews(review_path)
    critical_flags = set(critical_flags or ())
    semantic_ids = {record["clip_id"] for record in semantics}
    quality_index = index_quality_evidence(
        quality_root,
        rejected_quality_root,
        clip_ids=semantic_ids,
    )

    records = []
    for semantic in semantics:
        clip_id = semantic["clip_id"]
        quality = inspect_18d_quality(
            quality_root,
            clip_id,
            evidence=quality_index.get(clip_id),
            expected_contract=expected_contract,
            expected_action_dim=expected_action_dim,
            expected_fps=expected_fps,
            semantic=semantic,
        )
        records.append(
            adjudicate_record(
                semantic,
                quality,
                review_id,
                reviews.get(clip_id),
                critical_flags=critical_flags,
            )
        )
    _validate_adjudicated(records)

    output_metadata = {}
    for status, filename in OUTPUT_FILENAMES.items():
        path = output_dir / filename
        selected = [record for record in records if record["adjudication"]["status"] == status]
        _atomic_write(path, _jsonl_bytes(selected))
        output_metadata[status] = {
            "path": str(path),
            "records": len(selected),
            "sha256": sha256_file(path),
        }
    report = _build_report(
        records,
        semantics_path=semantics_path,
        review_path=review_path,
        quality_root=quality_root,
        rejected_quality_root=rejected_quality_root,
        quality_ids=set(quality_index),
        output_dir=output_dir,
        expected_count=expected_count,
        review_ids=set(reviews),
        output_metadata=output_metadata,
    )
    report_path = output_dir / "dataset_scale_report.json"
    report_bytes = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write(report_path, report_bytes)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantics", type=Path, required=True)
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument(
        "--rejected-quality-root",
        type=Path,
        help="Optional quarantine root containing failed 18D quality evidence.",
    )
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-count",
        type=int,
        default=100,
        help="Expected semantics count; use 0 to disable the count assertion.",
    )
    parser.add_argument("--expected-contract", default="ula_v2_18d_head_v1")
    parser.add_argument("--expected-action-dim", type=int, default=18)
    parser.add_argument("--expected-fps", type=float, default=30.0)
    parser.add_argument(
        "--semantic-critical-flag",
        action="append",
        default=[],
        help="Additional source_text_quality flag that forces rejection; repeat as needed.",
    )
    parser.add_argument(
        "--require-complete-qc",
        action="store_true",
        help="Write outputs but return code 2 while any 18D quality record is missing.",
    )
    args = parser.parse_args()
    report = adjudicate_dataset(
        args.semantics,
        args.quality_root,
        args.review,
        args.output_dir,
        rejected_quality_root=args.rejected_quality_root,
        expected_count=None if args.expected_count == 0 else args.expected_count,
        expected_contract=args.expected_contract,
        expected_action_dim=args.expected_action_dim,
        expected_fps=args.expected_fps,
        critical_flags=set(args.semantic_critical_flag),
    )
    print(
        json.dumps(
            {
                "counts": report["counts"],
                "quality_states": report["quality_states"],
                "quality_complete": report["coverage"]["quality_complete"],
                "output_dir": report["output_dir"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.require_complete_qc and not report["coverage"]["quality_complete"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
