"""Fail-closed BEAT2 emotion-supervision ingestion.

The module deliberately separates two kinds of targets:

* ``human_confirmed_observable`` targets come only from two independent blind
  reviewers that agree, or from a distinct third adjudicator after a
  disagreement;
* ``intended_metadata_weak`` targets come from the official BEAT2 filename
  protocol and always receive a lower loss weight.

A human decision applies only to the exact reviewed trajectory SHA256.  It is
never propagated to another window from the same source recording.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "1.0.0"
CONFIG_KIND = "beat2_emotion_supervision_ingest_config_v1"
RECORD_KIND = "beat2_emotion_supervision_record_v1"
AUDIT_KIND = "beat2_emotion_supervision_ingest_audit_v1"
CONTROLLER_KIND = "beat2_robot_observable_emotion_review_queue_record_v1"
MAPPING_KIND = "beat2_emotion_blind_review_hidden_mapping_v1"
PRIMARY_KIND = "beat2_blind_robot_affect_primary_review_task_v1"
ADJUDICATION_KIND = "beat2_blind_robot_affect_adjudication_task_v1"
PROTOCOL = "two_independent_blind_robot_affect_reviews_then_adjudication_v1"

EMOTIONS = ("neutral", "sad", "happy", "angry", "surprise", "fear")
OBSERVABILITY = ("observable", "not_observable", "ambiguous")
SPLITS = ("train", "validation", "test")
PRIMARY_ROLES = ("primary_1", "primary_2")
FORBIDDEN_DATASET_MARKER = "kimodo"
FORBIDDEN_CONTROLLER_LABEL_KEYS = {
    "emotion_id",
    "source_emotion_label",
    "official_emotion",
    "official_emotion_id",
}

PRIMARY_SUBMISSION_KEYS = {
    "schema_version",
    "artifact_kind",
    "sample_id",
    "assignment_id",
    "assignment_role",
    "video_path",
    "video_sha256",
    "sample_binding_sha256",
    "review_protocol",
    "allowed_observability",
    "allowed_observed_emotions",
    "observability",
    "observed_emotion",
    "confidence",
    "reviewer_id",
    "submitted_at_utc",
    "paired_primary_reviewer_must_be_distinct",
    "third_adjudication_required_on_any_disagreement",
    "source_label_visible",
    "accepted_for_training",
}
ADJUDICATION_SUBMISSION_KEYS = {
    "schema_version",
    "artifact_kind",
    "sample_id",
    "assignment_id",
    "assignment_role",
    "video_path",
    "video_sha256",
    "sample_binding_sha256",
    "review_protocol",
    "allowed_observability",
    "allowed_observed_emotions",
    "observability",
    "observed_emotion",
    "confidence",
    "reviewer_id",
    "submitted_at_utc",
    "source_label_visible",
    "primary_decisions_visible",
    "adjudicator_must_differ_from_primary_reviewers",
    "accepted_for_training",
}

WEAK_TIER = "intended_metadata_weak"
STRONG_TIER = "human_confirmed_observable"
WEAK_SOURCE = "official_beat2_filename_protocol_intended_metadata"
STRONG_SOURCE = "independent_blind_human_review"


def stable_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _value_sha256(value: object) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _token(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]}"


def _reject_forbidden(value: object, *, context: str = "$") -> None:
    """Reject Kimodo in paths, values, and mapping keys, case-insensitively."""

    if isinstance(value, str):
        if FORBIDDEN_DATASET_MARKER in value.casefold():
            raise ValueError(f"{context} contains forbidden dataset marker Kimodo")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_forbidden(str(key), context=f"{context}.<key>")
            _reject_forbidden(child, context=f"{context}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden(child, context=f"{context}[{index}]")


def _forbidden_key_paths(
    value: object,
    *,
    prefix: str = "$",
) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{prefix}.{key}"
            if str(key).casefold() in FORBIDDEN_CONTROLLER_LABEL_KEYS:
                found.append(child_path)
            found.extend(_forbidden_key_paths(child, prefix=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_key_paths(child, prefix=f"{prefix}[{index}]"))
    return found


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_text(path, "".join(stable_json(dict(row)) + "\n" for row in rows))


def _read_jsonl_bound(path: Path) -> list[tuple[dict[str, Any], str]]:
    _reject_forbidden(str(path), context="input_path")
    rows: list[tuple[dict[str, Any], str]] = []
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            payload = raw[:-1] if raw.endswith(b"\n") else raw
            if payload.endswith(b"\r"):
                raise ValueError(f"CRLF JSONL is not accepted: {path}:{line_number}")
            if not payload.strip():
                continue
            try:
                row = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid JSONL: {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"expected object: {path}:{line_number}")
            _reject_forbidden(row, context=f"{path}:{line_number}")
            rows.append((row, hashlib.sha256(payload).hexdigest()))
    return rows


def _resolve_path(value: object, *, base: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    _reject_forbidden(str(path), context=field)
    return path


def _bound_input(
    value: object,
    *,
    base: Path,
    field: str,
    required: bool = True,
) -> tuple[Path, str] | None:
    if value is None and not required:
        return None
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{field} must contain exactly path and sha256")
    path = _resolve_path(value["path"], base=base, field=f"{field}.path")
    expected = value["sha256"]
    if not _is_sha256(expected):
        raise ValueError(f"{field}.sha256 is invalid")
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"{field} hash binding mismatch")
    return path, str(expected)


def _require_split(value: object, *, context: str) -> str:
    if value not in SPLITS:
        raise ValueError(f"{context}: invalid fixed split {value!r}")
    return str(value)


def _require_identity(
    row: Mapping[str, Any],
    *,
    context: str,
) -> tuple[str, str, str]:
    split = _require_split(row.get("fixed_split_assignment"), context=context)
    speaker = row.get("speaker_key")
    source = row.get("source_group_key")
    if not isinstance(speaker, str) or not speaker:
        raise ValueError(f"{context}: missing speaker_key")
    if not isinstance(source, str) or not source:
        raise ValueError(f"{context}: missing source_group_key")
    return split, _token("speaker", speaker), _token("source", source)


def _trajectory_binding(
    row: Mapping[str, Any],
    *,
    manifest_path: Path,
    context: str,
) -> tuple[Path, str]:
    nested = row.get("motion_18d")
    nested = nested if isinstance(nested, Mapping) else {}
    path_value = row.get("safe_csv") or nested.get("safe_csv")
    digest = row.get("safe_csv_sha256") or nested.get("safe_csv_sha256")
    if not isinstance(path_value, str) or not _is_sha256(digest):
        raise ValueError(f"{context}: missing trajectory path/SHA256")
    trajectory = Path(path_value)
    if not trajectory.is_absolute():
        trajectory = manifest_path.parent / trajectory
    trajectory = trajectory.resolve()
    _reject_forbidden(str(trajectory), context=f"{context}.trajectory")
    if not trajectory.is_file() or sha256_file(trajectory) != digest:
        raise ValueError(f"{context}: trajectory hash mismatch")
    return trajectory, str(digest)


def _weak_record(
    row: Mapping[str, Any],
    *,
    manifest_path: Path,
    manifest_sha256: str,
    line_sha256: str,
    weight: float,
) -> dict[str, Any]:
    context = str(row.get("clip_id") or row.get("task_id") or line_sha256[:12])
    if (
        row.get("dataset") != "BEAT2"
        or row.get("status") != "passed"
        or row.get("emotion_id") not in EMOTIONS
        or row.get("emotion_label_source") != "official_beat2_filename_protocol"
        or row.get("source_emotion_label_verified") is not True
        or row.get("official_emotion_conditioning_enabled") is not False
    ):
        raise ValueError(f"{context}: invalid BEAT2 intended-emotion source")
    quality = row.get("quality_gate")
    nested = row.get("motion_18d")
    if not isinstance(quality, Mapping) and isinstance(nested, Mapping):
        quality = nested.get("quality_gate")
    if not isinstance(quality, Mapping) or quality.get("passed") is not True:
        raise ValueError(f"{context}: physical quality gate is not passed")
    split, speaker_token, source_token = _require_identity(row, context=context)
    trajectory, trajectory_sha256 = _trajectory_binding(
        row,
        manifest_path=manifest_path,
        context=context,
    )
    sample_id = row.get("clip_id") or row.get("task_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"{context}: missing clip/task identity")
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": RECORD_KIND,
        "sample_id": sample_id,
        "dataset": "BEAT2",
        "fixed_split_assignment": split,
        "speaker_group_token": speaker_token,
        "source_group_token": source_token,
        "trajectory_path": str(trajectory),
        "trajectory_sha256": trajectory_sha256,
        "emotion_id": str(row["emotion_id"]),
        "supervision_tier": WEAK_TIER,
        "emotion_target_source": WEAK_SOURCE,
        "human_confirmed_observable": False,
        "emotion_supervision_mask": False,
        "weak_emotion_training_mask": True,
        "emotion_loss_enabled": True,
        "emotion_loss_weight": float(weight),
        "review_resolution": "not_reviewed_official_intended_metadata_only",
        "review_receipt": None,
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": manifest_sha256,
        "source_record_line_sha256": line_sha256,
        "formal_release_eligible": False,
    }


def _require_controller(
    row: Mapping[str, Any],
    *,
    line_sha256: str,
    controller_path: Path,
) -> dict[str, Any]:
    sample_id = row.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id.startswith("beat2_affect_"):
        raise ValueError("controller row has invalid sample_id")
    exact = {
        "artifact_kind": row.get("artifact_kind") == CONTROLLER_KIND,
        "dataset": row.get("dataset") == "BEAT2",
        "protocol": row.get("review_protocol") == PROTOCOL,
        "official_label_hidden": row.get("official_emotion_field_present") is False,
        "automated_label_absent": row.get("automated_emotion_label_assigned") is False,
        "primary_distinct": row.get("primary_reviewer_ids_must_be_distinct") is True,
        "primary_agreement": row.get("primary_agreement_required") is True,
        "third_required": row.get(
            "third_adjudication_required_on_any_primary_disagreement"
        )
        is True,
        "emotion_closed": row.get("emotion_supervision_mask") is False,
        "affect_closed": row.get("affect_observable_supervision_mask") is False,
        "training_closed": row.get("accepted_for_training") is False,
    }
    failed = sorted(key for key, passed in exact.items() if not passed)
    if failed:
        raise ValueError(f"{sample_id}: controller contract failed: {failed}")
    forbidden_label_paths = _forbidden_key_paths(row)
    if forbidden_label_paths:
        raise ValueError(
            f"{sample_id}: controller queue exposes source emotion fields: "
            f"{forbidden_label_paths}"
        )
    primary = row.get("primary_reviews")
    if not isinstance(primary, list) or len(primary) != 2:
        raise ValueError(f"{sample_id}: controller must have two pristine primary slots")
    for index, slot in enumerate(primary):
        if (
            not isinstance(slot, Mapping)
            or slot.get("slot") != f"reviewer_{index + 1}"
            or slot.get("status") != "pending_independent_review"
            or any(
                slot.get(field) is not None
                for field in (
                    "reviewer_id",
                    "observability",
                    "observed_emotion",
                    "confidence",
                    "submitted_at_utc",
                )
            )
        ):
            raise ValueError(f"{sample_id}: controller primary slot is not pristine")
    third = row.get("third_adjudication")
    if (
        not isinstance(third, Mapping)
        or third.get("status") != "not_requested_pending_primary_reviews"
        or third.get("must_differ_from_primary_reviewers") is not True
    ):
        raise ValueError(f"{sample_id}: controller third-review slot is not pristine")
    split = _require_split(
        row.get("fixed_split_assignment"),
        context=sample_id,
    )
    speaker = row.get("speaker_group_token")
    source = row.get("source_group_token")
    trajectory_value = row.get("trajectory_path")
    trajectory_sha256 = row.get("trajectory_sha256")
    if (
        not isinstance(speaker, str)
        or not speaker.startswith("speaker_")
        or not isinstance(source, str)
        or not source.startswith("source_")
        or not isinstance(trajectory_value, str)
        or not _is_sha256(trajectory_sha256)
    ):
        raise ValueError(f"{sample_id}: invalid controller identity binding")
    trajectory = Path(trajectory_value)
    if not trajectory.is_absolute():
        trajectory = controller_path.parent / trajectory
    trajectory = trajectory.resolve()
    _reject_forbidden(str(trajectory), context=f"{sample_id}.trajectory")
    if not trajectory.is_file() or sha256_file(trajectory) != trajectory_sha256:
        raise ValueError(f"{sample_id}: controller trajectory hash mismatch")
    if not _is_sha256(line_sha256):
        raise AssertionError("controller line digest is not SHA256")
    return {
        "sample_id": sample_id,
        "split": split,
        "speaker_group_token": speaker,
        "source_group_token": source,
        "trajectory_path": trajectory,
        "trajectory_sha256": str(trajectory_sha256),
        "controller_record_line_sha256": line_sha256,
    }


def _index_unique(
    rows: Sequence[tuple[dict[str, Any], str]],
    *,
    key: str,
    context: str,
) -> dict[str, tuple[dict[str, Any], str]]:
    result: dict[str, tuple[dict[str, Any], str]] = {}
    for row, digest in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{context}: missing {key}")
        if value in result:
            raise ValueError(f"{context}: duplicate {key} {value}")
        result[value] = (row, digest)
    return result


def _require_mapping(
    row: Mapping[str, Any],
    *,
    line_sha256: str,
    controller: Mapping[str, Any],
    mapping_path: Path,
) -> dict[str, Any]:
    public_id = row.get("public_sample_id")
    if (
        row.get("artifact_kind") != MAPPING_KIND
        or row.get("controller_sample_id") != controller["sample_id"]
        or row.get("controller_record_line_sha256")
        != controller["controller_record_line_sha256"]
        or row.get("trajectory_sha256") != controller["trajectory_sha256"]
        or not isinstance(public_id, str)
        or not public_id.startswith("motion_")
        or not _is_sha256(row.get("video_sha256"))
        or row.get("primary_reviewer_ids_must_be_distinct") is not True
        or row.get("third_adjudication_required_on_any_primary_disagreement")
        is not True
        or row.get("third_adjudicator_must_differ_from_primary_reviewers")
        is not True
        or row.get("accepted_for_training") is not False
    ):
        raise ValueError(f"{controller['sample_id']}: hidden mapping binding failed")
    assignments = row.get("primary_assignments")
    if not isinstance(assignments, Mapping) or set(assignments) != set(PRIMARY_ROLES):
        raise ValueError(f"{public_id}: primary assignment mapping is incomplete")
    normalized_assignments: dict[str, dict[str, str]] = {}
    for role in PRIMARY_ROLES:
        assignment = assignments[role]
        if (
            not isinstance(assignment, Mapping)
            or not isinstance(assignment.get("assignment_id"), str)
            or not _is_sha256(assignment.get("sample_binding_sha256"))
        ):
            raise ValueError(f"{public_id}: invalid {role} assignment binding")
        normalized_assignments[role] = {
            "assignment_id": str(assignment["assignment_id"]),
            "sample_binding_sha256": str(assignment["sample_binding_sha256"]),
        }
    if len(
        {
            assignment["assignment_id"]
            for assignment in normalized_assignments.values()
        }
    ) != len(PRIMARY_ROLES):
        raise ValueError(f"{public_id}: primary assignment IDs must be distinct")
    video_value = row.get("anonymous_video_path")
    if not isinstance(video_value, str) or not video_value:
        raise ValueError(f"{public_id}: hidden mapping has no anonymous video path")
    video_path = Path(video_value)
    if not video_path.is_absolute():
        video_path = mapping_path.parent / video_path
    video_path = video_path.resolve()
    _reject_forbidden(str(video_path), context=f"{public_id}.anonymous_video_path")
    if (
        not video_path.is_file()
        or sha256_file(video_path) != row.get("video_sha256")
    ):
        raise ValueError(f"{public_id}: reviewed anonymous video hash mismatch")
    return {
        "public_sample_id": public_id,
        "video_path": str(video_path),
        "video_sha256": str(row["video_sha256"]),
        "mapping_record_line_sha256": line_sha256,
        "primary_assignments": normalized_assignments,
    }


def _submitted_at(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}: submitted_at_utc is required")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{context}: submitted_at_utc is invalid") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{context}: submitted_at_utc must include timezone")
    return value


def _decision(
    row: Mapping[str, Any],
    *,
    context: str,
) -> tuple[str, str | None, float, str, str]:
    observability = row.get("observability")
    emotion = row.get("observed_emotion")
    confidence = row.get("confidence")
    reviewer = row.get("reviewer_id")
    if observability not in OBSERVABILITY:
        raise ValueError(f"{context}: invalid observability")
    if observability == "observable":
        if emotion not in EMOTIONS:
            raise ValueError(f"{context}: observable review needs an emotion")
    elif emotion is not None:
        raise ValueError(f"{context}: non-observable review carries an emotion")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise ValueError(f"{context}: invalid confidence")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError(f"{context}: reviewer_id is required")
    submitted_at = _submitted_at(row.get("submitted_at_utc"), context=context)
    return (
        str(observability),
        str(emotion) if emotion is not None else None,
        float(confidence),
        reviewer,
        submitted_at,
    )


def _require_primary_submission(
    row: Mapping[str, Any],
    *,
    role: str,
    mapping: Mapping[str, Any],
) -> dict[str, Any]:
    public_id = mapping["public_sample_id"]
    if set(row) != PRIMARY_SUBMISSION_KEYS:
        raise ValueError(
            f"{public_id}/{role}: primary submission fields changed"
        )
    expected = mapping["primary_assignments"][role]
    exact = {
        "artifact_kind": row.get("artifact_kind") == PRIMARY_KIND,
        "sample_id": row.get("sample_id") == public_id,
        "assignment_role": row.get("assignment_role") == role,
        "assignment_id": row.get("assignment_id") == expected["assignment_id"],
        "binding": row.get("sample_binding_sha256")
        == expected["sample_binding_sha256"],
        "video": row.get("video_sha256") == mapping["video_sha256"],
        "protocol": row.get("review_protocol") == PROTOCOL,
        "source_hidden": row.get("source_label_visible") is False,
        "training_closed": row.get("accepted_for_training") is False,
        "paired_distinct": row.get("paired_primary_reviewer_must_be_distinct")
        is True,
        "third_required": row.get(
            "third_adjudication_required_on_any_disagreement"
        )
        is True,
    }
    failed = sorted(key for key, passed in exact.items() if not passed)
    if failed:
        raise ValueError(f"{public_id}/{role}: submission binding failed: {failed}")
    if row.get("allowed_observability") != list(OBSERVABILITY):
        raise ValueError(f"{public_id}/{role}: observability vocabulary changed")
    if row.get("allowed_observed_emotions") != list(EMOTIONS):
        raise ValueError(f"{public_id}/{role}: emotion vocabulary changed")
    observability, emotion, confidence, reviewer, submitted_at = _decision(
        row,
        context=f"{public_id}/{role}",
    )
    return {
        "role": role,
        "observability": observability,
        "observed_emotion": emotion,
        "confidence": confidence,
        "reviewer_id": reviewer,
        "submitted_at_utc": submitted_at,
        "submission_record_sha256": _value_sha256(row),
    }


def _require_adjudication_submission(
    row: Mapping[str, Any],
    *,
    mapping: Mapping[str, Any],
    primary_reviewer_ids: set[str],
) -> dict[str, Any]:
    public_id = mapping["public_sample_id"]
    if set(row) != ADJUDICATION_SUBMISSION_KEYS:
        raise ValueError(
            f"{public_id}/adjudication: adjudication submission fields changed"
        )
    exact = {
        "artifact_kind": row.get("artifact_kind") == ADJUDICATION_KIND,
        "sample_id": row.get("sample_id") == public_id,
        "assignment_role": row.get("assignment_role") == "adjudication",
        "binding": row.get("sample_binding_sha256")
        in {
            mapping["primary_assignments"][role]["sample_binding_sha256"]
            for role in PRIMARY_ROLES
        },
        "video": row.get("video_sha256") == mapping["video_sha256"],
        "protocol": row.get("review_protocol") == PROTOCOL,
        "source_hidden": row.get("source_label_visible") is False,
        "primary_hidden": row.get("primary_decisions_visible") is False,
        "training_closed": row.get("accepted_for_training") is False,
        "distinct_required": row.get(
            "adjudicator_must_differ_from_primary_reviewers"
        )
        is True,
    }
    failed = sorted(key for key, passed in exact.items() if not passed)
    if failed:
        raise ValueError(
            f"{public_id}/adjudication: submission binding failed: {failed}"
        )
    observability, emotion, confidence, reviewer, submitted_at = _decision(
        row,
        context=f"{public_id}/adjudication",
    )
    if reviewer in primary_reviewer_ids:
        raise ValueError(f"{public_id}: adjudicator must differ from both primaries")
    return {
        "role": "adjudication",
        "observability": observability,
        "observed_emotion": emotion,
        "confidence": confidence,
        "reviewer_id": reviewer,
        "submitted_at_utc": submitted_at,
        "submission_record_sha256": _value_sha256(row),
    }


def _label_decision(review: Mapping[str, Any]) -> tuple[str, str | None]:
    return str(review["observability"]), review.get("observed_emotion")


def _strong_record(
    *,
    controller: Mapping[str, Any],
    mapping: Mapping[str, Any],
    emotion: str,
    weight: float,
    resolution: str,
    primary: Sequence[Mapping[str, Any]],
    adjudication: Mapping[str, Any] | None,
    controller_queue_sha256: str,
    mapping_manifest_sha256: str,
) -> dict[str, Any]:
    receipt = {
        "protocol": PROTOCOL,
        "public_sample_id": mapping["public_sample_id"],
        "video_sha256": mapping["video_sha256"],
        "controller_queue_sha256": controller_queue_sha256,
        "controller_record_line_sha256": controller[
            "controller_record_line_sha256"
        ],
        "hidden_mapping_sha256": mapping_manifest_sha256,
        "mapping_record_line_sha256": mapping["mapping_record_line_sha256"],
        "primary_reviews": [dict(value) for value in primary],
        "adjudication": dict(adjudication) if adjudication is not None else None,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": RECORD_KIND,
        "sample_id": controller["sample_id"],
        "dataset": "BEAT2",
        "fixed_split_assignment": controller["split"],
        "speaker_group_token": controller["speaker_group_token"],
        "source_group_token": controller["source_group_token"],
        "trajectory_path": str(controller["trajectory_path"]),
        "trajectory_sha256": controller["trajectory_sha256"],
        "emotion_id": emotion,
        "supervision_tier": STRONG_TIER,
        "emotion_target_source": STRONG_SOURCE,
        "human_confirmed_observable": True,
        "emotion_supervision_mask": True,
        "weak_emotion_training_mask": False,
        "emotion_loss_enabled": True,
        "emotion_loss_weight": float(weight),
        "review_resolution": resolution,
        "review_receipt": receipt,
        "source_manifest_path": None,
        "source_manifest_sha256": None,
        "source_record_line_sha256": controller[
            "controller_record_line_sha256"
        ],
        "formal_release_eligible": False,
    }


def _assert_no_split_leakage(rows: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    checks: dict[str, defaultdict[str, set[str]]] = {
        field: defaultdict(set)
        for field in (
            "speaker_group_token",
            "source_group_token",
            "trajectory_sha256",
        )
    }
    trajectory_sources: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = _require_split(
            row.get("fixed_split_assignment"),
            context=str(row.get("sample_id")),
        )
        for field, values in checks.items():
            value = row.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{row.get('sample_id')}: missing {field}")
            values[value].add(split)
        trajectory_sources[str(row["trajectory_sha256"])].add(
            str(row["source_group_token"])
        )
    failures = {
        field: sorted(value for value, splits in values.items() if len(splits) > 1)
        for field, values in checks.items()
    }
    multi_source_trajectory = sorted(
        digest for digest, sources in trajectory_sources.items() if len(sources) > 1
    )
    if any(failures.values()) or multi_source_trajectory:
        raise ValueError(
            "cross split/source leakage detected: "
            + stable_json(
                {
                    **failures,
                    "trajectory_bound_to_multiple_sources": multi_source_trajectory,
                }
            )
        )
    return {
        "speaker_disjoint_split": True,
        "source_group_disjoint_split": True,
        "trajectory_disjoint_split": True,
        "trajectory_bound_to_one_source": True,
    }


def _validate_record(row: Mapping[str, Any]) -> None:
    sample_id = row.get("sample_id")
    if (
        row.get("schema_version") != SCHEMA_VERSION
        or row.get("artifact_kind") != RECORD_KIND
        or row.get("dataset") != "BEAT2"
        or not isinstance(sample_id, str)
        or not sample_id
        or row.get("emotion_id") not in EMOTIONS
        or row.get("emotion_loss_enabled") is not True
        or row.get("formal_release_eligible") is not False
    ):
        raise ValueError(f"{sample_id}: invalid emotion supervision record")
    weight = row.get("emotion_loss_weight")
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(float(weight))
        or float(weight) <= 0.0
    ):
        raise ValueError(f"{sample_id}: invalid emotion loss weight")
    tier = row.get("supervision_tier")
    if tier == STRONG_TIER:
        if not (
            row.get("emotion_target_source") == STRONG_SOURCE
            and row.get("human_confirmed_observable") is True
            and row.get("emotion_supervision_mask") is True
            and row.get("weak_emotion_training_mask") is False
            and isinstance(row.get("review_receipt"), Mapping)
        ):
            raise ValueError(f"{sample_id}: strong supervision is not human-confirmed")
    elif tier == WEAK_TIER:
        if not (
            row.get("emotion_target_source") == WEAK_SOURCE
            and row.get("human_confirmed_observable") is False
            and row.get("emotion_supervision_mask") is False
            and row.get("weak_emotion_training_mask") is True
            and row.get("review_receipt") is None
        ):
            raise ValueError(f"{sample_id}: weak metadata target is mislabeled")
    else:
        raise ValueError(f"{sample_id}: unsupported supervision tier")
    trajectory = Path(str(row.get("trajectory_path") or "")).resolve()
    if (
        not trajectory.is_file()
        or not _is_sha256(row.get("trajectory_sha256"))
        or sha256_file(trajectory) != row.get("trajectory_sha256")
    ):
        raise ValueError(f"{sample_id}: trajectory binding changed")


def load_emotion_supervision_manifest(path: Path) -> list[dict[str, Any]]:
    """Load the only accepted emotion-loss ingress format."""

    path = path.resolve()
    rows = [row for row, _ in _read_jsonl_bound(path)]
    seen_samples: set[str] = set()
    seen_trajectories: set[str] = set()
    for row in rows:
        _validate_record(row)
        sample = str(row["sample_id"])
        trajectory = str(row["trajectory_sha256"])
        if sample in seen_samples:
            raise ValueError(f"duplicate emotion sample_id: {sample}")
        if trajectory in seen_trajectories:
            raise ValueError(f"duplicate emotion trajectory: {trajectory}")
        seen_samples.add(sample)
        seen_trajectories.add(trajectory)
    _assert_no_split_leakage(rows)
    strong_weights = [
        float(row["emotion_loss_weight"])
        for row in rows
        if row["supervision_tier"] == STRONG_TIER
    ]
    weak_weights = [
        float(row["emotion_loss_weight"])
        for row in rows
        if row["supervision_tier"] == WEAK_TIER
    ]
    if strong_weights and weak_weights and max(weak_weights) >= min(strong_weights):
        raise ValueError("weak intended-metadata weight must be below strong human weight")
    return rows


def load_emotion_training_rows(path: Path) -> list[dict[str, Any]]:
    """Return only the fixed training split from a validated manifest.

    Training code should use this function instead of filtering the full
    manifest itself.  Validation and test targets remain available through
    :func:`load_emotion_supervision_manifest` for evaluation only.
    """

    rows = load_emotion_supervision_manifest(path)
    return [
        row for row in rows if row["fixed_split_assignment"] == "train"
    ]


def _load_bound_many(
    values: object,
    *,
    base: Path,
    field: str,
) -> list[tuple[Path, str]]:
    if not isinstance(values, list):
        raise ValueError(f"{field} must be a list")
    result: list[tuple[Path, str]] = []
    for index, value in enumerate(values):
        bound = _bound_input(
            value,
            base=base,
            field=f"{field}[{index}]",
        )
        assert bound is not None
        result.append(bound)
    return result


def _read_submission_union(
    inputs: Sequence[tuple[Path, str]],
    *,
    context: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    assignment_ids: set[str] = set()
    for path, _ in inputs:
        for row, _ in _read_jsonl_bound(path):
            sample_id = row.get("sample_id")
            assignment_id = row.get("assignment_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"{context}: submission has no sample_id")
            if sample_id in indexed:
                raise ValueError(f"{context}: duplicate sample submission {sample_id}")
            if not isinstance(assignment_id, str) or not assignment_id:
                raise ValueError(f"{context}: submission has no assignment_id")
            if assignment_id in assignment_ids:
                raise ValueError(f"{context}: duplicate assignment_id {assignment_id}")
            assignment_ids.add(assignment_id)
            indexed[sample_id] = row
    return indexed


def _pending_adjudication(
    *,
    mapping: Mapping[str, Any],
) -> dict[str, Any]:
    public_id = str(mapping["public_sample_id"])
    binding = mapping["primary_assignments"]["primary_1"]["sample_binding_sha256"]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ADJUDICATION_KIND,
        "sample_id": public_id,
        "assignment_id": "adjudication_" + hashlib.sha256(
            f"{public_id}\0{binding}".encode("utf-8")
        ).hexdigest()[:24],
        "assignment_role": "adjudication",
        "video_path": mapping["video_path"],
        "video_sha256": mapping["video_sha256"],
        "sample_binding_sha256": binding,
        "review_protocol": PROTOCOL,
        "allowed_observability": list(OBSERVABILITY),
        "allowed_observed_emotions": list(EMOTIONS),
        "observability": None,
        "observed_emotion": None,
        "confidence": None,
        "reviewer_id": None,
        "submitted_at_utc": None,
        "source_label_visible": False,
        "primary_decisions_visible": False,
        "adjudicator_must_differ_from_primary_reviewers": True,
        "accepted_for_training": False,
    }


def build_emotion_supervision_from_config(config_path: Path) -> dict[str, Any]:
    """Build an audited strong/weak BEAT2 emotion-loss manifest."""

    config_path = config_path.resolve()
    _reject_forbidden(str(config_path), context="config_path")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read emotion supervision config: {error}") from error
    if not isinstance(config, dict):
        raise ValueError("emotion supervision config must be an object")
    _reject_forbidden(config, context="config")
    if (
        config.get("schema_version") != SCHEMA_VERSION
        or config.get("artifact_kind") != CONFIG_KIND
        or config.get("allowed_dataset") != "BEAT2"
    ):
        raise ValueError("invalid BEAT2 emotion supervision config contract")
    base = config_path.parent
    weights = config.get("weights")
    if not isinstance(weights, Mapping) or set(weights) != {
        "human_confirmed_observable",
        "intended_metadata_weak",
    }:
        raise ValueError("weights must define exactly strong and weak tiers")
    strong_weight = weights["human_confirmed_observable"]
    weak_weight = weights["intended_metadata_weak"]
    if (
        isinstance(strong_weight, bool)
        or not isinstance(strong_weight, (int, float))
        or not math.isfinite(float(strong_weight))
        or isinstance(weak_weight, bool)
        or not isinstance(weak_weight, (int, float))
        or not math.isfinite(float(weak_weight))
        or not 0.0 < float(weak_weight) < float(strong_weight)
    ):
        raise ValueError("weak weight must be positive and below strong weight")

    weak_inputs = _load_bound_many(
        config.get("weak_source_manifests"),
        base=base,
        field="weak_source_manifests",
    )
    controller_bound = _bound_input(
        config.get("controller_queue"),
        base=base,
        field="controller_queue",
    )
    mapping_bound = _bound_input(
        config.get("hidden_mapping"),
        base=base,
        field="hidden_mapping",
    )
    assert controller_bound is not None and mapping_bound is not None
    controller_path, controller_sha = controller_bound
    mapping_path, mapping_sha = mapping_bound

    primary_config = config.get("primary_submissions")
    if not isinstance(primary_config, Mapping) or set(primary_config) != set(
        PRIMARY_ROLES
    ):
        raise ValueError("primary_submissions must define primary_1 and primary_2")
    primary_inputs = {
        role: _load_bound_many(
            primary_config[role],
            base=base,
            field=f"primary_submissions.{role}",
        )
        for role in PRIMARY_ROLES
    }
    adjudication_inputs = _load_bound_many(
        config.get("adjudication_submissions"),
        base=base,
        field="adjudication_submissions",
    )

    output_manifest = _resolve_path(
        config.get("output_manifest"),
        base=base,
        field="output_manifest",
    )
    output_audit = _resolve_path(
        config.get("output_audit"),
        base=base,
        field="output_audit",
    )
    output_adjudication = _resolve_path(
        config.get("output_adjudication_queue"),
        base=base,
        field="output_adjudication_queue",
    )
    for path in (output_manifest, output_audit, output_adjudication):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite emotion artifact: {path}")

    weak_records: list[dict[str, Any]] = []
    for path, digest in weak_inputs:
        for row, line_sha256 in _read_jsonl_bound(path):
            weak_records.append(
                _weak_record(
                    row,
                    manifest_path=path,
                    manifest_sha256=digest,
                    line_sha256=line_sha256,
                    weight=float(weak_weight),
                )
            )
    weak_by_trajectory: dict[str, dict[str, Any]] = {}
    weak_samples: set[str] = set()
    for row in weak_records:
        trajectory = str(row["trajectory_sha256"])
        if trajectory in weak_by_trajectory:
            raise ValueError(f"duplicate weak trajectory SHA256: {trajectory}")
        if row["sample_id"] in weak_samples:
            raise ValueError(f"duplicate weak sample_id: {row['sample_id']}")
        weak_by_trajectory[trajectory] = row
        weak_samples.add(str(row["sample_id"]))

    controller_rows = _read_jsonl_bound(controller_path)
    controllers: dict[str, dict[str, Any]] = {}
    for row, line_sha256 in controller_rows:
        controller = _require_controller(
            row,
            line_sha256=line_sha256,
            controller_path=controller_path,
        )
        if controller["sample_id"] in controllers:
            raise ValueError(f"duplicate controller sample {controller['sample_id']}")
        controllers[controller["sample_id"]] = controller

    mapping_rows = _read_jsonl_bound(mapping_path)
    mapping_raw = _index_unique(
        mapping_rows,
        key="controller_sample_id",
        context="hidden mapping",
    )
    mappings: dict[str, dict[str, Any]] = {}
    public_ids: set[str] = set()
    for controller_id, (row, line_sha256) in mapping_raw.items():
        if controller_id not in controllers:
            raise ValueError(f"mapping references unknown controller {controller_id}")
        mapping = _require_mapping(
            row,
            line_sha256=line_sha256,
            controller=controllers[controller_id],
            mapping_path=mapping_path,
        )
        if mapping["public_sample_id"] in public_ids:
            raise ValueError("hidden mapping reuses a public sample ID")
        public_ids.add(str(mapping["public_sample_id"]))
        mappings[controller_id] = mapping

    primary_rows = {
        role: _read_submission_union(primary_inputs[role], context=role)
        for role in PRIMARY_ROLES
    }
    adjudication_rows = _read_submission_union(
        adjudication_inputs,
        context="adjudication",
    )
    known_public_ids = {
        str(mapping["public_sample_id"]) for mapping in mappings.values()
    }
    submitted_public_ids = (
        set(primary_rows["primary_1"])
        | set(primary_rows["primary_2"])
        | set(adjudication_rows)
    )
    unknown = sorted(submitted_public_ids - known_public_ids)
    if unknown:
        raise ValueError(f"review submissions reference unknown samples: {unknown}")

    strong_records: list[dict[str, Any]] = []
    pending_adjudication: list[dict[str, Any]] = []
    review_counts: Counter[str] = Counter()
    suppressed_weak_trajectories: set[str] = set()
    used_adjudication_public_ids: set[str] = set()
    for controller_id in sorted(mappings):
        controller = controllers[controller_id]
        mapping = mappings[controller_id]
        public_id = str(mapping["public_sample_id"])
        raw_primary = [
            primary_rows[role].get(public_id) for role in PRIMARY_ROLES
        ]
        present = sum(row is not None for row in raw_primary)
        if present == 0:
            review_counts["zero_reviews_keep_weak"] += 1
            continue
        suppressed_weak_trajectories.add(str(controller["trajectory_sha256"]))
        if present == 1:
            review_counts["partial_primary_excluded"] += 1
            continue
        primary = [
            _require_primary_submission(
                raw_primary[index],
                role=role,
                mapping=mapping,
            )
            for index, role in enumerate(PRIMARY_ROLES)
        ]
        reviewer_ids = {str(review["reviewer_id"]) for review in primary}
        if len(reviewer_ids) != 2:
            raise ValueError(f"{public_id}: primary reviewers must be distinct")
        decisions = [_label_decision(review) for review in primary]
        if decisions[0] == decisions[1]:
            if decisions[0][0] == "observable":
                strong_records.append(
                    _strong_record(
                        controller=controller,
                        mapping=mapping,
                        emotion=str(decisions[0][1]),
                        weight=float(strong_weight),
                        resolution="two_independent_primary_reviewers_agreed",
                        primary=primary,
                        adjudication=None,
                        controller_queue_sha256=controller_sha,
                        mapping_manifest_sha256=mapping_sha,
                    )
                )
                review_counts["primary_agreement_observable"] += 1
            else:
                review_counts[
                    f"primary_agreement_{decisions[0][0]}_excluded"
                ] += 1
            if public_id in adjudication_rows:
                raise ValueError(
                    f"{public_id}: adjudication supplied without primary disagreement"
                )
            continue

        raw_adjudication = adjudication_rows.get(public_id)
        if raw_adjudication is None:
            pending_adjudication.append(_pending_adjudication(mapping=mapping))
            review_counts["primary_disagreement_pending_adjudication"] += 1
            continue
        adjudication = _require_adjudication_submission(
            raw_adjudication,
            mapping=mapping,
            primary_reviewer_ids=reviewer_ids,
        )
        used_adjudication_public_ids.add(public_id)
        if adjudication["observability"] == "observable":
            strong_records.append(
                _strong_record(
                    controller=controller,
                    mapping=mapping,
                    emotion=str(adjudication["observed_emotion"]),
                    weight=float(strong_weight),
                    resolution="distinct_third_reviewer_adjudicated_disagreement",
                    primary=primary,
                    adjudication=adjudication,
                    controller_queue_sha256=controller_sha,
                    mapping_manifest_sha256=mapping_sha,
                )
            )
            review_counts["adjudicated_observable"] += 1
        else:
            review_counts[
                f"adjudicated_{adjudication['observability']}_excluded"
            ] += 1

    unused_adjudications = sorted(
        set(adjudication_rows) - used_adjudication_public_ids
    )
    if unused_adjudications:
        raise ValueError(
            "unused or invalid adjudication submissions: "
            f"{unused_adjudications}"
        )

    retained_weak = [
        row
        for digest, row in weak_by_trajectory.items()
        if digest not in suppressed_weak_trajectories
    ]
    strong_trajectories = {
        str(row["trajectory_sha256"]) for row in strong_records
    }
    if strong_trajectories & {
        str(row["trajectory_sha256"]) for row in retained_weak
    }:
        raise AssertionError("strong reviewed trajectory was not promoted atomically")
    combined = sorted(
        [*retained_weak, *strong_records],
        key=lambda row: (
            str(row["fixed_split_assignment"]),
            str(row["supervision_tier"]),
            str(row["sample_id"]),
        ),
    )
    combined_samples = [str(row["sample_id"]) for row in combined]
    combined_trajectories = [str(row["trajectory_sha256"]) for row in combined]
    if (
        len(set(combined_samples)) != len(combined_samples)
        or len(set(combined_trajectories)) != len(combined_trajectories)
    ):
        raise ValueError("emotion supervision output has duplicate sample/trajectory")
    for row in combined:
        _validate_record(row)
    leakage = _assert_no_split_leakage(combined)
    if any(
        float(row["emotion_loss_weight"]) >= float(strong_weight)
        for row in retained_weak
    ):
        raise AssertionError("weak target is not down-weighted")

    tier_counts = Counter(str(row["supervision_tier"]) for row in combined)
    split_tier_counts: dict[str, dict[str, int]] = {}
    split_emotion_counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        split_tier_counts[split] = {
            tier: sum(
                row["fixed_split_assignment"] == split
                and row["supervision_tier"] == tier
                for row in combined
            )
            for tier in (STRONG_TIER, WEAK_TIER)
        }
        split_emotion_counts[split] = {
            emotion: sum(
                row["fixed_split_assignment"] == split
                and row["emotion_id"] == emotion
                for row in combined
            )
            for emotion in EMOTIONS
        }

    _atomic_jsonl(output_manifest, combined)
    _atomic_jsonl(output_adjudication, pending_adjudication)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": AUDIT_KIND,
        "dataset": "BEAT2",
        "policy": {
            "strong_tier": STRONG_TIER,
            "strong_source": STRONG_SOURCE,
            "strong_weight": float(strong_weight),
            "weak_tier": WEAK_TIER,
            "weak_source": WEAK_SOURCE,
            "weak_weight": float(weak_weight),
            "human_label_propagation": "exact_reviewed_trajectory_only",
            "zero_review_policy": "retain_intended_metadata_as_weak",
            "partial_or_unresolved_review_policy": "exclude_from_emotion_loss",
            "non_observable_policy": "exclude_from_emotion_loss",
            "kimodo_allowed": False,
        },
        "counts": {
            "controller_review_candidates": len(controllers),
            "rendered_mapped_candidates": len(mappings),
            "unrendered_controller_candidates": len(controllers) - len(mappings),
            "input_weak_records": len(weak_records),
            "retained_weak_records": len(retained_weak),
            "human_confirmed_observable_records": len(strong_records),
            "output_records": len(combined),
            "pending_adjudications": len(pending_adjudication),
            "suppressed_weak_trajectories": len(suppressed_weak_trajectories),
            "by_tier": dict(sorted(tier_counts.items())),
            "review_resolutions": dict(sorted(review_counts.items())),
            "by_split_tier": split_tier_counts,
            "by_split_emotion": split_emotion_counts,
        },
        "integrity": {
            **leakage,
            "kimodo_strings_present": False,
            "duplicate_trajectory_records": False,
            "weak_weight_below_strong_weight": True,
            "human_review_not_propagated_across_windows": True,
        },
        "inputs": {
            "weak_source_manifests": [
                {"path": str(path), "sha256": digest}
                for path, digest in weak_inputs
            ],
            "controller_queue": {
                "path": str(controller_path),
                "sha256": controller_sha,
            },
            "hidden_mapping": {
                "path": str(mapping_path),
                "sha256": mapping_sha,
            },
            "primary_submission_files": {
                role: [
                    {"path": str(path), "sha256": digest}
                    for path, digest in primary_inputs[role]
                ]
                for role in PRIMARY_ROLES
            },
            "adjudication_submission_files": [
                {"path": str(path), "sha256": digest}
                for path, digest in adjudication_inputs
            ],
        },
        "outputs": {
            "manifest": str(output_manifest),
            "manifest_sha256": sha256_file(output_manifest),
            "pending_adjudication_queue": str(output_adjudication),
            "pending_adjudication_queue_sha256": sha256_file(
                output_adjudication
            ),
            "audit": str(output_audit),
        },
        "formal_release_eligible": False,
    }
    _atomic_json(output_audit, audit)
    return audit
