#!/usr/bin/env python3
"""Merge independent expression-turn reviews into fail-closed training tiers.

The merger is shared by BEAT2 v8 base/expanded and ULA0513 native blind bundles. It
qualifies only evidence visible in the anonymous silent robot video.  Hidden
source labels are used solely to recover motion lineage and are never copied
into, or compared by, the qualification decision.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from tools.human_motion_review.expression_turn_contract import (
    ACTION_PROTOCOL,
    ACTION_RESULTS,
    ACTION_TEXT_PROVENANCE,
    AFFECT_ADJUDICATION_PROTOCOL,
    AFFECT_CLASSES,
    AFFECT_PROTOCOL,
    AFFECT_RESULTS,
    ARC_PROTOCOL,
    ARTIFACT_KIND as EXPRESSION_REVIEW_ARTIFACT_KIND,
    COMPLETE_PHASE_BASES,
    CONTEXT_POLICY,
    CONTRACT_VERSION as EXPRESSION_TURN_CONTRACT_VERSION,
    MIN_AFFECT_CONFIDENCE,
    PHASE_STATUSES,
    REPRESENTATION as EXPRESSION_TURN_REPRESENTATION,
    contract_definition,
    evaluate_expression_turn,
)
from tools.human_motion_review.expression_turn_retarget_contract import (
    REQUIRED_18D_GATES,
    RETARGET_SEGMENT_REPRESENTATION,
)
from upper_body_skeleton.retarget_v2_18d import (
    JOINT_LIMITS_18D,
    JOINT_ORDER_18D,
)
from upper_body_skeleton.ula_v2_expression_turn_episode import (
    FORMAL_ELIGIBILITY_MODE,
    FORMAL_EPISODE_CONTRACT,
    HUMAN_RETARGET_PHYSICAL_PROFILE,
    MOTION_FORM_PROMPT_PROFILE,
    NATIVE_ROBOT_PHYSICAL_PROFILE,
    NATIVE_ROBOT_REQUIRED_18D_GATES,
    NATIVE_ROBOT_RETARGET_SEGMENT_REPRESENTATION,
    PROMPT_TEXT_PROVENANCE,
    load_expression_turn_v8_episodes,
    validate_expression_turn_v8_episode,
)


SCHEMA_VERSION = "1.0.0"
REVIEW_RECORD_KIND = "expression_turn_independent_blind_review_record_v1"
TRAIN_READY_KIND = "expression_turn_v8_adjudicated_train_ready_record"
NATIVE_MATCHER_PROTOCOL = "expression_turn_action_semantics_blind_v1"
NATIVE_MAX_VELOCITY_RAD_S = 12.0
COLLISION_FRAME_RATE_MAX = 0.05
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROBOT_URDF = (
    PROJECT_ROOT
    / "urdf_V2_20260514/urdf/xacro/robot_modify_meshdir.urdf"
)
SUPPORTED_BUNDLES = {
    "expression_turn_v8_separate_blind_review_bundle": "beat2_expression_turn_v8",
    "expression_turn_v8_expansion_separate_blind_review_bundle_v1": (
        "beat2_expression_turn_v8"
    ),
    "ula0513_native_separate_blind_review_bundle": "ula0513_native_expression_turn_v1",
}
EXPANSION_BUNDLE_KIND = (
    "expression_turn_v8_expansion_separate_blind_review_bundle_v1"
)
FORBIDDEN_SUBMISSION_KEYS = {
    "canonical_action",
    "canonical_prompt",
    "emotion_id",
    "emotion_label",
    "official_emotion",
    "official_gesture_category",
    "source_behavior_label",
    "source_clip_id",
    "source_label",
    "source_text",
    "speaker_id",
    "speaker_key",
    "transcript",
}
FORBIDDEN_OUTPUT_KEYS = {
    "semantic_event",
    "official_emotion_label",
    "official_gesture_category",
    "source_behavior_label",
    "source_text",
    "transcript",
}


def stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def value_sha256(value: object) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def ascii_value_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_text(path, "".join(stable_json(row) + "\n" for row in rows))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _walk_keys(value: object):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _reject_hidden_labels(value: object, *, context: str) -> None:
    leaked = sorted({key for key in _walk_keys(value) if key.lower() in FORBIDDEN_SUBMISSION_KEYS})
    if leaked:
        raise ValueError(f"{context}: blind submission leaks hidden metadata: {leaked}")


def _assert_train_output_has_no_hidden_labels(value: object) -> None:
    leaked = sorted({key for key in _walk_keys(value) if key.lower() in FORBIDDEN_OUTPUT_KEYS})
    if leaked:
        raise AssertionError(f"training output leaked hidden labels: {leaked}")


def _resolve_declared(path_value: object, *, owner: Path, field: str) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{owner}: missing {field}")
    path = Path(path_value)
    return path.resolve() if path.is_absolute() else (owner.parent / path).resolve()


def _index_unique(rows: list[dict[str, Any]], key: str, *, context: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{context}: missing {key}")
        if value in result:
            raise ValueError(f"{context}: duplicate {key}: {value}")
        result[value] = row
    return result


def _verified_public_bundle(summary_path: Path) -> tuple[str, Path, Path, dict[str, dict[str, Any]]]:
    summary_path = summary_path.resolve()
    summary = read_json(summary_path)
    artifact_kind = summary.get("artifact_kind")
    if artifact_kind not in SUPPORTED_BUNDLES:
        raise ValueError("unsupported public blind bundle")
    if summary.get("accepted_for_training") not in (False, 0):
        raise ValueError("public blind bundle must remain fail-closed")
    if artifact_kind == EXPANSION_BUNDLE_KIND and (
        summary.get("all_samples_native_variable_length") is not True
        or summary.get("fixed_duration_window_used") is not False
    ):
        raise ValueError("expanded public blind bundle violates native duration")
    arc_path = _resolve_declared(summary.get("arc_action_queue"), owner=summary_path, field="arc_action_queue")
    affect_path = _resolve_declared(summary.get("affect_queue"), owner=summary_path, field="affect_queue")
    for path, hash_field in (
        (arc_path, "arc_action_queue_sha256"),
        (affect_path, "affect_queue_sha256"),
    ):
        if not path.is_file() or sha256(path) != summary.get(hash_field):
            raise ValueError(f"public queue evidence mismatch: {hash_field}")
    arc = _index_unique(read_jsonl(arc_path), "sample_id", context="arc queue")
    affect = _index_unique(read_jsonl(affect_path), "sample_id", context="affect queue")
    if set(arc) != set(affect):
        raise ValueError("arc and affect queue sample sets differ")
    for sample_id, arc_row in arc.items():
        affect_row = affect[sample_id]
        common = (
            "sample_id",
            "video_path",
            "video_sha256",
            "context_level",
            "audio_available",
            "label_metadata_exposed",
        )
        if any(arc_row.get(key) != affect_row.get(key) for key in common):
            raise ValueError(f"{sample_id}: public queues use different evidence")
        if (
            arc_row.get("arc_protocol_version") != ARC_PROTOCOL
            or arc_row.get("action_protocol_version") != ACTION_PROTOCOL
            or affect_row.get("affect_protocol_version") != AFFECT_PROTOCOL
            or arc_row.get("audio_available") is not False
            or arc_row.get("label_metadata_exposed") is not False
        ):
            raise ValueError(f"{sample_id}: invalid blind queue protocol")
        if artifact_kind == EXPANSION_BUNDLE_KIND:
            frames = arc_row.get("frame_count")
            if (
                arc_row.get("native_duration_preserved") is not True
                or arc_row.get("fixed_duration_window_used") is not False
                or affect_row.get("native_duration_preserved") is not True
                or affect_row.get("fixed_duration_window_used") is not False
                or isinstance(frames, bool)
                or not isinstance(frames, int)
                or frames < 3
                or affect_row.get("frame_count") != frames
                or not math.isclose(float(arc_row.get("fps", 0.0)), 30.0, abs_tol=1e-9)
                or affect_row.get("fps") != arc_row.get("fps")
            ):
                raise ValueError(f"{sample_id}: expanded queue changed native duration")
        video = Path(str(arc_row.get("video_path") or "")).resolve()
        if not video.is_file() or sha256(video) != arc_row.get("video_sha256"):
            raise ValueError(f"{sample_id}: anonymous video hash mismatch")
        _reject_hidden_labels(arc_row, context=f"{sample_id}/arc_queue")
        _reject_hidden_labels(affect_row, context=f"{sample_id}/affect_queue")
    return SUPPORTED_BUNDLES[str(artifact_kind)], arc_path, affect_path, arc


def _validate_common(row: dict[str, Any], queue: dict[str, Any], *, role: str) -> None:
    _reject_hidden_labels(row, context=f"{queue['sample_id']}/{role}")
    for key in ("sample_id", "video_sha256", "context_level", "audio_available", "label_metadata_exposed"):
        if row.get(key) != queue.get(key):
            raise ValueError(f"{queue['sample_id']}/{role}: {key} binding mismatch")
    submitted_video = Path(str(row.get("video_path") or "")).resolve()
    queued_video = Path(str(queue.get("video_path") or "")).resolve()
    if submitted_video != queued_video:
        raise ValueError(f"{queue['sample_id']}/{role}: video_path binding mismatch")


def _load_optional_index(path: Path | None, key: str, *, context: str) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    return _index_unique(read_jsonl(path.resolve()), key, context=context)


def _validate_author(
    row: dict[str, Any], queue: dict[str, Any], *, frames: int
) -> dict[str, Any]:
    sample_id = str(queue["sample_id"])
    _validate_common(row, queue, role="author")
    if row.get("arc_protocol_version") != ARC_PROTOCOL or row.get("action_protocol_version") != ACTION_PROTOCOL:
        raise ValueError(f"{sample_id}/author: protocol mismatch")
    arc_id = row.get("arc_review_id")
    action_id = row.get("action_review_id")
    arc_reviewer = row.get("arc_reviewer_id")
    action_reviewer = row.get("action_reviewer_id")
    if not all(isinstance(value, str) and value for value in (arc_id, action_id, arc_reviewer, action_reviewer)):
        raise ValueError(f"{sample_id}/author: review IDs and reviewers are required")
    if arc_id == action_id:
        raise ValueError(f"{sample_id}/author: arc/action review IDs must differ")

    evidence: dict[str, int] = {}
    phases: dict[str, dict[str, Any]] = {}
    for phase in ("onset", "apex", "offset"):
        status = row.get(f"{phase}_status")
        frame = row.get(f"{phase}_evidence_frame")
        basis = row.get(f"{phase}_basis")
        if status not in PHASE_STATUSES:
            raise ValueError(f"{sample_id}/author: invalid {phase} status")
        if frame is not None and (
            isinstance(frame, bool) or not isinstance(frame, int) or not 0 <= frame < frames
        ):
            raise ValueError(f"{sample_id}/author: {phase} frame is outside video")
        if status == "complete":
            if frame is None or basis not in COMPLETE_PHASE_BASES[phase]:
                raise ValueError(f"{sample_id}/author: complete {phase} lacks evidence")
            evidence[phase] = int(frame)
        phases[phase] = {"status": status, "evidence_frame": frame, "basis": basis}
    if set(evidence) == {"onset", "apex", "offset"} and not (
        evidence["onset"] < evidence["apex"] < evidence["offset"]
    ):
        raise ValueError(f"{sample_id}/author: arc evidence is not ordered")

    action_result = row.get("action_result")
    if action_result not in ACTION_RESULTS:
        raise ValueError(f"{sample_id}/author: invalid action result")
    candidate = row.get("candidate_text")
    candidate_hash = row.get("candidate_text_sha256")
    description = row.get("observable_description")
    text_valid = bool(
        isinstance(candidate, str)
        and candidate.strip()
        and row.get("candidate_text_provenance") == ACTION_TEXT_PROVENANCE
        and candidate_hash == text_sha256(candidate)
        and isinstance(description, str)
        and description.strip()
    )
    if any(value is not None for value in (candidate, candidate_hash, description)) and not text_valid:
        raise ValueError(f"{sample_id}/author: candidate text/hash/provenance mismatch")
    return {
        "arc_review_id": arc_id,
        "arc_reviewer_id": arc_reviewer,
        "action_review_id": action_id,
        "action_reviewer_id": action_reviewer,
        "action_result": action_result,
        "phases": phases,
        "candidate_text": candidate if text_valid else None,
        "candidate_text_sha256": candidate_hash if text_valid else None,
        "candidate_text_provenance": row.get("candidate_text_provenance") if text_valid else None,
        "observable_description": description if text_valid else None,
        "text_valid": text_valid,
        "submission_sha256": value_sha256(row),
    }


def _validate_matcher(
    row: dict[str, Any], queue: dict[str, Any], author: dict[str, Any]
) -> dict[str, Any]:
    sample_id = str(queue["sample_id"])
    _validate_common(row, queue, role="matcher")
    protocol = row.get("action_protocol_version")
    if protocol not in {ACTION_PROTOCOL, NATIVE_MATCHER_PROTOCOL}:
        raise ValueError(f"{sample_id}/matcher: protocol mismatch")
    author_binding = row.get("author_action_review_id")
    if author_binding is not None and author_binding != author["action_review_id"]:
        raise ValueError(f"{sample_id}/matcher: author review binding mismatch")
    if row.get("candidate_text") != author["candidate_text"] or row.get("candidate_text_sha256") != author["candidate_text_sha256"]:
        raise ValueError(f"{sample_id}/matcher: candidate text binding mismatch")
    if row.get("candidate_text_provenance", ACTION_TEXT_PROVENANCE) != ACTION_TEXT_PROVENANCE:
        raise ValueError(f"{sample_id}/matcher: candidate text provenance mismatch")
    review_id = row.get("action_match_review_id") or row.get("action_review_id")
    reviewer_id = row.get("action_match_reviewer_id") or row.get("action_reviewer_id")
    if not isinstance(review_id, str) or not review_id or not isinstance(reviewer_id, str) or not reviewer_id:
        raise ValueError(f"{sample_id}/matcher: review ID/reviewer are required")
    if review_id in {author["arc_review_id"], author["action_review_id"]} or reviewer_id in {
        author["arc_reviewer_id"], author["action_reviewer_id"]
    }:
        raise ValueError(f"{sample_id}/matcher: matcher is not independent")
    result = row.get("action_result")
    if result not in ACTION_RESULTS:
        raise ValueError(f"{sample_id}/matcher: invalid action result")
    description = row.get("observable_description")
    if description is not None and (not isinstance(description, str) or not description.strip()):
        raise ValueError(f"{sample_id}/matcher: invalid observable description")
    return {
        "review_id": review_id,
        "reviewer_id": reviewer_id,
        "result": result,
        # R2 judges the R1 candidate text.  The formal prompt remains exactly
        # that bound candidate, never a matcher-written paraphrase.
        "observable_description": author["candidate_text"] if result == "observable_match" else None,
        "author_action_review_id": author["action_review_id"],
        "candidate_text_sha256": author["candidate_text_sha256"],
        "protocol_version": protocol,
        "submission_sha256": value_sha256(row),
    }


def _validate_affect(row: dict[str, Any], queue: dict[str, Any]) -> dict[str, Any]:
    sample_id = str(queue["sample_id"])
    _validate_common(row, queue, role="affect")
    if row.get("affect_protocol_version") != AFFECT_PROTOCOL:
        raise ValueError(f"{sample_id}/affect: protocol mismatch")
    review_id = row.get("affect_review_id")
    reviewer_id = row.get("affect_reviewer_id")
    result = row.get("result")
    predicted = row.get("predicted_class")
    confidence = row.get("confidence")
    if not isinstance(review_id, str) or not review_id or not isinstance(reviewer_id, str) or not reviewer_id:
        raise ValueError(f"{sample_id}/affect: review ID/reviewer are required")
    if result not in AFFECT_RESULTS:
        raise ValueError(f"{sample_id}/affect: invalid result")
    if result == "observable":
        if predicted not in AFFECT_CLASSES or isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)) or not 0 <= float(confidence) <= 1:
            raise ValueError(f"{sample_id}/affect: invalid observable outcome")
        confidence = float(confidence)
    elif predicted is not None or confidence is not None:
        raise ValueError(f"{sample_id}/affect: non-observable result carries a class/confidence")
    return {
        "review_id": review_id,
        "reviewer_id": reviewer_id,
        "result": result,
        "predicted_class": predicted,
        "confidence": confidence,
        "submission_sha256": value_sha256(row),
    }


def _validate_adjudication(
    row: dict[str, Any], queue: dict[str, Any], reviews: list[dict[str, Any]]
) -> dict[str, Any]:
    sample_id = str(queue["sample_id"])
    _validate_common(row, queue, role="affect_adjudicator")
    if row.get("affect_adjudication_protocol_version") != AFFECT_ADJUDICATION_PROTOCOL:
        raise ValueError(f"{sample_id}/adjudicator: protocol mismatch")
    expected = {review["review_id"] for review in reviews}
    input_ids = row.get("input_review_ids")
    if not isinstance(input_ids, list) or len(input_ids) != len(set(input_ids)) or set(input_ids) != expected:
        raise ValueError(f"{sample_id}/adjudicator: incomplete input review binding")
    adapted = {
        **row,
        "affect_protocol_version": AFFECT_PROTOCOL,
        "affect_review_id": row.get("affect_adjudication_review_id"),
        "affect_reviewer_id": row.get("affect_adjudication_reviewer_id"),
    }
    outcome = _validate_affect(adapted, queue)
    outcome["input_review_ids"] = sorted(expected)
    outcome["submission_sha256"] = value_sha256(row)
    return outcome


def _trajectory_values(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != list(JOINT_ORDER_18D):
            raise ValueError(f"trajectory is not ordered ULA V2 18D: {path}")
        values: list[list[float]] = []
        for line_number, row in enumerate(reader, 2):
            if len(row) != len(JOINT_ORDER_18D):
                raise ValueError(f"trajectory row {line_number} is not 18D: {path}")
            try:
                parsed = [float(value) for value in row]
            except ValueError as error:
                raise ValueError(f"trajectory row {line_number} is not numeric: {path}") from error
            if not all(math.isfinite(value) for value in parsed):
                raise ValueError(f"trajectory row {line_number} is not finite: {path}")
            values.append(parsed)
    if len(values) < 3:
        raise ValueError(f"trajectory has no frames: {path}")
    return np.asarray(values, dtype=np.float64)


def _trajectory_frames(path: Path) -> int:
    return int(_trajectory_values(path).shape[0])


def _index_catalog(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path.resolve()):
        record_hash = row.get("record_sha256")
        if not _is_sha256(record_hash):
            continue
        payload = dict(row)
        payload.pop("record_sha256", None)
        if value_sha256(payload) != record_hash:
            raise ValueError(f"source catalog record hash mismatch: {row.get('clip_id')}")
        if record_hash in result:
            raise ValueError(f"duplicate source catalog record: {record_hash}")
        result[str(record_hash)] = row
    return result


_MUJOCO_MODEL_CACHE: dict[str, tuple[Any, Any, list[int]]] = {}


def _native_collision_metrics(
    trajectory: np.ndarray, *, urdf: Path
) -> dict[str, Any]:
    """Measure the same cross-arm/torso contacts used by the GMR QC."""

    import mujoco

    urdf = urdf.resolve()
    if not urdf.is_file():
        raise ValueError(f"native collision model is missing: {urdf}")
    key = str(urdf)
    cached = _MUJOCO_MODEL_CACHE.get(key)
    if cached is None:
        model = mujoco.MjModel.from_xml_path(key)
        addresses = []
        for name in JOINT_ORDER_18D:
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if joint_id < 0:
                raise ValueError(f"native collision model has no joint {name}")
            addresses.append(int(model.jnt_qposadr[joint_id]))
        cached = (mujoco, model, addresses)
        _MUJOCO_MODEL_CACHE[key] = cached
    mujoco_module, model, addresses = cached
    data = mujoco_module.MjData(model)

    def group(body_name: str) -> str | None:
        if body_name.startswith("link_l") and any(
            token in body_name for token in ("Shoulder", "Elbow", "Wrist")
        ):
            return "left_arm"
        if body_name.startswith("link_r") and any(
            token in body_name for token in ("Shoulder", "Elbow", "Wrist")
        ):
            return "right_arm"
        if body_name in {"link_torso", "link_pelvisYaw", "link_pelvisPitch"}:
            return "torso"
        return None

    collision_frames = 0
    pairs: Counter[str] = Counter()
    for row in trajectory:
        data.qpos[:] = 0.0
        data.qpos[addresses] = row
        mujoco_module.mj_forward(model, data)
        frame_has_collision = False
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            body1 = int(model.geom_bodyid[contact.geom1])
            body2 = int(model.geom_bodyid[contact.geom2])
            name1 = (
                mujoco_module.mj_id2name(
                    model, mujoco_module.mjtObj.mjOBJ_BODY, body1
                )
                or "world"
            )
            name2 = (
                mujoco_module.mj_id2name(
                    model, mujoco_module.mjtObj.mjOBJ_BODY, body2
                )
                or "world"
            )
            group1, group2 = group(name1), group(name2)
            if group1 is not None and group2 is not None and group1 != group2:
                pairs[" <-> ".join(sorted((name1, name2)))] += 1
                frame_has_collision = True
        collision_frames += int(frame_has_collision)
    return {
        "upper_body_collision_frames": collision_frames,
        "upper_body_collision_frame_rate": collision_frames / len(trajectory),
        "upper_body_collision_pairs": dict(pairs.most_common(20)),
        "collision_frame_rate_max": COLLISION_FRAME_RATE_MAX,
    }


def _native_physical_evidence(
    *,
    sample_id: str,
    motion: dict[str, Any],
    catalog: dict[str, Any],
    trajectory: np.ndarray,
    trajectory_sha256: str,
    fps: float,
) -> tuple[dict[str, bool], dict[str, Any]]:
    physical = catalog.get("physical_qc")
    if not isinstance(physical, dict) or physical.get("passed") is not True:
        raise ValueError(f"{sample_id}: native source physical QC is not passed")
    lower = np.asarray([JOINT_LIMITS_18D[name][0] for name in JOINT_ORDER_18D])
    upper = np.asarray([JOINT_LIMITS_18D[name][1] for name in JOINT_ORDER_18D])
    limit_violation = np.maximum(
        np.maximum(lower[None, :] - trajectory, 0.0),
        np.maximum(trajectory - upper[None, :], 0.0),
    )
    velocity = np.abs(np.diff(trajectory, axis=0) * fps)
    head_indices = [JOINT_ORDER_18D.index(name) for name in JOINT_ORDER_18D[-3:]]
    timing = physical.get("timing") or {}
    video = motion.get("video_check") or {}
    configured_urdf = (motion.get("render_config") or {}).get("urdf")
    urdf = Path(str(configured_urdf)).resolve() if configured_urdf else DEFAULT_ROBOT_URDF
    collision = _native_collision_metrics(trajectory, urdf=urdf)
    max_velocity = float(np.max(velocity, initial=0.0))
    head_max_velocity = float(np.max(velocity[:, head_indices], initial=0.0))
    max_limit_violation = float(np.max(limit_violation, initial=0.0))
    head_max_limit_violation = float(
        np.max(limit_violation[:, head_indices], initial=0.0)
    )
    safe_projection_pass = bool(
        physical.get("safe_projection_pass") is True
        and float(physical.get("max_safe_projection_rad", math.inf))
        <= float(physical.get("safe_projection_threshold_rad", 0.01)) + 1e-12
    )
    gates: dict[str, bool] = {
        "joint_limits_pass": max_limit_violation <= 1e-8,
        "velocity_pass": max_velocity <= NATIVE_MAX_VELOCITY_RAD_S + 1e-6,
        "timing_pass": bool(
            timing.get("starts_at_zero") is True
            and timing.get("strictly_increasing") is True
            and timing.get("native_30hz") is True
        ),
        "safe_projection_pass": safe_projection_pass,
        "head_joint_limits_pass": head_max_limit_violation <= 1e-8,
        "head_velocity_pass": head_max_velocity <= NATIVE_MAX_VELOCITY_RAD_S + 1e-6,
        "collision_pass": float(collision["upper_body_collision_frame_rate"])
        <= COLLISION_FRAME_RATE_MAX + 1e-12,
        "video_decode_pass": bool(video.get("fully_decodable") is True),
        "video_frame_count_pass": bool(
            video.get("decoded_frames") == len(trajectory)
            and video.get("expected_frames") == len(trajectory)
        ),
        "video_nonblank_pass": bool(video.get("nonblank") is True),
    }
    gates["passed"] = all(gates.values())
    if set(gates) != NATIVE_ROBOT_REQUIRED_18D_GATES or not all(gates.values()):
        failed = sorted(name for name, passed in gates.items() if not passed)
        raise ValueError(f"{sample_id}: native physical evidence failed: {failed}")
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "ula0513_native_full_asset_physical_evidence_v1",
        "sample_id": sample_id,
        "physical_evidence_profile": NATIVE_ROBOT_PHYSICAL_PROFILE,
        "trajectory_sha256": trajectory_sha256,
        "frames": int(len(trajectory)),
        "fps": fps,
        "joint_limit_max_violation_rad": max_limit_violation,
        "max_velocity_rad_s": max_velocity,
        "velocity_limit_rad_s": NATIVE_MAX_VELOCITY_RAD_S,
        "head_joint_limit_max_violation_rad": head_max_limit_violation,
        "head_max_velocity_rad_s": head_max_velocity,
        "source_physical_qc_record_sha256": value_sha256(physical),
        "robot_urdf_sha256": sha256(urdf),
        **collision,
        "video_evidence": {
            "video_sha256": motion.get("video_sha256"),
            "decoded_frames": video.get("decoded_frames"),
            "expected_frames": video.get("expected_frames"),
            "fully_decodable": video.get("fully_decodable"),
            "nonblank": video.get("nonblank"),
        },
        "quality_gate": gates,
    }
    evidence["record_sha256"] = value_sha256(evidence)
    return gates, evidence


def _human_physical_evidence(
    *,
    sample_id: str,
    motion: dict[str, Any],
    trajectory: Path,
    trajectory_sha256: str,
    frames: int,
    fps: float,
) -> tuple[dict[str, bool], dict[str, Any], dict[str, Any]]:
    declared_quality = motion.get("quality_json")
    quality_path = (
        Path(str(declared_quality)).resolve()
        if declared_quality
        else trajectory.parent / "quality.json"
    )
    if quality_path.is_file():
        quality = read_json(quality_path)
        if quality.get("safe_csv_sha256") != trajectory_sha256:
            raise ValueError(f"{sample_id}: retarget quality is not bound to final CSV")
        quality_file_sha256 = sha256(quality_path)
    else:
        inline = motion.get("quality_gate")
        if not isinstance(inline, dict):
            raise ValueError(f"{sample_id}: human retarget quality record is missing")
        quality = {
            "artifact_kind": "inline_test_human_retarget_quality",
            "quality_gate": inline,
            "safe_csv_sha256": trajectory_sha256,
            "frames": frames,
        }
        quality_file_sha256 = value_sha256(quality)
        quality_path = None
    raw_gates = quality.get("quality_gate")
    if not isinstance(raw_gates, dict):
        raise ValueError(f"{sample_id}: human retarget has no quality gates")
    gates = {name: raw_gates.get(name) is True for name in REQUIRED_18D_GATES}
    if set(gates) != REQUIRED_18D_GATES or not all(gates.values()):
        failed = sorted(name for name, passed in gates.items() if not passed)
        raise ValueError(f"{sample_id}: human retarget physical evidence failed: {failed}")
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "beat2_expression_turn_human_retarget_physical_evidence_v1",
        "sample_id": sample_id,
        "physical_evidence_profile": HUMAN_RETARGET_PHYSICAL_PROFILE,
        "trajectory_sha256": trajectory_sha256,
        "frames": frames,
        "fps": fps,
        "quality_record_path": str(quality_path) if quality_path else None,
        "quality_record_file_sha256": quality_file_sha256,
        "quality_record_sha256": value_sha256(quality),
        "quality_gate": gates,
    }
    evidence["record_sha256"] = value_sha256(evidence)
    return gates, evidence, quality


def _motion_evidence(
    *,
    sample_id: str,
    queue: dict[str, Any],
    hidden: dict[str, Any],
    motion: dict[str, Any],
    source_catalog: dict[str, dict[str, Any]],
    source_catalog_sha256: str | None,
) -> dict[str, Any]:
    expected_motion_hash = hidden.get("source_render_record_sha256") or hidden.get("render_record_sha256")
    if not _is_sha256(expected_motion_hash) or value_sha256(motion) != expected_motion_hash:
        raise ValueError(f"{sample_id}: hidden-to-motion record hash mismatch")
    if motion.get("status") != "passed" or motion.get("accepted_for_training") is not False:
        raise ValueError(f"{sample_id}: motion record is not a closed render pass")
    if motion.get("video_sha256") != queue.get("video_sha256"):
        raise ValueError(f"{sample_id}: motion/public video hash mismatch")

    trajectory_value = motion.get("trajectory_path") or motion.get("safe_csv")
    trajectory_hash = motion.get("trajectory_sha256") or motion.get("safe_csv_sha256")
    trajectory = Path(str(trajectory_value or "")).resolve()
    if not trajectory.is_file() or not _is_sha256(trajectory_hash) or sha256(trajectory) != trajectory_hash:
        raise ValueError(f"{sample_id}: final 18D trajectory hash mismatch")
    frames = motion.get("trajectory_frames") or motion.get("output_frame_count")
    trajectory_values = _trajectory_values(trajectory)
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 3 or len(trajectory_values) != frames:
        raise ValueError(f"{sample_id}: final 18D frame count mismatch")
    if hidden.get("frame_count") is not None and hidden.get("frame_count") != frames:
        raise ValueError(f"{sample_id}: hidden frame count lineage mismatch")
    if hidden.get("trajectory_sha256") is not None and hidden.get("trajectory_sha256") != trajectory_hash:
        raise ValueError(f"{sample_id}: hidden trajectory lineage mismatch")

    training = motion.get("training_segment")
    retarget = motion.get("retarget_segment")
    if not isinstance(training, dict) or not isinstance(retarget, dict):
        raise ValueError(f"{sample_id}: missing variable-length motion contracts")
    if (
        training.get("frame_count") is not None
        and retarget.get("source_frame_count") != training.get("frame_count")
    ) or retarget.get("output_frame_count") != frames:
        raise ValueError(f"{sample_id}: source/output frame lineage mismatch")
    for contract in (training, retarget):
        if contract.get("cropped") is not False:
            raise ValueError(f"{sample_id}: cropped motion is forbidden")
        for key in ("fixed_window_sec", "fixed_target_duration_sec", "target_duration_sec"):
            if contract.get(key) is not None:
                raise ValueError(f"{sample_id}: fixed/target duration is forbidden")
    fps = retarget.get("fps") or motion.get("fps") or motion.get("render_config", {}).get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isclose(float(fps), 30.0, abs_tol=1e-9):
        raise ValueError(f"{sample_id}: motion fps is not 30Hz")
    fps = float(fps)

    catalog = source_catalog.get(str(hidden.get("source_record_sha256")))
    native = catalog is not None
    if native:
        if not _is_sha256(source_catalog_sha256):
            raise ValueError(f"{sample_id}: native source catalog SHA256 is missing")
        quality_gate, physical_evidence = _native_physical_evidence(
            sample_id=sample_id,
            motion=motion,
            catalog=catalog,
            trajectory=trajectory_values,
            trajectory_sha256=str(trajectory_hash),
            fps=fps,
        )
        catalog_record_sha = str(catalog["record_sha256"])
        split_assignment_sha = value_sha256(
            {
                "artifact_kind": "ula0513_native_train_only_split_assignment_v1",
                "source_catalog_sha256": source_catalog_sha256,
                "fixed_split_assignment": motion.get("fixed_split_assignment") or "train",
            }
        )
        source = catalog.get("source") or {}
        lineage_fields = {
            "expression_turn_contract_sha256": value_sha256(contract_definition()),
            "source_inventory_manifest_sha256": source_catalog_sha256,
            "split_assignment_manifest_sha256": split_assignment_sha,
            "inventory_record_sha256": catalog_record_sha,
            "upstream_inventory_record_sha256": catalog_record_sha,
            "selected_record_sha256": catalog_record_sha,
            "retarget_input_manifest_sha256": source_catalog_sha256,
            "retarget_quality_record_sha256": physical_evidence["record_sha256"],
            "trajectory_sha256": trajectory_hash,
            "source_sha256": source.get("csv_sha256"),
        }
        anonymous_source_key = str(motion.get("task_id") or sample_id)
        provenance = {
            "source_clip_id": anonymous_source_key,
            "speaker_key": anonymous_source_key,
            "source_group_key": anonymous_source_key,
            "dataset_source": catalog.get("dataset") or "ULA0513_user_provided_robot_motion",
        }
        profile = NATIVE_ROBOT_PHYSICAL_PROFILE
        quality_record = None
    else:
        quality_gate, physical_evidence, quality_record = _human_physical_evidence(
            sample_id=sample_id,
            motion=motion,
            trajectory=trajectory,
            trajectory_sha256=str(trajectory_hash),
            frames=int(frames),
            fps=fps,
        )
        lineage_fields = {
            "expression_turn_contract_sha256": quality_record.get("expression_turn_contract_sha256")
            or motion.get("expression_turn_contract_sha256"),
            "source_inventory_manifest_sha256": quality_record.get("source_inventory_manifest_sha256")
            or motion.get("source_inventory_manifest_sha256"),
            "split_assignment_manifest_sha256": quality_record.get("split_assignment_manifest_sha256")
            or motion.get("split_assignment_manifest_sha256"),
            "inventory_record_sha256": quality_record.get("inventory_record_sha256")
            or motion.get("expression_turn_selection_record_sha256")
            or motion.get("expression_turn_record_sha256"),
            "upstream_inventory_record_sha256": quality_record.get("upstream_inventory_record_sha256")
            or motion.get("upstream_inventory_record_sha256"),
            "selected_record_sha256": quality_record.get("selected_record_sha256")
            or motion.get("selected_record_sha256"),
            "retarget_input_manifest_sha256": quality_record.get("retarget_input_manifest_sha256")
            or motion.get("retarget_input_manifest_sha256"),
            "retarget_quality_record_sha256": physical_evidence["record_sha256"],
            "trajectory_sha256": trajectory_hash,
            "source_sha256": quality_record.get("source_sha256")
            or quality_record.get("motion_sha256")
            or motion.get("source_sha256"),
        }
        provenance = {
            "source_clip_id": quality_record.get("source_clip_id") or motion.get("source_clip_id"),
            "speaker_key": quality_record.get("speaker_key") or motion.get("speaker_key"),
            "source_group_key": quality_record.get("source_group_key")
            or quality_record.get("source_clip_id")
            or motion.get("source_clip_id"),
            "dataset_source": quality_record.get("source_dataset")
            or quality_record.get("dataset_subset")
            or "BEAT2_human_motion",
        }
        profile = HUMAN_RETARGET_PHYSICAL_PROFILE
    invalid_lineage = sorted(
        key for key, value in lineage_fields.items() if not _is_sha256(value)
    )
    if invalid_lineage:
        raise ValueError(f"{sample_id}: incomplete physical lineage: {invalid_lineage}")
    if any(not str(value or "").strip() for value in provenance.values()):
        raise ValueError(f"{sample_id}: incomplete source provenance")
    return {
        "task_id": motion.get("task_id"),
        "trajectory": str(trajectory),
        "trajectory_sha256": trajectory_hash,
        "frames": frames,
        "fps": fps,
        "quality_gate": quality_gate,
        "physical_evidence": physical_evidence,
        "physical_evidence_profile": profile,
        "training_segment": training,
        "retarget_segment": retarget,
        "safety_monotonic_retime": motion.get("safety_monotonic_retime"),
        "core_interval": motion.get("core_interval")
        or (quality_record or {}).get("core_interval"),
        "context_plan": motion.get("context_plan")
        or (quality_record or {}).get("context_plan"),
        "fixed_split_assignment": motion.get("fixed_split_assignment") or "train",
        "motion_record_sha256": str(expected_motion_hash),
        "lineage_fields": lineage_fields,
        "provenance": provenance,
    }


def _affect_qualification(
    reviews: list[dict[str, Any]], adjudication: dict[str, Any] | None
) -> dict[str, Any]:
    if adjudication is not None and adjudication["result"] == "observable" and float(adjudication["confidence"]) >= MIN_AFFECT_CONFIDENCE:
        return {
            "eligible": True,
            "status": "qualified_independent_blind_adjudication",
            "basis": "independent_blind_adjudication",
            "blind_affect_class": adjudication["predicted_class"],
            "confidence": adjudication["confidence"],
        }
    consensus = bool(
        len(reviews) >= 2
        and all(
            review["result"] == "observable"
            and float(review["confidence"]) >= MIN_AFFECT_CONFIDENCE
            for review in reviews
        )
        and len({review["predicted_class"] for review in reviews}) == 1
    )
    if consensus:
        return {
            "eligible": True,
            "status": "qualified_independent_blind_consensus",
            "basis": "independent_blind_review_consensus",
            "blind_affect_class": reviews[0]["predicted_class"],
            "confidence": min(float(review["confidence"]) for review in reviews),
        }
    status = "pending_two_independent_affect_reviews" if len(reviews) < 2 else "conflicting_or_insufficient_affect_consensus"
    return {
        "eligible": False,
        "status": status,
        "basis": None,
        "blind_affect_class": None,
        "confidence": None,
    }


def _review_record(
    *,
    dataset_kind: str,
    queue: dict[str, Any],
    motion: dict[str, Any],
    author: dict[str, Any] | None,
    matcher: dict[str, Any] | None,
    affects: list[dict[str, Any]],
    adjudication: dict[str, Any] | None,
) -> dict[str, Any]:
    training = motion["training_segment"]
    start = int(training["start_frame"])
    end = int(training["end_frame_exclusive"])
    context_plan = motion.get("context_plan")
    core_interval = motion.get("core_interval")
    if context_plan is None:
        context_plan = {
            "policy": CONTEXT_POLICY,
            "same_source_only": True,
            "neighbor_crossing_allowed": False,
            "source_interval": {"start_frame": start, "end_frame_exclusive": end},
            "admissible_interval": {"start_frame": start, "end_frame_exclusive": end},
            "levels": [
                {
                    "level": 0,
                    "start_frame": start,
                    "end_frame_exclusive": end,
                    "left_boundary_basis": "complete_source_authored_asset_start",
                    "right_boundary_basis": "complete_source_authored_asset_end",
                }
            ],
            "selected_level": 0,
        }
    else:
        context_plan = json.loads(stable_json(context_plan))
    if core_interval is None:
        core_interval = {"start_frame": start, "end_frame_exclusive": end}
    else:
        core_interval = json.loads(stable_json(core_interval))
    selected_level = context_plan.get("selected_level")
    if selected_level != queue.get("context_level"):
        raise ValueError(
            f"{queue['sample_id']}: blind review context differs from selected interval"
        )

    def source_frame(output_frame: object) -> int | None:
        if output_frame is None:
            return None
        output_index = int(output_frame)
        source_frames = end - start
        output_frames = int(motion["frames"])
        if source_frames == output_frames:
            return start + output_index
        audit = motion.get("safety_monotonic_retime")
        time_map = audit.get("input_frame_output_times_sec") if isinstance(audit, dict) else None
        if not isinstance(time_map, list) or len(time_map) != source_frames:
            raise ValueError(
                f"{queue['sample_id']}: retimed blind evidence has no full source time map"
            )
        target_time = output_index / float(motion["fps"])
        right = bisect.bisect_left(time_map, target_time)
        candidates = [index for index in (right - 1, right) if 0 <= index < len(time_map)]
        relative = min(candidates, key=lambda index: abs(float(time_map[index]) - target_time))
        return start + relative

    arc_review = None
    action_author_review = None
    if author is not None:
        arc_review = {
            "protocol_version": ARC_PROTOCOL,
            "review_id": author["arc_review_id"],
            "reviewer_id": author["arc_reviewer_id"],
            "anonymous_video_sha256": queue["video_sha256"],
            "context_level": selected_level,
            "audio_available": False,
            "label_metadata_exposed": False,
            **{
                phase: {
                    "status": author["phases"][phase]["status"],
                    "evidence_frame": source_frame(
                        author["phases"][phase]["evidence_frame"]
                    ),
                    "basis": author["phases"][phase]["basis"],
                }
                for phase in ("onset", "apex", "offset")
            },
            "submission_sha256": author["submission_sha256"],
        }
        action_author_review = {
            "protocol_version": ACTION_PROTOCOL,
            "review_id": author["action_review_id"],
            "reviewer_id": author["action_reviewer_id"],
            "anonymous_video_sha256": queue["video_sha256"],
            "context_level": selected_level,
            "audio_available": False,
            "label_metadata_exposed": False,
            "result": author["action_result"],
            "candidate_text": author["candidate_text"],
            "candidate_text_sha256": author["candidate_text_sha256"],
            "candidate_text_provenance": author["candidate_text_provenance"],
            "observable_description": author["observable_description"],
            "submission_sha256": author["submission_sha256"],
        }
    else:
        arc_review = {
            "protocol_version": ARC_PROTOCOL,
            "review_id": f"{queue['sample_id']}__pending_arc_review",
            "reviewer_id": "pending_no_blind_arc_submission",
            "anonymous_video_sha256": queue["video_sha256"],
            "context_level": selected_level,
            "audio_available": False,
            "label_metadata_exposed": False,
            **{
                phase: {
                    "status": "pending",
                    "evidence_frame": None,
                    "basis": None,
                }
                for phase in ("onset", "apex", "offset")
            },
        }

    action_review = None
    if matcher is not None:
        action_review = {
            "protocol_version": ACTION_PROTOCOL,
            "review_id": matcher["review_id"],
            "reviewer_id": matcher["reviewer_id"],
            "anonymous_video_sha256": queue["video_sha256"],
            "context_level": selected_level,
            "audio_available": False,
            "label_metadata_exposed": False,
            "result": matcher["result"],
            "candidate_text_sha256": matcher["candidate_text_sha256"],
            "candidate_text_provenance": ACTION_TEXT_PROVENANCE,
            "observable_description": matcher["observable_description"],
            "author_action_review_id": matcher["author_action_review_id"],
            "author_submission_sha256": author["submission_sha256"] if author else None,
            "matcher_submission_sha256": matcher["submission_sha256"],
        }

    affect_reviews = [
        {
            "protocol_version": AFFECT_PROTOCOL,
            "review_id": affect["review_id"],
            "reviewer_id": affect["reviewer_id"],
            "anonymous_video_sha256": queue["video_sha256"],
            "context_level": selected_level,
            "audio_available": False,
            "label_metadata_exposed": False,
            "result": affect["result"],
            "predicted_class": affect["predicted_class"],
            "confidence": affect["confidence"],
            "submission_sha256": affect["submission_sha256"],
        }
        for affect in affects
    ]
    affect_adjudication = None
    if adjudication is not None:
        affect_adjudication = {
            "protocol_version": AFFECT_ADJUDICATION_PROTOCOL,
            "review_id": adjudication["review_id"],
            "reviewer_id": adjudication["reviewer_id"],
            "anonymous_video_sha256": queue["video_sha256"],
            "context_level": selected_level,
            "audio_available": False,
            "label_metadata_exposed": False,
            "input_review_ids": adjudication["input_review_ids"],
            "result": adjudication["result"],
            "predicted_class": adjudication["predicted_class"],
            "confidence": adjudication["confidence"],
            "submission_sha256": adjudication["submission_sha256"],
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": EXPRESSION_REVIEW_ARTIFACT_KIND,
        "contract_version": EXPRESSION_TURN_CONTRACT_VERSION,
        "dataset_kind": dataset_kind,
        "sample_id": queue["sample_id"],
        "clip_id": str(motion["task_id"]),
        "core_interval": core_interval,
        "context_plan": context_plan,
        "physical_qc": {
            "passed": True,
            "physical_evidence_profile": motion["physical_evidence_profile"],
            "physical_evidence_record_sha256": motion["physical_evidence"]["record_sha256"],
        },
        "motion_arc_review": arc_review,
        "action_text_author_review": action_author_review,
        "action_semantic_review": action_review,
        "affect_reviews": affect_reviews,
        "affect_adjudication": affect_adjudication,
        "official_action_or_emotion_used_for_qualification": False,
    }


def _normalized_retarget_segment(motion: dict[str, Any]) -> dict[str, Any]:
    training = motion["training_segment"]
    start = int(training["start_frame"])
    end = int(training["end_frame_exclusive"])
    source_frames = end - start
    output_frames = int(motion["frames"])
    fps = float(motion["fps"])
    representation = (
        NATIVE_ROBOT_RETARGET_SEGMENT_REPRESENTATION
        if motion["physical_evidence_profile"] == NATIVE_ROBOT_PHYSICAL_PROFILE
        else RETARGET_SEGMENT_REPRESENTATION
    )
    payload = {
        "representation": representation,
        "source_start_frame": start,
        "source_end_frame_exclusive": end,
        "source_frame_count": source_frames,
        "output_frame_count": output_frames,
        "fps": fps,
        "retimed": output_frames != source_frames,
        "cropped": False,
        "planner_duration_field": "output_sample_span_sec",
        "source_boundary_duration_field": "source_frame_coverage_sec",
        "legacy_quality_duration_sec_role": (
            "output_frame_coverage_compatibility_only_not_planner_target"
        ),
        "source_frame_coverage_sec": source_frames / fps,
        "output_sample_span_sec": max(0, output_frames - 1) / fps,
        "output_frame_coverage_sec": output_frames / fps,
    }
    return {**payload, "sha256": ascii_value_sha256(payload)}


def _stage_trajectory(source: str, destination: Path, expected_sha256: str) -> str:
    source_path = Path(source).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256(destination) == expected_sha256:
        return str(destination.resolve())
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copyfile(source_path, temporary)
    if sha256(temporary) != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError("staged trajectory SHA256 changed")
    os.replace(temporary, destination)
    return str(destination.resolve())


def _train_ready(
    *,
    dataset_kind: str,
    sample_id: str,
    motion: dict[str, Any],
    review: dict[str, Any],
    review_sha: str,
    qualification: dict[str, Any],
    qualification_sha: str,
) -> dict[str, Any]:
    highest = str(qualification["highest_qualification"])
    semantic_enabled = bool(
        qualification["qualifications"]["semantic_conditioning"]["eligible"]
    )
    affect_enabled = bool(
        qualification["qualifications"]["expressive_conditioning"]["eligible"]
    )
    action_review = review.get("action_semantic_review") or {}
    prompt = action_review.get("observable_description") if semantic_enabled else None
    emotion_id = qualification.get("blind_affect_class") if affect_enabled else None
    training = motion["training_segment"]
    training_segment = {
        "representation": EXPRESSION_TURN_REPRESENTATION,
        "start_frame": int(training["start_frame"]),
        "end_frame_exclusive": int(training["end_frame_exclusive"]),
        "frame_count": int(training["end_frame_exclusive"] - training["start_frame"]),
        "fixed_window_sec": None,
        "cropped": False,
    }
    retarget_segment = _normalized_retarget_segment(motion)
    channels = {
        "motion": True,
        "semantic_conditioning": semantic_enabled,
        "expressive_conditioning": affect_enabled,
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": TRAIN_READY_KIND,
        "formal_episode_contract": FORMAL_EPISODE_CONTRACT,
        "expression_turn_contract_version": EXPRESSION_TURN_CONTRACT_VERSION,
        "eligibility_mode": FORMAL_ELIGIBILITY_MODE,
        "dataset_kind": dataset_kind,
        "sample_id": sample_id,
        "clip_id": str(motion["task_id"]),
        "fixed_split_assignment": motion["fixed_split_assignment"],
        "training_qualification_tier": highest,
        "training_channel_masks": channels,
        "prompt_semantics_profile": MOTION_FORM_PROMPT_PROFILE,
        "prompt": prompt,
        "prompt_sha256": text_sha256(prompt) if prompt is not None else None,
        "prompt_text_provenance": PROMPT_TEXT_PROVENANCE if semantic_enabled else None,
        "prompt_review_id": action_review.get("review_id") if semantic_enabled else None,
        "emotion_id": emotion_id,
        "emotion_source": (
            "independent_blind_affect_consensus_or_adjudication_v1"
            if affect_enabled
            else None
        ),
        "emotion_supervision_mask": affect_enabled,
        "emotion_conditioning_mask": affect_enabled,
        "affect_observable_supervision_mask": affect_enabled,
        "official_emotion_conditioning_enabled": False,
        "official_category_conditioning_enabled": False,
        "behavior_supervision_mask": False,
        "behavior_id": None,
        "semantic_supervision_masks": {
            "official_category": False,
            "robot_observable_motion_form": semantic_enabled,
            "communicative_intent": False,
            "prompt_text": semantic_enabled,
            "legacy_gesture": False,
        },
        "training_segment": training_segment,
        "trajectory_sha256": motion["trajectory_sha256"],
        "physical_evidence_profile": motion["physical_evidence_profile"],
        "motion_18d": {
            "state": "passed",
            "safe_csv": motion["trajectory"],
            "safe_csv_sha256": motion["trajectory_sha256"],
            "frames": motion["frames"],
            "csv_rows": motion["frames"],
            "fps": motion["fps"],
            "source_window_frames": training_segment["frame_count"],
            "physical_evidence_profile": motion["physical_evidence_profile"],
            "quality_gate": motion["quality_gate"],
            "retarget_segment": retarget_segment,
        },
        "quality_gate": motion["quality_gate"],
        "retarget_segment": retarget_segment,
        "retarget_qc_passed": True,
        "quality_source_window_frames": training_segment["frame_count"],
        "quality_output_frame_count": motion["frames"],
        **motion["lineage_fields"],
        **motion["provenance"],
        "expression_turn_review_record": review,
        "expression_turn_review_record_sha256": review_sha,
        "qualification_report": qualification,
        "qualification_report_sha256": qualification_sha,
        "qualifications": qualification["qualifications"],
        "training_admission": {
            "contract": FORMAL_EPISODE_CONTRACT,
            "expression_turn_review_record_sha256": review_sha,
            "qualification_report_sha256": qualification_sha,
            "retarget_quality_record_sha256": motion["lineage_fields"]["retarget_quality_record_sha256"],
            "training_qualification_tier": highest,
            "training_channel_masks": channels,
        },
        "accepted_for_training": True,
    }
    _assert_train_output_has_no_hidden_labels(result)
    return result


def merge_reviews(
    *,
    public_summary: Path,
    hidden_mapping: Path,
    motion_manifest: Path,
    output_root: Path,
    author_submissions: Path | None = None,
    matcher_submissions: Path | None = None,
    affect_submissions: Iterable[Path] = (),
    affect_adjudications: Path | None = None,
    source_catalog: Path | None = None,
) -> dict[str, Any]:
    dataset_kind, arc_queue_path, affect_queue_path, queues = _verified_public_bundle(public_summary)
    hidden_mapping = hidden_mapping.resolve()
    motion_manifest = motion_manifest.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_catalog_path = source_catalog.resolve() if source_catalog is not None else None
    source_catalog_hash = sha256(source_catalog_path) if source_catalog_path else None
    hidden = _index_unique(read_jsonl(hidden_mapping), "sample_id", context="hidden mapping")
    if set(hidden) != set(queues):
        raise ValueError("hidden/public sample sets differ")
    motion_by_task = _index_unique(read_jsonl(motion_manifest), "task_id", context="motion manifest")
    catalog = _index_catalog(source_catalog_path)

    authors = _load_optional_index(author_submissions, "sample_id", context="author submissions")
    matchers = _load_optional_index(matcher_submissions, "sample_id", context="matcher submissions")
    adjudicators = _load_optional_index(affect_adjudications, "sample_id", context="affect adjudications")
    affect_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    affect_paths = [path.resolve() for path in affect_submissions]
    for path in affect_paths:
        for row in read_jsonl(path):
            sample_id = row.get("sample_id")
            if sample_id not in queues:
                raise ValueError(f"unknown affect sample: {sample_id}")
            affect_rows[str(sample_id)].append(row)
    for name, index in (("author", authors), ("matcher", matchers), ("adjudicator", adjudicators)):
        extra = sorted(set(index) - set(queues))
        if extra:
            raise ValueError(f"unknown {name} samples: {extra}")

    tiers: dict[str, list[dict[str, Any]]] = {
        "base_motion": [],
        "semantic_conditioning": [],
        "expressive_conditioning": [],
    }
    pending: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    review_records: list[dict[str, Any]] = []
    physical_evidence_records: list[dict[str, Any]] = []
    matcher_queue: list[dict[str, Any]] = []
    motion_manifest_hash = sha256(motion_manifest)
    hidden_mapping_hash = sha256(hidden_mapping)
    global_review_ids: dict[str, str] = {}

    def register_review_id(review_id: str, sample_id: str) -> None:
        previous = global_review_ids.get(review_id)
        if previous is not None:
            raise ValueError(
                f"review ID {review_id} is reused by {previous} and {sample_id}"
            )
        global_review_ids[review_id] = sample_id

    for sample_id in sorted(queues):
        queue = queues[sample_id]
        hidden_row = hidden[sample_id]
        task_id = hidden_row.get("task_id")
        if task_id not in motion_by_task:
            raise ValueError(f"{sample_id}: missing motion record for {task_id}")
        motion = _motion_evidence(
            sample_id=sample_id,
            queue=queue,
            hidden=hidden_row,
            motion=motion_by_task[str(task_id)],
            source_catalog=catalog,
            source_catalog_sha256=source_catalog_hash,
        )
        physical_evidence_records.append(motion["physical_evidence"])
        author = (
            _validate_author(authors[sample_id], queue, frames=motion["frames"])
            if sample_id in authors
            else None
        )
        if author is not None:
            register_review_id(str(author["arc_review_id"]), sample_id)
            register_review_id(str(author["action_review_id"]), sample_id)
        matcher = None
        if sample_id in matchers:
            if author is None or not author["text_valid"]:
                raise ValueError(f"{sample_id}: matcher exists without a valid author text")
            matcher = _validate_matcher(matchers[sample_id], queue, author)
            register_review_id(str(matcher["review_id"]), sample_id)
        affects = [_validate_affect(row, queue) for row in affect_rows[sample_id]]
        if len({item["review_id"] for item in affects}) != len(affects) or len(
            {item["reviewer_id"] for item in affects}
        ) != len(affects):
            raise ValueError(f"{sample_id}: affect reviews are not independent")
        for item in affects:
            register_review_id(str(item["review_id"]), sample_id)

        role_review_ids = set()
        role_reviewer_ids = set()
        if author is not None:
            role_review_ids.update({author["arc_review_id"], author["action_review_id"]})
            role_reviewer_ids.update({author["arc_reviewer_id"], author["action_reviewer_id"]})
        if matcher is not None:
            if matcher["review_id"] in role_review_ids or matcher["reviewer_id"] in role_reviewer_ids:
                raise ValueError(f"{sample_id}: matcher identity conflict")
            role_review_ids.add(matcher["review_id"])
            role_reviewer_ids.add(matcher["reviewer_id"])
        if any(item["review_id"] in role_review_ids or item["reviewer_id"] in role_reviewer_ids for item in affects):
            raise ValueError(f"{sample_id}: affect reviewer is not independent")
        role_review_ids.update(item["review_id"] for item in affects)
        role_reviewer_ids.update(item["reviewer_id"] for item in affects)

        adjudication = None
        if sample_id in adjudicators:
            if len(affects) < 2:
                raise ValueError(f"{sample_id}: adjudication requires two affect reviews")
            adjudication = _validate_adjudication(adjudicators[sample_id], queue, affects)
            if adjudication["review_id"] in role_review_ids or adjudication["reviewer_id"] in role_reviewer_ids:
                raise ValueError(f"{sample_id}: adjudicator is not independent")
            register_review_id(str(adjudication["review_id"]), sample_id)
        review = _review_record(
            dataset_kind=dataset_kind,
            queue=queue,
            motion=motion,
            author=author,
            matcher=matcher,
            affects=affects,
            adjudication=adjudication,
        )
        review_sha = value_sha256(review)
        qualification = evaluate_expression_turn(review)
        qualification_sha = value_sha256(qualification)
        review_records.append(
            {
                **review,
                "expression_turn_review_record_sha256": review_sha,
            }
        )
        reports.append({
            **qualification,
            "expression_turn_review_record_sha256": review_sha,
            "qualification_report_sha256": qualification_sha,
            "qualification_evidence": {
                "anonymous_video_and_blind_reviews_only": True,
                "hidden_label_fields_used": [],
                "official_emotion_used": False,
                "physical_evidence_record_sha256": motion["physical_evidence"]["record_sha256"],
            },
        })
        eligible = qualification["qualifications"]
        if eligible["base_motion"]["eligible"]:
            highest = str(qualification["highest_qualification"])
            if highest not in tiers:
                raise AssertionError(f"{sample_id}: invalid highest qualification {highest}")
            admitted_motion = dict(motion)
            admitted_motion["trajectory"] = _stage_trajectory(
                str(motion["trajectory"]),
                output_root / "trajectories" / f"{sample_id}.csv",
                str(motion["trajectory_sha256"]),
            )
            tiers[highest].append(
                _train_ready(
                    dataset_kind=dataset_kind,
                    sample_id=sample_id,
                    motion=admitted_motion,
                    review=review,
                    review_sha=review_sha,
                    qualification=qualification, qualification_sha=qualification_sha,
                )
            )
        if not eligible["base_motion"]["eligible"]:
            target = rejected if qualification["decision"] == "reject" else pending
            target.append({"sample_id": sample_id, "accepted_for_training": False, "qualification_report": qualification})
        elif not eligible["expressive_conditioning"]["eligible"]:
            pending.append({"sample_id": sample_id, "accepted_for_training": False, "qualification_report": qualification})

        if author is not None and author["text_valid"] and matcher is None:
            matcher_queue.append({
                "schema_version": SCHEMA_VERSION,
                "artifact_kind": "expression_turn_action_match_queue_record_v1",
                "sample_id": sample_id,
                "video_path": queue["video_path"],
                "video_sha256": queue["video_sha256"],
                "context_level": queue["context_level"],
                "audio_available": False,
                "label_metadata_exposed": False,
                "action_protocol_version": ACTION_PROTOCOL,
                "author_action_review_id": author["action_review_id"],
                "candidate_text": author["candidate_text"],
                "candidate_text_sha256": author["candidate_text_sha256"],
                "action_match_review_id": None,
                "action_match_reviewer_id": None,
                "action_result": None,
                "observable_description": None,
                "accepted_for_training": False,
            })

    paths = {
        "base_motion": output_root / "base_motion_train_ready.jsonl",
        "semantic_conditioning": output_root / "semantic_conditioning_train_ready.jsonl",
        "expressive_conditioning": output_root / "expressive_conditioning_train_ready.jsonl",
    }
    network_validation: dict[str, dict[str, Any]] = {}
    for tier, path in paths.items():
        records = tiers[tier]
        if not records:
            atomic_jsonl(path, [])
            network_validation[tier] = {
                "records": 0,
                "loader_invoked": False,
                "loader_validated_records": 0,
                "episode_validator_validated_records": 0,
                "network_contract_validation_passed": True,
            }
            continue
        validation_path = path.with_name(
            f".{path.name}.{os.getpid()}.network_validation.jsonl"
        )
        atomic_jsonl(validation_path, records)
        try:
            episodes = load_expression_turn_v8_episodes(validation_path)
            validated = [
                validate_expression_turn_v8_episode(episode)
                for episode in episodes
            ]
            if len(episodes) != len(records) or len(validated) != len(records):
                raise AssertionError(f"{tier}: network loader count mismatch")
            os.replace(validation_path, path)
        except Exception:
            validation_path.unlink(missing_ok=True)
            raise
        network_validation[tier] = {
            "records": len(records),
            "loader_invoked": True,
            "loader_validated_records": len(episodes),
            "episode_validator_validated_records": len(validated),
            "network_contract_validation_passed": True,
        }
    atomic_jsonl(output_root / "qualification_reports.jsonl", reports)
    atomic_jsonl(output_root / "formal_review_records.jsonl", review_records)
    physical_evidence_path = output_root / "physical_evidence_records.jsonl"
    atomic_jsonl(physical_evidence_path, physical_evidence_records)
    atomic_jsonl(output_root / "pending_manifest.jsonl", pending)
    atomic_jsonl(output_root / "rejected_manifest.jsonl", rejected)
    atomic_jsonl(output_root / "action_match_queue.jsonl", matcher_queue)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "expression_turn_three_tier_blind_review_merge_v1",
        "dataset_kind": dataset_kind,
        "records": len(queues),
        "counts": {
            "base_motion_train_ready": len(tiers["base_motion"]),
            "semantic_conditioning_train_ready": len(tiers["semantic_conditioning"]),
            "expressive_conditioning_train_ready": len(tiers["expressive_conditioning"]),
            "pending": len(pending),
            "rejected": len(rejected),
        },
        "inputs": {
            "public_summary": str(public_summary.resolve()),
            "public_summary_sha256": sha256(public_summary.resolve()),
            "arc_action_queue_sha256": sha256(arc_queue_path),
            "affect_queue_sha256": sha256(affect_queue_path),
            "hidden_mapping_sha256": hidden_mapping_hash,
            "motion_manifest_sha256": motion_manifest_hash,
            "source_catalog_sha256": source_catalog_hash,
            "author_submissions_sha256": sha256(author_submissions.resolve()) if author_submissions else None,
            "matcher_submissions_sha256": sha256(matcher_submissions.resolve()) if matcher_submissions else None,
            "affect_submissions_sha256": [sha256(path) for path in affect_paths],
            "affect_adjudications_sha256": sha256(affect_adjudications.resolve()) if affect_adjudications else None,
        },
        "outputs": {
            tier: {"path": str(path), "sha256": sha256(path), "records": len(tiers[tier])}
            for tier, path in paths.items()
        },
        "qualification_uses_hidden_or_official_labels": False,
        "tier_membership_policy": "disjoint_highest_qualification_only",
        "network_contract_validation": network_validation,
        "physical_evidence": {
            "path": str(physical_evidence_path),
            "sha256": sha256(physical_evidence_path),
            "records": len(physical_evidence_records),
            "records_with_verified_frame_count": sum(
                isinstance(record.get("frames"), int) and record.get("frames", 0) >= 3
                for record in physical_evidence_records
            ),
            "trajectory_frames_total": sum(
                int(record["frames"]) for record in physical_evidence_records
            ),
            "trajectory_frames_min": min(
                int(record["frames"]) for record in physical_evidence_records
            ),
            "trajectory_frames_max": max(
                int(record["frames"]) for record in physical_evidence_records
            ),
            "collision_evaluated_records": sum(
                "upper_body_collision_frame_rate" in record
                for record in physical_evidence_records
            ),
            "collision_pass_records": sum(
                (record.get("quality_gate") or {}).get("collision_pass") is True
                for record in physical_evidence_records
            ),
            "upper_body_collision_frames_total": sum(
                int(record.get("upper_body_collision_frames", 0))
                for record in physical_evidence_records
            ),
            "upper_body_collision_frame_rate_max": max(
                float(record.get("upper_body_collision_frame_rate", 0.0))
                for record in physical_evidence_records
            ),
            "collision_frame_rate_threshold": COLLISION_FRAME_RATE_MAX,
        },
        "merge_implementation_sha256": sha256(Path(__file__).resolve()),
        "formal_episode_contract": FORMAL_EPISODE_CONTRACT,
        "accepted_for_training": sum(len(rows) for rows in tiers.values()),
        "unique_train_ready_samples": len(
            {
                row["sample_id"]
                for rows in tiers.values()
                for row in rows
            }
        ),
    }
    atomic_json(output_root / "summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-summary", type=Path, required=True)
    parser.add_argument("--hidden-mapping", type=Path, required=True)
    parser.add_argument("--motion-manifest", type=Path, required=True)
    parser.add_argument("--source-catalog", type=Path)
    parser.add_argument("--author-submissions", type=Path)
    parser.add_argument("--matcher-submissions", type=Path)
    parser.add_argument("--affect-submission", type=Path, action="append", default=[])
    parser.add_argument("--affect-adjudications", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = merge_reviews(
        public_summary=args.public_summary,
        hidden_mapping=args.hidden_mapping,
        motion_manifest=args.motion_manifest,
        source_catalog=args.source_catalog,
        author_submissions=args.author_submissions,
        matcher_submissions=args.matcher_submissions,
        affect_submissions=args.affect_submission,
        affect_adjudications=args.affect_adjudications,
        output_root=args.output_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
