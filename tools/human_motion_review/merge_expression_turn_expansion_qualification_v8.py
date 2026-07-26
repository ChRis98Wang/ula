#!/usr/bin/env python3
"""Merge one BEAT2 v8 expansion round into fail-closed candidate tiers.

This is deliberately not a train-ready merger.  It binds the anonymous review
evidence to the hidden physical lineage and to a previously built natural-
boundary continuation decision.  Only samples declared complete by that
continuation decision are allowed to consume an affect judgment.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "1.0.0"
PUBLIC_KIND = "expression_turn_v8_expansion_separate_blind_review_bundle_v1"
HIDDEN_KIND = "expression_turn_v8_expansion_hidden_blind_mapping_v1"
PIPELINE_KIND = "beat2_expression_turn_v8_expansion_physical_pipeline_v1"
ARC_SUMMARY_KIND = (
    "beat2_expression_turn_v8_expansion_arc_action_review_submission_summary_v1"
)
AFFECT_SUMMARY_KIND = "robot_affect_blind_review_submission_summary_v1"
CONTINUATION_KIND = "expression_turn_v8_one_level_natural_context_expansion_plan"
COMPLETE_KIND = "expression_turn_v8_current_context_arc_action_complete"
REQUEST_KIND = "expression_turn_v8_one_level_natural_context_expansion_request"
EXHAUSTED_KIND = "expression_turn_v8_natural_context_exhausted"
TRAIN_CANDIDATE_KIND = "expression_turn_v8_expansion_train_candidate_v1"
NEEDS_KIND = "expression_turn_v8_expansion_needs_review_or_expansion_v1"
REJECT_KIND = "expression_turn_v8_expansion_reject_v1"
SUMMARY_KIND = "expression_turn_v8_expansion_qualification_summary_v1"

ARC_PROTOCOL = "robot_expression_arc_blind_video_v1"
ACTION_PROTOCOL = "robot_action_semantics_blind_video_v1"
AFFECT_PROTOCOL = "robot_affect_blind_video_v1"
AFFECT_CLASSES = ("angry", "fear", "happy", "neutral", "sad", "surprise")
PHASES = ("onset", "apex", "offset")
PHASE_STATUSES = {"complete", "incomplete", "ambiguous"}
INCOMPLETE_PHASE_STATUSES = {"incomplete", "ambiguous"}
PUBLIC_DURATION_POLICY = "one_next_predeclared_natural_boundary_level_no_fixed_window"
SELECTION_POLICY = "exactly_one_next_predeclared_natural_context_level"
PROCESSING_SCOPE = (
    "physical_retarget_and_silent_render_only_pending_fresh_blind_arc_action_and_affect_review"
)
QUALITY_KIND = "beat2_expression_turn_18d_v8_quality"
RETARGET_RESULT_KIND = "ula_v2_18d_expression_turn_retarget_v1"
REQUIRED_PHYSICAL_GATES = {
    "joint_limits_pass",
    "velocity_pass",
    "target_fit_pass",
    "collision_pass",
    "axis_direction_pass",
    "head_joint_limits_pass",
    "head_velocity_pass",
    "head_direction_pass",
    "head_continuity_pass",
    "safety_endpoint_pass",
    "safety_post_velocity_pass",
    "safety_slowdown_ratio_pass",
    "safety_time_map_pass",
    "passed",
}


def stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def value_sha256(value: object) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Unreadable JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_bound_jsonl(path: Path) -> list[tuple[dict[str, Any], str]]:
    rows: list[tuple[dict[str, Any], str]] = []
    try:
        handle = path.open("rb")
    except OSError as error:
        raise ValueError(f"Unreadable JSONL: {path}") from error
    with handle:
        for line_number, raw in enumerate(handle, 1):
            payload = raw[:-1] if raw.endswith(b"\n") else raw
            if payload.endswith(b"\r"):
                raise ValueError(f"CRLF JSONL is not accepted: {path}:{line_number}")
            if not payload.strip():
                continue
            try:
                value = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid JSONL record: {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object: {path}:{line_number}")
            rows.append((value, hashlib.sha256(payload).hexdigest()))
    return rows


def index_bound(
    rows: Iterable[tuple[dict[str, Any], str]], key: str, *, context: str
) -> dict[str, tuple[dict[str, Any], str]]:
    result: dict[str, tuple[dict[str, Any], str]] = {}
    for row, line_sha256 in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value or value in result:
            raise ValueError(f"{context} contains invalid or duplicate {key}")
        result[value] = (row, line_sha256)
    return result


def atomic_json(path: Path, value: object) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = "".join(stable_json(row) + "\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _declared_path(owner: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing declared path: {field}")
    path = Path(value)
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve()


def _require_declared_file(
    owner: Path,
    metadata: dict[str, Any],
    *,
    path_field: str,
    sha_field: str,
    expected: Path | None = None,
) -> Path:
    path = _declared_path(owner, metadata.get(path_field), field=path_field)
    if expected is not None and path != expected.resolve():
        raise ValueError(f"{path_field} does not resolve to the supplied input")
    if not path.is_file() or metadata.get(sha_field) != sha256_file(path):
        raise ValueError(f"{sha_field} does not bind {path_field}")
    return path


def _require_file_binding(
    owner: Path, binding: object, *, context: str, expected: Path
) -> None:
    if not isinstance(binding, dict):
        raise ValueError(f"{context} is not a file binding")
    path = _declared_path(owner, binding.get("path"), field=f"{context}.path")
    if (
        path != expected.resolve()
        or not path.is_file()
        or binding.get("sha256") != sha256_file(path)
    ):
        raise ValueError(f"{context} file binding mismatch")


def _verify_record_sha(row: dict[str, Any], field: str, *, context: str) -> None:
    expected = row.get(field)
    payload = dict(row)
    payload.pop(field, None)
    if expected != value_sha256(payload):
        raise ValueError(f"{context} record SHA mismatch")


def _interval(value: object, *, context: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} is not an interval")
    start = value.get("start_frame")
    end = value.get("end_frame_exclusive")
    inferred = end - start if isinstance(start, int) and isinstance(end, int) else None
    frames = value.get("frame_count", inferred)
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or isinstance(frames, bool)
        or not isinstance(frames, int)
        or start < 0
        or end <= start
        or frames != end - start
    ):
        raise ValueError(f"{context} is invalid")
    return {"start_frame": start, "end_frame_exclusive": end, "frame_count": frames}


def _strictly_contains(outer: dict[str, int], inner: dict[str, int]) -> bool:
    return (
        outer != inner
        and outer["start_frame"] <= inner["start_frame"]
        and inner["end_frame_exclusive"] <= outer["end_frame_exclusive"]
    )


def _verify_public_bundle(
    public_summary_path: Path,
) -> tuple[
    dict[str, Any],
    Path,
    Path,
    dict[str, tuple[dict[str, Any], str]],
    dict[str, tuple[dict[str, Any], str]],
]:
    summary = read_json(public_summary_path)
    if (
        summary.get("artifact_kind") != PUBLIC_KIND
        or summary.get("accepted_for_training") is not False
        or summary.get("duration_policy") != PUBLIC_DURATION_POLICY
        or summary.get("all_samples_native_variable_length") is not True
        or summary.get("fixed_duration_window_used") is not False
        or summary.get("same_anonymous_silent_video_used_by_both_reviews") is not True
        or summary.get("source_identity_official_action_text_and_affect_exposed") is not False
    ):
        raise ValueError("Public expansion bundle violates the blind variable-length contract")
    arc_path = _require_declared_file(
        public_summary_path,
        summary,
        path_field="arc_action_queue",
        sha_field="arc_action_queue_sha256",
    )
    affect_path = _require_declared_file(
        public_summary_path,
        summary,
        path_field="affect_queue",
        sha_field="affect_queue_sha256",
    )
    if arc_path.parent != public_summary_path.parent or affect_path.parent != public_summary_path.parent:
        raise ValueError("Public queues escape the public bundle")
    arc = index_bound(read_bound_jsonl(arc_path), "sample_id", context="arc queue")
    affect = index_bound(read_bound_jsonl(affect_path), "sample_id", context="affect queue")
    if set(arc) != set(affect) or summary.get("records") != len(arc):
        raise ValueError("Public arc and affect queue sets/counts differ")
    allowed = list(AFFECT_CLASSES)
    for sample_id in arc:
        arc_row = arc[sample_id][0]
        affect_row = affect[sample_id][0]
        common = (
            "sample_id",
            "video_path",
            "video_sha256",
            "context_level",
            "frame_count",
            "fps",
            "audio_available",
            "label_metadata_exposed",
            "native_duration_preserved",
            "fixed_duration_window_used",
        )
        if any(arc_row.get(key) != affect_row.get(key) for key in common):
            raise ValueError(f"{sample_id}: public queues bind different evidence")
        frames = arc_row.get("frame_count")
        if (
            arc_row.get("arc_protocol_version") != ARC_PROTOCOL
            or arc_row.get("action_protocol_version") != ACTION_PROTOCOL
            or affect_row.get("affect_protocol_version") != AFFECT_PROTOCOL
            or affect_row.get("allowed_classes") != allowed
            or arc_row.get("audio_available") is not False
            or arc_row.get("label_metadata_exposed") is not False
            or arc_row.get("native_duration_preserved") is not True
            or arc_row.get("fixed_duration_window_used") is not False
            or isinstance(frames, bool)
            or not isinstance(frames, int)
            or frames < 3
            or not math.isclose(float(arc_row.get("fps", 0.0)), 30.0, abs_tol=1e-9)
        ):
            raise ValueError(f"{sample_id}: invalid public queue contract")
        video = Path(str(arc_row.get("video_path") or "")).resolve()
        if video.parent != public_summary_path.parent / "videos":
            raise ValueError(f"{sample_id}: video escapes public video directory")
        if not video.is_file() or sha256_file(video) != arc_row.get("video_sha256"):
            raise ValueError(f"{sample_id}: anonymous video hash mismatch")
    return summary, arc_path, affect_path, arc, affect


def _verify_arc_reviews(
    *,
    submission_path: Path,
    summary_path: Path,
    public_summary_path: Path,
    queue_path: Path,
    queue: dict[str, tuple[dict[str, Any], str]],
) -> dict[str, tuple[dict[str, Any], str]]:
    reviews = index_bound(
        read_bound_jsonl(submission_path), "sample_id", context="arc/action submission"
    )
    summary = read_json(summary_path)
    exact = {
        "artifact_kind": ARC_SUMMARY_KIND,
        "validation_passed": True,
        "fixed_duration_window_used": False,
        "elapsed_duration_used_as_gate": False,
        "native_variable_length_reviewed": True,
    }
    failed = sorted(key for key, value in exact.items() if summary.get(key) != value)
    if failed or summary.get("records") != len(reviews):
        raise ValueError(f"Arc review summary violates contract: {failed}")
    declared = {
        "submission_path": submission_path,
        "public_summary_path": public_summary_path,
        "public_queue_path": queue_path,
    }
    for field, expected in declared.items():
        if _declared_path(summary_path, summary.get(field), field=field) != expected.resolve():
            raise ValueError(f"Arc review summary path mismatch: {field}")
    hashes = {
        "submission_sha256": sha256_file(submission_path),
        "public_summary_sha256": sha256_file(public_summary_path),
        "public_queue_sha256": sha256_file(queue_path),
    }
    if any(summary.get(key) != value for key, value in hashes.items()):
        raise ValueError("Arc review summary hash binding mismatch")
    if set(reviews) != set(queue):
        raise ValueError("Arc review coverage differs from public queue")
    review_ids: set[str] = set()
    for sample_id, (row, _line_sha) in reviews.items():
        queued = queue[sample_id][0]
        for key in ("sample_id", "video_path", "video_sha256", "context_level", "frame_count"):
            if row.get(key) != queued.get(key):
                raise ValueError(f"{sample_id}: arc review {key} binding mismatch")
        if (
            row.get("arc_protocol_version") != ARC_PROTOCOL
            or row.get("action_protocol_version") != ACTION_PROTOCOL
            or row.get("queue_sha256") != sha256_file(queue_path)
            or row.get("full_decode_to_eof") is not True
            or row.get("decoded_frame_count") != queued.get("frame_count")
            or row.get("native_duration_preserved") is not True
            or row.get("fixed_duration_window_used") is not False
            or row.get("audio_available") is not False
            or row.get("label_metadata_exposed") is not False
            or row.get("emotion_judgment_performed") is not False
            or row.get("training_admission") is not False
            or row.get("action_result") != "pass"
            or row.get("action_observability") != "observable"
            or not isinstance(row.get("observable_description"), str)
            or not row["observable_description"].strip()
        ):
            raise ValueError(f"{sample_id}: arc/action review violates blind contract")
        ids = (row.get("arc_review_id"), row.get("action_review_id"))
        reviewers = (row.get("arc_reviewer_id"), row.get("action_reviewer_id"))
        if not all(isinstance(value, str) and value for value in ids + reviewers):
            raise ValueError(f"{sample_id}: missing arc/action reviewer identity")
        if any(value in review_ids for value in ids):
            raise ValueError(f"{sample_id}: duplicate arc/action review ID")
        review_ids.update(ids)
        evidence: dict[str, int] = {}
        for phase in PHASES:
            status = row.get(f"{phase}_status")
            frame = row.get(f"{phase}_evidence_frame")
            if status not in PHASE_STATUSES:
                raise ValueError(f"{sample_id}: invalid {phase} status")
            if (
                isinstance(frame, bool)
                or not isinstance(frame, int)
                or not 0 <= frame < int(queued["frame_count"])
                or not isinstance(row.get(f"{phase}_basis"), str)
                or not row[f"{phase}_basis"]
            ):
                raise ValueError(f"{sample_id}: invalid {phase} evidence")
            if status == "complete":
                evidence[phase] = frame
        if set(evidence) == set(PHASES) and not (
            evidence["onset"] < evidence["apex"] < evidence["offset"]
        ):
            raise ValueError(f"{sample_id}: complete arc evidence is unordered")
    return reviews


def _verify_affect_reviews(
    *,
    submission_path: Path,
    summary_path: Path,
    queue_path: Path,
    queue: dict[str, tuple[dict[str, Any], str]],
    arc_reviews: dict[str, tuple[dict[str, Any], str]],
) -> dict[str, tuple[dict[str, Any], str]]:
    reviews = index_bound(
        read_bound_jsonl(submission_path), "sample_id", context="affect submission"
    )
    summary = read_json(summary_path)
    if (
        summary.get("artifact_kind") != AFFECT_SUMMARY_KIND
        or summary.get("training_admission") is not False
        or _declared_path(summary_path, summary.get("submission_path"), field="submission_path")
        != submission_path.resolve()
        or summary.get("submission_sha256") != sha256_file(submission_path)
        or _declared_path(summary_path, summary.get("public_queue_path"), field="public_queue_path")
        != queue_path.resolve()
        or summary.get("public_queue_sha256") != sha256_file(queue_path)
    ):
        raise ValueError("Affect review summary binding mismatch")
    coverage = summary.get("coverage")
    integrity = summary.get("integrity")
    if (
        not isinstance(coverage, dict)
        or coverage.get("expected_records") != len(queue)
        or coverage.get("reviewed_records") != len(reviews)
        or coverage.get("complete") is not True
        or not isinstance(integrity, dict)
        or integrity.get("all_full_video_reviewed") is not True
        or integrity.get("all_native_variable_length_reviewed") is not True
        or integrity.get("fixed_duration_window_used") is not False
    ):
        raise ValueError("Affect review summary is incomplete or duration-gated")
    if set(reviews) != set(queue):
        raise ValueError("Affect review coverage differs from public queue")
    review_ids: set[str] = set()
    result_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    for sample_id, (row, _line_sha) in reviews.items():
        queued = queue[sample_id][0]
        arc = arc_reviews[sample_id][0]
        if (
            row.get("video_sha256") != queued.get("video_sha256")
            or row.get("public_queue_sha256") != sha256_file(queue_path)
            or row.get("decoded_frame_count") != queued.get("frame_count")
            or row.get("full_video_reviewed") is not True
            or row.get("training_admission") is not False
            or row.get("affect_protocol_version") != AFFECT_PROTOCOL
            or row.get("allowed_classes") != list(AFFECT_CLASSES)
        ):
            raise ValueError(f"{sample_id}: affect review evidence binding mismatch")
        review_id = row.get("affect_review_id")
        reviewer_id = row.get("affect_reviewer_id")
        if (
            not isinstance(review_id, str)
            or not review_id
            or review_id in review_ids
            or not isinstance(reviewer_id, str)
            or not reviewer_id
            or reviewer_id in {arc.get("arc_reviewer_id"), arc.get("action_reviewer_id")}
        ):
            raise ValueError(f"{sample_id}: affect reviewer is missing, duplicate, or not independent")
        review_ids.add(review_id)
        result = row.get("result")
        predicted = row.get("predicted_class")
        confidence = row.get("confidence")
        if result == "observable":
            if (
                predicted not in AFFECT_CLASSES
                or isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(float(confidence))
                or not 0.0 <= float(confidence) <= 1.0
            ):
                raise ValueError(f"{sample_id}: invalid observable affect result")
            class_counts[str(predicted)] += 1
        elif result in {"ambiguous", "not_observable"}:
            if predicted is not None or confidence is not None:
                raise ValueError(f"{sample_id}: non-observable affect carries a pseudo-label")
        else:
            raise ValueError(f"{sample_id}: invalid affect result")
        result_counts[str(result)] += 1
    if summary.get("result_distribution") != {
        key: result_counts.get(key, 0)
        for key in ("observable", "ambiguous", "not_observable")
    }:
        raise ValueError("Affect summary result distribution differs from submission")
    return reviews


def _verify_physical_lineage(
    *,
    hidden_summary_path: Path,
    hidden_mapping_path: Path,
    pipeline_summary_path: Path,
    passed_manifest_path: Path,
    public_queue: dict[str, tuple[dict[str, Any], str]],
) -> tuple[
    dict[str, tuple[dict[str, Any], str]],
    dict[str, tuple[dict[str, Any], str]],
]:
    hidden_summary = read_json(hidden_summary_path)
    if (
        hidden_summary.get("artifact_kind") != HIDDEN_KIND
        or hidden_summary.get("accepted_for_training") is not False
        or hidden_summary.get("public_distribution_forbidden") is not True
    ):
        raise ValueError("Hidden mapping summary violates fail-closed policy")
    _require_declared_file(
        hidden_summary_path,
        hidden_summary,
        path_field="mapping",
        sha_field="mapping_sha256",
        expected=hidden_mapping_path,
    )
    _require_declared_file(
        hidden_summary_path,
        hidden_summary,
        path_field="pipeline_summary",
        sha_field="pipeline_summary_sha256",
        expected=pipeline_summary_path,
    )
    _require_declared_file(
        hidden_summary_path,
        hidden_summary,
        path_field="render_passed_manifest",
        sha_field="render_passed_manifest_sha256",
        expected=passed_manifest_path,
    )
    mappings = index_bound(
        read_bound_jsonl(hidden_mapping_path), "sample_id", context="hidden mapping"
    )
    if hidden_summary.get("records") != len(mappings) or set(mappings) != set(public_queue):
        raise ValueError("Hidden mapping coverage differs from public queue")

    pipeline = read_json(pipeline_summary_path)
    if (
        pipeline.get("artifact_kind") != PIPELINE_KIND
        or pipeline.get("stage") not in {"all", "render"}
        or pipeline.get("fixed_six_second_windows_used") is not False
        or pipeline.get("elapsed_seconds_used_for_selection") is not False
        or pipeline.get("accepted_for_training") is not False
        or pipeline.get("license_training_admission") is not False
        or pipeline.get("semantic_admission") is not False
        or pipeline.get("affect_admission") is not False
    ):
        raise ValueError("Physical pipeline violates fail-closed variable-length policy")
    audit = pipeline.get("input_audit")
    if not isinstance(audit, dict):
        raise ValueError("Physical pipeline lacks input audit")
    audit_payload = dict(audit)
    audit_sha = audit_payload.pop("sha256", None)
    if audit_sha != value_sha256(audit_payload):
        raise ValueError("Physical pipeline input audit SHA mismatch")
    render = pipeline.get("render_summary")
    if not isinstance(render, dict):
        raise ValueError("Physical pipeline lacks render summary")
    if (
        _declared_path(
            pipeline_summary_path, render.get("passed_manifest"), field="passed_manifest"
        )
        != passed_manifest_path.resolve()
        or render.get("passed_manifest_sha256") != sha256_file(passed_manifest_path)
        or (render.get("counts") or {}).get("passed") != len(mappings)
        or (render.get("counts") or {}).get("failed") != 0
        or render.get("render_pass_grants_training_admission") is not False
    ):
        raise ValueError("Physical render summary does not bind a complete pass manifest")

    passed = index_bound(
        read_bound_jsonl(passed_manifest_path), "task_id", context="physical pass manifest"
    )
    if len(passed) != len(mappings):
        raise ValueError("Physical pass count differs from hidden mapping")
    mapping_tasks: set[str] = set()
    for sample_id, (mapping, _mapping_line_sha) in mappings.items():
        queued = public_queue[sample_id][0]
        task_id = mapping.get("task_id")
        if not isinstance(task_id, str) or not task_id or task_id in mapping_tasks:
            raise ValueError(f"{sample_id}: invalid or duplicate physical task")
        mapping_tasks.add(task_id)
        physical_bound = passed.get(task_id)
        if physical_bound is None:
            raise ValueError(f"{sample_id}: hidden mapping has no physical pass record")
        physical = physical_bound[0]
        if (
            mapping.get("accepted_for_training") is not False
            or mapping.get("native_duration_preserved") is not True
            or mapping.get("fixed_duration_window_used") is not False
            or mapping.get("official_action_text_or_affect_exposed") is not False
            or mapping.get("displayed_context_level") != queued.get("context_level")
            or mapping.get("frame_count") != queued.get("frame_count")
            or mapping.get("video_sha256") != queued.get("video_sha256")
            or physical.get("status") != "passed"
            or physical.get("accepted_for_training") is not False
            or physical.get("license_training_admission") is not False
            or physical.get("render_pass_grants_training_admission") is not False
            or physical.get("processing_scope") != PROCESSING_SCOPE
            or physical.get("video_sha256") != mapping.get("video_sha256")
            or physical.get("trajectory_sha256") != mapping.get("trajectory_sha256")
            or physical.get("trajectory_frames") != mapping.get("frame_count")
            or mapping.get("source_render_record_sha256") != value_sha256(physical)
        ):
            raise ValueError(f"{sample_id}: hidden/public/physical record binding mismatch")
        video = Path(str(physical.get("video_path") or "")).resolve()
        trajectory = Path(str(physical.get("trajectory_path") or "")).resolve()
        if (
            not video.is_file()
            or sha256_file(video) != physical.get("video_sha256")
            or not trajectory.is_file()
            or sha256_file(trajectory) != physical.get("trajectory_sha256")
        ):
            raise ValueError(f"{sample_id}: physical video/trajectory hash mismatch")
        training = physical.get("training_segment")
        retarget = physical.get("retarget_segment")
        if (
            not isinstance(training, dict)
            or training.get("cropped") is not False
            or training.get("fixed_window_sec") is not None
            or not isinstance(retarget, dict)
            or retarget.get("cropped") is not False
            or retarget.get("fixed_target_duration_sec") is not None
            or retarget.get("output_frame_count") != mapping.get("frame_count")
        ):
            raise ValueError(f"{sample_id}: physical record is cropped or duration-targeted")

        result_path = Path(str(physical.get("retarget_result_json") or "")).resolve()
        if (
            not result_path.is_file()
            or physical.get("retarget_result_json_sha256") != sha256_file(result_path)
        ):
            raise ValueError(f"{sample_id}: retarget result JSON binding mismatch")
        result = read_json(result_path)
        result_gates = result.get("quality_gate")
        if (
            result.get("artifact_kind") != RETARGET_RESULT_KIND
            or result.get("status") != "passed"
            or result.get("accepted_for_training") is not False
            or not isinstance(result_gates, dict)
            or any(result_gates.get(gate) is not True for gate in REQUIRED_PHYSICAL_GATES)
            or (
                physical.get("quality_gate") is not None
                and result_gates != physical.get("quality_gate")
            )
            or Path(str(result.get("safe_csv") or "")).resolve() != trajectory
            or result.get("safe_csv_sha256") != mapping.get("trajectory_sha256")
            or result.get("frames") != mapping.get("frame_count")
        ):
            raise ValueError(f"{sample_id}: retarget result did not pass all required 18D gates")
        result_segment = result.get("retarget_segment")
        if (
            not isinstance(result_segment, dict)
            or result_segment.get("cropped") is not False
            or result_segment.get("fixed_target_duration_sec") is not None
            or result_segment.get("output_frame_count") != mapping.get("frame_count")
        ):
            raise ValueError(f"{sample_id}: retarget result duration contract mismatch")

        quality_path = Path(str(result.get("quality_json") or "")).resolve()
        if (
            not quality_path.is_file()
            or result.get("quality_json_sha256") != sha256_file(quality_path)
        ):
            raise ValueError(f"{sample_id}: quality JSON binding mismatch")
        quality = read_json(quality_path)
        quality_gates = quality.get("quality_gate")
        quality_outputs = quality.get("outputs")
        quality_segment = quality.get("retarget_segment")
        safety = quality.get("safety_monotonic_retime")
        if (
            quality.get("artifact_kind") != QUALITY_KIND
            or quality.get("accepted_for_training") is not False
            or quality.get("license_training_admission") is not False
            or quality.get("semantic_admission") is not False
            or quality.get("affect_admission") is not False
            or quality.get("physical_quality_only") is not True
            or quality.get("processing_scope") != PROCESSING_SCOPE
            or quality.get("action_dim") != 18
            or quality.get("frames") != mapping.get("frame_count")
            or not math.isclose(float(quality.get("fps", 0.0)), 30.0, abs_tol=1e-9)
            or quality.get("representation") != "native_variable_length_expression_turn_v1"
            or quality.get("output_contract") != "ula_v2_18d_head_v1"
            or quality.get("safe_csv_sha256") != mapping.get("trajectory_sha256")
            or not isinstance(quality_outputs, dict)
            or Path(str(quality_outputs.get("safe_csv") or "")).resolve() != trajectory
            or not isinstance(quality_gates, dict)
            or quality_gates != result_gates
            or any(quality_gates.get(gate) is not True for gate in REQUIRED_PHYSICAL_GATES)
            or not isinstance(quality_segment, dict)
            or quality_segment.get("cropped") is not False
            or quality_segment.get("fixed_target_duration_sec") is not None
            or quality_segment.get("output_frame_count") != mapping.get("frame_count")
            or not isinstance(safety, dict)
            or safety.get("artifact_kind") != "ula_18d_safety_monotonic_retime_v1"
            or safety.get("post_velocity_pass") is not True
            or safety.get("slowdown_ratio_pass") is not True
            or safety.get("time_map_strictly_increasing") is not True
            or safety.get("first_frame_preserved") is not True
            or safety.get("last_frame_preserved") is not True
            or safety.get("cropped") is not False
            or safety.get("tiled") is not False
            or safety.get("target_duration_sec") is not None
            or safety.get("output_frame_count") != mapping.get("frame_count")
        ):
            raise ValueError(f"{sample_id}: bound quality evidence violates the 18D safety contract")
    if set(passed) != mapping_tasks:
        raise ValueError("Physical pass manifest contains unmapped tasks")
    return mappings, passed


def _verify_continuation(
    *,
    continuation_summary_path: Path,
    public_summary_path: Path,
    arc_queue_path: Path,
    arc_submission_path: Path,
    arc_summary_path: Path,
    hidden_mapping_path: Path,
    hidden_summary_path: Path,
    pipeline_summary_path: Path,
    passed_manifest_path: Path,
    public_queue: dict[str, tuple[dict[str, Any], str]],
    arc_reviews: dict[str, tuple[dict[str, Any], str]],
    mappings: dict[str, tuple[dict[str, Any], str]],
) -> tuple[
    dict[str, tuple[dict[str, Any], str]],
    dict[str, tuple[dict[str, Any], str]],
    dict[str, tuple[dict[str, Any], str]],
]:
    summary = read_json(continuation_summary_path)
    if (
        summary.get("artifact_kind") != CONTINUATION_KIND
        or summary.get("accepted_for_training_count") != 0
        or summary.get("fixed_minimum_maximum_or_target_duration_used") is not False
        or summary.get("elapsed_duration_used_as_gate") is not False
        or summary.get("same_source_only") is not True
        or summary.get("neighbor_crossing_allowed") is not False
        or summary.get("one_level_only") is not True
        or summary.get("selection_policy") != SELECTION_POLICY
    ):
        raise ValueError("Continuation summary violates natural-boundary policy")
    expected_inputs = {
        "public_summary_sha256": sha256_file(public_summary_path),
        "arc_action_queue_sha256": sha256_file(arc_queue_path),
        "arc_review_submission_sha256": sha256_file(arc_submission_path),
        "arc_review_summary_sha256": sha256_file(arc_summary_path),
        "expansion_hidden_mapping_sha256": sha256_file(hidden_mapping_path),
        "expansion_hidden_summary_sha256": sha256_file(hidden_summary_path),
        "pipeline_summary_sha256": sha256_file(pipeline_summary_path),
        "render_passed_manifest_sha256": sha256_file(passed_manifest_path),
    }
    inputs = summary.get("inputs")
    if not isinstance(inputs, dict) or any(inputs.get(k) != v for k, v in expected_inputs.items()):
        raise ValueError("Continuation summary input SHA binding mismatch")
    output_specs = {
        "complete_current_context": COMPLETE_KIND,
        "expansion_requests": REQUEST_KIND,
        "context_exhausted": EXHAUSTED_KIND,
    }
    partitions: dict[str, dict[str, tuple[dict[str, Any], str]]] = {}
    outputs = summary.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Continuation summary lacks outputs")
    for name, expected_kind in output_specs.items():
        binding = outputs.get(name)
        if not isinstance(binding, dict):
            raise ValueError(f"Continuation output binding missing: {name}")
        path = _declared_path(
            continuation_summary_path, binding.get("path"), field=f"outputs.{name}.path"
        )
        if not path.is_file() or binding.get("sha256") != sha256_file(path):
            raise ValueError(f"Continuation output SHA mismatch: {name}")
        rows = index_bound(
            read_bound_jsonl(path), "reviewed_anonymous_sample_id", context=name
        )
        if binding.get("records") != len(rows):
            raise ValueError(f"Continuation output count mismatch: {name}")
        if any(row.get("artifact_kind") != expected_kind for row, _ in rows.values()):
            raise ValueError(f"Continuation output kind mismatch: {name}")
        partitions[name] = rows
    complete = partitions["complete_current_context"]
    requests = partitions["expansion_requests"]
    exhausted = partitions["context_exhausted"]
    sets = [set(complete), set(requests), set(exhausted)]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Continuation partitions overlap")
    if set().union(*sets) != set(public_queue):
        raise ValueError("Continuation partitions do not exactly cover public samples")
    canonical_ids: set[str] = set()
    for partition_name, rows in partitions.items():
        for anonymous_id, (row, _line_sha) in rows.items():
            _verify_record_sha(row, "plan_record_sha256", context=anonymous_id)
            queue_row, queue_line_sha = public_queue[anonymous_id]
            review = arc_reviews[anonymous_id][0]
            mapping, mapping_line_sha = mappings[anonymous_id]
            canonical_id = row.get("sample_id")
            if not isinstance(canonical_id, str) or not canonical_id or canonical_id in canonical_ids:
                raise ValueError(f"{anonymous_id}: duplicate/invalid canonical base sample ID")
            canonical_ids.add(canonical_id)
            statuses = {phase: review.get(f"{phase}_status") for phase in PHASES}
            evidence = [review.get(f"{phase}_evidence_frame") for phase in PHASES]
            lineage = row.get("continuation_lineage")
            expected_lineage = {
                "source_public_summary_sha256": sha256_file(public_summary_path),
                "source_arc_action_queue_sha256": sha256_file(arc_queue_path),
                "review_submission_sha256": sha256_file(arc_submission_path),
                "review_summary_sha256": sha256_file(arc_summary_path),
                "review_record_sha256": value_sha256(review),
                "review_queue_record_sha256": queue_line_sha,
                "expansion_hidden_mapping_sha256": sha256_file(hidden_mapping_path),
                "expansion_hidden_mapping_record_sha256": mapping_line_sha,
                "expansion_hidden_summary_sha256": sha256_file(hidden_summary_path),
                "pipeline_summary_sha256": sha256_file(pipeline_summary_path),
                "render_passed_manifest_sha256": sha256_file(passed_manifest_path),
                "source_render_record_sha256": mapping.get("source_render_record_sha256"),
            }
            if (
                not isinstance(lineage, dict)
                or any(lineage.get(k) != v for k, v in expected_lineage.items())
                or row.get("accepted_for_training") is not False
                or row.get("semantic_supervision_mask") is not False
                or row.get("emotion_supervision_mask") is not False
                or row.get("elapsed_duration_used_as_gate") is not False
                or row.get("reviewed_video_sha256") != queue_row.get("video_sha256")
                or row.get("reviewed_trajectory_sha256") != mapping.get("trajectory_sha256")
                or row.get("reviewed_output_frame_count") != mapping.get("frame_count")
                or row.get("reviewed_context_level") != queue_row.get("context_level")
                or row.get("review_phase_statuses") != statuses
                or row.get("review_evidence_frames") != evidence
            ):
                raise ValueError(f"{anonymous_id}: continuation lineage/decision binding mismatch")
            all_complete = all(status == "complete" for status in statuses.values())
            has_incomplete = any(status in INCOMPLETE_PHASE_STATUSES for status in statuses.values())
            if partition_name == "complete_current_context":
                if not all_complete or row.get("review_qualification_status") != "complete_arc_action_candidate":
                    raise ValueError(f"{anonymous_id}: incomplete arc entered complete partition")
            else:
                if all_complete or not has_incomplete:
                    raise ValueError(f"{anonymous_id}: complete arc entered non-complete partition")
                if row.get("review_qualification_status") != "natural_context_expansion_required":
                    raise ValueError(f"{anonymous_id}: non-complete continuation status mismatch")
            if partition_name == "expansion_requests":
                reviewed_level = row.get("reviewed_context_level")
                if row.get("requested_context_level") != reviewed_level + 1:
                    raise ValueError(f"{anonymous_id}: continuation skips a context level")
                if not _strictly_contains(
                    _interval(row.get("requested_interval"), context=f"{anonymous_id}.requested"),
                    _interval(row.get("reviewed_interval"), context=f"{anonymous_id}.reviewed"),
                ):
                    raise ValueError(f"{anonymous_id}: requested context does not strictly expand")
    return complete, requests, exhausted


def _closed_record_base(
    *,
    anonymous_id: str,
    continuation: dict[str, Any],
    mapping: dict[str, Any],
    physical: dict[str, Any],
    input_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": anonymous_id,
        "canonical_base_sample_id": continuation["sample_id"],
        "task_id": mapping["task_id"],
        "fixed_split_assignment": mapping.get("fixed_split_assignment"),
        "context_level": continuation["reviewed_context_level"],
        "frame_count": mapping["frame_count"],
        "fps": 30,
        "trajectory_path": physical["trajectory_path"],
        "trajectory_sha256": mapping["trajectory_sha256"],
        "video_sha256": mapping["video_sha256"],
        "native_variable_length": True,
        "fixed_duration_window_used": False,
        "license_training_admission": False,
        "accepted_for_training": False,
        "source_bindings": input_hashes,
        "continuation_record_sha256": continuation["plan_record_sha256"],
    }


def merge_qualification(
    *,
    public_summary: Path,
    hidden_summary: Path,
    hidden_mapping: Path,
    physical_pipeline_summary: Path,
    physical_passed_manifest: Path,
    arc_review_submission: Path,
    arc_review_summary: Path,
    affect_review_submission: Path,
    affect_review_summary: Path,
    continuation_summary: Path,
    output_root: Path,
) -> dict[str, Any]:
    supplied = {
        name: path.resolve()
        for name, path in {
            "public_summary": public_summary,
            "hidden_summary": hidden_summary,
            "hidden_mapping": hidden_mapping,
            "physical_pipeline_summary": physical_pipeline_summary,
            "physical_passed_manifest": physical_passed_manifest,
            "arc_review_submission": arc_review_submission,
            "arc_review_summary": arc_review_summary,
            "affect_review_submission": affect_review_submission,
            "affect_review_summary": affect_review_summary,
            "continuation_summary": continuation_summary,
        }.items()
    }
    for name, path in supplied.items():
        if not path.is_file():
            raise ValueError(f"Missing supplied input {name}: {path}")
    output_root = output_root.resolve()
    expected_output_names = {
        "train_candidate.jsonl",
        "needs_review_or_expansion.jsonl",
        "reject.jsonl",
        "summary.json",
    }
    if output_root.exists():
        unexpected = sorted(
            path.name for path in output_root.iterdir() if path.name not in expected_output_names
        )
        if unexpected:
            raise ValueError(f"Output root contains unexpected files: {unexpected}")

    public, arc_queue_path, affect_queue_path, arc_queue, affect_queue = _verify_public_bundle(
        supplied["public_summary"]
    )
    arc_reviews = _verify_arc_reviews(
        submission_path=supplied["arc_review_submission"],
        summary_path=supplied["arc_review_summary"],
        public_summary_path=supplied["public_summary"],
        queue_path=arc_queue_path,
        queue=arc_queue,
    )
    affect_reviews = _verify_affect_reviews(
        submission_path=supplied["affect_review_submission"],
        summary_path=supplied["affect_review_summary"],
        queue_path=affect_queue_path,
        queue=affect_queue,
        arc_reviews=arc_reviews,
    )
    mappings, passed = _verify_physical_lineage(
        hidden_summary_path=supplied["hidden_summary"],
        hidden_mapping_path=supplied["hidden_mapping"],
        pipeline_summary_path=supplied["physical_pipeline_summary"],
        passed_manifest_path=supplied["physical_passed_manifest"],
        public_queue=arc_queue,
    )
    complete, requests, exhausted = _verify_continuation(
        continuation_summary_path=supplied["continuation_summary"],
        public_summary_path=supplied["public_summary"],
        arc_queue_path=arc_queue_path,
        arc_submission_path=supplied["arc_review_submission"],
        arc_summary_path=supplied["arc_review_summary"],
        hidden_mapping_path=supplied["hidden_mapping"],
        hidden_summary_path=supplied["hidden_summary"],
        pipeline_summary_path=supplied["physical_pipeline_summary"],
        passed_manifest_path=supplied["physical_passed_manifest"],
        public_queue=arc_queue,
        arc_reviews=arc_reviews,
        mappings=mappings,
    )

    input_hashes = {name: sha256_file(path) for name, path in supplied.items()}
    input_hashes["arc_action_queue"] = sha256_file(arc_queue_path)
    input_hashes["affect_queue"] = sha256_file(affect_queue_path)
    train_candidates: list[dict[str, Any]] = []
    needs: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    complete_affect_results: Counter[str] = Counter()
    complete_affect_classes: Counter[str] = Counter()

    for anonymous_id in sorted(arc_queue):
        mapping = mappings[anonymous_id][0]
        physical = passed[str(mapping["task_id"])][0]
        continuation = (
            complete.get(anonymous_id)
            or requests.get(anonymous_id)
            or exhausted.get(anonymous_id)
        )[0]
        base = _closed_record_base(
            anonymous_id=anonymous_id,
            continuation=continuation,
            mapping=mapping,
            physical=physical,
            input_hashes=input_hashes,
        )
        arc = arc_reviews[anonymous_id][0]
        if anonymous_id in complete:
            affect = affect_reviews[anonymous_id][0]
            affect_result = str(affect["result"])
            affect_observable = affect_result == "observable"
            complete_affect_results[affect_result] += 1
            if affect_observable:
                complete_affect_classes[str(affect["predicted_class"])] += 1
            record = {
                **base,
                "artifact_kind": TRAIN_CANDIDATE_KIND,
                "qualification_status": "complete_arc_action_motion_candidate",
                "arc_action_complete": True,
                "action_result": "pass",
                "action_observability": "observable",
                "action_observable_description": arc["observable_description"],
                "semantic_supervision_mask": False,
                "semantic_supervision_reason": (
                    "blind_action_is_observable_but_no_bound_candidate_text_was_submitted"
                ),
                "affect_review_evaluated": True,
                "affect_result": affect_result,
                "emotion_supervision_mask": affect_observable,
                "emotion_class": affect["predicted_class"] if affect_observable else None,
                "emotion_confidence": affect["confidence"] if affect_observable else None,
                "emotion_label_provenance": (
                    "independent_blind_robot_motion_review" if affect_observable else None
                ),
                "training_candidate_scopes": (
                    ["base_motion", "affect_conditioning"]
                    if affect_observable
                    else ["base_motion"]
                ),
                "training_blockers": [
                    "license_training_admission_unconfirmed",
                    "formal_train_manifest_not_built",
                ],
                "arc_review_record_sha256": value_sha256(arc),
                "affect_review_record_sha256": value_sha256(affect),
            }
            train_candidates.append(record)
        elif anonymous_id in requests:
            record = {
                **base,
                "artifact_kind": NEEDS_KIND,
                "qualification_status": "next_natural_boundary_expansion_required",
                "review_phase_statuses": continuation["review_phase_statuses"],
                "requested_context_level": continuation["requested_context_level"],
                "requested_interval": continuation["requested_interval"],
                "semantic_supervision_mask": False,
                "affect_review_evaluated": False,
                "affect_result": None,
                "emotion_supervision_mask": False,
                "emotion_class": None,
                "emotion_confidence": None,
                "next_action": "retarget_render_and_repeat_blind_reviews_at_next_natural_boundary",
                "training_blockers": [
                    "arc_incomplete_at_current_natural_boundary",
                    "affect_judgment_not_admitted_before_complete_arc",
                    "license_training_admission_unconfirmed",
                ],
                "arc_review_record_sha256": value_sha256(arc),
                "affect_review_record_sha256": None,
            }
            needs.append(record)
        else:
            record = {
                **base,
                "artifact_kind": REJECT_KIND,
                "qualification_status": "reject_incomplete_arc_context_exhausted",
                "review_phase_statuses": continuation["review_phase_statuses"],
                "context_exhausted_at_level": continuation["context_exhausted_at_level"],
                "semantic_supervision_mask": False,
                "affect_review_evaluated": False,
                "affect_result": None,
                "emotion_supervision_mask": False,
                "emotion_class": None,
                "emotion_confidence": None,
                "rejection_reason": "natural_context_exhausted_without_complete_onset_apex_offset",
                "training_blockers": [
                    "arc_incomplete_and_no_predeclared_natural_context_remains",
                    "affect_judgment_not_admitted_before_complete_arc",
                    "license_training_admission_unconfirmed",
                ],
                "arc_review_record_sha256": value_sha256(arc),
                "affect_review_record_sha256": None,
            }
            rejected.append(record)

    if len(train_candidates) + len(needs) + len(rejected) != len(arc_queue):
        raise AssertionError("Qualification outputs do not cover all samples")
    for row in train_candidates + needs + rejected:
        if row["license_training_admission"] is not False or row["accepted_for_training"] is not False:
            raise AssertionError("Qualification output accidentally grants training admission")
        if row["emotion_supervision_mask"] and not (
            row.get("affect_review_evaluated") is True
            and row.get("affect_result") == "observable"
            and row.get("emotion_class") in AFFECT_CLASSES
        ):
            raise AssertionError("Emotion supervision escaped observable affect gate")

    output_root.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "train_candidate": output_root / "train_candidate.jsonl",
        "needs_review_or_expansion": output_root / "needs_review_or_expansion.jsonl",
        "reject": output_root / "reject.jsonl",
    }
    output_rows = {
        "train_candidate": train_candidates,
        "needs_review_or_expansion": needs,
        "reject": rejected,
    }
    for name, path in output_paths.items():
        atomic_jsonl(path, output_rows[name])
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": SUMMARY_KIND,
        "qualification_scope": (
            "candidate_only_pending_explicit_noncommercial_license_acceptance_and_formal_manifest"
        ),
        "license_training_admission": False,
        "accepted_for_training": False,
        "all_records_fail_closed": True,
        "affect_is_evaluated_only_after_complete_current_context": True,
        "ambiguous_or_not_observable_affect_is_never_pseudo_labeled": True,
        "fixed_duration_window_used": False,
        "native_variable_length": True,
        "inputs": {
            name: {"path": str(path), "sha256": input_hashes[name]}
            for name, path in supplied.items()
        }
        | {
            "arc_action_queue": {
                "path": str(arc_queue_path),
                "sha256": input_hashes["arc_action_queue"],
            },
            "affect_queue": {
                "path": str(affect_queue_path),
                "sha256": input_hashes["affect_queue"],
            },
        },
        "counts": {
            "input_samples": len(arc_queue),
            "train_candidate": len(train_candidates),
            "needs_review_or_expansion": len(needs),
            "reject": len(rejected),
            "affect_evaluated": len(train_candidates),
            "affect_intentionally_not_evaluated": len(needs) + len(rejected),
            "emotion_supervision_candidate": sum(
                row["emotion_supervision_mask"] for row in train_candidates
            ),
            "train_candidate_without_emotion_supervision": sum(
                not row["emotion_supervision_mask"] for row in train_candidates
            ),
            "semantic_text_supervision_candidate": 0,
            "accepted_for_training": 0,
        },
        "continuation_partition_counts": {
            "complete_current_context": len(complete),
            "expansion_requests": len(requests),
            "context_exhausted": len(exhausted),
        },
        "complete_context_affect_result_distribution": {
            key: complete_affect_results.get(key, 0)
            for key in ("observable", "ambiguous", "not_observable")
        },
        "complete_context_observable_affect_class_distribution": {
            key: complete_affect_classes.get(key, 0) for key in AFFECT_CLASSES
        },
        "outputs": {
            name: {
                "path": str(path),
                "records": len(output_rows[name]),
                "sha256": sha256_file(path),
            }
            for name, path in output_paths.items()
        },
        "validation_passed": True,
    }
    atomic_json(output_root / "summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-summary", type=Path, required=True)
    parser.add_argument("--hidden-summary", type=Path, required=True)
    parser.add_argument("--hidden-mapping", type=Path, required=True)
    parser.add_argument("--physical-pipeline-summary", type=Path, required=True)
    parser.add_argument("--physical-passed-manifest", type=Path, required=True)
    parser.add_argument("--arc-review-submission", type=Path, required=True)
    parser.add_argument("--arc-review-summary", type=Path, required=True)
    parser.add_argument("--affect-review-submission", type=Path, required=True)
    parser.add_argument("--affect-review-summary", type=Path, required=True)
    parser.add_argument("--continuation-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = merge_qualification(
        public_summary=args.public_summary,
        hidden_summary=args.hidden_summary,
        hidden_mapping=args.hidden_mapping,
        physical_pipeline_summary=args.physical_pipeline_summary,
        physical_passed_manifest=args.physical_passed_manifest,
        arc_review_submission=args.arc_review_submission,
        arc_review_summary=args.arc_review_summary,
        affect_review_submission=args.affect_review_submission,
        affect_review_summary=args.affect_review_summary,
        continuation_summary=args.continuation_summary,
        output_root=args.output_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
