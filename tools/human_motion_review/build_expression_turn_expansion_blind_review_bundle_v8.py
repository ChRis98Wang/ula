#!/usr/bin/env python3
"""Build anonymous blind-review queues for BEAT2 v8 context expansions.

Every public sample is the exact variable-length trajectory produced for the
next predeclared natural-boundary level.  This stage deliberately keeps action,
affect, licensing, and training admission closed.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
from pathlib import Path
from typing import Any

try:
    from tools.human_motion_review import build_expression_turn_blind_review_bundle_v8 as blind
    from tools.human_motion_review import build_expression_turn_video_queue_v8 as video_queue
    from tools.human_motion_review import render_beat2_annotation_review as renderer
    from tools.human_motion_review.expression_turn_contract import (
        ACTION_PROTOCOL,
        AFFECT_CLASSES,
        AFFECT_PROTOCOL,
        ARC_PROTOCOL,
    )
    from tools.human_motion_review.expression_turn_retarget_contract import (
        REQUIRED_18D_GATES,
        validate_safety_monotonic_retime,
    )
except ModuleNotFoundError:  # pragma: no cover - direct invocation by path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.human_motion_review import build_expression_turn_blind_review_bundle_v8 as blind
    from tools.human_motion_review import build_expression_turn_video_queue_v8 as video_queue
    from tools.human_motion_review import render_beat2_annotation_review as renderer
    from tools.human_motion_review.expression_turn_contract import (
        ACTION_PROTOCOL,
        AFFECT_CLASSES,
        AFFECT_PROTOCOL,
        ARC_PROTOCOL,
    )
    from tools.human_motion_review.expression_turn_retarget_contract import (
        REQUIRED_18D_GATES,
        validate_safety_monotonic_retime,
    )


SCHEMA_VERSION = "1.0.0"
PLAN_KIND = "expression_turn_v8_one_level_natural_context_expansion_plan"
REQUEST_KIND = "expression_turn_v8_one_level_natural_context_expansion_request"
PIPELINE_KIND = "beat2_expression_turn_v8_expansion_physical_pipeline_v1"
INPUT_AUDIT_KIND = "expression_turn_v8_expansion_input_audit_v1"
QUEUE_KIND = "expression_turn_v8_expansion_physical_review_queue_v1"
PROVENANCE_KIND = "expression_turn_v8_natural_context_expansion_lineage_v1"
SELECTION_POLICY = "exactly_one_next_predeclared_natural_context_level"
EXPANSION_UNIT = "one_predeclared_adjacent_natural_boundary_level"
PROCESSING_SCOPE = (
    "physical_retarget_and_silent_render_only_pending_fresh_blind_arc_action_"
    "and_affect_review"
)
EXPANSION_RENDER_AUDIT_FIELDS = (
    "expansion_provenance",
    "processing_scope",
    "semantic_supervision_mask",
    "license_training_admission",
    "license_training_admission_status",
    "retarget_result_json",
    "retarget_result_json_sha256",
)
SAMPLED_FIELDS_NOT_IN_RENDER_RESULT = {
    "input_record_sha256",
    "review_state",
    "sample_rank",
    "sampling",
    "sampling_seed",
    "sampling_stratum",
    "quality_json",
    "quality_json_sha256",
}
ARC_ACTION_PUBLIC_KEYS = blind.ARC_ACTION_PUBLIC_KEYS | {
    "native_duration_preserved",
    "fixed_duration_window_used",
    "frame_count",
    "fps",
}
AFFECT_PUBLIC_KEYS = blind.AFFECT_PUBLIC_KEYS | {
    "native_duration_preserved",
    "fixed_duration_window_used",
    "frame_count",
    "fps",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-summary", type=Path, required=True)
    parser.add_argument("--expansion-plan-summary", type=Path, required=True)
    parser.add_argument("--expansion-requests", type=Path, required=True)
    parser.add_argument("--render-passed-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--hidden-root", type=Path, required=True)
    parser.add_argument("--secret-hex")
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _file_binding(value: Any, *, context: str) -> tuple[Path, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} is not a file binding")
    path = Path(str(value.get("path") or "")).resolve()
    digest = str(value.get("sha256") or "")
    if not path.is_file() or blind.sha256(path) != digest:
        raise ValueError(f"{context} file binding mismatch")
    return path, digest


def _expected_sampled_projection(
    record: dict[str, Any], *, rank: int
) -> dict[str, Any]:
    projected = renderer.review_projection(
        record, rank=rank, sampling="sequential", seed=0
    )
    projected.update(
        {
            field: record[field]
            for field in EXPANSION_RENDER_AUDIT_FIELDS
            if field in record
        }
    )
    return projected


def _validate_queue_physical_evidence(
    record: dict[str, Any], queue_path: Path
) -> tuple[Path, int]:
    task_id = record.get("task_id")
    for field in ("quality_json", "quality_json_sha256"):
        if not isinstance(record.get(field), str) or not record[field]:
            raise ValueError(f"{task_id}: physical queue requires {field}")
    trajectory, parsed_frames = renderer.validate_trajectory(record, queue_path)
    quality_path = renderer.resolve_evidence_path(
        record["quality_json"], queue_path, "quality_json"
    )
    quality = _read_json(quality_path)
    gate = record.get("quality_gate")
    contract_validation = quality.get("expression_turn_output_contract_validation")
    if not isinstance(gate, dict):
        raise ValueError(f"{task_id}: physical queue requires quality_gate")
    missing_gates = sorted(REQUIRED_18D_GATES.difference(gate))
    if missing_gates:
        raise ValueError(
            f"{task_id}: physical quality evidence is missing required 18D gates: "
            f"{missing_gates}"
        )
    if (
        any(value is not True for value in gate.values())
        or quality.get("quality_gate") != gate
        or not isinstance(contract_validation, dict)
        or contract_validation.get("passed") is not True
        or quality.get("training_segment") != record.get("training_segment")
    ):
        raise ValueError(f"{task_id}: physical quality evidence is not all-pass")
    training = record.get("training_segment")
    retarget = record.get("retarget_segment")
    if not isinstance(training, dict) or not isinstance(retarget, dict):
        raise ValueError(f"{task_id}: missing natural/retarget segment contract")
    contract_record = dict(record)
    contract_record["frames"] = record.get("output_frame_count")
    output_frames, role, safety = video_queue._validate_final_output_contract(
        task_id=str(task_id),
        record=contract_record,
        training=training,
        retarget=retarget,
        quality=quality,
        trajectory=trajectory,
    )
    if safety is not None:
        validate_safety_monotonic_retime(
            safety,
            source_frames=int(training["frame_count"]),
            output_frames=output_frames,
            fps=float(record["fps"]),
            safe_csv_path=trajectory,
            safe_csv_sha256=str(record["trajectory_sha256"]),
        )
    if (
        parsed_frames != output_frames
        or record.get("trajectory_frames_expected") != output_frames
        or record.get("source_frame_count") != training.get("frame_count")
        or record.get("output_frame_count") != output_frames
        or record.get("final_trajectory_role") != role
        or record.get("blind_review_must_use_final_trajectory") is not True
        or record.get("safety_monotonic_retime") != safety
    ):
        raise ValueError(f"{task_id}: final physical trajectory contract mismatch")
    return trajectory, output_frames


def _index_requests(path: Path) -> dict[str, dict[str, Any]]:
    requests: dict[str, dict[str, Any]] = {}
    base_tasks: set[str] = set()
    for record in blind.read_jsonl(path):
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in requests:
            raise ValueError("expansion requests contain invalid or duplicate sample_id")
        required_strings = (
            "base_task_id",
            "source_clip_id",
            "fixed_split_assignment",
            "base_expression_turn_record_sha256",
            "comparison_record_sha256",
        )
        missing_strings = [
            field
            for field in required_strings
            if not isinstance(record.get(field), str) or not record[field].strip()
        ]
        if missing_strings:
            raise ValueError(f"{sample_id}: missing request lineage: {missing_strings}")
        base_task_id = record["base_task_id"]
        if base_task_id in base_tasks:
            raise ValueError(f"duplicate base_task_id in expansion requests: {base_task_id}")
        base_tasks.add(base_task_id)
        expected = record.get("plan_record_sha256")
        payload = dict(record)
        payload.pop("plan_record_sha256", None)
        exact = {
            "artifact_kind": REQUEST_KIND,
            "strictly_contains_reviewed_interval": True,
            "expansion_unit": EXPANSION_UNIT,
            "elapsed_duration_used_as_gate": False,
            "semantic_supervision_mask": False,
            "emotion_supervision_mask": False,
            "accepted_for_training": False,
        }
        failed = sorted(key for key, value in exact.items() if record.get(key) != value)
        if expected != blind.value_sha256(payload) or failed:
            raise ValueError(f"{sample_id}: invalid fail-closed expansion request: {failed}")
        reviewed = record.get("reviewed_context_level")
        requested = record.get("requested_context_level")
        if (
            isinstance(reviewed, bool)
            or not isinstance(reviewed, int)
            or reviewed < 0
            or requested != reviewed + 1
        ):
            raise ValueError(f"{sample_id}: expansion must advance exactly one context level")
        intervals: dict[str, dict[str, int]] = {}
        for name in ("reviewed_interval", "requested_interval"):
            interval = record.get(name)
            if not isinstance(interval, dict):
                raise ValueError(f"{sample_id}: {name} must be an interval object")
            values: dict[str, int] = {}
            for field in ("start_frame", "end_frame_exclusive", "frame_count"):
                value = interval.get(field)
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"{sample_id}: {name}.{field} must be an integer")
                values[field] = value
            start = values["start_frame"]
            end = values["end_frame_exclusive"]
            if start < 0 or end <= start or values["frame_count"] != end - start:
                raise ValueError(f"{sample_id}: invalid {name} frame arithmetic")
            intervals[name] = values
        reviewed_interval = intervals["reviewed_interval"]
        requested_interval = intervals["requested_interval"]
        contains = (
            requested_interval["start_frame"] <= reviewed_interval["start_frame"]
            and requested_interval["end_frame_exclusive"]
            >= reviewed_interval["end_frame_exclusive"]
            and (
                requested_interval["start_frame"] < reviewed_interval["start_frame"]
                or requested_interval["end_frame_exclusive"]
                > reviewed_interval["end_frame_exclusive"]
            )
        )
        if not contains:
            raise ValueError(
                f"{sample_id}: requested interval must strictly contain reviewed interval"
            )
        requests[sample_id] = record
    if not requests:
        raise ValueError("expansion request manifest is empty")
    return requests


def _validate_plan(
    plan_path: Path, requests_path: Path, requests: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    plan = _read_json(plan_path)
    if (
        plan.get("artifact_kind") != PLAN_KIND
        or plan.get("selection_policy") != SELECTION_POLICY
        or plan.get("fixed_minimum_maximum_or_target_duration_used") is not False
        or plan.get("accepted_for_training_count") != 0
    ):
        raise ValueError("expansion plan is not fail-closed")
    declared = (plan.get("outputs") or {}).get("expansion_requests") or {}
    if (
        Path(str(declared.get("path") or "")).resolve() != requests_path
        or declared.get("sha256") != blind.sha256(requests_path)
        or declared.get("records") != len(requests)
    ):
        raise ValueError("expansion plan does not bind the request manifest")
    return plan


def _validate_pipeline(
    pipeline_path: Path,
    *,
    plan_path: Path,
    requests_path: Path,
    passed_path: Path,
    requests: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    pipeline = _read_json(pipeline_path)
    exact = {
        "artifact_kind": PIPELINE_KIND,
        "selection_policy": SELECTION_POLICY,
        "fixed_six_second_windows_used": False,
        "elapsed_seconds_used_for_selection": False,
        "processing_scope": PROCESSING_SCOPE,
        "semantic_admission": False,
        "affect_admission": False,
        "license_training_admission": False,
        "accepted_for_training": False,
    }
    failed = sorted(key for key, value in exact.items() if pipeline.get(key) != value)
    if pipeline.get("stage") not in {"all", "render"}:
        failed.append("stage")
    if failed or pipeline.get("derived_candidate_count") != len(requests):
        raise ValueError(f"expansion pipeline is not complete/fail-closed: {failed}")
    audit = pipeline.get("input_audit")
    if not isinstance(audit, dict):
        raise ValueError("expansion pipeline lacks input audit")
    audit_hash = audit.get("sha256")
    audit_payload = dict(audit)
    audit_payload.pop("sha256", None)
    audit_exact = {
        "artifact_kind": INPUT_AUDIT_KIND,
        "selection_policy": SELECTION_POLICY,
        "fixed_minimum_maximum_or_target_duration_used": False,
        "semantic_admission": False,
        "affect_admission": False,
        "license_training_admission": False,
        "accepted_for_training": False,
    }
    audit_failed = sorted(
        key for key, value in audit_exact.items() if audit.get(key) != value
    )
    if audit_hash != blind.value_sha256(audit_payload) or audit_failed:
        raise ValueError(f"expansion input audit mismatch: {audit_failed}")
    inputs = audit.get("inputs") or {}
    bound_plan, _ = _file_binding(inputs.get("expansion_plan_summary"), context="plan")
    bound_requests, _ = _file_binding(inputs.get("expansion_requests"), context="requests")
    if bound_plan != plan_path or bound_requests != requests_path:
        raise ValueError("expansion pipeline input paths changed")

    queue = pipeline.get("review_queue_summary")
    render = pipeline.get("render_summary")
    if not isinstance(queue, dict) or not isinstance(render, dict):
        raise ValueError("expansion render summaries are missing")
    queue_path, queue_hash = _file_binding(
        {"path": queue.get("output"), "sha256": queue.get("output_sha256")},
        context="physical review queue",
    )
    queue_exact = {
        "artifact_kind": QUEUE_KIND,
        "selection_policy": SELECTION_POLICY,
        "elapsed_seconds_used_for_selection": False,
        "processing_scope": PROCESSING_SCOPE,
        "semantic_admission": False,
        "affect_admission": False,
        "license_training_admission": False,
        "accepted_for_training": False,
    }
    queue_failed = sorted(
        key for key, value in queue_exact.items() if queue.get(key) != value
    )
    queue_records = blind.read_jsonl(queue_path)
    queue_by_task: dict[str, dict[str, Any]] = {}
    queue_trajectory_by_task: dict[str, tuple[Path, int]] = {}
    for record in queue_records:
        task_id = record.get("task_id")
        if not isinstance(task_id, str) or not task_id or task_id in queue_by_task:
            raise ValueError("physical review queue contains invalid or duplicate task_id")
        queue_trajectory_by_task[task_id] = _validate_queue_physical_evidence(
            record, queue_path
        )
        queue_by_task[task_id] = record
    expected_tasks = {
        f"{request['base_task_id']}__ctxL{request['requested_context_level']:02d}"
        for request in requests.values()
    }
    if not set(queue_by_task).issubset(expected_tasks):
        raise ValueError("physical review queue contains an unrequested expansion task")
    if queue.get("records") != len(queue_records):
        raise ValueError("physical review queue summary count mismatch")

    sampling = render.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError("expansion render lacks sampled manifest binding")
    sampled_path, _sampled_hash = _file_binding(
        {
            "path": sampling.get("sampled_manifest"),
            "sha256": sampling.get("sampled_manifest_sha256"),
        },
        context="render sampled manifest",
    )
    sampled_records = blind.read_jsonl(sampled_path)
    sampled_by_task: dict[str, dict[str, Any]] = {}
    for record in sampled_records:
        task_id = record.get("task_id")
        if not isinstance(task_id, str) or not task_id or task_id in sampled_by_task:
            raise ValueError("sampled manifest contains invalid or duplicate task_id")
        sampled_by_task[task_id] = record
    selected_queue_records = renderer.select_records(
        queue_records, limit=None, sampling="sequential", seed=0
    )
    expected_sampled_records = [
        _expected_sampled_projection(record, rank=rank)
        for rank, record in enumerate(selected_queue_records)
    ]
    if (
        sampling.get("mode") != "sequential"
        or sampling.get("seed") != 0
        or sampling.get("limit") is not None
        or sampling.get("selected_records") != len(queue_records)
        or set(sampled_by_task) != set(queue_by_task)
        or sampled_records != expected_sampled_records
    ):
        raise ValueError("sampled manifest differs from the exact physical queue projection")

    bound_passed_path, _passed_hash = _file_binding(
        {
            "path": render.get("passed_manifest"),
            "sha256": render.get("passed_manifest_sha256"),
        },
        context="render passed manifest",
    )
    failed_path, _failed_hash = _file_binding(
        {
            "path": render.get("failed_manifest"),
            "sha256": render.get("failed_manifest_sha256"),
        },
        context="render failed manifest",
    )
    if bound_passed_path != passed_path or blind.read_jsonl(failed_path):
        raise ValueError("render terminal manifests are incomplete or changed")

    passed_records = blind.read_jsonl(passed_path)
    passed_by_task: dict[str, dict[str, Any]] = {}
    for record in passed_records:
        task_id = record.get("task_id")
        if not isinstance(task_id, str) or not task_id or task_id in passed_by_task:
            raise ValueError("render passed manifest contains invalid or duplicate task_id")
        passed_by_task[task_id] = record
    if set(passed_by_task) != set(queue_by_task):
        raise ValueError("render passed task set differs from the physical review queue")
    for task_id, record in passed_by_task.items():
        queue_record = queue_by_task[task_id]
        sampled_record = sampled_by_task[task_id]
        inherited_mismatches = sorted(
            field
            for field, value in sampled_record.items()
            if field not in SAMPLED_FIELDS_NOT_IN_RENDER_RESULT
            and record.get(field) != value
        )
        queue_trajectory, queue_frames = queue_trajectory_by_task[task_id]
        passed_trajectory = renderer.resolve_evidence_path(
            record.get("trajectory_path"), passed_path, "trajectory_path"
        )
        if record.get("input_fingerprint") != blind.value_sha256(queue_record):
            raise ValueError(f"{task_id}: render input fingerprint mismatch")
        if (
            inherited_mismatches
            or passed_trajectory != queue_trajectory
            or record.get("trajectory_sha256")
            != queue_record.get("trajectory_sha256")
            or record.get("trajectory_frames") != queue_frames
            or record.get("trajectory_frames_expected") != queue_frames
        ):
            raise ValueError(
                f"{task_id}: render result differs from queued trajectory/input: "
                f"{inherited_mismatches}"
            )

    counts = render.get("counts") or {}
    if (
        queue_failed
        or render.get("stage") != "beat2_annotation_review_video_render"
        or render.get("run_state") != "finished"
        or Path(str(render.get("review_queue") or "")).resolve() != queue_path
        or render.get("review_queue_sha256") != queue_hash
        or render.get("render_pass_grants_training_admission") is not False
        or render.get("accepted_for_training") != 0
        or render.get("queue_records") != len(queue_records)
        or counts.get("passed") != queue.get("records")
        or counts.get("failed") != 0
        or bound_passed_path != passed_path
    ):
        raise ValueError(f"expansion render is incomplete or unbound: {queue_failed}")
    return pipeline, queue_by_task, passed_records


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _validate_existing_public_tree(
    public_root: Path, expected_videos: dict[str, str]
) -> None:
    if not public_root.exists():
        return
    if public_root.is_symlink() or not public_root.is_dir():
        raise ValueError(f"public bundle root must be a real directory: {public_root}")
    allowed = {
        "videos",
        "arc_action_review_queue.jsonl",
        "affect_review_queue.jsonl",
        "summary.json",
    }
    unexpected = sorted(entry.name for entry in public_root.iterdir() if entry.name not in allowed)
    if unexpected:
        raise ValueError(f"stale unexpected public bundle entries: {unexpected}")
    for name in allowed.difference({"videos"}):
        path = public_root / name
        if path.exists() and (path.is_symlink() or not path.is_file()):
            raise ValueError(f"public bundle artifact is not a regular file: {path}")
    videos_root = public_root / "videos"
    if not videos_root.exists():
        return
    if videos_root.is_symlink() or not videos_root.is_dir():
        raise ValueError(f"public videos path must be a real directory: {videos_root}")
    actual_names = {entry.name for entry in videos_root.iterdir()}
    unexpected_videos = sorted(actual_names.difference(expected_videos))
    if unexpected_videos:
        raise ValueError(f"stale unexpected public videos: {unexpected_videos}")
    for entry in videos_root.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(f"public video is not a regular file: {entry}")
        if blind.sha256(entry) != expected_videos[entry.name]:
            raise ValueError(f"existing public video hash mismatch: {entry}")


def _validate_materialized_videos(
    public_root: Path, expected_videos: dict[str, str]
) -> None:
    videos_root = public_root / "videos"
    actual_names = {entry.name for entry in videos_root.iterdir()}
    if actual_names != set(expected_videos):
        raise ValueError("materialized public video set is incomplete or contains stale files")
    for entry in videos_root.iterdir():
        if (
            entry.is_symlink()
            or not entry.is_file()
            or entry.stat().st_nlink != 1
            or blind.sha256(entry) != expected_videos[entry.name]
        ):
            raise ValueError(f"materialized public video is not immutable: {entry}")


def _preview_secret(hidden_root: Path, provided_hex: str | None) -> bytes:
    """Resolve the stable anonymization key without creating output artifacts."""

    secret_path = hidden_root / "bundle_secret.json"
    if secret_path.exists():
        if secret_path.is_symlink() or not secret_path.is_file():
            raise ValueError(f"bundle secret path is not a file: {secret_path}")
        try:
            value = json.loads(secret_path.read_text(encoding="utf-8"))
            existing = bytes.fromhex(str(value["secret_hex"]))
        except (KeyError, OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid existing bundle secret: {secret_path}") from error
        if provided_hex is not None:
            try:
                provided = bytes.fromhex(provided_hex)
            except ValueError as error:
                raise ValueError("secret-hex must be valid hexadecimal") from error
            if provided != existing:
                raise ValueError("provided secret does not match existing bundle secret")
        secret = existing
    else:
        try:
            secret = bytes.fromhex(provided_hex) if provided_hex else secrets.token_bytes(32)
        except ValueError as error:
            raise ValueError("secret-hex must be valid hexadecimal") from error
    if len(secret) < 16:
        raise ValueError("bundle secret must contain at least 16 bytes")
    return secret


def _validate_expansion_render(
    record: dict[str, Any],
    *,
    request: dict[str, Any],
    plan_sha256: str,
    requests_sha256: str,
    evidence_manifest_path: Path,
) -> tuple[str, Path, str, int, int]:
    task_id, video, video_hash, context_level = blind._validated_final_render_evidence(
        record, evidence_manifest_path
    )
    provenance = record.get("expansion_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{task_id}: expansion provenance is missing")
    expected_task = f"{request['base_task_id']}__ctxL{request['requested_context_level']:02d}"
    exact = {
        "task_id": expected_task,
        "artifact_kind": PROVENANCE_KIND,
        "sample_id": request["sample_id"],
        "base_task_id": request["base_task_id"],
        "base_expression_turn_record_sha256": request[
            "base_expression_turn_record_sha256"
        ],
        "expansion_plan_summary_sha256": plan_sha256,
        "expansion_requests_sha256": requests_sha256,
        "plan_record_sha256": request["plan_record_sha256"],
        "comparison_record_sha256": request["comparison_record_sha256"],
        "reviewed_context_level": request["reviewed_context_level"],
        "requested_context_level": request["requested_context_level"],
        "selection_policy": SELECTION_POLICY,
        "expansion_unit": EXPANSION_UNIT,
        "elapsed_duration_used_as_gate": False,
        "physical_quality_only": True,
        "semantic_admission": False,
        "affect_admission": False,
        "license_training_admission": False,
        "accepted_for_training": False,
    }
    failed = []
    for key, value in exact.items():
        actual = record.get(key) if key == "task_id" else provenance.get(key)
        if actual != value:
            failed.append(key)
    segment = record.get("training_segment") or {}
    retarget = record.get("retarget_segment") or {}
    requested_interval = request["requested_interval"]
    frames = record.get("trajectory_frames")
    source_frames = requested_interval["frame_count"]
    if (
        record.get("processing_scope") != PROCESSING_SCOPE
        or record.get("license_training_admission") is not False
        or context_level != request["requested_context_level"]
        or segment.get("start_frame") != requested_interval["start_frame"]
        or segment.get("end_frame_exclusive") != requested_interval["end_frame_exclusive"]
        or segment.get("frame_count") != requested_interval["frame_count"]
        or segment.get("fixed_window_sec") is not None
        or segment.get("cropped") is not False
        or retarget.get("source_frame_count") != source_frames
        or retarget.get("output_frame_count") != frames
        or retarget.get("fixed_target_duration_sec") is not None
        or retarget.get("cropped") is not False
        or isinstance(frames, bool)
        or not isinstance(frames, int)
        or frames < source_frames
    ):
        failed.append("native_interval_binding")
    if failed:
        raise ValueError(f"{task_id}: expansion evidence mismatch: {sorted(set(failed))}")
    return task_id, video, video_hash, context_level, int(frames)


def build_bundle(
    *,
    pipeline_summary: Path,
    expansion_plan_summary: Path,
    expansion_requests: Path,
    render_passed_manifest: Path,
    output_root: Path,
    hidden_root: Path,
    secret_hex: str | None = None,
) -> dict[str, Any]:
    paths = [pipeline_summary, expansion_plan_summary, expansion_requests, render_passed_manifest]
    pipeline_summary, expansion_plan_summary, expansion_requests, render_passed_manifest = [
        path.resolve() for path in paths
    ]
    for path in (
        pipeline_summary,
        expansion_plan_summary,
        expansion_requests,
        render_passed_manifest,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    requests = _index_requests(expansion_requests)
    _validate_plan(expansion_plan_summary, expansion_requests, requests)
    pipeline, queue_by_task, passed_records = _validate_pipeline(
        pipeline_summary,
        plan_path=expansion_plan_summary,
        requests_path=expansion_requests,
        passed_path=render_passed_manifest,
        requests=requests,
    )

    output_root = output_root.resolve()
    public_entry = output_root / "public"
    if public_entry.is_symlink():
        raise ValueError("public bundle root must not be a symbolic link")
    public_root = public_entry.resolve()
    hidden_root = hidden_root.resolve()
    if _paths_overlap(public_root, hidden_root):
        raise ValueError("public and hidden bundle roots must be disjoint")
    for input_path in (
        pipeline_summary,
        expansion_plan_summary,
        expansion_requests,
        render_passed_manifest,
    ):
        if public_root == input_path or public_root in input_path.parents:
            raise ValueError("public bundle root overlaps an input artifact")
        if hidden_root == input_path or hidden_root in input_path.parents:
            raise ValueError("hidden bundle root overlaps an input artifact")
    plan_hash = blind.sha256(expansion_plan_summary)
    request_hash = blind.sha256(expansion_requests)
    passed_hash = blind.sha256(render_passed_manifest)

    arc_rows: list[dict[str, Any]] = []
    affect_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    validated_sources: list[tuple[str, Path, str, int, int, dict[str, Any], dict[str, Any]]] = []
    seen_base_samples: set[str] = set()
    for record in passed_records:
        provenance = record.get("expansion_provenance") or {}
        base_sample_id = provenance.get("sample_id")
        request = requests.get(base_sample_id)
        if request is None or base_sample_id in seen_base_samples:
            raise ValueError(f"unexpected or duplicate expansion sample: {base_sample_id}")
        seen_base_samples.add(base_sample_id)
        task_id, source_video, video_hash, context_level, frames = _validate_expansion_render(
            record,
            request=request,
            plan_sha256=plan_hash,
            requests_sha256=request_hash,
            evidence_manifest_path=render_passed_manifest,
        )
        if task_id not in queue_by_task:
            raise ValueError(f"{task_id}: render record is absent from physical queue")
        validated_sources.append(
            (task_id, source_video, video_hash, context_level, frames, request, record)
        )
    expected_render_passes = int(pipeline["review_queue_summary"]["records"])
    if len(seen_base_samples) != expected_render_passes:
        raise ValueError("render passed manifest count differs from the physical review queue")
    if blind.sha256(render_passed_manifest) != passed_hash:
        raise ValueError("render passed manifest changed during validation")

    secret = _preview_secret(hidden_root, secret_hex)
    materializations: list[tuple[Path, Path, str]] = []
    for task_id, source_video, video_hash, context_level, frames, request, record in validated_sources:
        sample_id = blind._anonymous_id(secret, task_id, video_hash)
        anonymous_video = public_root / "videos" / f"{sample_id}.mp4"
        materializations.append((source_video, anonymous_video, video_hash))
        common = {
            "schema_version": SCHEMA_VERSION,
            "sample_id": sample_id,
            "video_path": str(anonymous_video.resolve()),
            "video_sha256": video_hash,
            "context_level": context_level,
            "native_duration_preserved": True,
            "fixed_duration_window_used": False,
            "frame_count": frames,
            "fps": 30.0,
            "audio_available": False,
            "label_metadata_exposed": False,
        }
        arc = {
            **common,
            "arc_protocol_version": ARC_PROTOCOL,
            "arc_review_id": None,
            "arc_reviewer_id": None,
            "onset_status": None,
            "onset_evidence_frame": None,
            "onset_basis": None,
            "apex_status": None,
            "apex_evidence_frame": None,
            "apex_basis": None,
            "offset_status": None,
            "offset_evidence_frame": None,
            "offset_basis": None,
            "action_protocol_version": ACTION_PROTOCOL,
            "action_review_id": None,
            "action_reviewer_id": None,
            "action_result": None,
            "observable_description": None,
            "candidate_text": None,
            "candidate_text_sha256": None,
            "candidate_text_provenance": None,
        }
        affect = {
            **common,
            "affect_protocol_version": AFFECT_PROTOCOL,
            "affect_review_id": None,
            "affect_reviewer_id": None,
            "allowed_classes": sorted(AFFECT_CLASSES),
            "result": None,
            "predicted_class": None,
            "confidence": None,
        }
        blind._assert_public_privacy(arc, ARC_ACTION_PUBLIC_KEYS)
        blind._assert_public_privacy(affect, AFFECT_PUBLIC_KEYS)
        arc_rows.append(arc)
        affect_rows.append(affect)
        hidden_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": sample_id,
                # Use the request paired with this materialized task.  The
                # validation loop above has a different lifetime; reusing its
                # local base_sample_id here would stamp every row with the
                # final task's sample ID.
                "base_sample_id": request["sample_id"],
                "task_id": task_id,
                "derived_task_id": task_id,
                "base_task_id": request["base_task_id"],
                "source_clip_id": request["source_clip_id"],
                "fixed_split_assignment": request["fixed_split_assignment"],
                "reviewed_context_level": request["reviewed_context_level"],
                "displayed_context_level": context_level,
                "reviewed_interval": request["reviewed_interval"],
                "displayed_interval": request["requested_interval"],
                "plan_record_sha256": request["plan_record_sha256"],
                "comparison_record_sha256": request["comparison_record_sha256"],
                "source_render_record_sha256": blind.value_sha256(record),
                "video_sha256": video_hash,
                "frame_count": frames,
                "trajectory_sha256": record["trajectory_sha256"],
                "native_duration_preserved": True,
                "fixed_duration_window_used": False,
                "official_action_text_or_affect_exposed": False,
                "accepted_for_training": False,
            }
        )
    physical_exclusions = sorted(set(requests).difference(seen_base_samples))

    for rows in (arc_rows, affect_rows, hidden_rows):
        rows.sort(key=lambda row: row["sample_id"])
    expected_videos = {
        target.name: digest for _source, target, digest in materializations
    }
    _validate_existing_public_tree(public_root, expected_videos)

    public_root.mkdir(parents=True, exist_ok=True)
    hidden_root.mkdir(parents=True, exist_ok=True)
    if public_root.resolve() != public_root or _paths_overlap(
        public_root.resolve(), hidden_root.resolve()
    ):
        raise ValueError("created public and hidden roots are not disjoint real paths")
    os.chmod(hidden_root, 0o700)
    blind._secret(hidden_root, secret.hex())
    for source, target, digest in materializations:
        blind._materialize_video(source, target, digest)
    _validate_materialized_videos(public_root, expected_videos)
    arc_path = public_root / "arc_action_review_queue.jsonl"
    affect_path = public_root / "affect_review_queue.jsonl"
    hidden_path = hidden_root / "sample_mapping.jsonl"
    blind.atomic_jsonl(arc_path, arc_rows)
    blind.atomic_jsonl(affect_path, affect_rows)
    blind.atomic_jsonl(hidden_path, hidden_rows, mode=0o600)
    public_summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "expression_turn_v8_expansion_separate_blind_review_bundle_v1",
        "records": len(arc_rows),
        "arc_action_queue": str(arc_path),
        "arc_action_queue_sha256": blind.sha256(arc_path),
        "affect_queue": str(affect_path),
        "affect_queue_sha256": blind.sha256(affect_path),
        "affect_ontology": sorted(AFFECT_CLASSES),
        "duration_policy": "one_next_predeclared_natural_boundary_level_no_fixed_window",
        "all_samples_native_variable_length": True,
        "fixed_duration_window_used": False,
        "same_anonymous_silent_video_used_by_both_reviews": True,
        "action_and_affect_reviewers_must_be_independent": True,
        "source_identity_official_action_text_and_affect_exposed": False,
        "accepted_for_training": False,
    }
    hidden_summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "expression_turn_v8_expansion_hidden_blind_mapping_v1",
        "records": len(hidden_rows),
        "requested_records": len(requests),
        "physical_qc_excluded_records": len(physical_exclusions),
        "physical_qc_excluded_base_sample_ids": physical_exclusions,
        "mapping": str(hidden_path),
        "mapping_sha256": blind.sha256(hidden_path),
        "pipeline_summary": str(pipeline_summary),
        "pipeline_summary_sha256": blind.sha256(pipeline_summary),
        "expansion_plan_summary": str(expansion_plan_summary),
        "expansion_plan_summary_sha256": plan_hash,
        "expansion_requests": str(expansion_requests),
        "expansion_requests_sha256": request_hash,
        "render_passed_manifest": str(render_passed_manifest),
        "render_passed_manifest_sha256": passed_hash,
        "public_distribution_forbidden": True,
        "accepted_for_training": False,
    }
    blind.atomic_json(public_root / "summary.json", public_summary)
    blind.atomic_json(hidden_root / "summary.json", hidden_summary, mode=0o600)
    return {"public": public_summary, "hidden": hidden_summary}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_bundle(
        pipeline_summary=args.pipeline_summary,
        expansion_plan_summary=args.expansion_plan_summary,
        expansion_requests=args.expansion_requests,
        render_passed_manifest=args.render_passed_manifest,
        output_root=args.output_root,
        hidden_root=args.hidden_root,
        secret_hex=args.secret_hex,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
