#!/usr/bin/env python3
"""Build fail-closed round-N BEAT2 v8 natural-boundary continuation plans.

The builder consumes one rendered expansion round and its independent blind
arc/action review.  It either keeps the reviewed natural context, advances by
exactly one catalog-declared natural-boundary level, or records that the
catalog context is exhausted.  Elapsed seconds and arbitrary frame counts are
never selection inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any

try:
    from tools.human_motion_review.expression_turn_contract import (
        CONTEXT_POLICY,
        LEGACY_PILOT_LINEAGE_FIELDS,
        SELECTION_LINEAGE_FIELDS,
    )
except ModuleNotFoundError:  # pragma: no cover - direct invocation by path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.human_motion_review.expression_turn_contract import (
        CONTEXT_POLICY,
        LEGACY_PILOT_LINEAGE_FIELDS,
        SELECTION_LINEAGE_FIELDS,
    )


SCHEMA_VERSION = "1.0.0"
PLAN_KIND = "expression_turn_v8_one_level_natural_context_expansion_plan"
REQUEST_KIND = "expression_turn_v8_one_level_natural_context_expansion_request"
COMPLETE_KIND = "expression_turn_v8_current_context_arc_action_complete"
EXHAUSTED_KIND = "expression_turn_v8_natural_context_exhausted"
PUBLIC_KIND = "expression_turn_v8_expansion_separate_blind_review_bundle_v1"
HIDDEN_KIND = "expression_turn_v8_expansion_hidden_blind_mapping_v1"
REVIEW_SUMMARY_KIND = (
    "beat2_expression_turn_v8_expansion_arc_action_review_submission_summary_v1"
)
PIPELINE_KIND = "beat2_expression_turn_v8_expansion_physical_pipeline_v1"
SELECTION_POLICY = "exactly_one_next_predeclared_natural_context_level"
EXPANSION_UNIT = "one_predeclared_adjacent_natural_boundary_level"
PUBLIC_DURATION_POLICY = "one_next_predeclared_natural_boundary_level_no_fixed_window"
NATURAL_DURATION_POLICY = "natural_rest_to_natural_rest_no_fixed_or_max_duration"
PHASES = ("onset", "apex", "offset")
PHASE_STATUSES = {"complete", "incomplete", "ambiguous"}
INCOMPLETE_PHASE_STATUSES = {"incomplete", "ambiguous"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-summary", type=Path, required=True)
    parser.add_argument("--arc-review-submission", type=Path, required=True)
    parser.add_argument("--arc-review-summary", type=Path, required=True)
    parser.add_argument("--expansion-hidden-mapping", type=Path, required=True)
    parser.add_argument("--expansion-hidden-summary", type=Path, required=True)
    parser.add_argument("--base-hidden-mapping", type=Path, required=True)
    parser.add_argument("--previous-plan-summary", type=Path, required=True)
    parser.add_argument("--previous-expansion-requests", type=Path, required=True)
    parser.add_argument("--candidate-catalog", type=Path, required=True)
    parser.add_argument("--catalog-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


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
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
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
    rows: list[tuple[dict[str, Any], str]], key: str, *, context: str
) -> dict[str, tuple[dict[str, Any], str]]:
    result: dict[str, tuple[dict[str, Any], str]] = {}
    for row, line_sha256 in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value or value in result:
            raise ValueError(f"{context} contains invalid or duplicate {key}")
        result[value] = (row, line_sha256)
    return result


def index_rows(
    rows: list[tuple[dict[str, Any], str]], key: str, *, context: str
) -> dict[str, dict[str, Any]]:
    return {
        value: row
        for value, (row, _line_sha256) in index_bound(
            rows, key, context=context
        ).items()
    }


def atomic_json(path: Path, value: object) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(stable_json(row) + "\n" for row in rows).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _declared_path(owner: Path, value: Any, *, field: str) -> Path:
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


def _file_binding_path(value: Any, owner: Path, *, context: str) -> Path:
    if not isinstance(value, dict):
        raise ValueError(f"{context} is not a file binding")
    path = _declared_path(owner, value.get("path"), field=f"{context}.path")
    if not path.is_file() or value.get("sha256") != sha256_file(path):
        raise ValueError(f"{context} file binding mismatch")
    return path


def _interval(value: Any, *, context: str) -> dict[str, int]:
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


def _contains(outer: dict[str, int], inner: dict[str, int]) -> bool:
    return (
        outer["start_frame"] <= inner["start_frame"]
        and inner["end_frame_exclusive"] <= outer["end_frame_exclusive"]
    )


def _strictly_expands(current: dict[str, int], requested: dict[str, int]) -> bool:
    return _contains(requested, current) and requested != current


def _verify_record_sha(row: dict[str, Any], field: str, *, context: str) -> str:
    expected = row.get(field)
    payload = dict(row)
    payload.pop(field, None)
    actual = value_sha256(payload)
    if expected != actual:
        raise ValueError(f"{context} record SHA mismatch")
    return actual


def _context_levels(candidate: dict[str, Any]) -> dict[int, dict[str, int]]:
    task_id = candidate.get("task_id")
    context = candidate.get("context_plan")
    if not isinstance(context, dict):
        raise ValueError(f"Candidate lacks context plan: {task_id}")
    if (
        context.get("policy") != CONTEXT_POLICY
        or context.get("same_source_only") is not True
        or context.get("neighbor_crossing_allowed") is not False
    ):
        raise ValueError(f"Candidate permits non-natural or neighbor-crossing context: {task_id}")
    if context.get("selected_level") != 0:
        raise ValueError(f"Base candidate selected level is not zero: {task_id}")
    source = _interval(context.get("source_interval"), context=f"{task_id}.source_interval")
    admissible = _interval(
        context.get("admissible_interval"), context=f"{task_id}.admissible_interval"
    )
    if not _contains(source, admissible):
        raise ValueError(f"Admissible interval escapes source: {task_id}")
    levels: dict[int, dict[str, int]] = {}
    previous: dict[str, int] | None = None
    raw_levels = context.get("levels")
    if not isinstance(raw_levels, list) or not raw_levels:
        raise ValueError(f"Candidate has no context levels: {task_id}")
    for index, raw in enumerate(raw_levels):
        if not isinstance(raw, dict) or raw.get("level") != index:
            raise ValueError(f"Candidate context levels are not contiguous: {task_id}")
        interval = _interval(raw, context=f"{task_id}.level[{index}]")
        if not _contains(admissible, interval):
            raise ValueError(f"Context level crosses source/neighbor barrier: {task_id}")
        if previous is not None:
            if raw.get("parent_level") != index - 1 or not _strictly_expands(
                previous, interval
            ):
                raise ValueError(f"Context level does not strictly follow its parent: {task_id}")
        levels[index] = interval
        previous = interval
    exhausted = context.get("context_exhausted_at_level")
    if exhausted != max(levels):
        raise ValueError(f"Context exhaustion marker differs from final level: {task_id}")
    return levels


def _verify_candidate_hash(candidate: dict[str, Any]) -> None:
    expected = candidate.get("expression_turn_record_sha256")
    payload = {
        key: value
        for key, value in candidate.items()
        if key
        not in (
            {"expression_turn_record_sha256"}
            | SELECTION_LINEAGE_FIELDS
            | LEGACY_PILOT_LINEAGE_FIELDS
        )
    }
    if expected != value_sha256(payload):
        raise ValueError(f"Candidate record SHA mismatch: {candidate.get('task_id')}")


def _load_catalog(
    candidate_catalog: Path, catalog_summary: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    summary = read_json(catalog_summary)
    catalog_hash = sha256_file(candidate_catalog)
    contract = summary.get("expression_turn_contract")
    if (
        summary.get("artifact_kind") != "beat2_expression_turn_v8_candidate_catalog"
        or summary.get("accepted_for_training") not in (0, False)
        or summary.get("fixed_window_sec") is not None
        or not isinstance(contract, dict)
        or contract.get("duration_policy") != NATURAL_DURATION_POLICY
        or contract.get("fixed_window_sec") is not None
        or summary.get("expression_turn_contract_sha256") != value_sha256(contract)
        or (summary.get("output_sha256") or {}).get(candidate_catalog.name)
        != catalog_hash
    ):
        raise ValueError("Candidate catalog summary is unbound or duration-gated")
    rows = index_rows(
        read_bound_jsonl(candidate_catalog), "task_id", context="candidate catalog"
    )
    if summary.get("candidate_count") != len(rows):
        raise ValueError("Candidate catalog count differs from its summary")
    return rows, summary


def _verify_previous_request(row: dict[str, Any]) -> None:
    sample_id = row.get("sample_id")
    _verify_record_sha(row, "plan_record_sha256", context=f"previous request {sample_id}")
    exact = {
        "artifact_kind": REQUEST_KIND,
        "strictly_contains_reviewed_interval": True,
        "expansion_unit": EXPANSION_UNIT,
        "elapsed_duration_used_as_gate": False,
        "semantic_supervision_mask": False,
        "emotion_supervision_mask": False,
        "accepted_for_training": False,
    }
    failed = sorted(key for key, value in exact.items() if row.get(key) != value)
    reviewed = row.get("reviewed_context_level")
    requested = row.get("requested_context_level")
    if (
        failed
        or isinstance(reviewed, bool)
        or not isinstance(reviewed, int)
        or requested != reviewed + 1
        or not _strictly_expands(
            _interval(row.get("reviewed_interval"), context=f"{sample_id}.reviewed"),
            _interval(row.get("requested_interval"), context=f"{sample_id}.requested"),
        )
    ):
        raise ValueError(f"Invalid previous natural-context request: {sample_id}: {failed}")


def _verify_review_summary(
    summary_path: Path,
    summary: dict[str, Any],
    *,
    review_submission: Path,
    public_summary: Path,
    queue_path: Path,
    reviews: dict[str, dict[str, Any]],
) -> None:
    exact = {
        "artifact_kind": REVIEW_SUMMARY_KIND,
        "fixed_duration_window_used": False,
        "elapsed_duration_used_as_gate": False,
        "native_variable_length_reviewed": True,
        "validation_passed": True,
    }
    failed = sorted(key for key, value in exact.items() if summary.get(key) != value)
    if failed or summary.get("records") != len(reviews):
        raise ValueError(f"Arc/action review summary violates contract: {failed}")
    declared = {
        "submission_path": review_submission,
        "public_summary_path": public_summary,
        "public_queue_path": queue_path,
    }
    for field, expected in declared.items():
        if _declared_path(summary_path, summary.get(field), field=field) != expected:
            raise ValueError(f"Arc/action review summary path mismatch: {field}")
    hashes = {
        "submission_sha256": sha256_file(review_submission),
        "public_summary_sha256": sha256_file(public_summary),
        "public_queue_sha256": sha256_file(queue_path),
    }
    if any(summary.get(key) != value for key, value in hashes.items()):
        raise ValueError("Arc/action review summary hash binding mismatch")
    distributions = {
        "action_result_distribution": "action_result",
        "action_observability_distribution": "action_observability",
        "onset_status_distribution": "onset_status",
        "apex_status_distribution": "apex_status",
        "offset_status_distribution": "offset_status",
    }
    for summary_field, row_field in distributions.items():
        expected = dict(sorted(Counter(row[row_field] for row in reviews.values()).items()))
        if summary.get(summary_field) != expected:
            raise ValueError(f"Arc/action review summary distribution mismatch: {summary_field}")
    if summary.get("decoded_frame_count_total") != sum(
        int(row["decoded_frame_count"]) for row in reviews.values()
    ):
        raise ValueError("Arc/action review decoded-frame total mismatch")


def _verify_review(
    row: dict[str, Any],
    *,
    queue: dict[str, Any],
    queue_sha256: str,
) -> tuple[dict[str, str], list[int]]:
    sample_id = row.get("sample_id")
    exact = {
        "sample_id": queue.get("sample_id"),
        "video_path": queue.get("video_path"),
        "video_sha256": queue.get("video_sha256"),
        "context_level": queue.get("context_level"),
        "frame_count": queue.get("frame_count"),
        "fps": queue.get("fps"),
        "queue_sha256": queue_sha256,
        "action_result": "pass",
        "action_observability": "observable",
        "audio_available": False,
        "label_metadata_exposed": False,
        "native_duration_preserved": True,
        "fixed_duration_window_used": False,
        "full_decode_to_eof": True,
        "emotion_judgment_performed": False,
        "training_admission": False,
    }
    failed = sorted(key for key, value in exact.items() if row.get(key) != value)
    ordered_review_performed = (
        row.get("ordered_contact_sheet_review_performed") is True
        or row.get("ordered_full_video_review_performed") is True
    )
    if (
        failed
        or not ordered_review_performed
        or row.get("decoded_frame_count") != queue.get("frame_count")
    ):
        raise ValueError(f"Arc/action review mismatch for {sample_id}: {failed}")
    if row.get("elapsed_duration_used_as_gate") not in (None, False):
        raise ValueError(f"Arc/action review used elapsed duration: {sample_id}")
    if not isinstance(row.get("observable_description"), str) or not row[
        "observable_description"
    ].strip():
        raise ValueError(f"Arc/action review lacks observable description: {sample_id}")
    statuses: dict[str, str] = {}
    evidence: list[int] = []
    frame_count = int(queue["frame_count"])
    for phase in PHASES:
        status = row.get(f"{phase}_status")
        frame = row.get(f"{phase}_evidence_frame")
        basis = row.get(f"{phase}_basis")
        if (
            status not in PHASE_STATUSES
            or isinstance(frame, bool)
            or not isinstance(frame, int)
            or not 0 <= frame < frame_count
            or not isinstance(basis, str)
            or not basis.strip()
        ):
            raise ValueError(f"Invalid {phase} evidence for {sample_id}")
        statuses[phase] = status
        evidence.append(frame)
    return statuses, evidence


def _verify_public_queue_record(
    row: dict[str, Any], *, public_root: Path, queue_path: Path
) -> None:
    sample_id = row.get("sample_id")
    exact = {
        "native_duration_preserved": True,
        "fixed_duration_window_used": False,
        "audio_available": False,
        "label_metadata_exposed": False,
    }
    failed = sorted(key for key, value in exact.items() if row.get(key) != value)
    video = _declared_path(queue_path, row.get("video_path"), field="video_path")
    try:
        video.relative_to(public_root)
    except ValueError as error:
        raise ValueError(f"Public video escapes its bundle: {sample_id}") from error
    if failed or not video.is_file() or sha256_file(video) != row.get("video_sha256"):
        raise ValueError(f"Public video/queue contract mismatch: {sample_id}: {failed}")
    frames = row.get("frame_count")
    fps = row.get("fps")
    if (
        isinstance(frames, bool)
        or not isinstance(frames, int)
        or frames <= 0
        or isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or float(fps) <= 0
    ):
        raise ValueError(f"Invalid public queue timeline: {sample_id}")


def _verify_render_record(
    record: dict[str, Any],
    *,
    mapping: dict[str, Any],
    previous: dict[str, Any],
    candidate_catalog_sha256: str,
    previous_plan_summary_sha256: str,
    previous_expansion_requests_sha256: str,
) -> None:
    sample_id = mapping.get("sample_id")
    provenance = record.get("expansion_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"Render record lacks expansion provenance: {sample_id}")
    exact_provenance = {
        "base_candidate_catalog_sha256": candidate_catalog_sha256,
        "expansion_plan_summary_sha256": previous_plan_summary_sha256,
        "expansion_requests_sha256": previous_expansion_requests_sha256,
        "plan_record_sha256": previous.get("plan_record_sha256"),
        "comparison_record_sha256": previous.get("comparison_record_sha256"),
        "base_task_id": previous.get("base_task_id"),
        "base_expression_turn_record_sha256": previous.get(
            "base_expression_turn_record_sha256"
        ),
        "sample_id": previous.get("sample_id"),
        "reviewed_context_level": previous.get("reviewed_context_level"),
        "requested_context_level": previous.get("requested_context_level"),
        "selection_policy": SELECTION_POLICY,
        "expansion_unit": EXPANSION_UNIT,
        "elapsed_duration_used_as_gate": False,
        "accepted_for_training": False,
    }
    failed = sorted(
        key for key, value in exact_provenance.items() if provenance.get(key) != value
    )
    expected_task = (
        f"{previous['base_task_id']}__ctxL{previous['requested_context_level']:02d}"
    )
    segment = record.get("training_segment") or {}
    expected_interval = _interval(
        previous.get("requested_interval"), context=f"{sample_id}.previous_requested"
    )
    segment_interval = _interval(segment, context=f"{sample_id}.render_segment")
    if (
        failed
        or record.get("task_id") != expected_task
        or segment_interval != expected_interval
        or segment.get("fixed_window_sec") is not None
        or segment.get("cropped") is not False
        or record.get("processing_scope")
        != "physical_retarget_and_silent_render_only_pending_fresh_blind_arc_action_and_affect_review"
        or record.get("license_training_admission") is not False
        or record.get("render_pass_grants_training_admission") is not False
    ):
        raise ValueError(f"Render lineage mismatch for {sample_id}: {failed}")
    video = Path(str(record.get("video_path") or "")).resolve()
    trajectory = Path(str(record.get("trajectory_path") or "")).resolve()
    if (
        not video.is_file()
        or sha256_file(video) != record.get("video_sha256")
        or not trajectory.is_file()
        or sha256_file(trajectory) != record.get("trajectory_sha256")
        or mapping.get("video_sha256") != record.get("video_sha256")
        or mapping.get("trajectory_sha256") != record.get("trajectory_sha256")
        or mapping.get("source_render_record_sha256") != value_sha256(record)
        or mapping.get("frame_count") != record.get("trajectory_frames")
    ):
        raise ValueError(f"Render video/trajectory lineage mismatch for {sample_id}")
    retarget = record.get("retarget_segment") or {}
    if (
        retarget.get("source_start_frame") != expected_interval["start_frame"]
        or retarget.get("source_end_frame_exclusive")
        != expected_interval["end_frame_exclusive"]
        or retarget.get("source_frame_count") != expected_interval["frame_count"]
        or retarget.get("output_frame_count") != record.get("trajectory_frames")
        or retarget.get("fixed_target_duration_sec") is not None
        or retarget.get("cropped") is not False
    ):
        raise ValueError(f"Render retarget interval mismatch for {sample_id}")


def build_plan(
    *,
    public_summary: Path,
    arc_review_submission: Path,
    arc_review_summary: Path,
    expansion_hidden_mapping: Path,
    expansion_hidden_summary: Path,
    base_hidden_mapping: Path,
    previous_plan_summary: Path,
    previous_expansion_requests: Path,
    candidate_catalog: Path,
    catalog_summary: Path,
    output_root: Path,
) -> dict[str, Any]:
    supplied = {
        name: path.resolve()
        for name, path in {
            "public_summary": public_summary,
            "arc_review_submission": arc_review_submission,
            "arc_review_summary": arc_review_summary,
            "expansion_hidden_mapping": expansion_hidden_mapping,
            "expansion_hidden_summary": expansion_hidden_summary,
            "base_hidden_mapping": base_hidden_mapping,
            "previous_plan_summary": previous_plan_summary,
            "previous_expansion_requests": previous_expansion_requests,
            "candidate_catalog": candidate_catalog,
            "catalog_summary": catalog_summary,
        }.items()
    }
    for path in supplied.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    input_sha256 = {f"{name}_sha256": sha256_file(path) for name, path in supplied.items()}
    # Preserve the original runner contract: round-N requests are keyed by the
    # canonical base sample IDs and therefore use the original base mapping.
    input_sha256["hidden_mapping_sha256"] = input_sha256[
        "base_hidden_mapping_sha256"
    ]

    catalog, _catalog_meta = _load_catalog(
        supplied["candidate_catalog"], supplied["catalog_summary"]
    )
    previous_plan = read_json(supplied["previous_plan_summary"])
    if (
        previous_plan.get("artifact_kind") != PLAN_KIND
        or previous_plan.get("selection_policy") != SELECTION_POLICY
        or previous_plan.get("fixed_minimum_maximum_or_target_duration_used") is not False
        or previous_plan.get("accepted_for_training_count") != 0
        or (previous_plan.get("inputs") or {}).get("hidden_mapping_sha256")
        != input_sha256["base_hidden_mapping_sha256"]
        or (previous_plan.get("inputs") or {}).get("candidate_catalog_sha256")
        != input_sha256["candidate_catalog_sha256"]
    ):
        raise ValueError("Previous expansion plan is unbound or not fail-closed")
    previous_declared = (previous_plan.get("outputs") or {}).get("expansion_requests")
    if not isinstance(previous_declared, dict):
        raise ValueError("Previous plan does not declare expansion requests")
    if (
        _declared_path(
            supplied["previous_plan_summary"],
            previous_declared.get("path"),
            field="outputs.expansion_requests.path",
        )
        != supplied["previous_expansion_requests"]
        or previous_declared.get("sha256")
        != input_sha256["previous_expansion_requests_sha256"]
    ):
        raise ValueError("Previous plan request binding mismatch")
    previous_by_sample = index_rows(
        read_bound_jsonl(supplied["previous_expansion_requests"]),
        "sample_id",
        context="previous expansion requests",
    )
    if previous_declared.get("records") != len(previous_by_sample):
        raise ValueError("Previous expansion request count mismatch")
    previous_by_task: dict[str, dict[str, Any]] = {}
    for row in previous_by_sample.values():
        _verify_previous_request(row)
        task_id = row.get("base_task_id")
        if not isinstance(task_id, str) or task_id in previous_by_task:
            raise ValueError("Previous expansion requests contain duplicate base_task_id")
        previous_by_task[task_id] = row

    base_by_sample_bound = index_bound(
        read_bound_jsonl(supplied["base_hidden_mapping"]),
        "sample_id",
        context="base hidden mapping",
    )
    base_by_task: dict[str, tuple[dict[str, Any], str]] = {}
    for row, line_sha in base_by_sample_bound.values():
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id or task_id in base_by_task:
            raise ValueError("Base hidden mapping contains invalid or duplicate task_id")
        base_by_task[task_id] = (row, line_sha)

    public = read_json(supplied["public_summary"])
    if (
        public.get("artifact_kind") != PUBLIC_KIND
        or public.get("duration_policy") != PUBLIC_DURATION_POLICY
        or public.get("all_samples_native_variable_length") is not True
        or public.get("fixed_duration_window_used") is not False
        or public.get("accepted_for_training") is not False
    ):
        raise ValueError("Expansion public summary violates variable-length policy")
    queue_path = _require_declared_file(
        supplied["public_summary"],
        public,
        path_field="arc_action_queue",
        sha_field="arc_action_queue_sha256",
    )
    if queue_path.parent != supplied["public_summary"].parent:
        raise ValueError("Arc/action queue is outside the public bundle")
    queue_bound = index_bound(
        read_bound_jsonl(queue_path), "sample_id", context="public arc/action queue"
    )
    if public.get("records") != len(queue_bound):
        raise ValueError("Public expansion record count mismatch")

    reviews = index_rows(
        read_bound_jsonl(supplied["arc_review_submission"]),
        "sample_id",
        context="arc/action review submission",
    )
    expansion_mapping_bound = index_bound(
        read_bound_jsonl(supplied["expansion_hidden_mapping"]),
        "sample_id",
        context="expansion hidden mapping",
    )
    if set(queue_bound) != set(reviews) or set(queue_bound) != set(expansion_mapping_bound):
        raise ValueError("Public queue, review, and expansion mapping sample sets differ")
    _verify_review_summary(
        supplied["arc_review_summary"],
        read_json(supplied["arc_review_summary"]),
        review_submission=supplied["arc_review_submission"],
        public_summary=supplied["public_summary"],
        queue_path=queue_path,
        reviews=reviews,
    )
    input_sha256["arc_action_queue_sha256"] = sha256_file(queue_path)

    hidden_summary = read_json(supplied["expansion_hidden_summary"])
    if (
        hidden_summary.get("artifact_kind") != HIDDEN_KIND
        or hidden_summary.get("accepted_for_training") is not False
        or hidden_summary.get("public_distribution_forbidden") is not True
        or hidden_summary.get("records") != len(expansion_mapping_bound)
    ):
        raise ValueError("Expansion hidden summary violates fail-closed policy")
    _require_declared_file(
        supplied["expansion_hidden_summary"],
        hidden_summary,
        path_field="mapping",
        sha_field="mapping_sha256",
        expected=supplied["expansion_hidden_mapping"],
    )
    _require_declared_file(
        supplied["expansion_hidden_summary"],
        hidden_summary,
        path_field="expansion_plan_summary",
        sha_field="expansion_plan_summary_sha256",
        expected=supplied["previous_plan_summary"],
    )
    _require_declared_file(
        supplied["expansion_hidden_summary"],
        hidden_summary,
        path_field="expansion_requests",
        sha_field="expansion_requests_sha256",
        expected=supplied["previous_expansion_requests"],
    )
    pipeline_path = _require_declared_file(
        supplied["expansion_hidden_summary"],
        hidden_summary,
        path_field="pipeline_summary",
        sha_field="pipeline_summary_sha256",
    )
    passed_path = _require_declared_file(
        supplied["expansion_hidden_summary"],
        hidden_summary,
        path_field="render_passed_manifest",
        sha_field="render_passed_manifest_sha256",
    )
    input_sha256["pipeline_summary_sha256"] = sha256_file(pipeline_path)
    input_sha256["render_passed_manifest_sha256"] = sha256_file(passed_path)

    pipeline = read_json(pipeline_path)
    if (
        pipeline.get("artifact_kind") != PIPELINE_KIND
        or pipeline.get("stage") not in {"render", "all"}
        or pipeline.get("fixed_six_second_windows_used") is not False
        or pipeline.get("elapsed_seconds_used_for_selection") is not False
        or pipeline.get("accepted_for_training") is not False
    ):
        raise ValueError("Expansion physical pipeline is incomplete or duration-gated")
    audit = pipeline.get("input_audit")
    if not isinstance(audit, dict):
        raise ValueError("Expansion pipeline lacks input audit")
    audit_payload = dict(audit)
    audit_expected = audit_payload.pop("sha256", None)
    if audit_expected != value_sha256(audit_payload):
        raise ValueError("Expansion pipeline input audit SHA mismatch")
    audit_inputs = audit.get("inputs") or {}
    if (
        _file_binding_path(
            audit_inputs.get("expansion_plan_summary"),
            pipeline_path,
            context="pipeline expansion plan",
        )
        != supplied["previous_plan_summary"]
        or _file_binding_path(
            audit_inputs.get("expansion_requests"),
            pipeline_path,
            context="pipeline expansion requests",
        )
        != supplied["previous_expansion_requests"]
        or _file_binding_path(
            audit_inputs.get("candidate_catalog"),
            pipeline_path,
            context="pipeline candidate catalog",
        )
        != supplied["candidate_catalog"]
    ):
        raise ValueError("Expansion pipeline inputs differ from continuation inputs")
    render_summary = pipeline.get("render_summary")
    if not isinstance(render_summary, dict):
        raise ValueError("Expansion pipeline lacks render summary")
    render_passed = _declared_path(
        pipeline_path, render_summary.get("passed_manifest"), field="passed_manifest"
    )
    if (
        render_passed != passed_path
        or render_summary.get("passed_manifest_sha256") != sha256_file(passed_path)
        or (render_summary.get("counts") or {}).get("failed") != 0
        or (render_summary.get("counts") or {}).get("passed") != len(queue_bound)
    ):
        raise ValueError("Expansion render summary does not bind a complete passed manifest")
    passed_by_task = index_rows(
        read_bound_jsonl(passed_path), "task_id", context="render passed manifest"
    )
    if len(passed_by_task) != len(queue_bound):
        raise ValueError("Render passed manifest count differs from public queue")

    expansion_requests: list[dict[str, Any]] = []
    complete_current_context: list[dict[str, Any]] = []
    context_exhausted: list[dict[str, Any]] = []
    canonical_base_ids: set[str] = set()
    declared_base_id_mismatches = 0
    for anonymous_sample_id in sorted(queue_bound):
        queued, queue_line_sha = queue_bound[anonymous_sample_id]
        mapping, mapping_line_sha = expansion_mapping_bound[anonymous_sample_id]
        review = reviews[anonymous_sample_id]
        _verify_public_queue_record(
            queued, public_root=supplied["public_summary"].parent, queue_path=queue_path
        )
        statuses, evidence = _verify_review(
            review,
            queue=queued,
            queue_sha256=input_sha256["arc_action_queue_sha256"],
        )
        base_task_id = mapping.get("base_task_id")
        previous = previous_by_task.get(base_task_id)
        candidate = catalog.get(base_task_id)
        base_bound = base_by_task.get(base_task_id)
        if previous is None or candidate is None or base_bound is None:
            raise ValueError(f"Anonymous mapping references unknown base task: {anonymous_sample_id}")
        base_mapping, base_mapping_line_sha = base_bound
        canonical_base_id = previous["sample_id"]
        if base_mapping.get("sample_id") != canonical_base_id:
            raise ValueError(f"Previous request and base hidden mapping disagree: {base_task_id}")
        if canonical_base_id in canonical_base_ids:
            raise ValueError(f"Duplicate canonical base sample: {canonical_base_id}")
        canonical_base_ids.add(canonical_base_id)
        if mapping.get("base_sample_id") != canonical_base_id:
            declared_base_id_mismatches += 1

        _verify_candidate_hash(candidate)
        levels = _context_levels(candidate)
        current_level = mapping.get("displayed_context_level")
        if isinstance(current_level, bool) or not isinstance(current_level, int):
            raise ValueError(f"Invalid displayed context level: {anonymous_sample_id}")
        current_interval = levels.get(current_level)
        if current_interval is None:
            raise ValueError(f"Displayed context is absent from catalog: {anonymous_sample_id}")
        previous_reviewed = _interval(
            previous.get("reviewed_interval"), context=f"{canonical_base_id}.prior_reviewed"
        )
        previous_requested = _interval(
            previous.get("requested_interval"), context=f"{canonical_base_id}.prior_requested"
        )
        mapping_reviewed = _interval(
            mapping.get("reviewed_interval"), context=f"{anonymous_sample_id}.reviewed"
        )
        mapping_displayed = _interval(
            mapping.get("displayed_interval"), context=f"{anonymous_sample_id}.displayed"
        )
        expected_mapping = {
            "sample_id": anonymous_sample_id,
            "base_task_id": base_task_id,
            "source_clip_id": previous.get("source_clip_id"),
            "fixed_split_assignment": previous.get("fixed_split_assignment"),
            "reviewed_context_level": previous.get("reviewed_context_level"),
            "displayed_context_level": previous.get("requested_context_level"),
            "plan_record_sha256": previous.get("plan_record_sha256"),
            "comparison_record_sha256": previous.get("comparison_record_sha256"),
            "native_duration_preserved": True,
            "fixed_duration_window_used": False,
            "official_action_text_or_affect_exposed": False,
            "accepted_for_training": False,
        }
        failed = sorted(
            key for key, value in expected_mapping.items() if mapping.get(key) != value
        )
        if (
            failed
            or mapping_reviewed != previous_reviewed
            or mapping_displayed != previous_requested
            or mapping_displayed != current_interval
            or queued.get("context_level") != current_level
            or mapping.get("video_sha256") != queued.get("video_sha256")
            or mapping.get("frame_count") != queued.get("frame_count")
            or base_mapping.get("task_id") != base_task_id
            or base_mapping.get("source_clip_id") != previous.get("source_clip_id")
            or base_mapping.get("fixed_split_assignment")
            != previous.get("fixed_split_assignment")
            or base_mapping.get("expression_turn_record_sha256")
            != previous.get("base_expression_turn_record_sha256")
            or candidate.get("source_clip_id") != previous.get("source_clip_id")
            or candidate.get("fixed_split_assignment")
            != previous.get("fixed_split_assignment")
            or candidate.get("expression_turn_record_sha256")
            != previous.get("base_expression_turn_record_sha256")
        ):
            raise ValueError(f"Anonymous mapping/candidate lineage mismatch: {anonymous_sample_id}: {failed}")
        expected_task = f"{base_task_id}__ctxL{current_level:02d}"
        if mapping.get("task_id") != expected_task or mapping.get("derived_task_id") != expected_task:
            raise ValueError(f"Anonymous mapping derived task mismatch: {anonymous_sample_id}")
        render_record = passed_by_task.get(expected_task)
        if render_record is None:
            raise ValueError(f"Anonymous mapping has no render record: {anonymous_sample_id}")
        _verify_render_record(
            render_record,
            mapping=mapping,
            previous=previous,
            candidate_catalog_sha256=input_sha256["candidate_catalog_sha256"],
            previous_plan_summary_sha256=input_sha256["previous_plan_summary_sha256"],
            previous_expansion_requests_sha256=input_sha256[
                "previous_expansion_requests_sha256"
            ],
        )

        common = {
            "schema_version": SCHEMA_VERSION,
            "sample_id": canonical_base_id,
            "reviewed_anonymous_sample_id": anonymous_sample_id,
            "base_task_id": base_task_id,
            "source_clip_id": previous["source_clip_id"],
            "fixed_split_assignment": previous["fixed_split_assignment"],
            "base_expression_turn_record_sha256": previous[
                "base_expression_turn_record_sha256"
            ],
            "comparison_record_sha256": previous["comparison_record_sha256"],
            "review_qualification_status": "natural_context_expansion_required",
            "reviewed_context_level": current_level,
            "reviewed_interval": current_interval,
            "review_phase_statuses": statuses,
            "review_evidence_frames": evidence,
            "reviewed_video_sha256": mapping["video_sha256"],
            "reviewed_trajectory_sha256": mapping["trajectory_sha256"],
            "reviewed_output_frame_count": mapping["frame_count"],
            "continuation_lineage": {
                "role": "round_n_natural_boundary_continuation",
                "previous_reviewed_context_level": previous["reviewed_context_level"],
                "previous_requested_context_level": previous["requested_context_level"],
                "previous_plan_record_sha256": previous["plan_record_sha256"],
                "previous_plan_summary_sha256": input_sha256[
                    "previous_plan_summary_sha256"
                ],
                "previous_expansion_requests_sha256": input_sha256[
                    "previous_expansion_requests_sha256"
                ],
                "source_public_summary_sha256": input_sha256["public_summary_sha256"],
                "source_arc_action_queue_sha256": input_sha256[
                    "arc_action_queue_sha256"
                ],
                "review_submission_sha256": input_sha256[
                    "arc_review_submission_sha256"
                ],
                "review_summary_sha256": input_sha256["arc_review_summary_sha256"],
                "review_record_sha256": value_sha256(review),
                "review_queue_record_sha256": queue_line_sha,
                "expansion_hidden_mapping_sha256": input_sha256[
                    "expansion_hidden_mapping_sha256"
                ],
                "expansion_hidden_mapping_record_sha256": mapping_line_sha,
                "expansion_hidden_summary_sha256": input_sha256[
                    "expansion_hidden_summary_sha256"
                ],
                "base_hidden_mapping_sha256": input_sha256[
                    "base_hidden_mapping_sha256"
                ],
                "base_hidden_mapping_record_sha256": base_mapping_line_sha,
                "catalog_summary_sha256": input_sha256["catalog_summary_sha256"],
                "candidate_catalog_sha256": input_sha256[
                    "candidate_catalog_sha256"
                ],
                "pipeline_summary_sha256": input_sha256["pipeline_summary_sha256"],
                "render_passed_manifest_sha256": input_sha256[
                    "render_passed_manifest_sha256"
                ],
                "source_render_record_sha256": mapping[
                    "source_render_record_sha256"
                ],
                "input_mapping_declared_base_sample_id": mapping.get("base_sample_id"),
                "canonical_base_sample_id_derived_from_unique_base_task": canonical_base_id,
            },
            "elapsed_duration_used_as_gate": False,
            "semantic_supervision_mask": False,
            "emotion_supervision_mask": False,
            "accepted_for_training": False,
        }
        all_complete = all(status == "complete" for status in statuses.values())
        if all_complete:
            record = {
                **common,
                "artifact_kind": COMPLETE_KIND,
                "review_qualification_status": "complete_arc_action_candidate",
                "next_action": "hold_for_independent_affect_and_remaining_admission_gates",
            }
            record["plan_record_sha256"] = value_sha256(record)
            complete_current_context.append(record)
            continue
        if not any(status in INCOMPLETE_PHASE_STATUSES for status in statuses.values()):
            raise ValueError(f"Incomplete arc has no real incomplete/ambiguous boundary: {anonymous_sample_id}")
        next_level = current_level + 1
        requested = levels.get(next_level)
        if requested is None:
            record = {
                **common,
                "artifact_kind": EXHAUSTED_KIND,
                "context_exhausted_at_level": current_level,
                "next_action": "reject_or_manual_source_level_adjudication_no_further_predeclared_level",
            }
            record["plan_record_sha256"] = value_sha256(record)
            context_exhausted.append(record)
            continue
        if not _strictly_expands(current_interval, requested):
            raise ValueError(f"Next catalog level does not strictly expand: {anonymous_sample_id}")
        record = {
            **common,
            "artifact_kind": REQUEST_KIND,
            "requested_context_level": next_level,
            "requested_interval": requested,
            "strictly_contains_reviewed_interval": True,
            "expansion_unit": EXPANSION_UNIT,
            "next_action": "retarget_render_and_repeat_independent_blind_arc_action_review",
        }
        record["plan_record_sha256"] = value_sha256(record)
        expansion_requests.append(record)

    canonical_from_render = {
        str((row.get("expansion_provenance") or {}).get("sample_id"))
        for row in passed_by_task.values()
    }
    if canonical_from_render != canonical_base_ids:
        raise ValueError("Render provenance and canonical base sample sets differ")
    declared_exclusions = hidden_summary.get("physical_qc_excluded_base_sample_ids")
    expected_exclusions = sorted(set(previous_by_sample).difference(canonical_base_ids))
    if (
        declared_exclusions != expected_exclusions
        or hidden_summary.get("physical_qc_excluded_records") != len(expected_exclusions)
        or hidden_summary.get("requested_records") != len(previous_by_sample)
    ):
        raise ValueError("Expansion hidden summary physical-QC exclusion set mismatch")

    output_root = output_root.resolve()
    outputs: dict[str, dict[str, Any]] = {}
    for name, rows in (
        ("expansion_requests", expansion_requests),
        ("complete_current_context", complete_current_context),
        ("context_exhausted", context_exhausted),
    ):
        rows.sort(key=lambda row: row["sample_id"])
        path = output_root / f"{name}.jsonl"
        atomic_jsonl(path, rows)
        outputs[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "records": len(rows),
        }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": PLAN_KIND,
        "plan_role": "round_n_natural_boundary_continuation",
        "inputs": input_sha256,
        "input_record_counts": {
            "previous_expansion_requests": len(previous_by_sample),
            "physically_passed_and_reviewed_samples": len(queue_bound),
            "physical_qc_excluded_samples": len(expected_exclusions),
        },
        "review_phase_status_distributions": {
            phase: dict(
                sorted(Counter(row[f"{phase}_status"] for row in reviews.values()).items())
            )
            for phase in PHASES
        },
        "reviewed_context_level_distribution": dict(
            sorted(Counter(str(row["context_level"]) for row in reviews.values()).items())
        ),
        "requested_context_level_distribution": dict(
            sorted(
                Counter(
                    str(row["requested_context_level"])
                    for row in expansion_requests
                ).items()
            )
        ),
        "selection_policy": SELECTION_POLICY,
        "continuation_decision_basis": "observable_action_and_complete_onset_apex_offset_at_predeclared_natural_boundaries",
        "one_level_only": True,
        "same_source_only": True,
        "neighbor_crossing_allowed": False,
        "fixed_minimum_maximum_or_target_duration_used": False,
        "elapsed_duration_used_as_gate": False,
        "legacy_declared_base_sample_id_mismatch_count": declared_base_id_mismatches,
        "canonical_base_sample_id_recovered_from_unique_base_task_lineage": True,
        "outputs": outputs,
        "accepted_for_training_count": 0,
    }
    atomic_json(output_root / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_plan(
        public_summary=args.public_summary,
        arc_review_submission=args.arc_review_submission,
        arc_review_summary=args.arc_review_summary,
        expansion_hidden_mapping=args.expansion_hidden_mapping,
        expansion_hidden_summary=args.expansion_hidden_summary,
        base_hidden_mapping=args.base_hidden_mapping,
        previous_plan_summary=args.previous_plan_summary,
        previous_expansion_requests=args.previous_expansion_requests,
        candidate_catalog=args.candidate_catalog,
        catalog_summary=args.catalog_summary,
        output_root=args.output_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
