#!/usr/bin/env python3
"""Retarget and render one-level BEAT2 v8 natural-context expansions.

The input is the private expansion plan produced after independent arc/action
review.  This runner never chooses a window from elapsed seconds: every task
must advance from the reviewed context to exactly the next natural-boundary
level already declared in the immutable v8 catalog.  Its outputs are physical
and silent-video evidence only; semantic, affect, license, and training
admission remain closed.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from . import batch_retarget_beat2_expression_turns_v8 as base
    from . import batch_retarget_beat2_semantic_events_v2 as grouped
    from . import batch_retarget_beat2_v2 as ordinary
    from .retarget_beat2_grouped_v2 import GroupedRuntimeConfig
except ImportError:  # pragma: no cover - direct invocation by path
    import batch_retarget_beat2_expression_turns_v8 as base
    import batch_retarget_beat2_semantic_events_v2 as grouped
    import batch_retarget_beat2_v2 as ordinary
    from retarget_beat2_grouped_v2 import GroupedRuntimeConfig

from tools.human_motion_review import render_beat2_annotation_review as renderer
from tools.human_motion_review import build_expression_turn_video_queue_v8 as video_queue
from tools.human_motion_review.build_expression_turn_expansion_plan_v8 import (
    FORBIDDEN_OUTPUT_KEYS,
)
from tools.human_motion_review.expression_turn_contract import (
    CONTEXT_POLICY,
    SELECTION_LINEAGE_FIELDS,
    validate_expression_turn_candidate,
)
from upper_body_skeleton.retarget_v2_18d import JOINT_ORDER_18D


SCHEMA_VERSION = "1.0.0"
REQUEST_ARTIFACT_KIND = (
    "expression_turn_v8_one_level_natural_context_expansion_request"
)
PLAN_ARTIFACT_KIND = "expression_turn_v8_one_level_natural_context_expansion_plan"
EXPANSION_PROVENANCE_KIND = "expression_turn_v8_natural_context_expansion_lineage_v1"
PIPELINE_ARTIFACT_KIND = "beat2_expression_turn_v8_expansion_physical_pipeline_v1"
QUEUE_ARTIFACT_KIND = "expression_turn_v8_expansion_physical_review_queue_v1"
PROCESSING_SCOPE = (
    "physical_retarget_and_silent_render_only_pending_fresh_blind_arc_action_"
    "and_affect_review"
)
SELECTION_POLICY = "exactly_one_next_predeclared_natural_context_level"
EXPANSION_UNIT = "one_predeclared_adjacent_natural_boundary_level"
LICENSE_STATUS = "not_evaluated_physical_review_only"
SAFETY_ORDERED_METRIC_FIELDS = (
    "raw_max_velocity_rad_s_by_joint",
    "pre_retime_max_velocity_rad_s_by_joint",
    "post_retime_max_velocity_rad_s_by_joint",
    "post_retime_max_acceleration_rad_s2_by_joint",
)

DEFAULT_CATALOG_ROOT = Path(
    "/home/gez/nas/cloud/gez/human_motion/catalog/beat2_expression_turn_v8"
)
DEFAULT_CANDIDATE_CATALOG = (
    DEFAULT_CATALOG_ROOT / "beat2_expression_turn_v8.candidates.jsonl"
)
DEFAULT_CATALOG_SUMMARY = (
    DEFAULT_CATALOG_ROOT / "beat2_expression_turn_v8.summary.json"
)
DEFAULT_BEAT2_ROOT = Path("/home/gez/nas/cloud/gez/human_motion/raw/BEAT2")

EXPANSION_AUDIT_FIELDS = (
    "expansion_provenance",
    "processing_scope",
    "semantic_supervision_mask",
    "license_training_admission",
    "license_training_admission_status",
    "retarget_result_json",
    "retarget_result_json_sha256",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expansion-requests", type=Path, required=True)
    parser.add_argument("--expansion-plan-summary", type=Path, required=True)
    parser.add_argument("--hidden-mapping", type=Path, required=True)
    parser.add_argument(
        "--candidate-catalog", type=Path, default=DEFAULT_CANDIDATE_CATALOG
    )
    parser.add_argument(
        "--catalog-summary", type=Path, default=DEFAULT_CATALOG_SUMMARY
    )
    parser.add_argument("--beat2-root", type=Path, default=DEFAULT_BEAT2_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smplx-model", type=Path, default=ordinary.DEFAULT_MODEL)
    parser.add_argument("--gmr-root", type=Path, default=ordinary.DEFAULT_GMR_ROOT)
    parser.add_argument("--urdf", type=Path, default=ordinary.DEFAULT_URDF)
    parser.add_argument("--config", type=Path, default=ordinary.DEFAULT_CONFIG)
    parser.add_argument(
        "--renderer-python", type=Path, default=renderer.DEFAULT_RENDERER_PYTHON
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--render-workers", type=int, default=2)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--limit-sources", type=int)
    parser.add_argument("--limit-turns", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--stage", choices=("all", "retarget", "render"), default="all"
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Unreadable JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _index_unique(
    rows: list[dict[str, Any]], key: str, *, context: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{context} contains invalid {key}")
        if value in indexed:
            raise ValueError(f"{context} contains duplicate {key}: {value}")
        indexed[value] = row
    return indexed


def _record_sha256(record: dict[str, Any], field: str) -> str:
    expected = record.get(field)
    payload = dict(record)
    payload.pop(field, None)
    actual = ordinary.json_sha256(payload)
    if expected != actual:
        raise ValueError(f"Record SHA mismatch for {record.get('sample_id')}: {field}")
    return actual


def _require_interval(value: Any, *, context: str) -> tuple[int, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} is not an interval")
    start = value.get("start_frame")
    end = value.get("end_frame_exclusive")
    count = value.get("frame_count", end - start if isinstance(start, int) and isinstance(end, int) else None)
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or end <= start
        or count != end - start
    ):
        raise ValueError(f"{context} is invalid")
    return start, end


def _walk_keys(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _validate_request(record: dict[str, Any]) -> None:
    sample_id = record.get("sample_id")
    _record_sha256(record, "plan_record_sha256")
    exact = {
        "artifact_kind": REQUEST_ARTIFACT_KIND,
        "strictly_contains_reviewed_interval": True,
        "expansion_unit": EXPANSION_UNIT,
        "elapsed_duration_used_as_gate": False,
        "semantic_supervision_mask": False,
        "emotion_supervision_mask": False,
        "accepted_for_training": False,
    }
    failed = sorted(key for key, value in exact.items() if record.get(key) != value)
    if failed:
        raise ValueError(f"{sample_id}: fail-closed expansion request mismatch: {failed}")
    reviewed_level = record.get("reviewed_context_level")
    requested_level = record.get("requested_context_level")
    if (
        isinstance(reviewed_level, bool)
        or not isinstance(reviewed_level, int)
        or requested_level != reviewed_level + 1
    ):
        raise ValueError(f"{sample_id}: request does not advance exactly one level")
    reviewed = _require_interval(
        record.get("reviewed_interval"), context=f"{sample_id}.reviewed_interval"
    )
    requested = _require_interval(
        record.get("requested_interval"), context=f"{sample_id}.requested_interval"
    )
    if not (
        requested[0] <= reviewed[0]
        and requested[1] >= reviewed[1]
        and requested != reviewed
    ):
        raise ValueError(f"{sample_id}: requested interval is not a strict expansion")
    hidden = FORBIDDEN_OUTPUT_KEYS.intersection(_walk_keys(record))
    if hidden:
        raise ValueError(f"{sample_id}: expansion request leaks hidden labels: {sorted(hidden)}")


def _timeline(start: int, end: int, fps: float, *, relative: bool) -> dict[str, Any]:
    frame_count = end - start
    first = 0 if relative else start
    last = frame_count if relative else end
    return {
        "frame_axis": "turn_relative" if relative else "source_absolute",
        "start_frame": first,
        "end_frame_exclusive": last,
        "frame_count": frame_count,
        "frame_coverage_start_sec": round(first / fps, 6),
        "frame_coverage_end_sec": round(last / fps, 6),
        "frame_coverage_sec": round(frame_count / fps, 6),
        "sample_start_sec": round(first / fps, 6),
        "sample_end_sec": round((last - 1) / fps, 6),
        "sample_span_sec": round(max(0, frame_count - 1) / fps, 6),
    }


def _duration_band(frame_count: int, fps: float) -> str:
    if frame_count < 3 * fps:
        return "short_under_3s"
    if frame_count <= 6 * fps:
        return "medium_3_to_6s"
    return "long_over_6s"


def derive_expanded_candidate(
    candidate: dict[str, Any],
    request: dict[str, Any],
    *,
    plan_summary_sha256: str,
    expansion_requests_sha256: str,
    hidden_mapping_sha256: str,
    candidate_catalog_sha256: str,
) -> dict[str, Any]:
    """Create a valid v8 candidate at the one requested natural context level."""

    _validate_request(request)
    base_task_id = str(candidate.get("task_id") or "")
    if request.get("base_task_id") != base_task_id:
        raise ValueError("expansion request does not bind the candidate task")
    if request.get("base_expression_turn_record_sha256") != candidate.get(
        "expression_turn_record_sha256"
    ):
        raise ValueError(f"{base_task_id}: base candidate hash mismatch")

    context = copy.deepcopy(candidate.get("context_plan"))
    if not isinstance(context, dict) or context.get("policy") != CONTEXT_POLICY:
        raise ValueError(f"{base_task_id}: invalid catalog context plan")
    reviewed_level = int(request["reviewed_context_level"])
    requested_level = int(request["requested_context_level"])
    catalog_selected_level = context.get("selected_level")
    if catalog_selected_level != reviewed_level:
        # Round-N plans continue from a previously rendered catalog level while
        # still deriving from the immutable level-0 candidate.  The request's
        # record hash covers this lineage and the plan summary binds all prior
        # artifacts; arbitrary level jumps remain forbidden below.
        continuation = request.get("continuation_lineage")
        if (
            catalog_selected_level != 0
            or reviewed_level <= 0
            or not isinstance(continuation, dict)
            or continuation.get("role")
            != "round_n_natural_boundary_continuation"
            or continuation.get("previous_reviewed_context_level")
            != reviewed_level - 1
            or continuation.get("previous_requested_context_level")
            != reviewed_level
            or continuation.get("canonical_base_sample_id_derived_from_unique_base_task")
            != request.get("sample_id")
            or any(
                not ordinary._is_sha256(continuation.get(field))
                for field in (
                    "previous_plan_record_sha256",
                    "previous_plan_summary_sha256",
                    "previous_expansion_requests_sha256",
                )
            )
        ):
            raise ValueError(
                f"{base_task_id}: reviewed level differs from catalog selection "
                "without valid continuation lineage"
            )
    levels = context.get("levels")
    if not isinstance(levels, list) or requested_level >= len(levels):
        raise ValueError(f"{base_task_id}: requested natural context is unavailable")
    current = levels[reviewed_level]
    requested = levels[requested_level]
    if requested.get("parent_level") != reviewed_level:
        raise ValueError(f"{base_task_id}: requested level is not the declared child")
    current_interval = _require_interval(current, context=f"{base_task_id}.current")
    requested_interval = _require_interval(
        requested, context=f"{base_task_id}.requested"
    )
    if current_interval != _require_interval(
        request["reviewed_interval"], context=f"{base_task_id}.reviewed_request"
    ) or requested_interval != _require_interval(
        request["requested_interval"], context=f"{base_task_id}.requested_request"
    ):
        raise ValueError(f"{base_task_id}: request intervals differ from catalog levels")

    expanded = copy.deepcopy(candidate)
    for field in SELECTION_LINEAGE_FIELDS:
        expanded.pop(field, None)
    start, end = requested_interval
    frame_count = end - start
    fps = float(expanded["fps"])
    derived_task_id = f"{base_task_id}__ctxL{requested_level:02d}"
    expanded["clip_id"] = derived_task_id
    expanded["task_id"] = derived_task_id
    context["selected_level"] = requested_level
    expanded["context_plan"] = context
    expanded["training_segment"] = {
        "representation": base.INPUT_REPRESENTATION,
        "start_frame": start,
        "end_frame_exclusive": end,
        "frame_count": frame_count,
        "fixed_window_sec": None,
        "cropped": False,
        "duration_policy": base.NATURAL_DURATION_POLICY,
    }
    expanded["time_axes"] = {
        "source": _timeline(start, end, fps, relative=False),
        "turn": _timeline(start, end, fps, relative=True),
    }

    turn = copy.deepcopy(expanded["expression_turn"])
    for span in turn["included_event_spans"]:
        source_axis = span["source_time_axis"]
        span["turn_time_axis"] = {
            "start_sec": round(float(source_axis["start_sec"]) - start / fps, 6),
            "end_sec": round(float(source_axis["end_sec"]) - start / fps, 6),
            "start_frame": int(source_axis["start_frame_floor"]) - start,
            "end_frame_exclusive": int(source_axis["end_frame_exclusive_ceil"])
            - start,
        }
    peak = turn["peak"]
    peak_frame = int(peak["source_frame"])
    peak["turn_frame"] = peak_frame - start
    peak["turn_sec"] = round((peak_frame - start) / fps, 6)
    left_rest = float(requested["left_rest_score_rad_s"])
    right_rest = float(requested["right_rest_score_rad_s"])
    peak["prominence_over_boundaries"] = round(
        float(peak["energy_rad_s"]) / max(left_rest, right_rest, 0.05), 8
    )
    turn["left_natural_boundary"] = {
        "frame": start,
        "source_sec": requested["source_start_sec"],
        "rest_score_rad_s": requested["left_rest_score_rad_s"],
        "context_level": requested_level,
    }
    turn["right_natural_boundary"] = {
        "frame": end,
        "source_sec": requested["source_end_sec"],
        "rest_score_rad_s": requested["right_rest_score_rad_s"],
        "context_level": requested_level,
    }
    turn["onset_frame_count"] = peak_frame - start
    turn["recovery_frame_count"] = end - peak_frame
    expanded["expression_turn"] = turn

    expanded["window"] = {
        "start_frame": start,
        "end_frame_exclusive": end,
        "frame_count": frame_count,
        "start_sec": round(start / fps, 6),
        "end_sec": round(end / fps, 6),
        "duration_sec": round(frame_count / fps, 6),
        "motion_metric_role": (
            "not_reused_from_base_interval_physical_retarget_recomputes_quality"
        ),
    }
    expanded["duration_band"] = _duration_band(frame_count, fps)
    expanded["semantic_supervision_mask"] = False
    expanded["license_training_admission"] = False
    expanded["license_training_admission_status"] = LICENSE_STATUS
    expanded["processing_scope"] = PROCESSING_SCOPE
    expanded["expansion_provenance"] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": EXPANSION_PROVENANCE_KIND,
        "sample_id": request["sample_id"],
        "base_task_id": base_task_id,
        "base_expression_turn_record_sha256": candidate[
            "expression_turn_record_sha256"
        ],
        "base_candidate_catalog_sha256": candidate_catalog_sha256,
        "expansion_plan_summary_sha256": plan_summary_sha256,
        "expansion_requests_sha256": expansion_requests_sha256,
        "hidden_mapping_sha256": hidden_mapping_sha256,
        "plan_record_sha256": request["plan_record_sha256"],
        "comparison_record_sha256": request["comparison_record_sha256"],
        "reviewed_context_level": reviewed_level,
        "requested_context_level": requested_level,
        "selection_policy": SELECTION_POLICY,
        "expansion_unit": EXPANSION_UNIT,
        "elapsed_duration_used_as_gate": False,
        "physical_quality_only": True,
        "semantic_admission": False,
        "affect_admission": False,
        "license_training_admission": False,
        "accepted_for_training": False,
    }
    expanded.pop("expression_turn_record_sha256", None)
    expanded["expression_turn_record_sha256"] = ordinary.json_sha256(expanded)
    return expanded


def load_and_derive_candidates(
    *,
    expansion_requests: Path,
    expansion_plan_summary: Path,
    hidden_mapping: Path,
    candidate_catalog: Path,
    catalog_summary: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate every private/public binding and derive deterministic candidates."""

    paths = {
        "expansion_requests": expansion_requests.resolve(),
        "expansion_plan_summary": expansion_plan_summary.resolve(),
        "hidden_mapping": hidden_mapping.resolve(),
        "candidate_catalog": candidate_catalog.resolve(),
        "catalog_summary": catalog_summary.resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    request_hash = ordinary.sha256(paths["expansion_requests"])
    mapping_hash = ordinary.sha256(paths["hidden_mapping"])
    catalog_hash = ordinary.sha256(paths["candidate_catalog"])
    plan_hash = ordinary.sha256(paths["expansion_plan_summary"])

    plan = _load_json(paths["expansion_plan_summary"])
    if (
        plan.get("artifact_kind") != PLAN_ARTIFACT_KIND
        or plan.get("selection_policy") != SELECTION_POLICY
        or plan.get("fixed_minimum_maximum_or_target_duration_used") is not False
        or plan.get("accepted_for_training_count") != 0
    ):
        raise ValueError("expansion plan summary is not fail-closed")
    inputs = plan.get("inputs") or {}
    expected_inputs = {
        "hidden_mapping_sha256": mapping_hash,
        "candidate_catalog_sha256": catalog_hash,
    }
    if any(inputs.get(key) != value for key, value in expected_inputs.items()):
        raise ValueError("expansion plan input binding mismatch")
    output = (plan.get("outputs") or {}).get("expansion_requests") or {}
    declared_path = Path(str(output.get("path") or "")).resolve()
    if (
        declared_path != paths["expansion_requests"]
        or output.get("sha256") != request_hash
    ):
        raise ValueError("expansion request manifest is not bound by its plan summary")

    full_binding, catalog_audit = base.load_catalog_binding(
        paths["candidate_catalog"], paths["catalog_summary"], "full_pool"
    )
    if catalog_audit.get("catalog_candidate_manifest_sha256") != catalog_hash:
        raise ValueError("catalog summary does not bind the complete candidate manifest")
    validator_base_binding = {
        key: full_binding[key]
        for key in (
            "retarget_input_manifest_sha256",
            "expression_turn_contract_sha256",
            "source_inventory_manifest_sha256",
            "split_assignment_manifest_sha256",
            "selection_kind",
            "require_selection_record",
        )
    }

    requests = _index_unique(
        _read_jsonl(paths["expansion_requests"]),
        "sample_id",
        context="expansion requests",
    )
    if output.get("records") != len(requests) or not requests:
        raise ValueError("expansion request count differs from plan summary")
    for request in requests.values():
        _validate_request(request)
    mapping = _index_unique(
        _read_jsonl(paths["hidden_mapping"]),
        "sample_id",
        context="hidden mapping",
    )
    missing_samples = sorted(set(requests).difference(mapping))
    if missing_samples:
        raise ValueError(f"hidden mapping lacks expansion samples: {missing_samples[:3]}")

    requested_task_ids = {str(row["base_task_id"]) for row in requests.values()}
    candidates: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(paths["candidate_catalog"]):
        task_id = row.get("task_id")
        if task_id not in requested_task_ids:
            continue
        if task_id in candidates:
            raise ValueError(f"candidate catalog contains duplicate task: {task_id}")
        validate_expression_turn_candidate(row, catalog_binding=validator_base_binding)
        candidates[str(task_id)] = row
    missing_tasks = sorted(requested_task_ids.difference(candidates))
    if missing_tasks:
        raise ValueError(f"candidate catalog lacks expansion tasks: {missing_tasks[:3]}")

    derived: list[dict[str, Any]] = []
    for sample_id in sorted(requests):
        request = requests[sample_id]
        hidden = mapping[sample_id]
        candidate = candidates[str(request["base_task_id"])]
        exact = {
            "task_id": request["base_task_id"],
            "source_clip_id": request["source_clip_id"],
            "fixed_split_assignment": request["fixed_split_assignment"],
            "expression_turn_record_sha256": request[
                "base_expression_turn_record_sha256"
            ],
        }
        failed = sorted(key for key, value in exact.items() if hidden.get(key) != value)
        if failed:
            raise ValueError(f"{sample_id}: hidden mapping mismatch: {failed}")
        if any(candidate.get(key) != value for key, value in exact.items()):
            raise ValueError(f"{sample_id}: candidate/request binding mismatch")
        expanded = derive_expanded_candidate(
            candidate,
            request,
            plan_summary_sha256=plan_hash,
            expansion_requests_sha256=request_hash,
            hidden_mapping_sha256=mapping_hash,
            candidate_catalog_sha256=catalog_hash,
        )
        derived.append(expanded)

    audit = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "expression_turn_v8_expansion_input_audit_v1",
        "selection_policy": SELECTION_POLICY,
        "processing_scope": PROCESSING_SCOPE,
        "records": len(derived),
        "source_records": len({row["source_clip_id"] for row in derived}),
        "inputs": {
            key: ordinary.file_binding(value) for key, value in paths.items()
        },
        "catalog_binding": catalog_audit,
        "fixed_minimum_maximum_or_target_duration_used": False,
        "semantic_admission": False,
        "affect_admission": False,
        "license_training_admission": False,
        "accepted_for_training": False,
    }
    audit["sha256"] = ordinary.json_sha256(audit)
    return derived, audit


def materialize_derived_manifest(
    records: list[dict[str, Any]], path: Path, *, resume: bool
) -> str:
    payload = "".join(ordinary.stable_json(row) + "\n" for row in records)
    expected = ordinary.json_sha256(records)
    if path.exists():
        existing_rows = _read_jsonl(path)
        if ordinary.json_sha256(existing_rows) != expected:
            raise RuntimeError("existing derived candidate manifest changed")
        if not resume:
            raise RuntimeError(f"existing pipeline state requires --resume: {path}")
    else:
        ordinary.atomic_text(path, payload)
    return ordinary.sha256(path)


def tasks_from_derived_candidates(
    records: list[dict[str, Any]],
    *,
    derived_manifest_sha256: str,
    candidate_catalog_sha256: str,
    beat2_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    first = records[0]
    binding = {
        "retarget_input_manifest_sha256": derived_manifest_sha256,
        "expression_turn_contract_sha256": first[
            "expression_turn_contract_sha256"
        ],
        "source_inventory_manifest_sha256": first[
            "source_inventory_manifest_sha256"
        ],
        "split_assignment_manifest_sha256": first[
            "split_assignment_manifest_sha256"
        ],
        "selection_kind": None,
        "require_selection_record": False,
    }
    tasks: list[dict[str, Any]] = []
    source_hash_cache: dict[Path, str] = {}
    seen: set[str] = set()
    for record in records:
        report = validate_expression_turn_candidate(record, catalog_binding=binding)
        task_id = str(record["task_id"])
        if task_id in seen:
            raise ValueError(f"duplicate derived task: {task_id}")
        seen.add(task_id)
        source = ordinary.resolve_contained_path(
            beat2_root,
            str(record["motion_relpath"]),
            field="motion_relpath",
            clip_id=task_id,
        )
        if not source.is_file():
            raise FileNotFoundError(source)
        source_hash_cache.setdefault(source, ordinary.sha256(source))
        if source_hash_cache[source] != record.get("motion_sha256"):
            raise ValueError(f"{task_id}: source motion SHA mismatch")
        segment = record["training_segment"]
        tasks.append(
            {
                **record,
                **report["lineage"],
                "selected_record_sha256_role": (
                    "derived_one_level_natural_context_candidate_row"
                ),
                "upstream_inventory_manifest_sha256": candidate_catalog_sha256,
                "retarget_input_manifest_sha256_role": (
                    "derived_expansion_candidate_execution_manifest"
                ),
                "source": str(source),
                "start_frame": int(segment["start_frame"]),
                "end_frame_exclusive": int(segment["end_frame_exclusive"]),
                "source_warnings": [],
                "conditioning_text_status": (
                    "disabled_pending_fresh_independent_arc_action_review"
                ),
                "accepted_for_training": False,
            }
        )
    return tasks, binding


class ExpansionRuntime(base.ExpressionTurnRuntime):
    """Attach expansion and closed-admission lineage to physical quality."""

    def retarget_turn(
        self, task: dict[str, Any], turn: dict[str, Any], output_dir: Path
    ) -> dict[str, Any]:
        quality = super().retarget_turn(task, turn, output_dir)
        quality.update(
            {
                field: task[field]
                for field in EXPANSION_AUDIT_FIELDS
                if field in task
            }
        )
        quality.update(
            {
                "physical_quality_only": True,
                "semantic_admission": False,
                "affect_admission": False,
                "license_training_admission": False,
                "accepted_for_training": False,
            }
        )
        return quality


def expansion_worker_runtime(config: GroupedRuntimeConfig) -> ExpansionRuntime:
    return ExpansionRuntime(base.grouped.worker_runtime(config))


def _worker_entry(payload: tuple[Any, ...]) -> list[dict[str, Any]]:
    return base.run_source_group(*payload, runtime_factory=expansion_worker_runtime)


def build_run_contract(
    args: argparse.Namespace,
    input_audit: dict[str, Any],
    validator_binding: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    contract, _ = base.build_run_contract(args, input_audit["catalog_binding"])
    contract["artifacts"]["expansion_physical_pipeline"] = ordinary.file_binding(
        Path(__file__)
    )
    contract["expansion_input_audit"] = input_audit
    contract["input_contract"]["base_catalog_binding"] = contract[
        "input_contract"
    ].pop("catalog_binding")
    contract["input_contract"]["derived_execution_manifest"] = (
        ordinary.file_binding(args.inventory)
    )
    contract["input_contract"]["derived_validator_binding"] = dict(
        validator_binding
    )
    contract["expansion_policy"] = {
        "selection_policy": SELECTION_POLICY,
        "expansion_unit": EXPANSION_UNIT,
        "context_policy": CONTEXT_POLICY,
        "elapsed_seconds_used_for_selection": False,
        "fixed_six_second_windows_allowed": False,
        "processing_scope": PROCESSING_SCOPE,
        "semantic_admission": False,
        "affect_admission": False,
        "license_training_admission": False,
        "accepted_for_training": False,
    }
    return contract, ordinary.json_sha256(contract)


def _status(
    *,
    args: argparse.Namespace,
    input_audit: dict[str, Any],
    tasks: list[dict[str, Any]],
    selected_groups: list[list[dict[str, Any]]],
    results: list[dict[str, Any]],
    pending: list[dict[str, Any]],
    run_id: str,
    run_state: str,
    started_at: str,
    run_contract: dict[str, Any],
    run_contract_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "expression_turn_v8_expansion_retarget_status_v1",
        "run_id": run_id,
        "run_state": run_state,
        "started_at": started_at,
        "updated_at": ordinary.utc_now(),
        "input_audit": input_audit,
        "derived_manifest": str(args.inventory.resolve()),
        "derived_manifest_sha256": ordinary.sha256(args.inventory),
        "run_contract": run_contract,
        "run_contract_sha256": run_contract_hash,
        "eligible_turn_count": len(tasks),
        "eligible_source_count": len(grouped.group_tasks_by_source(tasks)),
        "selected_turn_count": sum(map(len, selected_groups)),
        "selected_source_count": len(selected_groups),
        "terminal_turn_count": len(results),
        "pending_turn_count": len(pending),
        "counts": dict(sorted(Counter(row.get("status") for row in results).items())),
        "processing_scope": PROCESSING_SCOPE,
        "semantic_admission": False,
        "affect_admission": False,
        "license_training_admission": False,
        "accepted_for_training": False,
    }


def run_retarget(
    *,
    args: argparse.Namespace,
    tasks: list[dict[str, Any]],
    validator_binding: dict[str, Any],
    input_audit: dict[str, Any],
) -> dict[str, Any]:
    retarget_root = args.output_root / "retarget"
    args.output_root = retarget_root
    all_groups = grouped.group_tasks_by_source(tasks)
    selected_groups = grouped.limit_groups(
        all_groups,
        limit_sources=args.limit_sources,
        limit_events=args.limit_turns,
    )
    run_contract, run_contract_hash = build_run_contract(
        args, input_audit, validator_binding
    )
    inventory_hash = ordinary.sha256(args.inventory)
    status_path = retarget_root / "status.json"
    if status_path.exists():
        if not args.resume:
            raise RuntimeError(f"existing retarget state requires --resume: {status_path}")
        ordinary.validate_resume_contract(
            status_path,
            retarget_root,
            inventory_hash,
            run_contract,
            run_contract_hash,
        )
        ordinary.validate_saved_result_contracts(
            retarget_root, tasks, run_contract_hash
        )
    elif (retarget_root / ordinary.RUN_CONTRACT_FILENAME).exists() or any(
        (retarget_root / "state/results").glob("*.json")
    ):
        raise RuntimeError("retarget state is incomplete; refusing unsafe reuse")

    selected_tasks = [task for group in selected_groups for task in group]
    runnable = base.select_runnable_tasks(
        selected_tasks, args, run_contract_hash, validator_binding
    )
    runnable_ids = {task["task_id"] for task in runnable}
    runnable_groups = [
        [task for task in group if task["task_id"] in runnable_ids]
        for group in selected_groups
    ]
    runnable_groups = [group for group in runnable_groups if group]
    retarget_root.mkdir(parents=True, exist_ok=True)
    contract_path = retarget_root / ordinary.RUN_CONTRACT_FILENAME
    if not contract_path.exists():
        ordinary.atomic_json(
            contract_path,
            {"run_contract_sha256": run_contract_hash, "run_contract": run_contract},
        )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    started_at = ordinary.utc_now()
    results, pending = ordinary.write_manifests(
        retarget_root,
        tasks,
        [],
        args.inventory,
        inventory_hash,
        run_contract_hash,
    )
    ordinary.atomic_json(
        status_path,
        _status(
            args=args,
            input_audit=input_audit,
            tasks=tasks,
            selected_groups=selected_groups,
            results=results,
            pending=pending,
            run_id=run_id,
            run_state="running",
            started_at=started_at,
            run_contract=run_contract,
            run_contract_hash=run_contract_hash,
        ),
    )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _worker_entry,
                (
                    group,
                    args,
                    inventory_hash,
                    run_contract_hash,
                    validator_binding,
                    run_id,
                ),
            ): group
            for group in runnable_groups
        }
        for index, future in enumerate(as_completed(futures), 1):
            group = futures[future]
            try:
                group_results = future.result()
            except Exception as error:
                grouped._record_group_failure(
                    group,
                    args,
                    inventory_hash,
                    run_contract_hash,
                    run_id,
                    error,
                )
                group_results = []
            results, pending = ordinary.write_manifests(
                retarget_root,
                tasks,
                [],
                args.inventory,
                inventory_hash,
                run_contract_hash,
            )
            ordinary.atomic_json(
                status_path,
                _status(
                    args=args,
                    input_audit=input_audit,
                    tasks=tasks,
                    selected_groups=selected_groups,
                    results=results,
                    pending=pending,
                    run_id=run_id,
                    run_state="running",
                    started_at=started_at,
                    run_contract=run_contract,
                    run_contract_hash=run_contract_hash,
                ),
            )
            print(
                f"[{index:04d}/{len(runnable_groups):04d}] "
                f"{group[0]['source_clip_id']}: "
                f"{dict(Counter(row['status'] for row in group_results))}",
                flush=True,
            )
    results, pending = ordinary.write_manifests(
        retarget_root,
        tasks,
        [],
        args.inventory,
        inventory_hash,
        run_contract_hash,
    )
    state = "finished" if not pending else "finished_partial_selection"
    status = _status(
        args=args,
        input_audit=input_audit,
        tasks=tasks,
        selected_groups=selected_groups,
        results=results,
        pending=pending,
        run_id=run_id,
        run_state=state,
        started_at=started_at,
        run_contract=run_contract,
        run_contract_hash=run_contract_hash,
    )
    ordinary.atomic_json(status_path, status)
    return status


def build_render_queue(
    *,
    tasks: list[dict[str, Any]],
    validator_binding: dict[str, Any],
    retarget_root: Path,
    output_path: Path,
    expected_run_contract_sha256: str,
) -> dict[str, Any]:
    status_path = retarget_root / "status.json"
    contract_path = retarget_root / ordinary.RUN_CONTRACT_FILENAME
    status = _load_json(status_path)
    wrapper = _load_json(contract_path)
    contract = wrapper.get("run_contract")
    if (
        not ordinary._is_sha256(expected_run_contract_sha256)
        or wrapper.get("run_contract_sha256") != expected_run_contract_sha256
        or not isinstance(contract, dict)
        or ordinary.json_sha256(contract) != expected_run_contract_sha256
        or status.get("run_contract_sha256") != expected_run_contract_sha256
        or status.get("run_contract") != contract
        or status.get("processing_scope") != PROCESSING_SCOPE
        or status.get("semantic_admission") is not False
        or status.get("affect_admission") is not False
        or status.get("license_training_admission") is not False
        or status.get("accepted_for_training") is not False
    ):
        raise ValueError("retarget state is not bound to the current closed run contract")
    records: list[dict[str, Any]] = []
    for task in tasks:
        result_path = ordinary.result_path(retarget_root, task)
        result = ordinary.load_result(result_path)
        if result is None or result.get("status") != "passed":
            continue
        if not expansion_pass_is_current(
            result, task, expected_run_contract_sha256, validator_binding
        ):
            raise ValueError(f"{task['task_id']}: retarget pass is not current")
        quality_path = Path(result["quality_json"]).resolve()
        quality = _load_json(quality_path)
        trajectory = Path(result["safe_csv"]).resolve()
        gates = quality.get("quality_gate") or {}
        if not gates or any(value is not True for value in gates.values()):
            raise ValueError(f"{task['task_id']}: physical quality gate failed")
        output_frames, final_trajectory_role, safety_retime = (
            video_queue._validate_final_output_contract(
                task_id=task["task_id"],
                record=result,
                training=task["training_segment"],
                retarget=result["retarget_segment"],
                quality=quality,
                trajectory=trajectory,
            )
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_kind": QUEUE_ARTIFACT_KIND,
                "task_id": task["task_id"],
                "source_clip_id": task["source_clip_id"],
                "speaker_key": task["speaker_key"],
                "official_split": task["official_split"],
                "fixed_split_assignment": task["fixed_split_assignment"],
                "robot_contract": renderer.ROBOT_CONTRACT,
                "fps": task["fps"],
                "canonical_action": None,
                "canonical_action_role": (
                    "disabled_pending_fresh_independent_blind_review"
                ),
                "canonical_prompt": {
                    "en": "Anonymous silent robot motion sample.",
                    "zh": "Anonymous silent robot motion sample.",
                },
                "canonical_prompt_role": (
                    "anonymous_renderer_placeholder_not_conditioning_text"
                ),
                "trajectory_path": str(trajectory),
                "trajectory_sha256": result["safe_csv_sha256"],
                "trajectory_frames_expected": output_frames,
                "source_frame_count": task["training_segment"]["frame_count"],
                "output_frame_count": output_frames,
                "final_trajectory_role": final_trajectory_role,
                "blind_review_must_use_final_trajectory": True,
                "quality_json": str(quality_path),
                "quality_json_sha256": result["quality_json_sha256"],
                "retarget_result_json": str(result_path.resolve()),
                "retarget_result_json_sha256": ordinary.sha256(result_path),
                "quality_gate": gates,
                "training_segment": task["training_segment"],
                "context_plan": task["context_plan"],
                "time_axes": task["time_axes"],
                "expression_turn": task["expression_turn"],
                "retarget_segment": result["retarget_segment"],
                "core_interval": task.get("core_interval"),
                "duration_band": task.get("duration_band"),
                "semantic_supervision_masks": dict(base.SEMANTIC_MASKS),
                "semantic_supervision_mask": False,
                "emotion_supervision_mask": False,
                "official_emotion_conditioning_enabled": False,
                "affect_observable_supervision_mask": False,
                "official_category_conditioning_enabled": False,
                "license_training_admission": False,
                "license_training_admission_status": LICENSE_STATUS,
                "expansion_provenance": task["expansion_provenance"],
                "expression_turn_contract_sha256": task.get(
                    "expression_turn_contract_sha256"
                ),
                "expression_turn_record_sha256": task.get(
                    "expression_turn_record_sha256"
                ),
                "source_inventory_manifest_sha256": task.get(
                    "source_inventory_manifest_sha256"
                ),
                "split_assignment_manifest_sha256": task.get(
                    "split_assignment_manifest_sha256"
                ),
                "upstream_inventory_record_sha256": task.get(
                    "upstream_inventory_record_sha256"
                ),
                "selected_record_sha256": task.get("selected_record_sha256"),
                "retarget_input_manifest_sha256": task.get(
                    "retarget_input_manifest_sha256"
                ),
                "processing_scope": PROCESSING_SCOPE,
                "manual_review_required": True,
                "semantic_action_completeness_review_required": True,
                "affect_observable_review_required": True,
                "render_pass_grants_training_admission": False,
                "speech_context_included": False,
                "accepted_for_training": False,
                **(
                    {"safety_monotonic_retime": safety_retime}
                    if safety_retime is not None
                    else {}
                ),
            }
        )
    if not records:
        raise ValueError("no current physical-QC passes are available to render")
    records.sort(key=lambda row: row["task_id"])
    ordinary.atomic_jsonl(output_path, records)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": QUEUE_ARTIFACT_KIND,
        "records": len(records),
        "output": str(output_path.resolve()),
        "output_sha256": ordinary.sha256(output_path),
        "context_levels": dict(
            sorted(
                Counter(
                    row["expansion_provenance"]["requested_context_level"]
                    for row in records
                ).items()
            )
        ),
        "selection_policy": SELECTION_POLICY,
        "retarget_status": ordinary.file_binding(status_path),
        "retarget_run_contract": ordinary.file_binding(contract_path),
        "retarget_run_contract_sha256": expected_run_contract_sha256,
        "elapsed_seconds_used_for_selection": False,
        "processing_scope": PROCESSING_SCOPE,
        "semantic_admission": False,
        "affect_admission": False,
        "license_training_admission": False,
        "accepted_for_training": False,
    }
    ordinary.atomic_json(output_path.with_suffix(".summary.json"), summary)
    return summary


def _canonicalize_safety_metric_key_order(
    quality: dict[str, Any],
) -> dict[str, Any]:
    """Restore semantic joint order after JSON writers sort mapping keys."""

    canonical = copy.deepcopy(quality)
    audit = canonical.get("safety_monotonic_retime")
    if not isinstance(audit, dict):
        return canonical
    expected = set(JOINT_ORDER_18D)
    for field in SAFETY_ORDERED_METRIC_FIELDS:
        values = audit.get(field)
        if isinstance(values, dict) and set(values) == expected:
            audit[field] = {joint: values[joint] for joint in JOINT_ORDER_18D}
    return canonical


def expansion_pass_is_current(
    result: dict[str, Any],
    task: dict[str, Any],
    run_contract_hash: str,
    validator_binding: dict[str, Any],
) -> bool:
    """Revalidate a disk result without treating JSON object order as data."""

    if (
        result.get("status") != "passed"
        or result.get("run_contract_sha256") != run_contract_hash
    ):
        return False
    output_dir = Path(str(result.get("output_dir") or ""))
    quality_path = output_dir / "quality.json"
    try:
        quality = _canonicalize_safety_metric_key_order(_load_json(quality_path))
        safe_csv = ordinary.only_safe_csv(output_dir)
        source_hash = ordinary.sha256(Path(task["source"]))
    except (OSError, ValueError):
        return False
    return bool(
        ordinary.result_lineage_matches(result, task)
        and base.expression_quality_passes(
            quality,
            task,
            source_hash,
            validator_binding,
            safe_csv_path=safe_csv,
        )
        and result.get("safe_csv_sha256") == ordinary.sha256(safe_csv)
        and result.get("quality_json_sha256") == ordinary.sha256(quality_path)
    )


def run_render(
    *, args: argparse.Namespace, queue_path: Path, render_root: Path
) -> dict[str, Any]:
    original_fields = renderer.SEMANTIC_AUDIT_FIELDS
    renderer.SEMANTIC_AUDIT_FIELDS = original_fields + tuple(
        field for field in EXPANSION_AUDIT_FIELDS if field not in original_fields
    )
    try:
        return renderer.run_review(
            queue_path=queue_path,
            output_root=render_root,
            renderer_python=args.renderer_python,
            urdf=args.urdf,
            limit=None,
            sampling="sequential",
            seed=0,
            workers=args.render_workers,
            width=args.width,
            height=args.height,
            resume=args.resume,
            retry_failed=args.retry_failed,
        )
    finally:
        renderer.SEMANTIC_AUDIT_FIELDS = original_fields


def _validate_cli(args: argparse.Namespace) -> None:
    for field in ("workers", "render_workers"):
        if getattr(args, field) < 1:
            raise ValueError(f"{field.replace('_', '-')} must be positive")
    for field in ("limit_sources", "limit_turns"):
        value = getattr(args, field)
        if value is not None and value < 1:
            raise ValueError(f"{field.replace('_', '-')} must be positive")
    if args.retry_failed and not args.resume:
        raise ValueError("--retry-failed requires --resume")
    if args.stage == "render" and (args.limit_sources or args.limit_turns):
        raise ValueError("render stage does not accept retarget selection limits")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_cli(args)
    for field in (
        "expansion_requests",
        "expansion_plan_summary",
        "hidden_mapping",
        "candidate_catalog",
        "catalog_summary",
        "beat2_root",
        "smplx_model",
        "gmr_root",
        "urdf",
        "config",
        "renderer_python",
    ):
        path = getattr(args, field).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        setattr(args, field, path)
    pipeline_root = args.output_root.resolve()
    pipeline_root.mkdir(parents=True, exist_ok=True)
    args.output_root = pipeline_root
    existing_pipeline_state = [
        path
        for path in (
            pipeline_root / "input_audit.json",
            pipeline_root / "derived_expanded_candidates.jsonl",
            pipeline_root / "pipeline_summary.json",
            pipeline_root / "retarget/status.json",
        )
        if path.exists()
    ]
    if args.stage != "render" and existing_pipeline_state and not args.resume:
        raise RuntimeError(
            "existing expansion pipeline state requires --resume: "
            f"{existing_pipeline_state[0]}"
        )
    if (
        args.stage == "render"
        and (pipeline_root / "rendered/status.json").exists()
        and not args.resume
    ):
        raise RuntimeError("existing render state requires --resume")

    records, input_audit = load_and_derive_candidates(
        expansion_requests=args.expansion_requests,
        expansion_plan_summary=args.expansion_plan_summary,
        hidden_mapping=args.hidden_mapping,
        candidate_catalog=args.candidate_catalog,
        catalog_summary=args.catalog_summary,
    )
    input_audit_path = pipeline_root / "input_audit.json"
    if input_audit_path.exists():
        if _load_json(input_audit_path) != input_audit:
            raise RuntimeError("expansion input audit changed; refusing unsafe reuse")
    else:
        ordinary.atomic_json(input_audit_path, input_audit)
    derived_manifest = pipeline_root / "derived_expanded_candidates.jsonl"
    derived_hash = materialize_derived_manifest(
        records, derived_manifest, resume=args.resume or args.stage == "render"
    )
    tasks, validator_binding = tasks_from_derived_candidates(
        records,
        derived_manifest_sha256=derived_hash,
        candidate_catalog_sha256=ordinary.sha256(args.candidate_catalog),
        beat2_root=args.beat2_root,
    )
    args.inventory = derived_manifest
    args.selection_kind = "full_pool"
    args.retry_selection_audit = None

    retarget_root = pipeline_root / "retarget"
    retarget_status = None
    if args.stage in {"all", "retarget"}:
        args.output_root = pipeline_root
        retarget_status = run_retarget(
            args=args,
            tasks=tasks,
            validator_binding=validator_binding,
            input_audit=input_audit,
        )
    queue_path = pipeline_root / "physical_review_queue.jsonl"
    queue_summary = None
    render_summary = None
    if args.stage in {"all", "render"}:
        args.output_root = retarget_root
        _current_contract, current_contract_hash = build_run_contract(
            args, input_audit, validator_binding
        )
        queue_summary = build_render_queue(
            tasks=tasks,
            validator_binding=validator_binding,
            retarget_root=retarget_root,
            output_path=queue_path,
            expected_run_contract_sha256=current_contract_hash,
        )
        render_summary = run_render(
            args=args, queue_path=queue_path, render_root=pipeline_root / "rendered"
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": PIPELINE_ARTIFACT_KIND,
        "stage": args.stage,
        "input_audit": input_audit,
        "derived_candidate_manifest": str(derived_manifest),
        "derived_candidate_manifest_sha256": derived_hash,
        "derived_candidate_count": len(records),
        "retarget_status": retarget_status,
        "review_queue_summary": queue_summary,
        "render_summary": render_summary,
        "selection_policy": SELECTION_POLICY,
        "fixed_six_second_windows_used": False,
        "elapsed_seconds_used_for_selection": False,
        "processing_scope": PROCESSING_SCOPE,
        "semantic_admission": False,
        "affect_admission": False,
        "license_training_admission": False,
        "accepted_for_training": False,
    }
    ordinary.atomic_json(pipeline_root / "pipeline_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
