#!/usr/bin/env python3
"""Retarget full BEAT2 v8 expression turns with safety-only monotonic slowdown."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from . import batch_retarget_beat2_semantic_events_v2 as grouped
    from . import batch_retarget_beat2_v2 as ordinary
    from . import retarget_beat2_grouped_v2 as grouped_runtime
    from .retarget_beat2_grouped_v2 import (
        GroupedBeat2RetargetRuntime,
        GroupedRuntimeConfig,
    )
except ImportError:  # pragma: no cover - direct invocation by path
    import batch_retarget_beat2_semantic_events_v2 as grouped
    import batch_retarget_beat2_v2 as ordinary
    import retarget_beat2_grouped_v2 as grouped_runtime
    from retarget_beat2_grouped_v2 import (
        GroupedBeat2RetargetRuntime,
        GroupedRuntimeConfig,
    )

from tools.human_motion_review.expression_turn_contract import (
    CONTEXT_POLICY,
    ExpressionTurnContractError,
    validate_expression_turn_candidate,
)
from tools.human_motion_review.expression_turn_retarget_contract import (
    QUALITY_ARTIFACT_KIND,
    RETARGET_SEGMENT_REPRESENTATION,
    TRAINING_ADMISSION_STATUS,
    validate_expression_turn_retarget_output,
)
from tools.gmr_v2.safety_monotonic_retime_v1 import (
    ALGORITHM_CONTRACT_SHA256,
    ALGORITHM_NAME,
    MAX_SLOWDOWN_RATIO,
    minimum_velocity_safety_retime,
)


DEFAULT_CATALOG_ROOT = Path(
    "/home/gez/nas/cloud/gez/human_motion/catalog/beat2_expression_turn_v8"
)
DEFAULT_INVENTORY = (
    DEFAULT_CATALOG_ROOT / "beat2_expression_turn_v8.representative100.jsonl"
)
DEFAULT_CATALOG_SUMMARY = DEFAULT_CATALOG_ROOT / "beat2_expression_turn_v8.summary.json"
DEFAULT_BEAT2_ROOT = Path("/home/gez/nas/cloud/gez/human_motion/raw/BEAT2")
DEFAULT_OUTPUT_ROOT = Path(
    "/home/gez/nas/cloud/gez/human_motion/processed/"
    "beat2_expression_turn_representative100_18d_v8"
)
BUILDER_SCRIPT = (
    PROJECT_ROOT
    / "tools/human_motion_collection/build_beat2_expression_turn_v8.py"
)
INPUT_CONTRACT_SCRIPT = (
    PROJECT_ROOT / "tools/human_motion_review/expression_turn_contract.py"
)
OUTPUT_CONTRACT_SCRIPT = (
    PROJECT_ROOT
    / "tools/human_motion_review/expression_turn_retarget_contract.py"
)
SAFETY_RETIME_SCRIPT = (
    PROJECT_ROOT / "tools/gmr_v2/safety_monotonic_retime_v1.py"
)

SCHEMA_VERSION = "1.0.0"
INPUT_ARTIFACT_KIND = "beat2_expression_turn_v8_candidate"
INPUT_REPRESENTATION = "native_variable_length_expression_turn_v1"
OUTPUT_ARTIFACT_KIND = "ula_v2_18d_expression_turn_retarget_v1"
NATURAL_DURATION_POLICY = "natural_rest_to_natural_rest_no_fixed_or_max_duration"
LEGACY_SEMANTIC_FIELDS = {
    "official_gesture_semantic_spans",
    "official_semantic_event",
    "prompt",
    "prompt_contract",
    "prompt_schema",
    "prompt_sha256",
    "prompt_source",
    "semantic_event",
    "semantic_gesture",
    "semantic_label_status",
}
SEMANTIC_MASKS = {
    "official_category": False,
    "robot_observable_motion_form": False,
    "communicative_intent": False,
    "prompt_text": False,
    "legacy_gesture": False,
}

EXPRESSION_PROVENANCE_FIELDS = (
    "clip_id",
    "task_id",
    "source_clip_id",
    "source_group_key",
    "speaker_key",
    "fixed_split_assignment",
    "split_grouping",
    "representation",
    "core_interval",
    "context_plan",
    "training_segment",
    "time_axes",
    "expression_turn",
    "window",
    "duration_band",
    "event_count_band",
    "source_speech_context",
    "source_speech_context_role",
    "canonical_prompt",
    "canonical_prompt_role",
    "canonical_action",
    "canonical_action_role",
    "robot_observable_motion_form",
    "communicative_intent",
    "semantic_mapping_status",
    "semantic_supervision_masks",
    "official_category_verified",
    "official_category_role",
    "official_category_conditioning_enabled",
    "official_category_condition_channel",
    "official_category_loss",
    "emotion_id",
    "source_emotion_label",
    "source_emotion_label_verified",
    "emotion_label_source",
    "emotion_supervision_mask",
    "emotion_supervision_role",
    "official_emotion_conditioning_enabled",
    "official_emotion_condition_channel",
    "official_emotion_loss",
    "affect_observable_review_status",
    "affect_observable_supervision_mask",
    "audio_enabled",
    "expression_turn_contract_sha256",
    "expression_turn_record_sha256",
    "expression_turn_selection_kind",
    "expression_turn_selection_rank",
    "expression_turn_selection_status",
    "expression_turn_selection_record_sha256",
    "source_inventory_manifest_sha256",
    "split_assignment_manifest_sha256",
    "upstream_event_record_sha256",
    "inventory_record_sha256",
    "upstream_inventory_record_sha256",
    "selected_record_sha256",
    "selected_record_sha256_role",
    "upstream_inventory_manifest_sha256",
    "retarget_input_manifest_sha256",
    "retarget_input_manifest_sha256_role",
    "training_admission_status",
    "accepted_for_training",
)

EXPRESSION_EXECUTION_POLICY = {
    **grouped.GROUPED_EXECUTION_POLICY,
    "scheduling_unit": "source_clip_with_expression_turns",
    "isolation_unit": "expression_turn",
    "source_npz_loads_per_group": 1,
    "natural_training_segment_only": True,
    "preserve_native_frame_count": False,
    "preserve_every_source_frame_in_strictly_monotonic_time_map": True,
    "safety_derived_monotonic_slowdown_only": True,
    "safety_retime_algorithm": ALGORITHM_NAME,
    "safety_retime_contract_sha256": ALGORITHM_CONTRACT_SHA256,
    "max_slowdown_ratio": MAX_SLOWDOWN_RATIO,
    "crop_tile_or_target_duration_allowed": False,
    "legacy_semantic_event_fields_allowed": False,
    "fixed_duration_windows_allowed": False,
}

TASK_TRANSPORT_FIELDS = {
    "conditioning_text_status",
    "end_frame_exclusive",
    "inventory_record_sha256",
    "retarget_input_manifest_sha256",
    "retarget_input_manifest_sha256_role",
    "selected_record_sha256",
    "selected_record_sha256_role",
    "source",
    "source_warnings",
    "start_frame",
    "upstream_inventory_manifest_sha256",
    "upstream_inventory_record_sha256",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument(
        "--selection-kind",
        choices=("representative100", "stress100", "full_pool"),
        default="representative100",
    )
    parser.add_argument(
        "--catalog-summary", type=Path, default=DEFAULT_CATALOG_SUMMARY
    )
    parser.add_argument("--beat2-root", type=Path, default=DEFAULT_BEAT2_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--smplx-model", type=Path, default=ordinary.DEFAULT_MODEL)
    parser.add_argument("--gmr-root", type=Path, default=ordinary.DEFAULT_GMR_ROOT)
    parser.add_argument("--urdf", type=Path, default=ordinary.DEFAULT_URDF)
    parser.add_argument("--config", type=Path, default=ordinary.DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit-sources", type=int)
    parser.add_argument("--limit-turns", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--retry-from-diagnostic-root",
        type=Path,
        help=(
            "Select only prior outputs whose physical gates passed and whose "
            "sole failure was the obsolete native-frame retime prohibition."
        ),
    )
    return parser.parse_args(argv)


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unreadable JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_catalog_binding(
    inventory: Path, catalog_summary: Path, selection_kind: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind a sampled review set or the full candidate pool to its catalog."""

    inventory = inventory.resolve()
    catalog_summary = catalog_summary.resolve()
    summary = _load_json_object(catalog_summary)
    if summary.get("artifact_kind") != "beat2_expression_turn_v8_candidate_catalog":
        raise ValueError("catalog summary artifact_kind is invalid")

    inventory_hash = ordinary.sha256(inventory)
    output_hashes = summary.get("output_sha256")
    if not isinstance(output_hashes, dict) or output_hashes.get(inventory.name) != inventory_hash:
        raise ValueError("review-set manifest is not bound by the catalog summary")
    candidate_manifest_hash = output_hashes.get(
        "beat2_expression_turn_v8.candidates.jsonl"
    )
    if not ordinary._is_sha256(candidate_manifest_hash):
        raise ValueError("catalog candidate manifest hash is invalid")
    full_pool = selection_kind == "full_pool"
    if full_pool and (
        inventory.name != "beat2_expression_turn_v8.candidates.jsonl"
        or inventory_hash != candidate_manifest_hash
    ):
        raise ValueError("full_pool must use the catalog-bound complete candidate manifest")

    contract = summary.get("expression_turn_contract")
    if not isinstance(contract, dict):
        raise ValueError("catalog summary lacks expression_turn_contract")
    contract_hash = ordinary.json_sha256(contract)
    if summary.get("expression_turn_contract_sha256") != contract_hash:
        raise ValueError("catalog expression-turn contract hash mismatch")
    if contract.get("duration_policy") != NATURAL_DURATION_POLICY:
        raise ValueError("catalog duration policy is not natural-boundary variable length")
    if contract.get("fixed_window_sec") is not None:
        raise ValueError("catalog contains a fixed-duration window")
    if contract.get("semantic_supervision_masks") != SEMANTIC_MASKS:
        raise ValueError("catalog semantic masks are not fail-closed")
    if contract.get("emotion_supervision_mask") is not False:
        raise ValueError("catalog emotion supervision is not fail-closed")

    builder_hash = ordinary.sha256(BUILDER_SCRIPT)
    if contract.get("builder_script_sha256") != builder_hash:
        raise ValueError("catalog builder implementation hash mismatch")
    source_hash = summary.get("input_sha256")
    split_hash = summary.get("split_assignment_sha256")
    if contract.get("source_manifest_sha256") != source_hash:
        raise ValueError("catalog source inventory binding mismatch")
    split_contract = contract.get("split_assignment")
    if not isinstance(split_contract, dict) or split_contract.get("manifest_sha256") != split_hash:
        raise ValueError("catalog split assignment binding mismatch")
    for field, value in (
        ("source inventory", source_hash),
        ("split assignment", split_hash),
        ("builder", builder_hash),
        ("expression-turn contract", contract_hash),
    ):
        if not ordinary._is_sha256(value):
            raise ValueError(f"catalog {field} hash is invalid")

    source_path = Path(str(summary.get("input") or ""))
    split_path = Path(str(summary.get("split_assignment") or ""))
    if not source_path.is_file() or ordinary.sha256(source_path) != source_hash:
        raise ValueError("catalog source inventory changed or is unavailable")
    if not split_path.is_file() or ordinary.sha256(split_path) != split_hash:
        raise ValueError("catalog split assignment changed or is unavailable")

    binding = {
        "retarget_input_manifest_sha256": inventory_hash,
        "expression_turn_contract_sha256": contract_hash,
        "source_inventory_manifest_sha256": source_hash,
        "split_assignment_manifest_sha256": split_hash,
        "selection_kind": None if full_pool else selection_kind,
        "require_selection_record": not full_pool,
    }
    audit = {
        **binding,
        "execution_selection_kind": selection_kind,
        "full_pool_candidate_manifest": full_pool,
        "catalog_summary": ordinary.file_binding(catalog_summary),
        "catalog_builder": ordinary.file_binding(BUILDER_SCRIPT),
        "catalog_candidate_manifest_sha256": candidate_manifest_hash,
        "duration_policy": NATURAL_DURATION_POLICY,
        "context_policy": CONTEXT_POLICY,
    }
    return binding, audit


def _require_no_legacy_semantic_fields(record: dict[str, Any], clip_id: str) -> None:
    present = sorted(LEGACY_SEMANTIC_FIELDS.intersection(record))
    if present:
        raise ExpressionTurnContractError(
            f"{clip_id}: legacy semantic-event fields are forbidden: {present}"
        )


def read_expression_turn_inventory(
    inventory: Path,
    beat2_root: Path,
    catalog_binding: dict[str, Any],
    candidate_manifest_hash: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read only contract-valid selected v8 expression-turn review records."""

    tasks: list[dict[str, Any]] = []
    seen_clip_ids: set[str] = set()
    seen_task_ids: set[str] = set()
    source_hash_cache: dict[Path, str] = {}
    with inventory.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {inventory}:{line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected object at {inventory}:{line_number}")
            report = validate_expression_turn_candidate(
                record, catalog_binding=catalog_binding
            )
            clip_id = str(record["clip_id"])
            task_id = str(record.get("task_id") or clip_id)
            if clip_id in seen_clip_ids:
                raise ValueError(f"Duplicate expression-turn clip_id: {clip_id}")
            if task_id in seen_task_ids:
                raise ValueError(f"Duplicate expression-turn task_id: {task_id}")
            seen_clip_ids.add(clip_id)
            seen_task_ids.add(task_id)
            _require_no_legacy_semantic_fields(record, clip_id)

            motion_relpath = str(record.get("motion_relpath") or "").strip()
            if not motion_relpath:
                raise ValueError(f"{clip_id}: missing motion_relpath")
            source = ordinary.resolve_contained_path(
                beat2_root,
                motion_relpath,
                field="motion_relpath",
                clip_id=clip_id,
            )
            if not source.is_file():
                raise FileNotFoundError(source)
            if source not in source_hash_cache:
                source_hash_cache[source] = ordinary.sha256(source)
            actual_source_hash = source_hash_cache[source]
            if actual_source_hash != record.get("motion_sha256"):
                raise ValueError(f"{clip_id}: motion_sha256 mismatch")

            segment = record["training_segment"]
            lineage = report["lineage"]
            tasks.append(
                {
                    **record,
                    **lineage,
                    "selected_record_sha256_role": (
                        "canonical_full_row_in_current_v8_review_set_manifest"
                    ),
                    "upstream_inventory_manifest_sha256": (
                        candidate_manifest_hash
                        if ordinary._is_sha256(candidate_manifest_hash)
                        else record["source_inventory_manifest_sha256"]
                    ),
                    "retarget_input_manifest_sha256_role": (
                        "current_v8_expression_turn_review_set_manifest"
                    ),
                    "source": str(source),
                    "task_id": task_id,
                    "start_frame": int(segment["start_frame"]),
                    "end_frame_exclusive": int(segment["end_frame_exclusive"]),
                    "source_warnings": [],
                    "conditioning_text_status": (
                        "disabled_pending_independent_action_semantic_review"
                    ),
                    "accepted_for_training": False,
                }
            )
    if not tasks:
        raise ValueError(f"Expression-turn inventory is empty: {inventory}")
    return tasks, []


def build_expression_retarget_segment_contract(
    task: dict[str, Any],
    *,
    source_frame_count: int,
    output_frame_count: int,
    fps: float,
) -> dict[str, Any]:
    payload = {
        "representation": RETARGET_SEGMENT_REPRESENTATION,
        "source_start_frame": int(task["start_frame"]),
        "source_end_frame_exclusive": int(task["end_frame_exclusive"]),
        "source_frame_count": int(source_frame_count),
        "source_frame_coverage_sec": float(source_frame_count / fps),
        "output_frame_count": int(output_frame_count),
        "output_sample_span_sec": float(max(0, output_frame_count - 1) / fps),
        "output_frame_coverage_sec": float(output_frame_count / fps),
        "fps": float(fps),
        "retimed": int(output_frame_count) != int(source_frame_count),
        "cropped": False,
        "duration_policy": NATURAL_DURATION_POLICY,
        "retime_policy": ALGORITHM_NAME,
        "max_slowdown_ratio": MAX_SLOWDOWN_RATIO,
        "fixed_target_duration_sec": None,
        "planner_duration_field": "output_sample_span_sec",
        "source_boundary_duration_field": "source_frame_coverage_sec",
        "legacy_quality_duration_sec_role": (
            "output_frame_coverage_compatibility_only_not_planner_target"
        ),
    }
    return {**payload, "sha256": ordinary.json_sha256(payload)}


def _read_joint_csv(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
    if header != list(grouped_runtime.JOINT_ORDER_18D):
        raise ValueError(f"Unexpected 18D joint header: {path}")
    values = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if values.ndim != 2 or values.shape[1] != len(grouped_runtime.JOINT_ORDER_18D):
        raise ValueError(f"Unexpected 18D joint matrix: {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite 18D joint values: {path}")
    return np.asarray(values, dtype=np.float64)


def apply_expression_turn_safety_retime(
    quality: dict[str, Any],
    turn: dict[str, Any],
    delegate: GroupedBeat2RetargetRuntime,
) -> dict[str, Any]:
    """Replace the shared adapter's retime with the auditable v8 contract."""

    outputs = quality.get("outputs") or {}
    raw_csv = Path(str(outputs.get("raw_csv") or ""))
    safe_csv = Path(str(outputs.get("safe_csv") or ""))
    raw = _read_joint_csv(raw_csv)
    safe, key_times, output_times, retime_audit = minimum_velocity_safety_retime(
        raw,
        fps=float(quality["fps"]),
        max_velocity_rad_s=float(delegate.config.max_velocity_rad_s),
        smoothing_window=int(delegate.config.smoothing_window),
        joint_order=grouped_runtime.JOINT_ORDER_18D,
        joint_limits=grouped_runtime.JOINT_LIMITS_18D,
        elbow_enforcer=lambda values: grouped_runtime.enforce_safe_elbow_branch(
            values, joint_order=grouped_runtime.JOINT_ORDER_18D
        ),
    )
    grouped_runtime.write_csv(
        safe_csv, safe, joint_order=grouped_runtime.JOINT_ORDER_18D
    )

    source_targets = [
        turn["target_for_frame"](index) for index in range(len(raw))
    ]
    safe_targets = grouped_runtime.retime_targets(
        source_targets, key_times, output_times
    )
    raw_pose_metrics = grouped_runtime.rendered_pose_metrics(
        delegate.retargeter.model,
        delegate.mujoco,
        raw,
        source_targets,
        joint_order=grouped_runtime.JOINT_ORDER_18D,
    )
    pose_metrics = grouped_runtime.rendered_pose_metrics(
        delegate.retargeter.model,
        delegate.mujoco,
        safe,
        safe_targets,
        joint_order=grouped_runtime.JOINT_ORDER_18D,
    )
    pose_metrics.update(
        {
            f"raw_{key}": value
            for key, value in raw_pose_metrics.items()
            if key.startswith("limb_target_error")
        }
    )
    metadata = dict(quality)
    metadata.update(
        {
            "frames": int(len(safe)),
            "duration_sec": float(len(safe) / float(quality["fps"])),
            "retime_factor": float(len(safe) / len(raw)),
            "safety_monotonic_retime": retime_audit,
            "outputs": {
                "raw_csv": str(raw_csv.resolve()),
                "safe_csv": str(safe_csv.resolve()),
            },
        }
    )
    report = grouped_runtime.quality_report(
        raw,
        safe,
        float(quality["fps"]),
        float(delegate.config.max_velocity_rad_s),
        pose_metrics,
        metadata,
        joint_order=grouped_runtime.JOINT_ORDER_18D,
        joint_limits=grouped_runtime.JOINT_LIMITS_18D,
    )
    direction = grouped_runtime.axis_direction_metrics(
        safe,
        turn["alignment"],
        joint_order=grouped_runtime.JOINT_ORDER_18D,
    )
    direction["axis_policy"] = grouped_runtime.BEAT2_AXIS_POLICY
    report.update(direction)
    report["quality_gate"]["axis_direction_pass"] = direction[
        "axis_direction_pass"
    ]
    head = grouped_runtime.head_quality_metrics(
        turn["head_relative_rotations"],
        raw,
        safe,
        float(quality["fps"]),
        float(delegate.config.max_velocity_rad_s),
        joint_order=grouped_runtime.JOINT_ORDER_18D,
    )
    report.update(head)
    for key in (
        "head_joint_limits_pass",
        "head_velocity_pass",
        "head_direction_pass",
        "head_continuity_pass",
    ):
        report["quality_gate"][key] = head[key]
    report["quality_gate"].update(
        {
            "safety_slowdown_ratio_pass": retime_audit["slowdown_ratio_pass"],
            "safety_time_map_pass": retime_audit[
                "time_map_strictly_increasing"
            ],
            "safety_endpoint_pass": bool(
                retime_audit["first_frame_preserved"]
                and retime_audit["last_frame_preserved"]
            ),
            "safety_post_velocity_pass": retime_audit["post_velocity_pass"],
        }
    )
    report["quality_gate"]["passed"] = all(
        value
        for key, value in report["quality_gate"].items()
        if key != "passed"
    )
    return report


class ExpressionTurnRuntime:
    """Add v8 provenance and remove legacy semantic-event fields from quality."""

    def __init__(self, delegate: GroupedBeat2RetargetRuntime):
        self.delegate = delegate

    @property
    def source_hash(self) -> str:
        return self.delegate.source_hash

    def load_source(self, source: Path) -> None:
        self.delegate.load_source(source)

    def reset_turn(self, task: dict[str, Any]) -> dict[str, Any]:
        return self.delegate.reset_event(task)

    def retarget_turn(
        self,
        task: dict[str, Any],
        turn: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        quality = self.delegate.retarget_event(task, turn, output_dir)
        quality = apply_expression_turn_safety_retime(
            quality, turn, self.delegate
        )
        for field in LEGACY_SEMANTIC_FIELDS:
            quality.pop(field, None)
        quality.update(
            {
                field: task[field]
                for field in EXPRESSION_PROVENANCE_FIELDS
                if field in task
            }
        )
        quality.update(
            {
                "artifact_kind": QUALITY_ARTIFACT_KIND,
                "input_artifact_kind": INPUT_ARTIFACT_KIND,
                "input_representation": INPUT_REPRESENTATION,
                "accepted_for_training": False,
                "training_admission_status": TRAINING_ADMISSION_STATUS,
                "retarget_segment": build_expression_retarget_segment_contract(
                    task,
                    source_frame_count=int(quality["source_window_frames"]),
                    output_frame_count=int(quality["frames"]),
                    fps=float(quality["fps"]),
                ),
            }
        )
        safe_csv = ordinary.only_safe_csv(output_dir)
        quality["safe_csv_sha256"] = ordinary.sha256(safe_csv)
        return quality


def expression_worker_runtime(config: GroupedRuntimeConfig) -> ExpressionTurnRuntime:
    return ExpressionTurnRuntime(grouped.worker_runtime(config))


def expression_quality_passes(
    quality: dict[str, Any],
    task: dict[str, Any],
    source_hash: str,
    catalog_binding: dict[str, Any],
    *,
    safe_csv_path: Path | None = None,
) -> bool:
    expected_frames = task["end_frame_exclusive"] - task["start_frame"]
    segment = quality.get("retarget_segment") or {}
    provenance_matches = all(
        quality.get(field) == task.get(field)
        for field in (
            "expression_turn_contract_sha256",
            "expression_turn_record_sha256",
            "expression_turn_selection_record_sha256",
            "inventory_record_sha256",
            "upstream_inventory_record_sha256",
            "selected_record_sha256",
            "retarget_input_manifest_sha256",
            "training_segment",
            "time_axes",
            "context_plan",
            "expression_turn",
        )
    )
    basic_passes = bool(
        ordinary.quality_passes(quality, task, source_hash)
        and quality.get("artifact_kind") == QUALITY_ARTIFACT_KIND
        and quality.get("input_representation") == INPUT_REPRESENTATION
        and isinstance(quality.get("frames"), int)
        and quality.get("frames", 0) >= expected_frames
        and quality.get("accepted_for_training") is False
        and quality.get("semantic_supervision_masks") == SEMANTIC_MASKS
        and quality.get("emotion_supervision_mask") is False
        and quality.get("affect_observable_supervision_mask") is False
        and quality.get("official_category_conditioning_enabled") is False
        and quality.get("official_emotion_conditioning_enabled") is False
        and not LEGACY_SEMANTIC_FIELDS.intersection(quality)
        and segment.get("representation") == RETARGET_SEGMENT_REPRESENTATION
        and segment.get("source_frame_count") == expected_frames
        and segment.get("output_frame_count") == quality.get("frames")
        and segment.get("cropped") is False
        and segment.get("duration_policy") == NATURAL_DURATION_POLICY
        and segment.get("retime_policy") == ALGORITHM_NAME
        and segment.get("max_slowdown_ratio") == MAX_SLOWDOWN_RATIO
        and segment.get("fixed_target_duration_sec") is None
        and isinstance(quality.get("safety_monotonic_retime"), dict)
        and provenance_matches
    )
    if not basic_passes:
        quality["expression_turn_output_contract_validation"] = {
            "passed": False,
            "status": "failed_prevalidation",
        }
        return False
    try:
        validate_expression_turn_retarget_output(
            quality,
            input_record=input_record_from_task(task),
            catalog_binding=catalog_binding,
            safe_csv_path=safe_csv_path,
        )
    except (ExpressionTurnContractError, OSError, ValueError) as error:
        quality["expression_turn_output_contract_validation"] = {
            "passed": False,
            "status": "failed_independent_output_contract",
            "error": str(error),
        }
        return False
    quality["expression_turn_output_contract_validation"] = {
        "passed": True,
        "status": "validated_physical_only_training_still_closed",
    }
    return True


def input_record_from_task(task: dict[str, Any]) -> dict[str, Any]:
    """Recover the exact catalog row without retarget transport metadata."""

    return {
        key: value
        for key, value in task.items()
        if key not in TASK_TRANSPORT_FIELDS
    }


def build_run_contract(
    args: argparse.Namespace, catalog_audit: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    contract, _old_hash = grouped.build_run_contract(args)
    contract["artifacts"]["expression_turn_batch_runner"] = ordinary.file_binding(
        Path(__file__)
    )
    contract["artifacts"]["expression_turn_candidate_contract"] = (
        ordinary.file_binding(INPUT_CONTRACT_SCRIPT)
    )
    contract["artifacts"]["expression_turn_retarget_output_contract"] = (
        ordinary.file_binding(OUTPUT_CONTRACT_SCRIPT)
    )
    contract["artifacts"]["expression_turn_safety_monotonic_retime"] = (
        ordinary.file_binding(SAFETY_RETIME_SCRIPT)
    )
    contract["input_contract"] = {
        "artifact_kind": INPUT_ARTIFACT_KIND,
        "representation": INPUT_REPRESENTATION,
        "duration_policy": NATURAL_DURATION_POLICY,
        "context_policy": CONTEXT_POLICY,
        "catalog_binding": catalog_audit,
        "semantic_supervision_masks": dict(SEMANTIC_MASKS),
        "emotion_supervision_mask": False,
        "accepted_for_training": False,
        "legacy_semantic_event_fields_allowed": False,
    }
    contract["grouped_execution_policy"] = dict(EXPRESSION_EXECUTION_POLICY)
    contract["output_artifact_kind"] = OUTPUT_ARTIFACT_KIND
    contract["retarget_segment_representation"] = RETARGET_SEGMENT_REPRESENTATION
    contract["safety_monotonic_retime"] = {
        "algorithm": ALGORITHM_NAME,
        "algorithm_contract_sha256": ALGORITHM_CONTRACT_SHA256,
        "max_slowdown_ratio": MAX_SLOWDOWN_RATIO,
        "crop_tile_or_target_duration_allowed": False,
        "blind_review_uses_retimed_output": True,
    }
    retry_audit = getattr(args, "retry_selection_audit", None)
    if retry_audit is not None:
        contract["retry_selection_audit"] = retry_audit
    return contract, ordinary.json_sha256(contract)


def _publish_turn_result(
    task: dict[str, Any],
    args: argparse.Namespace,
    inventory_hash: str,
    run_contract_hash: str,
    run_id: str,
    started_at: str,
    started: float,
    source_hash: str,
    catalog_binding: dict[str, Any],
    stage_dir: Path,
    quality: dict[str, Any],
    log_path: Path,
) -> dict[str, Any]:
    stage_safe_csv = ordinary.only_safe_csv(stage_dir)
    passed = expression_quality_passes(
        quality,
        task,
        source_hash,
        catalog_binding,
        safe_csv_path=stage_safe_csv,
    )
    category = "passed" if passed else "failed"
    destination = args.output_root / category / task["task_id"]
    ordinary.publish_directory(
        stage_dir, destination, args.output_root / "superseded" / category
    )
    quality_path = destination / "quality.json"
    safe_csv = ordinary.only_safe_csv(destination)
    quality["outputs"] = {
        "raw_csv": str(
            next(iter(sorted(destination.glob("*_gmr_raw_18d.csv"))), "")
        ),
        "safe_csv": str(safe_csv.resolve()),
    }
    ordinary.atomic_json(quality_path, quality)
    common = grouped._event_common(
        task,
        args,
        inventory_hash,
        run_contract_hash,
        run_id,
        started_at,
        started,
        source_hash,
        log_path,
    )
    result = {
        **common,
        "artifact_kind": OUTPUT_ARTIFACT_KIND,
        "returncode": 0,
        "status": "passed" if passed else "quality_failed",
        "accepted_for_training": False,
        "output_dir": str(destination.resolve()),
        "quality_json": str(quality_path.resolve()),
        "quality_json_sha256": ordinary.sha256(quality_path),
        "safe_csv": str(safe_csv.resolve()),
        "safe_csv_sha256": ordinary.sha256(safe_csv),
        "quality_gate": quality.get("quality_gate", {}),
        "frames": quality.get("frames"),
        "duration_sec": quality.get("duration_sec"),
        "retarget_segment": quality.get("retarget_segment"),
    }
    ordinary.atomic_json(ordinary.result_path(args.output_root, task), result)
    return result


def run_source_group(
    tasks: list[dict[str, Any]],
    args: argparse.Namespace,
    inventory_hash: str,
    run_contract_hash: str,
    catalog_binding: dict[str, Any],
    run_id: str,
    *,
    runtime_factory: Callable[[GroupedRuntimeConfig], ExpressionTurnRuntime] = (
        expression_worker_runtime
    ),
) -> list[dict[str, Any]]:
    if not tasks:
        return []
    source = Path(tasks[0]["source"]).resolve()
    if any(Path(task["source"]).resolve() != source for task in tasks):
        raise ValueError("run_source_group received tasks from multiple sources")

    runtime = runtime_factory(grouped.runtime_config(args))
    runtime.load_source(source)
    results: list[dict[str, Any]] = []
    for task in tasks:
        started_at = ordinary.utc_now()
        started = time.perf_counter()
        stage_dir = args.output_root / "staging" / run_id / task["task_id"]
        log_path = grouped._event_log_path(args.output_root, task, run_id)
        try:
            turn = runtime.reset_turn(task)
            quality = runtime.retarget_turn(task, turn, stage_dir)
            ordinary.atomic_json(stage_dir / "quality.json", quality)
            grouped._write_event_log(
                log_path,
                {
                    "status": "expression_turn_retarget_complete",
                    "task_id": task["task_id"],
                    "source_clip_id": task["source_clip_id"],
                    "turn_reset_ordinal": turn.get("event_reset_ordinal"),
                    "quality_passed": (quality.get("quality_gate") or {}).get(
                        "passed"
                    ),
                },
            )
            result = _publish_turn_result(
                task,
                args,
                inventory_hash,
                run_contract_hash,
                run_id,
                started_at,
                started,
                runtime.source_hash,
                catalog_binding,
                stage_dir,
                quality,
                log_path,
            )
        except Exception as error:  # one turn must not poison sibling turns
            result = grouped._record_event_failure(
                task,
                args,
                inventory_hash,
                run_contract_hash,
                run_id,
                started_at,
                started,
                getattr(runtime, "source_hash", None),
                stage_dir,
                log_path,
                error,
            )
        results.append(result)
    return results


def _worker_entry(payload: tuple[Any, ...]) -> list[dict[str, Any]]:
    return run_source_group(*payload)


def completed_pass_is_current(
    result: dict[str, Any],
    task: dict[str, Any],
    run_contract_hash: str,
    catalog_binding: dict[str, Any],
) -> bool:
    if result.get("status") != "passed":
        return False
    if result.get("run_contract_sha256") != run_contract_hash:
        return False
    output_dir = Path(str(result.get("output_dir") or ""))
    quality_path = output_dir / "quality.json"
    try:
        quality = _load_json_object(quality_path)
        safe_csv = ordinary.only_safe_csv(output_dir)
        source_hash = ordinary.sha256(Path(task["source"]))
    except (OSError, ValueError):
        return False
    return bool(
        ordinary.result_lineage_matches(result, task)
        and expression_quality_passes(
            quality,
            task,
            source_hash,
            catalog_binding,
            safe_csv_path=safe_csv,
        )
        and result.get("safe_csv_sha256") == ordinary.sha256(safe_csv)
        and result.get("quality_json_sha256") == ordinary.sha256(quality_path)
    )


def select_runnable_tasks(
    tasks: list[dict[str, Any]],
    args: argparse.Namespace,
    run_contract_hash: str,
    catalog_binding: dict[str, Any],
) -> list[dict[str, Any]]:
    runnable = []
    for task in tasks:
        previous = ordinary.load_result(ordinary.result_path(args.output_root, task))
        if previous is None:
            runnable.append(task)
        elif not ordinary.result_lineage_matches(previous, task):
            runnable.append(task)
        elif completed_pass_is_current(
            previous, task, run_contract_hash, catalog_binding
        ):
            continue
        elif previous.get("status") == "passed" or args.retry_failed:
            runnable.append(task)
    return runnable


RETRY_SELECTION_POLICY = (
    "prior_physical_gates_all_true_and_only_obsolete_native_frame_"
    "prohibition_failed_v1"
)


def select_safety_retime_retry_tasks(
    tasks: list[dict[str, Any]],
    diagnostic_root: Path,
    *,
    inventory_hash: str,
    selection_kind: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Select only the old contract's physically valid retimed outputs."""

    root = diagnostic_root.resolve()
    required_files = {
        name: root / name
        for name in (
            "status.json",
            ordinary.RUN_CONTRACT_FILENAME,
            "passed_manifest.jsonl",
            "failed_manifest.jsonl",
            "pending_manifest.jsonl",
            "excluded_manifest.jsonl",
        )
    }
    if any(not path.is_file() for path in required_files.values()):
        missing = [name for name, path in required_files.items() if not path.is_file()]
        raise ValueError(f"retry diagnostic root is incomplete: {missing}")
    prior_status = _load_json_object(required_files["status.json"])
    prior_contract_wrapper = _load_json_object(
        required_files[ordinary.RUN_CONTRACT_FILENAME]
    )
    prior_contract = prior_contract_wrapper.get("run_contract")
    prior_contract_hash = prior_contract_wrapper.get("run_contract_sha256")
    if (
        not isinstance(prior_contract, dict)
        or not ordinary._is_sha256(prior_contract_hash)
        or ordinary.json_sha256(prior_contract) != prior_contract_hash
        or prior_status.get("run_contract") != prior_contract
        or prior_status.get("run_contract_sha256") != prior_contract_hash
    ):
        raise ValueError("retry diagnostic run contract is invalid")
    if (
        prior_status.get("run_state") != "finished"
        or prior_status.get("inventory_sha256") != inventory_hash
        or prior_status.get("selection_kind") != selection_kind
        or prior_status.get("accepted_for_training") is not False
        or prior_status.get("terminal_turn_count") != len(tasks)
        or prior_status.get("pending_turn_count") != 0
    ):
        raise ValueError("retry diagnostic status is not a complete matching run")
    old_policy = prior_status.get("execution_policy") or {}
    if (
        old_policy.get("preserve_native_frame_count") is not True
        or old_policy.get("fixed_duration_windows_allowed") is not False
        or old_policy.get("natural_training_segment_only") is not True
    ):
        raise ValueError("retry diagnostic is not the obsolete native-frame run")

    selected = []
    excluded = []
    selected_lineage = []
    exclusion_counts: Counter[str] = Counter()
    ratios = []
    for task in tasks:
        result_path = ordinary.result_path(root, task)
        result = ordinary.load_result(result_path)
        if result is None or not ordinary.result_lineage_matches(result, task):
            raise ValueError(
                f"retry diagnostic result is missing or has wrong lineage: {task['task_id']}"
            )
        result_hash = ordinary.sha256(result_path)
        reason = None
        if result.get("status") == "passed":
            reason = "prior_native_frame_output_passed_no_retry"
        elif result.get("status") == "quality_failed":
            output_dir = Path(str(result.get("output_dir") or "")).resolve()
            if root != output_dir and root not in output_dir.parents:
                raise ValueError(
                    f"retry diagnostic output escapes its root: {task['task_id']}"
                )
            quality_path = output_dir / "quality.json"
            quality = _load_json_object(quality_path)
            if result.get("quality_json_sha256") != ordinary.sha256(quality_path):
                raise ValueError(
                    f"retry diagnostic quality hash mismatch: {task['task_id']}"
                )
            gate = quality.get("quality_gate") or {}
            physical_gates_pass = bool(gate) and all(
                value is True for value in gate.values()
            )
            segment = quality.get("retarget_segment") or {}
            expected_frames = int(task["end_frame_exclusive"]) - int(
                task["start_frame"]
            )
            obsolete_frame_only_failure = bool(
                physical_gates_pass
                and quality.get("expression_turn_output_contract_validation")
                == {"passed": False, "status": "failed_prevalidation"}
                and quality.get("source_window_frames") == expected_frames
                and quality.get("frames") == segment.get("output_frame_count")
                and segment.get("source_frame_count") == expected_frames
                and segment.get("output_frame_count", 0) > expected_frames
                and segment.get("retimed") is True
                and segment.get("cropped") is False
                and segment.get("duration_policy") == NATURAL_DURATION_POLICY
            )
            if obsolete_frame_only_failure:
                selected.append(task)
                ratio = float(segment["output_frame_count"] / expected_frames)
                ratios.append(ratio)
                selected_lineage.append(
                    {
                        "task_id": task["task_id"],
                        "prior_result_json_sha256": result_hash,
                        "prior_quality_json_sha256": result[
                            "quality_json_sha256"
                        ],
                        "prior_source_frame_count": expected_frames,
                        "prior_output_frame_count": segment[
                            "output_frame_count"
                        ],
                        "prior_retime_ratio": ratio,
                    }
                )
                continue
            reason = "prior_true_physical_quality_failure_no_retry"
        else:
            raise ValueError(
                f"retry diagnostic has unexpected terminal status: {task['task_id']}"
            )
        exclusion_counts[str(reason)] += 1
        excluded.append(
            {
                "artifact_kind": "ula_v2_18d_expression_turn_retry_exclusion_v1",
                "task_id": task["task_id"],
                "clip_id": task["clip_id"],
                "source_clip_id": task["source_clip_id"],
                "status": "excluded",
                "exclusion_reason": reason,
                "prior_result_json": str(result_path),
                "prior_result_json_sha256": result_hash,
                "accepted_for_training": False,
            }
        )
    if not selected or len(selected) + len(excluded) != len(tasks):
        raise ValueError("retry diagnostic classification is incomplete")
    audit_payload = {
        "artifact_kind": "ula_v2_18d_expression_turn_safety_retime_retry_selection_v1",
        "policy": RETRY_SELECTION_POLICY,
        "diagnostic_root": str(root),
        "diagnostic_files_sha256": {
            name: ordinary.sha256(path)
            for name, path in sorted(required_files.items())
        },
        "prior_run_contract_sha256": prior_contract_hash,
        "inventory_sha256": inventory_hash,
        "selection_kind": selection_kind,
        "input_task_count": len(tasks),
        "selected_retry_count": len(selected),
        "excluded_count": len(excluded),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "prior_retime_ratio_min": min(ratios),
        "prior_retime_ratio_max": max(ratios),
        "selected_lineage": selected_lineage,
        "accepted_for_training": False,
    }
    return selected, excluded, {
        **audit_payload,
        "sha256": ordinary.json_sha256(audit_payload),
    }


def status_payload(
    args: argparse.Namespace,
    inventory_hash: str,
    eligible: list[dict[str, Any]],
    selected_groups: list[list[dict[str, Any]]],
    results: list[dict[str, Any]],
    pending: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    run_id: str,
    run_state: str,
    started_at: str,
    run_contract: dict[str, Any],
    run_contract_hash: str,
) -> dict[str, Any]:
    selected = [task for group in selected_groups for task in group]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "ula_v2_18d_expression_turn_batch_status_v1",
        "run_id": run_id,
        "run_state": run_state,
        "started_at": started_at,
        "updated_at": ordinary.utc_now(),
        "inventory": str(args.inventory.resolve()),
        "inventory_sha256": inventory_hash,
        "catalog_summary": str(args.catalog_summary.resolve()),
        "catalog_summary_sha256": ordinary.sha256(args.catalog_summary),
        "selection_kind": args.selection_kind,
        "run_contract": run_contract,
        "run_contract_sha256": run_contract_hash,
        "execution_policy": dict(EXPRESSION_EXECUTION_POLICY),
        "eligible_turn_count": len(eligible),
        "eligible_source_count": len(grouped.group_tasks_by_source(eligible)),
        "selected_turn_count": len(selected),
        "selected_source_count": len(selected_groups),
        "terminal_turn_count": len(results),
        "pending_turn_count": len(pending),
        "excluded_turn_count": len(excluded),
        "inventory_turn_count": len(eligible) + len(excluded),
        "counts": dict(Counter(row.get("status") for row in results)),
        "coverage_complete": len(results) + len(pending) == len(eligible),
        "accepted_for_training": False,
        "retry_selection_audit": getattr(args, "retry_selection_audit", None),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    for name in ("limit_sources", "limit_turns"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.retry_failed and not args.resume:
        raise ValueError("--retry-failed requires --resume")
    if args.retry_from_diagnostic_root is not None and not args.retry_failed:
        raise ValueError(
            "--retry-from-diagnostic-root requires --retry-failed"
        )
    for name in (
        "inventory",
        "catalog_summary",
        "beat2_root",
        "smplx_model",
        "gmr_root",
        "urdf",
        "config",
    ):
        path = getattr(args, name).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        setattr(args, name, path)
    args.output_root = args.output_root.resolve()
    if args.retry_from_diagnostic_root is not None:
        args.retry_from_diagnostic_root = args.retry_from_diagnostic_root.resolve()
        if not args.retry_from_diagnostic_root.is_dir():
            raise FileNotFoundError(args.retry_from_diagnostic_root)

    catalog_binding, catalog_audit = load_catalog_binding(
        args.inventory, args.catalog_summary, args.selection_kind
    )
    # The validator receives only its public binding fields.
    validator_binding = {
        key: catalog_binding[key]
        for key in (
            "retarget_input_manifest_sha256",
            "expression_turn_contract_sha256",
            "source_inventory_manifest_sha256",
            "split_assignment_manifest_sha256",
            "selection_kind",
            "require_selection_record",
        )
    }
    inventory_hash = ordinary.sha256(args.inventory)
    all_eligible, reader_excluded = read_expression_turn_inventory(
        args.inventory,
        args.beat2_root,
        validator_binding,
        str(catalog_audit["catalog_candidate_manifest_sha256"]),
    )
    if reader_excluded:
        raise RuntimeError("v8 candidate reader must fail closed, not exclude records")
    excluded: list[dict[str, Any]] = []
    eligible = all_eligible
    args.retry_selection_audit = None
    if args.retry_from_diagnostic_root is not None:
        eligible, excluded, args.retry_selection_audit = (
            select_safety_retime_retry_tasks(
                all_eligible,
                args.retry_from_diagnostic_root,
                inventory_hash=inventory_hash,
                selection_kind=args.selection_kind,
            )
        )
    all_groups = grouped.group_tasks_by_source(eligible)
    selected_groups = grouped.limit_groups(
        all_groups,
        limit_sources=args.limit_sources,
        limit_events=args.limit_turns,
    )

    run_contract, run_contract_hash = build_run_contract(args, catalog_audit)
    status_path = args.output_root / "status.json"
    if status_path.exists():
        if not args.resume:
            raise RuntimeError(f"Existing batch state requires --resume: {status_path}")
        ordinary.validate_resume_contract(
            status_path,
            args.output_root,
            inventory_hash,
            run_contract,
            run_contract_hash,
        )
    elif (args.output_root / ordinary.RUN_CONTRACT_FILENAME).exists() or any(
        (args.output_root / "state/results").glob("*.json")
    ):
        raise RuntimeError(
            "Retarget output contains state without status.json; refusing unsafe reuse"
        )
    if status_path.exists():
        ordinary.validate_saved_result_contracts(
            args.output_root, eligible, run_contract_hash
        )

    selected_tasks = [task for group in selected_groups for task in group]
    runnable = select_runnable_tasks(
        selected_tasks, args, run_contract_hash, validator_binding
    )
    runnable_ids = {task["task_id"] for task in runnable}
    runnable_groups = [
        [task for task in group if task["task_id"] in runnable_ids]
        for group in selected_groups
    ]
    runnable_groups = [group for group in runnable_groups if group]

    args.output_root.mkdir(parents=True, exist_ok=True)
    contract_path = args.output_root / ordinary.RUN_CONTRACT_FILENAME
    if not contract_path.exists():
        ordinary.atomic_json(
            contract_path,
            {
                "run_contract_sha256": run_contract_hash,
                "run_contract": run_contract,
            },
        )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    started_at = ordinary.utc_now()
    results, pending = ordinary.write_manifests(
        args.output_root,
        eligible,
        excluded,
        args.inventory,
        inventory_hash,
        run_contract_hash,
    )
    ordinary.atomic_json(
        status_path,
        status_payload(
            args,
            inventory_hash,
            eligible,
            selected_groups,
            results,
            pending,
            excluded,
            run_id,
            "running",
            started_at,
            run_contract,
            run_contract_hash,
        ),
    )

    try:
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
                    args.output_root,
                    eligible,
                    excluded,
                    args.inventory,
                    inventory_hash,
                    run_contract_hash,
                )
                ordinary.atomic_json(
                    status_path,
                    status_payload(
                        args,
                        inventory_hash,
                        eligible,
                        selected_groups,
                        results,
                        pending,
                        excluded,
                        run_id,
                        "running",
                        started_at,
                        run_contract,
                        run_contract_hash,
                    ),
                )
                source_id = group[0]["source_clip_id"]
                counts = Counter(row["status"] for row in group_results)
                print(
                    f"[{index:04d}/{len(runnable_groups):04d}] "
                    f"{source_id}: {dict(counts)}",
                    flush=True,
                )
    except KeyboardInterrupt:
        results, pending = ordinary.write_manifests(
            args.output_root,
            eligible,
            excluded,
            args.inventory,
            inventory_hash,
            run_contract_hash,
        )
        ordinary.atomic_json(
            status_path,
            status_payload(
                args,
                inventory_hash,
                eligible,
                selected_groups,
                results,
                pending,
                excluded,
                run_id,
                "interrupted_resumable",
                started_at,
                run_contract,
                run_contract_hash,
            ),
        )
        return 130
    except Exception:
        traceback.print_exc()
        raise

    results, pending = ordinary.write_manifests(
        args.output_root,
        eligible,
        excluded,
        args.inventory,
        inventory_hash,
        run_contract_hash,
    )
    ordinary.atomic_json(
        status_path,
        status_payload(
            args,
            inventory_hash,
            eligible,
            selected_groups,
            results,
            pending,
            excluded,
            run_id,
            "finished",
            started_at,
            run_contract,
            run_contract_hash,
        ),
    )
    print(json.dumps(Counter(row["status"] for row in results), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
