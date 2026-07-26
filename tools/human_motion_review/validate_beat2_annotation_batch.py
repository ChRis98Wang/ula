#!/usr/bin/env python3
"""Fail-closed validation for a BEAT2 18D annotation batch.

The validator joins every artifact by the zero-based half-open window task ID
``{clip_id}_f{start:06d}-{end:06d}``.  Legacy records containing only the base
clip ID are accepted only when that clip maps to exactly one inventory window.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0.0"
ROBOT_CONTRACT = "ula_v2_18d_head_v1"
ACTION_DIM = 18
EXPECTED_ELIGIBLE_COUNT = 280
ELIGIBLE_SELECTION_STATUSES = {
    "full_nonoverlap_boundary_validated",
    "selected_nonstatic_low_dynamic_with_aligned_speech",
}
NON_BLOCKING_INVENTORY_WARNINGS = {
    "motion_audio_duration_mismatch_gt_0_3s",
    "textgrid_transcript_mismatch",
}
JOINT_ORDER = [
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
REQUIRED_QUALITY_GATES = {
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
}


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    payload = "".join(stable_json(record) + "\n" for record in records)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            records.append(record)
    return records


def _task_id(clip_id: str, start: int, end: int) -> str:
    return f"{clip_id}_f{start:06d}-{end:06d}"


def inventory_index(records: list[dict[str, Any]]) -> tuple[dict[str, dict], set[str], set[str]]:
    tasks: dict[str, dict] = {}
    eligible: set[str] = set()
    excluded: set[str] = set()
    clip_ids: set[str] = set()
    for index, record in enumerate(records):
        clip_id = record.get("clip_id")
        window = record.get("window") or {}
        start = window.get("start_frame")
        end = window.get("end_frame_exclusive")
        if not isinstance(clip_id, str) or not clip_id:
            raise ValueError(f"inventory record {index} has no clip_id")
        if clip_id in clip_ids:
            raise ValueError(f"duplicate inventory clip_id: {clip_id}")
        clip_ids.add(clip_id)
        if isinstance(start, bool) or not isinstance(start, int):
            raise ValueError(f"{clip_id}: invalid window.start_frame")
        if isinstance(end, bool) or not isinstance(end, int) or end <= start:
            raise ValueError(f"{clip_id}: invalid window.end_frame_exclusive")
        declared_task_id = record.get("task_id")
        source_clip_id = record.get("source_clip_id")
        if declared_task_id is not None:
            if not isinstance(declared_task_id, str) or not declared_task_id:
                raise ValueError(f"{clip_id}: invalid task_id")
            if not isinstance(source_clip_id, str) or not source_clip_id:
                raise ValueError(f"{clip_id}: explicit task_id requires source_clip_id")
            expected_task_id = _task_id(source_clip_id, start, end)
            if declared_task_id != expected_task_id or clip_id != declared_task_id:
                raise ValueError(
                    f"{clip_id}: task_id is not bound to source_clip_id and window"
                )
            task_id = declared_task_id
        else:
            task_id = _task_id(clip_id, start, end)
        task = {
            "task_id": task_id,
            "clip_id": clip_id,
            "start_frame": start,
            "end_frame_exclusive": end,
            "fps": float(record.get("fps", 0.0)),
            "official_split": record.get("official_split"),
            "speaker_key": record.get("speaker_key"),
        }
        if task_id in tasks:
            raise ValueError(f"duplicate inventory task_id: {task_id}")
        tasks[task_id] = task
        issues = set(record.get("issues") or [])
        blocking = issues - NON_BLOCKING_INVENTORY_WARNINGS
        if (
            window.get("selection_status") in ELIGIBLE_SELECTION_STATUSES
            and not blocking
        ):
            eligible.add(task_id)
        else:
            excluded.add(task_id)
    return tasks, eligible, excluded


def _clip_to_tasks(tasks: dict[str, dict]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for task_id, task in tasks.items():
        result[task["clip_id"]].append(task_id)
    return result


def canonical_task_id(
    record: dict[str, Any],
    tasks: dict[str, dict],
    clip_to_tasks: dict[str, list[str]],
) -> tuple[str | None, str | None]:
    """Map current and legacy record IDs to one inventory task ID."""
    for key in ("task_id", "sample_id", "record_id", "clip_id"):
        value = record.get(key)
        if not isinstance(value, str) or not value:
            continue
        if value in tasks:
            return value, None
        candidates = clip_to_tasks.get(value, [])
        if len(candidates) == 1:
            return candidates[0], f"legacy_base_clip_id:{key}"
        if len(candidates) > 1:
            return None, f"ambiguous base clip ID {value!r} in {key}"
    return None, "record has no inventory task ID"


def index_manifest(
    name: str,
    records: list[dict[str, Any]],
    tasks: dict[str, dict],
    clip_to_tasks: dict[str, list[str]],
    errors: list[str],
) -> tuple[dict[str, dict], int]:
    indexed: dict[str, dict] = {}
    legacy_mapped = 0
    for index, record in enumerate(records):
        task_id, adaptation = canonical_task_id(record, tasks, clip_to_tasks)
        if task_id is None:
            errors.append(f"{name}[{index}]: {adaptation}")
            continue
        if task_id in indexed:
            errors.append(f"{name}: duplicate task_id {task_id}")
            continue
        if adaptation:
            legacy_mapped += 1
        indexed[task_id] = record
    return indexed, legacy_mapped


def _field(record: dict[str, Any], key: str) -> Any:
    if key in record:
        return record[key]
    source = record.get("source")
    if isinstance(source, dict):
        return source.get(key)
    return None


def _has_field(record: dict[str, Any], key: str) -> bool:
    if key in record:
        return True
    source = record.get("source")
    return isinstance(source, dict) and key in source


def validate_preserved_metadata(
    name: str,
    indexed: dict[str, dict],
    tasks: dict[str, dict],
    errors: list[str],
    *,
    require_window: bool,
) -> None:
    for task_id, record in indexed.items():
        expected = tasks[task_id]
        for key in ("official_split", "speaker_key"):
            if not _has_field(record, key) or _field(record, key) != expected[key]:
                errors.append(f"{name}:{task_id}: {key} was not preserved")
        declared_clip = record.get("clip_id") or _field(record, "source_clip_id")
        if declared_clip is not None and declared_clip != expected["clip_id"]:
            errors.append(f"{name}:{task_id}: clip_id does not match inventory")
        if require_window:
            if record.get("start_frame") != expected["start_frame"]:
                errors.append(f"{name}:{task_id}: start_frame does not match inventory")
            if record.get("end_frame_exclusive") != expected["end_frame_exclusive"]:
                errors.append(f"{name}:{task_id}: end_frame_exclusive does not match inventory")
            try:
                fps = float(record.get("fps"))
            except (TypeError, ValueError):
                fps = math.nan
            if not math.isclose(fps, expected["fps"], rel_tol=0.0, abs_tol=1e-9):
                errors.append(f"{name}:{task_id}: fps does not match inventory")


def validate_state_statuses(indexed: dict[str, dict[str, dict]], errors: list[str]) -> None:
    allowed_failures = {
        "process_failed",
        "invalid_quality_report",
        "quality_failed",
        "worker_failed",
    }
    for task_id, record in indexed["failed"].items():
        if record.get("status") not in allowed_failures:
            errors.append(f"failed:{task_id}: unsupported failure status")
    for task_id, record in indexed["pending"].items():
        if record.get("status") != "pending":
            errors.append(f"pending:{task_id}: status must be pending")
    for task_id, record in indexed["excluded"].items():
        if record.get("status") != "excluded":
            errors.append(f"excluded:{task_id}: status must be excluded")


def _resolve(path_value: Any, manifest_path: Path) -> Path | None:
    if not isinstance(path_value, str) or not path_value:
        return None
    path = Path(path_value)
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def _validate_safe_csv(path: Path, expected_rows: int | None) -> list[str]:
    reasons: list[str] = []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header != JOINT_ORDER:
                reasons.append("trajectory_joint_order_mismatch")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error):
        return ["trajectory_unreadable"]
    if expected_rows is not None and len(rows) != expected_rows:
        reasons.append("trajectory_row_count_mismatch")
    for row in rows:
        if len(row) != ACTION_DIM:
            reasons.append("trajectory_row_width_mismatch")
            break
        try:
            values = [float(value) for value in row]
        except ValueError:
            reasons.append("trajectory_non_numeric")
            break
        if not all(math.isfinite(value) for value in values):
            reasons.append("trajectory_non_finite")
            break
    return sorted(set(reasons))


def validate_passed_evidence(
    indexed: dict[str, dict],
    tasks: dict[str, dict],
    manifest_path: Path,
    errors: list[str],
) -> None:
    for task_id, record in indexed.items():
        prefix = f"passed:{task_id}"
        if record.get("status") != "passed":
            errors.append(f"{prefix}: status must be passed")
        if record.get("retarget_contract") != ROBOT_CONTRACT:
            errors.append(f"{prefix}: retarget_contract must be {ROBOT_CONTRACT}")
        if record.get("accepted_for_training") is not False:
            errors.append(f"{prefix}: retarget pass cannot grant training admission")

        safe_csv = _resolve(record.get("safe_csv"), manifest_path)
        quality_json = _resolve(record.get("quality_json"), manifest_path)
        if safe_csv is None or not safe_csv.is_file():
            errors.append(f"{prefix}: safe_csv is missing")
            continue
        if quality_json is None or not quality_json.is_file():
            errors.append(f"{prefix}: quality_json is missing")
            continue
        safe_hash = record.get("safe_csv_sha256")
        quality_hash = record.get("quality_json_sha256")
        actual_safe_hash = sha256(safe_csv)
        actual_quality_hash = sha256(quality_json)
        if safe_hash != actual_safe_hash:
            errors.append(f"{prefix}: safe_csv_sha256 mismatch")
        if quality_hash != actual_quality_hash:
            errors.append(f"{prefix}: quality_json_sha256 mismatch")
        try:
            quality = json.loads(quality_json.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            errors.append(f"{prefix}: quality_json is invalid")
            continue
        if not isinstance(quality, dict):
            errors.append(f"{prefix}: quality_json must contain an object")
            continue
        if quality.get("output_contract") != ROBOT_CONTRACT:
            errors.append(f"{prefix}: quality output_contract mismatch")
        if quality.get("action_dim") != ACTION_DIM:
            errors.append(f"{prefix}: quality action_dim mismatch")
        if quality.get("joint_order") != JOINT_ORDER:
            errors.append(f"{prefix}: quality joint_order mismatch")
        expected = tasks[task_id]
        if quality.get("source_window_start_frame") != expected["start_frame"]:
            errors.append(f"{prefix}: quality source window start mismatch")
        if quality.get("source_window_end_frame_exclusive") != expected["end_frame_exclusive"]:
            errors.append(f"{prefix}: quality source window end mismatch")
        try:
            quality_fps = float(quality.get("fps"))
        except (TypeError, ValueError):
            quality_fps = math.nan
        if not math.isclose(quality_fps, expected["fps"], rel_tol=0.0, abs_tol=1e-9):
            errors.append(f"{prefix}: quality fps mismatch")
        gates = quality.get("quality_gate")
        if not isinstance(gates, dict):
            errors.append(f"{prefix}: quality_gate is missing")
        else:
            missing = sorted(REQUIRED_QUALITY_GATES - set(gates))
            if missing:
                errors.append(f"{prefix}: missing quality gates {missing}")
            failed = sorted(key for key in REQUIRED_QUALITY_GATES if gates.get(key) is not True)
            if failed:
                errors.append(f"{prefix}: quality gates did not pass {failed}")
        frames = quality.get("frames")
        expected_rows = frames if isinstance(frames, int) and not isinstance(frames, bool) else None
        if expected_rows is None or expected_rows <= 0:
            errors.append(f"{prefix}: quality frames is invalid")
            expected_rows = None
        for reason in _validate_safe_csv(safe_csv, expected_rows):
            errors.append(f"{prefix}: {reason}")
        outputs_safe = _resolve((quality.get("outputs") or {}).get("safe_csv"), quality_json)
        if outputs_safe != safe_csv:
            errors.append(f"{prefix}: quality outputs.safe_csv does not match manifest")


def validate_annotations(
    indexed: dict[str, dict],
    passed: dict[str, dict],
    tasks: dict[str, dict],
    annotation_manifest: Path,
    passed_manifest: Path,
    errors: list[str],
) -> None:
    for task_id, annotation in indexed.items():
        prefix = f"annotations:{task_id}"
        if task_id not in passed:
            errors.append(f"{prefix}: only retarget-passed tasks may be annotated")
            continue
        if annotation.get("accepted_for_training") is not False:
            errors.append(f"{prefix}: accepted_for_training must be false")
        manual_fields = [
            annotation[key]
            for key in ("manual_review_required", "manual_human_review_required")
            if key in annotation
        ]
        if not manual_fields or any(value is not True for value in manual_fields):
            errors.append(
                f"{prefix}: manual_review_required or "
                "manual_human_review_required must be true"
            )
        if annotation.get("robot_contract") != ROBOT_CONTRACT:
            errors.append(f"{prefix}: robot_contract must be {ROBOT_CONTRACT}")
        prompt = annotation.get("canonical_prompt")
        if not isinstance(prompt, dict) or not all(
            isinstance(prompt.get(language), str) and prompt[language].strip()
            for language in ("en", "zh")
        ):
            errors.append(f"{prefix}: canonical_prompt must contain non-empty en and zh")

        passed_record = passed[task_id]
        annotation_path = _resolve(annotation.get("trajectory_path"), annotation_manifest)
        passed_path = _resolve(passed_record.get("safe_csv"), passed_manifest)
        if annotation_path != passed_path:
            errors.append(f"{prefix}: trajectory_path does not match retarget evidence")
        if annotation.get("trajectory_sha256") != passed_record.get("safe_csv_sha256"):
            errors.append(f"{prefix}: trajectory_sha256 does not match retarget evidence")
        annotation_quality = _resolve(
            annotation.get("quality_json"), annotation_manifest
        )
        passed_quality = _resolve(passed_record.get("quality_json"), passed_manifest)
        if annotation_quality != passed_quality:
            errors.append(f"{prefix}: quality_json does not match retarget evidence")
        if annotation.get("quality_json_sha256") != passed_record.get(
            "quality_json_sha256"
        ):
            errors.append(
                f"{prefix}: quality_json_sha256 does not match retarget evidence"
            )

        expected = tasks[task_id]
        for key in ("official_split", "speaker_key"):
            if not _has_field(annotation, key) or _field(annotation, key) != expected[key]:
                errors.append(f"{prefix}: {key} was not preserved")


def _set_difference_summary(expected: set[str], actual: set[str]) -> dict[str, Any]:
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    return {
        "matches": not missing and not unexpected,
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "missing_task_ids": missing,
        "unexpected_task_ids": unexpected,
    }


def _distribution(task_ids: set[str], tasks: dict[str, dict], key: str) -> dict[str, int]:
    counts = Counter(str(tasks[task_id].get(key) or "<missing>") for task_id in task_ids)
    return dict(sorted(counts.items()))


def build_review_queue(
    annotations: dict[str, dict], passed: dict[str, dict], tasks: dict[str, dict]
) -> list[dict[str, Any]]:
    queue = []
    for task_id in sorted(
        annotations, key=lambda value: (str(tasks[value]["speaker_key"]), value)
    ):
        annotation = annotations[task_id]
        retarget = passed[task_id]
        task = tasks[task_id]
        queue.append(
            {
                "schema_version": SCHEMA_VERSION,
                "task_id": task_id,
                "source_clip_id": task["clip_id"],
                "speaker_key": task["speaker_key"],
                "official_split": task["official_split"],
                "robot_contract": ROBOT_CONTRACT,
                "canonical_action": annotation.get("canonical_action"),
                "canonical_prompt": annotation.get("canonical_prompt"),
                "observable_features": annotation.get("observable_features"),
                "semantic_confidence": annotation.get("semantic_confidence"),
                "review_flags": annotation.get("review_flags", []),
                "prompt_provenance": annotation.get("prompt_provenance"),
                "speech_context_included": False,
                "trajectory_path": retarget["safe_csv"],
                "trajectory_sha256": retarget["safe_csv_sha256"],
                "quality_json": retarget["quality_json"],
                "quality_json_sha256": retarget["quality_json_sha256"],
                "review_state": "pending_independent_motion_text_review",
                "manual_review_required": True,
                "accepted_for_training": False,
            }
        )
    return queue


def validate_batch(
    *,
    inventory_path: Path,
    passed_path: Path,
    failed_path: Path,
    pending_path: Path,
    excluded_path: Path,
    annotations_path: Path,
    expected_eligible_count: int | None = EXPECTED_ELIGIBLE_COUNT,
    review_queue_output: Path | None = None,
) -> dict[str, Any]:
    paths = {
        "inventory": inventory_path.resolve(),
        "passed": passed_path.resolve(),
        "failed": failed_path.resolve(),
        "pending": pending_path.resolve(),
        "excluded": excluded_path.resolve(),
        "annotations": annotations_path.resolve(),
    }
    inventory_records = read_jsonl(paths["inventory"])
    tasks, eligible_ids, expected_excluded_ids = inventory_index(inventory_records)
    clip_to_tasks = _clip_to_tasks(tasks)
    errors: list[str] = []
    if expected_eligible_count is not None and len(eligible_ids) != expected_eligible_count:
        errors.append(
            f"inventory eligible count {len(eligible_ids)} != expected {expected_eligible_count}"
        )

    indexed: dict[str, dict[str, dict]] = {}
    legacy_counts: dict[str, int] = {}
    for name in ("passed", "failed", "pending", "excluded", "annotations"):
        records = read_jsonl(paths[name])
        indexed[name], legacy_counts[name] = index_manifest(
            name, records, tasks, clip_to_tasks, errors
        )

    state_names = ("passed", "failed", "pending", "excluded")
    for left_index, left in enumerate(state_names):
        for right in state_names[left_index + 1 :]:
            overlap = sorted(set(indexed[left]) & set(indexed[right]))
            if overlap:
                errors.append(f"state manifests {left}/{right} overlap: {overlap}")

    eligible_state_ids = set().union(
        set(indexed["passed"]), set(indexed["failed"]), set(indexed["pending"])
    )
    eligible_closure = _set_difference_summary(eligible_ids, eligible_state_ids)
    if not eligible_closure["matches"]:
        errors.append("eligible inventory is not exactly passed + failed + pending")
    excluded_closure = _set_difference_summary(
        expected_excluded_ids, set(indexed["excluded"])
    )
    if not excluded_closure["matches"]:
        errors.append("excluded manifest does not exactly match ineligible inventory")

    for name in state_names:
        validate_preserved_metadata(
            name,
            indexed[name],
            tasks,
            errors,
            require_window=name != "excluded",
        )
    validate_state_statuses(indexed, errors)
    validate_passed_evidence(indexed["passed"], tasks, paths["passed"], errors)
    validate_annotations(
        indexed["annotations"],
        indexed["passed"],
        tasks,
        paths["annotations"],
        paths["passed"],
        errors,
    )
    annotation_closure = _set_difference_summary(
        set(indexed["passed"]), set(indexed["annotations"])
    )
    if not annotation_closure["matches"]:
        errors.append("annotations must cover exactly the retarget-passed tasks")

    status_counts = {name: len(indexed[name]) for name in state_names}
    input_hashes = {name: sha256(path) for name, path in paths.items()}
    summary = {
        "schema_version": SCHEMA_VERSION,
        "validator": "validate_beat2_annotation_batch",
        "valid": not errors,
        "pipeline_ready": not errors,
        "error_count": len(errors),
        "errors": errors,
        "contract": {
            "robot_contract": ROBOT_CONTRACT,
            "action_dim": ACTION_DIM,
            "joint_order": JOINT_ORDER,
            "task_id_convention": (
                "inventory.task_id when bound to source_clip_id+window; otherwise "
                "{clip_id}_f{start:06d}-{end:06d}"
            ),
        },
        "counts": {
            "inventory": len(tasks),
            "eligible": len(eligible_ids),
            "inventory_excluded": len(expected_excluded_ids),
            **status_counts,
            "annotations": len(indexed["annotations"]),
            "train_ready": 0,
        },
        "distributions": {
            "eligible_by_official_split": _distribution(
                eligible_ids, tasks, "official_split"
            ),
            "eligible_by_speaker_key": _distribution(
                eligible_ids, tasks, "speaker_key"
            ),
            "passed_by_official_split": _distribution(
                set(indexed["passed"]), tasks, "official_split"
            ),
            "passed_by_speaker_key": _distribution(
                set(indexed["passed"]), tasks, "speaker_key"
            ),
            "annotations_by_official_split": _distribution(
                set(indexed["annotations"]), tasks, "official_split"
            ),
            "annotations_by_speaker_key": _distribution(
                set(indexed["annotations"]), tasks, "speaker_key"
            ),
        },
        "closure": {
            "eligible_equals_passed_failed_pending": eligible_closure,
            "excluded_equals_inventory_ineligible": excluded_closure,
            "annotations_equal_passed": annotation_closure,
            "state_manifests_pairwise_disjoint": not any(
                set(indexed[left]) & set(indexed[right])
                for i, left in enumerate(state_names)
                for right in state_names[i + 1 :]
            ),
        },
        "legacy_base_clip_id_adaptations": legacy_counts,
        "inputs": {
            name: {"path": str(path), "sha256": input_hashes[name]}
            for name, path in paths.items()
        },
        "policies": {
            "eligible_selection_statuses": sorted(ELIGIBLE_SELECTION_STATUSES),
            "non_blocking_inventory_warnings": sorted(
                NON_BLOCKING_INVENTORY_WARNINGS
            ),
            "retarget_pass_grants_training_admission": False,
            "annotations_require_manual_human_review": True,
            "speech_context_is_action_label": False,
            "train_ready_requires_later_independent_review": True,
        },
    }
    if review_queue_output is not None:
        review_queue_output = review_queue_output.resolve()
        summary["review_queue"] = {
            "path": str(review_queue_output),
            "generated": False,
            "records": 0,
            "sha256": None,
            "order": "speaker_key_then_task_id",
        }
        if summary["valid"]:
            queue = build_review_queue(indexed["annotations"], indexed["passed"], tasks)
            atomic_jsonl(review_queue_output, queue)
            summary["review_queue"].update(
                generated=True,
                records=len(queue),
                sha256=sha256(review_queue_output),
            )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--passed", type=Path, required=True)
    parser.add_argument("--failed", type=Path, required=True)
    parser.add_argument("--pending", type=Path, required=True)
    parser.add_argument("--excluded", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--review-queue-output", type=Path)
    parser.add_argument(
        "--expected-eligible-count",
        type=int,
        default=EXPECTED_ELIGIBLE_COUNT,
        help="Use zero to disable the expected-count assertion.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = validate_batch(
            inventory_path=args.inventory,
            passed_path=args.passed,
            failed_path=args.failed,
            pending_path=args.pending,
            excluded_path=args.excluded,
            annotations_path=args.annotations,
            expected_eligible_count=(
                None if args.expected_eligible_count == 0 else args.expected_eligible_count
            ),
            review_queue_output=args.review_queue_output,
        )
    except (FileNotFoundError, OSError, UnicodeError, ValueError) as error:
        summary = {
            "schema_version": SCHEMA_VERSION,
            "validator": "validate_beat2_annotation_batch",
            "valid": False,
            "pipeline_ready": False,
            "error_count": 1,
            "errors": [f"fatal_input_error: {error}"],
        }
    if args.output is not None:
        atomic_json(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
