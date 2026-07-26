#!/usr/bin/env python3
"""Build internal MuJoCo queues for v8 expression-turn physical QC passes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from tools.human_motion_review.expression_turn_contract import REPRESENTATION
from tools.human_motion_review.expression_turn_retarget_contract import (
    RETARGET_SEGMENT_REPRESENTATION,
)
from upper_body_skeleton.retarget_v2_18d import JOINT_ORDER_18D


ROBOT_CONTRACT = "ula_v2_18d_head_v1"
INPUT_REPRESENTATION = REPRESENTATION
OUTPUT_ARTIFACT_KIND = "ula_v2_18d_expression_turn_retarget_v1"
NATURAL_DURATION_POLICY = "natural_rest_to_natural_rest_no_fixed_or_max_duration"
LEGACY_NATIVE_NOOP_REPRESENTATION = (
    "native_variable_length_expression_turn_retimed_30hz_v1"
)
ALLOWED_RETARGET_REPRESENTATIONS = {
    LEGACY_NATIVE_NOOP_REPRESENTATION,
    RETARGET_SEGMENT_REPRESENTATION,
}
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
SELECTION_KINDS = {"representative100", "stress100"}
ANONYMOUS_PROMPT = {
    "en": "Anonymous silent robot motion sample.",
    "zh": "匿名无声机器人动作样本。",
}


def stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    atomic_text(path, "".join(stable_json(record) + "\n" for record in records))


def atomic_json(path: Path, value: object) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
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
            records.append(value)
    return records


def _verified_file(record: dict[str, Any], path_key: str, hash_key: str) -> Path:
    task_id = record.get("task_id")
    value = record.get(path_key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{task_id}: missing {path_key}")
    path = Path(value).resolve()
    if not path.is_file() or sha256(path) != record.get(hash_key):
        raise ValueError(f"{task_id}: {path_key}/{hash_key} evidence mismatch")
    return path


def _trajectory_frame_count(path: Path, *, task_id: str) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"{task_id}: safe_csv is empty") from error
        if header != list(JOINT_ORDER_18D):
            raise ValueError(f"{task_id}: safe_csv is not ordered ULA V2 18D")
        frames = sum(1 for row in reader if row)
    if frames < 1:
        raise ValueError(f"{task_id}: safe_csv contains no frames")
    return frames


def _validate_final_output_contract(
    *,
    task_id: str,
    record: dict[str, Any],
    training: dict[str, Any],
    retarget: dict[str, Any],
    quality: dict[str, Any],
    trajectory: Path,
) -> tuple[int, str, dict[str, Any] | None]:
    source_frames = training.get("frame_count")
    output_frames = retarget.get("output_frame_count")
    fps = record.get("fps")
    if (
        isinstance(source_frames, bool)
        or not isinstance(source_frames, int)
        or source_frames < 1
        or isinstance(output_frames, bool)
        or not isinstance(output_frames, int)
        or output_frames < source_frames
        or isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isclose(float(fps), 30.0, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise ValueError(f"{task_id}: invalid source/output frame contract")
    if retarget.get("source_frame_count") != source_frames:
        raise ValueError(f"{task_id}: retarget source coverage mismatch")
    if record.get("frames") != output_frames:
        raise ValueError(f"{task_id}: result frames do not bind retarget output")
    actual_frames = _trajectory_frame_count(trajectory, task_id=task_id)
    if actual_frames != output_frames:
        raise ValueError(f"{task_id}: safe_csv rows do not match retarget output")

    expected_durations = {
        "source_frame_coverage_sec": source_frames / float(fps),
        "output_frame_coverage_sec": output_frames / float(fps),
        "output_sample_span_sec": max(0, output_frames - 1) / float(fps),
    }
    for field, expected in expected_durations.items():
        value = retarget.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-9)
        ):
            raise ValueError(f"{task_id}: retarget {field} is invalid")
    if quality.get("retarget_segment") != retarget:
        raise ValueError(f"{task_id}: quality changed the retarget segment")

    representation = retarget.get("representation")
    if representation not in ALLOWED_RETARGET_REPRESENTATIONS:
        raise ValueError(f"{task_id}: unsupported final retarget representation")
    if representation == LEGACY_NATIVE_NOOP_REPRESENTATION:
        if (
            output_frames != source_frames
            or retarget.get("retimed") is not False
            or quality.get("safety_monotonic_retime") is not None
        ):
            raise ValueError(f"{task_id}: legacy native output is not an identity timeline")
        return output_frames, "native_identity_timeline_no_slowdown_required", None

    safety = quality.get("safety_monotonic_retime")
    if not isinstance(safety, dict):
        raise ValueError(f"{task_id}: missing safety monotonic retime audit")
    exact = {
        "artifact_kind": "ula_18d_safety_monotonic_retime_v1",
        "blind_review_must_use_retimed_output": True,
        "source_frame_count": source_frames,
        "output_frame_count": output_frames,
        "minimum_output_frame_count": output_frames,
        "time_map_strictly_increasing": True,
        "first_frame_preserved": True,
        "last_frame_preserved": True,
        "post_velocity_pass": True,
        "slowdown_ratio_pass": True,
        "cropped": False,
        "tiled": False,
        "target_duration_sec": None,
    }
    failed = sorted(key for key, value in exact.items() if safety.get(key) != value)
    ratio = safety.get("retime_ratio")
    max_ratio = safety.get("max_slowdown_ratio")
    if (
        isinstance(ratio, bool)
        or not isinstance(ratio, (int, float))
        or isinstance(max_ratio, bool)
        or not isinstance(max_ratio, (int, float))
        or float(ratio) < 1.0
        or float(ratio) > float(max_ratio) + 1e-12
        or float(max_ratio) > 1.25 + 1e-12
    ):
        failed.append("retime_ratio")
    time_map = safety.get("input_frame_output_times_sec")
    if (
        not isinstance(time_map, list)
        or len(time_map) != source_frames
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in time_map
        )
        or any(float(right) <= float(left) for left, right in zip(time_map, time_map[1:]))
    ):
        failed.append("input_frame_output_times_sec")
    if failed:
        raise ValueError(f"{task_id}: invalid safety retime audit: {sorted(set(failed))}")
    return output_frames, "safety_monotonic_retimed_final_output", dict(safety)


def _require_all_true_gate(record: dict[str, Any]) -> dict[str, bool]:
    task_id = record.get("task_id")
    gate = record.get("quality_gate")
    if not isinstance(gate, dict) or not gate:
        raise ValueError(f"{task_id}: missing quality_gate")
    if any(not isinstance(value, bool) for value in gate.values()):
        raise ValueError(f"{task_id}: quality gates must be boolean")
    if any(value is not True for value in gate.values()):
        raise ValueError(f"{task_id}: every quality gate must pass")
    return dict(gate)


def _reject_legacy_fields(record: dict[str, Any], *, context: str) -> None:
    present = sorted(LEGACY_SEMANTIC_FIELDS.intersection(record))
    if present:
        raise ValueError(f"{context}: legacy semantic-event fields are forbidden: {present}")


def queue_record(
    record: dict[str, Any], *, expected_selection_kind: str
) -> dict[str, Any]:
    task_id = str(record.get("task_id") or "")
    if not task_id:
        raise ValueError("retarget pass is missing task_id")
    if record.get("artifact_kind") != OUTPUT_ARTIFACT_KIND or record.get("status") != "passed":
        raise ValueError(f"{task_id}: only v8 physical-QC passes may be rendered")
    _reject_legacy_fields(record, context=task_id)
    if record.get("expression_turn_selection_kind") != expected_selection_kind:
        raise ValueError(f"{task_id}: review-set selection kind mismatch")
    if record.get("accepted_for_training") is not False:
        raise ValueError(f"{task_id}: training admission must remain false")
    if record.get("semantic_supervision_masks") != SEMANTIC_MASKS:
        raise ValueError(f"{task_id}: semantic masks are not fail-closed")
    for field in (
        "emotion_supervision_mask",
        "official_emotion_conditioning_enabled",
        "affect_observable_supervision_mask",
        "official_category_conditioning_enabled",
    ):
        if record.get(field) is not False:
            raise ValueError(f"{task_id}: {field} must remain false")
    if record.get("canonical_prompt") is not None or record.get("canonical_action") is not None:
        raise ValueError(f"{task_id}: pre-review action text is forbidden")

    training = record.get("training_segment")
    retarget = record.get("retarget_segment")
    if not isinstance(training, dict) or not isinstance(retarget, dict):
        raise ValueError(f"{task_id}: missing natural/retarget segment contracts")
    if (
        training.get("representation") != INPUT_REPRESENTATION
        or training.get("duration_policy") != NATURAL_DURATION_POLICY
        or training.get("fixed_window_sec") is not None
        or training.get("cropped") is not False
    ):
        raise ValueError(f"{task_id}: training segment is not native natural length")
    if (
        retarget.get("cropped") is not False
        or retarget.get("duration_policy") != NATURAL_DURATION_POLICY
        or retarget.get("fixed_target_duration_sec") is not None
    ):
        raise ValueError(f"{task_id}: retarget is not a full natural expression turn")

    gate = _require_all_true_gate(record)
    trajectory = _verified_file(record, "safe_csv", "safe_csv_sha256")
    quality_path = _verified_file(record, "quality_json", "quality_json_sha256")
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    if not isinstance(quality, dict):
        raise ValueError(f"{task_id}: quality JSON is not an object")
    _reject_legacy_fields(quality, context=f"{task_id}/quality")
    validation = quality.get("expression_turn_output_contract_validation")
    if not isinstance(validation, dict) or validation.get("passed") is not True:
        raise ValueError(f"{task_id}: independent output contract did not pass")
    if quality.get("training_segment") != training:
        raise ValueError(f"{task_id}: quality changed the natural segment")
    output_frames, final_trajectory_role, safety_retime = _validate_final_output_contract(
        task_id=task_id,
        record=record,
        training=training,
        retarget=retarget,
        quality=quality,
        trajectory=trajectory,
    )

    queued = {
        "schema_version": "1.0.0",
        "artifact_kind": "expression_turn_v8_internal_video_review_queue_record",
        "task_id": task_id,
        "status": "passed",
        "source_clip_id": record.get("source_clip_id"),
        "speaker_key": record.get("speaker_key"),
        "official_split": record.get("official_split"),
        "fixed_split_assignment": record.get("fixed_split_assignment"),
        "robot_contract": ROBOT_CONTRACT,
        "fps": record.get("fps"),
        "canonical_action": None,
        "canonical_action_role": "disabled_pending_independent_blind_review",
        "canonical_prompt": dict(ANONYMOUS_PROMPT),
        "canonical_prompt_role": "anonymous_renderer_placeholder_not_conditioning_text",
        "robot_observable_motion_form": "candidate_unreviewed",
        "communicative_intent": "candidate_unreviewed",
        "semantic_supervision_masks": dict(SEMANTIC_MASKS),
        "emotion_id": record.get("emotion_id"),
        "emotion_supervision_mask": False,
        "source_emotion_label_verified": record.get("source_emotion_label_verified"),
        "emotion_supervision_role": "disabled_pending_robot_affect_review",
        "official_emotion_conditioning_enabled": False,
        "official_emotion_condition_channel": None,
        "official_emotion_loss": None,
        "official_category_conditioning_enabled": False,
        "affect_observable_review_status": "candidate_unreviewed",
        "affect_observable_supervision_mask": False,
        "trajectory_path": str(trajectory),
        "trajectory_sha256": record["safe_csv_sha256"],
        "trajectory_frames_expected": output_frames,
        "source_frame_count": training["frame_count"],
        "output_frame_count": output_frames,
        "final_trajectory_role": final_trajectory_role,
        "blind_review_must_use_final_trajectory": True,
        "safe_csv": str(trajectory),
        "safe_csv_sha256": record["safe_csv_sha256"],
        "quality_json": str(quality_path),
        "quality_json_sha256": record["quality_json_sha256"],
        "quality_gate": gate,
        "core_interval": record.get("core_interval"),
        "context_plan": record.get("context_plan"),
        "training_segment": training,
        "time_axes": record.get("time_axes"),
        "expression_turn": record.get("expression_turn"),
        "retarget_segment": retarget,
        "duration_band": record.get("duration_band"),
        "event_count_band": record.get("event_count_band"),
        "expression_turn_contract_sha256": record.get(
            "expression_turn_contract_sha256"
        ),
        "expression_turn_record_sha256": record.get(
            "expression_turn_record_sha256"
        ),
        "expression_turn_selection_kind": record.get(
            "expression_turn_selection_kind"
        ),
        "expression_turn_selection_rank": record.get(
            "expression_turn_selection_rank"
        ),
        "expression_turn_selection_status": record.get(
            "expression_turn_selection_status"
        ),
        "expression_turn_selection_record_sha256": record.get(
            "expression_turn_selection_record_sha256"
        ),
        "source_inventory_manifest_sha256": record.get(
            "source_inventory_manifest_sha256"
        ),
        "split_assignment_manifest_sha256": record.get(
            "split_assignment_manifest_sha256"
        ),
        "upstream_event_record_sha256": record.get(
            "upstream_event_record_sha256"
        ),
        "upstream_inventory_record_sha256": record.get(
            "upstream_inventory_record_sha256"
        ),
        "selected_record_sha256": record.get("selected_record_sha256"),
        "retarget_input_manifest_sha256": record.get(
            "retarget_input_manifest_sha256"
        ),
        "review_state": "pending_separate_blind_arc_action_and_affect_review",
        "manual_review_required": True,
        "semantic_action_completeness_review_required": True,
        "affect_observable_review_required": True,
        "render_pass_grants_training_admission": False,
        "speech_context_included": False,
        "accepted_for_training": False,
    }
    if safety_retime is not None:
        queued["safety_monotonic_retime"] = safety_retime
    return queued


def build_queue(
    passed_manifest: Path | Iterable[Path],
    output: Path,
    *,
    selection_kind: str,
) -> dict[str, Any]:
    if selection_kind not in SELECTION_KINDS:
        raise ValueError(f"invalid selection kind: {selection_kind}")
    manifests = (
        [passed_manifest]
        if isinstance(passed_manifest, Path)
        else list(passed_manifest)
    )
    if not manifests:
        raise ValueError("at least one passed manifest is required")
    manifests = [path.resolve() for path in manifests]
    records = [
        queue_record(record, expected_selection_kind=selection_kind)
        for path in manifests
        for record in read_jsonl(path)
    ]
    task_ids = [record["task_id"] for record in records]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("duplicate task_id in retarget passed manifest")
    records.sort(key=lambda item: item["task_id"])
    output = output.resolve()
    atomic_jsonl(output, records)
    summary = {
        "schema_version": "1.0.0",
        "artifact_kind": "expression_turn_v8_internal_video_review_queue",
        "selection_kind": selection_kind,
        "records": len(records),
        "counts_by_split": dict(
            sorted(Counter(item["fixed_split_assignment"] for item in records).items())
        ),
        "counts_by_duration_band": dict(
            sorted(Counter(item["duration_band"] for item in records).items())
        ),
        "counts_by_event_count_band": dict(
            sorted(Counter(item["event_count_band"] for item in records).items())
        ),
        "passed_manifests": [str(path) for path in manifests],
        "passed_manifest_sha256": [sha256(path) for path in manifests],
        "output": str(output),
        "output_sha256": sha256(output),
        "renderer_prompt_contains_action_or_emotion_target": False,
        "blind_bundle_must_strip_source_identity_and_official_metadata": True,
        "natural_full_length_render_required": True,
        "final_trajectory_only_render_required": True,
        "source_frame_count_total": sum(item["source_frame_count"] for item in records),
        "output_frame_count_total": sum(item["output_frame_count"] for item in records),
        "accepted_for_training": 0,
    }
    atomic_json(output.with_suffix(".summary.json"), summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--passed-manifest", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--selection-kind", choices=sorted(SELECTION_KINDS), required=True
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_queue(
        args.passed_manifest,
        args.output,
        selection_kind=args.selection_kind,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
