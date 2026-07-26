#!/usr/bin/env python3
"""Resumable batch runner for strict BEAT2 -> ULA V2 18D retargeting."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

try:
    from .retarget_beat2_v2 import (
        BEAT2_AXIS_POLICY,
        DEFAULT_CONFIG,
        DEFAULT_GMR_ROOT,
        DEFAULT_MODEL,
        DEFAULT_URDF,
        JOINT_LIMITS_18D,
        JOINT_ORDER_18D,
        ULA_V2_18D_CONTRACT,
    )
except ImportError:  # pragma: no cover - direct CLI execution
    from retarget_beat2_v2 import (
        BEAT2_AXIS_POLICY,
        DEFAULT_CONFIG,
        DEFAULT_GMR_ROOT,
        DEFAULT_MODEL,
        DEFAULT_URDF,
        JOINT_LIMITS_18D,
        JOINT_ORDER_18D,
        ULA_V2_18D_CONTRACT,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BEAT2_ROOT = Path(
    "/home/gez/nas/cloud/gez/human_motion/raw/BEAT2/beat_chinese_v2.0.0"
)
DEFAULT_INVENTORY = (
    PROJECT_ROOT
    / "deliverables/interactive_human_motion_v1/catalog/beat2_interaction_full_inventory_v1.jsonl"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "deliverables/interactive_human_motion_v1/batch_annotation_v1/retarget"
)
RETARGET_SCRIPT = Path(__file__).with_name("retarget_beat2_v2.py")
RETARGET_IMPLEMENTATION = Path(__file__).with_name("retarget_motionx322_v2.py")
RETARGET_COMMON_IMPLEMENTATION = Path(__file__).with_name("retarget_xsens_v2.py")
ULA_18D_CONTRACT_IMPLEMENTATION = (
    PROJECT_ROOT / "upper_body_skeleton/retarget_v2_18d.py"
)
GMR_VENV_PYTHON = Path("/home/gez/shuaiwang/.venvs/gmr/bin/python")
DEFAULT_RETARGET_PYTHON = GMR_VENV_PYTHON if GMR_VENV_PYTHON.is_file() else Path(sys.executable)
SELECTION_STATUS = "selected_nonstatic_low_dynamic_with_aligned_speech"
FULL_WINDOW_SELECTION_STATUS = "full_nonoverlap_boundary_validated"
SEMANTIC_EVENT_SELECTION_STATUS = (
    "official_semantic_event_variable_length_boundary_validated"
)
ELIGIBLE_SELECTION_STATUSES = {
    SELECTION_STATUS,
    FULL_WINDOW_SELECTION_STATUS,
    SEMANTIC_EVENT_SELECTION_STATUS,
}
NON_BLOCKING_WARNINGS = {
    "motion_audio_duration_mismatch_gt_0_3s",
    "textgrid_transcript_mismatch",
}
SCHEMA_VERSION = "1.0.0"
RUN_CONTRACT_SCHEMA_VERSION = "1.0.0"
RUN_CONTRACT_FILENAME = "run_contract.json"
VARIABLE_SEGMENT_REPRESENTATION = "native_variable_length_semantic_clip_v1"

# Keep label evidence attached to every retarget result.  A physical retarget
# pass is still not semantic admission, but downstream review must not have to
# recover these fields by loosely joining a second manifest.
INVENTORY_METADATA_FIELDS = (
    "dataset",
    "dataset_subset",
    "language",
    "language_code",
    "annotation_kind",
    "annotation_relpath",
    "textgrid_relpath",
    "interaction_label",
    "interaction_label_source",
    "interaction_scope",
    "interaction_scope_status",
    "semantic_label_status",
    "semantic_gesture",
    "canonical_action",
    "canonical_action_role",
    "official_category_verified",
    "official_category_role",
    "official_category_condition_channel",
    "official_category_loss",
    "official_category_conditioning_enabled",
    "robot_observable_motion_form",
    "communicative_intent",
    "semantic_supervision_masks",
    "official_gesture_semantic_spans",
    "official_gesture_semantic_role",
    "official_semantic_event",
    "semantic_event",
    "semantic_mapping_status",
    "interaction_scope_role",
    "training_segment",
    "behavior_id",
    "behavior_review_status",
    "behavior_supervision_mask",
    "behavior_source",
    "behavior_mapping_contract",
    "behavior_label_status",
    "emotion_id",
    "emotion_review_status",
    "emotion_supervision_mask",
    "source_emotion_label_verified",
    "emotion_supervision_role",
    "official_emotion_conditioning_enabled",
    "official_emotion_condition_channel",
    "official_emotion_loss",
    "affect_observable_review_status",
    "affect_observable_supervision_mask",
    "emotion_protocol_contract",
    "emotion_label_status",
    "emotion_source",
    "emotion_label_source",
    "emotion_protocol_kind",
    "source_emotion_id",
    "source_emotion_label",
    "source_emotion_protocol",
    "window_transcript_context",
    "window_transcript_role",
    "source_text",
    "source_text_role",
    "canonical_prompt",
    "canonical_prompt_role",
    "prompt",
    "prompt_schema",
    "prompt_source",
    "prompt_sha256",
    "prompt_contract",
    "motion_sha256",
    "inventory_record_sha256",
    "inventory_record_sha256_role",
    "upstream_inventory_record_sha256",
    "inventory_manifest_sha256",
    "inventory_manifest_sha256_role",
    "upstream_inventory_manifest_sha256",
    "pilot_selector_contract_sha256",
    "pilot_source_group_sha256",
    "pilot_speaker_group_sha256",
    "pilot_dynamic_band",
    "pilot_stratum",
    "pilot_split",
    "fixed_split_assignment",
    "pilot_selection_status",
    "pilot_selection_pass",
    "training_pool_selection_status",
    "training_pool_contract_sha256",
    "qc_replacement_round",
    "qc_replacement_for_stratum",
    "qc_replacement_selection_status",
    "qc_replacement_contract_sha256",
    "training_admission_status",
)

# These are passed explicitly to the retarget subprocess. Keep the corresponding
# quality thresholds here as an auditable contract rather than relying only on
# argparse defaults embedded in another file.
RETARGET_PARAMETERS = {
    "warmup_frames": 0,
    "max_velocity_rad_s": 3.0,
    "smoothing_window": 7,
    "posture_cost": 0.02,
    "solver": "daqp",
    "model_sha_check": "strict",
}
QUALITY_POLICY = {
    "all_reported_quality_gates_must_be_boolean_true": True,
    "joint_limit_tolerance_rad": 1e-8,
    "velocity_tolerance_rad_s": 1e-6,
    "limb_target_error_p95_max_m": 0.04,
    "upper_body_collision_frame_rate_max": 0.05,
    "axis_alignment_determinant_max": -0.999,
    "positive_elbow_branch_tolerance_rad": 1e-6,
    "head_direction_roundtrip_max_rad": 1e-5,
    "head_continuity_raw_component_step_max_rad": 1.5707963267948966,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: object) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def python_tree_sha256(root: Path) -> tuple[str, int]:
    """Hash importable GMR Python sources in a path-independent order."""
    files = sorted(
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts and path.is_file()
    )
    entries = [
        {"relative_path": path.relative_to(root).as_posix(), "sha256": sha256(path)}
        for path in files
    ]
    return json_sha256(entries), len(entries)


def file_binding(path: Path) -> dict:
    absolute = path.absolute()
    resolved = absolute.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(absolute),
        "resolved_path": str(resolved),
        "sha256": sha256(resolved),
    }


def build_run_contract(args) -> tuple[dict, str]:
    gmr_python_root = args.gmr_root / "general_motion_retargeting"
    if not gmr_python_root.is_dir():
        # Focused tests and custom adapters may provide an empty GMR root. The
        # empty tree is still deterministic and remains bound to this root.
        gmr_python_root = args.gmr_root
    gmr_tree_hash, gmr_python_file_count = python_tree_sha256(gmr_python_root)
    contract = {
        "schema_version": RUN_CONTRACT_SCHEMA_VERSION,
        "artifacts": {
            "batch_runner": file_binding(Path(__file__)),
            "retarget_entrypoint": file_binding(args.retarget_script),
            "retarget_motionx_implementation": file_binding(RETARGET_IMPLEMENTATION),
            "retarget_common_implementation": file_binding(
                RETARGET_COMMON_IMPLEMENTATION
            ),
            "ula_18d_contract_implementation": file_binding(
                ULA_18D_CONTRACT_IMPLEMENTATION
            ),
            "gmr_python_tree": {
                "path": str(gmr_python_root.resolve()),
                "sha256": gmr_tree_hash,
                "python_file_count": gmr_python_file_count,
            },
            "robot_urdf": file_binding(args.urdf),
            "smplx_model": file_binding(args.smplx_model),
            "retarget_config": file_binding(args.config),
            "python_interpreter": file_binding(args.python),
        },
        "retarget_parameters": dict(RETARGET_PARAMETERS),
        "output_contract": ULA_V2_18D_CONTRACT,
        "axis_policy": BEAT2_AXIS_POLICY,
        "joint_order": list(JOINT_ORDER_18D),
        "joint_limits_rad": {
            name: [float(lower), float(upper)]
            for name, (lower, upper) in JOINT_LIMITS_18D.items()
        },
        "quality_policy": dict(QUALITY_POLICY),
    }
    return contract, json_sha256(contract)


def resolve_contained_path(
    beat2_root: Path, raw_path: str, *, field: str, clip_id: str
) -> Path:
    root = beat2_root.resolve()
    candidate = (root / raw_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"Inventory {field} for {clip_id} escapes BEAT2 root: {raw_path!r}"
        ) from error
    return candidate


def atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def atomic_jsonl(path: Path, records: list[dict]) -> None:
    payload = "".join(stable_json(record) + "\n" for record in records)
    atomic_text(path, payload)


def record_sha256(record: dict) -> str:
    return hashlib.sha256(stable_json(record).encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def executable_path(path: Path) -> Path:
    """Return an absolute executable path without dereferencing venv symlinks."""
    absolute = path.absolute()
    if not absolute.is_file():
        raise FileNotFoundError(absolute)
    return absolute


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--beat2-root", type=Path, default=DEFAULT_BEAT2_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--smplx-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gmr-root", type=Path, default=DEFAULT_GMR_ROOT)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--clip-id",
        action="append",
        help="Select a specific eligible clip; repeat for more than one",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--python",
        type=Path,
        default=DEFAULT_RETARGET_PYTHON,
        help="Python interpreter containing GMR, mink, MuJoCo and SMPL-X",
    )
    parser.add_argument(
        "--retarget-script",
        type=Path,
        default=RETARGET_SCRIPT,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def read_inventory(path: Path, beat2_root: Path) -> tuple[list[dict], list[dict]]:
    eligible = []
    excluded = []
    seen = set()
    seen_task_ids = set()
    retarget_input_manifest_sha256 = sha256(path)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            clip_id = str(record.get("clip_id", "")).strip()
            if not clip_id:
                raise ValueError(f"Missing clip_id at {path}:{line_number}")
            if clip_id in seen:
                raise ValueError(f"Duplicate clip_id in inventory: {clip_id}")
            seen.add(clip_id)
            source_clip_id = str(record.get("source_clip_id") or clip_id).strip()
            if not source_clip_id:
                raise ValueError(f"Missing source_clip_id at {path}:{line_number}")
            motion_relpath = str(record.get("motion_relpath") or "").strip()
            audio_relpath = str(record.get("audio_relpath") or "").strip()
            source = (
                resolve_contained_path(
                    beat2_root,
                    motion_relpath,
                    field="motion_relpath",
                    clip_id=clip_id,
                )
                if motion_relpath
                else None
            )
            audio_source = (
                resolve_contained_path(
                    beat2_root,
                    audio_relpath,
                    field="audio_relpath",
                    clip_id=clip_id,
                )
                if audio_relpath
                else None
            )
            window = record.get("window") or {}
            status = window.get("selection_status")
            issues = sorted(set(record.get("issues") or []))
            blocking = [issue for issue in issues if issue not in NON_BLOCKING_WARNINGS]
            reasons = []
            if status not in ELIGIBLE_SELECTION_STATUSES:
                reasons.append(f"selection_status:{status}")
            if status == SEMANTIC_EVENT_SELECTION_STATUS:
                segment = record.get("training_segment")
                if not isinstance(segment, dict):
                    reasons.append("semantic_event:missing_training_segment")
                else:
                    if segment.get("representation") != VARIABLE_SEGMENT_REPRESENTATION:
                        reasons.append("semantic_event:invalid_variable_length_representation")
                    if segment.get("fixed_window_sec") is not None:
                        reasons.append("semantic_event:fixed_window_forbidden")
                    if not str(segment.get("boundary_source") or "").strip():
                        reasons.append("semantic_event:missing_boundary_source")
            reasons.extend(f"blocking_issue:{issue}" for issue in blocking)
            selected_record_sha256 = record_sha256(record)
            upstream_record_sha256 = record.get(
                "upstream_inventory_record_sha256"
            ) or record.get("inventory_record_sha256")
            upstream_manifest_sha256 = record.get(
                "upstream_inventory_manifest_sha256"
            ) or record.get("inventory_manifest_sha256")
            if record.get("pilot_selector_contract_sha256") is not None:
                if not _is_sha256(record.get("upstream_inventory_record_sha256")):
                    raise ValueError(
                        f"{clip_id}: formal selector row lacks "
                        "upstream_inventory_record_sha256"
                    )
                if not _is_sha256(
                    record.get("upstream_inventory_manifest_sha256")
                ):
                    raise ValueError(
                        f"{clip_id}: formal selector row lacks "
                        "upstream_inventory_manifest_sha256"
                    )
            if (
                record.get("inventory_record_sha256") is not None
                and upstream_record_sha256 != record.get("inventory_record_sha256")
            ):
                raise ValueError(f"{clip_id}: ambiguous upstream inventory row hashes")
            lineage = {
                "inventory_record_sha256": (
                    upstream_record_sha256 or selected_record_sha256
                ),
                "upstream_inventory_record_sha256": (
                    upstream_record_sha256 or selected_record_sha256
                ),
                "selected_record_sha256": selected_record_sha256,
                "selected_record_sha256_role": (
                    "canonical_row_in_current_retarget_input_manifest"
                ),
                "retarget_input_manifest_sha256": retarget_input_manifest_sha256,
                "retarget_input_manifest_sha256_role": (
                    "current_retarget_input_manifest"
                ),
            }
            if upstream_manifest_sha256 is not None:
                lineage["upstream_inventory_manifest_sha256"] = (
                    upstream_manifest_sha256
                )
            inventory_metadata = {
                field: record[field]
                for field in INVENTORY_METADATA_FIELDS
                if field in record
            }
            if reasons:
                start = window.get("start_frame")
                end = window.get("end_frame_exclusive")
                task_id = str(record.get("task_id") or "").strip() or (
                    f"{clip_id}_f{start:06d}-{end:06d}"
                    if isinstance(start, int) and not isinstance(start, bool)
                    and isinstance(end, int) and not isinstance(end, bool)
                    else f"{clip_id}_unwindowed"
                )
                excluded.append({
                    **inventory_metadata,
                    **lineage,
                    "accepted_for_training": False,
                    "clip_id": clip_id,
                    "source_clip_id": source_clip_id,
                    "source_group_key": str(
                        record.get("source_group_key")
                        or record.get("source_group_id")
                        or source_clip_id
                    ),
                    "task_id": task_id,
                    "start_frame": start,
                    "end_frame_exclusive": end,
                    "fps": record.get("fps"),
                    "official_split": record.get("official_split"),
                    "speaker_key": record.get("speaker_key"),
                    "issues": issues,
                    "reasons": reasons,
                    "selection_status": status,
                    "status": "excluded",
                })
                continue
            if not motion_relpath:
                raise ValueError(f"Eligible record {clip_id} has no motion_relpath")
            start = window.get("start_frame")
            end = window.get("end_frame_exclusive")
            if isinstance(start, bool) or not isinstance(start, int):
                raise ValueError(f"Eligible record {clip_id} has invalid start_frame")
            if isinstance(end, bool) or not isinstance(end, int) or end <= start:
                raise ValueError(f"Eligible record {clip_id} has invalid end_frame_exclusive")
            task_id = str(record.get("task_id") or "").strip() or (
                f"{clip_id}_f{start:06d}-{end:06d}"
            )
            if task_id in seen_task_ids:
                raise ValueError(f"Duplicate task_id in inventory: {task_id}")
            seen_task_ids.add(task_id)
            assert source is not None
            if not source.is_file():
                raise FileNotFoundError(source)
            eligible.append(
                {
                    **inventory_metadata,
                    **lineage,
                    "clip_id": clip_id,
                    "source_clip_id": source_clip_id,
                    "source_group_key": str(
                        record.get("source_group_key")
                        or record.get("source_group_id")
                        or source_clip_id
                    ),
                    "task_id": task_id,
                    "source": str(source),
                    "motion_relpath": motion_relpath,
                    "audio_source": (
                        str(audio_source) if audio_source is not None else None
                    ),
                    "audio_relpath": audio_relpath or None,
                    "start_frame": start,
                    "end_frame_exclusive": end,
                    "fps": float(record["fps"]),
                    "official_split": record.get("official_split"),
                    "speaker_key": record.get("speaker_key"),
                    "source_warnings": [
                        issue for issue in issues if issue in NON_BLOCKING_WARNINGS
                    ],
                    "interaction_label": record.get("interaction_label"),
                    "source_speech_context": record.get("window_transcript_context", ""),
                    "source_speech_context_role": (
                        record.get("window_transcript_role")
                        or "speech_context_only_not_action_label"
                    ),
                    "conditioning_text_status": (
                        record.get("conditioning_text_status")
                        or "speech_context_only_not_approved_action_prompt"
                    ),
                    "accepted_for_training": False,
                }
            )
    return eligible, excluded


def quality_passes(quality: dict, task: dict, source_hash: str) -> bool:
    gate = quality.get("quality_gate") or {}
    strict_gate_values = [value for key, value in gate.items() if key != "passed"]
    gate_values_pass = bool(strict_gate_values) and all(
        value is True for value in strict_gate_values
    )
    expected_frames = task["end_frame_exclusive"] - task["start_frame"]
    lineage_matches = all(
        task.get(field) is None or quality.get(field) == task.get(field)
        for field in (
            "inventory_record_sha256",
            "upstream_inventory_record_sha256",
            "selected_record_sha256",
            "retarget_input_manifest_sha256",
        )
    )
    return bool(
        quality.get("axis_policy") == BEAT2_AXIS_POLICY
        and quality.get("output_contract") == ULA_V2_18D_CONTRACT
        and quality.get("action_dim") == 18
        and quality.get("joint_order") == list(JOINT_ORDER_18D)
        and quality.get("fps") == task["fps"]
        and quality.get("source_window_frames") == expected_frames
        and isinstance(quality.get("frames"), int)
        and quality.get("frames", 0) > 0
        and quality.get("source_sha256") == source_hash
        and quality.get("source_window_start_frame") == task["start_frame"]
        and quality.get("source_window_end_frame_exclusive")
        == task["end_frame_exclusive"]
        and lineage_matches
        and gate.get("passed") is True
        and gate_values_pass
    )


def only_safe_csv(directory: Path) -> Path:
    candidates = sorted(directory.glob("*_gmr_safe_18d.csv"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one safe 18D CSV in {directory}, found {len(candidates)}")
    return candidates[0]


def publish_directory(source: Path, destination: Path, archive_root: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        archive_root.mkdir(parents=True, exist_ok=True)
        archived = archive_root / f"{destination.name}_{time.time_ns()}"
        os.replace(destination, archived)
    os.replace(source, destination)


def build_command(task: dict, args, stage_dir: Path) -> list[str]:
    return [
        str(args.python),
        str(args.retarget_script),
        "--beat2",
        task["source"],
        "--start-frame",
        str(task["start_frame"]),
        "--end-frame",
        str(task["end_frame_exclusive"]),
        "--fps",
        str(task["fps"]),
        "--output-contract",
        ULA_V2_18D_CONTRACT,
        "--smplx-model",
        str(args.smplx_model),
        "--gmr-root",
        str(args.gmr_root),
        "--urdf",
        str(args.urdf),
        "--config",
        str(args.config),
        "--warmup-frames",
        str(RETARGET_PARAMETERS["warmup_frames"]),
        "--max-velocity",
        str(RETARGET_PARAMETERS["max_velocity_rad_s"]),
        "--smoothing-window",
        str(RETARGET_PARAMETERS["smoothing_window"]),
        "--posture-cost",
        str(RETARGET_PARAMETERS["posture_cost"]),
        "--solver",
        str(RETARGET_PARAMETERS["solver"]),
        "--output-dir",
        str(stage_dir),
    ]


def result_path(output_root: Path, task: dict) -> Path:
    return output_root / "state/results" / f"{task['task_id']}.json"


def load_result(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def completed_pass_is_current(
    result: dict, task: dict, run_contract_hash: str
) -> bool:
    if result.get("status") != "passed":
        return False
    if result.get("run_contract_sha256") != run_contract_hash:
        return False
    output_dir = Path(str(result.get("output_dir", "")))
    quality_path = output_dir / "quality.json"
    if not quality_path.is_file():
        return False
    try:
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        source_hash = sha256(Path(task["source"]))
        safe_csv = only_safe_csv(output_dir)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        result_lineage_matches(result, task)
        and quality_passes(quality, task, source_hash)
        and result.get("safe_csv_sha256") == sha256(safe_csv)
        and result.get("quality_json_sha256") == sha256(quality_path)
    )


def result_lineage_matches(result: dict, task: dict) -> bool:
    return all(
        result.get(field) == task.get(field)
        for field in (
            "inventory_record_sha256",
            "upstream_inventory_record_sha256",
            "selected_record_sha256",
            "retarget_input_manifest_sha256",
        )
    )


def _base_result(
    task: dict,
    inventory: Path,
    inventory_hash: str,
    output_root: Path,
    run_contract_hash: str,
) -> dict:
    if task.get("retarget_input_manifest_sha256") != inventory_hash:
        raise ValueError(
            f"{task.get('task_id')}: task is not bound to the current retarget manifest"
        )
    lineage_hashes = {
        "upstream_inventory_manifest_sha256": task.get(
            "upstream_inventory_manifest_sha256"
        ),
        "upstream_inventory_record_sha256": task.get(
            "upstream_inventory_record_sha256"
        ),
        "retarget_input_manifest_sha256": inventory_hash,
        "selected_record_sha256": task.get("selected_record_sha256"),
    }
    return {
        **task,
        "schema_version": SCHEMA_VERSION,
        "source": str(Path(task["source"]).resolve()),
        "source_manifest": str(inventory.resolve()),
        "source_manifest_sha256": inventory_hash,
        "source_manifest_sha256_role": "current_retarget_input_manifest_compatibility",
        "lineage_hashes": lineage_hashes,
        "retarget_contract": ULA_V2_18D_CONTRACT,
        "axis_policy": BEAT2_AXIS_POLICY,
        "run_contract_sha256": run_contract_hash,
        "run_contract_path": str((output_root / RUN_CONTRACT_FILENAME).resolve()),
        "semantic_admission": "pending_separate_review",
    }


def run_task(
    task: dict,
    args,
    inventory_hash: str,
    run_contract_hash: str,
    run_id: str,
) -> dict:
    started_at = utc_now()
    started = time.perf_counter()
    source_hash = sha256(Path(task["source"]))
    stage_dir = args.output_root / "staging" / run_id / task["task_id"]
    stage_dir.mkdir(parents=True, exist_ok=False)
    log_dir = args.output_root / "logs" / task["task_id"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_id}.log"
    command = build_command(task, args, stage_dir)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command=" + stable_json(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, text=True, check=False
        )
    base = _base_result(
        task,
        args.inventory,
        inventory_hash,
        args.output_root,
        run_contract_hash,
    )
    common = {
        **base,
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": utc_now(),
        "elapsed_sec": round(time.perf_counter() - started, 6),
        "log_path": str(log_path.resolve()),
        "source_sha256": source_hash,
        "returncode": completed.returncode,
    }
    if completed.returncode != 0:
        destination = args.output_root / "failed" / task["task_id"]
        publish_directory(stage_dir, destination, args.output_root / "superseded/failed")
        result = {**common, "status": "process_failed", "output_dir": str(destination)}
        atomic_json(result_path(args.output_root, task), result)
        return result

    quality_path = stage_dir / "quality.json"
    try:
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        destination = args.output_root / "failed" / task["task_id"]
        publish_directory(stage_dir, destination, args.output_root / "superseded/failed")
        result = {
            **common,
            "status": "invalid_quality_report",
            "error": str(error),
            "output_dir": str(destination),
        }
        atomic_json(result_path(args.output_root, task), result)
        return result

    quality.update(
        {
            field: task[field]
            for field in (
                "inventory_record_sha256",
                "upstream_inventory_record_sha256",
                "selected_record_sha256",
                "retarget_input_manifest_sha256",
            )
            if field in task
        }
    )
    passed = quality_passes(quality, task, source_hash)
    category = "passed" if passed else "failed"
    destination = args.output_root / category / task["task_id"]
    publish_directory(
        stage_dir, destination, args.output_root / "superseded" / category
    )
    quality_path = destination / "quality.json"
    safe_csv = only_safe_csv(destination)
    quality["outputs"] = {
        "raw_csv": str(next(iter(sorted(destination.glob("*_gmr_raw_18d.csv"))), "")),
        "safe_csv": str(safe_csv.resolve()),
    }
    atomic_json(quality_path, quality)
    result = {
        **common,
        "status": "passed" if passed else "quality_failed",
        "output_dir": str(destination.resolve()),
        "quality_json": str(quality_path.resolve()),
        "quality_json_sha256": sha256(quality_path),
        "safe_csv": str(safe_csv.resolve()),
        "safe_csv_sha256": sha256(safe_csv),
        "quality_gate": quality.get("quality_gate", {}),
        "frames": quality.get("frames"),
        "duration_sec": quality.get("duration_sec"),
    }
    atomic_json(result_path(args.output_root, task), result)
    return result


def all_saved_results(output_root: Path, tasks: list[dict]) -> list[dict]:
    records = []
    for task in tasks:
        record = load_result(result_path(output_root, task))
        if record is not None:
            records.append(record)
    return sorted(records, key=lambda item: item["task_id"])


def validate_saved_result_contracts(
    output_root: Path, tasks: list[dict], run_contract_hash: str
) -> None:
    for task in tasks:
        path = result_path(output_root, task)
        if not path.exists():
            continue
        record = load_result(path)
        if record is None:
            raise RuntimeError(f"Unreadable saved retarget result; refusing resume: {path}")
        if record.get("run_contract_sha256") != run_contract_hash:
            raise RuntimeError(
                "Saved retarget result contract changed or is missing; "
                f"refusing unsafe resume: {path}"
            )


def write_manifests(output_root: Path, tasks: list[dict], excluded: list[dict],
                    inventory: Path, inventory_hash: str,
                    run_contract_hash: str) -> tuple[list[dict], list[dict]]:
    records = all_saved_results(output_root, tasks)
    completed_ids = {record["task_id"] for record in records}
    pending = [
        {
            **_base_result(
                task,
                inventory,
                inventory_hash,
                output_root,
                run_contract_hash,
            ),
            "status": "pending",
            "pending_reason": "no_terminal_retarget_result",
        }
        for task in tasks
        if task["task_id"] not in completed_ids
    ]
    excluded_records = [
        {
            **record,
            "schema_version": SCHEMA_VERSION,
            "source_manifest": str(inventory.resolve()),
            "source_manifest_sha256": inventory_hash,
            "run_contract_sha256": run_contract_hash,
            "run_contract_path": str(
                (output_root / RUN_CONTRACT_FILENAME).resolve()
            ),
        }
        for record in excluded
    ]
    atomic_jsonl(output_root / "passed_manifest.jsonl", [
        record for record in records if record.get("status") == "passed"
    ])
    atomic_jsonl(output_root / "failed_manifest.jsonl", [
        record for record in records if record.get("status") != "passed"
    ])
    atomic_jsonl(output_root / "pending_manifest.jsonl", pending)
    atomic_jsonl(output_root / "excluded_manifest.jsonl", excluded_records)
    return records, pending


def status_payload(args, inventory_hash: str, eligible: list[dict], excluded: list[dict],
                   selected: list[dict], results: list[dict], run_id: str,
                   run_state: str, started_at: str,
                   pending: list[dict], run_contract: dict,
                   run_contract_hash: str) -> dict:
    counts = Counter(record.get("status", "unknown") for record in results)
    completed_ids = {record["task_id"] for record in results}
    pending_ids = {record["task_id"] for record in pending}
    eligible_ids = {record["task_id"] for record in eligible}
    coverage_complete = bool(
        completed_ids.isdisjoint(pending_ids)
        and completed_ids | pending_ids == eligible_ids
        and len(completed_ids) + len(pending_ids) == len(eligible)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "run_state": run_state,
        "started_at": started_at,
        "updated_at": utc_now(),
        "inventory": str(args.inventory.resolve()),
        "inventory_sha256": inventory_hash,
        "run_contract": run_contract,
        "run_contract_sha256": run_contract_hash,
        "run_contract_path": str(
            (args.output_root / RUN_CONTRACT_FILENAME).resolve()
        ),
        "beat2_root": str(args.beat2_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "retarget_python": str(args.python),
        "selection_status_allowed": sorted(ELIGIBLE_SELECTION_STATUSES),
        "non_blocking_warnings": sorted(NON_BLOCKING_WARNINGS),
        "strict_quality_policy": "all_retarget_beat2_v2_quality_gates_must_pass",
        "output_contract": ULA_V2_18D_CONTRACT,
        "axis_policy": BEAT2_AXIS_POLICY,
        "semantic_admission_policy": "retarget_pass_does_not_imply_train_ready",
        "workers": args.workers,
        "resume": args.resume,
        "retry_failed": args.retry_failed,
        "limit": args.limit,
        "requested_clip_ids": args.clip_id or [],
        "inventory_record_count": len(eligible) + len(excluded),
        "eligible_task_count": len(eligible),
        "excluded_task_count": len(excluded),
        "selected_this_run_count": len(selected),
        "saved_result_count": len(results),
        "pending_count": len(pending),
        "coverage_complete": coverage_complete,
        "counts": dict(sorted(counts.items())),
    }


def validate_resume_contract(
    status_path: Path,
    output_root: Path,
    inventory_hash: str,
    run_contract: dict,
    run_contract_hash: str,
) -> None:
    try:
        previous = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Unreadable saved batch status; refusing resume: {status_path}"
        ) from error
    if not isinstance(previous, dict):
        raise RuntimeError("Saved batch status is not a JSON object; refusing resume")
    if previous.get("inventory_sha256") != inventory_hash:
        raise RuntimeError("Inventory changed since the saved batch; refusing unsafe resume")
    if previous.get("run_contract_sha256") != run_contract_hash:
        raise RuntimeError(
            "Retarget run contract changed or is missing; refusing unsafe resume"
        )
    if previous.get("run_contract") != run_contract:
        raise RuntimeError(
            "Saved status run contract payload does not match its expected contract"
        )

    contract_path = output_root / RUN_CONTRACT_FILENAME
    try:
        saved = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Unreadable saved retarget run contract; refusing resume: {contract_path}"
        ) from error
    if not isinstance(saved, dict):
        raise RuntimeError("Saved retarget run contract is not a JSON object")
    if (
        saved.get("run_contract_sha256") != run_contract_hash
        or saved.get("run_contract") != run_contract
        or json_sha256(saved.get("run_contract")) != run_contract_hash
    ):
        raise RuntimeError(
            "Saved retarget run contract changed or is corrupt; refusing unsafe resume"
        )


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    if args.retry_failed and not args.resume:
        raise ValueError("--retry-failed requires --resume")
    for name in ("inventory", "beat2_root", "smplx_model", "gmr_root", "urdf", "config", "retarget_script"):
        path = getattr(args, name).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        setattr(args, name, path)
    # Resolving a venv interpreter symlink bypasses the venv's site-packages.
    args.python = executable_path(args.python)
    args.output_root = args.output_root.resolve()
    status_path = args.output_root / "status.json"
    inventory_hash = sha256(args.inventory)
    run_contract, run_contract_hash = build_run_contract(args)
    if status_path.exists():
        if not args.resume:
            raise RuntimeError(f"Existing batch state requires --resume: {status_path}")
        validate_resume_contract(
            status_path,
            args.output_root,
            inventory_hash,
            run_contract,
            run_contract_hash,
        )
    elif (args.output_root / RUN_CONTRACT_FILENAME).exists() or any(
        (args.output_root / "state/results").glob("*.json")
    ):
        raise RuntimeError(
            "Retarget output contains state without status.json; refusing unsafe reuse"
        )
    eligible, excluded = read_inventory(args.inventory, args.beat2_root)
    if status_path.exists():
        validate_saved_result_contracts(
            args.output_root, eligible, run_contract_hash
        )
    selected = eligible
    if args.clip_id:
        requested = set(args.clip_id)
        selected = [task for task in eligible if task["clip_id"] in requested]
        missing = requested - {task["clip_id"] for task in selected}
        if missing:
            raise ValueError(f"Requested clip IDs are not eligible: {sorted(missing)}")
    if args.limit is not None:
        selected = selected[: args.limit]
    args.output_root.mkdir(parents=True, exist_ok=True)
    contract_path = args.output_root / RUN_CONTRACT_FILENAME
    if not contract_path.exists():
        atomic_json(
            contract_path,
            {
                "run_contract_sha256": run_contract_hash,
                "run_contract": run_contract,
            },
        )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    started_at = utc_now()

    runnable = []
    for task in selected:
        previous = load_result(result_path(args.output_root, task))
        if previous is None:
            runnable.append(task)
        elif not result_lineage_matches(previous, task):
            runnable.append(task)
        elif completed_pass_is_current(previous, task, run_contract_hash):
            continue
        elif previous.get("status") == "passed" or args.retry_failed:
            runnable.append(task)

    results, pending = write_manifests(
        args.output_root,
        eligible,
        excluded,
        args.inventory,
        inventory_hash,
        run_contract_hash,
    )
    atomic_json(status_path, status_payload(
        args, inventory_hash, eligible, excluded, selected, results, run_id,
        "running", started_at, pending, run_contract, run_contract_hash,
    ))
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    run_task,
                    task,
                    args,
                    inventory_hash,
                    run_contract_hash,
                    run_id,
                ): task
                for task in runnable
            }
            for index, future in enumerate(as_completed(futures), 1):
                task = futures[future]
                try:
                    result = future.result()
                except Exception as error:  # per-task progress survives worker bugs
                    result = {
                        **_base_result(
                            task,
                            args.inventory,
                            inventory_hash,
                            args.output_root,
                            run_contract_hash,
                        ),
                        "status": "worker_failed",
                        "error": repr(error),
                        "run_id": run_id,
                        "finished_at": utc_now(),
                    }
                    atomic_json(result_path(args.output_root, task), result)
                results, pending = write_manifests(
                    args.output_root,
                    eligible,
                    excluded,
                    args.inventory,
                    inventory_hash,
                    run_contract_hash,
                )
                atomic_json(status_path, status_payload(
                    args, inventory_hash, eligible, excluded, selected, results,
                    run_id, "running", started_at, pending, run_contract,
                    run_contract_hash,
                ))
                print(f"[{index:03d}/{len(runnable):03d}] {result['status']}: {task['task_id']}", flush=True)
    except KeyboardInterrupt:
        results, pending = write_manifests(
            args.output_root,
            eligible,
            excluded,
            args.inventory,
            inventory_hash,
            run_contract_hash,
        )
        atomic_json(status_path, status_payload(
            args, inventory_hash, eligible, excluded, selected, results, run_id,
            "interrupted_resumable", started_at, pending, run_contract,
            run_contract_hash,
        ))
        return 130

    results, pending = write_manifests(
        args.output_root,
        eligible,
        excluded,
        args.inventory,
        inventory_hash,
        run_contract_hash,
    )
    atomic_json(status_path, status_payload(
        args, inventory_hash, eligible, excluded, selected, results, run_id,
        "finished", started_at, pending, run_contract, run_contract_hash,
    ))
    print(json.dumps(Counter(record["status"] for record in results), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
