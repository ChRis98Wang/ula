#!/usr/bin/env python3
"""Run the resumable, audio-disabled BEAT2 18D post-training pipeline.

The orchestrator never reads audio or audio-derived labels.  It verifies the
completed retarget batch, builds trajectory-only text labels and semantics,
caches the versioned 264D text/style condition, and finally launches guarded
interaction-domain post-training with Kimodo replay.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_INVENTORY = (
    PROJECT_ROOT
    / "deliverables/interactive_human_motion_v1/catalog/beat2_interaction_full_6s_windows_v1.jsonl"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/gez/nas/cloud/gez/human_motion/processed/beat2_ula_v2_18d_full_v1"
)
DEFAULT_BASE_CHECKPOINT = (
    PROJECT_ROOT
    / "training/runs/motionx_ula_v2_18d_head_train_ready_v2_distilled/ula_fm_checkpoint.pt"
)
DEFAULT_QWEN_CHECKPOINT = (
    PROJECT_ROOT / "training/runs/kimodo_qwen_motion_latent_lora_v2/best.pt"
)
DEFAULT_KIMODO_DATASET = PROJECT_ROOT / "datasets/kimodo_lerobot_mmdit_lite"
DEFAULT_KIMODO_SPLIT = (
    PROJECT_ROOT / "training/runs/kimodo_motion_latent_v1/motion_latent_checkpoint.pt"
)
DEFAULT_POSTTRAIN_OUTPUT = (
    PROJECT_ROOT / "training/runs/beat2_18d_posttrain_full_v1"
)
GMR_PYTHON = Path("/home/gez/shuaiwang/.venvs/gmr/bin/python")
DEFAULT_PYTHON = GMR_PYTHON if GMR_PYTHON.is_file() else Path(sys.executable)

LABEL_SCRIPT = PROJECT_ROOT / "tools/human_motion_collection/label_ula_v2_18d_motion.py"
SEMANTICS_SCRIPT = (
    PROJECT_ROOT / "tools/human_motion_collection/build_beat2_network_semantics.py"
)
CACHE_SCRIPT = PROJECT_ROOT / "tools/train_ula_v2_18d_head.py"
POSTTRAIN_SCRIPT = PROJECT_ROOT / "tools/train_ula_v2_18d_posttrain.py"
HEAD_MODULE = PROJECT_ROOT / "upper_body_skeleton/ula_v2_18d_head.py"
POSTTRAIN_MODULE = PROJECT_ROOT / "upper_body_skeleton/ula_v2_18d_posttrain.py"
CROSS_MODAL_MODULE = PROJECT_ROOT / "upper_body_skeleton/cross_modal_latent.py"
CONDITIONING_MODULE = PROJECT_ROOT / "upper_body_skeleton/ula_v2_conditioning.py"
TRAINING_MODULE = PROJECT_ROOT / "upper_body_skeleton/ula_training.py"
SEMANTICS_MODULE = PROJECT_ROOT / "upper_body_skeleton/kimodo_semantics.py"
RETARGET_MODULE = PROJECT_ROOT / "upper_body_skeleton/retarget_v2.py"
RETARGET_18D_MODULE = PROJECT_ROOT / "upper_body_skeleton/retarget_v2_18d.py"
MOTION_LATENT_MODULE = PROJECT_ROOT / "upper_body_skeleton/motion_latent.py"
SEMANTIC_ADAPTER_MODULE = PROJECT_ROOT / "upper_body_skeleton/semantic_adapter.py"

STAGES = ("verify-retarget", "label", "semantics", "cache", "posttrain")
SCHEMA_VERSION = "1.0.0"
ARTIFACT_KIND = "beat2_ula_v2_18d_no_audio_posttrain_pipeline"
AUDIO_POLICY = "disabled_not_loaded"
SPEECH_CONTEXT_POLICY = "label_audit_only_not_forwarded_to_training"
CONDITIONING_MODALITIES = (
    "trajectory_derived_action_text",
    "trajectory_derived_motion_style",
    "qwen_text_motion_latent",
)
RETARGET_CONTRACT = "ula_v2_18d_head_v1"
CONDITION_DIM = 264
DEFAULT_SPLIT_FRACTIONS = {"train": 0.8, "validation": 0.1, "test": 0.1}
EXPECTED_DEFAULT_SPLIT_COUNTS = {"train": 2026, "validation": 1125, "test": 344}
EXPECTED_DEFAULT_SPEAKER_SPLIT = {
    "12_zhao": "validation",
    "13_lu": "train",
    "22_luqi": "train",
    "23_hailing": "train",
    "24_kexin": "test",
}
QUALITY_GATE_KEYS = {
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
JOINT_ORDER_18D = (
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
)
POSTTRAIN_ARTIFACT_KIND = "ula_mmdit_v2_18d_interaction_posttrain"
POSTTRAIN_TRAINING_STATE_KEYS = {
    "raw_model_state_dict",
    "ema_state_dict",
    "best_model_state_dict",
    "optimizer_state_dict",
    "sampler_state_dict",
    "frame_rng_state",
    "torch_rng_state",
    "cuda_rng_state_all",
    "stale_validations",
}
FORBIDDEN_TRAINING_METADATA_KEY_PARTS = (
    "audio",
    "waveform",
    "transcript",
    "speech_context",
)


class PipelineError(RuntimeError):
    """A fail-closed pipeline contract or execution error."""


class StageCommandError(PipelineError):
    def __init__(self, stage: str, returncode: int):
        super().__init__(f"stage {stage!r} failed with return code {returncode}")
        self.returncode = int(returncode)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def json_sha256(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("ascii")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: str | Path, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: str | Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PipelineError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise PipelineError(f"JSON payload must be an object: {path}")
    return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise PipelineError(f"cannot read JSONL {path}: {error}") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise PipelineError(f"invalid JSON at {path}:{line_number}: {error}") from error
        if not isinstance(record, dict):
            raise PipelineError(f"record at {path}:{line_number} must be an object")
        records.append(record)
    return records


def tree_binding(path: str | Path) -> dict[str, Any]:
    path = Path(path).absolute()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        return {
            "path": str(path),
            "kind": "file",
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    if not path.is_dir():
        raise PipelineError(f"unsupported input path type: {path}")
    records = []
    total_size = 0
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        size = child.stat().st_size
        total_size += size
        records.append(
            {
                "relative_path": child.relative_to(path).as_posix(),
                "size": size,
                "sha256": sha256_file(child),
            }
        )
    return {
        "path": str(path),
        "kind": "directory",
        "file_count": len(records),
        "size": total_size,
        "sha256": json_sha256(records),
    }


def planned_binding(path: Path, producer: str, producer_contract_sha256: str) -> dict[str, Any]:
    return {
        "path": str(path.absolute()),
        "kind": "planned_output",
        "producer_stage": producer,
        "producer_contract_sha256": producer_contract_sha256,
    }


def executable_path(path: str | Path) -> Path:
    # Keep the venv path rather than resolving its symlink to the Conda binary.
    path = Path(path).absolute()
    if not path.is_file():
        raise FileNotFoundError(path)
    if not os.access(path, os.X_OK):
        raise PermissionError(f"Python interpreter is not executable: {path}")
    return path


def record_id(record: Mapping[str, Any]) -> str:
    value = (
        record.get("task_id")
        or record.get("clip_id")
        or record.get("record_id")
        or record.get("sample_id")
    )
    if not isinstance(value, str) or not value.strip():
        raise PipelineError("record is missing a non-empty task/clip id")
    return value.strip()


def resolve_record_path(value: Any, manifest: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PipelineError(f"retarget record is missing {field}")
    path = Path(value)
    return path if path.is_absolute() else (manifest.parent / path).resolve()


def _manifest_ids(records: Sequence[Mapping[str, Any]], name: str) -> list[str]:
    values = [record_id(record) for record in records]
    if len(values) != len(set(values)):
        raise PipelineError(f"{name} contains duplicate record ids")
    return values


def verify_retarget_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    """Verify the complete batch and hash every accepted trajectory artifact."""
    retarget_root = args.output_root / "retarget"
    status_path = retarget_root / "status.json"
    manifest_paths = {
        name: retarget_root / f"{name}_manifest.jsonl"
        for name in ("passed", "failed", "pending", "excluded")
    }
    for path in (args.inventory, status_path, *manifest_paths.values()):
        if not path.is_file():
            raise FileNotFoundError(path)

    inventory = read_jsonl(args.inventory)
    inventory_ids = _manifest_ids(inventory, "inventory")
    if len(inventory) != args.expected_inventory_count:
        raise PipelineError(
            f"inventory count changed: {len(inventory)} != {args.expected_inventory_count}"
        )
    inventory_sha = sha256_file(args.inventory)
    status = load_json(status_path)
    expected_status = {
        "run_state": "finished",
        "coverage_complete": True,
        "pending_count": 0,
        "inventory_record_count": args.expected_inventory_count,
        "eligible_task_count": args.expected_inventory_count,
        "saved_result_count": args.expected_inventory_count,
        "inventory_sha256": inventory_sha,
        "output_contract": RETARGET_CONTRACT,
    }
    for field, expected in expected_status.items():
        if status.get(field) != expected:
            raise PipelineError(
                f"retarget status {field} changed: {status.get(field)!r} != {expected!r}"
            )

    manifests = {name: read_jsonl(path) for name, path in manifest_paths.items()}
    counts = {name: len(records) for name, records in manifests.items()}
    if counts != {
        "passed": args.expected_passed_count,
        "failed": args.expected_failed_count,
        "pending": 0,
        "excluded": 0,
    }:
        raise PipelineError(f"retarget manifest counts changed: {counts}")
    if status.get("counts") != {
        "passed": args.expected_passed_count,
        "quality_failed": args.expected_failed_count,
    }:
        raise PipelineError(f"retarget terminal status counts changed: {status.get('counts')}")

    manifest_id_sets = {
        name: set(_manifest_ids(records, f"{name}_manifest"))
        for name, records in manifests.items()
    }
    all_terminal_ids: set[str] = set()
    for name, values in manifest_id_sets.items():
        overlap = all_terminal_ids & values
        if overlap:
            raise PipelineError(f"retarget ids overlap at {name}: {sorted(overlap)[:5]}")
        all_terminal_ids.update(values)
    if all_terminal_ids != set(inventory_ids):
        missing = sorted(set(inventory_ids) - all_terminal_ids)
        extra = sorted(all_terminal_ids - set(inventory_ids))
        raise PipelineError(
            f"retarget coverage does not match inventory: missing={missing[:5]} extra={extra[:5]}"
        )

    accepted_artifacts = []
    speakers: Counter[str] = Counter()
    total_frames = 0
    total_duration_sec = 0.0
    for record in manifests["passed"]:
        clip_id = record_id(record)
        if record.get("status") != "passed":
            raise PipelineError(f"{clip_id}: passed manifest status is not passed")
        gate = record.get("quality_gate")
        if (
            not isinstance(gate, dict)
            or set(gate) != QUALITY_GATE_KEYS
            or not all(value is True for value in gate.values())
        ):
            raise PipelineError(f"{clip_id}: strict quality gate is not fully passed")
        if record.get("retarget_contract") != RETARGET_CONTRACT:
            raise PipelineError(f"{clip_id}: retarget contract mismatch")
        safe_csv = resolve_record_path(record.get("safe_csv"), manifest_paths["passed"], "safe_csv")
        quality_json = resolve_record_path(
            record.get("quality_json"), manifest_paths["passed"], "quality_json"
        )
        for path in (safe_csv, quality_json):
            if not path.is_file():
                raise FileNotFoundError(path)
        safe_hash = sha256_file(safe_csv)
        quality_hash = sha256_file(quality_json)
        if safe_hash != record.get("safe_csv_sha256"):
            raise PipelineError(f"{clip_id}: safe CSV hash mismatch")
        if quality_hash != record.get("quality_json_sha256"):
            raise PipelineError(f"{clip_id}: quality JSON hash mismatch")
        quality = load_json(quality_json)
        try:
            with safe_csv.open(newline="", encoding="utf-8") as stream:
                header = next(csv.reader(stream))
        except (OSError, UnicodeError, StopIteration, csv.Error) as error:
            raise PipelineError(f"{clip_id}: cannot read 18D CSV header: {error}") from error
        if header != list(JOINT_ORDER_18D):
            raise PipelineError(f"{clip_id}: safe CSV does not use the exact 18D joint order")
        if quality.get("output_contract") != RETARGET_CONTRACT:
            raise PipelineError(f"{clip_id}: quality output contract mismatch")
        if quality.get("action_dim") != 18:
            raise PipelineError(f"{clip_id}: quality action_dim is not 18")
        if quality.get("joint_order") != list(JOINT_ORDER_18D):
            raise PipelineError(f"{clip_id}: quality joint order is not the 18D contract")
        if quality.get("quality_gate") != gate:
            raise PipelineError(f"{clip_id}: manifest and quality gates differ")
        accepted_artifacts.append(
            {
                "clip_id": clip_id,
                "safe_csv_sha256": safe_hash,
                "quality_json_sha256": quality_hash,
            }
        )
        speakers[str(record.get("speaker_key") or "unknown")] += 1
        total_frames += int(record.get("frames") or 0)
        total_duration_sec += float(record.get("duration_sec") or 0.0)

    for record in manifests["failed"]:
        if record.get("status") != "quality_failed":
            raise PipelineError(
                f"{record_id(record)}: failed manifest contains a non-quality terminal result"
            )
    legacy_audit_binding = None
    legacy_audit_value = status.get("legacy_finalization_audit")
    legacy_audit_sha = status.get("legacy_finalization_audit_sha256")
    if legacy_audit_value is not None or legacy_audit_sha is not None:
        if not isinstance(legacy_audit_value, str) or not legacy_audit_value:
            raise PipelineError("retarget legacy finalization audit path is invalid")
        legacy_audit_path = Path(legacy_audit_value)
        if not legacy_audit_path.is_absolute():
            legacy_audit_path = (status_path.parent / legacy_audit_path).resolve()
        if not legacy_audit_path.is_file():
            raise FileNotFoundError(legacy_audit_path)
        actual_audit_sha = sha256_file(legacy_audit_path)
        if legacy_audit_sha != actual_audit_sha:
            raise PipelineError("retarget legacy finalization audit hash mismatch")
        legacy_audit_binding = tree_binding(legacy_audit_path)

    return {
        "inventory_count": len(inventory),
        "inventory_sha256": inventory_sha,
        "passed_count": counts["passed"],
        "failed_count": counts["failed"],
        "pending_count": 0,
        "accepted_frames": total_frames,
        "accepted_duration_sec": total_duration_sec,
        "speaker_counts": dict(sorted(speakers.items())),
        "accepted_artifacts_sha256": json_sha256(
            sorted(accepted_artifacts, key=lambda item: item["clip_id"])
        ),
        "status_sha256": sha256_file(status_path),
        "manifest_sha256": {
            name: sha256_file(path) for name, path in manifest_paths.items()
        },
        "retarget_provenance_status": (
            "legacy_finalized_snapshot_no_complete_run_contract"
            if legacy_audit_binding is not None
            else "terminal_snapshot_verified_no_separate_finalization_audit"
        ),
        "legacy_finalization_audit": legacy_audit_binding,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--qwen-checkpoint", type=Path, default=DEFAULT_QWEN_CHECKPOINT)
    parser.add_argument("--kimodo-dataset-dir", type=Path, default=DEFAULT_KIMODO_DATASET)
    parser.add_argument("--kimodo-split-checkpoint", type=Path, default=DEFAULT_KIMODO_SPLIT)
    parser.add_argument("--posttrain-output-dir", type=Path, default=DEFAULT_POSTTRAIN_OUTPUT)
    parser.add_argument("--expected-inventory-count", type=int, default=4570)
    parser.add_argument("--expected-passed-count", type=int, default=3495)
    parser.add_argument("--expected-failed-count", type=int, default=1075)
    parser.add_argument("--condition-batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--stage",
        action="append",
        choices=STAGES,
        help="Run only this stage; repeat for multiple stages. Default: all stages.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="Permit the explicitly unsafe, trajectory-only BEAT2 supervision path.",
    )
    return parser.parse_args(argv)


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    args.python = executable_path(args.python)
    for name in (
        "inventory",
        "output_root",
        "base_checkpoint",
        "qwen_checkpoint",
        "kimodo_dataset_dir",
        "kimodo_split_checkpoint",
        "posttrain_output_dir",
    ):
        setattr(args, name, Path(getattr(args, name)).absolute())
    for name in (
        "expected_inventory_count",
        "expected_passed_count",
        "expected_failed_count",
        "condition_batch_size",
        "steps",
        "batch_size",
    ):
        if int(getattr(args, name)) <= 0 and name not in {"expected_failed_count"}:
            raise ValueError(f"{name} must be positive")
    if args.expected_failed_count < 0:
        raise ValueError("expected_failed_count cannot be negative")
    if not 0.0 < float(args.lr) <= 1e-4:
        raise ValueError("lr must be in (0, 1e-4]")
    requested = tuple(stage for stage in STAGES if stage in set(args.stage or STAGES))
    args.requested_stages = requested
    if any(stage in requested for stage in ("cache", "posttrain")) and not args.allow_unreviewed:
        raise PipelineError(
            "the no-review semantics produced by this pipeline require --allow-unreviewed "
            "for cache/posttrain"
        )
    required = [args.inventory, SCRIPT_PATH]
    if "label" in requested:
        required.append(LABEL_SCRIPT)
    if "semantics" in requested:
        required.append(SEMANTICS_SCRIPT)
    if "cache" in requested:
        required.extend((CACHE_SCRIPT, args.base_checkpoint, args.qwen_checkpoint))
    if "posttrain" in requested:
        required.extend(
            (
                POSTTRAIN_SCRIPT,
                args.base_checkpoint,
                args.qwen_checkpoint,
                args.kimodo_dataset_dir,
                args.kimodo_split_checkpoint,
            )
        )
    for path in required:
        if not Path(path).exists():
            raise FileNotFoundError(path)
    return args


def pipeline_paths(args: argparse.Namespace) -> dict[str, Path]:
    state_root = args.output_root / "posttrain_pipeline"
    annotations = args.output_root / "annotations"
    semantics = args.output_root / "network_semantics"
    cache = args.output_root / "conditions_qwen_lora_v2.npz"
    return {
        "retarget": args.output_root / "retarget",
        "annotations": annotations,
        "drafts": annotations / "draft_prompts.jsonl",
        "semantics_root": semantics,
        "semantics": semantics / "network_semantics.jsonl",
        "cache": cache,
        "cache_metadata": cache.with_suffix(cache.suffix + ".json"),
        "state_root": state_root,
        "status": state_root / "pipeline_status.json",
        "stage_state": state_root / "stages",
        "logs": state_root / "logs",
        "posttrain_config": state_root / "posttrain_config_speaker_split_v1.json",
    }


def label_command(args: argparse.Namespace, paths: Mapping[str, Path]) -> list[str]:
    return [
        str(args.python),
        str(LABEL_SCRIPT),
        "--input-manifest",
        str(paths["retarget"] / "passed_manifest.jsonl"),
        "--output-dir",
        str(paths["annotations"]),
        "--resume",
    ]


def semantics_command(args: argparse.Namespace, paths: Mapping[str, Path]) -> list[str]:
    return [
        str(args.python),
        str(SEMANTICS_SCRIPT),
        "--input-annotations",
        str(paths["drafts"]),
        "--output-dir",
        str(paths["semantics_root"]),
    ]


def cache_command(args: argparse.Namespace, paths: Mapping[str, Path]) -> list[str]:
    command = [
        str(args.python),
        str(CACHE_SCRIPT),
        "cache-conditions",
        "--manifest",
        str(paths["semantics"]),
        "--base-checkpoint",
        str(args.base_checkpoint),
        "--qwen-checkpoint",
        str(args.qwen_checkpoint),
        "--output",
        str(paths["cache"]),
        "--device",
        str(args.device),
        "--batch-size",
        str(args.condition_batch_size),
    ]
    if args.allow_unreviewed:
        command.append("--allow-unreviewed")
    return command


def desired_posttrain_config(args: argparse.Namespace, paths: Mapping[str, Path]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "audio_policy": AUDIO_POLICY,
        "pipeline_metadata": {
            "artifact_kind": ARTIFACT_KIND,
            "audio_policy": AUDIO_POLICY,
            "speech_context_policy": SPEECH_CONTEXT_POLICY,
            "conditioning_modalities": list(CONDITIONING_MODALITIES),
            "split_policy": "strict_disjoint_speaker_and_source_group",
            "requested_split_fractions": dict(DEFAULT_SPLIT_FRACTIONS),
            "expected_full_dataset_split_counts": dict(EXPECTED_DEFAULT_SPLIT_COUNTS),
            "expected_full_dataset_speaker_split": dict(
                EXPECTED_DEFAULT_SPEAKER_SPLIT
            ),
        },
        "initial_checkpoint": str(args.base_checkpoint),
        "beat_manifest": str(paths["semantics"]),
        "condition_cache": str(paths["cache"]),
        "output_dir": str(args.posttrain_output_dir),
        "kimodo_dataset_dir": str(args.kimodo_dataset_dir),
        "kimodo_split_checkpoint": str(args.kimodo_split_checkpoint),
        "qwen_checkpoint": str(args.qwen_checkpoint),
        "allow_unreviewed": bool(args.allow_unreviewed),
        "allow_unsafe_condition_cache": False,
        "allow_unsafe_training_data": bool(args.allow_unreviewed),
        "training": {
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "validation_batch_size": 32,
            "phase_frame_choices": [64, 96, 128],
            "lr": float(args.lr),
            "minimum_lr_ratio": 0.1,
            "warmup_steps": 200,
            "weight_decay": 1e-4,
            "adam_eps": 1e-6,
            "max_grad_norm": 1.0,
            "ema_decay": 0.9995,
            "validation_interval": 100,
            "checkpoint_interval": 100,
            "log_interval": 25,
            "replay_evaluation_count": 128,
            "maximum_replay_regression_fraction": 0.03,
            "maximum_replay_regression_absolute": 0.02,
            "early_stopping_patience": 8,
            "early_stopping_min_delta": 1e-4,
            "split_fractions": dict(DEFAULT_SPLIT_FRACTIONS),
            "seed": 7,
            "device": str(args.device),
            "overwrite": False,
            "allow_unsafe_training_data": bool(args.allow_unreviewed),
            "loss": {
                "flow": 1.0,
                "position": 0.25,
                "body": 0.1,
                "velocity": 0.01,
                "acceleration": 0.0005,
            },
        },
    }


def posttrain_command(args: argparse.Namespace, paths: Mapping[str, Path]) -> list[str]:
    command = [
        str(args.python),
        str(POSTTRAIN_SCRIPT),
        "--config",
        str(paths["posttrain_config"]),
    ]
    if args.allow_unreviewed:
        command.append("--allow-unreviewed")
    return command


def _implementation_files(stage: str) -> tuple[Path, ...]:
    common = (SCRIPT_PATH,)
    if stage == "verify-retarget":
        return common
    if stage == "label":
        return (*common, LABEL_SCRIPT)
    if stage == "semantics":
        return (*common, SEMANTICS_SCRIPT, SEMANTICS_MODULE)
    if stage == "cache":
        return (
            *common,
            CACHE_SCRIPT,
            HEAD_MODULE,
            CROSS_MODAL_MODULE,
            CONDITIONING_MODULE,
            TRAINING_MODULE,
            SEMANTICS_MODULE,
            RETARGET_MODULE,
            RETARGET_18D_MODULE,
            MOTION_LATENT_MODULE,
            SEMANTIC_ADAPTER_MODULE,
        )
    if stage == "posttrain":
        return (
            *common,
            POSTTRAIN_SCRIPT,
            POSTTRAIN_MODULE,
            HEAD_MODULE,
            CROSS_MODAL_MODULE,
            CONDITIONING_MODULE,
            TRAINING_MODULE,
            SEMANTICS_MODULE,
            RETARGET_MODULE,
            RETARGET_18D_MODULE,
            MOTION_LATENT_MODULE,
            SEMANTIC_ADAPTER_MODULE,
        )
    raise KeyError(stage)


def _forbidden_metadata_keys(value: Any, prefix: str = "") -> list[str]:
    found = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if any(part in normalized for part in FORBIDDEN_TRAINING_METADATA_KEY_PARTS):
                found.append(child_prefix)
            found.extend(_forbidden_metadata_keys(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_metadata_keys(child, f"{prefix}[{index}]"))
    return found


def _index_records(
    records: Sequence[Mapping[str, Any]], name: str
) -> dict[str, Mapping[str, Any]]:
    ids = _manifest_ids(records, name)
    return dict(zip(ids, records))


def _resolved_output_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PipelineError(f"output record is missing {field}")
    return Path(value).absolute()


def _english_prompt(record: Mapping[str, Any]) -> str:
    prompt = record.get("canonical_prompt")
    if not isinstance(prompt, Mapping):
        raise PipelineError(f"{record_id(record)}: canonical_prompt must be an object")
    value = prompt.get("en")
    if not isinstance(value, str) or not value.strip():
        raise PipelineError(f"{record_id(record)}: canonical_prompt.en is empty")
    return value


def validate_label_outputs(paths: Mapping[str, Path], snapshot: Mapping[str, Any]) -> dict[str, Any]:
    summary = load_json(paths["annotations"] / "summary.json")
    drafts = read_jsonl(paths["drafts"])
    review_queue = read_jsonl(paths["annotations"] / "needs_human_review.jsonl")
    rejected = read_jsonl(paths["annotations"] / "rejected.jsonl")
    passed_manifest = paths["retarget"] / "passed_manifest.jsonl"
    passed = read_jsonl(passed_manifest)
    expected = int(snapshot["passed_count"])
    if (
        summary.get("input_records") != expected
        or summary.get("draft_records") != expected
        or summary.get("rejected_records") != 0
        or len(drafts) != expected
        or len(review_queue) != expected
        or len(passed) != expected
        or rejected
    ):
        raise PipelineError("label outputs do not cover every passed retarget record")
    if summary.get("prompt_provenance") != (
        "trajectory_only_ula_v2_18d_features_no_speech_semantics"
    ):
        raise PipelineError("label prompt provenance is not trajectory-only")
    passed_by_id = _index_records(passed, "passed retarget manifest")
    drafts_by_id = _index_records(drafts, "label drafts")
    review_by_id = _index_records(review_queue, "label human review queue")
    if set(drafts_by_id) != set(passed_by_id) or review_queue != drafts:
        raise PipelineError("label ids/review queue do not exactly match passed retarget ids")
    if set(review_by_id) != set(passed_by_id):
        raise PipelineError("label review queue ids do not match passed retarget ids")
    for clip_id, source in passed_by_id.items():
        draft = drafts_by_id[clip_id]
        trajectory = resolve_record_path(
            source.get("safe_csv"), passed_manifest, "safe_csv"
        ).absolute()
        quality = resolve_record_path(
            source.get("quality_json"), passed_manifest, "quality_json"
        ).absolute()
        if (
            _resolved_output_path(draft.get("trajectory_path"), "trajectory_path")
            != trajectory
            or draft.get("trajectory_sha256") != source.get("safe_csv_sha256")
            or _resolved_output_path(draft.get("quality_json"), "quality_json")
            != quality
            or draft.get("quality_json_sha256") != source.get("quality_json_sha256")
            or draft.get("retarget_quality_gate") != source.get("quality_gate")
            or draft.get("robot_contract") != RETARGET_CONTRACT
            or draft.get("speaker_key") != source.get("speaker_key")
        ):
            raise PipelineError(f"{clip_id}: label provenance differs from retarget input")
        if (
            draft.get("accepted_for_training") is not False
            or draft.get("manual_human_review_required") is not True
            or draft.get("decision") != "needs_human_review"
            or draft.get("prompt_provenance")
            != "trajectory_only_ula_v2_18d_features_no_speech_semantics"
        ):
            raise PipelineError(f"{clip_id}: label review/admission state is unsafe")
        _english_prompt(draft)
    return {"record_count": expected, "rejected_count": 0}


def validate_semantics_outputs(paths: Mapping[str, Path], snapshot: Mapping[str, Any]) -> dict[str, Any]:
    summary = load_json(paths["semantics_root"] / "summary.json")
    records = read_jsonl(paths["semantics"])
    for record in records:
        forbidden = _forbidden_metadata_keys(record)
        if forbidden:
            raise PipelineError(
                f"{record_id(record)}: training semantics contains disabled metadata "
                f"{forbidden[:3]}"
            )
    drafts = read_jsonl(paths["drafts"])
    review_queue = read_jsonl(paths["semantics_root"] / "human_review_queue.jsonl")
    emotion_supervised = read_jsonl(paths["semantics_root"] / "emotion_supervised.jsonl")
    rejected = read_jsonl(paths["semantics_root"] / "rejected.jsonl")
    expected = int(snapshot["passed_count"])
    if (
        summary.get("input_records") != expected
        or summary.get("output_records") != expected
        or summary.get("human_reviews_applied") != 0
        or len(records) != expected
        or len(drafts) != expected
        or len(review_queue) != expected
        or emotion_supervised
        or rejected
    ):
        raise PipelineError("semantic output count does not match passed retarget count")
    if summary.get("transcript_or_audio_metadata_used_for_labels") is not False:
        raise PipelineError("semantic summary does not certify the no-audio policy")
    if summary.get("behavior_supervised_records") != 0:
        raise PipelineError("automatic behavior candidates must remain unsupervised")
    if summary.get("emotion_supervised_records") != 0:
        raise PipelineError("emotion supervision must remain disabled")
    draft_by_id = _index_records(drafts, "label drafts")
    semantic_by_id = _index_records(records, "network semantics")
    queue_by_id = _index_records(review_queue, "semantic human review queue")
    if set(semantic_by_id) != set(draft_by_id) or set(queue_by_id) != set(draft_by_id):
        raise PipelineError("semantic ids do not exactly match label draft ids")
    for clip_id, record in semantic_by_id.items():
        draft = draft_by_id[clip_id]
        if record.get("behavior_supervision_mask") is not False:
            raise PipelineError(f"{clip_id}: behavior supervision is unexpectedly enabled")
        if record.get("emotion_supervision_mask") is not False or record.get("emotion_id") is not None:
            raise PipelineError(f"{clip_id}: emotion supervision is unexpectedly enabled")
        source_clip = str(draft.get("source_clip_id") or clip_id.split("_f", 1)[0])
        expected_group = f"beat2/{draft.get('speaker_key')}/{source_clip}"
        for field in (
            "canonical_prompt",
            "trajectory_path",
            "trajectory_sha256",
            "quality_json",
            "quality_json_sha256",
            "speaker_key",
            "robot_contract",
            "source_clip_id",
        ):
            if record.get(field) != draft.get(field):
                raise PipelineError(f"{clip_id}: semantic {field} differs from its label draft")
        if (
            record.get("source_group_key") != expected_group
            or record.get("review_required") is not True
            or record.get("network_semantic_supervision_ready") is not False
            or record.get("behavior_review_status") != "candidate_unreviewed"
            or record.get("emotion_review_status") != "unresolved"
        ):
            raise PipelineError(f"{clip_id}: semantic review/provenance contract mismatch")
    return {
        "record_count": expected,
        "behavior_supervised_count": 0,
        "emotion_supervised_count": 0,
        "audio_policy": AUDIO_POLICY,
    }


def validate_cache_outputs(
    args: argparse.Namespace, paths: Mapping[str, Path], snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        import numpy as np
        from upper_body_skeleton.ula_v2_18d_head import load_condition_cache
    except ImportError as error:
        raise PipelineError("condition cache validation requires NumPy and ULA V2") from error
    try:
        deep_clip_ids, deep_prompts, deep_conditions, deep_metadata = (
            load_condition_cache(paths["cache"], allow_unsafe_metadata=False)
        )
    except Exception as error:
        raise PipelineError(f"deep condition-cache validation failed: {error}") from error
    if deep_metadata.get("unsafe_condition_cache") is not False:
        raise PipelineError("condition cache was admitted through an unsafe metadata path")
    metadata = load_json(paths["cache_metadata"])
    semantics = read_jsonl(paths["semantics"])
    expected = int(snapshot["passed_count"])
    if (
        metadata.get("artifact_kind") != "ula_v2_qwen_motion_condition_cache"
        or metadata.get("schema_version") != 2
        or metadata.get("count") != expected
        or metadata.get("condition_dim") != CONDITION_DIM
    ):
        raise PipelineError("condition cache metadata contract mismatch")
    if metadata.get("cache_sha256") != sha256_file(paths["cache"]):
        raise PipelineError("condition cache payload hash mismatch")
    if metadata.get("generator_checkpoint_sha256") != sha256_file(args.base_checkpoint):
        raise PipelineError("condition cache targets a different generator checkpoint")
    if metadata.get("qwen_checkpoint_sha256") != sha256_file(args.qwen_checkpoint):
        raise PipelineError("condition cache targets a different Qwen checkpoint")
    if metadata.get("behavior_supervised_count") != 0 or metadata.get("emotion_supervised_count") != 0:
        raise PipelineError("condition cache unexpectedly contains supervised behavior/emotion")
    if (
        metadata.get("behavior_unsupervised_count") != expected
        or metadata.get("emotion_unresolved_count") != expected
        or _resolved_output_path(
            metadata.get("generator_checkpoint"), "generator_checkpoint"
        )
        != args.base_checkpoint
        or _resolved_output_path(metadata.get("qwen_checkpoint"), "qwen_checkpoint")
        != args.qwen_checkpoint
    ):
        raise PipelineError("condition cache provenance/count contract mismatch")
    semantic_ids = _manifest_ids(semantics, "network semantics")
    semantic_prompts = [_english_prompt(record) for record in semantics]
    if deep_clip_ids != semantic_ids or deep_prompts != semantic_prompts:
        raise PipelineError("deep-validated cache ids/prompts differ from network semantics")
    if deep_conditions.shape != (expected, CONDITION_DIM):
        raise PipelineError("deep-validated condition cache shape is invalid")
    episodes = metadata.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != expected:
        raise PipelineError("condition cache episode metadata count is invalid")
    with np.load(paths["cache"], allow_pickle=False) as payload:
        required_arrays = {
            "clip_ids",
            "prompts",
            "conditions",
            "behavior_ids",
            "behavior_review_statuses",
            "behavior_supervision_mask",
            "emotion_ids",
            "emotion_supervision_mask",
            "style_features",
            "style_controls",
        }
        if set(payload.files) != required_arrays:
            raise PipelineError("condition cache array set is incomplete or unexpected")
        clip_ids = payload["clip_ids"].astype(str).tolist()
        prompts = payload["prompts"].astype(str).tolist()
        conditions = payload["conditions"]
        behavior_mask = payload["behavior_supervision_mask"]
        emotion_mask = payload["emotion_supervision_mask"]
        style_features = payload["style_features"]
        style_controls = payload["style_controls"]
        if clip_ids != semantic_ids or prompts != semantic_prompts:
            raise PipelineError("condition cache ids/prompts differ from network semantics")
        if len(set(clip_ids)) != expected:
            raise PipelineError("condition cache contains duplicate clip ids")
        if conditions.shape != (expected, CONDITION_DIM) or not np.isfinite(conditions).all():
            raise PipelineError(f"condition cache has unexpected shape {conditions.shape}")
        if (
            behavior_mask.dtype != np.dtype(np.bool_)
            or emotion_mask.dtype != np.dtype(np.bool_)
            or bool(behavior_mask.any())
            or bool(emotion_mask.any())
        ):
            raise PipelineError("condition cache supervision masks must remain all-zero")
        if (
            style_features.shape != (expected, 3)
            or style_controls.shape != (expected, 3)
            or not np.isfinite(style_features).all()
            or not np.isfinite(style_controls).all()
        ):
            raise PipelineError("condition cache trajectory style arrays are invalid")
        if bool(conditions[:, 92:133].any()):
            raise PipelineError("unsupervised behavior/emotion condition slices must be zero")
        behavior_ids = payload["behavior_ids"].astype(str).tolist()
        review_statuses = payload["behavior_review_statuses"].astype(str).tolist()
        emotion_ids = payload["emotion_ids"].astype(str).tolist()
        for index, (semantic, episode) in enumerate(zip(semantics, episodes)):
            prompt_sha = hashlib.sha256(prompts[index].encode("utf-8")).hexdigest()
            if (
                not isinstance(episode, Mapping)
                or episode.get("clip_id") != clip_ids[index]
                or episode.get("prompt_sha256") != prompt_sha
                or episode.get("behavior_supervision_mask") is not False
                or episode.get("emotion_supervision_mask") is not False
                or behavior_ids[index] != semantic.get("behavior_id")
                or review_statuses[index] != semantic.get("behavior_review_status")
                or emotion_ids[index] != ""
            ):
                raise PipelineError("condition cache episode metadata differs from semantics")
    return {"record_count": expected, "condition_dim": CONDITION_DIM, "audio_policy": AUDIO_POLICY}


def validate_posttrain_outputs(
    args: argparse.Namespace, paths: Mapping[str, Path], snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    summary_path = args.posttrain_output_dir / "training_summary.json"
    split_path = args.posttrain_output_dir / "split_manifest.json"
    summary = load_json(summary_path)
    split = load_json(split_path)
    if summary.get("target_steps") != int(args.steps):
        raise PipelineError("posttrain target step count does not match the pipeline config")
    completed = int(summary.get("completed_steps") or 0)
    stopped_early = summary.get("stopped_early") is True
    if completed <= 0 or (completed != int(args.steps) and not stopped_early):
        raise PipelineError("posttrain is neither target-complete nor explicitly early-stopped")
    if args.allow_unreviewed:
        if summary.get("formal_release_eligible") is not False:
            raise PipelineError("unreviewed posttrain must not be formal-release eligible")
        if summary.get("artifact_status") != "experimental_unreviewed_unsafe":
            raise PipelineError("unreviewed posttrain artifact status is not fail-closed")
    split_payload = dict(split)
    recorded_split_sha = split_payload.pop("sha256", None)
    encoded_split = json.dumps(
        split_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    actual_split_sha = hashlib.sha256(encoded_split).hexdigest()
    if (
        recorded_split_sha != actual_split_sha
        or recorded_split_sha != summary.get("split_contract_sha256")
    ):
        raise PipelineError("posttrain split manifest hash does not match its summary")
    if (
        split.get("contract_type") != "speaker_source_group_strict_split"
        or split.get("contract_version") != 1
        or split.get("fractions") != DEFAULT_SPLIT_FRACTIONS
        or split.get("seed") != 7
    ):
        raise PipelineError("posttrain split policy differs from the requested 0.8/0.1/0.1")
    episodes = split.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != int(snapshot["passed_count"]):
        raise PipelineError("posttrain split does not cover the accepted BEAT2 episodes")
    split_ids = _manifest_ids(episodes, "posttrain split episodes")
    semantic_ids = _manifest_ids(read_jsonl(paths["semantics"]), "network semantics")
    if set(split_ids) != set(semantic_ids):
        raise PipelineError("posttrain split ids differ from network semantics")
    calculated_counts = Counter(str(item.get("split")) for item in episodes)
    if dict(split.get("counts") or {}) != {
        name: calculated_counts[name] for name in ("train", "validation", "test")
    }:
        raise PipelineError("posttrain split counts do not match its episodes")
    speaker_to_split = split.get("speaker_to_split")
    source_to_split = split.get("source_group_to_split")
    if not isinstance(speaker_to_split, dict) or not isinstance(source_to_split, dict):
        raise PipelineError("posttrain split is missing speaker/source assignments")
    observed_sources = set()
    for item in episodes:
        speaker = item.get("speaker_key")
        source_group = item.get("source_group_key")
        observed_sources.add(source_group)
        if (
            speaker_to_split.get(speaker) != item.get("split")
            or source_to_split.get(source_group) != item.get("split")
        ):
            raise PipelineError("posttrain split leaks or misassigns a speaker/source group")
    if observed_sources != set(source_to_split):
        raise PipelineError("posttrain split source-group index differs from its episodes")
    if int(snapshot["passed_count"]) == sum(EXPECTED_DEFAULT_SPLIT_COUNTS.values()):
        if (
            split.get("counts") != EXPECTED_DEFAULT_SPLIT_COUNTS
            or speaker_to_split != EXPECTED_DEFAULT_SPEAKER_SPLIT
        ):
            raise PipelineError("full BEAT2 strict speaker split changed unexpectedly")
    provenance = summary.get("data_provenance") or {}
    cache = provenance.get("condition_cache") or {}
    if cache.get("cache_sha256") != sha256_file(paths["cache"]):
        raise PipelineError("posttrain summary is not bound to the selected condition cache")
    if provenance.get("beat_counts") != split.get("counts"):
        raise PipelineError("posttrain provenance and split counts differ")
    replay_guard = summary.get("final_replay_regression_guard") or {}
    release = summary.get("formal_release_decision") or {}
    if (
        replay_guard.get("applicable") is not True
        or replay_guard.get("passed") is not True
        or release.get("replay_regression_guard") != replay_guard
        or release.get("replay_regression_guard_required") is not True
        or release.get("artifact_status") != summary.get("artifact_status")
        or release.get("formal_release_eligible")
        != summary.get("formal_release_eligible")
    ):
        raise PipelineError("posttrain Kimodo replay/formal-release guard is invalid")

    best_path = args.posttrain_output_dir / "ula_fm_checkpoint.pt"
    last_path = args.posttrain_output_dir / "last.pt"
    if (
        _resolved_output_path(summary.get("checkpoint"), "checkpoint") != best_path
        or _resolved_output_path(summary.get("last_checkpoint"), "last_checkpoint")
        != last_path
    ):
        raise PipelineError("posttrain summary checkpoint paths differ from pipeline outputs")
    try:
        import torch
    except ImportError as error:
        raise PipelineError("posttrain checkpoint validation requires PyTorch") from error
    base_checkpoint_sha = sha256_file(args.base_checkpoint)
    for name, checkpoint_path, expected_step, require_training_state in (
        ("best", best_path, int(summary.get("best_step") or 0), False),
        ("last", last_path, completed, True),
    ):
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
        except Exception as error:
            raise PipelineError(f"cannot load {name} posttrain checkpoint: {error}") from error
        if not isinstance(checkpoint, Mapping):
            raise PipelineError(f"{name} posttrain checkpoint must contain a mapping")
        training_contract = checkpoint.get("training_contract") or {}
        checkpoint_data_contract = checkpoint.get("posttrain_data_contract") or {}
        data_payload = dict(checkpoint_data_contract)
        recorded_data_sha = data_payload.pop("sha256", None)
        actual_data_sha = hashlib.sha256(
            json.dumps(
                data_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        source = checkpoint.get("posttrain_source") or {}
        if (
            checkpoint.get("posttrain_artifact_kind") != POSTTRAIN_ARTIFACT_KIND
            or checkpoint.get("action_dim") != 18
            or checkpoint.get("condition_dim") != CONDITION_DIM
            or int(checkpoint.get("posttrain_step") or -1) != expected_step
            or int(checkpoint.get("global_step") or -1)
            != int(source.get("source_global_step") or 0) + expected_step
            or checkpoint.get("artifact_status") != summary.get("artifact_status")
            or checkpoint.get("formal_release_eligible")
            != summary.get("formal_release_eligible")
            or source.get("checkpoint_sha256") != base_checkpoint_sha
            or recorded_data_sha != actual_data_sha
            or recorded_data_sha != summary.get("data_contract_sha256")
            or checkpoint_data_contract.get("split_contract_sha256")
            != recorded_split_sha
            or checkpoint.get("posttrain_split_contract") != split
            or (checkpoint.get("posttrain_config") or {}).get("split_fractions")
            != DEFAULT_SPLIT_FRACTIONS
            or training_contract.get("replay_regression_guard") != replay_guard
            or training_contract.get("formal_release_decision") != release
        ):
            raise PipelineError(f"{name} posttrain checkpoint contract mismatch")
        state = checkpoint.get("training_state")
        if require_training_state:
            if not isinstance(state, Mapping) or set(state) != POSTTRAIN_TRAINING_STATE_KEYS:
                raise PipelineError("last posttrain checkpoint lacks exact resume state")
        elif "training_state" in checkpoint:
            raise PipelineError("release posttrain checkpoint must not contain training state")
    return {
        "completed_steps": completed,
        "best_step": int(summary.get("best_step") or 0),
        "artifact_status": summary.get("artifact_status"),
        "formal_release_eligible": summary.get("formal_release_eligible"),
        "audio_policy": AUDIO_POLICY,
    }


def output_paths_for_stage(
    stage: str, args: argparse.Namespace, paths: Mapping[str, Path]
) -> tuple[Path, ...]:
    if stage == "verify-retarget":
        return ()
    if stage == "label":
        return (
            paths["drafts"],
            paths["annotations"] / "needs_human_review.jsonl",
            paths["annotations"] / "rejected.jsonl",
            paths["annotations"] / "summary.json",
            paths["annotations"] / "text/en",
            paths["annotations"] / "text/zh",
        )
    if stage == "semantics":
        return tuple(
            paths["semantics_root"] / name
            for name in (
                "network_semantics.jsonl",
                "human_review_queue.jsonl",
                "emotion_supervised.jsonl",
                "rejected.jsonl",
                "summary.json",
            )
        )
    if stage == "cache":
        return (paths["cache"], paths["cache_metadata"])
    if stage == "posttrain":
        return (
            args.posttrain_output_dir / "ula_fm_checkpoint.pt",
            args.posttrain_output_dir / "last.pt",
            args.posttrain_output_dir / "split_manifest.json",
            args.posttrain_output_dir / "training_summary.json",
        )
    raise KeyError(stage)


def input_paths_for_stage(
    stage: str, args: argparse.Namespace, paths: Mapping[str, Path]
) -> tuple[Path, ...]:
    if stage == "verify-retarget":
        return (
            args.inventory,
            paths["retarget"] / "status.json",
            paths["retarget"] / "passed_manifest.jsonl",
            paths["retarget"] / "failed_manifest.jsonl",
            paths["retarget"] / "pending_manifest.jsonl",
            paths["retarget"] / "excluded_manifest.jsonl",
        )
    if stage == "label":
        return (args.inventory, paths["retarget"] / "passed_manifest.jsonl")
    if stage == "semantics":
        return (paths["drafts"], paths["annotations"] / "summary.json")
    if stage == "cache":
        return (
            paths["semantics"],
            paths["semantics_root"] / "summary.json",
            args.base_checkpoint,
            args.qwen_checkpoint,
        )
    if stage == "posttrain":
        return (
            paths["semantics"],
            paths["cache"],
            paths["cache_metadata"],
            args.base_checkpoint,
            args.qwen_checkpoint,
            args.kimodo_dataset_dir,
            args.kimodo_split_checkpoint,
        )
    raise KeyError(stage)


def command_for_stage(
    stage: str, args: argparse.Namespace, paths: Mapping[str, Path]
) -> list[str]:
    if stage == "verify-retarget":
        return ["internal:verify-retarget"]
    if stage == "label":
        return label_command(args, paths)
    if stage == "semantics":
        return semantics_command(args, paths)
    if stage == "cache":
        return cache_command(args, paths)
    if stage == "posttrain":
        return posttrain_command(args, paths)
    raise KeyError(stage)


def validator_for_stage(stage: str) -> Callable[..., dict[str, Any]]:
    return {
        "label": validate_label_outputs,
        "semantics": validate_semantics_outputs,
        "cache": validate_cache_outputs,
        "posttrain": validate_posttrain_outputs,
    }[stage]


def stage_contract(
    stage: str,
    args: argparse.Namespace,
    paths: Mapping[str, Path],
    snapshot: Mapping[str, Any],
    *,
    planned_outputs: Mapping[Path, tuple[str, str]] | None = None,
) -> tuple[dict[str, Any], str]:
    planned_outputs = planned_outputs or {}
    inputs = []
    for path in input_paths_for_stage(stage, args, paths):
        absolute = path.absolute()
        if path.exists():
            inputs.append(tree_binding(path))
        elif absolute in planned_outputs:
            producer, digest = planned_outputs[absolute]
            inputs.append(planned_binding(path, producer, digest))
        else:
            raise FileNotFoundError(path)
    command = command_for_stage(stage, args, paths)
    virtual_inputs: dict[str, Any] = {
        "retarget_accepted_artifacts_sha256": snapshot["accepted_artifacts_sha256"],
        "retarget_passed_count": snapshot["passed_count"],
    }
    if stage == "posttrain":
        config = desired_posttrain_config(args, paths)
        virtual_inputs["posttrain_config"] = config
        virtual_inputs["posttrain_config_sha256"] = json_sha256(config)
    contract = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": f"{ARTIFACT_KIND}_stage_contract",
        "stage": stage,
        "audio_policy": AUDIO_POLICY,
        "speech_context_policy": SPEECH_CONTEXT_POLICY,
        "conditioning_modalities": list(CONDITIONING_MODALITIES),
        "python": tree_binding(args.python),
        "command": command,
        "command_sha256": json_sha256(command),
        "inputs": inputs,
        "virtual_inputs": virtual_inputs,
        "implementations": [tree_binding(path) for path in _implementation_files(stage)],
    }
    return contract, json_sha256(contract)


def receipt_path(stage: str, paths: Mapping[str, Path]) -> Path:
    return paths["stage_state"] / f"{stage}.json"


def output_bindings(output_paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [tree_binding(path) for path in output_paths]


def decide_stage_action(
    stage: str,
    contract: Mapping[str, Any],
    contract_sha: str,
    output_paths: Sequence[Path],
    paths: Mapping[str, Path],
) -> tuple[str, dict[str, Any] | None]:
    path = receipt_path(stage, paths)
    if not path.is_file():
        return "run", None
    receipt = load_json(path)
    if (
        receipt.get("stage_contract_sha256") != contract_sha
        or receipt.get("stage_contract") != contract
    ):
        raise PipelineError(f"stage {stage} contract/input hash drift; refusing reuse")
    state = receipt.get("state")
    if state == "succeeded":
        current = output_bindings(output_paths)
        if current != receipt.get("outputs"):
            raise PipelineError(f"stage {stage} output hash drift; refusing reuse")
        return "reuse", receipt
    if state in {"running", "failed"}:
        return "resume" if stage == "posttrain" else "retry", receipt
    raise PipelineError(f"stage {stage} receipt has unsupported state {state!r}")


def _ensure_posttrain_config(args: argparse.Namespace, paths: Mapping[str, Path]) -> None:
    expected = desired_posttrain_config(args, paths)
    path = paths["posttrain_config"]
    if path.exists():
        if load_json(path) != expected:
            raise PipelineError("posttrain config drift; refusing to overwrite fixed config")
        return
    atomic_json(path, expected)


def resumed_command(
    stage: str,
    command: Sequence[str],
    action: str,
    args: argparse.Namespace,
) -> list[str]:
    actual = list(command)
    if stage != "posttrain" or action not in {"resume", "retry"}:
        return actual
    last = args.posttrain_output_dir / "last.pt"
    if not last.is_file():
        raise PipelineError("interrupted posttrain has no last.pt; refusing unsafe restart")
    step = posttrain_checkpoint_step(last)
    if step >= int(args.steps):
        raise PipelineError(
            "posttrain last.pt already reached the target step but terminal outputs are "
            "incomplete or invalid; refusing a zero-step resume"
        )
    actual.extend(("--resume-from", str(last)))
    return actual


def posttrain_checkpoint_step(path: str | Path) -> int:
    try:
        import torch
    except ImportError as error:
        raise PipelineError("posttrain resume validation requires PyTorch") from error
    try:
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    except Exception as error:
        raise PipelineError(f"cannot inspect posttrain resume checkpoint {path}: {error}") from error
    if not isinstance(checkpoint, Mapping):
        raise PipelineError("posttrain resume checkpoint must contain a mapping")
    try:
        step = int(checkpoint["posttrain_step"])
    except (KeyError, TypeError, ValueError) as error:
        raise PipelineError("posttrain resume checkpoint is missing posttrain_step") from error
    if step < 0:
        raise PipelineError("posttrain resume checkpoint has a negative step")
    return step


def subprocess_environment() -> dict[str, str]:
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(PROJECT_ROOT) if not existing else str(PROJECT_ROOT) + os.pathsep + existing
    )
    return environment


class Pipeline:
    def __init__(self, args: argparse.Namespace):
        self.args = resolve_args(args)
        self.paths = pipeline_paths(self.args)
        self.snapshot = verify_retarget_snapshot(self.args)
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")

    def _validate_outputs(self, stage: str) -> dict[str, Any]:
        if stage == "verify-retarget":
            return dict(self.snapshot)
        validator = validator_for_stage(stage)
        if stage in {"label", "semantics"}:
            return validator(self.paths, self.snapshot)
        return validator(self.args, self.paths, self.snapshot)

    def _decide_stage_action(
        self,
        stage: str,
        contract: Mapping[str, Any],
        digest: str,
        outputs: Sequence[Path],
    ) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
        action, previous = decide_stage_action(
            stage, contract, digest, outputs, self.paths
        )
        existing = [path.exists() for path in outputs]
        if action == "run" and previous is None:
            if outputs and any(existing) and not all(existing):
                raise PipelineError(
                    f"stage {stage} has partial outputs without a pipeline receipt; "
                    "refusing overwrite"
                )
            if outputs and all(existing):
                verification = self._validate_outputs(stage)
                return "adopt", None, verification
            if (
                stage == "posttrain"
                and self.args.posttrain_output_dir.is_dir()
                and any(self.args.posttrain_output_dir.iterdir())
            ):
                raise PipelineError(
                    "posttrain output is non-empty without a complete adoptable artifact set"
                )
            return action, previous, None
        if action in {"resume", "retry"} and outputs and all(existing):
            try:
                verification = self._validate_outputs(stage)
            except Exception:
                if stage == "posttrain":
                    last = self.args.posttrain_output_dir / "last.pt"
                    if last.is_file() and posttrain_checkpoint_step(last) >= int(
                        self.args.steps
                    ):
                        raise PipelineError(
                            "posttrain reached its target step but terminal artifacts do not "
                            "pass validation; refusing retry"
                        )
            else:
                return "finalize_existing", previous, verification
        if action == "resume":
            last = self.args.posttrain_output_dir / "last.pt"
            if last.is_file() and posttrain_checkpoint_step(last) >= int(self.args.steps):
                raise PipelineError(
                    "posttrain reached its target step but terminal artifacts are incomplete; "
                    "refusing retry"
                )
        return action, previous, None

    def _record_existing_artifacts(
        self,
        stage: str,
        action: str,
        previous: Mapping[str, Any] | None,
        contract: Mapping[str, Any],
        digest: str,
        outputs: Sequence[Path],
        verification: Mapping[str, Any],
    ) -> dict[str, Any]:
        if stage == "posttrain":
            _ensure_posttrain_config(self.args, self.paths)
        attempts = list((previous or {}).get("attempts") or [])
        attempt_id = f"{self.run_id}_{len(attempts) + 1:02d}"
        log_path = self.paths["logs"] / f"{attempt_id}_{stage}_adoption.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = command_for_stage(stage, self.args, self.paths)
        atomic_text(
            log_path,
            "audio_policy="
            + AUDIO_POLICY
            + "\nexecution=not_run_existing_artifacts_validated\n"
            + "stage_contract_sha256="
            + digest
            + "\ncommand_not_executed="
            + json.dumps(command, ensure_ascii=False)
            + "\n",
        )
        now = utc_now()
        attempts.append(
            {
                "attempt_id": attempt_id,
                "attempt_kind": "existing_artifact_adoption",
                "state": "succeeded",
                "started_at": now,
                "finished_at": now,
                "command": command,
                "command_sha256": json_sha256(command),
                "command_executed": False,
                "returncode": None,
                "log_path": str(log_path.resolve()),
                "log_sha256": sha256_file(log_path),
            }
        )
        finished = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": f"{ARTIFACT_KIND}_stage_receipt",
            "stage": stage,
            "state": "succeeded",
            "audio_policy": AUDIO_POLICY,
            "stage_contract": dict(contract),
            "stage_contract_sha256": digest,
            "execution_provenance": (
                "validated_existing_artifacts_not_original_execution"
            ),
            "adoption_action": action,
            "finished_at": now,
            "outputs": output_bindings(outputs),
            "verification": dict(verification),
            "attempts": attempts,
        }
        atomic_json(receipt_path(stage, self.paths), finished)
        return {
            "stage": stage,
            "state": "adopted" if action == "adopt" else "finalized_existing",
            "stage_contract_sha256": digest,
            "verification": dict(verification),
            "receipt": str(receipt_path(stage, self.paths).resolve()),
        }

    def dry_run_plan(self) -> dict[str, Any]:
        planned: dict[Path, tuple[str, str]] = {}
        stages = []
        for stage in self.args.requested_stages:
            contract, digest = stage_contract(
                stage,
                self.args,
                self.paths,
                self.snapshot,
                planned_outputs=planned,
            )
            has_planned_input = any(
                item.get("kind") == "planned_output" for item in contract["inputs"]
            )
            if has_planned_input:
                action = "run_after_dependencies"
            else:
                action, _, _ = self._decide_stage_action(
                    stage,
                    contract,
                    digest,
                    output_paths_for_stage(stage, self.args, self.paths),
                )
            command = command_for_stage(stage, self.args, self.paths)
            if stage == "posttrain" and action == "resume":
                command = resumed_command(stage, command, action, self.args)
            stages.append(
                {
                    "stage": stage,
                    "action": action,
                    "command": command,
                    "stage_contract_sha256": digest,
                    "input_bindings": contract["inputs"],
                    "output_paths": [
                        str(path.absolute())
                        for path in output_paths_for_stage(stage, self.args, self.paths)
                    ],
                }
            )
            for path in output_paths_for_stage(stage, self.args, self.paths):
                planned[path.absolute()] = (stage, digest)
        plan = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": f"{ARTIFACT_KIND}_dry_run_plan",
            "dry_run": True,
            "audio_policy": AUDIO_POLICY,
            "speech_context_policy": SPEECH_CONTEXT_POLICY,
            "conditioning_modalities": list(CONDITIONING_MODALITIES),
            "requested_stages": list(self.args.requested_stages),
            "retarget_verification": self.snapshot,
            "stages": stages,
        }
        plan["plan_sha256"] = json_sha256(plan)
        return plan

    def _run_stage(
        self, stage: str, contract: dict[str, Any], digest: str
    ) -> dict[str, Any]:
        outputs = output_paths_for_stage(stage, self.args, self.paths)
        action, previous, existing_verification = self._decide_stage_action(
            stage, contract, digest, outputs
        )
        if action == "reuse":
            verification = self._validate_outputs(stage)
            return {
                "stage": stage,
                "state": "reused",
                "stage_contract_sha256": digest,
                "verification": verification,
                "receipt": str(receipt_path(stage, self.paths).resolve()),
            }

        if action in {"adopt", "finalize_existing"}:
            if existing_verification is None:
                raise AssertionError("existing artifact adoption requires validation")
            return self._record_existing_artifacts(
                stage,
                action,
                previous,
                contract,
                digest,
                outputs,
                existing_verification,
            )

        if stage == "posttrain":
            _ensure_posttrain_config(self.args, self.paths)
        command = command_for_stage(stage, self.args, self.paths)
        actual_command = resumed_command(stage, command, action, self.args)
        attempts = list((previous or {}).get("attempts") or [])
        attempt_id = f"{self.run_id}_{len(attempts) + 1:02d}"
        log_path = self.paths["logs"] / f"{attempt_id}_{stage}.log"
        attempt = {
            "attempt_id": attempt_id,
            "state": "running",
            "started_at": utc_now(),
            "command": actual_command,
            "command_sha256": json_sha256(actual_command),
            "log_path": str(log_path.resolve()),
        }
        attempts.append(attempt)
        running = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": f"{ARTIFACT_KIND}_stage_receipt",
            "stage": stage,
            "state": "running",
            "audio_policy": AUDIO_POLICY,
            "stage_contract": contract,
            "stage_contract_sha256": digest,
            "attempts": attempts,
        }
        atomic_json(receipt_path(stage, self.paths), running)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        returncode = 0
        try:
            with log_path.open("w", encoding="utf-8") as log:
                log.write("audio_policy=" + AUDIO_POLICY + "\n")
                log.write("stage_contract_sha256=" + digest + "\n")
                log.write("command=" + json.dumps(actual_command, ensure_ascii=False) + "\n")
                log.flush()
                if stage == "verify-retarget":
                    log.write(json.dumps(self.snapshot, ensure_ascii=False, sort_keys=True) + "\n")
                else:
                    completed = subprocess.run(
                        actual_command,
                        cwd=PROJECT_ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                        env=subprocess_environment(),
                    )
                    returncode = int(completed.returncode)
            attempt.update(
                {
                    "state": "succeeded" if returncode == 0 else "failed",
                    "returncode": returncode,
                    "finished_at": utc_now(),
                    "log_sha256": sha256_file(log_path),
                }
            )
            if returncode != 0:
                raise StageCommandError(stage, returncode)
            verification = self._validate_outputs(stage)
            finished = {
                **running,
                "state": "succeeded",
                "finished_at": utc_now(),
                "outputs": output_bindings(outputs),
                "verification": verification,
                "attempts": attempts,
            }
            atomic_json(receipt_path(stage, self.paths), finished)
            return {
                "stage": stage,
                "state": "succeeded",
                "stage_contract_sha256": digest,
                "verification": verification,
                "receipt": str(receipt_path(stage, self.paths).resolve()),
            }
        except Exception as error:
            attempt.update(
                {
                    "state": "failed",
                    "returncode": returncode or 1,
                    "finished_at": utc_now(),
                    "error": str(error),
                    "log_sha256": sha256_file(log_path) if log_path.is_file() else None,
                }
            )
            atomic_json(
                receipt_path(stage, self.paths),
                {
                    **running,
                    "state": "failed",
                    "finished_at": utc_now(),
                    "error": {"type": type(error).__name__, "message": str(error)},
                    "attempts": attempts,
                },
            )
            raise

    def run(self) -> int:
        self.paths["state_root"].mkdir(parents=True, exist_ok=True)
        status = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": ARTIFACT_KIND,
            "run_id": self.run_id,
            "state": "running",
            "started_at": utc_now(),
            "updated_at": utc_now(),
            "audio_policy": AUDIO_POLICY,
            "speech_context_policy": SPEECH_CONTEXT_POLICY,
            "conditioning_modalities": list(CONDITIONING_MODALITIES),
            "requested_stages": list(self.args.requested_stages),
            "retarget_verification": self.snapshot,
            "stages": {},
        }
        atomic_json(self.paths["status"], status)
        try:
            for stage in self.args.requested_stages:
                contract, digest = stage_contract(
                    stage, self.args, self.paths, self.snapshot
                )
                result = self._run_stage(stage, contract, digest)
                status["stages"][stage] = result
                status["updated_at"] = utc_now()
                atomic_json(self.paths["status"], status)
        except Exception as error:
            status.update(
                {
                    "state": "failed_resumable",
                    "updated_at": utc_now(),
                    "finished_at": utc_now(),
                    "failure": {"type": type(error).__name__, "message": str(error)},
                }
            )
            atomic_json(self.paths["status"], status)
            raise
        status.update(
            {"state": "finished", "updated_at": utc_now(), "finished_at": utc_now()}
        )
        atomic_json(self.paths["status"], status)
        print(json.dumps(status, ensure_ascii=False, sort_keys=True, indent=2))
        return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        pipeline = Pipeline(args)
        if pipeline.args.dry_run:
            print(
                json.dumps(
                    pipeline.dry_run_plan(),
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
            )
            return 0
        return pipeline.run()
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return int(error.returncode) if isinstance(error, StageCommandError) else 1


if __name__ == "__main__":
    raise SystemExit(main())
