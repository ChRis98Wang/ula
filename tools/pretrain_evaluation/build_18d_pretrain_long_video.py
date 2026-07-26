#!/usr/bin/env python3
"""Build a provenance-bound long MuJoCo video after the 18D motion-only pretrain.

This tool assembles real model samples only.  It never runs inference and it
never manufactures placeholder trajectories.  Every generated CSV must bind to
the completed checkpoint and held-out split before rendering starts.

Captions have deliberately narrow semantics:

* ``reviewed_robot_observable_text`` is accepted only through an independent
  blind review record bound to the generated CSV.
* ``motion_only_metadata`` is displayed with an unavoidable disclosure that it
  was metadata, not a text condition or a verified semantic interpretation.

The final MP4 contains two selectable subtitle tracks.  Sidecar SRT and chapter
files are retained beside a machine-readable, hash-bound index.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import imageio_ffmpeg
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.human_motion_review.render_beat2_annotation_review import (
    CAMERA_LOOKAT_Z_OFFSET,
    CAMERA_MARGIN,
    DEFAULT_URDF,
    validate_render_summary,
    validate_video,
)
from upper_body_skeleton.retarget_v2_18d import JOINT_ORDER_18D


SCHEMA_VERSION = "1.0.0"
ARTIFACT_KIND = "ula_v2_18d_motion_only_pretrain_long_video_v1"
SEGMENT_ARTIFACT_KIND = "ula_v2_18d_motion_only_pretrain_generation_v1"
REVIEW_ARTIFACT_KIND = "robot_observable_generated_motion_caption_review_v1"
REVIEW_PROTOCOL = "independent_blind_generated_motion_caption_v1"
ROBOT_CONTRACT = "ula_v2_18d_head_v1"
CHECKPOINT_ARTIFACT_KIND = "ula_mmdit_v2_generator"
CHECKPOINT_ARCHITECTURE = "ula_mmdit_v2"
CONDITION_DIM = 264
MOTION_ONLY_SCOPE = "motion_head_style_duration_only_v1"
MOTION_ONLY_EPISODE_CONTRACT = "ula_v2_18d_motion_only_physical_qc_v1"
FULL_RANDOM_INIT_MODE = "full_generator_random_qwen_lora_frozen_v1"
FORMAL_SCOPE = "formal_variable_length_semantic_units"
TEMPORAL_POLICY = "full_semantic_unit_variable_length_30hz"
NATIVE_DURATION_POLICY = "held_out_native_variable_length_no_crop_no_pad"
DEFAULT_OUTPUT_DIR = Path(
    "/home/gez/nas/cloud/gez/human_motion/deliverables/"
    "ula_v2_18d_pretrain_v1/long_video"
)
FPS = 30.0
WIDTH = 1920
HEIGHT = 1080
MIN_LONG_VIDEO_SEC = 60.0
ALLOWED_HELD_OUT_SPLITS = {"validation", "test"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
FORBIDDEN_FIXED_DURATION_KEYS = {
    "clip_frames",
    "crop_frames",
    "fixed_duration_sec",
    "fixed_frame_count",
    "fixed_window_sec",
    "max_duration_sec",
    "min_duration_sec",
    "target_duration_sec",
    "target_frame_count",
    "window_frames",
}


class LongVideoContractError(RuntimeError):
    """Raised when an evaluation artifact cannot be proven safe to publish."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def value_sha256(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: str | Path, payload: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
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
        raise LongVideoContractError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise LongVideoContractError(f"JSON payload must be an object: {path}")
    return value


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise LongVideoContractError(f"cannot read JSONL {path}: {error}") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise LongVideoContractError(
                f"invalid JSON at {path}:{line_number}: {error}"
            ) from error
        if not isinstance(record, dict):
            raise LongVideoContractError(
                f"record at {path}:{line_number} must be an object"
            )
        records.append(record)
    if not records:
        raise LongVideoContractError(f"JSONL contains no records: {path}")
    return records


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise LongVideoContractError(f"{field} must be a lowercase SHA256")
    return value


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LongVideoContractError(f"{field} must be a non-empty string")
    return value.strip()


def _resolve_input(value: Any, *, relative_to: Path, field: str) -> Path:
    raw = _require_string(value, field)
    path = Path(raw)
    path = path.resolve() if path.is_absolute() else (relative_to / path).resolve()
    if not path.is_file():
        raise LongVideoContractError(f"{field} does not exist: {path}")
    return path


def _walk_forbidden_fixed_duration_keys(value: Any, *, prefix: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{prefix}.{key}"
            if key in FORBIDDEN_FIXED_DURATION_KEYS and child is not None:
                raise LongVideoContractError(
                    f"{child_path} is forbidden by the native-duration contract"
                )
            _walk_forbidden_fixed_duration_keys(child, prefix=child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_forbidden_fixed_duration_keys(child, prefix=f"{prefix}[{index}]")


def validate_checkpoint_file(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise LongVideoContractError(f"cannot load checkpoint {path}: {error}") from error
    if not isinstance(checkpoint, Mapping):
        raise LongVideoContractError("checkpoint payload must be an object")
    exact_checkpoint = {
        "artifact_kind": CHECKPOINT_ARTIFACT_KIND,
        "architecture": CHECKPOINT_ARCHITECTURE,
        "condition_dim": CONDITION_DIM,
        "action_dim": 18,
        "joint_order": list(JOINT_ORDER_18D),
    }
    mismatched_checkpoint = [
        key for key, expected in exact_checkpoint.items() if checkpoint.get(key) != expected
    ]
    action_contract = checkpoint.get("action_contract")
    if (
        mismatched_checkpoint
        or not isinstance(action_contract, Mapping)
        or action_contract.get("version") != ROBOT_CONTRACT
        or action_contract.get("legacy_prefix_dim") != 15
    ):
        raise LongVideoContractError(
            "checkpoint is not the exact ULA V2 18D generator contract: "
            f"{mismatched_checkpoint}"
        )
    config = checkpoint.get("config") or {}
    hidden_dim = int(config.get("hidden_dim", 384))
    state = checkpoint.get("model_state_dict")
    expected_shapes = {
        "input.weight": (hidden_dim, 18),
        "output.weight": (18, hidden_dim),
        "output.bias": (18,),
    }
    if not isinstance(state, Mapping):
        raise LongVideoContractError("checkpoint model_state_dict is missing")
    for name, expected_shape in expected_shapes.items():
        value = state.get(name)
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != expected_shape
            or not torch.isfinite(value).all()
        ):
            raise LongVideoContractError(
                f"checkpoint {name} must be finite with shape {expected_shape}"
            )
    stats = checkpoint.get("action_stats")
    if not isinstance(stats, Mapping):
        raise LongVideoContractError("checkpoint action_stats are missing")
    for name in ("mean", "std"):
        value = torch.as_tensor(stats.get(name))
        if tuple(value.shape) != (18,) or not torch.isfinite(value).all():
            raise LongVideoContractError(f"checkpoint action_stats.{name} must be finite [18]")
        if name == "std" and torch.any(value <= 0):
            raise LongVideoContractError("checkpoint action_stats.std must be positive")
    if checkpoint.get("formal_release_eligible") is not True:
        raise LongVideoContractError(
            "checkpoint is not marked formal_release_eligible; refusing evaluation release"
        )
    if checkpoint.get("artifact_status") != "adjudicated_posttrain_candidate":
        raise LongVideoContractError(
            "checkpoint artifact_status is not adjudicated_posttrain_candidate"
        )
    if checkpoint.get("formal_episode_contract") != MOTION_ONLY_EPISODE_CONTRACT:
        raise LongVideoContractError(
            "checkpoint was not trained with the formal motion-only episode contract"
        )
    random_initialization = checkpoint.get("random_initialization")
    if (
        not isinstance(random_initialization, Mapping)
        or random_initialization.get("mode") != FULL_RANDOM_INIT_MODE
    ):
        raise LongVideoContractError(
            "checkpoint is not the requested full-random from-scratch pretrain"
        )
    provenance = checkpoint.get("data_provenance")
    if not isinstance(provenance, Mapping):
        raise LongVideoContractError("checkpoint data_provenance is missing")
    required = {
        "training_scope": FORMAL_SCOPE,
        "formal_training_enabled": True,
        "temporal_unit_policy": TEMPORAL_POLICY,
        "batching_mode": "native_variable_length",
        "training_policy": "full_network",
        "unsafe_training_data": False,
    }
    mismatched = [key for key, expected in required.items() if provenance.get(key) != expected]
    if mismatched:
        raise LongVideoContractError(
            f"checkpoint formal motion pretrain provenance mismatch: {mismatched}"
        )
    data_contract = checkpoint.get("posttrain_data_contract")
    if not isinstance(data_contract, Mapping):
        raise LongVideoContractError("checkpoint posttrain_data_contract is missing")
    declared_data_hash = _require_sha256(
        data_contract.get("sha256"), "checkpoint.posttrain_data_contract.sha256"
    )
    data_payload = {key: value for key, value in data_contract.items() if key != "sha256"}
    if value_sha256(data_payload) != declared_data_hash:
        raise LongVideoContractError("checkpoint posttrain_data_contract SHA256 is invalid")
    records = data_contract.get("records")
    if not isinstance(records, list) or not records:
        raise LongVideoContractError("checkpoint posttrain_data_contract.records is empty")
    native_frames: dict[str, int] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise LongVideoContractError(f"checkpoint data record {index} is not an object")
        if record.get("split") not in ALLOWED_HELD_OUT_SPLITS:
            continue
        clip_id = _require_string(
            record.get("clip_id"), f"checkpoint data record {index}.clip_id"
        )
        count = record.get("quality_output_frame_count")
        retarget = record.get("retarget_segment")
        training_segment = record.get("training_segment")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 3
            or not isinstance(retarget, Mapping)
            or retarget.get("output_frame_count") != count
            or retarget.get("cropped") is not False
            or not isinstance(training_segment, Mapping)
            or training_segment.get("fixed_window_sec") is not None
        ):
            raise LongVideoContractError(
                f"{clip_id}: checkpoint lacks an exact uncropped native output frame count"
            )
        if clip_id in native_frames:
            raise LongVideoContractError(f"duplicate checkpoint data record: {clip_id}")
        native_frames[clip_id] = count
    if not native_frames:
        raise LongVideoContractError("checkpoint data contract has no held-out native episodes")
    split_contract = checkpoint.get("posttrain_split_contract")
    if not isinstance(split_contract, Mapping):
        raise LongVideoContractError("checkpoint posttrain_split_contract is missing")
    split_hash = _require_sha256(
        split_contract.get("sha256"), "checkpoint.posttrain_split_contract.sha256"
    )
    split_payload = {key: value for key, value in split_contract.items() if key != "sha256"}
    if value_sha256(split_payload) != split_hash:
        raise LongVideoContractError("checkpoint posttrain_split_contract SHA256 is invalid")
    return {
        "global_step": int(checkpoint.get("global_step") or 0),
        "artifact_status": checkpoint["artifact_status"],
        "formal_release_eligible": True,
        "action_dim": 18,
        "robot_contract": ROBOT_CONTRACT,
        "formal_episode_contract": MOTION_ONLY_EPISODE_CONTRACT,
        "data_contract_sha256": declared_data_hash,
        "split_contract_sha256": split_hash,
        "_held_out_native_frames": native_frames,
        "_split_contract": dict(split_contract),
    }


def validate_training_summary(
    path: Path,
    *,
    checkpoint: Path,
) -> dict[str, Any]:
    summary = load_json(path)
    try:
        summary_checkpoint = Path(_require_string(summary.get("checkpoint"), "summary.checkpoint")).resolve()
    except OSError as error:
        raise LongVideoContractError(f"cannot resolve summary checkpoint: {error}") from error
    if summary_checkpoint != checkpoint.resolve():
        raise LongVideoContractError("training summary is bound to a different checkpoint")
    completed = summary.get("completed_steps")
    target = summary.get("target_steps")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (completed, target)):
        raise LongVideoContractError("training summary completed_steps/target_steps are invalid")
    if completed != target and summary.get("stopped_early") is not True:
        raise LongVideoContractError("training did not reach target_steps or a recorded early stop")
    best_step = summary.get("best_step")
    if (
        isinstance(best_step, bool)
        or not isinstance(best_step, int)
        or best_step < 1
        or best_step > completed
    ):
        raise LongVideoContractError("training summary best_step is invalid")
    required = {
        "artifact_status": "adjudicated_posttrain_candidate",
        "formal_release_eligible": True,
        "training_scope": FORMAL_SCOPE,
        "formal_training_enabled": True,
        "temporal_unit_policy": TEMPORAL_POLICY,
        "training_policy": "full_network",
    }
    mismatched = [key for key, expected in required.items() if summary.get(key) != expected]
    if mismatched:
        raise LongVideoContractError(f"training summary formal contract mismatch: {mismatched}")
    return {
        "completed_steps": completed,
        "target_steps": target,
        "stopped_early": bool(summary.get("stopped_early")),
        "best_step": best_step,
        "artifact_status": summary["artifact_status"],
        "formal_release_eligible": True,
    }


def load_split_membership(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise LongVideoContractError("split manifest must contain a non-empty episodes list")
    by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(episodes):
        if not isinstance(raw, Mapping):
            raise LongVideoContractError(f"split episodes[{index}] must be an object")
        clip_id = _require_string(raw.get("clip_id"), f"split episodes[{index}].clip_id")
        split = _require_string(raw.get("split"), f"split episodes[{index}].split")
        if split not in {"train", "validation", "test"}:
            raise LongVideoContractError(f"{clip_id}: unsupported split {split!r}")
        if clip_id in by_id:
            raise LongVideoContractError(f"duplicate split clip_id: {clip_id}")
        by_id[clip_id] = dict(raw)
    return by_id


def validate_generated_csv(path: Path, *, expected_frames: int) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise LongVideoContractError(f"empty generated CSV: {path}") from error
        expected_without_time = list(JOINT_ORDER_18D)
        expected_with_time = ["time_sec", *JOINT_ORDER_18D]
        if header not in (expected_without_time, expected_with_time):
            raise LongVideoContractError(
                f"generated CSV is not the ordered ULA V2 18D contract: {path}"
            )
        has_time = header == expected_with_time
        frame_count = 0
        for line_number, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise LongVideoContractError(f"{path}:{line_number} has the wrong column count")
            try:
                values = [float(value) for value in row]
            except ValueError as error:
                raise LongVideoContractError(f"{path}:{line_number} contains non-numeric data") from error
            if not all(math.isfinite(value) for value in values):
                raise LongVideoContractError(f"{path}:{line_number} contains non-finite data")
            if has_time and not math.isclose(values[0], frame_count / FPS, abs_tol=5e-6):
                raise LongVideoContractError(
                    f"{path}:{line_number} time_sec is not on the exact 30 Hz grid"
                )
            frame_count += 1
    if frame_count < 3:
        raise LongVideoContractError(f"generated CSV must contain at least three frames: {path}")
    if frame_count != expected_frames:
        raise LongVideoContractError(
            f"generated CSV has {frame_count} frames, expected native {expected_frames}"
        )
    return {
        "frames": frame_count,
        "fps": FPS,
        "sample_span_sec": (frame_count - 1) / FPS,
        "video_duration_sec": frame_count / FPS,
        "action_dim": 18,
        "joint_order": list(JOINT_ORDER_18D),
    }


def _load_review_caption(
    caption: Mapping[str, Any],
    *,
    generated_csv_sha256: str,
    checkpoint_sha256: str,
    frame_count: int,
    base_dir: Path,
) -> dict[str, Any]:
    artifact_path = _resolve_input(
        caption.get("review_artifact"),
        relative_to=base_dir,
        field="caption.review_artifact",
    )
    artifact_hash = _require_sha256(
        caption.get("review_artifact_sha256"), "caption.review_artifact_sha256"
    )
    if sha256_file(artifact_path) != artifact_hash:
        raise LongVideoContractError("caption review artifact SHA256 mismatch")
    record_id = _require_string(caption.get("review_record_id"), "caption.review_record_id")
    expected_record_hash = _require_sha256(
        caption.get("review_record_sha256"), "caption.review_record_sha256"
    )
    matches = [
        row
        for row in load_jsonl(artifact_path)
        if row.get("review_id") == record_id
    ]
    if len(matches) != 1:
        raise LongVideoContractError(
            f"caption review record {record_id!r} must occur exactly once"
        )
    record = matches[0]
    if value_sha256(record) != expected_record_hash:
        raise LongVideoContractError("caption review record SHA256 mismatch")
    exact = {
        "artifact_kind": REVIEW_ARTIFACT_KIND,
        "protocol_version": REVIEW_PROTOCOL,
        "review_status": "accepted_robot_observable_text",
        "full_decode_to_eof": True,
        "label_metadata_exposed": False,
        "emotion_inference_performed": False,
        "text_conditioning_claimed": False,
        "generated_csv_sha256": generated_csv_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "decoded_frame_count": frame_count,
        "fps": FPS,
        "native_duration_preserved": True,
        "fixed_duration_window_used": False,
    }
    mismatched = [key for key, expected in exact.items() if record.get(key) != expected]
    if mismatched:
        raise LongVideoContractError(
            f"caption review record contract mismatch: {mismatched}"
        )
    text = record.get("robot_observable_text")
    if not isinstance(text, Mapping):
        raise LongVideoContractError("review record robot_observable_text must be an object")
    zh = str(text.get("zh") or "").strip()
    en = str(text.get("en") or "").strip()
    if not zh and not en:
        raise LongVideoContractError("reviewed robot-observable caption is empty")
    return {
        "kind": "reviewed_robot_observable_text",
        "semantic_role": "reviewed_observation_not_pretrain_text_condition",
        "text_zh": zh or en,
        "text_en": en or zh,
        "review_artifact": str(artifact_path),
        "review_artifact_sha256": artifact_hash,
        "review_record_id": record_id,
        "review_record_sha256": expected_record_hash,
    }


def validate_caption(
    value: Any,
    *,
    generated_csv_sha256: str,
    checkpoint_sha256: str,
    frame_count: int,
    base_dir: Path,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LongVideoContractError("caption must be an object")
    kind = value.get("kind")
    if kind == "reviewed_robot_observable_text":
        return _load_review_caption(
            value,
            generated_csv_sha256=generated_csv_sha256,
            checkpoint_sha256=checkpoint_sha256,
            frame_count=frame_count,
            base_dir=base_dir,
        )
    if kind != "motion_only_metadata":
        raise LongVideoContractError(
            "caption.kind must be reviewed_robot_observable_text or motion_only_metadata"
        )
    if value.get("semantic_role") != "metadata_not_model_condition_not_verified_semantics":
        raise LongVideoContractError(
            "motion_only_metadata requires the explicit non-condition/non-semantic role"
        )
    zh = _require_string(value.get("text_zh"), "caption.text_zh")
    en = str(value.get("text_en") or "").strip() or zh
    return {
        "kind": "motion_only_metadata",
        "semantic_role": "metadata_not_model_condition_not_verified_semantics",
        "text_zh": zh,
        "text_en": en,
        "review_artifact": None,
        "review_artifact_sha256": None,
        "review_record_id": None,
        "review_record_sha256": None,
    }


def validate_generation_record(
    record: Mapping[str, Any],
    *,
    index: int,
    manifest_dir: Path,
    checkpoint_sha256: str,
    split_manifest_sha256: str,
    split_membership: Mapping[str, Mapping[str, Any]],
    authoritative_native_frames: Mapping[str, int],
) -> dict[str, Any]:
    prefix = f"generation[{index}]"
    _walk_forbidden_fixed_duration_keys(record, prefix=prefix)
    if record.get("schema_version") != SCHEMA_VERSION:
        raise LongVideoContractError(f"{prefix}.schema_version mismatch")
    if record.get("artifact_kind") != SEGMENT_ARTIFACT_KIND:
        raise LongVideoContractError(f"{prefix}.artifact_kind mismatch")
    segment_id = _require_string(record.get("segment_id"), f"{prefix}.segment_id")
    if not ID_RE.fullmatch(segment_id):
        raise LongVideoContractError(f"{prefix}.segment_id contains unsafe characters")
    held_out_clip_id = _require_string(
        record.get("held_out_clip_id"), f"{prefix}.held_out_clip_id"
    )
    split = _require_string(record.get("split"), f"{prefix}.split")
    if split not in ALLOWED_HELD_OUT_SPLITS:
        raise LongVideoContractError(f"{segment_id}: split must be validation or test")
    split_record = split_membership.get(held_out_clip_id)
    if split_record is None or split_record.get("split") != split:
        raise LongVideoContractError(
            f"{segment_id}: held-out clip membership does not match the split manifest"
        )
    if _require_sha256(
        record.get("generated_from_checkpoint_sha256"),
        f"{prefix}.generated_from_checkpoint_sha256",
    ) != checkpoint_sha256:
        raise LongVideoContractError(f"{segment_id}: generated checkpoint binding mismatch")
    if _require_sha256(
        record.get("split_manifest_sha256"), f"{prefix}.split_manifest_sha256"
    ) != split_manifest_sha256:
        raise LongVideoContractError(f"{segment_id}: split manifest binding mismatch")
    if record.get("output_kind") != "model_generated_motion":
        raise LongVideoContractError(f"{segment_id}: output_kind is not model_generated_motion")
    if record.get("robot_contract") != ROBOT_CONTRACT or record.get("action_dim") != 18:
        raise LongVideoContractError(f"{segment_id}: robot/action contract mismatch")
    if record.get("fps") != FPS:
        raise LongVideoContractError(f"{segment_id}: generation FPS must be exactly 30 Hz")
    conditioning = record.get("conditioning")
    exact_conditioning = {
        "mode": "motion_only",
        "text_conditioning_used": False,
        "emotion_conditioning_used": False,
        "audio_conditioning_used": False,
    }
    if not isinstance(conditioning, Mapping) or any(
        conditioning.get(key) != expected for key, expected in exact_conditioning.items()
    ):
        raise LongVideoContractError(f"{segment_id}: motion-only conditioning contract mismatch")
    native = record.get("native_duration")
    if not isinstance(native, Mapping) or native.get("policy") != NATIVE_DURATION_POLICY:
        raise LongVideoContractError(f"{segment_id}: native duration policy is missing")
    native_frames = native.get("held_out_episode_frame_count")
    requested_frames = native.get("requested_generation_frame_count")
    if (
        isinstance(native_frames, bool)
        or not isinstance(native_frames, int)
        or native_frames < 3
        or requested_frames != native_frames
        or native.get("cropped") is not False
        or native.get("padded") is not False
    ):
        raise LongVideoContractError(f"{segment_id}: native duration frame binding is invalid")
    authoritative_frames = authoritative_native_frames.get(held_out_clip_id)
    if authoritative_frames is None or native_frames != authoritative_frames:
        raise LongVideoContractError(
            f"{segment_id}: native frame count does not match the checkpoint data contract"
        )
    csv_path = _resolve_input(
        record.get("generated_csv"), relative_to=manifest_dir, field=f"{prefix}.generated_csv"
    )
    csv_hash = _require_sha256(
        record.get("generated_csv_sha256"), f"{prefix}.generated_csv_sha256"
    )
    if sha256_file(csv_path) != csv_hash:
        raise LongVideoContractError(f"{segment_id}: generated CSV SHA256 mismatch")
    csv_info = validate_generated_csv(csv_path, expected_frames=native_frames)
    caption = validate_caption(
        record.get("caption"),
        generated_csv_sha256=csv_hash,
        checkpoint_sha256=checkpoint_sha256,
        frame_count=native_frames,
        base_dir=manifest_dir,
    )
    return {
        "segment_id": segment_id,
        "held_out_clip_id": held_out_clip_id,
        "split": split,
        "source_group_key": split_record.get("source_group_key"),
        "speaker_key": split_record.get("speaker_key"),
        "generated_csv": str(csv_path),
        "generated_csv_sha256": csv_hash,
        "checkpoint_sha256": checkpoint_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "output_kind": "model_generated_motion",
        "conditioning": dict(exact_conditioning),
        "native_duration": {
            "policy": NATIVE_DURATION_POLICY,
            "held_out_episode_frame_count": native_frames,
            "requested_generation_frame_count": requested_frames,
            "cropped": False,
            "padded": False,
        },
        "trajectory": csv_info,
        "caption": caption,
        "generation_record_sha256": value_sha256(record),
    }


def validate_inputs(
    *,
    checkpoint: Path,
    training_summary: Path,
    split_manifest: Path,
    generation_manifest: Path,
    min_duration_sec: float = MIN_LONG_VIDEO_SEC,
) -> dict[str, Any]:
    paths = [checkpoint, training_summary, split_manifest, generation_manifest]
    for path in paths:
        if not Path(path).resolve().is_file():
            raise LongVideoContractError(f"required input file is missing: {Path(path).resolve()}")
    checkpoint = checkpoint.resolve()
    training_summary = training_summary.resolve()
    split_manifest = split_manifest.resolve()
    generation_manifest = generation_manifest.resolve()
    checkpoint_hash = sha256_file(checkpoint)
    checkpoint_info = validate_checkpoint_file(checkpoint)
    authoritative_native_frames = checkpoint_info.pop("_held_out_native_frames")
    checkpoint_split_contract = checkpoint_info.pop("_split_contract")
    summary_info = validate_training_summary(training_summary, checkpoint=checkpoint)
    if checkpoint_info["global_step"] != summary_info["best_step"]:
        raise LongVideoContractError(
            "best checkpoint global_step does not match training summary best_step"
        )
    split_hash = sha256_file(split_manifest)
    if load_json(split_manifest) != checkpoint_split_contract:
        raise LongVideoContractError(
            "split manifest content does not match the checkpoint posttrain split contract"
        )
    split_membership = load_split_membership(split_manifest)
    raw_records = load_jsonl(generation_manifest)
    segments = [
        validate_generation_record(
            record,
            index=index,
            manifest_dir=generation_manifest.parent,
            checkpoint_sha256=checkpoint_hash,
            split_manifest_sha256=split_hash,
            split_membership=split_membership,
            authoritative_native_frames=authoritative_native_frames,
        )
        for index, record in enumerate(raw_records)
    ]
    segment_ids = [row["segment_id"] for row in segments]
    if len(segment_ids) != len(set(segment_ids)):
        raise LongVideoContractError("generation manifest contains duplicate segment_id values")
    if len(segments) < 2:
        raise LongVideoContractError("long-video evaluation requires at least two native segments")
    frame_counts = [row["trajectory"]["frames"] for row in segments]
    if len(set(frame_counts)) < 2:
        raise LongVideoContractError(
            "selected evaluation set does not demonstrate variable native durations"
        )
    total_frames = sum(frame_counts)
    minimum = float(min_duration_sec)
    if not math.isfinite(minimum) or minimum <= 0.0:
        raise LongVideoContractError("min_duration_sec must be finite and positive")
    if total_frames / FPS < minimum:
        raise LongVideoContractError(
            f"long video would be {total_frames / FPS:.3f}s, below required {minimum:.3f}s"
        )
    return {
        "artifact_kind": ARTIFACT_KIND,
        "schema_version": SCHEMA_VERSION,
        "status": "validated_inputs_not_rendered",
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_hash,
            **checkpoint_info,
        },
        "training_summary": {
            "path": str(training_summary),
            "sha256": sha256_file(training_summary),
            **summary_info,
        },
        "split_manifest": {
            "path": str(split_manifest),
            "sha256": split_hash,
        },
        "generation_manifest": {
            "path": str(generation_manifest),
            "sha256": sha256_file(generation_manifest),
            "record_count": len(segments),
        },
        "evaluation_contract": {
            "mode": "motion_only_unconditioned_generation",
            "text_conditioning_used": False,
            "emotion_conditioning_used": False,
            "audio_conditioning_used": False,
            "caption_is_never_presented_as_text_condition": True,
            "held_out_splits_only": sorted(ALLOWED_HELD_OUT_SPLITS),
            "robot_contract": ROBOT_CONTRACT,
            "fps": FPS,
            "width": WIDTH,
            "height": HEIGHT,
            "native_duration_policy": NATIVE_DURATION_POLICY,
            "conditioning_scope": MOTION_ONLY_SCOPE,
            "fixed_six_second_units": False,
        },
        "segments": segments,
        "total_frames": total_frames,
        "video_duration_sec": total_frames / FPS,
        "sample_span_sec": sum((frames - 1) / FPS for frames in frame_counts),
        "unique_frame_counts": sorted(set(frame_counts)),
    }


def build_renderer_command(
    *,
    renderer_python: Path,
    csv_path: Path,
    output_mp4: Path,
    summary_json: Path,
    urdf: Path,
) -> list[str]:
    return [
        str(renderer_python),
        "-m",
        "upper_body_skeleton.mujoco_playback",
        "--joint-csv",
        str(csv_path),
        "--output-mp4",
        str(output_mp4),
        "--summary-json",
        str(summary_json),
        "--fps",
        str(FPS),
        "--width",
        str(WIDTH),
        "--height",
        str(HEIGHT),
        "--camera-margin",
        str(CAMERA_MARGIN),
        "--camera-lookat-z-offset",
        str(CAMERA_LOOKAT_Z_OFFSET),
        "--urdf",
        str(urdf),
    ]


def _run(command: Sequence[str], *, stage: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.setdefault("MUJOCO_GL", "egl")
    environment.setdefault("PYOPENGL_PLATFORM", "egl")
    python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(PROJECT_ROOT)
        if not python_path
        else str(PROJECT_ROOT) + os.pathsep + python_path
    )
    completed = subprocess.run(
        [str(value) for value in command],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout or "").strip()[-4000:]
        raise LongVideoContractError(f"{stage} failed ({completed.returncode}): {detail}")
    return completed


def render_segments(
    plan: Mapping[str, Any],
    *,
    staging: Path,
    renderer_python: Path,
    urdf: Path,
) -> list[dict[str, Any]]:
    if not renderer_python.is_file():
        raise LongVideoContractError(f"renderer Python does not exist: {renderer_python}")
    if not urdf.is_file():
        raise LongVideoContractError(f"ULA V2 URDF does not exist: {urdf}")
    videos = staging / "segments"
    summaries = staging / "render_summaries"
    videos.mkdir(parents=True, exist_ok=False)
    summaries.mkdir(parents=True, exist_ok=False)
    rendered: list[dict[str, Any]] = []
    for index, segment in enumerate(plan["segments"], 1):
        stem = f"{index:03d}_{segment['segment_id']}"
        output_mp4 = videos / f"{stem}.mp4"
        summary_json = summaries / f"{stem}.json"
        _run(
            build_renderer_command(
                renderer_python=renderer_python,
                csv_path=Path(segment["generated_csv"]),
                output_mp4=output_mp4,
                summary_json=summary_json,
                urdf=urdf,
            ),
            stage=f"render segment {segment['segment_id']}",
        )
        render_summary = load_json(summary_json)
        frames = int(segment["trajectory"]["frames"])
        try:
            validate_render_summary(
                render_summary,
                expected_frames=frames,
                expected_width=WIDTH,
                expected_height=HEIGHT,
                expected_urdf=urdf,
            )
            video_check = validate_video(
                output_mp4,
                expected_frames=frames,
                expected_width=WIDTH,
                expected_height=HEIGHT,
                expected_fps=FPS,
            )
        except (OSError, ValueError) as error:
            raise LongVideoContractError(
                f"rendered segment {segment['segment_id']} failed validation: {error}"
            ) from error
        # The staging directory is renamed at publication. Keep the durable path
        # relative to index.json instead of leaving a dead absolute staging path.
        render_summary["output_mp4_at_render_time"] = render_summary.get("output_mp4")
        render_summary["output_mp4"] = str(output_mp4.relative_to(staging))
        render_summary["output_path_base"] = "published_long_video_directory"
        atomic_json(summary_json, render_summary)
        rendered.append(
            {
                **segment,
                "segment_video": str(output_mp4.relative_to(staging)),
                "segment_video_sha256": sha256_file(output_mp4),
                "render_summary": str(summary_json.relative_to(staging)),
                "render_summary_sha256": sha256_file(summary_json),
                "video_validation": video_check,
            }
        )
    return rendered


def _single_line(text: str) -> str:
    return " ".join(str(text).replace("\ufeff", "").split())


def _srt_timestamp(frame: int) -> str:
    milliseconds = int(round(frame * 1000.0 / FPS))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def _clock_timestamp(frame: int) -> str:
    return _srt_timestamp(frame).replace(",", ".")


def _ffmetadata_escape(text: str) -> str:
    escaped = _single_line(text)
    for token in ("\\", "=", ";", "#"):
        escaped = escaped.replace(token, f"\\{token}")
    return escaped


def build_text_artifacts(
    segments: Sequence[Mapping[str, Any]], *, output_dir: Path
) -> dict[str, Any]:
    zh_blocks = []
    en_blocks = []
    chapter_lines = [
        "ULA V2 18D motion-only pretrain held-out long video",
        "All captions are observations/metadata and were NOT model text conditions.",
        "",
    ]
    metadata = [
        ";FFMETADATA1",
        "title=ULA V2 18D motion-only pretrain held-out evaluation",
        "comment=Captions were not model text conditions",
    ]
    timeline = []
    cursor = 0
    for number, segment in enumerate(segments, 1):
        frames = int(segment["trajectory"]["frames"])
        start = cursor
        end = cursor + frames
        caption = segment["caption"]
        if caption["kind"] == "reviewed_robot_observable_text":
            zh_disclosure = "[审核后的机器人可观察描述；未作为预训练文本条件]"
            en_disclosure = "[REVIEWED ROBOT-OBSERVABLE DESCRIPTION; NOT A PRETRAIN TEXT CONDITION]"
        else:
            zh_disclosure = "[动作元数据；不是文本条件；语义未经验证]"
            en_disclosure = "[MOTION-ONLY METADATA; NOT A TEXT CONDITION; SEMANTICS NOT VERIFIED]"
        zh_text = _single_line(caption["text_zh"])
        en_text = _single_line(caption["text_en"])
        zh_blocks.extend(
            [str(number), f"{_srt_timestamp(start)} --> {_srt_timestamp(end)}", f"{zh_disclosure}\n{zh_text}", ""]
        )
        en_blocks.extend(
            [str(number), f"{_srt_timestamp(start)} --> {_srt_timestamp(end)}", f"{en_disclosure}\n{en_text}", ""]
        )
        title = f"{number:02d} {segment['segment_id']} [{caption['kind']}]"
        chapter_lines.append(
            f"{_clock_timestamp(start)} - {_clock_timestamp(end)} | {title} | {zh_text}"
        )
        metadata.extend(
            [
                "[CHAPTER]",
                "TIMEBASE=1/30",
                f"START={start}",
                f"END={end}",
                f"title={_ffmetadata_escape(title)}",
            ]
        )
        timeline.append(
            {
                "number": number,
                "segment_id": segment["segment_id"],
                "start_frame": start,
                "end_frame_exclusive": end,
                "video_duration_sec": frames / FPS,
                "sample_span_sec": (frames - 1) / FPS,
                "caption_kind": caption["kind"],
                "caption_semantic_role": caption["semantic_role"],
                "text_zh": zh_text,
                "text_en": en_text,
            }
        )
        cursor = end
    paths = {
        "zh_srt": output_dir / "ula_v2_18d_pretrain_v1.zh-Hans.srt",
        "en_srt": output_dir / "ula_v2_18d_pretrain_v1.en.srt",
        "chapters_txt": output_dir / "ula_v2_18d_pretrain_v1.chapters.txt",
        "chapters_ffmetadata": output_dir / "ula_v2_18d_pretrain_v1.chapters.ffmetadata",
    }
    atomic_text(paths["zh_srt"], "\n".join(zh_blocks).rstrip() + "\n")
    atomic_text(paths["en_srt"], "\n".join(en_blocks).rstrip() + "\n")
    atomic_text(paths["chapters_txt"], "\n".join(chapter_lines).rstrip() + "\n")
    atomic_text(paths["chapters_ffmetadata"], "\n".join(metadata).rstrip() + "\n")
    return {"paths": paths, "timeline": timeline, "total_frames": cursor}


def _concat_quote(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "'\\''") + "'"


def build_ffmpeg_command(
    *,
    ffmpeg: Path,
    concat_file: Path,
    zh_srt: Path,
    en_srt: Path,
    chapters: Path,
    output_mp4: Path,
) -> list[str]:
    return [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-i",
        str(zh_srt),
        "-i",
        str(en_srt),
        "-i",
        str(chapters),
        "-map",
        "0:v:0",
        "-map",
        "1:0",
        "-map",
        "2:0",
        "-map_metadata",
        "3",
        "-map_chapters",
        "3",
        "-c:v",
        "copy",
        "-c:s",
        "mov_text",
        "-metadata:s:s:0",
        "language=zho",
        "-metadata:s:s:0",
        "title=Motion metadata / reviewed observation (not a model condition)",
        "-metadata:s:s:1",
        "language=eng",
        "-metadata:s:s:1",
        "title=Motion metadata / reviewed observation (not a model condition)",
        "-disposition:s:0",
        "default",
        "-disposition:s:1",
        "0",
        "-metadata",
        "title=ULA V2 18D motion-only pretrain held-out evaluation",
        "-metadata",
        "comment=Subtitles were not model text conditions",
        "-movflags",
        "+faststart",
        "-y",
        str(output_mp4),
    ]


def validate_final_streams(
    path: Path, *, expected_chapters: int | None = None
) -> dict[str, Any]:
    completed = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [line.strip() for line in completed.stderr.splitlines() if "Stream #" in line]
    video = [line for line in lines if " Video:" in line]
    audio = [line for line in lines if " Audio:" in line]
    subtitles = [line for line in lines if " Subtitle:" in line]
    chapter_lines = [
        line.strip()
        for line in completed.stderr.splitlines()
        if line.strip().startswith("Chapter #")
    ]
    if len(video) != 1 or audio or len(subtitles) != 2:
        raise LongVideoContractError(
            "final MP4 must contain one video, zero audio, and exactly two subtitle streams"
        )
    if "Video: h264" not in video[0] or "yuv420p" not in video[0]:
        raise LongVideoContractError("final MP4 must be H.264 yuv420p")
    if not all("mov_text" in line for line in subtitles):
        raise LongVideoContractError("final MP4 subtitle streams must use mov_text")
    if expected_chapters is not None and len(chapter_lines) != int(expected_chapters):
        raise LongVideoContractError(
            f"final MP4 has {len(chapter_lines)} chapters, expected {expected_chapters}"
        )
    return {
        "video_streams": 1,
        "audio_streams": 0,
        "subtitle_streams": 2,
        "video_codec": "h264",
        "pixel_format": "yuv420p",
        "subtitle_codec": "mov_text",
        "chapters": len(chapter_lines),
    }


def assemble_video(
    rendered: Sequence[Mapping[str, Any]], *, staging: Path
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    text_artifacts = build_text_artifacts(rendered, output_dir=staging)
    concat = staging / ".concat.txt"
    concat_lines = [
        f"file {_concat_quote(staging / str(segment['segment_video']))}"
        for segment in rendered
    ]
    atomic_text(concat, "\n".join(concat_lines) + "\n")
    output = staging / "ula_v2_18d_pretrain_v1_long_video.mp4"
    ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    _run(
        build_ffmpeg_command(
            ffmpeg=ffmpeg,
            concat_file=concat,
            zh_srt=text_artifacts["paths"]["zh_srt"],
            en_srt=text_artifacts["paths"]["en_srt"],
            chapters=text_artifacts["paths"]["chapters_ffmetadata"],
            output_mp4=output,
        ),
        stage="assemble long video",
    )
    concat.unlink(missing_ok=True)
    try:
        video_validation = validate_video(
            output,
            expected_frames=text_artifacts["total_frames"],
            expected_width=WIDTH,
            expected_height=HEIGHT,
            expected_fps=FPS,
        )
    except (OSError, ValueError) as error:
        raise LongVideoContractError(f"final long video failed full decode: {error}") from error
    video_validation.update(
        validate_final_streams(output, expected_chapters=len(rendered))
    )
    return output, text_artifacts, video_validation


def _artifact_binding(path: Path, *, relative_to: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(relative_to)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def publish_staging_directory(staging: Path, output_dir: Path) -> None:
    """Publish once, with bounded retries for transient CIFS close/rename races."""
    last_error: OSError | None = None
    for attempt in range(8):
        try:
            os.replace(staging, output_dir)
            return
        except OSError as error:
            last_error = error
            if output_dir.exists():
                raise LongVideoContractError(
                    f"publication target appeared during publish: {output_dir}"
                ) from error
            if attempt < 7:
                time.sleep(min(0.25 * (2**attempt), 2.0))
    raise LongVideoContractError(
        f"could not publish verified staging directory after 8 attempts: {last_error}"
    ) from last_error


def build_long_video(
    *,
    checkpoint: Path,
    training_summary: Path,
    split_manifest: Path,
    generation_manifest: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    renderer_python: Path = Path(sys.executable),
    urdf: Path = DEFAULT_URDF,
    min_duration_sec: float = MIN_LONG_VIDEO_SEC,
) -> dict[str, Any]:
    plan = validate_inputs(
        checkpoint=checkpoint,
        training_summary=training_summary,
        split_manifest=split_manifest,
        generation_manifest=generation_manifest,
        min_duration_sec=min_duration_sec,
    )
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise LongVideoContractError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging.", dir=output_dir.parent)
    )
    try:
        rendered = render_segments(
            plan,
            staging=staging,
            renderer_python=renderer_python.resolve(),
            urdf=urdf.resolve(),
        )
        video, text_artifacts, video_validation = assemble_video(rendered, staging=staging)
        index = {
            **{key: value for key, value in plan.items() if key != "segments"},
            "status": "complete_verified",
            "created_at": utc_now(),
            "output_directory": str(output_dir),
            "segments": rendered,
            "timeline": text_artifacts["timeline"],
            "artifacts": {
                "long_video": _artifact_binding(video, relative_to=staging),
                "zh_srt": _artifact_binding(
                    text_artifacts["paths"]["zh_srt"], relative_to=staging
                ),
                "en_srt": _artifact_binding(
                    text_artifacts["paths"]["en_srt"], relative_to=staging
                ),
                "chapters_txt": _artifact_binding(
                    text_artifacts["paths"]["chapters_txt"], relative_to=staging
                ),
                "chapters_ffmetadata": _artifact_binding(
                    text_artifacts["paths"]["chapters_ffmetadata"], relative_to=staging
                ),
            },
            "video_validation": video_validation,
            "publication_policy": {
                "final_index_written_after_full_decode": True,
                "staging_directory_atomically_published": True,
                "placeholder_or_fake_training_results_allowed": False,
                "caption_disclosure_forced": True,
            },
        }
        atomic_json(staging / "index.json", index)
        publish_staging_directory(staging, output_dir)
        return {**index, "index": str(output_dir / "index.json")}
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--renderer-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--min-duration-sec", type=float, default=MIN_LONG_VIDEO_SEC)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate real inputs and print a plan; do not render or publish artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    kwargs = {
        "checkpoint": args.checkpoint,
        "training_summary": args.training_summary,
        "split_manifest": args.split_manifest,
        "generation_manifest": args.generation_manifest,
        "min_duration_sec": args.min_duration_sec,
    }
    if args.validate_only:
        result = validate_inputs(**kwargs)
    else:
        result = build_long_video(
            **kwargs,
            output_dir=args.output_dir,
            renderer_python=args.renderer_python,
            urdf=args.urdf,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
