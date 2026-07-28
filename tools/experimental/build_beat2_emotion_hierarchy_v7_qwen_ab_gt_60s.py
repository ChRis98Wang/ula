#!/usr/bin/env python3
"""Build a gated 60-second BEAT2 GT | frozen-Qwen | LoRA-Qwen video.

Preparation is CPU-only and is safe while the LoRA B arm trains.  Generation
cannot begin until the real B checkpoint and completion summary exist and pass
the existing V7 admission gates.  Both generator arms reuse the audited V7
60-second builder, prompts, seeds, sampler, and initial-noise implementation.

The left pane is a deterministic held-out BEAT2 physical-QC 18D reference for
the exact prompt.  It is a qualitative target example, not a claim that
text-conditioned motion has one deterministic framewise ground truth and not
raw human skeleton data.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
from typing import Any, Iterable, Mapping, Sequence

import imageio_ffmpeg
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.experimental import build_beat2_clean_abc_video as abc_video  # noqa: E402
from tools.experimental import build_beat2_emotion_hierarchy_v7_60s as v7_video  # noqa: E402
from tools.experimental import build_beat2_style_emotion_v2_60s as style_video  # noqa: E402
from tools.train_beat2_emotion_hierarchy_v7 import TRAINING_POLICY  # noqa: E402
from upper_body_skeleton.ula_v2_18d_head import (  # noqa: E402
    JOINT_ORDER_18D,
    read_joint_csv,
)


SCHEMA_VERSION = 1
CONFIG_ARTIFACT_KIND = (
    "beat2_emotion_hierarchy_v7_qwen_ab_gt_60s_config_v1"
)
PLAN_ARTIFACT_KIND = "beat2_v7_qwen_ab_gt_60s_cpu_preparation_v1"
SUMMARY_ARTIFACT_KIND = "beat2_v7_qwen_ab_gt_60s_comparison_v1"
DATA_POLICY = "beat2_only_no_external_motion_dataset_v1"
FPS = 30.0
ACTION_DIM = 18
EXPECTED_FRAMES = 1800
EXPECTED_EVENT_COUNT = 24
MINIMUM_EVENT_FRAMES = 45
MAXIMUM_EVENT_FRAMES = 120
FORBIDDEN_SOURCE_TOKENS = ("kimodo", "hanyang")
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "beat2_emotion_hierarchy_v7_qwen_ab_gt_60s.json"
)
PLAN_FILENAME = "prepared_plan.json"
SUMMARY_FILENAME = "summary.json"
FINAL_VIDEO_FILENAME = "BEAT2_GT_vs_frozen_Qwen_vs_LoRA_Qwen_60s.mp4"
COMPARISON_CONTRACT = {
    "same_prompt": True,
    "same_seed": True,
    "same_initial_noise": True,
    "gt_loaded_before_generation": False,
    "gt_used_as_generation_input": False,
    "generation_frames_equal_gt_event_frames": True,
    "planner_duration_diagnostic_only": True,
    "static_padding_frames": 0,
    "endpoint_hold": False,
    "additional_smoothing": False,
    "time_warp": False,
    "network_frame_crop": False,
    "boundary_blend": False,
    "forced_last_frame_blend": False,
    "forced_return_to_zero": False,
}
METRIC_CONTRACT = {
    "jerk": (
        "raw_segment_rms_rad_s3_excluding_endpoint_hold_and_slot_boundaries"
    ),
    "expression": (
        "raw_segment_joint_range_rms_rad_proxy_not_emotion_accuracy"
    ),
    "head_activity": "raw_segment_head_velocity_rms_rad_s",
}
TARGET_EMOTION_ALIASES = {
    "neutral": "neutral",
    "happy": "happy",
    "angry": "angry",
    "surprised": "surprise",
    "sad": "sad",
    "fearful": "fear",
}


class ComparisonError(RuntimeError):
    """Raised when the A/B/GT comparison cannot prove its contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _resolve_path(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ComparisonError(f"{field} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    _reject_source_path(path, field=field)
    return path


def _reject_source_path(path: str | Path, *, field: str) -> None:
    normalized = "".join(
        character
        for character in str(path).casefold()
        if character.isalnum()
    )
    for token in FORBIDDEN_SOURCE_TOKENS:
        if token in normalized:
            raise ComparisonError(
                f"{field} references forbidden external source {token!r}"
            )


def _load_json(path: Path, *, field: str) -> dict[str, Any]:
    if not path.is_file():
        raise ComparisonError(f"{field} does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ComparisonError(f"{field} must be a JSON object")
    return value


def target_emotion_from_prompt(prompt: str) -> str:
    lower = str(prompt).casefold()
    matches = [
        emotion
        for token, emotion in TARGET_EMOTION_ALIASES.items()
        if f"{token} affect" in lower
    ]
    if len(matches) != 1:
        raise ComparisonError(
            f"prompt must contain exactly one supported target emotion: {prompt}"
        )
    return matches[0]


def validate_config(
    config: Mapping[str, Any],
    *,
    config_path: Path,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "artifact_kind",
        "data_policy",
        "no_external_data",
        "no_kimodo",
        "no_hanyang",
        "base_frozen_video_config",
        "base_frozen_video_config_sha256",
        "lora_training_overlay",
        "lora_training_overlay_sha256",
        "lora_training_service_unit",
        "target_duration_sec",
        "event_count",
        "minimum_event_frames",
        "maximum_event_frames",
        "selection_seed",
        "generation_seed_base",
        "gt_selection_policy",
        "comparison_contract",
        "metric_contract",
        "render",
        "output_dir",
    }
    if set(config) != required:
        raise ComparisonError("A/B/GT config fields changed")
    if (
        config.get("schema_version") != SCHEMA_VERSION
        or config.get("artifact_kind") != CONFIG_ARTIFACT_KIND
        or config.get("data_policy") != DATA_POLICY
        or config.get("no_external_data") is not True
        or config.get("no_kimodo") is not True
        or config.get("no_hanyang") is not True
        or config.get("comparison_contract") != COMPARISON_CONTRACT
        or config.get("metric_contract") != METRIC_CONTRACT
        or config.get("gt_selection_policy")
        != (
            "six_emotion_coverage_then_seeded_round_robin_exact_1800_"
            "frames_24_fixed_test_events_v1"
        )
        or not math.isclose(
            float(config.get("target_duration_sec", 0.0)),
            60.0,
            abs_tol=1e-9,
        )
        or config.get("lora_training_service_unit")
        != "beat2-emotion-hierarchy-v7-lora-qwen.service"
        or int(config.get("event_count", 0)) != EXPECTED_EVENT_COUNT
        or int(config.get("minimum_event_frames", 0))
        != MINIMUM_EVENT_FRAMES
        or int(config.get("maximum_event_frames", 0))
        != MAXIMUM_EVENT_FRAMES
        or int(config.get("selection_seed", 0)) != 20260728
        or int(config.get("generation_seed_base", 0)) != 202607280000
    ):
        raise ComparisonError("A/B/GT config contract is invalid")
    base_path = _resolve_path(
        config["base_frozen_video_config"],
        field="base_frozen_video_config",
    )
    overlay_path = _resolve_path(
        config["lora_training_overlay"],
        field="lora_training_overlay",
    )
    for path, expected, field in (
        (
            base_path,
            config["base_frozen_video_config_sha256"],
            "base_frozen_video_config",
        ),
        (
            overlay_path,
            config["lora_training_overlay_sha256"],
            "lora_training_overlay",
        ),
    ):
        if not path.is_file() or not _is_sha256(expected):
            raise ComparisonError(f"{field} pin is invalid")
        if abc_video.sha256_file(path) != expected:
            raise ComparisonError(f"{field} SHA256 changed")
    render = config.get("render")
    if (
        not isinstance(render, Mapping)
        or set(render)
        != {
            "pane_width",
            "pane_height",
            "panel_width",
            "simplified",
            "backend",
            "fps",
        }
        or min(
            int(render["pane_width"]),
            int(render["pane_height"]),
            int(render["panel_width"]),
        )
        <= 0
        or int(render["fps"]) != int(FPS)
        or render["backend"] != "mujoco_egl"
        or not isinstance(render["simplified"], bool)
    ):
        raise ComparisonError("render contract is invalid")
    output_dir = _resolve_path(config["output_dir"], field="output_dir")
    base_config = _load_json(base_path, field="base frozen video config")
    validated_base = v7_video.validate_config(base_config)
    validated_base.pop("_validated_comparison_interface", None)
    overlay = _load_json(overlay_path, field="LoRA training overlay")
    if (
        overlay.get("qwen_condition_variant") != "lora_finetuned"
        or overlay.get("base_frozen_config")
        != "configs/beat2_emotion_hierarchy_v7.json"
    ):
        raise ComparisonError("LoRA overlay is not the locked V7 B arm")
    return {
        **deepcopy(dict(config)),
        "_config_path": str(config_path),
        "_config_sha256": abc_video.sha256_file(config_path),
        "_base_path": str(base_path),
        "_base": validated_base,
        "_overlay_path": str(overlay_path),
        "_overlay": overlay,
        "_output_dir": str(output_dir),
    }


def _reference_candidate_valid(
    record: Mapping[str, Any],
    *,
    prompt: str,
    target_emotion: str,
) -> bool:
    motion = record.get("motion_18d")
    quality = motion.get("quality_gate") if isinstance(motion, Mapping) else None
    return bool(
        record.get("dataset") == "BEAT2"
        and record.get("fixed_split_assignment") == "test"
        and str(record.get("prompt", "")).strip() == prompt
        and record.get("emotion_id") == target_emotion
        and record.get("accepted_for_training") is True
        and (record.get("adjudication") or {}).get("status")
        == "motion_only_train_ready"
        and record.get("training_admission_status")
        == "motion_only_physical_qc_train_ready"
        and isinstance(motion, Mapping)
        and motion.get("state") == "passed"
        and int(motion.get("action_dim", -1)) == ACTION_DIM
        and float(motion.get("fps", -1)) == FPS
        and 4 <= int(motion.get("frames", -1)) <= MAXIMUM_EVENT_FRAMES
        and isinstance(quality, Mapping)
        and quality.get("passed") is True
    )


def select_gt_reference(
    records: Mapping[str, Mapping[str, Any]],
    *,
    prompt: str,
    target_emotion: str,
) -> tuple[Mapping[str, Any], list[str]]:
    candidates = [
        record
        for record in records.values()
        if _reference_candidate_valid(
            record,
            prompt=prompt,
            target_emotion=target_emotion,
        )
    ]
    if not candidates:
        raise ComparisonError(
            f"no physical-QC fixed-test BEAT2 reference for prompt: {prompt}"
        )
    candidates.sort(
        key=lambda record: (
            -int(record["motion_18d"]["frames"]),
            str(record["clip_id"]),
        )
    )
    return candidates[0], [str(record["clip_id"]) for record in candidates]


def select_native_event_montage(
    records: Mapping[str, Mapping[str, Any]],
    *,
    event_count: int = EXPECTED_EVENT_COUNT,
    target_frames: int = EXPECTED_FRAMES,
    minimum_frames: int = MINIMUM_EVENT_FRAMES,
    maximum_frames: int = MAXIMUM_EVENT_FRAMES,
    selection_seed: int = 20260728,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Select 24 native fixed-test events that sum to exactly 1,800 frames."""

    emotions = ("neutral", "sad", "happy", "angry", "surprise", "fear")

    def valid(record: Mapping[str, Any]) -> bool:
        motion = record.get("motion_18d")
        quality = (
            motion.get("quality_gate") if isinstance(motion, Mapping) else None
        )
        prompt = str(record.get("prompt", "")).strip()
        try:
            prompt_emotion = target_emotion_from_prompt(prompt)
        except ComparisonError:
            return False
        return bool(
            record.get("dataset") == "BEAT2"
            and record.get("fixed_split_assignment") == "test"
            and record.get("emotion_id") in emotions
            and prompt_emotion == record.get("emotion_id")
            and record.get("accepted_for_training") is True
            and (record.get("adjudication") or {}).get("status")
            == "motion_only_train_ready"
            and record.get("training_admission_status")
            == "motion_only_physical_qc_train_ready"
            and isinstance(motion, Mapping)
            and motion.get("state") == "passed"
            and int(motion.get("action_dim", -1)) == ACTION_DIM
            and float(motion.get("fps", -1)) == FPS
            and minimum_frames
            <= int(motion.get("frames", -1))
            <= maximum_frames
            and isinstance(quality, Mapping)
            and quality.get("passed") is True
        )

    candidates = [record for record in records.values() if valid(record)]
    by_emotion = {
        emotion: [
            record
            for record in candidates
            if record.get("emotion_id") == emotion
        ]
        for emotion in emotions
    }
    if any(not values for values in by_emotion.values()):
        raise ComparisonError("native montage cannot cover all six emotions")

    def stable_digest(record: Mapping[str, Any]) -> str:
        return hashlib.sha256(
            f"{selection_seed}:{record['clip_id']}".encode("utf-8")
        ).hexdigest()

    # One high-duration record per emotion guarantees coverage.  A seeded hash
    # resolves ties without visual or model-output selection.
    baseline = [
        min(
            by_emotion[emotion],
            key=lambda record: (
                -int(record["motion_18d"]["frames"]),
                stable_digest(record),
                str(record["clip_id"]),
            ),
        )
        for emotion in emotions
    ]
    baseline_ids = {str(record["clip_id"]) for record in baseline}
    remaining_count = int(event_count) - len(baseline)
    remaining_frames = int(target_frames) - sum(
        int(record["motion_18d"]["frames"]) for record in baseline
    )
    if remaining_count <= 0 or remaining_frames <= 0:
        raise ComparisonError("native montage target is incompatible with coverage")

    # Balance the deterministic search pool across emotions, then solve exact
    # event-count/subset-sum without looking at actions or model output.
    per_emotion_pool = {
        emotion: sorted(
            (
                record
                for record in by_emotion[emotion]
                if record["clip_id"] not in baseline_ids
            ),
            key=lambda record: (stable_digest(record), str(record["clip_id"])),
        )[:50]
        for emotion in emotions
    }
    pool = [
        per_emotion_pool[emotion][index]
        for index in range(50)
        for emotion in emotions
        if index < len(per_emotion_pool[emotion])
    ]
    states: list[dict[int, tuple[int, ...]]] = [
        {} for _ in range(remaining_count + 1)
    ]
    states[0][0] = ()
    for pool_index, record in enumerate(pool):
        frames = int(record["motion_18d"]["frames"])
        for count in range(
            min(remaining_count, pool_index + 1),
            0,
            -1,
        ):
            for frame_sum, path in list(states[count - 1].items()):
                new_sum = frame_sum + frames
                if (
                    new_sum <= remaining_frames
                    and new_sum not in states[count]
                ):
                    states[count][new_sum] = path + (pool_index,)
        if remaining_frames in states[remaining_count]:
            break
    selected_path = states[remaining_count].get(remaining_frames)
    if selected_path is None:
        raise ComparisonError(
            "cannot select an exact 1,800-frame native-event montage"
        )
    selected = baseline + [pool[index] for index in selected_path]
    if (
        len(selected) != event_count
        or len({record["clip_id"] for record in selected}) != event_count
        or sum(int(record["motion_18d"]["frames"]) for record in selected)
        != target_frames
        or {record["emotion_id"] for record in selected} != set(emotions)
    ):
        raise ComparisonError("native-event montage invariant failed")
    return selected, {
        "candidate_count": len(candidates),
        "event_count": len(selected),
        "total_frames": target_frames,
        "duration_sec": target_frames / FPS,
        "minimum_event_frames": min(
            int(record["motion_18d"]["frames"]) for record in selected
        ),
        "maximum_event_frames": max(
            int(record["motion_18d"]["frames"]) for record in selected
        ),
        "unique_prompt_count": len(
            {str(record["prompt"]) for record in selected}
        ),
        "unique_speaker_count": len(
            {str(record.get("speaker_key")) for record in selected}
        ),
        "emotion_counts": {
            emotion: sum(
                record["emotion_id"] == emotion for record in selected
            )
            for emotion in emotions
        },
        "selection_used_action_values": False,
        "selection_used_generator_outputs": False,
        "selection_used_visual_review": False,
    }


def _validate_gt_csv(
    record: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    motion = record["motion_18d"]
    csv_path = _resolve_path(motion["safe_csv"], field="GT safe_csv")
    if not csv_path.is_file():
        raise ComparisonError(f"GT safe CSV is missing: {csv_path}")
    csv_sha256 = abc_video.sha256_file(csv_path)
    if csv_sha256 != motion.get("safe_csv_sha256"):
        raise ComparisonError(f"GT safe CSV SHA changed: {record['clip_id']}")
    values = read_joint_csv(csv_path)
    if values.shape != (int(motion["frames"]), ACTION_DIM):
        raise ComparisonError(f"GT safe CSV shape changed: {record['clip_id']}")
    return values, {
        "clip_id": record["clip_id"],
        "source_clip_id": record.get("source_clip_id"),
        "speaker_key": record.get("speaker_key"),
        "fixed_split_assignment": "test",
        "safe_csv": str(csv_path),
        "safe_csv_sha256": csv_sha256,
        "frames": int(len(values)),
        "fps": FPS,
        "quality_json": motion.get("quality_json"),
        "quality_sha256": motion.get("quality_sha256"),
        "record_sha256": record.get("record_sha256"),
    }


def _declared_gt_receipt(record: Mapping[str, Any]) -> dict[str, Any]:
    """Seal GT identity metadata without reading the trajectory action values."""

    motion = record["motion_18d"]
    csv_path = _resolve_path(motion["safe_csv"], field="GT safe_csv")
    if not csv_path.is_file():
        raise ComparisonError(f"GT safe CSV is missing: {csv_path}")
    if not _is_sha256(motion.get("safe_csv_sha256")):
        raise ComparisonError(
            f"GT safe CSV has no declared SHA: {record['clip_id']}"
        )
    return {
        "clip_id": record["clip_id"],
        "source_clip_id": record.get("source_clip_id"),
        "speaker_key": record.get("speaker_key"),
        "fixed_split_assignment": "test",
        "safe_csv": str(csv_path),
        "declared_safe_csv_sha256": motion["safe_csv_sha256"],
        "declared_frames": int(motion["frames"]),
        "declared_fps": FPS,
        "quality_json": motion.get("quality_json"),
        "declared_quality_sha256": motion.get("quality_sha256"),
        "record_sha256": record.get("record_sha256"),
        "action_values_loaded": False,
        "file_content_hashed_during_preparation": False,
    }


def _b_paths(validated: Mapping[str, Any]) -> dict[str, Path]:
    output = _resolve_path(
        validated["_overlay"]["output_dir"],
        field="LoRA B output_dir",
    )
    return {
        "output": output,
        "summary": output / "training_summary_v7.json",
        "checkpoint": output / "generator_emotion_hierarchy_v7.pt",
    }


def _training_service_active(unit: str) -> bool:
    completed = subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", str(unit)],
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def b_completion_state(validated: Mapping[str, Any]) -> dict[str, Any]:
    paths = _b_paths(validated)
    if not paths["summary"].is_file():
        return {
            "ready": False,
            "status": "training_in_progress",
            "summary": str(paths["summary"]),
            "checkpoint": str(paths["checkpoint"]),
        }
    summary = _load_json(paths["summary"], field="LoRA B training summary")
    status = summary.get("status")
    if status in {
        "rejected_condition_collapse_or_quality_gate",
        "smoke_complete",
    }:
        raise ComparisonError(f"LoRA B ended without admission: {status}")
    if status != "experimental_candidate":
        return {
            "ready": False,
            "status": str(status or "incomplete_summary"),
            "summary": str(paths["summary"]),
            "checkpoint": str(paths["checkpoint"]),
        }
    if not paths["checkpoint"].is_file():
        raise ComparisonError("LoRA B summary is complete but checkpoint is absent")
    if summary.get("checkpoint_sha256") != abc_video.sha256_file(
        paths["checkpoint"]
    ):
        raise ComparisonError("LoRA B checkpoint differs from its summary")
    unit = str(validated.get("lora_training_service_unit") or "")
    if unit and _training_service_active(unit):
        return {
            "ready": False,
            "status": "checkpoint_complete_waiting_for_training_service_exit",
            "summary": str(paths["summary"]),
            "checkpoint": str(paths["checkpoint"]),
            "training_service_unit": unit,
            "training_service_active": True,
        }
    return {
        "ready": True,
        "status": status,
        "summary": str(paths["summary"]),
        "summary_sha256": abc_video.sha256_file(paths["summary"]),
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_sha256": abc_video.sha256_file(paths["checkpoint"]),
        "completed_steps": int(summary.get("completed_steps", -1)),
        "training_service_unit": unit or None,
        "training_service_active": False,
    }


def prepare_plan(
    validated: Mapping[str, Any],
) -> dict[str, Any]:
    base = validated["_base"]
    manifest_path = _resolve_path(
        base["source_manifest"], field="BEAT2 source_manifest"
    )
    records, manifest_sha256 = abc_video.load_manifest(
        manifest_path,
        expected_sha256=base["source_manifest_sha256"],
    )
    selected_records, selection_receipt = select_native_event_montage(
        records,
        event_count=int(validated["event_count"]),
        target_frames=EXPECTED_FRAMES,
        minimum_frames=int(validated["minimum_event_frames"]),
        maximum_frames=int(validated["maximum_event_frames"]),
        selection_seed=int(validated["selection_seed"]),
    )
    selections = []
    elapsed_frames = 0
    for index, record in enumerate(selected_records):
        prompt = str(record["prompt"]).strip()
        target_emotion = str(record["emotion_id"])
        frames = int(record["motion_18d"]["frames"])
        receipt = _declared_gt_receipt(record)
        selections.append(
            {
                "index": index + 1,
                "start_frame": elapsed_frames,
                "end_frame": elapsed_frames + frames,
                "frames": frames,
                "native_duration_sec": frames / FPS,
                "start_sec": elapsed_frames / FPS,
                "end_sec": (elapsed_frames + frames) / FPS,
                "prompt": prompt,
                "target_emotion": target_emotion,
                "seed": int(validated["generation_seed_base"]) + index,
                "selection_policy": validated["gt_selection_policy"],
                "gt_contract_checks": {
                    "fixed_test_split": True,
                    "exact_prompt_match": (
                        str(record.get("prompt", "")).strip() == prompt
                    ),
                    "exact_target_emotion_match": (
                        record.get("emotion_id") == target_emotion
                    ),
                    "physical_qc_passed": True,
                },
                "reference": receipt,
            }
        )
        elapsed_frames += frames
    if elapsed_frames != EXPECTED_FRAMES:
        raise ComparisonError("prepared native-event timeline is not 60 seconds")
    b_state = b_completion_state(validated)
    output_dir = Path(validated["_output_dir"])
    plan = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": PLAN_ARTIFACT_KIND,
        "status": (
            "ready_for_post_training_generation"
            if b_state["ready"]
            else "waiting_for_real_lora_B_completion"
        ),
        "created_utc": utc_now(),
        "config": validated["_config_path"],
        "config_sha256": validated["_config_sha256"],
        "data_policy": DATA_POLICY,
        "no_external_data": True,
        "no_kimodo": True,
        "no_hanyang": True,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": manifest_sha256,
        "target_duration_sec": 60.0,
        "expected_frames": EXPECTED_FRAMES,
        "event_count": EXPECTED_EVENT_COUNT,
        "selection_receipt": selection_receipt,
        "static_padding_frames": 0,
        "endpoint_hold_frames": 0,
        "gt_identity_metadata_selected_before_generation": True,
        "gt_action_values_loaded_before_generation": False,
        "gt_used_as_generation_input": False,
        "comparison_contract": deepcopy(COMPARISON_CONTRACT),
        "metric_contract": deepcopy(METRIC_CONTRACT),
        "selections": selections,
        "frozen_A": {
            "checkpoint": base["generator_checkpoint"],
            "training_summary": base["generator_training_summary"],
            "condition_cache": base["qwen_condition_cache"],
        },
        "lora_B": b_state,
        "generation_or_render_executed": False,
        "gpu_accessed": False,
        "fake_video_created": False,
    }
    plan["sha256"] = abc_video.value_sha256(plan)
    _atomic_json(output_dir / PLAN_FILENAME, plan)
    return plan


def derived_branch_config(
    validated: Mapping[str, Any],
    *,
    variant: str,
) -> dict[str, Any]:
    if variant not in {"frozen_base", "lora_finetuned"}:
        raise ComparisonError(f"unknown branch variant: {variant}")
    config = deepcopy(validated["_base"])
    output_dir = Path(validated["_output_dir"]) / "branches" / variant
    config["output_dir"] = str(output_dir)
    if variant == "frozen_base":
        return config
    b_paths = _b_paths(validated)
    overlay = validated["_overlay"]
    config["qwen_condition_cache"] = overlay["qwen_condition_cache"]
    config["generator_checkpoint"] = str(b_paths["checkpoint"])
    config["generator_training_summary"] = str(b_paths["summary"])
    interface = config["qwen_generator_comparison"]
    interface["active_variant"] = "lora_finetuned"
    arm = interface["variants"]["lora_finetuned"]
    arm.update(
        {
            "generation_enabled": True,
            "generator_checkpoint": str(b_paths["checkpoint"]),
            "generator_training_summary": str(b_paths["summary"]),
            "expected_training_policy": TRAINING_POLICY,
        }
    )
    return config


def _write_branch_config(
    validated: Mapping[str, Any], *, variant: str
) -> Path:
    path = (
        Path(validated["_output_dir"])
        / "derived_configs"
        / f"{variant}.json"
    )
    _atomic_json(path, derived_branch_config(validated, variant=variant))
    return path


def _generate_branch(
    validated: Mapping[str, Any],
    *,
    variant: str,
    plan: Mapping[str, Any],
    manifest_records: Mapping[str, Mapping[str, Any]],
    shared_noises: Sequence[np.ndarray],
) -> dict[str, Any]:
    config_path = _write_branch_config(validated, variant=variant)
    branch_config = _load_json(config_path, field=f"{variant} derived config")
    checked = v7_video.validate_config(branch_config)
    comparison = checked.pop("_validated_comparison_interface")
    active = comparison["active"]
    device = torch.device(str(checked["sampling"]["device"]))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ComparisonError("post-training A/B sampling requires available CUDA")
    prompts = [str(item["prompt"]) for item in plan["selections"]]
    if len(prompts) != EXPECTED_EVENT_COUNT:
        raise ComparisonError("native-event prompt count changed")
    with v7_video._patched_v7_video_engine(
        checked,
        comparison,
    ):
        model, style_head, checkpoint_receipt = (
            style_video._load_v2_checkpoint(
                Path(active["generator_checkpoint"]),
                Path(active["generator_training_summary"]),
                expected_manifest_sha256=checked[
                    "source_manifest_sha256"
                ],
                expected_qwen_cache_sha256=active[
                    "condition_cache_sha256"
                ],
                minimum_training_steps=int(
                    checked["minimum_training_steps"]
                ),
                device=device,
            )
        )
        prompt_latents, cache_receipt = style_video._load_prompt_latents(
            Path(active["condition_cache"]),
            prompts=prompts,
            manifest_records=manifest_records,
            manifest_sha256=checked["source_manifest_sha256"],
        )
        raw_segments: list[np.ndarray] = []
        event_receipts: list[dict[str, Any]] = []
        steps = int(checked["sampling"]["steps"])
        guidance_scale = style_video.validate_guidance_scale(
            checked["sampling"]["guidance_scale"]
        )
        for index, (selection, noise) in enumerate(
            zip(plan["selections"], shared_noises, strict=True)
        ):
            frames = int(selection["frames"])
            prompt = str(selection["prompt"])
            noise = np.asarray(noise, dtype=np.float32)
            if noise.shape != (frames, ACTION_DIM):
                raise ComparisonError(
                    f"{variant} event noise shape changed at {index + 1}"
                )
            latent = prompt_latents[prompt]
            condition, predicted_style = (
                style_video.compose_text_style_condition(
                    style_head,
                    latent,
                    device=device,
                )
            )
            planner_duration = style_video.predict_duration_sec(
                model,
                condition,
                device=device,
            )
            trajectory = style_video.sample_trajectory_cfg(
                model,
                condition,
                initial_noise=noise,
                steps=steps,
                guidance_scale=guidance_scale,
                device=device,
            )
            trajectory = np.asarray(trajectory, dtype=np.float32)
            if (
                trajectory.shape != (frames, ACTION_DIM)
                or not np.isfinite(trajectory).all()
            ):
                raise ComparisonError(
                    f"{variant} event output shape changed at {index + 1}"
                )
            raw_segments.append(trajectory)
            event_receipts.append(
                {
                    "index": index + 1,
                    "frames": frames,
                    "native_comparison_duration_sec": frames / FPS,
                    "planner_predicted_duration_sec": planner_duration,
                    "planner_duration_used_for_comparison_frames": False,
                    "prompt": prompt,
                    "seed": int(selection["seed"]),
                    "condition_sha256": hashlib.sha256(
                        condition.tobytes()
                    ).hexdigest(),
                    "qwen_latent_sha256": hashlib.sha256(
                        np.asarray(latent, dtype=np.float32).tobytes()
                    ).hexdigest(),
                    "predicted_style": predicted_style.tolist(),
                    "initial_noise_sha256": hashlib.sha256(
                        noise.tobytes()
                    ).hexdigest(),
                    "trajectory_sha256": hashlib.sha256(
                        trajectory.tobytes()
                    ).hexdigest(),
                }
            )
    del model, style_head
    gc.collect()
    torch.cuda.empty_cache()
    display = np.concatenate(raw_segments, axis=0)
    if display.shape != (EXPECTED_FRAMES, ACTION_DIM):
        raise ComparisonError(f"{variant} native-event montage is not 60 seconds")
    return {
        "variant": variant,
        "derived_config": str(config_path),
        "derived_config_sha256": abc_video.sha256_file(config_path),
        "display": display,
        "raw_segments": raw_segments,
        "noise_segments": [np.asarray(value).copy() for value in shared_noises],
        "checkpoint_admission": checkpoint_receipt,
        "condition_cache": cache_receipt,
        "events": event_receipts,
        "fixed_frames_equal_gt": True,
        "planner_duration_diagnostic_only": True,
        "static_padding_frames": 0,
        "endpoint_hold_frames": 0,
    }


def _same_noise_prefix(
    a: Sequence[np.ndarray], b: Sequence[np.ndarray]
) -> list[dict[str, Any]]:
    if len(a) != EXPECTED_EVENT_COUNT or len(b) != EXPECTED_EVENT_COUNT:
        raise ComparisonError("A/B noise segment count changed")
    receipts = []
    for index, (left, right) in enumerate(zip(a, b, strict=True)):
        if (
            left.shape != right.shape
            or len(left) < 4
            or not np.array_equal(left, right)
        ):
            raise ComparisonError(
                f"A/B initial noise differs for segment {index + 1}"
            )
        receipts.append(
            {
                "index": index + 1,
                "shared_frames": len(left),
                "exact_full_event_noise_match": True,
                "sha256": hashlib.sha256(
                    left.tobytes()
                ).hexdigest(),
            }
        )
    return receipts


def _metric_receipt(values: np.ndarray) -> dict[str, Any]:
    metrics = abc_video.trajectory_metrics(values, fps=FPS)
    return {
        "jerk_rms_rad_s3": metrics["jerk_rad_s3"]["rms"],
        "expression_amplitude_joint_range_rms_rad": metrics["amplitude"][
            "joint_range_rms_rad"
        ],
        "head_velocity_rms_rad_s": metrics["head_activity"][
            "velocity_rad_s"
        ]["rms"],
        "full": metrics,
    }


def _ass_time(seconds: float) -> str:
    centiseconds = int(round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, centis = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _ass_escape(value: object) -> str:
    return (
        str(value)
        .replace("\\", "／")
        .replace("{", "(")
        .replace("}", ")")
        .replace("\n", " ")
    )


def build_ass_document(
    timeline: Sequence[Mapping[str, Any]],
    *,
    robot_width: int,
    panel_width: int,
    height: int,
) -> str:
    if len(timeline) != EXPECTED_EVENT_COUNT:
        raise ComparisonError("ASS timeline must contain 24 native events")
    width = robot_width + panel_width
    panel_left = robot_width + 24
    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        (
            "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,"
            "OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,"
            "ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,"
            "MarginR,MarginV,Encoding"
        ),
        (
            f"Style:Header,DejaVu Sans,22,&H0050CD89,&H0050CD89,"
            f"&H000F172A,&H000F172A,-1,0,0,0,100,100,0,0,1,0,0,7,"
            f"{panel_left},20,24,1"
        ),
        (
            f"Style:Body,DejaVu Sans,20,&H00FFFFFF,&H00FFFFFF,&H000F172A,"
            f"&H000F172A,0,0,0,0,100,100,0,0,1,0,0,7,"
            f"{panel_left},24,78,1"
        ),
        (
            f"Style:Receipt,DejaVu Sans,15,&H00C8D2DC,&H00C8D2DC,"
            f"&H000F172A,&H000F172A,0,0,0,0,100,100,0,0,1,0,0,1,"
            f"{panel_left},24,24,1"
        ),
        (
            f"Style:Clock,DejaVu Sans,18,&H00FFFFFF,&H00FFFFFF,&H000F172A,"
            f"&H000F172A,-1,0,0,0,100,100,0,0,1,0,0,9,"
            f"20,20,20,1"
        ),
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]
    events = [
        (
            f"Dialogue: 0,0:00:00.00,0:01:00.00,Header,,0,0,0,,"
            "SAME TEXT · HELD-OUT BEAT2 TARGET · QWEN A/B"
        ),
        (
            f"Dialogue: 0,0:00:00.00,0:01:00.00,Receipt,,0,0,0,,"
            "GT = QC-PASSED RETARGETED BEAT2 18D TARGET EXAMPLE, NOT RAW HUMAN\\N"
            "ONE-TO-MANY QUALITATIVE VIEW · NOT FRAMEWISE DETERMINISTIC TRUTH\\N"
            "ALL 3 LANES USE EXACT GT EVENT FRAMES · STATIC PADDING = 0\\N"
            "PLANNER DURATION IS SHOWN AS DIAGNOSTIC, NOT USED TO RETIME\\N"
            "JERK/EXPRESSION USE NATIVE EVENTS; EVENT BOUNDARIES EXCLUDED\\N"
            "EXPRESSION = JOINT-RANGE RMS PROXY, NOT EMOTION ACCURACY\\N"
            "NO EXTRA SMOOTHING / TIME-WARP / CROP / BLEND / RETURN-TO-ZERO\\N"
            "BEAT2 ONLY · EXTERNAL MOTION DATA FORBIDDEN"
        ),
    ]
    wrap_width = max(32, panel_width // 14)
    for item in timeline:
        start = _ass_time(float(item["start_sec"]))
        end = _ass_time(float(item["end_sec"]))
        wrapped = "\\N".join(
            textwrap.wrap(
                _ass_escape(item["prompt"]),
                width=wrap_width,
            )
        )
        metrics = item["metrics"]
        planners = item["planner_duration_diagnostics_sec"]
        label = (
            f"EVENT {int(item['index']):02d}/24  ·  "
            f"TARGET EMOTION: {_ass_escape(item['target_emotion']).upper()}\\N\\N"
            f"FULL TEXT:\\N{wrapped}\\N\\N"
            "RAW JERK RMS  rad/s³\\N"
            f"GT {metrics['gt']['jerk_rms_rad_s3']:.1f}   ·   "
            f"A {metrics['frozen_A']['jerk_rms_rad_s3']:.1f}   ·   "
            f"B {metrics['lora_B']['jerk_rms_rad_s3']:.1f}\\N\\N"
            "EXPRESSION AMPLITUDE  joint-range RMS rad\\N"
            f"GT {metrics['gt']['expression_amplitude_joint_range_rms_rad']:.3f}"
            f"   ·   A "
            f"{metrics['frozen_A']['expression_amplitude_joint_range_rms_rad']:.3f}"
            f"   ·   B "
            f"{metrics['lora_B']['expression_amplitude_joint_range_rms_rad']:.3f}"
            f"\\N\\NNATIVE {float(item['native_duration_sec']):.2f}s"
            f"   ·   PLANNER A {planners['frozen_A']:.2f}s"
            f"   ·   B {planners['lora_B']:.2f}s"
            f"\\N\\NGT clip: {_ass_escape(item['gt']['clip_id'])}\\N"
            f"seed {int(item['seed'])}"
        )
        events.append(
            f"Dialogue: 0,{start},{end},Body,,0,0,0,,{label}"
        )
    for second in range(60):
        events.append(
            "Dialogue: 1,"
            f"{_ass_time(second)},{_ass_time(second + 1)},Clock,,0,0,0,,"
            f"TIME {second:02d}.0s / 60.0s"
        )
    return "\n".join(header + events) + "\n"


def _compose_final(
    *,
    side_by_side: Path,
    ass_path: Path,
    output_path: Path,
    robot_width: int,
    panel_width: int,
    height: int,
) -> dict[str, Any]:
    ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    escaped_ass = (
        str(ass_path.resolve())
        .replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
    )
    temporary = output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.tmp.mp4"
    )
    command = [
        str(ffmpeg),
        "-y",
        "-i",
        str(side_by_side),
        "-filter_complex",
        (
            f"[0:v]scale={robot_width}:{height}[r];"
            f"[r]pad={robot_width + panel_width}:{height}:0:0:"
            f"color=0x0f172a[p];[p]ass='{escaped_ass}'[out]"
        ),
        "-map",
        "[out]",
        "-an",
        "-r",
        "30",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise ComparisonError(
            "ffmpeg final composition failed: " + completed.stderr[-2000:]
        )
    decode = subprocess.run(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-i",
            str(temporary),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    if decode.returncode != 0:
        raise ComparisonError(
            "final MP4 full decode failed: " + decode.stderr[-2000:]
        )
    frames, duration = imageio_ffmpeg.count_frames_and_secs(str(temporary))
    if abs(int(frames) - EXPECTED_FRAMES) > 1 or not math.isclose(
        float(duration), 60.0, abs_tol=1 / FPS
    ):
        raise ComparisonError(
            f"final MP4 is not 60 seconds: frames={frames}, duration={duration}"
        )
    os.replace(temporary, output_path)
    return {
        "path": str(output_path),
        "sha256": abc_video.sha256_file(output_path),
        "decoded_frames": int(frames),
        "duration_sec": float(duration),
        "full_decode_passed": True,
        "ffmpeg": str(ffmpeg),
    }


def _build_comparison(
    validated: Mapping[str, Any],
    *,
    overwrite: bool,
) -> dict[str, Any]:
    b_state = b_completion_state(validated)
    if not b_state["ready"]:
        raise ComparisonError(
            "real LoRA B completion is not available; no video was generated"
        )
    base_config = validated["_base"]
    lora_arm = base_config["qwen_generator_comparison"]["variants"][
        "lora_finetuned"
    ]
    # CPU-only, full B admission occurs before either branch samples a frame.
    # This rechecks completion, frozen-parameter audit, anti-collapse, the full
    # 54-example/six-prototype diagnostic, and BEAT2-only cache lineage.
    pre_generation_b_admission = v7_video._strict_v7_checkpoint_admission(
        Path(b_state["checkpoint"]),
        Path(b_state["summary"]),
        expected_manifest_sha256=base_config["source_manifest_sha256"],
        expected_qwen_cache_sha256=lora_arm["condition_cache_sha256"],
        minimum_training_steps=int(base_config["minimum_training_steps"]),
        expected_training_policy=TRAINING_POLICY,
        gate_config=base_config["six_emotion_full_diagnostic_gate"],
    )
    output_dir = Path(validated["_output_dir"])
    summary_path = output_dir / SUMMARY_FILENAME
    final_video = output_dir / FINAL_VIDEO_FILENAME
    if (summary_path.exists() or final_video.exists()) and not overwrite:
        raise ComparisonError(
            "comparison output already exists; pass --overwrite to rebuild"
        )

    plan = prepare_plan(validated)
    if plan["lora_B"]["ready"] is not True:
        raise ComparisonError("LoRA B changed state before branch generation")
    prompts = [item["prompt"] for item in plan["selections"]]
    seeds = [int(item["seed"]) for item in plan["selections"]]
    manifest_records, _ = abc_video.load_manifest(
        plan["source_manifest"],
        expected_sha256=plan["source_manifest_sha256"],
    )
    shared_noises = [
        style_video.shared_initial_noise(
            seed=int(selection["seed"]),
            frames=int(selection["frames"]),
        )
        for selection in plan["selections"]
    ]

    # Both branches are sampled at each GT event's exact native frame count.
    # GT action values remain unopened until both generator branches finish.
    frozen = _generate_branch(
        validated,
        variant="frozen_base",
        plan=plan,
        manifest_records=manifest_records,
        shared_noises=shared_noises,
    )
    lora = _generate_branch(
        validated,
        variant="lora_finetuned",
        plan=plan,
        manifest_records=manifest_records,
        shared_noises=shared_noises,
    )
    noise_receipts = _same_noise_prefix(
        frozen["noise_segments"], lora["noise_segments"]
    )

    # Only now load the sealed GT actions; they cannot affect branch sampling.
    gt_raw_segments: list[np.ndarray] = []
    timeline: list[dict[str, Any]] = []
    for index, selection in enumerate(plan["selections"]):
        record = manifest_records[selection["reference"]["clip_id"]]
        values, gt_receipt = _validate_gt_csv(record)
        expected_frames = int(selection["frames"])
        if (
            len(values) != expected_frames
            or len(frozen["raw_segments"][index]) != expected_frames
            or len(lora["raw_segments"][index]) != expected_frames
        ):
            raise ComparisonError(
                f"GT/A/B event frame mismatch at event {index + 1}"
            )
        gt_raw_segments.append(values)
        timeline.append(
            {
                **deepcopy(selection),
                "gt": gt_receipt,
                "metrics": {
                    "gt": _metric_receipt(values),
                    "frozen_A": _metric_receipt(
                        frozen["raw_segments"][index]
                    ),
                    "lora_B": _metric_receipt(
                        lora["raw_segments"][index]
                    ),
                },
                "planner_duration_diagnostics_sec": {
                    "frozen_A": frozen["events"][index][
                        "planner_predicted_duration_sec"
                    ],
                    "lora_B": lora["events"][index][
                        "planner_predicted_duration_sec"
                    ],
                },
                "same_initial_noise": noise_receipts[index],
                "static_padding_frames": 0,
                "endpoint_hold_frames": 0,
                "additional_smoothing": False,
                "time_warp": False,
            }
        )
    gt_display = np.concatenate(gt_raw_segments, axis=0)
    if gt_display.shape != (EXPECTED_FRAMES, ACTION_DIM):
        raise ComparisonError("GT display shape changed")

    render = validated["render"]
    pane_width = int(render["pane_width"])
    pane_height = int(render["pane_height"])
    panel_width = int(render["panel_width"])
    output_dir.mkdir(parents=True, exist_ok=True)
    render_receipts = {}
    videos = []
    for name, trajectory in (
        ("gt", gt_display),
        ("frozen_A", frozen["display"]),
        ("lora_B", lora["display"]),
    ):
        csv_path = output_dir / f"{name}_native_events_no_padding.csv"
        mp4_path = output_dir / f"{name}_robot.mp4"
        render_receipts[name] = abc_video._render_single(
            trajectory,
            csv_path=csv_path,
            mp4_path=mp4_path,
            fps=FPS,
            width=pane_width,
            height=pane_height,
            simplified=bool(render["simplified"]),
        )
        videos.append(mp4_path)
    side_by_side = output_dir / "GT_A_B_fixed_camera.mp4"
    stack_receipt = abc_video.build_side_by_side(
        videos,
        side_by_side,
        labels=(
            "GT · HELD-OUT BEAT2 18D",
            "A · FROZEN QWEN",
            "B · LORA QWEN",
        ),
        pane_width=pane_width,
    )
    ass_path = output_dir / "text_emotion_metrics_timeline.ass"
    ass_path.write_text(
        build_ass_document(
            timeline,
            robot_width=pane_width * 3,
            panel_width=panel_width,
            height=pane_height + 40,
        ),
        encoding="utf-8",
    )
    final_receipt = _compose_final(
        side_by_side=side_by_side,
        ass_path=ass_path,
        output_path=final_video,
        robot_width=pane_width * 3,
        panel_width=panel_width,
        height=pane_height + 40,
    )
    trajectory_path = output_dir / "comparison_trajectories.npz"
    _atomic_npz(
        trajectory_path,
        gt_native_events=gt_display,
        frozen_A_fixed_gt_event_frames=frozen["display"],
        lora_B_fixed_gt_event_frames=lora["display"],
        prompts=np.asarray(prompts),
        target_emotions=np.asarray(
            [item["target_emotion"] for item in timeline]
        ),
        seeds=np.asarray(seeds, dtype=np.int64),
        event_frames=np.asarray(
            [item["frames"] for item in timeline], dtype=np.int64
        ),
        event_offsets=np.asarray(
            [0]
            + [
                int(item["end_frame"])
                for item in timeline
            ],
            dtype=np.int64,
        ),
        frozen_A_planner_duration_sec=np.asarray(
            [
                item["planner_duration_diagnostics_sec"]["frozen_A"]
                for item in timeline
            ],
            dtype=np.float32,
        ),
        lora_B_planner_duration_sec=np.asarray(
            [
                item["planner_duration_diagnostics_sec"]["lora_B"]
                for item in timeline
            ],
            dtype=np.float32,
        ),
        fps=np.asarray(FPS, dtype=np.float32),
        static_padding_frames=np.asarray(0, dtype=np.int64),
        endpoint_hold_frames=np.asarray(0, dtype=np.int64),
        joint_order=np.asarray(JOINT_ORDER_18D),
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": SUMMARY_ARTIFACT_KIND,
        "status": "complete",
        "experimental_only": True,
        "formal_release_eligible": False,
        "created_utc": utc_now(),
        "config": validated["_config_path"],
        "config_sha256": validated["_config_sha256"],
        "data_policy": DATA_POLICY,
        "no_external_data": True,
        "no_kimodo": True,
        "no_hanyang": True,
        "target_duration_sec": 60.0,
        "decoded_frames_expected": EXPECTED_FRAMES,
        "native_event_count": EXPECTED_EVENT_COUNT,
        "static_padding_frames": 0,
        "endpoint_hold_frames": 0,
        "comparison_contract": deepcopy(COMPARISON_CONTRACT),
        "metric_contract": deepcopy(METRIC_CONTRACT),
        "gt_disclosure": {
            "representation": "held_out_BEAT2_physical_qc_passed_18d_robot_motion",
            "raw_human_skeleton": False,
            "paired_deterministic_ground_truth": False,
            "role": "same_prompt_qualitative_target_example",
            "selection_before_visual_review": True,
            "loaded_after_generator_sampling": True,
            "generation_input": False,
        },
        "timeline": timeline,
        "branches": {
            branch["variant"]: {
                "derived_config": branch["derived_config"],
                "derived_config_sha256": branch[
                    "derived_config_sha256"
                ],
                "checkpoint_admission": branch["checkpoint_admission"],
                "condition_cache": branch["condition_cache"],
                "events": branch["events"],
                "fixed_frames_equal_gt": True,
                "planner_duration_diagnostic_only": True,
                "static_padding_frames": 0,
                "endpoint_hold_frames": 0,
            }
            for branch in (frozen, lora)
        },
        "pre_generation_lora_B_admission": pre_generation_b_admission,
        "lora_B_admitted_before_any_generator_sampling": True,
        "same_initial_noise_receipts": noise_receipts,
        "metrics": {
            "aggregate_raw_segment_mean": {
                lane: {
                    metric: float(
                        np.mean(
                            [
                                item["metrics"][lane][metric]
                                for item in timeline
                            ]
                        )
                    )
                    for metric in (
                        "jerk_rms_rad_s3",
                        "expression_amplitude_joint_range_rms_rad",
                        "head_velocity_rms_rad_s",
                    )
                }
                for lane in ("gt", "frozen_A", "lora_B")
            },
            "full_montage_derivatives_not_used_for_claims": True,
            "event_boundary_derivatives_excluded": True,
            "static_padding_frames": 0,
            "endpoint_hold_frames": 0,
            "emotion_accuracy_claimed": False,
        },
        "artifacts": {
            "comparison_trajectories": {
                "path": str(trajectory_path),
                "sha256": abc_video.sha256_file(trajectory_path),
            },
            "text_timeline_ass": {
                "path": str(ass_path),
                "sha256": abc_video.sha256_file(ass_path),
            },
            "side_by_side_robot_video": stack_receipt,
            "final_video": final_receipt,
            "render_receipts": render_receipts,
        },
    }
    summary["sha256"] = abc_video.value_sha256(summary)
    _atomic_json(summary_path, summary)
    return summary


def wait_for_completion_and_build(
    validated: Mapping[str, Any],
    *,
    overwrite: bool,
    poll_seconds: float,
    timeout_seconds: float | None,
) -> dict[str, Any]:
    if (
        not math.isfinite(poll_seconds)
        or poll_seconds <= 0
        or (
            timeout_seconds is not None
            and (
                not math.isfinite(timeout_seconds)
                or timeout_seconds <= 0
            )
        )
    ):
        raise ComparisonError("poll and timeout must be positive")
    started = time.monotonic()
    while True:
        state = b_completion_state(validated)
        if state["ready"]:
            return _build_comparison(validated, overwrite=overwrite)
        if (
            timeout_seconds is not None
            and time.monotonic() - started >= timeout_seconds
        ):
            raise ComparisonError("timed out before real LoRA B completion")
        time.sleep(poll_seconds)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--wait-for-completion", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config_path = args.config.resolve()
        config = _load_json(config_path, field="comparison config")
        validated = validate_config(config, config_path=config_path)
        plan = prepare_plan(validated)
        if args.prepare_only:
            print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.wait_for_completion:
            summary = wait_for_completion_and_build(
                validated,
                overwrite=args.overwrite,
                poll_seconds=float(args.poll_seconds),
                timeout_seconds=args.timeout_seconds,
            )
        else:
            summary = _build_comparison(
                validated,
                overwrite=args.overwrite,
            )
    except (
        ComparisonError,
        v7_video.EmotionHierarchyVideoError,
        style_video.StyleEmotionVideoError,
        abc_video.EvaluationContractError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": summary["status"],
                "video": summary["artifacts"]["final_video"]["path"],
                "video_sha256": summary["artifacts"]["final_video"]["sha256"],
                "duration_sec": summary["artifacts"]["final_video"][
                    "duration_sec"
                ],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
