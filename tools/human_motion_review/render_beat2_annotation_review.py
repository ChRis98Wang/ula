#!/usr/bin/env python3
"""Render and verify silent MuJoCo videos for a BEAT2 annotation review queue.

This stage is deliberately separate from training admission. A successful render
only proves that the queued 18D trajectory can be inspected as a valid video; it
does not prove that the draft text matches the motion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import imageio_ffmpeg


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REVIEW_QUEUE = (
    PROJECT_ROOT
    / "deliverables/interactive_human_motion_v1/batch_annotation_v1/review/review_queue.jsonl"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_REVIEW_QUEUE.parent / "videos_v1"
ENV_ISAACLAB_PYTHON = Path("/home/gez/miniconda3/envs/env_isaaclab/bin/python")
DEFAULT_RENDERER_PYTHON = (
    ENV_ISAACLAB_PYTHON if ENV_ISAACLAB_PYTHON.is_file() else Path(sys.executable)
)
DEFAULT_URDF = (
    PROJECT_ROOT
    / "urdf_V2_20260514/urdf/xacro/robot_modify_meshdir.urdf"
)

SCHEMA_VERSION = "1.0.0"
ROBOT_CONTRACT = "ula_v2_18d_head_v1"
FPS = 30.0
CAMERA_MARGIN = 1.12
CAMERA_LOOKAT_Z_OFFSET = -0.06
MIN_VIDEO_BYTES = 1024
TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]+$")
SEMANTIC_AUDIT_FIELDS = (
    "fixed_split_assignment",
    "canonical_prompt_role",
    "canonical_action_role",
    "official_category_verified",
    "official_category_role",
    "official_category_condition_channel",
    "official_category_loss",
    "official_category_conditioning_enabled",
    "robot_observable_motion_form",
    "communicative_intent",
    "semantic_supervision_masks",
    "semantic_event",
    "semantic_mapping_status",
    "emotion_id",
    "emotion_supervision_mask",
    "source_emotion_label_verified",
    "emotion_supervision_role",
    "official_emotion_conditioning_enabled",
    "official_emotion_condition_channel",
    "official_emotion_loss",
    "affect_observable_review_status",
    "affect_observable_supervision_mask",
    "semantic_action_completeness_review_required",
    "affect_observable_review_required",
    "retarget_segment",
    "safety_monotonic_retime",
    "trajectory_frames_expected",
    "source_frame_count",
    "output_frame_count",
    "final_trajectory_role",
    "blind_review_must_use_final_trajectory",
    "upstream_inventory_record_sha256",
    "selected_record_sha256",
    "retarget_input_manifest_sha256",
    "expression_turn",
    "context_plan",
    "training_segment",
    "time_axes",
    "expression_turn_contract_sha256",
    "expression_turn_record_sha256",
    "expression_turn_selection_kind",
    "expression_turn_selection_rank",
    "expression_turn_selection_status",
    "expression_turn_selection_record_sha256",
    "source_inventory_manifest_sha256",
    "split_assignment_manifest_sha256",
    "upstream_event_record_sha256",
)

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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_sha256(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    atomic_text(path, "".join(stable_json(record) + "\n" for record in records))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            records.append(value)
    return records


def _require_string(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"review record requires non-empty {key}")
    return value


def validate_queue_structure(records: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for record in records:
        task_id = _require_string(record, "task_id")
        if not TASK_ID_PATTERN.fullmatch(task_id) or task_id in {".", ".."}:
            raise ValueError(f"unsafe task_id: {task_id!r}")
        if task_id in seen:
            raise ValueError(f"duplicate task_id in review queue: {task_id}")
        seen.add(task_id)
        _require_string(record, "speaker_key")
        _require_string(record, "official_split")
        _require_string(record, "trajectory_path")
        _require_string(record, "trajectory_sha256")
        if record.get("robot_contract") != ROBOT_CONTRACT:
            raise ValueError(f"{task_id}: robot_contract must be {ROBOT_CONTRACT}")
        if record.get("accepted_for_training") is not False:
            raise ValueError(f"{task_id}: accepted_for_training must remain false")
        if record.get("manual_review_required") is not True:
            raise ValueError(f"{task_id}: manual_review_required must be true")
        prompt = record.get("canonical_prompt")
        if not isinstance(prompt, dict) or not all(
            isinstance(prompt.get(language), str) and prompt[language].strip()
            for language in ("en", "zh")
        ):
            raise ValueError(f"{task_id}: canonical_prompt requires non-empty en/zh")


def _semantic_key(record: dict[str, Any]) -> str:
    features = record.get("observable_features")
    if isinstance(features, dict):
        arm = features.get("arm") if isinstance(features.get("arm"), dict) else {}
        amplitude = arm.get("amplitude", arm.get("scale"))
        continuity = arm.get("continuity", arm.get("activity"))
        coordination = arm.get(
            "bilateral_temporally_coordinated", arm.get("coordination")
        )
        parts = [
            f"laterality={arm.get('laterality')}",
            f"amplitude={amplitude}",
            f"continuity={continuity}",
            f"bilateral_coordinated={coordination}",
            f"regularly_repeated={arm.get('regularly_repeated')}",
            f"head={features.get('head_motion')}",
            f"torso={features.get('torso_motion')}",
        ]
        key = "/".join(part for part in parts if not part.endswith("=None"))
        if key:
            return key
    action = record.get("canonical_action")
    if isinstance(action, str) and action:
        return action
    return "unspecified_robot_observable_motion"


def sampling_stratum(record: dict[str, Any]) -> tuple[str, str, str]:
    event = record.get("semantic_event")
    emotion = record.get("emotion_id")
    if isinstance(event, dict) and emotion:
        category = event.get("category")
        intensity = event.get("intensity")
        if category and intensity:
            return (
                str(record["speaker_key"]),
                str(
                    record.get("fixed_split_assignment")
                    or record["official_split"]
                ),
                f"{emotion}/{category}/{intensity}",
            )
    return (
        str(record["speaker_key"]),
        str(record["official_split"]),
        _semantic_key(record),
    )


def _stable_rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()


def duration_span_sec(record: dict[str, Any]) -> float:
    """Return the planner duration for a native variable-length event."""
    segment = record.get("retarget_segment")
    value = segment.get("output_sample_span_sec") if isinstance(segment, dict) else None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(
            f"{record.get('task_id', '<unknown>')}: missing output_sample_span_sec"
        )
    span = float(value)
    if not math.isfinite(span) or span < 0.0:
        raise ValueError(
            f"{record.get('task_id', '<unknown>')}: invalid output_sample_span_sec"
        )
    return span


def _duration_quantile_records(
    records: list[dict[str, Any]], limit: int | None
) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda item: (duration_span_sec(item), item["task_id"]))
    if limit is None or limit >= len(ordered):
        return ordered
    if limit == 1:
        return [ordered[len(ordered) // 2]]

    # Inclusive endpoints make a small review set prove the actual duration range.
    indices = [round(rank * (len(ordered) - 1) / (limit - 1)) for rank in range(limit)]
    return [ordered[index] for index in indices]


def select_records(
    records: list[dict[str, Any]],
    *,
    limit: int | None,
    sampling: str,
    seed: int,
) -> list[dict[str, Any]]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if sampling == "sequential":
        ordered = sorted(records, key=lambda item: item["task_id"])
    elif sampling == "duration_quantiles":
        return _duration_quantile_records(records, limit)
    elif sampling == "stratified":
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            groups[sampling_stratum(record)].append(record)
        for group in groups.values():
            group.sort(key=lambda item: _stable_rank(seed, str(item["task_id"])))
        group_keys = sorted(
            groups,
            key=lambda key: (_stable_rank(seed, "\0".join(key)), key),
        )
        ordered = []
        offset = 0
        while True:
            added = False
            for key in group_keys:
                group = groups[key]
                if offset < len(group):
                    ordered.append(group[offset])
                    added = True
            if not added:
                break
            offset += 1
    else:
        raise ValueError(
            "sampling must be 'stratified', 'sequential', or 'duration_quantiles'"
        )
    return ordered if limit is None else ordered[:limit]


def resolve_evidence_path(value: Any, queue_path: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing {field}")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (queue_path.parent / path).resolve()


def validate_trajectory(record: dict[str, Any], queue_path: Path) -> tuple[Path, int]:
    task_id = record["task_id"]
    path = resolve_evidence_path(record.get("trajectory_path"), queue_path, "trajectory_path")
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_hash = record.get("trajectory_sha256")
    actual_hash = sha256(path)
    if expected_hash != actual_hash:
        raise ValueError(f"{task_id}: trajectory_sha256 mismatch")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"{task_id}: empty trajectory CSV") from error
        if header != JOINT_ORDER:
            raise ValueError(f"{task_id}: trajectory is not the ordered ULA V2 18D contract")
        frames = 0
        for line_number, row in enumerate(reader, 2):
            if len(row) != len(JOINT_ORDER):
                raise ValueError(f"{task_id}: CSV row {line_number} is not 18D")
            try:
                values = [float(value) for value in row]
            except ValueError as error:
                raise ValueError(f"{task_id}: non-numeric CSV row {line_number}") from error
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{task_id}: non-finite CSV row {line_number}")
            frames += 1
    if frames < 1:
        raise ValueError(f"{task_id}: trajectory contains no frames")
    expected_frames = record.get("trajectory_frames_expected")
    if expected_frames is not None and (
        isinstance(expected_frames, bool)
        or not isinstance(expected_frames, int)
        or frames != expected_frames
    ):
        raise ValueError(f"{task_id}: trajectory frames do not match queue contract")

    quality_value = record.get("quality_json")
    quality_hash = record.get("quality_json_sha256")
    if quality_value is not None or quality_hash is not None:
        quality_path = resolve_evidence_path(quality_value, queue_path, "quality_json")
        if not quality_path.is_file():
            raise FileNotFoundError(quality_path)
        if not isinstance(quality_hash, str) or sha256(quality_path) != quality_hash:
            raise ValueError(f"{task_id}: quality_json_sha256 mismatch")
    return path, frames


def render_config(
    *, width: int, height: int, urdf: Path, renderer_python: Path
) -> dict[str, Any]:
    return {
        "renderer": "upper_body_skeleton.mujoco_playback",
        "renderer_python": str(renderer_python),
        "model": "real_ula_v2_urdf",
        "urdf": str(urdf),
        "robot_contract": ROBOT_CONTRACT,
        "fps": FPS,
        "width": int(width),
        "height": int(height),
        "camera_mode": "auto_full_body",
        "camera_margin": CAMERA_MARGIN,
        "camera_lookat_z_offset": CAMERA_LOOKAT_Z_OFFSET,
        "audio": "none",
    }


def build_renderer_command(
    *,
    renderer_python: Path,
    trajectory: Path,
    output_mp4: Path,
    summary_json: Path,
    urdf: Path,
    width: int,
    height: int,
) -> list[str]:
    return [
        str(renderer_python),
        "-m",
        "upper_body_skeleton.mujoco_playback",
        "--joint-csv",
        str(trajectory),
        "--output-mp4",
        str(output_mp4),
        "--summary-json",
        str(summary_json),
        "--fps",
        str(FPS),
        "--width",
        str(width),
        "--height",
        str(height),
        "--camera-margin",
        str(CAMERA_MARGIN),
        "--camera-lookat-z-offset",
        str(CAMERA_LOOKAT_Z_OFFSET),
        "--urdf",
        str(urdf),
    ]


def _stream_report(path: Path) -> dict[str, Any]:
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    report = completed.stderr
    stream_lines = [line.strip() for line in report.splitlines() if "Stream #" in line]
    video_lines = [line for line in stream_lines if " Video:" in line]
    audio_lines = [line for line in stream_lines if " Audio:" in line]
    if len(video_lines) != 1:
        raise ValueError(f"expected exactly one video stream, found {len(video_lines)}")
    if audio_lines:
        raise ValueError(f"review MP4 must be silent, found {len(audio_lines)} audio stream(s)")
    if "Video: h264" not in video_lines[0] or "yuv420p" not in video_lines[0]:
        raise ValueError("review MP4 must use broadly decodable H.264 yuv420p")
    return {
        "video_streams": len(video_lines),
        "audio_streams": len(audio_lines),
        "codec": "h264",
        "pixel_format": "yuv420p",
    }


def validate_video(
    path: Path,
    *,
    expected_frames: int,
    expected_width: int,
    expected_height: int,
    expected_fps: float = FPS,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    file_bytes = path.stat().st_size
    if file_bytes < MIN_VIDEO_BYTES:
        raise ValueError(f"video is empty or truncated: {file_bytes} bytes")
    mp4_bytes = path.read_bytes()
    moov_offset = mp4_bytes.find(b"moov")
    mdat_offset = mp4_bytes.find(b"mdat")
    if moov_offset < 0 or mdat_offset < 0 or moov_offset >= mdat_offset:
        raise ValueError("review MP4 must place the moov atom before mdat for faststart")
    stream_check = _stream_report(path)
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    try:
        metadata = next(reader)
        size = tuple(int(value) for value in metadata.get("size", ()))
        if size != (int(expected_width), int(expected_height)):
            raise ValueError(f"decoded video size {size} != expected {(expected_width, expected_height)}")
        fps = float(metadata.get("fps") or 0.0)
        if not math.isclose(fps, expected_fps, rel_tol=0.0, abs_tol=1e-3):
            raise ValueError(f"decoded video FPS {fps} != expected {expected_fps}")
        expected_frame_bytes = int(expected_width) * int(expected_height) * 3
        decoded_frames = 0
        pixel_min = 255
        pixel_max = 0
        for frame in reader:
            if len(frame) != expected_frame_bytes:
                raise ValueError(
                    f"decoded frame has {len(frame)} bytes, expected {expected_frame_bytes}"
                )
            decoded_frames += 1
            pixel_min = min(pixel_min, min(frame))
            pixel_max = max(pixel_max, max(frame))
    finally:
        reader.close()
    if decoded_frames != expected_frames:
        raise ValueError(f"decoded {decoded_frames} frames, expected {expected_frames}")
    if decoded_frames < 1 or pixel_max - pixel_min < 8:
        raise ValueError("decoded video is visually blank")
    return {
        "passed": True,
        "fully_decodable": True,
        "nonblank": True,
        "file_bytes": file_bytes,
        "decoded_frames": decoded_frames,
        "expected_frames": int(expected_frames),
        "width": int(expected_width),
        "height": int(expected_height),
        "fps": fps,
        "pixel_value_min": pixel_min,
        "pixel_value_max": pixel_max,
        "faststart": True,
        "moov_offset": moov_offset,
        "mdat_offset": mdat_offset,
        **stream_check,
    }


def validate_render_summary(
    summary: dict[str, Any],
    *,
    expected_frames: int,
    expected_width: int,
    expected_height: int,
    expected_urdf: Path,
) -> None:
    checks = {
        "output_contract": summary.get("output_contract") == ROBOT_CONTRACT,
        "action_dim": summary.get("action_dim") == len(JOINT_ORDER),
        "joint_order": summary.get("joint_order") == JOINT_ORDER,
        "frames": summary.get("frames") == expected_frames,
        "fps": math.isclose(float(summary.get("fps", 0.0)), FPS, abs_tol=1e-6),
        "width": summary.get("width") == expected_width,
        "height": summary.get("height") == expected_height,
        "model_source": Path(str(summary.get("model_source", ""))).resolve()
        == expected_urdf.resolve(),
    }
    framing = summary.get("camera_framing")
    checks.update(
        {
            "camera_mode": isinstance(framing, dict)
            and framing.get("mode") == "auto_full_body",
            "camera_margin": isinstance(framing, dict)
            and math.isclose(
                float(framing.get("margin", 0.0)), CAMERA_MARGIN, abs_tol=1e-6
            ),
            "camera_z_offset": isinstance(framing, dict)
            and math.isclose(
                float(framing.get("lookat_z_offset", 0.0)),
                CAMERA_LOOKAT_Z_OFFSET,
                abs_tol=1e-6,
            ),
            "subject_screen_bias": isinstance(framing, dict)
            and framing.get("subject_screen_bias") == "up",
        }
    )
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"render summary contract mismatch: {failed}")


def review_projection(
    record: dict[str, Any], *, rank: int, sampling: str, seed: int
) -> dict[str, Any]:
    projected = {
        "schema_version": SCHEMA_VERSION,
        "task_id": record["task_id"],
        "source_clip_id": record.get("source_clip_id"),
        "speaker_key": record["speaker_key"],
        "official_split": record["official_split"],
        "robot_contract": ROBOT_CONTRACT,
        "canonical_action": record.get("canonical_action"),
        "canonical_prompt": record["canonical_prompt"],
        "trajectory_path": record["trajectory_path"],
        "trajectory_sha256": record["trajectory_sha256"],
        "quality_json": record.get("quality_json"),
        "quality_json_sha256": record.get("quality_json_sha256"),
        "sample_rank": rank,
        "sampling": sampling,
        "sampling_seed": seed,
        "sampling_stratum": list(sampling_stratum(record)),
        "review_state": "pending_independent_motion_text_video_review",
        "manual_review_required": True,
        "accepted_for_training": False,
        "speech_context_included": False,
        "render_pass_grants_training_admission": False,
    }
    if isinstance(record.get("observable_features"), dict):
        projected["observable_features"] = record["observable_features"]
    projected.update(
        {field: record[field] for field in SEMANTIC_AUDIT_FIELDS if field in record}
    )
    projected["input_record_sha256"] = value_sha256(record)
    return projected


def _load_result(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _resumable_result(
    result_path: Path,
    *,
    input_fingerprint: str,
    config_fingerprint: str,
    retry_failed: bool,
    width: int,
    height: int,
) -> dict[str, Any] | None:
    previous = _load_result(result_path)
    if not previous:
        return None
    if previous.get("input_fingerprint") != input_fingerprint:
        return None
    if previous.get("render_config_fingerprint") != config_fingerprint:
        return None
    if previous.get("status") == "failed" and not retry_failed:
        reused = dict(previous)
        reused["resume_reused"] = True
        return reused
    if previous.get("status") != "passed":
        return None
    video_path = Path(str(previous.get("video_path", "")))
    expected_frames = int(previous.get("trajectory_frames", 0))
    try:
        check = validate_video(
            video_path,
            expected_frames=expected_frames,
            expected_width=width,
            expected_height=height,
        )
    except (FileNotFoundError, OSError, ValueError):
        return None
    reused = dict(previous)
    reused["video_check"] = check
    reused["resume_reused"] = True
    return reused


def render_one(
    sample: dict[str, Any],
    *,
    queue_path: Path,
    output_root: Path,
    renderer_python: Path,
    urdf: Path,
    width: int,
    height: int,
    run_id: str,
    resume: bool,
    retry_failed: bool,
) -> dict[str, Any]:
    task_id = sample["task_id"]
    config = render_config(
        width=width,
        height=height,
        urdf=urdf,
        renderer_python=renderer_python,
    )
    input_fingerprint = sample["input_record_sha256"]
    config_fingerprint = value_sha256(config)
    result_path = output_root / "results" / f"{task_id}.json"
    if resume:
        reusable = _resumable_result(
            result_path,
            input_fingerprint=input_fingerprint,
            config_fingerprint=config_fingerprint,
            retry_failed=retry_failed,
            width=width,
            height=height,
        )
        if reusable is not None:
            atomic_json(result_path, reusable)
            return reusable

    log_path = output_root / "logs" / task_id / f"{run_id}.log"
    video_path = output_root / "videos" / f"{task_id}.mp4"
    render_summary_path = output_root / "render_summaries" / f"{task_id}.json"
    staging = output_root / "staging" / run_id / task_id
    staging_video = staging / f"{task_id}.mp4"
    staging_summary = staging / f"{task_id}.render.json"
    staging.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    base = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "source_clip_id": sample.get("source_clip_id"),
        "speaker_key": sample["speaker_key"],
        "official_split": sample["official_split"],
        "canonical_action": sample.get("canonical_action"),
        "canonical_prompt": sample["canonical_prompt"],
        "robot_contract": ROBOT_CONTRACT,
        "input_fingerprint": input_fingerprint,
        "render_config": config,
        "render_config_fingerprint": config_fingerprint,
        "started_at": started_at,
        "accepted_for_training": False,
        "manual_review_required": True,
        "render_pass_grants_training_admission": False,
        "speech_context_included": False,
        "resume_reused": False,
    }
    base.update(
        {field: sample[field] for field in SEMANTIC_AUDIT_FIELDS if field in sample}
    )
    command: list[str] | None = None
    returncode: int | None = None
    try:
        trajectory, trajectory_frames = validate_trajectory(sample, queue_path)
        command = build_renderer_command(
            renderer_python=renderer_python,
            trajectory=trajectory,
            output_mp4=staging_video,
            summary_json=staging_summary,
            urdf=urdf,
            width=width,
            height=height,
        )
        environment = os.environ.copy()
        environment["MUJOCO_GL"] = "egl"
        environment["PYOPENGL_PLATFORM"] = "egl"
        current_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(PROJECT_ROOT)
            if not current_pythonpath
            else str(PROJECT_ROOT) + os.pathsep + current_pythonpath
        )
        with log_path.open("w", encoding="utf-8") as log:
            log.write("command=" + stable_json(command) + "\n")
            log.flush()
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        returncode = completed.returncode
        if returncode != 0:
            raise RuntimeError(f"renderer returned {returncode}")
        try:
            render_summary = json.loads(staging_summary.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid renderer summary: {error}") from error
        if not isinstance(render_summary, dict):
            raise ValueError("renderer summary is not a JSON object")
        validate_render_summary(
            render_summary,
            expected_frames=trajectory_frames,
            expected_width=width,
            expected_height=height,
            expected_urdf=urdf,
        )
        video_check = validate_video(
            staging_video,
            expected_frames=trajectory_frames,
            expected_width=width,
            expected_height=height,
        )
        video_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging_video, video_path)
        render_summary["output_mp4"] = str(video_path.resolve())
        render_summary["video_verification"] = video_check
        atomic_json(render_summary_path, render_summary)
        final_output_binding = {
            "trajectory_path": str(trajectory),
            "trajectory_sha256": sha256(trajectory),
            "output_frame_count": trajectory_frames,
            "fps": FPS,
            "video_path": str(video_path.resolve()),
            "video_sha256": sha256(video_path),
            "video_decoded_frames": video_check["decoded_frames"],
        }
        final_output_binding["sha256"] = value_sha256(final_output_binding)
        result = {
            **base,
            "status": "passed",
            "review_state": "video_render_verified_pending_independent_motion_text_review",
            "finished_at": utc_now(),
            "renderer_returncode": returncode,
            "trajectory_path": str(trajectory),
            "trajectory_sha256": final_output_binding["trajectory_sha256"],
            "trajectory_frames": trajectory_frames,
            "video_path": str(video_path.resolve()),
            "video_sha256": final_output_binding["video_sha256"],
            "video_check": video_check,
            "final_output_binding": final_output_binding,
            "render_summary_path": str(render_summary_path.resolve()),
            "render_summary_sha256": sha256(render_summary_path),
            "log_path": str(log_path.resolve()),
            "log_sha256": sha256(log_path),
        }
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
        if not log_path.exists():
            atomic_text(log_path, f"controller_error={error}\n")
        result = {
            **base,
            "status": "failed",
            "review_state": "video_render_or_verification_failed",
            "finished_at": utc_now(),
            "renderer_returncode": returncode,
            "error": f"{type(error).__name__}: {error}",
            "command": command,
            "log_path": str(log_path.resolve()),
            "log_sha256": sha256(log_path),
        }
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    atomic_json(result_path, result)
    return result


def _counts(results: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "passed": sum(result.get("status") == "passed" for result in results),
        "failed": sum(result.get("status") == "failed" for result in results),
        "resume_reused": sum(result.get("resume_reused") is True for result in results),
    }


def run_review(
    *,
    queue_path: Path,
    output_root: Path,
    renderer_python: Path,
    urdf: Path,
    limit: int | None,
    sampling: str,
    seed: int,
    workers: int,
    width: int,
    height: int,
    resume: bool,
    retry_failed: bool,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    if width < 2 or height < 2 or width % 2 or height % 2:
        raise ValueError("video width and height must be positive even integers")
    records = read_jsonl(queue_path)
    validate_queue_structure(records)
    selected = select_records(
        records,
        limit=limit,
        sampling=sampling,
        seed=seed,
    )
    samples = [
        review_projection(record, rank=index, sampling=sampling, seed=seed)
        for index, record in enumerate(selected)
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    sample_manifest = output_root / "sampled_manifest.jsonl"
    atomic_jsonl(sample_manifest, samples)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    started_at = utc_now()
    config = render_config(
        width=width,
        height=height,
        urdf=urdf,
        renderer_python=renderer_python,
    )
    results: list[dict[str, Any]] = []
    status_path = output_root / "status.json"

    def write_status(state: str) -> None:
        counts = _counts(results)
        atomic_json(
            status_path,
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "run_state": state,
                "started_at": started_at,
                "updated_at": utc_now(),
                "queue_records": len(records),
                "selected_records": len(samples),
                "completed_records": len(results),
                "pending_records": len(samples) - len(results),
                "counts": counts,
                "accepted_for_training": 0,
            },
        )

    write_status("running")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                render_one,
                sample,
                queue_path=queue_path,
                output_root=output_root,
                renderer_python=renderer_python,
                urdf=urdf,
                width=width,
                height=height,
                run_id=run_id,
                resume=resume,
                retry_failed=retry_failed,
            ): sample["task_id"]
            for sample in samples
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                stable_json(
                    {
                        "task_id": result["task_id"],
                        "status": result["status"],
                        "completed": len(results),
                        "selected": len(samples),
                    }
                ),
                flush=True,
            )
            write_status("running")

    results.sort(key=lambda item: item["task_id"])
    passed = [result for result in results if result["status"] == "passed"]
    failed = [result for result in results if result["status"] == "failed"]
    passed_manifest = output_root / "passed_manifest.jsonl"
    failed_manifest = output_root / "failed_manifest.jsonl"
    atomic_jsonl(passed_manifest, passed)
    atomic_jsonl(failed_manifest, failed)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": "beat2_annotation_review_video_render",
        "run_id": run_id,
        "run_state": "finished" if not failed else "finished_with_failures",
        "started_at": started_at,
        "finished_at": utc_now(),
        "review_queue": str(queue_path.resolve()),
        "review_queue_sha256": sha256(queue_path),
        "queue_records": len(records),
        "sampling": {
            "mode": sampling,
            "seed": seed,
            "limit": limit,
            "selected_records": len(samples),
            "sampled_manifest": str(sample_manifest.resolve()),
            "sampled_manifest_sha256": sha256(sample_manifest),
        },
        "render_config": config,
        "render_config_fingerprint": value_sha256(config),
        "counts": _counts(results),
        "passed_manifest": str(passed_manifest.resolve()),
        "passed_manifest_sha256": sha256(passed_manifest),
        "failed_manifest": str(failed_manifest.resolve()),
        "failed_manifest_sha256": sha256(failed_manifest),
        "manual_motion_text_review_still_required": True,
        "render_pass_grants_training_admission": False,
        "accepted_for_training": 0,
        "speech_context_used_as_action_prompt": False,
    }
    atomic_json(output_root / "summary.json", summary)
    write_status(summary["run_state"])
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-queue", type=Path, default=DEFAULT_REVIEW_QUEUE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--renderer-python", type=Path, default=DEFAULT_RENDERER_PYTHON)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--sampling",
        choices=("stratified", "sequential", "duration_quantiles"),
        default="stratified",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args(argv)


def _executable(path: Path) -> Path:
    absolute = path.absolute()
    if not absolute.is_file():
        raise FileNotFoundError(absolute)
    if not os.access(absolute, os.X_OK):
        raise PermissionError(f"renderer Python is not executable: {absolute}")
    return absolute


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    queue_path = args.review_queue.resolve()
    output_root = args.output_root.resolve()
    renderer_python = _executable(args.renderer_python)
    urdf = args.urdf.resolve()
    if not queue_path.is_file():
        raise FileNotFoundError(queue_path)
    if not urdf.is_file():
        raise FileNotFoundError(urdf)
    summary = run_review(
        queue_path=queue_path,
        output_root=output_root,
        renderer_python=renderer_python,
        urdf=urdf,
        limit=args.limit,
        sampling=args.sampling,
        seed=args.seed,
        workers=args.workers,
        width=args.width,
        height=args.height,
        resume=args.resume,
        retry_failed=args.retry_failed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["counts"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
