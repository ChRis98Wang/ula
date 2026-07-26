#!/usr/bin/env python3
"""Build a conservative BEAT2 conversational-interaction inventory.

The transcript is retained only as speech context.  It is never promoted to a
frame-level motion label: every record uses the single broad interaction label
``co_speech_conversational_gesture`` and remains denied for training pending
motion/text review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import string
import struct
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path(
    "/home/gez/nas/cloud/gez/human_motion/raw/BEAT2/beat_chinese_v2.0.0"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "deliverables/interactive_human_motion_v1/catalog"
)
OUTPUT_STEM = "beat2_interaction_full_inventory_v1"
EXPECTED_CLIP_COUNT = 310
FPS = 30.0
TARGET_WINDOW_SEC = 6.0
INTERACTION_LABEL = "co_speech_conversational_gesture"
TRANSCRIPT_ROLE = "clip_level_speech_context_only_not_action_label"
WINDOW_TRANSCRIPT_ROLE = "time_aligned_speech_context_only_not_action_label"
MAX_MOTION_AUDIO_DURATION_DIFF_SEC = 0.3
TRANSCRIPT_PUNCTUATION = string.punctuation + "，。！？；：、“”‘’（）《》【】…—"
TRANSCRIPT_NORMALIZATION_PATTERN = re.compile(
    r"[\s" + re.escape(TRANSCRIPT_PUNCTUATION) + r"]"
)

# SMPL-X pose order: root, 21 body joints, jaw/eyes, then 30 finger joints.
# Root translation and all face/finger channels are intentionally excluded.
SMPLX_BODY_JOINT_NAMES = {
    3: "spine1",
    6: "spine2",
    9: "spine3",
    12: "neck",
    13: "left_collar",
    14: "right_collar",
    15: "head",
    16: "left_shoulder",
    17: "right_shoulder",
    18: "left_elbow",
    19: "right_elbow",
    20: "left_wrist",
    21: "right_wrist",
}
UPPER_BODY_JOINT_INDICES = tuple(SMPLX_BODY_JOINT_NAMES)
HEAD_NECK_JOINT_INDICES = (12, 15)
IGNORED_POSE_JOINT_INDICES = {
    "root_orientation": [0],
    "lower_body": [1, 2, 4, 5, 7, 8, 10, 11],
    "jaw_and_eyes": [22, 23, 24],
    "left_and_right_fingers": list(range(25, 55)),
}

SPEAKER_PATTERN = re.compile(
    r"^(?P<speaker_id>\d+)_(?P<speaker_name>[^_]+)_(?P<session>\d+)_"
)
TEXTGRID_INTERVAL_PATTERN = re.compile(
    r"intervals\s*\[\d+\]\s*:\s*"
    r"xmin\s*=\s*([-+0-9.eE]+)\s*"
    r"xmax\s*=\s*([-+0-9.eE]+)\s*"
    r'text\s*=\s*"((?:""|[^"])*)"',
    re.MULTILINE,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window-sec", type=float, default=TARGET_WINDOW_SEC)
    parser.add_argument("--stride-sec", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--expected-clips",
        type=int,
        default=EXPECTED_CLIP_COUNT,
        help="Set to zero to allow any non-empty clip count.",
    )
    return parser.parse_args()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def normalize_text(value: str) -> str:
    return " ".join(value.strip().split())


def normalize_transcript_for_alignment(value: str) -> str:
    return TRANSCRIPT_NORMALIZATION_PATTERN.sub("", value).lower()


def relative_path(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def load_official_splits(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing official split CSV: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id", "type"]:
            raise ValueError(
                f"Unexpected split columns {reader.fieldnames}; expected ['id', 'type']"
            )
        rows = list(reader)
    splits: dict[str, str] = {}
    allowed = {"train", "val", "test", "additional"}
    for row_number, row in enumerate(rows, 2):
        clip_id = (row.get("id") or "").strip()
        split = (row.get("type") or "").strip()
        if not clip_id or split not in allowed:
            raise ValueError(f"Invalid official split row {row_number}: {row}")
        if clip_id in splits:
            raise ValueError(f"Duplicate clip_id in official split: {clip_id}")
        splits[clip_id] = split
    return splits


def parse_speaker(clip_id: str) -> dict[str, str]:
    match = SPEAKER_PATTERN.match(clip_id)
    if not match:
        raise ValueError(f"Cannot parse BEAT2 speaker from clip_id: {clip_id}")
    values = match.groupdict()
    return {
        "speaker_id": values["speaker_id"],
        "speaker_name": values["speaker_name"],
        "speaker_key": f"{values['speaker_id']}_{values['speaker_name']}",
        "session_id": values["session"],
        "speaker_source": "official_beat2_filename",
    }


def read_wav_metadata(path: Path) -> dict[str, int | float | str]:
    """Read PCM/IEEE-float RIFF metadata without decoding the audio payload."""
    with path.open("rb") as handle:
        header = handle.read(12)
        if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
            raise ValueError(f"Not a RIFF/WAVE file: {path}")
        format_fields = None
        data_size = None
        while True:
            chunk_header = handle.read(8)
            if not chunk_header:
                break
            if len(chunk_header) != 8:
                raise ValueError(f"Truncated WAV chunk header: {path}")
            chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
            if chunk_id == b"fmt ":
                payload = handle.read(chunk_size)
                if len(payload) != chunk_size or chunk_size < 16:
                    raise ValueError(f"Truncated WAV fmt chunk: {path}")
                format_fields = struct.unpack("<HHIIHH", payload[:16])
            elif chunk_id == b"data":
                data_size = chunk_size
                handle.seek(chunk_size, os.SEEK_CUR)
            else:
                handle.seek(chunk_size, os.SEEK_CUR)
            if chunk_size % 2:
                handle.seek(1, os.SEEK_CUR)

    if format_fields is None or data_size is None:
        raise ValueError(f"WAV is missing fmt or data chunk: {path}")
    audio_format, channels, sample_rate, _byte_rate, block_align, bits = format_fields
    dtype_by_format = {
        (1, 8): "uint8",
        (1, 16): "int16",
        (1, 24): "int24",
        (1, 32): "int32",
        (3, 32): "float32",
        (3, 64): "float64",
    }
    dtype = dtype_by_format.get((audio_format, bits))
    if dtype is None:
        raise ValueError(
            f"Unsupported WAV format tag={audio_format}, bits={bits}: {path}"
        )
    expected_block_align = channels * ((bits + 7) // 8)
    if channels < 1 or sample_rate < 1 or block_align != expected_block_align:
        raise ValueError(f"Invalid WAV channel/rate/block alignment: {path}")
    if data_size % block_align:
        raise ValueError(f"WAV data is not frame aligned: {path}")
    frame_count = data_size // block_align
    return {
        "sample_rate": int(sample_rate),
        "channels": int(channels),
        "dtype": dtype,
        "frame_count": int(frame_count),
        "duration_sec": round(float(frame_count) / float(sample_rate), 6),
        "format": "pcm" if audio_format == 1 else "ieee_float",
    }


def parse_textgrid_intervals(path: Path) -> list[tuple[float, float, str]]:
    payload = path.read_text(encoding="utf-8", errors="strict")
    if 'File type = "ooTextFile"' not in payload or 'name = "words"' not in payload:
        raise ValueError(f"Unsupported or missing words tier in TextGrid: {path}")
    intervals = []
    previous_end = -np.inf
    for match in TEXTGRID_INTERVAL_PATTERN.finditer(payload):
        start_sec = float(match.group(1))
        end_sec = float(match.group(2))
        text = match.group(3).replace('""', '"')
        if not np.isfinite(start_sec) or not np.isfinite(end_sec) or end_sec < start_sec:
            raise ValueError(f"Invalid TextGrid interval in {path}: {match.group(0)!r}")
        if start_sec + 1e-8 < previous_end:
            raise ValueError(f"Overlapping or unsorted TextGrid intervals: {path}")
        intervals.append((start_sec, end_sec, text))
        previous_end = end_sec
    if not intervals:
        raise ValueError(f"No intervals parsed from TextGrid: {path}")
    return intervals


def aligned_transcript_context(
    intervals: list[tuple[float, float, str]], start_sec: float, end_sec: float
) -> str:
    tokens = [
        text.strip()
        for interval_start, interval_end, text in intervals
        if text.strip() and interval_end > start_sec and interval_start < end_sec
    ]
    return "".join(tokens)


def axis_angle_to_quaternion(axis_angle: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(axis_angle, axis=-1)
    half_angle = angle * 0.5
    scale = np.divide(
        np.sin(half_angle),
        angle,
        out=np.full_like(angle, 0.5),
        where=angle > 1e-8,
    )
    return np.concatenate(
        [np.cos(half_angle)[..., None], axis_angle * scale[..., None]], axis=-1
    )


def joint_angular_speed(poses: np.ndarray, joint_indices: tuple[int, ...]) -> np.ndarray:
    joints = poses.reshape(poses.shape[0], 55, 3)[:, joint_indices]
    quaternion = axis_angle_to_quaternion(joints.astype(np.float64, copy=False))
    dot = np.abs(np.sum(quaternion[1:] * quaternion[:-1], axis=-1))
    angular_distance = 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))
    return angular_distance * FPS


def _window_metrics(
    upper_joint_speed: np.ndarray,
    head_neck_joint_speed: np.ndarray,
    start_frame: int,
    frame_count: int,
) -> dict[str, float | int]:
    # A T-frame window contains T-1 frame-to-frame velocity observations.
    stop_velocity = max(start_frame + 1, start_frame + frame_count - 1)
    upper = upper_joint_speed[start_frame:stop_velocity]
    head_neck = head_neck_joint_speed[start_frame:stop_velocity]
    upper_per_frame = np.mean(upper, axis=1)
    head_neck_per_frame = np.mean(head_neck, axis=1)
    combined = 0.8 * upper_per_frame + 0.2 * head_neck_per_frame
    return {
        "start_frame": int(start_frame),
        "frame_count": int(frame_count),
        "upper_body_mean_rad_s": float(np.mean(upper_per_frame)),
        "upper_body_p95_rad_s": float(np.percentile(upper_per_frame, 95)),
        "head_neck_mean_rad_s": float(np.mean(head_neck_per_frame)),
        "head_neck_p95_rad_s": float(np.percentile(head_neck_per_frame, 95)),
        "interaction_energy_mean_rad_s": float(np.mean(combined)),
        "interaction_energy_p95_rad_s": float(np.percentile(combined, 95)),
        "active_frame_fraction": float(np.mean(combined >= 0.02)),
    }


def select_interaction_window(
    poses: np.ndarray,
    window_sec: float = TARGET_WINDOW_SEC,
    stride_sec: float = 0.5,
    min_nonstatic_energy_rad_s: float = 0.02,
    max_p95_energy_rad_s: float = 4.0,
    speech_intervals: list[tuple[float, float, str]] | None = None,
) -> dict[str, float | int | str]:
    if poses.ndim != 2 or poses.shape[1] != 165 or poses.shape[0] < 2:
        raise ValueError(f"Expected poses[T>=2,165], got {poses.shape}")
    if not np.isfinite(poses).all():
        raise ValueError("poses contain non-finite values")
    if window_sec <= 0 or stride_sec <= 0:
        raise ValueError("window_sec and stride_sec must be positive")

    target_frames = max(2, int(round(window_sec * FPS)))
    frame_count = min(int(poses.shape[0]), target_frames)
    stride_frames = max(1, int(round(stride_sec * FPS)))
    last_start = int(poses.shape[0]) - frame_count
    starts = list(range(0, last_start + 1, stride_frames))
    if starts[-1] != last_start:
        starts.append(last_start)

    upper_speed = joint_angular_speed(poses, UPPER_BODY_JOINT_INDICES)
    head_neck_speed = joint_angular_speed(poses, HEAD_NECK_JOINT_INDICES)
    candidates = [
        _window_metrics(upper_speed, head_neck_speed, start, frame_count)
        for start in starts
    ]
    if speech_intervals is not None:
        for item in candidates:
            start_sec = float(item["start_frame"]) / FPS
            end_sec = float(item["start_frame"] + frame_count) / FPS
            item["aligned_speech_unit_count"] = sum(
                1
                for interval_start, interval_end, text in speech_intervals
                if text.strip()
                and interval_end > start_sec
                and interval_start < end_sec
            )
        candidate_pool = [
            item for item in candidates if item["aligned_speech_unit_count"] > 0
        ]
    else:
        for item in candidates:
            item["aligned_speech_unit_count"] = None
        candidate_pool = candidates

    bounded = [
        item
        for item in candidate_pool
        if item["interaction_energy_mean_rad_s"] >= min_nonstatic_energy_rad_s
        and item["interaction_energy_p95_rad_s"] <= max_p95_energy_rad_s
    ]
    if bounded:
        target_energy = float(
            np.quantile(
                [item["interaction_energy_mean_rad_s"] for item in bounded], 0.25
            )
        )
        chosen = min(
            bounded,
            key=lambda item: (
                abs(item["interaction_energy_mean_rad_s"] - target_energy),
                item["interaction_energy_p95_rad_s"],
                item["start_frame"],
            ),
        )
        status = (
            "selected_nonstatic_low_dynamic_with_aligned_speech"
            if speech_intervals is not None
            else "selected_nonstatic_low_dynamic"
        )
    else:
        if not candidate_pool:
            candidate_pool = candidates
        nonstatic = [
            item
            for item in candidate_pool
            if item["interaction_energy_mean_rad_s"] >= min_nonstatic_energy_rad_s
        ]
        if speech_intervals is not None and not any(
            item["aligned_speech_unit_count"] for item in candidates
        ):
            chosen = min(
                nonstatic or candidates,
                key=lambda item: (
                    item["interaction_energy_p95_rad_s"],
                    item["interaction_energy_mean_rad_s"],
                    item["start_frame"],
                ),
            )
            status = "fallback_no_aligned_speech_window"
        elif nonstatic:
            chosen = min(
                nonstatic,
                key=lambda item: (
                    item["interaction_energy_p95_rad_s"],
                    item["interaction_energy_mean_rad_s"],
                    item["start_frame"],
                ),
            )
            status = "fallback_no_non_high_dynamic_window"
        else:
            chosen = max(
                candidates,
                key=lambda item: (
                    item["interaction_energy_mean_rad_s"],
                    -item["start_frame"],
                ),
            )
            status = "fallback_no_nonstatic_window"

    result = dict(chosen)
    result.update(
        {
            "end_frame_exclusive": int(chosen["start_frame"] + frame_count),
            "start_sec": round(float(chosen["start_frame"]) / FPS, 6),
            "end_sec": round(
                float(chosen["start_frame"] + frame_count) / FPS, 6
            ),
            "duration_sec": round(float(frame_count) / FPS, 6),
            "selection_status": status,
            "candidate_count": len(candidates),
            "target_quantile": 0.25,
            "min_nonstatic_energy_rad_s": min_nonstatic_energy_rad_s,
            "max_p95_energy_rad_s": max_p95_energy_rad_s,
        }
    )
    for key, value in list(result.items()):
        if isinstance(value, float):
            result[key] = round(value, 8)
    return result


def validate_motion_npz(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"poses", "trans", "mocap_frame_rate"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"Missing NPZ fields {missing}: {path}")
        poses = np.asarray(archive["poses"])
        trans = np.asarray(archive["trans"])
        frame_rate = np.asarray(archive["mocap_frame_rate"])
    if poses.ndim != 2 or poses.shape[1] != 165 or poses.shape[0] < 2:
        raise ValueError(f"Expected poses[T>=2,165], got {poses.shape}: {path}")
    if trans.shape != (poses.shape[0], 3):
        raise ValueError(
            f"Expected trans[{poses.shape[0]},3], got {trans.shape}: {path}"
        )
    if frame_rate.size != 1 or float(frame_rate.reshape(-1)[0]) != FPS:
        raise ValueError(f"Expected mocap_frame_rate=30, got {frame_rate}: {path}")
    if not np.isfinite(poses).all() or not np.isfinite(trans).all():
        raise ValueError(f"Non-finite poses or trans: {path}")
    return poses, trans, FPS


def _build_record(arguments: tuple) -> dict:
    (
        root,
        motion_path,
        transcript_path,
        textgrid_path,
        audio_path,
        official_split,
        window_sec,
        stride_sec,
    ) = arguments
    root = Path(root)
    motion_path = Path(motion_path)
    transcript_path = Path(transcript_path)
    textgrid_path = Path(textgrid_path) if textgrid_path else None
    audio_path = Path(audio_path)
    clip_id = motion_path.stem

    if not transcript_path.is_file():
        raise FileNotFoundError(f"Missing transcript for {clip_id}: {transcript_path}")
    transcript = normalize_text(transcript_path.read_text(encoding="utf-8"))
    if not transcript:
        raise ValueError(f"Empty transcript for {clip_id}: {transcript_path}")
    if not audio_path.is_file():
        raise FileNotFoundError(f"Missing audio for {clip_id}: {audio_path}")
    audio = read_wav_metadata(audio_path)

    intervals = None
    textgrid_relpath = None
    textgrid_transcript_matches = None
    if textgrid_path and textgrid_path.is_file():
        intervals = parse_textgrid_intervals(textgrid_path)
        textgrid_relpath = relative_path(textgrid_path, root)
        textgrid_transcript_matches = normalize_transcript_for_alignment(
            "".join(text for _start, _end, text in intervals)
        ) == normalize_transcript_for_alignment(transcript)
    poses, trans, fps = validate_motion_npz(motion_path)
    window = select_interaction_window(
        poses, window_sec, stride_sec, speech_intervals=intervals
    )

    issues = []
    aligned_context = ""
    if intervals is not None:
        aligned_context = aligned_transcript_context(
            intervals, window["start_sec"], window["end_sec"]
        )
        if not aligned_context:
            issues.append("selected_window_has_no_aligned_speech")
        if not textgrid_transcript_matches:
            issues.append("textgrid_transcript_mismatch")
    else:
        issues.append("missing_textgrid_alignment")
    if official_split is None:
        issues.append("missing_official_split")
    if not window["selection_status"].startswith("selected_"):
        issues.append(window["selection_status"])
    motion_audio_duration_diff = abs(
        float(poses.shape[0]) / fps - float(audio["duration_sec"])
    )
    if motion_audio_duration_diff > MAX_MOTION_AUDIO_DURATION_DIFF_SEC:
        issues.append("motion_audio_duration_mismatch_gt_0_3s")

    speaker = parse_speaker(clip_id)
    return {
        "schema_version": "1.0.0",
        "dataset": "BEAT2",
        "dataset_subset": "beat_chinese_v2.0.0",
        "clip_id": clip_id,
        **speaker,
        "official_split": official_split,
        "interaction_label": INTERACTION_LABEL,
        "label_source": "dataset_scope_only_not_clip_action_semantics",
        "transcript": transcript,
        "transcript_role": TRANSCRIPT_ROLE,
        "window_transcript_context": aligned_context,
        "window_transcript_role": WINDOW_TRANSCRIPT_ROLE,
        "textgrid_transcript_matches": textgrid_transcript_matches,
        "motion_relpath": relative_path(motion_path, root),
        "transcript_relpath": relative_path(transcript_path, root),
        "textgrid_relpath": textgrid_relpath,
        "audio_relpath": relative_path(audio_path, root),
        "audio_sample_rate": audio["sample_rate"],
        "audio_channels": audio["channels"],
        "audio_dtype": audio["dtype"],
        "audio_format": audio["format"],
        "audio_frame_count": audio["frame_count"],
        "audio_duration_sec": audio["duration_sec"],
        "motion_audio_duration_abs_diff_sec": round(
            motion_audio_duration_diff, 6
        ),
        "source_frame_count": int(poses.shape[0]),
        "pose_feature_dim": int(poses.shape[1]),
        "trans_feature_dim": int(trans.shape[1]),
        "fps": fps,
        "source_duration_sec": round(float(poses.shape[0]) / fps, 6),
        "window": window,
        "energy_joint_indices": list(UPPER_BODY_JOINT_INDICES),
        "energy_joint_names": [
            SMPLX_BODY_JOINT_NAMES[index] for index in UPPER_BODY_JOINT_INDICES
        ],
        "head_neck_joint_indices": list(HEAD_NECK_JOINT_INDICES),
        "ignored_pose_joint_indices": IGNORED_POSE_JOINT_INDICES,
        "issues": issues,
        "review_state": "machine_windowed_pending_interaction_review",
        "manual_review_required": True,
        "accepted_for_training": False,
    }


def build_inventory(
    root: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    window_sec: float = TARGET_WINDOW_SEC,
    stride_sec: float = 0.5,
    expected_clip_count: int | None = EXPECTED_CLIP_COUNT,
    workers: int = 1,
) -> dict:
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    motion_root = root / "smplxflame_30"
    transcript_root = root / "text"
    textgrid_root = root / "textgrid"
    audio_root = root / "wave16k"
    split_path = root / "train_test_split.csv"
    motion_paths = sorted(motion_root.glob("*.npz"))
    if not motion_paths:
        raise FileNotFoundError(f"No BEAT2 NPZ files under {motion_root}")
    if expected_clip_count and len(motion_paths) != expected_clip_count:
        raise ValueError(
            f"Expected {expected_clip_count} BEAT2 NPZ files, found {len(motion_paths)}"
        )
    if len({path.stem for path in motion_paths}) != len(motion_paths):
        raise ValueError("Duplicate BEAT2 motion stems")
    if workers < 1:
        raise ValueError("workers must be at least 1")

    official_splits = load_official_splits(split_path)
    motion_stems = {path.stem for path in motion_paths}
    transcript_stems = {path.stem for path in transcript_root.glob("*.txt")}
    textgrid_by_stem = {
        path.stem: path for path in sorted(textgrid_root.glob("*.TextGrid"))
    }
    audio_stems = {path.stem for path in audio_root.glob("*.wav")}
    missing_transcripts = sorted(motion_stems - transcript_stems)
    if missing_transcripts:
        raise ValueError(
            f"Missing transcript stems for {len(missing_transcripts)} motions: "
            f"{missing_transcripts[:10]}"
        )
    missing_audio = sorted(motion_stems - audio_stems)
    if missing_audio:
        raise ValueError(
            f"Missing audio stems for {len(missing_audio)} motions: {missing_audio[:10]}"
        )

    arguments = [
        (
            root,
            motion_path,
            transcript_root / f"{motion_path.stem}.txt",
            textgrid_by_stem.get(motion_path.stem),
            audio_root / f"{motion_path.stem}.wav",
            official_splits.get(motion_path.stem),
            window_sec,
            stride_sec,
        )
        for motion_path in motion_paths
    ]
    if workers == 1:
        records = [_build_record(item) for item in arguments]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            records = list(executor.map(_build_record, arguments))
    records.sort(key=lambda record: record["clip_id"])

    jsonl_path = output_dir / f"{OUTPUT_STEM}.jsonl"
    csv_path = output_dir / f"{OUTPUT_STEM}.csv"
    summary_path = output_dir / f"{OUTPUT_STEM}.summary.json"
    jsonl_bytes = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for record in records
    ).encode("utf-8")
    fields = [
        "clip_id",
        "speaker_id",
        "speaker_name",
        "speaker_key",
        "session_id",
        "official_split",
        "interaction_label",
        "label_source",
        "transcript_role",
        "window_transcript_context",
        "window_transcript_role",
        "textgrid_transcript_matches",
        "motion_relpath",
        "transcript_relpath",
        "textgrid_relpath",
        "audio_relpath",
        "audio_sample_rate",
        "audio_channels",
        "audio_dtype",
        "audio_format",
        "audio_duration_sec",
        "motion_audio_duration_abs_diff_sec",
        "source_frame_count",
        "fps",
        "source_duration_sec",
        "window_start_frame",
        "window_end_frame_exclusive",
        "window_duration_sec",
        "selection_status",
        "upper_body_mean_rad_s",
        "head_neck_mean_rad_s",
        "interaction_energy_mean_rad_s",
        "interaction_energy_p95_rad_s",
        "issues",
        "review_state",
        "manual_review_required",
        "accepted_for_training",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for record in records:
        window = record["window"]
        writer.writerow(
            {
                **{key: record.get(key) for key in fields},
                "window_start_frame": window["start_frame"],
                "window_end_frame_exclusive": window["end_frame_exclusive"],
                "window_duration_sec": window["duration_sec"],
                "selection_status": window["selection_status"],
                "upper_body_mean_rad_s": window["upper_body_mean_rad_s"],
                "head_neck_mean_rad_s": window["head_neck_mean_rad_s"],
                "interaction_energy_mean_rad_s": window[
                    "interaction_energy_mean_rad_s"
                ],
                "interaction_energy_p95_rad_s": window[
                    "interaction_energy_p95_rad_s"
                ],
                "issues": "|".join(record["issues"]),
            }
        )
    csv_bytes = buffer.getvalue().encode("utf-8")

    issue_counts = Counter(issue for record in records for issue in record["issues"])
    selection_counts = Counter(
        record["window"]["selection_status"] for record in records
    )
    split_counts = Counter(record["official_split"] or "unassigned" for record in records)
    total_frames = sum(record["source_frame_count"] for record in records)
    selected_frames = sum(record["window"]["frame_count"] for record in records)
    all_textgrid_mismatch_count = 0
    comparable_textgrid_count = 0
    for stem, path in textgrid_by_stem.items():
        transcript_path = transcript_root / f"{stem}.txt"
        if not transcript_path.is_file():
            continue
        intervals = parse_textgrid_intervals(path)
        grid_text = "".join(text for _start, _end, text in intervals)
        source_text = transcript_path.read_text(encoding="utf-8")
        comparable_textgrid_count += 1
        if normalize_transcript_for_alignment(
            grid_text
        ) != normalize_transcript_for_alignment(source_text):
            all_textgrid_mismatch_count += 1
    audio_format_counts = Counter(
        f"{record['audio_sample_rate']}Hz/{record['audio_channels']}ch/"
        f"{record['audio_dtype']}"
        for record in records
    )
    summary = {
        "schema_version": "1.0.0",
        "dataset": "BEAT2",
        "dataset_subset": "beat_chinese_v2.0.0",
        "record_count": len(records),
        "source_frame_count": total_frames,
        "source_duration_sec": round(total_frames / FPS, 6),
        "selected_window_frame_count": selected_frames,
        "selected_window_duration_sec": round(selected_frames / FPS, 6),
        "speaker_count": len({record["speaker_key"] for record in records}),
        "speakers": sorted({record["speaker_key"] for record in records}),
        "counts_by_official_split": dict(sorted(split_counts.items())),
        "counts_by_selection_status": dict(sorted(selection_counts.items())),
        "counts_by_issue": dict(sorted(issue_counts.items())),
        "motion_stem_count": len(motion_stems),
        "transcript_stem_count": len(transcript_stems),
        "audio_stem_count": len(audio_stems),
        "textgrid_stem_count": len(textgrid_by_stem),
        "official_split_stem_count": len(official_splits),
        "missing_textgrid_for_motion_count": len(motion_stems - set(textgrid_by_stem)),
        "missing_official_split_for_motion_count": len(
            motion_stems - set(official_splits)
        ),
        "orphan_transcript_stem_count": len(transcript_stems - motion_stems),
        "orphan_audio_stem_count": len(audio_stems - motion_stems),
        "orphan_textgrid_stem_count": len(set(textgrid_by_stem) - motion_stems),
        "orphan_official_split_stem_count": len(set(official_splits) - motion_stems),
        "interaction_label_policy": INTERACTION_LABEL,
        "counts_by_audio_format": dict(sorted(audio_format_counts.items())),
        "motion_audio_duration_mismatch_threshold_sec": (
            MAX_MOTION_AUDIO_DURATION_DIFF_SEC
        ),
        "textgrid_transcript_comparable_all_source_count": (
            comparable_textgrid_count
        ),
        "textgrid_transcript_mismatch_all_source_count": (
            all_textgrid_mismatch_count
        ),
        "textgrid_transcript_mismatch_motion_count": issue_counts.get(
            "textgrid_transcript_mismatch", 0
        ),
        "textgrid_transcript_normalization_policy": (
            "lowercase and remove whitespace, ASCII punctuation and common Chinese "
            "punctuation"
        ),
        "transcript_policy": (
            "speech context only; never treated as a precise motion/action label"
        ),
        "channel_policy": (
            "window energy uses torso, neck, head, shoulders, elbows and wrists; "
            "face, eyes, jaw and fingers are ignored"
        ),
        "window_policy": {
            "target_duration_sec": window_sec,
            "stride_sec": stride_sec,
            "relative_energy_target_quantile": 0.25,
            "min_nonstatic_energy_rad_s": 0.02,
            "max_p95_energy_rad_s": 4.0,
        },
        "conditioning_policy": "deny_until_interaction_motion_review",
        "accepted_for_training_count": 0,
        "output_sha256": {
            "jsonl": hashlib.sha256(jsonl_bytes).hexdigest(),
            "csv": hashlib.sha256(csv_bytes).hexdigest(),
        },
    }
    summary_bytes = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    atomic_write(jsonl_path, jsonl_bytes)
    atomic_write(csv_path, csv_bytes)
    atomic_write(summary_path, summary_bytes)
    return summary


def main() -> None:
    args = parse_args()
    summary = build_inventory(
        args.root,
        args.output_dir,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        expected_clip_count=args.expected_clips or None,
        workers=args.workers,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
