#!/usr/bin/env python3
"""Build boundary-validated, non-overlapping BEAT2 six-second windows.

Only sources with a TextGrid are windowed. Speech is retained as aligned
context and is explicitly not used to infer behavior or emotion labels. Every
record remains denied for training until retarget QC and semantic review pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

try:
    from tools.human_motion_collection.build_beat2_interaction_inventory import (
        DEFAULT_ROOT,
        FPS,
        INTERACTION_LABEL,
        TRANSCRIPT_ROLE,
        WINDOW_TRANSCRIPT_ROLE,
        _window_metrics,
        aligned_transcript_context,
        atomic_write,
        joint_angular_speed,
        parse_speaker,
        parse_textgrid_intervals,
        read_wav_metadata,
        validate_motion_npz,
        UPPER_BODY_JOINT_INDICES,
        HEAD_NECK_JOINT_INDICES,
    )
except ModuleNotFoundError:  # pragma: no cover - direct invocation by path
    from build_beat2_interaction_inventory import (
        DEFAULT_ROOT,
        FPS,
        INTERACTION_LABEL,
        TRANSCRIPT_ROLE,
        WINDOW_TRANSCRIPT_ROLE,
        _window_metrics,
        aligned_transcript_context,
        atomic_write,
        joint_angular_speed,
        parse_speaker,
        parse_textgrid_intervals,
        read_wav_metadata,
        validate_motion_npz,
        UPPER_BODY_JOINT_INDICES,
        HEAD_NECK_JOINT_INDICES,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_INVENTORY = (
    PROJECT_ROOT
    / "deliverables/interactive_human_motion_v1/catalog/beat2_interaction_full_inventory_v1.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "deliverables/interactive_human_motion_v1/catalog"
)
OUTPUT_STEM = "beat2_interaction_full_6s_windows_v1"
SCHEMA_VERSION = "1.0.0"
TARGET_WINDOW_SEC = 6.0
EXPECTED_SOURCE_CLIP_COUNT = 310
EXPECTED_ALIGNED_SOURCE_CLIP_COUNT = 280
EXPECTED_WINDOW_COUNT = 4570
SELECTION_STATUS = "full_nonoverlap_boundary_validated"
SEMANTIC_STATUS = "unlabeled_no_inference_from_speech"
TIME_TOLERANCE_SEC = 1e-6


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-inventory", type=Path, default=DEFAULT_SOURCE_INVENTORY)
    parser.add_argument("--beat2-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window-sec", type=float, default=TARGET_WINDOW_SEC)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--expected-source-clips",
        type=int,
        default=EXPECTED_SOURCE_CLIP_COUNT,
        help="Set to zero to allow any non-empty source inventory size.",
    )
    parser.add_argument(
        "--expected-aligned-source-clips",
        type=int,
        default=EXPECTED_ALIGNED_SOURCE_CLIP_COUNT,
        help="Set to zero to allow any non-empty TextGrid-aligned source count.",
    )
    parser.add_argument(
        "--expected-windows",
        type=int,
        default=EXPECTED_WINDOW_COUNT,
        help="Set to zero to allow any non-empty output window count.",
    )
    return parser.parse_args(argv)


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


def record_sha256(record: dict[str, Any]) -> str:
    return hashlib.sha256(stable_json(record).encode("utf-8")).hexdigest()


def read_source_inventory(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            clip_id = str(record.get("clip_id", "")).strip()
            if not clip_id:
                raise ValueError(f"Missing clip_id at {path}:{line_number}")
            if clip_id in seen:
                raise ValueError(f"Duplicate source clip_id: {clip_id}")
            seen.add(clip_id)
            records.append(record)
    if not records:
        raise ValueError(f"Source inventory is empty: {path}")
    return records


def resolve_source_path(root: Path, relpath: object, field: str, clip_id: str) -> Path:
    value = str(relpath or "").strip()
    if not value:
        raise ValueError(f"Source {clip_id} has no {field}")
    root = root.resolve()
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Source {clip_id} {field} escapes BEAT2 root: {value}") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def audio_sample_for_frame(frame: int, sample_rate: int, fps: float) -> int:
    return int(round(float(frame) * float(sample_rate) / float(fps)))


def overlapping_textgrid_units(
    intervals: list[tuple[float, float, str]], start_sec: float, end_sec: float
) -> list[dict[str, Any]]:
    return [
        {
            "source_start_sec": round(interval_start, 6),
            "source_end_sec": round(interval_end, 6),
            "window_start_sec": round(max(interval_start, start_sec) - start_sec, 6),
            "window_end_sec": round(min(interval_end, end_sec) - start_sec, 6),
            "text": text.strip(),
        }
        for interval_start, interval_end, text in intervals
        if text.strip() and interval_end > start_sec and interval_start < end_sec
    ]


def validate_source_metadata(
    record: dict[str, Any],
    source_clip_id: str,
    poses_frame_count: int,
    fps: float,
    audio: dict[str, int | float | str],
) -> None:
    recorded_frames = record.get("source_frame_count")
    if recorded_frames is not None and recorded_frames != poses_frame_count:
        raise ValueError(
            f"Source {source_clip_id} frame count changed: "
            f"inventory={recorded_frames}, file={poses_frame_count}"
        )
    recorded_fps = record.get("fps")
    if recorded_fps is not None and float(recorded_fps) != fps:
        raise ValueError(
            f"Source {source_clip_id} FPS changed: inventory={recorded_fps}, file={fps}"
        )
    for key in ("audio_sample_rate", "audio_frame_count"):
        recorded_value = record.get(key)
        actual_value = audio[key.replace("audio_", "")]
        if recorded_value is not None and recorded_value != actual_value:
            raise ValueError(
                f"Source {source_clip_id} {key} changed: "
                f"inventory={recorded_value}, file={actual_value}"
            )


def _build_source_windows(arguments: tuple[Any, ...]) -> dict[str, Any]:
    record, beat2_root, window_sec = arguments
    beat2_root = Path(beat2_root)
    source_clip_id = str(record["clip_id"])
    speaker = parse_speaker(source_clip_id)
    for key in ("speaker_id", "speaker_name", "speaker_key", "session_id"):
        recorded = record.get(key)
        if recorded is not None and recorded != speaker[key]:
            raise ValueError(
                f"Source {source_clip_id} has inconsistent {key}: {recorded!r}"
            )

    motion_path = resolve_source_path(
        beat2_root, record.get("motion_relpath"), "motion_relpath", source_clip_id
    )
    audio_path = resolve_source_path(
        beat2_root, record.get("audio_relpath"), "audio_relpath", source_clip_id
    )
    textgrid_path = resolve_source_path(
        beat2_root, record.get("textgrid_relpath"), "textgrid_relpath", source_clip_id
    )
    poses, _trans, fps = validate_motion_npz(motion_path)
    audio = read_wav_metadata(audio_path)
    intervals = parse_textgrid_intervals(textgrid_path)
    validate_source_metadata(record, source_clip_id, len(poses), fps, audio)

    window_frames_float = window_sec * fps
    window_frames = int(round(window_frames_float))
    if window_frames < 2 or not math.isclose(
        window_frames_float, window_frames, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError(
            f"window_sec={window_sec} does not map to an integer frame count at {fps} FPS"
        )
    textgrid_start_sec = min(start for start, _end, _text in intervals)
    textgrid_end_sec = max(end for _start, end, _text in intervals)
    sample_rate = int(audio["sample_rate"])
    audio_frame_count = int(audio["frame_count"])
    upper_speed = joint_angular_speed(poses, UPPER_BODY_JOINT_INDICES)
    head_neck_speed = joint_angular_speed(poses, HEAD_NECK_JOINT_INDICES)

    full_motion_window_count = len(poses) // window_frames
    windows: list[dict[str, Any]] = []
    rejected_boundary_window_count = 0
    for source_window_index in range(full_motion_window_count):
        start_frame = source_window_index * window_frames
        end_frame = start_frame + window_frames
        start_sec = float(start_frame) / fps
        end_sec = float(end_frame) / fps
        audio_start_sample = audio_sample_for_frame(start_frame, sample_rate, fps)
        audio_end_sample = audio_sample_for_frame(end_frame, sample_rate, fps)
        motion_bounds_valid = 0 <= start_frame < end_frame <= len(poses)
        audio_bounds_valid = (
            0 <= audio_start_sample < audio_end_sample <= audio_frame_count
        )
        textgrid_bounds_valid = (
            start_sec + TIME_TOLERANCE_SEC >= textgrid_start_sec
            and end_sec <= textgrid_end_sec + TIME_TOLERANCE_SEC
        )
        if not (motion_bounds_valid and audio_bounds_valid and textgrid_bounds_valid):
            rejected_boundary_window_count += 1
            continue

        window_id = f"{source_clip_id}_f{start_frame:06d}-{end_frame:06d}"
        units = overlapping_textgrid_units(intervals, start_sec, end_sec)
        context = aligned_transcript_context(intervals, start_sec, end_sec)
        metrics = _window_metrics(
            upper_speed, head_neck_speed, start_frame, window_frames
        )
        metrics = {
            key: round(value, 8) if isinstance(value, float) else value
            for key, value in metrics.items()
        }
        source_issues = sorted(set(record.get("issues") or []))
        windows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset": "BEAT2",
                "dataset_subset": "beat_chinese_v2.0.0",
                "clip_id": window_id,
                "window_id": window_id,
                "task_id": window_id,
                "source_clip_id": source_clip_id,
                "source_group_id": source_clip_id,
                "split_group_id": source_clip_id,
                "source_group_policy": "all_windows_from_source_clip_share_split",
                **speaker,
                "official_split": record.get("official_split"),
                "interaction_label": INTERACTION_LABEL,
                "label_source": "dataset_scope_only_not_window_action_semantics",
                "behavior_id": None,
                "emotion_id": None,
                "semantic_label_status": SEMANTIC_STATUS,
                "emotion_supervision_mask": False,
                "transcript_role": TRANSCRIPT_ROLE,
                "window_transcript_context": context,
                "window_transcript_role": WINDOW_TRANSCRIPT_ROLE,
                "speech_context_available": bool(context),
                "textgrid_units": units,
                "motion_relpath": str(record["motion_relpath"]),
                "audio_relpath": str(record["audio_relpath"]),
                "transcript_relpath": record.get("transcript_relpath"),
                "textgrid_relpath": str(record["textgrid_relpath"]),
                "textgrid_transcript_matches": record.get(
                    "textgrid_transcript_matches"
                ),
                "source_frame_count": int(len(poses)),
                "fps": fps,
                "source_duration_sec": round(float(len(poses)) / fps, 6),
                "audio_sample_rate": sample_rate,
                "audio_channels": audio["channels"],
                "audio_dtype": audio["dtype"],
                "audio_format": audio["format"],
                "audio_frame_count": audio_frame_count,
                "audio_duration_sec": audio["duration_sec"],
                "window": {
                    **metrics,
                    "source_window_index": source_window_index,
                    "start_frame": start_frame,
                    "end_frame_exclusive": end_frame,
                    "frame_count": window_frames,
                    "start_sec": round(start_sec, 6),
                    "end_sec": round(end_sec, 6),
                    "duration_sec": round(window_sec, 6),
                    "selection_status": SELECTION_STATUS,
                    "stride_frames": window_frames,
                    "stride_sec": round(window_sec, 6),
                    "overlap_frames": 0,
                    "audio_start_sample": audio_start_sample,
                    "audio_end_sample_exclusive": audio_end_sample,
                    "audio_sample_count": audio_end_sample - audio_start_sample,
                    "textgrid_start_sec": round(start_sec, 6),
                    "textgrid_end_sec": round(end_sec, 6),
                    "aligned_speech_unit_count": len(units),
                    "motion_bounds_valid": motion_bounds_valid,
                    "audio_bounds_valid": audio_bounds_valid,
                    "textgrid_bounds_valid": textgrid_bounds_valid,
                },
                "source_inventory_record_sha256": record_sha256(record),
                "issues": source_issues,
                "review_state": "windowed_pending_retarget_and_semantic_review",
                "manual_review_required": True,
                "accepted_for_training": False,
            }
        )

    for output_index, window in enumerate(windows):
        window["window"]["source_output_window_index"] = output_index
        window["window"]["source_output_window_count"] = len(windows)
    return {
        "source_clip_id": source_clip_id,
        "speaker_key": speaker["speaker_key"],
        "official_split": record.get("official_split"),
        "source_frame_count": int(len(poses)),
        "window_frames": window_frames,
        "full_motion_window_count": full_motion_window_count,
        "output_window_count": len(windows),
        "trailing_short_frame_count": int(len(poses) % window_frames),
        "rejected_boundary_window_count": rejected_boundary_window_count,
        "textgrid_start_sec": textgrid_start_sec,
        "textgrid_end_sec": textgrid_end_sec,
        "windows": windows,
    }


def validate_output_windows(records: list[dict[str, Any]]) -> dict[str, bool]:
    if not records:
        raise ValueError("No boundary-safe windows were generated")
    clip_ids = [str(record["clip_id"]) for record in records]
    if len(clip_ids) != len(set(clip_ids)):
        raise ValueError("Generated window clip_id values are not unique")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["source_clip_id"])].append(record)
        window = record["window"]
        if window["frame_count"] <= 0:
            raise ValueError(f"Invalid frame count for {record['clip_id']}")
        if not all(
            window[key]
            for key in (
                "motion_bounds_valid",
                "audio_bounds_valid",
                "textgrid_bounds_valid",
            )
        ):
            raise ValueError(f"Unvalidated boundary in {record['clip_id']}")
    for source_clip_id, source_records in grouped.items():
        ordered = sorted(source_records, key=lambda item: item["window"]["start_frame"])
        previous_end = None
        for record in ordered:
            start = record["window"]["start_frame"]
            end = record["window"]["end_frame_exclusive"]
            if previous_end is not None and start < previous_end:
                raise ValueError(f"Overlapping windows for source {source_clip_id}")
            previous_end = end
            if record["source_group_id"] != source_clip_id:
                raise ValueError(f"Incorrect source group for {record['clip_id']}")
    return {
        "unique_window_ids_valid": True,
        "zero_overlap_valid": True,
        "motion_bounds_valid": True,
        "audio_bounds_valid": True,
        "textgrid_bounds_valid": True,
        "source_grouping_valid": True,
    }


def csv_payload(records: list[dict[str, Any]]) -> bytes:
    fields = [
        "clip_id",
        "source_clip_id",
        "source_group_id",
        "speaker_key",
        "session_id",
        "official_split",
        "motion_relpath",
        "audio_relpath",
        "textgrid_relpath",
        "start_frame",
        "end_frame_exclusive",
        "frame_count",
        "start_sec",
        "end_sec",
        "audio_start_sample",
        "audio_end_sample_exclusive",
        "aligned_speech_unit_count",
        "speech_context_available",
        "interaction_energy_mean_rad_s",
        "interaction_energy_p95_rad_s",
        "behavior_id",
        "emotion_id",
        "semantic_label_status",
        "accepted_for_training",
        "issues",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for record in records:
        window = record["window"]
        row = {field: record.get(field) for field in fields}
        row.update({field: window[field] for field in fields if field in window})
        row["issues"] = "|".join(record["issues"])
        writer.writerow(
            row
        )
    return buffer.getvalue().encode("utf-8")


def build_inventory(
    source_inventory: Path = DEFAULT_SOURCE_INVENTORY,
    beat2_root: Path = DEFAULT_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    window_sec: float = TARGET_WINDOW_SEC,
    expected_source_clip_count: int | None = EXPECTED_SOURCE_CLIP_COUNT,
    expected_aligned_source_clip_count: int | None = EXPECTED_ALIGNED_SOURCE_CLIP_COUNT,
    expected_window_count: int | None = EXPECTED_WINDOW_COUNT,
    workers: int = 1,
) -> dict[str, Any]:
    source_inventory = Path(source_inventory).resolve()
    beat2_root = Path(beat2_root).resolve()
    output_dir = Path(output_dir).resolve()
    if window_sec <= 0:
        raise ValueError("window_sec must be positive")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    source_records = read_source_inventory(source_inventory)
    if expected_source_clip_count and len(source_records) != expected_source_clip_count:
        raise ValueError(
            f"Expected {expected_source_clip_count} source clips, found {len(source_records)}"
        )
    aligned_records = [
        record for record in source_records if str(record.get("textgrid_relpath") or "").strip()
    ]
    if not aligned_records:
        raise ValueError("Source inventory contains no TextGrid-aligned clips")
    if (
        expected_aligned_source_clip_count
        and len(aligned_records) != expected_aligned_source_clip_count
    ):
        raise ValueError(
            f"Expected {expected_aligned_source_clip_count} TextGrid-aligned clips, "
            f"found {len(aligned_records)}"
        )

    arguments = [(record, beat2_root, window_sec) for record in aligned_records]
    if workers == 1:
        source_results = [_build_source_windows(item) for item in arguments]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            source_results = list(executor.map(_build_source_windows, arguments))
    source_results.sort(key=lambda result: result["source_clip_id"])
    records = [
        record
        for result in source_results
        for record in sorted(
            result["windows"], key=lambda item: item["window"]["start_frame"]
        )
    ]
    if expected_window_count and len(records) != expected_window_count:
        raise ValueError(
            f"Expected {expected_window_count} boundary-safe windows, found {len(records)}"
        )
    validations = validate_output_windows(records)

    jsonl_bytes = "".join(stable_json(record) + "\n" for record in records).encode(
        "utf-8"
    )
    csv_bytes = csv_payload(records)
    issue_counts = Counter(issue for record in records for issue in record["issues"])
    windows_by_speaker = Counter(record["speaker_key"] for record in records)
    windows_by_split = Counter(record.get("official_split") or "unassigned" for record in records)
    sources_by_speaker = Counter(result["speaker_key"] for result in source_results)
    window_frame_count = sum(record["window"]["frame_count"] for record in records)
    aligned_source_frame_count = sum(result["source_frame_count"] for result in source_results)
    trailing_short_frame_count = sum(
        result["trailing_short_frame_count"] for result in source_results
    )
    rejected_boundary_window_count = sum(
        result["rejected_boundary_window_count"] for result in source_results
    )
    full_motion_window_count = sum(
        result["full_motion_window_count"] for result in source_results
    )
    if full_motion_window_count != len(records) + rejected_boundary_window_count:
        raise ValueError("Full-window accounting is inconsistent")
    if aligned_source_frame_count != (
        full_motion_window_count * records[0]["window"]["frame_count"]
        + trailing_short_frame_count
    ):
        raise ValueError("Source frame accounting is inconsistent")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "BEAT2",
        "dataset_subset": "beat_chinese_v2.0.0",
        "source_inventory": str(source_inventory),
        "source_inventory_sha256": sha256(source_inventory),
        "source_inventory_record_count": len(source_records),
        "textgrid_aligned_source_clip_count": len(source_results),
        "excluded_missing_textgrid_source_clip_count": len(source_records) - len(source_results),
        "window_count": len(records),
        "window_duration_sec": round(window_sec, 6),
        "window_frame_count_each": records[0]["window"]["frame_count"],
        "total_window_frame_count": window_frame_count,
        "total_window_duration_sec": round(window_frame_count / FPS, 6),
        "total_window_duration_hours": round(window_frame_count / FPS / 3600.0, 6),
        "aligned_source_frame_count": aligned_source_frame_count,
        "aligned_source_duration_sec": round(aligned_source_frame_count / FPS, 6),
        "trailing_short_frame_count": trailing_short_frame_count,
        "trailing_short_duration_sec": round(trailing_short_frame_count / FPS, 6),
        "full_motion_window_count_before_boundary_validation": full_motion_window_count,
        "rejected_boundary_window_count": rejected_boundary_window_count,
        "speaker_count": len(sources_by_speaker),
        "source_clip_counts_by_speaker": dict(sorted(sources_by_speaker.items())),
        "window_counts_by_speaker": dict(sorted(windows_by_speaker.items())),
        "window_counts_by_official_split": dict(sorted(windows_by_split.items())),
        "window_counts_by_issue": dict(sorted(issue_counts.items())),
        "windows_with_speech_context_count": sum(
            bool(record["speech_context_available"]) for record in records
        ),
        "windows_without_speech_context_count": sum(
            not bool(record["speech_context_available"]) for record in records
        ),
        "window_policy": {
            "mode": "full_nonoverlap_from_frame_zero",
            "duration_sec": round(window_sec, 6),
            "stride_sec": round(window_sec, 6),
            "overlap_frames": 0,
            "short_tail_policy": "drop_tail_shorter_than_one_full_window",
            "source_requirement": "TextGrid alignment present",
        },
        "semantic_policy": {
            "behavior_id": "unset",
            "emotion_id": "unset",
            "speech_context": "alignment_context_only_not_action_or_emotion_label",
            "training_admission": "deny_until_retarget_qc_and_semantic_review",
        },
        "validation": {
            **validations,
            "frame_accounting_valid": True,
            "source_inventory_metadata_revalidated": True,
        },
        "manual_review_required_count": len(records),
        "accepted_for_training_count": 0,
        "output_sha256": {
            "jsonl": hashlib.sha256(jsonl_bytes).hexdigest(),
            "csv": hashlib.sha256(csv_bytes).hexdigest(),
        },
    }
    summary_bytes = (json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    atomic_write(output_dir / f"{OUTPUT_STEM}.jsonl", jsonl_bytes)
    atomic_write(output_dir / f"{OUTPUT_STEM}.csv", csv_bytes)
    atomic_write(output_dir / f"{OUTPUT_STEM}.summary.json", summary_bytes)
    return summary


def main() -> None:
    args = parse_args()
    summary = build_inventory(
        source_inventory=args.source_inventory,
        beat2_root=args.beat2_root,
        output_dir=args.output_dir,
        window_sec=args.window_sec,
        expected_source_clip_count=args.expected_source_clips or None,
        expected_aligned_source_clip_count=args.expected_aligned_source_clips or None,
        expected_window_count=args.expected_windows or None,
        workers=args.workers,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
