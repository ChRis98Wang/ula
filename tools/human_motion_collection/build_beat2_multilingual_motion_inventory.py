#!/usr/bin/env python3
"""Prepare auditable, resumable BEAT2 multilingual motion-only windows.

The pipeline reads SMPL-X/FLAME motion, official split metadata, TextGrid
alignment, and the labels shipped with BEAT2. It never opens or requires an
audio file. English ``sem`` labels are retained verbatim as time spans;
Japanese and Spanish text is retained only as speech context. No source is
assigned an inferred action or emotion label.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from tools.human_motion_collection.build_beat2_interaction_inventory import (
        FPS,
        HEAD_NECK_JOINT_INDICES,
        INTERACTION_LABEL,
        UPPER_BODY_JOINT_INDICES,
        _window_metrics,
        aligned_transcript_context,
        joint_angular_speed,
        parse_speaker,
        validate_motion_npz,
    )
    from tools.human_motion_collection.build_beat2_full_window_inventory import (
        overlapping_textgrid_units,
    )
except ModuleNotFoundError:  # pragma: no cover - direct invocation by path
    from build_beat2_interaction_inventory import (
        FPS,
        HEAD_NECK_JOINT_INDICES,
        INTERACTION_LABEL,
        UPPER_BODY_JOINT_INDICES,
        _window_metrics,
        aligned_transcript_context,
        joint_angular_speed,
        parse_speaker,
        validate_motion_npz,
    )
    from build_beat2_full_window_inventory import overlapping_textgrid_units


DEFAULT_ROOT = Path("/home/gez/nas/cloud/gez/human_motion/raw/BEAT2")
DEFAULT_OUTPUT_DIR = Path(
    "/home/gez/nas/cloud/gez/human_motion/catalog/"
    "beat2_multilingual_motion_only_v1"
)
DEFAULT_ACQUISITION_MANIFEST = Path(
    "/home/gez/nas/cloud/gez/human_motion/catalog/"
    "beat2_motion_only_acquisition.json"
)
OUTPUT_STEM = "beat2_multilingual_motion_only_6s_v1"
SCHEMA_VERSION = "1.0.0"
STATE_SCHEMA_VERSION = "1.0.0"
WINDOW_SEC = 6.0
SELECTION_STATUS = "full_nonoverlap_boundary_validated"
TIME_TOLERANCE_SEC = 1e-6
PRIORITY_MIN_ENERGY_RAD_S = 0.02
PRIORITY_MAX_P95_ENERGY_RAD_S = 4.0
ALLOWED_SPLITS = {"train", "val", "test", "additional"}
SUBSETS = {
    "english": {
        "dataset_subset": "beat_english_v2.0.0",
        "language_code": "en",
        "annotation_dir": "sem",
        "annotation_kind": "official_gesture_semantics",
    },
    "japanese": {
        "dataset_subset": "beat_japanese_v2.0.0",
        "language_code": "ja",
        "annotation_dir": "text",
        "annotation_kind": "speech_context_only",
    },
    "spanish": {
        "dataset_subset": "beat_spanish_v2.0.0",
        "language_code": "es",
        "annotation_dir": "text",
        "annotation_kind": "speech_context_only",
    },
}
TEXTGRID_ITEM_PATTERN = re.compile(
    r"^\s*item\s*\[\d+\]\s*:\s*(.*?)(?=^\s*item\s*\[\d+\]\s*:|\Z)",
    re.MULTILINE | re.DOTALL,
)
TEXTGRID_INTERVAL_PATTERN = re.compile(
    r"intervals\s*\[\d+\]\s*:\s*"
    r"xmin\s*=\s*([-+0-9.eE]+)\s*"
    r"xmax\s*=\s*([-+0-9.eE]+)\s*"
    r'text\s*=\s*"((?:""|[^"])*)"',
    re.MULTILINE,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--languages", nargs="+", choices=tuple(SUBSETS), default=tuple(SUBSETS)
    )
    parser.add_argument("--window-sec", type=float, default=WINDOW_SEC)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--acquisition-manifest",
        type=Path,
        default=DEFAULT_ACQUISITION_MANIFEST,
        help="Optional downloader manifest used to bind source revision/completeness",
    )
    parser.add_argument(
        "--require-acquisition-manifest",
        action="store_true",
        help="Fail if the downloader verification manifest is unavailable",
    )
    parser.add_argument(
        "--revalidate",
        action="store_true",
        help="Ignore per-source state and validate every input again",
    )
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def atomic_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> str:
    payload = "".join(stable_json(record) + "\n" for record in records).encode("utf-8")
    atomic_bytes(path, payload)
    return hashlib.sha256(payload).hexdigest()


def relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def file_binding(path: Path, root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": relative_path(path, root),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def read_official_splits(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing official split CSV: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id", "type"]:
            raise ValueError(
                f"Unexpected split columns {reader.fieldnames}; expected ['id', 'type']"
            )
        rows = list(reader)
    result: dict[str, str] = {}
    for line_number, row in enumerate(rows, 2):
        clip_id = str(row.get("id") or "").strip()
        split = str(row.get("type") or "").strip()
        if not clip_id or split not in ALLOWED_SPLITS:
            raise ValueError(f"Invalid official split row {line_number}: {row}")
        if clip_id in result:
            raise ValueError(f"Duplicate official split id: {clip_id}")
        result[clip_id] = split
    if not result:
        raise ValueError(f"Official split is empty: {path}")
    return result


def parse_words_textgrid_intervals(path: Path) -> list[tuple[float, float, str]]:
    """Parse only BEAT2's words tier, ignoring parallel phoneme tiers."""
    payload = path.read_text(encoding="utf-8", errors="strict")
    if 'File type = "ooTextFile"' not in payload:
        raise ValueError(f"Unsupported TextGrid header: {path}")
    words_blocks = [
        match.group(1)
        for match in TEXTGRID_ITEM_PATTERN.finditer(payload)
        if re.search(r'^\s*name\s*=\s*"words"\s*$', match.group(1), re.MULTILINE)
    ]
    if len(words_blocks) != 1:
        raise ValueError(
            f"Expected exactly one words tier, found {len(words_blocks)}: {path}"
        )
    intervals: list[tuple[float, float, str]] = []
    previous_end = -math.inf
    for match in TEXTGRID_INTERVAL_PATTERN.finditer(words_blocks[0]):
        start_sec = float(match.group(1))
        end_sec = float(match.group(2))
        text = match.group(3).replace('""', '"')
        if (
            not math.isfinite(start_sec)
            or not math.isfinite(end_sec)
            or end_sec < start_sec
        ):
            raise ValueError(f"Invalid words interval in {path}: {match.group(0)!r}")
        if start_sec + 1e-8 < previous_end:
            raise ValueError(f"Overlapping or unsorted words intervals: {path}")
        intervals.append((start_sec, end_sec, text))
        previous_end = end_sec
    if not intervals:
        raise ValueError(f"No words intervals parsed from TextGrid: {path}")
    return intervals


def parse_english_semantic_spans(path: Path) -> list[dict[str, Any]]:
    """Parse BEAT2 ``sem`` rows without normalizing their source labels."""
    spans: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue
            columns = line.split("\t")
            if len(columns) < 5:
                columns = line.split(maxsplit=5)
            if len(columns) < 5:
                raise ValueError(f"Expected at least five sem columns at {path}:{line_number}")
            source_label = columns[0].strip()
            if not source_label:
                raise ValueError(f"Empty sem source label at {path}:{line_number}")
            try:
                start_sec, end_sec, duration_sec, source_score = map(
                    float, columns[1:5]
                )
            except ValueError as error:
                raise ValueError(
                    f"Non-numeric sem boundary at {path}:{line_number}"
                ) from error
            values = (start_sec, end_sec, duration_sec, source_score)
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Non-finite sem value at {path}:{line_number}")
            if start_sec < 0 or end_sec < start_sec or duration_sec < 0:
                raise ValueError(f"Invalid sem interval at {path}:{line_number}")
            if not math.isclose(
                end_sec - start_sec, duration_sec, rel_tol=0.0, abs_tol=0.002
            ):
                raise ValueError(f"Inconsistent sem duration at {path}:{line_number}")
            lexical_anchor = "\t".join(columns[5:]).strip() or None
            spans.append(
                {
                    "source_label": source_label,
                    "source_start_sec": round(start_sec, 6),
                    "source_end_sec": round(end_sec, 6),
                    "source_duration_sec": round(duration_sec, 6),
                    "source_score": source_score,
                    "source_lexical_anchor": lexical_anchor,
                    "source_line_number": line_number,
                }
            )
    if not spans:
        raise ValueError(f"No semantic spans parsed from {path}")
    return spans


def semantic_spans_for_window(
    spans: list[dict[str, Any]], start_sec: float, end_sec: float
) -> list[dict[str, Any]]:
    result = []
    for span in spans:
        source_start = float(span["source_start_sec"])
        source_end = float(span["source_end_sec"])
        if source_end <= start_sec or source_start >= end_sec:
            continue
        result.append(
            {
                **span,
                "window_start_sec": round(max(source_start, start_sec) - start_sec, 6),
                "window_end_sec": round(min(source_end, end_sec) - start_sec, 6),
            }
        )
    return result


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return round(ordered[lower], 8)
    weight = position - lower
    return round(ordered[lower] * (1.0 - weight) + ordered[upper] * weight, 8)


def select_priority_windows(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select at most one representative low-dynamic window per source."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["source_group_id"])].append(record)
    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for source_group_id, source_records in sorted(grouped.items()):
        candidates = [
            record
            for record in source_records
            if float(record["window"]["interaction_energy_mean_rad_s"])
            >= PRIORITY_MIN_ENERGY_RAD_S
            and float(record["window"]["interaction_energy_p95_rad_s"])
            <= PRIORITY_MAX_P95_ENERGY_RAD_S
        ]
        first = source_records[0]
        if not candidates:
            excluded.append(
                {
                    "dataset_subset": first["dataset_subset"],
                    "language": first["language"],
                    "speaker_key": first["speaker_key"],
                    "source_clip_id": first["source_clip_id"],
                    "source_group_id": source_group_id,
                    "reason": "no_nonstatic_low_dynamic_window",
                    "candidate_window_count": len(source_records),
                    "min_energy_rad_s": PRIORITY_MIN_ENERGY_RAD_S,
                    "max_p95_energy_rad_s": PRIORITY_MAX_P95_ENERGY_RAD_S,
                }
            )
            continue
        semantic_candidates = [
            record
            for record in candidates
            if record["language"] == "english"
            and any(
                span["source_label"] != "01_beat_align"
                for span in record["official_gesture_semantic_spans"]
            )
        ]
        pool = semantic_candidates or candidates
        target_head = _median(
            [float(record["window"]["head_neck_mean_rad_s"]) for record in pool]
        )
        target_energy = _median(
            [
                float(record["window"]["interaction_energy_mean_rad_s"])
                for record in pool
            ]
        )
        chosen = min(
            pool,
            key=lambda record: (
                abs(float(record["window"]["head_neck_mean_rad_s"]) - target_head),
                abs(
                    float(record["window"]["interaction_energy_mean_rad_s"])
                    - target_energy
                ),
                float(record["window"]["interaction_energy_p95_rad_s"]),
                int(record["window"]["start_frame"]),
            ),
        )
        non_beat_labels = sorted(
            {
                str(span["source_label"])
                for span in chosen["official_gesture_semantic_spans"]
                if span["source_label"] != "01_beat_align"
            }
        )
        selected.append(
            {
                **chosen,
                "priority_selection": {
                    "policy": "one_representative_nonstatic_low_dynamic_window_per_source",
                    "reason": (
                        "english_official_non_beat_semantic_low_dynamic"
                        if semantic_candidates
                        else "representative_low_dynamic"
                    ),
                    "source_candidate_window_count": len(source_records),
                    "low_dynamic_candidate_window_count": len(candidates),
                    "selection_pool_window_count": len(pool),
                    "english_semantic_candidate_window_count": len(
                        semantic_candidates
                    ),
                    "official_non_beat_source_labels": non_beat_labels,
                    "target_head_neck_mean_rad_s": round(target_head, 8),
                    "target_interaction_energy_mean_rad_s": round(target_energy, 8),
                    "min_energy_rad_s": PRIORITY_MIN_ENERGY_RAD_S,
                    "max_p95_energy_rad_s": PRIORITY_MAX_P95_ENERGY_RAD_S,
                    "high_dynamic_fallback_allowed": False,
                },
            }
        )
    return selected, excluded


def _source_binding(
    *,
    root: Path,
    motion_path: Path,
    textgrid_path: Path,
    annotation_path: Path,
    subset: str,
    language: str,
    official_split: str | None,
    window_sec: float,
) -> dict[str, Any]:
    files = {}
    for name, path in (
        ("motion", motion_path),
        ("textgrid", textgrid_path),
        ("annotation", annotation_path),
    ):
        files[name] = file_binding(path, root) if path.is_file() else None
    return {
        "state_schema_version": STATE_SCHEMA_VERSION,
        "pipeline_schema_version": SCHEMA_VERSION,
        "dataset_subset": subset,
        "language": language,
        "official_split": official_split,
        "window_sec": window_sec,
        "audio_enabled": False,
        "files": files,
    }


def _reject_source(task: dict[str, Any], reason: str, detail: str) -> dict[str, Any]:
    return {
        "binding": task["binding"],
        "source": {
            "dataset_subset": task["dataset_subset"],
            "language": task["language"],
            "language_code": task["language_code"],
            "source_clip_id": task["source_clip_id"],
            "source_group_id": task["source_group_id"],
            "motion_relpath": task["motion_relpath"],
            "official_split": task["official_split"],
        },
        "windows": [],
        "discards": [
            {
                "discard_scope": "source",
                "dataset_subset": task["dataset_subset"],
                "language": task["language"],
                "source_clip_id": task["source_clip_id"],
                "source_group_id": task["source_group_id"],
                "reason": reason,
                "detail": detail,
            }
        ],
        "source_stats": {
            "source_frame_count": 0,
            "full_motion_window_count": 0,
            "output_window_count": 0,
            "rejected_boundary_window_count": 0,
            "trailing_short_frame_count": 0,
        },
    }


def _process_source(task: dict[str, Any]) -> dict[str, Any]:
    root = Path(task["root"])
    motion_path = root / task["motion_relpath"]
    textgrid_path = root / task["textgrid_relpath"]
    annotation_path = root / task["annotation_relpath"]
    if task["official_split"] is None:
        return _reject_source(task, "missing_official_split", "No matching split CSV row")
    if not textgrid_path.is_file():
        return _reject_source(task, "missing_textgrid", task["textgrid_relpath"])
    if not annotation_path.is_file():
        reason = (
            "missing_official_semantic_annotation"
            if task["annotation_kind"] == "official_gesture_semantics"
            else "missing_speech_context_text"
        )
        return _reject_source(task, reason, task["annotation_relpath"])

    try:
        poses, _trans, fps = validate_motion_npz(motion_path)
    except (OSError, ValueError) as error:
        return _reject_source(task, "invalid_motion_npz", str(error))
    try:
        intervals = parse_words_textgrid_intervals(textgrid_path)
    except (OSError, UnicodeError, ValueError) as error:
        return _reject_source(task, "invalid_textgrid", str(error))

    semantic_spans: list[dict[str, Any]] = []
    source_text: str | None = None
    if task["annotation_kind"] == "official_gesture_semantics":
        try:
            semantic_spans = parse_english_semantic_spans(annotation_path)
        except (OSError, UnicodeError, ValueError) as error:
            return _reject_source(task, "invalid_official_semantic_annotation", str(error))
    else:
        try:
            source_text = " ".join(
                annotation_path.read_text(encoding="utf-8-sig").strip().split()
            )
        except (OSError, UnicodeError) as error:
            return _reject_source(task, "invalid_speech_context_text", str(error))
        if not source_text:
            return _reject_source(
                task, "empty_speech_context_text", task["annotation_relpath"]
            )

    window_frames_float = float(task["window_sec"]) * fps
    window_frames = int(round(window_frames_float))
    if window_frames < 2 or not math.isclose(
        window_frames_float, window_frames, rel_tol=0.0, abs_tol=1e-9
    ):
        return _reject_source(
            task,
            "invalid_window_contract",
            f"{task['window_sec']} sec is not an integer frame count at {fps} Hz",
        )
    full_motion_window_count = len(poses) // window_frames
    if full_motion_window_count == 0:
        result = _reject_source(
            task,
            "source_shorter_than_window",
            f"frames={len(poses)}, required={window_frames}",
        )
        result["source_stats"]["source_frame_count"] = int(len(poses))
        result["source_stats"]["trailing_short_frame_count"] = int(len(poses))
        return result

    textgrid_start_sec = min(start for start, _end, _text in intervals)
    textgrid_end_sec = max(end for _start, end, _text in intervals)
    upper_speed = joint_angular_speed(poses, UPPER_BODY_JOINT_INDICES)
    head_neck_speed = joint_angular_speed(poses, HEAD_NECK_JOINT_INDICES)
    speaker = parse_speaker(task["source_clip_id"])
    source_sha256 = sha256_file(motion_path)
    textgrid_sha256 = sha256_file(textgrid_path)
    annotation_sha256 = sha256_file(annotation_path)
    windows: list[dict[str, Any]] = []
    discards: list[dict[str, Any]] = []

    for source_window_index in range(full_motion_window_count):
        start_frame = source_window_index * window_frames
        end_frame = start_frame + window_frames
        start_sec = start_frame / fps
        end_sec = end_frame / fps
        if not (
            start_sec + TIME_TOLERANCE_SEC >= textgrid_start_sec
            and end_sec <= textgrid_end_sec + TIME_TOLERANCE_SEC
        ):
            discards.append(
                {
                    "discard_scope": "window",
                    "dataset_subset": task["dataset_subset"],
                    "language": task["language"],
                    "source_clip_id": task["source_clip_id"],
                    "source_group_id": task["source_group_id"],
                    "start_frame": start_frame,
                    "end_frame_exclusive": end_frame,
                    "reason": "textgrid_boundary_mismatch",
                    "detail": (
                        f"window=[{start_sec:.6f},{end_sec:.6f}], "
                        f"textgrid=[{textgrid_start_sec:.6f},{textgrid_end_sec:.6f}]"
                    ),
                }
            )
            continue

        raw_window_id = (
            f"{task['source_clip_id']}_f{start_frame:06d}-{end_frame:06d}"
        )
        window_id = f"{task['dataset_subset']}__{raw_window_id}"
        textgrid_units = overlapping_textgrid_units(intervals, start_sec, end_sec)
        speech_context = aligned_transcript_context(intervals, start_sec, end_sec)
        metrics = _window_metrics(
            upper_speed, head_neck_speed, start_frame, window_frames
        )
        metrics = {
            key: round(value, 8) if isinstance(value, float) else value
            for key, value in metrics.items()
        }
        official_spans = semantic_spans_for_window(
            semantic_spans, start_sec, end_sec
        )
        windows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset": "BEAT2",
                "dataset_subset": task["dataset_subset"],
                "language": task["language"],
                "language_code": task["language_code"],
                "clip_id": window_id,
                "window_id": window_id,
                "task_id": window_id,
                "source_clip_id": task["source_clip_id"],
                "source_group_id": task["source_group_id"],
                "source_group_key": task["source_group_id"],
                "split_group_id": task["source_group_id"],
                "source_group_policy": "all_windows_from_source_clip_share_split",
                **speaker,
                "official_split": task["official_split"],
                "interaction_label": INTERACTION_LABEL,
                "interaction_label_source": "dataset_scope_only",
                "behavior_id": None,
                "emotion_id": None,
                "emotion_supervision_mask": False,
                "emotion_label_status": "not_provided_not_inferred",
                "semantic_label_status": (
                    "official_gesture_spans_preserved_pending_robot_mapping"
                    if task["annotation_kind"] == "official_gesture_semantics"
                    else "speech_context_only_no_gesture_semantics"
                ),
                "official_gesture_semantic_spans": official_spans,
                "official_gesture_semantic_role": (
                    "source_labels_preserved_verbatim_not_emotion_labels"
                    if task["annotation_kind"] == "official_gesture_semantics"
                    else "unavailable"
                ),
                "window_transcript_context": speech_context,
                "window_transcript_role": (
                    "time_aligned_speech_context_only_not_action_or_emotion_label"
                ),
                "speech_context_available": bool(speech_context),
                "textgrid_units": textgrid_units,
                "source_text": source_text,
                "source_text_role": (
                    "clip_level_speech_context_only_not_action_or_emotion_label"
                    if source_text is not None
                    else "unavailable_sem_file_is_not_transcript"
                ),
                "motion_relpath": task["motion_relpath"],
                "textgrid_relpath": task["textgrid_relpath"],
                "annotation_relpath": task["annotation_relpath"],
                "annotation_kind": task["annotation_kind"],
                "audio_enabled": False,
                "audio_policy": "disabled_not_read_not_required",
                "source_frame_count": int(len(poses)),
                "source_duration_sec": round(len(poses) / fps, 6),
                "fps": fps,
                "source_motion_sha256": source_sha256,
                "source_textgrid_sha256": textgrid_sha256,
                "source_annotation_sha256": annotation_sha256,
                "window": {
                    **metrics,
                    "source_window_index": source_window_index,
                    "start_frame": start_frame,
                    "end_frame_exclusive": end_frame,
                    "frame_count": window_frames,
                    "start_sec": round(start_sec, 6),
                    "end_sec": round(end_sec, 6),
                    "duration_sec": round(float(task["window_sec"]), 6),
                    "selection_status": SELECTION_STATUS,
                    "stride_frames": window_frames,
                    "stride_sec": round(float(task["window_sec"]), 6),
                    "overlap_frames": 0,
                    "motion_bounds_valid": True,
                    "textgrid_bounds_valid": True,
                    "aligned_speech_unit_count": len(textgrid_units),
                    "official_gesture_semantic_span_count": len(official_spans),
                },
                "issues": [],
                "review_state": "windowed_pending_retarget_and_semantic_admission",
                "manual_review_required": True,
                "accepted_for_training": False,
            }
        )

    for output_index, window in enumerate(windows):
        window["window"]["source_output_window_index"] = output_index
        window["window"]["source_output_window_count"] = len(windows)
    return {
        "binding": task["binding"],
        "source": {
            "dataset_subset": task["dataset_subset"],
            "language": task["language"],
            "language_code": task["language_code"],
            "source_clip_id": task["source_clip_id"],
            "source_group_id": task["source_group_id"],
            "motion_relpath": task["motion_relpath"],
            "official_split": task["official_split"],
            "motion_sha256": source_sha256,
            "textgrid_sha256": textgrid_sha256,
            "annotation_sha256": annotation_sha256,
        },
        "windows": windows,
        "discards": discards,
        "source_stats": {
            "source_frame_count": int(len(poses)),
            "full_motion_window_count": full_motion_window_count,
            "output_window_count": len(windows),
            "rejected_boundary_window_count": len(discards),
            "trailing_short_frame_count": int(len(poses) % window_frames),
            "official_semantic_span_count": len(semantic_spans),
        },
    }


def _state_path(state_dir: Path, task: dict[str, Any]) -> Path:
    key = f"{task['dataset_subset']}/{task['source_clip_id']}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return state_dir / task["dataset_subset"] / f"{task['source_clip_id']}.{digest}.json"


def _read_cached_result(path: Path, binding: dict[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("binding") != binding:
        return None
    if not isinstance(value.get("windows"), list) or not isinstance(
        value.get("discards"), list
    ):
        return None
    return value


def _load_acquisition_manifest(
    path: Path | None, root: Path, languages: list[str], require: bool
) -> dict[str, Any]:
    if path is None or not path.is_file():
        if require:
            raise FileNotFoundError(f"Missing acquisition manifest: {path}")
        return {"status": "not_available", "path": str(path) if path else None}
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("artifact_kind") != "beat2_motion_only_acquisition":
        raise ValueError(f"Unexpected acquisition manifest kind: {path}")
    if value.get("audio_policy") != "excluded_not_downloaded":
        raise ValueError("Acquisition manifest does not prove audio exclusion")
    manifest_root = Path(str(value.get("root") or "")).resolve()
    if manifest_root != root.resolve():
        raise ValueError(
            f"Acquisition root mismatch: manifest={manifest_root}, requested={root}"
        )
    manifest_languages = set(value.get("languages") or [])
    missing_languages = sorted(set(languages) - manifest_languages)
    if missing_languages:
        raise ValueError(
            f"Acquisition manifest does not cover languages: {missing_languages}"
        )
    verification = value.get("verification") or {}
    if verification.get("all_selected_files_present") is not True or verification.get(
        "all_selected_sizes_match"
    ) is not True:
        raise ValueError("Acquisition manifest is not fully verified")
    return {
        "status": "verified",
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "source": value.get("source"),
        "selection_sha256": value.get("selection_sha256"),
        "audio_policy": value.get("audio_policy"),
    }


def _discover_tasks(
    root: Path, languages: list[str], window_sec: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    metadata_discards: list[dict[str, Any]] = []
    discovery: dict[str, Any] = {}
    for language in languages:
        layout = SUBSETS[language]
        subset = str(layout["dataset_subset"])
        subset_root = root / subset
        motion_dir = subset_root / "smplxflame_30"
        if not motion_dir.is_dir():
            raise FileNotFoundError(f"Missing motion directory: {motion_dir}")
        splits = read_official_splits(subset_root / "train_test_split.csv")
        motion_paths = sorted(motion_dir.glob("*.npz"))
        if not motion_paths:
            raise ValueError(f"No motion NPZ files found: {motion_dir}")
        motion_ids = {path.stem for path in motion_paths}
        for missing_motion_id in sorted(set(splits) - motion_ids):
            metadata_discards.append(
                {
                    "discard_scope": "official_split_entry",
                    "dataset_subset": subset,
                    "language": language,
                    "source_clip_id": missing_motion_id,
                    "source_group_id": f"BEAT2/{subset}/{missing_motion_id}",
                    "reason": "official_split_entry_missing_motion",
                    "detail": f"official_split={splits[missing_motion_id]}",
                }
            )
        for motion_path in motion_paths:
            source_clip_id = motion_path.stem
            textgrid_path = subset_root / "textgrid" / f"{source_clip_id}.TextGrid"
            annotation_path = (
                subset_root
                / str(layout["annotation_dir"])
                / f"{source_clip_id}.txt"
            )
            official_split = splits.get(source_clip_id)
            source_group_id = f"BEAT2/{subset}/{source_clip_id}"
            task = {
                "root": str(root),
                "dataset_subset": subset,
                "language": language,
                "language_code": layout["language_code"],
                "annotation_kind": layout["annotation_kind"],
                "source_clip_id": source_clip_id,
                "source_group_id": source_group_id,
                "official_split": official_split,
                "motion_relpath": relative_path(motion_path, root),
                "textgrid_relpath": relative_path(textgrid_path, root),
                "annotation_relpath": relative_path(annotation_path, root),
                "window_sec": window_sec,
            }
            task["binding"] = _source_binding(
                root=root,
                motion_path=motion_path,
                textgrid_path=textgrid_path,
                annotation_path=annotation_path,
                subset=subset,
                language=language,
                official_split=official_split,
                window_sec=window_sec,
            )
            tasks.append(task)
        discovery[subset] = {
            "language": language,
            "language_code": layout["language_code"],
            "motion_npz_count": len(motion_paths),
            "official_split_entry_count": len(splits),
            "motion_with_official_split_count": sum(path.stem in splits for path in motion_paths),
            "official_split_entry_missing_motion_count": len(set(splits) - motion_ids),
            "textgrid_file_count": len(list((subset_root / "textgrid").glob("*.TextGrid"))),
            "annotation_file_count": len(
                list((subset_root / str(layout["annotation_dir"])).glob("*.txt"))
            ),
        }
    tasks.sort(key=lambda item: (item["dataset_subset"], item["source_clip_id"]))
    return tasks, metadata_discards, discovery


def _validate_windows(records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("No boundary-valid multilingual windows were generated")
    ids = [record["clip_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate multilingual window clip_id")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("audio_enabled") is not False or "audio_relpath" in record:
            raise ValueError(f"Audio contract violated by {record['clip_id']}")
        if record.get("emotion_id") is not None or record.get(
            "emotion_supervision_mask"
        ) is not False:
            raise ValueError(f"Emotion inference contract violated by {record['clip_id']}")
        grouped[record["source_group_id"]].append(record)
    for source_group, source_records in grouped.items():
        previous_end = None
        for record in sorted(
            source_records, key=lambda item: item["window"]["start_frame"]
        ):
            window = record["window"]
            if window["selection_status"] != SELECTION_STATUS:
                raise ValueError(f"Unexpected selection status: {record['clip_id']}")
            if previous_end is not None and window["start_frame"] < previous_end:
                raise ValueError(f"Overlapping windows for {source_group}")
            previous_end = window["end_frame_exclusive"]


def build_inventory(
    root: Path = DEFAULT_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    languages: Iterable[str] = tuple(SUBSETS),
    window_sec: float = WINDOW_SEC,
    workers: int = 4,
    acquisition_manifest: Path | None = DEFAULT_ACQUISITION_MANIFEST,
    require_acquisition_manifest: bool = False,
    revalidate: bool = False,
) -> dict[str, Any]:
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    languages = list(dict.fromkeys(languages))
    unknown = sorted(set(languages) - set(SUBSETS))
    if unknown or not languages:
        raise ValueError(f"Invalid language selection: {unknown or languages}")
    if window_sec <= 0:
        raise ValueError("window_sec must be positive")
    if workers < 1:
        raise ValueError("workers must be at least one")
    acquisition = _load_acquisition_manifest(
        acquisition_manifest, root, languages, require_acquisition_manifest
    )
    tasks, metadata_discards, discovery = _discover_tasks(root, languages, window_sec)
    state_dir = output_dir / "state"
    results: list[dict[str, Any]] = []
    pending: list[tuple[dict[str, Any], Path]] = []
    reused_state_count = 0
    for task in tasks:
        state_path = _state_path(state_dir, task)
        cached = None if revalidate else _read_cached_result(state_path, task["binding"])
        if cached is not None:
            results.append(cached)
            reused_state_count += 1
        else:
            pending.append((task, state_path))

    if workers == 1:
        for task, state_path in pending:
            result = _process_source(task)
            atomic_json(state_path, result)
            results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_state = {
                executor.submit(_process_source, task): state_path
                for task, state_path in pending
            }
            for future in as_completed(future_to_state):
                result = future.result()
                atomic_json(future_to_state[future], result)
                results.append(result)

    results.sort(
        key=lambda item: (
            item["source"]["dataset_subset"], item["source"]["source_clip_id"]
        )
    )
    records = sorted(
        (record for result in results for record in result["windows"]),
        key=lambda record: (
            record["dataset_subset"],
            record["source_clip_id"],
            record["window"]["start_frame"],
        ),
    )
    discards = sorted(
        [*metadata_discards, *(item for result in results for item in result["discards"])],
        key=lambda item: (
            item["dataset_subset"],
            item.get("source_clip_id", ""),
            item.get("start_frame", -1),
            item["reason"],
        ),
    )
    _validate_windows(records)
    priority_records, priority_excluded = select_priority_windows(records)
    priority_source_groups = [record["source_group_id"] for record in priority_records]
    if len(priority_source_groups) != len(set(priority_source_groups)):
        raise ValueError("Priority manifest contains more than one window per source")
    if any(
        float(record["window"]["interaction_energy_p95_rad_s"])
        > PRIORITY_MAX_P95_ENERGY_RAD_S
        for record in priority_records
    ):
        raise ValueError("Priority manifest contains a high-dynamic window")
    inventory_path = output_dir / f"{OUTPUT_STEM}.jsonl"
    discard_path = output_dir / f"{OUTPUT_STEM}.discarded.jsonl"
    priority_path = output_dir / "priority_manifest.jsonl"
    priority_excluded_path = output_dir / "priority_excluded.jsonl"
    inventory_sha256 = atomic_jsonl(inventory_path, records)
    discard_sha256 = atomic_jsonl(discard_path, discards)
    priority_sha256 = atomic_jsonl(priority_path, priority_records)
    priority_excluded_sha256 = atomic_jsonl(
        priority_excluded_path, priority_excluded
    )

    windows_by_subset = Counter(record["dataset_subset"] for record in records)
    windows_by_language = Counter(record["language"] for record in records)
    windows_by_split = Counter(record["official_split"] for record in records)
    sources_by_subset = Counter(
        result["source"]["dataset_subset"] for result in results if result["windows"]
    )
    discard_reasons = Counter(item["reason"] for item in discards)
    priority_by_language = Counter(
        record["language"] for record in priority_records
    )
    priority_by_speaker = Counter(
        record["speaker_key"] for record in priority_records
    )
    priority_by_subset = Counter(
        record["dataset_subset"] for record in priority_records
    )
    priority_by_split = Counter(
        record["official_split"] for record in priority_records
    )
    priority_excluded_reasons = Counter(
        record["reason"] for record in priority_excluded
    )
    priority_energy_mean = [
        float(record["window"]["interaction_energy_mean_rad_s"])
        for record in priority_records
    ]
    priority_energy_p95 = [
        float(record["window"]["interaction_energy_p95_rad_s"])
        for record in priority_records
    ]
    priority_head_mean = [
        float(record["window"]["head_neck_mean_rad_s"])
        for record in priority_records
    ]
    total_frames = sum(record["window"]["frame_count"] for record in records)
    total_source_frames = sum(
        result["source_stats"]["source_frame_count"] for result in results
    )
    total_tail_frames = sum(
        result["source_stats"]["trailing_short_frame_count"] for result in results
    )
    source_rejected_count = sum(not result["windows"] for result in results)
    english_records = [record for record in records if record["language"] == "english"]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "beat2_multilingual_motion_only_window_inventory",
        "created_at_utc": utc_now(),
        "root": str(root),
        "languages": languages,
        "dataset_subsets": [SUBSETS[language]["dataset_subset"] for language in languages],
        "source_acquisition": acquisition,
        "audio_policy": {
            "enabled": False,
            "audio_files_read": False,
            "audio_files_required": False,
            "contract": "motion_text_metadata_only",
        },
        "source_clip_count": len(tasks),
        "source_clip_with_output_count": len(tasks) - source_rejected_count,
        "source_clip_rejected_count": source_rejected_count,
        "window_count": len(records),
        "window_duration_sec": window_sec,
        "window_frame_count_each": int(round(window_sec * FPS)),
        "total_window_frame_count": total_frames,
        "total_window_duration_sec": round(total_frames / FPS, 6),
        "total_window_duration_hours": round(total_frames / FPS / 3600.0, 6),
        "total_discovered_source_frame_count": total_source_frames,
        "total_discovered_source_duration_hours": round(
            total_source_frames / FPS / 3600.0, 6
        ),
        "total_short_tail_frame_count": total_tail_frames,
        "source_clip_counts_by_subset_with_output": dict(sorted(sources_by_subset.items())),
        "window_counts_by_subset": dict(sorted(windows_by_subset.items())),
        "window_counts_by_language": dict(sorted(windows_by_language.items())),
        "window_counts_by_official_split": dict(sorted(windows_by_split.items())),
        "discard_counts_by_reason": dict(sorted(discard_reasons.items())),
        "english_window_with_official_gesture_span_count": sum(
            bool(record["official_gesture_semantic_spans"])
            for record in english_records
        ),
        "english_window_without_official_gesture_span_count": sum(
            not bool(record["official_gesture_semantic_spans"])
            for record in english_records
        ),
        "official_gesture_semantic_span_instances_in_windows": sum(
            len(record["official_gesture_semantic_spans"])
            for record in english_records
        ),
        "speech_context_window_count": sum(
            bool(record["speech_context_available"]) for record in records
        ),
        "discovery": discovery,
        "window_policy": {
            "mode": "full_nonoverlap_from_frame_zero",
            "stride_sec": window_sec,
            "overlap_frames": 0,
            "short_tail_policy": "discard_and_count",
            "boundary_contract": "motion_and_words_textgrid",
            "fps": FPS,
        },
        "semantic_policy": {
            "english": "official_sem_source_labels_preserved_verbatim_as_spans",
            "japanese": "text_and_textgrid_are_speech_context_only",
            "spanish": "text_and_textgrid_are_speech_context_only",
            "emotion": "not_provided_not_inferred_supervision_mask_false",
            "training_admission": "false_pending_retarget_qc_and_semantic_mapping",
        },
        "priority_manifest": {
            "window_count": len(priority_records),
            "source_group_count": len(priority_source_groups),
            "max_windows_per_source": 1,
            "counts_by_subset": dict(sorted(priority_by_subset.items())),
            "counts_by_language": dict(sorted(priority_by_language.items())),
            "counts_by_speaker": dict(sorted(priority_by_speaker.items())),
            "counts_by_official_split": dict(sorted(priority_by_split.items())),
            "english_official_non_beat_semantic_count": sum(
                record["priority_selection"]["reason"]
                == "english_official_non_beat_semantic_low_dynamic"
                for record in priority_records
            ),
            "excluded_source_count": len(priority_excluded),
            "excluded_counts_by_reason": dict(
                sorted(priority_excluded_reasons.items())
            ),
            "selection_policy": {
                "nonstatic_interaction_energy_mean_min_rad_s": (
                    PRIORITY_MIN_ENERGY_RAD_S
                ),
                "low_dynamic_interaction_energy_p95_max_rad_s": (
                    PRIORITY_MAX_P95_ENERGY_RAD_S
                ),
                "english_first_tier": (
                    "contains_official_span_whose_source_label_is_not_01_beat_align"
                ),
                "representative_head_policy": (
                    "closest_to_per_source_candidate_median_not_maximum"
                ),
                "high_dynamic_fallback_allowed": False,
            },
            "interaction_energy_mean_rad_s": {
                "p50": _percentile(priority_energy_mean, 0.5),
                "p90": _percentile(priority_energy_mean, 0.9),
                "p95": _percentile(priority_energy_mean, 0.95),
                "max": max(priority_energy_mean) if priority_energy_mean else None,
            },
            "interaction_energy_p95_rad_s": {
                "p50": _percentile(priority_energy_p95, 0.5),
                "p90": _percentile(priority_energy_p95, 0.9),
                "p95": _percentile(priority_energy_p95, 0.95),
                "max": max(priority_energy_p95) if priority_energy_p95 else None,
            },
            "head_neck_mean_rad_s": {
                "p50": _percentile(priority_head_mean, 0.5),
                "p90": _percentile(priority_head_mean, 0.9),
                "p95": _percentile(priority_head_mean, 0.95),
                "max": max(priority_head_mean) if priority_head_mean else None,
            },
        },
        "resume": {
            "state_dir": str(state_dir),
            "state_binding": "path_size_mtime_split_window_contract",
            "source_state_reused_count": reused_state_count,
            "source_state_computed_count": len(pending),
            "revalidate": revalidate,
        },
        "validation": {
            "motion_npz_30hz_validated_per_emitted_source": True,
            "official_split_validated_per_emitted_source": True,
            "textgrid_boundaries_validated_per_window": True,
            "unique_window_ids": True,
            "nonoverlapping_windows": True,
            "priority_at_most_one_window_per_source": True,
            "priority_high_dynamic_fallback_absent": True,
            "audio_disabled": True,
            "emotion_not_inferred": True,
        },
        "outputs": {
            "inventory_jsonl": str(inventory_path),
            "inventory_sha256": inventory_sha256,
            "discarded_jsonl": str(discard_path),
            "discarded_sha256": discard_sha256,
            "priority_manifest_jsonl": str(priority_path),
            "priority_manifest_sha256": priority_sha256,
            "priority_excluded_jsonl": str(priority_excluded_path),
            "priority_excluded_sha256": priority_excluded_sha256,
        },
        "accepted_for_training_count": 0,
    }
    summary_path = output_dir / f"{OUTPUT_STEM}.summary.json"
    summary["outputs"]["summary_json"] = str(summary_path)
    atomic_json(summary_path, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_inventory(
        root=args.root,
        output_dir=args.output_dir,
        languages=args.languages,
        window_sec=args.window_sec,
        workers=args.workers,
        acquisition_manifest=args.acquisition_manifest,
        require_acquisition_manifest=args.require_acquisition_manifest,
        revalidate=args.revalidate,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
