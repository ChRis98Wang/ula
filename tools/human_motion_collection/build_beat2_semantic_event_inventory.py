#!/usr/bin/env python3
"""Build variable-length BEAT2 English semantic-event motion candidates.

Official ``sem`` intervals, rather than a fixed-duration window, define each
candidate. Motion-derived low-speed boundaries may add a small amount of
context without crossing another official semantic event. Audio is disabled.
All candidates remain denied for training until robot retarget quality control.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from tools.human_motion_collection.build_beat2_full_window_inventory import (
        overlapping_textgrid_units,
    )
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
except ModuleNotFoundError:  # pragma: no cover - direct invocation by path
    from build_beat2_full_window_inventory import overlapping_textgrid_units
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


DEFAULT_ROOT = Path("/home/gez/nas/cloud/gez/human_motion/raw/BEAT2")
DEFAULT_OUTPUT_DIR = Path(
    "/home/gez/nas/cloud/gez/human_motion/catalog/"
    "beat2_semantic_event_inventory_v1"
)
DATASET_SUBSET = "beat_english_v2.0.0"
OUTPUT_STEM = "beat2_semantic_event_inventory_v1"
SCHEMA_VERSION = "1.0.0"
STATE_SCHEMA_VERSION = "1.0.0"
REPRESENTATION = "native_variable_length_semantic_clip_v1"
SELECTION_STATUS = "official_semantic_event_variable_length_boundary_validated"
DEFAULT_MAX_CONTEXT_SEC = 0.75
TIME_TOLERANCE_SEC = 1e-6
ALLOWED_SPLITS = {"train", "val", "test", "additional"}
CONTROL_LABELS = {"01_beat_align", "00_nogesture", "habit", "need_cut"}
INTENSITY_NAMES = {"l": "low", "m": "medium", "h": "high"}
OFFICIAL_EVENT_LABELS = {
    f"{code:02d}_{category}_{intensity}": (category, intensity)
    for code, category, intensity in (
        (2, "deictic", "l"),
        (3, "deictic", "m"),
        (4, "deictic", "h"),
        (5, "iconic", "l"),
        (6, "iconic", "m"),
        (7, "iconic", "h"),
        (8, "metaphoric", "l"),
        (9, "metaphoric", "m"),
        (10, "metaphoric", "h"),
    )
}
FILENAME_PATTERN = re.compile(
    r"^(?P<speaker_id>\d+)_(?P<speaker_name>.+)_"
    r"(?P<recording_type>\d+)_(?P<start_id>\d+)_(?P<end_id>\d+)$"
)
CONVERSATION_RECORDING_TYPES = {1, 3, 5, 7}
NETWORK_EMOTIONS = {"neutral", "sad", "happy", "angry", "surprise", "fear"}
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
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--max-context-sec", type=float, default=DEFAULT_MAX_CONTEXT_SEC
    )
    parser.add_argument("--revalidate", action="store_true")
    parser.add_argument(
        "--estimate-sem-only",
        action="store_true",
        help="Print a static label estimate without requiring motion or split files",
    )
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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
    return sha256_bytes(payload)


def relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def file_stat_binding(path: Path, root: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
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
        result: dict[str, str] = {}
        for line_number, row in enumerate(reader, 2):
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


def parse_semantic_spans(path: Path) -> list[dict[str, Any]]:
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
            try:
                start_sec, end_sec, duration_sec, score = map(float, columns[1:5])
            except ValueError as error:
                raise ValueError(
                    f"Non-numeric sem boundary at {path}:{line_number}"
                ) from error
            if not all(
                math.isfinite(value)
                for value in (start_sec, end_sec, duration_sec, score)
            ):
                raise ValueError(f"Non-finite sem value at {path}:{line_number}")
            if start_sec < 0 or end_sec < start_sec or duration_sec < 0:
                raise ValueError(f"Invalid sem interval at {path}:{line_number}")
            if not math.isclose(
                end_sec - start_sec, duration_sec, rel_tol=0.0, abs_tol=0.002
            ):
                raise ValueError(f"Inconsistent sem duration at {path}:{line_number}")
            lexical_anchor = "\t".join(columns[5:]).strip() or None
            event_descriptor = OFFICIAL_EVENT_LABELS.get(source_label)
            spans.append(
                {
                    "source_label": source_label,
                    "source_start_sec": round(start_sec, 6),
                    "source_end_sec": round(end_sec, 6),
                    "source_duration_sec": round(duration_sec, 6),
                    "source_score": score,
                    "source_lexical_anchor": lexical_anchor,
                    "source_line_number": line_number,
                    "is_training_semantic_event": event_descriptor is not None,
                    "category": event_descriptor[0] if event_descriptor else None,
                    "intensity_code": event_descriptor[1] if event_descriptor else None,
                    "intensity": (
                        INTENSITY_NAMES[event_descriptor[1]]
                        if event_descriptor
                        else None
                    ),
                }
            )
    if not spans:
        raise ValueError(f"No semantic spans parsed from {path}")
    return spans


def speech_emotion_for_id(speech_id: int) -> tuple[str, str | None]:
    ranges = (
        (0, 64, "neutral", "neutral"),
        (65, 72, "happiness", "happy"),
        (73, 80, "anger", "angry"),
        (81, 86, "sadness", "sad"),
        (87, 94, "contempt", None),
        (95, 102, "surprise", "surprise"),
        (103, 110, "fear", "fear"),
        (111, 118, "disgust", None),
    )
    for lower, upper, raw_label, network_label in ranges:
        if lower <= speech_id <= upper:
            return raw_label, network_label
    raise ValueError(f"speech_id_outside_official_protocol:{speech_id}")


def parse_filename_emotion(clip_id: str) -> dict[str, Any]:
    match = FILENAME_PATTERN.fullmatch(clip_id)
    if match is None:
        raise ValueError("filename_does_not_match_official_protocol")
    fields = match.groupdict()
    recording_type = int(fields["recording_type"])
    start_id = int(fields["start_id"])
    end_id = int(fields["end_id"])
    if end_id < start_id:
        raise ValueError("filename_id_range_reversed")
    if recording_type == 0:
        start_raw, start_network = speech_emotion_for_id(start_id)
        end_raw, end_network = speech_emotion_for_id(end_id)
        if (start_raw, start_network) != (end_raw, end_network):
            raise ValueError(
                "ambiguous_speech_id_range_crosses_emotion_protocol_boundary"
            )
        raw_label = start_raw
        network_label = start_network
        protocol_kind = "speech_id_range"
    elif recording_type in CONVERSATION_RECORDING_TYPES:
        raw_label = "neutral"
        network_label = "neutral"
        protocol_kind = "conversation_recording_type"
    else:
        raise ValueError(f"unsupported_recording_type:{recording_type}")
    return {
        "recording_type": recording_type,
        "recording_id_start": start_id,
        "recording_id_end": end_id,
        "emotion_protocol_kind": protocol_kind,
        "source_emotion_label": raw_label,
        "emotion_id": network_label,
        "emotion_supervision_mask": network_label in NETWORK_EMOTIONS,
        "emotion_label_status": (
            "official_filename_protocol_network_supported"
            if network_label in NETWORK_EMOTIONS
            else "official_filename_protocol_preserved_network_unsupported"
        ),
        "emotion_label_source": "official_beat2_filename_protocol",
    }


def _aligned_core_frames(
    start_sec: float, end_sec: float, frame_count: int
) -> tuple[int, int]:
    start_frame = max(0, int(math.floor(start_sec * FPS + 1e-9)))
    end_frame = min(frame_count, int(math.ceil(end_sec * FPS - 1e-9)))
    return start_frame, end_frame


def _boundary_energy(energy: np.ndarray, boundary_frame: int, frame_count: int) -> float:
    if energy.ndim != 1 or energy.size != max(0, frame_count - 1):
        raise ValueError(
            "boundary energy must contain exactly one value per frame transition"
        )
    if energy.size == 0:
        raise ValueError("at least two motion frames are required for boundary scoring")
    if boundary_frame <= 0:
        return float(energy[0])
    if boundary_frame >= frame_count:
        return float(energy[-1])
    # ``energy`` describes transitions, so it has N-1 entries for N frames.
    # Clamp the right transition at the final in-sequence boundary (N-1).
    left = min(boundary_frame - 1, energy.size - 1)
    right = min(boundary_frame, energy.size - 1)
    return float(0.5 * (energy[left] + energy[right]))


def _choose_low_speed_boundary(
    energy: np.ndarray,
    lower_frame: int,
    upper_frame: int,
    core_boundary_frame: int,
    frame_count: int,
) -> tuple[int, float]:
    candidates = range(lower_frame, upper_frame + 1)
    chosen = min(
        candidates,
        key=lambda frame: (
            _boundary_energy(energy, frame, frame_count),
            -abs(frame - core_boundary_frame),
            frame,
        ),
    )
    return chosen, _boundary_energy(energy, chosen, frame_count)


def _span_audit_fields(span: dict[str, Any]) -> dict[str, Any]:
    return {
        key: span[key]
        for key in (
            "source_label",
            "source_start_sec",
            "source_end_sec",
            "source_duration_sec",
            "source_score",
            "source_lexical_anchor",
            "source_line_number",
            "category",
            "intensity_code",
            "intensity",
        )
    }


def _discard_for_span(
    task: dict[str, Any], span: dict[str, Any], reason: str, detail: str
) -> dict[str, Any]:
    return {
        "discard_scope": "official_semantic_span",
        "dataset": "BEAT2",
        "dataset_subset": DATASET_SUBSET,
        "source_clip_id": task["source_clip_id"],
        "source_group_id": task["source_group_id"],
        "official_split": task["official_split"],
        "reason": reason,
        "detail": detail,
        "accepted_for_training": False,
        "official_span": _span_audit_fields(span),
    }


def _reject_source(task: dict[str, Any], reason: str, detail: str) -> dict[str, Any]:
    return {
        "binding": task["binding"],
        "records": [],
        "discards": [
            {
                "discard_scope": "source",
                "dataset": "BEAT2",
                "dataset_subset": DATASET_SUBSET,
                "source_clip_id": task["source_clip_id"],
                "source_group_id": task["source_group_id"],
                "official_split": task["official_split"],
                "reason": reason,
                "detail": detail,
                "accepted_for_training": False,
            }
        ],
        "source_stats": {
            "source_frame_count": 0,
            "official_span_count": 0,
            "semantic_event_span_count": 0,
            "output_candidate_count": 0,
        },
    }


def _overlap(a: dict[str, Any], b: dict[str, Any]) -> float:
    return max(
        0.0,
        min(float(a["source_end_sec"]), float(b["source_end_sec"]))
        - max(float(a["source_start_sec"]), float(b["source_start_sec"])),
    )


def _nearest_barrier_frames(
    span: dict[str, Any],
    all_spans: list[dict[str, Any]],
    frame_count: int,
) -> tuple[int, int, dict[str, Any] | None, dict[str, Any] | None]:
    relevant = [
        item
        for item in all_spans
        if item is not span
        and (
            item["is_training_semantic_event"]
            or item["source_label"] != "01_beat_align"
        )
    ]
    previous = [
        item
        for item in relevant
        if float(item["source_end_sec"])
        <= float(span["source_start_sec"]) + TIME_TOLERANCE_SEC
    ]
    following = [
        item
        for item in relevant
        if float(item["source_start_sec"])
        >= float(span["source_end_sec"]) - TIME_TOLERANCE_SEC
    ]
    previous_span = max(previous, key=lambda item: item["source_end_sec"], default=None)
    following_span = min(following, key=lambda item: item["source_start_sec"], default=None)
    lower = 0
    upper = frame_count
    if previous_span is not None:
        lower = min(
            frame_count,
            int(math.ceil(float(previous_span["source_end_sec"]) * FPS - 1e-9)),
        )
    if following_span is not None:
        upper = max(
            0,
            int(math.floor(float(following_span["source_start_sec"]) * FPS + 1e-9)),
        )
    return lower, upper, previous_span, following_span


def _process_source(task: dict[str, Any]) -> dict[str, Any]:
    root = Path(task["root"])
    motion_path = root / task["motion_relpath"]
    textgrid_path = root / task["textgrid_relpath"]
    sem_path = root / task["annotation_relpath"]
    if task["official_split"] is None:
        return _reject_source(task, "missing_official_split", "No matching split CSV row")
    for name, path in (
        ("motion", motion_path),
        ("textgrid", textgrid_path),
        ("official_semantic_annotation", sem_path),
    ):
        if not path.is_file():
            return _reject_source(task, f"missing_{name}", relative_path(path, root))
    try:
        emotion = parse_filename_emotion(task["source_clip_id"])
    except ValueError as error:
        return _reject_source(task, "invalid_or_ambiguous_filename_emotion", str(error))
    try:
        poses, _trans, fps = validate_motion_npz(motion_path)
    except (OSError, ValueError) as error:
        return _reject_source(task, "invalid_motion_npz", str(error))
    try:
        intervals = parse_words_textgrid_intervals(textgrid_path)
    except (OSError, UnicodeError, ValueError) as error:
        return _reject_source(task, "invalid_textgrid", str(error))
    try:
        spans = parse_semantic_spans(sem_path)
    except (OSError, UnicodeError, ValueError) as error:
        return _reject_source(task, "invalid_official_semantic_annotation", str(error))

    source_frame_count = int(poses.shape[0])
    source_duration_sec = source_frame_count / fps
    textgrid_start_sec = min(start for start, _end, _text in intervals)
    textgrid_end_sec = max(end for _start, end, _text in intervals)
    upper_speed = joint_angular_speed(poses, UPPER_BODY_JOINT_INDICES)
    head_speed = joint_angular_speed(poses, HEAD_NECK_JOINT_INDICES)
    combined_energy = 0.8 * np.mean(upper_speed, axis=1) + 0.2 * np.mean(
        head_speed, axis=1
    )
    speaker = parse_speaker(task["source_clip_id"])
    source_hashes = {
        "motion_sha256": sha256_file(motion_path),
        "textgrid_sha256": sha256_file(textgrid_path),
        "annotation_sha256": sha256_file(sem_path),
    }
    records: list[dict[str, Any]] = []
    discards: list[dict[str, Any]] = []
    events = [span for span in spans if span["is_training_semantic_event"]]

    for span in spans:
        if span["is_training_semantic_event"]:
            continue
        reason = (
            "official_control_label_denied_for_training"
            if span["source_label"] in CONTROL_LABELS
            else "unsupported_official_semantic_label"
        )
        discards.append(
            _discard_for_span(
                task,
                span,
                reason,
                "Audited source span is not a deictic/iconic/metaphoric event (02-10)",
            )
        )

    for span in events:
        conflicts = [
            other for other in events if other is not span and _overlap(span, other)
        ]
        if conflicts:
            discards.append(
                _discard_for_span(
                    task,
                    span,
                    "ambiguous_overlapping_semantic_events",
                    "Overlaps official event lines "
                    + ",".join(str(item["source_line_number"]) for item in conflicts),
                )
            )
            continue
        denied_control_conflicts = [
            other
            for other in spans
            if other is not span
            and not other["is_training_semantic_event"]
            and other["source_label"] != "01_beat_align"
            and _overlap(span, other)
        ]
        if denied_control_conflicts:
            discards.append(
                _discard_for_span(
                    task,
                    span,
                    "semantic_event_overlaps_denied_or_unknown_span",
                    "Overlaps denied official lines "
                    + ",".join(
                        str(item["source_line_number"])
                        for item in denied_control_conflicts
                    ),
                )
            )
            continue
        core_start, core_end = _aligned_core_frames(
            float(span["source_start_sec"]),
            float(span["source_end_sec"]),
            source_frame_count,
        )
        if (
            float(span["source_end_sec"]) > source_duration_sec + TIME_TOLERANCE_SEC
            or core_end - core_start < 2
        ):
            discards.append(
                _discard_for_span(
                    task,
                    span,
                    "semantic_event_outside_motion_or_too_short",
                    f"aligned_core=[{core_start},{core_end}), source_frames={source_frame_count}",
                )
            )
            continue
        if not (
            float(span["source_start_sec"]) + TIME_TOLERANCE_SEC >= textgrid_start_sec
            and float(span["source_end_sec"])
            <= textgrid_end_sec + TIME_TOLERANCE_SEC
        ):
            discards.append(
                _discard_for_span(
                    task,
                    span,
                    "semantic_event_outside_textgrid",
                    f"textgrid=[{textgrid_start_sec},{textgrid_end_sec}]",
                )
            )
            continue

        barrier_lower, barrier_upper, previous_span, following_span = (
            _nearest_barrier_frames(span, spans, source_frame_count)
        )
        if core_start < barrier_lower or core_end > barrier_upper:
            discards.append(
                _discard_for_span(
                    task,
                    span,
                    "ambiguous_after_30hz_alignment",
                    f"core=[{core_start},{core_end}), barriers=[{barrier_lower},{barrier_upper}]",
                )
            )
            continue
        max_context_frames = int(round(float(task["max_context_sec"]) * fps))
        left_lower = max(barrier_lower, core_start - max_context_frames)
        right_upper = min(barrier_upper, core_end + max_context_frames)
        start_frame, left_energy = _choose_low_speed_boundary(
            combined_energy,
            left_lower,
            core_start,
            core_start,
            source_frame_count,
        )
        end_frame, right_energy = _choose_low_speed_boundary(
            combined_energy,
            core_end,
            right_upper,
            core_end,
            source_frame_count,
        )
        if end_frame - start_frame < 2:
            discards.append(
                _discard_for_span(
                    task,
                    span,
                    "adaptive_boundary_too_short",
                    f"training_segment=[{start_frame},{end_frame})",
                )
            )
            continue

        start_sec = start_frame / fps
        end_sec = end_frame / fps
        frame_count = end_frame - start_frame
        textgrid_units = overlapping_textgrid_units(intervals, start_sec, end_sec)
        speech_context = aligned_transcript_context(intervals, start_sec, end_sec)
        metrics = _window_metrics(upper_speed, head_speed, start_frame, frame_count)
        metrics = {
            key: round(value, 8) if isinstance(value, float) else value
            for key, value in metrics.items()
        }
        raw_event_id = (
            f"{task['source_clip_id']}_sem{span['source_line_number']:04d}_"
            f"f{start_frame:06d}-{end_frame:06d}"
        )
        clip_id = f"{DATASET_SUBSET}__{raw_event_id}"
        event_fields = _span_audit_fields(span)
        official_span = {
            **event_fields,
            "window_start_sec": round(float(span["source_start_sec"]) - start_sec, 6),
            "window_end_sec": round(float(span["source_end_sec"]) - start_sec, 6),
        }
        boundary_source = {
            "mode": "official_sem_core_plus_motion_low_speed_context",
            "official_annotation_required": True,
            "official_core_start_sec": span["source_start_sec"],
            "official_core_end_sec": span["source_end_sec"],
            "official_core_start_frame_floor": core_start,
            "official_core_end_frame_exclusive_ceil": core_end,
            "max_context_sec_per_side": float(task["max_context_sec"]),
            "left_motion_boundary_energy_rad_s": round(left_energy, 8),
            "right_motion_boundary_energy_rad_s": round(right_energy, 8),
            "previous_barrier_source_label": (
                previous_span["source_label"] if previous_span else None
            ),
            "previous_barrier_end_frame": barrier_lower if previous_span else None,
            "following_barrier_source_label": (
                following_span["source_label"] if following_span else None
            ),
            "following_barrier_start_frame": barrier_upper if following_span else None,
            "neighbor_crossing_allowed": False,
        }
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset": "BEAT2",
                "dataset_subset": DATASET_SUBSET,
                "language": "english",
                "language_code": "en",
                "clip_id": clip_id,
                "window_id": clip_id,
                "task_id": clip_id,
                "source_clip_id": task["source_clip_id"],
                "source_group_id": task["source_group_id"],
                "source_group_key": task["source_group_id"],
                "split_group_id": task["source_group_id"],
                "source_group_policy": "all_events_from_source_clip_share_official_split",
                **speaker,
                "official_split": task["official_split"],
                "interaction_label": INTERACTION_LABEL,
                "interaction_label_source": "dataset_scope_only",
                "interaction_scope": "human_co_speech_interaction",
                "interaction_scope_status": (
                    "source_context_only_pending_robot_observability_qc"
                ),
                "behavior_id": None,
                **emotion,
                "semantic_event": event_fields,
                "official_gesture_semantic_spans": [official_span],
                "official_gesture_semantic_role": (
                    "official_event_label_not_emotion_lexical_anchor_not_emotion"
                ),
                "semantic_label_status": (
                    "official_semantic_event_preserved_pending_robot_retarget_qc"
                ),
                "semantic_mapping_status": "unmapped_pending_retarget_qc",
                "window_transcript_context": speech_context,
                "window_transcript_role": (
                    "time_aligned_speech_context_only_not_action_or_emotion_label"
                ),
                "speech_context_available": bool(speech_context),
                "textgrid_units": textgrid_units,
                "motion_relpath": task["motion_relpath"],
                "textgrid_relpath": task["textgrid_relpath"],
                "annotation_relpath": task["annotation_relpath"],
                "annotation_kind": "official_gesture_semantic_event",
                "audio_enabled": False,
                "audio_policy": "disabled_not_read_not_required",
                "fps": fps,
                "window": {
                    "selection_status": SELECTION_STATUS,
                    "start_frame": start_frame,
                    "end_frame_exclusive": end_frame,
                    "frame_count": frame_count,
                    "start_sec": round(start_sec, 6),
                    "end_sec": round(end_sec, 6),
                    "duration_sec": round(frame_count / fps, 6),
                    **{
                        key: value
                        for key, value in metrics.items()
                        if key not in {"start_frame", "frame_count"}
                    },
                },
                "training_segment": {
                    "representation": REPRESENTATION,
                    "fixed_window_sec": None,
                    "start_frame": start_frame,
                    "end_frame_exclusive": end_frame,
                    "frame_count": frame_count,
                    "boundary_source": boundary_source,
                },
                "accepted_for_training": False,
                "training_admission_status": "pending_retarget_qc",
                "issues": [],
                **source_hashes,
            }
        )

    return {
        "binding": task["binding"],
        "records": records,
        "discards": discards,
        "source_stats": {
            "source_frame_count": source_frame_count,
            "official_span_count": len(spans),
            "semantic_event_span_count": len(events),
            "output_candidate_count": len(records),
        },
    }


def _source_binding(
    root: Path,
    motion_path: Path,
    textgrid_path: Path,
    sem_path: Path,
    official_split: str | None,
    max_context_sec: float,
    pipeline_sha256: str,
) -> dict[str, Any]:
    return {
        "state_schema_version": STATE_SCHEMA_VERSION,
        "pipeline_schema_version": SCHEMA_VERSION,
        "dataset_subset": DATASET_SUBSET,
        "official_split": official_split,
        "max_context_sec": max_context_sec,
        "representation": REPRESENTATION,
        "selection_status": SELECTION_STATUS,
        "pipeline_implementation_sha256": pipeline_sha256,
        "audio_enabled": False,
        "files": {
            "motion": file_stat_binding(motion_path, root),
            "textgrid": file_stat_binding(textgrid_path, root),
            "annotation": file_stat_binding(sem_path, root),
        },
    }


def _state_path(state_dir: Path, source_clip_id: str) -> Path:
    digest = hashlib.sha256(source_clip_id.encode("utf-8")).hexdigest()[:20]
    return state_dir / f"{digest}.json"


def _load_reusable_state(path: Path, binding: dict[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if state.get("binding") != binding or not isinstance(state.get("result"), dict):
        return None
    return state["result"]


def _discover_tasks(
    root: Path,
    splits: dict[str, str],
    max_context_sec: float,
    pipeline_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    subset_root = root / DATASET_SUBSET
    directories = {
        "motion": subset_root / "smplxflame_30",
        "textgrid": subset_root / "textgrid",
        "annotation": subset_root / "sem",
    }
    ids_by_kind = {
        "motion": {path.stem for path in directories["motion"].glob("*.npz")},
        "textgrid": {path.stem for path in directories["textgrid"].glob("*.TextGrid")},
        "annotation": {path.stem for path in directories["annotation"].glob("*.txt")},
        "official_split": set(splits),
    }
    source_ids = sorted(set().union(*ids_by_kind.values()))
    tasks = []
    for clip_id in source_ids:
        motion_path = directories["motion"] / f"{clip_id}.npz"
        textgrid_path = directories["textgrid"] / f"{clip_id}.TextGrid"
        sem_path = directories["annotation"] / f"{clip_id}.txt"
        source_group_id = f"BEAT2/{DATASET_SUBSET}/{clip_id}"
        binding = _source_binding(
            root,
            motion_path,
            textgrid_path,
            sem_path,
            splits.get(clip_id),
            max_context_sec,
            pipeline_sha256,
        )
        tasks.append(
            {
                "root": str(root),
                "source_clip_id": clip_id,
                "source_group_id": source_group_id,
                "official_split": splits.get(clip_id),
                "motion_relpath": relative_path(motion_path, root),
                "textgrid_relpath": relative_path(textgrid_path, root),
                "annotation_relpath": relative_path(sem_path, root),
                "max_context_sec": max_context_sec,
                "binding": binding,
            }
        )
    return tasks, {
        "source_ids_by_component": {
            key: len(value) for key, value in sorted(ids_by_kind.items())
        },
        "source_id_union_count": len(source_ids),
    }


def estimate_sem_directory(sem_dir: Path) -> dict[str, Any]:
    label_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    emotion_counts: Counter[str] = Counter()
    network_supported = 0
    source_failures: Counter[str] = Counter()
    source_count = 0
    for path in sorted(sem_dir.glob("*.txt")):
        source_count += 1
        try:
            emotion = parse_filename_emotion(path.stem)
            spans = parse_semantic_spans(path)
        except (OSError, UnicodeError, ValueError) as error:
            source_failures[str(error)] += 1
            continue
        emotion_counts[str(emotion["source_emotion_label"])] += sum(
            span["is_training_semantic_event"] for span in spans
        )
        for span in spans:
            label_counts[str(span["source_label"])] += 1
            if span["is_training_semantic_event"]:
                category_counts[str(span["category"])] += 1
                if emotion["emotion_supervision_mask"]:
                    network_supported += 1
    semantic_count = sum(category_counts.values())
    return {
        "source_annotation_file_count": source_count,
        "semantic_event_candidate_count_before_motion_qc": semantic_count,
        "network_emotion_supported_candidate_count_before_motion_qc": network_supported,
        "label_counts": dict(sorted(label_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "candidate_counts_by_source_emotion": dict(sorted(emotion_counts.items())),
        "source_failure_counts": dict(sorted(source_failures.items())),
    }


def build_inventory(
    root: Path,
    output_dir: Path,
    *,
    workers: int = 4,
    max_context_sec: float = DEFAULT_MAX_CONTEXT_SEC,
    revalidate: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    output_dir = output_dir.resolve()
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if not math.isfinite(max_context_sec) or max_context_sec < 0:
        raise ValueError("max_context_sec must be finite and >= 0")
    subset_root = root / DATASET_SUBSET
    split_path = subset_root / "train_test_split.csv"
    splits = read_official_splits(split_path)
    pipeline_sha256 = sha256_file(Path(__file__).resolve())
    tasks, discovery = _discover_tasks(
        root, splits, max_context_sec, pipeline_sha256
    )
    if not tasks:
        raise ValueError(f"No BEAT2 English source components found under {subset_root}")

    state_dir = output_dir / ".state"
    results_by_clip: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    reused = 0
    for task in tasks:
        state_path = _state_path(state_dir, task["source_clip_id"])
        result = None if revalidate else _load_reusable_state(state_path, task["binding"])
        if result is None:
            pending.append(task)
        else:
            results_by_clip[task["source_clip_id"]] = result
            reused += 1

    def store(task: dict[str, Any], result: dict[str, Any]) -> None:
        results_by_clip[task["source_clip_id"]] = result
        atomic_json(
            _state_path(state_dir, task["source_clip_id"]),
            {
                "state_schema_version": STATE_SCHEMA_VERSION,
                "created_at_utc": utc_now(),
                "source_clip_id": task["source_clip_id"],
                "binding": task["binding"],
                "result": result,
            },
        )

    if workers == 1:
        for task in pending:
            store(task, _process_source(task))
    elif pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_source, task): task for task in pending}
            for future in as_completed(futures):
                store(futures[future], future.result())

    results = [results_by_clip[task["source_clip_id"]] for task in tasks]
    records = sorted(
        (record for result in results for record in result["records"]),
        key=lambda record: (
            record["source_clip_id"],
            record["semantic_event"]["source_line_number"],
        ),
    )
    supported = [record for record in records if record["emotion_supervision_mask"]]
    discards = sorted(
        (record for result in results for record in result["discards"]),
        key=lambda record: (
            record["source_clip_id"],
            (record.get("official_span") or {}).get("source_line_number", -1),
            record["reason"],
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / f"{OUTPUT_STEM}.full_candidates.jsonl"
    supported_path = output_dir / f"{OUTPUT_STEM}.network_emotion_supported.jsonl"
    discarded_path = output_dir / f"{OUTPUT_STEM}.discarded.jsonl"
    output_hashes = {
        full_path.name: atomic_jsonl(full_path, records),
        supported_path.name: atomic_jsonl(supported_path, supported),
        discarded_path.name: atomic_jsonl(discarded_path, discards),
    }

    candidate_by_label = Counter(
        str(record["semantic_event"]["source_label"]) for record in records
    )
    candidate_by_category = Counter(
        str(record["semantic_event"]["category"]) for record in records
    )
    candidate_by_emotion = Counter(
        str(record["source_emotion_label"]) for record in records
    )
    candidate_by_split = Counter(str(record["official_split"]) for record in records)
    discard_reasons = Counter(str(record["reason"]) for record in discards)
    frame_counts = [int(record["window"]["frame_count"]) for record in records]
    source_failure_count = sum(
        any(discard["discard_scope"] == "source" for discard in result["discards"])
        for result in results
    )
    static_estimate = estimate_sem_directory(subset_root / "sem")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "beat2_variable_length_semantic_event_inventory",
        "created_at_utc": utc_now(),
        "root": str(root),
        "dataset_subset": DATASET_SUBSET,
        "source_clip_union_count": len(tasks),
        "source_clip_failure_count": source_failure_count,
        "candidate_count": len(records),
        "network_emotion_supported_candidate_count": len(supported),
        "candidate_frame_count": sum(frame_counts),
        "candidate_duration_hours": round(sum(frame_counts) / FPS / 3600.0, 6),
        "candidate_frame_count_min": min(frame_counts) if frame_counts else None,
        "candidate_frame_count_max": max(frame_counts) if frame_counts else None,
        "distinct_candidate_frame_count_count": len(set(frame_counts)),
        "candidate_counts_by_source_label": dict(sorted(candidate_by_label.items())),
        "candidate_counts_by_category": dict(sorted(candidate_by_category.items())),
        "candidate_counts_by_source_emotion": dict(sorted(candidate_by_emotion.items())),
        "candidate_counts_by_official_split": dict(sorted(candidate_by_split.items())),
        "discard_counts_by_reason": dict(sorted(discard_reasons.items())),
        "static_sem_estimate_before_motion_qc": static_estimate,
        "discovery": discovery,
        "resume": {
            "source_state_reused_count": reused,
            "source_state_computed_count": len(pending),
        },
        "segment_policy": {
            "representation": REPRESENTATION,
            "fixed_window_sec": None,
            "fps": FPS,
            "selection_status": SELECTION_STATUS,
            "official_training_event_labels": "02-10 deictic/iconic/metaphoric only",
            "max_context_sec_per_side": max_context_sec,
            "adaptive_context": "motion_low_speed_boundary",
            "neighbor_crossing_allowed": False,
        },
        "emotion_policy": {
            "source": "official_filename_protocol",
            "network_labels": sorted(NETWORK_EMOTIONS),
            "unsupported_preserved_without_supervision": ["contempt", "disgust"],
            "ambiguous_cross_protocol_range": "fail_closed",
            "lexical_anchor_role": "semantic_anchor_only_never_emotion",
        },
        "audio_policy": {
            "enabled": False,
            "audio_files_read": False,
            "audio_files_required": False,
        },
        "training_admission": "false_pending_retarget_qc",
        "output_sha256": output_hashes,
    }
    summary_path = output_dir / f"{OUTPUT_STEM}.summary.json"
    atomic_json(summary_path, summary)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "pipeline_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "source_binding_sha256": sha256_bytes(
            stable_json([task["binding"] for task in tasks]).encode("utf-8")
        ),
        "source_binding_count": len(tasks),
        "official_split": {
            "path": relative_path(split_path, root),
            "sha256": sha256_file(split_path),
        },
        "output_sha256": {
            **output_hashes,
            summary_path.name: sha256_file(summary_path),
        },
        "contract": {
            "representation": REPRESENTATION,
            "fixed_window_sec": None,
            "selection_status": SELECTION_STATUS,
            "accepted_for_training": False,
            "audio_enabled": False,
        },
    }
    atomic_json(output_dir / f"{OUTPUT_STEM}.provenance.json", provenance)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.estimate_sem_only:
        estimate = estimate_sem_directory(
            args.root.resolve() / DATASET_SUBSET / "sem"
        )
        print(json.dumps(estimate, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    summary = build_inventory(
        args.root,
        args.output_dir,
        workers=args.workers,
        max_context_sec=args.max_context_sec,
        revalidate=args.revalidate,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
