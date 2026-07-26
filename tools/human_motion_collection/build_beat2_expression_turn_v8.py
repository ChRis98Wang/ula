#!/usr/bin/env python3
"""Build variable-length BEAT2 expression-turn candidates.

An expression turn is bounded by natural low-speed motion, not by a fixed
window or a time-gap threshold. Adjacent official gesture events are merged
unless a hard annotation barrier or natural-rest basin separates them. Motion
energy supplies candidate evidence only; independent blind video review must
confirm the onset/apex/offset arc and semantics. Official category and clip
emotion remain metadata-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from tools.human_motion_collection.build_beat2_interaction_inventory import (
        HEAD_NECK_JOINT_INDICES,
        UPPER_BODY_JOINT_INDICES,
        _window_metrics,
        aligned_transcript_context,
        joint_angular_speed,
        validate_motion_npz,
    )
    from tools.human_motion_collection.build_beat2_semantic_event_inventory import (
        FPS,
        _aligned_core_frames,
        _boundary_energy,
        parse_semantic_spans,
        parse_words_textgrid_intervals,
    )
except ModuleNotFoundError:  # pragma: no cover - direct invocation by path
    from build_beat2_interaction_inventory import (
        HEAD_NECK_JOINT_INDICES,
        UPPER_BODY_JOINT_INDICES,
        _window_metrics,
        aligned_transcript_context,
        joint_angular_speed,
        validate_motion_npz,
    )
    from build_beat2_semantic_event_inventory import (
        FPS,
        _aligned_core_frames,
        _boundary_energy,
        parse_semantic_spans,
        parse_words_textgrid_intervals,
    )


DEFAULT_INPUT = Path(
    "/home/gez/nas/cloud/gez/human_motion/catalog/"
    "beat2_semantic_event_inventory_v1/"
    "beat2_semantic_event_inventory_v1.network_emotion_supported.jsonl"
)
DEFAULT_ROOT = Path("/home/gez/nas/cloud/gez/human_motion/raw/BEAT2")
DEFAULT_SPLITS = Path(
    "/home/gez/nas/cloud/gez/human_motion/catalog/"
    "beat2_semantic_event_pilot_v7_full/"
    "beat2_semantic_event_pilot_v7_full.split_assignments.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/gez/nas/cloud/gez/human_motion/catalog/beat2_expression_turn_v8"
)
OUTPUT_STEM = "beat2_expression_turn_v8"
SCHEMA_VERSION = "1.0.0"
REPRESENTATION = "native_variable_length_expression_turn_v1"
CONTEXT_POLICY = "evidence_anchored_progressive_expansion_no_fixed_duration_v1"
NETWORK_EMOTIONS = ("neutral", "sad", "happy", "angry", "surprise", "fear")
SEMANTIC_CATEGORIES = ("deictic", "iconic", "metaphoric")
SPLITS = ("train", "validation", "test")
STRESS_SPLIT_COUNTS = {"train": 70, "validation": 15, "test": 15}
# Compatibility name for callers of the pre-review producer API. Formal output
# names the set stress100 and never treats it as a pool acceptance estimator.
PILOT_SPLIT_COUNTS = STRESS_SPLIT_COUNTS
HARD_BARRIER_LABELS = {"00_nogesture", "habit", "need_cut"}
SEMANTIC_MASKS = {
    "official_category": False,
    "robot_observable_motion_form": False,
    "communicative_intent": False,
    "prompt_text": False,
    "legacy_gesture": False,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--beat2-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split-assignments", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pilot-count", type=int, default=100)
    parser.add_argument("--rest-energy-max", type=float, default=0.14)
    parser.add_argument("--rest-radius-frames", type=int, default=2)
    parser.add_argument("--max-energy-p95", type=float, default=4.0)
    parser.add_argument("--limit-sources", type=int)
    return parser.parse_args(argv)


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


def record_sha256(record: dict[str, Any]) -> str:
    return sha256_bytes(stable_json(record).encode("utf-8"))


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    atomic_bytes(path, payload)
    return sha256_bytes(payload)


def atomic_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> str:
    payload = "".join(stable_json(record) + "\n" for record in records).encode("utf-8")
    atomic_bytes(path, payload)
    return sha256_bytes(payload)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            value["_expression_turn_input_line"] = line_number
            value["_expression_turn_input_record_sha256"] = record_sha256(value)
            # The line number is transport metadata and must not change lineage.
            value["_expression_turn_input_record_sha256"] = record_sha256(
                {
                    key: item
                    for key, item in value.items()
                    if not key.startswith("_expression_turn_")
                }
            )
            records.append(value)
    if not records:
        raise ValueError(f"input inventory is empty: {path}")
    return records


def load_split_assignment(path: Path) -> tuple[dict[str, str], str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    mapping = value.get("speaker_to_split") if isinstance(value, dict) else None
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("split assignment requires speaker_to_split")
    cleaned = {str(key): str(split) for key, split in mapping.items()}
    if any(split not in SPLITS for split in cleaned.values()):
        raise ValueError("split assignment contains an unsupported split")
    expected = sha256_bytes(stable_json(cleaned).encode("utf-8"))
    if value.get("sha256") != expected:
        raise ValueError("split assignment self-hash mismatch")
    return cleaned, sha256_file(path)


def combined_energy(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    upper_speed = joint_angular_speed(poses, UPPER_BODY_JOINT_INDICES)
    head_speed = joint_angular_speed(poses, HEAD_NECK_JOINT_INDICES)
    energy = 0.8 * np.mean(upper_speed, axis=1) + 0.2 * np.mean(head_speed, axis=1)
    return upper_speed, head_speed, energy


def boundary_rest_score(
    energy: np.ndarray, frame: int, frame_count: int, radius: int
) -> float:
    if radius < 0:
        raise ValueError("rest radius cannot be negative")
    if energy.size != max(0, frame_count - 1):
        raise ValueError("energy/frame count mismatch")
    center = min(max(frame, 1), frame_count - 1) - 1
    lower = max(0, center - radius)
    upper = min(energy.size, center + radius + 1)
    return float(np.mean(energy[lower:upper]))


def rest_boundaries(
    energy: np.ndarray,
    lower: int,
    upper: int,
    *,
    frame_count: int,
    rest_energy_max: float,
    radius: int,
) -> list[int]:
    lower = max(0, int(lower))
    upper = min(frame_count, int(upper))
    if lower > upper:
        return []
    return [
        frame
        for frame in range(lower, upper + 1)
        if boundary_rest_score(energy, frame, frame_count, radius)
        <= rest_energy_max
    ]


def _hard_barrier_between(
    first_end_sec: float, second_start_sec: float, spans: list[dict[str, Any]]
) -> bool:
    for span in spans:
        if span.get("source_label") not in HARD_BARRIER_LABELS:
            continue
        start = float(span["source_start_sec"])
        end = float(span["source_end_sec"])
        if end > first_end_sec + 1e-6 and start < second_start_sec - 1e-6:
            return True
    return False


def cluster_adjacent_events(
    events: list[dict[str, Any]],
    spans: list[dict[str, Any]],
    energy: np.ndarray,
    *,
    frame_count: int,
    fps: float,
    rest_energy_max: float,
    rest_radius_frames: int,
) -> list[list[dict[str, Any]]]:
    ordered = sorted(
        events,
        key=lambda record: (
            float(record["semantic_event"]["source_start_sec"]),
            float(record["semantic_event"]["source_end_sec"]),
            str(record["clip_id"]),
        ),
    )
    if not ordered:
        return []
    groups = [[ordered[0]]]
    for current in ordered[1:]:
        previous = groups[-1][-1]
        previous_event = previous["semantic_event"]
        current_event = current["semantic_event"]
        previous_end = float(previous_event["source_end_sec"])
        current_start = float(current_event["source_start_sec"])
        gap_sec = current_start - previous_end
        gap_start = min(frame_count, int(math.ceil(previous_end * fps - 1e-9)))
        gap_end = max(0, int(math.floor(current_start * fps + 1e-9)))
        gap_has_rest = bool(
            rest_boundaries(
                energy,
                gap_start,
                gap_end,
                frame_count=frame_count,
                rest_energy_max=rest_energy_max,
                radius=rest_radius_frames,
            )
        )
        merge = (
            gap_sec >= -1e-6
            and not _hard_barrier_between(previous_end, current_start, spans)
            and not gap_has_rest
        )
        if merge:
            groups[-1].append(current)
        else:
            groups.append([current])
    return groups


def _rest_basin_representatives(frames: list[int], *, side: str) -> list[int]:
    if not frames:
        return []
    basins: list[list[int]] = [[frames[0]]]
    for frame in frames[1:]:
        if frame == basins[-1][-1] + 1:
            basins[-1].append(frame)
        else:
            basins.append([frame])
    if side == "left":
        return [max(basin) for basin in reversed(basins)]
    if side == "right":
        return [min(basin) for basin in basins]
    raise ValueError("side must be left or right")


def build_context_plan(
    energy: np.ndarray,
    *,
    core_start: int,
    core_end: int,
    lower_barrier: int,
    upper_barrier: int,
    source_frame_count: int,
    fps: float,
    rest_energy_max: float,
    rest_radius_frames: int,
) -> dict[str, Any] | None:
    left_choices = _rest_basin_representatives(
        rest_boundaries(
            energy,
            lower_barrier,
            core_start,
            frame_count=source_frame_count,
            rest_energy_max=rest_energy_max,
            radius=rest_radius_frames,
        ),
        side="left",
    )
    right_choices = _rest_basin_representatives(
        rest_boundaries(
            energy,
            core_end,
            upper_barrier,
            frame_count=source_frame_count,
            rest_energy_max=rest_energy_max,
            radius=rest_radius_frames,
        ),
        side="right",
    )
    if not left_choices or not right_choices:
        return None
    levels = []
    for index in range(max(len(left_choices), len(right_choices))):
        left = left_choices[min(index, len(left_choices) - 1)]
        right = right_choices[min(index, len(right_choices) - 1)]
        if levels and (
            left == levels[-1]["start_frame"]
            and right == levels[-1]["end_frame_exclusive"]
        ):
            continue
        level = {
            "level": len(levels),
            "start_frame": left,
            "end_frame_exclusive": right,
            "left_boundary_basis": "preceding_natural_low_motion_basin",
            "right_boundary_basis": "following_natural_low_motion_basin",
            "left_rest_score_rad_s": round(
                boundary_rest_score(
                    energy, left, source_frame_count, rest_radius_frames
                ),
                8,
            ),
            "right_rest_score_rad_s": round(
                boundary_rest_score(
                    energy, right, source_frame_count, rest_radius_frames
                ),
                8,
            ),
            "source_start_sec": round(left / fps, 6),
            "source_end_sec": round(right / fps, 6),
        }
        if levels:
            level.update(
                {
                    "parent_level": len(levels) - 1,
                    "expansion_reason": "resolve_arc_or_semantic_ambiguity",
                }
            )
        levels.append(level)
    if any(
        not (
            levels[index]["start_frame"] <= levels[index - 1]["start_frame"]
            and levels[index]["end_frame_exclusive"]
            >= levels[index - 1]["end_frame_exclusive"]
            and (
                levels[index]["start_frame"] < levels[index - 1]["start_frame"]
                or levels[index]["end_frame_exclusive"]
                > levels[index - 1]["end_frame_exclusive"]
            )
        )
        for index in range(1, len(levels))
    ):
        raise ValueError("context levels are not strictly nested")
    return {
        "policy": CONTEXT_POLICY,
        "same_source_only": True,
        "neighbor_crossing_allowed": False,
        "source_interval": {
            "start_frame": 0,
            "end_frame_exclusive": source_frame_count,
        },
        "admissible_interval": {
            "start_frame": lower_barrier,
            "end_frame_exclusive": upper_barrier,
        },
        "selected_level": 0,
        "levels": levels,
        "context_exhausted_at_level": len(levels) - 1,
    }


def _nearest_outer_barriers(
    group: list[dict[str, Any]],
    spans: list[dict[str, Any]],
    frame_count: int,
    fps: float,
) -> tuple[int, int]:
    first_start = float(group[0]["semantic_event"]["source_start_sec"])
    last_end = float(group[-1]["semantic_event"]["source_end_sec"])
    relevant = [span for span in spans if span.get("source_label") != "01_beat_align"]
    previous = [
        span
        for span in relevant
        if float(span["source_end_sec"]) <= first_start + 1e-6
        and int(span.get("source_line_number") or -1)
        not in {
            int(record["semantic_event"]["source_line_number"]) for record in group
        }
    ]
    following = [
        span
        for span in relevant
        if float(span["source_start_sec"]) >= last_end - 1e-6
        and int(span.get("source_line_number") or -1)
        not in {
            int(record["semantic_event"]["source_line_number"]) for record in group
        }
    ]
    lower = 0
    upper = frame_count
    if previous:
        lower = min(
            frame_count,
            int(math.ceil(max(float(span["source_end_sec"]) for span in previous) * fps)),
        )
    if following:
        upper = min(
            frame_count,
            max(
                0,
                int(
                    math.floor(
                        min(float(span["source_start_sec"]) for span in following)
                        * fps
                    )
                ),
            ),
        )
    return lower, upper


def _timeline(start_frame: int, end_frame: int, fps: float, *, relative: bool) -> dict[str, Any]:
    frame_count = end_frame - start_frame
    first = 0 if relative else start_frame
    last_exclusive = frame_count if relative else end_frame
    return {
        "frame_axis": "turn_relative" if relative else "source_absolute",
        "start_frame": first,
        "end_frame_exclusive": last_exclusive,
        "frame_count": frame_count,
        "frame_coverage_start_sec": round(first / fps, 6),
        "frame_coverage_end_sec": round(last_exclusive / fps, 6),
        "frame_coverage_sec": round(frame_count / fps, 6),
        "sample_start_sec": round(first / fps, 6),
        "sample_end_sec": round((last_exclusive - 1) / fps, 6),
        "sample_span_sec": round(max(0, frame_count - 1) / fps, 6),
    }


def _dominant_category(group: list[dict[str, Any]]) -> str:
    duration = Counter()
    for record in group:
        event = record["semantic_event"]
        duration[str(event["category"])] += float(event["source_duration_sec"])
    return min(SEMANTIC_CATEGORIES, key=lambda name: (-duration[name], name))


def _event_span_projection(
    record: dict[str, Any], start_frame: int, fps: float, source_frame_count: int
) -> dict[str, Any]:
    event = record["semantic_event"]
    core_start, core_end = _aligned_core_frames(
        float(event["source_start_sec"]),
        float(event["source_end_sec"]),
        source_frame_count,
    )
    return {
        "upstream_clip_id": record["clip_id"],
        "upstream_inventory_record_sha256": record[
            "_expression_turn_input_record_sha256"
        ],
        "source_label": event["source_label"],
        "category": event["category"],
        "intensity": event["intensity"],
        "intensity_code": event.get("intensity_code"),
        "source_score": event.get("source_score"),
        "source_lexical_anchor": event.get("source_lexical_anchor"),
        "source_line_number": event["source_line_number"],
        "source_time_axis": {
            "start_sec": event["source_start_sec"],
            "end_sec": event["source_end_sec"],
            "start_frame_floor": core_start,
            "end_frame_exclusive_ceil": core_end,
        },
        "turn_time_axis": {
            "start_sec": round(float(event["source_start_sec"]) - start_frame / fps, 6),
            "end_sec": round(float(event["source_end_sec"]) - start_frame / fps, 6),
            "start_frame": core_start - start_frame,
            "end_frame_exclusive": core_end - start_frame,
        },
    }


def _reject(group: list[dict[str, Any]], source: dict[str, Any], reason: str, detail: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "beat2_expression_turn_v8_rejection",
        "source_clip_id": source["source_clip_id"],
        "source_group_key": source["source_group_key"],
        "speaker_key": source["speaker_key"],
        "included_upstream_clip_ids": [record["clip_id"] for record in group],
        "included_source_line_numbers": [
            record["semantic_event"]["source_line_number"] for record in group
        ],
        "reason": reason,
        "detail": detail,
        "accepted_for_training": False,
    }


def build_turn_for_group(
    group: list[dict[str, Any]],
    *,
    source: dict[str, Any],
    spans: list[dict[str, Any]],
    intervals: list[tuple[float, float, str]],
    upper_speed: np.ndarray,
    head_speed: np.ndarray,
    energy: np.ndarray,
    source_frame_count: int,
    fps: float,
    split: str,
    contract_sha256: str,
    source_manifest_sha256: str,
    split_manifest_sha256: str,
    rest_energy_max: float,
    rest_radius_frames: int,
    max_energy_p95: float,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    core_start, _ = _aligned_core_frames(
        float(group[0]["semantic_event"]["source_start_sec"]),
        float(group[0]["semantic_event"]["source_end_sec"]),
        source_frame_count,
    )
    _, core_end = _aligned_core_frames(
        float(group[-1]["semantic_event"]["source_start_sec"]),
        float(group[-1]["semantic_event"]["source_end_sec"]),
        source_frame_count,
    )
    lower_barrier, upper_barrier = _nearest_outer_barriers(
        group, spans, source_frame_count, fps
    )
    context_plan = build_context_plan(
        energy,
        core_start=core_start,
        core_end=core_end,
        lower_barrier=lower_barrier,
        upper_barrier=upper_barrier,
        source_frame_count=source_frame_count,
        fps=fps,
        rest_energy_max=rest_energy_max,
        rest_radius_frames=rest_radius_frames,
    )
    if context_plan is None:
        left_exists = bool(
            rest_boundaries(
                energy,
                lower_barrier,
                core_start,
                frame_count=source_frame_count,
                rest_energy_max=rest_energy_max,
                radius=rest_radius_frames,
            )
        )
        return None, _reject(
            group,
            source,
            (
                "missing_natural_recovery_boundary"
                if left_exists
                else "missing_natural_onset_boundary"
            ),
            f"admissible=[{lower_barrier},{upper_barrier}], core=[{core_start},{core_end}]",
        )
    selected_context = context_plan["levels"][context_plan["selected_level"]]
    start_frame = int(selected_context["start_frame"])
    end_frame = int(selected_context["end_frame_exclusive"])
    left = {
        "frame": start_frame,
        "source_sec": selected_context["source_start_sec"],
        "rest_score_rad_s": selected_context["left_rest_score_rad_s"],
        "context_level": context_plan["selected_level"],
    }
    right = {
        "frame": end_frame,
        "source_sec": selected_context["source_end_sec"],
        "rest_score_rad_s": selected_context["right_rest_score_rad_s"],
        "context_level": context_plan["selected_level"],
    }
    if end_frame - start_frame < 3:
        return None, _reject(group, source, "turn_too_short", "fewer than three frames")

    peak_slice = energy[core_start : max(core_start + 1, core_end - 1)]
    if peak_slice.size == 0:
        return None, _reject(group, source, "missing_expression_peak", "empty core energy")
    peak_offset = int(np.argmax(peak_slice))
    peak_energy = float(peak_slice[peak_offset])
    peak_frame = core_start + peak_offset + 1
    peak_prominence = peak_energy / max(
        float(left["rest_score_rad_s"]),
        float(right["rest_score_rad_s"]),
        0.05,
    )
    if not start_frame < peak_frame < end_frame - 1:
        return None, _reject(
            group,
            source,
            "incomplete_onset_peak_recovery_arc",
            f"onset={start_frame}, apex={peak_frame}, offset={end_frame-1}",
        )

    frame_count = end_frame - start_frame
    metrics = _window_metrics(upper_speed, head_speed, start_frame, frame_count)
    metrics = {
        key: round(value, 8) if isinstance(value, float) else value
        for key, value in metrics.items()
    }
    if float(metrics["interaction_energy_p95_rad_s"]) > max_energy_p95:
        return None, _reject(
            group,
            source,
            "high_dynamic_above_interaction_energy_ceiling",
            f"p95={metrics['interaction_energy_p95_rad_s']} > {max_energy_p95}",
        )
    first_line = int(group[0]["semantic_event"]["source_line_number"])
    last_line = int(group[-1]["semantic_event"]["source_line_number"])
    clip_id = (
        f"beat_english_v2.0.0__{source['source_clip_id']}_turn"
        f"{first_line:04d}-{last_line:04d}_f{start_frame:06d}-{end_frame:06d}"
    )
    included = [
        _event_span_projection(record, start_frame, fps, source_frame_count)
        for record in group
    ]
    categories = sorted({item["category"] for item in included})
    intensities = sorted({item["intensity"] for item in included})
    start_sec = start_frame / fps
    end_sec = end_frame / fps
    transcript = aligned_transcript_context(intervals, start_sec, end_sec)
    base = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "beat2_expression_turn_v8_candidate",
        "dataset": "BEAT2",
        "dataset_subset": "beat_english_v2.0.0",
        "clip_id": clip_id,
        "task_id": clip_id,
        "source_clip_id": source["source_clip_id"],
        "source_group_key": source["source_group_key"],
        "speaker_key": source["speaker_key"],
        "speaker_id": source.get("speaker_id"),
        "speaker_name": source.get("speaker_name"),
        "official_split": source.get("official_split"),
        "fixed_split_assignment": split,
        "split_grouping": "speaker_strict_source_inherits_speaker_split",
        "motion_relpath": source["motion_relpath"],
        "motion_sha256": source["motion_sha256"],
        "annotation_relpath": source["annotation_relpath"],
        "annotation_sha256": source.get("annotation_sha256"),
        "textgrid_relpath": source["textgrid_relpath"],
        "textgrid_sha256": source.get("textgrid_sha256"),
        "fps": fps,
        "representation": REPRESENTATION,
        "core_interval": {
            "start_frame": core_start,
            "end_frame_exclusive": core_end,
        },
        "context_plan": context_plan,
        "training_segment": {
            "representation": REPRESENTATION,
            "start_frame": start_frame,
            "end_frame_exclusive": end_frame,
            "frame_count": frame_count,
            "fixed_window_sec": None,
            "cropped": False,
            "duration_policy": "natural_rest_to_natural_rest_no_fixed_or_max_duration",
        },
        "time_axes": {
            "source": _timeline(start_frame, end_frame, fps, relative=False),
            "turn": _timeline(start_frame, end_frame, fps, relative=True),
        },
        "expression_turn": {
            "completeness_contract": "natural_rest_onset_peak_natural_rest_recovery_v1",
            "complete_motion_arc_verified": False,
            "automated_motion_arc_candidate_status": (
                "natural_boundary_and_apex_proxy_pending_blind_video_review"
            ),
            "included_event_count": len(included),
            "included_event_spans": included,
            "official_categories": categories,
            "official_intensities": intensities,
            "dominant_official_category": _dominant_category(group),
            "left_natural_boundary": left,
            "right_natural_boundary": right,
            "peak": {
                "source_frame": peak_frame,
                "turn_frame": peak_frame - start_frame,
                "source_sec": round(peak_frame / fps, 6),
                "turn_sec": round((peak_frame - start_frame) / fps, 6),
                "energy_rad_s": round(peak_energy, 8),
                "prominence_over_boundaries": round(peak_prominence, 8),
            },
            "onset_frame_count": peak_frame - start_frame,
            "recovery_frame_count": end_frame - peak_frame,
        },
        "window": {
            "start_frame": start_frame,
            "end_frame_exclusive": end_frame,
            "frame_count": frame_count,
            "start_sec": round(start_sec, 6),
            "end_sec": round(end_sec, 6),
            "duration_sec": round(frame_count / fps, 6),
            **metrics,
        },
        "duration_band": (
            "short_under_3s"
            if frame_count < 3 * fps
            else "medium_3_to_6s"
            if frame_count <= 6 * fps
            else "long_over_6s"
        ),
        "event_count_band": (
            "single" if len(included) == 1 else "pair" if len(included) == 2 else "multi_3plus"
        ),
        "source_speech_context": transcript,
        "source_speech_context_role": "time_aligned_context_only_not_action_or_emotion_label",
        "canonical_prompt": None,
        "canonical_prompt_role": "disabled_pending_expression_turn_video_review",
        "canonical_action": None,
        "canonical_action_role": "disabled_pending_expression_turn_video_review",
        "robot_observable_motion_form": "candidate_unreviewed",
        "communicative_intent": "candidate_unreviewed",
        "semantic_mapping_status": "official_event_sequence_metadata_only",
        "semantic_supervision_masks": dict(SEMANTIC_MASKS),
        "official_category_verified": True,
        "official_category_role": "verified_event_span_metadata_split_and_evaluation_only",
        "official_category_conditioning_enabled": False,
        "official_category_condition_channel": None,
        "official_category_loss": None,
        "emotion_id": source["emotion_id"],
        "source_emotion_label": source["source_emotion_label"],
        "source_emotion_label_verified": True,
        "emotion_label_source": "official_beat2_filename_protocol",
        "emotion_supervision_mask": False,
        "emotion_supervision_role": "disabled_pending_robot_affect_review",
        "official_emotion_conditioning_enabled": False,
        "official_emotion_condition_channel": None,
        "official_emotion_loss": None,
        "affect_observable_review_status": "candidate_unreviewed",
        "affect_observable_supervision_mask": False,
        "audio_enabled": False,
        "expression_turn_contract_sha256": contract_sha256,
        "source_inventory_manifest_sha256": source_manifest_sha256,
        "split_assignment_manifest_sha256": split_manifest_sha256,
        "upstream_event_record_sha256": [
            record["_expression_turn_input_record_sha256"] for record in group
        ],
        "training_admission_status": "pending_retarget_and_independent_video_review",
        "accepted_for_training": False,
    }
    base["expression_turn_record_sha256"] = record_sha256(base)
    return base, None


def _process_source(task: dict[str, Any]) -> dict[str, Any]:
    records = task["records"]
    source = records[0]
    root = Path(task["beat2_root"])
    motion_path = root / source["motion_relpath"]
    sem_path = root / source["annotation_relpath"]
    textgrid_path = root / source["textgrid_relpath"]
    try:
        poses, _trans, fps = validate_motion_npz(motion_path)
        if not math.isclose(float(fps), float(FPS), abs_tol=1e-6):
            raise ValueError(f"fps {fps} != {FPS}")
        spans = parse_semantic_spans(sem_path)
        intervals = parse_words_textgrid_intervals(textgrid_path)
        upper_speed, head_speed, energy = combined_energy(poses)
    except (OSError, UnicodeError, ValueError) as error:
        return {
            "candidates": [],
            "rejections": [
                _reject(records, source, "source_decode_failed", f"{type(error).__name__}: {error}")
            ],
        }
    groups = cluster_adjacent_events(
        records,
        spans,
        energy,
        frame_count=int(poses.shape[0]),
        fps=float(fps),
        rest_energy_max=task["rest_energy_max"],
        rest_radius_frames=task["rest_radius_frames"],
    )
    candidates = []
    rejections = []
    for group in groups:
        candidate, rejection = build_turn_for_group(
            group,
            source=source,
            spans=spans,
            intervals=intervals,
            upper_speed=upper_speed,
            head_speed=head_speed,
            energy=energy,
            source_frame_count=int(poses.shape[0]),
            fps=float(fps),
            split=task["split"],
            contract_sha256=task["contract_sha256"],
            source_manifest_sha256=task["source_manifest_sha256"],
            split_manifest_sha256=task["split_manifest_sha256"],
            rest_energy_max=task["rest_energy_max"],
            rest_radius_frames=task["rest_radius_frames"],
            max_energy_p95=task["max_energy_p95"],
        )
        if candidate is not None:
            candidates.append(candidate)
        if rejection is not None:
            rejections.append(rejection)
    return {"candidates": candidates, "rejections": rejections}


def _balanced_quotas(total: int, labels: tuple[str, ...], offset: int) -> dict[str, int]:
    base, remainder = divmod(total, len(labels))
    quotas = {label: base for label in labels}
    for index in range(remainder):
        quotas[labels[(offset + index) % len(labels)]] += 1
    return quotas


def _pilot_priority(record: dict[str, Any], seed: int) -> tuple[Any, ...]:
    turn = record["expression_turn"]
    return (
        0 if int(turn["included_event_count"]) >= 2 else 1,
        -int(turn["included_event_count"]),
        sha256_bytes(f"{seed}\0{record['clip_id']}".encode("utf-8")),
    )


def _attach_selection_lineage(
    records: list[dict[str, Any]], *, kind: str, status: str
) -> list[dict[str, Any]]:
    result = []
    for rank, record in enumerate(records, 1):
        value = {
            **record,
            "expression_turn_selection_kind": kind,
            "expression_turn_selection_rank": rank,
            "expression_turn_selection_status": status,
            "accepted_for_training": False,
        }
        value["expression_turn_selection_record_sha256"] = record_sha256(value)
        result.append(value)
    return result


def select_stress_set(
    candidates: list[dict[str, Any]], *, pilot_count: int, seed: int = 20260724
) -> list[dict[str, Any]]:
    if pilot_count != sum(STRESS_SPLIT_COUNTS.values()):
        raise ValueError("v8 stress-set contract currently requires exactly 100 records")
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        key = (
            str(record["fixed_split_assignment"]),
            str(record["emotion_id"]),
            str(record["expression_turn"]["dominant_official_category"]),
        )
        buckets[key].append(record)
    for bucket in buckets.values():
        bucket.sort(key=lambda record: _pilot_priority(record, seed))

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    used_sources: set[str] = set()
    for split_index, split in enumerate(SPLITS):
        emotion_quotas = _balanced_quotas(
            STRESS_SPLIT_COUNTS[split], NETWORK_EMOTIONS, split_index * 2
        )
        for emotion in NETWORK_EMOTIONS:
            category_cursor = 0
            for _ in range(emotion_quotas[emotion]):
                choice = None
                for relaxation in (False, True):
                    for offset in range(len(SEMANTIC_CATEGORIES)):
                        category = SEMANTIC_CATEGORIES[
                            (category_cursor + offset) % len(SEMANTIC_CATEGORIES)
                        ]
                        choice = next(
                            (
                                record
                                for record in buckets[(split, emotion, category)]
                                if record["clip_id"] not in selected_ids
                                and (
                                    relaxation
                                    or record["source_clip_id"] not in used_sources
                                )
                            ),
                            None,
                        )
                        if choice is not None:
                            category_cursor = (category_cursor + offset + 1) % len(
                                SEMANTIC_CATEGORIES
                            )
                            break
                    if choice is not None:
                        break
                if choice is None:
                    raise ValueError(f"unfillable v8 pilot stratum: {split}/{emotion}")
                selected_ids.add(str(choice["clip_id"]))
                used_sources.add(str(choice["source_clip_id"]))
                selected.append(choice)
    selected.sort(
        key=lambda record: (
            SPLITS.index(str(record["fixed_split_assignment"])),
            NETWORK_EMOTIONS.index(str(record["emotion_id"])),
            str(record["expression_turn"]["dominant_official_category"]),
            _pilot_priority(record, seed),
        )
    )
    return _attach_selection_lineage(
        selected,
        kind="stress100",
        status="selected_stress_pending_retarget_qc",
    )


def select_stratified_pilot(
    candidates: list[dict[str, Any]], *, pilot_count: int, seed: int = 20260724
) -> list[dict[str, Any]]:
    """Compatibility alias for the now-explicit stress review set."""

    return select_stress_set(candidates, pilot_count=pilot_count, seed=seed)


def _proportional_quotas(
    counts: dict[tuple[str, ...], int], total: int
) -> dict[tuple[str, ...], int]:
    population = sum(counts.values())
    if population <= 0 or total < 0:
        raise ValueError("invalid proportional quota population")
    raw = {key: total * count / population for key, count in counts.items()}
    quotas = {key: int(math.floor(value)) for key, value in raw.items()}
    remainder = total - sum(quotas.values())
    order = sorted(
        counts,
        key=lambda key: (-(raw[key] - quotas[key]), str(key)),
    )
    for key in order[:remainder]:
        quotas[key] += 1
    return quotas


def _representative_priority(record: dict[str, Any], seed: int) -> tuple[str, str]:
    return (
        sha256_bytes(f"{seed}\0{record['clip_id']}".encode("utf-8")),
        str(record["clip_id"]),
    )


def select_representative_set(
    candidates: list[dict[str, Any]], *, representative_count: int, seed: int = 20260725
) -> list[dict[str, Any]]:
    """Approximate the pool's duration/event joint distribution without emotion."""

    if representative_count != 100:
        raise ValueError("v8 representative-set contract currently requires exactly 100 records")
    joint_counts = Counter(
        (str(record["duration_band"]), str(record["event_count_band"]))
        for record in candidates
    )
    joint_quotas = _proportional_quotas(dict(joint_counts), representative_count)
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        key = (
            str(record["duration_band"]),
            str(record["event_count_band"]),
            str(record["fixed_split_assignment"]),
        )
        buckets[key].append(record)
    for bucket in buckets.values():
        bucket.sort(key=lambda record: _representative_priority(record, seed))

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    used_sources: set[str] = set()
    for joint in sorted(joint_quotas):
        quota = joint_quotas[joint]
        split_counts = {
            (split,): len(buckets[(joint[0], joint[1], split)]) for split in SPLITS
        }
        split_quotas = _proportional_quotas(split_counts, quota)
        for split in SPLITS:
            for _ in range(split_quotas[(split,) ]):
                choice = next(
                    (
                        record
                        for record in buckets[(joint[0], joint[1], split)]
                        if record["clip_id"] not in selected_ids
                        and record["source_clip_id"] not in used_sources
                    ),
                    None,
                )
                if choice is None:
                    choice = next(
                        (
                            record
                            for fallback_split in SPLITS
                            for record in buckets[(joint[0], joint[1], fallback_split)]
                            if record["clip_id"] not in selected_ids
                            and record["source_clip_id"] not in used_sources
                        ),
                        None,
                    )
                if choice is None:
                    raise ValueError(
                        f"unfillable representative stratum: {joint}, quota={quota}"
                    )
                selected_ids.add(str(choice["clip_id"]))
                used_sources.add(str(choice["source_clip_id"]))
                selected.append(choice)
    if len(selected) != representative_count:
        raise ValueError("representative selection count mismatch")
    if {record["fixed_split_assignment"] for record in selected} != set(SPLITS):
        raise ValueError("representative set does not cover all fixed splits")
    selected.sort(
        key=lambda record: (
            str(record["duration_band"]),
            str(record["event_count_band"]),
            str(record["fixed_split_assignment"]),
            _representative_priority(record, seed),
        )
    )
    return _attach_selection_lineage(
        selected,
        kind="representative100",
        status="selected_representative_pending_retarget_qc",
    )


def _counts(records: Iterable[dict[str, Any]], key) -> dict[str, int]:
    return dict(sorted(Counter(str(key(record)) for record in records).items()))


def _duration_accounting(prefix: str, frames: list[int]) -> dict[str, Any]:
    return {
        f"{prefix}_frame_coverage_hours_at_30hz": round(
            sum(frames) / FPS / 3600, 8
        ),
        f"{prefix}_sample_span_hours_at_30hz": round(
            sum(max(0, frame_count - 1) for frame_count in frames) / FPS / 3600,
            8,
        ),
    }


def _review_set_summary(
    prefix: str, records: list[dict[str, Any]], *, role: str
) -> dict[str, Any]:
    frames = [record["training_segment"]["frame_count"] for record in records]
    return {
        f"{prefix}_role": role,
        f"{prefix}_count": len(records),
        f"{prefix}_counts_by_split": _counts(
            records, lambda item: item["fixed_split_assignment"]
        ),
        f"{prefix}_counts_by_emotion_metadata": _counts(
            records, lambda item: item["emotion_id"]
        ),
        f"{prefix}_counts_by_dominant_category": _counts(
            records,
            lambda item: item["expression_turn"]["dominant_official_category"],
        ),
        f"{prefix}_counts_by_event_count_band": _counts(
            records, lambda item: item["event_count_band"]
        ),
        f"{prefix}_counts_by_duration_band": _counts(
            records, lambda item: item["duration_band"]
        ),
        f"{prefix}_source_count": len(
            {record["source_clip_id"] for record in records}
        ),
        f"{prefix}_speaker_count": len(
            {record["speaker_key"] for record in records}
        ),
        f"{prefix}_frame_count": sum(frames),
        f"{prefix}_frame_count_min": min(frames) if frames else None,
        f"{prefix}_frame_count_max": max(frames) if frames else None,
        f"{prefix}_source_disjoint": len(
            {record["source_clip_id"] for record in records}
        )
        == len(records),
        **_duration_accounting(prefix, frames),
    }


def build_catalog(
    input_path: Path,
    beat2_root: Path,
    split_path: Path,
    output_dir: Path,
    *,
    workers: int = 4,
    pilot_count: int = 100,
    rest_energy_max: float = 0.14,
    rest_radius_frames: int = 2,
    max_energy_p95: float = 4.0,
    limit_sources: int | None = None,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    if pilot_count not in {0, 100}:
        raise ValueError("v8 pilot count must be 100, or 0 for a limited diagnostic")
    if pilot_count == 0 and limit_sources is None:
        raise ValueError("pilot_count=0 is allowed only with limit_sources")
    if rest_energy_max < 0:
        raise ValueError("invalid rest threshold")
    input_path = input_path.resolve()
    beat2_root = beat2_root.resolve()
    split_path = split_path.resolve()
    output_dir = output_dir.resolve()
    source_manifest_sha = sha256_file(input_path)
    split_assignment, split_manifest_sha = load_split_assignment(split_path)
    policy = {
        "schema_version": SCHEMA_VERSION,
        "representation": REPRESENTATION,
        "duration_policy": "natural_rest_to_natural_rest_no_fixed_or_max_duration",
        "fixed_window_sec": None,
        "cropping_allowed": False,
        "adjacency": {
            "hard_barrier_labels": sorted(HARD_BARRIER_LABELS),
            "natural_rest_basin_splits_turn": True,
            "time_gap_used_as_acceptance_or_split_gate": False,
            "continuous_motion_can_merge_across_arbitrary_gap_duration": True,
        },
        "completeness": {
            "rest_energy_max_rad_s": rest_energy_max,
            "rest_radius_frames": rest_radius_frames,
            "context_expansion_basis": (
                "ordered_natural_rest_basins_until_same_source_neighbor_barrier"
            ),
            "apex_energy_role": "diagnostic_candidate_evidence_only_not_rejection_gate",
            "reject_if_incomplete": True,
        },
        "interaction_energy_p95_max_rad_s": max_energy_p95,
        "split_assignment": {
            "policy": "reuse_v7_speaker_strict_partition",
            "manifest_sha256": split_manifest_sha,
        },
        "review_sets": {
            "stress100": {
                "records": pilot_count,
                "counts_by_split": STRESS_SPLIT_COUNTS if pilot_count else {},
                "role": "compound_long_sequence_stress_review_not_pool_acceptance_estimator",
                "stratification": (
                    "split_then_official_emotion_metadata_then_dominant_category_"
                    "with_compound_turn_priority"
                ),
                "official_emotion_role": (
                    "hidden_blind_sampling_stratum_only_not_review_payload_or_supervision"
                ),
            },
            "representative100": {
                "records": pilot_count,
                "role": "pool_physical_quality_rate_estimator",
                "stratification": (
                    "full_pool_duration_band_x_event_count_band_proportional_"
                    "source_disjoint_with_split_coverage"
                ),
                "official_emotion_used_for_selection": False,
            },
        },
        "semantic_supervision_masks": dict(SEMANTIC_MASKS),
        "official_emotion_conditioning_enabled": False,
        "emotion_supervision_mask": False,
        "affect_observable_supervision_mask": False,
        "audio_enabled": False,
        "source_manifest_sha256": source_manifest_sha,
        "builder_script_sha256": sha256_file(Path(__file__).resolve()),
    }
    contract_sha = sha256_bytes(stable_json(policy).encode("utf-8"))
    records = load_jsonl(input_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        speaker = str(record.get("speaker_key") or "")
        if speaker not in split_assignment:
            raise ValueError(f"speaker missing from strict split assignment: {speaker}")
        grouped[str(record["source_group_key"])].append(record)
    source_keys = sorted(grouped)
    if limit_sources is not None:
        if limit_sources < 1:
            raise ValueError("limit_sources must be positive")
        source_keys = source_keys[:limit_sources]

    tasks = []
    for key in source_keys:
        group = grouped[key]
        source = group[0]
        consistent = (
            "source_clip_id",
            "speaker_key",
            "motion_relpath",
            "motion_sha256",
            "annotation_relpath",
            "textgrid_relpath",
            "emotion_id",
        )
        for field in consistent:
            if len({stable_json(record.get(field)) for record in group}) != 1:
                raise ValueError(f"{key}: inconsistent source field {field}")
        tasks.append(
            {
                "records": group,
                "beat2_root": str(beat2_root),
                "split": split_assignment[str(source["speaker_key"])],
                "contract_sha256": contract_sha,
                "source_manifest_sha256": source_manifest_sha,
                "split_manifest_sha256": split_manifest_sha,
                "rest_energy_max": rest_energy_max,
                "rest_radius_frames": rest_radius_frames,
                "max_energy_p95": max_energy_p95,
            }
        )

    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_source, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            candidates.extend(result["candidates"])
            rejected.extend(result["rejections"])
            completed += 1
            if completed % 25 == 0 or completed == len(tasks):
                print(
                    stable_json(
                        {
                            "completed_sources": completed,
                            "total_sources": len(tasks),
                            "candidates": len(candidates),
                            "rejected": len(rejected),
                        }
                    ),
                    flush=True,
                )
    candidates.sort(key=lambda record: record["clip_id"])
    rejected.sort(
        key=lambda record: (
            str(record.get("source_group_key")),
            str(record.get("included_source_line_numbers")),
            str(record.get("reason")),
        )
    )
    clip_ids = [record["clip_id"] for record in candidates]
    if len(set(clip_ids)) != len(clip_ids):
        raise ValueError("duplicate expression turn clip_id")
    stress = (
        select_stress_set(candidates, pilot_count=pilot_count)
        if pilot_count
        else []
    )
    representative = (
        select_representative_set(
            candidates, representative_count=pilot_count
        )
        if pilot_count
        else []
    )

    candidate_path = output_dir / f"{OUTPUT_STEM}.candidates.jsonl"
    rejected_path = output_dir / f"{OUTPUT_STEM}.rejected.jsonl"
    stress_path = output_dir / f"{OUTPUT_STEM}.stress100.jsonl"
    representative_path = output_dir / f"{OUTPUT_STEM}.representative100.jsonl"
    output_hashes = {
        candidate_path.name: atomic_jsonl(candidate_path, candidates),
        rejected_path.name: atomic_jsonl(rejected_path, rejected),
        stress_path.name: atomic_jsonl(stress_path, stress),
        representative_path.name: atomic_jsonl(
            representative_path, representative
        ),
    }
    candidate_frames = [record["training_segment"]["frame_count"] for record in candidates]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "beat2_expression_turn_v8_candidate_catalog",
        "expression_turn_contract": policy,
        "expression_turn_contract_sha256": contract_sha,
        "input": str(input_path),
        "input_sha256": source_manifest_sha,
        "split_assignment": str(split_path),
        "split_assignment_sha256": split_manifest_sha,
        "source_count": len(tasks),
        "candidate_count": len(candidates),
        "rejected_count": len(rejected),
        "rejection_counts_by_reason": _counts(rejected, lambda item: item["reason"]),
        "candidate_counts_by_split": _counts(
            candidates, lambda item: item["fixed_split_assignment"]
        ),
        "candidate_counts_by_emotion_metadata": _counts(
            candidates, lambda item: item["emotion_id"]
        ),
        "candidate_counts_by_event_count_band": _counts(
            candidates, lambda item: item["event_count_band"]
        ),
        "candidate_counts_by_duration_band": _counts(
            candidates, lambda item: item["duration_band"]
        ),
        "candidate_frame_count_min": min(candidate_frames),
        "candidate_frame_count_max": max(candidate_frames),
        "candidate_distinct_frame_count_count": len(set(candidate_frames)),
        **_duration_accounting("candidate", candidate_frames),
        "duration_accounting": {
            "frame_coverage": "sum(frame_count)/fps",
            "sample_span": "sum(max(frame_count-1,0))/fps",
            "planner_duration_basis": "sample_span",
            "frame_coverage_is_not_planner_target": True,
        },
        **_review_set_summary(
            "stress100",
            stress,
            role=(
                "compound_long_sequence_stress_review_not_pool_acceptance_estimator"
            ),
        ),
        **_review_set_summary(
            "representative100",
            representative,
            role="pool_physical_quality_rate_estimator",
        ),
        "speaker_disjoint_splits": all(
            len({record["fixed_split_assignment"] for record in candidates if record["speaker_key"] == speaker}) == 1
            for speaker in {record["speaker_key"] for record in candidates}
        ),
        "source_disjoint_splits": all(
            len({record["fixed_split_assignment"] for record in candidates if record["source_clip_id"] == source}) == 1
            for source in {record["source_clip_id"] for record in candidates}
        ),
        "fixed_window_sec": None,
        "accepted_for_training": 0,
        "output_sha256": output_hashes,
    }
    summary_path = output_dir / f"{OUTPUT_STEM}.summary.json"
    atomic_json(summary_path, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_catalog(
        args.input,
        args.beat2_root,
        args.split_assignments,
        args.output_dir,
        workers=args.workers,
        pilot_count=args.pilot_count,
        rest_energy_max=args.rest_energy_max,
        rest_radius_frames=args.rest_radius_frames,
        max_energy_p95=args.max_energy_p95,
        limit_sources=args.limit_sources,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
