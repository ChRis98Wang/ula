#!/usr/bin/env python3
"""Audit the BEAT2 semantic-event ingest gap and plan motion-only chunks.

Only motion-container headers, semantic label/timing columns, and the existing
speaker split are read.  The generated inventory contains no text, audio,
semantic, behavior, or affect conditioning payloads and is not train-ready
until every chunk has passed the unchanged 18D physical-QC contract.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATASET_ROOT = Path(
    "/home/gez/nas/cloud/gez/human_motion/raw/BEAT2/beat_english_v2.0.0"
)
DEFAULT_MOTION_ROOT = RAW_DATASET_ROOT / "smplxflame_30"
DEFAULT_SEMANTIC_ROOT = RAW_DATASET_ROOT / "sem"
DEFAULT_SELECTOR_SUMMARY = Path(
    "/home/gez/nas/cloud/gez/human_motion/catalog/"
    "beat2_semantic_event_pilot_v7_full/"
    "beat2_semantic_event_pilot_v7_full.summary.json"
)
DEFAULT_FULL_CANDIDATES = Path(
    "/home/gez/nas/cloud/gez/human_motion/catalog/"
    "beat2_semantic_event_inventory_v1/"
    "beat2_semantic_event_inventory_v1.full_candidates.jsonl"
)
DEFAULT_SUPPORTED = Path(
    "/home/gez/nas/cloud/gez/human_motion/catalog/"
    "beat2_semantic_event_inventory_v1/"
    "beat2_semantic_event_inventory_v1.network_emotion_supported.jsonl"
)
DEFAULT_POOL = Path(
    "/home/gez/nas/cloud/gez/human_motion/catalog/"
    "beat2_semantic_event_pilot_v7_full/training_pool_low_medium.jsonl"
)
DEFAULT_V7_RELEASE_REPORT = Path(
    "/home/gez/nas/cloud/gez/human_motion/processed/"
    "beat2_semantic_event_training_pool_18d_v7_full/"
    "adjudication_min30f/motion_only_release_report.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/gez/nas/cloud/gez/human_motion/processed/"
    "beat2_semantic_event_training_pool_18d_v8/expansion/ingest_gap"
)
DEFAULT_BEAT2_ROOT = Path("/home/gez/nas/cloud/gez/human_motion/raw/BEAT2")
DEFAULT_GMR_PYTHON = Path("/home/gez/shuaiwang/.venvs/gmr/bin/python")
DEFAULT_SMOKE_ROOT = Path(
    "/home/gez/nas/cloud/gez/human_motion/processed/"
    "beat2_semantic_event_training_pool_18d_v8/expansion/"
    "smoke_motion_foundation_cpu_v3"
)
FORBIDDEN_TOKEN = "kimodo"
FOUNDATION_ANNOTATION_KIND = "motion_foundation_unlabeled_contiguous_chunk"
FOUNDATION_REPRESENTATION = "motion_foundation_contiguous_nonoverlap_chunk_v1"
FULL_WINDOW_SELECTION_STATUS = "full_nonoverlap_boundary_validated"
SEMANTIC_MASKS = {
    "official_category": False,
    "robot_observable_motion_form": False,
    "communicative_intent": False,
    "prompt_text": False,
    "legacy_gesture": False,
}


def stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite ingest-gap artifact: {path}")
    atomic_text(path, "".join(stable_json(record) + "\n" for record in records))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            records.append(value)
    return records


def assert_no_forbidden_reference(value: Any, *, label: str) -> None:
    if isinstance(value, str):
        if FORBIDDEN_TOKEN in value.casefold():
            raise ValueError(f"{label} contains a forbidden cross-dataset reference")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            assert_no_forbidden_reference(str(key), label=label)
            assert_no_forbidden_reference(child, label=label)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for child in value:
            assert_no_forbidden_reference(child, label=label)


def npy_shape(archive: zipfile.ZipFile, member: str) -> tuple[int, ...]:
    with archive.open(member) as stream:
        version = np.lib.format.read_magic(stream)
        shape, _fortran_order, _dtype = np.lib.format._read_array_header(
            stream, version
        )
    return tuple(int(value) for value in shape)


def motion_frame_count(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        if "poses.npy" not in names or "mocap_frame_rate.npy" not in names:
            raise ValueError(f"BEAT2 motion container is incomplete: {path}")
        shape = npy_shape(archive, "poses.npy")
        with archive.open("mocap_frame_rate.npy") as stream:
            rate = float(np.load(stream, allow_pickle=False))
    if len(shape) != 2 or shape[0] < 3:
        raise ValueError(f"invalid poses shape in {path}: {shape}")
    if not np.isclose(rate, 30.0, rtol=0.0, atol=1e-9):
        raise ValueError(f"motion container is not 30 Hz: {path} ({rate})")
    return shape[0]


def speaker_key(source_clip_id: str) -> str:
    parts = source_clip_id.split("_")
    if len(parts) < 3 or not parts[0].isdigit():
        raise ValueError(f"cannot parse BEAT2 speaker from {source_clip_id}")
    return "_".join(parts[:2])


def speaker_split(summary: Mapping[str, Any]) -> dict[str, str]:
    mapping = (
        (summary.get("speaker_assignment") or {}).get("speaker_to_split") or {}
    )
    result = {str(key): str(value) for key, value in mapping.items()}
    if set(result.values()) != {"train", "validation", "test"}:
        raise ValueError("selector summary has no complete fixed speaker split")
    return result


def chunk_bounds(
    total_frames: int, *, max_frames: int, min_frames: int
) -> list[tuple[int, int]]:
    if min_frames < 3 or max_frames < min_frames:
        raise ValueError("invalid chunk frame limits")
    if total_frames < min_frames:
        return []
    chunks = [
        (start, min(total_frames, start + max_frames))
        for start in range(0, total_frames, max_frames)
    ]
    if len(chunks) > 1 and chunks[-1][1] - chunks[-1][0] < min_frames:
        chunks[-2] = (chunks[-2][0], chunks[-1][1])
        chunks.pop()
    return chunks


def union_frame_count(intervals: Sequence[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    total = 0
    left, right = sorted(intervals)[0]
    for start, end in sorted(intervals)[1:]:
        if start <= right:
            right = max(right, end)
        else:
            total += right - left
            left, right = start, end
    return total + right - left


def event_inventory_summary(
    path: Path, source_frames: Mapping[str, int]
) -> dict[str, Any]:
    records = read_jsonl(path)
    intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    declared_summed_frames = 0
    source_bounded_summed_frames = 0
    core_seconds = 0.0
    out_of_bounds_records: list[dict[str, Any]] = []
    for record in records:
        assert_no_forbidden_reference(record, label=str(path))
        source = str(record["source_clip_id"])
        segment = record["training_segment"]
        start = int(segment["start_frame"])
        declared_end = int(segment["end_frame_exclusive"])
        if source not in source_frames:
            raise ValueError(f"event references a missing source: {source}")
        if not 0 <= start < declared_end or start >= source_frames[source]:
            raise ValueError(
                f"event has no valid source-bounded interval: "
                f"{source}[{start}:{declared_end}]"
            )
        bounded_end = min(declared_end, source_frames[source])
        if bounded_end != declared_end:
            out_of_bounds_records.append(
                {
                    "task_id": str(record.get("task_id") or record.get("clip_id")),
                    "source_clip_id": source,
                    "start_frame": start,
                    "declared_end_frame_exclusive": declared_end,
                    "source_end_frame_exclusive": source_frames[source],
                    "declared_excess_frames": declared_end - bounded_end,
                }
            )
        intervals[source].append((start, bounded_end))
        declared_summed_frames += declared_end - start
        source_bounded_summed_frames += bounded_end - start
        event = record.get("semantic_event") or {}
        core_seconds += float(event.get("source_duration_sec") or 0.0)
    union_frames = sum(union_frame_count(spans) for spans in intervals.values())
    return {
        "records": len(records),
        "source_count": len(intervals),
        # Keep declared totals so the audit reconciles exactly with the existing
        # 15,054 -> 14,973 inventory accounting.  Use the source-bounded values
        # for physical-coverage and ingest-gap calculations.
        "summed_frames": declared_summed_frames,
        "summed_hours": declared_summed_frames / 108000.0,
        "source_bounded_summed_frames": source_bounded_summed_frames,
        "source_bounded_summed_hours": source_bounded_summed_frames / 108000.0,
        "union_frames": union_frames,
        "union_hours": union_frames / 108000.0,
        "overlap_hours": (source_bounded_summed_frames - union_frames) / 108000.0,
        "out_of_bounds_record_count": len(out_of_bounds_records),
        "declared_excess_frames": sum(
            record["declared_excess_frames"] for record in out_of_bounds_records
        ),
        "out_of_bounds_records": out_of_bounds_records,
        "official_core_hours": core_seconds / 3600.0,
        "adaptive_context_hours": declared_summed_frames / 108000.0
        - core_seconds / 3600.0,
    }


def semantic_label_timing_summary(
    semantic_root: Path, source_frames: Mapping[str, int]
) -> dict[str, Any]:
    label_seconds: Counter[str] = Counter()
    label_events: Counter[str] = Counter()
    missing = []
    union_seconds = 0.0
    overlap_seconds = 0.0
    for source, frames in source_frames.items():
        path = semantic_root / f"{source}.txt"
        if not path.is_file():
            missing.append(source)
            continue
        duration = frames / 30.0
        intervals: list[tuple[float, float]] = []
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                columns = line.rstrip("\n").split("\t")
                if len(columns) < 3:
                    continue
                # Deliberately ignore every column after label/start/end.
                label = columns[0].strip()
                start = max(0.0, float(columns[1]))
                end = min(duration, float(columns[2]))
                if end <= start:
                    continue
                label_seconds[label] += end - start
                label_events[label] += 1
                intervals.append((start, end))
        if intervals:
            merged = 0.0
            left, right = sorted(intervals)[0]
            for start, end in sorted(intervals)[1:]:
                if start <= right:
                    overlap_seconds += max(0.0, min(right, end) - start)
                    right = max(right, end)
                else:
                    merged += right - left
                    left, right = start, end
            union_seconds += merged + right - left
    groups = {
        "no_gesture": lambda label: label.startswith("00_"),
        "beat_aligned": lambda label: label.startswith("01_"),
        "official_semantic_02_10": lambda label: (
            label[:2].isdigit() and 2 <= int(label[:2]) <= 10
        ),
        "other_control_or_cleanup": lambda label: not (
            label.startswith("00_")
            or label.startswith("01_")
            or (label[:2].isdigit() and 2 <= int(label[:2]) <= 10)
        ),
    }
    return {
        "missing_semantic_timing_files": missing,
        "union_annotation_hours": union_seconds / 3600.0,
        "overlap_hours": overlap_seconds / 3600.0,
        "by_label": {
            label: {
                "events": label_events[label],
                "hours": seconds / 3600.0,
            }
            for label, seconds in sorted(label_seconds.items())
        },
        "by_label_group_hours": {
            name: sum(
                seconds
                for label, seconds in label_seconds.items()
                if predicate(label)
            )
            / 3600.0
            for name, predicate in groups.items()
        },
        "text_columns_read": False,
    }


def source_scale(
    sources: Iterable[str],
    source_frames: Mapping[str, int],
    splits: Mapping[str, str],
) -> dict[str, Any]:
    source_set = set(sources)
    by_split_frames = Counter()
    by_split_sources = Counter()
    for source in source_set:
        split = splits[speaker_key(source)]
        by_split_frames[split] += source_frames[source]
        by_split_sources[split] += 1
    frames = sum(source_frames[source] for source in source_set)
    return {
        "sources": len(source_set),
        "frames": frames,
        "hours": frames / 108000.0,
        "by_fixed_split": {
            split: {
                "sources": by_split_sources[split],
                "frames": by_split_frames[split],
                "hours": by_split_frames[split] / 108000.0,
            }
            for split in ("train", "validation", "test")
        },
    }


def build_chunk_inventory(
    *,
    motion_root: Path,
    source_frames: Mapping[str, int],
    splits: Mapping[str, str],
    max_frames: int,
    min_frames: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in sorted(source_frames):
        speaker = speaker_key(source)
        fixed_split = splits[speaker]
        relative_motion = (
            Path("beat_english_v2.0.0") / motion_root.name / f"{source}.npz"
        )
        bounds = chunk_bounds(
            source_frames[source],
            max_frames=max_frames,
            min_frames=min_frames,
        )
        for index, (start, end) in enumerate(bounds):
            task_id = (
                f"beat2_motion_foundation__{source}_"
                f"chunk{index:04d}_f{start:06d}-{end:06d}"
            )
            records.append(
                {
                    "schema_version": "1.0.0",
                    "clip_id": task_id,
                    "task_id": task_id,
                    "dataset": "BEAT2",
                    "dataset_subset": "beat_english_v2.0.0",
                    "language": "english",
                    "language_code": "en",
                    "source_clip_id": source,
                    "source_group_key": (
                        f"BEAT2/beat_english_v2.0.0/{source}"
                    ),
                    "speaker_key": speaker,
                    "fixed_split_assignment": fixed_split,
                    "fps": 30.0,
                    "motion_relpath": relative_motion.as_posix(),
                    "annotation_kind": FOUNDATION_ANNOTATION_KIND,
                    "semantic_label_status": "absent_motion_foundation",
                    "semantic_supervision_masks": dict(SEMANTIC_MASKS),
                    "behavior_supervision_mask": False,
                    "emotion_supervision_mask": False,
                    "affect_observable_supervision_mask": False,
                    "official_category_conditioning_enabled": False,
                    "official_emotion_conditioning_enabled": False,
                    "window": {
                        "selection_status": FULL_WINDOW_SELECTION_STATUS,
                        "start_frame": start,
                        "end_frame_exclusive": end,
                        "frame_count": end - start,
                    },
                    "training_segment": {
                        "representation": FOUNDATION_REPRESENTATION,
                        "start_frame": start,
                        "end_frame_exclusive": end,
                        "frame_count": end - start,
                        "fixed_window_sec": None,
                        "nominal_chunk_frames": max_frames,
                        "maximum_chunk_frames": max_frames + min_frames - 1,
                        "minimum_chunk_frames": min_frames,
                        "overlap_frames": 0,
                        "tail_policy": "sub_minimum_tail_absorbed_into_previous_chunk",
                        "boundary_source": "source_container_frame_bounds",
                    },
                    "issues": [],
                    "accepted_for_training": False,
                    "training_admission_status": (
                        "pending_18d_retarget_and_unchanged_physical_qc"
                    ),
                }
            )
    return records


def validate_chunk_inventory(
    records: Sequence[Mapping[str, Any]],
    *,
    source_frames: Mapping[str, int],
    splits: Mapping[str, str],
) -> dict[str, Any]:
    ids = set()
    intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    by_split = Counter()
    by_split_frames = Counter()
    for record in records:
        assert_no_forbidden_reference(record, label="foundation inventory")
        task_id = str(record.get("task_id") or "")
        if not task_id or task_id in ids:
            raise ValueError(f"missing or duplicate task_id: {task_id}")
        ids.add(task_id)
        if record.get("dataset") != "BEAT2":
            raise ValueError("foundation inventory contains a non-BEAT2 record")
        if record.get("semantic_supervision_masks") != SEMANTIC_MASKS:
            raise ValueError("foundation inventory semantic masks changed")
        if any(
            field in record
            for field in (
                "audio_relpath",
                "canonical_prompt",
                "prompt",
                "source_text",
                "window_transcript_context",
            )
        ):
            raise ValueError("foundation inventory contains conditioning metadata")
        source = str(record["source_clip_id"])
        segment = record["training_segment"]
        start, end = int(segment["start_frame"]), int(segment["end_frame_exclusive"])
        if not 0 <= start < end <= source_frames[source]:
            raise ValueError(f"chunk escapes source bounds: {task_id}")
        if end - start > int(segment["maximum_chunk_frames"]):
            raise ValueError(f"chunk exceeds its declared maximum: {task_id}")
        if record.get("fixed_split_assignment") != splits[speaker_key(source)]:
            raise ValueError(f"chunk changes fixed speaker split: {task_id}")
        intervals[source].append((start, end))
        split = str(record["fixed_split_assignment"])
        by_split[split] += 1
        by_split_frames[split] += end - start
    for source, spans in intervals.items():
        spans = sorted(spans)
        if spans[0][0] != 0 or spans[-1][1] != source_frames[source]:
            raise ValueError(f"chunks do not cover source endpoints: {source}")
        if any(left[1] != right[0] for left, right in zip(spans, spans[1:])):
            raise ValueError(f"chunks have a gap or overlap: {source}")
    if set(intervals) != set(source_frames):
        raise ValueError("chunk inventory dropped source recordings")
    return {
        "records": len(records),
        "sources": len(intervals),
        "frames": sum(by_split_frames.values()),
        "hours": sum(by_split_frames.values()) / 108000.0,
        "by_fixed_split": {
            split: {
                "records": by_split[split],
                "frames": by_split_frames[split],
                "hours": by_split_frames[split] / 108000.0,
            }
            for split in ("train", "validation", "test")
        },
        "source_coverage_complete": True,
        "source_overlap_frames": 0,
    }


def projection(
    source_frames: Mapping[str, int],
    splits: Mapping[str, str],
    *,
    max_seconds: int,
    min_frames: int,
) -> dict[str, Any]:
    max_frames = max_seconds * 30
    records = 0
    by_split = Counter()
    for source, frames in source_frames.items():
        count = len(
            chunk_bounds(frames, max_frames=max_frames, min_frames=min_frames)
        )
        records += count
        by_split[splits[speaker_key(source)]] += count
    return {
        "nominal_chunk_seconds": max_seconds,
        "maximum_chunk_seconds_after_tail_absorption": (
            max_seconds + (min_frames - 1) / 30.0
        ),
        "records": records,
        "unique_source_hours": sum(source_frames.values()) / 108000.0,
        "by_fixed_split_records": {
            split: by_split[split] for split in ("train", "validation", "test")
        },
    }


def tree_bytes(root: Path, *, excluded_names: frozenset[str] = frozenset()) -> int:
    return sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and path.name not in excluded_names
    )


def smoke_cost_basis(smoke_root: Path, *, planned_records: int) -> dict[str, Any]:
    if not smoke_root.is_dir():
        return {
            "available": False,
            "path": str(smoke_root.absolute()),
            "reason": "bounded CPU-only smoke has not been run",
        }
    status = read_json(smoke_root / "status.json")
    passed = read_jsonl(smoke_root / "passed_manifest.jsonl")
    started = datetime.fromisoformat(str(status["started_at"]))
    updated = datetime.fromisoformat(str(status["updated_at"]))
    event_elapsed = sum(float(record.get("elapsed_sec") or 0.0) for record in passed)
    output_frames = sum(int(record.get("frames") or 0) for record in passed)
    non_pending_bytes = tree_bytes(
        smoke_root, excluded_names=frozenset({"pending_manifest.jsonl"})
    )
    projected_bytes = (
        non_pending_bytes * planned_records / len(passed) if passed else 0.0
    )
    return {
        "available": True,
        "path": str(smoke_root.absolute()),
        "mode": "cpu_only_cuda_hidden_nice15_ionice7_one_worker_two_cores",
        "selected_records": int(status.get("selected_event_count") or 0),
        "passed_records": len(passed),
        "output_frames": output_frames,
        "wall_seconds": (updated - started).total_seconds(),
        "summed_per_event_elapsed_seconds": event_elapsed,
        "non_pending_artifact_bytes": non_pending_bytes,
        "naive_artifact_projection_gib": projected_bytes / (1024**3),
        "caution": (
            "two records validate compatibility but are too small for a narrow "
            "capacity forecast"
        ),
    }


def guarded_retarget_command(
    *,
    python: Path,
    inventory: Path,
    beat2_root: Path,
    output_root: Path,
    workers: int,
    cpu_affinity: str,
) -> list[str]:
    return [
        "/usr/bin/env",
        "CUDA_VISIBLE_DEVICES=",
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1",
        "TOKENIZERS_PARALLELISM=false",
        "/usr/bin/ionice",
        "-c",
        "2",
        "-n",
        "7",
        "/usr/bin/nice",
        "-n",
        "15",
        "/usr/bin/taskset",
        "-c",
        cpu_affinity,
        str(python.absolute()),
        str(
            PROJECT_ROOT
            / "tools/gmr_v2/batch_retarget_beat2_semantic_events_v2.py"
        ),
        "--inventory",
        str(inventory),
        "--beat2-root",
        str(beat2_root.resolve()),
        "--output-root",
        str(output_root),
        "--workers",
        str(workers),
        "--neutral-limit-margin-rad",
        "1e-6",
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite ingest-gap directory: {output_dir}")
    configured = {
        "motion_root": str(args.motion_root.resolve()),
        "semantic_root": str(args.semantic_root.resolve()),
        "selector_summary": str(args.selector_summary.resolve()),
        "full_candidates": str(args.full_candidates.resolve()),
        "supported": str(args.supported.resolve()),
        "pool": str(args.pool.resolve()),
        "v7_release_report": str(args.v7_release_report.resolve()),
        "output_dir": str(output_dir),
        "beat2_root": str(args.beat2_root.resolve()),
        # Preserve the virtual-environment entry point.  Resolving this symlink
        # would produce the base interpreter path and lose the venv context.
        "gmr_python": str(args.gmr_python.absolute()),
    }
    assert_no_forbidden_reference(configured, label="configured paths")
    for name, value in configured.items():
        if name == "output_dir":
            continue
        if not Path(value).exists():
            raise FileNotFoundError(value)
    summary = read_json(args.selector_summary)
    splits = speaker_split(summary)
    motion_paths = sorted(args.motion_root.glob("*.npz"))
    source_frames = {
        path.stem: motion_frame_count(path) for path in motion_paths
    }
    if len(source_frames) != len(motion_paths):
        raise ValueError("duplicate BEAT2 motion source ids")
    source_speakers = {speaker_key(source) for source in source_frames}
    if source_speakers != set(splits):
        raise ValueError("raw motion speakers differ from locked speaker split")

    inventories = {
        "full_official_semantic_candidates": event_inventory_summary(
            args.full_candidates, source_frames
        ),
        "network_emotion_supported": event_inventory_summary(
            args.supported, source_frames
        ),
        "low_medium_pool": event_inventory_summary(args.pool, source_frames),
    }
    full_sources = {
        str(record["source_clip_id"]) for record in read_jsonl(args.full_candidates)
    }
    supported_sources = {
        str(record["source_clip_id"]) for record in read_jsonl(args.supported)
    }
    raw_scale = source_scale(source_frames, source_frames, splits)
    source_coverage = {
        "supported_event_sources": source_scale(
            supported_sources, source_frames, splits
        ),
        "unsupported_emotion_candidate_sources": source_scale(
            full_sources - supported_sources, source_frames, splits
        ),
        "no_valid_semantic_candidate_sources": source_scale(
            set(source_frames) - full_sources, source_frames, splits
        ),
    }
    semantic_timing = semantic_label_timing_summary(
        args.semantic_root, source_frames
    )
    chunks = build_chunk_inventory(
        motion_root=args.motion_root,
        source_frames=source_frames,
        splits=splits,
        max_frames=args.max_chunk_seconds * 30,
        min_frames=args.min_frames,
    )
    chunk_scale = validate_chunk_inventory(
        chunks, source_frames=source_frames, splits=splits
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    inventory_path = output_dir / (
        f"beat2_motion_foundation_nonoverlap_nominal{args.max_chunk_seconds}s.jsonl"
    )
    write_jsonl(inventory_path, chunks)

    release = read_json(args.v7_release_report)
    observed_pool_hours = inventories["low_medium_pool"]["union_hours"]
    observed_pass_hours = float(
        (release.get("scale") or {}).get("total_frames") or 0
    ) / 108000.0
    observed_yield = (
        observed_pass_hours / observed_pool_hours
        if observed_pool_hours
        else 0.0
    )
    conservative_projection = {
        "basis": (
            "v7_min30_output_frame_coverage_divided_by_low_medium_source_coverage"
        ),
        "observed_ratio": observed_yield,
        "not_a_guarantee": True,
        "requires_full_retarget_qc": True,
        "projected_high_qc_hours": raw_scale["hours"] * observed_yield,
        "projected_by_fixed_split_hours": {
            split: raw_scale["by_fixed_split"][split]["hours"] * observed_yield
            for split in ("train", "validation", "test")
        },
    }
    retarget_root = (
        output_dir.parent
        / f"motion_foundation_retarget_nominal{args.max_chunk_seconds}s"
    )
    command = guarded_retarget_command(
        python=args.gmr_python,
        inventory=inventory_path,
        beat2_root=args.beat2_root,
        output_root=retarget_root,
        workers=args.recommended_workers,
        cpu_affinity=args.cpu_affinity,
    )
    pipeline_root = (
        output_dir.parent
        / f"motion_foundation_pipeline_nominal{args.max_chunk_seconds}s"
    )
    qc_receipt = pipeline_root / "qc_verification.json"
    adjudication_dir = pipeline_root / "adjudication"
    adjudication_report = (
        adjudication_dir / "motion_only_adjudication_report.json"
    )
    pending_lock = pipeline_root / "provenance_lock.pending.json"
    adjudication_tool = (
        PROJECT_ROOT / "tools/adjudicate_beat2_motion_foundation.py"
    )
    staged_commands = {
        "01_retarget_cpu_guarded": command,
        "01_resume_if_interrupted": [*command, "--resume"],
        "02_verify_original_qc": [
            str(args.gmr_python.absolute()),
            str(adjudication_tool),
            "verify-qc",
            "--inventory",
            str(inventory_path),
            "--retarget-root",
            str(retarget_root),
            "--selector-summary",
            str(args.selector_summary.resolve()),
            "--output",
            str(qc_receipt),
            "--verify-artifacts",
        ],
        "03_adjudicate_motion_only_unadmitted": [
            str(args.gmr_python.absolute()),
            str(adjudication_tool),
            "adjudicate",
            "--qc-verification",
            str(qc_receipt),
            "--output-dir",
            str(adjudication_dir),
        ],
        "04_hash_lock_pending": [
            str(args.gmr_python.absolute()),
            str(adjudication_tool),
            "lock",
            "--qc-verification",
            str(qc_receipt),
            "--release-report",
            str(adjudication_report),
            "--output",
            str(pending_lock),
        ],
    }
    smoke_basis = smoke_cost_basis(
        args.smoke_root.resolve(), planned_records=len(chunks)
    )
    report = {
        "schema_version": 1,
        "artifact_kind": "beat2_motion_foundation_ingest_gap_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "dataset": "BEAT2",
            "speaker_split_reused_without_change": True,
            "speaker_disjoint": True,
            "text_fields_in_inventory": 0,
            "audio_fields_in_inventory": 0,
            "semantic_behavior_affect_conditioning_enabled": False,
            "accepted_for_training": False,
            "admission_requirement": "retarget_and_pass_all_unchanged_18d_physical_qc",
            "forbidden_cross_dataset_reference_count": 0,
        },
        "inputs": {
            "motion_root": str(args.motion_root.resolve()),
            "motion_source_count": len(source_frames),
            "selector_summary": {
                "path": str(args.selector_summary.resolve()),
                "sha256": sha256_file(args.selector_summary),
            },
            "full_candidates": {
                "path": str(args.full_candidates.resolve()),
                "sha256": sha256_file(args.full_candidates),
            },
            "supported": {
                "path": str(args.supported.resolve()),
                "sha256": sha256_file(args.supported),
            },
            "pool": {
                "path": str(args.pool.resolve()),
                "sha256": sha256_file(args.pool),
            },
        },
        "raw_motion_scale": raw_scale,
        "official_scale_context": {
            "user_supplied_approximate_hours": 60.0,
            "local_snapshot_hours": raw_scale["hours"],
            "local_snapshot_source_count": len(source_frames),
            "interpretation": (
                "all exact counts in this report refer to the locally available "
                "1,620-container snapshot"
            ),
        },
        "semantic_event_scale": inventories,
        "semantic_timing_gap": semantic_timing,
        "source_coverage_classes": source_coverage,
        "gap": {
            "raw_motion_hours": raw_scale["hours"],
            "supported_event_hours": inventories[
                "network_emotion_supported"
            ]["union_hours"],
            "supported_fraction_of_raw": (
                inventories["network_emotion_supported"]["union_hours"]
                / raw_scale["hours"]
            ),
            "raw_hours_outside_supported_events": (
                raw_scale["hours"]
                - inventories["network_emotion_supported"]["union_hours"]
            ),
            "primary_cause": (
                "semantic-event-only sampling; beat-aligned and non-event motion "
                "dominates the source timeline"
            ),
        },
        "recommended_nonoverlap_plan": {
            "inventory": {
                "path": str(inventory_path),
                "records": len(chunks),
                "sha256": sha256_file(inventory_path),
            },
            "scale": chunk_scale,
            "nominal_chunk_seconds": args.max_chunk_seconds,
            "actual_maximum_chunk_frames": max(
                int(record["training_segment"]["frame_count"])
                for record in chunks
            ),
            "actual_maximum_chunk_seconds": max(
                int(record["training_segment"]["frame_count"])
                for record in chunks
            )
            / 30.0,
            "minimum_frames": args.min_frames,
            "tail_policy": "sub_minimum_tail_absorbed_into_previous_chunk",
            "unique_coverage_only": True,
            "retarget_command": command,
            "staged_commands": staged_commands,
            "full_launch_authorized": False,
        },
        "alternative_nonoverlap_projections": {
            str(seconds): projection(
                source_frames,
                splits,
                max_seconds=seconds,
                min_frames=args.min_frames,
            )
            for seconds in (4, 6, 10, 20, 30)
        },
        "sliding_window_caution": {
            "window_seconds": 6,
            "stride_seconds": 3,
            "projected_records": sum(
                (
                    0
                    if frames < 180
                    else len(range(0, max(1, frames - 180 + 1), 90))
                    + (
                        1
                        if (frames - 180) % 90 != 0
                        else 0
                    )
                )
                for frames in source_frames.values()
            ),
            "approximately_doubles_sample_hours_but_not_unique_motion_hours": True,
            "unique_motion_hours": raw_scale["hours"],
            "recommendation": "prefer_nonoverlap_for_first_physical_qc_pass",
        },
        "conservative_high_qc_projection": conservative_projection,
        "execution_cost_estimate": {
            "empirical_smoke": smoke_basis,
            "estimated_cpu_core_hours": {
                "low": 8.0,
                "high": 12.0,
            },
            "estimated_wall_hours_recommended_two_workers": {
                "otherwise_idle": [4.0, 7.0],
                "while_gpu_training_is_active": [6.0, 12.0],
            },
            "estimated_output_disk_gib": [2.5, 4.5],
            "recommended_free_disk_gib": 6.0,
            "gpu_allocation": "none_enforced_by_empty_CUDA_VISIBLE_DEVICES",
            "resource_guard": {
                "workers": args.recommended_workers,
                "cpu_affinity": args.cpu_affinity,
                "library_threads_per_worker": 1,
                "nice": 15,
                "ionice_class": 2,
                "ionice_priority": 7,
                "recommended_during_gpu_training": True,
                "increase_to_four_workers_only_after_gpu_training_or_cpu_contention_review": True,
            },
            "estimate_caveat": (
                "range extrapolates a two-chunk compatibility smoke plus the "
                "observed 513-record retry throughput; NAS contention and the "
                "full-corpus QC pass rate can move wall time and disk use"
            ),
        },
    }
    assert_no_forbidden_reference(report, label="ingest-gap report")
    atomic_json(output_dir / "ingest_gap_report.json", report)
    atomic_json(
        output_dir / "retarget_command.json",
        {"argv": command, "launch_authorized": False},
    )
    atomic_json(
        output_dir / "staged_commands.json",
        {
            "commands": staged_commands,
            "full_launch_authorized": False,
            "training_launch_authorized": False,
        },
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-root", type=Path, default=DEFAULT_MOTION_ROOT)
    parser.add_argument("--semantic-root", type=Path, default=DEFAULT_SEMANTIC_ROOT)
    parser.add_argument(
        "--selector-summary", type=Path, default=DEFAULT_SELECTOR_SUMMARY
    )
    parser.add_argument(
        "--full-candidates", type=Path, default=DEFAULT_FULL_CANDIDATES
    )
    parser.add_argument("--supported", type=Path, default=DEFAULT_SUPPORTED)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument(
        "--v7-release-report", type=Path, default=DEFAULT_V7_RELEASE_REPORT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--beat2-root", type=Path, default=DEFAULT_BEAT2_ROOT)
    parser.add_argument("--gmr-python", type=Path, default=DEFAULT_GMR_PYTHON)
    parser.add_argument("--smoke-root", type=Path, default=DEFAULT_SMOKE_ROOT)
    parser.add_argument("--max-chunk-seconds", type=int, default=10)
    parser.add_argument("--min-frames", type=int, default=30)
    parser.add_argument("--recommended-workers", type=int, default=2)
    parser.add_argument("--cpu-affinity", default="20-23")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_chunk_seconds < 1:
        raise ValueError("--max-chunk-seconds must be positive")
    if args.min_frames < 3:
        raise ValueError("--min-frames must be at least three")
    if args.recommended_workers < 1:
        raise ValueError("--recommended-workers must be positive")
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
