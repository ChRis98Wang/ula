#!/usr/bin/env python3
"""Audit and build a fail-closed BEAT2 v8 motion-only expansion.

The tool never mutates the locked v7 manifests.  Expansion retries are
restricted to the existing fixed train split and every recovered trajectory
must pass the unchanged 18D physical-QC gate set before it can be appended.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CATALOG_ROOT = Path(
    "/home/gez/nas/cloud/gez/human_motion/catalog"
)
PROCESSED_ROOT = Path(
    "/home/gez/nas/cloud/gez/human_motion/processed"
)
INVENTORY_ROOT = CATALOG_ROOT / "beat2_semantic_event_inventory_v1"
PILOT_ROOT = CATALOG_ROOT / "beat2_semantic_event_pilot_v7_full"
V7_ROOT = PROCESSED_ROOT / "beat2_semantic_event_training_pool_18d_v7_full"
DEFAULT_OUTPUT_DIR = (
    PROCESSED_ROOT
    / "beat2_semantic_event_training_pool_18d_v8"
    / "expansion"
)
DEFAULT_BEAT2_ROOT = Path(
    "/home/gez/nas/cloud/gez/human_motion/raw/BEAT2"
)
DEFAULT_SUPPORTED = (
    INVENTORY_ROOT
    / "beat2_semantic_event_inventory_v1.network_emotion_supported.jsonl"
)
DEFAULT_CANDIDATES = (
    PILOT_ROOT / "beat2_semantic_event_pilot_v7_full.candidates.jsonl"
)
DEFAULT_SELECTOR_SUMMARY = (
    PILOT_ROOT / "beat2_semantic_event_pilot_v7_full.summary.json"
)
DEFAULT_POOL = PILOT_ROOT / "training_pool_low_medium.jsonl"
DEFAULT_FAILED = V7_ROOT / "failed_manifest.jsonl"
DEFAULT_PASSED = V7_ROOT / "passed_manifest.jsonl"
DEFAULT_LOCKED_PASSED_MIN30 = V7_ROOT / "passed_manifest_min30f.jsonl"
DEFAULT_LOCKED_TRAIN_READY = (
    V7_ROOT / "adjudication_min30f" / "train_ready.jsonl"
)
DEFAULT_LOCK = PILOT_ROOT / "motion_only_pretrain_provenance_lock_min30f.json"
DEFAULT_GMR_PYTHON = Path("/home/gez/shuaiwang/.venvs/gmr/bin/python")
REQUIRED_GATES = frozenset(
    {
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
)
FORBIDDEN_TOKEN = "kimodo"
SCHEMA_VERSION = 1
CONTRACT_VERSION = "ula_v2_18d_head_v1"
MOTION_ONLY_EPISODE_CONTRACT = "ula_v2_18d_motion_only_physical_qc_v1"
FORMAL_SEMANTIC_SUPERVISION_MASKS = {
    "official_category": False,
    "robot_observable_motion_form": False,
    "communicative_intent": False,
    "prompt_text": False,
    "legacy_gesture": False,
}
MOTION_ONLY_REQUIRED_RELEASE_INVARIANTS = frozenset(
    {
        "every_record_passes_18d_physical_qc",
        "all_semantic_supervision_masked",
        "all_affect_supervision_masked",
        "all_prompt_latents_masked_zero",
        "all_sequences_native_variable_length",
        "no_fixed_duration_training_unit",
        "no_semantic_review_claimed",
    }
)


def stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite expansion artifact: {path}")
    atomic_text(path, "".join(stable_json(record) + "\n" for record in records))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            records.append(value)
    return records


def clip_id(record: Mapping[str, Any]) -> str:
    value = str(record.get("clip_id") or record.get("task_id") or "").strip()
    if not value:
        raise ValueError("record is missing clip_id/task_id")
    return value


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_strings(child)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for child in value:
            yield from _walk_strings(child)


def assert_no_forbidden_reference(value: Any, *, label: str) -> None:
    for text in _walk_strings(value):
        if FORBIDDEN_TOKEN in text.casefold():
            raise ValueError(f"{label} contains a forbidden cross-dataset reference")


def assert_beat2_record(record: Mapping[str, Any], *, label: str) -> None:
    assert_no_forbidden_reference(record, label=label)
    dataset = record.get("dataset")
    if dataset is not None and dataset != "BEAT2":
        raise ValueError(f"{label} is not a BEAT2 record")
    subset = record.get("dataset_subset")
    if subset is not None and subset != "beat_english_v2.0.0":
        raise ValueError(f"{label} has an unexpected dataset subset")
    group = record.get("source_group_key")
    if group is not None and not str(group).startswith("BEAT2/"):
        raise ValueError(f"{label} has a non-BEAT2 source group")
    source = record.get("source")
    if source is not None:
        source_path = Path(str(source))
        if "BEAT2" not in source_path.parts:
            raise ValueError(f"{label} has a source outside the BEAT2 tree")


def validate_source_record(
    record: Mapping[str, Any], *, verify_artifacts: bool
) -> None:
    key = clip_id(record)
    errors: list[str] = []
    if record.get("status") != "passed":
        errors.append("source status is not passed")
    frames = record.get("frames")
    rate = record.get("fps")
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 2:
        errors.append("frames must be an integer >= 2")
    if (
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not math.isclose(float(rate), 30.0, abs_tol=1e-9)
    ):
        errors.append("fps must be exactly 30 Hz")
    gates = record.get("quality_gate")
    if not isinstance(gates, Mapping) or not REQUIRED_GATES.issubset(gates):
        errors.append("quality gate is incomplete")
    elif any(value is not True for value in gates.values()):
        errors.append("a declared quality gate did not pass")
    segment = record.get("training_segment")
    if not isinstance(segment, Mapping):
        errors.append("training_segment is missing")
    elif (
        segment.get("representation") != "native_variable_length_semantic_clip_v1"
        or segment.get("fixed_window_sec") is not None
    ):
        errors.append("training_segment is not native variable length")
    retarget = record.get("retarget_segment")
    if not isinstance(retarget, Mapping) or retarget.get("cropped") is not False:
        errors.append("retarget segment is missing or cropped")
    if record.get("semantic_supervision_masks") != FORMAL_SEMANTIC_SUPERVISION_MASKS:
        errors.append("semantic supervision masks are not all false")
    for field in (
        "behavior_supervision_mask",
        "emotion_supervision_mask",
        "affect_observable_supervision_mask",
        "official_category_conditioning_enabled",
        "official_emotion_conditioning_enabled",
    ):
        if record.get(field) is not False:
            errors.append(f"{field} is not false")
    for path_field, hash_field in (
        ("safe_csv", "safe_csv_sha256"),
        ("quality_json", "quality_json_sha256"),
    ):
        value = str(record.get(path_field) or "").strip()
        path = Path(value).resolve() if value else None
        expected = record.get(hash_field)
        if path is None or not path.is_file():
            errors.append(f"{path_field} is missing")
        elif not is_sha256(expected):
            errors.append(f"{hash_field} is invalid")
        elif verify_artifacts and sha256_file(path) != expected:
            errors.append(f"{path_field} hash mismatch")
    if errors:
        raise ValueError(f"{key}: " + "; ".join(errors))


def build_record(source: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(source)
    record.update(
        {
            "formal_episode_contract": MOTION_ONLY_EPISODE_CONTRACT,
            "accepted_for_training": True,
            "training_admission_status": "motion_only_physical_qc_train_ready",
            "emotion_conditioning_mask": False,
            "independent_review": {
                "present": False,
                "status": "not_performed_motion_only",
                "training_acceptance": False,
                "scope": "semantic_and_affect_not_required_for_motion_only_loss",
            },
            "adjudication": {
                "status": "motion_only_train_ready",
                "reasons": [],
            },
            "motion_only_admission": {
                "physical_qc_only": True,
                "semantic_review_required": False,
                "independent_semantic_review_claimed": False,
                "text_conditioning_enabled": False,
                "emotion_conditioning_enabled": False,
                "audio_conditioning_enabled": False,
                "native_variable_length": True,
                "fixed_duration_training_unit": False,
                "source_record_sha256": canonical_sha256(source),
            },
            "motion_18d": {
                "state": "passed",
                "partition": "accepted_motion_only",
                "reasons": [],
                "output_contract": CONTRACT_VERSION,
                "action_dim": 18,
                "frames": int(source["frames"]),
                "csv_rows": int(source["frames"]),
                "fps": float(source["fps"]),
                "quality_gate": dict(source["quality_gate"]),
                "quality_json": str(Path(source["quality_json"]).resolve()),
                "quality_sha256": source["quality_json_sha256"],
                "safe_csv": str(Path(source["safe_csv"]).resolve()),
                "safe_csv_sha256": source["safe_csv_sha256"],
                "retarget_segment": dict(source["retarget_segment"]),
                "source_window_frames": int(
                    source["training_segment"]["frame_count"]
                ),
                "upstream_lineage": dict(source.get("lineage_hashes") or {}),
            },
        }
    )
    return record


def build_motion_only_release(
    *,
    passed_manifest: Path,
    output_dir: Path,
    verify_artifacts: bool,
) -> dict[str, Any]:
    passed_manifest = passed_manifest.resolve()
    output_dir = output_dir.resolve()
    output_manifest = output_dir / "train_ready.jsonl"
    report_path = output_dir / "motion_only_release_report.json"
    existing = [str(path) for path in (output_manifest, report_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite motion-only release: {existing}")
    source_records = read_jsonl(passed_manifest)
    if not source_records:
        raise ValueError("passed manifest contains no records")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in source_records:
        validate_source_record(source, verify_artifacts=verify_artifacts)
        key = clip_id(source)
        if key in seen:
            raise ValueError(f"duplicate clip_id: {key}")
        record = build_record(source)
        record["clip_id"] = key
        records.append(record)
        seen.add(key)
    write_jsonl(output_manifest, records)
    frame_counts = [int(record["motion_18d"]["frames"]) for record in records]
    spans = [(frames - 1) / 30.0 for frames in frame_counts]
    report = {
        "schema_version": 1,
        "artifact_kind": "beat2_v8_expansion_motion_only_physical_qc_release_v1",
        "formal_episode_contract": MOTION_ONLY_EPISODE_CONTRACT,
        "conditioning_policy": "all_text_behavior_emotion_affect_channels_masked_zero",
        "source": {
            "passed_manifest": str(passed_manifest),
            "passed_manifest_sha256": sha256_file(passed_manifest),
            "records": len(source_records),
            "artifact_hashes_verified_during_build": bool(verify_artifacts),
        },
        "outputs": {
            "train_ready": {
                "path": str(output_manifest),
                "records": len(records),
                "sha256": sha256_file(output_manifest),
            }
        },
        "scale": {
            "train_ready_clips": len(records),
            "total_frames": sum(frame_counts),
            "total_sample_span_sec": float(sum(spans)),
            "frame_count_min": min(frame_counts),
            "frame_count_median": float(statistics.median(frame_counts)),
            "frame_count_max": max(frame_counts),
            "distinct_frame_count_count": len(set(frame_counts)),
            "speaker_count": len(
                {str(record["speaker_key"]) for record in records}
            ),
            "source_group_count": len(
                {str(record["source_group_key"]) for record in records}
            ),
            "official_split_counts": dict(
                sorted(
                    Counter(
                        str(record.get("official_split")) for record in records
                    ).items()
                )
            ),
        },
        "invariants": {
            name: True
            for name in sorted(MOTION_ONLY_REQUIRED_RELEASE_INVARIANTS)
        },
        "semantic_claims": {
            "text_conditioned_training_ready": False,
            "emotion_conditioned_training_ready": False,
            "captions_may_be_used_as_review_metadata_only": True,
        },
    }
    atomic_json(report_path, report)
    return report


def index_records(
    records: Sequence[dict[str, Any]], *, label: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        key = clip_id(record)
        if key in result:
            raise ValueError(f"{label} contains duplicate clip_id {key}")
        assert_beat2_record(record, label=f"{label}:{key}")
        result[key] = record
    return result


def frame_count(record: Mapping[str, Any]) -> int:
    for value in (
        record.get("frames"),
        (record.get("window") or {}).get("frame_count"),
        (record.get("training_segment") or {}).get("frame_count"),
    ):
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return 0


def fps(record: Mapping[str, Any]) -> float:
    value = record.get("fps", 30.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{clip_id(record)} has invalid fps")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{clip_id(record)} has invalid fps")
    return result


def sample_span_sec(record: Mapping[str, Any]) -> float:
    return max(0, frame_count(record) - 1) / fps(record)


def frame_coverage_sec(record: Mapping[str, Any]) -> float:
    return frame_count(record) / fps(record)


def fixed_split(
    record: Mapping[str, Any], speaker_to_split: Mapping[str, str]
) -> str:
    value = record.get("fixed_split_assignment") or record.get("pilot_split")
    if value is None:
        value = speaker_to_split.get(str(record.get("speaker_key")))
    if value not in {"train", "validation", "test"}:
        raise ValueError(f"{clip_id(record)} has no valid fixed split")
    return str(value)


def stage_summary(
    records: Sequence[dict[str, Any]],
    speaker_to_split: Mapping[str, str],
) -> dict[str, Any]:
    split_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        split_records[fixed_split(record, speaker_to_split)].append(record)

    def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        frames = sum(frame_count(record) for record in rows)
        span = sum(sample_span_sec(record) for record in rows)
        coverage = sum(frame_coverage_sec(record) for record in rows)
        return {
            "records": len(rows),
            "frames": frames,
            "sample_span_sec": span,
            "sample_span_hours": span / 3600.0,
            "frame_coverage_sec": coverage,
            "frame_coverage_hours": coverage / 3600.0,
            "speakers": len({str(record.get("speaker_key")) for record in rows}),
            "source_groups": len(
                {str(record.get("source_group_key")) for record in rows}
            ),
        }

    return {
        **summarize(records),
        "by_fixed_split": {
            split: summarize(split_records.get(split, []))
            for split in ("train", "validation", "test")
        },
    }


def speaker_summary(
    stages: Mapping[str, Sequence[dict[str, Any]]],
    speaker_to_split: Mapping[str, str],
) -> dict[str, Any]:
    speakers = sorted(speaker_to_split)
    result: dict[str, Any] = {}
    for speaker in speakers:
        entry: dict[str, Any] = {"fixed_split": speaker_to_split[speaker]}
        for stage, records in stages.items():
            rows = [
                record
                for record in records
                if str(record.get("speaker_key")) == speaker
            ]
            entry[stage] = {
                "records": len(rows),
                "frames": sum(frame_count(record) for record in rows),
                "sample_span_hours": (
                    sum(sample_span_sec(record) for record in rows) / 3600.0
                ),
            }
        result[speaker] = entry
    return result


def failed_gate_tuple(record: Mapping[str, Any]) -> tuple[str, ...]:
    gates = record.get("quality_gate")
    if not isinstance(gates, Mapping):
        return ()
    return tuple(
        sorted(
            key
            for key, value in gates.items()
            if key != "passed" and value is not True
        )
    )


def process_error_text(
    record: Mapping[str, Any], *, quality_root: Path
) -> str:
    inline = str(record.get("error") or "").strip()
    if inline:
        return inline
    log_value = str(record.get("log_path") or "").strip()
    if not log_value:
        return ""
    log_path = Path(log_value).resolve()
    if quality_root.resolve() not in log_path.parents:
        raise ValueError(f"event log escapes v7 BEAT2 root: {log_path}")
    payload = read_json(log_path)
    assert_no_forbidden_reference(
        payload, label=f"event_log:{clip_id(record)}"
    )
    return str(payload.get("error") or "")


def _metric_bin(value: float, boundaries: Sequence[float]) -> str:
    for boundary in boundaries:
        if value <= boundary:
            return f"<= {boundary:g}"
    return f"> {boundaries[-1]:g}"


def quality_failure_summary(
    failed: Sequence[dict[str, Any]],
    *,
    quality_root: Path,
    speaker_to_split: Mapping[str, str],
) -> dict[str, Any]:
    status_counts = Counter(str(record.get("status")) for record in failed)
    gate_combinations = Counter()
    individual_gates = Counter()
    by_split_status = Counter()
    target_bins = Counter()
    collision_bins = Counter()
    process_errors: list[dict[str, Any]] = []
    for record in failed:
        split = fixed_split(record, speaker_to_split)
        status = str(record.get("status"))
        by_split_status[(split, status)] += 1
        if status != "quality_failed":
            process_errors.append(
                {
                    "clip_id": clip_id(record),
                    "fixed_split_assignment": split,
                    "source_frames_declared": frame_count(record),
                    "error": process_error_text(
                        record, quality_root=quality_root
                    ),
                }
            )
            continue
        bad = failed_gate_tuple(record)
        gate_combinations[bad] += 1
        individual_gates.update(bad)
        quality_path = Path(str(record.get("quality_json") or "")).resolve()
        if quality_root.resolve() not in quality_path.parents:
            raise ValueError(f"quality path escapes v7 BEAT2 root: {quality_path}")
        quality = read_json(quality_path)
        assert_no_forbidden_reference(
            quality, label=f"quality:{clip_id(record)}"
        )
        target_value = quality.get("limb_target_error_p95_m")
        if isinstance(target_value, (int, float)) and not isinstance(
            target_value, bool
        ):
            target_bins[_metric_bin(float(target_value), (0.04, 0.041, 0.045, 0.05, 0.075))] += 1
        collision_value = quality.get("upper_body_collision_frame_rate")
        if isinstance(collision_value, (int, float)) and not isinstance(
            collision_value, bool
        ):
            collision_bins[
                _metric_bin(float(collision_value), (0.05, 0.06, 0.1, 0.25, 0.5))
            ] += 1
    return {
        "status_counts": dict(sorted(status_counts.items())),
        "failed_gate_combinations": {
            "+".join(keys) if keys else "none_recorded": value
            for keys, value in sorted(
                gate_combinations.items(), key=lambda item: (-item[1], item[0])
            )
        },
        "failed_individual_gates": dict(sorted(individual_gates.items())),
        "by_fixed_split_and_status": {
            f"{split}|{status}": value
            for (split, status), value in sorted(by_split_status.items())
        },
        "target_fit_p95_m_histogram_all_quality_failures": dict(target_bins),
        "collision_frame_rate_histogram_all_quality_failures": dict(
            collision_bins
        ),
        "processing_errors": process_errors,
    }


def build_retry_candidates(
    pool_by_id: Mapping[str, dict[str, Any]],
    failed: Sequence[dict[str, Any]],
    speaker_to_split: Mapping[str, str],
    *,
    quality_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retry: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for failure in failed:
        key = clip_id(failure)
        source = pool_by_id[key]
        split = fixed_split(source, speaker_to_split)
        reason: str | None = None
        if split != "train":
            reason = "fixed_eval_split_immutable"
        elif frame_count(failure) and frame_count(failure) < 30:
            reason = "prior_output_below_min30"
        elif (
            failure.get("status") == "event_process_failed"
            and "Invalid semantic event"
            in process_error_text(failure, quality_root=quality_root)
        ):
            reason = "source_bounds_invalid_and_clipped_interval_below_min30"
        if reason:
            excluded.append(
                {
                    "clip_id": key,
                    "fixed_split_assignment": split,
                    "reason": reason,
                }
            )
            continue
        retry.append(source)
    return sorted(retry, key=clip_id), sorted(
        excluded, key=lambda item: item["clip_id"]
    )


def build_no_smoothing_retry_candidates(
    pool_by_id: Mapping[str, dict[str, Any]],
    failed: Sequence[dict[str, Any]],
    speaker_to_split: Mapping[str, str],
    *,
    quality_root: Path,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    retry: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for failure in failed:
        key = clip_id(failure)
        source = pool_by_id[key]
        if fixed_split(source, speaker_to_split) != "train":
            continue
        if frame_count(failure) and frame_count(failure) < 30:
            continue
        status = failure.get("status")
        if status == "event_process_failed":
            error = process_error_text(failure, quality_root=quality_root)
            if "Invalid semantic event" in error:
                continue
            retry.append(source)
            reasons["event_process_limit_boundary_retry"] += 1
            continue
        if status != "quality_failed":
            continue
        quality_path = Path(str(failure.get("quality_json") or "")).resolve()
        if quality_root.resolve() not in quality_path.parents:
            raise ValueError(f"quality path escapes v7 BEAT2 root: {quality_path}")
        quality = read_json(quality_path)
        gates = failure.get("quality_gate") or {}
        raw_fit = quality.get("raw_limb_target_error_p95_m")
        safe_fit = quality.get("limb_target_error_p95_m")
        smoothing_fit_regression = bool(
            isinstance(raw_fit, (int, float))
            and not isinstance(raw_fit, bool)
            and isinstance(safe_fit, (int, float))
            and not isinstance(safe_fit, bool)
            and float(raw_fit) <= 0.04
            and float(safe_fit) > 0.04
        )
        collision_failure = gates.get("collision_pass") is False
        if not smoothing_fit_regression and not collision_failure:
            continue
        retry.append(source)
        if smoothing_fit_regression:
            reasons["raw_fit_pass_but_smoothed_fit_fail"] += 1
        if collision_failure:
            reasons["collision_retry_without_savgol_filter"] += 1
    return sorted(retry, key=clip_id), reasons


def adjacent_short_pairs(
    passed: Sequence[dict[str, Any]],
    speaker_to_split: Mapping[str, str],
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in passed:
        if fixed_split(record, speaker_to_split) != "train":
            continue
        if frame_count(record) >= 30:
            continue
        groups[str(record.get("source_group_key"))].append(record)
    pairs: list[dict[str, Any]] = []
    for source_group, records in groups.items():
        rows = sorted(
            records,
            key=lambda record: (
                int(record.get("start_frame") or 0),
                int(record.get("end_frame_exclusive") or 0),
                clip_id(record),
            ),
        )
        for left, right in zip(rows, rows[1:]):
            left_end = int(left.get("end_frame_exclusive") or 0)
            right_start = int(right.get("start_frame") or 0)
            union_start = int(left.get("start_frame") or 0)
            union_end = int(right.get("end_frame_exclusive") or 0)
            if left_end != right_start or union_end - union_start < 30:
                continue
            pairs.append(
                {
                    "source_group_key": source_group,
                    "left_clip_id": clip_id(left),
                    "right_clip_id": clip_id(right),
                    "start_frame": union_start,
                    "end_frame_exclusive": union_end,
                    "source_frame_count": union_end - union_start,
                    "fixed_split_assignment": "train",
                    "admission": (
                        "candidate_only_requires_union_retarget_and_all_unchanged_qc_gates"
                    ),
                }
            )
    return sorted(
        pairs,
        key=lambda item: (
            item["source_group_key"],
            item["start_frame"],
            item["end_frame_exclusive"],
        ),
    )


def _loss_summary(
    source: Mapping[str, dict[str, Any]],
    target: Mapping[str, dict[str, Any]],
    speaker_to_split: Mapping[str, str],
) -> dict[str, Any]:
    lost = [source[key] for key in sorted(set(source) - set(target))]
    return {
        **stage_summary(lost, speaker_to_split),
        "records": len(lost),
    }


def _assert_subset(
    parent: Mapping[str, Any], child: Mapping[str, Any], *, label: str
) -> None:
    extra = sorted(set(child) - set(parent))
    if extra:
        raise ValueError(f"{label} adds records not in its parent: {extra[:3]}")


def _speaker_mapping(summary: Mapping[str, Any]) -> dict[str, str]:
    mapping = (
        (summary.get("speaker_assignment") or {}).get("speaker_to_split") or {}
    )
    if not isinstance(mapping, dict) or set(mapping.values()) != {
        "train",
        "validation",
        "test",
    }:
        raise ValueError("selector summary has no complete fixed speaker split")
    return {str(key): str(value) for key, value in mapping.items()}


def _command_plan(
    output_dir: Path,
    *,
    beat2_root: Path,
    gmr_python: Path,
) -> dict[str, list[str]]:
    retry_inventory = (
        output_dir / "audit" / "retry_candidates.no_smoothing.train.min30.jsonl"
    )
    broad_inventory = output_dir / "audit" / "retry_candidates.train.min30.jsonl"
    retry_root = output_dir / "retry_no_smoothing"
    return {
        "retry_train_only": [
            str(gmr_python),
            str(
                PROJECT_ROOT
                / "tools/gmr_v2/batch_retarget_beat2_semantic_events_v2.py"
            ),
            "--inventory",
            str(retry_inventory),
            "--beat2-root",
            str(beat2_root),
            "--output-root",
            str(retry_root),
            "--solver",
            "daqp",
            "--neutral-limit-margin-rad",
            "1e-6",
            "--smoothing-window",
            "1",
            "--workers",
            "4",
        ],
        "optional_broad_solver_retry": [
            str(gmr_python),
            str(
                PROJECT_ROOT
                / "tools/gmr_v2/batch_retarget_beat2_semantic_events_v2.py"
            ),
            "--inventory",
            str(broad_inventory),
            "--beat2-root",
            str(beat2_root),
            "--output-root",
            str(output_dir / "retry_quadprog_optional"),
            "--solver",
            "quadprog",
            "--neutral-limit-margin-rad",
            "1e-6",
            "--workers",
            "4",
        ],
        "finalize": [
            "python",
            str(Path(__file__).resolve()),
            "finalize",
            "--output-dir",
            str(output_dir),
            "--retry-passed",
            str(retry_root / "passed_manifest.jsonl"),
            "--verify-artifacts",
        ],
        "verify": [
            "python",
            str(PROJECT_ROOT / "tools/verify_beat2_v8_expansion.py"),
            "--output-dir",
            str(output_dir),
            "--verify-artifacts",
        ],
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite expansion directory: {output_dir}")
    configured_paths = {
        name: str(getattr(args, name).resolve())
        for name in (
            "supported",
            "candidates",
            "selector_summary",
            "pool",
            "failed",
            "passed",
            "locked_passed_min30",
            "locked_train_ready",
            "locked_provenance",
            "beat2_root",
            "gmr_python",
            "output_dir",
        )
    }
    assert_no_forbidden_reference(configured_paths, label="configured paths")
    for name, value in configured_paths.items():
        if name == "output_dir":
            continue
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(path)

    summary = read_json(args.selector_summary)
    speaker_to_split = _speaker_mapping(summary)
    stages = {
        "supported": read_jsonl(args.supported),
        "candidates": read_jsonl(args.candidates),
        "pool": read_jsonl(args.pool),
        "passed": read_jsonl(args.passed),
        "locked_min30": read_jsonl(args.locked_passed_min30),
    }
    indexed = {
        name: index_records(records, label=name)
        for name, records in stages.items()
    }
    failed = read_jsonl(args.failed)
    failed_by_id = index_records(failed, label="failed")
    locked_train_ready = read_jsonl(args.locked_train_ready)
    index_records(locked_train_ready, label="locked_train_ready")
    _assert_subset(indexed["supported"], indexed["candidates"], label="candidates")
    _assert_subset(indexed["candidates"], indexed["pool"], label="pool")
    _assert_subset(indexed["pool"], indexed["passed"], label="passed")
    _assert_subset(indexed["passed"], indexed["locked_min30"], label="locked_min30")
    if set(indexed["pool"]) != set(indexed["passed"]) | set(failed_by_id):
        raise ValueError("passed and failed manifests do not exactly partition the pool")

    provenance = read_json(args.locked_provenance)
    locked_bindings = provenance.get("locked_artifacts") or {}
    expected_locked_hashes = {
        "passed_min30": (
            locked_bindings.get("physical_qc_passed_manifest") or {}
        ).get("sha256"),
        "train_ready": (locked_bindings.get("train_ready_manifest") or {}).get(
            "sha256"
        ),
    }
    actual_locked_hashes = {
        "passed_min30": sha256_file(args.locked_passed_min30),
        "train_ready": sha256_file(args.locked_train_ready),
    }
    if expected_locked_hashes != actual_locked_hashes:
        raise ValueError("locked min30 artifact hash mismatch")

    retry, retry_excluded = build_retry_candidates(
        indexed["pool"],
        failed,
        speaker_to_split,
        quality_root=V7_ROOT,
    )
    no_smoothing_retry, no_smoothing_reasons = (
        build_no_smoothing_retry_candidates(
            indexed["pool"],
            failed,
            speaker_to_split,
            quality_root=V7_ROOT,
        )
    )
    short_pairs = adjacent_short_pairs(stages["passed"], speaker_to_split)
    audit_dir = output_dir / "audit"
    output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(audit_dir / "retry_candidates.train.min30.jsonl", retry)
    write_jsonl(
        audit_dir / "retry_candidates.no_smoothing.train.min30.jsonl",
        no_smoothing_retry,
    )
    write_jsonl(audit_dir / "retry_excluded.jsonl", retry_excluded)
    write_jsonl(audit_dir / "adjacent_short_merge_candidates.train.jsonl", short_pairs)

    losses = {
        "supported_to_candidates": _loss_summary(
            indexed["supported"], indexed["candidates"], speaker_to_split
        ),
        "candidates_to_low_medium_pool": _loss_summary(
            indexed["candidates"], indexed["pool"], speaker_to_split
        ),
        "pool_to_physical_qc_pass": _loss_summary(
            indexed["pool"], indexed["passed"], speaker_to_split
        ),
        "physical_qc_pass_to_locked_min30": _loss_summary(
            indexed["passed"], indexed["locked_min30"], speaker_to_split
        ),
    }
    supported_rejects = [
        indexed["supported"][key]
        for key in sorted(set(indexed["supported"]) - set(indexed["candidates"]))
    ]
    high_fallback = [
        indexed["candidates"][key]
        for key in sorted(set(indexed["candidates"]) - set(indexed["pool"]))
    ]
    losses["supported_to_candidates"]["reason_counts"] = {
        "high_dynamic_above_interaction_energy_ceiling": len(supported_rejects)
    }
    losses["candidates_to_low_medium_pool"]["reason_counts"] = dict(
        Counter(str(record.get("pilot_dynamic_band")) for record in high_fallback)
    )
    short = [
        indexed["passed"][key]
        for key in sorted(set(indexed["passed"]) - set(indexed["locked_min30"]))
    ]
    losses["physical_qc_pass_to_locked_min30"]["frame_count_histogram"] = dict(
        sorted(Counter(frame_count(record) for record in short).items())
    )
    failure = quality_failure_summary(
        failed,
        quality_root=V7_ROOT,
        speaker_to_split=speaker_to_split,
    )
    command_plan = _command_plan(
        output_dir, beat2_root=args.beat2_root, gmr_python=args.gmr_python
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "beat2_v8_expansion_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "dataset": "BEAT2",
            "fixed_split_assignment_immutable": True,
            "test_split_immutable": True,
            "locked_min30_manifest_immutable": True,
            "physical_qc_thresholds_immutable": True,
            "semantic_and_affect_supervision_remain_masked": True,
            "retry_admission_split": "train",
            "minimum_output_frames": 30,
            "forbidden_cross_dataset_reference_count": 0,
        },
        "inputs": {
            name: {"path": value, "sha256": sha256_file(Path(value))}
            for name, value in configured_paths.items()
            if name
            not in {"output_dir", "beat2_root"}
            and Path(value).is_file()
        },
        "locked_artifacts": {
            "hashes_verified": True,
            **actual_locked_hashes,
        },
        "stages": {
            name: stage_summary(records, speaker_to_split)
            for name, records in stages.items()
        },
        "losses": losses,
        "physical_qc_failures": failure,
        "speaker_stage_scale": speaker_summary(stages, speaker_to_split),
        "retry_plan": {
            "strategy": (
                "no_savgol_retry_with_event_local_limit_interiorization"
            ),
            "quality_threshold_change": False,
            "fixed_test_records_selected": 0,
            "candidate_records": len(no_smoothing_retry),
            "candidate_source_sample_span_hours": (
                sum(sample_span_sec(record) for record in no_smoothing_retry)
                / 3600.0
            ),
            "selection_reason_counts": dict(sorted(no_smoothing_reasons.items())),
            "broad_solver_retry_candidate_records": len(retry),
            "broad_solver_retry_source_sample_span_hours": (
                sum(sample_span_sec(record) for record in retry) / 3600.0
            ),
            "excluded_records": len(retry_excluded),
            "inventory": str(
                audit_dir
                / "retry_candidates.no_smoothing.train.min30.jsonl"
            ),
            "commands": command_plan,
        },
        "short_segment_merge_audit": {
            "short_pass_records": len(short),
            "short_pass_sample_span_hours": (
                sum(sample_span_sec(record) for record in short) / 3600.0
            ),
            "strict_contiguous_train_pairs": len(short_pairs),
            "automatic_admission": False,
            "reason": (
                "a merged interval requires a fresh union retarget and all unchanged "
                "physical QC gates; independent safe CSVs are never concatenated"
            ),
            "candidates": str(
                audit_dir / "adjacent_short_merge_candidates.train.jsonl"
            ),
        },
    }
    atomic_json(audit_dir / "expansion_report.json", report)
    atomic_json(audit_dir / "commands.json", command_plan)
    return report


def _validate_recovered_record(
    record: Mapping[str, Any],
    *,
    candidate_ids: set[str],
    verify_artifacts: bool,
) -> None:
    key = clip_id(record)
    assert_beat2_record(record, label=f"recovered:{key}")
    if key not in candidate_ids:
        raise ValueError(f"recovered record was not in the frozen retry inventory: {key}")
    if record.get("fixed_split_assignment") != "train":
        raise ValueError(f"recovered record is not fixed-split train: {key}")
    if frame_count(record) < 30:
        raise ValueError(f"recovered record is shorter than 30 frames: {key}")
    gates = record.get("quality_gate")
    if not isinstance(gates, Mapping) or not REQUIRED_GATES.issubset(gates):
        raise ValueError(f"recovered record has incomplete physical QC: {key}")
    if any(value is not True for value in gates.values()):
        raise ValueError(f"recovered record did not pass every physical QC gate: {key}")
    validate_source_record(record, verify_artifacts=verify_artifacts)


def _canonical_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    records = read_jsonl(path)
    return records, {clip_id(record): stable_json(record) for record in records}


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    audit_report_path = output_dir / "audit" / "expansion_report.json"
    retry_inventory_path = (
        output_dir / "audit" / "retry_candidates.train.min30.jsonl"
    )
    release_dir = output_dir / "release"
    if release_dir.exists():
        raise FileExistsError(f"refusing to overwrite expansion release: {release_dir}")
    report = read_json(audit_report_path)
    assert_no_forbidden_reference(report, label="audit report")
    if report.get("policy", {}).get("test_split_immutable") is not True:
        raise ValueError("audit report does not lock the test split")
    locked = read_json(args.locked_provenance)
    bindings = locked.get("locked_artifacts") or {}
    expected_passed_hash = (
        bindings.get("physical_qc_passed_manifest") or {}
    ).get("sha256")
    expected_train_ready_hash = (
        bindings.get("train_ready_manifest") or {}
    ).get("sha256")
    if sha256_file(args.locked_passed_min30) != expected_passed_hash:
        raise ValueError("locked passed min30 manifest changed")
    if sha256_file(args.locked_train_ready) != expected_train_ready_hash:
        raise ValueError("locked train-ready manifest changed")

    candidates = index_records(
        read_jsonl(retry_inventory_path), label="retry_candidates"
    )
    retry_passed = read_jsonl(args.retry_passed)
    retry_index = index_records(retry_passed, label="retry_passed")
    for record in retry_passed:
        _validate_recovered_record(
            record,
            candidate_ids=set(candidates),
            verify_artifacts=args.verify_artifacts,
        )
    base_passed = read_jsonl(args.locked_passed_min30)
    base_index = index_records(base_passed, label="locked_passed_min30")
    overlap = sorted(set(base_index) & set(retry_index))
    if overlap:
        raise ValueError(f"retry passed records duplicate locked records: {overlap[:3]}")

    release_dir.mkdir(parents=True, exist_ok=False)
    union_path = release_dir / "passed_manifest_min30f_expanded.jsonl"
    union = sorted([*base_passed, *retry_passed], key=clip_id)
    write_jsonl(union_path, union)
    release_report = build_motion_only_release(
        passed_manifest=union_path,
        output_dir=release_dir / "adjudication_min30f",
        verify_artifacts=args.verify_artifacts,
    )
    expanded_train_ready = (
        release_dir / "adjudication_min30f" / "train_ready.jsonl"
    )
    base_ready, base_canonical = _canonical_records(args.locked_train_ready)
    expanded_ready, expanded_canonical = _canonical_records(expanded_train_ready)
    base_ready_by_id = {clip_id(record): record for record in base_ready}
    expanded_ready_by_id = {clip_id(record): record for record in expanded_ready}
    for key, value in base_canonical.items():
        if expanded_canonical.get(key) != value:
            raise ValueError(f"locked base record changed in expansion: {key}")
    base_test = {
        key: value
        for key, value in base_canonical.items()
        if base_ready_by_id[key].get("fixed_split_assignment") == "test"
    }
    expanded_test = {
        key: value
        for key, value in expanded_canonical.items()
        if expanded_ready_by_id[key].get("fixed_split_assignment") == "test"
    }
    if expanded_test != base_test:
        raise ValueError("fixed test records changed")

    selector_summary = read_json(args.selector_summary)
    speaker_to_split = _speaker_mapping(selector_summary)
    base_scale = stage_summary(base_ready, speaker_to_split)
    expanded_scale = stage_summary(expanded_ready, speaker_to_split)
    release_report.update(
        {
            "artifact_kind": "beat2_v8_expansion_motion_only_physical_qc_release_v1",
            "expansion": {
                "base_locked_train_ready": {
                    "path": str(args.locked_train_ready.resolve()),
                    "records": len(base_ready),
                    "sha256": expected_train_ready_hash,
                },
                "retry_passed": {
                    "path": str(args.retry_passed.resolve()),
                    "records": len(retry_passed),
                    "sha256": sha256_file(args.retry_passed),
                },
                "recovered_fixed_split_counts": dict(
                    sorted(
                        Counter(
                            str(record.get("fixed_split_assignment"))
                            for record in retry_passed
                        ).items()
                    )
                ),
                "test_records_unchanged": True,
                "base_records_unchanged": True,
                "physical_qc_thresholds_changed": False,
            },
        }
    )
    atomic_json(
        release_dir / "adjudication_min30f" / "motion_only_release_report.json",
        release_report,
    )
    final_report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "beat2_v8_expansion_final_report_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "accepted_for_training": True,
        "policy": {
            "dataset": "BEAT2",
            "fixed_test_records_unchanged": True,
            "base_locked_records_unchanged": True,
            "physical_qc_thresholds_changed": False,
            "minimum_frames": 30,
            "semantic_and_affect_supervision_masked": True,
            "forbidden_cross_dataset_reference_count": 0,
        },
        "bindings": {
            "audit_report": {
                "path": str(audit_report_path),
                "sha256": sha256_file(audit_report_path),
            },
            "locked_provenance": {
                "path": str(args.locked_provenance.resolve()),
                "sha256": sha256_file(args.locked_provenance),
            },
            "locked_train_ready": {
                "path": str(args.locked_train_ready.resolve()),
                "sha256": expected_train_ready_hash,
            },
            "retry_passed": {
                "path": str(args.retry_passed.resolve()),
                "sha256": sha256_file(args.retry_passed),
            },
            "expanded_passed_min30": {
                "path": str(union_path),
                "sha256": sha256_file(union_path),
            },
            "expanded_train_ready": {
                "path": str(expanded_train_ready),
                "sha256": sha256_file(expanded_train_ready),
            },
        },
        "scale": {
            "base": base_scale,
            "expanded": expanded_scale,
            "added_records": len(expanded_ready) - len(base_ready),
            "added_sample_span_hours": (
                expanded_scale["sample_span_hours"]
                - base_scale["sample_span_hours"]
            ),
            "added_frame_coverage_hours": (
                expanded_scale["frame_coverage_hours"]
                - base_scale["frame_coverage_hours"]
            ),
        },
    }
    assert_no_forbidden_reference(final_report, label="final report")
    atomic_json(release_dir / "expansion_final_report.json", final_report)
    return final_report


def verify_expansion(
    output_dir: Path,
    *,
    locked_provenance: Path = DEFAULT_LOCK,
    locked_train_ready: Path = DEFAULT_LOCKED_TRAIN_READY,
    verify_artifacts: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    final_path = output_dir / "release" / "expansion_final_report.json"
    final = read_json(final_path)
    assert_no_forbidden_reference(final, label="final report")
    if final.get("accepted_for_training") is not True:
        raise ValueError("expansion is not accepted for training")
    policy = final.get("policy") or {}
    required_policy = {
        "dataset": "BEAT2",
        "fixed_test_records_unchanged": True,
        "base_locked_records_unchanged": True,
        "physical_qc_thresholds_changed": False,
        "minimum_frames": 30,
        "semantic_and_affect_supervision_masked": True,
        "forbidden_cross_dataset_reference_count": 0,
    }
    for key, expected in required_policy.items():
        if policy.get(key) != expected:
            raise ValueError(f"expansion policy mismatch for {key}")
    bindings = final.get("bindings") or {}
    for name, binding in bindings.items():
        path = Path(str((binding or {}).get("path") or "")).resolve()
        expected = (binding or {}).get("sha256")
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"expansion binding mismatch: {name}")
    locked = read_json(locked_provenance)
    expected_locked = (
        (locked.get("locked_artifacts") or {}).get("train_ready_manifest") or {}
    ).get("sha256")
    if sha256_file(locked_train_ready) != expected_locked:
        raise ValueError("locked train-ready artifact changed")

    expanded_path = Path(
        bindings["expanded_train_ready"]["path"]
    ).resolve()
    expanded = read_jsonl(expanded_path)
    base = read_jsonl(locked_train_ready)
    expanded_by_id = index_records(expanded, label="expanded_train_ready")
    base_by_id = index_records(base, label="locked_train_ready")
    if not set(base_by_id).issubset(expanded_by_id):
        raise ValueError("expanded release dropped locked records")
    for key, record in base_by_id.items():
        if stable_json(expanded_by_id[key]) != stable_json(record):
            raise ValueError(f"expanded release changed locked record: {key}")
    added = [
        record for key, record in expanded_by_id.items() if key not in base_by_id
    ]
    for record in added:
        if record.get("fixed_split_assignment") != "train":
            raise ValueError("expanded release added a non-train record")
        if frame_count(record) < 30:
            raise ValueError("expanded release added a sub-min30 record")
        gates = (record.get("motion_18d") or {}).get("quality_gate")
        if not isinstance(gates, Mapping) or any(
            gates.get(key) is not True for key in REQUIRED_GATES
        ):
            raise ValueError("expanded release added a record without full QC")
        if verify_artifacts:
            motion = record.get("motion_18d") or {}
            for path_key, hash_key in (
                ("safe_csv", "safe_csv_sha256"),
                ("quality_json", "quality_sha256"),
            ):
                path = Path(str(motion.get(path_key) or "")).resolve()
                if not path.is_file() or sha256_file(path) != motion.get(hash_key):
                    raise ValueError(
                        f"expanded artifact mismatch for {clip_id(record)}:{path_key}"
                    )
    base_test = {
        key: stable_json(record)
        for key, record in base_by_id.items()
        if record.get("fixed_split_assignment") == "test"
    }
    expanded_test = {
        key: stable_json(record)
        for key, record in expanded_by_id.items()
        if record.get("fixed_split_assignment") == "test"
    }
    if expanded_test != base_test:
        raise ValueError("expanded test split changed")
    return {
        "verified": True,
        "base_records": len(base),
        "expanded_records": len(expanded),
        "added_train_records": len(added),
        "test_records_unchanged": True,
        "expanded_train_ready_sha256": sha256_file(expanded_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--supported", type=Path, default=DEFAULT_SUPPORTED)
    audit_parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    audit_parser.add_argument(
        "--selector-summary", type=Path, default=DEFAULT_SELECTOR_SUMMARY
    )
    audit_parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    audit_parser.add_argument("--failed", type=Path, default=DEFAULT_FAILED)
    audit_parser.add_argument("--passed", type=Path, default=DEFAULT_PASSED)
    audit_parser.add_argument(
        "--locked-passed-min30", type=Path, default=DEFAULT_LOCKED_PASSED_MIN30
    )
    audit_parser.add_argument(
        "--locked-train-ready", type=Path, default=DEFAULT_LOCKED_TRAIN_READY
    )
    audit_parser.add_argument(
        "--locked-provenance", type=Path, default=DEFAULT_LOCK
    )
    audit_parser.add_argument("--beat2-root", type=Path, default=DEFAULT_BEAT2_ROOT)
    audit_parser.add_argument("--gmr-python", type=Path, default=DEFAULT_GMR_PYTHON)
    audit_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    finalize_parser.add_argument("--retry-passed", type=Path, required=True)
    finalize_parser.add_argument(
        "--selector-summary", type=Path, default=DEFAULT_SELECTOR_SUMMARY
    )
    finalize_parser.add_argument(
        "--locked-passed-min30", type=Path, default=DEFAULT_LOCKED_PASSED_MIN30
    )
    finalize_parser.add_argument(
        "--locked-train-ready", type=Path, default=DEFAULT_LOCKED_TRAIN_READY
    )
    finalize_parser.add_argument(
        "--locked-provenance", type=Path, default=DEFAULT_LOCK
    )
    finalize_parser.add_argument("--verify-artifacts", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        result = audit(args)
    elif args.command == "finalize":
        result = finalize(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
