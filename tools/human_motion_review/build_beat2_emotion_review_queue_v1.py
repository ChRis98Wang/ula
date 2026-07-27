#!/usr/bin/env python3
"""Build a balanced, fail-closed BEAT2 robot-affect human-review queue.

This tool does not create emotion labels and does not admit anything for
training.  Official BEAT2 filename emotion metadata is used only to stratify
the controller-side sample.  Every selected natural expression turn remains
masked until two independent reviewers agree, or a distinct third reviewer
adjudicates a disagreement in a later (separate) workflow.

To prevent copied-window and source leakage, at most one turn is selected from
each ``source_group_key`` across the complete queue.  The queue references the
already physical-QC-passed trajectory in place and never copies or rewrites it.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "1.0.0"
CONFIG_KIND = "beat2_robot_observable_emotion_review_queue_config_v1"
QUEUE_KIND = "beat2_robot_observable_emotion_review_queue_record_v1"
AUDIT_KIND = "beat2_robot_observable_emotion_review_queue_audit_v1"
SOURCE_ARTIFACT_KIND = "ula_v2_18d_expression_turn_retarget_v1"
SOURCE_LABEL_ROLE = (
    "official_beat2_metadata_for_balanced_sampling_only_"
    "not_robot_observable_emotion_supervision"
)
REVIEW_PROTOCOL = "two_independent_blind_robot_affect_reviews_then_adjudication_v1"
NATIVE_DURATION_POLICY = "natural_rest_to_natural_rest_no_fixed_or_max_duration"

EMOTIONS = ("neutral", "sad", "happy", "angry", "surprise", "fear")
SOURCE_EMOTION_LABELS = {
    "neutral": "neutral",
    "sad": "sadness",
    "happy": "happiness",
    "angry": "anger",
    "surprise": "surprise",
    "fear": "fear",
}
SPLITS = ("train", "validation", "test")
DURATION_BINS = (
    "short_under_3s",
    "medium_3_to_6s",
    "long_over_6s",
)
OBSERVABILITY_OPTIONS = ("observable", "not_observable", "ambiguous")
TARGETS = {
    "train": {emotion: 100 for emotion in EMOTIONS},
    "validation": {emotion: 30 for emotion in EMOTIONS},
    "test": {emotion: 50 for emotion in EMOTIONS},
}
MAX_TOTAL = 1080


def stable_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_text(path, "".join(stable_json(row) + "\n" for row in rows))


def read_jsonl_bound(path: Path) -> list[tuple[dict[str, Any], str]]:
    rows: list[tuple[dict[str, Any], str]] = []
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            payload = raw[:-1] if raw.endswith(b"\n") else raw
            if payload.endswith(b"\r"):
                raise ValueError(f"CRLF JSONL is not accepted: {path}:{line_number}")
            if not payload.strip():
                continue
            try:
                row = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid JSONL: {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"expected object: {path}:{line_number}")
            rows.append((row, hashlib.sha256(payload).hexdigest()))
    return rows


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _hashed_group(prefix: str, value: str) -> str:
    return f"{prefix}_{text_sha256(value)[:16]}"


def _category_key(row: Mapping[str, Any]) -> str:
    expression_turn = row.get("expression_turn")
    categories = (
        expression_turn.get("official_categories")
        if isinstance(expression_turn, Mapping)
        else None
    )
    if not isinstance(categories, list):
        return "unavailable"
    clean = sorted(
        {
            value.strip()
            for value in categories
            if isinstance(value, str) and value.strip()
        }
    )
    return "+".join(clean) if clean else "unavailable"


def _duration_bin(row: Mapping[str, Any]) -> str:
    value = row.get("duration_band")
    if value in DURATION_BINS:
        return str(value)
    duration = row.get("duration_sec")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise ValueError(f"{row.get('task_id')}: missing duration bin")
    if float(duration) < 3.0:
        return DURATION_BINS[0]
    if float(duration) <= 6.0:
        return DURATION_BINS[1]
    return DURATION_BINS[2]


@dataclass(frozen=True)
class Candidate:
    row: dict[str, Any]
    source_line_sha256: str
    task_id: str
    split: str
    emotion: str
    speaker: str
    source_group: str
    category: str
    duration_bin: str
    duration_sec: float
    trajectory_path: Path
    trajectory_sha256: str
    source_interval: tuple[int, int]


def _candidate(row: dict[str, Any], source_line_sha256: str) -> Candidate:
    task_id = row.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("source row has no task_id")
    exact = {
        "artifact_kind": row.get("artifact_kind") == SOURCE_ARTIFACT_KIND,
        "status": row.get("status") == "passed",
        "dataset": row.get("dataset") == "BEAT2",
        "accepted_for_training": row.get("accepted_for_training") is False,
        "emotion_supervision_mask": row.get("emotion_supervision_mask") is False,
        "official_emotion_conditioning_enabled": (
            row.get("official_emotion_conditioning_enabled") is False
        ),
        "affect_observable_supervision_mask": (
            row.get("affect_observable_supervision_mask") is False
        ),
        "canonical_prompt": row.get("canonical_prompt") is None,
        "canonical_action": row.get("canonical_action") is None,
    }
    failed = sorted(name for name, passed in exact.items() if not passed)
    if failed:
        raise ValueError(f"{task_id}: source is not fail-closed BEAT2: {failed}")

    semantic_masks = row.get("semantic_supervision_masks")
    if (
        not isinstance(semantic_masks, dict)
        or any(value is not False for value in semantic_masks.values())
    ):
        raise ValueError(f"{task_id}: semantic supervision is not fully masked")
    quality_gate = row.get("quality_gate")
    if (
        not isinstance(quality_gate, dict)
        or quality_gate.get("passed") is not True
        or any(value is not True for value in quality_gate.values())
    ):
        raise ValueError(f"{task_id}: physical quality gate is not fully passed")

    split = row.get("fixed_split_assignment")
    if split not in SPLITS:
        raise ValueError(f"{task_id}: invalid fixed speaker-disjoint split")
    speaker = row.get("speaker_key")
    source_group = row.get("source_group_key")
    if not isinstance(speaker, str) or not speaker:
        raise ValueError(f"{task_id}: missing speaker_key")
    if not isinstance(source_group, str) or not source_group:
        raise ValueError(f"{task_id}: missing source_group_key")

    emotion = row.get("emotion_id")
    if (
        emotion not in EMOTIONS
        or row.get("source_emotion_label") != SOURCE_EMOTION_LABELS.get(emotion)
        or row.get("source_emotion_label_verified") is not True
        or row.get("emotion_label_source") != "official_beat2_filename_protocol"
    ):
        raise ValueError(f"{task_id}: invalid official emotion sampling metadata")

    fps = row.get("fps")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isclose(float(fps), 30.0, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise ValueError(f"{task_id}: expression turn is not 30 Hz")
    training = row.get("training_segment")
    retarget = row.get("retarget_segment")
    if (
        not isinstance(training, dict)
        or training.get("duration_policy") != NATIVE_DURATION_POLICY
        or training.get("fixed_window_sec") is not None
        or training.get("cropped") is not False
        or not isinstance(retarget, dict)
        or retarget.get("duration_policy") != NATIVE_DURATION_POLICY
        or retarget.get("fixed_target_duration_sec") is not None
        or retarget.get("cropped") is not False
    ):
        raise ValueError(f"{task_id}: source is not a complete native-length turn")
    start = training.get("start_frame")
    end = training.get("end_frame_exclusive")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or end <= start
        or training.get("frame_count") != end - start
    ):
        raise ValueError(f"{task_id}: invalid natural source interval")

    duration = row.get("duration_sec")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or float(duration) <= 0.0
    ):
        raise ValueError(f"{task_id}: invalid duration")
    trajectory_value = row.get("safe_csv")
    trajectory_sha = row.get("safe_csv_sha256")
    if (
        not isinstance(trajectory_value, str)
        or not trajectory_value
        or not _is_sha256(trajectory_sha)
    ):
        raise ValueError(f"{task_id}: invalid trajectory binding")
    trajectory_path = Path(trajectory_value).resolve()
    if not trajectory_path.is_file():
        raise ValueError(f"{task_id}: trajectory is unavailable")

    return Candidate(
        row=row,
        source_line_sha256=source_line_sha256,
        task_id=task_id,
        split=str(split),
        emotion=str(emotion),
        speaker=speaker,
        source_group=source_group,
        category=_category_key(row),
        duration_bin=_duration_bin(row),
        duration_sec=float(duration),
        trajectory_path=trajectory_path,
        trajectory_sha256=str(trajectory_sha),
        source_interval=(start, end),
    )


def _require_fixed_split_integrity(candidates: list[Candidate]) -> dict[str, Any]:
    speaker_splits: dict[str, set[str]] = defaultdict(set)
    group_splits: dict[str, set[str]] = defaultdict(set)
    task_ids: set[str] = set()
    for item in candidates:
        if item.task_id in task_ids:
            raise ValueError(f"duplicate task_id: {item.task_id}")
        task_ids.add(item.task_id)
        speaker_splits[item.speaker].add(item.split)
        group_splits[item.source_group].add(item.split)
    bad_speakers = sorted(key for key, values in speaker_splits.items() if len(values) != 1)
    bad_groups = sorted(key for key, values in group_splits.items() if len(values) != 1)
    if bad_speakers:
        raise ValueError("speaker leakage across fixed splits")
    if bad_groups:
        raise ValueError("source-group leakage across fixed splits")
    return {
        "speaker_count": len(speaker_splits),
        "source_group_count": len(group_splits),
        "speaker_split_conflicts": 0,
        "source_group_split_conflicts": 0,
    }


def _target_rows(config: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    targets = config.get("targets")
    if targets != TARGETS:
        raise ValueError("config targets must equal the fixed 600/180/300 six-emotion plan")
    if config.get("max_total") != MAX_TOTAL:
        raise ValueError("config max_total must be 1080")
    return {split: dict(values) for split, values in TARGETS.items()}


def _tie(seed: int, item: Candidate) -> str:
    return text_sha256(f"{seed}\0{item.source_line_sha256}\0{item.task_id}")


def select_candidates(
    candidates: list[Candidate],
    *,
    targets: Mapping[str, Mapping[str, int]],
    seed: int,
) -> tuple[list[Candidate], dict[tuple[str, str], int]]:
    """Greedily balance speaker, category, and duration under hard uniqueness."""

    by_stratum: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for item in candidates:
        by_stratum[(item.split, item.emotion)].append(item)
    capacity = {
        key: len({item.source_group for item in values})
        for key, values in by_stratum.items()
    }
    strata = [(split, emotion) for split in SPLITS for emotion in EMOTIONS]
    strata.sort(
        key=lambda key: (
            capacity.get(key, 0) / max(1, int(targets[key[0]][key[1]])),
            SPLITS.index(key[0]),
            EMOTIONS.index(key[1]),
        )
    )

    selected: list[Candidate] = []
    used_groups: set[str] = set()
    used_trajectory_hashes: set[str] = set()
    used_intervals: set[tuple[str, int, int]] = set()
    for split, emotion in strata:
        speaker_counts: Counter[str] = Counter()
        category_counts: Counter[str] = Counter()
        duration_counts: Counter[str] = Counter()
        joint_counts: Counter[tuple[str, str, str]] = Counter()
        requested = int(targets[split][emotion])
        while sum(
            item.split == split and item.emotion == emotion for item in selected
        ) < requested:
            available = [
                item
                for item in by_stratum.get((split, emotion), [])
                if item.source_group not in used_groups
                and item.trajectory_sha256 not in used_trajectory_hashes
                and (
                    item.source_group,
                    item.source_interval[0],
                    item.source_interval[1],
                )
                not in used_intervals
            ]
            if not available:
                break
            item = min(
                available,
                key=lambda value: (
                    speaker_counts[value.speaker],
                    category_counts[value.category],
                    duration_counts[value.duration_bin],
                    joint_counts[
                        (value.speaker, value.category, value.duration_bin)
                    ],
                    _tie(seed, value),
                ),
            )
            selected.append(item)
            used_groups.add(item.source_group)
            used_trajectory_hashes.add(item.trajectory_sha256)
            used_intervals.add(
                (item.source_group, item.source_interval[0], item.source_interval[1])
            )
            speaker_counts[item.speaker] += 1
            category_counts[item.category] += 1
            duration_counts[item.duration_bin] += 1
            joint_counts[(item.speaker, item.category, item.duration_bin)] += 1
    return selected, capacity


def _review_slot(slot: str) -> dict[str, Any]:
    return {
        "slot": slot,
        "reviewer_id": None,
        "observability": None,
        "observed_emotion": None,
        "confidence": None,
        "submitted_at_utc": None,
        "status": "pending_independent_review",
    }


def _queue_record(item: Candidate, *, seed: int) -> dict[str, Any]:
    sample_id = "beat2_affect_" + text_sha256(
        f"{seed}\0{item.source_line_sha256}\0{item.trajectory_sha256}"
    )[:24]
    expression_sha = item.row.get("expression_turn_record_sha256")
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": QUEUE_KIND,
        "sample_id": sample_id,
        "dataset": "BEAT2",
        "controller_only_render_queue": True,
        "reviewers_must_not_receive_controller_queue": True,
        "reviewer_visible_projection_required": True,
        "fixed_split_assignment": item.split,
        "speaker_group_token": _hashed_group("speaker", item.speaker),
        "source_group_token": _hashed_group("source", item.source_group),
        "source_task_token": _hashed_group("turn", item.task_id),
        "source_record_line_sha256": item.source_line_sha256,
        "expression_turn_record_sha256": (
            expression_sha if _is_sha256(expression_sha) else None
        ),
        "source_interval": {
            "start_frame": item.source_interval[0],
            "end_frame_exclusive": item.source_interval[1],
        },
        "trajectory_path": str(item.trajectory_path),
        "trajectory_sha256": item.trajectory_sha256,
        "trajectory_reference_policy": (
            "reference_existing_physical_qc_pass_only_no_copy_no_rewindow"
        ),
        "trajectory_copied": False,
        "fps": 30.0,
        "duration_sec": item.duration_sec,
        "duration_bin": item.duration_bin,
        "gesture_category_balance_key": item.category,
        "source_official_emotion_exposed_to_reviewers": False,
        "official_emotion_field_present": False,
        "official_emotion_is_trusted_supervision": False,
        "automated_emotion_label_assigned": False,
        "review_protocol": REVIEW_PROTOCOL,
        "allowed_observability": list(OBSERVABILITY_OPTIONS),
        "allowed_observed_emotions": list(EMOTIONS),
        "primary_reviews": [
            _review_slot("reviewer_1"),
            _review_slot("reviewer_2"),
        ],
        "primary_reviewer_ids_must_be_distinct": True,
        "primary_agreement_required": True,
        "third_adjudication_required_on_any_primary_disagreement": True,
        "third_adjudication": {
            "reviewer_id": None,
            "must_differ_from_primary_reviewers": True,
            "observability": None,
            "observed_emotion": None,
            "confidence": None,
            "submitted_at_utc": None,
            "status": "not_requested_pending_primary_reviews",
        },
        "review_state": "pending_two_independent_reviews",
        "supervision_gate_status": (
            "closed_pending_primary_agreement_or_distinct_third_adjudication"
        ),
        "emotion_supervision_mask": False,
        "affect_observable_supervision_mask": False,
        "official_emotion_conditioning_enabled": False,
        "accepted_for_training": False,
    }


def _nested_counts(
    selected: Iterable[Candidate], attribute: str
) -> dict[str, dict[str, dict[str, int]]]:
    result: dict[str, dict[str, dict[str, int]]] = {}
    selected_list = list(selected)
    for split in SPLITS:
        result[split] = {}
        for emotion in EMOTIONS:
            counts = Counter(
                str(getattr(item, attribute))
                for item in selected_list
                if item.split == split and item.emotion == emotion
            )
            result[split][emotion] = dict(sorted(counts.items()))
    return result


def _split_emotion_counts(
    candidates: Iterable[Candidate], *, unique_groups: bool
) -> dict[str, dict[str, int]]:
    values = list(candidates)
    result: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        result[split] = {}
        for emotion in EMOTIONS:
            matched = [
                item
                for item in values
                if item.split == split and item.emotion == emotion
            ]
            result[split][emotion] = (
                len({item.source_group for item in matched})
                if unique_groups
                else len(matched)
            )
    return result


def _resolve_path(config_path: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing config path: {field}")
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def build_from_config(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable config: {config_path}") from error
    if not isinstance(config, dict) or config.get("artifact_kind") != CONFIG_KIND:
        raise ValueError("invalid review queue config")
    targets = _target_rows(config)
    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("config seed must be an integer")
    source_manifest = _resolve_path(
        config_path, config.get("source_manifest"), field="source_manifest"
    )
    output_manifest = _resolve_path(
        config_path, config.get("output_manifest"), field="output_manifest"
    )
    output_audit = _resolve_path(
        config_path, config.get("output_audit"), field="output_audit"
    )
    if output_manifest == source_manifest or output_audit == source_manifest:
        raise ValueError("outputs must not overwrite the source manifest")
    expected_source_sha = config.get("source_manifest_sha256")
    actual_source_sha = sha256_file(source_manifest)
    if not _is_sha256(expected_source_sha) or expected_source_sha != actual_source_sha:
        raise ValueError("source manifest SHA256 does not match config")
    if config.get("allowed_dataset") != "BEAT2":
        raise ValueError("the only allowed dataset is BEAT2")
    if config.get("source_group_policy") != "one_turn_per_source_group":
        raise ValueError("source_group_policy must be one_turn_per_source_group")

    bound_rows = read_jsonl_bound(source_manifest)
    candidates = [_candidate(row, line_sha) for row, line_sha in bound_rows]
    if not candidates:
        raise ValueError("source manifest contains no candidates")
    split_integrity = _require_fixed_split_integrity(candidates)
    selected, capacity = select_candidates(candidates, targets=targets, seed=seed)
    queue = [_queue_record(item, seed=seed) for item in selected]
    queue.sort(
        key=lambda row: (
            SPLITS.index(str(row["fixed_split_assignment"])),
            str(row["sample_id"]),
        )
    )

    groups = [item.source_group for item in selected]
    trajectories = [item.trajectory_sha256 for item in selected]
    intervals = [
        (item.source_group, item.source_interval[0], item.source_interval[1])
        for item in selected
    ]
    if len(groups) != len(set(groups)):
        raise AssertionError("selection violated source-group uniqueness")
    if len(trajectories) != len(set(trajectories)):
        raise AssertionError("selection contains copied trajectory content")
    if len(intervals) != len(set(intervals)):
        raise AssertionError("selection contains duplicate source windows")
    for item in selected:
        if sha256_file(item.trajectory_path) != item.trajectory_sha256:
            raise ValueError(f"{item.task_id}: selected trajectory hash mismatch")

    atomic_jsonl(output_manifest, queue)
    selected_counts = _split_emotion_counts(selected, unique_groups=False)
    raw_counts = _split_emotion_counts(candidates, unique_groups=False)
    unique_capacity = _split_emotion_counts(candidates, unique_groups=True)
    target_total = sum(sum(values.values()) for values in targets.values())
    selected_total = len(selected)
    shortfalls = {
        split: {
            emotion: targets[split][emotion] - selected_counts[split][emotion]
            for emotion in EMOTIONS
        }
        for split in SPLITS
    }
    audit = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": AUDIT_KIND,
        "validation_passed": True,
        "controller_only_contains_official_sampling_aggregates": True,
        "reviewer_distribution_forbidden": True,
        "source": {
            "dataset": "BEAT2",
            "manifest": str(source_manifest),
            "manifest_sha256": actual_source_sha,
            "records": len(candidates),
            "artifact_kind": SOURCE_ARTIFACT_KIND,
            "external_dataset_records": 0,
            "official_emotion_role": SOURCE_LABEL_ROLE,
        },
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "seed": seed,
        },
        "requested": {
            "max_total": MAX_TOTAL,
            "target_total": target_total,
            "targets_by_split_emotion": targets,
            "target_split_counts": {
                split: sum(targets[split].values()) for split in SPLITS
            },
        },
        "availability": {
            "raw_turn_counts_by_split_emotion": raw_counts,
            "unique_source_group_capacity_by_split_emotion": unique_capacity,
        },
        "selection": {
            "records": selected_total,
            "complete_target_met": selected_total == target_total,
            "counts_by_split_emotion": selected_counts,
            "counts_by_split": dict(
                sorted(Counter(item.split for item in selected).items())
            ),
            "shortfall_by_split_emotion": shortfalls,
            "shortfall_total": target_total - selected_total,
            "shortfall_policy": (
                "never_fill_with_second_turn_from_same_source_group"
            ),
            "capacity_diagnostic": {
                f"{split}/{emotion}": capacity.get((split, emotion), 0)
                for split in SPLITS
                for emotion in EMOTIONS
            },
            "speaker_balance_by_split_emotion": _nested_counts(
                selected, "speaker"
            ),
            "gesture_category_balance_by_split_emotion": _nested_counts(
                selected, "category"
            ),
            "duration_balance_by_split_emotion": _nested_counts(
                selected, "duration_bin"
            ),
        },
        "leakage_and_copy_audit": {
            **split_integrity,
            "selected_source_groups": len(set(groups)),
            "selected_records": selected_total,
            "one_turn_per_source_group": len(set(groups)) == selected_total,
            "unique_trajectory_hashes": len(set(trajectories)) == selected_total,
            "unique_source_intervals": len(set(intervals)) == selected_total,
            "trajectory_files_copied": 0,
            "source_windows_created": 0,
            "fixed_speaker_disjoint_split_preserved": True,
        },
        "review_contract": {
            "protocol": REVIEW_PROTOCOL,
            "official_source_labels_trusted": False,
            "automated_labels_created": 0,
            "minimum_independent_primary_reviewers": 2,
            "primary_reviewer_ids_must_be_distinct": True,
            "allowed_observability": list(OBSERVABILITY_OPTIONS),
            "allowed_observed_emotions": list(EMOTIONS),
            "observable_requires_same_observed_emotion": True,
            "any_primary_disagreement_requires_distinct_third_adjudicator": True,
            "third_adjudicator_must_differ_from_primary_reviewers": True,
            "supervision_may_only_be_considered_after_agreement_or_adjudication": True,
            "emotion_supervision_masks_enabled": 0,
            "affect_observable_supervision_masks_enabled": 0,
            "accepted_for_training_records": 0,
        },
        "output": {
            "manifest": str(output_manifest),
            "manifest_sha256": sha256_file(output_manifest),
            "audit": str(output_audit),
        },
    }
    atomic_json(output_audit, audit)
    return audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audit = build_from_config(args.config)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
