#!/usr/bin/env python3
"""Run resumable 15D migration, 18D head pretrain, and joint post-training."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from upper_body_skeleton.ula_v2_18d_head import (
    LEGACY_ACTION_DIM,
    attach_condition_cache,
    build_condition_cache,
    compute_18d_action_stats,
    load_18d_episodes,
    load_condition_cache,
    load_contract_checkpoint,
    migrate_15d_checkpoint,
    sha256_file,
    validate_condition_cache_for_generator,
)
from upper_body_skeleton.ula_v2_18d_posttrain import (
    load_kimodo_replay_splits,
    resolve_posttrain_config,
    strict_group_split,
    train_18d_posttrain,
)
from upper_body_skeleton.ula_training import KIMODO_V2_CONDITION_DIM


SCHEMA_VERSION = 1
STAGES = ("migrate", "head-cache", "head-pretrain", "joint-cache", "joint")


def _json_hash(value) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_hash(value) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _raw_episode_id(episode: Mapping) -> str:
    for field in ("clip_id", "episode_id", "episode_index"):
        value = episode.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError("episode is missing clip_id/episode_id/episode_index")


def _episode_sample_span(episode: Mapping) -> tuple[float, float]:
    fps = float(episode.get("fps") or 30.0)
    frame_count = int(np.asarray(episode["actions"]).shape[0])
    if not np.isfinite(fps) or fps <= 0.0 or frame_count < 2:
        raise ValueError(f"{_raw_episode_id(episode)}: invalid fps/frame count")
    sample_span = float((frame_count - 1) / fps)
    declared = episode.get("duration_sec")
    if declared is not None and not np.isclose(
        float(declared), sample_span, rtol=0.0, atol=1e-6
    ):
        raise ValueError(
            f"{_raw_episode_id(episode)}: duration_sec must equal "
            "(frame_count-1)/fps; N/fps is frame coverage only"
        )
    return fps, sample_span


def episodes_content_contract(episodes: Sequence[Mapping]) -> dict:
    semantic_fields = (
        "behavior_id",
        "emotion_id",
        "intent",
        "observed_affect",
        "motion_style",
        "source_motion_style",
        "semantic_gesture",
    )
    records = []
    for episode in sorted(episodes, key=lambda row: str(row["clip_id"])):
        prompt = str(episode.get("prompt") or "")
        fps, duration = _episode_sample_span(episode)
        records.append(
            {
                "clip_id": str(episode["clip_id"]),
                "dataset_source": _episode_dataset_source_for_receipt(episode),
                "actions_sha256": _array_hash(episode["actions"]),
                "trajectory_sha256": episode.get("trajectory_sha256"),
                "prompt_sha256": episode.get("prompt_sha256")
                or hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "semantics": {
                    field: episode.get(field) for field in semantic_fields
                },
                "fps": fps,
                "duration_sec": duration,
                "duration_time_axis": "sample_span=(frame_count-1)/fps",
                "frame_coverage_sec": float(len(episode["actions"]) / fps),
                "speaker_key": str(episode.get("speaker_key") or ""),
                "source_group_key": str(episode.get("source_group_key") or ""),
                "fixed_split_assignment": episode.get("fixed_split_assignment"),
                "temporal_quarantine_challenge": bool(
                    episode.get("temporal_quarantine_challenge", False)
                ),
                "temporal_quarantine_failed_gates": list(
                    episode.get("temporal_quarantine_failed_gates") or []
                ),
            }
        )
    contract = {
        "contract_type": "ula_v2_18d_episode_content",
        "contract_version": 1,
        "episode_count": len(records),
        "records": records,
    }
    contract["sha256"] = _json_hash(contract)
    return contract


def replay_content_contract(episodes: Sequence[Mapping], *, role: str) -> dict:
    records = []
    for episode in sorted(episodes, key=_raw_episode_id):
        fps, duration = _episode_sample_span(episode)
        records.append({
            "episode_id": _raw_episode_id(episode),
            "actions_sha256": _array_hash(episode["actions"]),
            "condition_sha256": _array_hash(episode["condition"]),
            "source_split": episode.get("replay_source_split"),
            "fps": fps,
            "effective_duration_sec": duration,
            "duration_time_axis": "sample_span=(frame_count-1)/fps",
            "frame_coverage_sec": float(
                np.asarray(episode["actions"]).shape[0] / fps
            ),
        })
    contract = {
        "contract_type": "ula_v2_kimodo_replay_content",
        "contract_version": 1,
        "role": str(role),
        "episode_count": len(records),
        "records": records,
    }
    contract["sha256"] = _json_hash(contract)
    return contract


def _atomic_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_config(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("staged training config must contain a JSON object")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if value.get("audio_policy") != "disabled_not_loaded":
        raise ValueError("current staged training requires audio_policy=disabled_not_loaded")
    training_scope = value.get("training_scope")
    if training_scope == "formal_variable_length_semantic_units":
        raise ValueError(
            "formal variable-length training is not implemented: require real 30 Hz "
            "length bucketing or temporal masks plus full-semantic-unit boundaries"
        )
    if (
        training_scope != "head_mechanism_experiment_only"
        or value.get("formal_training_enabled") is not False
    ):
        raise ValueError(
            "this staged runner is restricted to head_mechanism_experiment_only"
        )
    if not isinstance(value.get("motion_sources"), list) or not value["motion_sources"]:
        raise ValueError("motion_sources must be a non-empty list")
    if any(
        source.get("temporal_unit") != "fixed_window_experimental"
        for source in value["motion_sources"]
        if isinstance(source, Mapping)
    ):
        raise ValueError(
            "experimental staged sources must declare temporal_unit=fixed_window_experimental"
        )
    for name in ("base_15d_checkpoint", "qwen_checkpoint", "output_root"):
        if not isinstance(value.get(name), str) or not value[name].strip():
            raise ValueError(f"missing required config path: {name}")
    for name, policy in (
        ("head_pretrain", "head_projection_only"),
        ("joint_train", "full_network"),
    ):
        stage = value.get(name)
        if not isinstance(stage, Mapping):
            raise ValueError(f"{name} must be a mapping")
        if stage.get("training_policy") != policy:
            raise ValueError(f"{name}.training_policy must be {policy}")
        sampler = stage.get("sampler") or {}
        if sampler.get("mode") != "source_speaker_activity":
            raise ValueError(
                f"{name}.sampler.mode must be source_speaker_activity"
            )
    if value["joint_train"].get("require_disjoint_replay_evaluation") is not True:
        raise ValueError(
            "joint_train.require_disjoint_replay_evaluation must be true"
        )
    head_seed = int(value["head_pretrain"].get("seed", 7))
    joint_seed = int(value["joint_train"].get("seed", 7))
    if head_seed != joint_seed:
        raise ValueError("head and joint stages must use the same strict split seed")
    if value["head_pretrain"].get("split_fractions") != value["joint_train"].get(
        "split_fractions"
    ):
        raise ValueError("head and joint stages must use identical split_fractions")
    quarantine = value.get("temporal_quarantine")
    expected_quarantine = {
        "fps": 30.0,
        "acceleration_reduction": "per_axis_absolute_peak_and_p99_then_max_axis",
        "high_frequency_method": (
            "centered_5_frame_moving_average_interior_residual_per_axis_rms_then_max_axis"
        ),
        "high_frequency_window_frames": 5,
        "policy": "exclude_failed_train_keep_validation_test_as_challenge",
    }
    if not isinstance(quarantine, Mapping):
        raise ValueError("temporal_quarantine must be a mapping")
    for name, expected in expected_quarantine.items():
        if quarantine.get(name) != expected:
            raise ValueError(f"temporal_quarantine.{name} must be {expected!r}")
    for name in (
        "peak_acceleration_rad_s2",
        "p99_acceleration_rad_s2",
        "high_frequency_rms_degrees",
    ):
        if float(quarantine.get(name, 0.0)) <= 0:
            raise ValueError(f"temporal_quarantine.{name} must be positive")
    return value


def stage_paths(config: Mapping) -> dict[str, Path]:
    root = Path(config["output_root"]).resolve()
    return {
        "root": root,
        "receipts": root / "stage_receipts",
        "migrated": root / "migrated_from_15d.pt",
        "initial_split": root / "initial_split_manifest.json",
        "pre_quarantine_split": root / "pre_quarantine_split_manifest.json",
        "temporal_summary": root / "temporal_quarantine_summary.json",
        "temporal_exclusions": root / "temporal_quarantine_train_exclusions.jsonl",
        "temporal_challenges": root / "temporal_quarantine_eval_challenges.jsonl",
        "head_cache": root / "head_conditions.npz",
        "head_run": root / "head_pretrain",
        "joint_cache": root / "joint_conditions.npz",
        "joint_run": root / "joint_train",
        "status": root / "pipeline_status.json",
    }


def source_contract(config: Mapping) -> dict:
    records = []
    for source in config["motion_sources"]:
        if not isinstance(source, Mapping):
            raise ValueError("every motion source must be a mapping")
        path = Path(source["manifest"]).resolve()
        dataset_source = str(source.get("dataset_source") or "").strip()
        if not dataset_source:
            raise ValueError("every motion source requires dataset_source")
        records.append(
            {
                "dataset_source": dataset_source,
                "temporal_unit": source.get("temporal_unit"),
                "manifest": str(path),
                "manifest_sha256": sha256_file(path),
            }
        )
    if len({row["dataset_source"] for row in records}) != len(records):
        raise ValueError("motion source dataset_source values must be unique")
    contract = {
        "audio_policy": "disabled_not_loaded",
        "training_scope": config["training_scope"],
        "formal_training_enabled": config["formal_training_enabled"],
        "sources": records,
    }
    contract["sha256"] = _json_hash(contract)
    return contract


def load_motion_sources(config: Mapping) -> list[dict]:
    episodes = []
    seen = set()
    allow_unreviewed = bool(config.get("allow_unreviewed", False))
    for source in config["motion_sources"]:
        dataset_source = str(source["dataset_source"])
        manifest = Path(source["manifest"])
        raw_by_clip = {}
        with manifest.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON in {manifest}:{line_number}"
                    ) from exc
                clip_id = str(
                    record.get("clip_id") or record.get("sample_id") or ""
                ).strip()
                if clip_id:
                    raw_by_clip[clip_id] = record
        loaded = load_18d_episodes(
            manifest=manifest,
            allow_unreviewed=allow_unreviewed,
        )
        for episode in loaded:
            item = dict(episode)
            raw = raw_by_clip.get(str(item["clip_id"]), {})
            for field in (
                "speaker_key",
                "source_group_key",
                "source_group_id",
                "source_clip_id",
            ):
                if item.get(field) in (None, "") and raw.get(field) not in (None, ""):
                    item[field] = raw[field]
            split = raw.get("split") if isinstance(raw.get("split"), Mapping) else {}
            raw_source = (
                raw.get("source") if isinstance(raw.get("source"), Mapping) else {}
            )
            meta = raw.get("meta") if isinstance(raw.get("meta"), Mapping) else {}
            if item.get("source_group_key") in (None, ""):
                item["source_group_key"] = (
                    split.get("source_group_key")
                    or raw_source.get("source_group_key")
                    or raw_source.get("source_clip_id")
                )
            if item.get("speaker_key") in (None, ""):
                item["speaker_key"] = meta.get("speaker_key") or raw_source.get(
                    "speaker_key"
                )
            item["dataset_source"] = dataset_source
            clip_id = str(item["clip_id"])
            if clip_id in seen:
                raise ValueError(f"duplicate clip_id across motion sources: {clip_id}")
            seen.add(clip_id)
            episodes.append(item)
    return episodes


def initial_strict_split(config: Mapping, episodes: Sequence[dict]):
    # Conditions do not participate in grouping. A fixed dummy only satisfies
    # the canonical 18D record contract while deriving the pre-cache split.
    provisional = []
    for episode in episodes:
        item = dict(episode)
        item["condition"] = np.zeros(KIMODO_V2_CONDITION_DIM, dtype=np.float32)
        provisional.append(item)
    training = config["head_pretrain"]
    return strict_group_split(
        provisional,
        seed=int(training.get("seed", 7)),
        fractions=training.get("split_fractions"),
    )


def temporal_quarantine(config: Mapping, episodes: Sequence[dict]):
    splits, split_contract = initial_strict_split(config, episodes)
    assignment = {
        str(episode["clip_id"]): split_name
        for split_name, rows in splits.items()
        for episode in rows
    }
    thresholds = dict(config["temporal_quarantine"])
    required_fps = float(thresholds["fps"])
    peak_limit = float(thresholds["peak_acceleration_rad_s2"])
    p99_limit = float(thresholds["p99_acceleration_rad_s2"])
    hf_limit_rad = float(np.deg2rad(thresholds["high_frequency_rms_degrees"]))
    window = int(thresholds["high_frequency_window_frames"])
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padding = window // 2
    records = []
    excluded_train_ids = set()
    counts = Counter()
    reason_counts = Counter()
    for episode in sorted(episodes, key=lambda row: str(row["clip_id"])):
        clip_id = str(episode["clip_id"])
        split_name = assignment[clip_id]
        actions = np.asarray(episode["actions"], dtype=np.float64)
        fps = float(episode.get("fps") or required_fps)
        if not np.isfinite(fps) or not np.isclose(fps, required_fps, atol=1e-8):
            raise ValueError(
                f"temporal quarantine requires {required_fps:g} Hz input; "
                f"{clip_id} declares {fps!r} Hz"
            )
        head = actions[:, LEGACY_ACTION_DIM:]
        if head.shape[0] < window:
            raise ValueError(
                f"temporal quarantine needs at least {window} frames: {clip_id}"
            )
        acceleration = np.diff(head, n=2, axis=0) * (fps**2)
        absolute_acceleration = np.abs(acceleration)
        peak_acceleration_by_axis = absolute_acceleration.max(axis=0)
        p99_acceleration_by_axis = np.quantile(
            absolute_acceleration, 0.99, axis=0
        )
        smoothed = np.stack(
            [np.convolve(head[:, axis], kernel, mode="valid") for axis in range(3)],
            axis=-1,
        )
        high_frequency = head[padding:-padding] - smoothed
        high_frequency_rms_by_axis = np.sqrt(
            np.mean(np.square(high_frequency), axis=0)
        )
        metrics = {
            "peak_abs_acceleration_by_axis_rad_s2": [
                float(value) for value in peak_acceleration_by_axis
            ],
            "peak_abs_acceleration_max_axis_rad_s2": float(
                peak_acceleration_by_axis.max()
            ),
            "p99_abs_acceleration_by_axis_rad_s2": [
                float(value) for value in p99_acceleration_by_axis
            ],
            "p99_abs_acceleration_max_axis_rad_s2": float(
                p99_acceleration_by_axis.max()
            ),
            "moving_average_residual_rms_by_axis_rad": [
                float(value) for value in high_frequency_rms_by_axis
            ],
            "moving_average_residual_rms_max_axis_rad": float(
                high_frequency_rms_by_axis.max()
            ),
            "moving_average_residual_rms_by_axis_degrees": [
                float(value) for value in np.rad2deg(high_frequency_rms_by_axis)
            ],
            "moving_average_residual_rms_max_axis_degrees": float(
                np.rad2deg(high_frequency_rms_by_axis.max())
            ),
        }
        failed = []
        if metrics["peak_abs_acceleration_max_axis_rad_s2"] > peak_limit:
            failed.append("peak_acceleration")
        if metrics["p99_abs_acceleration_max_axis_rad_s2"] > p99_limit:
            failed.append("p99_acceleration")
        if metrics["moving_average_residual_rms_max_axis_rad"] > hf_limit_rad:
            failed.append("high_frequency_5frame_rms")
        counts[(split_name, "total")] += 1
        counts[(split_name, "failed" if failed else "passed")] += 1
        for reason in failed:
            reason_counts[(split_name, reason)] += 1
        excluded = bool(split_name == "train" and failed)
        if excluded:
            excluded_train_ids.add(clip_id)
        records.append(
            {
                "clip_id": clip_id,
                "dataset_source": _episode_dataset_source_for_receipt(episode),
                "speaker_key": episode["speaker_key"],
                "source_group_key": episode["source_group_key"],
                "split": split_name,
                "trajectory_path": episode.get("trajectory_path"),
                "trajectory_sha256": episode.get("trajectory_sha256"),
                "actions_sha256": _array_hash(actions),
                "prompt_sha256": episode.get("prompt_sha256")
                or hashlib.sha256(
                    str(episode.get("prompt") or "").encode("utf-8")
                ).hexdigest(),
                "fps": fps,
                "metrics": metrics,
                "failed_gates": failed,
                "excluded_from_training": excluded,
                "evaluation_challenge_retained": bool(
                    split_name in {"validation", "test"} and failed
                ),
            }
        )
    filtered = []
    record_by_id = {record["clip_id"]: record for record in records}
    for episode in episodes:
        clip_id = str(episode["clip_id"])
        if clip_id in excluded_train_ids:
            continue
        item = dict(episode)
        item["fixed_split_assignment"] = assignment[clip_id]
        quarantine_record = record_by_id[clip_id]
        item["temporal_quarantine_challenge"] = bool(
            quarantine_record["evaluation_challenge_retained"]
        )
        item["temporal_quarantine_failed_gates"] = list(
            quarantine_record["failed_gates"]
        )
        filtered.append(item)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "audio_policy": "disabled_not_loaded",
        "thresholds": thresholds,
        "pre_quarantine_split_contract_sha256": split_contract["sha256"],
        "counts": {
            split_name: {
                "total": counts[(split_name, "total")],
                "passed": counts[(split_name, "passed")],
                "failed": counts[(split_name, "failed")],
                "excluded_from_training": (
                    counts[(split_name, "failed")] if split_name == "train" else 0
                ),
                "retained_as_challenge": (
                    counts[(split_name, "failed")] if split_name != "train" else 0
                ),
            }
            for split_name in ("train", "validation", "test")
        },
        "failed_gate_counts": {
            split_name: {
                reason: reason_counts[(split_name, reason)]
                for reason in (
                    "peak_acceleration",
                    "p99_acceleration",
                    "high_frequency_5frame_rms",
                )
            }
            for split_name in ("train", "validation", "test")
        },
        "input_episode_count": len(episodes),
        "filtered_episode_count": len(filtered),
        "train_exclusion_count": len(excluded_train_ids),
        "evaluated_records_sha256": _json_hash(records),
        "input_episode_content_sha256": episodes_content_contract(episodes)[
            "sha256"
        ],
        "filtered_episode_content_sha256": episodes_content_contract(filtered)[
            "sha256"
        ],
    }
    summary["contract_sha256"] = _json_hash(summary)
    exclusions = [row for row in records if row["excluded_from_training"]]
    challenges = [row for row in records if row["evaluation_challenge_retained"]]
    return filtered, summary, exclusions, challenges, split_contract


def _episode_dataset_source_for_receipt(episode: Mapping) -> str:
    return str(
        episode.get("dataset_source")
        or episode.get("source_dataset")
        or episode.get("dataset_id")
        or "unknown"
    )


def _jsonl_text(records: Sequence[Mapping]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )


def write_temporal_quarantine_artifacts(
    paths: Mapping[str, Path],
    summary: Mapping,
    exclusions: Sequence[Mapping],
    challenges: Sequence[Mapping],
    pre_split: Mapping,
) -> None:
    json_outputs = {
        paths["temporal_summary"]: dict(summary),
        paths["pre_quarantine_split"]: dict(pre_split),
    }
    text_outputs = {
        paths["temporal_exclusions"]: _jsonl_text(exclusions),
        paths["temporal_challenges"]: _jsonl_text(challenges),
    }
    for path, payload in json_outputs.items():
        expected = json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != expected:
            raise RuntimeError(f"temporal quarantine artifact changed: {path}")
        if not path.exists():
            _atomic_json(payload, path)
    for path, expected in text_outputs.items():
        if path.exists() and path.read_text(encoding="utf-8") != expected:
            raise RuntimeError(f"temporal quarantine artifact changed: {path}")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_text(expected, encoding="utf-8")
            temporary.replace(path)


def _receipt_path(paths: Mapping[str, Path], stage: str) -> Path:
    return paths["receipts"] / f"{stage}.json"


def _valid_receipt(path: Path, contract_sha256: str, outputs: Sequence[Path]) -> bool:
    if not path.is_file():
        return False
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("contract_sha256") != contract_sha256:
        return False
    expected = {str(path.resolve()): sha256_file(path) for path in outputs if path.is_file()}
    return len(expected) == len(outputs) and receipt.get("outputs") == expected


def _write_receipt(
    paths: Mapping[str, Path], stage: str, contract: Mapping, outputs: Sequence[Path]
) -> dict:
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "contract_sha256": _json_hash(contract),
        "contract": dict(contract),
        "outputs": {str(path.resolve()): sha256_file(path) for path in outputs},
    }
    _atomic_json(receipt, _receipt_path(paths, stage))
    return receipt


def _require_clean_or_valid(
    paths: Mapping[str, Path],
    stage: str,
    contract: Mapping,
    outputs: Sequence[Path],
) -> bool:
    digest = _json_hash(contract)
    receipt = _receipt_path(paths, stage)
    if _valid_receipt(receipt, digest, outputs):
        return True
    if receipt.exists() or any(path.exists() for path in outputs):
        raise RuntimeError(
            f"{stage} has outputs without a matching hash receipt; move them aside "
            "or restore the matching config before continuing"
        )
    return False


def run_migrate(
    config: Mapping,
    paths: Mapping[str, Path],
    episodes: Sequence[dict],
    quarantine_summary: Mapping,
):
    base = Path(config["base_15d_checkpoint"]).resolve()
    splits, split_contract = initial_strict_split(config, episodes)
    contract = {
        "base_15d_checkpoint": str(base),
        "base_15d_checkpoint_sha256": sha256_file(base),
        "motion_sources": source_contract(config),
        "strict_train_only_stats": True,
        "split_contract_sha256": split_contract["sha256"],
        "temporal_quarantine_contract_sha256": quarantine_summary["contract_sha256"],
        "temporal_quarantine_thresholds": quarantine_summary["thresholds"],
        "temporal_train_exclusion_count": quarantine_summary[
            "train_exclusion_count"
        ],
        "episodes_content": episodes_content_contract(episodes),
    }
    outputs = [paths["migrated"], paths["initial_split"]]
    if _require_clean_or_valid(paths, "migrate", contract, outputs):
        return
    _, checkpoint = load_contract_checkpoint(base, expected_action_dim=15)
    stats = compute_18d_action_stats(
        [episode["actions"] for episode in splits["train"]],
        checkpoint["action_stats"],
    )
    migrate_15d_checkpoint(base, paths["migrated"], action_stats=stats)
    _atomic_json(split_contract, paths["initial_split"])
    _write_receipt(paths, "migrate", contract, outputs)


def run_cache(
    config: Mapping,
    paths: Mapping[str, Path],
    episodes: Sequence[dict],
    *,
    stage: str,
    generator_checkpoint: Path,
    output: Path,
    allow_download: bool,
):
    qwen = Path(config["qwen_checkpoint"]).resolve()
    contract = {
        "generator_checkpoint": str(generator_checkpoint.resolve()),
        "generator_checkpoint_sha256": sha256_file(generator_checkpoint),
        "qwen_checkpoint": str(qwen),
        "qwen_checkpoint_sha256": sha256_file(qwen),
        "motion_sources": source_contract(config),
        "audio_policy": "disabled_not_loaded",
        "temporal_quarantine_summary_sha256": sha256_file(
            paths["temporal_summary"]
        ),
        "episodes_content": episodes_content_contract(episodes),
    }
    metadata = output.with_name(output.name + ".json")
    outputs = [output, metadata]
    def validate_output():
        _, _, _, provenance = load_condition_cache(output)
        _, checkpoint = load_contract_checkpoint(
            generator_checkpoint, expected_action_dim=18
        )
        validate_condition_cache_for_generator(
            checkpoint,
            provenance,
            generator_checkpoint_path=generator_checkpoint,
        )

    if _require_clean_or_valid(paths, stage, contract, outputs):
        validate_output()
        return
    build_condition_cache(
        episodes,
        qwen,
        output,
        base_checkpoint=generator_checkpoint,
        device=str(config.get("cache_device") or "auto"),
        local_files_only=not allow_download,
        batch_size=int(config.get("cache_batch_size") or 16),
    )
    validate_output()
    _write_receipt(paths, stage, contract, outputs)


def _training_complete(
    output_dir: Path, target_steps: int, expected_config: Mapping
) -> bool:
    summary_path = output_dir / "training_summary.json"
    checkpoint = output_dir / "ula_fm_checkpoint.pt"
    if not summary_path.is_file() or not checkpoint.is_file():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    stored_config = dict(payload.get("posttrain_config") or {})
    ignored = {"resume_from", "overwrite"}
    compared = (set(stored_config) | set(expected_config)) - ignored
    if any(stored_config.get(key) != expected_config.get(key) for key in compared):
        return False
    return bool(
        int(summary.get("target_steps", -1)) == int(target_steps)
        and (
            int(summary.get("completed_steps", -1)) == int(target_steps)
            or summary.get("stopped_early") is True
        )
    )


def _validate_completed_training_lineage(
    output_dir: Path,
    *,
    initial_checkpoint_sha256: str,
    condition_cache: Path,
    attached_episodes: Sequence[Mapping],
    replay: Sequence[Mapping],
    replay_probe: Sequence[Mapping],
    replay_test: Sequence[Mapping],
    replay_provenance: Mapping | None,
) -> None:
    checkpoint_path = output_dir / "ula_fm_checkpoint.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (payload.get("posttrain_source") or {}).get(
        "checkpoint_sha256"
    ) != initial_checkpoint_sha256:
        raise RuntimeError("completed training source checkpoint lineage changed")
    _, _, _, cache = load_condition_cache(condition_cache)
    lineage_cache = (payload.get("data_provenance") or {}).get(
        "condition_cache"
    ) or {}
    for field in (
        "cache_sha256",
        "generator_checkpoint_sha256",
        "qwen_checkpoint_sha256",
        "qwen_model_name",
        "qwen_revision",
        "style_contract_sha256",
    ):
        if lineage_cache.get(field) != cache.get(field):
            raise RuntimeError(
                f"completed training condition-cache lineage changed field {field}"
            )
    split_contract = payload.get("posttrain_split_contract") or {}
    split_body = dict(split_contract)
    split_digest = split_body.pop("sha256", None)
    if split_digest != _json_hash(split_body):
        raise RuntimeError("completed training split contract hash is invalid")
    expected_split = {
        str(episode["clip_id"]): episode.get("fixed_split_assignment")
        for episode in attached_episodes
    }
    actual_split = {
        str(record["clip_id"]): record["split"]
        for record in split_contract.get("episodes") or []
    }
    if actual_split != expected_split or any(
        value not in {"train", "validation", "test"}
        for value in expected_split.values()
    ):
        raise RuntimeError("completed training did not preserve the fixed BEAT split")
    data_contract = payload.get("posttrain_data_contract") or {}
    data_body = dict(data_contract)
    data_digest = data_body.pop("sha256", None)
    if data_digest != _json_hash(data_body):
        raise RuntimeError("completed training data contract hash is invalid")
    if data_contract.get("split_contract_sha256") != split_digest:
        raise RuntimeError("completed training data/split contracts are not bound")
    records = data_contract.get("records") or []
    train_replay_ids = {
        record["clip_id"]
        for record in records
        if record.get("split") == "train_replay_only"
    }
    probe_ids = {
        record["clip_id"]
        for record in records
        if record.get("split") == "replay_evaluation_only"
    }
    test_ids = {
        record["clip_id"]
        for record in records
        if record.get("split") == "replay_test_only"
    }
    expected_train_replay = {
        f"kimodo:{_raw_episode_id(row)}" for row in replay
    }
    expected_probe = {
        f"kimodo:{_raw_episode_id(row)}" for row in replay_probe
    }
    expected_test = {
        f"kimodo:{_raw_episode_id(row)}" for row in replay_test
    }
    if (
        train_replay_ids != expected_train_replay
        or probe_ids != expected_probe
        or test_ids != expected_test
    ):
        raise RuntimeError("completed training replay data lineage changed")
    if (train_replay_ids & probe_ids) or (train_replay_ids & test_ids) or (
        probe_ids & test_ids
    ):
        raise RuntimeError("completed training replay probe leaked into optimization")
    actual_beat = {
        record["clip_id"]: (
            record["actions_sha256"],
            record["condition_sha256"],
            bool(record.get("temporal_quarantine_challenge", False)),
        )
        for record in records
        if record.get("domain") == "beat2"
    }
    expected_beat = {
        str(episode["clip_id"]): (
            _array_hash(episode["actions"]),
            _array_hash(episode["condition"]),
            bool(episode.get("temporal_quarantine_challenge", False)),
        )
        for episode in attached_episodes
    }
    if actual_beat != expected_beat:
        raise RuntimeError("completed training BEAT trajectory/condition lineage changed")
    actual_replay = {
        record["clip_id"]: (
            record["actions_sha256"],
            record["condition_sha256"],
        )
        for record in records
        if record.get("domain") == "kimodo"
    }
    expected_replay = {
        f"kimodo:{_raw_episode_id(episode)}": (
            _array_hash(episode["actions"]),
            _array_hash(episode["condition"]),
        )
        for episode in (*replay, *replay_probe, *replay_test)
    }
    if actual_replay != expected_replay:
        raise RuntimeError("completed training Kimodo content lineage changed")
    stored_replay = (payload.get("data_provenance") or {}).get("replay") or {}
    if stored_replay != dict(replay_provenance or {}):
        raise RuntimeError("completed training replay provenance changed")


def run_training(
    config: Mapping,
    paths: Mapping[str, Path],
    episodes: Sequence[dict],
    *,
    stage: str,
    initial_checkpoint: Path,
    condition_cache: Path,
    output_dir: Path,
    training_config: Mapping,
    replay: Sequence[dict] = (),
    replay_probe: Sequence[dict] = (),
    replay_test: Sequence[dict] = (),
    replay_provenance: Mapping | None = None,
):
    target_steps = int(training_config["steps"])
    stage_config = dict(training_config)
    stage_config["allow_unsafe_training_data"] = bool(
        config.get("allow_unsafe_training_data", False)
    )
    stage_config["training_scope"] = config["training_scope"]
    stage_config["formal_training_enabled"] = config["formal_training_enabled"]
    stage_config["temporal_unit_policy"] = "fixed_window_experimental"
    expected_config = resolve_posttrain_config(stage_config)
    contract = {
        "initial_checkpoint": str(initial_checkpoint.resolve()),
        "initial_checkpoint_sha256": sha256_file(initial_checkpoint),
        "condition_cache": str(condition_cache.resolve()),
        "condition_cache_sha256": sha256_file(condition_cache),
        "condition_cache_metadata_sha256": sha256_file(
            condition_cache.with_name(condition_cache.name + ".json")
        ),
        "training": dict(expected_config),
        "motion_sources": source_contract(config),
        "audio_policy": "disabled_not_loaded",
        "temporal_quarantine_summary_sha256": sha256_file(
            paths["temporal_summary"]
        ),
        "replay_provenance": dict(replay_provenance or {}),
        "replay_train_episode_ids_sha256": _json_hash(
            sorted(_raw_episode_id(row) for row in replay)
        ),
        "replay_probe_episode_ids_sha256": _json_hash(
            sorted(_raw_episode_id(row) for row in replay_probe)
        ),
        "replay_test_episode_ids_sha256": _json_hash(
            sorted(_raw_episode_id(row) for row in replay_test)
        ),
        "episodes_content": episodes_content_contract(episodes),
        "replay_train_content": replay_content_contract(
            replay, role="optimization_train"
        ),
        "replay_validation_content": replay_content_contract(
            replay_probe, role="validation_guard"
        ),
        "replay_test_content": replay_content_contract(
            replay_test, role="final_test_only"
        ),
    }
    summary_path = output_dir / "training_summary.json"
    checkpoint_path = output_dir / "ula_fm_checkpoint.pt"
    outputs = [summary_path, checkpoint_path, output_dir / "split_manifest.json"]
    receipt_path = _receipt_path(paths, stage)
    if _valid_receipt(receipt_path, _json_hash(contract), outputs):
        return
    if receipt_path.exists():
        raise RuntimeError(f"{stage} receipt does not match current inputs; refusing reissue")
    if _training_complete(output_dir, target_steps, expected_config):
        raise RuntimeError(
            f"{stage} is complete but lacks its matching receipt; refusing reissue"
        )
    last = output_dir / "last.pt"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not last.is_file():
            raise RuntimeError(f"{stage} is incomplete but has no resumable last.pt")
        stage_config["resume_from"] = str(last.resolve())
    attached = attach_condition_cache(
        episodes,
        condition_cache,
        allow_unsafe_metadata=bool(config.get("allow_unsafe_condition_cache", False)),
    )
    train_18d_posttrain(
        initial_checkpoint_path=initial_checkpoint,
        beat_episodes=attached,
        kimodo_replay_episodes=replay,
        kimodo_replay_probe_episodes=replay_probe,
        kimodo_replay_test_episodes=replay_test,
        replay_provenance=replay_provenance,
        output_dir=output_dir,
        config=stage_config,
    )
    if not _training_complete(output_dir, target_steps, expected_config):
        raise RuntimeError(f"{stage} returned without a completed or early-stopped summary")
    _validate_completed_training_lineage(
        output_dir,
        initial_checkpoint_sha256=contract["initial_checkpoint_sha256"],
        condition_cache=condition_cache,
        attached_episodes=attached,
        replay=replay,
        replay_probe=replay_probe,
        replay_test=replay_test,
        replay_provenance=replay_provenance,
    )
    _write_receipt(paths, stage, contract, outputs)


def _receipt_self_valid(path: Path, *, expected_stage: str) -> bool:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        contract = receipt["contract"]
        outputs = receipt["outputs"]
        if (
            receipt.get("schema_version") != SCHEMA_VERSION
            or receipt.get("stage") != expected_stage
            or receipt.get("contract_sha256") != _json_hash(contract)
            or not isinstance(outputs, Mapping)
            or not outputs
        ):
            return False
        return all(
            Path(output).is_file() and sha256_file(output) == digest
            for output, digest in outputs.items()
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _status(config: Mapping, paths: Mapping[str, Path], completed: Sequence[str]):
    del completed
    verified = [
        stage
        for stage in STAGES
        if _receipt_self_valid(
            _receipt_path(paths, stage), expected_stage=stage
        )
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "audio_policy": "disabled_not_loaded",
        "completed_stages": verified,
        "source_contract": source_contract(config),
        "temporal_quarantine_summary": str(paths["temporal_summary"].resolve()),
        "temporal_quarantine_summary_sha256": sha256_file(
            paths["temporal_summary"]
        ),
        "final_checkpoint": (
            str((paths["joint_run"] / "ula_fm_checkpoint.pt").resolve())
            if "joint" in verified
            else None
        ),
    }
    _atomic_json(payload, paths["status"])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("all",) + STAGES, default="all")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = read_config(args.config)
    paths = stage_paths(config)
    paths["root"].mkdir(parents=True, exist_ok=True)
    raw_episodes = load_motion_sources(config)
    (
        episodes,
        quarantine_summary,
        quarantine_exclusions,
        quarantine_challenges,
        pre_quarantine_split,
    ) = temporal_quarantine(config, raw_episodes)
    write_temporal_quarantine_artifacts(
        paths,
        quarantine_summary,
        quarantine_exclusions,
        quarantine_challenges,
        pre_quarantine_split,
    )
    completed = []

    def requested(stage):
        return args.stage == "all" or args.stage == stage

    if requested("migrate") or args.stage != "migrate":
        run_migrate(config, paths, episodes, quarantine_summary)
        completed.append("migrate")
    if requested("head-cache") or args.stage in ("head-pretrain", "joint-cache", "joint"):
        run_cache(
            config,
            paths,
            episodes,
            stage="head-cache",
            generator_checkpoint=paths["migrated"],
            output=paths["head_cache"],
            allow_download=args.allow_download,
        )
        completed.append("head-cache")
    if requested("head-pretrain") or args.stage in ("joint-cache", "joint"):
        run_training(
            config,
            paths,
            episodes,
            stage="head-pretrain",
            initial_checkpoint=paths["migrated"],
            condition_cache=paths["head_cache"],
            output_dir=paths["head_run"],
            training_config=config["head_pretrain"],
        )
        completed.append("head-pretrain")
    if requested("joint-cache") or args.stage == "joint":
        run_cache(
            config,
            paths,
            episodes,
            stage="joint-cache",
            generator_checkpoint=paths["head_run"] / "ula_fm_checkpoint.pt",
            output=paths["joint_cache"],
            allow_download=args.allow_download,
        )
        completed.append("joint-cache")
    if requested("joint"):
        replay, replay_probe, replay_test, replay_provenance = load_kimodo_replay_splits(
            config["kimodo_dataset_dir"],
            config["kimodo_split_checkpoint"],
            config["qwen_checkpoint"],
            device=str(config["joint_train"].get("device") or "auto"),
            local_files_only=not args.allow_download,
        )
        run_training(
            config,
            paths,
            episodes,
            stage="joint",
            initial_checkpoint=paths["head_run"] / "ula_fm_checkpoint.pt",
            condition_cache=paths["joint_cache"],
            output_dir=paths["joint_run"],
            training_config=config["joint_train"],
            replay=replay,
            replay_probe=replay_probe,
            replay_test=replay_test,
            replay_provenance=replay_provenance,
        )
        head_split = json.loads(
            (paths["head_run"] / "split_manifest.json").read_text(encoding="utf-8")
        )
        joint_split = json.loads(
            (paths["joint_run"] / "split_manifest.json").read_text(encoding="utf-8")
        )
        initial_split = json.loads(
            paths["initial_split"].read_text(encoding="utf-8")
        )
        if len(
            {
                initial_split.get("sha256"),
                head_split.get("sha256"),
                joint_split.get("sha256"),
            }
        ) != 1:
            raise RuntimeError(
                "migration stats, head, and joint stages did not reuse the same speaker split"
            )
        completed.append("joint")
    _status(config, paths, completed)
    print(json.dumps({"completed_stages": completed, "paths": {key: str(value) for key, value in paths.items()}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
