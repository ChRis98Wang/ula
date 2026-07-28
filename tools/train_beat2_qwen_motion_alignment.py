#!/usr/bin/env python3
"""BEAT2-only Qwen frozen-base versus LoRA text/motion alignment experiment.

This is intentionally independent from the older language/motion trainers.  It
accepts one adjudicated BEAT2 min-30-frame manifest, raw 18-DoF CSV trajectories,
and an official pinned Qwen base model.  There is no input checkpoint: the
motion encoder and both alignment heads are initialized in this run.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import csv
import gc
import hashlib
import io
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import yaml
except ImportError:  # pragma: no cover - deployment dependency
    yaml = None


SCHEMA_VERSION = 1
ARTIFACT_KIND = "beat2_qwen_motion_alignment_ab_v1"
MOTION_ARTIFACT_KIND = "beat2_random_init_motion_encoder_v1"
BASELINE_ARTIFACT_KIND = "beat2_qwen_frozen_base_alignment_v1"
LORA_ARTIFACT_KIND = "beat2_qwen_lora_alignment_v1"
NO_EXTERNAL_DATA_POLICY = "beat2_only_no_external_motion_dataset_v1"
SEMANTIC_SCOPE_ACKNOWLEDGEMENT = (
    "experimental_official_metadata_alignment_only_not_formal_generator_supervision"
)
SPLIT_NAMES = ("train", "validation", "test")
EXPECTED_DATASET = "BEAT2"
EXPECTED_TRAINING_ADMISSION = "motion_only_physical_qc_train_ready"
EXPECTED_LOCK_KIND = "ula_v2_18d_motion_only_pretrain_provenance_lock_v1"
EXPECTED_QWEN_MODEL = "Qwen/Qwen3-Embedding-0.6B"
FORBIDDEN_SOURCE_TOKEN = "kimodo"
FPS = 30.0

JOINT_NAMES = (
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
)

DEFAULT_CONFIG = {
    "manifest_path": (
        "/home/gez/nas/cloud/gez/human_motion/processed/"
        "beat2_semantic_event_training_pool_18d_v7_full/"
        "adjudication_min30f/train_ready.jsonl"
    ),
    "provenance_lock_path": (
        "/home/gez/nas/cloud/gez/human_motion/catalog/"
        "beat2_semantic_event_pilot_v7_full/"
        "motion_only_pretrain_provenance_lock_min30f.json"
    ),
    "allowed_csv_root": (
        "/home/gez/nas/cloud/gez/human_motion/processed/"
        "beat2_semantic_event_training_pool_18d_v7_full"
    ),
    "output_dir": "training/runs/beat2_qwen_motion_alignment_ab_v1",
    "expected_manifest_sha256": (
        "2b3692c4f0a9ea8e10f3bde74fa178800556ed9bb79d0918a770b963a7f7c7fd"
    ),
    "semantic_scope_acknowledgement": SEMANTIC_SCOPE_ACKNOWLEDGEMENT,
    "min_frames": 30,
    "verify_csv_hashes": True,
    "phase_samples": 24,
    "seed": 20260726,
    "device": "auto",
    "model_name": EXPECTED_QWEN_MODEL,
    "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    "local_files_only": True,
    "attention_backend": "eager",
    "max_length": 96,
    "instruction": (
        "Represent the requested upper-body gesture category, motion intensity, "
        "and explicitly stated affect for matching to a BEAT2 robot-retargeted motion."
    ),
    "qwen_component_dim": 512,
    "motion": {
        "hidden_dim": 256,
        "latent_dim": 128,
        "dropout": 0.05,
        "steps": 4000,
        "batch_size": 256,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "max_grad_norm": 5.0,
        "eval_interval": 250,
        "log_interval": 50,
        "reconstruction_weight": 1.0,
        "category_weight": 0.5,
        "intensity_weight": 0.5,
        "emotion_weight": 0.5,
        "group_weight": 1.0,
    },
    "alignment": {
        "hidden_dim": 256,
        # Zero keeps the shared projector update path deterministic across B/C;
        # stochasticity unique to C remains confined to its LoRA modules.
        "dropout": 0.0,
        "frozen_steps": 1500,
        "lora_steps": 1500,
        "projector_lr": 0.0003,
        "lora_lr": 0.000005,
        "weight_decay": 0.0001,
        "max_grad_norm": 1.0,
        "warmup_steps": 100,
        "minimum_lr_ratio": 0.1,
        "temperature": 0.07,
        "retrieval_weight": 1.0,
        "cosine_weight": 0.5,
        "category_weight": 0.25,
        "intensity_weight": 0.25,
        "emotion_weight": 0.25,
        "eval_interval": 100,
        "log_interval": 25,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "lora_top_layers": 2,
        "lora_target_projections": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    "generator_pairing": {
        "foundation_checkpoint": None,
        "posttrain_seed": 20260726,
        "posttrain_steps": 50000,
        "condition_dim": 128,
    },
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_clean_generator_foundation(path: str | Path) -> dict:
    """Load and fail closed on anything except the clean BEAT2-only foundation."""
    checkpoint_path = Path(path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"clean generator foundation checkpoint is missing: {checkpoint_path}"
        )
    _reject_forbidden_source(
        checkpoint_path, field="generator_pairing.foundation_checkpoint"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("clean generator foundation must contain a checkpoint mapping")
    try:
        from upper_body_skeleton.ula_v2_18d_head import (
            validate_checkpoint_contract,
            validate_motion_only_checkpoint_isolation,
        )

        validate_checkpoint_contract(checkpoint, expected_action_dim=18)
        isolation = validate_motion_only_checkpoint_isolation(checkpoint)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "generator foundation is not a contract-valid random-init "
            "BEAT2-only, Qwen-free motion checkpoint"
        ) from exc
    random_initialization = checkpoint["random_initialization"]
    return {
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "architecture": str(checkpoint["architecture"]),
        "formal_episode_contract": str(checkpoint["formal_episode_contract"]),
        "data_isolation_contract_sha256": str(isolation["sha256"]),
        "random_initialization_mode": str(random_initialization["mode"]),
        "dataset_family_whitelist": list(isolation["dataset_family_whitelist"]),
        "generator_checkpoint_inputs": list(
            isolation["generator_checkpoint_inputs"]
        ),
        "qwen_policy": str(isolation["qwen_policy"]),
    }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _atomic_json_save(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_torch_save(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _append_jsonl(value: Any, path: str | Path) -> None:
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _deep_merge(defaults: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    unknown = sorted(set(override) - set(defaults))
    if unknown:
        raise ValueError(f"unknown BEAT2 Qwen A/B config keys: {unknown}")
    result = deepcopy(dict(defaults))
    for key, value in override.items():
        if isinstance(result.get(key), Mapping):
            if not isinstance(value, Mapping):
                raise ValueError(f"config section {key!r} must be a mapping")
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        if yaml is None:
            raise RuntimeError("PyYAML is required for YAML configuration")
        raw = yaml.safe_load(text) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("BEAT2 Qwen A/B config must be a mapping")
    config = _deep_merge(DEFAULT_CONFIG, raw)
    return validate_config(config)


def _positive_int(config: Mapping[str, Any], key: str) -> int:
    value = int(config[key])
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _positive_float(config: Mapping[str, Any], key: str) -> float:
    value = float(config[key])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{key} must be finite and positive")
    return value


def _reject_forbidden_source(value: str | Path, *, field: str) -> None:
    normalized = str(value).casefold()
    if FORBIDDEN_SOURCE_TOKEN in normalized:
        raise ValueError(
            f"{field} contains forbidden external-data token "
            f"{FORBIDDEN_SOURCE_TOKEN!r}: {value}"
        )


def validate_config(config: Mapping[str, Any]) -> dict:
    values = deepcopy(dict(config))
    for key in (
        "manifest_path",
        "provenance_lock_path",
        "allowed_csv_root",
        "output_dir",
        "model_name",
        "revision",
        "instruction",
        "semantic_scope_acknowledgement",
        "expected_manifest_sha256",
        "device",
        "attention_backend",
    ):
        values[key] = str(values[key])
    for key in (
        "manifest_path",
        "provenance_lock_path",
        "allowed_csv_root",
        "output_dir",
        "instruction",
    ):
        _reject_forbidden_source(values[key], field=key)
    if values["model_name"] != EXPECTED_QWEN_MODEL:
        raise ValueError(
            f"model_name must be the official base checkpoint {EXPECTED_QWEN_MODEL!r}"
        )
    if not values["revision"] or values["revision"].casefold() in {"main", "none"}:
        raise ValueError("revision must pin the official Qwen base to an immutable commit")
    if values["semantic_scope_acknowledgement"] != SEMANTIC_SCOPE_ACKNOWLEDGEMENT:
        raise ValueError(
            "the current BEAT2 release masks prompt supervision; this controlled A/B "
            "requires the exact metadata-only semantic-scope acknowledgement"
        )
    if values["attention_backend"] not in {"eager", "sdpa"}:
        raise ValueError("attention_backend must be eager or sdpa")
    for key in ("min_frames", "phase_samples", "seed", "max_length", "qwen_component_dim"):
        values[key] = int(values[key])
    if values["min_frames"] < 30:
        raise ValueError("min_frames may not weaken the locked 30-frame data filter")
    if values["phase_samples"] < 4:
        raise ValueError("phase_samples must be at least four")
    if values["max_length"] <= 0 or values["qwen_component_dim"] <= 0:
        raise ValueError("max_length and qwen_component_dim must be positive")
    values["local_files_only"] = bool(values["local_files_only"])
    values["verify_csv_hashes"] = bool(values["verify_csv_hashes"])

    motion = values["motion"]
    for key in (
        "hidden_dim",
        "latent_dim",
        "steps",
        "batch_size",
        "eval_interval",
        "log_interval",
    ):
        motion[key] = _positive_int(motion, key)
    for key in (
        "learning_rate",
        "weight_decay",
        "max_grad_norm",
        "reconstruction_weight",
        "category_weight",
        "intensity_weight",
        "emotion_weight",
        "group_weight",
    ):
        motion[key] = _positive_float(motion, key)
    motion["dropout"] = float(motion["dropout"])

    alignment = values["alignment"]
    for key in (
        "hidden_dim",
        "frozen_steps",
        "lora_steps",
        "warmup_steps",
        "eval_interval",
        "log_interval",
        "lora_rank",
        "lora_alpha",
        "lora_top_layers",
    ):
        alignment[key] = int(alignment[key])
        if alignment[key] < (0 if key == "warmup_steps" else 1):
            raise ValueError(f"alignment.{key} is invalid")
    for key in (
        "projector_lr",
        "lora_lr",
        "weight_decay",
        "max_grad_norm",
        "temperature",
        "retrieval_weight",
        "cosine_weight",
        "category_weight",
        "intensity_weight",
        "emotion_weight",
    ):
        alignment[key] = _positive_float(alignment, key)
    alignment["minimum_lr_ratio"] = float(alignment["minimum_lr_ratio"])
    alignment["dropout"] = float(alignment["dropout"])
    alignment["lora_dropout"] = float(alignment["lora_dropout"])
    if not 0 < alignment["minimum_lr_ratio"] <= 1:
        raise ValueError("alignment.minimum_lr_ratio must be in (0, 1]")
    alignment["lora_target_projections"] = [
        str(name) for name in alignment["lora_target_projections"]
    ]
    if alignment["frozen_steps"] != alignment["lora_steps"]:
        raise ValueError(
            "frozen_steps and lora_steps must match for the controlled B/C budget"
        )
    pairing = values["generator_pairing"]
    pairing["foundation_checkpoint"] = (
        None
        if pairing["foundation_checkpoint"] in (None, "")
        else str(pairing["foundation_checkpoint"])
    )
    if pairing["foundation_checkpoint"] is not None:
        _reject_forbidden_source(
            pairing["foundation_checkpoint"], field="generator_pairing.foundation_checkpoint"
        )
    pairing["posttrain_seed"] = int(pairing["posttrain_seed"])
    pairing["posttrain_steps"] = _positive_int(pairing, "posttrain_steps")
    pairing["condition_dim"] = _positive_int(pairing, "condition_dim")
    if pairing["condition_dim"] != 128 or motion["latent_dim"] != 128:
        raise ValueError(
            "motion.latent_dim and generator_pairing.condition_dim must both be 128"
        )
    return values


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _path_bearing_strings(value: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(item, str) and (
                "path" in str(key).casefold()
                or "csv" in str(key).casefold()
                or str(key).casefold()
                in {"source", "source_manifest", "motion_relpath", "annotation_relpath"}
            ):
                yield name, item
            else:
                yield from _path_bearing_strings(item, name)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            yield from _path_bearing_strings(item, f"{prefix}[{index}]")


def _iter_text_lines(path: str | Path) -> Iterable[tuple[int, str]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                yield line_number, line


def _semantic_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    event = row.get("semantic_event") or {}
    return (
        str(event.get("category", "")).strip(),
        str(event.get("intensity", "")).strip(),
        str(row.get("emotion_id", "")).strip(),
    )


def _prompt_text(row: Mapping[str, Any]) -> str:
    prompt = str(row.get("prompt", "")).strip()
    canonical = row.get("canonical_prompt") or {}
    if prompt != str(canonical.get("en", "")).strip():
        raise ValueError(f"canonical prompt mismatch for {row.get('task_id')}")
    if sha256_bytes(prompt.encode("utf-8")) != row.get("prompt_sha256"):
        raise ValueError(f"prompt hash mismatch for {row.get('task_id')}")
    return prompt


def _validate_quality_gate(row: Mapping[str, Any]) -> None:
    quality = row.get("quality_gate")
    if not isinstance(quality, Mapping) or not quality:
        raise ValueError(f"missing quality gate for {row.get('task_id')}")
    if quality.get("passed") is not True:
        raise ValueError(f"failed quality gate in train-ready manifest: {row.get('task_id')}")
    invalid = {key: value for key, value in quality.items() if value is not True}
    if invalid:
        raise ValueError(
            f"not all physical quality gates passed for {row.get('task_id')}: {invalid}"
        )


def _group_limited_rows(
    rows: Sequence[dict], max_records_per_group_per_split: int | None
) -> list[dict]:
    if max_records_per_group_per_split is None:
        return list(rows)
    limit = int(max_records_per_group_per_split)
    if limit <= 0:
        raise ValueError("max_records_per_group_per_split must be positive")
    counts: dict[tuple[str, tuple[str, str, str]], int] = defaultdict(int)
    selected = []
    for row in rows:
        key = (row["fixed_split_assignment"], _semantic_key(row))
        if counts[key] < limit:
            selected.append(row)
            counts[key] += 1
    return selected


def audit_dataset(
    config: Mapping[str, Any],
    *,
    max_records_per_group_per_split: int | None = None,
    write_outputs: bool = True,
) -> tuple[list[dict], dict]:
    manifest_path = Path(config["manifest_path"]).resolve()
    lock_path = Path(config["provenance_lock_path"]).resolve()
    csv_root = Path(config["allowed_csv_root"]).resolve()
    for path, label in (
        (manifest_path, "manifest"),
        (lock_path, "provenance lock"),
        (csv_root, "allowed CSV root"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != config["expected_manifest_sha256"]:
        raise ValueError(
            "BEAT2 train-ready manifest hash mismatch: "
            f"expected {config['expected_manifest_sha256']}, got {manifest_sha}"
        )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("artifact_kind") != EXPECTED_LOCK_KIND:
        raise ValueError("unexpected BEAT2 provenance lock artifact kind")
    if lock.get("accepted_for_training") is not True:
        raise ValueError("BEAT2 provenance lock is not accepted for training")
    locked_manifest = (lock.get("locked_artifacts") or {}).get("train_ready_manifest") or {}
    if locked_manifest.get("sha256") != manifest_sha:
        raise ValueError("BEAT2 provenance lock does not bind the configured train-ready manifest")
    if Path(str(locked_manifest.get("path", ""))).resolve() != manifest_path:
        raise ValueError("BEAT2 provenance lock train-ready path mismatch")
    lock_scale = lock.get("dataset_scale") or {}
    if int(lock_scale.get("episode_count", -1)) <= 0:
        raise ValueError("BEAT2 provenance lock has no dataset scale")

    all_rows: list[dict] = []
    seen_tasks: set[str] = set()
    for line_number, line in _iter_text_lines(manifest_path):
        row = json.loads(line)
        task_id = str(row.get("task_id", "")).strip()
        if not task_id or task_id in seen_tasks:
            raise ValueError(f"missing or duplicate task_id at manifest line {line_number}")
        seen_tasks.add(task_id)
        if row.get("dataset") != EXPECTED_DATASET:
            raise ValueError(f"non-BEAT2 row rejected at manifest line {line_number}")
        if row.get("accepted_for_training") is not True:
            raise ValueError(f"unadmitted row at manifest line {line_number}")
        if row.get("training_admission_status") != EXPECTED_TRAINING_ADMISSION:
            raise ValueError(f"unexpected training admission at manifest line {line_number}")
        if row.get("status") != "passed":
            raise ValueError(f"non-passed row at manifest line {line_number}")
        frames = int(row.get("frames", -1))
        if frames < int(config["min_frames"]):
            raise ValueError(f"short row escaped min-frame filter at manifest line {line_number}")
        split = row.get("fixed_split_assignment")
        if split not in SPLIT_NAMES:
            raise ValueError(f"invalid fixed split at manifest line {line_number}")
        if not str(row.get("speaker_key", "")).strip():
            raise ValueError(f"missing speaker key at manifest line {line_number}")
        _validate_quality_gate(row)
        prompt = _prompt_text(row)
        _reject_forbidden_source(prompt, field=f"manifest row {line_number}:prompt")
        if row.get("official_category_verified") is not True:
            raise ValueError(f"unverified official category at manifest line {line_number}")
        masks = row.get("semantic_supervision_masks") or {}
        if masks.get("prompt_text") is not False:
            raise ValueError(
                "this experiment expects the source release's fail-closed prompt mask "
                "and records its explicit metadata-only scope"
            )
        semantic_key = _semantic_key(row)
        if any(not value for value in semantic_key):
            raise ValueError(f"incomplete official semantic metadata at line {line_number}")
        safe_csv = Path(str(row.get("safe_csv", ""))).resolve()
        if not safe_csv.is_file() or not _is_under(safe_csv, csv_root):
            raise ValueError(f"unsafe or missing 18D CSV at line {line_number}: {safe_csv}")
        for field, source in _path_bearing_strings(row):
            _reject_forbidden_source(source, field=f"manifest row {line_number}:{field}")
        # Keep only audited fields used downstream.  In particular, speech/text
        # context fields are deliberately not retained as motion instructions.
        all_rows.append(
            {
                "task_id": task_id,
                "clip_id": str(row.get("clip_id") or task_id),
                "fixed_split_assignment": str(split),
                "speaker_key": str(row["speaker_key"]),
                "source_group_key": str(row["source_group_key"]),
                "frames": frames,
                "safe_csv": str(safe_csv),
                "safe_csv_sha256": str(row["safe_csv_sha256"]),
                "prompt": str(row["prompt"]),
                "canonical_prompt": {"en": str(row["canonical_prompt"]["en"])},
                "prompt_sha256": str(row["prompt_sha256"]),
                "semantic_event": {
                    "category": semantic_key[0],
                    "intensity": semantic_key[1],
                },
                "emotion_id": semantic_key[2],
            }
        )

    if len(all_rows) != int(lock_scale["episode_count"]):
        raise ValueError(
            f"manifest/lock episode count mismatch: {len(all_rows)} vs "
            f"{lock_scale['episode_count']}"
        )
    full_release_speakers = {
        split: {
            str(row["speaker_key"])
            for row in all_rows
            if row["fixed_split_assignment"] == split
        }
        for split in SPLIT_NAMES
    }
    expected_speaker_counts = {"train": 17, "validation": 4, "test": 4}
    full_speaker_counts = {
        split: len(full_release_speakers[split]) for split in SPLIT_NAMES
    }
    if full_speaker_counts != expected_speaker_counts:
        raise ValueError(
            "BEAT2 fixed speaker split must remain exactly 17/4/4; "
            f"got {full_speaker_counts}"
        )
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            if full_release_speakers[left] & full_release_speakers[right]:
                raise ValueError(f"full-release speaker leakage between {left} and {right}")
    rows = _group_limited_rows(all_rows, max_records_per_group_per_split)
    clip_ids = [str(row["clip_id"]) for row in rows]
    if len(set(clip_ids)) != len(clip_ids):
        raise ValueError("selected BEAT2 condition-cache clip ids are not unique")
    split_rows = {
        split: [row for row in rows if row["fixed_split_assignment"] == split]
        for split in SPLIT_NAMES
    }
    if any(not split_rows[split] for split in SPLIT_NAMES):
        raise ValueError("speaker-fixed train/validation/test splits must all be non-empty")
    speakers = {
        split: {str(row["speaker_key"]) for row in split_rows[split]}
        for split in SPLIT_NAMES
    }
    source_groups = {
        split: {str(row["source_group_key"]) for row in split_rows[split]}
        for split in SPLIT_NAMES
    }
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            if speakers[left] & speakers[right]:
                raise ValueError(f"speaker leakage between {left} and {right}")
            if source_groups[left] & source_groups[right]:
                raise ValueError(f"source-group leakage between {left} and {right}")
    prompts_by_group: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in rows:
        prompts_by_group[_semantic_key(row)].add(_prompt_text(row))
    ambiguous = {key: values for key, values in prompts_by_group.items() if len(values) != 1}
    if ambiguous:
        raise ValueError(f"semantic groups do not map to one canonical prompt: {ambiguous}")
    unique_prompts = {str(row["prompt"]) for row in rows}
    if len(prompts_by_group) != 54 or len(unique_prompts) != 54:
        raise ValueError(
            "the locked BEAT2 A/B design expects exactly 54 canonical semantic "
            f"groups/prompts, got groups={len(prompts_by_group)} "
            f"prompts={len(unique_prompts)}"
        )
    train_groups = {_semantic_key(row) for row in split_rows["train"]}
    for split in ("validation", "test"):
        unseen = {_semantic_key(row) for row in split_rows[split]} - train_groups
        if unseen:
            raise ValueError(f"{split} contains semantic groups absent from train: {unseen}")

    categories = sorted({_semantic_key(row)[0] for row in rows})
    intensities = sorted({_semantic_key(row)[1] for row in rows})
    emotions = sorted({_semantic_key(row)[2] for row in rows})
    groups = sorted(train_groups)
    split_manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": f"{ARTIFACT_KIND}_speaker_fixed_split",
        "data_policy": NO_EXTERNAL_DATA_POLICY,
        "no_kimodo": True,
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha},
        "provenance_lock": {"path": str(lock_path), "sha256": sha256_file(lock_path)},
        "semantic_scope": config["semantic_scope_acknowledgement"],
        "full_release_episode_count": len(all_rows),
        "full_release_speaker_counts": full_speaker_counts,
        "full_release_speakers": {
            split: sorted(full_release_speakers[split]) for split in SPLIT_NAMES
        },
        "selected_episode_count": len(rows),
        "smoke_group_cap": max_records_per_group_per_split,
        "labels": {
            "categories": categories,
            "intensities": intensities,
            "emotions": emotions,
            "groups": [list(group) for group in groups],
        },
        "splits": {
            split: {
                "episodes": len(split_rows[split]),
                "frames": sum(int(row["frames"]) for row in split_rows[split]),
                "speakers": sorted(speakers[split]),
                "source_group_count": len(source_groups[split]),
                "semantic_group_count": len(
                    {_semantic_key(row) for row in split_rows[split]}
                ),
                "task_ids": [row["task_id"] for row in split_rows[split]],
            }
            for split in SPLIT_NAMES
        },
        "speaker_sets_pairwise_disjoint": True,
        "source_group_sets_pairwise_disjoint": True,
        "input_checkpoint_count": 0,
        "csv_set_sha256": sha256_bytes(
            _canonical_json(
                sorted(
                    (str(row["task_id"]), str(row["safe_csv_sha256"])) for row in rows
                )
            )
        ),
        "prompt_set_sha256": sha256_bytes(
            _canonical_json(
                sorted(
                    (
                        list(key),
                        next(iter(prompt_set)),
                    )
                    for key, prompt_set in prompts_by_group.items()
                )
            )
        ),
        "unique_canonical_prompt_count": len(unique_prompts),
        "text_generalization_limit": (
            "canonical training text contains only one deterministic sentence per "
            "category/intensity/emotion group (54 total); canonical-prompt metrics "
            "must not be presented as open-text generalization"
        ),
        "speech_context_policy": (
            "source_speech_context and window_transcript_context are not loaded, "
            "trained on, or treated as motion instructions"
        ),
    }
    if write_outputs:
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json_save(split_manifest, output_dir / "split_manifest.json")
        _atomic_json_save(
            {
                "artifact_kind": f"{ARTIFACT_KIND}_audit",
                "data_policy": NO_EXTERNAL_DATA_POLICY,
                "no_kimodo": True,
                "status": "passed",
                "manifest_sha256": manifest_sha,
                "provenance_lock_sha256": split_manifest["provenance_lock"]["sha256"],
                "episode_count": len(rows),
                "speaker_count": len(set().union(*speakers.values())),
                "split_counts": {
                    split: len(split_rows[split]) for split in SPLIT_NAMES
                },
                "label_cardinalities": {
                    "category": len(categories),
                    "intensity": len(intensities),
                    "emotion": len(emotions),
                    "joint_group": len(groups),
                },
                "semantic_scope": config["semantic_scope_acknowledgement"],
            },
            output_dir / "audit.json",
        )
    return rows, split_manifest


def motion_descriptor(actions: np.ndarray, *, phase_samples: int, fps: float = FPS) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != len(JOINT_NAMES):
        raise ValueError(f"18D trajectory must have shape [frames, 18], got {actions.shape}")
    if actions.shape[0] < 3 or not np.isfinite(actions).all():
        raise ValueError("18D trajectory is too short or non-finite")
    source_phase = np.linspace(0.0, 1.0, actions.shape[0], dtype=np.float64)
    target_phase = np.linspace(0.0, 1.0, int(phase_samples), dtype=np.float64)
    phase = np.stack(
        [
            np.interp(target_phase, source_phase, actions[:, joint])
            for joint in range(actions.shape[1])
        ],
        axis=1,
    ).astype(np.float32)
    velocity = np.diff(actions, axis=0) * float(fps)
    acceleration = np.diff(velocity, axis=0) * float(fps)
    position_features = np.stack(
        [
            actions.mean(axis=0),
            actions.std(axis=0),
            actions.min(axis=0),
            actions.max(axis=0),
            actions[0],
            actions[-1],
            np.ptp(actions, axis=0),
            actions[-1] - actions[0],
        ],
        axis=0,
    )
    abs_velocity = np.abs(velocity)
    velocity_features = np.stack(
        [
            velocity.mean(axis=0),
            velocity.std(axis=0),
            np.sqrt(np.mean(np.square(velocity), axis=0)),
            abs_velocity.mean(axis=0),
            np.quantile(abs_velocity, 0.95, axis=0),
            abs_velocity.max(axis=0),
        ],
        axis=0,
    )
    abs_acceleration = np.abs(acceleration)
    acceleration_features = np.stack(
        [
            abs_acceleration.mean(axis=0),
            np.sqrt(np.mean(np.square(acceleration), axis=0)),
            np.quantile(abs_acceleration, 0.95, axis=0),
        ],
        axis=0,
    )
    global_features = np.asarray(
        [
            math.log1p(actions.shape[0]),
            (actions.shape[0] - 1) / float(fps),
            np.abs(np.diff(actions, axis=0)).sum(axis=0).mean(),
            np.sqrt(np.mean(np.square(velocity))),
        ],
        dtype=np.float32,
    )
    descriptor = np.concatenate(
        [
            phase.reshape(-1),
            position_features.reshape(-1),
            velocity_features.reshape(-1),
            acceleration_features.reshape(-1),
            global_features,
        ]
    ).astype(np.float32, copy=False)
    if not np.isfinite(descriptor).all():
        raise ValueError("motion descriptor contains non-finite values")
    return descriptor


def _read_18d_csv(row: Mapping[str, Any], *, verify_hash: bool) -> np.ndarray:
    path = Path(row["safe_csv"])
    payload = path.read_bytes()
    if verify_hash and sha256_bytes(payload) != row.get("safe_csv_sha256"):
        raise ValueError(f"18D CSV hash mismatch: {path}")
    header = next(csv.reader([payload.splitlines()[0].decode("utf-8-sig")]))
    if tuple(header) != JOINT_NAMES:
        raise ValueError(f"18D joint order mismatch: {path}")
    actions = np.loadtxt(
        io.BytesIO(payload), delimiter=",", skiprows=1, dtype=np.float32, ndmin=2
    )
    if actions.shape != (int(row["frames"]), len(JOINT_NAMES)):
        raise ValueError(
            f"18D CSV frame/shape mismatch for {row['task_id']}: "
            f"{actions.shape} vs {(row['frames'], len(JOINT_NAMES))}"
        )
    return actions


def _label_contract(split_manifest: Mapping[str, Any]) -> dict:
    labels = split_manifest["labels"]
    return {
        "categories": list(labels["categories"]),
        "intensities": list(labels["intensities"]),
        "emotions": list(labels["emotions"]),
        "groups": [tuple(group) for group in labels["groups"]],
    }


def _encode_labels(rows: Sequence[Mapping[str, Any]], label_contract: Mapping[str, Any]) -> dict:
    category_to_index = {
        label: index for index, label in enumerate(label_contract["categories"])
    }
    intensity_to_index = {
        label: index for index, label in enumerate(label_contract["intensities"])
    }
    emotion_to_index = {
        label: index for index, label in enumerate(label_contract["emotions"])
    }
    group_to_index = {
        tuple(label): index for index, label in enumerate(label_contract["groups"])
    }
    keys = [_semantic_key(row) for row in rows]
    return {
        "category": np.asarray([category_to_index[key[0]] for key in keys], dtype=np.int64),
        "intensity": np.asarray([intensity_to_index[key[1]] for key in keys], dtype=np.int64),
        "emotion": np.asarray([emotion_to_index[key[2]] for key in keys], dtype=np.int64),
        "group": np.asarray([group_to_index[key] for key in keys], dtype=np.int64),
    }


def _atomic_npz_save(path: str | Path, **arrays: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def build_descriptor_cache(
    rows: Sequence[dict],
    split_manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    output_dir = Path(config["output_dir"])
    cache_path = output_dir / "beat2_motion_descriptors_v1.npz"
    if cache_path.is_file() and not overwrite:
        load_descriptor_cache(cache_path, split_manifest, config)
        return cache_path
    started = time.monotonic()
    descriptors = []
    for index, row in enumerate(rows, start=1):
        actions = _read_18d_csv(row, verify_hash=config["verify_csv_hashes"])
        descriptors.append(
            motion_descriptor(actions, phase_samples=config["phase_samples"])
        )
        if index == 1 or index % 500 == 0 or index == len(rows):
            print(
                json.dumps(
                    {
                        "stage": "descriptor_cache",
                        "episodes": index,
                        "total": len(rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    raw = np.stack(descriptors).astype(np.float32, copy=False)
    split_names = np.asarray(
        [str(row["fixed_split_assignment"]) for row in rows], dtype="U10"
    )
    train_mask = split_names == "train"
    mean = raw[train_mask].mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = raw[train_mask].std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, np.float32(1e-5))
    standardized = ((raw - mean) / scale).astype(np.float32)
    standardized = np.clip(standardized, -20.0, 20.0)
    label_contract = _label_contract(split_manifest)
    labels = _encode_labels(rows, label_contract)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "beat2_18d_motion_descriptor_cache_v1",
        "data_policy": NO_EXTERNAL_DATA_POLICY,
        "no_kimodo": True,
        "manifest_sha256": split_manifest["manifest"]["sha256"],
        "csv_set_sha256": split_manifest["csv_set_sha256"],
        "phase_samples": int(config["phase_samples"]),
        "fps": FPS,
        "joint_names": list(JOINT_NAMES),
        "descriptor_dim": int(standardized.shape[1]),
        "normalization_fit_split": "train",
        "episode_count": len(rows),
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_npz_save(
        cache_path,
        descriptors=standardized,
        descriptor_mean=mean,
        descriptor_scale=scale,
        split_names=split_names,
        task_ids=np.asarray([str(row["task_id"]) for row in rows], dtype="U160"),
        speaker_keys=np.asarray([str(row["speaker_key"]) for row in rows], dtype="U64"),
        prompts=np.asarray([_prompt_text(row) for row in rows], dtype="U256"),
        category_labels=labels["category"],
        intensity_labels=labels["intensity"],
        emotion_labels=labels["emotion"],
        group_labels=labels["group"],
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    _atomic_json_save(metadata, output_dir / "descriptor_cache_summary.json")
    return cache_path


def load_descriptor_cache(
    path: str | Path,
    split_manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        cache = {name: archive[name] for name in archive.files}
    metadata = json.loads(str(cache.pop("metadata_json").item()))
    expected = {
        "artifact_kind": "beat2_18d_motion_descriptor_cache_v1",
        "data_policy": NO_EXTERNAL_DATA_POLICY,
        "no_kimodo": True,
        "manifest_sha256": split_manifest["manifest"]["sha256"],
        "csv_set_sha256": split_manifest["csv_set_sha256"],
        "phase_samples": int(config["phase_samples"]),
        "joint_names": list(JOINT_NAMES),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"descriptor cache metadata mismatch for {key}")
    if cache["descriptors"].shape[0] != int(metadata["episode_count"]):
        raise ValueError("descriptor cache episode count mismatch")
    if not np.isfinite(cache["descriptors"]).all():
        raise ValueError("descriptor cache contains non-finite values")
    cache["metadata"] = metadata
    return cache


class MotionEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        label_sizes: Mapping[str, int],
        *,
        dropout: float,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.input_dim),
        )
        self.category_head = nn.Linear(self.latent_dim, int(label_sizes["category"]))
        self.intensity_head = nn.Linear(self.latent_dim, int(label_sizes["intensity"]))
        self.emotion_head = nn.Linear(self.latent_dim, int(label_sizes["emotion"]))
        self.group_head = nn.Linear(self.latent_dim, int(label_sizes["group"]))

    def forward(self, descriptors: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.encoder(descriptors)
        return {
            "raw": raw,
            "embedding": F.normalize(raw, dim=-1),
            "reconstruction": self.decoder(raw),
            "category_logits": self.category_head(raw),
            "intensity_logits": self.intensity_head(raw),
            "emotion_logits": self.emotion_head(raw),
            "group_logits": self.group_head(raw),
        }


def _motion_loss(
    output: Mapping[str, torch.Tensor],
    descriptors: torch.Tensor,
    targets: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    reconstruction = F.smooth_l1_loss(output["reconstruction"], descriptors)
    category = F.cross_entropy(output["category_logits"], targets["category"])
    intensity = F.cross_entropy(output["intensity_logits"], targets["intensity"])
    emotion = F.cross_entropy(output["emotion_logits"], targets["emotion"])
    group = F.cross_entropy(output["group_logits"], targets["group"])
    total = (
        config["reconstruction_weight"] * reconstruction
        + config["category_weight"] * category
        + config["intensity_weight"] * intensity
        + config["emotion_weight"] * emotion
        + config["group_weight"] * group
    )
    return {
        "total": total,
        "reconstruction": reconstruction,
        "category": category,
        "intensity": intensity,
        "emotion": emotion,
        "group": group,
    }


def _encode_motion_array(
    model: MotionEncoder,
    descriptors: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 1024,
) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    was_training = model.training
    model.eval()
    embeddings = []
    logits: dict[str, list[np.ndarray]] = {
        "category": [],
        "intensity": [],
        "emotion": [],
        "group": [],
    }
    reconstruction_losses = []
    with torch.inference_mode():
        for start in range(0, len(descriptors), int(batch_size)):
            batch = torch.as_tensor(
                descriptors[start : start + int(batch_size)],
                dtype=torch.float32,
                device=device,
            )
            output = model(batch)
            embeddings.append(output["embedding"].cpu().numpy())
            for name in logits:
                logits[name].append(output[f"{name}_logits"].cpu().numpy())
            reconstruction_losses.append(
                (
                    F.smooth_l1_loss(
                        output["reconstruction"], batch, reduction="none"
                    )
                    .mean(dim=1)
                    .cpu()
                    .numpy()
                )
            )
    if was_training:
        model.train()
    return (
        np.concatenate(embeddings).astype(np.float32, copy=False),
        {name: np.concatenate(parts) for name, parts in logits.items()},
        float(np.concatenate(reconstruction_losses).mean()),
    )


def _nearest_centroid_accuracy(
    reference_embeddings: np.ndarray,
    reference_groups: np.ndarray,
    query_embeddings: np.ndarray,
    query_groups: np.ndarray,
    group_count: int,
) -> float:
    centroids = np.stack(
        [reference_embeddings[reference_groups == group].mean(axis=0) for group in range(group_count)]
    )
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-8)
    predicted = query_embeddings @ centroids.T
    return float((predicted.argmax(axis=1) == query_groups).mean())


def evaluate_motion_encoder(
    model: MotionEncoder,
    cache: Mapping[str, np.ndarray],
    *,
    split: str,
    device: torch.device,
    label_contract: Mapping[str, Any],
) -> tuple[dict, np.ndarray]:
    split_names = cache["split_names"]
    train_mask = split_names == "train"
    mask = split_names == split
    embeddings, logits, reconstruction = _encode_motion_array(
        model, cache["descriptors"][mask], device=device
    )
    expected = {
        "category": cache["category_labels"][mask],
        "intensity": cache["intensity_labels"][mask],
        "emotion": cache["emotion_labels"][mask],
        "group": cache["group_labels"][mask],
    }
    metrics = {
        "count": int(mask.sum()),
        "reconstruction_smooth_l1": reconstruction,
    }
    for name in expected:
        metrics[f"{name}_accuracy"] = float(
            (logits[name].argmax(axis=1) == expected[name]).mean()
        )
    train_embeddings, _, _ = _encode_motion_array(
        model, cache["descriptors"][train_mask], device=device
    )
    metrics["nearest_train_group_centroid_accuracy"] = _nearest_centroid_accuracy(
        train_embeddings,
        cache["group_labels"][train_mask],
        embeddings,
        expected["group"],
        len(label_contract["groups"]),
    )
    return metrics, embeddings


def _balanced_motion_batch_indices(
    group_labels: np.ndarray,
    *,
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    group_to_rows = [
        np.flatnonzero(group_labels == group)
        for group in range(int(group_labels.max()) + 1)
    ]
    if any(len(rows) == 0 for rows in group_to_rows):
        raise ValueError("balanced motion sampler found an empty train semantic group")
    sampled_groups = rng.integers(0, len(group_to_rows), size=int(batch_size))
    return np.asarray(
        [rows[int(rng.integers(0, len(rows)))] for rows in map(group_to_rows.__getitem__, sampled_groups)],
        dtype=np.int64,
    )


def _motion_checkpoint_payload(
    model: MotionEncoder,
    optimizer: torch.optim.Optimizer,
    *,
    config: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    label_contract: Mapping[str, Any],
    step: int,
    validation: Mapping[str, Any],
    best_score: Sequence[float],
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": MOTION_ARTIFACT_KIND,
        "data_policy": NO_EXTERNAL_DATA_POLICY,
        "no_kimodo": True,
        "initialization": "random_seeded_no_input_checkpoint",
        "input_checkpoint_count": 0,
        "step": int(step),
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": {
            "input_dim": model.input_dim,
            "hidden_dim": int(config["motion"]["hidden_dim"]),
            "latent_dim": model.latent_dim,
            "dropout": float(config["motion"]["dropout"]),
        },
        "label_contract": {
            key: [list(value) if isinstance(value, tuple) else value for value in values]
            for key, values in label_contract.items()
        },
        "source_manifest_sha256": split_manifest["manifest"]["sha256"],
        "csv_set_sha256": split_manifest["csv_set_sha256"],
        "validation": dict(validation),
        "best_score": list(best_score),
        "seed": int(config["seed"]),
    }


def train_motion_encoder(
    cache: Mapping[str, np.ndarray],
    split_manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> tuple[Path, dict]:
    output_dir = Path(config["output_dir"])
    best_path = output_dir / "motion_encoder_best.pt"
    summary_path = output_dir / "motion_encoder_summary.json"
    if best_path.is_file() and summary_path.is_file() and not overwrite:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        _load_motion_encoder(best_path, split_manifest, cache, device=torch.device("cpu"))
        return best_path, summary

    seed_everything(config["seed"])
    device = resolve_device(config["device"])
    labels = _label_contract(split_manifest)
    label_sizes = {
        "category": len(labels["categories"]),
        "intensity": len(labels["intensities"]),
        "emotion": len(labels["emotions"]),
        "group": len(labels["groups"]),
    }
    motion_config = config["motion"]
    model = MotionEncoder(
        cache["descriptors"].shape[1],
        motion_config["hidden_dim"],
        motion_config["latent_dim"],
        label_sizes,
        dropout=motion_config["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=motion_config["learning_rate"],
        weight_decay=motion_config["weight_decay"],
    )
    train_mask = cache["split_names"] == "train"
    train_descriptors = cache["descriptors"][train_mask]
    train_targets = {
        name: cache[f"{name}_labels"][train_mask]
        for name in ("category", "intensity", "emotion", "group")
    }
    rng = np.random.default_rng(int(config["seed"]))
    progress_path = output_dir / "motion_progress.jsonl"
    progress_path.write_text("", encoding="utf-8")
    best_score: tuple[float, ...] | None = None
    best_step = 0
    best_validation: dict = {}
    started = time.monotonic()
    for step in range(1, int(motion_config["steps"]) + 1):
        model.train()
        local_indices = _balanced_motion_batch_indices(
            train_targets["group"],
            batch_size=motion_config["batch_size"],
            rng=rng,
        )
        batch = torch.as_tensor(
            train_descriptors[local_indices], dtype=torch.float32, device=device
        )
        targets = {
            name: torch.as_tensor(values[local_indices], dtype=torch.long, device=device)
            for name, values in train_targets.items()
        }
        output = model(batch)
        losses = _motion_loss(output, batch, targets, motion_config)
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(f"non-finite motion encoder loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), motion_config["max_grad_norm"], error_if_nonfinite=True
        )
        optimizer.step()

        should_eval = (
            step == 1
            or step % int(motion_config["eval_interval"]) == 0
            or step == int(motion_config["steps"])
        )
        event = {
            "stage": "motion_encoder",
            "step": step,
            "steps": int(motion_config["steps"]),
            "loss": float(losses["total"].detach().cpu()),
            "reconstruction_loss": float(losses["reconstruction"].detach().cpu()),
            "category_loss": float(losses["category"].detach().cpu()),
            "intensity_loss": float(losses["intensity"].detach().cpu()),
            "emotion_loss": float(losses["emotion"].detach().cpu()),
            "group_loss": float(losses["group"].detach().cpu()),
            "grad_norm": float(grad_norm.detach().cpu()),
        }
        if should_eval:
            validation, _ = evaluate_motion_encoder(
                model,
                cache,
                split="validation",
                device=device,
                label_contract=labels,
            )
            score = (
                validation["nearest_train_group_centroid_accuracy"],
                validation["group_accuracy"],
                validation["category_accuracy"],
                validation["intensity_accuracy"],
                validation["emotion_accuracy"],
                -validation["reconstruction_smooth_l1"],
            )
            is_best = best_score is None or score > best_score
            if is_best:
                best_score = score
                best_step = step
                best_validation = validation
                _atomic_torch_save(
                    _motion_checkpoint_payload(
                        model,
                        optimizer,
                        config=config,
                        split_manifest=split_manifest,
                        label_contract=labels,
                        step=step,
                        validation=validation,
                        best_score=score,
                    ),
                    best_path,
                )
            event["validation"] = validation
            event["is_best"] = is_best
        if (
            step == 1
            or step % int(motion_config["log_interval"]) == 0
            or should_eval
            or step == int(motion_config["steps"])
        ):
            print(json.dumps(event, sort_keys=True), flush=True)
            _append_jsonl(event, progress_path)

    best_model, _ = _load_motion_encoder(best_path, split_manifest, cache, device=device)
    test_metrics, _ = evaluate_motion_encoder(
        best_model, cache, split="test", device=device, label_contract=labels
    )
    summary = {
        "artifact_kind": f"{MOTION_ARTIFACT_KIND}_summary",
        "data_policy": NO_EXTERNAL_DATA_POLICY,
        "no_kimodo": True,
        "initialization": "random_seeded_no_input_checkpoint",
        "best_step": best_step,
        "best_validation": best_validation,
        "test": test_metrics,
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint": str(best_path),
        "checkpoint_sha256": sha256_file(best_path),
    }
    _atomic_json_save(summary, summary_path)
    return best_path, summary


def _load_motion_encoder(
    path: str | Path,
    split_manifest: Mapping[str, Any],
    cache: Mapping[str, np.ndarray],
    *,
    device: torch.device,
) -> tuple[MotionEncoder, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "artifact_kind": MOTION_ARTIFACT_KIND,
        "data_policy": NO_EXTERNAL_DATA_POLICY,
        "no_kimodo": True,
        "source_manifest_sha256": split_manifest["manifest"]["sha256"],
        "csv_set_sha256": split_manifest["csv_set_sha256"],
        "initialization": "random_seeded_no_input_checkpoint",
        "input_checkpoint_count": 0,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"motion encoder checkpoint mismatch for {key}")
    model_config = checkpoint["model_config"]
    if int(model_config["input_dim"]) != cache["descriptors"].shape[1]:
        raise ValueError("motion encoder input dimension mismatch")
    labels = checkpoint["label_contract"]
    label_sizes = {
        "category": len(labels["categories"]),
        "intensity": len(labels["intensities"]),
        "emotion": len(labels["emotions"]),
        "group": len(labels["groups"]),
    }
    model = MotionEncoder(
        model_config["input_dim"],
        model_config["hidden_dim"],
        model_config["latent_dim"],
        label_sizes,
        dropout=model_config["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False).to(device).eval()
    return model, checkpoint


class TextAlignmentHead(nn.Module):
    def __init__(
        self,
        qwen_dim: int,
        hidden_dim: int,
        latent_dim: int,
        label_sizes: Mapping[str, int],
        *,
        dropout: float,
    ):
        super().__init__()
        self.qwen_dim = int(qwen_dim)
        self.latent_dim = int(latent_dim)
        self.projector = nn.Sequential(
            nn.Linear(self.qwen_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.latent_dim),
        )
        self.category_head = nn.Linear(self.latent_dim, int(label_sizes["category"]))
        self.intensity_head = nn.Linear(self.latent_dim, int(label_sizes["intensity"]))
        self.emotion_head = nn.Linear(self.latent_dim, int(label_sizes["emotion"]))

    def forward(self, qwen_components: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.projector(qwen_components.float())
        return {
            "raw": raw,
            "embedding": F.normalize(raw, dim=-1),
            "category_logits": self.category_head(raw),
            "intensity_logits": self.intensity_head(raw),
            "emotion_logits": self.emotion_head(raw),
        }


def format_qwen_instruction(text: str, instruction: str) -> str:
    text = str(text).strip()
    instruction = str(instruction).strip()
    if not text or not instruction:
        raise ValueError("Qwen instruction and motion prompt must be non-empty")
    return f"Instruct: {instruction}\nQuery:{text}"


def last_token_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    left_padding = bool(
        int(attention_mask[:, -1].sum().item()) == int(attention_mask.shape[0])
    )
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    rows = torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device)
    return last_hidden_states[rows, sequence_lengths]


def tokenize_prompts(
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    instruction: str,
    max_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    formatted = [format_qwen_instruction(prompt, instruction) for prompt in prompts]
    tokens = tokenizer(
        formatted,
        padding=True,
        truncation=True,
        max_length=int(max_length),
        return_tensors="pt",
    )
    return {name: value.to(device) for name, value in tokens.items()}


def _load_official_qwen_base(
    config: Mapping[str, Any], *, device: torch.device
) -> tuple[nn.Module, Any, dict]:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("transformers is required for the Qwen A/B stages") from exc
    if config["model_name"] != EXPECTED_QWEN_MODEL:
        raise ValueError("only the official pinned Qwen base model is allowed")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        revision=config["revision"],
        local_files_only=config["local_files_only"],
        padding_side="left",
        trust_remote_code=False,
    )
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    model = AutoModel.from_pretrained(
        config["model_name"],
        revision=config["revision"],
        local_files_only=config["local_files_only"],
        trust_remote_code=False,
        dtype=dtype,
        attn_implementation=config["attention_backend"],
    ).to(device)
    model.config.use_cache = False
    actual_revision = str(
        getattr(model.config, "_commit_hash", None) or config["revision"]
    )
    if actual_revision != config["revision"]:
        raise ValueError(
            f"resolved Qwen revision changed: {actual_revision} != {config['revision']}"
        )
    metadata = {
        "source": "official_huggingface_base",
        "model_name": config["model_name"],
        "revision": actual_revision,
        "hidden_size": int(model.config.hidden_size),
        "layers": int(model.config.num_hidden_layers),
        "dtype": str(dtype).replace("torch.", ""),
        "local_files_only": bool(config["local_files_only"]),
        "input_checkpoint_kind": "official_base_only",
    }
    if int(config["qwen_component_dim"]) > metadata["hidden_size"]:
        raise ValueError("qwen_component_dim exceeds the official base hidden size")
    return model, tokenizer, metadata


def _qwen_components(
    qwen: nn.Module,
    token_batch: Mapping[str, torch.Tensor],
    *,
    component_dim: int,
) -> torch.Tensor:
    output = qwen(**token_batch, use_cache=False)
    pooled = last_token_pool(output.last_hidden_state, token_batch["attention_mask"])
    return pooled[:, : int(component_dim)].float()


def _group_prompts(
    rows: Sequence[Mapping[str, Any]], label_contract: Mapping[str, Any]
) -> list[str]:
    prompt_sets: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in rows:
        prompt_sets[_semantic_key(row)].add(_prompt_text(row))
    prompts = []
    for group in label_contract["groups"]:
        values = prompt_sets.get(tuple(group), set())
        if len(values) != 1:
            raise ValueError(f"group {group} does not have exactly one canonical prompt")
        prompts.append(next(iter(values)))
    return prompts


def _diagnostic_template_probe_prompts(
    label_contract: Mapping[str, Any],
    canonical_prompts: Sequence[str],
) -> list[str]:
    """Create deterministic unseen wording for diagnostics, never supervision."""
    probes = [
        (
            f"Create an upper-body {category} gesture; move with {intensity} "
            f"intensity while conveying {emotion} affect."
        )
        for category, intensity, emotion in label_contract["groups"]
    ]
    canonical_hashes = {
        sha256_bytes(str(prompt).casefold().encode("utf-8"))
        for prompt in canonical_prompts
    }
    probe_hashes = {
        sha256_bytes(str(prompt).casefold().encode("utf-8")) for prompt in probes
    }
    if len(probes) != len(set(probes)) or canonical_hashes & probe_hashes:
        raise RuntimeError("diagnostic template probes are not unique and held out")
    return probes


def _group_semantic_targets(label_contract: Mapping[str, Any]) -> dict[str, np.ndarray]:
    category_to_index = {
        label: index for index, label in enumerate(label_contract["categories"])
    }
    intensity_to_index = {
        label: index for index, label in enumerate(label_contract["intensities"])
    }
    emotion_to_index = {
        label: index for index, label in enumerate(label_contract["emotions"])
    }
    groups = [tuple(group) for group in label_contract["groups"]]
    return {
        "category": np.asarray(
            [category_to_index[group[0]] for group in groups], dtype=np.int64
        ),
        "intensity": np.asarray(
            [intensity_to_index[group[1]] for group in groups], dtype=np.int64
        ),
        "emotion": np.asarray(
            [emotion_to_index[group[2]] for group in groups], dtype=np.int64
        ),
    }


def prepare_motion_alignment_data(
    motion_checkpoint_path: str | Path,
    cache: Mapping[str, np.ndarray],
    split_manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict:
    model, _ = _load_motion_encoder(
        motion_checkpoint_path, split_manifest, cache, device=device
    )
    embeddings, _, _ = _encode_motion_array(
        model, cache["descriptors"], device=device
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    group_count = len(split_manifest["labels"]["groups"])
    split_data = {}
    for split in SPLIT_NAMES:
        mask = cache["split_names"] == split
        split_embeddings = embeddings[mask]
        split_groups = cache["group_labels"][mask]
        available_groups = sorted(set(int(value) for value in split_groups.tolist()))
        centroids = np.stack(
            [
                split_embeddings[split_groups == group].mean(axis=0)
                for group in available_groups
            ]
        )
        centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-8)
        split_data[split] = {
            "embeddings": split_embeddings,
            "groups": split_groups,
            "available_groups": np.asarray(available_groups, dtype=np.int64),
            "centroids": centroids.astype(np.float32),
        }
    if len(split_data["train"]["available_groups"]) != group_count:
        raise ValueError("train split does not provide every semantic motion group")
    train_centroids = np.empty(
        (group_count, embeddings.shape[1]), dtype=np.float32
    )
    for row, group in enumerate(split_data["train"]["available_groups"]):
        train_centroids[int(group)] = split_data["train"]["centroids"][row]
    return {
        "all_embeddings": embeddings,
        "split": split_data,
        "train_centroids": train_centroids,
        "motion_checkpoint_sha256": sha256_file(motion_checkpoint_path),
    }


def bidirectional_group_retrieval_loss(
    text_embeddings: torch.Tensor,
    motion_centroids: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if text_embeddings.shape != motion_centroids.shape:
        raise ValueError("text embeddings and train motion centroids must have equal shape")
    logits = text_embeddings @ motion_centroids.T / float(temperature)
    target = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)
    )


def _text_alignment_loss(
    output: Mapping[str, torch.Tensor],
    motion_centroids: torch.Tensor,
    semantic_targets: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    retrieval = bidirectional_group_retrieval_loss(
        output["embedding"],
        motion_centroids,
        temperature=config["temperature"],
    )
    cosine = (
        1.0 - (output["embedding"] * motion_centroids).sum(dim=-1)
    ).mean()
    category = F.cross_entropy(
        output["category_logits"], semantic_targets["category"]
    )
    intensity = F.cross_entropy(
        output["intensity_logits"], semantic_targets["intensity"]
    )
    emotion = F.cross_entropy(output["emotion_logits"], semantic_targets["emotion"])
    total = (
        config["retrieval_weight"] * retrieval
        + config["cosine_weight"] * cosine
        + config["category_weight"] * category
        + config["intensity_weight"] * intensity
        + config["emotion_weight"] * emotion
    )
    return {
        "total": total,
        "retrieval": retrieval,
        "cosine": cosine,
        "category": category,
        "intensity": intensity,
        "emotion": emotion,
    }


def _retrieval_ranks(similarity: np.ndarray) -> np.ndarray:
    order = np.argsort(-np.asarray(similarity), axis=1, kind="stable")
    target = np.arange(order.shape[0])[:, None]
    return np.argmax(order == target, axis=1) + 1


def _first_positive_ranks(
    similarity: np.ndarray, positive_mask: np.ndarray
) -> np.ndarray:
    similarity = np.asarray(similarity)
    positive_mask = np.asarray(positive_mask, dtype=bool)
    if similarity.shape != positive_mask.shape:
        raise ValueError("similarity and positive mask shapes do not match")
    valid = positive_mask.any(axis=1)
    if not valid.any():
        raise ValueError("retrieval evaluation has no positive queries")
    order = np.argsort(-similarity[valid], axis=1, kind="stable")
    ordered_positive = np.take_along_axis(positive_mask[valid], order, axis=1)
    return np.argmax(ordered_positive, axis=1) + 1


def _effective_rank(embeddings: np.ndarray, eps: float = 1e-12) -> float:
    singular_values = np.linalg.svd(
        np.asarray(embeddings, dtype=np.float64), compute_uv=False
    )
    energy = singular_values**2
    probability = energy / max(float(energy.sum()), eps)
    return float(
        np.exp(-(probability * np.log(np.maximum(probability, eps))).sum())
    )


def evaluate_text_motion_alignment(
    text_output: Mapping[str, torch.Tensor | np.ndarray],
    alignment_data: Mapping[str, Any],
    *,
    split: str,
    semantic_targets: Mapping[str, np.ndarray],
    temperature: float,
) -> dict:
    text_embeddings = np.asarray(
        text_output["embedding"].detach().cpu()
        if isinstance(text_output["embedding"], torch.Tensor)
        else text_output["embedding"],
        dtype=np.float32,
    )
    split_data = alignment_data["split"][split]
    motion_embeddings = np.asarray(split_data["embeddings"], dtype=np.float32)
    motion_groups = np.asarray(split_data["groups"], dtype=np.int64)
    available = np.asarray(split_data["available_groups"], dtype=np.int64)
    centroids = np.asarray(split_data["centroids"], dtype=np.float32)
    text_available = text_embeddings[available]
    centroid_similarity = text_available @ centroids.T
    text_to_centroid_ranks = _retrieval_ranks(centroid_similarity)
    centroid_to_text_ranks = _retrieval_ranks(centroid_similarity.T)

    text_to_episode_similarity = text_embeddings @ motion_embeddings.T
    text_to_episode_positive = (
        np.arange(text_embeddings.shape[0])[:, None] == motion_groups[None, :]
    )
    text_to_episode_ranks = _first_positive_ranks(
        text_to_episode_similarity, text_to_episode_positive
    )
    episode_to_text_similarity = motion_embeddings @ text_embeddings.T
    episode_to_text_positive = (
        motion_groups[:, None] == np.arange(text_embeddings.shape[0])[None, :]
    )
    episode_to_text_ranks = _first_positive_ranks(
        episode_to_text_similarity, episode_to_text_positive
    )
    predicted_group = episode_to_text_similarity.argmax(axis=1)
    positive_cosine = episode_to_text_similarity[
        np.arange(len(motion_groups)), motion_groups
    ]
    negative_mask = ~episode_to_text_positive
    negative_cosine = float(episode_to_text_similarity[negative_mask].mean())

    semantic_predictions = {
        name: values[predicted_group] for name, values in semantic_targets.items()
    }
    semantic_expected = {
        name: values[motion_groups] for name, values in semantic_targets.items()
    }
    text_logits = {}
    for name in ("category", "intensity", "emotion"):
        value = text_output[f"{name}_logits"]
        text_logits[name] = np.asarray(
            value.detach().cpu() if isinstance(value, torch.Tensor) else value
        )
    joint_motion_correct = np.ones(len(motion_groups), dtype=bool)
    metrics = {
        "split": split,
        "episode_count": int(len(motion_groups)),
        "semantic_group_count": int(len(available)),
        "group_text_to_motion_recall_at_1": float(
            np.mean(text_to_centroid_ranks <= 1)
        ),
        "group_text_to_motion_recall_at_5": float(
            np.mean(text_to_centroid_ranks <= 5)
        ),
        "group_motion_to_text_recall_at_1": float(
            np.mean(centroid_to_text_ranks <= 1)
        ),
        "group_motion_to_text_recall_at_5": float(
            np.mean(centroid_to_text_ranks <= 5)
        ),
        "group_text_to_motion_median_rank": float(
            np.median(text_to_centroid_ranks)
        ),
        "group_motion_to_text_median_rank": float(
            np.median(centroid_to_text_ranks)
        ),
        "episode_text_to_motion_recall_at_1": float(
            np.mean(text_to_episode_ranks <= 1)
        ),
        "episode_text_to_motion_recall_at_5": float(
            np.mean(text_to_episode_ranks <= 5)
        ),
        "episode_text_to_motion_recall_at_10": float(
            np.mean(text_to_episode_ranks <= 10)
        ),
        "episode_motion_to_text_recall_at_1": float(
            np.mean(episode_to_text_ranks <= 1)
        ),
        "episode_motion_to_text_recall_at_5": float(
            np.mean(episode_to_text_ranks <= 5)
        ),
        "episode_motion_to_text_median_rank": float(
            np.median(episode_to_text_ranks)
        ),
        "positive_episode_cosine": float(positive_cosine.mean()),
        "negative_episode_cosine": negative_cosine,
        "episode_cosine_gap": float(positive_cosine.mean() - negative_cosine),
        "episode_motion_latent_rmse": float(
            np.sqrt(
                np.mean(
                    np.square(
                        motion_embeddings - text_embeddings[motion_groups]
                    )
                )
            )
        ),
        "group_centroid_positive_cosine": float(
            np.diagonal(centroid_similarity).mean()
        ),
        "text_effective_rank": _effective_rank(text_embeddings),
        "motion_effective_rank": _effective_rank(motion_embeddings),
    }
    for name in ("category", "intensity", "emotion"):
        correct = semantic_predictions[name] == semantic_expected[name]
        joint_motion_correct &= correct
        metrics[f"retrieved_motion_{name}_accuracy"] = float(correct.mean())
        metrics[f"text_{name}_head_accuracy"] = float(
            (text_logits[name].argmax(axis=1) == semantic_targets[name]).mean()
        )
    metrics["retrieved_motion_joint_semantic_accuracy"] = float(
        joint_motion_correct.mean()
    )
    logits = torch.as_tensor(centroid_similarity / float(temperature))
    target = torch.arange(len(available))
    metrics["group_retrieval_loss"] = float(
        0.5
        * (
            F.cross_entropy(logits, target)
            + F.cross_entropy(logits.T, target)
        )
    )
    return metrics


def _alignment_selection_score(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        0.5
        * (
            float(metrics["group_text_to_motion_recall_at_1"])
            + float(metrics["group_motion_to_text_recall_at_1"])
        ),
        float(metrics["episode_motion_to_text_recall_at_1"]),
        float(metrics["episode_cosine_gap"]),
        -float(metrics["episode_motion_latent_rmse"]),
    )


def _alignment_lr_scale(
    step: int, *, total_steps: int, warmup_steps: int, minimum_ratio: float
) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return step / warmup_steps
    if step >= total_steps:
        return float(minimum_ratio)
    denominator = max(total_steps - warmup_steps, 1)
    progress = (step - warmup_steps) / denominator
    cosine = 0.5 * (1.0 + math.cos(math.pi * max(0.0, progress)))
    return float(minimum_ratio) + (1.0 - float(minimum_ratio)) * cosine


def _new_text_head(
    config: Mapping[str, Any],
    label_contract: Mapping[str, Any],
    *,
    device: torch.device,
) -> TextAlignmentHead:
    label_sizes = {
        "category": len(label_contract["categories"]),
        "intensity": len(label_contract["intensities"]),
        "emotion": len(label_contract["emotions"]),
    }
    with torch.random.fork_rng(
        devices=[device] if device.type == "cuda" else [], enabled=True
    ):
        torch.manual_seed(int(config["seed"]) + 1009)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(config["seed"]) + 1009)
        head = TextAlignmentHead(
            config["qwen_component_dim"],
            config["alignment"]["hidden_dim"],
            config["motion"]["latent_dim"],
            label_sizes,
            dropout=config["alignment"]["dropout"],
        ).to(device)
    return head


def _text_checkpoint_payload(
    *,
    variant: str,
    head: TextAlignmentHead,
    qwen: nn.Module | None,
    qwen_metadata: Mapping[str, Any],
    config: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    alignment_data: Mapping[str, Any],
    step: int,
    validation: Mapping[str, Any],
    best_score: Sequence[float],
) -> dict:
    artifact_kind = (
        BASELINE_ARTIFACT_KIND if variant == "frozen_base" else LORA_ARTIFACT_KIND
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": artifact_kind,
        "variant": variant,
        "data_policy": NO_EXTERNAL_DATA_POLICY,
        "no_kimodo": True,
        "latent_dim": 128,
        "qwen_policy": (
            "official_base_frozen"
            if variant == "frozen_base"
            else "official_base_plus_beat2_only_lora"
        ),
        "qwen": dict(qwen_metadata),
        "text_head_initialization_seed": int(config["seed"]) + 1009,
        "text_head_state_dict": {
            name: value.detach().cpu() for name, value in head.state_dict().items()
        },
        "step": int(step),
        "validation": dict(validation),
        "best_score": list(best_score),
        "sources": {
            "manifest_sha256": split_manifest["manifest"]["sha256"],
            "csv_set_sha256": split_manifest["csv_set_sha256"],
            "prompt_set_sha256": split_manifest["prompt_set_sha256"],
            "motion_encoder_checkpoint_sha256": alignment_data[
                "motion_checkpoint_sha256"
            ],
            "input_motion_or_text_checkpoint_count": 0,
        },
        "alignment_config": deepcopy(config["alignment"]),
        "qwen_component_dim": int(config["qwen_component_dim"]),
        "instruction": config["instruction"],
        "semantic_scope": config["semantic_scope_acknowledgement"],
    }
    if variant == "lora_finetuned":
        if qwen is None:
            raise ValueError("LoRA checkpoint requires the in-memory Qwen PEFT model")
        from peft import get_peft_model_state_dict

        payload["qwen_lora_state_dict"] = {
            name: value.detach().cpu()
            for name, value in get_peft_model_state_dict(qwen).items()
        }
    else:
        payload["qwen_lora_state_dict"] = None
    return payload


def export_128d_condition_cache(
    *,
    variant: str,
    rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    split_manifest: Mapping[str, Any],
    group_text_embeddings: np.ndarray,
    adapter_checkpoint_path: str | Path,
    qwen_metadata: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[Path, dict]:
    group_text_embeddings = np.asarray(group_text_embeddings, dtype=np.float32)
    if group_text_embeddings.shape != (
        len(split_manifest["labels"]["groups"]),
        128,
    ):
        raise ValueError(
            f"{variant} group text latents must have shape "
            f"[{len(split_manifest['labels']['groups'])}, 128]"
        )
    conditions = group_text_embeddings[cache["group_labels"]]
    if conditions.shape != (len(rows), 128) or not np.isfinite(conditions).all():
        raise ValueError(f"{variant} condition cache has an invalid shape or values")
    output_dir = Path(config["output_dir"])
    path = output_dir / f"conditions_128d_{variant}.npz"
    _atomic_npz_save(
        path,
        clip_ids=np.asarray([str(row["clip_id"]) for row in rows], dtype="U180"),
        task_ids=cache["task_ids"],
        prompts=cache["prompts"],
        conditions=conditions,
        motion_latents=conditions,
        fixed_split_assignments=cache["split_names"],
        speaker_keys=cache["speaker_keys"],
        semantic_group_indices=cache["group_labels"],
        trajectory_sha256=np.asarray(
            [str(row["safe_csv_sha256"]) for row in rows], dtype="U64"
        ),
    )
    checkpoint_path = Path(adapter_checkpoint_path).resolve()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "beat2_qwen_motion_latent_condition_cache_v1",
        "variant": variant,
        "data_policy": NO_EXTERNAL_DATA_POLICY,
        "no_kimodo": True,
        "condition_dim": 128,
        "motion_latent_dim": 128,
        "base_condition_dim": 0,
        "count": len(rows),
        "unique_canonical_prompt_count": 54,
        "unique_condition_count": int(
            np.unique(conditions, axis=0).shape[0]
        ),
        "condition_normalization": "unit_l2_per_canonical_prompt",
        "cache_sha256": sha256_file(path),
        "source_manifest_sha256": split_manifest["manifest"]["sha256"],
        "csv_set_sha256": split_manifest["csv_set_sha256"],
        "prompt_set_sha256": split_manifest["prompt_set_sha256"],
        "speaker_split_contract": "fixed_17_train_4_validation_4_test",
        "clip_order_sha256": sha256_bytes(
            _canonical_json([str(row["clip_id"]) for row in rows])
        ),
        "trajectory_order_sha256": sha256_bytes(
            _canonical_json([str(row["safe_csv_sha256"]) for row in rows])
        ),
        "adapter_checkpoint": str(checkpoint_path),
        "adapter_checkpoint_sha256": sha256_file(checkpoint_path),
        "qwen": dict(qwen_metadata),
        "semantic_scope": config["semantic_scope_acknowledgement"],
        "generator_contract": {
            "role": "direct_128d_text_to_learned_motion_latent_condition",
            "consumer": "clean_beat2_latent_to_motion_prior",
            "requires_same_foundation_for_ab": True,
            "condition_is_not_episode_motion_encoder_output": True,
            "condition_is_text_prediction_in_shared_motion_latent_space": True,
        },
    }
    _atomic_json_save(metadata, path.with_suffix(path.suffix + ".json"))
    return path, metadata


def _state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(_canonical_json(list(value.shape)))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _apply_beat2_lora(
    base_qwen: nn.Module,
    config: Mapping[str, Any],
    qwen_metadata: Mapping[str, Any],
) -> tuple[nn.Module, dict]:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("peft is required for the BEAT2-only LoRA branch") from exc
    alignment = config["alignment"]
    layer_count = int(qwen_metadata["layers"])
    top_layers = int(alignment["lora_top_layers"])
    if top_layers > layer_count:
        raise ValueError("lora_top_layers exceeds the official Qwen base layer count")
    layer_indices = list(range(layer_count - top_layers, layer_count))
    target_modules = [
        f"layers.{layer}.{projection}"
        if ".self_attn." in projection
        else f"layers.{layer}.self_attn.{projection}"
        for layer in layer_indices
        for projection in alignment["lora_target_projections"]
    ]
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=int(alignment["lora_rank"]),
        lora_alpha=int(alignment["lora_alpha"]),
        lora_dropout=float(alignment["lora_dropout"]),
        target_modules=target_modules,
        bias="none",
    )
    qwen = get_peft_model(base_qwen, lora_config)
    trainable_names = [
        name for name, parameter in qwen.named_parameters() if parameter.requires_grad
    ]
    if not trainable_names or any("lora_" not in name for name in trainable_names):
        raise RuntimeError("Qwen LoRA branch exposed non-LoRA trainable parameters")
    metadata = dict(qwen_metadata)
    metadata["lora"] = {
        "training_data": "BEAT2_only",
        "rank": int(alignment["lora_rank"]),
        "alpha": int(alignment["lora_alpha"]),
        "dropout": float(alignment["lora_dropout"]),
        "layer_indices": layer_indices,
        "target_modules": target_modules,
        "trainable_parameter_names": trainable_names,
        "trainable_parameters": int(
            sum(
                parameter.numel()
                for parameter in qwen.parameters()
                if parameter.requires_grad
            )
        ),
    }
    return qwen, metadata


def train_text_alignment_variant(
    variant: str,
    *,
    rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    split_manifest: Mapping[str, Any],
    motion_checkpoint_path: str | Path,
    config: Mapping[str, Any],
    overwrite: bool = False,
) -> tuple[Path, Path, dict]:
    if variant not in {"frozen_base", "lora_finetuned"}:
        raise ValueError(f"unknown Qwen A/B variant: {variant}")
    output_dir = Path(config["output_dir"])
    checkpoint_path = output_dir / f"qwen_{variant}_best.pt"
    summary_path = output_dir / f"qwen_{variant}_summary.json"
    condition_path = output_dir / f"conditions_128d_{variant}.npz"
    if (
        checkpoint_path.is_file()
        and summary_path.is_file()
        and condition_path.is_file()
        and not overwrite
    ):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        expected_kind = (
            BASELINE_ARTIFACT_KIND
            if variant == "frozen_base"
            else LORA_ARTIFACT_KIND
        )
        if (
            checkpoint.get("artifact_kind") != expected_kind
            or checkpoint.get("no_kimodo") is not True
            or checkpoint.get("sources", {}).get("manifest_sha256")
            != split_manifest["manifest"]["sha256"]
        ):
            raise ValueError(f"existing {variant} checkpoint provenance mismatch")
        metadata = json.loads(
            condition_path.with_suffix(condition_path.suffix + ".json").read_text(
                encoding="utf-8"
            )
        )
        if metadata.get("cache_sha256") != sha256_file(condition_path):
            raise ValueError(f"existing {variant} condition cache hash mismatch")
        return checkpoint_path, condition_path, json.loads(
            summary_path.read_text(encoding="utf-8")
        )

    seed_everything(config["seed"])
    device = resolve_device(config["device"])
    label_contract = _label_contract(split_manifest)
    prompts = _group_prompts(rows, label_contract)
    probe_prompts = _diagnostic_template_probe_prompts(label_contract, prompts)
    semantic_numpy = _group_semantic_targets(label_contract)
    semantic_targets = {
        name: torch.as_tensor(values, dtype=torch.long, device=device)
        for name, values in semantic_numpy.items()
    }
    alignment_data = prepare_motion_alignment_data(
        motion_checkpoint_path, cache, split_manifest, config, device=device
    )
    motion_centroids = torch.as_tensor(
        alignment_data["train_centroids"], dtype=torch.float32, device=device
    )
    if motion_centroids.shape[1] != 128:
        raise ValueError("BEAT2 motion encoder did not produce the required 128D latent")

    base_qwen, tokenizer, qwen_metadata = _load_official_qwen_base(
        config, device=device
    )
    token_batch = tokenize_prompts(
        tokenizer,
        prompts,
        instruction=config["instruction"],
        max_length=config["max_length"],
        device=device,
    )
    probe_token_batch = tokenize_prompts(
        tokenizer,
        probe_prompts,
        instruction=config["instruction"],
        max_length=config["max_length"],
        device=device,
    )
    fixed_components: torch.Tensor | None = None
    fixed_probe_components: torch.Tensor | None = None
    qwen: nn.Module | None
    if variant == "frozen_base":
        base_qwen.requires_grad_(False).eval()
        with torch.no_grad():
            fixed_components = _qwen_components(
                base_qwen,
                token_batch,
                component_dim=config["qwen_component_dim"],
            ).detach().clone()
            fixed_probe_components = _qwen_components(
                base_qwen,
                probe_token_batch,
                component_dim=config["qwen_component_dim"],
            ).detach().clone()
        qwen = None
        del base_qwen
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        qwen, qwen_metadata = _apply_beat2_lora(
            base_qwen, config, qwen_metadata
        )

    head = _new_text_head(config, label_contract, device=device)
    initial_head_sha256 = _state_dict_sha256(head.state_dict())
    alignment_config = config["alignment"]
    optimizer_groups = [
        {
            "params": list(head.parameters()),
            "lr": float(alignment_config["projector_lr"]),
            "base_lr": float(alignment_config["projector_lr"]),
            "name": "identical_text_projector",
        }
    ]
    if qwen is not None:
        optimizer_groups.append(
            {
                "params": [
                    parameter
                    for parameter in qwen.parameters()
                    if parameter.requires_grad
                ],
                "lr": float(alignment_config["lora_lr"]),
                "base_lr": float(alignment_config["lora_lr"]),
                "name": "beat2_only_qwen_lora",
            }
        )
    optimizer = torch.optim.AdamW(
        optimizer_groups, weight_decay=float(alignment_config["weight_decay"])
    )
    trainable_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    total_steps = int(
        alignment_config[
            "frozen_steps" if variant == "frozen_base" else "lora_steps"
        ]
    )
    progress_path = output_dir / f"qwen_{variant}_progress.jsonl"
    progress_path.write_text("", encoding="utf-8")
    best_score: tuple[float, ...] | None = None
    best_step = 0
    best_validation: dict = {}
    started = time.monotonic()

    def forward_text(*, training: bool) -> dict[str, torch.Tensor]:
        if qwen is None:
            if fixed_components is None:
                raise RuntimeError("frozen Qwen components were not prepared")
            return head(fixed_components)
        if training:
            qwen.train()
            head.train()
        else:
            qwen.eval()
            head.eval()
        components = _qwen_components(
            qwen,
            token_batch,
            component_dim=config["qwen_component_dim"],
        )
        return head(components)

    def forward_probe() -> dict[str, torch.Tensor]:
        if qwen is None:
            if fixed_probe_components is None:
                raise RuntimeError("frozen Qwen probe components were not prepared")
            return head(fixed_probe_components)
        components = _qwen_components(
            qwen,
            probe_token_batch,
            component_dim=config["qwen_component_dim"],
        )
        return head(components)

    for step in range(1, total_steps + 1):
        head.train()
        output = forward_text(training=True)
        losses = _text_alignment_loss(
            output, motion_centroids, semantic_targets, alignment_config
        )
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(f"non-finite {variant} loss at step {step}")
        scale = _alignment_lr_scale(
            step,
            total_steps=total_steps,
            warmup_steps=min(
                int(alignment_config["warmup_steps"]), max(total_steps - 1, 0)
            ),
            minimum_ratio=alignment_config["minimum_lr_ratio"],
        )
        for group in optimizer.param_groups:
            group["lr"] = float(group["base_lr"]) * scale
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            float(alignment_config["max_grad_norm"]),
            error_if_nonfinite=True,
        )
        optimizer.step()

        should_eval = (
            step == 1
            or step % int(alignment_config["eval_interval"]) == 0
            or step == total_steps
        )
        event = {
            "stage": "qwen_text_to_motion_latent",
            "variant": variant,
            "step": step,
            "steps": total_steps,
            "loss": float(losses["total"].detach().cpu()),
            "retrieval_loss": float(losses["retrieval"].detach().cpu()),
            "cosine_loss": float(losses["cosine"].detach().cpu()),
            "category_loss": float(losses["category"].detach().cpu()),
            "intensity_loss": float(losses["intensity"].detach().cpu()),
            "emotion_loss": float(losses["emotion"].detach().cpu()),
            "grad_norm": float(grad_norm.detach().cpu()),
            "lr_scale": scale,
        }
        if should_eval:
            head.eval()
            if qwen is not None:
                qwen.eval()
            with torch.inference_mode():
                validation_output = forward_text(training=False)
                probe_validation_output = forward_probe()
            validation = evaluate_text_motion_alignment(
                validation_output,
                alignment_data,
                split="validation",
                semantic_targets=semantic_numpy,
                temperature=alignment_config["temperature"],
            )
            validation["diagnostic_unseen_template_probe"] = (
                evaluate_text_motion_alignment(
                    probe_validation_output,
                    alignment_data,
                    split="validation",
                    semantic_targets=semantic_numpy,
                    temperature=alignment_config["temperature"],
                )
            )
            score = _alignment_selection_score(validation)
            is_best = best_score is None or score > best_score
            if is_best:
                best_score = score
                best_step = step
                best_validation = validation
                payload = _text_checkpoint_payload(
                    variant=variant,
                    head=head,
                    qwen=qwen,
                    qwen_metadata=qwen_metadata,
                    config=config,
                    split_manifest=split_manifest,
                    alignment_data=alignment_data,
                    step=step,
                    validation=validation,
                    best_score=score,
                )
                payload["text_head_initial_state_sha256"] = initial_head_sha256
                _atomic_torch_save(payload, checkpoint_path)
            event["validation"] = validation
            event["is_best"] = is_best
        if (
            step == 1
            or step % int(alignment_config["log_interval"]) == 0
            or should_eval
            or step == total_steps
        ):
            print(json.dumps(event, sort_keys=True), flush=True)
            _append_jsonl(event, progress_path)

    best_checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    head.load_state_dict(best_checkpoint["text_head_state_dict"], strict=True)
    if qwen is not None:
        from peft import set_peft_model_state_dict

        load_result = set_peft_model_state_dict(
            qwen, best_checkpoint["qwen_lora_state_dict"]
        )
        if getattr(load_result, "unexpected_keys", None):
            raise ValueError(
                f"unexpected BEAT2 LoRA keys: {load_result.unexpected_keys}"
            )
    head.eval()
    if qwen is not None:
        qwen.eval()
    with torch.inference_mode():
        best_output = forward_text(training=False)
        best_probe_output = forward_probe()
    test_metrics = evaluate_text_motion_alignment(
        best_output,
        alignment_data,
        split="test",
        semantic_targets=semantic_numpy,
        temperature=alignment_config["temperature"],
    )
    probe_test_metrics = evaluate_text_motion_alignment(
        best_probe_output,
        alignment_data,
        split="test",
        semantic_targets=semantic_numpy,
        temperature=alignment_config["temperature"],
    )
    group_text_embeddings = (
        best_output["embedding"].detach().cpu().numpy().astype(np.float32)
    )
    condition_path, condition_metadata = export_128d_condition_cache(
        variant=variant,
        rows=rows,
        cache=cache,
        split_manifest=split_manifest,
        group_text_embeddings=group_text_embeddings,
        adapter_checkpoint_path=checkpoint_path,
        qwen_metadata=qwen_metadata,
        config=config,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": f"{best_checkpoint['artifact_kind']}_summary",
        "variant": variant,
        "data_policy": NO_EXTERNAL_DATA_POLICY,
        "no_kimodo": True,
        "latent_role": "text_to_learned_128d_motion_latent",
        "best_step": best_step,
        "best_validation": best_validation,
        "test": test_metrics,
        "diagnostic_unseen_template_probe_test": probe_test_metrics,
        "diagnostic_probe_scope": (
            "deterministic held-out wording only; not human-authored, not formal "
            "semantic supervision, and not an open-text benchmark"
        ),
        "unique_canonical_training_prompts": 54,
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "condition_cache": str(condition_path),
        "condition_cache_sha256": condition_metadata["cache_sha256"],
        "condition_dim": 128,
        "motion_encoder_checkpoint_sha256": alignment_data[
            "motion_checkpoint_sha256"
        ],
        "text_head_initial_state_sha256": initial_head_sha256,
        "qwen": qwen_metadata,
    }
    _atomic_json_save(summary, summary_path)
    del head, optimizer
    if qwen is not None:
        del qwen
    del tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return checkpoint_path, condition_path, summary


def _numeric_delta(left: Any, right: Any) -> Any:
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        return float(right) - float(left)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return {
            key: _numeric_delta(left[key], right[key])
            for key in sorted(set(left) & set(right))
            if isinstance(left[key], (Mapping, int, float))
            and isinstance(right[key], (Mapping, int, float))
        }
    return None


def build_ab_comparison(
    config: Mapping[str, Any], split_manifest: Mapping[str, Any]
) -> dict:
    output_dir = Path(config["output_dir"])
    baseline_summary = json.loads(
        (output_dir / "qwen_frozen_base_summary.json").read_text(encoding="utf-8")
    )
    lora_summary = json.loads(
        (output_dir / "qwen_lora_finetuned_summary.json").read_text(
            encoding="utf-8"
        )
    )
    baseline_cache = output_dir / "conditions_128d_frozen_base.npz"
    lora_cache = output_dir / "conditions_128d_lora_finetuned.npz"
    with np.load(baseline_cache, allow_pickle=False) as left, np.load(
        lora_cache, allow_pickle=False
    ) as right:
        for name in (
            "clip_ids",
            "task_ids",
            "prompts",
            "fixed_split_assignments",
            "speaker_keys",
            "semantic_group_indices",
            "trajectory_sha256",
        ):
            if not np.array_equal(left[name], right[name]):
                raise ValueError(f"B/C condition cache pairing mismatch for {name}")
        if left["conditions"].shape != right["conditions"].shape or left[
            "conditions"
        ].shape[1] != 128:
            raise ValueError("B/C condition caches must share an [episodes, 128] shape")
        mean_condition_delta = float(
            np.linalg.norm(
                left["conditions"].astype(np.float64)
                - right["conditions"].astype(np.float64),
                axis=1,
            ).mean()
        )
    if (
        baseline_summary["text_head_initial_state_sha256"]
        != lora_summary["text_head_initial_state_sha256"]
    ):
        raise ValueError("B/C projectors did not start from identical initialization")
    if (
        baseline_summary["motion_encoder_checkpoint_sha256"]
        != lora_summary["motion_encoder_checkpoint_sha256"]
    ):
        raise ValueError("B/C branches did not use the same learned motion latent space")
    if config["alignment"]["frozen_steps"] != config["alignment"]["lora_steps"]:
        raise ValueError("B/C Qwen branches do not have equal optimization steps")

    foundation = config["generator_pairing"]["foundation_checkpoint"]
    foundation_binding = {
        "status": "awaiting_clean_foundation_checkpoint",
        "path": None,
        "sha256": None,
    }
    if foundation is not None:
        validation = validate_clean_generator_foundation(foundation)
        foundation_binding = {
            "status": "bound",
            **validation,
        }
    comparison = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": f"{ARTIFACT_KIND}_comparison",
        "data_policy": NO_EXTERNAL_DATA_POLICY,
        "no_kimodo": True,
        "question": "Does BEAT2-only Qwen LoRA improve text-to-motion latent conditioning over a frozen official Qwen base?",
        "controlled_variables": {
            "source_manifest_sha256": split_manifest["manifest"]["sha256"],
            "csv_set_sha256": split_manifest["csv_set_sha256"],
            "prompt_set_sha256": split_manifest["prompt_set_sha256"],
            "speaker_split": "fixed_17_train_4_validation_4_test",
            "motion_encoder_checkpoint_sha256": baseline_summary[
                "motion_encoder_checkpoint_sha256"
            ],
            "condition_dim": 128,
            "projector_initial_state_sha256": baseline_summary[
                "text_head_initial_state_sha256"
            ],
            "projector_steps_each": int(config["alignment"]["frozen_steps"]),
            "projector_learning_rate": float(
                config["alignment"]["projector_lr"]
            ),
            "loss_and_evaluation_protocol_identical": True,
        },
        "independent_variable": {
            "frozen_base": "official Qwen base frozen; projector trainable",
            "lora_finetuned": "same official Qwen base plus BEAT2-only LoRA; identical projector trainable",
        },
        "baseline": baseline_summary,
        "lora_finetuned": lora_summary,
        "delta_lora_minus_frozen": {
            "validation": _numeric_delta(
                baseline_summary["best_validation"],
                lora_summary["best_validation"],
            ),
            "test": _numeric_delta(
                baseline_summary["test"], lora_summary["test"]
            ),
        },
        "condition_cache_pair": {
            "frozen_base": str(baseline_cache),
            "frozen_base_sha256": sha256_file(baseline_cache),
            "lora_finetuned": str(lora_cache),
            "lora_finetuned_sha256": sha256_file(lora_cache),
            "same_clip_prompt_split_order": True,
            "shape": [split_manifest["selected_episode_count"], 128],
            "mean_per_episode_l2_delta": mean_condition_delta,
        },
        "generator_posttrain_pairing": {
            "foundation": foundation_binding,
            "required_foundation_policy": "clean_random_init_BEAT2_only_no_external_checkpoint",
            "same_foundation_checkpoint_required": True,
            "same_optimizer_seed": int(
                config["generator_pairing"]["posttrain_seed"]
            ),
            "same_posttrain_steps": int(
                config["generator_pairing"]["posttrain_steps"]
            ),
            "only_allowed_difference": "128D condition cache / Qwen policy",
            "status": (
                "ready_for_paired_generator_posttrain"
                if foundation_binding["status"] == "bound"
                else "condition_artifacts_ready_foundation_binding_pending"
            ),
        },
        "semantic_scope": config["semantic_scope_acknowledgement"],
    }
    _atomic_json_save(comparison, output_dir / "comparison.json")
    return comparison


def _prepare_pipeline_inputs(
    config: Mapping[str, Any],
    *,
    max_records_per_group_per_split: int | None,
    overwrite: bool,
) -> tuple[list[dict], dict, dict, Path]:
    rows, split_manifest = audit_dataset(
        config,
        max_records_per_group_per_split=max_records_per_group_per_split,
        write_outputs=True,
    )
    cache_path = build_descriptor_cache(
        rows, split_manifest, config, overwrite=overwrite
    )
    cache = load_descriptor_cache(cache_path, split_manifest, config)
    motion_path, _ = train_motion_encoder(
        cache, split_manifest, config, overwrite=overwrite
    )
    return rows, split_manifest, cache, motion_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the strictly BEAT2-only frozen-Qwen versus BEAT2-LoRA "
            "text-to-128D-motion-latent controlled experiment"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        choices=(
            "audit",
            "cache",
            "motion",
            "frozen",
            "lora",
            "compare",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use one episode per semantic group/split and two optimizer steps",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.output_dir:
        config["output_dir"] = str(args.output_dir)
    if args.device:
        config["device"] = str(args.device)
    group_cap = None
    if args.smoke_test:
        if not args.output_dir:
            output = Path(config["output_dir"])
            config["output_dir"] = str(output.with_name(output.name + "_smoke"))
        group_cap = 1
        config["motion"]["steps"] = 2
        config["motion"]["eval_interval"] = 1
        config["motion"]["log_interval"] = 1
        config["motion"]["batch_size"] = min(
            64, int(config["motion"]["batch_size"])
        )
        config["alignment"]["frozen_steps"] = 2
        config["alignment"]["lora_steps"] = 2
        config["alignment"]["warmup_steps"] = 0
        config["alignment"]["eval_interval"] = 1
        config["alignment"]["log_interval"] = 1
    config = validate_config(config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json_save(
        {
            **config,
            "runtime": {
                "smoke_test": bool(args.smoke_test),
                "stage": args.stage,
                "max_records_per_group_per_split": group_cap,
            },
        },
        output_dir / "resolved_config.json",
    )

    if args.stage == "audit":
        audit_dataset(
            config,
            max_records_per_group_per_split=group_cap,
            write_outputs=True,
        )
        audit = json.loads(
            (Path(config["output_dir"]) / "audit.json").read_text(encoding="utf-8")
        )
        print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    rows, split_manifest = audit_dataset(
        config,
        max_records_per_group_per_split=group_cap,
        write_outputs=True,
    )
    cache_path = build_descriptor_cache(
        rows, split_manifest, config, overwrite=args.overwrite
    )
    if args.stage == "cache":
        print(
            json.dumps(
                {
                    "stage": "cache",
                    "cache": str(cache_path),
                    "sha256": sha256_file(cache_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    cache = load_descriptor_cache(cache_path, split_manifest, config)
    motion_path, motion_summary = train_motion_encoder(
        cache, split_manifest, config, overwrite=args.overwrite
    )
    if args.stage == "motion":
        print(json.dumps(motion_summary, indent=2, sort_keys=True))
        return 0

    if args.stage in {"frozen", "all"}:
        _, _, summary = train_text_alignment_variant(
            "frozen_base",
            rows=rows,
            cache=cache,
            split_manifest=split_manifest,
            motion_checkpoint_path=motion_path,
            config=config,
            overwrite=args.overwrite,
        )
        if args.stage == "frozen":
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
    if args.stage in {"lora", "all"}:
        _, _, summary = train_text_alignment_variant(
            "lora_finetuned",
            rows=rows,
            cache=cache,
            split_manifest=split_manifest,
            motion_checkpoint_path=motion_path,
            config=config,
            overwrite=args.overwrite,
        )
        if args.stage == "lora":
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
    comparison = build_ab_comparison(config, split_manifest)
    print(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
