#!/usr/bin/env python3
"""Create an untrained, full-random 18D ULA MMDiT V2 checkpoint.

The command performs no optimization.  It refuses unsafe/unreviewed or fixed
window manifests and never accepts a pretrained generator checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Mapping

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from upper_body_skeleton.ula_v2_18d_head import load_18d_episodes, sha256_file
from upper_body_skeleton.ula_v2_18d_random_init import (
    DEFAULT_LENGTH_BUCKETS,
    DEFAULT_SPLIT_FRACTIONS,
    build_random_18d_checkpoint,
)
from upper_body_skeleton.ula_v2_expression_turn_episode import (
    is_expression_turn_v8_episode,
    load_expression_turn_v8_episodes,
    validate_expression_turn_v8_episode,
)


CONFIG_SCHEMA_VERSION = 1
FORBIDDEN_CONFIG_KEYS = {
    "base_checkpoint",
    "base_15d_checkpoint",
    "initial_checkpoint",
    "generator_checkpoint",
    "resume_from",
}


def _atomic_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"manifest row must be an object: {path}:{line_number}")
            records.append(value)
    return records


def _nested(record: Mapping, *paths: tuple[str, ...]):
    for path in paths:
        value = record
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                value = None
                break
            value = value[key]
        if value is not None and str(value).strip():
            return value
    return None


def read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("random initialization config must be a JSON object")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    forbidden = sorted(FORBIDDEN_CONFIG_KEYS.intersection(config))
    if forbidden:
        raise ValueError(
            "full-random initialization refuses generator checkpoint inputs: "
            f"{forbidden}"
        )
    if config.get("allow_unreviewed"):
        raise ValueError("full-random formal initialization cannot allow unreviewed data")
    if not isinstance(config.get("motion_sources"), list) or not config["motion_sources"]:
        raise ValueError("motion_sources must be a non-empty list")
    for field in ("qwen_checkpoint", "output_dir"):
        if not str(config.get(field) or "").strip():
            raise ValueError(f"{field} is required")
    return config


def load_formal_sources(config: Mapping) -> tuple[list[dict], list[dict]]:
    episodes = []
    provenance = []
    known_clip_ids = set()
    for source in config["motion_sources"]:
        if not isinstance(source, Mapping):
            raise ValueError("every motion source must be an object")
        dataset_source = str(source.get("dataset_source") or "").strip()
        manifest = Path(str(source.get("manifest") or "")).resolve()
        if not dataset_source or not manifest.is_file():
            raise ValueError("every motion source needs dataset_source and an existing manifest")
        speaker_namespace = str(source.get("speaker_namespace") or dataset_source).strip()
        group_namespace = str(source.get("source_group_namespace") or dataset_source).strip()
        if not speaker_namespace or not group_namespace:
            raise ValueError("speaker/source-group namespaces cannot be empty")
        raw_records = _read_jsonl(manifest)
        raw_by_clip = {
            str(record.get("clip_id") or record.get("sample_id") or "").strip(): record
            for record in raw_records
        }
        if "" in raw_by_clip or len(raw_by_clip) != len(raw_records):
            raise ValueError(f"manifest has missing or duplicate clip ids: {manifest}")
        v8_flags = [is_expression_turn_v8_episode(record) for record in raw_records]
        if any(v8_flags) and not all(v8_flags):
            raise ValueError(f"manifest mixes v8 and legacy episode contracts: {manifest}")
        expression_turn_v8 = bool(v8_flags and all(v8_flags))
        loaded = (
            load_expression_turn_v8_episodes(manifest)
            if expression_turn_v8
            else load_18d_episodes(manifest=manifest, allow_unreviewed=False)
        )
        for episode in loaded:
            clip_id = str(episode["clip_id"])
            if clip_id in known_clip_ids:
                raise ValueError(f"clip_id is not globally unique: {clip_id}")
            raw = raw_by_clip[clip_id]
            speaker = _nested(
                raw,
                ("speaker_key",),
                ("speaker_id",),
                ("meta", "speaker_key"),
                ("source", "speaker_key"),
            )
            source_group = _nested(
                raw,
                ("source_group_key",),
                ("source_group_id",),
                ("source_clip_id",),
                ("split", "source_group_key"),
                ("source", "source_group_key"),
                ("source", "source_clip_id"),
            )
            if speaker is None or source_group is None:
                raise ValueError(
                    f"{clip_id}: manifest lacks speaker/source group for strict split"
                )
            item = dict(episode)
            motion_18d = raw.get("motion_18d")
            retarget_qc_passed = bool(
                isinstance(motion_18d, Mapping)
                and motion_18d.get("state") == "passed"
                and isinstance(motion_18d.get("quality_gate"), Mapping)
                and motion_18d["quality_gate"]
                and all(value is True for value in motion_18d["quality_gate"].values())
            )
            item.update(
                {
                    "dataset_source": dataset_source,
                    "speaker_key": f"{speaker_namespace}:{speaker}",
                    "source_group_key": f"{group_namespace}:{source_group}",
                    "retarget_qc_passed": retarget_qc_passed,
                    "action_dim_mask": np.ones(18, dtype=np.bool_),
                }
            )
            if expression_turn_v8:
                validate_expression_turn_v8_episode(item)
            else:
                item.update(
                    {
                        "training_segment": raw.get("training_segment"),
                        "window": raw.get("window"),
                        "selection_status": raw.get("selection_status"),
                        "source_clip_id": raw.get("source_clip_id"),
                        "source_sha256": raw.get("source_sha256"),
                        "semantic_event": raw.get("semantic_event"),
                        "annotation_kind": raw.get("annotation_kind"),
                        "interaction_scope": raw.get("interaction_scope"),
                        "behavior_source": raw.get("behavior_source"),
                        "behavior_mapping_contract": raw.get(
                            "behavior_mapping_contract"
                        ),
                        "independent_review": raw.get("independent_review"),
                        "human_review": raw.get("human_review"),
                        "emotion_source": raw.get("emotion_source"),
                        "emotion_label_source": raw.get("emotion_label_source"),
                        "emotion_protocol_contract": raw.get(
                            "emotion_protocol_contract"
                        ),
                    }
                )
            if source.get("use_manifest_fixed_split") is True:
                assignment = _nested(
                    raw,
                    ("fixed_split_assignment",),
                    ("split", "assignment"),
                )
                item["fixed_split_assignment"] = assignment
            episodes.append(item)
            known_clip_ids.add(clip_id)
        provenance.append(
            {
                "dataset_source": dataset_source,
                "manifest": str(manifest),
                "manifest_sha256": sha256_file(manifest),
                "formal_loader_policy": (
                    "expression_turn_v8_three_tier_adjudicated_train_ready_only"
                    if expression_turn_v8
                    else "adjudicated_train_ready_only"
                ),
                "speaker_namespace": speaker_namespace,
                "source_group_namespace": group_namespace,
                "loaded_clip_count": len(loaded),
            }
        )
    return episodes, provenance


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    output_dir = Path(config["output_dir"]).resolve()
    outputs = {
        "checkpoint": output_dir / "random_init.pt",
        "split": output_dir / "split_manifest.json",
        "report": output_dir / "initialization_report.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing and not bool(config.get("overwrite", False)):
        raise FileExistsError(f"refusing to overwrite random initialization outputs: {existing}")

    episodes, source_provenance = load_formal_sources(config)
    model_config = dict(config.get("model") or {})
    checkpoint, split_contract, report = build_random_18d_checkpoint(
        episodes,
        qwen_checkpoint=Path(config["qwen_checkpoint"]).resolve(),
        source_provenance=source_provenance,
        seed=int(config.get("seed", 7)),
        fractions=dict(config.get("split_fractions") or DEFAULT_SPLIT_FRACTIONS),
        hidden_dim=int(model_config.get("hidden_dim", 384)),
        layers=int(model_config.get("layers", 6)),
        semantic_tokens=int(model_config.get("semantic_tokens", 7)),
        style_clip=float(config.get("style_clip", 5.0)),
        length_buckets=tuple(config.get("length_buckets") or DEFAULT_LENGTH_BUCKETS),
    )
    _atomic_torch_save(checkpoint, outputs["checkpoint"])
    _atomic_json(split_contract, outputs["split"])
    report.update(
        {
            "checkpoint": str(outputs["checkpoint"]),
            "checkpoint_sha256": sha256_file(outputs["checkpoint"]),
            "split_manifest": str(outputs["split"]),
            "sources": source_provenance,
            "training_command_executed": False,
        }
    )
    _atomic_json(report, outputs["report"])
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
