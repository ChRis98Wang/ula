#!/usr/bin/env python3
"""Create an untrained, full-random 18D ULA MMDiT V2 checkpoint.

The command performs no optimization.  It refuses unsafe/unreviewed or fixed
window manifests and never accepts a pretrained generator checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
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

from upper_body_skeleton.ula_v2_18d_head import (
    MOTION_ONLY_EPISODE_CONTRACT,
    load_18d_episodes,
    sha256_file,
)
from upper_body_skeleton.data_source_registry import (
    EXPRESSION_GENERATOR_ROLE,
    GENERATOR_FOUNDATION_ROLE,
    SEMANTIC_GENERATOR_ROLE,
    assert_no_forbidden_data_lineage,
    assert_no_forbidden_source_reference,
    registered_source,
)
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
from upper_body_skeleton.ula_v2_conversational_realization_episode import (
    FORMAL_EPISODE_CONTRACT as CONVERSATIONAL_REALIZATION_V9_EPISODE_CONTRACT,
    is_conversational_realization_v9_episode,
    load_conversational_realization_v9_episodes,
    validate_conversational_realization_v9_episode,
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
    if not str(config.get("output_dir") or "").strip():
        raise ValueError("output_dir is required")
    motion_only = (
        config.get("formal_episode_contract") == MOTION_ONLY_EPISODE_CONTRACT
    )
    expression_conditioned = config.get("formal_episode_contract") in {
        CONVERSATIONAL_REALIZATION_V9_EPISODE_CONTRACT,
        "beat2_expression_turn_v8_train_episode_v1",
    }
    source_role = (
        GENERATOR_FOUNDATION_ROLE
        if motion_only
        else EXPRESSION_GENERATOR_ROLE
        if expression_conditioned
        else SEMANTIC_GENERATOR_ROLE
    )
    assert_no_forbidden_data_lineage(config, context="random_init_config")
    for index, source in enumerate(config["motion_sources"]):
        if not isinstance(source, Mapping):
            raise ValueError(f"motion_sources[{index}] must be an object")
        registered_source(source.get("dataset_source"), role=source_role)
    if motion_only:
        for field in ("qwen_checkpoint", "qwen_checkpoint_sha256"):
            if field in config:
                raise ValueError(
                    f"motion-only BEAT2 initialization forbids {field}"
                )
        for index, source in enumerate(config["motion_sources"]):
            if not isinstance(source, Mapping) or source.get(
                "use_manifest_fixed_split"
            ) is not True:
                raise ValueError(
                    f"motion_sources[{index}] must enable use_manifest_fixed_split"
                )
            dataset_source = str(source.get("dataset_source") or "").lower()
            if not dataset_source.startswith(("beat2_", "beat2-")):
                raise ValueError(
                    f"motion_sources[{index}] is outside the BEAT2 source whitelist"
                )
    elif not str(config.get("qwen_checkpoint") or "").strip():
        raise ValueError("qwen_checkpoint is required")
    return config


def load_formal_sources(config: Mapping) -> tuple[list[dict], list[dict]]:
    episodes = []
    provenance = []
    known_clip_ids = set()
    motion_only = (
        config.get("formal_episode_contract") == MOTION_ONLY_EPISODE_CONTRACT
    )
    expression_conditioned = config.get("formal_episode_contract") in {
        CONVERSATIONAL_REALIZATION_V9_EPISODE_CONTRACT,
        "beat2_expression_turn_v8_train_episode_v1",
    }
    source_role = (
        GENERATOR_FOUNDATION_ROLE
        if motion_only
        else EXPRESSION_GENERATOR_ROLE
        if expression_conditioned
        else SEMANTIC_GENERATOR_ROLE
    )
    for source in config["motion_sources"]:
        if not isinstance(source, Mapping):
            raise ValueError("every motion source must be an object")
        dataset_source = str(source.get("dataset_source") or "").strip()
        manifest = Path(str(source.get("manifest") or "")).resolve()
        registration = registered_source(dataset_source, role=source_role)
        assert_no_forbidden_data_lineage(
            source, context=f"motion_source[{dataset_source or '<empty>'}]"
        )
        assert_no_forbidden_source_reference(
            manifest, context=f"{dataset_source}.manifest"
        )
        if not dataset_source or not manifest.is_file():
            raise ValueError("every motion source needs dataset_source and an existing manifest")
        if motion_only and (
            not dataset_source.lower().startswith(("beat2_", "beat2-"))
            or source.get("use_manifest_fixed_split") is not True
        ):
            raise ValueError(
                "motion-only sources must be BEAT2 and enable the manifest fixed split"
            )
        speaker_namespace = str(source.get("speaker_namespace") or dataset_source).strip()
        group_namespace = str(source.get("source_group_namespace") or dataset_source).strip()
        if not speaker_namespace or not group_namespace:
            raise ValueError("speaker/source-group namespaces cannot be empty")
        raw_records = _read_jsonl(manifest)
        for row_index, record in enumerate(raw_records):
            assert_no_forbidden_data_lineage(
                record,
                context=f"{dataset_source}.manifest[{row_index}]",
            )
        raw_by_clip = {
            str(record.get("clip_id") or record.get("sample_id") or "").strip(): record
            for record in raw_records
        }
        if "" in raw_by_clip or len(raw_by_clip) != len(raw_records):
            raise ValueError(f"manifest has missing or duplicate clip ids: {manifest}")
        v8_flags = [is_expression_turn_v8_episode(record) for record in raw_records]
        conversational_flags = [
            is_conversational_realization_v9_episode(record) for record in raw_records
        ]
        marked_contracts = sum((any(v8_flags), any(conversational_flags)))
        if marked_contracts > 1 or (
            any(v8_flags) and not all(v8_flags)
        ) or (any(conversational_flags) and not all(conversational_flags)):
            raise ValueError(f"manifest mixes incompatible episode contracts: {manifest}")
        expression_turn_v8 = bool(v8_flags and all(v8_flags))
        conversational_realization_v9 = bool(
            conversational_flags and all(conversational_flags)
        )
        loaded = (
            load_expression_turn_v8_episodes(manifest)
            if expression_turn_v8
            else load_conversational_realization_v9_episodes(manifest)
            if conversational_realization_v9
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
            elif conversational_realization_v9:
                validate_conversational_realization_v9_episode(item)
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
                if assignment not in {"train", "validation", "test"}:
                    raise ValueError(
                        f"{clip_id}: manifest fixed split assignment is missing or invalid"
                    )
                item["fixed_split_assignment"] = assignment
            episodes.append(item)
            known_clip_ids.add(clip_id)
        fixed_assignments = [
            {
                "clip_id": str(item["clip_id"]),
                "split": str(item.get("fixed_split_assignment")),
            }
            for item in episodes[-len(loaded) :]
        ]
        fixed_assignment_sha256 = hashlib.sha256(
            json.dumps(
                sorted(fixed_assignments, key=lambda item: item["clip_id"]),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()
        provenance.append(
            {
                "dataset_source": dataset_source,
                "data_source_registration": registration,
                "manifest": str(manifest),
                "manifest_sha256": sha256_file(manifest),
                "manifest_fixed_split": source.get("use_manifest_fixed_split") is True,
                "fixed_split_assignment_sha256": fixed_assignment_sha256,
                "fixed_split_counts": {
                    split_name: sum(
                        item["split"] == split_name for item in fixed_assignments
                    )
                    for split_name in ("train", "validation", "test")
                },
                **(
                    {"license_gate": dict(source["license_gate"])}
                    if isinstance(source.get("license_gate"), Mapping)
                    else {}
                ),
                "formal_loader_policy": (
                    "expression_turn_v8_three_tier_adjudicated_train_ready_only"
                    if expression_turn_v8
                    else "conversational_realization_v9_hash_bound_train_ready_only"
                    if conversational_realization_v9
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
        qwen_checkpoint=(
            None
            if config.get("formal_episode_contract") == MOTION_ONLY_EPISODE_CONTRACT
            else Path(config["qwen_checkpoint"]).resolve()
        ),
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
