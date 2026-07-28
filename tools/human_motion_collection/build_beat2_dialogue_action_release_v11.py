#!/usr/bin/env python3
"""Promote audited BEAT2 v10 rows to complete 18D dialogue/action semantics."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from upper_body_skeleton.ula_v2_dialogue_action_episode import (
    ACTION_ROLE,
    ACTION_SUPERVISION_SCOPE,
    ARTIFACT_KIND,
    DIALOGUE_ROLE,
    FORMAL_ELIGIBILITY_MODE,
    FORMAL_EPISODE_CONTRACT,
    TRAINING_SEGMENT_REPRESENTATION,
    action_directive_from_motion_style,
    canonical_sha256,
    sha256_file,
    text_sha256,
    validate_dialogue_action_v11_episode,
)


DEFAULT_V10_DIR = Path(
    "/home/gez/nas/cloud/gez/human_motion/processed/beat2_dialogue_directive_release_v10"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/gez/nas/cloud/gez/human_motion/processed/beat2_dialogue_action_release_v11"
)
SUMMARY_KIND = "beat2_dialogue_action_release_v11_summary"
LOCK_KIND = "ula_v2_dialogue_action_v11_provenance_lock_v1"
COUNTERFACTUAL_KIND = "ula_v2_dialogue_action_v11_counterfactual_pair_v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    return rows


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
    os.replace(temporary, path)


def build_release(v10_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    v10_dir = Path(v10_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    v10_manifest = v10_dir / "train_ready.jsonl"
    v10_pairs_path = v10_dir / "counterfactual_pairs.jsonl"
    v10_audit_path = v10_dir / "integrity_audit.json"
    audit = _read_json(v10_audit_path)
    if (
        audit.get("passed") is not True
        or audit.get("train_ready_manifest_sha256") != sha256_file(v10_manifest)
        or audit.get("counterfactual_pair_manifest_sha256") != sha256_file(v10_pairs_path)
    ):
        raise ValueError("v10 release has not passed its bound integrity audit")
    v10_rows = _read_jsonl(v10_manifest)
    v10_pairs = _read_jsonl(v10_pairs_path)
    old_pair_by_id = {row["anchor_clip_id"]: row for row in v10_pairs}
    if len(old_pair_by_id) != len(v10_rows):
        raise ValueError("v10 counterfactual membership changed")

    rows = []
    for source in v10_rows:
        clip_id = str(source["clip_id"])
        directive = action_directive_from_motion_style(source)
        row = deepcopy(source)
        upstream_admission = row.pop("training_admission")
        row.pop("directive_text", None)
        row.pop("directive_text_sha256", None)
        row.pop("directive_conditioning_mask", None)
        row.pop("directive_contract", None)
        row.pop("dialogue_motion_alignment", None)
        row.pop("self_speech_gesture_supervision_mask", None)
        row.update(
            {
                "schema_version": "1.0.0",
                "artifact_kind": ARTIFACT_KIND,
                "formal_episode_contract": FORMAL_EPISODE_CONTRACT,
                "eligibility_mode": FORMAL_ELIGIBILITY_MODE,
                "prompt": directive,
                "prompt_sha256": text_sha256(directive),
                "action_directive_text": directive,
                "action_directive_text_sha256": text_sha256(directive),
                "action_directive_conditioning_mask": True,
                "action_directive_contract": {
                    "role": ACTION_ROLE,
                    "primary_control": True,
                    "derived_from_dialogue_text": False,
                    "supervision_scope": ACTION_SUPERVISION_SCOPE,
                    "source": "verified_18d_trajectory_style_action_directive_v1",
                },
                "action_supervision_scope": ACTION_SUPERVISION_SCOPE,
                "complete_18d_action_supervision_mask": True,
                "self_speech_action_context_mask": True,
                "upstream_v10_record_sha256": canonical_sha256(source),
                "upstream_v10_training_admission": upstream_admission,
                "training_segment": {
                    **deepcopy(source["training_segment"]),
                    "representation": TRAINING_SEGMENT_REPRESENTATION,
                    "fixed_window_sec": None,
                    "cropped": False,
                },
                "dialogue_contract": {
                    **deepcopy(source["dialogue_contract"]),
                    "role": DIALOGUE_ROLE,
                    "auxiliary_context": True,
                    "action_label_supervision": False,
                    "partner_response_supervision": False,
                },
            }
        )
        rows.append(row)

    by_id = {row["clip_id"]: row for row in rows}
    pairs = []
    for row in rows:
        old_pair = old_pair_by_id[row["clip_id"]]
        dialogue_source = by_id[old_pair["dialogue_shuffled"]["source_clip_id"]]
        directive_source = by_id[old_pair["directive_shuffled"]["source_clip_id"]]
        if directive_source["action_directive_text_sha256"] == row[
            "action_directive_text_sha256"
        ]:
            candidates = [
                candidate
                for candidate in rows
                if candidate["fixed_split_assignment"] == row["fixed_split_assignment"]
                and candidate["source_group_key"] != row["source_group_key"]
                and candidate["action_directive_text_sha256"]
                != row["action_directive_text_sha256"]
            ]
            if not candidates:
                raise ValueError(f"{row['clip_id']}: no true action-directive negative")
            directive_source = min(candidates, key=lambda item: item["clip_id"])
        pair = {
            "schema_version": "1.0.0",
            "artifact_kind": COUNTERFACTUAL_KIND,
            "anchor_clip_id": row["clip_id"],
            "fixed_split_assignment": row["fixed_split_assignment"],
            "matched": {
                "action_directive_text_sha256": row["action_directive_text_sha256"],
                "dialogue_text_sha256": row["dialogue_text_sha256"],
                "trajectory_sha256": row["trajectory_sha256"],
            },
            "dialogue_shuffled": {
                "source_clip_id": dialogue_source["clip_id"],
                "dialogue_text_sha256": dialogue_source["dialogue_text_sha256"],
                "same_split": True,
                "different_source_group": True,
            },
            "action_directive_shuffled": {
                "source_clip_id": directive_source["clip_id"],
                "action_directive_text_sha256": directive_source[
                    "action_directive_text_sha256"
                ],
                "same_split": True,
                "different_source_group": True,
            },
            "evaluation_policy": (
                "compare_matched_vs_dialogue_shuffled_vs_action_directive_shuffled_"
                "with_identical_complete_18d_action_target"
            ),
            "accepted_as_positive_action_target": False,
        }
        pair["record_sha256"] = canonical_sha256(pair)
        row["dialogue_action_alignment"] = {
            "positive_pair": True,
            "evidence": "exact_dialogue_overlap_with_complete_native_18d_action_interval",
            "supervision": "weak_temporal_dialogue_action_alignment",
            "specific_action_label": False,
            "hard_negative_required": True,
            "hard_negative_record_sha256": pair["record_sha256"],
        }
        row["training_admission"] = {
            "contract": FORMAL_EPISODE_CONTRACT,
            "trajectory_sha256": row["trajectory_sha256"],
            "action_directive_text_sha256": row["action_directive_text_sha256"],
            "dialogue_text_sha256": row["dialogue_text_sha256"],
            "hard_negative_record_sha256": pair["record_sha256"],
            "training_channel_masks": {
                "complete_motion_18d": True,
                "action_directive_text": True,
                "dialogue_text": True,
                "dialogue_action_alignment": True,
                "trajectory_style": True,
                "primary_intent": False,
                "emotion": False,
                "partner_response": False,
                "audio": False,
            },
        }
        validate_dialogue_action_v11_episode(row)
        pairs.append(pair)

    manifest_path = output_dir / "train_ready.jsonl"
    pair_path = output_dir / "counterfactual_pairs.jsonl"
    _atomic_jsonl(manifest_path, rows)
    _atomic_jsonl(pair_path, pairs)
    scale = {
        "episode_count": len(rows),
        "counterfactual_pair_count": len(pairs),
        "frame_count": sum(int(row["frames"]) for row in rows),
        "sample_span_sec": sum((int(row["frames"]) - 1) / 30.0 for row in rows),
        "minimum_frames": min(int(row["frames"]) for row in rows),
        "maximum_frames": max(int(row["frames"]) for row in rows),
        "distinct_action_directive_count": len(
            {row["action_directive_text"] for row in rows}
        ),
        "distinct_dialogue_count": len({row["dialogue_text"] for row in rows}),
        "split_counts": dict(
            sorted(Counter(row["fixed_split_assignment"] for row in rows).items())
        ),
    }
    lock = {
        "schema_version": 1,
        "artifact_kind": LOCK_KIND,
        "accepted_for_training": True,
        "formal_episode_contract": FORMAL_EPISODE_CONTRACT,
        "supervision_scope": ACTION_SUPERVISION_SCOPE,
        "active_prompt_uses_gesture_narrowing": False,
        "duration_policy": "native_variable_length_no_fixed_or_target_duration_v1",
        "source_v10_integrity_audit": str(v10_audit_path),
        "source_v10_integrity_audit_sha256": sha256_file(v10_audit_path),
        "source_v10_manifest": str(v10_manifest),
        "source_v10_manifest_sha256": sha256_file(v10_manifest),
        "train_ready_manifest": str(manifest_path),
        "train_ready_manifest_sha256": sha256_file(manifest_path),
        "counterfactual_pair_manifest": str(pair_path),
        "counterfactual_pair_manifest_sha256": sha256_file(pair_path),
        "dataset_scale": scale,
        "scenario_support": {
            "complete_observed_18d_upper_body_action": "directly_supervised",
            "brain_action_directive_plus_dialogue": "directly_conditioned",
            "user_speaks_robot_listener_response": "not_directly_supervised",
        },
    }
    _atomic_json(output_dir / "provenance_lock.json", lock)
    summary = {
        "schema_version": "1.0.0",
        "artifact_kind": SUMMARY_KIND,
        "passed": True,
        "network_modified_or_trained": False,
        "formal_episode_contract": FORMAL_EPISODE_CONTRACT,
        "train_ready_manifest": str(manifest_path),
        "train_ready_manifest_sha256": lock["train_ready_manifest_sha256"],
        "counterfactual_pair_manifest": str(pair_path),
        "counterfactual_pair_manifest_sha256": lock[
            "counterfactual_pair_manifest_sha256"
        ],
        "dataset_scale": scale,
        "supervision_scope": ACTION_SUPERVISION_SCOPE,
        "next_gate": "dual_role_network_and_counterfactual_gradient_smoke",
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v10-dir", type=Path, default=DEFAULT_V10_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    print(json.dumps(build_release(args.v10_dir, args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
