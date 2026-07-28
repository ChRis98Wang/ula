#!/usr/bin/env python3
"""Build concise, Kimodo-like BEAT2 action summaries without intent invention."""

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

from upper_body_skeleton.beat2_observable_action_summary import (  # noqa: E402
    DEFAULT_ONTOLOGY_PATH,
    SUMMARY_SOURCE,
    ontology_sha256,
    summarize_observable_action,
    validate_action_summary,
)
from upper_body_skeleton.ula_v2_dialogue_action_episode import (  # noqa: E402
    canonical_sha256,
    sha256_file,
    text_sha256,
    validate_dialogue_action_v11_episode,
)


DEFAULT_V11_DIR = Path(
    "/home/gez/nas/cloud/gez/human_motion/processed/beat2_dialogue_action_release_v11"
)
DEFAULT_STYLE_CATALOG = (
    PROJECT_ROOT
    / "deliverables/expressive_human_motion_v2/robot_observable_intents_v1/"
    "beat2_full12148_intent_candidates_v9_tiera/style_catalog.jsonl"
)
DEFAULT_OFFICIAL_SOURCE = Path(
    "/home/gez/nas/cloud/gez/human_motion/processed/"
    "beat2_semantic_event_training_pool_18d_v8/expansion/release/"
    "adjudication_min30f/train_ready.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/gez/nas/cloud/gez/human_motion/processed/"
    "beat2_dialogue_named_action_release_v12_expanded29"
)
SUMMARY_KIND = "beat2_dialogue_named_action_release_v12_summary"
PAIR_KIND = "ula_v2_dialogue_named_action_v12_counterfactual_pair_v1"
LOCK_KIND = "ula_v2_dialogue_named_action_v12_provenance_lock_v1"


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


def _negative_source(
    anchor: Mapping[str, Any],
    rows: list[dict[str, Any]],
    *,
    old_source_id: str,
) -> dict[str, Any]:
    by_id = {row["clip_id"]: row for row in rows}
    old_source = by_id.get(old_source_id)
    action_id = anchor["action_summary"]["action_id"]
    if (
        old_source is not None
        and old_source["fixed_split_assignment"] == anchor["fixed_split_assignment"]
        and old_source["source_group_key"] != anchor["source_group_key"]
        and old_source["action_summary"]["action_id"] != action_id
    ):
        return old_source
    candidates = [
        row
        for row in rows
        if row["fixed_split_assignment"] == anchor["fixed_split_assignment"]
        and row["source_group_key"] != anchor["source_group_key"]
        and row["action_summary"]["action_id"] != action_id
    ]
    if not candidates:
        raise ValueError(f"{anchor['clip_id']}: no different named-action negative")
    return min(candidates, key=lambda row: row["clip_id"])


def action_summary_coverage(rows: list[Mapping[str, Any]], label_ids: list[str]) -> dict[str, Any]:
    """Report label coverage without equating a valid label with enough training data."""

    split_counts: dict[str, Counter[str]] = {
        label_id: Counter() for label_id in label_ids
    }
    for row in rows:
        action_id = str(row["action_summary"]["action_id"])
        if action_id not in split_counts:
            raise ValueError(f"release contains an unknown action summary: {action_id}")
        split_counts[action_id][str(row["fixed_split_assignment"])] += 1

    support = {}
    for label_id in label_ids:
        counts = split_counts[label_id]
        train_count = int(counts["train"])
        if train_count == 0:
            status = "no_train_examples"
        elif train_count < 20:
            status = "too_few_for_standalone_claim"
        elif train_count < 50:
            status = "limited_train_support"
        else:
            status = "train_supported"
        support[label_id] = {
            "train": train_count,
            "validation": int(counts["validation"]),
            "test": int(counts["test"]),
            "total": int(sum(counts.values())),
            "support_status": status,
        }
    return {
        "defined_action_summary_count": len(label_ids),
        "observed_action_summary_count": sum(
            value["total"] > 0 for value in support.values()
        ),
        "train_observed_action_summary_count": sum(
            value["train"] > 0 for value in support.values()
        ),
        "fully_train_supported_action_summary_count": sum(
            value["support_status"] == "train_supported"
            for value in support.values()
        ),
        "unobserved_action_ids": sorted(
            label_id for label_id, value in support.items() if value["total"] == 0
        ),
        "no_train_example_action_ids": sorted(
            label_id for label_id, value in support.items() if value["train"] == 0
        ),
        "under_20_train_example_action_ids": sorted(
            label_id
            for label_id, value in support.items()
            if 0 < value["train"] < 20
        ),
        "support_threshold_policy": {
            "no_train_examples": 0,
            "too_few_for_standalone_claim": "1-19",
            "limited_train_support": "20-49",
            "train_supported": ">=50",
            "thresholds_are_coverage_diagnostics_not_quality_guarantees": True,
        },
        "by_action_id": support,
        "release_integrity_passed": True,
        "full_ontology_train_coverage": all(
            value["train"] > 0 for value in support.values()
        ),
    }


def build_release(
    v11_dir: str | Path,
    style_catalog: str | Path,
    official_source: str | Path,
    output_dir: str | Path,
    *,
    ontology_path: str | Path = DEFAULT_ONTOLOGY_PATH,
) -> dict[str, Any]:
    v11_dir = Path(v11_dir).resolve()
    style_catalog = Path(style_catalog).resolve()
    official_source = Path(official_source).resolve()
    output_dir = Path(output_dir).resolve()
    ontology_path = Path(ontology_path).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    v11_manifest = v11_dir / "train_ready.jsonl"
    v11_pairs_path = v11_dir / "counterfactual_pairs.jsonl"
    v11_summary = _read_json(v11_dir / "summary.json")
    if (
        v11_summary.get("passed") is not True
        or v11_summary.get("train_ready_manifest_sha256") != sha256_file(v11_manifest)
        or v11_summary.get("counterfactual_pair_manifest_sha256")
        != sha256_file(v11_pairs_path)
    ):
        raise ValueError("V11 release does not match its passed summary")
    source_rows = _read_jsonl(v11_manifest)
    old_pairs = {
        row["anchor_clip_id"]: row for row in _read_jsonl(v11_pairs_path)
    }
    style_rows = _read_jsonl(style_catalog)
    styles = {row["task_id"]: row for row in style_rows}
    if len(styles) != len(style_rows):
        raise ValueError("style catalog contains duplicate task IDs")
    official_rows = _read_jsonl(official_source)
    official_by_id = {row["task_id"]: row for row in official_rows}
    if len(official_by_id) != len(official_rows):
        raise ValueError("official source contains duplicate task IDs")
    missing = sorted({row["clip_id"] for row in source_rows} - set(styles))
    if missing:
        raise ValueError(f"V11 rows are missing style evidence: {missing[:5]}")
    missing_official = sorted(
        {row["clip_id"] for row in source_rows} - set(official_by_id)
    )
    if missing_official:
        raise ValueError(
            f"V11 rows are missing official category evidence: {missing_official[:5]}"
        )

    rows = []
    ontology = json.loads(ontology_path.read_text(encoding="utf-8"))
    label_ids = [str(label["id"]) for label in ontology["labels"]]
    for source in source_rows:
        validate_dialogue_action_v11_episode(source)
        official = official_by_id[source["clip_id"]]
        semantic_event = official.get("semantic_event") or {}
        if official.get("official_category_verified") is not True:
            raise ValueError(f"{source['clip_id']}: official category is not verified")
        summary = summarize_observable_action(
            styles[source["clip_id"]],
            official_category=str(semantic_event.get("category") or ""),
            ontology_path=ontology_path,
        )
        validate_action_summary(summary, ontology_path=ontology_path)
        directive = summary["prompt_en"]
        row = deepcopy(source)
        row.update(
            {
                "schema_version": "1.1.0",
                "release_revision": "beat2_dialogue_named_action_v12",
                "base_v11_record_sha256": canonical_sha256(source),
                "action_summary": summary,
                "prompt": directive,
                "prompt_sha256": text_sha256(directive),
                "action_directive_text": directive,
                "action_directive_text_sha256": text_sha256(directive),
                "action_directive_contract": {
                    **source["action_directive_contract"],
                    "source": SUMMARY_SOURCE,
                },
            }
        )
        rows.append(row)

    by_id = {row["clip_id"]: row for row in rows}
    pairs = []
    for row in rows:
        old_pair = old_pairs[row["clip_id"]]
        dialogue_source = by_id[old_pair["dialogue_shuffled"]["source_clip_id"]]
        action_source = _negative_source(
            row,
            rows,
            old_source_id=old_pair["action_directive_shuffled"]["source_clip_id"],
        )
        pair = {
            "schema_version": "1.0.0",
            "artifact_kind": PAIR_KIND,
            "anchor_clip_id": row["clip_id"],
            "fixed_split_assignment": row["fixed_split_assignment"],
            "matched": {
                "action_id": row["action_summary"]["action_id"],
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
                "source_clip_id": action_source["clip_id"],
                "action_id": action_source["action_summary"]["action_id"],
                "action_directive_text_sha256": action_source[
                    "action_directive_text_sha256"
                ],
                "same_split": True,
                "different_source_group": True,
                "different_action_id": True,
            },
            "evaluation_policy": (
                "same_trajectory_compare_matched_vs_dialogue_shuffled_vs_"
                "different_named_action_shuffled"
            ),
            "accepted_as_positive_action_target": False,
        }
        pair["record_sha256"] = canonical_sha256(pair)
        row["dialogue_action_alignment"] = {
            **row["dialogue_action_alignment"],
            "supervision": "weak_dialogue_alignment_plus_trajectory_action_summary_v12",
            "specific_action_label": True,
            "hard_negative_record_sha256": pair["record_sha256"],
        }
        row["training_admission"] = {
            **row["training_admission"],
            "action_directive_text_sha256": row["action_directive_text_sha256"],
            "hard_negative_record_sha256": pair["record_sha256"],
            "named_action_summary": {
                "action_id": row["action_summary"]["action_id"],
                "ontology_sha256": row["action_summary"]["ontology_sha256"],
                "supervision_mask": True,
                "pragmatic_intent_supervision_mask": False,
            },
        }
        validate_dialogue_action_v11_episode(row)
        pairs.append(pair)

    manifest_path = output_dir / "train_ready.jsonl"
    pair_path = output_dir / "counterfactual_pairs.jsonl"
    _atomic_jsonl(manifest_path, rows)
    _atomic_jsonl(pair_path, pairs)
    counts = Counter(row["action_summary"]["action_id"] for row in rows)
    confidence_counts = Counter(row["action_summary"]["confidence"] for row in rows)
    coverage = action_summary_coverage(rows, label_ids)
    scale = {
        "episode_count": len(rows),
        "frame_count": sum(int(row["frames"]) for row in rows),
        "sample_span_sec": sum((int(row["frames"]) - 1) / 30.0 for row in rows),
        "distinct_action_summary_count": len(counts),
        "distinct_dialogue_count": len({row["dialogue_text"] for row in rows}),
        "split_counts": dict(
            sorted(Counter(row["fixed_split_assignment"] for row in rows).items())
        ),
        "action_summary_counts": dict(sorted(counts.items())),
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "action_summary_coverage": coverage,
    }
    summary = {
        "schema_version": "1.0.0",
        "artifact_kind": SUMMARY_KIND,
        "passed": True,
        "base_v11_manifest": str(v11_manifest),
        "base_v11_manifest_sha256": sha256_file(v11_manifest),
        "style_catalog": str(style_catalog),
        "style_catalog_sha256": sha256_file(style_catalog),
        "official_source_manifest": str(official_source),
        "official_source_manifest_sha256": sha256_file(official_source),
        "action_summary_ontology": str(ontology_path),
        "action_summary_ontology_sha256": ontology_sha256(ontology_path),
        "train_ready_manifest": str(manifest_path),
        "train_ready_manifest_sha256": sha256_file(manifest_path),
        "counterfactual_pair_manifest": str(pair_path),
        "counterfactual_pair_manifest_sha256": sha256_file(pair_path),
        "dataset_scale": scale,
        "label_policy": {
            "kimodo_data_used": False,
            "dialogue_used_to_assign_action": False,
            "emotion_used_to_assign_action": False,
            "pragmatic_intent_claimed": False,
            "physical_action_summary_supervision": True,
        },
        "coverage_interpretation": (
            "passed means the release is contract-valid; sparse labels are not claimed "
            "as independently learnable unless their training support is adequate"
        ),
    }
    _atomic_json(output_dir / "summary.json", summary)
    lock = {
        "schema_version": "1.0.0",
        "artifact_kind": LOCK_KIND,
        "summary_sha256": sha256_file(output_dir / "summary.json"),
        "train_ready_manifest_sha256": sha256_file(manifest_path),
        "counterfactual_pair_manifest_sha256": sha256_file(pair_path),
        "action_summary_ontology_sha256": ontology_sha256(ontology_path),
    }
    lock["sha256"] = canonical_sha256(lock)
    _atomic_json(output_dir / "provenance_lock.json", lock)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v11-dir", type=Path, default=DEFAULT_V11_DIR)
    parser.add_argument("--style-catalog", type=Path, default=DEFAULT_STYLE_CATALOG)
    parser.add_argument("--official-source", type=Path, default=DEFAULT_OFFICIAL_SOURCE)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    result = build_release(
        args.v11_dir,
        args.style_catalog,
        args.official_source,
        args.output_dir,
        ontology_path=args.ontology,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
