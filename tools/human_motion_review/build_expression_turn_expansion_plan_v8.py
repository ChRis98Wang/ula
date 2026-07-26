#!/usr/bin/env python3
"""Build one-level-at-a-time natural-context expansion requests from blind review."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any


COMPLETE_STATUS = "complete_arc_action_candidate"
EXPANDABLE_STATUSES = {
    "natural_context_expansion_required",
    "action_unusable_or_conflicting",
}
FORBIDDEN_OUTPUT_KEYS = {
    "canonical_action",
    "canonical_prompt",
    "emotion_id",
    "official_categories",
    "official_emotion",
    "source_text",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-records", type=Path, required=True)
    parser.add_argument("--hidden-mapping", type=Path, required=True)
    parser.add_argument("--candidate-catalog", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def value_sha256(value: object) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def index_unique(
    rows: list[dict[str, Any]], key: str, *, context: str
) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{context} contains an invalid {key}")
        if value in result:
            raise ValueError(f"{context} contains duplicate {key}: {value}")
        result[value] = row
    return result


def atomic_json(path: Path, value: object) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(stable_json(row) + "\n" for row in rows).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _assert_no_hidden_labels(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_OUTPUT_KEYS or key.startswith("official_"):
                raise ValueError(f"Expansion output leaks hidden label at {path}.{key}")
            _assert_no_hidden_labels(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_hidden_labels(child, f"{path}[{index}]")


def _verified_comparison(row: dict[str, Any]) -> dict[str, Any]:
    expected = row.get("comparison_record_sha256")
    value = dict(row)
    value.pop("comparison_record_sha256", None)
    if expected != value_sha256(value):
        raise ValueError(f"Comparison record SHA mismatch: {row.get('sample_id')}")
    if row.get("accepted_for_training") is not False:
        raise ValueError("Review comparison unexpectedly admits training")
    if row.get("elapsed_duration_used_as_gate") is not False:
        raise ValueError("Review comparison used duration as a gate")
    status = row.get("qualification_status")
    if status not in {COMPLETE_STATUS, *EXPANDABLE_STATUSES}:
        raise ValueError(f"Unknown review qualification status: {status}")
    return row


def _level_map(candidate: dict[str, Any]) -> dict[int, dict[str, Any]]:
    context = candidate.get("context_plan") or {}
    levels = context.get("levels") or []
    result = {}
    for level in levels:
        index = level.get("level")
        if not isinstance(index, int) or index in result:
            raise ValueError(f"Invalid context level in {candidate.get('task_id')}")
        start = level.get("start_frame")
        end = level.get("end_frame_exclusive")
        if not isinstance(start, int) or not isinstance(end, int) or end <= start:
            raise ValueError(f"Invalid context interval in {candidate.get('task_id')}")
        result[index] = level
    return result


def build_plan(
    *,
    comparison_records: Path,
    hidden_mapping: Path,
    candidate_catalog: Path,
    output_root: Path,
) -> dict[str, Any]:
    comparison_records = comparison_records.resolve()
    hidden_mapping = hidden_mapping.resolve()
    candidate_catalog = candidate_catalog.resolve()
    comparisons = index_unique(
        [_verified_comparison(row) for row in read_jsonl(comparison_records)],
        "sample_id",
        context="review comparisons",
    )
    hidden = index_unique(
        read_jsonl(hidden_mapping), "sample_id", context="hidden mapping"
    )
    if set(comparisons) != set(hidden):
        raise ValueError("Review comparison and hidden mapping sample sets differ")
    catalog = index_unique(
        read_jsonl(candidate_catalog), "task_id", context="candidate catalog"
    )

    expansion_requests = []
    complete_current_context = []
    context_exhausted = []
    for sample_id in sorted(comparisons):
        comparison = comparisons[sample_id]
        mapping = hidden[sample_id]
        task_id = mapping.get("task_id")
        candidate = catalog.get(task_id)
        if candidate is None:
            raise ValueError(f"Hidden task is absent from candidate catalog: {task_id}")
        if mapping.get("video_sha256") != comparison.get("video_sha256"):
            raise ValueError(f"Video SHA mismatch for {sample_id}")
        if mapping.get("expression_turn_record_sha256") != candidate.get(
            "expression_turn_record_sha256"
        ):
            raise ValueError(f"Expression-turn record binding mismatch for {sample_id}")
        context = candidate.get("context_plan") or {}
        current_level = context.get("selected_level")
        if current_level != comparison.get("context_level") or not isinstance(
            current_level, int
        ):
            raise ValueError(f"Reviewed context level mismatch for {sample_id}")
        levels = _level_map(candidate)
        current = levels.get(current_level)
        if current is None:
            raise ValueError(f"Reviewed context is absent from plan for {sample_id}")
        status = comparison["qualification_status"]
        common = {
            "schema_version": "1.0.0",
            "sample_id": sample_id,
            "base_task_id": task_id,
            "source_clip_id": mapping["source_clip_id"],
            "fixed_split_assignment": mapping["fixed_split_assignment"],
            "base_expression_turn_record_sha256": mapping[
                "expression_turn_record_sha256"
            ],
            "comparison_record_sha256": comparison["comparison_record_sha256"],
            "review_qualification_status": status,
            "reviewed_context_level": current_level,
            "reviewed_interval": {
                "start_frame": current["start_frame"],
                "end_frame_exclusive": current["end_frame_exclusive"],
                "frame_count": current["end_frame_exclusive"]
                - current["start_frame"],
            },
            "elapsed_duration_used_as_gate": False,
            "semantic_supervision_mask": False,
            "emotion_supervision_mask": False,
            "accepted_for_training": False,
        }
        if status == COMPLETE_STATUS:
            record = {
                **common,
                "artifact_kind": "expression_turn_v8_current_context_arc_action_complete",
                "next_action": "hold_for_independent_affect_and_remaining_admission_gates",
            }
            record["plan_record_sha256"] = value_sha256(record)
            complete_current_context.append(record)
            continue

        next_level = current_level + 1
        requested = levels.get(next_level)
        if requested is None:
            record = {
                **common,
                "artifact_kind": "expression_turn_v8_natural_context_exhausted",
                "next_action": "reject_or_manual_source_level_adjudication_no_further_predeclared_level",
                "context_exhausted_at_level": context.get("context_exhausted_at_level"),
            }
            record["plan_record_sha256"] = value_sha256(record)
            context_exhausted.append(record)
            continue
        record = {
            **common,
            "artifact_kind": "expression_turn_v8_one_level_natural_context_expansion_request",
            "requested_context_level": next_level,
            "requested_interval": {
                "start_frame": requested["start_frame"],
                "end_frame_exclusive": requested["end_frame_exclusive"],
                "frame_count": requested["end_frame_exclusive"]
                - requested["start_frame"],
            },
            "strictly_contains_reviewed_interval": (
                requested["start_frame"] <= current["start_frame"]
                and requested["end_frame_exclusive"] >= current["end_frame_exclusive"]
                and (
                    requested["start_frame"] < current["start_frame"]
                    or requested["end_frame_exclusive"] > current["end_frame_exclusive"]
                )
            ),
            "expansion_unit": "one_predeclared_adjacent_natural_boundary_level",
            "next_action": "retarget_render_and_repeat_independent_blind_arc_action_review",
        }
        if record["strictly_contains_reviewed_interval"] is not True:
            raise ValueError(f"Next context level does not expand {sample_id}")
        record["plan_record_sha256"] = value_sha256(record)
        expansion_requests.append(record)

    for rows in (expansion_requests, complete_current_context, context_exhausted):
        _assert_no_hidden_labels(rows)
    output_root = output_root.resolve()
    outputs = {}
    for name, rows in (
        ("expansion_requests", expansion_requests),
        ("complete_current_context", complete_current_context),
        ("context_exhausted", context_exhausted),
    ):
        path = output_root / f"{name}.jsonl"
        atomic_jsonl(path, rows)
        outputs[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "records": len(rows),
        }
    summary = {
        "schema_version": "1.0.0",
        "artifact_kind": "expression_turn_v8_one_level_natural_context_expansion_plan",
        "inputs": {
            "comparison_records_sha256": sha256_file(comparison_records),
            "hidden_mapping_sha256": sha256_file(hidden_mapping),
            "candidate_catalog_sha256": sha256_file(candidate_catalog),
        },
        "review_status_distribution": dict(
            sorted(
                Counter(
                    row["qualification_status"] for row in comparisons.values()
                ).items()
            )
        ),
        "selection_policy": "exactly_one_next_predeclared_natural_context_level",
        "fixed_minimum_maximum_or_target_duration_used": False,
        "outputs": outputs,
        "accepted_for_training_count": 0,
    }
    atomic_json(output_root / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_plan(
        comparison_records=args.comparison_records,
        hidden_mapping=args.hidden_mapping,
        candidate_catalog=args.candidate_catalog,
        output_root=args.output_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
