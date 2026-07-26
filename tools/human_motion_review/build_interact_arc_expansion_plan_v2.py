#!/usr/bin/env python3
"""Build fail-closed one-level InterAct dyadic natural-context expansions."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any


COMPLETE = "complete"
INCOMPLETE = "incomplete_requires_natural_boundary_expansion"
ARC_PROTOCOL = "interact_dyadic_arc_action_blind_video_native_bvh_v2"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-summary", type=Path, required=True)
    parser.add_argument("--review-submission", type=Path, required=True)
    parser.add_argument("--review-summary", type=Path, required=True)
    parser.add_argument("--migration-evidence", type=Path, required=True)
    parser.add_argument("--hidden-mapping", type=Path, required=True)
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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


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


def index_rows(rows: list[dict[str, Any]], *, context: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in result:
            raise ValueError(f"{context} contains invalid or duplicate sample_id")
        result[sample_id] = row
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


def _levels(mapping: dict[str, Any]) -> dict[int, dict[str, Any]]:
    context = mapping.get("context_plan") or {}
    if context.get("duration_gate_used") is not False:
        raise ValueError("InterAct context plan used a duration gate")
    result = {}
    for level in context.get("levels") or []:
        index = level.get("level")
        start = level.get("start_frame")
        end = level.get("end_frame_exclusive")
        if (
            not isinstance(index, int)
            or index in result
            or not isinstance(start, int)
            or not isinstance(end, int)
            or end <= start
        ):
            raise ValueError(f"Invalid context plan for {mapping.get('sample_id')}")
        result[index] = level
    return result


def _interval(level: dict[str, Any]) -> dict[str, int]:
    start = int(level["start_frame"])
    end = int(level["end_frame_exclusive"])
    return {"start_frame": start, "end_frame_exclusive": end, "frame_count": end - start}


def build_plan(
    *,
    public_summary: Path,
    review_submission: Path,
    review_summary: Path,
    migration_evidence: Path,
    hidden_mapping: Path,
    output_root: Path,
) -> dict[str, Any]:
    paths = [
        public_summary.resolve(),
        review_submission.resolve(),
        review_summary.resolve(),
        migration_evidence.resolve(),
        hidden_mapping.resolve(),
    ]
    public_summary, review_submission, review_summary, migration_evidence, hidden_mapping = paths
    public = read_json(public_summary)
    migration = read_json(migration_evidence)
    review_meta = read_json(review_summary)
    if migration.get("artifact_kind") != "interact_blind_review_axis_only_bundle_migration_evidence_v2":
        raise ValueError("Unexpected review migration evidence")
    migration_payload = dict(migration)
    migration_record_hash = migration_payload.pop("evidence_record_sha256", None)
    if migration_record_hash != value_sha256(migration_payload):
        raise ValueError("Review migration evidence record SHA mismatch")
    if migration.get("new_public_summary", {}).get("sha256") != sha256_file(public_summary):
        raise ValueError("Migration is not bound to the current public bundle")
    carried = migration.get("carried_forward_review") or {}
    if carried.get("submission_sha256") != sha256_file(review_submission):
        raise ValueError("Migration is not bound to the arc/action submission")
    if carried.get("summary_sha256") != sha256_file(review_summary):
        raise ValueError("Migration is not bound to the arc/action review summary")
    if migration.get("review_axes_allowed_to_carry_forward") != ["arc_action"]:
        raise ValueError("Migration does not explicitly carry arc/action evidence")
    if review_meta.get("accepted_for_training") is not False:
        raise ValueError("Review summary unexpectedly admits training")
    if review_meta.get("submission_jsonl_sha256") != sha256_file(review_submission):
        raise ValueError("Review summary submission SHA mismatch")

    queue_path = public_summary.parent / Path(public["arc_action_queue"]).name
    if sha256_file(queue_path) != public.get("arc_action_queue_sha256"):
        raise ValueError("Current arc/action queue SHA mismatch")
    queue = index_rows(read_jsonl(queue_path), context="arc/action queue")
    reviews = index_rows(read_jsonl(review_submission), context="arc/action reviews")
    hidden = index_rows(read_jsonl(hidden_mapping), context="hidden mapping")
    if set(queue) != set(reviews) or set(queue) != set(hidden):
        raise ValueError("Queue, review, and hidden mapping sample sets differ")

    expansions = []
    complete = []
    for sample_id in sorted(queue):
        queued = queue[sample_id]
        review = reviews[sample_id]
        mapping = hidden[sample_id]
        if review.get("protocol_version") != ARC_PROTOCOL:
            raise ValueError(f"Unexpected review protocol: {sample_id}")
        if review.get("video_sha256") != queued.get("video_sha256"):
            raise ValueError(f"Review video binding mismatch: {sample_id}")
        if (
            review.get("accepted_for_training") is not False
            or review.get("fixed_duration_window_used") is not False
            or review.get("native_duration_preserved") is not True
        ):
            raise ValueError(f"Review violates fail-closed duration contract: {sample_id}")
        current_level = mapping.get("displayed_context_level")
        if review.get("context_level") != current_level or not isinstance(current_level, int):
            raise ValueError(f"Reviewed context level mismatch: {sample_id}")
        levels = _levels(mapping)
        current = levels.get(current_level)
        if current is None:
            raise ValueError(f"Current context is missing: {sample_id}")
        common = {
            "schema_version": "2.0.0",
            "sample_id": sample_id,
            "turn_id": mapping["turn_id"],
            "base_video_sha256": queued["video_sha256"],
            "reviewed_context_level": current_level,
            "reviewed_interval": _interval(current),
            "review_submission_sha256": sha256_file(review_submission),
            "migration_evidence_record_sha256": migration_record_hash,
            "temporal_unit": "complete_natural_interaction_arc",
            "elapsed_duration_used_as_gate": False,
            "semantic_supervision_mask": False,
            "emotion_supervision_mask": False,
            "accepted_for_training": False,
        }
        status = review.get("expression_completeness_result")
        if status == COMPLETE:
            record = {
                **common,
                "artifact_kind": "interact_dyadic_current_context_arc_action_complete_v2",
                "next_action": "hold_for_independent_affect_and_remaining_admission_gates",
            }
            record["plan_record_sha256"] = value_sha256(record)
            complete.append(record)
            continue
        if status != INCOMPLETE:
            raise ValueError(f"Unknown expression completeness result: {status}")
        request = review.get("expansion_request") or {}
        next_level = request.get("next_context_level")
        if (
            request.get("required") is not True
            or request.get("boundary_policy") != "next_predeclared_natural_context_level"
            or next_level != current_level + 1
        ):
            raise ValueError(f"Invalid natural-boundary expansion request: {sample_id}")
        requested = levels.get(next_level)
        if requested is None:
            raise ValueError(f"Requested context level is missing: {sample_id}")
        current_interval = _interval(current)
        requested_interval = _interval(requested)
        if not (
            requested_interval["start_frame"] <= current_interval["start_frame"]
            and requested_interval["end_frame_exclusive"]
            >= current_interval["end_frame_exclusive"]
            and requested_interval != current_interval
        ):
            raise ValueError(f"Next context level does not expand: {sample_id}")
        record = {
            **common,
            "artifact_kind": "interact_dyadic_one_level_natural_context_expansion_request_v2",
            "requested_context_level": next_level,
            "requested_interval": requested_interval,
            "review_requested_directions": {
                "extend_before": request.get("extend_before") is True,
                "extend_after": request.get("extend_after") is True,
            },
            "actual_one_level_expansion": {
                "extended_before": requested_interval["start_frame"]
                < current_interval["start_frame"],
                "extended_after": requested_interval["end_frame_exclusive"]
                > current_interval["end_frame_exclusive"],
            },
            "expansion_unit": "exactly_one_next_predeclared_shared_rest_boundary_level",
            "next_action": "render_and_repeat_independent_blind_arc_action_review",
        }
        record["plan_record_sha256"] = value_sha256(record)
        expansions.append(record)

    output_root = output_root.resolve()
    outputs = {}
    for name, rows in (("expansion_requests", expansions), ("complete_current_context", complete)):
        path = output_root / f"{name}.jsonl"
        atomic_jsonl(path, rows)
        outputs[name] = {"path": str(path), "sha256": sha256_file(path), "records": len(rows)}
    summary = {
        "schema_version": "2.0.0",
        "artifact_kind": "interact_dyadic_arc_action_one_level_expansion_plan_v2",
        "inputs": {
            "public_summary_sha256": sha256_file(public_summary),
            "review_submission_sha256": sha256_file(review_submission),
            "review_summary_sha256": sha256_file(review_summary),
            "migration_evidence_sha256": sha256_file(migration_evidence),
            "hidden_mapping_sha256": sha256_file(hidden_mapping),
        },
        "expression_completeness_distribution": dict(
            sorted(Counter(row["expression_completeness_result"] for row in reviews.values()).items())
        ),
        "selection_policy": "exactly_one_next_predeclared_shared_rest_boundary_level",
        "fixed_minimum_maximum_or_target_duration_used": False,
        "outputs": outputs,
        "accepted_for_training_count": 0,
    }
    atomic_json(output_root / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_plan(
        public_summary=args.public_summary,
        review_submission=args.review_submission,
        review_summary=args.review_summary,
        migration_evidence=args.migration_evidence,
        hidden_mapping=args.hidden_mapping,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
