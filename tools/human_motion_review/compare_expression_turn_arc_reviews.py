#!/usr/bin/env python3
"""Fail-closed comparison of two anonymous expression arc/action reviews."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any


R1_ACTION_PASS = "observable_match"
R2_ACTION_PASS = "pass"
PHASES = ("onset", "apex", "offset")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-summary", type=Path, required=True)
    parser.add_argument("--review-r1", type=Path, required=True)
    parser.add_argument("--review-r2", type=Path, required=True)
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


def index_unique(rows: list[dict[str, Any]], *, context: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{context} contains a missing sample_id")
        if sample_id in result:
            raise ValueError(f"{context} contains duplicate sample_id {sample_id}")
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


def _reviewer_id(row: dict[str, Any]) -> str:
    arc = row.get("arc_reviewer_id")
    action = row.get("action_reviewer_id")
    if not isinstance(arc, str) or not arc or arc != action:
        raise ValueError(f"{row.get('sample_id')}: arc/action reviewer identity is invalid")
    return arc


def _validate_common(
    row: dict[str, Any], queue: dict[str, Any], *, queue_hash: str, role: str
) -> None:
    sample_id = queue["sample_id"]
    for key in (
        "sample_id",
        "video_path",
        "video_sha256",
        "context_level",
        "audio_available",
        "label_metadata_exposed",
    ):
        if row.get(key) != queue.get(key):
            raise ValueError(f"{sample_id}/{role}: {key} binding mismatch")
    if row.get("queue_sha256") != queue_hash:
        raise ValueError(f"{sample_id}/{role}: queue SHA mismatch")
    if row.get("training_admission") is not False:
        raise ValueError(f"{sample_id}/{role}: training admission must remain false")
    if row.get("audio_available") is not False or row.get("label_metadata_exposed") is not False:
        raise ValueError(f"{sample_id}/{role}: blind-review boundary is invalid")
    if row.get("arc_protocol_version") != "robot_expression_arc_blind_video_v1":
        raise ValueError(f"{sample_id}/{role}: arc protocol mismatch")
    if row.get("action_protocol_version") != "robot_action_semantics_blind_video_v1":
        raise ValueError(f"{sample_id}/{role}: action protocol mismatch")


def _normalized_review(row: dict[str, Any], *, role: str) -> dict[str, Any]:
    phase_status = {phase: row.get(f"{phase}_status") for phase in PHASES}
    if any(not isinstance(status, str) or not status for status in phase_status.values()):
        raise ValueError(f"{row.get('sample_id')}/{role}: missing phase status")
    action_result = row.get("action_result")
    action_passed = (
        action_result == R1_ACTION_PASS
        if role == "r1"
        else action_result == R2_ACTION_PASS
        and row.get("action_observability") == "observable"
    )
    return {
        "reviewer_id": _reviewer_id(row),
        "arc_review_id": row.get("arc_review_id"),
        "action_review_id": row.get("action_review_id"),
        "phase_status": phase_status,
        "all_phases_complete": all(status == "complete" for status in phase_status.values()),
        "action_result": action_result,
        "action_passed": action_passed,
        "observable_description": row.get("observable_description"),
        "submission_record_sha256": value_sha256(row),
    }


def compare_reviews(
    *, public_summary: Path, review_r1: Path, review_r2: Path, output_root: Path
) -> dict[str, Any]:
    public_summary = public_summary.resolve()
    review_r1 = review_r1.resolve()
    review_r2 = review_r2.resolve()
    public = json.loads(public_summary.read_text(encoding="utf-8"))
    if public.get("artifact_kind") != "expression_turn_v8_separate_blind_review_bundle":
        raise ValueError("Public bundle artifact_kind is invalid")
    if public.get("source_identity_official_action_text_and_emotion_exposed") is not False:
        raise ValueError("Public bundle exposed hidden labels")
    if public.get("accepted_for_training") is not False:
        raise ValueError("Public bundle unexpectedly admits training")
    queue_path = Path(public["arc_action_queue"]).resolve()
    queue_hash = sha256_file(queue_path)
    if public.get("arc_action_queue_sha256") != queue_hash:
        raise ValueError("Public arc/action queue SHA mismatch")
    queue = index_unique(read_jsonl(queue_path), context="public queue")
    r1 = index_unique(read_jsonl(review_r1), context="R1 review")
    r2 = index_unique(read_jsonl(review_r2), context="R2 review")
    if not (set(queue) == set(r1) == set(r2)):
        raise ValueError("R1/R2/public sample coverage differs")

    records = []
    for sample_id in sorted(queue):
        queued = queue[sample_id]
        _validate_common(r1[sample_id], queued, queue_hash=queue_hash, role="r1")
        _validate_common(r2[sample_id], queued, queue_hash=queue_hash, role="r2")
        first = _normalized_review(r1[sample_id], role="r1")
        second = _normalized_review(r2[sample_id], role="r2")
        if first["reviewer_id"] == second["reviewer_id"]:
            raise ValueError(f"{sample_id}: R1 and R2 are not independent")
        action_passed = first["action_passed"] and second["action_passed"]
        both_complete = (
            action_passed
            and first["all_phases_complete"]
            and second["all_phases_complete"]
        )
        if not action_passed:
            status = "action_unusable_or_conflicting"
        elif both_complete:
            status = "complete_arc_action_candidate"
        else:
            status = "natural_context_expansion_required"
        disagreement = (
            first["action_passed"] != second["action_passed"]
            or any(
                (first["phase_status"][phase] == "complete")
                != (second["phase_status"][phase] == "complete")
                for phase in PHASES
            )
        )
        record = {
            "schema_version": "1.0.0",
            "artifact_kind": "expression_turn_v8_dual_arc_action_review_comparison",
            "sample_id": sample_id,
            "video_path": queued["video_path"],
            "video_sha256": queued["video_sha256"],
            "context_level": queued["context_level"],
            "r1": first,
            "r2": second,
            "qualification_status": status,
            "review_disagreement": disagreement,
            "expansion_policy": (
                "expand_along_predeclared_adjacent_natural_boundaries_when_either_"
                "reviewer_marks_any_arc_phase_incomplete"
            ),
            "elapsed_duration_used_as_gate": False,
            "emotion_review_performed": False,
            "accepted_for_training": False,
        }
        record["comparison_record_sha256"] = value_sha256(record)
        records.append(record)

    output_root = output_root.resolve()
    groups = {
        "all_records": records,
        "complete_arc_action_candidates": [
            row
            for row in records
            if row["qualification_status"] == "complete_arc_action_candidate"
        ],
        "expansion_required": [
            row
            for row in records
            if row["qualification_status"] == "natural_context_expansion_required"
        ],
        "action_unusable": [
            row
            for row in records
            if row["qualification_status"] == "action_unusable_or_conflicting"
        ],
        "review_disagreements": [row for row in records if row["review_disagreement"]],
    }
    paths = {}
    for name, rows in groups.items():
        path = output_root / f"{name}.jsonl"
        atomic_jsonl(path, rows)
        paths[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "records": len(rows),
        }
    summary = {
        "schema_version": "1.0.0",
        "artifact_kind": "expression_turn_v8_dual_arc_action_review_comparison_summary",
        "public_summary": str(public_summary),
        "public_summary_sha256": sha256_file(public_summary),
        "arc_action_queue_sha256": queue_hash,
        "review_r1": str(review_r1),
        "review_r1_sha256": sha256_file(review_r1),
        "review_r2": str(review_r2),
        "review_r2_sha256": sha256_file(review_r2),
        "records": len(records),
        "qualification_status_distribution": dict(
            sorted(Counter(row["qualification_status"] for row in records).items())
        ),
        "review_disagreement_count": sum(row["review_disagreement"] for row in records),
        "duration_policy": "observable_complete_arc_not_fixed_elapsed_seconds",
        "accepted_for_training_count": 0,
        "outputs": paths,
    }
    atomic_json(output_root / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = compare_reviews(
        public_summary=args.public_summary,
        review_r1=args.review_r1,
        review_r2=args.review_r2,
        output_root=args.output_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
