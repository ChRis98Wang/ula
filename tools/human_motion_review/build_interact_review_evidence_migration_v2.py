#!/usr/bin/env python3
"""Prove which InterAct blind-review evidence survives an axis-only bundle rebuild."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ARC_PROTOCOL = "interact_dyadic_arc_action_blind_video_native_bvh_v2"
PUBLIC_KIND = "interact_native_bvh_separate_anonymous_blind_review_bundle_v2"
ARC_REVIEW_KIND = "interact_dyadic_arc_action_blind_review_submission_v2"
UNCHANGED_SUMMARY_KEYS = (
    "arc_action_queue_sha256",
    "affect_queue_sha256",
    "dyad_run_state_sha256",
    "duration_policy",
    "dyadic_arc_action_records",
    "dyadic_affect_records",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-public-summary", type=Path, required=True)
    parser.add_argument("--new-public-summary", type=Path, required=True)
    parser.add_argument("--arc-review-submission", type=Path, required=True)
    parser.add_argument("--arc-review-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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


def atomic_json(path: Path, value: object) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _local_artifact(summary_path: Path, summary: dict[str, Any], key: str) -> Path:
    declared = summary.get(key)
    if not isinstance(declared, str) or not declared:
        raise ValueError(f"Public summary has invalid {key}")
    return summary_path.parent / Path(declared).name


def _index(rows: list[dict[str, Any]], *, context: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in result:
            raise ValueError(f"{context} has invalid or duplicate sample_id")
        result[sample_id] = row
    return result


def build_migration(
    *,
    old_public_summary: Path,
    new_public_summary: Path,
    arc_review_submission: Path,
    arc_review_summary: Path,
    output: Path,
) -> dict[str, Any]:
    old_public_summary = old_public_summary.resolve()
    new_public_summary = new_public_summary.resolve()
    arc_review_submission = arc_review_submission.resolve()
    arc_review_summary = arc_review_summary.resolve()
    old = read_json(old_public_summary)
    new = read_json(new_public_summary)
    if old.get("artifact_kind") != PUBLIC_KIND or new.get("artifact_kind") != PUBLIC_KIND:
        raise ValueError("Unexpected InterAct public bundle kind")
    if old.get("accepted_for_training") is not False or new.get("accepted_for_training") is not False:
        raise ValueError("Public bundle unexpectedly admits training")
    if old.get("fixed_duration_window_used") is not False or new.get(
        "fixed_duration_window_used"
    ) is not False:
        raise ValueError("Public bundle used a fixed-duration window")
    for key in UNCHANGED_SUMMARY_KEYS:
        if old.get(key) != new.get(key):
            raise ValueError(f"Dyadic evidence changed during axis rebuild: {key}")
    if old.get("axis_queue_sha256") == new.get("axis_queue_sha256"):
        raise ValueError("Axis queue did not change; this is not an axis-evidence migration")
    if old.get("axis_run_state_sha256") == new.get("axis_run_state_sha256"):
        raise ValueError("Axis run state did not change")

    actual_paths: dict[str, dict[str, Path]] = {}
    for version, summary_path, summary in (
        ("old", old_public_summary, old),
        ("new", new_public_summary, new),
    ):
        actual_paths[version] = {}
        for queue_key in ("axis_queue", "arc_action_queue", "affect_queue"):
            path = _local_artifact(summary_path, summary, queue_key)
            expected = summary[f"{queue_key}_sha256"]
            if sha256_file(path) != expected:
                raise ValueError(f"{version} {queue_key} SHA mismatch")
            actual_paths[version][queue_key] = path
    for queue_key in ("arc_action_queue", "affect_queue"):
        old_path = actual_paths["old"][queue_key]
        new_path = actual_paths["new"][queue_key]
        if old_path.read_bytes() != new_path.read_bytes():
            raise ValueError(f"{queue_key} bytes changed during axis rebuild")

    review_summary = read_json(arc_review_summary)
    if review_summary.get("artifact_kind") != ARC_REVIEW_KIND:
        raise ValueError("Unexpected arc/action review summary kind")
    if review_summary.get("accepted_for_training") is not False:
        raise ValueError("Arc/action review unexpectedly admits training")
    integrity = review_summary.get("input_integrity") or {}
    old_summary_hash = sha256_file(old_public_summary)
    if integrity.get("public_summary_sha256") != old_summary_hash:
        raise ValueError("Arc/action review is not bound to the archived public summary")
    if integrity.get("arc_action_queue_sha256") != old["arc_action_queue_sha256"]:
        raise ValueError("Arc/action review is not bound to the unchanged queue")
    if review_summary.get("submission_jsonl_sha256") != sha256_file(
        arc_review_submission
    ):
        raise ValueError("Arc/action submission SHA mismatch")

    queue = _index(
        read_jsonl(actual_paths["new"]["arc_action_queue"]), context="new arc queue"
    )
    reviews = _index(read_jsonl(arc_review_submission), context="arc review submission")
    if set(queue) != set(reviews):
        raise ValueError("Arc queue and review sample sets differ")
    for sample_id, review in reviews.items():
        queued = queue[sample_id]
        if review.get("video_sha256") != queued.get("video_sha256"):
            raise ValueError(f"Arc video binding changed: {sample_id}")
        if review.get("protocol_version") != ARC_PROTOCOL:
            raise ValueError(f"Unexpected arc review protocol: {sample_id}")
        if (
            review.get("accepted_for_training") is not False
            or review.get("fixed_duration_window_used") is not False
            or review.get("native_duration_preserved") is not True
            or review.get("temporal_unit") != "complete_natural_interaction_arc"
        ):
            raise ValueError(f"Arc review violates fail-closed duration contract: {sample_id}")
        video_path = Path(queued["video_path"]).resolve()
        if sha256_file(video_path) != queued["video_sha256"]:
            raise ValueError(f"Current dyad video SHA mismatch: {sample_id}")

    record = {
        "schema_version": "2.0.0",
        "artifact_kind": "interact_blind_review_axis_only_bundle_migration_evidence_v2",
        "old_public_summary": {
            "path": str(old_public_summary),
            "sha256": old_summary_hash,
        },
        "new_public_summary": {
            "path": str(new_public_summary),
            "sha256": sha256_file(new_public_summary),
        },
        "unchanged_dyadic_evidence": {
            "arc_action_queue_sha256": new["arc_action_queue_sha256"],
            "affect_queue_sha256": new["affect_queue_sha256"],
            "dyad_run_state_sha256": new["dyad_run_state_sha256"],
            "arc_action_queue_byte_identical": True,
            "affect_queue_byte_identical": True,
        },
        "changed_axis_evidence": {
            "old_axis_queue_sha256": old["axis_queue_sha256"],
            "new_axis_queue_sha256": new["axis_queue_sha256"],
            "old_axis_run_state_sha256": old["axis_run_state_sha256"],
            "new_axis_run_state_sha256": new["axis_run_state_sha256"],
        },
        "carried_forward_review": {
            "review_axis": "arc_action",
            "submission_path": str(arc_review_submission),
            "submission_sha256": sha256_file(arc_review_submission),
            "summary_path": str(arc_review_summary),
            "summary_sha256": sha256_file(arc_review_summary),
            "records": len(reviews),
            "all_sample_and_video_hashes_match_new_queue": True,
        },
        "review_axes_allowed_to_carry_forward": ["arc_action"],
        "review_axes_invalidated_and_require_fresh_blind_review": ["axis"],
        "fixed_duration_window_used": False,
        "accepted_for_training": False,
    }
    record["evidence_record_sha256"] = value_sha256(record)
    atomic_json(output.resolve(), record)
    return record


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_migration(
        old_public_summary=args.old_public_summary,
        new_public_summary=args.new_public_summary,
        arc_review_submission=args.arc_review_submission,
        arc_review_summary=args.arc_review_summary,
        output=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
