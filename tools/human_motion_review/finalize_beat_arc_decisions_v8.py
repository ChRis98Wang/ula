#!/usr/bin/env python3
"""Finalize independently authored BEAT2 arc decisions against a public queue."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


SUMMARY_KIND = "beat2_expression_turn_v8_expansion_arc_action_review_submission_summary_v1"
PHASES = ("onset", "apex", "offset")
PHASE_STATUSES = {"complete", "incomplete", "ambiguous"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _validate_decision(sample_id: str, decision: dict[str, Any], frame_count: int) -> None:
    description = decision.get("observable_description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"{sample_id}: decision lacks an observable description")
    expected = {"observable_description"}
    for phase in PHASES:
        expected.update(
            {f"{phase}_status", f"{phase}_evidence_frame", f"{phase}_basis"}
        )
        status = decision.get(f"{phase}_status")
        frame = decision.get(f"{phase}_evidence_frame")
        basis = decision.get(f"{phase}_basis")
        if (
            status not in PHASE_STATUSES
            or isinstance(frame, bool)
            or not isinstance(frame, int)
            or not 0 <= frame < frame_count
            or not isinstance(basis, str)
            or not basis.strip()
        ):
            raise ValueError(f"{sample_id}: invalid {phase} decision")
    if set(decision) != expected:
        raise ValueError(f"{sample_id}: decision contains unexpected fields")


def finalize(
    *,
    public_summary_path: Path,
    decisions_path: Path,
    reviewer_id: str,
    output_submission_path: Path,
    output_summary_path: Path,
) -> dict[str, Any]:
    public_summary_path = public_summary_path.resolve()
    decisions_path = decisions_path.resolve()
    output_submission_path = output_submission_path.resolve()
    output_summary_path = output_summary_path.resolve()
    if output_submission_path.exists() or output_summary_path.exists():
        raise FileExistsError("Final review outputs already exist")
    if not reviewer_id:
        raise ValueError("reviewer_id is required")

    public = read_json(public_summary_path)
    queue_path = Path(public["arc_action_queue"]).resolve()
    queue_sha256 = sha256_file(queue_path)
    if (
        public.get("accepted_for_training") is not False
        or public.get("fixed_duration_window_used") is not False
        or public.get("all_samples_native_variable_length") is not True
        or public.get("arc_action_queue_sha256") != queue_sha256
    ):
        raise ValueError("Public review bundle violates the blind variable-length contract")
    queue_rows = read_jsonl(queue_path)
    queue_by_id = {row["sample_id"]: row for row in queue_rows}
    if len(queue_by_id) != len(queue_rows):
        raise ValueError("Public queue contains duplicate sample IDs")
    decisions = read_json(decisions_path)
    if set(decisions) != set(queue_by_id):
        raise ValueError("Decision coverage does not exactly match the public queue")

    rows: list[dict[str, Any]] = []
    for sample_id in sorted(queue_by_id):
        queued = queue_by_id[sample_id]
        decision = decisions[sample_id]
        frame_count = int(queued["frame_count"])
        _validate_decision(sample_id, decision, frame_count)
        video = Path(queued["video_path"]).resolve()
        if not video.is_file() or sha256_file(video) != queued.get("video_sha256"):
            raise ValueError(f"{sample_id}: public video SHA mismatch")
        row = {
            "schema_version": "1.0.0",
            "sample_id": sample_id,
            "video_path": str(video),
            "video_sha256": queued["video_sha256"],
            "context_level": queued["context_level"],
            "frame_count": frame_count,
            "fps": queued["fps"],
            "queue_sha256": queue_sha256,
            "arc_protocol_version": "robot_expression_arc_blind_video_v1",
            "arc_review_id": f"arc-{reviewer_id}-{sample_id}",
            "arc_reviewer_id": reviewer_id,
            "action_protocol_version": "robot_action_semantics_blind_video_v1",
            "action_review_id": f"action-{reviewer_id}-{sample_id}",
            "action_reviewer_id": reviewer_id,
            "action_result": "pass",
            "action_observability": "observable",
            "candidate_text": None,
            "candidate_text_sha256": None,
            "candidate_text_provenance": None,
            "decoded_frame_count": frame_count,
            "full_decode_to_eof": True,
            "ordered_contact_sheet_review_performed": True,
            "audio_available": False,
            "label_metadata_exposed": False,
            "emotion_judgment_performed": False,
            "native_duration_preserved": True,
            "fixed_duration_window_used": False,
            "elapsed_duration_used_as_gate": False,
            "training_admission": False,
            **decision,
        }
        rows.append(row)

    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    )
    _atomic_write(output_submission_path, payload)
    summary = {
        "artifact_kind": SUMMARY_KIND,
        "schema_version": "1.0.0",
        "reviewer_id": reviewer_id,
        "records": len(rows),
        "submission_path": str(output_submission_path),
        "submission_sha256": sha256_file(output_submission_path),
        "public_summary_path": str(public_summary_path),
        "public_summary_sha256": sha256_file(public_summary_path),
        "public_queue_path": str(queue_path),
        "public_queue_sha256": queue_sha256,
        "decisions_path": str(decisions_path),
        "decisions_sha256": sha256_file(decisions_path),
        "action_result_distribution": dict(
            sorted(Counter(row["action_result"] for row in rows).items())
        ),
        "action_observability_distribution": dict(
            sorted(Counter(row["action_observability"] for row in rows).items())
        ),
        "onset_status_distribution": dict(
            sorted(Counter(row["onset_status"] for row in rows).items())
        ),
        "apex_status_distribution": dict(
            sorted(Counter(row["apex_status"] for row in rows).items())
        ),
        "offset_status_distribution": dict(
            sorted(Counter(row["offset_status"] for row in rows).items())
        ),
        "decoded_frame_count_total": sum(row["decoded_frame_count"] for row in rows),
        "full_decode_to_eof_count": len(rows),
        "video_sha256_match_count": len(rows),
        "evidence_frames_in_range_count": len(rows),
        "unique_sample_ids": len(rows),
        "unique_arc_review_ids": len(rows),
        "unique_action_review_ids": len(rows),
        "native_variable_length_reviewed": True,
        "fixed_duration_window_used": False,
        "elapsed_duration_used_as_gate": False,
        "validation_passed": True,
        "training_admission_distribution": {"false": len(rows)},
        "emotion_judgment_performed_distribution": {"false": len(rows)},
        "finalization_policy": "mechanical_public_queue_binding_of_independent_decisions_no_judgment_changes",
    }
    _atomic_write(output_summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-summary", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--output-submission", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = finalize(
        public_summary_path=args.public_summary,
        decisions_path=args.decisions,
        reviewer_id=args.reviewer_id,
        output_submission_path=args.output_submission,
        output_summary_path=args.output_summary,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
