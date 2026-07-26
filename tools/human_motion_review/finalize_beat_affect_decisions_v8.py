#!/usr/bin/env python3
"""Bind independent BEAT2 v8 affect decisions to an anonymous public queue."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

from tools.human_motion_review.finalize_interact_full_dyad_review_shard_v3 import (
    atomic_write,
    probe_video,
    read_json,
    read_jsonl,
    sha256_file,
    stable_json,
)


PUBLIC_KIND = "expression_turn_v8_expansion_separate_blind_review_bundle_v1"
PROTOCOL = "robot_affect_blind_video_v1"
SUMMARY_KIND = "robot_affect_blind_review_submission_summary_v1"
RESULTS = {"observable", "ambiguous", "not_observable"}
CLASSES = {"angry", "fear", "happy", "neutral", "sad", "surprise"}
INTENSITIES = {"low", "medium", "high"}


def _validate_decision(sample_id: str, decision: object, frames: int) -> None:
    expected = {
        "result",
        "predicted_class",
        "confidence",
        "intensity",
        "evidence",
        "evidence_frames",
        "full_video_reviewed",
    }
    if (
        not isinstance(decision, dict)
        or set(decision) != expected
        or decision.get("full_video_reviewed") is not True
        or decision.get("result") not in RESULTS
        or not isinstance(decision.get("evidence"), str)
        or not decision["evidence"].strip()
    ):
        raise ValueError(f"{sample_id}: malformed affect decision")
    evidence_frames = decision.get("evidence_frames")
    if (
        not isinstance(evidence_frames, list)
        or not evidence_frames
        or len(set(evidence_frames)) != len(evidence_frames)
        or any(
            isinstance(frame, bool) or not isinstance(frame, int) or not 0 <= frame < frames
            for frame in evidence_frames
        )
    ):
        raise ValueError(f"{sample_id}: invalid affect evidence frames")
    predicted = decision.get("predicted_class")
    confidence = decision.get("confidence")
    intensity = decision.get("intensity")
    if decision["result"] == "observable":
        if (
            predicted not in CLASSES
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
            or intensity not in INTENSITIES
        ):
            raise ValueError(f"{sample_id}: invalid observable affect decision")
    elif any(value is not None for value in (predicted, confidence, intensity)):
        raise ValueError(f"{sample_id}: non-observable decision carries a pseudo-label")


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
        raise FileExistsError("Final affect outputs already exist")
    if not reviewer_id.strip():
        raise ValueError("reviewer_id is required")

    public = read_json(public_summary_path)
    queue_path = Path(str(public.get("affect_queue") or "")).resolve()
    queue_sha256 = sha256_file(queue_path) if queue_path.is_file() else None
    if (
        public.get("artifact_kind") != PUBLIC_KIND
        or public.get("accepted_for_training") is not False
        or public.get("all_samples_native_variable_length") is not True
        or public.get("fixed_duration_window_used") is not False
        or queue_path.parent != public_summary_path.parent
        or public.get("affect_queue_sha256") != queue_sha256
    ):
        raise ValueError("Public affect bundle violates its queue binding contract")
    queue_rows = read_jsonl(queue_path)
    queue = {row.get("sample_id"): row for row in queue_rows}
    if (
        len(queue) != len(queue_rows)
        or any(not isinstance(sample_id, str) or not sample_id for sample_id in queue)
    ):
        raise ValueError("Public affect queue contains invalid or duplicate sample IDs")
    decisions = read_json(decisions_path)
    if set(decisions) != set(queue):
        raise ValueError("Affect decision coverage does not exactly match the queue")

    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for sample_id in sorted(queue):
        queued = queue[sample_id]
        video = Path(str(queued.get("video_path") or "")).resolve()
        if not video.is_file() or sha256_file(video) != queued.get("video_sha256"):
            raise ValueError(f"{sample_id}: public video SHA mismatch")
        probe = probe_video(video)
        if (
            probe["frames"] != queued.get("frame_count")
            or not math.isclose(float(probe["fps"]), float(queued.get("fps")), abs_tol=1e-6)
        ):
            raise ValueError(f"{sample_id}: decoded video metadata differs from queue")
        decision = decisions[sample_id]
        _validate_decision(sample_id, decision, probe["frames"])
        row = dict(queued)
        row.update(decision)
        row.update(
            {
                "affect_review_id": f"affect-{reviewer_id}-{sample_id}",
                "affect_reviewer_id": reviewer_id,
                "source_affect_queue_sha256": queue_sha256,
                "full_video_reviewed": True,
                "decode_started_at_frame": 0,
                "decode_reached_eof": True,
                "decoded_frame_count": probe["frames"],
                "frame_count_verified": True,
                "video_sha256_verified": True,
                "training_admission": False,
            }
        )
        rows.append(row)
        counts[decision["result"]] += 1

    atomic_write(output_submission_path, "".join(stable_json(row) + "\n" for row in rows))
    summary = {
        "artifact_kind": SUMMARY_KIND,
        "schema_version": "1.0.0",
        "reviewer_id": reviewer_id,
        "records_expected": len(queue),
        "records_reviewed": len(rows),
        "coverage": {
            "expected_records": len(queue),
            "reviewed_records": len(rows),
            "complete": True,
            "fraction": 1,
        },
        "result_distribution": {
            result: counts.get(result, 0) for result in sorted(RESULTS)
        },
        "submission_path": str(output_submission_path),
        "submission_sha256": sha256_file(output_submission_path),
        "source_public_summary_path": str(public_summary_path),
        "source_public_summary_sha256": sha256_file(public_summary_path),
        "source_affect_queue_path": str(queue_path),
        "source_affect_queue_sha256": queue_sha256,
        "strict_blind_attestation": {
            "all_full_video_reviewed": True,
            "all_native_variable_length_reviewed": True,
            "all_training_admission_false": True,
            "audio_used": False,
            "body_motion_only": True,
            "fixed_seconds_or_fixed_duration_window_used": False,
            "hidden_mapping_or_original_labels_used": False,
        },
        "training_admission": False,
    }
    atomic_write(output_summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
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
