#!/usr/bin/env python3
"""Bind independent InterAct v3 shard decisions to their anonymous queue."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any


SHARD_KIND = "anonymous_blind_review_queue_shard_v1"
SUBMISSION_KIND = "interact_full_dyad_blind_review_shard_submission_v3"
SUMMARY_KIND = "interact_full_dyad_blind_review_shard_submission_summary_v3"
PHASES = ("onset", "apex", "offset")
PHASE_STATUSES = {"complete", "incomplete", "ambiguous"}
AFFECT_RESULTS = {"observable", "ambiguous", "not_observable"}
AFFECT_CLASSES = {"neutral", "sad", "happy", "angry", "surprise", "fear"}
INTENSITIES = {"low", "medium", "high"}


def stable_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_sha256(value: object) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def atomic_write(path: Path, payload: str) -> None:
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


def probe_video(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return _probe_video_opencv(path)
    command = [
        ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames,avg_frame_rate,width,height",
        "-of",
        "json",
        str(path),
    ]
    try:
        payload = json.loads(subprocess.check_output(command, text=True))
        stream = payload["streams"][0]
        numerator, denominator = (int(part) for part in stream["avg_frame_rate"].split("/"))
        result = {
            "frames": int(stream["nb_read_frames"]),
            "fps": numerator / denominator,
            "width": int(stream["width"]),
            "height": int(stream["height"]),
        }
    except (
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        ZeroDivisionError,
        subprocess.CalledProcessError,
    ) as error:
        raise ValueError(f"Could not completely probe review video: {path}") from error
    if (
        result["frames"] <= 0
        or not math.isclose(result["fps"], 30.0, rel_tol=0.0, abs_tol=1e-6)
        or result["width"] != 1280
        or result["height"] != 720
    ):
        raise ValueError(f"Review video violates the v3 evidence contract: {path}")
    return result


def _probe_video_opencv(path: Path) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as error:
        raise ValueError("Neither ffprobe nor OpenCV is available for full decode") from error
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open review video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        reported_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        decoded_frames = 0
        while True:
            decoded, frame = capture.read()
            if not decoded:
                break
            if frame is None or frame.shape[:2] != (height, width):
                raise ValueError(f"Decoded frame geometry changed: {path}")
            decoded_frames += 1
    finally:
        capture.release()
    result = {
        "frames": decoded_frames,
        "fps": fps,
        "width": width,
        "height": height,
    }
    if (
        decoded_frames <= 0
        or reported_frames != decoded_frames
        or not math.isclose(fps, 30.0, rel_tol=0.0, abs_tol=1e-6)
        or width != 1280
        or height != 720
    ):
        raise ValueError(f"Review video violates the v3 evidence contract: {path}")
    return result


def _text(value: object, *, optional: bool = False) -> bool:
    return (optional and value is None) or (isinstance(value, str) and bool(value.strip()))


def _validate_arc_decision(sample_id: str, decision: dict[str, Any], frames: int) -> None:
    expected = {
        "full_video_reviewed",
        "onset_status",
        "onset_evidence_frame",
        "apex_status",
        "apex_evidence_frame",
        "offset_status",
        "offset_evidence_frame",
        "interaction_observable_result",
        "interaction_description_en",
        "robot_a_observable_motion_en",
        "robot_b_observable_motion_en",
        "expression_completeness_result",
        "expansion_request",
        "review_notes",
    }
    if set(decision) != expected or decision.get("full_video_reviewed") is not True:
        raise ValueError(f"{sample_id}: malformed arc decision")
    statuses = []
    for phase in PHASES:
        status = decision.get(f"{phase}_status")
        frame = decision.get(f"{phase}_evidence_frame")
        if (
            status not in PHASE_STATUSES
            or isinstance(frame, bool)
            or not isinstance(frame, int)
            or not 0 <= frame < frames
        ):
            raise ValueError(f"{sample_id}: invalid {phase} evidence")
        statuses.append(status)
    expected_completeness = (
        "complete"
        if all(status == "complete" for status in statuses)
        else "ambiguous"
        if any(status == "ambiguous" for status in statuses)
        else "incomplete"
    )
    if decision.get("expression_completeness_result") != expected_completeness:
        raise ValueError(f"{sample_id}: phase/completeness disagreement")
    observable = decision.get("interaction_observable_result")
    if observable not in {"observable", "ambiguous", "not_observable"}:
        raise ValueError(f"{sample_id}: invalid interaction observability")
    descriptions = (
        decision.get("interaction_description_en"),
        decision.get("robot_a_observable_motion_en"),
        decision.get("robot_b_observable_motion_en"),
    )
    if observable == "observable":
        if not all(_text(value) for value in descriptions):
            raise ValueError(f"{sample_id}: observable interaction lacks descriptions")
    elif any(value is not None for value in descriptions):
        raise ValueError(f"{sample_id}: non-observable interaction carries pseudo-text")
    if decision.get("expansion_request") is not None:
        raise ValueError(f"{sample_id}: full-span review cannot request an inside-context crop")
    if not _text(decision.get("review_notes")):
        raise ValueError(f"{sample_id}: arc review notes are required")


def _validate_actor_affect(sample_id: str, actor: str, value: object, frames: int) -> None:
    expected = {
        "observability_result",
        "predicted_class",
        "confidence",
        "intensity",
        "evidence_frames",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{sample_id}/{actor}: malformed affect decision")
    result = value.get("observability_result")
    evidence = value.get("evidence_frames")
    if (
        result not in AFFECT_RESULTS
        or not isinstance(evidence, list)
        or not evidence
        or any(
            isinstance(frame, bool) or not isinstance(frame, int) or not 0 <= frame < frames
            for frame in evidence
        )
        or len(set(evidence)) != len(evidence)
    ):
        raise ValueError(f"{sample_id}/{actor}: invalid affect evidence")
    predicted = value.get("predicted_class")
    confidence = value.get("confidence")
    intensity = value.get("intensity")
    if result == "observable":
        if (
            predicted not in AFFECT_CLASSES
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
            or intensity not in INTENSITIES
        ):
            raise ValueError(f"{sample_id}/{actor}: invalid observable affect")
    elif any(value is not None for value in (predicted, confidence, intensity)):
        raise ValueError(f"{sample_id}/{actor}: non-observable affect carries a pseudo-label")


def _validate_affect_decision(sample_id: str, decision: dict[str, Any], frames: int) -> None:
    expected = {
        "full_video_reviewed",
        "robot_a",
        "robot_b",
        "interaction_affect_relation_en",
        "review_notes",
    }
    if set(decision) != expected or decision.get("full_video_reviewed") is not True:
        raise ValueError(f"{sample_id}: malformed affect decision")
    _validate_actor_affect(sample_id, "robot_a", decision.get("robot_a"), frames)
    _validate_actor_affect(sample_id, "robot_b", decision.get("robot_b"), frames)
    if not _text(decision.get("interaction_affect_relation_en"), optional=True):
        raise ValueError(f"{sample_id}: invalid affect relation text")
    if not _text(decision.get("review_notes")):
        raise ValueError(f"{sample_id}: affect review notes are required")


def finalize(
    *,
    shard_summary_path: Path,
    decisions_path: Path,
    reviewer_id: str,
    output_submission_path: Path,
    output_summary_path: Path,
) -> dict[str, Any]:
    shard_summary_path = shard_summary_path.resolve()
    decisions_path = decisions_path.resolve()
    output_submission_path = output_submission_path.resolve()
    output_summary_path = output_summary_path.resolve()
    if output_submission_path.exists() or output_summary_path.exists():
        raise FileExistsError("Final review outputs already exist")
    if not reviewer_id.strip():
        raise ValueError("reviewer_id is required")

    shard = read_json(shard_summary_path)
    queue_kind = shard.get("queue_kind")
    queue_path = Path(str(shard.get("review_queue") or "")).resolve()
    if (
        shard.get("artifact_kind") != SHARD_KIND
        or queue_kind not in {"arc_action", "affect"}
        or shard.get("accepted_for_training") is not False
        or shard.get("fixed_duration_window_used") is not False
        or not queue_path.is_file()
        or queue_path.parent != shard_summary_path.parent
        or sha256_file(queue_path) != shard.get("review_queue_sha256")
    ):
        raise ValueError("Shard summary violates its fail-closed queue binding")
    queue_rows = read_jsonl(queue_path)
    sample_ids = [row.get("sample_id") for row in queue_rows]
    queue = {sample_id: row for sample_id, row in zip(sample_ids, queue_rows)}
    if (
        any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids)
        or len(queue) != len(queue_rows)
    ):
        raise ValueError("Shard queue contains invalid or duplicate sample IDs")
    decisions = read_json(decisions_path)
    if set(decisions) != set(queue):
        raise ValueError("Decision coverage does not exactly match the shard queue")

    rows: list[dict[str, Any]] = []
    result_counts: Counter[str] = Counter()
    actor_result_counts: Counter[str] = Counter()
    for sample_id in sorted(queue):
        queued = queue[sample_id]
        if (
            queued.get("accepted_for_training") is not False
            or queued.get("fixed_duration_window_used") is not False
            or queued.get("native_variable_length") is not True
        ):
            raise ValueError(f"{sample_id}: source queue is not fail closed")
        video = Path(str(queued.get("video_path") or "")).resolve()
        if not video.is_file() or sha256_file(video) != queued.get("video_sha256"):
            raise ValueError(f"{sample_id}: anonymous review video SHA mismatch")
        video_info = probe_video(video)
        decision = decisions[sample_id]
        if not isinstance(decision, dict):
            raise ValueError(f"{sample_id}: decision must be an object")
        if queue_kind == "arc_action":
            _validate_arc_decision(sample_id, decision, video_info["frames"])
            result_counts[decision["expression_completeness_result"]] += 1
        else:
            _validate_affect_decision(sample_id, decision, video_info["frames"])
            for actor in ("robot_a", "robot_b"):
                actor_result_counts[decision[actor]["observability_result"]] += 1
        row = dict(queued)
        row.update(decision)
        row.update(
            {
                "artifact_kind": SUBMISSION_KIND,
                "review_id": f"{queue_kind}-{reviewer_id}-{sample_id}",
                "reviewer_id": reviewer_id,
                "decode_complete": True,
                "decoded_frame_count": video_info["frames"],
                "blind_review_provenance": {
                    "shard_summary_sha256": sha256_file(shard_summary_path),
                    "shard_queue_sha256": shard["review_queue_sha256"],
                    "source_public_summary_sha256": shard["source_public_summary_sha256"],
                    "source_queue_sha256": shard["source_queue_sha256"],
                    "queue_record_hash_method": "sha256_stable_json_utf8",
                    "queue_record_sha256": value_sha256(queued),
                },
            }
        )
        rows.append(row)

    atomic_write(output_submission_path, "".join(stable_json(row) + "\n" for row in rows))
    summary = {
        "artifact_kind": SUMMARY_KIND,
        "schema_version": "3.0.0",
        "queue_kind": queue_kind,
        "shard_index": shard["shard_index"],
        "shard_count": shard["shard_count"],
        "reviewer_id": reviewer_id,
        "records": len(rows),
        "submission": str(output_submission_path),
        "submission_sha256": sha256_file(output_submission_path),
        "shard_summary": str(shard_summary_path),
        "shard_summary_sha256": sha256_file(shard_summary_path),
        "shard_queue_sha256": shard["review_queue_sha256"],
        "source_public_summary_sha256": shard["source_public_summary_sha256"],
        "source_queue_sha256": shard["source_queue_sha256"],
        "decisions": str(decisions_path),
        "decisions_sha256": sha256_file(decisions_path),
        "coverage_complete_without_overlap": True,
        "all_videos_hash_verified_and_probed_to_eof": True,
        "expression_completeness_distribution": dict(sorted(result_counts.items())),
        "actor_affect_observability_distribution": dict(sorted(actor_result_counts.items())),
        "fixed_duration_window_used": False,
        "accepted_for_training": False,
    }
    atomic_write(output_summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-summary", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--output-submission", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = finalize(
        shard_summary_path=args.shard_summary,
        decisions_path=args.decisions,
        reviewer_id=args.reviewer_id,
        output_submission_path=args.output_submission,
        output_summary_path=args.output_summary,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
