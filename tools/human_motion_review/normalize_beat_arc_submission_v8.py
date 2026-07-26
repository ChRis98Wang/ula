#!/usr/bin/env python3
"""Normalize BEAT2 arc-review enums and bindings without changing judgments."""

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
STATUS_MAP = {
    "observed": "complete",
    "not_observed_needs_extension": "incomplete",
    "complete": "complete",
    "incomplete": "incomplete",
    "ambiguous": "ambiguous",
}


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
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected object at {path}:{line_number}")
        rows.append(value)
    return rows


def normalize_status(value: Any) -> str:
    try:
        return STATUS_MAP[value]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Unknown arc phase status: {value}") from error


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


def normalize(
    *,
    public_summary_path: Path,
    source_submission_path: Path,
    source_summary_path: Path,
    output_submission_path: Path,
    output_summary_path: Path,
) -> dict[str, Any]:
    paths = [
        public_summary_path,
        source_submission_path,
        source_summary_path,
        output_submission_path,
        output_summary_path,
    ]
    public_summary_path, source_submission_path, source_summary_path, output_submission_path, output_summary_path = (
        path.resolve() for path in paths
    )
    if output_submission_path.exists() or output_summary_path.exists():
        raise FileExistsError("Normalization outputs already exist")

    public = read_json(public_summary_path)
    queue_path = Path(public["arc_action_queue"]).resolve()
    queue_sha256 = sha256_file(queue_path)
    public_sha256 = sha256_file(public_summary_path)
    if (
        public.get("arc_action_queue_sha256") != queue_sha256
        or public.get("accepted_for_training") is not False
        or public.get("fixed_duration_window_used") is not False
        or public.get("all_samples_native_variable_length") is not True
    ):
        raise ValueError("Public BEAT2 arc bundle violates its binding contract")

    source_summary = read_json(source_summary_path)
    source_submission_sha256 = sha256_file(source_submission_path)
    if (
        source_summary.get("submission_sha256") != source_submission_sha256
        or source_summary.get("public_summary_sha256") != public_sha256
        or source_summary.get("public_arc_action_queue_sha256") != queue_sha256
        or source_summary.get("all_videos_decoded_from_frame_zero_to_eof") is not True
        or source_summary.get("fixed_duration_window_used") is not False
        or source_summary.get("training") is not False
    ):
        raise ValueError("Source BEAT2 arc summary binding mismatch")
    queue_rows = read_jsonl(queue_path)
    source_rows = read_jsonl(source_submission_path)
    queue = {row["sample_id"]: row for row in queue_rows}
    source = {row["sample_id"]: row for row in source_rows}
    if len(queue) != len(queue_rows) or len(source) != len(source_rows) or set(queue) != set(source):
        raise ValueError("Source BEAT2 arc submission coverage mismatch")

    normalized_rows: list[dict[str, Any]] = []
    for sample_id in sorted(queue):
        queued = queue[sample_id]
        row = source[sample_id]
        frame_count = int(queued["frame_count"])
        if (
            row.get("video_path") != queued.get("video_path")
            or row.get("video_sha256") != queued.get("video_sha256")
            or row.get("context_level") != queued.get("context_level")
            or row.get("frame_count") != frame_count
            or row.get("decoded_frame_count") != frame_count
            or row.get("video_sha256_verified") is not True
            or row.get("frame_count_verified") is not True
            or row.get("fps_verified") is not True
            or row.get("action_result") != "pass"
            or row.get("action_observability") != "observable"
            or row.get("training") is not False
            or row.get("emotion_judgment") is not False
        ):
            raise ValueError(f"Source BEAT2 arc row binding mismatch: {sample_id}")
        video = Path(row["video_path"]).resolve()
        if not video.is_file() or sha256_file(video) != row["video_sha256"]:
            raise ValueError(f"Source BEAT2 arc video SHA mismatch: {sample_id}")
        phases = {}
        for phase in ("onset", "apex", "offset"):
            status = normalize_status(row.get(f"{phase}_status"))
            frame = row.get(f"{phase}_evidence_frame")
            basis = row.get(f"{phase}_basis")
            if (
                isinstance(frame, bool)
                or not isinstance(frame, int)
                or not 0 <= frame < frame_count
                or not isinstance(basis, str)
                or not basis.strip()
            ):
                raise ValueError(f"Invalid {phase} evidence: {sample_id}")
            phases.update(
                {
                    f"{phase}_status": status,
                    f"{phase}_evidence_frame": frame,
                    f"{phase}_basis": basis,
                }
            )
        expected_arc_result = (
            "complete" if all(phases[f"{phase}_status"] == "complete" for phase in ("onset", "apex", "offset")) else "needs_extension"
        )
        if row.get("arc_result") != expected_arc_result:
            raise ValueError(f"Arc-result/phase disagreement: {sample_id}")
        normalized_rows.append(
            {
                "schema_version": "1.0.0",
                "sample_id": sample_id,
                "video_path": row["video_path"],
                "video_sha256": row["video_sha256"],
                "context_level": row["context_level"],
                "frame_count": frame_count,
                "decoded_frame_count": frame_count,
                "fps": queued["fps"],
                "queue_sha256": queue_sha256,
                "arc_protocol_version": "robot_expression_arc_blind_video_v1",
                "arc_review_id": row["arc_review_id"],
                "arc_reviewer_id": row["arc_reviewer_id"],
                "action_protocol_version": "robot_action_semantics_blind_video_v1",
                "action_review_id": row["action_review_id"],
                "action_reviewer_id": row["action_reviewer_id"],
                "action_result": "pass",
                "action_observability": "observable",
                "observable_description": row["observable_description"],
                "candidate_text": None,
                "candidate_text_sha256": None,
                "candidate_text_provenance": None,
                "full_decode_to_eof": True,
                "ordered_full_video_review_performed": True,
                "audio_available": False,
                "label_metadata_exposed": False,
                "emotion_judgment_performed": False,
                "native_duration_preserved": True,
                "fixed_duration_window_used": False,
                "elapsed_duration_used_as_gate": False,
                "training_admission": False,
                **phases,
            }
        )

    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in normalized_rows
    )
    _atomic_write(output_submission_path, payload)
    summary = {
        "artifact_kind": SUMMARY_KIND,
        "schema_version": "1.0.0",
        "reviewer_id": source_summary.get("reviewer_id"),
        "records": len(normalized_rows),
        "submission_path": str(output_submission_path),
        "submission_sha256": sha256_file(output_submission_path),
        "public_summary_path": str(public_summary_path),
        "public_summary_sha256": public_sha256,
        "public_queue_path": str(queue_path),
        "public_queue_sha256": queue_sha256,
        "action_result_distribution": {"pass": len(normalized_rows)},
        "action_observability_distribution": {"observable": len(normalized_rows)},
        "onset_status_distribution": dict(sorted(Counter(row["onset_status"] for row in normalized_rows).items())),
        "apex_status_distribution": dict(sorted(Counter(row["apex_status"] for row in normalized_rows).items())),
        "offset_status_distribution": dict(sorted(Counter(row["offset_status"] for row in normalized_rows).items())),
        "decoded_frame_count_total": sum(row["decoded_frame_count"] for row in normalized_rows),
        "native_variable_length_reviewed": True,
        "fixed_duration_window_used": False,
        "elapsed_duration_used_as_gate": False,
        "validation_passed": True,
        "normalization": {
            "judgments_changed": False,
            "source_submission_sha256": source_submission_sha256,
            "source_summary_sha256": sha256_file(source_summary_path),
            "phase_enum_mapping": STATUS_MAP,
        },
    }
    _atomic_write(output_summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-summary", type=Path, required=True)
    parser.add_argument("--source-submission", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--output-submission", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = normalize(
        public_summary_path=args.public_summary,
        source_submission_path=args.source_submission,
        source_summary_path=args.source_summary,
        output_submission_path=args.output_submission,
        output_summary_path=args.output_summary,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
