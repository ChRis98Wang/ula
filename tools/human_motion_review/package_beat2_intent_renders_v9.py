#!/usr/bin/env python3
"""Bind verified BEAT2 v9 render results into the generic blind-review package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


SCHEMA_VERSION = "1.0.0"
PACKAGE_CONTRACT = "beat2_observable_intent_render_package_v9"
JOINT_COUNT = 18


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_sha256(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    atomic_text(path, "".join(stable_json(record) + "\n" for record in records))


def trajectory_motion_check(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)
    if len(header) != JOINT_COUNT or len(rows) < 2:
        raise ValueError(f"invalid 18D trajectory: {path}")
    values = np.asarray(rows, dtype=np.float64)
    if values.shape != (len(rows), JOINT_COUNT) or not np.isfinite(values).all():
        raise ValueError(f"invalid numeric 18D trajectory: {path}")
    peak_to_peak = np.ptp(values, axis=0)
    moving_joint_count = int(np.count_nonzero(peak_to_peak >= np.radians(0.5)))
    has_motion = moving_joint_count >= 1
    return {
        "has_motion": has_motion,
        "moving_joint_count_0p5deg": moving_joint_count,
        "max_joint_excursion_deg": round(float(np.degrees(np.max(peak_to_peak))), 6),
        "frames": len(values),
    }


def build_package(
    *,
    review_queue: Path,
    passed_manifest: Path,
    output_manifest: Path,
    output_summary: Path,
) -> dict[str, Any]:
    queue = read_jsonl(review_queue)
    passed = read_jsonl(passed_manifest)
    queue_by_id = {str(item.get("task_id") or ""): item for item in queue}
    passed_by_id = {str(item.get("task_id") or ""): item for item in passed}
    if len(queue_by_id) != len(queue) or "" in queue_by_id:
        raise ValueError("review queue has missing or duplicate task_id values")
    if len(passed_by_id) != len(passed) or "" in passed_by_id:
        raise ValueError("passed render manifest has missing or duplicate task_id values")
    if set(queue_by_id) != set(passed_by_id):
        missing = sorted(set(queue_by_id) - set(passed_by_id))
        extra = sorted(set(passed_by_id) - set(queue_by_id))
        raise ValueError(f"render coverage mismatch: missing={missing[:3]}, extra={extra[:3]}")

    records: list[dict[str, Any]] = []
    total_frames = 0
    total_seconds = 0.0
    for task_id in sorted(queue_by_id):
        source = queue_by_id[task_id]
        render = passed_by_id[task_id]
        if source.get("accepted_for_training") is not False:
            raise ValueError(f"{task_id}: intent candidate must remain excluded")
        if source.get("manual_video_review_required") is not True:
            raise ValueError(f"{task_id}: intent candidate must require video review")
        if source.get("quality_gate", {}).get("passed") is not True:
            raise ValueError(f"{task_id}: source physical QC is not passed")
        if render.get("status") != "passed":
            raise ValueError(f"{task_id}: render status is not passed")
        trajectory = Path(str(source.get("trajectory_path") or "")).resolve()
        trajectory_digest = str(source.get("trajectory_sha256") or "")
        if not trajectory.is_file() or file_sha256(trajectory) != trajectory_digest:
            raise ValueError(f"{task_id}: trajectory binding mismatch")
        final_binding = render.get("final_output_binding")
        if not isinstance(final_binding, Mapping):
            raise ValueError(f"{task_id}: render final_output_binding is missing")
        if final_binding.get("trajectory_sha256") != trajectory_digest:
            raise ValueError(f"{task_id}: rendered trajectory SHA256 mismatch")
        video = Path(str(render.get("video_path") or "")).resolve()
        video_digest = str(render.get("video_sha256") or "")
        if not video.is_file() or file_sha256(video) != video_digest:
            raise ValueError(f"{task_id}: rendered video SHA256 mismatch")
        video_check = render.get("video_check")
        if not isinstance(video_check, Mapping) or video_check.get("passed") is not True:
            raise ValueError(f"{task_id}: video verification is incomplete")
        if video_check.get("nonblank") is not True or video_check.get("audio_streams") != 0:
            raise ValueError(f"{task_id}: video is blank or contains audio")
        render_summary = Path(str(render.get("render_summary_path") or "")).resolve()
        if not render_summary.is_file() or file_sha256(render_summary) != render.get(
            "render_summary_sha256"
        ):
            raise ValueError(f"{task_id}: render summary binding mismatch")
        render_metadata = json.loads(render_summary.read_text(encoding="utf-8"))
        framing = render_metadata.get("camera_framing")
        full_frame = isinstance(framing, Mapping) and framing.get("mode") == "auto_full_body"
        motion_check = trajectory_motion_check(trajectory)
        if motion_check["has_motion"] is not True:
            raise ValueError(f"{task_id}: trajectory has no measurable motion")
        frames = int(video_check["decoded_frames"])
        fps = float(video_check["fps"])
        if frames != int(source["frames"]) or not np.isclose(fps, float(source["fps"])):
            raise ValueError(f"{task_id}: rendered time axis mismatch")
        total_frames += frames
        total_seconds += max(0, frames - 1) / fps
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_kind": PACKAGE_CONTRACT,
                "clip_id": task_id,
                "accepted_for_training": False,
                "manual_video_review_required": True,
                "training_exclusion_reason": "observable intent blind review is incomplete",
                "physical_qc": {
                    "passed": True,
                    "source_quality_gate": source["quality_gate"],
                    "trajectory_motion_check": motion_check,
                },
                "trajectory": {
                    "path": str(trajectory),
                    "sha256": trajectory_digest,
                    "fps": fps,
                    "frames": frames,
                    "native_variable_length": True,
                },
                "motion_realization": source.get("motion_realization"),
                "motion_style": source.get("motion_style"),
                "candidate_routes": source.get("candidate_routes"),
                "render": {
                    "video": video.name,
                    "video_sha256": video_digest,
                    "summary": str(render_summary),
                    "summary_sha256": render["render_summary_sha256"],
                    "checks": {
                        "nonblank": True,
                        "has_motion": True,
                        "full_frame_uncropped": full_frame,
                        "fully_decodable": True,
                        "silent": True,
                        "frame_count_match": True,
                    },
                },
                "source_queue_record_sha256": record_sha256(source),
                "render_result_record_sha256": record_sha256(render),
            }
        )
    write_jsonl(output_manifest, records)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": PACKAGE_CONTRACT,
        "review_queue": str(review_queue.resolve()),
        "review_queue_sha256": file_sha256(review_queue),
        "passed_render_manifest": str(passed_manifest.resolve()),
        "passed_render_manifest_sha256": file_sha256(passed_manifest),
        "package_manifest": str(output_manifest.resolve()),
        "package_manifest_sha256": file_sha256(output_manifest),
        "source_records": len(queue),
        "packaged_records": len(records),
        "total_frames": total_frames,
        "total_sample_span_sec": round(total_seconds, 6),
        "accepted_for_training": 0,
        "anonymous_review_still_required": True,
    }
    atomic_text(output_summary, json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument("--passed-render-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_package(
        review_queue=args.review_queue,
        passed_manifest=args.passed_render_manifest,
        output_manifest=args.output_manifest,
        output_summary=args.output_summary,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
