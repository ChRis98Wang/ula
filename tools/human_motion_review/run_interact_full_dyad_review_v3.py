#!/usr/bin/env python3
"""Resumably render full-span InterAct 2x2 blind-review evidence."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import traceback
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from tools.human_motion_review.build_interact_full_dyad_review_manifest_v3 import (
    DEFAULT_OUTPUT_ROOT as DEFAULT_MANIFEST_ROOT,
    atomic_json,
    load_json,
    load_jsonl,
    sha256_file,
    value_sha256,
)
from tools.human_motion_review.render_beat2_annotation_review import validate_video
from tools.human_motion_review.render_interact_full_dyad_evidence_v3 import (
    HEIGHT,
    LINEAGE_CONTRACT,
    WIDTH,
    implementation_binding as renderer_implementation_binding,
    render_record,
)
from upper_body_skeleton.v2_axis_calibration import DEFAULT_URDF


DEFAULT_MANIFEST_SUMMARY = DEFAULT_MANIFEST_ROOT / "summary.json"
DEFAULT_PILOT_SELECTION = DEFAULT_MANIFEST_ROOT / "interact_full_dyad_pilot8_v3.jsonl"
DEFAULT_OUTPUT_ROOT = Path(
    "/home/gez/nas/cloud/gez/human_motion/review/"
    "interact_full_dyad_review_v3/pilot8_staging"
)
STATE_NAME = "interact_full_dyad_review_v3.run_state.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-summary", type=Path, default=DEFAULT_MANIFEST_SUMMARY)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--selection-manifest", type=Path, default=DEFAULT_PILOT_SELECTION)
    selection.add_argument("--all", action="store_true", help="Render all manifest dyads")
    parser.add_argument(
        "--confirm-full-count",
        type=int,
        help="Required with --all; must equal the hash-bound full manifest count",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--smoothing-window", type=int, default=7)
    return parser.parse_args(argv)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _evidence_id(record: dict[str, Any]) -> str:
    value = f"interact-full-evidence-v3\0{record['dyad_id']}\0{record['dyad_record_sha256']}"
    return "stg_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _paths(output_root: Path, evidence_id: str) -> dict[str, Path]:
    return {
        "video": output_root / "videos" / f"{evidence_id}.mp4",
        "lineage": output_root / "lineage" / f"{evidence_id}.json",
        "summary": output_root / "summaries" / f"{evidence_id}.json",
    }


def _validate_lineage(lineage: dict[str, Any], summary: dict[str, Any]) -> None:
    if lineage.get("artifact_kind") != "interact_full_dyad_explicit_frame_lineage_v3":
        raise ValueError("Evidence lineage has the wrong artifact kind")
    if lineage.get("lineage_contract") != LINEAGE_CONTRACT:
        raise ValueError("Evidence lineage has the wrong mapping contract")
    if lineage.get("evidence_frames") != summary.get("frames"):
        raise ValueError("Evidence lineage/video frame counts differ")
    evidence_frames = int(lineage["evidence_frames"])
    columns = {
        "evidence_frame": (0, evidence_frames - 1),
        "source_local_frame": (0, int(lineage["source_frames"]) - 1),
        "robot_a_safe_frame": (0, int(lineage["robot_a_output_frames"]) - 1),
        "robot_b_safe_frame": (0, int(lineage["robot_b_output_frames"]) - 1),
    }
    for name, endpoints in columns.items():
        values = lineage.get(name)
        if not isinstance(values, list) or len(values) != evidence_frames:
            raise ValueError(f"Evidence lineage column is incomplete: {name}")
        if values[0] != endpoints[0] or values[-1] != endpoints[1]:
            raise ValueError(f"Evidence lineage does not preserve endpoints: {name}")
        if any(right < left for left, right in zip(values, values[1:])):
            raise ValueError(f"Evidence lineage is not monotonic: {name}")
    if (
        lineage.get("all_source_and_output_endpoints_included") is not True
        or lineage.get("source_or_output_frames_cropped") is not False
        or lineage.get("fixed_duration_target_used") is not False
    ):
        raise ValueError("Evidence lineage violates the full-span contract")


def validate_result(
    record: dict[str, Any], output_root: Path, result: dict[str, Any] | None = None
) -> dict[str, Any]:
    evidence_id = _evidence_id(record)
    paths = _paths(output_root, evidence_id)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = load_json(paths["summary"])
    lineage = load_json(paths["lineage"])
    if summary.get("artifact_kind") != "interact_full_dyad_2x2_review_evidence_v3":
        raise ValueError("Evidence summary has the wrong artifact kind")
    if summary.get("implementation_binding") != renderer_implementation_binding():
        raise ValueError("Evidence renderer implementation binding changed")
    if summary.get("dyad_id") != record["dyad_id"] or summary.get(
        "dyad_record_sha256"
    ) != record["dyad_record_sha256"]:
        raise ValueError("Evidence summary is not bound to the selected dyad")
    if Path(summary["output_mp4"]).resolve() != paths["video"].resolve():
        raise ValueError("Evidence summary video path mismatch")
    video_hash = sha256_file(paths["video"])
    lineage_hash = sha256_file(paths["lineage"])
    summary_hash = sha256_file(paths["summary"])
    if summary.get("output_mp4_sha256") != video_hash or summary.get(
        "frame_lineage_json_sha256"
    ) != lineage_hash:
        raise ValueError("Evidence summary artifact SHA mismatch")
    if lineage.get("dyad_id") != record["dyad_id"] or lineage.get(
        "dyad_record_sha256"
    ) != record["dyad_record_sha256"]:
        raise ValueError("Frame lineage is not bound to the selected dyad")
    _validate_lineage(lineage, summary)
    video_validation = validate_video(
        paths["video"],
        expected_frames=int(summary["frames"]),
        expected_width=WIDTH,
        expected_height=HEIGHT,
        expected_fps=float(summary["fps"]),
    )
    if result is not None:
        expected_hashes = {
            "video_sha256": video_hash,
            "lineage_sha256": lineage_hash,
            "summary_sha256": summary_hash,
        }
        if any(result.get(key) != value for key, value in expected_hashes.items()):
            raise ValueError("Resume state evidence SHA mismatch")
    normalized = {
        "evidence_id": evidence_id,
        "dyad_id": record["dyad_id"],
        "dyad_record_sha256": record["dyad_record_sha256"],
        "status": "rendered_pending_blind_review",
        "video": str(paths["video"].resolve()),
        "video_sha256": video_hash,
        "lineage": str(paths["lineage"].resolve()),
        "lineage_sha256": lineage_hash,
        "summary": str(paths["summary"].resolve()),
        "summary_sha256": summary_hash,
        "frames": int(summary["frames"]),
        "source_frames": int(lineage["source_frames"]),
        "robot_a_output_frames": int(lineage["robot_a_output_frames"]),
        "robot_b_output_frames": int(lineage["robot_b_output_frames"]),
        "any_safety_retimed": bool(record["any_safety_retimed"]),
        "video_validation": video_validation,
        "semantic_supervision_mask": False,
        "emotion_supervision_mask": False,
        "license_training_mask": False,
        "accepted_for_training": False,
    }
    normalized["evidence_result_record_sha256"] = value_sha256(normalized)
    if result is not None and result.get("evidence_result_record_sha256") != normalized[
        "evidence_result_record_sha256"
    ]:
        raise ValueError("Resume state evidence result record SHA mismatch")
    return normalized


def _render_worker(payload: dict[str, Any]) -> dict[str, Any]:
    record = payload["record"]
    output_root = Path(payload["output_root"])
    evidence_id = _evidence_id(record)
    paths = _paths(output_root, evidence_id)
    render_record(
        record,
        paths["video"],
        paths["lineage"],
        paths["summary"],
        urdf=Path(payload["urdf"]),
        smoothing_window=int(payload["smoothing_window"]),
    )
    return validate_result(record, output_root)


def _failure(record: dict[str, Any], error: BaseException) -> dict[str, Any]:
    return {
        "evidence_id": _evidence_id(record),
        "dyad_id": record["dyad_id"],
        "dyad_record_sha256": record["dyad_record_sha256"],
        "status": "render_failed",
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc()[-8000:],
        "semantic_supervision_mask": False,
        "emotion_supervision_mask": False,
        "license_training_mask": False,
        "accepted_for_training": False,
    }


def _load_selection(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    summary_path = args.manifest_summary.resolve()
    manifest_summary = load_json(summary_path)
    if manifest_summary.get("artifact_kind") != "interact_full_dyad_initial_review_manifest_summary_v3":
        raise ValueError("Wrong InterAct full-dyad manifest summary kind")
    if manifest_summary.get("accepted_for_training") is not False:
        raise ValueError("Manifest summary unexpectedly admits training")
    manifest_path = Path(manifest_summary["manifest"]).resolve()
    if sha256_file(manifest_path) != manifest_summary.get("manifest_sha256"):
        raise ValueError("Full dyad manifest SHA mismatch")
    all_rows = load_jsonl(manifest_path)
    if len(all_rows) != manifest_summary.get("dyad_count"):
        raise ValueError("Full dyad manifest count mismatch")
    by_id = {row["dyad_id"]: row for row in all_rows}
    if len(by_id) != len(all_rows):
        raise ValueError("Full dyad manifest IDs are duplicated")
    if args.all:
        if args.confirm_full_count != len(all_rows):
            raise ValueError("--all requires --confirm-full-count equal to the bound full count")
        selected = all_rows
        selection_binding = {
            "selection_kind": "all_hash_bound_full_manifest_records",
            "selection_manifest": None,
            "selection_manifest_sha256": None,
            "selected_count": len(selected),
        }
    else:
        if args.confirm_full_count is not None:
            raise ValueError("--confirm-full-count is only valid with --all")
        selection_path = args.selection_manifest.resolve()
        selectors = load_jsonl(selection_path)
        if selection_path == Path(manifest_summary["pilot_manifest"]).resolve() and sha256_file(
            selection_path
        ) != manifest_summary.get("pilot_manifest_sha256"):
            raise ValueError("Pilot selection manifest SHA mismatch")
        selected = []
        seen = set()
        for selector in selectors:
            dyad_id = selector.get("dyad_id")
            record = by_id.get(dyad_id)
            if record is None or selector.get("dyad_record_sha256") != record.get(
                "dyad_record_sha256"
            ):
                raise ValueError(f"Selection does not bind a current dyad record: {dyad_id}")
            if dyad_id in seen:
                raise ValueError(f"Duplicate selected dyad: {dyad_id}")
            seen.add(dyad_id)
            selected.append(record)
        selection_binding = {
            "selection_kind": "explicit_hash_bound_selection_manifest",
            "selection_manifest": str(selection_path),
            "selection_manifest_sha256": sha256_file(selection_path),
            "selected_count": len(selected),
        }
    return manifest_summary, selected, {
        "manifest_summary": str(summary_path),
        "manifest_summary_sha256": sha256_file(summary_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        **selection_binding,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.smoothing_window < 1:
        raise ValueError("smoothing-window must be positive")
    manifest_summary, selected, input_binding = _load_selection(args)
    if not selected:
        raise ValueError("InterAct evidence selection is empty")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    os.chmod(output_root, 0o700)
    for name in ("videos", "lineage", "summaries"):
        (output_root / name).mkdir(parents=True, exist_ok=True)
        os.chmod(output_root / name, 0o700)
    state_path = output_root / STATE_NAME
    runtime = {
        "urdf": str(args.urdf.resolve()),
        "urdf_sha256": sha256_file(args.urdf.resolve()),
        "smoothing_window": args.smoothing_window,
        "workers": args.workers,
    }
    implementation = {
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        **renderer_implementation_binding(),
    }
    prior = load_json(state_path) if args.resume and state_path.is_file() else None
    if prior is not None:
        if prior.get("input_binding") != input_binding:
            raise ValueError("Resume state input binding differs")
        if (prior.get("runtime") or {}).get("urdf_sha256") != runtime["urdf_sha256"] or (
            prior.get("runtime") or {}
        ).get("smoothing_window") != runtime["smoothing_window"]:
            raise ValueError("Resume state rendering contract differs")
        if prior.get("implementation_binding") != implementation:
            raise ValueError("Resume state implementation binding differs")
        state = prior
        state["status"] = "running"
        state["runtime"] = runtime
    else:
        state = {
            "schema_version": "3.0.0",
            "artifact_kind": "interact_full_dyad_review_v3_run_state",
            "created_at_utc": _utc_now(),
            "status": "running",
            "input_binding": input_binding,
            "runtime": runtime,
            "implementation_binding": implementation,
            "selected_count": len(selected),
            "duration_policy": manifest_summary["duration_policy"],
            "duration_threshold_or_fixed_window_used": False,
            "evidence_layout": "2x2_source_dyad_xz_yz_plus_mujoco_robot_a_b",
            "accepted_for_training": False,
            "results": {},
        }

    pending = []
    for record in selected:
        previous = (state.get("results") or {}).get(record["dyad_id"])
        if args.resume and previous and previous.get("status") == "rendered_pending_blind_review":
            state["results"][record["dyad_id"]] = validate_result(
                record, output_root, previous
            )
        else:
            pending.append(record)
    atomic_json(state_path, state)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _render_worker,
                {
                    "record": record,
                    "output_root": str(output_root),
                    "urdf": runtime["urdf"],
                    "smoothing_window": args.smoothing_window,
                },
            ): record
            for record in pending
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            record = futures[future]
            try:
                result = future.result()
            except BaseException as error:
                result = _failure(record, error)
            state["results"][record["dyad_id"]] = result
            counts = Counter(row.get("status", "unknown") for row in state["results"].values())
            state["status_counts"] = dict(sorted(counts.items()))
            state["updated_at_utc"] = _utc_now()
            atomic_json(state_path, state)
            print(
                f"[{completed:04d}/{len(pending):04d}] {record['dyad_id']}: {result['status']}",
                flush=True,
            )

    counts = Counter(row.get("status", "unknown") for row in state["results"].values())
    state["status_counts"] = dict(sorted(counts.items()))
    state["rendered_count"] = counts["rendered_pending_blind_review"]
    state["failure_count"] = sum(count for status, count in counts.items() if status != "rendered_pending_blind_review")
    state["status"] = (
        "complete_pending_blind_review"
        if state["rendered_count"] == len(selected) and state["failure_count"] == 0
        else "complete_with_failures"
    )
    state["completed_at_utc"] = _utc_now()
    atomic_json(state_path, state)
    return state


def main(argv: list[str] | None = None) -> None:
    state = run(parse_args(argv))
    print(
        json.dumps(
            {
                "status": state["status"],
                "selected_count": state["selected_count"],
                "status_counts": state["status_counts"],
                "accepted_for_training": state["accepted_for_training"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
