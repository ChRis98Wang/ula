#!/usr/bin/env python3
"""Render one-level InterAct dyadic natural-context expansions for blind review."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.gmr_v2.interact_bvh_adapter import INTERACT_NATIVE_AXIS_POLICY
from tools.human_motion_review.build_interact_arc_expansion_plan_v2 import (
    read_json,
    read_jsonl,
    sha256_file,
    value_sha256,
)
from tools.human_motion_review.render_beat2_annotation_review import validate_video
from tools.human_motion_review.run_interact_dyadic_review_v2 import atomic_json


RENDERER = PROJECT_ROOT / "tools/human_motion_review/render_interact_dyadic_review.py"
STATE_NAME = "interact_dyadic_expansion_review_v2.run_state.json"
PLAN_KIND = "interact_dyadic_arc_action_one_level_expansion_plan_v2"
REQUEST_KIND = "interact_dyadic_one_level_natural_context_expansion_request_v2"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-summary", type=Path, required=True)
    parser.add_argument("--expansion-requests", type=Path, required=True)
    parser.add_argument("--hidden-mapping", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _index(rows: list[dict[str, Any]], key: str, *, context: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value or value in result:
            raise ValueError(f"{context} contains invalid or duplicate {key}")
        result[value] = row
    return result


def _verify_request(row: dict[str, Any]) -> None:
    expected = row.get("plan_record_sha256")
    value = dict(row)
    value.pop("plan_record_sha256", None)
    if expected != value_sha256(value):
        raise ValueError(f"Expansion request record SHA mismatch: {row.get('sample_id')}")
    if row.get("artifact_kind") != REQUEST_KIND:
        raise ValueError("Unexpected InterAct expansion request kind")
    if (
        row.get("accepted_for_training") is not False
        or row.get("semantic_supervision_mask") is not False
        or row.get("emotion_supervision_mask") is not False
        or row.get("elapsed_duration_used_as_gate") is not False
        or row.get("expansion_unit")
        != "exactly_one_next_predeclared_shared_rest_boundary_level"
    ):
        raise ValueError("InterAct expansion request violates fail-closed contract")
    if row.get("requested_context_level") != row.get("reviewed_context_level", -2) + 1:
        raise ValueError("InterAct expansion request skips a context level")
    current = row.get("reviewed_interval") or {}
    requested = row.get("requested_interval") or {}
    for interval in (current, requested):
        if (
            not isinstance(interval.get("start_frame"), int)
            or not isinstance(interval.get("end_frame_exclusive"), int)
            or interval.get("frame_count")
            != interval.get("end_frame_exclusive") - interval.get("start_frame")
        ):
            raise ValueError("InterAct expansion request has an invalid interval")
    if not (
        requested["start_frame"] <= current["start_frame"]
        and requested["end_frame_exclusive"] >= current["end_frame_exclusive"]
        and requested != current
    ):
        raise ValueError("InterAct requested interval is not a strict natural-context expansion")


def load_groups(
    *,
    plan_summary: Path,
    expansion_requests: Path,
    hidden_mapping: Path,
    receipt: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan_summary = plan_summary.resolve()
    expansion_requests = expansion_requests.resolve()
    hidden_mapping = hidden_mapping.resolve()
    receipt = receipt.resolve()
    plan = read_json(plan_summary)
    if plan.get("artifact_kind") != PLAN_KIND:
        raise ValueError("Unexpected InterAct expansion plan kind")
    if plan.get("accepted_for_training_count") != 0:
        raise ValueError("Expansion plan unexpectedly admits training")
    declared = plan.get("outputs", {}).get("expansion_requests") or {}
    if declared.get("sha256") != sha256_file(expansion_requests):
        raise ValueError("Expansion request manifest SHA mismatch")
    if plan.get("inputs", {}).get("hidden_mapping_sha256") != sha256_file(hidden_mapping):
        raise ValueError("Expansion plan hidden mapping SHA mismatch")
    requests = read_jsonl(expansion_requests)
    if declared.get("records") != len(requests):
        raise ValueError("Expansion request count mismatch")
    for row in requests:
        _verify_request(row)
    requests_by_sample = _index(requests, "sample_id", context="expansion requests")
    hidden = _index(read_jsonl(hidden_mapping), "sample_id", context="hidden mapping")
    receipt_value = read_json(receipt)
    selected = receipt_value.get("selected") or []
    tasks = _index(selected, "episode_task_id", context="receipt selected tasks")

    groups = []
    for sample_id, request in sorted(requests_by_sample.items()):
        mapping = hidden.get(sample_id)
        if mapping is None or mapping.get("turn_id") != request.get("turn_id"):
            raise ValueError(f"Expansion request hidden mapping mismatch: {sample_id}")
        actor_mapping = mapping.get("actor_mapping") or {}
        if set(actor_mapping) != {"A", "B"}:
            raise ValueError(f"Expansion request has incomplete actor mapping: {sample_id}")
        actor_rows = []
        for role in ("A", "B"):
            episode_task_id = actor_mapping[role].get("episode_task_id")
            task = tasks.get(episode_task_id)
            if task is None:
                raise ValueError(f"Receipt lacks expansion actor task: {episode_task_id}")
            source = Path(task["source_bvh"]).resolve()
            if sha256_file(source) != task.get("source_bvh_sha256"):
                raise ValueError(f"Expansion source BVH SHA mismatch: {episode_task_id}")
            actor_rows.append((role, task, source))
        interval = request["requested_interval"]
        expansion_task_id = (
            f"{sample_id}_ctx{request['requested_context_level']:02d}_"
            f"f{interval['start_frame']:06d}-{interval['end_frame_exclusive']:06d}"
        )
        groups.append(
            {
                "expansion_task_id": expansion_task_id,
                "sample_id": sample_id,
                "turn_id": request["turn_id"],
                "reviewed_context_level": request["reviewed_context_level"],
                "requested_context_level": request["requested_context_level"],
                "source_interval": [interval["start_frame"], interval["end_frame_exclusive"]],
                "expected_frame_count": interval["frame_count"],
                "actor_a_bvh": str(actor_rows[0][2]),
                "actor_a_bvh_sha256": actor_rows[0][1]["source_bvh_sha256"],
                "actor_b_bvh": str(actor_rows[1][2]),
                "actor_b_bvh_sha256": actor_rows[1][1]["source_bvh_sha256"],
                "plan_record_sha256": request["plan_record_sha256"],
                "fixed_duration_window_used": False,
                "accepted_for_training": False,
            }
        )
    binding = {
        "plan_summary": str(plan_summary),
        "plan_summary_sha256": sha256_file(plan_summary),
        "expansion_requests": str(expansion_requests),
        "expansion_requests_sha256": sha256_file(expansion_requests),
        "hidden_mapping": str(hidden_mapping),
        "hidden_mapping_sha256": sha256_file(hidden_mapping),
        "receipt": str(receipt),
        "receipt_sha256": sha256_file(receipt),
    }
    return groups, binding


def validate_result(summary_path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    summary = read_json(summary_path)
    if summary.get("axis_policy") != INTERACT_NATIVE_AXIS_POLICY:
        raise ValueError("Dyadic expansion render uses the wrong InterAct axis policy")
    if summary.get("source_interval") != expected["source_interval"]:
        raise ValueError("Dyadic expansion render changed the requested natural boundary")
    if summary.get("frames") != expected["expected_frame_count"]:
        raise ValueError("Dyadic expansion render changed the requested frame count")
    for role in ("a", "b"):
        if summary.get(f"actor_{role}_bvh_sha256") != expected[f"actor_{role}_bvh_sha256"]:
            raise ValueError(f"Dyadic expansion actor {role.upper()} source SHA mismatch")
    if summary.get("official_scenario_or_emotion_rendered") is not False:
        raise ValueError("Dyadic expansion render exposed hidden metadata")
    video = Path(summary["output_mp4"]).resolve()
    check = validate_video(
        video,
        expected_frames=expected["expected_frame_count"],
        expected_width=1280,
        expected_height=720,
        expected_fps=30.0,
    )
    return {
        "expansion_task_id": expected["expansion_task_id"],
        "sample_id": expected["sample_id"],
        "turn_id": expected["turn_id"],
        "status": "rendered_pending_repeat_blind_review",
        "reviewed_context_level": expected["reviewed_context_level"],
        "requested_context_level": expected["requested_context_level"],
        "source_interval": expected["source_interval"],
        "native_duration_preserved": True,
        "fixed_duration_window_used": False,
        "summary_json": str(summary_path.resolve()),
        "summary_json_sha256": sha256_file(summary_path),
        "video": str(video),
        "video_sha256": sha256_file(video),
        "video_validation": check,
        "semantic_supervision_mask": False,
        "emotion_supervision_mask": False,
        "accepted_for_training": False,
    }


def render_one(group: dict[str, Any], output_root: Path, resume: bool) -> dict[str, Any]:
    stem = group["expansion_task_id"]
    output_mp4 = output_root / f"{stem}_dyad.mp4"
    summary_json = output_root / f"{stem}_dyad_summary.json"
    if resume and output_mp4.is_file() and summary_json.is_file():
        return validate_result(summary_json, group)
    subprocess.run(
        [
            sys.executable,
            str(RENDERER),
            "--actor-a-bvh",
            group["actor_a_bvh"],
            "--actor-b-bvh",
            group["actor_b_bvh"],
            "--start-frame",
            str(group["source_interval"][0]),
            "--end-frame",
            str(group["source_interval"][1]),
            "--output-mp4",
            str(output_mp4),
            "--summary-json",
            str(summary_json),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return validate_result(summary_json, group)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    groups, binding = load_groups(
        plan_summary=args.plan_summary,
        expansion_requests=args.expansion_requests,
        hidden_mapping=args.hidden_mapping,
        receipt=args.receipt,
    )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / STATE_NAME
    state: dict[str, Any] = {
        "schema_version": "2.0.0",
        "artifact_kind": "interact_dyadic_natural_context_expansion_review_v2_run_state",
        "status": "running",
        "inputs": binding,
        "axis_policy": INTERACT_NATIVE_AXIS_POLICY,
        "duration_policy": "one_predeclared_shared_rest_boundary_level_no_fixed_window",
        "renderer_sha256": sha256_file(RENDERER),
        "task_count": len(groups),
        "results": {},
        "semantic_supervision_mask": False,
        "emotion_supervision_mask": False,
        "accepted_for_training": False,
    }
    atomic_json(state_path, state)
    lock = threading.Lock()
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(render_one, group, output_root, args.resume): group for group in groups
        }
        for future in as_completed(futures):
            group = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                failures += 1
                result = {
                    "expansion_task_id": group["expansion_task_id"],
                    "sample_id": group["sample_id"],
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "accepted_for_training": False,
                }
            with lock:
                state["results"][group["expansion_task_id"]] = result
                atomic_json(state_path, state)
            print(
                f"[{len(state['results']):02d}/{len(groups):02d}] "
                f"{group['expansion_task_id']}: {result['status']}",
                flush=True,
            )
    state["status"] = "failed" if failures else "complete_pending_repeat_blind_review"
    state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    state["failure_count"] = failures
    atomic_json(state_path, state)
    print(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
