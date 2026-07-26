#!/usr/bin/env python3
"""Build anonymous arc/action and affect queues for InterAct context expansions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
from typing import Any

try:
    from tools.human_motion_review import build_interact_blind_review_bundle as v1
    from tools.human_motion_review import build_interact_blind_review_bundle_v2 as v2
    from tools.human_motion_review.build_interact_arc_expansion_plan_v2 import (
        read_json,
        read_jsonl,
        sha256_file,
    )
except ModuleNotFoundError:  # pragma: no cover - direct invocation by path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.human_motion_review import build_interact_blind_review_bundle as v1
    from tools.human_motion_review import build_interact_blind_review_bundle_v2 as v2
    from tools.human_motion_review.build_interact_arc_expansion_plan_v2 import (
        read_json,
        read_jsonl,
        sha256_file,
    )


RUN_KIND = "interact_dyadic_natural_context_expansion_review_v2_run_state"
PLAN_KIND = "interact_dyadic_arc_action_one_level_expansion_plan_v2"
PUBLIC_ENTRY_WHITELIST = {
    "videos",
    "arc_action_review_queue.jsonl",
    "affect_review_queue.jsonl",
    "summary.json",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-summary", type=Path, required=True)
    parser.add_argument("--expansion-requests", type=Path, required=True)
    parser.add_argument("--run-state", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--hidden-root", type=Path, required=True)
    parser.add_argument("--secret-file", type=Path, required=True)
    return parser.parse_args(argv)


def _index(rows: list[dict[str, Any]], *, context: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in result:
            raise ValueError(f"{context} contains invalid or duplicate sample_id")
        result[sample_id] = row
    return result


def _load_secret(path: Path) -> bytes:
    value = read_json(path.resolve())
    secret_hex = value.get("secret_hex")
    if not isinstance(secret_hex, str):
        raise ValueError("InterAct bundle secret file is invalid")
    secret = bytes.fromhex(secret_hex)
    if len(secret) < 16:
        raise ValueError("InterAct bundle secret is too short")
    return secret


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _validate_existing_public_tree(public_root: Path) -> None:
    if public_root.is_symlink():
        raise ValueError(f"Public bundle root must not be a symlink: {public_root}")
    if not public_root.exists():
        return
    if not public_root.is_dir():
        raise ValueError(f"Public bundle root must be a directory: {public_root}")
    unexpected = sorted(
        entry.name
        for entry in public_root.iterdir()
        if entry.name not in PUBLIC_ENTRY_WHITELIST
    )
    if unexpected:
        raise ValueError(f"stale unexpected public bundle entries: {unexpected}")
    for name in PUBLIC_ENTRY_WHITELIST.difference({"videos"}):
        path = public_root / name
        if path.exists() and (
            path.is_symlink() or not stat.S_ISREG(path.stat().st_mode)
        ):
            raise ValueError(f"Public bundle artifact is not a regular file: {path}")
    videos_root = public_root / "videos"
    if videos_root.exists() and (videos_root.is_symlink() or not videos_root.is_dir()):
        raise ValueError(f"Public videos path must be a real directory: {videos_root}")


def _validate_public_bundle(
    public_root: Path,
    *,
    artifact_hashes: dict[str, str],
    video_hashes: dict[str, str],
    video_sources: dict[str, Path],
) -> None:
    if public_root.is_symlink() or not public_root.is_dir():
        raise ValueError(f"Public bundle root must be a real directory: {public_root}")
    actual_entries = {entry.name for entry in public_root.iterdir()}
    unexpected = sorted(actual_entries.difference(PUBLIC_ENTRY_WHITELIST))
    if unexpected:
        raise ValueError(f"stale unexpected public bundle entries: {unexpected}")
    missing = sorted(PUBLIC_ENTRY_WHITELIST.difference(actual_entries))
    if missing:
        raise ValueError(f"Public bundle is missing required entries: {missing}")
    if set(artifact_hashes) != PUBLIC_ENTRY_WHITELIST.difference({"videos"}):
        raise ValueError("Public bundle artifact hash whitelist is incomplete")
    for name, expected_hash in artifact_hashes.items():
        path = public_root / name
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError(f"Public bundle artifact is not a regular file: {path}")
        if sha256_file(path) != expected_hash:
            raise ValueError(f"Public bundle artifact SHA mismatch: {path}")

    videos_root = public_root / "videos"
    if videos_root.is_symlink() or not videos_root.is_dir():
        raise ValueError(f"Public videos path must be a real directory: {videos_root}")
    actual_videos = {entry.name for entry in videos_root.iterdir()}
    if actual_videos != set(video_hashes) or set(video_sources) != set(video_hashes):
        raise ValueError("Public video set does not match the exact whitelist")
    for name, expected_hash in video_hashes.items():
        path = videos_root / name
        path_stat = path.lstat()
        if (
            not stat.S_ISREG(path_stat.st_mode)
            or path_stat.st_nlink != 1
            or os.path.samefile(video_sources[name], path)
            or sha256_file(path) != expected_hash
        ):
            raise ValueError(f"Public video isolation or integrity failure: {path}")


def validate_state(
    state: dict[str, Any],
    *,
    plan_summary: Path,
    expansion_requests: Path,
) -> None:
    if state.get("artifact_kind") != RUN_KIND:
        raise ValueError("Unexpected InterAct expansion run-state kind")
    if state.get("status") != "complete_pending_repeat_blind_review":
        raise ValueError("InterAct expansion run is not complete")
    if state.get("failure_count") != 0:
        raise ValueError("InterAct expansion run contains failures")
    if state.get("accepted_for_training") is not False:
        raise ValueError("InterAct expansion run unexpectedly admits training")
    inputs = state.get("inputs") or {}
    if inputs.get("plan_summary_sha256") != sha256_file(plan_summary):
        raise ValueError("InterAct expansion run plan SHA mismatch")
    if inputs.get("expansion_requests_sha256") != sha256_file(expansion_requests):
        raise ValueError("InterAct expansion run request SHA mismatch")


def build_bundle(
    *,
    plan_summary: Path,
    expansion_requests: Path,
    run_state: Path,
    output_root: Path,
    hidden_root: Path,
    secret_file: Path,
) -> dict[str, Any]:
    plan_summary = plan_summary.resolve()
    expansion_requests = expansion_requests.resolve()
    run_state = run_state.resolve()
    plan = read_json(plan_summary)
    if plan.get("artifact_kind") != PLAN_KIND or plan.get("accepted_for_training_count") != 0:
        raise ValueError("Unexpected or training-admitted InterAct expansion plan")
    declared = plan.get("outputs", {}).get("expansion_requests") or {}
    if declared.get("sha256") != sha256_file(expansion_requests):
        raise ValueError("InterAct expansion request SHA mismatch")
    requests = _index(read_jsonl(expansion_requests), context="expansion requests")
    if declared.get("records") != len(requests):
        raise ValueError("InterAct expansion request count mismatch")
    state = read_json(run_state)
    validate_state(state, plan_summary=plan_summary, expansion_requests=expansion_requests)
    results = state.get("results") or {}
    if len(results) != len(requests):
        raise ValueError("InterAct expansion render result count mismatch")

    output_root = output_root.resolve()
    public_root = output_root / "public"
    hidden_root = hidden_root.resolve()
    if _paths_overlap(public_root, hidden_root):
        raise ValueError("Public and hidden InterAct bundle roots must be disjoint")
    _validate_existing_public_tree(public_root)
    hidden_root.mkdir(parents=True, exist_ok=True)
    os.chmod(hidden_root, 0o700)
    secret = _load_secret(secret_file)
    arc_queue = []
    affect_queue = []
    hidden_rows = []
    video_hashes: dict[str, str] = {}
    video_sources: dict[str, Path] = {}
    for base_sample_id, request in sorted(requests.items()):
        matching = [row for row in results.values() if row.get("sample_id") == base_sample_id]
        if len(matching) != 1:
            raise ValueError(f"InterAct expansion result binding mismatch: {base_sample_id}")
        result = matching[0]
        interval = request["requested_interval"]
        expected_interval = [interval["start_frame"], interval["end_frame_exclusive"]]
        if (
            result.get("status") != "rendered_pending_repeat_blind_review"
            or result.get("source_interval") != expected_interval
            or result.get("requested_context_level") != request["requested_context_level"]
            or result.get("fixed_duration_window_used") is not False
            or result.get("accepted_for_training") is not False
        ):
            raise ValueError(f"InterAct expansion result changed plan: {base_sample_id}")
        video = Path(result["video"]).resolve()
        video_hash = result["video_sha256"]
        if sha256_file(video) != video_hash:
            raise ValueError(f"InterAct expansion video SHA mismatch: {base_sample_id}")
        validation = result.get("video_validation") or {}
        if (
            validation.get("passed") is not True
            or validation.get("decoded_frames") != interval["frame_count"]
        ):
            raise ValueError(f"InterAct expansion video validation failed: {base_sample_id}")
        sample_id = v1.anonymous_id(
            secret,
            "dyadexpv2",
            f"{base_sample_id}:context:{request['requested_context_level']}",
            video_hash,
        )
        anonymous_video = public_root / "videos" / f"{sample_id}.mp4"
        if anonymous_video.name in video_hashes:
            raise ValueError(f"Duplicate anonymous expansion video: {anonymous_video.name}")
        video_hashes[anonymous_video.name] = video_hash
        video_sources[anonymous_video.name] = video
        v1.materialize_video(video, anonymous_video, video_hash)
        arc = v2._arc_action_record(sample_id, anonymous_video, video_hash)
        affect = v2._affect_record(sample_id, anonymous_video, video_hash)
        arc["context_level"] = request["requested_context_level"]
        affect["context_level"] = request["requested_context_level"]
        v1.assert_public_privacy(arc)
        v1.assert_public_privacy(affect)
        arc_queue.append(arc)
        affect_queue.append(affect)
        hidden_rows.append(
            {
                "sample_id": sample_id,
                "base_sample_id": base_sample_id,
                "turn_id": request["turn_id"],
                "reviewed_context_level": request["reviewed_context_level"],
                "displayed_context_level": request["requested_context_level"],
                "displayed_interval": interval,
                "plan_record_sha256": request["plan_record_sha256"],
                "video_sha256": video_hash,
                "native_duration_preserved": True,
                "official_scenario_or_emotion_exposed": False,
                "accepted_for_training": False,
            }
        )

    arc_queue.sort(key=lambda row: row["sample_id"])
    affect_queue.sort(key=lambda row: row["sample_id"])
    hidden_rows.sort(key=lambda row: row["sample_id"])
    paths = {
        "arc": public_root / "arc_action_review_queue.jsonl",
        "affect": public_root / "affect_review_queue.jsonl",
        "hidden": hidden_root / "sample_mapping.jsonl",
    }
    v1.atomic_jsonl(paths["arc"], arc_queue)
    v1.atomic_jsonl(paths["affect"], affect_queue)
    v1.atomic_jsonl(paths["hidden"], hidden_rows, mode=0o600)
    public_summary = {
        "schema_version": "2.0.0",
        "artifact_kind": "interact_dyadic_natural_context_expansion_anonymous_bundle_v2",
        "arc_action_records": len(arc_queue),
        "affect_records": len(affect_queue),
        "arc_action_queue": str(paths["arc"]),
        "arc_action_queue_sha256": sha256_file(paths["arc"]),
        "affect_queue": str(paths["affect"]),
        "affect_queue_sha256": sha256_file(paths["affect"]),
        "plan_summary_sha256": sha256_file(plan_summary),
        "run_state_sha256": sha256_file(run_state),
        "duration_policy": "one_predeclared_shared_rest_boundary_level_no_fixed_window",
        "fixed_duration_window_used": False,
        "identity_scenario_official_text_or_emotion_exposed": False,
        "accepted_for_training": False,
    }
    v1.assert_public_privacy(public_summary)
    hidden_summary = {
        "schema_version": "2.0.0",
        "artifact_kind": "interact_dyadic_natural_context_expansion_hidden_mapping_v2",
        "public_summary": str(public_root / "summary.json"),
        "plan_summary": str(plan_summary),
        "plan_summary_sha256": sha256_file(plan_summary),
        "expansion_requests": str(expansion_requests),
        "expansion_requests_sha256": sha256_file(expansion_requests),
        "run_state": str(run_state),
        "run_state_sha256": sha256_file(run_state),
        "sample_mapping": str(paths["hidden"]),
        "sample_mapping_sha256": sha256_file(paths["hidden"]),
        "accepted_for_training": False,
    }
    public_summary_path = public_root / "summary.json"
    v1.atomic_json(public_summary_path, public_summary)
    v1.atomic_json(hidden_root / "summary.json", hidden_summary, mode=0o600)
    _validate_public_bundle(
        public_root,
        artifact_hashes={
            paths["arc"].name: public_summary["arc_action_queue_sha256"],
            paths["affect"].name: public_summary["affect_queue_sha256"],
            public_summary_path.name: sha256_file(public_summary_path),
        },
        video_hashes=video_hashes,
        video_sources=video_sources,
    )
    return {"public": public_summary, "hidden": hidden_summary}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_bundle(
        plan_summary=args.plan_summary,
        expansion_requests=args.expansion_requests,
        run_state=args.run_state,
        output_root=args.output_root,
        hidden_root=args.hidden_root,
        secret_file=args.secret_file,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
