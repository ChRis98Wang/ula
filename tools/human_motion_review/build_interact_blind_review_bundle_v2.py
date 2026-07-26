#!/usr/bin/env python3
"""Build the native-BVH InterAct v2 anonymous blind-review bundle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

try:
    from tools.gmr_v2.interact_bvh_adapter import INTERACT_NATIVE_AXIS_POLICY
    from tools.human_motion_review import build_interact_blind_review_bundle as v1
    from tools.human_motion_review.render_beat2_annotation_review import validate_video
except ModuleNotFoundError:  # pragma: no cover - direct invocation by path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.gmr_v2.interact_bvh_adapter import INTERACT_NATIVE_AXIS_POLICY
    from tools.human_motion_review import build_interact_blind_review_bundle as v1
    from tools.human_motion_review.render_beat2_annotation_review import validate_video


SCHEMA_VERSION = "2.0.0"
AXIS_PROTOCOL = "interact_robot_axis_blind_video_native_bvh_v2"
ARC_ACTION_PROTOCOL = "interact_dyadic_arc_action_blind_video_native_bvh_v2"
AFFECT_PROTOCOL = "interact_dyadic_affect_blind_video_native_bvh_v2"
DURATION_POLICY = (
    "cataloged_complete_natural_interaction_boundary;"
    "duration_follows_observable_semantic_arc;no_fixed_window"
)
DEFAULT_RECEIPT = Path(
    "/home/gez/nas/cloud/gez/human_motion/catalog/"
    "interact_axis_smoke_four_performance_v1_receipt.json"
)
DEFAULT_AXIS_STATE = Path(
    "/home/gez/nas/cloud/gez/human_motion/review/"
    "interact_18d_axis_smoke_v16_native_bvh_camera_corrected_review_v2/"
    "interact_native_bvh_axis_smoke_v2.run_state.json"
)
DEFAULT_DYAD_STATE = Path(
    "/home/gez/nas/cloud/gez/human_motion/review/"
    "interact_blind_expression_v2/staging/"
    "interact_native_bvh_dyads_v2.run_state.json"
)
DEFAULT_OUTPUT = Path(
    "/home/gez/nas/cloud/gez/human_motion/review/interact_blind_expression_v2"
)
DEFAULT_HIDDEN = Path(
    "/home/gez/shuaiwang/.private_human_motion/interact_blind_expression_v2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--axis-run-state", type=Path, default=DEFAULT_AXIS_STATE)
    parser.add_argument("--dyad-run-state", type=Path, default=DEFAULT_DYAD_STATE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hidden-root", type=Path, default=DEFAULT_HIDDEN)
    parser.add_argument("--secret-hex")
    return parser.parse_args()


def _axis_record(sample_id: str, video: Path, video_hash: str) -> dict[str, Any]:
    record = v1.axis_record(sample_id, video, video_hash)
    record.update(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": AXIS_PROTOCOL,
            "temporal_unit": "complete_natural_interaction_arc",
            "fixed_duration_window_used": False,
            "native_duration_preserved": True,
            "accepted_for_training": False,
        }
    )
    return record


def _arc_action_record(
    sample_id: str, video: Path, video_hash: str
) -> dict[str, Any]:
    record = v1.arc_action_record(sample_id, video, video_hash)
    record.update(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": ARC_ACTION_PROTOCOL,
            "temporal_unit": "complete_natural_interaction_arc",
            "fixed_duration_window_used": False,
            "native_duration_preserved": True,
            "accepted_for_training": False,
        }
    )
    return record


def _affect_record(sample_id: str, video: Path, video_hash: str) -> dict[str, Any]:
    record = v1.affect_record(sample_id, video, video_hash)
    record.update(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": AFFECT_PROTOCOL,
            "temporal_unit": "complete_natural_interaction_arc",
            "fixed_duration_window_used": False,
            "native_duration_preserved": True,
            "accepted_for_training": False,
        }
    )
    return record


def _assert_complete_state(
    state: dict[str, Any], *, artifact_kind: str, receipt_hash: str
) -> None:
    if state.get("artifact_kind") != artifact_kind:
        raise ValueError(f"Unexpected run-state kind: {state.get('artifact_kind')}")
    if state.get("status") != "complete_pending_blind_review":
        raise ValueError(f"Run state is not complete: {state.get('status')}")
    if state.get("failure_count") != 0:
        raise ValueError("Run state contains rendering failures")
    if state.get("axis_policy") != INTERACT_NATIVE_AXIS_POLICY:
        raise ValueError("Run state does not use the native InterAct BVH policy")
    state_receipt_hash = state.get("input_receipt_sha256", state.get("receipt_sha256"))
    if state_receipt_hash != receipt_hash:
        raise ValueError("Run state and catalog receipt hashes differ")


def _same_interval(axis_interval: dict[str, Any], receipt_interval: dict[str, Any]) -> bool:
    keys = ("start_frame", "end_frame_exclusive", "frame_count")
    return all(axis_interval.get(key) == receipt_interval.get(key) for key in keys)


def main() -> None:
    args = parse_args()
    receipt_path = args.receipt.resolve()
    axis_state_path = args.axis_run_state.resolve()
    dyad_state_path = args.dyad_run_state.resolve()
    receipt = v1.load_json(receipt_path)
    selected = receipt.get("selected") or []
    if len(selected) != 8 or receipt.get("accepted_for_training") is not False:
        raise ValueError("InterAct v2 bundle requires the fail-closed eight-task receipt")

    receipt_hash = v1.sha256_file(receipt_path)
    axis_state = v1.load_json(axis_state_path)
    dyad_state = v1.load_json(dyad_state_path)
    _assert_complete_state(
        axis_state,
        artifact_kind="interact_native_bvh_axis_smoke_v2_run_state",
        receipt_hash=receipt_hash,
    )
    if axis_state.get("public_review_video_identity_metadata_exposed") is not False:
        raise ValueError("Native InterAct axis evidence is not anonymous")
    if (
        axis_state.get("front_camera_projection_corrected") is not True
        or axis_state.get("robot_front_camera_screen_right_axis") != "+Y"
    ):
        raise ValueError("Native InterAct axis evidence uses a mirrored front camera")
    _assert_complete_state(
        dyad_state,
        artifact_kind="interact_native_bvh_dyadic_review_v2_run_state",
        receipt_hash=receipt_hash,
    )
    if len(axis_state.get("results") or {}) != 8:
        raise ValueError("Native InterAct axis run state must contain eight results")
    if len(dyad_state.get("results") or {}) != 4:
        raise ValueError("Native InterAct dyad run state must contain four results")

    output_root = args.output_root.resolve()
    public_root = output_root / "public"
    hidden_root = args.hidden_root.resolve()
    hidden_root.mkdir(parents=True, exist_ok=True)
    os.chmod(hidden_root, 0o700)
    secret = v1.bundle_secret(hidden_root, args.secret_hex)
    task_manifest = Path(receipt["catalog_task_manifest"]).resolve()
    tasks = v1.load_tasks(task_manifest)

    axis_queue: list[dict[str, Any]] = []
    axis_hidden: list[dict[str, Any]] = []
    for row in sorted(selected, key=lambda item: item["episode_task_id"]):
        task_id = row["episode_task_id"]
        result = (axis_state.get("results") or {}).get(task_id)
        if result is None or result.get("status") != "rendered_pending_blind_review":
            raise ValueError(f"Missing completed native axis result: {task_id}")
        if not _same_interval(result.get("source_interval") or {}, row["source_interval"]):
            raise ValueError(f"Native axis result changed natural boundary: {task_id}")
        artifact = result["artifacts"]["comparison_mp4"]
        source_video = Path(artifact["path"]).resolve()
        video_hash = artifact["sha256"]
        if v1.sha256_file(source_video) != video_hash:
            raise ValueError(f"Native axis video SHA mismatch: {task_id}")
        validation = result.get("video_validation", {}).get("comparison", {})
        if not validation.get("passed"):
            raise ValueError(f"Native axis video validation failed: {task_id}")
        axis_summary = v1.load_json(
            Path(result["artifacts"]["axis_summary"]["path"]).resolve()
        )
        if (
            axis_summary.get("public_frame_labels_anonymous") is not True
            or axis_summary.get("identity_or_partner_metadata_drawn") is not False
        ):
            raise ValueError(f"Native axis video exposes identity metadata: {task_id}")
        if (
            axis_summary.get("robot_front_camera_screen_right_axis") != "+Y"
            or axis_summary.get("source_projection")
            != "episode_aligned_dual_view_front_plus_y_z_and_side_plus_x_z"
            or axis_summary.get("robot_side_label_positions")
            != {"screen_left": "ROBOT LEFT", "screen_right": "ROBOT RIGHT"}
        ):
            raise ValueError(f"Native axis video uses a mirrored camera contract: {task_id}")
        sample_id = v1.anonymous_id(secret, "axisv2", task_id, video_hash)
        anonymous_video = public_root / "videos" / f"{sample_id}.mp4"
        v1.materialize_video(source_video, anonymous_video, video_hash)
        public = _axis_record(sample_id, anonymous_video, video_hash)
        v1.assert_public_privacy(public)
        axis_queue.append(public)
        axis_hidden.append(
            {
                "sample_id": sample_id,
                "episode_task_id": task_id,
                "performance_id": row["performance_id"],
                "turn_id": row["turn_id"],
                "target_actor_id": row["target_actor_id"],
                "partner_actor_id": row["partner_actor_id"],
                "video_sha256": video_hash,
                "automated_quality_passed": result["automated_quality_passed"],
                "failed_automated_gates": result["failed_automated_gates"],
                "native_duration_preserved": result["native_duration_preserved"],
                "accepted_for_training": False,
            }
        )

    by_turn: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        by_turn.setdefault(row["turn_id"], []).append(row)
    if len(by_turn) != 4 or any(len(rows) != 2 for rows in by_turn.values()):
        raise ValueError("Native dyadic bundle requires four complete partner turns")

    arc_queue: list[dict[str, Any]] = []
    affect_queue: list[dict[str, Any]] = []
    dyad_hidden: list[dict[str, Any]] = []
    for turn_id, rows in sorted(by_turn.items()):
        result = (dyad_state.get("results") or {}).get(turn_id)
        if result is None or result.get("status") != "rendered_pending_blind_review":
            raise ValueError(f"Missing completed native dyad result: {turn_id}")
        expected_interval = [
            rows[0]["source_interval"]["start_frame"],
            rows[0]["source_interval"]["end_frame_exclusive"],
        ]
        if result.get("source_interval") != expected_interval:
            raise ValueError(f"Native dyad result changed natural boundary: {turn_id}")
        summary_path = Path(result["summary_json"]).resolve()
        summary = v1.load_json(summary_path)
        if summary.get("axis_policy") != INTERACT_NATIVE_AXIS_POLICY:
            raise ValueError(f"Native dyad policy mismatch: {turn_id}")
        source_video = Path(result["video"]).resolve()
        video_hash = result["video_sha256"]
        if summary.get("output_mp4_sha256") != video_hash:
            raise ValueError(f"Native dyad summary SHA mismatch: {turn_id}")
        if v1.sha256_file(source_video) != video_hash:
            raise ValueError(f"Native dyad video SHA mismatch: {turn_id}")
        check = validate_video(
            source_video,
            expected_frames=int(summary["frames"]),
            expected_width=1280,
            expected_height=720,
            expected_fps=30.0,
        )
        sample_id = v1.anonymous_id(secret, "dyadv2", turn_id, video_hash)
        anonymous_video = public_root / "videos" / f"{sample_id}.mp4"
        v1.materialize_video(source_video, anonymous_video, video_hash)
        arc_public = _arc_action_record(sample_id, anonymous_video, video_hash)
        affect_public = _affect_record(sample_id, anonymous_video, video_hash)
        v1.assert_public_privacy(arc_public)
        v1.assert_public_privacy(affect_public)
        arc_queue.append(arc_public)
        affect_queue.append(affect_public)

        hashes_to_role = {
            summary["actor_a_bvh_sha256"]: "A",
            summary["actor_b_bvh_sha256"]: "B",
        }
        actor_mapping: dict[str, dict[str, Any]] = {}
        for row in rows:
            role = hashes_to_role.get(row["source_bvh_sha256"])
            if role is None or role in actor_mapping:
                raise ValueError(f"Cannot bind anonymous native dyad roles: {turn_id}")
            actor_mapping[role] = {
                "episode_task_id": row["episode_task_id"],
                "actor_id": row["target_actor_id"],
                "partner_actor_id": row["partner_actor_id"],
            }
        dyad_hidden.append(
            {
                "sample_id": sample_id,
                "performance_id": rows[0]["performance_id"],
                "turn_id": turn_id,
                "actor_mapping": actor_mapping,
                "displayed_context_level": 0,
                "context_plan": tasks[rows[0]["episode_task_id"]]["context_plan"],
                "dyad_summary": str(summary_path),
                "dyad_summary_sha256": v1.sha256_file(summary_path),
                "video_validation": check,
                "native_duration_preserved": True,
                "official_scenario_or_emotion_exposed": False,
                "accepted_for_training": False,
            }
        )

    axis_queue.sort(key=lambda item: item["sample_id"])
    arc_queue.sort(key=lambda item: item["sample_id"])
    affect_queue.sort(key=lambda item: item["sample_id"])
    axis_hidden.sort(key=lambda item: item["sample_id"])
    dyad_hidden.sort(key=lambda item: item["sample_id"])
    paths = {
        "axis": public_root / "axis_review_queue.jsonl",
        "arc_action": public_root / "arc_action_review_queue.jsonl",
        "affect": public_root / "affect_review_queue.jsonl",
        "axis_hidden": hidden_root / "axis_mapping.jsonl",
        "dyad_hidden": hidden_root / "dyad_mapping.jsonl",
    }
    v1.atomic_jsonl(paths["axis"], axis_queue)
    v1.atomic_jsonl(paths["arc_action"], arc_queue)
    v1.atomic_jsonl(paths["affect"], affect_queue)
    v1.atomic_jsonl(paths["axis_hidden"], axis_hidden, mode=0o600)
    v1.atomic_jsonl(paths["dyad_hidden"], dyad_hidden, mode=0o600)

    public_summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "interact_native_bvh_separate_anonymous_blind_review_bundle_v2",
        "axis_records": len(axis_queue),
        "dyadic_arc_action_records": len(arc_queue),
        "dyadic_affect_records": len(affect_queue),
        "axis_queue": str(paths["axis"]),
        "axis_queue_sha256": v1.sha256_file(paths["axis"]),
        "arc_action_queue": str(paths["arc_action"]),
        "arc_action_queue_sha256": v1.sha256_file(paths["arc_action"]),
        "affect_queue": str(paths["affect"]),
        "affect_queue_sha256": v1.sha256_file(paths["affect"]),
        "native_bvh_axis_policy": INTERACT_NATIVE_AXIS_POLICY,
        "axis_run_state_sha256": v1.sha256_file(axis_state_path),
        "dyad_run_state_sha256": v1.sha256_file(dyad_state_path),
        "duration_policy": DURATION_POLICY,
        "fixed_duration_window_used": False,
        "identity_scenario_official_text_or_emotion_exposed": False,
        "axis_arc_action_and_affect_reviewers_must_be_independent": True,
        "incomplete_arc_action_requires_next_predeclared_natural_context_level": True,
        "accepted_for_training": False,
    }
    v1.assert_public_privacy(public_summary)
    hidden_summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "interact_native_bvh_hidden_blind_mapping_v2",
        "receipt": str(receipt_path),
        "receipt_sha256": receipt_hash,
        "task_manifest": str(task_manifest),
        "task_manifest_sha256": v1.sha256_file(task_manifest),
        "axis_run_state": str(axis_state_path),
        "axis_run_state_sha256": v1.sha256_file(axis_state_path),
        "dyad_run_state": str(dyad_state_path),
        "dyad_run_state_sha256": v1.sha256_file(dyad_state_path),
        "axis_mapping": str(paths["axis_hidden"]),
        "axis_mapping_sha256": v1.sha256_file(paths["axis_hidden"]),
        "dyad_mapping": str(paths["dyad_hidden"]),
        "dyad_mapping_sha256": v1.sha256_file(paths["dyad_hidden"]),
        "duration_policy": DURATION_POLICY,
        "public_distribution_forbidden": True,
        "accepted_for_training": False,
    }
    v1.atomic_json(public_root / "summary.json", public_summary)
    v1.atomic_json(hidden_root / "summary.json", hidden_summary, mode=0o600)
    v1.enforce_hidden_permissions(hidden_root)
    print(json.dumps({"public": public_summary, "hidden": hidden_summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
