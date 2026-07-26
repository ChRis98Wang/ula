#!/usr/bin/env python3
"""Build an anonymous fail-closed bundle from full-dyad 2x2 evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
from typing import Any

from tools.human_motion_review import build_interact_blind_review_bundle as shared
from tools.human_motion_review.build_interact_full_dyad_review_manifest_v3 import (
    load_json,
    load_jsonl,
    sha256_file,
    value_sha256,
)
from tools.human_motion_review.run_interact_full_dyad_review_v3 import (
    STATE_NAME,
    _validate_lineage,
    validate_result,
)


DEFAULT_STAGING = Path(
    "/home/gez/nas/cloud/gez/human_motion/review/"
    "interact_full_dyad_review_v3/pilot8_staging"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/gez/nas/cloud/gez/human_motion/review/"
    "interact_full_dyad_review_v3/pilot8_blind_bundle/public"
)
DEFAULT_HIDDEN_ROOT = Path(
    "/home/gez/shuaiwang/.private_human_motion/"
    "interact_full_dyad_review_v3/pilot8_blind_bundle"
)
ARC_PROTOCOL = "interact_full_dyad_arc_action_blind_2x2_v3"
AFFECT_PROTOCOL = "interact_full_dyad_affect_blind_2x2_v3"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--run-state", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--hidden-root", type=Path, default=DEFAULT_HIDDEN_ROOT)
    parser.add_argument("--secret-hex")
    return parser.parse_args(argv)


def _public_common(sample_id: str, video: Path, video_hash: str) -> dict[str, Any]:
    return {
        "schema_version": "3.0.0",
        "sample_id": sample_id,
        "video_path": str(video.resolve()),
        "video_sha256": video_hash,
        "video_layout": "2x2_human_dyad_xz_yz_plus_mujoco_robot_a_b",
        "natural_context_level": 0,
        "temporal_unit": "predeclared_natural_interaction_context",
        "native_variable_length": True,
        "fixed_duration_window_used": False,
        "inside_context_crop_used": False,
        "audio_available": False,
        "face_geometry_available": False,
        "finger_geometry_available": False,
        "label_metadata_exposed": False,
        "identity_scene_text_or_official_affect_exposed": False,
        "semantic_supervision_mask": False,
        "emotion_supervision_mask": False,
        "license_training_mask": False,
        "accepted_for_training": False,
    }


def arc_record(sample_id: str, video: Path, video_hash: str) -> dict[str, Any]:
    return {
        **_public_common(sample_id, video, video_hash),
        "protocol_version": ARC_PROTOCOL,
        "review_id": None,
        "reviewer_id": None,
        "decode_complete": None,
        "decoded_frame_count": None,
        "onset_status": None,
        "onset_evidence_frame": None,
        "apex_status": None,
        "apex_evidence_frame": None,
        "offset_status": None,
        "offset_evidence_frame": None,
        "interaction_observable_result": None,
        "interaction_description_en": None,
        "robot_a_observable_motion_en": None,
        "robot_b_observable_motion_en": None,
        "expression_completeness_result": None,
        "expansion_request": None,
        "review_notes": None,
    }


def affect_record(sample_id: str, video: Path, video_hash: str) -> dict[str, Any]:
    actor_fields = {
        "observability_result": None,
        "predicted_class": None,
        "confidence": None,
        "intensity": None,
        "evidence_frames": None,
    }
    return {
        **_public_common(sample_id, video, video_hash),
        "protocol_version": AFFECT_PROTOCOL,
        "allowed_classes": ["neutral", "sad", "happy", "angry", "surprise", "fear"],
        "review_id": None,
        "reviewer_id": None,
        "decode_complete": None,
        "decoded_frame_count": None,
        "robot_a": dict(actor_fields),
        "robot_b": dict(actor_fields),
        "interaction_affect_relation_en": None,
        "review_notes": None,
    }


def _assert_fail_closed_public(record: dict[str, Any]) -> None:
    shared.assert_public_privacy(record)
    required_false = (
        "fixed_duration_window_used",
        "inside_context_crop_used",
        "audio_available",
        "face_geometry_available",
        "finger_geometry_available",
        "label_metadata_exposed",
        "identity_scene_text_or_official_affect_exposed",
        "semantic_supervision_mask",
        "emotion_supervision_mask",
        "license_training_mask",
        "accepted_for_training",
    )
    if any(record.get(name) is not False for name in required_false):
        raise ValueError("Public InterAct full-dyad record opens a fail-closed field")
    if record.get("native_variable_length") is not True:
        raise ValueError("Public InterAct full-dyad record is not variable length")


def _tree_entries(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    if not root.exists():
        return files, directories
    for path in root.rglob("*"):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            raise ValueError(f"Public bundle may not contain symlinks: {relative}")
        if path.is_dir():
            directories.add(relative)
        elif path.is_file():
            files.add(relative)
        else:
            raise ValueError(f"Public bundle contains a non-file entry: {relative}")
    return files, directories


def assert_public_tree_exact(root: Path, expected_files: set[str]) -> None:
    files, directories = _tree_entries(root)
    expected_directories = {"videos"}
    if files != expected_files:
        raise ValueError(
            f"Public bundle file whitelist mismatch: extra={sorted(files - expected_files)}, "
            f"missing={sorted(expected_files - files)}"
        )
    if directories != expected_directories:
        raise ValueError(
            f"Public bundle directory whitelist mismatch: {sorted(directories)}"
        )
    for relative in expected_files:
        path = root / relative
        mode = path.stat().st_mode
        if not stat.S_ISREG(mode):
            raise ValueError(f"Public bundle entry is not a regular file: {relative}")
        if relative.startswith("videos/") and path.stat().st_nlink != 1:
            raise ValueError(f"Public video is not an independent copy: {relative}")


def _write_public_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    shared.atomic_jsonl(path, rows, mode=0o644)


def _write_public_json(path: Path, value: dict[str, Any]) -> None:
    shared.atomic_json(path, value, mode=0o644)


def materialize_or_reuse_video(source: Path, target: Path, expected_hash: str) -> str:
    if target.exists() or target.is_symlink():
        if (
            target.is_symlink()
            or not target.is_file()
            or target.stat().st_nlink != 1
            or sha256_file(target) != expected_hash
        ):
            raise ValueError(f"Existing anonymous video is unsafe or mismatched: {target}")
        return "resume_reused"
    shared.materialize_video(source, target, expected_hash)
    if target.stat().st_nlink != 1:
        raise ValueError(f"Materialized anonymous video is not independent: {target}")
    return "copied"


def validate_cached_result(
    record: dict[str, Any], staging: Path, result: dict[str, Any]
) -> dict[str, Any]:
    payload = dict(result)
    declared_record_hash = payload.pop("evidence_result_record_sha256", None)
    if declared_record_hash != value_sha256(payload):
        raise ValueError("Cached evidence result record SHA mismatch")
    required = {
        "dyad_id": record.get("dyad_id"),
        "dyad_record_sha256": record.get("dyad_record_sha256"),
        "status": "rendered_pending_blind_review",
        "semantic_supervision_mask": False,
        "emotion_supervision_mask": False,
        "license_training_mask": False,
        "accepted_for_training": False,
    }
    if any(result.get(key) != value for key, value in required.items()):
        raise ValueError("Cached evidence result violates its fail-closed binding")
    paths = {name: Path(result[name]).resolve() for name in ("video", "lineage", "summary")}
    for path in paths.values():
        try:
            path.relative_to(staging)
        except ValueError as error:
            raise ValueError("Cached evidence path escapes staging") from error
        if not path.is_file():
            raise FileNotFoundError(path)
    if (
        sha256_file(paths["lineage"]) != result.get("lineage_sha256")
        or sha256_file(paths["summary"]) != result.get("summary_sha256")
    ):
        raise ValueError("Cached evidence metadata SHA mismatch")
    summary = load_json(paths["summary"])
    lineage = load_json(paths["lineage"])
    _validate_lineage(lineage, summary)
    video_validation = result.get("video_validation") or {}
    if (
        video_validation.get("passed") is not True
        or video_validation.get("fully_decodable") is not True
        or video_validation.get("decoded_frames") != result.get("frames")
        or video_validation.get("expected_frames") != result.get("frames")
        or video_validation.get("audio_streams") != 0
        or video_validation.get("width") != 1280
        or video_validation.get("height") != 720
        or video_validation.get("fps") != 30
    ):
        raise ValueError("Cached evidence lacks a valid complete-decode proof")
    return dict(result)


def build(args: argparse.Namespace) -> dict[str, Any]:
    staging = args.staging_root.resolve()
    run_state_path = (args.run_state or (staging / STATE_NAME)).resolve()
    state = load_json(run_state_path)
    if state.get("artifact_kind") != "interact_full_dyad_review_v3_run_state":
        raise ValueError("Wrong InterAct full-dyad renderer state kind")
    if state.get("status") != "complete_pending_blind_review" or state.get(
        "failure_count"
    ) != 0:
        raise ValueError("InterAct full-dyad evidence run is not successfully complete")
    if state.get("accepted_for_training") is not False:
        raise ValueError("InterAct evidence run unexpectedly admits training")
    binding = state.get("input_binding") or {}
    manifest_summary_path = Path(binding["manifest_summary"]).resolve()
    manifest_path = Path(binding["manifest"]).resolve()
    if sha256_file(manifest_summary_path) != binding.get("manifest_summary_sha256"):
        raise ValueError("Evidence state manifest-summary SHA mismatch")
    if sha256_file(manifest_path) != binding.get("manifest_sha256"):
        raise ValueError("Evidence state full-manifest SHA mismatch")
    records = {row["dyad_id"]: row for row in load_jsonl(manifest_path)}
    if len(records) != load_json(manifest_summary_path).get("dyad_count"):
        raise ValueError("Full manifest IDs/count changed")
    results = state.get("results") or {}
    if len(results) != state.get("selected_count") or len(results) != state.get(
        "rendered_count"
    ):
        raise ValueError("Evidence state is incomplete")

    output_root = args.output_root.resolve()
    hidden_root = args.hidden_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "videos").mkdir(parents=True, exist_ok=True)
    os.chmod(output_root, 0o755)
    os.chmod(output_root / "videos", 0o755)
    hidden_root.mkdir(parents=True, exist_ok=True)
    os.chmod(hidden_root, 0o700)
    secret = shared.bundle_secret(hidden_root, args.secret_hex)

    arc_queue: list[dict[str, Any]] = []
    affect_queue: list[dict[str, Any]] = []
    hidden_mapping: list[dict[str, Any]] = []
    expected_files = {
        "arc_action_review_queue.jsonl",
        "affect_review_queue.jsonl",
        "summary.json",
    }
    for dyad_id, prior_result in sorted(results.items()):
        record = records.get(dyad_id)
        if record is None or record.get("dyad_record_sha256") != prior_result.get(
            "dyad_record_sha256"
        ):
            raise ValueError(f"Evidence result is not bound to a current dyad: {dyad_id}")
        video_hash = prior_result.get("video_sha256")
        if not isinstance(video_hash, str) or len(video_hash) != 64:
            raise ValueError(f"Evidence result lacks a video SHA: {dyad_id}")
        sample_id = shared.anonymous_id(secret, "dyadfullv3", dyad_id, video_hash)
        public_video = output_root / "videos" / f"{sample_id}.mp4"
        if public_video.exists() or public_video.is_symlink():
            materialize_or_reuse_video(Path(prior_result["video"]), public_video, video_hash)
            result = validate_cached_result(record, staging, prior_result)
        else:
            result = validate_result(record, staging, prior_result)
            materialize_or_reuse_video(Path(result["video"]), public_video, video_hash)
        arc = arc_record(sample_id, public_video, video_hash)
        affect = affect_record(sample_id, public_video, video_hash)
        _assert_fail_closed_public(arc)
        _assert_fail_closed_public(affect)
        arc_queue.append(arc)
        affect_queue.append(affect)
        expected_files.add(f"videos/{sample_id}.mp4")
        hidden_mapping.append(
            {
                "sample_id": sample_id,
                "dyad_id": dyad_id,
                "dyad_record_sha256": record["dyad_record_sha256"],
                "natural_context_levels": record["natural_context_levels"],
                "displayed_context_level": record["initial_review_level"],
                "source_interval": record["source_interval"],
                "actors": record["actors"],
                "physical_evidence_result": result,
                "physical_evidence_result_record_sha256": result[
                    "evidence_result_record_sha256"
                ],
                "official_scenario_text_or_emotion_exposed": False,
                "semantic_supervision_mask": False,
                "emotion_supervision_mask": False,
                "license_training_mask": False,
                "accepted_for_training": False,
            }
        )
    arc_queue.sort(key=lambda row: row["sample_id"])
    affect_queue.sort(key=lambda row: row["sample_id"])
    hidden_mapping.sort(key=lambda row: row["sample_id"])
    arc_path = output_root / "arc_action_review_queue.jsonl"
    affect_path = output_root / "affect_review_queue.jsonl"
    mapping_path = hidden_root / "mapping.jsonl"
    _write_public_jsonl(arc_path, arc_queue)
    _write_public_jsonl(affect_path, affect_queue)
    shared.atomic_jsonl(mapping_path, hidden_mapping, mode=0o600)
    public_summary = {
        "schema_version": "3.0.0",
        "artifact_kind": "interact_full_dyad_anonymous_blind_review_bundle_v3",
        "records": len(arc_queue),
        "arc_action_queue": str(arc_path),
        "arc_action_queue_sha256": sha256_file(arc_path),
        "affect_queue": str(affect_path),
        "affect_queue_sha256": sha256_file(affect_path),
        "video_layout": "2x2_human_dyad_xz_yz_plus_mujoco_robot_a_b",
        "videos_are_independent_copies_nlink_one": True,
        "public_tree_exact_whitelist": True,
        "native_variable_length": True,
        "fixed_duration_window_used": False,
        "identity_scene_text_or_official_affect_exposed": False,
        "semantic_supervision_mask": False,
        "emotion_supervision_mask": False,
        "license_training_mask": False,
        "accepted_for_training": False,
    }
    _write_public_json(output_root / "summary.json", public_summary)
    hidden_summary = {
        "schema_version": "3.0.0",
        "artifact_kind": "interact_full_dyad_hidden_blind_mapping_v3",
        "run_state": str(run_state_path),
        "run_state_sha256": sha256_file(run_state_path),
        "manifest_summary": str(manifest_summary_path),
        "manifest_summary_sha256": sha256_file(manifest_summary_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "mapping": str(mapping_path),
        "mapping_sha256": sha256_file(mapping_path),
        "public_summary": str((output_root / "summary.json").resolve()),
        "public_summary_sha256": sha256_file(output_root / "summary.json"),
        "bundle_implementation": str(Path(__file__).resolve()),
        "bundle_implementation_sha256": sha256_file(Path(__file__).resolve()),
        "public_distribution_forbidden": True,
        "semantic_supervision_mask": False,
        "emotion_supervision_mask": False,
        "license_training_mask": False,
        "accepted_for_training": False,
    }
    shared.atomic_json(hidden_root / "summary.json", hidden_summary, mode=0o600)
    shared.enforce_hidden_permissions(hidden_root)
    assert_public_tree_exact(output_root, expected_files)
    return {"public": public_summary, "hidden": hidden_summary}


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(build(parse_args(argv)), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
