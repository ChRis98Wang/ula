#!/usr/bin/env python3
"""Build separate anonymous arc/action and affect queues from v8 render passes."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
import shutil
from pathlib import Path
from typing import Any, Iterable

from tools.human_motion_review import render_beat2_annotation_review as renderer
from tools.human_motion_review.expression_turn_contract import (
    ACTION_PROTOCOL,
    AFFECT_CLASSES,
    AFFECT_PROTOCOL,
    ARC_PROTOCOL,
)


SCHEMA_VERSION = "1.0.0"
SELECTION_KINDS = {"representative100", "stress100"}
ARC_ACTION_PUBLIC_KEYS = {
    "schema_version",
    "sample_id",
    "video_path",
    "video_sha256",
    "context_level",
    "audio_available",
    "label_metadata_exposed",
    "arc_protocol_version",
    "arc_review_id",
    "arc_reviewer_id",
    "onset_status",
    "onset_evidence_frame",
    "onset_basis",
    "apex_status",
    "apex_evidence_frame",
    "apex_basis",
    "offset_status",
    "offset_evidence_frame",
    "offset_basis",
    "action_protocol_version",
    "action_review_id",
    "action_reviewer_id",
    "action_result",
    "observable_description",
    "candidate_text",
    "candidate_text_sha256",
    "candidate_text_provenance",
}
AFFECT_PUBLIC_KEYS = {
    "schema_version",
    "sample_id",
    "video_path",
    "video_sha256",
    "context_level",
    "audio_available",
    "label_metadata_exposed",
    "affect_protocol_version",
    "affect_review_id",
    "affect_reviewer_id",
    "allowed_classes",
    "result",
    "predicted_class",
    "confidence",
}
FORBIDDEN_PUBLIC_KEYS = {
    "canonical_action",
    "canonical_prompt",
    "category",
    "emotion_id",
    "emotion_label",
    "event_label",
    "official_emotion",
    "official_gesture_category",
    "prompt",
    "source",
    "source_clip_id",
    "source_label",
    "source_text",
    "speaker_id",
    "speaker_key",
    "transcript",
}


def stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_sha256(value: object) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def atomic_text(path: Path, payload: str, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    if mode is not None:
        os.chmod(temporary, mode)
    os.replace(temporary, path)


def atomic_jsonl(
    path: Path, records: Iterable[dict[str, Any]], *, mode: int | None = None
) -> None:
    atomic_text(
        path,
        "".join(stable_json(record) + "\n" for record in records),
        mode=mode,
    )


def atomic_json(path: Path, value: object, *, mode: int | None = None) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        mode=mode,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            records.append(value)
    return records


def _walk_keys(value: object):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _assert_public_privacy(record: dict[str, Any], allowed: set[str]) -> None:
    if set(record) != allowed:
        raise AssertionError("public blind record allowlist changed")
    for key in _walk_keys(record):
        lowered = key.lower()
        if (
            lowered in FORBIDDEN_PUBLIC_KEYS
            or lowered.startswith("official_")
            or lowered.startswith("source_")
            or "transcript" in lowered
        ):
            raise ValueError(f"public blind record leaks forbidden key: {key}")


def _secret(hidden_root: Path, provided_hex: str | None) -> bytes:
    secret_path = hidden_root / "bundle_secret.json"
    if secret_path.is_file():
        if secret_path.is_symlink():
            raise ValueError("bundle secret must not be a symbolic link")
        value = json.loads(secret_path.read_text(encoding="utf-8"))
        existing = bytes.fromhex(str(value["secret_hex"]))
        if provided_hex is not None and existing != bytes.fromhex(provided_hex):
            raise ValueError("provided secret does not match existing bundle secret")
        os.chmod(secret_path, 0o600)
        return existing
    secret = bytes.fromhex(provided_hex) if provided_hex else secrets.token_bytes(32)
    if len(secret) < 16:
        raise ValueError("bundle secret must contain at least 16 bytes")
    atomic_json(
        secret_path,
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "expression_turn_v8_blind_bundle_secret",
            "secret_hex": secret.hex(),
            "public_distribution_forbidden": True,
        },
        mode=0o600,
    )
    return secret


def _anonymous_id(secret: bytes, task_id: str, video_hash: str) -> str:
    payload = f"{task_id}\0{video_hash}".encode("utf-8")
    return "expr_" + hmac.new(secret, payload, hashlib.sha256).hexdigest()[:24]


def _materialize_video(source: Path, target: Path, expected_hash: str) -> None:
    source = source.resolve()
    target = Path(os.path.abspath(target))
    if source == target:
        raise ValueError("anonymous video target must differ from its source")
    if not source.is_file() or sha256(source) != expected_hash:
        raise ValueError(f"source video integrity failure: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or sha256(target) != expected_hash:
            raise ValueError(f"existing anonymous video does not match: {target}")

    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    try:
        shutil.copy2(source, temporary)
        if sha256(temporary) != expected_hash:
            raise ValueError(
                f"anonymous video temporary copy integrity failure: {target}"
            )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    if (
        target.is_symlink()
        or sha256(target) != expected_hash
        or os.path.samefile(source, target)
        or target.stat().st_nlink != 1
    ):
        raise ValueError(f"anonymous video integrity failure: {target}")


def _validated_final_render_evidence(
    record: dict[str, Any],
    queue_path: Path | None = None,
) -> tuple[str, Path, str, int]:
    """Validate the immutable final CSV/frame/video evidence for blind review."""

    task_id = str(record.get("task_id") or "")
    if not task_id:
        raise ValueError("render pass requires task_id")
    checks = {
        "status": record.get("status") == "passed",
        "accepted": record.get("accepted_for_training") is False,
        "render_admission": record.get("render_pass_grants_training_admission") is False,
        "emotion_mask": record.get("emotion_supervision_mask") is False,
        "emotion_conditioning": record.get("official_emotion_conditioning_enabled") is False,
        "affect_mask": record.get("affect_observable_supervision_mask") is False,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"{task_id}: render record is not fail-closed: {failed}")
    if "semantic_event" in record or "official_semantic_event" in record:
        raise ValueError(f"{task_id}: render record contains legacy semantic event")
    if record.get("blind_review_must_use_final_trajectory") is not True:
        raise ValueError(f"{task_id}: blind review is not pinned to the final trajectory")

    evidence_queue_path = (
        queue_path.resolve()
        if queue_path is not None
        else (Path.cwd() / "render_passed_manifest.jsonl").resolve()
    )
    trajectory = renderer.resolve_evidence_path(
        record.get("trajectory_path"), evidence_queue_path, "trajectory_path"
    )
    trajectory_record = dict(record)
    trajectory_record["trajectory_path"] = str(trajectory)
    if record.get("quality_json") is not None:
        trajectory_record["quality_json"] = str(
            renderer.resolve_evidence_path(
                record.get("quality_json"), evidence_queue_path, "quality_json"
            )
        )
    trajectory, parsed_trajectory_frames = renderer.validate_trajectory(
        trajectory_record, evidence_queue_path
    )

    video = renderer.resolve_evidence_path(
        record.get("video_path"), evidence_queue_path, "video_path"
    )
    video_hash = str(record.get("video_sha256") or "")
    if not video.is_file() or sha256(video) != video_hash:
        raise ValueError(f"{task_id}: video evidence mismatch")
    video_check = record.get("video_check")
    if (
        not isinstance(video_check, dict)
        or video_check.get("passed") is not True
        or video_check.get("fully_decodable") is not True
        or video_check.get("audio_streams") != 0
        or video_check.get("nonblank") is not True
    ):
        raise ValueError(f"{task_id}: video is not verified silent/nonblank")
    declared_width = video_check.get("width")
    declared_height = video_check.get("height")
    if (
        isinstance(declared_width, bool)
        or not isinstance(declared_width, int)
        or declared_width < 1
        or isinstance(declared_height, bool)
        or not isinstance(declared_height, int)
        or declared_height < 1
        or video_check.get("fps") != renderer.FPS
    ):
        raise ValueError(f"{task_id}: invalid declared video dimensions or FPS")
    decoded_video_check = renderer.validate_video(
        video,
        expected_frames=parsed_trajectory_frames,
        expected_width=declared_width,
        expected_height=declared_height,
        expected_fps=renderer.FPS,
    )
    declaration_mismatches = sorted(
        key
        for key, actual_value in decoded_video_check.items()
        if video_check.get(key) != actual_value
    )
    if declaration_mismatches:
        raise ValueError(
            f"{task_id}: decoded video does not match declared video_check: "
            f"{declaration_mismatches}"
        )

    binding = record.get("final_output_binding")
    if not isinstance(binding, dict):
        raise ValueError(f"{task_id}: missing final CSV/frame/video binding")
    binding_hash = binding.get("sha256")
    binding_payload = {key: value for key, value in binding.items() if key != "sha256"}
    if binding_hash != value_sha256(binding_payload):
        raise ValueError(f"{task_id}: final output binding SHA256 mismatch")
    trajectory_hash = str(record.get("trajectory_sha256") or "")
    output_frames = record.get("trajectory_frames")
    final_checks = {
        "trajectory_exists": trajectory.is_file(),
        "trajectory_hash": trajectory.is_file() and sha256(trajectory) == trajectory_hash,
        "parsed_trajectory_frames": output_frames == parsed_trajectory_frames,
        "queue_expected_frames": record.get("trajectory_frames_expected")
        == parsed_trajectory_frames,
        "retarget_output_frames": record.get("output_frame_count")
        == parsed_trajectory_frames,
        "binding_trajectory_path": binding.get("trajectory_path") == str(trajectory),
        "binding_trajectory_hash": binding.get("trajectory_sha256") == trajectory_hash,
        "binding_output_frames": binding.get("output_frame_count")
        == parsed_trajectory_frames,
        "binding_video_path": binding.get("video_path") == str(video),
        "binding_video_hash": binding.get("video_sha256") == video_hash,
        "binding_decoded_frames": binding.get("video_decoded_frames")
        == parsed_trajectory_frames,
        "verified_decoded_frames": decoded_video_check.get("decoded_frames")
        == parsed_trajectory_frames,
        "binding_fps": binding.get("fps") == renderer.FPS,
    }
    failed = sorted(name for name, passed in final_checks.items() if not passed)
    if failed:
        raise ValueError(f"{task_id}: final output binding mismatch: {failed}")
    role = record.get("final_trajectory_role")
    if role == "safety_monotonic_retimed_final_output":
        safety = record.get("safety_monotonic_retime")
        if (
            not isinstance(safety, dict)
            or safety.get("blind_review_must_use_retimed_output") is not True
            or safety.get("output_frame_count") != output_frames
        ):
            raise ValueError(f"{task_id}: safety-retimed blind evidence mismatch")
    elif role != "native_identity_timeline_no_slowdown_required":
        raise ValueError(f"{task_id}: unsupported final trajectory role")
    context = record.get("context_plan")
    context_level = context.get("selected_level") if isinstance(context, dict) else None
    if isinstance(context_level, bool) or not isinstance(context_level, int):
        raise ValueError(f"{task_id}: selected context level is invalid")
    return task_id, video, video_hash, context_level


def _validated_render_record(
    record: dict[str, Any],
    *,
    expected_selection_kind: str,
    queue_path: Path | None = None,
) -> tuple[str, Path, str, int]:
    evidence = _validated_final_render_evidence(record, queue_path)
    task_id = evidence[0]
    if record.get("expression_turn_selection_kind") != expected_selection_kind:
        raise ValueError(f"{task_id}: render record selection_kind mismatch")
    return evidence


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _preview_secret(hidden_root: Path, provided_hex: str | None) -> bytes:
    """Resolve the stable anonymization key without creating bundle artifacts."""

    secret_path = hidden_root / "bundle_secret.json"
    if secret_path.exists():
        if secret_path.is_symlink() or not secret_path.is_file():
            raise ValueError(f"bundle secret path is not a regular file: {secret_path}")
        try:
            value = json.loads(secret_path.read_text(encoding="utf-8"))
            secret = bytes.fromhex(str(value["secret_hex"]))
        except (KeyError, OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid existing bundle secret: {secret_path}") from error
        if provided_hex is not None:
            try:
                provided = bytes.fromhex(provided_hex)
            except ValueError as error:
                raise ValueError("secret-hex must be valid hexadecimal") from error
            if provided != secret:
                raise ValueError("provided secret does not match existing bundle secret")
    else:
        try:
            secret = bytes.fromhex(provided_hex) if provided_hex else secrets.token_bytes(32)
        except ValueError as error:
            raise ValueError("secret-hex must be valid hexadecimal") from error
    if len(secret) < 16:
        raise ValueError("bundle secret must contain at least 16 bytes")
    return secret


def _validate_existing_public_tree(
    public_root: Path, expected_videos: dict[str, str]
) -> None:
    if not public_root.exists():
        return
    if public_root.is_symlink() or not public_root.is_dir():
        raise ValueError(f"public bundle root must be a real directory: {public_root}")
    allowed = {
        "videos",
        "arc_action_review_queue.jsonl",
        "affect_review_queue.jsonl",
        "summary.json",
    }
    unexpected = sorted(
        entry.name for entry in public_root.iterdir() if entry.name not in allowed
    )
    if unexpected:
        raise ValueError(f"stale unexpected public bundle entries: {unexpected}")
    for name in allowed.difference({"videos"}):
        path = public_root / name
        if path.exists() and (path.is_symlink() or not path.is_file()):
            raise ValueError(f"public bundle artifact is not a regular file: {path}")
    videos_root = public_root / "videos"
    if not videos_root.exists():
        return
    if videos_root.is_symlink() or not videos_root.is_dir():
        raise ValueError(f"public videos path must be a real directory: {videos_root}")
    actual_names = {entry.name for entry in videos_root.iterdir()}
    unexpected_videos = sorted(actual_names.difference(expected_videos))
    if unexpected_videos:
        raise ValueError(f"stale unexpected public videos: {unexpected_videos}")
    for entry in videos_root.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(f"public video is not a regular file: {entry}")
        if sha256(entry) != expected_videos[entry.name]:
            raise ValueError(f"existing public video hash mismatch: {entry}")


def _validate_materialized_videos(
    public_root: Path, expected_videos: dict[str, str]
) -> None:
    videos_root = public_root / "videos"
    actual_names = {entry.name for entry in videos_root.iterdir()}
    if actual_names != set(expected_videos):
        raise ValueError("materialized public video set is incomplete or contains stale files")
    for entry in videos_root.iterdir():
        if (
            entry.is_symlink()
            or not entry.is_file()
            or entry.stat().st_nlink != 1
            or sha256(entry) != expected_videos[entry.name]
        ):
            raise ValueError(f"materialized public video is not immutable: {entry}")


def build_bundle(
    render_passed_manifest: Path,
    output_root: Path,
    *,
    selection_kind: str,
    hidden_root: Path | None = None,
    secret_hex: str | None = None,
) -> dict[str, Any]:
    if selection_kind not in SELECTION_KINDS:
        raise ValueError(f"invalid selection kind: {selection_kind}")
    render_passed_manifest = render_passed_manifest.resolve()
    output_root = output_root.resolve()
    public_entry = output_root / "public"
    if public_entry.is_symlink():
        raise ValueError("public bundle root must not be a symbolic link")
    public_root = public_entry.resolve()
    hidden_root = (
        hidden_root.resolve() if hidden_root is not None else output_root / "hidden"
    )
    if _paths_overlap(public_root, hidden_root):
        raise ValueError("public and hidden bundle roots must be disjoint")
    if public_root == render_passed_manifest or public_root in render_passed_manifest.parents:
        raise ValueError("public bundle root overlaps the render manifest")
    if hidden_root == render_passed_manifest or hidden_root in render_passed_manifest.parents:
        raise ValueError("hidden bundle root overlaps the render manifest")
    secret = _preview_secret(hidden_root, secret_hex)

    arc_action_records = []
    affect_records = []
    hidden_records = []
    materializations: list[tuple[Path, Path, str]] = []
    seen_tasks: set[str] = set()
    seen_samples: set[str] = set()
    for record in read_jsonl(render_passed_manifest):
        task_id, source_video, video_hash, context_level = _validated_render_record(
            record,
            expected_selection_kind=selection_kind,
            queue_path=render_passed_manifest,
        )
        if task_id in seen_tasks:
            raise ValueError(f"duplicate render task: {task_id}")
        seen_tasks.add(task_id)
        sample_id = _anonymous_id(secret, task_id, video_hash)
        if sample_id in seen_samples:
            raise ValueError(f"anonymous sample collision: {sample_id}")
        seen_samples.add(sample_id)
        anonymous_video = public_root / "videos" / f"{sample_id}.mp4"
        materializations.append((source_video, anonymous_video, video_hash))

        common = {
            "schema_version": SCHEMA_VERSION,
            "sample_id": sample_id,
            "video_path": str(anonymous_video.resolve()),
            "video_sha256": video_hash,
            "context_level": context_level,
            "audio_available": False,
            "label_metadata_exposed": False,
        }
        arc_action = {
            **common,
            "arc_protocol_version": ARC_PROTOCOL,
            "arc_review_id": None,
            "arc_reviewer_id": None,
            "onset_status": None,
            "onset_evidence_frame": None,
            "onset_basis": None,
            "apex_status": None,
            "apex_evidence_frame": None,
            "apex_basis": None,
            "offset_status": None,
            "offset_evidence_frame": None,
            "offset_basis": None,
            "action_protocol_version": ACTION_PROTOCOL,
            "action_review_id": None,
            "action_reviewer_id": None,
            "action_result": None,
            "observable_description": None,
            "candidate_text": None,
            "candidate_text_sha256": None,
            "candidate_text_provenance": None,
        }
        affect = {
            **common,
            "affect_protocol_version": AFFECT_PROTOCOL,
            "affect_review_id": None,
            "affect_reviewer_id": None,
            "allowed_classes": sorted(AFFECT_CLASSES),
            "result": None,
            "predicted_class": None,
            "confidence": None,
        }
        _assert_public_privacy(arc_action, ARC_ACTION_PUBLIC_KEYS)
        _assert_public_privacy(affect, AFFECT_PUBLIC_KEYS)
        arc_action_records.append(arc_action)
        affect_records.append(affect)

        turn = record.get("expression_turn")
        hidden_records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": sample_id,
                "task_id": task_id,
                "source_clip_id": record.get("source_clip_id"),
                "speaker_key": record.get("speaker_key"),
                "fixed_split_assignment": record.get("fixed_split_assignment"),
                "selection_kind": selection_kind,
                "official_emotion": record.get("emotion_id"),
                "official_categories": (
                    turn.get("official_categories") if isinstance(turn, dict) else None
                ),
                "source_video_path": str(source_video),
                "anonymous_video_path": str(anonymous_video.resolve()),
                "video_sha256": video_hash,
                "source_render_record_sha256": value_sha256(record),
                "expression_turn_record_sha256": record.get(
                    "expression_turn_record_sha256"
                ),
                "expression_turn_selection_record_sha256": record.get(
                    "expression_turn_selection_record_sha256"
                ),
                "selected_record_sha256": record.get("selected_record_sha256"),
                "retarget_input_manifest_sha256": record.get(
                    "retarget_input_manifest_sha256"
                ),
                "match_official_metadata_only_after_blind_submission": True,
                "accepted_for_training": False,
            }
        )

    arc_action_records.sort(key=lambda item: item["sample_id"])
    affect_records.sort(key=lambda item: item["sample_id"])
    hidden_records.sort(key=lambda item: item["sample_id"])
    expected_videos = {
        target.name: digest for _source, target, digest in materializations
    }
    _validate_existing_public_tree(public_root, expected_videos)

    public_root.mkdir(parents=True, exist_ok=True)
    hidden_root.mkdir(parents=True, exist_ok=True)
    if public_root.resolve() != public_root or _paths_overlap(
        public_root.resolve(), hidden_root.resolve()
    ):
        raise ValueError("created public and hidden roots are not disjoint real paths")
    os.chmod(hidden_root, 0o700)
    _secret(hidden_root, secret.hex())
    for source, target, digest in materializations:
        _materialize_video(source, target, digest)
    _validate_materialized_videos(public_root, expected_videos)

    arc_action_path = public_root / "arc_action_review_queue.jsonl"
    affect_path = public_root / "affect_review_queue.jsonl"
    hidden_path = hidden_root / "sample_mapping.jsonl"
    atomic_jsonl(arc_action_path, arc_action_records)
    atomic_jsonl(affect_path, affect_records)
    atomic_jsonl(hidden_path, hidden_records, mode=0o600)

    public_summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "expression_turn_v8_separate_blind_review_bundle",
        "selection_kind": selection_kind,
        "records": len(arc_action_records),
        "arc_action_queue": str(arc_action_path),
        "arc_action_queue_sha256": sha256(arc_action_path),
        "affect_queue": str(affect_path),
        "affect_queue_sha256": sha256(affect_path),
        "affect_ontology": sorted(AFFECT_CLASSES),
        "same_anonymous_silent_video_used_by_both_reviews": True,
        "all_reviews_pinned_to_final_csv_frame_video_binding": True,
        "action_and_affect_review_ids_and_reviewers_must_be_independent": True,
        "source_identity_official_action_text_and_emotion_exposed": False,
        "accepted_for_training": False,
    }
    hidden_summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "expression_turn_v8_hidden_blind_mapping",
        "selection_kind": selection_kind,
        "records": len(hidden_records),
        "mapping": str(hidden_path),
        "mapping_sha256": sha256(hidden_path),
        "render_passed_manifest": str(render_passed_manifest),
        "render_passed_manifest_sha256": sha256(render_passed_manifest),
        "public_distribution_forbidden": True,
        "accepted_for_training": False,
    }
    atomic_json(public_root / "summary.json", public_summary)
    atomic_json(hidden_root / "summary.json", hidden_summary, mode=0o600)
    return {"public": public_summary, "hidden": hidden_summary}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-passed-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--hidden-root", type=Path)
    parser.add_argument(
        "--selection-kind", choices=sorted(SELECTION_KINDS), required=True
    )
    parser.add_argument("--secret-hex")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_bundle(
        args.render_passed_manifest,
        args.output_root,
        selection_kind=args.selection_kind,
        hidden_root=args.hidden_root,
        secret_hex=args.secret_hex,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
