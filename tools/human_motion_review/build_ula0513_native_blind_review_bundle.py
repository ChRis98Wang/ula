#!/usr/bin/env python3
"""Publish label-free blind review queues for rendered ULA0513 motions."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import shutil
from typing import Any, Iterable

from tools.human_motion_review.build_expression_turn_blind_review_bundle_v8 import (
    ACTION_PROTOCOL,
    AFFECT_PROTOCOL,
    AFFECT_CLASSES,
    ARC_ACTION_PUBLIC_KEYS,
    ARC_PROTOCOL,
    AFFECT_PUBLIC_KEYS,
    FORBIDDEN_PUBLIC_KEYS,
)


SCHEMA_VERSION = "1.0.0"
ROBOT_CONTRACT = "ula_v2_18d_head_v1"


def stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def value_sha256(value: object) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, payload: str, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    if mode is not None:
        os.chmod(temporary, mode)
    os.replace(temporary, path)


def atomic_json(path: Path, value: object, *, mode: int | None = None) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        mode=mode,
    )


def atomic_jsonl(
    path: Path, rows: Iterable[dict[str, Any]], *, mode: int | None = None
) -> None:
    atomic_text(
        path,
        "".join(stable_json(row) + "\n" for row in rows),
        mode=mode,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
            rows.append(value)
    return rows


def _secret(hidden_root: Path, provided_hex: str | None) -> bytes:
    path = hidden_root / "bundle_secret.json"
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        existing = bytes.fromhex(str(value["secret_hex"]))
        if provided_hex is not None and existing != bytes.fromhex(provided_hex):
            raise ValueError("provided secret does not match existing bundle secret")
        os.chmod(path, 0o600)
        return existing
    secret = bytes.fromhex(provided_hex) if provided_hex else secrets.token_bytes(32)
    if len(secret) < 16:
        raise ValueError("bundle secret must contain at least 16 bytes")
    atomic_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "ula0513_native_blind_bundle_secret",
            "secret_hex": secret.hex(),
            "public_distribution_forbidden": True,
        },
        mode=0o600,
    )
    return secret


def _sample_id(secret: bytes, task_id: str, video_hash: str) -> str:
    payload = f"{task_id}\0{video_hash}".encode("utf-8")
    return "expr_" + hmac.new(secret, payload, hashlib.sha256).hexdigest()[:24]


def _walk_keys(value: object):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _assert_public(record: dict[str, Any], allowed: set[str]) -> None:
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


def _materialize_video(source: Path, target: Path, expected_hash: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or sha256(target) != expected_hash:
            raise ValueError(f"existing anonymous video does not match: {target}")
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    if sha256(target) != expected_hash:
        raise ValueError(f"anonymous video integrity failure: {target}")


def _mapping_by_task(mapping_path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(mapping_path):
        task_id = str(record.get("task_id") or "")
        if not task_id or task_id in result:
            raise ValueError("hidden task mapping has missing/duplicate task_id")
        if record.get("accepted_for_training") is not False:
            raise ValueError(f"{task_id}: hidden mapping is not fail-closed")
        result[task_id] = record
    return result


def _validated_render(
    record: dict[str, Any], mapping: dict[str, Any]
) -> tuple[Path, str, int, float]:
    task_id = str(record.get("task_id") or "")
    checks = {
        "status": record.get("status") == "passed",
        "accepted": record.get("accepted_for_training") is False,
        "render_admission": record.get("render_pass_grants_training_admission") is False,
        "robot_contract": record.get("robot_contract") == ROBOT_CONTRACT,
        "canonical_action": record.get("canonical_action") is None,
        "emotion_mask": record.get("emotion_supervision_mask") is False,
        "affect_mask": record.get("affect_observable_supervision_mask") is False,
        "official_emotion_conditioning": record.get(
            "official_emotion_conditioning_enabled"
        )
        is False,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"{task_id}: render record is not fail-closed: {failed}")
    frames = int(mapping.get("frame_count", -1))
    span = float(mapping.get("sample_span_sec", -1.0))
    if record.get("trajectory_frames") != frames:
        raise ValueError(f"{task_id}: render changed native frame count")
    segment = record.get("training_segment")
    if not isinstance(segment, dict):
        raise ValueError(f"{task_id}: missing native training segment")
    segment_checks = {
        "frame_count": segment.get("frame_count") == frames,
        "fixed_window": segment.get("fixed_window_sec") is None,
        "cropped": segment.get("cropped") is False,
        "resampled": segment.get("resampled") is False,
        "tiled": segment.get("tiled") is False,
    }
    failed = sorted(name for name, passed in segment_checks.items() if not passed)
    if failed:
        raise ValueError(f"{task_id}: native-length segment failed: {failed}")
    if record.get("trajectory_sha256") != mapping.get("trajectory_sha256"):
        raise ValueError(f"{task_id}: trajectory lineage mismatch")
    check = record.get("video_check")
    if not isinstance(check, dict):
        raise ValueError(f"{task_id}: missing video verification")
    video_checks = {
        "passed": check.get("passed") is True,
        "fully_decodable": check.get("fully_decodable") is True,
        "nonblank": check.get("nonblank") is True,
        "silent": check.get("audio_streams") == 0,
        "one_video": check.get("video_streams") == 1,
        "codec": check.get("codec") == "h264",
        "pixel_format": check.get("pixel_format") == "yuv420p",
        "frames": check.get("decoded_frames") == frames,
        "fps": float(check.get("fps", 0.0)) == 30.0,
        "faststart": check.get("faststart") is True,
    }
    failed = sorted(name for name, passed in video_checks.items() if not passed)
    if failed:
        raise ValueError(f"{task_id}: video verification failed: {failed}")
    video = Path(str(record.get("video_path") or "")).resolve()
    video_hash = str(record.get("video_sha256") or "")
    if not video.is_file() or sha256(video) != video_hash:
        raise ValueError(f"{task_id}: video evidence mismatch")
    return video, video_hash, frames, span


def build_bundle(
    passed_manifest: Path,
    task_mapping: Path,
    output_root: Path,
    hidden_root: Path,
    *,
    secret_hex: str | None = None,
) -> dict[str, Any]:
    passed_manifest = passed_manifest.resolve()
    task_mapping = task_mapping.resolve()
    output_root = output_root.resolve()
    hidden_root = hidden_root.resolve()
    public_root = output_root / "public"
    public_root.mkdir(parents=True, exist_ok=True)
    hidden_root.mkdir(parents=True, exist_ok=True)
    os.chmod(hidden_root, 0o700)
    secret = _secret(hidden_root, secret_hex)
    mapping = _mapping_by_task(task_mapping)
    render_records = read_jsonl(passed_manifest)
    render_ids = [str(record.get("task_id") or "") for record in render_records]
    if len(render_ids) != len(set(render_ids)):
        raise ValueError("render manifest has duplicate task_id")
    if set(render_ids) != set(mapping):
        missing = sorted(set(mapping) - set(render_ids))
        extra = sorted(set(render_ids) - set(mapping))
        raise ValueError(f"render/mapping task set mismatch: missing={missing}, extra={extra}")

    arc_action_rows: list[dict[str, Any]] = []
    affect_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    for record in render_records:
        task_id = str(record["task_id"])
        source = mapping[task_id]
        video, video_hash, frames, span = _validated_render(record, source)
        sample_id = _sample_id(secret, task_id, video_hash)
        anonymous_video = public_root / "videos" / f"{sample_id}.mp4"
        _materialize_video(video, anonymous_video, video_hash)
        common = {
            "schema_version": SCHEMA_VERSION,
            "sample_id": sample_id,
            "video_path": str(anonymous_video),
            "video_sha256": video_hash,
            "context_level": 0,
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
        _assert_public(arc_action, ARC_ACTION_PUBLIC_KEYS)
        _assert_public(affect, AFFECT_PUBLIC_KEYS)
        arc_action_rows.append(arc_action)
        affect_rows.append(affect)
        hidden_rows.append(
            {
                **source,
                "sample_id": sample_id,
                "anonymous_video_path": str(anonymous_video),
                "video_sha256": video_hash,
                "render_record_sha256": value_sha256(record),
                "frame_count": frames,
                "sample_span_sec": span,
                "match_source_asset_name_only_after_blind_submissions": True,
                "accepted_for_training": False,
            }
        )

    arc_action_rows.sort(key=lambda row: row["sample_id"])
    affect_rows.sort(key=lambda row: row["sample_id"])
    hidden_rows.sort(key=lambda row: row["sample_id"])
    arc_path = public_root / "arc_action_review_queue.jsonl"
    affect_path = public_root / "affect_review_queue.jsonl"
    hidden_path = hidden_root / "sample_mapping.jsonl"
    atomic_jsonl(arc_path, arc_action_rows)
    atomic_jsonl(affect_path, affect_rows)
    atomic_jsonl(hidden_path, hidden_rows, mode=0o600)
    spans = [float(row["sample_span_sec"]) for row in hidden_rows]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "ula0513_native_separate_blind_review_bundle",
        "records": len(hidden_rows),
        "arc_action_queue": str(arc_path),
        "arc_action_queue_sha256": sha256(arc_path),
        "affect_queue": str(affect_path),
        "affect_queue_sha256": sha256(affect_path),
        "affect_ontology": sorted(AFFECT_CLASSES),
        "native_sample_span_sec_min": min(spans) if spans else None,
        "native_sample_span_sec_max": max(spans) if spans else None,
        "fixed_window_sec": None,
        "duration_gate_used": False,
        "all_videos_silent_fully_decodable_nonblank_h264_yuv420p": True,
        "source_identity_action_and_affect_hints_exposed": False,
        "arc_action_and_affect_reviewers_must_be_independent": True,
        "at_least_two_independent_affect_submissions_required": True,
        "render_pass_grants_training_admission": False,
        "accepted_for_training": 0,
    }
    private_summary = {
        **summary,
        "artifact_kind": "ula0513_native_hidden_blind_review_mapping",
        "passed_manifest": str(passed_manifest),
        "passed_manifest_sha256": sha256(passed_manifest),
        "task_mapping": str(task_mapping),
        "task_mapping_sha256": sha256(task_mapping),
        "sample_mapping": str(hidden_path),
        "sample_mapping_sha256": sha256(hidden_path),
        "public_distribution_forbidden": True,
    }
    atomic_json(public_root / "summary.json", summary)
    atomic_json(hidden_root / "summary.json", private_summary, mode=0o600)
    return {"public": summary, "hidden": private_summary}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passed-manifest", type=Path, required=True)
    parser.add_argument("--task-mapping", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--hidden-root", type=Path, required=True)
    parser.add_argument("--secret-hex")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_bundle(
        args.passed_manifest,
        args.task_mapping,
        args.output_root,
        args.hidden_root,
        secret_hex=args.secret_hex,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
