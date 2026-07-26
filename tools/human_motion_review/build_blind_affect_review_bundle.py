#!/usr/bin/env python3
"""Build separated public/hidden bundles for blind robot-affect review."""

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

from tools.human_motion_review.adjudicate_training_dataset import (
    BLIND_AFFECT_PROTOCOL_VERSION,
)


SCHEMA_VERSION = "1.0.0"
BLIND_PROTOCOL_VERSION = BLIND_AFFECT_PROTOCOL_VERSION
OBSERVED_AFFECTS = (
    "neutral",
    "sad",
    "happy",
    "angry",
    "surprise",
    "fear",
    "not_observable",
    "ambiguous",
)
PUBLIC_RECORD_KEYS = {
    "schema_version",
    "sample_id",
    "video_path",
    "video_sha256",
    "observed_affect",
    "confidence",
    "reviewer_id",
    "blind_protocol_version",
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
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            records.append(value)
    return records


def _require_source_record(record: dict[str, Any]) -> tuple[str, Path, str]:
    task_id = str(record.get("task_id") or "")
    if not task_id:
        raise ValueError("render pass requires a task_id")
    fail_closed = {
        "status": record.get("status") == "passed",
        "accepted_for_training": record.get("accepted_for_training") is False,
        "render_admission": record.get("render_pass_grants_training_admission") is False,
        "emotion_mask": record.get("emotion_supervision_mask") is False,
        "emotion_conditioning": record.get("official_emotion_conditioning_enabled") is False,
        "affect_status": record.get("affect_observable_review_status")
        == "candidate_unreviewed",
        "affect_mask": record.get("affect_observable_supervision_mask") is False,
    }
    failed = sorted(name for name, passed in fail_closed.items() if not passed)
    if failed:
        raise ValueError(f"{task_id}: source is not fail-closed: {failed}")
    emotion = str(record.get("emotion_id") or "")
    if emotion not in OBSERVED_AFFECTS[:6]:
        raise ValueError(f"{task_id}: invalid hidden official emotion")
    path = Path(str(record.get("video_path") or "")).resolve()
    expected_sha = str(record.get("video_sha256") or "")
    if not path.is_file() or not expected_sha or sha256(path) != expected_sha:
        raise ValueError(f"{task_id}: video evidence mismatch")
    check = record.get("video_check")
    if not isinstance(check, dict) or check.get("passed") is not True:
        raise ValueError(f"{task_id}: video verification is not passed")
    if check.get("audio_streams") != 0:
        raise ValueError(f"{task_id}: blind affect video must be silent")
    return task_id, path, expected_sha


def _secret(hidden_root: Path, provided_hex: str | None) -> bytes:
    secret_path = hidden_root / "bundle_secret.json"
    if secret_path.is_file():
        value = json.loads(secret_path.read_text(encoding="utf-8"))
        existing = bytes.fromhex(str(value["secret_hex"]))
        if provided_hex is not None and existing != bytes.fromhex(provided_hex):
            raise ValueError("provided secret does not match existing hidden bundle secret")
        os.chmod(secret_path, 0o600)
        return existing
    secret = bytes.fromhex(provided_hex) if provided_hex is not None else secrets.token_bytes(32)
    if len(secret) < 16:
        raise ValueError("bundle secret must contain at least 16 bytes")
    atomic_json(
        secret_path,
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "blind_affect_bundle_secret",
            "secret_hex": secret.hex(),
            "public_distribution_forbidden": True,
        },
        mode=0o600,
    )
    return secret


def _anonymous_id(secret: bytes, task_id: str, video_sha256: str) -> str:
    message = f"{task_id}\0{video_sha256}".encode("utf-8")
    return "affect_" + hmac.new(secret, message, hashlib.sha256).hexdigest()[:24]


def _materialize_anonymous_video(source: Path, target: Path, expected_sha: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or sha256(target) != expected_sha:
            raise ValueError(f"existing anonymous video does not match: {target}")
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    if sha256(target) != expected_sha:
        raise ValueError(f"anonymous video copy failed integrity check: {target}")


def build_bundle(
    passed_paths: list[Path],
    output_root: Path,
    *,
    secret_hex: str | None = None,
    hidden_root: Path | None = None,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    public_root = output_root / "public_A"
    hidden_root = (
        hidden_root.resolve() if hidden_root is not None else output_root / "hidden_B"
    )
    public_root.mkdir(parents=True, exist_ok=True)
    hidden_root.mkdir(parents=True, exist_ok=True)
    os.chmod(hidden_root, 0o700)
    secret = _secret(hidden_root, secret_hex)

    public_records: list[dict[str, Any]] = []
    hidden_records: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    seen_samples: set[str] = set()
    for passed_path in passed_paths:
        for record in read_jsonl(passed_path.resolve()):
            task_id, source_video, video_sha = _require_source_record(record)
            if task_id in seen_tasks:
                raise ValueError(f"duplicate task_id across render manifests: {task_id}")
            seen_tasks.add(task_id)
            sample_id = _anonymous_id(secret, task_id, video_sha)
            if sample_id in seen_samples:
                raise ValueError(f"anonymous sample_id collision: {sample_id}")
            seen_samples.add(sample_id)
            anonymous_video = public_root / "videos" / f"{sample_id}.mp4"
            _materialize_anonymous_video(source_video, anonymous_video, video_sha)
            public_record = {
                "schema_version": SCHEMA_VERSION,
                "sample_id": sample_id,
                "video_path": str(anonymous_video.resolve()),
                "video_sha256": video_sha,
                "observed_affect": None,
                "confidence": None,
                "reviewer_id": None,
                "blind_protocol_version": BLIND_PROTOCOL_VERSION,
            }
            if set(public_record) != PUBLIC_RECORD_KEYS:
                raise AssertionError("public blind record allowlist changed")
            public_records.append(public_record)
            event = record.get("semantic_event")
            hidden_records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "sample_id": sample_id,
                    "task_id": task_id,
                    "source_clip_id": record.get("source_clip_id"),
                    "speaker_key": record.get("speaker_key"),
                    "fixed_split_assignment": record.get("fixed_split_assignment"),
                    "official_emotion": record.get("emotion_id"),
                    "official_category": event.get("category")
                    if isinstance(event, dict)
                    else None,
                    "official_intensity": event.get("intensity")
                    if isinstance(event, dict)
                    else None,
                    "source_video_path": str(source_video),
                    "anonymous_video_path": str(anonymous_video.resolve()),
                    "video_sha256": video_sha,
                    "source_render_record_sha256": value_sha256(record),
                    "upstream_inventory_record_sha256": record.get(
                        "upstream_inventory_record_sha256"
                    ),
                    "selected_record_sha256": record.get("selected_record_sha256"),
                    "retarget_input_manifest_sha256": record.get(
                        "retarget_input_manifest_sha256"
                    ),
                    "match_official_label_only_after_review_submission": True,
                    "accepted_for_training": False,
                }
            )

    public_records.sort(key=lambda item: item["sample_id"])
    hidden_records.sort(key=lambda item: item["sample_id"])
    public_manifest = public_root / "blind_review_queue.jsonl"
    hidden_mapping = hidden_root / "sample_mapping.jsonl"
    atomic_jsonl(public_manifest, public_records)
    atomic_jsonl(hidden_mapping, hidden_records, mode=0o600)
    atomic_json(
        public_root / "review_schema.json",
        {
            "schema_version": SCHEMA_VERSION,
            "blind_protocol_version": BLIND_PROTOCOL_VERSION,
            "allowed_observed_affect": list(OBSERVED_AFFECTS),
            "required_submission_fields": [
                "sample_id",
                "observed_affect",
                "confidence",
                "reviewer_id",
                "blind_protocol_version",
            ],
            "confidence_contract": "number_in_closed_interval_0_to_1",
            "identity_and_label_metadata_in_public_bundle": False,
            "accepted_for_training": False,
        },
    )
    public_summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "anonymous_blind_robot_affect_review_bundle_A",
        "blind_protocol_version": BLIND_PROTOCOL_VERSION,
        "records": len(public_records),
        "queue": str(public_manifest),
        "queue_sha256": sha256(public_manifest),
        "video_filenames_anonymous": True,
        "source_identity_included": False,
        "prompt_category_and_official_emotion_included": False,
        "accepted_for_training": False,
    }
    atomic_json(public_root / "summary.json", public_summary)
    hidden_summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "hidden_blind_robot_affect_mapping_bundle_B",
        "records": len(hidden_records),
        "mapping": str(hidden_mapping),
        "mapping_sha256": sha256(hidden_mapping),
        "public_queue_sha256": public_summary["queue_sha256"],
        "passed_manifests": [str(path.resolve()) for path in passed_paths],
        "passed_manifest_sha256": [sha256(path.resolve()) for path in passed_paths],
        "public_distribution_forbidden": True,
        "accepted_for_training": False,
    }
    atomic_json(hidden_root / "summary.json", hidden_summary, mode=0o600)
    return {"public_A": public_summary, "hidden_B": hidden_summary}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-passed-manifest", action="append", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--hidden-root",
        type=Path,
        help="Store bundle B outside the reviewer-visible output root.",
    )
    parser.add_argument("--secret-hex")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_bundle(
        args.render_passed_manifest,
        args.output_root,
        secret_hex=args.secret_hex,
        hidden_root=args.hidden_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
