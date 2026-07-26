#!/usr/bin/env python3
"""Build a private, anonymous render queue for native-length ULA0513 motions.

The source asset names are useful provenance, but they are also semantic and
affect hints.  This stage therefore assigns stable HMAC task IDs and writes the
source mapping to a private file.  Every admitted render task remains the full
source-authored motion: no crop, tile, resample, or duration gate is allowed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
from typing import Any, Iterable

from upper_body_skeleton.retarget_v2_18d import (
    CONTRACT_VERSION,
    JOINT_ORDER_18D,
)


SCHEMA_VERSION = "1.0.0"
SOURCE_ARTIFACT_KIND = "ula0513_native_expression_turn_candidate"
SOURCE_REPRESENTATION = "native_variable_length_robot_expression_turn_v1"
QUEUE_ARTIFACT_KIND = "ula0513_native_private_video_review_queue_record"
ROBOT_CONTRACT = "ula_v2_18d_head_v1"
ANONYMOUS_PROMPT = {
    "en": "Anonymous silent robot motion sample.",
    "zh": "Anonymous silent robot motion sample.",
}
EXPECTED_SEMANTIC_MASKS = {
    "communicative_intent": False,
    "prompt_text": False,
    "robot_observable_motion_form": False,
    "source_behavior_label": False,
}


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
    path: Path, records: Iterable[dict[str, Any]], *, mode: int | None = None
) -> None:
    atomic_text(
        path,
        "".join(stable_json(record) + "\n" for record in records),
        mode=mode,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            records.append(record)
    return records


def _secret(hidden_root: Path, provided_hex: str | None) -> bytes:
    path = hidden_root / "queue_secret.json"
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        existing = bytes.fromhex(str(value["secret_hex"]))
        if provided_hex is not None and existing != bytes.fromhex(provided_hex):
            raise ValueError("provided secret does not match existing queue secret")
        os.chmod(path, 0o600)
        return existing
    secret = bytes.fromhex(provided_hex) if provided_hex else secrets.token_bytes(32)
    if len(secret) < 16:
        raise ValueError("queue secret must contain at least 16 bytes")
    atomic_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "ula0513_native_review_queue_secret",
            "secret_hex": secret.hex(),
            "public_distribution_forbidden": True,
        },
        mode=0o600,
    )
    return secret


def _anonymous_task_id(secret: bytes, record: dict[str, Any]) -> str:
    identity = (
        f"{record['clip_id']}\0{record['record_sha256']}\0"
        f"{record['motion_18d']['safe_csv_sha256']}"
    ).encode("utf-8")
    return "ula_native_" + hmac.new(secret, identity, hashlib.sha256).hexdigest()[:24]


def _verified_record_hash(record: dict[str, Any]) -> None:
    expected = record.get("record_sha256")
    payload = dict(record)
    payload.pop("record_sha256", None)
    if expected != value_sha256(payload):
        raise ValueError(f"{record.get('clip_id')}: record_sha256 mismatch")


def _trajectory_frames(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"empty trajectory: {path}") from error
        if header != JOINT_ORDER_18D:
            raise ValueError(f"trajectory is not ordered ULA V2 18D: {path}")
        count = 0
        for line_number, row in enumerate(reader, 2):
            if len(row) != len(JOINT_ORDER_18D):
                raise ValueError(f"trajectory row {line_number} is not 18D: {path}")
            count += 1
    if count < 1:
        raise ValueError(f"trajectory contains no frames: {path}")
    return count


def _validated_source(record: dict[str, Any]) -> tuple[Path, int, float]:
    clip_id = str(record.get("clip_id") or "")
    if not clip_id:
        raise ValueError("source record is missing clip_id")
    _verified_record_hash(record)
    checks = {
        "artifact_kind": record.get("artifact_kind") == SOURCE_ARTIFACT_KIND,
        "representation": record.get("representation") == SOURCE_REPRESENTATION,
        "physical_qc": record.get("physical_qc", {}).get("passed") is True,
        "accepted": record.get("accepted_for_training") is False,
        "base": record.get("base_motion_eligible") is False,
        "semantic": record.get("semantic_conditioning_eligible") is False,
        "expressive": record.get("expressive_conditioning_eligible") is False,
        "emotion_mask": record.get("emotion_supervision_mask") is False,
        "affect_mask": record.get("affect_observable_supervision_mask") is False,
        "semantic_masks": record.get("semantic_supervision_masks")
        == EXPECTED_SEMANTIC_MASKS,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"{clip_id}: source is not fail-closed/physical-pass: {failed}")

    segment = record.get("training_segment")
    if not isinstance(segment, dict):
        raise ValueError(f"{clip_id}: missing training_segment")
    variable_length_checks = {
        "fixed_window_none": segment.get("fixed_window_sec") is None,
        "not_cropped": segment.get("cropped") is False,
        "not_resampled": segment.get("resampled") is False,
        "not_tiled": segment.get("tiled") is False,
        "complete_asset": segment.get("duration_policy")
        == "one_complete_source_authored_motion_asset",
        "starts_at_zero": segment.get("start_frame") == 0,
        "representation": segment.get("representation") == SOURCE_REPRESENTATION,
    }
    failed = sorted(name for name, passed in variable_length_checks.items() if not passed)
    if failed:
        raise ValueError(f"{clip_id}: native-length contract failed: {failed}")

    motion = record.get("motion_18d")
    if not isinstance(motion, dict):
        raise ValueError(f"{clip_id}: missing motion_18d")
    if motion.get("contract_version") != CONTRACT_VERSION:
        raise ValueError(f"{clip_id}: wrong 18D contract")
    if motion.get("joint_order") != JOINT_ORDER_18D:
        raise ValueError(f"{clip_id}: wrong 18D joint order")
    if motion.get("native_head_3dof_present") is not True:
        raise ValueError(f"{clip_id}: native head 3DoF is required")
    if motion.get("head_mapping_or_synthesis_used") is not False:
        raise ValueError(f"{clip_id}: mapped/synthesized head is forbidden")
    trajectory = Path(str(motion.get("safe_csv_path") or "")).resolve()
    if not trajectory.is_file() or sha256(trajectory) != motion.get("safe_csv_sha256"):
        raise ValueError(f"{clip_id}: safe trajectory evidence mismatch")
    frames = _trajectory_frames(trajectory)
    expected_frames = int(segment.get("frame_count", -1))
    if frames != expected_frames or frames != int(motion.get("frame_count", -1)):
        raise ValueError(f"{clip_id}: trajectory frame count changed")
    if segment.get("end_frame_exclusive") != frames:
        raise ValueError(f"{clip_id}: complete source interval mismatch")
    time_axes = record.get("time_axes", {}).get("output", {})
    span = time_axes.get("sample_span_sec")
    if not isinstance(span, (int, float)) or isinstance(span, bool) or float(span) < 0:
        raise ValueError(f"{clip_id}: invalid native sample span")
    expected_span = (frames - 1) / float(record.get("fps", 0.0))
    if abs(float(span) - expected_span) > 1e-8:
        raise ValueError(f"{clip_id}: native sample span mismatch")
    return trajectory, frames, float(span)


def build_queue(
    manifest: Path,
    output_queue: Path,
    hidden_root: Path,
    *,
    secret_hex: str | None = None,
) -> dict[str, Any]:
    manifest = manifest.resolve()
    output_queue = output_queue.resolve()
    hidden_root = hidden_root.resolve()
    hidden_root.mkdir(parents=True, exist_ok=True)
    os.chmod(hidden_root, 0o700)
    secret = _secret(hidden_root, secret_hex)

    queue: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    skipped_physical_fail = 0
    for record in read_jsonl(manifest):
        physical = record.get("physical_qc")
        if not isinstance(physical, dict) or physical.get("passed") is not True:
            skipped_physical_fail += 1
            continue
        trajectory, frames, span = _validated_source(record)
        task_id = _anonymous_task_id(secret, record)
        queue_record = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": QUEUE_ARTIFACT_KIND,
            "task_id": task_id,
            "speaker_key": task_id,
            "official_split": "train",
            "fixed_split_assignment": "train",
            "robot_contract": ROBOT_CONTRACT,
            "canonical_action": None,
            "canonical_action_role": "disabled_pending_independent_blind_review",
            "canonical_prompt": dict(ANONYMOUS_PROMPT),
            "canonical_prompt_role": "anonymous_renderer_placeholder_not_conditioning_text",
            "trajectory_path": str(trajectory),
            "trajectory_sha256": record["motion_18d"]["safe_csv_sha256"],
            "manual_review_required": True,
            "semantic_action_completeness_review_required": True,
            "affect_observable_review_required": True,
            "semantic_supervision_masks": dict(EXPECTED_SEMANTIC_MASKS),
            "emotion_supervision_mask": False,
            "affect_observable_supervision_mask": False,
            "official_emotion_conditioning_enabled": False,
            "accepted_for_training": False,
            "render_pass_grants_training_admission": False,
            "speech_context_included": False,
            "training_segment": dict(record["training_segment"]),
            "time_axes": record["time_axes"],
            "retarget_segment": {
                "representation": SOURCE_REPRESENTATION,
                "source_frame_count": frames,
                "output_frame_count": frames,
                "output_sample_span_sec": span,
                "cropped": False,
                "resampled": False,
                "tiled": False,
                "fixed_window_sec": None,
            },
            "source_inventory_record_sha256": record["record_sha256"],
        }
        queue_record["input_record_sha256"] = value_sha256(queue_record)
        queue.append(queue_record)
        mapping.append(
            {
                "schema_version": SCHEMA_VERSION,
                "task_id": task_id,
                "clip_id": record["clip_id"],
                "source_clip_id": record["source_clip_id"],
                "source_behavior_label": record["source_behavior_label"],
                "source_behavior_label_role": record["source_behavior_label_role"],
                "source_record_sha256": record["record_sha256"],
                "trajectory_path": str(trajectory),
                "trajectory_sha256": record["motion_18d"]["safe_csv_sha256"],
                "frame_count": frames,
                "sample_span_sec": span,
                "source_asset_name_must_not_be_exposed_before_blind_submission": True,
                "accepted_for_training": False,
            }
        )

    ids = [record["task_id"] for record in queue]
    if len(ids) != len(set(ids)):
        raise ValueError("anonymous task ID collision")
    queue.sort(key=lambda record: record["task_id"])
    mapping.sort(key=lambda record: record["task_id"])
    atomic_jsonl(output_queue, queue, mode=0o600)
    mapping_path = hidden_root / "task_mapping.jsonl"
    atomic_jsonl(mapping_path, mapping, mode=0o600)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "ula0513_native_private_video_review_queue",
        "source_manifest": str(manifest),
        "source_manifest_sha256": sha256(manifest),
        "records": len(queue),
        "skipped_physical_fail": skipped_physical_fail,
        "frame_count": sum(int(record["retarget_segment"]["output_frame_count"]) for record in queue),
        "frame_count_min": (
            min(int(record["retarget_segment"]["output_frame_count"]) for record in queue)
            if queue
            else None
        ),
        "frame_count_max": (
            max(int(record["retarget_segment"]["output_frame_count"]) for record in queue)
            if queue
            else None
        ),
        "fixed_window_sec": None,
        "cropped": False,
        "resampled": False,
        "tiled": False,
        "duration_gate_used": False,
        "output_queue": str(output_queue),
        "output_queue_sha256": sha256(output_queue),
        "hidden_mapping": str(mapping_path),
        "hidden_mapping_sha256": sha256(mapping_path),
        "labels_exposed_in_render_task_ids": False,
        "accepted_for_training": 0,
    }
    atomic_json(hidden_root / "queue_summary.json", summary, mode=0o600)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-queue", type=Path, required=True)
    parser.add_argument("--hidden-root", type=Path, required=True)
    parser.add_argument("--secret-hex")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_queue(
        args.manifest,
        args.output_queue,
        args.hidden_root,
        secret_hex=args.secret_hex,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
