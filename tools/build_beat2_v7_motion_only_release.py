#!/usr/bin/env python3
"""Build a strict physical-QC-only BEAT2 v7 pretraining release.

This release does not claim that action text or affect is observable. All
semantic, prompt, behavior, and emotion supervision stays masked.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from upper_body_skeleton.retarget_v2_18d import CONTRACT_VERSION
from upper_body_skeleton.ula_v2_18d_head import (
    ACTION_DIM,
    FORMAL_REQUIRED_18D_GATES,
    FORMAL_SEMANTIC_SUPERVISION_MASKS,
    MOTION_ONLY_EPISODE_CONTRACT,
    MOTION_ONLY_RELEASE_REPORT_FILENAME,
    MOTION_ONLY_REQUIRED_RELEASE_INVARIANTS,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"record must be an object at {path}:{line_number}")
            records.append(value)
    return records


def validate_source_record(record: Mapping[str, Any], *, verify_artifacts: bool) -> None:
    clip_id = str(record.get("clip_id") or record.get("task_id") or "<unknown>")
    errors: list[str] = []
    if record.get("status") != "passed":
        errors.append("source status is not passed")
    frames = record.get("frames")
    fps = record.get("fps")
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 2:
        errors.append("frames must be an integer >= 2")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isclose(float(fps), 30.0, abs_tol=1e-9)
    ):
        errors.append("fps must be exactly 30 Hz")
    gates = record.get("quality_gate")
    if not isinstance(gates, Mapping) or not FORMAL_REQUIRED_18D_GATES.issubset(gates):
        errors.append("quality gate is incomplete")
    elif any(value is not True for value in gates.values()):
        errors.append("a declared quality gate did not pass")
    segment = record.get("training_segment")
    if not isinstance(segment, Mapping):
        errors.append("training_segment is missing")
    elif (
        segment.get("representation") != "native_variable_length_semantic_clip_v1"
        or segment.get("fixed_window_sec") is not None
    ):
        errors.append("training_segment is not native variable length")
    retarget = record.get("retarget_segment")
    if not isinstance(retarget, Mapping) or retarget.get("cropped") is not False:
        errors.append("retarget segment is missing or cropped")
    if record.get("semantic_supervision_masks") != FORMAL_SEMANTIC_SUPERVISION_MASKS:
        errors.append("semantic supervision masks are not all false")
    for field in (
        "behavior_supervision_mask",
        "emotion_supervision_mask",
        "affect_observable_supervision_mask",
        "official_category_conditioning_enabled",
        "official_emotion_conditioning_enabled",
    ):
        if record.get(field) is not False:
            errors.append(f"{field} is not false")
    for path_field, hash_field in (
        ("safe_csv", "safe_csv_sha256"),
        ("quality_json", "quality_json_sha256"),
    ):
        value = str(record.get(path_field) or "").strip()
        path = Path(value).resolve() if value else None
        expected = record.get(hash_field)
        if path is None or not path.is_file():
            errors.append(f"{path_field} is missing")
        elif not is_sha256(expected):
            errors.append(f"{hash_field} is invalid")
        elif verify_artifacts and sha256_file(path) != expected:
            errors.append(f"{path_field} hash mismatch")
    if errors:
        raise ValueError(f"{clip_id}: " + "; ".join(errors))


def build_record(source: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(source)
    source_record_sha256 = canonical_sha256(source)
    record.update(
        {
            "formal_episode_contract": MOTION_ONLY_EPISODE_CONTRACT,
            "accepted_for_training": True,
            "training_admission_status": "motion_only_physical_qc_train_ready",
            "emotion_conditioning_mask": False,
            "independent_review": {
                "present": False,
                "status": "not_performed_motion_only",
                "training_acceptance": False,
                "scope": "semantic_and_affect_not_required_for_motion_only_loss",
            },
            "adjudication": {
                "status": "motion_only_train_ready",
                "reasons": [],
            },
            "motion_only_admission": {
                "physical_qc_only": True,
                "semantic_review_required": False,
                "independent_semantic_review_claimed": False,
                "text_conditioning_enabled": False,
                "emotion_conditioning_enabled": False,
                "audio_conditioning_enabled": False,
                "native_variable_length": True,
                "fixed_duration_training_unit": False,
                "source_record_sha256": source_record_sha256,
            },
            "motion_18d": {
                "state": "passed",
                "partition": "accepted_motion_only",
                "reasons": [],
                "output_contract": CONTRACT_VERSION,
                "action_dim": ACTION_DIM,
                "frames": int(source["frames"]),
                "csv_rows": int(source["frames"]),
                "fps": float(source["fps"]),
                "quality_gate": dict(source["quality_gate"]),
                "quality_json": str(Path(source["quality_json"]).resolve()),
                "quality_sha256": source["quality_json_sha256"],
                "safe_csv": str(Path(source["safe_csv"]).resolve()),
                "safe_csv_sha256": source["safe_csv_sha256"],
                "retarget_segment": dict(source["retarget_segment"]),
                "source_window_frames": int(source["training_segment"]["frame_count"]),
                "upstream_lineage": dict(source.get("lineage_hashes") or {}),
            },
        }
    )
    return record


def build_release(
    *, passed_manifest: Path, output_dir: Path, verify_artifacts: bool = False
) -> dict[str, Any]:
    passed_manifest = passed_manifest.resolve()
    output_dir = output_dir.resolve()
    output_manifest = output_dir / "train_ready.jsonl"
    report_path = output_dir / MOTION_ONLY_RELEASE_REPORT_FILENAME
    existing = [str(path) for path in (output_manifest, report_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite motion-only release: {existing}")
    source_records = read_jsonl(passed_manifest)
    if not source_records:
        raise ValueError("passed manifest contains no records")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in source_records:
        validate_source_record(source, verify_artifacts=verify_artifacts)
        clip_id = str(source.get("clip_id") or source.get("task_id") or "").strip()
        if not clip_id or clip_id in seen:
            raise ValueError(f"missing or duplicate clip_id: {clip_id!r}")
        record = build_record(source)
        record["clip_id"] = clip_id
        records.append(record)
        seen.add(clip_id)

    payload = "".join(
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for record in records
    )
    atomic_text(output_manifest, payload)
    output_hash = sha256_file(output_manifest)
    frame_counts = [int(record["motion_18d"]["frames"]) for record in records]
    spans = [(frames - 1) / 30.0 for frames in frame_counts]
    report = {
        "schema_version": 1,
        "artifact_kind": "beat2_v7_motion_only_physical_qc_release_v1",
        "formal_episode_contract": MOTION_ONLY_EPISODE_CONTRACT,
        "conditioning_policy": "all_text_behavior_emotion_affect_channels_masked_zero",
        "source": {
            "passed_manifest": str(passed_manifest),
            "passed_manifest_sha256": sha256_file(passed_manifest),
            "records": len(source_records),
            "artifact_hashes_verified_during_build": bool(verify_artifacts),
        },
        "outputs": {
            "train_ready": {
                "path": str(output_manifest),
                "records": len(records),
                "sha256": output_hash,
            }
        },
        "scale": {
            "train_ready_clips": len(records),
            "total_frames": sum(frame_counts),
            "total_sample_span_sec": float(sum(spans)),
            "frame_count_min": min(frame_counts),
            "frame_count_median": float(statistics.median(frame_counts)),
            "frame_count_max": max(frame_counts),
            "distinct_frame_count_count": len(set(frame_counts)),
            "speaker_count": len({str(record["speaker_key"]) for record in records}),
            "source_group_count": len(
                {str(record["source_group_key"]) for record in records}
            ),
            "official_split_counts": dict(
                sorted(Counter(str(record.get("official_split")) for record in records).items())
            ),
        },
        "invariants": {
            name: True for name in sorted(MOTION_ONLY_REQUIRED_RELEASE_INVARIANTS)
        },
        "semantic_claims": {
            "text_conditioned_training_ready": False,
            "emotion_conditioned_training_ready": False,
            "captions_may_be_used_as_review_metadata_only": True,
        },
    }
    atomic_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passed-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-artifacts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_release(
        passed_manifest=args.passed_manifest,
        output_dir=args.output_dir,
        verify_artifacts=args.verify_artifacts,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
