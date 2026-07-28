#!/usr/bin/env python3
"""Build a hash-bound provenance lock for the expanded BEAT2 v8 motion pool."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


LOCK_KIND = "ula_v2_18d_motion_only_pretrain_provenance_lock_v1"
FINAL_REPORT_KIND = "beat2_v8_expansion_final_report_v1"
RELEASE_REPORT_KIND = "beat2_v8_expansion_motion_only_physical_qc_release_v1"
EPISODE_CONTRACT = "ula_v2_18d_motion_only_physical_qc_v1"
DURATION_POLICY = "native_variable_length_physical_qc_no_fixed_duration_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _binding(owner: Path, value: object, *, name: str) -> tuple[Path, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Missing binding: {name}")
    raw_path = value.get("path")
    expected = value.get("sha256")
    if not isinstance(raw_path, str) or not raw_path or not isinstance(expected, str):
        raise ValueError(f"Invalid binding: {name}")
    path = Path(raw_path)
    if not path.is_absolute():
        path = owner.parent / path
    path = path.resolve()
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"Binding SHA mismatch: {name}")
    return path, expected


def _require_same(actual: Path, expected: Path, *, name: str) -> None:
    if actual.resolve() != expected.resolve():
        raise ValueError(f"{name} path does not match the supplied artifact")


def build_lock(
    *,
    base_lock_path: Path,
    expansion_report_path: Path,
    release_report_path: Path,
    passed_manifest_path: Path,
    train_ready_manifest_path: Path,
) -> dict[str, Any]:
    paths = [
        base_lock_path,
        expansion_report_path,
        release_report_path,
        passed_manifest_path,
        train_ready_manifest_path,
    ]
    base_lock_path, expansion_report_path, release_report_path, passed_manifest_path, train_ready_manifest_path = (
        path.resolve() for path in paths
    )
    for path in paths:
        if not path.resolve().is_file():
            raise FileNotFoundError(path)

    base = read_json(base_lock_path)
    if (
        base.get("artifact_kind") != LOCK_KIND
        or base.get("formal_episode_contract") != EPISODE_CONTRACT
        or base.get("accepted_for_training") is not True
        or base.get("formal_release_allowed") is not True
        or base.get("duration_policy") != DURATION_POLICY
    ):
        raise ValueError("Base provenance lock is not an admitted motion-only lock")
    base_license = base.get("license_gate")
    base_artifacts = base.get("locked_artifacts")
    if (
        not isinstance(base_license, Mapping)
        or base_license.get("training_authorized_by_this_lock") is not True
        or base_license.get("formal_release_blocked") is not False
        or not isinstance(base_artifacts, Mapping)
    ):
        raise ValueError("Base provenance lock license/artifact gate is incomplete")

    expansion = read_json(expansion_report_path)
    if (
        expansion.get("artifact_kind") != FINAL_REPORT_KIND
        or expansion.get("accepted_for_training") is not True
        or not isinstance(expansion.get("bindings"), Mapping)
    ):
        raise ValueError("Expansion final report is not accepted")
    bound_passed, _ = _binding(
        expansion_report_path,
        expansion["bindings"].get("expanded_passed_min30"),
        name="expanded_passed_min30",
    )
    bound_train_ready, _ = _binding(
        expansion_report_path,
        expansion["bindings"].get("expanded_train_ready"),
        name="expanded_train_ready",
    )
    _require_same(bound_passed, passed_manifest_path, name="passed manifest")
    _require_same(bound_train_ready, train_ready_manifest_path, name="train-ready manifest")

    release = read_json(release_report_path)
    scale = release.get("scale")
    invariants = release.get("invariants")
    semantic = release.get("semantic_claims")
    output = (release.get("outputs") or {}).get("train_ready")
    if (
        release.get("artifact_kind") != RELEASE_REPORT_KIND
        or release.get("formal_episode_contract") != EPISODE_CONTRACT
        or release.get("conditioning_policy")
        != "all_text_behavior_emotion_affect_channels_masked_zero"
        or not isinstance(scale, Mapping)
        or not isinstance(invariants, Mapping)
        or not invariants
        or any(value is not True for value in invariants.values())
        or not isinstance(semantic, Mapping)
        or semantic.get("text_conditioned_training_ready") is not False
        or semantic.get("emotion_conditioned_training_ready") is not False
        or not isinstance(output, Mapping)
    ):
        raise ValueError("Motion-only release report violates its fail-closed contract")
    release_train_ready, _ = _binding(
        release_report_path, output, name="release train_ready"
    )
    _require_same(release_train_ready, train_ready_manifest_path, name="release train_ready")
    episode_count = scale.get("train_ready_clips")
    if isinstance(episode_count, bool) or not isinstance(episode_count, int) or episode_count < 1:
        raise ValueError("Release report has no valid training scale")
    if output.get("records") != episode_count:
        raise ValueError("Release output count differs from release scale")

    required_base_artifacts = {
        "acquisition_receipt",
        "training_pool_low_medium",
        "user_confirmation_receipt",
    }
    if not required_base_artifacts.issubset(base_artifacts):
        raise ValueError("Base lock lacks acquisition/inventory/confirmation artifacts")
    locked_artifacts = {
        name: deepcopy(dict(base_artifacts[name])) for name in required_base_artifacts
    }
    locked_artifacts.update(
        {
            "physical_qc_passed_manifest": {
                "path": str(passed_manifest_path),
                "sha256": sha256_file(passed_manifest_path),
            },
            "train_ready_manifest": {
                "path": str(train_ready_manifest_path),
                "sha256": sha256_file(train_ready_manifest_path),
            },
            "motion_only_release_report": {
                "path": str(release_report_path),
                "sha256": sha256_file(release_report_path),
            },
            "expansion_final_report": {
                "path": str(expansion_report_path),
                "sha256": sha256_file(expansion_report_path),
            },
        }
    )
    return {
        "schema_version": 1,
        "artifact_kind": LOCK_KIND,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_episode_contract": EPISODE_CONTRACT,
        "duration_policy": DURATION_POLICY,
        "derived_from_lock": str(base_lock_path),
        "derived_from_lock_sha256": sha256_file(base_lock_path),
        "dataset_filter": {
            "policy": "v8_full_retarget_physical_qc_plus_minimum_30_frames",
            "min_frame_count": 30,
            "source_release": "beat2_v8_expansion_motion_only_release",
        },
        "dataset_scale": {
            "episode_count": episode_count,
            "total_frames": scale.get("total_frames"),
            "frame_count_min": scale.get("frame_count_min"),
            "frame_count_max": scale.get("frame_count_max"),
            "distinct_frame_count_count": scale.get("distinct_frame_count_count"),
            "speaker_count": scale.get("speaker_count"),
            "source_group_count": scale.get("source_group_count"),
            "total_sample_span_sec": scale.get("total_sample_span_sec"),
        },
        "license_gate": deepcopy(dict(base_license)),
        "locked_artifacts": locked_artifacts,
        "policy": {
            "audio_enabled": False,
            "face_and_fingers_retargeted": False,
            "native_variable_event_duration": True,
            "physical_qc_cannot_enable_semantic_or_affect_supervision": True,
            "semantic_and_affect_supervision_fail_closed": True,
            "base_v7_records_hash_locked_unchanged": True,
        },
        "experimental_local_processing_allowed": True,
        "accepted_for_training": True,
        "formal_release_allowed": True,
    }


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-lock", type=Path, required=True)
    parser.add_argument("--expansion-report", type=Path, required=True)
    parser.add_argument("--release-report", type=Path, required=True)
    parser.add_argument("--passed-manifest", type=Path, required=True)
    parser.add_argument("--train-ready-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_lock(
        base_lock_path=args.base_lock,
        expansion_report_path=args.expansion_report,
        release_report_path=args.release_report,
        passed_manifest_path=args.passed_manifest,
        train_ready_manifest_path=args.train_ready_manifest,
    )
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": sha256_file(args.output.resolve()),
                "dataset_scale": result["dataset_scale"],
                "accepted_for_training": result["accepted_for_training"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
