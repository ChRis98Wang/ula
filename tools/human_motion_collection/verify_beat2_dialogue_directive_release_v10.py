#!/usr/bin/env python3
"""Independently verify a built BEAT2 dialogue/directive v10 release."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.human_motion_collection.build_beat2_dialogue_directive_release_v10 import (
    COUNTERFACTUAL_KIND,
    EXCLUSION_KIND,
    PROVENANCE_LOCK_KIND,
)
from upper_body_skeleton.ula_v2_dialogue_directive_episode import (
    FORMAL_EPISODE_CONTRACT,
    canonical_sha256,
    validate_dialogue_directive_v10_episode,
)


DEFAULT_RELEASE_DIR = Path(
    "/home/gez/nas/cloud/gez/human_motion/processed/"
    "beat2_dialogue_directive_release_v10"
)
AUDIT_KIND = "beat2_dialogue_directive_release_v10_integrity_audit_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            records.append(value)
    return records


def _index(
    records: list[dict[str, Any]], field: str, *, context: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        key = str(record.get(field) or "").strip()
        if not key or key in result:
            raise ValueError(f"{context}: missing or duplicate {field}: {key!r}")
        result[key] = record
    return result


def _assert_file_binding(
    lock: Mapping[str, Any], release_dir: Path, path_field: str, hash_field: str
) -> Path:
    path = Path(str(lock.get(path_field) or "")).resolve()
    if path.parent != release_dir or not path.is_file():
        raise ValueError(f"release lock has an invalid {path_field}")
    if lock.get(hash_field) != _sha256_file(path):
        raise ValueError(f"release lock {hash_field} changed")
    return path


def verify_release(release_dir: str | Path) -> dict[str, Any]:
    release_dir = Path(release_dir).resolve()
    lock_path = release_dir / "provenance_lock.json"
    lock = _read_json(lock_path)
    if (
        lock.get("artifact_kind") != PROVENANCE_LOCK_KIND
        or lock.get("accepted_for_training") is not True
        or lock.get("formal_episode_contract") != FORMAL_EPISODE_CONTRACT
        or lock.get("fixed_six_second_training_unit") is not False
        or lock.get("duration_policy")
        != "native_variable_length_no_fixed_or_target_duration_v1"
    ):
        raise ValueError("release provenance lock contract changed")

    train_path = _assert_file_binding(
        lock,
        release_dir,
        "train_ready_manifest",
        "train_ready_manifest_sha256",
    )
    counterfactual_path = _assert_file_binding(
        lock,
        release_dir,
        "counterfactual_pair_manifest",
        "counterfactual_pair_manifest_sha256",
    )
    exclusion_path = _assert_file_binding(
        lock,
        release_dir,
        "dialogue_unavailable_manifest",
        "dialogue_unavailable_manifest_sha256",
    )

    episodes = _read_jsonl(train_path)
    counterfactuals = _read_jsonl(counterfactual_path)
    exclusions = _read_jsonl(exclusion_path)
    episode_by_id = _index(episodes, "clip_id", context="train-ready manifest")
    counterfactual_by_id = _index(
        counterfactuals, "anchor_clip_id", context="counterfactual manifest"
    )
    exclusion_by_id = _index(exclusions, "clip_id", context="exclusion manifest")
    if set(episode_by_id) != set(counterfactual_by_id):
        raise ValueError("train-ready and counterfactual membership differs")
    if set(episode_by_id) & set(exclusion_by_id):
        raise ValueError("dialogue-ready and excluded memberships overlap")

    for clip_id, episode in episode_by_id.items():
        validate_dialogue_directive_v10_episode(episode)
        pair = counterfactual_by_id[clip_id]
        pair_without_hash = dict(pair)
        declared_pair_hash = pair_without_hash.pop("record_sha256", None)
        if (
            pair.get("artifact_kind") != COUNTERFACTUAL_KIND
            or pair.get("accepted_as_positive_motion_target") is not False
            or declared_pair_hash != canonical_sha256(pair_without_hash)
        ):
            raise ValueError(f"{clip_id}: counterfactual record contract changed")
        if episode["dialogue_motion_alignment"]["hard_negative_record_sha256"] != declared_pair_hash:
            raise ValueError(f"{clip_id}: counterfactual binding changed")
        matched = pair["matched"]
        if matched != {
            "directive_text_sha256": episode["directive_text_sha256"],
            "dialogue_text_sha256": episode["dialogue_text_sha256"],
            "trajectory_sha256": episode["trajectory_sha256"],
        }:
            raise ValueError(f"{clip_id}: matched counterfactual identity changed")
        for channel, hash_field in (
            ("dialogue_shuffled", "dialogue_text_sha256"),
            ("directive_shuffled", "directive_text_sha256"),
        ):
            negative = pair[channel]
            source_id = str(negative.get("source_clip_id") or "")
            source = episode_by_id.get(source_id)
            if source is None:
                raise ValueError(f"{clip_id}: {channel} source is outside the release")
            if (
                negative.get("same_split") is not True
                or source["fixed_split_assignment"] != episode["fixed_split_assignment"]
                or pair["fixed_split_assignment"] != episode["fixed_split_assignment"]
            ):
                raise ValueError(f"{clip_id}: {channel} crosses a data split")
            if (
                negative.get("different_source_group") is not True
                or source["source_group_key"] == episode["source_group_key"]
            ):
                raise ValueError(f"{clip_id}: {channel} leaks the source group")
            if negative.get(hash_field) != source[hash_field]:
                raise ValueError(f"{clip_id}: {channel} text hash does not bind its source")
            matched_hash = matched[hash_field]
            if negative[hash_field] == matched_hash:
                raise ValueError(f"{clip_id}: {channel} is not a true text negative")

    for clip_id, exclusion in exclusion_by_id.items():
        exclusion_without_hash = dict(exclusion)
        declared_hash = exclusion_without_hash.pop("record_sha256", None)
        if (
            exclusion.get("artifact_kind") != EXCLUSION_KIND
            or exclusion.get("accepted_for_dialogue_conditioning") is not False
            or exclusion.get("retained_for_base_v9_directive_and_motion_training") is not True
            or declared_hash != canonical_sha256(exclusion_without_hash)
        ):
            raise ValueError(f"{clip_id}: exclusion record contract changed")

    scale = lock["dataset_scale"]
    frame_counts = [int(row["frames"]) for row in episodes]
    split_counts = Counter(row["fixed_split_assignment"] for row in episodes)
    recomputed = {
        "base_v9_episode_count": len(episodes) + len(exclusions),
        "dialogue_conditioned_episode_count": len(episodes),
        "dialogue_unavailable_retained_in_v9_count": len(exclusions),
        "all_base_motion_accounted_for": True,
        "frame_count": sum(frame_counts),
        "sample_span_sec": sum((frames - 1) / 30.0 for frames in frame_counts),
        "minimum_frames": min(frame_counts),
        "maximum_frames": max(frame_counts),
        "distinct_frame_counts": len(set(frame_counts)),
        "speaker_count": len({row["speaker_key"] for row in episodes}),
        "source_group_count": len({row["source_group_key"] for row in episodes}),
        "distinct_directive_text_count": len({row["directive_text"] for row in episodes}),
        "distinct_dialogue_text_count": len({row["dialogue_text"] for row in episodes}),
        "minimum_dialogue_token_count": min(
            row["dialogue_contract"]["token_count"] for row in episodes
        ),
        "maximum_dialogue_token_count": max(
            row["dialogue_contract"]["token_count"] for row in episodes
        ),
        "fixed_split_counts": dict(sorted(split_counts.items())),
        "counterfactual_pair_count": len(counterfactuals),
        "dialogue_negative_tier_counts": dict(
            sorted(Counter(row["dialogue_shuffled"]["selection_tier"] for row in counterfactuals).items())
        ),
        "directive_negative_tier_counts": dict(
            sorted(Counter(row["directive_shuffled"]["selection_tier"] for row in counterfactuals).items())
        ),
    }
    if scale != recomputed:
        raise ValueError("release dataset_scale does not match recomputed values")

    return {
        "schema_version": "1.0.0",
        "artifact_kind": AUDIT_KIND,
        "passed": True,
        "release_dir": str(release_dir),
        "provenance_lock_sha256": _sha256_file(lock_path),
        "train_ready_manifest_sha256": _sha256_file(train_path),
        "counterfactual_pair_manifest_sha256": _sha256_file(counterfactual_path),
        "dialogue_unavailable_manifest_sha256": _sha256_file(exclusion_path),
        "verified_scale": recomputed,
        "verified_invariants": {
            "episode_contract": True,
            "counterfactual_record_hashes": True,
            "counterfactual_episode_bindings": True,
            "split_isolation": True,
            "source_group_isolation": True,
            "true_text_negatives": True,
            "fixed_six_second_training_unit": False,
            "all_base_motion_accounted_for": True,
        },
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = verify_release(args.release_dir)
    if args.output is not None:
        _atomic_json(args.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
