#!/usr/bin/env python3
"""Build a hash-bound multisource lock for reviewed expression-turn manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.train_ula_v2_18d_formal_from_scratch import (
    EXPRESSION_TURN_V8_PROVENANCE_LOCK_KIND,
)
from upper_body_skeleton.ula_v2_expression_turn_episode import (
    FORMAL_EPISODE_CONTRACT,
    load_expression_turn_v8_episodes,
)


SPEC_ARTIFACT_KIND = "ula_v2_expression_turn_v8_provenance_lock_spec_v1"
DURATION_POLICY = "complete_expression_arc_variable_length_no_fixed_duration_v1"
MINIMUM_SCALE_FIELDS = {
    "episode_count",
    "semantic_conditioned_count",
    "expressive_conditioned_count",
    "unique_prompt_count",
    "source_group_count",
    "speaker_count",
    "distinct_frame_count_count",
    "total_sample_span_sec",
    "duration_span_sec",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _resolve_file(value: object, *, spec_path: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = spec_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{field} is missing: {path}")
    return path


def _artifact(path: Path, *, role: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "role": role,
    }


def build_lock(spec: Mapping[str, Any], *, spec_path: Path) -> dict[str, Any]:
    """Validate every train manifest and return a deterministic provenance lock."""

    if spec.get("schema_version") != 1 or spec.get("artifact_kind") != SPEC_ARTIFACT_KIND:
        raise ValueError("expression-turn provenance lock spec contract is invalid")
    sources = spec.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("provenance lock spec requires at least one source")

    locked_artifacts: dict[str, dict[str, Any]] = {}
    validations: list[dict[str, Any]] = []
    all_episodes: list[dict[str, Any]] = []

    def add_artifact(key: object, path: Path, *, role: str) -> None:
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{role} artifact key is missing")
        key = key.strip()
        if key in locked_artifacts:
            raise ValueError(f"duplicate provenance artifact key: {key}")
        locked_artifacts[key] = _artifact(path, role=role)

    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise ValueError(f"sources[{index}] must be an object")
        dataset_source = str(source.get("dataset_source") or "").strip()
        if not dataset_source:
            raise ValueError(f"sources[{index}].dataset_source is required")
        inventory = _resolve_file(
            source.get("source_inventory"),
            spec_path=spec_path,
            field=f"sources[{index}].source_inventory",
        )
        manifest = _resolve_file(
            source.get("manifest"),
            spec_path=spec_path,
            field=f"sources[{index}].manifest",
        )
        review_summary = _resolve_file(
            source.get("review_summary"),
            spec_path=spec_path,
            field=f"sources[{index}].review_summary",
        )
        add_artifact(
            source.get("inventory_artifact_key"),
            inventory,
            role="source_inventory",
        )
        add_artifact(
            source.get("manifest_artifact_key"),
            manifest,
            role="strict_train_manifest",
        )
        add_artifact(
            source.get("review_artifact_key"),
            review_summary,
            role="independent_blind_review_summary",
        )

        episodes = load_expression_turn_v8_episodes(manifest)
        all_episodes.extend(dict(episode) for episode in episodes)
        tier_counts = Counter(
            str(episode["training_qualification_tier"]) for episode in episodes
        )
        prompt_profile_counts = Counter(
            str(episode["prompt_semantics_profile"]) for episode in episodes
        )
        frame_counts = [int(episode["actions"].shape[0]) for episode in episodes]
        sample_spans = [
            float((frame_count - 1) / float(episode.get("fps") or 30.0))
            for frame_count, episode in zip(frame_counts, episodes, strict=True)
        ]
        semantic_episodes = [
            episode
            for episode in episodes
            if str(episode["training_qualification_tier"])
            in {"semantic_conditioning", "expressive_conditioning"}
        ]
        expressive_episodes = [
            episode
            for episode in episodes
            if str(episode["training_qualification_tier"])
            == "expressive_conditioning"
        ]
        validations.append(
            {
                "dataset_source": dataset_source,
                "inventory_artifact_key": source["inventory_artifact_key"],
                "manifest_artifact_key": source["manifest_artifact_key"],
                "review_artifact_key": source["review_artifact_key"],
                "manifest_records": len(episodes),
                "qualification_tier_counts": dict(sorted(tier_counts.items())),
                "prompt_semantics_profile_counts": dict(
                    sorted(prompt_profile_counts.items())
                ),
                "frame_count_min": min(frame_counts),
                "frame_count_max": max(frame_counts),
                "distinct_frame_count_count": len(set(frame_counts)),
                "sample_span_sec_min": min(sample_spans),
                "sample_span_sec_median": float(statistics.median(sample_spans)),
                "sample_span_sec_max": max(sample_spans),
                "total_sample_span_sec": float(sum(sample_spans)),
                "semantic_conditioned_count": len(semantic_episodes),
                "expressive_conditioned_count": len(expressive_episodes),
                "unique_prompt_count": len(
                    {
                        str(episode.get("prompt_sha256") or episode.get("prompt") or "")
                        for episode in semantic_episodes
                        if str(
                            episode.get("prompt_sha256")
                            or episode.get("prompt")
                            or ""
                        ).strip()
                    }
                ),
                "source_group_count": len(
                    {
                        str(episode.get("source_group_key"))
                        for episode in episodes
                        if str(episode.get("source_group_key") or "").strip()
                    }
                ),
                "speaker_count": len(
                    {
                        str(episode.get("speaker_key"))
                        for episode in episodes
                        if str(episode.get("speaker_key") or "").strip()
                    }
                ),
                "full_native_sequence_loader_validated": True,
            }
        )

    frame_counts = [int(episode["actions"].shape[0]) for episode in all_episodes]
    sample_spans = [
        float((frame_count - 1) / float(episode.get("fps") or 30.0))
        for frame_count, episode in zip(frame_counts, all_episodes, strict=True)
    ]
    semantic_episodes = [
        episode
        for episode in all_episodes
        if str(episode["training_qualification_tier"])
        in {"semantic_conditioning", "expressive_conditioning"}
    ]
    expressive_episodes = [
        episode
        for episode in all_episodes
        if str(episode["training_qualification_tier"])
        == "expressive_conditioning"
    ]
    dataset_scale = {
        "episode_count": len(all_episodes),
        "semantic_conditioned_count": len(semantic_episodes),
        "expressive_conditioned_count": len(expressive_episodes),
        "unique_prompt_count": len(
            {
                str(episode.get("prompt_sha256") or episode.get("prompt") or "")
                for episode in semantic_episodes
                if str(
                    episode.get("prompt_sha256") or episode.get("prompt") or ""
                ).strip()
            }
        ),
        "source_group_count": len(
            {
                f"{episode.get('dataset_source')}:{episode.get('source_group_key')}"
                for episode in all_episodes
                if str(episode.get("source_group_key") or "").strip()
            }
        ),
        "speaker_count": len(
            {
                f"{episode.get('dataset_source')}:{episode.get('speaker_key')}"
                for episode in all_episodes
                if str(episode.get("speaker_key") or "").strip()
            }
        ),
        "distinct_frame_count_count": len(set(frame_counts)),
        "frame_count_min": min(frame_counts),
        "frame_count_max": max(frame_counts),
        "sample_span_sec_min": min(sample_spans),
        "sample_span_sec_median": float(statistics.median(sample_spans)),
        "sample_span_sec_max": max(sample_spans),
        "total_sample_span_sec": float(sum(sample_spans)),
        "duration_span_sec": float(max(sample_spans) - min(sample_spans)),
    }
    minimum_scale = spec.get("minimum_training_scale") or {}
    if not isinstance(minimum_scale, Mapping):
        raise ValueError("minimum_training_scale must be an object")
    unknown_scale_fields = set(minimum_scale).difference(MINIMUM_SCALE_FIELDS)
    if unknown_scale_fields:
        raise ValueError(
            f"minimum_training_scale has unsupported fields: {sorted(unknown_scale_fields)}"
        )
    normalized_minimum_scale: dict[str, float | int] = {}
    deficits: list[str] = []
    for field, required in minimum_scale.items():
        if isinstance(required, bool) or not isinstance(required, (int, float)):
            raise ValueError(f"minimum_training_scale.{field} must be numeric")
        if not math.isfinite(float(required)) or float(required) < 0.0:
            raise ValueError(f"minimum_training_scale.{field} must be finite and non-negative")
        normalized = float(required) if field.endswith("_sec") else int(required)
        if not field.endswith("_sec") and float(required) != normalized:
            raise ValueError(f"minimum_training_scale.{field} must be an integer")
        normalized_minimum_scale[field] = normalized
        if float(dataset_scale[field]) < float(normalized):
            deficits.append(f"{field}:{dataset_scale[field]}<{normalized}")
    if deficits:
        raise ValueError("dataset scale gate failed: " + ", ".join(deficits))

    evidence = spec.get("evidence_artifacts")
    if not isinstance(evidence, Mapping) or not evidence:
        raise ValueError("provenance lock spec requires independent evidence artifacts")
    for key, value in sorted(evidence.items()):
        path = _resolve_file(
            value,
            spec_path=spec_path,
            field=f"evidence_artifacts.{key}",
        )
        add_artifact(key, path, role="additional_training_evidence")

    return {
        "schema_version": 1,
        "artifact_kind": EXPRESSION_TURN_V8_PROVENANCE_LOCK_KIND,
        "formal_episode_contract": FORMAL_EPISODE_CONTRACT,
        "duration_policy": DURATION_POLICY,
        "source_count": len(sources),
        "accepted_for_training": True,
        "formal_release_allowed": True,
        "dataset_scale": dataset_scale,
        "minimum_training_scale": normalized_minimum_scale,
        "scale_gate_passed": True,
        "license_gate": {
            "authority_policy": "separate_per_source_license_gates_v1",
            "formal_release_blocked": False,
        },
        "locked_artifacts": dict(sorted(locked_artifacts.items())),
        "source_validations": validations,
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "spec_path": str(spec_path),
            "spec_sha256": sha256_file(spec_path),
        },
    }


def write_lock(spec_path: Path, output: Path) -> dict[str, Any]:
    spec_path = spec_path.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite provenance lock: {output}")
    lock = build_lock(_read_json(spec_path), spec_path=spec_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return lock | {"output": str(output), "output_sha256": sha256_file(output)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            write_lock(args.spec, args.output),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
