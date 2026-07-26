#!/usr/bin/env python3
"""Run auditable BEAT 18D interaction-domain post-training with Kimodo replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from upper_body_skeleton.ula_v2_18d_posttrain import (
    load_attached_beat_episodes,
    load_kimodo_replay_episodes,
    train_18d_posttrain,
)
from upper_body_skeleton.ula_v2_18d_head import validate_qwen_checkpoint_for_generator
from upper_body_skeleton.ula_v2_18d_head import sha256_file


def _read_config(path: Path | None) -> dict:
    if path is None:
        return {}
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML configs require PyYAML") from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("post-training config must contain a mapping")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument(
        "--beat-manifest",
        type=Path,
        action="append",
        help="Repeat to provide the complete checkpoint-bound formal manifest set.",
    )
    parser.add_argument("--condition-cache", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--kimodo-dataset-dir", type=Path)
    parser.add_argument("--kimodo-split-checkpoint", type=Path)
    parser.add_argument("--qwen-checkpoint", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--device")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="Explicitly permit unadjudicated BEAT motion for an unsafe experiment.",
    )
    parser.add_argument(
        "--allow-unsafe-condition-cache",
        action="store_true",
        help="Explicitly permit an unversioned condition cache for an unsafe experiment.",
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _value(args, config, argument, config_name=None):
    value = getattr(args, argument)
    return value if value is not None else config.get(config_name or argument)


def resolve_bound_motion_sources(
    initial_payload: dict, requested_manifests=None
) -> list[dict]:
    """Resolve the complete immutable motion-manifest set for formal training."""
    requested = [Path(value).resolve() for value in (requested_manifests or [])]
    random_initialization = initial_payload.get("random_initialization")
    if not random_initialization:
        if len(requested) != 1:
            raise ValueError("non-random post-training requires exactly one beat manifest")
        if not requested[0].is_file():
            raise FileNotFoundError(f"BEAT manifest is missing: {requested[0]}")
        return [{"manifest": requested[0]}]

    stored = (initial_payload.get("sources") or {}).get("motion_manifests")
    if not isinstance(stored, list) or not stored:
        raise ValueError("random-init checkpoint has no bound motion manifests")
    stored_by_hash = {}
    for source in stored:
        if not isinstance(source, dict):
            raise ValueError("random-init motion source binding must be an object")
        digest = source.get("manifest_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("random-init motion source has no manifest SHA256")
        if digest in stored_by_hash:
            raise ValueError("random-init motion manifest hashes must be unique")
        for field in ("dataset_source", "speaker_namespace", "source_group_namespace"):
            if not str(source.get(field) or "").strip():
                raise ValueError(f"random-init motion source is missing {field}")
        stored_by_hash[digest] = dict(source)

    if requested:
        requested_by_hash = {}
        for path in requested:
            if not path.is_file():
                raise FileNotFoundError(f"formal motion manifest is missing: {path}")
            digest = sha256_file(path)
            if digest in requested_by_hash:
                raise ValueError("requested formal motion manifests contain duplicate content")
            requested_by_hash[digest] = path
        if set(requested_by_hash) != set(stored_by_hash):
            raise ValueError(
                "formal training manifests must exactly match the complete random-init source set"
            )
    else:
        requested_by_hash = {}
        for digest, source in stored_by_hash.items():
            path_value = source.get("manifest")
            if not str(path_value or "").strip():
                raise ValueError(
                    "random-init source has no manifest path; provide beat_manifests explicitly"
                )
            path = Path(path_value).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"checkpoint-bound motion manifest is missing: {path}")
            if sha256_file(path) != digest:
                raise ValueError(f"checkpoint-bound motion manifest changed: {path}")
            requested_by_hash[digest] = path

    return [
        dict(source) | {"manifest": requested_by_hash[digest]}
        for digest, source in stored_by_hash.items()
    ]


def main():
    args = parse_args()
    loaded = _read_config(args.config)
    initial_checkpoint = _value(args, loaded, "initial_checkpoint")
    cli_beat_manifests = list(args.beat_manifest or [])
    beat_manifest = loaded.get("beat_manifest")
    beat_manifests = loaded.get("beat_manifests")
    if beat_manifests is not None and (
        not isinstance(beat_manifests, list) or not beat_manifests
    ):
        raise ValueError("beat_manifests must be a non-empty list when provided")
    if beat_manifest not in (None, "") and beat_manifests is not None:
        raise ValueError("use beat_manifest or beat_manifests, not both")
    condition_cache = _value(args, loaded, "condition_cache")
    output_dir = _value(args, loaded, "output_dir")
    missing = [
        name
        for name, value in (
            ("initial_checkpoint", initial_checkpoint),
            ("condition_cache", condition_cache),
            ("output_dir", output_dir),
        )
        if value in (None, "")
    ]
    if missing:
        raise ValueError(f"missing required post-training paths: {missing}")

    allow_unreviewed = bool(args.allow_unreviewed or loaded.get("allow_unreviewed", False))
    allow_unsafe_cache = bool(
        args.allow_unsafe_condition_cache
        or loaded.get("allow_unsafe_condition_cache", False)
    )
    import torch

    initial_payload = torch.load(initial_checkpoint, map_location="cpu", weights_only=True)
    requested_manifests = cli_beat_manifests or beat_manifests or (
        [beat_manifest] if beat_manifest not in (None, "") else []
    )
    source_bindings = resolve_bound_motion_sources(
        initial_payload, requested_manifests=requested_manifests
    )
    beat = []
    seen_clip_ids = set()
    for source_binding in source_bindings:
        loaded_episodes = load_attached_beat_episodes(
            source_binding["manifest"],
            condition_cache,
            allow_unreviewed=allow_unreviewed,
            allow_unsafe_condition_cache=allow_unsafe_cache,
            dataset_source=source_binding.get("dataset_source"),
            speaker_namespace=source_binding.get("speaker_namespace"),
            source_group_namespace=source_binding.get("source_group_namespace"),
        )
        duplicate_ids = seen_clip_ids.intersection(
            str(episode["clip_id"]) for episode in loaded_episodes
        )
        if duplicate_ids:
            raise ValueError(
                "formal motion manifests contain duplicate clip IDs: "
                f"{sorted(duplicate_ids)[:5]}"
            )
        seen_clip_ids.update(str(episode["clip_id"]) for episode in loaded_episodes)
        beat.extend(loaded_episodes)

    kimodo_dataset = _value(args, loaded, "kimodo_dataset_dir")
    kimodo_split = _value(args, loaded, "kimodo_split_checkpoint")
    qwen_checkpoint = _value(args, loaded, "qwen_checkpoint")
    replay_paths = (kimodo_dataset, kimodo_split, qwen_checkpoint)
    if any(value not in (None, "") for value in replay_paths) and not all(
        value not in (None, "") for value in replay_paths
    ):
        raise ValueError(
            "Kimodo replay requires kimodo_dataset_dir, kimodo_split_checkpoint, "
            "and qwen_checkpoint together"
        )
    replay = []
    replay_provenance = {}
    if all(value not in (None, "") for value in replay_paths):
        validate_qwen_checkpoint_for_generator(initial_payload, qwen_checkpoint)
        replay_device = args.device or loaded.get("device", "cpu")
        if replay_device == "auto":
            replay_device = "cuda" if torch.cuda.is_available() else "cpu"
        replay, replay_provenance = load_kimodo_replay_episodes(
            kimodo_dataset,
            kimodo_split,
            qwen_checkpoint,
            device=replay_device,
            local_files_only=not args.allow_download,
        )

    training_config = dict(loaded.get("training") or {})
    for name in ("steps", "batch_size", "lr", "device"):
        value = getattr(args, name)
        if value is not None:
            training_config[name] = value
        elif name in loaded and name not in training_config:
            training_config[name] = loaded[name]
    resume_from = args.resume_from or loaded.get("resume_from") or training_config.get(
        "resume_from"
    )
    if resume_from is not None:
        training_config["resume_from"] = str(resume_from)
    training_config["overwrite"] = bool(
        args.overwrite or loaded.get("overwrite", False) or training_config.get("overwrite", False)
    )
    training_config["allow_unsafe_training_data"] = bool(
        allow_unreviewed
        or allow_unsafe_cache
        or loaded.get("allow_unsafe_training_data", False)
        or training_config.get("allow_unsafe_training_data", False)
    )
    result = train_18d_posttrain(
        initial_checkpoint_path=initial_checkpoint,
        beat_episodes=beat,
        kimodo_replay_episodes=replay,
        replay_provenance=replay_provenance,
        output_dir=output_dir,
        config=training_config,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
