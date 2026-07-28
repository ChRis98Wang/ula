#!/usr/bin/env python3
"""Prepare, smoke-test, and train a BEAT2 dual-text action generator."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from upper_body_skeleton.ula_v2_18d_head import (
    load_contract_checkpoint,
    validate_condition_cache_for_generator,
)
from upper_body_skeleton.ula_v2_18d_posttrain import (
    _counterfactual_conditions_for_episodes,
    load_attached_beat_episodes,
    masked_18d_objective,
    native_variable_length_batch_tensors,
    train_18d_posttrain,
)
from upper_body_skeleton.ula_v2_dialogue_action_episode import (
    load_dialogue_action_v11_episodes,
    load_dialogue_action_v11_records,
    sha256_file,
)
from upper_body_skeleton.ula_v2_dialogue_action_training import (
    build_dual_text_condition_cache,
    migrate_v3_foundation_to_v4_dual_text,
)


def _read_config(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("dialogue/action config must contain an object")
    required = {
        "source_checkpoint",
        "qwen_checkpoint",
        "manifest",
        "counterfactual_manifest",
        "migrated_checkpoint",
        "condition_cache",
        "smoke_report",
        "training_output_dir",
        "training",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"dialogue/action config is missing {missing}")
    return value


def prepare(config: dict) -> dict:
    migrated = Path(config["migrated_checkpoint"]).resolve()
    cache = Path(config["condition_cache"]).resolve()
    if not migrated.exists():
        migration = migrate_v3_foundation_to_v4_dual_text(
            config["source_checkpoint"],
            config["qwen_checkpoint"],
            migrated,
        )
    else:
        _, checkpoint = load_contract_checkpoint(migrated, expected_action_dim=18)
        migration = {
            "checkpoint": str(migrated),
            "checkpoint_sha256": sha256_file(migrated),
            "dual_text_conditioning_contract": checkpoint[
                "dual_text_conditioning_contract"
            ],
            "reused_existing": True,
        }
    if not cache.exists():
        episodes = load_dialogue_action_v11_records(config["manifest"])
        metadata = build_dual_text_condition_cache(
            episodes,
            config["counterfactual_manifest"],
            config["qwen_checkpoint"],
            migrated,
            cache,
            device=str(config.get("condition_cache_device") or "auto"),
            batch_size=int(config.get("condition_cache_batch_size") or 16),
        )
        del episodes
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        metadata_path = cache.with_suffix(cache.suffix + ".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("cache_sha256") != sha256_file(cache):
            raise ValueError("existing dual-text cache hash changed")
    checkpoint = torch.load(migrated, map_location="cpu", weights_only=True)
    validation = validate_condition_cache_for_generator(
        checkpoint,
        metadata,
        generator_checkpoint_path=migrated,
    )
    cache_summary = {
        key: value for key, value in metadata.items() if key != "episodes"
    }
    cache_summary["episode_binding_count"] = len(metadata.get("episodes") or [])
    return {
        "migration": migration,
        "condition_cache": cache_summary,
        "validation": validation,
    }


def gradient_smoke(config: dict) -> dict:
    report_path = Path(config["smoke_report"]).resolve()
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("passed") is not True:
            raise ValueError("existing gradient smoke did not pass")
        return report
    episodes = load_attached_beat_episodes(
        config["manifest"],
        config["condition_cache"],
        allow_unreviewed=False,
        allow_unsafe_condition_cache=False,
        dataset_source="beat2_dialogue_action_v11",
    )
    candidates = sorted(
        (row for row in episodes if row["fixed_split_assignment"] == "train"),
        key=lambda row: (len(row["actions"]), row["clip_id"]),
    )
    selected = candidates[:1]
    first_action_id = (selected[0].get("action_summary") or {}).get("action_id")
    if first_action_id is None:
        selected = candidates[:2]
    else:
        selected.extend(
            row
            for row in candidates[1:]
            if (row.get("action_summary") or {}).get("action_id") != first_action_id
        )
        selected = selected[:2]
    if len(selected) != 2:
        raise ValueError("gradient smoke requires two train episodes")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_contract_checkpoint(
        config["migrated_checkpoint"], expected_action_dim=18, device=device
    )
    model.train()
    actions, conditions, masks, durations, frame_valid = native_variable_length_batch_tensors(
        selected,
        buckets=[48, 64, 96, 128, 192, 256, 384, 512],
        action_stats=checkpoint["action_stats"],
        device=device,
    )
    negatives = _counterfactual_conditions_for_episodes(selected, device=device)
    losses = masked_18d_objective(
        model,
        actions,
        conditions,
        masks,
        durations,
        loss_weights={
            "flow": 1.0,
            "position": 0.25,
            "condition_contrastive": float(
                config["training"]["loss"]["condition_contrastive"]
            ),
        },
        frame_valid_mask=frame_valid,
        counterfactual_conditions=negatives,
        condition_contrastive_margin=float(
            config["training"]["condition_contrastive_margin"]
        ),
    )
    losses["total"].backward()
    directive_grad = float(
        model.action_directive_condition[0].weight.grad.detach().norm().cpu()
    )
    dialogue_grad = float(model.dialogue_condition[0].weight.grad.detach().norm().cpu())
    report = {
        "artifact_kind": str(
            config.get("gradient_smoke_artifact_kind")
            or "beat2_dialogue_action_v11_gradient_smoke_v1"
        ),
        "passed": bool(
            torch.isfinite(losses["total"]) and directive_grad > 0 and dialogue_grad > 0
        ),
        "checkpoint_sha256": sha256_file(config["migrated_checkpoint"]),
        "condition_cache_sha256": sha256_file(config["condition_cache"]),
        "clip_ids": [row["clip_id"] for row in selected],
        "action_ids": [
            (row.get("action_summary") or {}).get("action_id") for row in selected
        ],
        "losses": {
            name: float(value.detach().cpu()) for name, value in losses.items()
        },
        "action_directive_projection_gradient_norm": directive_grad,
        "dialogue_projection_gradient_norm": dialogue_grad,
        "both_text_roles_receive_gradient": directive_grad > 0 and dialogue_grad > 0,
    }
    if not report["passed"]:
        raise RuntimeError(f"dual-text gradient smoke failed: {report}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def train(config: dict) -> dict:
    episodes = load_attached_beat_episodes(
        config["manifest"],
        config["condition_cache"],
        allow_unreviewed=False,
        allow_unsafe_condition_cache=False,
        dataset_source=str(
            config.get("dataset_source") or "beat2_dialogue_action_v11"
        ),
    )
    return train_18d_posttrain(
        initial_checkpoint_path=config["migrated_checkpoint"],
        beat_episodes=episodes,
        output_dir=config["training_output_dir"],
        kimodo_replay_episodes=(),
        replay_provenance={},
        config=config["training"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("prepare", "smoke", "train", "all"), default="all"
    )
    args = parser.parse_args(argv)
    config = _read_config(args.config.resolve())
    result = {}
    if args.stage in {"prepare", "all"}:
        result["prepare"] = prepare(config)
    if args.stage in {"smoke", "all"}:
        result["smoke"] = gradient_smoke(config)
    if args.stage in {"train", "all"}:
        result["train"] = train(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
