#!/usr/bin/env python3
"""Train resumable short head-loss candidates on one strict BEAT split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from upper_body_skeleton.ula_v2_18d_posttrain import (
    resolve_posttrain_config,
    train_18d_posttrain,
)
from upper_body_skeleton.ula_v2_18d_head import attach_condition_cache
from tools import train_ula_v2_18d_staged as staged


def read_config(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("loss sweep config must use schema_version=1")
    if value.get("audio_policy") != "disabled_not_loaded":
        raise ValueError("head loss sweep requires audio_policy=disabled_not_loaded")
    candidates = value.get("candidates")
    if not isinstance(candidates, dict) or len(candidates) < 3:
        raise ValueError("head loss sweep requires at least three candidates")
    return value


def _completed(output: Path, config: dict) -> bool:
    summary_path = output / "training_summary.json"
    checkpoint_path = output / "ula_fm_checkpoint.pt"
    if not summary_path.is_file() or not checkpoint_path.is_file():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return bool(
        summary.get("target_steps") == config["steps"]
        and (
            summary.get("completed_steps") == config["steps"]
            or summary.get("stopped_early") is True
        )
        and summary.get("training_policy") == "head_projection_only"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = read_config(args.config)
    staged_config = staged.read_config(Path(config["staged_config"]))
    raw_episodes = staged.load_motion_sources(staged_config)
    episodes, quarantine, _, _, _ = staged.temporal_quarantine(
        staged_config, raw_episodes
    )
    expected_quarantine = Path(staged.stage_paths(staged_config)["temporal_summary"])
    stored_quarantine = json.loads(expected_quarantine.read_text(encoding="utf-8"))
    if stored_quarantine.get("contract_sha256") != quarantine["contract_sha256"]:
        raise ValueError("temporal quarantine contract changed before the loss sweep")
    episodes = attach_condition_cache(
        episodes,
        config["condition_cache"],
        allow_unsafe_metadata=bool(config.get("allow_unsafe_condition_cache", False)),
    )
    for episode in episodes:
        episode["dataset_source"] = str(config["dataset_source"])
    output_root = Path(config["output_root"])
    results = {}
    for name, loss in config["candidates"].items():
        training = dict(config["common_training"])
        training["loss"] = dict(loss)
        training["allow_unsafe_training_data"] = bool(
            config.get("allow_unsafe_training_data", False)
        )
        training["training_scope"] = staged_config["training_scope"]
        training["formal_training_enabled"] = staged_config[
            "formal_training_enabled"
        ]
        training["temporal_unit_policy"] = "fixed_window_experimental"
        training = resolve_posttrain_config(training)
        output = output_root / name
        if not _completed(output, training):
            last = output / "last.pt"
            if output.exists() and any(output.iterdir()):
                if not last.is_file():
                    raise RuntimeError(f"incomplete candidate {name} has no last.pt")
                training["resume_from"] = str(last.resolve())
            summary = train_18d_posttrain(
                initial_checkpoint_path=config["initial_checkpoint"],
                beat_episodes=episodes,
                output_dir=output,
                config=training,
            )
        else:
            summary = json.loads(
                (output / "training_summary.json").read_text(encoding="utf-8")
            )
        results[name] = {
            "checkpoint": summary["checkpoint"],
            "best_step": summary["best_step"],
            "initial_validation": summary["initial_validation"],
            "final_validation": summary["final_validation"],
            "frozen_weight_max_abs_error": summary["frozen_weight_max_abs_error"],
        }
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / "sweep_training_summary.json"
    result_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
