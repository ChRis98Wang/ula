#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from upper_body_skeleton.semantic_adapter import (
    DEFAULT_BEHAVIOR_INSTRUCTION,
    DEFAULT_EMOTION_INSTRUCTION,
    DEFAULT_QWEN_MODEL,
)
from upper_body_skeleton.kimodo_semantics import KIMODO_BEHAVIOR_IDS, KIMODO_EMOTION_IDS


DEFAULTS = {
    "model_name": DEFAULT_QWEN_MODEL,
    "revision": None,
    "behavior_instruction": DEFAULT_BEHAVIOR_INSTRUCTION,
    "emotion_instruction": DEFAULT_EMOTION_INSTRUCTION,
    "local_files_only": False,
    "attention_backend": "eager",
    "max_length": 128,
    "qwen_component_dim": 128,
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_top_layers": 4,
    "lora_target_projections": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "latent_dim": 128,
    "projection_hidden_dim": 256,
    "projection_dropout": 0.1,
    "motion_trainable_prefixes": [
        "backbone.6.",
        "backbone.7.",
        "projection.",
        "behavior_head.",
        "emotion_head.",
        "descriptor_head.",
    ],
    "steps": 100_000,
    "batch_size": 24,
    "eval_batch_size": 16,
    "qwen_lr": 5e-6,
    "projector_lr": 1e-4,
    "motion_lr": 1e-5,
    "weight_decay": 1e-4,
    "adam_eps": 1e-6,
    "warmup_steps": 500,
    "lr_decay_steps": 100_000,
    "minimum_lr_ratio": 0.1,
    "max_grad_norm": 1.0,
    "temperature": 0.07,
    "alignment_weight": 1.0,
    "text_behavior_weight": 0.5,
    "text_emotion_weight": 0.5,
    "motion_metric_weight": 0.25,
    "motion_anchor_weight": 0.1,
    "variance_weight": 1.0,
    "covariance_weight": 0.1,
    "minimum_effective_rank": 3.0,
    "seed": 7,
    "device": "auto",
    "max_episodes": 1620,
    "log_interval": 25,
    "eval_interval": 500,
    "checkpoint_interval": 500,
    "resume_from": None,
    "overwrite": False,
}

REQUIRED = {
    "dataset_dir",
    "prompt_csv",
    "paraphrases_json",
    "motion_checkpoint",
    "output_dir",
    "revision",
}


def load_train_config(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    if yaml is None:
        raise RuntimeError("PyYAML is required for YAML configs")
    return yaml.safe_load(text) or {}


def validated_train_config(config):
    config = dict(config)
    unknown = sorted(set(config) - set(DEFAULTS) - REQUIRED)
    if unknown:
        raise ValueError(f"unknown Qwen motion latent config keys: {unknown}")
    values = DEFAULTS | config
    missing = sorted(key for key in REQUIRED if values.get(key) in (None, ""))
    if missing:
        raise ValueError(f"Qwen motion latent config requires: {missing}")
    integer_positive = (
        "steps",
        "batch_size",
        "eval_batch_size",
        "max_length",
        "qwen_component_dim",
        "lora_rank",
        "lora_alpha",
        "lora_top_layers",
        "latent_dim",
        "projection_hidden_dim",
        "lr_decay_steps",
        "log_interval",
        "eval_interval",
        "checkpoint_interval",
    )
    for key in integer_positive:
        values[key] = int(values[key])
        if values[key] <= 0:
            raise ValueError(f"{key} must be positive")
    values["warmup_steps"] = int(values["warmup_steps"])
    values["seed"] = int(values["seed"])
    values["max_episodes"] = None if values["max_episodes"] is None else int(values["max_episodes"])
    for key in (
        "qwen_lr",
        "projector_lr",
        "motion_lr",
        "weight_decay",
        "adam_eps",
        "minimum_lr_ratio",
        "max_grad_norm",
        "temperature",
        "alignment_weight",
        "text_behavior_weight",
        "text_emotion_weight",
        "motion_metric_weight",
        "motion_anchor_weight",
        "variance_weight",
        "covariance_weight",
        "minimum_effective_rank",
        "lora_dropout",
        "projection_dropout",
    ):
        values[key] = float(values[key])
    if values["attention_backend"] != "eager":
        raise ValueError("attention_backend must be eager on the current RTX 5090 training host")
    if values["warmup_steps"] < 0 or values["warmup_steps"] >= values["lr_decay_steps"]:
        raise ValueError("warmup_steps must be non-negative and smaller than lr_decay_steps")
    if not 0.0 < values["minimum_lr_ratio"] <= 1.0:
        raise ValueError("minimum_lr_ratio must be in (0, 1]")
    if values["batch_size"] > len(KIMODO_BEHAVIOR_IDS) * len(KIMODO_EMOTION_IDS):
        raise ValueError("batch_size cannot exceed the 162 unique Kimodo semantic groups")
    values["local_files_only"] = bool(values["local_files_only"])
    values["overwrite"] = bool(values["overwrite"])
    values["resume_from"] = None if values["resume_from"] in (None, "") else str(values["resume_from"])
    for key in ("dataset_dir", "prompt_csv", "paraphrases_json", "motion_checkpoint", "output_dir"):
        values[key] = str(values[key])
    values["model_name"] = str(values["model_name"])
    values["revision"] = str(values["revision"])
    values["device"] = str(values["device"])
    values["behavior_instruction"] = str(values["behavior_instruction"])
    values["emotion_instruction"] = str(values["emotion_instruction"])
    values["lora_target_projections"] = [str(value) for value in values["lora_target_projections"]]
    values["motion_trainable_prefixes"] = [str(value) for value in values["motion_trainable_prefixes"]]
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train Qwen LoRA against the Kimodo motion latent space")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--resume-from")
    parser.add_argument("--overwrite", action="store_true", default=None)
    args = parser.parse_args(argv)
    config = load_train_config(args.config)
    for key in ("output_dir", "steps", "device", "resume_from", "overwrite"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    return validated_train_config(config)


def main(argv=None):
    from upper_body_skeleton.cross_modal_latent import train_qwen_motion_alignment

    summary = train_qwen_motion_alignment(parse_args(argv))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
