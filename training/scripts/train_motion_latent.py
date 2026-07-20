#!/usr/bin/env python3
import argparse
import json
from argparse import Namespace
import os
from pathlib import Path
import sys


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

try:
    import yaml
except ImportError:  # pragma: no cover - JSON and direct CLI remain available
    yaml = None


DEFAULTS = {
    "steps": 20_000,
    "batch_size": 64,
    "lr": 3e-4,
    "latent_dim": 128,
    "hidden_dim": 128,
    "device": "auto",
    "seed": 7,
    "log_interval": 100,
    "prototypes_per_group": 1,
    "weight_decay": 1e-4,
    "max_episodes": None,
    "deterministic": True,
    "overwrite": False,
}


def load_train_config(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    if yaml is None:
        raise RuntimeError("PyYAML is required for YAML configs; use JSON or direct CLI options")
    return yaml.safe_load(text) or {}


def _required(config, key):
    value = config.get(key)
    if value in (None, ""):
        raise ValueError(f"motion latent training config requires {key}")
    return value


def training_args_from_config(config):
    allowed = set(DEFAULTS) | {"dataset_dir", "output_dir"}
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"unknown motion latent config keys: {unknown}")
    values = DEFAULTS | {key: config[key] for key in DEFAULTS if key in config}
    return Namespace(
        dataset_dir=str(_required(config, "dataset_dir")),
        output_dir=str(_required(config, "output_dir")),
        steps=int(values["steps"]),
        batch_size=int(values["batch_size"]),
        lr=float(values["lr"]),
        latent_dim=int(values["latent_dim"]),
        hidden_dim=int(values["hidden_dim"]),
        device=str(values["device"]),
        seed=int(values["seed"]),
        log_interval=int(values["log_interval"]),
        prototypes_per_group=int(values["prototypes_per_group"]),
        weight_decay=float(values["weight_decay"]),
        max_episodes=None if values["max_episodes"] is None else int(values["max_episodes"]),
        deterministic=bool(values["deterministic"]),
        overwrite=bool(values["overwrite"]),
    )


def parse_training_args(argv=None):
    parser = argparse.ArgumentParser(description="Train and diagnose the Kimodo motion metric latent space")
    parser.add_argument("--config")
    parser.add_argument("--dataset-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--latent-dim", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--prototypes-per-group", type=int)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--max-episodes", type=int)
    deterministic = parser.add_mutually_exclusive_group()
    deterministic.add_argument("--deterministic", dest="deterministic", action="store_true", default=None)
    deterministic.add_argument("--non-deterministic", dest="deterministic", action="store_false")
    parser.add_argument("--overwrite", action="store_true", default=None)
    cli = parser.parse_args(argv)

    config = load_train_config(cli.config) if cli.config else {}
    for key, value in vars(cli).items():
        if key != "config" and value is not None:
            config[key] = value
    return training_args_from_config(config)


def main(argv=None):
    from upper_body_skeleton.motion_latent import load_motion_latent_episodes, train_motion_latent

    args = parse_training_args(argv)
    episodes = load_motion_latent_episodes(args.dataset_dir, max_episodes=args.max_episodes)
    if not episodes:
        raise SystemExit("no episodes loaded")
    summary = train_motion_latent(
        episodes,
        output_dir=args.output_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        device=args.device,
        seed=args.seed,
        log_interval=args.log_interval,
        prototypes_per_group=args.prototypes_per_group,
        weight_decay=args.weight_decay,
        deterministic=args.deterministic,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
