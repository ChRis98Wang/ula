#!/usr/bin/env python3
import argparse
import json
from argparse import Namespace
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only on minimal servers
    yaml = None

from upper_body_skeleton.ula_training import main as ula_training_main


DEFAULTS = {
    "steps": 1_000_000,
    "batch_size": 32,
    "lr": 1e-4,
    "max_episodes": 51_101,
    "hidden_dim": 384,
    "layers": 6,
    "device": "auto",
    "log_interval": 200,
}


PREVIEW_DEFAULTS = {
    "every_steps": 1000,
    "text": "紧张地解释一件困难的事情，然后逐渐缓和下来，动作要体现长程情绪变化",
    "mode": "long",
    "frames": 120,
    "fps": 30.0,
    "seed": 7,
    "duration_sec": 24.0,
    "min_segment_sec": 3.0,
    "max_segment_sec": 3.0,
    "min_segments": 8,
    "max_segments": 8,
    "max_velocity_rad_s": 3.0,
    "smooth_window": 5,
    "sampling_steps": 32,
    "width": 1280,
    "height": 720,
}


def load_train_config(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    if yaml is None:
        raise RuntimeError("PyYAML is required for YAML config files; use JSON or install pyyaml")
    data = yaml.safe_load(text)
    return data or {}


def _required(config, key):
    value = config.get(key)
    if value in (None, ""):
        raise ValueError(f"training config requires {key}")
    return value


def training_args_from_config(config):
    dataset_dir = str(_required(config, "dataset_dir"))
    output_dir = str(_required(config, "output_dir"))
    preview = dict(PREVIEW_DEFAULTS | (config.get("preview") or {}))
    preview_dir = preview.get("dir") or str(Path(output_dir) / "previews")
    values = DEFAULTS | {key: config[key] for key in DEFAULTS if key in config}
    return Namespace(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        steps=int(values["steps"]),
        batch_size=int(values["batch_size"]),
        lr=float(values["lr"]),
        max_episodes=None if values["max_episodes"] is None else int(values["max_episodes"]),
        hidden_dim=int(values["hidden_dim"]),
        layers=int(values["layers"]),
        device=str(values["device"]),
        log_interval=int(values["log_interval"]),
        preview_every_steps=int(preview["every_steps"]),
        preview_text=str(preview["text"]),
        preview_dir=str(preview_dir),
        preview_mode=str(preview["mode"]),
        preview_frames=int(preview["frames"]),
        preview_sampling_steps=int(preview["sampling_steps"]),
        preview_fps=float(preview["fps"]),
        preview_seed=int(preview["seed"]),
        preview_width=int(preview["width"]),
        preview_height=int(preview["height"]),
        preview_duration_sec=float(preview["duration_sec"]),
        preview_min_segment_sec=float(preview["min_segment_sec"]),
        preview_max_segment_sec=float(preview["max_segment_sec"]),
        preview_min_segments=int(preview["min_segments"]),
        preview_max_segments=int(preview["max_segments"]),
        preview_max_velocity_rad_s=float(preview["max_velocity_rad_s"]),
        preview_smooth_window=int(preview["smooth_window"]),
    )


def argv_from_training_args(args):
    argv = [
        "--dataset-dir",
        args.dataset_dir,
        "--output-dir",
        args.output_dir,
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--hidden-dim",
        str(args.hidden_dim),
        "--layers",
        str(args.layers),
        "--device",
        args.device,
        "--log-interval",
        str(args.log_interval),
        "--preview-every-steps",
        str(args.preview_every_steps),
        "--preview-text",
        args.preview_text,
        "--preview-dir",
        args.preview_dir,
        "--preview-mode",
        args.preview_mode,
        "--preview-frames",
        str(args.preview_frames),
        "--preview-sampling-steps",
        str(args.preview_sampling_steps),
        "--preview-fps",
        str(args.preview_fps),
        "--preview-seed",
        str(args.preview_seed),
        "--preview-width",
        str(args.preview_width),
        "--preview-height",
        str(args.preview_height),
        "--preview-duration-sec",
        str(args.preview_duration_sec),
        "--preview-min-segment-sec",
        str(args.preview_min_segment_sec),
        "--preview-max-segment-sec",
        str(args.preview_max_segment_sec),
        "--preview-min-segments",
        str(args.preview_min_segments),
        "--preview-max-segments",
        str(args.preview_max_segments),
        "--preview-max-velocity-rad-s",
        str(args.preview_max_velocity_rad_s),
        "--preview-smooth-window",
        str(args.preview_smooth_window),
    ]
    if args.max_episodes is not None:
        argv.extend(["--max-episodes", str(args.max_episodes)])
    return argv


def main():
    parser = argparse.ArgumentParser(description="Train ULA-FM from a YAML/JSON config")
    parser.add_argument("--config", required=True)
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()
    train_args = training_args_from_config(load_train_config(args.config))
    argv = argv_from_training_args(train_args)
    if args.print_command:
        command = "python -m upper_body_skeleton.ula_training " + " ".join(argv)
        print(command)
        return
    ula_training_main(argv)


if __name__ == "__main__":
    main()
