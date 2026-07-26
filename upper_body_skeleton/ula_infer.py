#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from upper_body_skeleton.retarget_v2 import JOINT_ORDER
from upper_body_skeleton.ula_training import (
    KIMODO_CONDITION_DIM,
    ULA_FM_LEGACY_ARCHITECTURE,
    build_condition_from_text,
    choose_device,
    create_ula_model,
    sample_trajectory,
    write_generated_csv,
)


def model_from_checkpoint(checkpoint, device, *, strict=True):
    config = checkpoint.get("config", {})
    model = create_ula_model(
        checkpoint.get("architecture", ULA_FM_LEGACY_ARCHITECTURE),
        action_dim=checkpoint.get("action_dim", len(JOINT_ORDER)),
        condition_dim=checkpoint.get("condition_dim", 92),
        hidden_dim=int(config.get("hidden_dim", 256)),
        layers=int(config.get("layers", 4)),
        semantic_tokens=int(config.get("semantic_tokens", 4)),
    )
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=bool(strict))
    if incompatible.unexpected_keys:
        raise RuntimeError(f"unexpected checkpoint keys: {incompatible.unexpected_keys}")
    if incompatible.missing_keys:
        allowed_missing_prefixes = ("plan.", "duration_head.", "transition_head.")
        disallowed_missing = [
            key for key in incompatible.missing_keys if not key.startswith(allowed_missing_prefixes)
        ]
        if disallowed_missing:
            raise RuntimeError(f"checkpoint is missing required keys: {disallowed_missing}")
    if checkpoint.get("action_stats") is not None:
        model.action_stats = checkpoint["action_stats"]
    model.planner_supervision_contract = dict(
        checkpoint.get("planner_supervision_contract")
        or (checkpoint.get("training_contract") or {}).get("planner_supervision")
        or {}
    )
    model.to(device)
    return model, checkpoint


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # The older public loader keeps planner-head backward compatibility. New
    # direct-PT paths validate current checkpoints and request strict loading.
    return model_from_checkpoint(checkpoint, device, strict=False)


def main():
    parser = argparse.ArgumentParser(description="Generate V2 upper-body joint motion from text with ULA-FM")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-npz")
    parser.add_argument("--summary-json")
    parser.add_argument("--intent")
    parser.add_argument("--affect")
    parser.add_argument("--style")
    parser.add_argument("--gesture")
    parser.add_argument("--behavior-id")
    parser.add_argument("--emotion-id")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--sampling-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    expected_condition_dim = checkpoint.get("condition_dim", KIMODO_CONDITION_DIM)
    condition = build_condition_from_text(
        args.text,
        intent=args.intent,
        affect=args.affect,
        style=args.style,
        gesture=args.gesture,
        behavior_id=args.behavior_id,
        emotion_id=args.emotion_id,
        condition_dim=expected_condition_dim,
    )
    if condition.shape[0] != expected_condition_dim:
        raise SystemExit(
            f"condition dim mismatch: built {condition.shape[0]}, checkpoint expects {expected_condition_dim}"
        )
    trajectory = sample_trajectory(
        model,
        condition=condition,
        frames=args.frames,
        action_dim=checkpoint.get("action_dim", len(JOINT_ORDER)),
        steps=args.sampling_steps,
        device=device,
        seed=args.seed,
        action_stats=getattr(model, "action_stats", None),
    )
    write_generated_csv(args.output_csv, trajectory, fps=args.fps)
    if args.output_npz:
        Path(args.output_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output_npz,
            trajectory=trajectory.astype(np.float32),
            joint_order=np.asarray(JOINT_ORDER[: trajectory.shape[1]], dtype=object),
            text=args.text,
            fps=np.asarray(args.fps, dtype=np.float32),
        )
    summary = {
        "checkpoint": str(args.checkpoint),
        "text": args.text,
        "output_csv": str(args.output_csv),
        "output_npz": str(args.output_npz) if args.output_npz else None,
        "frames": int(trajectory.shape[0]),
        "action_dim": int(trajectory.shape[1]),
        "fps": float(args.fps),
        "device": device,
        "seed": args.seed,
        "architecture": getattr(model, "architecture", ULA_FM_LEGACY_ARCHITECTURE),
        "behavior_id": args.behavior_id,
        "emotion_id": args.emotion_id,
    }
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
