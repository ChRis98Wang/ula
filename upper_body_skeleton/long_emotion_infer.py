#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from upper_body_skeleton.mujoco_playback import render_motion
from upper_body_skeleton.retarget_v2 import JOINT_ORDER
from upper_body_skeleton.ula_infer import load_model
from upper_body_skeleton.ula_training import (
    TRANSITION_IDS,
    build_condition_from_text,
    choose_device,
    sample_trajectory,
    write_generated_csv,
)


TRANSITION_NAMES = {value: key for key, value in TRANSITION_IDS.items()}


def _clamp(value, low, high):
    return max(float(low), min(float(high), float(value)))


def _condition_for_segment(text, segment_index, previous_transition):
    suffix = f" segment {segment_index}; previous transition {previous_transition}"
    return build_condition_from_text(text + suffix)


def _predict_plan(model, condition, device):
    tensor = torch.as_tensor(condition, dtype=torch.float32, device=device)
    with torch.no_grad():
        plan = model.plan_condition(tensor)
        probs = torch.softmax(plan["transition_logits"], dim=-1)[0].detach().cpu().numpy()
        duration = float(plan["duration_sec"][0].detach().cpu())
    transition_id = int(np.argmax(probs))
    return duration, transition_id, probs


def generate_long_emotion_motion(
    model,
    *,
    text,
    output_dir,
    fps=30.0,
    max_duration_sec=30.0,
    min_segment_sec=2.0,
    max_segment_sec=8.0,
    min_segments=2,
    max_segments=8,
    sampling_steps=32,
    device="cpu",
    seed=7,
    render=True,
    width=1280,
    height=720,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    model.eval()

    segments = []
    trajectories = []
    elapsed = 0.0
    previous_transition = "start"
    last_pose = None

    for segment_index in range(int(max_segments)):
        condition = _condition_for_segment(text, segment_index, previous_transition)
        predicted_duration, transition_id, transition_probs = _predict_plan(model, condition, device)
        remaining = max(0.0, float(max_duration_sec) - elapsed)
        if remaining <= 1.0 / float(fps):
            break
        duration_sec = _clamp(predicted_duration, min_segment_sec, max_segment_sec)
        duration_sec = min(duration_sec, remaining)
        frames = max(2, int(round(duration_sec * float(fps))))
        trajectory = sample_trajectory(
            model,
            condition=condition,
            frames=frames,
            action_dim=len(JOINT_ORDER),
            steps=sampling_steps,
            device=device,
            seed=None if seed is None else int(seed) + segment_index,
        )
        if last_pose is not None and len(trajectory):
            trajectory[0] = last_pose
        last_pose = trajectory[-1].copy()
        transition_name = TRANSITION_NAMES.get(transition_id, "continue")
        start_sec = elapsed
        elapsed += frames / float(fps)
        segments.append(
            {
                "segment_index": segment_index,
                "start_sec": start_sec,
                "end_sec": elapsed,
                "frames": int(frames),
                "predicted_duration_sec": predicted_duration,
                "used_duration_sec": float(frames / float(fps)),
                "transition": transition_name,
                "transition_probs": {TRANSITION_NAMES[i]: float(v) for i, v in enumerate(transition_probs)},
            }
        )
        trajectories.append(trajectory)
        previous_transition = transition_name
        if transition_name == "end" and len(segments) >= int(min_segments):
            break

    if not trajectories:
        condition = build_condition_from_text(text)
        frames = max(2, int(round(float(min_segment_sec) * float(fps))))
        trajectories.append(
            sample_trajectory(
                model,
                condition=condition,
                frames=frames,
                action_dim=len(JOINT_ORDER),
                steps=sampling_steps,
                device=device,
                seed=seed,
            )
        )
        elapsed = frames / float(fps)
        segments.append(
            {
                "segment_index": 0,
                "start_sec": 0.0,
                "end_sec": elapsed,
                "frames": int(frames),
                "predicted_duration_sec": float(min_segment_sec),
                "used_duration_sec": elapsed,
                "transition": "end",
                "transition_probs": {},
            }
        )

    trajectory = np.concatenate(trajectories, axis=0).astype(np.float32)
    csv_path = output_dir / "long_motion.csv"
    npz_path = output_dir / "long_motion.npz"
    plan_path = output_dir / "plan.json"
    mp4_path = output_dir / "long_motion_original_v2.mp4"
    summary_path = output_dir / "summary.json"

    write_generated_csv(csv_path, trajectory, fps=fps)
    np.savez_compressed(
        npz_path,
        trajectory=trajectory,
        joint_order=np.asarray(JOINT_ORDER, dtype=object),
        text=text,
        fps=np.asarray(fps, dtype=np.float32),
        segments=np.asarray(json.dumps(segments, ensure_ascii=False), dtype=object),
    )
    rendered_mp4 = None
    render_summary = None
    if render:
        render_summary = render_motion(csv_path, mp4_path, fps=fps, width=width, height=height)
        rendered_mp4 = str(mp4_path)

    plan = {
        "text": text,
        "fps": float(fps),
        "max_duration_sec": float(max_duration_sec),
        "segments": segments,
        "csv": str(csv_path),
        "npz": str(npz_path),
        "rendered_mp4": rendered_mp4,
    }
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "output_dir": str(output_dir),
        "text": text,
        "segments": len(segments),
        "frames": int(trajectory.shape[0]),
        "duration_sec": float(trajectory.shape[0] / float(fps)),
        "csv": str(csv_path),
        "npz": str(npz_path),
        "plan_json": str(plan_path),
        "rendered_mp4": rendered_mp4,
        "render": render_summary,
        "last_pose": [float(v) for v in trajectory[-1]],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate long-horizon emotion-aware V2 upper-body motion")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output-dir", default="/Users/demo/Desktop/upper_body_motion_roadmap/deliverables/long_emotion_previews/manual")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-duration-sec", type=float, default=30.0)
    parser.add_argument("--min-segment-sec", type=float, default=2.0)
    parser.add_argument("--max-segment-sec", type=float, default=8.0)
    parser.add_argument("--min-segments", type=int, default=2)
    parser.add_argument("--max-segments", type=int, default=8)
    parser.add_argument("--sampling-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    device = choose_device(args.device)
    model, _ = load_model(args.checkpoint, device)
    summary = generate_long_emotion_motion(
        model,
        text=args.text,
        output_dir=args.output_dir,
        fps=args.fps,
        max_duration_sec=args.max_duration_sec,
        min_segment_sec=args.min_segment_sec,
        max_segment_sec=args.max_segment_sec,
        min_segments=args.min_segments,
        max_segments=args.max_segments,
        sampling_steps=args.sampling_steps,
        device=device,
        seed=args.seed,
        render=not args.no_render,
        width=args.width,
        height=args.height,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
