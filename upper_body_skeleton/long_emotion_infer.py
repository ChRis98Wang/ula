#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from upper_body_skeleton.kimodo_semantics import kimodo_condition_metadata
from upper_body_skeleton.mujoco_playback import play_motion, render_motion
from upper_body_skeleton.retarget_v2 import JOINT_ORDER
from upper_body_skeleton.ula_infer import load_model
from upper_body_skeleton.ula_training import (
    KIMODO_CONDITION_DIM,
    LEGACY_CONDITION_DIM,
    TRANSITION_IDS,
    build_condition_from_text,
    choose_device,
    sample_trajectory,
    write_generated_csv,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LONG_EMOTION_OUTPUT_DIR = REPO_ROOT / "deliverables" / "long_emotion_previews" / "manual"
TRANSITION_NAMES = {value: key for key, value in TRANSITION_IDS.items()}


def _clamp(value, low, high):
    return max(float(low), min(float(high), float(value)))


def _condition_for_segment(text, segment_index, previous_transition, *, behavior_id=None, emotion_id=None, condition_dim=KIMODO_CONDITION_DIM):
    suffix = f" segment {segment_index}; previous transition {previous_transition}"
    return build_condition_from_text(
        text + suffix,
        behavior_id=behavior_id,
        emotion_id=emotion_id,
        condition_dim=condition_dim,
    )


def _predict_plan(model, condition, device):
    tensor = torch.as_tensor(condition, dtype=torch.float32, device=device)
    with torch.no_grad():
        plan = model.plan_condition(tensor)
        probs = torch.softmax(plan["transition_logits"], dim=-1)[0].detach().cpu().numpy()
        duration = float(plan["duration_sec"][0].detach().cpu())
    transition_id = int(np.argmax(probs))
    return duration, transition_id, probs


def limit_joint_velocity(trajectory, *, fps=30.0, max_velocity_rad_s=3.0):
    arr = np.asarray(trajectory, dtype=np.float32).copy()
    if len(arr) < 2:
        return arr
    max_delta = float(max_velocity_rad_s) / float(fps)
    for index in range(1, len(arr)):
        delta = np.clip(arr[index] - arr[index - 1], -max_delta, max_delta)
        arr[index] = arr[index - 1] + delta
    return arr


def smooth_trajectory(trajectory, window=5, *, fps=30.0, max_velocity_rad_s=3.0):
    arr = np.asarray(trajectory, dtype=np.float32)
    window = int(max(1, window))
    if window <= 1 or len(arr) < 3:
        return arr.copy()
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded = np.pad(arr, ((radius, radius), (0, 0)), mode="edge")
    out = np.empty_like(arr)
    for index in range(len(arr)):
        out[index] = np.mean(padded[index : index + window], axis=0)
    out[0] = arr[0]
    # Re-apply velocity limiting because moving averages can introduce a small
    # start-frame jump when the raw second frame is far away.
    return limit_joint_velocity(out, fps=fps, max_velocity_rad_s=max_velocity_rad_s)


def postprocess_trajectory(trajectory, *, fps=30.0, max_velocity_rad_s=3.0, smooth_window=5):
    limited = limit_joint_velocity(trajectory, fps=fps, max_velocity_rad_s=max_velocity_rad_s)
    smoothed = smooth_trajectory(limited, window=smooth_window, fps=fps, max_velocity_rad_s=max_velocity_rad_s)
    return limit_joint_velocity(smoothed, fps=fps, max_velocity_rad_s=max_velocity_rad_s)


def trajectory_quality(trajectory, *, fps=30.0):
    arr = np.asarray(trajectory, dtype=np.float32)
    if len(arr) < 2:
        return {
            "frames": int(arr.shape[0]),
            "max_delta_rad_per_frame": 0.0,
            "mean_delta_rad_per_frame": 0.0,
            "max_velocity_rad_s": 0.0,
            "mean_velocity_rad_s": 0.0,
        }
    delta = np.abs(np.diff(arr, axis=0))
    max_delta = float(delta.max())
    mean_delta = float(delta.mean())
    return {
        "frames": int(arr.shape[0]),
        "max_delta_rad_per_frame": max_delta,
        "mean_delta_rad_per_frame": mean_delta,
        "max_velocity_rad_s": float(max_delta * float(fps)),
        "mean_velocity_rad_s": float(mean_delta * float(fps)),
    }


def generate_long_emotion_motion(
    model,
    *,
    text,
    behavior_id=None,
    emotion_id=None,
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
    max_velocity_rad_s=3.0,
    smooth_window=5,
    play=False,
    play_loops=0,
    play_realtime=True,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    model.eval()
    condition_dim = int(getattr(model, "condition_dim", KIMODO_CONDITION_DIM))
    kimodo_meta = kimodo_condition_metadata(behavior_id=behavior_id, emotion_id=emotion_id, text=text)
    behavior_id = kimodo_meta["behavior_id"] if condition_dim != LEGACY_CONDITION_DIM else behavior_id
    emotion_id = kimodo_meta["emotion_id"] if condition_dim != LEGACY_CONDITION_DIM else emotion_id

    segments = []
    trajectories = []
    elapsed = 0.0
    previous_transition = "start"
    last_pose = None

    for segment_index in range(int(max_segments)):
        condition = _condition_for_segment(
            text,
            segment_index,
            previous_transition,
            behavior_id=behavior_id,
            emotion_id=emotion_id,
            condition_dim=condition_dim,
        )
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
        condition = build_condition_from_text(
            text,
            behavior_id=behavior_id,
            emotion_id=emotion_id,
            condition_dim=condition_dim,
        )
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

    raw_trajectory = np.concatenate(trajectories, axis=0).astype(np.float32)
    trajectory = postprocess_trajectory(
        raw_trajectory,
        fps=fps,
        max_velocity_rad_s=max_velocity_rad_s,
        smooth_window=smooth_window,
    ).astype(np.float32)
    csv_path = output_dir / "long_motion.csv"
    npz_path = output_dir / "long_motion.npz"
    plan_path = output_dir / "plan.json"
    mp4_path = output_dir / "long_motion_original_v2.mp4"
    summary_path = output_dir / "summary.json"

    write_generated_csv(csv_path, trajectory, fps=fps)
    np.savez_compressed(
        npz_path,
        trajectory=trajectory,
        raw_trajectory=raw_trajectory,
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
    viewer_summary = None
    if play:
        viewer_summary = play_motion(csv_path, fps=fps, loops=play_loops, realtime=play_realtime)

    plan = {
        "text": text,
        "behavior_id": behavior_id,
        "emotion_id": emotion_id,
        "fps": float(fps),
        "max_duration_sec": float(max_duration_sec),
        "segments": segments,
        "csv": str(csv_path),
        "npz": str(npz_path),
        "rendered_mp4": rendered_mp4,
        "viewer": viewer_summary,
        "kimodo_condition": kimodo_meta,
    }
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "output_dir": str(output_dir),
        "text": text,
        "behavior_id": behavior_id,
        "emotion_id": emotion_id,
        "segments": len(segments),
        "frames": int(trajectory.shape[0]),
        "duration_sec": float(trajectory.shape[0] / float(fps)),
        "csv": str(csv_path),
        "npz": str(npz_path),
        "plan_json": str(plan_path),
        "rendered_mp4": rendered_mp4,
        "render": render_summary,
        "viewer": viewer_summary,
        "kimodo_condition": kimodo_meta,
        "postprocess": {
            "max_velocity_rad_s": float(max_velocity_rad_s),
            "smooth_window": int(smooth_window),
        },
        "trajectory_quality": {
            "raw": trajectory_quality(raw_trajectory, fps=fps),
            "processed": trajectory_quality(trajectory, fps=fps),
        },
        "last_pose": [float(v) for v in trajectory[-1]],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate long-horizon emotion-aware V2 upper-body motion")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--behavior-id")
    parser.add_argument("--emotion-id")
    parser.add_argument("--output-dir", default=str(DEFAULT_LONG_EMOTION_OUTPUT_DIR))
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
    parser.add_argument("--max-velocity-rad-s", type=float, default=3.0)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--viewer", action="store_true", help="Open a MuJoCo viewer and play the generated motion")
    parser.add_argument("--viewer-loops", type=int, default=0, help="Viewer loops to play; 0 means until the viewer closes")
    parser.add_argument("--viewer-no-realtime", action="store_true", help="Play in the viewer as fast as possible")
    args = parser.parse_args()

    device = choose_device(args.device)
    model, _ = load_model(args.checkpoint, device)
    summary = generate_long_emotion_motion(
        model,
        text=args.text,
        behavior_id=args.behavior_id,
        emotion_id=args.emotion_id,
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
        max_velocity_rad_s=args.max_velocity_rad_s,
        smooth_window=args.smooth_window,
        play=args.viewer,
        play_loops=args.viewer_loops,
        play_realtime=not args.viewer_no_realtime,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
