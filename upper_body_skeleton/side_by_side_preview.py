#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image

from upper_body_skeleton.retarget_v2 import JOINT_ORDER
from upper_body_skeleton.v2_axis_calibration import DEFAULT_URDF


def preview_output_path(sample_key, output_dir):
    safe = str(sample_key).replace("/", "__")
    return Path(output_dir) / f"{safe}.original_vs_v2_mujoco.mp4"


def _resize_rgb(frame, target_height):
    image = Image.fromarray(frame.astype(np.uint8), mode="RGB")
    width = max(1, round(image.width * target_height / image.height))
    return np.asarray(image.resize((width, target_height), Image.Resampling.BILINEAR))


def concat_side_by_side(original_frame, robot_frame, target_height=480):
    original = _resize_rgb(original_frame, target_height)
    robot = _resize_rgb(robot_frame, target_height)
    return np.concatenate([original, robot], axis=1)


def _joint_addresses(model):
    addresses = {}
    for joint_name in JOINT_ORDER:
        try:
            addresses[joint_name] = int(model.joint(joint_name).qposadr[0])
        except KeyError:
            continue
    return addresses


def _read_joint_rows(joint_csv):
    with open(joint_csv, newline="", encoding="utf-8") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def build_front_camera():
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.35]
    camera.distance = 2.3
    camera.elevation = -5
    camera.azimuth = 180
    return camera


def build_upper_body_camera():
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.16]
    camera.distance = 1.05
    camera.elevation = -4
    camera.azimuth = 180
    return camera


def build_camera(view):
    if view == "upper":
        return build_upper_body_camera()
    if view == "front":
        return build_front_camera()
    raise ValueError("camera view must be 'front' or 'upper'")


def update_renderer_scene(renderer, data, camera=None):
    if camera is None:
        renderer.update_scene(data)
    else:
        renderer.update_scene(data, camera=camera)


def render_robot_frames(joint_csv, urdf_path=DEFAULT_URDF, width=640, height=480, max_frames=None, camera=None):
    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = camera or build_front_camera()
    addresses = _joint_addresses(model)
    rows = _read_joint_rows(joint_csv)
    if max_frames is not None:
        rows = rows[:max_frames]

    frames = []
    for row in rows:
        data.qpos[:] = 0.0
        for joint_name, address in addresses.items():
            data.qpos[address] = row[joint_name]
        mujoco.mj_forward(model, data)
        update_renderer_scene(renderer, data, camera=camera)
        frames.append(renderer.render().copy())
    renderer.close()
    return frames


def write_robot_video(joint_csv, output_path, urdf_path=DEFAULT_URDF, fps=30.0, width=640, height=480, max_frames=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = render_robot_frames(joint_csv, urdf_path=urdf_path, width=width, height=height, max_frames=max_frames)
    with imageio.get_writer(output_path, fps=fps, codec="libx264", quality=7, macro_block_size=8) as writer:
        for frame in frames:
            writer.append_data(frame)
    return {"frames": len(frames), "output": str(output_path)}


def write_side_by_side_video(
    original_video,
    joint_csv,
    output_path,
    urdf_path=DEFAULT_URDF,
    fps=30.0,
    width=640,
    height=480,
    target_height=480,
    max_frames=None,
    start_frame=0,
    camera_view="front",
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    robot_frames = render_robot_frames(
        joint_csv,
        urdf_path=urdf_path,
        width=width,
        height=height,
        max_frames=max_frames,
        camera=build_camera(camera_view),
    )
    reader = imageio.get_reader(original_video)
    written = 0
    with imageio.get_writer(output_path, fps=fps, codec="libx264", quality=7, macro_block_size=8) as writer:
        try:
            for offset, robot_frame in enumerate(robot_frames):
                original_frame = reader.get_data(int(start_frame) + offset)
                if original_frame.shape[-1] == 4:
                    original_frame = original_frame[:, :, :3]
                writer.append_data(concat_side_by_side(original_frame, robot_frame, target_height=target_height))
                written += 1
        finally:
            reader.close()
    return {"frames": written, "output": str(output_path)}


def skip_video_frames(reader, count):
    for index in range(max(0, int(count))):
        if hasattr(reader, "get_data"):
            reader.get_data(index)
        else:
            next(reader)


def load_manifest_rows(manifest_path):
    with open(manifest_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select_rows(rows, sample_keys=None, limit=10):
    if sample_keys:
        wanted = set(sample_keys)
        selected = [row for row in rows if row["sample"] in wanted]
        return selected[:limit]
    processed = [row for row in rows if row.get("status") in {"processed", "skipped_existing"}]
    scored = sorted(
        processed,
        key=lambda row: (
            float(row.get("max_cross_body_intent") or 0.0),
            float(row.get("flagged_frame_count") or 0.0) / max(1.0, float(row.get("frame_count") or 1.0)),
        ),
        reverse=True,
    )
    return scored[:limit]


def render_manifest_samples(
    manifest_path,
    output_dir,
    urdf_path=DEFAULT_URDF,
    sample_keys=None,
    limit=10,
    fps=30.0,
    width=640,
    height=480,
    max_frames=None,
    start_frame=0,
    camera_view="front",
):
    rows = select_rows(load_manifest_rows(manifest_path), sample_keys=sample_keys, limit=limit)
    rendered = []
    for row in rows:
        output_path = preview_output_path(row["sample"], output_dir)
        result = write_side_by_side_video(
            row["video_path"],
            row["joint_csv"],
            output_path,
            urdf_path=urdf_path,
            fps=fps,
            width=width,
            height=height,
            target_height=height,
            max_frames=max_frames,
            start_frame=start_frame,
            camera_view=camera_view,
        )
        result.update({"sample": row["sample"], "video_path": row["video_path"], "joint_csv": row["joint_csv"]})
        rendered.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return rendered


def main():
    parser = argparse.ArgumentParser(description="Render V2 MuJoCo previews side-by-side with original videos")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--urdf", default=str(DEFAULT_URDF))
    parser.add_argument("--samples", help="Comma-separated manifest sample keys. Defaults to top scored rows.")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--camera-view", choices=["front", "upper"], default="front")
    args = parser.parse_args()

    sample_keys = [s.strip() for s in args.samples.split(",")] if args.samples else None
    rendered = render_manifest_samples(
        args.manifest,
        args.output_dir,
        urdf_path=args.urdf,
        sample_keys=sample_keys,
        limit=args.limit,
        fps=args.fps,
        width=args.width,
        height=args.height,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        camera_view=args.camera_view,
    )
    print(json.dumps({"rendered": len(rendered), "output_dir": args.output_dir}, indent=2), flush=True)


if __name__ == "__main__":
    main()
