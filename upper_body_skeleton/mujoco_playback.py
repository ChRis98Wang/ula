#!/usr/bin/env python3
import argparse
import csv
import json
import os
import tempfile
import time
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from upper_body_skeleton.retarget_v2 import JOINT_ORDER
from upper_body_skeleton.side_by_side_preview import build_camera, build_front_camera
from upper_body_skeleton.v2_axis_calibration import DEFAULT_URDF, resolve_mujoco_urdf


V2_UPPER_BODY_XML = """
<mujoco model="v2_upper_body_preview">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <quality shadowsize="2048"/>
    <map force="0.1" znear="0.01"/>
  </visual>
  <asset>
    <material name="torso" rgba="0.22 0.26 0.30 1"/>
    <material name="left" rgba="0.15 0.45 0.95 1"/>
    <material name="right" rgba="0.90 0.35 0.18 1"/>
    <material name="joint" rgba="0.92 0.92 0.86 1"/>
    <material name="floor" rgba="0.18 0.20 0.22 1"/>
  </asset>
  <worldbody>
    <light name="key" pos="0 -3 4" dir="0 1 -1" directional="true" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="3 3 0.05" pos="0 0 -1.05" material="floor"/>
    <body name="pelvis" pos="0 0 0">
      <joint name="joint_pelvisYaw" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
      <joint name="joint_pelvisPitch" type="hinge" axis="0 1 0" range="0 1.046"/>
      <joint name="joint_pelvisRoll" type="hinge" axis="1 0 0" range="-0.35 0.35"/>
      <geom name="torso" type="capsule" fromto="0 0 -0.35 0 0 0.45" size="0.14" material="torso"/>
      <geom name="chest" type="box" pos="0 0 0.25" size="0.26 0.10 0.18" material="torso"/>
      <body name="left_shoulder" pos="0 0.24 0.33">
        <joint name="joint_lShoulderPitch" type="hinge" axis="0 1 0" range="-1.4 4.2"/>
        <joint name="joint_lShoulderRoll" type="hinge" axis="1 0 0" range="-1.41 1.57"/>
        <joint name="joint_lShoulderYaw" type="hinge" axis="0 0 1" range="-2.79 2.79"/>
        <geom name="left_upper_arm" type="capsule" fromto="0 0 0 0 0 -0.28" size="0.045" material="left"/>
        <body name="left_elbow" pos="0 0 -0.28">
          <joint name="joint_lElbow" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
          <geom name="left_forearm" type="capsule" fromto="0 0 0 0 0 -0.25" size="0.038" material="left"/>
          <body name="left_wrist" pos="0 0 -0.25">
            <joint name="joint_lWristRoll" type="hinge" axis="0 0 1" range="-2.79 2.79"/>
            <joint name="joint_lWristPitch" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
            <geom name="left_hand" type="sphere" size="0.055" material="joint"/>
          </body>
        </body>
      </body>
      <body name="right_shoulder" pos="0 -0.24 0.33">
        <joint name="joint_rShoulderPitch" type="hinge" axis="0 1 0" range="-1.4 4.2"/>
        <joint name="joint_rShoulderRoll" type="hinge" axis="1 0 0" range="-1.41 1.57"/>
        <joint name="joint_rShoulderYaw" type="hinge" axis="0 0 1" range="-2.79 2.79"/>
        <geom name="right_upper_arm" type="capsule" fromto="0 0 0 0 0 -0.28" size="0.045" material="right"/>
        <body name="right_elbow" pos="0 0 -0.28">
          <joint name="joint_rElbow" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
          <geom name="right_forearm" type="capsule" fromto="0 0 0 0 0 -0.25" size="0.038" material="right"/>
          <body name="right_wrist" pos="0 0 -0.25">
            <joint name="joint_rWristRoll" type="hinge" axis="0 0 1" range="-2.79 2.79"/>
            <joint name="joint_rWristPitch" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
            <geom name="right_hand" type="sphere" size="0.055" material="joint"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def read_joint_csv(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append([float(row[joint]) for joint in JOINT_ORDER])
    return np.asarray(rows, dtype=np.float32)


def load_preview_model(urdf_path=DEFAULT_URDF, *, simplified=False):
    if simplified:
        model = mujoco.MjModel.from_xml_string(V2_UPPER_BODY_XML)
        model_source = "simplified_builtin_xml"
    else:
        urdf_path = resolve_mujoco_urdf(urdf_path)
        model = mujoco.MjModel.from_xml_path(str(urdf_path))
        model_source = str(urdf_path)
    joint_to_qpos = {}
    for index, joint in enumerate(JOINT_ORDER):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if joint_id < 0:
            raise ValueError(f"missing joint in MuJoCo model: {joint}")
        joint_to_qpos[index] = int(model.jnt_qposadr[joint_id])
    return model, joint_to_qpos, model_source


def build_preview_model(urdf_path=DEFAULT_URDF, *, simplified=False):
    model, joint_to_qpos, _ = load_preview_model(urdf_path=urdf_path, simplified=simplified)
    return model, joint_to_qpos


def _validated_trajectory(values, *, name):
    trajectory = np.asarray(values, dtype=np.float32)
    if trajectory.ndim != 2 or trajectory.shape[1] != len(JOINT_ORDER):
        raise ValueError(f"{name} trajectory must have shape [frames, {len(JOINT_ORDER)}]")
    if trajectory.shape[0] < 1:
        raise ValueError(f"{name} trajectory must contain at least one frame")
    if not np.isfinite(trajectory).all():
        raise ValueError(f"{name} trajectory contains non-finite values")
    return trajectory


def compose_labeled_comparison_frame(
    network_frame,
    reference_frame,
    *,
    network_label="NETWORK OUTPUT",
    reference_label="DATASET REFERENCE",
    title_height=40,
):
    network_frame = np.asarray(network_frame, dtype=np.uint8)
    reference_frame = np.asarray(reference_frame, dtype=np.uint8)
    if network_frame.ndim != 3 or network_frame.shape[-1] != 3:
        raise ValueError("network frame must be an RGB image")
    if reference_frame.shape != network_frame.shape:
        raise ValueError("network and reference frames must have identical RGB shapes")
    title_height = int(title_height)
    if title_height < 24:
        raise ValueError("title_height must be at least 24 pixels")

    height, width, _ = network_frame.shape
    image = Image.new("RGB", (width * 2, height + title_height), (18, 21, 25))
    image.paste(Image.fromarray(network_frame), (0, title_height))
    image.paste(Image.fromarray(reference_frame), (width, title_height))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except OSError:  # pragma: no cover - Pillow always ships a default fallback
        font = ImageFont.load_default()
    for offset, label, color in (
        (0, str(network_label), (71, 174, 255)),
        (width, str(reference_label), (80, 205, 137)),
    ):
        bounds = draw.textbbox((0, 0), label, font=font)
        text_width = bounds[2] - bounds[0]
        text_height = bounds[3] - bounds[1]
        x = offset + max(8, (width - text_width) // 2)
        y = max(2, (title_height - text_height) // 2 - bounds[1])
        draw.text((x, y), label, fill=color, font=font)
    draw.line((width, 0, width, height + title_height), fill=(224, 228, 232), width=2)
    return np.asarray(image)


def render_trajectory_comparison(
    network_trajectory,
    reference_trajectory,
    output_mp4,
    *,
    fps=30.0,
    pane_width=640,
    pane_height=640,
    title_height=40,
    urdf_path=DEFAULT_URDF,
    simplified=False,
    camera_view="upper",
):
    network_trajectory = _validated_trajectory(network_trajectory, name="network")
    reference_trajectory = _validated_trajectory(reference_trajectory, name="reference")
    if network_trajectory.shape != reference_trajectory.shape:
        raise ValueError("network and reference trajectories must have identical shapes")
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    pane_width = int(pane_width)
    pane_height = int(pane_height)
    title_height = int(title_height)
    if pane_width <= 0 or pane_height <= 0:
        raise ValueError("comparison pane dimensions must be positive")
    if title_height < 24:
        raise ValueError("title_height must be at least 24 pixels")
    if (pane_height + title_height) % 2:
        raise ValueError("comparison video height must be even for H.264 encoding")
    if camera_view not in {"front", "upper"}:
        raise ValueError("camera_view must be 'front' or 'upper'")

    model, joint_to_qpos, model_source = load_preview_model(urdf_path=urdf_path, simplified=simplified)
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), pane_width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), pane_height)
    data = mujoco.MjData(model)
    camera = build_camera(camera_view)
    output_mp4 = Path(output_mp4)
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", imageio_ffmpeg.get_ffmpeg_exe())
    frames_written = 0
    renderer = None
    temporary_path = None

    def render_values(values):
        data.qpos[:] = 0.0
        for action_index, value in enumerate(values):
            data.qpos[joint_to_qpos[action_index]] = float(value)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        return renderer.render().copy()

    try:
        renderer = mujoco.Renderer(model, height=pane_height, width=pane_width)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_mp4.stem}.",
            suffix=output_mp4.suffix,
            dir=output_mp4.parent,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        with imageio.get_writer(
            temporary_path,
            fps=fps,
            codec="libx264",
            quality=7,
            macro_block_size=2,
        ) as writer:
            for network_values, reference_values in zip(network_trajectory, reference_trajectory):
                network_frame = render_values(network_values)
                reference_frame = render_values(reference_values)
                writer.append_data(
                    compose_labeled_comparison_frame(
                        network_frame,
                        reference_frame,
                        title_height=title_height,
                    )
                )
                frames_written += 1
        temporary_path.replace(output_mp4)
        temporary_path = None
    finally:
        if renderer is not None:
            renderer.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return {
        "output_mp4": str(output_mp4),
        "layout": {"left": "network_output", "right": "dataset_reference"},
        "frames": frames_written,
        "duration_sec": frames_written / fps,
        "fps": fps,
        "width": pane_width * 2,
        "height": pane_height + title_height,
        "camera_view": camera_view,
        "model_source": model_source,
        "model_nq": int(model.nq),
        "model_nbody": int(model.nbody),
    }


def render_motion(
    joint_csv,
    output_mp4,
    *,
    fps=30.0,
    width=1280,
    height=720,
    urdf_path=DEFAULT_URDF,
    simplified=False,
    camera_distance=None,
):
    trajectory = read_joint_csv(joint_csv)
    model, joint_to_qpos, model_source = load_preview_model(urdf_path=urdf_path, simplified=simplified)
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(width))
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(height))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = build_front_camera()
    if camera_distance is not None:
        camera.distance = float(camera_distance)

    frames = []
    for values in trajectory:
        data.qpos[:] = 0.0
        for action_index, value in enumerate(values):
            data.qpos[joint_to_qpos[action_index]] = float(value)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render().copy())
    renderer.close()
    output_mp4 = Path(output_mp4)
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", imageio_ffmpeg.get_ffmpeg_exe())
    imageio.mimwrite(output_mp4, frames, fps=fps, codec="libx264", quality=7, macro_block_size=8)
    return {
        "input_csv": str(joint_csv),
        "output_mp4": str(output_mp4),
        "frames": int(trajectory.shape[0]),
        "duration_sec": float(trajectory.shape[0] / fps) if fps else None,
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "model_source": model_source,
        "model_nq": int(model.nq),
        "model_nbody": int(model.nbody),
    }


def _launch_passive_viewer(model, data):
    return mujoco.viewer.launch_passive(model, data)


class MujocoMotionPlayer:
    def __init__(self, *, fps=30.0, urdf_path=DEFAULT_URDF, simplified=False):
        self.fps = float(fps)
        self.model, self.joint_to_qpos, self.model_source = load_preview_model(
            urdf_path=urdf_path,
            simplified=simplified,
        )
        self.data = mujoco.MjData(self.model)
        self._viewer_context = None
        self.viewer = None

    def __enter__(self):
        self._viewer_context = _launch_passive_viewer(self.model, self.data)
        self.viewer = self._viewer_context.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._viewer_context is not None:
                return self._viewer_context.__exit__(exc_type, exc, tb)
            return False
        finally:
            self.viewer = None
            self._viewer_context = None

    def play_csv(self, joint_csv, *, loops=1, realtime=True, stop_event=None):
        return self.play_trajectory(
            read_joint_csv(joint_csv),
            loops=loops,
            realtime=realtime,
            input_csv=joint_csv,
            stop_event=stop_event,
        )

    def play_trajectory(self, trajectory, *, loops=1, realtime=True, input_csv=None, stop_event=None):
        if self.viewer is None:
            raise RuntimeError("MujocoMotionPlayer must be used as a context manager")
        trajectory = np.asarray(trajectory, dtype=np.float32)
        frame_period = 1.0 / self.fps if self.fps else 0.0
        frames_played = 0
        loops_completed = 0
        interrupted = False

        while self.viewer.is_running():
            loop_finished = True
            for values in trajectory:
                if stop_event is not None and stop_event.is_set():
                    interrupted = True
                    loop_finished = False
                    break
                if not self.viewer.is_running():
                    loop_finished = False
                    break
                start = time.monotonic()
                self.data.qpos[:] = 0.0
                for action_index, value in enumerate(values):
                    self.data.qpos[self.joint_to_qpos[action_index]] = float(value)
                mujoco.mj_forward(self.model, self.data)
                self.viewer.sync()
                frames_played += 1
                if stop_event is not None and stop_event.is_set():
                    interrupted = True
                    loop_finished = False
                    break
                if realtime and frame_period > 0:
                    elapsed = time.monotonic() - start
                    if elapsed < frame_period:
                        time.sleep(frame_period - elapsed)
            if not loop_finished:
                break
            loops_completed += 1
            if loops and loops_completed >= int(loops):
                break

        return {
            "input_csv": str(input_csv) if input_csv is not None else None,
            "frames": int(trajectory.shape[0]),
            "frames_played": int(frames_played),
            "loops_completed": int(loops_completed),
            "interrupted": bool(interrupted),
            "fps": float(self.fps),
            "model_source": self.model_source,
            "model_nq": int(self.model.nq),
            "model_nbody": int(self.model.nbody),
        }


def play_motion(
    joint_csv,
    *,
    fps=30.0,
    urdf_path=DEFAULT_URDF,
    simplified=False,
    loops=0,
    realtime=True,
):
    with MujocoMotionPlayer(fps=fps, urdf_path=urdf_path, simplified=simplified) as player:
        return player.play_csv(joint_csv, loops=loops, realtime=realtime)


def main():
    parser = argparse.ArgumentParser(description="Render generated V2 upper-body joint CSV in MuJoCo")
    parser.add_argument("--joint-csv", required=True)
    parser.add_argument("--output-mp4")
    parser.add_argument("--summary-json")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--urdf", default=str(DEFAULT_URDF))
    parser.add_argument("--simplified", action="store_true", help="Use the old built-in simplified preview model")
    parser.add_argument("--viewer", action="store_true", help="Open a MuJoCo viewer and play the motion interactively")
    parser.add_argument("--loops", type=int, default=0, help="Viewer loops to play; 0 means until the viewer closes")
    parser.add_argument("--no-realtime", action="store_true", help="Viewer playback as fast as possible")
    args = parser.parse_args()
    if args.viewer:
        summary = play_motion(
            args.joint_csv,
            fps=args.fps,
            urdf_path=args.urdf,
            simplified=args.simplified,
            loops=args.loops,
            realtime=not args.no_realtime,
        )
    else:
        if not args.output_mp4:
            parser.error("--output-mp4 is required unless --viewer is set")
        summary = render_motion(
            args.joint_csv,
            args.output_mp4,
            fps=args.fps,
            width=args.width,
            height=args.height,
            urdf_path=args.urdf,
            simplified=args.simplified,
        )
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
