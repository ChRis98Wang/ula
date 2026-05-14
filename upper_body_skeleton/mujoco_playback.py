#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path

import imageio_ffmpeg
import mediapy as media
import mujoco
import numpy as np

from upper_body_skeleton.retarget_v2 import JOINT_ORDER


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


def build_preview_model():
    model = mujoco.MjModel.from_xml_string(V2_UPPER_BODY_XML)
    joint_to_qpos = {}
    for index, joint in enumerate(JOINT_ORDER):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if joint_id < 0:
            raise ValueError(f"missing joint in MuJoCo model: {joint}")
        joint_to_qpos[index] = int(model.jnt_qposadr[joint_id])
    return model, joint_to_qpos


def render_motion(joint_csv, output_mp4, *, fps=30.0, width=1280, height=720, camera_distance=2.2):
    trajectory = read_joint_csv(joint_csv)
    model, joint_to_qpos = build_preview_model()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, -0.05]
    camera.distance = camera_distance
    camera.azimuth = 180
    camera.elevation = -10

    frames = []
    for values in trajectory:
        for action_index, value in enumerate(values):
            data.qpos[joint_to_qpos[action_index]] = float(value)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render())
    renderer.close()
    output_mp4 = Path(output_mp4)
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("FFMPEG_BINARY", imageio_ffmpeg.get_ffmpeg_exe())
    media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    media.write_video(output_mp4, frames, fps=fps)
    return {
        "input_csv": str(joint_csv),
        "output_mp4": str(output_mp4),
        "frames": int(trajectory.shape[0]),
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
    }


def main():
    parser = argparse.ArgumentParser(description="Render generated V2 upper-body joint CSV in MuJoCo")
    parser.add_argument("--joint-csv", required=True)
    parser.add_argument("--output-mp4", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    summary = render_motion(args.joint_csv, args.output_mp4, fps=args.fps, width=args.width, height=args.height)
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
