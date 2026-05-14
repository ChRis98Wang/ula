#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


DEFAULT_URDF = Path(
    "/Users/demo/Desktop/upper_body_motion_roadmap/video/seamless_interaction_50g/upper_body_skeletons/"
    "mujoco_robot_v2_meshdir.urdf"
)

V2_ARMS_DOWN_SHOULDER_ROLL = -1.4
DELTA_RAD = 0.2


JOINT_TO_WRIST_BODY = {
    "joint_lShoulderPitch": "link_lWristPitch",
    "joint_lShoulderRoll": "link_lWristPitch",
    "joint_lShoulderYaw": "link_lWristPitch",
    "joint_lElbow": "link_lWristPitch",
    "joint_lWristRoll": "link_lWristPitch",
    "joint_lWristPitch": "link_lWristPitch",
    "joint_rShoulderPitch": "link_rWristPitch",
    "joint_rShoulderRoll": "link_rWristPitch",
    "joint_rShoulderYaw": "link_rWristPitch",
    "joint_rElbow": "link_rWristPitch",
    "joint_rWristRoll": "link_rWristPitch",
    "joint_rWristPitch": "link_rWristPitch",
}


def _name_maps(model):
    return (
        {model.joint(i).name: int(model.joint(i).qposadr[0]) for i in range(model.njnt)},
        {model.body(i).name: i for i in range(model.nbody)},
    )


def _body_xpos(model, data, qpos, body_name, body_ids):
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    return data.xpos[body_ids[body_name]].copy()


def analyze_v2_axes(urdf_path=DEFAULT_URDF, delta_rad=DELTA_RAD):
    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    data = mujoco.MjData(model)
    qpos_addr, body_ids = _name_maps(model)

    base_qpos = np.zeros(model.nq)
    if "joint_lShoulderRoll" in qpos_addr:
        base_qpos[qpos_addr["joint_lShoulderRoll"]] = V2_ARMS_DOWN_SHOULDER_ROLL
    if "joint_rShoulderRoll" in qpos_addr:
        base_qpos[qpos_addr["joint_rShoulderRoll"]] = V2_ARMS_DOWN_SHOULDER_ROLL

    joint_report = {}
    for joint, wrist_body in JOINT_TO_WRIST_BODY.items():
        if joint not in qpos_addr or wrist_body not in body_ids:
            continue
        q_plus = base_qpos.copy()
        q_plus[qpos_addr[joint]] += delta_rad
        base = _body_xpos(model, data, base_qpos, wrist_body, body_ids)
        moved = _body_xpos(model, data, q_plus, wrist_body, body_ids)
        delta = moved - base
        joint_report[joint] = {
            "wrist_body": wrist_body,
            "delta_rad": float(delta_rad),
            "base_wrist_position": [float(v) for v in base],
            "moved_wrist_position": [float(v) for v in moved],
            "wrist_delta": [float(v) for v in delta],
            "delta_norm": float(np.linalg.norm(delta)),
        }

    return {
        "urdf_path": str(urdf_path),
        "base_pose": {
            "joint_lShoulderRoll": V2_ARMS_DOWN_SHOULDER_ROLL,
            "joint_rShoulderRoll": V2_ARMS_DOWN_SHOULDER_ROLL,
        },
        "joints": joint_report,
        "calibration": {
            "joint_lShoulderRoll": {
                "arms_down_offset_rad": V2_ARMS_DOWN_SHOULDER_ROLL,
                "human_abduction_sign": 1.0,
            },
            "joint_rShoulderRoll": {
                "arms_down_offset_rad": V2_ARMS_DOWN_SHOULDER_ROLL,
                "human_abduction_sign": 1.0,
            },
            "joint_lElbow": {
                "human_flexion_sign": -1.0,
                "reason": "positive V2 left elbow moves wrist toward x-, opposite of right elbow",
            },
            "joint_rElbow": {
                "human_flexion_sign": 1.0,
                "reason": "positive V2 right elbow moves wrist toward x+",
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze V2 MuJoCo joint axis directions by small perturbations")
    parser.add_argument("--urdf", default=str(DEFAULT_URDF))
    parser.add_argument("--output", required=True)
    parser.add_argument("--delta-rad", type=float, default=DELTA_RAD)
    args = parser.parse_args()
    report = analyze_v2_axes(args.urdf, delta_rad=args.delta_rad)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "joint_count": len(report["joints"])}, indent=2))


if __name__ == "__main__":
    main()
