#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np

from upper_body_skeleton.retarget_v2 import JOINT_LIMITS, JOINT_ORDER, retarget_frame, write_joint_csv
from upper_body_skeleton.v2_axis_calibration import DEFAULT_URDF


IK_JOINTS = [
    "joint_lShoulderPitch",
    "joint_lShoulderRoll",
    "joint_lShoulderYaw",
    "joint_lElbow",
    "joint_lWristRoll",
    "joint_lWristPitch",
    "joint_rShoulderPitch",
    "joint_rShoulderRoll",
    "joint_rShoulderYaw",
    "joint_rElbow",
    "joint_rWristRoll",
    "joint_rWristPitch",
]

BODY_TARGETS = {
    "left_elbow": "link_lElbow",
    "left_wrist": "link_lWristPitch",
    "right_elbow": "link_rElbow",
    "right_wrist": "link_rWristPitch",
}

ROBOT_SHOULDER_ANCHORS = {
    "left_shoulder": np.array([0.0, -0.115, 0.103]),
    "right_shoulder": np.array([0.0, 0.115, 0.103]),
}

ROBOT_UPPER_ARM_LENGTH = 0.103
ROBOT_FOREARM_LENGTH = 0.092
ROBOT_ARM_NEUTRAL_DEPTH = 0.12
ROBOT_CROSS_BODY_DEPTH_OFFSET = 0.05
ROBOT_TORSO_FRONT_X = 0.055
ARM_POINT_CLEARANCE = 0.075
VIDEO_PLANE_AXES = np.array([1, 2], dtype=int)
UNKNOWN_DEPTH_WEIGHT = 0.35
ARM_CLEARANCE_WEIGHT = 3.5
TORSO_FRONT_WEIGHT = 5.0
CONTACT_VIOLATION_WEIGHT = 220.0
CONTACT_CLEARANCE_MARGIN = 0.006

LEFT_ARM_PREFIXES = ("link_lShoulder", "link_lElbow", "link_lWrist")
RIGHT_ARM_PREFIXES = ("link_rShoulder", "link_rElbow", "link_rWrist")
TORSO_BODY_NAMES = {"link_torso", "link_pelvisTorso", "link_pelvisYaw", "link_pelvisPitch"}


def _vec(point):
    return np.asarray(point, dtype=float)


def normalized_arm_targets(landmarks):
    targets = {
        "left_shoulder": ROBOT_SHOULDER_ANCHORS["left_shoulder"].copy(),
        "right_shoulder": ROBOT_SHOULDER_ANCHORS["right_shoulder"].copy(),
    }
    for side in ("left", "right"):
        source_shoulder = _vec(landmarks[f"{side}_shoulder"])
        source_elbow = _vec(landmarks[f"{side}_elbow"])
        source_wrist = _vec(landmarks[f"{side}_wrist"])
        robot_shoulder = targets[f"{side}_shoulder"]
        mirror_y = -1.0
        upper_plane = source_elbow[VIDEO_PLANE_AXES] - source_shoulder[VIDEO_PLANE_AXES]
        fore_plane = source_wrist[VIDEO_PLANE_AXES] - source_elbow[VIDEO_PLANE_AXES]
        upper_plane = upper_plane * np.array([mirror_y, 1.0])
        fore_plane = fore_plane * np.array([mirror_y, 1.0])
        upper_direction = _unit_plane(upper_plane, np.array([0.0, -1.0]))
        fore_direction = _unit_plane(fore_plane, upper_direction)
        cross_body = _cross_body_strength_from_plane(fore_direction, side)
        depth = ROBOT_ARM_NEUTRAL_DEPTH + _depth_layer_sign(side) * ROBOT_CROSS_BODY_DEPTH_OFFSET * cross_body
        elbow = robot_shoulder.copy()
        elbow[VIDEO_PLANE_AXES] += upper_direction * ROBOT_UPPER_ARM_LENGTH
        elbow[0] = depth
        wrist = elbow.copy()
        wrist[VIDEO_PLANE_AXES] += fore_direction * ROBOT_FOREARM_LENGTH
        wrist[0] = depth
        targets[f"{side}_elbow"] = elbow
        targets[f"{side}_wrist"] = wrist
    return targets


def projected_target_delta(robot_point, source_target):
    robot_point = _vec(robot_point)
    source_target = _vec(source_target)
    return robot_point[VIDEO_PLANE_AXES] - source_target[VIDEO_PLANE_AXES]


def _unit_plane(vector, fallback):
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return np.asarray(fallback, dtype=float)
    return np.asarray(vector, dtype=float) / norm


def _cross_body_strength_from_plane(fore_direction, side):
    inward_sign = 1.0 if side == "left" else -1.0
    return float(np.clip(inward_sign * fore_direction[0], 0.0, 1.0))


def _depth_layer_sign(side):
    return 1.0 if side == "left" else -1.0


def arm_clearance_penalty(points):
    total = 0.0
    for left_name in ("left_elbow", "left_wrist"):
        for right_name in ("right_elbow", "right_wrist"):
            delta = _vec(points[left_name]) - _vec(points[right_name])
            distance = float(np.linalg.norm(delta))
            violation = max(0.0, ARM_POINT_CLEARANCE - distance)
            total += violation * violation
    return total


def torso_front_penalty(points):
    total = 0.0
    for name in ("left_elbow", "left_wrist", "right_elbow", "right_wrist"):
        point = _vec(points[name])
        in_chest_yz = abs(point[1]) < 0.09 and -0.08 < point[2] < 0.14
        if in_chest_yz:
            violation = max(0.0, ROBOT_TORSO_FRONT_X - float(point[0]))
            total += violation * violation
    return total


def contact_violation_penalty(contacts):
    total = 0.0
    for body_a, body_b, distance in contacts:
        distance = float(distance)
        if _is_relevant_collision(body_a, body_b):
            violation = max(0.0, CONTACT_CLEARANCE_MARGIN - distance)
            total += violation * violation
    return total


def _is_relevant_collision(body_a, body_b):
    return _is_left_right_arm_collision(body_a, body_b) or _is_arm_torso_collision(body_a, body_b)


def _is_left_right_arm_collision(body_a, body_b):
    return (_is_left_arm(body_a) and _is_right_arm(body_b)) or (_is_left_arm(body_b) and _is_right_arm(body_a))


def _is_arm_torso_collision(body_a, body_b):
    return (_is_arm(body_a) and body_b in TORSO_BODY_NAMES) or (_is_arm(body_b) and body_a in TORSO_BODY_NAMES)


def _is_left_arm(body_name):
    return str(body_name).startswith(LEFT_ARM_PREFIXES)


def _is_right_arm(body_name):
    return str(body_name).startswith(RIGHT_ARM_PREFIXES)


def _is_arm(body_name):
    return _is_left_arm(body_name) or _is_right_arm(body_name)


def _contacts_from_data(model, data):
    contacts = []
    for index in range(data.ncon):
        contact = data.contact[index]
        body_a = model.body(model.geom_bodyid[contact.geom1]).name
        body_b = model.body(model.geom_bodyid[contact.geom2]).name
        contacts.append((body_a, body_b, float(contact.dist)))
    return contacts


def _maps(model):
    qpos_addr = {model.joint(i).name: int(model.joint(i).qposadr[0]) for i in range(model.njnt)}
    body_ids = {model.body(i).name: i for i in range(model.nbody)}
    return qpos_addr, body_ids


def _clip_joint(name, value):
    lower, upper = JOINT_LIMITS[name]
    return float(np.clip(value, lower, upper))


def _qpos_from_row(model, qpos_addr, row):
    qpos = np.zeros(model.nq)
    for joint in JOINT_ORDER:
        if joint in qpos_addr:
            qpos[qpos_addr[joint]] = _clip_joint(joint, row.get(joint, 0.0))
    return qpos


def _loss(model, data, qpos, qpos_addr, body_ids, targets, regularization_qpos):
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    total = 0.0
    for target_name, body_name in BODY_TARGETS.items():
        if body_name not in body_ids:
            continue
        robot_point = data.xpos[body_ids[body_name]]
        plane_delta = projected_target_delta(robot_point, targets[target_name])
        depth_delta = float(robot_point[0] - targets[target_name][0])
        weight = 1.0 if "wrist" in target_name else 0.45
        total += weight * float(plane_delta @ plane_delta)
        total += weight * UNKNOWN_DEPTH_WEIGHT * depth_delta * depth_delta
    points = {
        target_name: data.xpos[body_ids[body_name]].copy()
        for target_name, body_name in BODY_TARGETS.items()
        if body_name in body_ids
    }
    if len(points) == len(BODY_TARGETS):
        total += ARM_CLEARANCE_WEIGHT * arm_clearance_penalty(points)
        total += TORSO_FRONT_WEIGHT * torso_front_penalty(points)
    total += CONTACT_VIOLATION_WEIGHT * contact_violation_penalty(_contacts_from_data(model, data))
    for joint in IK_JOINTS:
        if joint in qpos_addr:
            diff = qpos[qpos_addr[joint]] - regularization_qpos[qpos_addr[joint]]
            regularization_weight = 0.05 if "Wrist" in joint else 0.015
            total += regularization_weight * float(diff * diff)
    return total


def optimize_frame(model, data, qpos_addr, body_ids, frame, previous_qpos=None, iterations=18):
    seed_row = retarget_frame(frame)
    qpos = _qpos_from_row(model, qpos_addr, seed_row)
    if previous_qpos is not None:
        for joint in IK_JOINTS:
            if joint in qpos_addr:
                qpos[qpos_addr[joint]] = previous_qpos[qpos_addr[joint]]
    regularization = _qpos_from_row(model, qpos_addr, seed_row)
    targets = normalized_arm_targets(frame["landmarks_3d"])
    step_sizes = [0.35, 0.18, 0.09, 0.04]
    for step in step_sizes:
        for _ in range(iterations):
            improved = False
            base_loss = _loss(model, data, qpos, qpos_addr, body_ids, targets, regularization)
            for joint in IK_JOINTS:
                if joint not in qpos_addr:
                    continue
                address = qpos_addr[joint]
                best_value = qpos[address]
                best_loss = base_loss
                for direction in (-1.0, 1.0):
                    candidate = qpos.copy()
                    candidate[address] = _clip_joint(joint, candidate[address] + direction * step)
                    candidate_loss = _loss(model, data, candidate, qpos_addr, body_ids, targets, regularization)
                    if candidate_loss < best_loss:
                        best_loss = candidate_loss
                        best_value = candidate[address]
                if best_value != qpos[address]:
                    qpos[address] = best_value
                    improved = True
            if not improved:
                break
    row = dict(seed_row)
    for joint in IK_JOINTS:
        if joint in qpos_addr:
            row[joint] = _clip_joint(joint, qpos[qpos_addr[joint]])
    return row, qpos


def retarget_payload_to_rows_ik(payload, urdf_path=DEFAULT_URDF, max_frames=None):
    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    data = mujoco.MjData(model)
    qpos_addr, body_ids = _maps(model)
    rows = []
    previous_qpos = None
    frames = payload.get("frames", [])
    if max_frames is not None:
        frames = frames[:max_frames]
    for frame in frames:
        row, previous_qpos = optimize_frame(model, data, qpos_addr, body_ids, frame, previous_qpos=previous_qpos)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Retarget upper-body skeleton to V2 using MuJoCo FK-guided IK")
    parser.add_argument("skeleton_json")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--urdf", default=str(DEFAULT_URDF))
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    payload = json.loads(Path(args.skeleton_json).read_text(encoding="utf-8"))
    rows = retarget_payload_to_rows_ik(payload, urdf_path=args.urdf, max_frames=args.max_frames)
    write_joint_csv(rows, args.output_csv)
    print(json.dumps({"rows": len(rows), "output_csv": args.output_csv}, indent=2))


if __name__ == "__main__":
    main()
