import numpy as np
import mujoco

from upper_body_skeleton.retarget_v2_ik import (
    ARM_POINT_CLEARANCE,
    IK_JOINTS,
    ROBOT_ARM_NEUTRAL_DEPTH,
    ROBOT_FOREARM_LENGTH,
    ROBOT_TORSO_FRONT_X,
    ROBOT_CROSS_BODY_DEPTH_OFFSET,
    ROBOT_UPPER_ARM_LENGTH,
    arm_clearance_penalty,
    CONTACT_CLEARANCE_MARGIN,
    contact_violation_penalty,
    projected_target_delta,
    normalized_arm_targets,
    torso_front_penalty,
    retarget_payload_to_rows_ik,
)
from upper_body_skeleton.retarget_v2 import JOINT_ORDER
from upper_body_skeleton.v2_axis_calibration import DEFAULT_URDF, resolve_mujoco_urdf


def test_normalized_arm_targets_map_video_left_right_into_v2_coordinate_sides():
    landmarks = {
        "left_shoulder": [0.0, 0.20, 0.50],
        "left_elbow": [0.0, 0.10, 0.35],
        "left_wrist": [0.0, 0.02, 0.25],
        "right_shoulder": [0.0, -0.20, 0.50],
        "right_elbow": [0.0, -0.10, 0.35],
        "right_wrist": [0.0, -0.02, 0.25],
    }

    targets = normalized_arm_targets(landmarks)

    assert targets["left_shoulder"][1] < targets["right_shoulder"][1]
    assert targets["left_wrist"][1] > targets["left_shoulder"][1]
    assert targets["right_wrist"][1] < targets["right_shoulder"][1]
    assert targets["left_elbow"][2] < targets["left_shoulder"][2]
    assert targets["right_elbow"][2] < targets["right_shoulder"][2]


def test_normalized_arm_targets_scale_video_segments_to_v2_arm_lengths():
    landmarks = {
        "left_shoulder": [0.0, 0.20, 0.50],
        "left_elbow": [0.0, -0.20, 0.20],
        "left_wrist": [0.0, -0.50, 0.10],
        "right_shoulder": [0.0, -0.20, 0.50],
        "right_elbow": [0.0, 0.20, 0.20],
        "right_wrist": [0.0, 0.50, 0.10],
    }

    targets = normalized_arm_targets(landmarks)

    for side in ("left", "right"):
        shoulder = targets[f"{side}_shoulder"]
        elbow = targets[f"{side}_elbow"]
        wrist = targets[f"{side}_wrist"]
        assert np.isclose(np.linalg.norm((elbow - shoulder)[[1, 2]]), ROBOT_UPPER_ARM_LENGTH)
        assert np.isclose(np.linalg.norm((wrist - elbow)[[1, 2]]), ROBOT_FOREARM_LENGTH)


def test_cross_body_targets_use_depth_layers_to_avoid_wrist_overlap():
    landmarks = {
        "left_shoulder": [0.0, 0.20, 0.50],
        "left_elbow": [0.0, 0.10, 0.35],
        "left_wrist": [0.0, -0.04, 0.34],
        "right_shoulder": [0.0, -0.20, 0.50],
        "right_elbow": [0.0, -0.10, 0.35],
        "right_wrist": [0.0, 0.04, 0.34],
    }

    targets = normalized_arm_targets(landmarks)

    assert targets["left_wrist"][0] > ROBOT_ARM_NEUTRAL_DEPTH
    assert targets["right_wrist"][0] < ROBOT_ARM_NEUTRAL_DEPTH
    assert targets["left_wrist"][0] - targets["right_wrist"][0] > ROBOT_CROSS_BODY_DEPTH_OFFSET
    assert targets["right_wrist"][0] > ROBOT_TORSO_FRONT_X


def test_normalized_arm_targets_keep_real_close_wrist_frames_close():
    landmarks = {
        "left_shoulder": [0.0, 0.1332473156104575, 0.3818654019182112],
        "left_elbow": [0.0, 0.14964312396218032, 0.17577304891482604],
        "left_wrist": [0.0, 0.04939654444305562, 0.0383874744694212],
        "right_shoulder": [0.0, -0.15574520455475324, 0.36474957651895235],
        "right_elbow": [0.0, -0.17876356762281423, 0.1513540395255078],
        "right_wrist": [0.0, -0.02762881651556784, -0.018018939399902456],
    }

    source_yz_distance = np.linalg.norm(
        (np.array(landmarks["left_wrist"]) - np.array(landmarks["right_wrist"]))[[1, 2]]
    )
    targets = normalized_arm_targets(landmarks)
    target_yz_distance = np.linalg.norm(
        (targets["left_wrist"] - targets["right_wrist"])[[1, 2]]
    )

    assert source_yz_distance < 0.10
    assert target_yz_distance < 0.08


def test_projected_target_delta_prioritizes_video_plane_over_unknown_depth():
    robot_point = np.array([0.40, 0.12, 0.31])
    source_target = np.array([0.00, 0.10, 0.35])

    delta = projected_target_delta(robot_point, source_target)

    assert np.allclose(delta, [0.02, -0.04])


def test_arm_clearance_penalty_penalizes_overlapping_left_right_arm_points():
    points = {
        "left_elbow": np.array([0.08, -0.03, 0.08]),
        "left_wrist": np.array([0.08, 0.00, 0.05]),
        "right_elbow": np.array([0.08, 0.03, 0.08]),
        "right_wrist": np.array([0.08, 0.01, 0.05]),
    }
    separated = dict(points)
    separated["right_elbow"] = np.array([0.08 + ARM_POINT_CLEARANCE * 2.0, 0.03, 0.08])
    separated["right_wrist"] = np.array([0.08 + ARM_POINT_CLEARANCE * 2.0, 0.01, 0.05])

    assert arm_clearance_penalty(points) > 0.0
    assert arm_clearance_penalty(separated) == 0.0


def test_torso_front_penalty_penalizes_arm_points_inside_chest_front():
    points = {
        "left_elbow": np.array([ROBOT_TORSO_FRONT_X - 0.02, 0.00, 0.08]),
        "left_wrist": np.array([ROBOT_TORSO_FRONT_X + 0.04, 0.12, 0.05]),
        "right_elbow": np.array([ROBOT_TORSO_FRONT_X + 0.04, 0.12, 0.05]),
        "right_wrist": np.array([ROBOT_TORSO_FRONT_X + 0.04, 0.12, 0.05]),
    }

    assert torso_front_penalty(points) > 0.0

    points["left_elbow"] = np.array([ROBOT_TORSO_FRONT_X + 0.04, 0.00, 0.08])
    assert torso_front_penalty(points) == 0.0


def test_contact_violation_penalty_only_counts_arm_collisions():
    contacts = [
        ("link_lWristPitch", "link_rWristPitch", -0.03),
        ("link_lElbow", "link_torso", -0.02),
        ("link_pelvisYaw", "link_torso", -0.05),
        ("link_lWristPitch", "link_rWristPitch", 0.02),
    ]

    assert contact_violation_penalty(contacts) > 0.0
    assert contact_violation_penalty([contacts[2]]) == 0.0
    assert contact_violation_penalty([contacts[3]]) == 0.0


def test_contact_violation_penalty_enforces_positive_clearance_margin():
    near_clearance = CONTACT_CLEARANCE_MARGIN * 0.5
    safe_clearance = CONTACT_CLEARANCE_MARGIN * 1.5

    assert contact_violation_penalty([("link_lWristPitch", "link_rWristPitch", near_clearance)]) > 0.0
    assert contact_violation_penalty([("link_lWristPitch", "link_rWristPitch", safe_clearance)]) == 0.0


def test_ik_joints_include_wrist_dofs_for_collision_clearance():
    assert "joint_lWristRoll" in IK_JOINTS
    assert "joint_lWristPitch" in IK_JOINTS
    assert "joint_rWristRoll" in IK_JOINTS
    assert "joint_rWristPitch" in IK_JOINTS


def test_ik_retarget_keeps_adjacent_joint_steps_continuous():
    def frame(time_sec, left_wrist_y, right_wrist_y):
        return {
            "time_sec": time_sec,
            "landmarks_3d": {
                "left_shoulder": [0.0, 0.20, 0.50],
                "left_elbow": [0.0, 0.10, 0.35],
                "left_wrist": [0.0, left_wrist_y, 0.30],
                "right_shoulder": [0.0, -0.20, 0.50],
                "right_elbow": [0.0, -0.10, 0.35],
                "right_wrist": [0.0, right_wrist_y, 0.30],
            },
        }

    payload = {
        "frames": [
            frame(0.0, 0.08, -0.08),
            frame(1.0 / 30.0, -0.02, 0.02),
            frame(2.0 / 30.0, -0.04, 0.04),
        ]
    }

    rows = retarget_payload_to_rows_ik(payload)

    for previous, current in zip(rows, rows[1:]):
        for joint in IK_JOINTS:
            assert abs(current[joint] - previous[joint]) <= 0.25


def test_ik_retarget_brings_robot_wrists_together_when_source_wrists_meet():
    payload = {
        "frames": [
            {
                "time_sec": 0.0,
                "landmarks_3d": {
                    "left_shoulder": [0.0, 0.20, 0.50],
                    "left_elbow": [0.0, 0.10, 0.35],
                    "left_wrist": [0.0, 0.02, 0.30],
                    "right_shoulder": [0.0, -0.20, 0.50],
                    "right_elbow": [0.0, -0.10, 0.35],
                    "right_wrist": [0.0, -0.02, 0.30],
                },
            }
        ]
    }

    row = retarget_payload_to_rows_ik(payload)[0]
    model = mujoco.MjModel.from_xml_path(str(resolve_mujoco_urdf(DEFAULT_URDF)))
    data = mujoco.MjData(model)
    qpos_addr = {model.joint(i).name: int(model.joint(i).qposadr[0]) for i in range(model.njnt)}
    body_ids = {model.body(i).name: i for i in range(model.nbody)}
    data.qpos[:] = 0.0
    for joint in JOINT_ORDER:
        data.qpos[qpos_addr[joint]] = row[joint]
    mujoco.mj_forward(model, data)
    left_wrist = data.xpos[body_ids["link_lWristPitch"]]
    right_wrist = data.xpos[body_ids["link_rWristPitch"]]

    assert np.linalg.norm((left_wrist - right_wrist)[[1, 2]]) < 0.09


def test_ik_retarget_preserves_real_close_wrist_frame_from_preview_sample():
    frame = {
        "time_sec": 10.3,
        "landmarks_3d": {
            "left_shoulder": [0.0, 0.1332473156104575, 0.3818654019182112],
            "left_elbow": [0.0, 0.14964312396218032, 0.17577304891482604],
            "left_wrist": [0.0, 0.04939654444305562, 0.0383874744694212],
            "right_shoulder": [0.0, -0.15574520455475324, 0.36474957651895235],
            "right_elbow": [0.0, -0.17876356762281423, 0.1513540395255078],
            "right_wrist": [0.0, -0.02762881651556784, -0.018018939399902456],
        },
    }

    row = retarget_payload_to_rows_ik({"frames": [frame]})[0]
    model = mujoco.MjModel.from_xml_path(str(resolve_mujoco_urdf(DEFAULT_URDF)))
    data = mujoco.MjData(model)
    qpos_addr = {model.joint(i).name: int(model.joint(i).qposadr[0]) for i in range(model.njnt)}
    body_ids = {model.body(i).name: i for i in range(model.nbody)}
    data.qpos[:] = 0.0
    for joint in JOINT_ORDER:
        data.qpos[qpos_addr[joint]] = row[joint]
    mujoco.mj_forward(model, data)
    left_wrist = data.xpos[body_ids["link_lWristPitch"]]
    right_wrist = data.xpos[body_ids["link_rWristPitch"]]

    assert np.linalg.norm((left_wrist - right_wrist)[[1, 2]]) < 0.09
