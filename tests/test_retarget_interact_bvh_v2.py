import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from tools.gmr_v2.interact_bvh_adapter import CANONICAL_BODY_ORDER
from tools.gmr_v2.retarget_interact_bvh_v2 import (
    episode_world_alignment,
    independent_head_fk_metrics,
    prepare_interact_target_builder,
)
from upper_body_skeleton.mujoco_playback import load_preview_model
from upper_body_skeleton.retarget_v2_18d import JOINT_ORDER_18D


def test_episode_alignment_uses_natural_onset_not_warmup_frame():
    frames = Rotation.from_euler("Z", [0.0, 90.0, 120.0], degrees=True).as_matrix()
    alignment = episode_world_alignment(frames, np.eye(3), reference_frame_index=1)
    np.testing.assert_allclose(alignment @ frames[1], np.eye(3), atol=1e-12)
    assert not np.allclose(alignment @ frames[0], np.eye(3), atol=1e-6)


def test_target_builder_keeps_unilateral_right_raise_on_right_target():
    model, _, _ = load_preview_model(joint_order=JOINT_ORDER_18D)
    index = {name: offset for offset, name in enumerate(CANONICAL_BODY_ORDER)}
    positions = np.zeros((2, len(CANONICAL_BODY_ORDER), 3), dtype=np.float64)
    positions[:, index["Chest4"]] = [0.0, 0.0, 1.0]
    positions[:, index["LeftShoulder"]] = [0.0, -0.4, 1.0]
    positions[:, index["LeftElbow"]] = [0.0, -0.4, 0.7]
    positions[:, index["LeftWrist"]] = [0.0, -0.4, 0.4]
    positions[:, index["RightShoulder"]] = [0.0, 0.4, 1.0]
    positions[0, index["RightElbow"]] = [0.0, 0.4, 0.7]
    positions[0, index["RightWrist"]] = [0.0, 0.4, 0.4]
    positions[1, index["RightElbow"]] = [0.0, 0.4, 1.3]
    positions[1, index["RightWrist"]] = [0.0, 0.4, 1.6]
    quaternions = np.zeros((2, len(CANONICAL_BODY_ORDER), 4), dtype=np.float64)
    quaternions[..., 0] = 1.0
    anatomical = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()

    _, target_for_frame, report = prepare_interact_target_builder(
        model,
        mujoco,
        positions,
        quaternions,
        anatomical,
        reference_frame_index=0,
    )
    before = target_for_frame(0)
    after = target_for_frame(1)
    assert after["RightWrist"][0][2] > before["RightWrist"][0][2]
    np.testing.assert_allclose(after["LeftWrist"][0], before["LeftWrist"][0])
    assert report["left_right_identity_swapped"] is False


def test_independent_head_fk_checks_real_urdf_joint_order_and_yaw_sign():
    model, _, _ = load_preview_model(joint_order=JOINT_ORDER_18D)
    trajectory = np.zeros((3, len(JOINT_ORDER_18D)), dtype=np.float64)
    head = np.array(
        [[0.0, 0.0, 0.0], [0.2, -0.1, 0.3], [-0.15, 0.25, -0.2]],
        dtype=np.float64,
    )
    trajectory[:, -3:] = head
    desired = Rotation.from_euler(
        "XYZ", np.column_stack((head[:, 0], head[:, 1], -head[:, 2]))
    ).as_matrix()
    metrics = independent_head_fk_metrics(model, mujoco, trajectory, desired)
    assert metrics["independent_head_fk_direction_pass"] is True
    assert metrics["independent_head_fk_error_max_deg"] < 1e-6
