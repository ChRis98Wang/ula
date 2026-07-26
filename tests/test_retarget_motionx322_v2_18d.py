import numpy as np
from scipy.spatial.transform import Rotation

from tools.gmr_v2.retarget_motionx322_v2 import (
    canonical_head_relative_rotations,
    decompose_v2_head_rotations,
    head_quality_metrics,
    reconstruct_v2_head_rotations,
)
from upper_body_skeleton.retarget_v2 import JOINT_ORDER
from upper_body_skeleton.retarget_v2_18d import (
    HEAD_JOINT_ORDER,
    JOINT_ORDER_18D,
)


def test_18d_contract_is_append_only():
    assert JOINT_ORDER_18D[: len(JOINT_ORDER)] == JOINT_ORDER
    assert JOINT_ORDER_18D[len(JOINT_ORDER) :] == HEAD_JOINT_ORDER


def test_v2_head_decomposition_uses_negative_z_yaw_axis():
    expected = np.array([[0.20, -0.30, 0.40], [0.22, -0.28, 0.45]])
    matrices = Rotation.from_euler(
        "XYZ",
        np.column_stack((expected[:, 0], expected[:, 1], -expected[:, 2])),
    ).as_matrix()

    actual = decompose_v2_head_rotations(matrices)

    assert np.allclose(actual, expected, atol=1e-7)
    assert np.allclose(reconstruct_v2_head_rotations(actual), matrices, atol=1e-7)


def test_head_chain_applies_front_reflection_to_spine3_neck_head_rotation():
    frames = 2
    local = np.zeros((frames, 16, 3), dtype=np.float64)
    parents = np.zeros(16, dtype=np.int64)
    parents[0] = -1
    parents[9] = 0
    parents[12] = 9
    parents[15] = 12
    alignment = np.diag([-1.0, 1.0, 1.0])
    canonical = Rotation.from_euler("XYZ", [[0.1, 0.2, -0.3], [0.12, 0.18, -0.28]]).as_matrix()
    source = alignment.T @ canonical @ alignment
    local[:, 12] = Rotation.from_matrix(source).as_rotvec()

    actual = canonical_head_relative_rotations(local, parents, alignment)

    assert np.allclose(actual, canonical, atol=1e-7)


def test_head_quality_gate_checks_direction_limits_velocity_and_continuity():
    head = np.array([[0.0, 0.0, 0.0], [0.02, -0.03, 0.04], [0.04, -0.05, 0.07]])
    source = reconstruct_v2_head_rotations(head)
    trajectory = np.zeros((3, len(JOINT_ORDER_18D)), dtype=np.float64)
    trajectory[:, -3:] = head

    metrics = head_quality_metrics(source, trajectory, trajectory, fps=30.0, max_velocity=3.0)

    assert metrics["head_joint_limits_pass"] is True
    assert metrics["head_velocity_pass"] is True
    assert metrics["head_direction_pass"] is True
    assert metrics["head_continuity_pass"] is True
    assert metrics["head_yaw_sign_convention"] == "qpos_yaw_is_negative_canonical_Z_euler"
