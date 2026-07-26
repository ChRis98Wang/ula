import numpy as np

from tools.human_motion_review import render_interact_axis_review as REVIEW


def test_front_projection_matches_real_mujoco_camera_screen_right_axis():
    positions = np.array(
        [[0.0, -0.25, 0.0], [0.0, 0.25, 0.0]], dtype=np.float64
    )
    horizontal = REVIEW._horizontal_values(positions, "front")
    assert horizontal[0] < horizontal[1]
    np.testing.assert_array_equal(
        REVIEW.ROBOT_FRONT_CAMERA_SCREEN_RIGHT_AXIS,
        np.array([0.0, 1.0, 0.0]),
    )


def test_side_projection_keeps_robot_forward_as_screen_right():
    positions = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    horizontal = REVIEW._horizontal_values(positions, "side")
    assert horizontal.tolist() == [-0.5, 0.5]
