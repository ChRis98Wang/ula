import math

from upper_body_skeleton.extract import (
    SMPLH_BODY_INDEX,
    UpperBodyFrameBuilder,
    axis_angle_to_matrix,
    build_zero_pose,
    build_urdf_zero_frame_from_keypoints,
    frame_indices,
)


def assert_vec_close(actual, expected, tol=1e-6):
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert abs(a - e) <= tol


def test_zero_pose_root_matches_urdf_zero_and_excludes_lower_body():
    builder = UpperBodyFrameBuilder()
    frame = builder.build_frame(
        frame_index=0,
        time_sec=0.0,
        body_pose=build_zero_pose(),
        global_orient=[0.0, 0.0, 0.0],
        translation=[1.0, 2.0, 3.0],
        valid=True,
    )

    assert_vec_close(frame["root"]["position"], [0.0, 0.0, 0.0])
    assert frame["root"]["coordinate_type"] == "urdf_zero_upper_body_local"
    assert "left_knee" not in frame["landmarks_3d"]
    assert "right_ankle" not in frame["landmarks_3d"]
    assert set(frame["landmarks_3d"]).issuperset(
        {
            "pelvis_origin",
            "torso_origin",
            "neck",
            "head",
            "left_shoulder",
            "left_elbow",
            "left_wrist",
            "right_shoulder",
            "right_elbow",
            "right_wrist",
        }
    )


def test_zero_pose_has_symmetric_reasonable_arm_lengths():
    builder = UpperBodyFrameBuilder()
    frame = builder.build_frame(
        frame_index=0,
        time_sec=0.0,
        body_pose=build_zero_pose(),
        global_orient=[0.0, 0.0, 0.0],
        translation=[0.0, 0.0, 0.0],
        valid=True,
    )
    lm = frame["landmarks_3d"]

    assert lm["left_shoulder"][1] > 0.0
    assert lm["right_shoulder"][1] < 0.0
    assert lm["left_elbow"][2] < lm["left_shoulder"][2]
    assert lm["right_elbow"][2] < lm["right_shoulder"][2]

    left_upper = builder.segment_length("left_upper_arm")
    right_upper = builder.segment_length("right_upper_arm")
    assert abs(left_upper - right_upper) < 1e-9
    assert 0.15 <= left_upper <= 0.40


def test_axis_angle_rotation_changes_left_upper_arm_direction():
    builder = UpperBodyFrameBuilder()
    zero = builder.build_frame(
        frame_index=0,
        time_sec=0.0,
        body_pose=build_zero_pose(),
        global_orient=[0.0, 0.0, 0.0],
        translation=[0.0, 0.0, 0.0],
        valid=True,
    )
    pose = build_zero_pose()
    pose[SMPLH_BODY_INDEX["left_shoulder"]] = [0.0, math.pi / 2.0, 0.0]
    moved = builder.build_frame(
        frame_index=1,
        time_sec=1 / 30,
        body_pose=pose,
        global_orient=[0.0, 0.0, 0.0],
        translation=[0.0, 0.0, 0.0],
        valid=True,
    )

    zero_vec = [
        zero["landmarks_3d"]["left_elbow"][i] - zero["landmarks_3d"]["left_shoulder"][i]
        for i in range(3)
    ]
    moved_vec = [
        moved["landmarks_3d"]["left_elbow"][i] - moved["landmarks_3d"]["left_shoulder"][i]
        for i in range(3)
    ]
    assert abs(zero_vec[0] - moved_vec[0]) > 0.1
    assert abs(moved_vec[2]) < 0.05


def test_global_orientation_is_ignored_for_urdf_zero_local_skeleton():
    builder = UpperBodyFrameBuilder()
    zero = builder.build_frame(
        frame_index=0,
        time_sec=0.0,
        body_pose=build_zero_pose(),
        global_orient=[0.0, 0.0, 0.0],
        translation=[0.0, 0.0, 0.0],
        valid=True,
    )
    rotated = builder.build_frame(
        frame_index=0,
        time_sec=0.0,
        body_pose=build_zero_pose(),
        global_orient=[math.pi, 0.0, 0.0],
        translation=[0.0, 0.0, 0.0],
        valid=True,
    )

    assert_vec_close(
        rotated["landmarks_3d"]["torso_origin"],
        zero["landmarks_3d"]["torso_origin"],
    )
    assert rotated["landmarks_3d"]["head"][2] > rotated["landmarks_3d"]["torso_origin"][2]


def test_axis_angle_to_matrix_identity_and_z_quarter_turn():
    identity = axis_angle_to_matrix([0.0, 0.0, 0.0])
    assert_vec_close(identity[0], [1.0, 0.0, 0.0])
    assert_vec_close(identity[1], [0.0, 1.0, 0.0])
    assert_vec_close(identity[2], [0.0, 0.0, 1.0])

    rot = axis_angle_to_matrix([0.0, 0.0, math.pi / 2.0])
    x_rotated = [
        rot[0][0] * 1.0 + rot[0][1] * 0.0 + rot[0][2] * 0.0,
        rot[1][0] * 1.0 + rot[1][1] * 0.0 + rot[1][2] * 0.0,
        rot[2][0] * 1.0 + rot[2][1] * 0.0 + rot[2][2] * 0.0,
    ]
    assert_vec_close(x_rotated, [0.0, 1.0, 0.0], tol=1e-6)


def test_keypoint_mapping_preserves_image_aligned_upper_body_shape():
    keypoints = [[0.0, 0.0, 0.0] for _ in range(133)]
    keypoints[5] = [700.0, 300.0, 0.9]   # left shoulder appears on image right
    keypoints[6] = [300.0, 300.0, 0.9]   # right shoulder appears on image left
    keypoints[7] = [760.0, 500.0, 0.9]
    keypoints[8] = [240.0, 500.0, 0.9]
    keypoints[9] = [790.0, 700.0, 0.9]
    keypoints[10] = [210.0, 700.0, 0.9]
    keypoints[11] = [620.0, 760.0, 0.8]
    keypoints[12] = [380.0, 760.0, 0.8]

    frame = build_urdf_zero_frame_from_keypoints(
        frame_index=0,
        time_sec=0.0,
        keypoints=keypoints,
        image_size=[1000, 1000],
        valid=True,
    )
    lm = frame["landmarks_3d"]

    assert_vec_close(frame["root"]["position"], [0.0, 0.0, 0.0])
    assert lm["left_shoulder"][1] > 0.0
    assert lm["right_shoulder"][1] < 0.0
    assert lm["left_elbow"][2] < lm["left_shoulder"][2]
    assert lm["right_elbow"][2] < lm["right_shoulder"][2]
    assert abs(lm["left_shoulder"][1] + lm["right_shoulder"][1]) < 1e-6
    assert frame["confidence"]["left_wrist"] == 0.9
    assert frame["features"]["left_elbow_flexion_rad"] > 0.0


def test_frame_indices_support_start_frame_stride_and_max_frames():
    assert list(frame_indices(total=10, start_frame=3, stride=2, max_frames=3)) == [3, 5, 7]
    assert list(frame_indices(total=5, start_frame=9, stride=1, max_frames=None)) == []
