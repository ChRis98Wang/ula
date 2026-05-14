from unittest.mock import Mock

import numpy as np

from upper_body_skeleton.side_by_side_preview import (
    build_front_camera,
    build_upper_body_camera,
    concat_side_by_side,
    preview_output_path,
    skip_video_frames,
    update_renderer_scene,
)


def test_preview_output_path_flattens_sample_key_safely(tmp_path):
    output = preview_output_path(
        "improvised__dev__0000__0000/V00_S0644_I00000129_P0799",
        tmp_path,
    )

    assert output == tmp_path / "improvised__dev__0000__0000__V00_S0644_I00000129_P0799.original_vs_v2_mujoco.mp4"


def test_concat_side_by_side_resizes_both_frames_to_target_height():
    original = np.zeros((100, 50, 3), dtype=np.uint8)
    robot = np.ones((80, 80, 3), dtype=np.uint8) * 255

    combined = concat_side_by_side(original, robot, target_height=40)

    assert combined.shape == (40, 60, 3)
    assert combined[:, :20].mean() == 0
    assert combined[:, 20:].mean() == 255


def test_update_renderer_scene_uses_default_camera_without_passing_none():
    renderer = Mock()
    data = object()

    update_renderer_scene(renderer, data)

    renderer.update_scene.assert_called_once_with(data)


def test_update_renderer_scene_passes_explicit_front_camera():
    renderer = Mock()
    data = object()
    camera = object()

    update_renderer_scene(renderer, data, camera=camera)

    renderer.update_scene.assert_called_once_with(data, camera=camera)


def test_build_front_camera_faces_robot_from_front():
    camera = build_front_camera()

    assert camera.azimuth == 180
    assert camera.distance > 1.0
    assert camera.lookat[2] > 0.0


def test_build_upper_body_camera_is_front_facing_and_closer():
    front = build_front_camera()
    upper = build_upper_body_camera()

    assert upper.azimuth == front.azimuth
    assert upper.distance < front.distance
    assert 0.05 < upper.lookat[2] < front.lookat[2]


def test_skip_video_frames_advances_reader():
    reader = iter([0, 1, 2, 3])

    skip_video_frames(reader, 2)

    assert next(reader) == 2


def test_skip_video_frames_supports_imageio_reader_get_data():
    class Reader:
        def __init__(self):
            self.requested = []

        def get_data(self, index):
            self.requested.append(index)

    reader = Reader()

    skip_video_frames(reader, 3)

    assert reader.requested == [0, 1, 2]
