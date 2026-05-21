import csv
from pathlib import Path

import numpy as np

from upper_body_skeleton.mujoco_playback import (
    MujocoMotionPlayer,
    build_preview_model,
    play_motion,
    read_joint_csv,
    render_motion,
)
from upper_body_skeleton.retarget_v2 import JOINT_ORDER
from upper_body_skeleton.v2_axis_calibration import DEFAULT_URDF, NEW_URDF_SOURCE, ensure_mujoco_urdf


def test_preview_model_uses_original_v2_urdf_by_default():
    model, joint_to_qpos = build_preview_model()

    assert Path(DEFAULT_URDF).exists()
    assert str(DEFAULT_URDF).endswith("robot_modify_meshdir.urdf")
    assert model.nq > len(JOINT_ORDER)
    assert model.nbody >= 20
    assert sorted(joint_to_qpos) == list(range(len(JOINT_ORDER)))


def test_new_0514_robot_modify_urdf_is_rewritten_for_mujoco_mesh_loading():
    mujoco_urdf = ensure_mujoco_urdf(NEW_URDF_SOURCE)
    text = Path(mujoco_urdf).read_text(encoding="utf-8")

    assert Path(NEW_URDF_SOURCE).name == "robot_modify.urdf"
    assert Path(mujoco_urdf).name == "robot_modify_meshdir.urdf"
    assert "package://" not in text
    assert "link_lShoulderPitch.STL" in text
    assert str(Path(NEW_URDF_SOURCE).parents[2] / "TorsoArm_urdf" / "meshes") in text


def test_new_0514_head_meshes_are_converted_instead_of_placeholder():
    mujoco_urdf = ensure_mujoco_urdf(NEW_URDF_SOURCE)
    text = Path(mujoco_urdf).read_text(encoding="utf-8")
    asset_root = Path(NEW_URDF_SOURCE).parents[2] / ".mujoco_assets"

    assert "small_placeholder" not in text
    assert str(asset_root / "base_link.obj") in text
    assert str(asset_root / "head_yaw_Link.obj") in text
    assert (asset_root / "base_link.obj").is_file()
    assert (asset_root / "head_yaw_Link.obj").is_file()


def test_read_joint_csv_returns_joint_matrix(tmp_path):
    path = tmp_path / "motion.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["time_sec"] + JOINT_ORDER)
        writer.writeheader()
        for frame in range(3):
            row = {"time_sec": frame / 30.0}
            row.update({joint: 0.1 * frame for joint in JOINT_ORDER})
            writer.writerow(row)

    values = read_joint_csv(path)

    assert values.shape == (3, len(JOINT_ORDER))
    assert np.allclose(values[2], 0.2)


def test_render_motion_uses_original_v2_at_requested_resolution(tmp_path):
    path = tmp_path / "motion.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["time_sec"] + JOINT_ORDER)
        writer.writeheader()
        for frame in range(2):
            row = {"time_sec": frame / 30.0}
            row.update({joint: 0.0 for joint in JOINT_ORDER})
            writer.writerow(row)

    summary = render_motion(path, tmp_path / "preview.mp4", width=800, height=450)

    assert summary["model_source"] == str(DEFAULT_URDF)
    assert summary["model_nq"] > len(JOINT_ORDER)
    assert summary["frames"] == 2
    assert summary["width"] == 800
    assert summary["height"] == 450
    assert (tmp_path / "preview.mp4").is_file()


def test_play_motion_streams_joint_frames_to_mujoco_viewer(tmp_path, monkeypatch):
    path = tmp_path / "motion.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["time_sec"] + JOINT_ORDER)
        writer.writeheader()
        for frame in range(3):
            row = {"time_sec": frame / 30.0}
            row.update({joint: 0.05 * frame for joint in JOINT_ORDER})
            writer.writerow(row)

    sync_calls = []

    class FakeViewer:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def is_running(self):
            return True

        def sync(self):
            sync_calls.append(1)

    monkeypatch.setattr("upper_body_skeleton.mujoco_playback._launch_passive_viewer", lambda model, data: FakeViewer())

    summary = play_motion(path, loops=1, realtime=False)

    assert summary["frames"] == 3
    assert summary["frames_played"] == 3
    assert summary["loops_completed"] == 1
    assert len(sync_calls) == 3


def test_reusable_player_streams_multiple_trajectories_through_one_viewer(monkeypatch):
    launch_calls = []
    sync_calls = []

    class FakeViewer:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def is_running(self):
            return True

        def sync(self):
            sync_calls.append(1)

    def fake_launch(model, data):
        launch_calls.append((model, data))
        return FakeViewer()

    monkeypatch.setattr("upper_body_skeleton.mujoco_playback._launch_passive_viewer", fake_launch)

    first = np.zeros((2, len(JOINT_ORDER)), dtype=np.float32)
    second = np.ones((3, len(JOINT_ORDER)), dtype=np.float32) * 0.05

    with MujocoMotionPlayer(simplified=True) as player:
        first_summary = player.play_trajectory(first, loops=1, realtime=False)
        second_summary = player.play_trajectory(second, loops=1, realtime=False)

    assert len(launch_calls) == 1
    assert first_summary["frames_played"] == 2
    assert second_summary["frames_played"] == 3
    assert len(sync_calls) == 5


def test_player_can_stop_playback_mid_trajectory(monkeypatch):
    sync_calls = []

    class StopAfterTwo:
        def __init__(self):
            self.calls = 0

        def is_set(self):
            self.calls += 1
            return self.calls > 2

    class FakeViewer:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def is_running(self):
            return True

        def sync(self):
            sync_calls.append(1)

    monkeypatch.setattr("upper_body_skeleton.mujoco_playback._launch_passive_viewer", lambda model, data: FakeViewer())

    trajectory = np.zeros((10, len(JOINT_ORDER)), dtype=np.float32)
    with MujocoMotionPlayer(simplified=True) as player:
        summary = player.play_trajectory(trajectory, loops=1, realtime=False, stop_event=StopAfterTwo())

    assert summary["interrupted"] is True
    assert summary["frames_played"] == 1
    assert len(sync_calls) == 1
