import csv
import math
from pathlib import Path

import mujoco
import numpy as np

from upper_body_skeleton.mujoco_playback import (
    MujocoMotionPlayer,
    apply_camera_lookat_z_offset,
    build_preview_model,
    fit_full_body_camera,
    load_preview_model,
    play_motion,
    read_joint_csv,
    render_motion,
)
from upper_body_skeleton.retarget_v2 import JOINT_ORDER
from upper_body_skeleton.retarget_v2_18d import JOINT_ORDER_18D
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


def test_read_joint_csv_auto_detects_versioned_18d_contract(tmp_path):
    path = tmp_path / "motion_18d.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=JOINT_ORDER_18D)
        writer.writeheader()
        writer.writerow({joint: 0.01 * index for index, joint in enumerate(JOINT_ORDER_18D)})

    values = read_joint_csv(path)

    assert values.shape == (1, len(JOINT_ORDER_18D))
    assert np.isclose(values[0, JOINT_ORDER_18D.index("head_yaw_joint")], 0.17)


def test_preview_model_maps_18d_head_joints_and_negative_yaw_axis():
    model, joint_to_qpos, _ = load_preview_model(joint_order=JOINT_ORDER_18D)
    yaw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "head_yaw_joint")

    assert len(joint_to_qpos) == len(JOINT_ORDER_18D)
    assert joint_to_qpos[JOINT_ORDER_18D.index("head_roll_joint")] == 15
    assert joint_to_qpos[JOINT_ORDER_18D.index("head_pitch_joint")] == 16
    assert joint_to_qpos[JOINT_ORDER_18D.index("head_yaw_joint")] == 17
    assert np.allclose(model.jnt_axis[yaw_id], [0.0, 0.0, -1.0])


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
    assert summary["camera_framing"]["mode"] == "auto_full_body"
    assert np.allclose(summary["camera_lookat"], summary["camera_framing"]["bounds_center"])
    assert summary["camera_distance"] == summary["camera_framing"]["auto_distance"]
    assert (tmp_path / "preview.mp4").is_file()


def test_full_body_camera_fits_union_of_all_trajectory_poses():
    model, joint_to_qpos, _ = load_preview_model(simplified=True)
    neutral = np.zeros((1, len(JOINT_ORDER)), dtype=np.float32)
    raised = neutral.copy()
    raised[0, JOINT_ORDER.index("joint_lShoulderPitch")] = 1.4
    raised[0, JOINT_ORDER.index("joint_rShoulderPitch")] = -1.4
    combined = np.concatenate([neutral, raised], axis=0)

    def fit(values):
        return fit_full_body_camera(
            model,
            mujoco.MjData(model),
            values,
            joint_to_qpos,
            width=1280,
            height=720,
            margin=1.12,
        )

    camera, framing = fit(combined)
    _, neutral_framing = fit(neutral)
    _, raised_framing = fit(raised)
    lower = np.asarray(framing["bounds_min"])
    upper = np.asarray(framing["bounds_max"])

    assert np.all(lower <= np.minimum(neutral_framing["bounds_min"], raised_framing["bounds_min"]))
    assert np.all(upper >= np.maximum(neutral_framing["bounds_max"], raised_framing["bounds_max"]))
    assert np.allclose(camera.lookat, 0.5 * (lower + upper))
    limiting_half_fov = min(
        math.radians(framing["vertical_fov_deg"]) / 2.0,
        math.atan(math.tan(math.radians(framing["vertical_fov_deg"]) / 2.0) * 1280 / 720),
    )
    required_distance = framing["margin"] * framing["bounds_radius"] / math.sin(limiting_half_fov)
    assert camera.distance >= required_distance - 1e-12


def test_full_body_camera_rejects_margin_that_can_crop_bounds():
    model, joint_to_qpos, _ = load_preview_model(simplified=True)
    trajectory = np.zeros((1, len(JOINT_ORDER)), dtype=np.float32)

    try:
        fit_full_body_camera(
            model,
            mujoco.MjData(model),
            trajectory,
            joint_to_qpos,
            width=1280,
            height=720,
            margin=0.99,
        )
    except ValueError as exc:
        assert "margin" in str(exc)
    else:
        raise AssertionError("camera fit accepted a margin below 1.0")


def test_camera_lookat_z_offset_moves_subject_up_within_reserved_margin():
    model, joint_to_qpos, _ = load_preview_model(simplified=True)
    trajectory = np.zeros((1, len(JOINT_ORDER)), dtype=np.float32)
    camera, framing = fit_full_body_camera(
        model,
        mujoco.MjData(model),
        trajectory,
        joint_to_qpos,
        width=1280,
        height=720,
        margin=1.25,
    )
    original_z = float(camera.lookat[2])

    camera, framing = apply_camera_lookat_z_offset(camera, framing, -0.08)

    assert np.isclose(camera.lookat[2], original_z - 0.08)
    assert framing["lookat_z_offset"] == -0.08
    assert framing["subject_screen_bias"] == "up"
    assert framing["lookat_z_offset_limit"] >= 0.08


def test_camera_lookat_z_offset_rejects_shift_outside_reserved_margin():
    model, joint_to_qpos, _ = load_preview_model(simplified=True)
    trajectory = np.zeros((1, len(JOINT_ORDER)), dtype=np.float32)
    camera, framing = fit_full_body_camera(
        model,
        mujoco.MjData(model),
        trajectory,
        joint_to_qpos,
        width=1280,
        height=720,
        margin=1.01,
    )

    try:
        apply_camera_lookat_z_offset(camera, framing, -0.08)
    except ValueError as exc:
        assert "reserved full-body framing margin" in str(exc)
    else:
        raise AssertionError("camera accepted a look-at shift outside its framing margin")


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
