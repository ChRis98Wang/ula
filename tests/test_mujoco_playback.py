import csv
from pathlib import Path

import numpy as np

from upper_body_skeleton.mujoco_playback import build_preview_model, read_joint_csv, render_motion
from upper_body_skeleton.retarget_v2 import JOINT_ORDER
from upper_body_skeleton.v2_axis_calibration import DEFAULT_URDF


def test_preview_model_uses_original_v2_urdf_by_default():
    model, joint_to_qpos = build_preview_model()

    assert Path(DEFAULT_URDF).exists()
    assert model.nq > len(JOINT_ORDER)
    assert model.nbody >= 20
    assert sorted(joint_to_qpos) == list(range(len(JOINT_ORDER)))


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
