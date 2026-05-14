import csv

import numpy as np

from upper_body_skeleton.mujoco_playback import build_preview_model, read_joint_csv
from upper_body_skeleton.retarget_v2 import JOINT_ORDER


def test_preview_model_has_all_v2_upper_body_joints():
    model, joint_to_qpos = build_preview_model()

    assert model.nq == len(JOINT_ORDER)
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
