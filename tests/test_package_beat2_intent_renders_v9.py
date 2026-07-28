from __future__ import annotations

import csv

import numpy as np

from tools.human_motion_review.package_beat2_intent_renders_v9 import (
    trajectory_motion_check,
)


def test_trajectory_motion_check_detects_real_motion(tmp_path) -> None:
    path = tmp_path / "motion.csv"
    values = np.zeros((4, 18), dtype=np.float64)
    values[:, 3] = np.radians([0.0, 1.0, 2.0, 3.0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"j{i}" for i in range(18)])
        writer.writerows(values)
    result = trajectory_motion_check(path)
    assert result["has_motion"] is True
    assert result["moving_joint_count_0p5deg"] == 1
    assert result["max_joint_excursion_deg"] == 3.0


def test_trajectory_motion_check_rejects_static_motion(tmp_path) -> None:
    path = tmp_path / "static.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"j{i}" for i in range(18)])
        writer.writerows(np.zeros((3, 18), dtype=np.float64))
    assert trajectory_motion_check(path)["has_motion"] is False
