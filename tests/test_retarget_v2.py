import csv
import math

from upper_body_skeleton.retarget_v2 import JOINT_ORDER, retarget_payload_to_rows, write_joint_csv


def make_frame(time_sec, left_elbow, left_wrist, right_elbow, right_wrist):
    landmarks = {
        "pelvis_origin": [0.0, 0.0, 0.0],
        "torso_origin": [0.0, 0.0, 0.30],
        "neck": [0.0, 0.0, 0.52],
        "head": [0.0, 0.0, 0.64],
        "left_shoulder": [0.0, 0.16, 0.48],
        "left_elbow": left_elbow,
        "left_wrist": left_wrist,
        "right_shoulder": [0.0, -0.16, 0.48],
        "right_elbow": right_elbow,
        "right_wrist": right_wrist,
    }
    return {
        "frame_index": int(round(time_sec * 30.0)),
        "time_sec": time_sec,
        "landmarks_3d": landmarks,
        "confidence": {name: 1.0 for name in landmarks},
    }


def make_payload(frames):
    return {
        "schema_name": "upper_body_skeleton_sequence",
        "fps": 30.0,
        "frames": frames,
    }


def test_down_pose_retargets_to_v2_arms_down_calibration_and_within_limits():
    payload = make_payload(
        [
            make_frame(
                0.0,
                [0.0, 0.16, 0.24],
                [0.0, 0.16, 0.05],
                [0.0, -0.16, 0.24],
                [0.0, -0.16, 0.05],
            )
        ]
    )

    rows, report = retarget_payload_to_rows(payload, output_hz=30.0)
    row = rows[0]

    assert report["row_count"] == 1
    assert set(row).issuperset(JOINT_ORDER)
    assert -1.41 <= row["joint_lShoulderRoll"] <= -1.35
    assert -1.41 <= row["joint_rShoulderRoll"] <= -1.35
    assert abs(row["joint_lElbow"]) < 1e-6
    assert abs(row["joint_rElbow"]) < 1e-6
    for name in JOINT_ORDER:
        assert math.isfinite(row[name])


def test_side_raise_maps_to_matching_shoulder_rolls():
    payload = make_payload(
        [
            make_frame(
                0.0,
                [0.0, 0.40, 0.48],
                [0.0, 0.59, 0.48],
                [0.0, -0.40, 0.48],
                [0.0, -0.59, 0.48],
            )
        ]
    )

    rows, _ = retarget_payload_to_rows(payload, output_hz=30.0)
    row = rows[0]

    assert -0.65 <= row["joint_lShoulderRoll"] <= 0.25
    assert -0.65 <= row["joint_rShoulderRoll"] <= 0.25
    assert abs(row["joint_lElbow"]) < 1e-6
    assert abs(row["joint_rElbow"]) < 1e-6


def test_elbow_bend_maps_to_elbow_joint_and_csv_header(tmp_path):
    payload = make_payload(
        [
            make_frame(
                0.0,
                [0.0, 0.16, 0.24],
                [0.0, 0.34, 0.24],
                [0.0, -0.16, 0.24],
                [0.0, -0.34, 0.24],
            )
        ]
    )
    rows, _ = retarget_payload_to_rows(payload, output_hz=30.0)

    assert rows[0]["joint_lElbow"] < -0.8
    assert rows[0]["joint_rElbow"] > 0.8

    out = tmp_path / "joints.csv"
    write_joint_csv(rows, out)
    with out.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
    assert header == ["time_sec"] + JOINT_ORDER


def test_moderate_elbow_bend_uses_stronger_v2_amplitude():
    payload = make_payload(
        [
            make_frame(
                0.0,
                [0.0, 0.16, 0.24],
                [0.0, 0.31, 0.09],
                [0.0, -0.16, 0.24],
                [0.0, -0.31, 0.09],
            )
        ]
    )

    rows, _ = retarget_payload_to_rows(payload, output_hz=30.0)
    row = rows[0]

    assert row["joint_lElbow"] < -0.95
    assert row["joint_rElbow"] > 0.95


def test_forearms_turning_inward_drive_cross_body_shoulder_yaw():
    payload = make_payload(
        [
            make_frame(
                0.0,
                [0.0, 0.34, 0.30],
                [0.0, 0.02, 0.31],
                [0.0, -0.34, 0.30],
                [0.0, -0.02, 0.31],
            )
        ]
    )

    rows, _ = retarget_payload_to_rows(payload, output_hz=30.0)
    row = rows[0]

    assert row["joint_lShoulderYaw"] > 0.85
    assert row["joint_rShoulderYaw"] < -0.85
    assert row["joint_lShoulderPitch"] < -1.05
    assert row["joint_rShoulderPitch"] < -1.05


def test_strong_cross_body_left_elbow_keeps_large_amplitude():
    payload = make_payload(
        [
            make_frame(
                0.0,
                [0.0, 0.34, 0.30],
                [0.0, 0.02, 0.31],
                [0.0, -0.34, 0.30],
                [0.0, -0.02, 0.31],
            )
        ]
    )

    rows, _ = retarget_payload_to_rows(payload, output_hz=30.0)
    row = rows[0]

    assert -1.57 <= row["joint_lElbow"] < -1.35
    assert row["joint_rElbow"] > 0.35


def test_moderate_cross_body_keeps_left_elbow_amplitude():
    payload = make_payload(
        [
            make_frame(
                0.0,
                [0.0, 0.30, 0.30],
                [0.0, 0.18, 0.30],
                [0.0, -0.30, 0.30],
                [0.0, -0.18, 0.30],
            )
        ]
    )

    rows, _ = retarget_payload_to_rows(payload, output_hz=30.0)
    row = rows[0]

    assert row["joint_lShoulderPitch"] < -1.05
    assert row["joint_lShoulderYaw"] > 1.05
    assert row["joint_lElbow"] < -1.35
    assert row["joint_rShoulderPitch"] < -1.05
    assert row["joint_rShoulderYaw"] < -1.05
    assert row["joint_rElbow"] > 0.35
