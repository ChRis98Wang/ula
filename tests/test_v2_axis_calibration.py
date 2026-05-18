from pathlib import Path

from upper_body_skeleton.v2_axis_calibration import NEW_URDF_SOURCE, REPO_ROOT, analyze_v2_axes


def test_default_0514_urdf_is_resolved_from_repo_root():
    assert REPO_ROOT == Path(__file__).resolve().parents[1]
    assert NEW_URDF_SOURCE == REPO_ROOT / "urdf_V2_20260514" / "urdf" / "xacro" / "robot_modify.urdf"


def test_v2_axis_analysis_reports_expected_shoulder_roll_and_elbow_directions():
    report = analyze_v2_axes()
    joints = report["joints"]

    assert joints["joint_lShoulderRoll"]["wrist_delta"][1] < 0.0
    assert joints["joint_rShoulderRoll"]["wrist_delta"][1] > 0.0
    assert joints["joint_lElbow"]["wrist_delta"][0] < 0.0
    assert joints["joint_rElbow"]["wrist_delta"][0] < 0.0
    assert report["calibration"]["joint_lShoulderRoll"]["arms_down_offset_rad"] == -1.4
    assert report["calibration"]["joint_rShoulderRoll"]["arms_down_offset_rad"] == -1.4
    assert report["calibration"]["joint_lElbow"]["human_flexion_sign"] == -1.0
    assert report["calibration"]["joint_rElbow"]["human_flexion_sign"] == -1.0
