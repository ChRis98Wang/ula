from upper_body_skeleton.v2_axis_calibration import analyze_v2_axes


def test_v2_axis_analysis_reports_expected_shoulder_roll_and_elbow_directions():
    report = analyze_v2_axes()
    joints = report["joints"]

    assert joints["joint_lShoulderRoll"]["wrist_delta"][1] < 0.0
    assert joints["joint_rShoulderRoll"]["wrist_delta"][1] > 0.0
    assert joints["joint_lElbow"]["wrist_delta"][0] < 0.0
    assert joints["joint_rElbow"]["wrist_delta"][0] > 0.0
    assert report["calibration"]["joint_lShoulderRoll"]["arms_down_offset_rad"] == -1.4
    assert report["calibration"]["joint_rShoulderRoll"]["arms_down_offset_rad"] == -1.4
