import json

from upper_body_skeleton.retarget_monitor import analyze_retarget_quality


def test_monitor_detects_yaw_under_response_for_cross_body_pose(tmp_path):
    skeleton = {
        "frames": [
            {
                "time_sec": 0.0,
                "landmarks_3d": {
                    "pelvis_origin": [0.0, 0.0, 0.0],
                    "torso_origin": [0.0, 0.0, 0.3],
                    "neck": [0.0, 0.0, 0.5],
                    "head": [0.0, 0.0, 0.6],
                    "left_shoulder": [0.0, 0.16, 0.48],
                    "left_elbow": [0.0, 0.34, 0.30],
                    "left_wrist": [0.0, 0.02, 0.31],
                    "right_shoulder": [0.0, -0.16, 0.48],
                    "right_elbow": [0.0, -0.34, 0.30],
                    "right_wrist": [0.0, -0.02, 0.31],
                },
            }
        ]
    }
    csv_path = tmp_path / "joints.csv"
    csv_path.write_text(
        "time_sec,joint_lShoulderYaw,joint_rShoulderYaw,joint_lElbow,joint_rElbow,"
        "joint_lShoulderRoll,joint_rShoulderRoll\n"
        "0.000000,0.000000,0.000000,-1.570000,1.570000,-1.000000,-1.000000\n",
        encoding="utf-8",
    )

    report = analyze_retarget_quality(skeleton, csv_path)

    assert report["summary"]["frame_count"] == 1
    assert report["summary"]["max_cross_body_intent"] > 0.5
    assert report["summary"]["max_yaw_under_response"] > 0.5
    assert report["summary"]["max_elbow_overfold"] > 0.2
    assert report["frames"][0]["flags"]
