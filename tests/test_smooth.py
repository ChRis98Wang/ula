import math

from upper_body_skeleton.smooth import smooth_payload


def make_payload(frames):
    return {
        "schema_name": "upper_body_skeleton_sequence",
        "schema_version": "0.2",
        "landmark_order": [
            "pelvis_origin",
            "torso_origin",
            "neck",
            "head",
            "left_shoulder",
            "left_elbow",
            "left_wrist",
            "right_shoulder",
            "right_elbow",
            "right_wrist",
        ],
        "frames": frames,
        "quality": {"frame_count": len(frames), "lower_body_excluded": True},
    }


def make_frame(index, left_elbow_y):
    landmarks = {
        "pelvis_origin": [0.0, 0.0, 0.0],
        "torso_origin": [0.0, 0.0, 0.30],
        "neck": [0.0, 0.0, 0.52],
        "head": [0.0, 0.0, 0.64],
        "left_shoulder": [0.0, 0.16, 0.48],
        "left_elbow": [0.0, left_elbow_y, 0.24],
        "left_wrist": [0.0, left_elbow_y, 0.04],
        "right_shoulder": [0.0, -0.16, 0.48],
        "right_elbow": [0.0, -0.16, 0.24],
        "right_wrist": [0.0, -0.16, 0.04],
    }
    return {
        "frame_index": index,
        "time_sec": index / 30.0,
        "root": {"position": [0.0, 0.0, 0.0]},
        "landmarks_3d": landmarks,
        "confidence": {name: 1.0 for name in landmarks},
        "features": {},
    }


def segment_length(frame, start, end):
    a = frame["landmarks_3d"][start]
    b = frame["landmarks_3d"][end]
    return math.sqrt(sum((float(b[i]) - float(a[i])) ** 2 for i in range(3)))


def test_smooth_payload_preserves_frame_count_and_urdf_zero_origin():
    payload = make_payload([make_frame(i, 0.16) for i in range(5)])

    smoothed = smooth_payload(payload, window_frames=3)

    assert len(smoothed["frames"]) == 5
    assert smoothed["quality"]["smoothing"]["window_frames"] == 3
    for frame in smoothed["frames"]:
        assert frame["landmarks_3d"]["pelvis_origin"] == [0.0, 0.0, 0.0]
        assert frame["root"]["position"] == [0.0, 0.0, 0.0]


def test_smooth_payload_stabilizes_arm_lengths_after_temporal_smoothing():
    payload = make_payload(
        [
            make_frame(0, 0.16),
            make_frame(1, 0.26),
            make_frame(2, 0.06),
            make_frame(3, 0.20),
            make_frame(4, 0.12),
        ]
    )

    smoothed = smooth_payload(payload, window_frames=3)

    left_upper_lengths = [
        segment_length(frame, "left_shoulder", "left_elbow") for frame in smoothed["frames"]
    ]
    left_fore_lengths = [segment_length(frame, "left_elbow", "left_wrist") for frame in smoothed["frames"]]
    assert max(left_upper_lengths) - min(left_upper_lengths) < 1e-6
    assert max(left_fore_lengths) - min(left_fore_lengths) < 1e-6
    assert "left_upper_arm_dir" in smoothed["frames"][0]["features"]
