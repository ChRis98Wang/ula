#!/usr/bin/env python3
import argparse
import copy
import json
import statistics
from pathlib import Path

from upper_body_skeleton.extract import UPPER_BODY_LANDMARKS, UpperBodyFrameBuilder, vec_norm, vec_sub


ARM_SEGMENTS = [
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
]


def _odd_window(window_frames, frame_count):
    window = max(1, int(window_frames))
    window = min(window, frame_count)
    if window % 2 == 0:
        window -= 1
    return max(1, window)


def _smooth_series(values, window_frames):
    if not values:
        return []
    window = _odd_window(window_frames, len(values))
    if window <= 1:
        return [float(v) for v in values]
    radius = window // 2
    smoothed = []
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        smoothed.append(sum(float(v) for v in values[start:end]) / (end - start))
    return smoothed


def _segment_lengths(frames, parent, child):
    lengths = []
    for frame in frames:
        landmarks = frame.get("landmarks_3d", {})
        if parent not in landmarks or child not in landmarks:
            continue
        length = vec_norm(vec_sub(landmarks[child], landmarks[parent]))
        if length > 1e-9:
            lengths.append(length)
    return lengths


def _unit_from_parent_to_child(landmarks, parent, child):
    vector = vec_sub(landmarks[child], landmarks[parent])
    length = vec_norm(vector)
    if length < 1e-9:
        return [0.0, 0.0, -1.0]
    return [vector[0] / length, vector[1] / length, vector[2] / length]


def _stabilize_arm_lengths(frames):
    target_lengths = {}
    for parent, child in ARM_SEGMENTS:
        lengths = _segment_lengths(frames, parent, child)
        if lengths:
            target_lengths[(parent, child)] = statistics.median(lengths)

    for frame in frames:
        landmarks = frame["landmarks_3d"]
        for parent, child in ARM_SEGMENTS:
            target = target_lengths.get((parent, child))
            if target is None or parent not in landmarks or child not in landmarks:
                continue
            unit = _unit_from_parent_to_child(landmarks, parent, child)
            landmarks[child] = [
                float(landmarks[parent][axis]) + unit[axis] * target for axis in range(3)
            ]


def smooth_payload(payload, window_frames=11):
    smoothed = copy.deepcopy(payload)
    frames = smoothed.get("frames", [])
    if not frames:
        smoothed.setdefault("quality", {})["smoothing"] = {
            "method": "moving_average_with_arm_length_stabilization",
            "window_frames": int(window_frames),
        }
        return smoothed

    landmark_order = smoothed.get("landmark_order") or UPPER_BODY_LANDMARKS
    for landmark in landmark_order:
        for axis in range(3):
            values = [
                frame.get("landmarks_3d", {}).get(landmark, [0.0, 0.0, 0.0])[axis]
                for frame in frames
            ]
            axis_values = _smooth_series(values, window_frames)
            for frame, value in zip(frames, axis_values):
                if landmark in frame.get("landmarks_3d", {}):
                    frame["landmarks_3d"][landmark][axis] = float(value)

    for frame in frames:
        frame.setdefault("root", {})["position"] = [0.0, 0.0, 0.0]
        if "pelvis_origin" in frame.get("landmarks_3d", {}):
            frame["landmarks_3d"]["pelvis_origin"] = [0.0, 0.0, 0.0]

    _stabilize_arm_lengths(frames)

    feature_builder = UpperBodyFrameBuilder()
    for frame in frames:
        frame["features"] = feature_builder._features(frame["landmarks_3d"])

    smoothed.setdefault("quality", {})["smoothing"] = {
        "method": "moving_average_with_arm_length_stabilization",
        "window_frames": _odd_window(window_frames, len(frames)),
        "arm_lengths_stabilized": True,
    }
    return smoothed


def main():
    parser = argparse.ArgumentParser(description="Smooth upper-body skeleton JSON and stabilize arm lengths")
    parser.add_argument("skeleton_json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--window-frames", type=int, default=11)
    args = parser.parse_args()

    payload = json.loads(Path(args.skeleton_json).read_text(encoding="utf-8"))
    smoothed = smooth_payload(payload, window_frames=args.window_frames)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(smoothed, indent=2), encoding="utf-8")
    print(json.dumps(smoothed.get("quality", {}).get("smoothing", {}), indent=2))


if __name__ == "__main__":
    main()
