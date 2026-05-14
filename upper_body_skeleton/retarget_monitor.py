#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

from upper_body_skeleton.retarget_v2 import CROSS_BODY_YAW_LIMIT


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def load_joint_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def side_cross_body_intent(landmarks, side):
    elbow = landmarks[f"{side}_elbow"]
    wrist = landmarks[f"{side}_wrist"]
    fore_y = float(wrist[1]) - float(elbow[1])
    inward_sign = -1.0 if side == "left" else 1.0
    # Normalize roughly by forearm length in the current 2.5D skeleton.
    intent = inward_sign * fore_y / 0.18
    return clamp(intent, 0.0, 1.0)


def analyze_frame(frame, row):
    lm = frame["landmarks_3d"]
    left_intent = side_cross_body_intent(lm, "left")
    right_intent = side_cross_body_intent(lm, "right")
    expected_left_yaw = left_intent * CROSS_BODY_YAW_LIMIT
    expected_right_yaw = -right_intent * CROSS_BODY_YAW_LIMIT
    left_yaw = row.get("joint_lShoulderYaw", 0.0)
    right_yaw = row.get("joint_rShoulderYaw", 0.0)
    left_yaw_gap = max(0.0, expected_left_yaw - left_yaw)
    right_yaw_gap = max(0.0, abs(expected_right_yaw) - abs(right_yaw))
    left_elbow = abs(row.get("joint_lElbow", 0.0))
    right_elbow = abs(row.get("joint_rElbow", 0.0))
    elbow_overfold = max(0.0, max(left_elbow, right_elbow) - 1.25)
    flags = []
    if max(left_yaw_gap, right_yaw_gap) > 0.25:
        flags.append("yaw_under_response")
    if elbow_overfold > 0.1:
        flags.append("elbow_overfold")
    return {
        "time_sec": float(frame.get("time_sec", row.get("time_sec", 0.0))),
        "left_cross_body_intent": left_intent,
        "right_cross_body_intent": right_intent,
        "expected_left_yaw": expected_left_yaw,
        "expected_right_yaw": expected_right_yaw,
        "actual_left_yaw": left_yaw,
        "actual_right_yaw": right_yaw,
        "yaw_under_response": max(left_yaw_gap, right_yaw_gap),
        "left_elbow_abs": left_elbow,
        "right_elbow_abs": right_elbow,
        "elbow_overfold": elbow_overfold,
        "flags": flags,
    }


def analyze_retarget_quality(skeleton_payload, joint_csv_path):
    frames = skeleton_payload.get("frames", [])
    rows = load_joint_rows(joint_csv_path)
    count = min(len(frames), len(rows))
    frame_reports = [analyze_frame(frames[i], rows[i]) for i in range(count)]
    summary = {
        "frame_count": count,
        "max_cross_body_intent": max(
            [max(f["left_cross_body_intent"], f["right_cross_body_intent"]) for f in frame_reports] or [0.0]
        ),
        "max_yaw_under_response": max([f["yaw_under_response"] for f in frame_reports] or [0.0]),
        "max_elbow_overfold": max([f["elbow_overfold"] for f in frame_reports] or [0.0]),
        "flagged_frame_count": sum(1 for f in frame_reports if f["flags"]),
    }
    return {"summary": summary, "frames": frame_reports}


def write_frame_csv(report, output_csv):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "time_sec",
        "left_cross_body_intent",
        "right_cross_body_intent",
        "expected_left_yaw",
        "expected_right_yaw",
        "actual_left_yaw",
        "actual_right_yaw",
        "yaw_under_response",
        "left_elbow_abs",
        "right_elbow_abs",
        "elbow_overfold",
        "flags",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for frame in report["frames"]:
            row = dict(frame)
            row["flags"] = "|".join(row["flags"])
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Monitor V2 retarget quality against source upper-body skeleton")
    parser.add_argument("skeleton_json")
    parser.add_argument("joint_csv")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    with open(args.skeleton_json, "r", encoding="utf-8") as f:
        skeleton = json.load(f)
    report = analyze_retarget_quality(skeleton, args.joint_csv)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_frame_csv(report, args.output_csv)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
