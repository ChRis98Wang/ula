#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path


JOINT_ORDER = [
    "joint_pelvisYaw",
    "joint_pelvisPitch",
    "joint_pelvisRoll",
    "joint_lShoulderPitch",
    "joint_lShoulderRoll",
    "joint_lShoulderYaw",
    "joint_lElbow",
    "joint_lWristRoll",
    "joint_lWristPitch",
    "joint_rShoulderPitch",
    "joint_rShoulderRoll",
    "joint_rShoulderYaw",
    "joint_rElbow",
    "joint_rWristRoll",
    "joint_rWristPitch",
]


JOINT_LIMITS = {
    "joint_pelvisYaw": (-1.57, 1.57),
    "joint_pelvisPitch": (0.0, 1.046),
    "joint_pelvisRoll": (-0.35, 0.35),
    "joint_lShoulderPitch": (-1.4, 4.2),
    "joint_lShoulderRoll": (-1.41, 1.57),
    "joint_lShoulderYaw": (-2.79, 2.79),
    "joint_lElbow": (-1.57, 1.57),
    "joint_lWristRoll": (-2.79, 2.79),
    "joint_lWristPitch": (-1.57, 1.57),
    "joint_rShoulderPitch": (-1.4, 4.2),
    "joint_rShoulderRoll": (-1.41, 1.57),
    "joint_rShoulderYaw": (-2.79, 2.79),
    "joint_rElbow": (-1.57, 1.57),
    "joint_rWristRoll": (-2.79, 2.79),
    "joint_rWristPitch": (-1.57, 1.57),
}


NEUTRAL = {joint: 0.0 for joint in JOINT_ORDER}
V2_ARMS_DOWN_SHOULDER_ROLL = -1.4
CROSS_BODY_YAW_GAIN = 4.5
CROSS_BODY_YAW_LIMIT = 1.55
ELBOW_FLEXION_GAIN = 1.35
CROSS_BODY_PITCH_GAIN = -1.25
CROSS_BODY_INTENT_SCALE_M = 0.12


def vec_sub(a, b):
    return [float(a[0]) - float(b[0]), float(a[1]) - float(b[1]), float(a[2]) - float(b[2])]


def vec_norm(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def angle_between(a, b):
    an = vec_norm(a)
    bn = vec_norm(b)
    if an < 1e-9 or bn < 1e-9:
        return 0.0
    dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (an * bn)
    return math.acos(max(-1.0, min(1.0, dot)))


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def joint_clamp(name, value):
    lower, upper = JOINT_LIMITS[name]
    return clamp(float(value), lower, upper)


def arm_angles(landmarks, side):
    shoulder = landmarks[f"{side}_shoulder"]
    elbow = landmarks[f"{side}_elbow"]
    wrist = landmarks[f"{side}_wrist"]
    upper = vec_sub(elbow, shoulder)
    fore = vec_sub(wrist, elbow)

    down = [0.0, 0.0, -1.0]
    outward_sign = 1.0 if side == "left" else -1.0
    human_abduction = math.atan2(outward_sign * upper[1], -upper[2])
    roll = V2_ARMS_DOWN_SHOULDER_ROLL + human_abduction
    inward_sign = -1.0 if side == "left" else 1.0
    inward_wrist_motion = inward_sign * fore[1]
    cross_body_strength = cross_body_strength_from_inward_motion(inward_wrist_motion)
    pitch = math.atan2(upper[0], -upper[2]) + CROSS_BODY_PITCH_GAIN * cross_body_strength
    elbow_sign = -1.0 if side == "left" else 1.0
    elbow_flexion = elbow_sign * angle_between(upper, fore) * ELBOW_FLEXION_GAIN
    yaw_sign = 1.0 if side == "left" else -1.0
    yaw = yaw_sign * CROSS_BODY_YAW_LIMIT * cross_body_strength
    return {
        "pitch": pitch,
        "roll": roll,
        "yaw": yaw,
        "elbow": elbow_flexion,
        "wrist_roll": 0.0,
        "wrist_pitch": 0.0,
    }


def cross_body_strength_from_inward_motion(inward_wrist_motion):
    raw = max(0.0, float(inward_wrist_motion)) / CROSS_BODY_INTENT_SCALE_M
    return clamp(raw * raw, 0.0, 1.0)


def retarget_frame(frame):
    lm = frame["landmarks_3d"]
    left = arm_angles(lm, "left")
    right = arm_angles(lm, "right")
    row = dict(NEUTRAL)
    row.update(
        {
            "time_sec": float(frame["time_sec"]),
            "joint_lShoulderPitch": left["pitch"],
            "joint_lShoulderRoll": left["roll"],
            "joint_lShoulderYaw": left["yaw"],
            "joint_lElbow": left["elbow"],
            "joint_lWristRoll": left["wrist_roll"],
            "joint_lWristPitch": left["wrist_pitch"],
            "joint_rShoulderPitch": right["pitch"],
            "joint_rShoulderRoll": right["roll"],
            "joint_rShoulderYaw": right["yaw"],
            "joint_rElbow": right["elbow"],
            "joint_rWristRoll": right["wrist_roll"],
            "joint_rWristPitch": right["wrist_pitch"],
        }
    )
    for joint in JOINT_ORDER:
        row[joint] = joint_clamp(joint, row[joint])
    return row


def resample_frames(frames, output_hz):
    if not frames:
        return []
    if len(frames) == 1:
        return [frames[0]]
    source_hz = 1.0 / max(1e-9, frames[1]["time_sec"] - frames[0]["time_sec"])
    if abs(source_hz - output_hz) < 1e-3:
        return frames
    # MVP policy: nearest-neighbor resampling keeps landmarks valid without
    # introducing invented bends before a full interpolation layer exists.
    start = frames[0]["time_sec"]
    end = frames[-1]["time_sec"]
    count = int(round((end - start) * output_hz)) + 1
    out = []
    cursor = 0
    for i in range(count):
        t = start + i / output_hz
        while cursor + 1 < len(frames) and abs(frames[cursor + 1]["time_sec"] - t) < abs(frames[cursor]["time_sec"] - t):
            cursor += 1
        out.append(frames[cursor])
    return out


def retarget_payload_to_rows(payload, output_hz=30.0):
    frames = resample_frames(payload.get("frames", []), output_hz)
    rows = [retarget_frame(frame) for frame in frames]
    report = {
        "source_frame_count": len(payload.get("frames", [])),
        "row_count": len(rows),
        "output_hz": float(output_hz),
        "joint_order": JOINT_ORDER,
        "joint_limits": JOINT_LIMITS,
        "notes": [
            "2.5D skeleton source: shoulder pitch/yaw and wrist rotations are conservative placeholders.",
            "Lower body and balance are intentionally excluded.",
        ],
    }
    return rows, report


def write_joint_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["time_sec"] + JOINT_ORDER)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: f"{row[name]:.6f}" for name in ["time_sec"] + JOINT_ORDER})


def main():
    parser = argparse.ArgumentParser(description="Retarget smoothed upper-body skeleton JSON to V2 joint CSV")
    parser.add_argument("skeleton_json")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output-hz", type=float, default=30.0)
    args = parser.parse_args()

    with open(args.skeleton_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rows, report = retarget_payload_to_rows(payload, output_hz=args.output_hz)
    write_joint_csv(rows, args.output_csv)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "csv": args.output_csv, "report": args.report}, indent=2))


if __name__ == "__main__":
    main()
