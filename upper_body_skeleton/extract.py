#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np


SMPLH_BODY_INDEX = {
    "left_hip": 0,
    "right_hip": 1,
    "spine1": 2,
    "left_knee": 3,
    "right_knee": 4,
    "spine2": 5,
    "left_ankle": 6,
    "right_ankle": 7,
    "spine3": 8,
    "left_foot": 9,
    "right_foot": 10,
    "neck": 11,
    "left_collar": 12,
    "right_collar": 13,
    "head": 14,
    "left_shoulder": 15,
    "right_shoulder": 16,
    "left_elbow": 17,
    "right_elbow": 18,
    "left_wrist": 19,
    "right_wrist": 20,
}


UPPER_BODY_LANDMARKS = [
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
]


KEYPOINT_INDEX = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
}


def build_zero_pose():
    return [[0.0, 0.0, 0.0] for _ in range(21)]


def axis_angle_to_matrix(axis_angle):
    x, y, z = [float(v) for v in axis_angle]
    theta = math.sqrt(x * x + y * y + z * z)
    if theta < 1e-12:
        return [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    x /= theta
    y /= theta
    z /= theta
    c = math.cos(theta)
    s = math.sin(theta)
    one_c = 1.0 - c
    return [
        [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
        [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
        [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
    ]


def matmul(a, b):
    return [
        [
            a[row][0] * b[0][col] + a[row][1] * b[1][col] + a[row][2] * b[2][col]
            for col in range(3)
        ]
        for row in range(3)
    ]


def matvec(m, v):
    return [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]


def vec_add(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def vec_sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def vec_norm(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def vec_scale(v, scale):
    return [v[0] * scale, v[1] * scale, v[2] * scale]


def vec_normalize(v, fallback):
    n = vec_norm(v)
    if n < 1e-9:
        return list(fallback)
    return [v[0] / n, v[1] / n, v[2] / n]


def keypoint_confidence(keypoints, name):
    return float(keypoints[KEYPOINT_INDEX[name]][2])


def build_urdf_zero_frame_from_keypoints(frame_index, time_sec, keypoints, image_size, valid=True):
    """Build a video-aligned upper-body skeleton from 2D COCO/OpenPose keypoints.

    This is intentionally 2.5D: x is set to 0 because monocular 2D keypoints do
    not contain reliable depth. y/z preserve the visible upper-body geometry in
    a URDF-zero local frame, centered at the hip midpoint.
    """
    image_width, image_height = [float(v) for v in image_size]
    if image_width <= 0.0 or image_height <= 0.0:
        raise ValueError("image_size must contain positive width and height")

    left_hip = keypoints[KEYPOINT_INDEX["left_hip"]]
    right_hip = keypoints[KEYPOINT_INDEX["right_hip"]]
    left_shoulder = keypoints[KEYPOINT_INDEX["left_shoulder"]]
    right_shoulder = keypoints[KEYPOINT_INDEX["right_shoulder"]]
    hip_mid = [(left_hip[0] + right_hip[0]) / 2.0, (left_hip[1] + right_hip[1]) / 2.0]
    shoulder_mid = [
        (left_shoulder[0] + right_shoulder[0]) / 2.0,
        (left_shoulder[1] + right_shoulder[1]) / 2.0,
    ]
    shoulder_width_px = abs(float(left_shoulder[0]) - float(right_shoulder[0]))
    torso_height_px = vec_norm([shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1], 0.0])
    scale_px = max(shoulder_width_px / 0.29, torso_height_px / 0.38, image_width * 0.02, 1.0)

    def point_from_keypoint(name):
        x_px, y_px, conf = keypoints[KEYPOINT_INDEX[name]]
        return [
            0.0,
            (float(x_px) - hip_mid[0]) / scale_px,
            (hip_mid[1] - float(y_px)) / scale_px,
        ]

    positions = {
        "pelvis_origin": [0.0, 0.0, 0.0],
        "left_shoulder": point_from_keypoint("left_shoulder"),
        "right_shoulder": point_from_keypoint("right_shoulder"),
        "left_elbow": point_from_keypoint("left_elbow"),
        "right_elbow": point_from_keypoint("right_elbow"),
        "left_wrist": point_from_keypoint("left_wrist"),
        "right_wrist": point_from_keypoint("right_wrist"),
    }
    positions["torso_origin"] = [
        0.0,
        (positions["left_shoulder"][1] + positions["right_shoulder"][1]) / 2.0,
        (positions["left_shoulder"][2] + positions["right_shoulder"][2]) / 2.0 - 0.08,
    ]
    positions["neck"] = [
        0.0,
        (positions["left_shoulder"][1] + positions["right_shoulder"][1]) / 2.0,
        (positions["left_shoulder"][2] + positions["right_shoulder"][2]) / 2.0 + 0.03,
    ]
    positions["head"] = [0.0, positions["neck"][1], positions["neck"][2] + 0.12]

    confidence = {
        "pelvis_origin": min(keypoint_confidence(keypoints, "left_hip"), keypoint_confidence(keypoints, "right_hip")),
        "torso_origin": min(
            keypoint_confidence(keypoints, "left_shoulder"),
            keypoint_confidence(keypoints, "right_shoulder"),
        ),
        "neck": min(
            keypoint_confidence(keypoints, "left_shoulder"),
            keypoint_confidence(keypoints, "right_shoulder"),
        ),
        "head": min(
            keypoint_confidence(keypoints, "left_shoulder"),
            keypoint_confidence(keypoints, "right_shoulder"),
        ),
        "left_shoulder": keypoint_confidence(keypoints, "left_shoulder"),
        "left_elbow": keypoint_confidence(keypoints, "left_elbow"),
        "left_wrist": keypoint_confidence(keypoints, "left_wrist"),
        "right_shoulder": keypoint_confidence(keypoints, "right_shoulder"),
        "right_elbow": keypoint_confidence(keypoints, "right_elbow"),
        "right_wrist": keypoint_confidence(keypoints, "right_wrist"),
    }
    if not valid:
        confidence = {name: 0.0 for name in confidence}

    return {
        "frame_index": int(frame_index),
        "time_sec": float(time_sec),
        "root": {
            "position": [0.0, 0.0, 0.0],
            "coordinate_type": "urdf_zero_upper_body_keypoint_local",
            "source_image_size": [image_width, image_height],
        },
        "landmarks_3d": {k: [float(x) for x in positions[k]] for k in UPPER_BODY_LANDMARKS},
        "confidence": confidence,
        "features": UpperBodyFrameBuilder()._features(positions),
    }


class UpperBodyFrameBuilder:
    """Approximate SMPL-H upper-body FK in a V2 URDF-zero local frame.

    This does not require SMPL-H model files. It uses the SMPL-H axis-angle
    pose hierarchy to rotate a conservative upper-body stick skeleton whose
    neutral dimensions are aligned to the V2 URDF upper-body zero pose.
    """

    def __init__(self):
        self.parents = {
            "pelvis_origin": None,
            "torso_origin": "pelvis_origin",
            "neck": "torso_origin",
            "head": "neck",
            "left_shoulder": "neck",
            "left_elbow": "left_shoulder",
            "left_wrist": "left_elbow",
            "right_shoulder": "neck",
            "right_elbow": "right_shoulder",
            "right_wrist": "right_elbow",
        }
        self.pose_sources = {
            "pelvis_origin": None,
            "torso_origin": "spine3",
            "neck": "neck",
            "head": "head",
            "left_shoulder": "left_shoulder",
            "left_elbow": "left_elbow",
            "left_wrist": "left_wrist",
            "right_shoulder": "right_shoulder",
            "right_elbow": "right_elbow",
            "right_wrist": "right_wrist",
        }
        self.offsets = {
            "pelvis_origin": [0.0, 0.0, 0.0],
            "torso_origin": [0.0, 0.0, 0.18],
            "neck": [0.0, 0.0, 0.20],
            "head": [0.0, 0.0, 0.10],
            "left_shoulder": [0.0, 0.145, 0.02],
            "left_elbow": [0.0, 0.0, -0.24],
            "left_wrist": [0.0, 0.0, -0.21],
            "right_shoulder": [0.0, -0.145, 0.02],
            "right_elbow": [0.0, 0.0, -0.24],
            "right_wrist": [0.0, 0.0, -0.21],
        }

    def segment_length(self, name):
        mapping = {
            "left_upper_arm": "left_elbow",
            "left_forearm": "left_wrist",
            "right_upper_arm": "right_elbow",
            "right_forearm": "right_wrist",
        }
        return vec_norm(self.offsets[mapping[name]])

    def build_frame(self, frame_index, time_sec, body_pose, global_orient, translation, valid=True):
        rotations = {}
        positions = {}
        confidence = {}
        root_rotation = axis_angle_to_matrix([0.0, 0.0, 0.0])
        for name in UPPER_BODY_LANDMARKS:
            parent = self.parents[name]
            source = self.pose_sources[name]
            local_rotation = (
                axis_angle_to_matrix(body_pose[SMPLH_BODY_INDEX[source]])
                if source is not None
                else axis_angle_to_matrix([0.0, 0.0, 0.0])
            )
            if parent is None:
                rotations[name] = root_rotation
                positions[name] = [0.0, 0.0, 0.0]
            else:
                rotations[name] = matmul(rotations[parent], local_rotation)
                positions[name] = vec_add(positions[parent], matvec(rotations[parent], self.offsets[name]))
            confidence[name] = 1.0 if valid else 0.0

        features = self._features(positions)
        return {
            "frame_index": int(frame_index),
            "time_sec": float(time_sec),
            "root": {
                "position": [0.0, 0.0, 0.0],
                "coordinate_type": "urdf_zero_upper_body_local",
                "ignored_source_global_orient": [float(v) for v in global_orient],
                "ignored_source_translation": [float(v) for v in translation],
            },
            "landmarks_3d": {k: [float(x) for x in positions[k]] for k in UPPER_BODY_LANDMARKS},
            "confidence": confidence,
            "features": features,
        }

    def _features(self, positions):
        left_upper = vec_sub(positions["left_elbow"], positions["left_shoulder"])
        left_fore = vec_sub(positions["left_wrist"], positions["left_elbow"])
        right_upper = vec_sub(positions["right_elbow"], positions["right_shoulder"])
        right_fore = vec_sub(positions["right_wrist"], positions["right_elbow"])
        return {
            "left_upper_arm_dir": vec_normalize(left_upper, [0.0, 0.0, -1.0]),
            "left_forearm_dir": vec_normalize(left_fore, [0.0, 0.0, -1.0]),
            "right_upper_arm_dir": vec_normalize(right_upper, [0.0, 0.0, -1.0]),
            "right_forearm_dir": vec_normalize(right_fore, [0.0, 0.0, -1.0]),
            "left_elbow_flexion_rad": elbow_flexion(left_upper, left_fore),
            "right_elbow_flexion_rad": elbow_flexion(right_upper, right_fore),
        }


def elbow_flexion(upper_arm, forearm):
    a = vec_normalize(upper_arm, [0.0, 0.0, -1.0])
    b = vec_normalize(forearm, [0.0, 0.0, -1.0])
    dot = max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))
    return math.acos(dot)


def frame_indices(total, start_frame=0, stride=1, max_frames=None):
    indices = range(int(start_frame), int(total), int(stride))
    if max_frames is not None:
        indices = list(indices)[:max_frames]
    return indices


def load_npz_frames(npz_path, max_frames=None, stride=1, start_frame=0):
    data = np.load(npz_path, allow_pickle=True)
    body_pose = data["smplh:body_pose"]
    global_orient = data["smplh:global_orient"]
    translation = data["smplh:translation"]
    valid = data["smplh:is_valid"]
    total = int(body_pose.shape[0])
    indices = frame_indices(total, start_frame=start_frame, stride=stride, max_frames=max_frames)
    for frame_index in indices:
        yield (
            frame_index,
            body_pose[frame_index].tolist(),
            global_orient[frame_index].tolist(),
            translation[frame_index].tolist(),
            bool(valid[frame_index]),
        )


def load_npz_keypoint_frames(npz_path, max_frames=None, stride=1, start_frame=0):
    data = np.load(npz_path, allow_pickle=True)
    keypoints = data["boxes_and_keypoints:keypoints"]
    valid_box = data["boxes_and_keypoints:is_valid_box"]
    total = int(keypoints.shape[0])
    indices = frame_indices(total, start_frame=start_frame, stride=stride, max_frames=max_frames)
    for frame_index in indices:
        yield frame_index, keypoints[frame_index].tolist(), bool(valid_box[frame_index])


def convert_npz(npz_path, output_path, fps=30.0, max_frames=None, stride=1, start_frame=0):
    npz_path = Path(npz_path)
    builder = UpperBodyFrameBuilder()
    frames = []
    valid_count = 0
    for frame_index, body_pose, global_orient, translation, valid in load_npz_frames(
        npz_path, max_frames=max_frames, stride=stride, start_frame=start_frame
    ):
        if valid:
            valid_count += 1
        frames.append(
            builder.build_frame(
                frame_index=frame_index,
                time_sec=frame_index / fps,
                body_pose=body_pose,
                global_orient=global_orient,
                translation=translation,
                valid=valid,
            )
        )
    payload = {
        "schema_name": "upper_body_skeleton_sequence",
        "schema_version": "0.1",
        "source_npz": str(npz_path),
        "source_type": "seamless_interaction_smplh_axis_angle",
        "pose_tool": "seamless_smplh_urdf_zero_fk_adapter",
        "coordinate_type": "urdf_zero_upper_body_local",
        "fps": float(fps),
        "zero_point": {
            "definition": "pelvis_origin at V2 URDF upper-body zero; source translation is retained but ignored",
            "robot_reference": "/Users/demo/Desktop/systemidentification/urdf_V2_20260424/urdf/robot.urdf",
        },
        "landmark_order": UPPER_BODY_LANDMARKS,
        "quality": {
            "frame_count": len(frames),
            "valid_smplh_frame_count": valid_count,
            "valid_smplh_frame_ratio": valid_count / len(frames) if frames else 0.0,
            "lower_body_excluded": True,
        },
        "frames": frames,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def convert_npz_keypoints(npz_path, output_path, fps=30.0, max_frames=None, stride=1, image_size=None, start_frame=0):
    npz_path = Path(npz_path)
    frames = []
    valid_count = 0
    image_size = image_size or [1080.0, 1920.0]
    for frame_index, keypoints, valid in load_npz_keypoint_frames(
        npz_path, max_frames=max_frames, stride=stride, start_frame=start_frame
    ):
        if valid:
            valid_count += 1
        frames.append(
            build_urdf_zero_frame_from_keypoints(
                frame_index=frame_index,
                time_sec=frame_index / fps,
                keypoints=keypoints,
                image_size=image_size,
                valid=valid,
            )
        )
    payload = {
        "schema_name": "upper_body_skeleton_sequence",
        "schema_version": "0.2",
        "source_npz": str(npz_path),
        "source_type": "seamless_interaction_boxes_and_keypoints",
        "pose_tool": "seamless_2d_keypoints_urdf_zero_adapter",
        "coordinate_type": "urdf_zero_upper_body_keypoint_local",
        "fps": float(fps),
        "zero_point": {
            "definition": "pelvis_origin at hip midpoint projected into V2 URDF upper-body zero; x-depth is 0 because source is 2D",
            "robot_reference": "/Users/demo/Desktop/systemidentification/urdf_V2_20260424/urdf/robot.urdf",
        },
        "landmark_order": UPPER_BODY_LANDMARKS,
        "quality": {
            "frame_count": len(frames),
            "valid_keypoint_frame_count": valid_count,
            "valid_keypoint_frame_ratio": valid_count / len(frames) if frames else 0.0,
            "lower_body_excluded": True,
            "depth_is_estimated": False,
        },
        "frames": frames,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Extract URDF-zero upper-body skeleton JSON from Seamless NPZ")
    parser.add_argument("npz_path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--source",
        choices=["smplh-approx", "keypoints"],
        default="smplh-approx",
        help="Use approximate SMPL-H FK or video-aligned 2D keypoints.",
    )
    parser.add_argument("--image-width", type=float, default=1080.0)
    parser.add_argument("--image-height", type=float, default=1920.0)
    args = parser.parse_args()
    if args.source == "keypoints":
        payload = convert_npz_keypoints(
            args.npz_path,
            args.output,
            fps=args.fps,
            max_frames=args.max_frames,
            stride=args.stride,
            image_size=[args.image_width, args.image_height],
            start_frame=args.start_frame,
        )
    else:
        payload = convert_npz(
            args.npz_path,
            args.output,
            fps=args.fps,
            max_frames=args.max_frames,
            stride=args.stride,
            start_frame=args.start_frame,
        )
    print(json.dumps(payload["quality"], indent=2))


if __name__ == "__main__":
    main()
