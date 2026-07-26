#!/usr/bin/env python3
"""Compose InterAct source-stick and real-URDF MuJoCo retarget evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.gmr_v2.interact_bvh_adapter import (
    INTERACT_NATIVE_AXIS_POLICY,
    load_interact_bvh_native_v2,
)


SOURCE_PANE_WIDTH = 640
ROBOT_FRONT_CAMERA_POSITION_AXIS = "+X"
ROBOT_FRONT_CAMERA_VIEW_DIRECTION = "-X"
ROBOT_FRONT_CAMERA_SCREEN_RIGHT_AXIS = np.array([0.0, 1.0, 0.0])
COLORS = {
    "torso": (185, 190, 196),
    "left": (71, 142, 255),
    "right": (245, 112, 69),
    "joint": (235, 238, 240),
    "text": (235, 238, 240),
    "muted": (150, 158, 168),
    "head_axis": (78, 214, 181),
    "head_forward": (247, 202, 82),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bvh", type=Path, required=True)
    parser.add_argument("--source-start-frame", type=int, required=True)
    parser.add_argument("--source-end-frame", type=int, required=True)
    parser.add_argument("--robot-mp4", type=Path, required=True)
    parser.add_argument("--output-mp4", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--quality-json", type=Path, required=True)
    parser.add_argument("--actor-id", required=True)
    parser.add_argument("--partner-actor-id", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _horizontal_values(positions: np.ndarray, view: str) -> np.ndarray:
    if view == "front":
        # MuJoCo azimuth=180 puts the camera on +X looking toward -X.
        # Its screen-right vector is +Y, verified from MjvScene camera axes.
        return positions @ ROBOT_FRONT_CAMERA_SCREEN_RIGHT_AXIS
    if view == "side":
        return positions[..., 0]
    raise ValueError(f"Unknown InterAct review projection: {view}")


def _projection_bounds(
    positions: np.ndarray, view: str
) -> tuple[float, float, float, float]:
    x = _horizontal_values(positions, view)
    z = positions[..., 2]
    x_min, x_max = float(np.min(x)), float(np.max(x))
    z_min, z_max = float(np.min(z)), float(np.max(z))
    if x_max - x_min < 1e-5 or z_max - z_min < 1e-5:
        raise ValueError("Source skeleton projection is degenerate")
    return x_min, x_max, z_min, z_max


def render_source_frame(
    positions: np.ndarray,
    head_rotation: np.ndarray,
    joint_order: list[str],
    bounds: dict[str, tuple[float, float, float, float]],
    *,
    width: int,
    height: int,
    source_frame: int,
) -> np.ndarray:
    image = Image.new("RGB", (width, height), (7, 9, 12))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    index = {name: offset for offset, name in enumerate(joint_order)}
    segments = (
        ("Hips", "Spine3", "torso"),
        ("Spine3", "Head", "torso"),
        ("Spine3", "LeftArm", "left"),
        ("LeftArm", "LeftForeArm", "left"),
        ("LeftForeArm", "LeftHand", "left"),
        ("Spine3", "RightArm", "right"),
        ("RightArm", "RightForeArm", "right"),
        ("RightForeArm", "RightHand", "right"),
    )

    def draw_view(view: str, top: int, bottom: int) -> None:
        x_min, x_max, z_min, z_max = bounds[view]
        title_height = 34
        margin_x = 38
        margin_y = 22
        available_width = width - 2 * margin_x
        available_height = bottom - top - title_height - 2 * margin_y
        scale = min(
            available_width / (x_max - x_min),
            available_height / (z_max - z_min),
        )
        center_x = 0.5 * (x_min + x_max)
        center_z = 0.5 * (z_min + z_max)
        screen_center_x = width / 2.0
        screen_center_y = top + title_height + margin_y + available_height / 2.0

        def projected_point(value: np.ndarray) -> tuple[int, int]:
            horizontal = float(_horizontal_values(np.asarray(value), view))
            return (
                int(round(screen_center_x + (horizontal - center_x) * scale)),
                int(round(screen_center_y - (value[2] - center_z) * scale)),
            )

        def point(name: str) -> tuple[int, int]:
            return projected_point(positions[index[name]])

        for start, end, color in segments:
            draw.line((point(start), point(end)), fill=COLORS[color], width=7)
        for name in joint_order:
            x, y = point(name)
            radius = 6 if name != "Head" else 10
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=COLORS["joint"],
            )
        head_position = positions[index["Head"]]
        head_up_endpoint = head_position + 0.18 * np.asarray(head_rotation)[:, 2]
        head_forward_endpoint = head_position + 0.18 * np.asarray(head_rotation)[:, 0]
        draw.line(
            (*projected_point(head_position), *projected_point(head_up_endpoint)),
            fill=COLORS["head_axis"],
            width=5,
        )
        draw.line(
            (*projected_point(head_position), *projected_point(head_forward_endpoint)),
            fill=COLORS["head_forward"],
            width=5,
        )
        label = "FRONT (+Y horizontal, +Z up)" if view == "front" else "SIDE (+X forward, +Z up)"
        draw.text((18, top + 24), label, fill=COLORS["muted"], font=font)

    midpoint = height // 2
    draw_view("front", 0, midpoint)
    draw.line((0, midpoint, width, midpoint), fill=(55, 61, 68), width=2)
    draw_view("side", midpoint, height)
    draw.text(
        (18, 8),
        f"ANONYMOUS SOURCE | frame {source_frame}",
        fill=COLORS["text"],
        font=font,
    )
    draw.text((width - 224, 8), "LEFT", fill=COLORS["left"], font=font)
    draw.text((width - 178, 8), "RIGHT", fill=COLORS["right"], font=font)
    draw.text((width - 124, 8), "UP", fill=COLORS["head_axis"], font=font)
    draw.text((width - 92, 8), "FORWARD", fill=COLORS["head_forward"], font=font)
    return np.asarray(image)


def compose_axis_review(
    source_positions: np.ndarray,
    source_head_rotations: np.ndarray,
    source_joint_order: list[str],
    robot_mp4: Path,
    output_mp4: Path,
    *,
    source_start_frame: int,
) -> dict:
    robot_reader = imageio.get_reader(robot_mp4)
    metadata = robot_reader.get_meta_data()
    fps = float(metadata.get("fps") or 0.0)
    if not np.isclose(fps, 30.0, atol=1e-3):
        robot_reader.close()
        raise ValueError(f"Robot evidence FPS must be 30, got {fps}")
    robot_frames = [np.asarray(frame) for frame in robot_reader]
    robot_reader.close()
    if not robot_frames:
        raise ValueError("Robot evidence contains no frames")
    if source_head_rotations.shape != (len(source_positions), 3, 3):
        raise ValueError("Source head rotations do not match source positions")
    robot_height, robot_width = robot_frames[0].shape[:2]
    if any(frame.shape[:2] != (robot_height, robot_width) for frame in robot_frames):
        raise ValueError("Robot evidence changes resolution")
    bounds = {
        view: _projection_bounds(source_positions, view)
        for view in ("front", "side")
    }
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_mp4.stem}.", suffix=".mp4", dir=output_mp4.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", imageio_ffmpeg.get_ffmpeg_exe())
    try:
        with imageio.get_writer(
            temporary,
            fps=30.0,
            codec="libx264",
            pixelformat="yuv420p",
            quality=7,
            macro_block_size=2,
            output_params=["-movflags", "+faststart"],
        ) as writer:
            for robot_index, robot_frame in enumerate(robot_frames):
                if len(robot_frames) == 1:
                    source_index = 0
                else:
                    source_index = int(
                        round(robot_index * (len(source_positions) - 1) / (len(robot_frames) - 1))
                    )
                source_frame = render_source_frame(
                    source_positions[source_index],
                    source_head_rotations[source_index],
                    source_joint_order,
                    bounds,
                    width=SOURCE_PANE_WIDTH,
                    height=robot_height,
                    source_frame=source_start_frame + source_index,
                )
                robot_image = Image.fromarray(robot_frame)
                robot_draw = ImageDraw.Draw(robot_image)
                robot_draw.rectangle((0, 0, robot_width, 28), fill=(7, 9, 12))
                robot_draw.text(
                    (robot_width // 2 - 112, 10),
                    "RETARGET | REAL ULA V2 URDF | MUJOCO",
                    fill=COLORS["text"],
                    font=ImageFont.load_default(),
                )
                robot_draw.text(
                    (18, 10),
                    "ROBOT LEFT",
                    fill=COLORS["left"],
                    font=ImageFont.load_default(),
                )
                robot_draw.text(
                    (robot_width - 86, 10),
                    "ROBOT RIGHT",
                    fill=COLORS["right"],
                    font=ImageFont.load_default(),
                )
                writer.append_data(
                    np.concatenate((source_frame, np.asarray(robot_image)), axis=1)
                )
        temporary.replace(output_mp4)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "frames": len(robot_frames),
        "fps": 30.0,
        "width": SOURCE_PANE_WIDTH + robot_width,
        "height": robot_height,
        "source_frames": len(source_positions),
        "source_to_robot_time_mapping": "endpoint_preserving_linear_index",
        "source_projection": (
            "episode_aligned_dual_view_front_plus_y_z_and_side_plus_x_z"
        ),
        "robot_front_camera_position_axis": ROBOT_FRONT_CAMERA_POSITION_AXIS,
        "robot_front_camera_view_direction": ROBOT_FRONT_CAMERA_VIEW_DIRECTION,
        "robot_front_camera_screen_right_axis": "+Y",
        "robot_side_label_positions": {
            "screen_left": "ROBOT LEFT",
            "screen_right": "ROBOT RIGHT",
        },
        "source_head_orientation_visible": True,
        "public_frame_labels_anonymous": True,
        "identity_or_partner_metadata_drawn": False,
        "audio": "none",
    }


def main() -> None:
    args = parse_args()
    quality = json.loads(args.quality_json.read_text(encoding="utf-8"))
    if quality.get("axis_policy") != INTERACT_NATIVE_AXIS_POLICY:
        raise ValueError("Quality artifact does not use the native InterAct v2 axis policy")
    if quality.get("legacy_gmr_euler_component_reorder_used") is not False:
        raise ValueError("Quality artifact does not reject the legacy GMR Euler reorder")
    if Path(quality["source_bvh"]).resolve() != args.source_bvh.resolve():
        raise ValueError("Quality/source BVH path mismatch")
    interval = quality["source_interval"]
    if [interval["start_frame"], interval["end_frame_exclusive"]] != [
        args.source_start_frame,
        args.source_end_frame,
    ]:
        raise ValueError("Quality/source interval mismatch")
    source = load_interact_bvh_native_v2(
        args.source_bvh,
        start_frame=args.source_start_frame,
        end_frame=args.source_end_frame,
    )
    source_positions = np.asarray(source["review_joint_positions"], dtype=np.float64)
    source_rotations = np.asarray(source["review_joint_rotations"], dtype=np.float64)
    episode_alignment = np.asarray(
        quality["episode_frame_alignment"]["source_world_to_robot_world_matrix"],
        dtype=np.float64,
    )
    if episode_alignment.shape != (3, 3) or not np.allclose(
        episode_alignment.T @ episode_alignment, np.eye(3), atol=1e-6
    ) or not np.isclose(np.linalg.det(episode_alignment), 1.0, atol=1e-6):
        raise ValueError("Quality episode alignment is not a proper rotation")
    source_positions = source_positions @ episode_alignment.T
    source_rotations = episode_alignment @ source_rotations
    head_index = list(source["review_joint_order"]).index("Head")
    result = compose_axis_review(
        source_positions,
        source_rotations[:, head_index],
        list(source["review_joint_order"]),
        args.robot_mp4,
        args.output_mp4,
        source_start_frame=args.source_start_frame,
    )
    summary = {
        "schema_version": "2.0.0",
        "artifact_kind": "interact_native_bvh_source_vs_mujoco_axis_review",
        "source_bvh": str(args.source_bvh.resolve()),
        "source_bvh_sha256": sha256_file(args.source_bvh),
        "quality_json": str(args.quality_json.resolve()),
        "quality_json_sha256": sha256_file(args.quality_json),
        "source_interval": [args.source_start_frame, args.source_end_frame],
        "actor_id": args.actor_id,
        "partner_actor_id": args.partner_actor_id,
        "robot_mp4": str(args.robot_mp4.resolve()),
        "robot_mp4_sha256": sha256_file(args.robot_mp4),
        "output_mp4": str(args.output_mp4.resolve()),
        **result,
        "pilot_scope": "smoke_only_not_representative_of_dataset_pass_rate",
        "axis_visual_qc_status": "pending_blind_human_review",
        "accepted_for_retarget_batch": False,
        "accepted_for_training": False,
    }
    summary["output_mp4_sha256"] = sha256_file(args.output_mp4)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
