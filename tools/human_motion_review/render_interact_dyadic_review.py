#!/usr/bin/env python3
"""Render an anonymous two-view InterAct dyad for blind expression review."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from tools.gmr_v2.interact_bvh_adapter import (
        INTERACT_NATIVE_AXIS_POLICY,
        load_interact_bvh_native_v2,
    )
except ModuleNotFoundError:  # pragma: no cover - direct invocation by path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.gmr_v2.interact_bvh_adapter import (
        INTERACT_NATIVE_AXIS_POLICY,
        load_interact_bvh_native_v2,
    )


FPS = 30.0
WIDTH = 1280
HEIGHT = 720
PANE_WIDTH = WIDTH // 2
BACKGROUND = (7, 9, 12)
TEXT = (225, 230, 235)
MUTED = (145, 153, 164)
ACTOR_COLORS = {
    "A": {"torso": (109, 158, 219), "limb": (66, 133, 244), "joint": (198, 222, 255)},
    "B": {"torso": (221, 149, 102), "limb": (239, 99, 62), "joint": (255, 218, 195)},
}
SEGMENTS = (
    ("Hips", "Spine3", "torso"),
    ("Spine3", "Head", "torso"),
    ("Spine3", "LeftArm", "limb"),
    ("LeftArm", "LeftForeArm", "limb"),
    ("LeftForeArm", "LeftHand", "limb"),
    ("Spine3", "RightArm", "limb"),
    ("RightArm", "RightForeArm", "limb"),
    ("RightForeArm", "RightHand", "limb"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-a-bvh", type=Path, required=True)
    parser.add_argument("--actor-b-bvh", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--output-mp4", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bounds(
    actor_a: np.ndarray,
    actor_b: np.ndarray,
    horizontal_axis: int,
) -> tuple[float, float, float, float]:
    horizontal = np.concatenate(
        (actor_a[..., horizontal_axis].ravel(), actor_b[..., horizontal_axis].ravel())
    )
    vertical = np.concatenate((actor_a[..., 2].ravel(), actor_b[..., 2].ravel()))
    horizontal_min, horizontal_max = float(horizontal.min()), float(horizontal.max())
    vertical_min, vertical_max = float(vertical.min()), float(vertical.max())
    if horizontal_max - horizontal_min < 1e-5 or vertical_max - vertical_min < 1e-5:
        raise ValueError("InterAct dyadic projection is degenerate")
    return horizontal_min, horizontal_max, vertical_min, vertical_max


def _draw_actor(
    draw: ImageDraw.ImageDraw,
    positions: np.ndarray,
    order: list[str],
    bounds: tuple[float, float, float, float],
    horizontal_axis: int,
    pane_offset: int,
    actor_key: str,
    head_rotation: np.ndarray,
) -> None:
    horizontal_min, horizontal_max, vertical_min, vertical_max = bounds
    margin_x = 44
    title_height = 54
    margin_bottom = 40
    usable_width = PANE_WIDTH - 2 * margin_x
    usable_height = HEIGHT - title_height - margin_bottom
    scale = min(
        usable_width / (horizontal_max - horizontal_min),
        usable_height / (vertical_max - vertical_min),
    )
    center_horizontal = 0.5 * (horizontal_min + horizontal_max)
    center_vertical = 0.5 * (vertical_min + vertical_max)
    center_x = pane_offset + PANE_WIDTH / 2.0
    center_y = title_height + usable_height / 2.0
    indices = {name: index for index, name in enumerate(order)}

    def point(name: str) -> tuple[int, int]:
        value = positions[indices[name]]
        return (
            int(round(center_x + (value[horizontal_axis] - center_horizontal) * scale)),
            int(round(center_y - (value[2] - center_vertical) * scale)),
        )

    colors = ACTOR_COLORS[actor_key]
    for start, end, role in SEGMENTS:
        draw.line((point(start), point(end)), fill=colors[role], width=7)
    for name in order:
        x, y = point(name)
        radius = 6 if name != "Head" else 10
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=colors["joint"])
    head_position = positions[indices["Head"]]

    def project(value: np.ndarray) -> tuple[int, int]:
        return (
            int(
                round(
                    center_x
                    + (value[horizontal_axis] - center_horizontal) * scale
                )
            ),
            int(round(center_y - (value[2] - center_vertical) * scale)),
        )

    head_up = head_position + 0.18 * np.asarray(head_rotation)[:, 2]
    head_forward = head_position + 0.18 * np.asarray(head_rotation)[:, 0]
    draw.line((*project(head_position), *project(head_up)), fill=(78, 214, 181), width=5)
    draw.line(
        (*project(head_position), *project(head_forward)),
        fill=(247, 202, 82),
        width=5,
    )


def render_dyad(
    actor_a: np.ndarray,
    actor_b: np.ndarray,
    actor_a_head_rotations: np.ndarray,
    actor_b_head_rotations: np.ndarray,
    order: list[str],
    output_mp4: Path,
) -> dict:
    if actor_a.shape != actor_b.shape or actor_a.ndim != 3 or actor_a.shape[2] != 3:
        raise ValueError("Aligned InterAct partners must have equal [frames,joints,3] data")
    expected_head_shape = (len(actor_a), 3, 3)
    if actor_a_head_rotations.shape != expected_head_shape or actor_b_head_rotations.shape != expected_head_shape:
        raise ValueError("InterAct dyadic head rotations do not match partner frames")
    bounds_xz = _bounds(actor_a, actor_b, 0)
    bounds_yz = _bounds(actor_a, actor_b, 1)
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_mp4.stem}.", suffix=".mp4", dir=output_mp4.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", imageio_ffmpeg.get_ffmpeg_exe())
    font = ImageFont.load_default()
    try:
        with imageio.get_writer(
            temporary,
            fps=FPS,
            codec="libx264",
            pixelformat="yuv420p",
            quality=7,
            macro_block_size=2,
            output_params=["-movflags", "+faststart"],
        ) as writer:
            for frame_index in range(len(actor_a)):
                image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
                draw = ImageDraw.Draw(image)
                draw.line((PANE_WIDTH, 0, PANE_WIDTH, HEIGHT), fill=(42, 47, 54), width=2)
                draw.text((20, 15), "ANONYMOUS DYAD | X-Z VIEW", fill=TEXT, font=font)
                draw.text((PANE_WIDTH + 20, 15), "ANONYMOUS DYAD | Y-Z VIEW", fill=TEXT, font=font)
                draw.text((20, 34), f"frame {frame_index}", fill=MUTED, font=font)
                draw.text((PANE_WIDTH - 78, 15), "A", fill=ACTOR_COLORS["A"]["limb"], font=font)
                draw.text((PANE_WIDTH - 42, 15), "B", fill=ACTOR_COLORS["B"]["limb"], font=font)
                for actor_key, positions, head_rotations in (
                    ("A", actor_a, actor_a_head_rotations),
                    ("B", actor_b, actor_b_head_rotations),
                ):
                    _draw_actor(
                        draw,
                        positions[frame_index],
                        order,
                        bounds_xz,
                        0,
                        0,
                        actor_key,
                        head_rotations[frame_index],
                    )
                    _draw_actor(
                        draw,
                        positions[frame_index],
                        order,
                        bounds_yz,
                        1,
                        PANE_WIDTH,
                        actor_key,
                        head_rotations[frame_index],
                    )
                writer.append_data(np.asarray(image))
        temporary.replace(output_mp4)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "frames": int(len(actor_a)),
        "fps": FPS,
        "width": WIDTH,
        "height": HEIGHT,
        "audio": "none",
        "views": ["robot_aligned_x_z", "robot_aligned_y_z"],
        "partner_world_translation_preserved": True,
        "root_translation_reset_to_zero": False,
        "source_head_orientation_visible": True,
        "axis_policy": INTERACT_NATIVE_AXIS_POLICY,
    }


def main() -> None:
    args = parse_args()
    if args.start_frame < 0 or args.end_frame <= args.start_frame:
        raise ValueError("Dyadic source interval must be non-empty and half-open")
    sources = [
        load_interact_bvh_native_v2(
            path,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
        )
        for path in (args.actor_a_bvh, args.actor_b_bvh)
    ]
    if sources[0]["review_joint_order"] != sources[1]["review_joint_order"]:
        raise ValueError("InterAct partner skeleton contracts differ")
    head_index = list(sources[0]["review_joint_order"]).index("Head")
    result = render_dyad(
        np.asarray(sources[0]["review_joint_positions"]),
        np.asarray(sources[1]["review_joint_positions"]),
        np.asarray(sources[0]["review_joint_rotations"])[:, head_index],
        np.asarray(sources[1]["review_joint_rotations"])[:, head_index],
        list(sources[0]["review_joint_order"]),
        args.output_mp4.resolve(),
    )
    summary = {
        "schema_version": "2.0.0",
        "artifact_kind": "interact_native_bvh_anonymous_dyadic_expression_review_render",
        "source_interval": [args.start_frame, args.end_frame],
        "actor_a_bvh": str(args.actor_a_bvh.resolve()),
        "actor_a_bvh_sha256": sha256_file(args.actor_a_bvh),
        "actor_b_bvh": str(args.actor_b_bvh.resolve()),
        "actor_b_bvh_sha256": sha256_file(args.actor_b_bvh),
        "output_mp4": str(args.output_mp4.resolve()),
        "output_mp4_sha256": sha256_file(args.output_mp4),
        **result,
        "official_scenario_or_emotion_rendered": False,
        "accepted_for_training": False,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
