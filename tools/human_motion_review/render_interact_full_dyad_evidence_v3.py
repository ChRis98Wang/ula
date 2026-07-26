#!/usr/bin/env python3
"""Render full-span 2x2 source-dyad and real-URDF robot review evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import imageio.v2 as imageio
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.gmr_v2.interact_bvh_adapter import load_interact_bvh_native_v2
from tools.gmr_v2.retarget_xsens_v2 import smooth_and_limit
from tools.human_motion_review.build_interact_full_dyad_review_manifest_v3 import (
    DYAD_KIND,
    load_json,
    sha256_file,
)
from upper_body_skeleton.mujoco_playback import (
    _set_joint_values,
    apply_camera_lookat_z_offset,
    fit_full_body_camera,
    load_preview_model,
    read_joint_csv,
)
from upper_body_skeleton.retarget_v2_18d import (
    CONTRACT_VERSION as ULA_V2_18D_CONTRACT,
    JOINT_LIMITS_18D,
    JOINT_ORDER_18D,
)
from upper_body_skeleton.v2_axis_calibration import DEFAULT_URDF


FPS = 30.0
WIDTH = 1280
HEIGHT = 720
PANE_WIDTH = WIDTH // 2
PANE_HEIGHT = HEIGHT // 2
BACKGROUND = (7, 9, 12)
TEXT = (232, 236, 240)
MUTED = (146, 155, 166)
ACTOR_COLORS = {
    "A": {"torso": (132, 176, 229), "limb": (66, 133, 244), "joint": (211, 230, 255)},
    "B": {"torso": (232, 159, 112), "limb": (239, 99, 62), "joint": (255, 222, 201)},
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
RETIME_RECONSTRUCTION_TOLERANCE_RAD = 3e-6
LINEAGE_CONTRACT = "interact_explicit_monotonic_full_span_frame_lineage_v3"


def implementation_binding() -> dict[str, str]:
    paths = {
        "renderer": Path(__file__).resolve(),
        "native_bvh_adapter": Path(
            __import__(
                "tools.gmr_v2.interact_bvh_adapter", fromlist=["__file__"]
            ).__file__
        ).resolve(),
        "safety_retime_implementation": Path(
            __import__("tools.gmr_v2.retarget_xsens_v2", fromlist=["__file__"]).__file__
        ).resolve(),
        "mujoco_playback": Path(
            __import__("upper_body_skeleton.mujoco_playback", fromlist=["__file__"]).__file__
        ).resolve(),
    }
    return {
        f"{name}_path": str(path)
        for name, path in paths.items()
    } | {
        f"{name}_sha256": sha256_file(path)
        for name, path in paths.items()
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-json", type=Path, required=True)
    parser.add_argument("--output-mp4", type=Path, required=True)
    parser.add_argument("--lineage-json", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--smoothing-window", type=int, default=7)
    return parser.parse_args(argv)


def _atomic_json(path: Path, value: object) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(payload)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_file(path_value: str, digest: str) -> Path:
    path = Path(path_value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != digest:
        raise ValueError(f"Input SHA mismatch: {path}")
    return path


def reconstruct_actor_retime(
    actor: dict[str, Any], *, smoothing_window: int
) -> dict[str, Any]:
    if smoothing_window < 1:
        raise ValueError("smoothing_window must be positive")
    source = _verify_file(actor["source_bvh"], actor["source_bvh_sha256"])
    quality_path = _verify_file(actor["quality_json"], actor["quality_json_sha256"])
    raw_path = _verify_file(actor["raw_csv"], actor["raw_csv_sha256"])
    safe_path = _verify_file(actor["safe_csv"], actor["safe_csv_sha256"])
    quality = load_json(quality_path)
    if quality.get("episode_task_id") != actor["episode_task_id"]:
        raise ValueError("Quality/task binding changed before evidence rendering")
    if quality.get("episode_task_record_sha256") != actor["episode_task_record_sha256"]:
        raise ValueError("Quality/task record SHA changed before evidence rendering")
    if quality.get("source_sha256") != actor["source_bvh_sha256"]:
        raise ValueError("Quality/source SHA changed before evidence rendering")
    if quality.get("output_contract") != ULA_V2_18D_CONTRACT or quality.get("action_dim") != 18:
        raise ValueError("Evidence input is not the real 18D robot contract")
    if quality.get("quality_gate", {}).get("passed") is not True:
        raise ValueError("Evidence input did not pass physical quality gates")
    raw = np.asarray(read_joint_csv(raw_path), dtype=np.float64)
    safe = np.asarray(read_joint_csv(safe_path), dtype=np.float64)
    if raw.shape != (actor["source_frames"], len(JOINT_ORDER_18D)):
        raise ValueError("Raw CSV does not match source-frame lineage")
    if safe.shape != (actor["output_frames"], len(JOINT_ORDER_18D)):
        raise ValueError("Safe CSV does not match output-frame lineage")
    regenerated, key_times, output_times = smooth_and_limit(
        raw,
        FPS,
        float(quality["max_velocity_rad_s"]),
        smoothing_window,
        joint_order=JOINT_ORDER_18D,
        joint_limits=JOINT_LIMITS_18D,
    )
    if regenerated.shape != safe.shape:
        raise ValueError("Reconstructed safety retime changed output frame count")
    max_error = float(np.max(np.abs(regenerated - safe)))
    if max_error > RETIME_RECONSTRUCTION_TOLERANCE_RAD:
        raise ValueError(
            f"Reconstructed safety trajectory differs from safe CSV: {max_error:.3e} rad"
        )
    if bool(len(safe) != len(raw)) is not actor["safety_retimed"]:
        raise ValueError("Manifest safety-retime flag does not match trajectory lengths")
    if not np.all(np.diff(key_times) > 0.0) or not np.all(np.diff(output_times) > 0.0):
        raise ValueError("Reconstructed retime axes are not strictly monotonic")
    return {
        "source_path": source,
        "quality_path": quality_path,
        "raw_path": raw_path,
        "safe_path": safe_path,
        "quality": quality,
        "raw": raw,
        "safe": safe,
        "key_times": np.asarray(key_times, dtype=np.float64),
        "output_times": np.asarray(output_times, dtype=np.float64),
        "reconstruction_max_abs_error_rad": max_error,
    }


def _nearest_monotonic_indices(values: np.ndarray, axis: np.ndarray) -> np.ndarray:
    fractional = np.interp(values, axis, np.arange(len(axis), dtype=np.float64))
    indices = np.floor(fractional + 0.5).astype(np.int64)
    indices = np.maximum.accumulate(np.clip(indices, 0, len(axis) - 1))
    indices[0] = 0
    indices[-1] = len(axis) - 1
    return indices


def build_full_span_lineage(
    *,
    source_start_frame: int,
    source_frames: int,
    actor_retimes: list[dict[str, Any]],
) -> dict[str, Any]:
    if source_frames < 2 or len(actor_retimes) != 2:
        raise ValueError("Full-span dyad lineage requires two actors and at least two source frames")
    if any(len(actor["raw"]) != source_frames for actor in actor_retimes):
        raise ValueError("Actor raw trajectories do not share the dyad source interval")
    evidence_frames = max(source_frames, *(len(actor["safe"]) for actor in actor_retimes))
    evidence_index = np.arange(evidence_frames, dtype=np.int64)
    source_float = np.linspace(0.0, float(source_frames - 1), evidence_frames)
    source_local = np.floor(source_float + 0.5).astype(np.int64)
    source_local = np.maximum.accumulate(source_local)
    source_local[0] = 0
    source_local[-1] = source_frames - 1
    robot_indices = []
    for actor in actor_retimes:
        source_key_axis = np.arange(source_frames, dtype=np.float64)
        stretched_time = np.interp(source_float, source_key_axis, actor["key_times"])
        indices = _nearest_monotonic_indices(stretched_time, actor["output_times"])
        robot_indices.append(indices)
    for values, expected_last in (
        (source_local, source_frames - 1),
        (robot_indices[0], len(actor_retimes[0]["safe"]) - 1),
        (robot_indices[1], len(actor_retimes[1]["safe"]) - 1),
    ):
        if values[0] != 0 or values[-1] != expected_last or np.any(np.diff(values) < 0):
            raise ValueError("Full-span evidence mapping is not endpoint-preserving monotonic")
    return {
        "schema_version": "3.0.0",
        "artifact_kind": "interact_full_dyad_explicit_frame_lineage_v3",
        "lineage_contract": LINEAGE_CONTRACT,
        "evidence_fps": FPS,
        "evidence_frames": evidence_frames,
        "source_frames": source_frames,
        "robot_a_output_frames": len(actor_retimes[0]["safe"]),
        "robot_b_output_frames": len(actor_retimes[1]["safe"]),
        "evidence_frame": evidence_index.tolist(),
        "source_local_frame": source_local.tolist(),
        "source_absolute_frame": (source_start_frame + source_local).tolist(),
        "robot_a_safe_frame": robot_indices[0].tolist(),
        "robot_b_safe_frame": robot_indices[1].tolist(),
        "mapping_policy": (
            "evidence_full_span_to_source_linear_endpoints_then_each_actor_"
            "reconstructed_piecewise_safety_retime_to_safe_frame"
        ),
        "all_mappings_monotonic": True,
        "all_source_and_output_endpoints_included": True,
        "source_or_output_frames_cropped": False,
        "fixed_duration_target_used": False,
    }


def _projection_bounds(
    positions_a: np.ndarray, positions_b: np.ndarray, horizontal_axis: int
) -> tuple[float, float, float, float]:
    horizontal = np.concatenate(
        (positions_a[..., horizontal_axis].ravel(), positions_b[..., horizontal_axis].ravel())
    )
    vertical = np.concatenate((positions_a[..., 2].ravel(), positions_b[..., 2].ravel()))
    x_min, x_max = float(horizontal.min()), float(horizontal.max())
    z_min, z_max = float(vertical.min()), float(vertical.max())
    if x_max - x_min < 1e-5 or z_max - z_min < 1e-5:
        raise ValueError("Source dyad projection is degenerate")
    x_padding = max(0.08, 0.05 * (x_max - x_min))
    z_padding = max(0.08, 0.05 * (z_max - z_min))
    return x_min - x_padding, x_max + x_padding, z_min - z_padding, z_max + z_padding


def _draw_source_actor(
    draw: ImageDraw.ImageDraw,
    positions: np.ndarray,
    head_rotation: np.ndarray,
    joint_order: list[str],
    bounds: tuple[float, float, float, float],
    horizontal_axis: int,
    role: str,
) -> None:
    x_min, x_max, z_min, z_max = bounds
    margin_x, title_height, margin_bottom = 32, 34, 18
    usable_width = PANE_WIDTH - 2 * margin_x
    usable_height = PANE_HEIGHT - title_height - margin_bottom
    scale = min(usable_width / (x_max - x_min), usable_height / (z_max - z_min))
    center_x = 0.5 * (x_min + x_max)
    center_z = 0.5 * (z_min + z_max)
    screen_center_x = PANE_WIDTH / 2.0
    screen_center_y = title_height + usable_height / 2.0
    indices = {name: index for index, name in enumerate(joint_order)}

    def point(value: np.ndarray) -> tuple[int, int]:
        return (
            int(round(screen_center_x + (value[horizontal_axis] - center_x) * scale)),
            int(round(screen_center_y - (value[2] - center_z) * scale)),
        )

    colors = ACTOR_COLORS[role]
    for start, end, color in SEGMENTS:
        draw.line(
            (*point(positions[indices[start]]), *point(positions[indices[end]])),
            fill=colors[color],
            width=5,
        )
    for name in joint_order:
        x, y = point(positions[indices[name]])
        radius = 4 if name != "Head" else 7
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=colors["joint"])
    head = positions[indices["Head"]]
    up = head + 0.16 * np.asarray(head_rotation)[:, 2]
    forward = head + 0.16 * np.asarray(head_rotation)[:, 0]
    draw.line((*point(head), *point(up)), fill=(78, 214, 181), width=3)
    draw.line((*point(head), *point(forward)), fill=(247, 202, 82), width=3)


def _source_pane(
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    head_a: np.ndarray,
    head_b: np.ndarray,
    joint_order: list[str],
    bounds: tuple[float, float, float, float],
    horizontal_axis: int,
    evidence_frame: int,
    evidence_frames: int,
) -> np.ndarray:
    image = Image.new("RGB", (PANE_WIDTH, PANE_HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title = "SOURCE DYAD | X-Z" if horizontal_axis == 0 else "SOURCE DYAD | Y-Z"
    draw.text((14, 10), title, fill=TEXT, font=font)
    draw.text(
        (PANE_WIDTH - 132, 10),
        f"{evidence_frame + 1}/{evidence_frames}",
        fill=MUTED,
        font=font,
    )
    draw.text((PANE_WIDTH - 48, 10), "A", fill=ACTOR_COLORS["A"]["limb"], font=font)
    draw.text((PANE_WIDTH - 26, 10), "B", fill=ACTOR_COLORS["B"]["limb"], font=font)
    _draw_source_actor(draw, positions_a, head_a, joint_order, bounds, horizontal_axis, "A")
    _draw_source_actor(draw, positions_b, head_b, joint_order, bounds, horizontal_axis, "B")
    return np.asarray(image)


def _label_robot(frame: np.ndarray, role: str, evidence_frame: int, evidence_frames: int) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, PANE_WIDTH, 28), fill=BACKGROUND)
    draw.text((14, 9), f"MUJOCO | REAL ULA V2 18D | ROBOT {role}", fill=TEXT, font=font)
    draw.text(
        (PANE_WIDTH - 132, 9),
        f"{evidence_frame + 1}/{evidence_frames}",
        fill=MUTED,
        font=font,
    )
    return np.asarray(image)


def render_record(
    record: dict[str, Any],
    output_mp4: Path,
    lineage_json: Path,
    summary_json: Path,
    *,
    urdf: Path = DEFAULT_URDF,
    smoothing_window: int = 7,
) -> dict[str, Any]:
    if record.get("artifact_kind") != DYAD_KIND or record.get("accepted_for_training") is not False:
        raise ValueError("Renderer requires a fail-closed InterAct full-dyad task")
    if record.get("fixed_duration_window_used") is not False or record.get(
        "duration_gate_used_for_full_pool_admission"
    ) is not False:
        raise ValueError("Renderer refuses a fixed-duration or duration-gated task")
    runtime = record.get("physical_runtime_contract") or {}
    if runtime.get("smoothing_window") != smoothing_window:
        raise ValueError("Renderer smoothing window differs from the physical run contract")
    actors = record.get("actors") or []
    if len(actors) != 2 or [actor.get("anonymous_evidence_role") for actor in actors] != ["A", "B"]:
        raise ValueError("Renderer requires exactly anonymous actor roles A and B")
    actor_retimes = [
        reconstruct_actor_retime(actor, smoothing_window=smoothing_window) for actor in actors
    ]
    if any(
        float(actor["quality"]["max_velocity_rad_s"])
        != float(runtime.get("max_velocity"))
        for actor in actor_retimes
    ):
        raise ValueError("Renderer max-velocity lineage differs from the physical run contract")
    interval = record["source_interval"]
    lineage = build_full_span_lineage(
        source_start_frame=int(interval["start_frame"]),
        source_frames=int(interval["frame_count"]),
        actor_retimes=actor_retimes,
    )
    lineage.update(
        {
            "dyad_id": record["dyad_id"],
            "dyad_record_sha256": record["dyad_record_sha256"],
            "actor_a_retime_reconstruction_max_abs_error_rad": actor_retimes[0][
                "reconstruction_max_abs_error_rad"
            ],
            "actor_b_retime_reconstruction_max_abs_error_rad": actor_retimes[1][
                "reconstruction_max_abs_error_rad"
            ],
        }
    )
    _atomic_json(lineage_json, lineage)

    sources = [
        load_interact_bvh_native_v2(
            actor["source_path"],
            start_frame=interval["start_frame"],
            end_frame=interval["end_frame_exclusive"],
        )
        for actor in actor_retimes
    ]
    if sources[0]["review_joint_order"] != sources[1]["review_joint_order"]:
        raise ValueError("Dyad source skeleton contracts differ")
    joint_order = list(sources[0]["review_joint_order"])
    positions = [np.asarray(source["review_joint_positions"], dtype=np.float64) for source in sources]
    rotations = [np.asarray(source["review_joint_rotations"], dtype=np.float64) for source in sources]
    if any(len(value) != interval["frame_count"] for value in positions):
        raise ValueError("Decoded source dyad does not match the natural interval")
    head_index = joint_order.index("Head")
    bounds = {
        axis: _projection_bounds(positions[0], positions[1], axis) for axis in (0, 1)
    }

    urdf = urdf.resolve()
    if not urdf.is_file():
        raise FileNotFoundError(urdf)
    model, joint_to_qpos, model_source = load_preview_model(
        urdf_path=urdf,
        simplified=False,
        joint_order=JOINT_ORDER_18D,
    )
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), PANE_WIDTH)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), PANE_HEIGHT)
    data = [mujoco.MjData(model), mujoco.MjData(model)]
    cameras = []
    camera_contracts = []
    for actor_index in range(2):
        camera, framing = fit_full_body_camera(
            model,
            data[actor_index],
            actor_retimes[actor_index]["safe"],
            joint_to_qpos,
            width=PANE_WIDTH,
            height=PANE_HEIGHT,
            margin=1.12,
        )
        camera, framing = apply_camera_lookat_z_offset(camera, framing, -0.04)
        cameras.append(camera)
        camera_contracts.append(
            {
                "distance": float(camera.distance),
                "lookat": [float(value) for value in camera.lookat],
                "azimuth_deg": float(camera.azimuth),
                "elevation_deg": float(camera.elevation),
                "framing": framing,
            }
        )

    output_mp4 = output_mp4.resolve()
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_mp4.stem}.", suffix=".mp4", dir=output_mp4.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", imageio_ffmpeg.get_ffmpeg_exe())
    renderer = mujoco.Renderer(model, height=PANE_HEIGHT, width=PANE_WIDTH)
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
            for evidence_frame in range(lineage["evidence_frames"]):
                source_index = lineage["source_local_frame"][evidence_frame]
                source_panes = [
                    _source_pane(
                        positions[0][source_index],
                        positions[1][source_index],
                        rotations[0][source_index, head_index],
                        rotations[1][source_index, head_index],
                        joint_order,
                        bounds[axis],
                        axis,
                        evidence_frame,
                        lineage["evidence_frames"],
                    )
                    for axis in (0, 1)
                ]
                robot_panes = []
                for actor_index, role in enumerate(("A", "B")):
                    safe_index = lineage[f"robot_{role.lower()}_safe_frame"][evidence_frame]
                    _set_joint_values(
                        data[actor_index],
                        actor_retimes[actor_index]["safe"][safe_index],
                        joint_to_qpos,
                    )
                    mujoco.mj_forward(model, data[actor_index])
                    renderer.update_scene(data[actor_index], camera=cameras[actor_index])
                    robot_panes.append(
                        _label_robot(
                            renderer.render().copy(),
                            role,
                            evidence_frame,
                            lineage["evidence_frames"],
                        )
                    )
                frame = np.concatenate(
                    (
                        np.concatenate(source_panes, axis=1),
                        np.concatenate(robot_panes, axis=1),
                    ),
                    axis=0,
                )
                writer.append_data(frame)
        os.chmod(temporary, 0o600)
        os.replace(temporary, output_mp4)
    finally:
        renderer.close()
        temporary.unlink(missing_ok=True)

    summary = {
        "schema_version": "3.0.0",
        "artifact_kind": "interact_full_dyad_2x2_review_evidence_v3",
        "dyad_id": record["dyad_id"],
        "dyad_record_sha256": record["dyad_record_sha256"],
        "output_mp4": str(output_mp4),
        "output_mp4_sha256": sha256_file(output_mp4),
        "frame_lineage_json": str(lineage_json.resolve()),
        "frame_lineage_json_sha256": sha256_file(lineage_json),
        "frames": lineage["evidence_frames"],
        "fps": FPS,
        "width": WIDTH,
        "height": HEIGHT,
        "layout": "2x2_source_dyad_xz_yz_plus_mujoco_robot_a_b",
        "source_view_contract": "shared_world_dyad_xz_and_yz_no_face_or_finger_geometry",
        "robot_contract": ULA_V2_18D_CONTRACT,
        "robot_model_source": model_source,
        "robot_urdf": str(urdf),
        "robot_urdf_sha256": sha256_file(urdf),
        "robot_camera_contracts": camera_contracts,
        "implementation_binding": implementation_binding(),
        "lineage_contract": LINEAGE_CONTRACT,
        "all_source_and_output_endpoints_included": True,
        "source_or_output_frames_cropped": False,
        "fixed_duration_target_used": False,
        "audio": "none",
        "face_geometry_used": False,
        "finger_geometry_used": False,
        "identity_scene_official_emotion_drawn": False,
        "public_frame_labels_anonymous": True,
        "semantic_supervision_mask": False,
        "emotion_supervision_mask": False,
        "license_training_mask": False,
        "accepted_for_training": False,
    }
    _atomic_json(summary_json, summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    record = load_json(args.record_json.resolve())
    summary = render_record(
        record,
        args.output_mp4,
        args.lineage_json,
        args.summary_json,
        urdf=args.urdf,
        smoothing_window=args.smoothing_window,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
