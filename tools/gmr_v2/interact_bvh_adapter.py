#!/usr/bin/env python3
"""Parse InterAct BVH into an auditable ULA-V2 source-body adapter.

This module intentionally stops before robot-axis admission.  It proves the
source skeleton/channel mapping and preserves the full Spine3-to-Head chain,
but an independently rendered MuJoCo axis check is still required before its
18D targets may enter training.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable
import warnings

import numpy as np
from scipy.spatial.transform import Rotation


SOURCE_TO_CANONICAL = {
    "Spine3": "Chest4",
    "LeftArm": "LeftShoulder",
    "LeftForeArm": "LeftElbow",
    "LeftHand": "LeftWrist",
    "RightArm": "RightShoulder",
    "RightForeArm": "RightElbow",
    "RightHand": "RightWrist",
}
CANONICAL_BODY_ORDER = (
    "Chest4",
    "LeftShoulder",
    "LeftElbow",
    "LeftWrist",
    "RightShoulder",
    "RightElbow",
    "RightWrist",
)
HEAD_CHAIN = ("Spine3", "Neck", "Neck1", "Head")
ANATOMICAL_FRAME_JOINTS = ("Hips", "Spine3", "LeftArm", "RightArm")
ENERGY_JOINTS = (
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Spine3",
    "Neck",
    "Neck1",
    "Head",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightArm",
    "RightForeArm",
    "RightHand",
)
EXPECTED_FRAME_TIME_SEC = 1.0 / 30.0
FRAME_TIME_TOLERANCE_SEC = 5e-6
AXIS_POLICY = "interact_native_bvh_y_up_robot_alignment_pending_mujoco_v1"
GMR_AXIS_POLICY = "interact_gmr_z_up_explicit_robot_basis_pending_mujoco_v2"
GMR_FRONT_REFLECTION = np.diag([1.0, -1.0, 1.0])
# GMR's converted InterAct skeleton is z-up with anatomical right along +x.
# ULA V2 is z-up with anatomical right along +y and forward along +x.  This
# orthogonal basis maps source (right, forward, up) to robot coordinates while
# retaining the audited front-axis reflection required by the robot chirality.
GMR_IN_PLANE_BASIS_ROTATION = np.array(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
GMR_SOURCE_TO_ROBOT_BASIS = GMR_IN_PLANE_BASIS_ROTATION @ GMR_FRONT_REFLECTION

# InterAct BVH files declare a Y-up frame with forward +X and anatomical right
# +Z.  The source and robot use opposite coordinate handedness, so the direct
# orthogonal basis change has determinant -1.  Conjugating already-composed
# rotation matrices by this basis still yields proper rotations.  Keeping this
# as one post-FK change of basis is important: reordering Euler components is
# not equivalent to respecting each BVH joint's declared rotation channels.
INTERACT_BVH_FORWARD_AXIS = np.array([1.0, 0.0, 0.0], dtype=np.float64)
INTERACT_BVH_RIGHT_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)
INTERACT_BVH_UP_AXIS = np.array([0.0, 1.0, 0.0], dtype=np.float64)
INTERACT_NATIVE_TO_ROBOT_BASIS = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
INTERACT_NATIVE_AXIS_POLICY = (
    "interact_native_bvh_declared_channel_order_robot_basis_v2"
)
GMR_REVIEW_JOINT_ORDER = (
    "Hips",
    "Spine3",
    "Head",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightArm",
    "RightForeArm",
    "RightHand",
)


@dataclass(frozen=True)
class BVHStructure:
    names: tuple[str, ...]
    parents: np.ndarray
    offsets: np.ndarray
    channel_names: tuple[tuple[str, ...], ...]
    channel_starts: np.ndarray
    frame_count: int
    frame_time: float
    motion: np.ndarray | None


def read_bvh_header(path: Path) -> tuple[int, float]:
    """Read frame count/rate without loading the potentially large motion body."""
    frame_count = None
    frame_time = None
    with path.open(encoding="utf-8") as handle:
        in_motion = False
        for raw in handle:
            stripped = raw.strip()
            if stripped == "MOTION":
                in_motion = True
                continue
            if not in_motion:
                continue
            normalized = stripped.replace(":", " ").split()
            if len(normalized) == 2 and normalized[0] == "Frames":
                frame_count = int(normalized[1])
            elif len(normalized) == 3 and normalized[:2] == ["Frame", "Time"]:
                frame_time = float(normalized[2])
                break
    if frame_count is None or frame_time is None:
        raise ValueError("BVH MOTION header is incomplete")
    if frame_count < 2 or frame_time <= 0:
        raise ValueError("BVH must contain at least two positive-time frames")
    return frame_count, frame_time


def _parse_hierarchy(lines: list[str]) -> tuple[
    list[str], list[int], list[list[float]], list[tuple[str, ...]], list[int], int
]:
    names: list[str] = []
    parents: list[int] = []
    offsets: list[list[float]] = []
    channels: list[tuple[str, ...]] = []
    channel_starts: list[int] = []
    stack: list[int | None] = []
    pending: int | None | str = None
    channel_cursor = 0
    motion_line = -1
    for line_number, raw in enumerate(lines):
        stripped = raw.strip()
        if stripped == "MOTION":
            motion_line = line_number
            break
        parts = stripped.split()
        if not parts or parts[0] == "HIERARCHY":
            continue
        if parts[0] in {"ROOT", "JOINT"}:
            if len(parts) != 2:
                raise ValueError(f"Malformed BVH joint declaration at line {line_number + 1}")
            parent = stack[-1] if stack else None
            if parent is None and names:
                raise ValueError("BVH joint cannot be parented to an End Site")
            names.append(parts[1])
            parents.append(-1 if parent is None else int(parent))
            offsets.append([0.0, 0.0, 0.0])
            channels.append(())
            channel_starts.append(channel_cursor)
            pending = len(names) - 1
        elif parts[:2] == ["End", "Site"]:
            pending = "end_site"
        elif parts[0] == "{":
            if pending == "end_site":
                stack.append(None)
            elif isinstance(pending, int):
                stack.append(pending)
            else:
                raise ValueError(f"Unexpected BVH opening brace at line {line_number + 1}")
            pending = None
        elif parts[0] == "}":
            if not stack:
                raise ValueError(f"Unexpected BVH closing brace at line {line_number + 1}")
            stack.pop()
        elif parts[0] == "OFFSET":
            if len(parts) != 4:
                raise ValueError(f"Malformed BVH offset at line {line_number + 1}")
            if stack and stack[-1] is not None:
                offsets[int(stack[-1])] = [float(value) for value in parts[1:]]
        elif parts[0] == "CHANNELS":
            if not stack or stack[-1] is None:
                raise ValueError(f"Channels outside a BVH joint at line {line_number + 1}")
            count = int(parts[1])
            if len(parts) != count + 2:
                raise ValueError(f"Malformed BVH channels at line {line_number + 1}")
            joint = int(stack[-1])
            channel_starts[joint] = channel_cursor
            channels[joint] = tuple(parts[2:])
            channel_cursor += count
    if motion_line < 0:
        raise ValueError("BVH is missing MOTION section")
    if stack:
        raise ValueError("BVH hierarchy has unclosed braces")
    if len(names) != len(set(names)):
        raise ValueError("BVH joint names must be unique")
    return names, parents, offsets, channels, channel_starts, motion_line


def parse_bvh(path: Path, *, load_motion: bool = True) -> BVHStructure:
    lines = path.read_text(encoding="utf-8").splitlines()
    names, parents, offsets, channels, channel_starts, motion_line = _parse_hierarchy(lines)
    if motion_line + 2 >= len(lines):
        raise ValueError("BVH MOTION header is incomplete")
    frames_parts = lines[motion_line + 1].strip().replace(":", " ").split()
    frame_time_parts = lines[motion_line + 2].strip().replace(":", " ").split()
    if len(frames_parts) != 2 or frames_parts[0] != "Frames":
        raise ValueError("BVH is missing Frames header")
    if len(frame_time_parts) != 3 or frame_time_parts[:2] != ["Frame", "Time"]:
        raise ValueError("BVH is missing Frame Time header")
    frame_count = int(frames_parts[1])
    frame_time = float(frame_time_parts[2])
    if frame_count < 2 or frame_time <= 0:
        raise ValueError("BVH must contain at least two positive-time frames")
    total_channels = sum(len(value) for value in channels)
    motion = None
    if load_motion:
        values = np.fromstring("\n".join(lines[motion_line + 3 :]), sep=" ")
        expected = frame_count * total_channels
        if values.size != expected:
            raise ValueError(
                f"BVH motion payload has {values.size} values; expected {expected}"
            )
        motion = values.reshape(frame_count, total_channels)
        if not np.isfinite(motion).all():
            raise ValueError("BVH motion contains non-finite values")
    return BVHStructure(
        names=tuple(names),
        parents=np.asarray(parents, dtype=np.int64),
        offsets=np.asarray(offsets, dtype=np.float64),
        channel_names=tuple(channels),
        channel_starts=np.asarray(channel_starts, dtype=np.int64),
        frame_count=frame_count,
        frame_time=frame_time,
        motion=motion,
    )


def is_30hz(frame_time: float) -> bool:
    return abs(float(frame_time) - EXPECTED_FRAME_TIME_SEC) <= FRAME_TIME_TOLERANCE_SEC


def local_rotation_matrices(
    structure: BVHStructure, joint_names: Iterable[str] | None = None
) -> dict[str, np.ndarray]:
    if structure.motion is None:
        raise ValueError("BVH motion was not loaded")
    selected = set(structure.names if joint_names is None else joint_names)
    unknown = selected.difference(structure.names)
    if unknown:
        raise ValueError(f"BVH is missing required joints: {sorted(unknown)}")
    result: dict[str, np.ndarray] = {}
    for joint, name in enumerate(structure.names):
        if name not in selected:
            continue
        channel_names = structure.channel_names[joint]
        start = int(structure.channel_starts[joint])
        rotation_columns = [
            (offset, channel[0].upper())
            for offset, channel in enumerate(channel_names)
            if channel.lower().endswith("rotation")
        ]
        if not rotation_columns:
            result[name] = np.broadcast_to(
                np.eye(3), (structure.frame_count, 3, 3)
            ).copy()
            continue
        sequence = "".join(axis for _offset, axis in rotation_columns)
        angles = structure.motion[
            :, [start + offset for offset, _axis in rotation_columns]
        ]
        result[name] = Rotation.from_euler(sequence, angles, degrees=True).as_matrix()
    return result


def _local_positions(structure: BVHStructure, scale: float) -> np.ndarray:
    if structure.motion is None:
        raise ValueError("BVH motion was not loaded")
    local = np.broadcast_to(
        structure.offsets[None, :, :] * scale,
        (structure.frame_count, len(structure.names), 3),
    ).copy()
    axes = {"x": 0, "y": 1, "z": 2}
    for joint, channel_names in enumerate(structure.channel_names):
        start = int(structure.channel_starts[joint])
        for offset, channel in enumerate(channel_names):
            if channel.lower().endswith("position"):
                local[:, joint, axes[channel[0].lower()]] += (
                    structure.motion[:, start + offset] * scale
                )
    return local


def forward_kinematics(
    structure: BVHStructure, *, scale: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    rotations_by_name = local_rotation_matrices(structure)
    local_positions = _local_positions(structure, scale)
    frame_count = structure.frame_count
    joint_count = len(structure.names)
    global_rotations = np.empty((frame_count, joint_count, 3, 3), dtype=np.float64)
    global_positions = np.empty((frame_count, joint_count, 3), dtype=np.float64)
    for joint, name in enumerate(structure.names):
        local_rotation = rotations_by_name[name]
        parent = int(structure.parents[joint])
        if parent < 0:
            global_rotations[:, joint] = local_rotation
            global_positions[:, joint] = local_positions[:, joint]
        else:
            global_rotations[:, joint] = global_rotations[:, parent] @ local_rotation
            global_positions[:, joint] = global_positions[:, parent] + np.einsum(
                "fij,fj->fi", global_rotations[:, parent], local_positions[:, joint]
            )
    return global_positions, global_rotations


def adapt_interact_bvh(path: Path, *, scale: float = 0.01) -> dict[str, object]:
    structure = parse_bvh(path, load_motion=True)
    required = set(SOURCE_TO_CANONICAL) | set(HEAD_CHAIN)
    missing = sorted(required.difference(structure.names))
    if missing:
        raise ValueError(f"InterAct BVH is missing adapter joints: {missing}")
    global_positions, global_rotations = forward_kinematics(structure, scale=scale)
    indices = {name: structure.names.index(name) for name in required}
    canonical_positions = {
        canonical: global_positions[:, indices[source]].copy()
        for source, canonical in SOURCE_TO_CANONICAL.items()
    }
    canonical_quaternions_wxyz = {
        canonical: Rotation.from_matrix(
            global_rotations[:, indices[source]]
        ).as_quat(scalar_first=True)
        for source, canonical in SOURCE_TO_CANONICAL.items()
    }
    canonical_positions_stacked = np.stack(
        [canonical_positions[name] for name in CANONICAL_BODY_ORDER], axis=1
    )
    canonical_quaternions_stacked_wxyz = np.stack(
        [canonical_quaternions_wxyz[name] for name in CANONICAL_BODY_ORDER], axis=1
    )
    spine3_global = global_rotations[:, indices["Spine3"]]
    head_global = global_rotations[:, indices["Head"]]
    head_relative = np.swapaxes(spine3_global, 1, 2) @ head_global
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        head_xyz = Rotation.from_matrix(head_relative).as_euler("XYZ")
    head_xyz = np.unwrap(head_xyz, axis=0)
    head_native_3dof = np.column_stack(
        (head_xyz[:, 0], head_xyz[:, 1], -head_xyz[:, 2])
    )
    return {
        "structure": structure,
        "canonical_positions": canonical_positions,
        "canonical_quaternions_wxyz": canonical_quaternions_wxyz,
        "canonical_body_order": list(CANONICAL_BODY_ORDER),
        "canonical_positions_stacked": canonical_positions_stacked,
        "canonical_quaternions_stacked_wxyz": canonical_quaternions_stacked_wxyz,
        "head_relative_rotations": head_relative,
        "head_native_unaligned_3dof": head_native_3dof,
        "source_to_canonical": dict(SOURCE_TO_CANONICAL),
        "head_chain": list(HEAD_CHAIN),
        "axis_policy": AXIS_POLICY,
    }


def canonicalize_gmr_global_data(
    names: Iterable[str],
    global_positions: np.ndarray,
    global_quaternions_wxyz: np.ndarray,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> dict[str, object]:
    """Map GMR's parsed InterAct skeleton to the existing retarget source API."""
    names = tuple(names)
    if len(names) != len(set(names)):
        raise ValueError("GMR BVH joint names must be unique")
    positions = np.asarray(global_positions, dtype=np.float64)
    quaternions = np.asarray(global_quaternions_wxyz, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[1:] != (len(names), 3):
        raise ValueError("GMR global positions have the wrong shape")
    if quaternions.shape != (len(positions), len(names), 4):
        raise ValueError("GMR global quaternions have the wrong shape")
    required = set(SOURCE_TO_CANONICAL) | set(HEAD_CHAIN) | set(ANATOMICAL_FRAME_JOINTS)
    missing = sorted(required.difference(names))
    if missing:
        raise ValueError(f"InterAct GMR parse is missing adapter joints: {missing}")
    end = len(positions) if end_frame is None else int(end_frame)
    start = int(start_frame)
    if start < 0 or end <= start or end > len(positions):
        raise ValueError(f"Invalid half-open GMR frame interval [{start}, {end})")
    indices = {name: names.index(name) for name in required}
    selected_positions = positions[start:end]
    selected_quaternions = quaternions[start:end]
    matrices = Rotation.from_quat(
        selected_quaternions.reshape(-1, 4), scalar_first=True
    ).as_matrix().reshape(len(selected_positions), len(names), 3, 3)
    aligned_positions = selected_positions @ GMR_SOURCE_TO_ROBOT_BASIS.T
    aligned_matrices = (
        GMR_SOURCE_TO_ROBOT_BASIS
        @ matrices
        @ GMR_SOURCE_TO_ROBOT_BASIS.T
    )
    anatomical_frame_rotations = anatomical_frames_from_landmarks(
        aligned_positions[:, indices["Hips"]],
        aligned_positions[:, indices["Spine3"]],
        aligned_positions[:, indices["LeftArm"]],
        aligned_positions[:, indices["RightArm"]],
    )
    canonical_positions = np.stack(
        [
            aligned_positions[:, indices[source]]
            for source, canonical in SOURCE_TO_CANONICAL.items()
            if canonical in CANONICAL_BODY_ORDER
        ],
        axis=1,
    )
    # Dict insertion order matches the explicit canonical order declaration.
    produced_order = tuple(SOURCE_TO_CANONICAL.values())
    if produced_order != CANONICAL_BODY_ORDER:
        raise RuntimeError("InterAct source mapping no longer matches retarget body order")
    canonical_matrices = np.stack(
        [aligned_matrices[:, indices[source]] for source in SOURCE_TO_CANONICAL],
        axis=1,
    )
    canonical_quaternions = Rotation.from_matrix(
        canonical_matrices.reshape(-1, 3, 3)
    ).as_quat(scalar_first=True).reshape(
        len(selected_positions), len(CANONICAL_BODY_ORDER), 4
    )
    spine3 = aligned_matrices[:, indices["Spine3"]]
    head = aligned_matrices[:, indices["Head"]]
    head_relative = np.swapaxes(spine3, 1, 2) @ head
    review_positions = np.stack(
        [aligned_positions[:, names.index(name)] for name in GMR_REVIEW_JOINT_ORDER],
        axis=1,
    )
    return {
        "canonical_body_order": list(CANONICAL_BODY_ORDER),
        "canonical_positions_stacked": canonical_positions,
        "canonical_quaternions_stacked_wxyz": canonical_quaternions,
        "head_relative_rotations": head_relative,
        "anatomical_frame_rotations": anatomical_frame_rotations,
        "review_joint_order": list(GMR_REVIEW_JOINT_ORDER),
        "review_joint_positions": review_positions,
        "axis_alignment": GMR_SOURCE_TO_ROBOT_BASIS.copy(),
        "axis_alignment_determinant": float(np.linalg.det(GMR_SOURCE_TO_ROBOT_BASIS)),
        "axis_front_reflection": GMR_FRONT_REFLECTION.copy(),
        "axis_in_plane_basis_rotation": GMR_IN_PLANE_BASIS_ROTATION.copy(),
        "source_right_axis_in_robot_coordinates": (
            GMR_SOURCE_TO_ROBOT_BASIS @ np.array([1.0, 0.0, 0.0])
        ),
        "source_forward_axis_in_robot_coordinates": (
            GMR_SOURCE_TO_ROBOT_BASIS @ np.array([0.0, 1.0, 0.0])
        ),
        "source_up_axis_in_robot_coordinates": (
            GMR_SOURCE_TO_ROBOT_BASIS @ np.array([0.0, 0.0, 1.0])
        ),
        "axis_policy": GMR_AXIS_POLICY,
        "source_start_frame": start,
        "source_end_frame_exclusive": end,
        "source_total_frames": len(positions),
    }


def load_interact_bvh_native_v2(
    path: Path,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> dict[str, object]:
    """Parse InterAct in its declared BVH channel order, then change basis.

    The legacy GMR Xsens parser permutes Euler components into an XYZ vector
    before composing them.  InterAct declares Zrotation/Yrotation/Xrotation,
    so that shortcut changes the actual pose.  This loader first performs FK in
    the native BVH frame and only then applies the explicit orthogonal basis
    change used by the InterAct-to-robot contract.
    """
    structure = parse_bvh(Path(path), load_motion=True)
    required = (
        set(SOURCE_TO_CANONICAL)
        | set(HEAD_CHAIN)
        | set(ANATOMICAL_FRAME_JOINTS)
        | set(GMR_REVIEW_JOINT_ORDER)
    )
    missing = sorted(required.difference(structure.names))
    if missing:
        raise ValueError(f"InterAct BVH is missing native-v2 adapter joints: {missing}")
    end = structure.frame_count if end_frame is None else int(end_frame)
    start = int(start_frame)
    if start < 0 or end <= start or end > structure.frame_count:
        raise ValueError(f"Invalid half-open native BVH frame interval [{start}, {end})")

    global_positions, global_rotations = forward_kinematics(structure, scale=0.01)
    selected_positions = global_positions[start:end]
    selected_rotations = global_rotations[start:end]
    basis = INTERACT_NATIVE_TO_ROBOT_BASIS
    aligned_positions = selected_positions @ basis.T
    aligned_rotations = basis @ selected_rotations @ basis.T
    indices = {name: structure.names.index(name) for name in required}

    anatomical_frame_rotations = anatomical_frames_from_landmarks(
        aligned_positions[:, indices["Hips"]],
        aligned_positions[:, indices["Spine3"]],
        aligned_positions[:, indices["LeftArm"]],
        aligned_positions[:, indices["RightArm"]],
    )
    canonical_positions = np.stack(
        [aligned_positions[:, indices[source]] for source in SOURCE_TO_CANONICAL],
        axis=1,
    )
    if tuple(SOURCE_TO_CANONICAL.values()) != CANONICAL_BODY_ORDER:
        raise RuntimeError("InterAct source mapping no longer matches retarget body order")
    canonical_matrices = np.stack(
        [aligned_rotations[:, indices[source]] for source in SOURCE_TO_CANONICAL],
        axis=1,
    )
    canonical_quaternions = Rotation.from_matrix(
        canonical_matrices.reshape(-1, 3, 3)
    ).as_quat(scalar_first=True).reshape(
        len(selected_positions), len(CANONICAL_BODY_ORDER), 4
    )
    spine3 = aligned_rotations[:, indices["Spine3"]]
    head = aligned_rotations[:, indices["Head"]]
    spine3_relative = np.swapaxes(spine3, 1, 2) @ head
    head_relative = head_rotations_in_anatomical_parent_frame(
        anatomical_frame_rotations, head
    )
    review_positions = np.stack(
        [aligned_positions[:, indices[name]] for name in GMR_REVIEW_JOINT_ORDER],
        axis=1,
    )
    review_rotations = np.stack(
        [aligned_rotations[:, indices[name]] for name in GMR_REVIEW_JOINT_ORDER],
        axis=1,
    )

    return {
        "canonical_body_order": list(CANONICAL_BODY_ORDER),
        "canonical_positions_stacked": canonical_positions,
        "canonical_quaternions_stacked_wxyz": canonical_quaternions,
        "head_relative_rotations": head_relative,
        "head_spine3_relative_rotations_diagnostic": spine3_relative,
        "head_parent_frame_policy": "per_frame_anatomical_torso_frame",
        "anatomical_frame_rotations": anatomical_frame_rotations,
        "review_joint_order": list(GMR_REVIEW_JOINT_ORDER),
        "review_joint_positions": review_positions,
        "review_joint_rotations": review_rotations,
        "axis_alignment": basis.copy(),
        "axis_alignment_determinant": float(np.linalg.det(basis)),
        "source_forward_axis_in_robot_coordinates": (
            basis @ INTERACT_BVH_FORWARD_AXIS
        ),
        "source_right_axis_in_robot_coordinates": basis @ INTERACT_BVH_RIGHT_AXIS,
        "source_up_axis_in_robot_coordinates": basis @ INTERACT_BVH_UP_AXIS,
        "axis_policy": INTERACT_NATIVE_AXIS_POLICY,
        "bvh_rotation_composition": "declared_channel_order_intrinsic",
        "legacy_gmr_euler_component_reorder_used": False,
        "source_start_frame": start,
        "source_end_frame_exclusive": end,
        "source_total_frames": structure.frame_count,
        "frame_time_sec": float(structure.frame_time),
        "root_translation_reset_to_zero": False,
    }


def head_rotations_in_anatomical_parent_frame(
    anatomical_frame_rotations: np.ndarray,
    head_global_rotations: np.ndarray,
) -> np.ndarray:
    """Express head orientation in the same torso frame used by robot IK."""
    anatomical = np.asarray(anatomical_frame_rotations, dtype=np.float64)
    head = np.asarray(head_global_rotations, dtype=np.float64)
    if anatomical.shape != head.shape or anatomical.ndim != 3 or anatomical.shape[1:] != (
        3,
        3,
    ):
        raise ValueError("Anatomical and head rotations must both have shape [frames, 3, 3]")
    relative = np.swapaxes(anatomical, 1, 2) @ head
    if not np.allclose(np.linalg.det(relative), 1.0, atol=1e-6):
        raise ValueError("Anatomical-parent head rotations must be proper")
    return relative


def anatomical_frames_from_landmarks(
    hips: np.ndarray,
    chest: np.ndarray,
    left_shoulder: np.ndarray,
    right_shoulder: np.ndarray,
) -> np.ndarray:
    """Build proper [forward, right, up] body frames from observable landmarks."""
    hips = np.asarray(hips, dtype=np.float64)
    chest = np.asarray(chest, dtype=np.float64)
    left_shoulder = np.asarray(left_shoulder, dtype=np.float64)
    right_shoulder = np.asarray(right_shoulder, dtype=np.float64)
    if not (
        hips.shape == chest.shape == left_shoulder.shape == right_shoulder.shape
        and hips.ndim == 2
        and hips.shape[1] == 3
    ):
        raise ValueError("InterAct anatomical landmarks must all have shape [frames, 3]")

    def normalize_rows(values: np.ndarray, label: str) -> np.ndarray:
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if np.any(norms < 1e-8):
            raise ValueError(f"Degenerate InterAct anatomical {label} axis")
        return values / norms

    right = normalize_rows(right_shoulder - left_shoulder, "right")
    up_hint = chest - hips
    up_orthogonal = up_hint - np.sum(up_hint * right, axis=1, keepdims=True) * right
    up = normalize_rows(up_orthogonal, "up")
    forward = normalize_rows(np.cross(right, up), "forward")
    up = normalize_rows(np.cross(forward, right), "recomputed_up")
    frames = np.stack((forward, right, up), axis=2)
    if not np.allclose(
        np.swapaxes(frames, 1, 2) @ frames,
        np.eye(3),
        atol=1e-6,
    ) or not np.allclose(np.linalg.det(frames), 1.0, atol=1e-6):
        raise ValueError("InterAct anatomical body frames must be proper rotations")
    return frames


def load_interact_bvh_for_gmr(
    path: Path,
    gmr_root: Path,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
    reset_to_zero: bool = True,
) -> dict[str, object]:
    """Use GMR's existing BVH parser, then apply the InterAct name adapter."""
    import sys

    root = str(Path(gmr_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from general_motion_retargeting.utils.lafan_vendor import utils
    from general_motion_retargeting.utils.xsens_vendor.BVHParser import Anim, BVHParser

    parser = BVHParser(axis_order="zxy", scale=0.01)
    rotations, _ = parser.parse(Path(path).read_text(encoding="utf-8"))
    quaternions, positions, offsets, parents = parser._MOTION_data_post_processing(
        rotations,
        np.copy(parser.positions),
        reset_to_zero=reset_to_zero,
    )
    animation = Anim(quaternions, positions, offsets, parents, parser.names)
    global_quaternions, global_positions = utils.quat_fk(
        animation.quats, animation.pos, animation.parents
    )
    result = canonicalize_gmr_global_data(
        parser.names,
        global_positions,
        global_quaternions,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    result["frame_time_sec"] = float(parser.frame_time)
    result["source_names"] = list(parser.names)
    result["root_translation_reset_to_zero"] = bool(reset_to_zero)
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_smoke_report(path: Path) -> dict[str, object]:
    adapted = adapt_interact_bvh(path)
    structure = adapted["structure"]
    assert isinstance(structure, BVHStructure)
    positions = adapted["canonical_positions"]
    head = np.asarray(adapted["head_native_unaligned_3dof"])
    geometry_finite = all(np.isfinite(value).all() for value in positions.values())
    nondegenerate_limbs = all(
        float(np.median(np.linalg.norm(positions[wrist] - positions[elbow], axis=1)))
        > 1e-4
        for elbow, wrist in (("LeftElbow", "LeftWrist"), ("RightElbow", "RightWrist"))
    )
    parser_mapping_passed = bool(
        is_30hz(structure.frame_time)
        and geometry_finite
        and np.isfinite(head).all()
        and nondegenerate_limbs
    )
    return {
        "schema_version": "1.0.0",
        "artifact_kind": "interact_bvh_to_ula_v2_18d_adapter_smoke",
        "source_bvh": str(path.resolve()),
        "source_bvh_sha256": sha256_file(path),
        "frame_count": structure.frame_count,
        "frame_time_sec": structure.frame_time,
        "fps": 1.0 / structure.frame_time,
        "source_to_canonical": dict(SOURCE_TO_CANONICAL),
        "head_chain": list(HEAD_CHAIN),
        "face_channels_used": False,
        "finger_joints_used": False,
        "audio_used": False,
        "parser_mapping_smoke_passed": parser_mapping_passed,
        "axis_policy": AXIS_POLICY,
        "axis_visual_qc_status": "pending_mujoco_blind_direction_review",
        "accepted_for_18d_retarget": False,
        "admission_blockers": ["robot_axis_alignment_not_yet_visually_verified"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = adapter_smoke_report(args.bvh)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
