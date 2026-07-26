#!/usr/bin/env python3
"""Retarget one native-length InterAct actor episode to ULA V2 18D.

This is a smoke/QC adapter, not a training admission path.  It uses the same
GMR/Mink body solver and 18D append-only contract as the existing adapters, but
keeps the InterAct axis visual gate and CC-BY-NC-SA use gate explicitly false.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    from .interact_bvh_adapter import (
        CANONICAL_BODY_ORDER,
        INTERACT_BVH_FORWARD_AXIS,
        INTERACT_BVH_RIGHT_AXIS,
        INTERACT_BVH_UP_AXIS,
        INTERACT_NATIVE_AXIS_POLICY,
        INTERACT_NATIVE_TO_ROBOT_BASIS,
        is_30hz,
        load_interact_bvh_native_v2,
    )
    from .retarget_motionx322_v2 import (
        DEFAULT_CONFIG,
        DEFAULT_GMR_ROOT,
        DEFAULT_URDF,
        JOINT_LIMITS_18D,
        JOINT_ORDER_18D,
        ULA_V2_18D_CONTRACT,
        configure_retargeter,
        decompose_v2_head_rotations,
        enforce_human_elbow_branch,
        enforce_safe_elbow_branch,
        extract_joint_row,
        head_quality_metrics,
        quality_report,
        rendered_pose_metrics,
        retime_targets,
        sha256,
        smooth_and_limit,
        write_csv,
    )
except ImportError:  # pragma: no cover - direct invocation
    from interact_bvh_adapter import (
        CANONICAL_BODY_ORDER,
        INTERACT_BVH_FORWARD_AXIS,
        INTERACT_BVH_RIGHT_AXIS,
        INTERACT_BVH_UP_AXIS,
        INTERACT_NATIVE_AXIS_POLICY,
        INTERACT_NATIVE_TO_ROBOT_BASIS,
        is_30hz,
        load_interact_bvh_native_v2,
    )
    from retarget_motionx322_v2 import (
        DEFAULT_CONFIG,
        DEFAULT_GMR_ROOT,
        DEFAULT_URDF,
        JOINT_LIMITS_18D,
        JOINT_ORDER_18D,
        ULA_V2_18D_CONTRACT,
        configure_retargeter,
        decompose_v2_head_rotations,
        enforce_human_elbow_branch,
        enforce_safe_elbow_branch,
        extract_joint_row,
        head_quality_metrics,
        quality_report,
        rendered_pose_metrics,
        retime_targets,
        sha256,
        smooth_and_limit,
        write_csv,
    )


INTERACT_REVISION = "152ba832f379c465f5b1e10c67166d646014d675"
INTERACT_LICENSE = "CC-BY-NC-SA-4.0"
OUTPUT_FPS = 30.0
SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-task-id", required=True)
    parser.add_argument("--episode-task-record-sha256")
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--partner-actor-id")
    parser.add_argument("--gmr-root", type=Path, default=DEFAULT_GMR_ROOT)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--warmup-source-frames", type=int, default=30)
    parser.add_argument("--max-velocity", type=float, default=3.0)
    parser.add_argument("--smoothing-window", type=int, default=7)
    parser.add_argument("--posture-cost", type=float, default=0.02)
    parser.add_argument("--solver", default="daqp")
    parser.add_argument(
        "--elbow-branch",
        choices=("unconstrained", "motionx_negative"),
        default="unconstrained",
        help="InterAct/Xsens defaults to unconstrained; Motion-X's branch is an A/B only",
    )
    parser.add_argument(
        "--processing-scope",
        choices=(
            "axis_smoke_only_not_representative_of_dataset_pass_rate",
            "full_pool_physical_preview_pending_blind_semantic_affect_review",
        ),
        default="axis_smoke_only_not_representative_of_dataset_pass_rate",
    )
    return parser.parse_args(argv)


def output_stem(task_id: str) -> str:
    value = SAFE_ID.sub("_", task_id).strip("._")
    if not value:
        raise ValueError("episode-task-id does not contain a safe filename character")
    return value


def _body_pose(model, data, mujoco, name: str) -> tuple[np.ndarray, np.ndarray]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"Body not found in V2 model: {name}")
    return data.xpos[body_id].copy(), data.xmat[body_id].reshape(3, 3).copy()


def _normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("Degenerate InterAct limb or shoulder vector")
    return np.asarray(vector, dtype=np.float64) / norm


def episode_world_alignment(
    anatomical_frame_rotations: np.ndarray,
    robot_reference_rotation: np.ndarray,
    reference_frame_index: int,
) -> np.ndarray:
    """Align the cataloged episode onset to the neutral robot torso frame."""
    frames = np.asarray(anatomical_frame_rotations, dtype=np.float64)
    robot = np.asarray(robot_reference_rotation, dtype=np.float64)
    reference_frame_index = int(reference_frame_index)
    if frames.ndim != 3 or frames.shape[1:] != (3, 3):
        raise ValueError("InterAct anatomical frames must have shape [frames, 3, 3]")
    if not 0 <= reference_frame_index < len(frames):
        raise ValueError("InterAct episode reference frame is outside solver context")
    if robot.shape != (3, 3):
        raise ValueError("Robot reference rotation must have shape [3, 3]")
    alignment = robot @ frames[reference_frame_index].T
    if not np.allclose(alignment.T @ alignment, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(alignment), 1.0, atol=1e-6
    ):
        raise ValueError("InterAct episode frame alignment must be a proper rotation")
    return alignment


def prepare_interact_target_builder(
    model,
    mujoco,
    positions,
    quaternions,
    anatomical_frame_rotations,
    *,
    reference_frame_index: int,
):
    """Align the complete source episode frame to the neutral robot torso frame."""
    positions = np.asarray(positions, dtype=np.float64)
    quaternions = np.asarray(quaternions, dtype=np.float64)
    anatomical_frame_rotations = np.asarray(
        anatomical_frame_rotations, dtype=np.float64
    )
    if anatomical_frame_rotations.shape != (len(positions), 3, 3):
        raise ValueError("InterAct anatomical frames do not match source positions")
    reference_frame_index = int(reference_frame_index)
    if not 0 <= reference_frame_index < len(positions):
        raise ValueError("InterAct episode reference frame is outside solver context")
    source_index = {name: index for index, name in enumerate(CANONICAL_BODY_ORDER)}
    data = mujoco.MjData(model)
    neutral_qpos = np.zeros(model.nq, dtype=np.float64)
    for joint_name in ("joint_lShoulderRoll", "joint_rShoulderRoll"):
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        neutral_qpos[model.jnt_qposadr[joint_id]] = -1.4
    data.qpos[:] = neutral_qpos
    mujoco.mj_forward(model, data)

    neutral = {
        name: _body_pose(model, data, mujoco, name)
        for name in (
            "link_torso",
            "link_lShoulderPitch",
            "link_lElbow",
            "link_lWristPitch",
            "link_rShoulderPitch",
            "link_rElbow",
            "link_rWristPitch",
        )
    }
    source_reference_rotation = anatomical_frame_rotations[reference_frame_index]
    robot_reference_rotation = neutral["link_torso"][1]
    source_world_to_robot_world = episode_world_alignment(
        anatomical_frame_rotations,
        robot_reference_rotation,
        reference_frame_index,
    )
    reference_wrist_rotation = {
        name: Rotation.from_quat(
            quaternions[reference_frame_index, source_index[name]], scalar_first=True
        ).as_matrix()
        for name in ("LeftWrist", "RightWrist")
    }

    source_right = source_reference_rotation[:, 1]
    robot_right = _normalized(
        neutral["link_rShoulderPitch"][0]
        - neutral["link_lShoulderPitch"][0]
    )
    aligned_source_right = _normalized(source_world_to_robot_world @ source_right)
    alignment_report = {
        "policy": "native_declared_bvh_fk_then_proper_episode_frame_rotation_v2",
        "source_reference_frame_index_in_solver_context": reference_frame_index,
        "source_reference_role": "cataloged_natural_episode_onset_not_warmup",
        "source_world_to_robot_world_matrix": source_world_to_robot_world.tolist(),
        "determinant": float(np.linalg.det(source_world_to_robot_world)),
        "source_right_axis": source_right.tolist(),
        "aligned_source_right_axis": aligned_source_right.tolist(),
        "robot_right_axis": robot_right.tolist(),
        "right_axis_cosine_after_alignment": float(
            np.dot(aligned_source_right, robot_right)
        ),
        "left_right_identity_swapped": False,
        "per_joint_sign_override_used": False,
    }

    def target_for_frame(frame_index: int) -> dict:
        chest_rotation = anatomical_frame_rotations[frame_index]
        torso_pos, torso_neutral_rotation = neutral["link_torso"]
        torso_target_rotation = source_world_to_robot_world @ chest_rotation
        robot_torso_delta = torso_target_rotation @ torso_neutral_rotation.T
        target = {
            "Chest4": [
                torso_pos.copy(),
                Rotation.from_matrix(torso_target_rotation).as_quat(
                    scalar_first=True
                ),
            ]
        }
        for side in ("Left", "Right"):
            shoulder_name = f"{side}Shoulder"
            elbow_name = f"{side}Elbow"
            wrist_name = f"{side}Wrist"
            prefix = "l" if side == "Left" else "r"
            shoulder_body = f"link_{prefix}ShoulderPitch"
            elbow_body = f"link_{prefix}Elbow"
            wrist_body = f"link_{prefix}WristPitch"
            shoulder_neutral = neutral[shoulder_body][0]
            shoulder_target = torso_pos + robot_torso_delta @ (
                shoulder_neutral - torso_pos
            )
            upper_length = float(
                np.linalg.norm(neutral[elbow_body][0] - shoulder_neutral)
            )
            fore_length = float(
                np.linalg.norm(neutral[wrist_body][0] - neutral[elbow_body][0])
            )
            shoulder = positions[frame_index, source_index[shoulder_name]]
            elbow = positions[frame_index, source_index[elbow_name]]
            wrist = positions[frame_index, source_index[wrist_name]]
            upper_direction = source_world_to_robot_world @ _normalized(
                elbow - shoulder
            )
            fore_direction = source_world_to_robot_world @ _normalized(wrist - elbow)
            elbow_target = shoulder_target + upper_length * upper_direction
            wrist_target = elbow_target + fore_length * fore_direction

            wrist_rotation = Rotation.from_quat(
                quaternions[frame_index, source_index[wrist_name]],
                scalar_first=True,
            ).as_matrix()
            source_wrist_delta = (
                wrist_rotation @ reference_wrist_rotation[wrist_name].T
            )
            robot_wrist_delta = (
                source_world_to_robot_world
                @ source_wrist_delta
                @ source_world_to_robot_world.T
            )
            wrist_target_rotation = robot_wrist_delta @ neutral[wrist_body][1]
            target[elbow_name] = [
                elbow_target,
                np.array([1.0, 0.0, 0.0, 0.0]),
            ]
            target[wrist_name] = [
                wrist_target,
                Rotation.from_matrix(wrist_target_rotation).as_quat(
                    scalar_first=True
                ),
            ]
        return target

    return neutral_qpos, target_for_frame, alignment_report


def limb_direction_metrics(model, mujoco, trajectory, targets) -> dict:
    """Compare target and achieved upper/forearm directions, independent of labels."""
    data = mujoco.MjData(model)
    joint_addresses = []
    for name in JOINT_ORDER_18D:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        joint_addresses.append(int(model.jnt_qposadr[joint_id]))
    values: dict[str, list[float]] = {
        "left_upper": [],
        "left_fore": [],
        "right_upper": [],
        "right_fore": [],
    }

    def body_position(name: str) -> np.ndarray:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        return data.xpos[body_id].copy()

    def cosine(left: np.ndarray, right: np.ndarray) -> float:
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        return 1.0 if denominator < 1e-12 else float(np.dot(left, right) / denominator)

    for row, target in zip(trajectory, targets):
        data.qpos[:] = 0.0
        data.qpos[joint_addresses] = row
        mujoco.mj_forward(model, data)
        for side, prefix, key_prefix in (
            ("Left", "l", "left"),
            ("Right", "r", "right"),
        ):
            shoulder = body_position(f"link_{prefix}ShoulderPitch")
            elbow = body_position(f"link_{prefix}Elbow")
            wrist = body_position(f"link_{prefix}WristPitch")
            target_elbow = np.asarray(target[f"{side}Elbow"][0])
            target_wrist = np.asarray(target[f"{side}Wrist"][0])
            values[f"{key_prefix}_upper"].append(
                cosine(target_elbow - shoulder, elbow - shoulder)
            )
            values[f"{key_prefix}_fore"].append(
                cosine(target_wrist - target_elbow, wrist - elbow)
            )
    flattened = np.concatenate([np.asarray(item) for item in values.values()])
    per_segment = {
        name: {
            "minimum": float(np.min(item)),
            "p01": float(np.percentile(item, 1)),
            "p05": float(np.percentile(item, 5)),
            "mean": float(np.mean(item)),
        }
        for name, raw in values.items()
        for item in (np.asarray(raw),)
    }
    p01 = float(np.percentile(flattened, 1))
    return {
        "limb_direction_cosine": per_segment,
        "limb_direction_cosine_all_p01": p01,
        "limb_direction_not_reversed_pass": p01 >= 0.0,
    }


def _retime_vectors(
    values: np.ndarray, key_times: np.ndarray, output_times: np.ndarray
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    key_times = np.asarray(key_times, dtype=np.float64)
    output_times = np.asarray(output_times, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) != len(key_times):
        raise ValueError("Retimed source vectors must have shape [key_times, 3]")
    clipped_times = np.clip(output_times, key_times[0], key_times[-1])
    result = np.column_stack(
        [np.interp(clipped_times, key_times, values[:, axis]) for axis in range(3)]
    )
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("Retimed source limb direction became degenerate")
    return result / norms


def independent_native_limb_metrics(
    model,
    mujoco,
    trajectory: np.ndarray,
    source_positions: np.ndarray,
    source_world_to_robot_world: np.ndarray,
    key_times: np.ndarray,
    output_times: np.ndarray,
) -> dict[str, object]:
    """Compare robot FK directly with native-parser source geometry."""
    source_positions = np.asarray(source_positions, dtype=np.float64)
    alignment = np.asarray(source_world_to_robot_world, dtype=np.float64)
    if source_positions.shape[1:] != (len(CANONICAL_BODY_ORDER), 3):
        raise ValueError("Native source positions have the wrong canonical shape")
    source_index = {name: index for index, name in enumerate(CANONICAL_BODY_ORDER)}
    desired: dict[str, np.ndarray] = {}
    for side, key_prefix in (("Left", "left"), ("Right", "right")):
        shoulder = source_positions[:, source_index[f"{side}Shoulder"]]
        elbow = source_positions[:, source_index[f"{side}Elbow"]]
        wrist = source_positions[:, source_index[f"{side}Wrist"]]
        upper = (elbow - shoulder) @ alignment.T
        fore = (wrist - elbow) @ alignment.T
        upper /= np.linalg.norm(upper, axis=1, keepdims=True)
        fore /= np.linalg.norm(fore, axis=1, keepdims=True)
        desired[f"{key_prefix}_upper"] = _retime_vectors(
            upper, key_times, output_times
        )
        desired[f"{key_prefix}_fore"] = _retime_vectors(fore, key_times, output_times)

    data = mujoco.MjData(model)
    joint_addresses = []
    for name in JOINT_ORDER_18D:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        joint_addresses.append(int(model.jnt_qposadr[joint_id]))
    achieved: dict[str, list[np.ndarray]] = {key: [] for key in desired}

    def position(name: str) -> np.ndarray:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        return data.xpos[body_id].copy()

    for row in np.asarray(trajectory, dtype=np.float64):
        data.qpos[:] = 0.0
        data.qpos[joint_addresses] = row
        mujoco.mj_forward(model, data)
        for side, prefix, key_prefix in (
            ("Left", "l", "left"),
            ("Right", "r", "right"),
        ):
            shoulder = position(f"link_{prefix}ShoulderPitch")
            elbow = position(f"link_{prefix}Elbow")
            wrist = position(f"link_{prefix}WristPitch")
            achieved[f"{key_prefix}_upper"].append(_normalized(elbow - shoulder))
            achieved[f"{key_prefix}_fore"].append(_normalized(wrist - elbow))

    cosines = {
        key: np.sum(desired[key] * np.asarray(achieved[key]), axis=1)
        for key in desired
    }
    flattened = np.concatenate(list(cosines.values()))
    p01 = float(np.percentile(flattened, 1))
    return {
        "independent_native_limb_direction_cosine": {
            key: {
                "minimum": float(values.min()),
                "p01": float(np.percentile(values, 1)),
                "p05": float(np.percentile(values, 5)),
                "mean": float(values.mean()),
            }
            for key, values in cosines.items()
        },
        "independent_native_limb_direction_cosine_all_p01": p01,
        "independent_native_limb_direction_pass": p01 >= 0.80,
        "independent_native_limb_gate_threshold_p01": 0.80,
    }


def _retime_rotations(
    rotations: np.ndarray, key_times: np.ndarray, output_times: np.ndarray
) -> np.ndarray:
    rotations = np.asarray(rotations, dtype=np.float64)
    key_times = np.asarray(key_times, dtype=np.float64)
    output_times = np.asarray(output_times, dtype=np.float64)
    if rotations.shape != (len(key_times), 3, 3):
        raise ValueError("Retimed source rotations must have shape [key_times, 3, 3]")
    clipped_times = np.clip(output_times, key_times[0], key_times[-1])
    if len(rotations) == 1:
        return np.broadcast_to(rotations, (len(output_times), 3, 3)).copy()
    return Slerp(key_times, Rotation.from_matrix(rotations))(clipped_times).as_matrix()


def independent_head_fk_metrics(
    model,
    mujoco,
    trajectory: np.ndarray,
    desired_head_relative_rotations: np.ndarray,
) -> dict[str, object]:
    """Validate the real URDF head chain against the anatomical-parent target."""
    trajectory = np.asarray(trajectory, dtype=np.float64)
    desired = np.asarray(desired_head_relative_rotations, dtype=np.float64)
    if desired.shape != (len(trajectory), 3, 3):
        raise ValueError("Desired head rotations do not match the robot trajectory")
    data = mujoco.MjData(model)
    joint_addresses = []
    for name in JOINT_ORDER_18D:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        joint_addresses.append(int(model.jnt_qposadr[joint_id]))
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_torso")
    head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head_yaw_Link")
    if torso_id < 0 or head_id < 0:
        raise ValueError("Real ULA V2 URDF is missing torso or head body")
    achieved = []
    for row in trajectory:
        data.qpos[:] = 0.0
        data.qpos[joint_addresses] = row
        mujoco.mj_forward(model, data)
        torso = data.xmat[torso_id].reshape(3, 3).copy()
        head = data.xmat[head_id].reshape(3, 3).copy()
        achieved.append(torso.T @ head)
    achieved = np.asarray(achieved)
    errors = Rotation.from_matrix(np.swapaxes(achieved, 1, 2) @ desired).magnitude()
    p95_deg = float(np.rad2deg(np.percentile(errors, 95)))
    return {
        "independent_head_fk_error_mean_deg": float(np.rad2deg(errors.mean())),
        "independent_head_fk_error_p95_deg": p95_deg,
        "independent_head_fk_error_max_deg": float(np.rad2deg(errors.max())),
        "independent_head_fk_direction_pass": p95_deg <= 5.0,
        "independent_head_fk_gate_p95_deg": 5.0,
        "head_parent_frame_policy": "per_frame_anatomical_torso_frame",
    }


def run(args: argparse.Namespace) -> dict:
    for path in (args.bvh, args.gmr_root, args.urdf, args.config):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    if args.start_frame < 0 or args.end_frame <= args.start_frame:
        raise ValueError("Source interval must be a non-empty half-open frame interval")
    if args.end_frame - args.start_frame < 3:
        raise ValueError("InterAct retarget smoke requires at least three source frames")
    if args.warmup_source_frames < 0:
        raise ValueError("warmup-source-frames cannot be negative")
    if args.max_velocity <= 0:
        raise ValueError("max-velocity must be positive")

    started = time.perf_counter()
    source_hash = sha256(args.bvh)
    if args.expected_source_sha256 and source_hash != args.expected_source_sha256:
        raise ValueError("InterAct source SHA256 does not match actor episode task")
    solve_start = max(0, args.start_frame - args.warmup_source_frames)
    source = load_interact_bvh_native_v2(
        args.bvh,
        start_frame=solve_start,
        end_frame=args.end_frame,
    )
    frame_time = float(source["frame_time_sec"])
    if not is_30hz(frame_time):
        raise ValueError(f"InterAct source is not 30 Hz: frame_time={frame_time}")
    output_offset = args.start_frame - solve_start
    positions = np.asarray(source["canonical_positions_stacked"], dtype=np.float64)
    quaternions = np.asarray(
        source["canonical_quaternions_stacked_wxyz"], dtype=np.float64
    )
    head_relative = np.asarray(source["head_relative_rotations"], dtype=np.float64)
    anatomical_frames = np.asarray(
        source["anatomical_frame_rotations"], dtype=np.float64
    )
    raw_head_all = decompose_v2_head_rotations(head_relative)

    retargeter, mujoco, mink = configure_retargeter(
        args.gmr_root, args.urdf, args.config, args.solver
    )
    if args.elbow_branch == "motionx_negative":
        enforce_human_elbow_branch(retargeter, mujoco, mink)
    neutral_qpos, target_for_frame, episode_frame_alignment = (
        prepare_interact_target_builder(
            retargeter.model,
            mujoco,
            positions,
            quaternions,
            anatomical_frames,
            reference_frame_index=output_offset,
        )
    )
    retargeter.configuration.update(neutral_qpos)
    if args.posture_cost > 0:
        posture_task = mink.PostureTask(
            retargeter.model, cost=args.posture_cost, lm_damping=1.0
        )
        posture_task.set_target(neutral_qpos)
        retargeter.tasks1.append(posture_task)

    first_target = target_for_frame(0)
    for _ in range(20):
        retargeter.retarget(first_target)
    raw_body_rows = []
    output_targets = []
    ik_errors = []
    for local_frame in range(len(positions)):
        target = target_for_frame(local_frame)
        qpos = retargeter.retarget(target)
        if local_frame >= output_offset:
            raw_body_rows.append(
                extract_joint_row(retargeter.model, qpos, mujoco)
            )
            output_targets.append(target)
            ik_errors.append(float(retargeter.error1()))

    raw_body = np.asarray(raw_body_rows, dtype=np.float64)
    raw_head = raw_head_all[output_offset:]
    raw = np.column_stack((raw_body, raw_head))
    safe, key_times, output_times = smooth_and_limit(
        raw,
        OUTPUT_FPS,
        args.max_velocity,
        args.smoothing_window,
        joint_order=JOINT_ORDER_18D,
        joint_limits=JOINT_LIMITS_18D,
    )
    if args.elbow_branch == "motionx_negative":
        safe = enforce_safe_elbow_branch(safe, joint_order=JOINT_ORDER_18D)
    safe_targets = retime_targets(output_targets, key_times, output_times)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(args.episode_task_id)
    raw_csv = output_dir / f"{stem}_raw_18d.csv"
    safe_csv = output_dir / f"{stem}_safe_18d.csv"
    quality_json = output_dir / f"{stem}_quality.json"
    write_csv(raw_csv, raw, joint_order=JOINT_ORDER_18D)
    write_csv(safe_csv, safe, joint_order=JOINT_ORDER_18D)

    raw_pose_metrics = rendered_pose_metrics(
        retargeter.model,
        mujoco,
        raw,
        output_targets,
        joint_order=JOINT_ORDER_18D,
    )
    pose_metrics = rendered_pose_metrics(
        retargeter.model,
        mujoco,
        safe,
        safe_targets,
        joint_order=JOINT_ORDER_18D,
    )
    pose_metrics.update(
        {
            f"raw_{key}": value
            for key, value in raw_pose_metrics.items()
            if key.startswith("limb_target_error")
        }
    )
    source_frames = args.end_frame - args.start_frame
    metadata = {
        "schema_version": "2.0.0",
        "artifact_kind": "interact_actor_episode_ula_v2_18d_native_bvh_retarget_smoke",
        "episode_task_id": args.episode_task_id,
        "episode_task_record_sha256": args.episode_task_record_sha256,
        "interaction_partner_actor_id": args.partner_actor_id,
        "source_bvh": str(Path(args.bvh).resolve()),
        "source_sha256": source_hash,
        "source_dataset": "InterAct",
        "source_revision": INTERACT_REVISION,
        "source_license": INTERACT_LICENSE,
        "source_fps_observed": 1.0 / frame_time,
        "source_fps_contract": OUTPUT_FPS,
        "source_interval": {
            "start_frame": args.start_frame,
            "end_frame_exclusive": args.end_frame,
            "frame_count": source_frames,
            "sample_span_sec": (source_frames - 1) / OUTPUT_FPS,
            "sample_span_sec_role": "diagnostic_only_never_a_cut_or_admission_gate",
        },
        "solver_context": {
            "start_frame": solve_start,
            "end_frame_exclusive": args.end_frame,
            "warmup_source_frames_used": output_offset,
            "context_not_exported_as_episode": True,
            "episode_reference_frame_in_solver_context": output_offset,
            "warmup_used_as_episode_reference": False,
        },
        "robot_urdf": str(Path(args.urdf).resolve()),
        "output_contract": ULA_V2_18D_CONTRACT,
        "action_dim": len(JOINT_ORDER_18D),
        "joint_order": list(JOINT_ORDER_18D),
        "frames": int(len(safe)),
        "fps": OUTPUT_FPS,
        "output_sample_span_sec": (len(safe) - 1) / OUTPUT_FPS,
        "output_frame_coverage_sec": len(safe) / OUTPUT_FPS,
        "retimed": len(safe) != len(raw),
        "temporal_selection": {
            "interval_role": "core_axis_fit_preview_not_training_selection",
            "elapsed_time_cut_used": False,
            "duration_measurements_are_diagnostic_only": True,
            "training_interval_requires_blind_expression_completeness_review": True,
        },
        "max_velocity_rad_s": float(args.max_velocity),
        "posture_cost": float(args.posture_cost),
        "elbow_branch_policy": args.elbow_branch,
        "mean_final_ik_objective": float(np.mean(ik_errors)),
        "processing_sec": float(time.perf_counter() - started),
        "axis_policy": INTERACT_NATIVE_AXIS_POLICY,
        "axis_alignment_matrix": INTERACT_NATIVE_TO_ROBOT_BASIS.tolist(),
        "axis_alignment_determinant": float(
            np.linalg.det(INTERACT_NATIVE_TO_ROBOT_BASIS)
        ),
        "bvh_rotation_composition": "declared_channel_order_intrinsic",
        "legacy_gmr_euler_component_reorder_used": False,
        "head_parent_frame_policy": source["head_parent_frame_policy"],
        "warmup_reference_policy": "warmup_initializes_ik_only_episode_onset_defines_alignment",
        "episode_frame_alignment": episode_frame_alignment,
        "source_anatomical_axes_in_robot_coordinates": {
            "right": (
                INTERACT_NATIVE_TO_ROBOT_BASIS @ INTERACT_BVH_RIGHT_AXIS
            ).tolist(),
            "forward": (
                INTERACT_NATIVE_TO_ROBOT_BASIS @ INTERACT_BVH_FORWARD_AXIS
            ).tolist(),
            "up": (
                INTERACT_NATIVE_TO_ROBOT_BASIS @ INTERACT_BVH_UP_AXIS
            ).tolist(),
        },
        "mapped_source_joints": [
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
        ],
        "ignored_modalities": ["audio", "face", "finger"],
        "outputs": {
            "raw_csv": str(raw_csv),
            "safe_csv": str(safe_csv),
            "quality_json": str(quality_json),
        },
    }
    report = quality_report(
        raw,
        safe,
        OUTPUT_FPS,
        args.max_velocity,
        pose_metrics,
        metadata,
        joint_order=JOINT_ORDER_18D,
        joint_limits=JOINT_LIMITS_18D,
    )
    direction_metrics = limb_direction_metrics(
        retargeter.model, mujoco, safe, safe_targets
    )
    report.update(direction_metrics)
    report["quality_gate"]["limb_direction_not_reversed_pass"] = direction_metrics[
        "limb_direction_not_reversed_pass"
    ]
    independent_limb = independent_native_limb_metrics(
        retargeter.model,
        mujoco,
        safe,
        positions[output_offset:],
        np.asarray(
            episode_frame_alignment["source_world_to_robot_world_matrix"],
            dtype=np.float64,
        ),
        key_times,
        output_times,
    )
    report.update(independent_limb)
    report["quality_gate"]["independent_native_limb_direction_pass"] = (
        independent_limb["independent_native_limb_direction_pass"]
    )
    elbow_indices = [
        JOINT_ORDER_18D.index("joint_lElbow"),
        JOINT_ORDER_18D.index("joint_rElbow"),
    ]
    elbows = safe[:, elbow_indices]
    report.update(
        {
            "positive_elbow_branch_values": int(np.sum(elbows > 1e-6)),
            "negative_elbow_branch_values": int(np.sum(elbows < -1e-6)),
            "elbow_branch_observation_only": args.elbow_branch == "unconstrained",
            "axis_policy": INTERACT_NATIVE_AXIS_POLICY,
        }
    )
    head_metrics = head_quality_metrics(
        head_relative[output_offset:],
        raw,
        safe,
        OUTPUT_FPS,
        args.max_velocity,
        joint_order=JOINT_ORDER_18D,
    )
    report.update(head_metrics)
    for key in (
        "head_joint_limits_pass",
        "head_velocity_pass",
        "head_direction_pass",
        "head_continuity_pass",
    ):
        report["quality_gate"][key] = head_metrics[key]
    desired_safe_head = _retime_rotations(
        head_relative[output_offset:], key_times, output_times
    )
    independent_head = independent_head_fk_metrics(
        retargeter.model, mujoco, safe, desired_safe_head
    )
    report.update(independent_head)
    report["quality_gate"]["independent_head_fk_direction_pass"] = independent_head[
        "independent_head_fk_direction_pass"
    ]
    report["quality_gate"]["passed"] = all(
        value for key, value in report["quality_gate"].items() if key != "passed"
    )
    report["axis_visual_qc"] = {
        "status": "pending_mujoco_blind_direction_review",
        "passed": False,
    }
    report["license_gate"] = {
        "noncommercial_use_confirmation_status": "pending",
        "training_authorized": False,
    }
    report["admission_gate"] = {
        "automated_physical_qc_passed": report["quality_gate"]["passed"],
        "axis_visual_qc_passed": False,
        "semantic_review_passed": False,
        "emotion_review_passed": False,
        "license_training_use_confirmed": False,
        "passed": False,
    }
    report["processing_scope"] = args.processing_scope
    report["accepted_for_training"] = False
    quality_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    report = run(parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
