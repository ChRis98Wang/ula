#!/usr/bin/env python3
"""Retarget Motion-X++ SMPL-X 322D into versioned ULA V2 contracts."""

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from smplx import SMPLX
from smplx.joint_names import JOINT_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from upper_body_skeleton.retarget_v2_18d import (  # noqa: E402
    CONTRACT_VERSION as ULA_V2_18D_CONTRACT,
    HEAD_JOINT_AXES,
    HEAD_JOINT_LIMITS,
    HEAD_JOINT_ORDER,
    HEAD_JOINT_VELOCITY_LIMITS,
    JOINT_LIMITS_18D,
    JOINT_ORDER_18D,
)

try:  # Support both direct execution and package imports in focused tests.
    from .retarget_xsens_v2 import (
        JOINT_LIMITS,
        JOINT_ORDER,
        SOURCE_BODIES,
        configure_retargeter,
        extract_joint_row,
        prepare_target_builder,
        quality_report,
        rendered_pose_metrics,
        retime_targets,
        smooth_and_limit,
        write_csv,
    )
except ImportError:  # pragma: no cover - exercised by the command-line script
    from retarget_xsens_v2 import (
        JOINT_LIMITS,
        JOINT_ORDER,
        SOURCE_BODIES,
        configure_retargeter,
        extract_joint_row,
        prepare_target_builder,
        quality_report,
        rendered_pose_metrics,
        retime_targets,
        smooth_and_limit,
        write_csv,
    )


DEFAULT_GMR_ROOT = WORKSPACE_ROOT / "GMR"
DEFAULT_URDF = PROJECT_ROOT / "urdf_V2_20260514/urdf/xacro/robot_modify_meshdir.urdf"
DEFAULT_CONFIG = Path(__file__).with_name("xsens_to_ula_v2.json")
DEFAULT_MODEL = Path(
    "/home/gez/nas/cloud/gez/human_motion/models/smplx2020/"
    "smplx_models/smplx/SMPLX_NEUTRAL_2020.npz"
)
EXPECTED_MODEL_SHA256 = "bdf06146e27d92022fe5dadad3b9203373f6879eca8e4d8235359ee3ec6a5a74"
AXIS_POLICY = "motionx_anatomical_right_up_with_v2_front_reflection_v2"
ULA_V2_15D_CONTRACT = "ula_v2_15d_v1"
ELBOW_JOINTS = ("joint_lElbow", "joint_rElbow")
HEAD_CONTINUITY_MAX_RAW_STEP_RAD = float(np.deg2rad(90.0))

if JOINT_ORDER_18D[: len(JOINT_ORDER)] != JOINT_ORDER:
    raise RuntimeError("ULA V2 18D contract must preserve the complete 15D prefix")

SMPLX_TO_CANONICAL = {
    "spine3": "Chest4",
    "left_shoulder": "LeftShoulder",
    "left_elbow": "LeftElbow",
    "left_wrist": "LeftWrist",
    "right_shoulder": "RightShoulder",
    "right_elbow": "RightElbow",
    "right_wrist": "RightWrist",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motionx", type=Path, required=True)
    parser.add_argument("--smplx-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gmr-root", type=Path, default=DEFAULT_GMR_ROOT)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--warmup-frames", type=int, default=0)
    parser.add_argument("--max-velocity", type=float, default=3.0)
    parser.add_argument("--smoothing-window", type=int, default=7)
    parser.add_argument("--posture-cost", type=float, default=0.02)
    parser.add_argument("--solver", default="daqp")
    parser.add_argument(
        "--output-contract",
        choices=(ULA_V2_15D_CONTRACT, ULA_V2_18D_CONTRACT),
        default=ULA_V2_15D_CONTRACT,
        help="Keep the legacy 15D output or append the three V2 head joints",
    )
    parser.add_argument("--skip-model-sha-check", action="store_true")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        raise ValueError("Cannot construct anatomical basis from a degenerate vector")
    return vector / norm


def decode_motionx322(path):
    motion = np.load(path)
    if motion.ndim != 2 or motion.shape[1] != 322:
        raise ValueError(f"Expected Motion-X++ [frames, 322], got {motion.shape}")
    if len(motion) < 3:
        raise ValueError("Motion-X++ clip must contain at least three frames")
    if not np.isfinite(motion).all():
        raise ValueError("Motion-X++ clip contains non-finite values")
    return {
        "root_orient": motion[:, 0:3].astype(np.float32),
        "pose_body": motion[:, 3:66].astype(np.float32),
        "trans": motion[:, 309:312].astype(np.float32),
        "betas": np.median(motion[:, 312:322], axis=0, keepdims=True).astype(np.float32),
        "frame_count": int(len(motion)),
    }


def forward_smplx(decoded, model_path):
    model = SMPLX(str(model_path), gender="neutral", use_pca=False)
    frame_count = decoded["frame_count"]
    zeros_hand = torch.zeros((frame_count, 45), dtype=torch.float32)
    zeros_face = torch.zeros((frame_count, 3), dtype=torch.float32)
    zeros_expression = torch.zeros(
        (frame_count, model.num_expression_coeffs), dtype=torch.float32
    )
    betas = torch.from_numpy(decoded["betas"]).expand(frame_count, -1)
    with torch.inference_mode():
        output = model(
            betas=betas,
            global_orient=torch.from_numpy(decoded["root_orient"]),
            body_pose=torch.from_numpy(decoded["pose_body"]),
            transl=torch.from_numpy(decoded["trans"]),
            left_hand_pose=zeros_hand,
            right_hand_pose=zeros_hand,
            jaw_pose=zeros_face,
            leye_pose=zeros_face,
            reye_pose=zeros_face,
            expression=zeros_expression,
            return_full_pose=True,
        )
    joints = output.joints[:, : len(model.parents)].detach().cpu().numpy()
    full_pose = output.full_pose.detach().cpu().numpy().reshape(frame_count, -1, 3)
    return joints, full_pose[:, : len(model.parents)], model.parents.detach().cpu().numpy()


def global_joint_rotations(local_rotvecs, parents):
    frame_count, joint_count, _ = local_rotvecs.shape
    local = Rotation.from_rotvec(local_rotvecs.reshape(-1, 3)).as_matrix()
    local = local.reshape(frame_count, joint_count, 3, 3)
    global_rotations = np.empty_like(local)
    for joint_index, parent_index in enumerate(parents):
        if parent_index < 0:
            global_rotations[:, joint_index] = local[:, joint_index]
        else:
            global_rotations[:, joint_index] = (
                global_rotations[:, parent_index] @ local[:, joint_index]
            )
    return global_rotations


def canonical_head_relative_rotations(local_rotvecs, parents, alignment):
    """Return the reflected spine3-to-head rotation through neck and head."""
    local_rotvecs = np.asarray(local_rotvecs, dtype=np.float64)
    parents = np.asarray(parents, dtype=np.int64)
    alignment = np.asarray(alignment, dtype=np.float64)
    if local_rotvecs.ndim != 3 or local_rotvecs.shape[2] != 3:
        raise ValueError("SMPL-X local rotations must have shape [frames, joints, 3]")
    if parents.shape != (local_rotvecs.shape[1],):
        raise ValueError("SMPL-X parent array does not match local rotations")
    if alignment.shape != (3, 3) or not np.allclose(
        alignment @ alignment.T, np.eye(3), atol=1e-6
    ):
        raise ValueError("Motion-X anatomical alignment must be an orthogonal 3x3 matrix")

    names = JOINT_NAMES[: local_rotvecs.shape[1]]
    indices = {name: index for index, name in enumerate(names)}
    missing = [name for name in ("spine3", "neck", "head") if name not in indices]
    if missing:
        raise ValueError(f"SMPL-X model is missing head-chain joints: {missing}")
    spine3 = indices["spine3"]
    neck = indices["neck"]
    head = indices["head"]
    if int(parents[neck]) != spine3 or int(parents[head]) != neck:
        raise ValueError("Expected SMPL-X head chain spine3 -> neck -> head")

    global_rotations = global_joint_rotations(local_rotvecs, parents)
    relative = np.swapaxes(global_rotations[:, spine3], 1, 2) @ global_rotations[:, head]
    return alignment @ relative @ alignment.T


def decompose_v2_head_rotations(relative_rotations):
    """Decompose as Rx(roll) Ry(pitch) Rz(-yaw), matching the V2 URDF axes."""
    relative_rotations = np.asarray(relative_rotations, dtype=np.float64)
    if relative_rotations.ndim != 3 or relative_rotations.shape[1:] != (3, 3):
        raise ValueError("Head rotations must have shape [frames, 3, 3]")
    if len(relative_rotations) < 1 or not np.isfinite(relative_rotations).all():
        raise ValueError("Head rotations must contain finite frames")
    determinants = np.linalg.det(relative_rotations)
    if not np.allclose(determinants, 1.0, atol=1e-5):
        raise ValueError("Head-relative matrices must be proper rotations")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        xyz = Rotation.from_matrix(relative_rotations).as_euler("XYZ")
    xyz = np.unwrap(xyz, axis=0)
    return np.column_stack((xyz[:, 0], xyz[:, 1], -xyz[:, 2]))


def reconstruct_v2_head_rotations(head_trajectory):
    head_trajectory = np.asarray(head_trajectory, dtype=np.float64)
    if head_trajectory.ndim != 2 or head_trajectory.shape[1] != len(HEAD_JOINT_ORDER):
        raise ValueError("V2 head trajectory must have shape [frames, 3]")
    xyz = np.column_stack(
        (head_trajectory[:, 0], head_trajectory[:, 1], -head_trajectory[:, 2])
    )
    return Rotation.from_euler("XYZ", xyz).as_matrix()


def head_quality_metrics(
    source_relative_rotations,
    raw,
    safe,
    fps,
    max_velocity,
    *,
    joint_order=JOINT_ORDER_18D,
):
    head_indices = [joint_order.index(name) for name in HEAD_JOINT_ORDER]
    raw_head = np.asarray(raw, dtype=np.float64)[:, head_indices]
    safe_head = np.asarray(safe, dtype=np.float64)[:, head_indices]
    reconstructed = reconstruct_v2_head_rotations(raw_head)
    error_rotations = np.swapaxes(reconstructed, 1, 2) @ np.asarray(
        source_relative_rotations, dtype=np.float64
    )
    roundtrip_error = Rotation.from_matrix(error_rotations).magnitude()

    lower = np.asarray([HEAD_JOINT_LIMITS[name][0] for name in HEAD_JOINT_ORDER])
    upper = np.asarray([HEAD_JOINT_LIMITS[name][1] for name in HEAD_JOINT_ORDER])
    safe_limit_violations = int(((safe_head < lower - 1e-8) | (safe_head > upper + 1e-8)).sum())
    raw_steps = np.abs(np.diff(raw_head, axis=0))
    safe_steps = np.abs(np.diff(safe_head, axis=0))
    source_steps = Rotation.from_matrix(
        np.swapaxes(source_relative_rotations[:-1], 1, 2)
        @ source_relative_rotations[1:]
    ).magnitude()
    safe_velocity = safe_steps * float(fps)
    physical_velocity_limit = min(HEAD_JOINT_VELOCITY_LIMITS.values())
    effective_velocity_limit = min(float(max_velocity), float(physical_velocity_limit))
    raw_max_step = float(raw_steps.max(initial=0.0))
    safe_max_step = float(safe_steps.max(initial=0.0))
    safe_max_velocity = float(safe_velocity.max(initial=0.0))
    direction_error_max = float(roundtrip_error.max(initial=0.0))

    return {
        "head_joint_order": list(HEAD_JOINT_ORDER),
        "head_joint_axes": {name: list(HEAD_JOINT_AXES[name]) for name in HEAD_JOINT_ORDER},
        "head_yaw_sign_convention": "qpos_yaw_is_negative_canonical_Z_euler",
        "head_direction_roundtrip_max_rad": direction_error_max,
        "head_direction_roundtrip_max_deg": float(np.rad2deg(direction_error_max)),
        "head_source_max_rotation_step_rad": float(source_steps.max(initial=0.0)),
        "head_raw_max_component_step_rad": raw_max_step,
        "head_safe_max_component_step_rad": safe_max_step,
        "head_safe_max_velocity_rad_s": safe_max_velocity,
        "head_effective_velocity_limit_rad_s": effective_velocity_limit,
        "head_physical_velocity_limit_rad_s": float(physical_velocity_limit),
        "head_safe_joint_limit_violations": safe_limit_violations,
        "head_joint_limits_pass": safe_limit_violations == 0,
        "head_velocity_pass": safe_max_velocity <= effective_velocity_limit + 1e-6,
        "head_direction_pass": direction_error_max <= 1e-5,
        "head_continuity_pass": bool(
            raw_max_step <= HEAD_CONTINUITY_MAX_RAW_STEP_RAD + 1e-6
            and safe_max_step <= effective_velocity_limit / float(fps) + 1e-6
        ),
    }


def anatomical_alignment(joints, name_to_index):
    left_shoulder = joints[:, name_to_index["left_shoulder"]]
    right_shoulder = joints[:, name_to_index["right_shoulder"]]
    spine = joints[:, name_to_index["spine3"]]
    pelvis = joints[:, name_to_index["pelvis"]]
    right = normalize(np.median(right_shoulder - left_shoulder, axis=0))
    up = np.median(spine - pelvis, axis=0)
    up = normalize(up - np.dot(up, right) * right)
    # Motion-X/SMPL-X and this V2 URDF use opposite front-axis chirality once
    # anatomical right and up are aligned.  The reflection is intentional: a
    # human elbow bend must move toward V2 x+ and select the negative elbow
    # branch documented by v2_axis_calibration.py.
    forward = -normalize(np.cross(right, up))
    source_basis = np.column_stack((forward, right, up))
    alignment = source_basis.T
    if not np.isclose(np.linalg.det(alignment), -1.0, atol=1e-4):
        raise ValueError("Anatomical alignment must contain the front-axis reflection")
    return alignment


def canonical_source_data(joints, local_rotvecs, parents):
    joint_names = JOINT_NAMES[: len(parents)]
    name_to_index = {name: index for index, name in enumerate(joint_names)}
    missing = sorted(set(SMPLX_TO_CANONICAL) - set(name_to_index))
    if missing:
        raise ValueError(f"SMPL-X model is missing joints: {missing}")
    alignment = anatomical_alignment(joints, name_to_index)
    global_rotations = global_joint_rotations(local_rotvecs, parents)

    positions = []
    rotations = []
    canonical_to_smplx = {
        canonical: smplx_name for smplx_name, canonical in SMPLX_TO_CANONICAL.items()
    }
    for canonical_name in SOURCE_BODIES:
        smplx_name = canonical_to_smplx[canonical_name]
        index = name_to_index[smplx_name]
        positions.append(joints[:, index] @ alignment.T)
        rotations.append(alignment @ global_rotations[:, index] @ alignment.T)
    positions = np.stack(positions, axis=1)
    rotation_matrices = np.stack(rotations, axis=1)
    quaternions = Rotation.from_matrix(rotation_matrices.reshape(-1, 3, 3)).as_quat(
        scalar_first=True
    )
    quaternions = quaternions.reshape(len(joints), len(SOURCE_BODIES), 4)
    return positions, quaternions, alignment


def enforce_human_elbow_branch(retargeter, mujoco, mink):
    """Restrict Motion-X retargeting to V2's calibrated human-flexion branch."""
    for joint_name in ELBOW_JOINTS:
        joint_id = mujoco.mj_name2id(
            retargeter.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        if joint_id < 0:
            raise ValueError(f"Joint not found in V2 model: {joint_name}")
        retargeter.model.jnt_range[joint_id, 1] = 0.0
    retargeter.ik_limits = [mink.ConfigurationLimit(retargeter.model)]


def enforce_safe_elbow_branch(trajectory, *, joint_order=JOINT_ORDER):
    safe = np.asarray(trajectory).copy()
    for joint_name in ELBOW_JOINTS:
        joint_index = joint_order.index(joint_name)
        safe[:, joint_index] = np.minimum(safe[:, joint_index], 0.0)
    return safe


def axis_direction_metrics(trajectory, alignment, *, joint_order=JOINT_ORDER):
    elbow_indices = [joint_order.index(name) for name in ELBOW_JOINTS]
    elbows = np.asarray(trajectory)[:, elbow_indices]
    positive = elbows > 1e-6
    determinant = float(np.linalg.det(alignment))
    return {
        "axis_policy": AXIS_POLICY,
        "alignment_determinant": determinant,
        "positive_elbow_branch_values": int(positive.sum()),
        "positive_elbow_branch_frames": int(np.any(positive, axis=1).sum()),
        "max_elbow_angle_rad": float(elbows.max()),
        "axis_direction_pass": bool(
            determinant < -0.999 and not np.any(positive)
        ),
    }


def main():
    args = parse_args()
    for path in (args.motionx, args.smplx_model, args.gmr_root, args.urdf, args.config):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.fps <= 0 or args.max_velocity <= 0:
        raise ValueError("fps and max-velocity must be positive")
    if args.warmup_frames < 0:
        raise ValueError("warmup-frames cannot be negative")

    started = time.perf_counter()
    model_hash = None
    model_hash_status = "skipped_not_computed"
    if not args.skip_model_sha_check:
        model_hash = sha256(args.smplx_model)
        model_hash_status = "verified"
        if model_hash != EXPECTED_MODEL_SHA256:
            raise ValueError(
                f"Unexpected SMPL-X model SHA256 {model_hash}; expected {EXPECTED_MODEL_SHA256}"
            )
    source_hash = sha256(args.motionx)
    decoded = decode_motionx322(args.motionx)
    joints, local_rotvecs, parents = forward_smplx(decoded, args.smplx_model)
    positions, quaternions, alignment = canonical_source_data(
        joints, local_rotvecs, parents
    )
    include_head = args.output_contract == ULA_V2_18D_CONTRACT
    joint_order = JOINT_ORDER_18D if include_head else JOINT_ORDER
    joint_limits = JOINT_LIMITS_18D if include_head else JOINT_LIMITS
    head_relative_rotations = None
    raw_head = None
    if include_head:
        head_relative_rotations = canonical_head_relative_rotations(
            local_rotvecs, parents, alignment
        )
        raw_head = decompose_v2_head_rotations(head_relative_rotations)

    retargeter, mujoco, mink = configure_retargeter(
        args.gmr_root, args.urdf, args.config, args.solver
    )
    enforce_human_elbow_branch(retargeter, mujoco, mink)
    neutral_qpos, target_for_frame = prepare_target_builder(
        retargeter.model, mujoco, positions, quaternions
    )
    retargeter.configuration.update(neutral_qpos)
    if args.posture_cost > 0:
        posture_task = mink.PostureTask(
            retargeter.model, cost=args.posture_cost, lm_damping=1.0
        )
        posture_task.set_target(neutral_qpos)
        retargeter.tasks1.append(posture_task)

    first_target = target_for_frame(0)
    for _ in range(20 + args.warmup_frames):
        retargeter.retarget(first_target)

    raw_rows = []
    output_targets = []
    ik_errors = []
    for frame_index in range(decoded["frame_count"]):
        target = target_for_frame(frame_index)
        qpos = retargeter.retarget(target)
        raw_rows.append(extract_joint_row(retargeter.model, qpos, mujoco))
        output_targets.append(target)
        ik_errors.append(float(retargeter.error1()))

    raw_body = np.asarray(raw_rows)
    raw = np.column_stack((raw_body, raw_head)) if include_head else raw_body
    safe, retime_key_times, retime_output_times = smooth_and_limit(
        raw,
        args.fps,
        args.max_velocity,
        args.smoothing_window,
        joint_order=joint_order,
        joint_limits=joint_limits,
    )
    safe = enforce_safe_elbow_branch(safe, joint_order=joint_order)
    safe_targets = retime_targets(output_targets, retime_key_times, retime_output_times)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.motionx.stem
    dimension_label = "18d" if include_head else "15d"
    raw_csv = args.output_dir / f"{stem}_gmr_raw_{dimension_label}.csv"
    safe_csv = args.output_dir / f"{stem}_gmr_safe_{dimension_label}.csv"
    quality_json = args.output_dir / "quality.json"
    write_csv(raw_csv, raw, joint_order=joint_order)
    write_csv(safe_csv, safe, joint_order=joint_order)

    raw_pose_metrics = rendered_pose_metrics(
        retargeter.model, mujoco, raw, output_targets, joint_order=joint_order
    )
    pose_metrics = rendered_pose_metrics(
        retargeter.model, mujoco, safe, safe_targets, joint_order=joint_order
    )
    pose_metrics.update(
        {
            f"raw_{key}": value
            for key, value in raw_pose_metrics.items()
            if key.startswith("limb_target_error")
        }
    )
    metadata = {
        "source_motionx": str(args.motionx.resolve()),
        "source_sha256": source_hash,
        "source_dataset": "Motion-X++/HAA500",
        "source_revision": "c74fa62247289ed31e407b6133d954d3c171db43",
        "source_fps": float(args.fps),
        "source_frames": int(decoded["frame_count"]),
        "source_feature_dim": 322,
        "smplx_model": str(args.smplx_model.resolve()),
        "smplx_model_sha256": model_hash,
        "smplx_model_sha256_status": model_hash_status,
        "smplx_model_revision": "a57d1dfb1162c2a9cc20013f0ab212c21f211e78",
        "robot_urdf": str(args.urdf.resolve()),
        "frames": int(len(safe)),
        "fps": float(args.fps),
        "duration_sec": float(len(safe) / args.fps),
        "source_window_frames": int(len(raw)),
        "source_window_duration_sec": float(len(raw) / args.fps),
        "retime_factor": float(len(safe) / len(raw)),
        "max_velocity_rad_s": float(args.max_velocity),
        "posture_cost": float(args.posture_cost),
        "mean_final_ik_objective": float(np.mean(ik_errors)),
        "anatomical_alignment_matrix": alignment.tolist(),
        "anatomical_alignment_determinant": float(np.linalg.det(alignment)),
        "axis_policy": AXIS_POLICY,
        "output_contract": args.output_contract,
        "action_dim": len(joint_order),
        "joint_order": list(joint_order),
        "mapped_channels": [
            "root_orientation",
            "body_pose_spine3",
            "shoulders",
            "elbows",
            "wrists",
        ]
        + (["body_pose_neck", "body_pose_head"] if include_head else []),
        "ignored_channels": [
            "hands",
            "jaw",
            "eyes",
            "face_expression",
            "face_shape",
        ]
        + ([] if include_head else ["head"]),
        "processing_sec": float(time.perf_counter() - started),
        "outputs": {
            "raw_csv": str(raw_csv.resolve()),
            "safe_csv": str(safe_csv.resolve()),
        },
    }
    report = quality_report(
        raw,
        safe,
        args.fps,
        args.max_velocity,
        pose_metrics,
        metadata,
        joint_order=joint_order,
        joint_limits=joint_limits,
    )
    direction_metrics = axis_direction_metrics(safe, alignment, joint_order=joint_order)
    report.update(direction_metrics)
    report["quality_gate"]["axis_direction_pass"] = direction_metrics[
        "axis_direction_pass"
    ]
    if include_head:
        head_metrics = head_quality_metrics(
            head_relative_rotations,
            raw,
            safe,
            args.fps,
            args.max_velocity,
            joint_order=joint_order,
        )
        report.update(head_metrics)
        for key in (
            "head_joint_limits_pass",
            "head_velocity_pass",
            "head_direction_pass",
            "head_continuity_pass",
        ):
            report["quality_gate"][key] = head_metrics[key]
    report["quality_gate"]["passed"] = all(
        value
        for key, value in report["quality_gate"].items()
        if key != "passed"
    )
    quality_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
