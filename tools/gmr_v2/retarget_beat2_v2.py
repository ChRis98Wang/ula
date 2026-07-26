#!/usr/bin/env python3
"""Retarget a BEAT2 SMPL-X/FLAME NPZ clip or window to ULA V2 18D."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:  # Support package imports and direct command-line execution.
    from .retarget_motionx322_v2 import (
        DEFAULT_CONFIG,
        DEFAULT_GMR_ROOT,
        DEFAULT_MODEL,
        DEFAULT_URDF,
        EXPECTED_MODEL_SHA256,
        JOINT_LIMITS_18D,
        JOINT_ORDER_18D,
        ULA_V2_18D_CONTRACT,
        axis_direction_metrics,
        canonical_head_relative_rotations,
        canonical_source_data,
        configure_retargeter,
        decompose_v2_head_rotations,
        enforce_human_elbow_branch,
        enforce_safe_elbow_branch,
        extract_joint_row,
        forward_smplx,
        head_quality_metrics,
        prepare_target_builder,
        quality_report,
        rendered_pose_metrics,
        retime_targets,
        sha256,
        smooth_and_limit,
        write_csv,
    )
except ImportError:  # pragma: no cover - exercised by direct script execution
    from retarget_motionx322_v2 import (
        DEFAULT_CONFIG,
        DEFAULT_GMR_ROOT,
        DEFAULT_MODEL,
        DEFAULT_URDF,
        EXPECTED_MODEL_SHA256,
        JOINT_LIMITS_18D,
        JOINT_ORDER_18D,
        ULA_V2_18D_CONTRACT,
        axis_direction_metrics,
        canonical_head_relative_rotations,
        canonical_source_data,
        configure_retargeter,
        decompose_v2_head_rotations,
        enforce_human_elbow_branch,
        enforce_safe_elbow_branch,
        extract_joint_row,
        forward_smplx,
        head_quality_metrics,
        prepare_target_builder,
        quality_report,
        rendered_pose_metrics,
        retime_targets,
        sha256,
        smooth_and_limit,
        write_csv,
    )


BEAT2_DATASET_ID = "beat2_chinese_v2"
BEAT2_DATASET_NAME = "BEAT2 Chinese v2.0.0"
BEAT2_SOURCE_REVISION = "8689ecb43513ba31964fd60e0ca69be02d3b0872"
BEAT2_SOURCE_URL = "https://huggingface.co/datasets/H-Liu1997/BEAT2"
BEAT2_AXIS_POLICY = "beat2_smplx_anatomical_right_up_with_v2_front_reflection_v1"
BEAT2_POSE_FEATURE_DIM = 165
BEAT2_BODY_END = 66
BEAT2_BETA_COUNT = 10


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--beat2", type=Path, required=True, help="BEAT2 smplxflame_30 NPZ"
    )
    parser.add_argument("--start-frame", type=int, default=0, help="Inclusive source frame")
    parser.add_argument("--end-frame", type=int, help="Exclusive source frame")
    parser.add_argument("--smplx-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gmr-root", type=Path, default=DEFAULT_GMR_ROOT)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fps",
        type=float,
        help="Optional assertion; must equal the NPZ mocap_frame_rate",
    )
    parser.add_argument("--warmup-frames", type=int, default=0)
    parser.add_argument("--max-velocity", type=float, default=3.0)
    parser.add_argument("--smoothing-window", type=int, default=7)
    parser.add_argument("--posture-cost", type=float, default=0.02)
    parser.add_argument("--solver", default="daqp")
    parser.add_argument(
        "--output-contract",
        choices=(ULA_V2_18D_CONTRACT,),
        default=ULA_V2_18D_CONTRACT,
        help="BEAT2 adapter emits the append-only ULA V2 18D contract",
    )
    parser.add_argument("--skip-model-sha-check", action="store_true")
    return parser.parse_args(argv)


def _scalar(npz, key, *, required=False):
    if key not in npz.files:
        if required:
            raise ValueError(f"BEAT2 NPZ is missing required field: {key}")
        return None
    value = np.asarray(npz[key])
    if value.ndim != 0:
        raise ValueError(f"BEAT2 field {key} must be scalar, got {value.shape}")
    return value.item()


def decode_beat2_smplx(path, *, start_frame=0, end_frame=None):
    """Decode only global/body pose, translation, and fixed clip-level shape."""
    with np.load(path, allow_pickle=False) as source:
        missing = sorted(
            {"poses", "trans", "betas", "mocap_frame_rate"} - set(source.files)
        )
        if missing:
            raise ValueError(f"BEAT2 NPZ is missing required fields: {missing}")

        poses = np.asarray(source["poses"])
        trans = np.asarray(source["trans"])
        betas = np.asarray(source["betas"])
        if poses.ndim != 2 or poses.shape[1] != BEAT2_POSE_FEATURE_DIM:
            raise ValueError(
                f"Expected BEAT2 poses [frames, {BEAT2_POSE_FEATURE_DIM}], got {poses.shape}"
            )
        total_frames = int(poses.shape[0])
        if trans.shape != (total_frames, 3):
            raise ValueError(f"Expected BEAT2 trans [{total_frames}, 3], got {trans.shape}")
        if betas.ndim != 1 or betas.shape[0] < BEAT2_BETA_COUNT:
            raise ValueError(
                f"Expected clip-level BEAT2 betas [{BEAT2_BETA_COUNT}+], got {betas.shape}"
            )

        if isinstance(start_frame, bool) or not isinstance(start_frame, (int, np.integer)):
            raise ValueError("start-frame must be an integer")
        if end_frame is not None and (
            isinstance(end_frame, bool) or not isinstance(end_frame, (int, np.integer))
        ):
            raise ValueError("end-frame must be an integer")
        start = int(start_frame)
        end = total_frames if end_frame is None else int(end_frame)
        if start < 0 or end > total_frames or end <= start:
            raise ValueError(
                f"Invalid BEAT2 frame window [{start}, {end}) for {total_frames} frames"
            )
        if end - start < 3:
            raise ValueError("BEAT2 frame window must contain at least three frames")

        used_pose = poses[start:end, :BEAT2_BODY_END]
        used_trans = trans[start:end]
        used_betas = betas[:BEAT2_BETA_COUNT]
        if not np.isfinite(used_pose).all():
            raise ValueError("BEAT2 global/body pose window contains non-finite values")
        if not np.isfinite(used_trans).all():
            raise ValueError("BEAT2 translation window contains non-finite values")
        if not np.isfinite(used_betas).all():
            raise ValueError("BEAT2 first 10 betas contain non-finite values")

        mocap_frame_rate = float(_scalar(source, "mocap_frame_rate", required=True))
        if not np.isfinite(mocap_frame_rate) or mocap_frame_rate <= 0:
            raise ValueError("BEAT2 mocap_frame_rate must be finite and positive")
        source_model = _scalar(source, "model")
        source_gender = _scalar(source, "gender")
        source_keys = sorted(source.files)

    return {
        "root_orient": used_pose[:, 0:3].astype(np.float32, copy=True),
        "pose_body": used_pose[:, 3:66].astype(np.float32, copy=True),
        "trans": used_trans.astype(np.float32, copy=True),
        "betas": used_betas.astype(np.float32, copy=True)[None, :],
        "frame_count": int(end - start),
        "source_total_frames": total_frames,
        "source_start_frame": start,
        "source_end_frame": end,
        "source_pose_dim": int(poses.shape[1]),
        "source_beta_dim": int(betas.shape[0]),
        "source_keys": source_keys,
        "source_model": None if source_model is None else str(source_model),
        "source_gender": None if source_gender is None else str(source_gender),
        "mocap_frame_rate": mocap_frame_rate,
    }


def build_beat2_provenance(source_path, decoded, source_hash):
    beta_hash = hashlib.sha256(
        np.ascontiguousarray(decoded["betas"], dtype=np.float32).tobytes()
    ).hexdigest()
    return {
        "source_beat2_npz": str(Path(source_path).resolve()),
        "source_sha256": source_hash,
        "source_dataset": BEAT2_DATASET_NAME,
        "source_dataset_id": BEAT2_DATASET_ID,
        "source_url": BEAT2_SOURCE_URL,
        "source_revision": BEAT2_SOURCE_REVISION,
        "source_format": "BEAT2 SMPL-X/FLAME NPZ",
        "source_fps": float(decoded["mocap_frame_rate"]),
        "source_total_frames": int(decoded["source_total_frames"]),
        "source_frames": int(decoded["frame_count"]),
        "source_window_start_frame": int(decoded["source_start_frame"]),
        "source_window_end_frame_exclusive": int(decoded["source_end_frame"]),
        "source_window_frames": int(decoded["frame_count"]),
        "source_window_convention": "zero_based_half_open_[start,end)",
        "source_feature_dim": int(decoded["source_pose_dim"]),
        "source_pose_feature_dim": int(decoded["source_pose_dim"]),
        "source_beta_feature_dim": int(decoded["source_beta_dim"]),
        "source_container_keys": list(decoded["source_keys"]),
        "source_model_label": decoded["source_model"],
        "source_gender_label": decoded["source_gender"],
        "pose_decode": {
            "global_orient": "poses[:, 0:3]",
            "body_pose": "poses[:, 3:66]",
            "ignored_pose": "poses[:, 66:165]",
        },
        "betas_policy": "first_10_clip_level_coefficients_fixed_across_all_frames",
        "selected_betas_sha256": beta_hash,
    }


def output_stem(source_path, decoded):
    return (
        f"{Path(source_path).stem}"
        f"_f{decoded['source_start_frame']:06d}-{decoded['source_end_frame']:06d}"
    )


def run(args):
    for path in (args.beat2, args.smplx_model, args.gmr_root, args.urdf, args.config):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.max_velocity <= 0:
        raise ValueError("max-velocity must be positive")
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

    source_hash = sha256(args.beat2)
    decoded = decode_beat2_smplx(
        args.beat2,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    fps = float(decoded["mocap_frame_rate"])
    if args.fps is not None and not np.isclose(args.fps, fps, rtol=0.0, atol=1e-6):
        raise ValueError(
            f"--fps {args.fps} does not match BEAT2 mocap_frame_rate {fps}; "
            "this adapter does not resample source frames"
        )

    joints, local_rotvecs, parents = forward_smplx(decoded, args.smplx_model)
    positions, quaternions, alignment = canonical_source_data(
        joints, local_rotvecs, parents
    )
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
    raw = np.column_stack((raw_body, raw_head))
    safe, retime_key_times, retime_output_times = smooth_and_limit(
        raw,
        fps,
        args.max_velocity,
        args.smoothing_window,
        joint_order=JOINT_ORDER_18D,
        joint_limits=JOINT_LIMITS_18D,
    )
    safe = enforce_safe_elbow_branch(safe, joint_order=JOINT_ORDER_18D)
    safe_targets = retime_targets(output_targets, retime_key_times, retime_output_times)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(args.beat2, decoded)
    raw_csv = args.output_dir / f"{stem}_gmr_raw_18d.csv"
    safe_csv = args.output_dir / f"{stem}_gmr_safe_18d.csv"
    quality_json = args.output_dir / "quality.json"
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

    metadata = build_beat2_provenance(args.beat2, decoded, source_hash)
    metadata.update(
        {
            "smplx_model": str(args.smplx_model.resolve()),
            "smplx_model_sha256": model_hash,
            "smplx_model_sha256_status": model_hash_status,
            "smplx_model_revision": "a57d1dfb1162c2a9cc20013f0ab212c21f211e78",
            "robot_urdf": str(args.urdf.resolve()),
            "frames": int(len(safe)),
            "fps": fps,
            "duration_sec": float(len(safe) / fps),
            "source_window_duration_sec": float(decoded["frame_count"] / fps),
            "retime_factor": float(len(safe) / len(raw)),
            "max_velocity_rad_s": float(args.max_velocity),
            "posture_cost": float(args.posture_cost),
            "mean_final_ik_objective": float(np.mean(ik_errors)),
            "anatomical_alignment_matrix": alignment.tolist(),
            "anatomical_alignment_determinant": float(np.linalg.det(alignment)),
            "axis_policy": BEAT2_AXIS_POLICY,
            "output_contract": args.output_contract,
            "action_dim": len(JOINT_ORDER_18D),
            "joint_order": list(JOINT_ORDER_18D),
            "mapped_channels": [
                "poses[:,0:3] global_orientation",
                "poses[:,3:66] body_pose_spine3_shoulders_elbows_wrists_neck_head",
                "trans[:,0:3] root_translation_for_source_joint_positions",
                "betas[:10] fixed_clip_shape",
            ],
            "ignored_channels": [
                "poses[:,66:165] jaw_eyes_hands",
                "expressions",
                "betas[10:]",
                "face_shape",
                "fingers",
            ],
            "processing_sec": float(time.perf_counter() - started),
            "outputs": {
                "raw_csv": str(raw_csv.resolve()),
                "safe_csv": str(safe_csv.resolve()),
            },
        }
    )
    report = quality_report(
        raw,
        safe,
        fps,
        args.max_velocity,
        pose_metrics,
        metadata,
        joint_order=JOINT_ORDER_18D,
        joint_limits=JOINT_LIMITS_18D,
    )
    direction_metrics = axis_direction_metrics(
        safe, alignment, joint_order=JOINT_ORDER_18D
    )
    direction_metrics["axis_policy"] = BEAT2_AXIS_POLICY
    report.update(direction_metrics)
    report["quality_gate"]["axis_direction_pass"] = direction_metrics[
        "axis_direction_pass"
    ]
    head_metrics = head_quality_metrics(
        head_relative_rotations,
        raw,
        safe,
        fps,
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
    report["quality_gate"]["passed"] = all(
        value
        for key, value in report["quality_gate"].items()
        if key != "passed"
    )
    quality_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
