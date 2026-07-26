#!/usr/bin/env python3
"""Retarget an Xsens BVH clip to the ULA V2 15-DoF training contract."""

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_GMR_ROOT = WORKSPACE_ROOT / "GMR"
DEFAULT_BVH = DEFAULT_GMR_ROOT / "assets/xsens_bvh_test/251021_04_boxing_120Hz_cm_3DsMax.bvh"
DEFAULT_URDF = PROJECT_ROOT / "urdf_V2_20260514/urdf/xacro/robot_modify_meshdir.urdf"
DEFAULT_CONFIG = Path(__file__).with_name("xsens_to_ula_v2.json")
DEFAULT_OUTPUT = PROJECT_ROOT / "deliverables/gmr_v2_quality/boxing_xsens"

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
    "joint_lShoulderPitch": (-1.75, 1.75),
    "joint_lShoulderRoll": (-1.65, 1.65),
    "joint_lShoulderYaw": (-2.79, 2.79),
    "joint_lElbow": (-1.75, 1.57),
    "joint_lWristRoll": (-2.79, 2.79),
    "joint_lWristPitch": (-1.75, 1.57),
    "joint_rShoulderPitch": (-1.75, 1.75),
    "joint_rShoulderRoll": (-1.65, 1.65),
    "joint_rShoulderYaw": (-2.79, 2.79),
    "joint_rElbow": (-1.75, 1.57),
    "joint_rWristRoll": (-2.79, 2.79),
    "joint_rWristPitch": (-1.75, 1.57),
}

SOURCE_BODIES = [
    "Chest4",
    "LeftShoulder",
    "LeftElbow",
    "LeftWrist",
    "RightShoulder",
    "RightElbow",
    "RightWrist",
]
REFLECT_Y = np.diag([1.0, -1.0, 1.0])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh", type=Path, default=DEFAULT_BVH)
    parser.add_argument("--gmr-root", type=Path, default=DEFAULT_GMR_ROOT)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration-sec", type=float, default=8.0)
    parser.add_argument("--start-sec", type=float, help="Default: select the most active window")
    parser.add_argument("--warmup-sec", type=float, default=1.0)
    parser.add_argument("--max-velocity", type=float, default=3.0)
    parser.add_argument("--smoothing-window", type=int, default=7)
    parser.add_argument("--posture-cost", type=float, default=0.02)
    parser.add_argument("--solver", default="daqp")
    return parser.parse_args()


def load_xsens_bvh(path, gmr_root):
    sys.path.insert(0, str(gmr_root))
    from general_motion_retargeting.utils.lafan_vendor import utils
    from general_motion_retargeting.utils.xsens_vendor.BVHParser import Anim, BVHParser

    parser = BVHParser(axis_order="zxy", scale=0.01)
    rotations, _ = parser.parse(path.read_text(encoding="utf-8"))
    quats, positions, offsets, parents = parser._MOTION_data_post_processing(
        rotations,
        np.copy(parser.positions),
        reset_to_zero=True,
    )
    anim = Anim(quats, positions, offsets, parents, parser.names)
    global_quats, global_positions = utils.quat_fk(anim.quats, anim.pos, anim.parents)
    indices = {name: parser.names.index(name) for name in SOURCE_BODIES}
    positions = np.stack([global_positions[:, indices[name]] for name in SOURCE_BODIES], axis=1)
    quats = np.stack([global_quats[:, indices[name]] for name in SOURCE_BODIES], axis=1)

    positions = positions @ REFLECT_Y.T
    matrices = Rotation.from_quat(quats.reshape(-1, 4), scalar_first=True).as_matrix()
    matrices = REFLECT_Y @ matrices @ REFLECT_Y
    quats = Rotation.from_matrix(matrices).as_quat(scalar_first=True).reshape(quats.shape)
    return positions, quats, float(parser.frame_time)


def sample_indices(frame_count, frame_time, fps):
    end_time = (frame_count - 1) * frame_time
    times = np.arange(0.0, end_time + 1e-9, 1.0 / fps)
    indices = np.rint(times / frame_time).astype(np.int64)
    indices = np.clip(indices, 0, frame_count - 1)
    keep = np.r_[True, np.diff(indices) > 0]
    return indices[keep], times[keep]


def choose_window(positions, indices, times, duration_sec, fps, start_sec):
    frame_count = max(1, int(round(duration_sec * fps)))
    frame_count = min(frame_count, len(indices))
    if start_sec is not None:
        start = int(np.searchsorted(times, start_sec, side="left"))
        start = min(max(0, start), len(indices) - frame_count)
        return start, frame_count, None

    body_index = {name: i for i, name in enumerate(SOURCE_BODIES)}
    wrists = positions[indices][:, [body_index["LeftWrist"], body_index["RightWrist"]]]
    activity = np.linalg.norm(np.diff(wrists, axis=0), axis=2).sum(axis=1)
    if frame_count >= len(indices):
        return 0, frame_count, float(activity.sum())
    cumulative = np.r_[0.0, np.cumsum(activity)]
    scores = cumulative[frame_count:] - cumulative[:-frame_count]
    start = int(np.argmax(scores))
    return start, frame_count, float(scores[start])


def body_pose(model, data, mujoco, name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"Body not found in V2 model: {name}")
    return data.xpos[body_id].copy(), data.xmat[body_id].reshape(3, 3).copy()


def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        raise ValueError("Degenerate human limb segment")
    return vector / norm


def prepare_target_builder(model, mujoco, positions, quats):
    data = mujoco.MjData(model)
    neutral_qpos = np.zeros(model.nq, dtype=np.float64)
    for joint_name in ("joint_lShoulderRoll", "joint_rShoulderRoll"):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        neutral_qpos[model.jnt_qposadr[joint_id]] = -1.4
    data.qpos[:] = neutral_qpos
    mujoco.mj_forward(model, data)

    neutral = {}
    for name in (
        "link_torso",
        "link_lShoulderPitch",
        "link_lElbow",
        "link_lWristPitch",
        "link_rShoulderPitch",
        "link_rElbow",
        "link_rWristPitch",
    ):
        neutral[name] = body_pose(model, data, mujoco, name)

    source_index = {name: i for i, name in enumerate(SOURCE_BODIES)}
    reference_rotation = {
        name: Rotation.from_quat(quats[0, source_index[name]], scalar_first=True).as_matrix()
        for name in ("Chest4", "LeftWrist", "RightWrist")
    }

    def target_for_frame(frame_index):
        chest_rotation = Rotation.from_quat(
            quats[frame_index, source_index["Chest4"]], scalar_first=True
        ).as_matrix()
        torso_delta = chest_rotation @ reference_rotation["Chest4"].T
        torso_pos, torso_neutral_rotation = neutral["link_torso"]
        torso_target_rotation = torso_delta @ torso_neutral_rotation

        target = {
            "Chest4": [
                torso_pos.copy(),
                Rotation.from_matrix(torso_target_rotation).as_quat(scalar_first=True),
            ]
        }
        for side in ("Left", "Right"):
            shoulder_name = f"{side}Shoulder"
            elbow_name = f"{side}Elbow"
            wrist_name = f"{side}Wrist"
            robot_prefix = "l" if side == "Left" else "r"
            shoulder_body = f"link_{robot_prefix}ShoulderPitch"
            elbow_body = f"link_{robot_prefix}Elbow"
            wrist_body = f"link_{robot_prefix}WristPitch"

            shoulder_neutral = neutral[shoulder_body][0]
            shoulder_target = torso_pos + torso_delta @ (shoulder_neutral - torso_pos)
            upper_length = np.linalg.norm(neutral[elbow_body][0] - shoulder_neutral)
            fore_length = np.linalg.norm(neutral[wrist_body][0] - neutral[elbow_body][0])
            shoulder = positions[frame_index, source_index[shoulder_name]]
            elbow = positions[frame_index, source_index[elbow_name]]
            wrist = positions[frame_index, source_index[wrist_name]]
            elbow_target = shoulder_target + upper_length * normalize(elbow - shoulder)
            wrist_target = elbow_target + fore_length * normalize(wrist - elbow)

            wrist_rotation = Rotation.from_quat(
                quats[frame_index, source_index[wrist_name]], scalar_first=True
            ).as_matrix()
            wrist_delta = wrist_rotation @ reference_rotation[wrist_name].T
            wrist_target_rotation = wrist_delta @ neutral[wrist_body][1]
            target[elbow_name] = [elbow_target, np.array([1.0, 0.0, 0.0, 0.0])]
            target[wrist_name] = [
                wrist_target,
                Rotation.from_matrix(wrist_target_rotation).as_quat(scalar_first=True),
            ]
        return target

    return neutral_qpos, target_for_frame


def configure_retargeter(gmr_root, urdf, config, solver):
    sys.path.insert(0, str(gmr_root))
    import mink
    import mujoco
    from general_motion_retargeting import GeneralMotionRetargeting
    from general_motion_retargeting.params import IK_CONFIG_DICT, ROBOT_XML_DICT

    ROBOT_XML_DICT["ula_v2"] = urdf
    IK_CONFIG_DICT.setdefault("bvh_xsens_ula", {})["ula_v2"] = config
    retargeter = GeneralMotionRetargeting(
        src_human="bvh_xsens_ula",
        tgt_robot="ula_v2",
        solver=solver,
        damping=0.1,
        verbose=False,
        use_velocity_limit=False,
    )
    retargeter.max_iter = 20
    return retargeter, mujoco, mink


def extract_joint_row(model, qpos, mujoco, joint_order=JOINT_ORDER):
    values = []
    for name in joint_order:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Joint not found in V2 model: {name}")
        values.append(qpos[model.jnt_qposadr[joint_id]])
    return np.asarray(values, dtype=np.float64)


def smooth_and_limit(
    raw,
    fps,
    max_velocity,
    smoothing_window,
    *,
    joint_order=JOINT_ORDER,
    joint_limits=JOINT_LIMITS,
):
    lower = np.asarray([joint_limits[name][0] for name in joint_order])
    upper = np.asarray([joint_limits[name][1] for name in joint_order])
    clipped = np.clip(raw, lower, upper)
    window = min(int(smoothing_window), len(clipped) if len(clipped) % 2 else len(clipped) - 1)
    if window >= 5:
        if window % 2 == 0:
            window -= 1
        filtered = savgol_filter(clipped, window_length=window, polyorder=2, axis=0, mode="interp")
    else:
        filtered = clipped.copy()
    filtered = np.clip(filtered, lower, upper)
    nominal_dt = 1.0 / float(fps)
    segment_delta = np.max(np.abs(np.diff(filtered, axis=0)), axis=1)
    segment_duration = np.maximum(nominal_dt, segment_delta / float(max_velocity))
    key_times = np.r_[0.0, np.cumsum(segment_duration)]
    output_times = np.arange(0.0, key_times[-1] + nominal_dt * 0.5, nominal_dt)
    safe = np.column_stack(
        [np.interp(output_times, key_times, filtered[:, joint]) for joint in range(filtered.shape[1])]
    )
    return np.clip(safe, lower, upper), key_times, output_times


def retime_targets(targets, key_times, output_times):
    retimed = []
    for output_time in output_times:
        right = int(np.searchsorted(key_times, output_time, side="right"))
        right = min(max(1, right), len(targets) - 1)
        left = right - 1
        span = key_times[right] - key_times[left]
        alpha = 0.0 if span <= 0 else float((output_time - key_times[left]) / span)
        frame = {}
        for body_name in targets[left]:
            left_pos = np.asarray(targets[left][body_name][0])
            right_pos = np.asarray(targets[right][body_name][0])
            frame[body_name] = [
                (1.0 - alpha) * left_pos + alpha * right_pos,
                np.array([1.0, 0.0, 0.0, 0.0]),
            ]
        retimed.append(frame)
    return retimed


def write_csv(path, trajectory, *, joint_order=JOINT_ORDER):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(joint_order)
        writer.writerows([[f"{value:.8f}" for value in row] for row in trajectory])


def derivatives(trajectory, fps):
    velocity = np.diff(trajectory, axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps
    jerk = np.diff(acceleration, axis=0) * fps
    return velocity, acceleration, jerk


def rendered_pose_metrics(model, mujoco, trajectory, targets, *, joint_order=JOINT_ORDER):
    data = mujoco.MjData(model)
    joint_addresses = []
    for name in joint_order:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        joint_addresses.append(int(model.jnt_qposadr[joint_id]))
    body_map = {
        "LeftElbow": "link_lElbow",
        "LeftWrist": "link_lWristPitch",
        "RightElbow": "link_rElbow",
        "RightWrist": "link_rWristPitch",
    }
    errors = []
    collision_frames = 0
    collision_pairs = Counter()

    def group(body_name):
        if body_name.startswith("link_l") and any(key in body_name for key in ("Shoulder", "Elbow", "Wrist")):
            return "left_arm"
        if body_name.startswith("link_r") and any(key in body_name for key in ("Shoulder", "Elbow", "Wrist")):
            return "right_arm"
        if body_name in {"link_torso", "link_pelvisYaw", "link_pelvisPitch"}:
            return "torso"
        return None

    for row, target in zip(trajectory, targets):
        data.qpos[:] = 0.0
        data.qpos[joint_addresses] = row
        mujoco.mj_forward(model, data)
        frame_errors = []
        for human_name, robot_name in body_map.items():
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_name)
            frame_errors.append(np.linalg.norm(data.xpos[body_id] - target[human_name][0]))
        errors.extend(frame_errors)

        frame_has_collision = False
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            body1 = int(model.geom_bodyid[contact.geom1])
            body2 = int(model.geom_bodyid[contact.geom2])
            name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1) or "world"
            name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2) or "world"
            group1, group2 = group(name1), group(name2)
            if group1 is not None and group2 is not None and group1 != group2:
                pair = " <-> ".join(sorted((name1, name2)))
                collision_pairs[pair] += 1
                frame_has_collision = True
        collision_frames += int(frame_has_collision)

    errors = np.asarray(errors)
    return {
        "limb_target_error_mean_m": float(errors.mean()),
        "limb_target_error_p95_m": float(np.percentile(errors, 95)),
        "limb_target_error_max_m": float(errors.max()),
        "upper_body_collision_frames": int(collision_frames),
        "upper_body_collision_frame_rate": float(collision_frames / len(trajectory)),
        "upper_body_collision_pairs": dict(collision_pairs.most_common(20)),
    }


def quality_report(
    raw,
    safe,
    fps,
    max_velocity,
    pose_metrics,
    metadata,
    *,
    joint_order=JOINT_ORDER,
    joint_limits=JOINT_LIMITS,
):
    lower = np.asarray([joint_limits[name][0] for name in joint_order])
    upper = np.asarray([joint_limits[name][1] for name in joint_order])
    raw_limit_mask = (raw < lower) | (raw > upper)
    safe_limit_mask = (safe < lower - 1e-8) | (safe > upper + 1e-8)
    raw_velocity, _, _ = derivatives(raw, fps)
    velocity, acceleration, jerk = derivatives(safe, fps)
    ranges = {
        name: {
            "min_rad": float(safe[:, index].min()),
            "max_rad": float(safe[:, index].max()),
            "range_deg": float(np.rad2deg(np.ptp(safe[:, index]))),
        }
        for index, name in enumerate(joint_order)
    }
    report = {
        **metadata,
        "raw_joint_limit_violation_fraction": float(raw_limit_mask.mean()),
        "safe_joint_limit_violations": int(safe_limit_mask.sum()),
        "raw_velocity_violation_fraction": float((np.abs(raw_velocity) > max_velocity).mean()),
        "safe_max_velocity_rad_s": float(np.abs(velocity).max(initial=0.0)),
        "safe_p95_velocity_rad_s": float(np.percentile(np.abs(velocity), 95)),
        "safe_max_acceleration_rad_s2": float(np.abs(acceleration).max(initial=0.0)),
        "safe_rms_jerk_rad_s3": float(np.sqrt(np.mean(np.square(jerk)))) if jerk.size else 0.0,
        "joint_ranges": ranges,
        **pose_metrics,
    }
    report["quality_gate"] = {
        "joint_limits_pass": report["safe_joint_limit_violations"] == 0,
        "velocity_pass": report["safe_max_velocity_rad_s"] <= max_velocity + 1e-6,
        "target_fit_pass": report["limb_target_error_p95_m"] <= 0.04,
        "collision_pass": report["upper_body_collision_frame_rate"] <= 0.05,
    }
    report["quality_gate"]["passed"] = all(report["quality_gate"].values())
    return report


def main():
    args = parse_args()
    for path in (args.bvh, args.gmr_root, args.urdf, args.config):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.fps <= 0 or args.duration_sec <= 0 or args.max_velocity <= 0:
        raise ValueError("fps, duration-sec, and max-velocity must be positive")

    started = time.perf_counter()
    positions, quats, frame_time = load_xsens_bvh(args.bvh, args.gmr_root)
    indices, times = sample_indices(len(positions), frame_time, args.fps)
    window_start, window_frames, activity_score = choose_window(
        positions, indices, times, args.duration_sec, args.fps, args.start_sec
    )
    window_end = window_start + window_frames
    warmup_frames = int(round(args.warmup_sec * args.fps))
    solve_start = max(0, window_start - warmup_frames)

    retargeter, mujoco, mink = configure_retargeter(
        args.gmr_root, args.urdf, args.config, args.solver
    )
    neutral_qpos, target_for_frame = prepare_target_builder(
        retargeter.model, mujoco, positions, quats
    )
    retargeter.configuration.update(neutral_qpos)
    if args.posture_cost > 0:
        posture_task = mink.PostureTask(
            retargeter.model,
            cost=args.posture_cost,
            lm_damping=1.0,
        )
        posture_task.set_target(neutral_qpos)
        retargeter.tasks1.append(posture_task)

    first_target = target_for_frame(int(indices[solve_start]))
    for _ in range(20):
        retargeter.retarget(first_target)

    raw_rows = []
    output_targets = []
    ik_errors = []
    for sample_index in range(solve_start, window_end):
        target = target_for_frame(int(indices[sample_index]))
        qpos = retargeter.retarget(target)
        if sample_index >= window_start:
            raw_rows.append(extract_joint_row(retargeter.model, qpos, mujoco))
            output_targets.append(target)
            ik_errors.append(float(retargeter.error1()))

    raw = np.asarray(raw_rows)
    safe, retime_key_times, retime_output_times = smooth_and_limit(
        raw, args.fps, args.max_velocity, args.smoothing_window
    )
    safe_targets = retime_targets(output_targets, retime_key_times, retime_output_times)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = args.output_dir / "boxing_gmr_raw_15d.csv"
    safe_csv = args.output_dir / "boxing_gmr_safe_15d.csv"
    quality_json = args.output_dir / "quality.json"
    write_csv(raw_csv, raw)
    write_csv(safe_csv, safe)

    raw_pose_metrics = rendered_pose_metrics(retargeter.model, mujoco, raw, output_targets)
    pose_metrics = rendered_pose_metrics(retargeter.model, mujoco, safe, safe_targets)
    pose_metrics.update(
        {
            f"raw_{key}": value
            for key, value in raw_pose_metrics.items()
            if key.startswith("limb_target_error")
        }
    )
    metadata = {
        "source_bvh": str(args.bvh.resolve()),
        "source_url": "https://github.com/YanjieZe/GMR/blob/master/assets/xsens_bvh_test/251021_04_boxing_120Hz_cm_3DsMax.bvh",
        "gmr_commit": "bb1bbe40774794fceb2a7c579a3464a28e68c844",
        "robot_urdf": str(args.urdf.resolve()),
        "frames": int(len(safe)),
        "fps": float(args.fps),
        "duration_sec": float(len(safe) / args.fps),
        "source_window_frames": int(len(raw)),
        "source_window_duration_sec": float(len(raw) / args.fps),
        "retime_factor": float(len(safe) / len(raw)),
        "source_fps": float(1.0 / frame_time),
        "selected_start_sec": float(times[window_start]),
        "selected_end_sec": float(times[window_end - 1] + 1.0 / args.fps),
        "selection": "explicit_start" if args.start_sec is not None else "max_wrist_activity",
        "activity_score": activity_score,
        "max_velocity_rad_s": float(args.max_velocity),
        "posture_cost": float(args.posture_cost),
        "mean_final_ik_objective": float(np.mean(ik_errors)),
        "processing_sec": float(time.perf_counter() - started),
        "outputs": {
            "raw_csv": str(raw_csv.resolve()),
            "safe_csv": str(safe_csv.resolve()),
        },
    }
    report = quality_report(raw, safe, args.fps, args.max_velocity, pose_metrics, metadata)
    quality_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
