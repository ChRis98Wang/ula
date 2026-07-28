#!/usr/bin/env python3
"""Retarget one Hanyang 19-point clip to a partial-observation ULA V2 18D clip.

The training candidate always remains exactly 150 frames at 30 Hz.  A separate
velocity-safe, time-dilated trajectory is emitted for deployment preview only
and is explicitly forbidden from emotion-timing supervision.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping
from xml.etree import ElementTree

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from upper_body_skeleton.hanyang_emotion_retarget import (  # noqa: E402
    ACTION_DIM_MASK_18D,
    AXIS_POLICY,
    DATASET,
    DATASET_ID,
    DATASET_LICENSE,
    DATASET_REVISION,
    HEAD_OBSERVATION_POLICY,
    SOURCE_FPS,
    SOURCE_FRAMES,
    SOURCE_JOINTS,
    SOURCE_TO_ROBOT_BASIS,
    UNOBSERVED_18D_JOINTS,
    WRIST_OBSERVATION_POLICY,
    json_hash,
    load_hanyang_csv,
    observation_confidence_18d,
    observed_head_angles,
    observed_head_rotations,
    reject_forbidden_dataset_marker,
    robot_torso_rotations,
    sha256_file,
    source_geometry_quality,
)
from upper_body_skeleton.retarget_v2_18d import (  # noqa: E402
    CONTRACT_VERSION as ULA_V2_18D_CONTRACT,
    JOINT_LIMITS_18D,
    JOINT_ORDER_18D,
)

try:  # noqa: E402
    from .retarget_motionx322_v2 import (
        DEFAULT_GMR_ROOT,
        DEFAULT_URDF,
        configure_retargeter,
        enforce_human_elbow_branch,
        enforce_safe_elbow_branch,
        extract_joint_row,
        reconstruct_v2_head_rotations,
        write_csv,
    )
except ImportError:  # pragma: no cover - direct invocation
    from retarget_motionx322_v2 import (
        DEFAULT_GMR_ROOT,
        DEFAULT_URDF,
        configure_retargeter,
        enforce_human_elbow_branch,
        enforce_safe_elbow_branch,
        extract_joint_row,
        reconstruct_v2_head_rotations,
        write_csv,
    )


DEFAULT_CONFIG = Path(__file__).with_name("hanyang_positions_to_ula_v2.json")
DEFAULT_OUTPUT = PROJECT_ROOT / "deliverables/external_emotion_research/hanyang_retarget_smoke"
UNOBSERVED_INDICES = tuple(
    JOINT_ORDER_18D.index(name) for name in sorted(UNOBSERVED_18D_JOINTS)
)
SOURCE_POSITION_INDEX = {name: index for index, name in enumerate(SOURCE_JOINTS)}
DEFAULT_MAX_VELOCITY_RAD_S = 3.0
DEFAULT_SMOOTHING_WINDOW = 5
DEFAULT_POSTURE_COST = 0.02
TARGET_POSITION_P95_LIMIT_M = 0.04
DIRECTION_MEAN_LIMIT_DEG = 10.0
DIRECTION_P95_LIMIT_DEG = 15.0
COLLISION_FRAME_RATE_LIMIT = 0.05
COLLISION_CONSECUTIVE_LIMIT = 3
SATURATION_FRACTION_LIMIT = 0.01
SATURATION_CONSECUTIVE_LIMIT = 3
HEAD_PROXY_DIRECTION_P95_LIMIT_DEG = 5.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--human-evaluation-json", type=Path)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--gmr-root", type=Path, default=DEFAULT_GMR_ROOT)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--solver", default="daqp")
    parser.add_argument(
        "--max-velocity",
        type=float,
        default=DEFAULT_MAX_VELOCITY_RAD_S,
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=DEFAULT_SMOOTHING_WINDOW,
    )
    parser.add_argument(
        "--posture-cost",
        type=float,
        default=DEFAULT_POSTURE_COST,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _body_pose(model, data, mujoco, name: str) -> tuple[np.ndarray, np.ndarray]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"Body not found in V2 model: {name}")
    return data.xpos[body_id].copy(), data.xmat[body_id].reshape(3, 3).copy()


def _normalize(vector: np.ndarray, *, context: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError(f"degenerate Hanyang vector: {context}")
    return vector / norm


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(value), allow_pickle=False)
    os.replace(temporary, path)


def _longest_true_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in np.asarray(values, dtype=np.bool_).tolist():
        current = current + 1 if value else 0
        longest = max(longest, current)
    return int(longest)


def _software_limit_arrays() -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(
        [JOINT_LIMITS_18D[name][0] for name in JOINT_ORDER_18D],
        dtype=np.float64,
    )
    upper = np.asarray(
        [JOINT_LIMITS_18D[name][1] for name in JOINT_ORDER_18D],
        dtype=np.float64,
    )
    return lower, upper


def urdf_velocity_limits(
    urdf: str | Path, *, maximum_rad_s: float
) -> tuple[np.ndarray, dict[str, float]]:
    root = ElementTree.parse(Path(urdf)).getroot()
    declared: dict[str, float] = {}
    for joint in root.findall(".//joint"):
        name = joint.attrib.get("name")
        if name not in JOINT_ORDER_18D:
            continue
        limit = joint.find("limit")
        if limit is None or "velocity" not in limit.attrib:
            raise ValueError(f"ULA URDF joint is missing velocity limit: {name}")
        declared[name] = float(limit.attrib["velocity"])
    missing = sorted(set(JOINT_ORDER_18D) - set(declared))
    if missing:
        raise ValueError(f"ULA URDF is missing 18D velocity limits: {missing}")
    effective = {
        name: min(float(maximum_rad_s), declared[name])
        for name in JOINT_ORDER_18D
    }
    return (
        np.asarray([effective[name] for name in JOINT_ORDER_18D], dtype=np.float64),
        effective,
    )


def smooth_source_faithful(
    raw: np.ndarray, *, smoothing_window: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Smooth IK jitter without changing frame count, FPS, or endpoints."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape != (SOURCE_FRAMES, len(JOINT_ORDER_18D)):
        raise ValueError("raw Hanyang IK must have shape [150, 18]")
    lower, upper = _software_limit_arrays()
    raw_limit_mask = (raw < lower) | (raw > upper)
    clipped = np.clip(raw, lower, upper)
    window = min(
        int(smoothing_window),
        len(clipped) if len(clipped) % 2 else len(clipped) - 1,
    )
    if window >= 5:
        if window % 2 == 0:
            window -= 1
        filtered = savgol_filter(
            clipped,
            window_length=window,
            polyorder=2,
            axis=0,
            mode="interp",
        )
        # Preserve the captured endpoints; do not synthesize a terminal hold.
        filtered[0] = clipped[0]
        filtered[-1] = clipped[-1]
    else:
        filtered = clipped.copy()
    filtered = np.clip(filtered, lower, upper)
    filtered = enforce_safe_elbow_branch(
        filtered, joint_order=JOINT_ORDER_18D
    )
    filtered[:, UNOBSERVED_INDICES] = 0.0
    return filtered, {
        "raw_joint_limit_violation_count": int(raw_limit_mask.sum()),
        "raw_joint_limit_violation_fraction": float(raw_limit_mask.mean()),
        "smoothing_window": int(window if window >= 5 else 0),
        "endpoint_policy": "source_first_and_last_frames_preserved_no_terminal_hold",
        "retimed": False,
        "retime_factor": 1.0,
    }


def velocity_safe_preview(
    trajectory: np.ndarray,
    *,
    fps: float,
    velocity_limits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a deployment-only time dilation using per-joint velocity limits."""
    trajectory = np.asarray(trajectory, dtype=np.float64)
    velocity_limits = np.asarray(velocity_limits, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != len(JOINT_ORDER_18D):
        raise ValueError("trajectory must have shape [frames, 18]")
    if velocity_limits.shape != (len(JOINT_ORDER_18D),) or np.any(
        velocity_limits <= 0
    ):
        raise ValueError("velocity limits must contain 18 positive values")
    nominal_dt = 1.0 / float(fps)
    required = np.abs(np.diff(trajectory, axis=0)) / velocity_limits[None, :]
    segment_duration = np.maximum(nominal_dt, required.max(axis=1))
    key_times = np.r_[0.0, np.cumsum(segment_duration)]
    output_times = np.arange(
        0.0, key_times[-1] + nominal_dt * 0.5, nominal_dt
    )
    output = np.column_stack(
        [
            np.interp(output_times, key_times, trajectory[:, dimension])
            for dimension in range(trajectory.shape[1])
        ]
    )
    output[:, UNOBSERVED_INDICES] = 0.0
    return output, key_times, output_times


def _retime_targets(
    targets: list[dict[str, list[np.ndarray]]],
    key_times: np.ndarray,
    output_times: np.ndarray,
) -> list[dict[str, list[np.ndarray]]]:
    result = []
    for output_time in output_times:
        right = int(np.searchsorted(key_times, output_time, side="right"))
        right = min(max(1, right), len(targets) - 1)
        left = right - 1
        span = key_times[right] - key_times[left]
        alpha = 0.0 if span <= 0 else float(
            (output_time - key_times[left]) / span
        )
        frame: dict[str, list[np.ndarray]] = {}
        for body_name in targets[left]:
            left_position = np.asarray(targets[left][body_name][0])
            right_position = np.asarray(targets[right][body_name][0])
            frame[body_name] = [
                (1.0 - alpha) * left_position + alpha * right_position,
                np.asarray(targets[left][body_name][1]),
            ]
        result.append(frame)
    return result


def _collision_group(body_name: str) -> str | None:
    if body_name.startswith("link_l") and any(
        token in body_name for token in ("Shoulder", "Elbow", "Wrist")
    ):
        return "left_arm"
    if body_name.startswith("link_r") and any(
        token in body_name for token in ("Shoulder", "Elbow", "Wrist")
    ):
        return "right_arm"
    if body_name in {"link_torso", "link_pelvisYaw", "link_pelvisPitch"}:
        return "torso"
    return None


def fk_quality_metrics(
    model,
    mujoco,
    trajectory: np.ndarray,
    targets: list[dict[str, list[np.ndarray]]],
) -> dict[str, Any]:
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if len(trajectory) != len(targets):
        raise ValueError("trajectory and target frame counts differ")
    data = mujoco.MjData(model)
    joint_addresses = []
    for name in JOINT_ORDER_18D:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        joint_addresses.append(int(model.jnt_qposadr[joint_id]))

    def position(name: str) -> np.ndarray:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Body not found in V2 model: {name}")
        return data.xpos[body_id].copy()

    target_errors: list[float] = []
    direction_errors: dict[str, list[float]] = {
        "left_upper": [],
        "left_fore": [],
        "right_upper": [],
        "right_fore": [],
    }
    collision_mask: list[bool] = []
    for row, target in zip(trajectory, targets, strict=True):
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
            target_elbow = np.asarray(target[f"{side}Elbow"][0])
            target_wrist = np.asarray(target[f"{side}Wrist"][0])
            target_errors.extend(
                (
                    float(np.linalg.norm(elbow - target_elbow)),
                    float(np.linalg.norm(wrist - target_wrist)),
                )
            )
            for label, desired, achieved in (
                (
                    f"{key_prefix}_upper",
                    target_elbow - shoulder,
                    elbow - shoulder,
                ),
                (
                    f"{key_prefix}_fore",
                    target_wrist - target_elbow,
                    wrist - elbow,
                ),
            ):
                cosine = float(
                    np.clip(
                        np.dot(
                            _normalize(desired, context=label),
                            _normalize(achieved, context=label),
                        ),
                        -1.0,
                        1.0,
                    )
                )
                direction_errors[label].append(float(np.rad2deg(np.arccos(cosine))))

        collision = False
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            body1 = int(model.geom_bodyid[contact.geom1])
            body2 = int(model.geom_bodyid[contact.geom2])
            name1 = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1)
                or "world"
            )
            name2 = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2)
                or "world"
            )
            group1 = _collision_group(name1)
            group2 = _collision_group(name2)
            if group1 is not None and group2 is not None and group1 != group2:
                collision = True
                break
        collision_mask.append(collision)

    target_errors_array = np.asarray(target_errors, dtype=np.float64)
    flattened_direction = np.concatenate(
        [np.asarray(values) for values in direction_errors.values()]
    )
    collision_array = np.asarray(collision_mask, dtype=np.bool_)
    return {
        "limb_target_error_mean_m": float(target_errors_array.mean()),
        "limb_target_error_p95_m": float(
            np.percentile(target_errors_array, 95)
        ),
        "limb_target_error_max_m": float(target_errors_array.max()),
        "limb_direction_error_mean_deg": float(flattened_direction.mean()),
        "limb_direction_error_p95_deg": float(
            np.percentile(flattened_direction, 95)
        ),
        "limb_direction_error_max_deg": float(flattened_direction.max()),
        "limb_direction_error_by_segment": {
            name: {
                "mean_deg": float(np.mean(values)),
                "p95_deg": float(np.percentile(values, 95)),
                "max_deg": float(np.max(values)),
            }
            for name, values in direction_errors.items()
        },
        "upper_body_collision_frames": int(collision_array.sum()),
        "upper_body_collision_frame_rate": float(collision_array.mean()),
        "upper_body_collision_max_consecutive_frames": _longest_true_run(
            collision_array
        ),
    }


def trajectory_quality(
    trajectory: np.ndarray,
    *,
    fps: float,
    velocity_limits: np.ndarray,
) -> dict[str, Any]:
    trajectory = np.asarray(trajectory, dtype=np.float64)
    lower, upper = _software_limit_arrays()
    limit_mask = (trajectory < lower - 1e-8) | (trajectory > upper + 1e-8)
    velocity = np.diff(trajectory, axis=0) * float(fps)
    acceleration = np.diff(velocity, axis=0) * float(fps)
    jerk = np.diff(acceleration, axis=0) * float(fps)
    velocity_ratio = (
        np.abs(velocity) / np.asarray(velocity_limits)[None, :]
        if len(velocity)
        else np.zeros((0, len(JOINT_ORDER_18D)))
    )
    lower_saturation = np.isclose(trajectory, lower[None, :], atol=1e-5)
    upper_saturation = np.isclose(trajectory, upper[None, :], atol=1e-5)
    # Pelvis pitch is intentionally one-sided and its upright neutral pose is
    # exactly the lower bound.  Remaining there is not evidence of clipping.
    lower_saturation[:, JOINT_ORDER_18D.index("joint_pelvisPitch")] = False
    saturation = lower_saturation | upper_saturation
    observed = np.asarray(ACTION_DIM_MASK_18D, dtype=np.bool_)
    saturation_fraction = saturation[:, observed].mean(axis=0)
    saturation_runs = [
        _longest_true_run(saturation[:, index])
        for index in np.flatnonzero(observed)
    ]
    return {
        "joint_limit_violations": int(limit_mask.sum()),
        "max_velocity_rad_s": float(np.abs(velocity).max(initial=0.0)),
        "max_velocity_limit_ratio": float(velocity_ratio.max(initial=0.0)),
        "per_joint_max_velocity_rad_s": {
            name: float(np.abs(velocity[:, index]).max(initial=0.0))
            for index, name in enumerate(JOINT_ORDER_18D)
        },
        "max_acceleration_rad_s2": float(
            np.abs(acceleration).max(initial=0.0)
        ),
        "rms_jerk_rad_s3": (
            float(np.sqrt(np.mean(np.square(jerk)))) if jerk.size else 0.0
        ),
        "observed_joint_max_saturation_fraction": float(
            saturation_fraction.max(initial=0.0)
        ),
        "per_joint_saturation_fraction": {
            name: float(saturation[:, index].mean())
            for index, name in enumerate(JOINT_ORDER_18D)
        },
        "observed_joint_max_consecutive_saturation_frames": int(
            max(saturation_runs, default=0)
        ),
        "per_joint_max_consecutive_saturation_frames": {
            name: _longest_true_run(saturation[:, index])
            for index, name in enumerate(JOINT_ORDER_18D)
        },
    }


class HanyangRetargetRuntime:
    """Reusable GMR/Mink runtime; one instance should be kept per batch worker."""

    def __init__(
        self,
        *,
        gmr_root: str | Path = DEFAULT_GMR_ROOT,
        urdf: str | Path = DEFAULT_URDF,
        config: str | Path = DEFAULT_CONFIG,
        solver: str = "daqp",
        posture_cost: float = DEFAULT_POSTURE_COST,
    ):
        self.gmr_root = Path(gmr_root).resolve()
        self.urdf = Path(urdf).resolve()
        self.config = Path(config).resolve()
        reject_forbidden_dataset_marker(
            [str(self.gmr_root), str(self.urdf), str(self.config)],
            context="retarget_runtime",
        )
        for path in (self.gmr_root, self.urdf, self.config):
            if not path.exists():
                raise FileNotFoundError(path)
        self.retargeter, self.mujoco, self.mink = configure_retargeter(
            self.gmr_root, self.urdf, self.config, solver
        )
        enforce_human_elbow_branch(self.retargeter, self.mujoco, self.mink)
        self.posture_cost = float(posture_cost)
        self.neutral_qpos, self.neutral = self._build_neutral()
        if self.posture_cost > 0:
            posture_task = self.mink.PostureTask(
                self.retargeter.model,
                cost=self.posture_cost,
                lm_damping=1.0,
            )
            posture_task.set_target(self.neutral_qpos)
            self.retargeter.tasks1.append(posture_task)

    def _build_neutral(self) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
        model = self.retargeter.model
        data = self.mujoco.MjData(model)
        neutral_qpos = np.zeros(model.nq, dtype=np.float64)
        for name in ("joint_lShoulderRoll", "joint_rShoulderRoll"):
            joint_id = self.mujoco.mj_name2id(
                model, self.mujoco.mjtObj.mjOBJ_JOINT, name
            )
            neutral_qpos[model.jnt_qposadr[joint_id]] = -1.4
        data.qpos[:] = neutral_qpos
        self.mujoco.mj_forward(model, data)
        neutral = {
            name: _body_pose(model, data, self.mujoco, name)
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
        return neutral_qpos, neutral

    def targets_for_positions(
        self, positions: np.ndarray
    ) -> list[dict[str, list[np.ndarray]]]:
        positions = np.asarray(positions, dtype=np.float64)
        torso_rotations = robot_torso_rotations(positions)
        torso_position, torso_neutral_rotation = self.neutral["link_torso"]
        targets: list[dict[str, list[np.ndarray]]] = []
        for frame_index, torso_target_rotation in enumerate(torso_rotations):
            torso_delta = torso_target_rotation @ torso_neutral_rotation.T
            target: dict[str, list[np.ndarray]] = {
                "Chest4": [
                    torso_position.copy(),
                    Rotation.from_matrix(torso_target_rotation).as_quat(
                        scalar_first=True
                    ),
                ]
            }
            for side, prefix in (("Left", "l"), ("Right", "r")):
                shoulder_body = f"link_{prefix}ShoulderPitch"
                elbow_body = f"link_{prefix}Elbow"
                wrist_body = f"link_{prefix}WristPitch"
                shoulder_neutral = self.neutral[shoulder_body][0]
                shoulder_target = torso_position + torso_delta @ (
                    shoulder_neutral - torso_position
                )
                upper_length = float(
                    np.linalg.norm(
                        self.neutral[elbow_body][0] - shoulder_neutral
                    )
                )
                fore_length = float(
                    np.linalg.norm(
                        self.neutral[wrist_body][0]
                        - self.neutral[elbow_body][0]
                    )
                )
                shoulder = positions[
                    frame_index, SOURCE_POSITION_INDEX[f"{side}Arm"]
                ]
                elbow = positions[
                    frame_index, SOURCE_POSITION_INDEX[f"{side}ForeArm"]
                ]
                wrist = positions[
                    frame_index, SOURCE_POSITION_INDEX[f"{side}Hand"]
                ]
                upper_direction = SOURCE_TO_ROBOT_BASIS @ _normalize(
                    elbow - shoulder, context=f"{side} upper arm"
                )
                fore_direction = SOURCE_TO_ROBOT_BASIS @ _normalize(
                    wrist - elbow, context=f"{side} forearm"
                )
                elbow_target = shoulder_target + upper_length * upper_direction
                wrist_target = elbow_target + fore_length * fore_direction
                target[f"{side}Elbow"] = [
                    elbow_target,
                    np.asarray((1.0, 0.0, 0.0, 0.0)),
                ]
                # Rotation cost is zero in the Hanyang GMR configuration.
                target[f"{side}Wrist"] = [
                    wrist_target,
                    Rotation.from_matrix(self.neutral[wrist_body][1]).as_quat(
                        scalar_first=True
                    ),
                ]
            targets.append(target)
        return targets

    def retarget(
        self,
        source_csv: str | Path,
        output_dir: str | Path,
        *,
        human_evaluation: Mapping[str, Any] | None = None,
        expected_source_sha256: str | None = None,
        max_velocity: float = DEFAULT_MAX_VELOCITY_RAD_S,
        smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        source = load_hanyang_csv(source_csv)
        if (
            expected_source_sha256
            and source["source_sha256"] != expected_source_sha256
        ):
            raise ValueError("Hanyang source SHA256 does not match inventory")
        if human_evaluation is not None:
            reject_forbidden_dataset_marker(
                human_evaluation, context="human_evaluation"
            )
            if human_evaluation.get("clip_id") != source["clip_id"]:
                raise ValueError("human evaluation does not match Hanyang clip")
        if max_velocity <= 0:
            raise ValueError("max_velocity must be positive")

        output_dir = Path(output_dir).resolve()
        reject_forbidden_dataset_marker(str(output_dir), context="output_dir")
        quality_json = output_dir / "quality.json"
        if quality_json.exists() and not overwrite:
            raise FileExistsError(quality_json)
        output_dir.mkdir(parents=True, exist_ok=True)

        positions = np.asarray(source["positions"], dtype=np.float64)
        source_quality = source_geometry_quality(positions)
        frame_confidence = observation_confidence_18d(positions)
        head_rotations = observed_head_rotations(positions)
        head_angles = observed_head_angles(positions)
        targets = self.targets_for_positions(positions)

        self.retargeter.configuration.update(self.neutral_qpos)
        for _ in range(20):
            self.retargeter.retarget(targets[0])
        raw_body_rows = []
        ik_errors = []
        for target in targets:
            qpos = self.retargeter.retarget(target)
            raw_body_rows.append(
                extract_joint_row(
                    self.retargeter.model,
                    qpos,
                    self.mujoco,
                )
            )
            ik_errors.append(float(self.retargeter.error1()))
        raw_body = np.asarray(raw_body_rows, dtype=np.float64)
        raw = np.column_stack((raw_body, head_angles))
        raw[:, UNOBSERVED_INDICES] = 0.0
        faithful, smoothing = smooth_source_faithful(
            raw, smoothing_window=smoothing_window
        )
        velocity_limits, effective_velocity_limits = urdf_velocity_limits(
            self.urdf, maximum_rad_s=max_velocity
        )
        preview, preview_key_times, preview_output_times = velocity_safe_preview(
            faithful,
            fps=SOURCE_FPS,
            velocity_limits=velocity_limits,
        )
        preview_targets = _retime_targets(
            targets, preview_key_times, preview_output_times
        )

        raw_csv = output_dir / f"{source['source_stem']}_raw_18d.csv"
        faithful_csv = (
            output_dir
            / f"{source['source_stem']}_source_faithful_partial_18d.csv"
        )
        preview_csv = (
            output_dir
            / f"{source['source_stem']}_deployment_safe_partial_18d.csv"
        )
        confidence_npy = output_dir / "observation_confidence_18d.npy"
        write_csv(raw_csv, raw, joint_order=JOINT_ORDER_18D)
        write_csv(faithful_csv, faithful, joint_order=JOINT_ORDER_18D)
        write_csv(preview_csv, preview, joint_order=JOINT_ORDER_18D)
        _atomic_npy(confidence_npy, frame_confidence)

        fk_metrics = fk_quality_metrics(
            self.retargeter.model, self.mujoco, faithful, targets
        )
        preview_fk_metrics = fk_quality_metrics(
            self.retargeter.model,
            self.mujoco,
            preview,
            preview_targets,
        )
        trajectory_metrics = trajectory_quality(
            faithful,
            fps=SOURCE_FPS,
            velocity_limits=velocity_limits,
        )
        reconstructed_head = reconstruct_v2_head_rotations(
            faithful[
                :,
                [
                    JOINT_ORDER_18D.index("head_roll_joint"),
                    JOINT_ORDER_18D.index("head_pitch_joint"),
                    JOINT_ORDER_18D.index("head_yaw_joint"),
                ],
            ]
        )
        head_error = Rotation.from_matrix(
            np.swapaxes(reconstructed_head, 1, 2) @ head_rotations
        ).magnitude()
        head_error_p95_deg = float(np.rad2deg(np.percentile(head_error, 95)))

        gate = {
            "source_geometry_pass": bool(
                source_quality["quality_gate"]["passed"]
            ),
            "fixed_150_frames_30hz_pass": bool(
                len(faithful) == SOURCE_FRAMES and SOURCE_FPS == 30.0
            ),
            "retime_factor_exactly_one_pass": True,
            "joint_limits_pass": (
                trajectory_metrics["joint_limit_violations"] == 0
            ),
            "per_joint_velocity_pass": (
                trajectory_metrics["max_velocity_limit_ratio"] <= 1.0 + 1e-6
            ),
            "target_fit_pass": (
                fk_metrics["limb_target_error_p95_m"]
                <= TARGET_POSITION_P95_LIMIT_M
            ),
            "limb_direction_pass": bool(
                fk_metrics["limb_direction_error_mean_deg"]
                <= DIRECTION_MEAN_LIMIT_DEG
                and fk_metrics["limb_direction_error_p95_deg"]
                <= DIRECTION_P95_LIMIT_DEG
            ),
            "collision_pass": bool(
                fk_metrics["upper_body_collision_frame_rate"]
                <= COLLISION_FRAME_RATE_LIMIT
                and fk_metrics["upper_body_collision_max_consecutive_frames"]
                <= COLLISION_CONSECUTIVE_LIMIT
            ),
            "saturation_pass": bool(
                trajectory_metrics["observed_joint_max_saturation_fraction"]
                <= SATURATION_FRACTION_LIMIT
                and trajectory_metrics[
                    "observed_joint_max_consecutive_saturation_frames"
                ]
                <= SATURATION_CONSECUTIVE_LIMIT
            ),
            "head_tilt_proxy_pass": (
                head_error_p95_deg <= HEAD_PROXY_DIRECTION_P95_LIMIT_DEG
            ),
        }
        gate["passed"] = all(gate.values())

        evaluation_payload = (
            dict(human_evaluation) if human_evaluation is not None else None
        )
        report: dict[str, Any] = {
            "schema_version": "1.0.0",
            "artifact_kind": "hanyang_partial_18d_retarget_quality_v1",
            "dataset": DATASET,
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "dataset_license": DATASET_LICENSE,
            "clip_id": source["clip_id"],
            "participant_id": source["participant_id"],
            "block_id": source["block_id"],
            "trial_id": source["trial_id"],
            "emotion_index": source["emotion_index"],
            "emotion_id": source["emotion_id"],
            "fixed_split_assignment": source["fixed_split_assignment"],
            "speaker_key": source["speaker_key"],
            "source_group_key": source["source_group_key"],
            "source_csv": source["path"],
            "source_sha256": source["source_sha256"],
            "source_frames": SOURCE_FRAMES,
            "source_fps": SOURCE_FPS,
            "source_sample_span_sec": (SOURCE_FRAMES - 1) / SOURCE_FPS,
            "source_frame_coverage_sec": SOURCE_FRAMES / SOURCE_FPS,
            "source_joint_order": list(SOURCE_JOINTS),
            "axis_policy": AXIS_POLICY,
            "source_to_robot_basis": SOURCE_TO_ROBOT_BASIS.tolist(),
            "robot_urdf": str(self.urdf),
            "robot_urdf_sha256": sha256_file(self.urdf),
            "gmr_config": str(self.config),
            "gmr_config_sha256": sha256_file(self.config),
            "output_contract": ULA_V2_18D_CONTRACT,
            "joint_order": list(JOINT_ORDER_18D),
            "action_dim": len(JOINT_ORDER_18D),
            "action_dim_mask": list(ACTION_DIM_MASK_18D),
            "unobserved_joints": sorted(UNOBSERVED_18D_JOINTS),
            "wrist_observation_policy": WRIST_OBSERVATION_POLICY,
            "head_observation_policy": HEAD_OBSERVATION_POLICY,
            "per_frame_observation_confidence": {
                "path": str(confidence_npy),
                "sha256": sha256_file(confidence_npy),
                "shape": list(frame_confidence.shape),
                "minimum": float(frame_confidence.min()),
                "mean": float(frame_confidence.mean()),
                "maximum": float(frame_confidence.max()),
            },
            "emotion_evaluation": evaluation_payload,
            "source_geometry": source_quality,
            "smoothing": smoothing,
            "trajectory": trajectory_metrics,
            "effective_velocity_limits_rad_s": effective_velocity_limits,
            "mean_final_ik_objective": float(np.mean(ik_errors)),
            **fk_metrics,
            "head_tilt_proxy_error_mean_deg": float(
                np.rad2deg(head_error.mean())
            ),
            "head_tilt_proxy_error_p95_deg": head_error_p95_deg,
            "head_tilt_proxy_error_max_deg": float(
                np.rad2deg(head_error.max())
            ),
            "quality_gate": gate,
            "outputs": {
                "raw_csv": str(raw_csv),
                "raw_csv_sha256": sha256_file(raw_csv),
                "source_faithful_partial_18d_csv": str(faithful_csv),
                "source_faithful_partial_18d_csv_sha256": sha256_file(
                    faithful_csv
                ),
                "deployment_safe_partial_18d_csv": str(preview_csv),
                "deployment_safe_partial_18d_csv_sha256": sha256_file(
                    preview_csv
                ),
            },
            "deployment_preview": {
                "optimization_eligible": False,
                "emotion_timing_supervision_eligible": False,
                "reason": "per_joint_velocity_time_dilation_changes_emotional_timing",
                "frames": int(len(preview)),
                "fps": SOURCE_FPS,
                "retime_factor": float(len(preview) / SOURCE_FRAMES),
                "frame_coverage_sec": float(len(preview) / SOURCE_FPS),
                "max_velocity_limit_ratio": trajectory_quality(
                    preview,
                    fps=SOURCE_FPS,
                    velocity_limits=velocity_limits,
                )["max_velocity_limit_ratio"],
                "limb_target_error_p95_m": preview_fk_metrics[
                    "limb_target_error_p95_m"
                ],
            },
            "admission": {
                "external_retarget_pool_eligible": bool(gate["passed"]),
                "emotion_critic_candidate": bool(
                    gate["passed"] and evaluation_payload is not None
                ),
                "hard_emotion_supervision_candidate": bool(
                    gate["passed"]
                    and evaluation_payload
                    and evaluation_payload.get("intended_high_confidence")
                ),
                "generator_training_eligible": False,
                "generator_blockers": [
                    "partial_18d_loader_with_per_frame_confidence_not_yet_qualified",
                    "robot_observable_emotion_blind_review_pending",
                    "cross_domain_bridge_gate_pending",
                ],
                "current_beat2_training_mutated": False,
                "foundation_ingest_allowed": False,
            },
            "processing_sec": float(time.perf_counter() - started),
        }
        report["record_sha256"] = json_hash(
            {key: value for key, value in report.items() if key != "record_sha256"}
        )
        _atomic_json(quality_json, report)
        return report


def _load_human_evaluation(path: Path | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("human evaluation JSON must contain one object")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runtime = HanyangRetargetRuntime(
        gmr_root=args.gmr_root,
        urdf=args.urdf,
        config=args.config,
        solver=args.solver,
        posture_cost=args.posture_cost,
    )
    report = runtime.retarget(
        args.csv,
        args.output_dir,
        human_evaluation=_load_human_evaluation(args.human_evaluation_json),
        expected_source_sha256=args.expected_source_sha256,
        max_velocity=args.max_velocity,
        smoothing_window=args.smoothing_window,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
