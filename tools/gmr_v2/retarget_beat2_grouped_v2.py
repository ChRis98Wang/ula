#!/usr/bin/env python3
"""In-process BEAT2 source runtime for isolated variable-length 18D events.

The ordinary BEAT2 adapter is intentionally a one-window command.  This
module keeps the expensive SMPL-X model and GMR retargeter alive inside a
worker, loads a source NPZ once, and then treats every semantic event as a
fresh retarget job.  IK state, warmup, smoothing, retiming, and quality
measurement are all event-local.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


EVENT_REVIEW_PROVENANCE_FIELDS = (
    "annotation_kind",
    "language",
    "language_code",
    "canonical_action",
    "canonical_action_role",
    "canonical_prompt",
    "canonical_prompt_role",
    "prompt",
    "prompt_schema",
    "prompt_source",
    "prompt_sha256",
    "prompt_contract",
    "behavior_id",
    "behavior_review_status",
    "behavior_supervision_mask",
    "behavior_source",
    "behavior_mapping_contract",
    "official_category_verified",
    "official_category_role",
    "official_category_condition_channel",
    "official_category_loss",
    "official_category_conditioning_enabled",
    "robot_observable_motion_form",
    "communicative_intent",
    "semantic_supervision_masks",
    "semantic_mapping_status",
    "emotion_id",
    "emotion_review_status",
    "emotion_supervision_mask",
    "source_emotion_label_verified",
    "emotion_supervision_role",
    "official_emotion_conditioning_enabled",
    "official_emotion_condition_channel",
    "official_emotion_loss",
    "affect_observable_review_status",
    "affect_observable_supervision_mask",
    "emotion_source",
    "emotion_protocol_contract",
    "semantic_gesture",
    "fixed_split_assignment",
    "pilot_split",
    "inventory_manifest_sha256",
    "pilot_selector_contract_sha256",
    "pilot_source_group_sha256",
    "pilot_speaker_group_sha256",
    "motion_sha256",
    "inventory_record_sha256",
    "upstream_inventory_record_sha256",
    "selected_record_sha256",
    "selected_record_sha256_role",
    "upstream_inventory_manifest_sha256",
    "retarget_input_manifest_sha256",
    "retarget_input_manifest_sha256_role",
    "training_pool_selection_status",
    "training_pool_contract_sha256",
    "qc_replacement_round",
    "qc_replacement_for_stratum",
    "qc_replacement_selection_status",
    "qc_replacement_contract_sha256",
    "training_admission_status",
)
RETARGET_SEGMENT_REPRESENTATION = (
    "native_variable_length_semantic_event_retimed_30hz_v1"
)
MOTION_FOUNDATION_ANNOTATION_KIND = "motion_foundation_unlabeled_contiguous_chunk"
MOTION_FOUNDATION_RETARGET_SEGMENT_REPRESENTATION = (
    "motion_foundation_contiguous_chunk_retimed_30hz_v1"
)


def build_retarget_segment_contract(
    task: dict[str, Any], *, source_frame_count: int, output_frame_count: int, fps: float
) -> dict[str, Any]:
    payload = {
        "representation": (
            MOTION_FOUNDATION_RETARGET_SEGMENT_REPRESENTATION
            if task.get("annotation_kind") == MOTION_FOUNDATION_ANNOTATION_KIND
            else RETARGET_SEGMENT_REPRESENTATION
        ),
        "source_start_frame": int(task["start_frame"]),
        "source_end_frame_exclusive": int(task["end_frame_exclusive"]),
        "source_frame_count": int(source_frame_count),
        "source_frame_coverage_sec": float(source_frame_count / fps),
        "output_frame_count": int(output_frame_count),
        "output_sample_span_sec": float(max(0, output_frame_count - 1) / fps),
        "output_frame_coverage_sec": float(output_frame_count / fps),
        "fps": float(fps),
        "retimed": int(output_frame_count) != int(source_frame_count),
        "cropped": False,
        "planner_duration_field": "output_sample_span_sec",
        "source_boundary_duration_field": "source_frame_coverage_sec",
        "legacy_quality_duration_sec_role": (
            "output_frame_coverage_compatibility_only_not_planner_target"
        ),
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return {**payload, "sha256": digest}

try:
    from . import retarget_motionx322_v2 as motionx
    from .retarget_beat2_v2 import (
        BEAT2_AXIS_POLICY,
        EXPECTED_MODEL_SHA256,
        JOINT_LIMITS_18D,
        JOINT_ORDER_18D,
        ULA_V2_18D_CONTRACT,
        axis_direction_metrics,
        build_beat2_provenance,
        canonical_head_relative_rotations,
        canonical_source_data,
        decode_beat2_smplx,
        decompose_v2_head_rotations,
        enforce_human_elbow_branch,
        enforce_safe_elbow_branch,
        extract_joint_row,
        head_quality_metrics,
        prepare_target_builder,
        quality_report,
        rendered_pose_metrics,
        retime_targets,
        sha256,
        smooth_and_limit,
        write_csv,
    )
except ImportError:  # pragma: no cover - direct invocation by path
    import retarget_motionx322_v2 as motionx
    from retarget_beat2_v2 import (
        BEAT2_AXIS_POLICY,
        EXPECTED_MODEL_SHA256,
        JOINT_LIMITS_18D,
        JOINT_ORDER_18D,
        ULA_V2_18D_CONTRACT,
        axis_direction_metrics,
        build_beat2_provenance,
        canonical_head_relative_rotations,
        canonical_source_data,
        decode_beat2_smplx,
        decompose_v2_head_rotations,
        enforce_human_elbow_branch,
        enforce_safe_elbow_branch,
        extract_joint_row,
        head_quality_metrics,
        prepare_target_builder,
        quality_report,
        rendered_pose_metrics,
        retime_targets,
        sha256,
        smooth_and_limit,
        write_csv,
    )


@dataclass(frozen=True)
class GroupedRuntimeConfig:
    smplx_model: Path
    gmr_root: Path
    urdf: Path
    config: Path
    warmup_frames: int = 0
    max_velocity_rad_s: float = 3.0
    smoothing_window: int = 7
    posture_cost: float = 0.02
    solver: str = "daqp"
    neutral_limit_margin_rad: float = 0.0


def interiorize_neutral_qpos(
    model,
    mujoco,
    neutral_qpos: np.ndarray,
    margin_rad: float,
) -> np.ndarray:
    """Move exact-bound hinge initializers into the feasible-set interior.

    Some QP backends reject a value that is numerically equal to a joint
    limit after MuJoCo/Mink conversions.  The opt-in margin changes only the
    event-local IK initializer; output trajectories still have to pass the
    unchanged physical quality gates.
    """

    margin = float(margin_rad)
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("neutral_limit_margin_rad must be finite and non-negative")
    result = np.asarray(neutral_qpos, dtype=np.float64).copy()
    if margin == 0.0:
        return result
    for joint_name in JOINT_ORDER_18D:
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        if joint_id < 0 or not bool(model.jnt_limited[joint_id]):
            continue
        lower, upper = (float(value) for value in model.jnt_range[joint_id])
        if upper - lower <= 2.0 * margin:
            raise ValueError(
                f"neutral limit margin {margin} is too large for {joint_name}"
            )
        qpos_address = int(model.jnt_qposadr[joint_id])
        result[qpos_address] = np.clip(
            result[qpos_address], lower + margin, upper - margin
        )
    return result


def slice_decoded_source(
    decoded: dict[str, Any], start_frame: int, end_frame_exclusive: int
) -> dict[str, Any]:
    """Return one event without reopening or joining the source container."""
    total = int(decoded["source_total_frames"])
    if (
        isinstance(start_frame, bool)
        or isinstance(end_frame_exclusive, bool)
        or not isinstance(start_frame, (int, np.integer))
        or not isinstance(end_frame_exclusive, (int, np.integer))
    ):
        raise ValueError("Semantic event frame bounds must be integers")
    start = int(start_frame)
    end = int(end_frame_exclusive)
    if start < 0 or end > total or end - start < 3:
        raise ValueError(
            f"Invalid semantic event [{start}, {end}) for {total} source frames; "
            "at least three frames are required"
        )
    result = dict(decoded)
    result.update(
        {
            "root_orient": decoded["root_orient"][start:end].copy(),
            "pose_body": decoded["pose_body"][start:end].copy(),
            "trans": decoded["trans"][start:end].copy(),
            "betas": decoded["betas"].copy(),
            "frame_count": end - start,
            "source_start_frame": start,
            "source_end_frame": end,
        }
    )
    return result


def forward_smplx_with_loaded_model(decoded: dict[str, Any], model):
    """Run one event through an already constructed SMPL-X model."""
    frame_count = int(decoded["frame_count"])
    torch = motionx.torch
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
            return_verts=False,
        )
    parent_count = len(model.parents)
    joints = output.joints[:, :parent_count].detach().cpu().numpy()
    local = output.full_pose.detach().cpu().numpy().reshape(frame_count, -1, 3)
    parents = model.parents.detach().cpu().numpy()
    return joints, local[:, :parent_count], parents


class GroupedBeat2RetargetRuntime:
    """Reusable worker runtime with strict event reset boundaries."""

    def __init__(self, config: GroupedRuntimeConfig):
        self.config = config
        for path in (
            config.smplx_model,
            config.gmr_root,
            config.urdf,
            config.config,
        ):
            if not Path(path).exists():
                raise FileNotFoundError(path)
        if config.warmup_frames < 0:
            raise ValueError("warmup_frames cannot be negative")
        if config.max_velocity_rad_s <= 0:
            raise ValueError("max_velocity_rad_s must be positive")

        self.model_hash = sha256(config.smplx_model)
        if self.model_hash != EXPECTED_MODEL_SHA256:
            raise ValueError(
                f"Unexpected SMPL-X model SHA256 {self.model_hash}; "
                f"expected {EXPECTED_MODEL_SHA256}"
            )
        self.smplx_model = motionx.SMPLX(
            str(config.smplx_model), gender="neutral", use_pca=False
        )
        self.smplx_model.eval()
        self.retargeter, self.mujoco, self.mink = motionx.configure_retargeter(
            config.gmr_root, config.urdf, config.config, config.solver
        )
        enforce_human_elbow_branch(self.retargeter, self.mujoco, self.mink)
        self.posture_task = None
        if config.posture_cost > 0:
            self.posture_task = self.mink.PostureTask(
                self.retargeter.model,
                cost=config.posture_cost,
                lm_damping=1.0,
            )
            self.retargeter.tasks1.append(self.posture_task)

        self.source_path: Path | None = None
        self.source_hash: str | None = None
        self.decoded_source: dict[str, Any] | None = None
        self.source_load_count = 0
        self.event_reset_count = 0
        self.current_source_event_reset_count = 0
        self.current_event_limit_margin_interventions = 0

    def load_source(self, source_path: Path) -> None:
        """Load exactly one source container for a grouped worker call."""
        source = Path(source_path).resolve()
        self.source_path = source
        self.source_hash = sha256(source)
        self.decoded_source = decode_beat2_smplx(source)
        self.source_load_count += 1
        self.current_source_event_reset_count = 0

    def reset_event(self, task: dict[str, Any]) -> dict[str, Any]:
        """Create event-local targets and reset IK before every event."""
        if self.source_path is None or self.decoded_source is None:
            raise RuntimeError("load_source must be called before reset_event")
        if Path(task["source"]).resolve() != self.source_path:
            raise ValueError("Task source does not match the currently loaded source")
        fps = float(self.decoded_source["mocap_frame_rate"])
        if not np.isclose(float(task["fps"]), fps, rtol=0.0, atol=1e-6):
            raise ValueError(
                f"Inventory fps {task['fps']} does not match source fps {fps}"
            )

        decoded = slice_decoded_source(
            self.decoded_source,
            task["start_frame"],
            task["end_frame_exclusive"],
        )
        joints, local_rotvecs, parents = forward_smplx_with_loaded_model(
            decoded, self.smplx_model
        )
        positions, quaternions, alignment = canonical_source_data(
            joints, local_rotvecs, parents
        )
        head_relative_rotations = canonical_head_relative_rotations(
            local_rotvecs, parents, alignment
        )
        raw_head = decompose_v2_head_rotations(head_relative_rotations)
        neutral_qpos, target_for_frame = prepare_target_builder(
            self.retargeter.model, self.mujoco, positions, quaternions
        )
        neutral_qpos = interiorize_neutral_qpos(
            self.retargeter.model,
            self.mujoco,
            neutral_qpos,
            self.config.neutral_limit_margin_rad,
        )

        # This is the hard event boundary. No configuration or warm-start state
        # from the preceding event is allowed to seed the current event.
        self.retargeter.configuration.update(neutral_qpos)
        self.current_event_limit_margin_interventions = 0
        if self.posture_task is not None:
            self.posture_task.set_target(neutral_qpos)
        first_target = target_for_frame(0)
        for _ in range(20 + self.config.warmup_frames):
            self._enforce_configuration_limit_margin()
            self.retargeter.retarget(first_target)

        self.event_reset_count += 1
        self.current_source_event_reset_count += 1
        return {
            "decoded": decoded,
            "fps": fps,
            "alignment": alignment,
            "head_relative_rotations": head_relative_rotations,
            "raw_head": raw_head,
            "target_for_frame": target_for_frame,
            "event_reset_ordinal": self.current_source_event_reset_count,
            "neutral_limit_margin_rad": self.config.neutral_limit_margin_rad,
        }

    def _enforce_configuration_limit_margin(self) -> None:
        if self.config.neutral_limit_margin_rad == 0.0:
            return
        current = self.retargeter.configuration.data.qpos.copy()
        interior = interiorize_neutral_qpos(
            self.retargeter.model,
            self.mujoco,
            current,
            self.config.neutral_limit_margin_rad,
        )
        if not np.array_equal(interior, current):
            self.retargeter.configuration.update(interior)
            self.current_event_limit_margin_interventions += 1

    def retarget_event(
        self,
        task: dict[str, Any],
        event: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        """Retarget, smooth, and quality-check one event in isolation."""
        if self.source_path is None or self.source_hash is None:
            raise RuntimeError("No source is loaded")
        started = time.perf_counter()
        decoded = event["decoded"]
        fps = float(event["fps"])
        target_for_frame = event["target_for_frame"]

        raw_rows: list[np.ndarray] = []
        output_targets: list[dict[str, Any]] = []
        ik_errors: list[float] = []
        for frame_index in range(int(decoded["frame_count"])):
            target = target_for_frame(frame_index)
            self._enforce_configuration_limit_margin()
            qpos = self.retargeter.retarget(target)
            raw_rows.append(
                extract_joint_row(self.retargeter.model, qpos, self.mujoco)
            )
            output_targets.append(target)
            ik_errors.append(float(self.retargeter.error1()))

        raw_body = np.asarray(raw_rows)
        raw = np.column_stack((raw_body, event["raw_head"]))
        safe, key_times, output_times = smooth_and_limit(
            raw,
            fps,
            self.config.max_velocity_rad_s,
            self.config.smoothing_window,
            joint_order=JOINT_ORDER_18D,
            joint_limits=JOINT_LIMITS_18D,
        )
        safe = enforce_safe_elbow_branch(safe, joint_order=JOINT_ORDER_18D)
        safe_targets = retime_targets(output_targets, key_times, output_times)

        output_dir.mkdir(parents=True, exist_ok=False)
        stem = (
            f"{self.source_path.stem}_f{decoded['source_start_frame']:06d}-"
            f"{decoded['source_end_frame']:06d}"
        )
        raw_csv = output_dir / f"{stem}_gmr_raw_18d.csv"
        safe_csv = output_dir / f"{stem}_gmr_safe_18d.csv"
        write_csv(raw_csv, raw, joint_order=JOINT_ORDER_18D)
        write_csv(safe_csv, safe, joint_order=JOINT_ORDER_18D)

        raw_pose_metrics = rendered_pose_metrics(
            self.retargeter.model,
            self.mujoco,
            raw,
            output_targets,
            joint_order=JOINT_ORDER_18D,
        )
        pose_metrics = rendered_pose_metrics(
            self.retargeter.model,
            self.mujoco,
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

        metadata = build_beat2_provenance(
            self.source_path, decoded, self.source_hash
        )
        metadata.update(
            {
                field: task[field]
                for field in EVENT_REVIEW_PROVENANCE_FIELDS
                if field in task
            }
        )
        metadata.update(
            {
                "source_dataset": "BEAT2 English v2.0.0",
                "source_dataset_id": "beat_english_v2.0.0",
                "dataset_subset": task.get("dataset_subset"),
                "inventory_record_sha256": task.get("inventory_record_sha256"),
                "upstream_inventory_record_sha256": task.get(
                    "upstream_inventory_record_sha256"
                ),
                "selected_record_sha256": task.get("selected_record_sha256"),
                "retarget_input_manifest_sha256": task.get(
                    "retarget_input_manifest_sha256"
                ),
                "semantic_event": task.get("semantic_event"),
                "official_semantic_event": task.get("official_semantic_event"),
                "official_gesture_semantic_spans": task.get(
                    "official_gesture_semantic_spans"
                ),
                "training_segment": task.get("training_segment"),
                "semantic_label_status": task.get("semantic_label_status"),
                "emotion_id": task.get("emotion_id"),
                "emotion_supervision_mask": task.get(
                    "emotion_supervision_mask"
                ),
                "emotion_label_status": task.get("emotion_label_status"),
                "smplx_model": str(self.config.smplx_model.resolve()),
                "smplx_model_sha256": self.model_hash,
                "smplx_model_sha256_status": "verified",
                "smplx_model_revision": (
                    "a57d1dfb1162c2a9cc20013f0ab212c21f211e78"
                ),
                "robot_urdf": str(self.config.urdf.resolve()),
                "frames": int(len(safe)),
                "fps": fps,
                "duration_sec": float(len(safe) / fps),
                "source_window_duration_sec": float(decoded["frame_count"] / fps),
                "retarget_segment": build_retarget_segment_contract(
                    task,
                    source_frame_count=int(decoded["frame_count"]),
                    output_frame_count=int(len(safe)),
                    fps=fps,
                ),
                "retime_factor": float(len(safe) / len(raw)),
                "max_velocity_rad_s": float(self.config.max_velocity_rad_s),
                "posture_cost": float(self.config.posture_cost),
                "mean_final_ik_objective": float(np.mean(ik_errors)),
                "anatomical_alignment_matrix": event["alignment"].tolist(),
                "anatomical_alignment_determinant": float(
                    np.linalg.det(event["alignment"])
                ),
                "axis_policy": BEAT2_AXIS_POLICY,
                "output_contract": ULA_V2_18D_CONTRACT,
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
                "grouped_execution": {
                    "source_npz_loads_for_group": 1,
                    "smplx_model_scope": "one_cached_instance_per_worker_process",
                    "gmr_runtime_scope": "one_cached_instance_per_worker_process",
                    "event_reset_ordinal": event["event_reset_ordinal"],
                    "ik_reset": "neutral_qpos_before_each_event",
                    "neutral_limit_margin_rad": float(
                        event["neutral_limit_margin_rad"]
                    ),
                    "limit_margin_interventions": int(
                        self.current_event_limit_margin_interventions
                    ),
                    "warmup_iterations_per_event": 20
                    + self.config.warmup_frames,
                    "smoothing_scope": "single_event_only_no_cross_event_frames",
                    "quality_scope": "single_event_only",
                },
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
            self.config.max_velocity_rad_s,
            pose_metrics,
            metadata,
            joint_order=JOINT_ORDER_18D,
            joint_limits=JOINT_LIMITS_18D,
        )
        direction = axis_direction_metrics(
            safe, event["alignment"], joint_order=JOINT_ORDER_18D
        )
        direction["axis_policy"] = BEAT2_AXIS_POLICY
        report.update(direction)
        report["quality_gate"]["axis_direction_pass"] = direction[
            "axis_direction_pass"
        ]
        head = head_quality_metrics(
            event["head_relative_rotations"],
            raw,
            safe,
            fps,
            self.config.max_velocity_rad_s,
            joint_order=JOINT_ORDER_18D,
        )
        report.update(head)
        for key in (
            "head_joint_limits_pass",
            "head_velocity_pass",
            "head_direction_pass",
            "head_continuity_pass",
        ):
            report["quality_gate"][key] = head[key]
        report["quality_gate"]["passed"] = all(
            value
            for key, value in report["quality_gate"].items()
            if key != "passed"
        )
        return report
