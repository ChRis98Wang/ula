#!/usr/bin/env python3
"""Build GT | V8.1 control | V8.1 safe-Hanyang treatment at native speed.

This builder is deliberately fail-closed.  It accepts only the two completed
60k formal arms sealed by the retry4 supervisor and samples only their
``best_admissible`` checkpoints.  A technical smoke or an in-progress best
checkpoint is never eligible.

The left lane reuses the exact 24 held-out BEAT2 robot-GT events selected
before the earlier Qwen comparison.  All lanes use each event's native frame
count, the same prompt, seed, and initial noise.  There is no smoothing,
padding, endpoint hold, time warp, crop, or boundary blend.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import supervise_hanyang_beat2_emotion_preserving_v81 as supervisor  # noqa: E402
from tools import train_hanyang_beat2_emotion_preserving_v81 as trainer  # noqa: E402
from tools.experimental import build_beat2_clean_abc_video as abc_video  # noqa: E402
from tools.experimental import (  # noqa: E402
    build_beat2_emotion_hierarchy_v7_60s as v7_video,
)
from tools.experimental import (  # noqa: E402
    build_beat2_emotion_hierarchy_v7_qwen_ab_gt_60s as prior_gt,
)
from tools.experimental import (  # noqa: E402
    build_beat2_style_emotion_v2_60s as style_video,
)
from upper_body_skeleton.ula_v2_18d_head import (  # noqa: E402
    JOINT_ORDER_18D,
)


SCHEMA_VERSION = 1
CONFIG_ARTIFACT_KIND = (
    "hanyang_beat2_emotion_preserving_v81_ab_gt_60s_config_v1"
)
PLAN_ARTIFACT_KIND = (
    "hanyang_beat2_emotion_preserving_v81_ab_gt_60s_plan_v1"
)
SUMMARY_ARTIFACT_KIND = (
    "hanyang_beat2_emotion_preserving_v81_ab_gt_60s_summary_v1"
)
DATA_POLICY = (
    "held_out_beat2_gt_vs_v81_control_and_safe323_hanyang_treatment_v1"
)
CONTROL_ARM = "winner_control_0pct_hanyang"
TREATMENT_ARM = "winner_isolated_5pct_hanyang"
ARMS = (CONTROL_ARM, TREATMENT_ARM)
ARM_LABELS = {
    CONTROL_ARM: "CONTROL · 0% HANYANG",
    TREATMENT_ARM: "TREATMENT · 5% SAFE323 HANYANG",
}
ARM_SHORT = {
    CONTROL_ARM: "C",
    TREATMENT_ARM: "T",
}
FPS = 30.0
EXPECTED_FRAMES = 1800
EXPECTED_EVENTS = 24
EXPECTED_FORMAL_STEPS = 60_000
ACTION_DIM = 18
CONDITION_DIM = 264
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "hanyang_beat2_emotion_preserving_v81_ab_gt_60s.json"
)
PLAN_FILENAME = "prepared_plan_v8_1.json"
SUMMARY_FILENAME = "summary_v8_1.json"
CLAIM_INVALIDATION_FILENAME = "CLAIM_INVALIDATION.json"
FINAL_VIDEO_FILENAME = (
    "BEAT2_GT_vs_V81_control_vs_safe323_Hanyang_treatment_60s.mp4"
)
TRAJECTORY_FILENAME = "comparison_trajectories_v8_1.npz"
ASS_FILENAME = "text_emotion_jerk_expression_response_v8_1.ass"
MATCHED_AUDIT_PURPOSE = (
    "matched_step_deterministic_replay_not_formal_candidate"
)
MATCHED_AUDIT_SUMMARY_ARTIFACT_KIND = (
    "hanyang_beat2_emotion_preserving_v81_matched_step_gt_60s_summary_v1"
)
MATCHED_AUDIT_DATA_POLICY = (
    "held_out_beat2_gt_vs_v81_matched_step1500_audit_replay_v1"
)
MATCHED_AUDIT_SUMMARY_FILENAME = "summary_matched_step1500_v1.json"
MATCHED_AUDIT_FINAL_VIDEO_FILENAME = (
    "BEAT2_GT_vs_V81_control_vs_safe323_Hanyang_matched_step1500_60s.mp4"
)
COMPARISON_CONTRACT = {
    "same_held_out_gt_events": True,
    "same_prompt": True,
    "same_seed": True,
    "same_initial_noise": True,
    "gt_loaded_after_both_generator_branches": True,
    "gt_used_as_generation_input": False,
    "generation_frames_equal_gt_event_frames": True,
    "planner_duration_diagnostic_only": True,
    "formal_best_admissible_only": True,
    "smoke_checkpoints_allowed": False,
    "temporal_padding_frames": 0,
    "static_padding_frames": 0,
    "endpoint_hold": False,
    "additional_smoothing": False,
    "time_warp": False,
    "network_frame_crop": False,
    "boundary_blend": False,
    "forced_last_frame_blend": False,
    "forced_return_to_zero": False,
}
MATCHED_AUDIT_COMPARISON_CONTRACT = {
    **COMPARISON_CONTRACT,
    "formal_best_admissible_only": False,
    "matched_step_audit_replay_only": True,
    "audit_checkpoints_are_formal_candidates": False,
    "deterministic_replay_must_equal_original_log": True,
}
METRIC_CONTRACT = {
    "jerk": (
        "raw_segment_rms_rad_s3_excluding_endpoint_hold_and_event_boundaries"
    ),
    "expression": (
        "raw_segment_joint_range_rms_rad_proxy_not_emotion_accuracy"
    ),
    "head_activity": "raw_segment_head_velocity_rms_rad_s",
    "emotion_response": (
        "formal_best_checkpoint_fixed_beat2_diagnostic_retention_"
        "not_per_event_accuracy"
    ),
}


class V81ComparisonError(RuntimeError):
    """Raised when the formal V8.1 comparison contract cannot be proved."""


def _reject_claim_invalidated(output_dir: str | Path) -> None:
    receipt_path = Path(output_dir) / CLAIM_INVALIDATION_FILENAME
    if not receipt_path.is_file():
        return
    receipt = _load_json(receipt_path, field="claim invalidation receipt")
    status = receipt.get("artifact_status")
    eligibility = receipt.get("claim_eligibility") or {}
    if (
        status
        != "invalidated_for_hanyang_benefit_and_text_semantic_claims"
        or eligibility.get("hanyang_training_benefit") is not False
        or eligibility.get("text_to_motion_semantic_alignment") is not False
        or eligibility.get("emotion_accuracy") is not False
    ):
        raise V81ComparisonError("malformed claim invalidation receipt")
    raise V81ComparisonError(
        "completed V8.1 output is invalidated for Hanyang benefit and "
        "text-semantic claims"
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _load_json(path: Path, *, field: str) -> dict[str, Any]:
    if not path.is_file():
        raise V81ComparisonError(f"{field} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise V81ComparisonError(f"{field} is not valid JSON") from error
    if not isinstance(value, dict):
        raise V81ComparisonError(f"{field} must be a JSON object")
    return value


def _resolve_path(
    value: object,
    *,
    field: str,
    must_exist: bool,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise V81ComparisonError(f"{field} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if "kimodo" in str(path).casefold() or "komodo" in str(path).casefold():
        raise V81ComparisonError(f"{field} contains forbidden Kimodo lineage")
    if must_exist and not path.is_file():
        raise V81ComparisonError(f"{field} is missing: {path}")
    return path


def _validate_file_pin(
    path_value: object,
    sha_value: object,
    *,
    field: str,
) -> Path:
    path = _resolve_path(path_value, field=field, must_exist=True)
    if not _is_sha256(sha_value) or abc_video.sha256_file(path) != sha_value:
        raise V81ComparisonError(f"{field} SHA256 mismatch")
    return path


def _reject_source_bearing_kimodo(
    value: object,
    *,
    field: str = "root",
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in {"deny_policy", "kimodo_admitted_count"}:
                continue
            if key_text in {
                "model_state_dict",
                "qwen_style_head_state_dict",
            }:
                continue
            _reject_source_bearing_kimodo(
                child, field=f"{field}.{key_text}"
            )
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, child in enumerate(value):
            _reject_source_bearing_kimodo(
                child, field=f"{field}[{index}]"
            )
        return
    if not isinstance(value, str):
        return
    folded = value.casefold()
    if "kimodo" not in folded and "komodo" not in folded:
        return
    leaf = field.rsplit(".", 1)[-1].casefold()
    source_bearing = any(
        token in leaf
        for token in (
            "path",
            "source",
            "dataset",
            "manifest",
            "checkpoint",
            "cache",
            "input",
            "output",
        )
    )
    if source_bearing or "/" in value or "\\" in value:
        raise V81ComparisonError(
            f"{field} contains forbidden Kimodo lineage"
        )


def _validate_gt_plan(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_canonical_sha256: str,
) -> dict[str, Any]:
    if abc_video.sha256_file(path) != expected_file_sha256:
        raise V81ComparisonError("held-out GT plan file SHA256 mismatch")
    plan = _load_json(path, field="held-out GT plan")
    unsigned = deepcopy(plan)
    claimed = unsigned.pop("sha256", None)
    if (
        claimed != expected_canonical_sha256
        or abc_video.value_sha256(unsigned) != claimed
        or plan.get("artifact_kind") != prior_gt.PLAN_ARTIFACT_KIND
        or plan.get("schema_version") != prior_gt.SCHEMA_VERSION
        or plan.get("no_kimodo") is not True
        or plan.get("no_hanyang") is not True
        or plan.get("expected_frames") != EXPECTED_FRAMES
        or plan.get("event_count") != EXPECTED_EVENTS
        or plan.get("static_padding_frames") != 0
        or plan.get("endpoint_hold_frames") != 0
        or plan.get("gt_used_as_generation_input") is not False
        or plan.get("gt_action_values_loaded_before_generation") is not False
    ):
        raise V81ComparisonError("held-out GT plan contract changed")
    selections = plan.get("selections")
    if not isinstance(selections, list) or len(selections) != EXPECTED_EVENTS:
        raise V81ComparisonError("held-out GT selection count changed")
    expected_start = 0
    emotions: set[str] = set()
    clip_ids: set[str] = set()
    for index, selection in enumerate(selections, start=1):
        if not isinstance(selection, Mapping):
            raise V81ComparisonError("held-out GT selection is invalid")
        frames = int(selection.get("frames", -1))
        reference = selection.get("reference")
        prompt = str(selection.get("prompt", "")).strip()
        emotion = str(selection.get("target_emotion", ""))
        if (
            selection.get("index") != index
            or selection.get("start_frame") != expected_start
            or selection.get("end_frame") != expected_start + frames
            or not 4 <= frames <= prior_gt.MAXIMUM_EVENT_FRAMES
            or prior_gt.target_emotion_from_prompt(prompt) != emotion
            or not isinstance(reference, Mapping)
            or reference.get("fixed_split_assignment") != "test"
            or not _is_sha256(
                reference.get("declared_safe_csv_sha256")
            )
            or not isinstance(reference.get("clip_id"), str)
        ):
            raise V81ComparisonError(
                f"held-out GT event {index} contract changed"
            )
        expected_start += frames
        emotions.add(emotion)
        clip_ids.add(str(reference["clip_id"]))
    if (
        expected_start != EXPECTED_FRAMES
        or len(clip_ids) != EXPECTED_EVENTS
        or emotions != {"neutral", "sad", "happy", "angry", "surprise", "fear"}
    ):
        raise V81ComparisonError("held-out GT timeline changed")
    manifest = _resolve_path(
        plan.get("source_manifest"),
        field="held-out GT source manifest",
        must_exist=True,
    )
    if abc_video.sha256_file(manifest) != plan.get(
        "source_manifest_sha256"
    ):
        raise V81ComparisonError("held-out GT source manifest changed")
    _reject_source_bearing_kimodo(plan, field="held_out_gt_plan")
    return deepcopy(plan)


def read_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    config = _load_json(config_path, field="V8.1 video config")
    required = {
        "schema_version",
        "artifact_kind",
        "data_policy",
        "no_kimodo",
        "kimodo_admitted_count",
        "hanyang_source_count",
        "hanyang_boundary_excluded_count",
        "hanyang_training_eligible_count",
        "hanyang_boundary_admitted_count",
        "base_v7_video_config",
        "expected_base_v7_video_config_sha256",
        "held_out_gt_plan",
        "expected_held_out_gt_plan_file_sha256",
        "expected_held_out_gt_plan_canonical_sha256",
        "promoted_v81_config",
        "expected_promoted_v81_config_sha256",
        "retry4_supervisor_config",
        "expected_retry4_supervisor_config_sha256",
        "retry4_supervisor_receipt",
        "retry4_supervisor_service_unit",
        "formal_steps_per_arm",
        "control_arm",
        "treatment_arm",
        "target_duration_sec",
        "expected_frames",
        "event_count",
        "comparison_contract",
        "metric_contract",
        "gpu_wait_gate",
        "render",
        "output_dir",
    }
    if set(config) != required:
        raise V81ComparisonError("V8.1 video config fields changed")
    if (
        config.get("schema_version") != SCHEMA_VERSION
        or config.get("artifact_kind") != CONFIG_ARTIFACT_KIND
        or config.get("data_policy") != DATA_POLICY
        or config.get("no_kimodo") is not True
        or config.get("kimodo_admitted_count") != 0
        or config.get("hanyang_source_count")
        != trainer.HANYANG_SOURCE_POOL_COUNT
        or config.get("hanyang_boundary_excluded_count")
        != trainer.HANYANG_BOUNDARY_COUNT
        or config.get("hanyang_training_eligible_count")
        != trainer.HANYANG_TRAINING_ELIGIBLE_COUNT
        or config.get("hanyang_boundary_admitted_count") != 0
        or config.get("retry4_supervisor_service_unit")
        != "hanyang-beat2-emotion-preserving-v81-retry4.service"
        or config.get("formal_steps_per_arm") != EXPECTED_FORMAL_STEPS
        or config.get("control_arm") != CONTROL_ARM
        or config.get("treatment_arm") != TREATMENT_ARM
        or not math.isclose(
            float(config.get("target_duration_sec", 0.0)),
            60.0,
            abs_tol=1e-12,
        )
        or config.get("expected_frames") != EXPECTED_FRAMES
        or config.get("event_count") != EXPECTED_EVENTS
        or config.get("comparison_contract") != COMPARISON_CONTRACT
        or config.get("metric_contract") != METRIC_CONTRACT
    ):
        raise V81ComparisonError("V8.1 video identity contract changed")
    base_path = _validate_file_pin(
        config["base_v7_video_config"],
        config["expected_base_v7_video_config_sha256"],
        field="base_v7_video_config",
    )
    gt_plan_path = _validate_file_pin(
        config["held_out_gt_plan"],
        config["expected_held_out_gt_plan_file_sha256"],
        field="held_out_gt_plan",
    )
    promoted_path = _validate_file_pin(
        config["promoted_v81_config"],
        config["expected_promoted_v81_config_sha256"],
        field="promoted_v81_config",
    )
    supervisor_config_path = _validate_file_pin(
        config["retry4_supervisor_config"],
        config["expected_retry4_supervisor_config_sha256"],
        field="retry4_supervisor_config",
    )
    receipt_path = _resolve_path(
        config["retry4_supervisor_receipt"],
        field="retry4_supervisor_receipt",
        must_exist=False,
    )
    output_dir = _resolve_path(
        config["output_dir"], field="output_dir", must_exist=False
    )
    base = v7_video.validate_config(
        _load_json(base_path, field="base V7 video config")
    )
    base.pop("_validated_comparison_interface", None)
    promoted = trainer.read_config(promoted_path)
    supervised = supervisor.read_config(supervisor_config_path)
    gt_plan = _validate_gt_plan(
        gt_plan_path,
        expected_file_sha256=config[
            "expected_held_out_gt_plan_file_sha256"
        ],
        expected_canonical_sha256=config[
            "expected_held_out_gt_plan_canonical_sha256"
        ],
    )
    if (
        promoted.get("training_policy") != trainer.V81_TRAINING_POLICY
        or promoted.get("deny_policy")
        != (
            "kimodo_permanent_hard_deny_raw_cache_normalizer_"
            "split_checkpoint_v1"
        )
        or promoted["qwen_ab_selection_gate"].get("winner_selected") is not True
        or promoted["approval_gate"].get("status") != "approved"
        or set(promoted["winner_overlay_arms"]) - {
            "launch_policy",
            "shared_selected_foundation_sha256",
            "shared_selected_qwen_variant",
            "shared_selected_condition_cache_sha256",
            "shared_seed",
            "shared_noise_seed",
            "shared_split_contract",
            *ARMS,
        }
        or Path(supervised["promoted_config"]).resolve()
        != promoted_path
        or Path(supervised["receipt"]).resolve() != receipt_path
    ):
        raise V81ComparisonError("promoted V8.1/supervisor contract changed")
    for arm in ARMS:
        output = Path(
            promoted["winner_overlay_arms"][arm]["output_dir"]
        ).resolve()
        if output.name != arm:
            raise V81ComparisonError(f"{arm} output directory changed")
    render = config["render"]
    if (
        not isinstance(render, Mapping)
        or set(render)
        != {
            "pane_width",
            "pane_height",
            "panel_width",
            "simplified",
            "backend",
            "fps",
        }
        or min(
            int(render["pane_width"]),
            int(render["pane_height"]),
            int(render["panel_width"]),
        )
        <= 0
        or render["backend"] != "mujoco_egl"
        or int(render["fps"]) != int(FPS)
        or not isinstance(render["simplified"], bool)
    ):
        raise V81ComparisonError("render contract changed")
    gpu = config["gpu_wait_gate"]
    if (
        not isinstance(gpu, Mapping)
        or set(gpu)
        != {
            "index",
            "minimum_free_memory_mib",
            "required_consecutive_passes",
            "unknown_compute_policy",
        }
        or gpu.get("index") != 0
        or float(gpu.get("minimum_free_memory_mib", 0)) < 24576
        or gpu.get("required_consecutive_passes") != 2
        or gpu.get("unknown_compute_policy") != "wait_never_kill"
    ):
        raise V81ComparisonError("GPU wait gate changed")
    _reject_source_bearing_kimodo(config, field="config")
    return {
        **deepcopy(config),
        "_config_path": str(config_path),
        "_config_sha256": abc_video.sha256_file(config_path),
        "_base": base,
        "_base_path": str(base_path),
        "_gt_plan": gt_plan,
        "_gt_plan_path": str(gt_plan_path),
        "_promoted": promoted,
        "_promoted_path": str(promoted_path),
        "_supervisor": supervised,
        "_supervisor_config_path": str(supervisor_config_path),
        "_supervisor_receipt_path": str(receipt_path),
        "_output_dir": str(output_dir),
    }


def _service_snapshot(unit: str) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "--property=LoadState",
            "--property=ActiveState",
            "--property=SubState",
            "--property=Result",
            "--property=ExecMainStatus",
            "--no-pager",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    values = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    missing_detail = (
        completed.stdout + "\n" + completed.stderr
    ).casefold()
    transient_reclaimed = (
        values.get("LoadState") == "not-found"
        or "not found" in missing_detail
        or "could not be found" in missing_detail
    )
    if completed.returncode != 0 and not transient_reclaimed:
        return {
            "known": False,
            "active": False,
            "failed": False,
            "transient_reclaimed": False,
            "detail": completed.stderr.strip()[-500:],
        }
    active = values.get("ActiveState") == "active"
    failed = (
        values.get("ActiveState") == "failed"
        or values.get("Result") == "failed"
        or (
            values.get("ExecMainStatus", "0") not in {"", "0"}
            and not active
        )
    )
    return {
        "known": True,
        "active": active,
        "failed": failed,
        "transient_reclaimed": transient_reclaimed,
        **values,
    }


def _validate_supervisor_receipt(
    validated: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(validated["_supervisor_receipt_path"])
    receipt = _load_json(path, field="retry4 supervisor receipt")
    if (
        receipt.get("sha256") != supervisor.canonical_sha256(receipt)
        or receipt.get("schema_version") != supervisor.SCHEMA_VERSION
        or receipt.get("artifact_kind")
        != supervisor.RECEIPT_ARTIFACT_KIND
        or receipt.get("status") != "complete"
        or Path(str(receipt.get("config", ""))).resolve()
        != Path(validated["_supervisor_config_path"]).resolve()
        or receipt.get("config_sha256")
        != validated["expected_retry4_supervisor_config_sha256"]
        or receipt.get("stage_order") != list(supervisor.STAGES)
        or receipt.get("formal_arms_sequential") is not True
        or receipt.get("arm_initialization")
        != "same_selected_winner_independent_no_cross_arm_warm_start"
        or receipt.get("boundary_hanyang_admitted_count") != 0
        or receipt.get("kimodo_admitted_count") != 0
    ):
        raise V81ComparisonError("retry4 supervisor receipt changed")
    state_path = _resolve_path(
        receipt.get("state"), field="supervisor receipt state", must_exist=True
    )
    if abc_video.sha256_file(state_path) != receipt.get("state_sha256"):
        raise V81ComparisonError("retry4 supervisor state SHA256 changed")
    stages = receipt.get("stages")
    if not isinstance(stages, Mapping):
        raise V81ComparisonError("retry4 supervisor stages are missing")
    for stage in ("control_formal", "treatment_formal"):
        if (
            not isinstance(stages.get(stage), Mapping)
            or stages[stage].get("status") != "succeeded"
            or not isinstance(stages[stage].get("record"), Mapping)
        ):
            raise V81ComparisonError(f"retry4 {stage} is incomplete")
    return {
        "path": str(path.resolve()),
        "file_sha256": abc_video.sha256_file(path),
        "canonical_sha256": receipt["sha256"],
        "value": receipt,
    }


def _validate_checkpoint_contract(
    checkpoint: Mapping[str, Any],
    *,
    arm: str,
    promoted: Mapping[str, Any],
    summary_best: Mapping[str, Any],
    expected_role: str = "best_admissible",
    expected_candidate_eligible: bool = True,
    expected_step: int | None = None,
    expected_audit_replay: bool = False,
) -> None:
    step = int(checkpoint.get("v8_1_step", -1))
    selected = promoted["qwen_ab_selection_gate"]
    expected_exposure = trainer.expected_exposure(step, arm=arm)
    audit = {
        "hanyang_source_pool_count": trainer.HANYANG_SOURCE_POOL_COUNT,
        "hanyang_boundary_candidate_count": trainer.HANYANG_BOUNDARY_COUNT,
        "hanyang_boundary_excluded_count": trainer.HANYANG_BOUNDARY_COUNT,
        "hanyang_boundary_admitted_count": 0,
        "boundary_hanyang_admitted_count": 0,
        "hanyang_training_eligible_count": (
            trainer.HANYANG_TRAINING_ELIGIBLE_COUNT
        ),
        "hanyang_safe_clip_ids_sha256": trainer.HANYANG_SAFE_CLIP_IDS_SHA256,
        "hanyang_excluded_clip_ids_sha256": (
            trainer.HANYANG_EXCLUDED_CLIP_IDS_SHA256
        ),
        "hanyang_condition_labels_masked": True,
        "kimodo_admitted_count": 0,
    }
    if (
        checkpoint.get("schema_version") != trainer.SCHEMA_VERSION
        or checkpoint.get("artifact_kind")
        != trainer.CHECKPOINT_ARTIFACT_KIND
        or checkpoint.get("arm") != arm
        or checkpoint.get("checkpoint_role") != expected_role
        or checkpoint.get("candidate_eligible")
        is not expected_candidate_eligible
        or checkpoint.get("smoke_test") is not False
        or checkpoint.get("formal_release_eligible") is not False
        or checkpoint.get("target_steps") != EXPECTED_FORMAL_STEPS
        or not 1 <= step <= EXPECTED_FORMAL_STEPS
        or (expected_step is not None and step != int(expected_step))
        or (
            expected_audit_replay
            and (
                checkpoint.get("audit_replay") is not True
                or checkpoint.get("audit_stop_step") != int(expected_step)
                or checkpoint.get("audit_purpose") != MATCHED_AUDIT_PURPOSE
            )
        )
        or (
            not expected_audit_replay
            and checkpoint.get("audit_replay") is True
        )
        or checkpoint.get("architecture")
        != trainer.V7_RUNTIME_MODEL_CONFIG["architecture"]
        or checkpoint.get("action_dim") != ACTION_DIM
        or checkpoint.get("condition_dim") != CONDITION_DIM
        or checkpoint.get("joint_order") != list(JOINT_ORDER_18D)
        or checkpoint.get("condition_policy")
        != trainer.V7_CONDITION_POLICY
        or checkpoint.get("training_policy")
        != trainer.V81_TRAINING_POLICY
        or checkpoint.get("selected_qwen_variant")
        != selected["selected_qwen_variant"]
        or checkpoint.get("selected_foundation_sha256")
        != selected["selected_foundation_sha256"]
        or checkpoint.get("exposure") != expected_exposure
        or checkpoint.get("diagnostic_gate") != summary_best.get("gate")
        or not isinstance(checkpoint.get("model_state_dict"), Mapping)
        or not isinstance(
            checkpoint.get("qwen_style_head_state_dict"), Mapping
        )
        or not isinstance(checkpoint.get("qwen_style_head_config"), Mapping)
        or not isinstance(checkpoint.get("action_stats"), Mapping)
        or any(checkpoint.get(key) != value for key, value in audit.items())
    ):
        message = (
            "best checkpoint is not the formal admissible artifact"
            if expected_role == "best_admissible"
            and expected_candidate_eligible
            and not expected_audit_replay
            else "checkpoint does not satisfy its evidence contract"
        )
        raise V81ComparisonError(f"{arm} {message}")
    input_contract = checkpoint.get("input_contract")
    unsigned_input = (
        {
            key: value
            for key, value in input_contract.items()
            if key != "sha256"
        }
        if isinstance(input_contract, Mapping)
        else {}
    )
    if (
        not isinstance(input_contract, Mapping)
        or not _is_sha256(input_contract.get("sha256"))
        or input_contract.get("sha256")
        != trainer.canonical_sha256(unsigned_input)
        or input_contract.get("arm") != arm
        or input_contract.get("target_steps") != EXPECTED_FORMAL_STEPS
        or input_contract.get("selected_qwen_variant")
        != selected["selected_qwen_variant"]
        or input_contract.get("selected_foundation_sha256")
        != selected["selected_foundation_sha256"]
        or input_contract.get("selected_condition_cache_sha256")
        != selected["selected_condition_cache_sha256"]
        or input_contract.get("hanyang_training_eligible_count")
        != trainer.HANYANG_TRAINING_ELIGIBLE_COUNT
        or input_contract.get("hanyang_boundary_admitted_count") != 0
        or input_contract.get("kimodo_admitted_count") != 0
    ):
        raise V81ComparisonError(f"{arm} input contract changed")
    for state_name in ("model_state_dict", "qwen_style_head_state_dict"):
        state = checkpoint[state_name]
        if (
            not state
            or any(
                not isinstance(tensor, torch.Tensor)
                or not torch.isfinite(tensor).all()
                for tensor in state.values()
            )
        ):
            raise V81ComparisonError(
                f"{arm} {state_name} contains non-finite or non-tensor values"
            )
    _reject_source_bearing_kimodo(
        {
            key: value
            for key, value in checkpoint.items()
            if key
            not in {"model_state_dict", "qwen_style_head_state_dict"}
        },
        field=f"{arm}.checkpoint",
    )


def _validate_formal_arm(
    validated: Mapping[str, Any],
    *,
    arm: str,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise V81ComparisonError("unknown V8.1 arm")
    try:
        record = supervisor._validate_arm_summary(
            validated["_supervisor"], arm=arm, smoke=False
        )
    except (supervisor.SupervisorError, ValueError) as error:
        raise V81ComparisonError(
            f"{arm} formal summary is not admissible: {error}"
        ) from error
    summary_path = Path(record["summary"]).resolve()
    summary = _load_json(summary_path, field=f"{arm} formal summary")
    best = summary.get("best_admissible")
    checkpoint_path = Path(
        str(record["best_admissible_checkpoint"])
    ).resolve()
    expected_root = Path(
        validated["_promoted"]["winner_overlay_arms"][arm]["output_dir"]
    ).resolve()
    if (
        summary_path != expected_root / "training_summary_v8_1.json"
        or checkpoint_path
        != expected_root / "best_admissible_generator_v8_1.pt"
        or not isinstance(best, Mapping)
        or best.get("checkpoint_sha256")
        != record["best_admissible_checkpoint_sha256"]
        or summary.get("completed_steps") != EXPECTED_FORMAL_STEPS
        or summary.get("smoke_test") is not False
        or summary.get("run_status") != "candidate_available"
        or summary.get("candidate_available") is not True
        or summary.get("prefix_schedule_assertion_passed") is not True
        or summary.get("exposure")
        != trainer.expected_exposure(EXPECTED_FORMAL_STEPS, arm=arm)
        or summary.get("exposure") != summary.get("expected_exposure")
        or summary.get("hanyang_training_eligible_count")
        != trainer.HANYANG_TRAINING_ELIGIBLE_COUNT
        or summary.get("hanyang_boundary_admitted_count") != 0
        or summary.get("kimodo_admitted_count") != 0
    ):
        raise V81ComparisonError(f"{arm} formal summary contract changed")
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise V81ComparisonError(
            f"cannot load {arm} formal best checkpoint: {error}"
        ) from error
    if not isinstance(checkpoint, Mapping):
        raise V81ComparisonError(f"{arm} checkpoint is not a mapping")
    _validate_checkpoint_contract(
        checkpoint,
        arm=arm,
        promoted=validated["_promoted"],
        summary_best=best,
    )
    _reject_source_bearing_kimodo(summary, field=f"{arm}.summary")
    return {
        "arm": arm,
        "summary": str(summary_path),
        "summary_sha256": abc_video.sha256_file(summary_path),
        "completed_steps": EXPECTED_FORMAL_STEPS,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": abc_video.sha256_file(checkpoint_path),
        "checkpoint_v8_1_step": int(checkpoint["v8_1_step"]),
        "checkpoint_global_step": int(checkpoint["global_step"]),
        "checkpoint_role": "best_admissible",
        "smoke_test": False,
        "candidate_eligible": True,
        "diagnostic_gate": deepcopy(dict(checkpoint["diagnostic_gate"])),
        "input_contract": deepcopy(dict(checkpoint["input_contract"])),
        "exposure_at_best": deepcopy(dict(checkpoint["exposure"])),
        "final_exposure": deepcopy(dict(summary["exposure"])),
        "no_kimodo": True,
        "kimodo_admitted_count": 0,
        "hanyang_source_count": trainer.HANYANG_SOURCE_POOL_COUNT,
        "hanyang_boundary_excluded_count": trainer.HANYANG_BOUNDARY_COUNT,
        "hanyang_training_eligible_count": (
            trainer.HANYANG_TRAINING_ELIGIBLE_COUNT
        ),
        "hanyang_boundary_admitted_count": 0,
    }


def formal_completion_state(
    validated: Mapping[str, Any],
    *,
    service_probe: Callable[[str], Mapping[str, Any]] = _service_snapshot,
) -> dict[str, Any]:
    service = dict(
        service_probe(validated["retry4_supervisor_service_unit"])
    )
    receipt_path = Path(validated["_supervisor_receipt_path"])
    if service.get("failed") is True:
        raise V81ComparisonError("retry4 supervisor failed")
    if not receipt_path.is_file():
        state_path = Path(validated["_supervisor"]["state"])
        current_stage = None
        if state_path.is_file():
            current_stage = _load_json(
                state_path, field="retry4 supervisor state"
            ).get("current_stage")
        return {
            "ready": False,
            "status": "waiting_for_retry4_formal_completion",
            "current_stage": current_stage,
            "service": service,
            "supervisor_receipt": str(receipt_path),
        }
    receipt = _validate_supervisor_receipt(validated)
    if (
        service.get("known") is not True
        and service.get("transient_reclaimed") is not True
    ):
        raise V81ComparisonError(
            "cannot prove retry4 supervisor has released the GPU"
        )
    if service.get("active") is True:
        return {
            "ready": False,
            "status": "waiting_for_retry4_supervisor_exit",
            "current_stage": "complete_receipt_written",
            "service": service,
            "supervisor_receipt": receipt["path"],
            "supervisor_receipt_file_sha256": receipt["file_sha256"],
        }
    arms = {
        arm: _validate_formal_arm(validated, arm=arm)
        for arm in ARMS
    }
    stages = receipt["value"]["stages"]
    for arm, stage in (
        (CONTROL_ARM, "control_formal"),
        (TREATMENT_ARM, "treatment_formal"),
    ):
        sealed = stages[stage]["record"]
        current = arms[arm]
        if (
            sealed.get("summary_sha256") != current["summary_sha256"]
            or sealed.get("best_admissible_checkpoint_sha256")
            != current["checkpoint_sha256"]
            or sealed.get("completed_steps") != EXPECTED_FORMAL_STEPS
            or sealed.get("smoke_test") is not False
            or (sealed.get("emotion_candidate_gate") or {}).get(
                "absolute_v7_gate_passed"
            )
            is not True
        ):
            raise V81ComparisonError(
                f"retry4 receipt does not bind {arm} formal artifacts"
            )
    control_input = arms[CONTROL_ARM]["input_contract"]
    treatment_input = arms[TREATMENT_ARM]["input_contract"]
    shared_fields = (
        "target_steps",
        "selected_qwen_variant",
        "selected_foundation_sha256",
        "selected_condition_cache_sha256",
        "v7_reference_config_sha256",
        "hanyang_strict_manifest_sha256",
        "hanyang_pool_receipt_sha256",
        "hanyang_rejected_manifest_sha256",
        "seed",
        "noise_seed",
        "diagnostic_seed_policy",
        "diagnostic_seed",
        "paired_beat2_slot_policy",
        "hanyang_source_pool_count",
        "hanyang_boundary_excluded_count",
        "hanyang_training_eligible_count",
        "hanyang_safe_clip_ids_sha256",
        "hanyang_excluded_clip_ids_sha256",
        "kimodo_admitted_count",
    )
    if any(
        control_input.get(field) != treatment_input.get(field)
        for field in shared_fields
    ):
        raise V81ComparisonError(
            "formal control/treatment input contracts are not paired"
        )
    return {
        "ready": True,
        "status": "formal_control_and_treatment_admissible",
        "service": service,
        "supervisor_receipt": {
            "path": receipt["path"],
            "file_sha256": receipt["file_sha256"],
            "canonical_sha256": receipt["canonical_sha256"],
        },
        "arms": arms,
        "same_selected_foundation": True,
        "same_selected_qwen_condition_cache": True,
        "same_seed_noise_split_diagnostic": True,
        "formal_arms_sequential": True,
        "independent_initialization_no_cross_arm_warm_start": True,
        "no_kimodo": True,
        "kimodo_admitted_count": 0,
    }


def _jsonl_step_record(path: Path, *, step: int) -> dict[str, Any]:
    if not path.is_file():
        raise V81ComparisonError(f"original progress log is missing: {path}")
    matches: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise V81ComparisonError(
                f"invalid original progress JSON at line {line_number}"
            ) from error
        if isinstance(value, dict) and value.get("step") == step:
            matches.append(value)
    if len(matches) != 1:
        raise V81ComparisonError(
            f"original progress must contain exactly one step {step} record"
        )
    return matches[0]


def _validate_matched_audit_arm(
    validated: Mapping[str, Any],
    *,
    audit_root: Path,
    arm: str,
    step: int,
) -> dict[str, Any]:
    root = (audit_root / arm).resolve()
    summary_path = root / "training_summary_v8_1.json"
    checkpoint_path = root / "last_generator_v8_1.pt"
    summary = _load_json(summary_path, field=f"{arm} audit summary")
    expected_exposure = trainer.expected_exposure(step, arm=arm)
    last = summary.get("last_diagnostics")
    if (
        summary.get("arm") != arm
        or summary.get("completed_steps") != step
        or summary.get("target_steps") != EXPECTED_FORMAL_STEPS
        or summary.get("audit_replay") is not True
        or summary.get("audit_stop_step") != step
        or summary.get("audit_purpose") != MATCHED_AUDIT_PURPOSE
        or summary.get("run_status")
        != "matched_step_audit_replay_completed_not_candidate"
        or summary.get("candidate_available") is not False
        or summary.get("best_admissible") is not None
        or summary.get("smoke_test") is not False
        or summary.get("formal_release_eligible") is not False
        or summary.get("last_checkpoint_status") != "admissible"
        or summary.get("last_checkpoint") != str(checkpoint_path)
        or summary.get("exposure") != expected_exposure
        or summary.get("expected_exposure") != expected_exposure
        or summary.get("prefix_schedule_assertion_passed") is not True
        or not isinstance(last, Mapping)
        or last.get("step") != step
        or (last.get("gate") or {}).get("admissible") is not True
        or summary.get("hanyang_training_eligible_count")
        != trainer.HANYANG_TRAINING_ELIGIBLE_COUNT
        or summary.get("hanyang_boundary_admitted_count") != 0
        or summary.get("kimodo_admitted_count") != 0
    ):
        raise V81ComparisonError(f"{arm} audit summary contract changed")
    checkpoint_sha = abc_video.sha256_file(checkpoint_path)
    if checkpoint_sha != summary.get("last_checkpoint_sha256"):
        raise V81ComparisonError(f"{arm} audit checkpoint SHA256 changed")
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise V81ComparisonError(
            f"cannot load {arm} matched-step audit checkpoint: {error}"
        ) from error
    if not isinstance(checkpoint, Mapping):
        raise V81ComparisonError(f"{arm} audit checkpoint is not a mapping")
    _validate_checkpoint_contract(
        checkpoint,
        arm=arm,
        promoted=validated["_promoted"],
        summary_best=last,
        expected_role="last_audit_only",
        expected_candidate_eligible=False,
        expected_step=step,
        expected_audit_replay=True,
    )
    original_root = Path(
        validated["_promoted"]["winner_overlay_arms"][arm]["output_dir"]
    ).resolve()
    original = _jsonl_step_record(
        original_root / "progress_v8_1.jsonl", step=step
    )
    deterministic_checks = {
        "diagnostics": original.get("held_out_diagnostics") == last,
        "exposure": original.get("exposure") == summary.get("exposure"),
        "expected_exposure": (
            original.get("expected_exposure")
            == summary.get("expected_exposure")
        ),
        "beat2_metrics": (
            original.get("beat2") == (summary.get("last_event") or {}).get("beat2")
        ),
        "hanyang_metrics": (
            original.get("hanyang")
            == (summary.get("last_event") or {}).get("hanyang")
        ),
        "beat2_batch_receipts": (
            original.get("beat2_batch_receipts")
            == (summary.get("last_event") or {}).get("beat2_batch_receipts")
        ),
        "hanyang_clip_ids": (
            original.get("hanyang_clip_ids")
            == (summary.get("last_event") or {}).get("hanyang_clip_ids")
        ),
        "learning_rates": (
            original.get("learning_rates")
            == (summary.get("last_event") or {}).get("learning_rates")
        ),
        "gradient_norms": (
            original.get("gradient_norms_before_clip")
            == (summary.get("last_event") or {}).get(
                "gradient_norms_before_clip"
            )
        ),
    }
    if not all(deterministic_checks.values()):
        raise V81ComparisonError(
            f"{arm} audit replay differs from original training log"
        )
    _reject_source_bearing_kimodo(summary, field=f"{arm}.audit_summary")
    return {
        "arm": arm,
        "checkpoint_mode": "matched_step_audit",
        "summary": str(summary_path),
        "summary_sha256": abc_video.sha256_file(summary_path),
        "completed_steps": step,
        "formal_schedule_target_steps": EXPECTED_FORMAL_STEPS,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_v8_1_step": step,
        "checkpoint_global_step": int(checkpoint["global_step"]),
        "checkpoint_role": "last_audit_only",
        "smoke_test": False,
        "candidate_eligible": False,
        "diagnostic_gate": deepcopy(dict(last["gate"])),
        "diagnostics": deepcopy(dict(last["diagnostics"])),
        "input_contract": deepcopy(dict(checkpoint["input_contract"])),
        "exposure_at_best": deepcopy(dict(checkpoint["exposure"])),
        "final_exposure": deepcopy(dict(summary["exposure"])),
        "deterministic_replay_checks": deterministic_checks,
        "original_progress": str(original_root / "progress_v8_1.jsonl"),
        "original_progress_sha256": abc_video.sha256_file(
            original_root / "progress_v8_1.jsonl"
        ),
        "no_kimodo": True,
        "kimodo_admitted_count": 0,
    }


def matched_step_audit_completion(
    validated: Mapping[str, Any],
    *,
    audit_root: str | Path,
    step: int,
) -> dict[str, Any]:
    if step <= 1 or step > EXPECTED_FORMAL_STEPS:
        raise V81ComparisonError("matched audit step is outside the useful range")
    root = Path(audit_root).expanduser().resolve()
    receipt = _validate_supervisor_receipt(validated)
    arms = {
        arm: _validate_matched_audit_arm(
            validated, audit_root=root, arm=arm, step=step
        )
        for arm in ARMS
    }
    control = arms[CONTROL_ARM]["diagnostics"]
    treatment = arms[TREATMENT_ARM]["diagnostics"]
    control_hanyang = float(control["hanyang"]["total"])
    treatment_hanyang = float(treatment["hanyang"]["total"])
    hanyang_improvement = (
        control_hanyang - treatment_hanyang
    ) / control_hanyang
    control_beat2 = float(control["beat2"]["aligned_flow_loss"])
    treatment_beat2 = float(treatment["beat2"]["aligned_flow_loss"])
    beat2_relative_change = (treatment_beat2 - control_beat2) / control_beat2
    if (
        not math.isfinite(hanyang_improvement)
        or hanyang_improvement <= 0
        or not math.isfinite(beat2_relative_change)
        or beat2_relative_change > 0.01
    ):
        raise V81ComparisonError(
            "matched-step audit does not show bounded held-out improvement"
        )
    completion = {
        "ready": True,
        "status": "matched_step_audit_replay_admissible",
        "evidence_mode": "matched_step_audit",
        "audit_root": str(root),
        "audit_step": step,
        "audit_purpose": MATCHED_AUDIT_PURPOSE,
        "formal_candidate": False,
        "supervisor_receipt": {
            "path": receipt["path"],
            "file_sha256": receipt["file_sha256"],
            "canonical_sha256": receipt["canonical_sha256"],
        },
        "arms": arms,
        "paired_metrics": {
            "control_hanyang_total": control_hanyang,
            "treatment_hanyang_total": treatment_hanyang,
            "hanyang_relative_improvement": hanyang_improvement,
            "control_beat2_aligned_flow": control_beat2,
            "treatment_beat2_aligned_flow": treatment_beat2,
            "beat2_aligned_flow_relative_change": beat2_relative_change,
            "control_correct_flow_win_rate": float(
                control["beat2"]["correct_flow_win_rate"]
            ),
            "treatment_correct_flow_win_rate": float(
                treatment["beat2"]["correct_flow_win_rate"]
            ),
            "control_minimum_emotion_retention": float(
                arms[CONTROL_ARM]["diagnostic_gate"][
                    "minimum_emotion_retention"
                ]
            ),
            "treatment_minimum_emotion_retention": float(
                arms[TREATMENT_ARM]["diagnostic_gate"][
                    "minimum_emotion_retention"
                ]
            ),
        },
        "same_selected_foundation": True,
        "same_selected_qwen_condition_cache": True,
        "same_seed_noise_split_diagnostic": True,
        "same_training_step": True,
        "independent_initialization_no_cross_arm_warm_start": True,
        "strict_single_variable_causal_claim": True,
        "no_kimodo": True,
        "kimodo_admitted_count": 0,
    }
    completion["sha256"] = abc_video.value_sha256(completion)
    return completion


def prepare_plan(
    validated: Mapping[str, Any],
    *,
    completion: Mapping[str, Any] | None = None,
    evidence_mode: str = "formal_best",
) -> dict[str, Any]:
    completion = (
        dict(completion)
        if completion is not None
        else formal_completion_state(validated)
    )
    gt_plan = validated["_gt_plan"]
    if evidence_mode not in {"formal_best", "matched_step_audit"}:
        raise V81ComparisonError("unknown comparison evidence mode")
    audit_mode = evidence_mode == "matched_step_audit"
    comparison_contract = (
        MATCHED_AUDIT_COMPARISON_CONTRACT
        if audit_mode
        else COMPARISON_CONTRACT
    )
    plan = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": PLAN_ARTIFACT_KIND,
        "status": (
            (
                "ready_for_matched_step_audit_generation"
                if audit_mode
                else "ready_for_formal_generation"
            )
            if completion.get("ready") is True
            else "waiting_for_retry4_formal_completion"
        ),
        "created_utc": utc_now(),
        "config": validated["_config_path"],
        "config_sha256": validated["_config_sha256"],
        "data_policy": (
            MATCHED_AUDIT_DATA_POLICY if audit_mode else DATA_POLICY
        ),
        "evidence_mode": evidence_mode,
        "no_kimodo": True,
        "kimodo_admitted_count": 0,
        "hanyang_source_count": trainer.HANYANG_SOURCE_POOL_COUNT,
        "hanyang_boundary_excluded_count": trainer.HANYANG_BOUNDARY_COUNT,
        "hanyang_training_eligible_count": (
            trainer.HANYANG_TRAINING_ELIGIBLE_COUNT
        ),
        "hanyang_boundary_admitted_count": 0,
        "held_out_gt_plan": {
            "path": validated["_gt_plan_path"],
            "file_sha256": validated[
                "expected_held_out_gt_plan_file_sha256"
            ],
            "canonical_sha256": validated[
                "expected_held_out_gt_plan_canonical_sha256"
            ],
        },
        "source_manifest": gt_plan["source_manifest"],
        "source_manifest_sha256": gt_plan["source_manifest_sha256"],
        "target_duration_sec": 60.0,
        "expected_frames": EXPECTED_FRAMES,
        "event_count": EXPECTED_EVENTS,
        "selections": deepcopy(gt_plan["selections"]),
        "comparison_contract": deepcopy(comparison_contract),
        "metric_contract": deepcopy(METRIC_CONTRACT),
        "formal_completion": deepcopy(dict(completion)),
        "gt_identity_selected_before_generation": True,
        "gt_action_values_loaded_before_generation": False,
        "generation_or_render_executed": False,
        "gpu_accessed": False,
        "smoke_checkpoint_used": False,
        "static_padding_frames": 0,
        "temporal_padding_frames": 0,
        "endpoint_hold_frames": 0,
    }
    plan["sha256"] = abc_video.value_sha256(plan)
    _atomic_json(Path(validated["_output_dir"]) / PLAN_FILENAME, plan)
    return plan


def _load_formal_model(
    validated: Mapping[str, Any],
    *,
    arm: str,
    formal: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    checkpoint_path = Path(formal["checkpoint"])
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    summary = _load_json(Path(formal["summary"]), field=f"{arm} summary")
    _validate_checkpoint_contract(
        checkpoint,
        arm=arm,
        promoted=validated["_promoted"],
        summary_best=summary["best_admissible"],
    )
    shape = trainer.V7_RUNTIME_MODEL_CONFIG
    model = style_video.create_ula_model(
        checkpoint["architecture"],
        action_dim=ACTION_DIM,
        condition_dim=CONDITION_DIM,
        hidden_dim=int(shape["hidden_dim"]),
        layers=int(shape["layers"]),
        semantic_tokens=int(shape["semantic_tokens"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    action_stats = checkpoint["action_stats"]
    model.action_stats = {
        name: torch.as_tensor(
            action_stats[name], dtype=torch.float32
        ).clone()
        for name in ("mean", "std")
    }
    if (
        any(value.shape != (ACTION_DIM,) for value in model.action_stats.values())
        or any(
            not torch.isfinite(value).all()
            for value in model.action_stats.values()
        )
        or torch.any(model.action_stats["std"] <= 0)
    ):
        raise V81ComparisonError(f"{arm} action stats are invalid")
    try:
        style_head = style_video.QwenStyleHead.from_config(
            checkpoint["qwen_style_head_config"]
        )
        style_head.load_state_dict(
            checkpoint["qwen_style_head_state_dict"], strict=True
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise V81ComparisonError(
            f"{arm} Qwen style head is invalid: {error}"
        ) from error
    model = model.to(device).eval().requires_grad_(False)
    style_head = style_head.to(device).eval().requires_grad_(False)
    return model, style_head, {
        "path": str(checkpoint_path.resolve()),
        "sha256": formal["checkpoint_sha256"],
        "summary": formal["summary"],
        "summary_sha256": formal["summary_sha256"],
        "arm": arm,
        "checkpoint_role": "best_admissible",
        "v8_1_step": int(checkpoint["v8_1_step"]),
        "global_step": int(checkpoint["global_step"]),
        "smoke_test": False,
        "candidate_eligible": True,
        "diagnostic_gate": deepcopy(dict(checkpoint["diagnostic_gate"])),
        "no_kimodo": True,
        "kimodo_admitted_count": 0,
    }


def _load_matched_audit_model(
    validated: Mapping[str, Any],
    *,
    arm: str,
    formal: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    checkpoint_path = Path(formal["checkpoint"])
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    summary = _load_json(Path(formal["summary"]), field=f"{arm} audit summary")
    _validate_checkpoint_contract(
        checkpoint,
        arm=arm,
        promoted=validated["_promoted"],
        summary_best=summary["last_diagnostics"],
        expected_role="last_audit_only",
        expected_candidate_eligible=False,
        expected_step=int(formal["checkpoint_v8_1_step"]),
        expected_audit_replay=True,
    )
    shape = trainer.V7_RUNTIME_MODEL_CONFIG
    model = style_video.create_ula_model(
        checkpoint["architecture"],
        action_dim=ACTION_DIM,
        condition_dim=CONDITION_DIM,
        hidden_dim=int(shape["hidden_dim"]),
        layers=int(shape["layers"]),
        semantic_tokens=int(shape["semantic_tokens"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    action_stats = checkpoint["action_stats"]
    model.action_stats = {
        name: torch.as_tensor(
            action_stats[name], dtype=torch.float32
        ).clone()
        for name in ("mean", "std")
    }
    if (
        any(value.shape != (ACTION_DIM,) for value in model.action_stats.values())
        or any(
            not torch.isfinite(value).all()
            for value in model.action_stats.values()
        )
        or torch.any(model.action_stats["std"] <= 0)
    ):
        raise V81ComparisonError(f"{arm} audit action stats are invalid")
    try:
        style_head = style_video.QwenStyleHead.from_config(
            checkpoint["qwen_style_head_config"]
        )
        style_head.load_state_dict(
            checkpoint["qwen_style_head_state_dict"], strict=True
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise V81ComparisonError(
            f"{arm} audit Qwen style head is invalid: {error}"
        ) from error
    model = model.to(device).eval().requires_grad_(False)
    style_head = style_head.to(device).eval().requires_grad_(False)
    return model, style_head, {
        "path": str(checkpoint_path.resolve()),
        "sha256": formal["checkpoint_sha256"],
        "summary": formal["summary"],
        "summary_sha256": formal["summary_sha256"],
        "arm": arm,
        "checkpoint_role": "last_audit_only",
        "evidence_mode": "matched_step_audit",
        "audit_replay": True,
        "audit_purpose": MATCHED_AUDIT_PURPOSE,
        "v8_1_step": int(checkpoint["v8_1_step"]),
        "global_step": int(checkpoint["global_step"]),
        "smoke_test": False,
        "candidate_eligible": False,
        "diagnostic_gate": deepcopy(dict(checkpoint["diagnostic_gate"])),
        "deterministic_replay_checks": deepcopy(
            dict(formal["deterministic_replay_checks"])
        ),
        "no_kimodo": True,
        "kimodo_admitted_count": 0,
    }


def _generate_arm(
    validated: Mapping[str, Any],
    *,
    arm: str,
    formal: Mapping[str, Any],
    plan: Mapping[str, Any],
    prompt_latents: Mapping[str, np.ndarray],
    shared_noises: Sequence[np.ndarray],
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_mode = formal.get("checkpoint_mode", "formal_best")
    if checkpoint_mode == "matched_step_audit":
        model, style_head, checkpoint_receipt = _load_matched_audit_model(
            validated, arm=arm, formal=formal, device=device
        )
    elif checkpoint_mode == "formal_best":
        model, style_head, checkpoint_receipt = _load_formal_model(
            validated, arm=arm, formal=formal, device=device
        )
    else:
        raise V81ComparisonError(f"unknown checkpoint mode: {checkpoint_mode}")
    sampling = validated["_base"]["sampling"]
    steps = int(sampling["steps"])
    guidance_scale = style_video.validate_guidance_scale(
        sampling["guidance_scale"]
    )
    raw_segments: list[np.ndarray] = []
    events: list[dict[str, Any]] = []
    for index, (selection, noise) in enumerate(
        zip(plan["selections"], shared_noises, strict=True),
        start=1,
    ):
        frames = int(selection["frames"])
        prompt = str(selection["prompt"])
        latent = np.asarray(prompt_latents[prompt], dtype=np.float32)
        condition, predicted_style = (
            style_video.compose_text_style_condition(
                style_head, latent, device=device
            )
        )
        planner_duration = style_video.predict_duration_sec(
            model, condition, device=device
        )
        noise = np.asarray(noise, dtype=np.float32)
        trajectory = np.asarray(
            style_video.sample_trajectory_cfg(
                model,
                condition,
                initial_noise=noise,
                steps=steps,
                guidance_scale=guidance_scale,
                device=device,
            ),
            dtype=np.float32,
        )
        if (
            noise.shape != (frames, ACTION_DIM)
            or trajectory.shape != (frames, ACTION_DIM)
            or not np.isfinite(trajectory).all()
        ):
            raise V81ComparisonError(
                f"{arm} event {index} output shape changed"
            )
        raw_segments.append(trajectory)
        events.append(
            {
                "index": index,
                "frames": frames,
                "native_duration_sec": frames / FPS,
                "prompt": prompt,
                "target_emotion": selection["target_emotion"],
                "seed": int(selection["seed"]),
                "initial_noise_sha256": hashlib.sha256(
                    noise.tobytes()
                ).hexdigest(),
                "condition_sha256": hashlib.sha256(
                    condition.tobytes()
                ).hexdigest(),
                "predicted_style": predicted_style.tolist(),
                "planner_predicted_duration_sec": planner_duration,
                "planner_duration_used_for_frames": False,
                "trajectory_sha256": hashlib.sha256(
                    trajectory.tobytes()
                ).hexdigest(),
            }
        )
    display = np.concatenate(raw_segments, axis=0)
    del model, style_head
    gc.collect()
    torch.cuda.empty_cache()
    if display.shape != (EXPECTED_FRAMES, ACTION_DIM):
        raise V81ComparisonError(f"{arm} montage is not exactly 60 seconds")
    return {
        "arm": arm,
        "display": display,
        "raw_segments": raw_segments,
        "events": events,
        "checkpoint_admission": checkpoint_receipt,
        "diagnostic_response": deepcopy(
            dict(checkpoint_receipt["diagnostic_gate"])
        ),
        "fixed_frames_equal_gt": True,
        "static_padding_frames": 0,
        "endpoint_hold_frames": 0,
        "additional_smoothing": False,
    }


def _metric_receipt(values: np.ndarray) -> dict[str, Any]:
    return prior_gt._metric_receipt(values)


def _shared_union_camera(
    trajectories: Sequence[np.ndarray],
    *,
    width: int,
    height: int,
    simplified: bool,
) -> dict[str, Any]:
    """Fit one camera to the union of all three complete trajectories."""

    from upper_body_skeleton import mujoco_playback

    union = np.concatenate(
        [np.asarray(value, dtype=np.float32) for value in trajectories],
        axis=0,
    )
    model, joint_to_qpos, _ = mujoco_playback.load_preview_model(
        simplified=simplified,
        joint_order=JOINT_ORDER_18D,
    )
    data = mujoco_playback.mujoco.MjData(model)
    camera, framing = mujoco_playback.fit_full_body_camera(
        model,
        data,
        union,
        joint_to_qpos,
        width=width,
        height=height,
    )
    return {
        "distance": float(camera.distance),
        "lookat": [float(value) for value in camera.lookat],
        "azimuth_deg": float(camera.azimuth),
        "elevation_deg": float(camera.elevation),
        "framing": deepcopy(dict(framing)),
    }


def _render_with_shared_camera(
    trajectory: np.ndarray,
    *,
    csv_path: Path,
    mp4_path: Path,
    fps: float,
    width: int,
    height: int,
    simplified: bool,
    camera: Mapping[str, Any],
) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    from upper_body_skeleton.mujoco_playback import render_motion

    abc_video.write_generated_csv(csv_path, trajectory, fps=fps)
    receipt = render_motion(
        csv_path,
        mp4_path,
        fps=fps,
        width=width,
        height=height,
        simplified=simplified,
        camera_override=dict(camera),
    )
    observed = {
        "distance": receipt["camera_distance"],
        "lookat": receipt["camera_lookat"],
        "azimuth_deg": receipt["camera_azimuth_deg"],
        "elevation_deg": receipt["camera_elevation_deg"],
    }
    expected = {
        key: camera[key]
        for key in ("distance", "lookat", "azimuth_deg", "elevation_deg")
    }
    if observed != expected:
        raise V81ComparisonError("render lane did not use shared union camera")
    receipt["shared_union_camera"] = deepcopy(dict(camera))
    return receipt


def _response_line(gate: Mapping[str, Any]) -> str:
    values = gate["retentions"]
    return (
        f"min {float(gate['minimum_emotion_retention']):.3f} · "
        f"align {float(values['aligned_vs_zero']):.3f} · "
        f"cross {float(values['aligned_vs_cross_group']):.3f} · "
        f"flow {float(values['flow_gap']):.3f} · "
        f"Q2/Q6/G54 {float(values['q2_recall']):.3f}/"
        f"{float(values['q6_recall']):.3f}/"
        f"{float(values['global54_recall']):.3f}"
    )


def build_ass_document(
    timeline: Sequence[Mapping[str, Any]],
    *,
    responses: Mapping[str, Mapping[str, Any]],
    best_steps: Mapping[str, int],
    robot_width: int,
    panel_width: int,
    height: int,
    evidence_mode: str = "formal_best",
) -> str:
    if len(timeline) != EXPECTED_EVENTS:
        raise V81ComparisonError("ASS timeline must contain 24 events")
    width = robot_width + panel_width
    panel_left = robot_width + 22
    if evidence_mode == "matched_step_audit":
        header_text = "MATCHED STEP AUDIT · HELD-OUT BEAT2 · SAFE323"
        receipt_text = (
            "MATCHED STEP 1500 DETERMINISTIC AUDIT REPLAY · NOT A FORMAL CANDIDATE\\N"
            "BOTH HELD-OUT GATES PASS · ORIGINAL LOG REPLAY CHECKS PASS\\N"
            "NATIVE EVENT SPEED · STATIC PADDING = 0 · ENDPOINT HOLD = 0\\N"
            "NO SMOOTHING / TIME-WARP / CROP / BLEND / RETURN-TO-ZERO\\N"
            "JERK/EXPRESSION ARE PER NATIVE EVENT; BOUNDARIES EXCLUDED\\N"
            "EXPRESSION = JOINT-RANGE RMS PROXY, NOT EMOTION ACCURACY\\N"
            "HANYANG SAFE323 MOTION-ONLY · BOUNDARY21 EXCLUDED · KIMODO = 0"
        )
        step_prefix = "MATCHED AUDIT STEP"
    elif evidence_mode == "formal_best":
        header_text = (
            "HELD-OUT BEAT2 GT · V8.1 CONTROL · SAFE323 HANYANG TREATMENT"
        )
        receipt_text = (
            "FORMAL 60K BEST_ADMISSIBLE CHECKPOINTS ONLY · SMOKE = FORBIDDEN\\N"
            "NATIVE EVENT SPEED · STATIC PADDING = 0 · ENDPOINT HOLD = 0\\N"
            "NO SMOOTHING / TIME-WARP / CROP / BLEND / RETURN-TO-ZERO\\N"
            "JERK/EXPRESSION ARE PER NATIVE EVENT; BOUNDARIES EXCLUDED\\N"
            "EXPRESSION = JOINT-RANGE RMS PROXY, NOT EMOTION ACCURACY\\N"
            "RESPONSE = FIXED HELD-OUT BEAT2 DIAGNOSTIC RETENTION\\N"
            "HANYANG SAFE323 MOTION-ONLY · BOUNDARY21 EXCLUDED · KIMODO = 0"
        )
        step_prefix = "FORMAL BEST STEP"
    else:
        raise V81ComparisonError("unknown ASS evidence mode")
    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        (
            "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,"
            "OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,"
            "ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,"
            "MarginR,MarginV,Encoding"
        ),
        (
            f"Style:Header,DejaVu Sans,20,&H0050CD89,&H0050CD89,"
            f"&H000F172A,&H000F172A,-1,0,0,0,100,100,0,0,1,0,0,7,"
            f"{panel_left},18,20,1"
        ),
        (
            f"Style:Body,DejaVu Sans,17,&H00FFFFFF,&H00FFFFFF,&H000F172A,"
            f"&H000F172A,0,0,0,0,100,100,0,0,1,0,0,7,"
            f"{panel_left},20,66,1"
        ),
        (
            f"Style:Receipt,DejaVu Sans,13,&H00C8D2DC,&H00C8D2DC,"
            f"&H000F172A,&H000F172A,0,0,0,0,100,100,0,0,1,0,0,1,"
            f"{panel_left},20,18,1"
        ),
        (
            "Style:Clock,DejaVu Sans,18,&H00FFFFFF,&H00FFFFFF,&H000F172A,"
            "&H000F172A,-1,0,0,0,100,100,0,0,1,0,0,9,20,20,20,1"
        ),
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]
    events = [
        (
            "Dialogue: 0,0:00:00.00,0:01:00.00,Header,,0,0,0,,"
            + header_text
        ),
        (
            "Dialogue: 0,0:00:00.00,0:01:00.00,Receipt,,0,0,0,,"
            + receipt_text
        ),
    ]
    wrap_width = max(34, panel_width // 13)
    response_control = prior_gt._ass_escape(
        _response_line(responses[CONTROL_ARM])
    )
    response_treatment = prior_gt._ass_escape(
        _response_line(responses[TREATMENT_ARM])
    )
    for item in timeline:
        start = prior_gt._ass_time(float(item["start_sec"]))
        end = prior_gt._ass_time(float(item["end_sec"]))
        prompt = "\\N".join(
            textwrap.wrap(
                prior_gt._ass_escape(item["prompt"]),
                width=wrap_width,
            )
        )
        metrics = item["metrics"]
        planners = item["planner_duration_diagnostics_sec"]
        body = (
            f"EVENT {int(item['index']):02d}/24 · "
            f"EMOTION: {prior_gt._ass_escape(item['target_emotion']).upper()}"
            f" · NATIVE {float(item['native_duration_sec']):.2f}s\\N\\N"
            f"TEXT:\\N{prompt}\\N\\N"
            "RAW JERK RMS rad/s³\\N"
            f"GT {metrics['gt']['jerk_rms_rad_s3']:.1f} · "
            f"C {metrics[CONTROL_ARM]['jerk_rms_rad_s3']:.1f} · "
            f"T {metrics[TREATMENT_ARM]['jerk_rms_rad_s3']:.1f}\\N"
            "EXPRESSION amplitude joint-range RMS rad\\N"
            f"GT "
            f"{metrics['gt']['expression_amplitude_joint_range_rms_rad']:.3f}"
            f" · C "
            f"{metrics[CONTROL_ARM]['expression_amplitude_joint_range_rms_rad']:.3f}"
            f" · T "
            f"{metrics[TREATMENT_ARM]['expression_amplitude_joint_range_rms_rad']:.3f}"
            "\\NHEAD activity velocity RMS rad/s\\N"
            f"GT {metrics['gt']['head_velocity_rms_rad_s']:.3f} · "
            f"C {metrics[CONTROL_ARM]['head_velocity_rms_rad_s']:.3f} · "
            f"T {metrics[TREATMENT_ARM]['head_velocity_rms_rad_s']:.3f}\\N\\N"
            f"C RESPONSE: {response_control}\\N"
            f"T RESPONSE: {response_treatment}\\N\\N"
            f"{step_prefix} C {int(best_steps[CONTROL_ARM])} · "
            f"T {int(best_steps[TREATMENT_ARM])}\\N"
            f"PLANNER diagnostic C {planners[CONTROL_ARM]:.2f}s · "
            f"T {planners[TREATMENT_ARM]:.2f}s\\N"
            f"GT clip: {prior_gt._ass_escape(item['gt']['clip_id'])} · "
            f"seed {int(item['seed'])}"
        )
        events.append(
            f"Dialogue: 0,{start},{end},Body,,0,0,0,,{body}"
        )
    for second in range(60):
        events.append(
            "Dialogue: 1,"
            f"{prior_gt._ass_time(second)},"
            f"{prior_gt._ass_time(second + 1)},Clock,,0,0,0,,"
            f"TIME {second:02d}.0s / 60.0s"
        )
    return "\n".join(header + events) + "\n"


def _build_comparison(
    validated: Mapping[str, Any],
    *,
    overwrite: bool,
    completion_override: Mapping[str, Any] | None = None,
    evidence_mode: str = "formal_best",
    summary_filename: str = SUMMARY_FILENAME,
    final_video_filename: str = FINAL_VIDEO_FILENAME,
) -> dict[str, Any]:
    if evidence_mode not in {"formal_best", "matched_step_audit"}:
        raise V81ComparisonError("unknown comparison evidence mode")
    audit_mode = evidence_mode == "matched_step_audit"
    completion = (
        deepcopy(dict(completion_override))
        if completion_override is not None
        else formal_completion_state(validated)
    )
    if completion.get("ready") is not True:
        raise V81ComparisonError(
            "retry4 formal control/treatment are incomplete; no GPU work ran"
        )
    output_dir = Path(validated["_output_dir"])
    summary_path = output_dir / summary_filename
    final_video = output_dir / final_video_filename
    if (summary_path.exists() or final_video.exists()) and not overwrite:
        raise V81ComparisonError(
            "V8.1 comparison exists; pass --overwrite to rebuild"
        )
    plan = prepare_plan(
        validated, completion=completion, evidence_mode=evidence_mode
    )
    manifest_records, manifest_sha = abc_video.load_manifest(
        plan["source_manifest"],
        expected_sha256=plan["source_manifest_sha256"],
    )
    if manifest_sha != plan["source_manifest_sha256"]:
        raise V81ComparisonError("held-out manifest SHA changed")
    prompts = [str(item["prompt"]) for item in plan["selections"]]
    selected = validated["_promoted"]["qwen_ab_selection_gate"]
    condition_cache = Path(selected["selected_condition_cache"])
    prompt_latents, cache_receipt = style_video._load_prompt_latents(
        condition_cache,
        prompts=prompts,
        manifest_records=manifest_records,
        manifest_sha256=manifest_sha,
    )
    if cache_receipt["sha256"] != selected[
        "selected_condition_cache_sha256"
    ]:
        raise V81ComparisonError("selected Qwen condition cache changed")
    shared_noises = [
        style_video.shared_initial_noise(
            seed=int(selection["seed"]),
            frames=int(selection["frames"]),
        )
        for selection in plan["selections"]
    ]
    device = torch.device(str(validated["_base"]["sampling"]["device"]))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise V81ComparisonError(
            "formal post-training sampling requires available CUDA"
        )
    branches = {}
    for arm in ARMS:
        branches[arm] = _generate_arm(
            validated,
            arm=arm,
            formal=completion["arms"][arm],
            plan=plan,
            prompt_latents=prompt_latents,
            shared_noises=shared_noises,
            device=device,
        )
    for index in range(EXPECTED_EVENTS):
        left = branches[CONTROL_ARM]["events"][index][
            "initial_noise_sha256"
        ]
        right = branches[TREATMENT_ARM]["events"][index][
            "initial_noise_sha256"
        ]
        if left != right:
            raise V81ComparisonError(
                f"control/treatment noise differs at event {index + 1}"
            )

    # GT action arrays are intentionally opened only after both branches have
    # completed generation.
    gt_segments: list[np.ndarray] = []
    timeline: list[dict[str, Any]] = []
    gt_csv_receipts: list[dict[str, Any]] = []
    for index, selection in enumerate(plan["selections"]):
        clip_id = selection["reference"]["clip_id"]
        record = manifest_records[clip_id]
        gt_values, gt_receipt = prior_gt._validate_gt_csv(record)
        quality_path = _resolve_path(
            gt_receipt.get("quality_json"),
            field=f"GT quality_json event {index + 1}",
            must_exist=True,
        )
        if (
            not _is_sha256(gt_receipt.get("quality_sha256"))
            or abc_video.sha256_file(quality_path)
            != gt_receipt["quality_sha256"]
        ):
            raise V81ComparisonError(
                f"GT quality receipt changed at event {index + 1}"
            )
        gt_receipt["quality_json"] = str(quality_path)
        frames = int(selection["frames"])
        if (
            len(gt_values) != frames
            or any(
                len(branches[arm]["raw_segments"][index]) != frames
                for arm in ARMS
            )
            or gt_receipt["safe_csv_sha256"]
            != selection["reference"]["declared_safe_csv_sha256"]
        ):
            raise V81ComparisonError(
                f"GT/control/treatment frame or hash mismatch at {index + 1}"
            )
        gt_segments.append(gt_values)
        gt_csv_receipts.append(deepcopy(gt_receipt))
        timeline.append(
            {
                **deepcopy(selection),
                "gt": gt_receipt,
                "metrics": {
                    "gt": _metric_receipt(gt_values),
                    **{
                        arm: _metric_receipt(
                            branches[arm]["raw_segments"][index]
                        )
                        for arm in ARMS
                    },
                },
                "planner_duration_diagnostics_sec": {
                    arm: branches[arm]["events"][index][
                        "planner_predicted_duration_sec"
                    ]
                    for arm in ARMS
                },
                "same_initial_noise_sha256": branches[CONTROL_ARM][
                    "events"
                ][index]["initial_noise_sha256"],
                "static_padding_frames": 0,
                "temporal_padding_frames": 0,
                "endpoint_hold_frames": 0,
                "additional_smoothing": False,
                "time_warp": False,
            }
        )
    gt_display = np.concatenate(gt_segments, axis=0)
    if gt_display.shape != (EXPECTED_FRAMES, ACTION_DIM):
        raise V81ComparisonError("held-out GT montage shape changed")

    render = validated["render"]
    pane_width = int(render["pane_width"])
    pane_height = int(render["pane_height"])
    panel_width = int(render["panel_width"])
    output_dir.mkdir(parents=True, exist_ok=True)
    render_receipts = {}
    videos = []
    lanes = (
        ("gt", gt_display),
        (CONTROL_ARM, branches[CONTROL_ARM]["display"]),
        (TREATMENT_ARM, branches[TREATMENT_ARM]["display"]),
    )
    shared_camera = _shared_union_camera(
        [trajectory for _, trajectory in lanes],
        width=pane_width,
        height=pane_height,
        simplified=bool(render["simplified"]),
    )
    for name, trajectory in lanes:
        csv_path = output_dir / f"{name}_native_no_padding.csv"
        mp4_path = output_dir / f"{name}_robot.mp4"
        render_receipts[name] = _render_with_shared_camera(
            trajectory,
            csv_path=csv_path,
            mp4_path=mp4_path,
            fps=FPS,
            width=pane_width,
            height=pane_height,
            simplified=bool(render["simplified"]),
            camera=shared_camera,
        )
        videos.append(mp4_path)
    side_by_side = output_dir / "GT_control_treatment_fixed_camera.mp4"
    stack_receipt = abc_video.build_side_by_side(
        videos,
        side_by_side,
        labels=(
            "GT · HELD-OUT BEAT2 18D",
            ARM_LABELS[CONTROL_ARM],
            ARM_LABELS[TREATMENT_ARM],
        ),
        pane_width=pane_width,
    )
    responses = {
        arm: branches[arm]["diagnostic_response"] for arm in ARMS
    }
    ass_path = output_dir / ASS_FILENAME
    ass_path.write_text(
        build_ass_document(
            timeline,
            responses=responses,
            best_steps={
                arm: int(
                    completion["arms"][arm]["checkpoint_v8_1_step"]
                )
                for arm in ARMS
            },
            robot_width=pane_width * 3,
            panel_width=panel_width,
            height=pane_height + 40,
            evidence_mode=evidence_mode,
        ),
        encoding="utf-8",
    )
    final_receipt = prior_gt._compose_final(
        side_by_side=side_by_side,
        ass_path=ass_path,
        output_path=final_video,
        robot_width=pane_width * 3,
        panel_width=panel_width,
        height=pane_height + 40,
    )
    trajectory_path = output_dir / TRAJECTORY_FILENAME
    trajectory_arrays = {
        "gt_native_events": gt_display,
        "prompts": np.asarray(prompts),
        "target_emotions": np.asarray(
            [item["target_emotion"] for item in timeline]
        ),
        "seeds": np.asarray(
            [int(item["seed"]) for item in timeline], dtype=np.int64
        ),
        "event_frames": np.asarray(
            [int(item["frames"]) for item in timeline], dtype=np.int64
        ),
        "event_offsets": np.asarray(
            [0] + [int(item["end_frame"]) for item in timeline],
            dtype=np.int64,
        ),
        "fps": np.asarray(FPS, dtype=np.float32),
        "static_padding_frames": np.asarray(0, dtype=np.int64),
        "endpoint_hold_frames": np.asarray(0, dtype=np.int64),
        "joint_order": np.asarray(JOINT_ORDER_18D),
    }
    if audit_mode:
        trajectory_arrays.update(
            {
                "control_matched_step_audit_native_events": branches[
                    CONTROL_ARM
                ]["display"],
                "treatment_matched_step_audit_native_events": branches[
                    TREATMENT_ARM
                ]["display"],
            }
        )
    else:
        trajectory_arrays.update(
            {
                "control_formal_best_native_events": branches[CONTROL_ARM][
                    "display"
                ],
                "treatment_formal_best_native_events": branches[
                    TREATMENT_ARM
                ]["display"],
            }
        )
    _atomic_npz(trajectory_path, **trajectory_arrays)
    branch_records = {}
    for arm in ARMS:
        record = {
            "label": ARM_LABELS[arm],
            "checkpoint_admission": branches[arm]["checkpoint_admission"],
            "diagnostic_response": branches[arm]["diagnostic_response"],
            "events": branches[arm]["events"],
            "fixed_frames_equal_gt": True,
            "static_padding_frames": 0,
            "temporal_padding_frames": 0,
            "endpoint_hold_frames": 0,
            "additional_smoothing": False,
        }
        if audit_mode:
            record.update(
                {
                    "audit_summary": completion["arms"][arm]["summary"],
                    "audit_summary_sha256": completion["arms"][arm][
                        "summary_sha256"
                    ],
                    "matched_step_audit_checkpoint": completion["arms"][arm][
                        "checkpoint"
                    ],
                    "matched_step_audit_checkpoint_sha256": completion[
                        "arms"
                    ][arm]["checkpoint_sha256"],
                    "formal_candidate": False,
                    "deterministic_replay_checks": completion["arms"][arm][
                        "deterministic_replay_checks"
                    ],
                }
            )
        else:
            record.update(
                {
                    "formal_summary": completion["arms"][arm]["summary"],
                    "formal_summary_sha256": completion["arms"][arm][
                        "summary_sha256"
                    ],
                    "formal_best_checkpoint": completion["arms"][arm][
                        "checkpoint"
                    ],
                    "formal_best_checkpoint_sha256": completion["arms"][arm][
                        "checkpoint_sha256"
                    ],
                }
            )
        branch_records[arm] = record
    comparison_contract = (
        MATCHED_AUDIT_COMPARISON_CONTRACT if audit_mode else COMPARISON_CONTRACT
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": (
            MATCHED_AUDIT_SUMMARY_ARTIFACT_KIND
            if audit_mode
            else SUMMARY_ARTIFACT_KIND
        ),
        "status": "complete",
        "experimental_only": True,
        "formal_release_eligible": False,
        "created_utc": utc_now(),
        "config": validated["_config_path"],
        "config_sha256": validated["_config_sha256"],
        "data_policy": (
            MATCHED_AUDIT_DATA_POLICY if audit_mode else DATA_POLICY
        ),
        "no_kimodo": True,
        "kimodo_admitted_count": 0,
        "hanyang_source_count": trainer.HANYANG_SOURCE_POOL_COUNT,
        "hanyang_boundary_excluded_count": trainer.HANYANG_BOUNDARY_COUNT,
        "hanyang_training_eligible_count": (
            trainer.HANYANG_TRAINING_ELIGIBLE_COUNT
        ),
        "hanyang_boundary_admitted_count": 0,
        "supervisor_receipt": deepcopy(
            dict(completion["supervisor_receipt"])
        ),
        "checkpoint_steps": {
            arm: int(completion["arms"][arm]["checkpoint_v8_1_step"])
            for arm in ARMS
        },
        "same_checkpoint_step": (
            completion["arms"][CONTROL_ARM]["checkpoint_v8_1_step"]
            == completion["arms"][TREATMENT_ARM]["checkpoint_v8_1_step"]
        ),
        "strict_single_variable_causal_claim": bool(audit_mode),
        "smoke_checkpoint_used": False,
        "target_duration_sec": 60.0,
        "decoded_frames_expected": EXPECTED_FRAMES,
        "native_event_count": EXPECTED_EVENTS,
        "static_padding_frames": 0,
        "temporal_padding_frames": 0,
        "endpoint_hold_frames": 0,
        "additional_smoothing": False,
        "shared_union_camera": shared_camera,
        "all_lanes_exact_same_camera": all(
            render_receipts[name]["shared_union_camera"] == shared_camera
            for name, _ in lanes
        ),
        "comparison_contract": deepcopy(comparison_contract),
        "metric_contract": deepcopy(METRIC_CONTRACT),
        "held_out_gt_plan": deepcopy(plan["held_out_gt_plan"]),
        "held_out_gt_csvs": gt_csv_receipts,
        "gt_disclosure": {
            "representation": (
                "held_out_BEAT2_physical_qc_passed_18d_robot_motion"
            ),
            "raw_human_skeleton": False,
            "paired_deterministic_ground_truth": False,
            "role": "same_prompt_qualitative_target_example",
            "loaded_after_both_generator_branches": True,
            "generation_input": False,
            "upstream_retarget_may_have_retimed_source": True,
            "no_additional_video_stage_retiming": True,
        },
        "timeline": timeline,
        "branches": branch_records,
        "selected_qwen_condition_cache": {
            "path": str(condition_cache.resolve()),
            "sha256": cache_receipt["sha256"],
            "variant": selected["selected_qwen_variant"],
        },
        "metrics": {
            "aggregate_raw_segment_mean": {
                lane: {
                    metric: float(
                        np.mean(
                            [
                                item["metrics"][lane][metric]
                                for item in timeline
                            ]
                        )
                    )
                    for metric in (
                        "jerk_rms_rad_s3",
                        "expression_amplitude_joint_range_rms_rad",
                        "head_velocity_rms_rad_s",
                    )
                }
                for lane in ("gt", *ARMS)
            },
            (
                "matched_step_beat2_response_retention"
                if audit_mode
                else "formal_beat2_response_retention"
            ): responses,
            "event_boundary_derivatives_excluded": True,
            "emotion_accuracy_claimed": False,
        },
        "artifacts": {
            "comparison_trajectories": {
                "path": str(trajectory_path),
                "sha256": abc_video.sha256_file(trajectory_path),
            },
            "text_emotion_jerk_expression_response_ass": {
                "path": str(ass_path),
                "sha256": abc_video.sha256_file(ass_path),
            },
            "side_by_side_robot_video": stack_receipt,
            "final_video": final_receipt,
            "render_receipts": render_receipts,
        },
    }
    if audit_mode:
        summary.update(
            {
                "evidence_mode": "matched_step_audit",
                "audit_step": int(completion["audit_step"]),
                "audit_purpose": MATCHED_AUDIT_PURPOSE,
                "audit_checkpoints_are_formal_candidates": False,
                "formal_schedule_target_steps": EXPECTED_FORMAL_STEPS,
                "paired_audit_metrics": deepcopy(
                    dict(completion["paired_metrics"])
                ),
                "matched_step_completion_sha256": completion["sha256"],
            }
        )
    else:
        summary.update(
            {
                "formal_steps_per_arm": EXPECTED_FORMAL_STEPS,
                "formal_best_admissible_only": True,
                "formal_best_steps": deepcopy(summary["checkpoint_steps"]),
                "same_best_step": summary["same_checkpoint_step"],
            }
        )
    summary["sha256"] = abc_video.value_sha256(summary)
    _atomic_json(summary_path, summary)
    return summary


def build_matched_step_audit_comparison(
    validated: Mapping[str, Any],
    *,
    audit_root: str | Path,
    output_dir: str | Path,
    step: int = 1500,
    overwrite: bool = False,
) -> dict[str, Any]:
    audit_validated = deepcopy(dict(validated))
    audit_output = Path(output_dir).expanduser().resolve()
    audit_validated["_output_dir"] = str(audit_output)
    completion = matched_step_audit_completion(
        audit_validated, audit_root=audit_root, step=step
    )
    return _build_comparison(
        audit_validated,
        overwrite=overwrite,
        completion_override=completion,
        evidence_mode="matched_step_audit",
        summary_filename=MATCHED_AUDIT_SUMMARY_FILENAME,
        final_video_filename=MATCHED_AUDIT_FINAL_VIDEO_FILENAME,
    )


def validate_completed_matched_step_audit(
    validated: Mapping[str, Any],
    *,
    audit_root: str | Path,
    output_dir: str | Path,
    step: int = 1500,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    path = output / MATCHED_AUDIT_SUMMARY_FILENAME
    summary = _load_json(path, field="completed matched-step video summary")
    unsigned = deepcopy(summary)
    claimed = unsigned.pop("sha256", None)
    completion = matched_step_audit_completion(
        validated, audit_root=audit_root, step=step
    )
    if (
        claimed != abc_video.value_sha256(unsigned)
        or summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("artifact_kind")
        != MATCHED_AUDIT_SUMMARY_ARTIFACT_KIND
        or summary.get("status") != "complete"
        or summary.get("evidence_mode") != "matched_step_audit"
        or summary.get("audit_step") != step
        or summary.get("audit_purpose") != MATCHED_AUDIT_PURPOSE
        or summary.get("audit_checkpoints_are_formal_candidates") is not False
        or summary.get("formal_release_eligible") is not False
        or summary.get("strict_single_variable_causal_claim") is not True
        or summary.get("checkpoint_steps")
        != {arm: step for arm in ARMS}
        or summary.get("same_checkpoint_step") is not True
        or summary.get("paired_audit_metrics")
        != completion["paired_metrics"]
        or summary.get("matched_step_completion_sha256")
        != completion["sha256"]
        or summary.get("no_kimodo") is not True
        or summary.get("kimodo_admitted_count") != 0
        or summary.get("hanyang_boundary_admitted_count") != 0
        or summary.get("smoke_checkpoint_used") is not False
        or summary.get("static_padding_frames") != 0
        or summary.get("temporal_padding_frames") != 0
        or summary.get("endpoint_hold_frames") != 0
        or summary.get("additional_smoothing") is not False
        or summary.get("all_lanes_exact_same_camera") is not True
        or summary.get("comparison_contract")
        != MATCHED_AUDIT_COMPARISON_CONTRACT
        or len(summary.get("held_out_gt_csvs") or []) != EXPECTED_EVENTS
    ):
        raise V81ComparisonError("matched-step video summary changed")
    for gt in summary["held_out_gt_csvs"]:
        csv_path = _resolve_path(
            gt.get("safe_csv"), field="audit summary GT CSV", must_exist=True
        )
        quality_path = _resolve_path(
            gt.get("quality_json"),
            field="audit summary GT quality JSON",
            must_exist=True,
        )
        if (
            abc_video.sha256_file(csv_path) != gt.get("safe_csv_sha256")
            or abc_video.sha256_file(quality_path) != gt.get("quality_sha256")
        ):
            raise V81ComparisonError("matched-step GT artifact changed")
    for arm in ARMS:
        branch = (summary.get("branches") or {}).get(arm)
        current = completion["arms"][arm]
        if (
            not isinstance(branch, Mapping)
            or branch.get("formal_candidate") is not False
            or branch.get("deterministic_replay_checks")
            != current["deterministic_replay_checks"]
            or not all(branch["deterministic_replay_checks"].values())
        ):
            raise V81ComparisonError(f"matched-step branch changed: {arm}")
        for path_key, sha_key, expected_path, expected_sha in (
            (
                "audit_summary",
                "audit_summary_sha256",
                current["summary"],
                current["summary_sha256"],
            ),
            (
                "matched_step_audit_checkpoint",
                "matched_step_audit_checkpoint_sha256",
                current["checkpoint"],
                current["checkpoint_sha256"],
            ),
        ):
            artifact = _resolve_path(
                branch.get(path_key),
                field=f"matched-step {arm} {path_key}",
                must_exist=True,
            )
            if (
                str(artifact) != expected_path
                or branch.get(sha_key) != expected_sha
                or abc_video.sha256_file(artifact) != expected_sha
            ):
                raise V81ComparisonError(
                    f"matched-step {arm} artifact SHA256 changed"
                )
    artifacts = summary.get("artifacts") or {}
    final = artifacts.get("final_video") or {}
    final_path = _resolve_path(
        final.get("path"), field="matched-step final video", must_exist=True
    )
    if (
        final_path != output / MATCHED_AUDIT_FINAL_VIDEO_FILENAME
        or abc_video.sha256_file(final_path) != final.get("sha256")
        or int(final.get("decoded_frames", -1)) != EXPECTED_FRAMES
        or not math.isclose(
            float(final.get("duration_sec", -1.0)),
            60.0,
            abs_tol=1 / FPS,
        )
    ):
        raise V81ComparisonError("matched-step final video changed")
    return {
        "summary": str(path),
        "summary_file_sha256": abc_video.sha256_file(path),
        "summary_canonical_sha256": claimed,
        "video": str(final_path),
        "video_sha256": final["sha256"],
        "duration_sec": float(final["duration_sec"]),
        "decoded_frames": int(final["decoded_frames"]),
        "audit_step": step,
        "hanyang_relative_improvement": completion["paired_metrics"][
            "hanyang_relative_improvement"
        ],
        "formal_candidate": False,
        "no_kimodo": True,
        "kimodo_admitted_count": 0,
    }


def validate_completed_output(
    validated: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate the sealed summary and its externally stored artifacts."""

    _reject_claim_invalidated(validated["_output_dir"])
    path = Path(validated["_output_dir"]) / SUMMARY_FILENAME
    summary = _load_json(path, field="completed V8.1 video summary")
    unsigned = deepcopy(summary)
    claimed = unsigned.pop("sha256", None)
    if (
        claimed != abc_video.value_sha256(unsigned)
        or summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("artifact_kind") != SUMMARY_ARTIFACT_KIND
        or summary.get("status") != "complete"
        or summary.get("no_kimodo") is not True
        or summary.get("kimodo_admitted_count") != 0
        or summary.get("hanyang_training_eligible_count")
        != trainer.HANYANG_TRAINING_ELIGIBLE_COUNT
        or summary.get("hanyang_boundary_admitted_count") != 0
        or summary.get("smoke_checkpoint_used") is not False
        or summary.get("formal_best_admissible_only") is not True
        or summary.get("static_padding_frames") != 0
        or summary.get("temporal_padding_frames") != 0
        or summary.get("endpoint_hold_frames") != 0
        or summary.get("additional_smoothing") is not False
        or summary.get("all_lanes_exact_same_camera") is not True
        or summary.get("comparison_contract") != COMPARISON_CONTRACT
        or len(summary.get("held_out_gt_csvs") or []) != EXPECTED_EVENTS
    ):
        raise V81ComparisonError("completed V8.1 video summary changed")
    for gt in summary["held_out_gt_csvs"]:
        csv_path = _resolve_path(
            gt.get("safe_csv"), field="summary GT CSV", must_exist=True
        )
        quality_path = _resolve_path(
            gt.get("quality_json"),
            field="summary GT quality JSON",
            must_exist=True,
        )
        if (
            abc_video.sha256_file(csv_path) != gt.get("safe_csv_sha256")
            or abc_video.sha256_file(quality_path)
            != gt.get("quality_sha256")
        ):
            raise V81ComparisonError("summary GT artifact SHA256 changed")
    for arm in ARMS:
        branch = (summary.get("branches") or {}).get(arm)
        if not isinstance(branch, Mapping):
            raise V81ComparisonError(f"summary branch missing: {arm}")
        for path_key, sha_key in (
            ("formal_summary", "formal_summary_sha256"),
            (
                "formal_best_checkpoint",
                "formal_best_checkpoint_sha256",
            ),
        ):
            artifact = _resolve_path(
                branch.get(path_key),
                field=f"summary {arm} {path_key}",
                must_exist=True,
            )
            if abc_video.sha256_file(artifact) != branch.get(sha_key):
                raise V81ComparisonError(
                    f"summary {arm} artifact SHA256 changed"
                )
    artifacts = summary.get("artifacts") or {}
    final = artifacts.get("final_video") or {}
    final_path = _resolve_path(
        final.get("path"), field="summary final video", must_exist=True
    )
    if (
        abc_video.sha256_file(final_path) != final.get("sha256")
        or int(final.get("decoded_frames", -1)) != EXPECTED_FRAMES
        or not math.isclose(
            float(final.get("duration_sec", -1.0)),
            60.0,
            abs_tol=1 / FPS,
        )
    ):
        raise V81ComparisonError("completed final video changed")
    return {
        "summary": str(path.resolve()),
        "summary_file_sha256": abc_video.sha256_file(path),
        "summary_canonical_sha256": claimed,
        "video": str(final_path),
        "video_sha256": final["sha256"],
        "duration_sec": float(final["duration_sec"]),
        "decoded_frames": int(final["decoded_frames"]),
        "no_kimodo": True,
        "kimodo_admitted_count": 0,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validated = read_config(args.config)
        if args.prepare_only:
            result = prepare_plan(validated)
            output = {
                "status": result["status"],
                "plan": str(
                    Path(validated["_output_dir"]) / PLAN_FILENAME
                ),
                "formal_completion": result["formal_completion"],
                "gpu_accessed": False,
                "generation_or_render_executed": False,
            }
        else:
            summary = _build_comparison(
                validated, overwrite=bool(args.overwrite)
            )
            output = {
                "status": summary["status"],
                "summary": str(
                    Path(validated["_output_dir"]) / SUMMARY_FILENAME
                ),
                "video": summary["artifacts"]["final_video"]["path"],
                "video_sha256": summary["artifacts"]["final_video"][
                    "sha256"
                ],
                "duration_sec": summary["artifacts"]["final_video"][
                    "duration_sec"
                ],
            }
    except (
        V81ComparisonError,
        supervisor.SupervisorError,
        v7_video.EmotionHierarchyVideoError,
        style_video.StyleEmotionVideoError,
        abc_video.EvaluationContractError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
