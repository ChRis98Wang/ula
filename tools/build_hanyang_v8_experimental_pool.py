#!/usr/bin/env python3
"""Build the hash-bound Hanyang v8 partial-18D experimental pool.

The pool is intentionally *not* a formal generator-foundation admission.  It
contains only source-faithful 150-frame/30-Hz partial-18D trajectories and
their observation-confidence tensors.  Raw IK and deployment-preview outputs
are validated as ineligible by policy and are never emitted as training
artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from upper_body_skeleton.data_source_registry import (  # noqa: E402
    EMOTION_CRITIC_ROLE,
    HANYANG_EMOTIONAL_BODY_SOURCE_ID,
    assert_no_forbidden_data_lineage,
    build_data_source_registry_contract,
    validate_data_source_registry_contract,
)
from upper_body_skeleton.hanyang_emotion_retarget import (  # noqa: E402
    DATASET_REVISION,
    SOURCE_FPS,
    SOURCE_FRAMES,
    fixed_split_for_participant,
    json_hash,
    sha256_file,
    stable_json,
)
from upper_body_skeleton.hanyang_expanded_training import (  # noqa: E402
    HANYANG_ACTION_ORDER_18D,
    HANYANG_P7_ORDER,
    HANYANG_Q2_ORDER,
    HANYANG_Q6_ORDER,
    PERMANENTLY_DISABLED_CONDITION_LANES,
    PERMANENTLY_UNOBSERVED_DOF_INDICES,
    hanyang_p7_to_hierarchy_targets,
    hanyang_training_admission,
    observation_weight_sha256,
    validate_observation_weight,
)


DEFAULT_RETARGET_ROOT = Path(
    "/home/gez/nas/cloud/gez/human_motion/external_emotion_research/"
    "hanyang_emotional_body_motion_zenodo_10052504_v1/retarget_v1"
)
DEFAULT_PASSED_MANIFEST = DEFAULT_RETARGET_ROOT / "passed_manifest.jsonl"
DEFAULT_OUTPUT_DIR = DEFAULT_RETARGET_ROOT / "experimental_pool_v8"
FINAL_PASSED_MANIFEST_SHA256 = (
    "be9dcf53f0aa2acc2695475e8625d1f05550a07fb8403e46d0e0c8e3f633daab"
)
EXPECTED_PASSED_COUNT = 344
POOL_ROW_KIND = "hanyang_v8_experimental_partial18d_pool_row_v1"
POOL_RECEIPT_KIND = "hanyang_v8_experimental_pool_admission_receipt_v1"
REVIEW_ROW_KIND = "hanyang_v8_source_faithful_human_review_row_v1"
SCHEMA_VERSION = "1.0.0"
REQUIRED_QUALITY_GATES = (
    "collision_pass",
    "fixed_150_frames_30hz_pass",
    "head_tilt_proxy_pass",
    "joint_limits_pass",
    "limb_direction_pass",
    "passed",
    "per_joint_velocity_pass",
    "retime_factor_exactly_one_pass",
    "saturation_pass",
    "source_geometry_pass",
    "target_fit_pass",
)
REQUIRED_SPLITS = ("train", "validation", "test")
STRICT_GLOBAL_EMOTION_COVERAGE_POLICY = (
    "every_p7_class_has_locally_admissible_examples_in_train_validation_test_v1"
)
FORMAL_REJECTION_REASON = "insufficient_non_neutral_coverage"
SOURCE_FAITHFUL_ENDPOINT_POLICY = (
    "source_first_and_last_frames_preserved_no_terminal_hold"
)
_NEGATIVE_PROOF_KEY = "kimodo_accessed_or_used"
_FORBIDDEN_TOKEN = "kimodo"
EXPECTED_REVIEW_CLIP_IDS = (
    "hanyang:11_1_4_1",
    "hanyang:18_2_3_2",
    "hanyang:28_4_4_2",
    "hanyang:2_3_4_3",
    "hanyang:23_1_2_3",
    "hanyang:26_2_4_3",
    "hanyang:13_3_1_4",
    "hanyang:24_1_5_4",
    "hanyang:27_3_1_4",
    "hanyang:3_1_3_5",
    "hanyang:27_2_1_5",
    "hanyang:13_3_3_6",
    "hanyang:23_3_1_6",
    "hanyang:28_1_1_6",
    "hanyang:1_2_4_7",
    "hanyang:22_1_1_7",
    "hanyang:26_1_1_7",
    "hanyang:19_3_2_5",
    "hanyang:13_2_4_4",
    "hanyang:24_3_4_4",
    "hanyang:6_2_1_6",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _record_hash(record: Mapping[str, Any]) -> str:
    return json_hash(
        {key: value for key, value in record.items() if key != "record_sha256"}
    )


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(stable_json(dict(row)) + "\n")
    os.replace(temporary, path)


def _reject_forbidden_dataset(
    payload: object,
    *,
    context: str,
) -> None:
    """Fail closed on forbidden source references while allowing one proof.

    The literal negative proof ``kimodo_accessed_or_used: false`` is allowed.
    Every other key, string, and path is scanned after punctuation removal.
    """

    def normalized(value: object) -> str:
        return "".join(character for character in str(value).casefold() if character.isalnum())

    def walk(value: object, field: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key == _NEGATIVE_PROOF_KEY:
                    if child is not False:
                        raise ValueError(
                            f"{field}.{key} must be exactly false"
                        )
                    continue
                walk(key, f"{field}.<key>")
                walk(child, f"{field}.{key}")
            return
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for index, child in enumerate(value):
                walk(child, f"{field}[{index}]")
            return
        if isinstance(value, os.PathLike):
            walk(os.fspath(value), field)
            return
        if isinstance(value, str) and _FORBIDDEN_TOKEN in normalized(value):
            raise ValueError(
                f"{field} contains a permanently forbidden dataset token"
            )

    walk(payload, context)


def _require_descendant(path: Path, root: Path, *, context: str) -> Path:
    resolved = path.resolve()
    root = root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{context} escapes the retarget root: {resolved}") from error
    return resolved


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            rows.append(value)
    return rows


def _validate_csv(path: Path) -> None:
    with path.open("r", encoding="utf-8") as handle:
        header = tuple(handle.readline().strip().split(","))
    if header != HANYANG_ACTION_ORDER_18D:
        raise ValueError(f"source-faithful CSV header changed: {path}")
    values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if values.shape != (SOURCE_FRAMES, len(HANYANG_ACTION_ORDER_18D)):
        raise ValueError(
            f"source-faithful CSV must have shape (150,18): {path}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"source-faithful CSV contains non-finite values: {path}")
    if np.count_nonzero(values[:, PERMANENTLY_UNOBSERVED_DOF_INDICES]) != 0:
        raise ValueError(
            f"source-faithful CSV invents permanently unobserved DOFs: {path}"
        )


def _validate_confidence(path: Path) -> tuple[np.ndarray, str]:
    confidence = np.load(path, allow_pickle=False)
    if confidence.shape != (SOURCE_FRAMES, len(HANYANG_ACTION_ORDER_18D)):
        raise ValueError(f"observation confidence must have shape (150,18): {path}")
    if not np.issubdtype(confidence.dtype, np.floating):
        raise ValueError(f"observation confidence must be floating point: {path}")
    if not np.isfinite(confidence).all():
        raise ValueError(f"observation confidence contains non-finite values: {path}")
    if np.any((confidence < 0.0) | (confidence > 1.0)):
        raise ValueError(f"observation confidence must lie in [0,1]: {path}")
    if np.count_nonzero(confidence[:, PERMANENTLY_UNOBSERVED_DOF_INDICES]) != 0:
        raise ValueError(
            f"observation confidence enables permanently unobserved DOFs: {path}"
        )
    tensor = torch.from_numpy(np.ascontiguousarray(confidence)).unsqueeze(0)
    validate_observation_weight(tensor)
    return confidence, observation_weight_sha256(tensor)


def _float_list(value: torch.Tensor) -> list[float]:
    return [float(item) for item in value.detach().cpu().reshape(-1).tolist()]


def _soft_targets(
    emotion_evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    distribution = emotion_evaluation.get("soft_emotion_distribution")
    if not isinstance(distribution, Mapping):
        raise ValueError("quality record has no P7 soft-emotion distribution")
    if set(distribution) != set(HANYANG_P7_ORDER):
        raise ValueError("P7 soft-emotion distribution classes changed")
    p7 = torch.tensor(
        [distribution[name] for name in HANYANG_P7_ORDER],
        dtype=torch.float64,
    )
    targets = hanyang_p7_to_hierarchy_targets(p7)
    return {
        "contract": "hanyang_p7_v7_hierarchy_soft_targets_v1",
        "p7_order": list(HANYANG_P7_ORDER),
        "p7": _float_list(p7),
        "q2_order": list(HANYANG_Q2_ORDER),
        "q2": _float_list(targets.q2),
        "q6_order": list(HANYANG_Q6_ORDER),
        "q6": _float_list(targets.q6),
        "disgust_mass": float(targets.disgust_mass.item()),
        "q6_supervision_weight": float(
            targets.six_class_supervision_weight.item()
        ),
    }


def _validate_and_build_local_row(
    upstream: Mapping[str, Any],
    *,
    retarget_root: Path,
    upstream_manifest_sha256: str,
) -> dict[str, Any]:
    _reject_forbidden_dataset(upstream, context="upstream_passed_row")
    assert_no_forbidden_data_lineage(upstream, context="upstream_passed_row")
    if upstream.get(_NEGATIVE_PROOF_KEY) is not False:
        raise ValueError("upstream passed row lacks the negative isolation proof")
    if upstream.get("record_sha256") != _record_hash(upstream):
        raise ValueError("upstream passed row record hash changed")
    if upstream.get("status") != "passed":
        raise ValueError("experimental pool accepts only passed retarget rows")
    if upstream.get("dataset_id") != HANYANG_EMOTIONAL_BODY_SOURCE_ID:
        raise ValueError("unexpected dataset ID in passed retarget row")

    clip_id = str(upstream.get("clip_id") or "")
    source_stem = str(upstream.get("source_stem") or "")
    if clip_id != f"hanyang:{source_stem}":
        raise ValueError(f"clip identity changed: {clip_id!r}")
    participant_id = int(upstream["participant_id"])
    fixed_split = fixed_split_for_participant(participant_id)
    if upstream.get("fixed_split_assignment") != fixed_split:
        raise ValueError(f"participant-disjoint split changed for {clip_id}")

    quality_path = _require_descendant(
        Path(str(upstream["quality_json"])),
        retarget_root / "clips" / source_stem,
        context=f"{clip_id}.quality_json",
    )
    if quality_path.name != "quality.json" or not quality_path.is_file():
        raise ValueError(f"missing canonical quality report for {clip_id}")
    quality_sha256 = sha256_file(quality_path)
    if quality_sha256 != upstream.get("quality_json_sha256"):
        raise ValueError(f"quality report file hash changed for {clip_id}")
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    _reject_forbidden_dataset(quality, context=f"{clip_id}.quality")
    assert_no_forbidden_data_lineage(quality, context=f"{clip_id}.quality")
    if quality.get("record_sha256") != _record_hash(quality):
        raise ValueError(f"quality record hash changed for {clip_id}")
    if quality.get("record_sha256") != upstream.get("quality_record_sha256"):
        raise ValueError(f"quality record binding changed for {clip_id}")
    for key, expected in (
        ("clip_id", clip_id),
        ("dataset_id", HANYANG_EMOTIONAL_BODY_SOURCE_ID),
        ("participant_id", participant_id),
        ("fixed_split_assignment", fixed_split),
        ("source_frames", SOURCE_FRAMES),
        ("source_fps", SOURCE_FPS),
    ):
        if quality.get(key) != expected:
            raise ValueError(f"{clip_id} quality identity field {key!r} changed")
    if quality.get("action_dim") != len(HANYANG_ACTION_ORDER_18D):
        raise ValueError(f"{clip_id} quality action dimension changed")
    action_dim_mask = tuple(bool(value) for value in quality["action_dim_mask"])
    expected_mask = tuple(
        index not in PERMANENTLY_UNOBSERVED_DOF_INDICES
        for index in range(len(HANYANG_ACTION_ORDER_18D))
    )
    if action_dim_mask != expected_mask:
        raise ValueError(f"{clip_id} partial-18D action mask changed")
    quality_gate = quality.get("quality_gate")
    if not isinstance(quality_gate, Mapping) or any(
        quality_gate.get(name) is not True for name in REQUIRED_QUALITY_GATES
    ):
        raise ValueError(f"{clip_id} did not pass every required quality gate")
    if dict(quality_gate) != upstream.get("quality_gate"):
        raise ValueError(f"{clip_id} upstream quality-gate binding changed")
    smoothing = quality.get("smoothing")
    if (
        not isinstance(smoothing, Mapping)
        or smoothing.get("retime_factor") != 1.0
        or smoothing.get("retimed") is not False
    ):
        raise ValueError(f"{clip_id} source-faithful timing changed")
    smoothing_window = int(smoothing.get("smoothing_window", 0))
    if smoothing_window < 1 or smoothing_window % 2 != 1:
        raise ValueError(f"{clip_id} has an invalid upstream smoothing window")
    if smoothing.get("endpoint_policy") != SOURCE_FAITHFUL_ENDPOINT_POLICY:
        raise ValueError(f"{clip_id} source-faithful endpoint policy changed")

    outputs = quality.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError(f"{clip_id} quality report has no outputs")
    trajectory_path = _require_descendant(
        Path(str(outputs["source_faithful_partial_18d_csv"])),
        retarget_root / "clips" / source_stem,
        context=f"{clip_id}.source_faithful_partial_18d_csv",
    )
    if trajectory_path.name != f"{source_stem}_source_faithful_partial_18d.csv":
        raise ValueError(f"{clip_id} source-faithful artifact name changed")
    if not trajectory_path.is_file():
        raise FileNotFoundError(trajectory_path)
    trajectory_sha256 = sha256_file(trajectory_path)
    if trajectory_sha256 != outputs.get(
        "source_faithful_partial_18d_csv_sha256"
    ):
        raise ValueError(f"{clip_id} source-faithful CSV hash changed")
    _validate_csv(trajectory_path)

    confidence_record = quality.get("per_frame_observation_confidence")
    if not isinstance(confidence_record, Mapping):
        raise ValueError(f"{clip_id} has no observation-confidence record")
    confidence_path = _require_descendant(
        Path(str(confidence_record["path"])),
        retarget_root / "clips" / source_stem,
        context=f"{clip_id}.observation_confidence",
    )
    if confidence_path.name != "observation_confidence_18d.npy":
        raise ValueError(f"{clip_id} confidence artifact name changed")
    if not confidence_path.is_file():
        raise FileNotFoundError(confidence_path)
    confidence_file_sha256 = sha256_file(confidence_path)
    if confidence_file_sha256 != confidence_record.get("sha256"):
        raise ValueError(f"{clip_id} observation-confidence file hash changed")
    confidence, confidence_tensor_sha256 = _validate_confidence(confidence_path)
    if confidence_record.get("shape") != [SOURCE_FRAMES, len(HANYANG_ACTION_ORDER_18D)]:
        raise ValueError(f"{clip_id} confidence shape receipt changed")
    confidence_statistics = {
        "minimum": float(confidence.min()),
        "mean": float(confidence.mean()),
        "maximum": float(confidence.max()),
        "positive_fraction": float(np.count_nonzero(confidence) / confidence.size),
    }
    trajectory_quality = quality.get("trajectory")
    if not isinstance(trajectory_quality, Mapping):
        raise ValueError(f"{clip_id} quality report has no trajectory metrics")
    kinematic_quality_summary = {
        "rms_jerk_rad_s3": float(trajectory_quality["rms_jerk_rad_s3"]),
        "max_acceleration_rad_s2": float(
            trajectory_quality["max_acceleration_rad_s2"]
        ),
        "max_velocity_limit_ratio": float(
            trajectory_quality["max_velocity_limit_ratio"]
        ),
        "limb_target_error_p95_m": float(
            quality["limb_target_error_p95_m"]
        ),
        "limb_direction_error_p95_deg": float(
            quality["limb_direction_error_p95_deg"]
        ),
        "upper_body_collision_frame_rate": float(
            quality["upper_body_collision_frame_rate"]
        ),
    }
    if not all(
        np.isfinite(value) for value in kinematic_quality_summary.values()
    ):
        raise ValueError(f"{clip_id} has non-finite kinematic quality metrics")

    emotion_evaluation = quality.get("emotion_evaluation")
    if not isinstance(emotion_evaluation, Mapping):
        raise ValueError(f"{clip_id} quality report has no human evaluation")
    soft_targets = _soft_targets(emotion_evaluation)
    intended_high_confidence = bool(
        emotion_evaluation.get("intended_high_confidence", False)
    )
    intended_majority_agrees = emotion_evaluation.get(
        "intended_majority_agrees"
    )
    rater_coverage_pass = emotion_evaluation.get("rater_coverage_pass")
    local_admission = hanyang_training_admission(
        qc_pass=True,
        rater_coverage_pass=rater_coverage_pass,
        intended_majority_agrees=intended_majority_agrees,
        intended_share=emotion_evaluation.get("intended_share"),
        lineage={
            "dataset_id": HANYANG_EMOTIONAL_BODY_SOURCE_ID,
            "clip_id": clip_id,
            "source_manifest": str(retarget_root / "passed_manifest.jsonl"),
            "quality_path": str(quality_path),
            "trajectory_path": str(trajectory_path),
            "confidence_path": str(confidence_path),
        },
    )
    if any(
        local_admission[f"{lane}_condition_eligible"] is not False
        for lane in PERMANENTLY_DISABLED_CONDITION_LANES
    ):
        raise ValueError(f"{clip_id} enabled a permanently disabled condition lane")

    row: dict[str, Any] = {
        "artifact_kind": POOL_ROW_KIND,
        "schema_version": SCHEMA_VERSION,
        "clip_id": clip_id,
        "dataset_id": HANYANG_EMOTIONAL_BODY_SOURCE_ID,
        "dataset_revision": DATASET_REVISION,
        "source_stem": source_stem,
        "source_sha256": quality["source_sha256"],
        "participant_id": participant_id,
        "source_group_key": quality["source_group_key"],
        "fixed_split_assignment": fixed_split,
        "emotion_id": quality["emotion_id"],
        "frames": SOURCE_FRAMES,
        "fps": SOURCE_FPS,
        "action_dim": len(HANYANG_ACTION_ORDER_18D),
        "action_order": list(HANYANG_ACTION_ORDER_18D),
        "action_dim_mask": list(action_dim_mask),
        "quality_gate": dict(quality_gate),
        "quality_json": str(quality_path),
        "quality_json_sha256": quality_sha256,
        "quality_record_sha256": quality["record_sha256"],
        "source_faithful_partial_18d_csv": str(trajectory_path),
        "source_faithful_partial_18d_csv_sha256": trajectory_sha256,
        "observation_confidence_npy": str(confidence_path),
        "observation_confidence_npy_sha256": confidence_file_sha256,
        "observation_confidence_tensor_sha256": confidence_tensor_sha256,
        "observation_confidence_shape": list(confidence.shape),
        "observation_confidence_statistics": confidence_statistics,
        "kinematic_quality_summary": kinematic_quality_summary,
        "source_faithful_processing": {
            "upstream_smoothing_preserved": True,
            "upstream_source_faithful_smoothing_window": smoothing_window,
            "endpoints_preserved": True,
            "endpoint_policy": SOURCE_FAITHFUL_ENDPOINT_POLICY,
            "additional_review_smoothing": False,
            "retimed": False,
            "retime_factor": 1.0,
        },
        "label_audit": {
            "intended_share": float(emotion_evaluation["intended_share"]),
            "intended_high_confidence": intended_high_confidence,
            "intended_majority_agrees": intended_majority_agrees,
            "rater_coverage_pass": rater_coverage_pass,
        },
        "soft_emotion_targets": soft_targets,
        "local_admission": local_admission,
        "unconditional_motion_eligible": local_admission[
            "unconditional_motion_eligible"
        ],
        "local_emotion_condition_candidate": local_admission[
            "emotion_condition_eligible"
        ],
        # The global coverage gate is applied after every row is validated.
        "emotion_condition_eligible": False,
        "emotion_condition_mask": False,
        "group54_condition_eligible": False,
        "group54_condition_mask": False,
        "style_condition_eligible": False,
        "style_condition_mask": False,
        "duration_condition_eligible": False,
        "duration_condition_mask": False,
        "semantic_condition_eligible": False,
        "semantic_condition_mask": False,
        "admission_tier": "unconditional_motion_only_pending_global_gate",
        "formal_training_eligible": False,
        "generator_foundation_eligible": False,
        "deployment_preview_eligible": False,
        "raw_ik_eligible": False,
        _NEGATIVE_PROOF_KEY: False,
        "upstream_passed_manifest_sha256": upstream_manifest_sha256,
        "upstream_passed_record_sha256": upstream["record_sha256"],
    }
    row["record_sha256"] = _record_hash(row)
    return row


def _coverage_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    local = [
        row for row in rows if row["local_emotion_condition_candidate"] is True
    ]
    by_emotion = Counter(str(row["emotion_id"]) for row in local)
    by_split_emotion = Counter(
        (str(row["fixed_split_assignment"]), str(row["emotion_id"]))
        for row in local
    )
    missing_cells = [
        {"split": split, "emotion_id": emotion}
        for split in REQUIRED_SPLITS
        for emotion in HANYANG_P7_ORDER
        if by_split_emotion[(split, emotion)] == 0
    ]
    non_neutral = [
        row for row in local if row["emotion_id"] != "neutral"
    ]
    return {
        "policy": STRICT_GLOBAL_EMOTION_COVERAGE_POLICY,
        "passed": not missing_cells,
        "required_splits": list(REQUIRED_SPLITS),
        "required_emotions": list(HANYANG_P7_ORDER),
        "local_candidate_count": len(local),
        "local_candidate_count_by_emotion": {
            emotion: by_emotion[emotion] for emotion in HANYANG_P7_ORDER
        },
        "local_candidate_count_by_split_and_emotion": {
            split: {
                emotion: by_split_emotion[(split, emotion)]
                for emotion in HANYANG_P7_ORDER
            }
            for split in REQUIRED_SPLITS
        },
        "non_neutral_local_candidate_count": len(non_neutral),
        "non_neutral_local_candidate_count_by_split": {
            split: sum(
                row["fixed_split_assignment"] == split
                for row in non_neutral
            )
            for split in REQUIRED_SPLITS
        },
        "missing_split_emotion_cells": missing_cells,
        "failure_reason": None if not missing_cells else FORMAL_REJECTION_REASON,
    }


def _build_human_review_bundle(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Select 14--21 deterministic samples across labels, tiers, and extremes."""

    ordered_rows = sorted(rows, key=lambda row: str(row["clip_id"]))
    selected: dict[str, dict[str, Any]] = {}
    selection_order: list[str] = []

    def confidence_tier(row: Mapping[str, Any]) -> str:
        return (
            "strict_local_candidate"
            if row["local_emotion_condition_candidate"] is True
            else "motion_only"
        )

    def pick(candidates: Iterable[Mapping[str, Any]], reason: str) -> None:
        candidates = list(candidates)
        if not candidates:
            return
        row = min(candidates, key=lambda item: str(item["clip_id"]))
        entry = selected.setdefault(
            str(row["clip_id"]),
            {"row": row, "reasons": set()},
        )
        if str(row["clip_id"]) not in selection_order:
            selection_order.append(str(row["clip_id"]))
        entry["reasons"].add(reason)

    # Cover all 17 nonempty intended-emotion × participant-split cells first.
    # Highest intended rater share wins; ties prefer high-confidence/agreement,
    # then the stable clip ID.  Emotion-major order is the review order.
    for emotion_id in HANYANG_P7_ORDER:
        for split in REQUIRED_SPLITS:
            candidates = [
                row
                for row in ordered_rows
                if row["fixed_split_assignment"] == split
                and row["emotion_id"] == emotion_id
            ]
            if not candidates:
                continue
            best = min(
                candidates,
                key=lambda row: (
                    -float(row["label_audit"]["intended_share"]),
                    -int(row["label_audit"]["intended_high_confidence"]),
                    -int(row["label_audit"]["intended_majority_agrees"]),
                    str(row["clip_id"]),
                ),
            )
            pick(
                (best,),
                f"split_emotion_coverage:{split}:{emotion_id}",
            )

    # Add four global QC sentinels after coverage, in this fixed order.
    extremes = (
        (
            max(
                ordered_rows,
                key=lambda row: row["kinematic_quality_summary"][
                    "max_velocity_limit_ratio"
                ],
            ),
            "quality_extreme:max_velocity_limit_ratio:maximum",
        ),
        (
            max(
                ordered_rows,
                key=lambda row: row["kinematic_quality_summary"][
                    "rms_jerk_rad_s3"
                ],
            ),
            "quality_extreme:rms_jerk_rad_s3:maximum",
        ),
        (
            min(
                ordered_rows,
                key=lambda row: row["observation_confidence_statistics"][
                    "mean"
                ],
            ),
            "quality_extreme:observation_confidence_mean:minimum",
        ),
        (
            max(
                ordered_rows,
                key=lambda row: row["kinematic_quality_summary"][
                    "limb_target_error_p95_m"
                ],
            ),
            "quality_extreme:limb_target_error_p95_m:maximum",
        ),
    )
    for row, reason in extremes:
        pick((row,), reason)
    if len(rows) == EXPECTED_PASSED_COUNT:
        if tuple(selection_order) != EXPECTED_REVIEW_CLIP_IDS:
            raise ValueError(
                "production review selection changed from its audited "
                "21-clip binding"
            )

    review_rows: list[dict[str, Any]] = []
    selected_entries = [selected[clip_id] for clip_id in selection_order]
    for selected_entry in selected_entries:
        source = selected_entry["row"]
        split = str(source["fixed_split_assignment"])
        emotion_id = str(source["emotion_id"])
        tier = confidence_tier(source)
        p7 = source["soft_emotion_targets"]
        intended_index = p7["p7_order"].index(emotion_id)
        review_row: dict[str, Any] = {
            "artifact_kind": REVIEW_ROW_KIND,
            "schema_version": SCHEMA_VERSION,
            "clip_id": source["clip_id"],
            "dataset_id": source["dataset_id"],
            "participant_id": source["participant_id"],
            "fixed_split_assignment": split,
            "intended_emotion_id": emotion_id,
            "confidence_tier": tier,
            "training_lane": "motion_only_pending_human_approval",
            "intended_share": p7["p7"][intended_index],
            "selection_reasons": sorted(selected_entry["reasons"]),
            "source_faithful_partial_18d_csv": source[
                "source_faithful_partial_18d_csv"
            ],
            "source_faithful_partial_18d_csv_sha256": source[
                "source_faithful_partial_18d_csv_sha256"
            ],
            "observation_confidence_npy": source[
                "observation_confidence_npy"
            ],
            "observation_confidence_npy_sha256": source[
                "observation_confidence_npy_sha256"
            ],
            "quality_json": source["quality_json"],
            "quality_json_sha256": source["quality_json_sha256"],
            "quality_gate": source["quality_gate"],
            "observation_confidence_statistics": source[
                "observation_confidence_statistics"
            ],
            "kinematic_quality_summary": source[
                "kinematic_quality_summary"
            ],
            "source_faithful_upstream_smoothing_preserved": True,
            "upstream_source_faithful_smoothing_window": source[
                "source_faithful_processing"
            ]["upstream_source_faithful_smoothing_window"],
            "endpoints_preserved": True,
            "direct_playback_fps": SOURCE_FPS,
            "direct_playback_frames": SOURCE_FRAMES,
            "additional_review_smoothing": False,
            "additional_review_smoothing_applied": False,
            "retiming_applied_for_review": False,
            "retimed": False,
            "review_status": "pending",
            "human_review_approved": False,
            "training_launch_allowed": False,
            _NEGATIVE_PROOF_KEY: False,
        }
        review_row["record_sha256"] = _record_hash(review_row)
        review_rows.append(review_row)
    review_dir = output_dir / "human_review_bundle"
    review_manifest = review_dir / "manifest.jsonl"
    _atomic_jsonl(review_manifest, review_rows)
    split_counts = Counter(
        str(row["fixed_split_assignment"]) for row in review_rows
    )
    emotion_counts = Counter(
        str(row["intended_emotion_id"]) for row in review_rows
    )
    confidence_counts = Counter(
        str(row["confidence_tier"]) for row in review_rows
    )
    return {
        "artifact_kind": "hanyang_v8_human_review_selection_manifest_v1",
        "manifest": str(review_manifest),
        "manifest_sha256": sha256_file(review_manifest),
        "sample_count": len(review_rows),
        "selection_policy": (
            "14_to_21_source_faithful_samples_covering_all_nonempty_split_"
            "intended_emotion_cells_confidence_tiers_and_quality_extremes_v1"
        ),
        "counts_by_split": {
            split: split_counts[split] for split in REQUIRED_SPLITS
        },
        "counts_by_intended_emotion": {
            emotion: emotion_counts[emotion] for emotion in HANYANG_P7_ORDER
        },
        "counts_by_confidence_tier": dict(sorted(confidence_counts.items())),
        "source_faithful_direct_playback_only": True,
        "source_faithful_upstream_smoothing_preserved": True,
        "upstream_source_faithful_smoothing_windows": sorted(
            {
                int(row["upstream_source_faithful_smoothing_window"])
                for row in review_rows
            }
        ),
        "endpoints_preserved": True,
        "additional_review_smoothing": False,
        "additional_review_smoothing_applied": False,
        "retiming_applied": False,
        "selection_manifest_only": True,
        "contains_rendered_video": False,
        "human_review_evidence_complete": False,
        "review_status": "HUMAN_REVIEW_BLOCKED",
    }


def build_experimental_pool(
    passed_manifest: str | Path,
    output_dir: str | Path,
    *,
    expected_count: int = EXPECTED_PASSED_COUNT,
    expected_upstream_sha256: str = FINAL_PASSED_MANIFEST_SHA256,
) -> dict[str, Any]:
    """Validate all upstream artifacts and atomically build the v8 pool."""

    passed_manifest = Path(passed_manifest).resolve()
    output_dir = Path(output_dir).resolve()
    _reject_forbidden_dataset(
        {
            "source_manifest": str(passed_manifest),
            "output_path": str(output_dir),
        },
        context="build_arguments",
    )
    if not passed_manifest.is_file():
        raise FileNotFoundError(passed_manifest)
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    if (
        len(expected_upstream_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_upstream_sha256)
    ):
        raise ValueError("expected_upstream_sha256 must be a lowercase SHA-256")
    upstream_sha256 = sha256_file(passed_manifest)
    if upstream_sha256 != expected_upstream_sha256:
        raise ValueError(
            "passed manifest hash changed: "
            f"{upstream_sha256} != {expected_upstream_sha256}"
        )
    upstream_rows = _read_jsonl(passed_manifest)
    if len(upstream_rows) != expected_count:
        raise ValueError(
            f"passed manifest must contain {expected_count} rows, "
            f"got {len(upstream_rows)}"
        )
    clip_ids = [str(row.get("clip_id") or "") for row in upstream_rows]
    if len(set(clip_ids)) != len(clip_ids):
        raise ValueError("passed manifest contains duplicate clip IDs")
    retarget_root = passed_manifest.parent
    rows = [
        _validate_and_build_local_row(
            upstream,
            retarget_root=retarget_root,
            upstream_manifest_sha256=upstream_sha256,
        )
        for upstream in upstream_rows
    ]
    rows.sort(
        key=lambda row: (
            int(row["participant_id"]),
            tuple(int(value) for value in row["source_stem"].split("_")[1:]),
        )
    )
    coverage = _coverage_report(rows)
    global_emotion_gate = bool(coverage["passed"])
    for row in rows:
        local = bool(row["local_emotion_condition_candidate"])
        row["emotion_condition_eligible"] = bool(local and global_emotion_gate)
        row["emotion_condition_mask"] = row["emotion_condition_eligible"]
        row["admission_tier"] = (
            "strict_emotion_condition"
            if row["emotion_condition_eligible"]
            else "unconditional_motion_only"
        )
        row["formal_training_eligible"] = False
        row["record_sha256"] = _record_hash(row)

    registry_contract = build_data_source_registry_contract(
        [HANYANG_EMOTIONAL_BODY_SOURCE_ID],
        role=EMOTION_CRITIC_ROLE,
    )
    validate_data_source_registry_contract(
        registry_contract,
        expected_role=EMOTION_CRITIC_ROLE,
        expected_dataset_sources=[HANYANG_EMOTIONAL_BODY_SOURCE_ID],
    )
    manifest_path = output_dir / "manifest.jsonl"
    receipt_path = output_dir / "admission_receipt.json"
    _atomic_jsonl(manifest_path, rows)
    manifest_sha256 = sha256_file(manifest_path)
    human_review_bundle = _build_human_review_bundle(
        rows,
        output_dir=output_dir,
    )

    split_counts = Counter(str(row["fixed_split_assignment"]) for row in rows)
    emotion_counts = Counter(str(row["emotion_id"]) for row in rows)
    split_emotion_counts = Counter(
        (str(row["fixed_split_assignment"]), str(row["emotion_id"]))
        for row in rows
    )
    receipt: dict[str, Any] = {
        "artifact_kind": POOL_RECEIPT_KIND,
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "dataset_id": HANYANG_EMOTIONAL_BODY_SOURCE_ID,
        "dataset_revision": DATASET_REVISION,
        "pool_manifest": str(manifest_path),
        "pool_manifest_sha256": manifest_sha256,
        "pool_row_count": len(rows),
        "upstream_passed_manifest": str(passed_manifest),
        "upstream_passed_manifest_sha256": upstream_sha256,
        "upstream_passed_row_count": len(upstream_rows),
        "complete_upstream_coverage": len(rows) == len(upstream_rows),
        "counts_by_split": {
            split: split_counts[split] for split in REQUIRED_SPLITS
        },
        "counts_by_emotion": {
            emotion: emotion_counts[emotion] for emotion in HANYANG_P7_ORDER
        },
        "counts_by_split_and_emotion": {
            split: {
                emotion: split_emotion_counts[(split, emotion)]
                for emotion in HANYANG_P7_ORDER
            }
            for split in REQUIRED_SPLITS
        },
        "unconditional_motion_eligible_count": sum(
            row["unconditional_motion_eligible"] is True for row in rows
        ),
        "local_emotion_condition_candidate_count": sum(
            row["local_emotion_condition_candidate"] is True for row in rows
        ),
        "emotion_condition_eligible_count": sum(
            row["emotion_condition_eligible"] is True for row in rows
        ),
        "strict_global_emotion_coverage": coverage,
        "training_artifact_policy": {
            "source_faithful_partial_18d_only": True,
            "observation_confidence_required": True,
            "deployment_preview_ingest_allowed": False,
            "raw_ik_ingest_allowed": False,
            "fixed_frames": SOURCE_FRAMES,
            "fixed_fps": SOURCE_FPS,
            "permanently_unobserved_dof_indices": list(
                PERMANENTLY_UNOBSERVED_DOF_INDICES
            ),
        },
        "data_source_registry_contract": registry_contract,
        "registry_snapshot_role": EMOTION_CRITIC_ROLE,
        "registry_allowed_roles_unchanged": registry_contract["sources"][0][
            "allowed_roles"
        ],
        "experimental": True,
        "formal": False,
        "formal_training_admission": False,
        "formal_rejection_reason": FORMAL_REJECTION_REASON,
        "human_review_required": True,
        "human_review_approved": False,
        "human_review_status": "HUMAN_REVIEW_BLOCKED",
        "human_review_bundle": human_review_bundle,
        "training_launch_allowed": False,
        "generator_foundation_eligible": False,
        "current_beat2_training_mutated": False,
        _NEGATIVE_PROOF_KEY: False,
    }
    assert_no_forbidden_data_lineage(receipt, context="experimental_pool_receipt")
    if receipt[_NEGATIVE_PROOF_KEY] is not False:
        raise ValueError("experimental pool receipt lost the isolation proof")
    receipt["record_sha256"] = _record_hash(receipt)
    _atomic_json(receipt_path, receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--passed-manifest",
        type=Path,
        default=DEFAULT_PASSED_MANIFEST,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--expected-count",
        type=int,
        default=EXPECTED_PASSED_COUNT,
    )
    parser.add_argument(
        "--expected-upstream-sha256",
        default=FINAL_PASSED_MANIFEST_SHA256,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = build_experimental_pool(
        args.passed_manifest,
        args.output_dir,
        expected_count=args.expected_count,
        expected_upstream_sha256=args.expected_upstream_sha256,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
