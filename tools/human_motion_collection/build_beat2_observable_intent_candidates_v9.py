#!/usr/bin/env python3
"""Route BEAT2 18D motion into fail-closed v9 intent review candidates.

The router may prioritize clips for review, but it never emits semantic
supervision.  Source transcripts, filenames, legacy prompts, and emotion names
are metadata only and cannot become observable-intent labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.human_motion_collection.label_ula_v2_18d_motion import (  # noqa: E402
    load_trajectory,
)
from upper_body_skeleton.robot_observable_intents import (  # noqa: E402
    DEFAULT_ONTOLOGY_PATH,
    load_observable_intent_ontology,
    observable_intent_ids,
    ontology_sha256,
)
from upper_body_skeleton.robot_observable_motion_realizations import (  # noqa: E402
    DEFAULT_REALIZATION_ONTOLOGY_PATH,
    build_conversational_realization_annotation,
    realization_ontology_sha256,
    validate_conversational_realization_annotation,
)


SCHEMA_VERSION = "1.0.0"
ARTIFACT_KIND = "beat2_observable_intent_style_candidate_v9"
ROUTER_VERSION = "beat2_18d_trajectory_candidate_router_v1.1"
STYLE_VERSION = "beat2_18d_observable_style_v1"
SOURCE_TEXT_ROLE = "routing_context_metadata_only_never_semantic_supervision"
DEFAULT_SOURCE = Path(
    "/home/gez/nas/cloud/gez/human_motion/processed/"
    "beat2_semantic_event_training_pool_18d_v8/expansion/release/"
    "adjudication_min30f/train_ready.jsonl"
)
DEFAULT_STYLE_DRAFTS = (
    PROJECT_ROOT
    / "deliverables/expressive_human_motion_v2/robot_observable_intents_v1/"
    "beat2_full12148_style_v9/draft_prompts.jsonl"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "deliverables/expressive_human_motion_v2/robot_observable_intents_v1/"
    "beat2_full12148_intent_candidates_v9"
)


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_sha256(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    atomic_text(path, "".join(stable_json(record) + "\n" for record in records))


def _clip_control(value: float, low: float, high: float) -> float:
    if high <= low:
        raise ValueError("style control bounds must be increasing")
    return round(float(np.clip(2.0 * (value - low) / (high - low) - 1.0, -1.0, 1.0)), 6)


def _energy_label(mean_speed: float) -> str:
    if mean_speed < 0.55:
        return "restrained"
    if mean_speed < 0.85:
        return "moderate"
    if mean_speed < 1.15:
        return "energetic"
    return "emphatic"


def build_style_descriptor(features: Mapping[str, Any]) -> dict[str, Any]:
    groups = features["groups"]
    arm = features["arm"]
    overall = features["overall_motion"]
    arm_mean_speed = max(
        float(groups["left_arm"]["mean_speed_rad_s"]),
        float(groups["right_arm"]["mean_speed_rad_s"]),
    )
    arm_excursion = max(
        float(groups["left_arm"]["excursion_deg"]),
        float(groups["right_arm"]["excursion_deg"]),
    )
    change_rate = float(overall["normalized_change_rate_hz"])
    head_level = str(features["head_motion"])
    torso_level = str(features["torso_motion"])
    head_engagement = {
        "minimal": "quiet",
        "subtle": "engaged",
        "clear": "expressive",
    }[head_level]
    torso_engagement = {
        "minimal": "quiet",
        "subtle": "engaged",
        "clear": "expressive",
    }[torso_level]
    style_controls = {
        "amplitude": _clip_control(arm_excursion, 15.0, 70.0),
        "tempo": _clip_control(change_rate, 0.34, 0.82),
        "energy": _clip_control(arm_mean_speed, 0.34, 1.26),
    }
    return {
        "contract": STYLE_VERSION,
        "arm_amplitude": str(arm["amplitude"]),
        "laterality": str(arm["laterality"]),
        "energy_dominance": str(arm["energy_dominance"]),
        "bilateral_coordination": bool(arm["bilateral_temporally_coordinated"]),
        "continuity": str(arm["continuity"]),
        "regularly_repeated": bool(arm["regularly_repeated"]),
        "estimated_period_sec": arm["estimated_period_sec"],
        "pace": str(overall["pace"]),
        "energy": _energy_label(arm_mean_speed),
        "head_engagement": head_engagement,
        "head_dominant_axis": str(features["head"]["axis_motion"]["dominant_axis"]),
        "head_repeated_pattern": str(features["head"]["repeated_pattern"]["pattern"]),
        "torso_engagement": torso_engagement,
        "torso_dominant_axis": str(features["torso"]["axis_motion"]["dominant_axis"]),
        "style_controls": style_controls,
        "continuous_evidence": {
            "arm_excursion_deg": round(arm_excursion, 6),
            "arm_mean_speed_rad_s": round(arm_mean_speed, 6),
            "normalized_change_rate_hz": round(change_rate, 6),
            "head_excursion_deg": round(float(groups["head"]["excursion_deg"]), 6),
            "torso_excursion_deg": round(float(groups["torso"]["excursion_deg"]), 6),
        },
        "supervision_role": "trajectory_derived_style_only",
    }


def _trajectory_shape(values: np.ndarray) -> dict[str, float]:
    def subspace(group: np.ndarray) -> dict[str, float]:
        baseline = np.median(group[:baseline_count], axis=0)
        displacement = np.sqrt(np.mean(np.square(group - baseline), axis=1))
        peak = max(float(np.percentile(displacement, 95.0)), 1e-9)
        final = float(np.median(displacement[-tail_count:]))
        return {
            "peak_rad": round(peak, 6),
            "final_to_peak_ratio": round(final / peak, 6),
            "peak_hold_fraction": round(float(np.mean(displacement >= 0.82 * peak)), 6),
        }

    baseline_count = max(2, int(math.ceil(len(values) * 0.1)))
    tail_count = baseline_count
    baseline = np.median(values[:baseline_count], axis=0)
    displacement = np.sqrt(np.mean(np.square(values - baseline), axis=1))
    peak = max(float(np.percentile(displacement, 95.0)), 1e-9)
    final = float(np.median(displacement[-tail_count:]))
    hold_at_peak = float(np.mean(displacement >= 0.82 * peak))
    velocity = np.diff(displacement)
    reversal_count = int(
        np.count_nonzero(np.sign(velocity[1:]) * np.sign(velocity[:-1]) < 0)
    )
    head_pitch = values[:, 16]
    torso_pitch = values[:, 1]
    pitch_correlation = (
        float(np.corrcoef(head_pitch, torso_pitch)[0, 1])
        if np.std(head_pitch) > 1e-8 and np.std(torso_pitch) > 1e-8
        else 0.0
    )
    head_yaw = values[:, 17]
    yaw_low, yaw_high = np.percentile(head_yaw, [10.0, 90.0])
    yaw_center = 0.5 * float(yaw_low + yaw_high)
    yaw_half_range = max(0.5 * float(yaw_high - yaw_low), 1e-9)
    yaw_extreme_hold = float(
        np.mean(np.abs(head_yaw - yaw_center) >= 0.72 * yaw_half_range)
    )
    yaw_speed = np.r_[np.abs(np.diff(head_yaw)) * 30.0, 0.0]
    yaw_extreme_pause = float(
        np.mean(
            (np.abs(head_yaw - yaw_center) >= 0.72 * yaw_half_range)
            & (yaw_speed <= 0.12)
        )
    )
    return {
        "joint_space_peak_rad": round(peak, 6),
        "final_to_peak_ratio": round(final / peak, 6),
        "peak_hold_fraction": round(hold_at_peak, 6),
        "radial_reversal_count": reversal_count,
        "left_arm": subspace(values[:, 3:9]),
        "right_arm": subspace(values[:, 9:15]),
        "head": subspace(values[:, 15:18]),
        "torso": subspace(values[:, 0:3]),
        "head_torso_pitch_correlation": round(pitch_correlation, 6),
        "head_yaw_extreme_hold_fraction": round(yaw_extreme_hold, 6),
        "head_yaw_extreme_pause_fraction": round(yaw_extreme_pause, 6),
    }


def _shape_value(
    shape: Mapping[str, Any], group: str, field: str, fallback_field: str
) -> float:
    grouped = shape.get(group)
    if isinstance(grouped, Mapping) and grouped.get(field) is not None:
        return float(grouped[field])
    return float(shape[fallback_field])


def _route(
    intent_id: str,
    score: float,
    family: str,
    reasons: list[str],
    *,
    context_required: bool = False,
) -> dict[str, Any]:
    return {
        "candidate_intent_id": intent_id,
        "candidate_score": round(float(score), 6),
        "review_family": family,
        "routing_evidence": reasons,
        "context_evidence_required": context_required,
        "candidate_only": True,
        "grants_training_admission": False,
    }


def route_candidate_intents(
    source: Mapping[str, Any],
    features: Mapping[str, Any],
    shape: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Return review routes, never labels or semantic supervision."""

    routes: list[dict[str, Any]] = []
    arm = features["arm"]
    head = features["head"]
    torso = features["torso"]
    groups = features["groups"]
    pace = str(features["overall_motion"]["pace"])
    amp = str(arm["amplitude"])
    laterality = str(arm["laterality"])
    repeated = bool(arm["regularly_repeated"])
    period = arm["estimated_period_sec"]
    correlation = float(arm["bilateral_speed_correlation"])
    balance = float(arm["bilateral_energy_balance"])
    head_axes = head["axis_motion"]["per_axis"]
    torso_axes = torso["axis_motion"]["per_axis"]
    head_pitch = float(head_axes["pitch"]["excursion_deg"])
    head_yaw = float(head_axes["yaw"]["excursion_deg"])
    pitch_sweeps = int(head_axes["pitch"]["full_band_sweep_count"])
    yaw_sweeps = int(head_axes["yaw"]["full_band_sweep_count"])
    torso_pitch = float(torso_axes["pitch"]["excursion_deg"])
    torso_yaw = float(torso_axes["yaw"]["excursion_deg"])
    returned = float(shape["final_to_peak_ratio"]) <= 0.42
    sustained = float(shape["final_to_peak_ratio"]) >= 0.62
    left_hold = _shape_value(shape, "left_arm", "peak_hold_fraction", "peak_hold_fraction")
    right_hold = _shape_value(shape, "right_arm", "peak_hold_fraction", "peak_hold_fraction")
    left_return = _shape_value(shape, "left_arm", "final_to_peak_ratio", "final_to_peak_ratio")
    right_return = _shape_value(shape, "right_arm", "final_to_peak_ratio", "final_to_peak_ratio")
    head_return = _shape_value(shape, "head", "final_to_peak_ratio", "final_to_peak_ratio")
    torso_return = _shape_value(shape, "torso", "final_to_peak_ratio", "final_to_peak_ratio")
    active_arm_hold = left_hold if laterality == "left" else right_hold
    yaw_extreme_hold = float(shape.get("head_yaw_extreme_pause_fraction", 0.0))
    pitch_correlation = float(shape.get("head_torso_pitch_correlation", 0.0))
    duration = float(features["sample_span_sec"])

    if pitch_sweeps >= 4 and head_pitch >= 12.0:
        tier_a = bool(
            head_pitch >= 15.0
            and head["axis_motion"]["dominant_axis"] == "pitch"
            and torso_pitch < 8.0
            and 1.2 <= duration <= 5.0
        )
        routes.append(
            _route(
                "agree_nod",
                min(0.96, (0.86 if tier_a else 0.70) + 0.015 * pitch_sweeps),
                "head_response",
                [
                    f"tier_a={str(tier_a).lower()}",
                    f"head_pitch_sweeps={pitch_sweeps}",
                    f"head_pitch_excursion_deg={head_pitch:.2f}",
                ],
            )
        )
    if yaw_sweeps >= 4 and head_yaw >= 12.0:
        head_to_torso_yaw = head_yaw / max(torso_yaw, 1e-6)
        tier_a = bool(
            head_yaw >= 18.0
            and head["axis_motion"]["dominant_axis"] == "yaw"
            and torso_yaw < 12.0
            and head_to_torso_yaw >= 1.3
            and yaw_extreme_hold < 0.35
        )
        routes.append(
            _route(
                "disagree_head_shake",
                min(0.96, (0.86 if tier_a else 0.70) + 0.015 * yaw_sweeps),
                "head_response",
                [
                    f"tier_a={str(tier_a).lower()}",
                    f"head_yaw_sweeps={yaw_sweeps}",
                    f"head_yaw_excursion_deg={head_yaw:.2f}",
                    f"head_to_torso_yaw_ratio={head_to_torso_yaw:.2f}",
                ],
            )
        )
    if (
        head_yaw >= 22.0
        and 1 <= yaw_sweeps <= 3
        and duration >= 2.2
        and yaw_extreme_hold >= 0.40
        and head["repeated_pattern"]["pattern"] == "none"
    ):
        routes.append(
            _route(
                "search_scan",
                min(0.90, 0.68 + 0.006 * head_yaw),
                "head_scan_or_no_hard_negative",
                [
                    f"head_yaw_sweeps={yaw_sweeps}",
                    f"head_yaw_excursion_deg={head_yaw:.2f}",
                    f"yaw_extreme_hold_fraction={yaw_extreme_hold:.2f}",
                ],
            )
        )

    if (
        torso_pitch >= 10.0
        and head_pitch >= 8.0
        and pitch_correlation >= 0.55
        and head_return <= 0.45
        and torso_return <= 0.45
        and torso["axis_motion"]["dominant_axis"] == "pitch"
    ):
        routes.append(
            _route(
                "bow",
                min(0.90, 0.68 + 0.006 * torso_pitch),
                "forward_torso_arc",
                [f"torso_pitch_excursion_deg={torso_pitch:.2f}", "trajectory_returns_toward_start"],
            )
        )
    if torso_pitch >= 15.0 and head_pitch >= 10.0 and sustained and pace != "quick":
        routes.append(
            _route(
                "disappointment_slump",
                min(0.82, 0.64 + 0.004 * torso_pitch),
                "forward_torso_arc",
                [f"torso_pitch_excursion_deg={torso_pitch:.2f}", "final_pose_remains_displaced"],
            )
        )

    unilateral = laterality in {"left", "right"}
    if unilateral and amp in {"moderate", "large"} and repeated and period is not None and float(period) <= 1.4:
        routes.extend(
            [
                _route(
                    "wave_to_person",
                    0.75,
                    "unilateral_repeated_social_signal",
                    [f"laterality={laterality}", f"period_sec={float(period):.2f}", f"amplitude={amp}"],
                ),
                _route(
                    "beckon_come_here",
                    0.67,
                    "unilateral_repeated_social_signal",
                    ["wave-versus-beckon direction requires robot-video review"],
                ),
            ]
        )
    inactive_excursion_ratio = (
        float(groups["right_arm"]["excursion_deg"])
        / max(float(groups["left_arm"]["excursion_deg"]), 1e-6)
        if laterality == "left"
        else float(groups["left_arm"]["excursion_deg"])
        / max(float(groups["right_arm"]["excursion_deg"]), 1e-6)
    )
    raised_hold_tier_a = bool(
        unilateral
        and amp == "large"
        and not repeated
        and active_arm_hold >= 0.40
        and balance <= 0.45
        and inactive_excursion_ratio <= 0.55
        and float(groups["head"]["excursion_deg"]) < 15.0
        and float(groups["torso"]["excursion_deg"]) < 15.0
    )
    if raised_hold_tier_a:
        routes.append(
            _route(
                "raise_hand_get_attention",
                0.84,
                "unilateral_raised_hold",
                [
                    "tier_a=true",
                    f"laterality={laterality}",
                    f"active_arm_peak_hold_fraction={active_arm_hold:.2f}",
                    f"inactive_arm_excursion_ratio={inactive_excursion_ratio:.2f}",
                ],
            )
        )

    bilateral = laterality == "both" and balance >= 0.60
    coordinated = bilateral and correlation >= 0.55
    left_excursion = float(groups["left_arm"]["excursion_deg"])
    right_excursion = float(groups["right_arm"]["excursion_deg"])
    if (
        coordinated
        and repeated
        and period is not None
        and float(period) <= 0.9
        and left_excursion >= 28.0
        and right_excursion >= 28.0
        and balance >= 0.75
        and correlation >= 0.70
    ):
        routes.append(
            _route(
                "applaud",
                min(0.86, 0.69 + 0.10 * max(correlation, 0.0)),
                "bilateral_repeated_signal",
                [f"bilateral_speed_correlation={correlation:.2f}", f"energy_balance={balance:.2f}"],
            )
        )
    if (
        bilateral
        and left_excursion >= 50.0
        and right_excursion >= 50.0
        and balance >= 0.80
        and correlation >= 0.65
        and pace == "quick"
        and left_return <= 0.35
        and right_return <= 0.35
        and not repeated
    ):
        routes.append(
            _route(
                "celebrate",
                0.70,
                "bilateral_expansive_arc",
                ["large_bilateral_quick_arc", "trajectory_returns_toward_start"],
            )
        )
    if coordinated and amp in {"small", "moderate"} and duration <= 3.5 and returned and not repeated:
        routes.append(
            _route(
                "shrug_uncertain",
                0.67,
                "compact_bilateral_arc",
                [f"amplitude={amp}", f"duration_sec={duration:.2f}", "compact_returning_arc"],
            )
        )

    category = (source.get("semantic_event") or {}).get("category")
    if category == "deictic" and raised_hold_tier_a:
        reasons = [
            "official_deictic_category_used_for_review_routing_only",
            f"laterality={laterality}",
            "direction_requires_robot-video_review",
        ]
        for intent_id in ("point_left", "point_right", "point_forward"):
            routes.append(_route(intent_id, 0.69, "deictic_direction_unknown", reasons))

    # Ordinary BEAT2 co-speech gestures belong to the independent motion-
    # realization layer.  Kinematics alone cannot upgrade them to the primary
    # explain_present intent.
    if torso_yaw >= 25.0 and head_yaw >= 20.0 and sustained:
        routes.append(
            _route(
                "withdraw_turn_away",
                0.66,
                "sustained_head_torso_turn",
                [f"torso_yaw_excursion_deg={torso_yaw:.2f}", "final_pose_remains_displaced"],
            )
        )

    known_ids: set[str] = set()
    deduplicated: list[dict[str, Any]] = []
    for route in sorted(routes, key=lambda item: (-item["candidate_score"], item["candidate_intent_id"])):
        intent_id = str(route["candidate_intent_id"])
        if intent_id not in known_ids:
            known_ids.add(intent_id)
            deduplicated.append(route)
    return deduplicated


def _selection_key(record: Mapping[str, Any], seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{record['task_id']}".encode("utf-8")).hexdigest()


def select_balanced_review(
    candidates: list[dict[str, Any]],
    *,
    per_family: int,
    max_total: int,
    max_per_speaker_family: int,
    seed: int,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        family = str(record["candidate_routes"][0]["review_family"])
        groups[family].append(record)
    selected: list[dict[str, Any]] = []
    for family in sorted(groups):
        ranked = sorted(
            groups[family],
            key=lambda item: (
                -float(item["candidate_routes"][0]["candidate_score"]),
                _selection_key(item, seed),
            ),
        )
        speaker_counts: Counter[str] = Counter()
        family_selected = 0
        for item in ranked:
            speaker = str(item.get("speaker_key") or "unknown")
            if speaker_counts[speaker] >= max_per_speaker_family:
                continue
            output = dict(item)
            output["review_selection"] = {
                "selected": True,
                "review_family": family,
                "selection_seed": seed,
                "rank_within_family": family_selected + 1,
                "source_group_balanced": True,
            }
            selected.append(output)
            speaker_counts[speaker] += 1
            family_selected += 1
            if family_selected >= per_family:
                break
    return sorted(
        selected,
        key=lambda item: (
            str(item["review_selection"]["review_family"]),
            int(item["review_selection"]["rank_within_family"]),
        ),
    )[:max_total]


def build_candidates(
    *,
    source_manifest: Path,
    style_drafts: Path,
    ontology_path: Path,
    realization_ontology_path: Path,
    output_dir: Path,
    per_family: int,
    max_total: int,
    max_per_speaker_family: int,
    seed: int,
) -> dict[str, Any]:
    sources = read_jsonl(source_manifest)
    drafts = read_jsonl(style_drafts)
    ontology = load_observable_intent_ontology(ontology_path)
    ontology_digest = ontology_sha256(ontology_path)
    realization_digest = realization_ontology_sha256(realization_ontology_path)
    ontology_ids = set(observable_intent_ids(ontology))
    draft_by_id = {str(item["task_id"]): item for item in drafts}
    if len(draft_by_id) != len(drafts):
        raise ValueError("style drafts contain duplicate task_id values")
    if len(sources) != len(drafts):
        raise ValueError("source/style record counts do not match")

    style_catalog: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    realization_train_ready: list[dict[str, Any]] = []
    route_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    style_counts: dict[str, Counter[str]] = {
        name: Counter()
        for name in ("arm_amplitude", "laterality", "pace", "energy", "head_engagement", "torso_engagement")
    }
    for source in sources:
        task_id = str(source.get("task_id") or "")
        draft = draft_by_id.get(task_id)
        if draft is None:
            raise ValueError(f"{task_id}: missing trajectory style draft")
        if source.get("status") != "passed" or source.get("quality_gate", {}).get("passed") is not True:
            raise ValueError(f"{task_id}: source is not physical-QC passed")
        trajectory = Path(str(source.get("safe_csv") or "")).resolve()
        expected_sha = source.get("safe_csv_sha256")
        if not trajectory.is_file() or file_sha256(trajectory) != expected_sha:
            raise ValueError(f"{task_id}: trajectory evidence mismatch")
        if draft.get("trajectory_sha256") != expected_sha:
            raise ValueError(f"{task_id}: style draft trajectory binding mismatch")
        features = draft.get("observable_features")
        if not isinstance(features, dict):
            raise ValueError(f"{task_id}: style draft lacks observable_features")
        values = load_trajectory(trajectory)
        shape = _trajectory_shape(values)
        style = build_style_descriptor(features)
        realization = build_conversational_realization_annotation(
            source,
            style,
            ontology_path=realization_ontology_path,
        )
        validate_conversational_realization_annotation(
            realization,
            ontology_path=realization_ontology_path,
        )
        routes = route_candidate_intents(source, features, shape)
        unknown = {str(item["candidate_intent_id"]) for item in routes} - ontology_ids
        if unknown:
            raise ValueError(f"{task_id}: router emitted unknown ontology IDs {sorted(unknown)}")
        style_record = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": ARTIFACT_KIND,
            "router_version": ROUTER_VERSION,
            "task_id": task_id,
            "source_clip_id": source.get("source_clip_id"),
            "source_group_key": source.get("source_group_key"),
            "speaker_key": source.get("speaker_key"),
            "official_split": source.get("official_split"),
            "fixed_split_assignment": source.get("fixed_split_assignment"),
            "fps": float(source["fps"]),
            "frames": int(source["frames"]),
            "native_variable_length": True,
            "trajectory_path": str(trajectory),
            "trajectory_sha256": expected_sha,
            "physical_qc_passed": True,
            "quality_json": source.get("quality_json"),
            "quality_json_sha256": source.get("quality_json_sha256"),
            "quality_gate": source.get("quality_gate"),
            "robot_contract": "ula_v2_18d_head_v1",
            "canonical_action": "candidate_observable_interaction_pending_blind_review",
            "canonical_prompt": realization["motion_realization_prompt"],
            "observable_features": features,
            "motion_style": style,
            "motion_realization": realization,
            "trajectory_shape": shape,
            "intent_ontology_id": ontology["ontology_id"],
            "intent_ontology_sha256": ontology_digest,
            "candidate_routes": routes,
            "candidate_intent_ids": [item["candidate_intent_id"] for item in routes],
            "observable_intent_id": None,
            "intent_review_status": "candidate_pending_anonymous_robot_video_review" if routes else "not_routed",
            "intent_supervision_mask": False,
            "intent_conditioning_mask": False,
            "emotion_id_metadata_only": source.get("emotion_id"),
            "emotion_supervision_mask": False,
            "source_text_role": SOURCE_TEXT_ROLE,
            "source_text_used_for_routing": False,
            "source_text_used_for_admission": False,
            "legacy_prompt_used_for_routing": False,
            "automatic_intent_labels_emitted": 0,
            "accepted_for_training": False,
            "accepted_for_motion_realization_training": True,
            "accepted_for_intent_training": False,
            "manual_video_review_required": bool(routes),
            "manual_review_required": bool(routes),
            "source_record_sha256": record_sha256(source),
        }
        style_catalog.append(style_record)
        realization_train_ready.append(style_record)
        for field in style_counts:
            style_counts[field][str(style[field])] += 1
        if routes:
            candidates.append(style_record)
            family_counts[str(routes[0]["review_family"])] += 1
            for route in routes:
                route_counts[str(route["candidate_intent_id"])] += 1

    review_selection = select_balanced_review(
        candidates,
        per_family=per_family,
        max_total=max_total,
        max_per_speaker_family=max_per_speaker_family,
        seed=seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "style_catalog.jsonl", style_catalog)
    write_jsonl(output_dir / "intent_candidate_pool.jsonl", candidates)
    write_jsonl(output_dir / "review_selection.jsonl", review_selection)
    write_jsonl(
        output_dir / "motion_realization_train_ready.jsonl",
        realization_train_ready,
    )
    write_jsonl(output_dir / "train_ready.jsonl", [])
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "beat2_observable_intent_style_candidate_release_v9",
        "router_version": ROUTER_VERSION,
        "style_version": STYLE_VERSION,
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": file_sha256(source_manifest),
        "style_drafts": str(style_drafts.resolve()),
        "style_drafts_sha256": file_sha256(style_drafts),
        "ontology_path": str(ontology_path.resolve()),
        "ontology_id": ontology["ontology_id"],
        "ontology_sha256": ontology_digest,
        "motion_realization_ontology_path": str(realization_ontology_path.resolve()),
        "motion_realization_ontology_sha256": realization_digest,
        "source_records": len(sources),
        "style_records": len(style_catalog),
        "intent_candidate_records": len(candidates),
        "review_selection_records": len(review_selection),
        "automatic_intent_labels_emitted": 0,
        "motion_realization_train_ready_records": len(realization_train_ready),
        "train_ready_records": 0,
        "native_variable_length_preserved": True,
        "audio_used": False,
        "source_text_used_for_routing": False,
        "source_text_used_for_admission": False,
        "review_policy": {
            "anonymous_robot_video_required": True,
            "minimum_independent_reviewers": ontology["review_contract"]["minimum_independent_reviewers"],
            "exact_intent_consensus_required": True,
            "context_required_intents_need_separate_context_evidence": True,
        },
        "candidate_route_counts": dict(sorted(route_counts.items())),
        "primary_review_family_counts": dict(sorted(family_counts.items())),
        "style_counts": {field: dict(sorted(counts.items())) for field, counts in style_counts.items()},
        "selection_policy": {
            "per_family": per_family,
            "max_total": max_total,
            "max_per_speaker_family": max_per_speaker_family,
            "seed": seed,
        },
        "outputs": {
            "style_catalog": str((output_dir / "style_catalog.jsonl").resolve()),
            "intent_candidate_pool": str((output_dir / "intent_candidate_pool.jsonl").resolve()),
            "review_selection": str((output_dir / "review_selection.jsonl").resolve()),
            "motion_realization_train_ready": str(
                (output_dir / "motion_realization_train_ready.jsonl").resolve()
            ),
            "train_ready": str((output_dir / "train_ready.jsonl").resolve()),
        },
    }
    atomic_text(output_dir / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--style-drafts", type=Path, default=DEFAULT_STYLE_DRAFTS)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY_PATH)
    parser.add_argument(
        "--realization-ontology",
        type=Path,
        default=DEFAULT_REALIZATION_ONTOLOGY_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-family", type=int, default=24)
    parser.add_argument("--max-total", type=int, default=240)
    parser.add_argument("--max-per-speaker-family", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.per_family <= 0 or args.max_total <= 0 or args.max_per_speaker_family <= 0:
        raise ValueError("selection limits must be positive")
    summary = build_candidates(
        source_manifest=args.source_manifest,
        style_drafts=args.style_drafts,
        ontology_path=args.ontology,
        realization_ontology_path=args.realization_ontology,
        output_dir=args.output_dir,
        per_family=args.per_family,
        max_total=args.max_total,
        max_per_speaker_family=args.max_per_speaker_family,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
