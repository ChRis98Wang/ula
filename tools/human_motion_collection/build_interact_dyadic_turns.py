#!/usr/bin/env python3
"""Build paired InterAct inventory and natural-rest expression-turn pilots.

Turns are cut only at a basin where both partners' selected body/head joints
are at rest.  There is no target, minimum, maximum, or fixed turn duration. If
no internal joint-rest basin exists, the whole recording is preserved as one
candidate.  Scenario, relationship, and source emotion are provenance
metadata only; every supervision and admission mask starts false.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable

import numpy as np

try:
    from tools.gmr_v2.interact_bvh_adapter import (
        ENERGY_JOINTS,
        is_30hz,
        local_rotation_matrices,
        parse_bvh,
        read_bvh_header,
    )
except ModuleNotFoundError:  # pragma: no cover - direct invocation by path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.gmr_v2.interact_bvh_adapter import (
        ENERGY_JOINTS,
        is_30hz,
        local_rotation_matrices,
        parse_bvh,
        read_bvh_header,
    )


DEFAULT_ROOT = Path("/home/gez/nas/cloud/gez/human_motion/raw/InterAct")
DEFAULT_RECEIPT = Path(
    "/home/gez/nas/cloud/gez/human_motion/catalog/"
    "interact_motion_only_acquisition.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/gez/nas/cloud/gez/human_motion/catalog/interact_dyadic_turn_v1"
)
FILE_PATTERN = re.compile(
    r"bvhs/(?P<date>\d{8})_(?P<actor>\d{3})_(?P<scenario>\d{3})\.bvh\Z"
)
REST_ENERGY_THRESHOLD_RAD_S = 0.12
REST_EVIDENCE_CONSECUTIVE_FRAMES = 9
ENERGY_SMOOTHING_RADIUS_FRAMES = 2
SEMANTIC_MASKS = {
    "scenario_text": False,
    "relationship": False,
    "communicative_intent": False,
    "robot_observable_motion_form": False,
    "prompt_text": False,
}
NATURAL_DURATION_POLICY = (
    "semantic_affect_complete_at_predeclared_shared_rest_boundaries;"
    "no_fixed_target_minimum_or_maximum_duration"
)
COMPLETENESS_REVIEW_CRITERIA = (
    "complete_onset_apex_offset_and_complete_observable_action_semantics_and_affect_arc"
)
PILOT_COLLECTION_SCOPE = "pilot_subset_not_representative_of_dataset_pass_rate"
FULL_COLLECTION_SCOPE = "full_pool_all_locally_complete_paired_performances"
FORBIDDEN_DURATION_CONSTRAINT_KEYS = {
    "duration_sec",
    "fixed_duration_sec",
    "fixed_window_sec",
    "max_duration",
    "max_duration_sec",
    "min_duration",
    "min_duration_sec",
    "target_duration",
    "target_duration_sec",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--acquisition-receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--pilot-performance-count",
        type=int,
        default=4,
        help="Number of complete real paired recordings to segment for the pilot",
    )
    parser.add_argument(
        "--rest-energy-threshold-rad-s",
        type=float,
        default=REST_ENERGY_THRESHOLD_RAD_S,
    )
    parser.add_argument(
        "--rest-evidence-consecutive-frames",
        type=int,
        default=REST_EVIDENCE_CONSECUTIVE_FRAMES,
    )
    return parser.parse_args(argv)


def stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def record_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def duration_constraint_key_paths(value: Any, path: str = "$") -> list[str]:
    """Find elapsed-time fields that could silently become segmentation gates."""
    findings = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key).lower() in FORBIDDEN_DURATION_CONSTRAINT_KEYS:
                findings.append(child_path)
            findings.extend(duration_constraint_key_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(duration_constraint_key_paths(child, f"{path}[{index}]"))
    return findings


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> str:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return hashlib.sha256(payload).hexdigest()


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    payload = "".join(stable_json(row) + "\n" for row in rows).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return hashlib.sha256(payload).hexdigest()


def load_source_metadata(root: Path) -> dict[str, Any]:
    actors_connection = sqlite3.connect(root / "actors.db")
    scenarios_connection = sqlite3.connect(root / "scenarios.db")
    try:
        actors = {
            str(actor_id): {"actor_id": str(actor_id), "gender": str(gender)}
            for actor_id, gender in actors_connection.execute(
                "SELECT actor_id, gender FROM actors"
            )
        }
        sessions = {
            str(date): {"male_id": str(male_id), "female_id": str(female_id)}
            for date, male_id, female_id in actors_connection.execute(
                "SELECT date, male_id, female_id FROM sessions"
            )
        }
        relationships = {
            int(identifier): str(name)
            for identifier, name in scenarios_connection.execute(
                "SELECT id, name FROM relationships"
            )
        }
        emotions = {
            int(identifier): str(name)
            for identifier, name in scenarios_connection.execute(
                "SELECT id, name FROM emotions"
            )
        }
        scenarios = {}
        for identifier, relationship_id, emotion_id, character_setup, scenario in (
            scenarios_connection.execute(
                "SELECT id, relationship_id, primary_emotion_id, "
                "character_setup, scenario FROM scenarios"
            )
        ):
            scenarios[int(identifier)] = {
                "scenario_id": int(identifier),
                "relationship_id": int(relationship_id),
                "relationship_name": relationships[int(relationship_id)],
                "primary_emotion_id": int(emotion_id),
                "primary_emotion_name": emotions[int(emotion_id)],
                "character_setup": str(character_setup),
                "scenario": str(scenario),
            }
    finally:
        actors_connection.close()
        scenarios_connection.close()
    return {
        "actors": actors,
        "sessions": sessions,
        "relationships": relationships,
        "emotions": emotions,
        "scenarios": scenarios,
    }


def _source_rows(receipt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in receipt.get("files") or []:
        if FILE_PATTERN.fullmatch(str(row.get("path") or "")):
            rows[str(row["path"])] = row
    if not rows:
        raise ValueError("Acquisition receipt contains no InterAct BVH files")
    return rows


def build_paired_inventory(
    root: Path,
    acquisition_receipt: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_rows = _source_rows(acquisition_receipt)
    grouped: dict[tuple[str, int], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for path, source in source_rows.items():
        match = FILE_PATTERN.fullmatch(path)
        assert match is not None
        grouped[(match.group("date"), int(match.group("scenario")))].append(
            (match.group("actor"), source)
        )

    performances = []
    actor_clips = []
    receipt_sha256 = acquisition_receipt.get("receipt_sha256")
    for (date, scenario_id), members in sorted(grouped.items()):
        session = metadata["sessions"].get(date)
        scenario = metadata["scenarios"].get(scenario_id)
        expected_actor_ids = (
            sorted([session["male_id"], session["female_id"]]) if session else []
        )
        observed_actor_ids = sorted(actor_id for actor_id, _source in members)
        partner_records = []
        for actor_id, source in sorted(members):
            local_path = root / source["path"]
            header_error = None
            frame_count = None
            frame_time = None
            if source.get("local_status") == "verified" and local_path.is_file():
                try:
                    frame_count, frame_time = read_bvh_header(local_path)
                except (OSError, ValueError) as error:
                    header_error = str(error)
            role = None
            if session:
                if actor_id == session["male_id"]:
                    role = "male_actor_in_source_session"
                elif actor_id == session["female_id"]:
                    role = "female_actor_in_source_session"
            partner_records.append(
                {
                    "actor_id": actor_id,
                    "actor_gender_metadata": (
                        metadata["actors"].get(actor_id, {}).get("gender")
                    ),
                    "session_role_metadata": role,
                    "source_path": source["path"],
                    "source_local_path": str(local_path.resolve()),
                    "source_local_sha256": source.get("local_sha256"),
                    "source_remote_git_blob_oid_sha1": source.get(
                        "remote_git_blob_oid_sha1"
                    ),
                    "source_verification_status": source.get("local_status"),
                    "frame_count": frame_count,
                    "frame_time_sec": frame_time,
                    "fps": None if frame_time is None else 1.0 / frame_time,
                    "header_error": header_error,
                }
            )
        frame_counts = [p["frame_count"] for p in partner_records]
        frame_times = [p["frame_time_sec"] for p in partner_records]
        checks = {
            "scenario_metadata_found": scenario is not None,
            "session_metadata_found": session is not None,
            "exactly_two_source_partners": len(members) == 2,
            "source_actor_ids_match_session": observed_actor_ids == expected_actor_ids,
            "both_local_files_verified": all(
                p["source_verification_status"] == "verified" for p in partner_records
            ),
            "both_headers_readable": all(value is not None for value in frame_counts),
            "both_30hz": all(
                value is not None and is_30hz(float(value)) for value in frame_times
            ),
            "partner_frame_counts_equal": (
                len(frame_counts) == 2
                and all(value is not None for value in frame_counts)
                and frame_counts[0] == frame_counts[1]
            ),
        }
        pair_valid = all(checks.values())
        performance_id = f"interact__{date}__scenario_{scenario_id:03d}"
        performance = {
            "schema_version": "1.0.0",
            "artifact_kind": "interact_dyadic_performance_inventory",
            "performance_id": performance_id,
            "recording_date": date,
            "scenario_id": scenario_id,
            "partners": partner_records,
            "partner_pair_lineage_complete": len(partner_records) == 2,
            "source_metadata": None
            if scenario is None
            else {
                **scenario,
                "metadata_only": True,
                "semantic_validated": False,
                "emotion_observable_validated": False,
            },
            "pair_checks": checks,
            "pair_available_for_turn_candidate_extraction": pair_valid,
            "semantic_supervision_masks": dict(SEMANTIC_MASKS),
            "emotion_supervision_mask": False,
            "source_emotion_conditioning_mask": False,
            "relationship_conditioning_mask": False,
            "admission_mask": False,
            "accepted_for_training": False,
            "acquisition_receipt_sha256": receipt_sha256,
        }
        performance["inventory_record_sha256"] = record_sha256(performance)
        performances.append(performance)
        for partner in partner_records:
            other = [p for p in partner_records if p["actor_id"] != partner["actor_id"]]
            actor_record = {
                "schema_version": "1.0.0",
                "artifact_kind": "interact_actor_clip_inventory",
                "clip_id": f"{performance_id}__actor_{partner['actor_id']}",
                "performance_id": performance_id,
                "actor": partner,
                "partner_lineage": other[0] if len(other) == 1 else None,
                "partner_lineage_complete": len(other) == 1,
                "scenario_metadata_only": None if scenario is None else scenario,
                "semantic_supervision_masks": dict(SEMANTIC_MASKS),
                "emotion_supervision_mask": False,
                "admission_mask": False,
                "accepted_for_training": False,
            }
            actor_record["inventory_record_sha256"] = record_sha256(actor_record)
            actor_clips.append(actor_record)
    return performances, actor_clips


def angular_energy_rad_s(structure: Any) -> np.ndarray:
    rotations = local_rotation_matrices(structure, ENERGY_JOINTS)
    per_joint = []
    for joint in ENERGY_JOINTS:
        matrices = rotations[joint]
        delta = np.swapaxes(matrices[:-1], 1, 2) @ matrices[1:]
        cosine = np.clip((np.trace(delta, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
        per_joint.append(np.arccos(cosine) / structure.frame_time)
    energy = np.sqrt(np.mean(np.square(np.stack(per_joint, axis=1)), axis=1))
    energy = np.r_[energy[0], energy]
    radius = ENERGY_SMOOTHING_RADIUS_FRAMES
    if radius:
        kernel = np.ones(2 * radius + 1, dtype=np.float64) / (2 * radius + 1)
        energy = np.convolve(np.pad(energy, (radius, radius), mode="edge"), kernel, mode="valid")
    return energy


def dyadic_joint_energy_rad_s(left: Any, right: Any) -> np.ndarray:
    if left.frame_count != right.frame_count:
        raise ValueError("Partner frame counts differ; implicit truncation is forbidden")
    if abs(left.frame_time - right.frame_time) > 1e-9:
        raise ValueError("Partner frame times differ; implicit retiming is forbidden")
    # A shared rest requires each partner to be still, so max is safer than mean.
    return np.maximum(angular_energy_rad_s(left), angular_energy_rad_s(right))


def rest_basins(
    energy: np.ndarray,
    *,
    threshold_rad_s: float,
    evidence_consecutive_frames: int,
) -> list[dict[str, Any]]:
    if threshold_rad_s <= 0 or evidence_consecutive_frames <= 0:
        raise ValueError("Rest evidence settings must be positive")
    mask = np.asarray(energy) <= threshold_rad_s
    padded = np.r_[False, mask, False].astype(np.int8)
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    basins = []
    for start, end in zip(starts, ends, strict=True):
        if end - start < evidence_consecutive_frames:
            continue
        basin_energy = energy[start:end]
        minimum_value = float(np.min(basin_energy))
        minimum_candidates = np.flatnonzero(
            np.isclose(basin_energy, minimum_value, rtol=0.0, atol=1e-12)
        )
        midpoint = (end - start - 1) / 2.0
        minimum = int(
            start
            + minimum_candidates[
                int(np.argmin(np.abs(minimum_candidates.astype(float) - midpoint)))
            ]
        )
        basins.append(
            {
                "start_frame": int(start),
                "end_frame_exclusive": int(end),
                "cut_evidence_frame": minimum,
                "frame_count": int(end - start),
                "energy_min_rad_s": float(energy[minimum]),
                "energy_max_rad_s": float(np.max(energy[start:end])),
            }
        )
    return basins


def natural_rest_intervals(
    energy: np.ndarray,
    *,
    threshold_rad_s: float,
    evidence_consecutive_frames: int,
) -> tuple[list[tuple[int, int]], list[dict[str, Any]]]:
    basins = rest_basins(
        energy,
        threshold_rad_s=threshold_rad_s,
        evidence_consecutive_frames=evidence_consecutive_frames,
    )
    frame_count = len(energy)
    internal_cuts = sorted(
        {
            int(basin["cut_evidence_frame"])
            for basin in basins
            if basin["start_frame"] > 0 and basin["end_frame_exclusive"] < frame_count
        }
    )
    cuts = [0, *[cut for cut in internal_cuts if 0 < cut < frame_count], frame_count]
    return list(zip(cuts[:-1], cuts[1:], strict=True)), basins


def _boundary_basis(
    frame: int, frame_count: int, basins: list[dict[str, Any]], side: str
) -> str:
    if side == "left" and frame == 0:
        return (
            "recording_start_with_joint_rest_evidence"
            if any(b["start_frame"] == 0 for b in basins)
            else "recording_start_without_joint_rest_evidence"
        )
    if side == "right" and frame == frame_count - 1:
        return (
            "recording_end_with_joint_rest_evidence"
            if any(b["end_frame_exclusive"] == frame_count for b in basins)
            else "recording_end_without_joint_rest_evidence"
        )
    if any(b["start_frame"] <= frame < b["end_frame_exclusive"] for b in basins):
        return "shared_natural_joint_rest_basin"
    return "boundary_without_joint_rest_evidence"


def progressive_natural_context_levels(
    intervals: list[tuple[int, int]], core_index: int
) -> list[dict[str, Any]]:
    """Return nested rest-boundary expansions, eventually covering the recording."""
    if not 0 <= core_index < len(intervals):
        raise ValueError("Core interval index is out of range")
    left = core_index
    right = core_index
    levels = []

    def append_level(reason: str) -> None:
        levels.append(
            {
                "level": len(levels),
                "start_frame": intervals[left][0],
                "end_frame_exclusive": intervals[right][1],
                "included_natural_interval_indices": list(range(left, right + 1)),
                "expansion_reason": reason,
            }
        )

    append_level("core_shared_rest_to_shared_rest_candidate")
    while left > 0 or right < len(intervals) - 1:
        if left > 0:
            left -= 1
            append_level("prepend_adjacent_natural_interval_if_blind_arc_is_incomplete")
        if right < len(intervals) - 1:
            right += 1
            append_level("append_adjacent_natural_interval_if_blind_arc_is_incomplete")
    return levels


def segment_performance(
    performance: dict[str, Any],
    *,
    threshold_rad_s: float,
    evidence_consecutive_frames: int,
) -> list[dict[str, Any]]:
    if not performance["pair_available_for_turn_candidate_extraction"]:
        raise ValueError("Cannot segment an unavailable or invalid partner pair")
    partners = performance["partners"]
    left = parse_bvh(Path(partners[0]["source_local_path"]), load_motion=True)
    right = parse_bvh(Path(partners[1]["source_local_path"]), load_motion=True)
    energy = dyadic_joint_energy_rad_s(left, right)
    intervals, basins = natural_rest_intervals(
        energy,
        threshold_rad_s=threshold_rad_s,
        evidence_consecutive_frames=evidence_consecutive_frames,
    )
    candidates = []
    no_internal_rest = len(intervals) == 1
    scenario_metadata = performance["source_metadata"]
    for turn_index, (start, end) in enumerate(intervals):
        left_basis = _boundary_basis(start, len(energy), basins, "left")
        right_basis = _boundary_basis(end - 1, len(energy), basins, "right")
        apex = int(start + np.argmax(energy[start:end]))
        frame_count = end - start
        turn_id = f"{performance['performance_id']}__turn_{turn_index:04d}_f{start:06d}-{end:06d}"
        candidate = {
            "schema_version": "1.1.0",
            "artifact_kind": "interact_dyadic_natural_rest_expression_turn_candidate",
            "turn_id": turn_id,
            "performance_id": performance["performance_id"],
            "representation": "native_variable_length_dyadic_expression_turn_v1",
            "fps": 1.0 / left.frame_time,
            "source_interval": {
                "interval_role": "core_axis_fit_preview_not_training_selection",
                "start_frame": start,
                "end_frame_exclusive": end,
                "frame_count": frame_count,
                "sample_span_sec": (frame_count - 1) * left.frame_time,
                "sample_span_sec_role": "diagnostic_only_never_a_cut_or_admission_gate",
            },
            "natural_boundary_evidence": {
                "cut_policy": "shared_joint_rest_only_no_elapsed_time_cut",
                "dyadic_energy_reduction": "max_of_partner_rms_joint_angular_speeds",
                "energy_joint_names": list(ENERGY_JOINTS),
                "face_joints_used": False,
                "finger_joints_used": False,
                "audio_used": False,
                "rest_energy_threshold_rad_s": threshold_rad_s,
                "rest_evidence_consecutive_frames": evidence_consecutive_frames,
                "left_boundary_basis": left_basis,
                "right_boundary_basis": right_basis,
                "apex_proxy_frame": apex,
                "apex_proxy_energy_rad_s": float(energy[apex]),
                "complete_onset_apex_offset_verified": False,
                "requires_blind_video_arc_review": True,
                "no_internal_rest_preserved_as_single_turn": no_internal_rest,
            },
            "context_plan": {
                "policy": (
                    "nested_adjacent_natural_rest_expansion_until_blind_review_"
                    "observes_complete_expression"
                ),
                "core_preview_level": 0,
                "selected_level": None,
                "selection_status": "pending_blind_expression_completeness_review",
                "selected_training_interval": None,
                "duration_gate_used": False,
                "duration_policy": NATURAL_DURATION_POLICY,
                "source_recording_interval": [0, len(energy)],
                "levels": progressive_natural_context_levels(intervals, turn_index),
                "completeness_review": {
                    "criteria": COMPLETENESS_REVIEW_CRITERIA,
                    "reviewer_blinded_to_source_text_and_emotion": True,
                    "allowed_action": (
                        "select_first_complete_level_or_expand_to_next_predeclared_level"
                    ),
                    "elapsed_seconds_may_influence_decision": False,
                    "shrinking_below_core_or_cutting_inside_a_level_allowed": False,
                    "semantic_complete": None,
                    "affect_arc_complete": None,
                    "onset_apex_offset_complete": None,
                },
                "review_instruction": (
                    "expand_when_onset_apex_offset_action_or_affect_is_incomplete; "
                    "source_scenario_and_emotion_metadata_must_not_be_exposed"
                ),
            },
            "partner_lineage": [
                {
                    "actor_id": partner["actor_id"],
                    "actor_gender_metadata": partner["actor_gender_metadata"],
                    "session_role_metadata": partner["session_role_metadata"],
                    "source_path": partner["source_path"],
                    "source_local_sha256": partner["source_local_sha256"],
                    "partner_actor_id": partners[1 - index]["actor_id"],
                    "source_frame_interval": [start, end],
                }
                for index, partner in enumerate(partners)
            ],
            "source_scenario_metadata": scenario_metadata,
            "source_metadata_use_policy": (
                "provenance_only_never_self_certifies_action_semantics_or_affect"
            ),
            "semantic_supervision_masks": dict(SEMANTIC_MASKS),
            "emotion_supervision_mask": False,
            "source_emotion_conditioning_mask": False,
            "relationship_conditioning_mask": False,
            "scenario_conditioning_mask": False,
            "physical_qc_mask": False,
            "expression_completeness_mask": False,
            "admission_mask": False,
            "accepted_for_training": False,
        }
        candidate["candidate_record_sha256"] = record_sha256(candidate)
        candidates.append(candidate)
    return candidates


def build_actor_robot_episode_tasks(
    candidates: Iterable[dict[str, Any]],
    *,
    collection_scope: str = PILOT_COLLECTION_SCOPE,
) -> list[dict[str, Any]]:
    if collection_scope not in {PILOT_COLLECTION_SCOPE, FULL_COLLECTION_SCOPE}:
        raise ValueError("Unknown InterAct collection scope")
    tasks = []
    for candidate in candidates:
        partners = candidate["partner_lineage"]
        if len(partners) != 2:
            raise ValueError("Actor-specific episode expansion requires exactly two partners")
        for actor_index, actor in enumerate(partners):
            partner = partners[1 - actor_index]
            task = {
                "schema_version": "1.1.0",
                "artifact_kind": "interact_actor_specific_robot_episode_task",
                "episode_task_id": f"{candidate['turn_id']}__actor_{actor['actor_id']}",
                "turn_id": candidate["turn_id"],
                "performance_id": candidate["performance_id"],
                "target_actor_lineage": actor,
                "interaction_partner_lineage": partner,
                "source_interval": dict(candidate["source_interval"]),
                "training_source_interval": None,
                "natural_boundary_evidence": candidate["natural_boundary_evidence"],
                "context_plan": candidate["context_plan"],
                "source_scenario_metadata": candidate["source_scenario_metadata"],
                "retarget_task": {
                    "source_bvh": actor["source_path"],
                    "source_bvh_sha256": actor["source_local_sha256"],
                    "source_frame_interval": list(actor["source_frame_interval"]),
                    "source_frame_interval_role": (
                        "core_axis_fit_preview_only_pending_expression_completeness_review"
                    ),
                    "output_contract": "ula_v2_18d_head_v1",
                    "target_actor_only": True,
                    "partner_motion_mixed_into_target": False,
                    "partner_retained_for_interaction_review": True,
                    "axis_visual_qc_status": "pending_mujoco_blind_direction_review",
                },
                "collection_scope": collection_scope,
                "semantic_supervision_masks": dict(SEMANTIC_MASKS),
                "emotion_supervision_mask": False,
                "source_emotion_conditioning_mask": False,
                "relationship_conditioning_mask": False,
                "scenario_conditioning_mask": False,
                "physical_qc_mask": False,
                "axis_qc_mask": False,
                "expression_completeness_mask": False,
                "admission_mask": False,
                "accepted_for_retarget_batch": False,
                "accepted_for_training": False,
            }
            task["episode_task_record_sha256"] = record_sha256(task)
            tasks.append(task)
    return tasks


def build_summary(
    receipt: dict[str, Any],
    performances: list[dict[str, Any]],
    actor_clips: list[dict[str, Any]],
    pilot_performances: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    actor_episode_tasks: list[dict[str, Any]],
    *,
    collection_scope: str,
) -> dict[str, Any]:
    spans = [row["source_interval"]["sample_span_sec"] for row in candidates]
    source_emotions = Counter(
        row["source_scenario_metadata"]["primary_emotion_name"]
        for row in candidates
        if row.get("source_scenario_metadata")
    )
    duration_findings = duration_constraint_key_paths(
        {"candidates": candidates, "actor_episode_tasks": actor_episode_tasks}
    )
    if duration_findings:
        raise ValueError(
            "Forbidden duration constraint field(s): " + ", ".join(duration_findings)
        )
    return {
        "schema_version": "1.1.0",
        "artifact_kind": "interact_dyadic_natural_rest_catalog_summary",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": receipt["source"]["revision"],
        "license_gate": receipt["license_gate"],
        "inventory": {
            "remote_performance_count": len(performances),
            "remote_actor_clip_count": len(actor_clips),
            "locally_complete_valid_pair_count": sum(
                row["pair_available_for_turn_candidate_extraction"]
                for row in performances
            ),
            "paired_source_contract": "exactly_two_session-matched_actor_BVH_files",
        },
        "selection": {
            "scope": collection_scope,
            "real_performance_count": len(pilot_performances),
            "candidate_turn_count": len(candidates),
            "actor_specific_robot_episode_task_count": len(actor_episode_tasks),
            "candidate_sample_span_sec": {
                "minimum": min(spans) if spans else None,
                "median": float(np.median(spans)) if spans else None,
                "maximum": max(spans) if spans else None,
            },
            "no_internal_rest_single_turn_count": sum(
                row["natural_boundary_evidence"][
                    "no_internal_rest_preserved_as_single_turn"
                ]
                for row in candidates
            ),
            "source_emotion_metadata_distribution_unvalidated": dict(
                sorted(source_emotions.items())
            ),
            "semantic_validated_count": 0,
            "emotion_observable_validated_count": 0,
            "accepted_for_training_count": 0,
            "retarget_axis_visually_accepted_count": 0,
        },
        "duration_policy": NATURAL_DURATION_POLICY,
        "duration_contract_audit": {
            "forbidden_constraint_key_paths": duration_findings,
            "elapsed_time_values_are_diagnostic_only": True,
            "candidate_training_interval_selected_before_completeness_review": False,
        },
        "expression_completeness_policy": {
            "criteria": COMPLETENESS_REVIEW_CRITERIA,
            "selection_status_at_catalog_build": "pending",
            "expansion_unit": "adjacent_predeclared_shared_rest_interval",
            "elapsed_seconds_used_for_expansion": False,
        },
        "admission_policy": (
            "all_masks_false_until_18d_physical_qc_and_separate_blind_action_affect_reviews"
        ),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.pilot_performance_count < 0:
        raise ValueError("--pilot-performance-count cannot be negative")
    receipt_path = args.acquisition_receipt.resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["receipt_sha256"] = sha256_file(receipt_path)
    if receipt.get("artifact_kind") != "interact_motion_only_acquisition":
        raise ValueError("Wrong acquisition receipt artifact kind")
    if receipt.get("source", {}).get("revision") != (
        "152ba832f379c465f5b1e10c67166d646014d675"
    ):
        raise ValueError("InterAct acquisition receipt revision is not pinned")
    root = args.root.resolve()
    metadata = load_source_metadata(root)
    performances, actor_clips = build_paired_inventory(root, receipt, metadata)
    eligible = [
        row for row in performances if row["pair_available_for_turn_candidate_extraction"]
    ]
    # Stable-hash selection is independent of scenario/emotion metadata.
    eligible.sort(
        key=lambda row: hashlib.sha256(row["performance_id"].encode("utf-8")).hexdigest()
    )
    pilot_performances = eligible[: args.pilot_performance_count]
    collection_scope = (
        FULL_COLLECTION_SCOPE
        if len(pilot_performances) == len(eligible)
        else PILOT_COLLECTION_SCOPE
    )
    candidates = []
    for performance in pilot_performances:
        candidates.extend(
            segment_performance(
                performance,
                threshold_rad_s=args.rest_energy_threshold_rad_s,
                evidence_consecutive_frames=args.rest_evidence_consecutive_frames,
            )
        )
    actor_episode_tasks = build_actor_robot_episode_tasks(
        candidates, collection_scope=collection_scope
    )
    output = args.output_dir.resolve()
    artifacts = {
        "performances": {
            "path": str(output / "interact_dyadic_performances.jsonl"),
            "sha256": atomic_jsonl(
                output / "interact_dyadic_performances.jsonl", performances
            ),
        },
        "actor_clips": {
            "path": str(output / "interact_actor_clips.jsonl"),
            "sha256": atomic_jsonl(output / "interact_actor_clips.jsonl", actor_clips),
        },
        "pilot_candidates": {
            "path": str(output / "interact_natural_rest_pilot_candidates.jsonl"),
            "sha256": atomic_jsonl(
                output / "interact_natural_rest_pilot_candidates.jsonl", candidates
            ),
        },
        "actor_robot_episode_tasks": {
            "path": str(output / "interact_actor_robot_episode_tasks.jsonl"),
            "sha256": atomic_jsonl(
                output / "interact_actor_robot_episode_tasks.jsonl",
                actor_episode_tasks,
            ),
        },
    }
    summary = build_summary(
        receipt,
        performances,
        actor_clips,
        pilot_performances,
        candidates,
        actor_episode_tasks,
        collection_scope=collection_scope,
    )
    summary["artifacts"] = artifacts
    summary_sha256 = atomic_json(output / "summary.json", summary)
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "summary_sha256": summary_sha256,
                **summary["inventory"],
                **summary["selection"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
