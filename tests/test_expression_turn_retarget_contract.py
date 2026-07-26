import copy
import csv
import hashlib
import json

import numpy as np
import pytest

from tools.gmr_v2.safety_monotonic_retime_v1 import (
    minimum_velocity_safety_retime,
)
from tools.human_motion_review.expression_turn_contract import (
    CANDIDATE_ARTIFACT_KIND,
    CONTEXT_POLICY,
    ExpressionTurnContractError,
)
from tools.human_motion_review.expression_turn_retarget_contract import (
    QUALITY_ARTIFACT_KIND,
    REQUIRED_18D_GATES,
    RETARGET_SEGMENT_REPRESENTATION,
    TRAINING_ADMISSION_STATUS,
    validate_expression_turn_retarget_output,
)
from upper_body_skeleton.retarget_v2_18d import (
    CONTRACT_VERSION,
    JOINT_LIMITS_18D,
    JOINT_ORDER_18D,
)


def _stable_hash(value, *, ascii_only=False):
    payload = json.dumps(
        value,
        ensure_ascii=ascii_only,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii" if ascii_only else "utf-8")
    return hashlib.sha256(payload).hexdigest()


def _candidate():
    candidate = {
        "artifact_kind": CANDIDATE_ARTIFACT_KIND,
        "clip_id": "turn-v8",
        "representation": "native_variable_length_expression_turn_v1",
        "fps": 30.0,
        "motion_sha256": "1" * 64,
        "expression_turn_contract_sha256": "2" * 64,
        "source_inventory_manifest_sha256": "3" * 64,
        "split_assignment_manifest_sha256": "4" * 64,
        "upstream_event_record_sha256": ["5" * 64],
        "core_interval": {"start_frame": 100, "end_frame_exclusive": 104},
        "context_plan": {
            "policy": CONTEXT_POLICY,
            "same_source_only": True,
            "neighbor_crossing_allowed": False,
            "source_interval": {"start_frame": 0, "end_frame_exclusive": 500},
            "admissible_interval": {
                "start_frame": 90,
                "end_frame_exclusive": 120,
            },
            "selected_level": 0,
            "levels": [
                {
                    "level": 0,
                    "start_frame": 99,
                    "end_frame_exclusive": 105,
                    "left_boundary_basis": "natural_low_motion_basin",
                    "right_boundary_basis": "natural_low_motion_basin",
                }
            ],
        },
        "training_segment": {
            "representation": "native_variable_length_expression_turn_v1",
            "start_frame": 99,
            "end_frame_exclusive": 105,
            "frame_count": 6,
            "fixed_window_sec": None,
            "cropped": False,
            "duration_policy": "natural_rest_to_natural_rest_no_fixed_or_max_duration",
        },
        "time_axes": {
            "source": {
                "start_frame": 99,
                "end_frame_exclusive": 105,
                "frame_count": 6,
                "sample_span_sec": 0.166667,
            },
            "turn": {
                "start_frame": 0,
                "end_frame_exclusive": 6,
                "frame_count": 6,
                "sample_span_sec": 0.166667,
            },
        },
        "expression_turn": {
            "complete_motion_arc_verified": False,
            "automated_motion_arc_candidate_status": (
                "natural_boundary_and_apex_proxy_pending_blind_video_review"
            ),
            "peak": {"source_frame": 102},
            "included_event_count": 1,
            "included_event_spans": [
                {
                    "source_time_axis": {
                        "start_frame_floor": 100,
                        "end_frame_exclusive_ceil": 104,
                    },
                    "turn_time_axis": {
                        "start_frame": 1,
                        "end_frame_exclusive": 5,
                    },
                }
            ],
        },
        "semantic_supervision_masks": {
            "official_category": False,
            "robot_observable_motion_form": False,
            "communicative_intent": False,
            "prompt_text": False,
            "legacy_gesture": False,
        },
        "emotion_supervision_mask": False,
        "official_emotion_conditioning_enabled": False,
        "affect_observable_supervision_mask": False,
        "official_category_conditioning_enabled": False,
        "canonical_prompt": None,
        "canonical_action": None,
        "accepted_for_training": False,
        "expression_turn_selection_kind": "stress100",
        "expression_turn_selection_rank": 1,
        "expression_turn_selection_status": "selected_stress_pending_retarget_qc",
    }
    base_payload = {
        key: value
        for key, value in candidate.items()
        if key
        not in {
            "expression_turn_selection_kind",
            "expression_turn_selection_rank",
            "expression_turn_selection_status",
        }
    }
    candidate["expression_turn_record_sha256"] = _stable_hash(base_payload)
    candidate["expression_turn_selection_record_sha256"] = _stable_hash(candidate)
    return candidate


def _binding():
    return {
        "retarget_input_manifest_sha256": "6" * 64,
        "expression_turn_contract_sha256": "2" * 64,
        "source_inventory_manifest_sha256": "3" * 64,
        "split_assignment_manifest_sha256": "4" * 64,
        "selection_kind": "stress100",
        "require_selection_record": True,
    }


def _write_safe_csv(path, *, rows=6, nonfinite=False):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(JOINT_ORDER_18D)
        for row_index in range(rows):
            values = [0.01 * row_index] * len(JOINT_ORDER_18D)
            if nonfinite and row_index == 0:
                values[-1] = float("nan")
            writer.writerow(values)


def _write_trajectory(path, trajectory):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(JOINT_ORDER_18D)
        writer.writerows(trajectory)


def _quality(candidate, binding, safe_csv, safety_retime):
    selected_hash = _stable_hash(candidate)
    output_frames = safety_retime["output_frame_count"]
    retarget_segment = {
        "representation": RETARGET_SEGMENT_REPRESENTATION,
        "source_start_frame": 99,
        "source_end_frame_exclusive": 105,
        "source_frame_count": 6,
        "source_frame_coverage_sec": 0.2,
        "output_frame_count": output_frames,
        "output_sample_span_sec": (output_frames - 1) / 30,
        "output_frame_coverage_sec": output_frames / 30,
        "fps": 30.0,
        "retimed": output_frames != 6,
        "cropped": False,
        "duration_policy": "natural_rest_to_natural_rest_no_fixed_or_max_duration",
        "retime_policy": safety_retime["algorithm"],
        "max_slowdown_ratio": safety_retime["max_slowdown_ratio"],
        "fixed_target_duration_sec": None,
        "planner_duration_field": "output_sample_span_sec",
        "source_boundary_duration_field": "source_frame_coverage_sec",
        "legacy_quality_duration_sec_role": (
            "output_frame_coverage_compatibility_only_not_planner_target"
        ),
    }
    retarget_segment["sha256"] = _stable_hash(retarget_segment, ascii_only=True)
    return {
        "artifact_kind": QUALITY_ARTIFACT_KIND,
        "clip_id": candidate["clip_id"],
        "output_contract": CONTRACT_VERSION,
        "action_dim": 18,
        "joint_order": list(JOINT_ORDER_18D),
        "fps": 30.0,
        "frames": output_frames,
        "source_window_start_frame": 99,
        "source_window_end_frame_exclusive": 105,
        "source_window_frames": 6,
        "source_sha256": candidate["motion_sha256"],
        "training_segment": copy.deepcopy(candidate["training_segment"]),
        "context_plan": copy.deepcopy(candidate["context_plan"]),
        "time_axes": copy.deepcopy(candidate["time_axes"]),
        "expression_turn": copy.deepcopy(candidate["expression_turn"]),
        "expression_turn_contract_sha256": candidate[
            "expression_turn_contract_sha256"
        ],
        "source_inventory_manifest_sha256": candidate[
            "source_inventory_manifest_sha256"
        ],
        "split_assignment_manifest_sha256": candidate[
            "split_assignment_manifest_sha256"
        ],
        "motion_sha256": candidate["motion_sha256"],
        "upstream_event_record_sha256": candidate[
            "upstream_event_record_sha256"
        ],
        "inventory_record_sha256": candidate[
            "expression_turn_selection_record_sha256"
        ],
        "upstream_inventory_record_sha256": candidate[
            "expression_turn_record_sha256"
        ],
        "selected_record_sha256": selected_hash,
        "retarget_input_manifest_sha256": binding[
            "retarget_input_manifest_sha256"
        ],
        "retarget_segment": retarget_segment,
        "safety_monotonic_retime": safety_retime,
        "quality_gate": {gate: True for gate in REQUIRED_18D_GATES},
        "semantic_supervision_masks": copy.deepcopy(
            candidate["semantic_supervision_masks"]
        ),
        "emotion_supervision_mask": False,
        "official_emotion_conditioning_enabled": False,
        "affect_observable_supervision_mask": False,
        "official_category_conditioning_enabled": False,
        "canonical_prompt": None,
        "canonical_action": None,
        "training_admission_status": TRAINING_ADMISSION_STATUS,
        "accepted_for_training": False,
        "outputs": {"safe_csv": str(safe_csv.resolve())},
        "safe_csv_sha256": hashlib.sha256(safe_csv.read_bytes()).hexdigest(),
    }


def _fixture(tmp_path):
    candidate = _candidate()
    binding = _binding()
    safe_csv = tmp_path / "turn_gmr_safe_18d.csv"
    midpoint = np.asarray(
        [
            (JOINT_LIMITS_18D[name][0] + JOINT_LIMITS_18D[name][1]) / 2
            for name in JOINT_ORDER_18D
        ]
    )
    raw = np.tile(midpoint, (6, 1))
    safe, _key_times, _output_times, safety_retime = (
        minimum_velocity_safety_retime(
            raw,
            fps=30.0,
            max_velocity_rad_s=3.0,
            smoothing_window=3,
            joint_order=JOINT_ORDER_18D,
            joint_limits=JOINT_LIMITS_18D,
        )
    )
    _write_trajectory(safe_csv, safe)
    quality = _quality(candidate, binding, safe_csv, safety_retime)
    return candidate, binding, safe_csv, quality


def _retimed_fixture(tmp_path, delta):
    candidate = _candidate()
    binding = _binding()
    midpoint = np.asarray(
        [
            (JOINT_LIMITS_18D[name][0] + JOINT_LIMITS_18D[name][1]) / 2
            for name in JOINT_ORDER_18D
        ]
    )
    raw = np.tile(midpoint, (6, 1))
    raw[3:, 0] += delta
    safe, _key_times, _output_times, safety_retime = (
        minimum_velocity_safety_retime(
            raw,
            fps=30.0,
            max_velocity_rad_s=3.0,
            smoothing_window=3,
            joint_order=JOINT_ORDER_18D,
            joint_limits=JOINT_LIMITS_18D,
        )
    )
    safe_csv = tmp_path / "turn_gmr_safe_18d.csv"
    _write_trajectory(safe_csv, safe)
    quality = _quality(candidate, binding, safe_csv, safety_retime)
    return candidate, binding, safe_csv, quality


def test_valid_retarget_is_physical_only_and_keeps_training_closed(tmp_path):
    candidate, binding, safe_csv, quality = _fixture(tmp_path)
    result = validate_expression_turn_retarget_output(
        quality,
        input_record=candidate,
        catalog_binding=binding,
        safe_csv_path=safe_csv,
    )
    assert result["retarget_output_valid"] is True
    assert result["physical_qc_eligible"] is True
    assert result["base_motion_eligible"] is False
    assert result["semantic_conditioning_eligible"] is False
    assert result["expressive_conditioning_eligible"] is False
    assert result["emotion_supervision_mask"] is False


def test_retarget_cannot_change_natural_segment_or_dual_time_axes(tmp_path):
    candidate, binding, safe_csv, quality = _fixture(tmp_path)
    quality["training_segment"]["start_frame"] = 98
    with pytest.raises(ExpressionTurnContractError, match="natural expression-turn"):
        validate_expression_turn_retarget_output(
            quality, input_record=candidate, catalog_binding=binding
        )

    candidate, binding, safe_csv, quality = _fixture(tmp_path)
    quality["time_axes"]["turn"]["sample_span_sec"] = 0.2
    with pytest.raises(ExpressionTurnContractError, match="time_axes"):
        validate_expression_turn_retarget_output(
            quality, input_record=candidate, catalog_binding=binding
        )


def test_retarget_requires_exact_candidate_and_manifest_lineage(tmp_path):
    candidate, binding, safe_csv, quality = _fixture(tmp_path)
    quality["selected_record_sha256"] = "0" * 64
    with pytest.raises(ExpressionTurnContractError, match="selected_record_sha256"):
        validate_expression_turn_retarget_output(
            quality, input_record=candidate, catalog_binding=binding
        )

    candidate, binding, safe_csv, quality = _fixture(tmp_path)
    quality["expression_turn_contract_sha256"] = "0" * 64
    with pytest.raises(ExpressionTurnContractError, match="did not preserve"):
        validate_expression_turn_retarget_output(
            quality, input_record=candidate, catalog_binding=binding
        )


def test_legacy_semantic_event_or_fixed_window_contract_is_rejected(tmp_path):
    candidate, binding, safe_csv, quality = _fixture(tmp_path)
    quality["semantic_event"] = {"category": "deictic"}
    with pytest.raises(ExpressionTurnContractError, match="legacy semantic-event"):
        validate_expression_turn_retarget_output(
            quality, input_record=candidate, catalog_binding=binding
        )

    candidate = _candidate()
    candidate["training_segment"]["fixed_window_sec"] = 6.0
    binding = _binding()
    with pytest.raises(ExpressionTurnContractError, match="duration is diagnostic"):
        validate_expression_turn_retarget_output(
            {}, input_record=candidate, catalog_binding=binding
        )


def test_retarget_cannot_enable_semantic_emotion_or_training_admission(tmp_path):
    candidate, binding, safe_csv, quality = _fixture(tmp_path)
    quality["semantic_supervision_masks"]["prompt_text"] = True
    with pytest.raises(ExpressionTurnContractError, match="semantic masks"):
        validate_expression_turn_retarget_output(
            quality, input_record=candidate, catalog_binding=binding
        )

    candidate, binding, safe_csv, quality = _fixture(tmp_path)
    quality["emotion_supervision_mask"] = True
    with pytest.raises(ExpressionTurnContractError, match="enabled semantic"):
        validate_expression_turn_retarget_output(
            quality, input_record=candidate, catalog_binding=binding
        )


def test_all_required_and_reported_18d_quality_gates_must_pass(tmp_path):
    candidate, binding, safe_csv, quality = _fixture(tmp_path)
    quality["quality_gate"].pop("head_continuity_pass")
    with pytest.raises(ExpressionTurnContractError, match="missing 18D gates"):
        validate_expression_turn_retarget_output(
            quality, input_record=candidate, catalog_binding=binding
        )

    candidate, binding, safe_csv, quality = _fixture(tmp_path)
    quality["quality_gate"]["head_continuity_pass"] = False
    with pytest.raises(ExpressionTurnContractError, match="must pass"):
        validate_expression_turn_retarget_output(
            quality, input_record=candidate, catalog_binding=binding
        )


def test_safe_csv_hash_shape_values_and_row_count_are_verified(tmp_path):
    candidate, binding, safe_csv, quality = _fixture(tmp_path)
    quality["safe_csv_sha256"] = "0" * 64
    with pytest.raises(ExpressionTurnContractError, match="SHA256"):
        validate_expression_turn_retarget_output(
            quality, input_record=candidate, catalog_binding=binding
        )

    candidate, binding, safe_csv, quality = _fixture(tmp_path)
    _write_safe_csv(safe_csv, nonfinite=True)
    quality["safe_csv_sha256"] = hashlib.sha256(safe_csv.read_bytes()).hexdigest()
    with pytest.raises(ExpressionTurnContractError, match="non-finite"):
        validate_expression_turn_retarget_output(
            quality, input_record=candidate, catalog_binding=binding
        )

    candidate, binding, safe_csv, quality = _fixture(tmp_path)
    _write_safe_csv(safe_csv, rows=5)
    quality["safe_csv_sha256"] = hashlib.sha256(safe_csv.read_bytes()).hexdigest()
    with pytest.raises(ExpressionTurnContractError, match="row count"):
        validate_expression_turn_retarget_output(
            quality, input_record=candidate, catalog_binding=binding
        )


def test_retarget_segment_hash_and_duration_semantics_are_verified(tmp_path):
    candidate, binding, safe_csv, quality = _fixture(tmp_path)
    quality["retarget_segment"]["output_sample_span_sec"] = 0.2
    with pytest.raises(ExpressionTurnContractError, match="SHA256"):
        validate_expression_turn_retarget_output(
            quality, input_record=candidate, catalog_binding=binding
        )


def test_minimum_safety_slowdown_preserves_full_arc_and_is_accepted(tmp_path):
    candidate, binding, safe_csv, quality = _retimed_fixture(tmp_path, 0.11)

    result = validate_expression_turn_retarget_output(
        quality,
        input_record=candidate,
        catalog_binding=binding,
        safe_csv_path=safe_csv,
    )

    assert quality["frames"] == 7
    assert quality["retarget_segment"]["retimed"] is True
    assert result["retarget_output_valid"] is True


def test_safety_slowdown_above_ratio_cap_is_quarantined(tmp_path):
    candidate, binding, safe_csv, quality = _retimed_fixture(tmp_path, 0.5)

    with pytest.raises(ExpressionTurnContractError, match="slowdown_ratio_pass"):
        validate_expression_turn_retarget_output(
            quality,
            input_record=candidate,
            catalog_binding=binding,
            safe_csv_path=safe_csv,
        )
