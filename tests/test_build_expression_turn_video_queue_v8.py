import copy
import json
from pathlib import Path

import pytest

from tools.human_motion_review.build_expression_turn_video_queue_v8 import (
    ANONYMOUS_PROMPT,
    INPUT_REPRESENTATION,
    LEGACY_NATIVE_NOOP_REPRESENTATION,
    NATURAL_DURATION_POLICY,
    OUTPUT_ARTIFACT_KIND,
    RETARGET_SEGMENT_REPRESENTATION,
    SEMANTIC_MASKS,
    build_queue,
    queue_record,
    sha256,
)
from upper_body_skeleton.retarget_v2_18d import JOINT_ORDER_18D
from tools.human_motion_review.render_beat2_annotation_review import (
    review_projection,
    validate_queue_structure,
)


def _fixture(tmp_path):
    safe_csv = tmp_path / "turn_gmr_safe_18d.csv"
    safe_csv.write_text(
        ",".join(JOINT_ORDER_18D)
        + "\n"
        + "".join(",".join(["0"] * 18) + "\n" for _ in range(32)),
        encoding="utf-8",
    )
    training = {
        "representation": INPUT_REPRESENTATION,
        "start_frame": 90,
        "end_frame_exclusive": 120,
        "frame_count": 30,
        "fixed_window_sec": None,
        "cropped": False,
        "duration_policy": NATURAL_DURATION_POLICY,
    }
    retarget = {
        "representation": RETARGET_SEGMENT_REPRESENTATION,
        "source_frame_count": 30,
        "output_frame_count": 32,
        "source_frame_coverage_sec": 1.0,
        "output_frame_coverage_sec": 32 / 30,
        "output_sample_span_sec": 31 / 30,
        "duration_policy": NATURAL_DURATION_POLICY,
        "fixed_target_duration_sec": None,
        "retimed": True,
        "cropped": False,
    }
    safety = {
        "artifact_kind": "ula_18d_safety_monotonic_retime_v1",
        "blind_review_must_use_retimed_output": True,
        "source_frame_count": 30,
        "output_frame_count": 32,
        "minimum_output_frame_count": 32,
        "time_map_strictly_increasing": True,
        "input_frame_output_times_sec": [index * (31 / 30) / 29 for index in range(30)],
        "first_frame_preserved": True,
        "last_frame_preserved": True,
        "post_velocity_pass": True,
        "slowdown_ratio_pass": True,
        "retime_ratio": 32 / 30,
        "max_slowdown_ratio": 1.25,
        "cropped": False,
        "tiled": False,
        "target_duration_sec": None,
    }
    quality = {
        "training_segment": copy.deepcopy(training),
        "retarget_segment": copy.deepcopy(retarget),
        "safety_monotonic_retime": safety,
        "expression_turn_output_contract_validation": {
            "passed": True,
            "status": "validated_physical_only_training_still_closed",
        },
    }
    quality_path = tmp_path / "quality.json"
    quality_path.write_text(json.dumps(quality), encoding="utf-8")
    record = {
        "artifact_kind": OUTPUT_ARTIFACT_KIND,
        "status": "passed",
        "task_id": "turn-v8",
        "source_clip_id": "source-a",
        "speaker_key": "speaker-a",
        "official_split": "train",
        "fixed_split_assignment": "train",
        "fps": 30,
        "frames": 32,
        "accepted_for_training": False,
        "semantic_supervision_masks": dict(SEMANTIC_MASKS),
        "emotion_supervision_mask": False,
        "official_emotion_conditioning_enabled": False,
        "affect_observable_supervision_mask": False,
        "official_category_conditioning_enabled": False,
        "canonical_prompt": None,
        "canonical_action": None,
        "emotion_id": "fear",
        "source_emotion_label_verified": True,
        "training_segment": training,
        "retarget_segment": retarget,
        "core_interval": {"start_frame": 95, "end_frame_exclusive": 115},
        "context_plan": {"selected_level": 0},
        "time_axes": {"source": {}, "turn": {}},
        "expression_turn": {"complete_motion_arc_verified": False},
        "duration_band": "short_under_3s",
        "event_count_band": "single",
        "quality_gate": {
            "joint_limits_pass": True,
            "velocity_pass": True,
            "passed": True,
        },
        "safe_csv": str(safe_csv),
        "safe_csv_sha256": sha256(safe_csv),
        "quality_json": str(quality_path),
        "quality_json_sha256": sha256(quality_path),
        "expression_turn_contract_sha256": "1" * 64,
        "expression_turn_record_sha256": "2" * 64,
        "expression_turn_selection_kind": "representative100",
        "expression_turn_selection_rank": 1,
        "expression_turn_selection_status": (
            "selected_representative_pending_retarget_qc"
        ),
        "expression_turn_selection_record_sha256": "3" * 64,
        "source_inventory_manifest_sha256": "4" * 64,
        "split_assignment_manifest_sha256": "5" * 64,
        "upstream_event_record_sha256": ["6" * 64],
        "upstream_inventory_record_sha256": "2" * 64,
        "selected_record_sha256": "7" * 64,
        "retarget_input_manifest_sha256": "8" * 64,
    }
    return record


def _rewrite_evidence(record, *, frames):
    safe_csv = Path(record["safe_csv"])
    safe_csv.write_text(
        ",".join(JOINT_ORDER_18D)
        + "\n"
        + "".join(",".join(["0"] * 18) + "\n" for _ in range(frames)),
        encoding="utf-8",
    )
    record["safe_csv_sha256"] = sha256(safe_csv)
    quality_path = Path(record["quality_json"])
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["training_segment"] = copy.deepcopy(record["training_segment"])
    quality["retarget_segment"] = copy.deepcopy(record["retarget_segment"])
    quality_path.write_text(json.dumps(quality), encoding="utf-8")
    record["quality_json_sha256"] = sha256(quality_path)


def test_queue_preserves_natural_full_length_without_action_or_emotion_prompt(tmp_path):
    record = _fixture(tmp_path)
    queued = queue_record(record, expected_selection_kind="representative100")
    assert queued["training_segment"] == record["training_segment"]
    assert queued["retarget_segment"] == record["retarget_segment"]
    assert queued["expression_turn"] == record["expression_turn"]
    assert queued["source_frame_count"] == 30
    assert queued["output_frame_count"] == 32
    assert queued["trajectory_frames_expected"] == 32
    assert queued["blind_review_must_use_final_trajectory"] is True
    assert queued["safety_monotonic_retime"]["output_frame_count"] == 32
    assert queued["canonical_action"] is None
    assert queued["canonical_prompt"] == ANONYMOUS_PROMPT
    assert "fear" not in queued["canonical_prompt"]["en"].lower()
    assert "semantic_event" not in queued
    assert queued["accepted_for_training"] is False
    validate_queue_structure([queued])
    projected = review_projection(
        queued, rank=0, sampling="duration_quantiles", seed=0
    )
    assert projected["expression_turn"] == record["expression_turn"]
    assert projected["training_segment"] == record["training_segment"]
    assert "semantic_event" not in projected


def test_queue_rejects_legacy_fixed_or_invalid_output_record(tmp_path):
    record = _fixture(tmp_path)
    record["semantic_event"] = {"category": "deictic"}
    with pytest.raises(ValueError, match="legacy semantic-event"):
        queue_record(record, expected_selection_kind="representative100")

    record = _fixture(tmp_path)
    record["training_segment"]["fixed_window_sec"] = 6.0
    with pytest.raises(ValueError, match="natural length"):
        queue_record(record, expected_selection_kind="representative100")

    record = _fixture(tmp_path)
    record["retarget_segment"]["output_frame_count"] = 29
    with pytest.raises(ValueError, match="invalid source/output frame contract"):
        queue_record(record, expected_selection_kind="representative100")


def test_queue_accepts_verified_native_identity_timeline(tmp_path):
    record = _fixture(tmp_path)
    record["frames"] = 30
    record["retarget_segment"].update(
        {
            "representation": LEGACY_NATIVE_NOOP_REPRESENTATION,
            "output_frame_count": 30,
            "output_frame_coverage_sec": 1.0,
            "output_sample_span_sec": 29 / 30,
            "retimed": False,
        }
    )
    quality_path = Path(record["quality_json"])
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["safety_monotonic_retime"] = None
    quality_path.write_text(json.dumps(quality), encoding="utf-8")
    _rewrite_evidence(record, frames=30)

    queued = queue_record(record, expected_selection_kind="representative100")
    assert queued["output_frame_count"] == 30
    assert queued["final_trajectory_role"] == "native_identity_timeline_no_slowdown_required"
    assert "safety_monotonic_retime" not in queued


def test_build_queue_binds_passed_manifest_and_selection_kind(tmp_path):
    record = _fixture(tmp_path)
    passed = tmp_path / "passed.jsonl"
    passed.write_text(json.dumps(record) + "\n", encoding="utf-8")
    output = tmp_path / "review_queue.jsonl"
    summary = build_queue(
        passed, output, selection_kind="representative100"
    )
    assert summary["records"] == 1
    assert summary["selection_kind"] == "representative100"
    assert summary["natural_full_length_render_required"] is True
    assert summary["source_frame_count_total"] == 30
    assert summary["output_frame_count_total"] == 32
    assert summary["renderer_prompt_contains_action_or_emotion_target"] is False
    queued = json.loads(output.read_text(encoding="utf-8"))
    assert queued["expression_turn_selection_kind"] == "representative100"

    with pytest.raises(ValueError, match="selection kind mismatch"):
        build_queue(passed, output, selection_kind="stress100")
