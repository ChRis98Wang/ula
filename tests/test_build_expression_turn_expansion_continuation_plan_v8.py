import json
from pathlib import Path

import pytest

from tools.human_motion_review.build_expression_turn_expansion_continuation_plan_v8 import (
    EXPANSION_UNIT,
    NATURAL_DURATION_POLICY,
    PIPELINE_KIND,
    PLAN_KIND,
    PUBLIC_DURATION_POLICY,
    PUBLIC_KIND,
    REQUEST_KIND,
    REVIEW_SUMMARY_KIND,
    SELECTION_POLICY,
    build_plan,
    sha256_file,
    stable_json,
    value_sha256,
)
from tools.human_motion_review.expression_turn_contract import CONTEXT_POLICY


def _json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(stable_json(row) + "\n" for row in rows), encoding="utf-8")


def _binding(path: Path):
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _case(
    tmp_path: Path,
    *,
    phases=("complete", "complete", "incomplete"),
    final_level=2,
    previous_requested_level=1,
    malformed_level2=False,
    pipeline_stage="render",
):
    root = tmp_path
    task = "beat_source_turn0001"
    base_sample = "expr_base"
    anonymous = "expr_anon"
    source = "source_1"
    split = "train"
    levels = [
        {"level": 0, "start_frame": 20, "end_frame_exclusive": 40},
        {
            "level": 1,
            "parent_level": 0,
            "start_frame": 10,
            "end_frame_exclusive": 50,
        },
    ]
    if final_level >= 2:
        levels.append(
            {
                "level": 2,
                "parent_level": 1,
                "start_frame": 15 if malformed_level2 else 0,
                "end_frame_exclusive": 60,
            }
        )
    candidate = {
        "task_id": task,
        "source_clip_id": source,
        "fixed_split_assignment": split,
        "context_plan": {
            "policy": CONTEXT_POLICY,
            "same_source_only": True,
            "neighbor_crossing_allowed": False,
            "selected_level": 0,
            "source_interval": {"start_frame": 0, "end_frame_exclusive": 100},
            "admissible_interval": {"start_frame": 0, "end_frame_exclusive": 60},
            "context_exhausted_at_level": final_level,
            "levels": levels,
        },
    }
    candidate["expression_turn_record_sha256"] = value_sha256(candidate)
    catalog = root / "beat2_expression_turn_v8.candidates.jsonl"
    _jsonl(catalog, [candidate])
    contract = {"duration_policy": NATURAL_DURATION_POLICY, "fixed_window_sec": None}
    catalog_summary = root / "beat2_expression_turn_v8.summary.json"
    _json(
        catalog_summary,
        {
            "artifact_kind": "beat2_expression_turn_v8_candidate_catalog",
            "accepted_for_training": 0,
            "candidate_count": 1,
            "fixed_window_sec": None,
            "expression_turn_contract": contract,
            "expression_turn_contract_sha256": value_sha256(contract),
            "output_sha256": {catalog.name: sha256_file(catalog)},
        },
    )

    base_mapping = root / "base_mapping.jsonl"
    _jsonl(
        base_mapping,
        [
            {
                "sample_id": base_sample,
                "task_id": task,
                "source_clip_id": source,
                "fixed_split_assignment": split,
                "expression_turn_record_sha256": candidate[
                    "expression_turn_record_sha256"
                ],
                "accepted_for_training": False,
            }
        ],
    )
    previous = {
        "schema_version": "1.0.0",
        "sample_id": base_sample,
        "artifact_kind": REQUEST_KIND,
        "base_task_id": task,
        "source_clip_id": source,
        "fixed_split_assignment": split,
        "base_expression_turn_record_sha256": candidate[
            "expression_turn_record_sha256"
        ],
        "comparison_record_sha256": "c" * 64,
        "review_qualification_status": "natural_context_expansion_required",
        "reviewed_context_level": 0,
        "reviewed_interval": {
            "start_frame": 20,
            "end_frame_exclusive": 40,
            "frame_count": 20,
        },
        "requested_context_level": previous_requested_level,
        "requested_interval": {
            "start_frame": 10,
            "end_frame_exclusive": 50,
            "frame_count": 40,
        },
        "strictly_contains_reviewed_interval": True,
        "expansion_unit": EXPANSION_UNIT,
        "elapsed_duration_used_as_gate": False,
        "semantic_supervision_mask": False,
        "emotion_supervision_mask": False,
        "accepted_for_training": False,
    }
    previous["plan_record_sha256"] = value_sha256(previous)
    previous_requests = root / "previous_requests.jsonl"
    _jsonl(previous_requests, [previous])
    previous_plan = root / "previous_plan.json"
    _json(
        previous_plan,
        {
            "schema_version": "1.0.0",
            "artifact_kind": PLAN_KIND,
            "inputs": {
                "hidden_mapping_sha256": sha256_file(base_mapping),
                "candidate_catalog_sha256": sha256_file(catalog),
            },
            "selection_policy": SELECTION_POLICY,
            "fixed_minimum_maximum_or_target_duration_used": False,
            "accepted_for_training_count": 0,
            "outputs": {
                "expansion_requests": {
                    "path": str(previous_requests.resolve()),
                    "sha256": sha256_file(previous_requests),
                    "records": 1,
                }
            },
        },
    )

    render_video = root / "physical" / "video.mp4"
    trajectory = root / "physical" / "trajectory.csv"
    render_video.parent.mkdir(parents=True, exist_ok=True)
    render_video.write_bytes(b"variable length video evidence")
    trajectory.write_text("joint\n" + "0\n" * 40, encoding="utf-8")
    derived_task = f"{task}__ctxL01"
    render = {
        "task_id": derived_task,
        "video_path": str(render_video.resolve()),
        "video_sha256": sha256_file(render_video),
        "trajectory_path": str(trajectory.resolve()),
        "trajectory_sha256": sha256_file(trajectory),
        "trajectory_frames": 40,
        "training_segment": {
            "start_frame": 10,
            "end_frame_exclusive": 50,
            "frame_count": 40,
            "fixed_window_sec": None,
            "cropped": False,
        },
        "retarget_segment": {
            "source_start_frame": 10,
            "source_end_frame_exclusive": 50,
            "source_frame_count": 40,
            "output_frame_count": 40,
            "fixed_target_duration_sec": None,
            "cropped": False,
        },
        "processing_scope": "physical_retarget_and_silent_render_only_pending_fresh_blind_arc_action_and_affect_review",
        "license_training_admission": False,
        "render_pass_grants_training_admission": False,
        "expansion_provenance": {
            "base_candidate_catalog_sha256": sha256_file(catalog),
            "expansion_plan_summary_sha256": sha256_file(previous_plan),
            "expansion_requests_sha256": sha256_file(previous_requests),
            "plan_record_sha256": previous["plan_record_sha256"],
            "comparison_record_sha256": previous["comparison_record_sha256"],
            "base_task_id": task,
            "base_expression_turn_record_sha256": candidate[
                "expression_turn_record_sha256"
            ],
            "sample_id": base_sample,
            "reviewed_context_level": 0,
            "requested_context_level": 1,
            "selection_policy": SELECTION_POLICY,
            "expansion_unit": EXPANSION_UNIT,
            "elapsed_duration_used_as_gate": False,
            "accepted_for_training": False,
        },
    }
    passed = root / "physical" / "passed.jsonl"
    _jsonl(passed, [render])
    audit = {
        "inputs": {
            "expansion_plan_summary": _binding(previous_plan),
            "expansion_requests": _binding(previous_requests),
            "candidate_catalog": _binding(catalog),
        }
    }
    audit["sha256"] = value_sha256(audit)
    pipeline = root / "physical" / "pipeline.json"
    _json(
        pipeline,
        {
            "artifact_kind": PIPELINE_KIND,
            "stage": pipeline_stage,
            "fixed_six_second_windows_used": False,
            "elapsed_seconds_used_for_selection": False,
            "accepted_for_training": False,
            "input_audit": audit,
            "render_summary": {
                "passed_manifest": str(passed.resolve()),
                "passed_manifest_sha256": sha256_file(passed),
                "counts": {"passed": 1, "failed": 0},
            },
        },
    )

    public_root = root / "public"
    public_video = public_root / "videos" / f"{anonymous}.mp4"
    public_video.parent.mkdir(parents=True, exist_ok=True)
    public_video.write_bytes(render_video.read_bytes())
    queue = public_root / "arc_action_review_queue.jsonl"
    queue_row = {
        "sample_id": anonymous,
        "video_path": str(public_video.resolve()),
        "video_sha256": sha256_file(public_video),
        "context_level": 1,
        "frame_count": 40,
        "fps": 30.0,
        "native_duration_preserved": True,
        "fixed_duration_window_used": False,
        "audio_available": False,
        "label_metadata_exposed": False,
    }
    _jsonl(queue, [queue_row])
    public_summary = public_root / "summary.json"
    _json(
        public_summary,
        {
            "artifact_kind": PUBLIC_KIND,
            "duration_policy": PUBLIC_DURATION_POLICY,
            "all_samples_native_variable_length": True,
            "fixed_duration_window_used": False,
            "accepted_for_training": False,
            "records": 1,
            "arc_action_queue": str(queue.resolve()),
            "arc_action_queue_sha256": sha256_file(queue),
        },
    )

    mapping = root / "hidden" / "mapping.jsonl"
    mapping_row = {
        "sample_id": anonymous,
        "base_sample_id": "legacy_wrong_id",
        "base_task_id": task,
        "task_id": derived_task,
        "derived_task_id": derived_task,
        "source_clip_id": source,
        "fixed_split_assignment": split,
        "reviewed_context_level": 0,
        "displayed_context_level": 1,
        "reviewed_interval": previous["reviewed_interval"],
        "displayed_interval": previous["requested_interval"],
        "plan_record_sha256": previous["plan_record_sha256"],
        "comparison_record_sha256": previous["comparison_record_sha256"],
        "source_render_record_sha256": value_sha256(render),
        "video_sha256": render["video_sha256"],
        "trajectory_sha256": render["trajectory_sha256"],
        "frame_count": 40,
        "native_duration_preserved": True,
        "fixed_duration_window_used": False,
        "official_action_text_or_affect_exposed": False,
        "accepted_for_training": False,
    }
    _jsonl(mapping, [mapping_row])
    hidden_summary = root / "hidden" / "summary.json"
    _json(
        hidden_summary,
        {
            "artifact_kind": "expression_turn_v8_expansion_hidden_blind_mapping_v1",
            "accepted_for_training": False,
            "public_distribution_forbidden": True,
            "records": 1,
            "requested_records": 1,
            "physical_qc_excluded_records": 0,
            "physical_qc_excluded_base_sample_ids": [],
            "mapping": str(mapping.resolve()),
            "mapping_sha256": sha256_file(mapping),
            "expansion_plan_summary": str(previous_plan.resolve()),
            "expansion_plan_summary_sha256": sha256_file(previous_plan),
            "expansion_requests": str(previous_requests.resolve()),
            "expansion_requests_sha256": sha256_file(previous_requests),
            "pipeline_summary": str(pipeline.resolve()),
            "pipeline_summary_sha256": sha256_file(pipeline),
            "render_passed_manifest": str(passed.resolve()),
            "render_passed_manifest_sha256": sha256_file(passed),
        },
    )

    review = root / "review.jsonl"
    review_row = {
        **queue_row,
        "queue_sha256": sha256_file(queue),
        "action_result": "pass",
        "action_observability": "observable",
        "observable_description": "Raises both arms, reaches an apex, and starts to settle.",
        "decoded_frame_count": 40,
        "full_decode_to_eof": True,
        "ordered_contact_sheet_review_performed": True,
        "emotion_judgment_performed": False,
        "training_admission": False,
    }
    for phase, status, frame in zip(("onset", "apex", "offset"), phases, (3, 20, 38)):
        review_row[f"{phase}_status"] = status
        review_row[f"{phase}_evidence_frame"] = frame
        review_row[f"{phase}_basis"] = f"{phase}_visual_basis"
    _jsonl(review, [review_row])
    review_summary = root / "review.summary.json"
    distributions = {
        "action_result_distribution": {"pass": 1},
        "action_observability_distribution": {"observable": 1},
        "onset_status_distribution": {phases[0]: 1},
        "apex_status_distribution": {phases[1]: 1},
        "offset_status_distribution": {phases[2]: 1},
    }
    _json(
        review_summary,
        {
            "artifact_kind": REVIEW_SUMMARY_KIND,
            "fixed_duration_window_used": False,
            "elapsed_duration_used_as_gate": False,
            "native_variable_length_reviewed": True,
            "validation_passed": True,
            "records": 1,
            "decoded_frame_count_total": 40,
            "submission_path": str(review.resolve()),
            "submission_sha256": sha256_file(review),
            "public_summary_path": str(public_summary.resolve()),
            "public_summary_sha256": sha256_file(public_summary),
            "public_queue_path": str(queue.resolve()),
            "public_queue_sha256": sha256_file(queue),
            **distributions,
        },
    )
    return {
        "public_summary": public_summary,
        "arc_review_submission": review,
        "arc_review_summary": review_summary,
        "expansion_hidden_mapping": mapping,
        "expansion_hidden_summary": hidden_summary,
        "base_hidden_mapping": base_mapping,
        "previous_plan_summary": previous_plan,
        "previous_expansion_requests": previous_requests,
        "candidate_catalog": catalog,
        "catalog_summary": catalog_summary,
        "output_root": root / "out",
    }


def _build(case):
    return build_plan(**case)


@pytest.mark.parametrize(
    ("phases", "final_level", "expected"),
    [
        (("complete", "complete", "complete"), 2, (0, 1, 0)),
        (("complete", "complete", "incomplete"), 2, (1, 0, 0)),
        (("incomplete", "complete", "ambiguous"), 1, (0, 0, 1)),
    ],
)
def test_complete_expand_and_exhaust_are_content_driven(tmp_path, phases, final_level, expected):
    summary = _build(_case(tmp_path, phases=phases, final_level=final_level))
    outputs = summary["outputs"]
    assert (
        outputs["expansion_requests"]["records"],
        outputs["complete_current_context"]["records"],
        outputs["context_exhausted"]["records"],
    ) == expected
    assert summary["fixed_minimum_maximum_or_target_duration_used"] is False
    assert summary["elapsed_duration_used_as_gate"] is False
    assert summary["accepted_for_training_count"] == 0


def test_completed_all_stage_pipeline_is_accepted(tmp_path):
    summary = _build(_case(tmp_path, pipeline_stage="all"))
    assert summary["outputs"]["expansion_requests"]["records"] == 1


def test_request_advances_exactly_one_level_and_keeps_runner_kind(tmp_path):
    case = _case(tmp_path)
    _build(case)
    request = json.loads((case["output_root"] / "expansion_requests.jsonl").read_text())
    assert request["artifact_kind"] == REQUEST_KIND
    assert request["requested_context_level"] == request["reviewed_context_level"] + 1
    assert request["requested_interval"] == {
        "start_frame": 0,
        "end_frame_exclusive": 60,
        "frame_count": 60,
    }
    assert request["continuation_lineage"]["role"] == "round_n_natural_boundary_continuation"
    assert request["accepted_for_training"] is False


def test_ordered_full_video_review_is_accepted_without_contact_sheet_claim(tmp_path):
    case = _case(tmp_path)
    row = json.loads(case["arc_review_submission"].read_text())
    row.pop("ordered_contact_sheet_review_performed")
    row["ordered_full_video_review_performed"] = True
    _jsonl(case["arc_review_submission"], [row])
    summary = json.loads(case["arc_review_summary"].read_text())
    summary["submission_sha256"] = sha256_file(case["arc_review_submission"])
    _json(case["arc_review_summary"], summary)
    result = _build(case)
    assert result["outputs"]["expansion_requests"]["records"] == 1


def test_rejects_previous_plan_level_jump(tmp_path):
    case = _case(tmp_path, previous_requested_level=2)
    with pytest.raises(ValueError, match="previous natural-context request"):
        _build(case)


def test_rejects_anonymous_mapping_video_substitution(tmp_path):
    case = _case(tmp_path)
    row = json.loads(case["expansion_hidden_mapping"].read_text())
    row["video_sha256"] = "f" * 64
    _jsonl(case["expansion_hidden_mapping"], [row])
    hidden = json.loads(case["expansion_hidden_summary"].read_text())
    hidden["mapping_sha256"] = sha256_file(case["expansion_hidden_mapping"])
    _json(case["expansion_hidden_summary"], hidden)
    with pytest.raises(ValueError, match="lineage mismatch"):
        _build(case)


def test_rejects_anonymous_mapping_trajectory_substitution(tmp_path):
    case = _case(tmp_path)
    row = json.loads(case["expansion_hidden_mapping"].read_text())
    row["trajectory_sha256"] = "e" * 64
    _jsonl(case["expansion_hidden_mapping"], [row])
    hidden = json.loads(case["expansion_hidden_summary"].read_text())
    hidden["mapping_sha256"] = sha256_file(case["expansion_hidden_mapping"])
    _json(case["expansion_hidden_summary"], hidden)
    with pytest.raises(ValueError, match="trajectory lineage mismatch"):
        _build(case)


def test_rejects_review_record_tamper_even_when_summary_is_rebound(tmp_path):
    case = _case(tmp_path)
    row = json.loads(case["arc_review_submission"].read_text())
    row["action_result"] = "mismatch"
    _jsonl(case["arc_review_submission"], [row])
    summary = json.loads(case["arc_review_summary"].read_text())
    summary["submission_sha256"] = sha256_file(case["arc_review_submission"])
    summary["action_result_distribution"] = {"mismatch": 1}
    _json(case["arc_review_summary"], summary)
    with pytest.raises(ValueError, match="Arc/action review mismatch"):
        _build(case)


def test_rejects_review_summary_hash_tamper(tmp_path):
    case = _case(tmp_path)
    summary = json.loads(case["arc_review_summary"].read_text())
    summary["submission_sha256"] = "0" * 64
    _json(case["arc_review_summary"], summary)
    with pytest.raises(ValueError, match="hash binding mismatch"):
        _build(case)


def test_rejects_previous_plan_record_hash_tamper(tmp_path):
    case = _case(tmp_path)
    row = json.loads(case["previous_expansion_requests"].read_text())
    row["plan_record_sha256"] = "0" * 64
    _jsonl(case["previous_expansion_requests"], [row])
    plan = json.loads(case["previous_plan_summary"].read_text())
    plan["outputs"]["expansion_requests"]["sha256"] = sha256_file(
        case["previous_expansion_requests"]
    )
    _json(case["previous_plan_summary"], plan)
    with pytest.raises(ValueError, match="record SHA mismatch"):
        _build(case)


def test_rejects_fixed_duration_pollution(tmp_path):
    case = _case(tmp_path)
    public = json.loads(case["public_summary"].read_text())
    public["fixed_duration_window_used"] = True
    _json(case["public_summary"], public)
    with pytest.raises(ValueError, match="variable-length policy"):
        _build(case)


def test_rejects_elapsed_duration_gate(tmp_path):
    case = _case(tmp_path)
    summary = json.loads(case["arc_review_summary"].read_text())
    summary["elapsed_duration_used_as_gate"] = True
    _json(case["arc_review_summary"], summary)
    with pytest.raises(ValueError, match="review summary violates contract"):
        _build(case)


def test_rejects_forged_displayed_interval(tmp_path):
    case = _case(tmp_path)
    row = json.loads(case["expansion_hidden_mapping"].read_text())
    row["displayed_interval"] = {
        "start_frame": 11,
        "end_frame_exclusive": 50,
        "frame_count": 39,
    }
    _jsonl(case["expansion_hidden_mapping"], [row])
    hidden = json.loads(case["expansion_hidden_summary"].read_text())
    hidden["mapping_sha256"] = sha256_file(case["expansion_hidden_mapping"])
    _json(case["expansion_hidden_summary"], hidden)
    with pytest.raises(ValueError, match="mapping/candidate lineage mismatch"):
        _build(case)


def test_rejects_catalog_context_that_crosses_or_fails_to_contain_parent(tmp_path):
    case = _case(tmp_path, malformed_level2=True)
    with pytest.raises(ValueError, match="strictly follow its parent"):
        _build(case)
