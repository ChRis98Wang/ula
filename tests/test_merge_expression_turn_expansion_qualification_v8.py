from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.human_motion_review import (
    merge_expression_turn_expansion_qualification_v8 as MERGE,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(MERGE.stable_json(row) + "\n" for row in rows))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _fixture(tmp_path: Path, *, pipeline_stage: str = "render") -> dict[str, Path]:
    public = tmp_path / "public"
    physical = tmp_path / "physical"
    private = tmp_path / "private"
    reviews = tmp_path / "reviews"
    continuation = private / "continuation"
    sample_specs = [
        ("anon_complete_observable", "complete", "observable", "happy", 0.8),
        ("anon_complete_ambiguous", "complete", "ambiguous", None, None),
        # Affect is deliberately observable here.  The merger must validate it
        # for file integrity but must not adjudicate/copy it before arc completion.
        ("anon_request", "request", "observable", "surprise", 0.9),
        ("anon_exhausted", "exhausted", "not_observable", None, None),
    ]

    arc_queue_rows: list[dict] = []
    affect_queue_rows: list[dict] = []
    arc_review_rows: list[dict] = []
    affect_review_rows: list[dict] = []
    physical_rows: list[dict] = []
    mapping_rows: list[dict] = []
    for index, (sample_id, decision, affect_result, affect_class, confidence) in enumerate(
        sample_specs
    ):
        task_id = f"task_{index}__ctxL01"
        video = public / "videos" / f"{sample_id}.mp4"
        trajectory = physical / "trajectories" / f"{task_id}.csv"
        video.parent.mkdir(parents=True, exist_ok=True)
        trajectory.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(f"video-{sample_id}".encode())
        trajectory.write_bytes(f"trajectory-{sample_id}".encode())
        video_sha = MERGE.sha256_file(video)
        trajectory_sha = MERGE.sha256_file(trajectory)
        common = {
            "schema_version": MERGE.SCHEMA_VERSION,
            "sample_id": sample_id,
            "video_path": str(video),
            "video_sha256": video_sha,
            "context_level": 1,
            "frame_count": 5,
            "fps": 30,
            "audio_available": False,
            "label_metadata_exposed": False,
            "native_duration_preserved": True,
            "fixed_duration_window_used": False,
        }
        arc_queue_rows.append(
            {
                **common,
                "arc_protocol_version": MERGE.ARC_PROTOCOL,
                "action_protocol_version": MERGE.ACTION_PROTOCOL,
            }
        )
        affect_queue_rows.append(
            {
                **common,
                "affect_protocol_version": MERGE.AFFECT_PROTOCOL,
                "allowed_classes": list(MERGE.AFFECT_CLASSES),
            }
        )

        complete = decision == "complete"
        statuses = {
            "onset": "complete",
            "apex": "complete",
            "offset": "complete" if complete else "incomplete",
        }
        arc_review_rows.append(
            {
                **common,
                "arc_protocol_version": MERGE.ARC_PROTOCOL,
                "action_protocol_version": MERGE.ACTION_PROTOCOL,
                "arc_review_id": f"arc-{sample_id}",
                "action_review_id": f"action-{sample_id}",
                "arc_reviewer_id": "arc-reviewer",
                "action_reviewer_id": "arc-reviewer",
                "queue_sha256": None,
                "full_decode_to_eof": True,
                "decoded_frame_count": 5,
                "emotion_judgment_performed": False,
                "training_admission": False,
                "action_result": "pass",
                "action_observability": "observable",
                "observable_description": f"Observable action {index}.",
                "onset_status": statuses["onset"],
                "onset_evidence_frame": 0,
                "onset_basis": "coherent_motion_entry",
                "apex_status": statuses["apex"],
                "apex_evidence_frame": 2,
                "apex_basis": "distinct_motion_or_pose_peak",
                "offset_status": statuses["offset"],
                "offset_evidence_frame": 4,
                "offset_basis": (
                    "natural_settle_or_stable_transition"
                    if complete
                    else "video_ends_before_natural_settle"
                ),
            }
        )
        affect_review_rows.append(
            {
                "schema_version": MERGE.SCHEMA_VERSION,
                "affect_protocol_version": MERGE.AFFECT_PROTOCOL,
                "affect_review_id": f"affect-{sample_id}",
                "affect_reviewer_id": "affect-reviewer",
                "sample_id": sample_id,
                "result": affect_result,
                "predicted_class": affect_class,
                "confidence": confidence,
                "allowed_classes": list(MERGE.AFFECT_CLASSES),
                "public_queue_sha256": None,
                "video_sha256": video_sha,
                "decoded_frame_count": 5,
                "full_video_reviewed": True,
                "training_admission": False,
            }
        )
        gates = {gate: True for gate in MERGE.REQUIRED_PHYSICAL_GATES}
        retarget_segment = {
            "source_frame_count": 5,
            "output_frame_count": 5,
            "cropped": False,
            "fixed_target_duration_sec": None,
        }
        safety = {
            "artifact_kind": "ula_18d_safety_monotonic_retime_v1",
            "output_frame_count": 5,
            "post_velocity_pass": True,
            "slowdown_ratio_pass": True,
            "time_map_strictly_increasing": True,
            "first_frame_preserved": True,
            "last_frame_preserved": True,
            "cropped": False,
            "tiled": False,
            "target_duration_sec": None,
        }
        quality_path = physical / "quality" / task_id / "quality.json"
        quality = {
            "schema_version": MERGE.SCHEMA_VERSION,
            "artifact_kind": MERGE.QUALITY_KIND,
            "accepted_for_training": False,
            "license_training_admission": False,
            "semantic_admission": False,
            "affect_admission": False,
            "physical_quality_only": True,
            "processing_scope": MERGE.PROCESSING_SCOPE,
            "action_dim": 18,
            "frames": 5,
            "fps": 30,
            "representation": "native_variable_length_expression_turn_v1",
            "output_contract": "ula_v2_18d_head_v1",
            "safe_csv_sha256": trajectory_sha,
            "outputs": {"safe_csv": str(trajectory)},
            "quality_gate": gates,
            "retarget_segment": retarget_segment,
            "safety_monotonic_retime": safety,
        }
        _write_json(quality_path, quality)
        result_path = physical / "results" / f"{task_id}.json"
        result = {
            "schema_version": MERGE.SCHEMA_VERSION,
            "artifact_kind": MERGE.RETARGET_RESULT_KIND,
            "status": "passed",
            "accepted_for_training": False,
            "quality_gate": gates,
            "quality_json": str(quality_path),
            "quality_json_sha256": MERGE.sha256_file(quality_path),
            "safe_csv": str(trajectory),
            "safe_csv_sha256": trajectory_sha,
            "frames": 5,
            "retarget_segment": retarget_segment,
        }
        _write_json(result_path, result)
        physical_row = {
            "schema_version": MERGE.SCHEMA_VERSION,
            "task_id": task_id,
            "status": "passed",
            "accepted_for_training": False,
            "license_training_admission": False,
            "render_pass_grants_training_admission": False,
            "processing_scope": MERGE.PROCESSING_SCOPE,
            "video_path": str(video),
            "video_sha256": video_sha,
            "trajectory_path": str(trajectory),
            "trajectory_sha256": trajectory_sha,
            "trajectory_frames": 5,
            "quality_gate": gates,
            "retarget_result_json": str(result_path),
            "retarget_result_json_sha256": MERGE.sha256_file(result_path),
            "training_segment": {
                "start_frame": 10,
                "end_frame_exclusive": 15,
                "frame_count": 5,
                "cropped": False,
                "fixed_window_sec": None,
            },
            "retarget_segment": retarget_segment,
        }
        physical_rows.append(physical_row)
        mapping_rows.append(
            {
                "schema_version": MERGE.SCHEMA_VERSION,
                "sample_id": sample_id,
                "task_id": task_id,
                "base_task_id": f"task_{index}",
                "accepted_for_training": False,
                "native_duration_preserved": True,
                "fixed_duration_window_used": False,
                "official_action_text_or_affect_exposed": False,
                "displayed_context_level": 1,
                "frame_count": 5,
                "fixed_split_assignment": "train",
                "video_sha256": video_sha,
                "trajectory_sha256": trajectory_sha,
                "source_render_record_sha256": MERGE.value_sha256(physical_row),
            }
        )

    arc_queue = public / "arc_action_review_queue.jsonl"
    affect_queue = public / "affect_review_queue.jsonl"
    _write_jsonl(arc_queue, arc_queue_rows)
    _write_jsonl(affect_queue, affect_queue_rows)
    public_summary = public / "summary.json"
    _write_json(
        public_summary,
        {
            "schema_version": MERGE.SCHEMA_VERSION,
            "artifact_kind": MERGE.PUBLIC_KIND,
            "accepted_for_training": False,
            "duration_policy": MERGE.PUBLIC_DURATION_POLICY,
            "all_samples_native_variable_length": True,
            "fixed_duration_window_used": False,
            "same_anonymous_silent_video_used_by_both_reviews": True,
            "source_identity_official_action_text_and_affect_exposed": False,
            "records": len(sample_specs),
            "arc_action_queue": str(arc_queue),
            "arc_action_queue_sha256": MERGE.sha256_file(arc_queue),
            "affect_queue": str(affect_queue),
            "affect_queue_sha256": MERGE.sha256_file(affect_queue),
        },
    )

    for row in arc_review_rows:
        row["queue_sha256"] = MERGE.sha256_file(arc_queue)
    for row in affect_review_rows:
        row["public_queue_sha256"] = MERGE.sha256_file(affect_queue)
    arc_submission = reviews / "arc.jsonl"
    affect_submission = reviews / "affect.jsonl"
    _write_jsonl(arc_submission, arc_review_rows)
    _write_jsonl(affect_submission, affect_review_rows)
    arc_summary = reviews / "arc.summary.json"
    _write_json(
        arc_summary,
        {
            "schema_version": MERGE.SCHEMA_VERSION,
            "artifact_kind": MERGE.ARC_SUMMARY_KIND,
            "records": len(sample_specs),
            "submission_path": str(arc_submission),
            "submission_sha256": MERGE.sha256_file(arc_submission),
            "public_summary_path": str(public_summary),
            "public_summary_sha256": MERGE.sha256_file(public_summary),
            "public_queue_path": str(arc_queue),
            "public_queue_sha256": MERGE.sha256_file(arc_queue),
            "validation_passed": True,
            "fixed_duration_window_used": False,
            "elapsed_duration_used_as_gate": False,
            "native_variable_length_reviewed": True,
        },
    )
    affect_summary = reviews / "affect.summary.json"
    result_counts = {key: 0 for key in ("observable", "ambiguous", "not_observable")}
    for row in affect_review_rows:
        result_counts[row["result"]] += 1
    _write_json(
        affect_summary,
        {
            "schema_version": MERGE.SCHEMA_VERSION,
            "artifact_kind": MERGE.AFFECT_SUMMARY_KIND,
            "training_admission": False,
            "submission_path": str(affect_submission),
            "submission_sha256": MERGE.sha256_file(affect_submission),
            "public_queue_path": str(affect_queue),
            "public_queue_sha256": MERGE.sha256_file(affect_queue),
            "coverage": {
                "expected_records": len(sample_specs),
                "reviewed_records": len(sample_specs),
                "complete": True,
            },
            "integrity": {
                "all_full_video_reviewed": True,
                "all_native_variable_length_reviewed": True,
                "fixed_duration_window_used": False,
            },
            "result_distribution": result_counts,
        },
    )

    passed_manifest = physical / "passed.jsonl"
    _write_jsonl(passed_manifest, physical_rows)
    audit = {
        "artifact_kind": "test_input_audit",
        "accepted_for_training": False,
        "fixed_minimum_maximum_or_target_duration_used": False,
    }
    audit["sha256"] = MERGE.value_sha256(audit)
    pipeline_summary = physical / "pipeline_summary.json"
    _write_json(
        pipeline_summary,
        {
            "schema_version": MERGE.SCHEMA_VERSION,
            "artifact_kind": MERGE.PIPELINE_KIND,
            "stage": pipeline_stage,
            "accepted_for_training": False,
            "license_training_admission": False,
            "semantic_admission": False,
            "affect_admission": False,
            "fixed_six_second_windows_used": False,
            "elapsed_seconds_used_for_selection": False,
            "input_audit": audit,
            "render_summary": {
                "passed_manifest": str(passed_manifest),
                "passed_manifest_sha256": MERGE.sha256_file(passed_manifest),
                "counts": {"passed": len(sample_specs), "failed": 0},
                "render_pass_grants_training_admission": False,
            },
        },
    )
    hidden_mapping = private / "mapping.jsonl"
    _write_jsonl(hidden_mapping, mapping_rows)
    hidden_summary = private / "summary.json"
    _write_json(
        hidden_summary,
        {
            "schema_version": MERGE.SCHEMA_VERSION,
            "artifact_kind": MERGE.HIDDEN_KIND,
            "accepted_for_training": False,
            "public_distribution_forbidden": True,
            "records": len(sample_specs),
            "mapping": str(hidden_mapping),
            "mapping_sha256": MERGE.sha256_file(hidden_mapping),
            "pipeline_summary": str(pipeline_summary),
            "pipeline_summary_sha256": MERGE.sha256_file(pipeline_summary),
            "render_passed_manifest": str(passed_manifest),
            "render_passed_manifest_sha256": MERGE.sha256_file(passed_manifest),
        },
    )

    queue_bound = MERGE.index_bound(
        MERGE.read_bound_jsonl(arc_queue), "sample_id", context="test queue"
    )
    mapping_bound = MERGE.index_bound(
        MERGE.read_bound_jsonl(hidden_mapping), "sample_id", context="test mapping"
    )
    arc_by_id = {row["sample_id"]: row for row in arc_review_rows}
    complete_rows: list[dict] = []
    request_rows: list[dict] = []
    exhausted_rows: list[dict] = []
    for index, (sample_id, decision, *_rest) in enumerate(sample_specs):
        arc = arc_by_id[sample_id]
        mapping = mapping_bound[sample_id][0]
        statuses = {phase: arc[f"{phase}_status"] for phase in MERGE.PHASES}
        common = {
            "schema_version": MERGE.SCHEMA_VERSION,
            "sample_id": f"canonical_{index}",
            "reviewed_anonymous_sample_id": sample_id,
            "accepted_for_training": False,
            "semantic_supervision_mask": False,
            "emotion_supervision_mask": False,
            "elapsed_duration_used_as_gate": False,
            "reviewed_context_level": 1,
            "reviewed_interval": {
                "start_frame": 10,
                "end_frame_exclusive": 15,
                "frame_count": 5,
            },
            "review_phase_statuses": statuses,
            "review_evidence_frames": [
                arc["onset_evidence_frame"],
                arc["apex_evidence_frame"],
                arc["offset_evidence_frame"],
            ],
            "reviewed_video_sha256": mapping["video_sha256"],
            "reviewed_trajectory_sha256": mapping["trajectory_sha256"],
            "reviewed_output_frame_count": 5,
            "continuation_lineage": {
                "source_public_summary_sha256": MERGE.sha256_file(public_summary),
                "source_arc_action_queue_sha256": MERGE.sha256_file(arc_queue),
                "review_submission_sha256": MERGE.sha256_file(arc_submission),
                "review_summary_sha256": MERGE.sha256_file(arc_summary),
                "review_record_sha256": MERGE.value_sha256(arc),
                "review_queue_record_sha256": queue_bound[sample_id][1],
                "expansion_hidden_mapping_sha256": MERGE.sha256_file(hidden_mapping),
                "expansion_hidden_mapping_record_sha256": mapping_bound[sample_id][1],
                "expansion_hidden_summary_sha256": MERGE.sha256_file(hidden_summary),
                "pipeline_summary_sha256": MERGE.sha256_file(pipeline_summary),
                "render_passed_manifest_sha256": MERGE.sha256_file(passed_manifest),
                "source_render_record_sha256": mapping["source_render_record_sha256"],
            },
        }
        if decision == "complete":
            row = {
                **common,
                "artifact_kind": MERGE.COMPLETE_KIND,
                "review_qualification_status": "complete_arc_action_candidate",
            }
            complete_rows.append(row)
        elif decision == "request":
            row = {
                **common,
                "artifact_kind": MERGE.REQUEST_KIND,
                "review_qualification_status": "natural_context_expansion_required",
                "requested_context_level": 2,
                "requested_interval": {
                    "start_frame": 9,
                    "end_frame_exclusive": 16,
                    "frame_count": 7,
                },
            }
            request_rows.append(row)
        else:
            row = {
                **common,
                "artifact_kind": MERGE.EXHAUSTED_KIND,
                "review_qualification_status": "natural_context_expansion_required",
                "context_exhausted_at_level": 1,
            }
            exhausted_rows.append(row)
        row["plan_record_sha256"] = MERGE.value_sha256(row)

    complete_path = continuation / "complete.jsonl"
    request_path = continuation / "requests.jsonl"
    exhausted_path = continuation / "exhausted.jsonl"
    _write_jsonl(complete_path, complete_rows)
    _write_jsonl(request_path, request_rows)
    _write_jsonl(exhausted_path, exhausted_rows)
    continuation_summary = continuation / "summary.json"
    _write_json(
        continuation_summary,
        {
            "schema_version": MERGE.SCHEMA_VERSION,
            "artifact_kind": MERGE.CONTINUATION_KIND,
            "accepted_for_training_count": 0,
            "fixed_minimum_maximum_or_target_duration_used": False,
            "elapsed_duration_used_as_gate": False,
            "same_source_only": True,
            "neighbor_crossing_allowed": False,
            "one_level_only": True,
            "selection_policy": MERGE.SELECTION_POLICY,
            "inputs": {
                "public_summary_sha256": MERGE.sha256_file(public_summary),
                "arc_action_queue_sha256": MERGE.sha256_file(arc_queue),
                "arc_review_submission_sha256": MERGE.sha256_file(arc_submission),
                "arc_review_summary_sha256": MERGE.sha256_file(arc_summary),
                "expansion_hidden_mapping_sha256": MERGE.sha256_file(hidden_mapping),
                "expansion_hidden_summary_sha256": MERGE.sha256_file(hidden_summary),
                "pipeline_summary_sha256": MERGE.sha256_file(pipeline_summary),
                "render_passed_manifest_sha256": MERGE.sha256_file(passed_manifest),
            },
            "outputs": {
                "complete_current_context": {
                    "path": str(complete_path),
                    "records": len(complete_rows),
                    "sha256": MERGE.sha256_file(complete_path),
                },
                "expansion_requests": {
                    "path": str(request_path),
                    "records": len(request_rows),
                    "sha256": MERGE.sha256_file(request_path),
                },
                "context_exhausted": {
                    "path": str(exhausted_path),
                    "records": len(exhausted_rows),
                    "sha256": MERGE.sha256_file(exhausted_path),
                },
            },
        },
    )
    return {
        "public_summary": public_summary,
        "hidden_summary": hidden_summary,
        "hidden_mapping": hidden_mapping,
        "physical_pipeline_summary": pipeline_summary,
        "physical_passed_manifest": passed_manifest,
        "arc_review_submission": arc_submission,
        "arc_review_summary": arc_summary,
        "affect_review_submission": affect_submission,
        "affect_review_summary": affect_summary,
        "continuation_summary": continuation_summary,
        "output_root": tmp_path / "output",
        "request_path": request_path,
    }


def test_completed_all_stage_pipeline_is_accepted(tmp_path):
    fixture = _fixture(tmp_path / "fixture", pipeline_stage="all")
    result = MERGE.merge_qualification(
        **{key: value for key, value in fixture.items() if key != "request_path"}
    )
    assert result["counts"]["train_candidate"] == 2


def test_three_tiers_are_closed_and_affect_is_gated_by_complete_arc(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    summary = MERGE.merge_qualification(**{k: v for k, v in paths.items() if k != "request_path"})

    assert summary["counts"] == {
        "input_samples": 4,
        "train_candidate": 2,
        "needs_review_or_expansion": 1,
        "reject": 1,
        "affect_evaluated": 2,
        "affect_intentionally_not_evaluated": 2,
        "emotion_supervision_candidate": 1,
        "train_candidate_without_emotion_supervision": 1,
        "semantic_text_supervision_candidate": 0,
        "accepted_for_training": 0,
    }
    candidates = _read_jsonl(paths["output_root"] / "train_candidate.jsonl")
    assert [row["emotion_class"] for row in candidates] == [None, "happy"]
    assert [row["emotion_supervision_mask"] for row in candidates] == [False, True]
    needs = _read_jsonl(paths["output_root"] / "needs_review_or_expansion.jsonl")
    assert needs[0]["sample_id"] == "anon_request"
    assert needs[0]["affect_review_evaluated"] is False
    assert needs[0]["affect_result"] is None
    assert needs[0]["emotion_class"] is None
    all_rows = candidates + needs + _read_jsonl(paths["output_root"] / "reject.jsonl")
    assert all(row["accepted_for_training"] is False for row in all_rows)
    assert all(row["license_training_admission"] is False for row in all_rows)


def test_continuation_output_sha_tamper_fails_closed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["request_path"].write_text(paths["request_path"].read_text() + " \n")
    with pytest.raises(ValueError, match="Continuation output SHA mismatch"):
        MERGE.merge_qualification(**{k: v for k, v in paths.items() if k != "request_path"})


def test_ambiguous_affect_cannot_carry_pseudo_label(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    rows = _read_jsonl(paths["affect_review_submission"])
    target = next(row for row in rows if row["sample_id"] == "anon_complete_ambiguous")
    target["predicted_class"] = "happy"
    target["confidence"] = 0.9
    _write_jsonl(paths["affect_review_submission"], rows)
    summary = json.loads(paths["affect_review_summary"].read_text())
    summary["submission_sha256"] = MERGE.sha256_file(paths["affect_review_submission"])
    _write_json(paths["affect_review_summary"], summary)
    with pytest.raises(ValueError, match="non-observable affect carries a pseudo-label"):
        MERGE.merge_qualification(**{k: v for k, v in paths.items() if k != "request_path"})


def test_continuation_partition_cannot_omit_sample(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _write_jsonl(paths["request_path"], [])
    summary = json.loads(paths["continuation_summary"].read_text())
    binding = summary["outputs"]["expansion_requests"]
    binding["records"] = 0
    binding["sha256"] = MERGE.sha256_file(paths["request_path"])
    _write_json(paths["continuation_summary"], summary)
    with pytest.raises(ValueError, match="do not exactly cover"):
        MERGE.merge_qualification(**{k: v for k, v in paths.items() if k != "request_path"})
