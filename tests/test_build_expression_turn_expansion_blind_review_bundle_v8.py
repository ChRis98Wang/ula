import json
import os

import imageio.v2 as imageio
import numpy as np
import pytest

from tools.gmr_v2.safety_monotonic_retime_v1 import (
    minimum_velocity_safety_retime,
)
from tools.human_motion_review import build_expression_turn_blind_review_bundle_v8 as blind
from tools.human_motion_review import build_expression_turn_video_queue_v8 as video_queue
from tools.human_motion_review import render_beat2_annotation_review as renderer
from tools.human_motion_review.expression_turn_retarget_contract import (
    REQUIRED_18D_GATES,
)
from tools.human_motion_review.build_expression_turn_expansion_blind_review_bundle_v8 import (
    AFFECT_PUBLIC_KEYS,
    ARC_ACTION_PUBLIC_KEYS,
    EXPANSION_UNIT,
    EXPANSION_RENDER_AUDIT_FIELDS,
    INPUT_AUDIT_KIND,
    PIPELINE_KIND,
    PLAN_KIND,
    PROCESSING_SCOPE,
    PROVENANCE_KIND,
    QUEUE_KIND,
    REQUEST_KIND,
    SAMPLED_FIELDS_NOT_IN_RENDER_RESULT,
    SELECTION_POLICY,
    _index_requests,
    build_bundle,
)
from upper_body_skeleton.retarget_v2_18d import JOINT_LIMITS_18D


def _write_jsonl(path, rows):
    path.write_text(
        "".join(blind.stable_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _request(sample_id, base_task_id, start=10, end=25):
    record = {
        "schema_version": "1.0.0",
        "artifact_kind": REQUEST_KIND,
        "sample_id": sample_id,
        "base_task_id": base_task_id,
        "source_clip_id": f"source-{sample_id}",
        "fixed_split_assignment": "train",
        "base_expression_turn_record_sha256": "1" * 64,
        "comparison_record_sha256": "2" * 64,
        "review_qualification_status": "natural_context_expansion_required",
        "reviewed_context_level": 0,
        "reviewed_interval": {
            "start_frame": start + 2,
            "end_frame_exclusive": end - 2,
            "frame_count": end - start - 4,
        },
        "elapsed_duration_used_as_gate": False,
        "semantic_supervision_mask": False,
        "emotion_supervision_mask": False,
        "accepted_for_training": False,
        "requested_context_level": 1,
        "requested_interval": {
            "start_frame": start,
            "end_frame_exclusive": end,
            "frame_count": end - start,
        },
        "strictly_contains_reviewed_interval": True,
        "expansion_unit": EXPANSION_UNIT,
        "next_action": "retarget_render_and_repeat_independent_blind_arc_action_review",
    }
    record["plan_record_sha256"] = blind.value_sha256(record)
    return record


def _render_record(tmp_path, request):
    video = tmp_path / f"{request['sample_id']}.mp4"
    source_frames = request["requested_interval"]["frame_count"]
    raw = np.zeros((source_frames, len(renderer.JOINT_ORDER)), dtype=np.float64)
    raw[source_frames // 2, 0] = 0.11
    safe, _input_times, _output_times, safety_retime = (
        minimum_velocity_safety_retime(
            raw,
            fps=renderer.FPS,
            max_velocity_rad_s=3.0,
            smoothing_window=3,
            joint_order=renderer.JOINT_ORDER,
            joint_limits=JOINT_LIMITS_18D,
        )
    )
    frames = len(safe)
    video_frames = []
    for index in range(frames):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        offset = 2 + index % 24
        frame[8:32, offset : offset + 18] = (40, 160, 240)
        video_frames.append(frame)
    imageio.mimwrite(
        video,
        video_frames,
        fps=renderer.FPS,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=2,
        output_params=["-movflags", "+faststart"],
    )
    trajectory = tmp_path / f"{request['sample_id']}.csv"
    trajectory.write_text(
        ",".join(renderer.JOINT_ORDER)
        + "\n"
        + "".join(
            ",".join(f"{value:.8f}" for value in row)
            + "\n"
            for row in safe
        ),
        encoding="utf-8",
    )
    video_check = renderer.validate_video(
        video,
        expected_frames=frames,
        expected_width=64,
        expected_height=48,
    )
    training_segment = {
        **request["requested_interval"],
        "fixed_window_sec": None,
        "cropped": False,
        "duration_policy": video_queue.NATURAL_DURATION_POLICY,
        "representation": video_queue.INPUT_REPRESENTATION,
    }
    retarget_segment = {
        "source_frame_count": source_frames,
        "output_frame_count": frames,
        "source_frame_coverage_sec": source_frames / 30.0,
        "output_frame_coverage_sec": frames / 30.0,
        "output_sample_span_sec": (frames - 1) / 30.0,
        "fixed_target_duration_sec": None,
        "cropped": False,
        "duration_policy": video_queue.NATURAL_DURATION_POLICY,
        "representation": video_queue.RETARGET_SEGMENT_REPRESENTATION,
        "retimed": True,
    }
    quality_gate = {
        **{name: True for name in REQUIRED_18D_GATES},
        "safety_retime_passed": True,
    }
    quality = {
        "quality_gate": quality_gate,
        "expression_turn_output_contract_validation": {"passed": True},
        "training_segment": training_segment,
        "retarget_segment": retarget_segment,
        "safety_monotonic_retime": safety_retime,
    }
    quality_path = tmp_path / f"{request['sample_id']}.quality.json"
    quality_path.write_text(json.dumps(quality, sort_keys=True), encoding="utf-8")
    binding = {
        "trajectory_path": str(trajectory.resolve()),
        "trajectory_sha256": blind.sha256(trajectory),
        "output_frame_count": frames,
        "fps": 30.0,
        "video_path": str(video.resolve()),
        "video_sha256": blind.sha256(video),
        "video_decoded_frames": frames,
    }
    binding["sha256"] = blind.value_sha256(binding)
    provenance = {
        "schema_version": "1.0.0",
        "artifact_kind": PROVENANCE_KIND,
        "sample_id": request["sample_id"],
        "base_task_id": request["base_task_id"],
        "base_expression_turn_record_sha256": request[
            "base_expression_turn_record_sha256"
        ],
        "base_candidate_catalog_sha256": "3" * 64,
        "expansion_plan_summary_sha256": None,
        "expansion_requests_sha256": None,
        "hidden_mapping_sha256": "4" * 64,
        "plan_record_sha256": request["plan_record_sha256"],
        "comparison_record_sha256": request["comparison_record_sha256"],
        "reviewed_context_level": 0,
        "requested_context_level": 1,
        "selection_policy": SELECTION_POLICY,
        "expansion_unit": EXPANSION_UNIT,
        "elapsed_duration_used_as_gate": False,
        "physical_quality_only": True,
        "semantic_admission": False,
        "affect_admission": False,
        "license_training_admission": False,
        "accepted_for_training": False,
    }
    return {
        "schema_version": "1.0.0",
        "task_id": f"{request['base_task_id']}__ctxL01",
        "source_clip_id": request["source_clip_id"],
        "speaker_key": "speaker-a",
        "official_split": "train",
        "fixed_split_assignment": "train",
        "robot_contract": renderer.ROBOT_CONTRACT,
        "canonical_action": None,
        "canonical_action_role": "disabled_pending_fresh_independent_blind_review",
        "canonical_prompt": {
            "en": "Anonymous silent robot motion sample.",
            "zh": "Anonymous silent robot motion sample.",
        },
        "canonical_prompt_role": "anonymous_renderer_placeholder_not_conditioning_text",
        "status": "passed",
        "accepted_for_training": False,
        "render_pass_grants_training_admission": False,
        "emotion_supervision_mask": False,
        "official_emotion_conditioning_enabled": False,
        "affect_observable_supervision_mask": False,
        "official_category_conditioning_enabled": False,
        "manual_review_required": True,
        "speech_context_included": False,
        "semantic_action_completeness_review_required": True,
        "affect_observable_review_required": True,
        "blind_review_must_use_final_trajectory": True,
        "context_plan": {"selected_level": 1},
        "training_segment": training_segment,
        "trajectory_path": str(trajectory.resolve()),
        "trajectory_sha256": blind.sha256(trajectory),
        "trajectory_frames": frames,
        "trajectory_frames_expected": frames,
        "output_frame_count": frames,
        "source_frame_count": source_frames,
        "final_trajectory_role": "safety_monotonic_retimed_final_output",
        "retarget_segment": retarget_segment,
        "safety_monotonic_retime": safety_retime,
        "quality_json": str(quality_path.resolve()),
        "quality_json_sha256": blind.sha256(quality_path),
        "quality_gate": quality_gate,
        "video_path": str(video.resolve()),
        "video_sha256": blind.sha256(video),
        "video_check": video_check,
        "final_output_binding": binding,
        "processing_scope": PROCESSING_SCOPE,
        "license_training_admission": False,
        "license_training_admission_status": "not_evaluated_physical_review_only",
        "expansion_provenance": provenance,
    }


def _fixture(tmp_path, *, include_physical_exclusion=True, render_all=False):
    requests = [_request("base-a", "turn-a")]
    if include_physical_exclusion:
        requests.append(_request("base-b", "turn-b", 30, 55))
    requests_path = tmp_path / "requests.jsonl"
    _write_jsonl(requests_path, requests)
    plan_path = tmp_path / "plan.json"
    plan = {
        "schema_version": "1.0.0",
        "artifact_kind": PLAN_KIND,
        "selection_policy": SELECTION_POLICY,
        "fixed_minimum_maximum_or_target_duration_used": False,
        "accepted_for_training_count": 0,
        "outputs": {
            "expansion_requests": {
                "path": str(requests_path.resolve()),
                "sha256": blind.sha256(requests_path),
                "records": len(requests),
            }
        },
    }
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    rendered_requests = requests if render_all else requests[:1]
    render_records = [_render_record(tmp_path, request) for request in rendered_requests]
    for render_record in render_records:
        render_record["expansion_provenance"][
            "expansion_plan_summary_sha256"
        ] = blind.sha256(plan_path)
        render_record["expansion_provenance"][
            "expansion_requests_sha256"
        ] = blind.sha256(requests_path)
    queue_path = tmp_path / "physical_queue.jsonl"
    queue_records = []
    sampled_records = []
    for rank, render_record in enumerate(render_records):
        queue_record = dict(render_record)
        for field in (
            "status",
            "trajectory_frames",
            "video_path",
            "video_sha256",
            "video_check",
            "final_output_binding",
        ):
            queue_record.pop(field)
        queue_record["artifact_kind"] = QUEUE_KIND
        queue_record["fps"] = 30.0
        queue_records.append(queue_record)
        render_record["input_fingerprint"] = blind.value_sha256(queue_record)
        sampled_record = renderer.review_projection(
            queue_record, rank=rank, sampling="sequential", seed=0
        )
        sampled_record.update(
            {
                field: queue_record[field]
                for field in EXPANSION_RENDER_AUDIT_FIELDS
                if field in queue_record
            }
        )
        for field, value in sampled_record.items():
            if field not in SAMPLED_FIELDS_NOT_IN_RENDER_RESULT:
                render_record[field] = value
        sampled_records.append(sampled_record)
    _write_jsonl(queue_path, queue_records)
    passed_path = tmp_path / "passed.jsonl"
    _write_jsonl(passed_path, render_records)
    sampled_path = tmp_path / "sampled.jsonl"
    _write_jsonl(sampled_path, sampled_records)
    failed_path = tmp_path / "failed.jsonl"
    _write_jsonl(failed_path, [])

    audit = {
        "schema_version": "1.0.0",
        "artifact_kind": INPUT_AUDIT_KIND,
        "selection_policy": SELECTION_POLICY,
        "processing_scope": PROCESSING_SCOPE,
        "records": len(requests),
        "inputs": {
            "expansion_plan_summary": {
                "path": str(plan_path.resolve()),
                "sha256": blind.sha256(plan_path),
            },
            "expansion_requests": {
                "path": str(requests_path.resolve()),
                "sha256": blind.sha256(requests_path),
            },
        },
        "fixed_minimum_maximum_or_target_duration_used": False,
        "semantic_admission": False,
        "affect_admission": False,
        "license_training_admission": False,
        "accepted_for_training": False,
    }
    audit["sha256"] = blind.value_sha256(audit)
    pipeline = {
        "schema_version": "1.0.0",
        "artifact_kind": PIPELINE_KIND,
        "stage": "render",
        "input_audit": audit,
        "derived_candidate_count": len(requests),
        "review_queue_summary": {
            "artifact_kind": QUEUE_KIND,
            "records": len(queue_records),
            "output": str(queue_path.resolve()),
            "output_sha256": blind.sha256(queue_path),
            "selection_policy": SELECTION_POLICY,
            "elapsed_seconds_used_for_selection": False,
            "processing_scope": PROCESSING_SCOPE,
            "semantic_admission": False,
            "affect_admission": False,
            "license_training_admission": False,
            "accepted_for_training": False,
        },
        "render_summary": {
            "stage": "beat2_annotation_review_video_render",
            "run_state": "finished",
            "review_queue": str(queue_path.resolve()),
            "review_queue_sha256": blind.sha256(queue_path),
            "queue_records": len(queue_records),
            "sampling": {
                "mode": "sequential",
                "seed": 0,
                "limit": None,
                "selected_records": len(queue_records),
                "sampled_manifest": str(sampled_path.resolve()),
                "sampled_manifest_sha256": blind.sha256(sampled_path),
            },
            "counts": {
                "passed": len(queue_records),
                "failed": 0,
                "resume_reused": 0,
            },
            "passed_manifest": str(passed_path.resolve()),
            "passed_manifest_sha256": blind.sha256(passed_path),
            "failed_manifest": str(failed_path.resolve()),
            "failed_manifest_sha256": blind.sha256(failed_path),
            "render_pass_grants_training_admission": False,
            "accepted_for_training": 0,
        },
        "selection_policy": SELECTION_POLICY,
        "fixed_six_second_windows_used": False,
        "elapsed_seconds_used_for_selection": False,
        "processing_scope": PROCESSING_SCOPE,
        "semantic_admission": False,
        "affect_admission": False,
        "license_training_admission": False,
        "accepted_for_training": False,
    }
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(json.dumps(pipeline), encoding="utf-8")
    return pipeline_path, plan_path, requests_path, passed_path, render_records[0]


def _refresh_passed_binding(pipeline_path, passed_path):
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    pipeline["render_summary"]["passed_manifest_sha256"] = blind.sha256(passed_path)
    pipeline_path.write_text(json.dumps(pipeline), encoding="utf-8")


def _refresh_queue_binding(pipeline_path, queue_path):
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    digest = blind.sha256(queue_path)
    pipeline["review_queue_summary"]["output_sha256"] = digest
    pipeline["render_summary"]["review_queue_sha256"] = digest
    pipeline_path.write_text(json.dumps(pipeline), encoding="utf-8")


def test_bundle_is_anonymous_native_variable_length_and_allows_physical_exclusions(tmp_path):
    pipeline, plan, requests, passed, _record = _fixture(tmp_path)
    result = build_bundle(
        pipeline_summary=pipeline,
        expansion_plan_summary=plan,
        expansion_requests=requests,
        render_passed_manifest=passed,
        output_root=tmp_path / "public_bundle",
        hidden_root=tmp_path / "private_bundle",
        secret_hex="55" * 32,
    )
    arc = json.loads(
        (tmp_path / "public_bundle/public/arc_action_review_queue.jsonl").read_text()
    )
    affect = json.loads(
        (tmp_path / "public_bundle/public/affect_review_queue.jsonl").read_text()
    )
    hidden = json.loads(
        (tmp_path / "private_bundle/sample_mapping.jsonl").read_text()
    )
    assert set(arc) == ARC_ACTION_PUBLIC_KEYS
    assert set(affect) == AFFECT_PUBLIC_KEYS
    assert arc["native_duration_preserved"] is True
    assert arc["fixed_duration_window_used"] is False
    assert arc["frame_count"] == 16
    assert arc["sample_id"] == affect["sample_id"]
    assert "base-a" not in json.dumps(arc)
    assert hidden["base_sample_id"] == "base-a"
    assert hidden["task_id"] == "turn-a__ctxL01"
    assert hidden["frame_count"] == 16
    assert hidden["trajectory_sha256"]
    source_video = tmp_path / "base-a.mp4"
    anonymous_video = tmp_path / "public_bundle/public/videos" / f"{arc['sample_id']}.mp4"
    assert os.stat(source_video).st_ino != os.stat(anonymous_video).st_ino
    assert result["hidden"]["requested_records"] == 2
    assert result["hidden"]["physical_qc_excluded_records"] == 1
    assert result["public"]["accepted_for_training"] is False


def test_each_hidden_base_sample_id_comes_from_its_paired_request(tmp_path):
    pipeline, plan, requests, passed, _record = _fixture(
        tmp_path, include_physical_exclusion=True, render_all=True
    )
    build_bundle(
        pipeline_summary=pipeline,
        expansion_plan_summary=plan,
        expansion_requests=requests,
        render_passed_manifest=passed,
        output_root=tmp_path / "public_bundle",
        hidden_root=tmp_path / "private_bundle",
        secret_hex="56" * 32,
    )
    hidden_rows = blind.read_jsonl(tmp_path / "private_bundle/sample_mapping.jsonl")
    assert {row["base_sample_id"] for row in hidden_rows} == {"base-a", "base-b"}
    assert {
        (row["base_sample_id"], row["base_task_id"]) for row in hidden_rows
    } == {("base-a", "turn-a"), ("base-b", "turn-b")}


def test_bundle_accepts_completed_all_stage_pipeline(tmp_path):
    pipeline, plan, requests, passed, _record = _fixture(
        tmp_path, include_physical_exclusion=False
    )
    pipeline_value = json.loads(pipeline.read_text(encoding="utf-8"))
    pipeline_value["stage"] = "all"
    pipeline.write_text(json.dumps(pipeline_value), encoding="utf-8")

    result = build_bundle(
        pipeline_summary=pipeline,
        expansion_plan_summary=plan,
        expansion_requests=requests,
        render_passed_manifest=passed,
        output_root=tmp_path / "public_bundle",
        hidden_root=tmp_path / "private_bundle",
        secret_hex="5a" * 32,
    )

    assert result["public"]["records"] == 1


def test_bundle_rejects_fixed_window_or_changed_natural_interval(tmp_path):
    pipeline, plan, requests, passed, record = _fixture(
        tmp_path, include_physical_exclusion=False
    )
    record["training_segment"]["fixed_window_sec"] = 6.0
    _write_jsonl(passed, [record])
    _refresh_passed_binding(pipeline, passed)
    with pytest.raises(ValueError, match="differs from queued trajectory/input"):
        build_bundle(
            pipeline_summary=pipeline,
            expansion_plan_summary=plan,
            expansion_requests=requests,
            render_passed_manifest=passed,
            output_root=tmp_path / "public_bundle",
            hidden_root=tmp_path / "private_bundle",
            secret_hex="66" * 32,
        )


def test_bundle_rejects_incomplete_render_summary(tmp_path):
    pipeline, plan, requests, passed, _record = _fixture(
        tmp_path, include_physical_exclusion=False
    )
    value = json.loads(pipeline.read_text())
    value["render_summary"]["run_state"] = "finished_with_failures"
    pipeline.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete or unbound"):
        build_bundle(
            pipeline_summary=pipeline,
            expansion_plan_summary=plan,
            expansion_requests=requests,
            render_passed_manifest=passed,
            output_root=tmp_path / "public_bundle",
            hidden_root=tmp_path / "private_bundle",
            secret_hex="77" * 32,
        )


def test_request_rejects_false_interval_arithmetic_and_non_containment(tmp_path):
    request = _request("base-a", "turn-a")
    request["requested_interval"] = {
        "start_frame": 13,
        "end_frame_exclusive": 22,
        "frame_count": 180,
    }
    request["plan_record_sha256"] = blind.value_sha256(
        {key: value for key, value in request.items() if key != "plan_record_sha256"}
    )
    path = tmp_path / "invalid_requests.jsonl"
    _write_jsonl(path, [request])
    with pytest.raises(ValueError, match="frame arithmetic"):
        _index_requests(path)

    request["requested_interval"]["frame_count"] = 9
    request["plan_record_sha256"] = blind.value_sha256(
        {key: value for key, value in request.items() if key != "plan_record_sha256"}
    )
    _write_jsonl(path, [request])
    with pytest.raises(ValueError, match="strictly contain"):
        _index_requests(path)


@pytest.mark.parametrize(
    "hidden_relative",
    ("public_bundle/public", "public_bundle/public/private", "public_bundle"),
)
def test_bundle_rejects_overlapping_public_and_hidden_roots(tmp_path, hidden_relative):
    pipeline, plan, requests, passed, _record = _fixture(
        tmp_path, include_physical_exclusion=False
    )
    hidden = tmp_path / hidden_relative
    with pytest.raises(ValueError, match="must be disjoint"):
        build_bundle(
            pipeline_summary=pipeline,
            expansion_plan_summary=plan,
            expansion_requests=requests,
            render_passed_manifest=passed,
            output_root=tmp_path / "public_bundle",
            hidden_root=hidden,
            secret_hex="88" * 32,
        )
    assert not (hidden / "bundle_secret.json").exists()


def test_bundle_rejects_substituted_render_task_set(tmp_path):
    pipeline, plan, requests, passed, record = _fixture(
        tmp_path, include_physical_exclusion=False
    )
    record["task_id"] = "turn-b__ctxL01"
    _write_jsonl(passed, [record])
    _refresh_passed_binding(pipeline, passed)
    with pytest.raises(ValueError, match="task set differs"):
        build_bundle(
            pipeline_summary=pipeline,
            expansion_plan_summary=plan,
            expansion_requests=requests,
            render_passed_manifest=passed,
            output_root=tmp_path / "public_bundle",
            hidden_root=tmp_path / "private_bundle",
            secret_hex="99" * 32,
        )


def test_bundle_rejects_fake_video_even_when_manifest_claims_it_passed(tmp_path):
    pipeline, plan, requests, passed, record = _fixture(
        tmp_path, include_physical_exclusion=False
    )
    fake = tmp_path / "fake.mp4"
    fake.write_bytes(b"not-a-video")
    record["video_path"] = str(fake.resolve())
    record["video_sha256"] = blind.sha256(fake)
    binding = record["final_output_binding"]
    binding["video_path"] = str(fake.resolve())
    binding["video_sha256"] = blind.sha256(fake)
    binding["sha256"] = blind.value_sha256(
        {key: value for key, value in binding.items() if key != "sha256"}
    )
    _write_jsonl(passed, [record])
    _refresh_passed_binding(pipeline, passed)
    with pytest.raises(ValueError, match="video"):
        build_bundle(
            pipeline_summary=pipeline,
            expansion_plan_summary=plan,
            expansion_requests=requests,
            render_passed_manifest=passed,
            output_root=tmp_path / "public_bundle",
            hidden_root=tmp_path / "private_bundle",
            secret_hex="aa" * 32,
        )


def test_bundle_rejects_stale_unexpected_public_video(tmp_path):
    pipeline, plan, requests, passed, _record = _fixture(
        tmp_path, include_physical_exclusion=False
    )
    arguments = {
        "pipeline_summary": pipeline,
        "expansion_plan_summary": plan,
        "expansion_requests": requests,
        "render_passed_manifest": passed,
        "output_root": tmp_path / "public_bundle",
        "hidden_root": tmp_path / "private_bundle",
        "secret_hex": "bb" * 32,
    }
    build_bundle(**arguments)
    stale = tmp_path / "public_bundle/public/videos/stale.mp4"
    stale.write_bytes(b"stale")
    with pytest.raises(ValueError, match="stale unexpected"):
        build_bundle(**arguments)


def test_bundle_rejects_same_task_with_substituted_valid_trajectory(tmp_path):
    pipeline, plan, requests, passed, record = _fixture(
        tmp_path, include_physical_exclusion=False
    )
    original = tmp_path / "base-a.csv"
    substitute = tmp_path / "substitute.csv"
    rows = original.read_text(encoding="utf-8").splitlines()
    substitute.write_text(
        rows[0]
        + "\n"
        + "\n".join(
            ",".join(f"{float(value) + 0.1:.6f}" for value in row.split(","))
            for row in rows[1:]
        )
        + "\n",
        encoding="utf-8",
    )
    record["trajectory_path"] = str(substitute.resolve())
    record["trajectory_sha256"] = blind.sha256(substitute)
    binding = record["final_output_binding"]
    binding["trajectory_path"] = str(substitute.resolve())
    binding["trajectory_sha256"] = blind.sha256(substitute)
    binding["sha256"] = blind.value_sha256(
        {key: value for key, value in binding.items() if key != "sha256"}
    )
    _write_jsonl(passed, [record])
    _refresh_passed_binding(pipeline, passed)
    with pytest.raises(ValueError, match="differs from queued trajectory/input"):
        build_bundle(
            pipeline_summary=pipeline,
            expansion_plan_summary=plan,
            expansion_requests=requests,
            render_passed_manifest=passed,
            output_root=tmp_path / "public_bundle",
            hidden_root=tmp_path / "private_bundle",
            secret_hex="cc" * 32,
        )


def test_bundle_rejects_public_root_symlink_to_hidden_root(tmp_path):
    pipeline, plan, requests, passed, _record = _fixture(
        tmp_path, include_physical_exclusion=False
    )
    output = tmp_path / "public_bundle"
    hidden = tmp_path / "private_bundle"
    output.mkdir()
    hidden.mkdir()
    (output / "public").symlink_to(hidden, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        build_bundle(
            pipeline_summary=pipeline,
            expansion_plan_summary=plan,
            expansion_requests=requests,
            render_passed_manifest=passed,
            output_root=output,
            hidden_root=hidden,
            secret_hex="dd" * 32,
        )
    assert not (hidden / "bundle_secret.json").exists()


def test_bundle_rejects_stale_sensitive_file_in_public_root(tmp_path):
    pipeline, plan, requests, passed, _record = _fixture(
        tmp_path, include_physical_exclusion=False
    )
    arguments = {
        "pipeline_summary": pipeline,
        "expansion_plan_summary": plan,
        "expansion_requests": requests,
        "render_passed_manifest": passed,
        "output_root": tmp_path / "public_bundle",
        "hidden_root": tmp_path / "private_bundle",
        "secret_hex": "ee" * 32,
    }
    build_bundle(**arguments)
    stale = tmp_path / "public_bundle/public/sample_mapping.jsonl"
    stale.write_text('{"secret":"leak"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="stale unexpected public bundle entries"):
        build_bundle(**arguments)


def test_bundle_requires_quality_evidence_and_all_pass_gate(tmp_path):
    pipeline, plan, requests, passed, _record = _fixture(
        tmp_path, include_physical_exclusion=False
    )
    queue_path = tmp_path / "physical_queue.jsonl"
    queue_record = json.loads(queue_path.read_text(encoding="utf-8"))
    queue_record.pop("quality_json")
    queue_record.pop("quality_json_sha256")
    queue_record.pop("quality_gate")
    _write_jsonl(queue_path, [queue_record])
    _refresh_queue_binding(pipeline, queue_path)
    with pytest.raises(ValueError, match="physical queue requires quality_json"):
        build_bundle(
            pipeline_summary=pipeline,
            expansion_plan_summary=plan,
            expansion_requests=requests,
            render_passed_manifest=passed,
            output_root=tmp_path / "public_bundle",
            hidden_root=tmp_path / "private_bundle",
            secret_hex="ff" * 32,
        )


def test_bundle_rejects_incomplete_all_true_quality_gate(tmp_path):
    pipeline, plan, requests, passed, _record = _fixture(
        tmp_path, include_physical_exclusion=False
    )
    queue_path = tmp_path / "physical_queue.jsonl"
    queue_record = json.loads(queue_path.read_text(encoding="utf-8"))
    quality_path = tmp_path / "base-a.quality.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    incomplete_gate = {"made_up_gate": True}
    queue_record["quality_gate"] = incomplete_gate
    quality["quality_gate"] = incomplete_gate
    quality_path.write_text(json.dumps(quality), encoding="utf-8")
    queue_record["quality_json_sha256"] = blind.sha256(quality_path)
    _write_jsonl(queue_path, [queue_record])
    _refresh_queue_binding(pipeline, queue_path)

    with pytest.raises(ValueError, match="missing required 18D gates"):
        build_bundle(
            pipeline_summary=pipeline,
            expansion_plan_summary=plan,
            expansion_requests=requests,
            render_passed_manifest=passed,
            output_root=tmp_path / "public_bundle",
            hidden_root=tmp_path / "private_bundle",
            secret_hex="10" * 32,
        )


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("time_map_endpoint", "time map is not full"),
        ("algorithm_hash", "algorithm_contract_sha256 is invalid"),
        ("trajectory_endpoint", "retime_input_first_frame"),
    ),
)
def test_bundle_rejects_corrupt_complete_safety_audit(tmp_path, corruption, message):
    pipeline, plan, requests, passed, _record = _fixture(
        tmp_path, include_physical_exclusion=False
    )
    queue_path = tmp_path / "physical_queue.jsonl"
    queue_record = json.loads(queue_path.read_text(encoding="utf-8"))
    quality_path = tmp_path / "base-a.quality.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    safety = quality["safety_monotonic_retime"]
    if corruption == "time_map_endpoint":
        safety["input_frame_output_times_sec"][-1] += 10.0
    elif corruption == "algorithm_hash":
        safety["algorithm_contract_sha256"] = "0" * 64
    else:
        safety["retime_input_first_frame"][0] += 0.1
    queue_record["safety_monotonic_retime"] = safety
    quality_path.write_text(json.dumps(quality), encoding="utf-8")
    queue_record["quality_json_sha256"] = blind.sha256(quality_path)
    _write_jsonl(queue_path, [queue_record])
    _refresh_queue_binding(pipeline, queue_path)

    with pytest.raises(ValueError, match=message):
        build_bundle(
            pipeline_summary=pipeline,
            expansion_plan_summary=plan,
            expansion_requests=requests,
            render_passed_manifest=passed,
            output_root=tmp_path / "public_bundle",
            hidden_root=tmp_path / "private_bundle",
            secret_hex="20" * 32,
        )
