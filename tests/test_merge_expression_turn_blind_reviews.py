import copy
import csv
import json
from pathlib import Path

import pytest

from tools.human_motion_review import merge_expression_turn_blind_reviews as MERGE
from upper_body_skeleton.ula_v2_expression_turn_episode import (
    load_expression_turn_v8_episodes,
    validate_expression_turn_v8_episode,
)


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(MERGE.stable_json(row) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _fixture(tmp_path, *, ula0513=False):
    video = tmp_path / "public/videos/expr_test.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"anonymous-silent-video")
    video_hash = MERGE.sha256(video)
    common = {
        "schema_version": "1.0.0",
        "sample_id": "expr_test",
        "video_path": str(video),
        "video_sha256": video_hash,
        "context_level": 0,
        "audio_available": False,
        "label_metadata_exposed": False,
    }
    arc = {
        **common,
        "arc_protocol_version": MERGE.ARC_PROTOCOL,
        "action_protocol_version": MERGE.ACTION_PROTOCOL,
    }
    affect_queue = {**common, "affect_protocol_version": MERGE.AFFECT_PROTOCOL}
    arc_path = tmp_path / "public/arc.jsonl"
    affect_path = tmp_path / "public/affect.jsonl"
    _write_jsonl(arc_path, [arc])
    _write_jsonl(affect_path, [affect_queue])
    summary = {
        "schema_version": "1.0.0",
        "artifact_kind": (
            "ula0513_native_separate_blind_review_bundle"
            if ula0513
            else "expression_turn_v8_separate_blind_review_bundle"
        ),
        "accepted_for_training": 0 if ula0513 else False,
        "arc_action_queue": str(arc_path),
        "arc_action_queue_sha256": MERGE.sha256(arc_path),
        "affect_queue": str(affect_path),
        "affect_queue_sha256": MERGE.sha256(affect_path),
    }
    summary_path = tmp_path / "public/summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    trajectory = tmp_path / "motion/final.csv"
    trajectory.parent.mkdir(parents=True)
    with trajectory.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(MERGE.JOINT_ORDER_18D)
        for index in range(4):
            writer.writerow([index / 100.0] * 18)
    motion = {
        "schema_version": "1.0.0",
        "task_id": "motion-task",
        "status": "passed",
        "accepted_for_training": False,
        "video_sha256": video_hash,
        "trajectory_path": str(trajectory),
        "trajectory_sha256": MERGE.sha256(trajectory),
        "trajectory_frames": 4,
        "render_config": {"fps": 30},
        "video_check": {
            "fully_decodable": True,
            "decoded_frames": 4,
            "expected_frames": 4,
            "nonblank": True,
        },
        "fixed_split_assignment": "train",
        "training_segment": {
            "representation": "native_variable_length_expression_turn_v1",
            "start_frame": 0,
            "end_frame_exclusive": 4,
            "frame_count": 4,
            "fixed_window_sec": None,
            "cropped": False,
        },
        "retarget_segment": {
            "representation": "native_variable_length_expression_turn_safety_retimed_30hz_v2",
            "source_frame_count": 4,
            "output_frame_count": 4,
            "fps": 30,
            "cropped": False,
            "fixed_target_duration_sec": None,
        },
    }
    source_catalog_path = None
    if ula0513:
        source_record = {
            "schema_version": "1.0.0",
            "clip_id": "private-source-name",
            "dataset": "ULA0513_user_provided_robot_motion",
            "source": {"csv_sha256": "a" * 64},
            "physical_qc": {
                "passed": True,
                "joint_limits_pass": True,
                "safe_projection_pass": True,
                "max_safe_projection_rad": 0.0,
                "safe_projection_threshold_rad": 0.01,
                "timing": {
                    "starts_at_zero": True,
                    "strictly_increasing": True,
                    "native_30hz": True,
                },
            },
            "motion_18d": {"safe_csv_sha256": MERGE.sha256(trajectory)},
        }
        source_record["record_sha256"] = MERGE.value_sha256(source_record)
        source_catalog_path = tmp_path / "source_catalog.jsonl"
        _write_jsonl(source_catalog_path, [source_record])
        hidden = {
            "sample_id": "expr_test",
            "task_id": "motion-task",
            "render_record_sha256": MERGE.value_sha256(motion),
            "source_record_sha256": source_record["record_sha256"],
            "source_behavior_label": "HiddenWaveLabel",
            "frame_count": 4,
            "trajectory_sha256": MERGE.sha256(trajectory),
        }
    else:
        motion["quality_gate"] = {
            gate: True for gate in MERGE.REQUIRED_18D_GATES
        }
        motion.update({
            "expression_turn_contract_sha256": "1" * 64,
            "source_inventory_manifest_sha256": "2" * 64,
            "split_assignment_manifest_sha256": "3" * 64,
            "expression_turn_selection_record_sha256": "4" * 64,
            "upstream_inventory_record_sha256": "5" * 64,
            "selected_record_sha256": "6" * 64,
            "retarget_input_manifest_sha256": "7" * 64,
            "source_sha256": "8" * 64,
            "source_clip_id": "human-source-clip",
            "speaker_key": "human-speaker",
        })
        hidden = {
            "sample_id": "expr_test",
            "task_id": "motion-task",
            "source_render_record_sha256": MERGE.value_sha256(motion),
            "official_emotion": "hidden-fear",
        }
    motion_path = tmp_path / "motion_manifest.jsonl"
    hidden_path = tmp_path / "hidden.jsonl"
    _write_jsonl(motion_path, [motion])
    _write_jsonl(hidden_path, [hidden])

    text = "The robot raises both forearms, pauses, and settles to rest."
    author = {
        **common,
        "arc_protocol_version": MERGE.ARC_PROTOCOL,
        "arc_review_id": "arc-r1",
        "arc_reviewer_id": "author-r1",
        "onset_status": "complete",
        "onset_evidence_frame": 0,
        "onset_basis": "coherent_motion_entry",
        "apex_status": "complete",
        "apex_evidence_frame": 1,
        "apex_basis": "distinct_motion_or_pose_peak",
        "offset_status": "complete",
        "offset_evidence_frame": 3,
        "offset_basis": "natural_settle",
        "action_protocol_version": MERGE.ACTION_PROTOCOL,
        "action_review_id": "author-action-r1",
        "action_reviewer_id": "author-r1",
        "action_result": "observable_match",
        "observable_description": text,
        "candidate_text": text,
        "candidate_text_sha256": MERGE.text_sha256(text),
        "candidate_text_provenance": MERGE.ACTION_TEXT_PROVENANCE,
    }
    matcher = {
        **common,
        "action_protocol_version": MERGE.ACTION_PROTOCOL,
        "author_action_review_id": "author-action-r1",
        "candidate_text": text,
        "candidate_text_sha256": MERGE.text_sha256(text),
        "action_match_review_id": "matcher-r1",
        "action_match_reviewer_id": "matcher-reviewer-r1",
        "action_result": "observable_match",
        "observable_description": "The description matches the visible motion.",
    }
    if ula0513:
        matcher.pop("action_match_review_id")
        matcher.pop("action_match_reviewer_id")
        matcher.pop("author_action_review_id")
        matcher.pop("observable_description")
        matcher.update(
            {
                "action_protocol_version": MERGE.NATIVE_MATCHER_PROTOCOL,
                "action_review_id": "matcher-r1",
                "action_reviewer_id": "matcher-reviewer-r1",
                "candidate_text_provenance": MERGE.ACTION_TEXT_PROVENANCE,
                "match_confidence": 0.97,
                "mismatch_notes": None,
            }
        )
    affect1 = {
        **common,
        "affect_protocol_version": MERGE.AFFECT_PROTOCOL,
        "affect_review_id": "affect-r1",
        "affect_reviewer_id": "affect-reviewer-r1",
        "result": "observable",
        "predicted_class": "happy",
        "confidence": 0.81,
    }
    affect2 = {
        **common,
        "affect_protocol_version": MERGE.AFFECT_PROTOCOL,
        "affect_review_id": "affect-r2",
        "affect_reviewer_id": "affect-reviewer-r2",
        "result": "observable",
        "predicted_class": "happy",
        "confidence": 0.78,
    }
    paths = {}
    for name, rows in (
        ("author", [author]),
        ("matcher", [matcher]),
        ("affect1", [affect1]),
        ("affect2", [affect2]),
    ):
        paths[name] = tmp_path / f"{name}.jsonl"
        _write_jsonl(paths[name], rows)
    return {
        "summary": summary_path,
        "hidden": hidden_path,
        "motion": motion_path,
        "source_catalog": source_catalog_path,
        "paths": paths,
        "rows": {"author": author, "matcher": matcher, "affect1": affect1, "affect2": affect2},
    }


def _run(fixture, output, **overrides):
    values = {
        "public_summary": fixture["summary"],
        "hidden_mapping": fixture["hidden"],
        "motion_manifest": fixture["motion"],
        "source_catalog": fixture["source_catalog"],
        "author_submissions": fixture["paths"]["author"],
        "matcher_submissions": fixture["paths"]["matcher"],
        "affect_submissions": [fixture["paths"]["affect1"], fixture["paths"]["affect2"]],
        "output_root": output,
    }
    values.update(overrides)
    return MERGE.merge_reviews(**values)


def test_full_consensus_writes_only_highest_train_ready_tier(tmp_path):
    fixture = _fixture(tmp_path / "fixture")
    output = tmp_path / "output"
    summary = _run(fixture, output)
    assert summary["counts"] == {
        "base_motion_train_ready": 0,
        "semantic_conditioning_train_ready": 0,
        "expressive_conditioning_train_ready": 1,
        "pending": 0,
        "rejected": 0,
    }
    expressive = _read_jsonl(output / "expressive_conditioning_train_ready.jsonl")[0]
    assert expressive["accepted_for_training"] is True
    assert expressive["formal_episode_contract"] == MERGE.FORMAL_EPISODE_CONTRACT
    assert expressive["eligibility_mode"] == MERGE.FORMAL_ELIGIBILITY_MODE
    assert expressive["training_channel_masks"]["expressive_conditioning"] is True
    assert expressive["prompt_semantics_profile"] == MERGE.MOTION_FORM_PROMPT_PROFILE
    assert expressive["prompt_review_id"] == "matcher-r1"
    assert expressive["emotion_id"] == "happy"
    assert expressive["motion_18d"]["frames"] == 4
    assert expressive["motion_18d"]["safe_csv_sha256"]
    assert expressive["expression_turn_review_record_sha256"] == MERGE.value_sha256(
        expressive["expression_turn_review_record"]
    )
    assert expressive["qualification_report_sha256"] == MERGE.value_sha256(
        expressive["qualification_report"]
    )
    episodes = load_expression_turn_v8_episodes(
        output / "expressive_conditioning_train_ready.jsonl"
    )
    assert validate_expression_turn_v8_episode(episodes[0])["highest_qualification"] == (
        "expressive_conditioning"
    )
    assert summary["network_contract_validation"]["expressive_conditioning"] == {
        "records": 1,
        "loader_invoked": True,
        "loader_validated_records": 1,
        "episode_validator_validated_records": 1,
        "network_contract_validation_passed": True,
    }
    assert "semantic_event" not in json.dumps(expressive)


def test_author_only_qualifies_base_and_emits_matcher_queue(tmp_path):
    fixture = _fixture(tmp_path / "fixture")
    output = tmp_path / "output"
    summary = _run(
        fixture,
        output,
        matcher_submissions=None,
        affect_submissions=[],
    )
    assert summary["counts"]["base_motion_train_ready"] == 1
    assert summary["counts"]["semantic_conditioning_train_ready"] == 0
    assert summary["counts"]["expressive_conditioning_train_ready"] == 0
    matcher_queue = _read_jsonl(output / "action_match_queue.jsonl")[0]
    assert matcher_queue["candidate_text_sha256"] == fixture["rows"]["author"]["candidate_text_sha256"]
    assert matcher_queue["accepted_for_training"] is False


@pytest.mark.parametrize("mutation,match", [
    (lambda row: row.update(action_match_reviewer_id="author-r1"), "not independent"),
    (lambda row: row.update(candidate_text_sha256="f" * 64), "candidate text binding"),
    (lambda row: row.update(video_sha256="e" * 64), "video_sha256 binding"),
])
def test_matcher_identity_text_and_video_are_strictly_bound(tmp_path, mutation, match):
    fixture = _fixture(tmp_path / "fixture")
    matcher = copy.deepcopy(fixture["rows"]["matcher"])
    mutation(matcher)
    bad = tmp_path / "bad_matcher.jsonl"
    _write_jsonl(bad, [matcher])
    with pytest.raises(ValueError, match=match):
        _run(fixture, tmp_path / "output", matcher_submissions=bad)


def test_conflicting_affect_needs_independent_adjudication(tmp_path):
    fixture = _fixture(tmp_path / "fixture")
    conflicting = copy.deepcopy(fixture["rows"]["affect2"])
    conflicting["predicted_class"] = "sad"
    _write_jsonl(fixture["paths"]["affect2"], [conflicting])
    output1 = tmp_path / "without_adjudication"
    summary = _run(fixture, output1)
    assert summary["counts"]["semantic_conditioning_train_ready"] == 1
    assert summary["counts"]["expressive_conditioning_train_ready"] == 0

    common = {key: fixture["rows"]["author"][key] for key in (
        "sample_id", "video_path", "video_sha256", "context_level", "audio_available", "label_metadata_exposed"
    )}
    adjudication = {
        **common,
        "affect_adjudication_protocol_version": MERGE.AFFECT_ADJUDICATION_PROTOCOL,
        "affect_adjudication_review_id": "adjudication-r1",
        "affect_adjudication_reviewer_id": "adjudicator-r1",
        "input_review_ids": ["affect-r1", "affect-r2"],
        "result": "observable",
        "predicted_class": "happy",
        "confidence": 0.9,
    }
    adjudication_path = tmp_path / "adjudication.jsonl"
    _write_jsonl(adjudication_path, [adjudication])
    output2 = tmp_path / "with_adjudication"
    summary = _run(fixture, output2, affect_adjudications=adjudication_path)
    assert summary["counts"]["expressive_conditioning_train_ready"] == 1


def test_hidden_labels_do_not_change_qualification_or_leak(tmp_path):
    fixture = _fixture(tmp_path / "fixture")
    output1 = tmp_path / "output1"
    _run(fixture, output1)
    report1 = _read_jsonl(output1 / "qualification_reports.jsonl")[0]

    hidden = _read_jsonl(fixture["hidden"])[0]
    hidden["official_emotion"] = "totally-different-hidden-label"
    hidden["source_behavior_label"] = "private-source-label"
    _write_jsonl(fixture["hidden"], [hidden])
    output2 = tmp_path / "output2"
    _run(fixture, output2)
    report2 = _read_jsonl(output2 / "qualification_reports.jsonl")[0]
    assert report1["qualifications"] == report2["qualifications"]
    payload = (output2 / "expressive_conditioning_train_ready.jsonl").read_text(encoding="utf-8")
    assert "totally-different-hidden-label" not in payload
    assert "private-source-label" not in payload
    assert report2["qualification_evidence"]["hidden_label_fields_used"] == []


def test_ula0513_adapter_uses_verified_physical_catalog_not_source_label(tmp_path):
    fixture = _fixture(tmp_path / "fixture", ula0513=True)
    output = tmp_path / "output"
    summary = _run(fixture, output)
    assert summary["dataset_kind"] == "ula0513_native_expression_turn_v1"
    assert summary["counts"]["expressive_conditioning_train_ready"] == 1
    record = _read_jsonl(output / "expressive_conditioning_train_ready.jsonl")[0]
    assert record["quality_gate"]["passed"] is True
    assert record["quality_gate"]["collision_pass"] is True
    assert record["physical_evidence_profile"] == MERGE.NATIVE_ROBOT_PHYSICAL_PROFILE
    assert summary["physical_evidence"]["collision_evaluated_records"] == 1
    assert summary["physical_evidence"]["collision_pass_records"] == 1
    assert summary["physical_evidence"]["records_with_verified_frame_count"] == 1
    assert "HiddenWaveLabel" not in json.dumps(record)


def test_expansion_bundle_requires_and_preserves_native_variable_length(tmp_path):
    fixture = _fixture(tmp_path / "fixture")
    summary_path = fixture["summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "artifact_kind": MERGE.EXPANSION_BUNDLE_KIND,
            "all_samples_native_variable_length": True,
            "fixed_duration_window_used": False,
        }
    )
    for key in ("arc_action_queue", "affect_queue"):
        queue_path = Path(summary[key])
        rows = _read_jsonl(queue_path)
        for row in rows:
            row.update(
                {
                    "native_duration_preserved": True,
                    "fixed_duration_window_used": False,
                    "frame_count": 4,
                    "fps": 30.0,
                }
            )
        _write_jsonl(queue_path, rows)
    summary["arc_action_queue_sha256"] = MERGE.sha256(
        Path(summary["arc_action_queue"])
    )
    summary["affect_queue_sha256"] = MERGE.sha256(Path(summary["affect_queue"]))
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    output = tmp_path / "output"
    merged = _run(fixture, output)
    assert merged["dataset_kind"] == "beat2_expression_turn_v8"
    record = _read_jsonl(output / "expressive_conditioning_train_ready.jsonl")[0]
    assert record["training_segment"]["frame_count"] == 4
    assert record["training_segment"]["fixed_window_sec"] is None

    summary["fixed_duration_window_used"] = True
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="violates native duration"):
        _run(fixture, tmp_path / "rejected")
