import json
from argparse import Namespace

import pytest

from tools.gmr_v2 import batch_retarget_beat2_expression_turns_v8 as runner
from tools.gmr_v2 import batch_retarget_beat2_v2 as ordinary
from tools.human_motion_review.expression_turn_contract import (
    ExpressionTurnContractError,
)


def _record(source_hash):
    return {
        "artifact_kind": runner.INPUT_ARTIFACT_KIND,
        "clip_id": "turn-v8",
        "task_id": "turn-v8",
        "source_clip_id": "source-a",
        "source_group_key": "BEAT2/source-a",
        "speaker_key": "speaker-a",
        "motion_relpath": "motion/source-a.npz",
        "motion_sha256": source_hash,
        "fps": 30,
        "representation": runner.INPUT_REPRESENTATION,
        "training_segment": {
            "representation": runner.INPUT_REPRESENTATION,
            "start_frame": 90,
            "end_frame_exclusive": 120,
            "frame_count": 30,
            "fixed_window_sec": None,
            "cropped": False,
            "duration_policy": runner.NATURAL_DURATION_POLICY,
        },
        "context_plan": {"selected_level": 0},
        "time_axes": {"source": {}, "turn": {}},
        "expression_turn": {"complete_motion_arc_verified": False},
        "semantic_supervision_masks": dict(runner.SEMANTIC_MASKS),
        "emotion_supervision_mask": False,
        "official_emotion_conditioning_enabled": False,
        "affect_observable_supervision_mask": False,
        "official_category_conditioning_enabled": False,
        "canonical_prompt": None,
        "canonical_action": None,
        "accepted_for_training": False,
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
        "training_admission_status": (
            "pending_retarget_and_independent_video_review"
        ),
    }


def _binding():
    return {
        "retarget_input_manifest_sha256": "7" * 64,
        "expression_turn_contract_sha256": "1" * 64,
        "source_inventory_manifest_sha256": "4" * 64,
        "split_assignment_manifest_sha256": "5" * 64,
        "selection_kind": "representative100",
        "require_selection_record": True,
    }


def test_reader_maps_natural_segment_and_selection_lineage(tmp_path, monkeypatch):
    beat2_root = tmp_path / "beat2"
    source = beat2_root / "motion/source-a.npz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source-motion")
    record = _record(ordinary.sha256(source))
    inventory = tmp_path / "representative100.jsonl"
    inventory.write_text(json.dumps(record) + "\n", encoding="utf-8")
    lineage = {
        "inventory_record_sha256": "3" * 64,
        "upstream_inventory_record_sha256": "2" * 64,
        "selected_record_sha256": "8" * 64,
        "retarget_input_manifest_sha256": "7" * 64,
    }
    seen = {}

    def validate(value, *, catalog_binding):
        seen["record"] = value
        seen["binding"] = catalog_binding
        return {"lineage": lineage}

    monkeypatch.setattr(runner, "validate_expression_turn_candidate", validate)
    tasks, excluded = runner.read_expression_turn_inventory(
        inventory,
        beat2_root,
        _binding(),
        "9" * 64,
    )
    assert excluded == []
    assert len(tasks) == 1
    task = tasks[0]
    assert task["start_frame"] == 90
    assert task["end_frame_exclusive"] == 120
    assert task["inventory_record_sha256"] == "3" * 64
    assert task["upstream_inventory_record_sha256"] == "2" * 64
    assert task["upstream_inventory_manifest_sha256"] == "9" * 64
    assert runner.input_record_from_task(task) == record
    assert seen == {"record": record, "binding": _binding()}


def test_reader_rejects_legacy_semantic_event_even_after_candidate_validation(
    tmp_path, monkeypatch
):
    beat2_root = tmp_path / "beat2"
    source = beat2_root / "motion/source-a.npz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source-motion")
    record = _record(ordinary.sha256(source))
    record["semantic_event"] = {"category": "deictic"}
    inventory = tmp_path / "representative100.jsonl"
    inventory.write_text(json.dumps(record) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "validate_expression_turn_candidate",
        lambda *_args, **_kwargs: {"lineage": {}},
    )
    with pytest.raises(ExpressionTurnContractError, match="legacy semantic-event"):
        runner.read_expression_turn_inventory(
            inventory, beat2_root, _binding(), "9" * 64
        )


def test_expression_retarget_segment_preserves_native_frame_count_and_duration():
    task = {"start_frame": 90, "end_frame_exclusive": 120}
    segment = runner.build_expression_retarget_segment_contract(
        task,
        source_frame_count=30,
        output_frame_count=30,
        fps=30.0,
    )
    assert segment["representation"] == runner.RETARGET_SEGMENT_REPRESENTATION
    assert segment["source_frame_count"] == 30
    assert segment["output_frame_count"] == 30
    assert segment["retimed"] is False
    assert segment["cropped"] is False
    assert segment["duration_policy"] == runner.NATURAL_DURATION_POLICY
    payload = {key: value for key, value in segment.items() if key != "sha256"}
    assert segment["sha256"] == ordinary.json_sha256(payload)


def test_quality_gate_calls_independent_output_validator_with_exact_input(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.npz"
    source.write_bytes(b"source-motion")
    record = _record(ordinary.sha256(source))
    task = {
        **record,
        "source": str(source),
        "start_frame": 90,
        "end_frame_exclusive": 120,
        "inventory_record_sha256": "3" * 64,
        "upstream_inventory_record_sha256": "2" * 64,
        "selected_record_sha256": "8" * 64,
        "retarget_input_manifest_sha256": "7" * 64,
        "selected_record_sha256_role": "current",
        "retarget_input_manifest_sha256_role": "current",
        "upstream_inventory_manifest_sha256": "9" * 64,
        "source_warnings": [],
        "conditioning_text_status": "disabled",
    }
    segment = runner.build_expression_retarget_segment_contract(
        task,
        source_frame_count=30,
        output_frame_count=30,
        fps=30.0,
    )
    quality = {
        "artifact_kind": runner.QUALITY_ARTIFACT_KIND,
        "input_representation": runner.INPUT_REPRESENTATION,
        "frames": 30,
        "accepted_for_training": False,
        "semantic_supervision_masks": dict(runner.SEMANTIC_MASKS),
        "emotion_supervision_mask": False,
        "affect_observable_supervision_mask": False,
        "official_category_conditioning_enabled": False,
        "official_emotion_conditioning_enabled": False,
        "retarget_segment": segment,
        "safety_monotonic_retime": {"contract_present": True},
    }
    for field in (
        "expression_turn_contract_sha256",
        "expression_turn_record_sha256",
        "expression_turn_selection_record_sha256",
        "inventory_record_sha256",
        "upstream_inventory_record_sha256",
        "selected_record_sha256",
        "retarget_input_manifest_sha256",
        "training_segment",
        "time_axes",
        "context_plan",
        "expression_turn",
    ):
        quality[field] = task[field]
    seen = {}
    monkeypatch.setattr(runner.ordinary, "quality_passes", lambda *_args: True)

    def validate(value, *, input_record, catalog_binding, safe_csv_path):
        seen.update(
            quality=value,
            input_record=input_record,
            binding=catalog_binding,
            safe_csv_path=safe_csv_path,
        )
        return {"retarget_output_valid": True}

    monkeypatch.setattr(runner, "validate_expression_turn_retarget_output", validate)
    safe_csv = tmp_path / "safe.csv"
    safe_csv.write_text("placeholder\n", encoding="utf-8")
    assert runner.expression_quality_passes(
        quality,
        task,
        ordinary.sha256(source),
        _binding(),
        safe_csv_path=safe_csv,
    )
    assert seen["input_record"] == record
    assert seen["binding"] == _binding()
    assert seen["safe_csv_path"] == safe_csv

    quality["semantic_event"] = {}
    assert not runner.expression_quality_passes(
        quality,
        task,
        ordinary.sha256(source),
        _binding(),
        safe_csv_path=safe_csv,
    )


def test_catalog_binding_verifies_review_set_and_upstream_hashes(tmp_path):
    inventory = tmp_path / "beat2_expression_turn_v8.representative100.jsonl"
    inventory.write_text("{}\n", encoding="utf-8")
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text("{}\n", encoding="utf-8")
    contract = {
        "duration_policy": runner.NATURAL_DURATION_POLICY,
        "fixed_window_sec": None,
        "semantic_supervision_masks": dict(runner.SEMANTIC_MASKS),
        "emotion_supervision_mask": False,
        "builder_script_sha256": ordinary.sha256(runner.BUILDER_SCRIPT),
        "source_manifest_sha256": ordinary.sha256(source),
        "split_assignment": {"manifest_sha256": ordinary.sha256(split)},
    }
    summary = {
        "artifact_kind": "beat2_expression_turn_v8_candidate_catalog",
        "expression_turn_contract": contract,
        "expression_turn_contract_sha256": ordinary.json_sha256(contract),
        "input": str(source),
        "input_sha256": ordinary.sha256(source),
        "split_assignment": str(split),
        "split_assignment_sha256": ordinary.sha256(split),
        "output_sha256": {
            inventory.name: ordinary.sha256(inventory),
            "beat2_expression_turn_v8.candidates.jsonl": "a" * 64,
        },
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    binding, audit = runner.load_catalog_binding(
        inventory, summary_path, "representative100"
    )
    assert binding["selection_kind"] == "representative100"
    assert binding["require_selection_record"] is True
    assert binding["retarget_input_manifest_sha256"] == ordinary.sha256(inventory)
    assert audit["catalog_candidate_manifest_sha256"] == "a" * 64


def test_catalog_binding_supports_hash_bound_full_pool(tmp_path):
    inventory = tmp_path / "beat2_expression_turn_v8.candidates.jsonl"
    inventory.write_text("{}\n", encoding="utf-8")
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text("{}\n", encoding="utf-8")
    contract = {
        "duration_policy": runner.NATURAL_DURATION_POLICY,
        "fixed_window_sec": None,
        "semantic_supervision_masks": dict(runner.SEMANTIC_MASKS),
        "emotion_supervision_mask": False,
        "builder_script_sha256": ordinary.sha256(runner.BUILDER_SCRIPT),
        "source_manifest_sha256": ordinary.sha256(source),
        "split_assignment": {"manifest_sha256": ordinary.sha256(split)},
    }
    summary = {
        "artifact_kind": "beat2_expression_turn_v8_candidate_catalog",
        "expression_turn_contract": contract,
        "expression_turn_contract_sha256": ordinary.json_sha256(contract),
        "input": str(source),
        "input_sha256": ordinary.sha256(source),
        "split_assignment": str(split),
        "split_assignment_sha256": ordinary.sha256(split),
        "output_sha256": {inventory.name: ordinary.sha256(inventory)},
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    binding, audit = runner.load_catalog_binding(
        inventory, summary_path, "full_pool"
    )
    assert binding["selection_kind"] is None
    assert binding["require_selection_record"] is False
    assert audit["execution_selection_kind"] == "full_pool"
    assert audit["full_pool_candidate_manifest"] is True

    sampled = tmp_path / "sampled.jsonl"
    sampled.write_text("{}\n", encoding="utf-8")
    summary["output_sha256"][sampled.name] = ordinary.sha256(sampled)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="full_pool must use"):
        runner.load_catalog_binding(sampled, summary_path, "full_pool")


def test_cli_has_selection_kind_and_no_fixed_duration_controls():
    args = runner.parse_args(
        ["--selection-kind", "stress100", "--workers", "1"]
    )
    assert isinstance(args, Namespace)
    assert args.selection_kind == "stress100"
    assert runner.parse_args(["--selection-kind", "full_pool"]).selection_kind == (
        "full_pool"
    )
    with pytest.raises(SystemExit):
        runner.parse_args(["--fixed-window", "6"])


def test_retry_selector_only_admits_obsolete_frame_contract_failures(tmp_path):
    root = tmp_path / "diagnostic"
    root.mkdir()
    inventory_hash = "a" * 64
    tasks = []
    for index, prior_status in enumerate(("passed", "retry", "physical"), 1):
        task = {
            **_record("b" * 64),
            "task_id": f"turn-{index}",
            "clip_id": f"turn-{index}",
            "source": str(tmp_path / f"source-{index}.npz"),
            "start_frame": 90,
            "end_frame_exclusive": 120,
            "inventory_record_sha256": f"{index}" * 64,
            "upstream_inventory_record_sha256": "2" * 64,
            "selected_record_sha256": "8" * 64,
            "retarget_input_manifest_sha256": inventory_hash,
        }
        tasks.append(task)
        result = {
            "task_id": task["task_id"],
            "status": "passed" if prior_status == "passed" else "quality_failed",
            "inventory_record_sha256": task["inventory_record_sha256"],
            "upstream_inventory_record_sha256": task[
                "upstream_inventory_record_sha256"
            ],
            "selected_record_sha256": task["selected_record_sha256"],
            "retarget_input_manifest_sha256": inventory_hash,
        }
        if prior_status != "passed":
            output_dir = root / "failed" / task["task_id"]
            output_dir.mkdir(parents=True)
            gates = {
                "joint_limits_pass": True,
                "velocity_pass": True,
                "target_fit_pass": prior_status == "retry",
                "collision_pass": True,
                "passed": prior_status == "retry",
            }
            quality = {
                "quality_gate": gates,
                "expression_turn_output_contract_validation": {
                    "passed": False,
                    "status": "failed_prevalidation",
                },
                "source_window_frames": 30,
                "frames": 33,
                "retarget_segment": {
                    "source_frame_count": 30,
                    "output_frame_count": 33,
                    "retimed": True,
                    "cropped": False,
                    "duration_policy": runner.NATURAL_DURATION_POLICY,
                },
            }
            quality_path = output_dir / "quality.json"
            quality_path.write_text(json.dumps(quality), encoding="utf-8")
            result.update(
                output_dir=str(output_dir),
                quality_json_sha256=ordinary.sha256(quality_path),
            )
        result_path = ordinary.result_path(root, task)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result), encoding="utf-8")

    prior_contract = {"contract": "obsolete-native-frame"}
    prior_contract_hash = ordinary.json_sha256(prior_contract)
    (root / ordinary.RUN_CONTRACT_FILENAME).write_text(
        json.dumps(
            {
                "run_contract": prior_contract,
                "run_contract_sha256": prior_contract_hash,
            }
        ),
        encoding="utf-8",
    )
    (root / "status.json").write_text(
        json.dumps(
            {
                "run_state": "finished",
                "inventory_sha256": inventory_hash,
                "selection_kind": "representative100",
                "accepted_for_training": False,
                "terminal_turn_count": 3,
                "pending_turn_count": 0,
                "run_contract": prior_contract,
                "run_contract_sha256": prior_contract_hash,
                "execution_policy": {
                    "preserve_native_frame_count": True,
                    "fixed_duration_windows_allowed": False,
                    "natural_training_segment_only": True,
                },
            }
        ),
        encoding="utf-8",
    )
    for name in (
        "passed_manifest.jsonl",
        "failed_manifest.jsonl",
        "pending_manifest.jsonl",
        "excluded_manifest.jsonl",
    ):
        (root / name).write_text("", encoding="utf-8")

    selected, excluded, audit = runner.select_safety_retime_retry_tasks(
        tasks,
        root,
        inventory_hash=inventory_hash,
        selection_kind="representative100",
    )

    assert [task["task_id"] for task in selected] == ["turn-2"]
    assert len(excluded) == 2
    assert audit["selected_retry_count"] == 1
    assert audit["exclusion_counts"] == {
        "prior_native_frame_output_passed_no_retry": 1,
        "prior_true_physical_quality_failure_no_retry": 1,
    }
