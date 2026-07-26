import json
from pathlib import Path

import pytest

from tools.gmr_v2 import (
    batch_retarget_render_beat2_expression_turn_expansions_v8 as runner,
)
from tools.gmr_v2 import batch_retarget_beat2_v2 as ordinary
from upper_body_skeleton.retarget_v2_18d import JOINT_ORDER_18D


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(ordinary.stable_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _candidate():
    base_hash = "a" * 64
    return {
        "artifact_kind": runner.base.INPUT_ARTIFACT_KIND,
        "clip_id": "turn-001",
        "task_id": "turn-001",
        "source_clip_id": "source-001",
        "fixed_split_assignment": "train",
        "fps": 30.0,
        "representation": runner.base.INPUT_REPRESENTATION,
        "motion_relpath": "motion/source-001.npz",
        "motion_sha256": "b" * 64,
        "source_inventory_manifest_sha256": "c" * 64,
        "split_assignment_manifest_sha256": "d" * 64,
        "expression_turn_contract_sha256": "e" * 64,
        "expression_turn_record_sha256": base_hash,
        "context_plan": {
            "policy": runner.CONTEXT_POLICY,
            "selected_level": 0,
            "levels": [
                {
                    "level": 0,
                    "start_frame": 20,
                    "end_frame_exclusive": 50,
                    "source_start_sec": 0.666667,
                    "source_end_sec": 1.666667,
                    "left_rest_score_rad_s": 0.1,
                    "right_rest_score_rad_s": 0.1,
                },
                {
                    "level": 1,
                    "parent_level": 0,
                    "start_frame": 10,
                    "end_frame_exclusive": 70,
                    "source_start_sec": 0.333333,
                    "source_end_sec": 2.333333,
                    "left_rest_score_rad_s": 0.08,
                    "right_rest_score_rad_s": 0.09,
                },
                {
                    "level": 2,
                    "parent_level": 1,
                    "start_frame": 0,
                    "end_frame_exclusive": 90,
                    "source_start_sec": 0.0,
                    "source_end_sec": 3.0,
                    "left_rest_score_rad_s": 0.07,
                    "right_rest_score_rad_s": 0.08,
                },
            ],
        },
        "training_segment": {
            "representation": runner.base.INPUT_REPRESENTATION,
            "start_frame": 20,
            "end_frame_exclusive": 50,
            "frame_count": 30,
            "fixed_window_sec": None,
            "cropped": False,
            "duration_policy": runner.base.NATURAL_DURATION_POLICY,
        },
        "time_axes": {},
        "expression_turn": {
            "included_event_spans": [
                {
                    "source_time_axis": {
                        "start_sec": 1.0,
                        "end_sec": 1.3,
                        "start_frame_floor": 30,
                        "end_frame_exclusive_ceil": 39,
                    },
                    "turn_time_axis": {},
                }
            ],
            "peak": {
                "source_frame": 35,
                "source_sec": 1.166667,
                "turn_frame": 15,
                "turn_sec": 0.5,
                "energy_rad_s": 1.0,
                "prominence_over_boundaries": 10.0,
            },
        },
        "window": {"stale_metric": 123},
        "duration_band": "short_under_3s",
        "semantic_supervision_masks": dict(runner.base.SEMANTIC_MASKS),
        "emotion_supervision_mask": False,
        "affect_observable_supervision_mask": False,
        "official_emotion_conditioning_enabled": False,
        "official_category_conditioning_enabled": False,
        "canonical_prompt": None,
        "canonical_action": None,
        "accepted_for_training": False,
    }


def _request():
    record = {
        "artifact_kind": runner.REQUEST_ARTIFACT_KIND,
        "sample_id": "expr-001",
        "base_task_id": "turn-001",
        "source_clip_id": "source-001",
        "fixed_split_assignment": "train",
        "base_expression_turn_record_sha256": "a" * 64,
        "comparison_record_sha256": "f" * 64,
        "reviewed_context_level": 0,
        "requested_context_level": 1,
        "reviewed_interval": {
            "start_frame": 20,
            "end_frame_exclusive": 50,
            "frame_count": 30,
        },
        "requested_interval": {
            "start_frame": 10,
            "end_frame_exclusive": 70,
            "frame_count": 60,
        },
        "strictly_contains_reviewed_interval": True,
        "expansion_unit": runner.EXPANSION_UNIT,
        "elapsed_duration_used_as_gate": False,
        "semantic_supervision_mask": False,
        "emotion_supervision_mask": False,
        "accepted_for_training": False,
    }
    record["plan_record_sha256"] = ordinary.json_sha256(record)
    return record


def test_derivation_advances_one_declared_level_without_duration_selection():
    expanded = runner.derive_expanded_candidate(
        _candidate(),
        _request(),
        plan_summary_sha256="1" * 64,
        expansion_requests_sha256="2" * 64,
        hidden_mapping_sha256="3" * 64,
        candidate_catalog_sha256="4" * 64,
    )
    assert expanded["task_id"] == "turn-001__ctxL01"
    assert expanded["context_plan"]["selected_level"] == 1
    assert expanded["training_segment"] == {
        "representation": runner.base.INPUT_REPRESENTATION,
        "start_frame": 10,
        "end_frame_exclusive": 70,
        "frame_count": 60,
        "fixed_window_sec": None,
        "cropped": False,
        "duration_policy": runner.base.NATURAL_DURATION_POLICY,
    }
    assert expanded["window"]["motion_metric_role"].startswith("not_reused")
    assert "stale_metric" not in expanded["window"]
    assert expanded["time_axes"]["turn"]["end_frame_exclusive"] == 60
    assert expanded["expression_turn"]["peak"]["turn_frame"] == 25
    assert expanded["expression_turn_record_sha256"] == ordinary.json_sha256(
        {
            key: value
            for key, value in expanded.items()
            if key != "expression_turn_record_sha256"
        }
    )
    provenance = expanded["expansion_provenance"]
    assert provenance["elapsed_duration_used_as_gate"] is False
    assert provenance["license_training_admission"] is False
    assert provenance["accepted_for_training"] is False


def test_derivation_accepts_only_bound_round_n_continuation_lineage():
    request = _request()
    request.update(
        {
            "reviewed_context_level": 1,
            "requested_context_level": 2,
            "reviewed_interval": {
                "start_frame": 10,
                "end_frame_exclusive": 70,
                "frame_count": 60,
            },
            "requested_interval": {
                "start_frame": 0,
                "end_frame_exclusive": 90,
                "frame_count": 90,
            },
            "continuation_lineage": {
                "role": "round_n_natural_boundary_continuation",
                "previous_reviewed_context_level": 0,
                "previous_requested_context_level": 1,
                "previous_plan_record_sha256": "5" * 64,
                "previous_plan_summary_sha256": "6" * 64,
                "previous_expansion_requests_sha256": "7" * 64,
                "canonical_base_sample_id_derived_from_unique_base_task": "expr-001",
            },
        }
    )
    request["plan_record_sha256"] = ordinary.json_sha256(
        {key: value for key, value in request.items() if key != "plan_record_sha256"}
    )
    expanded = runner.derive_expanded_candidate(
        _candidate(),
        request,
        plan_summary_sha256="1" * 64,
        expansion_requests_sha256="2" * 64,
        hidden_mapping_sha256="3" * 64,
        candidate_catalog_sha256="4" * 64,
    )
    assert expanded["task_id"] == "turn-001__ctxL02"
    assert expanded["context_plan"]["selected_level"] == 2

    request["continuation_lineage"]["previous_requested_context_level"] = 0
    request["plan_record_sha256"] = ordinary.json_sha256(
        {key: value for key, value in request.items() if key != "plan_record_sha256"}
    )
    with pytest.raises(ValueError, match="without valid continuation lineage"):
        runner.derive_expanded_candidate(
            _candidate(),
            request,
            plan_summary_sha256="1" * 64,
            expansion_requests_sha256="2" * 64,
            hidden_mapping_sha256="3" * 64,
            candidate_catalog_sha256="4" * 64,
        )


def test_request_rejects_jump_or_hash_tampering():
    jumped = _request()
    jumped["requested_context_level"] = 2
    jumped["plan_record_sha256"] = ordinary.json_sha256(
        {key: value for key, value in jumped.items() if key != "plan_record_sha256"}
    )
    with pytest.raises(ValueError, match="exactly one level"):
        runner._validate_request(jumped)

    tampered = _request()
    tampered["requested_interval"]["start_frame"] = 9
    with pytest.raises(ValueError, match="Record SHA mismatch"):
        runner._validate_request(tampered)


def test_loader_checks_plan_mapping_catalog_bindings(tmp_path, monkeypatch):
    request = _request()
    candidate = _candidate()
    requests = tmp_path / "expansion_requests.jsonl"
    mapping = tmp_path / "sample_mapping.jsonl"
    catalog = tmp_path / "beat2_expression_turn_v8.candidates.jsonl"
    catalog_summary = tmp_path / "catalog_summary.json"
    plan_summary = tmp_path / "summary.json"
    _write_jsonl(requests, [request])
    _write_jsonl(
        mapping,
        [
            {
                "sample_id": "expr-001",
                "task_id": "turn-001",
                "source_clip_id": "source-001",
                "fixed_split_assignment": "train",
                "expression_turn_record_sha256": "a" * 64,
            }
        ],
    )
    _write_jsonl(catalog, [candidate])
    catalog_summary.write_text("{}\n", encoding="utf-8")
    plan = {
        "artifact_kind": runner.PLAN_ARTIFACT_KIND,
        "selection_policy": runner.SELECTION_POLICY,
        "fixed_minimum_maximum_or_target_duration_used": False,
        "accepted_for_training_count": 0,
        "inputs": {
            "hidden_mapping_sha256": ordinary.sha256(mapping),
            "candidate_catalog_sha256": ordinary.sha256(catalog),
        },
        "outputs": {
            "expansion_requests": {
                "path": str(requests.resolve()),
                "sha256": ordinary.sha256(requests),
                "records": 1,
            }
        },
    }
    plan_summary.write_text(json.dumps(plan), encoding="utf-8")
    binding = {
        "retarget_input_manifest_sha256": ordinary.sha256(catalog),
        "expression_turn_contract_sha256": "e" * 64,
        "source_inventory_manifest_sha256": "c" * 64,
        "split_assignment_manifest_sha256": "d" * 64,
        "selection_kind": None,
        "require_selection_record": False,
    }
    monkeypatch.setattr(
        runner.base,
        "load_catalog_binding",
        lambda *_args: (
            binding,
            {"catalog_candidate_manifest_sha256": ordinary.sha256(catalog)},
        ),
    )
    validated = []
    monkeypatch.setattr(
        runner,
        "validate_expression_turn_candidate",
        lambda row, *, catalog_binding: validated.append(
            (row["task_id"], catalog_binding)
        ),
    )
    rows, audit = runner.load_and_derive_candidates(
        expansion_requests=requests,
        expansion_plan_summary=plan_summary,
        hidden_mapping=mapping,
        candidate_catalog=catalog,
        catalog_summary=catalog_summary,
    )
    assert len(rows) == audit["records"] == 1
    assert validated == [("turn-001", binding)]
    assert audit["fixed_minimum_maximum_or_target_duration_used"] is False
    assert audit["license_training_admission"] is False

    hidden = json.loads(mapping.read_text())
    hidden["source_clip_id"] = "wrong-source"
    _write_jsonl(mapping, [hidden])
    plan["inputs"]["hidden_mapping_sha256"] = ordinary.sha256(mapping)
    plan_summary.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="hidden mapping mismatch"):
        runner.load_and_derive_candidates(
            expansion_requests=requests,
            expansion_plan_summary=plan_summary,
            hidden_mapping=mapping,
            candidate_catalog=catalog,
            catalog_summary=catalog_summary,
        )


def test_render_queue_remains_physical_only(tmp_path, monkeypatch):
    task = runner.derive_expanded_candidate(
        _candidate(),
        _request(),
        plan_summary_sha256="1" * 64,
        expansion_requests_sha256="2" * 64,
        hidden_mapping_sha256="3" * 64,
        candidate_catalog_sha256="4" * 64,
    )
    task.update(
        source=str(tmp_path / "source.npz"),
        inventory_record_sha256="5" * 64,
        upstream_inventory_record_sha256="6" * 64,
        selected_record_sha256="7" * 64,
        retarget_input_manifest_sha256="8" * 64,
        speaker_key="speaker-a",
        official_split="train",
    )
    safe = tmp_path / "safe.csv"
    safe.write_text(
        ",".join(JOINT_ORDER_18D)
        + "\n"
        + "".join(",".join(["0"] * 18) + "\n" for _ in range(61)),
        encoding="utf-8",
    )
    retarget_segment = {
        "representation": runner.base.RETARGET_SEGMENT_REPRESENTATION,
        "source_frame_count": 60,
        "output_frame_count": 61,
        "source_frame_coverage_sec": 2.0,
        "output_frame_coverage_sec": 61 / 30,
        "output_sample_span_sec": 2.0,
        "fps": 30.0,
        "retimed": True,
        "cropped": False,
        "duration_policy": runner.base.NATURAL_DURATION_POLICY,
        "fixed_target_duration_sec": None,
    }
    safety = {
        "artifact_kind": "ula_18d_safety_monotonic_retime_v1",
        "blind_review_must_use_retimed_output": True,
        "source_frame_count": 60,
        "output_frame_count": 61,
        "minimum_output_frame_count": 61,
        "time_map_strictly_increasing": True,
        "first_frame_preserved": True,
        "last_frame_preserved": True,
        "post_velocity_pass": True,
        "slowdown_ratio_pass": True,
        "cropped": False,
        "tiled": False,
        "target_duration_sec": None,
        "retime_ratio": 61 / 60,
        "max_slowdown_ratio": 1.25,
        "input_frame_output_times_sec": [index / 30 for index in range(60)],
    }
    quality = {
        "quality_gate": {"joint_limits_pass": True, "passed": True},
        "retarget_segment": retarget_segment,
        "safety_monotonic_retime": safety,
    }
    quality_path = tmp_path / "quality.json"
    quality_path.write_text(json.dumps(quality), encoding="utf-8")
    contract = {"contract": "current-expansion-physical-run"}
    contract_hash = ordinary.json_sha256(contract)
    retarget_root = tmp_path / "retarget"
    retarget_root.mkdir()
    (retarget_root / ordinary.RUN_CONTRACT_FILENAME).write_text(
        json.dumps(
            {"run_contract": contract, "run_contract_sha256": contract_hash}
        ),
        encoding="utf-8",
    )
    (retarget_root / "status.json").write_text(
        json.dumps(
            {
                "run_contract": contract,
                "run_contract_sha256": contract_hash,
                "processing_scope": runner.PROCESSING_SCOPE,
                "semantic_admission": False,
                "affect_admission": False,
                "license_training_admission": False,
                "accepted_for_training": False,
            }
        ),
        encoding="utf-8",
    )
    result = {
        "status": "passed",
        "run_contract_sha256": contract_hash,
        "quality_json": str(quality_path),
        "quality_json_sha256": ordinary.sha256(quality_path),
        "safe_csv": str(safe),
        "safe_csv_sha256": ordinary.sha256(safe),
        "frames": 61,
        "fps": 30.0,
        "retarget_segment": retarget_segment,
    }
    result_path = ordinary.result_path(retarget_root, task)
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(result), encoding="utf-8")
    monkeypatch.setattr(runner.ordinary, "load_result", lambda _path: result)
    monkeypatch.setattr(
        runner, "expansion_pass_is_current", lambda *_args, **_kwargs: True
    )
    output = tmp_path / "queue.jsonl"
    summary = runner.build_render_queue(
        tasks=[task],
        validator_binding={},
        retarget_root=retarget_root,
        output_path=output,
        expected_run_contract_sha256=contract_hash,
    )
    queued = json.loads(output.read_text())
    assert summary["records"] == 1
    assert queued["context_plan"]["selected_level"] == 1
    assert queued["training_segment"]["fixed_window_sec"] is None
    assert queued["source_frame_count"] == 60
    assert queued["output_frame_count"] == 61
    assert queued["trajectory_frames_expected"] == 61
    assert queued["final_trajectory_role"] == "safety_monotonic_retimed_final_output"
    assert queued["blind_review_must_use_final_trajectory"] is True
    assert queued["safety_monotonic_retime"]["source_frame_count"] == 60
    assert queued["semantic_supervision_mask"] is False
    assert queued["emotion_supervision_mask"] is False
    assert queued["affect_observable_supervision_mask"] is False
    assert queued["license_training_admission"] is False
    assert queued["render_pass_grants_training_admission"] is False
    assert queued["accepted_for_training"] is False
    with pytest.raises(ValueError, match="current closed run contract"):
        runner.build_render_queue(
            tasks=[task],
            validator_binding={},
            retarget_root=retarget_root,
            output_path=tmp_path / "stale-queue.jsonl",
            expected_run_contract_sha256="0" * 64,
        )


def test_disk_sorted_safety_metric_maps_are_restored_to_joint_order():
    sorted_map = {joint: float(index) for index, joint in enumerate(sorted(runner.JOINT_ORDER_18D))}
    quality = {
        "safety_monotonic_retime": {
            field: dict(sorted_map) for field in runner.SAFETY_ORDERED_METRIC_FIELDS
        }
    }
    canonical = runner._canonicalize_safety_metric_key_order(quality)
    for field in runner.SAFETY_ORDERED_METRIC_FIELDS:
        assert list(canonical["safety_monotonic_retime"][field]) == list(
            runner.JOINT_ORDER_18D
        )
    assert list(quality["safety_monotonic_retime"][field]) == sorted(
        runner.JOINT_ORDER_18D
    )


def test_cli_exposes_stages_but_no_fixed_duration_control():
    args = runner.parse_args(
        [
            "--expansion-requests",
            "requests.jsonl",
            "--expansion-plan-summary",
            "summary.json",
            "--hidden-mapping",
            "mapping.jsonl",
            "--output-root",
            "out",
            "--stage",
            "render",
        ]
    )
    assert args.stage == "render"
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--expansion-requests",
                "requests.jsonl",
                "--expansion-plan-summary",
                "summary.json",
                "--hidden-mapping",
                "mapping.jsonl",
                "--output-root",
                "out",
                "--fixed-window",
                "6",
            ]
        )
