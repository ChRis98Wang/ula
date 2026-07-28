from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from tools import supervise_hanyang_beat2_emotion_preserving_v81 as supervisor


ROOT = Path(__file__).resolve().parents[1]


def _tmp_config(tmp_path: Path) -> dict:
    return {
        "_config_path": str(tmp_path / "supervisor.json"),
        "_config_sha256": "a" * 64,
        "state": str(tmp_path / "state.json"),
        "receipt": str(tmp_path / "receipt.json"),
        "selected_winner_receipt": str(tmp_path / "winner.json"),
        "qwen_b_training_service_unit": "qwen.service",
        "video_waiter_service_unit": "video.service",
        "video_summary": str(tmp_path / "video-summary.json"),
        "video_mp4": str(tmp_path / supervisor.VIDEO_FILENAME),
        "base_blocked_config": str(tmp_path / "blocked.json"),
        "promoted_config": str(tmp_path / "approved.json"),
        "approval_receipt": str(tmp_path / "approval.json"),
        "control_smoke_output_dir": str(tmp_path / "smoke-control"),
        "treatment_smoke_output_dir": str(tmp_path / "smoke-treatment"),
        "python_executable": "/python",
        "trainer": "/trainer.py",
        "promoter": "/promoter.py",
        "gpu": {
            "index": 0,
            "minimum_free_memory_mib": 24576,
            "unknown_compute_policy": "record_never_kill",
        },
        "data_contract": {
            "hanyang_strict_count": 344,
            "hanyang_boundary_admitted_count": 0,
            "kimodo_admitted_count": 0,
            "formal_arms_sequential": True,
            "arm_initialization": (
                "same_selected_winner_independent_no_cross_arm_warm_start"
            ),
        },
    }


def _identity():
    return {
        "approved_by": "reviewer",
        "approved_utc": "2026-07-27T09:00:00Z",
        "decision_notes": "night pair approved after prerequisites",
    }


def _success_unit(unit: str, **_kwargs):
    return {
        "unit": unit,
        "available": True,
        "failed": False,
        "running": False,
        "succeeded_and_exited": True,
    }


def _winner(_path):
    return {
        "ready": True,
        "file_sha256": "b" * 64,
        "selected_qwen_variant": "frozen_base",
    }


def _gpu(free=26000.0, pids=None):
    return {
        "gpu_index": 0,
        "free_memory_mib": free,
        "minimum_free_memory_mib": 24576.0,
        "compute_processes": [
            {
                "pid": pid,
                "process_name": "other-RL",
                "used_memory_mib": 5000.0,
            }
            for pid in (pids or [])
        ],
        "unknown_compute_pids": list(pids or []),
        "memory_threshold_passed": free >= 24576,
        "ready": free >= 24576,
        "policy": "unknown_compute_record_only_never_kill",
    }


def _patch_happy(monkeypatch, calls, *, interrupt_stage=None):
    monkeypatch.setattr(supervisor, "_selected_receipt_snapshot", _winner)
    monkeypatch.setattr(supervisor, "_unit_snapshot", _success_unit)
    monkeypatch.setattr(
        supervisor,
        "_validate_archived_qwen_b_completion",
        lambda _path: {"status": "complete", "summary_sha256": "9" * 64},
    )
    monkeypatch.setattr(
        supervisor,
        "_validate_video",
        lambda _config: {"summary_sha256": "c" * 64, "video_sha256": "d" * 64},
    )
    monkeypatch.setattr(
        supervisor,
        "_run_promotion",
        lambda _config, _identity, **_kwargs: calls.append("promotion")
        or {"promoted_config_sha256": "e" * 64},
    )
    gpu_values = iter([_gpu(pids=[2238037])] * 16)
    monkeypatch.setattr(
        supervisor, "_gpu_snapshot", lambda *_args, **_kwargs: next(gpu_values)
    )

    interrupted = {"done": False}

    def run_arm(_config, *, arm, smoke, **_kwargs):
        stage = (
            ("control" if arm == supervisor.ARMS[0] else "treatment")
            + ("_smoke" if smoke else "_formal")
        )
        calls.append(stage)
        if stage == interrupt_stage and not interrupted["done"]:
            interrupted["done"] = True
            raise KeyboardInterrupt("synthetic process interruption")
        return {"arm": arm, "smoke_test": smoke, "summary_sha256": "f" * 64}

    monkeypatch.setattr(supervisor, "_run_arm", run_arm)

    def revalidate(_config, _identity, stage):
        calls.append("revalidate:" + stage)
        return {"stage": stage, "revalidated": True}

    monkeypatch.setattr(supervisor, "_revalidate_success", revalidate)


def test_checked_supervisor_config_is_explicit_and_safe():
    values = supervisor.read_config(
        ROOT / "configs/hanyang_beat2_emotion_preserving_v81_night_supervisor.json"
    )
    assert values["gpu"]["minimum_free_memory_mib"] == 24576
    assert values["gpu"]["unknown_compute_policy"] == "record_never_kill"
    assert values["data_contract"] == {
        "hanyang_strict_count": 344,
        "hanyang_boundary_admitted_count": 0,
        "kimodo_admitted_count": 0,
        "formal_arms_sequential": True,
        "arm_initialization": (
            "same_selected_winner_independent_no_cross_arm_warm_start"
        ),
    }
    assert Path(values["control_smoke_output_dir"]) != Path(
        values["treatment_smoke_output_dir"]
    )


def test_retry_config_uses_fresh_state_receipt_and_derived_lock():
    original = supervisor.read_config(
        ROOT / "configs/hanyang_beat2_emotion_preserving_v81_night_supervisor.json"
    )
    retry = supervisor.read_config(
        ROOT
        / "configs/hanyang_beat2_emotion_preserving_v81_night_supervisor_retry1.json"
    )
    retry2 = supervisor.read_config(
        ROOT
        / "configs/hanyang_beat2_emotion_preserving_v81_night_supervisor_retry2.json"
    )
    assert retry["state"] != original["state"]
    assert retry["receipt"] != original["receipt"]
    assert retry["state"] + ".lock" != original["state"] + ".lock"
    assert retry2["state"] not in {original["state"], retry["state"]}
    assert retry2["receipt"] not in {original["receipt"], retry["receipt"]}
    assert retry2["state"] + ".lock" not in {
        original["state"] + ".lock",
        retry["state"] + ".lock",
    }
    for field in (
        "promoted_config",
        "approval_receipt",
        "control_smoke_output_dir",
        "treatment_smoke_output_dir",
    ):
        assert retry2[field] != retry[field]
    assert (
        retry["data_contract"]["arm_initialization"]
        == "same_selected_winner_independent_no_cross_arm_warm_start"
    )


def test_transient_unit_not_found_is_observable_but_not_success():
    output = "\n".join(
        (
            "Result=success",
            "ExecMainStatus=0",
            "LoadState=not-found",
            "ActiveState=inactive",
            "SubState=dead",
        )
    )

    def runner(argv, **_kwargs):
        return subprocess.CompletedProcess(argv, 0, output, "")

    snapshot = supervisor._unit_snapshot(
        "collected.service",
        command_runner=runner,
        allow_transient_reclaimed=True,
    )
    assert snapshot["available"] is False
    assert snapshot["transient_unit_reclaimed"] is True
    assert snapshot["succeeded_and_exited"] is False
    with pytest.raises(supervisor.SupervisorError, match="unavailable"):
        supervisor._unit_snapshot(
            "collected.service",
            command_runner=runner,
        )


def test_reclaimed_qwen_requires_hash_valid_b_archive(tmp_path, monkeypatch):
    selected_receipt = tmp_path / "selected.json"
    selected_receipt.write_text("{}\n", encoding="utf-8")
    summary = tmp_path / "training_summary_v7.json"
    checkpoint = tmp_path / "generator_emotion_hierarchy_v7.pt"
    summary.write_bytes(b"sealed summary")
    checkpoint.write_bytes(b"sealed checkpoint")
    arm = {
        "variant": supervisor.selector.LORA_VARIANT,
        "completed": True,
        "summary": {
            "path": str(summary),
            "file_sha256": supervisor.sha256_file(summary),
            "status": "experimental_candidate",
            "completed_steps": supervisor.selector.EXPECTED_STEPS,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": supervisor.sha256_file(checkpoint),
            "global_step": 347000,
        },
        "anti_collapse_gate": {"passed": True},
        "no_external_data": True,
        "no_kimodo": True,
        "no_hanyang": True,
    }
    checked = {
        "sha256": "1" * 64,
        "arms": {supervisor.selector.LORA_VARIANT: arm},
        "sealed_ab_receipt": {
            "path": str(tmp_path / "sealed.json"),
            "file_sha256": "2" * 64,
        },
    }
    monkeypatch.setattr(
        supervisor.selector,
        "validate_selection_receipt",
        lambda _receipt, *, require_selected: checked,
    )
    monkeypatch.setattr(
        supervisor.selector,
        "validate_sealed_ab_receipt",
        lambda *_args, **_kwargs: {"sealed": True},
    )
    monkeypatch.setattr(
        supervisor.selector,
        "_validate_arm",
        lambda **_kwargs: dict(arm),
    )
    record = supervisor._validate_archived_qwen_b_completion(
        selected_receipt
    )
    assert record["completed_steps"] == supervisor.selector.EXPECTED_STEPS
    assert record["checkpoint_sha256"] == supervisor.sha256_file(checkpoint)
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(supervisor.SupervisorError, match="archived Qwen B"):
        supervisor._validate_archived_qwen_b_completion(selected_receipt)


def test_reclaimed_video_unit_requires_validated_video_artifacts(
    tmp_path, monkeypatch
):
    config = _tmp_config(tmp_path)
    reclaimed = {
        "unit": "video.service",
        "available": False,
        "transient_unit_reclaimed": True,
        "failed": False,
        "succeeded_and_exited": False,
    }
    monkeypatch.setattr(
        supervisor,
        "_validate_video",
        lambda _config: {
            "summary_sha256": "3" * 64,
            "video_sha256": "4" * 64,
            "duration_sec": 60.0,
        },
    )
    completion = supervisor._video_completion_snapshot(config, reclaimed)
    assert completion["ready"] is True
    assert completion["completion_evidence"].startswith(
        "transient_unit_reclaimed"
    )


def test_gpu_gate_records_unknown_pid_but_memory_controls_readiness(monkeypatch):
    outputs = iter(
        [
            subprocess.CompletedProcess(["nvidia-smi"], 0, "26000\n", ""),
            subprocess.CompletedProcess(
                ["nvidia-smi"], 0, "2238037, python, 5000\n", ""
            ),
            subprocess.CompletedProcess(["nvidia-smi"], 0, "23000\n", ""),
            subprocess.CompletedProcess(["nvidia-smi"], 0, "\n", ""),
        ]
    )

    def runner(argv, **_kwargs):
        assert isinstance(argv, list)
        return next(outputs)

    config = _tmp_config(Path("/tmp/supervisor-test"))
    busy_but_safe = supervisor._gpu_snapshot(config, command_runner=runner)
    assert busy_but_safe["ready"] is True
    assert busy_but_safe["unknown_compute_pids"] == [2238037]
    low = supervisor._gpu_snapshot(config, command_runner=runner)
    assert low["ready"] is False


def test_supervisor_orders_all_stages_and_requires_two_gpu_polls(
    tmp_path, monkeypatch
):
    config = _tmp_config(tmp_path)
    calls = []
    _patch_happy(monkeypatch, calls)
    sleeps = []
    receipt = supervisor.run_supervisor(
        config,
        **_identity(),
        poll_seconds=0.01,
        command_runner=lambda *_args, **_kwargs: None,
        sleeper=sleeps.append,
    )
    assert calls == [
        "promotion",
        "control_smoke",
        "treatment_smoke",
        "control_formal",
        "treatment_formal",
    ]
    assert len(sleeps) == 4
    assert receipt["status"] == "complete"
    assert list(receipt["stages"]) == list(supervisor.STAGES)
    for stage in supervisor.STAGES[2:]:
        gate = receipt["stages"][stage]["record"]["prelaunch_gpu_gate"]
        assert len(gate["observations"]) == 2
        assert gate["unknown_compute_policy"] == "record_only_never_kill"


def test_interrupted_stage_restarts_without_redoing_successes(
    tmp_path, monkeypatch
):
    config = _tmp_config(tmp_path)
    calls = []
    _patch_happy(monkeypatch, calls, interrupt_stage="treatment_smoke")
    with pytest.raises(KeyboardInterrupt, match="synthetic"):
        supervisor.run_supervisor(
            config,
            **_identity(),
            poll_seconds=0.01,
            command_runner=lambda *_args, **_kwargs: None,
            sleeper=lambda _seconds: None,
        )
    state = supervisor._read_json(config["state"], context="state")
    assert state["stages"]["treatment_smoke"]["status"] == "running"
    receipt = supervisor.run_supervisor(
        config,
        **_identity(),
        poll_seconds=0.01,
        command_runner=lambda *_args, **_kwargs: None,
        sleeper=lambda _seconds: None,
    )
    assert receipt["status"] == "complete"
    assert calls.count("promotion") == 1
    assert calls.count("control_smoke") == 1
    assert calls.count("treatment_smoke") == 2
    assert "revalidate:promotion" in calls
    assert "revalidate:control_smoke" in calls


def test_terminal_failure_stops_and_cannot_skip(tmp_path, monkeypatch):
    config = _tmp_config(tmp_path)
    calls = []
    monkeypatch.setattr(supervisor, "_selected_receipt_snapshot", _winner)
    monkeypatch.setattr(supervisor, "_unit_snapshot", _success_unit)
    monkeypatch.setattr(
        supervisor,
        "_validate_archived_qwen_b_completion",
        lambda _path: {"status": "complete"},
    )
    monkeypatch.setattr(supervisor, "_validate_video", lambda _config: {})

    def fail_promotion(*_args, **_kwargs):
        calls.append("promotion")
        raise supervisor.SupervisorError("synthetic promotion failure")

    monkeypatch.setattr(supervisor, "_run_promotion", fail_promotion)
    monkeypatch.setattr(
        supervisor,
        "_run_arm",
        lambda *_args, **_kwargs: calls.append("arm"),
    )
    with pytest.raises(supervisor.SupervisorError, match="synthetic"):
        supervisor.run_supervisor(
            config,
            **_identity(),
            poll_seconds=0.01,
            command_runner=lambda *_args, **_kwargs: None,
            sleeper=lambda _seconds: None,
        )
    state = supervisor._read_json(config["state"], context="state")
    assert state["status"] == "failed"
    assert state["failure"]["stage"] == "promotion"
    assert calls == ["promotion"]
    with pytest.raises(supervisor.SupervisorError, match="terminally failed"):
        supervisor.run_supervisor(
            config,
            **_identity(),
            poll_seconds=0.01,
            command_runner=lambda *_args, **_kwargs: None,
            sleeper=lambda _seconds: None,
        )


def test_runner_uses_argument_list_cuda_resume_and_unique_smoke_dir(
    tmp_path, monkeypatch
):
    config = _tmp_config(tmp_path)
    commands = []
    monkeypatch.setattr(
        supervisor,
        "_summary_path",
        lambda *_args, arm, smoke, **_kwargs: (
            tmp_path / ("control-smoke" if arm == supervisor.ARMS[0] else "other")
            / "training_summary_v8_1.json"
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "_validate_arm_summary",
        lambda *_args, arm, smoke, **_kwargs: {"arm": arm, "smoke_test": smoke},
    )

    def runner(argv, **kwargs):
        commands.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "{}", "")

    record = supervisor._run_arm(
        config,
        arm=supervisor.ARMS[0],
        smoke=True,
        command_runner=runner,
    )
    argv, kwargs = commands[0]
    assert isinstance(argv, list)
    assert "--resume" in argv
    assert argv[argv.index("--device") + 1] == "cuda"
    assert argv[argv.index("--smoke-output-dir") + 1].endswith("control-smoke")
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert record["smoke_test"] is True


def test_promotion_subprocess_receives_explicit_approval_cli(
    tmp_path, monkeypatch
):
    config = _tmp_config(tmp_path)
    commands = []
    monkeypatch.setattr(
        supervisor,
        "_validate_promotion_outputs",
        lambda _config, _identity: {"promoted_config_sha256": "e" * 64},
    )

    def runner(argv, **kwargs):
        commands.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "{}", "")

    record = supervisor._run_promotion(
        config, _identity(), command_runner=runner
    )
    argv, kwargs = commands[0]
    assert isinstance(argv, list)
    assert argv[argv.index("--approved-by") + 1] == "reviewer"
    assert argv[argv.index("--approved-utc") + 1] == "2026-07-27T09:00:00Z"
    assert (
        argv[argv.index("--decision-notes") + 1]
        == "night pair approved after prerequisites"
    )
    assert "shell" not in kwargs
    assert record["argv"] == argv


def test_video_receipt_requires_exact_mp4_hash(tmp_path):
    config = _tmp_config(tmp_path)
    mp4 = Path(config["video_mp4"])
    mp4.write_bytes(b"fake mp4")
    summary = {
        "schema_version": 1,
        "artifact_kind": supervisor.VIDEO_SUMMARY_ARTIFACT_KIND,
        "status": "complete",
        "no_external_data": True,
        "no_kimodo": True,
        "no_hanyang": True,
        "artifacts": {
            "final_video": {
                "path": str(mp4),
                "sha256": supervisor.sha256_file(mp4),
                "duration_sec": 60.0,
            }
        },
    }
    summary["sha256"] = hashlib.sha256(
        json.dumps(
            summary,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    Path(config["video_summary"]).write_text(
        json.dumps(summary), encoding="utf-8"
    )
    assert supervisor._validate_video(config)["duration_sec"] == 60.0
    mp4.write_bytes(b"tampered")
    with pytest.raises(supervisor.SupervisorError, match="MP4"):
        supervisor._validate_video(config)


def test_smoke_summary_is_strictly_noncandidate(tmp_path, monkeypatch):
    config = _tmp_config(tmp_path)
    output = Path(config["control_smoke_output_dir"])
    output.mkdir()
    last = output / "last_generator_v8_1.pt"
    state = output / "last_state_v8_1.pt"
    last.write_bytes(b"last")
    state.write_bytes(b"state")
    summary = {
        "schema_version": "8.1",
        "artifact_kind": supervisor.trainer.SUMMARY_ARTIFACT_KIND,
        "arm": supervisor.ARMS[0],
        "completed_steps": 1,
        "smoke_test": True,
        "run_status": "technical_smoke_completed_not_candidate",
        "formal_release_eligible": False,
        "prefix_schedule_assertion_passed": True,
        "last_checkpoint": str(last),
        "last_checkpoint_sha256": supervisor.sha256_file(last),
        "state": str(state),
        "state_sha256": supervisor.sha256_file(state),
        "exposure": {"beat2": 16, "hanyang": 0, "matched_noop": 0},
        "expected_exposure": {"beat2": 16, "hanyang": 0, "matched_noop": 0},
        "candidate_available": False,
        "best_admissible": None,
    }
    (output / "training_summary_v8_1.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    record = supervisor._validate_arm_summary(
        config, arm=supervisor.ARMS[0], smoke=True
    )
    assert record["run_status"] == "technical_smoke_completed_not_candidate"
    summary["candidate_available"] = True
    (output / "training_summary_v8_1.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    with pytest.raises(supervisor.SupervisorError, match="candidate"):
        supervisor._validate_arm_summary(
            config, arm=supervisor.ARMS[0], smoke=True
        )


def _passing_emotion_gate():
    names = {
        "aligned_vs_zero",
        "aligned_vs_cross_group",
        "flow_gap",
        "q2_recall",
        "q6_recall",
        "global54_recall",
    }
    return {
        "admissible": True,
        "absolute_v7_gate": {"passed": True, "failure_reasons": []},
        "retentions": {name: 0.95 for name in names},
        "retention_thresholds": {name: 0.9 for name in names},
        "retention_checks": {name: True for name in names},
        "hanyang_validation_finite": True,
        "minimum_emotion_retention": 0.95,
        "failure_reasons": [],
    }


@pytest.mark.parametrize(
    "failed_name",
    [
        "aligned_vs_zero",
        "aligned_vs_cross_group",
        "flow_gap",
        "q2_recall",
        "q6_recall",
        "global54_recall",
    ],
)
def test_every_emotion_retention_is_an_independent_hard_gate(failed_name):
    gate = _passing_emotion_gate()
    assert supervisor._validate_emotion_candidate_gate(gate)[
        "absolute_v7_gate_passed"
    ]
    failed = deepcopy(gate)
    failed["retentions"][failed_name] = 0.899
    failed["retention_checks"][failed_name] = False
    failed["minimum_emotion_retention"] = 0.899
    with pytest.raises(supervisor.SupervisorError, match=failed_name):
        supervisor._validate_emotion_candidate_gate(failed)


def test_absolute_v7_and_finite_hanyang_are_hard_gates():
    absolute = _passing_emotion_gate()
    absolute["absolute_v7_gate"]["passed"] = False
    absolute["admissible"] = False
    absolute["failure_reasons"] = ["absolute:aligned_response"]
    with pytest.raises(supervisor.SupervisorError, match="not admissible"):
        supervisor._validate_emotion_candidate_gate(absolute)
    finite = _passing_emotion_gate()
    finite["hanyang_validation_finite"] = False
    finite["admissible"] = False
    finite["failure_reasons"] = ["hanyang_validation_nonfinite"]
    with pytest.raises(supervisor.SupervisorError, match="not admissible"):
        supervisor._validate_emotion_candidate_gate(finite)
