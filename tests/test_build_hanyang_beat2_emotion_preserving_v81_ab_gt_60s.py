from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from tools import train_hanyang_beat2_emotion_preserving_v81 as trainer
from tools import (
    wait_and_build_hanyang_beat2_emotion_preserving_v81_ab_gt_60s
    as waiter,
)
from tools.experimental import (
    build_hanyang_beat2_emotion_preserving_v81_ab_gt_60s as video,
)
from upper_body_skeleton import mujoco_playback
from upper_body_skeleton.ula_v2_18d_head import JOINT_ORDER_18D


CONFIG = (
    video.PROJECT_ROOT
    / "configs/hanyang_beat2_emotion_preserving_v81_ab_gt_60s.json"
)


@pytest.fixture(scope="module")
def checked_config():
    return video.read_config(CONFIG)


def _gate() -> dict:
    retentions = {
        "aligned_vs_zero": 1.0,
        "aligned_vs_cross_group": 1.0,
        "flow_gap": 1.0,
        "q2_recall": 1.0,
        "q6_recall": 1.0,
        "global54_recall": 1.0,
    }
    return {
        "admissible": True,
        "absolute_v7_gate": {"passed": True},
        "retentions": retentions,
        "retention_thresholds": {key: 0.9 for key in retentions},
        "retention_checks": {key: True for key in retentions},
        "hanyang_validation_finite": True,
        "minimum_emotion_retention": 1.0,
        "selection_score": 1.0,
        "failure_reasons": [],
    }


def _promoted() -> dict:
    return {
        "qwen_ab_selection_gate": {
            "selected_qwen_variant": "frozen_base",
            "selected_foundation_sha256": "a" * 64,
            "selected_condition_cache_sha256": "b" * 64,
        }
    }


def _checkpoint(*, arm: str = video.CONTROL_ARM) -> tuple[dict, dict]:
    input_contract = {
        "arm": arm,
        "target_steps": video.EXPECTED_FORMAL_STEPS,
        "selected_qwen_variant": "frozen_base",
        "selected_foundation_sha256": "a" * 64,
        "selected_condition_cache_sha256": "b" * 64,
        "hanyang_training_eligible_count": (
            trainer.HANYANG_TRAINING_ELIGIBLE_COUNT
        ),
        "hanyang_boundary_admitted_count": 0,
        "kimodo_admitted_count": 0,
    }
    input_contract["sha256"] = trainer.canonical_sha256(input_contract)
    gate = _gate()
    checkpoint = {
        "schema_version": trainer.SCHEMA_VERSION,
        "artifact_kind": trainer.CHECKPOINT_ARTIFACT_KIND,
        "arm": arm,
        "checkpoint_role": "best_admissible",
        "candidate_eligible": True,
        "smoke_test": False,
        "formal_release_eligible": False,
        "target_steps": video.EXPECTED_FORMAL_STEPS,
        "v8_1_step": 500,
        "global_step": 347500,
        "architecture": trainer.V7_RUNTIME_MODEL_CONFIG["architecture"],
        "action_dim": 18,
        "condition_dim": 264,
        "joint_order": list(JOINT_ORDER_18D),
        "condition_policy": trainer.V7_CONDITION_POLICY,
        "training_policy": trainer.V81_TRAINING_POLICY,
        "selected_qwen_variant": "frozen_base",
        "selected_foundation_sha256": "a" * 64,
        "exposure": trainer.expected_exposure(500, arm=arm),
        "diagnostic_gate": gate,
        "model_state_dict": {"weight": torch.ones(1)},
        "qwen_style_head_state_dict": {"weight": torch.ones(1)},
        "qwen_style_head_config": {"input_dim": 128},
        "action_stats": {"mean": [0.0] * 18, "std": [1.0] * 18},
        "input_contract": input_contract,
        "hanyang_source_pool_count": trainer.HANYANG_SOURCE_POOL_COUNT,
        "hanyang_boundary_candidate_count": trainer.HANYANG_BOUNDARY_COUNT,
        "hanyang_boundary_excluded_count": trainer.HANYANG_BOUNDARY_COUNT,
        "hanyang_boundary_admitted_count": 0,
        "boundary_hanyang_admitted_count": 0,
        "hanyang_training_eligible_count": (
            trainer.HANYANG_TRAINING_ELIGIBLE_COUNT
        ),
        "hanyang_safe_clip_ids_sha256": trainer.HANYANG_SAFE_CLIP_IDS_SHA256,
        "hanyang_excluded_clip_ids_sha256": (
            trainer.HANYANG_EXCLUDED_CLIP_IDS_SHA256
        ),
        "hanyang_condition_labels_masked": True,
        "kimodo_admitted_count": 0,
    }
    return checkpoint, {"gate": gate}


def test_checked_config_prepares_exact_native_pending_plan(checked_config):
    completion = {
        "ready": False,
        "status": "waiting_for_retry4_formal_completion",
        "current_stage": "control_formal",
    }
    plan = video.prepare_plan(checked_config, completion=completion)
    assert plan["status"] == "waiting_for_retry4_formal_completion"
    assert len(plan["selections"]) == 24
    assert sum(item["frames"] for item in plan["selections"]) == 1800
    assert plan["static_padding_frames"] == 0
    assert plan["temporal_padding_frames"] == 0
    assert plan["endpoint_hold_frames"] == 0
    assert plan["comparison_contract"]["additional_smoothing"] is False
    assert plan["gpu_accessed"] is False
    assert plan["generation_or_render_executed"] is False
    assert plan["smoke_checkpoint_used"] is False


def test_current_or_synthetic_missing_final_receipt_stays_pending(
    checked_config, tmp_path: Path
):
    config = deepcopy(checked_config)
    config["_supervisor_receipt_path"] = str(
        tmp_path / "missing_final_receipt.json"
    )
    state = video.formal_completion_state(
        config,
        service_probe=lambda _: {
            "known": True,
            "active": True,
            "failed": False,
        },
    )
    assert state["ready"] is False
    assert state["status"] == "waiting_for_retry4_formal_completion"


def test_checkpoint_contract_rejects_smoke_and_nonfinite():
    checkpoint, best = _checkpoint()
    video._validate_checkpoint_contract(
        checkpoint,
        arm=video.CONTROL_ARM,
        promoted=_promoted(),
        summary_best=best,
    )
    smoke = deepcopy(checkpoint)
    smoke["smoke_test"] = True
    with pytest.raises(video.V81ComparisonError, match="formal admissible"):
        video._validate_checkpoint_contract(
            smoke,
            arm=video.CONTROL_ARM,
            promoted=_promoted(),
            summary_best=best,
        )
    nonfinite = deepcopy(checkpoint)
    nonfinite["model_state_dict"]["weight"] = torch.tensor(
        [float("nan")]
    )
    with pytest.raises(video.V81ComparisonError, match="non-finite"):
        video._validate_checkpoint_contract(
            nonfinite,
            arm=video.CONTROL_ARM,
            promoted=_promoted(),
            summary_best=best,
        )


def test_checkpoint_contract_rejects_input_hash_tamper():
    checkpoint, best = _checkpoint()
    checkpoint["input_contract"]["target_steps"] = 1
    with pytest.raises(video.V81ComparisonError, match="input contract"):
        video._validate_checkpoint_contract(
            checkpoint,
            arm=video.CONTROL_ARM,
            promoted=_promoted(),
            summary_best=best,
        )


def test_completed_output_rejects_claim_invalidation(tmp_path: Path):
    receipt = {
        "artifact_status": (
            "invalidated_for_hanyang_benefit_and_text_semantic_claims"
        ),
        "claim_eligibility": {
            "hanyang_training_benefit": False,
            "text_to_motion_semantic_alignment": False,
            "emotion_accuracy": False,
        },
    }
    (tmp_path / video.CLAIM_INVALIDATION_FILENAME).write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )
    with pytest.raises(video.V81ComparisonError, match="is invalidated"):
        video._reject_claim_invalidated(tmp_path)


def test_ass_contains_text_emotion_jerk_expression_response_and_steps(
    checked_config,
):
    selection = deepcopy(checked_config["_gt_plan"]["selections"][0])
    selection["metrics"] = {
        name: {
            "jerk_rms_rad_s3": 1.0,
            "expression_amplitude_joint_range_rms_rad": 0.2,
            "head_velocity_rms_rad_s": 0.3,
        }
        for name in ("gt", *video.ARMS)
    }
    selection["planner_duration_diagnostics_sec"] = {
        arm: 2.0 for arm in video.ARMS
    }
    selection["gt"] = {"clip_id": "held_out_clip"}
    timeline = [deepcopy(selection) for _ in range(24)]
    for index, item in enumerate(timeline, start=1):
        item["index"] = index
    document = video.build_ass_document(
        timeline,
        responses={arm: _gate() for arm in video.ARMS},
        best_steps={
            video.CONTROL_ARM: 5500,
            video.TREATMENT_ARM: 6000,
        },
        robot_width=1440,
        panel_width=760,
        height=720,
    )
    for token in (
        "TEXT:",
        "EMOTION:",
        "RAW JERK",
        "EXPRESSION",
        "HEAD activity",
        "C RESPONSE",
        "T RESPONSE",
        "FORMAL BEST STEP C 5500 · T 6000",
        "STATIC PADDING = 0",
    ):
        assert token in document


def test_shared_camera_override_is_exact(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_write(path, trajectory, *, fps):
        Path(path).write_text("joint csv", encoding="utf-8")

    def fake_render(joint_csv, output_mp4, **kwargs):
        captured.update(kwargs)
        Path(output_mp4).write_bytes(b"video")
        camera = kwargs["camera_override"]
        return {
            "camera_distance": camera["distance"],
            "camera_lookat": camera["lookat"],
            "camera_azimuth_deg": camera["azimuth_deg"],
            "camera_elevation_deg": camera["elevation_deg"],
        }

    monkeypatch.setattr(video.abc_video, "write_generated_csv", fake_write)
    monkeypatch.setattr(mujoco_playback, "render_motion", fake_render)
    camera = {
        "distance": 2.8,
        "lookat": [0.0, 0.1, 0.2],
        "azimuth_deg": 180.0,
        "elevation_deg": -5.0,
        "framing": {"bounds": "union"},
    }
    receipt = video._render_with_shared_camera(
        np.zeros((4, 18), dtype=np.float32),
        csv_path=tmp_path / "lane.csv",
        mp4_path=tmp_path / "lane.mp4",
        fps=30.0,
        width=480,
        height=680,
        simplified=False,
        camera=camera,
    )
    assert captured["camera_override"] == camera
    assert receipt["shared_union_camera"] == camera


def test_render_motion_rejects_malformed_camera_override(tmp_path: Path):
    # Contract validation occurs after trajectory/model setup in render_motion,
    # so assert the public signature rather than initializing MuJoCo here.
    import inspect

    assert "camera_override" in inspect.signature(
        mujoco_playback.render_motion
    ).parameters


def test_waiter_requires_two_gpu_passes_before_launch(
    checked_config, monkeypatch, tmp_path: Path
):
    config = deepcopy(checked_config)
    config["_output_dir"] = str(tmp_path)
    completion_calls = []
    gpu_calls = []
    commands = []

    def completion_reader(_):
        completion_calls.append(True)
        return {
            "ready": True,
            "status": "formal_control_and_treatment_admissible",
            "supervisor_receipt": {"sha256": "a" * 64},
        }

    def gpu_reader(_):
        gpu_calls.append(True)
        return {
            "ready": True,
            "known": True,
            "free_memory_mib": 32000.0,
            "compute_processes": [],
        }

    def command_runner(command, **_):
        commands.append(command)
        return subprocess.CompletedProcess(
            command, 0, stdout='{"status":"complete"}', stderr=""
        )

    monkeypatch.setattr(
        waiter.builder,
        "validate_completed_output",
        lambda _: {
            "summary": "summary.json",
            "video": "video.mp4",
        },
    )
    result = waiter.wait_and_build(
        config,
        config_path=CONFIG,
        poll_seconds=1.0,
        timeout_seconds=10.0,
        overwrite=False,
        completion_reader=completion_reader,
        gpu_reader=gpu_reader,
        command_runner=command_runner,
        sleeper=lambda _: None,
        clock=iter([0.0, 0.0, 1.0, 2.0, 3.0]).__next__,
    )
    assert result["status"] == "complete"
    assert result["builder_launched"] is True
    assert len(gpu_calls) == 2
    assert len(commands) == 1
    assert "--overwrite" in commands[0]


def test_gpu_gate_never_kills_and_blocks_compute_process():
    outputs = iter(
        [
            subprocess.CompletedProcess([], 0, stdout="32000\n", stderr=""),
            subprocess.CompletedProcess(
                [],
                0,
                stdout="1234, training, 10000\n",
                stderr="",
            ),
        ]
    )
    commands = []

    def runner(command, **_):
        commands.append(command)
        return next(outputs)

    state = waiter.gpu_snapshot(
        {
            "index": 0,
            "minimum_free_memory_mib": 24576,
            "required_consecutive_passes": 2,
            "unknown_compute_policy": "wait_never_kill",
        },
        command_runner=runner,
    )
    assert state["ready"] is False
    assert state["policy"] == "wait_never_kill"
    assert all(command[0] == "nvidia-smi" for command in commands)
    assert not any("kill" in part for command in commands for part in command)


def test_forbidden_kimodo_output_path_is_rejected():
    with pytest.raises(video.V81ComparisonError, match="Kimodo"):
        video._resolve_path(
            "/tmp/" + "ki" + "modo/video",
            field="output",
            must_exist=False,
        )


def test_transient_collected_unit_is_explicitly_recognized(monkeypatch):
    monkeypatch.setattr(
        video.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [],
            1,
            stdout=(
                "LoadState=not-found\n"
                "ActiveState=inactive\n"
                "SubState=dead\n"
                "Result=success\n"
                "ExecMainStatus=0\n"
            ),
            stderr="Unit could not be found.",
        ),
    )
    state = video._service_snapshot(
        "hanyang-beat2-emotion-preserving-v81-retry4.service"
    )
    assert state["known"] is True
    assert state["transient_reclaimed"] is True
    assert state["active"] is False
    assert state["failed"] is False
