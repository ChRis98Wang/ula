from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.experimental import (
    build_beat2_emotion_hierarchy_v7_qwen_ab_gt_60s as video,
)


def _record(
    clip_id: str,
    *,
    frames: int,
    prompt: str,
    emotion: str,
) -> dict:
    return {
        "clip_id": clip_id,
        "dataset": "BEAT2",
        "fixed_split_assignment": "test",
        "prompt": prompt,
        "emotion_id": emotion,
        "accepted_for_training": True,
        "adjudication": {"status": "motion_only_train_ready"},
        "training_admission_status": "motion_only_physical_qc_train_ready",
        "motion_18d": {
            "state": "passed",
            "action_dim": 18,
            "fps": 30.0,
            "frames": frames,
            "quality_gate": {"passed": True},
        },
    }


@pytest.mark.parametrize(
    ("prompt", "emotion"),
    [
        ("Perform a happy affect.", "happy"),
        ("Perform a neutral affect.", "neutral"),
        ("Perform an angry affect.", "angry"),
        ("Perform a surprised affect.", "surprise"),
        ("Perform a sad affect.", "sad"),
        ("Perform a fearful affect.", "fear"),
    ],
)
def test_target_emotion_parser_is_exact(prompt: str, emotion: str) -> None:
    assert video.target_emotion_from_prompt(prompt) == emotion


def test_gt_selection_prefers_longest_then_clip_id() -> None:
    prompt = "Perform a high-intensity iconic gesture with a happy affect."
    records = {
        row["clip_id"]: row
        for row in (
            _record("z", frames=80, prompt=prompt, emotion="happy"),
            _record("b", frames=100, prompt=prompt, emotion="happy"),
            _record("a", frames=100, prompt=prompt, emotion="happy"),
            _record("wrong", frames=200, prompt=prompt, emotion="sad"),
        )
    }
    selected, candidates = video.select_gt_reference(
        records,
        prompt=prompt,
        target_emotion="happy",
    )
    assert selected["clip_id"] == "a"
    assert candidates == ["a", "b", "z"]


def test_same_noise_prefix_accepts_only_exact_prefix() -> None:
    left = [np.arange((index + 4) * 18, dtype=np.float32).reshape(-1, 18) for index in range(24)]
    right = [value.copy() for value in left]
    receipts = video._same_noise_prefix(left, right)
    assert len(receipts) == 24
    assert all(item["exact_full_event_noise_match"] is True for item in receipts)
    right[2][0, 0] += 1
    with pytest.raises(video.ComparisonError, match="initial noise"):
        video._same_noise_prefix(left, right)


def test_ass_discloses_text_emotion_time_and_metric_meaning() -> None:
    metric = {
        "jerk_rms_rad_s3": 12.5,
        "expression_amplitude_joint_range_rms_rad": 0.25,
        "head_velocity_rms_rad_s": 0.1,
    }
    timeline = [
        {
            "index": index + 1,
            "start_sec": index * 2.5,
            "end_sec": (index + 1) * 2.5,
            "native_duration_sec": 2.5,
            "prompt": f"Complete long prompt number {index + 1} with a happy affect.",
            "target_emotion": "happy",
            "seed": 100 + index,
            "gt": {"clip_id": f"beat:{index}"},
            "planner_duration_diagnostics_sec": {
                "frozen_A": 1.5,
                "lora_B": 1.75,
            },
            "metrics": {
                "gt": metric,
                "frozen_A": metric,
                "lora_B": metric,
            },
        }
        for index in range(24)
    ]
    document = video.build_ass_document(
        timeline,
        robot_width=1440,
        panel_width=720,
        height=720,
    )
    assert "Complete long prompt number 1" in document
    assert "TARGET EMOTION: HAPPY" in document
    assert "RAW JERK RMS" in document
    assert "EXPRESSION = JOINT-RANGE RMS PROXY, NOT EMOTION ACCURACY" in document
    assert "TIME 59.0s / 60.0s" in document
    assert "RETURN-TO-ZERO" in document
    assert "STATIC PADDING = 0" in document


def test_native_montage_is_exactly_60s_without_padding() -> None:
    emotions = ("neutral", "sad", "happy", "angry", "surprise", "fear")
    records = {}
    for emotion in emotions:
        adjective = "surprised" if emotion == "surprise" else (
            "fearful" if emotion == "fear" else emotion
        )
        prompt = f"Perform a medium gesture with a {adjective} affect."
        for index in range(4):
            record = _record(
                f"{emotion}-{index}",
                frames=75,
                prompt=prompt,
                emotion=emotion,
            )
            records[record["clip_id"]] = record
    selected, receipt = video.select_native_event_montage(records)
    assert len(selected) == 24
    assert sum(row["motion_18d"]["frames"] for row in selected) == 1800
    assert receipt["duration_sec"] == 60.0
    assert receipt["selection_used_action_values"] is False
    assert video.COMPARISON_CONTRACT["static_padding_frames"] == 0
    assert video.COMPARISON_CONTRACT["endpoint_hold"] is False


def test_b_completion_does_not_accept_missing_summary(tmp_path: Path) -> None:
    validated = {
        "_overlay": {"output_dir": str(tmp_path / "b")},
    }
    assert video.b_completion_state(validated) == {
        "ready": False,
        "status": "training_in_progress",
        "summary": str(tmp_path / "b" / "training_summary_v7.json"),
        "checkpoint": str(
            tmp_path / "b" / "generator_emotion_hierarchy_v7.pt"
        ),
    }


def test_b_completion_waits_for_training_service_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "b"
    output.mkdir()
    checkpoint = output / "generator_emotion_hierarchy_v7.pt"
    checkpoint.write_bytes(b"real-artifact-placeholder-for-state-test")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (output / "training_summary_v7.json").write_text(
        json.dumps(
            {
                "status": "experimental_candidate",
                "checkpoint_sha256": checkpoint_sha,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(video, "_training_service_active", lambda unit: True)
    state = video.b_completion_state(
        {
            "_overlay": {"output_dir": str(output)},
            "lora_training_service_unit": "training.service",
        }
    )
    assert state["ready"] is False
    assert (
        state["status"]
        == "checkpoint_complete_waiting_for_training_service_exit"
    )
    assert state["training_service_active"] is True


def test_source_paths_reject_both_external_datasets() -> None:
    for token in ("kimodo", "hanyang"):
        with pytest.raises(video.ComparisonError, match="forbidden"):
            video._reject_source_path(
                f"/datasets/{token}/clip.csv",
                field="source",
            )
