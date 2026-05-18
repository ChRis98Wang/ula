import json

import numpy as np
import torch

from upper_body_skeleton.long_emotion_infer import (
    DEFAULT_LONG_EMOTION_OUTPUT_DIR,
    REPO_ROOT,
    generate_long_emotion_motion,
    limit_joint_velocity,
    smooth_trajectory,
)
from upper_body_skeleton.retarget_v2 import JOINT_ORDER
from upper_body_skeleton.ula_infer import load_model
from upper_body_skeleton.ula_training import UlaFmModel


def test_default_long_emotion_output_dir_is_repo_relative():
    assert DEFAULT_LONG_EMOTION_OUTPUT_DIR == REPO_ROOT / "deliverables" / "long_emotion_previews" / "manual"


def test_long_emotion_generation_writes_multisegment_outputs(tmp_path):
    model = UlaFmModel(action_dim=len(JOINT_ORDER), condition_dim=92, hidden_dim=64)

    summary = generate_long_emotion_motion(
        model,
        text="紧张地解释，然后逐渐缓和",
        output_dir=tmp_path / "long_preview",
        fps=30.0,
        max_duration_sec=0.4,
        min_segment_sec=0.1,
        max_segment_sec=0.2,
        max_segments=3,
        sampling_steps=2,
        device="cpu",
        seed=11,
        render=False,
    )

    plan = json.loads((tmp_path / "long_preview" / "plan.json").read_text(encoding="utf-8"))
    csv_lines = (tmp_path / "long_preview" / "long_motion.csv").read_text(encoding="utf-8").splitlines()

    assert summary["segments"] >= 2
    assert summary["frames"] > 4
    assert summary["duration_sec"] > 0.1
    assert summary["rendered_mp4"] is None
    assert len(plan["segments"]) == summary["segments"]
    assert csv_lines[0] == "time_sec," + ",".join(JOINT_ORDER)
    assert len(csv_lines) == summary["frames"] + 1
    assert summary["trajectory_quality"]["processed"]["max_velocity_rad_s"] <= 3.0 + 1e-5
    assert torch.isfinite(torch.tensor(summary["last_pose"])).all()


def test_long_emotion_generation_can_open_mujoco_viewer(tmp_path, monkeypatch):
    model = UlaFmModel(action_dim=len(JOINT_ORDER), condition_dim=92, hidden_dim=64)
    calls = []

    def fake_play_motion(path, **kwargs):
        calls.append((path, kwargs))
        return {"frames_played": 4, "loops_completed": kwargs["loops"]}

    monkeypatch.setattr("upper_body_skeleton.long_emotion_infer.play_motion", fake_play_motion)

    summary = generate_long_emotion_motion(
        model,
        text="开心地挥手",
        output_dir=tmp_path / "viewer_preview",
        fps=30.0,
        max_duration_sec=0.2,
        min_segment_sec=0.1,
        max_segment_sec=0.1,
        min_segments=1,
        max_segments=1,
        sampling_steps=2,
        device="cpu",
        seed=11,
        render=False,
        play=True,
        play_loops=1,
        play_realtime=False,
    )

    assert summary["viewer"]["frames_played"] == 4
    assert summary["viewer"]["loops_completed"] == 1
    assert calls[0][0] == tmp_path / "viewer_preview" / "long_motion.csv"
    assert calls[0][1]["loops"] == 1
    assert calls[0][1]["realtime"] is False


def test_joint_velocity_limiter_removes_large_frame_jumps():
    trajectory = np.zeros((6, len(JOINT_ORDER)), dtype=np.float32)
    trajectory[1, 3] = 3.0
    trajectory[2, 3] = -3.0
    trajectory[3, 6] = 2.5

    limited = limit_joint_velocity(trajectory, fps=30.0, max_velocity_rad_s=3.0)
    smoothed = smooth_trajectory(limited, window=3)
    max_delta = float(np.abs(np.diff(smoothed, axis=0)).max())

    assert max_delta <= 3.0 / 30.0 + 1e-6


def test_old_checkpoint_without_planner_heads_loads_strict_false(tmp_path):
    checkpoint = tmp_path / "old_checkpoint.pt"
    old_model = UlaFmModel(action_dim=len(JOINT_ORDER), condition_dim=92, hidden_dim=64)
    state = {
        key: value
        for key, value in old_model.state_dict().items()
        if not key.startswith("plan.") and not key.startswith("duration_head.") and not key.startswith("transition_head.")
    }
    torch.save(
        {
            "model_state_dict": state,
            "action_dim": len(JOINT_ORDER),
            "condition_dim": 92,
            "config": {"hidden_dim": 64, "layers": 4},
        },
        checkpoint,
    )

    model, loaded = load_model(checkpoint, "cpu")
    condition = torch.randn(1, 92)
    plan = model.plan_condition(condition)

    assert loaded["condition_dim"] == 92
    assert plan["duration_sec"].shape == (1,)
    assert plan["transition_logits"].shape == (1, 4)
