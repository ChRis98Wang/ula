import csv
import json

import pytest
import torch

from upper_body_skeleton.lerobot_export import export_lerobot_dataset
from upper_body_skeleton.retarget_v2 import JOINT_ORDER
from upper_body_skeleton.ula_training import (
    BASE_CONDITION_DIM,
    KIMODO_CONDITION_DIM,
    LEGACY_CONDITION_DIM,
    ULA_ADALN_LITE_ARCHITECTURE,
    ULA_FM_LEGACY_ARCHITECTURE,
    ULA_MMDIT_LITE_ARCHITECTURE,
    UlaAdaLNLiteModel,
    UlaFmModel,
    UlaMMDiTLiteModel,
    build_condition_from_text,
    clip_grad_norm_float64,
    create_ula_model,
    compute_action_normalization_stats,
    denormalize_action_tensor,
    load_lerobot_episodes,
    normalize_action_tensor,
    sample_trajectory,
    planner_loss,
    train_steps,
    model_checkpoint_payload,
    write_training_preview,
    write_generated_csv,
)


def write_joint_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["time_sec"] + JOINT_ORDER)
        writer.writeheader()
        for frame_index in range(rows):
            row = {"time_sec": frame_index / 30.0}
            row.update({joint: float(frame_index) / 10.0 for joint in JOINT_ORDER})
            writer.writerow(row)


def make_lerobot_fixture(tmp_path, rows_per_file=100):
    csv_path = tmp_path / "sample.csv"
    jsonl_path = tmp_path / "index.jsonl"
    write_joint_csv(csv_path, 6)
    records = []
    for idx in range(2):
        records.append(
            {
                "sample_id": f"sample_{idx}",
                "source": {"dataset": "test", "retarget_csv_path": str(csv_path), "video_path": "", "json_path": "", "npz_path": ""},
                "time_window": {"start_sec": 0.0, "end_sec": 6.0 / 30.0, "fps": 30.0},
                "language_condition": {
                    "raw_transcript": "hello",
                    "scenario_description": "hello",
                    "action_description": "upper body gesture",
                    "intent_text": "explain",
                    "mood_text": "neutral",
                    "rationale_text": "explain",
                    "instruction_variants": ["explain with restrained style"],
                },
                "labels": {
                    "intent": "explaining",
                    "observed_affect": "neutral",
                    "motion_style": "restrained",
                    "arousal": 0.1,
                    "valence": 0.2,
                    "arousal_token": 1,
                    "valence_token": 2,
                    "motion_energy": 0.01,
                    "label_sources": [],
                    "behavior_id": "Behavior.GreetingOwner01",
                    "emotion_id": "happy",
                },
                "meta_semantics": {"semantic_gesture": "upper_body_gesture", "behavior_id": "Behavior.GreetingOwner01", "emotion_id": "happy"},
                "action": {"retarget_csv_path": str(csv_path), "start_row": 0, "end_row": 6, "fps": 30.0, "joint_order": JOINT_ORDER},
                "quality": {"accepted_for_training": True, "frame_count": 6, "flagged_frame_count": 0, "max_elbow_overfold": 0, "max_yaw_under_response": 0},
            }
        )
    jsonl_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    out_dir = tmp_path / "lerobot"
    export_lerobot_dataset(jsonl_path, out_dir, rows_per_file=rows_per_file)
    return out_dir


def test_lerobot_episode_loader_returns_action_chunks(tmp_path):
    out_dir = make_lerobot_fixture(tmp_path)

    episodes = load_lerobot_episodes(out_dir, max_episodes=2)

    assert len(episodes) == 2
    assert episodes[0]["actions"].shape == (6, len(JOINT_ORDER))
    assert episodes[0]["condition"].shape[0] == KIMODO_CONDITION_DIM
    assert episodes[0]["meta"]["behavior_id"] == "Behavior.GreetingOwner01"
    assert episodes[0]["meta"]["emotion_id"] == "happy"


def test_lerobot_episode_loader_stops_after_requested_episode_count(tmp_path):
    out_dir = make_lerobot_fixture(tmp_path, rows_per_file=6)
    second_file = out_dir / "data" / "chunk-000" / "file-001.parquet"
    second_file.write_text("not a parquet file", encoding="utf-8")

    episodes = load_lerobot_episodes(out_dir, max_episodes=1)

    assert len(episodes) == 1
    assert episodes[0]["episode_index"] == 0


def test_ula_model_trains_one_flow_matching_step(tmp_path):
    out_dir = make_lerobot_fixture(tmp_path)
    episodes = load_lerobot_episodes(out_dir, max_episodes=2)
    model = UlaFmModel(action_dim=len(JOINT_ORDER), condition_dim=episodes[0]["condition"].shape[0], hidden_dim=64)

    losses = train_steps(model, episodes, steps=2, batch_size=2, lr=1e-3, device="cpu")

    assert len(losses) == 2
    assert all(torch.isfinite(torch.tensor(losses)))


def test_action_normalization_stats_roundtrip_episode_actions(tmp_path):
    out_dir = make_lerobot_fixture(tmp_path)
    episodes = load_lerobot_episodes(out_dir, max_episodes=2)
    stats = compute_action_normalization_stats(episodes)
    actions = torch.tensor(episodes[0]["actions"])

    normalized = normalize_action_tensor(actions, stats)
    restored = denormalize_action_tensor(normalized, stats)

    assert stats["mean"].shape == (len(JOINT_ORDER),)
    assert stats["std"].shape == (len(JOINT_ORDER),)
    assert torch.all(stats["std"] > 0)
    assert torch.allclose(restored, actions, atol=1e-6)


def test_float64_gradient_clipping_handles_large_finite_gradients_without_overflow():
    parameter = torch.nn.Parameter(torch.zeros(4))
    parameter.grad = torch.full_like(parameter, 1e30)

    original_norm = clip_grad_norm_float64([parameter], 1.0)

    assert original_norm == pytest.approx(2e30)
    assert torch.isfinite(parameter.grad).all()
    assert float(torch.linalg.vector_norm(parameter.grad.double())) == pytest.approx(1.0)


def test_text_condition_builder_uses_codes_and_text_signal():
    neutral = build_condition_from_text("explain calmly", style="restrained", affect="neutral", gesture="null")
    energetic = build_condition_from_text("wave excitedly", style="energetic", affect="excited", gesture="waving")

    assert neutral.shape == energetic.shape
    assert neutral.shape[0] == KIMODO_CONDITION_DIM
    assert not torch.equal(torch.tensor(neutral), torch.tensor(energetic))


def test_text_condition_builder_can_emit_legacy_condition_for_old_checkpoints():
    legacy = build_condition_from_text("开心地挥手", condition_dim=LEGACY_CONDITION_DIM)

    assert legacy.shape == (LEGACY_CONDITION_DIM,)


def test_text_condition_builder_uses_explicit_kimodo_behavior_and_emotion():
    greeting = build_condition_from_text(
        "开心地挥手",
        behavior_id="Behavior.GreetingOwner01",
        emotion_id="happy",
    )
    alert = build_condition_from_text(
        "开心地挥手",
        behavior_id="Behavior.Alert",
        emotion_id="happy",
    )

    assert greeting.shape == (KIMODO_CONDITION_DIM,)
    assert not torch.equal(torch.tensor(greeting[BASE_CONDITION_DIM:]), torch.tensor(alert[BASE_CONDITION_DIM:]))


def test_flow_sampler_exports_generated_joint_csv(tmp_path):
    out_dir = make_lerobot_fixture(tmp_path)
    episodes = load_lerobot_episodes(out_dir, max_episodes=2)
    model = UlaFmModel(action_dim=len(JOINT_ORDER), condition_dim=episodes[0]["condition"].shape[0], hidden_dim=64)
    train_steps(model, episodes, steps=1, batch_size=2, lr=1e-3, device="cpu")

    trajectory = sample_trajectory(
        model,
        condition=episodes[0]["condition"],
        frames=6,
        action_dim=len(JOINT_ORDER),
        steps=3,
        device="cpu",
        seed=7,
    )
    csv_path = tmp_path / "generated.csv"
    write_generated_csv(csv_path, trajectory, fps=30.0)

    assert trajectory.shape == (6, len(JOINT_ORDER))
    assert torch.isfinite(torch.tensor(trajectory)).all()
    assert model.training is True
    assert csv_path.read_text(encoding="utf-8").splitlines()[0] == "time_sec," + ",".join(JOINT_ORDER)


def test_flow_sampler_accepts_generation_pose_bounds():
    model = UlaFmModel(action_dim=len(JOINT_ORDER), condition_dim=92, hidden_dim=64)
    bounds = {
        "joint_rShoulderRoll": (-1.30, -0.05),
        "joint_rElbow": (-1.58, 0.35),
    }

    trajectory = sample_trajectory(
        model,
        condition=torch.zeros(92),
        frames=12,
        action_dim=len(JOINT_ORDER),
        steps=4,
        device="cpu",
        seed=3,
        pose_bounds=bounds,
    )

    shoulder = trajectory[:, JOINT_ORDER.index("joint_rShoulderRoll")]
    elbow = trajectory[:, JOINT_ORDER.index("joint_rElbow")]
    assert shoulder.min() >= -1.30 - 1e-6
    assert shoulder.max() <= -0.05 + 1e-6
    assert elbow.min() >= -1.58 - 1e-6
    assert elbow.max() <= 0.35 + 1e-6


def test_flow_sampler_denormalizes_action_stats_before_export():
    class ZeroVelocityModel(torch.nn.Module):
        def forward(self, x_t, t, condition):
            return torch.zeros_like(x_t)

    mean = torch.zeros(len(JOINT_ORDER))
    std = torch.ones(len(JOINT_ORDER)) * 0.02
    shoulder_index = JOINT_ORDER.index("joint_lShoulderRoll")
    mean[shoulder_index] = -0.98
    stats = {"mean": mean, "std": std}

    trajectory = sample_trajectory(
        ZeroVelocityModel(),
        condition=torch.zeros(92),
        frames=20,
        action_dim=len(JOINT_ORDER),
        steps=2,
        device="cpu",
        seed=5,
        action_stats=stats,
        pose_bounds={"joint_lShoulderRoll": (-1.30, -0.15)},
    )

    shoulder = trajectory[:, shoulder_index]
    assert shoulder.min() > -1.08
    assert shoulder.max() < -0.88


def test_ula_model_predicts_duration_and_transition_logits():
    condition = torch.randn(3, 92)
    model = UlaFmModel(action_dim=len(JOINT_ORDER), condition_dim=92, hidden_dim=64)

    plan = model.plan_condition(condition)

    assert plan["duration_sec"].shape == (3,)
    assert plan["transition_logits"].shape == (3, 4)
    assert torch.all(plan["duration_sec"] > 0.0)


def test_ula_model_uses_frame_position_signal():
    condition = torch.zeros(1, 92)
    model = UlaFmModel(action_dim=len(JOINT_ORDER), condition_dim=92, hidden_dim=64)
    model.eval()
    identical_frames = torch.zeros(1, 6, len(JOINT_ORDER))

    with torch.no_grad():
        output = model(identical_frames, torch.tensor([0.5]), condition)

    assert not torch.allclose(output[:, 0, :], output[:, 1, :])


def test_mmdit_lite_model_uses_semantic_tokens_and_preserves_motion_shape():
    model = UlaMMDiTLiteModel(
        action_dim=len(JOINT_ORDER),
        condition_dim=KIMODO_CONDITION_DIM,
        hidden_dim=64,
        layers=2,
        semantic_tokens=4,
    )
    x_t = torch.randn(2, 6, len(JOINT_ORDER))
    condition = torch.randn(2, KIMODO_CONDITION_DIM)

    output = model(x_t, torch.tensor([0.2, 0.8]), condition)
    plan = model.plan_condition(condition)

    assert output.shape == x_t.shape
    assert model.architecture == ULA_MMDIT_LITE_ARCHITECTURE
    assert model.semantic_tokens == 4
    assert model.last_joint_sequence_shape == (2, 10, 64)
    assert plan["duration_sec"].shape == (2,)
    assert plan["transition_logits"].shape == (2, 4)


def test_adaln_lite_model_modulates_motion_with_condition_and_time():
    model = UlaAdaLNLiteModel(
        action_dim=len(JOINT_ORDER),
        condition_dim=KIMODO_CONDITION_DIM,
        hidden_dim=64,
        layers=2,
    )
    x_t = torch.randn(2, 6, len(JOINT_ORDER))
    condition = torch.randn(2, KIMODO_CONDITION_DIM)

    output = model(x_t, torch.tensor([0.2, 0.8]), condition)
    shifted_condition_output = model(x_t, torch.tensor([0.2, 0.8]), condition + 0.5)
    plan = model.plan_condition(condition)

    assert output.shape == x_t.shape
    assert model.architecture == ULA_ADALN_LITE_ARCHITECTURE
    assert model.last_motion_sequence_shape == (2, 6, 64)
    assert not torch.allclose(output, shifted_condition_output)
    assert plan["duration_sec"].shape == (2,)
    assert plan["transition_logits"].shape == (2, 4)


def test_model_factory_keeps_legacy_and_mmdit_architectures_separate():
    legacy = create_ula_model(
        ULA_FM_LEGACY_ARCHITECTURE,
        action_dim=len(JOINT_ORDER),
        condition_dim=LEGACY_CONDITION_DIM,
        hidden_dim=32,
        layers=1,
    )
    mmdit = create_ula_model(
        ULA_MMDIT_LITE_ARCHITECTURE,
        action_dim=len(JOINT_ORDER),
        condition_dim=KIMODO_CONDITION_DIM,
        hidden_dim=32,
        layers=1,
        semantic_tokens=3,
    )
    adaln = create_ula_model(
        ULA_ADALN_LITE_ARCHITECTURE,
        action_dim=len(JOINT_ORDER),
        condition_dim=KIMODO_CONDITION_DIM,
        hidden_dim=32,
        layers=1,
    )

    assert isinstance(legacy, UlaFmModel)
    assert isinstance(mmdit, UlaMMDiTLiteModel)
    assert isinstance(adaln, UlaAdaLNLiteModel)
    assert legacy.architecture == ULA_FM_LEGACY_ARCHITECTURE
    assert mmdit.architecture == ULA_MMDIT_LITE_ARCHITECTURE
    assert adaln.architecture == ULA_ADALN_LITE_ARCHITECTURE


def test_mmdit_lite_model_trains_one_flow_matching_step(tmp_path):
    out_dir = make_lerobot_fixture(tmp_path)
    episodes = load_lerobot_episodes(out_dir, max_episodes=2)
    model = create_ula_model(
        ULA_MMDIT_LITE_ARCHITECTURE,
        action_dim=len(JOINT_ORDER),
        condition_dim=episodes[0]["condition"].shape[0],
        hidden_dim=64,
        layers=1,
        semantic_tokens=3,
    )

    losses = train_steps(model, episodes, steps=1, batch_size=2, lr=1e-3, device="cpu")

    assert len(losses) == 1
    assert torch.isfinite(torch.tensor(losses)).all()


def test_adaln_lite_model_trains_one_flow_matching_step(tmp_path):
    out_dir = make_lerobot_fixture(tmp_path)
    episodes = load_lerobot_episodes(out_dir, max_episodes=2)
    model = create_ula_model(
        ULA_ADALN_LITE_ARCHITECTURE,
        action_dim=len(JOINT_ORDER),
        condition_dim=episodes[0]["condition"].shape[0],
        hidden_dim=64,
        layers=1,
    )

    losses = train_steps(model, episodes, steps=1, batch_size=2, lr=1e-3, device="cpu")

    assert len(losses) == 1
    assert torch.isfinite(torch.tensor(losses)).all()


def test_train_steps_writes_best_and_interval_checkpoints(tmp_path):
    out_dir = make_lerobot_fixture(tmp_path)
    episodes = load_lerobot_episodes(out_dir, max_episodes=2)
    model = create_ula_model(
        ULA_ADALN_LITE_ARCHITECTURE,
        action_dim=len(JOINT_ORDER),
        condition_dim=episodes[0]["condition"].shape[0],
        hidden_dim=64,
        layers=1,
    )

    class Args:
        hidden_dim = 64
        layers = 1
        semantic_tokens = 4

    losses = train_steps(
        model,
        episodes,
        steps=2,
        batch_size=2,
        lr=1e-3,
        device="cpu",
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_every_steps=2,
        save_best=True,
        checkpoint_payload_fn=lambda checkpoint_model, step, loss: model_checkpoint_payload(
            checkpoint_model,
            episodes,
            Args(),
            "cpu",
            step=step,
            loss=loss,
        ),
    )

    assert len(losses) == 2
    assert (tmp_path / "checkpoints" / "ula_fm_best_checkpoint.pt").is_file()
    assert (tmp_path / "checkpoints" / "ula_fm_step_000002.pt").is_file()


def test_planner_loss_uses_duration_and_transition_targets():
    condition = torch.randn(2, 92)
    model = UlaFmModel(action_dim=len(JOINT_ORDER), condition_dim=92, hidden_dim=64)
    duration_target = torch.tensor([4.0, 12.0])
    transition_target = torch.tensor([0, 3])

    loss = planner_loss(model, condition, duration_target, transition_target)

    assert torch.isfinite(loss)


def test_training_preview_writes_mujoco_video_folder(tmp_path):
    out_dir = make_lerobot_fixture(tmp_path)
    episodes = load_lerobot_episodes(out_dir, max_episodes=2)
    model = UlaFmModel(action_dim=len(JOINT_ORDER), condition_dim=episodes[0]["condition"].shape[0], hidden_dim=64)
    train_steps(model, episodes, steps=1, batch_size=2, lr=1e-3, device="cpu")

    summary = write_training_preview(
        model,
        preview_root=tmp_path / "previews",
        step=1000,
        text="紧张地解释，同时双手做克制的上肢手势",
        frames=4,
        sampling_steps=2,
        fps=30.0,
        device="cpu",
        seed=3,
        width=160,
        height=120,
        preview_mode="long",
        long_duration_sec=0.6,
        min_segment_sec=0.2,
        max_segment_sec=0.2,
        min_segments=3,
        max_segments=3,
        max_velocity_rad_s=3.0,
        smooth_window=3,
    )

    preview_dir = tmp_path / "previews" / "step_001000"
    assert summary["step"] == 1000
    assert summary["preview_mode"] == "long"
    assert summary["duration_sec"] > 0.4
    assert summary["frames"] > 4
    assert (preview_dir / "long_motion.csv").is_file()
    assert (preview_dir / "long_motion.npz").is_file()
    assert (preview_dir / "plan.json").is_file()
    assert (preview_dir / "summary.json").is_file()
    assert (preview_dir / "long_motion_original_v2.mp4").is_file()
    assert summary["trajectory_quality"]["processed"]["max_velocity_rad_s"] <= 3.0 + 1e-5
