import csv
import json

import torch

from upper_body_skeleton.lerobot_export import export_lerobot_dataset
from upper_body_skeleton.retarget_v2 import JOINT_ORDER
from upper_body_skeleton.ula_training import (
    BASE_CONDITION_DIM,
    KIMODO_CONDITION_DIM,
    LEGACY_CONDITION_DIM,
    ULA_FM_LEGACY_ARCHITECTURE,
    ULA_MMDIT_LITE_ARCHITECTURE,
    UlaFmModel,
    UlaMMDiTLiteModel,
    build_condition_from_text,
    create_ula_model,
    load_lerobot_episodes,
    sample_trajectory,
    planner_loss,
    train_steps,
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
    assert csv_path.read_text(encoding="utf-8").splitlines()[0] == "time_sec," + ",".join(JOINT_ORDER)


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

    assert isinstance(legacy, UlaFmModel)
    assert isinstance(mmdit, UlaMMDiTLiteModel)
    assert legacy.architecture == ULA_FM_LEGACY_ARCHITECTURE
    assert mmdit.architecture == ULA_MMDIT_LITE_ARCHITECTURE


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
