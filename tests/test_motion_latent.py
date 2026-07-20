from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import upper_body_skeleton.motion_latent as motion_latent_module
from upper_body_skeleton.motion_latent import (
    MotionLatentIndex,
    MotionMetricEncoder,
    build_motion_features,
    build_raw_motion_index,
    compute_latent_diagnostics,
    compute_motion_descriptors,
    compute_motion_normalization,
    cross_set_retrieval_loss,
    encode_motion_episodes,
    load_motion_latent_episodes,
    load_motion_metric_checkpoint,
    motion_metric_loss,
    sample_semantic_pair_batch,
    select_prototype_medoids,
    stratified_episode_split,
    supervised_contrastive_loss,
    train_motion_latent,
)
from training.scripts.train_motion_latent import load_train_config, training_args_from_config


def make_episode(episode_index, behavior_id, emotion_id, value, frames=12, joints=15):
    actions = np.full((frames, joints), value, dtype=np.float32)
    actions += np.linspace(0.0, 0.1, frames, dtype=np.float32)[:, None]
    return {
        "episode_index": episode_index,
        "actions": actions,
        "meta": {
            "behavior_id": behavior_id,
            "emotion_id": emotion_id,
        },
    }


def make_grouped_episodes(samples_per_group=10):
    episodes = []
    for behavior_index, behavior in enumerate(("Behavior.GreetingOwner01", "Behavior.Alert")):
        for emotion_index, emotion in enumerate(("happy", "fear")):
            for sample_index in range(samples_per_group):
                episodes.append(
                    make_episode(
                        episode_index=len(episodes),
                        behavior_id=behavior,
                        emotion_id=emotion,
                        value=behavior_index + emotion_index * 0.2 + sample_index * 0.01,
                    )
                )
    return episodes


def semantic_counts(episodes):
    return Counter((episode["meta"]["behavior_id"], episode["meta"]["emotion_id"]) for episode in episodes)


def test_stratified_episode_split_is_deterministic_and_keeps_every_group_in_each_split():
    episodes = make_grouped_episodes(samples_per_group=10)

    first = stratified_episode_split(episodes, seed=17)
    second = stratified_episode_split(episodes, seed=17)

    assert [[item["episode_index"] for item in split] for split in first] == [
        [item["episode_index"] for item in split] for split in second
    ]
    reversed_input = stratified_episode_split(list(reversed(episodes)), seed=17)
    assert [sorted(item["episode_index"] for item in split) for split in first] == [
        sorted(item["episode_index"] for item in split) for split in reversed_input
    ]
    train, validation, test = first
    assert set(semantic_counts(train).values()) == {8}
    assert set(semantic_counts(validation).values()) == {1}
    assert set(semantic_counts(test).values()) == {1}
    ids = [set(item["episode_index"] for item in split) for split in first]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])


def test_motion_normalization_uses_only_supplied_episodes_and_roundtrips():
    train = [make_episode(0, "Behavior.Alert", "fear", 1.0), make_episode(1, "Behavior.Alert", "fear", 3.0)]
    held_out = make_episode(2, "Behavior.Alert", "fear", 100.0)

    stats = compute_motion_normalization(train)
    actions = torch.from_numpy(held_out["actions"])[None]
    normalized = (actions - stats["mean"]) / stats["std"]
    restored = normalized * stats["std"] + stats["mean"]

    assert stats["mean"].shape == (1, 1, 15)
    assert stats["std"].shape == (1, 1, 15)
    assert torch.all(stats["mean"] < 3.1)
    assert torch.all(stats["std"] > 0)
    assert torch.allclose(restored, actions, atol=1e-6)


def test_build_motion_features_concatenates_normalized_position_and_velocity():
    episode = make_episode(0, "Behavior.GreetingOwner01", "happy", 0.5, frames=7, joints=15)
    stats = compute_motion_normalization([episode])
    actions = torch.from_numpy(episode["actions"])[None]

    features = build_motion_features(actions, stats)

    assert features.shape == (1, 30, 7)
    assert torch.allclose(features[:, 15:, 0], torch.zeros(1, 15), atol=1e-7)
    assert torch.all(features[:, 15:, 1:] > 0)
    assert torch.isfinite(features).all()


def test_motion_metric_encoder_returns_normalized_embedding_and_auxiliary_heads():
    model = MotionMetricEncoder(action_dim=15, latent_dim=24, hidden_dim=32, behavior_classes=2, emotion_classes=3)
    features = torch.randn(4, 30, 17)

    output = model(features)

    assert output["embedding"].shape == (4, 24)
    assert torch.allclose(output["embedding"].norm(dim=-1), torch.ones(4), atol=1e-5)
    assert output["behavior_logits"].shape == (4, 2)
    assert output["emotion_logits"].shape == (4, 3)
    assert output["descriptors"].shape == (4, 6)


def test_motion_descriptors_are_finite_and_distinguish_static_from_active_motion():
    static = torch.zeros(1, 12, 15)
    active = torch.zeros(1, 12, 15)
    active[:, :, 3:9] = torch.linspace(0.0, 2.0, 12)[None, :, None]
    actions = torch.cat([static, active], dim=0)

    descriptors = compute_motion_descriptors(actions)

    assert descriptors.shape == (2, 6)
    assert torch.isfinite(descriptors).all()
    assert descriptors[1, 0] > descriptors[0, 0]
    assert descriptors[1, 1] > descriptors[0, 1]
    assert descriptors[1, 3] > descriptors[1, 4]


def test_supervised_contrastive_loss_rewards_same_label_embeddings_being_close():
    labels = torch.tensor([0, 0, 1, 1])
    separated = torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]])
    mixed = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.9, 0.1], [-0.9, -0.1]])

    separated_loss = supervised_contrastive_loss(separated, labels, temperature=0.1)
    mixed_loss = supervised_contrastive_loss(mixed, labels, temperature=0.1)

    assert separated_loss < mixed_loss


def test_supervised_contrastive_loss_without_positive_pairs_is_differentiable_zero():
    embeddings = torch.randn(3, 5, requires_grad=True)

    loss = supervised_contrastive_loss(embeddings, torch.tensor([0, 1, 2]))
    loss.backward()

    assert loss.item() == 0.0
    assert embeddings.grad is not None
    assert torch.equal(embeddings.grad, torch.zeros_like(embeddings))


def test_cross_set_retrieval_loss_uses_train_positives_for_single_validation_examples():
    reference = make_latent_index(
        embeddings=[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        episode_indices=[0, 1, 2, 3],
        behavior_ids=["wave", "wave", "alert", "alert"],
        emotion_ids=["happy", "happy", "fear", "fear"],
    )
    separated_query = make_latent_index(
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        episode_indices=[10, 11],
        behavior_ids=["wave", "alert"],
        emotion_ids=["happy", "fear"],
    )
    collapsed_query = make_latent_index(
        embeddings=np.ones((2, 2)),
        episode_indices=[10, 11],
        behavior_ids=["wave", "alert"],
        emotion_ids=["happy", "fear"],
    )

    assert cross_set_retrieval_loss(reference, separated_query) < cross_set_retrieval_loss(reference, collapsed_query)


def test_motion_metric_loss_combines_finite_training_objectives():
    model = MotionMetricEncoder(action_dim=15, latent_dim=16, hidden_dim=32, behavior_classes=2, emotion_classes=2)
    output = model(torch.randn(4, 30, 12))
    behavior = torch.tensor([0, 0, 1, 1])
    emotion = torch.tensor([0, 0, 1, 1])

    losses = motion_metric_loss(output, behavior, emotion, torch.randn(4, 6))

    assert set(losses) == {"total", "behavior", "emotion", "contrastive", "descriptor"}
    assert all(value.ndim == 0 and torch.isfinite(value) for value in losses.values())
    losses["total"].backward()


def test_semantic_pair_batch_places_distinct_positive_examples_next_to_each_other():
    episodes = make_grouped_episodes(samples_per_group=4)

    batch = sample_semantic_pair_batch(episodes, batch_size=8, seed=9)

    assert len(batch) == 8
    for index in range(0, len(batch), 2):
        first, second = batch[index : index + 2]
        assert first["episode_index"] != second["episode_index"]
        assert first["meta"]["behavior_id"] == second["meta"]["behavior_id"]
        assert first["meta"]["emotion_id"] == second["meta"]["emotion_id"]


def make_latent_index(embeddings, episode_indices, behavior_ids, emotion_ids):
    return MotionLatentIndex(
        embeddings=np.asarray(embeddings, dtype=np.float32),
        episode_indices=np.asarray(episode_indices, dtype=np.int64),
        behavior_ids=np.asarray(behavior_ids),
        emotion_ids=np.asarray(emotion_ids),
    )


def test_latent_diagnostics_queries_train_references_and_reports_separation():
    reference = make_latent_index(
        embeddings=[[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0], [0.1, 0.9, 0.0]],
        episode_indices=[0, 1, 2, 3],
        behavior_ids=["wave", "wave", "alert", "alert"],
        emotion_ids=["happy", "happy", "fear", "fear"],
    )
    query = make_latent_index(
        embeddings=[[0.95, 0.05, 0.0], [0.05, 0.95, 0.0]],
        episode_indices=[100, 101],
        behavior_ids=["wave", "alert"],
        emotion_ids=["happy", "fear"],
    )

    report = compute_latent_diagnostics(reference, query)

    assert report["reference_count"] == 4
    assert report["query_count"] == 2
    assert report["knn_behavior_accuracy"] == 1.0
    assert report["knn_emotion_accuracy"] == 1.0
    assert report["knn_joint_accuracy"] == 1.0
    assert report["mean_inter_class_distance"] > report["mean_intra_class_distance"]
    assert report["separation_ratio"] > 1.0
    assert report["effective_rank"] > 1.0


def test_latent_diagnostics_marks_identical_embeddings_as_collapsed():
    reference = make_latent_index(
        embeddings=np.ones((4, 3)),
        episode_indices=[0, 1, 2, 3],
        behavior_ids=["wave", "wave", "alert", "alert"],
        emotion_ids=["happy", "happy", "fear", "fear"],
    )
    query = make_latent_index(
        embeddings=np.ones((2, 3)),
        episode_indices=[10, 11],
        behavior_ids=["wave", "alert"],
        emotion_ids=["happy", "fear"],
    )

    report = compute_latent_diagnostics(reference, query)

    assert report["collapsed"] is True
    assert report["effective_rank"] <= 1.0
    assert all(np.isfinite(value) for value in report.values() if isinstance(value, float))


def test_latent_diagnostics_detects_collapsed_queries_even_when_references_are_diverse():
    reference = make_latent_index(
        embeddings=[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        episode_indices=[0, 1, 2, 3],
        behavior_ids=["wave", "wave", "alert", "alert"],
        emotion_ids=["happy", "happy", "fear", "fear"],
    )
    query = make_latent_index(
        embeddings=np.ones((2, 2)),
        episode_indices=[10, 11],
        behavior_ids=["wave", "alert"],
        emotion_ids=["happy", "fear"],
    )

    report = compute_latent_diagnostics(reference, query)

    assert report["reference_collapsed"] is False
    assert report["query_collapsed"] is True
    assert report["collapsed"] is True
    assert report["effective_rank"] == report["query_effective_rank"]


def test_prototype_medoids_cover_each_train_group_without_using_held_out_ids():
    train = make_latent_index(
        embeddings=[
            [0.98, -0.20],
            [1.00, 0.00],
            [0.94, 0.34],
            [-0.20, 0.98],
            [0.00, 1.00],
            [0.34, 0.94],
        ],
        episode_indices=[1, 2, 3, 4, 5, 6],
        behavior_ids=["wave", "wave", "wave", "alert", "alert", "alert"],
        emotion_ids=["happy", "happy", "happy", "fear", "fear", "fear"],
    )

    prototypes = select_prototype_medoids(train, per_group=1)

    assert {(item["behavior_id"], item["emotion_id"]) for item in prototypes} == {
        ("wave", "happy"),
        ("alert", "fear"),
    }
    assert {item["episode_indices"][0] for item in prototypes} == {2, 5}
    assert not ({100, 101} & {episode for item in prototypes for episode in item["episode_indices"]})

    multiple = select_prototype_medoids(train, per_group=2)
    assert all(len(item["episode_indices"]) == 2 for item in multiple)
    assert all(len(set(item["episode_indices"])) == 2 for item in multiple)


def test_raw_motion_index_exports_downsampled_position_velocity_baseline():
    episodes = make_grouped_episodes(samples_per_group=3)
    stats = compute_motion_normalization(episodes)

    index = build_raw_motion_index(episodes, stats, stride=3)

    assert index.embeddings.shape == (12, 30 * 4)
    assert index.episode_indices.tolist() == list(range(12))


def test_motion_latent_training_config_parses_yaml_defaults_and_overrides(tmp_path):
    config_path = tmp_path / "motion_latent.yaml"
    config_path.write_text(
        f"""
dataset_dir: {tmp_path / 'dataset'}
output_dir: {tmp_path / 'output'}
steps: 25
batch_size: 8
latent_dim: 32
hidden_dim: 48
lr: 0.0002
device: cpu
seed: 13
prototypes_per_group: 2
""",
        encoding="utf-8",
    )

    args = training_args_from_config(load_train_config(config_path))

    assert args.dataset_dir == str(tmp_path / "dataset")
    assert args.output_dir == str(tmp_path / "output")
    assert args.steps == 25
    assert args.batch_size == 8
    assert args.latent_dim == 32
    assert args.hidden_dim == 48
    assert args.lr == 0.0002
    assert args.device == "cpu"
    assert args.seed == 13
    assert args.prototypes_per_group == 2
    assert args.deterministic is True


def test_motion_latent_training_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown motion latent config keys"):
        training_args_from_config(
            {
                "dataset_dir": "/tmp/data",
                "output_dir": "/tmp/output",
                "prototype_per_group": 3,
            }
        )


def test_train_motion_latent_writes_checkpoint_indices_diagnostics_and_train_prototypes(tmp_path):
    episodes = make_grouped_episodes(samples_per_group=4)
    output_dir = tmp_path / "motion_latent_run"

    summary = train_motion_latent(
        episodes,
        output_dir=output_dir,
        steps=2,
        batch_size=8,
        lr=1e-3,
        latent_dim=8,
        hidden_dim=16,
        device="cpu",
        seed=5,
        log_interval=1,
        prototypes_per_group=2,
    )

    expected_files = {
        "motion_latent_checkpoint.pt",
        "embeddings.npz",
        "diagnostics.json",
        "prototypes.json",
        "progress.jsonl",
        "training_summary.json",
    }
    assert expected_files <= {path.name for path in output_dir.iterdir()}
    diagnostics = json.loads((output_dir / "diagnostics.json").read_text(encoding="utf-8"))
    prototypes = json.loads((output_dir / "prototypes.json").read_text(encoding="utf-8"))
    progress = [json.loads(line) for line in (output_dir / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
    embedding_data = np.load(output_dir / "embeddings.npz")

    assert set(diagnostics) == {"learned", "raw_feature_baseline", "comparison"}
    assert set(diagnostics["learned"]) == {"validation", "test"}
    assert set(diagnostics["raw_feature_baseline"]) == {"validation", "test"}
    assert diagnostics["comparison"]["raw_feature_dim"] > diagnostics["comparison"]["latent_dim"]
    assert len(prototypes) == 4
    assert all(len(item["episode_indices"]) == 2 for item in prototypes)
    assert len(progress) == 2
    assert embedding_data["embeddings"].shape == (16, 8)
    assert set(embedding_data["split"].tolist()) == {"train", "validation", "test"}
    held_out_ids = set(embedding_data["episode_indices"][embedding_data["split"] != "train"].tolist())
    prototype_ids = {episode for item in prototypes for episode in item["episode_indices"]}
    assert held_out_ids.isdisjoint(prototype_ids)
    checkpoint = torch.load(output_dir / "motion_latent_checkpoint.pt", map_location="cpu", weights_only=False)
    train_ids = set(checkpoint["split_episode_indices"]["train"])
    expected_stats = compute_motion_normalization([episode for episode in episodes if episode["episode_index"] in train_ids])
    assert torch.allclose(checkpoint["normalization"]["mean"], expected_stats["mean"])
    assert torch.allclose(checkpoint["normalization"]["std"], expected_stats["std"])
    best_event = min(progress, key=lambda event: (event["validation_retrieval_loss"], event["validation_loss"]))
    assert checkpoint["best_step"] == best_event["step"]
    assert checkpoint["best_validation_loss"] == best_event["validation_loss"]
    assert checkpoint["best_validation_retrieval_loss"] == best_event["validation_retrieval_loss"]
    loaded_model, loaded_stats, loaded_checkpoint = load_motion_metric_checkpoint(
        output_dir / "motion_latent_checkpoint.pt"
    )
    first_episode = episodes[0]
    reencoded = encode_motion_episodes(loaded_model, [first_episode], loaded_stats)
    stored_row = int(np.flatnonzero(embedding_data["episode_indices"] == first_episode["episode_index"])[0])
    assert np.allclose(reencoded.embeddings[0], embedding_data["embeddings"][stored_row], atol=1e-6)
    assert loaded_checkpoint["config"]["deterministic"] is True
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert summary["episodes"] == 16
    assert summary["best_step"] in {1, 2}


def test_train_motion_latent_checkpoints_best_model_before_diagnostics(tmp_path, monkeypatch):
    episodes = make_grouped_episodes(samples_per_group=4)
    output_dir = tmp_path / "interrupted_postprocess"

    def fail_diagnostics(*args, **kwargs):
        raise RuntimeError("diagnostic failure")

    monkeypatch.setattr(motion_latent_module, "compute_latent_diagnostics", fail_diagnostics)
    with pytest.raises(RuntimeError, match="diagnostic failure"):
        train_motion_latent(
            episodes,
            output_dir=output_dir,
            steps=1,
            batch_size=4,
            latent_dim=8,
            hidden_dim=16,
            device="cpu",
            log_interval=1,
        )

    assert (output_dir / "motion_latent_checkpoint.pt").is_file()


def test_train_motion_latent_rejects_nonempty_output_without_overwrite(tmp_path):
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    (output_dir / "old.txt").write_text("old run", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        train_motion_latent(
            make_grouped_episodes(samples_per_group=3),
            output_dir=output_dir,
            steps=1,
            batch_size=4,
            latent_dim=8,
            hidden_dim=16,
            device="cpu",
        )


def test_motion_latent_script_supports_direct_path_execution():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "training/scripts/train_motion_latent.py"), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Train and diagnose" in completed.stdout


def test_streaming_motion_latent_loader_reads_only_actions_and_semantic_labels(tmp_path):
    dataset_dir = tmp_path / "dataset"
    data_dir = dataset_dir / "data" / "chunk-000"
    meta_dir = dataset_dir / "meta"
    data_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)
    semantic_rows = []
    frame_rows = []
    for episode_index in range(4):
        semantic_rows.append(
            {
                "episode_index": episode_index,
                "behavior_id": "Behavior.GreetingOwner01" if episode_index < 2 else "Behavior.Alert",
                "emotion_id": "happy" if episode_index % 2 == 0 else "fear",
            }
        )
        for frame_index in range(5):
            frame_rows.append(
                {
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "observation.state": [float(episode_index + frame_index)] * 15,
                    "next.done": frame_index == 4,
                    "unused_large_text": "not needed" * 100,
                }
            )
    pq.write_table(pa.Table.from_pylist(semantic_rows), meta_dir / "semantic_index.parquet")
    pq.write_table(pa.Table.from_pylist(frame_rows), data_dir / "file-000.parquet")

    episodes = load_motion_latent_episodes(dataset_dir, max_episodes=3)

    assert len(episodes) == 3
    assert episodes[0]["actions"].shape == (5, 15)
    assert set(episodes[0]) == {"episode_index", "actions", "meta"}
    assert set(episodes[0]["meta"]) == {"behavior_id", "emotion_id"}
