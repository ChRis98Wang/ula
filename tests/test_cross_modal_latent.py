import numpy as np
import pytest
import torch

from training.scripts.train_qwen_motion_latent import validated_train_config
from upper_body_skeleton.cross_modal_latent import (
    CrossModalBatchSampler,
    LoRAMotionConditionBuilder,
    QwenMotionLatentAligner,
    TextMotionPrediction,
    bidirectional_alignment_loss,
    build_cross_modal_splits,
    cross_modal_training_loss,
    set_motion_trainable_policy,
    validation_selection_score,
    variance_covariance_loss,
)
from upper_body_skeleton.kimodo_semantics import (
    KIMODO_BEHAVIOR_IDS,
    KIMODO_CONDITION_CONTRACT_VERSION,
    KIMODO_CONDITION_SCHEMA_VERSION,
    KIMODO_EMOTION_IDS,
    kimodo_condition_vectors_sha256,
)
from upper_body_skeleton.motion_latent import MotionMetricEncoder
from upper_body_skeleton.semantic_adapter import SemanticPromptRecord


def _split_fixture():
    episodes = []
    records = []
    split_names = []
    episode_indices = {"train": [], "validation": [], "test": []}
    episode_index = 0
    for behavior_id in KIMODO_BEHAVIOR_IDS:
        for emotion_id in KIMODO_EMOTION_IDS:
            for split_name in ("train", "validation", "test"):
                episodes.append(
                    {
                        "episode_index": episode_index,
                        "actions": np.zeros((8, 15), dtype=np.float32),
                        "meta": {"behavior_id": behavior_id, "emotion_id": emotion_id},
                    }
                )
                episode_indices[split_name].append(episode_index)
                episode_index += 1
                records.append(
                    SemanticPromptRecord(
                        behavior_id,
                        emotion_id,
                        f"{split_name} {behavior_id} {emotion_id}",
                    )
                )
                split_names.append(split_name)
    checkpoint = {"split_episode_indices": episode_indices}
    return episodes, checkpoint, records, split_names


def _minimal_config():
    return {
        "dataset_dir": "dataset",
        "prompt_csv": "prompts.csv",
        "paraphrases_json": "paraphrases.json",
        "motion_checkpoint": "motion.pt",
        "output_dir": "output",
        "revision": "revision",
    }


def test_cross_modal_split_manifest_is_disjoint_and_complete():
    episodes, checkpoint, records, split_names = _split_fixture()

    splits, manifest = build_cross_modal_splits(episodes, checkpoint, records, split_names)

    assert set(splits) == {"train", "validation", "test"}
    assert manifest["counts"]["train"]["semantic_groups"] == 162
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert set(manifest["episode_indices"][left]).isdisjoint(manifest["episode_indices"][right])
        assert set(manifest["text_hashes"][left]).isdisjoint(manifest["text_hashes"][right])


def test_cross_modal_split_rejects_normalized_text_leakage():
    episodes, checkpoint, records, split_names = _split_fixture()
    train_row = split_names.index("train")
    validation_row = split_names.index("validation")
    leaked = records[validation_row]
    records[validation_row] = SemanticPromptRecord(
        leaked.behavior_id,
        leaked.emotion_id,
        f"  {records[train_row].text.upper()}  ",
    )

    with pytest.raises(ValueError, match="text hash"):
        build_cross_modal_splits(episodes, checkpoint, records, split_names)


def test_cross_modal_sampler_uses_unique_semantic_groups():
    episodes, checkpoint, records, split_names = _split_fixture()
    splits, _ = build_cross_modal_splits(episodes, checkpoint, records, split_names)
    sampler = CrossModalBatchSampler(splits["train"], seed=3)

    texts, motions, keys = sampler.sample(24)

    assert len(texts) == len(motions) == len(keys) == 24
    assert len(set(keys)) == 24
    assert all(motion["meta"]["behavior_id"] == key[0] for motion, key in zip(motions, keys))


def test_alignment_and_collapse_regularization_are_differentiable():
    text_raw = torch.randn(8, 16, requires_grad=True)
    motion_raw = torch.randn(8, 16, requires_grad=True)
    text_embedding = torch.nn.functional.normalize(text_raw, dim=-1)
    motion_embedding = torch.nn.functional.normalize(motion_raw, dim=-1)

    alignment = bidirectional_alignment_loss(text_embedding, motion_embedding)
    regularization = variance_covariance_loss(text_raw, motion_raw)
    total = alignment + regularization["variance"] + 0.01 * regularization["covariance"]
    total.backward()

    assert torch.isfinite(total)
    assert text_raw.grad is not None and torch.isfinite(text_raw.grad).all()
    assert motion_raw.grad is not None and torch.isfinite(motion_raw.grad).all()
    collapsed = variance_covariance_loss(torch.zeros(8, 16), torch.zeros(8, 16))
    assert collapsed["variance"] > 0.9


def test_full_cross_modal_loss_updates_both_modalities():
    batch_size = 8
    latent_dim = 16
    text_raw = torch.randn(batch_size, latent_dim, requires_grad=True)
    motion_raw = torch.randn(batch_size, latent_dim, requires_grad=True)
    metric_latent = torch.randn(batch_size, latent_dim, requires_grad=True)
    metric_embedding = torch.nn.functional.normalize(metric_latent, dim=-1)
    teacher = torch.nn.functional.normalize(torch.randn(batch_size, latent_dim), dim=-1)
    output = {
        "text": {
            "raw": text_raw,
            "embedding": torch.nn.functional.normalize(text_raw, dim=-1),
            "behavior_logits": torch.randn(batch_size, len(KIMODO_BEHAVIOR_IDS), requires_grad=True),
            "emotion_logits": torch.randn(batch_size, len(KIMODO_EMOTION_IDS), requires_grad=True),
        },
        "motion": {
            "raw": motion_raw,
            "embedding": torch.nn.functional.normalize(motion_raw, dim=-1),
            "teacher_embedding": teacher,
            "metric_output": {
                "embedding": metric_embedding,
                "behavior_logits": torch.randn(batch_size, len(KIMODO_BEHAVIOR_IDS), requires_grad=True),
                "emotion_logits": torch.randn(batch_size, len(KIMODO_EMOTION_IDS), requires_grad=True),
                "descriptors": torch.randn(batch_size, 6, requires_grad=True),
            },
        },
    }
    behavior = torch.arange(batch_size) % len(KIMODO_BEHAVIOR_IDS)
    emotion = torch.arange(batch_size) % len(KIMODO_EMOTION_IDS)

    losses = cross_modal_training_loss(output, behavior, emotion, torch.zeros(batch_size, 6))
    losses["total"].backward()

    assert set(losses) == {
        "total",
        "alignment",
        "text_behavior",
        "text_emotion",
        "motion_metric",
        "motion_anchor",
        "variance",
        "covariance",
    }
    assert text_raw.grad is not None
    assert motion_raw.grad is not None
    assert metric_latent.grad is not None


def test_motion_trainable_policy_only_unfreezes_requested_layers():
    model = MotionMetricEncoder(action_dim=15, latent_dim=16, hidden_dim=32)

    names = set_motion_trainable_policy(model, prefixes=("projection.",))

    assert names
    assert all(name.startswith("projection.") for name in names)
    assert all(parameter.requires_grad == name.startswith("projection.") for name, parameter in model.named_parameters())


def test_qwen_motion_config_rejects_unsafe_attention_and_unknown_keys():
    config = validated_train_config(_minimal_config())
    assert config["attention_backend"] == "eager"
    assert config["lora_rank"] == 8

    with pytest.raises(ValueError, match="attention_backend"):
        validated_train_config(_minimal_config() | {"attention_backend": "sdpa"})
    with pytest.raises(ValueError, match="unknown"):
        validated_train_config(_minimal_config() | {"surprise_option": True})


def test_validation_selection_prioritizes_bidirectional_recall():
    baseline = {
        "text_to_motion_recall_at_1": 0.2,
        "motion_to_text_recall_at_1": 0.2,
        "text_to_motion_recall_at_5": 0.5,
        "motion_to_text_recall_at_5": 0.5,
        "cosine_gap": 0.4,
        "retrieval_loss": 3.0,
    }
    higher_recall = baseline | {
        "text_to_motion_recall_at_1": 0.3,
        "retrieval_loss": 3.2,
    }

    assert validation_selection_score(higher_recall) > validation_selection_score(baseline)


def test_lora_motion_condition_builder_preserves_continuous_text_latent():
    vectors = np.zeros((len(KIMODO_BEHAVIOR_IDS), len(KIMODO_EMOTION_IDS), 136), dtype=np.float32)
    condition_bank = {
        "contract_version": KIMODO_CONDITION_CONTRACT_VERSION,
        "condition_schema_version": KIMODO_CONDITION_SCHEMA_VERSION,
        "condition_dim": 136,
        "behavior_ids": list(KIMODO_BEHAVIOR_IDS),
        "emotion_ids": list(KIMODO_EMOTION_IDS),
        "source_semantic_index_sha256": "0" * 64,
        "canonical_vectors_sha256": kimodo_condition_vectors_sha256(vectors),
        "vectors": torch.from_numpy(vectors),
    }
    latent = np.zeros(128, dtype=np.float32)
    latent[17] = 1.0

    class Encoder:
        def predict_one(self, text):
            return TextMotionPrediction(
                text=text,
                behavior_id=KIMODO_BEHAVIOR_IDS[3],
                emotion_id=KIMODO_EMOTION_IDS[2],
                behavior_confidence=0.8,
                emotion_confidence=0.9,
                motion_latent=latent,
            )

    builder = LoRAMotionConditionBuilder(Encoder(), condition_bank=condition_bank)
    condition = builder("new paraphrase", condition_dim=136)

    assert condition.shape == (136,)
    assert builder.last_prediction.behavior_id == KIMODO_BEHAVIOR_IDS[3]
    assert builder.last_prediction.emotion_id == KIMODO_EMOTION_IDS[2]
    assert np.array_equal(builder.last_motion_latent, latent)


def test_encode_motion_is_owned_by_cross_modal_aligner():
    assert "encode_motion" in QwenMotionLatentAligner.__dict__
    assert "encode_motion" not in LoRAMotionConditionBuilder.__dict__
