import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.train_beat2_qwen_motion_alignment import motion_descriptor
from upper_body_skeleton.beat2_semantic_perceptual import (
    BEAT2_DATA_POLICY,
    BEAT2_JOINT_NAMES,
    Beat2MotionEncoder,
    Beat2SemanticPerceptualLoss,
    beat2_motion_descriptor_tensor,
    descriptor_dim,
    group_aware_contrastive_loss,
    load_train_qwen_group_prototypes,
    motion_to_global_prototype_info_nce,
)


def _write_test_artifacts(tmp_path: Path, *, no_kimodo: bool = True):
    width = descriptor_dim(24)
    descriptor_path = tmp_path / "descriptors.npz"
    metadata = {
        "artifact_kind": "beat2_18d_motion_descriptor_cache_v1",
        "data_policy": BEAT2_DATA_POLICY,
        "no_kimodo": no_kimodo,
        "manifest_sha256": "beat2-manifest",
        "csv_set_sha256": "beat2-csv-set",
        "phase_samples": 24,
        "fps": 30.0,
        "joint_names": list(BEAT2_JOINT_NAMES),
        "descriptor_dim": width,
        "normalization_fit_split": "train",
    }
    np.savez_compressed(
        descriptor_path,
        descriptor_mean=np.zeros(width, dtype=np.float32),
        descriptor_scale=np.ones(width, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )

    labels = {
        "categories": ["deictic", "iconic"],
        "intensities": ["low", "high"],
        "emotions": ["happy", "sad"],
        "groups": [
            ["deictic", "low", "happy"],
            ["deictic", "high", "sad"],
            ["iconic", "high", "happy"],
        ],
    }
    model = Beat2MotionEncoder(
        width,
        32,
        128,
        {
            "category": len(labels["categories"]),
            "intensity": len(labels["intensities"]),
            "emotion": len(labels["emotions"]),
            "group": len(labels["groups"]),
        },
        dropout=0.05,
    )
    checkpoint_path = tmp_path / "motion_encoder.pt"
    torch.save(
        {
            "artifact_kind": "beat2_random_init_motion_encoder_v1",
            "data_policy": BEAT2_DATA_POLICY,
            "no_kimodo": True,
            "initialization": "random_seeded_no_input_checkpoint",
            "input_checkpoint_count": 0,
            "source_manifest_sha256": "beat2-manifest",
            "csv_set_sha256": "beat2-csv-set",
            "step": 10,
            "model_config": {
                "input_dim": width,
                "hidden_dim": 32,
                "latent_dim": 128,
                "dropout": 0.05,
            },
            "label_contract": labels,
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    return descriptor_path, checkpoint_path


def _write_condition_cache(
    tmp_path: Path,
    *,
    inconsistent_train_group: bool = False,
) -> Path:
    path = tmp_path / "conditions_128d_frozen_base.npz"
    prototypes = np.zeros((3, 128), dtype=np.float32)
    prototypes[0, 0] = 1.0
    prototypes[1, 1] = 1.0
    prototypes[2, 2] = 1.0
    group_zero_second = prototypes[0].copy()
    if inconsistent_train_group:
        group_zero_second[:] = 0.0
        group_zero_second[3] = 1.0
    validation_variant = np.zeros(128, dtype=np.float32)
    validation_variant[4] = 1.0
    conditions = np.stack(
        (
            prototypes[0],
            group_zero_second,
            prototypes[1],
            prototypes[2],
            validation_variant,
        )
    )
    np.savez_compressed(
        path,
        conditions=conditions,
        motion_latents=conditions.copy(),
        semantic_group_indices=np.asarray([0, 0, 1, 2, 0], dtype=np.int64),
        fixed_split_assignments=np.asarray(
            ["train", "train", "train", "train", "validation"]
        ),
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = {
        "artifact_kind": "beat2_qwen_motion_latent_condition_cache_v1",
        "data_policy": BEAT2_DATA_POLICY,
        "no_kimodo": True,
        "variant": "frozen_base",
        "condition_dim": 128,
        "motion_latent_dim": 128,
        "condition_normalization": "unit_l2_per_canonical_prompt",
        "source_manifest_sha256": "beat2-manifest",
        "csv_set_sha256": "beat2-csv-set",
        "unique_condition_count": 3,
        "cache_sha256": digest,
    }
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(sidecar, sort_keys=True), encoding="utf-8"
    )
    return path


def test_full_descriptor_matches_existing_numpy_implementation():
    time = np.linspace(0.0, 2.0, 61, dtype=np.float32)
    actions = np.stack(
        [
            0.15 * np.sin(time * (joint + 1) / 4.0)
            + 0.02 * np.cos(time * (joint + 2) / 3.0)
            for joint in range(18)
        ],
        axis=1,
    ).astype(np.float32)
    expected = motion_descriptor(actions, phase_samples=24, fps=30.0)
    actual = beat2_motion_descriptor_tensor(
        torch.from_numpy(actions)[None],
        durations_sec=torch.tensor([2.0]),
        phase_samples=24,
    )[0]
    assert actual.shape == (742,)
    np.testing.assert_allclose(
        actual.detach().numpy(), expected, rtol=2e-5, atol=2e-5
    )


def test_padding_is_ignored_and_has_zero_gradient():
    generator = torch.Generator().manual_seed(41)
    valid = torch.randn(1, 9, 18, generator=generator)
    first = torch.cat((valid, torch.zeros(1, 7, 18)), dim=1).requires_grad_(True)
    second = torch.cat(
        (valid, 1000.0 * torch.randn(1, 7, 18, generator=generator)), dim=1
    )
    mask = torch.zeros(1, 16, dtype=torch.bool)
    mask[:, :9] = True
    duration = torch.tensor([8.0 / 30.0])
    descriptor_a = beat2_motion_descriptor_tensor(
        first, frame_mask=mask, durations_sec=duration
    )
    descriptor_b = beat2_motion_descriptor_tensor(
        second, frame_mask=mask, durations_sec=duration
    )
    assert torch.equal(descriptor_a.detach(), descriptor_b)

    descriptor_a.square().mean().backward()
    assert first.grad is not None
    assert torch.isfinite(first.grad).all()
    assert first.grad[:, :9].abs().sum() > 0
    assert torch.count_nonzero(first.grad[:, 9:]) == 0


def test_mask_must_be_a_contiguous_prefix():
    actions = torch.zeros(1, 6, 18)
    mask = torch.tensor([[True, True, False, True, False, False]])
    with pytest.raises(ValueError, match="contiguous valid prefix"):
        beat2_motion_descriptor_tensor(actions, frame_mask=mask)


def test_group_aware_contrastive_does_not_make_same_group_false_negatives():
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    groups = torch.tensor([4, 4, 9])
    loss = group_aware_contrastive_loss(
        embeddings, embeddings, group_ids=groups, temperature=0.01
    )
    assert loss < 1e-5


def test_train_prototype_loader_is_train_only_and_checks_group_consistency(
    tmp_path,
):
    path = _write_condition_cache(tmp_path)
    prototypes, groups, metadata = load_train_qwen_group_prototypes(
        path,
        expected_manifest_sha256="beat2-manifest",
        expected_csv_set_sha256="beat2-csv-set",
        expected_group_count=3,
    )
    assert prototypes.shape == (3, 128)
    assert groups.tolist() == [0, 1, 2]
    assert torch.equal(prototypes, torch.eye(128)[:3])
    assert metadata["fit_split"] == "train"
    assert metadata["validation_or_test_row_count_used"] == 0
    assert metadata["maximum_within_group_linf_deviation"] == 0.0
    assert metadata["train_group_counts"] == {"0": 2, "1": 1, "2": 1}


def test_train_prototype_loader_rejects_inconsistent_latents_or_aggregates_explicitly(
    tmp_path,
):
    path = _write_condition_cache(tmp_path, inconsistent_train_group=True)
    with pytest.raises(ValueError, match="not latent-consistent"):
        load_train_qwen_group_prototypes(
            path, expected_group_count=3, aggregation="require_identical"
        )
    prototypes, groups, metadata = load_train_qwen_group_prototypes(
        path,
        expected_group_count=3,
        aggregation="normalized_train_mean",
    )
    assert groups.tolist() == [0, 1, 2]
    assert torch.allclose(prototypes.norm(dim=-1), torch.ones(3))
    assert metadata["aggregation"] == "normalized_train_mean"
    assert metadata["maximum_within_group_linf_deviation"] == 1.0


def test_global_prototype_info_nce_uses_all_absent_groups_as_negatives():
    prototypes = torch.zeros(3, 128)
    prototypes[0, 0] = 1.0
    prototypes[1, 1] = 1.0
    prototypes[2, 2] = 1.0
    result = motion_to_global_prototype_info_nce(
        prototypes[[0, 2]],
        torch.tensor([0, 2]),
        prototypes,
        torch.tensor([0, 1, 2]),
        temperature=0.05,
    )
    assert result["loss"] < 1e-5
    assert result["recall_at_1"] == 1.0
    assert result["hard_margin"] == 1.0
    assert result["prototype_count"] == 3

    wrong = motion_to_global_prototype_info_nce(
        prototypes[[1]],
        torch.tensor([0]),
        prototypes,
        torch.tensor([0, 1, 2]),
        temperature=0.05,
    )
    assert wrong["recall_at_1"] == 0.0
    assert wrong["hard_margin"] == -1.0
    assert wrong["loss"] > 10.0


def test_artifact_loss_is_frozen_differentiable_and_reports_cross_group_metrics(
    tmp_path,
):
    torch.manual_seed(7)
    descriptor_path, checkpoint_path = _write_test_artifacts(tmp_path)
    condition_path = _write_condition_cache(tmp_path)
    module = Beat2SemanticPerceptualLoss.from_artifacts(
        descriptor_cache_path=descriptor_path,
        motion_encoder_checkpoint_path=checkpoint_path,
        qwen_condition_cache_path=condition_path,
        action_stats={
            "mean": torch.zeros(18),
            "std": torch.ones(18),
        },
        cosine_weight=0.5,
        contrastive_weight=0.0,
        global_contrastive_weight=1.0,
        temperature=0.1,
    )
    module.train()
    assert module.training
    assert not module.motion_encoder.training
    assert not any(parameter.requires_grad for parameter in module.motion_encoder.parameters())

    actions = 0.03 * torch.randn(4, 12, 18)
    # BEAT2 clips can contain exactly static joints; their zero RMS must not
    # create the undefined sqrt(0) backward seen in a literal NumPy port.
    actions[..., ::3] = 0.0
    actions.requires_grad_(True)
    mask = torch.zeros(4, 12, dtype=torch.bool)
    lengths = torch.tensor([12, 9, 8, 11])
    for row, length in enumerate(lengths.tolist()):
        mask[row, :length] = True
    durations = (lengths.float() - 1.0) / 30.0
    qwen = module.global_prototype_embeddings[[0, 1, 2, 0]].clone()
    qwen.requires_grad_(True)
    result = module(
        actions,
        qwen,
        frame_mask=mask,
        durations_sec=durations,
        group_ids=torch.tensor([0, 1, 2, 0]),
    )
    expected_fields = {
        "total",
        "cosine",
        "contrastive",
        "global_contrastive",
        "aligned_cosine",
        "same_group_cosine",
        "cross_group_cosine",
        "cross_group_cosine_gap",
        "hard_cross_group_margin",
        "hard_cross_group_margin_positive_fraction",
        "motion_to_text_group_recall_at_1",
        "text_to_motion_group_recall_at_1",
        "motion_encoder_group_accuracy",
        "global_hard_cross_group_margin",
        "global_hard_cross_group_margin_positive_fraction",
        "global_motion_to_prototype_recall_at_1",
        "global_mean_positive_rank",
        "global_prototype_count",
    }
    assert expected_fields.issubset(result)
    assert torch.isfinite(result["total"])
    result["total"].backward()
    assert actions.grad is not None
    assert torch.isfinite(actions.grad).all()
    assert actions.grad.abs().sum() > 0
    assert qwen.grad is None
    assert all(parameter.grad is None for parameter in module.motion_encoder.parameters())


def test_artifact_loader_rejects_non_beat2_only_descriptor(tmp_path):
    descriptor_path, checkpoint_path = _write_test_artifacts(
        tmp_path, no_kimodo=False
    )
    with pytest.raises(ValueError, match="no_kimodo"):
        Beat2SemanticPerceptualLoss.from_artifacts(
            descriptor_cache_path=descriptor_path,
            motion_encoder_checkpoint_path=checkpoint_path,
            action_stats={"mean": torch.zeros(18), "std": torch.ones(18)},
        )
