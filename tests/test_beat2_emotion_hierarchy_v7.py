from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools import train_beat2_style_emotion_v2 as base_trainer
from tools.train_beat2_emotion_hierarchy_v7 import (
    CONFIG_ARTIFACT_KIND,
    LABEL_CONTRACT,
    SCHEMA_VERSION,
    TRAINING_POLICY,
    _patched_v7_engine,
    read_config,
    validate_config,
)
from upper_body_skeleton.beat2_emotion_hierarchy import (
    EMOTION_ORDER,
    INTENDED_WEAK_LABEL_ROLE,
    SAMPLING_POLICY,
    Beat2EmotionHierarchyLoss,
    SourceGroupFirstNativeBucketSampler,
    derive_emotion_hierarchy_prototypes,
)


PROJECT_ROOT = Path(__file__).parents[1]


def _episode(
    clip_id: str,
    *,
    emotion: str = "happy",
    speaker: str = "speaker_a",
    source_group: str = "source_a",
    frames: int = 40,
):
    return {
        "clip_id": clip_id,
        "dataset": "BEAT2",
        "fixed_split_assignment": "train",
        "emotion_id": emotion,
        "speaker_key": speaker,
        "source_group_key": source_group,
        "actions": np.zeros((frames, 18), dtype=np.float32),
    }


def _batching():
    return {
        "max_motion_tokens_per_microbatch": 4096,
        "max_attention_elements_per_microbatch": 8_000_000,
    }


def _sample(
    sampler: SourceGroupFirstNativeBucketSampler, count: int
):
    return sampler.sample_microbatch(
        remaining_effective_batch=count,
        semantic_tokens=7,
        max_batch_size=count,
        batching=_batching(),
    )


def test_checked_in_v7_config_is_isolated_intended_weak_hierarchy():
    config = read_config(
        PROJECT_ROOT / "configs" / "beat2_emotion_hierarchy_v7.json"
    )
    assert config["schema_version"] == SCHEMA_VERSION
    assert config["artifact_kind"] == CONFIG_ARTIFACT_KIND
    assert config["training_policy"] == TRAINING_POLICY
    assert config["intended_emotion_label_contract"] == LABEL_CONTRACT
    assert config["emotion_hierarchy"]["label_role"] == (
        INTENDED_WEAK_LABEL_ROLE
    )
    assert config["emotion_hierarchy"]["weak_label_weight"] == 0.1
    assert config["emotion_hierarchy"]["group_auxiliary_weight"] == 0.1
    assert (
        config["emotion_hierarchy"]["schedule_mode"]
        == "simultaneous_hierarchy_no_stage_schedule_v1"
    )
    assert (
        config["emotion_hierarchy"]["human_perceived_emotion_truth"] is False
    )
    assert (
        config["emotion_supervision_ingress"][
            "expected_human_confirmed_observable_rows"
        ]
        == 0
    )
    assert config["training"]["sampler"]["mode"] == SAMPLING_POLICY
    outer = config["semantic_perceptual"]["outer_weight"]
    weak = config["emotion_hierarchy"]["weak_label_weight"]
    binary = (
        outer * weak * config["emotion_hierarchy"]["binary_weight"]
    )
    emotion = (
        outer * weak * config["emotion_hierarchy"]["emotion_weight"]
    )
    cosine = outer * config["semantic_perceptual"]["cosine_weight"]
    group = (
        outer
        * weak
        * config["emotion_hierarchy"]["group_auxiliary_weight"]
    )
    assert binary == pytest.approx(0.02)
    assert emotion == pytest.approx(0.02)
    assert cosine == pytest.approx(0.002)
    assert group == pytest.approx(0.002)
    assert min(binary, emotion) >= 10 * max(cosine, group)
    assert "emotion_hierarchy_v7" in config["output_dir"]
    assert "global54_semantic_v6" not in config["output_dir"]


def test_v7_config_rejects_forbidden_external_token_anywhere(tmp_path):
    raw = json.loads(
        (
            PROJECT_ROOT / "configs" / "beat2_emotion_hierarchy_v7.json"
        ).read_text(encoding="utf-8")
    )
    raw["output_dir"] = str(tmp_path / "forbidden-kimodo-run")
    with pytest.raises(ValueError, match="forbidden external-data token"):
        validate_config(raw)


def test_v7_config_rejects_auxiliary_that_dominates_primary():
    raw = json.loads(
        (
            PROJECT_ROOT / "configs" / "beat2_emotion_hierarchy_v7.json"
        ).read_text(encoding="utf-8")
    )
    raw["semantic_perceptual"]["cosine_weight"] = 0.25
    with pytest.raises(ValueError, match="at least ten times"):
        validate_config(raw)


def test_v7_engine_patch_is_scoped_and_restored():
    config = read_config(
        PROJECT_ROOT / "configs" / "beat2_emotion_hierarchy_v7.json"
    )
    original_schema = base_trainer.SCHEMA_VERSION
    original_sampler = base_trainer.TemperedGroupNativeBucketSampler
    with _patched_v7_engine(config):
        assert base_trainer.SCHEMA_VERSION == SCHEMA_VERSION
        assert (
            base_trainer.TemperedGroupNativeBucketSampler
            is SourceGroupFirstNativeBucketSampler
        )
    assert base_trainer.SCHEMA_VERSION == original_schema
    assert base_trainer.TemperedGroupNativeBucketSampler is original_sampler


def test_source_group_is_selected_before_event_row_count():
    rows = [_episode("short_source", source_group="source_short")]
    rows.extend(
        _episode(f"long_event_{index}", source_group="source_long")
        for index in range(12)
    )
    sampler = SourceGroupFirstNativeBucketSampler(
        rows, buckets=(48,), seed=123
    )
    counts = {"source_short": 0, "source_long": 0}
    for _ in range(4000):
        batch, _ = _sample(sampler, 1)
        counts[batch[0]["source_group_key"]] += 1
    short_fraction = counts["source_short"] / sum(counts.values())
    assert short_fraction == pytest.approx(0.5, abs=0.04)


def test_source_groups_do_not_repeat_inside_microbatch_when_available():
    rows = [
        _episode(
            f"clip_{index}",
            emotion=EMOTION_ORDER[index % len(EMOTION_ORDER)],
            speaker=f"speaker_{index % 3}",
            source_group=f"source_{index}",
        )
        for index in range(12)
    ]
    sampler = SourceGroupFirstNativeBucketSampler(
        rows, buckets=(48,), seed=44
    )
    batch, plan = _sample(sampler, 8)
    sources = [row["source_group_key"] for row in batch]
    assert len(set(sources)) == len(sources)
    assert plan["unique_source_groups"] == len(sources)
    assert plan["sampling_policy"] == SAMPLING_POLICY


def test_source_group_sampler_resume_is_exact():
    rows = [
        _episode(
            f"clip_{index}",
            emotion=EMOTION_ORDER[index % len(EMOTION_ORDER)],
            speaker=f"speaker_{index % 4}",
            source_group=f"source_{index}",
            frames=40 if index % 2 else 60,
        )
        for index in range(24)
    ]
    first = SourceGroupFirstNativeBucketSampler(
        rows, buckets=(48, 64), seed=91
    )
    _sample(first, 5)
    state = deepcopy(first.state_dict())
    expected = []
    for _ in range(6):
        batch, plan = _sample(first, 4)
        expected.append(
            ([row["clip_id"] for row in batch], deepcopy(plan))
        )

    resumed = SourceGroupFirstNativeBucketSampler(
        rows, buckets=(48, 64), seed=91
    )
    resumed.load_state_dict(state)
    actual = []
    for _ in range(6):
        batch, plan = _sample(resumed, 4)
        actual.append(([row["clip_id"] for row in batch], deepcopy(plan)))
    assert actual == expected


def test_source_group_sampler_rejects_non_beat2_and_forbidden_path():
    wrong_dataset = _episode("wrong")
    wrong_dataset["dataset"] = "external"
    with pytest.raises(ValueError, match="BEAT2 only"):
        SourceGroupFirstNativeBucketSampler(
            [wrong_dataset], buckets=(48,), seed=1
        )
    forbidden = _episode("forbidden")
    forbidden["source_path"] = "/tmp/kimodo/source.csv"
    with pytest.raises(ValueError, match="forbidden external-data token"):
        SourceGroupFirstNativeBucketSampler(
            [forbidden], buckets=(48,), seed=1
        )


def _prototype_bank():
    generator = torch.Generator().manual_seed(17)
    groups = torch.randn(54, 128, generator=generator)
    group_ids = torch.arange(54, dtype=torch.long)
    mapping = {
        group: EMOTION_ORDER[group // 9] for group in range(54)
    }
    return derive_emotion_hierarchy_prototypes(
        groups, group_ids, mapping
    )


def test_hierarchy_prototypes_are_two_six_and_fifty_four():
    bank = _prototype_bank()
    assert bank.binary_embeddings.shape == (2, 128)
    assert bank.emotion_embeddings.shape == (6, 128)
    assert bank.group_embeddings.shape == (54, 128)
    assert torch.allclose(
        bank.binary_embeddings.norm(dim=-1), torch.ones(2)
    )
    assert torch.allclose(
        bank.emotion_embeddings.norm(dim=-1), torch.ones(6)
    )
    assert all(
        len(value) == 9
        for value in bank.metadata["groups_per_emotion"].values()
    )
    assert bank.metadata["label_role"] == INTENDED_WEAK_LABEL_ROLE
    assert bank.metadata["validation_or_test_rows_used"] == 0


def test_hierarchy_loss_uses_weak_weight_and_small_group_auxiliary():
    bank = _prototype_bank()
    module = Beat2EmotionHierarchyLoss(
        bank,
        binary_weight=1.0,
        emotion_weight=1.0,
        group_auxiliary_weight=0.1,
        weak_label_weight=0.1,
        temperature=0.07,
    )
    motion = torch.randn(6, 128, requires_grad=True)
    group_ids = torch.tensor([0, 9, 18, 27, 36, 45])
    result = module(motion, group_ids)
    expected = 0.1 * (
        result["binary_loss"]
        + result["emotion_loss"]
        + 0.1 * result["group_auxiliary_loss"]
    )
    assert torch.allclose(result["total"], expected)
    assert torch.isfinite(result["total"])
    assert int(result["binary_prototype_count"]) == 2
    assert int(result["emotion_prototype_count"]) == 6
    assert int(result["group_auxiliary_prototype_count"]) == 54
    result["total"].backward()
    assert motion.grad is not None
    assert torch.isfinite(motion.grad).all()


def test_group_objective_cannot_become_a_primary_weight():
    with pytest.raises(ValueError, match="small positive auxiliary"):
        Beat2EmotionHierarchyLoss(
            _prototype_bank(),
            binary_weight=1.0,
            emotion_weight=1.0,
            group_auxiliary_weight=0.5,
            weak_label_weight=0.1,
        )
