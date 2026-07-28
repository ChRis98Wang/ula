import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from tools.train_beat2_qwen_motion_alignment import (
    DEFAULT_CONFIG,
    TextAlignmentHead,
    _diagnostic_template_probe_prompts,
    _first_positive_ranks,
    export_128d_condition_cache,
    motion_descriptor,
    validate_clean_generator_foundation,
    validate_config,
)


def test_config_fail_closed_and_requires_equal_ab_budget():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["manifest_path"] = "/datasets/kimodo_forbidden/train.jsonl"
    with pytest.raises(ValueError, match="forbidden external-data token"):
        validate_config(config)

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["alignment"]["lora_steps"] += 1
    with pytest.raises(ValueError, match="must match"):
        validate_config(config)

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["motion"]["latent_dim"] = 64
    with pytest.raises(ValueError, match="must both be 128"):
        validate_config(config)


def test_generator_foundation_must_pass_clean_motion_only_contract(tmp_path):
    checkpoint = tmp_path / "unqualified-foundation.pt"
    torch.save({"artifact_kind": "unqualified"}, checkpoint)
    with pytest.raises(ValueError, match="not a contract-valid random-init"):
        validate_clean_generator_foundation(checkpoint)


def test_motion_descriptor_is_fixed_width_finite_and_deterministic():
    time = np.linspace(0.0, 2.0, 61, dtype=np.float32)
    actions = np.stack(
        [np.sin(time * (joint + 1) / 4.0) for joint in range(18)], axis=1
    )
    first = motion_descriptor(actions, phase_samples=24)
    second = motion_descriptor(actions.copy(), phase_samples=24)
    assert first.shape == (742,)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)


def test_positive_retrieval_ranks_accept_multiple_motion_positives():
    similarity = np.asarray(
        [
            [0.1, 0.9, 0.8, 0.0],
            [0.7, 0.6, 0.5, 0.4],
        ],
        dtype=np.float32,
    )
    positives = np.asarray(
        [
            [False, False, True, False],
            [False, True, False, True],
        ]
    )
    assert _first_positive_ranks(similarity, positives).tolist() == [2, 2]


def test_probe_templates_are_held_out_and_projector_is_128d():
    labels = {
        "categories": ["deictic"],
        "intensities": ["high", "low"],
        "emotions": ["happy"],
        "groups": [
            ("deictic", "high", "happy"),
            ("deictic", "low", "happy"),
        ],
    }
    canonical = [
        "Perform a high-intensity deictic pointing gesture with a happy affect.",
        "Perform a low-intensity deictic pointing gesture with a happy affect.",
    ]
    probes = _diagnostic_template_probe_prompts(labels, canonical)
    assert len(probes) == 2
    assert set(probes).isdisjoint(canonical)

    head = TextAlignmentHead(
        32,
        16,
        128,
        {"category": 1, "intensity": 2, "emotion": 1},
        dropout=0.0,
    )
    output = head(torch.randn(2, 32))
    assert output["embedding"].shape == (2, 128)
    assert torch.allclose(
        output["embedding"].norm(dim=-1), torch.ones(2), atol=1e-5
    )


def test_exported_condition_cache_is_direct_128d_and_source_bound(tmp_path):
    checkpoint = tmp_path / "adapter.pt"
    checkpoint.write_bytes(b"official-base-derived-test-adapter")
    rows = [
        {
            "clip_id": "clip-a",
            "safe_csv_sha256": hashlib.sha256(b"a").hexdigest(),
        },
        {
            "clip_id": "clip-b",
            "safe_csv_sha256": hashlib.sha256(b"b").hexdigest(),
        },
    ]
    cache = {
        "task_ids": np.asarray(["task-a", "task-b"]),
        "prompts": np.asarray(["prompt-a", "prompt-b"]),
        "split_names": np.asarray(["train", "test"]),
        "speaker_keys": np.asarray(["speaker-a", "speaker-b"]),
        "group_labels": np.asarray([0, 1], dtype=np.int64),
    }
    split_manifest = {
        "manifest": {"sha256": "manifest-sha"},
        "csv_set_sha256": "csv-set-sha",
        "prompt_set_sha256": "prompt-set-sha",
        "labels": {
            "groups": [
                ["deictic", "high", "happy"],
                ["iconic", "low", "sad"],
            ]
        },
    }
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["output_dir"] = str(tmp_path)
    latents = np.zeros((2, 128), dtype=np.float32)
    latents[0, 0] = 1.0
    latents[1, 1] = 1.0
    path, metadata = export_128d_condition_cache(
        variant="frozen_base",
        rows=rows,
        cache=cache,
        split_manifest=split_manifest,
        group_text_embeddings=latents,
        adapter_checkpoint_path=checkpoint,
        qwen_metadata={"model_name": "Qwen/Qwen3-Embedding-0.6B", "revision": "pin"},
        config=config,
    )
    with np.load(path, allow_pickle=False) as payload:
        assert payload["conditions"].shape == (2, 128)
        assert np.array_equal(payload["conditions"], payload["motion_latents"])
        assert payload["clip_ids"].tolist() == ["clip-a", "clip-b"]
    stored = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    assert stored == metadata
    assert metadata["no_kimodo"] is True
    assert metadata["base_condition_dim"] == 0
    assert metadata["condition_dim"] == 128
