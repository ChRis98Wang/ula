import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.experimental import build_beat2_clean_abc_video as abc_video


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def manifest_record(clip_id: str, *, split: str = "test") -> dict:
    return {
        "clip_id": clip_id,
        "dataset": "BEAT2",
        "fixed_split_assignment": split,
        "accepted_for_training": True,
        "frames": 8,
        "fps": 30.0,
        "prompt": "test prompt",
        "motion_18d": {
            "action_dim": 18,
            "output_contract": "ula_v2_18d_head_v1",
            "quality_gate": {"passed": True},
            "safe_csv_sha256": "a" * 64,
        },
    }


def write_npz_sidecar(path: Path, metadata: dict, **arrays) -> None:
    np.savez_compressed(path, **arrays)
    metadata = dict(metadata)
    metadata["cache_sha256"] = abc_video.sha256_file(path)
    write_json(Path(str(path) + ".json"), metadata)


def test_compose_condition_enforces_exact_clean_slices():
    style = np.zeros(264, dtype=np.float32)
    style[133:136] = [1.0, -2.0, 0.5]
    latent = np.linspace(-1.0, 1.0, 128, dtype=np.float32)

    arm_a = abc_video.compose_condition(style, None)
    arm_b = abc_video.compose_condition(style, latent)

    assert np.count_nonzero(arm_a) == 3
    assert np.array_equal(arm_a[:133], np.zeros(133, dtype=np.float32))
    assert np.array_equal(arm_a[136:], np.zeros(128, dtype=np.float32))
    assert np.array_equal(arm_b[:133], np.zeros(133, dtype=np.float32))
    assert np.array_equal(arm_b[133:136], style[133:136])
    assert np.array_equal(arm_b[136:], latent)

    polluted = style.copy()
    polluted[7] = 1.0
    with pytest.raises(abc_video.EvaluationContractError, match="outside"):
        abc_video.compose_condition(polluted, None)


def test_trajectory_metrics_include_jerk_amplitude_and_head_activity():
    fps = 30.0
    time = np.arange(10, dtype=np.float64) / fps
    trajectory = np.zeros((10, 18), dtype=np.float64)
    trajectory[:, 0] = time**3
    trajectory[:, 17] = 0.2 * time

    metrics = abc_video.trajectory_metrics(trajectory, fps=fps)

    assert metrics["jerk_rad_s3"]["rms"] == pytest.approx(
        6.0 / np.sqrt(18), rel=1e-6
    )
    assert metrics["jerk_rad_s3"]["max"] == pytest.approx(6.0, rel=1e-6)
    assert metrics["amplitude"]["joint_range_max_rad"] > 0
    assert metrics["head_activity"]["velocity_rad_s"]["max"] == pytest.approx(0.2)
    assert metrics["head_activity"]["joint_names"] == list(
        abc_video.JOINT_ORDER_18D[15:18]
    )


def test_style_and_qwen_caches_are_clip_and_provenance_bound(tmp_path):
    records = {"clip-a": manifest_record("clip-a")}
    style_path = tmp_path / "style.npz"
    style_conditions = np.zeros((1, 264), dtype=np.float32)
    style_conditions[0, 133:136] = [0.1, 0.2, 0.3]
    write_npz_sidecar(
        style_path,
        {
            "artifact_kind": "ula_v2_18d_motion_only_style_condition_cache",
            "condition_dim": 264,
            "condition_policy": abc_video.EXPECTED_STYLE_POLICY,
            "kimodo_policy": abc_video.EXPECTED_KIMODO_POLICY,
            "qwen_policy": abc_video.EXPECTED_QWEN_DISABLED_POLICY,
            "condition_exact_zero_ranges": [[0, 133], [136, 264]],
            "condition_nonzero_indices": [133, 134, 135],
        },
        clip_ids=np.asarray(["clip-a"]),
        conditions=style_conditions,
    )
    loaded_style = abc_video.load_style_cache(
        style_path, manifest_records=records
    )
    assert np.array_equal(loaded_style["clip-a"], style_conditions[0])

    qwen_path = tmp_path / "qwen.npz"
    qwen_conditions = np.ones((1, 128), dtype=np.float32)
    manifest_sha = "b" * 64
    write_npz_sidecar(
        qwen_path,
        {
            "artifact_kind": abc_video.EXPECTED_QWEN_CACHE_KIND,
            "condition_dim": 128,
            "motion_latent_dim": 128,
            "data_policy": "beat2_only_no_external_motion_dataset_v1",
            "no_kimodo": True,
            "semantic_scope": abc_video.EXPECTED_QWEN_SCOPE,
            "source_manifest_sha256": manifest_sha,
            "variant": "frozen_base",
            "qwen": {
                "source": "official_huggingface_base",
                "input_checkpoint_kind": "official_base_only",
            },
        },
        clip_ids=np.asarray(["clip-a"]),
        conditions=qwen_conditions,
        fixed_split_assignments=np.asarray(["test"]),
        trajectory_sha256=np.asarray(["a" * 64]),
    )
    loaded_qwen, _ = abc_video.load_qwen_cache(
        qwen_path,
        variant="frozen_base",
        manifest_records=records,
        manifest_sha256=manifest_sha,
    )
    assert np.array_equal(loaded_qwen["clip-a"], qwen_conditions[0])

    polluted_metadata = json.loads(Path(str(qwen_path) + ".json").read_text())
    polluted_metadata["no_kimodo"] = False
    write_json(Path(str(qwen_path) + ".json"), polluted_metadata)
    with pytest.raises(abc_video.EvaluationContractError, match="BEAT2-only"):
        abc_video.load_qwen_cache(
            qwen_path,
            variant="frozen_base",
            manifest_records=records,
            manifest_sha256=manifest_sha,
        )


def test_completion_summary_is_terminal_and_checkpoint_bound(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    summary = tmp_path / "training_summary.json"
    write_json(
        summary,
        {
            "checkpoint": str(checkpoint),
            "completed_steps": 100,
            "target_steps": 100,
            "stopped_early": False,
        },
    )

    result = abc_video.validate_completion_summary(
        summary, checkpoint_path=checkpoint, branch="A"
    )
    assert result["completed_steps"] == 100

    write_json(
        summary,
        {
            "checkpoint": str(checkpoint),
            "completed_steps": 50,
            "target_steps": 100,
            "stopped_early": False,
        },
    )
    with pytest.raises(abc_video.EvaluationContractError, match="incomplete"):
        abc_video.validate_completion_summary(
            summary, checkpoint_path=checkpoint, branch="A"
        )


def test_pair_contract_uses_canonical_self_hash_not_file_hash(tmp_path):
    pair_path = tmp_path / "pair_contract.json"
    payload = {
        "artifact_kind": "pair",
        "unicode_note": "自哈希",
        "nested": {"b": 2, "a": 1},
    }
    payload["sha256"] = abc_video.self_hashed_mapping_sha256(payload)
    pair_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )

    assert abc_video.sha256_file(pair_path) != payload["sha256"]
    assert (
        abc_video.validate_self_hashed_json(
            pair_path,
            expected_sha256=payload["sha256"],
            field="pair contract",
        )
        == payload["sha256"]
    )

    payload["nested"]["a"] = 99
    write_json(pair_path, payload)
    with pytest.raises(abc_video.EvaluationContractError, match="SHA mismatch"):
        abc_video.validate_self_hashed_json(
            pair_path,
            expected_sha256=payload["sha256"],
            field="pair contract",
        )


def test_experimental_training_contract_matches_producer_schema():
    latent_weight = torch.zeros((8, 128), dtype=torch.float32)
    plan_weight = torch.zeros((4, 264), dtype=torch.float32)
    mask = torch.zeros_like(plan_weight)
    mask[:, 136:264] = 1.0
    contract = {
        "policy": abc_video.EXPECTED_EXPERIMENTAL_TRAINING_POLICY,
        "full_network": False,
        "trainable_tensor_names": list(
            abc_video.EXPECTED_TRAINABLE_TENSOR_NAMES
        ),
        "effective_trainable_parameter_names": list(
            abc_video.EXPECTED_EFFECTIVE_TRAINABLE_PARAMETER_NAMES
        ),
        "optimizer_parameter_count": latent_weight.numel() + plan_weight.numel(),
        "effective_trainable_parameter_count": (
            latent_weight.numel() + plan_weight.shape[0] * 128
        ),
        "plan_gradient_mask_sha256": __import__("hashlib").sha256(
            mask.numpy().tobytes()
        ).hexdigest(),
    }
    state = {
        abc_video.MOTION_LATENT_WEIGHT_NAME: latent_weight,
        abc_video.PLAN_WEIGHT_NAME: plan_weight,
    }

    receipt = abc_video.validate_experimental_training_contract(
        contract, state, branch="B"
    )
    assert receipt["effective_trainable_parameter_count"] == 1536

    polluted = dict(contract)
    polluted["full_network"] = True
    with pytest.raises(
        abc_video.EvaluationContractError, match="auditable trainable"
    ):
        abc_video.validate_experimental_training_contract(
            polluted, state, branch="B"
        )


def test_checkpoint_rejects_wrong_architecture_before_model_load(tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "artifact_kind": abc_video.CHECKPOINT_ARTIFACT_KIND,
            "architecture": "ula_mmdit_v2",
            "action_dim": 18,
            "condition_dim": 264,
            "joint_order": list(abc_video.JOINT_ORDER_18D),
            "config": {
                "hidden_dim": 384,
                "layers": 6,
                "semantic_tokens": 7,
                "initialization_mode": abc_video.EXPECTED_INITIALIZATION_MODE,
            },
        },
        path,
    )

    with pytest.raises(abc_video.EvaluationContractError, match="full AdaLN"):
        abc_video.validate_checkpoint(
            path,
            expected_manifest_sha256="b" * 64,
            branch="A",
        )


def test_experimental_loader_rejects_clean_checkpoint_contract(tmp_path):
    path = tmp_path / "clean.pt"
    torch.save(
        {
            "schema_version": 1,
            "artifact_kind": abc_video.CHECKPOINT_ARTIFACT_KIND,
            "architecture": abc_video.ULA_MMDIT_V3_ADALN_ARCHITECTURE,
            "action_dim": 18,
            "condition_dim": 264,
            "joint_order": list(abc_video.JOINT_ORDER_18D),
            "config": {
                "hidden_dim": 384,
                "layers": 6,
                "semantic_tokens": 7,
            },
        },
        path,
    )

    with pytest.raises(
        abc_video.EvaluationContractError, match="experimental AdaLN"
    ):
        abc_video.validate_experimental_checkpoint(
            path,
            branch="B",
            variant="frozen_base",
            expected_manifest_sha256="a" * 64,
            foundation_checkpoint_sha256="b" * 64,
            source_128d_cache_sha256="c" * 64,
            style_cache_sha256="d" * 64,
            manifest_records={},
            expected_conditions={},
            device="cpu",
        )


def test_heldout_cases_must_be_fixed_test_and_physical_qc():
    records = {
        "test": manifest_record("test"),
        "train": manifest_record("train", split="train"),
    }
    cases = abc_video._validate_cases(
        [{"clip_id": "test", "seed": 7}], manifest_records=records
    )
    assert cases[0]["frames"] == 8

    with pytest.raises(abc_video.EvaluationContractError, match="fixed test"):
        abc_video._validate_cases(
            [{"clip_id": "train", "seed": 7}], manifest_records=records
        )


def test_condition_sensitivity_counterfactuals_preserve_expected_channels():
    condition = np.zeros(264, dtype=np.float32)
    condition[133:136] = [1.0, 2.0, 3.0]
    condition[136:] = 0.25

    counterfactuals = abc_video._condition_counterfactuals(condition)

    assert set(counterfactuals) == {
        "zero_condition",
        "style_ablated",
        "latent_ablated",
    }
    assert not np.any(counterfactuals["zero_condition"])
    assert not np.any(counterfactuals["style_ablated"][133:136])
    assert np.array_equal(
        counterfactuals["style_ablated"][136:], condition[136:]
    )
    assert not np.any(counterfactuals["latent_ablated"][136:])
    assert np.array_equal(
        counterfactuals["latent_ablated"][133:136], condition[133:136]
    )
