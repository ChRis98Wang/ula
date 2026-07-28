import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import upper_body_skeleton.xem_beat2_domain_bridge as bridge


def test_common_features_ignore_entity_identity_and_entity_count():
    phase = np.linspace(0.0, 4.0 * np.pi, 96)
    activity = np.abs(np.sin(phase))[:, None]
    eighteen = np.repeat(activity, 18, axis=1)
    twenty_three = np.repeat(activity, 23, axis=1)

    first, first_names = bridge.common_angular_activity_features(eighteen)
    permuted, permuted_names = bridge.common_angular_activity_features(
        eighteen[:, np.random.default_rng(7).permutation(18)]
    )
    other_count, other_names = bridge.common_angular_activity_features(twenty_three)

    assert first.shape == (41,)
    assert first_names == permuted_names == other_names
    np.testing.assert_allclose(first, permuted, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(first, other_count, atol=1e-6, rtol=1e-6)
    assert not any("joint_" in name or "segment_" in name for name in first_names)


def test_weak_neutral_reference_uses_only_requested_split():
    rng = np.random.default_rng(3)
    features = rng.normal(size=(48, 41))
    rows = []
    for index in range(48):
        rows.append(
            {
                "split": "train" if index < 24 else "validation",
                "emotion_id": "neutral",
                "speaker_id": f"speaker_{index // 8}",
                "label_role": bridge.TARGET_LABEL_ROLE,
            }
        )
    features[24:] += 1_000.0

    center, scale, receipt = bridge.robust_neutral_reference(
        features,
        rows,
        split="train",
    )

    np.testing.assert_allclose(center, np.median(features[:24], axis=0))
    assert np.all(scale > 0.0)
    assert receipt["samples"] == 24
    assert receipt["weak_label_used"] is True
    assert receipt["source_split"] == "train"


def _write_controller_csv(path: Path) -> str:
    trajectory = np.zeros((12, 18), dtype=np.float64)
    trajectory[:, 0] = np.linspace(0.0, 0.5, 12)
    np.savetxt(
        path,
        trajectory,
        delimiter=",",
        header=",".join(f"joint_{index}" for index in range(18)),
        comments="",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_beat2_loader_rejects_speaker_crossing_fixed_splits(tmp_path: Path):
    rows = []
    for index, split in enumerate(("train", "validation")):
        csv_path = tmp_path / f"motion_{index}.csv"
        csv_sha = _write_controller_csv(csv_path)
        rows.append(
            {
                "dataset": "BEAT2",
                "accepted_for_training": True,
                "clip_id": f"clip_{index}",
                "fixed_split_assignment": split,
                "speaker_key": "same_speaker",
                "source_group_key": f"source_{index}",
                "emotion_id": "neutral",
                "emotion_label_source": "official_beat2_filename_protocol",
                "emotion_supervision_mask": False,
                "emotion_conditioning_mask": False,
                "motion_18d": {
                    "action_dim": 18,
                    "frames": 12,
                    "fps": 30.0,
                    "safe_csv": str(csv_path),
                    "safe_csv_sha256": csv_sha,
                    "quality_gate": {"passed": True},
                },
            }
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="speaker_id crosses fixed splits"):
        bridge.load_beat2_weak_evaluation_features(
            manifest,
            expected_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        )


def test_gate_fails_closed_when_target_transfer_is_unstable():
    def metrics(balanced, macro, recall, maximum=0.5):
        return {
            "balanced_accuracy": balanced,
            "macro_f1": macro,
            "minimum_class_recall": recall,
            "maximum_prediction_fraction": maximum,
        }

    xem_report = {"test_metrics": {"balanced_accuracy": 0.75}}
    beat2_report = {
        "validation": {
            "source_group_level": metrics(0.39, 0.34, 0.12),
            "speaker_cluster_bootstrap": {"lower_95": 0.24},
        },
        "test": {
            "source_group_level": metrics(0.28, 0.22, 0.0, maximum=0.9),
            "speaker_cluster_bootstrap": {"lower_95": 0.20},
        },
    }
    decision = bridge.gate_decision(
        xem_report,
        beat2_report,
        {"range": 0.15},
    )

    assert decision["all_checks_passed"] is False
    assert decision["eligible_as_generator_training_gate"] is False
    assert decision["eligible_as_pseudolabel_source"] is False
    assert decision["research_screening_recommendation"].startswith("not_suitable")


def test_config_rejects_forbidden_dataset_path(tmp_path: Path):
    payload = {
        "schema_version": bridge.SCHEMA_VERSION,
        "artifact_kind": bridge.CONFIG_KIND,
        "research_scope": bridge.RESEARCH_SCOPE,
        "xem_archive_path": "/tmp/xem.zip",
        "expected_xem_archive_sha256": "a" * 64,
        "xem_metadata_path": "/tmp/xem.json",
        "expected_xem_metadata_sha256": "b" * 64,
        "xem_mat_member": "Data.mat",
        "xem_mat_variable": "VelocityData",
        "beat2_manifest_path": "/tmp/beat2.jsonl",
        "expected_beat2_manifest_sha256": "c" * 64,
        "overlapping_emotions": list(bridge.EMOTIONS),
        "ridge_alphas": [1.0],
        "bootstrap_replicates": 200,
        "seed": 1,
        "output_root": (
            "/tmp/external_emotion_research/" + bridge.FORBIDDEN_DATASET_MARKER
        ),
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden dataset marker"):
        bridge.load_config(config)
