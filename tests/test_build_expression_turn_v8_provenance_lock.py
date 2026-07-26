from __future__ import annotations

import json

import numpy as np
import pytest

from tools import build_expression_turn_v8_provenance_lock as LOCK
from upper_body_skeleton.ula_v2_expression_turn_episode import (
    FORMAL_EPISODE_CONTRACT,
    MOTION_FORM_PROMPT_PROFILE,
)


def _spec(tmp_path):
    inventory = tmp_path / "inventory.jsonl"
    manifest = tmp_path / "train.jsonl"
    review = tmp_path / "review.json"
    evidence = tmp_path / "qc.json"
    for path, value in (
        (inventory, "{}\n"),
        (manifest, "{}\n"),
        (review, "{}\n"),
        (evidence, "{}\n"),
    ):
        path.write_text(value, encoding="utf-8")
    return {
        "schema_version": 1,
        "artifact_kind": LOCK.SPEC_ARTIFACT_KIND,
        "sources": [
            {
                "dataset_source": "user-owned",
                "source_inventory": str(inventory),
                "manifest": str(manifest),
                "review_summary": str(review),
                "inventory_artifact_key": "user_inventory",
                "manifest_artifact_key": "user_train_manifest",
                "review_artifact_key": "user_review",
            }
        ],
        "evidence_artifacts": {"network_tests": str(evidence)},
    }


def test_build_lock_validates_full_manifests_and_pins_all_files(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    episode = {
        "formal_episode_contract": FORMAL_EPISODE_CONTRACT,
        "training_qualification_tier": "semantic_conditioning",
        "prompt_semantics_profile": MOTION_FORM_PROMPT_PROFILE,
        "actions": np.zeros((37, 18), dtype=np.float32),
    }
    calls = []

    def loader(path):
        calls.append(path)
        return [episode]

    monkeypatch.setattr(LOCK, "load_expression_turn_v8_episodes", loader)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    lock = LOCK.build_lock(spec, spec_path=spec_path)

    assert calls == [(tmp_path / "train.jsonl").resolve()]
    assert lock["formal_episode_contract"] == FORMAL_EPISODE_CONTRACT
    assert lock["duration_policy"] == LOCK.DURATION_POLICY
    assert lock["source_validations"][0]["frame_count_min"] == 37
    assert lock["dataset_scale"]["episode_count"] == 1
    assert lock["dataset_scale"]["semantic_conditioned_count"] == 1
    assert lock["dataset_scale"]["distinct_frame_count_count"] == 1
    assert lock["dataset_scale"]["total_sample_span_sec"] == pytest.approx(36 / 30)
    assert lock["scale_gate_passed"] is True
    assert set(lock["locked_artifacts"]) == {
        "network_tests",
        "user_inventory",
        "user_review",
        "user_train_manifest",
    }
    assert lock["license_gate"]["authority_policy"].startswith("separate_per_source")


def test_build_lock_rejects_duplicate_keys_or_missing_evidence(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    spec["sources"][0]["review_artifact_key"] = "user_inventory"
    monkeypatch.setattr(
        LOCK,
        "load_expression_turn_v8_episodes",
        lambda _path: [
            {
                "training_qualification_tier": "base_motion",
                "prompt_semantics_profile": MOTION_FORM_PROMPT_PROFILE,
                "actions": np.zeros((3, 18), dtype=np.float32),
            }
        ],
    )
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate provenance artifact key"):
        LOCK.build_lock(spec, spec_path=spec_path)

    spec = _spec(tmp_path)
    spec["evidence_artifacts"] = {}
    with pytest.raises(ValueError, match="requires independent evidence"):
        LOCK.build_lock(spec, spec_path=spec_path)


def test_build_lock_enforces_explicit_minimum_training_scale(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    spec["minimum_training_scale"] = {
        "episode_count": 2,
        "semantic_conditioned_count": 2,
        "expressive_conditioned_count": 1,
        "distinct_frame_count_count": 2,
        "duration_span_sec": 1.0,
    }
    monkeypatch.setattr(
        LOCK,
        "load_expression_turn_v8_episodes",
        lambda _path: [
            {
                "training_qualification_tier": "semantic_conditioning",
                "prompt_semantics_profile": MOTION_FORM_PROMPT_PROFILE,
                "prompt_sha256": "a" * 64,
                "actions": np.zeros((37, 18), dtype=np.float32),
                "fps": 30.0,
            }
        ],
    )
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="dataset scale gate failed") as error:
        LOCK.build_lock(spec, spec_path=spec_path)
    assert "episode_count:1<2" in str(error.value)
    assert "expressive_conditioned_count:0<1" in str(error.value)
