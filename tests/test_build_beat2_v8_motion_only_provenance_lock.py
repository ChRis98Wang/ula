import json
from pathlib import Path

import pytest

from tools import build_beat2_v8_motion_only_provenance_lock as build


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    artifacts = {}
    for name in ("acquisition", "inventory", "confirmation", "passed", "train_ready"):
        path = tmp_path / f"{name}.jsonl"
        path.write_text(f'{{"name":"{name}"}}\n', encoding="utf-8")
        artifacts[name] = path
    base = {
        "artifact_kind": build.LOCK_KIND,
        "formal_episode_contract": build.EPISODE_CONTRACT,
        "accepted_for_training": True,
        "formal_release_allowed": True,
        "duration_policy": build.DURATION_POLICY,
        "license_gate": {
            "training_authorized_by_this_lock": True,
            "formal_release_blocked": False,
        },
        "locked_artifacts": {
            key: {"path": str(artifacts[source]), "sha256": build.sha256_file(artifacts[source])}
            for key, source in {
                "acquisition_receipt": "acquisition",
                "training_pool_low_medium": "inventory",
                "user_confirmation_receipt": "confirmation",
            }.items()
        },
    }
    artifacts["base"] = tmp_path / "base.json"
    _write(artifacts["base"], base)
    release = {
        "artifact_kind": build.RELEASE_REPORT_KIND,
        "formal_episode_contract": build.EPISODE_CONTRACT,
        "conditioning_policy": "all_text_behavior_emotion_affect_channels_masked_zero",
        "scale": {
            "train_ready_clips": 1,
            "total_frames": 30,
            "frame_count_min": 30,
            "frame_count_max": 30,
            "distinct_frame_count_count": 1,
            "speaker_count": 1,
            "source_group_count": 1,
            "total_sample_span_sec": 29 / 30,
        },
        "invariants": {"native": True, "physical": True},
        "semantic_claims": {
            "text_conditioned_training_ready": False,
            "emotion_conditioned_training_ready": False,
        },
        "outputs": {
            "train_ready": {
                "path": str(artifacts["train_ready"]),
                "sha256": build.sha256_file(artifacts["train_ready"]),
                "records": 1,
            }
        },
    }
    artifacts["release"] = tmp_path / "release.json"
    _write(artifacts["release"], release)
    expansion = {
        "artifact_kind": build.FINAL_REPORT_KIND,
        "accepted_for_training": True,
        "bindings": {
            "expanded_passed_min30": {
                "path": str(artifacts["passed"]),
                "sha256": build.sha256_file(artifacts["passed"]),
            },
            "expanded_train_ready": {
                "path": str(artifacts["train_ready"]),
                "sha256": build.sha256_file(artifacts["train_ready"]),
            },
        },
    }
    artifacts["expansion"] = tmp_path / "expansion.json"
    _write(artifacts["expansion"], expansion)
    return artifacts


def _build(paths: dict[str, Path]) -> dict:
    return build.build_lock(
        base_lock_path=paths["base"],
        expansion_report_path=paths["expansion"],
        release_report_path=paths["release"],
        passed_manifest_path=paths["passed"],
        train_ready_manifest_path=paths["train_ready"],
    )


def test_builds_expanded_fail_closed_motion_lock(tmp_path):
    paths = _fixture(tmp_path)
    result = _build(paths)
    assert result["accepted_for_training"] is True
    assert result["dataset_scale"]["episode_count"] == 1
    assert result["policy"]["semantic_and_affect_supervision_fail_closed"] is True
    assert result["locked_artifacts"]["train_ready_manifest"]["sha256"] == build.sha256_file(
        paths["train_ready"]
    )


def test_rejects_tampered_expansion_binding(tmp_path):
    paths = _fixture(tmp_path)
    paths["passed"].write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Binding SHA mismatch"):
        _build(paths)


def test_rejects_semantic_release_claim(tmp_path):
    paths = _fixture(tmp_path)
    release = json.loads(paths["release"].read_text(encoding="utf-8"))
    release["semantic_claims"]["text_conditioned_training_ready"] = True
    _write(paths["release"], release)
    with pytest.raises(ValueError, match="fail-closed"):
        _build(paths)
