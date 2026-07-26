import hashlib
import json
from pathlib import Path

import pytest

from tools.train_ula_v2_18d_formal_from_scratch import (
    EXPRESSION_TURN_V8_PROVENANCE_LOCK_KIND,
    EXPRESSION_TURN_V8_CONDITIONING_POLICY,
    EXPRESSION_TURN_V8_OPTIMIZATION_TARGETS,
    EXPRESSION_TURN_V8_QWEN_POLICY,
    MOTION_ONLY_CONDITIONING_POLICY,
    MOTION_ONLY_OPTIMIZATION_TARGETS,
    MOTION_ONLY_PROVENANCE_LOCK_KIND,
    NONCOMMERCIAL_CONFIRMATION_TEXT,
    USER_CONFIRMATION_RECEIPT_KIND,
    _license_bound_source_provenance,
    _validate_checkpoint_license_binding,
    _validate_motion_only_conditioning,
    audit_formal_readiness,
    resolve_formal_config,
)
from upper_body_skeleton.ula_v2_18d_head import MOTION_ONLY_EPISODE_CONTRACT
from upper_body_skeleton.ula_v2_expression_turn_episode import FORMAL_EPISODE_CONTRACT


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _fixture(tmp_path: Path) -> dict:
    inventory = tmp_path / "pool.jsonl"
    inventory.write_text("{}\n", encoding="utf-8")
    receipt = tmp_path / "acquisition.json"
    _write_json(
        receipt,
        {
            "artifact_kind": "beat2_motion_only_acquisition",
            "audio_policy": "excluded_not_downloaded",
            "verification": {
                "all_selected_files_present": True,
                "all_selected_sha256_available": True,
                "all_selected_sha256_match": True,
                "all_selected_sizes_match": True,
                "forbidden_audio_selected": False,
            },
        },
    )
    lock = tmp_path / "provenance_lock.json"
    lock_payload = {
        "artifact_kind": "beat2_semantic_event_pilot_v7_provenance_lock",
        "accepted_for_training": True,
        "formal_release_allowed": True,
        "license_gate": {
            "training_authorized_by_this_lock": True,
            "formal_release_blocked": False,
        },
        "locked_artifacts": {
            "acquisition_receipt": {
                "path": str(receipt),
                "sha256": _sha256(receipt),
            },
            "training_pool_low_medium": {
                "path": str(inventory),
                "sha256": _sha256(inventory),
            },
        },
    }
    _write_json(lock, lock_payload)
    manifest = tmp_path / "train_ready.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    qwen = tmp_path / "qwen.pt"
    qwen.write_bytes(b"bound checkpoint fixture")
    return {
        "output_dir": str(tmp_path / "run"),
        "source_provenance_lock": str(lock),
        "source_provenance_lock_sha256": _sha256(lock),
        "qwen_checkpoint": str(qwen),
        "qwen_checkpoint_sha256": _sha256(qwen),
        "motion_sources": [
            {
                "dataset_source": "beat2-v7",
                "manifest": str(manifest),
                "source_inventory": str(inventory),
                "source_inventory_sha256": _sha256(inventory),
                "provenance_lock_artifact_key": "training_pool_low_medium",
                "license_gate": {
                    "policy": (
                        "noncommercial_research_user_confirmation_required_v1"
                    ),
                    "dataset_family": "BEAT2",
                    "terms_status": "conflicting_upstream_statements",
                    "metadata_statement": "apache-2.0",
                    "official_project_statement": "Non-commercial",
                    "allowed_scope": "non-commercial_research_only",
                    "user_confirmation": {
                        "confirmed": True,
                        "acknowledges_upstream_terms": True,
                        "authorized_scope": "non-commercial_research_only",
                        "confirmed_by": "fixture-user",
                        "confirmed_at": "2026-07-24T00:00:00Z",
                    },
                },
            }
        ],
    }


def test_authorized_pinned_source_is_ready_for_initialization(tmp_path):
    config = _fixture(tmp_path)
    report = audit_formal_readiness(config, stage="initialize")
    assert report["ready"] is True
    assert report["blockers"] == []
    assert report["license_audit"][0]["terms_conflict"] is True
    assert report["license_audit"][0]["user_confirmation_present"] is True


def test_metadata_license_and_old_provenance_lock_cannot_bypass_user_confirmation(
    tmp_path,
):
    config = _fixture(tmp_path)
    confirmation = config["motion_sources"][0]["license_gate"][
        "user_confirmation"
    ]
    confirmation.update(
        confirmed=False,
        acknowledges_upstream_terms=False,
        authorized_scope=None,
        confirmed_by=None,
        confirmed_at=None,
    )

    report = audit_formal_readiness(config, stage="initialize")
    assert report["ready"] is False
    assert (
        "motion_sources[0].noncommercial_research_use_not_explicitly_confirmed"
        in report["blockers"]
    )
    assert report["license_audit"][0]["metadata_statement"] == "apache-2.0"
    assert report["license_audit"][0]["official_project_statement"] == (
        "Non-commercial"
    )
    assert report["license_audit"][0]["terms_conflict"] is True


def test_user_owned_source_uses_independent_explicit_authorization(tmp_path):
    config = _fixture(tmp_path)
    source = config["motion_sources"][0]
    source["dataset_source"] = "ula0513_user_owned"
    source["license_gate"] = {
        "policy": "user_owned_explicit_authorization_v1",
        "dataset_family": "ula0513",
        "terms_status": "user_owned",
        "allowed_scope": "model_training_and_internal_evaluation",
        "user_authorization": {
            "confirmed": True,
            "ownership_asserted": True,
            "authorized_scope": "model_training_and_internal_evaluation",
            "confirmed_by": "fixture-owner",
            "confirmed_at": "2026-07-24T00:00:00Z",
        },
    }

    report = audit_formal_readiness(config, stage="initialize")
    assert report["ready"] is True
    assert report["license_audit"][0]["user_authorization_present"] is True


def test_checkpoint_binds_exact_license_confirmation(tmp_path):
    config = _fixture(tmp_path)
    audit = audit_formal_readiness(config, stage="initialize")["license_audit"]
    provenance = _license_bound_source_provenance(
        config,
        [{"dataset_source": "beat2-v7", "manifest_sha256": "a" * 64}],
        audit,
    )
    checkpoint = {"sources": {"motion_manifests": provenance}}
    _validate_checkpoint_license_binding(checkpoint, config)

    changed = json.loads(json.dumps(config))
    changed["motion_sources"][0]["license_gate"]["user_confirmation"][
        "confirmed_at"
    ] = "2026-07-25T00:00:00Z"
    with pytest.raises(ValueError, match="differs from initialization"):
        _validate_checkpoint_license_binding(checkpoint, changed)


def test_pending_license_blocks_formal_initialization(tmp_path):
    config = _fixture(tmp_path)
    lock_path = Path(config["source_provenance_lock"])
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["accepted_for_training"] = False
    lock["formal_release_allowed"] = False
    lock["license_gate"] = {
        "training_authorized_by_this_lock": False,
        "formal_release_blocked": True,
    }
    _write_json(lock_path, lock)
    config["source_provenance_lock_sha256"] = _sha256(lock_path)

    blockers = audit_formal_readiness(config, stage="initialize")["blockers"]
    assert "source_provenance_lock_training_not_accepted" in blockers
    assert "source_provenance_lock_formal_release_not_allowed" in blockers
    assert "source_provenance_lock_training_not_authorized" in blockers
    assert "source_provenance_lock_license_review_pending" in blockers


def test_tampered_receipt_blocks_formal_initialization(tmp_path):
    config = _fixture(tmp_path)
    lock = json.loads(
        Path(config["source_provenance_lock"]).read_text(encoding="utf-8")
    )
    receipt = Path(lock["locked_artifacts"]["acquisition_receipt"]["path"])
    receipt.write_text("{}", encoding="utf-8")

    blockers = audit_formal_readiness(config, stage="initialize")["blockers"]
    assert "acquisition_receipt_sha256_mismatch" in blockers


def test_motion_only_lock_binds_release_manifests_and_exact_confirmation(tmp_path):
    config = _fixture(tmp_path)
    config["formal_episode_contract"] = MOTION_ONLY_EPISODE_CONTRACT
    confirmation = tmp_path / "user_confirmation.json"
    confirmation_payload = {
        "artifact_kind": USER_CONFIRMATION_RECEIPT_KIND,
        "confirmed": True,
        "confirmation_text": NONCOMMERCIAL_CONFIRMATION_TEXT,
        "authorized_scope": "non-commercial_research_only",
        "acknowledges_upstream_terms": True,
        "confirmed_by": "fixture-user",
        "confirmed_at": "2026-07-24T00:00:00Z",
    }
    _write_json(confirmation, confirmation_payload)
    artifacts = {}
    for key in (
        "physical_qc_passed_manifest",
        "train_ready_manifest",
        "motion_only_release_report",
    ):
        path = tmp_path / f"{key}.json"
        path.write_text(f"{key}\n", encoding="utf-8")
        artifacts[key] = {"path": str(path), "sha256": _sha256(path)}

    lock_path = Path(config["source_provenance_lock"])
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock.update(
        artifact_kind=MOTION_ONLY_PROVENANCE_LOCK_KIND,
        formal_episode_contract=MOTION_ONLY_EPISODE_CONTRACT,
        duration_policy="native_variable_length_physical_qc_no_fixed_duration_v1",
        dataset_scale={"episode_count": 12345},
    )
    lock["locked_artifacts"].update(
        artifacts,
        user_confirmation_receipt={
            "path": str(confirmation),
            "sha256": _sha256(confirmation),
        },
    )
    _write_json(lock_path, lock)
    config["source_provenance_lock_sha256"] = _sha256(lock_path)

    report = audit_formal_readiness(config, stage="initialize")
    assert report["ready"] is True
    assert report["blockers"] == []

    Path(artifacts["train_ready_manifest"]["path"]).write_text(
        "tampered\n", encoding="utf-8"
    )
    blockers = audit_formal_readiness(config, stage="initialize")["blockers"]
    assert (
        "source_provenance_lock_artifact_sha256_mismatch:train_ready_manifest"
        in blockers
    )


def test_repository_formal_config_is_bound_to_v7_pool():
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs/beat2_18d_from_scratch_formal_v1.json").read_text(
            encoding="utf-8"
        )
    )
    source = config["motion_sources"][0]
    assert source["dataset_source"].endswith("training_pool_v7")
    assert "training_pool_18d_v7_full" in source["manifest"]
    assert source["provenance_lock_artifact_key"] == "training_pool_low_medium"
    assert "pilot_18d_v3" not in source["manifest"]
    assert config["conditioning_policy"] == MOTION_ONLY_CONDITIONING_POLICY
    assert config["training"]["optimization_targets"] == (
        MOTION_ONLY_OPTIMIZATION_TARGETS
    )
    license_gate = source["license_gate"]
    assert license_gate["terms_status"] == "conflicting_upstream_statements"
    assert license_gate["metadata_statement"] == "apache-2.0"
    assert license_gate["official_project_statement"] == "Non-commercial"
    assert license_gate["user_confirmation"]["confirmed"] is True
    assert license_gate["user_confirmation"]["authorized_scope"] == (
        "non-commercial_research_only"
    )
    assert license_gate["user_confirmation"]["confirmed_by"]
    assert license_gate["user_confirmation"]["confirmed_at"]


@pytest.mark.parametrize(
    "field",
    [
        "phase_frame_choices",
        "fixed_window_sec",
        "fixed_frame_count",
        "target_duration_sec",
        "max_duration_sec",
    ],
)
def test_formal_entry_rejects_fixed_or_cropped_temporal_controls(field):
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs/beat2_18d_from_scratch_formal_v1.json").read_text(
            encoding="utf-8"
        )
    )
    config["training"][field] = [64, 96] if field == "phase_frame_choices" else 6

    with pytest.raises(ValueError, match="fixed/cropped temporal controls"):
        resolve_formal_config(config)


def test_motion_only_entry_rejects_behavior_or_emotion_conditions():
    _validate_motion_only_conditioning(
        [
            {
                "clip_id": "masked",
                "behavior_supervision_mask": False,
                "emotion_conditioning_mask": False,
            }
        ]
    )
    with pytest.raises(ValueError, match="behavior conditioning masked"):
        _validate_motion_only_conditioning(
            [
                {
                    "clip_id": "behavior",
                    "behavior_supervision_mask": True,
                    "emotion_conditioning_mask": False,
                }
            ]
        )
    with pytest.raises(ValueError, match="emotion conditioning masked"):
        _validate_motion_only_conditioning(
            [
                {
                    "clip_id": "emotion",
                    "behavior_supervision_mask": False,
                    "emotion_conditioning_mask": True,
                }
            ]
        )


def test_explicit_v8_entry_enables_only_three_tier_blind_conditioning():
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs/beat2_18d_from_scratch_formal_v1.json").read_text(
            encoding="utf-8"
        )
    )
    config["formal_episode_contract"] = FORMAL_EPISODE_CONTRACT
    config["qwen_policy"] = EXPRESSION_TURN_V8_QWEN_POLICY
    config["conditioning_policy"] = EXPRESSION_TURN_V8_CONDITIONING_POLICY
    config["training"]["optimization_targets"] = EXPRESSION_TURN_V8_OPTIMIZATION_TARGETS

    resolved = resolve_formal_config(config)
    assert resolved["formal_episode_contract"] == FORMAL_EPISODE_CONTRACT
    assert resolved["qwen_policy"] == EXPRESSION_TURN_V8_QWEN_POLICY
    assert resolved["conditioning_policy"] == EXPRESSION_TURN_V8_CONDITIONING_POLICY
    assert resolved["training"]["optimization_targets"] == (
        EXPRESSION_TURN_V8_OPTIMIZATION_TARGETS
    )

    config["conditioning_policy"] = MOTION_ONLY_CONDITIONING_POLICY
    with pytest.raises(ValueError, match="scope"):
        resolve_formal_config(config)


def test_v8_multisource_lock_pins_each_inventory_without_legacy_receipt(tmp_path):
    config = _fixture(tmp_path)
    config["formal_episode_contract"] = FORMAL_EPISODE_CONTRACT
    lock_path = Path(config["source_provenance_lock"])
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock.update(
        artifact_kind=EXPRESSION_TURN_V8_PROVENANCE_LOCK_KIND,
        formal_episode_contract=FORMAL_EPISODE_CONTRACT,
        duration_policy="complete_expression_arc_variable_length_no_fixed_duration_v1",
        source_count=1,
        dataset_scale={"episode_count": 1},
        minimum_training_scale={"episode_count": 1},
        scale_gate_passed=True,
        license_gate={
            "authority_policy": "separate_per_source_license_gates_v1",
            "formal_release_blocked": False,
        },
    )
    lock["locked_artifacts"].pop("acquisition_receipt")
    _write_json(lock_path, lock)
    config["source_provenance_lock_sha256"] = _sha256(lock_path)

    report = audit_formal_readiness(config, stage="initialize")
    assert report["ready"] is True
    assert report["blockers"] == []

    inventory = Path(config["motion_sources"][0]["source_inventory"])
    inventory.write_text('{"changed":true}\n', encoding="utf-8")
    report = audit_formal_readiness(config, stage="initialize")
    assert report["ready"] is False
    assert any("artifact_sha256_mismatch" in item for item in report["blockers"])
