import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.human_motion_review.adjudicate_training_dataset import (
    ADJUDICATION_SCHEMA_VERSION,
    REQUIRED_18D_GATES,
    REQUIRED_REVIEW_GATES,
)
from upper_body_skeleton.retarget_v2 import JOINT_ORDER
from upper_body_skeleton.retarget_v2_18d import CONTRACT_VERSION, JOINT_ORDER_18D
from upper_body_skeleton.lerobot_export import read_joint_window
from upper_body_skeleton.pt_mujoco_infer import validate_generator_checkpoint
from upper_body_skeleton.kimodo_semantics import (
    KIMODO_BEHAVIOR_FAMILIES,
    KIMODO_BEHAVIOR_IDS,
    KIMODO_EMOTION_IDS,
)
from upper_body_skeleton.ula_training import (
    AFFECT_IDS,
    GESTURE_IDS,
    INTENT_IDS,
    KIMODO_CONDITION_DIM,
    KIMODO_V2_CONDITION_DIM,
    LEGACY_CONDITION_DIM,
    STYLE_IDS,
    ULA_MMDIT_V2_ARCHITECTURE,
    create_ula_model,
)
from upper_body_skeleton.ula_v2_conditioning import (
    extract_style_features,
    normalize_style_features,
)
from upper_body_skeleton.ula_v2_18d_head import (
    ACTION_DIM,
    ARTIFACT_KIND,
    CONDITION_CACHE_SCHEMA_VERSION,
    FORMAL_ADJUDICATION_SCHEMA_VERSION,
    FORMAL_REQUIRED_18D_GATES,
    FORMAL_REQUIRED_RELEASE_INVARIANTS,
    FORMAL_REQUIRED_REVIEW_GATES,
    FORMAL_SEMANTIC_SUPERVISION_MASKS,
    KIMODO_BEHAVIOR_FAMILY_SLICE,
    KIMODO_BEHAVIOR_SLICE,
    KIMODO_EMOTION_SLICE,
    LEGACY_AFFECT_SLICE,
    LEGACY_ACTION_DIM,
    LEGACY_GESTURE_SLICE,
    LEGACY_INTENT_SLICE,
    STYLE_CONTROL_SLICE,
    attach_condition_cache,
    body_sampling_drift_metrics,
    benchmark_contract_inference,
    build_condition_cache,
    compute_18d_action_stats,
    configure_head_adapter_policy,
    frozen_weight_max_error,
    instantiate_checkpoint_model,
    legacy_forward_max_error,
    load_18d_episodes,
    load_condition_cache,
    load_contract_checkpoint,
    migrate_15d_checkpoint,
    nonzero_head_forward_drift_metrics,
    predict_contract_frame_count,
    restore_frozen_weights,
    semantic_supervision_policy,
    train_head_adapter,
    validate_condition_cache_for_generator,
    validate_qwen_checkpoint_for_generator,
    validate_checkpoint_contract,
    verify_migrated_prefix,
    write_contract_csv,
    write_contract_npz,
)


def make_style_contract():
    payload = {
        "contract_type": "ula_v2_style_normalization",
        "contract_version": 1,
        "feature_names": [
            "signed_arm_balance",
            "log_arm_amplitude",
            "log_arm_speed",
        ],
        "feature_definition": {},
        "mean": [0.1, 0.2, 0.3],
        "std": [0.5, 0.25, 0.75],
        "eps": 1e-4,
        "clip": 5.0,
        "fit_split": "train",
        "fit_episode_count": 1,
        "fit_episode_indices": [0],
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return {**payload, "sha256": digest}


def make_15d_checkpoint(path):
    torch.manual_seed(4)
    model = create_ula_model(
        ULA_MMDIT_V2_ARCHITECTURE,
        action_dim=LEGACY_ACTION_DIM,
        condition_dim=KIMODO_V2_CONDITION_DIM,
        hidden_dim=32,
        layers=1,
        semantic_tokens=6,
    )
    style_contract = make_style_contract()
    payload = {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "architecture": ULA_MMDIT_V2_ARCHITECTURE,
        "condition_dim": KIMODO_V2_CONDITION_DIM,
        "action_dim": LEGACY_ACTION_DIM,
        "joint_order": list(JOINT_ORDER),
        "model_state_dict": model.state_dict(),
        "action_stats": {
            "mean": torch.linspace(-0.2, 0.2, LEGACY_ACTION_DIM),
            "std": torch.linspace(0.1, 0.3, LEGACY_ACTION_DIM),
        },
        "config": {"hidden_dim": 32, "layers": 1, "semantic_tokens": 6},
        "global_step": 12,
        "v2_contracts": {
            "style": style_contract,
            "condition": {
                "style_contract_sha256": style_contract["sha256"],
                "style_control_indices": list(
                    range(KIMODO_CONDITION_DIM - 3, KIMODO_CONDITION_DIM)
                ),
            },
        },
    }
    torch.save(payload, path)
    return payload


def make_trajectory(offset=0.0, frames=12):
    phase = np.linspace(0.0, 1.0, frames, dtype=np.float32)[:, None]
    scales = np.linspace(0.01, 0.18, ACTION_DIM, dtype=np.float32)[None, :]
    return offset + np.sin(phase * np.pi * 2.0) * scales


def write_18d_csv(path, values):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(JOINT_ORDER_18D)
        writer.writerows(values.tolist())


def legacy_semantics(**overrides):
    return {
        "intent": "explaining",
        "observed_affect": "low_confidence_unknown",
        "motion_style": "restrained",
        "semantic_gesture": "upper_body_gesture",
        **overrides,
    }


def official_semantic_episode(**overrides):
    category = overrides.pop("category", "deictic")
    enabled = overrides.pop("emotion_conditioning_enabled", False)
    record = {
        "clip_id": "official_event",
        "annotation_kind": "official_gesture_semantic_event",
        "semantic_event": {"category": category, "intensity": "high"},
        "canonical_action": f"official_gesture_category:{category}",
        "canonical_action_role": "official_category_metadata_split_key_only",
        "semantic_mapping_status": "official_category_verified_metadata_only",
        "official_category_verified": True,
        "official_category_conditioning_enabled": False,
        "official_category_role": "verified_metadata_split_and_evaluation_only",
        "official_category_condition_channel": None,
        "official_category_loss": None,
        "robot_observable_motion_form": "candidate_unreviewed",
        "communicative_intent": "candidate_unreviewed",
        "canonical_prompt": {
            "en": "Perform a high-intensity deictic gesture with an angry affect."
        },
        "canonical_prompt_role": "coarse_category_only",
        "semantic_supervision_masks": dict(FORMAL_SEMANTIC_SUPERVISION_MASKS),
        "behavior_id": "Behavior.InteractPresence",
        "behavior_review_status": "candidate_unreviewed",
        "behavior_supervision_mask": False,
        "emotion_id": "angry",
        "emotion_review_status": "official_protocol_confirmed",
        "source_emotion_label_verified": True,
        "emotion_supervision_mask": enabled,
        "official_emotion_conditioning_enabled": enabled,
        "emotion_supervision_role": (
            "enabled_verified_robot_affect_observable_in_18d"
            if enabled
            else "disabled_pending_robot_affect_review"
        ),
        "official_emotion_condition_channel": (
            "kimodo_emotion_one_hot_and_legacy_affect" if enabled else None
        ),
        "official_emotion_loss": None,
        "affect_observable_review_status": (
            "verified" if enabled else "candidate_unreviewed"
        ),
        "affect_observable_supervision_mask": enabled,
        "emotion_conditioning_mask": enabled,
        "intent": "explaining",
        "observed_affect": "low_confidence_unknown",
        "motion_style": "sharp",
        "semantic_gesture": "upper_body_gesture",
        "actions": make_trajectory(),
        "fps": 30.0,
    }
    record.update(overrides)
    return record


def write_formal_train_ready_release(
    root,
    *,
    record_forgery=None,
    report_forgery=None,
):
    root.mkdir(parents=True, exist_ok=True)
    clip_id = "reviewed_wave"
    values = make_trajectory()
    csv_path = root / f"{clip_id}_gmr_safe_18d.csv"
    write_18d_csv(csv_path, values)
    quality_gates = {gate: True for gate in FORMAL_REQUIRED_18D_GATES}
    quality_path = root / "quality.json"
    quality_payload = {
        "output_contract": CONTRACT_VERSION,
        "action_dim": ACTION_DIM,
        "joint_order": list(JOINT_ORDER_18D),
        "frames": len(values),
        "fps": 30.0,
        "quality_gate": quality_gates,
        "outputs": {"safe_csv": str(csv_path)},
    }
    quality_path.write_text(json.dumps(quality_payload, sort_keys=True), encoding="utf-8")
    record = {
        "adjudication_schema_version": FORMAL_ADJUDICATION_SCHEMA_VERSION,
        "clip_id": clip_id,
        "canonical_action": "one_hand_wave",
        "canonical_prompt": {"en": "Wave one hand in greeting."},
        "official_split": "train",
        "emotion_label_source": "fixture_unresolved",
        "semantic_label_status": "fixture_human_reviewed",
        "official_gesture_semantic_spans": [{"category": "iconic"}],
        "audio_policy": "disabled_not_read_not_required",
        "annotation_relpath": "annotations/reviewed_wave.txt",
        "textgrid_relpath": "textgrid/reviewed_wave.TextGrid",
        "motion_relpath": "motion/reviewed_wave.npz",
        "window_transcript_context": "hello",
        "window_transcript_role": "context_only",
        "behavior_id": "Behavior.GreetingOwner01",
        "behavior_review_status": "human_confirmed",
        "behavior_supervision_mask": True,
        "human_review": {
            "reviewer_id": "fixture-human",
            "reviewer_kind": "human",
            "reviewed_at": "2026-07-23T10:00:00+08:00",
            "decision": "behavior_confirmed",
        },
        "emotion_id": None,
        "emotion_review_status": "unresolved",
        "emotion_supervision_mask": False,
        "source_emotion_label_verified": False,
        "affect_observable_review_status": "not_verified",
        "affect_observable_supervision_mask": False,
        "emotion_conditioning_mask": False,
        "network_semantic_supervision_ready": False,
        **legacy_semantics(),
        "trajectory_path": str(csv_path),
        "adjudication": {
            "status": "train_ready",
            "reasons": [],
            "rejection_causes": [],
            "semantic_critical": False,
            "semantic_pending_reasons": [],
        },
        "independent_review": {
            "review_id": "independent_fixture_review",
            "present": True,
            "status": "agent_reviewed",
            "training_acceptance": True,
            "gates": {
                gate: gate != "affect_observable_in_18d"
                for gate in FORMAL_REQUIRED_REVIEW_GATES
            },
            "conflict_ids": [],
        },
        "motion_18d": {
            "state": "passed",
            "partition": "accepted",
            "quality_json": str(quality_path),
            "safe_csv": str(csv_path),
            "quality_sha256": hashlib.sha256(quality_path.read_bytes()).hexdigest(),
            "safe_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "reasons": [],
            "output_contract": CONTRACT_VERSION,
            "action_dim": ACTION_DIM,
            "frames": len(values),
            "fps": 30.0,
            "quality_gate": quality_gates,
            "csv_rows": len(values),
        },
        "training_eligibility": {
            "behavior": {
                "eligible": True,
                "status": "train_ready",
                "one_hot_supervision_mask": True,
                "requires": ["human_confirmed_behavior", "motion_style_train_ready"],
            },
            "motion_style": {
                "eligible": True,
                "status": "train_ready",
                "requires": ["passed_18d_qc", "independent_training_acceptance"],
            },
            "emotion": {
                "eligible": False,
                "status": "blocked_unresolved_emotion",
                "loss_mask": False,
                "source_label_verified": False,
                "affect_observable_verified": False,
                "conditioning_mask": False,
                "requires": [
                    "motion_style_train_ready",
                    "human_confirmed_emotion",
                    "verified_blind_robot_affect_review",
                ],
            },
        },
        "split": {
            "subject_key": "speaker_fixture",
            "subject_policy": "subject_disjoint",
            "action_key": "one_hand_wave",
            "source_group_key": "fixture/source_clip",
            "assignment": "train",
            "eval_eligible": False,
        },
    }
    if record_forgery is not None:
        record_forgery(record)
    manifest = root / "train_ready.jsonl"
    manifest.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "schema_version": 1,
        "adjudication_schema_version": FORMAL_ADJUDICATION_SCHEMA_VERSION,
        "counts": {"train_ready": 1},
        "scale": {"train_ready_clips": 1},
        "outputs": {
            "train_ready": {
                "path": str(manifest),
                "records": 1,
                "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            }
        },
        "invariants": {name: True for name in FORMAL_REQUIRED_RELEASE_INVARIANTS},
        "output_dir": str(root),
    }
    if report_forgery is not None:
        report_forgery(report)
    report_path = root / "dataset_scale_report.json"
    report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    return manifest, report_path


def forge_formal_record(record, case):
    if case == "missing_schema":
        record.pop("adjudication_schema_version")
    elif case == "partial_adjudication":
        record["adjudication"].pop("semantic_critical")
    elif case == "partial_review":
        record["independent_review"]["gates"].pop("observable_in_18d")
    elif case == "false_eligibility":
        record["training_eligibility"]["motion_style"]["eligible"] = False
    elif case == "quality_hash":
        record["motion_18d"]["quality_sha256"] = "0" * 64
    elif case == "trajectory_hash":
        record["motion_18d"]["safe_csv_sha256"] = "0" * 64
    else:
        raise AssertionError(case)


def with_posttrain_contract_hash(payload):
    result = dict(payload)
    result["sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return result


def make_posttrain_cache_lineage(root):
    root.mkdir(parents=True, exist_ok=True)
    base_path = root / "base_15d.pt"
    base = make_15d_checkpoint(base_path)
    source_path = root / "source_18d.pt"
    stats = compute_18d_action_stats([make_trajectory()], base["action_stats"])
    source, _ = migrate_15d_checkpoint(base_path, source_path, action_stats=stats)
    qwen_source = {
        "checkpoint_sha256": "1" * 64,
        "model_name": "Qwen/test",
        "revision": "pinned-revision",
    }
    source["v2_contracts"]["text_motion_latent"] = {"source": qwen_source}
    torch.save(source, source_path)
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    cache_provenance = {
        "schema_version": CONDITION_CACHE_SCHEMA_VERSION,
        "artifact_kind": "ula_v2_qwen_motion_condition_cache",
        "unsafe_condition_cache": False,
        "cache_sha256": "2" * 64,
        "qwen_checkpoint_sha256": qwen_source["checkpoint_sha256"],
        "qwen_model_name": qwen_source["model_name"],
        "qwen_revision": qwen_source["revision"],
        "generator_checkpoint": str(source_path),
        "generator_checkpoint_sha256": source_sha256,
        "style_contract_sha256": source["v2_contracts"]["style"]["sha256"],
    }
    split_contract = with_posttrain_contract_hash(
        {
            "contract_type": "speaker_source_group_strict_split",
            "contract_version": 1,
            "seed": 7,
        }
    )
    data_contract = with_posttrain_contract_hash(
        {
            "contract_type": "ula_v2_18d_interaction_posttrain_data",
            "contract_version": 1,
            "split_contract_sha256": split_contract["sha256"],
            "episode_count": 0,
            "records": [],
        }
    )
    derived = torch.load(source_path, map_location="cpu", weights_only=True)
    posttrain_step = 5
    derived.update(
        {
            "global_step": int(source["global_step"]) + posttrain_step,
            "posttrain_step": posttrain_step,
            "posttrain_artifact_kind": "ula_mmdit_v2_18d_interaction_posttrain",
            "posttrain_source": {
                "checkpoint": str(source_path),
                "checkpoint_sha256": source_sha256,
                "source_global_step": int(source["global_step"]),
            },
            "posttrain_data_contract": data_contract,
            "posttrain_split_contract": split_contract,
            "posttrain_config": {"lr": 1e-5, "steps": posttrain_step},
            "training_contract": {
                "mode": "low_lr_full_network_interaction_domain_posttrain",
                "all_model_parameters_trainable": True,
            },
            "data_provenance": {
                "data_contract_sha256": data_contract["sha256"],
                "condition_cache": dict(cache_provenance),
            },
        }
    )
    derived_path = root / "posttrained_18d.pt"
    torch.save(derived, derived_path)
    return derived_path, derived, source_path, cache_provenance


def test_formal_loader_contract_matches_adjudicator_contract():
    assert FORMAL_ADJUDICATION_SCHEMA_VERSION == ADJUDICATION_SCHEMA_VERSION
    assert FORMAL_REQUIRED_18D_GATES == frozenset(REQUIRED_18D_GATES)
    assert FORMAL_REQUIRED_REVIEW_GATES == frozenset(REQUIRED_REVIEW_GATES)


def test_migration_preserves_every_15d_weight_and_zero_padded_forward(tmp_path):
    base_path = tmp_path / "base.pt"
    base = make_15d_checkpoint(base_path)
    stats = compute_18d_action_stats([make_trajectory()], base["action_stats"])

    migrated, verification = migrate_15d_checkpoint(
        base_path, tmp_path / "migrated.pt", action_stats=stats
    )

    assert migrated["action_dim"] == ACTION_DIM
    assert migrated["joint_order"] == JOINT_ORDER_18D
    assert migrated["action_contract"]["version"] == CONTRACT_VERSION
    assert verification == {
        "weight_prefix_max_abs_error": 0.0,
        "legacy_forward_max_abs_error": 0.0,
        "legacy_forward_atol": 1e-5,
    }
    assert verify_migrated_prefix(base, migrated) == 0.0
    assert torch.count_nonzero(migrated["model_state_dict"]["input.weight"][:, 15:]) == 0
    assert torch.count_nonzero(migrated["model_state_dict"]["output.weight"][15:]) == 0


def test_gradient_policy_updates_only_new_projection_slices(tmp_path):
    base_path = tmp_path / "base.pt"
    base = make_15d_checkpoint(base_path)
    stats = compute_18d_action_stats([make_trajectory()], base["action_stats"])
    migrated, _ = migrate_15d_checkpoint(base_path, tmp_path / "migrated.pt", action_stats=stats)
    model = instantiate_checkpoint_model(migrated)
    policy = configure_head_adapter_policy(model)
    with torch.no_grad():
        model.output.weight[15:].normal_(std=0.01)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.1
    )
    before_input_head = model.input.weight[:, 15:].detach().clone()
    output = model(
        torch.randn(2, 8, ACTION_DIM),
        torch.tensor([0.2, 0.7]),
        torch.randn(2, KIMODO_V2_CONDITION_DIM),
    )
    (output[..., 15:] - 1.0).square().mean().backward()
    optimizer.step()
    restore_frozen_weights(model, policy)

    assert frozen_weight_max_error(model, policy) == 0.0
    assert not torch.equal(model.input.weight[:, 15:], before_input_head)
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if name not in {"input.weight", "output.weight", "output.bias"}
    )


def test_loader_requires_acceptance_and_strict_joint_order(tmp_path):
    motion_root = tmp_path / "motion"
    clip_dir = motion_root / "wave_clip1"
    clip_dir.mkdir(parents=True)
    path = clip_dir / "wave_clip1_gmr_safe_18d.csv"
    write_18d_csv(path, make_trajectory())
    semantics = tmp_path / "semantics.jsonl"
    record = {
        "clip_id": "wave_clip1",
        "canonical_prompt": {"en": "Wave one hand."},
        "behavior_id": "Behavior.GreetingOwner01",
        "emotion_id": "neutral",
        **legacy_semantics(motion_style="slow_safe"),
        "review_status": {"state": "agent_reviewed", "accepted_for_training": False},
    }
    semantics.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no eligible"):
        load_18d_episodes(motion_root=motion_root, semantics=semantics)
    episodes = load_18d_episodes(
        motion_root=motion_root, semantics=semantics, allow_unreviewed=True
    )
    assert episodes[0]["actions"].shape == (12, ACTION_DIM)
    assert episodes[0]["intent"] == "explaining"
    assert episodes[0]["observed_affect"] == "low_confidence_unknown"
    assert episodes[0]["semantic_gesture"] == "upper_body_gesture"
    assert episodes[0]["source_motion_style"] == "slow_safe"
    assert episodes[0]["motion_style"] == "restrained"

    path.write_text(",".join(reversed(JOINT_ORDER_18D)) + "\n0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly match"):
        load_18d_episodes(
            motion_root=motion_root, semantics=semantics, allow_unreviewed=True
        )


def test_merged_manifest_cannot_bypass_18d_qc_with_legacy_acceptance_bit(tmp_path):
    csv_path = tmp_path / "legacy_accepted_gmr_safe_18d.csv"
    write_18d_csv(csv_path, make_trajectory())
    manifest = tmp_path / "merged.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "clip_id": "legacy_accepted",
                "canonical_prompt": {"en": "Wave one hand."},
                "behavior_id": "Behavior.GreetingOwner01",
                "emotion_id": "neutral",
                **legacy_semantics(),
                "trajectory_path": str(csv_path),
                "review_status": {"accepted_for_training": True, "state": "accepted"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no eligible"):
        load_18d_episodes(manifest=manifest)
    unsafe = load_18d_episodes(manifest=manifest, allow_unreviewed=True)
    assert unsafe[0]["eligibility_mode"] == "unsafe_allow_unreviewed"
    assert unsafe[0]["accepted_for_training"] is False


def test_strict_loader_accepts_only_complete_report_bound_train_ready_release(tmp_path):
    manifest, _ = write_formal_train_ready_release(tmp_path / "release")

    episode = load_18d_episodes(manifest=manifest)[0]

    assert episode["accepted_for_training"] is True
    assert episode["eligibility_mode"] == "adjudicated_train_ready"
    assert episode["formal_source_metadata"]["official_split"] == "train"
    assert episode["formal_source_metadata"]["annotation_relpath"] == (
        "annotations/reviewed_wave.txt"
    )
    assert len(episode["source_record_sha256"]) == 64
    assert episode["review_state"] == "train_ready"
    assert episode["actions"].shape == (12, ACTION_DIM)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing_schema", "adjudication_schema_version"),
        ("partial_adjudication", "adjudication.semantic_critical"),
        ("partial_review", "independent_review.gates"),
        ("false_eligibility", "training_eligibility.motion_style.eligible"),
        ("quality_hash", "quality JSON hash mismatch"),
        ("trajectory_hash", "18D trajectory hash mismatch"),
    ],
)
def test_forged_partial_train_ready_record_fails_closed_but_unsafe_path_loads(
    tmp_path, case, message
):
    manifest, _ = write_formal_train_ready_release(
        tmp_path / case,
        record_forgery=lambda record: forge_formal_record(record, case),
    )

    with pytest.raises(ValueError, match=message):
        load_18d_episodes(manifest=manifest)

    unsafe = load_18d_episodes(manifest=manifest, allow_unreviewed=True)[0]
    assert unsafe["accepted_for_training"] is False
    assert unsafe["eligibility_mode"] == "unsafe_allow_unreviewed"


@pytest.mark.parametrize(
    ("report_forgery", "message"),
    [
        (lambda report: report.update(schema_version=0), "schema_version"),
        (
            lambda report: report["outputs"]["train_ready"].update(sha256="0" * 64),
            "outputs.train_ready.sha256",
        ),
        (
            lambda report: report["invariants"].update(
                every_semantic_record_adjudicated_once=False
            ),
            "required release invariant",
        ),
    ],
)
def test_forged_release_report_fails_closed_but_unsafe_path_loads(
    tmp_path, report_forgery, message
):
    manifest, _ = write_formal_train_ready_release(
        tmp_path / message.replace(".", "_"),
        report_forgery=report_forgery,
    )

    with pytest.raises(ValueError, match=message):
        load_18d_episodes(manifest=manifest)

    unsafe = load_18d_episodes(manifest=manifest, allow_unreviewed=True)[0]
    assert unsafe["accepted_for_training"] is False
    assert unsafe["eligibility_mode"] == "unsafe_allow_unreviewed"


def test_missing_release_report_fails_closed_but_unsafe_path_loads(tmp_path):
    manifest, report = write_formal_train_ready_release(tmp_path / "missing_report")
    report.unlink()

    with pytest.raises(ValueError, match="requires sibling dataset_scale_report.json"):
        load_18d_episodes(manifest=manifest)

    unsafe = load_18d_episodes(manifest=manifest, allow_unreviewed=True)[0]
    assert unsafe["accepted_for_training"] is False
    assert unsafe["eligibility_mode"] == "unsafe_allow_unreviewed"


def test_structured_condition_cache_overrides_text_and_preserves_unresolved_mask(
    tmp_path, monkeypatch
):
    class FakeEncoder:
        def encode(self, prompts, batch_size):
            assert batch_size == 2
            return np.zeros(
                (len(prompts), KIMODO_V2_CONDITION_DIM - KIMODO_CONDITION_DIM),
                dtype=np.float32,
            )

    qwen = tmp_path / "qwen.pt"
    qwen_record = {"model_name": "Qwen/test", "revision": "pinned-revision"}
    torch.save({"qwen": qwen_record}, qwen)
    base_checkpoint = tmp_path / "base.pt"
    base_payload = make_15d_checkpoint(base_checkpoint)
    base_payload["v2_contracts"]["text_motion_latent"] = {
        "source": {
            "checkpoint_sha256": hashlib.sha256(qwen.read_bytes()).hexdigest(),
            "model_name": qwen_record["model_name"],
            "revision": qwen_record["revision"],
        }
    }
    torch.save(base_payload, base_checkpoint)
    monkeypatch.setattr(
        "upper_body_skeleton.cross_modal_latent.load_qwen_motion_text_encoder",
        lambda *_args, **_kwargs: (FakeEncoder(), {"qwen": qwen_record}),
    )
    episodes = [
        {
            "clip_id": "misleading_text",
            "prompt": "Move both arms across the body happily, with a joyful greeting.",
            # Legacy Kimodo records keep their explicit IDs under labels.
            "labels": {
                "behavior_id": "Behavior.Alert",
                "emotion_id": "sad",
                "motion_style": "sharp",
            },
            "actions": make_trajectory(),
            "fps": 30.0,
        },
        {
            "clip_id": "unresolved_emotion",
            "prompt": "Wave happily and look surprised.",
            "behavior_id": "Behavior.Hesitate",
            "behavior_review_status": "candidate_unreviewed",
            "behavior_supervision_mask": False,
            "emotion_review_status": "unresolved",
            "emotion_supervision_mask": False,
            "actions": make_trajectory(0.02),
            "fps": 30.0,
        },
    ]
    cache = tmp_path / "conditions.npz"
    metadata = build_condition_cache(
        episodes,
        qwen,
        cache,
        base_checkpoint=base_checkpoint,
        device="cpu",
        batch_size=2,
    )

    _, _, conditions, loaded_metadata = load_condition_cache(cache)
    behavior_start = LEGACY_CONDITION_DIM
    emotion_start = behavior_start + len(KIMODO_BEHAVIOR_IDS)
    assert conditions[0, behavior_start + KIMODO_BEHAVIOR_IDS.index("Behavior.Alert")] == 1
    assert conditions[0, emotion_start + KIMODO_EMOTION_IDS.index("sad")] == 1
    assert conditions[0, emotion_start + KIMODO_EMOTION_IDS.index("happy")] == 0
    assert np.count_nonzero(
        conditions[1, behavior_start : behavior_start + len(KIMODO_BEHAVIOR_IDS)]
    ) == 0
    behavior_family_start = emotion_start + len(KIMODO_EMOTION_IDS)
    assert np.count_nonzero(
        conditions[
            1,
            behavior_family_start : behavior_family_start + len(KIMODO_BEHAVIOR_FAMILIES),
        ]
    ) == 0
    assert np.count_nonzero(
        conditions[1, emotion_start : emotion_start + len(KIMODO_EMOTION_IDS)]
    ) == 0
    gesture_start = len(INTENT_IDS) + len(AFFECT_IDS) + len(STYLE_IDS)
    style_start = len(INTENT_IDS) + len(AFFECT_IDS)
    assert conditions[0, INTENT_IDS["explaining"]] == 1
    assert conditions[0, len(INTENT_IDS) + AFFECT_IDS["low_confidence_unknown"]] == 1
    assert conditions[0, style_start + STYLE_IDS["energetic"]] == 1
    assert conditions[0, gesture_start + GESTURE_IDS["upper_body_gesture"]] == 1
    assert conditions[0, gesture_start + GESTURE_IDS["crossed_arms"]] == 0
    assert conditions[1, INTENT_IDS["explaining"]] == 1
    assert conditions[1, len(INTENT_IDS) + AFFECT_IDS["low_confidence_unknown"]] == 0
    assert conditions[1, gesture_start + GESTURE_IDS["upper_body_gesture"]] == 1
    expected_features = extract_style_features(episodes[0]["actions"][:, :15], fps=30.0)
    expected_controls = normalize_style_features(
        expected_features, base_payload["v2_contracts"]["style"]
    )
    assert np.array_equal(
        conditions[0, KIMODO_CONDITION_DIM - 3 : KIMODO_CONDITION_DIM],
        expected_controls,
    )

    with np.load(cache, allow_pickle=False) as payload:
        assert payload["behavior_ids"].astype(str).tolist() == [
            "Behavior.Alert",
            "Behavior.Hesitate",
        ]
        assert payload["behavior_review_statuses"].astype(str).tolist() == [
            "legacy_resolved",
            "candidate_unreviewed",
        ]
        assert payload["behavior_supervision_mask"].tolist() == [True, False]
        assert payload["emotion_ids"].astype(str).tolist() == ["sad", ""]
        assert payload["emotion_supervision_mask"].tolist() == [True, False]
        assert np.array_equal(payload["style_features"][0], expected_features)
        assert np.array_equal(payload["style_controls"][0], expected_controls)
    assert metadata["schema_version"] == 3
    assert metadata["emotion_supervised_count"] == 1
    assert metadata["emotion_unresolved_count"] == 1
    assert metadata["behavior_supervised_count"] == 1
    assert metadata["behavior_unsupervised_count"] == 1
    assert metadata["generator_checkpoint_sha256"] == hashlib.sha256(
        base_checkpoint.read_bytes()
    ).hexdigest()
    assert (
        metadata["style_contract_sha256"]
        == base_payload["v2_contracts"]["style"]["sha256"]
    )
    assert metadata["episodes"][1]["emotion_id"] is None
    assert metadata["episodes"][1]["emotion_supervision_mask"] is False
    assert loaded_metadata["episodes"] == metadata["episodes"]

    attached = attach_condition_cache(episodes, cache)
    assert attached[0]["behavior_id"] == "Behavior.Alert"
    assert attached[0]["emotion_id"] == "sad"
    assert attached[1]["emotion_id"] is None
    assert attached[1]["behavior_review_status"] == "candidate_unreviewed"
    assert attached[1]["behavior_supervision_mask"] is False
    assert attached[1]["emotion_review_status"] == "unresolved"
    assert attached[1]["emotion_supervision_mask"] is False
    assert validate_condition_cache_for_generator(
        base_payload,
        loaded_metadata,
        generator_checkpoint_path=base_checkpoint,
    )["validated"] is True

    wrong_checkpoint = tmp_path / "wrong_base.pt"
    wrong_payload = torch.load(base_checkpoint, map_location="cpu", weights_only=True)
    wrong_payload["model_state_dict"]["input.weight"] = (
        wrong_payload["model_state_dict"]["input.weight"] + 0.01
    )
    torch.save(wrong_payload, wrong_checkpoint)
    with pytest.raises(ValueError, match="different generator checkpoint"):
        validate_condition_cache_for_generator(
            wrong_payload,
            loaded_metadata,
            generator_checkpoint_path=wrong_checkpoint,
        )


def test_official_v7_cache_masks_category_prompt_and_unverified_affect(
    tmp_path, monkeypatch
):
    class NoPromptEncoder:
        def encode(self, _prompts, _batch_size):
            raise AssertionError("masked official prompts must not reach Qwen")

    qwen = tmp_path / "qwen.pt"
    qwen_record = {"model_name": "Qwen/test", "revision": "pinned-revision"}
    torch.save({"qwen": qwen_record}, qwen)
    base_checkpoint = tmp_path / "base.pt"
    base_payload = make_15d_checkpoint(base_checkpoint)
    base_payload["v2_contracts"]["text_motion_latent"] = {
        "source": {
            "checkpoint_sha256": hashlib.sha256(qwen.read_bytes()).hexdigest(),
            "model_name": qwen_record["model_name"],
            "revision": qwen_record["revision"],
        }
    }
    torch.save(base_payload, base_checkpoint)
    monkeypatch.setattr(
        "upper_body_skeleton.cross_modal_latent.load_qwen_motion_text_encoder",
        lambda *_args, **_kwargs: (NoPromptEncoder(), {"qwen": qwen_record}),
    )
    episodes = [
        official_semantic_episode(),
        official_semantic_episode(
            clip_id="official_event_affect_verified",
            emotion_conditioning_enabled=True,
        ),
    ]
    cache = tmp_path / "official_v7_conditions.npz"
    metadata = build_condition_cache(
        episodes,
        qwen,
        cache,
        base_checkpoint=base_checkpoint,
        device="cpu",
    )

    _, _, conditions, loaded = load_condition_cache(cache)
    assert np.count_nonzero(conditions[:, KIMODO_CONDITION_DIM:]) == 0
    assert np.count_nonzero(conditions[:, LEGACY_INTENT_SLICE]) == 0
    assert np.count_nonzero(conditions[:, LEGACY_GESTURE_SLICE]) == 0
    assert np.count_nonzero(conditions[:, KIMODO_BEHAVIOR_SLICE]) == 0
    assert np.count_nonzero(conditions[:, KIMODO_BEHAVIOR_FAMILY_SLICE]) == 0
    assert np.count_nonzero(conditions[0, KIMODO_EMOTION_SLICE]) == 0
    assert np.count_nonzero(conditions[0, LEGACY_AFFECT_SLICE]) == 0
    assert conditions[
        1, KIMODO_EMOTION_SLICE.start + KIMODO_EMOTION_IDS.index("angry")
    ] == 1.0
    assert np.count_nonzero(conditions[1, KIMODO_EMOTION_SLICE]) == 1
    assert conditions[
        1, LEGACY_AFFECT_SLICE.start + AFFECT_IDS["angry_like"]
    ] == 1.0
    assert np.count_nonzero(conditions[1, LEGACY_AFFECT_SLICE]) == 1
    assert np.array_equal(
        conditions[:, STYLE_CONTROL_SLICE],
        np.asarray([item["style_controls"] for item in metadata["episodes"]]),
    )
    assert metadata["official_category_conditioned_count"] == 0
    assert metadata["affect_observable_verified_count"] == 1
    assert metadata["emotion_conditioned_count"] == 1
    assert metadata["episodes"][0]["emotion_id"] == "angry"
    assert metadata["episodes"][0]["source_emotion_label_verified"] is True
    assert metadata["episodes"][0]["emotion_conditioning_mask"] is False
    assert metadata["episodes"][1]["observed_affect"] == "angry_like"
    assert loaded["episodes"] == metadata["episodes"]

    attached = attach_condition_cache(episodes, cache)
    assert attached[0]["official_category_conditioning_enabled"] is False
    assert attached[0]["emotion_id"] == "angry"
    assert attached[0]["emotion_conditioning_mask"] is False
    assert attached[1]["emotion_conditioning_mask"] is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "official_category_conditioning_enabled",
            True,
            "official_category_conditioning_enabled must be false",
        ),
        (
            "source_emotion_label_verified",
            False,
            "source_emotion_label_verified must be true",
        ),
    ],
)
def test_official_v7_semantic_policy_tampering_fails_closed(field, value, message):
    record = official_semantic_episode()
    record[field] = value
    with pytest.raises(ValueError, match=message):
        semantic_supervision_policy(record)


def test_official_v7_prompt_mask_tampering_fails_closed():
    record = official_semantic_episode()
    record["semantic_supervision_masks"]["prompt_text"] = True
    with pytest.raises(ValueError, match="semantic_supervision_masks"):
        semantic_supervision_policy(record)


@pytest.mark.parametrize(
    ("semantic_fields", "message"),
    [
        ({"emotion_id": "happy"}, "missing explicit behavior_id"),
        (
            {"behavior_id": "Behavior.NotReal", "emotion_id": "happy"},
            "unknown Kimodo behavior_id",
        ),
        (
            {"behavior_id": "Behavior.Alert", "emotion_id": "unknown"},
            "unknown Kimodo emotion_id",
        ),
    ],
)
def test_18d_manifest_rejects_missing_or_illegal_structured_ids(
    tmp_path, semantic_fields, message
):
    csv_path = tmp_path / "clip_gmr_safe_18d.csv"
    write_18d_csv(csv_path, make_trajectory())
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "clip_id": "clip",
                "canonical_prompt": {"en": "A deliberately misleading happy wave."},
                "trajectory_path": str(csv_path),
                **semantic_fields,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=message):
        load_18d_episodes(manifest=manifest, allow_unreviewed=True)


def test_unresolved_emotion_contract_requires_absent_id_and_false_mask(tmp_path):
    csv_path = tmp_path / "clip_gmr_safe_18d.csv"
    write_18d_csv(csv_path, make_trajectory())
    base_record = {
        "clip_id": "clip",
        "canonical_prompt": {"en": "A gesture with uncertain affect."},
        "trajectory_path": str(csv_path),
        "behavior_id": "Behavior.Hesitate",
        "emotion_review_status": "unresolved",
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(base_record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="requires emotion_supervision_mask=false"):
        load_18d_episodes(manifest=manifest, allow_unreviewed=True)

    base_record["emotion_supervision_mask"] = False
    base_record["emotion_id"] = "neutral"
    manifest.write_text(json.dumps(base_record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must not carry an emotion_id"):
        load_18d_episodes(manifest=manifest, allow_unreviewed=True)


@pytest.mark.parametrize(
    ("behavior_fields", "message"),
    [
        (
            {
                "behavior_review_status": "candidate_unreviewed",
                "behavior_supervision_mask": True,
            },
            "requires behavior_supervision_mask=false",
        ),
        (
            {
                "behavior_review_status": "human_confirmed",
                "behavior_supervision_mask": False,
            },
            "requires behavior_supervision_mask=true",
        ),
        (
            {
                "behavior_review_status": "model_guessed",
                "behavior_supervision_mask": False,
            },
            "unknown behavior_review_status",
        ),
    ],
)
def test_behavior_supervision_contract_fails_closed(tmp_path, behavior_fields, message):
    csv_path = tmp_path / "clip_gmr_safe_18d.csv"
    write_18d_csv(csv_path, make_trajectory())
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "clip_id": "clip",
                "canonical_prompt": {"en": "Move both arms."},
                "trajectory_path": str(csv_path),
                "behavior_id": "Behavior.InteractPresence",
                "emotion_review_status": "unresolved",
                "emotion_supervision_mask": False,
                **behavior_fields,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=message):
        load_18d_episodes(manifest=manifest, allow_unreviewed=True)


def test_candidate_behavior_without_legacy_mask_defaults_to_unsupervised(tmp_path):
    csv_path = tmp_path / "clip_gmr_safe_18d.csv"
    write_18d_csv(csv_path, make_trajectory())
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "clip_id": "clip",
                "canonical_prompt": {"en": "Move across multiple head axes."},
                "trajectory_path": str(csv_path),
                "behavior_id": "Behavior.InteractPresence",
                "behavior_review_status": "candidate_unreviewed",
                "emotion_review_status": "unresolved",
                "emotion_supervision_mask": False,
                "motion_style": "slow_safe",
                "observed_affect": {"label": None, "confidence": 0.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    episode = load_18d_episodes(manifest=manifest, allow_unreviewed=True)[0]
    assert episode["behavior_supervision_mask"] is False
    assert episode["observed_affect"] == "low_confidence_unknown"
    assert episode["motion_style"] == "restrained"
    assert episode["semantic_gesture"] == "upper_body_gesture"


def test_condition_cache_and_cpu_smoke_training(tmp_path):
    base_path = tmp_path / "base.pt"
    base = make_15d_checkpoint(base_path)
    episodes = []
    clip_ids = []
    prompts = []
    conditions = []
    for index in range(3):
        clip_id = f"clip_{index}"
        prompt = f"Observable action {index}"
        clip_ids.append(clip_id)
        prompts.append(prompt)
        conditions.append(np.full(KIMODO_V2_CONDITION_DIM, index / 10.0, dtype=np.float32))
        episodes.append(
            {
                "clip_id": clip_id,
                "prompt": prompt,
                "actions": make_trajectory(index * 0.01),
                "fps": 30.0,
                "duration_sec": 0.4,
            }
        )
    cache = tmp_path / "conditions.npz"
    np.savez_compressed(
        cache,
        clip_ids=np.asarray(clip_ids),
        prompts=np.asarray(prompts),
        conditions=np.stack(conditions),
    )
    with pytest.raises(ValueError, match="metadata is required"):
        attach_condition_cache(episodes, cache)
    episodes = attach_condition_cache(episodes, cache, allow_unsafe_metadata=True)

    summary = train_head_adapter(
        base_checkpoint_path=base_path,
        episodes=episodes,
        output_dir=tmp_path / "run",
        steps=2,
        batch_size=2,
        frames=8,
        lr=1e-3,
        device="cpu",
        seed=3,
        log_interval=1,
        allow_unsafe_condition_cache=True,
    )

    assert summary["steps"] == 2
    assert summary["frozen_weight_prefix_max_abs_error"] == 0.0
    assert summary["legacy_zero_padded_forward_max_abs_error"] == 0.0
    assert Path(summary["checkpoint"]).is_file()
    _, trained = load_contract_checkpoint(summary["checkpoint"], expected_action_dim=ACTION_DIM)
    assert validate_checkpoint_contract(trained) == JOINT_ORDER_18D
    assert torch.equal(trained["action_stats"]["mean"][:15], base["action_stats"]["mean"])
    assert trained["data_provenance"]["episode_count"] == 3
    assert trained["data_provenance"]["unsafe_training_data"] is True
    assert trained["data_provenance"]["unsafe_condition_cache"] is True
    assert trained["data_provenance"]["condition_cache"]["cache_sha256"]
    assert trained["training_contract"]["body_distillation_weight"] == pytest.approx(1.0)
    assert trained["training_contract"]["student_forward_mode"] == "eval_dropout_disabled"
    assert trained["body_compatibility"]["passed"] is True
    assert summary["body_compatibility"]["passed"] is True
    first_progress = json.loads(
        (tmp_path / "run" / "progress.jsonl").read_text().splitlines()[0]
    )
    assert first_progress["train"]["body_distillation"] < 1e-10


def test_contract_export_rejects_unknown_dimensions(tmp_path):
    values = make_trajectory(frames=4)
    write_contract_csv(tmp_path / "motion.csv", values)
    write_contract_npz(tmp_path / "motion.npz", values, prompt="wave")
    assert (tmp_path / "motion.csv").read_text().splitlines()[0] == "time_sec," + ",".join(
        JOINT_ORDER_18D
    )
    with pytest.raises(ValueError, match="15 or 18"):
        write_contract_csv(tmp_path / "bad.csv", np.zeros((4, 17), dtype=np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        write_contract_csv(tmp_path / "nan.csv", np.full((4, 18), np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="positive"):
        write_contract_npz(tmp_path / "zero_fps.npz", values, fps=0)
    with np.load(tmp_path / "motion.npz", allow_pickle=False) as payload:
        assert payload["action_contract"].item() == CONTRACT_VERSION


def test_legacy_inference_and_lerobot_export_refuse_to_truncate_18d(tmp_path):
    base_path = tmp_path / "base.pt"
    base = make_15d_checkpoint(base_path)
    stats = compute_18d_action_stats([make_trajectory()], base["action_stats"])
    migrated, _ = migrate_15d_checkpoint(base_path, tmp_path / "18d.pt", action_stats=stats)
    with pytest.raises(ValueError, match="will not truncate"):
        validate_generator_checkpoint(migrated)

    csv_path = tmp_path / "motion_18d.csv"
    write_18d_csv(csv_path, make_trajectory())
    with pytest.raises(ValueError, match="silently drop"):
        read_joint_window(csv_path, 0, 2)


def test_inference_benchmark_measures_full_trajectory(tmp_path):
    base_path = tmp_path / "base.pt"
    base = make_15d_checkpoint(base_path)
    stats = compute_18d_action_stats([make_trajectory()], base["action_stats"])
    migrated, _ = migrate_15d_checkpoint(base_path, tmp_path / "18d.pt", action_stats=stats)
    model = instantiate_checkpoint_model(migrated)

    metrics, trajectory = benchmark_contract_inference(
        model,
        np.zeros(KIMODO_V2_CONDITION_DIM, dtype=np.float32),
        frames=6,
        steps=2,
        warmup=1,
        repeats=2,
        device="cpu",
    )

    assert trajectory.shape == (6, ACTION_DIM)
    assert metrics["wall_seconds_median"] > 0
    assert metrics["playback_control_hz"] == 30.0
    assert "not a causal control-loop rate" in metrics["note"]


def test_native_inference_uses_learned_complete_arc_duration():
    payload = {
        "contract_type": "ula_v2_native_complete_expression_duration",
        "fixed_frame_count": None,
        "fixed_duration_sec": None,
        "trajectory_representation": "native_variable_length_complete_expression_arc",
        "duration_supervision_sec": {"min": 1.0, "median": 3.0, "max": 9.0},
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()

    class DurationModel:
        training = True

        def eval(self):
            self.training = False
            return self

        def train(self, mode=True):
            self.training = mode
            return self

        def plan_condition(self, condition):
            return {
                "duration_sec": torch.tensor([4.25], device=condition.device),
                "transition_logits": torch.zeros((1, 4), device=condition.device),
            }

    frames, predicted = predict_contract_frame_count(
        DurationModel(),
        {"v2_contracts": {"duration": payload}},
        np.zeros(KIMODO_V2_CONDITION_DIM, dtype=np.float32),
        fps=30.0,
    )
    assert predicted == pytest.approx(4.25)
    assert frames == 129

    fixed = dict(payload)
    fixed["fixed_duration_sec"] = 6.0
    fixed["sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in fixed.items() if key != "sha256"},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    with pytest.raises(ValueError, match="fixed-frame or fixed-duration"):
        predict_contract_frame_count(
            DurationModel(),
            {"v2_contracts": {"duration": fixed}},
            np.zeros(KIMODO_V2_CONDITION_DIM, dtype=np.float32),
        )


def test_versioned_condition_cache_is_bound_to_generator_qwen_contract(tmp_path):
    clip_ids = ["clip_a"]
    prompts = ["Wave one hand."]
    cache = tmp_path / "conditions.npz"
    np.savez_compressed(
        cache,
        clip_ids=np.asarray(clip_ids),
        prompts=np.asarray(prompts),
        conditions=np.zeros((1, KIMODO_V2_CONDITION_DIM), dtype=np.float32),
    )
    qwen = tmp_path / "qwen.pt"
    qwen_record = {"model_name": "Qwen/test", "revision": "pinned-revision"}
    torch.save(
        {
            "schema_version": 1,
            "artifact_kind": "qwen_motion_cross_modal_alignment",
            "qwen": qwen_record,
        },
        qwen,
    )
    qwen_sha = hashlib.sha256(qwen.read_bytes()).hexdigest()
    cache_sha = hashlib.sha256(cache.read_bytes()).hexdigest()
    metadata = {
        "schema_version": 1,
        "artifact_kind": "ula_v2_qwen_motion_condition_cache",
        "condition_dim": KIMODO_V2_CONDITION_DIM,
        "base_condition_dim": 136,
        "motion_latent_dim": 128,
        "count": 1,
        "cache_sha256": cache_sha,
        "qwen_checkpoint": str(qwen),
        "qwen_checkpoint_sha256": qwen_sha,
        "qwen_model_name": qwen_record["model_name"],
        "qwen_revision": qwen_record["revision"],
        "episodes": [
            {
                "clip_id": clip_ids[0],
                "prompt_sha256": hashlib.sha256(prompts[0].encode()).hexdigest(),
            }
        ],
    }
    metadata_path = cache.with_suffix(".npz.json")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    _, _, _, provenance = load_condition_cache(cache)
    generator = {
        "v2_contracts": {
            "text_motion_latent": {
                "source": {
                    "checkpoint_sha256": qwen_sha,
                    "model_name": qwen_record["model_name"],
                    "revision": qwen_record["revision"],
                }
            }
        }
    }
    assert validate_condition_cache_for_generator(generator, provenance)["validated"] is True
    assert validate_qwen_checkpoint_for_generator(generator, qwen)["validated"] is True

    wrong_generator = json.loads(json.dumps(generator))
    wrong_generator["v2_contracts"]["text_motion_latent"]["source"][
        "checkpoint_sha256"
    ] = "0" * 64
    with pytest.raises(ValueError, match="does not match"):
        validate_condition_cache_for_generator(wrong_generator, provenance)
    with pytest.raises(ValueError, match="does not match"):
        validate_qwen_checkpoint_for_generator(wrong_generator, qwen)

    metadata["qwen_revision"] = "wrong-revision"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        load_condition_cache(cache)


def test_condition_cache_accepts_verified_immediate_posttrain_lineage(tmp_path):
    derived_path, derived, source_path, cache = make_posttrain_cache_lineage(
        tmp_path / "valid_posttrain"
    )

    result = validate_condition_cache_for_generator(
        derived,
        cache,
        generator_checkpoint_path=derived_path,
    )

    assert result["validated"] is True
    assert result["generator_checkpoint_compatibility"] == (
        "verified_immediate_posttrain_source"
    )
    assert result["posttrain_lineage"]["source_checkpoint"] == str(
        source_path.resolve()
    )
    assert result["posttrain_lineage"]["source_checkpoint_sha256"] == cache[
        "generator_checkpoint_sha256"
    ]


def mutate_posttrain_lineage(checkpoint, case):
    if case == "missing_artifact_kind":
        checkpoint.pop("posttrain_artifact_kind")
    elif case == "missing_source":
        checkpoint.pop("posttrain_source")
    elif case == "missing_source_path":
        checkpoint["posttrain_source"].pop("checkpoint")
    elif case == "wrong_source_hash":
        checkpoint["posttrain_source"]["checkpoint_sha256"] = "0" * 64
    elif case == "wrong_data_contract_hash":
        checkpoint["posttrain_data_contract"]["sha256"] = "0" * 64
    elif case == "missing_training_contract":
        checkpoint.pop("training_contract")
    elif case == "wrong_cache_lineage_hash":
        checkpoint["data_provenance"]["condition_cache"][
            "generator_checkpoint_sha256"
        ] = "0" * 64
    elif case == "wrong_global_step":
        checkpoint["global_step"] += 1
    else:
        raise AssertionError(case)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing_artifact_kind", "posttrain_artifact_kind"),
        ("missing_source", "posttrain_source must be an object"),
        ("missing_source_path", "posttrain_source.checkpoint is missing"),
        ("wrong_source_hash", "immediate source hash"),
        ("wrong_data_contract_hash", "posttrain_data_contract hash is invalid"),
        ("missing_training_contract", "training_contract is missing"),
        ("wrong_cache_lineage_hash", "condition cache changed field"),
        ("wrong_global_step", "global_step does not equal"),
    ],
)
def test_condition_cache_rejects_incomplete_or_forged_posttrain_lineage(
    tmp_path, case, message
):
    derived_path, derived, _, cache = make_posttrain_cache_lineage(tmp_path / case)
    mutate_posttrain_lineage(derived, case)
    torch.save(derived, derived_path)

    with pytest.raises(ValueError, match=message):
        validate_condition_cache_for_generator(
            derived,
            cache,
            generator_checkpoint_path=derived_path,
        )


def test_condition_cache_rejects_multihop_posttrain_ancestor_match(tmp_path):
    derived_path, derived, _, cache = make_posttrain_cache_lineage(
        tmp_path / "multihop"
    )
    child = torch.load(derived_path, map_location="cpu", weights_only=True)
    child_step = 3
    child["posttrain_source"] = {
        "checkpoint": str(derived_path),
        "checkpoint_sha256": hashlib.sha256(derived_path.read_bytes()).hexdigest(),
        "source_global_step": int(derived["global_step"]),
    }
    child["posttrain_step"] = child_step
    child["global_step"] = int(derived["global_step"]) + child_step
    child_path = tmp_path / "multihop" / "child_posttrained_18d.pt"
    torch.save(child, child_path)

    with pytest.raises(ValueError, match="immediate source hash"):
        validate_condition_cache_for_generator(
            child,
            cache,
            generator_checkpoint_path=child_path,
        )


def test_versioned_condition_cache_requires_generator_checkpoint_path(tmp_path):
    _, derived, _, cache = make_posttrain_cache_lineage(tmp_path / "missing_current_path")

    with pytest.raises(ValueError, match="requires a generator checkpoint path"):
        validate_condition_cache_for_generator(derived, cache)


def test_body_compatibility_metrics_cover_nonzero_head_and_complete_sampling(tmp_path):
    base_path = tmp_path / "base.pt"
    base = make_15d_checkpoint(base_path)
    stats = compute_18d_action_stats([make_trajectory()], base["action_stats"])
    migrated, _ = migrate_15d_checkpoint(base_path, tmp_path / "18d.pt", action_stats=stats)
    base_model = instantiate_checkpoint_model(base)
    expanded_model = instantiate_checkpoint_model(migrated)

    forward = nonzero_head_forward_drift_metrics(
        base_model, expanded_model, seed=9, batch_size=2, frames=6
    )
    sampling = body_sampling_drift_metrics(
        base_model,
        expanded_model,
        [np.zeros(KIMODO_V2_CONDITION_DIM, dtype=np.float32)],
        action_stats=stats,
        frames=6,
        steps=2,
        seeds=(11,),
    )
    assert forward["max_abs_normalized"] == 0.0
    assert sampling["body_max_abs_rad"] == 0.0

    with torch.no_grad():
        expanded_model.input.weight[:, LEGACY_ACTION_DIM:].normal_(std=0.1)
    changed = nonzero_head_forward_drift_metrics(
        base_model, expanded_model, seed=9, batch_size=2, frames=6
    )
    assert changed["mean_abs_normalized"] > 0.0
