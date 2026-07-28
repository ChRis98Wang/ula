from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest
import torch

from tools.init_ula_v2_18d_random import read_config
from tools.train_ula_v2_18d_posttrain import resolve_bound_motion_sources
from upper_body_skeleton.kimodo_semantics import (
    KIMODO_BEHAVIOR_IDS,
    KIMODO_EMOTION_IDS,
)
from upper_body_skeleton.ula_training import (
    KIMODO_V2_CONDITION_DIM,
    ULA_MMDIT_V2_ARCHITECTURE,
    create_ula_model,
)
from upper_body_skeleton.ula_v2_18d_head import (
    ACTION_DIM,
    FORMAL_SELECTED_LINEAGE_FIELDS,
    MOTION_ONLY_EPISODE_CONTRACT,
    MOTION_ONLY_NO_KIMODO_POLICY,
    MOTION_ONLY_NO_QWEN_POLICY,
    MOTION_ONLY_RANDOM_INIT_MODE,
    STYLE_CONTROL_SLICE,
    build_motion_only_condition_cache,
    load_condition_cache,
    validate_checkpoint_contract,
    validate_motion_only_style_condition,
)
from upper_body_skeleton.ula_v2_18d_random_init import (
    FORMAL_SEMANTIC_EVENT_SELECTION_STATUS,
    PROJECT_BEHAVIOR_MAPPING_SOURCE,
    RANDOM_INIT_MODE,
    RETARGET_SEGMENT_REPRESENTATION,
    VARIABLE_SEGMENT_REPRESENTATION,
    build_random_18d_checkpoint,
    collate_variable_length_18d,
    default_style_evaluation_conditions,
    forward_with_frame_mask,
    validate_formal_variable_length_episode,
    validate_random_checkpoint_split,
)
from upper_body_skeleton.ula_v2_18d_posttrain import masked_18d_objective
from upper_body_skeleton.ula_v2_18d_posttrain import transition_supervision_contract


def _retarget_segment(start, end, output_frames, *, fps=30.0):
    source_frames = end - start
    payload = {
        "representation": RETARGET_SEGMENT_REPRESENTATION,
        "source_start_frame": start,
        "source_end_frame_exclusive": end,
        "source_frame_count": source_frames,
        "source_frame_coverage_sec": source_frames / fps,
        "output_frame_count": output_frames,
        "output_sample_span_sec": (output_frames - 1) / fps,
        "output_frame_coverage_sec": output_frames / fps,
        "fps": fps,
        "retimed": output_frames != source_frames,
        "cropped": False,
        "planner_duration_field": "output_sample_span_sec",
        "source_boundary_duration_field": "source_frame_coverage_sec",
        "legacy_quality_duration_sec_role": (
            "output_frame_coverage_compatibility_only_not_planner_target"
        ),
    }
    return payload | {
        "sha256": hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest()
    }


def _qwen_checkpoint(path):
    torch.save(
        {
            "schema_version": 1,
            "artifact_kind": "qwen_motion_cross_modal_alignment",
            "global_step": 12,
            "best_step": 11,
            "qwen": {
                "model_name": "Qwen/Qwen3-Embedding-0.6B",
                "revision": "pinned-test-revision",
            },
            "config": {"latent_dim": 128},
            "qwen_lora_state_dict": {"layer.lora_A": torch.ones(2, 2)},
        },
        path,
    )
    return path


def _episodes():
    rows = []
    for speaker_index, speaker in enumerate(("speaker-a", "speaker-b", "speaker-c")):
        for local_index, frames in enumerate((5, 7)):
            value = float(1 + speaker_index * 10 + local_index * 2)
            actions = np.full((frames, ACTION_DIM), value, dtype=np.float32)
            actions += np.arange(ACTION_DIM, dtype=np.float32)[None, :] * 0.01
            source_sha256 = hashlib.sha256(
                f"source:{speaker}:{local_index}".encode()
            ).hexdigest()
            behavior_payload = {
                "source": PROJECT_BEHAVIOR_MAPPING_SOURCE,
                "revision": "pilot-test-v1",
                "behavior_id": "Behavior.InteractPresence",
                "supervision": "weak_candidate_masked",
                "scope": "co_speech_interaction_coarse_only",
            }
            behavior_contract = behavior_payload | {
                "sha256": hashlib.sha256(
                    json.dumps(
                        behavior_payload,
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("ascii")
                ).hexdigest()
            }
            emotion_id = KIMODO_EMOTION_IDS[speaker_index]
            emotion_payload = {
                "source": "official_beat2_filename_protocol",
                "revision": "beat2-test-protocol-v1",
                "emotion_id": emotion_id,
                "source_sha256": source_sha256,
            }
            emotion_contract = emotion_payload | {
                "sha256": hashlib.sha256(
                    json.dumps(
                        emotion_payload,
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("ascii")
                ).hexdigest()
            }
            source_lineage = {
                field: hashlib.sha256(
                    f"{field}:{speaker}:{local_index}".encode("utf-8")
                ).hexdigest()
                for field in FORMAL_SELECTED_LINEAGE_FIELDS
            }
            rows.append(
                {
                    "clip_id": f"{speaker}-{local_index}",
                    "actions": actions,
                    "fps": 30.0,
                    "prompt": f"Visible interaction motion {speaker_index} {local_index}",
                    "behavior_id": "Behavior.InteractPresence",
                    "behavior_review_status": "candidate_unreviewed",
                    "behavior_supervision_mask": False,
                    "behavior_source": PROJECT_BEHAVIOR_MAPPING_SOURCE,
                    "behavior_mapping_contract": behavior_contract,
                    "emotion_id": emotion_id,
                    "emotion_review_status": "official_protocol_confirmed",
                    "emotion_supervision_mask": False,
                    "source_emotion_label_verified": True,
                    "emotion_supervision_role": (
                        "disabled_pending_robot_affect_review"
                    ),
                    "official_emotion_conditioning_enabled": False,
                    "official_emotion_condition_channel": None,
                    "official_emotion_loss": None,
                    "affect_observable_review_status": "not_verified",
                    "affect_observable_supervision_mask": False,
                    "emotion_conditioning_mask": False,
                    "emotion_source": "official_beat2_filename_protocol",
                    "emotion_protocol_contract": emotion_contract,
                    "accepted_for_training": True,
                    "eligibility_mode": "adjudicated_train_ready",
                    "dataset_source": "ula0513_user_owned",
                    "speaker_key": speaker,
                    "source_group_key": f"{speaker}/source-{local_index}",
                    "source_clip_id": f"{speaker}_source_{local_index}",
                    "source_sha256": source_sha256,
                    "trajectory_sha256": hashlib.sha256(
                        f"trajectory:{speaker}:{local_index}".encode()
                    ).hexdigest(),
                    "source_manifest_sha256": "a" * 64,
                    "source_record_sha256": hashlib.sha256(
                        f"record:{speaker}:{local_index}".encode()
                    ).hexdigest(),
                    "annotation_kind": "official_gesture_semantic_event",
                    "canonical_action": "official_gesture_category:deictic",
                    "canonical_action_role": (
                        "official_category_metadata_split_key_only"
                    ),
                    "semantic_mapping_status": (
                        "official_category_verified_metadata_only"
                    ),
                    "official_category_verified": True,
                    "official_category_conditioning_enabled": False,
                    "official_category_role": (
                        "verified_metadata_split_and_evaluation_only"
                    ),
                    "official_category_condition_channel": None,
                    "official_category_loss": None,
                    "robot_observable_motion_form": "candidate_unreviewed",
                    "communicative_intent": "candidate_unreviewed",
                    "canonical_prompt_role": "coarse_category_only",
                    "semantic_supervision_masks": {
                        "official_category": False,
                        "robot_observable_motion_form": False,
                        "communicative_intent": False,
                        "prompt_text": False,
                        "legacy_gesture": False,
                    },
                    "semantic_event": {
                        "category": "deictic",
                        "intensity": "high",
                        "lexical_anchor": "there",
                    },
                    "semantic_gesture": "pointing",
                    "retarget_qc_passed": True,
                    "retarget_segment": _retarget_segment(
                        100 * local_index,
                        100 * local_index + frames,
                        frames,
                    ),
                    "quality_source_window_frames": frames,
                    "quality_output_frame_count": frames,
                    "formal_source_metadata": dict(source_lineage),
                    "retarget_source_lineage": dict(source_lineage),
                    "window": {
                        "selection_status": FORMAL_SEMANTIC_EVENT_SELECTION_STATUS
                    },
                    "training_segment": {
                        "representation": VARIABLE_SEGMENT_REPRESENTATION,
                        "fixed_window_sec": None,
                        "start_frame": 100 * local_index,
                        "end_frame_exclusive": 100 * local_index + frames,
                        "frame_count": frames,
                        "boundary_source": {
                            "mode": "official_sem_core_plus_motion_low_speed_context"
                        },
                    },
                }
            )
    return rows


def _build(tmp_path, episodes=None, *, seed=7):
    episodes = episodes or _episodes()
    motion_only = all(
        episode.get("formal_episode_contract") == MOTION_ONLY_EPISODE_CONTRACT
        for episode in episodes
    )
    qwen = None if motion_only else _qwen_checkpoint(tmp_path / "qwen.pt")
    source_provenance = {
        "dataset_source": (
            "beat2_official_semantic_event_training_pool_v7"
            if motion_only
            else "ula0513_user_owned"
        ),
        "manifest_sha256": "a" * 64,
    }
    if motion_only:
        source_provenance.update(
            manifest_fixed_split=True,
            fixed_split_assignment_sha256="b" * 64,
            license_gate={"dataset_family": "BEAT2"},
        )
    return build_random_18d_checkpoint(
        episodes,
        qwen_checkpoint=qwen,
        source_provenance=[source_provenance],
        seed=seed,
        fractions={"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3},
        hidden_dim=32,
        layers=2,
        semantic_tokens=7,
        length_buckets=(8, 16),
    )


def _motion_only_episodes():
    episodes = deepcopy(_episodes())
    split_by_speaker = {
        "speaker-a": "train",
        "speaker-b": "validation",
        "speaker-c": "test",
    }
    for episode in episodes:
        episode["dataset_source"] = (
            "beat2_official_semantic_event_training_pool_v7"
        )
        episode["fixed_split_assignment"] = split_by_speaker[
            episode["speaker_key"]
        ]
        episode["formal_episode_contract"] = MOTION_ONLY_EPISODE_CONTRACT
        episode["motion_only_admission"] = {
            "physical_qc_only": True,
            "semantic_review_required": False,
            "independent_semantic_review_claimed": False,
            "text_conditioning_enabled": False,
            "emotion_conditioning_enabled": False,
            "audio_conditioning_enabled": False,
            "native_variable_length": True,
            "fixed_duration_training_unit": False,
            "source_record_sha256": episode["source_record_sha256"],
        }
    return episodes


def test_motion_only_random_checkpoint_preserves_masked_contract(tmp_path):
    checkpoint, _, report = _build(tmp_path, _motion_only_episodes())

    assert checkpoint["formal_episode_contract"] == MOTION_ONLY_EPISODE_CONTRACT
    assert checkpoint["config"]["formal_episode_contract"] == MOTION_ONLY_EPISODE_CONTRACT
    assert checkpoint["v2_contracts"]["formal_episode_contract"] == (
        MOTION_ONLY_EPISODE_CONTRACT
    )
    assert checkpoint["v2_contracts"]["batching"]["source_representation"] == (
        VARIABLE_SEGMENT_REPRESENTATION
    )
    duration = checkpoint["v2_contracts"]["duration"]
    assert duration["semantic_unit"] == "native_physical_qc_motion_segment"
    assert "onset_apex_offset" not in duration["duration_supervision_policy"]
    text = checkpoint["v2_contracts"]["text_motion_latent"]
    assert text["conditioned_episode_count"] == 0
    assert text["masked_episode_count"] == len(_episodes())
    assert text["text_field"] is None
    assert text["conditioning_policy"] == "all_text_latents_exact_zero"
    assert "source" not in text
    assert "qwen_checkpoint_sha256" not in checkpoint["sources"]
    assert checkpoint["random_initialization"]["mode"] == MOTION_ONLY_RANDOM_INIT_MODE
    assert checkpoint["random_initialization"]["qwen_policy"] == (
        MOTION_ONLY_NO_QWEN_POLICY
    )
    assert checkpoint["random_initialization"]["kimodo_policy"] == (
        MOTION_ONLY_NO_KIMODO_POLICY
    )
    supervision = checkpoint["semantic_supervision_contract"]
    assert supervision["semantic_conditioned_count"] == 0
    assert supervision["expressive_conditioned_count"] == 0
    assert supervision["audio_conditioned_count"] == 0
    assert report["formal_episode_contract"] == MOTION_ONLY_EPISODE_CONTRACT
    assert report["primary_evaluation"] == (
        "motion_only_zero_text_emotion_audio_default_style"
    )
    assert report["split_counts"] == {"train": 2, "validation": 2, "test": 2}


def test_motion_only_random_checkpoint_rejects_qwen_and_generated_split(tmp_path):
    episodes = _motion_only_episodes()
    with pytest.raises(ValueError, match="forbids a Qwen"):
        build_random_18d_checkpoint(
            episodes,
            qwen_checkpoint=_qwen_checkpoint(tmp_path / "qwen.pt"),
            source_provenance=[],
            hidden_dim=32,
            layers=2,
        )

    for episode in episodes:
        episode.pop("fixed_split_assignment")
    with pytest.raises(ValueError, match="fixed split"):
        build_random_18d_checkpoint(
            episodes,
            qwen_checkpoint=None,
            source_provenance=[],
            hidden_dim=32,
            layers=2,
        )


def test_motion_only_condition_is_style_only_fail_closed():
    condition = np.zeros(KIMODO_V2_CONDITION_DIM, dtype=np.float32)
    condition[STYLE_CONTROL_SLICE] = [0.25, -0.5, 0.75]
    validate_motion_only_style_condition(condition)
    for index in (0, 28, 91, 92, 132, 136, 263):
        tampered = condition.copy()
        tampered[index] = 1.0
        with pytest.raises(ValueError, match="exactly zero"):
            validate_motion_only_style_condition(tampered)


def test_motion_only_cache_has_no_qwen_lineage_and_rejects_hidden_channels(
    tmp_path,
):
    episodes = _motion_only_episodes()
    checkpoint, _, _ = _build(tmp_path, episodes)
    checkpoint_path = tmp_path / "random_init.pt"
    torch.save(checkpoint, checkpoint_path)
    cache_path = tmp_path / "conditions.npz"

    metadata = build_motion_only_condition_cache(
        episodes,
        cache_path,
        base_checkpoint=checkpoint_path,
    )

    assert metadata["qwen_policy"] == MOTION_ONLY_NO_QWEN_POLICY
    assert metadata["kimodo_policy"] == MOTION_ONLY_NO_KIMODO_POLICY
    assert not any(key.startswith("qwen_checkpoint") for key in metadata)
    _, _, conditions, loaded = load_condition_cache(cache_path)
    validate_motion_only_style_condition(conditions)
    assert loaded["unsafe_condition_cache"] is False

    with np.load(cache_path, allow_pickle=False) as payload:
        values = {name: payload[name].copy() for name in payload.files}
    values["conditions"][0, 28] = 1.0
    np.savez_compressed(cache_path, **values)
    metadata["cache_sha256"] = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    cache_path.with_suffix(".npz.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="exactly zero"):
        load_condition_cache(cache_path)


def test_random_checkpoint_rejects_mixed_motion_only_and_legacy_contracts(tmp_path):
    episodes = _motion_only_episodes()
    episodes[0].pop("formal_episode_contract")
    with pytest.raises(ValueError, match="cannot mix"):
        _build(tmp_path, episodes)


def test_full_random_checkpoint_has_no_generator_lineage_and_is_deterministic(tmp_path):
    torch.manual_seed(991)
    rng_before = torch.get_rng_state().clone()
    first, split, report = _build(tmp_path)
    rng_after = torch.get_rng_state().clone()
    second, _, _ = _build(tmp_path)

    assert torch.equal(rng_before, rng_after)
    assert validate_checkpoint_contract(first, expected_action_dim=18)
    assert first["global_step"] == 0
    assert first["artifact_status"].startswith("untrained_random")
    assert first["random_initialization"]["mode"] == RANDOM_INIT_MODE
    assert first["random_initialization"]["generator_checkpoint_inputs"] == []
    assert "migration_source" not in first
    assert "posttrain_source" not in first
    assert report["formal_training_started"] is False
    assert report["forgetting_guard_applicable"] is False
    assert split["counts"] == {"train": 2, "validation": 2, "test": 2}
    for name, value in first["model_state_dict"].items():
        assert torch.equal(value, second["model_state_dict"][name]), name

    layer_zero = first["model_state_dict"]["blocks.layers.0.self_attn.in_proj_weight"]
    layer_one = first["model_state_dict"]["blocks.layers.1.self_attn.in_proj_weight"]
    assert not torch.equal(layer_zero, layer_one)

    duration = first["v2_contracts"]["duration"]
    assert duration["fixed_frame_count"] is None
    assert duration["fixed_duration_sec"] is None
    assert duration["semantic_unit"] == "natural_onset_apex_offset_expression_arc"
    assert duration["duration_formula"] == "(frame_count-1)/fps"
    assert duration["duration_supervision_episode_count"] == 2
    assert duration["duration_supervision_sec"]["min"] != pytest.approx(6.0)
    assert set(duration["fit_clip_ids"]) == {
        row["clip_id"] for row in split["episodes"] if row["split"] == "train"
    }
    assert report["duration_contract_sha256"] == duration["sha256"]


def test_action_and_style_statistics_ignore_validation_and_test(tmp_path):
    episodes = _episodes()
    first, split, _ = _build(tmp_path, episodes)
    train_ids = {
        record["clip_id"] for record in split["episodes"] if record["split"] == "train"
    }
    changed = deepcopy(episodes)
    for episode in changed:
        if episode["clip_id"] not in train_ids:
            episode["actions"][:] = 100_000.0
    second, _, _ = _build(tmp_path, changed)

    assert torch.equal(first["action_stats"]["mean"], second["action_stats"]["mean"])
    assert torch.equal(first["action_stats"]["std"], second["action_stats"]["std"])
    first_contracts = first["v2_contracts"]
    second_contracts = second["v2_contracts"]
    assert first_contracts["action_statistics"] == second_contracts["action_statistics"]
    assert first_contracts["style"] == second_contracts["style"]
    assert first_contracts["duration"] == second_contracts["duration"]
    assert set(first_contracts["action_statistics"]["fit_clip_ids"]) == train_ids
    assert set(first_contracts["style"]["fit_clip_ids"]) == train_ids


def test_fixed_window_or_unresolved_labels_are_rejected():
    episode = _episodes()[0]
    episode["training_segment"]["fixed_window_sec"] = 6.0
    with pytest.raises(ValueError, match="fixed-window"):
        validate_formal_variable_length_episode(episode)

    episode = _episodes()[0]
    episode["emotion_supervision_mask"] = False
    episode["emotion_id"] = None
    with pytest.raises(ValueError, match="emotion_id"):
        validate_formal_variable_length_episode(episode)


def test_retimed_formal_episode_keeps_source_boundary_but_uses_output_duration():
    episode = _episodes()[0]
    episode["actions"] = np.zeros((124, ACTION_DIM), dtype=np.float32)
    episode["training_segment"].update(
        start_frame=900,
        end_frame_exclusive=998,
        frame_count=98,
    )
    episode["retarget_segment"] = _retarget_segment(900, 998, 124)
    episode["quality_source_window_frames"] = 98
    episode["quality_output_frame_count"] = 124

    validate_formal_variable_length_episode(episode)
    batch = collate_variable_length_18d([episode], buckets=(128,))

    assert batch["frame_counts"].tolist() == [124]
    assert batch["durations_sec"].tolist() == pytest.approx([123 / 30])
    assert episode["training_segment"]["frame_count"] == 98
    assert episode["retarget_segment"]["cropped"] is False

    forged = deepcopy(episode)
    forged["retarget_segment"] = _retarget_segment(900, 998, 123)
    with pytest.raises(ValueError, match="output_frame_count"):
        validate_formal_variable_length_episode(forged)

def test_native_length_collation_and_attention_mask_ignore_padding():
    episodes = _episodes()[:2]
    batch = collate_variable_length_18d(episodes, buckets=(8,))
    assert batch["actions"].shape == (2, 8, 18)
    assert batch["frame_valid_mask"].sum(dim=1).tolist() == [5, 7]
    assert batch["frame_counts"].tolist() == [5, 7]
    assert batch["durations_sec"].tolist() == pytest.approx([4 / 30, 6 / 30])

    torch.manual_seed(3)
    model = create_ula_model(
        ULA_MMDIT_V2_ARCHITECTURE,
        action_dim=ACTION_DIM,
        condition_dim=KIMODO_V2_CONDITION_DIM,
        hidden_dim=32,
        layers=1,
        semantic_tokens=7,
    ).eval()
    x = batch["actions"].clone()
    changed_padding = x.clone()
    changed_padding[0, 5:] = 999.0
    changed_padding[1, 7:] = -999.0
    t = torch.tensor([0.2, 0.8], dtype=torch.float32)
    condition = torch.zeros((2, KIMODO_V2_CONDITION_DIM), dtype=torch.float32)
    with torch.no_grad():
        baseline = forward_with_frame_mask(
            model, x, t, condition, batch["frame_valid_mask"]
        )
        changed = forward_with_frame_mask(
            model, changed_padding, t, condition, batch["frame_valid_mask"]
        )
    assert torch.allclose(baseline[0, :5], changed[0, :5], atol=1e-6)
    assert torch.allclose(baseline[1, :7], changed[1, :7], atol=1e-6)

    weights = {
        "flow": 1.0,
        "position": 1.0,
        "body": 0.0,
        "velocity": 1.0,
        "acceleration": 1.0,
        "jerk": 1.0,
        "head_flow": 1.0,
        "head_position": 1.0,
        "head_velocity": 1.0,
        "head_acceleration": 1.0,
        "head_jerk": 1.0,
    }
    first = masked_18d_objective(
        model,
        x,
        condition,
        batch["action_dim_mask"],
        batch["durations_sec"],
        loss_weights=weights,
        frame_valid_mask=batch["frame_valid_mask"],
        generator=torch.Generator().manual_seed(41),
    )
    second = masked_18d_objective(
        model,
        changed_padding,
        condition,
        batch["action_dim_mask"],
        batch["durations_sec"],
        loss_weights=weights,
        frame_valid_mask=batch["frame_valid_mask"],
        generator=torch.Generator().manual_seed(41),
    )
    assert first.keys() == second.keys()
    for name in first:
        assert torch.equal(first[name], second[name]), name

    larger_bucket = collate_variable_length_18d(episodes, buckets=(32,))
    with torch.no_grad():
        larger_output = forward_with_frame_mask(
            model,
            larger_bucket["actions"],
            t,
            condition,
            larger_bucket["frame_valid_mask"],
        )
    assert torch.allclose(baseline[0, :5], larger_output[0, :5], atol=1e-6)
    assert torch.allclose(baseline[1, :7], larger_output[1, :7], atol=1e-6)
    larger_loss = masked_18d_objective(
        model,
        larger_bucket["actions"],
        condition,
        larger_bucket["action_dim_mask"],
        larger_bucket["durations_sec"],
        loss_weights=weights,
        frame_valid_mask=larger_bucket["frame_valid_mask"],
        generator=torch.Generator().manual_seed(41),
    )
    for name in first:
        assert torch.allclose(first[name], larger_loss[name], atol=1e-5), name


def test_catalog_maximum_native_length_is_padded_not_cropped():
    episode = _episodes()[0]
    episode["actions"] = np.zeros((2589, ACTION_DIM), dtype=np.float32)
    batch = collate_variable_length_18d([episode], buckets=(512,))

    assert batch["actions"].shape == (1, 2592, ACTION_DIM)
    assert batch["frame_counts"].tolist() == [2589]
    assert batch["frame_valid_mask"].sum().item() == 2589
    assert not batch["frame_valid_mask"][0, 2589:].any()
    assert batch["durations_sec"].item() == pytest.approx(2588 / 30)


def test_primary_evaluation_zeros_only_oracle_style_controls():
    condition = np.arange(KIMODO_V2_CONDITION_DIM, dtype=np.float32)
    result = default_style_evaluation_conditions(condition)
    assert result[133:136].tolist() == [0.0, 0.0, 0.0]
    assert np.array_equal(result[:133], condition[:133])
    assert np.array_equal(result[136:], condition[136:])
    assert condition[133:136].tolist() != [0.0, 0.0, 0.0]


def test_random_split_contract_rejects_fraction_and_group_metadata_tampering(tmp_path):
    episodes = _episodes()
    checkpoint, _, _ = _build(tmp_path, episodes)
    with pytest.raises(ValueError, match="split fractions differ"):
        validate_random_checkpoint_split(
            checkpoint,
            episodes,
            requested_fractions={"train": 0.5, "validation": 0.25, "test": 0.25},
        )
    changed = deepcopy(episodes)
    changed[0]["speaker_key"] = "tampered-speaker"
    with pytest.raises(ValueError, match="speaker/source group differs"):
        validate_random_checkpoint_split(
            checkpoint,
            changed,
            requested_fractions={
                "train": 1 / 3,
                "validation": 1 / 3,
                "test": 1 / 3,
            },
        )


def test_native_planner_loss_updates_duration_and_transition_heads():
    episodes = _episodes()[:2]
    for episode in episodes:
        episode["condition"] = np.zeros(KIMODO_V2_CONDITION_DIM, dtype=np.float32)
    batch = collate_variable_length_18d(episodes, buckets=(8,))
    torch.manual_seed(31)
    model = create_ula_model(
        ULA_MMDIT_V2_ARCHITECTURE,
        action_dim=ACTION_DIM,
        condition_dim=KIMODO_V2_CONDITION_DIM,
        hidden_dim=32,
        layers=1,
        semantic_tokens=7,
    )
    losses = masked_18d_objective(
        model,
        batch["actions"],
        torch.zeros((2, KIMODO_V2_CONDITION_DIM)),
        batch["action_dim_mask"],
        batch["durations_sec"],
        loss_weights={
            "flow": 0.0,
            "position": 0.0,
            "body": 0.0,
            "velocity": 0.0,
            "acceleration": 0.0,
            "planner": 1.0,
        },
        frame_valid_mask=batch["frame_valid_mask"],
        transition_targets=torch.tensor([3, 2], dtype=torch.long),
        generator=torch.Generator().manual_seed(9),
    )
    losses["total"].backward()
    assert torch.count_nonzero(model.duration_head.weight.grad) > 0
    assert torch.count_nonzero(model.transition_head.weight.grad) > 0


def test_formal_duration_supervision_does_not_train_unlabeled_transition_head():
    model = create_ula_model(
        ULA_MMDIT_V2_ARCHITECTURE,
        action_dim=ACTION_DIM,
        condition_dim=KIMODO_V2_CONDITION_DIM,
        hidden_dim=16,
        layers=1,
        semantic_tokens=6,
    )
    actions = torch.zeros((2, 7, ACTION_DIM), dtype=torch.float32)
    conditions = torch.zeros((2, KIMODO_V2_CONDITION_DIM), dtype=torch.float32)
    dim_mask = torch.ones((2, ACTION_DIM), dtype=torch.bool)
    durations = torch.tensor([0.2, 0.4], dtype=torch.float32)
    losses = masked_18d_objective(
        model,
        actions,
        conditions,
        dim_mask,
        durations,
        loss_weights={
            "flow": 0.0,
            "position": 0.0,
            "body": 0.0,
            "velocity": 0.0,
            "acceleration": 0.0,
            "planner_duration": 1.0,
            "planner_transition": 0.0,
        },
        generator=torch.Generator().manual_seed(17),
    )
    losses["total"].backward()

    assert torch.count_nonzero(model.duration_head.weight.grad) > 0
    assert model.transition_head.weight.grad is None
    assert model.transition_head.bias.grad is None


def test_transition_supervision_is_masked_by_default_and_rejects_all_end():
    contract = transition_supervision_contract(_episodes())
    assert contract["transition_supervised_episode_count"] == 0
    assert contract["transition_head_trainable"] is False
    assert contract["transition_inference_enabled"] is False

    all_end = _episodes()[:2]
    for index, episode in enumerate(all_end):
        episode["transition_id"] = "end"
        episode["transition_supervision_mask"] = True
        episode["transition_label_source"] = "verified_adjacent_sequence"
        episode["transition_sequence_id"] = "sequence-a"
        episode["transition_sequence_index"] = index
    with pytest.raises(ValueError, match="all verified transition labels are end"):
        transition_supervision_contract(all_end)


def test_random_init_config_refuses_generator_checkpoint(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        """{
          "schema_version": 1,
          "base_15d_checkpoint": "forbidden.pt",
          "qwen_checkpoint": "qwen.pt",
          "output_dir": "out",
          "motion_sources": [{"dataset_source": "x", "manifest": "x.jsonl"}]
        }""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="refuses generator checkpoint"):
        read_config(config_path)


def test_formal_cli_resolves_complete_multi_manifest_checkpoint_set(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text('{"clip_id":"first"}\n', encoding="utf-8")
    second.write_text('{"clip_id":"second"}\n', encoding="utf-8")

    def binding(path, name):
        return {
            "dataset_source": name,
            "manifest": str(path),
            "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "speaker_namespace": f"{name}-speaker",
            "source_group_namespace": f"{name}-source",
        }

    checkpoint = {
        "random_initialization": {"mode": RANDOM_INIT_MODE},
        "sources": {"motion_manifests": [binding(first, "a"), binding(second, "b")]},
    }
    resolved = resolve_bound_motion_sources(checkpoint)
    assert [row["dataset_source"] for row in resolved] == ["a", "b"]
    assert [row["manifest"] for row in resolved] == [first.resolve(), second.resolve()]

    reordered = resolve_bound_motion_sources(
        checkpoint, requested_manifests=[second, first]
    )
    assert [row["manifest"] for row in reordered] == [first.resolve(), second.resolve()]

    with pytest.raises(ValueError, match="complete random-init source set"):
        resolve_bound_motion_sources(checkpoint, requested_manifests=[first])

    second.write_text('{"clip_id":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        resolve_bound_motion_sources(checkpoint)
