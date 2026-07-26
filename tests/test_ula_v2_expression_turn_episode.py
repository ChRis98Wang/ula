from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json

import numpy as np
import pytest
import torch

from tools.human_motion_review.expression_turn_contract import (
    ACTION_PROTOCOL,
    AFFECT_PROTOCOL,
    ARC_PROTOCOL,
    ARTIFACT_KIND,
    CONTRACT_VERSION,
    CONTEXT_POLICY,
    evaluate_expression_turn,
)
from tools.human_motion_review.expression_turn_retarget_contract import (
    REQUIRED_18D_GATES,
    RETARGET_SEGMENT_REPRESENTATION,
)
from upper_body_skeleton.ula_training import (
    KIMODO_CONDITION_DIM,
    KIMODO_V2_CONDITION_DIM,
)
from upper_body_skeleton.ula_v2_18d_head import (
    ACTION_DIM,
    KIMODO_EMOTION_SLICE,
    STYLE_CONTROL_SLICE,
)
from upper_body_skeleton.retarget_v2_18d import JOINT_ORDER_18D
from upper_body_skeleton.ula_v2_conditioning import STYLE_FEATURE_NAMES
from upper_body_skeleton.ula_v2_18d_random_init import (
    build_native_duration_contract,
    build_random_18d_checkpoint,
    validate_formal_variable_length_episode,
)
from upper_body_skeleton.ula_v2_expression_turn_episode import (
    DYADIC_INTERACTION_PROMPT_PROFILE,
    DYADIC_PROMPT_TEXT_PROVENANCE,
    EXPRESSION_TURN_REPRESENTATION,
    FORMAL_ELIGIBILITY_MODE,
    FORMAL_EPISODE_CONTRACT,
    HUMAN_RETARGET_PHYSICAL_PROFILE,
    NATIVE_ROBOT_PHYSICAL_PROFILE,
    NATIVE_ROBOT_REQUIRED_18D_GATES,
    NATIVE_ROBOT_RETARGET_SEGMENT_REPRESENTATION,
    MOTION_FORM_PROMPT_PROFILE,
    PROMPT_TEXT_PROVENANCE,
    build_expression_turn_v8_condition_vectors,
    load_expression_turn_v8_episodes,
    validate_expression_turn_v8_episode,
)


def _hash(value, *, ascii_only=False):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=ascii_only,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii" if ascii_only else "utf-8")
    ).hexdigest()


def _envelope(protocol, review_id, reviewer_id):
    return {
        "protocol_version": protocol,
        "review_id": review_id,
        "reviewer_id": reviewer_id,
        "anonymous_video_sha256": "a" * 64,
        "context_level": 0,
        "audio_available": False,
        "label_metadata_exposed": False,
    }


def _review(*, semantic=False, expressive=False):
    arc = _envelope(ARC_PROTOCOL, "arc-r1", "arc-reviewer")
    arc.update(
        {
            "onset": {
                "status": "complete",
                "evidence_frame": 100,
                "basis": "natural_rest_or_low_motion",
            },
            "apex": {
                "status": "complete",
                "evidence_frame": 102,
                "basis": "distinct_motion_or_pose_peak",
            },
            "offset": {
                "status": "complete",
                "evidence_frame": 105,
                "basis": "natural_settle",
            },
        }
    )
    action = None
    if semantic:
        action = _envelope(ACTION_PROTOCOL, "action-r1", "action-reviewer")
        action.update(
            {
                "result": "observable_match",
                "observable_description": "Raises one forearm, holds, then returns.",
                "candidate_text_sha256": "b" * 64,
                "candidate_text_provenance": (
                    "independently_authored_robot_observable_text_v1"
                ),
            }
        )
    affect_reviews = []
    reviewer_ids = ("affect-reviewer-a", "affect-reviewer-b") if expressive else (
        "affect-reviewer-a",
    )
    for index, reviewer_id in enumerate(reviewer_ids):
        affect = _envelope(AFFECT_PROTOCOL, f"affect-r{index}", reviewer_id)
        affect.update(
            {
                "result": "observable" if expressive else "not_observable",
                "predicted_class": "happy" if expressive else None,
                "confidence": 0.9,
            }
        )
        affect_reviews.append(affect)
    record = {
        "artifact_kind": ARTIFACT_KIND,
        "contract_version": CONTRACT_VERSION,
        "clip_id": "turn-v8",
        "core_interval": {"start_frame": 100, "end_frame_exclusive": 106},
        "context_plan": {
            "policy": CONTEXT_POLICY,
            "same_source_only": True,
            "neighbor_crossing_allowed": False,
            "source_interval": {"start_frame": 0, "end_frame_exclusive": 200},
            "admissible_interval": {"start_frame": 90, "end_frame_exclusive": 120},
            "selected_level": 0,
            "levels": [
                {
                    "level": 0,
                    "start_frame": 100,
                    "end_frame_exclusive": 106,
                    "left_boundary_basis": "natural_low_motion_basin",
                    "right_boundary_basis": "natural_low_motion_basin",
                }
            ],
        },
        "physical_qc": {"passed": True},
        "motion_arc_review": arc,
        "action_semantic_review": action,
        "affect_reviews": affect_reviews,
        "official_category_conditioning_enabled": False,
        "emotion_conditioning_enabled": False,
        "emotion_supervision_mask": False,
    }
    return record


def _retarget_segment():
    payload = {
        "representation": RETARGET_SEGMENT_REPRESENTATION,
        "source_start_frame": 100,
        "source_end_frame_exclusive": 106,
        "source_frame_count": 6,
        "source_frame_coverage_sec": 0.2,
        "output_frame_count": 6,
        "output_sample_span_sec": 5 / 30,
        "output_frame_coverage_sec": 0.2,
        "fps": 30.0,
        "retimed": False,
        "cropped": False,
        "planner_duration_field": "output_sample_span_sec",
        "source_boundary_duration_field": "source_frame_coverage_sec",
        "legacy_quality_duration_sec_role": (
            "output_frame_coverage_compatibility_only_not_planner_target"
        ),
    }
    return payload | {"sha256": _hash(payload, ascii_only=True)}


def _episode(*, semantic=False, expressive=False):
    review = _review(semantic=semantic, expressive=expressive)
    report = evaluate_expression_turn(review)
    channels = {
        "motion": True,
        "semantic_conditioning": semantic,
        "expressive_conditioning": expressive,
    }
    prompt = (
        review["action_semantic_review"]["observable_description"] if semantic else None
    )
    condition = np.zeros(KIMODO_V2_CONDITION_DIM, dtype=np.float32)
    condition[STYLE_CONTROL_SLICE] = [0.1, -0.2, 0.3]
    if semantic:
        condition[KIMODO_CONDITION_DIM] = 1.0
    if expressive:
        condition[KIMODO_EMOTION_SLICE.start + 2] = 1.0
    review_hash = _hash(review)
    report_hash = _hash(report)
    episode = {
        "formal_episode_contract": FORMAL_EPISODE_CONTRACT,
        "expression_turn_contract_version": CONTRACT_VERSION,
        "clip_id": "turn-v8",
        "accepted_for_training": True,
        "eligibility_mode": FORMAL_ELIGIBILITY_MODE,
        "expression_turn_review_record": review,
        "expression_turn_review_record_sha256": review_hash,
        "qualification_report": report,
        "qualification_report_sha256": report_hash,
        "qualifications": deepcopy(report["qualifications"]),
        "training_qualification_tier": report["highest_qualification"],
        "training_channel_masks": channels,
        "semantic_supervision_masks": {
            "official_category": False,
            "robot_observable_motion_form": semantic,
            "communicative_intent": False,
            "prompt_text": semantic,
            "legacy_gesture": False,
        },
        "prompt": prompt,
        "prompt_semantics_profile": MOTION_FORM_PROMPT_PROFILE,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest() if prompt else None,
        "prompt_text_provenance": PROMPT_TEXT_PROVENANCE if semantic else None,
        "prompt_review_id": "action-r1" if semantic else None,
        "emotion_id": "happy" if expressive else None,
        "emotion_source": (
            "independent_blind_affect_consensus_or_adjudication_v1"
            if expressive
            else None
        ),
        "emotion_supervision_mask": expressive,
        "emotion_conditioning_mask": expressive,
        "affect_observable_supervision_mask": expressive,
        "official_emotion_conditioning_enabled": False,
        "official_category_conditioning_enabled": False,
        "behavior_supervision_mask": False,
        "behavior_id": None,
        "actions": np.zeros((6, ACTION_DIM), dtype=np.float32),
        "action_dim_mask": np.ones(ACTION_DIM, dtype=np.bool_),
        "condition": condition,
        "fps": 30.0,
        "training_segment": {
            "representation": EXPRESSION_TURN_REPRESENTATION,
            "start_frame": 100,
            "end_frame_exclusive": 106,
            "frame_count": 6,
            "fixed_window_sec": None,
            "cropped": False,
            "duration_policy": "natural_rest_to_natural_rest_no_fixed_or_max_duration",
        },
        "retarget_segment": _retarget_segment(),
        "physical_evidence_profile": HUMAN_RETARGET_PHYSICAL_PROFILE,
        "quality_gate": {gate: True for gate in REQUIRED_18D_GATES},
        "retarget_qc_passed": True,
        "quality_source_window_frames": 6,
        "quality_output_frame_count": 6,
        "source_clip_id": "source-a",
        "speaker_key": "beat2:speaker-a",
        "source_group_key": "beat2:source-a",
        "dataset_source": "beat2_expression_turn_v8",
    }
    for index, field in enumerate(
        (
            "expression_turn_contract_sha256",
            "source_inventory_manifest_sha256",
            "split_assignment_manifest_sha256",
            "inventory_record_sha256",
            "upstream_inventory_record_sha256",
            "selected_record_sha256",
            "retarget_input_manifest_sha256",
            "retarget_quality_record_sha256",
            "trajectory_sha256",
            "source_sha256",
        )
    ):
        episode[field] = f"{index:x}" * 64
    episode["training_admission"] = {
        "contract": FORMAL_EPISODE_CONTRACT,
        "expression_turn_review_record_sha256": review_hash,
        "qualification_report_sha256": report_hash,
        "retarget_quality_record_sha256": episode["retarget_quality_record_sha256"],
        "training_qualification_tier": report["highest_qualification"],
        "training_channel_masks": channels,
    }
    return episode


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


def _v8_episode_set():
    episodes = []
    tier_flags = ((False, False), (True, False), (True, True))
    for index in range(6):
        semantic, expressive = tier_flags[index % len(tier_flags)]
        episode = _episode(semantic=semantic, expressive=expressive)
        speaker_index = index // 2
        local_index = index % 2
        frames = 6 + local_index
        clip_id = f"turn-v8-{index}"
        review = episode["expression_turn_review_record"]
        review["clip_id"] = clip_id
        review["core_interval"]["end_frame_exclusive"] = 100 + frames
        review["context_plan"]["levels"][0]["end_frame_exclusive"] = 100 + frames
        report = evaluate_expression_turn(review)
        review_hash = _hash(review)
        report_hash = _hash(report)
        episode.update(
            {
                "clip_id": clip_id,
                "expression_turn_review_record_sha256": review_hash,
                "qualification_report": report,
                "qualification_report_sha256": report_hash,
                "qualifications": deepcopy(report["qualifications"]),
                "training_qualification_tier": report["highest_qualification"],
                "actions": np.full(
                    (frames, ACTION_DIM), 0.01 * (index + 1), dtype=np.float32
                ),
                "quality_source_window_frames": frames,
                "quality_output_frame_count": frames,
                "source_clip_id": f"source-{index}",
                "speaker_key": f"beat2:speaker-{speaker_index}",
                "source_group_key": f"beat2:source-{index}",
            }
        )
        episode["training_segment"].update(
            end_frame_exclusive=100 + frames,
            frame_count=frames,
        )
        retarget = {
            "representation": RETARGET_SEGMENT_REPRESENTATION,
            "source_start_frame": 100,
            "source_end_frame_exclusive": 100 + frames,
            "source_frame_count": frames,
            "source_frame_coverage_sec": frames / 30,
            "output_frame_count": frames,
            "output_sample_span_sec": (frames - 1) / 30,
            "output_frame_coverage_sec": frames / 30,
            "fps": 30.0,
            "retimed": False,
            "cropped": False,
            "planner_duration_field": "output_sample_span_sec",
            "source_boundary_duration_field": "source_frame_coverage_sec",
            "legacy_quality_duration_sec_role": (
                "output_frame_coverage_compatibility_only_not_planner_target"
            ),
        }
        episode["retarget_segment"] = retarget | {
            "sha256": _hash(retarget, ascii_only=True)
        }
        episode["training_admission"].update(
            expression_turn_review_record_sha256=review_hash,
            qualification_report_sha256=report_hash,
            training_qualification_tier=report["highest_qualification"],
        )
        episodes.append(episode)
    return episodes


@pytest.mark.parametrize(
    ("semantic", "expressive", "tier"),
    (
        (False, False, "base_motion"),
        (True, False, "semantic_conditioning"),
        (True, True, "expressive_conditioning"),
    ),
)
def test_v8_episode_opens_only_review_qualified_channels(semantic, expressive, tier):
    episode = _episode(semantic=semantic, expressive=expressive)
    report = validate_expression_turn_v8_episode(
        episode, require_attached_condition=True
    )
    validate_formal_variable_length_episode(
        episode, require_attached_condition=True
    )
    assert report["highest_qualification"] == tier
    assert report["training_channel_masks"] == {
        "motion": True,
        "semantic_conditioning": semantic,
        "expressive_conditioning": expressive,
    }


def test_v8_episode_recomputes_review_and_rejects_forged_tier():
    episode = _episode()
    episode["qualifications"]["semantic_conditioning"]["eligible"] = True
    with pytest.raises(ValueError, match="three-tier qualification"):
        validate_expression_turn_v8_episode(episode)


def test_v8_episode_rejects_legacy_semantic_event_and_unqualified_qwen():
    legacy = _episode()
    legacy["semantic_event"] = {"category": "deictic"}
    with pytest.raises(ValueError, match="legacy semantic-event"):
        validate_expression_turn_v8_episode(legacy)

    leaked = _episode()
    leaked["condition"][KIMODO_CONDITION_DIM] = 1.0
    with pytest.raises(ValueError, match="unqualified condition channel"):
        validate_expression_turn_v8_episode(leaked)


def test_v8_episode_rejects_fixed_crop_or_review_interval_mismatch():
    fixed = _episode()
    fixed["training_segment"]["fixed_window_sec"] = 6.0
    with pytest.raises(ValueError, match="fixed duration"):
        validate_expression_turn_v8_episode(fixed)

    shifted = _episode()
    shifted["training_segment"]["start_frame"] = 99
    shifted["training_segment"]["frame_count"] = 7
    with pytest.raises(ValueError, match="reviewed anonymous video"):
        validate_expression_turn_v8_episode(shifted)


def test_v8_native_robot_profile_preserves_full_asset_and_requires_collision():
    episode = _episode(semantic=True)
    payload = {
        key: value
        for key, value in episode["retarget_segment"].items()
        if key != "sha256"
    }
    payload["representation"] = NATIVE_ROBOT_RETARGET_SEGMENT_REPRESENTATION
    episode["retarget_segment"] = payload | {
        "sha256": _hash(payload, ascii_only=True)
    }
    episode["physical_evidence_profile"] = NATIVE_ROBOT_PHYSICAL_PROFILE
    episode["quality_gate"] = {
        gate: True for gate in NATIVE_ROBOT_REQUIRED_18D_GATES
    }

    report = validate_expression_turn_v8_episode(
        episode, require_attached_condition=True
    )
    assert report["frame_count"] == report["source_frame_count"] == 6

    episode["quality_gate"]["collision_pass"] = False
    with pytest.raises(ValueError, match="physical quality gates"):
        validate_expression_turn_v8_episode(episode, require_attached_condition=True)


def test_v8_dyadic_prompt_enables_only_independently_observed_intent():
    episode = _episode(semantic=True)
    review = episode["expression_turn_review_record"]
    action = review["action_semantic_review"]
    prompt = (
        "The robot opens both forearms toward its partner to invite a response, "
        "then lowers them after the partner reacts."
    )
    action.update(
        {
            "candidate_text": prompt,
            "candidate_text_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "robot_observable_motion_description": (
                "The robot opens both forearms toward the other actor, holds, then lowers them."
            ),
            "communicative_intent_result": "observable",
            "communicative_intent_description": "Invites the partner to respond.",
            "robot_prompt": prompt,
            "robot_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        }
    )
    report = evaluate_expression_turn(review)
    review_hash = _hash(review)
    report_hash = _hash(report)
    episode.update(
        {
            "expression_turn_review_record_sha256": review_hash,
            "qualification_report": report,
            "qualification_report_sha256": report_hash,
            "qualifications": deepcopy(report["qualifications"]),
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt_text_provenance": DYADIC_PROMPT_TEXT_PROVENANCE,
            "prompt_semantics_profile": DYADIC_INTERACTION_PROMPT_PROFILE,
            "semantic_supervision_masks": {
                "official_category": False,
                "robot_observable_motion_form": True,
                "communicative_intent": True,
                "prompt_text": True,
                "legacy_gesture": False,
            },
        }
    )
    episode["training_admission"].update(
        expression_turn_review_record_sha256=review_hash,
        qualification_report_sha256=report_hash,
    )

    result = validate_expression_turn_v8_episode(
        episode, require_attached_condition=True
    )
    assert result["prompt_semantics_profile"] == DYADIC_INTERACTION_PROMPT_PROFILE

    action["communicative_intent_result"] = "ambiguous"
    episode["expression_turn_review_record_sha256"] = _hash(review)
    episode["training_admission"]["expression_turn_review_record_sha256"] = _hash(review)
    with pytest.raises(ValueError, match="dyadic prompt lacks"):
        validate_expression_turn_v8_episode(episode, require_attached_condition=True)


def test_v8_condition_vectors_open_only_blind_qualified_channels():
    episodes = _v8_episode_set()
    text_latents = {
        episode["clip_id"]: np.arange(1, 129, dtype=np.float32)
        for episode in episodes
        if episode["training_channel_masks"]["semantic_conditioning"]
    }
    style_contract = {
        "contract_version": 1,
        "feature_names": list(STYLE_FEATURE_NAMES),
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "clip": 5.0,
    }
    native_lengths = [len(episode["actions"]) for episode in episodes]

    conditions, features, controls = build_expression_turn_v8_condition_vectors(
        episodes,
        text_latents=text_latents,
        style_contract=style_contract,
    )

    assert conditions.shape == (6, KIMODO_V2_CONDITION_DIM)
    assert features.shape == controls.shape == (6, 3)
    assert [len(episode["actions"]) for episode in episodes] == native_lengths
    for episode, condition in zip(episodes, conditions, strict=True):
        semantic = episode["training_channel_masks"]["semantic_conditioning"]
        expressive = episode["training_channel_masks"]["expressive_conditioning"]
        assert bool(np.linalg.norm(condition[KIMODO_CONDITION_DIM:])) is semantic
        assert np.isclose(
            np.linalg.norm(condition[KIMODO_CONDITION_DIM:]), 1.0 if semantic else 0.0
        )
        assert int(np.count_nonzero(condition[KIMODO_EMOTION_SLICE])) == int(expressive)


def test_v8_manifest_loader_binds_full_native_csv_and_rejects_hash_change(tmp_path):
    episode = _episode(semantic=True)
    actions = np.arange(6 * ACTION_DIM, dtype=np.float32).reshape(6, ACTION_DIM) / 1000
    csv_path = tmp_path / "native_full_arc.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_sec", *JOINT_ORDER_18D])
        for index, row in enumerate(actions):
            writer.writerow([index / 30, *row.tolist()])
    trajectory_sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    record = deepcopy(episode)
    for field in ("actions", "action_dim_mask", "condition", "duration_sec"):
        record.pop(field, None)
    record["trajectory_sha256"] = trajectory_sha256
    record["motion_18d"] = {
        "state": "passed",
        "safe_csv": str(csv_path),
        "safe_csv_sha256": trajectory_sha256,
        "frames": 6,
        "csv_rows": 6,
        "fps": 30.0,
        "source_window_frames": 6,
        "physical_evidence_profile": record["physical_evidence_profile"],
        "quality_gate": deepcopy(record["quality_gate"]),
        "retarget_segment": deepcopy(record["retarget_segment"]),
    }
    manifest = tmp_path / "train_ready.jsonl"
    manifest.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")

    loaded = load_expression_turn_v8_episodes(manifest)
    assert len(loaded) == 1
    assert loaded[0]["actions"].shape == (6, ACTION_DIM)
    assert np.allclose(loaded[0]["actions"], actions)
    assert loaded[0]["duration_sec"] == pytest.approx(5 / 30)
    assert loaded[0]["training_segment"]["fixed_window_sec"] is None

    record["trajectory_sha256"] = "0" * 64
    manifest.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="top-level trajectory SHA256"):
        load_expression_turn_v8_episodes(manifest)


def test_v8_random_checkpoint_preserves_three_tier_conditioning_contract(tmp_path):
    episodes = _v8_episode_set()
    checkpoint, _, report = build_random_18d_checkpoint(
        episodes,
        qwen_checkpoint=_qwen_checkpoint(tmp_path / "qwen.pt"),
        source_provenance=[
            {
                "dataset_source": "beat2_expression_turn_v8",
                "manifest_sha256": "a" * 64,
            }
        ],
        seed=17,
        fractions={"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3},
        hidden_dim=16,
        layers=1,
        semantic_tokens=7,
        length_buckets=(8,),
    )

    assert checkpoint["formal_episode_contract"] == FORMAL_EPISODE_CONTRACT
    contracts = checkpoint["v2_contracts"]
    assert contracts["formal_episode_contract"] == FORMAL_EPISODE_CONTRACT
    assert contracts["batching"]["source_representation"] == (
        EXPRESSION_TURN_REPRESENTATION
    )
    text = contracts["text_motion_latent"]
    assert text["conditioned_episode_count"] == 4
    assert text["masked_episode_count"] == 2
    assert text["unique_text_count"] == 1
    assert text["qualification_required"] == "semantic_conditioning"
    assert text["prompt_semantics_profile_counts"] == {
        MOTION_FORM_PROMPT_PROFILE: 4
    }
    assert text["communicative_intent_conditioned_count"] == 0
    supervision = checkpoint["semantic_supervision_contract"]
    assert supervision["qualification_tier_counts"] == {
        "base_motion": 2,
        "semantic_conditioning": 2,
        "expressive_conditioning": 2,
    }
    assert supervision["semantic_conditioned_count"] == 4
    assert supervision["expressive_conditioned_count"] == 2
    assert supervision["prompt_semantics_profile_counts"] == {
        MOTION_FORM_PROMPT_PROFILE: 4
    }
    assert supervision["communicative_intent_conditioned_count"] == 0
    assert supervision["official_category_conditioning_enabled"] is False
    assert supervision["official_emotion_conditioning_enabled"] is False
    assert supervision["legacy_semantic_event_forbidden"] is True
    assert report["qualification_tier_counts"] == supervision[
        "qualification_tier_counts"
    ]


def test_v8_duration_supervision_rejects_partial_expression_arc():
    episode = _episode(semantic=True)
    episode["expression_turn_review_record"]["motion_arc_review"]["offset"][
        "status"
    ] = "incomplete"
    with pytest.raises(ValueError, match="complete onset/apex/offset arc"):
        build_native_duration_contract([episode])
