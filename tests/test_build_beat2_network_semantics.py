import json
from pathlib import Path

import pytest

from tools.human_motion_collection.build_beat2_network_semantics import (
    AUTO_BEHAVIOR_IDS,
    KIMODO_BEHAVIOR_IDS,
    KIMODO_EMOTION_IDS,
    UNRESOLVED_EMOTION_SOURCE,
    apply_human_review,
    build_automatic_record,
    build_semantics,
    main,
    validate_network_record,
)


def annotation(
    clip_id: str = "12_zhao_2_101_101_f000000-000180",
    *,
    speech_context: str = "我现在非常开心和生气",
    laterality: str = "both",
    amplitude: str = "large",
    pace: str = "quick",
    continuity: str = "continuous",
    head_motion: str = "clear",
    head_pattern: str = "none",
) -> dict:
    return {
        "task_id": clip_id,
        "source_clip_id": "12_zhao_2_101_101",
        "speaker_key": "12_zhao",
        "canonical_action": "robot_observable_upper_body_motion",
        "source_speech_context": speech_context,
        "audio_path": "/data/happy_angry.wav",
        "canonical_prompt": {
            "en": "Move both arms broadly, turn the head, and keep a quick pace.",
            "zh": "大幅移动双臂并转动头部，保持较快节奏。",
        },
        "observable_features": {
            "arm": {
                "laterality": laterality,
                "amplitude": amplitude,
                "continuity": continuity,
            },
            "overall_motion": {"pace": pace},
            "head_motion": head_motion,
            "torso_motion": "subtle",
            "head": {"repeated_pattern": {"pattern": head_pattern}},
        },
        "trajectory_path": "/data/motion.csv",
        "trajectory_sha256": "abc",
    }


def confirmed_review(clip_id: str, emotion: str = "happy") -> dict:
    return {
        "clip_id": clip_id,
        "decision": "confirmed",
        "reviewer_kind": "human",
        "reviewer_id": "reviewer-01",
        "reviewed_at": "2026-07-23T10:00:00+08:00",
        "behavior_id": "Behavior.InteractPresence",
        "emotion_id": emotion,
        "emotion_confidence": 0.9,
        "canonical_prompt": {
            "en": f"Raise both arms and turn the head with {emotion} emotion.",
            "zh": "抬起双臂并转动头部，表现出经过人工确认的情绪。",
        },
        "notes": "Visible in the review video.",
    }


def behavior_confirmed_review(clip_id: str) -> dict:
    return {
        "clip_id": clip_id,
        "decision": "behavior_confirmed",
        "reviewer_kind": "human",
        "reviewer_id": "reviewer-behavior-01",
        "reviewed_at": "2026-07-23T10:00:00+08:00",
        "behavior_id": "Behavior.InteractPresence",
        "canonical_prompt": {
            "en": "Raise both arms broadly and turn the head.",
            "zh": "大幅抬起双臂并转动头部。",
        },
        "notes": "Behavior is visible; affect remains ambiguous.",
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_network_ontology_has_exactly_27_behaviors_and_6_emotions():
    assert len(KIMODO_BEHAVIOR_IDS) == 27
    assert len(set(KIMODO_BEHAVIOR_IDS)) == 27
    assert KIMODO_EMOTION_IDS == (
        "neutral",
        "sad",
        "happy",
        "angry",
        "surprise",
        "fear",
    )


def test_automatic_record_is_fail_closed_and_ignores_speech_and_filename_emotion():
    record = build_automatic_record(annotation())

    assert record["behavior_id"] in AUTO_BEHAVIOR_IDS
    assert record["behavior_id"] == "Behavior.InteractPresence"
    assert record["behavior_review_status"] == "candidate_unreviewed"
    assert record["behavior_supervision_mask"] is False
    assert record["behavior_training_eligibility"] == "blocked_unreviewed_behavior"
    assert record["emotion_id"] is None
    assert record["emotion_review_status"] == "unresolved"
    assert record["emotion_source"] == UNRESOLVED_EMOTION_SOURCE
    assert record["emotion_confidence"] == 0.0
    assert record["emotion_supervision_mask"] is False
    assert record["observed_affect"]["label"] is None
    assert record["motion_style"] == "energetic"
    assert record["canonical_action"] == (
        "bilateral_large_continuous_arm_motion_with_clear_multi_axis_head_motion_"
        "with_subtle_multi_axis_torso_motion"
    )
    assert record["canonical_action_source"] == "trajectory_only_18d_observable_features"
    assert record["source_window_start_frame"] == 0
    assert record["source_window_end_frame_exclusive"] == 180
    assert record["motion_style_training_eligibility"] == "pending_adjudication"
    assert record["emotion_training_eligibility"] == "blocked_unresolved_emotion"
    assert record["source_group_key"] == "beat2/12_zhao/12_zhao_2_101_101"
    assert "source_speech_context" not in record
    assert "audio_path" not in record
    assert validate_network_record(record) == []


def test_concrete_input_action_is_preserved_without_folding_emotion_into_action():
    concrete = annotation()
    concrete["canonical_action"] = "bilateral_conversational_arm_sweep"
    preserved = build_automatic_record(concrete)
    assert preserved["canonical_action"] == "bilateral_conversational_arm_sweep"
    assert preserved["canonical_action_source"] == "preserved_input_canonical_action"

    emotion_only = annotation()
    emotion_only["canonical_action"] = "happy"
    generated = build_automatic_record(emotion_only)
    assert generated["canonical_action_source"] == "trajectory_only_18d_observable_features"
    assert "happy" not in generated["canonical_action"]


def test_multiple_windows_from_one_source_keep_distinct_window_provenance():
    first = annotation("12_zhao_2_101_101_f000000-000180")
    second = annotation("12_zhao_2_101_101_f000180-000360")
    records, _, _ = build_semantics([second, first])

    assert [record["clip_id"] for record in records] == [
        "12_zhao_2_101_101_f000000-000180",
        "12_zhao_2_101_101_f000180-000360",
    ]
    assert {record["source_group_key"] for record in records} == {
        "beat2/12_zhao/12_zhao_2_101_101"
    }
    assert [record["source_window_start_frame"] for record in records] == [0, 180]


def test_namespaced_semantic_event_id_is_bound_to_explicit_source():
    item = annotation(
        "beat_english_v2.0.0__12_zhao_2_101_101_sem0004_f000010-000190"
    )
    item["source_window_start_frame"] = 10
    item["source_window_end_frame_exclusive"] = 190

    record = build_automatic_record(item)

    assert record["source_window_start_frame"] == 10
    assert record["source_window_end_frame_exclusive"] == 190
    assert record["source_group_key"] == "beat2/12_zhao/12_zhao_2_101_101"


def test_explicit_window_conflicting_with_task_id_is_rejected():
    bad = annotation()
    bad["source_window_start_frame"] = 1
    bad["source_window_end_frame_exclusive"] = 180
    with pytest.raises(ValueError, match="explicit source window conflicts with task_id"):
        build_automatic_record(bad)


@pytest.mark.parametrize(
    ("kwargs", "behavior_id"),
    [
        (
            {
                "laterality": "none",
                "amplitude": "very_small",
                "head_motion": "minimal",
            },
            "Behavior.IdleQuiet",
        ),
        (
            {"laterality": "none", "amplitude": "very_small"},
            "Behavior.IdleAttentive",
        ),
        (
            {
                "laterality": "none",
                "amplitude": "very_small",
                "head_pattern": "repeated_pitch_nods",
            },
            "Behavior.ActiveListening",
        ),
    ],
)
def test_automatic_behavior_stays_with_conservative_conversation_ids(kwargs, behavior_id):
    record = build_automatic_record(annotation(**kwargs))
    assert record["behavior_id"] == behavior_id
    assert record["behavior_id"] in AUTO_BEHAVIOR_IDS


def test_human_confirmation_enables_emotion_supervision_only_after_strict_checks():
    automatic = build_automatic_record(annotation())
    reviewed = apply_human_review(
        automatic, confirmed_review(automatic["clip_id"], emotion="surprise")
    )

    assert reviewed["behavior_id"] in KIMODO_BEHAVIOR_IDS
    assert reviewed["behavior_review_status"] == "human_confirmed"
    assert reviewed["behavior_supervision_mask"] is True
    assert reviewed["emotion_id"] == "surprise"
    assert reviewed["emotion_review_status"] == "human_confirmed"
    assert reviewed["emotion_supervision_mask"] is True
    assert reviewed["network_semantic_supervision_ready"] is True
    assert reviewed["observed_affect"]["label"] == "surprise"
    assert validate_network_record(reviewed) == []


def test_behavior_can_be_confirmed_while_emotion_remains_explicitly_unresolved():
    automatic = build_automatic_record(annotation())
    reviewed = apply_human_review(
        automatic, behavior_confirmed_review(automatic["clip_id"])
    )

    assert reviewed["behavior_review_status"] == "human_confirmed"
    assert reviewed["behavior_supervision_mask"] is True
    assert reviewed["behavior_training_eligibility"] == "pending_adjudication"
    assert reviewed["emotion_id"] is None
    assert reviewed["emotion_review_status"] == "unresolved"
    assert reviewed["emotion_supervision_mask"] is False
    assert reviewed["emotion_training_eligibility"] == "blocked_unresolved_emotion"
    assert reviewed["network_semantic_supervision_ready"] is False
    assert "happy" not in reviewed["canonical_prompt"]["en"].lower()
    assert validate_network_record(reviewed) == []


def test_agent_cannot_be_declared_as_human_semantic_confirmation():
    automatic = build_automatic_record(annotation())
    review = behavior_confirmed_review(automatic["clip_id"])
    review["reviewer_kind"] = "agent"

    with pytest.raises(ValueError, match="requires reviewer_kind=human"):
        apply_human_review(automatic, review)


@pytest.mark.parametrize(
    ("mutation", "error_fragment"),
    [
        (
            lambda review: review["canonical_prompt"].update(
                {"en": "Raise both arms with a surprised expression."}
            ),
            "exact emotion word 'happy'",
        ),
        (
            lambda review: review["canonical_prompt"].update(
                {"en": "Perform an action with happy emotion."}
            ),
            "concrete robot-visible action",
        ),
        (
            lambda review: review["canonical_prompt"].update(
                {"zh": "表达情绪。"}
            ),
            "canonical_prompt.zh",
        ),
        (
            lambda review: review.update({"emotion_id": "excited"}),
            "unknown emotion_id",
        ),
        (
            lambda review: review.update({"behavior_id": "Behavior.NotReal"}),
            "unknown behavior_id",
        ),
    ],
)
def test_invalid_human_confirmation_is_rejected(mutation, error_fragment):
    automatic = build_automatic_record(annotation())
    review = confirmed_review(automatic["clip_id"])
    mutation(review)

    with pytest.raises(ValueError, match=error_fragment):
        apply_human_review(automatic, review)


def test_rejection_never_enables_emotion_supervision():
    automatic = build_automatic_record(annotation())
    review = {
        "clip_id": automatic["clip_id"],
        "decision": "rejected",
        "reviewer_kind": "human",
        "reviewer_id": "reviewer-02",
        "reviewed_at": "2026-07-23T11:00:00+08:00",
        "notes": "Affect is ambiguous.",
    }
    reviewed = apply_human_review(automatic, review)

    assert reviewed["emotion_id"] is None
    assert reviewed["emotion_review_status"] == "rejected"
    assert reviewed["behavior_review_status"] == "rejected"
    assert reviewed["behavior_supervision_mask"] is False
    assert reviewed["emotion_supervision_mask"] is False
    assert validate_network_record(reviewed) == []


def test_build_semantics_outputs_only_unresolved_records_to_review_queue():
    first = annotation("clip_a")
    second = annotation("clip_b")
    review = confirmed_review("clip_b", "neutral")

    records, queue, summary = build_semantics([first, second], [review])

    assert [record["clip_id"] for record in records] == ["clip_a", "clip_b"]
    assert [record["clip_id"] for record in queue] == ["clip_a"]
    assert summary["emotion_supervised_records"] == 1
    assert summary["human_review_queue_records"] == 1
    assert summary["transcript_or_audio_metadata_used_for_labels"] is False
    contract = queue[0]["review_contract"]
    assert contract["allowed_emotion_ids"] == [
        "neutral",
        "sad",
        "happy",
        "angry",
        "surprise",
        "fear",
    ]
    assert "transcript" in contract["emotion_policy"]


def test_cli_writes_network_queue_and_supervised_partitions(tmp_path):
    input_path = tmp_path / "drafts.jsonl"
    reviews_path = tmp_path / "reviews.jsonl"
    output = tmp_path / "semantic"
    write_jsonl(input_path, [annotation("clip_a"), annotation("clip_b")])
    write_jsonl(reviews_path, [confirmed_review("clip_b", "fear")])

    assert main(
        [
            "--input-annotations",
            str(input_path),
            "--human-reviews",
            str(reviews_path),
            "--output-dir",
            str(output),
        ]
    ) == 0

    network = [
        json.loads(line)
        for line in (output / "network_semantics.jsonl").read_text().splitlines()
    ]
    queue = [
        json.loads(line)
        for line in (output / "human_review_queue.jsonl").read_text().splitlines()
    ]
    supervised = [
        json.loads(line)
        for line in (output / "emotion_supervised.jsonl").read_text().splitlines()
    ]
    summary = json.loads((output / "summary.json").read_text())
    assert len(network) == 2
    assert [record["clip_id"] for record in queue] == ["clip_a"]
    assert [record["clip_id"] for record in supervised] == ["clip_b"]
    assert summary["output_records"] == 2
    assert summary["validation_errors"] == 0
