import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.human_motion_review.build_observable_intent_review_v1 import build_outputs
from tools.human_motion_review.build_rendered_intent_blind_bundle_v1 import (
    build_anonymous_bundle,
)
from upper_body_skeleton.robot_observable_intents import (
    DEFAULT_ONTOLOGY_PATH,
    build_observable_intent_one_hot,
    intent_definition,
    load_observable_intent_ontology,
    observable_intent_ids,
    ontology_sha256,
    validate_observable_intent_annotation,
)
from upper_body_skeleton.ula_v2_observable_intent_v9 import (
    KIMODO_V2_CONDITION_DIM,
    LEGACY_KIMODO_FAMILY_SLICE,
    OBSERVABLE_INTENT_SLICE_V9,
    apply_observable_intent_overlay_v9,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _source_record(sample_id: str, video_sha256: str) -> dict:
    return {
        "sample_id": sample_id,
        "clip_id": f"source_{sample_id}_GreetingOwner01",
        "prompt": "The robot moves one arm while turning its head.",
        "prompt_text_provenance": "independent_blind_action_observable_description_v1",
        "source_transcript": "hello",
        "expression_turn_review_record": {
            "action_semantic_review": {
                "anonymous_video_sha256": video_sha256,
            }
        },
    }


def _decision(
    sample_id: str,
    video_sha256: str,
    reviewer: str,
    *,
    intent_id: str = "wave_to_person",
    confidence: float = 0.9,
) -> dict:
    return {
        "protocol_version": "robot_observable_intent_blind_video_v1",
        "sample_id": sample_id,
        "video_sha256": video_sha256,
        "intent_ontology_sha256": ontology_sha256(DEFAULT_ONTOLOGY_PATH),
        "label_metadata_exposed": False,
        "audio_available": False,
        "result": "observable",
        "observable_intent_id": intent_id,
        "confidence": confidence,
        "hard_negative_checked": True,
        "hard_negative_notes": "Repeated lateral cycles, not a held raise or inward beckon.",
        "reviewer_id": reviewer,
        "review_id": f"{reviewer}/{sample_id}",
        "notes": "Visible in the anonymous full-robot video.",
    }


def test_ontology_has_stable_transparent_27_slot_contract():
    ontology = load_observable_intent_ontology()
    ids = observable_intent_ids(ontology)

    assert len(ids) == 27
    assert len(set(ids)) == 27
    assert tuple(intent["slot"] for intent in ontology["intents"]) == tuple(range(27))
    assert "wave_to_person" in ids
    assert "beckon_come_here" in ids
    assert "raise_hand_get_attention" in ids
    assert "salute" in ids
    assert "offer_fist_bump" in ids
    assert "dance_sway" not in ids
    assert "greeting" not in ids
    assert "farewell" not in ids


def test_greeting_and_farewell_are_context_roles_of_the_same_wave_motion():
    ontology = load_observable_intent_ontology()
    wave = intent_definition("wave_to_person", ontology)

    assert wave["allowed_pragmatic_roles"] == [
        "greeting",
        "farewell",
        "acknowledgement",
    ]
    assert {"raise_hand_get_attention", "beckon_come_here"}.issubset(
        wave["hard_negatives"]
    )


def test_explicit_intents_have_distinct_one_hot_vectors():
    ontology = load_observable_intent_ontology()
    wave = build_observable_intent_one_hot("wave_to_person", ontology)
    beckon = build_observable_intent_one_hot("beckon_come_here", ontology)
    attention = build_observable_intent_one_hot(
        "raise_hand_get_attention", ontology
    )

    assert wave.dtype == np.float32
    assert wave.shape == (27,)
    assert np.sum(wave) == 1.0
    assert not np.array_equal(wave, beckon)
    assert not np.array_equal(wave, attention)
    with pytest.raises(ValueError, match="unknown observable_intent_id"):
        build_observable_intent_one_hot("hello_from_filename", ontology)


def test_masked_annotation_cannot_smuggle_a_text_inferred_intent():
    ontology = load_observable_intent_ontology()
    record = {
        "intent_ontology_id": ontology["ontology_id"],
        "intent_ontology_sha256": ontology_sha256(DEFAULT_ONTOLOGY_PATH),
        "observable_intent_id": "wave_to_person",
        "intent_review_status": "pending_review",
        "intent_supervision_mask": False,
        "intent_conditioning_mask": False,
        "pragmatic_role": None,
        "pragmatic_role_supervision_mask": False,
    }
    with pytest.raises(ValueError, match="must not carry observable_intent_id"):
        validate_observable_intent_annotation(
            record,
            ontology,
            expected_ontology_sha256=ontology_sha256(DEFAULT_ONTOLOGY_PATH),
        )


def test_migration_is_fail_closed_even_when_filename_and_transcript_say_hello(
    tmp_path: Path,
):
    sample_id = "anonymous_sample_001"
    video_bytes = b"fake anonymous robot video"
    video_sha = _sha256(video_bytes)
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / f"{sample_id}.mp4").write_bytes(video_bytes)
    source_manifest = tmp_path / "source.jsonl"
    _write_jsonl(source_manifest, [_source_record(sample_id, video_sha)])
    output_dir = tmp_path / "out"

    summary = build_outputs(
        input_manifest=source_manifest,
        video_dir=video_dir,
        ontology_path=DEFAULT_ONTOLOGY_PATH,
        output_dir=output_dir,
        decision_paths=[],
    )

    assert summary["source_record_count"] == 1
    assert summary["pending_count"] == 1
    assert summary["train_ready_count"] == 0
    assert summary["automatic_intent_labels_emitted"] == 0
    pending = json.loads((output_dir / "intent_review_pending.jsonl").read_text())
    assert pending["observable_intent_id"] is None
    assert pending["intent_supervision_mask"] is False
    serialized = json.dumps(pending, ensure_ascii=False)
    assert "GreetingOwner01" not in serialized
    assert "hello" not in serialized


def test_two_independent_matching_reviews_create_v9_intent_overlay(tmp_path: Path):
    sample_id = "anonymous_sample_002"
    video_bytes = b"fake wave video"
    video_sha = _sha256(video_bytes)
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / f"{sample_id}.mp4").write_bytes(video_bytes)
    source_manifest = tmp_path / "source.jsonl"
    _write_jsonl(source_manifest, [_source_record(sample_id, video_sha)])
    review_a = tmp_path / "review_a.jsonl"
    review_b = tmp_path / "review_b.jsonl"
    _write_jsonl(review_a, [_decision(sample_id, video_sha, "reviewer_a")])
    _write_jsonl(review_b, [_decision(sample_id, video_sha, "reviewer_b")])
    output_dir = tmp_path / "out"

    summary = build_outputs(
        input_manifest=source_manifest,
        video_dir=video_dir,
        ontology_path=DEFAULT_ONTOLOGY_PATH,
        output_dir=output_dir,
        decision_paths=[review_a, review_b],
    )

    assert summary["train_ready_count"] == 1
    assert summary["pending_count"] == 0
    overlay = json.loads((output_dir / "intent_train_ready.jsonl").read_text())
    assert overlay["observable_intent_id"] == "wave_to_person"
    assert overlay["intent_supervision_mask"] is True
    assert overlay["intent_review_status"] == "independent_blind_consensus"
    assert sum(overlay["intent_one_hot"]) == 1.0
    assert "behavior_id" not in overlay
    assert overlay["pragmatic_role"] is None
    assert overlay["pragmatic_role_supervision_mask"] is False
    assert overlay["network_condition_contract"] == (
        "ula_v2_18d_observable_intent_v9"
    )
    assert overlay["primary_semantic_channel"] == "observable_intent_one_hot"
    assert overlay["semantic_prompt"].startswith(
        "Raise one hand and wave it side to side toward the person."
    )
    assert "Motion realization:" in overlay["semantic_prompt"]

    base_condition = np.ones(KIMODO_V2_CONDITION_DIM, dtype=np.float32)
    condition = apply_observable_intent_overlay_v9(base_condition, overlay)
    assert np.array_equal(
        condition[OBSERVABLE_INTENT_SLICE_V9], overlay["intent_one_hot"]
    )
    assert not np.any(condition[LEGACY_KIMODO_FAMILY_SLICE])
    assert np.array_equal(base_condition, np.ones_like(base_condition))


def test_review_disagreement_remains_pending_adjudication(tmp_path: Path):
    sample_id = "anonymous_sample_003"
    video_bytes = b"ambiguous arm gesture"
    video_sha = _sha256(video_bytes)
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / f"{sample_id}.mp4").write_bytes(video_bytes)
    source_manifest = tmp_path / "source.jsonl"
    _write_jsonl(source_manifest, [_source_record(sample_id, video_sha)])
    review_a = tmp_path / "review_a.jsonl"
    review_b = tmp_path / "review_b.jsonl"
    _write_jsonl(review_a, [_decision(sample_id, video_sha, "reviewer_a")])
    _write_jsonl(
        review_b,
        [
            _decision(
                sample_id,
                video_sha,
                "reviewer_b",
                intent_id="raise_hand_get_attention",
            )
        ],
    )

    summary = build_outputs(
        input_manifest=source_manifest,
        video_dir=video_dir,
        ontology_path=DEFAULT_ONTOLOGY_PATH,
        output_dir=tmp_path / "out",
        decision_paths=[review_a, review_b],
    )

    assert summary["train_ready_count"] == 0
    assert summary["pending_count"] == 1
    assert summary["review_status_counts"] == {"pending_adjudication": 1}


def test_source_metadata_in_review_decision_is_rejected(tmp_path: Path):
    sample_id = "anonymous_sample_004"
    video_bytes = b"fake wave video"
    video_sha = _sha256(video_bytes)
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / f"{sample_id}.mp4").write_bytes(video_bytes)
    source_manifest = tmp_path / "source.jsonl"
    _write_jsonl(source_manifest, [_source_record(sample_id, video_sha)])
    compromised = _decision(sample_id, video_sha, "reviewer_a")
    compromised["source_filename"] = "GreetingOwner01.csv"
    decision_path = tmp_path / "compromised.jsonl"
    _write_jsonl(decision_path, [compromised])

    with pytest.raises(ValueError, match="forbidden source fields"):
        build_outputs(
            input_manifest=source_manifest,
            video_dir=video_dir,
            ontology_path=DEFAULT_ONTOLOGY_PATH,
            output_dir=tmp_path / "out",
            decision_paths=[decision_path],
        )


def test_rendered_source_bundle_hides_action_name_from_public_review(tmp_path: Path):
    source_video_dir = tmp_path / "named_videos"
    source_video_dir.mkdir()
    source_video = source_video_dir / "salute_11_clip1.mp4"
    source_video.write_bytes(b"rendered salute sample")
    source_video_sha = _sha256(source_video.read_bytes())
    source_manifest = tmp_path / "named_source.jsonl"
    _write_jsonl(
        source_manifest,
        [
            {
                "clip_id": "salute_11_clip1",
                "source_action": "salute",
                "accepted_for_training": False,
                "manual_video_review_required": True,
                "physical_qc": {"passed": True},
                "render": {
                    "video": source_video.name,
                    "video_sha256": source_video_sha,
                    "checks": {
                        "nonblank": True,
                        "has_motion": True,
                        "full_frame_uncropped": True,
                    },
                },
            }
        ],
    )
    output_dir = tmp_path / "public_bundle"
    private_mapping = tmp_path / "private" / "mapping.jsonl"

    summary = build_anonymous_bundle(
        input_manifest=source_manifest,
        source_video_dir=source_video_dir,
        ontology_path=DEFAULT_ONTOLOGY_PATH,
        output_dir=output_dir,
        private_mapping_path=private_mapping,
    )

    assert summary["anonymous_video_count"] == 1
    assert summary["automatic_intent_labels_emitted"] == 0
    public_manifest = (output_dir / "public" / "anonymous_source_manifest.jsonl").read_text()
    public_queue = (output_dir / "review" / "intent_review_pending.jsonl").read_text()
    assert "salute" not in public_manifest.lower()
    assert "salute" not in public_queue.lower()
    assert "source_action" not in public_queue
    private_record = json.loads(private_mapping.read_text())
    assert private_record["source_clip_id"] == "salute_11_clip1"
    anonymous_video = next((output_dir / "public" / "videos").glob("intent_*.mp4"))
    assert anonymous_video.read_bytes() == source_video.read_bytes()
