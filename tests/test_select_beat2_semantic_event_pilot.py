import hashlib
import json
from collections import defaultdict
from pathlib import Path

from tools.human_motion_collection import select_beat2_semantic_event_pilot as pilot
from tools.human_motion_review import adjudicate_training_dataset as adjudicate


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _record(
    speaker: str,
    emotion: str,
    category: str,
    intensity: str,
    *,
    variant: str = "base",
    source_id: str | None = None,
    mean_energy: float = 0.2,
    p95_energy: float = 0.6,
) -> dict:
    source_label = pilot.SOURCE_LABELS[(category, intensity)]
    source_id = source_id or (
        f"{speaker}_0_{pilot.NETWORK_EMOTIONS.index(emotion):03d}_"
        f"{category}_{intensity}_{variant}"
    )
    clip_id = f"{source_id}__{emotion}_{category}_{intensity}_{variant}"
    start = 7
    frames = 31 + pilot.INTENSITIES.index(intensity) * 5
    end = start + frames
    return {
        "schema_version": "1.0.0",
        "dataset": "BEAT2",
        "dataset_subset": pilot.DATASET_SUBSET,
        "language": "english",
        "language_code": "en",
        "clip_id": clip_id,
        "task_id": clip_id,
        "source_clip_id": source_id,
        "source_group_id": source_id,
        "source_group_key": source_id,
        "speaker_key": speaker,
        "official_split": "train",
        "emotion_id": emotion,
        "emotion_supervision_mask": True,
        "emotion_label_status": "official_filename_protocol_network_supported",
        "emotion_label_source": "official_beat2_filename_protocol",
        "semantic_event": {
            "source_label": source_label,
            "category": category,
            "intensity": intensity,
            "intensity_code": pilot.INTENSITY_CODES[intensity],
            "source_lexical_anchor": "there",
            "source_score": 0.8,
        },
        "official_gesture_semantic_spans": [{"source_label": source_label}],
        "annotation_kind": pilot.ANNOTATION_KIND,
        "semantic_label_status": (
            "official_semantic_event_preserved_pending_robot_retarget_qc"
        ),
        "interaction_scope": "human_co_speech_interaction",
        "window_transcript_context": "please look over there",
        "motion_relpath": f"smplxflame_30/{source_id}.npz",
        "motion_sha256": "a" * 64,
        "audio_enabled": False,
        "accepted_for_training": False,
        "issues": [],
        "fps": 30.0,
        "window": {
            "selection_status": pilot.SELECTION_STATUS,
            "start_frame": start,
            "end_frame_exclusive": end,
            "frame_count": frames,
            "interaction_energy_mean_rad_s": mean_energy,
            "interaction_energy_p95_rad_s": p95_energy,
            "active_frame_fraction": 0.75,
        },
        "training_segment": {
            "representation": pilot.VARIABLE_REPRESENTATION,
            "fixed_window_sec": None,
            "start_frame": start,
            "end_frame_exclusive": end,
            "frame_count": frames,
            "boundary_source": {
                "mode": "official_sem_core_plus_motion_low_speed_context"
            },
        },
    }


def _complete_records(*, shared_source_per_speaker: bool = False) -> list[dict]:
    records = []
    for speaker in ("speaker_a", "speaker_b", "speaker_c"):
        for emotion, category, intensity in pilot.all_strata():
            source_id = f"{speaker}_shared_source" if shared_source_per_speaker else None
            records.append(
                _record(
                    speaker,
                    emotion,
                    category,
                    intensity,
                    source_id=source_id,
                )
            )
    return records


def _build(tmp_path: Path, records: list[dict], name: str = "pilot"):
    inventory = tmp_path / f"{name}.jsonl"
    output = tmp_path / f"{name}_out"
    _write_jsonl(inventory, records)
    summary = pilot.build_pilot(
        inventory,
        output,
        seed=17,
        assignment_trials=32,
        max_events_per_source=3,
    )
    return summary, output


def test_complete_pilot_is_balanced_and_speaker_source_disjoint(tmp_path):
    summary, output = _build(tmp_path, _complete_records())
    selected = _read_jsonl(output / f"{pilot.OUTPUT_STEM}.selected.jsonl")
    training_pool = _read_jsonl(output / pilot.TRAINING_POOL_FILENAME)

    assert summary["selection_complete"] is True
    assert summary["selected_count"] == 3 * len(pilot.all_strata()) == 162
    assert summary["selected_counts_by_pilot_split"] == {
        "test": 54,
        "train": 54,
        "validation": 54,
    }
    assert summary["selected_counts_by_emotion"] == {
        emotion: 27 for emotion in sorted(pilot.NETWORK_EMOTIONS)
    }
    assert summary["selected_counts_by_semantic_category"] == {
        category: 54 for category in sorted(pilot.SEMANTIC_CATEGORIES)
    }
    assert summary["selected_counts_by_semantic_intensity"] == {
        intensity: 54 for intensity in sorted(pilot.INTENSITIES)
    }
    assert summary["strict_split_audit"]["speaker_disjoint"] is True
    assert summary["strict_split_audit"]["source_disjoint"] is True
    assert summary["strict_split_audit"][
        "official_split_used_for_pilot_partition"
    ] is False

    speaker_splits = defaultdict(set)
    source_splits = defaultdict(set)
    for record in selected:
        speaker_splits[record["speaker_key"]].add(record["fixed_split_assignment"])
        source_splits[record["source_clip_id"]].add(record["fixed_split_assignment"])
        assert record["training_segment"]["fixed_window_sec"] is None
        assert record["accepted_for_training"] is False
        assert len(record["inventory_record_sha256"]) == 64
        assert record["upstream_inventory_record_sha256"] == record[
            "inventory_record_sha256"
        ]
        assert record["upstream_inventory_manifest_sha256"] == record[
            "inventory_manifest_sha256"
        ]
        assert record["inventory_record_sha256_role"] == (
            "upstream_beat2_semantic_event_inventory_canonical_row"
        )
        assert record["inventory_manifest_sha256_role"] == (
            "upstream_beat2_semantic_event_inventory_manifest"
        )
        assert "source_manifest_sha256" not in record
        assert len(record["pilot_source_group_sha256"]) == 64
        assert record["source_sha256"] == record["motion_sha256"]
        assert record["official_category_verified"] is True
        assert record["official_category_role"] == (
            "verified_metadata_split_and_evaluation_only"
        )
        assert record["official_category_condition_channel"] is None
        assert record["official_category_loss"] is None
        assert record["official_category_conditioning_enabled"] is False
        assert record["robot_observable_motion_form"] == "candidate_unreviewed"
        assert record["communicative_intent"] == "candidate_unreviewed"
        assert record["semantic_supervision_masks"] == {
            "official_category": False,
            "robot_observable_motion_form": False,
            "communicative_intent": False,
            "prompt_text": False,
            "legacy_gesture": False,
        }
        assert record["canonical_action"] == (
            f"official_gesture_category:{record['semantic_event']['category']}"
        )
        assert record["canonical_action_role"] == (
            "official_category_metadata_split_key_only"
        )
        assert record["semantic_mapping_status"] == (
            "official_category_verified_metadata_only"
        )
        assert record["behavior_id"] == "Behavior.InteractPresence"
        assert record["behavior_review_status"] == "candidate_unreviewed"
        assert record["behavior_supervision_mask"] is False
        assert record["behavior_source"] == pilot.BEHAVIOR_MAPPING_SOURCE
        behavior_contract = record["behavior_mapping_contract"]
        assert behavior_contract["revision"] == pilot.BEHAVIOR_MAPPING_REVISION
        assert behavior_contract["supervision"] == "weak_candidate_masked"
        assert behavior_contract["scope"] == (
            "beat2_human_co_speech_interaction_dataset_scope"
        )
        assert behavior_contract["rationale"] == (
            "project_weak_mapping_not_an_official_beat2_behavior_annotation"
        )
        assert behavior_contract["sha256"] == _contract_sha256(behavior_contract)
        assert record["emotion_review_status"] == "official_protocol_confirmed"
        assert record["emotion_supervision_mask"] is False
        assert record["source_emotion_label_verified"] is True
        assert record["emotion_supervision_role"] == (
            "disabled_pending_robot_affect_review"
        )
        assert record["official_emotion_conditioning_enabled"] is False
        assert record["official_emotion_condition_channel"] is None
        assert record["official_emotion_loss"] is None
        assert record["affect_observable_review_status"] == "candidate_unreviewed"
        assert record["affect_observable_supervision_mask"] is False
        assert record["emotion_source"] == pilot.OFFICIAL_EMOTION_SOURCE
        emotion_contract = record["emotion_protocol_contract"]
        assert emotion_contract["emotion_id"] == record["emotion_id"]
        assert emotion_contract["source_sha256"] == record["source_sha256"]
        assert emotion_contract["sha256"] == _contract_sha256(emotion_contract)
        expected_gesture = (
            "pointing"
            if record["semantic_event"]["category"] == "deictic"
            else "upper_body_gesture"
        )
        assert record["semantic_gesture"] == expected_gesture
        prompt = record["canonical_prompt"]["en"]
        assert record["prompt"] == prompt
        assert record["canonical_prompt_role"] == "coarse_category_only"
        assert record["prompt_source"] == pilot.PROMPT_SOURCE
        assert record["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
        assert record["prompt_contract"]["sha256"] == _contract_sha256(
            record["prompt_contract"]
        )
        prompt_schema = record["prompt_schema"]
        assert prompt_schema["schema_version"] == pilot.PROMPT_SCHEMA_VERSION
        assert prompt_schema["canonical_prompt_role"] == "coarse_category_only"
        assert prompt_schema["motion_instruction"] == {
            "emotion_source": pilot.OFFICIAL_EMOTION_SOURCE,
            "official_emotion_id": record["emotion_id"],
            "official_semantic_category": record["semantic_event"]["category"],
            "official_semantic_intensity": record["semantic_event"]["intensity"],
        }
        assert prompt_schema["speech_context"]["included_in_canonical_prompt"] is False
        assert adjudicate._project_weak_behavior(record) is True
        assert adjudicate._official_source_emotion_label_verified(record) is True
        assert record["official_emotion_conditioning_enabled"] is False
        assert record["emotion_supervision_mask"] is False
        assert record["affect_observable_supervision_mask"] is False
    assert all(len(values) == 1 for values in speaker_splits.values())
    assert all(len(values) == 1 for values in source_splits.values())
    assert set(summary["output_sha256"]) >= {
        f"{pilot.OUTPUT_STEM}.candidates.jsonl",
        f"{pilot.OUTPUT_STEM}.selected.jsonl",
        f"{pilot.OUTPUT_STEM}.rejected.jsonl",
        f"{pilot.OUTPUT_STEM}.split_assignments.json",
        pilot.TRAINING_POOL_FILENAME,
    }
    assert summary["selected_frame_count_min"] == 31
    assert summary["selected_frame_count_max"] == 41
    assert summary["selected_distinct_frame_count_count"] == 3
    selected_path = output / f"{pilot.OUTPUT_STEM}.selected.jsonl"
    assert summary["output_sha256"][selected_path.name] == hashlib.sha256(
        selected_path.read_bytes()
    ).hexdigest()
    assert summary["training_pool_count"] == len(training_pool) == 162
    assert summary["training_pool_counts_by_dynamic_band"] == {"low": 162}
    assert all(record["accepted_for_training"] is False for record in training_pool)
    assert all(
        record["training_pool_contract_sha256"]
        == summary["training_pool_contract_sha256"]
        for record in training_pool
    )
    pool_speaker_splits = defaultdict(set)
    for record in training_pool:
        pool_speaker_splits[record["speaker_key"]].add(
            record["fixed_split_assignment"]
        )
    assert all(len(values) == 1 for values in pool_speaker_splits.values())


def _contract_sha256(contract: dict) -> str:
    payload = {key: value for key, value in contract.items() if key != "sha256"}
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def test_selection_is_input_order_independent_for_same_records(tmp_path):
    records = _complete_records()
    first, first_output = _build(tmp_path, records, "first")
    second, second_output = _build(tmp_path, list(reversed(records)), "second")
    first_selected = _read_jsonl(
        first_output / f"{pilot.OUTPUT_STEM}.selected.jsonl"
    )
    second_selected = _read_jsonl(
        second_output / f"{pilot.OUTPUT_STEM}.selected.jsonl"
    )

    assert first["speaker_assignment"]["speaker_to_split"] == second[
        "speaker_assignment"
    ]["speaker_to_split"]
    assert [record["clip_id"] for record in first_selected] == [
        record["clip_id"] for record in second_selected
    ]


def test_low_dynamic_event_is_preferred_over_medium_and_high_fallback(tmp_path):
    records = _complete_records()
    target = ("neutral", "deictic", "low")
    for speaker in ("speaker_a", "speaker_b", "speaker_c"):
        records.extend(
            [
                _record(
                    speaker,
                    *target,
                    variant="medium_alt",
                    p95_energy=2.0,
                ),
                _record(
                    speaker,
                    *target,
                    variant="high_alt",
                    p95_energy=3.5,
                ),
            ]
        )

    summary, output = _build(tmp_path, records)
    selected = _read_jsonl(output / f"{pilot.OUTPUT_STEM}.selected.jsonl")
    training_pool = _read_jsonl(output / pilot.TRAINING_POOL_FILENAME)
    chosen = [
        record
        for record in selected
        if pilot.semantic_stratum(record) == target
    ]

    assert summary["selection_complete"] is True
    assert len(chosen) == 3
    assert {record["pilot_dynamic_band"] for record in chosen} == {"low"}
    assert all("_base" in record["clip_id"] for record in chosen)
    pool_target = [
        record
        for record in training_pool
        if pilot.semantic_stratum(record) == target
    ]
    assert len(pool_target) == 6
    assert {record["pilot_dynamic_band"] for record in pool_target} == {
        "low",
        "medium",
    }
    assert not any("high_alt" in record["clip_id"] for record in training_pool)


def test_lexical_emotion_word_never_overrides_official_emotion(tmp_path):
    records = _complete_records()
    target = next(
        record
        for record in records
        if record["speaker_key"] == "speaker_a"
        and pilot.semantic_stratum(record) == ("neutral", "metaphoric", "medium")
    )
    target["semantic_event"]["source_lexical_anchor"] = "angry and scared"
    target["window_transcript_context"] = "I am angry and scared"

    summary, output = _build(tmp_path, records, "anchor_conflict")
    selected = _read_jsonl(output / f"{pilot.OUTPUT_STEM}.selected.jsonl")
    chosen = next(record for record in selected if record["clip_id"] == target["clip_id"])

    assert summary["selection_complete"] is True
    assert chosen["emotion_id"] == "neutral"
    assert chosen["emotion_protocol_contract"]["emotion_id"] == "neutral"
    assert "neutral" in chosen["canonical_prompt"]["en"]
    assert "angry" not in chosen["canonical_prompt"]["en"].lower()
    assert chosen["prompt_schema"]["speech_context"]["lexical_anchor"] == (
        "angry and scared"
    )
    assert chosen["prompt_schema"]["emotion_resolution"] == {
        "lexical_anchor_used_for_emotion": False,
        "source": pilot.OFFICIAL_EMOTION_SOURCE,
        "transcript_used_for_emotion": False,
    }


def test_all_official_emotions_use_controlled_english_affect_phrases():
    expected = {
        "neutral": "with neutral affect.",
        "sad": "with a sad affect.",
        "happy": "with a happy affect.",
        "angry": "with an angry affect.",
        "surprise": "with a surprised affect.",
        "fear": "with a fearful affect.",
    }
    for emotion_id, suffix in expected.items():
        record = _record("speaker_a", emotion_id, "iconic", "medium")
        provenance = pilot.training_semantic_provenance(record)
        prompt = provenance["canonical_prompt"]["en"]
        assert prompt == provenance["prompt"]
        assert prompt.endswith(suffix)
        assert provenance["prompt_schema"]["motion_instruction"][
            "official_emotion_id"
        ] == emotion_id
        assert provenance["prompt_contract"]["revision"] == (
            pilot.PROMPT_SCHEMA_VERSION
        )


def test_fixed6_bad_emotion_static_high_dynamic_and_source_conflict_are_rejected(
    tmp_path,
):
    records = _complete_records()
    fixed = _record("speaker_a", "neutral", "deictic", "low", variant="fixed")
    fixed["training_segment"]["fixed_window_sec"] = 6.0
    bad_emotion = _record(
        "speaker_a", "neutral", "deictic", "low", variant="bad_emotion"
    )
    bad_emotion["emotion_id"] = "disgust"
    static = _record(
        "speaker_a",
        "neutral",
        "deictic",
        "low",
        variant="static",
        mean_energy=0.001,
    )
    dynamic = _record(
        "speaker_a",
        "neutral",
        "deictic",
        "low",
        variant="dynamic",
        p95_energy=5.0,
    )
    conflict_a = _record(
        "speaker_a",
        "neutral",
        "deictic",
        "low",
        variant="conflict_a",
        source_id="shared_conflicting_source",
    )
    conflict_b = _record(
        "speaker_b",
        "happy",
        "iconic",
        "medium",
        variant="conflict_b",
        source_id="shared_conflicting_source",
    )
    records.extend([fixed, bad_emotion, static, dynamic, conflict_a, conflict_b])

    summary, output = _build(tmp_path, records)
    rejected = _read_jsonl(output / f"{pilot.OUTPUT_STEM}.rejected.jsonl")
    reasons = {
        record["clip_id"]: set(record["pilot_rejection_reasons"])
        for record in rejected
    }

    assert summary["selection_complete"] is True
    assert "fixed_window_forbidden" in reasons[fixed["clip_id"]]
    assert "unsupported_network_emotion" in reasons[bad_emotion["clip_id"]]
    assert "static_below_interaction_energy_floor" in reasons[static["clip_id"]]
    assert "high_dynamic_above_interaction_energy_ceiling" in reasons[
        dynamic["clip_id"]
    ]
    assert reasons[conflict_a["clip_id"]] == {
        "source_clip_maps_to_multiple_speakers"
    }
    assert reasons[conflict_b["clip_id"]] == {
        "source_clip_maps_to_multiple_speakers"
    }


def test_source_diversity_cap_relaxes_only_to_complete_strata(tmp_path):
    records = _complete_records(shared_source_per_speaker=True)
    inventory = tmp_path / "shared.jsonl"
    output = tmp_path / "shared_out"
    _write_jsonl(inventory, records)

    summary = pilot.build_pilot(
        inventory,
        output,
        seed=9,
        assignment_trials=8,
        max_events_per_source=1,
    )
    selected = _read_jsonl(output / f"{pilot.OUTPUT_STEM}.selected.jsonl")

    assert summary["selection_complete"] is True
    assert sum(
        record["pilot_selection_pass"]
        == "source_cap_relaxed_for_stratum_coverage"
        for record in selected
    ) == 3 * (len(pilot.all_strata()) - 1)


def test_incomplete_inventory_is_reported_without_claiming_coverage(tmp_path):
    records = [
        _record(speaker, "neutral", "deictic", "low")
        for speaker in ("speaker_a", "speaker_b", "speaker_c")
    ]
    summary, output = _build(tmp_path, records)

    assert summary["selection_complete"] is False
    assert all(len(missing) == len(pilot.all_strata()) - 1 for missing in summary[
        "missing_strata_by_split"
    ].values())
    assert pilot.main(
        [
            "--input",
            str(tmp_path / "pilot.jsonl"),
            "--output-dir",
            str(tmp_path / "cli_out"),
            "--assignment-trials",
            "0",
        ]
    ) == 2
    assert pilot.main(
        [
            "--input",
            str(tmp_path / "pilot.jsonl"),
            "--output-dir",
            str(tmp_path / "cli_allow_out"),
            "--assignment-trials",
            "0",
            "--allow-incomplete",
        ]
    ) == 0
