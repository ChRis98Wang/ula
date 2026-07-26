import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools.human_motion_collection.build_beat2_expression_turn_v8 import (
    NETWORK_EMOTIONS,
    PILOT_SPLIT_COUNTS,
    SEMANTIC_MASKS,
    _duration_accounting,
    _nearest_outer_barriers,
    build_turn_for_group,
    build_context_plan,
    cluster_adjacent_events,
    select_representative_set,
    select_stress_set,
    select_stratified_pilot,
)


def _event(clip_id: str, start: float, end: float, line: int, category: str = "deictic"):
    return {
        "clip_id": clip_id,
        "semantic_event": {
            "source_start_sec": start,
            "source_end_sec": end,
            "source_duration_sec": end - start,
            "source_line_number": line,
            "source_label": "02_deictic_l",
            "category": category,
            "intensity": "low",
        },
    }


def test_natural_rest_or_hard_barrier_splits_adjacent_events():
    energy = np.full(299, 0.5)
    events = [_event("a", 1.0, 2.0, 1), _event("b", 2.4, 3.2, 2)]
    merged = cluster_adjacent_events(
        events,
        [],
        energy,
        frame_count=300,
        fps=30.0,
        rest_energy_max=0.14,
        rest_radius_frames=2,
    )
    assert [len(group) for group in merged] == [2]

    energy[65:75] = 0.01
    merged_with_internal_rest = cluster_adjacent_events(
        events,
        [],
        energy,
        frame_count=300,
        fps=30.0,
        rest_energy_max=0.14,
        rest_radius_frames=2,
    )
    assert [len(group) for group in merged_with_internal_rest] == [1, 1]

    barrier = [
        {
            "source_label": "need_cut",
            "source_start_sec": 2.1,
            "source_end_sec": 2.2,
        }
    ]
    split_on_barrier = cluster_adjacent_events(
        events,
        barrier,
        np.full(299, 0.5),
        frame_count=300,
        fps=30.0,
        rest_energy_max=0.14,
        rest_radius_frames=2,
    )
    assert [len(group) for group in split_on_barrier] == [1, 1]


def test_long_gap_without_rest_is_not_split_by_duration():
    events = [_event("a", 1.0, 2.0, 1), _event("b", 5.0, 6.0, 2)]
    groups = cluster_adjacent_events(
        events,
        [],
        np.full(299, 0.5),
        frame_count=300,
        fps=30.0,
        rest_energy_max=0.14,
        rest_radius_frames=2,
    )
    assert [len(group) for group in groups] == [2]


def test_context_plan_is_strictly_nested_and_neighbor_safe():
    energy = np.full(239, 0.5)
    energy[25:31] = 0.02
    energy[55:61] = 0.03
    energy[178:184] = 0.04
    energy[208:214] = 0.02
    plan = build_context_plan(
        energy,
        core_start=90,
        core_end=150,
        lower_barrier=10,
        upper_barrier=230,
        source_frame_count=240,
        fps=30.0,
        rest_energy_max=0.14,
        rest_radius_frames=2,
    )
    assert plan is not None
    assert plan["same_source_only"] is True
    assert plan["neighbor_crossing_allowed"] is False
    assert len(plan["levels"]) == 2
    first, second = plan["levels"]
    assert second["start_frame"] < first["start_frame"]
    assert second["end_frame_exclusive"] > first["end_frame_exclusive"]


def test_following_annotation_beyond_motion_is_clamped_to_source_frames():
    group = [_event("current", 8.0, 9.0, 1)]
    spans = [
        {
            "source_label": "02_deictic_l",
            "source_start_sec": 10.5,
            "source_end_sec": 11.0,
            "source_line_number": 2,
        }
    ]
    lower, upper = _nearest_outer_barriers(
        group, spans, frame_count=300, fps=30.0
    )
    assert lower == 0
    assert upper == 300


def test_low_amplitude_turn_is_candidate_not_rejected_by_absolute_peak():
    event = _event("low", 1.0, 2.0, 1)
    event.update(
        {
            "_expression_turn_input_record_sha256": "a" * 64,
            "motion_relpath": "motion.npz",
            "motion_sha256": "b" * 64,
            "annotation_relpath": "sem.txt",
            "annotation_sha256": "c" * 64,
            "textgrid_relpath": "words.TextGrid",
            "textgrid_sha256": "d" * 64,
            "source_clip_id": "low_source",
            "source_group_key": "BEAT2/low_source",
            "speaker_key": "speaker",
            "speaker_id": "1",
            "speaker_name": "speaker",
            "official_split": "train",
            "emotion_id": "neutral",
            "source_emotion_label": "neutral",
        }
    )
    event["semantic_event"].update(
        {
            "intensity_code": "l",
            "source_score": 0.8,
            "source_lexical_anchor": None,
        }
    )
    energy = np.full(119, 0.1)
    energy[10:21] = 0.005
    energy[75:86] = 0.005
    upper = np.repeat(energy[:, None], 3, axis=1)
    head = np.repeat(energy[:, None], 3, axis=1)
    candidate, rejection = build_turn_for_group(
        [event],
        source=event,
        spans=[],
        intervals=[(0.0, 4.0, "context")],
        upper_speed=upper,
        head_speed=head,
        energy=energy,
        source_frame_count=120,
        fps=30.0,
        split="train",
        contract_sha256="e" * 64,
        source_manifest_sha256="f" * 64,
        split_manifest_sha256="1" * 64,
        rest_energy_max=0.02,
        rest_radius_frames=2,
        max_energy_p95=4.0,
    )
    assert rejection is None
    assert candidate is not None
    assert candidate["expression_turn"]["peak"]["energy_rad_s"] < 0.25
    assert candidate["expression_turn"]["complete_motion_arc_verified"] is False


def _pilot_candidate(index: int, split: str, emotion: str, category: str):
    source = f"source_{split}_{emotion}_{index}"
    return {
        "clip_id": f"clip_{split}_{emotion}_{category}_{index}",
        "source_clip_id": source,
        "speaker_key": f"speaker_{split}_{index % 8}",
        "fixed_split_assignment": split,
        "emotion_id": emotion,
        "expression_turn": {
            "included_event_count": 2 + index % 2,
            "dominant_official_category": category,
            "left_natural_boundary": {"rest_score_rad_s": 0.03},
            "right_natural_boundary": {"rest_score_rad_s": 0.04},
            "peak": {"prominence_over_boundaries": 8.0},
        },
        "semantic_supervision_masks": dict(SEMANTIC_MASKS),
        "emotion_supervision_mask": False,
        "official_emotion_conditioning_enabled": False,
        "affect_observable_supervision_mask": False,
        "accepted_for_training": False,
    }


def test_stratified_pilot_is_100_fail_closed_and_split_exact():
    candidates = []
    for split in PILOT_SPLIT_COUNTS:
        for emotion in NETWORK_EMOTIONS:
            for index in range(16):
                category = ("deictic", "iconic", "metaphoric")[index % 3]
                candidates.append(_pilot_candidate(index, split, emotion, category))
    pilot = select_stress_set(candidates, pilot_count=100)
    assert len(pilot) == 100
    assert {
        split: sum(record["fixed_split_assignment"] == split for record in pilot)
        for split in PILOT_SPLIT_COUNTS
    } == PILOT_SPLIT_COUNTS
    assert all(record["accepted_for_training"] is False for record in pilot)
    assert all(record["semantic_supervision_masks"] == SEMANTIC_MASKS for record in pilot)
    assert all(record["emotion_supervision_mask"] is False for record in pilot)
    assert all(record["affect_observable_supervision_mask"] is False for record in pilot)
    assert all(record["official_emotion_conditioning_enabled"] is False for record in pilot)
    assert all(record["expression_turn_selection_kind"] == "stress100" for record in pilot)
    assert all(
        record["expression_turn_selection_status"]
        == "selected_stress_pending_retarget_qc"
        for record in pilot
    )
    assert len({record["expression_turn_selection_record_sha256"] for record in pilot}) == 100


def test_stratified_pilot_rejects_non_100_contract():
    with pytest.raises(ValueError, match="exactly 100"):
        select_stratified_pilot([], pilot_count=99)


def _representative_candidate(index, duration_band, event_band):
    split = ("train", "validation", "test")[index % 3]
    return {
        "clip_id": f"rep_{duration_band}_{event_band}_{index}",
        "source_clip_id": f"rep_source_{duration_band}_{event_band}_{index}",
        "speaker_key": f"rep_speaker_{index % 12}",
        "fixed_split_assignment": split,
        "emotion_id": NETWORK_EMOTIONS[index % len(NETWORK_EMOTIONS)],
        "duration_band": duration_band,
        "event_count_band": event_band,
        "training_segment": {"frame_count": 30 + index},
        "expression_turn": {
            "included_event_count": 1,
            "dominant_official_category": "deictic",
        },
        "semantic_supervision_masks": dict(SEMANTIC_MASKS),
        "emotion_supervision_mask": False,
        "official_emotion_conditioning_enabled": False,
        "affect_observable_supervision_mask": False,
        "accepted_for_training": False,
    }


def test_representative_set_tracks_joint_pool_distribution_without_emotion():
    candidates = []
    specification = (
        ("short_under_3s", "single", 100),
        ("medium_3_to_6s", "pair", 60),
        ("long_over_6s", "multi_3plus", 40),
    )
    for duration_band, event_band, count in specification:
        candidates.extend(
            _representative_candidate(index, duration_band, event_band)
            for index in range(count)
        )
    representative = select_representative_set(
        candidates, representative_count=100
    )
    counts = {
        (duration, event): sum(
            record["duration_band"] == duration
            and record["event_count_band"] == event
            for record in representative
        )
        for duration, event, _count in specification
    }
    assert counts == {
        ("short_under_3s", "single"): 50,
        ("medium_3_to_6s", "pair"): 30,
        ("long_over_6s", "multi_3plus"): 20,
    }
    assert len({record["source_clip_id"] for record in representative}) == 100
    assert {record["fixed_split_assignment"] for record in representative} == {
        "train",
        "validation",
        "test",
    }
    assert all(
        record["expression_turn_selection_kind"] == "representative100"
        for record in representative
    )
    relabeled = [dict(record, emotion_id="fear") for record in candidates]
    assert [record["clip_id"] for record in representative] == [
        record["clip_id"]
        for record in select_representative_set(
            relabeled, representative_count=100
        )
    ]


def test_duration_accounting_separates_coverage_from_sample_span():
    accounting = _duration_accounting("candidate", [30, 60])
    assert accounting["candidate_frame_coverage_hours_at_30hz"] == round(
        90 / 30 / 3600, 8
    )
    assert accounting["candidate_sample_span_hours_at_30hz"] == round(
        88 / 30 / 3600, 8
    )
    assert "candidate_duration_hours_at_30hz" not in accounting


def test_cli_smoke_has_no_duration_gap_or_peak_rejection_flags():
    script = (
        Path(__file__).resolve().parents[1]
        / "tools/human_motion_collection/build_beat2_expression_turn_v8.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--max-event-gap-sec" not in completed.stdout
    assert "--min-peak-energy" not in completed.stdout
    assert "--min-peak-prominence" not in completed.stdout
    assert "--fixed-window" not in completed.stdout
