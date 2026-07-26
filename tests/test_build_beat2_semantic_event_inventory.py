import json
import math
from pathlib import Path

import numpy as np
import pytest

from tools.human_motion_collection import build_beat2_semantic_event_inventory as inv


def write_motion(path: Path, frame_count: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    poses = np.zeros((frame_count, 165), dtype=np.float32)
    phase = np.linspace(0.0, 6.0 * np.pi, frame_count)
    poses[:, 12 * 3] = 0.08 * np.sin(phase)
    poses[:, 16 * 3] = 0.12 * np.sin(phase)
    np.savez(
        path,
        poses=poses,
        trans=np.zeros((frame_count, 3), dtype=np.float32),
        mocap_frame_rate=np.asarray(30, dtype=np.int32),
    )


def write_textgrid(path: Path, duration_sec: float = 10.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = {duration_sec}
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = {duration_sec}
        intervals: size = 3
        intervals [1]:
            xmin = 0
            xmax = 3
            text = "hello"
        intervals [2]:
            xmin = 3
            xmax = 6
            text = "semantic"
        intervals [3]:
            xmin = 6
            xmax = {duration_sec}
            text = "motion"
    item [2]:
        class = "IntervalTier"
        name = "phonemes"
        xmin = 0
        xmax = {duration_sec}
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = {duration_sec}
            text = "ignored"
''',
        encoding="utf-8",
    )


def write_source(
    root: Path,
    clip_id: str,
    sem_text: str,
    *,
    frame_count: int = 300,
    split: str = "train",
) -> None:
    subset = root / inv.DATASET_SUBSET
    write_motion(subset / "smplxflame_30" / f"{clip_id}.npz", frame_count)
    write_textgrid(subset / "textgrid" / f"{clip_id}.TextGrid", frame_count / 30)
    sem_path = subset / "sem" / f"{clip_id}.txt"
    sem_path.parent.mkdir(parents=True, exist_ok=True)
    sem_path.write_text(sem_text, encoding="utf-8")
    split_path = subset / "train_test_split.csv"
    split_path.write_text(f"id,type\n{clip_id},{split}\n", encoding="utf-8")


def build(root: Path, output: Path, **kwargs):
    return inv.build_inventory(root, output, workers=1, **kwargs)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def full_candidates(output: Path) -> list[dict]:
    return read_jsonl(output / f"{inv.OUTPUT_STEM}.full_candidates.jsonl")


def discards(output: Path) -> list[dict]:
    return read_jsonl(output / f"{inv.OUTPUT_STEM}.discarded.jsonl")


def test_official_events_produce_native_variable_lengths_without_six_second_window(
    tmp_path: Path,
) -> None:
    clip_id = "1_wayne_0_1_1"
    write_source(
        tmp_path,
        clip_id,
        "01_beat_align\t0\t0.6\t0.6\t0.1\n"
        "02_deictic_l\t0.6\t1.4\t0.8\t0.2\there\n"
        "01_beat_align\t1.4\t3.0\t1.6\t0.1\n"
        "07_iconic_h\t3.0\t5.2\t2.2\t0.7\twide\n"
        "01_beat_align\t5.2\t10.0\t4.8\t0.1\n",
    )
    output = tmp_path / "out"

    summary = build(tmp_path, output, max_context_sec=0.25)
    records = full_candidates(output)

    assert len(records) == 2
    assert records[0]["window"]["frame_count"] != records[1]["window"]["frame_count"]
    assert all(record["window"]["frame_count"] != 180 for record in records)
    assert all(record["window"]["selection_status"] == inv.SELECTION_STATUS for record in records)
    assert all(
        record["training_segment"]["representation"] == inv.REPRESENTATION
        and record["training_segment"]["fixed_window_sec"] is None
        and "official" in record["training_segment"]["boundary_source"]["mode"]
        for record in records
    )
    assert summary["segment_policy"]["fixed_window_sec"] is None
    assert summary["distinct_candidate_frame_count_count"] == 2
    assert all(record["accepted_for_training"] is False for record in records)
    assert all(record["behavior_id"] is None for record in records)
    assert all(
        record["interaction_scope"] == "human_co_speech_interaction"
        and record["semantic_mapping_status"] == "unmapped_pending_retarget_qc"
        for record in records
    )
    assert records[0]["semantic_event"] == {
        "category": "deictic",
        "intensity": "low",
        "intensity_code": "l",
        "source_duration_sec": 0.8,
        "source_end_sec": 1.4,
        "source_label": "02_deictic_l",
        "source_lexical_anchor": "here",
        "source_line_number": 2,
        "source_score": 0.2,
        "source_start_sec": 0.6,
    }


@pytest.mark.parametrize(
    ("clip_id", "raw", "network"),
    [
        ("1_wayne_0_0_64", "neutral", "neutral"),
        ("1_wayne_0_65_72", "happiness", "happy"),
        ("1_wayne_0_73_80", "anger", "angry"),
        ("1_wayne_0_81_86", "sadness", "sad"),
        ("1_wayne_0_95_102", "surprise", "surprise"),
        ("1_wayne_0_103_110", "fear", "fear"),
        ("1_wayne_1_1_12", "neutral", "neutral"),
        ("1_wayne_3_1_12", "neutral", "neutral"),
        ("1_wayne_5_1_12", "neutral", "neutral"),
        ("1_wayne_7_1_12", "neutral", "neutral"),
    ],
)
def test_official_filename_emotion_mapping(
    clip_id: str, raw: str, network: str
) -> None:
    result = inv.parse_filename_emotion(clip_id)

    assert result["source_emotion_label"] == raw
    assert result["emotion_id"] == network
    assert result["emotion_supervision_mask"] is True


@pytest.mark.parametrize(
    ("clip_id", "raw"),
    [
        ("1_wayne_0_87_94", "contempt"),
        ("1_wayne_0_111_118", "disgust"),
    ],
)
def test_unsupported_official_emotions_are_preserved_but_masked(
    clip_id: str, raw: str
) -> None:
    result = inv.parse_filename_emotion(clip_id)

    assert result["source_emotion_label"] == raw
    assert result["emotion_id"] is None
    assert result["emotion_supervision_mask"] is False
    assert "preserved_network_unsupported" in result["emotion_label_status"]


def test_cross_emotion_id_range_fails_closed() -> None:
    with pytest.raises(ValueError, match="ambiguous_speech_id_range"):
        inv.parse_filename_emotion("1_wayne_0_64_65")


def test_control_labels_are_audited_and_never_emitted_as_candidates(
    tmp_path: Path,
) -> None:
    write_source(
        tmp_path,
        "1_wayne_0_1_1",
        "01_beat_align\t0\t1\t1\t0.1\n"
        "00_nogesture\t1\t2\t1\t0.0\t00\n"
        "habit\t2\t3\t1\t0.0\t11\n"
        "need_cut\t3\t4\t1\t0.0\t12\n"
        "06_iconic_m\t4\t5.5\t1.5\t0.6\tshow\n"
        "02_iconic_l\t5.5\t6.5\t1\t0.2\tmalformed protocol label\n"
        "01_beat_align\t6.5\t10\t3.5\t0.1\n",
    )
    output = tmp_path / "out"

    build(tmp_path, output)
    records = full_candidates(output)
    rejected = discards(output)

    assert [record["semantic_event"]["source_label"] for record in records] == [
        "06_iconic_m"
    ]
    denied_labels = {
        item["official_span"]["source_label"]
        for item in rejected
        if item["reason"] == "official_control_label_denied_for_training"
    }
    assert denied_labels == {"01_beat_align", "00_nogesture", "habit", "need_cut"}
    assert "02_iconic_l" in {
        item["official_span"]["source_label"]
        for item in rejected
        if item["reason"] == "unsupported_official_semantic_label"
    }
    assert all(item["accepted_for_training"] is False for item in rejected)


def test_event_overlapping_denied_control_span_fails_closed(tmp_path: Path) -> None:
    write_source(
        tmp_path,
        "1_wayne_0_1_1",
        "05_iconic_l\t1\t3\t2\t0.5\tshape\n"
        "need_cut\t2\t4\t2\t0.0\t12\n",
    )
    output = tmp_path / "out"

    build(tmp_path, output)

    assert full_candidates(output) == []
    assert "semantic_event_overlaps_denied_or_unknown_span" in {
        item["reason"] for item in discards(output)
    }


def test_adaptive_context_does_not_cross_neighboring_official_events(
    tmp_path: Path,
) -> None:
    write_source(
        tmp_path,
        "1_wayne_0_1_1",
        "01_beat_align\t0\t1\t1\t0.1\n"
        "02_deictic_l\t1\t2\t1\t0.2\tthis\n"
        "01_beat_align\t2\t2.2\t0.2\t0.1\n"
        "05_iconic_l\t2.2\t3.0\t0.8\t0.5\tshape\n"
        "01_beat_align\t3\t10\t7\t0.1\n",
    )
    output = tmp_path / "out"

    build(tmp_path, output, max_context_sec=2.0)
    first, second = full_candidates(output)

    second_core_start = math.floor(2.2 * 30)
    first_core_end = math.ceil(2.0 * 30)
    assert first["window"]["end_frame_exclusive"] <= second_core_start
    assert second["window"]["start_frame"] >= first_core_end
    assert first["training_segment"]["boundary_source"][
        "following_barrier_source_label"
    ] == "05_iconic_l"
    assert second["training_segment"]["boundary_source"][
        "previous_barrier_source_label"
    ] == "02_deictic_l"


def test_semantic_event_can_end_at_last_motion_frame(tmp_path: Path) -> None:
    clip_id = "1_wayne_0_1_1"
    write_source(
        tmp_path,
        clip_id,
        "01_beat_align\t0\t8\t8\t0.1\n"
        "06_iconic_m\t8\t10\t2\t0.5\tfinish\n",
        frame_count=300,
    )
    output = tmp_path / "out"

    build(tmp_path, output, max_context_sec=0.75)
    record = full_candidates(output)[0]

    assert record["training_segment"]["end_frame_exclusive"] == 300
    assert record["window"]["end_frame_exclusive"] == 300
    assert math.isfinite(
        record["training_segment"]["boundary_source"][
            "right_motion_boundary_energy_rad_s"
        ]
    )


def test_lexical_anchor_is_not_used_as_emotion_and_supported_manifest_is_filtered(
    tmp_path: Path,
) -> None:
    write_source(
        tmp_path,
        "1_wayne_0_1_1",
        "09_metaphoric_m\t1\t2\t1\t0.9\tangry and scared\n",
    )
    output = tmp_path / "out"

    build(tmp_path, output)
    record = full_candidates(output)[0]
    supported = read_jsonl(
        output / f"{inv.OUTPUT_STEM}.network_emotion_supported.jsonl"
    )

    assert record["source_emotion_label"] == "neutral"
    assert record["emotion_id"] == "neutral"
    assert record["semantic_event"]["source_lexical_anchor"] == "angry and scared"
    assert supported == [record]


def test_unsupported_emotion_remains_in_full_but_not_network_manifest(
    tmp_path: Path,
) -> None:
    write_source(
        tmp_path,
        "1_wayne_0_87_87",
        "04_deictic_h\t1\t2.2\t1.2\t0.4\tyou\n",
    )
    output = tmp_path / "out"

    summary = build(tmp_path, output)

    assert len(full_candidates(output)) == 1
    assert read_jsonl(
        output / f"{inv.OUTPUT_STEM}.network_emotion_supported.jsonl"
    ) == []
    assert summary["network_emotion_supported_candidate_count"] == 0


def test_source_state_is_resumable_and_bound_to_annotation(tmp_path: Path) -> None:
    clip_id = "1_wayne_0_1_1"
    write_source(
        tmp_path,
        clip_id,
        "02_deictic_l\t1\t2\t1\t0.2\there\n",
    )
    output = tmp_path / "out"

    first = build(tmp_path, output)
    second = build(tmp_path, output)
    sem_path = tmp_path / inv.DATASET_SUBSET / "sem" / f"{clip_id}.txt"
    sem_path.write_text("05_iconic_l\t1\t2\t1\t0.5\tshape\n", encoding="utf-8")
    third = build(tmp_path, output)

    assert first["resume"]["source_state_computed_count"] == 1
    assert second["resume"]["source_state_reused_count"] == 1
    assert third["resume"]["source_state_computed_count"] == 1
    assert full_candidates(output)[0]["semantic_event"]["source_label"] == "05_iconic_l"
    assert (output / f"{inv.OUTPUT_STEM}.provenance.json").is_file()
