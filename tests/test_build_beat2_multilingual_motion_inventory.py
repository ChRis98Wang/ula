import json
from pathlib import Path

import numpy as np

from tools.gmr_v2 import batch_retarget_beat2_v2 as batch
from tools.human_motion_collection import (
    build_beat2_multilingual_motion_inventory as inventory,
)


CLIPS = {
    "english": "1_wayne_0_1_1",
    "japanese": "17_itoi_6_1_1",
    "spanish": "15_carlos_4_1_1",
}


def write_motion(path: Path, frame_count: int = 390, fps: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    poses = np.zeros((frame_count, 165), dtype=np.float32)
    phase = np.linspace(0.0, 4.0 * np.pi, frame_count)
    poses[:, 12 * 3] = 0.15 * np.sin(phase)
    poses[:, 16 * 3] = 0.2 * np.sin(phase)
    np.savez(
        path,
        poses=poses,
        trans=np.zeros((frame_count, 3), dtype=np.float32),
        mocap_frame_rate=np.asarray(fps, dtype=np.int32),
    )


def write_textgrid(path: Path, duration_sec: float = 13.0) -> None:
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
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = {duration_sec}
            text = "first"
    item [2]:
        class = "IntervalTier"
        name = "phonemes"
        xmin = 0
        xmax = {duration_sec}
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = {duration_sec}
            text = "parallel"
''',
        encoding="utf-8",
    )


def write_subset(
    root: Path,
    language: str,
    *,
    frame_count: int = 390,
    fps: int = 30,
    textgrid_duration_sec: float = 13.0,
    extra_split_rows: str = "",
) -> tuple[Path, str]:
    layout = inventory.SUBSETS[language]
    subset_root = root / layout["dataset_subset"]
    clip_id = CLIPS[language]
    write_motion(
        subset_root / "smplxflame_30" / f"{clip_id}.npz",
        frame_count=frame_count,
        fps=fps,
    )
    write_textgrid(
        subset_root / "textgrid" / f"{clip_id}.TextGrid",
        duration_sec=textgrid_duration_sec,
    )
    annotation = subset_root / layout["annotation_dir"] / f"{clip_id}.txt"
    annotation.parent.mkdir(parents=True, exist_ok=True)
    if language == "english":
        annotation.write_text(
            "01_beat_align   0   1   1   0.1\n"
            "02_deictic_l\t1\t7\t6\t0.2\tone\n"
            "08_metaphoric_l\t0.5\t1.5\t1\t0.8\toverlap\n"
            "09_metaphoric_m\t7\t13\t6\t0.9\tscared\n",
            encoding="utf-8",
        )
    else:
        annotation.write_text(
            "これは発話の文脈です" if language == "japanese" else "contexto hablado",
            encoding="utf-8",
        )
    (subset_root / "train_test_split.csv").write_text(
        f"id,type\n{clip_id},train\n{extra_split_rows}", encoding="utf-8"
    )
    return subset_root, clip_id


def build(root: Path, output: Path, languages: list[str], **kwargs):
    return inventory.build_inventory(
        root=root,
        output_dir=output,
        languages=languages,
        workers=1,
        acquisition_manifest=None,
        **kwargs,
    )


def read_records(output: Path):
    path = output / f"{inventory.OUTPUT_STEM}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def read_discards(output: Path):
    path = output / f"{inventory.OUTPUT_STEM}.discarded.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def read_priority(output: Path):
    path = output / "priority_manifest.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_english_preserves_official_semantic_spans_without_emotion(tmp_path):
    _subset_root, clip_id = write_subset(tmp_path, "english")
    output = tmp_path / "out"

    summary = build(tmp_path, output, ["english"])
    records = read_records(output)
    priority = read_priority(output)

    assert len(records) == 2
    assert summary["window_count"] == 2
    assert summary["total_window_duration_sec"] == 12.0
    assert len(priority) == 1
    assert summary["priority_manifest"]["counts_by_language"] == {"english": 1}
    assert summary["priority_manifest"]["counts_by_speaker"] == {"1_wayne": 1}
    assert summary["priority_manifest"]["max_windows_per_source"] == 1
    assert priority[0]["priority_selection"]["high_dynamic_fallback_allowed"] is False
    assert records[0]["dataset_subset"] == "beat_english_v2.0.0"
    assert records[0]["language"] == "english"
    assert records[0]["speaker_key"] == "1_wayne"
    assert records[0]["source_group_id"].endswith(f"/{clip_id}")
    assert records[0]["official_split"] == "train"
    assert records[0]["window"]["selection_status"] == batch.FULL_WINDOW_SELECTION_STATUS
    assert "audio_relpath" not in records[0]
    assert records[0]["audio_enabled"] is False
    assert records[0]["emotion_id"] is None
    assert records[0]["emotion_supervision_mask"] is False
    first_spans = records[0]["official_gesture_semantic_spans"]
    assert [span["source_label"] for span in first_spans] == [
        "01_beat_align",
        "02_deictic_l",
        "08_metaphoric_l",
    ]
    assert first_spans[1]["source_lexical_anchor"] == "one"
    assert first_spans[1]["window_end_sec"] == 6.0
    assert records[1]["official_gesture_semantic_spans"][0][
        "window_start_sec"
    ] == 0.0


def test_japanese_and_spanish_text_remains_speech_context_only(tmp_path):
    write_subset(tmp_path, "japanese")
    write_subset(tmp_path, "spanish")
    output = tmp_path / "out"

    summary = build(tmp_path, output, ["japanese", "spanish"])
    records = read_records(output)

    assert summary["window_counts_by_language"] == {"japanese": 2, "spanish": 2}
    assert all(not record["official_gesture_semantic_spans"] for record in records)
    assert all(
        record["semantic_label_status"] == "speech_context_only_no_gesture_semantics"
        for record in records
    )
    assert all(record["behavior_id"] is None for record in records)
    assert all(record["emotion_id"] is None for record in records)
    assert all(record["source_text"] for record in records)
    assert all("speech_context_only" in record["source_text_role"] for record in records)


def test_textgrid_boundary_failure_is_counted_and_not_emitted(tmp_path):
    write_subset(tmp_path, "spanish", textgrid_duration_sec=7.0)
    output = tmp_path / "out"

    summary = build(tmp_path, output, ["spanish"])
    records = read_records(output)
    discards = read_discards(output)

    assert len(records) == 1
    assert summary["discard_counts_by_reason"] == {"textgrid_boundary_mismatch": 1}
    assert discards[0]["discard_scope"] == "window"
    assert discards[0]["start_frame"] == 180


def test_invalid_motion_and_missing_split_are_audited_while_good_source_survives(
    tmp_path,
):
    subset_root, good_id = write_subset(
        tmp_path,
        "english",
        extra_split_rows="99_missing_0_1_1,test\n",
    )
    bad_id = "2_scott_0_1_1"
    write_motion(subset_root / "smplxflame_30" / f"{bad_id}.npz", fps=60)
    write_textgrid(subset_root / "textgrid" / f"{bad_id}.TextGrid")
    (subset_root / "sem" / f"{bad_id}.txt").write_text(
        "01_beat_align\t0\t13\t13\t0.1\n", encoding="utf-8"
    )
    with (subset_root / "train_test_split.csv").open("a", encoding="utf-8") as handle:
        handle.write(f"{bad_id},val\n")
    output = tmp_path / "out"

    summary = build(tmp_path, output, ["english"])
    records = read_records(output)
    discards = read_discards(output)

    assert {record["source_clip_id"] for record in records} == {good_id}
    assert summary["source_clip_rejected_count"] == 1
    assert summary["discard_counts_by_reason"] == {
        "invalid_motion_npz": 1,
        "official_split_entry_missing_motion": 1,
    }
    assert {item["reason"] for item in discards} == {
        "invalid_motion_npz",
        "official_split_entry_missing_motion",
    }


def test_state_is_reused_and_invalidated_by_annotation_change(tmp_path):
    subset_root, clip_id = write_subset(tmp_path, "japanese")
    output = tmp_path / "out"

    first = build(tmp_path, output, ["japanese"])
    second = build(tmp_path, output, ["japanese"])
    annotation = subset_root / "text" / f"{clip_id}.txt"
    annotation.write_text("更新した発話コンテキスト", encoding="utf-8")
    third = build(tmp_path, output, ["japanese"])

    assert first["resume"]["source_state_computed_count"] == 1
    assert second["resume"]["source_state_reused_count"] == 1
    assert third["resume"]["source_state_computed_count"] == 1
    assert read_records(output)[0]["source_text"] == "更新した発話コンテキスト"


def test_existing_batch_retarget_reader_consumes_motion_only_inventory(tmp_path):
    write_subset(tmp_path, "english")
    output = tmp_path / "out"
    build(tmp_path, output, ["english"])
    priority = read_priority(output)

    eligible, excluded = batch.read_inventory(
        output / "priority_manifest.jsonl", tmp_path
    )

    assert not excluded
    assert len(eligible) == 1
    assert all(task["audio_source"] is None for task in eligible)
    assert all(task["audio_relpath"] is None for task in eligible)
    assert eligible[0]["source_group_key"].startswith(
        "BEAT2/beat_english_v2.0.0/"
    )
    assert eligible[0]["start_frame"] == priority[0]["window"]["start_frame"]


def test_priority_selector_never_falls_back_to_high_dynamic():
    def record(source, start, *, mean, p95, head, label="01_beat_align"):
        return {
            "dataset_subset": "beat_english_v2.0.0",
            "language": "english",
            "speaker_key": "1_wayne",
            "source_clip_id": source,
            "source_group_id": f"BEAT2/beat_english_v2.0.0/{source}",
            "official_gesture_semantic_spans": [{"source_label": label}],
            "window": {
                "start_frame": start,
                "interaction_energy_mean_rad_s": mean,
                "interaction_energy_p95_rad_s": p95,
                "head_neck_mean_rad_s": head,
            },
        }

    records = [
        record("low", 0, mean=0.1, p95=1.0, head=0.1),
        record("low", 180, mean=0.2, p95=1.2, head=0.8, label="02_deictic_l"),
        record("high", 0, mean=0.5, p95=4.1, head=0.5, label="02_deictic_l"),
    ]

    selected, excluded = inventory.select_priority_windows(records)

    assert len(selected) == 1
    assert selected[0]["source_clip_id"] == "low"
    assert selected[0]["window"]["start_frame"] == 180
    assert selected[0]["priority_selection"]["official_non_beat_source_labels"] == [
        "02_deictic_l"
    ]
    assert excluded == [
        {
            "dataset_subset": "beat_english_v2.0.0",
            "language": "english",
            "speaker_key": "1_wayne",
            "source_clip_id": "high",
            "source_group_id": "BEAT2/beat_english_v2.0.0/high",
            "reason": "no_nonstatic_low_dynamic_window",
            "candidate_window_count": 1,
            "min_energy_rad_s": 0.02,
            "max_p95_energy_rad_s": 4.0,
        }
    ]


def test_acquisition_manifest_must_prove_audio_exclusion(tmp_path):
    write_subset(tmp_path, "spanish")
    manifest = tmp_path / "acquisition.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_kind": "beat2_motion_only_acquisition",
                "root": str(tmp_path.resolve()),
                "languages": ["spanish"],
                "audio_policy": "included",
                "verification": {
                    "all_selected_files_present": True,
                    "all_selected_sizes_match": True,
                },
            }
        ),
        encoding="utf-8",
    )

    try:
        inventory.build_inventory(
            root=tmp_path,
            output_dir=tmp_path / "out",
            languages=["spanish"],
            workers=1,
            acquisition_manifest=manifest,
            require_acquisition_manifest=True,
        )
    except ValueError as error:
        assert "audio exclusion" in str(error)
    else:  # pragma: no cover
        raise AssertionError("unsafe acquisition manifest was accepted")
