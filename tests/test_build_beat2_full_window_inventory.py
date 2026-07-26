import csv
import importlib.util
import json
import wave
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools/human_motion_collection/build_beat2_full_window_inventory.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_beat2_full_window_inventory", SCRIPT_PATH
)
assert SPEC and SPEC.loader
INVENTORY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INVENTORY)


def write_wav(path: Path, duration_sec: float, sample_rate: int = 16000) -> None:
    frame_count = int(round(duration_sec * sample_rate))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(np.zeros(frame_count, dtype="<i2").tobytes())


def write_textgrid(path: Path, duration_sec: float) -> None:
    path.write_text(
        f'''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = {duration_sec}
tiers? <exists>
size = 1
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
            text = "first"
        intervals [2]:
            xmin = 3
            xmax = 6
            text = ""
        intervals [3]:
            xmin = 6
            xmax = {duration_sec}
            text = "second"
''',
        encoding="utf-8",
    )


def write_source_fixture(
    root: Path,
    *,
    frame_count: int = 400,
    audio_duration_sec: float | None = None,
    textgrid_duration_sec: float | None = None,
    include_textgrid: bool = True,
) -> tuple[Path, str]:
    clip_id = "12_zhao_2_1_1"
    for directory in ("smplxflame_30", "wave16k", "textgrid", "text"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    poses = np.zeros((frame_count, 165), dtype=np.float32)
    phase = np.linspace(0.0, 4.0 * np.pi, frame_count)
    poses[:, 16 * 3] = 0.2 * np.sin(phase)
    np.savez(
        root / "smplxflame_30" / f"{clip_id}.npz",
        poses=poses,
        trans=np.zeros((frame_count, 3), dtype=np.float32),
        mocap_frame_rate=np.asarray(30, dtype=np.int32),
    )
    source_duration = frame_count / 30.0
    audio_duration = audio_duration_sec if audio_duration_sec is not None else source_duration
    textgrid_duration = (
        textgrid_duration_sec if textgrid_duration_sec is not None else source_duration
    )
    write_wav(root / "wave16k" / f"{clip_id}.wav", audio_duration)
    textgrid_relpath = None
    if include_textgrid:
        write_textgrid(root / "textgrid" / f"{clip_id}.TextGrid", textgrid_duration)
        textgrid_relpath = f"textgrid/{clip_id}.TextGrid"
    (root / "text" / f"{clip_id}.txt").write_text("first second", encoding="utf-8")
    record = {
        "clip_id": clip_id,
        "speaker_id": "12",
        "speaker_name": "zhao",
        "speaker_key": "12_zhao",
        "session_id": "2",
        "official_split": "train",
        "motion_relpath": f"smplxflame_30/{clip_id}.npz",
        "audio_relpath": f"wave16k/{clip_id}.wav",
        "transcript_relpath": f"text/{clip_id}.txt",
        "textgrid_relpath": textgrid_relpath,
        "textgrid_transcript_matches": True,
        "source_frame_count": frame_count,
        "fps": 30.0,
        "audio_sample_rate": 16000,
        "audio_frame_count": int(round(audio_duration * 16000)),
        "issues": [],
        "accepted_for_training": False,
    }
    inventory_path = root / "source.jsonl"
    inventory_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return inventory_path, clip_id


def run_fixture(root: Path, source_inventory: Path, expected_windows: int):
    return INVENTORY.build_inventory(
        source_inventory=source_inventory,
        beat2_root=root,
        output_dir=root / "output",
        expected_source_clip_count=1,
        expected_aligned_source_clip_count=1,
        expected_window_count=expected_windows,
    )


def test_builds_unique_grouped_nonoverlap_windows_and_drops_short_tail(tmp_path):
    source_inventory, source_clip_id = write_source_fixture(tmp_path)

    summary = run_fixture(tmp_path, source_inventory, expected_windows=2)
    records = [
        json.loads(line)
        for line in (tmp_path / "output" / f"{INVENTORY.OUTPUT_STEM}.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    with (tmp_path / "output" / f"{INVENTORY.OUTPUT_STEM}.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        csv_records = list(csv.DictReader(handle))

    assert [record["clip_id"] for record in records] == [
        f"{source_clip_id}_f000000-000180",
        f"{source_clip_id}_f000180-000360",
    ]
    assert {record["source_clip_id"] for record in records} == {source_clip_id}
    assert {record["source_group_id"] for record in records} == {source_clip_id}
    assert records[0]["window"]["end_frame_exclusive"] == records[1]["window"]["start_frame"]
    assert records[0]["window"]["audio_start_sample"] == 0
    assert records[0]["window"]["audio_end_sample_exclusive"] == 96000
    assert records[1]["window"]["audio_end_sample_exclusive"] == 192000
    assert all(record["window"]["overlap_frames"] == 0 for record in records)
    assert all(record["window"]["motion_bounds_valid"] for record in records)
    assert all(record["window"]["audio_bounds_valid"] for record in records)
    assert all(record["window"]["textgrid_bounds_valid"] for record in records)
    assert summary["window_count"] == 2
    assert csv_records[0]["clip_id"] == records[0]["clip_id"]
    assert csv_records[0]["source_clip_id"] == source_clip_id
    assert csv_records[0]["start_frame"] == "0"
    assert summary["total_window_frame_count"] == 360
    assert summary["trailing_short_frame_count"] == 40
    assert summary["validation"]["frame_accounting_valid"] is True


def test_speech_is_context_only_and_never_becomes_behavior_or_emotion(tmp_path):
    source_inventory, _ = write_source_fixture(tmp_path)
    run_fixture(tmp_path, source_inventory, expected_windows=2)
    records = [
        json.loads(line)
        for line in (tmp_path / "output" / f"{INVENTORY.OUTPUT_STEM}.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert records[0]["window_transcript_context"] == "first"
    assert records[1]["window_transcript_context"] == "second"
    assert records[0]["textgrid_units"][0]["text"] == "first"
    assert all(record["behavior_id"] is None for record in records)
    assert all(record["emotion_id"] is None for record in records)
    assert all(record["emotion_supervision_mask"] is False for record in records)
    assert all(record["accepted_for_training"] is False for record in records)
    assert all(record["semantic_label_status"].startswith("unlabeled") for record in records)


def test_boundary_limited_windows_are_not_emitted(tmp_path):
    source_inventory, _ = write_source_fixture(
        tmp_path,
        frame_count=400,
        audio_duration_sec=7.0,
        textgrid_duration_sec=7.0,
    )

    summary = run_fixture(tmp_path, source_inventory, expected_windows=1)

    assert summary["full_motion_window_count_before_boundary_validation"] == 2
    assert summary["window_count"] == 1
    assert summary["rejected_boundary_window_count"] == 1
    assert summary["validation"]["audio_bounds_valid"] is True
    assert summary["validation"]["textgrid_bounds_valid"] is True


def test_sources_without_textgrid_are_excluded_before_windowing(tmp_path):
    source_inventory, _ = write_source_fixture(tmp_path, include_textgrid=False)

    with pytest.raises(ValueError, match="no TextGrid-aligned clips"):
        INVENTORY.build_inventory(
            source_inventory=source_inventory,
            beat2_root=tmp_path,
            output_dir=tmp_path / "output",
            expected_source_clip_count=1,
            expected_aligned_source_clip_count=None,
            expected_window_count=None,
        )


def test_rejects_source_inventory_metadata_drift(tmp_path):
    source_inventory, _ = write_source_fixture(tmp_path)
    record = json.loads(source_inventory.read_text(encoding="utf-8"))
    record["source_frame_count"] += 1
    source_inventory.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="frame count changed"):
        run_fixture(tmp_path, source_inventory, expected_windows=2)
