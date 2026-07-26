import csv
import importlib.util
import json
import struct
import wave
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools/human_motion_collection/build_beat2_interaction_inventory.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_beat2_interaction_inventory", SCRIPT_PATH
)
assert SPEC and SPEC.loader
INVENTORY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INVENTORY)


def make_poses(frame_count=240, moving_start=30, moving_end=210, amplitude=0.25):
    poses = np.zeros((frame_count, 165), dtype=np.float32)
    phase = np.linspace(0.0, 4.0 * np.pi, moving_end - moving_start)
    for joint_index in (12, 15, 16, 17, 18, 19, 20, 21):
        poses[moving_start:moving_end, joint_index * 3] = amplitude * np.sin(phase)
    return poses


def write_textgrid(path, duration_sec=8.0):
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
        intervals: size = 2
        intervals [1]:
            xmin = 0
            xmax = 4
            text = "hello"
        intervals [2]:
            xmin = 4
            xmax = {duration_sec}
            text = "world"
''',
        encoding="utf-8",
    )


def write_wav(path, duration_sec=8.0, sample_rate=16000):
    frame_count = int(round(duration_sec * sample_rate))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(np.zeros(frame_count, dtype="<i2").tobytes())


def write_float_wav(path, frame_count=160, sample_rate=16000):
    data = np.zeros(frame_count, dtype="<f4").tobytes()
    fmt = struct.pack(
        "<HHIIHH", 3, 1, sample_rate, sample_rate * 4, 4, 32
    )
    riff_size = 4 + (8 + len(fmt)) + (8 + len(data))
    path.write_bytes(
        b"RIFF"
        + struct.pack("<I", riff_size)
        + b"WAVEfmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )


def write_fixture(root, *, with_textgrid=True, split=True, poses=None):
    clip_id = "12_zhao_2_1_1"
    motion_root = root / "smplxflame_30"
    text_root = root / "text"
    textgrid_root = root / "textgrid"
    audio_root = root / "wave16k"
    motion_root.mkdir(parents=True)
    text_root.mkdir(parents=True)
    textgrid_root.mkdir(parents=True)
    audio_root.mkdir(parents=True)
    poses = make_poses() if poses is None else poses
    np.savez(
        motion_root / f"{clip_id}.npz",
        poses=poses,
        trans=np.zeros((len(poses), 3), dtype=np.float32),
        mocap_frame_rate=np.asarray(30, dtype=np.int32),
    )
    (text_root / f"{clip_id}.txt").write_text(
        "Hello, world!", encoding="utf-8"
    )
    write_wav(audio_root / f"{clip_id}.wav", len(poses) / 30)
    if with_textgrid:
        write_textgrid(textgrid_root / f"{clip_id}.TextGrid", len(poses) / 30)
    split_rows = f"{clip_id},train\n" if split else "13_lu_2_2_2,test\n"
    (root / "train_test_split.csv").write_text(
        "id,type\n" + split_rows, encoding="utf-8"
    )
    return clip_id


def test_window_selector_prefers_nonstatic_low_dynamic_window():
    poses = make_poses(frame_count=360, moving_start=90, moving_end=330)
    result = INVENTORY.select_interaction_window(poses)

    assert result["frame_count"] == 180
    assert result["duration_sec"] == 6.0
    assert result["selection_status"] == "selected_nonstatic_low_dynamic"
    assert result["interaction_energy_mean_rad_s"] >= 0.02
    assert result["interaction_energy_p95_rad_s"] <= 4.0


def test_window_selector_requires_aligned_speech_when_available():
    poses = make_poses(frame_count=360, moving_start=0, moving_end=360)
    intervals = [(0.0, 6.0, ""), (6.0, 12.0, "spoken")]
    result = INVENTORY.select_interaction_window(
        poses, speech_intervals=intervals
    )

    assert result["selection_status"].endswith("with_aligned_speech")
    assert result["aligned_speech_unit_count"] > 0
    assert result["end_sec"] > 6.0


def test_energy_is_invariant_to_face_fingers_and_translation():
    poses = make_poses()
    baseline = INVENTORY.select_interaction_window(poses)
    ignored_changed = poses.copy()
    ignored_changed[:, 66:165] = np.linspace(0.0, 2.0, len(poses))[:, None]
    changed = INVENTORY.select_interaction_window(ignored_changed)

    assert changed == baseline
    assert max(INVENTORY.UPPER_BODY_JOINT_INDICES) == 21
    ignored = {
        index
        for indices in INVENTORY.IGNORED_POSE_JOINT_INDICES.values()
        for index in indices
    }
    assert ignored.isdisjoint(INVENTORY.UPPER_BODY_JOINT_INDICES)
    assert set(range(22, 55)).issubset(ignored)


def test_static_clip_is_flagged_not_silently_admitted():
    result = INVENTORY.select_interaction_window(
        np.zeros((240, 165), dtype=np.float32)
    )

    assert result["selection_status"] == "fallback_no_nonstatic_window"


def test_inventory_uses_only_broad_label_and_preserves_context(tmp_path):
    clip_id = write_fixture(tmp_path)
    output_dir = tmp_path / "catalog"
    summary = INVENTORY.build_inventory(
        tmp_path, output_dir, expected_clip_count=1
    )
    record = json.loads(
        (output_dir / f"{INVENTORY.OUTPUT_STEM}.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    with (output_dir / f"{INVENTORY.OUTPUT_STEM}.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        csv_record = next(csv.DictReader(handle))

    assert record["clip_id"] == clip_id
    assert record["interaction_label"] == "co_speech_conversational_gesture"
    assert csv_record["interaction_label"] == "co_speech_conversational_gesture"
    assert record["transcript"] == "Hello, world!"
    assert record["transcript_role"].endswith("not_action_label")
    assert record["window_transcript_role"].endswith("not_action_label")
    assert record["speaker_key"] == "12_zhao"
    assert record["official_split"] == "train"
    assert record["audio_relpath"].endswith(f"{clip_id}.wav")
    assert record["audio_sample_rate"] == 16000
    assert record["audio_channels"] == 1
    assert record["audio_dtype"] == "int16"
    assert record["motion_audio_duration_abs_diff_sec"] == 0.0
    assert record["textgrid_transcript_matches"] is True
    assert record["accepted_for_training"] is False
    assert record["manual_review_required"] is True
    assert summary["accepted_for_training_count"] == 0
    assert summary["record_count"] == 1


def test_inventory_reports_missing_textgrid_and_split_without_inventing(tmp_path):
    write_fixture(tmp_path, with_textgrid=False, split=False)
    output_dir = tmp_path / "catalog"
    summary = INVENTORY.build_inventory(
        tmp_path, output_dir, expected_clip_count=1
    )
    record = json.loads(
        (output_dir / f"{INVENTORY.OUTPUT_STEM}.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )

    assert record["official_split"] is None
    assert record["textgrid_relpath"] is None
    assert "missing_textgrid_alignment" in record["issues"]
    assert "missing_official_split" in record["issues"]
    assert summary["missing_textgrid_for_motion_count"] == 1
    assert summary["missing_official_split_for_motion_count"] == 1


def test_inventory_flags_audio_duration_and_textgrid_transcript_mismatch(tmp_path):
    clip_id = write_fixture(tmp_path)
    (tmp_path / "text" / f"{clip_id}.txt").write_text(
        "Different transcript", encoding="utf-8"
    )
    write_wav(tmp_path / "wave16k" / f"{clip_id}.wav", duration_sec=1.0)
    output_dir = tmp_path / "catalog"
    summary = INVENTORY.build_inventory(
        tmp_path, output_dir, expected_clip_count=1
    )
    record = json.loads(
        (output_dir / f"{INVENTORY.OUTPUT_STEM}.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )

    assert "textgrid_transcript_mismatch" in record["issues"]
    assert "motion_audio_duration_mismatch_gt_0_3s" in record["issues"]
    assert summary["textgrid_transcript_mismatch_motion_count"] == 1
    assert summary["textgrid_transcript_mismatch_all_source_count"] == 1


def test_transcript_alignment_normalization_is_explicit():
    assert INVENTORY.normalize_transcript_for_alignment(
        "Hello， world!"
    ) == INVENTORY.normalize_transcript_for_alignment("helloworld")


def test_wav_metadata_supports_ieee_float_without_decoding(tmp_path):
    path = tmp_path / "float.wav"
    write_float_wav(path)

    metadata = INVENTORY.read_wav_metadata(path)

    assert metadata == {
        "sample_rate": 16000,
        "channels": 1,
        "dtype": "float32",
        "frame_count": 160,
        "duration_sec": 0.01,
        "format": "ieee_float",
    }


@pytest.mark.parametrize(
    ("poses", "trans", "fps", "message"),
    [
        (np.zeros((10, 164)), np.zeros((10, 3)), 30, r"poses\[T>=2,165\]"),
        (np.zeros((10, 165)), np.zeros((9, 3)), 30, r"trans\[10,3\]"),
        (np.zeros((10, 165)), np.zeros((10, 3)), 60, "mocap_frame_rate=30"),
    ],
)
def test_npz_structural_validation(tmp_path, poses, trans, fps, message):
    path = tmp_path / "bad.npz"
    np.savez(path, poses=poses, trans=trans, mocap_frame_rate=fps)

    with pytest.raises(ValueError, match=message):
        INVENTORY.validate_motion_npz(path)


def test_inventory_requires_nonempty_transcript(tmp_path):
    clip_id = write_fixture(tmp_path)
    (tmp_path / "text" / f"{clip_id}.txt").write_text("  \n", encoding="utf-8")

    with pytest.raises(ValueError, match="Empty transcript"):
        INVENTORY.build_inventory(
            tmp_path, tmp_path / "catalog", expected_clip_count=1
        )


def test_expected_clip_count_is_enforced(tmp_path):
    write_fixture(tmp_path)

    with pytest.raises(ValueError, match="Expected 310 .* found 1"):
        INVENTORY.build_inventory(tmp_path, tmp_path / "catalog")
