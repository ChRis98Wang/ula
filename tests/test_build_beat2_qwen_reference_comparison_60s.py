import numpy as np
import pytest

from tools.experimental import (
    build_beat2_qwen_reference_comparison_60s as reference_video,
)


def _record(clip_id, *, prompt="point", frames=20):
    return {
        "clip_id": clip_id,
        "dataset": "BEAT2",
        "prompt": prompt,
        "fixed_split_assignment": "test",
        "accepted_for_training": True,
        "adjudication": {"status": "motion_only_train_ready"},
        "training_admission_status": "motion_only_physical_qc_train_ready",
        "motion_18d": {
            "state": "passed",
            "action_dim": 18,
            "fps": 30,
            "frames": frames,
            "csv_rows": frames,
            "quality_gate": {"passed": True},
        },
    }


def test_endpoint_hold_preserves_every_published_value():
    published = np.arange(5 * 18, dtype=np.float32).reshape(5, 18)
    original = published.copy()

    padded, valid = reference_video.pad_reference_with_endpoint_hold(
        published, slot_frames=9
    )

    assert np.array_equal(published, original)
    assert np.array_equal(padded[:5], published)
    assert np.array_equal(padded[5:], np.repeat(published[-1:], 4, axis=0))
    assert np.array_equal(valid, [True, True, True, True, True, False, False, False, False])


def test_endpoint_hold_rejects_reference_longer_than_slot():
    with pytest.raises(reference_video.ReferenceComparisonError, match="reference"):
        reference_video.pad_reference_with_endpoint_hold(
            np.zeros((10, 18), dtype=np.float32), slot_frames=9
        )


def test_sealed_reference_must_be_lexicographic_first_test_candidate():
    records = {
        "b": _record("b"),
        "a": _record("a"),
        "train": {
            **_record("train"),
            "fixed_split_assignment": "train",
        },
    }

    record, candidates = reference_video.select_sealed_reference(
        prompt="point",
        representative_clip_id="a",
        expected_candidate_count=2,
        manifest_records=records,
    )

    assert record["clip_id"] == "a"
    assert candidates == ["a", "b"]


def test_sealed_reference_rejects_visual_or_noncanonical_pick():
    records = {"b": _record("b"), "a": _record("a")}
    with pytest.raises(
        reference_video.ReferenceComparisonError,
        match="lexicographic first",
    ):
        reference_video.select_sealed_reference(
            prompt="point",
            representative_clip_id="b",
            expected_candidate_count=2,
            manifest_records=records,
        )


def test_ass_discloses_reference_origin_padding_and_non_ground_truth():
    timeline = [
        {
            "index": 1,
            "prompt": "Perform a pointing gesture.",
            "start_sec": 0.0,
            "end_sec": 60.0,
            "reference_source_clip_id": "speaker_clip",
            "reference_published_frames": 53,
            "reference_published_coverage_sec": 53 / 30,
            "upstream_retarget_retimed": True,
            "upstream_source_frame_count": 47,
            "reference_hold_frames": 127,
            "reference_candidate_count": 4,
            "reference_csv_sha256": "a" * 64,
        }
    ]

    document = reference_video.build_reference_ass_document(
        timeline,
        duration_sec=60.0,
        width=1920,
        height=720,
        robot_width=1280,
    )

    assert "BEAT2 → GMR RETARGET" in document
    assert "PROJECT FIXED-TEST BEAT2 18D" in document
    assert "NOT RAW HUMAN SMPL-X OR PAIRED GROUND TRUTH" in document
    assert "NEVER GENERATION INPUT" in document
    assert "THIS COMPARISON ADDS NO LOOP / TIME-WARP / CROP / SMOOTHING" in document
    assert "PUBLISHED 18D MAY INCLUDE DISCLOSED UPSTREAM GMR RETIMING" in document
    assert "Perform a pointing gesture." in document
    assert "published 53 frames" in document
    assert "upstream GMR retimed: true" in document
    assert "hold 127 frames" in document
    assert "0:01:00.00" in document
