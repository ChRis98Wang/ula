import numpy as np
import pytest

from tools.experimental import build_beat2_qwen_text_interaction_60s as long_video


def test_timeline_is_exactly_sixty_seconds_without_gaps():
    prompts = [f"prompt {index}" for index in range(10)]
    seeds = list(range(10))

    timeline = long_video.build_timeline(
        prompts,
        frames_per_segment=180,
        fps=30.0,
        seeds=seeds,
    )

    assert timeline[0]["start_frame"] == 0
    assert timeline[-1]["end_frame_exclusive"] == 1800
    assert timeline[-1]["end_sec"] == 60.0
    for left, right in zip(timeline[:-1], timeline[1:], strict=True):
        assert left["end_frame_exclusive"] == right["start_frame"]
        assert left["end_sec"] == right["start_sec"]


def test_stitch_preserves_network_arrays_and_adds_c0_continuity():
    first = np.zeros((12, 18), dtype=np.float32)
    first[:, 0] = np.linspace(0.0, 1.0, 12)
    second = np.full((12, 18), 4.0, dtype=np.float32)
    original_first = first.copy()
    original_second = second.copy()

    stitched, receipts = long_video.stitch_network_segments(
        [first, second], transition_frames=4
    )

    assert np.array_equal(first, original_first)
    assert np.array_equal(second, original_second)
    assert stitched.shape == (24, 18)
    assert np.array_equal(stitched[11], stitched[12])
    assert receipts[1]["first_frame_matches_previous"] is True
    assert receipts[1]["preblend_boundary_delta_rms_rad"] > 0
    assert np.array_equal(stitched[16:], second[4:])


def test_safety_playback_is_bounded_and_reduces_jerk_without_mutation():
    frames = 120
    source = np.zeros((frames, 18), dtype=np.float32)
    source[:, 0] = np.where(np.arange(frames) < 60, -4.0, 4.0)
    original = source.copy()

    playback = long_video.build_safety_playback(
        source,
        fps=30.0,
        max_velocity_rad_s=30.0,
        smooth_window=11,
        smooth_passes=2,
    )

    assert np.array_equal(source, original)
    assert playback.shape == source.shape
    assert np.isfinite(playback).all()
    raw_jerk = np.max(np.abs(np.diff(source[:, 0], n=3))) * 30.0**3
    playback_jerk = np.max(np.abs(np.diff(playback[:, 0], n=3))) * 30.0**3
    assert playback_jerk < raw_jerk
    assert float(np.max(np.abs(np.diff(playback, axis=0))) * 30.0) <= 30.0


def test_ass_panel_tracks_each_prompt_and_discloses_stitching():
    timeline = long_video.build_timeline(
        ["point left", "gesture happily"],
        frames_per_segment=900,
        fps=30.0,
        seeds=[7, 8],
    )
    timeline[0]["text_path_delta_rms_rad"] = 0.001234

    document = long_video.build_ass_document(
        timeline,
        duration_sec=60.0,
        width=1920,
        height=720,
        pane_width=640,
    )

    assert "A  DEFAULT / NO TEXT LATENT" in document
    assert "B  FROZEN QWEN TEXT LATENT" in document
    assert "NO REFERENCE TRAJECTORY USED FOR GENERATION" in document
    assert "DISPLAY POSTPROCESS ≠ EXACT RAW" in document
    assert "RAW SAVED IN NPZ" in document
    assert "CANONICAL 54-GROUP DEMO" in document
    assert "OPEN TEXT UNVALIDATED" in document
    assert "PATH SENSITIVITY ONLY" in document
    assert "CROSS-PROMPT SEEDS DIFFER" in document
    assert "point left" in document
    assert "gesture happily" in document
    assert "0.001234 rad RMS" in document
    assert "0:00:30.00" in document
    assert "0:01:00.00" in document


def test_timeline_rejects_unpaired_prompts_and_seeds():
    with pytest.raises(long_video.InteractionVideoError, match="paired"):
        long_video.build_timeline(
            ["one", "two"],
            frames_per_segment=180,
            fps=30.0,
            seeds=[1],
        )


def test_canonical_prompt_order_uses_semantic_group_indices(tmp_path):
    cache = tmp_path / "cache.npz"
    np.savez_compressed(
        cache,
        prompts=np.asarray(["group one", "group zero", "group one"]),
        semantic_group_indices=np.asarray([1, 0, 1], dtype=np.int64),
    )

    assert long_video.canonical_prompt_order_from_cache(cache) == [
        "group zero",
        "group one",
    ]
