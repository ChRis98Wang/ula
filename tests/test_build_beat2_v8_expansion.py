import json

import pytest

from tools import build_beat2_v8_expansion as expansion


def _pool_record(clip_id, *, start=0, end=40, split="train"):
    return {
        "clip_id": clip_id,
        "task_id": clip_id,
        "dataset": "BEAT2",
        "dataset_subset": "beat_english_v2.0.0",
        "source": "/data/BEAT2/beat_english_v2.0.0/smplxflame_30/source.npz",
        "source_group_key": "BEAT2/beat_english_v2.0.0/source",
        "speaker_key": "speaker",
        "fixed_split_assignment": split,
        "fps": 30,
        "training_segment": {
            "start_frame": start,
            "end_frame_exclusive": end,
            "frame_count": end - start,
        },
    }


def _failed_record(
    source,
    quality_json,
    *,
    target=False,
    collision=False,
    status="quality_failed",
):
    return {
        **source,
        "status": status,
        "frames": source["training_segment"]["frame_count"],
        "quality_json": str(quality_json),
        "quality_gate": {
            "passed": False,
            "target_fit_pass": not target,
            "collision_pass": not collision,
        },
    }


def test_no_smoothing_retry_selects_only_train_physical_regressions(tmp_path):
    fit = _pool_record("fit")
    collision = _pool_record("collision")
    eval_record = _pool_record("eval", split="test")
    quality_fit = tmp_path / "fit.json"
    quality_collision = tmp_path / "collision.json"
    quality_eval = tmp_path / "eval.json"
    quality_fit.write_text(
        json.dumps(
            {
                "raw_limb_target_error_p95_m": 0.039,
                "limb_target_error_p95_m": 0.041,
            }
        ),
        encoding="utf-8",
    )
    quality_collision.write_text(
        json.dumps(
            {
                "raw_limb_target_error_p95_m": 0.05,
                "limb_target_error_p95_m": 0.05,
            }
        ),
        encoding="utf-8",
    )
    quality_eval.write_text(
        json.dumps(
            {
                "raw_limb_target_error_p95_m": 0.039,
                "limb_target_error_p95_m": 0.041,
            }
        ),
        encoding="utf-8",
    )
    failed = [
        _failed_record(fit, quality_fit, target=True),
        _failed_record(collision, quality_collision, collision=True),
        _failed_record(eval_record, quality_eval, target=True),
    ]

    selected, reasons = expansion.build_no_smoothing_retry_candidates(
        {
            "fit": fit,
            "collision": collision,
            "eval": eval_record,
        },
        failed,
        {"speaker": "train"},
        quality_root=tmp_path,
    )

    assert [record["clip_id"] for record in selected] == ["collision", "fit"]
    assert reasons == {
        "collision_retry_without_savgol_filter": 1,
        "raw_fit_pass_but_smoothed_fit_fail": 1,
    }


def test_adjacent_short_pairs_are_train_only_and_require_exact_contiguity():
    left = {
        **_pool_record("left", start=10, end=28),
        "frames": 18,
        "start_frame": 10,
        "end_frame_exclusive": 28,
    }
    right = {
        **_pool_record("right", start=28, end=45),
        "frames": 17,
        "start_frame": 28,
        "end_frame_exclusive": 45,
    }
    gapped = {
        **_pool_record("gapped", start=46, end=60),
        "frames": 14,
        "start_frame": 46,
        "end_frame_exclusive": 60,
    }

    pairs = expansion.adjacent_short_pairs(
        [left, right, gapped], {"speaker": "train"}
    )

    assert len(pairs) == 1
    assert pairs[0]["left_clip_id"] == "left"
    assert pairs[0]["right_clip_id"] == "right"
    assert pairs[0]["source_frame_count"] == 35
    assert pairs[0]["admission"].startswith("candidate_only")


def test_forbidden_cross_dataset_reference_fails_closed():
    record = _pool_record("bad")
    record["source"] = "/data/KiMoDo/clip.npz"

    with pytest.raises(ValueError, match="forbidden"):
        expansion.assert_beat2_record(record, label="bad")
