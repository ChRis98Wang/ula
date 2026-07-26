import json
import os
from pathlib import Path

import numpy as np
import pytest

from tools.gmr_v2.retarget_xsens_v2 import smooth_and_limit, write_csv
from tools.human_motion_review.build_interact_full_dyad_blind_bundle_v3 import (
    _assert_fail_closed_public,
    affect_record,
    arc_record,
    assert_public_tree_exact,
    materialize_or_reuse_video,
    validate_cached_result,
)
from tools.human_motion_review.build_interact_full_dyad_review_manifest_v3 import (
    NATURAL_DURATION_POLICY,
    _select_pilot,
    _validate_context_plan,
    sha256_file,
)
from tools.human_motion_review.render_interact_full_dyad_evidence_v3 import (
    LINEAGE_CONTRACT,
    build_full_span_lineage,
    reconstruct_actor_retime,
)
from tools.human_motion_review.run_interact_full_dyad_review_v3 import _validate_lineage
from upper_body_skeleton.retarget_v2_18d import JOINT_LIMITS_18D, JOINT_ORDER_18D


def _context_task():
    return {
        "source_interval": {
            "start_frame": 10,
            "end_frame_exclusive": 20,
            "frame_count": 10,
        },
        "context_plan": {
            "duration_policy": NATURAL_DURATION_POLICY,
            "duration_gate_used": False,
            "selected_level": None,
            "selected_training_interval": None,
            "completeness_review": {
                "elapsed_seconds_may_influence_decision": False,
                "shrinking_below_core_or_cutting_inside_a_level_allowed": False,
            },
            "levels": [
                {
                    "level": 0,
                    "start_frame": 10,
                    "end_frame_exclusive": 20,
                },
                {
                    "level": 1,
                    "start_frame": 0,
                    "end_frame_exclusive": 20,
                },
            ],
            "source_recording_interval": [0, 20],
        },
    }


def test_natural_context_validation_never_uses_duration_gate():
    assert len(_validate_context_plan(_context_task())) == 2
    task = _context_task()
    task["context_plan"]["duration_gate_used"] = True
    with pytest.raises(ValueError, match="duration as an admission gate"):
        _validate_context_plan(task)


def test_natural_context_levels_must_expand_without_inside_cut():
    task = _context_task()
    task["context_plan"]["levels"][1] = {
        "level": 1,
        "start_frame": 11,
        "end_frame_exclusive": 21,
    }
    task["context_plan"]["source_recording_interval"] = [11, 21]
    with pytest.raises(ValueError, match="strictly nested"):
        _validate_context_plan(task)


def test_pilot_covers_four_length_ranks_in_each_retime_state():
    rows = []
    for retimed in (False, True):
        for index in range(20):
            rows.append(
                {
                    "dyad_id": f"d_{int(retimed)}_{index:02d}",
                    "dyad_record_sha256": f"{int(retimed)}{index:063d}"[-64:],
                    "source_interval": {"frame_count": 30 + index * 17},
                    "any_safety_retimed": retimed,
                }
            )
    selected = _select_pilot(rows)
    assert len(selected) == 8
    assert {row["any_safety_retimed"] for row in selected} == {False, True}
    assert all(row["pilot_coverage_only_not_admission"] is True for row in selected)
    for retimed in (False, True):
        subset = [row for row in selected if row["any_safety_retimed"] is retimed]
        assert len(subset) == 4
        assert len({row["source_frames"] for row in subset}) == 4


def _actor_retime(tmp_path: Path, raw: np.ndarray, smoothing_window: int):
    safe, _, _ = smooth_and_limit(
        raw,
        30.0,
        3.0,
        smoothing_window,
        joint_order=JOINT_ORDER_18D,
        joint_limits=JOINT_LIMITS_18D,
    )
    source = tmp_path / "source.bvh"
    source.write_bytes(b"synthetic source")
    raw_path = tmp_path / "raw.csv"
    safe_path = tmp_path / "safe.csv"
    quality_path = tmp_path / "quality.json"
    write_csv(raw_path, raw, joint_order=JOINT_ORDER_18D)
    write_csv(safe_path, safe, joint_order=JOINT_ORDER_18D)
    quality = {
        "episode_task_id": "task",
        "episode_task_record_sha256": "a" * 64,
        "source_sha256": sha256_file(source),
        "output_contract": "ula_v2_18d_head_v1",
        "action_dim": 18,
        "max_velocity_rad_s": 3.0,
        "quality_gate": {"passed": True},
    }
    quality_path.write_text(json.dumps(quality), encoding="utf-8")
    actor = {
        "episode_task_id": "task",
        "episode_task_record_sha256": "a" * 64,
        "source_bvh": str(source),
        "source_bvh_sha256": sha256_file(source),
        "quality_json": str(quality_path),
        "quality_json_sha256": sha256_file(quality_path),
        "raw_csv": str(raw_path),
        "raw_csv_sha256": sha256_file(raw_path),
        "safe_csv": str(safe_path),
        "safe_csv_sha256": sha256_file(safe_path),
        "source_frames": len(raw),
        "output_frames": len(safe),
        "safety_retimed": len(raw) != len(safe),
    }
    return actor, safe


def test_retime_reconstruction_binds_raw_and_safe_csv(tmp_path):
    raw = np.zeros((5, 18), dtype=np.float64)
    raw[-1, 0] = 1.2
    actor, safe = _actor_retime(tmp_path, raw, smoothing_window=1)
    reconstructed = reconstruct_actor_retime(actor, smoothing_window=1)
    assert len(reconstructed["safe"]) == len(safe)
    assert reconstructed["reconstruction_max_abs_error_rad"] < 3e-6
    assert len(safe) > len(raw)


def test_retime_reconstruction_rejects_tampered_safe_csv(tmp_path):
    raw = np.zeros((5, 18), dtype=np.float64)
    actor, _ = _actor_retime(tmp_path, raw, smoothing_window=1)
    safe_path = Path(actor["safe_csv"])
    payload = safe_path.read_text(encoding="utf-8").replace("0.00000000", "0.10000000", 1)
    safe_path.write_text(payload, encoding="utf-8")
    actor["safe_csv_sha256"] = sha256_file(safe_path)
    with pytest.raises(ValueError, match="differs from safe CSV"):
        reconstruct_actor_retime(actor, smoothing_window=1)


def test_full_span_lineage_is_monotonic_endpoint_preserving_and_uncropped():
    actor_a = {
        "raw": np.zeros((4, 18)),
        "safe": np.zeros((4, 18)),
        "key_times": np.array([0.0, 1.0, 2.0, 3.0]),
        "output_times": np.array([0.0, 1.0, 2.0, 3.0]),
    }
    actor_b = {
        "raw": np.zeros((4, 18)),
        "safe": np.zeros((7, 18)),
        "key_times": np.array([0.0, 2.0, 4.0, 6.0]),
        "output_times": np.arange(7, dtype=np.float64),
    }
    lineage = build_full_span_lineage(
        source_start_frame=100,
        source_frames=4,
        actor_retimes=[actor_a, actor_b],
    )
    assert lineage["evidence_frames"] == 7
    assert lineage["source_local_frame"][0] == 0
    assert lineage["source_local_frame"][-1] == 3
    assert lineage["source_absolute_frame"][0] == 100
    assert lineage["source_absolute_frame"][-1] == 103
    assert lineage["robot_a_safe_frame"][-1] == 3
    assert lineage["robot_b_safe_frame"][-1] == 6
    assert lineage["source_or_output_frames_cropped"] is False
    assert lineage["fixed_duration_target_used"] is False


def test_runner_lineage_validator_rejects_nonmonotonic_mapping():
    lineage = {
        "artifact_kind": "interact_full_dyad_explicit_frame_lineage_v3",
        "lineage_contract": LINEAGE_CONTRACT,
        "evidence_frames": 4,
        "source_frames": 4,
        "robot_a_output_frames": 4,
        "robot_b_output_frames": 4,
        "evidence_frame": [0, 1, 2, 3],
        "source_local_frame": [0, 1, 2, 3],
        "robot_a_safe_frame": [0, 1, 2, 3],
        "robot_b_safe_frame": [0, 1, 2, 3],
        "all_source_and_output_endpoints_included": True,
        "source_or_output_frames_cropped": False,
        "fixed_duration_target_used": False,
    }
    _validate_lineage(lineage, {"frames": 4})
    lineage["robot_b_safe_frame"] = [0, 2, 1, 3]
    with pytest.raises(ValueError, match="not monotonic"):
        _validate_lineage(lineage, {"frames": 4})


def test_public_queues_are_anonymous_variable_length_and_fail_closed(tmp_path):
    video = tmp_path / "videos" / "sample.mp4"
    video.parent.mkdir()
    video.write_bytes(b"video")
    for record in (
        arc_record("sample", video, "a" * 64),
        affect_record("sample", video, "a" * 64),
    ):
        _assert_fail_closed_public(record)
        assert record["native_variable_length"] is True
        assert record["accepted_for_training"] is False
        assert "turn_id" not in record
        assert "scenario" not in record


def test_public_tree_exact_whitelist_and_nlink_one(tmp_path):
    root = tmp_path / "public"
    videos = root / "videos"
    videos.mkdir(parents=True)
    (root / "summary.json").write_text("{}", encoding="utf-8")
    (root / "arc_action_review_queue.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "affect_review_queue.jsonl").write_text("{}\n", encoding="utf-8")
    (videos / "sample.mp4").write_bytes(b"independent")
    expected = {
        "summary.json",
        "arc_action_review_queue.jsonl",
        "affect_review_queue.jsonl",
        "videos/sample.mp4",
    }
    assert_public_tree_exact(root, expected)
    os.link(videos / "sample.mp4", tmp_path / "alias.mp4")
    with pytest.raises(ValueError, match="independent copy"):
        assert_public_tree_exact(root, expected)


def test_public_tree_rejects_stale_file(tmp_path):
    root = tmp_path / "public"
    (root / "videos").mkdir(parents=True)
    (root / "stale_private_mapping.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="whitelist mismatch"):
        assert_public_tree_exact(root, set())


def test_materialize_or_reuse_video_preserves_valid_existing_copy(tmp_path):
    source = tmp_path / "source.mp4"
    target = tmp_path / "public" / "sample.mp4"
    source.write_bytes(b"video")
    target.parent.mkdir()
    target.write_bytes(b"video")
    inode = target.stat().st_ino
    assert materialize_or_reuse_video(source, target, sha256_file(source)) == "resume_reused"
    assert target.stat().st_ino == inode


def test_materialize_or_reuse_video_rejects_mismatched_existing_copy(tmp_path):
    source = tmp_path / "source.mp4"
    target = tmp_path / "public" / "sample.mp4"
    source.write_bytes(b"video")
    target.parent.mkdir()
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="unsafe or mismatched"):
        materialize_or_reuse_video(source, target, sha256_file(source))


def test_validate_cached_result_rejects_unbound_record_before_reuse(tmp_path):
    record = {"dyad_id": "dyad", "dyad_record_sha256": "a" * 64}
    result = {
        "dyad_id": "dyad",
        "dyad_record_sha256": "a" * 64,
        "evidence_result_record_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="record SHA mismatch"):
        validate_cached_result(record, tmp_path, result)
