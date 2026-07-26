import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

from tools.human_motion_collection import build_interact_dyadic_turns as BUILD


TREE = (
    ("Spine", "Hips"),
    ("Spine1", "Spine"),
    ("Spine2", "Spine1"),
    ("Spine3", "Spine2"),
    ("Neck", "Spine3"),
    ("Neck1", "Neck"),
    ("Head", "Neck1"),
    ("LeftArm", "Spine3"),
    ("LeftForeArm", "LeftArm"),
    ("LeftHand", "LeftForeArm"),
    ("RightArm", "Spine3"),
    ("RightForeArm", "RightArm"),
    ("RightHand", "RightForeArm"),
)


def _children(parent):
    return [name for name, direct_parent in TREE if direct_parent == parent]


def _block(name, depth=0):
    indent = "\t" * depth
    declaration = "ROOT" if name == "Hips" else "JOINT"
    offsets = {
        "Hips": (0, 0, 0),
        "Spine": (0, 10, 0),
        "Spine1": (0, 10, 0),
        "Spine2": (0, 10, 0),
        "Spine3": (0, 10, 0),
        "Neck": (0, 10, 0),
        "Neck1": (0, 5, 0),
        "Head": (0, 5, 0),
        "LeftArm": (0, 0, -15),
        "LeftForeArm": (0, 0, -25),
        "LeftHand": (0, 0, -20),
        "RightArm": (0, 0, 15),
        "RightForeArm": (0, 0, 25),
        "RightHand": (0, 0, 20),
    }
    value = offsets[name]
    rows = [f"{indent}{declaration} {name}", f"{indent}{{"]
    rows.append(f"{indent}\tOFFSET {value[0]} {value[1]} {value[2]}")
    rows.append(
        f"{indent}\tCHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation"
        if name == "Hips"
        else f"{indent}\tCHANNELS 3 Zrotation Yrotation Xrotation"
    )
    for child in _children(name):
        rows.extend(_block(child, depth + 1))
    rows.append(f"{indent}}}")
    return rows


def _write_bvh(path: Path, frame_count: int, angle_by_frame=None):
    angle_by_frame = angle_by_frame or [0.0] * frame_count
    channels = 6 + len(TREE) * 3
    frames = []
    for angle in angle_by_frame:
        values = np.zeros(channels)
        values[-3] = angle
        frames.append(" ".join(str(float(value)) for value in values))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "HIERARCHY",
                *_block("Hips"),
                "MOTION",
                f"Frames: {frame_count}",
                "Frame Time: 0.033333",
                *frames,
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_databases(root: Path):
    actors = sqlite3.connect(root / "actors.db")
    actors.executescript(
        "CREATE TABLE actors(actor_id TEXT PRIMARY KEY, gender TEXT NOT NULL);"
        "CREATE TABLE sessions(date TEXT PRIMARY KEY, male_id TEXT, female_id TEXT);"
        "INSERT INTO actors VALUES('001','male'),('002','female');"
        "INSERT INTO sessions VALUES('20231119','001','002');"
    )
    actors.commit()
    actors.close()
    scenarios = sqlite3.connect(root / "scenarios.db")
    scenarios.executescript(
        "CREATE TABLE relationships(id INTEGER PRIMARY KEY, name TEXT);"
        "CREATE TABLE emotions(id INTEGER PRIMARY KEY, name TEXT);"
        "CREATE TABLE scenarios(id INTEGER PRIMARY KEY, relationship_id INTEGER, "
        "primary_emotion_id INTEGER, character_setup TEXT, scenario TEXT);"
        "INSERT INTO relationships VALUES(1,'coworkers');"
        "INSERT INTO emotions VALUES(1,'amusement');"
        "INSERT INTO scenarios VALUES(51,1,1,'two coworkers','one shares news');"
    )
    scenarios.commit()
    scenarios.close()


def _receipt(root: Path):
    rows = []
    for actor in ("001", "002"):
        relative = f"bvhs/20231119_{actor}_051.bvh"
        path = root / relative
        rows.append(
            {
                "path": relative,
                "local_path": str(path),
                "local_status": "verified",
                "local_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "remote_git_blob_oid_sha1": "a" * 40,
            }
        )
    return {
        "artifact_kind": "interact_motion_only_acquisition",
        "source": {"revision": "152ba832f379c465f5b1e10c67166d646014d675"},
        "license_gate": {
            "training_authorized_by_this_receipt": False,
            "formal_training_blocked": True,
        },
        "files": rows,
    }


def test_pair_inventory_maps_database_metadata_but_masks_all_labels(tmp_path):
    _write_databases(tmp_path)
    for actor in ("001", "002"):
        _write_bvh(tmp_path / f"bvhs/20231119_{actor}_051.bvh", 12)
    metadata = BUILD.load_source_metadata(tmp_path)
    performances, clips = BUILD.build_paired_inventory(
        tmp_path, _receipt(tmp_path), metadata
    )
    assert len(performances) == 1
    assert len(clips) == 2
    performance = performances[0]
    assert performance["pair_available_for_turn_candidate_extraction"] is True
    assert performance["source_metadata"]["primary_emotion_name"] == "amusement"
    assert performance["source_metadata"]["emotion_observable_validated"] is False
    assert not any(performance["semantic_supervision_masks"].values())
    assert performance["emotion_supervision_mask"] is False
    assert performance["admission_mask"] is False
    assert clips[0]["partner_lineage"]["actor_id"] != clips[0]["actor"]["actor_id"]


def test_frame_count_mismatch_rejects_pair_without_truncation(tmp_path):
    _write_databases(tmp_path)
    _write_bvh(tmp_path / "bvhs/20231119_001_051.bvh", 12)
    _write_bvh(tmp_path / "bvhs/20231119_002_051.bvh", 13)
    performances, _clips = BUILD.build_paired_inventory(
        tmp_path, _receipt(tmp_path), BUILD.load_source_metadata(tmp_path)
    )
    assert performances[0]["pair_checks"]["partner_frame_counts_equal"] is False
    assert performances[0]["pair_available_for_turn_candidate_extraction"] is False


def test_rest_basin_cuts_are_not_elapsed_time_windows():
    energy = np.r_[np.zeros(10), np.ones(10), np.zeros(10), np.ones(10), np.zeros(10)]
    intervals, basins = BUILD.natural_rest_intervals(
        energy, threshold_rad_s=0.1, evidence_consecutive_frames=5
    )
    assert len(basins) == 3
    assert intervals == [(0, 24), (24, 50)]
    # The cut lies inside the shared rest basin, not at an elapsed-time multiple.
    assert energy[intervals[0][1] - 1] == 0
    assert energy[intervals[1][0]] == 0


def test_no_internal_rest_preserves_arbitrarily_long_recording_as_one_turn():
    energy = np.ones(100_003)
    intervals, basins = BUILD.natural_rest_intervals(
        energy, threshold_rad_s=0.1, evidence_consecutive_frames=5
    )
    assert basins == []
    assert intervals == [(0, len(energy))]


def test_context_expansion_is_nested_and_eventually_reaches_whole_recording():
    intervals = [(0, 10), (10, 25), (25, 80), (80, 100)]
    levels = BUILD.progressive_natural_context_levels(intervals, 2)
    assert levels[0]["start_frame"] == 25
    assert levels[0]["end_frame_exclusive"] == 80
    assert levels[-1]["start_frame"] == 0
    assert levels[-1]["end_frame_exclusive"] == 100
    for previous, current in zip(levels, levels[1:]):
        assert current["start_frame"] <= previous["start_frame"]
        assert current["end_frame_exclusive"] >= previous["end_frame_exclusive"]


def test_duration_constraint_audit_rejects_fixed_min_max_and_target_fields():
    assert BUILD.duration_constraint_key_paths(
        {
            "fixed_window_sec": 6,
            "nested": {
                "min_duration_sec": 2,
                "max_duration_sec": 12,
                "target_duration_sec": 6,
            },
        }
    ) == [
        "$.fixed_window_sec",
        "$.nested.min_duration_sec",
        "$.nested.max_duration_sec",
        "$.nested.target_duration_sec",
    ]
    assert BUILD.duration_constraint_key_paths(
        {"sample_span_sec": 7.4, "sample_span_sec_role": "diagnostic_only"}
    ) == []


def test_real_candidate_schema_keeps_scenario_emotion_and_admission_masked(tmp_path):
    _write_databases(tmp_path)
    for actor in ("001", "002"):
        _write_bvh(tmp_path / f"bvhs/20231119_{actor}_051.bvh", 12)
    performances, _clips = BUILD.build_paired_inventory(
        tmp_path, _receipt(tmp_path), BUILD.load_source_metadata(tmp_path)
    )
    candidates = BUILD.segment_performance(
        performances[0], threshold_rad_s=0.12, evidence_consecutive_frames=5
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["source_interval"]["frame_count"] == 12
    assert candidate["natural_boundary_evidence"][
        "no_internal_rest_preserved_as_single_turn"
    ] is True
    assert not any(candidate["semantic_supervision_masks"].values())
    assert candidate["emotion_supervision_mask"] is False
    assert candidate["scenario_conditioning_mask"] is False
    assert candidate["accepted_for_training"] is False
    assert candidate["partner_lineage"][0]["partner_actor_id"] == "002"
    assert candidate["context_plan"]["duration_gate_used"] is False
    assert candidate["context_plan"]["selected_level"] is None
    assert candidate["context_plan"]["selected_training_interval"] is None
    assert candidate["context_plan"]["completeness_review"][
        "elapsed_seconds_may_influence_decision"
    ] is False
    assert candidate["context_plan"]["levels"][-1]["end_frame_exclusive"] == 12

    tasks = BUILD.build_actor_robot_episode_tasks(candidates)
    assert len(tasks) == 2
    assert {task["target_actor_lineage"]["actor_id"] for task in tasks} == {
        "001",
        "002",
    }
    assert tasks[0]["interaction_partner_lineage"]["actor_id"] != tasks[0][
        "target_actor_lineage"
    ]["actor_id"]
    assert tasks[0]["retarget_task"]["partner_motion_mixed_into_target"] is False
    assert tasks[0]["training_source_interval"] is None
    assert "preview_only" in tasks[0]["retarget_task"][
        "source_frame_interval_role"
    ]
    assert tasks[0]["collection_scope"] == BUILD.PILOT_COLLECTION_SCOPE
    assert tasks[0]["accepted_for_retarget_batch"] is False
    assert tasks[0]["accepted_for_training"] is False
