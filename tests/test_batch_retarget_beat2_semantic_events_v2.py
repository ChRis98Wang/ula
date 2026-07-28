import hashlib
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from tools.gmr_v2 import batch_retarget_beat2_semantic_events_v2 as grouped
from tools.gmr_v2 import batch_retarget_beat2_v2 as ordinary
from tools.gmr_v2.retarget_beat2_grouped_v2 import (
    EVENT_REVIEW_PROVENANCE_FIELDS,
    MOTION_FOUNDATION_RETARGET_SEGMENT_REPRESENTATION,
    RETARGET_SEGMENT_REPRESENTATION,
    build_retarget_segment_contract,
    interiorize_neutral_qpos,
)


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _semantic_record(
    clip_id,
    source_clip_id,
    start,
    end,
    *,
    fixed_window_sec=None,
):
    return {
        "clip_id": clip_id,
        "task_id": clip_id,
        "source_clip_id": source_clip_id,
        "source_group_id": source_clip_id,
        "dataset": "BEAT2",
        "dataset_subset": "beat_english_v2.0.0",
        "language": "english",
        "language_code": "en",
        "fps": 30.0,
        "motion_relpath": f"smplxflame_30/{source_clip_id}.npz",
        "official_split": "train",
        "speaker_key": "1_wayne",
        "annotation_kind": "official_gesture_semantic_event",
        "semantic_label_status": (
            "official_semantic_event_preserved_pending_robot_retarget_qc"
        ),
        "semantic_event": {
            "source_label": "04_deictic_h",
            "category": "deictic",
            "intensity": "high",
            "lexical_anchor": "there",
        },
        "official_gesture_semantic_spans": [
            {
                "source_label": "04_deictic_h",
                "category": "deictic",
                "intensity": "high",
            }
        ],
        "emotion_id": "angry",
        "emotion_review_status": "official_protocol_confirmed",
        "emotion_supervision_mask": False,
        "source_emotion_label_verified": True,
        "emotion_supervision_role": "disabled_pending_robot_affect_review",
        "official_emotion_conditioning_enabled": False,
        "official_emotion_condition_channel": None,
        "official_emotion_loss": None,
        "affect_observable_review_status": "candidate_unreviewed",
        "affect_observable_supervision_mask": False,
        "emotion_source": "official_beat2_filename_protocol",
        "emotion_protocol_contract": {"sha256": "e" * 64},
        "emotion_label_status": "official_beat_filename_protocol_mapped",
        "behavior_id": "Behavior.InteractPresence",
        "behavior_review_status": "candidate_unreviewed",
        "behavior_supervision_mask": False,
        "behavior_source": "project_dataset_scope_weak_mapping_v1",
        "behavior_mapping_contract": {"sha256": "b" * 64},
        "official_category_verified": True,
        "official_category_role": "verified_metadata_split_and_evaluation_only",
        "official_category_condition_channel": None,
        "official_category_loss": None,
        "official_category_conditioning_enabled": False,
        "robot_observable_motion_form": "candidate_unreviewed",
        "communicative_intent": "candidate_unreviewed",
        "semantic_supervision_masks": {
            "official_category": False,
            "robot_observable_motion_form": False,
            "communicative_intent": False,
            "prompt_text": False,
            "legacy_gesture": False,
        },
        "canonical_action": "official_gesture_category:deictic",
        "canonical_action_role": "official_category_metadata_split_key_only",
        "semantic_mapping_status": "official_category_verified_metadata_only",
        "canonical_prompt": {
            "en": "Perform a high-intensity deictic pointing gesture while expressing angry."
        },
        "canonical_prompt_role": "coarse_category_only",
        "prompt": (
            "Perform a high-intensity deictic pointing gesture while expressing angry."
        ),
        "prompt_schema": {
            "schema_version": "beat2_official_semantic_event_qwen_prompt_v1",
            "speech_context": {"included_in_canonical_prompt": False},
        },
        "prompt_source": "deterministic_official_semantic_event_and_emotion_v1",
        "prompt_sha256": "d" * 64,
        "prompt_contract": {"sha256": "c" * 64},
        "motion_sha256": "a" * 64,
        "inventory_record_sha256": "5" * 64,
        "upstream_inventory_record_sha256": "5" * 64,
        "inventory_manifest_sha256": "1" * 64,
        "upstream_inventory_manifest_sha256": "1" * 64,
        "pilot_selector_contract_sha256": "2" * 64,
        "pilot_source_group_sha256": "3" * 64,
        "pilot_speaker_group_sha256": "4" * 64,
        "fixed_split_assignment": "train",
        "window_transcript_context": "look over there",
        "window_transcript_role": "aligned_speech_context_not_motion_label",
        "issues": [],
        "window": {
            "selection_status": ordinary.SEMANTIC_EVENT_SELECTION_STATUS,
            "start_frame": start,
            "end_frame_exclusive": end,
            "frame_count": end - start,
        },
        "training_segment": {
            "representation": ordinary.VARIABLE_SEGMENT_REPRESENTATION,
            "fixed_window_sec": fixed_window_sec,
            "start_frame": start,
            "end_frame_exclusive": end,
            "frame_count": end - start,
            "boundary_source": {
                "mode": "official_sem_core_plus_motion_low_speed_context"
            },
        },
    }


def _motion_foundation_record(source_clip_id="source_a"):
    clip_id = f"beat2_motion_foundation__{source_clip_id}_chunk0000_f000000-000300"
    return {
        "schema_version": "1.0.0",
        "clip_id": clip_id,
        "task_id": clip_id,
        "dataset": "BEAT2",
        "dataset_subset": "beat_english_v2.0.0",
        "language": "english",
        "language_code": "en",
        "source_clip_id": source_clip_id,
        "source_group_key": f"BEAT2/beat_english_v2.0.0/{source_clip_id}",
        "speaker_key": "1_wayne",
        "fixed_split_assignment": "train",
        "fps": 30.0,
        "motion_relpath": (
            f"beat_english_v2.0.0/smplxflame_30/{source_clip_id}.npz"
        ),
        "annotation_kind": grouped.MOTION_FOUNDATION_ANNOTATION_KIND,
        "semantic_label_status": "absent_motion_foundation",
        "semantic_supervision_masks": dict(grouped.MOTION_FOUNDATION_MASKS),
        "behavior_supervision_mask": False,
        "emotion_supervision_mask": False,
        "affect_observable_supervision_mask": False,
        "official_category_conditioning_enabled": False,
        "official_emotion_conditioning_enabled": False,
        "window": {
            "selection_status": ordinary.FULL_WINDOW_SELECTION_STATUS,
            "start_frame": 0,
            "end_frame_exclusive": 300,
            "frame_count": 300,
        },
        "training_segment": {
            "representation": grouped.MOTION_FOUNDATION_SEGMENT_REPRESENTATION,
            "start_frame": 0,
            "end_frame_exclusive": 300,
            "frame_count": 300,
            "fixed_window_sec": None,
            "overlap_frames": 0,
            "boundary_source": "source_container_frame_bounds",
        },
        "issues": [],
        "accepted_for_training": False,
        "training_admission_status": (
            "pending_18d_retarget_and_unchanged_physical_qc"
        ),
    }


def _dependencies(tmp_path):
    model = tmp_path / "model.npz"
    urdf = tmp_path / "robot.urdf"
    config = tmp_path / "retarget.json"
    gmr = tmp_path / "gmr"
    model.write_bytes(b"model")
    urdf.write_text("urdf", encoding="utf-8")
    config.write_text("{}", encoding="utf-8")
    gmr.mkdir()
    return model, urdf, config, gmr


def _args(tmp_path, inventory, beat2_root):
    model, urdf, config, gmr = _dependencies(tmp_path)
    return Namespace(
        inventory=inventory.resolve(),
        beat2_root=beat2_root.resolve(),
        output_root=(tmp_path / "output").resolve(),
        smplx_model=model.resolve(),
        gmr_root=gmr.resolve(),
        urdf=urdf.resolve(),
        config=config.resolve(),
        workers=1,
        limit_sources=None,
        limit_events=None,
        resume=False,
        retry_failed=False,
    )


def _quality(task, source_hash, *, passed=True):
    frame_count = task["end_frame_exclusive"] - task["start_frame"]
    return {
        **{
            field: task[field]
            for field in EVENT_REVIEW_PROVENANCE_FIELDS
            if field in task
        },
        "axis_policy": ordinary.BEAT2_AXIS_POLICY,
        "output_contract": ordinary.ULA_V2_18D_CONTRACT,
        "action_dim": 18,
        "joint_order": list(ordinary.JOINT_ORDER_18D),
        "fps": task["fps"],
        "source_window_frames": frame_count,
        "source_sha256": source_hash,
        "source_window_start_frame": task["start_frame"],
        "source_window_end_frame_exclusive": task["end_frame_exclusive"],
        "inventory_record_sha256": task["inventory_record_sha256"],
        "semantic_event": task["semantic_event"],
        "training_segment": task["training_segment"],
        "emotion_id": task["emotion_id"],
        "emotion_supervision_mask": task["emotion_supervision_mask"],
        "quality_gate": {
            "passed": passed,
            "joint_limits_pass": passed,
            "axis_direction_pass": passed,
        },
        "frames": frame_count,
        "duration_sec": frame_count / task["fps"],
        "retarget_segment": build_retarget_segment_contract(
            task,
            source_frame_count=frame_count,
            output_frame_count=frame_count,
            fps=task["fps"],
        ),
    }


class FakeRuntime:
    def __init__(self, config, trace, fail_ids=()):
        self.config = config
        self.trace = trace
        self.fail_ids = set(fail_ids)
        self.source_hash = None
        self.reset_ordinal = 0
        trace["runtime_initializations"] += 1

    def load_source(self, source):
        self.trace["source_loads"].append(str(Path(source).resolve()))
        self.source_hash = ordinary.sha256(Path(source))

    def reset_event(self, task):
        self.reset_ordinal += 1
        frame_count = task["end_frame_exclusive"] - task["start_frame"]
        self.trace["resets"].append((task["task_id"], frame_count))
        return {
            "event_reset_ordinal": self.reset_ordinal,
            "frame_count": frame_count,
        }

    def retarget_event(self, task, event, output_dir):
        self.trace["retargets"].append(
            (task["task_id"], event["event_reset_ordinal"], event["frame_count"])
        )
        if task["task_id"] in self.fail_ids:
            raise RuntimeError(f"synthetic failure for {task['task_id']}")
        output_dir.mkdir(parents=True, exist_ok=False)
        stem = f"{task['task_id']}_f{task['start_frame']:06d}-{task['end_frame_exclusive']:06d}"
        (output_dir / f"{stem}_gmr_raw_18d.csv").write_text(
            "joint\n0\n", encoding="utf-8"
        )
        (output_dir / f"{stem}_gmr_safe_18d.csv").write_text(
            "joint\n0\n", encoding="utf-8"
        )
        return _quality(task, self.source_hash)


def _read_tasks(inventory, root):
    eligible, excluded = grouped.read_semantic_inventory(inventory, root)
    assert not excluded
    return eligible


def test_source_group_loads_once_resets_each_variable_event_and_isolates_failure(
    tmp_path,
):
    beat2_root = tmp_path / "beat2"
    motion_root = beat2_root / "smplxflame_30"
    motion_root.mkdir(parents=True)
    (motion_root / "source_a.npz").write_bytes(b"one-source")
    records = [
        _semantic_record("event_short", "source_a", 3, 8),
        _semantic_record("event_bad", "source_a", 20, 29),
        _semantic_record("event_long", "source_a", 40, 53),
    ]
    inventory = tmp_path / "inventory.jsonl"
    _write_jsonl(inventory, records)
    args = _args(tmp_path, inventory, beat2_root)
    tasks = _read_tasks(inventory, beat2_root)
    trace = {
        "runtime_initializations": 0,
        "source_loads": [],
        "resets": [],
        "retargets": [],
    }

    def factory(config):
        return FakeRuntime(config, trace, fail_ids={"event_bad"})

    results = grouped.run_source_group(
        tasks,
        args,
        ordinary.sha256(inventory),
        "contract-sha",
        "run-1",
        runtime_factory=factory,
    )

    assert trace["runtime_initializations"] == 1
    assert trace["source_loads"] == [str((motion_root / "source_a.npz").resolve())]
    assert trace["resets"] == [
        ("event_short", 5),
        ("event_bad", 9),
        ("event_long", 13),
    ]
    assert [ordinal for _task, ordinal, _frames in trace["retargets"]] == [1, 2, 3]
    assert [result["status"] for result in results] == [
        "passed",
        "event_process_failed",
        "passed",
    ]
    assert results[2]["frames"] == 13
    assert results[2]["semantic_event"]["category"] == "deictic"
    assert results[2]["training_segment"]["fixed_window_sec"] is None
    assert results[2]["emotion_id"] == "angry"
    assert results[2]["canonical_prompt"] == records[2]["canonical_prompt"]
    assert results[2]["canonical_prompt_role"] == "coarse_category_only"
    assert results[2]["official_category_verified"] is True
    assert results[2]["robot_observable_motion_form"] == "candidate_unreviewed"
    assert results[2]["communicative_intent"] == "candidate_unreviewed"
    assert results[2]["semantic_supervision_masks"]["prompt_text"] is False
    assert results[2]["semantic_supervision_masks"]["legacy_gesture"] is False
    assert results[2]["canonical_action"] == "official_gesture_category:deictic"
    assert results[2]["official_category_role"] == (
        "verified_metadata_split_and_evaluation_only"
    )
    assert results[2]["official_category_condition_channel"] is None
    assert results[2]["official_category_loss"] is None
    assert results[2]["semantic_mapping_status"] == (
        "official_category_verified_metadata_only"
    )
    assert results[2]["prompt_schema"] == records[2]["prompt_schema"]
    assert results[2]["behavior_mapping_contract"] == records[2][
        "behavior_mapping_contract"
    ]
    assert results[2]["emotion_protocol_contract"] == records[2][
        "emotion_protocol_contract"
    ]
    assert results[2]["inventory_manifest_sha256"] == "1" * 64
    assert results[2]["fixed_split_assignment"] == "train"
    assert results[2]["retarget_segment"]["representation"] == (
        RETARGET_SEGMENT_REPRESENTATION
    )
    assert results[2]["retarget_segment"]["source_frame_count"] == 13
    assert results[2]["retarget_segment"]["output_frame_count"] == 13
    assert results[2]["retarget_segment"]["cropped"] is False
    assert results[2]["inventory_record_sha256"] == "5" * 64
    assert results[2]["upstream_inventory_record_sha256"] == "5" * 64
    assert results[2]["selected_record_sha256"] == ordinary.record_sha256(records[2])
    assert results[2]["retarget_input_manifest_sha256"] == ordinary.sha256(
        inventory
    )
    assert results[2]["source_manifest_sha256"] == ordinary.sha256(inventory)
    assert Path(results[1]["output_dir"]).is_dir()
    assert Path(results[2]["safe_csv"]).is_file()
    quality = json.loads(Path(results[2]["quality_json"]).read_text())
    assert quality["inventory_record_sha256"] == results[2][
        "inventory_record_sha256"
    ]
    assert quality["upstream_inventory_record_sha256"] == "5" * 64
    assert quality["selected_record_sha256"] == results[2][
        "selected_record_sha256"
    ]
    assert quality["semantic_event"]["category"] == "deictic"
    assert quality["training_segment"]["frame_count"] == 13
    assert quality["emotion_id"] == "angry"
    assert quality["canonical_prompt"] == records[2]["canonical_prompt"]
    assert quality["canonical_prompt_role"] == "coarse_category_only"
    assert quality["official_category_verified"] is True
    assert quality["robot_observable_motion_form"] == "candidate_unreviewed"
    assert quality["communicative_intent"] == "candidate_unreviewed"
    assert quality["semantic_supervision_masks"] == records[2][
        "semantic_supervision_masks"
    ]
    assert quality["prompt_schema"] == records[2]["prompt_schema"]
    assert quality["behavior_supervision_mask"] is False
    assert quality["emotion_review_status"] == "official_protocol_confirmed"
    assert quality["inventory_manifest_sha256"] == "1" * 64
    assert quality["motion_sha256"] == "a" * 64
    assert quality["source_sha256"] == ordinary.sha256(
        motion_root / "source_a.npz"
    )


def test_resume_skips_hash_valid_pass_and_requeues_stale_pass(tmp_path):
    beat2_root = tmp_path / "beat2"
    motion_root = beat2_root / "smplxflame_30"
    motion_root.mkdir(parents=True)
    (motion_root / "source_a.npz").write_bytes(b"one-source")
    record = _semantic_record("event_a", "source_a", 1, 7)
    inventory = tmp_path / "inventory.jsonl"
    _write_jsonl(inventory, [record])
    args = _args(tmp_path, inventory, beat2_root)
    task = _read_tasks(inventory, beat2_root)[0]
    trace = {
        "runtime_initializations": 0,
        "source_loads": [],
        "resets": [],
        "retargets": [],
    }
    grouped.run_source_group(
        [task],
        args,
        ordinary.sha256(inventory),
        "contract-sha",
        "run-1",
        runtime_factory=lambda config: FakeRuntime(config, trace),
    )

    assert grouped.select_runnable_tasks([task], args, "contract-sha") == []
    saved_path = ordinary.result_path(args.output_root, task)
    saved = ordinary.load_result(saved_path)
    saved["selected_record_sha256"] = "0" * 64
    saved_path.write_text(json.dumps(saved), encoding="utf-8")
    assert grouped.select_runnable_tasks([task], args, "contract-sha") == [task]
    saved["selected_record_sha256"] = task["selected_record_sha256"]
    saved_path.write_text(json.dumps(saved), encoding="utf-8")
    Path(saved["safe_csv"]).write_text("changed\n", encoding="utf-8")
    assert grouped.select_runnable_tasks([task], args, "contract-sha") == [task]


def test_formal_selector_row_requires_explicit_two_level_lineage(tmp_path):
    beat2_root = tmp_path / "beat2"
    motion_root = beat2_root / "smplxflame_30"
    motion_root.mkdir(parents=True)
    (motion_root / "source_a.npz").write_bytes(b"source")
    record = _semantic_record("event_a", "source_a", 1, 7)
    record.pop("upstream_inventory_record_sha256")
    inventory = tmp_path / "inventory.jsonl"
    _write_jsonl(inventory, [record])

    with pytest.raises(ValueError, match="upstream_inventory_record_sha256"):
        grouped.read_semantic_inventory(inventory, beat2_root)


def test_inventory_rejects_fixed_six_second_event_and_preserves_valid_metadata(
    tmp_path,
):
    beat2_root = tmp_path / "beat2"
    motion_root = beat2_root / "smplxflame_30"
    motion_root.mkdir(parents=True)
    (motion_root / "source_a.npz").write_bytes(b"source")
    valid = _semantic_record("event_variable", "source_a", 10, 81)
    fixed = _semantic_record(
        "event_fixed6", "source_a", 100, 280, fixed_window_sec=6.0
    )
    inventory = tmp_path / "inventory.jsonl"
    _write_jsonl(inventory, [valid, fixed])

    eligible, excluded = grouped.read_semantic_inventory(inventory, beat2_root)

    assert [task["task_id"] for task in eligible] == ["event_variable"]
    assert eligible[0]["end_frame_exclusive"] - eligible[0]["start_frame"] == 71
    assert eligible[0]["semantic_event"] == valid["semantic_event"]
    assert eligible[0]["emotion_supervision_mask"] is False
    assert len(excluded) == 1
    assert "semantic_event:fixed_window_forbidden" in excluded[0]["reasons"]


def test_group_and_limits_keep_source_as_scheduling_unit():
    tasks = [
        {"source": "/tmp/a.npz", "source_clip_id": "a", "task_id": "a1"},
        {"source": "/tmp/a.npz", "source_clip_id": "a", "task_id": "a2"},
        {"source": "/tmp/b.npz", "source_clip_id": "b", "task_id": "b1"},
    ]

    groups = grouped.group_tasks_by_source(tasks)
    selected = grouped.limit_groups(groups, limit_sources=2, limit_events=2)

    assert [[task["task_id"] for task in group] for group in groups] == [
        ["a1", "a2"],
        ["b1"],
    ]
    assert [[task["task_id"] for task in group] for group in selected] == [
        ["a1", "a2"]
    ]


def test_inventory_record_hash_is_bound_to_original_json(tmp_path):
    beat2_root = tmp_path / "beat2"
    motion_root = beat2_root / "smplxflame_30"
    motion_root.mkdir(parents=True)
    source = motion_root / "source_a.npz"
    source.write_bytes(b"source")
    record = _semantic_record("event_a", "source_a", 2, 12)
    inventory = tmp_path / "inventory.jsonl"
    _write_jsonl(inventory, [record])

    task = _read_tasks(inventory, beat2_root)[0]

    assert task["inventory_record_sha256"] == "5" * 64
    assert task["upstream_inventory_record_sha256"] == "5" * 64
    assert task["selected_record_sha256"] == hashlib.sha256(
        ordinary.stable_json(record).encode("utf-8")
    ).hexdigest()


def test_default_beat2_root_matches_dataset_prefixed_inventory_paths():
    assert grouped.DEFAULT_BEAT2_ROOT.name == "BEAT2"


def test_retarget_segment_contract_distinguishes_source_and_retimed_output():
    task = {"start_frame": 900, "end_frame_exclusive": 998}
    contract = build_retarget_segment_contract(
        task, source_frame_count=98, output_frame_count=124, fps=30.0
    )
    payload = {key: value for key, value in contract.items() if key != "sha256"}
    expected_hash = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()

    assert contract == {
        "representation": RETARGET_SEGMENT_REPRESENTATION,
        "source_start_frame": 900,
        "source_end_frame_exclusive": 998,
        "source_frame_count": 98,
        "source_frame_coverage_sec": 98 / 30.0,
        "output_frame_count": 124,
        "output_sample_span_sec": 123 / 30.0,
        "output_frame_coverage_sec": 124 / 30.0,
        "fps": 30.0,
        "retimed": True,
        "cropped": False,
        "planner_duration_field": "output_sample_span_sec",
        "source_boundary_duration_field": "source_frame_coverage_sec",
        "legacy_quality_duration_sec_role": (
            "output_frame_coverage_compatibility_only_not_planner_target"
        ),
        "sha256": expected_hash,
    }


def test_optional_retry_parameters_are_bound_to_runtime_and_contract(tmp_path):
    beat2_root = tmp_path / "beat2"
    beat2_root.mkdir()
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text("", encoding="utf-8")
    args = _args(tmp_path, inventory, beat2_root)
    args.solver = "quadprog"
    args.neutral_limit_margin_rad = 1e-6
    args.smoothing_window = 1
    args.posture_cost = 0.0

    runtime = grouped.runtime_config(args)
    contract, _contract_hash = grouped.build_run_contract(args)

    assert runtime.solver == "quadprog"
    assert runtime.neutral_limit_margin_rad == 1e-6
    assert runtime.smoothing_window == 1
    assert runtime.posture_cost == 0.0
    assert contract["retarget_parameters"]["solver"] == "quadprog"
    assert contract["retarget_parameters"]["neutral_limit_margin_rad"] == 1e-6
    assert contract["retarget_parameters"]["smoothing_window"] == 1
    assert contract["retarget_parameters"]["posture_cost"] == 0.0
    assert contract["quality_policy"] == ordinary.QUALITY_POLICY


def test_interiorize_neutral_qpos_only_moves_exact_limited_boundaries():
    class FakeModel:
        jnt_limited = np.asarray([True, True, True])
        jnt_range = np.asarray([[-1.0, 1.0], [0.0, 1.0], [-2.0, 2.0]])
        jnt_qposadr = np.asarray([0, 1, 2])

    class FakeMujoco:
        class mjtObj:
            mjOBJ_JOINT = 1

        @staticmethod
        def mj_name2id(_model, _kind, name):
            return {
                "joint_pelvisYaw": 0,
                "joint_pelvisPitch": 1,
                "joint_pelvisRoll": 2,
            }.get(name, -1)

    result = interiorize_neutral_qpos(
        FakeModel(),
        FakeMujoco(),
        np.asarray([0.25, 0.0, -0.5]),
        1e-6,
    )

    assert result.tolist() == [0.25, 1e-6, -0.5]


def test_motion_foundation_inventory_is_unlabeled_and_uses_distinct_contract(
    tmp_path,
):
    beat2_root = tmp_path / "BEAT2"
    motion_root = beat2_root / "beat_english_v2.0.0/smplxflame_30"
    motion_root.mkdir(parents=True)
    (motion_root / "source_a.npz").write_bytes(b"source")
    record = _motion_foundation_record()
    inventory = tmp_path / "foundation.jsonl"
    _write_jsonl(inventory, [record])

    eligible, excluded = grouped.read_semantic_inventory(inventory, beat2_root)

    assert excluded == []
    assert len(eligible) == 1
    task = eligible[0]
    assert task["semantic_supervision_masks"] == grouped.MOTION_FOUNDATION_MASKS
    assert "canonical_prompt" not in task
    assert "audio_relpath" not in task
    assert "source_speech_context" not in task
    contract = build_retarget_segment_contract(
        task, source_frame_count=300, output_frame_count=300, fps=30.0
    )
    assert (
        contract["representation"]
        == MOTION_FOUNDATION_RETARGET_SEGMENT_REPRESENTATION
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("canonical_prompt", {"en": "leaked"}),
        ("prompt", "leaked"),
        ("source_text", "leaked"),
        ("window_transcript_context", "leaked"),
        ("audio_relpath", "beat_english_v2.0.0/wave16k/source_a.wav"),
        ("semantic_event", {"category": "deictic"}),
        ("behavior_id", "Behavior.InteractPresence"),
        ("emotion_id", "happy"),
    ],
)
def test_motion_foundation_inventory_rejects_conditioning_metadata(
    tmp_path, field, value
):
    beat2_root = tmp_path / "BEAT2"
    motion_root = beat2_root / "beat_english_v2.0.0/smplxflame_30"
    motion_root.mkdir(parents=True)
    (motion_root / "source_a.npz").write_bytes(b"source")
    if field == "audio_relpath":
        audio = beat2_root / str(value)
        audio.parent.mkdir(parents=True)
        audio.write_bytes(b"audio")
    record = _motion_foundation_record()
    record[field] = value
    inventory = tmp_path / f"foundation_{field}.jsonl"
    _write_jsonl(inventory, [record])

    eligible, excluded = grouped.read_semantic_inventory(inventory, beat2_root)

    assert eligible == []
    assert len(excluded) == 1
    assert (
        "motion_foundation:conditioning_or_audio_metadata_forbidden"
        in excluded[0]["reasons"]
    )
