import hashlib
import json
import sys
from pathlib import Path

import pytest

from tools.gmr_v2 import batch_retarget_beat2_v2 as batch


def _record(clip_id, *, status=batch.SELECTION_STATUS, issues=None):
    return {
        "clip_id": clip_id,
        "fps": 30.0,
        "motion_relpath": f"smplxflame_30/{clip_id}.npz",
        "official_split": "train",
        "speaker_key": "12_zhao",
        "interaction_label": "co_speech_conversational_gesture",
        "window_transcript_context": f"speech for {clip_id}",
        "issues": issues or [],
        "window": {
            "selection_status": status,
            "start_frame": 12,
            "end_frame_exclusive": 192,
        },
    }


def _semantic_event_record(clip_id="event_a"):
    record = _record(
        clip_id,
        status=batch.SEMANTIC_EVENT_SELECTION_STATUS,
    )
    record.update(
        {
            "dataset_subset": "beat_english_v2.0.0",
            "language": "english",
            "annotation_kind": "official_gesture_semantics",
            "semantic_label_status": "official_semantic_event_pending_retarget_qc",
            "semantic_gesture": "pointing",
            "official_semantic_event": {
                "source_label": "04_deictic_h",
                "category": "deictic",
                "intensity": "high",
                "lexical_anchor": "there",
            },
            "training_segment": {
                "representation": batch.VARIABLE_SEGMENT_REPRESENTATION,
                "fixed_window_sec": None,
                "boundary_source": "official_semantic_core_plus_motion_envelope",
            },
            "emotion_id": "angry",
            "emotion_supervision_mask": True,
            "emotion_label_status": "official_beat_filename_protocol_mapped",
            "emotion_source": "official_beat_recording_protocol",
            "source_emotion_label": "anger",
            "window_transcript_role": "aligned_speech_context_not_motion_label",
        }
    )
    record["window"].update(
        {
            "start_frame": 31,
            "end_frame_exclusive": 104,
        }
    )
    return record


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _touch_dependencies(tmp_path):
    paths = {}
    for name in ("model", "gmr", "urdf", "config"):
        path = tmp_path / name
        if name == "gmr":
            path.mkdir()
        else:
            path.write_text(name, encoding="utf-8")
        paths[name] = path
    return paths


def _fake_retarget(path, *, passed):
    path.write_text(
        f"""\
import argparse, hashlib, json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--beat2', type=Path)
p.add_argument('--start-frame', type=int)
p.add_argument('--end-frame', type=int)
p.add_argument('--output-dir', type=Path)
a,_=p.parse_known_args()
a.output_dir.mkdir(parents=True, exist_ok=True)
stem=f'{{a.beat2.stem}}_f{{a.start_frame:06d}}-{{a.end_frame:06d}}'
(a.output_dir/f'{{stem}}_gmr_raw_18d.csv').write_text('time,neck_yaw\\n0,0\\n', encoding='utf-8')
(a.output_dir/f'{{stem}}_gmr_safe_18d.csv').write_text('time,neck_yaw\\n0,0\\n', encoding='utf-8')
source_hash=hashlib.sha256(a.beat2.read_bytes()).hexdigest()
quality={{
  'axis_policy': {batch.BEAT2_AXIS_POLICY!r},
  'output_contract': {batch.ULA_V2_18D_CONTRACT!r},
  'action_dim': 18,
  'joint_order': {list(batch.JOINT_ORDER_18D)!r},
  'fps': 30.0,
  'source_window_frames': a.end_frame-a.start_frame,
  'source_sha256': source_hash,
  'source_window_start_frame': a.start_frame,
  'source_window_end_frame_exclusive': a.end_frame,
  'quality_gate': {{'passed': {passed!r}, 'joint_limits_pass': {passed!r}, 'axis_direction_pass': {passed!r}}},
  'frames': 180,
  'duration_sec': 6.0,
}}
(a.output_dir/'quality.json').write_text(json.dumps(quality), encoding='utf-8')
""",
        encoding="utf-8",
    )


def _main_args(tmp_path, inventory, beat2_root, output_root, fake_script):
    dependencies = _touch_dependencies(tmp_path)
    return [
        "--inventory", str(inventory),
        "--beat2-root", str(beat2_root),
        "--output-root", str(output_root),
        "--smplx-model", str(dependencies["model"]),
        "--gmr-root", str(dependencies["gmr"]),
        "--urdf", str(dependencies["urdf"]),
        "--config", str(dependencies["config"]),
        "--retarget-script", str(fake_script),
        "--python", sys.executable,
        "--workers", "1",
    ]


def _arg_path(args, option):
    return Path(args[args.index(option) + 1])


def test_inventory_selects_aligned_windows_and_allows_only_declared_warnings(tmp_path):
    root = tmp_path / "beat2"
    motion = root / "smplxflame_30"
    motion.mkdir(parents=True)
    records = [
        _record("eligible_clean"),
        _record("eligible_warning", issues=["textgrid_transcript_mismatch"]),
        _record("blocked", issues=["missing_textgrid_alignment"]),
        _record("wrong_status", status="selected_nonstatic_low_dynamic"),
    ]
    for record in records:
        (motion / f"{record['clip_id']}.npz").write_bytes(b"npz")
    inventory = tmp_path / "inventory.jsonl"
    _write_jsonl(inventory, records)

    eligible, excluded = batch.read_inventory(inventory, root)

    assert [task["clip_id"] for task in eligible] == [
        "eligible_clean", "eligible_warning"
    ]
    assert eligible[1]["source_warnings"] == ["textgrid_transcript_mismatch"]
    assert eligible[0]["task_id"] == "eligible_clean_f000012-000192"
    assert eligible[0]["source_clip_id"] == "eligible_clean"
    assert eligible[0]["source_speech_context"] == "speech for eligible_clean"
    assert eligible[0]["source_speech_context_role"] == "speech_context_only_not_action_label"
    assert {item["clip_id"] for item in excluded} == {"blocked", "wrong_status"}
    assert all(item["source_clip_id"] == item["clip_id"] for item in excluded)
    assert all(item["official_split"] == "train" for item in excluded)
    assert all(item["speaker_key"] == "12_zhao" for item in excluded)
    assert all(item["task_id"].endswith("_f000012-000192") for item in excluded)
    assert all(task["accepted_for_training"] is False for task in eligible)


def test_inventory_preserves_unique_window_and_shared_source_identity(tmp_path):
    root = tmp_path / "beat2"
    motion = root / "smplxflame_30"
    motion.mkdir(parents=True)
    (motion / "source_clip.npz").write_bytes(b"npz")
    records = []
    for index, start in enumerate((0, 180)):
        record = _record(f"source_clip_window_{index}")
        record.update(
            {
                "source_clip_id": "source_clip",
                "source_group_key": "speaker/session/source_clip",
                "task_id": f"source_clip_f{start:06d}-{start + 180:06d}",
                "motion_relpath": "smplxflame_30/source_clip.npz",
            }
        )
        record["window"].update(
            {"start_frame": start, "end_frame_exclusive": start + 180}
        )
        records.append(record)
    inventory = tmp_path / "inventory.jsonl"
    _write_jsonl(inventory, records)

    eligible, excluded = batch.read_inventory(inventory, root)

    assert not excluded
    assert [row["clip_id"] for row in eligible] == [
        "source_clip_window_0",
        "source_clip_window_1",
    ]
    assert {row["source_clip_id"] for row in eligible} == {"source_clip"}
    assert {row["source_group_key"] for row in eligible} == {
        "speaker/session/source_clip"
    }
    assert [row["task_id"] for row in eligible] == [
        "source_clip_f000000-000180",
        "source_clip_f000180-000360",
    ]


def test_inventory_accepts_boundary_validated_full_windows(tmp_path):
    root = tmp_path / "beat2"
    motion = root / "smplxflame_30"
    motion.mkdir(parents=True)
    (motion / "source_clip.npz").write_bytes(b"npz")
    record = _record(
        "source_clip_f000000-000180",
        status=batch.FULL_WINDOW_SELECTION_STATUS,
    )
    record.update(
        {
            "source_clip_id": "source_clip",
            "source_group_id": "source_clip",
            "task_id": "source_clip_f000000-000180",
            "motion_relpath": "smplxflame_30/source_clip.npz",
        }
    )
    record["window"].update({"start_frame": 0, "end_frame_exclusive": 180})
    inventory = tmp_path / "inventory.jsonl"
    _write_jsonl(inventory, [record])

    eligible, excluded = batch.read_inventory(inventory, root)

    assert not excluded
    assert eligible[0]["clip_id"] == "source_clip_f000000-000180"
    assert eligible[0]["task_id"] == "source_clip_f000000-000180"
    assert eligible[0]["source_clip_id"] == "source_clip"
    assert eligible[0]["source_group_key"] == "source_clip"


def test_inventory_preserves_variable_semantic_event_evidence(tmp_path):
    root = tmp_path / "beat2"
    motion = root / "smplxflame_30"
    motion.mkdir(parents=True)
    (motion / "event_a.npz").write_bytes(b"npz")
    record = _semantic_event_record()
    inventory = tmp_path / "inventory.jsonl"
    _write_jsonl(inventory, [record])

    eligible, excluded = batch.read_inventory(inventory, root)

    assert not excluded
    assert len(eligible) == 1
    task = eligible[0]
    assert task["start_frame"] == 31
    assert task["end_frame_exclusive"] == 104
    assert task["training_segment"]["fixed_window_sec"] is None
    assert task["official_semantic_event"]["category"] == "deictic"
    assert task["emotion_id"] == "angry"
    assert task["emotion_supervision_mask"] is True
    assert task["source_speech_context_role"] == (
        "aligned_speech_context_not_motion_label"
    )


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    (
        ("training_segment", None, "missing_training_segment"),
        (
            "representation",
            "fixed_resampled_window",
            "invalid_variable_length_representation",
        ),
        ("fixed_window_sec", 6.0, "fixed_window_forbidden"),
        ("boundary_source", "", "missing_boundary_source"),
    ),
)
def test_inventory_excludes_invalid_formal_semantic_event_contract(
    tmp_path, field, value, reason
):
    root = tmp_path / "beat2"
    motion = root / "smplxflame_30"
    motion.mkdir(parents=True)
    (motion / "event_a.npz").write_bytes(b"npz")
    record = _semantic_event_record()
    if field == "training_segment":
        record[field] = value
    else:
        record["training_segment"][field] = value
    inventory = tmp_path / "inventory.jsonl"
    _write_jsonl(inventory, [record])

    eligible, excluded = batch.read_inventory(inventory, root)

    assert not eligible
    assert len(excluded) == 1
    assert f"semantic_event:{reason}" in excluded[0]["reasons"]


@pytest.mark.parametrize(
    ("field", "outside_name"),
    (("motion_relpath", "outside.npz"), ("audio_relpath", "outside.wav")),
)
def test_inventory_rejects_motion_and_audio_paths_outside_beat2_root(
    tmp_path, field, outside_name
):
    root = tmp_path / "beat2"
    motion = root / "smplxflame_30"
    motion.mkdir(parents=True)
    (motion / "clip.npz").write_bytes(b"npz")
    (tmp_path / outside_name).write_bytes(b"outside")
    record = _record("clip")
    record[field] = f"../{outside_name}"
    inventory = tmp_path / "inventory.jsonl"
    _write_jsonl(inventory, [record])

    with pytest.raises(ValueError, match=rf"{field}.*escapes BEAT2 root"):
        batch.read_inventory(inventory, root)


def test_inventory_rejects_symlink_escape_from_beat2_root(tmp_path):
    root = tmp_path / "beat2"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "clip.npz").write_bytes(b"outside")
    (root / "linked").symlink_to(outside, target_is_directory=True)
    record = _record("clip")
    record["motion_relpath"] = "linked/clip.npz"
    inventory = tmp_path / "inventory.jsonl"
    _write_jsonl(inventory, [record])

    with pytest.raises(ValueError, match="motion_relpath.*escapes BEAT2 root"):
        batch.read_inventory(inventory, root)


def test_inventory_does_not_require_contained_audio_for_motion_only_run(tmp_path):
    root = tmp_path / "beat2"
    motion = root / "smplxflame_30"
    motion.mkdir(parents=True)
    (motion / "clip.npz").write_bytes(b"npz")
    record = _record("clip")
    record["audio_relpath"] = "wave16k/not_downloaded_yet.wav"
    inventory = tmp_path / "inventory.jsonl"
    _write_jsonl(inventory, [record])

    eligible, excluded = batch.read_inventory(inventory, root)

    assert not excluded
    assert eligible[0]["audio_relpath"] == "wave16k/not_downloaded_yet.wav"


def test_quality_check_requires_every_strict_gate_and_matching_provenance(tmp_path):
    source = tmp_path / "source.npz"
    source.write_bytes(b"source")
    source_hash = hashlib.sha256(b"source").hexdigest()
    task = {"start_frame": 12, "end_frame_exclusive": 192, "fps": 30.0}
    quality = {
        "axis_policy": batch.BEAT2_AXIS_POLICY,
        "output_contract": batch.ULA_V2_18D_CONTRACT,
        "action_dim": 18,
        "joint_order": list(batch.JOINT_ORDER_18D),
        "fps": 30.0,
        "source_window_frames": 180,
        "frames": 180,
        "source_sha256": source_hash,
        "source_window_start_frame": 12,
        "source_window_end_frame_exclusive": 192,
        "quality_gate": {
            "passed": True,
            "joint_limits_pass": True,
            "axis_direction_pass": True,
        },
    }

    assert batch.quality_passes(quality, task, source_hash)
    quality["quality_gate"]["axis_direction_pass"] = False
    assert not batch.quality_passes(quality, task, source_hash)
    quality["quality_gate"]["axis_direction_pass"] = 1
    assert not batch.quality_passes(quality, task, source_hash)
    quality["quality_gate"]["axis_direction_pass"] = True
    assert not batch.quality_passes(quality, task, "different-hash")


def test_runner_writes_atomic_state_manifests_and_resumes_pass(tmp_path):
    beat2_root = tmp_path / "beat2"
    motion = beat2_root / "smplxflame_30"
    motion.mkdir(parents=True)
    record = _record("clip_a")
    (motion / "clip_a.npz").write_bytes(b"source-a")
    (motion / "clip_pending.npz").write_bytes(b"source-pending")
    (motion / "clip_excluded.npz").write_bytes(b"source-excluded")
    inventory = tmp_path / "inventory.jsonl"
    _write_jsonl(inventory, [
        record,
        _record("clip_pending"),
        _record("clip_excluded", issues=["missing_textgrid_alignment"]),
    ])
    output_root = tmp_path / "out"
    fake_script = tmp_path / "fake_retarget.py"
    _fake_retarget(fake_script, passed=True)
    args = [
        *_main_args(tmp_path, inventory, beat2_root, output_root, fake_script),
        "--limit", "1",
    ]

    assert batch.main(args) == 0
    status = json.loads((output_root / "status.json").read_text(encoding="utf-8"))
    passed = [json.loads(line) for line in (output_root / "passed_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    result_before = passed[0]

    assert status["run_state"] == "finished"
    assert status["eligible_task_count"] == 2
    assert status["excluded_task_count"] == 1
    assert status["counts"] == {"passed": 1}
    assert status["pending_count"] == 1
    assert status["coverage_complete"] is True
    contract_document = json.loads(
        (output_root / batch.RUN_CONTRACT_FILENAME).read_text(encoding="utf-8")
    )
    assert status["run_contract"] == contract_document["run_contract"]
    assert status["run_contract_sha256"] == contract_document["run_contract_sha256"]
    assert result_before["run_contract_sha256"] == status["run_contract_sha256"]
    assert status["run_contract"]["artifacts"]["robot_urdf"]["sha256"]
    assert status["run_contract"]["artifacts"]["smplx_model"]["sha256"]
    assert status["run_contract"]["artifacts"]["retarget_config"]["sha256"]
    assert status["run_contract"]["artifacts"]["python_interpreter"]["sha256"]
    assert status["run_contract"]["quality_policy"] == batch.QUALITY_POLICY
    assert result_before["accepted_for_training"] is False
    assert result_before["source_manifest_sha256"] == batch.sha256(inventory)
    assert result_before["safe_csv_sha256"] == batch.sha256(Path(result_before["safe_csv"]))
    assert (output_root / "failed_manifest.jsonl").read_text(encoding="utf-8") == ""
    pending = json.loads((output_root / "pending_manifest.jsonl").read_text(encoding="utf-8"))
    excluded = json.loads((output_root / "excluded_manifest.jsonl").read_text(encoding="utf-8"))
    assert pending["clip_id"] == "clip_pending"
    assert pending["status"] == "pending"
    assert excluded["clip_id"] == "clip_excluded"
    assert excluded["status"] == "excluded"

    assert batch.main([*args, "--resume"]) == 0
    result_after = json.loads((output_root / "passed_manifest.jsonl").read_text(encoding="utf-8"))
    assert result_after["run_id"] == result_before["run_id"]
    assert not (output_root / "superseded").exists()


def test_failed_result_is_retried_only_when_requested(tmp_path):
    beat2_root = tmp_path / "beat2"
    motion = beat2_root / "smplxflame_30"
    motion.mkdir(parents=True)
    (motion / "clip_b.npz").write_bytes(b"source-b")
    inventory = tmp_path / "inventory.jsonl"
    _write_jsonl(inventory, [_record("clip_b")])
    output_root = tmp_path / "out"
    fake_script = tmp_path / "fake_retarget.py"
    _fake_retarget(fake_script, passed=False)
    args = _main_args(tmp_path, inventory, beat2_root, output_root, fake_script)

    assert batch.main(args) == 0
    failed_before = json.loads((output_root / "failed_manifest.jsonl").read_text(encoding="utf-8"))
    assert failed_before["status"] == "quality_failed"

    assert batch.main([*args, "--resume"]) == 0
    still_failed = json.loads((output_root / "failed_manifest.jsonl").read_text(encoding="utf-8"))
    assert still_failed["run_id"] == failed_before["run_id"]

    assert batch.main([*args, "--resume", "--retry-failed"]) == 0
    failed_after = json.loads((output_root / "failed_manifest.jsonl").read_text(encoding="utf-8"))
    assert failed_after["status"] == "quality_failed"
    assert failed_after["run_id"] != failed_before["run_id"]


@pytest.mark.parametrize(
    "artifact_option",
    ("--retarget-script", "--smplx-model", "--urdf", "--config", "--gmr-root"),
)
def test_resume_refuses_artifact_contract_drift(tmp_path, artifact_option):
    beat2_root = tmp_path / "beat2"
    motion = beat2_root / "smplxflame_30"
    motion.mkdir(parents=True)
    (motion / "clip.npz").write_bytes(b"source")
    inventory = tmp_path / "inventory.jsonl"
    _write_jsonl(inventory, [_record("clip")])
    output_root = tmp_path / "out"
    fake_script = tmp_path / "fake_retarget.py"
    _fake_retarget(fake_script, passed=True)
    args = _main_args(tmp_path, inventory, beat2_root, output_root, fake_script)
    artifact = _arg_path(args, artifact_option)
    if artifact.is_dir():
        artifact = artifact / "implementation.py"
        artifact.write_text("before\n", encoding="utf-8")

    assert batch.main(args) == 0
    artifact.write_text(artifact.read_text(encoding="utf-8") + "after\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="run contract changed"):
        batch.main([*args, "--resume"])


def test_resume_refuses_python_interpreter_contract_drift(tmp_path):
    beat2_root = tmp_path / "beat2"
    motion = beat2_root / "smplxflame_30"
    motion.mkdir(parents=True)
    (motion / "clip.npz").write_bytes(b"source")
    inventory = tmp_path / "inventory.jsonl"
    _write_jsonl(inventory, [_record("clip")])
    output_root = tmp_path / "out"
    fake_script = tmp_path / "fake_retarget.py"
    _fake_retarget(fake_script, passed=True)
    args = _main_args(tmp_path, inventory, beat2_root, output_root, fake_script)
    wrapper = tmp_path / "python-wrapper"
    wrapper.write_text(f"#!/bin/sh\nexec {sys.executable} \"$@\"\n", encoding="utf-8")
    wrapper.chmod(0o755)
    args[args.index("--python") + 1] = str(wrapper)

    assert batch.main(args) == 0
    wrapper.write_text(wrapper.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="run contract changed"):
        batch.main([*args, "--resume"])


def test_resume_refuses_implementation_and_quality_policy_drift(
    tmp_path, monkeypatch
):
    beat2_root = tmp_path / "beat2"
    motion = beat2_root / "smplxflame_30"
    motion.mkdir(parents=True)
    (motion / "clip.npz").write_bytes(b"source")
    inventory = tmp_path / "inventory.jsonl"
    _write_jsonl(inventory, [_record("clip")])
    fake_script = tmp_path / "fake_retarget.py"
    _fake_retarget(fake_script, passed=True)

    implementation = tmp_path / "implementation.py"
    implementation.write_text("before\n", encoding="utf-8")
    monkeypatch.setattr(batch, "RETARGET_IMPLEMENTATION", implementation)
    original_quality_policy = batch.QUALITY_POLICY
    original_axis_policy = batch.BEAT2_AXIS_POLICY
    implementation_output = tmp_path / "implementation-out"
    implementation_args = _main_args(
        tmp_path, inventory, beat2_root, implementation_output, fake_script
    )
    assert batch.main(implementation_args) == 0
    implementation.write_text("after\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="run contract changed"):
        batch.main([*implementation_args, "--resume"])

    implementation.write_text("before\n", encoding="utf-8")
    policy_output = tmp_path / "policy-out"
    policy_args = list(implementation_args)
    policy_args[policy_args.index("--output-root") + 1] = str(policy_output)
    assert batch.main(policy_args) == 0
    changed_policy = dict(batch.QUALITY_POLICY)
    changed_policy["limb_target_error_p95_max_m"] = 0.05
    monkeypatch.setattr(batch, "QUALITY_POLICY", changed_policy)
    with pytest.raises(RuntimeError, match="run contract changed"):
        batch.main([*policy_args, "--resume"])

    monkeypatch.setattr(batch, "QUALITY_POLICY", original_quality_policy)
    axis_output = tmp_path / "axis-out"
    axis_args = list(implementation_args)
    axis_args[axis_args.index("--output-root") + 1] = str(axis_output)
    assert batch.main(axis_args) == 0
    monkeypatch.setattr(batch, "BEAT2_AXIS_POLICY", "changed_axis_policy")
    with pytest.raises(RuntimeError, match="run contract changed"):
        batch.main([*axis_args, "--resume"])

    monkeypatch.setattr(batch, "BEAT2_AXIS_POLICY", original_axis_policy)
    parameter_output = tmp_path / "parameter-out"
    parameter_args = list(implementation_args)
    parameter_args[parameter_args.index("--output-root") + 1] = str(parameter_output)
    assert batch.main(parameter_args) == 0
    changed_parameters = dict(batch.RETARGET_PARAMETERS)
    changed_parameters["smoothing_window"] = 9
    monkeypatch.setattr(batch, "RETARGET_PARAMETERS", changed_parameters)
    with pytest.raises(RuntimeError, match="run contract changed"):
        batch.main([*parameter_args, "--resume"])


def test_resume_refuses_result_with_missing_contract_binding(tmp_path):
    beat2_root = tmp_path / "beat2"
    motion = beat2_root / "smplxflame_30"
    motion.mkdir(parents=True)
    (motion / "clip.npz").write_bytes(b"source")
    inventory = tmp_path / "inventory.jsonl"
    _write_jsonl(inventory, [_record("clip")])
    output_root = tmp_path / "out"
    fake_script = tmp_path / "fake_retarget.py"
    _fake_retarget(fake_script, passed=True)
    args = _main_args(tmp_path, inventory, beat2_root, output_root, fake_script)
    assert batch.main(args) == 0
    result_file = next((output_root / "state/results").glob("*.json"))
    result = json.loads(result_file.read_text(encoding="utf-8"))
    result.pop("run_contract_sha256")
    result_file.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(RuntimeError, match="Saved retarget result contract"):
        batch.main([*args, "--resume"])


def test_python_argument_keeps_virtualenv_symlink_path(tmp_path):
    real_python = tmp_path / "real-python"
    real_python.write_text("python", encoding="utf-8")
    venv_python = tmp_path / "venv-python"
    venv_python.symlink_to(real_python)

    assert batch.executable_path(venv_python) == venv_python
    assert batch.executable_path(venv_python) != real_python
