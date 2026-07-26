import json
import hashlib
from pathlib import Path
import sys

import pytest

from tools.human_motion_collection import run_beat2_18d_posttrain_pipeline as pipeline


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def make_complete_retarget(tmp_path: Path):
    output_root = tmp_path / "processed"
    retarget = output_root / "retarget"
    passed_dir = retarget / "passed/pass_clip"
    passed_dir.mkdir(parents=True)
    safe_csv = passed_dir / "pass_clip_gmr_safe_18d.csv"
    header = ",".join(pipeline.JOINT_ORDER_18D)
    row = ",".join("0" for _ in pipeline.JOINT_ORDER_18D)
    safe_csv.write_text(f"{header}\n{row}\n{row}\n", encoding="utf-8")
    gate = {key: True for key in pipeline.QUALITY_GATE_KEYS}
    quality = {
        "output_contract": pipeline.RETARGET_CONTRACT,
        "action_dim": 18,
        "joint_order": list(pipeline.JOINT_ORDER_18D),
        "quality_gate": gate,
    }
    quality_path = passed_dir / "quality.json"
    write_json(quality_path, quality)

    inventory = tmp_path / "inventory.jsonl"
    inventory_records = [{"task_id": "pass_clip"}, {"task_id": "failed_clip"}]
    write_jsonl(inventory, inventory_records)
    passed_record = {
        "task_id": "pass_clip",
        "clip_id": "pass_clip",
        "status": "passed",
        "retarget_contract": pipeline.RETARGET_CONTRACT,
        "quality_gate": gate,
        "safe_csv": str(safe_csv),
        "safe_csv_sha256": pipeline.sha256_file(safe_csv),
        "quality_json": str(quality_path),
        "quality_json_sha256": pipeline.sha256_file(quality_path),
        "speaker_key": "speaker_a",
        "frames": 2,
        "duration_sec": 2 / 30,
    }
    failed_record = {"task_id": "failed_clip", "status": "quality_failed"}
    write_jsonl(retarget / "passed_manifest.jsonl", [passed_record])
    write_jsonl(retarget / "failed_manifest.jsonl", [failed_record])
    write_jsonl(retarget / "pending_manifest.jsonl", [])
    write_jsonl(retarget / "excluded_manifest.jsonl", [])
    write_json(
        retarget / "status.json",
        {
            "run_state": "finished",
            "coverage_complete": True,
            "pending_count": 0,
            "inventory_record_count": 2,
            "eligible_task_count": 2,
            "saved_result_count": 2,
            "inventory_sha256": pipeline.sha256_file(inventory),
            "output_contract": pipeline.RETARGET_CONTRACT,
            "counts": {"passed": 1, "quality_failed": 1},
        },
    )
    return output_root, inventory, safe_csv


def pipeline_argv(tmp_path: Path, output_root: Path, inventory: Path, *extra: str):
    base = tmp_path / "base.pt"
    qwen = tmp_path / "qwen.pt"
    split = tmp_path / "split.pt"
    dataset = tmp_path / "kimodo"
    for path in (base, qwen, split):
        path.write_bytes(path.name.encode("ascii"))
    dataset.mkdir(exist_ok=True)
    (dataset / "data.parquet").write_bytes(b"dataset")
    return [
        "--inventory",
        str(inventory),
        "--output-root",
        str(output_root),
        "--python",
        sys.executable,
        "--base-checkpoint",
        str(base),
        "--qwen-checkpoint",
        str(qwen),
        "--kimodo-dataset-dir",
        str(dataset),
        "--kimodo-split-checkpoint",
        str(split),
        "--posttrain-output-dir",
        str(tmp_path / "posttrain"),
        "--expected-inventory-count",
        "2",
        "--expected-passed-count",
        "1",
        "--expected-failed-count",
        "1",
        "--allow-unreviewed",
        *extra,
    ]


def make_complete_labels(output_root: Path) -> None:
    passed_path = output_root / "retarget/passed_manifest.jsonl"
    source = json.loads(passed_path.read_text(encoding="utf-8").splitlines()[0])
    annotations = output_root / "annotations"
    prompt = {"en": "Move both arms.", "zh": "Move both arms (zh placeholder)."}
    draft = {
        "task_id": source["task_id"],
        "trajectory_path": source["safe_csv"],
        "trajectory_sha256": source["safe_csv_sha256"],
        "quality_json": source["quality_json"],
        "quality_json_sha256": source["quality_json_sha256"],
        "retarget_quality_gate": source["quality_gate"],
        "robot_contract": pipeline.RETARGET_CONTRACT,
        "speaker_key": source["speaker_key"],
        "accepted_for_training": False,
        "manual_human_review_required": True,
        "decision": "needs_human_review",
        "prompt_provenance": "trajectory_only_ula_v2_18d_features_no_speech_semantics",
        "canonical_prompt": prompt,
    }
    write_jsonl(annotations / "draft_prompts.jsonl", [draft])
    write_jsonl(annotations / "needs_human_review.jsonl", [draft])
    write_jsonl(annotations / "rejected.jsonl", [])
    write_json(
        annotations / "summary.json",
        {
            "input_records": 1,
            "draft_records": 1,
            "rejected_records": 0,
            "prompt_provenance": (
                "trajectory_only_ula_v2_18d_features_no_speech_semantics"
            ),
        },
    )
    for language in ("en", "zh"):
        text = annotations / f"text/{language}/{source['task_id']}.txt"
        text.parent.mkdir(parents=True, exist_ok=True)
        text.write_text(prompt[language] + "\n", encoding="utf-8")


def test_full_dry_run_is_deterministic_audio_disabled_and_write_free(
    tmp_path, monkeypatch, capsys
):
    output_root, inventory, _ = make_complete_retarget(tmp_path)
    argv = pipeline_argv(tmp_path, output_root, inventory, "--dry-run")

    def forbidden_subprocess(*args, **kwargs):
        raise AssertionError("dry-run launched a subprocess")

    monkeypatch.setattr(pipeline.subprocess, "run", forbidden_subprocess)
    assert pipeline.main(argv) == 0
    first = json.loads(capsys.readouterr().out)
    assert pipeline.main(argv) == 0
    second = json.loads(capsys.readouterr().out)

    assert first["plan_sha256"] == second["plan_sha256"]
    assert first["audio_policy"] == "disabled_not_loaded"
    assert first["requested_stages"] == list(pipeline.STAGES)
    assert [stage["stage"] for stage in first["stages"]] == list(pipeline.STAGES)
    assert first["stages"][0]["action"] == "run"
    assert first["stages"][2]["action"] == "run_after_dependencies"
    commands = json.dumps([stage["command"] for stage in first["stages"]])
    assert "pseudolabel" not in commands
    assert "--allow-download" not in commands
    assert not (output_root / "posttrain_pipeline").exists()


def test_verify_stage_receipt_reuses_exact_contract_and_rejects_manifest_drift(
    tmp_path, monkeypatch
):
    output_root, inventory, _ = make_complete_retarget(tmp_path)
    argv = pipeline_argv(
        tmp_path,
        output_root,
        inventory,
        "--stage",
        "verify-retarget",
    )

    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("internal verification launched a subprocess")
        ),
    )
    assert pipeline.main(argv) == 0
    receipt_path = output_root / "posttrain_pipeline/stages/verify-retarget.json"
    first_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert first_receipt["state"] == "succeeded"
    assert first_receipt["audio_policy"] == "disabled_not_loaded"
    assert len(first_receipt["attempts"]) == 1

    assert pipeline.main(argv) == 0
    reused_status = json.loads(
        (output_root / "posttrain_pipeline/pipeline_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert reused_status["stages"]["verify-retarget"]["state"] == "reused"
    assert len(json.loads(receipt_path.read_text())["attempts"]) == 1

    passed = output_root / "retarget/passed_manifest.jsonl"
    passed.write_text(passed.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert pipeline.main(argv) == 1
    failed_status = json.loads(
        (output_root / "posttrain_pipeline/pipeline_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert failed_status["state"] == "failed_resumable"
    assert "contract/input hash drift" in failed_status["failure"]["message"]


def test_verify_retarget_fails_closed_on_artifact_hash_change(tmp_path):
    output_root, inventory, safe_csv = make_complete_retarget(tmp_path)
    args = pipeline.resolve_args(
        pipeline.parse_args(
            pipeline_argv(
                tmp_path,
                output_root,
                inventory,
                "--stage",
                "verify-retarget",
                "--dry-run",
            )
        )
    )
    snapshot = pipeline.verify_retarget_snapshot(args)
    assert snapshot["passed_count"] == 1
    safe_csv.write_text("changed\n", encoding="utf-8")
    with pytest.raises(pipeline.PipelineError, match="safe CSV hash mismatch"):
        pipeline.verify_retarget_snapshot(args)


def test_finished_stage_output_tamper_is_not_reused(tmp_path):
    paths = {
        "stage_state": tmp_path / "state",
    }
    paths["stage_state"].mkdir()
    output = tmp_path / "output.json"
    output.write_text("first", encoding="utf-8")
    contract = {"stage": "label", "inputs": [{"sha256": "abc"}]}
    digest = pipeline.json_sha256(contract)
    write_json(
        pipeline.receipt_path("label", paths),
        {
            "stage": "label",
            "state": "succeeded",
            "stage_contract": contract,
            "stage_contract_sha256": digest,
            "outputs": pipeline.output_bindings([output]),
        },
    )
    action, _ = pipeline.decide_stage_action(
        "label", contract, digest, [output], paths
    )
    assert action == "reuse"
    output.write_text("tampered", encoding="utf-8")
    with pytest.raises(pipeline.PipelineError, match="output hash drift"):
        pipeline.decide_stage_action("label", contract, digest, [output], paths)


def test_complete_preexisting_label_outputs_are_adopted_without_execution(
    tmp_path, monkeypatch
):
    output_root, inventory, _ = make_complete_retarget(tmp_path)
    make_complete_labels(output_root)
    argv = pipeline_argv(
        tmp_path, output_root, inventory, "--stage", "label"
    )

    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("artifact adoption launched a subprocess")
        ),
    )
    assert pipeline.main(argv) == 0
    status = json.loads(
        (output_root / "posttrain_pipeline/pipeline_status.json").read_text()
    )
    assert status["stages"]["label"]["state"] == "adopted"
    receipt = json.loads(
        (output_root / "posttrain_pipeline/stages/label.json").read_text()
    )
    assert receipt["execution_provenance"] == (
        "validated_existing_artifacts_not_original_execution"
    )
    assert receipt["attempts"][-1]["command_executed"] is False


def test_partial_outputs_without_receipt_fail_closed(tmp_path):
    output_root, inventory, _ = make_complete_retarget(tmp_path)
    write_jsonl(output_root / "annotations/draft_prompts.jsonl", [])
    argv = pipeline_argv(
        tmp_path, output_root, inventory, "--stage", "label", "--dry-run"
    )
    assert pipeline.main(argv) == 1
    assert not (output_root / "posttrain_pipeline").exists()


def test_posttrain_config_uses_split80_and_resume_requires_last_checkpoint(tmp_path):
    output_root, inventory, _ = make_complete_retarget(tmp_path)
    args = pipeline.resolve_args(
        pipeline.parse_args(pipeline_argv(tmp_path, output_root, inventory))
    )
    paths = pipeline.pipeline_paths(args)
    config = pipeline.desired_posttrain_config(args, paths)

    assert config["training"]["split_fractions"] == {
        "train": 0.8,
        "validation": 0.1,
        "test": 0.1,
    }
    assert config["pipeline_metadata"]["audio_policy"] == "disabled_not_loaded"
    assert config["allow_unsafe_condition_cache"] is False
    assert "audio" not in pipeline.posttrain_command(args, paths)

    base_command = pipeline.posttrain_command(args, paths)
    with pytest.raises(pipeline.PipelineError, match="no last.pt"):
        pipeline.resumed_command("posttrain", base_command, "resume", args)
    args.posttrain_output_dir.mkdir(parents=True)
    last = args.posttrain_output_dir / "last.pt"
    import torch

    torch.save({"posttrain_step": 1}, last)
    resumed = pipeline.resumed_command("posttrain", base_command, "resume", args)
    assert resumed[-2:] == ["--resume-from", str(last)]
    torch.save({"posttrain_step": args.steps}, last)
    with pytest.raises(pipeline.PipelineError, match="zero-step resume"):
        pipeline.resumed_command("posttrain", base_command, "resume", args)


def test_posttrain_validation_binds_steps_split_cache_checkpoints_and_replay(tmp_path):
    import torch

    output_root, inventory, _ = make_complete_retarget(tmp_path)
    args = pipeline.resolve_args(
        pipeline.parse_args(pipeline_argv(tmp_path, output_root, inventory))
    )
    paths = pipeline.pipeline_paths(args)
    semantic_records = [
        {"clip_id": "clip_train"},
        {"clip_id": "clip_validation"},
        {"clip_id": "clip_test"},
    ]
    write_jsonl(paths["semantics"], semantic_records)
    paths["cache"].parent.mkdir(parents=True, exist_ok=True)
    paths["cache"].write_bytes(b"condition-cache")

    episodes = [
        {
            "clip_id": "clip_train",
            "speaker_key": "speaker_train",
            "source_group_key": "source_train",
            "split": "train",
        },
        {
            "clip_id": "clip_validation",
            "speaker_key": "speaker_validation",
            "source_group_key": "source_validation",
            "split": "validation",
        },
        {
            "clip_id": "clip_test",
            "speaker_key": "speaker_test",
            "source_group_key": "source_test",
            "split": "test",
        },
    ]
    split = {
        "contract_type": "speaker_source_group_strict_split",
        "contract_version": 1,
        "seed": 7,
        "fractions": dict(pipeline.DEFAULT_SPLIT_FRACTIONS),
        "counts": {"train": 1, "validation": 1, "test": 1},
        "speaker_to_split": {
            "speaker_train": "train",
            "speaker_validation": "validation",
            "speaker_test": "test",
        },
        "source_group_to_split": {
            "source_train": "train",
            "source_validation": "validation",
            "source_test": "test",
        },
        "episodes": episodes,
    }
    split["sha256"] = hashlib.sha256(
        json.dumps(
            split, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    data_contract = {
        "contract_type": "ula_v2_18d_interaction_posttrain_data",
        "contract_version": 1,
        "split_contract_sha256": split["sha256"],
        "episode_count": 3,
        "records": [],
    }
    data_contract["sha256"] = hashlib.sha256(
        json.dumps(
            data_contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    replay_guard = {
        "applicable": True,
        "passed": True,
        "baseline_total": 1.0,
        "current_total": 1.01,
        "delta": 0.01,
        "allowed_delta": 0.02,
    }
    release = {
        "formal_release_eligible": False,
        "artifact_status": "experimental_unreviewed_unsafe",
        "input_formal_release_eligible": False,
        "replay_regression_guard_required": True,
        "replay_regression_guard": replay_guard,
    }
    output = args.posttrain_output_dir
    output.mkdir(parents=True)
    best_path = output / "ula_fm_checkpoint.pt"
    last_path = output / "last.pt"
    common = {
        "action_dim": 18,
        "condition_dim": pipeline.CONDITION_DIM,
        "posttrain_artifact_kind": pipeline.POSTTRAIN_ARTIFACT_KIND,
        "artifact_status": "experimental_unreviewed_unsafe",
        "formal_release_eligible": False,
        "global_step": 2100,
        "posttrain_step": 2000,
        "posttrain_source": {
            "checkpoint_sha256": pipeline.sha256_file(args.base_checkpoint),
            "source_global_step": 100,
        },
        "posttrain_data_contract": data_contract,
        "posttrain_split_contract": split,
        "posttrain_config": {
            "split_fractions": dict(pipeline.DEFAULT_SPLIT_FRACTIONS)
        },
        "training_contract": {
            "replay_regression_guard": replay_guard,
            "formal_release_decision": release,
        },
    }
    torch.save(common, best_path)
    last = dict(common)
    last["training_state"] = {
        key: None for key in pipeline.POSTTRAIN_TRAINING_STATE_KEYS
    }
    torch.save(last, last_path)
    write_json(output / "split_manifest.json", split)
    write_json(
        output / "training_summary.json",
        {
            "target_steps": 2000,
            "completed_steps": 2000,
            "stopped_early": False,
            "best_step": 2000,
            "artifact_status": "experimental_unreviewed_unsafe",
            "formal_release_eligible": False,
            "split_contract_sha256": split["sha256"],
            "data_contract_sha256": data_contract["sha256"],
            "checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "final_replay_regression_guard": replay_guard,
            "formal_release_decision": release,
            "data_provenance": {
                "condition_cache": {
                    "cache_sha256": pipeline.sha256_file(paths["cache"])
                },
                "beat_counts": split["counts"],
            },
        },
    )

    result = pipeline.validate_posttrain_outputs(
        args, paths, {"passed_count": 3}
    )
    assert result["completed_steps"] == 2000
    last["posttrain_step"] = 1999
    torch.save(last, last_path)
    with pytest.raises(pipeline.PipelineError, match="checkpoint contract mismatch"):
        pipeline.validate_posttrain_outputs(args, paths, {"passed_count": 3})


def test_semantics_validation_rejects_any_audio_or_speech_metadata(tmp_path):
    root = tmp_path / "network_semantics"
    root.mkdir()
    paths = {
        "semantics_root": root,
        "semantics": root / "network_semantics.jsonl",
    }
    write_json(
        root / "summary.json",
        {
            "output_records": 1,
            "transcript_or_audio_metadata_used_for_labels": False,
            "behavior_supervised_records": 0,
            "emotion_supervised_records": 0,
        },
    )
    write_jsonl(
        paths["semantics"],
        [
            {
                "clip_id": "clip",
                "behavior_supervision_mask": False,
                "emotion_supervision_mask": False,
                "emotion_id": None,
                "audio_path": "/disabled.wav",
            }
        ],
    )
    with pytest.raises(pipeline.PipelineError, match="disabled metadata"):
        pipeline.validate_semantics_outputs(paths, {"passed_count": 1})


def test_cache_and_posttrain_require_explicit_unreviewed_authorization(tmp_path):
    output_root, inventory, _ = make_complete_retarget(tmp_path)
    argv = pipeline_argv(tmp_path, output_root, inventory)
    argv.remove("--allow-unreviewed")
    with pytest.raises(pipeline.PipelineError, match="--allow-unreviewed"):
        pipeline.Pipeline(pipeline.parse_args(argv))
