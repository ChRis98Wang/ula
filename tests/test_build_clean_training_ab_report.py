import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "build_clean_training_ab_report.py"
)
SPEC = importlib.util.spec_from_file_location("clean_ab_report", MODULE_PATH)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def test_parse_progress_ignores_partial_tail_and_finds_best(tmp_path):
    path = tmp_path / "progress.jsonl"
    path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "step": 10,
                        "steps": 100,
                        "train": {"total": 4.0},
                        "validation": {"total": 3.0},
                    }
                ),
                json.dumps(
                    {
                        "step": 20,
                        "steps": 100,
                        "train": {"total": 2.0},
                        "validation": {"total": 2.5},
                    }
                ),
                '{"step": 30',
            )
        ),
        encoding="utf-8",
    )

    result = REPORT.parse_progress(path)

    assert result["status"] == "available"
    assert result["current_step"] == 20
    assert result["target_steps"] == 100
    assert result["best_validation_step"] == 20
    assert result["best_validation"]["total"] == 2.5
    assert result["malformed_records"] == 1


def test_motion_only_condition_audit_detects_forbidden_slice_leak(tmp_path):
    path = tmp_path / "conditions.npz"
    values = np.zeros((3, 264), dtype=np.float32)
    values[:, 133:136] = np.asarray([0.0, 1.0, 0.0])
    np.savez_compressed(path, conditions=values, style_controls=values[:, 133:136])

    clean = REPORT.audit_condition_cache(
        path, formal_episode_contract="ula_v2_18d_motion_only_physical_qc_v1"
    )
    assert clean["passed"] is True

    values[0, 136] = 0.25
    np.savez_compressed(path, conditions=values, style_controls=values[:, 133:136])
    leaked = REPORT.audit_condition_cache(
        path, formal_episode_contract="ula_v2_18d_motion_only_physical_qc_v1"
    )
    assert leaked["passed"] is False
    assert leaked["zero_required_violations"] == ["qwen_motion_latent"]


def test_discovers_experimental_checkpoint_and_prepared_condition_cache(tmp_path):
    root = tmp_path / "paired"
    branch = root / "frozen_base"
    prepared = root / "prepared"
    branch.mkdir(parents=True)
    prepared.mkdir()
    checkpoint = branch / "generator_experimental.pt"
    checkpoint.write_bytes(b"checkpoint")
    cache = prepared / "conditions_264d_frozen_base.experimental.npz"
    np.savez_compressed(cache, conditions=np.zeros((1, 264), dtype=np.float32))

    assert REPORT._discover_generator_checkpoint(branch) == checkpoint
    assert REPORT._discover_condition_cache(branch) == cache


def test_experimental_cache_requires_identity_and_fixed_split_layout(tmp_path):
    path = tmp_path / "conditions_264d_frozen_base.experimental.npz"
    conditions = np.zeros((3, 264), dtype=np.float32)
    conditions[:, 133:136] = 0.1
    conditions[:, 136:] = 0.2
    np.savez_compressed(
        path,
        conditions=conditions,
        clip_ids=np.asarray(["a", "b", "c"]),
        fixed_split_assignments=np.asarray(["train", "validation", "test"]),
        style_controls=conditions[:, 133:136],
    )

    clean = REPORT.audit_condition_cache(
        path, experimental_variant="frozen_base"
    )
    assert clean["passed"] is True
    assert clean["experimental_layout_valid"] is True
    assert clean["slices"]["qwen_motion_latent"]["expected_zero"] is False

    conditions[0, 92] = 1.0
    np.savez_compressed(
        path,
        conditions=conditions,
        clip_ids=np.asarray(["a", "b", "c"]),
        fixed_split_assignments=np.asarray(["train", "validation", "test"]),
        style_controls=conditions[:, 133:136],
    )
    polluted = REPORT.audit_condition_cache(
        path, experimental_variant="frozen_base"
    )
    assert polluted["passed"] is False
    assert "behavior_one_hot" in polluted["zero_required_violations"]


def test_experimental_model_audit_never_falls_back_to_formal_loader(tmp_path):
    checkpoint = tmp_path / "generator_experimental.pt"
    torch.save({"artifact_kind": "anything"}, checkpoint)

    audit, metadata = REPORT._model_audit(
        checkpoint,
        device="cpu",
        frames=4,
        sampling_steps=1,
        seeds=(1,),
        style_delta=1.0,
        padding_tolerance=1e-6,
        min_generation_rms=0.0,
        min_style_response=0.0,
    )

    assert audit["status"] == "unavailable"
    assert "--abc-video-config" in audit["reason"]
    assert metadata == {}


def test_video_summary_requires_complete_hashed_side_by_side(tmp_path):
    mp4 = tmp_path / "abc.mp4"
    mp4.write_bytes(b"video")
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "artifact_kind": (
                    "beat2_clean_adaln_abc_experimental_video_v1"
                ),
                "status": "complete",
                "cases": [
                    {
                        "branches": {"A": {}, "B": {}, "C": {}},
                        "side_by_side": {
                            "output_mp4": str(mp4),
                            "sha256": REPORT._sha256_file(mp4),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = REPORT.summarize_video_artifact(summary)
    assert result["status"] == "complete"
    mp4.write_bytes(b"changed")
    assert REPORT.summarize_video_artifact(summary)["status"] == "unavailable"


def test_wait_for_artifacts_requires_explicit_abc_and_video(tmp_path):
    parser = REPORT._parser()
    args = parser.parse_args(
        [
            "--foundation-run",
            str(tmp_path),
            "--output-json",
            str(tmp_path / "report.json"),
            "--wait-for-artifacts",
        ]
    )
    with pytest.raises(ValueError, match="abc-video-config"):
        REPORT.wait_for_terminal_artifacts(
            args, poll_seconds=1.0, timeout_hours=1.0
        )


def test_qwen_comparison_separates_held_out_splits_and_signs_loss():
    frozen = {
        "metrics": {
            "validation": {
                "count": 20.0,
                "retrieval_loss": 4.0,
                "text_to_motion_recall_at_1": 0.1,
            },
            "test": None,
        },
        "sources": {"eval_sha256": "same"},
        "foundation_checkpoint_sha256": "foundation",
    }
    finetuned = {
        "metrics": {
            "validation": {
                "count": 20.0,
                "retrieval_loss": 3.0,
                "text_to_motion_recall_at_1": 0.3,
            },
            "test": None,
        },
        "sources": {"eval_sha256": "same"},
        "foundation_checkpoint_sha256": "foundation",
    }

    result = REPORT.compare_qwen_artifacts(frozen, finetuned)

    assert result["status"] == "available"
    assert result["comparability"]["passed"] is True
    deltas = result["splits"]["validation"]["deltas"]
    assert deltas["retrieval_loss"]["delta"] == -1.0
    assert deltas["retrieval_loss"]["improvement"] == 1.0
    assert abs(deltas["text_to_motion_recall_at_1"]["improvement"] - 0.2) < 1e-12
    assert result["splits"]["test"]["status"] == "unavailable"


def _native_qwen_metrics(split, *, recall_at_1, retrieval_loss, episode_count):
    return {
        "split": split,
        "episode_count": episode_count,
        "semantic_group_count": 54,
        "group_text_to_motion_recall_at_1": recall_at_1,
        "group_text_to_motion_recall_at_5": recall_at_1 + 0.1,
        "group_motion_to_text_recall_at_1": recall_at_1 + 0.2,
        "group_motion_to_text_recall_at_5": recall_at_1 + 0.3,
        "group_text_to_motion_median_rank": 8.0,
        "group_motion_to_text_median_rank": 9.0,
        "group_retrieval_loss": retrieval_loss,
        # Deliberately differ from group recall to ensure the report uses the
        # model-selection group-centroid metric for its compact table.
        "episode_text_to_motion_recall_at_1": 0.99,
        "episode_motion_to_text_recall_at_1": 0.98,
        "positive_episode_cosine": 0.4,
        "negative_episode_cosine": 0.1,
        "episode_cosine_gap": 0.3,
    }


def _qwen_summary(variant, *, validation_recall, test_recall):
    is_lora = variant == "lora_finetuned"
    return {
        "artifact_kind": (
            "beat2_qwen_lora_alignment_v1_summary"
            if is_lora
            else "beat2_qwen_frozen_base_alignment_v1_summary"
        ),
        "variant": variant,
        "best_step": 100,
        "best_validation": _native_qwen_metrics(
            "validation",
            recall_at_1=validation_recall,
            retrieval_loss=4.0 - validation_recall,
            episode_count=1629,
        ),
        "test": _native_qwen_metrics(
            "test",
            recall_at_1=test_recall,
            retrieval_loss=5.0 - test_recall,
            episode_count=2988,
        ),
        "qwen": {
            "model_name": "Qwen/Qwen3-Embedding-0.6B",
            **({"lora": {"rank": 8}} if is_lora else {}),
        },
    }


def test_qwen_pipeline_summary_layout_exposes_validation_and_test(tmp_path):
    frozen_path = tmp_path / "qwen_frozen_base_summary.json"
    lora_path = tmp_path / "qwen_lora_finetuned_summary.json"
    frozen_path.write_text(
        json.dumps(
            _qwen_summary(
                "frozen_base", validation_recall=0.1, test_recall=0.2
            )
        ),
        encoding="utf-8",
    )
    lora_path.write_text(
        json.dumps(
            _qwen_summary(
                "lora_finetuned", validation_recall=0.3, test_recall=0.4
            )
        ),
        encoding="utf-8",
    )

    frozen = REPORT.summarize_qwen_artifact(
        "official_frozen", frozen_path, ("kimodo",)
    )
    finetuned = REPORT.summarize_qwen_artifact(
        "beat2_only_lora", lora_path, ("kimodo",)
    )

    assert frozen["status"] == "available"
    assert frozen["metrics"]["validation"]["count"] == 54.0
    assert frozen["metrics"]["validation"]["episode_count"] == 1629.0
    assert frozen["metrics"]["validation"]["text_to_motion_recall_at_1"] == 0.1
    assert frozen["metrics"]["test"]["text_to_motion_recall_at_1"] == 0.2
    assert finetuned["metrics"]["validation"]["text_to_motion_recall_at_1"] == 0.3
    assert finetuned["metrics"]["test"]["text_to_motion_recall_at_1"] == 0.4
    assert finetuned["lora_state_present"] is True

    comparison = REPORT.compare_qwen_artifacts(frozen, finetuned)
    assert comparison["status"] == "available"
    assert (
        comparison["splits"]["validation"]["deltas"][
            "text_to_motion_recall_at_1"
        ]["improvement"]
        == 0.19999999999999998
    )
    assert (
        comparison["splits"]["test"]["deltas"]["text_to_motion_recall_at_1"][
            "improvement"
        ]
        == 0.2
    )

    foundation_run = tmp_path / "foundation"
    foundation_run.mkdir()
    output = tmp_path / "morning_report.json"
    assert (
        REPORT.main(
            [
                "--foundation-run",
                str(foundation_run),
                "--qwen-frozen",
                str(frozen_path),
                "--qwen-finetuned",
                str(lora_path),
                "--output-json",
                str(output),
                "--no-model-audits",
            ]
        )
        == 0
    )
    markdown = output.with_suffix(".md").read_text(encoding="utf-8")
    assert "| frozen | validation | 54 | 0.1000 |" in markdown
    assert "| frozen | test | 54 | 0.2000 |" in markdown
    assert "| finetuned | validation | 54 | 0.3000 |" in markdown
    assert "| finetuned | test | 54 | 0.4000 |" in markdown


def test_qwen_comparison_json_and_run_directory_select_the_requested_arm(tmp_path):
    baseline = _qwen_summary(
        "frozen_base", validation_recall=0.11, test_recall=0.12
    )
    lora = _qwen_summary(
        "lora_finetuned", validation_recall=0.21, test_recall=0.22
    )
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "artifact_kind": "beat2_qwen_motion_alignment_ab_v1_comparison",
                "baseline": baseline,
                "lora_finetuned": lora,
                "delta_lora_minus_frozen": {},
            }
        ),
        encoding="utf-8",
    )

    from_comparison_frozen = REPORT.summarize_qwen_artifact(
        "official_frozen", comparison_path, ()
    )
    from_comparison_lora = REPORT.summarize_qwen_artifact(
        "beat2_only_lora", comparison_path, ()
    )
    assert (
        from_comparison_frozen["metrics"]["validation"][
            "text_to_motion_recall_at_1"
        ]
        == 0.11
    )
    assert (
        from_comparison_lora["metrics"]["test"]["text_to_motion_recall_at_1"]
        == 0.22
    )
    assert (
        from_comparison_lora["container_artifact_kind"]
        == "beat2_qwen_motion_alignment_ab_v1_comparison"
    )

    (tmp_path / "qwen_frozen_base_summary.json").write_text(
        json.dumps(baseline), encoding="utf-8"
    )
    (tmp_path / "qwen_lora_finetuned_summary.json").write_text(
        json.dumps(lora), encoding="utf-8"
    )
    from_directory_frozen = REPORT.summarize_qwen_artifact(
        "official_frozen", tmp_path, ()
    )
    from_directory_lora = REPORT.summarize_qwen_artifact(
        "beat2_only_lora", tmp_path, ()
    )
    assert Path(from_directory_frozen["selected_artifact"]).name == (
        "qwen_frozen_base_summary.json"
    )
    assert Path(from_directory_lora["selected_artifact"]).name == (
        "qwen_lora_finetuned_summary.json"
    )


def test_cli_reports_missing_artifacts_as_unavailable(tmp_path):
    run = tmp_path / "clean_run"
    run.mkdir()
    output = tmp_path / "report.json"

    exit_code = REPORT.main(
        [
            "--foundation-run",
            str(run),
            "--qwen-frozen",
            str(tmp_path / "missing-frozen.json"),
            "--qwen-finetuned",
            str(tmp_path / "missing-lora.json"),
            "--output-json",
            str(output),
            "--no-model-audits",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["experiment_arms"]["A"]["generator"] == "A_motion_only_foundation"
    assert payload["experiment_arms"]["B"]["status"] == "unavailable"
    assert payload["experiment_arms"]["C"]["status"] == "unavailable"
    assert payload["report_contract"][
        "generator_quality_separate_from_text_motion_alignment"
    ]
    assert output.with_suffix(".md").is_file()
