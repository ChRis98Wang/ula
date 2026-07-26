import argparse
import json
from pathlib import Path

from tools.human_motion_collection.run_beat2_annotation_pipeline import (
    build_label_command,
    build_render_command,
    build_retarget_command,
    build_validate_command,
    summarize_outputs,
)


def args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        python=tmp_path / "gmr-python",
        inventory=tmp_path / "inventory.jsonl",
        beat2_root=tmp_path / "beat2",
        renderer_python=tmp_path / "renderer-python",
        workers=3,
        render_workers=1,
        render_limit=24,
        retry_failed=True,
        limit=7,
        expected_eligible_count=280,
    )


def test_commands_join_the_versioned_batch_contract_and_auto_resume(tmp_path):
    options = args(tmp_path)
    output = tmp_path / "batch"
    retarget = output / "retarget"
    labels = output / "annotations"
    retarget.mkdir(parents=True)
    (retarget / "status.json").write_text("{}", encoding="utf-8")
    labels.mkdir()
    (labels / "draft_prompts.jsonl").write_text("", encoding="utf-8")

    retarget_command = build_retarget_command(options, retarget)
    label_command = build_label_command(options, retarget, labels)
    validate_command = build_validate_command(options, retarget, labels, output)
    render_command = build_render_command(options, output)

    assert "--resume" in retarget_command
    assert "--retry-failed" in retarget_command
    assert retarget_command[retarget_command.index("--workers") + 1] == "3"
    assert retarget_command[retarget_command.index("--limit") + 1] == "7"
    assert "--resume" in label_command
    assert str(retarget / "passed_manifest.jsonl") in label_command
    assert str(retarget / "pending_manifest.jsonl") in validate_command
    assert str(retarget / "excluded_manifest.jsonl") in validate_command
    assert str(labels / "draft_prompts.jsonl") in validate_command
    assert str(output / "review/review_queue.jsonl") in validate_command
    assert str(output / "review/review_queue.jsonl") in render_command
    assert str(output / "review/videos_v1") in render_command
    assert render_command[render_command.index("--limit") + 1] == "24"
    assert str(options.renderer_python) in render_command


def test_fresh_retarget_does_not_request_resume_or_retry(tmp_path):
    options = args(tmp_path)
    command = build_retarget_command(options, tmp_path / "fresh")

    assert "--resume" not in command
    assert "--retry-failed" not in command


def test_summary_separates_batch_completion_from_training_admission(tmp_path):
    output = tmp_path / "batch"
    (output / "retarget").mkdir(parents=True)
    (output / "annotations").mkdir()
    (output / "retarget/status.json").write_text(
        json.dumps(
            {
                "run_state": "finished",
                "coverage_complete": True,
                "eligible_task_count": 280,
                "excluded_task_count": 30,
                "pending_count": 0,
                "counts": {"passed": 268, "quality_failed": 12},
            }
        ),
        encoding="utf-8",
    )
    (output / "annotations/summary.json").write_text(
        json.dumps(
            {
                "draft_records": 268,
                "rejected_records": 0,
                "needs_human_review_records": 268,
                "accepted_for_training_records": 0,
            }
        ),
        encoding="utf-8",
    )
    (output / "validation_summary.json").write_text(
        json.dumps(
            {
                "valid": True,
                "error_count": 0,
                "counts": {"passed": 268, "annotations": 268, "train_ready": 0},
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_outputs(output)

    assert summary["batch_complete"] is True
    assert summary["ready_for_human_review"] is True
    assert summary["ready_for_training"] is False
    assert summary["annotations"]["drafts"] == 268
    assert summary["training_blocker"] == (
        "independent_motion_text_video_review_not_completed"
    )
