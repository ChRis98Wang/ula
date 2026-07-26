#!/usr/bin/env python3
"""Run the resumable BEAT2 -> ULA V2 18D -> motion-text draft pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = (
    PROJECT_ROOT
    / "deliverables/interactive_human_motion_v1/catalog/beat2_interaction_full_inventory_v1.jsonl"
)
DEFAULT_BEAT2_ROOT = Path(
    "/home/gez/nas/cloud/gez/human_motion/raw/BEAT2/beat_chinese_v2.0.0"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "deliverables/interactive_human_motion_v1/batch_annotation_v1"
)
GMR_PYTHON = Path("/home/gez/shuaiwang/.venvs/gmr/bin/python")
DEFAULT_PYTHON = GMR_PYTHON if GMR_PYTHON.is_file() else Path(sys.executable)
RETARGET_SCRIPT = PROJECT_ROOT / "tools/gmr_v2/batch_retarget_beat2_v2.py"
LABEL_SCRIPT = PROJECT_ROOT / "tools/human_motion_collection/label_ula_v2_18d_motion.py"
VALIDATE_SCRIPT = (
    PROJECT_ROOT / "tools/human_motion_review/validate_beat2_annotation_batch.py"
)
RENDER_SCRIPT = (
    PROJECT_ROOT / "tools/human_motion_review/render_beat2_annotation_review.py"
)
ENV_ISAACLAB_PYTHON = Path("/home/gez/miniconda3/envs/env_isaaclab/bin/python")
DEFAULT_RENDERER_PYTHON = (
    ENV_ISAACLAB_PYTHON if ENV_ISAACLAB_PYTHON.is_file() else Path(sys.executable)
)
CORE_STAGES = ("retarget", "label", "validate")
STAGE_ORDER = (*CORE_STAGES, "render")
SCHEMA_VERSION = "1.0.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--beat2-root", type=Path, default=DEFAULT_BEAT2_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--renderer-python", type=Path, default=DEFAULT_RENDERER_PYTHON
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--stage",
        action="append",
        choices=STAGE_ORDER,
        help="Run only this stage; repeat for multiple stages. Default: all.",
    )
    parser.add_argument("--expected-eligible-count", type=int, default=280)
    parser.add_argument("--render-limit", type=int, default=24)
    parser.add_argument("--render-workers", type=int, default=1)
    return parser.parse_args(argv)


def resolved_executable(path: Path) -> Path:
    # Do not resolve a virtualenv symlink: that can bypass its site-packages.
    absolute = path.absolute()
    if not absolute.is_file():
        raise FileNotFoundError(absolute)
    if not os.access(absolute, os.X_OK):
        raise PermissionError(f"Python interpreter is not executable: {absolute}")
    return absolute


def build_retarget_command(args: argparse.Namespace, retarget_root: Path) -> list[str]:
    command = [
        str(args.python),
        str(RETARGET_SCRIPT),
        "--inventory",
        str(args.inventory),
        "--beat2-root",
        str(args.beat2_root),
        "--output-root",
        str(retarget_root),
        "--python",
        str(args.python),
        "--workers",
        str(args.workers),
    ]
    has_state = (retarget_root / "status.json").is_file()
    if has_state:
        command.append("--resume")
    if args.retry_failed and has_state:
        command.append("--retry-failed")
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    return command


def build_label_command(args: argparse.Namespace, retarget_root: Path, label_root: Path) -> list[str]:
    command = [
        str(args.python),
        str(LABEL_SCRIPT),
        "--input-manifest",
        str(retarget_root / "passed_manifest.jsonl"),
        "--output-dir",
        str(label_root),
    ]
    if (label_root / "draft_prompts.jsonl").is_file():
        command.append("--resume")
    return command


def build_validate_command(
    args: argparse.Namespace,
    retarget_root: Path,
    label_root: Path,
    output_root: Path,
) -> list[str]:
    return [
        str(args.python),
        str(VALIDATE_SCRIPT),
        "--inventory",
        str(args.inventory),
        "--passed",
        str(retarget_root / "passed_manifest.jsonl"),
        "--failed",
        str(retarget_root / "failed_manifest.jsonl"),
        "--pending",
        str(retarget_root / "pending_manifest.jsonl"),
        "--excluded",
        str(retarget_root / "excluded_manifest.jsonl"),
        "--annotations",
        str(label_root / "draft_prompts.jsonl"),
        "--expected-eligible-count",
        str(args.expected_eligible_count),
        "--review-queue-output",
        str(output_root / "review/review_queue.jsonl"),
        "--output",
        str(output_root / "validation_summary.json"),
    ]


def build_render_command(args: argparse.Namespace, output_root: Path) -> list[str]:
    render_root = output_root / "review/videos_v1"
    command = [
        str(args.python),
        str(RENDER_SCRIPT),
        "--review-queue",
        str(output_root / "review/review_queue.jsonl"),
        "--output-root",
        str(render_root),
        "--renderer-python",
        str(args.renderer_python),
        "--sampling",
        "stratified",
        "--workers",
        str(args.render_workers),
    ]
    if args.render_limit is not None:
        command.extend(("--limit", str(args.render_limit)))
    has_state = (render_root / "status.json").is_file()
    if has_state:
        command.append("--resume")
    if args.retry_failed and has_state:
        command.append("--retry-failed")
    return command


def run_stage(
    name: str,
    command: list[str],
    output_root: Path,
    run_id: str,
) -> dict[str, Any]:
    log_path = output_root / "logs" / f"{run_id}_{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command=" + json.dumps(command, ensure_ascii=False) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return {
        "name": name,
        "state": "finished" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "started_at": started_at,
        "finished_at": utc_now(),
        "command": command,
        "log_path": str(log_path.resolve()),
        "log_sha256": sha256(log_path),
    }


def summarize_outputs(output_root: Path) -> dict[str, Any]:
    retarget_status_path = output_root / "retarget/status.json"
    label_summary_path = output_root / "annotations/summary.json"
    validation_path = output_root / "validation_summary.json"
    render_summary_path = output_root / "review/videos_v1/summary.json"
    retarget = load_json(retarget_status_path) if retarget_status_path.is_file() else {}
    labels = load_json(label_summary_path) if label_summary_path.is_file() else {}
    validation = load_json(validation_path) if validation_path.is_file() else {}
    render = load_json(render_summary_path) if render_summary_path.is_file() else {}
    pending = int(retarget.get("pending_count", 0)) if retarget else None
    batch_complete = bool(
        retarget
        and retarget.get("run_state") == "finished"
        and pending == 0
        and retarget.get("coverage_complete") is True
    )
    return {
        "batch_complete": batch_complete,
        "retarget": {
            "eligible": retarget.get("eligible_task_count"),
            "excluded": retarget.get("excluded_task_count"),
            "pending": pending,
            "terminal_counts": retarget.get("counts", {}),
        },
        "annotations": {
            "drafts": labels.get("draft_records"),
            "technical_rejections": labels.get("rejected_records"),
            "needs_human_review": labels.get("needs_human_review_records"),
            "accepted_for_training": labels.get("accepted_for_training_records"),
        },
        "validation": {
            "valid": validation.get("valid"),
            "error_count": validation.get("error_count"),
            "counts": validation.get("counts", {}),
        },
        "review_videos": {
            "passed": (render.get("counts") or {}).get("passed"),
            "failed": (render.get("counts") or {}).get("failed"),
            "accepted_for_training": render.get("accepted_for_training"),
            "manual_motion_text_review_still_required": render.get(
                "manual_motion_text_review_still_required"
            ),
        },
        "ready_for_human_review": bool(validation.get("valid")),
        "ready_for_training": False,
        "training_blocker": "independent_motion_text_video_review_not_completed",
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    if args.expected_eligible_count < 0:
        raise ValueError("expected eligible count cannot be negative")
    if args.render_limit is not None and args.render_limit < 1:
        raise ValueError("render limit must be positive")
    if args.render_workers < 1:
        raise ValueError("render workers must be positive")
    args.python = resolved_executable(args.python)
    args.inventory = args.inventory.resolve()
    args.beat2_root = args.beat2_root.resolve()
    args.output_root = args.output_root.resolve()
    for path in (
        args.inventory,
        args.beat2_root,
        RETARGET_SCRIPT,
        LABEL_SCRIPT,
        VALIDATE_SCRIPT,
        RENDER_SCRIPT,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    requested = set(args.stage or CORE_STAGES)
    if "render" in requested:
        args.renderer_python = resolved_executable(args.renderer_python)
    stages = [stage for stage in STAGE_ORDER if stage in requested]
    retarget_root = args.output_root / "retarget"
    label_root = args.output_root / "annotations"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    status_path = args.output_root / "pipeline_status.json"
    status: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "state": "running",
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "inventory": str(args.inventory),
        "inventory_sha256": sha256(args.inventory),
        "beat2_root": str(args.beat2_root),
        "output_root": str(args.output_root),
        "python": str(args.python),
        "workers": args.workers,
        "limit": args.limit,
        "retry_failed": args.retry_failed,
        "requested_stages": stages,
        "stages": [],
        "training_admission_policy": "deny_until_independent_video_review",
    }
    atomic_json(status_path, status)

    commands = {
        "retarget": build_retarget_command(args, retarget_root),
        "label": build_label_command(args, retarget_root, label_root),
        "validate": build_validate_command(
            args, retarget_root, label_root, args.output_root
        ),
        "render": build_render_command(args, args.output_root),
    }
    for stage in stages:
        result = run_stage(stage, commands[stage], args.output_root, run_id)
        status["stages"].append(result)
        status["updated_at"] = utc_now()
        if result["returncode"] != 0:
            status["state"] = "failed_resumable"
            status["failed_stage"] = stage
            status["finished_at"] = utc_now()
            atomic_json(status_path, status)
            return int(result["returncode"] or 1)
        atomic_json(status_path, status)

    status["state"] = "finished"
    status["finished_at"] = utc_now()
    status["summary"] = summarize_outputs(args.output_root)
    atomic_json(status_path, status)
    atomic_json(args.output_root / "pipeline_summary.json", status["summary"])
    train_ready = args.output_root / "manifests/train_ready.jsonl"
    if not train_ready.exists():
        atomic_text(train_ready, "")
    print(json.dumps(status["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
