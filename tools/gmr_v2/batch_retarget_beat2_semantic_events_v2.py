#!/usr/bin/env python3
"""Grouped, resumable BEAT2 semantic-event -> ULA V2 18D retargeting."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from . import batch_retarget_beat2_v2 as ordinary
    from .retarget_beat2_grouped_v2 import (
        GroupedBeat2RetargetRuntime,
        GroupedRuntimeConfig,
        build_retarget_segment_contract,
    )
except ImportError:  # pragma: no cover - direct invocation by path
    import batch_retarget_beat2_v2 as ordinary
    from retarget_beat2_grouped_v2 import (
        GroupedBeat2RetargetRuntime,
        GroupedRuntimeConfig,
        build_retarget_segment_contract,
    )


DEFAULT_INVENTORY = Path(
    "/home/gez/nas/cloud/gez/human_motion/catalog/"
    "beat2_semantic_event_inventory_v1/beat2_semantic_event_inventory_v1.jsonl"
)
DEFAULT_BEAT2_ROOT = Path(
    "/home/gez/nas/cloud/gez/human_motion/raw/BEAT2"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/gez/nas/cloud/gez/human_motion/processed/"
    "beat2_english_semantic_events_18d_v1/retarget"
)
SCHEMA_VERSION = "1.0.0"
RUN_CONTRACT_SCHEMA_VERSION = "1.0.0"
GROUPED_EXECUTION_POLICY = {
    "scheduling_unit": "source_clip",
    "source_npz_loads_per_group": 1,
    "smplx_model_instances_per_worker_process": 1,
    "gmr_runtime_instances_per_worker_process": 1,
    "ik_reset": "neutral_qpos_before_every_event",
    "warmup_scope": "repeated_for_every_event",
    "smoothing_scope": "event_local_no_cross_event_frames",
    "quality_scope": "event_local",
    "failure_scope": "event_local_continue_source_group",
    "fixed_duration_windows_allowed": False,
}

_WORKER_RUNTIME: GroupedBeat2RetargetRuntime | None = None
_WORKER_RUNTIME_KEY: tuple[str, ...] | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--beat2-root", type=Path, default=DEFAULT_BEAT2_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--smplx-model", type=Path, default=ordinary.DEFAULT_MODEL)
    parser.add_argument("--gmr-root", type=Path, default=ordinary.DEFAULT_GMR_ROOT)
    parser.add_argument("--urdf", type=Path, default=ordinary.DEFAULT_URDF)
    parser.add_argument("--config", type=Path, default=ordinary.DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-sources", type=int)
    parser.add_argument("--limit-events", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args(argv)


def runtime_config(args: argparse.Namespace) -> GroupedRuntimeConfig:
    return GroupedRuntimeConfig(
        smplx_model=args.smplx_model,
        gmr_root=args.gmr_root,
        urdf=args.urdf,
        config=args.config,
        warmup_frames=int(ordinary.RETARGET_PARAMETERS["warmup_frames"]),
        max_velocity_rad_s=float(
            ordinary.RETARGET_PARAMETERS["max_velocity_rad_s"]
        ),
        smoothing_window=int(ordinary.RETARGET_PARAMETERS["smoothing_window"]),
        posture_cost=float(ordinary.RETARGET_PARAMETERS["posture_cost"]),
        solver=str(ordinary.RETARGET_PARAMETERS["solver"]),
    )


def _runtime_key(config: GroupedRuntimeConfig) -> tuple[str, ...]:
    return (
        str(config.smplx_model),
        str(config.gmr_root),
        str(config.urdf),
        str(config.config),
        str(config.warmup_frames),
        str(config.max_velocity_rad_s),
        str(config.smoothing_window),
        str(config.posture_cost),
        config.solver,
    )


def worker_runtime(config: GroupedRuntimeConfig) -> GroupedBeat2RetargetRuntime:
    """Create expensive model state once in each persistent pool worker."""
    global _WORKER_RUNTIME, _WORKER_RUNTIME_KEY
    key = _runtime_key(config)
    if _WORKER_RUNTIME is None:
        _WORKER_RUNTIME = GroupedBeat2RetargetRuntime(config)
        _WORKER_RUNTIME_KEY = key
    elif _WORKER_RUNTIME_KEY != key:
        raise RuntimeError("Worker runtime configuration changed inside one process")
    return _WORKER_RUNTIME


def read_semantic_inventory(
    inventory: Path, beat2_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Admit only native variable-length semantic events."""
    eligible, excluded = ordinary.read_inventory(inventory, beat2_root)
    semantic: list[dict[str, Any]] = []
    for task in eligible:
        segment = task.get("training_segment")
        reasons = []
        if not isinstance(segment, dict):
            reasons.append("grouped_semantic_event:missing_training_segment")
        else:
            if segment.get("representation") != ordinary.VARIABLE_SEGMENT_REPRESENTATION:
                reasons.append("grouped_semantic_event:invalid_variable_length_representation")
            if segment.get("fixed_window_sec") is not None:
                reasons.append("grouped_semantic_event:fixed_window_forbidden")
            if not segment.get("boundary_source"):
                reasons.append("grouped_semantic_event:missing_boundary_source")
        if not isinstance(task.get("semantic_event"), dict) and not isinstance(
            task.get("official_semantic_event"), dict
        ):
            reasons.append("grouped_semantic_event:missing_official_semantic_event")
        if reasons:
            excluded.append(
                {
                    **task,
                    "status": "excluded",
                    "accepted_for_training": False,
                    "reasons": reasons,
                }
            )
        else:
            semantic.append(task)
    return semantic, excluded


def group_tasks_by_source(
    tasks: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    source_ids: dict[str, str] = {}
    for task in tasks:
        source = str(Path(task["source"]).resolve())
        source_clip_id = str(task["source_clip_id"])
        previous_id = source_ids.setdefault(source, source_clip_id)
        if previous_id != source_clip_id:
            raise ValueError(
                f"One source path has conflicting source_clip_id values: {source}"
            )
        groups.setdefault(source, []).append(task)
    return list(groups.values())


def limit_groups(
    groups: list[list[dict[str, Any]]],
    *,
    limit_sources: int | None,
    limit_events: int | None,
) -> list[list[dict[str, Any]]]:
    selected = groups[:limit_sources] if limit_sources is not None else groups
    if limit_events is None:
        return selected
    result: list[list[dict[str, Any]]] = []
    remaining = limit_events
    for group in selected:
        if remaining <= 0:
            break
        kept = group[:remaining]
        if kept:
            result.append(kept)
        remaining -= len(kept)
    return result


def build_run_contract(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    gmr_python_root = args.gmr_root / "general_motion_retargeting"
    if not gmr_python_root.is_dir():
        gmr_python_root = args.gmr_root
    gmr_hash, gmr_count = ordinary.python_tree_sha256(gmr_python_root)
    contract = {
        "schema_version": RUN_CONTRACT_SCHEMA_VERSION,
        "artifacts": {
            "grouped_batch_runner": ordinary.file_binding(Path(__file__)),
            "ordinary_batch_contract_helpers": ordinary.file_binding(
                Path(ordinary.__file__)
            ),
            "grouped_event_runtime": ordinary.file_binding(
                Path(__file__).with_name("retarget_beat2_grouped_v2.py")
            ),
            "beat2_decoder_and_provenance": ordinary.file_binding(
                Path(__file__).with_name("retarget_beat2_v2.py")
            ),
            "retarget_motionx_implementation": ordinary.file_binding(
                ordinary.RETARGET_IMPLEMENTATION
            ),
            "retarget_common_implementation": ordinary.file_binding(
                ordinary.RETARGET_COMMON_IMPLEMENTATION
            ),
            "ula_18d_contract_implementation": ordinary.file_binding(
                ordinary.ULA_18D_CONTRACT_IMPLEMENTATION
            ),
            "gmr_python_tree": {
                "path": str(gmr_python_root.resolve()),
                "sha256": gmr_hash,
                "python_file_count": gmr_count,
            },
            "robot_urdf": ordinary.file_binding(args.urdf),
            "smplx_model": ordinary.file_binding(args.smplx_model),
            "retarget_config": ordinary.file_binding(args.config),
            "python_interpreter": ordinary.file_binding(Path(sys.executable)),
        },
        "retarget_parameters": dict(ordinary.RETARGET_PARAMETERS),
        "output_contract": ordinary.ULA_V2_18D_CONTRACT,
        "axis_policy": ordinary.BEAT2_AXIS_POLICY,
        "joint_order": list(ordinary.JOINT_ORDER_18D),
        "joint_limits_rad": {
            name: [float(lower), float(upper)]
            for name, (lower, upper) in ordinary.JOINT_LIMITS_18D.items()
        },
        "quality_policy": dict(ordinary.QUALITY_POLICY),
        "grouped_execution_policy": dict(GROUPED_EXECUTION_POLICY),
    }
    return contract, ordinary.json_sha256(contract)


def _event_log_path(
    output_root: Path, task: dict[str, Any], run_id: str
) -> Path:
    return output_root / "logs" / task["task_id"] / f"{run_id}.log"


def _write_event_log(path: Path, payload: dict[str, Any]) -> None:
    ordinary.atomic_text(
        path, json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )


def _event_common(
    task: dict[str, Any],
    args: argparse.Namespace,
    inventory_hash: str,
    run_contract_hash: str,
    run_id: str,
    started_at: str,
    started: float,
    source_hash: str | None,
    log_path: Path,
) -> dict[str, Any]:
    return {
        **ordinary._base_result(
            task,
            args.inventory,
            inventory_hash,
            args.output_root,
            run_contract_hash,
        ),
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": ordinary.utc_now(),
        "elapsed_sec": round(time.perf_counter() - started, 6),
        "log_path": str(log_path.resolve()),
        "source_sha256": source_hash,
        "execution_mode": "source_grouped_in_process_event_isolated",
    }


def _publish_event_result(
    task: dict[str, Any],
    args: argparse.Namespace,
    inventory_hash: str,
    run_contract_hash: str,
    run_id: str,
    started_at: str,
    started: float,
    source_hash: str,
    stage_dir: Path,
    quality: dict[str, Any],
    log_path: Path,
) -> dict[str, Any]:
    passed = ordinary.quality_passes(quality, task, source_hash)
    category = "passed" if passed else "failed"
    destination = args.output_root / category / task["task_id"]
    ordinary.publish_directory(
        stage_dir, destination, args.output_root / "superseded" / category
    )
    quality_path = destination / "quality.json"
    safe_csv = ordinary.only_safe_csv(destination)
    quality["outputs"] = {
        "raw_csv": str(
            next(iter(sorted(destination.glob("*_gmr_raw_18d.csv"))), "")
        ),
        "safe_csv": str(safe_csv.resolve()),
    }
    ordinary.atomic_json(quality_path, quality)
    common = _event_common(
        task,
        args,
        inventory_hash,
        run_contract_hash,
        run_id,
        started_at,
        started,
        source_hash,
        log_path,
    )
    result = {
        **common,
        "returncode": 0,
        "status": "passed" if passed else "quality_failed",
        "output_dir": str(destination.resolve()),
        "quality_json": str(quality_path.resolve()),
        "quality_json_sha256": ordinary.sha256(quality_path),
        "safe_csv": str(safe_csv.resolve()),
        "safe_csv_sha256": ordinary.sha256(safe_csv),
        "quality_gate": quality.get("quality_gate", {}),
        "frames": quality.get("frames"),
        "duration_sec": quality.get("duration_sec"),
        "retarget_segment": quality.get("retarget_segment"),
    }
    ordinary.atomic_json(ordinary.result_path(args.output_root, task), result)
    return result


def _record_event_failure(
    task: dict[str, Any],
    args: argparse.Namespace,
    inventory_hash: str,
    run_contract_hash: str,
    run_id: str,
    started_at: str,
    started: float,
    source_hash: str | None,
    stage_dir: Path,
    log_path: Path,
    error: BaseException,
) -> dict[str, Any]:
    log_payload = {
        "status": "event_process_failed",
        "task_id": task["task_id"],
        "error": repr(error),
        "traceback": traceback.format_exc(),
    }
    _write_event_log(log_path, log_payload)
    destination = args.output_root / "failed" / task["task_id"]
    if not stage_dir.exists():
        stage_dir.mkdir(parents=True, exist_ok=False)
    ordinary.publish_directory(
        stage_dir, destination, args.output_root / "superseded/failed"
    )
    result = {
        **_event_common(
            task,
            args,
            inventory_hash,
            run_contract_hash,
            run_id,
            started_at,
            started,
            source_hash,
            log_path,
        ),
        "returncode": 1,
        "status": "event_process_failed",
        "error": repr(error),
        "output_dir": str(destination.resolve()),
    }
    ordinary.atomic_json(ordinary.result_path(args.output_root, task), result)
    return result


def run_source_group(
    tasks: list[dict[str, Any]],
    args: argparse.Namespace,
    inventory_hash: str,
    run_contract_hash: str,
    run_id: str,
    *,
    runtime_factory: Callable[
        [GroupedRuntimeConfig], GroupedBeat2RetargetRuntime
    ] = worker_runtime,
) -> list[dict[str, Any]]:
    """Process one source; an event failure never aborts later events."""
    if not tasks:
        return []
    source = Path(tasks[0]["source"]).resolve()
    if any(Path(task["source"]).resolve() != source for task in tasks):
        raise ValueError("run_source_group received tasks from multiple sources")

    runtime = runtime_factory(runtime_config(args))
    runtime.load_source(source)
    results: list[dict[str, Any]] = []
    for task in tasks:
        started_at = ordinary.utc_now()
        started = time.perf_counter()
        stage_dir = args.output_root / "staging" / run_id / task["task_id"]
        log_path = _event_log_path(args.output_root, task, run_id)
        try:
            event = runtime.reset_event(task)
            quality = runtime.retarget_event(task, event, stage_dir)
            ordinary.atomic_json(stage_dir / "quality.json", quality)
            _write_event_log(
                log_path,
                {
                    "status": "retarget_complete",
                    "task_id": task["task_id"],
                    "source_clip_id": task["source_clip_id"],
                    "event_reset_ordinal": event.get("event_reset_ordinal"),
                    "quality_passed": (quality.get("quality_gate") or {}).get(
                        "passed"
                    ),
                },
            )
            result = _publish_event_result(
                task,
                args,
                inventory_hash,
                run_contract_hash,
                run_id,
                started_at,
                started,
                runtime.source_hash,
                stage_dir,
                quality,
                log_path,
            )
        except Exception as error:  # one bad event must not poison its siblings
            result = _record_event_failure(
                task,
                args,
                inventory_hash,
                run_contract_hash,
                run_id,
                started_at,
                started,
                runtime.source_hash,
                stage_dir,
                log_path,
                error,
            )
        results.append(result)
    return results


def _worker_entry(payload: tuple[Any, ...]) -> list[dict[str, Any]]:
    return run_source_group(*payload)


def select_runnable_tasks(
    tasks: list[dict[str, Any]],
    args: argparse.Namespace,
    run_contract_hash: str,
) -> list[dict[str, Any]]:
    runnable = []
    for task in tasks:
        previous = ordinary.load_result(ordinary.result_path(args.output_root, task))
        if previous is None:
            runnable.append(task)
        elif not ordinary.result_lineage_matches(previous, task):
            runnable.append(task)
        elif ordinary.completed_pass_is_current(
            previous, task, run_contract_hash
        ):
            continue
        elif previous.get("status") == "passed" or args.retry_failed:
            runnable.append(task)
    return runnable


def status_payload(
    args: argparse.Namespace,
    inventory_hash: str,
    eligible: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    selected_groups: list[list[dict[str, Any]]],
    results: list[dict[str, Any]],
    pending: list[dict[str, Any]],
    run_id: str,
    run_state: str,
    started_at: str,
    run_contract: dict[str, Any],
    run_contract_hash: str,
) -> dict[str, Any]:
    selected = [task for group in selected_groups for task in group]
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "run_state": run_state,
        "started_at": started_at,
        "updated_at": ordinary.utc_now(),
        "inventory": str(args.inventory.resolve()),
        "inventory_sha256": inventory_hash,
        "run_contract": run_contract,
        "run_contract_sha256": run_contract_hash,
        "execution_policy": dict(GROUPED_EXECUTION_POLICY),
        "eligible_event_count": len(eligible),
        "eligible_source_count": len(group_tasks_by_source(eligible)),
        "excluded_event_count": len(excluded),
        "selected_event_count": len(selected),
        "selected_source_count": len(selected_groups),
        "terminal_event_count": len(results),
        "pending_event_count": len(pending),
        "counts": dict(Counter(row.get("status") for row in results)),
        "coverage_complete": len(results) + len(pending) == len(eligible),
    }


def _record_group_failure(
    group: list[dict[str, Any]],
    args: argparse.Namespace,
    inventory_hash: str,
    run_contract_hash: str,
    run_id: str,
    error: BaseException,
) -> None:
    for task in group:
        state_path = ordinary.result_path(args.output_root, task)
        if state_path.exists():
            continue
        now = ordinary.utc_now()
        log_path = _event_log_path(args.output_root, task, run_id)
        _write_event_log(
            log_path,
            {
                "status": "source_worker_failed",
                "task_id": task["task_id"],
                "error": repr(error),
                "traceback": traceback.format_exc(),
            },
        )
        result = {
            **ordinary._base_result(
                task,
                args.inventory,
                inventory_hash,
                args.output_root,
                run_contract_hash,
            ),
            "run_id": run_id,
            "started_at": now,
            "finished_at": now,
            "elapsed_sec": 0.0,
            "log_path": str(log_path.resolve()),
            "status": "source_worker_failed",
            "error": repr(error),
            "accepted_for_training": False,
        }
        ordinary.atomic_json(state_path, result)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    for name in ("limit_sources", "limit_events"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.retry_failed and not args.resume:
        raise ValueError("--retry-failed requires --resume")
    for name in (
        "inventory",
        "beat2_root",
        "smplx_model",
        "gmr_root",
        "urdf",
        "config",
    ):
        path = getattr(args, name).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        setattr(args, name, path)
    args.output_root = args.output_root.resolve()

    inventory_hash = ordinary.sha256(args.inventory)
    run_contract, run_contract_hash = build_run_contract(args)
    status_path = args.output_root / "status.json"
    if status_path.exists():
        if not args.resume:
            raise RuntimeError(f"Existing batch state requires --resume: {status_path}")
        ordinary.validate_resume_contract(
            status_path,
            args.output_root,
            inventory_hash,
            run_contract,
            run_contract_hash,
        )
    elif (args.output_root / ordinary.RUN_CONTRACT_FILENAME).exists() or any(
        (args.output_root / "state/results").glob("*.json")
    ):
        raise RuntimeError(
            "Retarget output contains state without status.json; refusing unsafe reuse"
        )

    eligible, excluded = read_semantic_inventory(args.inventory, args.beat2_root)
    if status_path.exists():
        ordinary.validate_saved_result_contracts(
            args.output_root, eligible, run_contract_hash
        )
    all_groups = group_tasks_by_source(eligible)
    selected_groups = limit_groups(
        all_groups,
        limit_sources=args.limit_sources,
        limit_events=args.limit_events,
    )
    selected_tasks = [task for group in selected_groups for task in group]
    runnable = select_runnable_tasks(selected_tasks, args, run_contract_hash)
    runnable_ids = {task["task_id"] for task in runnable}
    runnable_groups = [
        [task for task in group if task["task_id"] in runnable_ids]
        for group in selected_groups
    ]
    runnable_groups = [group for group in runnable_groups if group]

    args.output_root.mkdir(parents=True, exist_ok=True)
    contract_path = args.output_root / ordinary.RUN_CONTRACT_FILENAME
    if not contract_path.exists():
        ordinary.atomic_json(
            contract_path,
            {
                "run_contract_sha256": run_contract_hash,
                "run_contract": run_contract,
            },
        )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    started_at = ordinary.utc_now()

    results, pending = ordinary.write_manifests(
        args.output_root,
        eligible,
        excluded,
        args.inventory,
        inventory_hash,
        run_contract_hash,
    )
    ordinary.atomic_json(
        status_path,
        status_payload(
            args,
            inventory_hash,
            eligible,
            excluded,
            selected_groups,
            results,
            pending,
            run_id,
            "running",
            started_at,
            run_contract,
            run_contract_hash,
        ),
    )

    try:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _worker_entry,
                    (
                        group,
                        args,
                        inventory_hash,
                        run_contract_hash,
                        run_id,
                    ),
                ): group
                for group in runnable_groups
            }
            for index, future in enumerate(as_completed(futures), 1):
                group = futures[future]
                try:
                    group_results = future.result()
                except Exception as error:
                    _record_group_failure(
                        group,
                        args,
                        inventory_hash,
                        run_contract_hash,
                        run_id,
                        error,
                    )
                    group_results = []
                results, pending = ordinary.write_manifests(
                    args.output_root,
                    eligible,
                    excluded,
                    args.inventory,
                    inventory_hash,
                    run_contract_hash,
                )
                ordinary.atomic_json(
                    status_path,
                    status_payload(
                        args,
                        inventory_hash,
                        eligible,
                        excluded,
                        selected_groups,
                        results,
                        pending,
                        run_id,
                        "running",
                        started_at,
                        run_contract,
                        run_contract_hash,
                    ),
                )
                source_id = group[0]["source_clip_id"]
                counts = Counter(row["status"] for row in group_results)
                print(
                    f"[{index:04d}/{len(runnable_groups):04d}] "
                    f"{source_id}: {dict(counts)}",
                    flush=True,
                )
    except KeyboardInterrupt:
        results, pending = ordinary.write_manifests(
            args.output_root,
            eligible,
            excluded,
            args.inventory,
            inventory_hash,
            run_contract_hash,
        )
        ordinary.atomic_json(
            status_path,
            status_payload(
                args,
                inventory_hash,
                eligible,
                excluded,
                selected_groups,
                results,
                pending,
                run_id,
                "interrupted_resumable",
                started_at,
                run_contract,
                run_contract_hash,
            ),
        )
        return 130

    results, pending = ordinary.write_manifests(
        args.output_root,
        eligible,
        excluded,
        args.inventory,
        inventory_hash,
        run_contract_hash,
    )
    ordinary.atomic_json(
        status_path,
        status_payload(
            args,
            inventory_hash,
            eligible,
            excluded,
            selected_groups,
            results,
            pending,
            run_id,
            "finished",
            started_at,
            run_contract,
            run_contract_hash,
        ),
    )
    print(json.dumps(Counter(row["status"] for row in results), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
