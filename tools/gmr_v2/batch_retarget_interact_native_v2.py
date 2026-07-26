#!/usr/bin/env python3
"""Durably retarget the hash-bound InterAct natural-boundary catalog to 18D.

This stage performs physical conversion only.  It requires an independent
anonymous v2 axis review to pass, preserves each cataloged natural interval,
and leaves semantic, affect, license-training, and training-admission gates
closed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG_ROOT = Path(
    "/home/gez/nas/cloud/gez/human_motion/catalog/interact_dyadic_turn_v3_full"
)
DEFAULT_TASK_MANIFEST = DEFAULT_CATALOG_ROOT / "interact_actor_robot_episode_tasks.jsonl"
DEFAULT_CATALOG_SUMMARY = DEFAULT_CATALOG_ROOT / "summary.json"
DEFAULT_RAW_ROOT = Path("/home/gez/nas/cloud/gez/human_motion/raw/InterAct")
DEFAULT_PUBLIC_REVIEW_ROOT = Path(
    "/home/gez/nas/cloud/gez/human_motion/review/interact_blind_expression_v2/public"
)
DEFAULT_AXIS_QUEUE = DEFAULT_PUBLIC_REVIEW_ROOT / "axis_review_queue.jsonl"
DEFAULT_PUBLIC_SUMMARY = DEFAULT_PUBLIC_REVIEW_ROOT / "summary.json"
DEFAULT_AXIS_REVIEW = (
    PROJECT_ROOT
    / "deliverables/interactive_human_motion_v2/interact_blind_expression_v2/"
    "review_submissions/axis_reviewer_r2_v16.jsonl"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/gez/nas/cloud/gez/human_motion/processed/"
    "interact_natural_turn_18d_v3_full_physical_preview"
)

TASK_ARTIFACT_KIND = "interact_actor_specific_robot_episode_task"
CATALOG_ARTIFACT_KIND = "interact_dyadic_natural_rest_catalog_summary"
AXIS_PROTOCOL = "interact_robot_axis_blind_video_native_bvh_v2"
RETARGET_ARTIFACT_KIND = (
    "interact_actor_episode_ula_v2_18d_native_bvh_retarget_smoke"
)
NATURAL_DURATION_POLICY = (
    "semantic_affect_complete_at_predeclared_shared_rest_boundaries;"
    "no_fixed_target_minimum_or_maximum_duration"
)
TERMINAL_STATUSES = {"passed", "quality_failed", "processing_failed"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-manifest", type=Path, default=DEFAULT_TASK_MANIFEST)
    parser.add_argument("--catalog-summary", type=Path, default=DEFAULT_CATALOG_SUMMARY)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--public-summary", type=Path, default=DEFAULT_PUBLIC_SUMMARY)
    parser.add_argument("--axis-queue", type=Path, default=DEFAULT_AXIS_QUEUE)
    parser.add_argument("--axis-review", type=Path, default=DEFAULT_AXIS_REVIEW)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--warmup-source-frames", type=int, default=30)
    parser.add_argument("--max-velocity", type=float, default=3.0)
    parser.add_argument("--smoothing-window", type=int, default=7)
    parser.add_argument("--posture-cost", type=float, default=0.02)
    parser.add_argument("--solver", default="daqp")
    parser.add_argument(
        "--elbow-branch",
        choices=("unconstrained", "motionx_negative"),
        default="unconstrained",
    )
    return parser.parse_args(argv)


def stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def json_sha256(value: object) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"JSONL is empty: {path}")
    return rows


def atomic_json(path: Path, value: object) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    for attempt in range(120):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 119:
                raise
            time.sleep(0.25)


def _without_record_hash(row: dict[str, Any]) -> dict[str, Any]:
    value = dict(row)
    value.pop("episode_task_record_sha256", None)
    return value


def _validate_task(
    row: dict[str, Any],
    raw_root: Path,
    source_hash_cache: dict[Path, str] | None = None,
) -> dict[str, Any]:
    if row.get("artifact_kind") != TASK_ARTIFACT_KIND:
        raise ValueError("InterAct task artifact_kind is invalid")
    record_hash = row.get("episode_task_record_sha256")
    if record_hash != json_sha256(_without_record_hash(row)):
        raise ValueError(f"InterAct task record hash mismatch: {row.get('episode_task_id')}")
    if row.get("accepted_for_training") is not False:
        raise ValueError("Catalog task unexpectedly admits training")
    if row.get("admission_mask") is not False:
        raise ValueError("Catalog task admission mask is open")
    if row.get("emotion_supervision_mask") is not False:
        raise ValueError("Catalog task emotion mask is open")
    if any((row.get("semantic_supervision_masks") or {}).values()):
        raise ValueError("Catalog task semantic mask is open")
    context = row.get("context_plan") or {}
    if context.get("duration_policy") != NATURAL_DURATION_POLICY:
        raise ValueError("Catalog task does not use the native semantic-duration policy")
    if context.get("duration_gate_used") is not False:
        raise ValueError("Catalog task used elapsed duration as a gate")
    if context.get("selected_training_interval") is not None:
        raise ValueError("Catalog task selected a training interval before blind review")
    interval = row.get("source_interval") or {}
    start = interval.get("start_frame")
    end = interval.get("end_frame_exclusive")
    frames = interval.get("frame_count")
    if not all(isinstance(value, int) for value in (start, end, frames)):
        raise ValueError("Catalog source interval is not integer-valued")
    if start < 0 or end <= start or frames != end - start:
        raise ValueError("Catalog source interval is invalid")
    retarget = row.get("retarget_task") or {}
    if retarget.get("source_frame_interval") != [start, end]:
        raise ValueError("Retarget interval differs from the natural catalog interval")
    if retarget.get("partner_motion_mixed_into_target") is not False:
        raise ValueError("Partner motion may not be mixed into the target trajectory")
    relative_source = Path(str(retarget.get("source_bvh") or ""))
    if relative_source.is_absolute() or ".." in relative_source.parts:
        raise ValueError("InterAct source path must be a safe path below raw-root")
    source = (raw_root / relative_source).resolve()
    try:
        source.relative_to(raw_root.resolve())
    except ValueError as error:
        raise ValueError("InterAct source escapes raw-root") from error
    if not source.is_file():
        raise FileNotFoundError(source)
    source_hash_cache = source_hash_cache if source_hash_cache is not None else {}
    observed_source_hash = source_hash_cache.get(source)
    if observed_source_hash is None:
        observed_source_hash = sha256_file(source)
        source_hash_cache[source] = observed_source_hash
    if observed_source_hash != retarget.get("source_bvh_sha256"):
        raise ValueError(f"InterAct source SHA mismatch: {source}")
    return {
        "episode_task_id": row["episode_task_id"],
        "episode_task_record_sha256": record_hash,
        "source_bvh": str(source),
        "source_bvh_sha256": retarget["source_bvh_sha256"],
        "start_frame": start,
        "end_frame": end,
        "partner_actor_id": row["interaction_partner_lineage"]["actor_id"],
    }


def load_catalog(
    task_manifest: Path, catalog_summary: Path, raw_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task_manifest = task_manifest.resolve()
    catalog_summary = catalog_summary.resolve()
    summary = load_json(catalog_summary)
    if summary.get("artifact_kind") != CATALOG_ARTIFACT_KIND:
        raise ValueError("InterAct catalog summary artifact_kind is invalid")
    if summary.get("duration_policy") != NATURAL_DURATION_POLICY:
        raise ValueError("InterAct catalog summary duration policy is invalid")
    if summary.get("duration_contract_audit", {}).get(
        "forbidden_constraint_key_paths"
    ) != []:
        raise ValueError("InterAct catalog contains a forbidden duration constraint")
    selection = summary.get("selection") or {}
    if selection.get("scope") != (
        "full_pool_all_locally_complete_paired_performances"
    ):
        raise ValueError("InterAct catalog is not the complete local full pool")
    if selection.get("accepted_for_training_count") != 0:
        raise ValueError("InterAct catalog unexpectedly admits training")
    if summary.get("license_gate", {}).get("training_authorized_by_this_receipt") is not False:
        raise ValueError("Catalog license receipt unexpectedly authorizes training")
    artifact = (summary.get("artifacts") or {}).get("actor_robot_episode_tasks") or {}
    if Path(str(artifact.get("path") or "")).resolve() != task_manifest:
        raise ValueError("Task manifest path is not bound by the catalog summary")
    manifest_hash = sha256_file(task_manifest)
    if artifact.get("sha256") != manifest_hash:
        raise ValueError("Task manifest SHA is not bound by the catalog summary")
    rows = load_jsonl(task_manifest)
    expected_count = selection.get(
        "actor_specific_robot_episode_task_count"
    )
    if expected_count != len(rows):
        raise ValueError("Task manifest count differs from the catalog summary")
    seen = set()
    jobs = []
    source_hash_cache: dict[Path, str] = {}
    for row in rows:
        job = _validate_task(row, raw_root.resolve(), source_hash_cache)
        if job["episode_task_id"] in seen:
            raise ValueError(f"Duplicate task ID: {job['episode_task_id']}")
        seen.add(job["episode_task_id"])
        jobs.append(job)
    binding = {
        "task_manifest": str(task_manifest),
        "task_manifest_sha256": manifest_hash,
        "catalog_summary": str(catalog_summary),
        "catalog_summary_sha256": sha256_file(catalog_summary),
        "catalog_task_count": len(rows),
        "unique_source_bvh_count": len(source_hash_cache),
        "duration_policy": NATURAL_DURATION_POLICY,
        "training_authorized_by_catalog": False,
    }
    return jobs, binding


def validate_axis_review(
    public_summary_path: Path, queue_path: Path, review_path: Path
) -> dict[str, Any]:
    public_summary_path = public_summary_path.resolve()
    queue_path = queue_path.resolve()
    review_path = review_path.resolve()
    public = load_json(public_summary_path)
    if public.get("artifact_kind") != (
        "interact_native_bvh_separate_anonymous_blind_review_bundle_v2"
    ):
        raise ValueError("InterAct public review bundle is not native-BVH v2")
    if public.get("fixed_duration_window_used") is not False:
        raise ValueError("InterAct public review used a fixed duration window")
    if public.get("identity_scenario_official_text_or_emotion_exposed") is not False:
        raise ValueError("InterAct public review exposed hidden metadata")
    if Path(str(public.get("axis_queue") or "")).resolve() != queue_path:
        raise ValueError("Axis queue path differs from the public review summary")
    queue_hash = sha256_file(queue_path)
    public_summary_hash = sha256_file(public_summary_path)
    if public.get("axis_queue_sha256") != queue_hash:
        raise ValueError("Axis queue SHA differs from the public review summary")
    queue = load_jsonl(queue_path)
    if public.get("axis_records") != len(queue):
        raise ValueError("Axis queue count differs from the public review summary")
    queue_by_id = {row.get("sample_id"): row for row in queue}
    if None in queue_by_id or len(queue_by_id) != len(queue):
        raise ValueError("Axis queue sample IDs are missing or duplicated")
    reviews = load_jsonl(review_path)
    if len(reviews) != len(queue):
        raise ValueError("Axis blind review is incomplete")
    review_ids = set()
    reviewer_ids = set()
    for review in reviews:
        sample_id = review.get("sample_id")
        queued = queue_by_id.get(sample_id)
        if queued is None:
            raise ValueError(f"Unknown axis-review sample: {sample_id}")
        if review.get("protocol_version") != AXIS_PROTOCOL:
            raise ValueError("Axis blind review protocol is invalid")
        if review.get("axis_queue_sha256") != queue_hash:
            raise ValueError("Axis blind review is not bound to the current queue")
        if review.get("public_summary_sha256") != public_summary_hash:
            raise ValueError("Axis blind review is not bound to the current public summary")
        if review.get("video_sha256") != queued.get("video_sha256"):
            raise ValueError("Axis blind review video SHA mismatch")
        if (
            review.get("video_sha256_verified") is not True
            or review.get("decode_complete") is not True
            or review.get("declared_frame_count")
            != review.get("decoded_frame_count")
        ):
            raise ValueError("Axis blind review did not completely decode its video")
        if (
            review.get("native_duration_preserved") is not True
            or review.get("fixed_duration_window_used") is not False
        ):
            raise ValueError("Axis blind review violates the native-duration contract")
        if review.get("label_metadata_exposed") is not False:
            raise ValueError("Axis blind reviewer observed metadata leakage")
        if review.get("overall_result") != "pass":
            raise ValueError(f"Axis blind review did not pass: {sample_id}")
        if review.get("accepted_for_training", review.get("training_admission")) is not False:
            raise ValueError("Axis review unexpectedly admits training")
        review_id = review.get("review_id")
        reviewer_id = review.get("reviewer_id")
        if not review_id or review_id in review_ids or not reviewer_id:
            raise ValueError("Axis review IDs are missing or duplicated")
        review_ids.add(review_id)
        reviewer_ids.add(reviewer_id)
    return {
        "public_summary": str(public_summary_path),
        "public_summary_sha256": public_summary_hash,
        "axis_queue": str(queue_path),
        "axis_queue_sha256": queue_hash,
        "axis_review": str(review_path),
        "axis_review_sha256": sha256_file(review_path),
        "axis_review_count": len(reviews),
        "axis_reviewer_ids": sorted(reviewer_ids),
        "all_axis_reviews_passed": True,
        "training_admission_granted": False,
    }


def _artifact_paths(task_id: str, output_root: Path) -> dict[str, Path]:
    from tools.gmr_v2.retarget_interact_bvh_v2 import output_stem

    stem = output_stem(task_id)
    return {
        "raw_csv": output_root / f"{stem}_raw_18d.csv",
        "safe_csv": output_root / f"{stem}_safe_18d.csv",
        "quality_json": output_root / f"{stem}_quality.json",
    }


def validate_output(job: dict[str, Any], output_root: Path) -> dict[str, Any]:
    paths = _artifact_paths(job["episode_task_id"], output_root)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    quality = load_json(paths["quality_json"])
    if quality.get("artifact_kind") != RETARGET_ARTIFACT_KIND:
        raise ValueError("InterAct retarget artifact_kind is invalid")
    if quality.get("episode_task_id") != job["episode_task_id"]:
        raise ValueError("InterAct retarget task ID mismatch")
    if quality.get("episode_task_record_sha256") != job["episode_task_record_sha256"]:
        raise ValueError("InterAct retarget task record SHA mismatch")
    if quality.get("source_sha256") != job["source_bvh_sha256"]:
        raise ValueError("InterAct retarget source SHA mismatch")
    expected_interval = {
        "start_frame": job["start_frame"],
        "end_frame_exclusive": job["end_frame"],
    }
    interval = quality.get("source_interval") or {}
    if any(interval.get(key) != value for key, value in expected_interval.items()):
        raise ValueError("InterAct retarget changed the natural source interval")
    if quality.get("temporal_selection", {}).get("elapsed_time_cut_used") is not False:
        raise ValueError("InterAct retarget used an elapsed-time cut")
    if quality.get("accepted_for_training") is not False:
        raise ValueError("InterAct physical retarget unexpectedly admits training")
    if quality.get("license_gate", {}).get("training_authorized") is not False:
        raise ValueError("InterAct physical retarget unexpectedly authorizes training")
    gate_passed = quality.get("quality_gate", {}).get("passed") is True
    return {
        "episode_task_id": job["episode_task_id"],
        "episode_task_record_sha256": job["episode_task_record_sha256"],
        "status": "passed" if gate_passed else "quality_failed",
        "quality_gate_passed": gate_passed,
        "source_frames": int(job["end_frame"] - job["start_frame"]),
        "output_frames": int(quality["frames"]),
        "safety_retimed": quality.get("retimed") is True,
        "source_sample_span_sec": interval.get("sample_span_sec"),
        "output_sample_span_sec": quality.get("output_sample_span_sec"),
        "failed_quality_gates": sorted(
            key
            for key, value in (quality.get("quality_gate") or {}).items()
            if key != "passed" and value is not True
        ),
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "semantic_supervision_mask": False,
        "emotion_supervision_mask": False,
        "accepted_for_training": False,
    }


def _run_job(payload: dict[str, Any]) -> dict[str, Any]:
    from argparse import Namespace
    from tools.gmr_v2 import retarget_interact_bvh_v2 as retarget

    job = payload["job"]
    output_root = Path(payload["output_root"])
    args = Namespace(
        bvh=Path(job["source_bvh"]),
        start_frame=job["start_frame"],
        end_frame=job["end_frame"],
        output_dir=output_root,
        episode_task_id=job["episode_task_id"],
        episode_task_record_sha256=job["episode_task_record_sha256"],
        expected_source_sha256=job["source_bvh_sha256"],
        partner_actor_id=job["partner_actor_id"],
        gmr_root=Path(payload["gmr_root"]),
        urdf=Path(payload["urdf"]),
        config=Path(payload["config"]),
        warmup_source_frames=payload["warmup_source_frames"],
        max_velocity=payload["max_velocity"],
        smoothing_window=payload["smoothing_window"],
        posture_cost=payload["posture_cost"],
        solver=payload["solver"],
        elbow_branch=payload["elbow_branch"],
        processing_scope=(
            "full_pool_physical_preview_pending_blind_semantic_affect_review"
        ),
    )
    retarget.run(args)
    return validate_output(job, output_root)


def _failure_result(job: dict[str, Any], error: BaseException) -> dict[str, Any]:
    return {
        "episode_task_id": job["episode_task_id"],
        "episode_task_record_sha256": job["episode_task_record_sha256"],
        "status": "processing_failed",
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc()[-8000:],
        "semantic_supervision_mask": False,
        "emotion_supervision_mask": False,
        "accepted_for_training": False,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _summarize_state(state: dict[str, Any]) -> None:
    results = list((state.get("results") or {}).values())
    counts = Counter(row.get("status", "unknown") for row in results)
    state["status_counts"] = dict(sorted(counts.items()))
    state["terminal_task_count"] = sum(counts[key] for key in TERMINAL_STATUSES)
    state["physical_quality_passed_count"] = counts["passed"]
    state["training_ready_count"] = 0
    state["updated_at_utc"] = _utc_now()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    if args.warmup_source_frames < 0:
        raise ValueError("warmup-source-frames cannot be negative")

    jobs, catalog_binding = load_catalog(
        args.task_manifest, args.catalog_summary, args.raw_root
    )
    axis_binding = validate_axis_review(
        args.public_summary, args.axis_queue, args.axis_review
    )
    if args.limit is not None:
        jobs = jobs[: args.limit]

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "interact_native_bvh_full_retarget_v2.run_state.json"
    prior = load_json(state_path) if args.resume and state_path.is_file() else None
    if prior:
        if prior.get("catalog_binding") != catalog_binding:
            raise ValueError("Resume state catalog binding differs from current inputs")
        if prior.get("axis_review_binding") != axis_binding:
            raise ValueError("Resume state axis review binding differs from current inputs")
        state = prior
        state["status"] = "running"
    else:
        state = {
            "schema_version": "1.0.0",
            "artifact_kind": "interact_native_bvh_full_physical_retarget_v2_run_state",
            "status": "running",
            "created_at_utc": _utc_now(),
            "catalog_binding": catalog_binding,
            "axis_review_binding": axis_binding,
            "execution_task_count": len(jobs),
            "duration_policy": NATURAL_DURATION_POLICY,
            "fixed_minimum_maximum_or_target_duration_used": False,
            "safety_retime_may_only_lengthen_motion": True,
            "semantic_review_required_after_physical_retarget": True,
            "affect_review_required_after_physical_retarget": True,
            "license_training_confirmation_required": True,
            "accepted_for_training": False,
            "results": {},
        }

    from tools.gmr_v2 import retarget_interact_bvh_v2 as retarget

    runtime = {
        "output_root": str(output_root),
        "gmr_root": str(retarget.DEFAULT_GMR_ROOT),
        "urdf": str(retarget.DEFAULT_URDF),
        "config": str(retarget.DEFAULT_CONFIG),
        "warmup_source_frames": args.warmup_source_frames,
        "max_velocity": args.max_velocity,
        "smoothing_window": args.smoothing_window,
        "posture_cost": args.posture_cost,
        "solver": args.solver,
        "elbow_branch": args.elbow_branch,
    }
    state["runtime_contract"] = runtime
    state["implementation_sha256"] = sha256_file(Path(__file__).resolve())
    state["retarget_implementation_sha256"] = sha256_file(
        Path(retarget.__file__).resolve()
    )

    pending = []
    resume_recovered_existing_output_count = 0
    for job in jobs:
        previous = state["results"].get(job["episode_task_id"])
        if previous and previous.get("status") in {"passed", "quality_failed"}:
            try:
                state["results"][job["episode_task_id"]] = validate_output(
                    job, output_root
                )
                continue
            except (OSError, ValueError, KeyError, TypeError):
                if not args.retry_failed:
                    raise
        if previous and previous.get("status") == "processing_failed" and not args.retry_failed:
            continue
        if args.resume and previous is None:
            try:
                state["results"][job["episode_task_id"]] = validate_output(
                    job, output_root
                )
            except (OSError, ValueError, KeyError, TypeError):
                pass
            else:
                resume_recovered_existing_output_count += 1
                continue
        pending.append(job)

    state["resume_recovered_existing_output_count"] = (
        resume_recovered_existing_output_count
    )
    _summarize_state(state)
    atomic_json(state_path, state)
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_run_job, {"job": job, **runtime}): job
                for job in pending
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                job = futures[future]
                try:
                    result = future.result()
                except BaseException as error:  # preserve every failed task as evidence
                    result = _failure_result(job, error)
                state["results"][job["episode_task_id"]] = result
                _summarize_state(state)
                atomic_json(state_path, state)
                print(
                    f"[{completed:04d}/{len(pending):04d}] "
                    f"{job['episode_task_id']}: {result['status']}",
                    flush=True,
                )

    _summarize_state(state)
    state["status"] = (
        "complete"
        if state["terminal_task_count"] == len(jobs)
        and state["status_counts"].get("processing_failed", 0) == 0
        else "complete_with_failures"
    )
    state["completed_at_utc"] = _utc_now()
    atomic_json(state_path, state)
    return state


def main(argv: list[str] | None = None) -> None:
    state = run(parse_args(argv))
    print(
        json.dumps(
            {
                "status": state["status"],
                "execution_task_count": state["execution_task_count"],
                "status_counts": state["status_counts"],
                "training_ready_count": state["training_ready_count"],
                "accepted_for_training": state["accepted_for_training"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
