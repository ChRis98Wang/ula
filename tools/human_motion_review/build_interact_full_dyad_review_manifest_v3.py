#!/usr/bin/env python3
"""Build the fail-closed initial blind-review manifest for the full InterAct pool.

Every admitted dyad is a cataloged natural-boundary turn for which both actor
retargets passed physical QC.  Duration is never an admission criterion.  The
separate pilot selector uses frame-count quantiles only to exercise short,
middle, and long rendering paths; it does not change full-pool membership.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG_ROOT = Path(
    "/home/gez/nas/cloud/gez/human_motion/catalog/interact_dyadic_turn_v3_full"
)
DEFAULT_TASKS = DEFAULT_CATALOG_ROOT / "interact_actor_robot_episode_tasks.jsonl"
DEFAULT_CATALOG_SUMMARY = DEFAULT_CATALOG_ROOT / "summary.json"
DEFAULT_PHYSICAL_ROOT = Path(
    "/home/gez/nas/cloud/gez/human_motion/processed/"
    "interact_natural_turn_18d_v3_full_physical_preview"
)
DEFAULT_PHYSICAL_STATE = (
    DEFAULT_PHYSICAL_ROOT / "interact_native_bvh_full_retarget_v2.run_state.json"
)
DEFAULT_RAW_ROOT = Path("/home/gez/nas/cloud/gez/human_motion/raw/InterAct")
DEFAULT_OUTPUT_ROOT = Path(
    "/home/gez/shuaiwang/.private_human_motion/"
    "interact_full_dyad_review_v3/initial_manifest"
)

TASK_KIND = "interact_actor_specific_robot_episode_task"
CATALOG_KIND = "interact_dyadic_natural_rest_catalog_summary"
PHYSICAL_STATE_KIND = "interact_native_bvh_full_physical_retarget_v2_run_state"
QUALITY_KIND = "interact_actor_episode_ula_v2_18d_native_bvh_retarget_smoke"
DYAD_KIND = "interact_full_physical_passed_dyad_initial_review_task_v3"
NATURAL_DURATION_POLICY = (
    "semantic_affect_complete_at_predeclared_shared_rest_boundaries;"
    "no_fixed_target_minimum_or_maximum_duration"
)
PILOT_TARGET_QUANTILES = (0.08, 0.35, 0.65, 0.92)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-manifest", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--catalog-summary", type=Path, default=DEFAULT_CATALOG_SUMMARY)
    parser.add_argument("--physical-state", type=Path, default=DEFAULT_PHYSICAL_STATE)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--expected-dyads", type=int, default=1476)
    return parser.parse_args(argv)


def stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def value_sha256(value: object) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"JSONL is empty: {path}")
    return rows


def atomic_write(path: Path, payload: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.chmod(temporary, mode)
    os.replace(temporary, path)
    os.chmod(path, mode)


def atomic_json(path: Path, value: object) -> None:
    atomic_write(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(stable_json(row) + "\n" for row in rows).encode("utf-8")
    atomic_write(path, payload)


def _without_task_hash(row: dict[str, Any]) -> dict[str, Any]:
    value = dict(row)
    value.pop("episode_task_record_sha256", None)
    return value


def _all_false_masks(row: dict[str, Any]) -> bool:
    scalar_names = (
        "accepted_for_retarget_batch",
        "accepted_for_training",
        "admission_mask",
        "axis_qc_mask",
        "emotion_supervision_mask",
        "expression_completeness_mask",
        "physical_qc_mask",
        "relationship_conditioning_mask",
        "scenario_conditioning_mask",
        "source_emotion_conditioning_mask",
    )
    return all(row.get(name) is False for name in scalar_names) and not any(
        (row.get("semantic_supervision_masks") or {}).values()
    )


def _validate_context_plan(task: dict[str, Any]) -> list[dict[str, Any]]:
    context = task.get("context_plan") or {}
    if context.get("duration_policy") != NATURAL_DURATION_POLICY:
        raise ValueError("Task uses the wrong natural-duration policy")
    if context.get("duration_gate_used") is not False:
        raise ValueError("Task used duration as an admission gate")
    if context.get("selected_level") is not None or context.get(
        "selected_training_interval"
    ) is not None:
        raise ValueError("Task selected a training interval before blind review")
    completeness = context.get("completeness_review") or {}
    if completeness.get("elapsed_seconds_may_influence_decision") is not False:
        raise ValueError("Context expansion may not use elapsed seconds")
    if completeness.get("shrinking_below_core_or_cutting_inside_a_level_allowed") is not False:
        raise ValueError("Context plan permits an inside-level cut")
    levels = context.get("levels")
    if not isinstance(levels, list) or not levels:
        raise ValueError("Task has no predeclared natural context levels")
    interval = task.get("source_interval") or {}
    previous: tuple[int, int] | None = None
    for expected_level, level in enumerate(levels):
        if level.get("level") != expected_level:
            raise ValueError("Natural context levels are not consecutive")
        start = level.get("start_frame")
        end = level.get("end_frame_exclusive")
        if not isinstance(start, int) or not isinstance(end, int) or end <= start:
            raise ValueError("Natural context level has an invalid interval")
        current = (start, end)
        if expected_level == 0 and current != (
            interval.get("start_frame"),
            interval.get("end_frame_exclusive"),
        ):
            raise ValueError("Level zero differs from the physical core interval")
        if previous is not None:
            if current[0] > previous[0] or current[1] < previous[1] or current == previous:
                raise ValueError("Natural context levels are not strictly nested")
        previous = current
    recording = context.get("source_recording_interval")
    if recording != [levels[-1]["start_frame"], levels[-1]["end_frame_exclusive"]]:
        raise ValueError("Final natural context level does not cover the recording plan")
    return levels


def _resolve_source(task: dict[str, Any], raw_root: Path) -> Path:
    relative = Path(str((task.get("retarget_task") or {}).get("source_bvh") or ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Source BVH path is not safely relative")
    source = (raw_root / relative).resolve()
    try:
        source.relative_to(raw_root.resolve())
    except ValueError as error:
        raise ValueError("Source BVH escapes raw root") from error
    if not source.is_file():
        raise FileNotFoundError(source)
    return source


def _validate_task(
    task: dict[str, Any], raw_root: Path, source_hash_cache: dict[Path, str]
) -> dict[str, Any]:
    task_id = task.get("episode_task_id")
    if task.get("artifact_kind") != TASK_KIND or not task_id:
        raise ValueError("Invalid InterAct episode task")
    task_hash = task.get("episode_task_record_sha256")
    if task_hash != value_sha256(_without_task_hash(task)):
        raise ValueError(f"Task record SHA mismatch: {task_id}")
    if not _all_false_masks(task):
        raise ValueError(f"Catalog task unexpectedly opens a mask: {task_id}")
    levels = _validate_context_plan(task)
    source = _resolve_source(task, raw_root)
    source_hash = source_hash_cache.get(source)
    if source_hash is None:
        source_hash = sha256_file(source)
        source_hash_cache[source] = source_hash
    declared_hash = (task.get("retarget_task") or {}).get("source_bvh_sha256")
    if source_hash != declared_hash:
        raise ValueError(f"Source BVH SHA mismatch: {task_id}")
    interval = task["source_interval"]
    if interval.get("frame_count") != interval["end_frame_exclusive"] - interval["start_frame"]:
        raise ValueError(f"Task source frame count is invalid: {task_id}")
    return {
        "task": task,
        "task_id": task_id,
        "task_hash": task_hash,
        "source": source,
        "source_hash": source_hash,
        "levels": levels,
    }


def _validated_artifact(
    result: dict[str, Any], name: str, expected_root: Path
) -> tuple[Path, str]:
    artifact = (result.get("artifacts") or {}).get(name) or {}
    path = Path(str(artifact.get("path") or "")).resolve()
    try:
        path.relative_to(expected_root.resolve())
    except ValueError as error:
        raise ValueError(f"Physical artifact escapes output root: {name}") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    if digest != artifact.get("sha256"):
        raise ValueError(f"Physical artifact SHA mismatch: {path}")
    return path, digest


def _validate_physical_result(
    validated_task: dict[str, Any], result: dict[str, Any], physical_root: Path
) -> dict[str, Any]:
    task = validated_task["task"]
    task_id = validated_task["task_id"]
    if result.get("status") != "passed" or result.get("quality_gate_passed") is not True:
        raise ValueError(f"Non-passed physical result was admitted: {task_id}")
    if result.get("episode_task_record_sha256") != validated_task["task_hash"]:
        raise ValueError(f"Physical result/task SHA mismatch: {task_id}")
    if result.get("accepted_for_training") is not False:
        raise ValueError(f"Physical result unexpectedly admits training: {task_id}")
    if result.get("semantic_supervision_mask") is not False or result.get(
        "emotion_supervision_mask"
    ) is not False:
        raise ValueError(f"Physical result opens semantic/affect masks: {task_id}")
    paths = {
        name: _validated_artifact(result, name, physical_root)
        for name in ("quality_json", "raw_csv", "safe_csv")
    }
    quality = load_json(paths["quality_json"][0])
    if quality.get("artifact_kind") != QUALITY_KIND:
        raise ValueError(f"Wrong physical quality artifact kind: {task_id}")
    if quality.get("episode_task_id") != task_id or quality.get(
        "episode_task_record_sha256"
    ) != validated_task["task_hash"]:
        raise ValueError(f"Quality/task binding mismatch: {task_id}")
    if quality.get("source_sha256") != validated_task["source_hash"]:
        raise ValueError(f"Quality/source binding mismatch: {task_id}")
    if Path(str(quality.get("source_bvh") or "")).resolve() != validated_task["source"]:
        raise ValueError(f"Quality/source path mismatch: {task_id}")
    interval = task["source_interval"]
    if any(
        quality.get("source_interval", {}).get(key) != interval[key]
        for key in ("start_frame", "end_frame_exclusive", "frame_count")
    ):
        raise ValueError(f"Quality changed the natural source interval: {task_id}")
    if quality.get("frames") != result.get("output_frames") or result.get(
        "source_frames"
    ) != interval["frame_count"]:
        raise ValueError(f"Physical frame lineage mismatch: {task_id}")
    if quality.get("retimed") is not result.get("safety_retimed"):
        raise ValueError(f"Physical retime flag mismatch: {task_id}")
    if quality.get("quality_gate", {}).get("passed") is not True:
        raise ValueError(f"Quality gate is not passed: {task_id}")
    if quality.get("temporal_selection", {}).get("elapsed_time_cut_used") is not False:
        raise ValueError(f"Physical result used an elapsed-time cut: {task_id}")
    if quality.get("accepted_for_training") is not False or quality.get(
        "license_gate", {}
    ).get("training_authorized") is not False:
        raise ValueError(f"Physical artifact unexpectedly authorizes training: {task_id}")
    for name, (path, _) in paths.items():
        declared = (quality.get("outputs") or {}).get(name)
        if declared is not None and Path(declared).resolve() != path:
            raise ValueError(f"Quality output path mismatch for {name}: {task_id}")
    return {
        "episode_task_id": task_id,
        "episode_task_record_sha256": validated_task["task_hash"],
        "physical_result_record_sha256": value_sha256(result),
        "source_bvh": str(validated_task["source"]),
        "source_bvh_sha256": validated_task["source_hash"],
        "quality_json": str(paths["quality_json"][0]),
        "quality_json_sha256": paths["quality_json"][1],
        "raw_csv": str(paths["raw_csv"][0]),
        "raw_csv_sha256": paths["raw_csv"][1],
        "safe_csv": str(paths["safe_csv"][0]),
        "safe_csv_sha256": paths["safe_csv"][1],
        "source_frames": int(result["source_frames"]),
        "output_frames": int(result["output_frames"]),
        "safety_retimed": bool(result["safety_retimed"]),
        "physical_quality_passed": True,
        "accepted_for_training": False,
        "semantic_supervision_mask": False,
        "emotion_supervision_mask": False,
        "license_training_mask": False,
    }


def _dyad_record(
    turn_id: str,
    pair: list[dict[str, Any]],
    results: dict[str, Any],
    physical_root: Path,
    runtime_contract: dict[str, Any],
) -> dict[str, Any]:
    pair = sorted(pair, key=lambda item: item["task"]["target_actor_lineage"]["actor_id"])
    left, right = pair
    left_task, right_task = left["task"], right["task"]
    if left_task["performance_id"] != right_task["performance_id"]:
        raise ValueError(f"Dyad partners have different performances: {turn_id}")
    if left_task["source_interval"] != right_task["source_interval"]:
        raise ValueError(f"Dyad partners have different natural intervals: {turn_id}")
    if left["levels"] != right["levels"]:
        raise ValueError(f"Dyad partners have different context plans: {turn_id}")
    if left_task["target_actor_lineage"]["actor_id"] != right_task[
        "interaction_partner_lineage"
    ]["actor_id"] or right_task["target_actor_lineage"]["actor_id"] != left_task[
        "interaction_partner_lineage"
    ]["actor_id"]:
        raise ValueError(f"Dyad partner lineage is not reciprocal: {turn_id}")
    actors = []
    for role, item in zip(("A", "B"), pair, strict=True):
        result = results.get(item["task_id"])
        if result is None:
            raise ValueError(f"Missing physical state result: {item['task_id']}")
        actor = _validate_physical_result(item, result, physical_root)
        actor["anonymous_evidence_role"] = role
        actors.append(actor)
    interval = left_task["source_interval"]
    record = {
        "schema_version": "3.0.0",
        "artifact_kind": DYAD_KIND,
        "dyad_id": turn_id,
        "performance_id": left_task["performance_id"],
        "initial_review_level": 0,
        "source_interval": {
            "start_frame": interval["start_frame"],
            "end_frame_exclusive": interval["end_frame_exclusive"],
            "frame_count": interval["frame_count"],
        },
        "natural_context_levels": left["levels"],
        "natural_context_expansion_policy": left_task["context_plan"]["policy"],
        "duration_policy": NATURAL_DURATION_POLICY,
        "duration_gate_used_for_full_pool_admission": False,
        "fixed_duration_window_used": False,
        "inside_natural_level_cut_allowed": False,
        "actors": actors,
        "physical_runtime_contract": runtime_contract,
        "both_actor_physical_quality_passed": True,
        "any_safety_retimed": any(actor["safety_retimed"] for actor in actors),
        "evidence_contract": {
            "layout": "2x2_source_dyad_xz_yz_plus_mujoco_robot_a_b",
            "source_face_used": False,
            "source_fingers_used": False,
            "audio_used": False,
            "identity_scene_official_emotion_exposed": False,
            "synchronization": "explicit_monotonic_full_span_source_output_frame_lineage",
            "cropping_or_fixed_duration_resampling_allowed": False,
        },
        "semantic_supervision_masks": {
            "prompt_text": False,
            "scenario_text": False,
            "communicative_intent": False,
            "robot_observable_motion_form": False,
        },
        "emotion_supervision_mask": False,
        "license_training_mask": False,
        "accepted_for_training": False,
    }
    record["dyad_record_sha256"] = value_sha256(record)
    return record


def _select_pilot(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for retimed in (False, True):
        group = sorted(
            (row for row in rows if row["any_safety_retimed"] is retimed),
            key=lambda row: (row["source_interval"]["frame_count"], row["dyad_id"]),
        )
        if len(group) < len(PILOT_TARGET_QUANTILES):
            raise ValueError(f"Insufficient dyads for retime pilot stratum: {retimed}")
        used: set[str] = set()
        for target in PILOT_TARGET_QUANTILES:
            target_index = target * (len(group) - 1)
            candidates = sorted(
                enumerate(group),
                key=lambda pair: (
                    abs(pair[0] - target_index),
                    pair[1]["source_interval"]["frame_count"],
                    pair[1]["dyad_id"],
                ),
            )
            chosen_index, chosen = next(
                pair for pair in candidates if pair[1]["dyad_id"] not in used
            )
            used.add(chosen["dyad_id"])
            selected.append(
                {
                    "dyad_id": chosen["dyad_id"],
                    "dyad_record_sha256": chosen["dyad_record_sha256"],
                    "source_frames": chosen["source_interval"]["frame_count"],
                    "any_safety_retimed": retimed,
                    "coverage_quantile_target": target,
                    "coverage_rank_in_retime_stratum": chosen_index,
                    "coverage_stratum_size": len(group),
                    "pilot_coverage_only_not_admission": True,
                    "accepted_for_training": False,
                }
            )
    return sorted(selected, key=lambda row: row["dyad_id"])


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.expected_dyads < 1:
        raise ValueError("expected-dyads must be positive")
    task_manifest = args.task_manifest.resolve()
    catalog_summary_path = args.catalog_summary.resolve()
    physical_state_path = args.physical_state.resolve()
    raw_root = args.raw_root.resolve()
    physical_root = physical_state_path.parent.resolve()
    summary = load_json(catalog_summary_path)
    if summary.get("artifact_kind") != CATALOG_KIND:
        raise ValueError("Wrong InterAct catalog artifact kind")
    if summary.get("duration_policy") != NATURAL_DURATION_POLICY:
        raise ValueError("Catalog duration policy is invalid")
    if summary.get("duration_contract_audit", {}).get("forbidden_constraint_key_paths") != []:
        raise ValueError("Catalog contains a forbidden duration constraint")
    task_artifact = (summary.get("artifacts") or {}).get("actor_robot_episode_tasks") or {}
    if Path(str(task_artifact.get("path") or "")).resolve() != task_manifest:
        raise ValueError("Catalog does not bind the requested task manifest path")
    task_manifest_hash = sha256_file(task_manifest)
    if task_artifact.get("sha256") != task_manifest_hash:
        raise ValueError("Catalog task manifest SHA mismatch")
    state = load_json(physical_state_path)
    if state.get("artifact_kind") != PHYSICAL_STATE_KIND or state.get("status") != "complete":
        raise ValueError("Physical state is not a successful completed full run")
    if state.get("terminal_task_count") != state.get("execution_task_count"):
        raise ValueError("Physical state is not terminal for every catalog task")
    if state.get("status_counts", {}).get("processing_failed", 0) != 0:
        raise ValueError("Physical state contains processing failures")
    binding = state.get("catalog_binding") or {}
    if (
        binding.get("task_manifest_sha256") != task_manifest_hash
        or binding.get("catalog_summary_sha256") != sha256_file(catalog_summary_path)
        or Path(str(binding.get("task_manifest") or "")).resolve() != task_manifest
        or Path(str(binding.get("catalog_summary") or "")).resolve()
        != catalog_summary_path
    ):
        raise ValueError("Physical state catalog binding mismatch")
    implementation = PROJECT_ROOT / "tools/gmr_v2/batch_retarget_interact_native_v2.py"
    retarget_implementation = PROJECT_ROOT / "tools/gmr_v2/retarget_interact_bvh_v2.py"
    if state.get("implementation_sha256") != sha256_file(implementation):
        raise ValueError("Physical batch implementation changed since the completed run")
    if state.get("retarget_implementation_sha256") != sha256_file(retarget_implementation):
        raise ValueError("Physical retarget implementation changed since the completed run")

    tasks = load_jsonl(task_manifest)
    if len(tasks) != state.get("execution_task_count"):
        raise ValueError("Task count differs from completed physical state")
    source_hash_cache: dict[Path, str] = {}
    validated = [_validate_task(task, raw_root, source_hash_cache) for task in tasks]
    by_turn: dict[str, list[dict[str, Any]]] = {}
    for item in validated:
        by_turn.setdefault(item["task"]["turn_id"], []).append(item)
    if any(len(pair) != 2 for pair in by_turn.values()):
        raise ValueError("Catalog does not contain exactly two actor tasks per turn")
    results = state.get("results") or {}
    admitted_turns = [
        turn_id
        for turn_id, pair in sorted(by_turn.items())
        if all((results.get(item["task_id"]) or {}).get("status") == "passed" for item in pair)
    ]
    if len(admitted_turns) != args.expected_dyads:
        raise ValueError(
            f"Expected {args.expected_dyads} both-actor-passed dyads, found {len(admitted_turns)}"
        )
    dyads = [
        _dyad_record(
            turn_id,
            by_turn[turn_id],
            results,
            physical_root,
            state.get("runtime_contract") or {},
        )
        for turn_id in admitted_turns
    ]
    pilot = _select_pilot(dyads)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    os.chmod(output_root, 0o700)
    manifest_path = output_root / "interact_full_dyad_initial_review_tasks_v3.jsonl"
    pilot_path = output_root / "interact_full_dyad_pilot8_v3.jsonl"
    atomic_jsonl(manifest_path, dyads)
    atomic_jsonl(pilot_path, pilot)
    frame_counts = sorted(row["source_interval"]["frame_count"] for row in dyads)
    output = {
        "schema_version": "3.0.0",
        "artifact_kind": "interact_full_dyad_initial_review_manifest_summary_v3",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_manifest": str(task_manifest),
        "task_manifest_sha256": task_manifest_hash,
        "catalog_summary": str(catalog_summary_path),
        "catalog_summary_sha256": sha256_file(catalog_summary_path),
        "physical_state": str(physical_state_path),
        "physical_state_sha256": sha256_file(physical_state_path),
        "physical_implementation_sha256": state["implementation_sha256"],
        "retarget_implementation_sha256": state["retarget_implementation_sha256"],
        "physical_runtime_contract": state.get("runtime_contract") or {},
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "dyad_count": len(dyads),
        "actor_task_count": len(dyads) * 2,
        "source_bvh_count": len(source_hash_cache),
        "selection_contract": "same_turn_exactly_two_actors_and_both_physical_status_passed",
        "duration_policy": NATURAL_DURATION_POLICY,
        "duration_threshold_or_fixed_window_used_for_admission": False,
        "source_frame_count_diagnostics": {
            "minimum": frame_counts[0],
            "median": frame_counts[len(frame_counts) // 2],
            "maximum": frame_counts[-1],
        },
        "retime_distribution": dict(
            sorted(Counter(str(row["any_safety_retimed"]).lower() for row in dyads).items())
        ),
        "pilot_manifest": str(pilot_path),
        "pilot_manifest_sha256": sha256_file(pilot_path),
        "pilot_count": len(pilot),
        "pilot_sampling_contract": (
            "four_frame_count_quantile_coverage_points_per_any_retime_state;"
            "coverage_only_not_full_pool_admission"
        ),
        "semantic_supervision_mask": False,
        "emotion_supervision_mask": False,
        "license_training_mask": False,
        "accepted_for_training": False,
    }
    summary_path = output_root / "summary.json"
    atomic_json(summary_path, output)
    output["summary"] = str(summary_path)
    output["summary_sha256"] = sha256_file(summary_path)
    return output


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(build(parse_args(argv)), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
