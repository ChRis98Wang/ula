#!/usr/bin/env python3
"""Prepare and batch-retarget official Hanyang emotional body motions.

This job is deliberately isolated from generator foundation training.  It
creates a participant-disjoint, hash-bound partial-18D research pool for the
emotion critic/calibration lane.  Source-faithful clips remain 150 frames at
30 Hz; time-dilated deployment previews are separate, ineligible artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import multiprocessing
import queue
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any, Iterable, Mapping
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.gmr_v2.retarget_hanyang_positions_v1 import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_GMR_ROOT,
    DEFAULT_MAX_VELOCITY_RAD_S,
    DEFAULT_POSTURE_COST,
    DEFAULT_SMOOTHING_WINDOW,
    DEFAULT_URDF,
    HanyangRetargetRuntime,
)
from upper_body_skeleton.data_source_registry import (  # noqa: E402
    EMOTION_CRITIC_ROLE,
    assert_no_forbidden_data_lineage,
    build_data_source_registry_contract,
    validate_data_source_registry_contract,
)
from upper_body_skeleton.hanyang_emotion_retarget import (  # noqa: E402
    DATASET,
    DATASET_ID,
    DATASET_LICENSE,
    DATASET_REVISION,
    DUPLICATE_EXCLUDED_STEMS,
    EVALUATION_PROTOCOL_ANOMALY_STEMS,
    HUMAN_EVALUATION_BYTES,
    HUMAN_EVALUATION_MD5,
    HUMAN_EVALUATION_NAME,
    SOURCE_ARCHIVE_BYTES,
    SOURCE_ARCHIVE_MD5,
    SOURCE_ARCHIVE_NAME,
    SOURCE_CLIP_COUNT,
    json_hash,
    load_human_evaluations,
    parse_clip_name,
    reject_forbidden_dataset_marker,
    sha256_file,
    stable_json,
    validate_official_file,
)


DEFAULT_RESEARCH_ROOT = Path(
    "/home/gez/nas/cloud/gez/human_motion/external_emotion_research/"
    "hanyang_emotional_body_motion_zenodo_10052504_v1"
)
ACQUISITION_PROVENANCE_NAME = "provenance_inventory.json"
ACQUISITION_PROVENANCE_SHA256 = (
    "ea8063aeee7a0f1c0c5a2fc553fc7217de3673737d7b4cb1f4ad408fb4911877"
)
OUTPUT_REVISION = "retarget_v1"
INVENTORY_KIND = "hanyang_partial_18d_source_inventory_v1"
RECEIPT_KIND = "hanyang_partial_18d_batch_retarget_receipt_v1"
STATUS_KIND = "hanyang_partial_18d_batch_status_v1"
DEFAULT_WORKERS = 4
WORKER_INITIALIZATION_TIMEOUT_SEC = 120.0

_WORKER_RUNTIME: HanyangRetargetRuntime | None = None
_WORKER_EVALUATIONS: dict[str, dict[str, Any]] = {}
_WORKER_OPTIONS: dict[str, Any] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(stable_json(dict(row)) + "\n")
    os.replace(temporary, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, default=DEFAULT_RESEARCH_ROOT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--gmr-root", type=Path, default=DEFAULT_GMR_ROOT)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--solver", default="daqp")
    parser.add_argument(
        "--max-velocity",
        type=float,
        default=DEFAULT_MAX_VELOCITY_RAD_S,
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=DEFAULT_SMOOTHING_WINDOW,
    )
    parser.add_argument(
        "--posture-cost",
        type=float,
        default=DEFAULT_POSTURE_COST,
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    reject_forbidden_dataset_marker(vars(args), context="batch_arguments")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be positive")
    if args.max_velocity <= 0:
        raise ValueError("--max-velocity must be positive")


def _official_paths(research_root: Path) -> dict[str, Path]:
    return {
        "archive": research_root / "raw" / SOURCE_ARCHIVE_NAME,
        "evaluation": research_root / "raw" / HUMAN_EVALUATION_NAME,
        "acquisition_provenance": research_root / ACQUISITION_PROVENANCE_NAME,
        "source_csv_root": research_root / "source" / "csv_v1",
        "output_root": research_root / OUTPUT_REVISION,
    }


def validate_acquisition(
    research_root: Path,
) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any]]:
    research_root = research_root.resolve()
    reject_forbidden_dataset_marker(str(research_root), context="research_root")
    paths = _official_paths(research_root)
    archive = validate_official_file(
        paths["archive"],
        expected_bytes=SOURCE_ARCHIVE_BYTES,
        expected_md5=SOURCE_ARCHIVE_MD5,
    )
    evaluation = validate_official_file(
        paths["evaluation"],
        expected_bytes=HUMAN_EVALUATION_BYTES,
        expected_md5=HUMAN_EVALUATION_MD5,
    )
    provenance_path = paths["acquisition_provenance"]
    if not provenance_path.is_file():
        raise FileNotFoundError(provenance_path)
    provenance_sha256 = sha256_file(provenance_path)
    if provenance_sha256 != ACQUISITION_PROVENANCE_SHA256:
        raise ValueError("Hanyang acquisition provenance changed after audit")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    # The receipt intentionally contains the negative policy field
    # ``kimodo_accessed_or_used=false``.  Scan only source-bearing lineage here
    # so that the denial proof is not confused with an actual source reference.
    assert_no_forbidden_data_lineage(
        provenance, context="acquisition_provenance"
    )
    if provenance["scope_and_isolation"]["kimodo_accessed_or_used"] is not False:
        raise ValueError("acquisition provenance does not guarantee Kimodo isolation")
    upper_zero = set(
        provenance["motion_quality_anomalies"][
            "upper_body_any_exact_zero_files"
        ]
    )
    if len(upper_zero) != 92:
        raise ValueError("unexpected Hanyang upper-body zero anomaly inventory")
    recorded_protocol_anomalies = set(
        provenance["human_evaluation_workbook"][
            "evaluation_protocol_anomaly_motions"
        ]
    )
    if recorded_protocol_anomalies != set(EVALUATION_PROTOCOL_ANOMALY_STEMS):
        raise ValueError("evaluation protocol anomaly inventory changed")
    return paths, {
        "archive": archive,
        "evaluation": evaluation,
        "acquisition_provenance": {
            "path": str(provenance_path),
            "sha256": provenance_sha256,
        },
    }, provenance


def _safe_archive_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = [member for member in archive.infolist() if not member.is_dir()]
    if len(members) != SOURCE_CLIP_COUNT:
        raise ValueError(
            f"expected {SOURCE_CLIP_COUNT} Hanyang CSV members, got {len(members)}"
        )
    names = [member.filename for member in members]
    if len(names) != len(set(names)):
        raise ValueError("Hanyang archive contains duplicate member names")
    for member in members:
        member_path = Path(member.filename)
        if (
            member_path.is_absolute()
            or len(member_path.parts) != 1
            or member_path.name != member.filename
            or member_path.suffix.casefold() != ".csv"
        ):
            raise ValueError(f"unsafe or unexpected archive member: {member.filename!r}")
        parse_clip_name(member.filename)
    return sorted(
        members,
        key=lambda member: tuple(
            parse_clip_name(member.filename)[key]
            for key in ("participant_id", "block_id", "trial_id", "emotion_index")
        ),
    )


def extract_and_inventory(
    paths: Mapping[str, Path],
    provenance: Mapping[str, Any],
    evaluations: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    csv_root = paths["source_csv_root"]
    output_root = paths["output_root"]
    inventory_path = output_root / "source_inventory.jsonl"
    if inventory_path.is_file():
        rows = [
            json.loads(line)
            for line in inventory_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if (
            len(rows) == SOURCE_CLIP_COUNT
            and all(row.get("artifact_kind") == INVENTORY_KIND for row in rows)
            and all(row.get("dataset_id") == DATASET_ID for row in rows)
            and all(Path(row["source_csv"]).is_file() for row in rows)
        ):
            return rows
        raise ValueError("existing Hanyang source inventory is incomplete or incompatible")

    csv_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    upper_zero = set(
        provenance["motion_quality_anomalies"][
            "upper_body_any_exact_zero_files"
        ]
    )
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(paths["archive"]) as archive:
        members = _safe_archive_members(archive)
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"Hanyang archive CRC failure: {bad_member}")
        for member in members:
            target = csv_root / member.filename
            if not target.is_file() or target.stat().st_size != member.file_size:
                temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
                with archive.open(member) as source, temporary.open("wb") as sink:
                    shutil.copyfileobj(source, sink, length=8 * 1024 * 1024)
                os.replace(temporary, target)
            if target.stat().st_size != member.file_size:
                raise ValueError(f"extracted size mismatch: {target}")
            clip = parse_clip_name(target.name)
            evaluation = evaluations.get(clip["clip_id"])
            if evaluation is None:
                raise ValueError(f"missing official human evaluation: {clip['clip_id']}")
            stem = clip["source_stem"]
            exclusion_reasons: list[str] = []
            if stem in upper_zero:
                exclusion_reasons.append("upper_body_any_exact_zero_triplet")
            if stem in EVALUATION_PROTOCOL_ANOMALY_STEMS:
                exclusion_reasons.append("official_evaluation_protocol_anomaly")
            if stem in DUPLICATE_EXCLUDED_STEMS:
                exclusion_reasons.append("exact_duplicate_content_noncanonical")
            row: dict[str, Any] = {
                "schema_version": "1.0.0",
                "artifact_kind": INVENTORY_KIND,
                "dataset": DATASET,
                "dataset_id": DATASET_ID,
                "dataset_revision": DATASET_REVISION,
                "dataset_license": DATASET_LICENSE,
                **clip,
                "source_csv": str(target.resolve()),
                "source_bytes": target.stat().st_size,
                "source_sha256": sha256_file(target),
                "human_evaluation_sha256": evaluation["sha256"],
                "rater_count": evaluation["rater_count"],
                "intended_share": evaluation["intended_share"],
                "intended_high_confidence": evaluation[
                    "intended_high_confidence"
                ],
                "pre_retarget_exclusion_reasons": exclusion_reasons,
                "retarget_candidate": not exclusion_reasons,
                "generator_foundation_eligible": False,
                "kimodo_accessed_or_used": False,
            }
            row["record_sha256"] = json_hash(row)
            rows.append(row)
    if len(rows) != SOURCE_CLIP_COUNT:
        raise ValueError("Hanyang inventory construction did not cover all clips")
    if len({row["source_stem"] for row in rows}) != SOURCE_CLIP_COUNT:
        raise ValueError("Hanyang inventory source stems are not unique")
    atomic_jsonl(inventory_path, rows)
    return rows


def _init_worker(
    options: Mapping[str, Any],
    initialization_queue: multiprocessing.Queue,
) -> None:
    global _WORKER_RUNTIME, _WORKER_EVALUATIONS, _WORKER_OPTIONS
    try:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        _WORKER_OPTIONS = dict(options)
        _WORKER_EVALUATIONS = load_human_evaluations(
            _WORKER_OPTIONS["human_evaluation_xlsx"]
        )
        _WORKER_RUNTIME = HanyangRetargetRuntime(
            gmr_root=_WORKER_OPTIONS["gmr_root"],
            urdf=_WORKER_OPTIONS["urdf"],
            config=_WORKER_OPTIONS["config"],
            solver=_WORKER_OPTIONS["solver"],
            posture_cost=_WORKER_OPTIONS["posture_cost"],
        )
    except BaseException as error:
        initialization_queue.put(
            {
                "ok": False,
                "pid": os.getpid(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(limit=12),
            }
        )
        raise
    initialization_queue.put({"ok": True, "pid": os.getpid()})


def _await_worker_initialization(
    pool: multiprocessing.pool.Pool,
    initialization_queue: multiprocessing.Queue,
    *,
    expected_workers: int,
    timeout_sec: float = WORKER_INITIALIZATION_TIMEOUT_SEC,
) -> list[int]:
    """Wait for every initial worker or terminate the pool on the first error.

    ``multiprocessing.Pool`` otherwise keeps replacing a process whose
    initializer raises, which can turn a missing dependency into an unbounded
    respawn loop.  The explicit handshake makes startup fail closed.
    """

    deadline = time.monotonic() + float(timeout_sec)
    initialized_pids: list[int] = []
    while len(initialized_pids) < expected_workers:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            pool.terminate()
            pool.join()
            raise TimeoutError(
                "timed out waiting for Hanyang retarget workers to initialize"
            )
        try:
            message = initialization_queue.get(timeout=remaining)
        except queue.Empty as error:
            pool.terminate()
            pool.join()
            raise TimeoutError(
                "timed out waiting for Hanyang retarget workers to initialize"
            ) from error
        if not message.get("ok"):
            pool.terminate()
            pool.join()
            raise RuntimeError(
                "Hanyang retarget worker initializer failed: "
                f"{message.get('error_type')}: {message.get('error')}\n"
                f"{message.get('traceback', '')}"
            )
        initialized_pids.append(int(message["pid"]))
    if len(set(initialized_pids)) != expected_workers:
        pool.terminate()
        pool.join()
        raise RuntimeError("Hanyang worker initialization reused a process")
    return initialized_pids


def _valid_existing_report(
    quality_path: Path,
    *,
    source_sha256: str,
) -> dict[str, Any]:
    report = json.loads(quality_path.read_text(encoding="utf-8"))
    if (
        report.get("dataset_id") != DATASET_ID
        or report.get("source_sha256") != source_sha256
        or report.get("admission", {}).get("foundation_ingest_allowed") is not False
    ):
        raise ValueError(f"incompatible existing quality report: {quality_path}")
    expected_record_sha256 = json_hash(
        {
            key: value
            for key, value in report.items()
            if key != "record_sha256"
        }
    )
    if report.get("record_sha256") != expected_record_sha256:
        raise ValueError(f"quality report hash mismatch: {quality_path}")
    return report


def _retarget_worker(row: Mapping[str, Any]) -> dict[str, Any]:
    if _WORKER_RUNTIME is None:
        raise RuntimeError("Hanyang batch worker was not initialized")
    started = time.perf_counter()
    stem = str(row["source_stem"])
    output_dir = Path(_WORKER_OPTIONS["clips_root"]) / stem
    quality_path = output_dir / "quality.json"
    try:
        if quality_path.is_file() and not _WORKER_OPTIONS["overwrite"]:
            report = _valid_existing_report(
                quality_path,
                source_sha256=str(row["source_sha256"]),
            )
            resumed = True
        else:
            report = _WORKER_RUNTIME.retarget(
                row["source_csv"],
                output_dir,
                human_evaluation=_WORKER_EVALUATIONS[row["clip_id"]],
                expected_source_sha256=row["source_sha256"],
                max_velocity=_WORKER_OPTIONS["max_velocity"],
                smoothing_window=_WORKER_OPTIONS["smoothing_window"],
                overwrite=_WORKER_OPTIONS["overwrite"],
            )
            resumed = False
        return {
            "source_stem": stem,
            "clip_id": row["clip_id"],
            "status": (
                "passed" if report["quality_gate"]["passed"] else "failed_quality"
            ),
            "resumed": resumed,
            "source_sha256": row["source_sha256"],
            "quality_json": str(quality_path.resolve()),
            "quality_json_sha256": sha256_file(quality_path),
            "quality_record_sha256": report["record_sha256"],
            "emotion_id": report["emotion_id"],
            "fixed_split_assignment": report["fixed_split_assignment"],
            "intended_high_confidence": bool(
                report.get("emotion_evaluation", {}).get(
                    "intended_high_confidence", False
                )
            ),
            "quality_gate": report["quality_gate"],
            "processing_sec": float(time.perf_counter() - started),
        }
    except Exception as error:  # keep the long batch alive and audit every failure
        return {
            "source_stem": stem,
            "clip_id": row["clip_id"],
            "status": "processing_error",
            "resumed": False,
            "source_sha256": row["source_sha256"],
            "quality_json": str(quality_path.resolve()),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(limit=8),
            "processing_sec": float(time.perf_counter() - started),
        }


def _manifest_row(
    inventory_by_stem: Mapping[str, Mapping[str, Any]],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    source = inventory_by_stem[result["source_stem"]]
    row = {
        "schema_version": "1.0.0",
        "dataset_id": DATASET_ID,
        "source_stem": source["source_stem"],
        "clip_id": source["clip_id"],
        "participant_id": source["participant_id"],
        "emotion_id": source["emotion_id"],
        "fixed_split_assignment": source["fixed_split_assignment"],
        "source_group_key": source["source_group_key"],
        "source_sha256": source["source_sha256"],
        "intended_share": source["intended_share"],
        "intended_high_confidence": source["intended_high_confidence"],
        "status": result["status"],
        "quality_json": result["quality_json"],
        "quality_json_sha256": result.get("quality_json_sha256"),
        "quality_record_sha256": result.get("quality_record_sha256"),
        "quality_gate": result.get("quality_gate"),
        "error_type": result.get("error_type"),
        "error": result.get("error"),
        "generator_foundation_eligible": False,
        "emotion_critic_candidate": result["status"] == "passed",
        "kimodo_accessed_or_used": False,
    }
    row["record_sha256"] = json_hash(row)
    return row


def _excluded_row(source: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        "schema_version": "1.0.0",
        "dataset_id": DATASET_ID,
        "source_stem": source["source_stem"],
        "clip_id": source["clip_id"],
        "participant_id": source["participant_id"],
        "emotion_id": source["emotion_id"],
        "fixed_split_assignment": source["fixed_split_assignment"],
        "source_sha256": source["source_sha256"],
        "status": "excluded_before_retarget",
        "exclusion_reasons": source["pre_retarget_exclusion_reasons"],
        "generator_foundation_eligible": False,
        "kimodo_accessed_or_used": False,
    }
    row["record_sha256"] = json_hash(row)
    return row


def _nested_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    by_status = Counter()
    by_emotion: dict[str, Counter[str]] = defaultdict(Counter)
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    high_confidence_passed = 0
    for row in rows:
        status = str(row["status"])
        by_status[status] += 1
        if row.get("emotion_id"):
            by_emotion[str(row["emotion_id"])][status] += 1
        if row.get("fixed_split_assignment"):
            by_split[str(row["fixed_split_assignment"])][status] += 1
        if status == "passed" and row.get("intended_high_confidence"):
            high_confidence_passed += 1
    return {
        "by_status": dict(sorted(by_status.items())),
        "by_emotion_and_status": {
            key: dict(sorted(value.items()))
            for key, value in sorted(by_emotion.items())
        },
        "by_split_and_status": {
            key: dict(sorted(value.items()))
            for key, value in sorted(by_split.items())
        },
        "passed_and_intended_high_confidence": high_confidence_passed,
    }


def materialize_progress(
    output_root: Path,
    *,
    inventory: list[dict[str, Any]],
    selected_candidate_count: int,
    results: Mapping[str, Mapping[str, Any]],
    started_utc: str,
    started_monotonic: float,
    complete: bool,
    registry_contract: Mapping[str, Any],
) -> dict[str, Any]:
    inventory_by_stem = {row["source_stem"]: row for row in inventory}
    excluded = [_excluded_row(row) for row in inventory if not row["retarget_candidate"]]
    processed = [
        _manifest_row(inventory_by_stem, result)
        for _, result in sorted(results.items())
    ]
    passed = [row for row in processed if row["status"] == "passed"]
    failed = [row for row in processed if row["status"] != "passed"]
    atomic_jsonl(output_root / "passed_manifest.jsonl", passed)
    atomic_jsonl(output_root / "failed_manifest.jsonl", failed)
    atomic_jsonl(output_root / "excluded_manifest.jsonl", excluded)
    elapsed = max(1e-9, time.monotonic() - started_monotonic)
    rate = len(processed) / elapsed
    remaining = max(0, selected_candidate_count - len(processed))
    status = {
        "schema_version": "1.0.0",
        "artifact_kind": STATUS_KIND,
        "updated_utc": utc_now(),
        "started_utc": started_utc,
        "dataset_id": DATASET_ID,
        "phase": "complete" if complete else "retargeting",
        "complete": complete,
        "source_inventory_count": len(inventory),
        "pre_retarget_excluded_count": len(excluded),
        "selected_retarget_candidate_count": selected_candidate_count,
        "processed_count": len(processed),
        "remaining_count": remaining,
        "clips_per_second": rate,
        "eta_seconds": (remaining / rate) if rate > 0 else None,
        "counts": _nested_counts([*processed, *excluded]),
        "data_source_registry_sha256": registry_contract["sha256"],
        "generator_foundation_ingest_allowed": False,
        "current_beat2_training_mutated": False,
        "kimodo_accessed_or_used": False,
    }
    status["record_sha256"] = json_hash(status)
    atomic_json(output_root / "status.json", status)
    return status


def write_receipt(
    output_root: Path,
    *,
    official_inventory: Mapping[str, Any],
    inventory: list[dict[str, Any]],
    status: Mapping[str, Any],
    registry_contract: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    artifact_files = {
        name: output_root / name
        for name in (
            "source_inventory.jsonl",
            "passed_manifest.jsonl",
            "failed_manifest.jsonl",
            "excluded_manifest.jsonl",
            "status.json",
            "data_source_registry.json",
        )
    }
    receipt: dict[str, Any] = {
        "schema_version": "1.0.0",
        "artifact_kind": RECEIPT_KIND,
        "created_utc": utc_now(),
        "dataset": DATASET,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_license": DATASET_LICENSE,
        "official_input_inventory": official_inventory,
        "source_inventory_count": len(inventory),
        "batch_status": dict(status),
        "batch_parameters": {
            "workers": args.workers,
            "limit": args.limit,
            "gmr_root": str(args.gmr_root.resolve()),
            "urdf": str(args.urdf.resolve()),
            "config": str(args.config.resolve()),
            "solver": args.solver,
            "max_velocity_rad_s": args.max_velocity,
            "smoothing_window": args.smoothing_window,
            "posture_cost": args.posture_cost,
        },
        "artifact_sha256": {
            name: sha256_file(path) for name, path in artifact_files.items()
        },
        "data_source_registry": dict(registry_contract),
        "admission": {
            "allowed_role": EMOTION_CRITIC_ROLE,
            "generator_foundation_ingest_allowed": False,
            "generator_training_started_from_this_pool": False,
            "robot_observable_blind_review_required_before_generator_ab": True,
            "current_beat2_training_mutated": False,
        },
        "kimodo_accessed_or_used": False,
    }
    receipt["record_sha256"] = json_hash(receipt)
    atomic_json(output_root / "expansion_receipt.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    started_utc = utc_now()
    started_monotonic = time.monotonic()
    paths, official_inventory, provenance = validate_acquisition(
        args.research_root
    )
    evaluations = load_human_evaluations(paths["evaluation"])
    inventory = extract_and_inventory(paths, provenance, evaluations)
    output_root = paths["output_root"]
    registry_contract = build_data_source_registry_contract(
        [DATASET_ID], role=EMOTION_CRITIC_ROLE
    )
    validate_data_source_registry_contract(
        registry_contract,
        expected_role=EMOTION_CRITIC_ROLE,
        expected_dataset_sources=[DATASET_ID],
    )
    atomic_json(output_root / "data_source_registry.json", registry_contract)

    candidates = [row for row in inventory if row["retarget_candidate"]]
    if args.limit is not None:
        candidates = candidates[: args.limit]
    pre_status = materialize_progress(
        output_root,
        inventory=inventory,
        selected_candidate_count=len(candidates),
        results={},
        started_utc=started_utc,
        started_monotonic=started_monotonic,
        complete=args.prepare_only,
        registry_contract=registry_contract,
    )
    if args.prepare_only:
        receipt = write_receipt(
            output_root,
            official_inventory=official_inventory,
            inventory=inventory,
            status=pre_status,
            registry_contract=registry_contract,
            args=args,
        )
        print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    worker_options = {
        "human_evaluation_xlsx": str(paths["evaluation"]),
        "clips_root": str(output_root / "clips"),
        "gmr_root": str(args.gmr_root.resolve()),
        "urdf": str(args.urdf.resolve()),
        "config": str(args.config.resolve()),
        "solver": args.solver,
        "max_velocity": args.max_velocity,
        "smoothing_window": args.smoothing_window,
        "posture_cost": args.posture_cost,
        "overwrite": args.overwrite,
    }
    results: dict[str, dict[str, Any]] = {}
    context = multiprocessing.get_context("spawn")
    worker_count = min(args.workers, len(candidates))
    initialization_queue = context.Queue()
    with context.Pool(
        processes=worker_count,
        initializer=_init_worker,
        initargs=(worker_options, initialization_queue),
    ) as pool:
        _await_worker_initialization(
            pool,
            initialization_queue,
            expected_workers=worker_count,
        )
        for result in pool.imap_unordered(_retarget_worker, candidates, chunksize=1):
            results[result["source_stem"]] = result
            if (
                len(results) % args.progress_every == 0
                or len(results) == len(candidates)
            ):
                status = materialize_progress(
                    output_root,
                    inventory=inventory,
                    selected_candidate_count=len(candidates),
                    results=results,
                    started_utc=started_utc,
                    started_monotonic=started_monotonic,
                    complete=len(results) == len(candidates),
                    registry_contract=registry_contract,
                )
                print(
                    stable_json(
                        {
                            "processed": status["processed_count"],
                            "remaining": status["remaining_count"],
                            "counts": status["counts"]["by_status"],
                            "eta_seconds": status["eta_seconds"],
                        }
                    ),
                    flush=True,
                )
    final_status = materialize_progress(
        output_root,
        inventory=inventory,
        selected_candidate_count=len(candidates),
        results=results,
        started_utc=started_utc,
        started_monotonic=started_monotonic,
        complete=True,
        registry_contract=registry_contract,
    )
    receipt = write_receipt(
        output_root,
        official_inventory=official_inventory,
        inventory=inventory,
        status=final_status,
        registry_contract=registry_contract,
        args=args,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not final_status["counts"]["by_status"].get("processing_error") else 2


if __name__ == "__main__":
    raise SystemExit(main())
