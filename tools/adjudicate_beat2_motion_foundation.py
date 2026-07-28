#!/usr/bin/env python3
"""Verify, adjudicate, and hash-lock an unlabeled BEAT2 motion foundation.

The stages are deliberately separate and fail closed:

1. ``verify-qc`` checks retarget accounting, the unchanged 18D quality
   contract, fixed speaker splits, lineage, and (optionally) every artifact
   hash.
2. ``adjudicate`` emits a motion-only physical-QC manifest.  It never marks
   records train-ready; loader and provenance admission remain a later review.
3. ``lock`` binds the inventory, retarget contract, QC receipt, adjudicated
   manifest, and release report into a non-authorizing provenance lock.

No stage reads or emits source text or audio content.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_beat2_motion_foundation_ingest_gap import (
    FOUNDATION_ANNOTATION_KIND,
    FOUNDATION_REPRESENTATION,
    SEMANTIC_MASKS,
)
from tools.gmr_v2 import batch_retarget_beat2_v2 as ordinary
from tools.gmr_v2.retarget_beat2_grouped_v2 import (
    MOTION_FOUNDATION_RETARGET_SEGMENT_REPRESENTATION,
)


DEFAULT_SELECTOR_SUMMARY = Path(
    "/home/gez/nas/cloud/gez/human_motion/catalog/"
    "beat2_semantic_event_pilot_v7_full/"
    "beat2_semantic_event_pilot_v7_full.summary.json"
)
REQUIRED_GATES = frozenset(
    {
        "joint_limits_pass",
        "velocity_pass",
        "target_fit_pass",
        "collision_pass",
        "axis_direction_pass",
        "head_joint_limits_pass",
        "head_velocity_pass",
        "head_direction_pass",
        "head_continuity_pass",
        "passed",
    }
)
FALSE_FLAGS = (
    "behavior_supervision_mask",
    "emotion_supervision_mask",
    "affect_observable_supervision_mask",
    "official_category_conditioning_enabled",
    "official_emotion_conditioning_enabled",
)
CONDITIONING_PAYLOAD_FIELDS = (
    "audio_relpath",
    "audio_source",
    "canonical_action",
    "canonical_prompt",
    "prompt",
    "prompt_schema",
    "prompt_source",
    "prompt_sha256",
    "prompt_contract",
    "source_text",
    "source_speech_context",
    "window_transcript_context",
    "semantic_event",
    "official_semantic_event",
    "official_gesture_semantic_spans",
    "behavior_id",
    "emotion_id",
    "source_emotion_id",
    "source_emotion_label",
)
STRIP_FROM_ADJUDICATED = frozenset(
    {
        *CONDITIONING_PAYLOAD_FIELDS,
        "source_speech_context_role",
        "window_transcript_role",
        "source_text_role",
        "conditioning_text_status",
        "interaction_label",
        "semantic_admission",
    }
)
FORBIDDEN_TOKEN = "kimodo"
MOTION_ONLY_EPISODE_CONTRACT = "ula_v2_18d_motion_only_physical_qc_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_and_lines(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    lines = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            lines += chunk.count(b"\n")
    return digest.hexdigest(), lines


def is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite staged artifact: {path}")
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite staged artifact: {path}")
    atomic_text(path, "".join(stable_json(record) + "\n" for record in records))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            records.append(value)
    return records


def clip_id(record: Mapping[str, Any]) -> str:
    value = str(record.get("clip_id") or record.get("task_id") or "").strip()
    if not value:
        raise ValueError("record is missing clip_id/task_id")
    return value


def index_records(
    records: Sequence[dict[str, Any]], *, label: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        key = clip_id(record)
        if key in result:
            raise ValueError(f"{label} has duplicate id: {key}")
        result[key] = record
    return result


def assert_no_forbidden_reference(value: Any, *, label: str) -> None:
    if isinstance(value, str):
        if FORBIDDEN_TOKEN in value.casefold():
            raise ValueError(f"{label} contains a forbidden cross-dataset reference")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            assert_no_forbidden_reference(str(key), label=label)
            assert_no_forbidden_reference(child, label=label)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for child in value:
            assert_no_forbidden_reference(child, label=label)


def _speaker_mapping(summary: Mapping[str, Any]) -> dict[str, str]:
    mapping = (
        (summary.get("speaker_assignment") or {}).get("speaker_to_split") or {}
    )
    result = {str(key): str(value) for key, value in mapping.items()}
    if set(result.values()) != {"train", "validation", "test"}:
        raise ValueError("selector summary has no complete fixed speaker split")
    return result


def _assert_within(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes {root}: {resolved}") from error
    return resolved


def validate_foundation_metadata(
    record: Mapping[str, Any],
    *,
    speaker_to_split: Mapping[str, str],
    label: str,
) -> None:
    errors: list[str] = []
    assert_no_forbidden_reference(record, label=label)
    if record.get("dataset") != "BEAT2":
        errors.append("dataset is not BEAT2")
    if record.get("dataset_subset") != "beat_english_v2.0.0":
        errors.append("dataset subset changed")
    if record.get("annotation_kind") != FOUNDATION_ANNOTATION_KIND:
        errors.append("annotation kind is not unlabeled motion foundation")
    if record.get("semantic_supervision_masks") != SEMANTIC_MASKS:
        errors.append("semantic supervision masks are not all false")
    if any(record.get(field) is not False for field in FALSE_FLAGS):
        errors.append("behavior/emotion/affect supervision is not all false")
    for field in CONDITIONING_PAYLOAD_FIELDS:
        if field in record and record.get(field) not in (None, "", [], {}):
            errors.append(f"{field} contains forbidden conditioning metadata")
    segment = record.get("training_segment")
    if not isinstance(segment, Mapping):
        errors.append("training segment is missing")
    else:
        if segment.get("representation") != FOUNDATION_REPRESENTATION:
            errors.append("training segment representation changed")
        if segment.get("fixed_window_sec") is not None:
            errors.append("fixed-duration training window is forbidden")
        if segment.get("overlap_frames") != 0:
            errors.append("source chunks overlap")
        if segment.get("boundary_source") != "source_container_frame_bounds":
            errors.append("source boundary contract changed")
    speaker = str(record.get("speaker_key") or "")
    expected_split = speaker_to_split.get(speaker)
    if expected_split is None:
        errors.append("speaker is missing from fixed split")
    elif record.get("fixed_split_assignment") != expected_split:
        errors.append("fixed split assignment changed")
    source_group = str(record.get("source_group_key") or "")
    if not source_group.startswith("BEAT2/beat_english_v2.0.0/"):
        errors.append("source group is outside BEAT2 English")
    if errors:
        raise ValueError(f"{label}: " + "; ".join(errors))


def _validate_run_contract(
    *,
    status: Mapping[str, Any],
    saved: Mapping[str, Any],
    inventory_sha256: str,
) -> tuple[dict[str, Any], str]:
    contract = saved.get("run_contract")
    digest = saved.get("run_contract_sha256")
    if not isinstance(contract, dict) or not is_sha256(digest):
        raise ValueError("retarget run contract is incomplete")
    if ordinary.json_sha256(contract) != digest:
        raise ValueError("retarget run contract canonical hash mismatch")
    if (
        status.get("run_contract") != contract
        or status.get("run_contract_sha256") != digest
    ):
        raise ValueError("status and saved run contract disagree")
    if status.get("inventory_sha256") != inventory_sha256:
        raise ValueError("retarget status is bound to a different inventory")
    if contract.get("quality_policy") != ordinary.QUALITY_POLICY:
        raise ValueError("original 18D physical-QC thresholds changed")
    if contract.get("output_contract") != ordinary.ULA_V2_18D_CONTRACT:
        raise ValueError("18D output contract changed")
    if contract.get("axis_policy") != ordinary.BEAT2_AXIS_POLICY:
        raise ValueError("BEAT2 axis policy changed")
    parameters = contract.get("retarget_parameters") or {}
    required_parameters = {
        "max_velocity_rad_s": ordinary.RETARGET_PARAMETERS["max_velocity_rad_s"],
        "smoothing_window": ordinary.RETARGET_PARAMETERS["smoothing_window"],
        "posture_cost": ordinary.RETARGET_PARAMETERS["posture_cost"],
        "solver": ordinary.RETARGET_PARAMETERS["solver"],
        "warmup_frames": ordinary.RETARGET_PARAMETERS["warmup_frames"],
        "neutral_limit_margin_rad": 1e-6,
    }
    if any(parameters.get(key) != value for key, value in required_parameters.items()):
        raise ValueError("retarget parameters differ from the reviewed foundation plan")
    return contract, str(digest)


def _verify_current_file_bindings(contract: Mapping[str, Any]) -> int:
    verified = 0
    for name, binding in (contract.get("artifacts") or {}).items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"run contract artifact binding is invalid: {name}")
        path_value = str(binding.get("resolved_path") or binding.get("path") or "")
        path = Path(path_value)
        expected = binding.get("sha256")
        if path.is_file():
            if not is_sha256(expected) or sha256_file(path) != expected:
                raise ValueError(f"run contract file binding changed: {name}")
            verified += 1
    return verified


def validate_passed_record(
    record: Mapping[str, Any],
    *,
    inventory_record: Mapping[str, Any],
    inventory_sha256: str,
    run_contract_sha256: str,
    retarget_root: Path,
    speaker_to_split: Mapping[str, str],
    verify_artifacts: bool,
    source_hash_cache: dict[Path, str],
) -> None:
    key = clip_id(record)
    validate_foundation_metadata(
        record, speaker_to_split=speaker_to_split, label=f"passed:{key}"
    )
    errors: list[str] = []
    if record.get("status") != "passed":
        errors.append("status is not passed")
    if record.get("accepted_for_training") is not False:
        errors.append("retarget result prematurely claims training admission")
    if record.get("selected_record_sha256") != canonical_sha256(inventory_record):
        errors.append("selected inventory row hash mismatch")
    if record.get("retarget_input_manifest_sha256") != inventory_sha256:
        errors.append("retarget input manifest hash mismatch")
    if record.get("run_contract_sha256") != run_contract_sha256:
        errors.append("run contract hash mismatch")
    frames = record.get("frames")
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 30:
        errors.append("output has fewer than 30 frames")
    if not math.isclose(float(record.get("fps") or 0.0), 30.0, abs_tol=1e-9):
        errors.append("output is not exactly 30 Hz")
    gates = record.get("quality_gate")
    if not isinstance(gates, Mapping) or not REQUIRED_GATES.issubset(gates):
        errors.append("physical-QC gate set is incomplete")
    elif any(value is not True for value in gates.values()):
        errors.append("not every declared physical-QC gate passed")
    retarget = record.get("retarget_segment")
    if not isinstance(retarget, Mapping):
        errors.append("retarget segment is missing")
    else:
        if (
            retarget.get("representation")
            != MOTION_FOUNDATION_RETARGET_SEGMENT_REPRESENTATION
        ):
            errors.append("retarget segment representation changed")
        if retarget.get("cropped") is not False:
            errors.append("retarget output was cropped")
        if retarget.get("output_frame_count") != frames:
            errors.append("retarget output frame count mismatch")
    quality_value = str(record.get("quality_json") or "")
    safe_value = str(record.get("safe_csv") or "")
    quality_path = (
        _assert_within(Path(quality_value), retarget_root, label="quality_json")
        if quality_value
        else None
    )
    safe_path = (
        _assert_within(Path(safe_value), retarget_root, label="safe_csv")
        if safe_value
        else None
    )
    if quality_path is None or not quality_path.is_file():
        errors.append("quality JSON is missing")
    if safe_path is None or not safe_path.is_file():
        errors.append("safe 18D CSV is missing")
    if not is_sha256(record.get("quality_json_sha256")):
        errors.append("quality JSON hash is invalid")
    if not is_sha256(record.get("safe_csv_sha256")):
        errors.append("safe CSV hash is invalid")
    if errors:
        raise ValueError(f"passed:{key}: " + "; ".join(errors))

    assert quality_path is not None and safe_path is not None
    quality = read_json(quality_path)
    if quality.get("quality_gate") != gates:
        raise ValueError(f"passed:{key}: manifest/quality gate mismatch")
    if quality.get("frames") != frames or quality.get("fps") != record.get("fps"):
        raise ValueError(f"passed:{key}: manifest/quality scale mismatch")
    if quality.get("annotation_kind") != FOUNDATION_ANNOTATION_KIND:
        raise ValueError(f"passed:{key}: quality lost foundation provenance")
    if quality.get("semantic_supervision_masks") != SEMANTIC_MASKS:
        raise ValueError(f"passed:{key}: quality semantic masks changed")
    if any(quality.get(field) is not False for field in FALSE_FLAGS):
        raise ValueError(f"passed:{key}: quality supervision flags changed")
    if (quality.get("retarget_segment") or {}) != retarget:
        raise ValueError(f"passed:{key}: quality retarget segment mismatch")
    outputs = quality.get("outputs") or {}
    if Path(str(outputs.get("safe_csv") or "")).resolve() != safe_path:
        raise ValueError(f"passed:{key}: quality safe CSV path mismatch")

    source_path = Path(str(record.get("source") or "")).resolve()
    if not source_path.is_file() or "BEAT2" not in source_path.parts:
        raise ValueError(f"passed:{key}: source is outside the BEAT2 tree")
    source_hash = source_hash_cache.get(source_path)
    if source_hash is None:
        source_hash = sha256_file(source_path)
        source_hash_cache[source_path] = source_hash
    if not ordinary.quality_passes(quality, dict(record), source_hash):
        raise ValueError(f"passed:{key}: original batch quality contract does not pass")

    if verify_artifacts:
        if sha256_file(quality_path) != record["quality_json_sha256"]:
            raise ValueError(f"passed:{key}: quality JSON hash mismatch")
        safe_hash, safe_lines = sha256_and_lines(safe_path)
        if safe_hash != record["safe_csv_sha256"]:
            raise ValueError(f"passed:{key}: safe CSV hash mismatch")
        if safe_lines != frames + 1:
            raise ValueError(f"passed:{key}: safe CSV row count mismatch")


def verify_qc(args: argparse.Namespace) -> dict[str, Any]:
    inventory = args.inventory.resolve()
    retarget_root = args.retarget_root.resolve()
    output = args.output.resolve()
    selector_summary = args.selector_summary.resolve()
    for path in (inventory, retarget_root, selector_summary):
        if not path.exists():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite QC receipt: {output}")
    configured = {
        "inventory": str(inventory),
        "retarget_root": str(retarget_root),
        "selector_summary": str(selector_summary),
        "output": str(output),
    }
    assert_no_forbidden_reference(configured, label="configured paths")
    inventory_records = read_jsonl(inventory)
    inventory_index = index_records(inventory_records, label="inventory")
    speaker_to_split = _speaker_mapping(read_json(selector_summary))
    for key, record in inventory_index.items():
        validate_foundation_metadata(
            record, speaker_to_split=speaker_to_split, label=f"inventory:{key}"
        )
    inventory_hash = sha256_file(inventory)
    status_path = retarget_root / "status.json"
    contract_path = retarget_root / ordinary.RUN_CONTRACT_FILENAME
    status = read_json(status_path)
    saved_contract = read_json(contract_path)
    contract, contract_hash = _validate_run_contract(
        status=status,
        saved=saved_contract,
        inventory_sha256=inventory_hash,
    )
    if status.get("run_state") != "finished":
        raise ValueError("retarget run is not terminal")
    if status.get("excluded_event_count") != 0:
        raise ValueError("foundation inventory produced excluded retarget tasks")
    if status.get("eligible_event_count") != len(inventory_records):
        raise ValueError("retarget eligible count differs from inventory")

    manifests = {
        name: retarget_root / f"{name}_manifest.jsonl"
        for name in ("passed", "failed", "pending", "excluded")
    }
    records = {name: read_jsonl(path) for name, path in manifests.items()}
    indexed = {
        name: index_records(rows, label=f"{name}_manifest")
        for name, rows in records.items()
    }
    all_ids: set[str] = set()
    for name, rows in indexed.items():
        overlap = all_ids & set(rows)
        if overlap:
            raise ValueError(f"retarget manifests overlap at {name}: {sorted(overlap)[:3]}")
        all_ids.update(rows)
    if all_ids != set(inventory_index):
        raise ValueError("retarget manifests do not exactly account for the inventory")
    if indexed["excluded"]:
        raise ValueError("foundation retarget has excluded records")

    terminal_count = len(indexed["passed"]) + len(indexed["failed"])
    is_full = (
        len(indexed["pending"]) == 0
        and terminal_count == len(inventory_records)
        and status.get("selected_event_count") == len(inventory_records)
        and status.get("terminal_event_count") == len(inventory_records)
    )
    if not is_full and not args.allow_partial_smoke:
        raise ValueError(
            "retarget coverage is partial; use --allow-partial-smoke only for a bounded smoke"
        )
    if not is_full and terminal_count < 1:
        raise ValueError("partial smoke contains no terminal retarget records")
    if status.get("terminal_event_count") != terminal_count:
        raise ValueError("status terminal count differs from result manifests")

    source_hash_cache: dict[Path, str] = {}
    for key, record in indexed["passed"].items():
        validate_passed_record(
            record,
            inventory_record=inventory_index[key],
            inventory_sha256=inventory_hash,
            run_contract_sha256=contract_hash,
            retarget_root=retarget_root,
            speaker_to_split=speaker_to_split,
            verify_artifacts=args.verify_artifacts,
            source_hash_cache=source_hash_cache,
        )
    for key, record in indexed["failed"].items():
        assert_no_forbidden_reference(record, label=f"failed:{key}")
        if record.get("status") == "passed" or record.get("accepted_for_training") is True:
            raise ValueError(f"failed:{key}: invalid failed-record admission")
        if key not in inventory_index:
            raise ValueError(f"failed:{key}: not present in the inventory")

    file_bindings_verified = (
        _verify_current_file_bindings(contract) if args.verify_artifacts else 0
    )
    split_records = Counter(
        str(record["fixed_split_assignment"]) for record in indexed["passed"].values()
    )
    split_frames = Counter(
        {
            split: sum(
                int(record["frames"])
                for record in indexed["passed"].values()
                if record["fixed_split_assignment"] == split
            )
            for split in ("train", "validation", "test")
        }
    )
    receipt = {
        "schema_version": 1,
        "artifact_kind": "beat2_motion_foundation_18d_qc_verification_v1",
        "created_at_utc": utc_now(),
        "verified": True,
        "mode": "full" if is_full else "partial_smoke",
        "accepted_for_training": False,
        "policy": {
            "dataset": "BEAT2",
            "original_18d_physical_qc_thresholds_unchanged": True,
            "speaker_split_unchanged": True,
            "text_audio_semantic_behavior_affect_payloads_absent": True,
            "all_conditioning_and_supervision_masks_false": True,
            "artifact_hashes_verified": bool(args.verify_artifacts),
            "current_run_contract_file_bindings_verified": file_bindings_verified,
        },
        "bindings": {
            "inventory": {
                "path": str(inventory),
                "records": len(inventory_records),
                "sha256": inventory_hash,
            },
            "selector_summary": {
                "path": str(selector_summary),
                "sha256": sha256_file(selector_summary),
            },
            "retarget_root": str(retarget_root),
            "status": {
                "path": str(status_path),
                "sha256": sha256_file(status_path),
            },
            "run_contract": {
                "path": str(contract_path),
                "sha256": sha256_file(contract_path),
                "canonical_contract_sha256": contract_hash,
            },
            "passed_manifest": {
                "path": str(manifests["passed"]),
                "records": len(indexed["passed"]),
                "sha256": sha256_file(manifests["passed"]),
            },
            "failed_manifest": {
                "path": str(manifests["failed"]),
                "records": len(indexed["failed"]),
                "sha256": sha256_file(manifests["failed"]),
            },
            "pending_manifest": {
                "path": str(manifests["pending"]),
                "records": len(indexed["pending"]),
                "sha256": sha256_file(manifests["pending"]),
            },
        },
        "accounting": {
            "inventory_records": len(inventory_records),
            "terminal_records": terminal_count,
            "passed_records": len(indexed["passed"]),
            "failed_records": len(indexed["failed"]),
            "pending_records": len(indexed["pending"]),
            "full_coverage_terminal": is_full,
        },
        "passed_scale": {
            "records": len(indexed["passed"]),
            "frames": sum(int(record["frames"]) for record in indexed["passed"].values()),
            "frame_coverage_hours": (
                sum(int(record["frames"]) for record in indexed["passed"].values())
                / 108000.0
            ),
            "by_fixed_split": {
                split: {
                    "records": split_records[split],
                    "frames": split_frames[split],
                    "frame_coverage_hours": split_frames[split] / 108000.0,
                }
                for split in ("train", "validation", "test")
            },
        },
    }
    assert_no_forbidden_reference(receipt, label="QC receipt")
    atomic_json(output, receipt)
    return receipt


def _binding_path(
    parent: Mapping[str, Any], name: str, *, verify_hash: bool = True
) -> Path:
    binding = (parent.get("bindings") or {}).get(name)
    if not isinstance(binding, Mapping):
        raise ValueError(f"missing artifact binding: {name}")
    path = Path(str(binding.get("path") or "")).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if verify_hash and sha256_file(path) != binding.get("sha256"):
        raise ValueError(f"artifact binding hash mismatch: {name}")
    return path


def _adjudicated_record(source: Mapping[str, Any], *, smoke_only: bool) -> dict[str, Any]:
    record = {
        key: value for key, value in source.items() if key not in STRIP_FROM_ADJUDICATED
    }
    source_hash = canonical_sha256(source)
    record.update(
        {
            "formal_episode_contract": MOTION_ONLY_EPISODE_CONTRACT,
            "accepted_for_training": False,
            "training_admission_status": (
                "smoke_only_not_training_admitted"
                if smoke_only
                else "physical_qc_adjudicated_pending_loader_and_provenance_admission"
            ),
            "emotion_conditioning_mask": False,
            "independent_review": {
                "present": False,
                "status": "not_applicable_unlabeled_motion_foundation",
                "training_acceptance": False,
                "scope": "physical_motion_only",
            },
            "adjudication": {
                "status": "motion_foundation_physical_qc_verified",
                "reasons": [],
                "training_admitted": False,
            },
            "motion_only_admission": {
                "physical_qc_only": True,
                "semantic_review_required": False,
                "independent_semantic_review_claimed": False,
                "text_conditioning_enabled": False,
                "emotion_conditioning_enabled": False,
                "audio_conditioning_enabled": False,
                "native_variable_length": True,
                "fixed_duration_training_unit": False,
                "source_record_sha256": source_hash,
            },
            "motion_18d": {
                "state": "passed",
                "partition": "adjudicated_motion_foundation_not_training_admitted",
                "reasons": [],
                "output_contract": ordinary.ULA_V2_18D_CONTRACT,
                "action_dim": 18,
                "frames": int(source["frames"]),
                "csv_rows": int(source["frames"]),
                "fps": float(source["fps"]),
                "quality_gate": dict(source["quality_gate"]),
                "quality_json": str(Path(str(source["quality_json"])).resolve()),
                "quality_sha256": source["quality_json_sha256"],
                "safe_csv": str(Path(str(source["safe_csv"])).resolve()),
                "safe_csv_sha256": source["safe_csv_sha256"],
                "retarget_segment": dict(source["retarget_segment"]),
                "source_window_frames": int(source["training_segment"]["frame_count"]),
                "upstream_lineage": dict(source.get("lineage_hashes") or {}),
            },
        }
    )
    return record


def adjudicate(args: argparse.Namespace) -> dict[str, Any]:
    receipt_path = args.qc_verification.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite adjudication directory: {output_dir}")
    receipt = read_json(receipt_path)
    assert_no_forbidden_reference(receipt, label="QC receipt")
    if (
        receipt.get("verified") is not True
        or receipt.get("accepted_for_training") is not False
    ):
        raise ValueError("QC receipt is not a verified non-training receipt")
    smoke_only = receipt.get("mode") == "partial_smoke"
    if smoke_only != bool(args.smoke_only):
        raise ValueError("--smoke-only must exactly match the QC receipt mode")
    passed_manifest = _binding_path(receipt, "passed_manifest")
    records = read_jsonl(passed_manifest)
    expected_records = (
        ((receipt.get("bindings") or {}).get("passed_manifest") or {}).get("records")
    )
    if len(records) != expected_records:
        raise ValueError("passed manifest record count changed")
    if not records:
        raise ValueError("QC receipt contains no passed motion")
    adjudicated = [_adjudicated_record(record, smoke_only=smoke_only) for record in records]
    for record in adjudicated:
        assert_no_forbidden_reference(record, label=f"adjudicated:{clip_id(record)}")
        if any(field in record for field in CONDITIONING_PAYLOAD_FIELDS):
            raise ValueError(f"adjudicated:{clip_id(record)} retained conditioning metadata")
        if record.get("accepted_for_training") is not False:
            raise ValueError(f"adjudicated:{clip_id(record)} became training-admitted")

    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "adjudicated_motion_only.jsonl"
    write_jsonl(manifest_path, adjudicated)
    frame_counts = [int(record["frames"]) for record in adjudicated]
    split_records = Counter(
        str(record["fixed_split_assignment"]) for record in adjudicated
    )
    split_frames = Counter(
        {
            split: sum(
                int(record["frames"])
                for record in adjudicated
                if record["fixed_split_assignment"] == split
            )
            for split in ("train", "validation", "test")
        }
    )
    report = {
        "schema_version": 1,
        "artifact_kind": "beat2_motion_foundation_physical_qc_adjudication_v1",
        "created_at_utc": utc_now(),
        "mode": receipt["mode"],
        "accepted_for_training": False,
        "formal_release_allowed": False,
        "reason": (
            "bounded_smoke_only"
            if smoke_only
            else "loader_contract_and_explicit_provenance_admission_pending"
        ),
        "policy": {
            "dataset": "BEAT2",
            "physical_qc_only": True,
            "source_fixed_speaker_splits_preserved": True,
            "semantic_behavior_emotion_affect_supervision_masked": True,
            "text_audio_conditioning_absent": True,
            "no_fixed_duration_training_unit": True,
        },
        "bindings": {
            "qc_verification": {
                "path": str(receipt_path),
                "sha256": sha256_file(receipt_path),
            },
            "source_passed_manifest": {
                "path": str(passed_manifest),
                "sha256": sha256_file(passed_manifest),
            },
        },
        "outputs": {
            "adjudicated_motion_only": {
                "path": str(manifest_path),
                "records": len(adjudicated),
                "sha256": sha256_file(manifest_path),
            }
        },
        "scale": {
            "records": len(adjudicated),
            "frames": sum(frame_counts),
            "frame_coverage_hours": sum(frame_counts) / 108000.0,
            "sample_span_hours": (
                sum(max(0, frames - 1) for frames in frame_counts) / 108000.0
            ),
            "frame_count_min": min(frame_counts),
            "frame_count_median": float(statistics.median(frame_counts)),
            "frame_count_max": max(frame_counts),
            "by_fixed_split": {
                split: {
                    "records": split_records[split],
                    "frames": split_frames[split],
                    "frame_coverage_hours": split_frames[split] / 108000.0,
                }
                for split in ("train", "validation", "test")
            },
        },
    }
    atomic_json(output_dir / "motion_only_adjudication_report.json", report)
    return report


def lock(args: argparse.Namespace) -> dict[str, Any]:
    receipt_path = args.qc_verification.resolve()
    report_path = args.release_report.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite provenance lock: {output}")
    receipt = read_json(receipt_path)
    report = read_json(report_path)
    assert_no_forbidden_reference(receipt, label="QC receipt")
    assert_no_forbidden_reference(report, label="adjudication report")
    if receipt.get("verified") is not True:
        raise ValueError("QC receipt is not verified")
    if report.get("accepted_for_training") is not False:
        raise ValueError("adjudication report unexpectedly admits training")
    qc_binding = (report.get("bindings") or {}).get("qc_verification") or {}
    if (
        Path(str(qc_binding.get("path") or "")).resolve() != receipt_path
        or qc_binding.get("sha256") != sha256_file(receipt_path)
    ):
        raise ValueError("adjudication report is not bound to this QC receipt")
    output_binding = (report.get("outputs") or {}).get(
        "adjudicated_motion_only"
    )
    if not isinstance(output_binding, Mapping):
        raise ValueError("adjudication output binding is missing")
    adjudicated_path = Path(str(output_binding.get("path") or "")).resolve()
    if (
        not adjudicated_path.is_file()
        or sha256_file(adjudicated_path) != output_binding.get("sha256")
    ):
        raise ValueError("adjudicated manifest hash mismatch")
    inventory_binding = (receipt.get("bindings") or {}).get("inventory") or {}
    run_binding = (receipt.get("bindings") or {}).get("run_contract") or {}
    lock_value = {
        "schema_version": 1,
        "artifact_kind": "beat2_motion_foundation_pending_provenance_lock_v1",
        "created_at_utc": utc_now(),
        "mode": receipt["mode"],
        "accepted_for_training": False,
        "formal_release_allowed": False,
        "training_authorized_by_this_lock": False,
        "admission_blockers": [
            (
                "bounded_smoke_not_full_dataset"
                if receipt["mode"] == "partial_smoke"
                else "explicit_loader_contract_and_training_authorization_pending"
            )
        ],
        "policy": {
            "dataset": "BEAT2",
            "fixed_speaker_splits": True,
            "original_18d_physical_qc_thresholds_unchanged": True,
            "motion_only": True,
            "text_audio_conditioning_absent": True,
            "semantic_behavior_emotion_affect_supervision_masked": True,
        },
        "locked_artifacts": {
            "source_inventory": dict(inventory_binding),
            "retarget_run_contract": dict(run_binding),
            "qc_verification": {
                "path": str(receipt_path),
                "sha256": sha256_file(receipt_path),
            },
            "adjudication_report": {
                "path": str(report_path),
                "sha256": sha256_file(report_path),
            },
            "adjudicated_motion_only": dict(output_binding),
        },
        "dataset_scale": dict(report.get("scale") or {}),
    }
    assert_no_forbidden_reference(lock_value, label="provenance lock")
    atomic_json(output, lock_value)
    return lock_value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify-qc")
    verify.add_argument("--inventory", type=Path, required=True)
    verify.add_argument("--retarget-root", type=Path, required=True)
    verify.add_argument(
        "--selector-summary", type=Path, default=DEFAULT_SELECTOR_SUMMARY
    )
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--verify-artifacts", action="store_true")
    verify.add_argument("--allow-partial-smoke", action="store_true")

    adjudication = subparsers.add_parser("adjudicate")
    adjudication.add_argument("--qc-verification", type=Path, required=True)
    adjudication.add_argument("--output-dir", type=Path, required=True)
    adjudication.add_argument("--smoke-only", action="store_true")

    lock_parser = subparsers.add_parser("lock")
    lock_parser.add_argument("--qc-verification", type=Path, required=True)
    lock_parser.add_argument("--release-report", type=Path, required=True)
    lock_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "verify-qc":
        result = verify_qc(args)
    elif args.command == "adjudicate":
        result = adjudicate(args)
    elif args.command == "lock":
        result = lock(args)
    else:  # pragma: no cover
        raise ValueError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
