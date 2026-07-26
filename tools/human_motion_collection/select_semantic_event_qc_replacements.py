#!/usr/bin/env python3
"""Select deterministic same-stratum replacements after terminal 18D QC."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "1.0.0"
ALLOWED_SPLITS = ("train", "validation", "test")
ALLOWED_DYNAMIC_BANDS = {"low", "medium"}


def stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> str:
    payload = "".join(stable_json(record) + "\n" for record in records).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: object) -> str:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return hashlib.sha256(payload).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            records.append(value)
    return records


def stratum(record: dict[str, Any]) -> tuple[str, str, str, str]:
    event = record.get("semantic_event")
    if not isinstance(event, dict):
        raise ValueError(f"{record.get('clip_id')}: semantic_event is missing")
    split = str(record.get("fixed_split_assignment") or record.get("pilot_split") or "")
    value = (
        split,
        str(record.get("emotion_id") or ""),
        str(event.get("category") or ""),
        str(event.get("intensity") or ""),
    )
    if split not in ALLOWED_SPLITS or any(not item for item in value[1:]):
        raise ValueError(f"{record.get('clip_id')}: invalid fixed split or semantic stratum")
    return value


def stratum_key(value: tuple[str, str, str, str]) -> str:
    return "|".join(value)


def _unique_by_clip(records: list[dict[str, Any]], *, name: str) -> dict[str, dict]:
    indexed = {}
    for record in records:
        clip_id = str(record.get("clip_id") or record.get("task_id") or "").strip()
        if not clip_id or clip_id in indexed:
            raise ValueError(f"{name} requires unique non-empty clip_id values")
        indexed[clip_id] = record
    return indexed


def _result_records(paths: list[Path], expected_status: str) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        for record in load_jsonl(path):
            status = record.get("status")
            if expected_status == "passed" and status != "passed":
                raise ValueError(f"{path}: non-passed status {status!r} in passed manifest")
            if expected_status == "failed" and status == "passed":
                raise ValueError(f"{path}: passed row in failed manifest")
            records.append(record)
    return records


def build_replacements(
    candidates_path: Path,
    selected_path: Path,
    passed_paths: list[Path],
    failed_paths: list[Path],
    output_path: Path,
    *,
    round_number: int = 1,
) -> dict[str, Any]:
    if round_number < 1:
        raise ValueError("round_number must be positive")
    candidates_path = candidates_path.resolve()
    selected_path = selected_path.resolve()
    passed_paths = [path.resolve() for path in passed_paths]
    failed_paths = [path.resolve() for path in failed_paths]
    candidates = load_jsonl(candidates_path)
    selected = load_jsonl(selected_path)
    passed = _result_records(passed_paths, "passed")
    failed = _result_records(failed_paths, "failed")
    candidate_by_id = _unique_by_clip(candidates, name="candidates")
    selected_by_id = _unique_by_clip(selected, name="selected")
    passed_by_id = _unique_by_clip(passed, name="passed manifests")
    failed_by_id = _unique_by_clip(failed, name="failed manifests")
    attempted_ids = set(passed_by_id) | set(failed_by_id)
    if set(passed_by_id) & set(failed_by_id):
        raise ValueError("One clip appears in both passed and failed manifests")
    pending_selected = sorted(set(selected_by_id) - attempted_ids)
    if pending_selected:
        raise ValueError(
            f"Original selected pilot still has {len(pending_selected)} pending clips"
        )
    unknown_results = sorted(attempted_ids - set(candidate_by_id))
    if unknown_results:
        raise ValueError(f"QC manifests contain unknown candidates: {unknown_results[:3]}")

    target_counts = Counter(stratum(record) for record in selected)
    passed_counts = Counter(stratum(record) for record in passed)
    deficits = {
        key: max(0, target_counts[key] - passed_counts[key])
        for key in sorted(target_counts)
    }
    buckets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        clip_id = str(record.get("clip_id") or "")
        if clip_id in attempted_ids:
            continue
        if record.get("pilot_dynamic_band") not in ALLOWED_DYNAMIC_BANDS:
            continue
        segment = record.get("training_segment")
        if not isinstance(segment, dict) or segment.get("fixed_window_sec") is not None:
            continue
        buckets[stratum(record)].append(record)
    for bucket in buckets.values():
        bucket.sort(
            key=lambda record: (
                int(record.get("pilot_candidate_rank_within_split_stratum") or 10**12),
                str(record["clip_id"]),
            )
        )

    contract = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "semantic_event_same_split_stratum_qc_replacements",
        "round_number": int(round_number),
        "candidate_manifest_sha256": sha256_file(candidates_path),
        "selected_manifest_sha256": sha256_file(selected_path),
        "passed_manifest_sha256": [sha256_file(path) for path in passed_paths],
        "failed_manifest_sha256": [sha256_file(path) for path in failed_paths],
        "dynamic_bands": sorted(ALLOWED_DYNAMIC_BANDS),
        "fixed_window_sec": None,
        "selection_order": "existing_pilot_candidate_rank_then_clip_id",
        "split_constraint": "same_fixed_speaker_split",
        "stratum_constraint": "same_emotion_category_intensity",
        "accepted_for_training": False,
    }
    contract_sha = hashlib.sha256(stable_json(contract).encode("utf-8")).hexdigest()
    replacements = []
    unfillable = {}
    for key in sorted(deficits):
        required = deficits[key]
        available = buckets.get(key, [])
        chosen = available[:required]
        for record in chosen:
            replacements.append(
                {
                    **record,
                    "qc_replacement_round": int(round_number),
                    "qc_replacement_for_stratum": stratum_key(key),
                    "qc_replacement_selection_status": "selected_pending_retarget_qc",
                    "qc_replacement_contract_sha256": contract_sha,
                    "training_admission_status": "pending_retarget_qc",
                    "accepted_for_training": False,
                }
            )
        if len(chosen) < required:
            unfillable[stratum_key(key)] = required - len(chosen)

    replacements.sort(
        key=lambda record: (
            ALLOWED_SPLITS.index(record["fixed_split_assignment"]),
            stratum(record),
            int(record.get("pilot_candidate_rank_within_split_stratum") or 10**12),
            str(record["clip_id"]),
        )
    )
    output_sha = atomic_jsonl(output_path.resolve(), replacements)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": contract["artifact_kind"],
        "replacement_contract": contract,
        "replacement_contract_sha256": contract_sha,
        "target_count": sum(target_counts.values()),
        "passed_count": sum(min(passed_counts[key], target_counts[key]) for key in target_counts),
        "failed_or_missing_target_count": sum(deficits.values()),
        "replacement_count": len(replacements),
        "unfillable_deficits": unfillable,
        "coverage_fillable": not unfillable,
        "deficits_by_split_stratum": {
            stratum_key(key): value for key, value in deficits.items() if value
        },
        "replacement_counts_by_split": dict(
            sorted(Counter(record["fixed_split_assignment"] for record in replacements).items())
        ),
        "output": {
            "path": str(output_path.resolve()),
            "records": len(replacements),
            "sha256": output_sha,
        },
    }
    atomic_json(output_path.with_suffix(".summary.json").resolve(), summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--passed-manifest", type=Path, action="append", required=True)
    parser.add_argument("--failed-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--round", type=int, default=1, dest="round_number")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_replacements(
        args.candidates,
        args.selected,
        args.passed_manifest,
        args.failed_manifest,
        args.output,
        round_number=args.round_number,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["coverage_fillable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
