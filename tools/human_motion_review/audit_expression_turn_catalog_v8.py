#!/usr/bin/env python3
"""Fail-closed audit for a BEAT2 variable-length expression-turn catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.human_motion_review.expression_turn_contract import (
    SELECTION_LINEAGE_FIELDS,
    validate_expression_turn_candidate,
)


STEM = "beat2_expression_turn_v8"
EXPECTED_FILES = {
    f"{STEM}.candidates.jsonl",
    f"{STEM}.rejected.jsonl",
    f"{STEM}.representative100.jsonl",
    f"{STEM}.stress100.jsonl",
    f"{STEM}.summary.json",
}
SPLITS = {"train", "validation", "test"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{line_number} is not an object")
            records.append(value)
    return records


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def proportional_quotas(counts: Counter[tuple[str, str]], total: int) -> dict[str, int]:
    population = sum(counts.values())
    raw = {key: total * count / population for key, count in counts.items()}
    quotas = {key: math.floor(value) for key, value in raw.items()}
    remainder = total - sum(quotas.values())
    order = sorted(counts, key=lambda key: (-(raw[key] - quotas[key]), str(key)))
    for key in order[:remainder]:
        quotas[key] += 1
    return {"|".join(key): value for key, value in sorted(quotas.items())}


def joint_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    values = Counter(
        (str(record["duration_band"]), str(record["event_count_band"]))
        for record in records
    )
    return {"|".join(key): value for key, value in sorted(values.items())}


def counts(records: Iterable[dict[str, Any]], key) -> dict[str, int]:
    return dict(sorted(Counter(str(key(record)) for record in records).items()))


def npz_array_shape(path: Path, member: str) -> tuple[int, ...]:
    with zipfile.ZipFile(path) as archive, archive.open(f"{member}.npy") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, _fortran, _dtype = np.lib.format.read_array_header_1_0(handle)
        else:
            shape, _fortran, _dtype = np.lib.format.read_array_header_2_0(handle)
    return tuple(int(value) for value in shape)


class Audit:
    def __init__(self) -> None:
        self.checks: dict[str, bool] = {}
        self.errors: list[dict[str, Any]] = []

    def check(self, name: str, condition: bool, detail: Any = None) -> None:
        passed = bool(condition)
        self.checks[name] = self.checks.get(name, True) and passed
        if not passed and len(self.errors) < 200:
            self.errors.append({"check": name, "detail": detail})

    def capture(self, name: str, fn) -> Any:
        try:
            return fn()
        except Exception as error:  # Audit must emit a report on malformed input.
            self.check(name, False, f"{type(error).__name__}: {error}")
            return None


def _summary_counts(audit: Audit, summary: dict[str, Any], prefix: str,
                    records: list[dict[str, Any]]) -> None:
    frames = [int(record["training_segment"]["frame_count"]) for record in records]
    expected = {
        f"{prefix}_count": len(records),
        f"{prefix}_counts_by_split": counts(
            records, lambda item: item["fixed_split_assignment"]
        ),
        f"{prefix}_counts_by_emotion_metadata": counts(
            records, lambda item: item["emotion_id"]
        ),
        f"{prefix}_counts_by_dominant_category": counts(
            records, lambda item: item["expression_turn"]["dominant_official_category"]
        ),
        f"{prefix}_counts_by_event_count_band": counts(
            records, lambda item: item["event_count_band"]
        ),
        f"{prefix}_counts_by_duration_band": counts(
            records, lambda item: item["duration_band"]
        ),
        f"{prefix}_source_count": len({record["source_clip_id"] for record in records}),
        f"{prefix}_source_disjoint": (
            len({record["source_clip_id"] for record in records}) == len(records)
        ),
        f"{prefix}_speaker_count": len({record["speaker_key"] for record in records}),
        f"{prefix}_frame_count": sum(frames),
        f"{prefix}_frame_count_min": min(frames),
        f"{prefix}_frame_count_max": max(frames),
        f"{prefix}_frame_coverage_hours_at_30hz": round(sum(frames) / 30 / 3600, 8),
        f"{prefix}_sample_span_hours_at_30hz": round(
            sum(max(frame - 1, 0) for frame in frames) / 30 / 3600, 8
        ),
    }
    for field, value in expected.items():
        audit.check(f"summary.{field}", summary.get(field) == value, {
            "expected": value, "actual": summary.get(field)
        })


def audit_catalog(catalog_dir: Path, beat2_root: Path) -> dict[str, Any]:
    audit = Audit()
    paths = {name: catalog_dir / name for name in EXPECTED_FILES}
    actual_names = {path.name for path in catalog_dir.iterdir() if path.is_file()}
    audit.check("catalog.exact_file_set", actual_names == EXPECTED_FILES, {
        "expected": sorted(EXPECTED_FILES), "actual": sorted(actual_names)
    })
    audit.check("catalog.no_legacy_pilot_file", not any("pilot" in name for name in actual_names))
    for name, path in paths.items():
        audit.check(f"catalog.file_exists.{name}", path.is_file())

    summary = audit.capture(
        "summary.decode",
        lambda: json.loads(paths[f"{STEM}.summary.json"].read_text(encoding="utf-8")),
    )
    if not isinstance(summary, dict):
        return {"passed": False, "checks": audit.checks, "errors": audit.errors}

    candidates = audit.capture(
        "candidates.decode", lambda: read_jsonl(paths[f"{STEM}.candidates.jsonl"])
    ) or []
    rejected = audit.capture(
        "rejected.decode", lambda: read_jsonl(paths[f"{STEM}.rejected.jsonl"])
    ) or []
    representative = audit.capture(
        "representative100.decode",
        lambda: read_jsonl(paths[f"{STEM}.representative100.jsonl"]),
    ) or []
    stress = audit.capture(
        "stress100.decode", lambda: read_jsonl(paths[f"{STEM}.stress100.jsonl"])
    ) or []

    manifest_hashes = {name: sha256_file(path) for name, path in paths.items() if name != f"{STEM}.summary.json"}
    audit.check("summary.output_sha256", summary.get("output_sha256") == manifest_hashes, {
        "expected": manifest_hashes, "actual": summary.get("output_sha256")
    })
    audit.check("summary.candidate_count", summary.get("candidate_count") == len(candidates))
    audit.check("summary.rejected_count", summary.get("rejected_count") == len(rejected))
    audit.check("summary.accepted_for_training_zero", summary.get("accepted_for_training") == 0)
    audit.check("summary.duration_accounting", summary.get("duration_accounting") == {
        "frame_coverage": "sum(frame_count)/fps",
        "frame_coverage_is_not_planner_target": True,
        "planner_duration_basis": "sample_span",
        "sample_span": "sum(max(frame_count-1,0))/fps",
    })
    audit.check("summary.no_ambiguous_duration_hours_fields", not any(
        key.endswith("_duration_hours_at_30hz") for key in summary
    ))
    contract = summary.get("expression_turn_contract")
    audit.check("summary.contract_hash", isinstance(contract, dict) and stable_sha256(contract) == summary.get("expression_turn_contract_sha256"))

    external_paths = {
        "source_inventory": Path(str(summary.get("input", ""))),
        "split_assignment": Path(str(summary.get("split_assignment", ""))),
        "builder": Path(__file__).parents[1] / "human_motion_collection" / "build_beat2_expression_turn_v8.py",
    }
    expected_external_hashes = {
        "source_inventory": summary.get("input_sha256"),
        "split_assignment": summary.get("split_assignment_sha256"),
        "builder": contract.get("builder_script_sha256") if isinstance(contract, dict) else None,
    }
    for label, path in external_paths.items():
        audit.check(f"external.{label}.exists", path.is_file(), str(path))
        if path.is_file():
            audit.check(
                f"external.{label}.sha256",
                sha256_file(path) == expected_external_hashes[label],
                str(path),
            )

    source_records = audit.capture(
        "source_inventory.decode", lambda: read_jsonl(external_paths["source_inventory"])
    ) or []
    source_events: dict[str, tuple[str, dict[str, Any]]] = {}
    source_metadata: dict[str, dict[str, Any]] = {}
    for record in source_records:
        event_hash = stable_sha256(record)
        source_events[str(record["clip_id"])] = (event_hash, record)
        source_metadata.setdefault(str(record["source_group_key"]), record)

    binding_common = {
        "expression_turn_contract_sha256": summary.get("expression_turn_contract_sha256"),
        "source_inventory_manifest_sha256": summary.get("input_sha256"),
        "split_assignment_manifest_sha256": summary.get("split_assignment_sha256"),
    }
    sets = (
        ("candidates", candidates, f"{STEM}.candidates.jsonl", None, False),
        ("stress100", stress, f"{STEM}.stress100.jsonl", "stress100", True),
        ("representative100", representative, f"{STEM}.representative100.jsonl", "representative100", True),
    )
    validation_counts: dict[str, int] = {}
    for set_name, records, filename, selection_kind, require_selection in sets:
        failures = 0
        binding = {
            **binding_common,
            "retarget_input_manifest_sha256": manifest_hashes.get(filename),
            "selection_kind": selection_kind,
            "require_selection_record": require_selection,
        }
        for index, record in enumerate(records, 1):
            try:
                validate_expression_turn_candidate(record, catalog_binding=binding)
            except Exception as error:
                failures += 1
                audit.check(
                    f"contract.{set_name}", False,
                    {"line": index, "clip_id": record.get("clip_id"), "error": f"{type(error).__name__}: {error}"},
                )
        audit.check(f"contract.{set_name}", failures == 0, {"failures": failures})
        validation_counts[set_name] = len(records) - failures

    candidate_by_hash = {str(record.get("expression_turn_record_sha256")): record for record in candidates}
    audit.check("candidates.unique_clip_id", len({record.get("clip_id") for record in candidates}) == len(candidates))
    audit.check("candidates.unique_record_sha256", len(candidate_by_hash) == len(candidates))
    source_to_split: dict[str, set[str]] = defaultdict(set)
    speaker_to_split: dict[str, set[str]] = defaultdict(set)
    source_interval_by_group: dict[str, tuple[int, int]] = {}
    source_motion: dict[str, tuple[str, str]] = {}
    upstream_failures = 0
    for record in candidates:
        source_key = str(record["source_group_key"])
        split = str(record["fixed_split_assignment"])
        source_to_split[source_key].add(split)
        speaker_to_split[str(record["speaker_key"])].add(split)
        interval = record["context_plan"]["source_interval"]
        source_interval_by_group.setdefault(source_key, (int(interval["start_frame"]), int(interval["end_frame_exclusive"])))
        audit.check("boundaries.source_interval_consistent_per_source", source_interval_by_group[source_key] == (int(interval["start_frame"]), int(interval["end_frame_exclusive"])), source_key)
        source_motion.setdefault(source_key, (str(record["motion_relpath"]), str(record["motion_sha256"])))
        audit.check("source.motion_metadata_consistent_per_source", source_motion[source_key] == (str(record["motion_relpath"]), str(record["motion_sha256"])), source_key)
        upstream_ids = [str(span["upstream_clip_id"]) for span in record["expression_turn"]["included_event_spans"]]
        upstream_hashes = [str(value) for value in record["upstream_event_record_sha256"]]
        expected_hashes = [source_events.get(event_id, (None, None))[0] for event_id in upstream_ids]
        if upstream_hashes != expected_hashes:
            upstream_failures += 1
        source_record = source_metadata.get(source_key)
        if source_record is None or any(
            record.get(field) != source_record.get(field)
            for field in ("motion_relpath", "motion_sha256", "annotation_relpath", "annotation_sha256", "textgrid_relpath", "textgrid_sha256")
        ):
            upstream_failures += 1
    audit.check("source.upstream_event_and_asset_lineage", upstream_failures == 0, {"failures": upstream_failures})
    audit.check("splits.source_disjoint", all(len(values) == 1 for values in source_to_split.values()))
    audit.check("splits.speaker_disjoint", all(len(values) == 1 for values in speaker_to_split.values()))

    source_header_failures = 0
    for source_key, (motion_relpath, _motion_hash) in source_motion.items():
        motion_path = beat2_root / motion_relpath
        try:
            shape = npz_array_shape(motion_path, "poses")
            expected_interval = source_interval_by_group[source_key]
            if len(shape) < 1 or expected_interval != (0, shape[0]):
                source_header_failures += 1
        except Exception:
            source_header_failures += 1
    audit.check("boundaries.source_interval_matches_npz_pose_shape", source_header_failures == 0, {"failures": source_header_failures})

    candidate_frames = [int(record["training_segment"]["frame_count"]) for record in candidates]
    candidate_expected = {
        "candidate_counts_by_duration_band": counts(candidates, lambda item: item["duration_band"]),
        "candidate_counts_by_event_count_band": counts(candidates, lambda item: item["event_count_band"]),
        "candidate_counts_by_split": counts(candidates, lambda item: item["fixed_split_assignment"]),
        "candidate_counts_by_emotion_metadata": counts(candidates, lambda item: item["emotion_id"]),
        "candidate_distinct_frame_count_count": len(set(candidate_frames)),
        "candidate_frame_count_min": min(candidate_frames),
        "candidate_frame_count_max": max(candidate_frames),
        "candidate_frame_coverage_hours_at_30hz": round(sum(candidate_frames) / 30 / 3600, 8),
        "candidate_sample_span_hours_at_30hz": round(sum(frame - 1 for frame in candidate_frames) / 30 / 3600, 8),
    }
    for field, value in candidate_expected.items():
        audit.check(f"summary.{field}", summary.get(field) == value, {"expected": value, "actual": summary.get(field)})
    rejection_counts = counts(rejected, lambda item: item["reason"])
    audit.check("summary.rejection_counts_by_reason", summary.get("rejection_counts_by_reason") == rejection_counts)
    audit.check("rejected.fail_closed", all(record.get("accepted_for_training") is False for record in rejected))

    for name, records in (("stress100", stress), ("representative100", representative)):
        ranks = [record.get("expression_turn_selection_rank") for record in records]
        audit.check(f"{name}.ranks_complete", sorted(ranks) == list(range(1, len(records) + 1)))
        audit.check(f"{name}.source_disjoint", len({record["source_clip_id"] for record in records}) == len(records))
        audit.check(f"{name}.membership_exact", all(
            record.get("expression_turn_record_sha256") in candidate_by_hash
            and {key: value for key, value in record.items() if key not in SELECTION_LINEAGE_FIELDS}
            == candidate_by_hash[record["expression_turn_record_sha256"]]
            for record in records
        ))
        _summary_counts(audit, summary, name, records)

    audit.check("stress100.expected_split_counts", counts(stress, lambda item: item["fixed_split_assignment"]) == {"test": 15, "train": 70, "validation": 15})
    audit.check("stress100.role", summary.get("stress100_role") == "compound_long_sequence_stress_review_not_pool_acceptance_estimator")
    expected_joint = proportional_quotas(Counter(
        (str(record["duration_band"]), str(record["event_count_band"])) for record in candidates
    ), len(representative))
    actual_joint = joint_counts(representative)
    audit.check("representative100.joint_proportional_quotas", actual_joint == expected_joint, {"expected": expected_joint, "actual": actual_joint})
    audit.check("representative100.split_coverage", {record["fixed_split_assignment"] for record in representative} == SPLITS)
    audit.check("representative100.role", summary.get("representative100_role") == "pool_physical_quality_rate_estimator")
    review_sets = contract.get("review_sets", {}) if isinstance(contract, dict) else {}
    audit.check("representative100.official_emotion_not_selection", review_sets.get("representative100", {}).get("official_emotion_used_for_selection") is False)

    metrics = {
        "candidate_records": len(candidates),
        "rejected_records": len(rejected),
        "stress100_records": len(stress),
        "representative100_records": len(representative),
        "source_records": len(source_motion),
        "contract_valid_records": validation_counts,
        "candidate_frame_count_min": min(candidate_frames) if candidate_frames else None,
        "candidate_frame_count_max": max(candidate_frames) if candidate_frames else None,
        "candidate_distinct_frame_counts": len(set(candidate_frames)),
        "candidate_frame_coverage_hours_at_30hz": round(sum(candidate_frames) / 30 / 3600, 8),
        "candidate_sample_span_hours_at_30hz": round(sum(frame - 1 for frame in candidate_frames) / 30 / 3600, 8),
        "representative100_joint_counts": actual_joint,
        "representative100_expected_joint_quotas": expected_joint,
        "manifest_sha256": manifest_hashes,
    }
    return {
        "artifact_kind": "beat2_expression_turn_v8_catalog_local_audit",
        "schema_version": "1.0.0",
        "catalog_dir": str(catalog_dir.resolve()),
        "passed": all(audit.checks.values()) and not audit.errors,
        "checks": dict(sorted(audit.checks.items())),
        "metrics": metrics,
        "errors": audit.errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-dir", type=Path, required=True)
    parser.add_argument(
        "--beat2-root", type=Path,
        default=Path("/home/gez/nas/cloud/gez/human_motion/raw/BEAT2"),
    )
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_catalog(args.catalog_dir, args.beat2_root)
    report["audit_payload_sha256"] = stable_sha256(report)
    payload = (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_write(args.report, payload)
    report_sha256 = hashlib.sha256(payload).hexdigest()
    atomic_write(args.report.with_suffix(args.report.suffix + ".sha256"), f"{report_sha256}  {args.report.name}\n".encode("ascii"))
    print(json.dumps({"passed": report["passed"], "report": str(args.report), "report_sha256": report_sha256, "metrics": report["metrics"], "errors": report["errors"]}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
