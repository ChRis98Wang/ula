#!/usr/bin/env python3
"""Independently audit a rebuilt BEAT2 expression-turn v8 catalog."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import zipfile

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.human_motion_review.expression_turn_contract import (  # noqa: E402
    validate_expression_turn_candidate,
)


FILE_NAMES = {
    "candidates": "beat2_expression_turn_v8.candidates.jsonl",
    "rejected": "beat2_expression_turn_v8.rejected.jsonl",
    "representative100": "beat2_expression_turn_v8.representative100.jsonl",
    "stress100": "beat2_expression_turn_v8.stress100.jsonl",
    "summary": "beat2_expression_turn_v8.summary.json",
}
EXPECTED_SEMANTIC_MASKS = {
    "official_category": False,
    "robot_observable_motion_form": False,
    "communicative_intent": False,
    "prompt_text": False,
    "legacy_gesture": False,
}
FAIL_CLOSED_FIELDS = {
    "accepted_for_training": False,
    "affect_observable_supervision_mask": False,
    "emotion_supervision_mask": False,
    "official_category_conditioning_enabled": False,
    "official_emotion_conditioning_enabled": False,
}
FORBIDDEN_LEGACY_KEYS = {
    "expression_turn_pilot_rank",
    "expression_turn_pilot_record_sha256",
    "expression_turn_pilot_selection_status",
    "official_semantic_event",
    "semantic_event",
    "semantic_gesture",
}
AMBIGUOUS_DURATION_KEYS = {
    "candidate_duration_hours_at_30hz",
    "representative100_duration_hours_at_30hz",
    "stress100_duration_hours_at_30hz",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-dir", type=Path, required=True)
    parser.add_argument("--beat2-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def stable_json(value) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            records.append(value)
    return records


def walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_keys(child)


def npy_shape_from_npz(path: Path, member="poses.npy") -> tuple[int, ...]:
    """Infer BEAT2 pose shape from stored NPY member sizes.

    Official BEAT2 motion archives use NumPy v1 headers padded to 128 bytes and
    store poses[T,165]/expressions[T,200] as float32 and trans[T,3] as float64.
    Reading the ZIP central directory avoids a second random NAS seek per source
    while the three independent member sizes still have to agree on T.
    """

    if member != "poses.npy":
        raise ValueError("catalog audit supports only the official poses.npy member")
    with zipfile.ZipFile(path) as archive:
        poses_size = archive.getinfo("poses.npy").file_size
        trans_size = archive.getinfo("trans.npy").file_size
        expression_size = archive.getinfo("expressions.npy").file_size
        rate_size = archive.getinfo("mocap_frame_rate.npy").file_size
    header_bytes = 128
    payloads = {
        "poses": (poses_size - header_bytes, 165 * 4),
        "trans": (trans_size - header_bytes, 3 * 8),
        "expressions": (expression_size - header_bytes, 200 * 4),
    }
    if rate_size != header_bytes + 4:
        raise ValueError("mocap_frame_rate.npy is not one float32 scalar")
    frame_counts = {}
    for name, (payload_size, bytes_per_frame) in payloads.items():
        if payload_size <= 0 or payload_size % bytes_per_frame:
            raise ValueError(f"{name}.npy does not match the official numeric layout")
        frame_counts[name] = payload_size // bytes_per_frame
    if len(set(frame_counts.values())) != 1:
        raise ValueError(f"BEAT2 time-axis members disagree: {frame_counts}")
    return (int(frame_counts["poses"]), 165)


def hamilton_counts(counts: Counter, total: int) -> dict[tuple[str, str], int]:
    population = sum(counts.values())
    quotas = {key: value * total / population for key, value in counts.items()}
    selected = {key: math.floor(value) for key, value in quotas.items()}
    remaining = total - sum(selected.values())
    order = sorted(
        quotas,
        key=lambda key: (-(quotas[key] - selected[key]), key),
    )
    for key in order[:remaining]:
        selected[key] += 1
    return selected


def nested_count(records: list[dict], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(record[field]) for record in records).items()))


def duration_accounting(records: list[dict]) -> dict:
    frames = [int(record["training_segment"]["frame_count"]) for record in records]
    return {
        "frame_count": sum(frames),
        "frame_count_min": min(frames),
        "frame_count_max": max(frames),
        "distinct_frame_count_count": len(set(frames)),
        "frame_coverage_hours_at_30hz": round(sum(frames) / 30 / 3600, 8),
        "sample_span_hours_at_30hz": round(
            sum(max(0, value - 1) for value in frames) / 30 / 3600, 8
        ),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    catalog_dir = args.catalog_dir.resolve()
    beat2_root = args.beat2_root.resolve()
    output = args.output.resolve()
    paths = {name: catalog_dir / filename for name, filename in FILE_NAMES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"catalog files are missing: {missing}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit report: {output}")

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    candidates = read_jsonl(paths["candidates"])
    rejected = read_jsonl(paths["rejected"])
    representative = read_jsonl(paths["representative100"])
    stress = read_jsonl(paths["stress100"])
    blockers: list[str] = []
    warnings: list[str] = []

    actual_hashes = {
        path.name: sha256_file(path)
        for name, path in paths.items()
        if name != "summary"
    }
    if actual_hashes != summary.get("output_sha256"):
        blockers.append("summary_output_sha256_mismatch")
    actual_counts = {
        "candidate_count": len(candidates),
        "rejected_count": len(rejected),
        "representative100_count": len(representative),
        "stress100_count": len(stress),
    }
    for field, value in actual_counts.items():
        if summary.get(field) != value:
            blockers.append(f"summary_{field}_mismatch")

    contract_errors = []
    for index, record in enumerate(candidates, 1):
        try:
            validate_expression_turn_candidate(record)
        except ValueError as error:
            contract_errors.append(
                {"file": paths["candidates"].name, "line": index, "error": str(error)}
            )
    if contract_errors:
        blockers.append("candidate_contract_validation_failed")

    candidate_by_clip = {str(record.get("clip_id")): record for record in candidates}
    if len(candidate_by_clip) != len(candidates):
        blockers.append("candidate_clip_ids_missing_or_duplicate")

    selection_reports = {}
    selection_errors = []
    selection_bindings = {
        "representative100": representative,
        "stress100": stress,
    }
    for kind, records in selection_bindings.items():
        manifest_path = paths[kind]
        binding = {
            "retarget_input_manifest_sha256": actual_hashes[manifest_path.name],
            "expression_turn_contract_sha256": summary[
                "expression_turn_contract_sha256"
            ],
            "source_inventory_manifest_sha256": summary["input_sha256"],
            "split_assignment_manifest_sha256": summary[
                "split_assignment_sha256"
            ],
            "selection_kind": kind,
            "require_selection_record": True,
        }
        expected_status = f"selected_{kind.removesuffix('100')}_pending_retarget_qc"
        ranks = []
        source_ids = []
        speaker_ids = []
        base_binding_mismatches = 0
        for index, record in enumerate(records, 1):
            try:
                validate_expression_turn_candidate(record, catalog_binding=binding)
            except ValueError as error:
                selection_errors.append(
                    {"file": manifest_path.name, "line": index, "error": str(error)}
                )
            ranks.append(record.get("expression_turn_selection_rank"))
            source_ids.append(str(record.get("source_clip_id")))
            speaker_ids.append(str(record.get("speaker_key")))
            if record.get("expression_turn_selection_status") != expected_status:
                selection_errors.append(
                    {
                        "file": manifest_path.name,
                        "line": index,
                        "error": "selection status does not match kind",
                    }
                )
            base = candidate_by_clip.get(str(record.get("clip_id")))
            if base is None:
                base_binding_mismatches += 1
            else:
                projected = {
                    key: value
                    for key, value in record.items()
                    if not key.startswith("expression_turn_selection_")
                }
                if projected != base:
                    base_binding_mismatches += 1
        if ranks != list(range(1, len(records) + 1)):
            selection_errors.append(
                {"file": manifest_path.name, "error": "selection ranks are not 1..N"}
            )
        if base_binding_mismatches:
            selection_errors.append(
                {
                    "file": manifest_path.name,
                    "error": f"{base_binding_mismatches} rows differ from base candidates",
                }
            )
        selection_reports[kind] = {
            "record_count": len(records),
            "unique_clip_count": len({record["clip_id"] for record in records}),
            "source_count": len(set(source_ids)),
            "source_disjoint": len(set(source_ids)) == len(source_ids),
            "speaker_count": len(set(speaker_ids)),
            "rank_sequence_exact": ranks == list(range(1, len(records) + 1)),
            "base_candidate_binding_mismatches": base_binding_mismatches,
            "counts_by_split": nested_count(records, "fixed_split_assignment"),
            "counts_by_duration_band": nested_count(records, "duration_band"),
            "counts_by_event_count_band": nested_count(records, "event_count_band"),
            "manifest_sha256": actual_hashes[manifest_path.name],
        }
        if not selection_reports[kind]["source_disjoint"]:
            selection_errors.append(
                {"file": manifest_path.name, "error": "sources are not disjoint"}
            )
    if selection_errors:
        blockers.append("selection_contract_validation_failed")

    source_shapes = {}
    source_errors = []
    for record in candidates:
        relative = str(record.get("motion_relpath") or "")
        if relative in source_shapes:
            continue
        motion_path = beat2_root / relative
        if not motion_path.is_file():
            source_errors.append({"motion_relpath": relative, "error": "missing"})
            continue
        try:
            shape = npy_shape_from_npz(motion_path)
        except (OSError, ValueError, zipfile.BadZipFile, KeyError) as error:
            source_errors.append(
                {"motion_relpath": relative, "error": f"invalid poses.npy: {error}"}
            )
            continue
        if not shape or shape[0] < 1:
            source_errors.append(
                {"motion_relpath": relative, "error": f"invalid pose shape {shape}"}
            )
            continue
        source_shapes[relative] = shape

    bound_errors = []
    maximum_margin = None
    minimum_margin = None
    for index, record in enumerate(candidates, 1):
        shape = source_shapes.get(record["motion_relpath"])
        if shape is None:
            continue
        actual_frames = int(shape[0])
        intervals = {
            "training_segment": record["training_segment"],
            "core_interval": record["core_interval"],
            "context_plan.source_interval": record["context_plan"]["source_interval"],
            "context_plan.admissible_interval": record["context_plan"][
                "admissible_interval"
            ],
        }
        intervals.update(
            {
                f"context_plan.levels[{level_index}]": level
                for level_index, level in enumerate(record["context_plan"]["levels"])
            }
        )
        for name, interval in intervals.items():
            start = int(interval["start_frame"])
            end = int(interval["end_frame_exclusive"])
            if start < 0 or end > actual_frames or end <= start:
                bound_errors.append(
                    {
                        "line": index,
                        "clip_id": record["clip_id"],
                        "interval": name,
                        "start": start,
                        "end": end,
                        "actual_frames": actual_frames,
                    }
                )
        margin = actual_frames - int(
            record["context_plan"]["admissible_interval"]["end_frame_exclusive"]
        )
        minimum_margin = margin if minimum_margin is None else min(minimum_margin, margin)
        maximum_margin = margin if maximum_margin is None else max(maximum_margin, margin)
    if source_errors:
        blockers.append("source_motion_header_validation_failed")
    if bound_errors:
        blockers.append("candidate_interval_exceeds_actual_source_motion")

    fail_closed_errors = []
    legacy_key_counts = Counter()
    ambiguous_duration_key_counts = Counter()
    for file_name, records in (
        (paths["candidates"].name, candidates),
        (paths["representative100"].name, representative),
        (paths["stress100"].name, stress),
    ):
        for index, record in enumerate(records, 1):
            if record.get("semantic_supervision_masks") != EXPECTED_SEMANTIC_MASKS:
                fail_closed_errors.append(
                    {"file": file_name, "line": index, "field": "semantic_supervision_masks"}
                )
            for field, expected in FAIL_CLOSED_FIELDS.items():
                if record.get(field) is not expected:
                    fail_closed_errors.append(
                        {"file": file_name, "line": index, "field": field}
                    )
            if record.get("canonical_prompt") is not None or record.get(
                "canonical_action"
            ) is not None:
                fail_closed_errors.append(
                    {"file": file_name, "line": index, "field": "canonical_text"}
                )
            if record.get("training_admission_status") != (
                "pending_retarget_and_independent_video_review"
            ):
                fail_closed_errors.append(
                    {"file": file_name, "line": index, "field": "training_admission_status"}
                )
            for key in walk_keys(record):
                if key in FORBIDDEN_LEGACY_KEYS:
                    legacy_key_counts[key] += 1
                if key in AMBIGUOUS_DURATION_KEYS:
                    ambiguous_duration_key_counts[key] += 1
    for key in walk_keys(summary):
        if key in AMBIGUOUS_DURATION_KEYS:
            ambiguous_duration_key_counts[key] += 1
    if fail_closed_errors:
        blockers.append("candidate_training_gate_not_fail_closed")
    if legacy_key_counts:
        blockers.append("legacy_pilot_or_semantic_event_fields_present")
    if ambiguous_duration_key_counts:
        blockers.append("ambiguous_duration_accounting_fields_present")

    full_joint = Counter(
        (record["duration_band"], record["event_count_band"])
        for record in candidates
    )
    representative_joint = Counter(
        (record["duration_band"], record["event_count_band"])
        for record in representative
    )
    expected_joint = hamilton_counts(full_joint, len(representative))
    all_joint_keys = sorted(set(full_joint) | set(representative_joint))
    joint_rows = []
    total_variation = 0.0
    for key in all_joint_keys:
        full_fraction = full_joint[key] / len(candidates)
        sample_fraction = representative_joint[key] / len(representative)
        total_variation += abs(full_fraction - sample_fraction)
        joint_rows.append(
            {
                "duration_band": key[0],
                "event_count_band": key[1],
                "pool_count": full_joint[key],
                "pool_fraction": round(full_fraction, 8),
                "hamilton_target_count": expected_joint.get(key, 0),
                "representative_count": representative_joint[key],
                "representative_fraction": round(sample_fraction, 8),
            }
        )
    total_variation *= 0.5
    hamilton_exact = all(
        representative_joint[key] == expected_joint.get(key, 0)
        for key in set(expected_joint) | set(representative_joint)
    )
    if not hamilton_exact:
        blockers.append("representative100_joint_distribution_not_hamilton_proportional")

    full_duration = duration_accounting(candidates)
    representative_duration = duration_accounting(representative)
    stress_duration = duration_accounting(stress)
    expected_summary_durations = {
        "candidate_frame_coverage_hours_at_30hz": full_duration[
            "frame_coverage_hours_at_30hz"
        ],
        "candidate_sample_span_hours_at_30hz": full_duration[
            "sample_span_hours_at_30hz"
        ],
        "representative100_frame_coverage_hours_at_30hz": representative_duration[
            "frame_coverage_hours_at_30hz"
        ],
        "representative100_sample_span_hours_at_30hz": representative_duration[
            "sample_span_hours_at_30hz"
        ],
        "stress100_frame_coverage_hours_at_30hz": stress_duration[
            "frame_coverage_hours_at_30hz"
        ],
        "stress100_sample_span_hours_at_30hz": stress_duration[
            "sample_span_hours_at_30hz"
        ],
    }
    duration_mismatches = {
        field: {"summary": summary.get(field), "recomputed": expected}
        for field, expected in expected_summary_durations.items()
        if summary.get(field) != expected
    }
    if duration_mismatches:
        blockers.append("summary_duration_accounting_mismatch")

    rejection_counts = dict(sorted(Counter(row.get("reason") for row in rejected).items()))
    if rejection_counts != summary.get("rejection_counts_by_reason"):
        blockers.append("summary_rejection_reason_counts_mismatch")
    if any(row.get("accepted_for_training") is not False for row in rejected):
        blockers.append("rejection_record_training_gate_not_closed")

    contract_sha = hashlib.sha256(
        stable_json(summary["expression_turn_contract"]).encode("utf-8")
    ).hexdigest()
    if contract_sha != summary.get("expression_turn_contract_sha256"):
        blockers.append("summary_expression_turn_contract_sha256_mismatch")
    builder_path = PROJECT_ROOT / "tools/human_motion_collection/build_beat2_expression_turn_v8.py"
    if sha256_file(builder_path) != summary["expression_turn_contract"].get(
        "builder_script_sha256"
    ):
        blockers.append("builder_script_sha256_mismatch")

    cross_set_sources = sorted(
        {record["source_clip_id"] for record in representative}
        & {record["source_clip_id"] for record in stress}
    )
    if cross_set_sources:
        warnings.append(
            "stress100 and representative100 share sources; each set remains internally source-disjoint"
        )

    report = {
        "artifact_kind": "beat2_expression_turn_v8_independent_catalog_audit",
        "audit_contract_version": 1,
        "audit_status": "signed_off" if not blockers else "rejected",
        "catalog_dir": str(catalog_dir),
        "catalog_summary_sha256": sha256_file(paths["summary"]),
        "audit_tool_sha256": sha256_file(Path(__file__).resolve()),
        "scope": (
            "boundary_candidates_and_review_set_selection_only; does_not_grant_"
            "retarget_QC_blind_review_license_or_training_admission"
        ),
        "counts": actual_counts,
        "file_sha256": actual_hashes,
        "summary_hashes_match": actual_hashes == summary.get("output_sha256"),
        "contract_validation": {
            "candidate_valid_count": len(candidates) - len(contract_errors),
            "candidate_error_count": len(contract_errors),
            "candidate_errors": contract_errors[:20],
            "selection_error_count": len(selection_errors),
            "selection_errors": selection_errors[:20],
        },
        "source_motion_bounds": {
            "unique_declared_source_count": len(
                {record["motion_relpath"] for record in candidates}
            ),
            "validated_npz_pose_header_count": len(source_shapes),
            "source_header_error_count": len(source_errors),
            "source_header_errors": source_errors[:20],
            "interval_error_count": len(bound_errors),
            "interval_errors": bound_errors[:20],
            "minimum_admissible_end_margin_frames": minimum_margin,
            "maximum_admissible_end_margin_frames": maximum_margin,
        },
        "training_gate": {
            "fail_closed_error_count": len(fail_closed_errors),
            "fail_closed_errors": fail_closed_errors[:20],
            "legacy_key_counts": dict(sorted(legacy_key_counts.items())),
            "ambiguous_duration_key_counts": dict(
                sorted(ambiguous_duration_key_counts.items())
            ),
            "accepted_for_training_count": sum(
                record.get("accepted_for_training") is True for record in candidates
            ),
        },
        "selection_sets": selection_reports,
        "selection_cross_set_source_overlap_count": len(cross_set_sources),
        "selection_cross_set_source_overlap": cross_set_sources,
        "representative100_joint_distribution": {
            "allocation_policy": "hamilton_largest_remainder_from_full_pool",
            "hamilton_target_exact": hamilton_exact,
            "total_variation_distance": round(total_variation, 8),
            "cells": joint_rows,
        },
        "duration_accounting": {
            "candidate": full_duration,
            "representative100": representative_duration,
            "stress100": stress_duration,
            "summary_mismatches": duration_mismatches,
            "planner_target": "sample_span_sum(N-1)/30_only",
        },
        "rejections": {
            "counts_by_reason": rejection_counts,
            "summary_counts_match": rejection_counts
            == summary.get("rejection_counts_by_reason"),
        },
        "blockers": blockers,
        "warnings": warnings,
        "signoff": {
            "catalog_boundary_contract": not blockers,
            "retarget_allowed_for_selected_review_sets": not blockers,
            "formal_training_allowed_by_this_audit": False,
            "license_release_allowed_by_this_audit": False,
        },
    }
    report["report_sha256"] = hashlib.sha256(
        stable_json(report).encode("utf-8")
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
