#!/usr/bin/env python3
"""Validate an independent agent motion-review bundle and conflict queue."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from tools.human_motion_review.adjudicate_training_dataset import (
    MOTION_ADMISSION_REVIEW_GATES,
    REQUIRED_REVIEW_GATES,
    _validate_blind_affect_review,
)


ALLOWED_STATUSES = {"agent_reviewed", "needs_human", "rejected"}
REQUIRED_GATES = REQUIRED_REVIEW_GATES
LEGACY_OBSERVABILITY_GATE = "observable_in_15d"
ALLOWED_ASSIGNMENTS = {None, "train", "val", "test", "ood_action_test"}


def review_gate_template():
    """Return a fresh fail-closed gate template for an ULA V2 18D review."""

    return {gate: False for gate in sorted(REQUIRED_GATES)}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_evidence(sample, errors, check_files):
    evidence = sample.get("evidence")
    if not isinstance(evidence, dict):
        errors.append(f"{sample.get('sample_id')}: evidence must be an object")
        return
    for key in ("preview", "raw_label", "quality_report"):
        item = evidence.get(key)
        if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
            errors.append(f"{sample.get('sample_id')}: incomplete evidence.{key}")
            continue
        if check_files:
            path = Path(item["path"])
            if not path.is_file():
                errors.append(f"{sample.get('sample_id')}: missing evidence file {path}")
            elif _sha256(path) != item["sha256"]:
                errors.append(f"{sample.get('sample_id')}: sha256 mismatch for {path}")


def validate_bundle(review, conflicts, *, check_files=False):
    errors = []
    if review.get("schema_version") != 1 or conflicts.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if review.get("review_id") != conflicts.get("review_id"):
        errors.append("review and conflict queue review_id differ")
    reviewer = review.get("reviewer") or {}
    if reviewer.get("kind") != "agent" or reviewer.get("independent_of_annotation_logic") is not True:
        errors.append("reviewer must be an independent agent")

    samples = review.get("samples")
    if not isinstance(samples, list) or not samples:
        errors.append("samples must be a non-empty list")
        samples = []
    sample_ids = [sample.get("sample_id") for sample in samples]
    if None in sample_ids or len(sample_ids) != len(set(sample_ids)):
        errors.append("sample_id values must be present and unique")

    conflict_items = conflicts.get("items")
    if not isinstance(conflict_items, list):
        errors.append("conflict items must be a list")
        conflict_items = []
    conflict_ids = [item.get("conflict_id") for item in conflict_items]
    if None in conflict_ids or len(conflict_ids) != len(set(conflict_ids)):
        errors.append("conflict_id values must be present and unique")
    conflict_by_id = {item.get("conflict_id"): item for item in conflict_items}

    for sample in samples:
        sample_id = sample.get("sample_id")
        status = sample.get("status")
        if status not in ALLOWED_STATUSES:
            errors.append(f"{sample_id}: invalid status {status!r}")
        scores = sample.get("scores") or {}
        for key in ("recognizability", "text_consistency", "observable_dof_coverage"):
            value = scores.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
                errors.append(f"{sample_id}: scores.{key} must be in [0, 1]")

        gates = sample.get("gates")
        if isinstance(gates, dict) and LEGACY_OBSERVABILITY_GATE in gates:
            errors.append(
                f"{sample_id}: legacy {LEGACY_OBSERVABILITY_GATE} cannot validate an 18D "
                "review; regenerate the bundle with observable_in_18d"
            )
            gates = {}
        elif not isinstance(gates, dict) or set(gates) != REQUIRED_GATES:
            errors.append(f"{sample_id}: gates must contain exactly {sorted(REQUIRED_GATES)}")
            gates = {}
        elif not all(isinstance(value, bool) for value in gates.values()):
            errors.append(f"{sample_id}: every gate must be boolean")

        accepted = sample.get("training_acceptance")
        if not isinstance(accepted, bool):
            errors.append(f"{sample_id}: training_acceptance must be boolean")
        if gates.get("affect_observable_in_18d") is True:
            try:
                _validate_blind_affect_review(sample)
            except ValueError as error:
                errors.append(str(error))
        if accepted and (
            status != "agent_reviewed"
            or not all(gates.get(gate) is True for gate in MOTION_ADMISSION_REVIEW_GATES)
        ):
            errors.append(
                f"{sample_id}: accepted sample must be agent_reviewed and pass every "
                "motion-admission gate"
            )
        if status in {"needs_human", "rejected"} and accepted:
            errors.append(f"{sample_id}: unresolved or rejected sample cannot be accepted")

        split = sample.get("split") or {}
        assignment = split.get("assignment")
        if assignment not in ALLOWED_ASSIGNMENTS:
            errors.append(f"{sample_id}: invalid split assignment {assignment!r}")
        if not split.get("action_key") or not split.get("source_group_key"):
            errors.append(f"{sample_id}: split action/source group keys are required")
        if accepted and assignment != "train":
            errors.append(f"{sample_id}: this review bundle only accepts samples into train")
        if accepted and split.get("subject_key") is None and split.get("subject_policy") != "train_only_unknown":
            errors.append(f"{sample_id}: unknown subject is allowed only under train_only_unknown")
        if split.get("subject_key") is None and assignment in {"val", "test", "ood_action_test"}:
            errors.append(f"{sample_id}: unknown subject cannot enter evaluation")
        if split.get("subject_key") is None and split.get("eval_eligible") is not False:
            errors.append(f"{sample_id}: unknown subject must set eval_eligible=false")

        listed_conflicts = sample.get("conflict_ids") or []
        for conflict_id in listed_conflicts:
            item = conflict_by_id.get(conflict_id)
            if item is None:
                errors.append(f"{sample_id}: unknown conflict {conflict_id}")
            elif item.get("scope") == "sample" and item.get("sample_id") != sample_id:
                errors.append(f"{sample_id}: conflict {conflict_id} belongs to another sample")
        _validate_evidence(sample, errors, check_files)

    source_assignments = defaultdict(set)
    subject_assignments = defaultdict(set)
    action_assignments = defaultdict(set)
    for sample in samples:
        split = sample.get("split") or {}
        assignment = split.get("assignment")
        if assignment is None:
            continue
        source_assignments[split.get("source_group_key")].add(assignment)
        action_assignments[split.get("action_key")].add(assignment)
        if split.get("subject_key") is not None:
            subject_assignments[split["subject_key"]].add(assignment)

    for source_group, assignments in source_assignments.items():
        if len(assignments) > 1:
            errors.append(f"source group {source_group!r} crosses splits: {sorted(assignments)}")
    for subject, assignments in subject_assignments.items():
        if len(assignments) > 1:
            errors.append(f"subject {subject!r} crosses splits: {sorted(assignments)}")
    for action, assignments in action_assignments.items():
        if "ood_action_test" in assignments and assignments != {"ood_action_test"}:
            errors.append(f"OOD action {action!r} also appears outside ood_action_test")

    for item in conflict_items:
        conflict_id = item.get("conflict_id")
        if item.get("status") not in ALLOWED_STATUSES:
            errors.append(f"{conflict_id}: invalid status {item.get('status')!r}")
        if item.get("severity") not in {"warning", "blocking"}:
            errors.append(f"{conflict_id}: severity must be warning or blocking")
        if item.get("scope") == "sample" and item.get("sample_id") not in set(sample_ids):
            errors.append(f"{conflict_id}: unknown sample_id {item.get('sample_id')!r}")
        affected = item.get("affected_sample_ids") or []
        if any(sample_id not in set(sample_ids) for sample_id in affected):
            errors.append(f"{conflict_id}: affected_sample_ids contains an unknown sample")

    expected = Counter(sample.get("status") for sample in samples)
    if review.get("summary", {}).get("status_counts") != dict(sorted(expected.items())):
        errors.append("summary.status_counts does not match samples")
    accepted_count = sum(sample.get("training_acceptance") is True for sample in samples)
    if review.get("summary", {}).get("training_accepted") != accepted_count:
        errors.append("summary.training_accepted does not match samples")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--conflicts", type=Path, required=True)
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args()
    review = json.loads(args.review.read_text(encoding="utf-8"))
    conflicts = json.loads(args.conflicts.read_text(encoding="utf-8"))
    errors = validate_bundle(review, conflicts, check_files=args.check_files)
    print(json.dumps({"valid": not errors, "error_count": len(errors), "errors": errors}, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
