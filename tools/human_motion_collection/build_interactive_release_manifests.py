#!/usr/bin/env python3
"""Build admission manifests for the low-dynamic interaction dataset.

The release deliberately keeps physical retargeting quality separate from
semantic admission.  BEAT2 transcripts are speech context, not action labels,
and HAA500 category names still require a blind video review before training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG_ROOT = PROJECT_ROOT / "deliverables/interactive_human_motion_v1/catalog"
DEFAULT_BEAT2_ROOT = Path(
    "/home/gez/nas/cloud/gez/human_motion/raw/BEAT2/beat_chinese_v2.0.0"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "deliverables/interactive_human_motion_v1/manifests"
)
SCHEMA_VERSION = "1.0.0"
NON_BLOCKING_BEAT2_WARNINGS = {
    "motion_audio_duration_mismatch_gt_0_3s",
    "textgrid_transcript_mismatch",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--beat2-inventory",
        type=Path,
        default=CATALOG_ROOT / "beat2_interaction_full_inventory_v1.jsonl",
    )
    parser.add_argument("--beat2-root", type=Path, default=DEFAULT_BEAT2_ROOT)
    parser.add_argument(
        "--haa-primitives",
        type=Path,
        default=CATALOG_ROOT / "haa500/communicative_primitives_pending_review.jsonl",
    )
    parser.add_argument(
        "--haa-partner-offers",
        type=Path,
        default=CATALOG_ROOT / "haa500/partner_offers_needs_review.jsonl",
    )
    parser.add_argument(
        "--haa-excluded",
        type=Path,
        default=CATALOG_ROOT / "haa500/excluded.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            records.append(record)
    return records


def stable_json(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_optional(root: Path, relative: str | None) -> str | None:
    if not relative:
        return None
    return str((root / relative).resolve())


def normalize_beat2(record: dict, beat2_root: Path, source_manifest: Path) -> dict:
    clip_id = record["clip_id"]
    issues = sorted(set(record.get("issues", [])))
    source_warnings = [
        issue for issue in issues if issue in NON_BLOCKING_BEAT2_WARNINGS
    ]
    blocking_issues = [
        issue for issue in issues if issue not in NON_BLOCKING_BEAT2_WARNINGS
    ]
    window = record["window"]
    selection_status = str(window.get("selection_status", ""))
    low_dynamic = selection_status.startswith("selected_nonstatic_low_dynamic")
    if blocking_issues or not low_dynamic:
        decision = "rejected"
        decision_reasons = [f"source_issue:{issue}" for issue in blocking_issues]
        if not low_dynamic:
            decision_reasons.append("not_selected_nonstatic_low_dynamic")
        review_state = "rejected_for_current_text_conditioned_training"
    else:
        decision = "pending_review"
        decision_reasons = ["requires_blind_motion_text_review"]
        review_state = "pending_interaction_video_review"

    return {
        "accepted_for_training": False,
        "clip_id": clip_id,
        "conditioning_text": record.get("window_transcript_context", ""),
        "conditioning_text_status": "speech_context_only_not_approved_action_prompt",
        "dataset": "BEAT2",
        "decision": decision,
        "decision_reasons": decision_reasons,
        "decision_scope": "ula_v2_18d_text_conditioned_interaction_training_v1",
        "fps": record["fps"],
        "interaction_label": "co_speech_conversational_gesture",
        "motion_path": _resolved_optional(beat2_root, record.get("motion_relpath")),
        "audio_path": _resolved_optional(beat2_root, record.get("audio_relpath")),
        "textgrid_path": _resolved_optional(beat2_root, record.get("textgrid_relpath")),
        "transcript_path": _resolved_optional(beat2_root, record.get("transcript_relpath")),
        "official_split": record.get("official_split"),
        "record_id": f"beat2:{clip_id}",
        "review_state": review_state,
        "schema_version": SCHEMA_VERSION,
        "source_kind": "continuous_co_speech_conversational_gesture",
        "source_manifest": str(source_manifest.resolve()),
        "source_warnings": source_warnings,
        "speaker_key": record.get("speaker_key"),
        "window": window,
    }


def normalize_haa_pending(record: dict, source_manifest: Path, kind: str) -> dict:
    clip_id = record["clip_id"]
    return {
        "accepted_for_training": False,
        "clip_id": clip_id,
        "conditioning_text": record["canonical_prompt_en"],
        "conditioning_text_status": "canonical_prompt_pending_blind_video_review",
        "context_dependency": record.get("context_dependency"),
        "dataset": "HAA500",
        "decision": "pending_review",
        "decision_reasons": ["requires_blind_video_semantic_review"],
        "decision_scope": "ula_v2_18d_text_conditioned_interaction_training_v1",
        "fps": record["trajectory"]["fps"],
        "interaction_label": record["communicative_intent"],
        "record_id": f"haa500:{clip_id}",
        "review_state": record["candidate_state"],
        "schema_version": SCHEMA_VERSION,
        "source_kind": kind,
        "source_manifest": str(source_manifest.resolve()),
        "trajectory": record["trajectory"],
    }


def normalize_haa_rejected(record: dict, source_manifest: Path) -> dict:
    clip_id = record["clip_id"]
    return {
        "accepted_for_training": False,
        "clip_id": clip_id,
        "dataset": "HAA500",
        "decision": "rejected",
        "decision_reasons": record["exclusion_reasons"],
        "decision_scope": "ula_v2_18d_text_conditioned_interaction_training_v1",
        "interaction_label": None,
        "record_id": f"haa500:{clip_id}",
        "review_state": "rejected_by_interactive_only_filter",
        "schema_version": SCHEMA_VERSION,
        "source_action": record.get("source_action"),
        "source_kind": "non_interaction_or_unusable_haa500_source",
        "source_manifest": str(source_manifest.resolve()),
    }


def _duration(record: dict) -> float:
    if "window" in record:
        return float(record["window"]["duration_sec"])
    if "trajectory" in record:
        return float(record["trajectory"]["duration_sec"])
    return 0.0


def build_release(
    *,
    beat2_inventory: Path,
    beat2_root: Path,
    haa_primitives: Path,
    haa_partner_offers: Path,
    haa_excluded: Path,
    output_dir: Path,
) -> dict:
    for path in (beat2_inventory, haa_primitives, haa_partner_offers, haa_excluded):
        if not path.is_file():
            raise FileNotFoundError(path)

    records = [
        normalize_beat2(record, beat2_root, beat2_inventory)
        for record in read_jsonl(beat2_inventory)
    ]
    records.extend(
        normalize_haa_pending(record, haa_primitives, "single_person_interaction_primitive")
        for record in read_jsonl(haa_primitives)
    )
    records.extend(
        normalize_haa_pending(
            record, haa_partner_offers, "partner_conditioned_offer_probe"
        )
        for record in read_jsonl(haa_partner_offers)
    )
    records.extend(
        normalize_haa_rejected(record, haa_excluded)
        for record in read_jsonl(haa_excluded)
    )

    record_ids = [record["record_id"] for record in records]
    duplicates = sorted(key for key, count in Counter(record_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate release record IDs: {duplicates[:10]}")
    if any(record["accepted_for_training"] for record in records):
        raise ValueError("Unreviewed records cannot be accepted for training")

    groups = {"train_ready": [], "pending_review": [], "rejected": []}
    for record in records:
        destination = "train_ready" if record["accepted_for_training"] else record["decision"]
        groups[destination].append(record)
    for group in groups.values():
        group.sort(key=lambda record: record["record_id"])

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {}
    for name, group in groups.items():
        path = output_dir / f"{name}.jsonl"
        payload = "".join(f"{stable_json(record)}\n" for record in group)
        atomic_write(path, payload)
        output_paths[name] = {
            "path": str(path.resolve()),
            "records": len(group),
            "duration_sec_with_known_window": sum(_duration(record) for record in group),
            "sha256": sha256(path),
        }

    counts_by_dataset_and_decision = Counter(
        f"{record['dataset']}:{record['decision']}" for record in records
    )
    rejection_reason_counts = Counter(
        reason
        for record in groups["rejected"]
        for reason in record["decision_reasons"]
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "low_dynamic_robot_observable_interaction_motion_only",
        "high_dynamic_actions_included": False,
        "face_fingers_used": False,
        "head_neck_used": True,
        "training_admission_policy": (
            "deny until blind motion/text semantic review; physical QC alone is insufficient"
        ),
        "beat2_scope": (
            "continuous co-speech conversational gesture, not dyadic physical contact"
        ),
        "haa500_scope": "single-person interaction primitives and partner-offer probes only",
        "counts_by_dataset_and_decision": dict(sorted(counts_by_dataset_and_decision.items())),
        "rejection_reason_counts": dict(sorted(rejection_reason_counts.items())),
        "outputs": output_paths,
    }
    summary_path = output_dir / "release_summary.json"
    atomic_write(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main(argv=None):
    args = parse_args(argv)
    summary = build_release(
        beat2_inventory=args.beat2_inventory,
        beat2_root=args.beat2_root,
        haa_primitives=args.haa_primitives,
        haa_partner_offers=args.haa_partner_offers,
        haa_excluded=args.haa_excluded,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
