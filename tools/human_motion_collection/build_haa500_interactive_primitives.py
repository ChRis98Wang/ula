#!/usr/bin/env python3
"""Build a conservative HAA500 supplement for robot interaction gestures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


DEFAULT_ROOT = Path("/home/gez/nas/cloud/gez/human_motion")
DEFAULT_OUTPUT = Path("deliverables/interactive_human_motion_v1/catalog/haa500")
SCHEMA_VERSION = "1.0.0"

COMMUNICATIVE_PRIMITIVES = {
    "applauding": {
        "communicative_intent": "positive_feedback_applause",
        "canonical_prompt_en": (
            "Repeatedly bring both forearms together in front of the torso as applause."
        ),
        "context_dependency": "none",
    },
    "bowing_fullbody": {
        "communicative_intent": "respectful_deep_bow",
        "canonical_prompt_en": (
            "Lean the torso forward in a deliberate deep bow, then return upright."
        ),
        "context_dependency": "none",
    },
    "bowing_waist": {
        "communicative_intent": "respectful_bow",
        "canonical_prompt_en": (
            "Bend forward at the waist in a respectful bow, then return upright."
        ),
        "context_dependency": "none",
    },
    "hailing_taxi": {
        "communicative_intent": "raise_hand_to_get_attention",
        "canonical_prompt_en": (
            "Raise one arm clearly to get another person's attention, then lower it."
        ),
        "context_dependency": "observer_required",
    },
    "salute": {
        "communicative_intent": "formal_greeting_salute",
        "canonical_prompt_en": (
            "Raise the right hand toward the forehead in a formal salute, then lower it."
        ),
        "context_dependency": "none",
    },
}

PARTNER_OFFERS = {
    "fist_bump": {
        "communicative_intent": "offer_fist_bump",
        "canonical_prompt_en": (
            "Extend one forearm forward and pause to offer a fist-bump interaction."
        ),
        "context_dependency": "partner_target_required",
    },
    "high_five": {
        "communicative_intent": "offer_high_five",
        "canonical_prompt_en": (
            "Raise one forearm and pause with the hand presented for a high-five."
        ),
        "context_dependency": "partner_target_required",
    },
    "hugging_human": {
        "communicative_intent": "offer_hug",
        "canonical_prompt_en": (
            "Open both arms toward a partner and hold the invitation to hug."
        ),
        "context_dependency": "partner_geometry_required",
    },
}

KNOWN_HARD_EXCLUSIONS = {
    "arm_wave": "dataset action is a dance-style arm wave, not a greeting wave",
    "blowing_kisses": "meaning depends on face, mouth, and finger detail",
    "hand_in_hand": "meaning depends on a second person's trajectory and locomotion",
    "kiss": "meaning depends on face, mouth, and a second person",
    "rock_paper_scissors": "meaning depends on finger articulation",
    "shaking_head": "current retargets are roll-dominant and do not verify refusal yaw",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inventory(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "clip_id" not in rows[0] or "action" not in rows[0]:
        raise ValueError("inventory must contain clip_id and action")
    clip_ids = [row["clip_id"] for row in rows]
    if len(clip_ids) != len(set(clip_ids)):
        raise ValueError("inventory clip_id values must be unique")
    return sorted(rows, key=lambda row: row["clip_id"])


def load_passed(path: Path) -> dict[str, dict]:
    records = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            clip_id = record.get("clip_id")
            if not isinstance(clip_id, str) or not clip_id:
                raise ValueError(f"invalid clip_id at {path}:{line_number}")
            if clip_id in records:
                raise ValueError(f"duplicate passed clip_id: {clip_id}")
            if record.get("retarget_status") != "retarget_qc_passed":
                raise ValueError(f"passed manifest contains a non-passing record: {clip_id}")
            records[clip_id] = record
    return records


def source_flags(row: dict) -> list[str]:
    return sorted({value for value in (row.get("source_text_flags") or "").split("|") if value})


def _candidate_record(row: dict, physical: dict, spec: dict, *, candidate_state: str) -> dict:
    safe_csv = physical["output_18d"]["safe_csv"]
    return {
        "schema_version": SCHEMA_VERSION,
        "clip_id": row["clip_id"],
        "source_action": row["action"],
        "communicative_intent": spec["communicative_intent"],
        "canonical_prompt_en": spec["canonical_prompt_en"],
        "candidate_state": candidate_state,
        "context_dependency": spec["context_dependency"],
        "robot_contract": "ula_v2_18d_head_v1",
        "trajectory": {
            "path": safe_csv["path"],
            "sha256": safe_csv["sha256"],
            "frames": physical["output_18d"]["frames"],
            "fps": physical["output_18d"]["fps"],
            "duration_sec": physical["output_18d"]["duration_sec"],
        },
        "source": {
            "path": physical["source"]["path"],
            "sha256": physical["source"]["sha256"],
            "dataset_revision": row.get("dataset_revision"),
        },
        "physical_qc": {
            "passed": True,
            "quality_json": physical["output_18d"]["quality_json"],
            "quality_sha256": physical["output_18d"]["quality_sha256"],
        },
        "semantic_review": {
            "source_text_flags": [],
            "manual_video_review_required": True,
            "accepted_for_training": False,
            "rule": "physical QC and category filtering do not grant semantic admission",
        },
    }


def _jsonl_bytes(records: list[dict]) -> bytes:
    if not records:
        return b""
    return (
        "\n".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for record in records
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def build_subset(inventory_path: Path, passed_path: Path, output_dir: Path) -> dict:
    inventory_path = inventory_path.resolve()
    passed_path = passed_path.resolve()
    output_dir = output_dir.resolve()
    inventory = load_inventory(inventory_path)
    passed = load_passed(passed_path)
    inventory_ids = {row["clip_id"] for row in inventory}
    unknown_passed = sorted(set(passed) - inventory_ids)
    if unknown_passed:
        raise ValueError(f"passed manifest contains clips outside inventory: {unknown_passed[:3]}")

    primitives = []
    partner_offers = []
    excluded = []
    exclusion_reasons = Counter()
    selected_ids = set()
    for row in inventory:
        clip_id = row["clip_id"]
        action = row["action"]
        flags = source_flags(row)
        physical = passed.get(clip_id)
        spec = COMMUNICATIVE_PRIMITIVES.get(action)
        partner_spec = PARTNER_OFFERS.get(action)
        reasons = []
        if spec is None and partner_spec is None:
            reasons.append("not_in_interactive_ontology")
            if action in KNOWN_HARD_EXCLUSIONS:
                reasons.append(f"known_hard_exclusion:{KNOWN_HARD_EXCLUSIONS[action]}")
        if physical is None:
            reasons.append("retarget_qc_not_passed")
        reasons.extend(f"source_text_risk:{flag}" for flag in flags)

        if not reasons and spec is not None:
            primitives.append(
                _candidate_record(
                    row,
                    physical,
                    spec,
                    candidate_state="communicative_primitive_pending_video_review",
                )
            )
            selected_ids.add(clip_id)
        elif not reasons and partner_spec is not None:
            partner_offers.append(
                _candidate_record(
                    row,
                    physical,
                    partner_spec,
                    candidate_state="partner_conditioned_probe_pending_review",
                )
            )
            selected_ids.add(clip_id)
        else:
            for reason in reasons:
                exclusion_reasons[reason] += 1
            excluded.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "clip_id": clip_id,
                    "source_action": action,
                    "retarget_qc_passed": physical is not None,
                    "source_text_flags": flags,
                    "exclusion_reasons": sorted(reasons),
                    "accepted_for_training": False,
                }
            )

    if len(primitives) + len(partner_offers) + len(excluded) != len(inventory):
        raise AssertionError("interactive partition does not cover the inventory exactly once")
    if selected_ids & {record["clip_id"] for record in excluded}:
        raise AssertionError("interactive candidate also appears in excluded partition")

    paths = {
        "communicative_primitives": output_dir / "communicative_primitives_pending_review.jsonl",
        "partner_offers": output_dir / "partner_offers_needs_review.jsonl",
        "train_ready": output_dir / "train_ready.jsonl",
        "excluded": output_dir / "excluded.jsonl",
    }
    payloads = {
        "communicative_primitives": primitives,
        "partner_offers": partner_offers,
        "train_ready": [],
        "excluded": excluded,
    }
    for name, path in paths.items():
        _atomic_write(path, _jsonl_bytes(payloads[name]))

    def scale(records: list[dict]) -> dict:
        return {
            "clips": len(records),
            "frames": sum(record["trajectory"]["frames"] for record in records),
            "duration_sec": sum(record["trajectory"]["duration_sec"] for record in records),
        }

    summary = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "conservative HAA500 supplement for robot-observable interaction gestures",
        "policy": {
            "haa500_role": "single_person_interaction_primitive_supplement_only",
            "high_dynamic_actions_included": False,
            "fine_fingers_face_and_objects_used": False,
            "partner_offers_require_future_partner_conditioning": True,
            "retarget_qc_grants_training_admission": False,
        },
        "inputs": {
            "inventory": {"path": str(inventory_path), "sha256": sha256_file(inventory_path)},
            "physical_pass_manifest": {"path": str(passed_path), "sha256": sha256_file(passed_path)},
        },
        "counts": {
            "inventory": len(inventory),
            "physical_pass_pool": len(passed),
            "communicative_primitives_pending_review": len(primitives),
            "partner_offers_needs_review": len(partner_offers),
            "train_ready": 0,
            "excluded": len(excluded),
        },
        "scale": {
            "communicative_primitives_pending_review": scale(primitives),
            "partner_offers_needs_review": scale(partner_offers),
        },
        "counts_by_communicative_intent": dict(
            sorted(Counter(record["communicative_intent"] for record in primitives).items())
        ),
        "counts_by_partner_offer": dict(
            sorted(Counter(record["communicative_intent"] for record in partner_offers).items())
        ),
        "exclusion_reason_counts": dict(sorted(exclusion_reasons.items())),
        "known_hard_exclusions": KNOWN_HARD_EXCLUSIONS,
        "outputs": {},
    }
    for name, path in paths.items():
        summary["outputs"][name] = {
            "path": str(path),
            "records": len(payloads[name]),
            "sha256": sha256_file(path),
        }
    summary_path = output_dir / "summary.json"
    _atomic_write(
        summary_path,
        (json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_ROOT / "catalog/motionx_haa500_full_inventory_v1.csv",
    )
    parser.add_argument(
        "--physical-pass-manifest",
        type=Path,
        default=(
            DEFAULT_ROOT
            / "catalog/ula_v2_18d_head_v1_full_retarget_qa/full_retarget_passed.jsonl"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = build_subset(args.inventory, args.physical_pass_manifest, args.output_dir)
    print(json.dumps({"counts": summary["counts"], "output_dir": str(args.output_dir.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
