#!/usr/bin/env python3
"""Build fail-closed intent review and v9 semantic overlay manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from upper_body_skeleton.robot_observable_intents import (
    DEFAULT_ONTOLOGY_PATH,
    build_observable_intent_one_hot,
    intent_definition,
    load_observable_intent_ontology,
    observable_intent_ids,
    ontology_sha256,
    validate_observable_intent_annotation,
)


PROTOCOL_VERSION = "robot_observable_intent_blind_video_v1"
OUTPUT_SCHEMA_VERSION = "1.0.0"
FORBIDDEN_REVIEW_FIELDS = frozenset(
    {
        "source_action_name",
        "source_filename",
        "source_transcript",
        "official_category",
        "candidate_behavior_id",
    }
)
VERIFIED_MOTION_PROMPT_PROVENANCES = frozenset(
    {
        "independent_blind_action_observable_description_v1",
        "independent_blind_dyadic_interaction_prompt_v1",
    }
)
OBSERVABLE_RESULT = "observable"
NON_OBSERVABLE_RESULTS = frozenset({"not_observable", "not_in_ontology"})
PENDING_RESULTS = frozenset({"ambiguous"})
QUALITY_FAILURE_RESULT = "quality_failure"
ALL_RESULTS = frozenset(
    {OBSERVABLE_RESULT, *NON_OBSERVABLE_RESULTS, *PENDING_RESULTS, QUALITY_FAILURE_RESULT}
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _record_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL record must be an object")
            records.append(value)
    return records


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _sample_id(record: Mapping[str, Any]) -> str:
    value = record.get("sample_id") or record.get("clip_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("source record lacks sample_id/clip_id")
    return value.strip()


def _expected_review_video_sha256(record: Mapping[str, Any]) -> str:
    review = record.get("expression_turn_review_record")
    if not isinstance(review, Mapping):
        raise ValueError(f"{_sample_id(record)}: expression_turn_review_record is missing")
    action_review = review.get("action_semantic_review")
    if not isinstance(action_review, Mapping):
        raise ValueError(f"{_sample_id(record)}: action_semantic_review is missing")
    value = action_review.get("anonymous_video_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{_sample_id(record)}: anonymous video SHA256 is missing")
    return value


def build_review_queue(
    source_records: list[dict[str, Any]],
    *,
    source_manifest: Path,
    video_dir: Path,
    ontology_path: Path,
) -> list[dict[str, Any]]:
    ontology = load_observable_intent_ontology(ontology_path)
    ontology_digest = ontology_sha256(ontology_path)
    source_manifest_digest = _file_sha256(source_manifest)
    ids_seen: set[str] = set()
    queue: list[dict[str, Any]] = []
    for source_record in sorted(source_records, key=_sample_id):
        sample_id = _sample_id(source_record)
        if sample_id in ids_seen:
            raise ValueError(f"duplicate source sample_id: {sample_id}")
        ids_seen.add(sample_id)
        video_path = (video_dir / f"{sample_id}.mp4").resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"{sample_id}: missing anonymous video {video_path}")
        video_digest = _file_sha256(video_path)
        if video_digest != _expected_review_video_sha256(source_record):
            raise ValueError(f"{sample_id}: anonymous video does not match reviewed source")
        queue.append(
            {
                "schema_version": OUTPUT_SCHEMA_VERSION,
                "artifact_kind": "robot_observable_intent_blind_review_item",
                "protocol_version": PROTOCOL_VERSION,
                "sample_id": sample_id,
                "anonymous_video_path": str(video_path),
                "anonymous_video_sha256": video_digest,
                "audio_available": False,
                "label_metadata_exposed": False,
                "intent_ontology_id": ontology["ontology_id"],
                "intent_ontology_path": str(ontology_path.resolve()),
                "intent_ontology_sha256": ontology_digest,
                "observable_intent_id": None,
                "intent_review_status": "pending_review",
                "intent_supervision_mask": False,
                "intent_conditioning_mask": False,
                "pragmatic_role": None,
                "pragmatic_role_supervision_mask": False,
                "review_instruction": (
                    "Watch the anonymous robot video without source name, transcript, audio, "
                    "or prior prompt. Select an ontology intent only when its visual signature "
                    "and hard-negative distinction are both visible."
                ),
                "decision_template": {
                    "intent_ontology_sha256": ontology_digest,
                    "result": None,
                    "observable_intent_id": None,
                    "confidence": None,
                    "hard_negative_checked": None,
                    "hard_negative_notes": None,
                    "reviewer_id": None,
                    "review_id": None,
                    "notes": None,
                },
                "sealed_source_provenance": {
                    "source_manifest_sha256": source_manifest_digest,
                    "source_record_sha256": _record_sha256(source_record),
                    "not_reviewer_visible": True,
                },
            }
        )
    return queue


def _validate_decision(
    decision: Mapping[str, Any],
    queue_item: Mapping[str, Any],
    ontology: Mapping[str, Any],
) -> dict[str, Any]:
    sample_id = queue_item["sample_id"]
    if decision.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"{sample_id}: decision protocol_version is invalid")
    if decision.get("sample_id") != sample_id:
        raise ValueError(f"{sample_id}: decision sample_id mismatch")
    if decision.get("video_sha256") != queue_item["anonymous_video_sha256"]:
        raise ValueError(f"{sample_id}: decision video_sha256 mismatch")
    if decision.get("intent_ontology_sha256") != queue_item["intent_ontology_sha256"]:
        raise ValueError(f"{sample_id}: decision intent ontology SHA256 mismatch")
    if decision.get("label_metadata_exposed") is not False:
        raise ValueError(f"{sample_id}: source label metadata was exposed")
    if decision.get("audio_available") is not False:
        raise ValueError(f"{sample_id}: audio may not be used for primary intent review")
    exposed_fields = FORBIDDEN_REVIEW_FIELDS & set(decision)
    if exposed_fields:
        raise ValueError(
            f"{sample_id}: decision contains forbidden source fields {sorted(exposed_fields)}"
        )
    reviewer_id = decision.get("reviewer_id")
    review_id = decision.get("review_id")
    if not isinstance(reviewer_id, str) or not reviewer_id.strip():
        raise ValueError(f"{sample_id}: decision lacks reviewer_id")
    if not isinstance(review_id, str) or not review_id.strip():
        raise ValueError(f"{sample_id}: decision lacks review_id")
    result = decision.get("result")
    if result not in ALL_RESULTS:
        raise ValueError(f"{sample_id}: unknown review result {result!r}")

    normalized = dict(decision)
    normalized["reviewer_id"] = reviewer_id.strip()
    normalized["review_id"] = review_id.strip()
    if result == OBSERVABLE_RESULT:
        intent_id = decision.get("observable_intent_id")
        if intent_id not in set(observable_intent_ids(ontology)):
            raise ValueError(f"{sample_id}: observable result lacks a known intent")
        confidence = decision.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise ValueError(f"{sample_id}: observable decision confidence is invalid")
        if decision.get("hard_negative_checked") is not True:
            raise ValueError(f"{sample_id}: hard-negative check is required")
        normalized["confidence"] = float(confidence)
    elif decision.get("observable_intent_id") is not None:
        raise ValueError(f"{sample_id}: non-observable result must not carry an intent")
    return normalized


def _pending_annotation(queue_item: Mapping[str, Any], status: str) -> dict[str, Any]:
    return {
        "intent_ontology_id": queue_item["intent_ontology_id"],
        "intent_ontology_sha256": queue_item["intent_ontology_sha256"],
        "observable_intent_id": None,
        "intent_review_status": status,
        "intent_supervision_mask": False,
        "intent_conditioning_mask": False,
        "pragmatic_role": None,
        "pragmatic_role_supervision_mask": False,
    }


def adjudicate_item(
    queue_item: Mapping[str, Any],
    raw_decisions: list[Mapping[str, Any]],
    ontology: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    sample_id = queue_item["sample_id"]
    if not raw_decisions:
        return "pending", _pending_annotation(queue_item, "pending_review")
    decisions = [
        _validate_decision(decision, queue_item, ontology) for decision in raw_decisions
    ]
    reviewer_ids = [decision["reviewer_id"] for decision in decisions]
    if len(set(reviewer_ids)) != len(reviewer_ids):
        raise ValueError(f"{sample_id}: reviewer ids must be independent and unique")

    minimum_reviewers = int(ontology["review_contract"]["minimum_independent_reviewers"])
    if len(decisions) < minimum_reviewers:
        return "pending", _pending_annotation(queue_item, "pending_review")

    results = {decision["result"] for decision in decisions}
    observable = [
        decision for decision in decisions if decision["result"] == OBSERVABLE_RESULT
    ]
    if len(observable) == len(decisions):
        intent_ids = {decision["observable_intent_id"] for decision in observable}
        minimum_confidence = min(float(decision["confidence"]) for decision in observable)
        required_confidence = float(ontology["review_contract"]["minimum_confidence"])
        if len(intent_ids) == 1 and minimum_confidence >= required_confidence:
            intent_id = next(iter(intent_ids))
            definition = intent_definition(intent_id, ontology)
            prompt = definition["canonical_prompt_en"]
            annotation = {
                "intent_ontology_id": queue_item["intent_ontology_id"],
                "intent_ontology_sha256": queue_item["intent_ontology_sha256"],
                "observable_intent_id": intent_id,
                "intent_slot": int(definition["slot"]),
                "intent_one_hot": build_observable_intent_one_hot(
                    intent_id, ontology
                ).tolist(),
                "intent_review_status": "independent_blind_consensus",
                "intent_supervision_mask": True,
                "intent_conditioning_mask": True,
                "intent_prompt": prompt,
                "intent_prompt_zh": definition["canonical_prompt_zh"],
                "intent_prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                "intent_review_evidence": {
                    "protocol_version": PROTOCOL_VERSION,
                    "review_ids": [decision["review_id"] for decision in observable],
                    "reviewer_ids": reviewer_ids,
                    "video_sha256": queue_item["anonymous_video_sha256"],
                    "minimum_confidence": minimum_confidence,
                    "label_metadata_exposed": False,
                    "audio_available": False,
                    "hard_negative_checked": True,
                },
                "pragmatic_role": None,
                "pragmatic_role_supervision_mask": False,
            }
            validate_observable_intent_annotation(
                annotation,
                ontology,
                expected_ontology_sha256=queue_item["intent_ontology_sha256"],
            )
            return "train_ready", annotation
        return "pending", _pending_annotation(queue_item, "pending_adjudication")

    if results.issubset(NON_OBSERVABLE_RESULTS) and len(decisions) >= minimum_reviewers:
        return "rejected", _pending_annotation(queue_item, "rejected")
    if results == {QUALITY_FAILURE_RESULT}:
        return "rejected", _pending_annotation(queue_item, "rejected")
    return "pending", _pending_annotation(queue_item, "pending_adjudication")


def _load_decisions(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        for decision in _read_jsonl(path):
            sample_id = decision.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(f"{path}: decision lacks sample_id")
            by_sample[sample_id.strip()].append(decision)
    return by_sample


def build_outputs(
    *,
    input_manifest: Path,
    video_dir: Path,
    ontology_path: Path,
    output_dir: Path,
    decision_paths: list[Path],
) -> dict[str, Any]:
    source_records = _read_jsonl(input_manifest)
    source_by_sample = {_sample_id(record): record for record in source_records}
    if len(source_by_sample) != len(source_records):
        raise ValueError("input manifest contains duplicate sample ids")
    queue = build_review_queue(
        source_records,
        source_manifest=input_manifest,
        video_dir=video_dir,
        ontology_path=ontology_path,
    )
    ontology = load_observable_intent_ontology(ontology_path)
    decisions = _load_decisions(decision_paths)
    queue_ids = {item["sample_id"] for item in queue}
    unknown_decisions = set(decisions) - queue_ids
    if unknown_decisions:
        raise ValueError(f"decisions reference unknown samples: {sorted(unknown_decisions)}")

    pending: list[dict[str, Any]] = []
    train_ready: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    intent_counts: Counter[str] = Counter()
    for item in queue:
        disposition, annotation = adjudicate_item(
            item, decisions.get(item["sample_id"], []), ontology
        )
        status_counts[annotation["intent_review_status"]] += 1
        if disposition == "train_ready":
            intent_counts[annotation["observable_intent_id"]] += 1
            source_record = source_by_sample[item["sample_id"]]
            motion_prompt = source_record.get("prompt")
            motion_prompt_provenance = source_record.get("prompt_text_provenance")
            if (
                not isinstance(motion_prompt, str)
                or not motion_prompt.strip()
                or motion_prompt_provenance not in VERIFIED_MOTION_PROMPT_PROVENANCES
            ):
                motion_prompt = None
                motion_prompt_provenance = None
            semantic_prompt = annotation["intent_prompt"]
            semantic_prompt_source = "reviewed_observable_intent_canonical_prompt_v1"
            if motion_prompt is not None:
                semantic_prompt = (
                    f"{semantic_prompt} Motion realization: {motion_prompt.strip()}"
                )
                semantic_prompt_source = (
                    "reviewed_observable_intent_plus_existing_motion_form_prompt_v1"
                )
            train_ready.append(
                {
                    "schema_version": OUTPUT_SCHEMA_VERSION,
                    "artifact_kind": "ula_v9_observable_intent_training_overlay",
                    "network_condition_contract": "ula_v2_18d_observable_intent_v9",
                    "primary_semantic_channel": "observable_intent_one_hot",
                    "text_channel_role": "auxiliary_semantic_prompt",
                    "legacy_behavior_conditioning_enabled": False,
                    "legacy_intent_conditioning_enabled": False,
                    "sample_id": item["sample_id"],
                    "source_record_sha256": item["sealed_source_provenance"][
                        "source_record_sha256"
                    ],
                    "source_manifest_sha256": item["sealed_source_provenance"][
                        "source_manifest_sha256"
                    ],
                    "semantic_prompt": semantic_prompt,
                    "semantic_prompt_sha256": hashlib.sha256(
                        semantic_prompt.encode("utf-8")
                    ).hexdigest(),
                    "semantic_prompt_source": semantic_prompt_source,
                    "motion_realization_prompt": motion_prompt,
                    "motion_realization_prompt_provenance": motion_prompt_provenance,
                    **annotation,
                }
            )
        elif disposition == "rejected":
            rejected.append(
                {
                    "schema_version": OUTPUT_SCHEMA_VERSION,
                    "artifact_kind": "ula_v9_observable_intent_reject",
                    "sample_id": item["sample_id"],
                    "reason": "independent_review_not_observable_in_18d_ontology",
                    **annotation,
                }
            )
        else:
            pending_item = dict(item)
            pending_item.update(annotation)
            pending_item["received_review_count"] = len(decisions.get(item["sample_id"], []))
            pending.append(pending_item)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "intent_review_all.jsonl", queue)
    _write_jsonl(output_dir / "intent_review_pending.jsonl", pending)
    _write_jsonl(output_dir / "intent_train_ready.jsonl", train_ready)
    _write_jsonl(output_dir / "intent_reject.jsonl", rejected)
    for reviewer_name in ("reviewer_a", "reviewer_b"):
        _write_jsonl(
            output_dir / f"{reviewer_name}_decision_template.jsonl",
            (
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "sample_id": item["sample_id"],
                    "video_sha256": item["anonymous_video_sha256"],
                    "intent_ontology_sha256": item["intent_ontology_sha256"],
                    "label_metadata_exposed": False,
                    "audio_available": False,
                    "result": None,
                    "observable_intent_id": None,
                    "confidence": None,
                    "hard_negative_checked": None,
                    "hard_negative_notes": None,
                    "reviewer_id": None,
                    "review_id": None,
                    "notes": None,
                }
                for item in pending
            ),
        )

    summary = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "artifact_kind": "ula_v9_observable_intent_review_summary",
        "ontology_id": ontology["ontology_id"],
        "ontology_path": str(ontology_path.resolve()),
        "ontology_sha256": ontology_sha256(ontology_path),
        "input_manifest": str(input_manifest.resolve()),
        "input_manifest_sha256": _file_sha256(input_manifest),
        "source_record_count": len(source_records),
        "review_queue_count": len(queue),
        "train_ready_count": len(train_ready),
        "pending_count": len(pending),
        "rejected_count": len(rejected),
        "review_status_counts": dict(sorted(status_counts.items())),
        "train_ready_intent_counts": dict(sorted(intent_counts.items())),
        "automatic_intent_labels_emitted": 0,
        "filename_or_transcript_used_for_intent": False,
        "audio_used_for_primary_intent": False,
        "pragmatic_role_conditioning_enabled": False,
        "network_admission": (
            "Only intent_train_ready.jsonl may enable the v9 explicit intent one-hot. "
            "Pending and rejected records must keep the intent block zero."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--review-decisions",
        type=Path,
        action="append",
        default=[],
        help="Independent blind decision JSONL; may be supplied more than once.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_outputs(
        input_manifest=args.input_manifest,
        video_dir=args.video_dir,
        ontology_path=args.ontology,
        output_dir=args.output_dir,
        decision_paths=args.review_decisions,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
