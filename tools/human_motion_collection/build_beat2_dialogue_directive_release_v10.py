#!/usr/bin/env python3
"""Build BEAT2 directive + dialogue motion-alignment data before model changes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.human_motion_collection.build_beat2_semantic_event_inventory import (
    parse_words_textgrid_intervals,
)
from upper_body_skeleton.ula_v2_dialogue_directive_episode import (
    ARTIFACT_KIND,
    DIALOGUE_ALIGNMENT_POLICY,
    DIALOGUE_ROLE,
    DIRECTIVE_ROLE,
    FORMAL_ELIGIBILITY_MODE,
    FORMAL_EPISODE_CONTRACT,
    TRAINING_SEGMENT_REPRESENTATION,
    canonical_sha256,
    text_sha256,
    validate_dialogue_directive_v10_episode,
)


DEFAULT_BASE_V9_DIR = (
    PROJECT_ROOT
    / "deliverables/expressive_human_motion_v2/robot_observable_intents_v1/"
    "beat2_conversational_realization_release_v9"
)
DEFAULT_BASE_V9_MANIFEST = DEFAULT_BASE_V9_DIR / "train_ready.jsonl"
DEFAULT_BASE_V9_LOCK = DEFAULT_BASE_V9_DIR / "provenance_lock.json"
DEFAULT_MOTION_MANIFEST = Path(
    "/home/gez/nas/cloud/gez/human_motion/processed/"
    "beat2_semantic_event_training_pool_18d_v8/expansion/release/"
    "adjudication_min30f/train_ready.jsonl"
)
DEFAULT_BEAT2_ROOT = Path("/home/gez/nas/cloud/gez/human_motion/raw/BEAT2")
DEFAULT_OUTPUT_DIR = Path(
    "/home/gez/nas/cloud/gez/human_motion/processed/"
    "beat2_dialogue_directive_release_v10"
)
SUMMARY_KIND = "beat2_dialogue_directive_release_v10_summary"
PROVENANCE_LOCK_KIND = "ula_v2_dialogue_directive_v10_provenance_lock_v1"
COUNTERFACTUAL_KIND = "ula_v2_dialogue_directive_v10_counterfactual_pair_v1"
EXCLUSION_KIND = "ula_v2_dialogue_directive_v10_exclusion_v1"
BASE_V9_CONTRACT = "ula_v2_18d_conversational_realization_v9_episode_v1"
BASE_V9_LOCK_KIND = "ula_v2_conversational_realization_v9_provenance_lock_v1"
WINDOW_TRANSCRIPT_ROLE = "time_aligned_speech_context_only_not_action_or_emotion_label"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            records.append(value)
    if not records:
        raise ValueError(f"manifest is empty: {path}")
    return records


def _index(records: Sequence[Mapping[str, Any]], *, context: str) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for record in records:
        clip_id = str(record.get("clip_id") or record.get("task_id") or "").strip()
        if not clip_id or clip_id in indexed:
            raise ValueError(f"{context} has a missing or duplicate clip_id: {clip_id!r}")
        indexed[clip_id] = record
    return indexed


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
    os.replace(temporary, path)


def _resolve_under(root: Path, relative: object, *, clip_id: str) -> Path:
    value = str(relative or "").strip()
    if not value:
        raise ValueError(f"{clip_id}: textgrid_relpath is missing")
    root = root.resolve()
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{clip_id}: TextGrid path escapes BEAT2 root") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def aligned_dialogue_tokens(
    intervals: Sequence[tuple[float, float, str]],
    *,
    start_sec: float,
    end_sec: float,
) -> list[str]:
    """Return readable words aligned to one native motion interval."""

    if end_sec <= start_sec:
        raise ValueError("dialogue interval must have positive duration")
    return [
        " ".join(text.strip().split())
        for interval_start, interval_end, text in intervals
        if text.strip() and interval_end > start_sec and interval_start < end_sec
    ]


def _style_signature(record: Mapping[str, Any]) -> tuple[str, ...]:
    style = record.get("motion_style") or {}
    return tuple(
        str(style.get(field) or "unknown")
        for field in ("arm_amplitude", "laterality", "pace", "head_engagement")
    )


def _deterministic_choice(
    anchor: Mapping[str, Any],
    index: Mapping[str, Mapping[object, list[Mapping[str, Any]]]],
    *,
    channel: str,
) -> tuple[Mapping[str, Any], str]:
    anchor_id = str(anchor["clip_id"])
    split = anchor["fixed_split_assignment"]
    dialogue_hash = anchor["dialogue_text_sha256"]
    directive_hash = anchor["directive_text_sha256"]
    style = _style_signature(anchor)
    bucket_specs = (
        (
            index["speaker_style"].get((split, anchor["speaker_key"], style), []),
            "same_speaker_same_style_different_source_group",
        ),
        (
            index["speaker"].get((split, anchor["speaker_key"]), []),
            "same_speaker_nearest_style_different_source_group",
        ),
        (
            index["style"].get((split, style), []),
            "different_speaker_same_style_different_source_group",
        ),
        (
            index["split"].get(split, []),
            "split_safe_different_source_group_fallback",
        ),
    )

    def score(record: Mapping[str, Any]) -> tuple[float, str]:
        frame_ratio = max(record["frames"], anchor["frames"]) / max(
            1, min(record["frames"], anchor["frames"])
        )
        return (
            abs(frame_ratio - 1.0),
            hashlib.sha256(
                f"{channel}|{anchor_id}|{record['clip_id']}".encode("utf-8")
            ).hexdigest(),
        )

    for bucket, tier in bucket_specs:
        candidates = [
            record
            for record in bucket
            if record["clip_id"] != anchor_id
            and record["source_group_key"] != anchor["source_group_key"]
            and (
                record["dialogue_text_sha256"] != dialogue_hash
                if channel == "dialogue"
                else record["directive_text_sha256"] != directive_hash
            )
        ]
        if candidates:
            return min(candidates, key=score), tier
    raise ValueError(f"{anchor_id}: no split-safe {channel} counterfactual exists")


def _counterfactual_index(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[object, list[Mapping[str, Any]]]]:
    result: dict[str, dict[object, list[Mapping[str, Any]]]] = {
        "speaker_style": defaultdict(list),
        "speaker": defaultdict(list),
        "style": defaultdict(list),
        "split": defaultdict(list),
    }
    for record in records:
        split = record["fixed_split_assignment"]
        speaker = record["speaker_key"]
        style = _style_signature(record)
        result["speaker_style"][(split, speaker, style)].append(record)
        result["speaker"][(split, speaker)].append(record)
        result["style"][(split, style)].append(record)
        result["split"][split].append(record)
    return result


def _counterfactual_record(
    anchor: Mapping[str, Any],
    index: Mapping[str, Mapping[object, list[Mapping[str, Any]]]],
) -> dict[str, Any]:
    dialogue_negative, dialogue_tier = _deterministic_choice(
        anchor, index, channel="dialogue"
    )
    directive_negative, directive_tier = _deterministic_choice(
        anchor, index, channel="directive"
    )
    record = {
        "schema_version": "1.0.0",
        "artifact_kind": COUNTERFACTUAL_KIND,
        "anchor_clip_id": anchor["clip_id"],
        "fixed_split_assignment": anchor["fixed_split_assignment"],
        "matched": {
            "directive_text_sha256": anchor["directive_text_sha256"],
            "dialogue_text_sha256": anchor["dialogue_text_sha256"],
            "trajectory_sha256": anchor["trajectory_sha256"],
        },
        "dialogue_shuffled": {
            "source_clip_id": dialogue_negative["clip_id"],
            "dialogue_text_sha256": dialogue_negative["dialogue_text_sha256"],
            "different_source_group": True,
            "same_split": True,
            "selection_tier": dialogue_tier,
        },
        "directive_shuffled": {
            "source_clip_id": directive_negative["clip_id"],
            "directive_text_sha256": directive_negative["directive_text_sha256"],
            "different_source_group": True,
            "same_split": True,
            "selection_tier": directive_tier,
        },
        "evaluation_policy": (
            "compare_matched_vs_dialogue_shuffled_vs_directive_shuffled_with_"
            "identical_motion_target"
        ),
        "accepted_as_positive_motion_target": False,
    }
    record["record_sha256"] = canonical_sha256(record)
    return record


def _validate_base_lock(
    lock: Mapping[str, Any], *, base_manifest: Path, base_manifest_sha256: str
) -> None:
    if (
        lock.get("artifact_kind") != BASE_V9_LOCK_KIND
        or lock.get("formal_episode_contract") != BASE_V9_CONTRACT
        or lock.get("scale_gate_passed") is not True
        or lock.get("audio_conditioning_enabled") is not False
        or lock.get("primary_intent_conditioning_enabled") is not False
        or lock.get("emotion_conditioning_enabled") is not False
        or lock.get("duration_policy")
        != "native_variable_length_conversational_event_no_fixed_duration_v1"
    ):
        raise ValueError("base v9 provenance lock is not a formal conversational release")
    declared = Path(str(lock.get("train_ready_manifest") or "")).resolve()
    if declared != base_manifest or lock.get("train_ready_manifest_sha256") != base_manifest_sha256:
        raise ValueError("base v9 provenance lock does not bind the supplied manifest")


def build_release(
    *,
    base_v9_manifest: str | Path,
    base_v9_lock: str | Path,
    motion_manifest: str | Path,
    beat2_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    base_v9_manifest = Path(base_v9_manifest).resolve()
    base_v9_lock = Path(base_v9_lock).resolve()
    motion_manifest = Path(motion_manifest).resolve()
    beat2_root = Path(beat2_root).resolve()
    output_dir = Path(output_dir).resolve()
    for path in (base_v9_manifest, base_v9_lock, motion_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not beat2_root.is_dir():
        raise FileNotFoundError(beat2_root)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_hash = _sha256_file(base_v9_manifest)
    base_lock_hash = _sha256_file(base_v9_lock)
    motion_hash = _sha256_file(motion_manifest)
    lock = _read_json(base_v9_lock)
    _validate_base_lock(lock, base_manifest=base_v9_manifest, base_manifest_sha256=base_hash)
    base_rows = _read_jsonl(base_v9_manifest)
    motion_rows = _read_jsonl(motion_manifest)
    base_by_id = _index(base_rows, context="base v9 manifest")
    motion_by_id = _index(motion_rows, context="motion manifest")
    if set(base_by_id) != set(motion_by_id):
        raise ValueError("base v9 and source motion manifests have different membership")

    interval_cache: dict[Path, list[tuple[float, float, str]]] = {}
    textgrid_hash_cache: dict[Path, str] = {}
    candidates: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for clip_id in sorted(base_by_id):
        base = base_by_id[clip_id]
        source = motion_by_id[clip_id]
        source_record_sha256 = canonical_sha256(source)
        if (
            base.get("formal_episode_contract") != BASE_V9_CONTRACT
            or base.get("accepted_for_training") is not True
            or base.get("source_record_sha256") != source_record_sha256
            or base.get("source_manifest_sha256") != motion_hash
            or Path(str(base.get("source_manifest") or "")).resolve() != motion_manifest
        ):
            raise ValueError(f"{clip_id}: base v9/source motion binding changed")
        if source.get("window_transcript_role") != WINDOW_TRANSCRIPT_ROLE:
            raise ValueError(f"{clip_id}: source transcript role changed")
        start = int(source.get("start_frame", -1))
        end = int(source.get("end_frame_exclusive", -1))
        if start < 0 or end <= start:
            raise ValueError(f"{clip_id}: invalid native source interval")
        textgrid_path = _resolve_under(
            beat2_root, source.get("textgrid_relpath"), clip_id=clip_id
        )
        if textgrid_path not in interval_cache:
            interval_cache[textgrid_path] = parse_words_textgrid_intervals(textgrid_path)
            textgrid_hash_cache[textgrid_path] = _sha256_file(textgrid_path)
        tokens = aligned_dialogue_tokens(
            interval_cache[textgrid_path],
            start_sec=start / 30.0,
            end_sec=end / 30.0,
        )
        legacy_text = str(source.get("window_transcript_context") or "")
        if "".join(tokens) != legacy_text:
            raise ValueError(f"{clip_id}: rebuilt TextGrid overlap differs from source lineage")
        base_record_sha256 = canonical_sha256(base)
        if not tokens:
            exclusion = {
                "schema_version": "1.0.0",
                "artifact_kind": EXCLUSION_KIND,
                "clip_id": clip_id,
                "reason": "no_nonempty_words_tier_token_overlaps_native_motion_interval",
                "retained_for_base_v9_directive_and_motion_training": True,
                "accepted_for_dialogue_conditioning": False,
                "base_v9_record_sha256": base_record_sha256,
                "source_record_sha256": source_record_sha256,
                "trajectory_sha256": base["trajectory_sha256"],
            }
            exclusion["record_sha256"] = canonical_sha256(exclusion)
            exclusions.append(exclusion)
            continue

        directive = str(base.get("prompt") or "").strip()
        dialogue = " ".join(tokens)
        episode = deepcopy(dict(base))
        episode.pop("source_transcript_semantics_used", None)
        episode.update(
            {
                "schema_version": "1.0.0",
                "artifact_kind": ARTIFACT_KIND,
                "formal_episode_contract": FORMAL_EPISODE_CONTRACT,
                "eligibility_mode": FORMAL_ELIGIBILITY_MODE,
                "accepted_for_training": True,
                "native_variable_length": True,
                "prompt": directive,
                "prompt_sha256": text_sha256(directive),
                "directive_text": directive,
                "directive_text_sha256": text_sha256(directive),
                "directive_conditioning_mask": True,
                "directive_contract": {
                    "role": DIRECTIVE_ROLE,
                    "source": "verified_conversational_motion_realization_prompt_v9",
                    "primary_control": True,
                    "derived_from_dialogue_text": False,
                    "specific_intent_supervision": False,
                },
                "dialogue_text": dialogue,
                "dialogue_text_sha256": text_sha256(dialogue),
                "dialogue_conditioning_mask": True,
                "dialogue_contract": {
                    "role": DIALOGUE_ROLE,
                    "source": "beat2_words_textgrid_native_interval_overlap_v1",
                    "alignment_policy": DIALOGUE_ALIGNMENT_POLICY,
                    "auxiliary_context": True,
                    "action_label_supervision": False,
                    "emotion_label_supervision": False,
                    "partner_response_supervision": False,
                    "language_code": str(source.get("language_code") or "en"),
                    "interval_start_sec": start / 30.0,
                    "interval_end_sec": end / 30.0,
                    "token_count": len(tokens),
                    "token_sequence_sha256": canonical_sha256({"tokens": tokens}),
                    "textgrid_relpath": str(source["textgrid_relpath"]),
                    "textgrid_sha256": textgrid_hash_cache[textgrid_path],
                    "legacy_compact_context_sha256": text_sha256(legacy_text),
                },
                "source_transcript_conditioning_used": True,
                "source_transcript_used_as_action_or_emotion_label": False,
                "partner_response_supervision_mask": False,
                "self_speech_gesture_supervision_mask": True,
                "base_v9_manifest": str(base_v9_manifest),
                "base_v9_manifest_sha256": base_hash,
                "base_v9_record_sha256": base_record_sha256,
                "source_manifest": str(motion_manifest),
                "source_manifest_sha256": motion_hash,
                "source_record_sha256": source_record_sha256,
                "training_segment": {
                    **deepcopy(dict(base["training_segment"])),
                    "representation": TRAINING_SEGMENT_REPRESENTATION,
                    "fixed_window_sec": None,
                    "cropped": False,
                },
            }
        )
        candidates.append(episode)

    counterfactual_index = _counterfactual_index(candidates)
    counterfactuals = [
        _counterfactual_record(row, counterfactual_index) for row in candidates
    ]
    counterfactual_by_id = {row["anchor_clip_id"]: row for row in counterfactuals}
    formal: list[dict[str, Any]] = []
    for episode in candidates:
        counterfactual = counterfactual_by_id[episode["clip_id"]]
        episode["dialogue_motion_alignment"] = {
            "positive_pair": True,
            "evidence": "exact_words_tier_overlap_with_native_motion_interval",
            "supervision": "weak_temporal_co_speech_alignment",
            "specific_action_label": False,
            "hard_negative_required": True,
            "hard_negative_record_sha256": counterfactual["record_sha256"],
        }
        episode["training_admission"] = {
            "contract": FORMAL_EPISODE_CONTRACT,
            "trajectory_sha256": episode["trajectory_sha256"],
            "source_record_sha256": episode["source_record_sha256"],
            "base_v9_record_sha256": episode["base_v9_record_sha256"],
            "directive_text_sha256": episode["directive_text_sha256"],
            "dialogue_text_sha256": episode["dialogue_text_sha256"],
            "textgrid_sha256": episode["dialogue_contract"]["textgrid_sha256"],
            "hard_negative_record_sha256": counterfactual["record_sha256"],
            "training_channel_masks": {
                "motion_18d": True,
                "directive_text": True,
                "dialogue_text": True,
                "dialogue_motion_alignment": True,
                "trajectory_style": True,
                "primary_intent": False,
                "emotion": False,
                "partner_response": False,
                "audio": False,
            },
        }
        validate_dialogue_directive_v10_episode(episode)
        formal.append(episode)

    output_manifest = output_dir / "train_ready.jsonl"
    exclusion_manifest = output_dir / "dialogue_unavailable_retained_in_v9.jsonl"
    counterfactual_manifest = output_dir / "counterfactual_pairs.jsonl"
    _atomic_jsonl(output_manifest, formal)
    _atomic_jsonl(exclusion_manifest, exclusions)
    _atomic_jsonl(counterfactual_manifest, counterfactuals)
    output_hash = _sha256_file(output_manifest)
    exclusion_hash = _sha256_file(exclusion_manifest)
    counterfactual_hash = _sha256_file(counterfactual_manifest)

    split_counts = Counter(row["fixed_split_assignment"] for row in formal)
    dialogue_tiers = Counter(
        row["dialogue_shuffled"]["selection_tier"] for row in counterfactuals
    )
    directive_tiers = Counter(
        row["directive_shuffled"]["selection_tier"] for row in counterfactuals
    )
    token_counts = [row["dialogue_contract"]["token_count"] for row in formal]
    frame_counts = [int(row["frames"]) for row in formal]
    scale = {
        "base_v9_episode_count": len(base_rows),
        "dialogue_conditioned_episode_count": len(formal),
        "dialogue_unavailable_retained_in_v9_count": len(exclusions),
        "all_base_motion_accounted_for": len(formal) + len(exclusions) == len(base_rows),
        "frame_count": sum(frame_counts),
        "sample_span_sec": sum((frames - 1) / 30.0 for frames in frame_counts),
        "minimum_frames": min(frame_counts),
        "maximum_frames": max(frame_counts),
        "distinct_frame_counts": len(set(frame_counts)),
        "speaker_count": len({row["speaker_key"] for row in formal}),
        "source_group_count": len({row["source_group_key"] for row in formal}),
        "distinct_directive_text_count": len({row["directive_text"] for row in formal}),
        "distinct_dialogue_text_count": len({row["dialogue_text"] for row in formal}),
        "minimum_dialogue_token_count": min(token_counts),
        "maximum_dialogue_token_count": max(token_counts),
        "fixed_split_counts": dict(sorted(split_counts.items())),
        "counterfactual_pair_count": len(counterfactuals),
        "dialogue_negative_tier_counts": dict(sorted(dialogue_tiers.items())),
        "directive_negative_tier_counts": dict(sorted(directive_tiers.items())),
    }
    provenance_lock = {
        "schema_version": 1,
        "artifact_kind": PROVENANCE_LOCK_KIND,
        "accepted_for_training": True,
        "formal_episode_contract": FORMAL_EPISODE_CONTRACT,
        "duration_policy": "native_variable_length_no_fixed_or_target_duration_v1",
        "fixed_six_second_training_unit": False,
        "base_v9_manifest": str(base_v9_manifest),
        "base_v9_manifest_sha256": base_hash,
        "base_v9_provenance_lock": str(base_v9_lock),
        "base_v9_provenance_lock_sha256": base_lock_hash,
        "source_motion_manifest": str(motion_manifest),
        "source_motion_manifest_sha256": motion_hash,
        "beat2_root": str(beat2_root),
        "train_ready_manifest": str(output_manifest),
        "train_ready_manifest_sha256": output_hash,
        "counterfactual_pair_manifest": str(counterfactual_manifest),
        "counterfactual_pair_manifest_sha256": counterfactual_hash,
        "dialogue_unavailable_manifest": str(exclusion_manifest),
        "dialogue_unavailable_manifest_sha256": exclusion_hash,
        "dataset_scale": scale,
        "conditioning_contract": {
            "directive_text": "primary_robot_brain_control",
            "dialogue_text": "auxiliary_time_aligned_co_speech_context",
            "dialogue_motion_alignment": "positive_plus_split_safe_hard_negative",
            "audio": "disabled_current_stage",
            "primary_intent": "not_inferred_from_transcript",
            "emotion": "not_inferred_from_transcript",
        },
        "scenario_support": {
            "robot_speaks_with_content_appropriate_gestures": "directly_supervised",
            "brain_directive_plus_dialogue_context_schema": "supported",
            "user_speaks_robot_listener_response": (
                "not_directly_supervised_by_beat2_self_speech_motion_pairs"
            ),
        },
    }
    provenance_path = output_dir / "provenance_lock.json"
    _atomic_json(provenance_path, provenance_lock)
    summary = {
        "schema_version": "1.0.0",
        "artifact_kind": SUMMARY_KIND,
        "formal_episode_contract": FORMAL_EPISODE_CONTRACT,
        "train_ready_manifest": str(output_manifest),
        "train_ready_manifest_sha256": output_hash,
        "counterfactual_pair_manifest": str(counterfactual_manifest),
        "counterfactual_pair_manifest_sha256": counterfactual_hash,
        "dialogue_unavailable_manifest": str(exclusion_manifest),
        "dialogue_unavailable_manifest_sha256": exclusion_hash,
        "provenance_lock": str(provenance_path),
        "dataset_scale": scale,
        "all_motion_accounted_for": scale["all_base_motion_accounted_for"],
        "fixed_six_second_training_unit": False,
        "network_modified_or_trained": False,
        "next_gate": (
            "dataset_integrity_and_dialogue_motion_counterfactual_audit_before_"
            "network_change"
        ),
        "scenario_support": provenance_lock["scenario_support"],
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-v9-manifest", type=Path, default=DEFAULT_BASE_V9_MANIFEST)
    parser.add_argument("--base-v9-lock", type=Path, default=DEFAULT_BASE_V9_LOCK)
    parser.add_argument("--motion-manifest", type=Path, default=DEFAULT_MOTION_MANIFEST)
    parser.add_argument("--beat2-root", type=Path, default=DEFAULT_BEAT2_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_release(
        base_v9_manifest=args.base_v9_manifest,
        base_v9_lock=args.base_v9_lock,
        motion_manifest=args.motion_manifest,
        beat2_root=args.beat2_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
