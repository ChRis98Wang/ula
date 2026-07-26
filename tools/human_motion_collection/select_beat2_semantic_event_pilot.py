#!/usr/bin/env python3
"""Select a reproducible, speaker-disjoint BEAT2 semantic-event pilot.

Official BEAT2 source splits are retained as provenance only because speakers
occur in more than one official split.  This selector derives a strict pilot
split at speaker granularity, then balances native-length semantic events over
network emotion, official gesture category, and official intensity.  Audio is
never read and fixed-duration windows are denied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path(
    "/home/gez/nas/cloud/gez/human_motion/catalog/"
    "beat2_semantic_event_inventory_v1/"
    "beat2_semantic_event_inventory_v1.network_emotion_supported.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/gez/nas/cloud/gez/human_motion/catalog/"
    "beat2_semantic_event_pilot_v7_full"
)
OUTPUT_STEM = "beat2_semantic_event_pilot_v7_full"
TRAINING_POOL_FILENAME = "training_pool_low_medium.jsonl"
SCHEMA_VERSION = "3.3.0"
DATASET_SUBSET = "beat_english_v2.0.0"
VARIABLE_REPRESENTATION = "native_variable_length_semantic_clip_v1"
SELECTION_STATUS = "official_semantic_event_variable_length_boundary_validated"
ANNOTATION_KIND = "official_gesture_semantic_event"
BEHAVIOR_MAPPING_SOURCE = "project_dataset_scope_weak_mapping_v1"
BEHAVIOR_MAPPING_REVISION = "beat2_semantic_event_pilot_v7_full"
OFFICIAL_EMOTION_SOURCE = "official_beat2_filename_protocol"
OFFICIAL_EMOTION_REVISION = "beat_english_v2.0.0_filename_protocol_v1"
PROMPT_SCHEMA_VERSION = "beat2_official_semantic_event_qwen_prompt_v5"
PROMPT_SOURCE = "deterministic_official_semantic_event_and_emotion_v5"
NETWORK_EMOTIONS = ("neutral", "sad", "happy", "angry", "surprise", "fear")
SEMANTIC_CATEGORIES = ("deictic", "iconic", "metaphoric")
INTENSITIES = ("low", "medium", "high")
INTENSITY_CODES = {"low": "l", "medium": "m", "high": "h"}
SOURCE_LABELS = {
    ("deictic", "low"): "02_deictic_l",
    ("deictic", "medium"): "03_deictic_m",
    ("deictic", "high"): "04_deictic_h",
    ("iconic", "low"): "05_iconic_l",
    ("iconic", "medium"): "06_iconic_m",
    ("iconic", "high"): "07_iconic_h",
    ("metaphoric", "low"): "08_metaphoric_l",
    ("metaphoric", "medium"): "09_metaphoric_m",
    ("metaphoric", "high"): "10_metaphoric_h",
}
SPLITS = ("train", "validation", "test")
DEFAULT_SPLIT_FRACTIONS = {"train": 0.7, "validation": 0.15, "test": 0.15}
DYNAMIC_BAND_ORDER = {"low": 0, "medium": 1, "high_fallback": 2}
HEX_DIGITS = frozenset("0123456789abcdef")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--assignment-trials", type=int, default=4096)
    parser.add_argument("--train-per-stratum", type=int, default=1)
    parser.add_argument("--validation-per-stratum", type=int, default=1)
    parser.add_argument("--test-per-stratum", type=int, default=1)
    parser.add_argument("--max-events-per-source", type=int, default=3)
    parser.add_argument("--min-energy-mean", type=float, default=0.02)
    parser.add_argument("--low-energy-p95", type=float, default=1.0)
    parser.add_argument("--medium-energy-p95", type=float, default=3.0)
    parser.add_argument("--max-energy-p95", type=float, default=4.0)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Return success for a diagnostic pilot with missing requested strata",
    )
    return parser.parse_args(argv)


def stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_sha256(record: dict[str, Any]) -> str:
    return sha256_bytes(stable_json(record).encode("utf-8"))


def content_contract(payload: dict[str, Any]) -> dict[str, Any]:
    """Embed the canonical ASCII hash expected by downstream admission gates."""
    result = dict(payload)
    result["sha256"] = sha256_bytes(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    )
    return result


def qwen_prompt_provenance(record: dict[str, Any]) -> dict[str, Any]:
    event = record["semantic_event"]
    category = str(event["category"])
    intensity = str(event["intensity"])
    emotion_id = str(record["emotion_id"])
    category_phrase = {
        "deictic": "deictic pointing gesture",
        "iconic": "iconic upper-body gesture",
        "metaphoric": "metaphoric upper-body gesture",
    }[category]
    affect_phrase = {
        "neutral": "with neutral affect",
        "sad": "with a sad affect",
        "happy": "with a happy affect",
        "angry": "with an angry affect",
        "surprise": "with a surprised affect",
        "fear": "with a fearful affect",
    }[emotion_id]
    prompt = f"Perform a {intensity}-intensity {category_phrase} {affect_phrase}."
    lexical_anchor = event.get("source_lexical_anchor")
    lexical_anchor = lexical_anchor.strip() if isinstance(lexical_anchor, str) else None
    lexical_anchor = lexical_anchor or None
    prompt_sha256 = sha256_bytes(prompt.encode("utf-8"))
    schema = {
        "schema_version": PROMPT_SCHEMA_VERSION,
        "language": "en",
        "canonical_prompt_role": "coarse_category_only",
        "motion_instruction": {
            "official_semantic_category": category,
            "official_semantic_intensity": intensity,
            "official_emotion_id": emotion_id,
            "emotion_source": OFFICIAL_EMOTION_SOURCE,
        },
        "speech_context": {
            "lexical_anchor": lexical_anchor,
            "role": "speech_context_only_not_motion_instruction_or_emotion_label",
            "included_in_canonical_prompt": False,
        },
        "emotion_resolution": {
            "source": OFFICIAL_EMOTION_SOURCE,
            "lexical_anchor_used_for_emotion": False,
            "transcript_used_for_emotion": False,
        },
    }
    contract = content_contract(
        {
            "source": PROMPT_SOURCE,
            "revision": PROMPT_SCHEMA_VERSION,
            "prompt_sha256": prompt_sha256,
            "official_semantic_category": category,
            "official_semantic_intensity": intensity,
            "official_emotion_id": emotion_id,
            "canonical_prompt_role": "coarse_category_only",
            "prompt_text_supervision_mask": False,
            "lexical_anchor_role": schema["speech_context"]["role"],
            "lexical_anchor_included_in_canonical_prompt": False,
        }
    )
    return {
        "canonical_prompt": {"en": prompt},
        "canonical_prompt_role": "coarse_category_only",
        "prompt": prompt,
        "prompt_schema": schema,
        "prompt_source": PROMPT_SOURCE,
        "prompt_sha256": prompt_sha256,
        "prompt_contract": contract,
    }


def training_semantic_provenance(record: dict[str, Any]) -> dict[str, Any]:
    source_sha256 = str(record["motion_sha256"])
    emotion_id = str(record["emotion_id"])
    category = str(record["semantic_event"]["category"])
    return {
        "canonical_action": f"official_gesture_category:{category}",
        "canonical_action_role": "official_category_metadata_split_key_only",
        "semantic_mapping_status": "official_category_verified_metadata_only",
        "official_category_verified": True,
        "official_category_role": "verified_metadata_split_and_evaluation_only",
        "official_category_condition_channel": None,
        "official_category_loss": None,
        "official_category_conditioning_enabled": False,
        "robot_observable_motion_form": "candidate_unreviewed",
        "communicative_intent": "candidate_unreviewed",
        "semantic_supervision_masks": {
            "official_category": False,
            "robot_observable_motion_form": False,
            "communicative_intent": False,
            "prompt_text": False,
            "legacy_gesture": False,
        },
        "behavior_id": "Behavior.InteractPresence",
        "behavior_review_status": "candidate_unreviewed",
        "behavior_supervision_mask": False,
        "behavior_source": BEHAVIOR_MAPPING_SOURCE,
        "behavior_mapping_contract": content_contract(
            {
                "source": BEHAVIOR_MAPPING_SOURCE,
                "revision": BEHAVIOR_MAPPING_REVISION,
                "behavior_id": "Behavior.InteractPresence",
                "supervision": "weak_candidate_masked",
                "scope": "beat2_human_co_speech_interaction_dataset_scope",
                "rationale": (
                    "project_weak_mapping_not_an_official_beat2_behavior_annotation"
                ),
            }
        ),
        "semantic_gesture": (
            "pointing" if category == "deictic" else "upper_body_gesture"
        ),
        "emotion_review_status": "official_protocol_confirmed",
        "emotion_supervision_mask": False,
        "source_emotion_label_verified": True,
        "emotion_supervision_role": "disabled_pending_robot_affect_review",
        "official_emotion_conditioning_enabled": False,
        "official_emotion_condition_channel": None,
        "official_emotion_loss": None,
        "affect_observable_review_status": "candidate_unreviewed",
        "affect_observable_supervision_mask": False,
        "emotion_source": OFFICIAL_EMOTION_SOURCE,
        "emotion_protocol_contract": content_contract(
            {
                "source": OFFICIAL_EMOTION_SOURCE,
                "revision": OFFICIAL_EMOTION_REVISION,
                "emotion_id": emotion_id,
                "source_sha256": source_sha256,
                "source_emotion_label_verified": True,
                "supervision_role": "disabled_pending_robot_affect_review",
                "conditioning_enabled": False,
                "condition_channel": None,
                "loss": None,
            }
        ),
        "source_sha256": source_sha256,
        **qwen_prompt_provenance(record),
    }


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    atomic_bytes(path, payload)
    return sha256_bytes(payload)


def atomic_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> str:
    payload = "".join(stable_json(record) + "\n" for record in records).encode(
        "utf-8"
    )
    atomic_bytes(path, payload)
    return sha256_bytes(payload)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            value = dict(value)
            value["_selector_input_line"] = line_number
            records.append(value)
    if not records:
        raise ValueError(f"Semantic-event inventory is empty: {path}")
    return records


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _valid_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in HEX_DIGITS for character in text)


def semantic_stratum(record: dict[str, Any]) -> tuple[str, str, str]:
    event = record["semantic_event"]
    return (
        str(record["emotion_id"]),
        str(event["category"]),
        str(event["intensity"]),
    )


def all_strata() -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (emotion, category, intensity)
        for emotion in NETWORK_EMOTIONS
        for category in SEMANTIC_CATEGORIES
        for intensity in INTENSITIES
    )


def stratum_key(stratum: tuple[str, str, str]) -> str:
    return "|".join(stratum)


def dynamic_band(
    record: dict[str, Any], *, low_p95: float, medium_p95: float
) -> str:
    p95 = float(record["window"]["interaction_energy_p95_rad_s"])
    if p95 <= low_p95:
        return "low"
    if p95 <= medium_p95:
        return "medium"
    return "high_fallback"


def validate_candidate(
    record: dict[str, Any],
    *,
    min_energy_mean: float,
    max_energy_p95: float,
) -> list[str]:
    reasons = []
    if record.get("dataset_subset") != DATASET_SUBSET:
        reasons.append("dataset_subset_mismatch")
    if record.get("language") != "english" or record.get("language_code") != "en":
        reasons.append("language_contract_mismatch")
    if not str(record.get("clip_id") or "").strip():
        reasons.append("missing_clip_id")
    if not str(record.get("source_clip_id") or "").strip():
        reasons.append("missing_source_clip_id")
    if not str(record.get("speaker_key") or "").strip():
        reasons.append("missing_speaker_key")
    if record.get("emotion_id") not in NETWORK_EMOTIONS:
        reasons.append("unsupported_network_emotion")
    if record.get("emotion_supervision_mask") is not True:
        reasons.append("emotion_supervision_not_true")
    if record.get("emotion_label_source") != "official_beat2_filename_protocol":
        reasons.append("emotion_not_from_official_filename_protocol")
    if record.get("annotation_kind") != ANNOTATION_KIND:
        reasons.append("annotation_kind_mismatch")
    official_spans = record.get("official_gesture_semantic_spans")
    if not isinstance(official_spans, list) or not official_spans:
        reasons.append("missing_official_semantic_span_evidence")
    if record.get("audio_enabled") is not False:
        reasons.append("audio_must_be_disabled")
    if record.get("accepted_for_training") is not False:
        reasons.append("pre_retarget_training_admission_must_be_false")
    if record.get("issues") not in ([], None):
        reasons.append("inventory_has_unresolved_issues")
    if record.get("interaction_scope") != "human_co_speech_interaction":
        reasons.append("interaction_scope_mismatch")

    event = record.get("semantic_event")
    if not isinstance(event, dict):
        reasons.append("missing_semantic_event")
    else:
        category = event.get("category")
        intensity = event.get("intensity")
        if category not in SEMANTIC_CATEGORIES:
            reasons.append("unsupported_semantic_category")
        if intensity not in INTENSITIES:
            reasons.append("unsupported_semantic_intensity")
        lexical_anchor = event.get("source_lexical_anchor")
        if lexical_anchor is not None and not isinstance(lexical_anchor, str):
            reasons.append("invalid_lexical_anchor_type")
        if category in SEMANTIC_CATEGORIES and intensity in INTENSITIES:
            if event.get("source_label") != SOURCE_LABELS[(category, intensity)]:
                reasons.append("official_source_label_mismatch")
            expected_code = INTENSITY_CODES[intensity]
            if event.get("intensity_code") not in (None, expected_code):
                reasons.append("intensity_code_mismatch")

    window = record.get("window")
    segment = record.get("training_segment")
    if not isinstance(window, dict):
        reasons.append("missing_window")
        window = {}
    if not isinstance(segment, dict):
        reasons.append("missing_training_segment")
        segment = {}
    if window.get("selection_status") != SELECTION_STATUS:
        reasons.append("selection_status_mismatch")
    if segment.get("representation") != VARIABLE_REPRESENTATION:
        reasons.append("variable_length_representation_mismatch")
    if segment.get("fixed_window_sec") is not None:
        reasons.append("fixed_window_forbidden")
    boundary_source = segment.get("boundary_source")
    if (
        not isinstance(boundary_source, dict)
        or not str(boundary_source.get("mode") or "").strip()
    ):
        reasons.append("missing_semantic_boundary_source")

    start = window.get("start_frame")
    end = window.get("end_frame_exclusive")
    frames = window.get("frame_count")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or end - start < 3
        or frames != end - start
    ):
        reasons.append("invalid_variable_frame_bounds")
    elif any(
        segment.get(key) != expected
        for key, expected in (
            ("start_frame", start),
            ("end_frame_exclusive", end),
            ("frame_count", frames),
        )
    ):
        reasons.append("training_segment_window_mismatch")
    if not _finite_number(record.get("fps")) or not math.isclose(
        float(record.get("fps") or 0.0), 30.0, abs_tol=1e-6
    ):
        reasons.append("fps_not_30hz")

    metric_names = (
        "interaction_energy_mean_rad_s",
        "interaction_energy_p95_rad_s",
        "active_frame_fraction",
    )
    if any(not _finite_number(window.get(name)) for name in metric_names):
        reasons.append("invalid_interaction_energy_metrics")
    else:
        mean = float(window["interaction_energy_mean_rad_s"])
        p95 = float(window["interaction_energy_p95_rad_s"])
        active = float(window["active_frame_fraction"])
        if mean < min_energy_mean:
            reasons.append("static_below_interaction_energy_floor")
        if p95 > max_energy_p95:
            reasons.append("high_dynamic_above_interaction_energy_ceiling")
        if mean < 0 or p95 < 0 or not 0 <= active <= 1:
            reasons.append("interaction_energy_metric_out_of_range")
    if not _valid_sha256(record.get("motion_sha256")):
        reasons.append("missing_or_invalid_motion_sha256")
    return sorted(set(reasons))


def target_speaker_counts(
    speaker_count: int,
    fractions: dict[str, float],
) -> dict[str, int]:
    if speaker_count < len(SPLITS):
        raise ValueError("Strict train/validation/test needs at least three speakers")
    if set(fractions) != set(SPLITS) or not math.isclose(
        sum(float(fractions[name]) for name in SPLITS), 1.0, abs_tol=1e-9
    ):
        raise ValueError(f"Split fractions must define {SPLITS} and sum to one")
    raw = {name: float(fractions[name]) * speaker_count for name in SPLITS}
    counts = {name: max(1, int(math.floor(raw[name]))) for name in SPLITS}
    while sum(counts.values()) > speaker_count:
        choices = [name for name in SPLITS if counts[name] > 1]
        if not choices:
            raise ValueError("Cannot allocate at least one speaker to every split")
        name = max(choices, key=lambda item: (counts[item] - raw[item], counts[item]))
        counts[name] -= 1
    while sum(counts.values()) < speaker_count:
        name = max(SPLITS, key=lambda item: (raw[item] - counts[item], -SPLITS.index(item)))
        counts[name] += 1
    return counts


def _assignment_from_order(
    order: list[str], target_counts: dict[str, int]
) -> dict[str, str]:
    assignment = {}
    cursor = 0
    for split in SPLITS:
        stop = cursor + target_counts[split]
        assignment.update({speaker: split for speaker in order[cursor:stop]})
        cursor = stop
    if cursor != len(order):
        raise AssertionError("Speaker assignment did not consume every speaker")
    return assignment


def _assignment_score(
    assignment: dict[str, str],
    profiles: dict[str, Counter],
    preferred_profiles: dict[str, Counter],
    quotas: dict[str, int],
) -> tuple[Any, ...]:
    available = {split: Counter() for split in SPLITS}
    preferred = {split: Counter() for split in SPLITS}
    for speaker, split in assignment.items():
        available[split].update(profiles[speaker])
        preferred[split].update(preferred_profiles[speaker])
    deficits = {}
    preferred_deficits = {}
    for split in SPLITS:
        deficits[split] = sum(
            max(0, quotas[split] - available[split][stratum])
            for stratum in all_strata()
        )
        preferred_deficits[split] = sum(
            max(0, quotas[split] - preferred[split][stratum])
            for stratum in all_strata()
        )
    assignment_hash = sha256_bytes(stable_json(assignment).encode("utf-8"))
    return (
        sum(deficits.values()),
        max(deficits.values()),
        deficits["validation"] + deficits["test"],
        sum(preferred_deficits.values()),
        preferred_deficits["validation"] + preferred_deficits["test"],
        assignment_hash,
    )


def assign_speakers(
    records: list[dict[str, Any]],
    *,
    seed: int,
    trials: int,
    fractions: dict[str, float],
    quotas: dict[str, int],
) -> tuple[dict[str, str], dict[str, Any]]:
    speakers = sorted({str(record["speaker_key"]) for record in records})
    target_counts = target_speaker_counts(len(speakers), fractions)
    profiles = {speaker: Counter() for speaker in speakers}
    preferred = {speaker: Counter() for speaker in speakers}
    for record in records:
        speaker = str(record["speaker_key"])
        stratum = semantic_stratum(record)
        profiles[speaker][stratum] += 1
        if record["pilot_dynamic_band"] in {"low", "medium"}:
            preferred[speaker][stratum] += 1

    rng = random.Random(int(seed))
    orders = [
        sorted(
            speakers,
            key=lambda speaker: sha256_bytes(
                f"{int(seed)}:speaker:{speaker}".encode("utf-8")
            ),
        )
    ]
    for _ in range(int(trials)):
        order = list(speakers)
        rng.shuffle(order)
        orders.append(order)

    best_assignment = None
    best_score = None
    for order in orders:
        assignment = _assignment_from_order(order, target_counts)
        score = _assignment_score(assignment, profiles, preferred, quotas)
        if best_score is None or score < best_score:
            best_assignment, best_score = assignment, score
    assert best_assignment is not None and best_score is not None

    # Deterministic pair-swap refinement cannot change speaker counts.
    improved = True
    while improved:
        improved = False
        for first_index, first in enumerate(speakers):
            for second in speakers[first_index + 1 :]:
                if best_assignment[first] == best_assignment[second]:
                    continue
                candidate = dict(best_assignment)
                candidate[first], candidate[second] = candidate[second], candidate[first]
                score = _assignment_score(candidate, profiles, preferred, quotas)
                if score[:-1] < best_score[:-1]:
                    best_assignment, best_score = candidate, score
                    improved = True
        # Each full pass improves a finite integer deficit metric; equal-quality
        # assignments retain the deterministic multistart winner.

    return best_assignment, {
        "policy": "seeded_multistart_stratum_coverage_then_pair_swap_v1",
        "seed": int(seed),
        "random_assignment_trials": int(trials),
        "speaker_target_counts": target_counts,
        "score": {
            "total_quota_deficit": best_score[0],
            "maximum_split_quota_deficit": best_score[1],
            "evaluation_quota_deficit": best_score[2],
            "preferred_dynamic_quota_deficit": best_score[3],
            "preferred_dynamic_evaluation_quota_deficit": best_score[4],
        },
        "speaker_to_split": dict(sorted(best_assignment.items())),
        "sha256": sha256_bytes(stable_json(best_assignment).encode("utf-8")),
    }


def _event_priority(record: dict[str, Any], seed: int) -> tuple[Any, ...]:
    window = record["window"]
    event = record["semantic_event"]
    speech_context = str(record.get("window_transcript_context") or "").strip()
    lexical_anchor = str(event.get("source_lexical_anchor") or "").strip()
    source_score = event.get("source_score")
    score = float(source_score) if _finite_number(source_score) else 0.0
    tie = sha256_bytes(f"{int(seed)}:{record['clip_id']}".encode("utf-8"))
    return (
        DYNAMIC_BAND_ORDER[record["pilot_dynamic_band"]],
        0 if speech_context else 1,
        0 if lexical_anchor else 1,
        -float(window["active_frame_fraction"]),
        -score,
        float(window["interaction_energy_p95_rad_s"]),
        tie,
    )


def select_balanced_events(
    candidates: list[dict[str, Any]],
    assignment: dict[str, str],
    *,
    quotas: dict[str, int],
    max_events_per_source: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    buckets: dict[tuple[str, tuple[str, str, str]], list[dict[str, Any]]] = {
        (split, stratum): [] for split in SPLITS for stratum in all_strata()
    }
    for record in candidates:
        split = assignment[str(record["speaker_key"])]
        record["pilot_split"] = split
        record["fixed_split_assignment"] = split
        buckets[(split, semantic_stratum(record))].append(record)
    for bucket in buckets.values():
        bucket.sort(key=lambda record: _event_priority(record, seed))
        for rank, record in enumerate(bucket, 1):
            record["pilot_candidate_rank_within_split_stratum"] = rank

    selected_ids = set()
    selection_pass: dict[str, str] = {}
    source_uses = Counter()
    for quota_round in range(max(quotas.values())):
        for split in SPLITS:
            if quota_round >= quotas[split]:
                continue
            for stratum in all_strata():
                bucket = buckets[(split, stratum)]
                choice = next(
                    (
                        record
                        for record in bucket
                        if record["clip_id"] not in selected_ids
                        and source_uses[str(record["source_clip_id"])]
                        < max_events_per_source
                    ),
                    None,
                )
                if choice is None:
                    continue
                clip_id = str(choice["clip_id"])
                selected_ids.add(clip_id)
                source_uses[str(choice["source_clip_id"])] += 1
                selection_pass[clip_id] = "preferred_source_cap_respected"

    # Coverage is more important than the diversity cap. Relax only for a
    # stratum that is still below its explicit quota.
    selected_counts = Counter(
        (assignment[str(record["speaker_key"])], semantic_stratum(record))
        for record in candidates
        if record["clip_id"] in selected_ids
    )
    for split in SPLITS:
        for stratum in all_strata():
            while selected_counts[(split, stratum)] < quotas[split]:
                choice = next(
                    (
                        record
                        for record in buckets[(split, stratum)]
                        if record["clip_id"] not in selected_ids
                    ),
                    None,
                )
                if choice is None:
                    break
                clip_id = str(choice["clip_id"])
                selected_ids.add(clip_id)
                source_uses[str(choice["source_clip_id"])] += 1
                selection_pass[clip_id] = "source_cap_relaxed_for_stratum_coverage"
                selected_counts[(split, stratum)] += 1

    selected = [record for record in candidates if record["clip_id"] in selected_ids]
    selected.sort(
        key=lambda record: (
            SPLITS.index(record["pilot_split"]),
            semantic_stratum(record),
            record["pilot_candidate_rank_within_split_stratum"],
            record["clip_id"],
        )
    )
    return selected, selection_pass


def _group_hashes(
    records: list[dict[str, Any]], group_field: str
) -> dict[str, str]:
    grouped = defaultdict(list)
    for record in records:
        grouped[str(record[group_field])].append(record["inventory_record_sha256"])
    return {
        key: sha256_bytes(stable_json(sorted(values)).encode("utf-8"))
        for key, values in grouped.items()
    }


def _count_by(records: Iterable[dict[str, Any]], field_fn) -> dict[str, int]:
    return dict(sorted(Counter(str(field_fn(record)) for record in records).items()))


def build_pilot(
    input_path: Path,
    output_dir: Path,
    *,
    seed: int = 20260723,
    assignment_trials: int = 4096,
    quotas: dict[str, int] | None = None,
    split_fractions: dict[str, float] | None = None,
    max_events_per_source: int = 3,
    min_energy_mean: float = 0.02,
    low_energy_p95: float = 1.0,
    medium_energy_p95: float = 3.0,
    max_energy_p95: float = 4.0,
) -> dict[str, Any]:
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    quotas = dict(quotas or {name: 1 for name in SPLITS})
    fractions = dict(split_fractions or DEFAULT_SPLIT_FRACTIONS)
    if set(quotas) != set(SPLITS) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in quotas.values()
    ):
        raise ValueError(f"Quotas must define positive integers for {SPLITS}")
    if assignment_trials < 0:
        raise ValueError("assignment_trials cannot be negative")
    if max_events_per_source < 1:
        raise ValueError("max_events_per_source must be positive")
    thresholds = (
        min_energy_mean,
        low_energy_p95,
        medium_energy_p95,
        max_energy_p95,
    )
    if not all(math.isfinite(float(value)) for value in thresholds) or not (
        0 <= min_energy_mean <= low_energy_p95 <= medium_energy_p95 <= max_energy_p95
    ):
        raise ValueError("Interaction energy thresholds are invalid or unordered")

    source_manifest_sha = sha256_file(input_path)
    raw_records = load_jsonl(input_path)
    script_sha = sha256_file(Path(__file__).resolve())
    policy = {
        "schema_version": SCHEMA_VERSION,
        "seed": int(seed),
        "assignment_trials": int(assignment_trials),
        "quotas_per_emotion_category_intensity": quotas,
        "split_fractions_at_speaker_granularity": fractions,
        "max_events_per_source": int(max_events_per_source),
        "dynamic_thresholds_rad_s": {
            "minimum_mean": float(min_energy_mean),
            "low_p95_max": float(low_energy_p95),
            "medium_p95_max": float(medium_energy_p95),
            "hard_p95_max": float(max_energy_p95),
        },
        "fixed_window_sec": None,
        "duration_policy": "native_semantic_event_boundary_no_fixed_or_max_duration",
        "audio_enabled": False,
        "qwen_prompt": {
            "schema_version": PROMPT_SCHEMA_VERSION,
            "source": PROMPT_SOURCE,
            "lexical_anchor_role": (
                "speech_context_only_not_motion_instruction_or_emotion_label"
            ),
            "lexical_anchor_included_in_canonical_prompt": False,
            "emotion_source": OFFICIAL_EMOTION_SOURCE,
            "canonical_prompt_role": "coarse_category_only",
            "semantic_supervision_masks": {
                "official_category": False,
                "robot_observable_motion_form": False,
                "communicative_intent": False,
                "prompt_text": False,
                "legacy_gesture": False,
            },
        },
        "official_category_usage": {
            "status": "official_category_verified_metadata_split_eval_only",
            "verified": True,
            "role": "verified_metadata_split_and_evaluation_only",
            "conditioning_enabled": False,
            "condition_channel": None,
            "loss": None,
        },
        "official_emotion_usage": {
            "source_emotion_label_verified": True,
            "supervision_role": "disabled_pending_robot_affect_review",
            "conditioning_enabled": False,
            "condition_channel": None,
            "loss": None,
            "affect_observable_review_status": "candidate_unreviewed",
            "affect_observable_supervision_mask": False,
        },
        "behavior_supervision": {
            "source": BEHAVIOR_MAPPING_SOURCE,
            "revision": BEHAVIOR_MAPPING_REVISION,
            "scope": "beat2_human_co_speech_interaction_dataset_scope",
            "mask": False,
            "official_beat2_behavior_annotation": False,
        },
        "official_split_role": "provenance_only_not_pilot_partition",
        "split_grouping": "speaker_strict_source_inherits_speaker_split",
        "selector_script_sha256": script_sha,
        "source_manifest_sha256": source_manifest_sha,
    }
    policy_sha = sha256_bytes(stable_json(policy).encode("utf-8"))

    rejected_invalid: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    clip_id_counts = Counter(
        str(record.get("clip_id") or "") for record in raw_records
    )
    for raw in raw_records:
        record = dict(raw)
        line_number = record.pop("_selector_input_line")
        input_record_sha = record_sha256(record)
        reasons = validate_candidate(
            record,
            min_energy_mean=min_energy_mean,
            max_energy_p95=max_energy_p95,
        )
        clip_id = str(record.get("clip_id") or "")
        if clip_id_counts[clip_id] != 1:
            reasons.append("duplicate_clip_id")
        common = {
            **record,
            "inventory_record_sha256": input_record_sha,
            "upstream_inventory_record_sha256": input_record_sha,
            "inventory_record_sha256_role": (
                "upstream_beat2_semantic_event_inventory_canonical_row"
            ),
            "inventory_manifest_sha256": source_manifest_sha,
            "upstream_inventory_manifest_sha256": source_manifest_sha,
            "inventory_manifest_sha256_role": (
                "upstream_beat2_semantic_event_inventory_manifest"
            ),
            "pilot_selector_contract_sha256": policy_sha,
            "accepted_for_training": False,
        }
        if reasons:
            rejected_invalid.append(
                {
                    **common,
                    "pilot_selector_input_line": line_number,
                    "pilot_selection_status": "rejected_contract",
                    "pilot_rejection_reasons": sorted(set(reasons)),
                }
            )
            continue
        common["pilot_dynamic_band"] = dynamic_band(
            common, low_p95=low_energy_p95, medium_p95=medium_energy_p95
        )
        common.update(training_semantic_provenance(common))
        common["pilot_stratum"] = {
            "emotion_id": semantic_stratum(common)[0],
            "semantic_category": semantic_stratum(common)[1],
            "semantic_intensity": semantic_stratum(common)[2],
        }
        valid.append(common)

    source_speakers = defaultdict(set)
    for record in valid:
        source_speakers[str(record["source_clip_id"])].add(str(record["speaker_key"]))
    conflicting_sources = {
        source for source, speakers in source_speakers.items() if len(speakers) != 1
    }
    if conflicting_sources:
        retained = []
        for record in valid:
            if str(record["source_clip_id"]) in conflicting_sources:
                rejected_invalid.append(
                    {
                        **record,
                        "pilot_selection_status": "rejected_contract",
                        "pilot_rejection_reasons": [
                            "source_clip_maps_to_multiple_speakers"
                        ],
                    }
                )
            else:
                retained.append(record)
        valid = retained
    if not valid:
        raise ValueError("No contract-valid semantic-event pilot candidates")

    source_hashes = _group_hashes(valid, "source_clip_id")
    speaker_hashes = _group_hashes(valid, "speaker_key")
    for record in valid:
        record["pilot_source_group_sha256"] = source_hashes[
            str(record["source_clip_id"])
        ]
        record["pilot_speaker_group_sha256"] = speaker_hashes[
            str(record["speaker_key"])
        ]

    assignment, assignment_report = assign_speakers(
        valid,
        seed=seed,
        trials=assignment_trials,
        fractions=fractions,
        quotas=quotas,
    )
    selected, selection_pass = select_balanced_events(
        valid,
        assignment,
        quotas=quotas,
        max_events_per_source=max_events_per_source,
        seed=seed,
    )
    selected_ids = {str(record["clip_id"]) for record in selected}
    for record in valid:
        clip_id = str(record["clip_id"])
        if clip_id in selected_ids:
            record["pilot_selection_status"] = "selected_pending_retarget_qc"
            record["pilot_selection_pass"] = selection_pass[clip_id]
            record["training_admission_status"] = "pending_retarget_qc"
        else:
            record["pilot_selection_status"] = "not_selected"
            record["training_admission_status"] = "pilot_stratum_quota_filled"

    pool_contract = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "beat2_variable_length_low_medium_dynamic_training_pool",
        "source_manifest_sha256": source_manifest_sha,
        "pilot_selector_contract_sha256": policy_sha,
        "speaker_assignment_sha256": assignment_report["sha256"],
        "included_dynamic_bands": ["low", "medium"],
        "excluded_dynamic_bands": ["high_fallback"],
        "fixed_window_sec": None,
        "duration_policy": "native_semantic_event_boundary_no_fixed_or_max_duration",
        "split_policy": "speaker_strict_source_inherits_fixed_split_assignment",
        "test_split_role": "final_evaluation_only_not_model_selection",
        "accepted_for_training": False,
        "admission_requirement": "retarget_qc_then_independent_video_text_review",
        "official_category_usage": policy["official_category_usage"],
        "official_emotion_usage": policy["official_emotion_usage"],
        "semantic_supervision_masks": policy["qwen_prompt"][
            "semantic_supervision_masks"
        ],
    }
    pool_contract_sha = sha256_bytes(stable_json(pool_contract).encode("utf-8"))
    training_pool = [
        {
            **record,
            "training_pool_selection_status": "selected_pending_retarget_qc",
            "training_pool_contract_sha256": pool_contract_sha,
            "training_admission_status": "pending_retarget_qc",
            "accepted_for_training": False,
        }
        for record in valid
        if record["pilot_dynamic_band"] in {"low", "medium"}
    ]
    training_pool.sort(
        key=lambda record: (
            SPLITS.index(record["fixed_split_assignment"]),
            str(record["source_clip_id"]),
            int(record["window"]["start_frame"]),
            str(record["clip_id"]),
        )
    )

    candidates = sorted(valid, key=lambda record: str(record["clip_id"]))
    rejected_not_selected = [
        {
            **record,
            "pilot_selection_status": "rejected_not_selected",
            "pilot_rejection_reasons": ["split_stratum_quota_filled"],
        }
        for record in candidates
        if str(record["clip_id"]) not in selected_ids
    ]
    rejected = sorted(
        rejected_invalid + rejected_not_selected,
        key=lambda record: (
            str(record.get("clip_id") or ""),
            int(record.get("pilot_selector_input_line") or 0),
        ),
    )

    selected_counts = Counter(
        (record["pilot_split"], semantic_stratum(record)) for record in selected
    )
    missing = {
        split: [
            stratum_key(stratum)
            for stratum in all_strata()
            if selected_counts[(split, stratum)] < quotas[split]
        ]
        for split in SPLITS
    }
    complete = not any(missing.values())

    speakers_by_split = {
        split: sorted(speaker for speaker, value in assignment.items() if value == split)
        for split in SPLITS
    }
    selected_sources_by_split = {
        split: sorted(
            {
                str(record["source_clip_id"])
                for record in selected
                if record["pilot_split"] == split
            }
        )
        for split in SPLITS
    }
    leaked_speakers = sorted(
        speaker
        for speaker in assignment
        if sum(speaker in speakers_by_split[split] for split in SPLITS) != 1
    )
    source_selected_splits = defaultdict(set)
    for record in selected:
        source_selected_splits[str(record["source_clip_id"])].add(record["pilot_split"])
    leaked_sources = sorted(
        source for source, split_set in source_selected_splits.items() if len(split_set) != 1
    )
    if leaked_speakers or leaked_sources:
        raise AssertionError(
            f"Internal split leakage: speakers={leaked_speakers}, sources={leaked_sources}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / f"{OUTPUT_STEM}.candidates.jsonl"
    selected_path = output_dir / f"{OUTPUT_STEM}.selected.jsonl"
    rejected_path = output_dir / f"{OUTPUT_STEM}.rejected.jsonl"
    assignment_path = output_dir / f"{OUTPUT_STEM}.split_assignments.json"
    training_pool_path = output_dir / TRAINING_POOL_FILENAME
    output_hashes = {
        candidate_path.name: atomic_jsonl(candidate_path, candidates),
        selected_path.name: atomic_jsonl(selected_path, selected),
        rejected_path.name: atomic_jsonl(rejected_path, rejected),
        assignment_path.name: atomic_json(assignment_path, assignment_report),
        training_pool_path.name: atomic_jsonl(training_pool_path, training_pool),
    }
    for split in SPLITS:
        split_path = output_dir / f"{OUTPUT_STEM}.selected.{split}.jsonl"
        output_hashes[split_path.name] = atomic_jsonl(
            split_path,
            [record for record in selected if record["pilot_split"] == split],
        )
        pool_split_path = output_dir / f"training_pool_low_medium.{split}.jsonl"
        output_hashes[pool_split_path.name] = atomic_jsonl(
            pool_split_path,
            [
                record
                for record in training_pool
                if record["fixed_split_assignment"] == split
            ],
        )

    selected_frame_counts = [
        int(record["window"]["frame_count"]) for record in selected
    ]
    candidate_frame_counts = [
        int(record["window"]["frame_count"]) for record in candidates
    ]
    training_pool_frame_counts = [
        int(record["window"]["frame_count"]) for record in training_pool
    ]
    selected_frames = sum(selected_frame_counts)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "beat2_variable_length_semantic_event_balanced_pilot",
        "source_manifest": str(input_path),
        "source_manifest_sha256": source_manifest_sha,
        "selector_contract": policy,
        "selector_contract_sha256": policy_sha,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "rejected_count": len(rejected),
        "contract_rejected_count": len(rejected_invalid),
        "not_selected_count": len(rejected_not_selected),
        "selected_source_count": len(
            {str(record["source_clip_id"]) for record in selected}
        ),
        "selected_speaker_count": len(
            {str(record["speaker_key"]) for record in selected}
        ),
        "selected_frame_count": selected_frames,
        "selected_frame_count_min": min(selected_frame_counts, default=None),
        "selected_frame_count_max": max(selected_frame_counts, default=None),
        "selected_distinct_frame_count_count": len(set(selected_frame_counts)),
        "candidate_frame_count_min": min(candidate_frame_counts, default=None),
        "candidate_frame_count_max": max(candidate_frame_counts, default=None),
        "candidate_distinct_frame_count_count": len(set(candidate_frame_counts)),
        "selected_duration_hours_at_30hz": round(selected_frames / 30.0 / 3600.0, 8),
        "selection_complete": complete,
        "training_pool_contract": pool_contract,
        "training_pool_contract_sha256": pool_contract_sha,
        "training_pool_count": len(training_pool),
        "training_pool_source_count": len(
            {str(record["source_clip_id"]) for record in training_pool}
        ),
        "training_pool_speaker_count": len(
            {str(record["speaker_key"]) for record in training_pool}
        ),
        "training_pool_frame_count": sum(training_pool_frame_counts),
        "training_pool_duration_hours_at_30hz": round(
            sum(training_pool_frame_counts) / 30.0 / 3600.0, 8
        ),
        "training_pool_frame_count_min": min(
            training_pool_frame_counts, default=None
        ),
        "training_pool_frame_count_max": max(
            training_pool_frame_counts, default=None
        ),
        "training_pool_distinct_frame_count_count": len(
            set(training_pool_frame_counts)
        ),
        "training_pool_counts_by_split": _count_by(
            training_pool, lambda record: record["fixed_split_assignment"]
        ),
        "training_pool_counts_by_dynamic_band": _count_by(
            training_pool, lambda record: record["pilot_dynamic_band"]
        ),
        "training_pool_counts_by_emotion": _count_by(
            training_pool, lambda record: record["emotion_id"]
        ),
        "training_pool_counts_by_semantic_category": _count_by(
            training_pool, lambda record: record["semantic_event"]["category"]
        ),
        "training_pool_counts_by_semantic_intensity": _count_by(
            training_pool, lambda record: record["semantic_event"]["intensity"]
        ),
        "expected_stratum_count_per_split": len(all_strata()),
        "missing_strata_by_split": missing,
        "speaker_assignment": assignment_report,
        "strict_split_audit": {
            "speaker_disjoint": not leaked_speakers,
            "source_disjoint": not leaked_sources,
            "official_split_used_for_pilot_partition": False,
            "speakers_by_split": speakers_by_split,
            "selected_sources_by_split": selected_sources_by_split,
        },
        "candidate_counts_by_pilot_split": _count_by(
            candidates, lambda record: record["pilot_split"]
        ),
        "selected_counts_by_pilot_split": _count_by(
            selected, lambda record: record["pilot_split"]
        ),
        "selected_counts_by_emotion": _count_by(
            selected, lambda record: record["emotion_id"]
        ),
        "selected_counts_by_semantic_category": _count_by(
            selected, lambda record: record["semantic_event"]["category"]
        ),
        "selected_counts_by_semantic_intensity": _count_by(
            selected, lambda record: record["semantic_event"]["intensity"]
        ),
        "selected_counts_by_dynamic_band": _count_by(
            selected, lambda record: record["pilot_dynamic_band"]
        ),
        "selected_counts_by_split_stratum": {
            f"{split}|{stratum_key(stratum)}": selected_counts[(split, stratum)]
            for split in SPLITS
            for stratum in all_strata()
        },
        "rejection_counts_by_reason": dict(
            sorted(
                Counter(
                    reason
                    for record in rejected
                    for reason in record["pilot_rejection_reasons"]
                ).items()
            )
        ),
        "output_sha256": dict(sorted(output_hashes.items())),
    }
    summary_path = output_dir / f"{OUTPUT_STEM}.summary.json"
    summary["summary_sha256_without_self"] = sha256_bytes(
        stable_json(summary).encode("utf-8")
    )
    atomic_json(summary_path, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_pilot(
        args.input,
        args.output_dir,
        seed=args.seed,
        assignment_trials=args.assignment_trials,
        quotas={
            "train": args.train_per_stratum,
            "validation": args.validation_per_stratum,
            "test": args.test_per_stratum,
        },
        max_events_per_source=args.max_events_per_source,
        min_energy_mean=args.min_energy_mean,
        low_energy_p95=args.low_energy_p95,
        medium_energy_p95=args.medium_energy_p95,
        max_energy_p95=args.max_energy_p95,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["selection_complete"] or args.allow_incomplete else 2


if __name__ == "__main__":
    raise SystemExit(main())
