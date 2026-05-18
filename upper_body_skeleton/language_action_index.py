#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from upper_body_skeleton.retarget_v2 import JOINT_ORDER


WINDOW_SEC = 4.0
STRIDE_SEC = 2.0


REFUSAL_HINTS = ("no", "not", "don't", "do not", "refuse", "stop", "uncomfortable")
HELP_HINTS = ("help", "please", "need", "assist")
WARNING_HINTS = ("careful", "warning", "danger", "watch out")
GREETING_HINTS = ("hello", "hi", "hey", "welcome", "good morning")
UNCERTAIN_HINTS = ("unsure", "uncertain", "hesitant", "guarded", "nervous", "uncomfortable", "difficult")
FRIENDLY_HINTS = ("happy", "friendly", "smile", "excited", "glad", "welcome")
SAD_HINTS = ("sad", "sorry", "grief", "upset", "grandmother")
ANGRY_HINTS = ("angry", "mad", "frustrated")


def read_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_metadata(json_path):
    return json.loads(Path(json_path).read_text(encoding="utf-8"))


def source_json_path(row):
    npz_path = Path(row["npz_path"])
    return npz_path.with_suffix(".json")


def transcript_for_window(metadata, start_sec, end_sec):
    parts = []
    for item in metadata.get("metadata:transcript", []):
        if float(item.get("end", 0.0)) < start_sec or float(item.get("start", 0.0)) > end_sec:
            continue
        text = str(item.get("transcript", "")).strip()
        if text:
            parts.append(text)
    return " ".join(parts)


def annotations_for_window(metadata, start_sec, end_sec):
    out = {}
    for key, value in metadata.items():
        if not key.startswith("annotations:"):
            continue
        selected = []
        for item in value:
            item_start = float(item.get("start_ts", 0.0))
            item_end = float(item.get("end_ts", 0.0))
            if item_end < start_sec or item_start > end_sec:
                continue
            text = str(item.get("annotation", "")).strip()
            if text:
                selected.append(text)
        if selected:
            out[key] = selected
    return out


def safe_nanmean(values):
    if len(values) == 0:
        return None
    return float(np.nanmean(values))


def safe_token_mode(values):
    flat = np.asarray(values).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if len(flat) == 0:
        return None
    tokens, counts = np.unique(flat.astype(int), return_counts=True)
    return int(tokens[int(np.argmax(counts))])


def safe_vector_mean(values):
    arr = np.asarray(values)
    if arr.size == 0:
        return None
    mean = np.nanmean(arr, axis=0)
    return [float(x) for x in np.asarray(mean).reshape(-1)]


def npz_affect_stats(npz_path, start_sec, end_sec, fps=30.0):
    start = max(0, int(round(start_sec * fps)))
    end = max(start + 1, int(round(end_sec * fps)))
    stats = {
        "arousal": None,
        "valence": None,
        "arousal_token": None,
        "valence_token": None,
        "label_sources": [],
    }
    with np.load(npz_path, allow_pickle=True) as data:
        if "movement:emotion_arousal" in data:
            values = data["movement:emotion_arousal"][start:end]
            stats["arousal"] = safe_nanmean(values)
            if stats["arousal"] is not None:
                stats["label_sources"].append("movement:emotion_arousal")
        if "movement:emotion_valence" in data:
            values = data["movement:emotion_valence"][start:end]
            stats["valence"] = safe_nanmean(values)
            if stats["valence"] is not None:
                stats["label_sources"].append("movement:emotion_valence")
        if "movement:EmotionArousalToken" in data:
            stats["arousal_token"] = safe_token_mode(data["movement:EmotionArousalToken"][start:end])
            if stats["arousal_token"] is not None:
                stats["label_sources"].append("movement:EmotionArousalToken")
        if "movement:EmotionValenceToken" in data:
            stats["valence_token"] = safe_token_mode(data["movement:EmotionValenceToken"][start:end])
            if stats["valence_token"] is not None:
                stats["label_sources"].append("movement:EmotionValenceToken")
    return stats


def csv_row_count(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


def motion_style_from_csv(csv_path, start_row, end_row):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for index, row in enumerate(csv.DictReader(f)):
            if index < start_row:
                continue
            if index >= end_row:
                break
            rows.append([float(row[joint]) for joint in JOINT_ORDER])
    if len(rows) < 2:
        return "restrained", 0.0
    arr = np.asarray(rows, dtype=float)
    velocity = np.abs(np.diff(arr, axis=0))
    energy = float(np.nanmean(velocity))
    if energy > 0.06:
        return "energetic", energy
    if energy > 0.025:
        return "relaxed", energy
    return "restrained", energy


def contains_any(text, hints):
    text = text.lower()
    for hint in hints:
        pattern = r"(?<![a-z0-9_])" + re.escape(hint.lower()) + r"(?![a-z0-9_])"
        if re.search(pattern, text):
            return True
    return False


def infer_intent_label(transcript, action_text, intent_text, rationale_text):
    combined = " ".join([transcript, action_text, intent_text, rationale_text]).lower()
    if contains_any(combined, REFUSAL_HINTS):
        return "refusing"
    if contains_any(combined, HELP_HINTS):
        return "requesting_help"
    if contains_any(combined, WARNING_HINTS):
        return "warning"
    if contains_any(combined, GREETING_HINTS):
        return "greeting"
    if transcript or action_text or intent_text:
        return "explaining"
    return "waiting"


def infer_observed_affect(internal_state_text, transcript, affect_stats=None):
    combined = " ".join([internal_state_text, transcript]).lower()
    if contains_any(combined, ANGRY_HINTS):
        return "angry_like"
    if contains_any(combined, SAD_HINTS):
        return "sad_like"
    if contains_any(combined, FRIENDLY_HINTS):
        return "friendly"
    if contains_any(combined, UNCERTAIN_HINTS):
        return "uncertain"
    if affect_stats:
        arousal = affect_stats.get("arousal")
        valence = affect_stats.get("valence")
        if arousal is not None and valence is not None:
            if arousal > 0.55 and valence > 0.15:
                return "excited"
            if arousal > 0.55 and valence < -0.15:
                return "nervous"
            if valence < -0.35:
                return "sad_like"
            if valence > 0.25:
                return "friendly"
    return "neutral" if internal_state_text else "low_confidence_unknown"


def semantic_gesture_keyword(action_text):
    text = action_text.lower()
    if "cross" in text or "fold" in text:
        return "crossed_arms"
    if "point" in text:
        return "pointing"
    if "wave" in text:
        return "waving"
    if "shrug" in text:
        return "shrugging"
    if "arms" in text or "hands" in text:
        return "upper_body_gesture"
    return "null"


def gesture_function_from_semantics(intent_label, semantic_gesture, action_text):
    text = " ".join([intent_label, semantic_gesture, action_text]).lower()
    if any(word in text for word in ("hello", "hi", "welcome", "greeting", "wave")):
        return "social"
    if any(word in text for word in ("refus", "guard", "cross", "fold", "protect")):
        return "self_protective"
    if any(word in text for word in ("warn", "careful", "stop", "danger")):
        return "regulatory"
    if any(word in text for word in ("point", "show", "indicate", "explain", "describe")):
        return "representational"
    if semantic_gesture in {"upper_body_gesture", "shrugging"}:
        return "beat_or_emphasis"
    return "none"


def affect_bucket(observed_affect, arousal, valence):
    if observed_affect and observed_affect != "low_confidence_unknown":
        return observed_affect
    if arousal is None or valence is None:
        return "neutral"
    if arousal > 0.55 and valence < -0.15:
        return "nervous"
    if arousal > 0.55 and valence > 0.15:
        return "excited"
    if valence < -0.35:
        return "sad_like"
    if valence > 0.25:
        return "friendly"
    return "neutral"


def motion_control_codes(intent_label, observed_affect, semantic_gesture, action_text, affect_stats, motion_style, motion_energy, duration_sec):
    arousal = affect_stats.get("arousal") if affect_stats else None
    valence = affect_stats.get("valence") if affect_stats else None
    emotion = affect_bucket(observed_affect, arousal, valence)
    if arousal is not None:
        intensity = "high" if arousal > 0.62 else "medium" if arousal > 0.38 else "low"
    else:
        intensity = "high" if motion_energy > 0.06 else "medium" if motion_energy > 0.025 else "low"
    speed = "fast" if motion_energy > 0.06 else "medium" if motion_energy > 0.025 else "slow"
    text = action_text.lower()
    closed_hint = any(word in text for word in ("cross", "fold", "guard", "tight", "chest", "protect"))
    open_hint = any(word in text for word in ("open", "wide", "wave", "welcome", "point"))
    openness = "closed" if closed_hint or semantic_gesture == "crossed_arms" else "open" if open_hint else "neutral"
    tense_affects = {"nervous", "angry_like", "uncertain"}
    tension = "high" if emotion in tense_affects or closed_hint else "medium" if intensity == "medium" else "low"
    transition = "end" if duration_sec <= WINDOW_SEC + 1e-6 else "continue"
    return {
        "communicative_intent": intent_label,
        "gesture_function": gesture_function_from_semantics(intent_label, semantic_gesture, action_text),
        "emotion_trajectory": f"{emotion}_sustained",
        "intensity": intensity,
        "speed": speed,
        "openness": openness,
        "tension": tension,
        "duration_sec": float(duration_sec),
        "transition": transition,
        "semantic_confidence": 0.9 if action_text and action_text != "upper-body conversational gesture" else 0.45,
    }


def choose_intent_and_descriptions(transcript, annotations, affect_stats=None):
    action_text = " ".join(annotations.get("annotations:3P-V", []))
    mood_text = " ".join(annotations.get("annotations:1P-IS", []) + annotations.get("annotations:3P-IS", []))
    intent_text = " ".join(annotations.get("annotations:1P-R", []) + annotations.get("annotations:3P-R", []))
    rationale_text = intent_text
    scenario = transcript[:400]
    if not action_text:
        action_text = "upper-body conversational gesture"
    if not intent_text:
        intent_text = "conversational expression"
    if not mood_text:
        mood_text = "observable affect unknown"
    if not rationale_text:
        rationale_text = "rationale unknown"
    intent_label = infer_intent_label(transcript, action_text, intent_text, rationale_text)
    observed_affect = infer_observed_affect(mood_text, transcript, affect_stats=affect_stats)
    return {
        "scenario_description": scenario,
        "action_description": action_text,
        "intent_text": intent_text,
        "mood_text": mood_text,
        "rationale_text": rationale_text,
        "intent_label": intent_label,
        "observed_affect": observed_affect,
        "meta_semantics": {
            "visual_element": action_text,
            "internal_state": mood_text,
            "rationale": rationale_text,
            "semantic_gesture": semantic_gesture_keyword(action_text),
            "annotation_views": {
                "first_person_internal_state": annotations.get("annotations:1P-IS", []),
                "third_person_internal_state": annotations.get("annotations:3P-IS", []),
                "third_person_visual": annotations.get("annotations:3P-V", []),
                "first_person_rationale": annotations.get("annotations:1P-R", []),
                "third_person_rationale": annotations.get("annotations:3P-R", []),
            },
        },
    }


def windows_for_row(row, window_sec=WINDOW_SEC, stride_sec=STRIDE_SEC, fps=30.0):
    frame_count = int(float(row.get("frame_count") or 0))
    total_sec = frame_count / fps
    if frame_count <= 0:
        return []
    if total_sec <= window_sec:
        return [(0.0, total_sec)]
    windows = []
    start = 0.0
    while start + window_sec <= total_sec:
        windows.append((start, start + window_sec))
        start += stride_sec
    return windows


def build_records(manifest_path, output_jsonl, max_records=None, window_sec=WINDOW_SEC, stride_sec=STRIDE_SEC, fps=30.0):
    rows = [row for row in read_manifest(manifest_path) if row.get("status") in {"processed", "skipped_existing"}]
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_jsonl.open("w", encoding="utf-8") as out:
        for row in rows:
            json_path = source_json_path(row)
            if not json_path.exists() or not Path(row["joint_csv"]).exists():
                continue
            metadata = load_metadata(json_path)
            row_count = csv_row_count(row["joint_csv"])
            for start_sec, end_sec in windows_for_row(row, window_sec=window_sec, stride_sec=stride_sec, fps=fps):
                start_row = max(0, int(round(start_sec * fps)))
                end_row = min(row_count, int(round(end_sec * fps)))
                if end_row <= start_row:
                    continue
                transcript = transcript_for_window(metadata, start_sec, end_sec)
                annotations = annotations_for_window(metadata, start_sec, end_sec)
                affect = npz_affect_stats(row["npz_path"], start_sec, end_sec, fps=fps)
                motion_style, motion_energy = motion_style_from_csv(row["joint_csv"], start_row, end_row)
                text = choose_intent_and_descriptions(transcript, annotations, affect_stats=affect)
                semantic_gesture = text["meta_semantics"]["semantic_gesture"]
                duration_sec = float(end_sec - start_sec)
                control_codes = motion_control_codes(
                    text["intent_label"],
                    text["observed_affect"],
                    semantic_gesture,
                    text["action_description"],
                    affect,
                    motion_style,
                    motion_energy,
                    duration_sec,
                )
                sample_id = f"{row['sample']}__{start_row}_{end_row}".replace("/", "__")
                record = {
                    "sample_id": sample_id,
                    "source": {
                        "dataset": "seamless_interaction_50g",
                        "video_path": row["video_path"],
                        "json_path": str(json_path),
                        "npz_path": row["npz_path"],
                        "retarget_csv_path": row["joint_csv"],
                        "monitor_report_path": row["monitor_json"],
                    },
                    "time_window": {"start_sec": start_sec, "end_sec": end_sec, "fps": fps},
                    "language_condition": {
                        "raw_transcript": transcript,
                        "scenario_description": text["scenario_description"],
                        "action_description": text["action_description"],
                        "intent_text": text["intent_text"],
                        "mood_text": text["mood_text"],
                        "rationale_text": text["rationale_text"],
                        "instruction_variants": [
                            f"{text['intent_text']}; gesture style: {motion_style}; affect: {text['mood_text']}; rationale: {text['rationale_text']}",
                            f"Perform {text['meta_semantics']['semantic_gesture']} with {motion_style} style while expressing {text['observed_affect']}",
                        ],
                    },
                    "labels": {
                        "intent": text["intent_label"],
                        "communicative_intent": control_codes["communicative_intent"],
                        "observed_affect": text["observed_affect"],
                        "gesture_function": control_codes["gesture_function"],
                        "emotion_trajectory": control_codes["emotion_trajectory"],
                        "motion_style": motion_style,
                        "intensity": control_codes["intensity"],
                        "speed": control_codes["speed"],
                        "openness": control_codes["openness"],
                        "tension": control_codes["tension"],
                        "duration_sec": control_codes["duration_sec"],
                        "transition": control_codes["transition"],
                        "semantic_confidence": control_codes["semantic_confidence"],
                        "arousal": affect["arousal"],
                        "valence": affect["valence"],
                        "arousal_token": affect["arousal_token"],
                        "valence_token": affect["valence_token"],
                        "motion_energy": motion_energy,
                        "label_sources": sorted(set(affect["label_sources"] + list(annotations.keys()))),
                    },
                    "meta_semantics": {
                        **text["meta_semantics"],
                        "affect_codebook": {
                            "arousal_token": affect["arousal_token"],
                            "valence_token": affect["valence_token"],
                        },
                        "body_motion": {
                            "joint_space": "v2_upper_body_15d",
                            "action_shape": [end_row - start_row, len(JOINT_ORDER)],
                            "joint_order": JOINT_ORDER,
                            "motion_style": motion_style,
                            "motion_energy": motion_energy,
                            "semantic_gesture": text["meta_semantics"]["semantic_gesture"],
                        },
                        "body_expression": {
                            **control_codes,
                            "semantic_gesture": semantic_gesture,
                            "no_face_or_head": True,
                        },
                    },
                    "action": {
                        "retarget_csv_path": row["joint_csv"],
                        "start_row": start_row,
                        "end_row": end_row,
                        "fps": fps,
                        "joint_order": JOINT_ORDER,
                    },
                    "quality": {
                        "accepted_for_training": not row.get("status", "").startswith("error"),
                        "frame_count": int(float(row.get("frame_count") or 0)),
                        "flagged_frame_count": int(float(row.get("flagged_frame_count") or 0)),
                        "max_elbow_overfold": float(row.get("max_elbow_overfold") or 0.0),
                        "max_yaw_under_response": float(row.get("max_yaw_under_response") or 0.0),
                    },
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                if max_records is not None and count >= max_records:
                    return count
    return count


def main():
    parser = argparse.ArgumentParser(description="Build language-action JSONL index from retarget manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--window-sec", type=float, default=WINDOW_SEC)
    parser.add_argument("--stride-sec", type=float, default=STRIDE_SEC)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    count = build_records(
        args.manifest,
        args.output_jsonl,
        max_records=args.max_records,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        fps=args.fps,
    )
    print(json.dumps({"records": count, "output_jsonl": args.output_jsonl}, indent=2))


if __name__ == "__main__":
    main()
