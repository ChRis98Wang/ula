#!/usr/bin/env python3
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


KIMODO_BEHAVIOR_IDS = [
    "Behavior.IdleLowPower",
    "Behavior.IdleQuiet",
    "Behavior.IdleAttentive",
    "Behavior.InteractPresence",
    "Behavior.GreetingOwner01",
    "Behavior.GreetingOwner02",
    "Behavior.GreetingOwner03",
    "Behavior.GreetingOwner04",
    "Behavior.GreetingGuest",
    "Behavior.Farewell",
    "Behavior.Joy",
    "Behavior.Aversion",
    "Behavior.CuriousLook",
    "Behavior.Comfort",
    "Behavior.Alert",
    "Behavior.SeekAttention",
    "Behavior.ActiveListening",
    "Behavior.Disappointment",
    "Behavior.Withdrawal",
    "Behavior.Hesitate",
    "Behavior.Search",
    "Behavior.Error",
    "Behavior.Dance.Base",
    "Behavior.Dance.Sway",
    "Behavior.Dance.Accent",
    "Behavior.Dance.Stop",
    "Behavior.FingerHeart",
]

KIMODO_EMOTION_IDS = ["neutral", "sad", "happy", "angry", "surprise", "fear"]
KIMODO_BEHAVIOR_FAMILIES = [
    "idle",
    "greeting",
    "farewell",
    "comfort",
    "alert",
    "withdrawal",
    "dance",
    "attention_search_error",
]
KIMODO_CONDITION_SCHEMA_VERSION = 2.0
KIMODO_CONDITION_CONTRACT_VERSION = 1
KIMODO_CONDITION_EXTRA_DIM = len(KIMODO_BEHAVIOR_IDS) + len(KIMODO_EMOTION_IDS) + len(KIMODO_BEHAVIOR_FAMILIES) + 3


@dataclass(frozen=True)
class KimodoPromptRecord:
    behavior_id: str
    emotion_id: str
    emotion_zh_label: str
    prompt: str
    negative_prompt: str
    output_name: str
    output_format: str
    requires_bvh_without_t_pose: bool


def load_kimodo_prompt_index(csv_path):
    index = {}
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            record = KimodoPromptRecord(
                behavior_id=row["behavior_id"],
                emotion_id=row["emotion_id"],
                emotion_zh_label=row.get("emotion_zh_label", ""),
                prompt=row.get("prompt", ""),
                negative_prompt=row.get("negative_prompt", ""),
                output_name=row.get("output_name", ""),
                output_format=row.get("output_format", ""),
                requires_bvh_without_t_pose=str(row.get("requires_bvh_without_t_pose", "")).lower() == "true",
            )
            index[(record.behavior_id, record.emotion_id)] = record
    return index


def _one_hot(label, labels):
    vec = np.zeros(len(labels), dtype=np.float32)
    if label in labels:
        vec[labels.index(label)] = 1.0
    return vec


def kimodo_behavior_family(behavior_id):
    if behavior_id in {"Behavior.IdleLowPower", "Behavior.IdleQuiet", "Behavior.IdleAttentive", "Behavior.InteractPresence", "Behavior.ActiveListening"}:
        return "idle"
    if behavior_id.startswith("Behavior.Greeting") or behavior_id == "Behavior.FingerHeart":
        return "greeting"
    if behavior_id == "Behavior.Farewell":
        return "farewell"
    if behavior_id in {"Behavior.Comfort", "Behavior.Joy", "Behavior.CuriousLook"}:
        return "comfort"
    if behavior_id == "Behavior.Alert":
        return "alert"
    if behavior_id in {"Behavior.Aversion", "Behavior.Disappointment", "Behavior.Withdrawal", "Behavior.Hesitate"}:
        return "withdrawal"
    if behavior_id.startswith("Behavior.Dance"):
        return "dance"
    if behavior_id in {"Behavior.SeekAttention", "Behavior.Search", "Behavior.Error"}:
        return "attention_search_error"
    return "idle"


def infer_kimodo_ids_from_text(text):
    lowered = str(text).lower()
    emotion_id = "neutral"
    emotion_hints = [
        ("happy", ("happy", "joy", "开心", "高兴", "快乐", "兴奋")),
        ("sad", ("sad", "disappoint", "悲伤", "难过", "失望")),
        ("angry", ("angry", "mad", "愤怒", "生气")),
        ("surprise", ("surprise", "惊讶", "吃惊")),
        ("fear", ("fear", "afraid", "scared", "恐惧", "害怕", "紧张")),
    ]
    for label, hints in emotion_hints:
        if any(hint in lowered for hint in hints):
            emotion_id = label
            break

    behavior_id = "Behavior.IdleQuiet"
    behavior_hints = [
        ("Behavior.Alert", ("alert", "warning", "stop", "警告", "停止", "风险", "危险")),
        ("Behavior.FingerHeart", ("finger heart", "比心", "爱心")),
        ("Behavior.Farewell", ("farewell", "goodbye", "再见", "告别")),
        ("Behavior.Dance.Sway", ("dance", "sway", "跳舞", "摇摆")),
        ("Behavior.Search", ("search", "look for", "寻找", "搜索")),
        ("Behavior.Comfort", ("comfort", "安慰", "温柔")),
        ("Behavior.CuriousLook", ("curious", "好奇")),
        ("Behavior.SeekAttention", ("attention", "注意", "关注")),
        ("Behavior.GreetingOwner01", ("greet", "hello", "wave", "主人", "打招呼", "挥手", "你好")),
        ("Behavior.Withdrawal", ("withdraw", "退缩", "回避")),
        ("Behavior.Hesitate", ("hesitate", "犹豫")),
        ("Behavior.Error", ("confusion", "error", "困惑", "错误")),
    ]
    for label, hints in behavior_hints:
        if any(hint in lowered for hint in hints):
            behavior_id = label
            break
    return behavior_id, emotion_id


def build_kimodo_condition_extra(behavior_id=None, emotion_id=None, prompt=""):
    if behavior_id is None or emotion_id is None:
        inferred_behavior, inferred_emotion = infer_kimodo_ids_from_text(prompt)
        behavior_id = behavior_id or inferred_behavior
        emotion_id = emotion_id or inferred_emotion
    if behavior_id not in KIMODO_BEHAVIOR_IDS:
        raise ValueError(f"unknown Kimodo behavior_id: {behavior_id}")
    if emotion_id not in KIMODO_EMOTION_IDS:
        raise ValueError(f"unknown Kimodo emotion_id: {emotion_id}")
    family = kimodo_behavior_family(behavior_id)
    controls = np.asarray([KIMODO_CONDITION_SCHEMA_VERSION, 1.0, 0.0], dtype=np.float32)
    return np.concatenate(
        [
            _one_hot(behavior_id, KIMODO_BEHAVIOR_IDS),
            _one_hot(emotion_id, KIMODO_EMOTION_IDS),
            _one_hot(family, KIMODO_BEHAVIOR_FAMILIES),
            controls,
        ],
        axis=0,
    ).astype(np.float32)


def kimodo_condition_vectors_sha256(vectors):
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("Kimodo canonical condition vectors must be a rank-3 array")
    if values.shape[:2] != (len(KIMODO_BEHAVIOR_IDS), len(KIMODO_EMOTION_IDS)):
        raise ValueError("Kimodo canonical condition vector label dimensions do not match")
    if not np.isfinite(values).all():
        raise ValueError("Kimodo canonical condition vectors must be finite")
    header = {
        "contract_version": KIMODO_CONDITION_CONTRACT_VERSION,
        "condition_schema_version": KIMODO_CONDITION_SCHEMA_VERSION,
        "behavior_ids": KIMODO_BEHAVIOR_IDS,
        "emotion_ids": KIMODO_EMOTION_IDS,
        "shape": list(values.shape),
        "dtype": "float32-le",
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(header, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(values.astype("<f4", copy=False)).tobytes())
    return digest.hexdigest()


def kimodo_condition_metadata(behavior_id=None, emotion_id=None, text=""):
    inferred_behavior, inferred_emotion = infer_kimodo_ids_from_text(text)
    behavior_id = behavior_id or inferred_behavior
    emotion_id = emotion_id or inferred_emotion
    return {
        "behavior_id": behavior_id,
        "emotion_id": emotion_id,
        "behavior_family": kimodo_behavior_family(behavior_id),
        "upper_body_only": True,
        "finger_detail_available": False,
    }
