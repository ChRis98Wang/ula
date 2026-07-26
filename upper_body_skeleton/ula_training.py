#!/usr/bin/env python3
import argparse
from contextlib import nullcontext
import csv
import hashlib
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn

from upper_body_skeleton.kimodo_semantics import (
    KIMODO_BEHAVIOR_IDS,
    KIMODO_CONDITION_EXTRA_DIM,
    KIMODO_CONDITION_CONTRACT_VERSION,
    KIMODO_CONDITION_SCHEMA_VERSION,
    KIMODO_EMOTION_IDS,
    build_kimodo_condition_extra,
    kimodo_condition_vectors_sha256,
)
from upper_body_skeleton.retarget_v2 import JOINT_LIMITS, JOINT_ORDER
from upper_body_skeleton.retarget_v2_18d import (
    JOINT_LIMITS_18D,
    joint_order_for_action_dim,
)


INTENT_IDS = {"waiting": 0, "explaining": 1, "refusing": 2, "requesting_help": 3, "greeting": 4, "warning": 5}
AFFECT_IDS = {"low_confidence_unknown": 0, "neutral": 1, "sad_like": 2, "nervous": 3, "friendly": 4, "uncertain": 5, "angry_like": 6, "excited": 7}
STYLE_IDS = {"restrained": 0, "relaxed": 1, "energetic": 2}
GESTURE_IDS = {"null": 0, "upper_body_gesture": 1, "pointing": 2, "crossed_arms": 3, "shrugging": 4, "waving": 5}
BASE_CONDITION_DIM = len(INTENT_IDS) + len(AFFECT_IDS) + len(STYLE_IDS) + len(GESTURE_IDS) + 5
TEXT_EMBED_DIM = 64
LEGACY_CONDITION_DIM = BASE_CONDITION_DIM + TEXT_EMBED_DIM
KIMODO_CONDITION_DIM = LEGACY_CONDITION_DIM + KIMODO_CONDITION_EXTRA_DIM
KIMODO_MOTION_LATENT_DIM = 128
KIMODO_V2_CONDITION_DIM = KIMODO_CONDITION_DIM + KIMODO_MOTION_LATENT_DIM
TRANSITION_IDS = {"continue": 0, "emotion_change": 1, "action_change": 2, "end": 3}
ULA_FM_LEGACY_ARCHITECTURE = "ula_fm_legacy"
ULA_MMDIT_LITE_ARCHITECTURE = "ula_mmdit_lite"
ULA_MMDIT_V2_ARCHITECTURE = "ula_mmdit_v2"
ULA_ADALN_LITE_ARCHITECTURE = "ula_adaln_lite"


INTENT_KEYWORDS = [
    ("warning", ("warning", "warn", "stop", "danger", "careful", "警告", "小心", "停止", "危险")),
    ("requesting_help", ("help", "assist", "please", "request", "求助", "帮忙", "请求", "请")),
    ("greeting", ("hello", "hi", "greet", "wave", "你好", "打招呼", "挥手")),
    ("refusing", ("refuse", "reject", "no ", "don't", "cannot", "拒绝", "不要", "不行")),
    ("explaining", ("explain", "tell", "describe", "conversational", "解释", "说明", "表达", "讲")),
]
AFFECT_KEYWORDS = [
    ("excited", ("excited", "energetic", "happy", "eager", "兴奋", "激动", "高兴")),
    ("angry_like", ("angry", "frustrated", "annoyed", "生气", "愤怒", "不满")),
    ("nervous", ("nervous", "tense", "anxious", "紧张", "焦虑")),
    ("uncertain", ("uncertain", "unsure", "hesitant", "不确定", "犹豫")),
    ("sad_like", ("sad", "down", "upset", "难过", "低落")),
    ("friendly", ("friendly", "warm", "kind", "友好", "亲切")),
]
STYLE_KEYWORDS = [
    ("energetic", ("energetic", "large", "fast", "big", "active", "用力", "大幅度", "快速")),
    ("relaxed", ("relaxed", "soft", "calm", "loose", "放松", "柔和", "平静")),
    ("restrained", ("restrained", "small", "subtle", "reserved", "克制", "小幅度", "保守")),
]
GESTURE_KEYWORDS = [
    ("crossed_arms", ("cross", "fold arms", "arms crossed", "交叉", "抱臂")),
    ("pointing", ("point", "pointing", "指", "指向")),
    ("shrugging", ("shrug", "shrugging", "耸肩")),
    ("waving", ("wave", "waving", "挥手", "摆手")),
    ("upper_body_gesture", ("gesture", "body", "upper body", "手势", "肢体", "上肢")),
]
AFFECT_DEFAULTS = {
    "low_confidence_unknown": (0.0, 0.0, 0, 0),
    "neutral": (0.0, 0.0, 0, 0),
    "sad_like": (-0.25, -0.45, 0, 0),
    "nervous": (0.45, -0.2, 2, 0),
    "friendly": (0.25, 0.45, 1, 2),
    "uncertain": (0.2, -0.15, 1, 0),
    "angry_like": (0.65, -0.55, 3, 0),
    "excited": (0.75, 0.45, 3, 2),
}


def one_hot(index, size):
    vec = np.zeros(size, dtype=np.float32)
    vec[max(0, min(size - 1, int(index)))] = 1.0
    return vec


def _first_keyword_label(text, choices, default):
    lowered = f" {text.lower()} "
    for label, keywords in choices:
        if any(keyword in lowered for keyword in keywords):
            return label
    return default


def infer_codes_from_text(text):
    return {
        "intent": _first_keyword_label(text, INTENT_KEYWORDS, "explaining"),
        "observed_affect": _first_keyword_label(text, AFFECT_KEYWORDS, "neutral"),
        "motion_style": _first_keyword_label(text, STYLE_KEYWORDS, "restrained"),
        "semantic_gesture": _first_keyword_label(text, GESTURE_KEYWORDS, "upper_body_gesture"),
    }


def frozen_text_embedding(text, dim=TEXT_EMBED_DIM):
    """Small deterministic frozen text encoder used until a larger LM is wired in."""
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
    if not tokens:
        return vec
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
        for offset in range(0, len(digest), 2):
            index = digest[offset] % dim
            sign = 1.0 if digest[offset + 1] % 2 == 0 else -1.0
            vec[index] += sign
    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        vec /= norm
    return vec


def _semantic_text(meta_row):
    parts = [
        meta_row.get("language_instruction", ""),
        meta_row.get("raw_transcript", ""),
        meta_row.get("scenario_description", ""),
        meta_row.get("action_description", ""),
        meta_row.get("intent_text", ""),
        meta_row.get("mood_text", ""),
        meta_row.get("rationale_text", ""),
    ]
    return " ".join(str(part) for part in parts if part)


def condition_vector(meta_row):
    text = _semantic_text(meta_row)
    labels = []
    labels.append(one_hot(INTENT_IDS.get(meta_row.get("intent", ""), 0), len(INTENT_IDS)))
    labels.append(one_hot(AFFECT_IDS.get(meta_row.get("observed_affect", ""), 0), len(AFFECT_IDS)))
    labels.append(one_hot(STYLE_IDS.get(meta_row.get("motion_style", ""), 0), len(STYLE_IDS)))
    labels.append(one_hot(GESTURE_IDS.get(meta_row.get("semantic_gesture", ""), 0), len(GESTURE_IDS)))
    scalars = np.asarray(
        [
            meta_row.get("arousal") if meta_row.get("arousal") is not None else 0.0,
            meta_row.get("valence") if meta_row.get("valence") is not None else 0.0,
            meta_row.get("motion_energy") if meta_row.get("motion_energy") is not None else 0.0,
            meta_row.get("arousal_token") if meta_row.get("arousal_token") is not None else 0.0,
            meta_row.get("valence_token") if meta_row.get("valence_token") is not None else 0.0,
        ],
        dtype=np.float32,
    )
    legacy = np.concatenate([*labels, scalars, frozen_text_embedding(text)], axis=0)
    kimodo_extra = build_kimodo_condition_extra(
        behavior_id=meta_row.get("behavior_id") or None,
        emotion_id=meta_row.get("emotion_id") or None,
        prompt=text,
    )
    return np.concatenate([legacy, kimodo_extra], axis=0)


def build_condition_from_text(
    text,
    *,
    intent=None,
    affect=None,
    style=None,
    gesture=None,
    arousal=None,
    valence=None,
    arousal_token=None,
    valence_token=None,
    motion_energy=0.05,
    text_dim=TEXT_EMBED_DIM,
    behavior_id=None,
    emotion_id=None,
    condition_dim=KIMODO_CONDITION_DIM,
):
    codes = infer_codes_from_text(text)
    if intent is not None:
        codes["intent"] = intent
    if affect is not None:
        codes["observed_affect"] = affect
    if style is not None:
        codes["motion_style"] = style
    if gesture is not None:
        codes["semantic_gesture"] = gesture
    default_arousal, default_valence, default_arousal_token, default_valence_token = AFFECT_DEFAULTS.get(
        codes["observed_affect"], AFFECT_DEFAULTS["neutral"]
    )
    meta_row = {
        "language_instruction": text,
        "intent": codes["intent"],
        "observed_affect": codes["observed_affect"],
        "motion_style": codes["motion_style"],
        "semantic_gesture": codes["semantic_gesture"],
        "arousal": default_arousal if arousal is None else arousal,
        "valence": default_valence if valence is None else valence,
        "motion_energy": motion_energy,
        "arousal_token": default_arousal_token if arousal_token is None else arousal_token,
        "valence_token": default_valence_token if valence_token is None else valence_token,
        "behavior_id": behavior_id,
        "emotion_id": emotion_id,
    }
    if condition_dim == LEGACY_CONDITION_DIM and text_dim == TEXT_EMBED_DIM:
        return np.concatenate(
            [
                condition_vector(meta_row)[:BASE_CONDITION_DIM],
                frozen_text_embedding(text),
            ],
            axis=0,
        )
    base = condition_vector(meta_row)
    if text_dim != TEXT_EMBED_DIM:
        base = np.concatenate([base[:BASE_CONDITION_DIM], frozen_text_embedding(text, dim=text_dim), base[LEGACY_CONDITION_DIM:]], axis=0)
    if condition_dim is not None and base.shape[0] != int(condition_dim):
        if int(condition_dim) == LEGACY_CONDITION_DIM:
            return base[:LEGACY_CONDITION_DIM]
        raise ValueError(f"condition dim mismatch: built {base.shape[0]}, requested {condition_dim}")
    return base


def load_lerobot_episodes(dataset_dir, max_episodes=None):
    dataset_dir = Path(dataset_dir)
    semantic_path = dataset_dir / "meta" / "semantic_index.parquet"
    episode_meta = {row["episode_index"]: row for row in pq.read_table(semantic_path).to_pylist()}
    grouped = {}
    episodes = []

    def flush_complete(episode_index):
        frame_rows = sorted(grouped.pop(episode_index), key=lambda row: row["frame_index"])
        actions = np.asarray([row["observation.state"] for row in frame_rows], dtype=np.float32)
        meta = episode_meta.get(episode_index, {})
        episodes.append(
            {
                "episode_index": episode_index,
                "actions": actions,
                "condition": condition_vector(meta),
                "meta": meta,
                "task_index": frame_rows[0]["task_index"] if frame_rows else 0,
                "duration_sec": float(
                    max(0, len(actions) - 1)
                    / max(1e-6, float(meta.get("fps") or 30.0))
                ),
                "fps": float(meta.get("fps") or 30.0),
                "transition_id": TRANSITION_IDS["end"],
            }
        )

    for path in sorted((dataset_dir / "data").glob("chunk-*/*.parquet")):
        for row in pq.read_table(path).to_pylist():
            episode_index = row["episode_index"]
            grouped.setdefault(episode_index, []).append(row)
            if row.get("next.done"):
                flush_complete(episode_index)
                if max_episodes is not None and len(episodes) >= max_episodes:
                    return episodes
    for episode_index in sorted(grouped):
        if max_episodes is not None and len(episodes) >= max_episodes:
            break
        flush_complete(episode_index)
    return episodes


def build_kimodo_condition_contract(episodes, dataset_dir):
    episodes = list(episodes)
    if not episodes or np.asarray(episodes[0]["condition"]).shape != (KIMODO_CONDITION_DIM,):
        return None
    behavior_index = {label: index for index, label in enumerate(KIMODO_BEHAVIOR_IDS)}
    emotion_index = {label: index for index, label in enumerate(KIMODO_EMOTION_IDS)}
    vectors = np.full(
        (len(KIMODO_BEHAVIOR_IDS), len(KIMODO_EMOTION_IDS), KIMODO_CONDITION_DIM),
        np.nan,
        dtype=np.float32,
    )
    seen = set()
    for episode in episodes:
        meta = episode.get("meta") or {}
        behavior_id = meta.get("behavior_id")
        emotion_id = meta.get("emotion_id")
        if behavior_id not in behavior_index or emotion_id not in emotion_index:
            continue
        index = behavior_index[behavior_id], emotion_index[emotion_id]
        vector = np.asarray(episode["condition"], dtype=np.float32)
        if vector.shape != (KIMODO_CONDITION_DIM,) or not np.isfinite(vector).all():
            raise ValueError(f"invalid Kimodo condition for episode {episode.get('episode_index')}")
        if index in seen and not np.array_equal(vectors[index], vector):
            raise ValueError(f"inconsistent Kimodo condition for {behavior_id}/{emotion_id}")
        vectors[index] = vector
        seen.add(index)
    expected_count = len(KIMODO_BEHAVIOR_IDS) * len(KIMODO_EMOTION_IDS)
    if len(seen) != expected_count:
        return None
    semantic_path = Path(dataset_dir) / "meta" / "semantic_index.parquet"
    if not semantic_path.is_file():
        raise FileNotFoundError(f"Kimodo semantic index not found for checkpoint contract: {semantic_path}")
    return {
        "contract_version": KIMODO_CONDITION_CONTRACT_VERSION,
        "condition_schema_version": KIMODO_CONDITION_SCHEMA_VERSION,
        "condition_dim": KIMODO_CONDITION_DIM,
        "behavior_ids": list(KIMODO_BEHAVIOR_IDS),
        "emotion_ids": list(KIMODO_EMOTION_IDS),
        "source_semantic_index_sha256": hashlib.sha256(semantic_path.read_bytes()).hexdigest(),
        "canonical_vectors_sha256": kimodo_condition_vectors_sha256(vectors),
    }


def compute_action_normalization_stats(episodes, eps=1e-4):
    actions = np.concatenate([episode["actions"] for episode in episodes], axis=0).astype(np.float32)
    mean = torch.tensor(actions.mean(axis=0), dtype=torch.float32)
    std = torch.tensor(actions.std(axis=0), dtype=torch.float32).clamp_min(float(eps))
    return {"mean": mean, "std": std}


def _stats_tensor(stats, key, *, device, action_dim):
    value = stats[key]
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    return tensor[:action_dim].reshape(*([1] * 2), action_dim)


def normalize_action_tensor(actions, stats):
    mean = _stats_tensor(stats, "mean", device=actions.device, action_dim=actions.shape[-1])
    std = _stats_tensor(stats, "std", device=actions.device, action_dim=actions.shape[-1])
    return (actions - mean) / std


def denormalize_action_tensor(actions, stats):
    mean = _stats_tensor(stats, "mean", device=actions.device, action_dim=actions.shape[-1])
    std = _stats_tensor(stats, "std", device=actions.device, action_dim=actions.shape[-1])
    return actions * std + mean


def normalize_episode_actions(episodes, stats):
    normalized = []
    mean = stats["mean"].detach().cpu().numpy()
    std = stats["std"].detach().cpu().numpy()
    for episode in episodes:
        item = dict(episode)
        item["actions"] = ((episode["actions"] - mean) / std).astype(np.float32)
        normalized.append(item)
    return normalized


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half, device=t.device))
        angles = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class SinusoidalFrameEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def embed_positions(self, positions):
        half = self.dim // 2
        if half == 0:
            return torch.zeros(
                (*positions.shape, self.dim),
                dtype=torch.float32,
                device=positions.device,
            )
        freqs = torch.exp(
            torch.linspace(
                math.log(1.0), math.log(1000.0), half, device=positions.device
            )
        )
        angles = positions[..., None] * freqs
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
        return emb

    def forward(self, frame_count, device):
        positions = torch.linspace(0.0, 1.0, int(frame_count), device=device)
        return self.embed_positions(positions)


def stable_sdpa_context(tensor):
    if tensor.device.type != "cuda" or not hasattr(torch.nn, "attention"):
        return nullcontext()
    return torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH])


class UlaFmModel(nn.Module):
    def __init__(self, action_dim=15, condition_dim=BASE_CONDITION_DIM + TEXT_EMBED_DIM, hidden_dim=256, layers=4):
        super().__init__()
        self.architecture = ULA_FM_LEGACY_ARCHITECTURE
        self.action_dim = action_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.input = nn.Linear(action_dim, hidden_dim)
        self.time = SinusoidalTimeEmbedding(hidden_dim)
        self.frame = SinusoidalFrameEmbedding(hidden_dim)
        self.cond = nn.Sequential(nn.Linear(condition_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.plan = nn.Sequential(nn.Linear(condition_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.duration_head = nn.Linear(hidden_dim, 1)
        self.transition_head = nn.Linear(hidden_dim, len(TRANSITION_IDS))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x_t, t, condition):
        h = self.input(x_t)
        cond = self.cond(condition)[:, None, :]
        time = self.time(t)[:, None, :]
        frame = self.frame(x_t.shape[1], x_t.device)[None, :, :]
        h = h + cond + time + frame
        with stable_sdpa_context(h):
            h = self.blocks(h)
        return self.output(h)

    def plan_condition(self, condition):
        if condition.ndim == 1:
            condition = condition[None, :]
        h = self.plan(condition)
        duration_sec = torch.nn.functional.softplus(self.duration_head(h).squeeze(-1)) + 0.25
        return {"duration_sec": duration_sec, "transition_logits": self.transition_head(h)}


class UlaMMDiTLiteModel(nn.Module):
    def __init__(self, action_dim=15, condition_dim=KIMODO_CONDITION_DIM, hidden_dim=256, layers=4, semantic_tokens=4):
        super().__init__()
        self.architecture = ULA_MMDIT_LITE_ARCHITECTURE
        self.action_dim = action_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.semantic_tokens = int(semantic_tokens)
        if self.semantic_tokens <= 0:
            raise ValueError("semantic_tokens must be positive")
        self.input = nn.Linear(action_dim, hidden_dim)
        self.time = SinusoidalTimeEmbedding(hidden_dim)
        self.frame = SinusoidalFrameEmbedding(hidden_dim)
        self.condition_tokens = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim * self.semantic_tokens),
            nn.SiLU(),
            nn.Linear(hidden_dim * self.semantic_tokens, hidden_dim * self.semantic_tokens),
        )
        self.plan = nn.Sequential(nn.Linear(condition_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.duration_head = nn.Linear(hidden_dim, 1)
        self.transition_head = nn.Linear(hidden_dim, len(TRANSITION_IDS))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.output = nn.Linear(hidden_dim, action_dim)
        self.last_joint_sequence_shape = None

    def semantic_condition_tokens(self, condition):
        tokens = self.condition_tokens(condition)
        return tokens.reshape(condition.shape[0], self.semantic_tokens, self.hidden_dim)

    def forward(self, x_t, t, condition):
        motion = self.input(x_t)
        time = self.time(t)[:, None, :]
        frame = self.frame(x_t.shape[1], x_t.device)[None, :, :]
        motion = motion + time + frame
        semantic = self.semantic_condition_tokens(condition)
        h = torch.cat([semantic, motion], dim=1)
        self.last_joint_sequence_shape = tuple(h.shape)
        with stable_sdpa_context(h):
            h = self.blocks(h)
        motion_h = h[:, self.semantic_tokens :, :]
        return self.output(motion_h)

    def plan_condition(self, condition):
        if condition.ndim == 1:
            condition = condition[None, :]
        h = self.plan(condition)
        duration_sec = torch.nn.functional.softplus(self.duration_head(h).squeeze(-1)) + 0.25
        return {"duration_sec": duration_sec, "transition_logits": self.transition_head(h)}


class UlaMMDiTV2Model(nn.Module):
    """Structured motion transformer for semantic, style, and motion-latent conditioning."""

    def __init__(
        self,
        action_dim=15,
        condition_dim=KIMODO_V2_CONDITION_DIM,
        hidden_dim=384,
        layers=6,
        semantic_tokens=7,
    ):
        super().__init__()
        if int(condition_dim) != KIMODO_V2_CONDITION_DIM:
            raise ValueError(
                f"{ULA_MMDIT_V2_ARCHITECTURE} requires condition_dim={KIMODO_V2_CONDITION_DIM}"
            )
        self.architecture = ULA_MMDIT_V2_ARCHITECTURE
        self.action_dim = int(action_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)
        self.semantic_tokens = int(semantic_tokens)
        self.motion_latent_tokens = self.semantic_tokens - 5
        if self.motion_latent_tokens <= 0:
            raise ValueError("ula_mmdit_v2 semantic_tokens must be at least 6")

        self.input = nn.Linear(self.action_dim, self.hidden_dim)
        self.time = SinusoidalTimeEmbedding(self.hidden_dim)
        self.frame = SinusoidalFrameEmbedding(self.hidden_dim)

        def projection(input_dim, token_count=1):
            output_dim = self.hidden_dim * int(token_count)
            return nn.Sequential(
                nn.Linear(int(input_dim), output_dim),
                nn.SiLU(),
                nn.Linear(output_dim, output_dim),
            )

        behavior_dim = len(KIMODO_BEHAVIOR_IDS)
        emotion_dim = len(KIMODO_EMOTION_IDS)
        family_dim = KIMODO_CONDITION_EXTRA_DIM - behavior_dim - emotion_dim - 3
        self.legacy_condition = projection(LEGACY_CONDITION_DIM)
        self.behavior_condition = projection(behavior_dim)
        self.emotion_condition = projection(emotion_dim)
        self.family_condition = projection(family_dim)
        self.style_condition = projection(3)
        self.motion_latent_condition = projection(
            KIMODO_MOTION_LATENT_DIM,
            token_count=self.motion_latent_tokens,
        )

        self.plan = nn.Sequential(
            nn.Linear(self.condition_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.duration_head = nn.Linear(self.hidden_dim, 1)
        self.transition_head = nn.Linear(self.hidden_dim, len(TRANSITION_IDS))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=self.hidden_dim * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=self.layers)
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, self.action_dim)
        self.last_joint_sequence_shape = None

    def semantic_condition_tokens(self, condition):
        behavior_start = LEGACY_CONDITION_DIM
        emotion_start = behavior_start + len(KIMODO_BEHAVIOR_IDS)
        family_start = emotion_start + len(KIMODO_EMOTION_IDS)
        style_start = KIMODO_CONDITION_DIM - 3
        latent_start = KIMODO_CONDITION_DIM
        tokens = [
            self.legacy_condition(condition[:, :behavior_start])[:, None, :],
            self.behavior_condition(condition[:, behavior_start:emotion_start])[:, None, :],
            self.emotion_condition(condition[:, emotion_start:family_start])[:, None, :],
            self.family_condition(condition[:, family_start:style_start])[:, None, :],
            self.style_condition(condition[:, style_start:latent_start])[:, None, :],
        ]
        latent = self.motion_latent_condition(condition[:, latent_start:])
        tokens.append(latent.reshape(condition.shape[0], self.motion_latent_tokens, self.hidden_dim))
        return torch.cat(tokens, dim=1)

    def forward(self, x_t, t, condition):
        motion = self.input(x_t)
        motion = motion + self.time(t)[:, None, :]
        motion = motion + self.frame(x_t.shape[1], x_t.device)[None, :, :]
        semantic = self.semantic_condition_tokens(condition)
        h = torch.cat([semantic, motion], dim=1)
        self.last_joint_sequence_shape = tuple(h.shape)
        with stable_sdpa_context(h):
            h = self.blocks(h)
        return self.output(self.output_norm(h[:, self.semantic_tokens :, :]))

    def plan_condition(self, condition):
        if condition.ndim == 1:
            condition = condition[None, :]
        h = self.plan(condition)
        duration_sec = torch.nn.functional.softplus(self.duration_head(h).squeeze(-1)) + 0.25
        return {"duration_sec": duration_sec, "transition_logits": self.transition_head(h)}


class AdaLNTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, nhead=4):
        super().__init__()
        self.norm_attn = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_dim, nhead, batch_first=True)
        self.norm_ffn = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 6),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    @staticmethod
    def _modulate(x, shift, scale):
        return x * (1.0 + scale[:, None, :]) + shift[:, None, :]

    def forward(self, x, conditioning):
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = self.modulation(conditioning).chunk(6, dim=-1)
        attn_input = self._modulate(self.norm_attn(x), shift_attn, scale_attn)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + gate_attn[:, None, :] * attn_output
        ffn_input = self._modulate(self.norm_ffn(x), shift_ffn, scale_ffn)
        x = x + gate_ffn[:, None, :] * self.ffn(ffn_input)
        return x


class UlaAdaLNLiteModel(nn.Module):
    def __init__(self, action_dim=15, condition_dim=KIMODO_CONDITION_DIM, hidden_dim=256, layers=4):
        super().__init__()
        self.architecture = ULA_ADALN_LITE_ARCHITECTURE
        self.action_dim = action_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.input = nn.Linear(action_dim, hidden_dim)
        self.time = SinusoidalTimeEmbedding(hidden_dim)
        self.frame = SinusoidalFrameEmbedding(hidden_dim)
        self.condition = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.plan = nn.Sequential(nn.Linear(condition_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.duration_head = nn.Linear(hidden_dim, 1)
        self.transition_head = nn.Linear(hidden_dim, len(TRANSITION_IDS))
        self.blocks = nn.ModuleList([AdaLNTransformerBlock(hidden_dim, nhead=4) for _ in range(layers)])
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, action_dim)
        self.last_motion_sequence_shape = None

    def forward(self, x_t, t, condition):
        h = self.input(x_t)
        frame = self.frame(x_t.shape[1], x_t.device)[None, :, :]
        conditioning = self.condition(condition) + self.time(t)
        h = h + frame + conditioning[:, None, :]
        with stable_sdpa_context(h):
            for block in self.blocks:
                h = block(h, conditioning)
        self.last_motion_sequence_shape = tuple(h.shape)
        return self.output(self.output_norm(h))

    def plan_condition(self, condition):
        if condition.ndim == 1:
            condition = condition[None, :]
        h = self.plan(condition)
        duration_sec = torch.nn.functional.softplus(self.duration_head(h).squeeze(-1)) + 0.25
        return {"duration_sec": duration_sec, "transition_logits": self.transition_head(h)}


def create_ula_model(
    architecture=ULA_FM_LEGACY_ARCHITECTURE,
    *,
    action_dim=15,
    condition_dim=KIMODO_CONDITION_DIM,
    hidden_dim=256,
    layers=4,
    semantic_tokens=4,
):
    if architecture in (None, "", ULA_FM_LEGACY_ARCHITECTURE):
        return UlaFmModel(
            action_dim=action_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            layers=layers,
        )
    if architecture == ULA_MMDIT_LITE_ARCHITECTURE:
        return UlaMMDiTLiteModel(
            action_dim=action_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            layers=layers,
            semantic_tokens=semantic_tokens,
        )
    if architecture == ULA_MMDIT_V2_ARCHITECTURE:
        return UlaMMDiTV2Model(
            action_dim=action_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            layers=layers,
            semantic_tokens=semantic_tokens,
        )
    if architecture == ULA_ADALN_LITE_ARCHITECTURE:
        return UlaAdaLNLiteModel(
            action_dim=action_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            layers=layers,
        )
    raise ValueError(f"unknown ULA architecture: {architecture}")


def joint_limit_tensors(device, action_dim, pose_bounds=None):
    lowers = []
    uppers = []
    pose_bounds = pose_bounds or {}
    for joint in joint_order_for_action_dim(action_dim):
        lower, upper = JOINT_LIMITS_18D[joint]
        if joint in pose_bounds:
            pose_lower, pose_upper = pose_bounds[joint]
            lower = max(float(lower), float(pose_lower))
            upper = min(float(upper), float(pose_upper))
            if lower > upper:
                raise ValueError(f"pose bounds for {joint} do not overlap joint limits")
        lowers.append(lower)
        uppers.append(upper)
    return (
        torch.tensor(lowers, dtype=torch.float32, device=device)[None, None, :],
        torch.tensor(uppers, dtype=torch.float32, device=device)[None, None, :],
    )


def sample_trajectory(
    model,
    condition,
    frames=120,
    action_dim=15,
    steps=24,
    device="cpu",
    seed=None,
    pose_bounds=None,
    action_stats=None,
):
    if seed is not None:
        torch.manual_seed(int(seed))
    model.to(device)
    was_training = model.training
    model.eval()
    try:
        condition_tensor = torch.as_tensor(condition, dtype=torch.float32, device=device)
        if condition_tensor.ndim == 1:
            condition_tensor = condition_tensor[None, :]
        dt = 1.0 / float(max(1, steps))
        lower, upper = joint_limit_tensors(device, action_dim, pose_bounds=pose_bounds)
        if action_stats is None:
            action_stats = getattr(model, "action_stats", None)
        if action_stats is not None:
            lower = normalize_action_tensor(lower, action_stats)
            upper = normalize_action_tensor(upper, action_stats)
            lower, upper = torch.minimum(lower, upper), torch.maximum(lower, upper)
        x = torch.randn((condition_tensor.shape[0], frames, action_dim), dtype=torch.float32, device=device)
        x = torch.clamp(x, lower, upper)
        with torch.no_grad():
            for index in range(steps):
                t = torch.full((condition_tensor.shape[0],), index * dt, dtype=torch.float32, device=device)
                velocity = model(x, t, condition_tensor)
                x = torch.clamp(x + dt * velocity, lower, upper)
        if action_stats is not None:
            x = denormalize_action_tensor(x, action_stats)
        return x[0].detach().cpu().numpy()
    finally:
        if was_training:
            model.train()


def write_generated_csv(path, trajectory, fps=30.0):
    trajectory = np.asarray(trajectory)
    if trajectory.ndim != 2:
        raise ValueError("trajectory must have shape [frames, joints]")
    joint_order = joint_order_for_action_dim(trajectory.shape[1])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["time_sec"] + joint_order)
        writer.writeheader()
        for frame_index, values in enumerate(trajectory):
            row = {"time_sec": f"{frame_index / float(fps):.6f}"}
            row.update(
                {
                    joint: f"{float(values[i]):.6f}"
                    for i, joint in enumerate(joint_order)
                }
            )
            writer.writerow(row)


def write_training_preview(
    model,
    *,
    preview_root,
    step,
    text,
    frames,
    sampling_steps,
    fps,
    device,
    seed,
    width=1280,
    height=720,
    preview_mode="long",
    behavior_id=None,
    emotion_id=None,
    long_duration_sec=24.0,
    min_segment_sec=3.0,
    max_segment_sec=3.0,
    min_segments=8,
    max_segments=8,
    max_velocity_rad_s=3.0,
    smooth_window=5,
):
    preview_dir = Path(preview_root) / f"step_{int(step):06d}"
    preview_dir.mkdir(parents=True, exist_ok=True)

    if preview_mode == "long":
        from upper_body_skeleton.long_emotion_infer import generate_long_emotion_motion

        summary = generate_long_emotion_motion(
            model,
            text=text,
            behavior_id=behavior_id,
            emotion_id=emotion_id,
            output_dir=preview_dir,
            fps=fps,
            max_duration_sec=long_duration_sec,
            min_segment_sec=min_segment_sec,
            max_segment_sec=max_segment_sec,
            min_segments=min_segments,
            max_segments=max_segments,
            sampling_steps=sampling_steps,
            device=device,
            seed=seed,
            render=True,
            width=width,
            height=height,
            max_velocity_rad_s=max_velocity_rad_s,
            smooth_window=smooth_window,
        )
        summary.update(
            {
                "step": int(step),
                "preview_mode": "long",
                "generated_csv": summary["csv"],
                "generated_npz": summary["npz"],
                "preview_mp4": summary["rendered_mp4"],
                "sampling_steps": int(sampling_steps),
                "fps": float(fps),
            }
        )
        (preview_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    # Import lazily so training-only tests and environments do not need a renderer
    # until preview generation is explicitly requested.
    from upper_body_skeleton.long_emotion_infer import postprocess_trajectory, trajectory_quality
    from upper_body_skeleton.mujoco_playback import render_motion

    condition = build_condition_from_text(
        text,
        behavior_id=behavior_id,
        emotion_id=emotion_id,
        condition_dim=getattr(model, "condition_dim", KIMODO_CONDITION_DIM),
    )
    raw_trajectory = sample_trajectory(
        model,
        condition=condition,
        frames=frames,
        action_dim=len(JOINT_ORDER),
        steps=sampling_steps,
        device=device,
        seed=seed,
        action_stats=getattr(model, "action_stats", None),
    )
    trajectory = postprocess_trajectory(
        raw_trajectory,
        fps=fps,
        max_velocity_rad_s=max_velocity_rad_s,
        smooth_window=smooth_window,
    )
    csv_path = preview_dir / "generated.csv"
    npz_path = preview_dir / "generated.npz"
    mp4_path = preview_dir / "preview.mp4"
    summary_path = preview_dir / "summary.json"
    write_generated_csv(csv_path, trajectory, fps=fps)
    np.savez_compressed(
        npz_path,
        trajectory=trajectory.astype(np.float32),
        raw_trajectory=raw_trajectory.astype(np.float32),
        joint_order=np.asarray(JOINT_ORDER, dtype=object),
        text=text,
        fps=np.asarray(fps, dtype=np.float32),
        step=np.asarray(step, dtype=np.int64),
    )
    render_summary = render_motion(csv_path, mp4_path, fps=fps, width=width, height=height)
    summary = {
        "step": int(step),
        "text": text,
        "behavior_id": behavior_id,
        "emotion_id": emotion_id,
        "generated_csv": str(csv_path),
        "generated_npz": str(npz_path),
        "preview_mp4": str(mp4_path),
        "preview_mode": "chunk",
        "frames": int(trajectory.shape[0]),
        "sampling_steps": int(sampling_steps),
        "fps": float(fps),
        "render": render_summary,
        "postprocess": {
            "max_velocity_rad_s": float(max_velocity_rad_s),
            "smooth_window": int(smooth_window),
        },
        "trajectory_quality": {
            "raw": trajectory_quality(raw_trajectory, fps=fps),
            "processed": trajectory_quality(trajectory, fps=fps),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def sample_batch(episodes, batch_size, device):
    selected = random.choices(episodes, k=batch_size)
    actions = torch.tensor(np.stack([episode["actions"] for episode in selected]), dtype=torch.float32, device=device)
    condition = torch.tensor(np.stack([episode["condition"] for episode in selected]), dtype=torch.float32, device=device)
    duration = torch.tensor(
        [
            episode.get(
                "duration_sec",
                (episode["actions"].shape[0] - 1)
                / float(episode.get("fps") or 30.0),
            )
            for episode in selected
        ],
        dtype=torch.float32,
        device=device,
    )
    transition = torch.tensor([episode.get("transition_id", TRANSITION_IDS["end"]) for episode in selected], dtype=torch.long, device=device)
    return actions, condition, duration, transition


def flow_matching_loss(model, actions, condition):
    noise = torch.randn_like(actions)
    t = torch.rand(actions.shape[0], device=actions.device)
    x_t = (1.0 - t[:, None, None]) * noise + t[:, None, None] * actions
    target_v = actions - noise
    pred_v = model(x_t, t, condition)
    return torch.mean((pred_v - target_v) ** 2)


def planner_duration_loss(model, condition, duration_target):
    plan = model.plan_condition(condition)
    duration_target = duration_target.to(plan["duration_sec"].device, plan["duration_sec"].dtype)
    return torch.nn.functional.smooth_l1_loss(
        torch.log1p(plan["duration_sec"]), torch.log1p(duration_target)
    )


def planner_transition_loss(
    model, condition, transition_target, transition_supervision_mask=None
):
    plan = model.plan_condition(condition)
    transition_target = transition_target.to(plan["transition_logits"].device, torch.long)
    if transition_supervision_mask is None:
        transition_supervision_mask = torch.ones_like(
            transition_target, dtype=torch.bool
        )
    else:
        transition_supervision_mask = transition_supervision_mask.to(
            plan["transition_logits"].device, torch.bool
        )
    if transition_supervision_mask.shape != transition_target.shape:
        raise ValueError("transition supervision mask must match transition targets")
    if not torch.any(transition_supervision_mask):
        raise ValueError("transition loss requires at least one verified transition label")
    return torch.nn.functional.cross_entropy(
        plan["transition_logits"][transition_supervision_mask],
        transition_target[transition_supervision_mask],
    )


def planner_loss(
    model,
    condition,
    duration_target,
    transition_target=None,
    transition_supervision_mask=None,
):
    """Backward-compatible joint loss; transition supervision is optional."""
    duration = planner_duration_loss(model, condition, duration_target)
    if transition_target is None:
        return duration
    transition = planner_transition_loss(
        model,
        condition,
        transition_target,
        transition_supervision_mask=transition_supervision_mask,
    )
    return duration + 0.25 * transition


def clip_grad_norm_float64(parameters, max_norm):
    parameters = [parameter for parameter in parameters if parameter.grad is not None]
    max_norm = float(max_norm)
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    if not parameters:
        return 0.0
    squared_norm = torch.zeros((), dtype=torch.float64, device=parameters[0].grad.device)
    for parameter in parameters:
        gradient = parameter.grad.detach()
        squared_norm += gradient.double().square().sum()
    total_norm = squared_norm.sqrt()
    if not torch.isfinite(total_norm):
        raise FloatingPointError("non-finite global gradient norm encountered during training")
    scale = min(1.0, max_norm / (float(total_norm) + 1e-12))
    if scale < 1.0:
        for parameter in parameters:
            parameter.grad.mul_(scale)
    return float(total_norm)


def train_steps(
    model,
    episodes,
    steps,
    batch_size,
    lr,
    device,
    log_interval=0,
    progress_path=None,
    preview_interval=0,
    preview_root=None,
    preview_text=None,
    preview_frames=120,
    preview_sampling_steps=32,
    preview_fps=30.0,
    preview_seed=7,
    preview_width=1280,
    preview_height=720,
    preview_mode="long",
    preview_behavior_id=None,
    preview_emotion_id=None,
    preview_duration_sec=24.0,
    preview_min_segment_sec=3.0,
    preview_max_segment_sec=3.0,
    preview_min_segments=8,
    preview_max_segments=8,
    preview_max_velocity_rad_s=3.0,
    preview_smooth_window=5,
    checkpoint_dir=None,
    checkpoint_every_steps=0,
    checkpoint_payload_fn=None,
    save_best=False,
    weight_decay=0.01,
    adam_eps=1e-8,
    warmup_steps=0,
    max_grad_norm=1.0,
):
    lr = float(lr)
    weight_decay = float(weight_decay)
    adam_eps = float(adam_eps)
    warmup_steps = int(warmup_steps)
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative")
    if not math.isfinite(adam_eps) or adam_eps <= 0:
        raise ValueError("adam_eps must be finite and positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, eps=adam_eps)
    losses = []
    progress_path = Path(progress_path) if progress_path else None
    if progress_path:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    def write_checkpoint(path, step, loss_value):
        if checkpoint_payload_fn is None:
            return
        payload = checkpoint_payload_fn(model, step, loss_value)
        torch.save(payload, path)

    for step_index in range(steps):
        step = step_index + 1
        current_lr = lr if warmup_steps == 0 else lr * min(1.0, step / warmup_steps)
        for parameter_group in opt.param_groups:
            parameter_group["lr"] = current_lr
        actions, condition, durations, transitions = sample_batch(episodes, batch_size, device)
        loss = flow_matching_loss(model, actions, condition) + 0.05 * planner_loss(model, condition, durations, transitions)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at step {step_index + 1}: {float(loss.detach().cpu())}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = clip_grad_norm_float64(model.parameters(), max_grad_norm)
        opt.step()
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        if checkpoint_dir and save_best and loss_value < best_loss:
            best_loss = loss_value
            write_checkpoint(checkpoint_dir / "ula_fm_best_checkpoint.pt", step, loss_value)
        if checkpoint_dir and checkpoint_every_steps and (step % int(checkpoint_every_steps) == 0 or step == steps):
            write_checkpoint(checkpoint_dir / f"ula_fm_step_{step:06d}.pt", step, loss_value)
        if log_interval and (step == 1 or step % log_interval == 0 or step == steps):
            event = {
                "step": step,
                "steps": steps,
                "loss": loss_value,
                "lr": current_lr,
                "grad_norm": grad_norm,
            }
            print(json.dumps(event), flush=True)
            if progress_path:
                with progress_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(event) + "\n")
        if preview_interval and preview_root and preview_text and (step % preview_interval == 0 or step == steps):
            summary = write_training_preview(
                model,
                preview_root=preview_root,
                step=step,
                text=preview_text,
                behavior_id=preview_behavior_id,
                emotion_id=preview_emotion_id,
                frames=preview_frames,
                sampling_steps=preview_sampling_steps,
                fps=preview_fps,
                device=device,
                seed=preview_seed + step,
                width=preview_width,
                height=preview_height,
                preview_mode=preview_mode,
                long_duration_sec=preview_duration_sec,
                min_segment_sec=preview_min_segment_sec,
                max_segment_sec=preview_max_segment_sec,
                min_segments=preview_min_segments,
                max_segments=preview_max_segments,
                max_velocity_rad_s=preview_max_velocity_rad_s,
                smooth_window=preview_smooth_window,
            )
            print(json.dumps({"preview": summary["preview_mp4"], "step": step}), flush=True)
    return losses


def sample_span_to_frame_count(sample_span_sec, fps):
    """Convert an inclusive endpoint sample span to its 30 Hz-style frame count."""
    sample_span_sec = float(sample_span_sec)
    fps = float(fps)
    if not math.isfinite(sample_span_sec) or sample_span_sec < 0.0:
        raise ValueError("sample_span_sec must be finite and non-negative")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be finite and positive")
    return max(2, int(round(sample_span_sec * fps)) + 1)


def frame_count_to_sample_span(frame_count, fps):
    frame_count = int(frame_count)
    fps = float(fps)
    if frame_count < 2:
        raise ValueError("frame_count must be at least 2")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be finite and positive")
    return float((frame_count - 1) / fps)


def frame_count_to_coverage(frame_count, fps):
    frame_count = int(frame_count)
    fps = float(fps)
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be finite and positive")
    return float(frame_count / fps)


def choose_device(requested):
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def write_training_log(path, config, losses):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": config, "losses": losses, "final_loss": losses[-1] if losses else None}, indent=2), encoding="utf-8")


def model_checkpoint_payload(model, episodes, args, device, *, step=None, loss=None):
    config = vars(args) | {"device": device, "episodes_loaded": len(episodes)}
    if step is not None:
        config["checkpoint_step"] = int(step)
    if loss is not None:
        config["checkpoint_loss"] = float(loss)
    payload = {
        "model_state_dict": model.state_dict(),
        "joint_order": JOINT_ORDER,
        "condition_dim": episodes[0]["condition"].shape[0],
        "action_dim": len(JOINT_ORDER),
        "architecture": model.architecture,
        "config": config,
    }
    if getattr(model, "action_stats", None) is not None:
        payload["action_stats"] = {
            "mean": model.action_stats["mean"].detach().cpu(),
            "std": model.action_stats["std"].detach().cpu(),
        }
    condition_contract = build_kimodo_condition_contract(episodes, getattr(args, "dataset_dir", ""))
    if condition_contract is not None:
        payload["condition_contract"] = condition_contract
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a minimal body-only ULA-FM model from LeRobot parquet")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--attention-backend", choices=("math",), default="math")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument(
        "--architecture",
        choices=[
            ULA_FM_LEGACY_ARCHITECTURE,
            ULA_MMDIT_LITE_ARCHITECTURE,
            ULA_MMDIT_V2_ARCHITECTURE,
            ULA_ADALN_LITE_ARCHITECTURE,
        ],
        default=ULA_FM_LEGACY_ARCHITECTURE,
    )
    parser.add_argument("--semantic-tokens", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--preview-every-steps", type=int, default=0)
    parser.add_argument("--preview-text", default="紧张地解释，同时双手做克制的上肢手势")
    parser.add_argument("--preview-behavior-id")
    parser.add_argument("--preview-emotion-id")
    parser.add_argument("--preview-dir")
    parser.add_argument("--preview-mode", choices=["long", "chunk"], default="long")
    parser.add_argument("--preview-frames", type=int, default=120, help="Used only when --preview-mode chunk")
    parser.add_argument("--preview-sampling-steps", type=int, default=32)
    parser.add_argument("--preview-fps", type=float, default=30.0)
    parser.add_argument("--preview-seed", type=int, default=7)
    parser.add_argument("--preview-width", type=int, default=1280)
    parser.add_argument("--preview-height", type=int, default=720)
    parser.add_argument("--preview-duration-sec", type=float, default=24.0)
    parser.add_argument("--preview-min-segment-sec", type=float, default=3.0)
    parser.add_argument("--preview-max-segment-sec", type=float, default=3.0)
    parser.add_argument("--preview-min-segments", type=int, default=8)
    parser.add_argument("--preview-max-segments", type=int, default=8)
    parser.add_argument("--preview-max-velocity-rad-s", type=float, default=3.0)
    parser.add_argument("--preview-smooth-window", type=int, default=5)
    parser.add_argument("--checkpoint-every-steps", type=int, default=0)
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--normalize-actions", action="store_true")
    args = parser.parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)
    episodes = load_lerobot_episodes(args.dataset_dir, max_episodes=args.max_episodes)
    if not episodes:
        raise SystemExit("no episodes loaded")
    action_stats = compute_action_normalization_stats(episodes) if args.normalize_actions else None
    train_episodes = normalize_episode_actions(episodes, action_stats) if action_stats is not None else episodes
    model = create_ula_model(
        args.architecture,
        action_dim=len(JOINT_ORDER),
        condition_dim=train_episodes[0]["condition"].shape[0],
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        semantic_tokens=args.semantic_tokens,
    )
    if action_stats is not None:
        model.action_stats = action_stats
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    losses = train_steps(
        model,
        train_episodes,
        args.steps,
        args.batch_size,
        args.lr,
        device,
        log_interval=args.log_interval,
        progress_path=out / "progress.jsonl",
        preview_interval=args.preview_every_steps,
        preview_root=args.preview_dir or out / "previews",
        preview_text=args.preview_text,
        preview_frames=args.preview_frames,
        preview_sampling_steps=args.preview_sampling_steps,
        preview_fps=args.preview_fps,
        preview_seed=args.preview_seed,
        preview_width=args.preview_width,
        preview_height=args.preview_height,
        preview_mode=args.preview_mode,
        preview_behavior_id=args.preview_behavior_id,
        preview_emotion_id=args.preview_emotion_id,
        preview_duration_sec=args.preview_duration_sec,
        preview_min_segment_sec=args.preview_min_segment_sec,
        preview_max_segment_sec=args.preview_max_segment_sec,
        preview_min_segments=args.preview_min_segments,
        preview_max_segments=args.preview_max_segments,
        preview_max_velocity_rad_s=args.preview_max_velocity_rad_s,
        preview_smooth_window=args.preview_smooth_window,
        checkpoint_dir=out,
        checkpoint_every_steps=args.checkpoint_every_steps,
        save_best=args.save_best,
        checkpoint_payload_fn=lambda checkpoint_model, step, loss_value: model_checkpoint_payload(
            checkpoint_model,
            train_episodes,
            args,
            device,
            step=step,
            loss=loss_value,
        ),
        weight_decay=args.weight_decay,
        adam_eps=args.adam_eps,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
    )
    model.cpu()
    torch.save(model_checkpoint_payload(model, train_episodes, args, device), out / "ula_fm_checkpoint.pt")
    write_training_log(out / "train_log.json", vars(args) | {"device": device, "episodes_loaded": len(train_episodes)}, losses)
    print(json.dumps({"output_dir": str(out), "steps": args.steps, "episodes_loaded": len(train_episodes), "final_loss": losses[-1]}, indent=2))


if __name__ == "__main__":
    main()
