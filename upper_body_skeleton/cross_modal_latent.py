#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Sequence
import unicodedata

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from upper_body_skeleton.kimodo_semantics import KIMODO_BEHAVIOR_IDS, KIMODO_EMOTION_IDS
from upper_body_skeleton.motion_latent import (
    BEHAVIOR_TO_INDEX,
    EMOTION_TO_INDEX,
    MotionMetricEncoder,
    build_motion_features,
    compute_motion_descriptors,
    load_motion_latent_episodes,
    motion_metric_loss,
)
from upper_body_skeleton.semantic_adapter import (
    DEFAULT_BEHAVIOR_INSTRUCTION,
    DEFAULT_EMOTION_INSTRUCTION,
    SemanticPromptRecord,
    build_deployment_semantic_records,
    format_qwen_instruction,
    last_token_pool,
    load_semantic_paraphrase_config,
    load_semantic_prompt_catalog,
    validate_condition_bank,
)


CROSS_MODAL_CHECKPOINT_SCHEMA_VERSION = 1
CROSS_MODAL_ARTIFACT_KIND = "qwen_motion_cross_modal_alignment"
DEFAULT_MOTION_TRAINABLE_PREFIXES = (
    "backbone.6.",
    "backbone.7.",
    "projection.",
    "behavior_head.",
    "emotion_head.",
    "descriptor_head.",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_text(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def _text_hash(text):
    return hashlib.sha256(_normalized_text(text).encode("utf-8")).hexdigest()


def _semantic_key(value):
    if isinstance(value, SemanticPromptRecord):
        return value.behavior_id, value.emotion_id
    meta = value.get("meta", {})
    return str(meta.get("behavior_id", "")), str(meta.get("emotion_id", ""))


@dataclass(frozen=True)
class CrossModalSplit:
    name: str
    texts: tuple[SemanticPromptRecord, ...]
    motions: tuple[dict, ...]

    def grouped(self):
        text_groups = defaultdict(list)
        motion_groups = defaultdict(list)
        for record in self.texts:
            text_groups[_semantic_key(record)].append(record)
        for episode in self.motions:
            motion_groups[_semantic_key(episode)].append(episode)
        if set(text_groups) != set(motion_groups):
            raise ValueError(f"cross-modal {self.name} text/motion semantic keys do not match")
        if not text_groups:
            raise ValueError(f"cross-modal {self.name} split is empty")
        return {
            key: (tuple(text_groups[key]), tuple(motion_groups[key]))
            for key in sorted(text_groups)
        }


def _validate_disjoint_sets(named_sets, *, field):
    names = list(named_sets)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            overlap = named_sets[left_name] & named_sets[right_name]
            if overlap:
                raise ValueError(
                    f"cross-modal split leakage in {field}: {left_name}/{right_name} overlap {sorted(overlap)[:3]}"
                )


def build_cross_modal_splits(episodes, motion_checkpoint, text_records, text_split_names):
    episodes = list(episodes)
    text_records = list(text_records)
    text_split_names = list(text_split_names)
    if len(text_records) != len(text_split_names):
        raise ValueError("semantic text records and split names do not match")
    split_ids = motion_checkpoint.get("split_episode_indices")
    if not isinstance(split_ids, dict) or set(split_ids) != {"train", "validation", "test"}:
        raise ValueError("motion checkpoint does not contain a train/validation/test split manifest")

    episode_by_id = {int(episode["episode_index"]): episode for episode in episodes}
    if len(episode_by_id) != len(episodes):
        raise ValueError("motion episodes contain duplicate episode indices")
    episode_sets = {
        name: {int(index) for index in split_ids[name]}
        for name in ("train", "validation", "test")
    }
    _validate_disjoint_sets(episode_sets, field="episode_index")
    if set().union(*episode_sets.values()) != set(episode_by_id):
        raise ValueError("motion checkpoint split manifest does not cover the loaded dataset exactly")

    texts_by_split = {name: [] for name in ("train", "validation", "test")}
    for record, split_name in zip(text_records, text_split_names):
        if split_name not in texts_by_split:
            raise ValueError(f"unknown semantic text split: {split_name}")
        texts_by_split[split_name].append(record)
    text_hash_sets = {
        name: {_text_hash(record.text) for record in records}
        for name, records in texts_by_split.items()
    }
    _validate_disjoint_sets(text_hash_sets, field="normalized text hash")

    splits = {}
    for name in ("train", "validation", "test"):
        split = CrossModalSplit(
            name=name,
            texts=tuple(texts_by_split[name]),
            motions=tuple(episode_by_id[index] for index in sorted(episode_sets[name])),
        )
        groups = split.grouped()
        expected_keys = {
            (behavior_id, emotion_id)
            for behavior_id in KIMODO_BEHAVIOR_IDS
            for emotion_id in KIMODO_EMOTION_IDS
        }
        if set(groups) != expected_keys:
            raise ValueError(f"cross-modal {name} split does not cover the complete Kimodo semantic grid")
        splits[name] = split

    manifest = {
        "schema_version": 1,
        "episode_indices": {
            name: sorted(episode_sets[name])
            for name in ("train", "validation", "test")
        },
        "text_hashes": {
            name: sorted(text_hash_sets[name])
            for name in ("train", "validation", "test")
        },
        "counts": {
            name: {
                "texts": len(splits[name].texts),
                "motions": len(splits[name].motions),
                "semantic_groups": len(splits[name].grouped()),
            }
            for name in ("train", "validation", "test")
        },
    }
    return splits, manifest


class CrossModalBatchSampler:
    def __init__(self, split, *, seed=7):
        self.groups = split.grouped()
        self.keys = tuple(sorted(self.groups))
        self.rng = np.random.default_rng(int(seed))

    def sample(self, batch_size):
        batch_size = int(batch_size)
        if batch_size <= 1 or batch_size > len(self.keys):
            raise ValueError(f"batch_size must be between 2 and {len(self.keys)}")
        key_rows = self.rng.choice(len(self.keys), size=batch_size, replace=False)
        keys = [self.keys[int(row)] for row in key_rows]
        texts = []
        motions = []
        for key in keys:
            text_rows, motion_rows = self.groups[key]
            texts.append(text_rows[int(self.rng.integers(0, len(text_rows)))].text)
            motions.append(motion_rows[int(self.rng.integers(0, len(motion_rows)))])
        return texts, motions, keys

    def state_dict(self):
        return {"bit_generator_state": self.rng.bit_generator.state}

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state["bit_generator_state"]


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=256, dropout=0.1, *, residual=False):
        super().__init__()
        self.residual = bool(residual and int(input_dim) == int(latent_dim))
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(latent_dim)),
        )

    def forward(self, values):
        projected = self.network(values)
        return values + 0.1 * projected if self.residual else projected


class QwenMotionLatentAligner(nn.Module):
    def __init__(
        self,
        qwen,
        motion_encoder,
        motion_teacher,
        *,
        qwen_component_dim=128,
        latent_dim=128,
        projection_hidden_dim=256,
        projection_dropout=0.1,
    ):
        super().__init__()
        self.qwen = qwen
        self.motion_encoder = motion_encoder
        self.motion_teacher = motion_teacher
        self.qwen_component_dim = int(qwen_component_dim)
        self.latent_dim = int(latent_dim)
        self.text_projection = ProjectionHead(
            self.qwen_component_dim * 2,
            self.latent_dim,
            projection_hidden_dim,
            projection_dropout,
        )
        self.motion_projection = ProjectionHead(
            self.motion_encoder.latent_dim,
            self.latent_dim,
            projection_hidden_dim,
            projection_dropout,
            residual=self.motion_encoder.latent_dim == self.latent_dim,
        )
        self.text_behavior_head = nn.Linear(self.latent_dim, len(KIMODO_BEHAVIOR_IDS))
        self.text_emotion_head = nn.Linear(self.latent_dim, len(KIMODO_EMOTION_IDS))
        self.motion_teacher.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        self.motion_teacher.eval()
        return self

    def encode_text(self, token_batch, *, text_count):
        output = self.qwen(**token_batch, use_cache=False)
        pooled = last_token_pool(output.last_hidden_state, token_batch["attention_mask"])
        if pooled.shape[0] != int(text_count) * 2:
            raise ValueError("dual-instruction Qwen batch has an invalid size")
        pooled = pooled[:, : self.qwen_component_dim].float()
        behavior_component = pooled[:text_count]
        emotion_component = pooled[text_count:]
        raw = self.text_projection(torch.cat([behavior_component, emotion_component], dim=-1))
        return {
            "raw": raw,
            "embedding": F.normalize(raw, dim=-1),
            "behavior_logits": self.text_behavior_head(raw),
            "emotion_logits": self.text_emotion_head(raw),
        }

    def encode_motion(self, actions, normalization):
        features = build_motion_features(actions, normalization)
        motion_output = self.motion_encoder(features)
        with torch.no_grad():
            teacher_embedding = self.motion_teacher(features)["embedding"]
        raw = self.motion_projection(motion_output["embedding"])
        return {
            "raw": raw,
            "embedding": F.normalize(raw, dim=-1),
            "teacher_embedding": teacher_embedding,
            "metric_output": motion_output,
        }


@dataclass(frozen=True)
class TextMotionPrediction:
    text: str
    behavior_id: str
    emotion_id: str
    behavior_confidence: float
    emotion_confidence: float
    motion_latent: np.ndarray


class QwenMotionTextEncoder:
    def __init__(self, model, tokenizer, checkpoint, *, device):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.checkpoint = checkpoint
        self.config = dict(checkpoint["config"])
        self.device = torch.device(device)

    def predict(self, texts: Sequence[str], *, batch_size=16):
        raw_texts = [str(text).strip() for text in texts]
        if not raw_texts or any(not text for text in raw_texts):
            raise ValueError("Qwen motion text input must contain non-empty strings")
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(raw_texts), int(batch_size)):
                batch_texts = raw_texts[start : start + int(batch_size)]
                token_batch = tokenize_alignment_texts(
                    self.tokenizer,
                    batch_texts,
                    behavior_instruction=self.config["behavior_instruction"],
                    emotion_instruction=self.config["emotion_instruction"],
                    max_length=self.config["max_length"],
                    device=self.device,
                )
                output = self.model.encode_text(token_batch, text_count=len(batch_texts))
                behavior_probability = output["behavior_logits"].softmax(dim=-1)
                emotion_probability = output["emotion_logits"].softmax(dim=-1)
                behavior_confidence, behavior_index = behavior_probability.max(dim=-1)
                emotion_confidence, emotion_index = emotion_probability.max(dim=-1)
                embeddings = output["embedding"].detach().cpu().numpy().astype(np.float32, copy=False)
                for row, text in enumerate(batch_texts):
                    predictions.append(
                        TextMotionPrediction(
                            text=text,
                            behavior_id=KIMODO_BEHAVIOR_IDS[int(behavior_index[row])],
                            emotion_id=KIMODO_EMOTION_IDS[int(emotion_index[row])],
                            behavior_confidence=float(behavior_confidence[row].cpu()),
                            emotion_confidence=float(emotion_confidence[row].cpu()),
                            motion_latent=embeddings[row].copy(),
                        )
                    )
        return predictions

    def predict_one(self, text):
        return self.predict([text], batch_size=1)[0]

    def encode(self, texts: Sequence[str], *, batch_size=16):
        return np.stack(
            [prediction.motion_latent for prediction in self.predict(texts, batch_size=batch_size)]
        ).astype(np.float32, copy=False)


class LoRAMotionConditionBuilder:
    def __init__(self, text_encoder, *, condition_bank):
        self.text_encoder = text_encoder
        self.condition_bank = validate_condition_bank(condition_bank)
        self.last_prediction = None
        self.last_motion_latent = None

    def __call__(self, text, *, behavior_id=None, emotion_id=None, condition_dim=136, **_kwargs):
        if int(condition_dim) != 136:
            raise ValueError("Qwen Motion LoRA requires a 136-dimensional Kimodo base condition")
        predicted = self.text_encoder.predict_one(text)
        resolved_behavior = behavior_id or predicted.behavior_id
        resolved_emotion = emotion_id or predicted.emotion_id
        if resolved_behavior not in BEHAVIOR_TO_INDEX or resolved_emotion not in EMOTION_TO_INDEX:
            raise ValueError("Qwen Motion LoRA predicted an unknown Kimodo semantic label")
        self.last_prediction = TextMotionPrediction(
            text=predicted.text,
            behavior_id=resolved_behavior,
            emotion_id=resolved_emotion,
            behavior_confidence=1.0 if behavior_id else predicted.behavior_confidence,
            emotion_confidence=1.0 if emotion_id else predicted.emotion_confidence,
            motion_latent=np.asarray(predicted.motion_latent, dtype=np.float32).copy(),
        )
        self.last_motion_latent = self.last_prediction.motion_latent.copy()
        vector = self.condition_bank["vectors"][
            BEHAVIOR_TO_INDEX[resolved_behavior], EMOTION_TO_INDEX[resolved_emotion]
        ]
        return vector.detach().cpu().numpy().astype(np.float32, copy=True)

def tokenize_alignment_texts(
    tokenizer,
    texts: Sequence[str],
    *,
    behavior_instruction=DEFAULT_BEHAVIOR_INSTRUCTION,
    emotion_instruction=DEFAULT_EMOTION_INSTRUCTION,
    max_length=128,
    device="cpu",
):
    texts = [str(text).strip() for text in texts]
    if not texts or any(not text for text in texts):
        raise ValueError("alignment text batch must contain non-empty strings")
    formatted = [format_qwen_instruction(text, behavior_instruction) for text in texts]
    formatted.extend(format_qwen_instruction(text, emotion_instruction) for text in texts)
    batch = tokenizer(
        formatted,
        padding=True,
        truncation=True,
        max_length=int(max_length),
        return_tensors="pt",
    )
    return {name: value.to(device) for name, value in batch.items()}


def set_motion_trainable_policy(model, prefixes=DEFAULT_MOTION_TRAINABLE_PREFIXES):
    prefixes = tuple(str(prefix) for prefix in prefixes)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            parameter.requires_grad_(True)
            trainable.append(name)
    if not trainable:
        raise ValueError("motion trainable policy did not match any parameters")
    return trainable


def load_motion_alignment_checkpoint(path, *, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("config", {})
    required = {"action_dim", "latent_dim", "hidden_dim"}
    if not required.issubset(config):
        raise ValueError(f"motion checkpoint is missing config fields: {sorted(required - set(config))}")
    if checkpoint.get("behavior_ids") != list(KIMODO_BEHAVIOR_IDS):
        raise ValueError("motion checkpoint behavior label order mismatch")
    if checkpoint.get("emotion_ids") != list(KIMODO_EMOTION_IDS):
        raise ValueError("motion checkpoint emotion label order mismatch")
    model = MotionMetricEncoder(
        action_dim=int(config["action_dim"]),
        latent_dim=int(config["latent_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("motion checkpoint contains non-finite weights")
    normalization = {
        name: torch.as_tensor(value, dtype=torch.float32, device=device)
        for name, value in checkpoint["normalization"].items()
    }
    return model.to(device), normalization, checkpoint


def build_qwen_motion_aligner(config, *, device):
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional training dependency
        raise RuntimeError("transformers and peft are required for Qwen motion alignment") from exc

    device = torch.device(device)
    model_name = str(config["model_name"])
    revision = str(config["revision"])
    local_files_only = bool(config.get("local_files_only", False))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        local_files_only=local_files_only,
        padding_side="left",
        trust_remote_code=False,
    )
    qwen_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    base_qwen = AutoModel.from_pretrained(
        model_name,
        revision=revision,
        local_files_only=local_files_only,
        trust_remote_code=False,
        dtype=qwen_dtype,
        attn_implementation=str(config.get("attention_backend", "eager")),
    )
    base_qwen.config.use_cache = False
    layer_count = int(getattr(base_qwen.config, "num_hidden_layers", 0))
    top_layers = int(config["lora_top_layers"])
    if top_layers <= 0 or top_layers > layer_count:
        raise ValueError(f"lora_top_layers must be between 1 and {layer_count}")
    layer_indices = list(range(layer_count - top_layers, layer_count))
    projections = tuple(config.get("lora_target_projections", ("q_proj", "k_proj", "v_proj", "o_proj")))
    target_modules = [
        f"layers.{layer}.self_attn.{projection}"
        for layer in layer_indices
        for projection in projections
    ]
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=int(config["lora_rank"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        target_modules=target_modules,
        bias="none",
    )
    qwen = get_peft_model(base_qwen, lora_config).to(device)
    qwen_trainable = [name for name, parameter in qwen.named_parameters() if parameter.requires_grad]
    if not qwen_trainable or any("lora_" not in name for name in qwen_trainable):
        raise RuntimeError("Qwen PEFT trainable parameter policy is invalid")

    motion_encoder, normalization, motion_checkpoint = load_motion_alignment_checkpoint(
        config["motion_checkpoint"],
        device=device,
    )
    motion_teacher = deepcopy(motion_encoder).to(device).eval()
    motion_trainable = set_motion_trainable_policy(
        motion_encoder,
        config.get("motion_trainable_prefixes", DEFAULT_MOTION_TRAINABLE_PREFIXES),
    )
    aligner = QwenMotionLatentAligner(
        qwen,
        motion_encoder,
        motion_teacher,
        qwen_component_dim=int(config["qwen_component_dim"]),
        latent_dim=int(config["latent_dim"]),
        projection_hidden_dim=int(config["projection_hidden_dim"]),
        projection_dropout=float(config["projection_dropout"]),
    ).to(device)
    metadata = {
        "model_name": model_name,
        "revision": str(getattr(base_qwen.config, "_commit_hash", None) or revision),
        "hidden_size": int(base_qwen.config.hidden_size),
        "dtype": str(qwen_dtype).replace("torch.", ""),
        "attention_backend": str(config.get("attention_backend", "eager")),
        "lora": {
            "rank": int(config["lora_rank"]),
            "alpha": int(config["lora_alpha"]),
            "dropout": float(config["lora_dropout"]),
            "layer_indices": layer_indices,
            "target_modules": target_modules,
            "trainable_parameter_names": qwen_trainable,
            "trainable_parameters": sum(
                parameter.numel() for parameter in qwen.parameters() if parameter.requires_grad
            ),
        },
        "motion_trainable_parameter_names": motion_trainable,
    }
    return aligner, tokenizer, normalization, motion_checkpoint, metadata


def load_qwen_motion_text_encoder(checkpoint_path, *, device="auto", local_files_only=None):
    from peft import set_peft_model_state_dict

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != CROSS_MODAL_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"unsupported cross-modal checkpoint schema: {checkpoint_path}")
    if checkpoint.get("artifact_kind") != CROSS_MODAL_ARTIFACT_KIND:
        raise ValueError(f"not a cross-modal Qwen LoRA checkpoint: {checkpoint_path}")
    config = dict(checkpoint.get("config") or {})
    resolved_device = torch.device(
        device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    config["device"] = str(resolved_device)
    if local_files_only is not None:
        config["local_files_only"] = bool(local_files_only)
    model, tokenizer, _, _, metadata = build_qwen_motion_aligner(config, device=resolved_device)
    set_result = set_peft_model_state_dict(model.qwen, checkpoint["qwen_lora_state_dict"])
    if getattr(set_result, "unexpected_keys", None):
        raise ValueError(f"unexpected Qwen LoRA keys: {set_result.unexpected_keys}")
    model.motion_encoder.load_state_dict(checkpoint["motion_encoder_state_dict"], strict=True)
    model.text_projection.load_state_dict(checkpoint["text_projection_state_dict"], strict=True)
    model.motion_projection.load_state_dict(checkpoint["motion_projection_state_dict"], strict=True)
    model.text_behavior_head.load_state_dict(checkpoint["text_behavior_head_state_dict"], strict=True)
    model.text_emotion_head.load_state_dict(checkpoint["text_emotion_head_state_dict"], strict=True)
    recorded_qwen = checkpoint.get("qwen") or {}
    if recorded_qwen.get("model_name") != metadata.get("model_name"):
        raise ValueError("Qwen LoRA model name does not match its checkpoint")
    if recorded_qwen.get("revision") != metadata.get("revision"):
        raise ValueError("Qwen LoRA revision does not match its checkpoint")
    model.requires_grad_(False).eval()
    return QwenMotionTextEncoder(model, tokenizer, checkpoint, device=resolved_device), checkpoint


def bidirectional_alignment_loss(text_embedding, motion_embedding, *, temperature=0.07):
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("alignment temperature must be finite and positive")
    if text_embedding.shape != motion_embedding.shape or text_embedding.ndim != 2:
        raise ValueError("text and motion embeddings must have the same [batch, latent] shape")
    logits = text_embedding @ motion_embedding.T / temperature
    target = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def variance_covariance_loss(text_raw, motion_raw, *, variance_target=1.0, eps=1e-4):
    if text_raw.shape != motion_raw.shape or text_raw.ndim != 2:
        raise ValueError("variance/covariance inputs must have the same [batch, latent] shape")
    if text_raw.shape[0] < 2:
        raise ValueError("variance/covariance loss requires at least two samples")

    def component(values):
        centered = values - values.mean(dim=0, keepdim=True)
        std = torch.sqrt(centered.var(dim=0, unbiased=True) + float(eps))
        variance = F.relu(float(variance_target) - std).mean()
        covariance = centered.T @ centered / (values.shape[0] - 1)
        covariance = covariance - torch.diag(torch.diagonal(covariance))
        covariance = covariance.square().sum() / values.shape[1]
        return variance, covariance

    text_variance, text_covariance = component(text_raw)
    motion_variance, motion_covariance = component(motion_raw)
    return {
        "variance": 0.5 * (text_variance + motion_variance),
        "covariance": 0.5 * (text_covariance + motion_covariance),
    }


def cross_modal_training_loss(
    model_output,
    behavior_target,
    emotion_target,
    descriptor_target,
    *,
    temperature=0.07,
    alignment_weight=1.0,
    text_behavior_weight=0.5,
    text_emotion_weight=0.5,
    motion_metric_weight=0.25,
    motion_anchor_weight=0.1,
    variance_weight=0.1,
    covariance_weight=0.01,
):
    text = model_output["text"]
    motion = model_output["motion"]
    behavior_target = torch.as_tensor(behavior_target, dtype=torch.long, device=text["raw"].device)
    emotion_target = torch.as_tensor(emotion_target, dtype=torch.long, device=text["raw"].device)
    alignment = bidirectional_alignment_loss(
        text["embedding"], motion["embedding"], temperature=temperature
    )
    text_behavior = F.cross_entropy(text["behavior_logits"], behavior_target)
    text_emotion = F.cross_entropy(text["emotion_logits"], emotion_target)
    motion_losses = motion_metric_loss(
        motion["metric_output"],
        behavior_target,
        emotion_target,
        descriptor_target,
    )
    motion_anchor = (1.0 - (motion["metric_output"]["embedding"] * motion["teacher_embedding"]).sum(dim=-1)).mean()
    regularization = variance_covariance_loss(text["raw"], motion["raw"])
    total = (
        float(alignment_weight) * alignment
        + float(text_behavior_weight) * text_behavior
        + float(text_emotion_weight) * text_emotion
        + float(motion_metric_weight) * motion_losses["total"]
        + float(motion_anchor_weight) * motion_anchor
        + float(variance_weight) * regularization["variance"]
        + float(covariance_weight) * regularization["covariance"]
    )
    return {
        "total": total,
        "alignment": alignment,
        "text_behavior": text_behavior,
        "text_emotion": text_emotion,
        "motion_metric": motion_losses["total"],
        "motion_anchor": motion_anchor,
        "variance": regularization["variance"],
        "covariance": regularization["covariance"],
    }


def _effective_rank(embeddings, eps=1e-12):
    embeddings = np.asarray(embeddings, dtype=np.float64)
    singular_values = np.linalg.svd(embeddings, compute_uv=False)
    energy = singular_values**2
    probabilities = energy / max(float(energy.sum()), eps)
    return float(np.exp(-(probabilities * np.log(np.maximum(probabilities, eps))).sum()))


def _retrieval_ranks(similarity):
    order = np.argsort(-similarity, axis=1)
    target = np.arange(similarity.shape[0])[:, None]
    return np.argmax(order == target, axis=1) + 1


def validation_selection_score(metrics):
    return (
        0.5
        * (
            float(metrics["text_to_motion_recall_at_1"])
            + float(metrics["motion_to_text_recall_at_1"])
        ),
        0.5
        * (
            float(metrics["text_to_motion_recall_at_5"])
            + float(metrics["motion_to_text_recall_at_5"])
        ),
        float(metrics["cosine_gap"]),
        -float(metrics["retrieval_loss"]),
    )


def _split_evaluation_pairs(split):
    texts = []
    motions = []
    keys = []
    for key, (text_rows, motion_rows) in split.grouped().items():
        texts.append(sorted(text_rows, key=lambda record: record.text)[0].text)
        motions.append(sorted(motion_rows, key=lambda episode: int(episode["episode_index"]))[0])
        keys.append(key)
    return texts, motions, keys


def evaluate_cross_modal_alignment(
    model,
    tokenizer,
    split,
    normalization,
    *,
    device,
    behavior_instruction=DEFAULT_BEHAVIOR_INSTRUCTION,
    emotion_instruction=DEFAULT_EMOTION_INSTRUCTION,
    max_length=128,
    batch_size=16,
    temperature=0.07,
):
    device = torch.device(device)
    texts, motions, keys = _split_evaluation_pairs(split)
    was_training = model.training
    model.eval()
    text_embeddings = []
    motion_embeddings = []
    text_behavior_logits = []
    text_emotion_logits = []
    motion_behavior_logits = []
    motion_emotion_logits = []
    with torch.no_grad():
        for start in range(0, len(texts), int(batch_size)):
            text_batch = texts[start : start + int(batch_size)]
            motion_batch = motions[start : start + int(batch_size)]
            tokens = tokenize_alignment_texts(
                tokenizer,
                text_batch,
                behavior_instruction=behavior_instruction,
                emotion_instruction=emotion_instruction,
                max_length=max_length,
                device=device,
            )
            actions = torch.as_tensor(
                np.stack([episode["actions"] for episode in motion_batch]),
                dtype=torch.float32,
                device=device,
            )
            text_output = model.encode_text(tokens, text_count=len(text_batch))
            motion_output = model.encode_motion(actions, normalization)
            text_embeddings.append(text_output["embedding"].cpu())
            motion_embeddings.append(motion_output["embedding"].cpu())
            text_behavior_logits.append(text_output["behavior_logits"].cpu())
            text_emotion_logits.append(text_output["emotion_logits"].cpu())
            motion_behavior_logits.append(motion_output["metric_output"]["behavior_logits"].cpu())
            motion_emotion_logits.append(motion_output["metric_output"]["emotion_logits"].cpu())
    if was_training:
        model.train()

    text_tensor = torch.cat(text_embeddings)
    motion_tensor = torch.cat(motion_embeddings)
    similarity_tensor = text_tensor @ motion_tensor.T
    similarity = similarity_tensor.numpy()
    target = torch.arange(len(keys))
    retrieval_loss = 0.5 * (
        F.cross_entropy(similarity_tensor / float(temperature), target)
        + F.cross_entropy(similarity_tensor.T / float(temperature), target)
    )
    t2m_ranks = _retrieval_ranks(similarity)
    m2t_ranks = _retrieval_ranks(similarity.T)
    behavior_target = torch.tensor([BEHAVIOR_TO_INDEX[key[0]] for key in keys])
    emotion_target = torch.tensor([EMOTION_TO_INDEX[key[1]] for key in keys])
    text_behavior = torch.cat(text_behavior_logits).argmax(dim=-1)
    text_emotion = torch.cat(text_emotion_logits).argmax(dim=-1)
    motion_behavior = torch.cat(motion_behavior_logits).argmax(dim=-1)
    motion_emotion = torch.cat(motion_emotion_logits).argmax(dim=-1)
    diagonal = np.diag(similarity)
    negative_mask = ~np.eye(len(keys), dtype=bool)
    negative_mean = float(similarity[negative_mask].mean())
    text_std = text_tensor.std(dim=0, unbiased=False)
    motion_std = motion_tensor.std(dim=0, unbiased=False)

    def accuracy(predicted, expected):
        return float((predicted == expected).float().mean())

    return {
        "count": len(keys),
        "retrieval_loss": float(retrieval_loss),
        "text_to_motion_recall_at_1": float(np.mean(t2m_ranks <= 1)),
        "text_to_motion_recall_at_5": float(np.mean(t2m_ranks <= 5)),
        "motion_to_text_recall_at_1": float(np.mean(m2t_ranks <= 1)),
        "motion_to_text_recall_at_5": float(np.mean(m2t_ranks <= 5)),
        "text_to_motion_median_rank": float(np.median(t2m_ranks)),
        "motion_to_text_median_rank": float(np.median(m2t_ranks)),
        "positive_cosine": float(diagonal.mean()),
        "negative_cosine": negative_mean,
        "cosine_gap": float(diagonal.mean() - negative_mean),
        "text_effective_rank": _effective_rank(text_tensor.numpy()),
        "motion_effective_rank": _effective_rank(motion_tensor.numpy()),
        "text_mean_dimension_std": float(text_std.mean()),
        "motion_mean_dimension_std": float(motion_std.mean()),
        "text_behavior_accuracy": accuracy(text_behavior, behavior_target),
        "text_emotion_accuracy": accuracy(text_emotion, emotion_target),
        "text_joint_accuracy": accuracy(
            (text_behavior == behavior_target) & (text_emotion == emotion_target),
            torch.ones_like(behavior_target, dtype=torch.bool),
        ),
        "motion_behavior_accuracy": accuracy(motion_behavior, behavior_target),
        "motion_emotion_accuracy": accuracy(motion_emotion, emotion_target),
        "motion_joint_accuracy": accuracy(
            (motion_behavior == behavior_target) & (motion_emotion == emotion_target),
            torch.ones_like(behavior_target, dtype=torch.bool),
        ),
    }


def _atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _atomic_json_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _lr_scale(step, *, warmup_steps, decay_steps, minimum_ratio):
    step = int(step)
    warmup_steps = int(warmup_steps)
    decay_steps = int(decay_steps)
    if warmup_steps and step <= warmup_steps:
        return step / warmup_steps
    if decay_steps <= warmup_steps or step >= decay_steps:
        return float(minimum_ratio)
    progress = (step - warmup_steps) / (decay_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(minimum_ratio) + (1.0 - float(minimum_ratio)) * cosine


def _optimizer_groups(model, config):
    qwen = [parameter for parameter in model.qwen.parameters() if parameter.requires_grad]
    text_modules = [model.text_projection, model.text_behavior_head, model.text_emotion_head]
    text = [parameter for module in text_modules for parameter in module.parameters() if parameter.requires_grad]
    motion_modules = [model.motion_encoder, model.motion_projection]
    motion = [parameter for module in motion_modules for parameter in module.parameters() if parameter.requires_grad]
    groups = [
        {"params": qwen, "lr": float(config["qwen_lr"]), "base_lr": float(config["qwen_lr"]), "name": "qwen_lora"},
        {"params": text, "lr": float(config["projector_lr"]), "base_lr": float(config["projector_lr"]), "name": "text"},
        {"params": motion, "lr": float(config["motion_lr"]), "base_lr": float(config["motion_lr"]), "name": "motion"},
    ]
    parameter_ids = [id(parameter) for group in groups for parameter in group["params"]]
    expected_ids = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
    if len(parameter_ids) != len(set(parameter_ids)) or set(parameter_ids) != set(expected_ids):
        raise RuntimeError("optimizer parameter groups do not exactly cover trainable parameters")
    return groups


def _checkpoint_payload(
    model,
    optimizer,
    *,
    config,
    metadata,
    sources,
    manifest,
    sampler,
    global_step,
    best_step,
    best_validation_loss,
    best_validation_score,
    validation_metrics,
):
    from peft import get_peft_model_state_dict

    return {
        "schema_version": CROSS_MODAL_CHECKPOINT_SCHEMA_VERSION,
        "artifact_kind": CROSS_MODAL_ARTIFACT_KIND,
        "global_step": int(global_step),
        "target_steps": int(config["steps"]),
        "best_step": int(best_step),
        "best_validation_loss": float(best_validation_loss),
        "best_validation_score": list(best_validation_score or ()),
        "validation_metrics": dict(validation_metrics or {}),
        "qwen_lora_state_dict": {
            name: value.detach().cpu()
            for name, value in get_peft_model_state_dict(model.qwen).items()
        },
        "motion_encoder_state_dict": {
            name: value.detach().cpu() for name, value in model.motion_encoder.state_dict().items()
        },
        "text_projection_state_dict": {
            name: value.detach().cpu() for name, value in model.text_projection.state_dict().items()
        },
        "motion_projection_state_dict": {
            name: value.detach().cpu() for name, value in model.motion_projection.state_dict().items()
        },
        "text_behavior_head_state_dict": {
            name: value.detach().cpu() for name, value in model.text_behavior_head.state_dict().items()
        },
        "text_emotion_head_state_dict": {
            name: value.detach().cpu() for name, value in model.text_emotion_head.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "sampler_state_dict": sampler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "config": dict(config),
        "qwen": dict(metadata),
        "sources": dict(sources),
        "split_manifest": manifest,
        "behavior_ids": list(KIMODO_BEHAVIOR_IDS),
        "emotion_ids": list(KIMODO_EMOTION_IDS),
    }


def _load_training_checkpoint(path, model, optimizer, sampler, *, config, sources):
    from peft import set_peft_model_state_dict

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != CROSS_MODAL_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"unsupported cross-modal checkpoint schema: {path}")
    if checkpoint.get("artifact_kind") != CROSS_MODAL_ARTIFACT_KIND:
        raise ValueError(f"not a cross-modal alignment trainer checkpoint: {path}")
    for field in (
        "model_name",
        "revision",
        "motion_checkpoint",
        "latent_dim",
        "qwen_component_dim",
        "lora_rank",
        "lora_alpha",
        "lora_top_layers",
        "attention_backend",
        "batch_size",
        "qwen_lr",
        "projector_lr",
        "motion_lr",
        "warmup_steps",
        "lr_decay_steps",
    ):
        if checkpoint["config"].get(field) != config.get(field):
            raise ValueError(f"resume config mismatch for {field}")
    if checkpoint.get("sources") != sources:
        raise ValueError("resume source hashes do not match the current data/model files")
    set_result = set_peft_model_state_dict(model.qwen, checkpoint["qwen_lora_state_dict"])
    if getattr(set_result, "unexpected_keys", None):
        raise ValueError(f"unexpected Qwen LoRA resume keys: {set_result.unexpected_keys}")
    model.motion_encoder.load_state_dict(checkpoint["motion_encoder_state_dict"], strict=True)
    model.text_projection.load_state_dict(checkpoint["text_projection_state_dict"], strict=True)
    model.motion_projection.load_state_dict(checkpoint["motion_projection_state_dict"], strict=True)
    model.text_behavior_head.load_state_dict(checkpoint["text_behavior_head_state_dict"], strict=True)
    model.text_emotion_head.load_state_dict(checkpoint["text_emotion_head_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    sampler.load_state_dict(checkpoint["sampler_state_dict"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all"):
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    return checkpoint


def train_qwen_motion_alignment(config):
    config = dict(config)
    output_dir = Path(config["output_dir"])
    resume_from = config.get("resume_from")
    known_outputs = {
        "last.pt",
        "last.pt.tmp",
        "best.pt",
        "best.pt.tmp",
        "progress.jsonl",
        "split_manifest.json",
        "training_summary.json",
    }
    if not resume_from and output_dir.exists() and any(output_dir.iterdir()):
        if not config.get("overwrite", False):
            raise FileExistsError(f"cross-modal output directory is not empty: {output_dir}")
        for name in known_outputs:
            path = output_dir / name
            if path.is_file():
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(config["device"] if config["device"] != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    motion_checkpoint_path = Path(config["motion_checkpoint"])
    motion_checkpoint = torch.load(motion_checkpoint_path, map_location="cpu", weights_only=True)
    episodes = load_motion_latent_episodes(config["dataset_dir"], max_episodes=config.get("max_episodes"))
    canonical_records = load_semantic_prompt_catalog(config["prompt_csv"])
    paraphrases = load_semantic_paraphrase_config(config["paraphrases_json"])
    text_records, text_split_names = build_deployment_semantic_records(canonical_records, paraphrases)
    splits, manifest = build_cross_modal_splits(
        episodes,
        motion_checkpoint,
        text_records,
        text_split_names,
    )
    sources = {
        "motion_checkpoint_sha256": sha256_file(motion_checkpoint_path),
        "semantic_index_sha256": sha256_file(Path(config["dataset_dir"]) / "meta" / "semantic_index.parquet"),
        "prompt_csv_sha256": sha256_file(config["prompt_csv"]),
        "paraphrases_sha256": sha256_file(config["paraphrases_json"]),
    }
    manifest = manifest | {"sources": sources}
    _atomic_json_save(manifest, output_dir / "split_manifest.json")

    model, tokenizer, normalization, _, metadata = build_qwen_motion_aligner(config, device=device)
    parameter_groups = _optimizer_groups(model, config)
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(config["weight_decay"]),
        eps=float(config["adam_eps"]),
    )
    sampler = CrossModalBatchSampler(splits["train"], seed=seed)
    global_step = 0
    best_step = 0
    best_validation_loss = float("inf")
    best_validation_score = None
    validation_metrics = {}
    if resume_from:
        checkpoint = _load_training_checkpoint(
            resume_from,
            model,
            optimizer,
            sampler,
            config=config,
            sources=sources,
        )
        global_step = int(checkpoint["global_step"])
        best_step = int(checkpoint["best_step"])
        best_validation_loss = float(checkpoint["best_validation_loss"])
        validation_metrics = dict(checkpoint.get("validation_metrics", {}))
        stored_score = checkpoint.get("best_validation_score")
        if stored_score:
            best_validation_score = tuple(float(value) for value in stored_score)
        best_path = output_dir / "best.pt"
        if best_validation_score is None and best_path.is_file():
            best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
            best_metrics = best_checkpoint.get("validation_metrics", {})
            if best_metrics:
                best_validation_score = validation_selection_score(best_metrics)
    else:
        (output_dir / "progress.jsonl").write_text("", encoding="utf-8")

    total_steps = int(config["steps"])
    if total_steps <= global_step:
        raise ValueError(f"target steps {total_steps} must be greater than resumed step {global_step}")
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    loss_keys = {
        "temperature": float(config["temperature"]),
        "alignment_weight": float(config["alignment_weight"]),
        "text_behavior_weight": float(config["text_behavior_weight"]),
        "text_emotion_weight": float(config["text_emotion_weight"]),
        "motion_metric_weight": float(config["motion_metric_weight"]),
        "motion_anchor_weight": float(config["motion_anchor_weight"]),
        "variance_weight": float(config["variance_weight"]),
        "covariance_weight": float(config["covariance_weight"]),
    }
    progress_path = output_dir / "progress.jsonl"
    print(
        json.dumps(
            {
                "device": str(device),
                "start_step": global_step,
                "target_steps": total_steps,
                "qwen_lora_parameters": metadata["lora"]["trainable_parameters"],
                "total_trainable_parameters": sum(parameter.numel() for parameter in trainable_parameters),
                "split_counts": manifest["counts"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for step in range(global_step + 1, total_steps + 1):
        model.train()
        texts, motion_rows, keys = sampler.sample(config["batch_size"])
        token_batch = tokenize_alignment_texts(
            tokenizer,
            texts,
            behavior_instruction=config["behavior_instruction"],
            emotion_instruction=config["emotion_instruction"],
            max_length=config["max_length"],
            device=device,
        )
        actions = torch.as_tensor(
            np.stack([episode["actions"] for episode in motion_rows]),
            dtype=torch.float32,
            device=device,
        )
        text_output = model.encode_text(token_batch, text_count=len(texts))
        motion_output = model.encode_motion(actions, normalization)
        behavior_target = torch.tensor([BEHAVIOR_TO_INDEX[key[0]] for key in keys], device=device)
        emotion_target = torch.tensor([EMOTION_TO_INDEX[key[1]] for key in keys], device=device)
        descriptors = compute_motion_descriptors(actions, normalization)
        losses = cross_modal_training_loss(
            {"text": text_output, "motion": motion_output},
            behavior_target,
            emotion_target,
            descriptors,
            **loss_keys,
        )
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(f"non-finite cross-modal loss at step {step}")
        scale = _lr_scale(
            step,
            warmup_steps=config["warmup_steps"],
            decay_steps=config["lr_decay_steps"],
            minimum_ratio=config["minimum_lr_ratio"],
        )
        for group in optimizer.param_groups:
            group["lr"] = float(group["base_lr"]) * scale
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            float(config["max_grad_norm"]),
            error_if_nonfinite=True,
        )
        optimizer.step()
        global_step = step

        should_log = step == 1 or step % int(config["log_interval"]) == 0 or step == total_steps
        should_evaluate = step == 1 or step % int(config["eval_interval"]) == 0 or step == total_steps
        should_checkpoint = step == 1 or step % int(config["checkpoint_interval"]) == 0 or step == total_steps
        is_best = False
        if should_evaluate:
            validation_metrics = evaluate_cross_modal_alignment(
                model,
                tokenizer,
                splits["validation"],
                normalization,
                device=device,
                behavior_instruction=config["behavior_instruction"],
                emotion_instruction=config["emotion_instruction"],
                max_length=config["max_length"],
                batch_size=config["eval_batch_size"],
                temperature=config["temperature"],
            )
            rank_gate = min(
                validation_metrics["text_effective_rank"],
                validation_metrics["motion_effective_rank"],
            ) >= float(config["minimum_effective_rank"])
            candidate_score = validation_selection_score(validation_metrics)
            if rank_gate and (best_validation_score is None or candidate_score > best_validation_score):
                best_validation_score = candidate_score
                best_validation_loss = float(validation_metrics["retrieval_loss"])
                best_step = step
                is_best = True

        event = {
            "step": step,
            "steps": total_steps,
            "loss": float(losses["total"].detach().cpu()),
            "alignment_loss": float(losses["alignment"].detach().cpu()),
            "text_behavior_loss": float(losses["text_behavior"].detach().cpu()),
            "text_emotion_loss": float(losses["text_emotion"].detach().cpu()),
            "motion_metric_loss": float(losses["motion_metric"].detach().cpu()),
            "motion_anchor_loss": float(losses["motion_anchor"].detach().cpu()),
            "variance_loss": float(losses["variance"].detach().cpu()),
            "covariance_loss": float(losses["covariance"].detach().cpu()),
            "grad_norm": float(grad_norm.detach().cpu()),
            "lr_scale": scale,
        }
        if should_evaluate:
            event["validation"] = validation_metrics
            event["is_best"] = is_best
        if should_log or should_evaluate:
            print(json.dumps(event, sort_keys=True), flush=True)
            with progress_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, sort_keys=True) + "\n")

        if should_checkpoint or is_best:
            payload = _checkpoint_payload(
                model,
                optimizer,
                config=config,
                metadata=metadata,
                sources=sources,
                manifest=manifest,
                sampler=sampler,
                global_step=global_step,
                best_step=best_step,
                best_validation_loss=best_validation_loss,
                best_validation_score=best_validation_score,
                validation_metrics=validation_metrics,
            )
            if should_checkpoint:
                _atomic_torch_save(payload, output_dir / "last.pt")
            if is_best:
                _atomic_torch_save(payload, output_dir / "best.pt")

    test_metrics = evaluate_cross_modal_alignment(
        model,
        tokenizer,
        splits["test"],
        normalization,
        device=device,
        behavior_instruction=config["behavior_instruction"],
        emotion_instruction=config["emotion_instruction"],
        max_length=config["max_length"],
        batch_size=config["eval_batch_size"],
        temperature=config["temperature"],
    )
    summary = {
        "output_dir": str(output_dir),
        "steps": total_steps,
        "best_step": best_step,
        "best_validation_loss": best_validation_loss,
        "best_validation_score": list(best_validation_score or ()),
        "final_validation": validation_metrics,
        "final_test": test_metrics,
        "sources": sources,
    }
    _atomic_json_save(summary, output_dir / "training_summary.json")
    return summary
