#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Sequence
import unicodedata

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from upper_body_skeleton.kimodo_semantics import (
    KIMODO_BEHAVIOR_IDS,
    KIMODO_CONDITION_CONTRACT_VERSION,
    KIMODO_CONDITION_SCHEMA_VERSION,
    KIMODO_EMOTION_IDS,
    kimodo_condition_vectors_sha256,
)
from upper_body_skeleton.ula_training import KIMODO_CONDITION_DIM, build_condition_from_text, condition_vector


DEFAULT_QWEN_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_TASK_INSTRUCTION = (
    "Classify an upper-body motion instruction by its intended Kimodo robot behavior and expressed emotion"
)
DEFAULT_BEHAVIOR_INSTRUCTION = (
    "Represent the exact physical upper-body action requested for matching to a robot motion behavior. "
    "Ignore all emotion words and infer only the movement pattern."
)
DEFAULT_EMOTION_INSTRUCTION = (
    "Represent only the explicitly requested emotional tone for classification as neutral, sad, happy, angry, "
    "surprise, or fear. Ignore the action's inherent mood."
)
SEMANTIC_ADAPTER_SCHEMA_VERSION = 1
SEMANTIC_ADAPTER_ARCHITECTURES = {"shared", "dual_task"}

BEHAVIOR_TO_INDEX = {label: index for index, label in enumerate(KIMODO_BEHAVIOR_IDS)}
EMOTION_TO_INDEX = {label: index for index, label in enumerate(KIMODO_EMOTION_IDS)}


@dataclass(frozen=True)
class SemanticPromptRecord:
    behavior_id: str
    emotion_id: str
    text: str
    language: str = "en"
    source: str = "canonical_en"

    @property
    def key(self):
        return self.behavior_id, self.emotion_id


@dataclass(frozen=True)
class SemanticPrediction:
    text: str
    behavior_id: str
    emotion_id: str
    behavior_confidence: float
    emotion_confidence: float

    def __post_init__(self):
        if self.behavior_id not in BEHAVIOR_TO_INDEX:
            raise ValueError(f"unknown Kimodo behavior_id: {self.behavior_id}")
        if self.emotion_id not in EMOTION_TO_INDEX:
            raise ValueError(f"unknown Kimodo emotion_id: {self.emotion_id}")
        for name in ("behavior_confidence", "emotion_confidence"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and between zero and one")

    def as_dict(self):
        return {
            "text": self.text,
            "behavior_id": self.behavior_id,
            "emotion_id": self.emotion_id,
            "behavior_confidence": float(self.behavior_confidence),
            "emotion_confidence": float(self.emotion_confidence),
        }


def validate_semantic_labels(behavior_id, emotion_id):
    if behavior_id not in BEHAVIOR_TO_INDEX:
        raise ValueError(f"unknown Kimodo behavior_id: {behavior_id}")
    if emotion_id not in EMOTION_TO_INDEX:
        raise ValueError(f"unknown Kimodo emotion_id: {emotion_id}")


def load_semantic_prompt_catalog(path, *, require_complete=True):
    path = Path(path)
    records = []
    seen = set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row_number, row in enumerate(csv.DictReader(stream), start=2):
            behavior_id = str(row.get("behavior_id", "")).strip()
            emotion_id = str(row.get("emotion_id", "")).strip()
            text = str(row.get("prompt", "")).strip()
            try:
                validate_semantic_labels(behavior_id, emotion_id)
            except ValueError as exc:
                raise ValueError(f"invalid semantic catalog row {row_number}: {exc}") from exc
            if not text:
                raise ValueError(f"semantic catalog row {row_number} has no prompt")
            key = behavior_id, emotion_id
            if key in seen:
                raise ValueError(f"duplicate semantic catalog label pair at row {row_number}: {key}")
            seen.add(key)
            records.append(SemanticPromptRecord(behavior_id, emotion_id, text))

    expected = {(behavior, emotion) for behavior in KIMODO_BEHAVIOR_IDS for emotion in KIMODO_EMOTION_IDS}
    if require_complete and seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(f"semantic catalog must contain the complete Kimodo grid; missing={missing[:3]} extra={extra[:3]}")
    return sorted(records, key=lambda row: (BEHAVIOR_TO_INDEX[row.behavior_id], EMOTION_TO_INDEX[row.emotion_id]))


def validate_condition_bank(condition_bank, *, path="<memory>"):
    if not isinstance(condition_bank, dict):
        raise ValueError(f"Kimodo condition bank must be a dictionary: {path}")
    if condition_bank.get("behavior_ids") != list(KIMODO_BEHAVIOR_IDS):
        raise ValueError(f"Kimodo condition bank behavior label order mismatch: {path}")
    if condition_bank.get("emotion_ids") != list(KIMODO_EMOTION_IDS):
        raise ValueError(f"Kimodo condition bank emotion label order mismatch: {path}")
    if int(condition_bank.get("condition_dim", -1)) != KIMODO_CONDITION_DIM:
        raise ValueError(f"Kimodo condition bank dimension mismatch: {path}")
    if int(condition_bank.get("contract_version", -1)) != KIMODO_CONDITION_CONTRACT_VERSION:
        raise ValueError(f"Kimodo condition bank contract version mismatch: {path}")
    if float(condition_bank.get("condition_schema_version", -1)) != KIMODO_CONDITION_SCHEMA_VERSION:
        raise ValueError(f"Kimodo condition bank schema version mismatch: {path}")
    vectors = condition_bank.get("vectors")
    expected_shape = (len(KIMODO_BEHAVIOR_IDS), len(KIMODO_EMOTION_IDS), KIMODO_CONDITION_DIM)
    if not torch.is_tensor(vectors) or tuple(vectors.shape) != expected_shape:
        raise ValueError(f"Kimodo condition bank vectors must have shape {expected_shape}: {path}")
    if not torch.isfinite(vectors).all():
        raise ValueError(f"Kimodo condition bank contains non-finite values: {path}")
    source_hash = condition_bank.get("source_semantic_index_sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 64 or any(
        character not in "0123456789abcdef" for character in source_hash.lower()
    ):
        raise ValueError(f"Kimodo condition bank source hash is missing: {path}")
    expected_vectors_hash = kimodo_condition_vectors_sha256(vectors.detach().cpu().numpy())
    if condition_bank.get("canonical_vectors_sha256") != expected_vectors_hash:
        raise ValueError(f"Kimodo condition bank vector hash mismatch: {path}")
    return condition_bank


def load_kimodo_condition_bank(dataset_dir):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - core training environment includes pyarrow
        raise RuntimeError("pyarrow is required to build the Kimodo condition bank") from exc

    semantic_path = Path(dataset_dir) / "meta" / "semantic_index.parquet"
    if not semantic_path.is_file():
        raise FileNotFoundError(f"Kimodo semantic index not found: {semantic_path}")
    vectors = np.full(
        (len(KIMODO_BEHAVIOR_IDS), len(KIMODO_EMOTION_IDS), KIMODO_CONDITION_DIM),
        np.nan,
        dtype=np.float32,
    )
    seen = set()
    for row_number, row in enumerate(pq.read_table(semantic_path).to_pylist(), start=1):
        behavior_id = str(row.get("behavior_id", "")).strip()
        emotion_id = str(row.get("emotion_id", "")).strip()
        try:
            validate_semantic_labels(behavior_id, emotion_id)
        except ValueError as exc:
            raise ValueError(f"invalid Kimodo semantic index row {row_number}: {exc}") from exc
        vector = np.asarray(condition_vector(row), dtype=np.float32)
        if vector.shape != (KIMODO_CONDITION_DIM,) or not np.isfinite(vector).all():
            raise ValueError(f"invalid Kimodo condition at semantic index row {row_number}")
        index = BEHAVIOR_TO_INDEX[behavior_id], EMOTION_TO_INDEX[emotion_id]
        if index in seen and not np.array_equal(vectors[index], vector):
            raise ValueError(f"inconsistent repeated Kimodo condition at semantic index row {row_number}")
        vectors[index] = vector
        seen.add(index)
    expected = {
        (behavior_index, emotion_index)
        for behavior_index in range(len(KIMODO_BEHAVIOR_IDS))
        for emotion_index in range(len(KIMODO_EMOTION_IDS))
    }
    if seen != expected:
        raise ValueError(f"Kimodo condition bank is incomplete: {semantic_path}")
    condition_bank = {
        "contract_version": KIMODO_CONDITION_CONTRACT_VERSION,
        "condition_schema_version": KIMODO_CONDITION_SCHEMA_VERSION,
        "condition_dim": KIMODO_CONDITION_DIM,
        "behavior_ids": list(KIMODO_BEHAVIOR_IDS),
        "emotion_ids": list(KIMODO_EMOTION_IDS),
        "vectors": torch.from_numpy(vectors),
        "source_semantic_index": str(semantic_path),
        "source_semantic_index_sha256": hashlib.sha256(semantic_path.read_bytes()).hexdigest(),
        "canonical_vectors_sha256": kimodo_condition_vectors_sha256(vectors),
    }
    return validate_condition_bank(condition_bank, path=semantic_path)


def _normalized_prompt(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def _validated_text_list(value, *, field, expected_count):
    if not isinstance(value, list) or len(value) != expected_count:
        raise ValueError(f"{field} must contain exactly {expected_count} strings")
    texts = [str(text).strip() for text in value]
    if any(not text for text in texts):
        raise ValueError(f"{field} contains an empty string")
    if len({_normalized_prompt(text) for text in texts}) != len(texts):
        raise ValueError(f"{field} contains duplicate strings")
    return texts


def load_semantic_paraphrase_config(path):
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid semantic paraphrase JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict) or int(payload.get("schema_version", -1)) != 1:
        raise ValueError(f"unsupported semantic paraphrase schema: {path}")
    if set(payload) != {"schema_version", "behaviors", "emotions"}:
        raise ValueError(f"semantic paraphrase JSON has unexpected top-level fields: {path}")
    behaviors = payload.get("behaviors")
    emotions = payload.get("emotions")
    if not isinstance(behaviors, dict) or set(behaviors) != set(KIMODO_BEHAVIOR_IDS):
        raise ValueError(f"semantic paraphrases must define exactly all Kimodo behavior IDs: {path}")
    if not isinstance(emotions, dict) or set(emotions) != set(KIMODO_EMOTION_IDS):
        raise ValueError(f"semantic paraphrases must define exactly all Kimodo emotion IDs: {path}")

    validated_behaviors = {}
    for behavior_id in KIMODO_BEHAVIOR_IDS:
        row = behaviors[behavior_id]
        expected_fields = {"train_phrases", "validation_phrase", "test_phrase"}
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise ValueError(f"invalid paraphrase fields for {behavior_id}: {path}")
        train_phrases = _validated_text_list(
            row["train_phrases"], field=f"behaviors.{behavior_id}.train_phrases", expected_count=2
        )
        evaluation_phrases = {
            name: str(row[name]).strip()
            for name in ("validation_phrase", "test_phrase")
        }
        if any(not text for text in evaluation_phrases.values()):
            raise ValueError(f"behavior evaluation phrases must not be empty for {behavior_id}")
        all_phrases = train_phrases + list(evaluation_phrases.values())
        if len({_normalized_prompt(text) for text in all_phrases}) != len(all_phrases):
            raise ValueError(f"behavior train/validation/test phrases overlap for {behavior_id}")
        validated_behaviors[behavior_id] = {
            "train_phrases": train_phrases,
            **evaluation_phrases,
        }

    validated_emotions = {}
    for emotion_id in KIMODO_EMOTION_IDS:
        row = emotions[emotion_id]
        expected_fields = {"train_prefixes", "validation_prefix", "test_prefix"}
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise ValueError(f"invalid paraphrase fields for emotion {emotion_id}: {path}")
        train_prefixes = _validated_text_list(
            row["train_prefixes"], field=f"emotions.{emotion_id}.train_prefixes", expected_count=2
        )
        evaluation_prefixes = {
            name: str(row[name]).strip()
            for name in ("validation_prefix", "test_prefix")
        }
        if any(not text for text in evaluation_prefixes.values()):
            raise ValueError(f"emotion evaluation prefixes must not be empty for {emotion_id}")
        all_prefixes = train_prefixes + list(evaluation_prefixes.values())
        if len({_normalized_prompt(text) for text in all_prefixes}) != len(all_prefixes):
            raise ValueError(f"emotion train/validation/test prefixes overlap for {emotion_id}")
        validated_emotions[emotion_id] = {
            "train_prefixes": train_prefixes,
            **evaluation_prefixes,
        }

    for label_type, rows, fields in (
        (
            "behavior",
            validated_behaviors,
            ("train_phrases", "validation_phrase", "test_phrase"),
        ),
        (
            "emotion",
            validated_emotions,
            ("train_prefixes", "validation_prefix", "test_prefix"),
        ),
    ):
        owners = {}
        for label, row in rows.items():
            for field in fields:
                values = row[field] if isinstance(row[field], list) else [row[field]]
                for value in values:
                    normalized = _normalized_prompt(value)
                    previous = owners.setdefault(normalized, (label, field))
                    if previous != (label, field):
                        raise ValueError(
                            f"{label_type} paraphrase text {value!r} is reused by "
                            f"{previous[0]}.{previous[1]} and {label}.{field}"
                        )
    return {"schema_version": 1, "behaviors": validated_behaviors, "emotions": validated_emotions}


def build_multilingual_semantic_records(canonical_records, paraphrase_config, *, fold=0):
    canonical_records = list(canonical_records)
    canonical_splits = latin_square_semantic_split(canonical_records, fold=fold)
    split_by_key = {
        record.key: split_name
        for split_name, rows in canonical_splits.items()
        for record in rows
    }
    records = []
    split_names = []
    for canonical in canonical_records:
        split = split_by_key[canonical.key]
        records.append(
            SemanticPromptRecord(
                canonical.behavior_id,
                canonical.emotion_id,
                canonical.text,
                language="en",
                source="canonical_en",
            )
        )
        split_names.append(split)

        behavior = paraphrase_config["behaviors"][canonical.behavior_id]
        emotion = paraphrase_config["emotions"][canonical.emotion_id]
        if split == "train":
            texts = [
                f"{prefix}，{phrase}"
                for prefix in emotion["train_prefixes"]
                for phrase in behavior["train_phrases"]
            ]
            source = "paraphrase_zh_train"
        else:
            texts = [f'{emotion[f"{split}_prefix"]}，{behavior[f"{split}_phrase"]}']
            source = f"paraphrase_zh_{split}"
        for text in texts:
            records.append(
                SemanticPromptRecord(
                    canonical.behavior_id,
                    canonical.emotion_id,
                    text,
                    language="zh",
                    source=source,
                )
            )
            split_names.append(split)

    text_splits = {}
    for record, split in zip(records, split_names):
        normalized = _normalized_prompt(record.text)
        previous = text_splits.setdefault(normalized, split)
        if previous != split:
            raise ValueError(f"semantic paraphrase text leaks across {previous} and {split}: {record.text!r}")
    return records, split_names


def build_deployment_semantic_records(canonical_records, paraphrase_config):
    records = []
    split_names = []
    for canonical in canonical_records:
        behavior = paraphrase_config["behaviors"][canonical.behavior_id]
        emotion = paraphrase_config["emotions"][canonical.emotion_id]
        records.append(
            SemanticPromptRecord(
                canonical.behavior_id,
                canonical.emotion_id,
                canonical.text,
                language="en",
                source="canonical_en",
            )
        )
        split_names.append("train")
        for prefix in emotion["train_prefixes"]:
            for phrase in behavior["train_phrases"]:
                records.append(
                    SemanticPromptRecord(
                        canonical.behavior_id,
                        canonical.emotion_id,
                        f"{prefix}，{phrase}",
                        language="zh",
                        source="paraphrase_zh_train",
                    )
                )
                split_names.append("train")

        for evaluation_split in ("validation", "test"):
            records.append(
                SemanticPromptRecord(
                    canonical.behavior_id,
                    canonical.emotion_id,
                    f'{emotion[f"{evaluation_split}_prefix"]}，{behavior[f"{evaluation_split}_phrase"]}',
                    language="zh",
                    source=f"paraphrase_zh_{evaluation_split}",
                )
            )
            split_names.append(evaluation_split)

    text_splits = {}
    for record, split in zip(records, split_names):
        normalized = _normalized_prompt(record.text)
        previous = text_splits.setdefault(normalized, split)
        if previous != split:
            raise ValueError(f"semantic paraphrase text leaks across {previous} and {split}: {record.text!r}")
    return records, split_names


def latin_square_semantic_split(records, *, fold=0):
    fold = int(fold)
    if not 0 <= fold < len(KIMODO_EMOTION_IDS):
        raise ValueError(f"fold must be in [0, {len(KIMODO_EMOTION_IDS) - 1}]")
    splits = {"train": [], "validation": [], "test": []}
    for record in records:
        validate_semantic_labels(record.behavior_id, record.emotion_id)
        behavior_index = BEHAVIOR_TO_INDEX[record.behavior_id]
        emotion_index = EMOTION_TO_INDEX[record.emotion_id]
        test_emotion = (behavior_index + fold) % len(KIMODO_EMOTION_IDS)
        validation_emotion = (behavior_index + fold + 1) % len(KIMODO_EMOTION_IDS)
        if emotion_index == test_emotion:
            split = "test"
        elif emotion_index == validation_emotion:
            split = "validation"
        else:
            split = "train"
        splits[split].append(record)
    return splits


def format_qwen_instruction(text, instruction=DEFAULT_TASK_INSTRUCTION):
    text = str(text).strip()
    if not text:
        raise ValueError("semantic adapter text must not be empty")
    instruction = str(instruction).strip()
    if not instruction:
        raise ValueError("Qwen task instruction must not be empty")
    return f"Instruct: {instruction}\nQuery: {text}"


def last_token_pool(last_hidden_state, attention_mask):
    if last_hidden_state.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("last hidden state and attention mask have invalid shapes")
    if last_hidden_state.shape[:2] != attention_mask.shape:
        raise ValueError("last hidden state and attention mask shapes do not match")
    if bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item()):
        return last_hidden_state[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    rows = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
    return last_hidden_state[rows, sequence_lengths]


def _resolve_device(requested):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FrozenQwenTextEncoder:
    def __init__(
        self,
        model_name=DEFAULT_QWEN_MODEL,
        *,
        revision=None,
        instruction=DEFAULT_TASK_INSTRUCTION,
        secondary_instruction=None,
        embedding_dim=256,
        max_length=256,
        device="auto",
        local_files_only=False,
    ):
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional environment
            raise RuntimeError("transformers>=4.51 is required for the Qwen semantic adapter") from exc

        self.model_name = str(model_name)
        self.requested_revision = None if revision in (None, "") else str(revision)
        self.instruction = str(instruction)
        self.secondary_instruction = (
            None if secondary_instruction in (None, "") else str(secondary_instruction)
        )
        self.component_embedding_dim = int(embedding_dim)
        self.embedding_dim = self.component_embedding_dim * (2 if self.secondary_instruction else 1)
        self.max_length = int(max_length)
        self.device = _resolve_device(device)
        if not 32 <= self.component_embedding_dim <= 1024:
            raise ValueError("Qwen embedding_dim must be between 32 and 1024")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")

        load_kwargs = {
            "revision": self.requested_revision,
            "local_files_only": bool(local_files_only),
            "trust_remote_code": False,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left", **load_kwargs)
        dtype = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        self.model = AutoModel.from_pretrained(self.model_name, dtype=dtype, **load_kwargs)
        self.model.to(self.device).eval()
        self.model.requires_grad_(False)
        hidden_size = int(getattr(self.model.config, "hidden_size", 0))
        if self.component_embedding_dim > hidden_size:
            raise ValueError(
                f"embedding_dim {self.component_embedding_dim} exceeds Qwen hidden size {hidden_size}"
            )
        self.hidden_size = hidden_size
        self.revision = str(getattr(self.model.config, "_commit_hash", None) or self.requested_revision or "main")

    def encode(self, texts: Sequence[str], *, batch_size=16):
        raw_texts = [str(text).strip() for text in texts]
        if not raw_texts:
            raise ValueError("cannot encode an empty text collection")
        if any(not text for text in raw_texts):
            raise ValueError("semantic adapter text must not be empty")
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        components = []
        instructions = [self.instruction]
        if self.secondary_instruction:
            instructions.append(self.secondary_instruction)
        with torch.inference_mode():
            for instruction in instructions:
                task_texts = [format_qwen_instruction(text, instruction) for text in raw_texts]
                encoded = []
                for start in range(0, len(task_texts), batch_size):
                    batch = self.tokenizer(
                        task_texts[start : start + batch_size],
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )
                    batch = {name: value.to(self.device) for name, value in batch.items()}
                    output = self.model(**batch)
                    pooled = last_token_pool(output.last_hidden_state, batch["attention_mask"])
                    pooled = F.normalize(
                        pooled[:, : self.component_embedding_dim].float(),
                        p=2,
                        dim=-1,
                    )
                    encoded.append(pooled.cpu())
                components.append(torch.cat(encoded, dim=0))
        result = torch.cat(components, dim=-1)
        if not torch.isfinite(result).all():
            raise FloatingPointError("Qwen returned non-finite text embeddings")
        return result.numpy().astype(np.float32, copy=False)


class SemanticAdapterNetwork(nn.Module):
    def __init__(
        self,
        embedding_dim=256,
        hidden_dim=128,
        dropout=0.1,
        *,
        architecture="shared",
        component_embedding_dim=None,
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.architecture = str(architecture)
        self.component_embedding_dim = (
            None if component_embedding_dim is None else int(component_embedding_dim)
        )
        if self.embedding_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("adapter dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.architecture not in SEMANTIC_ADAPTER_ARCHITECTURES:
            raise ValueError(f"unknown semantic adapter architecture: {self.architecture}")
        if self.architecture == "dual_task":
            if self.component_embedding_dim is None:
                self.component_embedding_dim = self.embedding_dim // 2
            if self.embedding_dim != 2 * self.component_embedding_dim:
                raise ValueError("dual_task adapter input must contain two equal embedding components")
            self.behavior_backbone = self._make_backbone(self.component_embedding_dim)
            self.emotion_backbone = self._make_backbone(self.component_embedding_dim)
        else:
            if self.component_embedding_dim not in (None, self.embedding_dim):
                raise ValueError("shared adapter component_embedding_dim must equal embedding_dim")
            self.component_embedding_dim = self.embedding_dim
            self.backbone = self._make_backbone(self.embedding_dim)
        self.behavior_head = nn.Linear(self.hidden_dim, len(KIMODO_BEHAVIOR_IDS))
        self.emotion_head = nn.Linear(self.hidden_dim, len(KIMODO_EMOTION_IDS))

    def _make_backbone(self, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
        )

    def forward(self, embeddings):
        embeddings = torch.as_tensor(embeddings, dtype=torch.float32, device=self.behavior_head.weight.device)
        if embeddings.ndim != 2 or embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(f"embeddings must have shape [batch, {self.embedding_dim}]")
        if self.architecture == "dual_task":
            behavior_embedding, emotion_embedding = embeddings.split(self.component_embedding_dim, dim=-1)
            behavior_hidden = self.behavior_backbone(F.normalize(behavior_embedding, p=2, dim=-1))
            emotion_hidden = self.emotion_backbone(F.normalize(emotion_embedding, p=2, dim=-1))
        else:
            hidden = self.backbone(F.normalize(embeddings, p=2, dim=-1))
            behavior_hidden = hidden
            emotion_hidden = hidden
        return {
            "behavior_logits": self.behavior_head(behavior_hidden),
            "emotion_logits": self.emotion_head(emotion_hidden),
        }


def semantic_adapter_loss(output, behavior_target, emotion_target, *, emotion_weight=1.0):
    behavior_target = torch.as_tensor(
        behavior_target, dtype=torch.long, device=output["behavior_logits"].device
    )
    emotion_target = torch.as_tensor(emotion_target, dtype=torch.long, device=output["emotion_logits"].device)
    behavior = F.cross_entropy(output["behavior_logits"], behavior_target)
    emotion = F.cross_entropy(output["emotion_logits"], emotion_target)
    return {"total": behavior + float(emotion_weight) * emotion, "behavior": behavior, "emotion": emotion}


def _record_targets(records, device="cpu"):
    behavior = torch.tensor([BEHAVIOR_TO_INDEX[row.behavior_id] for row in records], dtype=torch.long, device=device)
    emotion = torch.tensor([EMOTION_TO_INDEX[row.emotion_id] for row in records], dtype=torch.long, device=device)
    return behavior, emotion


def _macro_accuracy(predicted, target):
    values = []
    for label in torch.unique(target).tolist():
        mask = target == int(label)
        values.append(float((predicted[mask] == target[mask]).float().mean().item()))
    return float(np.mean(values)) if values else 0.0


def evaluate_semantic_adapter(model, embeddings, records, *, device="cpu"):
    device = torch.device(device)
    embeddings = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    behavior, emotion = _record_targets(records, device=device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        output = model(embeddings)
        losses = semantic_adapter_loss(output, behavior, emotion)
        behavior_predicted = output["behavior_logits"].argmax(dim=-1)
        emotion_predicted = output["emotion_logits"].argmax(dim=-1)
    if was_training:
        model.train()
    behavior_correct = behavior_predicted == behavior
    emotion_correct = emotion_predicted == emotion
    return {
        "loss": float(losses["total"].cpu()),
        "behavior_accuracy": float(behavior_correct.float().mean().cpu()),
        "emotion_accuracy": float(emotion_correct.float().mean().cpu()),
        "joint_accuracy": float((behavior_correct & emotion_correct).float().mean().cpu()),
        "behavior_macro_accuracy": _macro_accuracy(behavior_predicted.cpu(), behavior.cpu()),
        "emotion_macro_accuracy": _macro_accuracy(emotion_predicted.cpu(), emotion.cpu()),
        "count": int(len(records)),
    }


def evaluate_semantic_adapter_groups(model, embeddings, records, *, device="cpu"):
    records = list(records)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.shape[0] != len(records):
        raise ValueError("grouped semantic evaluation records and embeddings do not match")
    result = {"by_source": {}, "by_language": {}}
    for field, output_name in (("source", "by_source"), ("language", "by_language")):
        values = sorted({str(getattr(record, field)) for record in records})
        for value in values:
            rows = [index for index, record in enumerate(records) if str(getattr(record, field)) == value]
            result[output_name][value] = evaluate_semantic_adapter(
                model,
                embeddings[rows],
                [records[index] for index in rows],
                device=device,
            )
    return result


class FrozenQwenSemanticAdapter:
    def __init__(self, text_encoder, network, *, device="auto"):
        self.text_encoder = text_encoder
        self.device = _resolve_device(device)
        self.network = network.to(self.device).eval()

    def predict(self, texts: Sequence[str], *, batch_size=16):
        raw_texts = [str(text).strip() for text in texts]
        if any(not text for text in raw_texts):
            raise ValueError("semantic adapter text must not be empty")
        embeddings = self.text_encoder.encode(raw_texts, batch_size=batch_size)
        with torch.no_grad():
            output = self.network(torch.as_tensor(embeddings, dtype=torch.float32, device=self.device))
            behavior_probability = output["behavior_logits"].softmax(dim=-1)
            emotion_probability = output["emotion_logits"].softmax(dim=-1)
            behavior_confidence, behavior_index = behavior_probability.max(dim=-1)
            emotion_confidence, emotion_index = emotion_probability.max(dim=-1)
        predictions = []
        for row, text in enumerate(raw_texts):
            predictions.append(
                SemanticPrediction(
                    text=text,
                    behavior_id=KIMODO_BEHAVIOR_IDS[int(behavior_index[row])],
                    emotion_id=KIMODO_EMOTION_IDS[int(emotion_index[row])],
                    behavior_confidence=float(behavior_confidence[row].cpu()),
                    emotion_confidence=float(emotion_confidence[row].cpu()),
                )
            )
        return predictions

    def predict_one(self, text):
        return self.predict([text], batch_size=1)[0]


class AdapterConditionBuilder:
    def __init__(self, adapter, *, condition_bank=None, base_builder=build_condition_from_text):
        self.adapter = adapter
        self.condition_bank = (
            None if condition_bank is None else validate_condition_bank(condition_bank)
        )
        self.base_builder = base_builder
        self.last_prediction = None

    def __call__(self, text, *, behavior_id=None, emotion_id=None, condition_dim=KIMODO_CONDITION_DIM, **kwargs):
        if int(condition_dim) != KIMODO_CONDITION_DIM:
            raise ValueError(
                f"Qwen semantic adapters require a {KIMODO_CONDITION_DIM}-dimensional Kimodo generator checkpoint"
            )
        predicted = self.adapter.predict_one(text)
        resolved_behavior = behavior_id or predicted.behavior_id
        resolved_emotion = emotion_id or predicted.emotion_id
        validate_semantic_labels(resolved_behavior, resolved_emotion)
        self.last_prediction = SemanticPrediction(
            text=str(text).strip(),
            behavior_id=resolved_behavior,
            emotion_id=resolved_emotion,
            behavior_confidence=1.0 if behavior_id else predicted.behavior_confidence,
            emotion_confidence=1.0 if emotion_id else predicted.emotion_confidence,
        )
        if self.condition_bank is not None:
            vector = self.condition_bank["vectors"][
                BEHAVIOR_TO_INDEX[resolved_behavior],
                EMOTION_TO_INDEX[resolved_emotion],
            ]
            return vector.detach().cpu().numpy().astype(np.float32, copy=True)
        return self.base_builder(
            text,
            behavior_id=resolved_behavior,
            emotion_id=resolved_emotion,
            condition_dim=condition_dim,
            **kwargs,
        )


def semantic_adapter_checkpoint_payload(
    model,
    *,
    model_name,
    model_revision,
    instruction,
    max_length,
    fold,
    best_step,
    validation_metrics,
    test_metrics,
    split_keys,
    validation_breakdown=None,
    test_breakdown=None,
    component_embedding_dim=None,
    secondary_instruction=None,
    split_mode="latin",
    condition_bank=None,
):
    component_embedding_dim = int(
        component_embedding_dim or getattr(model, "component_embedding_dim", model.embedding_dim)
    )
    instructions = [str(instruction)]
    if secondary_instruction not in (None, ""):
        instructions.append(str(secondary_instruction))
    payload = {
        "schema_version": SEMANTIC_ADAPTER_SCHEMA_VERSION,
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "qwen": {
            "model_name": str(model_name),
            "revision": str(model_revision),
            "instruction": str(instruction),
            "embedding_dim": int(model.embedding_dim),
            "component_embedding_dim": component_embedding_dim,
            "instructions": instructions,
            "max_length": int(max_length),
        },
        "adapter": {
            "architecture": str(model.architecture),
            "hidden_dim": int(model.hidden_dim),
            "dropout": float(model.dropout),
            "component_embedding_dim": int(model.component_embedding_dim),
        },
        "behavior_ids": list(KIMODO_BEHAVIOR_IDS),
        "emotion_ids": list(KIMODO_EMOTION_IDS),
        "fold": int(fold),
        "split_mode": str(split_mode),
        "best_step": int(best_step),
        "validation_metrics": dict(validation_metrics),
        "test_metrics": dict(test_metrics),
        "split_keys": {name: [list(key) for key in keys] for name, keys in split_keys.items()},
        "validation_breakdown": dict(validation_breakdown or {}),
        "test_breakdown": dict(test_breakdown or {}),
    }
    if condition_bank is not None:
        validate_condition_bank(condition_bank)
        payload["condition_bank"] = {
            **condition_bank,
            "vectors": condition_bank["vectors"].detach().cpu(),
        }
    return payload


def validate_semantic_adapter_checkpoint(checkpoint, *, path="<memory>"):
    if not isinstance(checkpoint, dict):
        raise ValueError(f"semantic adapter checkpoint must contain a dictionary: {path}")
    if int(checkpoint.get("schema_version", -1)) != SEMANTIC_ADAPTER_SCHEMA_VERSION:
        raise ValueError(f"unsupported semantic adapter schema version: {path}")
    if checkpoint.get("behavior_ids") != list(KIMODO_BEHAVIOR_IDS):
        raise ValueError(f"semantic adapter behavior label order mismatch: {path}")
    if checkpoint.get("emotion_ids") != list(KIMODO_EMOTION_IDS):
        raise ValueError(f"semantic adapter emotion label order mismatch: {path}")
    if not isinstance(checkpoint.get("model_state_dict"), dict) or not checkpoint["model_state_dict"]:
        raise ValueError(f"semantic adapter checkpoint has no model_state_dict: {path}")
    qwen = checkpoint.get("qwen")
    adapter = checkpoint.get("adapter")
    if not isinstance(qwen, dict) or not isinstance(adapter, dict):
        raise ValueError(f"semantic adapter checkpoint metadata is incomplete: {path}")
    for field in ("model_name", "revision", "instruction", "embedding_dim", "max_length"):
        if qwen.get(field) in (None, ""):
            raise ValueError(f"semantic adapter checkpoint is missing qwen.{field}: {path}")
    for field in ("hidden_dim", "dropout"):
        if adapter.get(field) is None:
            raise ValueError(f"semantic adapter checkpoint is missing adapter.{field}: {path}")
    architecture = str(adapter.get("architecture", "shared"))
    if architecture not in SEMANTIC_ADAPTER_ARCHITECTURES:
        raise ValueError(f"unsupported semantic adapter architecture {architecture!r}: {path}")
    instructions = qwen.get("instructions")
    if instructions is not None:
        if not isinstance(instructions, list) or len(instructions) not in (1, 2) or any(
            not isinstance(value, str) or not value.strip() for value in instructions
        ):
            raise ValueError(f"semantic adapter qwen.instructions is invalid: {path}")
        component_embedding_dim = int(qwen.get("component_embedding_dim", 0))
        if component_embedding_dim <= 0 or component_embedding_dim * len(instructions) != int(qwen["embedding_dim"]):
            raise ValueError(f"semantic adapter Qwen embedding component metadata is inconsistent: {path}")
        adapter_component_dim = int(adapter.get("component_embedding_dim", qwen["embedding_dim"]))
        if adapter_component_dim != component_embedding_dim:
            raise ValueError(f"semantic adapter and Qwen component dimensions do not match: {path}")
        if architecture == "dual_task" and len(instructions) != 2:
            raise ValueError(f"dual_task semantic adapter requires two Qwen instructions: {path}")
    for name, tensor in checkpoint["model_state_dict"].items():
        if not torch.is_tensor(tensor) or not torch.isfinite(tensor).all():
            raise ValueError(f"semantic adapter checkpoint contains an invalid tensor at {name}: {path}")
    if checkpoint.get("condition_bank") is not None:
        validate_condition_bank(checkpoint["condition_bank"], path=path)
    return checkpoint


def load_semantic_adapter(
    checkpoint_path,
    *,
    model_name=None,
    revision=None,
    device="auto",
    local_files_only=False,
    allow_incompatible_encoder=False,
    text_encoder=None,
    text_encoder_factory: Callable = FrozenQwenTextEncoder,
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_semantic_adapter_checkpoint(checkpoint, path=checkpoint_path)
    qwen = checkpoint["qwen"]
    adapter_config = checkpoint["adapter"]
    network = SemanticAdapterNetwork(
        embedding_dim=int(qwen["embedding_dim"]),
        hidden_dim=int(adapter_config["hidden_dim"]),
        dropout=float(adapter_config["dropout"]),
        architecture=str(adapter_config.get("architecture", "shared")),
        component_embedding_dim=int(
            adapter_config.get("component_embedding_dim", qwen["embedding_dim"])
        ),
    )
    network.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if text_encoder is None:
        requested_model = qwen["model_name"] if model_name in (None, "") else str(model_name)
        requested_revision = qwen["revision"] if revision in (None, "") else str(revision)
        mismatches = []
        if requested_model != str(qwen["model_name"]):
            mismatches.append(f"model {requested_model!r} != {qwen['model_name']!r}")
        if requested_revision != str(qwen["revision"]):
            mismatches.append(f"revision {requested_revision!r} != {qwen['revision']!r}")
        if mismatches and not allow_incompatible_encoder:
            raise ValueError(
                "semantic encoder override does not match the encoder used to train this adapter: "
                + "; ".join(mismatches)
                + ". Pass allow_incompatible_encoder=True only for an intentional compatibility experiment."
            )
        instructions = qwen.get("instructions") or [qwen["instruction"]]
        text_encoder = text_encoder_factory(
            model_name=requested_model,
            revision=requested_revision,
            instruction=instructions[0],
            secondary_instruction=instructions[1] if len(instructions) > 1 else None,
            embedding_dim=int(qwen.get("component_embedding_dim", qwen["embedding_dim"])),
            max_length=int(qwen["max_length"]),
            device=device,
            local_files_only=local_files_only,
        )
    elif not allow_incompatible_encoder:
        encoder_model = getattr(text_encoder, "model_name", None)
        encoder_revision = getattr(text_encoder, "revision", None)
        if encoder_model != qwen["model_name"] or encoder_revision != qwen["revision"]:
            raise ValueError(
                "injected semantic encoder must declare the model_name and revision stored in the adapter "
                "checkpoint, or allow_incompatible_encoder=True must be explicit"
            )
    if int(getattr(text_encoder, "embedding_dim", qwen["embedding_dim"])) != int(qwen["embedding_dim"]):
        raise ValueError("Qwen text encoder embedding dimension does not match adapter checkpoint")
    return FrozenQwenSemanticAdapter(text_encoder, network, device=device), checkpoint


SEMANTIC_ADAPTER_ARTIFACTS = {
    "semantic_adapter_checkpoint.pt",
    "semantic_adapter_checkpoint.pt.tmp",
    "metrics.json",
    "progress.jsonl",
    "split_manifest.json",
}


def _prepare_training_output(output_dir, *, overwrite):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"semantic adapter output directory is not empty: {output_dir}")
        for name in SEMANTIC_ADAPTER_ARTIFACTS:
            path = output_dir / name
            if path.is_file():
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _atomic_torch_save(payload, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_semantic_adapter_head(
    records,
    embeddings,
    split_names,
    *,
    output_dir,
    model_name,
    model_revision,
    instruction=DEFAULT_TASK_INSTRUCTION,
    max_length=256,
    fold=0,
    hidden_dim=128,
    dropout=0.1,
    steps=1_000,
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-4,
    eval_interval=25,
    early_stopping_patience=20,
    seed=7,
    device="auto",
    overwrite=False,
    architecture="shared",
    component_embedding_dim=None,
    secondary_instruction=None,
    split_mode="latin",
    condition_bank=None,
):
    records = list(records)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    split_names = np.asarray(split_names, dtype=str)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(records):
        raise ValueError("semantic embeddings must have shape [records, embedding_dim]")
    if split_names.shape != (len(records),):
        raise ValueError("split_names must contain one split for each semantic record")
    if not np.isfinite(embeddings).all():
        raise ValueError("semantic embeddings must be finite")
    if set(split_names.tolist()) != {"train", "validation", "test"}:
        raise ValueError("semantic adapter requires train, validation, and test rows")
    if steps <= 0 or batch_size <= 0 or eval_interval <= 0:
        raise ValueError("steps, batch_size, and eval_interval must be positive")

    output_dir = _prepare_training_output(output_dir, overwrite=bool(overwrite))
    progress_path = output_dir / "progress.jsonl"
    progress_path.write_text("", encoding="utf-8")
    resolved_device = _resolve_device(device)
    torch.manual_seed(int(seed))
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    rng = np.random.default_rng(int(seed))

    indices = {name: np.flatnonzero(split_names == name) for name in ("train", "validation", "test")}
    model = SemanticAdapterNetwork(
        embedding_dim=embeddings.shape[1],
        hidden_dim=hidden_dim,
        dropout=dropout,
        architecture=architecture,
        component_embedding_dim=component_embedding_dim,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    embedding_tensor = torch.as_tensor(embeddings, dtype=torch.float32, device=resolved_device)
    behavior_target, emotion_target = _record_targets(records, device=resolved_device)

    best_score = None
    best_state = None
    best_step = 0
    best_validation = None
    best_validation_breakdown = None
    evaluations_without_improvement = 0
    stopped_early = False
    train_rows = indices["train"]

    for step in range(1, int(steps) + 1):
        replace = len(train_rows) < int(batch_size)
        batch_rows = rng.choice(train_rows, size=int(batch_size), replace=replace)
        batch_rows = torch.as_tensor(batch_rows, dtype=torch.long, device=resolved_device)
        model.train()
        output = model(embedding_tensor[batch_rows])
        losses = semantic_adapter_loss(output, behavior_target[batch_rows], emotion_target[batch_rows])
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(f"non-finite semantic adapter loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()

        should_evaluate = step == 1 or step % int(eval_interval) == 0 or step == int(steps)
        if not should_evaluate:
            continue
        validation_rows = indices["validation"]
        validation = evaluate_semantic_adapter(
            model,
            embeddings[validation_rows],
            [records[index] for index in validation_rows],
            device=resolved_device,
        )
        validation_records = [records[index] for index in validation_rows]
        validation_breakdown = evaluate_semantic_adapter_groups(
            model,
            embeddings[validation_rows],
            validation_records,
            device=resolved_device,
        )
        source_joint_scores = [
            metrics["joint_accuracy"]
            for metrics in validation_breakdown["by_source"].values()
        ]
        score = (
            min(source_joint_scores),
            validation["joint_accuracy"],
            validation["behavior_macro_accuracy"],
            validation["emotion_macro_accuracy"],
            -validation["loss"],
        )
        is_best = best_score is None or score > best_score
        if is_best:
            best_score = score
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_step = step
            best_validation = validation
            best_validation_breakdown = validation_breakdown
            evaluations_without_improvement = 0
        else:
            evaluations_without_improvement += 1
        event = {
            "step": int(step),
            "steps": int(steps),
            "train_loss": float(losses["total"].detach().cpu()),
            "train_behavior_loss": float(losses["behavior"].detach().cpu()),
            "train_emotion_loss": float(losses["emotion"].detach().cpu()),
            "validation": validation,
            "validation_breakdown": validation_breakdown,
            "is_best": bool(is_best),
        }
        with progress_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        if early_stopping_patience and evaluations_without_improvement >= int(early_stopping_patience):
            stopped_early = True
            break

    if best_state is None:
        raise RuntimeError("semantic adapter training produced no checkpoint")
    model.load_state_dict(best_state)
    test_rows = indices["test"]
    test_metrics = evaluate_semantic_adapter(
        model,
        embeddings[test_rows],
        [records[index] for index in test_rows],
        device=resolved_device,
    )
    test_records = [records[index] for index in test_rows]
    test_breakdown = evaluate_semantic_adapter_groups(
        model,
        embeddings[test_rows],
        test_records,
        device=resolved_device,
    )
    split_keys = {
        name: sorted({records[index].key for index in rows})
        for name, rows in indices.items()
    }
    checkpoint = semantic_adapter_checkpoint_payload(
        model,
        model_name=model_name,
        model_revision=model_revision,
        instruction=instruction,
        max_length=max_length,
        fold=fold,
        best_step=best_step,
        validation_metrics=best_validation,
        test_metrics=test_metrics,
        split_keys=split_keys,
        validation_breakdown=best_validation_breakdown,
        test_breakdown=test_breakdown,
        component_embedding_dim=component_embedding_dim,
        secondary_instruction=secondary_instruction,
        split_mode=split_mode,
        condition_bank=condition_bank,
    )
    _atomic_torch_save(checkpoint, output_dir / "semantic_adapter_checkpoint.pt")
    split_manifest = {
        name: [
            {
                "behavior_id": records[index].behavior_id,
                "emotion_id": records[index].emotion_id,
                "text": records[index].text,
                "language": records[index].language,
                "source": records[index].source,
            }
            for index in rows
        ]
        for name, rows in indices.items()
    }
    write_json(output_dir / "split_manifest.json", split_manifest)
    summary = {
        "output_dir": str(output_dir),
        "qwen_model": str(model_name),
        "qwen_revision": str(model_revision),
        "adapter_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "embedding_dim": int(model.embedding_dim),
        "hidden_dim": int(model.hidden_dim),
        "fold": int(fold),
        "split_mode": str(split_mode),
        "adapter_architecture": str(model.architecture),
        "train_records": int(len(indices["train"])),
        "validation_records": int(len(indices["validation"])),
        "test_records": int(len(indices["test"])),
        "best_step": int(best_step),
        "stopped_early": bool(stopped_early),
        "validation": best_validation,
        "validation_breakdown": best_validation_breakdown,
        "test": test_metrics,
        "test_breakdown": test_breakdown,
    }
    write_json(output_dir / "metrics.json", summary)
    return summary


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Classify motion text with a frozen-Qwen Kimodo semantic adapter")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", action="append", required=True, help="Motion instruction; may be repeated")
    parser.add_argument("--model-name")
    parser.add_argument("--revision")
    parser.add_argument(
        "--allow-incompatible-encoder",
        action="store_true",
        help="Allow a model or revision that differs from the adapter training metadata",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)

    adapter, checkpoint = load_semantic_adapter(
        args.checkpoint,
        model_name=args.model_name,
        revision=args.revision,
        device=args.device,
        local_files_only=args.local_files_only,
        allow_incompatible_encoder=args.allow_incompatible_encoder,
    )
    result = {
        "checkpoint": str(args.checkpoint),
        "qwen": checkpoint["qwen"],
        "effective_encoder": {
            "model_name": getattr(adapter.text_encoder, "model_name", None),
            "revision": getattr(adapter.text_encoder, "revision", None),
        },
        "predictions": [prediction.as_dict() for prediction in adapter.predict(args.text)],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
