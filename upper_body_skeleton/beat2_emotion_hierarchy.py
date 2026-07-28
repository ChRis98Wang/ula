"""BEAT2-only intended-emotion hierarchy and source-group-first sampling.

The emotion labels consumed here are the official BEAT2 filename-protocol
performance intentions.  They are useful weak metadata, but they are never
promoted to perceived robot-affect truth.  Every artifact loader therefore
requires the original supervision masks to remain disabled.

This module is independent of the generator trainer.  It provides:

* an exact-resumable native-length sampler that chooses emotion, speaker,
  source recording, and finally one event;
* a train-only Qwen prototype hierarchy for neutral/non-neutral, six intended
  emotions, and the 54 category/intensity/emotion prompt groups; and
* a frozen motion-perceptual wrapper whose primary losses are the two coarse
  hierarchy levels and whose 54-group objective is auxiliary only.

The V7 implementation optimizes these levels simultaneously.  It is a
hierarchical objective, not a staged curriculum scheduler.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from upper_body_skeleton.beat2_semantic_perceptual import (
    BEAT2_DATA_POLICY,
    LATENT_DIM,
    Beat2SemanticPerceptualLoss,
    load_train_qwen_group_prototypes,
    motion_to_global_prototype_info_nce,
)
from upper_body_skeleton.beat2_emotion_supervision import (
    load_emotion_training_rows,
)
from upper_body_skeleton.ula_v2_18d_posttrain import (
    native_length_bucket,
    native_length_microbatch_capacity,
)


EMOTION_ORDER = ("neutral", "sad", "happy", "angry", "surprise", "fear")
EMOTION_TO_INDEX = {
    emotion: index for index, emotion in enumerate(EMOTION_ORDER)
}
INTENDED_WEAK_LABEL_ROLE = (
    "official_beat2_filename_intended_emotion_weak_metadata_"
    "not_perceived_robot_affect_truth_v1"
)
SAMPLING_POLICY = "source_group_first_uniform_emotion_native_length_v1"
PROTOTYPE_ARTIFACT_KIND = "beat2_intended_emotion_hierarchy_prototypes_v1"
FORBIDDEN_EXTERNAL_TOKEN = "kimodo"
FIXED_SPLITS = ("train", "validation", "test")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iter_strings(value: Any, path: str = "root"):
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield from _iter_strings(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, child in enumerate(value):
            yield from _iter_strings(child, f"{path}[{index}]")


def reject_forbidden_external_tokens(value: Any) -> None:
    """Fail closed if a config, row, or artifact references Kimodo."""

    for field, text in _iter_strings(value):
        if FORBIDDEN_EXTERNAL_TOKEN in text.lower():
            raise ValueError(
                f"{field} contains a forbidden external-data token"
            )


def _row_identity(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    emotion = str(row.get("emotion_id") or "")
    speaker = str(row.get("speaker_key") or "")
    source_group = str(row.get("source_group_key") or "")
    clip_id = str(row.get("clip_id") or "")
    if emotion not in EMOTION_TO_INDEX:
        raise ValueError(f"{clip_id or '<unknown>'}: unsupported emotion")
    if not speaker or not source_group or not clip_id:
        raise ValueError("sampler rows require speaker/source_group/clip_id")
    return emotion, speaker, source_group, clip_id


class SourceGroupFirstNativeBucketSampler:
    """Exact-resumable emotion/speaker/source/event sampler.

    Buckets are scheduled by unique source-group membership, never by event
    row count.  Inside a bucket, the selection order is:

    ``uniform emotion -> uniform speaker -> uniform source group -> event``.

    This prevents a long source recording with many semantic-event rows from
    receiving proportionally more probability than a recording with one row.
    Source groups are not repeated inside a microbatch when enough distinct
    groups are available.
    """

    STATE_VERSION = 1

    def __init__(
        self,
        episodes: Sequence[Mapping[str, Any]],
        *,
        buckets: Sequence[int],
        seed: int,
    ) -> None:
        if not episodes:
            raise ValueError("source-group-first sampler requires episodes")
        self.buckets = tuple(sorted({int(value) for value in buckets}))
        if not self.buckets or any(value <= 0 for value in self.buckets):
            raise ValueError("sampler buckets must be positive")
        self.seed = int(seed)

        group_identity: dict[str, tuple[str, str, str]] = {}
        tree: dict[
            int, dict[str, dict[str, dict[str, list[dict[str, Any]]]]]
        ] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )
        for raw_row in episodes:
            row = dict(raw_row)
            reject_forbidden_external_tokens(row)
            dataset = row.get("dataset")
            dataset_source = str(row.get("dataset_source") or "")
            if not (
                dataset == "BEAT2"
                or (
                    dataset is None
                    and dataset_source.casefold().startswith("beat2")
                )
            ):
                raise ValueError("source-group-first sampler accepts BEAT2 only")
            if row.get("fixed_split_assignment") != "train":
                raise ValueError(
                    "source-group-first sampler accepts the train split only"
                )
            emotion, speaker, source_group, clip_id = _row_identity(row)
            frames = int(np.asarray(row.get("actions")).shape[0])
            if frames <= 0:
                raise ValueError(f"{clip_id}: actions are missing")
            bucket = native_length_bucket(frames, self.buckets)
            identity = (emotion, speaker, "train")
            previous = group_identity.setdefault(source_group, identity)
            if previous != identity:
                raise ValueError(
                    f"{source_group}: source group changes emotion/speaker/split"
                )
            tree[bucket][emotion][speaker][source_group].append(row)

        self.tree = {
            bucket: {
                emotion: {
                    speaker: {
                        source_group: sorted(
                            rows, key=lambda row: str(row["clip_id"])
                        )
                        for source_group, rows in sorted(groups.items())
                    }
                    for speaker, groups in sorted(speakers.items())
                }
                for emotion, speakers in sorted(emotions.items())
            }
            for bucket, emotions in sorted(tree.items())
        }
        if not self.tree:
            raise ValueError("source-group-first sampler has no native buckets")

        self.schedule_rng = random.Random(self.seed + 7919)
        self.choice_rngs = {
            bucket: random.Random(
                self.seed
                + int.from_bytes(
                    hashlib.sha256(
                        f"source-group-bucket:{bucket}".encode("ascii")
                    ).digest()[:4],
                    byteorder="big",
                )
            )
            for bucket in self.tree
        }
        self.bucket_schedule = [
            bucket
            for bucket, emotions in self.tree.items()
            for _ in range(
                len(
                    {
                        source_group
                        for speakers in emotions.values()
                        for groups in speakers.values()
                        for source_group in groups
                    }
                )
            )
        ]
        self.schedule_rng.shuffle(self.bucket_schedule)
        self.schedule_cursor = 0

        structure = {
            "buckets": list(self.buckets),
            "policy": SAMPLING_POLICY,
            "selection_order": [
                "emotion",
                "speaker",
                "source_group",
                "event",
            ],
            "membership": {
                str(bucket): {
                    emotion: {
                        speaker: {
                            source_group: [
                                str(row["clip_id"]) for row in rows
                            ]
                            for source_group, rows in groups.items()
                        }
                        for speaker, groups in speakers.items()
                    }
                    for emotion, speakers in emotions.items()
                }
                for bucket, emotions in self.tree.items()
            },
        }
        self.structure_sha256 = hashlib.sha256(
            _canonical_json(structure)
        ).hexdigest()

    def _next_bucket(self) -> int:
        if self.schedule_cursor >= len(self.bucket_schedule):
            self.schedule_rng.shuffle(self.bucket_schedule)
            self.schedule_cursor = 0
        bucket = int(self.bucket_schedule[self.schedule_cursor])
        self.schedule_cursor += 1
        return bucket

    def _draw_row(
        self, bucket: int, used_source_groups: set[str]
    ) -> dict[str, Any]:
        rng = self.choice_rngs[bucket]
        tree = self.tree[bucket]

        def available_sources(
            emotion: str, speaker: str
        ) -> list[str]:
            return [
                source_group
                for source_group in tree[emotion][speaker]
                if source_group not in used_source_groups
            ]

        emotions = [
            emotion
            for emotion in EMOTION_ORDER
            if emotion in tree
            and any(
                available_sources(emotion, speaker)
                for speaker in tree[emotion]
            )
        ]
        if not emotions:
            used_source_groups.clear()
            emotions = [
                emotion for emotion in EMOTION_ORDER if emotion in tree
            ]
        emotion = rng.choice(emotions)
        speakers = [
            speaker
            for speaker in tree[emotion]
            if available_sources(emotion, speaker)
        ]
        if not speakers:
            # This can only occur immediately after exhausting all unique
            # source groups.  Starting a new without-replacement cycle is
            # deterministic and keeps the hierarchy unchanged.
            used_source_groups.clear()
            speakers = list(tree[emotion])
        speaker = rng.choice(speakers)
        source_groups = available_sources(emotion, speaker)
        if not source_groups:
            source_groups = list(tree[emotion][speaker])
        source_group = rng.choice(source_groups)
        used_source_groups.add(source_group)
        return dict(rng.choice(tree[emotion][speaker][source_group]))

    def sample_microbatch(
        self,
        *,
        remaining_effective_batch: int,
        semantic_tokens: int,
        max_batch_size: int,
        batching: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        bucket = self._next_bucket()
        plan = native_length_microbatch_capacity(
            bucket,
            semantic_tokens=int(semantic_tokens),
            max_batch_size=int(max_batch_size),
            max_motion_tokens=int(
                batching["max_motion_tokens_per_microbatch"]
            ),
            max_attention_elements=int(
                batching["max_attention_elements_per_microbatch"]
            ),
        )
        count = min(int(remaining_effective_batch), int(plan["capacity"]))
        if count <= 0:
            raise ValueError("remaining effective batch must be positive")
        used_source_groups: set[str] = set()
        rows = [
            self._draw_row(bucket, used_source_groups) for _ in range(count)
        ]
        return rows, plan | {
            "microbatch_size": count,
            "motion_tokens": count * bucket,
            "attention_elements": count
            * int(plan["per_episode_attention_elements"]),
            "sampling_policy": SAMPLING_POLICY,
            "selection_order": (
                "emotion_then_speaker_then_source_group_then_event"
            ),
            "unique_source_groups": len(
                {str(row["source_group_key"]) for row in rows}
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "state_version": self.STATE_VERSION,
            "structure_sha256": self.structure_sha256,
            "bucket_schedule": list(self.bucket_schedule),
            "schedule_cursor": int(self.schedule_cursor),
            "schedule_rng_state": self.schedule_rng.getstate(),
            "choice_rng_states": {
                bucket: rng.getstate()
                for bucket, rng in self.choice_rngs.items()
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (
            state.get("state_version") != self.STATE_VERSION
            or state.get("structure_sha256") != self.structure_sha256
        ):
            raise ValueError(
                "source-group-first sampler resume structure changed"
            )
        schedule = [int(value) for value in state.get("bucket_schedule") or ()]
        if sorted(schedule) != sorted(self.bucket_schedule):
            raise ValueError(
                "source-group-first sampler bucket schedule changed"
            )
        cursor = int(state.get("schedule_cursor", -1))
        if not 0 <= cursor <= len(schedule):
            raise ValueError(
                "source-group-first sampler cursor is invalid"
            )
        choice_states = dict(state.get("choice_rng_states") or {})
        if set(choice_states) != set(self.choice_rngs):
            raise ValueError(
                "source-group-first sampler RNG set changed"
            )
        self.bucket_schedule = schedule
        self.schedule_cursor = cursor
        self.schedule_rng.setstate(state["schedule_rng_state"])
        for bucket, rng in self.choice_rngs.items():
            rng.setstate(choice_states[bucket])


@dataclass(frozen=True)
class EmotionHierarchyPrototypeBank:
    binary_embeddings: torch.Tensor
    binary_ids: torch.Tensor
    emotion_embeddings: torch.Tensor
    emotion_ids: torch.Tensor
    group_embeddings: torch.Tensor
    group_ids: torch.Tensor
    group_to_emotion_index: torch.Tensor
    metadata: dict[str, Any]


def derive_emotion_hierarchy_prototypes(
    group_embeddings: torch.Tensor,
    group_ids: torch.Tensor,
    group_to_emotion: Mapping[int, str],
) -> EmotionHierarchyPrototypeBank:
    """Derive equal-group-weight six-way and binary Qwen prototypes."""

    groups = torch.as_tensor(group_embeddings, dtype=torch.float32)
    ids = torch.as_tensor(group_ids, dtype=torch.long)
    if groups.ndim != 2 or groups.shape[1] != LATENT_DIM:
        raise ValueError("group embeddings must have shape [groups, 128]")
    if ids.shape != (groups.shape[0],):
        raise ValueError("group IDs must have shape [groups]")
    if torch.unique(ids).numel() != ids.numel():
        raise ValueError("group IDs must be unique")
    if set(int(value) for value in ids.tolist()) != set(group_to_emotion):
        raise ValueError("group-to-emotion mapping does not cover the bank")
    groups = F.normalize(groups, dim=-1)

    emotion_rows = []
    groups_per_emotion: dict[str, list[int]] = {}
    for emotion in EMOTION_ORDER:
        selected = [
            index
            for index, group_id in enumerate(ids.tolist())
            if group_to_emotion[int(group_id)] == emotion
        ]
        if not selected:
            raise ValueError(f"prototype bank has no {emotion} group")
        groups_per_emotion[emotion] = [
            int(ids[index]) for index in selected
        ]
        prototype = groups[selected].mean(dim=0)
        if not torch.isfinite(prototype).all() or prototype.norm() <= 1e-8:
            raise ValueError(f"{emotion} prototype is non-finite or zero")
        emotion_rows.append(F.normalize(prototype, dim=0))
    emotions = torch.stack(emotion_rows)
    if len(ids) == 54 and any(
        len(groups_per_emotion[emotion]) != 9 for emotion in EMOTION_ORDER
    ):
        raise ValueError(
            "the 54-group bank must contain nine groups per emotion"
        )

    neutral = emotions[EMOTION_TO_INDEX["neutral"]]
    nonneutral = F.normalize(emotions[1:].mean(dim=0), dim=0)
    binary = torch.stack((neutral, nonneutral))
    lookup = torch.full(
        (int(ids.max().item()) + 1,), -1, dtype=torch.long
    )
    for group_id, emotion in group_to_emotion.items():
        lookup[int(group_id)] = EMOTION_TO_INDEX[emotion]
    metadata = {
        "artifact_kind": PROTOTYPE_ARTIFACT_KIND,
        "data_policy": BEAT2_DATA_POLICY,
        "no_kimodo": True,
        "label_role": INTENDED_WEAK_LABEL_ROLE,
        "fit_split": "train",
        "validation_or_test_rows_used": 0,
        "group_count": int(len(ids)),
        "emotion_count": len(EMOTION_ORDER),
        "binary_count": 2,
        "emotion_order": list(EMOTION_ORDER),
        "binary_order": ["neutral", "non_neutral"],
        "aggregation": (
            "equal_weight_normalized_qwen_group_mean_then_l2_normalize"
        ),
        "groups_per_emotion": groups_per_emotion,
    }
    return EmotionHierarchyPrototypeBank(
        binary_embeddings=binary,
        binary_ids=torch.arange(2, dtype=torch.long),
        emotion_embeddings=emotions,
        emotion_ids=torch.arange(len(EMOTION_ORDER), dtype=torch.long),
        group_embeddings=groups,
        group_ids=ids,
        group_to_emotion_index=lookup,
        metadata=metadata,
    )


def _read_manifest_rows(
    path: Path, *, expected_sha256: str
) -> list[dict[str, Any]]:
    if FORBIDDEN_EXTERNAL_TOKEN in str(path).lower():
        raise ValueError("manifest path contains a forbidden external token")
    if _sha256_file(path) != expected_sha256:
        raise ValueError("BEAT2 hierarchy manifest SHA256 changed")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"manifest row {line_number} is not an object"
                )
            reject_forbidden_external_tokens(value)
            clip_id = str(value.get("clip_id") or "")
            expected = {
                "dataset": value.get("dataset") == "BEAT2",
                "emotion_label_source": value.get("emotion_label_source")
                == "official_beat2_filename_protocol",
                "source_emotion_label_verified": value.get(
                    "source_emotion_label_verified"
                )
                is True,
                "emotion_supervision_mask": value.get(
                    "emotion_supervision_mask"
                )
                is False,
                "emotion_conditioning_mask": value.get(
                    "emotion_conditioning_mask"
                )
                is False,
                "affect_observable_supervision_mask": value.get(
                    "affect_observable_supervision_mask"
                )
                is False,
                "official_emotion_conditioning_enabled": value.get(
                    "official_emotion_conditioning_enabled"
                )
                is False,
            }
            prompt_contract = value.get("prompt_contract") or {}
            expected["prompt_text_supervision_mask"] = (
                prompt_contract.get("prompt_text_supervision_mask") is False
            )
            failed = sorted(
                name for name, passed in expected.items() if not passed
            )
            if failed:
                raise ValueError(
                    f"{clip_id or line_number}: invalid weak-label contract "
                    f"{failed}"
                )
            if value.get("emotion_id") not in EMOTION_TO_INDEX:
                raise ValueError(f"{clip_id}: unsupported emotion")
            if value.get("fixed_split_assignment") not in FIXED_SPLITS:
                raise ValueError(f"{clip_id}: invalid fixed split")
            rows.append(value)
    if not rows:
        raise ValueError("BEAT2 hierarchy manifest is empty")
    return rows


def load_beat2_emotion_hierarchy_prototypes(
    manifest_path: str | Path,
    condition_cache_path: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_qwen_variant: str = "frozen_base",
    prototype_aggregation: str = "require_identical",
    prototype_consistency_tolerance: float = 1e-6,
) -> EmotionHierarchyPrototypeBank:
    """Load train-only hierarchy prototypes with fail-closed weak labels."""

    manifest = Path(manifest_path).expanduser().resolve()
    cache = Path(condition_cache_path).expanduser().resolve()
    reject_forbidden_external_tokens(
        {"manifest_path": str(manifest), "condition_cache_path": str(cache)}
    )
    rows = _read_manifest_rows(
        manifest, expected_sha256=str(expected_manifest_sha256)
    )
    row_by_clip = {str(row["clip_id"]): row for row in rows}
    if len(row_by_clip) != len(rows):
        raise ValueError("BEAT2 hierarchy manifest clip IDs are not unique")

    group_embeddings, group_ids, group_metadata = (
        load_train_qwen_group_prototypes(
            cache,
            expected_manifest_sha256=str(expected_manifest_sha256),
            expected_group_count=54,
            expected_variant=str(expected_qwen_variant),
            aggregation=prototype_aggregation,
            consistency_tolerance=prototype_consistency_tolerance,
        )
    )
    with np.load(cache, allow_pickle=False) as archive:
        required = {
            "clip_ids",
            "semantic_group_indices",
            "fixed_split_assignments",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(
                f"Qwen cache identity fields are missing: {sorted(missing)}"
            )
        clip_ids = np.asarray(archive["clip_ids"]).astype(str)
        groups = np.asarray(
            archive["semantic_group_indices"], dtype=np.int64
        )
        splits = np.asarray(
            archive["fixed_split_assignments"]
        ).astype(str)
    if not (clip_ids.shape == groups.shape == splits.shape):
        raise ValueError("Qwen hierarchy cache identity shapes changed")

    group_emotions: dict[int, set[str]] = defaultdict(set)
    train_rows = 0
    for clip_id, group_id, split in zip(
        clip_ids.tolist(), groups.tolist(), splits.tolist()
    ):
        row = row_by_clip.get(str(clip_id))
        if row is None:
            raise ValueError(f"Qwen cache clip is absent from manifest: {clip_id}")
        if str(row["fixed_split_assignment"]) != str(split):
            raise ValueError(f"{clip_id}: Qwen/manifest split changed")
        if split == "train":
            train_rows += 1
            group_emotions[int(group_id)].add(str(row["emotion_id"]))
    expected_groups = set(int(value) for value in group_ids.tolist())
    if set(group_emotions) != expected_groups:
        raise ValueError("train rows do not cover every Qwen group")
    ambiguous = {
        group: sorted(emotions)
        for group, emotions in group_emotions.items()
        if len(emotions) != 1
    }
    if ambiguous:
        raise ValueError(
            f"Qwen groups do not map to one intended emotion: {ambiguous}"
        )
    mapping = {
        group: next(iter(emotions))
        for group, emotions in group_emotions.items()
    }
    bank = derive_emotion_hierarchy_prototypes(
        group_embeddings, group_ids, mapping
    )
    metadata = dict(bank.metadata)
    metadata.update(
        {
            "source_manifest": str(manifest),
            "source_manifest_sha256": str(expected_manifest_sha256),
            "source_qwen_cache": str(cache),
            "source_qwen_cache_sha256": _sha256_file(cache),
            "train_row_count": train_rows,
            "manifest_row_count": len(rows),
            "group_prototype_metadata": group_metadata,
            "formal_release_eligible": False,
            "human_perceived_emotion_supervision_count": 0,
        }
    )
    return EmotionHierarchyPrototypeBank(
        binary_embeddings=bank.binary_embeddings,
        binary_ids=bank.binary_ids,
        emotion_embeddings=bank.emotion_embeddings,
        emotion_ids=bank.emotion_ids,
        group_embeddings=bank.group_embeddings,
        group_ids=bank.group_ids,
        group_to_emotion_index=bank.group_to_emotion_index,
        metadata=metadata,
    )


def validate_emotion_supervision_binding(
    supervision_manifest_path: str | Path,
    supervision_audit_path: str | Path,
    source_manifest_path: str | Path,
    *,
    expected_supervision_manifest_sha256: str,
    expected_supervision_audit_sha256: str,
    expected_source_manifest_sha256: str,
    expected_weak_weight: float = 0.1,
) -> dict[str, Any]:
    """Bind V7 weak targets to the shared fail-closed supervision ingress."""

    supervision_manifest = Path(supervision_manifest_path).resolve()
    supervision_audit = Path(supervision_audit_path).resolve()
    source_manifest = Path(source_manifest_path).resolve()
    reject_forbidden_external_tokens(
        {
            "supervision_manifest": str(supervision_manifest),
            "supervision_audit": str(supervision_audit),
            "source_manifest": str(source_manifest),
        }
    )
    expected = {
        supervision_manifest: str(expected_supervision_manifest_sha256),
        supervision_audit: str(expected_supervision_audit_sha256),
        source_manifest: str(expected_source_manifest_sha256),
    }
    for path, digest in expected.items():
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not path.is_file()
            or _sha256_file(path) != digest
        ):
            raise ValueError(f"emotion hierarchy hash binding changed: {path}")

    source_rows = _read_manifest_rows(
        source_manifest,
        expected_sha256=str(expected_source_manifest_sha256),
    )
    train_source = {
        str(row["clip_id"]): row
        for row in source_rows
        if row["fixed_split_assignment"] == "train"
    }
    training_rows = load_emotion_training_rows(supervision_manifest)
    bound = {
        str(row["sample_id"]): row
        for row in training_rows
        if row.get("source_manifest_sha256")
        == str(expected_source_manifest_sha256)
    }
    if set(bound) != set(train_source):
        raise ValueError(
            "emotion supervision ingress does not exactly cover source train rows"
        )
    expected_weak_weight = float(expected_weak_weight)
    if (
        not math.isfinite(expected_weak_weight)
        or expected_weak_weight <= 0
    ):
        raise ValueError("expected weak emotion weight must be positive")
    for sample_id, row in bound.items():
        source = train_source[sample_id]
        if (
            row.get("supervision_tier") != "intended_metadata_weak"
            or row.get("emotion_target_source")
            != "official_beat2_filename_protocol_intended_metadata"
            or row.get("human_confirmed_observable") is not False
            or row.get("emotion_supervision_mask") is not False
            or row.get("weak_emotion_training_mask") is not True
            or float(row.get("emotion_loss_weight", -1))
            != expected_weak_weight
            or row.get("emotion_id") != source.get("emotion_id")
            or row.get("trajectory_sha256")
            != source.get("safe_csv_sha256")
        ):
            raise ValueError(
                f"{sample_id}: emotion supervision weak binding changed"
            )

    audit = json.loads(supervision_audit.read_text(encoding="utf-8"))
    reject_forbidden_external_tokens(audit)
    counts = audit.get("counts") or {}
    outputs = audit.get("outputs") or {}
    integrity = audit.get("integrity") or {}
    if (
        audit.get("artifact_kind")
        != "beat2_emotion_supervision_ingest_audit_v1"
        or audit.get("dataset") != "BEAT2"
        or audit.get("formal_release_eligible") is not False
        or int(counts.get("human_confirmed_observable_records", -1)) != 0
        or outputs.get("manifest_sha256")
        != str(expected_supervision_manifest_sha256)
        or integrity.get("kimodo_strings_present") is not False
        or integrity.get("speaker_disjoint_split") is not True
        or integrity.get("source_group_disjoint_split") is not True
        or integrity.get("trajectory_disjoint_split") is not True
    ):
        raise ValueError("emotion supervision audit contract changed")
    return {
        "artifact_kind": "beat2_emotion_hierarchy_supervision_binding_v1",
        "data_policy": BEAT2_DATA_POLICY,
        "no_kimodo": True,
        "label_role": INTENDED_WEAK_LABEL_ROLE,
        "supervision_manifest": {
            "path": str(supervision_manifest),
            "sha256": str(expected_supervision_manifest_sha256),
        },
        "supervision_audit": {
            "path": str(supervision_audit),
            "sha256": str(expected_supervision_audit_sha256),
        },
        "source_manifest": {
            "path": str(source_manifest),
            "sha256": str(expected_source_manifest_sha256),
        },
        "bound_train_rows": len(bound),
        "weak_weight": expected_weak_weight,
        "human_confirmed_observable_rows": 0,
        "formal_release_eligible": False,
    }


class Beat2EmotionHierarchyLoss(nn.Module):
    """Binary + six-way primary losses with a small 54-way auxiliary."""

    def __init__(
        self,
        bank: EmotionHierarchyPrototypeBank,
        *,
        binary_weight: float = 1.0,
        emotion_weight: float = 1.0,
        group_auxiliary_weight: float = 0.1,
        weak_label_weight: float = 0.1,
        temperature: float = 0.07,
        validate_inputs: bool = True,
    ) -> None:
        super().__init__()
        self.binary_weight = float(binary_weight)
        self.emotion_weight = float(emotion_weight)
        self.group_auxiliary_weight = float(group_auxiliary_weight)
        self.weak_label_weight = float(weak_label_weight)
        self.temperature = float(temperature)
        self.validate_inputs = bool(validate_inputs)
        for name, value in (
            ("binary_weight", self.binary_weight),
            ("emotion_weight", self.emotion_weight),
            ("group_auxiliary_weight", self.group_auxiliary_weight),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.binary_weight <= 0 or self.emotion_weight <= 0:
            raise ValueError("binary and six-emotion losses must be primary")
        if not 0 < self.group_auxiliary_weight <= 0.25:
            raise ValueError(
                "54-group objective must be a small positive auxiliary"
            )
        if not 0 < self.weak_label_weight < 1:
            raise ValueError(
                "intended weak-label weight must be positive and below one"
            )
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.metadata = dict(bank.metadata)
        self.register_buffer(
            "binary_embeddings", bank.binary_embeddings.float()
        )
        self.register_buffer("binary_ids", bank.binary_ids.long())
        self.register_buffer(
            "emotion_embeddings", bank.emotion_embeddings.float()
        )
        self.register_buffer("emotion_ids", bank.emotion_ids.long())
        self.register_buffer(
            "group_embeddings", bank.group_embeddings.float()
        )
        self.register_buffer("group_ids", bank.group_ids.long())
        self.register_buffer(
            "group_to_emotion_index",
            bank.group_to_emotion_index.long(),
        )

    def forward(
        self,
        motion_embeddings: torch.Tensor,
        semantic_group_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        groups = torch.as_tensor(
            semantic_group_ids,
            dtype=torch.long,
            device=motion_embeddings.device,
        )
        if groups.shape != (motion_embeddings.shape[0],):
            raise ValueError("semantic_group_ids must have shape [batch]")
        if self.validate_inputs and bool(
            torch.any(
                (groups < 0)
                | (groups >= self.group_to_emotion_index.numel())
            ).item()
        ):
            raise ValueError("semantic group is outside hierarchy lookup")
        emotion = self.group_to_emotion_index[groups]
        if self.validate_inputs and bool(torch.any(emotion < 0).item()):
            raise ValueError("semantic group has no intended-emotion mapping")
        binary = (emotion != EMOTION_TO_INDEX["neutral"]).long()

        binary_result = motion_to_global_prototype_info_nce(
            motion_embeddings,
            binary,
            self.binary_embeddings,
            self.binary_ids,
            temperature=self.temperature,
            validate=self.validate_inputs,
        )
        emotion_result = motion_to_global_prototype_info_nce(
            motion_embeddings,
            emotion,
            self.emotion_embeddings,
            self.emotion_ids,
            temperature=self.temperature,
            validate=self.validate_inputs,
        )
        group_result = motion_to_global_prototype_info_nce(
            motion_embeddings,
            groups,
            self.group_embeddings,
            self.group_ids,
            temperature=self.temperature,
            validate=self.validate_inputs,
        )
        unweighted_total = (
            self.binary_weight * binary_result["loss"]
            + self.emotion_weight * emotion_result["loss"]
            + self.group_auxiliary_weight * group_result["loss"]
        )
        total = self.weak_label_weight * unweighted_total
        result = {
            "total": total,
            "unweighted_total": unweighted_total,
            "binary_loss": binary_result["loss"],
            "emotion_loss": emotion_result["loss"],
            "group_auxiliary_loss": group_result["loss"],
        }
        for prefix, values in (
            ("binary", binary_result),
            ("emotion", emotion_result),
            ("group_auxiliary", group_result),
        ):
            for name, value in values.items():
                if name != "loss":
                    result[f"{prefix}_{name}"] = value.detach()
        return result


class Beat2EmotionHierarchyPerceptualLoss(nn.Module):
    """Frozen BEAT2 MotionEncoder plus the intended-emotion hierarchy."""

    def __init__(
        self,
        semantic_base: Beat2SemanticPerceptualLoss,
        hierarchy: Beat2EmotionHierarchyLoss,
        *,
        artifact_metadata: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.semantic_base = semantic_base
        self.hierarchy = hierarchy
        self.artifact_metadata = dict(artifact_metadata)
        self.requires_grad_(False)

    @classmethod
    def from_artifacts(
        cls,
        *,
        descriptor_cache_path: str | Path,
        motion_encoder_checkpoint_path: str | Path,
        action_stats: Mapping[str, Any],
        manifest_path: str | Path,
        expected_manifest_sha256: str,
        qwen_condition_cache_path: str | Path,
        expected_qwen_variant: str = "frozen_base",
        prototype_aggregation: str = "require_identical",
        prototype_consistency_tolerance: float = 1e-6,
        cosine_weight: float = 0.25,
        binary_weight: float = 1.0,
        emotion_weight: float = 1.0,
        group_auxiliary_weight: float = 0.1,
        weak_label_weight: float = 0.1,
        temperature: float = 0.07,
        validate_inputs: bool = True,
        device: str | torch.device | None = None,
    ) -> "Beat2EmotionHierarchyPerceptualLoss":
        bank = load_beat2_emotion_hierarchy_prototypes(
            manifest_path,
            qwen_condition_cache_path,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_qwen_variant=expected_qwen_variant,
            prototype_aggregation=prototype_aggregation,
            prototype_consistency_tolerance=(
                prototype_consistency_tolerance
            ),
        )
        semantic_base = Beat2SemanticPerceptualLoss.from_artifacts(
            descriptor_cache_path=descriptor_cache_path,
            motion_encoder_checkpoint_path=motion_encoder_checkpoint_path,
            action_stats=action_stats,
            qwen_condition_cache_path=None,
            cosine_weight=float(cosine_weight),
            contrastive_weight=0.0,
            global_contrastive_weight=0.0,
            temperature=float(temperature),
            validate_inputs=validate_inputs,
        )
        hierarchy = Beat2EmotionHierarchyLoss(
            bank,
            binary_weight=binary_weight,
            emotion_weight=emotion_weight,
            group_auxiliary_weight=group_auxiliary_weight,
            weak_label_weight=weak_label_weight,
            temperature=temperature,
            validate_inputs=validate_inputs,
        )
        module = cls(
            semantic_base,
            hierarchy,
            artifact_metadata={
                "data_policy": BEAT2_DATA_POLICY,
                "no_kimodo": True,
                "label_role": INTENDED_WEAK_LABEL_ROLE,
                "formal_release_eligible": False,
                "human_perceived_emotion_supervision_count": 0,
                "hierarchy_prototypes": bank.metadata,
            },
        )
        if device is not None:
            module = module.to(device)
        return module

    def train(
        self, mode: bool = True
    ) -> "Beat2EmotionHierarchyPerceptualLoss":
        super().train(mode)
        self.semantic_base.train(mode)
        return self

    def forward(
        self,
        normalized_actions: torch.Tensor,
        aligned_qwen_latents: torch.Tensor,
        *,
        frame_mask: torch.Tensor | None = None,
        durations_sec: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if group_ids is None:
            raise ValueError("emotion hierarchy requires semantic group IDs")
        base = self.semantic_base(
            normalized_actions,
            aligned_qwen_latents,
            frame_mask=frame_mask,
            durations_sec=durations_sec,
            group_ids=group_ids,
        )
        hierarchy = self.hierarchy(base["motion_embeddings"], group_ids)
        result = dict(base)
        result["total"] = base["total"] + hierarchy["total"]
        result["global_contrastive"] = hierarchy["total"]

        # Existing trainer diagnostics call this family of fields "global".
        # In V7 they intentionally report the primary six-emotion bank.
        result["global_positive_cosine"] = hierarchy[
            "emotion_positive_cosine"
        ]
        result["global_hard_negative_cosine"] = hierarchy[
            "emotion_hard_negative_cosine"
        ]
        result["global_hard_cross_group_margin"] = hierarchy[
            "emotion_hard_margin"
        ]
        result["global_hard_cross_group_margin_positive_fraction"] = (
            hierarchy["emotion_hard_margin_positive_fraction"]
        )
        result["global_motion_to_prototype_recall_at_1"] = hierarchy[
            "emotion_recall_at_1"
        ]
        result["global_mean_positive_rank"] = hierarchy[
            "emotion_mean_positive_rank"
        ]
        result["global_prototype_count"] = hierarchy[
            "emotion_prototype_count"
        ]
        result.update(
            {
                f"hierarchy_{name}": value
                for name, value in hierarchy.items()
            }
        )
        return result


__all__ = [
    "EMOTION_ORDER",
    "EMOTION_TO_INDEX",
    "INTENDED_WEAK_LABEL_ROLE",
    "SAMPLING_POLICY",
    "Beat2EmotionHierarchyLoss",
    "Beat2EmotionHierarchyPerceptualLoss",
    "EmotionHierarchyPrototypeBank",
    "SourceGroupFirstNativeBucketSampler",
    "derive_emotion_hierarchy_prototypes",
    "load_beat2_emotion_hierarchy_prototypes",
    "reject_forbidden_external_tokens",
    "validate_emotion_supervision_binding",
]
