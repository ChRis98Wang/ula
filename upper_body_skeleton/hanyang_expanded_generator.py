"""Fail-closed inputs for the isolated BEAT2 + Hanyang generator experiment.

Hanyang is admitted here only as a secondary, partially observed motion domain.
It is not promoted to a generator-foundation source and its seven intended
emotion labels are not treated as generator supervision.  The running BEAT2
V7 experiment and the global data-source registry are deliberately untouched.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from upper_body_skeleton.data_source_registry import (
    BEAT2_FORMAL_SOURCE_ID,
    EMOTION_CRITIC_ROLE,
    HANYANG_EMOTIONAL_BODY_SOURCE_ID,
    KIMODO_PERMANENT_DENY_POLICY,
    assert_no_forbidden_data_lineage,
    validate_data_source_registry_contract,
)
from upper_body_skeleton.hanyang_emotion_retarget import (
    ACTION_DIM_MASK_18D,
    DATASET_ID,
    SOURCE_FPS,
    SOURCE_FRAMES,
    fixed_split_for_participant,
    json_hash,
    reject_forbidden_dataset_marker,
    sha256_file,
)
from upper_body_skeleton.hanyang_expanded_training import (
    derivative_observation_weights,
    validate_observation_weight,
    weighted_masked_mean,
)
from upper_body_skeleton.retarget_v2_18d import JOINT_ORDER_18D
from upper_body_skeleton.ula_training import KIMODO_V2_CONDITION_DIM
from upper_body_skeleton.ula_v2_18d_head import ACTION_DIM
from upper_body_skeleton.ula_v2_18d_posttrain import (
    load_attached_beat_episodes,
)
from upper_body_skeleton.ula_v2_18d_random_init import (
    forward_with_frame_mask,
)


SCHEMA_VERSION = 8
EXPERIMENT_ARTIFACT_KIND = "hanyang_beat2_expanded_generator_experiment_v8"
ADMISSION_ARTIFACT_KIND = "hanyang_partial_motion_experimental_admission_v8"
DATA_CONTRACT_ARTIFACT_KIND = "hanyang_beat2_expanded_data_contract_v8"
NORMALIZER_ARTIFACT_KIND = "foundation_bound_action_normalizer_v8"
SPLIT_NAMES = ("train", "validation", "test")
EXPECTED_HANYANG_SPLIT_COUNTS = {
    "train": 291,
    "validation": 10,
    "test": 43,
}
EXPECTED_HANYANG_TOTAL = sum(EXPECTED_HANYANG_SPLIT_COUNTS.values())
HANYANG_DOMAIN_MAX_FRACTION = 0.0625
TEXT_LATENT_SLICE = slice(136, 264)
UNOBSERVED_INDICES = tuple(
    index for index, observed in enumerate(ACTION_DIM_MASK_18D) if not observed
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            records.append(value)
    return records


def _validate_record_sha256(record: Mapping[str, Any], *, context: str) -> None:
    expected = json_hash(
        {key: value for key, value in record.items() if key != "record_sha256"}
    )
    if record.get("record_sha256") != expected:
        raise ValueError(f"{context}: record_sha256 mismatch")


def _resolved_child(path: object, *, root: Path, field: str) -> Path:
    if not isinstance(path, str) or not path:
        raise ValueError(f"{field} must be a non-empty path")
    reject_forbidden_dataset_marker(path, context=field)
    result = Path(path).expanduser().resolve()
    try:
        result.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes the completed retarget root") from exc
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def _load_faithful_csv(
    quality: Mapping[str, Any], *, retarget_root: Path
) -> tuple[np.ndarray, Path, str]:
    outputs = quality.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("quality.outputs must be a mapping")
    faithful = _resolved_child(
        outputs.get("source_faithful_partial_18d_csv"),
        root=retarget_root,
        field="outputs.source_faithful_partial_18d_csv",
    )
    deployment = _resolved_child(
        outputs.get("deployment_safe_partial_18d_csv"),
        root=retarget_root,
        field="outputs.deployment_safe_partial_18d_csv",
    )
    if faithful == deployment:
        raise ValueError(
            "source-faithful training input aliases the deployment preview"
        )
    expected_hash = outputs.get("source_faithful_partial_18d_csv_sha256")
    observed_hash = sha256_file(faithful)
    if observed_hash != expected_hash:
        raise ValueError("source-faithful partial 18D CSV hash mismatch")
    with faithful.open("r", encoding="utf-8", newline="") as stream:
        header = tuple(next(csv.reader(stream)))
    if header != tuple(JOINT_ORDER_18D):
        raise ValueError("source-faithful partial 18D CSV joint order changed")
    actions = np.loadtxt(
        faithful, delimiter=",", skiprows=1, dtype=np.float32
    )
    if actions.shape != (SOURCE_FRAMES, ACTION_DIM):
        raise ValueError(
            f"source-faithful actions must have shape "
            f"{(SOURCE_FRAMES, ACTION_DIM)}"
        )
    if not np.isfinite(actions).all():
        raise ValueError("source-faithful actions contain non-finite values")
    return np.ascontiguousarray(actions), faithful, observed_hash


def _load_observation_confidence(
    quality: Mapping[str, Any], *, retarget_root: Path
) -> tuple[np.ndarray, Path, str]:
    descriptor = quality.get("per_frame_observation_confidence")
    if not isinstance(descriptor, Mapping):
        raise ValueError("per_frame_observation_confidence is missing")
    if descriptor.get("shape") != [SOURCE_FRAMES, ACTION_DIM]:
        raise ValueError("declared observation-confidence shape changed")
    path = _resolved_child(
        descriptor.get("path"),
        root=retarget_root,
        field="per_frame_observation_confidence.path",
    )
    observed_hash = sha256_file(path)
    if observed_hash != descriptor.get("sha256"):
        raise ValueError("observation-confidence NPY hash mismatch")
    confidence = np.load(path, allow_pickle=False)
    if confidence.shape != (SOURCE_FRAMES, ACTION_DIM):
        raise ValueError("observation-confidence array shape changed")
    confidence = np.asarray(confidence, dtype=np.float32)
    if (
        not np.isfinite(confidence).all()
        or np.any(confidence < 0.0)
        or np.any(confidence > 1.0)
    ):
        raise ValueError("observation confidence must be finite in [0, 1]")
    if np.any(confidence[:, UNOBSERVED_INDICES] != 0.0):
        raise ValueError("five permanently unobserved dimensions must be zero")
    observed_indices = tuple(
        index
        for index, observed in enumerate(ACTION_DIM_MASK_18D)
        if observed
    )
    if not np.any(confidence[:, observed_indices] > 0.0):
        raise ValueError("Hanyang clip has no observed generator target")
    return np.ascontiguousarray(confidence), path, observed_hash


def load_hanyang_partial_motion_episodes(
    passed_manifest: str | Path,
    *,
    expected_manifest_sha256: str,
    pool_receipt_path: str | Path | None = None,
    expected_pool_receipt_sha256: str | None = None,
    expected_upstream_passed_manifest_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load all QC-pass Hanyang rows as unconditional partial motion targets."""

    manifest = Path(passed_manifest).expanduser().resolve()
    reject_forbidden_dataset_marker(str(manifest), context="passed_manifest")
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    manifest_hash = sha256_file(manifest)
    if manifest_hash != expected_manifest_sha256:
        raise ValueError("Hanyang passed-manifest SHA256 mismatch")
    retarget_root = manifest.parent.resolve()
    records = _read_jsonl(manifest)
    if len(records) != EXPECTED_HANYANG_TOTAL:
        raise ValueError(
            f"Hanyang QC-pass count changed: {len(records)} != "
            f"{EXPECTED_HANYANG_TOTAL}"
        )
    pool_rows = bool(
        records
        and records[0].get("artifact_kind")
        == "hanyang_v8_experimental_partial18d_pool_row_v1"
    )
    if pool_rows:
        retarget_root = manifest.parent.parent.resolve()
    pool_receipt_hash = None
    pool_review_manifest_sha256 = None
    if pool_rows:
        if (
            pool_receipt_path is None
            or not isinstance(expected_pool_receipt_sha256, str)
            or not isinstance(
                expected_upstream_passed_manifest_sha256, str
            )
        ):
            raise ValueError(
                "experimental pool loading requires its receipt and upstream hash"
            )
        receipt_path = Path(pool_receipt_path).expanduser().resolve()
        pool_receipt_hash = sha256_file(receipt_path)
        if pool_receipt_hash != expected_pool_receipt_sha256:
            raise ValueError("Hanyang experimental-pool receipt hash mismatch")
        pool_receipt = _read_json(receipt_path)
        _validate_record_sha256(pool_receipt, context=str(receipt_path))
        registry = validate_data_source_registry_contract(
            pool_receipt.get("data_source_registry_contract"),
            expected_role=EMOTION_CRITIC_ROLE,
            expected_dataset_sources=[HANYANG_EMOTIONAL_BODY_SOURCE_ID],
        )
        registry_source = registry["sources"][0]
        review_bundle = pool_receipt.get("human_review_bundle") or {}
        pool_review_manifest_sha256 = review_bundle.get("manifest_sha256")
        if (
            pool_receipt.get("artifact_kind")
            != "hanyang_v8_experimental_pool_admission_receipt_v1"
            or pool_receipt.get("dataset_id")
            != HANYANG_EMOTIONAL_BODY_SOURCE_ID
            or pool_receipt.get("pool_manifest_sha256") != manifest_hash
            or Path(pool_receipt.get("pool_manifest", "")).resolve()
            != manifest
            or pool_receipt.get("pool_row_count") != EXPECTED_HANYANG_TOTAL
            or pool_receipt.get("upstream_passed_manifest_sha256")
            != expected_upstream_passed_manifest_sha256
            or pool_receipt.get("upstream_passed_row_count")
            != EXPECTED_HANYANG_TOTAL
            or pool_receipt.get("counts_by_split")
            != EXPECTED_HANYANG_SPLIT_COUNTS
            or pool_receipt.get("complete_upstream_coverage") is not True
            or pool_receipt.get("formal") is not False
            or pool_receipt.get("formal_training_admission") is not False
            or pool_receipt.get("generator_foundation_eligible") is not False
            or pool_receipt.get("emotion_condition_eligible_count") != 0
            or pool_receipt.get("human_review_required") is not True
            or pool_receipt.get("human_review_approved") is not False
            or pool_receipt.get("training_launch_allowed") is not False
            or pool_receipt.get("human_review_status")
            != "HUMAN_REVIEW_BLOCKED"
            or review_bundle.get("sample_count") != 21
            or review_bundle.get("review_status")
            != "HUMAN_REVIEW_BLOCKED"
            or not isinstance(pool_review_manifest_sha256, str)
            or sha256_file(review_bundle.get("manifest", ""))
            != pool_review_manifest_sha256
            or registry_source.get("generator_foundation_allowed") is not False
            or registry_source.get("isolation_domain")
            != "external_emotion_critic_only"
            or set(registry_source.get("allowed_roles") or ())
            != {
                "emotion_calibration",
                "emotion_critic",
                "emotion_evaluation",
            }
        ):
            raise ValueError(
                "Hanyang experimental-pool admission/registry contract changed"
            )
    elif pool_receipt_path is not None:
        raise ValueError("pool receipt supplied for a non-pool manifest")

    episodes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row_number, record in enumerate(records, 1):
        context = f"{manifest}:{row_number}"
        _validate_record_sha256(record, context=context)
        clip_id = str(record.get("clip_id") or "")
        if not clip_id.startswith("hanyang:") or clip_id in seen:
            raise ValueError(f"{context}: invalid or duplicate clip_id")
        seen.add(clip_id)
        if record.get("dataset_id") != DATASET_ID or record.get(
            "generator_foundation_eligible"
        ) is not False or record.get("kimodo_accessed_or_used") is not False:
            raise ValueError(f"{context}: source/admission status changed")
        if pool_rows:
            if (
                record.get("artifact_kind")
                != "hanyang_v8_experimental_partial18d_pool_row_v1"
                or record.get("unconditional_motion_eligible") is not True
                or record.get("admission_tier")
                != "unconditional_motion_only"
                or record.get("emotion_condition_eligible") is not False
                or record.get("emotion_condition_mask") is not False
                or record.get("style_condition_eligible") is not False
                or record.get("style_condition_mask") is not False
                or record.get("semantic_condition_eligible") is not False
                or record.get("semantic_condition_mask") is not False
                or record.get("duration_condition_eligible") is not False
                or record.get("duration_condition_mask") is not False
                or record.get("group54_condition_eligible") is not False
                or record.get("group54_condition_mask") is not False
                or record.get("formal_training_eligible") is not False
                or record.get("deployment_preview_eligible") is not False
                or record.get("raw_ik_eligible") is not False
                or record.get("frames") != SOURCE_FRAMES
                or not math.isclose(
                    float(record.get("fps", -1.0)),
                    SOURCE_FPS,
                    abs_tol=1e-9,
                )
                or record.get("action_order") != list(JOINT_ORDER_18D)
                or record.get("action_dim_mask")
                != list(ACTION_DIM_MASK_18D)
                or record.get("upstream_passed_manifest_sha256")
                != expected_upstream_passed_manifest_sha256
            ):
                raise ValueError(
                    f"{context}: experimental motion-only lane changed"
                )
        elif record.get("status") != "passed":
            raise ValueError(f"{context}: upstream status is not passed")
        participant_id = int(record.get("participant_id", -1))
        split = fixed_split_for_participant(participant_id)
        if record.get("fixed_split_assignment") != split:
            raise ValueError(f"{context}: fixed participant split changed")
        gates = record.get("quality_gate")
        if (
            not isinstance(gates, Mapping)
            or gates.get("passed") is not True
            or any(value is not True for value in gates.values())
        ):
            raise ValueError(f"{context}: quality gate is not fully passed")

        quality_path = _resolved_child(
            record.get("quality_json"),
            root=retarget_root,
            field=f"{context}.quality_json",
        )
        if sha256_file(quality_path) != record.get("quality_json_sha256"):
            raise ValueError(f"{context}: quality report file hash mismatch")
        quality = _read_json(quality_path)
        _validate_record_sha256(quality, context=str(quality_path))
        if quality.get("record_sha256") != record.get(
            "quality_record_sha256"
        ):
            raise ValueError(f"{context}: quality record lineage mismatch")
        if (
            quality.get("artifact_kind")
            != "hanyang_partial_18d_retarget_quality_v1"
            or quality.get("dataset_id") != DATASET_ID
            or quality.get("clip_id") != clip_id
            or quality.get("participant_id") != participant_id
            or quality.get("fixed_split_assignment") != split
            or quality.get("source_frames") != SOURCE_FRAMES
            or not math.isclose(
                float(quality.get("source_fps", -1.0)),
                SOURCE_FPS,
                abs_tol=1e-9,
            )
            or (quality.get("quality_gate") or {}).get("passed") is not True
        ):
            raise ValueError(f"{context}: quality identity/shape contract changed")
        if quality.get("action_dim_mask") != list(ACTION_DIM_MASK_18D):
            raise ValueError(f"{context}: permanent 18D mask changed")
        quality_emotion = quality.get("emotion_evaluation") or {}
        expected_source_group = (
            f"hanyang:participant:{participant_id:02d}"
            if pool_rows
            else str(quality.get("source_group_key"))
        )
        if (
            record.get("source_sha256") != quality.get("source_sha256")
            or record.get("emotion_id") != quality.get("emotion_id")
            or record.get("source_stem")
            != clip_id.removeprefix("hanyang:")
            or record.get("source_group_key") != expected_source_group
        ):
            raise ValueError(f"{context}: source/emotion lineage mismatch")
        if pool_rows:
            soft = record.get("soft_emotion_targets") or {}
            distribution = quality_emotion.get(
                "soft_emotion_distribution"
            ) or {}
            expected_p7 = [
                float(distribution.get(name, -1.0))
                for name in soft.get("p7_order") or ()
            ]
            if (
                soft.get("p7_order")
                != [
                    "happy",
                    "sad",
                    "surprise",
                    "angry",
                    "disgust",
                    "fear",
                    "neutral",
                ]
                or not np.allclose(
                    np.asarray(soft.get("p7"), dtype=np.float64),
                    np.asarray(expected_p7, dtype=np.float64),
                    atol=1e-12,
                    rtol=0.0,
                )
            ):
                raise ValueError(
                    f"{context}: rater distribution/intended share changed"
                )

        actions, action_path, action_hash = _load_faithful_csv(
            quality, retarget_root=retarget_root
        )
        confidence, confidence_path, confidence_hash = (
            _load_observation_confidence(
                quality, retarget_root=retarget_root
            )
        )
        if pool_rows and (
            Path(record.get("source_faithful_partial_18d_csv", "")).resolve()
            != action_path
            or record.get("source_faithful_partial_18d_csv_sha256")
            != action_hash
            or Path(record.get("observation_confidence_npy", "")).resolve()
            != confidence_path
            or record.get("observation_confidence_npy_sha256")
            != confidence_hash
            or record.get("observation_confidence_shape")
            != [SOURCE_FRAMES, ACTION_DIM]
        ):
            raise ValueError(
                f"{context}: pool action/confidence lineage mismatch"
            )
        condition = np.zeros(KIMODO_V2_CONDITION_DIM, dtype=np.float32)
        episodes.append(
            {
                "clip_id": clip_id,
                "actions": actions,
                "fps": SOURCE_FPS,
                "duration_sec": (SOURCE_FRAMES - 1) / SOURCE_FPS,
                "condition": condition,
                "observation_confidence": confidence,
                "action_dim_mask": np.asarray(
                    ACTION_DIM_MASK_18D, dtype=np.bool_
                ),
                "fixed_split_assignment": split,
                "speaker_key": f"hanyang:participant:{participant_id:02d}",
                "source_group_key": (
                    f"hanyang:participant:{participant_id:02d}"
                ),
                "dataset_source": HANYANG_EMOTIONAL_BODY_SOURCE_ID,
                "domain": "hanyang_partial_motion",
                "source_faithful_actions_path": str(action_path),
                "source_faithful_actions_sha256": action_hash,
                "observation_confidence_path": str(confidence_path),
                "observation_confidence_sha256": confidence_hash,
                "quality_json": str(quality_path),
                "quality_json_sha256": record["quality_json_sha256"],
                "emotion_conditioning_mask": False,
                "style_supervision_mask": False,
                "semantic_supervision_mask": False,
                "duration_supervision_mask": False,
                "intended_emotion_id_audit_only": quality.get("emotion_id"),
                "intended_emotion_high_confidence_audit_only": bool(
                    (quality.get("emotion_evaluation") or {}).get(
                        "intended_high_confidence", False
                    )
                ),
                "experimental_motion_only": True,
            }
        )

    split_counts = Counter(
        episode["fixed_split_assignment"] for episode in episodes
    )
    if dict(split_counts) != EXPECTED_HANYANG_SPLIT_COUNTS:
        raise ValueError(
            f"Hanyang fixed split counts changed: {dict(split_counts)}"
        )
    receipt = {
        "artifact_kind": "hanyang_partial_motion_loader_receipt_v8",
        "schema_version": SCHEMA_VERSION,
        "manifest": str(manifest),
        "manifest_sha256": manifest_hash,
        "experimental_pool_receipt_sha256": pool_receipt_hash,
        "human_review_manifest_sha256": pool_review_manifest_sha256,
        "upstream_passed_manifest_sha256": (
            expected_upstream_passed_manifest_sha256
            if pool_rows
            else manifest_hash
        ),
        "episode_count": len(episodes),
        "split_counts": dict(split_counts),
        "condition_policy": "all_zero_unconditional_no_emotion_ingress_v1",
        "emotion_conditioned_episode_count": 0,
        "source_faithful_csv_only": True,
        "deployment_preview_used": False,
        "per_frame_per_dimension_confidence_required": True,
        "permanent_unobserved_dimensions": [
            JOINT_ORDER_18D[index] for index in UNOBSERVED_INDICES
        ],
    }
    receipt["sha256"] = canonical_sha256(receipt)
    return episodes, receipt


def load_beat2_frozen_qwen_motion_episodes(
    manifest: str | Path,
    foundation_condition_cache: str | Path,
    frozen_qwen_cache: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_foundation_condition_cache_sha256: str,
    expected_frozen_qwen_cache_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load strict BEAT2 trajectories and use only the frozen Qwen 128D input."""

    manifest = Path(manifest).expanduser().resolve()
    foundation_condition_cache = Path(
        foundation_condition_cache
    ).expanduser().resolve()
    frozen_qwen_cache = Path(frozen_qwen_cache).expanduser().resolve()
    assert_no_forbidden_data_lineage(
        {
            "manifest_path": str(manifest),
            "cache_path": str(foundation_condition_cache),
            "qwen_cache_path": str(frozen_qwen_cache),
        },
        context="beat2_v8_inputs",
    )
    manifest_hash = sha256_file(manifest)
    if manifest_hash != expected_manifest_sha256:
        raise ValueError("BEAT2 manifest SHA256 mismatch")
    foundation_cache_hash = sha256_file(foundation_condition_cache)
    qwen_cache_hash = sha256_file(frozen_qwen_cache)
    if (
        foundation_cache_hash
        != expected_foundation_condition_cache_sha256
        or qwen_cache_hash != expected_frozen_qwen_cache_sha256
    ):
        raise ValueError("BEAT2 condition-cache SHA256 mismatch")
    episodes = load_attached_beat_episodes(
        manifest,
        foundation_condition_cache,
        dataset_source=BEAT2_FORMAL_SOURCE_ID,
        speaker_namespace="beat2",
        source_group_namespace="beat2-official-semantic-event",
    )
    with np.load(frozen_qwen_cache, allow_pickle=False) as archive:
        required = {
            "clip_ids",
            "conditions",
            "fixed_split_assignments",
            "trajectory_sha256",
        }
        if not required.issubset(archive.files):
            raise ValueError("frozen Qwen cache fields are incomplete")
        clip_ids = archive["clip_ids"].astype(str)
        conditions = np.asarray(archive["conditions"], dtype=np.float32)
        splits = archive["fixed_split_assignments"].astype(str)
        trajectory_hashes = archive["trajectory_sha256"].astype(str)
    if (
        conditions.shape != (len(clip_ids), TEXT_LATENT_SLICE.stop - TEXT_LATENT_SLICE.start)
        or not np.isfinite(conditions).all()
        or len(set(clip_ids.tolist())) != len(clip_ids)
    ):
        raise ValueError("frozen Qwen cache content is invalid")
    index = {clip_id: row for row, clip_id in enumerate(clip_ids.tolist())}
    by_clip = {str(episode["clip_id"]): episode for episode in episodes}
    if set(index) != set(by_clip):
        raise ValueError("BEAT2 manifest and frozen Qwen cache clip sets differ")

    result: list[dict[str, Any]] = []
    for clip_id in sorted(by_clip):
        item = dict(by_clip[clip_id])
        row = index[clip_id]
        if (
            splits[row] != str(item.get("fixed_split_assignment"))
            or trajectory_hashes[row] != str(item.get("trajectory_sha256"))
        ):
            raise ValueError(f"{clip_id}: Qwen identity/split lineage changed")
        condition = np.zeros(KIMODO_V2_CONDITION_DIM, dtype=np.float32)
        condition[TEXT_LATENT_SLICE] = conditions[row]
        item["condition"] = condition
        item.pop("condition_cache_provenance", None)
        item["observation_confidence"] = np.ones_like(
            np.asarray(item["actions"], dtype=np.float32)
        )
        item["action_dim_mask"] = np.ones(ACTION_DIM, dtype=np.bool_)
        item["domain"] = "beat2_frozen_qwen"
        item["qwen_condition_variant"] = "frozen_base"
        item["oracle_style_condition_used"] = False
        item["duration_supervision_mask"] = False
        result.append(item)

    receipt = {
        "artifact_kind": "beat2_frozen_qwen_motion_loader_receipt_v8",
        "schema_version": SCHEMA_VERSION,
        "manifest": str(manifest),
        "manifest_sha256": manifest_hash,
        "foundation_condition_cache": str(foundation_condition_cache),
        "foundation_condition_cache_sha256": foundation_cache_hash,
        "frozen_qwen_cache": str(frozen_qwen_cache),
        "frozen_qwen_cache_sha256": qwen_cache_hash,
        "episode_count": len(result),
        "split_counts": dict(
            Counter(
                episode["fixed_split_assignment"] for episode in result
            )
        ),
        "condition_policy": (
            "zero_0_136_frozen_qwen_text_136_264_no_oracle_style_v1"
        ),
    }
    receipt["sha256"] = canonical_sha256(receipt)
    return result, receipt


def build_experimental_admission_contract(
    *,
    hanyang_loader_receipt: Mapping[str, Any],
    beat2_loader_receipt: Mapping[str, Any],
    hanyang_batch_receipt_path: str | Path,
    expected_hanyang_batch_receipt_sha256: str,
    hanyang_domain_fraction: float,
    human_review: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the isolated admission record; fail closed on emotion claims."""

    fraction = float(hanyang_domain_fraction)
    if not 0.0 < fraction <= HANYANG_DOMAIN_MAX_FRACTION:
        raise ValueError("Hanyang training share must be in (0, 0.0625]")
    receipt_path = Path(hanyang_batch_receipt_path).expanduser().resolve()
    reject_forbidden_dataset_marker(
        str(receipt_path), context="hanyang_batch_receipt"
    )
    observed_receipt_hash = sha256_file(receipt_path)
    if observed_receipt_hash != expected_hanyang_batch_receipt_sha256:
        raise ValueError("Hanyang batch receipt SHA256 mismatch")
    batch_receipt = _read_json(receipt_path)
    _validate_record_sha256(batch_receipt, context=str(receipt_path))
    legacy_registry = validate_data_source_registry_contract(
        batch_receipt.get("data_source_registry"),
        expected_role=EMOTION_CRITIC_ROLE,
        expected_dataset_sources=[HANYANG_EMOTIONAL_BODY_SOURCE_ID],
    )
    counts = ((batch_receipt.get("batch_status") or {}).get("counts") or {})
    if (
        batch_receipt.get("artifact_kind")
        != "hanyang_partial_18d_batch_retarget_receipt_v1"
        or batch_receipt.get("dataset_id") != HANYANG_EMOTIONAL_BODY_SOURCE_ID
        or batch_receipt.get("kimodo_accessed_or_used") is not False
        or (batch_receipt.get("admission") or {}).get(
            "generator_foundation_ingest_allowed"
        )
        is not False
        or (batch_receipt.get("batch_status") or {}).get("phase")
        != "complete"
        or (batch_receipt.get("artifact_sha256") or {}).get(
            "passed_manifest.jsonl"
        )
        != hanyang_loader_receipt.get(
            "upstream_passed_manifest_sha256"
        )
        or (counts.get("by_status") or {}).get("passed")
        != EXPECTED_HANYANG_TOTAL
        or {
            split: int(
                ((counts.get("by_split_and_status") or {}).get(split) or {}).get(
                    "passed", -1
                )
            )
            for split in SPLIT_NAMES
        }
        != EXPECTED_HANYANG_SPLIT_COUNTS
        or legacy_registry["sources"][0].get(
            "generator_foundation_allowed"
        )
        is not False
        or legacy_registry["sources"][0].get("isolation_domain")
        != "external_emotion_critic_only"
    ):
        raise ValueError("Hanyang retarget receipt is not a completed isolated pool")
    if not isinstance(human_review, Mapping):
        raise ValueError("human review contract is required")
    review_required = human_review.get("required")
    review_approved = human_review.get("approved")
    if review_required is not True or type(review_approved) is not bool:
        raise ValueError("human review gate is invalid")
    sample_bundle = human_review.get("sample_bundle")
    if not isinstance(sample_bundle, Mapping):
        raise ValueError("human review sample bundle must be hash-bound")
    if sample_bundle.get(
        "expected_review_manifest_sha256"
    ) != hanyang_loader_receipt.get("human_review_manifest_sha256"):
        raise ValueError(
            "human review manifest is not bound to the admitted Hanyang pool"
        )
    contract = {
        "artifact_kind": ADMISSION_ARTIFACT_KIND,
        "schema_version": SCHEMA_VERSION,
        "experiment_artifact_kind": EXPERIMENT_ARTIFACT_KIND,
        "source_whitelist": [
            BEAT2_FORMAL_SOURCE_ID,
            HANYANG_EMOTIONAL_BODY_SOURCE_ID,
        ],
        "hanyang_role": (
            "secondary_partial_motion_domain_experiment_not_foundation"
        ),
        "generator_foundation_allowed": False,
        "formal_release_eligible": False,
        "experimental_only": True,
        "no_seven_class_claim": True,
        "insufficient_heldout_emotion_coverage": True,
        "hanyang_emotion_supervision_enabled": False,
        "hanyang_style_supervision_enabled": False,
        "hanyang_semantic_supervision_enabled": False,
        "hanyang_duration_supervision_enabled": False,
        "hanyang_condition_policy": "all_zero_unconditional_v1",
        "hanyang_domain_fraction": fraction,
        "hanyang_domain_fraction_maximum": HANYANG_DOMAIN_MAX_FRACTION,
        "deny_policy": KIMODO_PERMANENT_DENY_POLICY,
        "human_review_required": True,
        "human_review_approved": review_approved,
        "training_gate": (
            "APPROVED" if review_approved else "HUMAN_REVIEW_BLOCKED"
        ),
        "human_review_sample_bundle": deepcopy(dict(sample_bundle)),
        "hanyang_batch_receipt": {
            "path": str(receipt_path),
            "sha256": observed_receipt_hash,
            "legacy_registry_snapshot_preserved": True,
            "legacy_registry_snapshot_sha256": legacy_registry["sha256"],
        },
        "hanyang_loader_receipt_sha256": hanyang_loader_receipt["sha256"],
        "beat2_loader_receipt_sha256": beat2_loader_receipt["sha256"],
    }
    assert_no_forbidden_data_lineage(contract, context="v8_admission")
    contract["sha256"] = canonical_sha256(contract)
    return contract


def split_expanded_episodes(
    beat2: Sequence[Mapping[str, Any]],
    hanyang: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    """Preserve both source-defined splits without reassigning a participant."""

    sources = {
        "beat2": [dict(episode) for episode in beat2],
        "hanyang": [dict(episode) for episode in hanyang],
    }
    splits: dict[str, dict[str, list[dict[str, Any]]]] = {
        name: {"beat2": [], "hanyang": []} for name in SPLIT_NAMES
    }
    group_splits: dict[str, set[str]] = {}
    clip_ids: set[str] = set()
    for source_name, episodes in sources.items():
        for episode in episodes:
            clip_id = str(episode["clip_id"])
            split = str(episode.get("fixed_split_assignment"))
            group = f"{source_name}:{episode.get('source_group_key')}"
            if clip_id in clip_ids or split not in splits:
                raise ValueError("expanded episode identity/split is invalid")
            clip_ids.add(clip_id)
            splits[split][source_name].append(dict(episode))
            group_splits.setdefault(group, set()).add(split)
    leaked = sorted(group for group, values in group_splits.items() if len(values) != 1)
    if leaked:
        raise ValueError(f"source-group split leakage detected: {leaked[:3]}")
    hanyang_counts = {
        split: len(splits[split]["hanyang"]) for split in SPLIT_NAMES
    }
    if hanyang_counts != EXPECTED_HANYANG_SPLIT_COUNTS:
        raise ValueError("Hanyang participant split counts changed")
    contract = {
        "artifact_kind": "hanyang_beat2_fixed_split_contract_v8",
        "schema_version": SCHEMA_VERSION,
        "assignment_policy": "preserve_source_fixed_group_splits_no_resplit_v1",
        "source_counts": {
            source: {
                split: len(splits[split][source])
                for split in SPLIT_NAMES
            }
            for source in sources
        },
        "source_group_leakage_count": 0,
        "clip_overlap_count": 0,
        "hanyang_participant_assignment": {
            "train": "1-21",
            "validation": "22-25",
            "test": "26-29",
        },
    }
    contract["sha256"] = canonical_sha256(contract)
    return splits, contract


def collate_confidence_weighted_18d(
    episodes: Sequence[Mapping[str, Any]],
    *,
    buckets: Sequence[int],
    action_stats: Mapping[str, Any],
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Pad, normalize, and retain per-frame/per-dimension confidence."""

    if not episodes:
        raise ValueError("cannot collate an empty confidence-weighted batch")
    frame_counts = [
        int(np.asarray(episode["actions"]).shape[0]) for episode in episodes
    ]
    candidates = sorted({int(value) for value in buckets if int(value) >= 3})
    required = max(frame_counts)
    bucket = next((value for value in candidates if value >= required), None)
    if bucket is None:
        bucket = int(math.ceil(required / 32.0) * 32)
    mean = np.asarray(action_stats["mean"], dtype=np.float32)
    std = np.asarray(action_stats["std"], dtype=np.float32)
    if (
        mean.shape != (ACTION_DIM,)
        or std.shape != (ACTION_DIM,)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or np.any(std <= 0.0)
    ):
        raise ValueError("foundation 18D action statistics are invalid")

    actions = np.zeros(
        (len(episodes), bucket, ACTION_DIM), dtype=np.float32
    )
    confidence = np.zeros_like(actions)
    frame_valid = np.zeros((len(episodes), bucket), dtype=np.bool_)
    conditions = np.empty(
        (len(episodes), KIMODO_V2_CONDITION_DIM), dtype=np.float32
    )
    durations = np.empty(len(episodes), dtype=np.float32)
    dim_mask = np.zeros((len(episodes), ACTION_DIM), dtype=np.bool_)
    for row, episode in enumerate(episodes):
        values = np.asarray(episode["actions"], dtype=np.float32)
        weights = np.asarray(
            episode.get("observation_confidence"), dtype=np.float32
        )
        condition = np.asarray(episode["condition"], dtype=np.float32)
        mask = np.asarray(
            episode.get("action_dim_mask"), dtype=np.bool_
        )
        count = int(values.shape[0])
        if (
            values.shape != (count, ACTION_DIM)
            or count < 3
            or weights.shape != values.shape
            or condition.shape != (KIMODO_V2_CONDITION_DIM,)
            or mask.shape != (ACTION_DIM,)
            or not np.isfinite(values).all()
            or not np.isfinite(weights).all()
            or not np.isfinite(condition).all()
            or np.any(weights < 0.0)
            or np.any(weights > 1.0)
            or np.any(weights[:, ~mask] != 0.0)
        ):
            raise ValueError(f"{episode.get('clip_id')}: invalid weighted episode")
        observed = weights > 0.0
        normalized = (values - mean[None, :]) / std[None, :]
        normalized[~observed] = 0.0
        actions[row, :count] = normalized
        confidence[row, :count] = weights
        frame_valid[row, :count] = True
        conditions[row] = condition
        dim_mask[row] = mask
        fps = float(episode.get("fps", SOURCE_FPS))
        duration = (count - 1) / fps
        declared = float(episode.get("duration_sec", duration))
        if (
            not math.isfinite(fps)
            or fps <= 0.0
            or not math.isclose(declared, duration, abs_tol=1e-6)
        ):
            raise ValueError(f"{episode.get('clip_id')}: invalid native duration")
        durations[row] = duration
    return {
        "actions": torch.as_tensor(actions, device=device),
        "conditions": torch.as_tensor(conditions, device=device),
        "observation_confidence": torch.as_tensor(confidence, device=device),
        "frame_valid_mask": torch.as_tensor(frame_valid, device=device),
        "action_dim_mask": torch.as_tensor(dim_mask, device=device),
        "durations_sec": torch.as_tensor(durations, device=device),
        "frame_counts": torch.as_tensor(frame_counts, device=device),
        "bucket_frames": torch.tensor(bucket, device=device),
    }


def derivative_confidence_weights(
    observation_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use the minimum confidence across every finite-difference stencil."""

    validated = validate_observation_weight(observation_weights)
    derivatives = derivative_observation_weights(validated)
    return (
        derivatives.velocity,
        derivatives.acceleration,
        derivatives.jerk,
    )


def _general_derivative_confidence_weights(
    observation_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adjacent minima for fully observed BEAT2 batches."""

    velocity = torch.minimum(
        observation_weights[:, 1:], observation_weights[:, :-1]
    )
    acceleration = torch.minimum(
        torch.minimum(
            observation_weights[:, 2:], observation_weights[:, 1:-1]
        ),
        observation_weights[:, :-2],
    )
    jerk = torch.minimum(
        torch.minimum(
            observation_weights[:, 3:], observation_weights[:, 2:-1]
        ),
        torch.minimum(
            observation_weights[:, 1:-2], observation_weights[:, :-3]
        ),
    )
    return velocity, acceleration, jerk


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if values.shape != weights.shape:
        weights = torch.broadcast_to(weights, values.shape)
    return weighted_masked_mean(values, weights.to(values.dtype))


def _derivatives(
    actions: torch.Tensor,
    durations: torch.Tensor,
    frame_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    intervals = (frame_counts.to(durations.dtype) - 1.0).clamp_min(1.0)
    dt = (durations / intervals).clamp_min(1e-4)[:, None, None]
    velocity = (actions[:, 1:] - actions[:, :-1]) / dt
    acceleration = (velocity[:, 1:] - velocity[:, :-1]) / dt
    jerk = (acceleration[:, 1:] - acceleration[:, :-1]) / dt
    return velocity, acceleration, jerk


def confidence_weighted_18d_objective(
    model: torch.nn.Module,
    actions: torch.Tensor,
    conditions: torch.Tensor,
    observation_confidence: torch.Tensor,
    durations: torch.Tensor,
    frame_valid_mask: torch.Tensor,
    *,
    loss_weights: Mapping[str, float],
    noise: torch.Tensor | None = None,
    flow_times: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    require_hanyang_partial_weights: bool = False,
) -> dict[str, torch.Tensor]:
    """Flow/position/derivative objective for partially observed trajectories."""

    if (
        actions.ndim != 3
        or actions.shape[-1] != ACTION_DIM
        or conditions.shape
        != (actions.shape[0], KIMODO_V2_CONDITION_DIM)
        or observation_confidence.shape != actions.shape
        or frame_valid_mask.shape != actions.shape[:2]
        or frame_valid_mask.dtype != torch.bool
        or durations.shape != (actions.shape[0],)
    ):
        raise ValueError("invalid confidence-weighted objective tensors")
    tensor_inputs = (conditions, observation_confidence, durations)
    if any(
        value.device != actions.device for value in tensor_inputs
    ) or frame_valid_mask.device != actions.device:
        raise ValueError("all objective tensors must share a device")
    if (
        not actions.is_floating_point()
        or conditions.dtype != actions.dtype
        or observation_confidence.dtype != actions.dtype
        or durations.dtype != actions.dtype
    ):
        raise ValueError("all numerical objective tensors must share a float dtype")
    if (
        not torch.isfinite(actions).all()
        or not torch.isfinite(conditions).all()
        or not torch.isfinite(observation_confidence).all()
        or not torch.isfinite(durations).all()
        or torch.any(observation_confidence < 0.0)
        or torch.any(observation_confidence > 1.0)
        or torch.any(durations <= 0.0)
    ):
        raise ValueError("non-finite or out-of-range objective input")
    invalid_seen = (~frame_valid_mask).to(torch.int64).cumsum(dim=1) > 0
    if torch.any(frame_valid_mask & invalid_seen):
        raise ValueError("frame_valid_mask must be a contiguous valid prefix")
    if torch.any(
        observation_confidence.masked_select(
            ~frame_valid_mask[:, :, None].expand_as(observation_confidence)
        )
        != 0.0
    ):
        raise ValueError("padding must have zero observation confidence")
    allowed = {"flow", "position", "velocity", "acceleration", "jerk"}
    if set(loss_weights) != allowed:
        raise ValueError(
            f"v8 loss weights must be exactly {sorted(allowed)}"
        )
    numeric_loss_weights = {
        name: float(value) for name, value in loss_weights.items()
    }
    if (
        any(
            not math.isfinite(value) or value < 0.0
            for value in numeric_loss_weights.values()
        )
        or sum(numeric_loss_weights.values()) <= 0.0
    ):
        raise ValueError("v8 loss weights must be finite, nonnegative, and nonzero")
    if require_hanyang_partial_weights:
        validate_observation_weight(
            observation_confidence, frame_valid_mask=frame_valid_mask
        )
    weights = observation_confidence * frame_valid_mask[:, :, None]
    observed = weights > 0.0
    frame_counts = frame_valid_mask.sum(dim=1)
    if torch.any(frame_counts < 4):
        raise ValueError("jerk objective requires at least four valid frames")
    if noise is None:
        noise = torch.zeros_like(actions)
        for row, count in enumerate(frame_counts.tolist()):
            noise[row, :count] = torch.randn(
                (count, ACTION_DIM),
                dtype=actions.dtype,
                device=actions.device,
                generator=generator,
            )
    elif (
        noise.shape != actions.shape
        or noise.dtype != actions.dtype
        or noise.device != actions.device
        or not torch.isfinite(noise).all()
    ):
        raise ValueError("explicit noise must exactly match actions")
    if flow_times is None:
        flow_times = torch.rand(
            actions.shape[0],
            dtype=actions.dtype,
            device=actions.device,
            generator=generator,
        )
    elif (
        flow_times.shape != (actions.shape[0],)
        or flow_times.dtype != actions.dtype
        or flow_times.device != actions.device
        or not torch.isfinite(flow_times).all()
        or torch.any(flow_times < 0.0)
        or torch.any(flow_times > 1.0)
    ):
        raise ValueError("explicit flow times must be [B] in [0,1]")
    noise = noise * observed
    actions = actions * observed
    x_t = (
        (1.0 - flow_times[:, None, None]) * noise
        + flow_times[:, None, None] * actions
    ) * observed
    target = (actions - noise) * observed
    predicted = forward_with_frame_mask(
        model, x_t, flow_times, conditions, frame_valid_mask
    )
    if predicted.shape != actions.shape or not torch.isfinite(predicted).all():
        raise FloatingPointError("generator produced invalid V8 flow output")
    reconstructed = (
        x_t + (1.0 - flow_times[:, None, None]) * predicted
    )
    if require_hanyang_partial_weights:
        velocity_weights, acceleration_weights, jerk_weights = (
            derivative_confidence_weights(weights)
        )
    else:
        velocity_weights, acceleration_weights, jerk_weights = (
            _general_derivative_confidence_weights(weights)
        )
    reconstructed_derivatives = _derivatives(
        reconstructed, durations, frame_counts
    )
    target_derivatives = _derivatives(actions, durations, frame_counts)
    losses = {
        "flow": _weighted_mean((predicted - target).square(), weights),
        "position": _weighted_mean(
            F.smooth_l1_loss(
                reconstructed, actions, reduction="none"
            ),
            weights,
        ),
        "velocity": _weighted_mean(
            F.smooth_l1_loss(
                reconstructed_derivatives[0],
                target_derivatives[0],
                reduction="none",
            ),
            velocity_weights,
        ),
        "acceleration": _weighted_mean(
            F.smooth_l1_loss(
                reconstructed_derivatives[1],
                target_derivatives[1],
                reduction="none",
            ),
            acceleration_weights,
        ),
        "jerk": _weighted_mean(
            F.smooth_l1_loss(
                reconstructed_derivatives[2],
                target_derivatives[2],
                reduction="none",
            ),
            jerk_weights,
        ),
    }
    total = sum(
        numeric_loss_weights[name] * value for name, value in losses.items()
    )
    return losses | {
        "total": total,
        "observed_weight_mean": weights.sum()
        / frame_valid_mask.sum().clamp_min(1)
        / ACTION_DIM,
    }


def action_normalizer_contract(
    action_stats: Mapping[str, Any],
    *,
    foundation_checkpoint: str | Path,
    foundation_checkpoint_sha256: str,
) -> dict[str, Any]:
    checkpoint_path = Path(foundation_checkpoint).expanduser().resolve()
    observed_checkpoint_hash = sha256_file(checkpoint_path)
    if observed_checkpoint_hash != foundation_checkpoint_sha256:
        raise ValueError("foundation checkpoint hash changed before normalizer bind")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    mean = torch.as_tensor(action_stats["mean"], dtype=torch.float32).cpu()
    std = torch.as_tensor(action_stats["std"], dtype=torch.float32).cpu()
    checkpoint_mean = torch.as_tensor(
        checkpoint["action_stats"]["mean"], dtype=torch.float32
    ).cpu()
    checkpoint_std = torch.as_tensor(
        checkpoint["action_stats"]["std"], dtype=torch.float32
    ).cpu()
    if (
        tuple(mean.shape) != (ACTION_DIM,)
        or tuple(std.shape) != (ACTION_DIM,)
        or not torch.isfinite(mean).all()
        or not torch.isfinite(std).all()
        or torch.any(std <= 0.0)
        or not torch.equal(mean, checkpoint_mean)
        or not torch.equal(std, checkpoint_std)
    ):
        raise ValueError("foundation action normalizer is invalid")
    arrays_hash = hashlib.sha256(
        mean.numpy().tobytes() + std.numpy().tobytes()
    ).hexdigest()
    contract = {
        "artifact_kind": NORMALIZER_ARTIFACT_KIND,
        "schema_version": SCHEMA_VERSION,
        "statistics_source": "foundation_checkpoint_train_only_no_refit_v1",
        "foundation_checkpoint": str(checkpoint_path),
        "foundation_checkpoint_sha256": observed_checkpoint_hash,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "arrays_sha256": arrays_hash,
        "hanyang_refit_performed": False,
    }
    assert_no_forbidden_data_lineage(contract, context="v8_normalizer")
    contract["sha256"] = canonical_sha256(contract)
    return contract


__all__ = [
    "ADMISSION_ARTIFACT_KIND",
    "DATA_CONTRACT_ARTIFACT_KIND",
    "EXPERIMENT_ARTIFACT_KIND",
    "EXPECTED_HANYANG_SPLIT_COUNTS",
    "HANYANG_DOMAIN_MAX_FRACTION",
    "NORMALIZER_ARTIFACT_KIND",
    "SCHEMA_VERSION",
    "action_normalizer_contract",
    "build_experimental_admission_contract",
    "canonical_sha256",
    "collate_confidence_weighted_18d",
    "confidence_weighted_18d_objective",
    "derivative_confidence_weights",
    "load_beat2_frozen_qwen_motion_episodes",
    "load_hanyang_partial_motion_episodes",
    "split_expanded_episodes",
]
