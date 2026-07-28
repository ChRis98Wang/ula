"""Research-only XEM -> BEAT2 weak-emotion domain-shift diagnostic.

The bridge deliberately avoids a skeleton mapping.  XEM contributes only the
per-joint norm of its official angular-velocity vectors.  BEAT2 contributes
only absolute finite-difference speeds from its 18 controller angles.  Both
are reduced to anonymous, entity-count-independent sequence statistics before
classification.

This module is not imported by a generator or trainer and emits no
generator-compatible artifact.  BEAT2 filename-protocol emotions are weak
intended-performance labels, not robot-observable emotion truth.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from upper_body_skeleton import xem_emotion_research as xem


SCHEMA_VERSION = "1.0.0"
CONFIG_KIND = "xem_to_beat2_weak_emotion_domain_shift_config_v1"
REPORT_KIND = "xem_to_beat2_weak_emotion_domain_shift_report_v1"
AUDIT_KIND = "xem_to_beat2_weak_emotion_domain_shift_audit_v1"
RESEARCH_SCOPE = "isolated_cross_dataset_emotion_screening_research_only"
COMMON_FEATURE_POLICY = (
    "anonymous_angular_activity_statistics_no_skeleton_mapping_no_euler_mapping_v1"
)
TARGET_LABEL_ROLE = (
    "official_filename_protocol_intended_expression_weak_evaluation_only"
)
EMOTIONS = ("angry", "neutral", "happy", "sad")
SPLITS = ("train", "validation", "test")
RESAMPLED_FRAMES = 64
AMPLITUDE_FEATURE_COUNT = 10
FORBIDDEN_DATASET_MARKER = "kimodo"

CONFIG_KEYS = {
    "schema_version",
    "artifact_kind",
    "research_scope",
    "xem_archive_path",
    "expected_xem_archive_sha256",
    "xem_metadata_path",
    "expected_xem_metadata_sha256",
    "xem_mat_member",
    "xem_mat_variable",
    "beat2_manifest_path",
    "expected_beat2_manifest_sha256",
    "overlapping_emotions",
    "ridge_alphas",
    "bootstrap_replicates",
    "seed",
    "output_root",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_forbidden(value: object, *, context: str = "$") -> None:
    if isinstance(value, str):
        if FORBIDDEN_DATASET_MARKER in value.casefold():
            raise ValueError(f"{context} contains a forbidden dataset marker")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_forbidden(str(key), context=f"{context}.<key>")
            _reject_forbidden(child, context=f"{context}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden(child, context=f"{context}[{index}]")


def _resolve_path(value: object, *, base: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    _reject_forbidden(str(path), context=field)
    return path


def load_config(config_path: Path, *, require_new_output: bool = True) -> dict[str, Any]:
    config_path = config_path.resolve()
    _reject_forbidden(str(config_path), context="config_path")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load bridge config {config_path}: {error}") from error
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        keys = set(config) if isinstance(config, dict) else set()
        raise ValueError(
            "bridge config keys changed; "
            f"missing={sorted(CONFIG_KEYS - keys)}, unknown={sorted(keys - CONFIG_KEYS)}"
        )
    _reject_forbidden(config, context="config")
    if (
        config["schema_version"] != SCHEMA_VERSION
        or config["artifact_kind"] != CONFIG_KIND
        or config["research_scope"] != RESEARCH_SCOPE
    ):
        raise ValueError("invalid XEM/BEAT2 bridge config contract")
    if config["overlapping_emotions"] != list(EMOTIONS):
        raise ValueError("overlapping_emotions must preserve the fixed class order")
    for field in (
        "expected_xem_archive_sha256",
        "expected_xem_metadata_sha256",
        "expected_beat2_manifest_sha256",
    ):
        if not _is_sha256(config[field]):
            raise ValueError(f"{field} is not a SHA256")
    alphas = config["ridge_alphas"]
    if (
        not isinstance(alphas, list)
        or not alphas
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in alphas
        )
    ):
        raise ValueError("ridge_alphas must be positive finite numbers")
    if len({float(value) for value in alphas}) != len(alphas):
        raise ValueError("ridge_alphas must be unique")
    replicates = config["bootstrap_replicates"]
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates < 200
    ):
        raise ValueError("bootstrap_replicates must be an integer >= 200")
    seed = config["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    for field in ("xem_mat_member", "xem_mat_variable"):
        if not isinstance(config[field], str) or not config[field]:
            raise ValueError(f"{field} must be a non-empty string")

    base = config_path.parent
    resolved = dict(config)
    for field in (
        "xem_archive_path",
        "xem_metadata_path",
        "beat2_manifest_path",
        "output_root",
    ):
        resolved[field] = _resolve_path(config[field], base=base, field=field)
    output_root: Path = resolved["output_root"]
    if "external_emotion_research" not in output_root.parts:
        raise ValueError("bridge output must remain under external_emotion_research")
    if require_new_output and output_root.exists():
        raise FileExistsError(f"refusing to overwrite bridge output: {output_root}")
    return resolved


def _resample_columns(values: np.ndarray, frames: int = RESAMPLED_FRAMES) -> np.ndarray:
    source = np.linspace(0.0, 1.0, values.shape[0], dtype=np.float64)
    target = np.linspace(0.0, 1.0, frames, dtype=np.float64)
    return np.stack(
        [np.interp(target, source, values[:, column]) for column in range(values.shape[1])],
        axis=1,
    )


def _fractional_top_share(values: np.ndarray, fraction: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    total = float(ordered.sum())
    if total <= 1e-12:
        return 0.0
    target = fraction * ordered.size
    whole = int(math.floor(target))
    remainder = target - whole
    selected = float(ordered[:whole].sum())
    if whole < ordered.size:
        selected += remainder * float(ordered[whole])
    return selected / total


def common_angular_activity_features(
    angular_speed: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Return anonymous features shared by vector- and scalar-joint systems.

    ``angular_speed`` is frames x anonymous motion entities.  Entity ordering,
    names, axes, absolute poses, segment Euler angles, and skeleton topology
    are intentionally unavailable to this function.
    """

    speed = np.asarray(angular_speed, dtype=np.float64)
    if (
        speed.ndim != 2
        or speed.shape[0] < 8
        or speed.shape[1] < 4
        or not np.isfinite(speed).all()
        or np.any(speed < 0.0)
    ):
        raise ValueError("angular_speed must be finite non-negative [frames>=8, entities>=4]")

    values: list[float] = []
    names: list[str] = []

    flattened = speed.reshape(-1)
    for quantile in (0.50, 0.75, 0.90, 0.95, 0.99):
        values.append(float(np.log1p(np.quantile(flattened, quantile))))
        names.append(f"amplitude_log1p_speed_q{int(quantile * 100):02d}")
    values.append(float(np.log1p(np.sqrt(np.mean(flattened**2)))))
    names.append("amplitude_log1p_speed_rms")

    entity_rms = np.sqrt(np.mean(speed**2, axis=0))
    for quantile in (0.25, 0.50, 0.75, 0.90):
        values.append(float(np.log1p(np.quantile(entity_rms, quantile))))
        names.append(f"amplitude_log1p_entity_rms_q{int(quantile * 100):02d}")

    entity_energy = np.sum(speed**2, axis=0)
    energy_total = float(entity_energy.sum())
    if energy_total <= 1e-12:
        probabilities = np.full(speed.shape[1], 1.0 / speed.shape[1])
    else:
        probabilities = entity_energy / energy_total
    entropy = float(
        -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)))
        / math.log(speed.shape[1])
    )
    values.extend(
        [
            entropy,
            float(np.exp(entropy * math.log(speed.shape[1])) / speed.shape[1]),
            _fractional_top_share(entity_energy, 0.10),
            _fractional_top_share(entity_energy, 0.25),
            _fractional_top_share(entity_energy, 0.50),
        ]
    )
    names.extend(
        [
            "participation_normalized_entropy",
            "participation_effective_entity_fraction",
            "participation_top_10pct_energy_share",
            "participation_top_25pct_energy_share",
            "participation_top_50pct_energy_share",
        ]
    )

    frame_rms = np.sqrt(np.mean(speed**2, axis=1, keepdims=True))
    profile = _resample_columns(frame_rms)[:, 0]
    scale = max(float(np.quantile(profile, 0.90)), 1e-12)
    profile = profile / scale
    for quantile in (0.10, 0.25, 0.50, 0.75, 0.90):
        values.append(float(np.quantile(profile, quantile)))
        names.append(f"profile_q{int(quantile * 100):02d}_over_q90")
    values.extend(
        [
            float(np.std(profile)),
            float(np.max(profile)),
            float(np.mean(profile > 0.25)),
            float(np.mean(profile > 0.50)),
            float(np.mean(profile > 0.75)),
        ]
    )
    names.extend(
        [
            "profile_std_over_q90",
            "profile_peak_over_q90",
            "profile_duty_above_025_q90",
            "profile_duty_above_050_q90",
            "profile_duty_above_075_q90",
        ]
    )

    first = np.abs(np.diff(profile))
    second = np.abs(np.diff(profile, n=2))
    for prefix, derivative in (("profile_delta", first), ("profile_delta2", second)):
        values.extend(
            [
                float(np.median(derivative)),
                float(np.quantile(derivative, 0.90)),
                float(np.sqrt(np.mean(derivative**2))),
                float(np.max(derivative)),
            ]
        )
        names.extend(
            [
                f"{prefix}_median",
                f"{prefix}_q90",
                f"{prefix}_rms",
                f"{prefix}_max",
            ]
        )

    spectrum = np.abs(np.fft.rfft(profile - np.mean(profile))) ** 2
    spectrum = spectrum[1:]
    power = float(spectrum.sum())
    if power <= 1e-12:
        fractions = np.zeros(3, dtype=np.float64)
        spectral_entropy = 0.0
    else:
        fractions = np.asarray(
            [
                spectrum[:2].sum(),
                spectrum[2:8].sum(),
                spectrum[8:].sum(),
            ],
            dtype=np.float64,
        ) / power
        normalized = spectrum / power
        spectral_entropy = float(
            -np.sum(normalized * np.log(np.maximum(normalized, 1e-300)))
            / math.log(spectrum.size)
        )
    values.extend([float(value) for value in fractions] + [spectral_entropy])
    names.extend(
        [
            "spectrum_normalized_low_fraction_bins_1_2",
            "spectrum_normalized_mid_fraction_bins_3_8",
            "spectrum_normalized_high_fraction_bins_9_plus",
            "spectrum_normalized_entropy",
        ]
    )

    entity_profiles = _resample_columns(speed)
    varying = np.std(entity_profiles, axis=0) > 1e-10
    entity_profiles = entity_profiles[:, varying]
    if entity_profiles.shape[1] < 2:
        correlations = np.zeros(1, dtype=np.float64)
    else:
        matrix = np.corrcoef(entity_profiles, rowvar=False)
        correlations = matrix[np.triu_indices(matrix.shape[0], k=1)]
        correlations = correlations[np.isfinite(correlations)]
        if correlations.size == 0:
            correlations = np.zeros(1, dtype=np.float64)
    values.extend(
        [
            float(np.quantile(correlations, 0.10)),
            float(np.median(correlations)),
            float(np.quantile(correlations, 0.90)),
            float(np.mean(np.abs(correlations))),
        ]
    )
    names.extend(
        [
            "coordination_pairwise_correlation_q10",
            "coordination_pairwise_correlation_q50",
            "coordination_pairwise_correlation_q90",
            "coordination_pairwise_absolute_correlation_mean",
        ]
    )

    result = np.asarray(values, dtype=np.float64)
    if (
        result.shape != (41,)
        or len(names) != 41
        or AMPLITUDE_FEATURE_COUNT != 10
        or not np.isfinite(result).all()
    ):
        raise AssertionError("common angular activity feature contract changed")
    return result.astype(np.float32), names


def _stable_row_sha256(row: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(row),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_beat2_weak_evaluation_features(
    manifest_path: Path,
    *,
    expected_sha256: str,
) -> tuple[list[dict[str, Any]], np.ndarray, list[str], dict[str, Any]]:
    """Load only the four overlapping intended-label classes from BEAT2."""

    manifest_path = manifest_path.resolve()
    _reject_forbidden(str(manifest_path), context="beat2_manifest_path")
    if not manifest_path.is_file() or sha256_file(manifest_path) != expected_sha256:
        raise ValueError("BEAT2 weak-evaluation manifest SHA256 mismatch")

    rows: list[dict[str, Any]] = []
    features: list[np.ndarray] = []
    names: list[str] | None = None
    all_manifest_rows = 0
    excluded_emotions: Counter[str] = Counter()
    verified_csv_hashes: set[str] = set()
    for line_number, raw in enumerate(manifest_path.open("rb"), 1):
        payload = raw[:-1] if raw.endswith(b"\n") else raw
        if not payload.strip():
            continue
        all_manifest_rows += 1
        try:
            source = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid BEAT2 manifest line {line_number}") from error
        if not isinstance(source, dict):
            raise ValueError(f"BEAT2 manifest line {line_number} is not an object")
        _reject_forbidden(source, context=f"beat2_manifest[{line_number}]")
        emotion = str(source.get("emotion_id", ""))
        if emotion not in EMOTIONS:
            excluded_emotions[emotion] += 1
            continue
        split = str(source.get("fixed_split_assignment", ""))
        speaker = str(source.get("speaker_key", ""))
        source_group = str(source.get("source_group_key", ""))
        motion = source.get("motion_18d")
        if (
            source.get("dataset") != "BEAT2"
            or source.get("accepted_for_training") is not True
            or split not in SPLITS
            or not speaker
            or not source_group
            or source.get("emotion_label_source")
            != "official_beat2_filename_protocol"
            or source.get("emotion_supervision_mask") is not False
            or source.get("emotion_conditioning_mask") is not False
            or not isinstance(motion, Mapping)
            or motion.get("action_dim") != 18
            or motion.get("quality_gate", {}).get("passed") is not True
        ):
            raise ValueError(f"line {line_number}: invalid BEAT2 weak-evaluation row")
        csv_path = Path(str(motion.get("safe_csv", ""))).resolve()
        _reject_forbidden(str(csv_path), context=f"line {line_number}.safe_csv")
        csv_sha = str(motion.get("safe_csv_sha256", ""))
        if (
            not csv_path.is_file()
            or not _is_sha256(csv_sha)
            or sha256_file(csv_path) != csv_sha
        ):
            raise ValueError(f"line {line_number}: BEAT2 controller CSV mismatch")
        verified_csv_hashes.add(csv_sha)
        try:
            trajectory = np.loadtxt(
                csv_path,
                delimiter=",",
                skiprows=1,
                dtype=np.float64,
                ndmin=2,
            )
        except (OSError, ValueError) as error:
            raise ValueError(f"line {line_number}: cannot read controller CSV") from error
        frames = int(motion.get("frames", -1))
        fps = float(motion.get("fps", 0.0))
        if (
            trajectory.shape != (frames, 18)
            or frames < 9
            or not np.isfinite(trajectory).all()
            or not math.isfinite(fps)
            or fps <= 0.0
        ):
            raise ValueError(f"line {line_number}: invalid controller trajectory")
        angular_speed = np.abs(np.diff(trajectory, axis=0)) * fps
        feature, current_names = common_angular_activity_features(angular_speed)
        if names is None:
            names = current_names
        elif names != current_names:
            raise AssertionError("BEAT2 common feature names changed")
        features.append(feature)
        rows.append(
            {
                "sample_id": str(source.get("clip_id")),
                "split": split,
                "speaker_id": speaker,
                "source_group_id": source_group,
                "emotion_id": emotion,
                "label_role": TARGET_LABEL_ROLE,
                "human_confirmed_robot_observable": False,
                "emotion_supervision_mask": False,
                "controller_csv_sha256": csv_sha,
                "source_record_sha256": _stable_row_sha256(source),
            }
        )
    if not rows or names is None:
        raise ValueError("BEAT2 weak-evaluation overlap is empty")

    sample_ids = [row["sample_id"] for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("duplicate BEAT2 sample IDs in weak evaluation")
    for group_field in ("speaker_id", "source_group_id"):
        assignments: defaultdict[str, set[str]] = defaultdict(set)
        for row in rows:
            assignments[str(row[group_field])].add(str(row["split"]))
        leaked = sorted(key for key, value in assignments.items() if len(value) != 1)
        if leaked:
            raise ValueError(f"BEAT2 {group_field} crosses fixed splits: {leaked[:8]}")
    source_emotions: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        source_emotions[str(row["source_group_id"])].add(str(row["emotion_id"]))
    inconsistent = sorted(
        key for key, emotions in source_emotions.items() if len(emotions) != 1
    )
    if inconsistent:
        raise ValueError(
            "BEAT2 source groups have inconsistent intended emotions: "
            f"{inconsistent[:8]}"
        )
    matrix = np.stack(features).astype(np.float32)
    receipt = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": expected_sha256,
        "all_manifest_rows": all_manifest_rows,
        "overlap_rows": len(rows),
        "excluded_emotion_counts": dict(sorted(excluded_emotions.items())),
        "verified_controller_csv_records": len(rows),
        "unique_controller_csv_sha256": len(verified_csv_hashes),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "emotion_counts": dict(
            sorted(Counter(row["emotion_id"] for row in rows).items())
        ),
        "speaker_counts_by_split": {
            split: len({row["speaker_id"] for row in rows if row["split"] == split})
            for split in SPLITS
        },
        "source_group_counts_by_split": {
            split: len(
                {row["source_group_id"] for row in rows if row["split"] == split}
            )
            for split in SPLITS
        },
        "speaker_disjoint": True,
        "source_group_disjoint": True,
        "label_role": TARGET_LABEL_ROLE,
        "human_confirmed_robot_observable_labels": 0,
    }
    return rows, matrix, names, receipt


def load_xem_common_features(
    archive_path: Path,
    *,
    expected_archive_sha256: str,
    metadata_path: Path,
    expected_metadata_sha256: str,
    mat_member: str,
    mat_variable: str,
) -> tuple[list[dict[str, Any]], np.ndarray, list[str], dict[str, Any]]:
    """Load official XEM and reduce it without any skeleton/robot mapping."""

    metadata = xem.validate_official_metadata(
        metadata_path,
        expected_sha256=expected_metadata_sha256,
    )
    cells, inspection = xem._load_xem_cells(  # type: ignore[attr-defined]
        archive_path,
        mat_member=mat_member,
        mat_variable=mat_variable,
    )
    if inspection["archive"]["sha256"] != expected_archive_sha256:
        raise ValueError("XEM archive SHA256 mismatch")
    rows: list[dict[str, Any]] = []
    features: list[np.ndarray] = []
    names: list[str] | None = None
    for action_zero in range(5):
        for participant_zero in range(10):
            participant = participant_zero + 1
            split = next(
                name
                for name, participants in xem.DEFAULT_PARTICIPANT_SPLIT.items()
                if participant in participants
            )
            for example_zero in range(20):
                matrix = xem._sequence_matrix(  # type: ignore[attr-defined]
                    cells[action_zero, participant_zero, example_zero],
                    sample_id=(
                        f"xem_a{action_zero + 1:02d}_p{participant:02d}_"
                        f"e{example_zero + 1:02d}"
                    ),
                )
                angular_vectors = matrix[:, 139:208].reshape(matrix.shape[0], 23, 3)
                angular_speed = np.linalg.norm(angular_vectors, axis=2)
                feature, current_names = common_angular_activity_features(angular_speed)
                if names is None:
                    names = current_names
                elif names != current_names:
                    raise AssertionError("XEM common feature names changed")
                emotion = EMOTIONS[example_zero // 5]
                rows.append(
                    {
                        "sample_id": (
                            f"xem_a{action_zero + 1:02d}_p{participant:02d}_"
                            f"e{example_zero + 1:02d}"
                        ),
                        "split": split,
                        "participant_id": f"xem_participant_{participant:02d}",
                        "action_id": xem.ACTIONS[action_zero],
                        "emotion_id": emotion,
                        "label_role": (
                            "official_xem_protocol_intended_expression_research_only"
                        ),
                        "human_confirmed_robot_observable": False,
                    }
                )
                features.append(feature)
    if len(rows) != 1_000 or names is None:
        raise AssertionError("XEM common feature inventory changed")
    matrix = np.stack(features).astype(np.float32)
    receipt = {
        "archive": inspection["archive"],
        "official_metadata": metadata,
        "mat_member": inspection["mat_member"]["name"],
        "mat_variable": inspection["selected_mat_variable"]["name"],
        "rows": len(rows),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "participant_disjoint": True,
        "skeleton_mapping_used": False,
        "segment_euler_mapping_used": False,
        "input_columns": "official_joint_angular_velocity_23x3_only",
    }
    return rows, matrix, names, receipt


def robust_neutral_reference(
    features: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mask = np.asarray(
        [
            row["split"] == split and row["emotion_id"] == "neutral"
            for row in rows
        ],
        dtype=bool,
    )
    if int(mask.sum()) < 20:
        raise ValueError(f"{split} neutral reference has fewer than 20 examples")
    neutral = np.asarray(features[mask], dtype=np.float64)
    center = np.median(neutral, axis=0)
    mad = np.median(np.abs(neutral - center), axis=0) * 1.4826
    q25, q75 = np.quantile(neutral, [0.25, 0.75], axis=0)
    iqr_scale = (q75 - q25) / 1.349
    std = np.std(neutral, axis=0)
    scale = np.where(mad > 1e-8, mad, np.where(iqr_scale > 1e-8, iqr_scale, std))
    constant = scale <= 1e-8
    scale[constant] = 1.0
    speakers = sorted(
        {
            str(row.get("speaker_id", row.get("participant_id", "")))
            for row, keep in zip(rows, mask, strict=True)
            if keep
        }
    )
    return center, scale, {
        "source_split": split,
        "emotion": "neutral",
        "weak_label_used": any(
            row.get("label_role") == TARGET_LABEL_ROLE
            for row, keep in zip(rows, mask, strict=True)
            if keep
        ),
        "samples": int(mask.sum()),
        "groups": len(speakers),
        "constant_feature_count": int(constant.sum()),
    }


def apply_reference(
    features: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    result = (np.asarray(features, dtype=np.float64) - center) / scale
    if not np.isfinite(result).all():
        raise ValueError("neutral calibration produced NaN/Inf")
    return result


def _labels(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    lookup = {emotion: index for index, emotion in enumerate(EMOTIONS)}
    return np.asarray([lookup[str(row["emotion_id"])] for row in rows], dtype=np.int64)


def _fit_ridge(features: np.ndarray, labels: np.ndarray, alpha: float) -> np.ndarray:
    targets = np.eye(len(EMOTIONS), dtype=np.float64)[labels]
    augmented = np.concatenate(
        [features, np.ones((features.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    penalty = np.eye(augmented.shape[1], dtype=np.float64) * alpha
    penalty[-1, -1] = 0.0
    left = augmented.T @ augmented + penalty
    right = augmented.T @ targets
    try:
        return np.linalg.solve(left, right)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(left, right, rcond=None)[0]


def _scores(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    augmented = np.concatenate(
        [features, np.ones((features.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    return augmented @ weights


def classification_metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Any]:
    if truth.ndim != 1 or truth.shape != predicted.shape or truth.size == 0:
        raise ValueError("classification inputs are invalid")
    confusion = np.zeros((len(EMOTIONS), len(EMOTIONS)), dtype=np.int64)
    for expected, actual in zip(truth, predicted, strict=True):
        confusion[int(expected), int(actual)] += 1
    recalls: list[float] = []
    f1s: list[float] = []
    per_class: dict[str, dict[str, float | int]] = {}
    for index, emotion in enumerate(EMOTIONS):
        true_positive = int(confusion[index, index])
        support = int(confusion[index].sum())
        predicted_count = int(confusion[:, index].sum())
        recall = true_positive / support if support else 0.0
        precision = true_positive / predicted_count if predicted_count else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        recalls.append(recall)
        f1s.append(f1)
        per_class[emotion] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    distribution = np.bincount(predicted, minlength=len(EMOTIONS))
    return {
        "samples": int(truth.size),
        "accuracy": float(np.mean(truth == predicted)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "minimum_class_recall": float(min(recalls)),
        "confusion_matrix": confusion.tolist(),
        "class_order": list(EMOTIONS),
        "per_class": per_class,
        "prediction_distribution": {
            emotion: int(distribution[index])
            for index, emotion in enumerate(EMOTIONS)
        },
        "maximum_prediction_fraction": float(distribution.max() / truth.size),
    }


def _standardize_train(
    train: np.ndarray,
    *others: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0, dtype=np.float64)
    scale = np.std(train, axis=0, dtype=np.float64)
    scale[scale < 1e-10] = 1.0
    return (
        (train - mean) / scale,
        [(value - mean) / scale for value in others],
        mean,
        scale,
    )


def train_xem_classifier(
    features: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    ridge_alphas: Sequence[float],
    feature_indices: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    split_values = np.asarray([row["split"] for row in rows])
    labels = _labels(rows)
    if feature_indices is not None:
        features = features[:, feature_indices]
    masks = {split: split_values == split for split in SPLITS}
    train, others, mean, scale = _standardize_train(
        np.asarray(features[masks["train"]], dtype=np.float64),
        np.asarray(features[masks["validation"]], dtype=np.float64),
        np.asarray(features[masks["test"]], dtype=np.float64),
    )
    validation, test = others
    train_y = labels[masks["train"]]
    validation_y = labels[masks["validation"]]
    test_y = labels[masks["test"]]
    candidates: list[dict[str, Any]] = []
    weights_by_alpha: dict[float, np.ndarray] = {}
    for raw_alpha in ridge_alphas:
        alpha = float(raw_alpha)
        weights = _fit_ridge(train, train_y, alpha)
        predicted = np.argmax(_scores(validation, weights), axis=1)
        metrics = classification_metrics(validation_y, predicted)
        candidates.append({"alpha": alpha, "validation": metrics})
        weights_by_alpha[alpha] = weights
    selected = max(
        candidates,
        key=lambda item: (
            item["validation"]["macro_f1"],
            item["validation"]["balanced_accuracy"],
            -item["alpha"],
        ),
    )
    alpha = float(selected["alpha"])
    weights = weights_by_alpha[alpha]
    report = {
        "selected_alpha": alpha,
        "alpha_candidates": candidates,
        "train_metrics": classification_metrics(
            train_y,
            np.argmax(_scores(train, weights), axis=1),
        ),
        "validation_metrics": selected["validation"],
        "test_metrics": classification_metrics(
            test_y,
            np.argmax(_scores(test, weights), axis=1),
        ),
        "validation_used_only_for_ridge_alpha": True,
        "test_used_for_selection": False,
    }
    model = {
        "weights": weights,
        "feature_mean": mean,
        "feature_scale": scale,
        "feature_indices": (
            np.arange(features.shape[1], dtype=np.int64)
            if feature_indices is None
            else np.asarray(feature_indices, dtype=np.int64)
        ),
        "selected_alpha": np.asarray(alpha),
    }
    return report, model


def apply_classifier(features: np.ndarray, model: Mapping[str, np.ndarray]) -> np.ndarray:
    indices = np.asarray(model["feature_indices"], dtype=np.int64)
    selected = np.asarray(features, dtype=np.float64)[:, indices]
    normalized = (
        selected - np.asarray(model["feature_mean"], dtype=np.float64)
    ) / np.asarray(model["feature_scale"], dtype=np.float64)
    return _scores(normalized, np.asarray(model["weights"], dtype=np.float64))


def _aggregate_scores(
    scores: np.ndarray,
    truth: np.ndarray,
    group_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    grouped: defaultdict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(group_ids):
        grouped[str(group)].append(index)
    group_truth: list[int] = []
    group_scores: list[np.ndarray] = []
    ordered_groups = sorted(grouped)
    for group in ordered_groups:
        indices = np.asarray(grouped[group], dtype=np.int64)
        labels = set(int(value) for value in truth[indices])
        if len(labels) != 1:
            raise ValueError(f"group {group} spans multiple emotion labels")
        group_truth.append(next(iter(labels)))
        group_scores.append(np.mean(scores[indices], axis=0))
    return (
        np.asarray(group_truth, dtype=np.int64),
        np.stack(group_scores),
        ordered_groups,
    )


def cluster_bootstrap_balanced_accuracy(
    truth: np.ndarray,
    predicted: np.ndarray,
    group_ids: Sequence[str],
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    grouped: defaultdict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(group_ids):
        grouped[str(group)].append(index)
    groups = sorted(grouped)
    if len(groups) < 2:
        raise ValueError("cluster bootstrap requires at least two groups")
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(replicates):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate(
            [np.asarray(grouped[str(group)], dtype=np.int64) for group in sampled]
        )
        values.append(
            classification_metrics(truth[indices], predicted[indices])[
                "balanced_accuracy"
            ]
        )
    array = np.asarray(values, dtype=np.float64)
    return {
        "replicates": replicates,
        "groups": len(groups),
        "median": float(np.median(array)),
        "lower_95": float(np.quantile(array, 0.025)),
        "upper_95": float(np.quantile(array, 0.975)),
    }


def evaluate_beat2(
    scores: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    truth = _labels(rows)
    predicted = np.argmax(scores, axis=1)
    split_values = np.asarray([row["split"] for row in rows])
    result: dict[str, Any] = {}
    for offset, split in enumerate(("validation", "test")):
        mask = split_values == split
        split_truth = truth[mask]
        split_scores = scores[mask]
        split_predicted = predicted[mask]
        split_rows = [row for row, keep in zip(rows, mask, strict=True) if keep]
        source_truth, source_scores, source_groups = _aggregate_scores(
            split_scores,
            split_truth,
            [str(row["source_group_id"]) for row in split_rows],
        )
        speaker_metrics: dict[str, Any] = {}
        for speaker in sorted({str(row["speaker_id"]) for row in split_rows}):
            speaker_mask = np.asarray(
                [row["speaker_id"] == speaker for row in split_rows],
                dtype=bool,
            )
            speaker_metrics[speaker] = classification_metrics(
                split_truth[speaker_mask],
                split_predicted[speaker_mask],
            )
        result[split] = {
            "clip_level": classification_metrics(split_truth, split_predicted),
            "source_group_level": classification_metrics(
                source_truth,
                np.argmax(source_scores, axis=1),
            ),
            "source_groups": len(source_groups),
            "by_speaker": speaker_metrics,
            "speaker_cluster_bootstrap": cluster_bootstrap_balanced_accuracy(
                split_truth,
                split_predicted,
                [str(row["speaker_id"]) for row in split_rows],
                replicates=bootstrap_replicates,
                seed=seed + offset,
            ),
        }
    return result


def _calibration_sensitivity(
    beat2_features: np.ndarray,
    beat2_rows: Sequence[Mapping[str, Any]],
    *,
    xem_model: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    training_speakers = sorted(
        {
            str(row["speaker_id"])
            for row in beat2_rows
            if row["split"] == "train" and row["emotion_id"] == "neutral"
        }
    )
    test_mask = np.asarray([row["split"] == "test" for row in beat2_rows])
    test_rows = [row for row, keep in zip(beat2_rows, test_mask, strict=True) if keep]
    values: list[dict[str, Any]] = []
    for omitted in training_speakers:
        calibration_rows = [
            dict(row)
            for row in beat2_rows
            if not (
                row["split"] == "train"
                and row["emotion_id"] == "neutral"
                and row["speaker_id"] == omitted
            )
        ]
        keep = np.asarray(
            [
                not (
                    row["split"] == "train"
                    and row["emotion_id"] == "neutral"
                    and row["speaker_id"] == omitted
                )
                for row in beat2_rows
            ],
            dtype=bool,
        )
        center, scale, receipt = robust_neutral_reference(
            beat2_features[keep],
            calibration_rows,
            split="train",
        )
        calibrated = apply_reference(beat2_features[test_mask], center, scale)
        scores = apply_classifier(calibrated, xem_model)
        truth = _labels(test_rows)
        group_truth, group_scores, _ = _aggregate_scores(
            scores,
            truth,
            [str(row["source_group_id"]) for row in test_rows],
        )
        metrics = classification_metrics(
            group_truth,
            np.argmax(group_scores, axis=1),
        )
        values.append(
            {
                "omitted_train_neutral_speaker": omitted,
                "neutral_samples": receipt["samples"],
                "test_source_group_balanced_accuracy": metrics[
                    "balanced_accuracy"
                ],
            }
        )
    accuracies = np.asarray(
        [row["test_source_group_balanced_accuracy"] for row in values],
        dtype=np.float64,
    )
    return {
        "policy": "leave_one_beat2_train_neutral_speaker_out",
        "validation_or_test_labels_used_for_calibration": False,
        "runs": values,
        "minimum": float(accuracies.min()),
        "maximum": float(accuracies.max()),
        "range": float(accuracies.max() - accuracies.min()),
        "standard_deviation": float(np.std(accuracies)),
    }


def gate_decision(
    xem_report: Mapping[str, Any],
    beat2_report: Mapping[str, Any],
    calibration_sensitivity: Mapping[str, Any],
) -> dict[str, Any]:
    validation = beat2_report["validation"]
    test = beat2_report["test"]
    checks = {
        "xem_test_balanced_accuracy_at_least_0_50": (
            xem_report["test_metrics"]["balanced_accuracy"] >= 0.50
        ),
        "beat2_validation_source_group_balanced_accuracy_at_least_0_40": (
            validation["source_group_level"]["balanced_accuracy"] >= 0.40
        ),
        "beat2_test_source_group_balanced_accuracy_at_least_0_40": (
            test["source_group_level"]["balanced_accuracy"] >= 0.40
        ),
        "beat2_validation_source_group_macro_f1_at_least_0_35": (
            validation["source_group_level"]["macro_f1"] >= 0.35
        ),
        "beat2_test_source_group_macro_f1_at_least_0_35": (
            test["source_group_level"]["macro_f1"] >= 0.35
        ),
        "beat2_validation_minimum_class_recall_at_least_0_15": (
            validation["source_group_level"]["minimum_class_recall"] >= 0.15
        ),
        "beat2_test_minimum_class_recall_at_least_0_15": (
            test["source_group_level"]["minimum_class_recall"] >= 0.15
        ),
        "beat2_validation_speaker_bootstrap_lower_95_above_chance": (
            validation["speaker_cluster_bootstrap"]["lower_95"] > 0.25
        ),
        "beat2_test_speaker_bootstrap_lower_95_above_chance": (
            test["speaker_cluster_bootstrap"]["lower_95"] > 0.25
        ),
        "beat2_val_test_source_group_balanced_accuracy_gap_at_most_0_10": (
            abs(
                validation["source_group_level"]["balanced_accuracy"]
                - test["source_group_level"]["balanced_accuracy"]
            )
            <= 0.10
        ),
        "beat2_test_prediction_max_fraction_at_most_0_80": (
            test["source_group_level"]["maximum_prediction_fraction"] <= 0.80
        ),
        "beat2_train_neutral_calibration_loso_range_at_most_0_10": (
            calibration_sensitivity["range"] <= 0.10
        ),
    }
    passed = all(checks.values())
    return {
        "predeclared_thresholds": {
            "chance_balanced_accuracy": 0.25,
            "source_group_balanced_accuracy_minimum": 0.40,
            "source_group_macro_f1_minimum": 0.35,
            "minimum_class_recall": 0.15,
            "cluster_bootstrap_lower_95_must_exceed": 0.25,
            "val_test_balanced_accuracy_gap_maximum": 0.10,
            "maximum_prediction_fraction": 0.80,
            "neutral_calibration_loso_range_maximum": 0.10,
        },
        "checks": checks,
        "all_checks_passed": passed,
        "eligible_as_generator_training_gate": False,
        "eligible_as_pseudolabel_source": False,
        "research_screening_recommendation": (
            "candidate_for_human_review_queue_prioritization_only"
            if passed
            else "not_suitable_for_training_gate_pseudolabels_or_automatic_screening"
        ),
        "human_robot_observable_validation_required": True,
    }


def _distribution_shift(
    xem_calibrated: np.ndarray,
    xem_rows: Sequence[Mapping[str, Any]],
    beat2_calibrated: np.ndarray,
    beat2_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    xem_train = np.asarray([row["split"] == "train" for row in xem_rows])
    result: dict[str, Any] = {}
    train = np.asarray(xem_calibrated[xem_train], dtype=np.float64)
    train_mean = np.mean(train, axis=0)
    train_scale = np.std(train, axis=0)
    train_scale[train_scale < 1e-10] = 1.0
    for split in SPLITS:
        mask = np.asarray([row["split"] == split for row in beat2_rows])
        target = np.asarray(beat2_calibrated[mask], dtype=np.float64)
        standardized = (target - train_mean) / train_scale
        result[split] = {
            "samples": int(mask.sum()),
            "mean_absolute_standardized_feature": float(
                np.mean(np.abs(standardized))
            ),
            "fraction_values_outside_xem_train_plus_minus_3sd": float(
                np.mean(np.abs(standardized) > 3.0)
            ),
            "maximum_absolute_standardized_feature_mean": float(
                np.max(np.abs(np.mean(standardized, axis=0)))
            ),
        }
    return result


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def run_domain_shift_diagnostic(config_path: Path) -> dict[str, Any]:
    """Run the isolated XEM-train / BEAT2-evaluate diagnostic."""

    config_path = config_path.resolve()
    config_sha = sha256_file(config_path)
    config = load_config(config_path)
    output_root: Path = config["output_root"]

    xem_rows, xem_features, feature_names, xem_receipt = load_xem_common_features(
        config["xem_archive_path"],
        expected_archive_sha256=config["expected_xem_archive_sha256"],
        metadata_path=config["xem_metadata_path"],
        expected_metadata_sha256=config["expected_xem_metadata_sha256"],
        mat_member=config["xem_mat_member"],
        mat_variable=config["xem_mat_variable"],
    )
    beat2_rows, beat2_features, beat2_names, beat2_receipt = (
        load_beat2_weak_evaluation_features(
            config["beat2_manifest_path"],
            expected_sha256=config["expected_beat2_manifest_sha256"],
        )
    )
    if feature_names != beat2_names:
        raise AssertionError("XEM and BEAT2 common feature schemas differ")

    xem_center, xem_scale, xem_neutral_receipt = robust_neutral_reference(
        xem_features,
        xem_rows,
        split="train",
    )
    beat2_center, beat2_scale, beat2_neutral_receipt = robust_neutral_reference(
        beat2_features,
        beat2_rows,
        split="train",
    )
    if beat2_neutral_receipt["weak_label_used"] is not True:
        raise AssertionError("BEAT2 neutral calibration must be marked weak")
    xem_calibrated = apply_reference(xem_features, xem_center, xem_scale)
    beat2_calibrated = apply_reference(beat2_features, beat2_center, beat2_scale)

    calibrated_xem_report, calibrated_model = train_xem_classifier(
        xem_calibrated,
        xem_rows,
        ridge_alphas=config["ridge_alphas"],
    )
    calibrated_scores = apply_classifier(beat2_calibrated, calibrated_model)
    calibrated_beat2_report = evaluate_beat2(
        calibrated_scores,
        beat2_rows,
        bootstrap_replicates=int(config["bootstrap_replicates"]),
        seed=int(config["seed"]),
    )

    raw_xem_report, raw_model = train_xem_classifier(
        xem_features,
        xem_rows,
        ridge_alphas=config["ridge_alphas"],
    )
    raw_scores = apply_classifier(beat2_features, raw_model)
    raw_beat2_report = evaluate_beat2(
        raw_scores,
        beat2_rows,
        bootstrap_replicates=int(config["bootstrap_replicates"]),
        seed=int(config["seed"]) + 100,
    )

    shape_indices = np.arange(
        AMPLITUDE_FEATURE_COUNT,
        len(feature_names),
        dtype=np.int64,
    )
    shape_xem_report, shape_model = train_xem_classifier(
        xem_calibrated,
        xem_rows,
        ridge_alphas=config["ridge_alphas"],
        feature_indices=shape_indices,
    )
    shape_scores = apply_classifier(beat2_calibrated, shape_model)
    shape_beat2_report = evaluate_beat2(
        shape_scores,
        beat2_rows,
        bootstrap_replicates=int(config["bootstrap_replicates"]),
        seed=int(config["seed"]) + 200,
    )

    sensitivity = _calibration_sensitivity(
        beat2_features,
        beat2_rows,
        xem_model=calibrated_model,
    )
    decision = gate_decision(
        calibrated_xem_report,
        calibrated_beat2_report,
        sensitivity,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": REPORT_KIND,
        "research_scope": RESEARCH_SCOPE,
        "class_order": list(EMOTIONS),
        "feature_contract": {
            "policy": COMMON_FEATURE_POLICY,
            "feature_count": len(feature_names),
            "amplitude_feature_count": AMPLITUDE_FEATURE_COUNT,
            "shape_feature_count": len(feature_names) - AMPLITUDE_FEATURE_COUNT,
            "feature_names": feature_names,
            "xem_input": "joint_angular_velocity_vector_norms_23_anonymous_entities",
            "beat2_input": "absolute_controller_angle_finite_difference_18_anonymous_entities",
            "joint_identity_used": False,
            "skeleton_topology_used": False,
            "joint_correspondence_used": False,
            "segment_euler_mapping_used": False,
            "retargeting_used": False,
            "duration_or_sample_rate_feature_used": False,
        },
        "target_label_contract": {
            "role": TARGET_LABEL_ROLE,
            "human_confirmed_robot_observable_labels": 0,
            "emotion_supervision_masks": False,
            "labels_used_for_training": False,
            "labels_used_for_model_selection": False,
            "labels_used_for_reporting_only": True,
        },
        "neutral_calibration": {
            "xem": xem_neutral_receipt,
            "beat2": beat2_neutral_receipt,
            "beat2_allowed_source": "train_split_weak_neutral_only",
            "beat2_validation_labels_or_features_used": False,
            "beat2_test_labels_or_features_used": False,
            "calibration_sensitivity": sensitivity,
        },
        "primary_neutral_calibrated_all_features": {
            "xem": calibrated_xem_report,
            "beat2": calibrated_beat2_report,
        },
        "ablations": {
            "no_target_neutral_calibration": {
                "xem": raw_xem_report,
                "beat2": raw_beat2_report,
            },
            "neutral_calibrated_shape_features_only": {
                "xem": shape_xem_report,
                "beat2": shape_beat2_report,
            },
        },
        "domain_shift": _distribution_shift(
            xem_calibrated,
            xem_rows,
            beat2_calibrated,
            beat2_rows,
        ),
        "training_gate_decision": decision,
        "limitations": {
            "xem_labels_are_intended_not_perceived": True,
            "beat2_labels_are_intended_not_robot_observable": True,
            "xem_has_fixed_action_protocol": True,
            "beat2_is_co_speech_gesture": True,
            "different_sensors_skeletons_and_control_spaces": True,
            "neutral_calibration_uses_weak_target_train_labels": True,
            "this_experiment_cannot_validate_emotion_generation": True,
        },
        "production_training_eligible": False,
        "foundation_ingest_allowed": False,
        "generator_ingest_allowed": False,
        "generator_checkpoint_emitted": False,
    }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        report_path = staging / "domain_shift_report.json"
        features_path = staging / "research_features.npz"
        model_path = staging / "research_model.npz"
        audit_path = staging / "audit.json"
        _atomic_json(report_path, report)
        _atomic_npz(
            features_path,
            feature_names=np.asarray(feature_names),
            xem_sample_ids=np.asarray([row["sample_id"] for row in xem_rows]),
            xem_splits=np.asarray([row["split"] for row in xem_rows]),
            xem_emotions=np.asarray([row["emotion_id"] for row in xem_rows]),
            xem_features=xem_features,
            xem_neutral_calibrated_features=xem_calibrated.astype(np.float32),
            beat2_sample_ids=np.asarray([row["sample_id"] for row in beat2_rows]),
            beat2_splits=np.asarray([row["split"] for row in beat2_rows]),
            beat2_emotions=np.asarray([row["emotion_id"] for row in beat2_rows]),
            beat2_features=beat2_features,
            beat2_neutral_calibrated_features=beat2_calibrated.astype(np.float32),
            generator_compatible=np.asarray(False),
            research_scope=np.asarray(RESEARCH_SCOPE),
        )
        _atomic_npz(
            model_path,
            **calibrated_model,
            class_order=np.asarray(EMOTIONS),
            feature_names=np.asarray(feature_names),
            generator_compatible=np.asarray(False),
            research_scope=np.asarray(RESEARCH_SCOPE),
        )
        final_report = output_root / report_path.name
        final_features = output_root / features_path.name
        final_model = output_root / model_path.name
        final_audit = output_root / audit_path.name
        audit = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": AUDIT_KIND,
            "research_scope": RESEARCH_SCOPE,
            "inputs": {
                "config": {"path": str(config_path), "sha256": config_sha},
                "xem": xem_receipt,
                "beat2": beat2_receipt,
                "forbidden_dataset_input_count": 0,
            },
            "split_isolation": {
                "xem_participant_disjoint": True,
                "beat2_speaker_disjoint": True,
                "beat2_source_group_disjoint": True,
                "beat2_calibration_split": "train",
                "beat2_calibration_label": "weak_neutral",
                "beat2_validation_or_test_used_for_calibration": False,
                "beat2_validation_or_test_used_for_model_selection": False,
            },
            "mapping_isolation": {
                "skeleton_mapping_used": False,
                "segment_euler_mapping_used": False,
                "retargeting_used": False,
                "joint_identity_used": False,
            },
            "outputs": {
                "report": {
                    "path": str(final_report),
                    "sha256": sha256_file(report_path),
                },
                "research_features": {
                    "path": str(final_features),
                    "sha256": sha256_file(features_path),
                    "generator_compatible": False,
                },
                "research_model": {
                    "path": str(final_model),
                    "sha256": sha256_file(model_path),
                    "generator_compatible": False,
                },
                "audit": {"path": str(final_audit)},
            },
            "decision": decision,
            "production_training_eligible": False,
            "foundation_ingest_allowed": False,
            "generator_ingest_allowed": False,
        }
        _atomic_json(audit_path, audit)
        os.replace(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return audit
