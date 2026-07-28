"""Isolated, research-only XEM inventory and emotion baseline.

This module never imports the BEAT2 training stack and never produces a
generator-compatible checkpoint.  It reads the official XEM ZIP in place,
streams its single MAT member through an ephemeral spool, builds a
participant-disjoint inventory, and fits a small NumPy ridge classifier as a
diagnostic baseline.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import Any, BinaryIO, Iterable, Mapping, Sequence
import zipfile

import numpy as np
import scipy
from scipy.io import loadmat, whosmat


SCHEMA_VERSION = "1.0.0"
CONFIG_KIND = "xem_isolated_emotion_research_config_v1"
INVENTORY_KIND = "xem_isolated_sequence_inventory_v1"
AUDIT_KIND = "xem_isolated_emotion_research_audit_v1"
REPORT_KIND = "xem_participant_disjoint_emotion_baseline_v1"
RESEARCH_SCOPE = "isolated_external_emotion_research_only"
DATASET = "XEM"

OFFICIAL_ARCHIVE_BYTES = 271_751_792
OFFICIAL_ARCHIVE_MD5 = "4a7729c4d3d503f1cb0a71433d25d871"
OFFICIAL_DOI = "doi:10.57745/GZQCOY"
OFFICIAL_FILE_DOI = "doi:10.57745/GZQCOY/C8GGLO"
OFFICIAL_LICENSE = "etalab 2.0"
OFFICIAL_CELL_SHAPE = (5, 10, 20)
SEQUENCE_COLUMNS = 277
COLUMN_CONTRACT = {
    "time": [0, 1],
    "joint_positions_23x3": [1, 70],
    "joint_linear_velocity_23x3": [70, 139],
    "joint_angular_velocity_23x3": [139, 208],
    "segment_euler_angles_22x3": [208, 274],
    "center_of_mass_3": [274, 277],
}

ACTIONS = (
    "dancing",
    "moving_hands_up",
    "waving_hands",
    "stopping",
    "pointing",
)
EMOTIONS = ("angry", "neutral", "happy", "sad")
SOURCE_EMOTIONS = ("anger", "neutrality", "happiness", "sadness")
SPLITS = ("train", "validation", "test")
DEFAULT_PARTICIPANT_SPLIT = {
    "train": tuple(range(1, 7)),
    "validation": (7, 8),
    "test": (9, 10),
}

FORBIDDEN_INPUT_MARKERS = ("kimodo", "beat2")
FEATURE_POLICY = (
    "xem_full_body_motion_magnitude_statistics_no_identity_no_action_label_v1"
)
MODEL_POLICY = (
    "numpy_train_only_standardized_one_vs_rest_ridge_validation_alpha_v1"
)

CONFIG_KEYS = {
    "schema_version",
    "artifact_kind",
    "dataset",
    "research_scope",
    "archive_path",
    "expected_archive_bytes",
    "expected_archive_md5",
    "metadata_path",
    "expected_metadata_sha256",
    "mat_member",
    "mat_variable",
    "participant_split",
    "ridge_alphas",
    "seed",
    "output_root",
}


def stable_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
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
        folded = value.casefold()
        if any(marker in folded for marker in FORBIDDEN_INPUT_MARKERS):
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


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_text(path, "".join(stable_json(dict(row)) + "\n" for row in rows))


def _atomic_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _resolve_path(value: object, *, base: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    _reject_forbidden(str(path), context=field)
    return path


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    _reject_forbidden(payload, context=str(path))
    return payload


def _participant_split(value: object) -> dict[str, tuple[int, ...]]:
    if not isinstance(value, Mapping) or set(value) != set(SPLITS):
        raise ValueError("participant_split must define train/validation/test")
    result: dict[str, tuple[int, ...]] = {}
    seen: dict[int, str] = {}
    for split in SPLITS:
        participants = value[split]
        if (
            not isinstance(participants, list)
            or not participants
            or any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in participants
            )
        ):
            raise ValueError(f"participant_split.{split} is invalid")
        normalized = tuple(int(item) for item in participants)
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"participant_split.{split} contains duplicates")
        for participant in normalized:
            if not 1 <= participant <= 10:
                raise ValueError(f"participant index out of range: {participant}")
            if participant in seen:
                raise ValueError(
                    f"participant {participant} crosses {seen[participant]}/{split}"
                )
            seen[participant] = split
        result[split] = normalized
    if set(seen) != set(range(1, 11)):
        raise ValueError("participant split must cover exactly participants 1..10")
    if result != DEFAULT_PARTICIPANT_SPLIT:
        raise ValueError(
            "XEM v1 uses the fixed participant split 1-6/7-8/9-10"
        )
    return result


def load_config(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    _reject_forbidden(str(config_path), context="config_path")
    config = _load_json(config_path)
    if set(config) != CONFIG_KEYS:
        missing = sorted(CONFIG_KEYS - set(config))
        unknown = sorted(set(config) - CONFIG_KEYS)
        raise ValueError(f"XEM config keys changed; missing={missing}, unknown={unknown}")
    if (
        config["schema_version"] != SCHEMA_VERSION
        or config["artifact_kind"] != CONFIG_KIND
        or config["dataset"] != DATASET
        or config["research_scope"] != RESEARCH_SCOPE
    ):
        raise ValueError("invalid isolated XEM research config")
    if (
        config["expected_archive_bytes"] != OFFICIAL_ARCHIVE_BYTES
        or config["expected_archive_md5"] != OFFICIAL_ARCHIVE_MD5
    ):
        raise ValueError("official XEM archive size/MD5 contract changed")
    if not _is_sha256(config["expected_metadata_sha256"]):
        raise ValueError("expected_metadata_sha256 is invalid")
    if config["mat_member"] is not None and not isinstance(
        config["mat_member"], str
    ):
        raise ValueError("mat_member must be null or a ZIP member name")
    if config["mat_variable"] is not None and not isinstance(
        config["mat_variable"], str
    ):
        raise ValueError("mat_variable must be null or a MAT variable name")
    _participant_split(config["participant_split"])
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
        or len(set(float(value) for value in alphas)) != len(alphas)
    ):
        raise ValueError("ridge_alphas must be unique positive finite numbers")
    seed = config["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    base = config_path.parent
    resolved = dict(config)
    resolved["archive_path"] = _resolve_path(
        config["archive_path"], base=base, field="archive_path"
    )
    resolved["metadata_path"] = _resolve_path(
        config["metadata_path"], base=base, field="metadata_path"
    )
    resolved["output_root"] = _resolve_path(
        config["output_root"], base=base, field="output_root"
    )
    if "external_emotion_research" not in resolved["output_root"].parts:
        raise ValueError("XEM output must remain in an external_emotion_research root")
    if resolved["output_root"].exists():
        raise FileExistsError(
            f"refusing to overwrite XEM research output: {resolved['output_root']}"
        )
    return resolved


def _find_metadata_file(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    data = payload.get("data")
    latest = data.get("latestVersion") if isinstance(data, Mapping) else None
    files = latest.get("files") if isinstance(latest, Mapping) else None
    if not isinstance(files, list):
        raise ValueError("Dataverse metadata has no released file list")
    matches = [
        row.get("dataFile")
        for row in files
        if isinstance(row, Mapping)
        and isinstance(row.get("dataFile"), Mapping)
        and row["dataFile"].get("filename") == "xem-dataset.zip"
    ]
    if len(matches) != 1 or not isinstance(matches[0], Mapping):
        raise ValueError("Dataverse metadata must bind one xem-dataset.zip")
    return matches[0]


def validate_official_metadata(
    path: Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError("XEM Dataverse metadata SHA256 mismatch")
    payload = _load_json(path)
    data = payload.get("data")
    latest = data.get("latestVersion") if isinstance(data, Mapping) else None
    license_record = latest.get("license") if isinstance(latest, Mapping) else None
    source_file = _find_metadata_file(payload)
    if (
        payload.get("status") != "OK"
        or not isinstance(data, Mapping)
        or data.get("persistentUrl") != "https://doi.org/10.57745/GZQCOY"
        or not isinstance(latest, Mapping)
        or latest.get("versionState") != "RELEASED"
        or latest.get("datasetPersistentId") != OFFICIAL_DOI
        or not isinstance(license_record, Mapping)
        or str(license_record.get("name", "")).casefold() != OFFICIAL_LICENSE
        or source_file.get("persistentId") != OFFICIAL_FILE_DOI
        or source_file.get("filesize") != OFFICIAL_ARCHIVE_BYTES
        or source_file.get("md5") != OFFICIAL_ARCHIVE_MD5
        or (source_file.get("checksum") or {}).get("value")
        != OFFICIAL_ARCHIVE_MD5
    ):
        raise ValueError("official XEM Dataverse provenance contract changed")
    return {
        "path": str(path),
        "sha256": expected_sha256,
        "dataset_persistent_id": OFFICIAL_DOI,
        "file_persistent_id": OFFICIAL_FILE_DOI,
        "license": str(license_record["name"]),
        "version": (
            f"{latest.get('versionNumber')}.{latest.get('versionMinorNumber')}"
        ),
        "version_state": "RELEASED",
        "official_archive_bytes": OFFICIAL_ARCHIVE_BYTES,
        "official_archive_md5": OFFICIAL_ARCHIVE_MD5,
    }


def validate_archive(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size != OFFICIAL_ARCHIVE_BYTES:
        raise ValueError(
            f"incomplete XEM archive: {size} != {OFFICIAL_ARCHIVE_BYTES} bytes"
        )
    md5 = md5_file(path)
    if md5 != OFFICIAL_ARCHIVE_MD5:
        raise ValueError(f"XEM archive MD5 mismatch: {md5}")
    sha256 = sha256_file(path)
    return {
        "path": str(path),
        "bytes": size,
        "md5": md5,
        "sha256": sha256,
    }


def _safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if not infos or len(set(names)) != len(names):
        raise ValueError("XEM ZIP is empty or contains duplicate member names")
    for info in infos:
        _reject_forbidden(info.filename, context="zip_member")
        member = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        if (
            member.is_absolute()
            or ".." in member.parts
            or "\\" in info.filename
            or (mode and stat.S_ISLNK(mode))
            or info.flag_bits & 0x1
        ):
            raise ValueError(f"unsafe XEM ZIP member: {info.filename!r}")
    return infos


def _select_mat_member(
    infos: Sequence[zipfile.ZipInfo],
    requested: str | None,
) -> zipfile.ZipInfo:
    candidates = [
        info
        for info in infos
        if not info.is_dir() and info.filename.casefold().endswith(".mat")
    ]
    if requested is not None:
        matches = [info for info in candidates if info.filename == requested]
        if len(matches) != 1:
            raise ValueError(f"requested MAT member not found: {requested}")
        return matches[0]
    if len(candidates) != 1:
        raise ValueError(
            f"XEM ZIP must contain exactly one MAT file, found {len(candidates)}"
        )
    return candidates[0]


@contextmanager
def _spooled_member(
    archive_path: Path,
    member_name: str,
) -> Iterable[BinaryIO]:
    with zipfile.ZipFile(archive_path, "r") as archive:
        infos = _safe_members(archive)
        matched = [info for info in infos if info.filename == member_name]
        if len(matched) != 1:
            raise ValueError("selected MAT member disappeared")
        expected_size = matched[0].file_size
        with archive.open(matched[0], "r") as source:
            with tempfile.SpooledTemporaryFile(
                max_size=64 * 1024 * 1024,
                mode="w+b",
            ) as spool:
                copied = 0
                while True:
                    block = source.read(8 * 1024 * 1024)
                    if not block:
                        break
                    spool.write(block)
                    copied += len(block)
                if copied != expected_size:
                    raise ValueError("XEM MAT member decompressed size mismatch")
                spool.seek(0)
                yield spool


def _select_mat_variable(
    variables: Sequence[tuple[str, tuple[int, ...], str]],
    requested: str | None,
) -> tuple[str, tuple[int, ...], str]:
    if requested is not None:
        matches = [value for value in variables if value[0] == requested]
        if len(matches) != 1:
            raise ValueError(f"requested MAT variable not found: {requested}")
        selected = matches[0]
    else:
        matches = [
            value
            for value in variables
            if tuple(value[1]) == OFFICIAL_CELL_SHAPE and value[2] == "cell"
        ]
        if len(matches) != 1:
            raise ValueError(
                "MAT must contain exactly one 5x10x20 cell array; "
                f"found {len(matches)}"
            )
        selected = matches[0]
    if tuple(selected[1]) != OFFICIAL_CELL_SHAPE or selected[2] != "cell":
        raise ValueError("selected MAT variable is not the official 5x10x20 cell array")
    return selected


def inspect_xem_archive(
    archive_path: Path,
    *,
    mat_member: str | None = None,
    mat_variable: str | None = None,
) -> dict[str, Any]:
    """Verify the official archive and inspect its MAT schema without extraction."""

    archive_receipt = validate_archive(archive_path)
    with zipfile.ZipFile(archive_path, "r") as archive:
        infos = _safe_members(archive)
        member = _select_mat_member(infos, mat_member)
        member_rows = [
            {
                "name": info.filename,
                "uncompressed_bytes": info.file_size,
                "compressed_bytes": info.compress_size,
                "crc32": f"{info.CRC:08x}",
                "is_directory": info.is_dir(),
            }
            for info in infos
        ]
    with _spooled_member(archive_path, member.filename) as spool:
        try:
            variables = whosmat(spool)
        except NotImplementedError as error:
            raise ValueError(
                "XEM MAT is v7.3/HDF5; scipy-only loader cannot read it"
            ) from error
    selected = _select_mat_variable(variables, mat_variable)
    return {
        "archive": archive_receipt,
        "archive_members": member_rows,
        "archive_member_count": len(member_rows),
        "mat_member": {
            "name": member.filename,
            "uncompressed_bytes": member.file_size,
            "compressed_bytes": member.compress_size,
            "crc32": f"{member.CRC:08x}",
        },
        "mat_variables": [
            {"name": name, "shape": list(shape), "class": class_name}
            for name, shape, class_name in variables
        ],
        "selected_mat_variable": {
            "name": selected[0],
            "shape": list(selected[1]),
            "class": selected[2],
        },
        "read_policy": (
            "zip_member_streamed_to_ephemeral_spooled_file_"
            "no_persistent_mat_extraction"
        ),
    }


def _load_xem_cells(
    archive_path: Path,
    *,
    mat_member: str | None,
    mat_variable: str | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    inspection = inspect_xem_archive(
        archive_path,
        mat_member=mat_member,
        mat_variable=mat_variable,
    )
    member_name = inspection["mat_member"]["name"]
    variable_name = inspection["selected_mat_variable"]["name"]
    with _spooled_member(archive_path, member_name) as spool:
        try:
            payload = loadmat(
                spool,
                variable_names=[variable_name],
                squeeze_me=False,
                struct_as_record=False,
            )
        except NotImplementedError as error:
            raise ValueError(
                "XEM MAT is v7.3/HDF5; scipy-only loader cannot read it"
            ) from error
    cells = payload.get(variable_name)
    if (
        not isinstance(cells, np.ndarray)
        or cells.dtype != object
        or tuple(cells.shape) != OFFICIAL_CELL_SHAPE
    ):
        raise ValueError("decoded XEM variable is not a 5x10x20 object array")
    return cells, inspection


def _sequence_matrix(value: object, *, sample_id: str) -> np.ndarray:
    matrix = np.asarray(value)
    while matrix.dtype == object and matrix.size == 1:
        matrix = np.asarray(matrix.item())
    if matrix.dtype == object or matrix.ndim != 2:
        raise ValueError(f"{sample_id}: XEM cell is not a numeric 2D matrix")
    if matrix.shape[1] != SEQUENCE_COLUMNS:
        raise ValueError(
            f"{sample_id}: expected {SEQUENCE_COLUMNS} columns, got {matrix.shape}"
        )
    if matrix.shape[0] < 2:
        raise ValueError(f"{sample_id}: sequence has fewer than two frames")
    matrix = np.asarray(matrix, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{sample_id}: sequence contains NaN/Inf")
    delta = np.diff(matrix[:, 0])
    if not np.all(delta > 0.0):
        raise ValueError(f"{sample_id}: time column is not strictly increasing")
    return matrix


def _emotion_for_example(example_index: int) -> tuple[str, str, int]:
    if not 1 <= example_index <= 20:
        raise ValueError("XEM example index must be 1..20")
    group = (example_index - 1) // 5
    repetition = (example_index - 1) % 5 + 1
    return EMOTIONS[group], SOURCE_EMOTIONS[group], repetition


def _participant_assignment(
    participant_index: int,
    split: Mapping[str, Sequence[int]],
) -> str:
    matched = [
        name for name, participants in split.items() if participant_index in participants
    ]
    if len(matched) != 1:
        raise ValueError(f"participant {participant_index} has invalid split assignment")
    return matched[0]


def _decoded_matrix_sha256(matrix: np.ndarray) -> str:
    normalized = np.ascontiguousarray(matrix.astype("<f8", copy=False))
    digest = hashlib.sha256()
    digest.update(np.asarray(normalized.shape, dtype="<i8").tobytes())
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def _magnitude_stats(
    values: np.ndarray,
    *,
    prefix: str,
) -> tuple[np.ndarray, list[str]]:
    if values.ndim != 2:
        raise AssertionError("magnitude statistics expect frames x channels")
    statistics = (
        ("mean", np.mean(values, axis=0)),
        ("std", np.std(values, axis=0)),
        ("median", np.median(values, axis=0)),
        ("q90", np.quantile(values, 0.9, axis=0)),
        ("max", np.max(values, axis=0)),
    )
    features: list[np.ndarray] = []
    names: list[str] = []
    for statistic, vector in statistics:
        features.append(np.asarray(vector, dtype=np.float64))
        names.extend(
            f"{prefix}_{channel + 1:02d}_{statistic}"
            for channel in range(vector.shape[0])
        )
    return np.concatenate(features), names


def sequence_features(matrix: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Extract label-free motion dynamics from the official 277 columns."""

    matrix = _sequence_matrix(matrix, sample_id="feature_input")
    time = matrix[:, 0]
    delta_time = np.diff(time)

    position = matrix[:, 1:70].reshape(matrix.shape[0], 23, 3)
    position_delta = position - position[:1]
    position_magnitude = np.linalg.norm(position_delta, axis=2)

    linear_velocity = matrix[:, 70:139].reshape(matrix.shape[0], 23, 3)
    linear_speed = np.linalg.norm(linear_velocity, axis=2)

    angular_velocity = matrix[:, 139:208].reshape(matrix.shape[0], 23, 3)
    angular_speed = np.linalg.norm(angular_velocity, axis=2)

    euler = matrix[:, 208:274].reshape(matrix.shape[0], 22, 3)
    euler_delta = (euler - euler[:1] + 180.0) % 360.0 - 180.0
    euler_magnitude = np.linalg.norm(euler_delta, axis=2)

    center = matrix[:, 274:277]
    center_magnitude = np.linalg.norm(center - center[:1], axis=1, keepdims=True)

    vectors: list[np.ndarray] = []
    names: list[str] = []
    for prefix, values in (
        ("position_displacement_joint", position_magnitude),
        ("linear_speed_joint", linear_speed),
        ("angular_speed_joint", angular_speed),
        ("euler_displacement_segment", euler_magnitude),
        ("center_of_mass_displacement", center_magnitude),
    ):
        vector, block_names = _magnitude_stats(values, prefix=prefix)
        vectors.append(vector)
        names.extend(block_names)
    duration = float(time[-1] - time[0])
    median_dt = float(np.median(delta_time))
    temporal = np.asarray(
        [
            math.log1p(duration),
            math.log(float(matrix.shape[0])),
            math.log1p(1.0 / median_dt),
            float(np.std(delta_time) / max(median_dt, 1e-12)),
        ],
        dtype=np.float64,
    )
    vectors.append(temporal)
    names.extend(
        (
            "temporal_log1p_duration",
            "temporal_log_frames",
            "temporal_log1p_median_sample_rate",
            "temporal_delta_time_coefficient_of_variation",
        )
    )
    result = np.concatenate(vectors)
    if result.shape != (464,) or len(names) != 464 or not np.isfinite(result).all():
        raise AssertionError("XEM feature schema changed")
    return result.astype(np.float32), names


def build_inventory_and_features(
    cells: np.ndarray,
    *,
    participant_split: Mapping[str, Sequence[int]],
    archive_receipt: Mapping[str, Any],
    mat_member: str,
    mat_variable: str,
) -> tuple[list[dict[str, Any]], np.ndarray, list[str]]:
    if tuple(cells.shape) != OFFICIAL_CELL_SHAPE or cells.dtype != object:
        raise ValueError("XEM cell array shape changed")
    rows: list[dict[str, Any]] = []
    feature_rows: list[np.ndarray] = []
    feature_names: list[str] | None = None
    for action_zero in range(5):
        for participant_zero in range(10):
            for example_zero in range(20):
                action_index = action_zero + 1
                participant_index = participant_zero + 1
                example_index = example_zero + 1
                sample_id = (
                    f"xem_a{action_index:02d}_p{participant_index:02d}_"
                    f"e{example_index:02d}"
                )
                matrix = _sequence_matrix(
                    cells[action_zero, participant_zero, example_zero],
                    sample_id=sample_id,
                )
                emotion, source_emotion, repetition = _emotion_for_example(
                    example_index
                )
                split = _participant_assignment(
                    participant_index,
                    participant_split,
                )
                delta_time = np.diff(matrix[:, 0])
                decoded_sha = _decoded_matrix_sha256(matrix)
                feature, names = sequence_features(matrix)
                if feature_names is None:
                    feature_names = names
                elif names != feature_names:
                    raise AssertionError("feature names changed across sequences")
                feature_rows.append(feature)
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "artifact_kind": INVENTORY_KIND,
                        "sample_id": sample_id,
                        "dataset": DATASET,
                        "research_scope": RESEARCH_SCOPE,
                        "participant_id": f"xem_participant_{participant_index:02d}",
                        "participant_index": participant_index,
                        "fixed_split_assignment": split,
                        "action_id": ACTIONS[action_zero],
                        "action_index": action_index,
                        "emotion_id": emotion,
                        "source_emotion_label": source_emotion,
                        "emotion_label_source": (
                            "official_xem_example_index_protocol_intended_expression"
                        ),
                        "human_confirmed_robot_observable": False,
                        "repetition_index": repetition,
                        "example_index": example_index,
                        "frames": int(matrix.shape[0]),
                        "columns": int(matrix.shape[1]),
                        "duration_source_units": float(
                            matrix[-1, 0] - matrix[0, 0]
                        ),
                        "median_time_step_source_units": float(
                            np.median(delta_time)
                        ),
                        "decoded_matrix_sha256": decoded_sha,
                        "archive_sha256": archive_receipt["sha256"],
                        "mat_member": mat_member,
                        "mat_variable": mat_variable,
                        "feature_policy": FEATURE_POLICY,
                        "production_training_eligible": False,
                        "foundation_ingest_allowed": False,
                        "generator_ingest_allowed": False,
                    }
                )
    if len(rows) != 1_000 or feature_names is None:
        raise AssertionError("XEM inventory must contain exactly 1,000 sequences")
    features = np.stack(feature_rows).astype(np.float32)
    if features.shape != (1_000, 464):
        raise AssertionError("XEM feature matrix shape changed")
    _validate_inventory_rows(rows)
    return rows, features, feature_names


def _validate_inventory_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != 1_000:
        raise ValueError("XEM inventory must contain exactly 1,000 rows")
    sample_ids: set[str] = set()
    participant_splits: defaultdict[str, set[str]] = defaultdict(set)
    tuple_keys: set[tuple[int, int, int]] = set()
    decoded_hashes: set[str] = set()
    for row in rows:
        sample_id = row.get("sample_id")
        if (
            row.get("schema_version") != SCHEMA_VERSION
            or row.get("artifact_kind") != INVENTORY_KIND
            or row.get("dataset") != DATASET
            or row.get("research_scope") != RESEARCH_SCOPE
            or row.get("emotion_id") not in EMOTIONS
            or row.get("action_id") not in ACTIONS
            or row.get("fixed_split_assignment") not in SPLITS
            or row.get("human_confirmed_robot_observable") is not False
            or row.get("production_training_eligible") is not False
            or row.get("foundation_ingest_allowed") is not False
            or row.get("generator_ingest_allowed") is not False
            or not isinstance(sample_id, str)
            or not sample_id
        ):
            raise ValueError(f"{sample_id}: invalid isolated XEM inventory row")
        _reject_forbidden(row, context=sample_id)
        if sample_id in sample_ids:
            raise ValueError(f"duplicate XEM sample_id: {sample_id}")
        sample_ids.add(sample_id)
        participant = str(row["participant_id"])
        participant_splits[participant].add(str(row["fixed_split_assignment"]))
        key = (
            int(row["action_index"]),
            int(row["participant_index"]),
            int(row["example_index"]),
        )
        if key in tuple_keys:
            raise ValueError(f"duplicate XEM action/participant/example: {key}")
        tuple_keys.add(key)
        decoded = row.get("decoded_matrix_sha256")
        if not _is_sha256(decoded):
            raise ValueError(f"{sample_id}: decoded matrix hash is invalid")
        if decoded in decoded_hashes:
            raise ValueError(f"duplicate decoded XEM trajectory: {decoded}")
        decoded_hashes.add(str(decoded))
    leaked = sorted(
        participant
        for participant, splits in participant_splits.items()
        if len(splits) != 1
    )
    if leaked:
        raise ValueError(f"XEM participants cross splits: {leaked}")
    if set(participant_splits) != {
        f"xem_participant_{index:02d}" for index in range(1, 11)
    }:
        raise ValueError("XEM participant coverage changed")


def load_xem_inventory(path: Path) -> list[dict[str, Any]]:
    path = path.resolve()
    _reject_forbidden(str(path), context="inventory_path")
    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            payload = raw[:-1] if raw.endswith(b"\n") else raw
            if payload.endswith(b"\r"):
                raise ValueError(f"CRLF inventory is not accepted: {line_number}")
            if not payload.strip():
                continue
            try:
                row = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid XEM inventory line {line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"XEM inventory line {line_number} is not an object")
            rows.append(row)
    _validate_inventory_rows(rows)
    return rows


def _class_indices(values: Sequence[str]) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(EMOTIONS)}
    try:
        return np.asarray([lookup[str(value)] for value in values], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"unsupported XEM emotion: {error.args[0]}") from error


def _fit_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    targets = np.eye(len(EMOTIONS), dtype=np.float64)[train_y]
    augmented = np.concatenate(
        [train_x, np.ones((train_x.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    regularizer = np.eye(augmented.shape[1], dtype=np.float64) * float(alpha)
    regularizer[-1, -1] = 0.0
    left = augmented.T @ augmented + regularizer
    right = augmented.T @ targets
    try:
        return np.linalg.solve(left, right)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(left, right, rcond=None)[0]


def _predict_ridge(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    augmented = np.concatenate(
        [features, np.ones((features.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    return np.argmax(augmented @ weights, axis=1).astype(np.int64)


def _classification_metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Any]:
    if truth.shape != predicted.shape or truth.ndim != 1 or truth.size == 0:
        raise ValueError("classification metric inputs are invalid")
    confusion = np.zeros((len(EMOTIONS), len(EMOTIONS)), dtype=np.int64)
    for expected, actual in zip(truth, predicted, strict=True):
        confusion[int(expected), int(actual)] += 1
    recalls: list[float] = []
    f1_values: list[float] = []
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
        f1_values.append(f1)
        per_class[emotion] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "samples": int(truth.size),
        "accuracy": float(np.mean(truth == predicted)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_values)),
        "confusion_matrix": confusion.tolist(),
        "class_order": list(EMOTIONS),
        "per_class": per_class,
    }


def _standardize_train_only(
    train_x: np.ndarray,
    *others: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray, int]:
    mean = np.mean(train_x, axis=0, dtype=np.float64)
    std = np.std(train_x, axis=0, dtype=np.float64)
    constant = std < 1e-12
    scale = std.copy()
    scale[constant] = 1.0
    normalized_train = (train_x.astype(np.float64) - mean) / scale
    normalized_others = [
        (value.astype(np.float64) - mean) / scale for value in others
    ]
    return normalized_train, normalized_others, mean, scale, int(constant.sum())


def train_participant_disjoint_baseline(
    features: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    ridge_alphas: Sequence[float],
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if features.shape != (1_000, 464) or len(rows) != 1_000:
        raise ValueError("baseline expects the complete 1000x464 XEM feature matrix")
    split_values = np.asarray(
        [str(row["fixed_split_assignment"]) for row in rows]
    )
    action_values = np.asarray([str(row["action_id"]) for row in rows])
    labels = _class_indices([str(row["emotion_id"]) for row in rows])
    masks = {split: split_values == split for split in SPLITS}
    expected_counts = {"train": 600, "validation": 200, "test": 200}
    if {
        split: int(mask.sum()) for split, mask in masks.items()
    } != expected_counts:
        raise ValueError("participant-disjoint XEM split counts changed")

    train_x, normalized, mean, scale, constant_features = _standardize_train_only(
        features[masks["train"]],
        features[masks["validation"]],
        features[masks["test"]],
    )
    validation_x, test_x = normalized
    train_y = labels[masks["train"]]
    validation_y = labels[masks["validation"]]
    test_y = labels[masks["test"]]

    candidates: list[dict[str, Any]] = []
    candidate_weights: dict[float, np.ndarray] = {}
    for raw_alpha in ridge_alphas:
        alpha = float(raw_alpha)
        weights = _fit_ridge(train_x, train_y, alpha=alpha)
        predicted = _predict_ridge(validation_x, weights)
        metrics = _classification_metrics(validation_y, predicted)
        candidates.append({"alpha": alpha, "validation": metrics})
        candidate_weights[alpha] = weights
    chosen = max(
        candidates,
        key=lambda row: (
            row["validation"]["macro_f1"],
            row["validation"]["balanced_accuracy"],
            row["validation"]["accuracy"],
            -row["alpha"],
        ),
    )
    alpha = float(chosen["alpha"])
    weights = candidate_weights[alpha]
    train_predicted = _predict_ridge(train_x, weights)
    test_predicted = _predict_ridge(test_x, weights)

    rng = np.random.default_rng(seed)
    permuted_train_y = train_y.copy()
    rng.shuffle(permuted_train_y)
    permuted_weights = _fit_ridge(train_x, permuted_train_y, alpha=alpha)
    permuted_test = _predict_ridge(test_x, permuted_weights)

    train_actions = action_values[masks["train"]]
    test_actions = action_values[masks["test"]]
    action_predictions: dict[str, int] = {}
    for action in ACTIONS:
        counts = np.bincount(
            train_y[train_actions == action],
            minlength=len(EMOTIONS),
        )
        action_predictions[action] = int(np.argmax(counts))
    action_only_test = np.asarray(
        [action_predictions[str(action)] for action in test_actions],
        dtype=np.int64,
    )

    per_action: dict[str, dict[str, Any]] = {}
    for action in ACTIONS:
        mask = test_actions == action
        per_action[action] = _classification_metrics(
            test_y[mask],
            test_predicted[mask],
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": REPORT_KIND,
        "dataset": DATASET,
        "research_scope": RESEARCH_SCOPE,
        "model_policy": MODEL_POLICY,
        "feature_policy": FEATURE_POLICY,
        "participant_split": {
            split: list(DEFAULT_PARTICIPANT_SPLIT[split]) for split in SPLITS
        },
        "split_counts": expected_counts,
        "class_order": list(EMOTIONS),
        "action_order": list(ACTIONS),
        "alpha_candidates": candidates,
        "selected_alpha": alpha,
        "train_metrics": _classification_metrics(train_y, train_predicted),
        "validation_metrics": chosen["validation"],
        "test_metrics": _classification_metrics(test_y, test_predicted),
        "test_metrics_by_action": per_action,
        "negative_controls": {
            "action_only_train_contingency_test": _classification_metrics(
                test_y,
                action_only_test,
            ),
            "permuted_train_emotion_test": _classification_metrics(
                test_y,
                permuted_test,
            ),
        },
        "preprocessing": {
            "standardization_statistics_source": "train_participants_only",
            "constant_feature_count": constant_features,
            "validation_used_only_for_alpha_selection": True,
            "test_used_for_model_selection": False,
        },
        "label_limitations": {
            "labels_are_intended_performance_protocol_not_human_perception": True,
            "human_confirmed_robot_observable_labels": 0,
            "fixed_action_set_can_create_shortcuts": True,
            "baseline_does_not_establish_cross_dataset_generalization": True,
        },
        "production_training_eligible": False,
        "foundation_ingest_allowed": False,
        "generator_ingest_allowed": False,
        "generator_checkpoint_emitted": False,
    }
    model = {
        "weights": weights.astype(np.float64),
        "feature_mean": mean.astype(np.float64),
        "feature_scale": scale.astype(np.float64),
        "selected_alpha": np.asarray(alpha, dtype=np.float64),
        "class_order": np.asarray(EMOTIONS),
        "model_policy": np.asarray(MODEL_POLICY),
        "research_scope": np.asarray(RESEARCH_SCOPE),
        "generator_compatible": np.asarray(False),
    }
    return report, model


def _counts_by(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def _numeric_summary(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> dict[str, float | int]:
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    return {
        "sum": float(values.sum()),
        "minimum": float(values.min()),
        "median": float(np.median(values)),
        "maximum": float(values.max()),
    }


def run_xem_research_pipeline(config_path: Path) -> dict[str, Any]:
    """Build the isolated inventory, features, and participant-disjoint baseline."""

    config_path = config_path.resolve()
    config_sha256 = sha256_file(config_path)
    config = load_config(config_path)
    archive_path: Path = config["archive_path"]
    metadata_path: Path = config["metadata_path"]
    output_root: Path = config["output_root"]
    metadata_receipt = validate_official_metadata(
        metadata_path,
        expected_sha256=str(config["expected_metadata_sha256"]),
    )
    cells, inspection = _load_xem_cells(
        archive_path,
        mat_member=config["mat_member"],
        mat_variable=config["mat_variable"],
    )
    split = _participant_split(config["participant_split"])
    inventory, features, feature_names = build_inventory_and_features(
        cells,
        participant_split=split,
        archive_receipt=inspection["archive"],
        mat_member=inspection["mat_member"]["name"],
        mat_variable=inspection["selected_mat_variable"]["name"],
    )
    report, model = train_participant_disjoint_baseline(
        features,
        inventory,
        ridge_alphas=[float(value) for value in config["ridge_alphas"]],
        seed=int(config["seed"]),
    )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.",
            dir=output_root.parent,
        )
    )
    try:
        inventory_path = staging / "inventory.jsonl"
        features_path = staging / "features.npz"
        model_path = staging / "baseline_model.npz"
        report_path = staging / "baseline_report.json"
        audit_path = staging / "audit.json"
        _atomic_jsonl(inventory_path, inventory)
        _atomic_npz(
            features_path,
            sample_ids=np.asarray([row["sample_id"] for row in inventory]),
            participant_ids=np.asarray(
                [row["participant_id"] for row in inventory]
            ),
            fixed_split_assignments=np.asarray(
                [row["fixed_split_assignment"] for row in inventory]
            ),
            action_ids=np.asarray([row["action_id"] for row in inventory]),
            emotion_ids=np.asarray([row["emotion_id"] for row in inventory]),
            feature_names=np.asarray(feature_names),
            features=features,
            feature_policy=np.asarray(FEATURE_POLICY),
            research_scope=np.asarray(RESEARCH_SCOPE),
            generator_compatible=np.asarray(False),
        )
        _atomic_npz(model_path, **model)

        final_inventory = output_root / inventory_path.name
        final_features = output_root / features_path.name
        final_model = output_root / model_path.name
        final_report = output_root / report_path.name
        final_audit = output_root / audit_path.name
        report.update(
            {
                "inventory_path": str(final_inventory),
                "inventory_sha256": sha256_file(inventory_path),
                "features_path": str(final_features),
                "features_sha256": sha256_file(features_path),
                "model_path": str(final_model),
                "model_sha256": sha256_file(model_path),
            }
        )
        _atomic_json(report_path, report)
        audit = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": AUDIT_KIND,
            "dataset": DATASET,
            "research_scope": RESEARCH_SCOPE,
            "source": {
                "config": {
                    "path": str(config_path),
                    "sha256": config_sha256,
                },
                "archive": inspection["archive"],
                "official_metadata": metadata_receipt,
                "zip_and_mat_inspection": inspection,
                "sequence_column_contract_zero_based_half_open": COLUMN_CONTRACT,
            },
            "inventory": {
                "records": len(inventory),
                "features": int(features.shape[1]),
                "counts_by_split": _counts_by(
                    inventory,
                    "fixed_split_assignment",
                ),
                "counts_by_participant": _counts_by(
                    inventory,
                    "participant_id",
                ),
                "counts_by_action": _counts_by(inventory, "action_id"),
                "counts_by_emotion": _counts_by(inventory, "emotion_id"),
                "frame_count_summary": _numeric_summary(inventory, "frames"),
                "duration_source_units_summary": _numeric_summary(
                    inventory,
                    "duration_source_units",
                ),
                "median_time_step_source_units_summary": _numeric_summary(
                    inventory,
                    "median_time_step_source_units",
                ),
                "time_unit_policy": (
                    "preserve_official_raw_values_unit_not_asserted_"
                    "approximately_60hz_16_or_17_tick_steps_observed"
                ),
                "participant_disjoint": True,
                "unique_decoded_trajectories": len(
                    {row["decoded_matrix_sha256"] for row in inventory}
                ),
            },
            "isolation": {
                "external_inputs_only": True,
                "kimodo_input_count": 0,
                "beat2_input_count": 0,
                "foundation_ingest_allowed": False,
                "generator_ingest_allowed": False,
                "normalization_shared_with_generator": False,
                "checkpoint_shared_with_generator": False,
                "persistent_mat_extraction": False,
                "production_training_eligible": False,
            },
            "limitations": report["label_limitations"],
            "software": {
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "classifier_dependency": "numpy_only",
            },
            "implementation": {
                "module_path": str(Path(__file__).resolve()),
                "module_sha256": sha256_file(Path(__file__).resolve()),
            },
            "outputs": {
                "inventory": {
                    "path": str(final_inventory),
                    "sha256": sha256_file(inventory_path),
                },
                "features": {
                    "path": str(final_features),
                    "sha256": sha256_file(features_path),
                },
                "baseline_model": {
                    "path": str(final_model),
                    "sha256": sha256_file(model_path),
                    "generator_compatible": False,
                },
                "baseline_report": {
                    "path": str(final_report),
                    "sha256": sha256_file(report_path),
                },
                "audit": {"path": str(final_audit)},
            },
        }
        _atomic_json(audit_path, audit)
        os.replace(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return audit
