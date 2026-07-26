#!/usr/bin/env python3
"""Create fail-closed audio emotion pseudo-labels for BEAT2 windows.

The input is the boundary-validated six-second window inventory produced by
``build_beat2_full_window_inventory.py``.  Audio SER is only a review hint: this
tool never writes a trainable emotion label or enables emotion supervision.

The default backend follows the official emotion2vec+/FunASR interface:

    AutoModel(model="iic/emotion2vec_plus_large", ...)
    model.generate(input=wav_path, granularity="utterance",
                   extract_embedding=False)

FunASR and its model are loaded lazily.  Tests and offline audits can inject a
predictor object or use ``--predictor-factory package.module:callable`` without
installing FunASR or downloading model weights.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import numbers
import os
import struct
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Protocol


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_INVENTORY = (
    PROJECT_ROOT
    / "deliverables/interactive_human_motion_v1/catalog/"
    "beat2_interaction_full_6s_windows_v1.jsonl"
)
DEFAULT_BEAT2_ROOT = Path(
    "/home/gez/nas/cloud/gez/human_motion/raw/BEAT2/beat_chinese_v2.0.0"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "deliverables/interactive_human_motion_v1/catalog/beat2_audio_emotion_pseudo_v1"
)
OUTPUT_STEM = "beat2_audio_emotion_pseudo_v1"
PROGRESS_FILENAME = f".{OUTPUT_STEM}.progress.jsonl"
SCHEMA_VERSION = "beat2_audio_ser_pseudo_v1.0.0"
EMOTION_SOURCE = "audio_ser_pseudo"
EMOTION_REVIEW_STATUS = "pseudo_pending_human_review"
NETWORK_EMOTIONS = ("neutral", "sad", "happy", "angry", "surprise", "fear")
DEFAULT_MODEL_ID = "iic/emotion2vec_plus_large"
DEFAULT_MODEL_REVISION = "master"
DEFAULT_MIN_SCORE = 0.55
DEFAULT_MIN_MARGIN = 0.10
OFFICIAL_FUNASR_URL = "https://github.com/modelscope/FunASR"
OFFICIAL_EMOTION2VEC_URL = "https://github.com/ddlBoJack/emotion2vec"
OFFICIAL_MODEL_URL = "https://modelscope.cn/models/iic/emotion2vec_plus_large"

# emotion2vec+ publishes nine output classes.  Disgust, other, and unknown have
# no honest equivalent in the network's six-class contract and remain
# unresolved instead of being folded into neutral or another class.
RAW_LABEL_TO_NETWORK_EMOTION: dict[str, str | None] = {
    "angry": "angry",
    "anger": "angry",
    "happy": "happy",
    "happiness": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "fear": "fear",
    "fearful": "fear",
    "surprise": "surprise",
    "surprised": "surprise",
    "disgust": None,
    "disgusted": None,
    "other": None,
    "unknown": None,
}


class AudioEmotionPredictor(Protocol):
    """Minimal injectable predictor contract used by ``build_pseudolabels``."""

    model_id: str
    model_revision: str
    backend_name: str

    def predict(self, wav_path: Path, *, clip_id: str, sample_rate: int) -> Any:
        """Return a FunASR-like object containing ``labels`` and ``scores``."""


class FunASRAudioEmotionPredictor:
    """Lazy adapter around the official FunASR emotion2vec+ API."""

    backend_name = "funasr_emotion2vec_plus"

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        model_revision: str = DEFAULT_MODEL_REVISION,
        hub: str = "ms",
        device: str = "cpu",
    ) -> None:
        try:
            from funasr import AutoModel
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "FunASR is not installed. Install it with `pip install -U funasr "
                "modelscope`; constructing the default backend may then download "
                "the emotion2vec+ model. Use --predictor-factory for an offline "
                "or mock backend."
            ) from error

        self.model_id = model_id
        self.model_revision = model_revision
        self.hub = hub
        self.device = device
        self._model = AutoModel(
            model=model_id,
            model_revision=model_revision,
            hub=hub,
            device=device,
        )
        self.model_path = str(getattr(self._model, "model_path", "")) or None
        try:
            from importlib.metadata import version

            self.funasr_version = version("funasr")
        except Exception:  # pragma: no cover - package metadata can be absent
            self.funasr_version = None

    def predict(self, wav_path: Path, *, clip_id: str, sample_rate: int) -> Any:
        del clip_id, sample_rate
        return self._model.generate(
            input=str(wav_path),
            granularity="utterance",
            extract_embedding=False,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-inventory", type=Path, default=DEFAULT_INPUT_INVENTORY)
    parser.add_argument("--beat2-root", type=Path, default=DEFAULT_BEAT2_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--hub", default="ms")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--min-margin", type=float, default=DEFAULT_MIN_MARGIN)
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N inventory records (useful for a real-model probe).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the append-only, contract-validated progress journal.",
    )
    parser.add_argument(
        "--predictor-factory",
        help=(
            "Optional package.module:callable. The callable receives model_id, "
            "model_revision, hub, and device keyword arguments and returns an "
            "object with predict(wav_path, clip_id=..., sample_rate=...)."
        ),
    )
    parser.add_argument(
        "--continue-on-prediction-error",
        action="store_true",
        help="Emit an unresolved review item instead of aborting on predictor errors.",
    )
    return parser.parse_args(argv)


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_sha256(record: dict[str, Any]) -> str:
    return sha256_bytes(stable_json(record).encode("utf-8"))


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def append_durable_jsonl(path: Path, value: dict[str, Any]) -> None:
    """Append one recoverable record and make it durable before returning."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(stable_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"Cannot read window inventory {path}: {error}") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        clip_id = _required_string(record.get("clip_id"), "clip_id")
        if clip_id in seen:
            raise ValueError(f"Duplicate clip_id in window inventory: {clip_id}")
        seen.add(clip_id)
        records.append(record)
    if not records:
        raise ValueError(f"Window inventory is empty: {path}")
    return records


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _required_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def resolve_audio_path(beat2_root: Path, record: dict[str, Any]) -> Path:
    clip_id = _required_string(record.get("clip_id"), "clip_id")
    relpath = _required_string(record.get("audio_relpath"), f"{clip_id}.audio_relpath")
    root = beat2_root.resolve()
    path = (root / relpath).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{clip_id}.audio_relpath escapes BEAT2 root: {relpath}") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def normalize_raw_label(label: str) -> str:
    """Normalize known FunASR labels without substring-based guessing."""

    value = label.strip().lower().replace("-", "_").replace(" ", "_")
    for prefix in ("emo_", "emotion_"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
    direct = value.strip("_.,:;()[]{}")
    if direct in RAW_LABEL_TO_NETWORK_EMOTION:
        return direct
    # Some model revisions display bilingual labels such as ``生气/angry``.
    for separator in ("/", "|", ":"):
        for part in direct.split(separator):
            candidate = part.strip("_.,:;()[]{}")
            if candidate in RAW_LABEL_TO_NETWORK_EMOTION:
                return candidate
    return direct


def map_raw_label(label: str) -> str | None:
    return RAW_LABEL_TO_NETWORK_EMOTION.get(normalize_raw_label(label))


def _as_list(value: Any, field: str) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise ValueError(f"Predictor {field} must be a list")
    return value


def parse_prediction(raw_prediction: Any) -> dict[str, Any]:
    """Parse the documented FunASR ``labels``/``scores`` result."""

    if isinstance(raw_prediction, tuple):
        raw_prediction = list(raw_prediction)
    if isinstance(raw_prediction, list):
        if len(raw_prediction) != 1:
            raise ValueError(
                "Predictor must return exactly one utterance result per audio window"
            )
        raw_prediction = raw_prediction[0]
    if not isinstance(raw_prediction, dict):
        raise ValueError("Predictor result must be an object or one-item list")
    labels = _as_list(raw_prediction.get("labels"), "labels")
    scores = _as_list(raw_prediction.get("scores"), "scores")
    if not labels or len(labels) != len(scores):
        raise ValueError("Predictor labels and scores must be non-empty and equal length")

    pairs: list[tuple[int, str, float]] = []
    for index, (label, score) in enumerate(zip(labels, scores)):
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Predictor label at index {index} must be non-empty")
        if isinstance(score, bool) or not isinstance(score, numbers.Real):
            raise ValueError(f"Predictor score at index {index} must be numeric")
        numeric_score = float(score)
        if not math.isfinite(numeric_score) or not 0.0 <= numeric_score <= 1.0:
            raise ValueError(
                f"Predictor score at index {index} must be finite and in [0, 1]"
            )
        pairs.append((index, label.strip(), numeric_score))
    ranked = sorted(pairs, key=lambda item: (-item[2], item[0]))
    _top_index, top_label, top_score = ranked[0]
    if len(ranked) > 1:
        _runner_index, runner_label, runner_score = ranked[1]
    else:
        runner_label, runner_score = None, 0.0
    return {
        "labels": [label.strip() for label in labels],
        "scores": [float(score) for score in scores],
        "raw_model_label": top_label,
        "raw_model_score": top_score,
        "raw_runner_up_label": runner_label,
        "raw_runner_up_score": runner_score,
        "raw_model_margin": top_score - runner_score,
    }


def resolve_candidate(
    prediction: dict[str, Any], *, min_score: float, min_margin: float
) -> tuple[str | None, str | None]:
    mapped = map_raw_label(prediction["raw_model_label"])
    if mapped is None:
        return None, "raw_label_not_mappable_to_network_six_class_ontology"
    if prediction["raw_model_score"] < min_score:
        return None, "below_min_score"
    if prediction["raw_model_margin"] < min_margin:
        return None, "below_min_margin"
    return mapped, None


def _validate_threshold(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


def extract_exact_audio_window(
    source_path: Path,
    record: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    """Write exactly the inventory's half-open audio-frame range to a WAV.

    Python's standard ``wave`` reader rejects WAVE_FORMAT_IEEE_FLOAT, which is
    used by a small part of BEAT2.  Parsing the RIFF chunks directly preserves
    both PCM (format 1) and float32 (format 3) sample bytes without requiring
    soundfile or introducing a quantization step.
    """

    clip_id = _required_string(record.get("clip_id"), "clip_id")
    window = record.get("window")
    if not isinstance(window, dict):
        raise ValueError(f"{clip_id}.window must be an object")
    start = _required_integer(
        window.get("audio_start_sample"), f"{clip_id}.window.audio_start_sample"
    )
    end = _required_integer(
        window.get("audio_end_sample_exclusive"),
        f"{clip_id}.window.audio_end_sample_exclusive",
    )
    sample_count = _required_integer(
        window.get("audio_sample_count"), f"{clip_id}.window.audio_sample_count"
    )
    if start < 0 or end <= start or sample_count != end - start:
        raise ValueError(f"{clip_id} has an invalid audio sample interval")

    metadata = read_riff_wave_metadata(source_path)
    channels = metadata["channels"]
    sample_width = metadata["sample_width_bytes"]
    sample_rate = metadata["sample_rate"]
    source_frame_count = metadata["frame_count"]
    if end > source_frame_count:
        raise ValueError(
            f"{clip_id} audio interval [{start}, {end}) exceeds "
            f"{source_frame_count} source samples"
        )
    expected_sample_rate = record.get("audio_sample_rate")
    expected_channels = record.get("audio_channels")
    expected_frames = record.get("audio_frame_count")
    if expected_sample_rate is not None and expected_sample_rate != sample_rate:
        raise ValueError(f"{clip_id} audio_sample_rate no longer matches source WAV")
    if expected_channels is not None and expected_channels != channels:
        raise ValueError(f"{clip_id} audio_channels no longer matches source WAV")
    if expected_frames is not None and expected_frames != source_frame_count:
        raise ValueError(f"{clip_id} audio_frame_count no longer matches source WAV")
    expected_format = record.get("audio_format")
    actual_format = {1: "pcm", 3: "ieee_float"}[metadata["format_code"]]
    if expected_format is not None and expected_format != actual_format:
        raise ValueError(f"{clip_id} audio_format no longer matches source WAV")

    byte_start = metadata["data_offset"] + start * metadata["block_align"]
    expected_byte_count = sample_count * metadata["block_align"]
    with source_path.open("rb") as source:
        source.seek(byte_start)
        sample_bytes = source.read(expected_byte_count)
    if len(sample_bytes) != expected_byte_count:
        raise ValueError(
            f"{clip_id} exact audio read returned {len(sample_bytes)} bytes; "
            f"expected {expected_byte_count}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmt_payload = metadata["fmt_payload"]
    fmt_padding = b"\x00" if len(fmt_payload) % 2 else b""
    data_padding = b"\x00" if len(sample_bytes) % 2 else b""
    riff_body = (
        b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt_payload))
        + fmt_payload
        + fmt_padding
        + b"data"
        + struct.pack("<I", len(sample_bytes))
        + sample_bytes
        + data_padding
    )
    output_path.write_bytes(b"RIFF" + struct.pack("<I", len(riff_body)) + riff_body)
    return {
        "source_audio_path": str(source_path),
        "audio_start_sample": start,
        "audio_end_sample_exclusive": end,
        "audio_sample_count": sample_count,
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "source_wave_format_code": metadata["format_code"],
        "source_wave_format": actual_format,
        "window_sample_byte_count": len(sample_bytes),
        "window_sample_bytes_sha256": sha256_bytes(sample_bytes),
        "temporary_window_wav_sha256": sha256_file(output_path),
        "boundary_semantics": "half_open_source_sample_interval",
    }


def read_riff_wave_metadata(path: Path) -> dict[str, Any]:
    """Read the minimal RIFF/WAVE structure needed for byte-exact slicing."""

    file_size = path.stat().st_size
    with path.open("rb") as handle:
        header = handle.read(12)
        if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
            raise ValueError(f"Not a little-endian RIFF/WAVE file: {path}")
        declared_riff_size = struct.unpack("<I", header[4:8])[0]
        riff_end = declared_riff_size + 8
        if riff_end > file_size:
            raise ValueError(f"Truncated RIFF/WAVE file: {path}")
        fmt_payload = None
        data_offset = None
        data_size = None
        position = 12
        while position + 8 <= riff_end:
            handle.seek(position)
            chunk_header = handle.read(8)
            if len(chunk_header) != 8:
                raise ValueError(f"Truncated RIFF chunk header: {path}")
            chunk_id = chunk_header[:4]
            chunk_size = struct.unpack("<I", chunk_header[4:])[0]
            payload_offset = position + 8
            payload_end = payload_offset + chunk_size
            if payload_end > riff_end:
                raise ValueError(f"RIFF chunk exceeds declared file bounds: {path}")
            if chunk_id == b"fmt ":
                if fmt_payload is not None:
                    raise ValueError(f"WAV contains multiple fmt chunks: {path}")
                if chunk_size < 16 or chunk_size > 4096:
                    raise ValueError(f"Unsupported WAV fmt chunk size {chunk_size}: {path}")
                handle.seek(payload_offset)
                fmt_payload = handle.read(chunk_size)
            elif chunk_id == b"data":
                if data_offset is not None:
                    raise ValueError(f"WAV contains multiple data chunks: {path}")
                data_offset = payload_offset
                data_size = chunk_size
            position = payload_end + (chunk_size % 2)

    if fmt_payload is None or data_offset is None or data_size is None:
        raise ValueError(f"WAV requires exactly one fmt and one data chunk: {path}")
    format_code, channels, sample_rate, byte_rate, block_align, bits_per_sample = (
        struct.unpack("<HHIIHH", fmt_payload[:16])
    )
    if format_code not in {1, 3}:
        raise ValueError(f"Unsupported WAV format code {format_code}: {path}")
    if channels < 1 or sample_rate < 1 or block_align < channels:
        raise ValueError(f"Invalid WAV format metadata: {path}")
    if block_align % channels or bits_per_sample != (block_align // channels) * 8:
        raise ValueError(f"Inconsistent WAV block alignment: {path}")
    if byte_rate != sample_rate * block_align:
        raise ValueError(f"Inconsistent WAV byte rate: {path}")
    if data_size % block_align:
        raise ValueError(f"WAV data size is not frame aligned: {path}")
    if format_code == 3 and bits_per_sample != 32:
        raise ValueError(f"Only float32 IEEE WAV is supported: {path}")
    return {
        "format_code": format_code,
        "channels": channels,
        "sample_rate": sample_rate,
        "byte_rate": byte_rate,
        "block_align": block_align,
        "bits_per_sample": bits_per_sample,
        "sample_width_bytes": block_align // channels,
        "frame_count": data_size // block_align,
        "data_offset": data_offset,
        "data_size": data_size,
        "fmt_payload": fmt_payload,
    }


def load_predictor_factory(specification: str) -> Callable[..., Any]:
    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("predictor factory must use package.module:callable syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name, None)
    if not callable(factory):
        raise ValueError(f"predictor factory is not callable: {specification}")
    return factory


def _call_predictor(
    predictor: Any, wav_path: Path, *, clip_id: str, sample_rate: int
) -> Any:
    prediction_method = getattr(predictor, "predict", None)
    if callable(prediction_method):
        return prediction_method(wav_path, clip_id=clip_id, sample_rate=sample_rate)
    if callable(predictor):
        return predictor(wav_path, clip_id=clip_id, sample_rate=sample_rate)
    raise TypeError("predictor must be callable or expose a callable predict method")


def predictor_metadata(
    predictor: Any, *, fallback_model_id: str, fallback_model_revision: str
) -> dict[str, Any]:
    model_path = getattr(predictor, "model_path", None)
    funasr_version = getattr(predictor, "funasr_version", None)
    metadata = {
        "backend": str(
            getattr(
                predictor,
                "backend_name",
                f"{predictor.__class__.__module__}.{predictor.__class__.__qualname__}",
            )
        ),
        "model_id": str(getattr(predictor, "model_id", fallback_model_id)),
        "model_revision": str(
            getattr(predictor, "model_revision", fallback_model_revision)
        ),
        "model_path": str(model_path) if model_path is not None else None,
        "funasr_version": (
            str(funasr_version) if funasr_version is not None else None
        ),
        "inference_contract": {
            "granularity": "utterance",
            "extract_embedding": False,
            "result_fields": ["labels", "scores"],
        },
    }
    if model_path is not None:
        snapshot = Path(str(model_path))
        artifact_hashes = {}
        for name in ("model.pt", "config.yaml", "configuration.json", "tokens.txt"):
            artifact = snapshot / name
            if artifact.is_file():
                artifact_hashes[name] = {
                    "bytes": artifact.stat().st_size,
                    "sha256": sha256_file(artifact),
                }
        metadata["snapshot_artifacts"] = artifact_hashes
    else:
        metadata["snapshot_artifacts"] = {}
    return metadata


def _run_contract(
    *,
    input_inventory: Path,
    beat2_root: Path,
    model: dict[str, Any],
    min_score: float,
    min_margin: float,
) -> dict[str, Any]:
    return {
        "artifact_kind": "beat2_audio_emotion_pseudolabel_progress",
        "schema_version": SCHEMA_VERSION,
        "input_inventory": str(input_inventory),
        "input_inventory_sha256": sha256_file(input_inventory),
        "beat2_root": str(beat2_root),
        "model": model,
        "thresholds": {
            "min_score_inclusive": min_score,
            "min_margin_inclusive": min_margin,
        },
        "network_emotions": list(NETWORK_EMOTIONS),
        "mapping": RAW_LABEL_TO_NETWORK_EMOTION,
        "safety_policy": "candidate_only_never_enables_emotion_supervision",
    }


def _load_progress(
    progress_path: Path,
    *,
    contract: dict[str, Any],
    inventory_by_id: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    lines = progress_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty audio SER progress journal: {progress_path}")
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid audio SER progress header: {error}") from error
    expected_hash = record_sha256(contract)
    if (
        not isinstance(header, dict)
        or header.get("record_type") != "run_contract"
        or header.get("contract_sha256") != expected_hash
        or header.get("contract") != contract
    ):
        raise ValueError(
            "Audio SER progress contract does not match the current inventory, "
            "model artifacts, thresholds, or data root"
        )

    existing: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(lines[1:], 2):
        if not line.strip():
            continue
        try:
            journal_record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid audio SER progress record at line {line_number}: {error}"
            ) from error
        if not isinstance(journal_record, dict) or journal_record.get("record_type") != "output":
            raise ValueError(f"Invalid audio SER progress record at line {line_number}")
        output = journal_record.get("output")
        if not isinstance(output, dict):
            raise ValueError(f"Missing progress output at line {line_number}")
        clip_id = _required_string(output.get("clip_id"), "progress.clip_id")
        if clip_id in existing:
            raise ValueError(f"Duplicate clip_id in audio SER progress: {clip_id}")
        source = inventory_by_id.get(clip_id)
        if source is None:
            raise ValueError(f"Progress clip is absent from current inventory: {clip_id}")
        if output.get("source_window_inventory_record_sha256") != record_sha256(source):
            raise ValueError(f"Progress source record changed for {clip_id}")
        existing[clip_id] = output
    return existing


def _prediction_error_payload(error: Exception) -> dict[str, Any]:
    return {
        "type": type(error).__name__,
        "message": str(error),
    }


def build_pseudolabels(
    *,
    input_inventory: Path,
    beat2_root: Path,
    output_dir: Path,
    predictor: Any,
    min_score: float = DEFAULT_MIN_SCORE,
    min_margin: float = DEFAULT_MIN_MARGIN,
    model_id: str = DEFAULT_MODEL_ID,
    model_revision: str = DEFAULT_MODEL_REVISION,
    continue_on_prediction_error: bool = False,
    limit: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    input_inventory = Path(input_inventory).resolve()
    beat2_root = Path(beat2_root).resolve()
    output_dir = Path(output_dir).resolve()
    min_score = _validate_threshold(min_score, "min_score")
    min_margin = _validate_threshold(min_margin, "min_margin")
    records = read_jsonl(input_inventory)
    if limit is not None:
        if isinstance(limit, bool) or int(limit) <= 0:
            raise ValueError("limit must be a positive integer")
        limit = int(limit)
    selected_records = records if limit is None else records[:limit]
    model = predictor_metadata(
        predictor,
        fallback_model_id=model_id,
        fallback_model_revision=model_revision,
    )
    contract = _run_contract(
        input_inventory=input_inventory,
        beat2_root=beat2_root,
        model=model,
        min_score=min_score,
        min_margin=min_margin,
    )
    progress_path = output_dir / PROGRESS_FILENAME
    inventory_by_id = {
        _required_string(record.get("clip_id"), "clip_id"): record for record in records
    }
    if resume and progress_path.is_file():
        existing = _load_progress(
            progress_path,
            contract=contract,
            inventory_by_id=inventory_by_id,
        )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        header = {
            "record_type": "run_contract",
            "contract_sha256": record_sha256(contract),
            "contract": contract,
        }
        atomic_write(progress_path, (stable_json(header) + "\n").encode("utf-8"))
        existing = {}
    source_hash_cache: dict[Path, str] = {}
    outputs_by_id: dict[str, dict[str, Any]] = dict(existing)
    reused_count = 0
    processed_count = 0

    with tempfile.TemporaryDirectory(prefix="beat2_audio_ser_") as temporary_dir:
        temporary_root = Path(temporary_dir)
        for index, source_record in enumerate(selected_records):
            clip_id = _required_string(source_record.get("clip_id"), "clip_id")
            if clip_id in outputs_by_id:
                reused_count += 1
                continue
            if source_record.get("emotion_supervision_mask") is not False:
                raise ValueError(
                    f"{clip_id} must have emotion_supervision_mask=false before pseudo-labeling"
                )
            if source_record.get("emotion_id") not in (None, ""):
                raise ValueError(
                    f"{clip_id} already has emotion_id; refusing to overwrite reviewed semantics"
                )
            audio_path = resolve_audio_path(beat2_root, source_record)
            if audio_path not in source_hash_cache:
                source_hash_cache[audio_path] = sha256_file(audio_path)
            source_hash = source_hash_cache[audio_path]
            temporary_wav = temporary_root / f"window_{index:06d}.wav"
            audio_audit = extract_exact_audio_window(audio_path, source_record, temporary_wav)
            audio_audit["source_audio_sha256"] = source_hash

            prediction_error = None
            try:
                raw_result = _call_predictor(
                    predictor,
                    temporary_wav,
                    clip_id=clip_id,
                    sample_rate=audio_audit["sample_rate"],
                )
                prediction = parse_prediction(raw_result)
                candidate, unresolved_reason = resolve_candidate(
                    prediction, min_score=min_score, min_margin=min_margin
                )
            except Exception as error:
                if not continue_on_prediction_error:
                    raise RuntimeError(f"Audio emotion prediction failed for {clip_id}") from error
                prediction_error = _prediction_error_payload(error)
                prediction = {
                    "labels": [],
                    "scores": [],
                    "raw_model_label": None,
                    "raw_model_score": None,
                    "raw_runner_up_label": None,
                    "raw_runner_up_score": None,
                    "raw_model_margin": None,
                }
                candidate = None
                unresolved_reason = "prediction_error"
            finally:
                # Prediction is synchronous by contract; retaining every 6 s
                # slice would otherwise consume roughly the size of the source
                # corpus again during a full 4,570-window pass.
                temporary_wav.unlink(missing_ok=True)

            pseudolabel = {
                "schema_version": SCHEMA_VERSION,
                "candidate_emotion_id": candidate,
                "candidate_status": "candidate" if candidate else "unresolved",
                "unresolved_reason": unresolved_reason,
                "raw_model_label": prediction["raw_model_label"],
                "raw_model_score": prediction["raw_model_score"],
                "raw_model_margin": prediction["raw_model_margin"],
                "raw_runner_up_label": prediction["raw_runner_up_label"],
                "raw_runner_up_score": prediction["raw_runner_up_score"],
                "raw_labels": prediction["labels"],
                "raw_scores": prediction["scores"],
                "model": model,
                "thresholds": {
                    "min_score_inclusive": min_score,
                    "min_margin_inclusive": min_margin,
                },
                "audio": audio_audit,
                "prediction_error": prediction_error,
                "policy": {
                    "candidate_only": True,
                    "requires_human_review": True,
                    "may_enable_emotion_supervision": False,
                    "unmappable_raw_labels": ["disgusted", "other", "unknown"],
                },
            }
            output = copy.deepcopy(source_record)
            output.update(
                {
                    "emotion_id": None,
                    "emotion_candidate_id": candidate,
                    "emotion_confidence": (
                        float(prediction["raw_model_score"])
                        if prediction["raw_model_score"] is not None
                        else 0.0
                    ),
                    "emotion_source": EMOTION_SOURCE,
                    "emotion_review_status": EMOTION_REVIEW_STATUS,
                    "emotion_supervision_mask": False,
                    "network_semantic_supervision_ready": False,
                    "accepted_for_training": False,
                    "manual_review_required": True,
                    "audio_emotion_pseudolabel": pseudolabel,
                    "source_window_inventory_record_sha256": record_sha256(source_record),
                }
            )
            append_durable_jsonl(
                progress_path,
                {
                    "record_type": "output",
                    "clip_id": clip_id,
                    "output_sha256": record_sha256(output),
                    "output": output,
                },
            )
            outputs_by_id[clip_id] = output
            processed_count += 1

    outputs = [
        outputs_by_id[_required_string(record.get("clip_id"), "clip_id")]
        for record in selected_records
    ]

    if not all(
        record.get("emotion_source") == EMOTION_SOURCE
        and record.get("emotion_review_status") == EMOTION_REVIEW_STATUS
        and record.get("emotion_supervision_mask") is False
        and record.get("emotion_id") is None
        and record.get("accepted_for_training") is False
        for record in outputs
    ):
        raise AssertionError("Fail-closed audio pseudo-label invariants were violated")

    review_queue: list[dict[str, Any]] = []
    for record in outputs:
        queued = copy.deepcopy(record)
        queued["human_review_template"] = {
            "clip_id": record["clip_id"],
            "decision": "confirmed_or_rejected_or_unresolved",
            "emotion_id": record["emotion_candidate_id"],
            "emotion_confidence": None,
            "reviewer_id": "",
            "reviewed_at": "",
            "notes": "",
            "warning": (
                "Audio SER is a pseudo-label only. Confirm performed affect from "
                "the synchronized clip before enabling emotion supervision."
            ),
        }
        review_queue.append(queued)

    output_payload = "".join(stable_json(record) + "\n" for record in outputs).encode(
        "utf-8"
    )
    queue_payload = "".join(
        stable_json(record) + "\n" for record in review_queue
    ).encode("utf-8")
    candidate_counts = Counter(
        record["emotion_candidate_id"] or "unresolved" for record in outputs
    )
    unresolved_counts = Counter(
        record["audio_emotion_pseudolabel"]["unresolved_reason"]
        for record in outputs
        if record["audio_emotion_pseudolabel"]["unresolved_reason"] is not None
    )
    raw_label_counts = Counter(
        record["audio_emotion_pseudolabel"]["raw_model_label"] or "prediction_error"
        for record in outputs
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "BEAT2",
        "input_inventory": str(input_inventory),
        "input_inventory_sha256": sha256_file(input_inventory),
        "beat2_root": str(beat2_root),
        "inventory_record_count": len(records),
        "selected_record_count": len(selected_records),
        "selection_complete": len(selected_records) == len(records),
        "limit": limit,
        "output_record_count": len(outputs),
        "processed_this_run_count": processed_count,
        "reused_from_progress_count": reused_count,
        "review_queue_count": len(review_queue),
        "candidate_count": sum(record["emotion_candidate_id"] is not None for record in outputs),
        "unresolved_count": sum(record["emotion_candidate_id"] is None for record in outputs),
        "prediction_error_count": sum(
            record["audio_emotion_pseudolabel"]["prediction_error"] is not None
            for record in outputs
        ),
        "candidate_counts": dict(sorted(candidate_counts.items())),
        "unresolved_reason_counts": dict(sorted(unresolved_counts.items())),
        "raw_model_label_counts": dict(sorted(raw_label_counts.items())),
        "speaker_count": len({record.get("speaker_key") for record in outputs}),
        "source_clip_count": len({record.get("source_clip_id") for record in outputs}),
        "model": model,
        "official_references": {
            "funasr": OFFICIAL_FUNASR_URL,
            "emotion2vec": OFFICIAL_EMOTION2VEC_URL,
            "model": OFFICIAL_MODEL_URL,
        },
        "thresholds": {
            "min_score_inclusive": min_score,
            "min_margin_inclusive": min_margin,
        },
        "mapping": {
            "network_emotions": list(NETWORK_EMOTIONS),
            "raw_label_to_network_emotion": RAW_LABEL_TO_NETWORK_EMOTION,
        },
        "safety_invariants": {
            "all_emotion_source_audio_ser_pseudo": True,
            "all_emotion_review_status_pseudo_pending_human_review": True,
            "all_emotion_supervision_mask_false": True,
            "all_emotion_id_unset": True,
            "all_accepted_for_training_false": True,
        },
        "output_sha256": {
            "pseudolabels_jsonl": sha256_bytes(output_payload),
            "human_review_queue_jsonl": sha256_bytes(queue_payload),
            "progress_jsonl": sha256_file(progress_path),
        },
        "progress_journal": str(progress_path),
        "progress_contract_sha256": record_sha256(contract),
    }
    summary_payload = (
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    atomic_write(output_dir / f"{OUTPUT_STEM}.jsonl", output_payload)
    atomic_write(output_dir / f"{OUTPUT_STEM}.review_queue.jsonl", queue_payload)
    atomic_write(output_dir / f"{OUTPUT_STEM}.summary.json", summary_payload)
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.predictor_factory:
        factory = load_predictor_factory(args.predictor_factory)
        predictor = factory(
            model_id=args.model_id,
            model_revision=args.model_revision,
            hub=args.hub,
            device=args.device,
        )
    else:
        predictor = FunASRAudioEmotionPredictor(
            model_id=args.model_id,
            model_revision=args.model_revision,
            hub=args.hub,
            device=args.device,
        )
    summary = build_pseudolabels(
        input_inventory=args.input_inventory,
        beat2_root=args.beat2_root,
        output_dir=args.output_dir,
        predictor=predictor,
        min_score=args.min_score,
        min_margin=args.min_margin,
        model_id=args.model_id,
        model_revision=args.model_revision,
        continue_on_prediction_error=args.continue_on_prediction_error,
        limit=args.limit,
        resume=args.resume,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
