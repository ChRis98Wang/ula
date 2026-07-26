import hashlib
import importlib.util
import json
import struct
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools/human_motion_collection/pseudolabel_beat2_audio_emotion.py"
)
SPEC = importlib.util.spec_from_file_location(
    "pseudolabel_beat2_audio_emotion", SCRIPT_PATH
)
assert SPEC and SPEC.loader
PSEUDO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PSEUDO)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    pcm = np.asarray(samples, dtype="<i2").tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return pcm


def write_inventory(
    root: Path,
    *,
    start: int = 3,
    end: int = 8,
    clip_id: str = "12_zhao_2_1_1_f000003-000008",
) -> tuple[Path, Path, bytes, dict]:
    audio_dir = root / "wave16k"
    audio_dir.mkdir(parents=True)
    audio_path = audio_dir / "12_zhao_2_1_1.wav"
    source_pcm = write_wav(audio_path, np.arange(12, dtype=np.int16))
    record = {
        "schema_version": "1.0.0",
        "dataset": "BEAT2",
        "clip_id": clip_id,
        "window_id": clip_id,
        "task_id": clip_id,
        "source_clip_id": "12_zhao_2_1_1",
        "source_group_id": "12_zhao_2_1_1",
        "split_group_id": "12_zhao_2_1_1",
        "speaker_key": "12_zhao",
        "official_split": "train",
        "audio_relpath": "wave16k/12_zhao_2_1_1.wav",
        "audio_sample_rate": 16000,
        "audio_channels": 1,
        "audio_frame_count": 12,
        "behavior_id": None,
        "emotion_id": None,
        "emotion_supervision_mask": False,
        "accepted_for_training": False,
        "window": {
            "audio_start_sample": start,
            "audio_end_sample_exclusive": end,
            "audio_sample_count": end - start,
        },
    }
    inventory = root / "windows.jsonl"
    inventory.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return inventory, audio_path, source_pcm, record


class CapturingPredictor:
    backend_name = "unit_test_mock"
    model_id = "mock/emotion2vec"
    model_revision = "test-revision"

    def __init__(self, result):
        self.result = result
        self.calls = []

    def predict(self, wav_path, *, clip_id, sample_rate):
        with wave.open(str(wav_path), "rb") as handle:
            self.calls.append(
                {
                    "clip_id": clip_id,
                    "sample_rate": sample_rate,
                    "frame_count": handle.getnframes(),
                    "pcm": handle.readframes(handle.getnframes()),
                }
            )
        return self.result


def run_fixture(tmp_path: Path, predictor, **kwargs):
    inventory, audio_path, source_pcm, source_record = write_inventory(tmp_path)
    output_dir = tmp_path / "output"
    summary = PSEUDO.build_pseudolabels(
        input_inventory=inventory,
        beat2_root=tmp_path,
        output_dir=output_dir,
        predictor=predictor,
        **kwargs,
    )
    output = json.loads(
        (output_dir / f"{PSEUDO.OUTPUT_STEM}.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    queue = json.loads(
        (output_dir / f"{PSEUDO.OUTPUT_STEM}.review_queue.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    return summary, output, queue, audio_path, source_pcm, source_record


def test_exact_sample_slice_is_hashed_and_high_confidence_is_candidate_only(tmp_path):
    predictor = CapturingPredictor(
        [{"labels": ["neutral", "happy", "sad"], "scores": [0.05, 0.81, 0.14]}]
    )

    summary, output, queue, audio_path, source_pcm, source_record = run_fixture(
        tmp_path, predictor, min_score=0.6, min_margin=0.2
    )

    expected_pcm = source_pcm[3 * 2 : 8 * 2]
    assert predictor.calls == [
        {
            "clip_id": source_record["clip_id"],
            "sample_rate": 16000,
            "frame_count": 5,
            "pcm": expected_pcm,
        }
    ]
    assert output["source_group_id"] == source_record["source_group_id"]
    assert output["split_group_id"] == source_record["split_group_id"]
    assert output["speaker_key"] == "12_zhao"
    assert output["emotion_id"] is None
    assert output["emotion_candidate_id"] == "happy"
    assert output["emotion_source"] == "audio_ser_pseudo"
    assert output["emotion_review_status"] == "pseudo_pending_human_review"
    assert output["emotion_supervision_mask"] is False
    assert output["accepted_for_training"] is False
    pseudo = output["audio_emotion_pseudolabel"]
    assert pseudo["raw_model_label"] == "happy"
    assert pseudo["raw_model_score"] == pytest.approx(0.81)
    assert pseudo["raw_model_margin"] == pytest.approx(0.67)
    assert pseudo["model"]["model_revision"] == "test-revision"
    assert pseudo["audio"]["audio_start_sample"] == 3
    assert pseudo["audio"]["audio_end_sample_exclusive"] == 8
    assert pseudo["audio"]["window_sample_bytes_sha256"] == hashlib.sha256(
        expected_pcm
    ).hexdigest()
    assert pseudo["audio"]["source_audio_sha256"] == PSEUDO.sha256_file(audio_path)
    assert queue["human_review_template"]["emotion_id"] == "happy"
    assert summary["candidate_counts"] == {"happy": 1}
    assert summary["safety_invariants"]["all_emotion_id_unset"] is True


@pytest.mark.parametrize(
    ("result", "reason"),
    [
        ({"labels": ["happy", "sad"], "scores": [0.54, 0.20]}, "below_min_score"),
        ({"labels": ["angry", "sad"], "scores": [0.70, 0.65]}, "below_min_margin"),
        (
            {"labels": ["disgusted", "neutral"], "scores": [0.91, 0.09]},
            "raw_label_not_mappable_to_network_six_class_ontology",
        ),
    ],
)
def test_low_confidence_and_unmappable_predictions_stay_unresolved(
    tmp_path, result, reason
):
    predictor = CapturingPredictor(result)

    summary, output, _queue, _audio_path, _pcm, _record = run_fixture(
        tmp_path, predictor, min_score=0.55, min_margin=0.10
    )

    assert output["emotion_id"] is None
    assert output["emotion_candidate_id"] is None
    assert output["emotion_supervision_mask"] is False
    assert output["audio_emotion_pseudolabel"]["candidate_status"] == "unresolved"
    assert output["audio_emotion_pseudolabel"]["unresolved_reason"] == reason
    assert summary["unresolved_reason_counts"] == {reason: 1}


@pytest.mark.parametrize(
    ("raw_label", "expected"),
    [
        ("fearful", "fear"),
        ("surprised", "surprise"),
        ("emo_angry", "angry"),
        ("生气/angry", "angry"),
        ("other", None),
        ("unknown-new-label", None),
    ],
)
def test_raw_label_mapping_is_explicit(raw_label, expected):
    assert PSEUDO.map_raw_label(raw_label) == expected


def test_numpy_scores_from_predictor_are_accepted():
    prediction = PSEUDO.parse_prediction(
        {"labels": ["neutral", "sad"], "scores": [np.float32(0.8), np.float32(0.2)]}
    )

    assert prediction["raw_model_label"] == "neutral"
    assert prediction["raw_model_score"] == pytest.approx(0.8)


def test_prediction_error_is_fail_closed_when_explicitly_continued(tmp_path):
    class BrokenPredictor(CapturingPredictor):
        def predict(self, wav_path, *, clip_id, sample_rate):
            raise RuntimeError("mock failure")

    predictor = BrokenPredictor(None)
    summary, output, _queue, _audio_path, _pcm, _record = run_fixture(
        tmp_path, predictor, continue_on_prediction_error=True
    )

    assert output["emotion_candidate_id"] is None
    assert output["emotion_id"] is None
    assert output["emotion_supervision_mask"] is False
    assert output["audio_emotion_pseudolabel"]["unresolved_reason"] == "prediction_error"
    assert output["audio_emotion_pseudolabel"]["prediction_error"] == {
        "type": "RuntimeError",
        "message": "mock failure",
    }
    assert summary["prediction_error_count"] == 1


def test_refuses_out_of_bounds_window_before_calling_predictor(tmp_path):
    inventory, _audio_path, _source_pcm, record = write_inventory(tmp_path)
    record["window"]["audio_end_sample_exclusive"] = 13
    record["window"]["audio_sample_count"] = 10
    inventory.write_text(json.dumps(record) + "\n", encoding="utf-8")
    predictor = CapturingPredictor({"labels": ["neutral"], "scores": [1.0]})

    with pytest.raises(ValueError, match="exceeds 12 source samples"):
        PSEUDO.build_pseudolabels(
            input_inventory=inventory,
            beat2_root=tmp_path,
            output_dir=tmp_path / "output",
            predictor=predictor,
        )
    assert predictor.calls == []


def test_ieee_float32_wav_is_sliced_without_quantization(tmp_path):
    audio_dir = tmp_path / "wave16k"
    audio_dir.mkdir()
    audio_path = audio_dir / "float.wav"
    samples = np.asarray([0.0, 0.125, -0.25, 0.5, -0.75], dtype="<f4")
    fmt_payload = struct.pack("<HHIIHH", 3, 1, 16000, 64000, 4, 32)
    sample_bytes = samples.tobytes()
    body = (
        b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt_payload))
        + fmt_payload
        + b"data"
        + struct.pack("<I", len(sample_bytes))
        + sample_bytes
    )
    audio_path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    record = {
        "clip_id": "float_f000001-000004",
        "audio_relpath": "wave16k/float.wav",
        "audio_sample_rate": 16000,
        "audio_channels": 1,
        "audio_frame_count": 5,
        "audio_format": "ieee_float",
        "emotion_id": None,
        "emotion_supervision_mask": False,
        "accepted_for_training": False,
        "window": {
            "audio_start_sample": 1,
            "audio_end_sample_exclusive": 4,
            "audio_sample_count": 3,
        },
    }
    inventory = tmp_path / "float.jsonl"
    inventory.write_text(json.dumps(record) + "\n", encoding="utf-8")

    class FloatAwarePredictor(CapturingPredictor):
        def predict(self, wav_path, *, clip_id, sample_rate):
            metadata = PSEUDO.read_riff_wave_metadata(wav_path)
            with Path(wav_path).open("rb") as handle:
                handle.seek(metadata["data_offset"])
                raw = handle.read(metadata["data_size"])
            self.calls.append((metadata, raw))
            return {"labels": ["fearful", "neutral"], "scores": [0.9, 0.1]}

    predictor = FloatAwarePredictor(None)
    summary = PSEUDO.build_pseudolabels(
        input_inventory=inventory,
        beat2_root=tmp_path,
        output_dir=tmp_path / "output",
        predictor=predictor,
    )

    metadata, raw = predictor.calls[0]
    assert metadata["format_code"] == 3
    assert metadata["frame_count"] == 3
    assert raw == samples[1:4].tobytes()
    assert summary["candidate_counts"] == {"fear": 1}


def test_refuses_to_overwrite_human_reviewed_emotion(tmp_path):
    inventory, _audio_path, _source_pcm, record = write_inventory(tmp_path)
    record["emotion_id"] = "happy"
    record["emotion_supervision_mask"] = True
    inventory.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="emotion_supervision_mask=false"):
        PSEUDO.build_pseudolabels(
            input_inventory=inventory,
            beat2_root=tmp_path,
            output_dir=tmp_path / "output",
            predictor=CapturingPredictor({"labels": ["neutral"], "scores": [1.0]}),
        )


def test_limit_then_resume_reuses_durable_progress(tmp_path):
    inventory, _audio_path, _source_pcm, first = write_inventory(tmp_path)
    second = json.loads(json.dumps(first))
    second["clip_id"] = "12_zhao_2_1_1_f000008-000012"
    second["window"]["audio_start_sample"] = 8
    second["window"]["audio_end_sample_exclusive"] = 12
    second["window"]["audio_sample_count"] = 4
    inventory.write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8"
    )
    output_dir = tmp_path / "output"
    predictor = CapturingPredictor(
        {"labels": ["neutral", "happy"], "scores": [0.8, 0.2]}
    )

    partial = PSEUDO.build_pseudolabels(
        input_inventory=inventory,
        beat2_root=tmp_path,
        output_dir=output_dir,
        predictor=predictor,
        limit=1,
    )
    resumed = PSEUDO.build_pseudolabels(
        input_inventory=inventory,
        beat2_root=tmp_path,
        output_dir=output_dir,
        predictor=predictor,
        resume=True,
    )
    repeated = PSEUDO.build_pseudolabels(
        input_inventory=inventory,
        beat2_root=tmp_path,
        output_dir=output_dir,
        predictor=predictor,
        resume=True,
    )

    assert partial["selected_record_count"] == 1
    assert partial["selection_complete"] is False
    assert resumed["processed_this_run_count"] == 1
    assert resumed["reused_from_progress_count"] == 1
    assert repeated["processed_this_run_count"] == 0
    assert repeated["reused_from_progress_count"] == 2
    assert len(predictor.calls) == 2
    records = [
        json.loads(line)
        for line in (output_dir / f"{PSEUDO.OUTPUT_STEM}.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["clip_id"] for record in records] == [
        first["clip_id"],
        second["clip_id"],
    ]


def test_resume_rejects_changed_threshold_contract(tmp_path):
    inventory, _audio_path, _source_pcm, _record = write_inventory(tmp_path)
    output_dir = tmp_path / "output"
    predictor = CapturingPredictor({"labels": ["neutral"], "scores": [1.0]})
    PSEUDO.build_pseudolabels(
        input_inventory=inventory,
        beat2_root=tmp_path,
        output_dir=output_dir,
        predictor=predictor,
    )

    with pytest.raises(ValueError, match="progress contract does not match"):
        PSEUDO.build_pseudolabels(
            input_inventory=inventory,
            beat2_root=tmp_path,
            output_dir=output_dir,
            predictor=predictor,
            min_score=0.9,
            resume=True,
        )


def test_cli_predictor_factory_is_injectable_without_funasr(monkeypatch):
    module = types.ModuleType("test_audio_ser_plugin")
    sentinel = object()
    module.make_predictor = lambda **kwargs: (sentinel, kwargs)
    monkeypatch.setitem(sys.modules, module.__name__, module)

    factory = PSEUDO.load_predictor_factory("test_audio_ser_plugin:make_predictor")
    predictor, kwargs = factory(
        model_id="mock", model_revision="r1", hub="offline", device="cpu"
    )

    assert predictor is sentinel
    assert kwargs["model_revision"] == "r1"


def test_default_backend_reports_install_requirement_without_downloading(monkeypatch):
    monkeypatch.setitem(sys.modules, "funasr", None)
    with pytest.raises(RuntimeError, match="pip install -U funasr modelscope"):
        PSEUDO.FunASRAudioEmotionPredictor()
