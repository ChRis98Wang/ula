import json
from pathlib import Path

import pytest

from tools.human_motion_review import finalize_beat_affect_decisions_v8 as mod
from tools.human_motion_review import normalize_expression_turn_affect_submission_v8 as normalizer


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fixture(tmp_path: Path) -> tuple[Path, dict]:
    video = tmp_path / "public/videos/sample.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"anonymous-video")
    row = {
        "schema_version": "1.0.0",
        "sample_id": "sample",
        "video_path": str(video),
        "video_sha256": mod.sha256_file(video),
        "context_level": 5,
        "frame_count": 20,
        "fps": 30,
        "allowed_classes": ["angry", "fear", "happy", "neutral", "sad", "surprise"],
        "affect_protocol_version": mod.PROTOCOL,
        "native_duration_preserved": True,
        "fixed_duration_window_used": False,
        "audio_available": False,
        "label_metadata_exposed": False,
    }
    queue = tmp_path / "public/affect.jsonl"
    queue.write_text(mod.stable_json(row) + "\n", encoding="utf-8")
    summary = {
        "artifact_kind": mod.PUBLIC_KIND,
        "accepted_for_training": False,
        "all_samples_native_variable_length": True,
        "fixed_duration_window_used": False,
        "affect_queue": str(queue),
        "affect_queue_sha256": mod.sha256_file(queue),
    }
    summary_path = tmp_path / "public/summary.json"
    write_json(summary_path, summary)
    return summary_path, row


def decision(result: str = "observable") -> dict:
    return {
        "result": result,
        "predicted_class": "neutral" if result == "observable" else None,
        "confidence": 0.8 if result == "observable" else None,
        "intensity": "low" if result == "observable" else None,
        "evidence": "The robot remains upright with low-amplitude movements.",
        "evidence_frames": [2, 10, 18],
        "full_video_reviewed": True,
    }


def test_finalize_is_compatible_with_affect_normalizer_contract(tmp_path, monkeypatch):
    public, _ = fixture(tmp_path)
    decisions = tmp_path / "decisions.json"
    write_json(decisions, {"sample": decision()})
    monkeypatch.setattr(
        mod,
        "probe_video",
        lambda _: {"frames": 20, "fps": 30.0, "width": 1280, "height": 720},
    )

    submission = tmp_path / "out/submission.jsonl"
    submission_summary = tmp_path / "out/summary.json"
    summary = mod.finalize(
        public_summary_path=public,
        decisions_path=decisions,
        reviewer_id="affect-reviewer",
        output_submission_path=submission,
        output_summary_path=submission_summary,
    )
    row = mod.read_jsonl(submission)[0]
    assert summary["coverage"]["fraction"] == 1
    assert row["decode_started_at_frame"] == 0
    assert row["decode_reached_eof"] is True
    assert row["training_admission"] is False
    normalized = normalizer.normalize_submission(
        public_summary_path=public,
        source_submission_path=submission,
        source_summary_path=submission_summary,
        output_submission_path=tmp_path / "normalized/submission.jsonl",
        output_summary_path=tmp_path / "normalized/summary.json",
    )
    assert normalized["coverage"]["complete"] is True


def test_finalize_rejects_pseudo_label_on_ambiguous_decision(tmp_path, monkeypatch):
    public, _ = fixture(tmp_path)
    invalid = decision("ambiguous")
    invalid["predicted_class"] = "happy"
    decisions = tmp_path / "decisions.json"
    write_json(decisions, {"sample": invalid})
    monkeypatch.setattr(
        mod,
        "probe_video",
        lambda _: {"frames": 20, "fps": 30.0, "width": 1280, "height": 720},
    )
    with pytest.raises(ValueError, match="pseudo-label"):
        mod.finalize(
            public_summary_path=public,
            decisions_path=decisions,
            reviewer_id="affect-reviewer",
            output_submission_path=tmp_path / "out/submission.jsonl",
            output_summary_path=tmp_path / "out/summary.json",
        )


def test_finalize_requires_exact_coverage(tmp_path, monkeypatch):
    public, _ = fixture(tmp_path)
    decisions = tmp_path / "decisions.json"
    write_json(decisions, {})
    monkeypatch.setattr(mod, "probe_video", lambda _: {})
    with pytest.raises(ValueError, match="coverage"):
        mod.finalize(
            public_summary_path=public,
            decisions_path=decisions,
            reviewer_id="affect-reviewer",
            output_submission_path=tmp_path / "out/submission.jsonl",
            output_summary_path=tmp_path / "out/summary.json",
        )
