import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.human_motion_review import finalize_interact_full_dyad_review_shard_v3 as mod


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fixture(tmp_path: Path, queue_kind: str) -> tuple[Path, Path, dict]:
    video = tmp_path / "public/videos/anonymous.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"independent-anonymous-video")
    row = {
        "schema_version": "3.0.0",
        "protocol_version": f"protocol-{queue_kind}",
        "sample_id": "anonymous",
        "video_path": str(video),
        "video_sha256": mod.sha256_file(video),
        "native_variable_length": True,
        "fixed_duration_window_used": False,
        "accepted_for_training": False,
    }
    queue = tmp_path / "shard/review_queue.jsonl"
    queue.parent.mkdir(parents=True)
    queue.write_text(mod.stable_json(row) + "\n", encoding="utf-8")
    summary = {
        "artifact_kind": mod.SHARD_KIND,
        "queue_kind": queue_kind,
        "shard_index": 0,
        "shard_count": 1,
        "review_queue": str(queue),
        "review_queue_sha256": mod.sha256_file(queue),
        "source_public_summary_sha256": "a" * 64,
        "source_queue_sha256": "b" * 64,
        "fixed_duration_window_used": False,
        "accepted_for_training": False,
    }
    summary_path = tmp_path / "shard/summary.json"
    write_json(summary_path, summary)
    return summary_path, video, row


def arc_decision() -> dict:
    return {
        "full_video_reviewed": True,
        "onset_status": "complete",
        "onset_evidence_frame": 2,
        "apex_status": "complete",
        "apex_evidence_frame": 10,
        "offset_status": "complete",
        "offset_evidence_frame": 18,
        "interaction_observable_result": "observable",
        "interaction_description_en": "The pair coordinate a greeting gesture.",
        "robot_a_observable_motion_en": "Robot A raises and lowers one arm.",
        "robot_b_observable_motion_en": "Robot B turns and responds with one arm.",
        "expression_completeness_result": "complete",
        "expansion_request": None,
        "review_notes": "Reviewed silently from frame zero through EOF.",
    }


def actor_affect(result: str = "observable") -> dict:
    return {
        "observability_result": result,
        "predicted_class": "neutral" if result == "observable" else None,
        "confidence": 0.8 if result == "observable" else None,
        "intensity": "low" if result == "observable" else None,
        "evidence_frames": [2, 10, 18],
    }


def test_finalize_arc_binds_queue_and_decisions(tmp_path, monkeypatch):
    summary, _, queued = fixture(tmp_path, "arc_action")
    decisions = tmp_path / "decisions.json"
    write_json(decisions, {"anonymous": arc_decision()})
    monkeypatch.setattr(
        mod,
        "probe_video",
        lambda _: {"frames": 20, "fps": 30.0, "width": 1280, "height": 720},
    )

    result = mod.finalize(
        shard_summary_path=summary,
        decisions_path=decisions,
        reviewer_id="arc-reviewer-01",
        output_submission_path=tmp_path / "out/submission.jsonl",
        output_summary_path=tmp_path / "out/summary.json",
    )

    row = mod.read_jsonl(tmp_path / "out/submission.jsonl")[0]
    assert result["records"] == 1
    assert row["decoded_frame_count"] == 20
    assert row["blind_review_provenance"]["queue_record_sha256"] == mod.value_sha256(queued)
    assert row["accepted_for_training"] is False


def test_finalize_rejects_incomplete_decision_coverage(tmp_path, monkeypatch):
    summary, _, _ = fixture(tmp_path, "arc_action")
    decisions = tmp_path / "decisions.json"
    write_json(decisions, {})
    monkeypatch.setattr(mod, "probe_video", lambda _: {})

    with pytest.raises(ValueError, match="coverage"):
        mod.finalize(
            shard_summary_path=summary,
            decisions_path=decisions,
            reviewer_id="arc-reviewer-01",
            output_submission_path=tmp_path / "out/submission.jsonl",
            output_summary_path=tmp_path / "out/summary.json",
        )


def test_finalize_rejects_affect_pseudo_label(tmp_path, monkeypatch):
    summary, _, _ = fixture(tmp_path, "affect")
    invalid = actor_affect("ambiguous")
    invalid["predicted_class"] = "happy"
    decisions = tmp_path / "decisions.json"
    write_json(
        decisions,
        {
            "anonymous": {
                "full_video_reviewed": True,
                "robot_a": invalid,
                "robot_b": actor_affect(),
                "interaction_affect_relation_en": None,
                "review_notes": "Reviewed silently from frame zero through EOF.",
            }
        },
    )
    monkeypatch.setattr(
        mod,
        "probe_video",
        lambda _: {"frames": 20, "fps": 30.0, "width": 1280, "height": 720},
    )

    with pytest.raises(ValueError, match="pseudo-label"):
        mod.finalize(
            shard_summary_path=summary,
            decisions_path=decisions,
            reviewer_id="affect-reviewer-01",
            output_submission_path=tmp_path / "out/submission.jsonl",
            output_summary_path=tmp_path / "out/summary.json",
        )


def test_arc_phase_completeness_must_agree():
    decision = arc_decision()
    decision["offset_status"] = "incomplete"
    with pytest.raises(ValueError, match="disagreement"):
        mod._validate_arc_decision("sample", decision, 20)


def test_probe_video_falls_back_to_complete_opencv_decode(tmp_path, monkeypatch):
    video = tmp_path / "review.avi"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (1280, 720)
    )
    assert writer.isOpened()
    for value in (20, 80, 140):
        writer.write(np.full((720, 1280, 3), value, dtype=np.uint8))
    writer.release()
    monkeypatch.setattr(mod.shutil, "which", lambda _: None)

    result = mod.probe_video(video)

    assert result == {"frames": 3, "fps": 30.0, "width": 1280, "height": 720}
