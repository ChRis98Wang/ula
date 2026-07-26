import hashlib
import json
import os
from pathlib import Path

import pytest

from tools.human_motion_review import build_ula0513_native_blind_review_bundle as BUILD


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixtures(tmp_path: Path):
    task_id = "ula_native_0123456789abcdef01234567"
    trajectory = tmp_path / "private" / "joy_safe.csv"
    trajectory.parent.mkdir()
    trajectory.write_text("synthetic trajectory", encoding="utf-8")
    video = tmp_path / "private" / "render.mp4"
    video.write_bytes(b"synthetic fully verified h264 video")
    mapping = {
        "schema_version": "1.0.0",
        "task_id": task_id,
        "clip_id": "ula0513_native_v1__joy_01",
        "source_clip_id": "Robot_Model0530_V2_Joy01",
        "source_behavior_label": "Joy01",
        "source_behavior_label_role": "source_metadata_pending_review",
        "source_record_sha256": "a" * 64,
        "trajectory_path": str(trajectory),
        "trajectory_sha256": _sha(trajectory),
        "frame_count": 11,
        "sample_span_sec": 10 / 30,
        "source_asset_name_must_not_be_exposed_before_blind_submission": True,
        "accepted_for_training": False,
    }
    render = {
        "schema_version": "1.0.0",
        "task_id": task_id,
        "status": "passed",
        "robot_contract": BUILD.ROBOT_CONTRACT,
        "canonical_action": None,
        "emotion_supervision_mask": False,
        "affect_observable_supervision_mask": False,
        "official_emotion_conditioning_enabled": False,
        "accepted_for_training": False,
        "render_pass_grants_training_admission": False,
        "trajectory_frames": 11,
        "trajectory_sha256": _sha(trajectory),
        "training_segment": {
            "frame_count": 11,
            "fixed_window_sec": None,
            "cropped": False,
            "resampled": False,
            "tiled": False,
        },
        "video_path": str(video),
        "video_sha256": _sha(video),
        "video_check": {
            "passed": True,
            "fully_decodable": True,
            "nonblank": True,
            "audio_streams": 0,
            "video_streams": 1,
            "codec": "h264",
            "pixel_format": "yuv420p",
            "decoded_frames": 11,
            "fps": 30.0,
            "faststart": True,
        },
    }
    mapping_path = tmp_path / "mapping.jsonl"
    passed_path = tmp_path / "passed.jsonl"
    mapping_path.write_text(BUILD.stable_json(mapping) + "\n", encoding="utf-8")
    passed_path.write_text(BUILD.stable_json(render) + "\n", encoding="utf-8")
    return mapping_path, passed_path, mapping, render


def test_bundle_strips_source_label_and_requires_separate_protocols(tmp_path):
    mapping_path, passed_path, mapping, _render = _fixtures(tmp_path)
    output = tmp_path / "bundle"
    hidden = tmp_path / "hidden"
    result = BUILD.build_bundle(
        passed_path,
        mapping_path,
        output,
        hidden,
        secret_hex="11" * 32,
    )
    arc = json.loads((output / "public/arc_action_review_queue.jsonl").read_text())
    affect = json.loads((output / "public/affect_review_queue.jsonl").read_text())
    private = json.loads((hidden / "sample_mapping.jsonl").read_text())
    assert set(arc) == BUILD.ARC_ACTION_PUBLIC_KEYS
    assert set(affect) == BUILD.AFFECT_PUBLIC_KEYS
    assert arc["sample_id"] == affect["sample_id"]
    assert arc["video_sha256"] == affect["video_sha256"]
    assert arc["label_metadata_exposed"] is False
    assert affect["audio_available"] is False
    assert affect["allowed_classes"] == [
        "angry",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprise",
    ]
    assert result["public"]["affect_ontology"] == affect["allowed_classes"]
    assert private["source_behavior_label"] == "Joy01"
    assert result["public"]["fixed_window_sec"] is None
    assert result["public"]["at_least_two_independent_affect_submissions_required"] is True
    public_payload = "".join(
        path.read_text(encoding="utf-8")
        for path in (output / "public").glob("*.json*")
    ).lower()
    assert "joy" not in public_payload
    assert mapping["source_clip_id"].lower() not in public_payload
    assert (os.stat(hidden).st_mode & 0o777) == 0o700
    assert (os.stat(hidden / "sample_mapping.jsonl").st_mode & 0o777) == 0o600


def test_bundle_rejects_missing_task_or_incomplete_video(tmp_path):
    mapping_path, passed_path, _mapping, render = _fixtures(tmp_path)
    passed_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="task set mismatch"):
        BUILD.build_bundle(
            passed_path,
            mapping_path,
            tmp_path / "out-a",
            tmp_path / "hidden-a",
            secret_hex="22" * 32,
        )

    render["video_check"]["fully_decodable"] = False
    passed_path.write_text(BUILD.stable_json(render) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="video verification failed"):
        BUILD.build_bundle(
            passed_path,
            mapping_path,
            tmp_path / "out-b",
            tmp_path / "hidden-b",
            secret_hex="33" * 32,
        )


def test_bundle_rejects_fixed_window_or_emotion_mask(tmp_path):
    mapping_path, passed_path, _mapping, render = _fixtures(tmp_path)
    render["training_segment"]["fixed_window_sec"] = 6.0
    passed_path.write_text(BUILD.stable_json(render) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="native-length segment"):
        BUILD.build_bundle(
            passed_path,
            mapping_path,
            tmp_path / "out-a",
            tmp_path / "hidden-a",
            secret_hex="44" * 32,
        )

    render["training_segment"]["fixed_window_sec"] = None
    render["emotion_supervision_mask"] = True
    passed_path.write_text(BUILD.stable_json(render) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fail-closed"):
        BUILD.build_bundle(
            passed_path,
            mapping_path,
            tmp_path / "out-b",
            tmp_path / "hidden-b",
            secret_hex="55" * 32,
        )
