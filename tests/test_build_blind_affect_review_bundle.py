import hashlib
import json
from pathlib import Path

import pytest

from tools.human_motion_review.adjudicate_training_dataset import (
    BLIND_AFFECT_PROTOCOL_VERSION,
)
from tools.human_motion_review.build_blind_affect_review_bundle import (
    BLIND_PROTOCOL_VERSION,
    PUBLIC_RECORD_KEYS,
    build_bundle,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(tmp_path: Path, *, task_id: str = "task-visible") -> tuple[Path, Path]:
    video = tmp_path / f"{task_id}.mp4"
    video.write_bytes(b"opaque-video-evidence")
    record = {
        "task_id": task_id,
        "status": "passed",
        "accepted_for_training": False,
        "render_pass_grants_training_admission": False,
        "emotion_supervision_mask": False,
        "official_emotion_conditioning_enabled": False,
        "affect_observable_review_status": "candidate_unreviewed",
        "affect_observable_supervision_mask": False,
        "emotion_id": "happy",
        "semantic_event": {"category": "deictic", "intensity": "low"},
        "source_clip_id": "source-visible",
        "speaker_key": "speaker-visible",
        "fixed_split_assignment": "test",
        "video_path": str(video),
        "video_sha256": _sha(video),
        "video_check": {"passed": True, "audio_streams": 0},
    }
    manifest = tmp_path / "passed.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return manifest, video


def test_public_bundle_is_anonymous_and_hidden_mapping_is_separate(tmp_path: Path):
    manifest, source_video = _source(tmp_path)
    output = tmp_path / "bundle"
    result = build_bundle([manifest], output, secret_hex="11" * 32)

    public = json.loads(
        (output / "public_A/blind_review_queue.jsonl").read_text(encoding="utf-8")
    )
    hidden = json.loads(
        (output / "hidden_B/sample_mapping.jsonl").read_text(encoding="utf-8")
    )
    assert set(public) == PUBLIC_RECORD_KEYS
    assert public["sample_id"].startswith("affect_")
    assert public["blind_protocol_version"] == BLIND_PROTOCOL_VERSION
    assert BLIND_PROTOCOL_VERSION == BLIND_AFFECT_PROTOCOL_VERSION
    assert "task-visible" not in json.dumps(public)
    assert "source-visible" not in json.dumps(public)
    assert "speaker-visible" not in json.dumps(public)
    assert "happy" not in json.dumps(public)
    assert "deictic" not in json.dumps(public)
    assert Path(public["video_path"]).name == f'{public["sample_id"]}.mp4'
    assert _sha(Path(public["video_path"])) == _sha(source_video)
    assert hidden["task_id"] == "task-visible"
    assert hidden["official_emotion"] == "happy"
    assert hidden["official_category"] == "deictic"
    assert result["public_A"]["accepted_for_training"] is False
    assert result["hidden_B"]["accepted_for_training"] is False


def test_bundle_rejects_enabled_affect_supervision(tmp_path: Path):
    manifest, _ = _source(tmp_path)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["affect_observable_supervision_mask"] = True
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fail-closed"):
        build_bundle([manifest], tmp_path / "bundle", secret_hex="22" * 32)


def test_hidden_root_can_be_physically_separated(tmp_path: Path):
    manifest, _ = _source(tmp_path)
    public_output = tmp_path / "reviewer_visible"
    hidden_output = tmp_path / "controller_only"
    build_bundle(
        [manifest],
        public_output,
        secret_hex="33" * 32,
        hidden_root=hidden_output,
    )
    assert (public_output / "public_A/blind_review_queue.jsonl").is_file()
    assert not (public_output / "hidden_B").exists()
    assert (hidden_output / "sample_mapping.jsonl").is_file()
