import csv
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from tools.human_motion_review import render_beat2_annotation_review as renderer
from tools.human_motion_review.build_expression_turn_blind_review_bundle_v8 import (
    ACTION_PROTOCOL,
    AFFECT_PROTOCOL,
    ARC_ACTION_PUBLIC_KEYS,
    ARC_PROTOCOL,
    AFFECT_PUBLIC_KEYS,
    FORBIDDEN_PUBLIC_KEYS,
    _materialize_video,
    build_bundle,
    sha256,
    value_sha256,
)


def _write_video(path, frame_count):
    frames = []
    for index in range(frame_count):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        offset = 2 + index * 8
        frame[8:36, offset : offset + 18] = (40, 160, 240)
        frames.append(frame)
    imageio.mimwrite(
        path,
        frames,
        fps=renderer.FPS,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=2,
        output_params=["-movflags", "+faststart"],
    )


def _write_trajectory(path, frame_count):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(renderer.JOINT_ORDER)
        for frame_index in range(frame_count):
            writer.writerow(
                [
                    (frame_index + joint_index) / 1000.0
                    for joint_index in range(len(renderer.JOINT_ORDER))
                ]
            )


def _refresh_binding(record):
    record["video_sha256"] = sha256(Path(record["video_path"]))
    record["trajectory_sha256"] = sha256(Path(record["trajectory_path"]))
    binding = record["final_output_binding"]
    binding["video_sha256"] = record["video_sha256"]
    binding["trajectory_sha256"] = record["trajectory_sha256"]
    binding["sha256"] = value_sha256(
        {key: value for key, value in binding.items() if key != "sha256"}
    )


def _render_record(tmp_path, frame_count=3):
    video = tmp_path / "source.mp4"
    _write_video(video, frame_count)
    trajectory = tmp_path / "final.csv"
    _write_trajectory(trajectory, frame_count)
    quality = tmp_path / "quality.json"
    quality.write_text('{"passed":true}\n', encoding="utf-8")
    video_check = renderer.validate_video(
        video,
        expected_frames=frame_count,
        expected_width=64,
        expected_height=48,
    )
    binding = {
        "trajectory_path": str(trajectory.resolve()),
        "trajectory_sha256": sha256(trajectory),
        "output_frame_count": frame_count,
        "fps": 30.0,
        "video_path": str(video.resolve()),
        "video_sha256": sha256(video),
        "video_decoded_frames": frame_count,
    }
    binding["sha256"] = value_sha256(binding)
    return {
        "task_id": "turn-v8",
        "status": "passed",
        "accepted_for_training": False,
        "render_pass_grants_training_admission": False,
        "emotion_supervision_mask": False,
        "official_emotion_conditioning_enabled": False,
        "affect_observable_supervision_mask": False,
        "expression_turn_selection_kind": "representative100",
        "source_clip_id": "source-a",
        "speaker_key": "speaker-a",
        "fixed_split_assignment": "train",
        "emotion_id": "fear",
        "expression_turn": {
            "official_categories": ["deictic"],
            "complete_motion_arc_verified": False,
        },
        "context_plan": {"selected_level": 0},
        "trajectory_path": str(trajectory.resolve()),
        "trajectory_sha256": sha256(trajectory),
        "trajectory_frames": frame_count,
        "trajectory_frames_expected": frame_count,
        "quality_json": str(quality.resolve()),
        "quality_json_sha256": sha256(quality),
        "output_frame_count": frame_count,
        "final_trajectory_role": "native_identity_timeline_no_slowdown_required",
        "blind_review_must_use_final_trajectory": True,
        "video_path": str(video.resolve()),
        "video_sha256": sha256(video),
        "video_check": video_check,
        "final_output_binding": binding,
        "expression_turn_record_sha256": "1" * 64,
        "expression_turn_selection_record_sha256": "2" * 64,
        "selected_record_sha256": "3" * 64,
        "retarget_input_manifest_sha256": "4" * 64,
    }


def _read_one(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_bundle_uses_same_anonymous_video_but_separate_review_protocols(tmp_path):
    manifest = tmp_path / "render_passed.jsonl"
    manifest.write_text(json.dumps(_render_record(tmp_path)) + "\n", encoding="utf-8")
    output = tmp_path / "bundle"
    hidden = tmp_path / "private"
    summary = build_bundle(
        manifest,
        output,
        selection_kind="representative100",
        hidden_root=hidden,
        secret_hex="11" * 32,
    )
    arc = _read_one(output / "public/arc_action_review_queue.jsonl")
    affect = _read_one(output / "public/affect_review_queue.jsonl")
    mapping = _read_one(hidden / "sample_mapping.jsonl")
    assert set(arc) == ARC_ACTION_PUBLIC_KEYS
    assert set(affect) == AFFECT_PUBLIC_KEYS
    assert arc["sample_id"] == affect["sample_id"]
    assert arc["video_path"] == affect["video_path"]
    assert arc["video_sha256"] == affect["video_sha256"]
    assert arc["arc_protocol_version"] == ARC_PROTOCOL
    assert arc["action_protocol_version"] == ACTION_PROTOCOL
    assert affect["affect_protocol_version"] == AFFECT_PROTOCOL
    assert affect["allowed_classes"] == [
        "angry",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprise",
    ]
    assert summary["public"]["affect_ontology"] == affect["allowed_classes"]
    assert arc["audio_available"] is False
    assert affect["label_metadata_exposed"] is False
    assert summary["public"][
        "same_anonymous_silent_video_used_by_both_reviews"
    ] is True
    assert mapping["task_id"] == "turn-v8"
    assert mapping["official_emotion"] == "fear"
    assert mapping["official_categories"] == ["deictic"]
    assert (os.stat(hidden).st_mode & 0o777) == 0o700
    assert (os.stat(hidden / "sample_mapping.jsonl").st_mode & 0o777) == 0o600
    source = Path(mapping["source_video_path"])
    anonymous = Path(mapping["anonymous_video_path"])
    assert os.stat(source).st_ino != os.stat(anonymous).st_ino


def test_public_queues_have_no_identity_action_or_official_emotion_keys(tmp_path):
    manifest = tmp_path / "render_passed.jsonl"
    manifest.write_text(json.dumps(_render_record(tmp_path)) + "\n", encoding="utf-8")
    output = tmp_path / "bundle"
    build_bundle(
        manifest,
        output,
        selection_kind="representative100",
        hidden_root=tmp_path / "private",
        secret_hex="22" * 32,
    )
    for path in (
        output / "public/arc_action_review_queue.jsonl",
        output / "public/affect_review_queue.jsonl",
    ):
        record = _read_one(path)
        lowered = {key.lower() for key in record}
        assert lowered.isdisjoint(FORBIDDEN_PUBLIC_KEYS)
        assert not any(key.startswith("official_") for key in lowered)
        assert not any(key.startswith("source_") for key in lowered)
        payload_record = dict(record)
        payload_record.pop("allowed_classes", None)
        payload = json.dumps(payload_record, ensure_ascii=False).lower()
        assert "fear" not in payload
        assert "deictic" not in payload
        assert "speaker-a" not in payload
        assert "turn-v8" not in payload


def test_bundle_rejects_wrong_selection_or_audio(tmp_path):
    record = _render_record(tmp_path)
    manifest = tmp_path / "render_passed.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="selection_kind"):
        build_bundle(
            manifest,
            tmp_path / "bundle-a",
            selection_kind="stress100",
            hidden_root=tmp_path / "private-a",
            secret_hex="33" * 32,
        )

    record["video_check"]["audio_streams"] = 1
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="silent/nonblank"):
        build_bundle(
            manifest,
            tmp_path / "bundle-b",
            selection_kind="representative100",
            hidden_root=tmp_path / "private-b",
            secret_hex="44" * 32,
        )


def test_bundle_rejects_self_consistent_fake_video_declarations(tmp_path):
    record = _render_record(tmp_path)
    video = Path(record["video_path"])
    video.write_bytes(b"not-an-mp4" * 256)
    _refresh_binding(record)
    manifest = tmp_path / "render_passed.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="moov atom"):
        build_bundle(
            manifest,
            tmp_path / "bundle",
            selection_kind="representative100",
            hidden_root=tmp_path / "private",
            secret_hex="55" * 32,
        )


def test_bundle_rejects_csv_actual_frame_count_mismatch(tmp_path):
    record = _render_record(tmp_path, frame_count=3)
    _write_trajectory(Path(record["trajectory_path"]), frame_count=2)
    _refresh_binding(record)
    manifest = tmp_path / "render_passed.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="trajectory frames do not match queue contract"):
        build_bundle(
            manifest,
            tmp_path / "bundle",
            selection_kind="representative100",
            hidden_root=tmp_path / "private",
            secret_hex="66" * 32,
        )


def test_bundle_rejects_quality_json_hash_mismatch(tmp_path):
    record = _render_record(tmp_path)
    Path(record["quality_json"]).write_text('{"passed":false}\n', encoding="utf-8")
    manifest = tmp_path / "render_passed.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="quality_json_sha256 mismatch"):
        build_bundle(
            manifest,
            tmp_path / "bundle",
            selection_kind="representative100",
            hidden_root=tmp_path / "private",
            secret_hex="77" * 32,
        )


def test_materialize_detaches_existing_hardlink(tmp_path):
    source = tmp_path / "source.mp4"
    _write_video(source, frame_count=3)
    target = tmp_path / "public" / "anonymous.mp4"
    target.parent.mkdir()
    os.link(source, target)
    assert os.path.samefile(source, target)

    _materialize_video(source, target, sha256(source))

    assert sha256(target) == sha256(source)
    assert not os.path.samefile(source, target)
    assert target.stat().st_nlink == 1


@pytest.mark.parametrize("alias_kind", ("symlink", "hardlink"))
def test_materialize_detaches_alias_to_unrelated_same_hash_file(tmp_path, alias_kind):
    source = tmp_path / "source.mp4"
    _write_video(source, frame_count=3)
    external = tmp_path / "external.mp4"
    external.write_bytes(source.read_bytes())
    target = tmp_path / "public" / "anonymous.mp4"
    target.parent.mkdir()
    if alias_kind == "symlink":
        target.symlink_to(external)
    else:
        os.link(external, target)

    _materialize_video(source, target, sha256(source))

    assert not target.is_symlink()
    assert target.stat().st_nlink == 1
    assert not os.path.samefile(target, external)
    assert sha256(target) == sha256(source)


def test_base_bundle_rejects_public_hidden_overlap_and_public_symlink(tmp_path):
    manifest = tmp_path / "render_passed.jsonl"
    manifest.write_text(json.dumps(_render_record(tmp_path)) + "\n", encoding="utf-8")
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="must be disjoint"):
        build_bundle(
            manifest,
            output,
            selection_kind="representative100",
            hidden_root=output / "public",
            secret_hex="88" * 32,
        )

    output.mkdir()
    hidden = tmp_path / "private"
    hidden.mkdir()
    (output / "public").symlink_to(hidden, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        build_bundle(
            manifest,
            output,
            selection_kind="representative100",
            hidden_root=hidden,
            secret_hex="99" * 32,
        )


def test_base_bundle_rejects_stale_sensitive_public_entry(tmp_path):
    manifest = tmp_path / "render_passed.jsonl"
    manifest.write_text(json.dumps(_render_record(tmp_path)) + "\n", encoding="utf-8")
    arguments = {
        "render_passed_manifest": manifest,
        "output_root": tmp_path / "bundle",
        "selection_kind": "representative100",
        "hidden_root": tmp_path / "private",
        "secret_hex": "aa" * 32,
    }
    build_bundle(**arguments)
    stale = tmp_path / "bundle/public/sample_mapping.jsonl"
    stale.write_text(
        '{"official_emotion":"fear","speaker_key":"private"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="stale unexpected public bundle entries"):
        build_bundle(**arguments)
