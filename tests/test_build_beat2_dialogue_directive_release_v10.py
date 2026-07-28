from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.human_motion_collection.build_beat2_dialogue_directive_release_v10 import (
    BASE_V9_CONTRACT,
    BASE_V9_LOCK_KIND,
    build_release,
)
from tools.human_motion_collection.verify_beat2_dialogue_directive_release_v10 import (
    verify_release,
)
from upper_body_skeleton.ula_v2_dialogue_directive_episode import (
    TRAINING_SEGMENT_REPRESENTATION,
    canonical_sha256,
    validate_dialogue_directive_v10_episode,
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_words_textgrid(path: Path, words: list[tuple[float, float, str]]) -> None:
    intervals = "\n".join(
        f'''            intervals [{index}]:
                xmin = {start}
                xmax = {end}
                text = "{text}"'''
        for index, (start, end, text) in enumerate(words, 1)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 2
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 2
        intervals: size = {len(words)}
{intervals}
''',
        encoding="utf-8",
    )


def _fixtures(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    beat2_root = tmp_path / "BEAT2"
    specifications = [
        ("clip_a", "speaker_a", "group_a", "validation", "move calmly", ["hello", "there"]),
        ("clip_b", "speaker_a", "group_b", "validation", "move quickly", ["how", "are", "you"]),
        ("clip_c", "speaker_b", "group_c", "validation", "move broadly", ["that", "is", "good"]),
        ("clip_empty", "speaker_c", "group_d", "train", "move quietly", []),
    ]
    motion_rows = []
    for index, (clip_id, speaker, group, split, _directive, words) in enumerate(
        specifications
    ):
        relpath = f"english/textgrid/{clip_id}.TextGrid"
        if words:
            intervals = [
                (position * 0.2, (position + 1) * 0.2, word)
                for position, word in enumerate(words)
            ]
        else:
            intervals = [(0.0, 1.0, "")]
        _write_words_textgrid(beat2_root / relpath, intervals)
        motion_rows.append(
            {
                "clip_id": clip_id,
                "task_id": clip_id,
                "source_clip_id": clip_id,
                "speaker_key": speaker,
                "source_group_key": group,
                "fixed_split_assignment": split,
                "start_frame": 0,
                "end_frame_exclusive": 30 + index,
                "textgrid_relpath": relpath,
                "window_transcript_role": (
                    "time_aligned_speech_context_only_not_action_or_emotion_label"
                ),
                "window_transcript_context": "".join(words),
                "language_code": "en",
            }
        )
    motion_manifest = tmp_path / "motion.jsonl"
    _write_jsonl(motion_manifest, motion_rows)
    motion_hash = _sha256_file(motion_manifest)

    base_rows = []
    for index, (clip_id, speaker, group, split, directive, _words) in enumerate(
        specifications
    ):
        source = motion_rows[index]
        frames = 30 + index
        base_rows.append(
            {
                "formal_episode_contract": BASE_V9_CONTRACT,
                "accepted_for_training": True,
                "clip_id": clip_id,
                "task_id": clip_id,
                "source_clip_id": clip_id,
                "speaker_key": speaker,
                "source_group_key": group,
                "fixed_split_assignment": split,
                "frames": frames,
                "fps": 30.0,
                "prompt": directive,
                "prompt_sha256": hashlib.sha256(directive.encode()).hexdigest(),
                "trajectory_sha256": f"{index + 1:064x}",
                "source_manifest": str(motion_manifest),
                "source_manifest_sha256": motion_hash,
                "source_record_sha256": canonical_sha256(source),
                "source_transcript_semantics_used": False,
                "motion_style": {
                    "arm_amplitude": "moderate",
                    "laterality": "both",
                    "pace": "steady" if index != 1 else "quick",
                    "head_engagement": "engaged",
                },
                "training_segment": {
                    "representation": "native_variable_length_conversational_gesturing_30hz_v1",
                    "start_frame": 0,
                    "end_frame_exclusive": 30 + index,
                    "frame_count": 30 + index,
                    "output_frame_count": frames,
                    "fixed_window_sec": None,
                    "cropped": False,
                },
            }
        )
    base_manifest = tmp_path / "base_v9.jsonl"
    _write_jsonl(base_manifest, base_rows)
    base_hash = _sha256_file(base_manifest)
    base_lock = tmp_path / "base_v9_lock.json"
    base_lock.write_text(
        json.dumps(
            {
                "artifact_kind": BASE_V9_LOCK_KIND,
                "formal_episode_contract": BASE_V9_CONTRACT,
                "accepted_for_training": True,
                "scale_gate_passed": True,
                "audio_conditioning_enabled": False,
                "primary_intent_conditioning_enabled": False,
                "emotion_conditioning_enabled": False,
                "duration_policy": (
                    "native_variable_length_conversational_event_no_fixed_duration_v1"
                ),
                "train_ready_manifest": str(base_manifest),
                "train_ready_manifest_sha256": base_hash,
            }
        ),
        encoding="utf-8",
    )
    return base_manifest, base_lock, motion_manifest, beat2_root


def test_release_rebuilds_readable_dialogue_and_accounts_for_empty_alignment(
    tmp_path: Path,
) -> None:
    base, lock, motion, root = _fixtures(tmp_path)
    output = tmp_path / "release"
    summary = build_release(
        base_v9_manifest=base,
        base_v9_lock=lock,
        motion_manifest=motion,
        beat2_root=root,
        output_dir=output,
    )

    assert summary["dataset_scale"]["dialogue_conditioned_episode_count"] == 3
    assert summary["dataset_scale"]["dialogue_unavailable_retained_in_v9_count"] == 1
    assert summary["all_motion_accounted_for"] is True
    assert summary["fixed_six_second_training_unit"] is False
    assert summary["network_modified_or_trained"] is False

    episodes = [json.loads(line) for line in (output / "train_ready.jsonl").read_text().splitlines()]
    by_id = {row["clip_id"]: row for row in episodes}
    assert by_id["clip_a"]["dialogue_text"] == "hello there"
    assert by_id["clip_b"]["dialogue_text"] == "how are you"
    assert all(row["training_segment"]["fixed_window_sec"] is None for row in episodes)
    assert all(row["partner_response_supervision_mask"] is False for row in episodes)
    assert all(validate_dialogue_directive_v10_episode(row) for row in episodes)

    pairs = [
        json.loads(line)
        for line in (output / "counterfactual_pairs.jsonl").read_text().splitlines()
    ]
    assert len(pairs) == 3
    assert all(row["dialogue_shuffled"]["same_split"] is True for row in pairs)
    assert all(
        row["dialogue_shuffled"]["dialogue_text_sha256"]
        != row["matched"]["dialogue_text_sha256"]
        for row in pairs
    )
    excluded = json.loads(
        (output / "dialogue_unavailable_retained_in_v9.jsonl").read_text().strip()
    )
    assert excluded["clip_id"] == "clip_empty"
    assert excluded["retained_for_base_v9_directive_and_motion_training"] is True

    audit = verify_release(output)
    assert audit["passed"] is True
    assert audit["verified_invariants"]["split_isolation"] is True
    assert audit["verified_scale"] == summary["dataset_scale"]


def test_contract_rejects_fixed_window_and_transcript_label_promotion(
    tmp_path: Path,
) -> None:
    base, lock, motion, root = _fixtures(tmp_path)
    output = tmp_path / "release"
    build_release(
        base_v9_manifest=base,
        base_v9_lock=lock,
        motion_manifest=motion,
        beat2_root=root,
        output_dir=output,
    )
    episode = json.loads((output / "train_ready.jsonl").read_text().splitlines()[0])

    episode["training_segment"]["fixed_window_sec"] = 6.0
    with pytest.raises(ValueError, match="fixed-window"):
        validate_dialogue_directive_v10_episode(episode)

    episode["training_segment"]["fixed_window_sec"] = None
    episode["source_transcript_used_as_action_or_emotion_label"] = True
    with pytest.raises(ValueError, match="action/emotion label"):
        validate_dialogue_directive_v10_episode(episode)


def test_contract_uses_explicit_native_variable_length_representation() -> None:
    assert TRAINING_SEGMENT_REPRESENTATION.startswith("native_variable_length")
    assert "6" not in TRAINING_SEGMENT_REPRESENTATION


def test_release_audit_rejects_counterfactual_binding_tampering(
    tmp_path: Path,
) -> None:
    base, lock, motion, root = _fixtures(tmp_path)
    output = tmp_path / "release"
    build_release(
        base_v9_manifest=base,
        base_v9_lock=lock,
        motion_manifest=motion,
        beat2_root=root,
        output_dir=output,
    )
    pair_path = output / "counterfactual_pairs.jsonl"
    pairs = [json.loads(line) for line in pair_path.read_text().splitlines()]
    pairs[0]["dialogue_shuffled"]["source_clip_id"] = pairs[0]["anchor_clip_id"]
    pairs[0].pop("record_sha256")
    pairs[0]["record_sha256"] = canonical_sha256(pairs[0])
    _write_jsonl(pair_path, pairs)
    lock_path = output / "provenance_lock.json"
    release_lock = json.loads(lock_path.read_text())
    release_lock["counterfactual_pair_manifest_sha256"] = _sha256_file(pair_path)
    lock_path.write_text(json.dumps(release_lock), encoding="utf-8")

    with pytest.raises(ValueError, match="counterfactual binding changed"):
        verify_release(output)
