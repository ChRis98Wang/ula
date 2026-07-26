import csv
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
import pytest
import torch

from tools.pretrain_evaluation import build_18d_pretrain_long_video as long_video


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def write_motion(path: Path, frames: int, *, bad_time: bool = False) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_sec", *long_video.JOINT_ORDER_18D])
        for frame in range(frames):
            time_sec = frame / 30.0
            if bad_time and frame == frames - 1:
                time_sec += 0.01
            writer.writerow([time_sec, *([frame / 1000.0] * 18)])


def formal_summary(checkpoint: Path) -> dict:
    return {
        "checkpoint": str(checkpoint.resolve()),
        "completed_steps": 10_000,
        "target_steps": 10_000,
        "stopped_early": False,
        "best_step": 9_750,
        "artifact_status": "adjudicated_posttrain_candidate",
        "formal_release_eligible": True,
        "training_scope": long_video.FORMAL_SCOPE,
        "formal_training_enabled": True,
        "temporal_unit_policy": long_video.TEMPORAL_POLICY,
        "training_policy": "full_network",
    }


def metadata_caption(name: str) -> dict:
    return {
        "kind": "motion_only_metadata",
        "semantic_role": "metadata_not_model_condition_not_verified_semantics",
        "text_zh": f"保留集样本 {name}",
        "text_en": f"Held-out sample {name}",
    }


def generation_record(
    *,
    segment_id: str,
    clip_id: str,
    split: str,
    motion: Path,
    frames: int,
    checkpoint_hash: str,
    split_hash: str,
    caption: dict | None = None,
) -> dict:
    return {
        "schema_version": long_video.SCHEMA_VERSION,
        "artifact_kind": long_video.SEGMENT_ARTIFACT_KIND,
        "segment_id": segment_id,
        "held_out_clip_id": clip_id,
        "split": split,
        "generated_csv": str(motion),
        "generated_csv_sha256": digest(motion),
        "generated_from_checkpoint_sha256": checkpoint_hash,
        "split_manifest_sha256": split_hash,
        "output_kind": "model_generated_motion",
        "robot_contract": long_video.ROBOT_CONTRACT,
        "action_dim": 18,
        "fps": 30.0,
        "conditioning": {
            "mode": "motion_only",
            "text_conditioning_used": False,
            "emotion_conditioning_used": False,
            "audio_conditioning_used": False,
        },
        "native_duration": {
            "policy": long_video.NATIVE_DURATION_POLICY,
            "held_out_episode_frame_count": frames,
            "requested_generation_frame_count": frames,
            "cropped": False,
            "padded": False,
        },
        "caption": caption or metadata_caption(segment_id),
    }


def input_tree(tmp_path: Path, monkeypatch, *, frames=(61, 94)) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"real-checkpoint-fixture")
    checkpoint_hash = digest(checkpoint)
    training_summary = tmp_path / "training_summary.json"
    write_json(training_summary, formal_summary(checkpoint))
    split_manifest = tmp_path / "split_manifest.json"
    split_payload = {
        "episodes": [
            {
                "clip_id": "heldout-a",
                "split": "validation",
                "speaker_key": "speaker-a",
                "source_group_key": "group-a",
            },
            {
                "clip_id": "heldout-b",
                "split": "test",
                "speaker_key": "speaker-b",
                "source_group_key": "group-b",
            },
        ]
    }
    split_payload["sha256"] = long_video.value_sha256(split_payload)
    write_json(split_manifest, split_payload)
    split_hash = digest(split_manifest)
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    write_motion(first, frames[0])
    write_motion(second, frames[1])
    generation_manifest = tmp_path / "generation.jsonl"
    write_jsonl(
        generation_manifest,
        [
            generation_record(
                segment_id="sample-a",
                clip_id="heldout-a",
                split="validation",
                motion=first,
                frames=frames[0],
                checkpoint_hash=checkpoint_hash,
                split_hash=split_hash,
            ),
            generation_record(
                segment_id="sample-b",
                clip_id="heldout-b",
                split="test",
                motion=second,
                frames=frames[1],
                checkpoint_hash=checkpoint_hash,
                split_hash=split_hash,
            ),
        ],
    )
    monkeypatch.setattr(
        long_video,
        "validate_checkpoint_file",
        lambda path: {
            "global_step": 9_750,
            "artifact_status": "adjudicated_posttrain_candidate",
            "formal_release_eligible": True,
            "action_dim": 18,
            "robot_contract": long_video.ROBOT_CONTRACT,
            "data_contract_sha256": "d" * 64,
            "split_contract_sha256": split_payload["sha256"],
            "_held_out_native_frames": {
                "heldout-a": frames[0],
                "heldout-b": frames[1],
            },
            "_split_contract": split_payload,
        },
    )
    return {
        "checkpoint": checkpoint,
        "training_summary": training_summary,
        "split_manifest": split_manifest,
        "generation_manifest": generation_manifest,
    }


def test_generated_csv_requires_exact_18d_30hz_and_native_frame_count(tmp_path):
    valid = tmp_path / "valid.csv"
    write_motion(valid, 7)
    result = long_video.validate_generated_csv(valid, expected_frames=7)
    assert result["frames"] == 7
    assert result["action_dim"] == 18
    assert result["sample_span_sec"] == pytest.approx(0.2)

    bad_time = tmp_path / "bad_time.csv"
    write_motion(bad_time, 7, bad_time=True)
    with pytest.raises(long_video.LongVideoContractError, match="30 Hz grid"):
        long_video.validate_generated_csv(bad_time, expected_frames=7)
    with pytest.raises(long_video.LongVideoContractError, match="expected native 8"):
        long_video.validate_generated_csv(valid, expected_frames=8)


def test_validate_inputs_binds_checkpoint_held_out_splits_and_native_lengths(
    tmp_path, monkeypatch
):
    paths = input_tree(tmp_path, monkeypatch)
    plan = long_video.validate_inputs(**paths, min_duration_sec=1.0)

    assert plan["status"] == "validated_inputs_not_rendered"
    assert plan["total_frames"] == 155
    assert plan["unique_frame_counts"] == [61, 94]
    assert {row["split"] for row in plan["segments"]} == {"validation", "test"}
    assert plan["evaluation_contract"]["text_conditioning_used"] is False
    assert plan["evaluation_contract"]["fixed_six_second_units"] is False
    assert all(
        row["caption"]["semantic_role"]
        == "metadata_not_model_condition_not_verified_semantics"
        for row in plan["segments"]
    )


def test_generation_contract_rejects_text_condition_train_split_and_fixed_window(
    tmp_path, monkeypatch
):
    paths = input_tree(tmp_path, monkeypatch)
    records = long_video.load_jsonl(paths["generation_manifest"])

    records[0]["conditioning"]["text_conditioning_used"] = True
    write_jsonl(paths["generation_manifest"], records)
    with pytest.raises(long_video.LongVideoContractError, match="motion-only conditioning"):
        long_video.validate_inputs(**paths, min_duration_sec=1.0)

    records[0]["conditioning"]["text_conditioning_used"] = False
    records[0]["native_duration"]["fixed_window_sec"] = 6.0
    write_jsonl(paths["generation_manifest"], records)
    with pytest.raises(long_video.LongVideoContractError, match="forbidden"):
        long_video.validate_inputs(**paths, min_duration_sec=1.0)

    del records[0]["native_duration"]["fixed_window_sec"]
    records[0]["split"] = "train"
    write_jsonl(paths["generation_manifest"], records)
    with pytest.raises(long_video.LongVideoContractError, match="validation or test"):
        long_video.validate_inputs(**paths, min_duration_sec=1.0)


def test_native_frame_count_must_match_checkpoint_data_contract(tmp_path, monkeypatch):
    paths = input_tree(tmp_path, monkeypatch)
    records = long_video.load_jsonl(paths["generation_manifest"])
    records[0]["native_duration"]["held_out_episode_frame_count"] = 62
    records[0]["native_duration"]["requested_generation_frame_count"] = 62
    write_jsonl(paths["generation_manifest"], records)

    with pytest.raises(long_video.LongVideoContractError, match="checkpoint data contract"):
        long_video.validate_inputs(**paths, min_duration_sec=1.0)


def test_metadata_caption_requires_explicit_non_condition_role(tmp_path):
    with pytest.raises(long_video.LongVideoContractError, match="explicit"):
        long_video.validate_caption(
            {"kind": "motion_only_metadata", "text_zh": "挥手"},
            generated_csv_sha256="a" * 64,
            checkpoint_sha256="b" * 64,
            frame_count=31,
            base_dir=tmp_path,
        )


def test_reviewed_caption_is_bound_to_generated_csv_checkpoint_and_full_blind_decode(
    tmp_path,
):
    record = {
        "artifact_kind": long_video.REVIEW_ARTIFACT_KIND,
        "protocol_version": long_video.REVIEW_PROTOCOL,
        "review_id": "review-1",
        "review_status": "accepted_robot_observable_text",
        "full_decode_to_eof": True,
        "label_metadata_exposed": False,
        "emotion_inference_performed": False,
        "text_conditioning_claimed": False,
        "generated_csv_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "decoded_frame_count": 73,
        "fps": 30.0,
        "native_duration_preserved": True,
        "fixed_duration_window_used": False,
        "robot_observable_text": {
            "zh": "机器人先抬起左前臂，再将双臂向外展开。",
            "en": "The robot raises its left forearm, then opens both arms outward.",
        },
    }
    review = tmp_path / "review.jsonl"
    write_jsonl(review, [record])
    caption = {
        "kind": "reviewed_robot_observable_text",
        "review_artifact": str(review),
        "review_artifact_sha256": digest(review),
        "review_record_id": "review-1",
        "review_record_sha256": long_video.value_sha256(record),
    }

    result = long_video.validate_caption(
        caption,
        generated_csv_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        frame_count=73,
        base_dir=tmp_path,
    )
    assert result["kind"] == "reviewed_robot_observable_text"
    assert "未作为预训练文本条件" not in result["text_zh"]

    record["label_metadata_exposed"] = True
    write_jsonl(review, [record])
    caption["review_artifact_sha256"] = digest(review)
    caption["review_record_sha256"] = long_video.value_sha256(record)
    with pytest.raises(long_video.LongVideoContractError, match="contract mismatch"):
        long_video.validate_caption(
            caption,
            generated_csv_sha256="a" * 64,
            checkpoint_sha256="b" * 64,
            frame_count=73,
            base_dir=tmp_path,
        )


def test_text_artifacts_force_disclosure_and_keep_frame_exact_chapters(tmp_path):
    segments = [
        {
            "segment_id": "metadata",
            "trajectory": {"frames": 61},
            "caption": {
                "kind": "motion_only_metadata",
                "semantic_role": "metadata_not_model_condition_not_verified_semantics",
                "text_zh": "保留集样本 A",
                "text_en": "Held-out sample A",
            },
        },
        {
            "segment_id": "reviewed",
            "trajectory": {"frames": 94},
            "caption": {
                "kind": "reviewed_robot_observable_text",
                "semantic_role": "reviewed_observation_not_pretrain_text_condition",
                "text_zh": "机器人向外展开双臂。",
                "text_en": "The robot opens both arms outward.",
            },
        },
    ]
    result = long_video.build_text_artifacts(segments, output_dir=tmp_path)
    zh = result["paths"]["zh_srt"].read_text(encoding="utf-8")
    en = result["paths"]["en_srt"].read_text(encoding="utf-8")
    metadata = result["paths"]["chapters_ffmetadata"].read_text(encoding="utf-8")

    assert "不是文本条件；语义未经验证" in zh
    assert "未作为预训练文本条件" in zh
    assert "NOT A TEXT CONDITION" in en
    assert "TIMEBASE=1/30" in metadata
    assert "START=61" in metadata
    assert "END=155" in metadata
    assert result["total_frames"] == 155
    assert result["timeline"][1]["start_frame"] == 61


def test_renderer_and_ffmpeg_commands_fix_1080p_30hz_and_embed_two_subtitles(tmp_path):
    renderer = long_video.build_renderer_command(
        renderer_python=Path("/env/python"),
        csv_path=tmp_path / "motion.csv",
        output_mp4=tmp_path / "motion.mp4",
        summary_json=tmp_path / "motion.json",
        urdf=tmp_path / "robot.urdf",
    )
    assert renderer[renderer.index("--fps") + 1] == "30.0"
    assert renderer[renderer.index("--width") + 1] == "1920"
    assert renderer[renderer.index("--height") + 1] == "1080"
    assert "--simplified" not in renderer

    ffmpeg = long_video.build_ffmpeg_command(
        ffmpeg=Path("/bin/ffmpeg"),
        concat_file=tmp_path / "concat.txt",
        zh_srt=tmp_path / "zh.srt",
        en_srt=tmp_path / "en.srt",
        chapters=tmp_path / "chapters.ffmetadata",
        output_mp4=tmp_path / "long.mp4",
    )
    assert ffmpeg.count("-map") == 3
    assert "copy" in ffmpeg
    assert "mov_text" in ffmpeg
    assert "-map_chapters" in ffmpeg
    assert any("Subtitles were not model text conditions" in item for item in ffmpeg)


def test_ffmpeg_command_really_muxes_two_subtitle_streams_and_chapters(tmp_path):
    videos = []
    for video_index, count in enumerate((3, 4)):
        path = tmp_path / f"segment-{video_index}.mp4"
        frames = []
        for frame_index in range(count):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[5:35, 4 + frame_index : 25 + frame_index] = (
                40 + 50 * video_index,
                100,
                220,
            )
            frames.append(frame)
        imageio.mimwrite(
            path,
            frames,
            fps=30.0,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=2,
            output_params=["-movflags", "+faststart"],
        )
        videos.append(path)

    concat = tmp_path / "concat.txt"
    concat.write_text(
        "".join(f"file {long_video._concat_quote(path)}\n" for path in videos),
        encoding="utf-8",
    )
    text = long_video.build_text_artifacts(
        [
            {
                "segment_id": "a",
                "trajectory": {"frames": 3},
                "caption": {
                    "kind": "motion_only_metadata",
                    "semantic_role": "metadata_not_model_condition_not_verified_semantics",
                    "text_zh": "样本 A",
                    "text_en": "Sample A",
                },
            },
            {
                "segment_id": "b",
                "trajectory": {"frames": 4},
                "caption": {
                    "kind": "motion_only_metadata",
                    "semantic_role": "metadata_not_model_condition_not_verified_semantics",
                    "text_zh": "样本 B",
                    "text_en": "Sample B",
                },
            },
        ],
        output_dir=tmp_path,
    )
    output = tmp_path / "long.mp4"
    command = long_video.build_ffmpeg_command(
        ffmpeg=Path(imageio_ffmpeg.get_ffmpeg_exe()),
        concat_file=concat,
        zh_srt=text["paths"]["zh_srt"],
        en_srt=text["paths"]["en_srt"],
        chapters=text["paths"]["chapters_ffmetadata"],
        output_mp4=output,
    )
    long_video._run(command, stage="test mux")

    streams = long_video.validate_final_streams(output, expected_chapters=2)
    assert streams == {
        "video_streams": 1,
        "audio_streams": 0,
        "subtitle_streams": 2,
        "video_codec": "h264",
        "pixel_format": "yuv420p",
        "subtitle_codec": "mov_text",
        "chapters": 2,
    }


def test_long_video_requires_duration_diversity_and_minimum_runtime(tmp_path, monkeypatch):
    paths = input_tree(tmp_path, monkeypatch, frames=(60, 60))
    with pytest.raises(long_video.LongVideoContractError, match="variable native durations"):
        long_video.validate_inputs(**paths, min_duration_sec=1.0)

    paths = input_tree(tmp_path / "short", monkeypatch, frames=(30, 31))
    with pytest.raises(long_video.LongVideoContractError, match="below required"):
        long_video.validate_inputs(**paths, min_duration_sec=60.0)


def test_training_summary_must_be_complete_formal_and_bound_to_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    summary_path = tmp_path / "summary.json"
    summary = formal_summary(checkpoint)
    summary["completed_steps"] = 9999
    write_json(summary_path, summary)
    with pytest.raises(long_video.LongVideoContractError, match="did not reach"):
        long_video.validate_training_summary(summary_path, checkpoint=checkpoint)

    summary["completed_steps"] = 10_000
    summary["formal_release_eligible"] = False
    write_json(summary_path, summary)
    with pytest.raises(long_video.LongVideoContractError, match="formal contract"):
        long_video.validate_training_summary(summary_path, checkpoint=checkpoint)


def test_checkpoint_validator_requires_formal_motion_only_not_expression_training(tmp_path):
    split_contract = {
        "episodes": [
            {"clip_id": "heldout", "split": "test"},
        ]
    }
    split_contract["sha256"] = long_video.value_sha256(split_contract)
    data_contract = {
        "records": [
            {
                "clip_id": "heldout",
                "split": "test",
                "quality_output_frame_count": 73,
                "retarget_segment": {"output_frame_count": 73, "cropped": False},
                "training_segment": {"fixed_window_sec": None},
            }
        ]
    }
    data_contract["sha256"] = long_video.value_sha256(data_contract)
    checkpoint = {
        "artifact_kind": long_video.CHECKPOINT_ARTIFACT_KIND,
        "architecture": long_video.CHECKPOINT_ARCHITECTURE,
        "condition_dim": long_video.CONDITION_DIM,
        "action_dim": 18,
        "joint_order": list(long_video.JOINT_ORDER_18D),
        "action_contract": {
            "version": long_video.ROBOT_CONTRACT,
            "legacy_prefix_dim": 15,
        },
        "config": {"hidden_dim": 4},
        "model_state_dict": {
            "input.weight": torch.zeros(4, 18),
            "output.weight": torch.zeros(18, 4),
            "output.bias": torch.zeros(18),
        },
        "action_stats": {"mean": torch.zeros(18), "std": torch.ones(18)},
        "formal_release_eligible": True,
        "artifact_status": "adjudicated_posttrain_candidate",
        "formal_episode_contract": long_video.MOTION_ONLY_EPISODE_CONTRACT,
        "random_initialization": {"mode": long_video.FULL_RANDOM_INIT_MODE},
        "posttrain_data_contract": data_contract,
        "posttrain_split_contract": split_contract,
        "global_step": 10_000,
        "data_provenance": {
            "training_scope": long_video.FORMAL_SCOPE,
            "formal_training_enabled": True,
            "temporal_unit_policy": long_video.TEMPORAL_POLICY,
            "batching_mode": "native_variable_length",
            "training_policy": "full_network",
            "unsafe_training_data": False,
        },
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)

    result = long_video.validate_checkpoint_file(path)
    assert result["formal_episode_contract"] == long_video.MOTION_ONLY_EPISODE_CONTRACT

    checkpoint["formal_episode_contract"] = "beat2_expression_turn_v8_train_episode_v1"
    torch.save(checkpoint, path)
    with pytest.raises(long_video.LongVideoContractError, match="motion-only"):
        long_video.validate_checkpoint_file(path)
