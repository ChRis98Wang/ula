import csv
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from tools.human_motion_review import render_beat2_annotation_review as review


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def queue_record(
    task_id: str,
    *,
    speaker: str = "12_zhao",
    split: str = "train",
    laterality: str = "both",
) -> dict:
    return {
        "schema_version": "1.0.0",
        "task_id": task_id,
        "source_clip_id": task_id.rsplit("_f", 1)[0],
        "speaker_key": speaker,
        "official_split": split,
        "robot_contract": review.ROBOT_CONTRACT,
        "canonical_action": "robot_observable_upper_body_motion",
        "canonical_prompt": {
            "en": "Move both arms gently.",
            "zh": "轻柔地移动双臂。",
        },
        "observable_features": {
            "arm": {
                "laterality": laterality,
                "amplitude": "small",
                "continuity": "intermittent",
                "bilateral_temporally_coordinated": laterality == "both",
                "regularly_repeated": False,
            },
            "head_motion": "minimal",
            "torso_motion": "subtle",
        },
        "trajectory_path": f"evidence/{task_id}.csv",
        "trajectory_sha256": "0" * 64,
        "review_state": "pending_independent_motion_text_review",
        "manual_review_required": True,
        "accepted_for_training": False,
        "speech_context": "This must never enter the review prompt bundle.",
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def write_trajectory(path: Path, frames: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(review.JOINT_ORDER)
        for index in range(frames):
            writer.writerow([index / 100.0] * len(review.JOINT_ORDER))


def test_stratified_selection_is_deterministic_and_round_robins_strata():
    records = [
        queue_record("a_f000000-000180", speaker="12_zhao", laterality="left"),
        queue_record("b_f000000-000180", speaker="12_zhao", laterality="left"),
        queue_record("c_f000000-000180", speaker="13_lu", split="test"),
        queue_record("d_f000000-000180", speaker="13_lu", split="test"),
        queue_record("e_f000000-000180", speaker="22_luqi", laterality="right"),
        queue_record("f_f000000-000180", speaker="22_luqi", laterality="right"),
    ]

    selected = review.select_records(
        records, limit=3, sampling="stratified", seed=17
    )
    reversed_selected = review.select_records(
        list(reversed(records)), limit=3, sampling="stratified", seed=17
    )

    assert [item["task_id"] for item in selected] == [
        item["task_id"] for item in reversed_selected
    ]
    assert len({review.sampling_stratum(item) for item in selected}) == 3


def test_duration_quantile_selection_proves_native_range_and_median():
    records = []
    for index, span in enumerate((6.1, 0.5, 2.0, 1.0, 4.0)):
        record = queue_record(f"duration_{index}_f000000-000180")
        record["retarget_segment"] = {"output_sample_span_sec": span}
        records.append(record)

    selected = review.select_records(
        records, limit=3, sampling="duration_quantiles", seed=999
    )

    assert [review.duration_span_sec(item) for item in selected] == [0.5, 2.0, 6.1]


def test_duration_quantile_selection_rejects_missing_planner_span():
    with pytest.raises(ValueError, match="missing output_sample_span_sec"):
        review.select_records(
            [queue_record("missing_f000000-000180")],
            limit=1,
            sampling="duration_quantiles",
            seed=0,
        )


def test_semantic_stratum_supports_formal_features_and_legacy_aliases():
    formal = queue_record("formal_f000000-000180", laterality="both")
    legacy = queue_record("legacy_f000000-000180", laterality="both")
    legacy["observable_features"]["arm"] = {
        "laterality": "both",
        "scale": "small",
        "activity": "intermittent",
        "coordination": True,
        "regularly_repeated": False,
    }

    assert review._semantic_key(formal) == review._semantic_key(legacy)
    assert "amplitude=small" in review._semantic_key(formal)
    assert "continuity=intermittent" in review._semantic_key(formal)
    assert "bilateral_coordinated=True" in review._semantic_key(formal)


def test_sampling_stratum_prefers_official_semantics_and_fixed_split():
    record = queue_record("official_f000000-000180", split="train")
    record.update(
        {
            "fixed_split_assignment": "validation",
            "emotion_id": "fear",
            "semantic_event": {"category": "deictic", "intensity": "low"},
        }
    )

    assert review.sampling_stratum(record) == (
        "12_zhao",
        "validation",
        "fear/deictic/low",
    )


def test_sequential_selection_is_sorted_and_projection_excludes_speech():
    records = [
        queue_record("z_f000000-000180"),
        queue_record("a_f000000-000180"),
    ]
    selected = review.select_records(
        records, limit=None, sampling="sequential", seed=0
    )
    projected = review.review_projection(
        selected[0], rank=0, sampling="sequential", seed=0
    )

    assert [item["task_id"] for item in selected] == [
        "a_f000000-000180",
        "z_f000000-000180",
    ]
    assert all("speech" not in key.lower() or key == "speech_context_included" for key in projected)
    assert projected["speech_context_included"] is False
    assert projected["accepted_for_training"] is False
    assert projected["render_pass_grants_training_admission"] is False


def test_projection_preserves_fail_closed_semantic_audit_fields():
    record = queue_record("semantic_f000000-000180")
    record.update(
        {
            "canonical_prompt_role": "coarse_category_only",
            "official_category_verified": True,
            "official_category_role": "verified_metadata_split_and_evaluation_only",
            "official_category_condition_channel": None,
            "official_category_loss": None,
            "robot_observable_motion_form": "candidate_unreviewed",
            "communicative_intent": "candidate_unreviewed",
            "affect_observable_review_status": "candidate_unreviewed",
            "affect_observable_supervision_mask": False,
            "semantic_action_completeness_review_required": True,
            "affect_observable_review_required": True,
            "semantic_supervision_masks": {
                "official_category": False,
                "robot_observable_motion_form": False,
                "communicative_intent": False,
                "prompt_text": False,
                "legacy_gesture": False,
            },
        }
    )

    projected = review.review_projection(
        record, rank=0, sampling="sequential", seed=0
    )

    assert projected["canonical_prompt_role"] == "coarse_category_only"
    assert projected["official_category_verified"] is True
    assert projected["communicative_intent"] == "candidate_unreviewed"
    assert projected["affect_observable_review_status"] == "candidate_unreviewed"
    assert projected["affect_observable_supervision_mask"] is False
    assert projected["semantic_supervision_masks"]["prompt_text"] is False


def test_renderer_command_uses_real_18d_review_contract(tmp_path):
    command = review.build_renderer_command(
        renderer_python=Path("/env/bin/python"),
        trajectory=tmp_path / "motion.csv",
        output_mp4=tmp_path / "motion.mp4",
        summary_json=tmp_path / "motion.json",
        urdf=tmp_path / "robot.urdf",
        width=1280,
        height=720,
    )

    assert command[:3] == [
        "/env/bin/python",
        "-m",
        "upper_body_skeleton.mujoco_playback",
    ]
    assert "--simplified" not in command
    assert command[command.index("--fps") + 1] == "30.0"
    assert command[command.index("--camera-margin") + 1] == "1.12"
    assert command[command.index("--camera-lookat-z-offset") + 1] == "-0.06"


def test_trajectory_validation_checks_order_frames_and_hash(tmp_path):
    queue_path = tmp_path / "queue.jsonl"
    trajectory = tmp_path / "evidence/motion.csv"
    write_trajectory(trajectory, frames=4)
    record = queue_record("motion_f000000-000180")
    record["trajectory_path"] = "evidence/motion.csv"
    record["trajectory_sha256"] = digest(trajectory)

    path, frames = review.validate_trajectory(record, queue_path)

    assert path == trajectory.resolve()
    assert frames == 4
    record["trajectory_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="trajectory_sha256 mismatch"):
        review.validate_trajectory(record, queue_path)


def test_video_validation_fully_decodes_and_enforces_frame_count_and_nonblank(tmp_path):
    video = tmp_path / "valid.mp4"
    frames = []
    for offset in (2, 12, 22):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[8:28, offset : offset + 16] = (40, 160, 240)
        frames.append(frame)
    imageio.mimwrite(
        video,
        frames,
        fps=review.FPS,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=2,
        output_params=["-movflags", "+faststart"],
    )

    result = review.validate_video(
        video,
        expected_frames=3,
        expected_width=64,
        expected_height=48,
    )

    assert result["fully_decodable"] is True
    assert result["decoded_frames"] == 3
    assert result["audio_streams"] == 0
    assert result["faststart"] is True
    assert result["moov_offset"] < result["mdat_offset"]
    with pytest.raises(ValueError, match="decoded 3 frames, expected 2"):
        review.validate_video(
            video,
            expected_frames=2,
            expected_width=64,
            expected_height=48,
        )


def test_run_review_writes_closed_manifests_without_training_admission(
    tmp_path, monkeypatch
):
    queue_path = tmp_path / "review_queue.jsonl"
    records = [
        queue_record("a_f000000-000180"),
        queue_record("b_f000000-000180", speaker="13_lu", split="test"),
    ]
    write_jsonl(queue_path, records)

    def fake_render(sample, **kwargs):
        return {
            "schema_version": review.SCHEMA_VERSION,
            "task_id": sample["task_id"],
            "status": "passed",
            "resume_reused": False,
            "accepted_for_training": False,
            "manual_review_required": True,
            "render_pass_grants_training_admission": False,
        }

    monkeypatch.setattr(review, "render_one", fake_render)
    output = tmp_path / "output"
    summary = review.run_review(
        queue_path=queue_path,
        output_root=output,
        renderer_python=Path("/env/bin/python"),
        urdf=Path("/robot.urdf"),
        limit=1,
        sampling="stratified",
        seed=0,
        workers=1,
        width=1280,
        height=720,
        resume=False,
        retry_failed=False,
    )

    passed = [json.loads(line) for line in (output / "passed_manifest.jsonl").read_text().splitlines()]
    sampled = [json.loads(line) for line in (output / "sampled_manifest.jsonl").read_text().splitlines()]
    assert summary["counts"] == {"passed": 1, "failed": 0, "resume_reused": 0}
    assert summary["accepted_for_training"] == 0
    assert summary["render_pass_grants_training_admission"] is False
    assert summary["passed_manifest_sha256"] == review.sha256(
        output / "passed_manifest.jsonl"
    )
    assert summary["failed_manifest_sha256"] == review.sha256(
        output / "failed_manifest.jsonl"
    )
    assert len(passed) == len(sampled) == 1
    assert passed[0]["accepted_for_training"] is False
    assert sampled[0]["speech_context_included"] is False


def test_resume_reuses_failed_result_only_without_retry(tmp_path):
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "task_id": "failed_f000000-000180",
                "status": "failed",
                "input_fingerprint": "input-v1",
                "render_config_fingerprint": "render-v1",
                "resume_reused": False,
            }
        ),
        encoding="utf-8",
    )

    reused = review._resumable_result(
        result_path,
        input_fingerprint="input-v1",
        config_fingerprint="render-v1",
        retry_failed=False,
        width=1280,
        height=720,
    )
    retried = review._resumable_result(
        result_path,
        input_fingerprint="input-v1",
        config_fingerprint="render-v1",
        retry_failed=True,
        width=1280,
        height=720,
    )

    assert reused["resume_reused"] is True
    assert retried is None
