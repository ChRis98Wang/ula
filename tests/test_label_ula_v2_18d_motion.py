import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.human_motion_collection.label_ula_v2_18d_motion import (
    ALGORITHM_VERSION,
    JOINT_ORDER,
    PROMPT_PROVENANCE,
    PROMPT_TEMPLATE_VERSION,
    SPEECH_CONTEXT_ROLE,
    build_labels,
    extract_features,
    prompt_from_features,
)


def write_csv(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(JOINT_ORDER)
        writer.writerows(values)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def synthetic_motion(
    *,
    frames: int = 180,
    left: float = 0.0,
    right: float = 0.0,
    head: float = 0.0,
    torso: float = 0.0,
) -> np.ndarray:
    values = np.zeros((frames, 18), dtype=np.float64)
    phase = np.linspace(0.0, 6.0 * np.pi, frames)
    values[:, 3:9] = left * np.sin(phase)[:, None]
    values[:, 9:15] = right * np.sin(phase)[:, None]
    values[:, 15:18] = head * np.sin(phase * 0.5)[:, None]
    values[:, 0:3] = torso * np.sin(phase * 0.25)[:, None]
    return values


def axis_sine_motion(
    joint_index: int,
    *,
    amplitude: float = 0.3,
    frequency_hz: float = 0.7,
    frames: int = 180,
    fps: float = 30.0,
) -> np.ndarray:
    values = np.zeros((frames, 18), dtype=np.float64)
    time = np.arange(frames, dtype=np.float64) / fps
    values[:, joint_index] = amplitude * np.sin(
        2.0 * np.pi * frequency_hz * time
    )
    return values


def write_manifest(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records),
        encoding="utf-8",
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def passed_record(
    root: Path, task_id: str, trajectory: Path, **extra
) -> dict:
    gate = {
        "passed": True,
        "joint_limits_pass": True,
        "velocity_pass": True,
    }
    quality = root / f"{task_id.replace(':', '_')}.quality.json"
    quality.write_text(
        json.dumps(
            {
                "output_contract": "ula_v2_18d_head_v1",
                "action_dim": 18,
                "joint_order": JOINT_ORDER,
                "quality_gate": gate,
            }
        ),
        encoding="utf-8",
    )
    return {
        "task_id": task_id,
        "source_clip_id": extra.pop("source_clip_id", task_id),
        "status": "passed",
        "fps": 30,
        "safe_csv": str(trajectory),
        "safe_csv_sha256": digest(trajectory),
        "quality_json": str(quality),
        "quality_json_sha256": digest(quality),
        "quality_gate": gate,
        **extra,
    }


def test_features_and_prompt_describe_only_observable_left_arm_motion():
    features = extract_features(synthetic_motion(left=0.25, head=0.08), 30.0)
    prompt = prompt_from_features(features)

    assert features["arm"]["laterality"] == "left"
    assert features["head_motion"] in {"subtle", "clear"}
    assert features["duration_sec"] == pytest.approx((180 - 1) / 30.0)
    assert features["frame_coverage_sec"] == pytest.approx(180 / 30.0)
    assert features["duration_time_axis"].startswith("sample_span=")
    assert "left arm" in prompt["en"]
    assert "左臂" in prompt["zh"]
    for unsupported in ("conversation", "explain", "angry", "happy", "greet", "point"):
        assert unsupported not in prompt["en"].lower()


def test_bilateral_temporal_coordination_is_detected_without_claiming_intent():
    features = extract_features(synthetic_motion(left=0.2, right=0.2), 30.0)
    prompt = prompt_from_features(features)

    assert features["arm"]["laterality"] == "both"
    assert features["arm"]["energy_dominance"] == "balanced"
    assert features["arm"]["bilateral_temporally_coordinated"] is True
    assert prompt["en"].startswith("Move both arms together")
    assert prompt["zh"].startswith("双臂同步")
    assert "motion energy similar" in prompt["en"]


def test_bilateral_motion_reports_only_measured_arm_energy_dominance():
    features = extract_features(synthetic_motion(left=0.2, right=0.13), 30.0)
    prompt = prompt_from_features(features)

    assert features["arm"]["laterality"] == "both"
    assert features["arm"]["energy_dominance"] == "left"
    assert "more motion energy in the left arm" in prompt["en"]
    assert "左臂承担更多运动能量" in prompt["zh"]


def test_overall_pace_uses_amplitude_normalized_joint_change_rate():
    slow = extract_features(axis_sine_motion(3, frequency_hz=0.2), 30.0)
    quick = extract_features(axis_sine_motion(3, frequency_hz=0.7), 30.0)

    assert slow["overall_motion"]["pace"] == "slow"
    assert quick["overall_motion"]["pace"] == "quick"
    assert (
        slow["overall_motion"]["normalized_change_rate_hz"]
        < quick["overall_motion"]["normalized_change_rate_hz"]
    )
    assert "unhurried overall movement pace" in prompt_from_features(slow)["en"]
    assert "较快的整体动作节奏" in prompt_from_features(quick)["zh"]


@pytest.mark.parametrize(
    ("joint_index", "axis", "pattern", "english", "chinese"),
    [
        (15, "roll", "repeated_roll_tilts", "roll-axis side tilts", "滚转轴侧倾"),
        (16, "pitch", "repeated_pitch_nods", "pitch-axis nodding", "俯仰轴点头"),
        (17, "yaw", "repeated_yaw_turns", "yaw-axis head turns", "偏航轴转头"),
    ],
)
def test_head_axis_and_repeated_pattern_require_verifiable_joint_sweeps(
    joint_index, axis, pattern, english, chinese
):
    features = extract_features(axis_sine_motion(joint_index), 30.0)
    prompt = prompt_from_features(features)

    assert features["head"]["axis_motion"]["dominant_axis"] == axis
    repeated = features["head"]["repeated_pattern"]
    assert repeated["pattern"] == pattern
    assert repeated["sweep_count"] >= 4
    assert english in prompt["en"]
    assert chinese in prompt["zh"]


def test_large_one_way_head_change_is_not_called_repeated():
    values = np.zeros((180, 18), dtype=np.float64)
    values[:, 17] = np.linspace(-0.3, 0.3, len(values))
    features = extract_features(values, 30.0)

    assert features["head"]["axis_motion"]["dominant_axis"] == "yaw"
    assert features["head"]["repeated_pattern"]["pattern"] == "none"
    assert "repeated" not in prompt_from_features(features)["en"]


def test_mixed_axis_chinese_prompt_does_not_leak_english_subject_name():
    features = extract_features(synthetic_motion(head=0.2), 30.0)
    prompt = prompt_from_features(features)

    assert features["head"]["axis_motion"]["dominant_axis"] == "mixed"
    assert "跨多个头部关节轴" in prompt["zh"]
    assert "head" not in prompt["zh"]


def test_torso_axis_and_variation_intensity_are_joint_space_observations():
    features = extract_features(axis_sine_motion(0, frequency_hz=0.8), 30.0)
    prompt = prompt_from_features(features)

    assert features["torso"]["axis_motion"]["dominant_axis"] == "yaw"
    assert features["torso"]["variation_intensity"] == "high"
    assert "torso motion mainly around the yaw axis" in prompt["en"]
    assert "变化强度较高" in prompt["zh"]


def test_speech_context_cannot_change_motion_prompt(tmp_path):
    trajectory = tmp_path / "same.csv"
    write_csv(trajectory, synthetic_motion(right=0.18, torso=0.08))
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        manifest,
        [
            passed_record(
                tmp_path, "a", trajectory, speech_context="生气地拒绝"
            ),
            passed_record(
                tmp_path, "b", trajectory, speech_context="开心地欢迎"
            ),
        ],
    )

    summary = build_labels(manifest, tmp_path / "out")
    drafts = read_jsonl(tmp_path / "out/draft_prompts.jsonl")

    assert summary["draft_records"] == 2
    assert drafts[0]["canonical_prompt"] == drafts[1]["canonical_prompt"]
    assert drafts[0]["speech_context"] != drafts[1]["speech_context"]
    assert all(x["speech_context_role"] == SPEECH_CONTEXT_ROLE for x in drafts)
    assert all(x["prompt_provenance"] == PROMPT_PROVENANCE for x in drafts)
    assert all(x["accepted_for_training"] is False for x in drafts)


def test_build_outputs_bilingual_text_review_queue_rejections_and_resume(tmp_path):
    moving = tmp_path / "moving.csv"
    static = tmp_path / "static.csv"
    write_csv(moving, synthetic_motion(left=0.15))
    write_csv(static, synthetic_motion())
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        manifest,
        [
            passed_record(tmp_path, "moving", moving),
            passed_record(tmp_path, "static", static),
            {
                **passed_record(tmp_path, "failed_gate", moving),
                "quality_gate": {
                    "passed": True,
                    "joint_limits_pass": False,
                },
            },
        ],
    )
    output = tmp_path / "out"

    first = build_labels(manifest, output)
    first_payload = (output / "draft_prompts.jsonl").read_bytes()
    second = build_labels(manifest, output, resume=True)

    assert first["draft_records"] == 2
    assert first["rejected_records"] == 1
    assert first["accepted_for_training_records"] == 0
    assert first["label_diversity"]["unique_english_prompts"] == 2
    assert sum(first["observable_feature_counts"]["overall_pace"].values()) == 2
    assert (
        sum(
            first["observable_feature_counts"]["head_repeated_pattern"].values()
        )
        == 2
    )
    assert second["resume_reused_records"] == 2
    assert (output / "draft_prompts.jsonl").read_bytes() == first_payload
    assert (output / "needs_human_review.jsonl").read_bytes() == first_payload
    assert (output / "text/en/moving.txt").read_text().strip()
    assert (output / "text/zh/moving.txt").read_text().strip()
    drafts = {x["task_id"]: x for x in read_jsonl(output / "draft_prompts.jsonl")}
    assert drafts["static"]["semantic_confidence"] == "low"
    assert drafts["static"]["review_flags"] == ["near_static_observable_state"]
    assert drafts["static"]["canonical_prompt"]["en"] == (
        "Keep both arms nearly still."
    )
    reasons = {
        x["task_id"]: x["rejection_reason"]
        for x in read_jsonl(output / "rejected.jsonl")
    }
    assert reasons["failed_gate"] == "every strict quality gate must be boolean true"


def test_thresholds_and_algorithm_are_embedded_in_every_draft(tmp_path):
    trajectory = tmp_path / "motion.csv"
    write_csv(trajectory, synthetic_motion(left=0.12))
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        manifest,
        [passed_record(tmp_path, "clip", trajectory, source_clip_id="clip")],
    )

    build_labels(manifest, tmp_path / "out")
    draft = read_jsonl(tmp_path / "out/draft_prompts.jsonl")[0]
    summary = json.loads((tmp_path / "out/summary.json").read_text())

    assert draft["algorithm_version"] == ALGORITHM_VERSION
    assert draft["prompt_template_version"] == PROMPT_TEMPLATE_VERSION
    assert draft["task_id"] == "clip"
    assert draft["source_clip_id"] == "clip"
    assert draft["labeling_thresholds"]["minimum_frames"] == 12
    assert summary["algorithm_version"] == ALGORITHM_VERSION
    assert summary["prompt_template_version"] == PROMPT_TEMPLATE_VERSION
    assert summary["thresholds"]["minimum_frames"] == 12
    assert draft["canonical_action"] == "robot_observable_upper_body_motion"


def test_semantic_event_retarget_segment_preserves_dual_time_axes(tmp_path):
    trajectory = tmp_path / "motion.csv"
    write_csv(trajectory, synthetic_motion(left=0.12))
    record = passed_record(tmp_path, "dataset__source_sem0004_f000010-000190", trajectory)
    record["retarget_segment"] = {
        "fps": 30,
        "source_start_frame": 10,
        "source_end_frame_exclusive": 190,
        "source_frame_coverage_sec": 6.0,
        "output_frame_count": 180,
        "output_sample_span_sec": 179 / 30,
    }
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [record])

    build_labels(manifest, tmp_path / "out")
    draft = read_jsonl(tmp_path / "out/draft_prompts.jsonl")[0]

    assert draft["fps"] == 30
    assert draft["source_window_start_frame"] == 10
    assert draft["source_window_end_frame_exclusive"] == 190
    assert draft["retarget_segment"]["output_sample_span_sec"] == pytest.approx(179 / 30)


def test_existing_six_sample_regression_does_not_overclaim_semantics(tmp_path):
    project = Path(__file__).resolve().parents[1]
    manifest = (
        project
        / "deliverables/interactive_human_motion_v1"
        / "samples/beat2_conversation/sample_manifest.jsonl"
    )
    if not manifest.is_file():
        return

    strict_records = []
    sample_root = manifest.parent
    for sample in read_jsonl(manifest):
        safe_csv = (sample_root / sample["retarget"]["safe_csv"]).resolve()
        quality_json = (sample_root / sample["retarget"]["quality_json"]).resolve()
        quality = json.loads(quality_json.read_text(encoding="utf-8"))
        strict_records.append(
            {
                "task_id": sample["sample_id"],
                "source_clip_id": Path(sample["source_motion"]["relpath"]).stem,
                "status": "passed",
                "fps": sample["source_motion"]["fps"],
                "safe_csv": str(safe_csv),
                "safe_csv_sha256": digest(safe_csv),
                "quality_json": str(quality_json),
                "quality_json_sha256": digest(quality_json),
                "quality_gate": quality["quality_gate"],
                "source_speech_context": sample["window_transcript_context"],
                "official_split": sample["official_split"],
                "speaker_key": sample["speaker_key"],
                "source_warnings": sample["inventory_issues"],
            }
        )
    strict_manifest = tmp_path / "strict_six.jsonl"
    write_manifest(strict_manifest, strict_records)
    summary = build_labels(strict_manifest, tmp_path / "out")
    drafts = read_jsonl(tmp_path / "out/draft_prompts.jsonl")

    assert summary["draft_records"] == 6
    assert summary["rejected_records"] == 0
    banned = {"conversation", "explain", "greet", "salute", "angry", "happy", "sad"}
    for draft in drafts:
        assert draft["observable_features"]["frames"] >= 180
        assert not banned.intersection(draft["canonical_prompt"]["en"].lower().split())
        assert draft["accepted_for_training"] is False


def test_task_id_is_primary_and_source_clip_id_is_preserved(tmp_path):
    trajectory = tmp_path / "motion.csv"
    write_csv(trajectory, synthetic_motion(right=0.12))
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        manifest,
        [
            {
                **passed_record(
                    tmp_path,
                    "beat2:clip_f000010-000190",
                    trajectory,
                    source_clip_id="beat2_clip",
                ),
                "clip_id": "legacy_clip",
                "official_split": "test",
                "speaker_key": "speaker_1",
                "source_warnings": ["source_warning"],
            }
        ],
    )

    build_labels(manifest, tmp_path / "out")
    draft = read_jsonl(tmp_path / "out/draft_prompts.jsonl")[0]

    assert draft["task_id"] == "beat2:clip_f000010-000190"
    assert draft["source_clip_id"] == "beat2_clip"
    assert draft["official_split"] == "test"
    assert draft["speaker_key"] == "speaker_1"
    assert draft["source_warnings"] == ["source_warning"]
    assert draft["manual_review_required"] is True
    assert draft["manual_human_review_required"] is True
    assert (tmp_path / "out/text/en/beat2:clip_f000010-000190.txt").is_file()


def test_batch_admission_is_fail_closed_when_status_gate_or_hash_is_missing(tmp_path):
    trajectory = tmp_path / "motion.csv"
    write_csv(trajectory, synthetic_motion(left=0.12))
    missing_status = passed_record(tmp_path, "missing_status", trajectory)
    missing_status.pop("status")
    missing_gate = passed_record(tmp_path, "missing_gate", trajectory)
    missing_gate.pop("quality_gate")
    missing_hash = passed_record(tmp_path, "missing_hash", trajectory)
    missing_hash.pop("safe_csv_sha256")
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [missing_status, missing_gate, missing_hash])

    summary = build_labels(manifest, tmp_path / "out")
    rejected = read_jsonl(tmp_path / "out/rejected.jsonl")

    assert summary["draft_records"] == 0
    assert summary["rejected_records"] == 3
    reasons = {item["task_id"]: item["rejection_reason"] for item in rejected}
    assert reasons["missing_status"] == "status must be exactly 'passed'"
    assert reasons["missing_gate"] == "quality_gate.passed must be true"
    assert reasons["missing_hash"] == "safe_csv_sha256 is missing or does not match"
