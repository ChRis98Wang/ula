import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from tools.human_motion_review.adjudicate_training_dataset import (
    EXPECTED_18D_JOINT_ORDER,
    REQUIRED_18D_GATES,
    adjudicate_dataset,
    index_quality_evidence,
    load_independent_reviews,
    load_semantics,
    sha256_file,
)


def _semantic(clip_id, *, recommended_use="auxiliary_after_manual_review", flags=None):
    return {
        "schema_version": "1.0.0",
        "clip_id": clip_id,
        "canonical_action": clip_id.split("_")[0],
        "canonical_prompt": {"en": f"Perform {clip_id}.", "zh": "test"},
        "source": {"motion_relpath": f"raw/{clip_id}.npy"},
        "source_text_quality": {
            "recommended_use": recommended_use,
            "flags": list(flags or []),
        },
    }


def _review_sample(
    clip_id,
    status,
    accepted,
    *,
    failed_gates=(),
    affect_observable=False,
    observed_emotion_id=None,
):
    gates = {
        "action_recognizable": True,
        "text_consistent": True,
        "observable_in_18d": True,
        "context_available": True,
        "physical_qc": True,
        "subject_action_split_safe": True,
        "affect_observable_in_18d": bool(affect_observable),
    }
    for gate in failed_gates:
        gates[gate] = False
    sample = {
        "sample_id": clip_id,
        "status": status,
        "training_acceptance": accepted,
        "gates": gates,
        "conflict_ids": [f"conflict_{clip_id}"] if not accepted else [],
        "split": {
            "subject_key": None,
            "subject_policy": "train_only_unknown",
            "action_key": clip_id.split("_")[0],
            "source_group_key": f"fixture/{clip_id}",
            "assignment": "train" if accepted else None,
            "eval_eligible": False,
        },
    }
    if affect_observable:
        video_sha256 = "9" * 64
        sample["evidence"] = {
            "preview": {"path": f"anonymous/{clip_id}.mp4", "sha256": video_sha256}
        }
        sample["blind_affect_review"] = {
            "protocol_version": "robot_affect_blind_video_v1",
            "review_id": f"blind-{clip_id}",
            "anonymous_video_id": f"anonymous-{hashlib.sha256(clip_id.encode()).hexdigest()[:12]}",
            "video_sha256": video_sha256,
            "target_emotion_exposed": False,
            "audio_available": False,
            "blinded_to": [
                "audio",
                "canonical_prompt",
                "official_emotion_label",
                "official_gesture_category",
                "source_text",
            ],
            "reviewer": {
                "kind": "agent",
                "independent_of_annotation_logic": True,
                "reviewer_id": "blind-agent-fixture",
            },
            "observed_affect": {
                "status": "label",
                "emotion_id": observed_emotion_id or "neutral",
                "confidence": 0.9,
            },
        }
    return sample


def _write_inputs(root: Path):
    semantics = [
        _semantic("accepted_clip"),
        _semantic("accepted_missing_qc_clip"),
        _semantic("human_clip"),
        _semantic(
            "rejected_clip",
            recommended_use="reject",
            flags=["canonical_action_text_conflict", "cross_action_duplicate"],
        ),
        _semantic("unreviewed_clip"),
    ]
    semantics_path = root / "semantics.jsonl"
    semantics_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in semantics),
        encoding="utf-8",
    )
    review = {
        "schema_version": 1,
        "review_id": "fixture_review",
        "reviewer": {"kind": "agent", "independent_of_annotation_logic": True},
        "samples": [
            _review_sample("accepted_clip", "agent_reviewed", True),
            _review_sample("accepted_missing_qc_clip", "agent_reviewed", True),
            _review_sample(
                "human_clip",
                "needs_human",
                False,
                failed_gates=["text_consistent"],
            ),
            _review_sample(
                "rejected_clip",
                "rejected",
                False,
                failed_gates=["action_recognizable", "text_consistent"],
            ),
        ],
    }
    review_path = root / "review.json"
    review_path.write_text(json.dumps(review, sort_keys=True), encoding="utf-8")
    quality_root = root / "quality"
    _write_quality(quality_root, "accepted_clip")
    _write_quality(quality_root, "human_clip")
    _write_quality(quality_root, "unreviewed_clip")
    _write_quality(
        quality_root,
        "rejected_clip",
        failed_gates=["target_fit_pass", "passed"],
    )
    return semantics_path, quality_root, review_path


def _write_quality(quality_root: Path, clip_id: str, *, failed_gates=()):
    sample_dir = quality_root / clip_id
    sample_dir.mkdir(parents=True)
    safe_csv = sample_dir / f"{clip_id}_gmr_safe_18d.csv"
    raw_csv = sample_dir / f"{clip_id}_gmr_raw_18d.csv"
    rows = [[0.0] * len(EXPECTED_18D_JOINT_ORDER), [0.1] * len(EXPECTED_18D_JOINT_ORDER)]
    for path in (safe_csv, raw_csv):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(EXPECTED_18D_JOINT_ORDER)
            writer.writerows(rows)
    gates = {gate: True for gate in REQUIRED_18D_GATES}
    for gate in failed_gates:
        gates[gate] = False
    quality = {
        "source_motionx": str(root_motion := quality_root / f"{clip_id}.npy"),
        "output_contract": "ula_v2_18d_head_v1",
        "action_dim": 18,
        "joint_order": EXPECTED_18D_JOINT_ORDER,
        "frames": 2,
        "fps": 30,
        "quality_gate": gates,
        "outputs": {"raw_csv": str(raw_csv), "safe_csv": str(safe_csv)},
    }
    assert root_motion.stem == clip_id
    (sample_dir / "quality.json").write_text(json.dumps(quality), encoding="utf-8")


def _write_beat_quality(
    quality_root: Path,
    task_id: str,
    source_clip_id: str,
    start: int,
    end: int,
    *,
    actual_start: int | None = None,
    actual_end: int | None = None,
):
    _write_quality(quality_root, task_id)
    quality_path = quality_root / task_id / "quality.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality.pop("source_motionx")
    source_path = quality_root / "raw" / f"{source_clip_id}.npz"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(f"beat2:{source_clip_id}".encode())
    window_start = start if actual_start is None else actual_start
    window_end = end if actual_end is None else actual_end
    quality.update(
        {
            "source_beat2_npz": str(source_path),
            "source_sha256": sha256_file(source_path),
            "source_window_start_frame": window_start,
            "source_window_end_frame_exclusive": window_end,
            "source_window_convention": "zero_based_half_open_[start,end)",
            "source_window_frames": window_end - window_start,
            "source_frames": window_end - window_start,
        }
    )
    quality_path.write_text(json.dumps(quality), encoding="utf-8")
    return quality_path, source_path


def _beat_semantic(
    quality_root: Path,
    task_id: str,
    source_clip_id: str,
    start: int,
    end: int,
    quality_path: Path,
    source_path: Path,
):
    safe_csv = quality_root / task_id / f"{task_id}_gmr_safe_18d.csv"
    return {
        "schema_version": "beat2_kimodo_semantics_v1.1.0",
        "clip_id": task_id,
        "canonical_action": "bilateral_large_continuous_arm_motion",
        "canonical_prompt": {
            "en": "Move both arms broadly and continuously.",
            "zh": "连续大幅移动双臂。",
        },
        "speaker_key": "12_zhao",
        "source_clip_id": source_clip_id,
        "source_group_key": f"beat2/12_zhao/{source_clip_id}",
        "source_window_start_frame": start,
        "source_window_end_frame_exclusive": end,
        "source_beat2_npz": str(source_path),
        "source_sha256": sha256_file(source_path),
        "quality_json": str(quality_path),
        "quality_json_sha256": sha256_file(quality_path),
        "trajectory_path": str(safe_csv),
        "trajectory_sha256": sha256_file(safe_csv),
        "behavior_id": "Behavior.InteractPresence",
        "behavior_review_status": "human_confirmed",
        "behavior_supervision_mask": True,
        "emotion_id": None,
        "emotion_review_status": "unresolved",
        "emotion_supervision_mask": False,
        "network_semantic_supervision_ready": False,
        "motion_style": "energetic",
    }


def _write_semantics_and_reviews(root: Path, semantics: list[dict]):
    semantics_path = root / "beat_semantics.jsonl"
    semantics_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in semantics),
        encoding="utf-8",
    )
    review_path = root / "beat_review.json"
    review_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "review_id": "independent_beat_fixture",
                "reviewer": {
                    "kind": "agent",
                    "independent_of_annotation_logic": True,
                },
                "samples": [
                    _review_sample(record["clip_id"], "agent_reviewed", True)
                    for record in semantics
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return semantics_path, review_path


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _self_hashed_contract(payload):
    return dict(payload) | {
        "sha256": hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()
    }


def _add_verified_source_emotion(semantic, *, emotion_id="happy", source_sha256="a" * 64):
    semantic.update(
        {
            "emotion_id": emotion_id,
            "emotion_review_status": "official_protocol_confirmed",
            "emotion_supervision_mask": False,
            "source_emotion_label_verified": True,
            "emotion_source": "official_beat2_filename_protocol",
            "emotion_protocol_contract": _self_hashed_contract(
                {
                    "source": "official_beat2_filename_protocol",
                    "revision": "beat2-protocol-v1",
                    "emotion_id": emotion_id,
                    "source_sha256": source_sha256,
                }
            ),
            "source_sha256": source_sha256,
            "network_semantic_supervision_ready": False,
        }
    )
    return semantic


def test_official_source_emotion_without_blind_affect_stays_motion_only(
    tmp_path,
):
    clip_id = "official_event_clip"
    source_sha256 = "a" * 64
    semantic = _semantic(clip_id)
    semantic.update(
        {
            "behavior_id": "Behavior.InteractPresence",
            "behavior_review_status": "candidate_unreviewed",
            "behavior_supervision_mask": False,
            "behavior_source": "project_dataset_scope_weak_mapping_v1",
            "behavior_mapping_contract": _self_hashed_contract(
                {
                    "source": "project_dataset_scope_weak_mapping_v1",
                    "revision": "pilot-v1",
                    "behavior_id": "Behavior.InteractPresence",
                    "supervision": "weak_candidate_masked",
                }
            ),
            "emotion_id": "happy",
            "emotion_review_status": "official_protocol_confirmed",
            "emotion_supervision_mask": False,
            "source_emotion_label_verified": True,
            "emotion_source": "official_beat2_filename_protocol",
            "emotion_protocol_contract": _self_hashed_contract(
                {
                    "source": "official_beat2_filename_protocol",
                    "revision": "beat2-protocol-v1",
                    "emotion_id": "happy",
                    "source_sha256": source_sha256,
                }
            ),
            "source_sha256": source_sha256,
            "network_semantic_supervision_ready": False,
        }
    )
    semantics_path = tmp_path / "semantics.jsonl"
    semantics_path.write_text(json.dumps(semantic) + "\n", encoding="utf-8")
    review_path = tmp_path / "review.json"
    review_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "review_id": "weak-event-review",
                "reviewer": {
                    "kind": "agent",
                    "independent_of_annotation_logic": True,
                },
                "samples": [_review_sample(clip_id, "agent_reviewed", True)],
            }
        ),
        encoding="utf-8",
    )
    quality_root = tmp_path / "quality"
    _write_quality(quality_root, clip_id)

    adjudicate_dataset(
        semantics_path, quality_root, review_path, tmp_path / "output", expected_count=1
    )
    ready = _read_jsonl(tmp_path / "output" / "train_ready.jsonl")
    assert [record["clip_id"] for record in ready] == [clip_id]
    assert ready[0]["training_eligibility"]["behavior"] == {
        "eligible": False,
        "status": "masked_project_weak_candidate",
        "one_hot_supervision_mask": False,
        "requires": [
            "project_weak_mapping_provenance",
            "behavior_condition_channels_zero",
        ],
    }
    emotion = ready[0]["training_eligibility"]["emotion"]
    assert emotion["eligible"] is False
    assert emotion["source_label_verified"] is True
    assert emotion["affect_observable_verified"] is False
    assert emotion["conditioning_mask"] is False
    assert ready[0]["emotion_supervision_mask"] is False
    assert ready[0]["affect_observable_supervision_mask"] is False
    assert ready[0]["emotion_conditioning_mask"] is False


def test_matching_anonymous_blind_affect_enables_emotion_conditioning(tmp_path):
    clip_id = "blind_affect_match"
    semantic = _add_verified_source_emotion(_semantic(clip_id), emotion_id="happy")
    semantics_path = tmp_path / "semantics.jsonl"
    semantics_path.write_text(json.dumps(semantic) + "\n", encoding="utf-8")
    review = _review_sample(
        clip_id,
        "agent_reviewed",
        True,
        affect_observable=True,
        observed_emotion_id="happy",
    )
    review_path = tmp_path / "review.json"
    review_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "review_id": "blind-match-bundle",
                "reviewer": {
                    "kind": "agent",
                    "independent_of_annotation_logic": True,
                },
                "samples": [review],
            }
        ),
        encoding="utf-8",
    )
    quality_root = tmp_path / "quality"
    _write_quality(quality_root, clip_id)

    adjudicate_dataset(
        semantics_path, quality_root, review_path, tmp_path / "output", expected_count=1
    )
    record = _read_jsonl(tmp_path / "output/train_ready.jsonl")[0]
    assert record["training_eligibility"]["emotion"]["eligible"] is True
    assert record["emotion_supervision_mask"] is True
    assert record["affect_observable_supervision_mask"] is True
    assert record["emotion_conditioning_mask"] is True
    assert record["independent_review"]["blind_affect_review"][
        "observed_affect"
    ]["emotion_id"] == "happy"


def test_mismatched_blind_affect_keeps_emotion_masked(tmp_path):
    clip_id = "blind_affect_mismatch"
    semantic = _add_verified_source_emotion(_semantic(clip_id), emotion_id="happy")
    semantics_path = tmp_path / "semantics.jsonl"
    semantics_path.write_text(json.dumps(semantic) + "\n", encoding="utf-8")
    review = _review_sample(
        clip_id,
        "agent_reviewed",
        True,
        affect_observable=True,
        observed_emotion_id="neutral",
    )
    review_path = tmp_path / "review.json"
    review_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "review_id": "blind-mismatch-bundle",
                "reviewer": {
                    "kind": "agent",
                    "independent_of_annotation_logic": True,
                },
                "samples": [review],
            }
        ),
        encoding="utf-8",
    )
    quality_root = tmp_path / "quality"
    _write_quality(quality_root, clip_id)

    adjudicate_dataset(
        semantics_path, quality_root, review_path, tmp_path / "output", expected_count=1
    )
    record = _read_jsonl(tmp_path / "output/train_ready.jsonl")[0]
    assert record["training_eligibility"]["emotion"]["eligible"] is False
    assert record["training_eligibility"]["motion_style"]["eligible"] is True
    assert record["emotion_supervision_mask"] is False
    assert record["affect_observable_supervision_mask"] is False
    assert record["emotion_conditioning_mask"] is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda sample: sample["blind_affect_review"].update(
                target_emotion_exposed=True
            ),
            "exposed the target emotion",
        ),
        (
            lambda sample: sample["blind_affect_review"]["observed_affect"].update(
                confidence=0.2
            ),
            "below the formal threshold",
        ),
        (
            lambda sample: sample["blind_affect_review"].update(video_sha256="8" * 64),
            "not bound to the reviewed video",
        ),
    ],
)
def test_blind_affect_proof_fails_closed(tmp_path, mutate, message):
    sample = _review_sample(
        "blind_invalid",
        "agent_reviewed",
        True,
        affect_observable=True,
        observed_emotion_id="happy",
    )
    mutate(sample)
    path = tmp_path / "review.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "review_id": "invalid-blind-proof",
                "reviewer": {
                    "kind": "agent",
                    "independent_of_annotation_logic": True,
                },
                "samples": [sample],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=message):
        load_independent_reviews(path)


def test_adjudication_enforces_review_qc_and_semantic_precedence(tmp_path):
    semantics, quality_root, review = _write_inputs(tmp_path)
    output = tmp_path / "output"
    report = adjudicate_dataset(
        semantics,
        quality_root,
        review,
        output,
        expected_count=5,
    )

    train_ready = _read_jsonl(output / "train_ready.jsonl")
    needs_human = _read_jsonl(output / "needs_human.jsonl")
    rejected = _read_jsonl(output / "rejected.jsonl")
    assert [record["clip_id"] for record in train_ready] == ["accepted_clip"]
    assert [record["clip_id"] for record in needs_human] == [
        "accepted_missing_qc_clip",
        "human_clip",
        "unreviewed_clip",
    ]
    assert [record["clip_id"] for record in rejected] == ["rejected_clip"]

    accepted = train_ready[0]
    assert accepted["motion_18d"]["state"] == "passed"
    assert accepted["independent_review"]["training_acceptance"] is True
    assert accepted["split"]["assignment"] == "train"
    assert accepted["split"]["eval_eligible"] is False

    rejected_reasons = rejected[0]["adjudication"]["reasons"]
    assert "semantic_recommended_use_reject" in rejected_reasons
    assert "semantic_flag:canonical_action_text_conflict" in rejected_reasons
    assert "qc_gate_failed:target_fit_pass" in rejected_reasons
    assert "qc_gate_failed:passed" in rejected_reasons
    assert "independent_review_rejected" in rejected_reasons
    assert "independent_review_gate_failed:action_recognizable" in rejected_reasons
    assert report["counts"] == {"needs_human": 3, "rejected": 1, "train_ready": 1}
    assert report["quality_states"] == {"failed": 1, "missing": 1, "passed": 3}


def test_unreviewed_qc_pass_is_never_train_ready(tmp_path):
    semantics, quality_root, review = _write_inputs(tmp_path)
    output = tmp_path / "output"
    adjudicate_dataset(semantics, quality_root, review, output, expected_count=5)
    by_id = {record["clip_id"]: record for record in _read_jsonl(output / "needs_human.jsonl")}
    unreviewed = by_id["unreviewed_clip"]
    assert unreviewed["motion_18d"]["state"] == "passed"
    assert unreviewed["independent_review"]["present"] is False
    assert unreviewed["adjudication"]["reasons"] == ["independent_review_missing"]


def test_bad_safe_csv_turns_qc_into_rejection(tmp_path):
    semantics, quality_root, review = _write_inputs(tmp_path)
    path = quality_root / "unreviewed_clip/unreviewed_clip_gmr_safe_18d.csv"
    path.write_text("wrong,header\n0,0\n", encoding="utf-8")
    output = tmp_path / "output"
    adjudicate_dataset(semantics, quality_root, review, output, expected_count=5)
    by_id = {record["clip_id"]: record for record in _read_jsonl(output / "rejected.jsonl")}
    reasons = by_id["unreviewed_clip"]["adjudication"]["rejection_causes"]
    assert "qc_safe_csv_header_mismatch" in reasons
    assert "qc_safe_csv_row_width_mismatch" in reasons
    assert "qc_safe_csv_row_count_mismatch" in reasons


def test_outputs_are_byte_reproducible(tmp_path):
    semantics, quality_root, review = _write_inputs(tmp_path)
    output = tmp_path / "output"
    adjudicate_dataset(semantics, quality_root, review, output, expected_count=5)
    names = [*OUTPUT_NAMES, "dataset_scale_report.json"]
    first = {name: (output / name).read_bytes() for name in names}
    adjudicate_dataset(semantics, quality_root, review, output, expected_count=5)
    second = {name: (output / name).read_bytes() for name in names}
    assert first == second


def test_duplicate_semantics_are_rejected(tmp_path):
    path = tmp_path / "semantics.jsonl"
    record = _semantic("duplicate_clip")
    path.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate semantics clip_id"):
        load_semantics(path)


def test_rejected_quality_root_is_discovered_recursively(tmp_path):
    accepted_root = tmp_path / "accepted"
    rejected_root = tmp_path / "rejected/batch_1"
    _write_quality(rejected_root, "failed_clip", failed_gates=["passed"])
    index = index_quality_evidence(accepted_root, tmp_path / "rejected")
    assert index["failed_clip"]["partition"] == "rejected"


def test_duplicate_quality_evidence_is_rejected(tmp_path):
    accepted_root = tmp_path / "accepted"
    rejected_root = tmp_path / "rejected"
    _write_quality(accepted_root, "duplicate_clip")
    _write_quality(rejected_root, "duplicate_clip", failed_gates=["passed"])
    with pytest.raises(ValueError, match="duplicate 18D quality evidence"):
        index_quality_evidence(accepted_root, rejected_root)


def test_beat2_multiple_windows_pass_strict_source_and_window_provenance(tmp_path):
    quality_root = tmp_path / "quality"
    source_clip_id = "12_zhao_2_101_101"
    semantics = []
    for start, end in ((0, 180), (180, 360)):
        task_id = f"{source_clip_id}_f{start:06d}-{end:06d}"
        quality_path, source_path = _write_beat_quality(
            quality_root, task_id, source_clip_id, start, end
        )
        semantics.append(
            _beat_semantic(
                quality_root,
                task_id,
                source_clip_id,
                start,
                end,
                quality_path,
                source_path,
            )
        )
    semantics_path, review_path = _write_semantics_and_reviews(tmp_path, semantics)
    output = tmp_path / "output"

    report = adjudicate_dataset(
        semantics_path, quality_root, review_path, output, expected_count=2
    )

    records = _read_jsonl(output / "train_ready.jsonl")
    assert len(records) == 2
    assert report["quality_states"] == {"failed": 0, "missing": 0, "passed": 2}
    assert all(
        "qc_source_clip_mismatch" not in record["adjudication"]["reasons"]
        for record in records
    )
    assert {record["motion_18d"]["source_provenance"]["kind"] for record in records} == {
        "beat2"
    }
    assert {record["split"]["source_group_key"] for record in records} == {
        f"beat2/12_zhao/{source_clip_id}"
    }
    assert [
        record["motion_18d"]["source_provenance"]["window_start_frame"]
        for record in records
    ] == [0, 180]
    assert all(record["training_eligibility"]["motion_style"]["eligible"] for record in records)
    assert all(
        not record["training_eligibility"]["emotion"]["eligible"]
        and not record["training_eligibility"]["emotion"]["loss_mask"]
        and record["training_eligibility"]["emotion"]["status"]
        == "blocked_unresolved_emotion"
        for record in records
    )


def test_beat2_wrong_window_is_rejected_even_with_matching_file_hashes(tmp_path):
    quality_root = tmp_path / "quality"
    source_clip_id = "12_zhao_2_101_101"
    task_id = f"{source_clip_id}_f000000-000180"
    quality_path, source_path = _write_beat_quality(
        quality_root,
        task_id,
        source_clip_id,
        0,
        180,
        actual_start=1,
        actual_end=181,
    )
    semantic = _beat_semantic(
        quality_root,
        task_id,
        source_clip_id,
        0,
        180,
        quality_path,
        source_path,
    )
    semantics_path, review_path = _write_semantics_and_reviews(tmp_path, [semantic])
    output = tmp_path / "output"

    adjudicate_dataset(semantics_path, quality_root, review_path, output, expected_count=1)

    rejected = _read_jsonl(output / "rejected.jsonl")
    assert len(rejected) == 1
    reasons = rejected[0]["adjudication"]["rejection_causes"]
    assert "qc_source_window_start_mismatch" in reasons
    assert "qc_source_window_end_mismatch" in reasons


def test_beat2_wrong_source_clip_is_rejected(tmp_path):
    quality_root = tmp_path / "quality"
    expected_source = "12_zhao_2_101_101"
    task_id = f"{expected_source}_f000000-000180"
    quality_path, wrong_source_path = _write_beat_quality(
        quality_root, task_id, "13_lu_2_101_101", 0, 180
    )
    semantic = _beat_semantic(
        quality_root,
        task_id,
        expected_source,
        0,
        180,
        quality_path,
        wrong_source_path,
    )
    semantics_path, review_path = _write_semantics_and_reviews(tmp_path, [semantic])
    output = tmp_path / "output"

    adjudicate_dataset(semantics_path, quality_root, review_path, output, expected_count=1)

    rejected = _read_jsonl(output / "rejected.jsonl")
    assert "qc_source_clip_mismatch" in rejected[0]["adjudication"]["rejection_causes"]


def test_beat2_declared_trajectory_hash_mismatch_is_rejected(tmp_path):
    quality_root = tmp_path / "quality"
    source_clip_id = "12_zhao_2_101_101"
    task_id = f"{source_clip_id}_f000000-000180"
    quality_path, source_path = _write_beat_quality(
        quality_root, task_id, source_clip_id, 0, 180
    )
    semantic = _beat_semantic(
        quality_root,
        task_id,
        source_clip_id,
        0,
        180,
        quality_path,
        source_path,
    )
    semantic["trajectory_sha256"] = "0" * 64
    semantics_path, review_path = _write_semantics_and_reviews(tmp_path, [semantic])
    output = tmp_path / "output"

    adjudicate_dataset(semantics_path, quality_root, review_path, output, expected_count=1)

    rejected = _read_jsonl(output / "rejected.jsonl")
    assert "qc_safe_csv_sha256_mismatch" in rejected[0]["adjudication"][
        "rejection_causes"
    ]


def test_candidate_behavior_cannot_enter_motion_style_training(tmp_path):
    quality_root = tmp_path / "quality"
    source_clip_id = "12_zhao_2_101_101"
    task_id = f"{source_clip_id}_f000000-000180"
    quality_path, source_path = _write_beat_quality(
        quality_root, task_id, source_clip_id, 0, 180
    )
    semantic = _beat_semantic(
        quality_root,
        task_id,
        source_clip_id,
        0,
        180,
        quality_path,
        source_path,
    )
    semantic["behavior_review_status"] = "candidate_unreviewed"
    semantic["behavior_supervision_mask"] = False
    semantics_path, review_path = _write_semantics_and_reviews(tmp_path, [semantic])
    output = tmp_path / "output"

    adjudicate_dataset(semantics_path, quality_root, review_path, output, expected_count=1)

    pending = _read_jsonl(output / "needs_human.jsonl")
    assert len(pending) == 1
    assert pending[0]["motion_18d"]["state"] == "passed"
    assert pending[0]["independent_review"]["training_acceptance"] is True
    assert "semantic_behavior_confirmation_missing" in pending[0]["adjudication"][
        "reasons"
    ]
    assert pending[0]["training_eligibility"]["behavior"] == {
        "eligible": False,
        "one_hot_supervision_mask": False,
        "requires": ["human_confirmed_behavior", "motion_style_train_ready"],
        "status": "blocked_unconfirmed_behavior",
    }
    assert pending[0]["training_eligibility"]["motion_style"]["eligible"] is False


def test_head_only_motion_uses_18d_observability_review_gate(tmp_path):
    clip_id = "head_only_yaw_turn"
    semantics_path = tmp_path / "semantics.jsonl"
    semantic = _semantic(clip_id)
    semantic["canonical_action"] = "repeated_head_yaw_turns"
    semantic["canonical_prompt"] = {
        "en": "Turn the head repeatedly around the yaw axis.",
        "zh": "沿偏航轴反复转动头部。",
    }
    semantics_path.write_text(json.dumps(semantic) + "\n", encoding="utf-8")
    quality_root = tmp_path / "quality"
    _write_quality(quality_root, clip_id)
    safe_csv = quality_root / clip_id / f"{clip_id}_gmr_safe_18d.csv"
    with safe_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(EXPECTED_18D_JOINT_ORDER)
        writer.writerow([0.0] * 18)
        writer.writerow([0.0] * 15 + [0.0, 0.0, 0.2])
    _, review_path = _write_semantics_and_reviews(tmp_path, [semantic])
    output = tmp_path / "output"

    adjudicate_dataset(semantics_path, quality_root, review_path, output, expected_count=1)

    ready = _read_jsonl(output / "train_ready.jsonl")
    assert [record["clip_id"] for record in ready] == [clip_id]
    assert ready[0]["independent_review"]["gates"]["observable_in_18d"] is True


def test_legacy_15d_observability_gate_cannot_admit_18d_review(tmp_path):
    sample = _review_sample("head_only_yaw_turn", "agent_reviewed", True)
    sample["gates"]["observable_in_15d"] = sample["gates"].pop(
        "observable_in_18d"
    )
    review_path = tmp_path / "legacy_review.json"
    review_path.write_text(
        json.dumps(
            {
                "review_id": "legacy_15d_review",
                "reviewer": {
                    "kind": "agent",
                    "independent_of_annotation_logic": True,
                },
                "samples": [sample],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="independent review gates are incomplete"):
        load_independent_reviews(review_path)


OUTPUT_NAMES = ["train_ready.jsonl", "needs_human.jsonl", "rejected.jsonl"]
