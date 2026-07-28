import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.human_motion_review.build_beat2_emotion_review_queue_v1 import (
    CONFIG_KIND,
    EMOTIONS,
    MAX_TOTAL,
    OBSERVABILITY_OPTIONS,
    QUEUE_KIND,
    SOURCE_ARTIFACT_KIND,
    SOURCE_EMOTION_LABELS,
    TARGETS,
    build_from_config,
    sha256_file,
)


def _source_row(
    tmp_path: Path,
    *,
    task_id: str,
    source_group: str,
    speaker: str = "speaker-a",
    split: str = "train",
    emotion: str = "happy",
    category: str = "iconic",
    duration_bin: str = "short_under_3s",
    start: int = 10,
) -> dict:
    trajectory = tmp_path / f"{task_id}.csv"
    trajectory.write_text(f"unique,{task_id}\n", encoding="utf-8")
    duration = {
        "short_under_3s": 2.0,
        "medium_3_to_6s": 4.0,
        "long_over_6s": 7.0,
    }[duration_bin]
    frame_count = int(duration * 30)
    return {
        "artifact_kind": SOURCE_ARTIFACT_KIND,
        "status": "passed",
        "task_id": task_id,
        "dataset": "BEAT2",
        "accepted_for_training": False,
        "emotion_supervision_mask": False,
        "official_emotion_conditioning_enabled": False,
        "affect_observable_supervision_mask": False,
        "canonical_prompt": None,
        "canonical_action": None,
        "semantic_supervision_masks": {
            "official_category": False,
            "prompt_text": False,
        },
        "quality_gate": {"passed": True, "velocity_pass": True},
        "fixed_split_assignment": split,
        "official_split": "train",
        "speaker_key": speaker,
        "source_group_key": source_group,
        "emotion_id": emotion,
        "source_emotion_label": SOURCE_EMOTION_LABELS[emotion],
        "source_emotion_label_verified": True,
        "emotion_label_source": "official_beat2_filename_protocol",
        "fps": 30.0,
        "duration_sec": duration,
        "duration_band": duration_bin,
        "training_segment": {
            "duration_policy": (
                "natural_rest_to_natural_rest_no_fixed_or_max_duration"
            ),
            "fixed_window_sec": None,
            "cropped": False,
            "start_frame": start,
            "end_frame_exclusive": start + frame_count,
            "frame_count": frame_count,
        },
        "retarget_segment": {
            "duration_policy": (
                "natural_rest_to_natural_rest_no_fixed_or_max_duration"
            ),
            "fixed_target_duration_sec": None,
            "cropped": False,
        },
        "expression_turn": {"official_categories": [category]},
        "expression_turn_record_sha256": hashlib.sha256(
            task_id.encode("utf-8")
        ).hexdigest(),
        "safe_csv": str(trajectory),
        "safe_csv_sha256": sha256_file(trajectory),
        "source_speech_context": "must never enter the review queue",
    }


def _write_config(tmp_path: Path, rows: list[dict]) -> tuple[Path, Path, Path]:
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    output = tmp_path / "review_queue.jsonl"
    audit = tmp_path / "audit.json"
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "allowed_dataset": "BEAT2",
                "artifact_kind": CONFIG_KIND,
                "max_total": MAX_TOTAL,
                "output_audit": str(audit),
                "output_manifest": str(output),
                "seed": 17,
                "source_group_policy": "one_turn_per_source_group",
                "source_manifest": str(source),
                "source_manifest_sha256": sha256_file(source),
                "targets": TARGETS,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return config, output, audit


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_queue_is_fail_closed_and_initializes_independent_review_slots(tmp_path: Path):
    row = _source_row(
        tmp_path,
        task_id="turn-a",
        source_group="source-a",
        emotion="happy",
    )
    config, output, audit_path = _write_config(tmp_path, [row])
    audit = build_from_config(config)
    queued = _read_jsonl(output)

    assert len(queued) == 1
    record = queued[0]
    assert record["artifact_kind"] == QUEUE_KIND
    assert record["dataset"] == "BEAT2"
    assert "emotion_id" not in record
    assert "source_emotion_label" not in record
    assert "source_speech_context" not in record
    assert record["controller_only_render_queue"] is True
    assert record["reviewers_must_not_receive_controller_queue"] is True
    assert record["reviewer_visible_projection_required"] is True
    assert record["source_official_emotion_exposed_to_reviewers"] is False
    assert record["official_emotion_field_present"] is False
    assert record["official_emotion_is_trusted_supervision"] is False
    assert record["automated_emotion_label_assigned"] is False
    assert record["allowed_observability"] == list(OBSERVABILITY_OPTIONS)
    assert len(record["primary_reviews"]) == 2
    assert all(slot["reviewer_id"] is None for slot in record["primary_reviews"])
    assert all(slot["observability"] is None for slot in record["primary_reviews"])
    assert record["primary_reviewer_ids_must_be_distinct"] is True
    assert record["primary_agreement_required"] is True
    assert record["third_adjudication_required_on_any_primary_disagreement"] is True
    assert record["emotion_supervision_mask"] is False
    assert record["affect_observable_supervision_mask"] is False
    assert record["accepted_for_training"] is False
    assert record["trajectory_copied"] is False
    assert audit["review_contract"]["automated_labels_created"] == 0
    assert audit["review_contract"]["emotion_supervision_masks_enabled"] == 0
    assert json.loads(audit_path.read_text())["output"]["manifest_sha256"] == sha256_file(
        output
    )


def test_selection_uses_only_one_natural_turn_per_source_group(tmp_path: Path):
    rows = [
        _source_row(
            tmp_path,
            task_id="turn-a1",
            source_group="source-a",
            emotion="fear",
            category="iconic",
            start=10,
        ),
        _source_row(
            tmp_path,
            task_id="turn-a2",
            source_group="source-a",
            emotion="fear",
            category="deictic",
            start=100,
        ),
        _source_row(
            tmp_path,
            task_id="turn-b",
            source_group="source-b",
            emotion="fear",
            speaker="speaker-b",
            duration_bin="medium_3_to_6s",
        ),
    ]
    config, output, _ = _write_config(tmp_path, rows)
    audit = build_from_config(config)
    queued = _read_jsonl(output)

    assert len(queued) == 2
    assert len({row["source_group_token"] for row in queued}) == 2
    leakage = audit["leakage_and_copy_audit"]
    assert leakage["one_turn_per_source_group"] is True
    assert leakage["unique_trajectory_hashes"] is True
    assert leakage["unique_source_intervals"] is True
    assert leakage["trajectory_files_copied"] == 0
    assert audit["availability"]["raw_turn_counts_by_split_emotion"]["train"]["fear"] == 3
    assert (
        audit["availability"]["unique_source_group_capacity_by_split_emotion"][
            "train"
        ]["fear"]
        == 2
    )


def test_fixed_target_caps_and_balances_speakers(tmp_path: Path):
    rows = [
        _source_row(
            tmp_path,
            task_id=f"turn-{index:03d}",
            source_group=f"source-{index:03d}",
            speaker=f"speaker-{index % 4}",
            emotion="angry",
            category=("iconic", "deictic", "metaphoric")[index % 3],
            duration_bin=(
                "short_under_3s",
                "medium_3_to_6s",
                "long_over_6s",
            )[index % 3],
            start=index * 300,
        )
        for index in range(104)
    ]
    config, output, _ = _write_config(tmp_path, rows)
    audit = build_from_config(config)
    queued = _read_jsonl(output)

    assert len(queued) == 100
    assert audit["selection"]["counts_by_split_emotion"]["train"]["angry"] == 100
    speaker_counts = audit["selection"]["speaker_balance_by_split_emotion"]["train"][
        "angry"
    ]
    assert max(speaker_counts.values()) - min(speaker_counts.values()) <= 1
    assert audit["selection"]["shortfall_by_split_emotion"]["train"]["angry"] == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"dataset": "external"}, "fail-closed BEAT2"),
        ({"affect_observable_supervision_mask": True}, "fail-closed BEAT2"),
    ],
)
def test_rejects_external_or_pretrusted_sources(
    tmp_path: Path, mutation: dict, message: str
):
    row = _source_row(
        tmp_path, task_id="turn-a", source_group="source-a", emotion="sad"
    )
    row.update(mutation)
    config, _, _ = _write_config(tmp_path, [row])
    with pytest.raises(ValueError, match=message):
        build_from_config(config)


def test_rejects_speaker_or_source_group_split_leakage(tmp_path: Path):
    base = _source_row(
        tmp_path,
        task_id="turn-a",
        source_group="source-a",
        speaker="speaker-a",
        split="train",
    )
    speaker_leak = _source_row(
        tmp_path,
        task_id="turn-b",
        source_group="source-b",
        speaker="speaker-a",
        split="test",
    )
    config, _, _ = _write_config(tmp_path, [base, speaker_leak])
    with pytest.raises(ValueError, match="speaker leakage"):
        build_from_config(config)

    group_leak = copy.deepcopy(speaker_leak)
    group_leak["speaker_key"] = "speaker-b"
    group_leak["source_group_key"] = "source-a"
    config, _, _ = _write_config(tmp_path, [base, group_leak])
    with pytest.raises(ValueError, match="source-group leakage"):
        build_from_config(config)
