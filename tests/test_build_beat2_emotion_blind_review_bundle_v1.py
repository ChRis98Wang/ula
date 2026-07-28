from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from tools.human_motion_review import render_beat2_annotation_review as renderer
from tools.human_motion_review.build_beat2_emotion_blind_review_bundle_v1 import (
    EMOTIONS,
    OBSERVABILITY,
    PRIMARY_ROLES,
    PUBLIC_TASK_KEYS,
    build_bundle,
    read_jsonl_bound,
    sha256_file,
    value_sha256,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _trajectory(path: Path, frames: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(renderer.JOINT_ORDER)
        for frame in range(frames):
            writer.writerow([frame / 100.0] * len(renderer.JOINT_ORDER))


def _source_row(trajectory: Path, *, suffix: str = "001") -> dict:
    digest = sha256_file(trajectory)
    return {
        "schema_version": "1.0.0",
        "artifact_kind": "beat2_robot_observable_emotion_review_queue_record_v1",
        "sample_id": f"beat2_affect_{suffix:0>24}",
        "dataset": "BEAT2",
        "controller_only_render_queue": True,
        "reviewers_must_not_receive_controller_queue": True,
        "reviewer_visible_projection_required": True,
        "fixed_split_assignment": "train",
        "speaker_group_token": "speaker_0123456789abcdef",
        "source_group_token": "source_0123456789abcdef",
        "source_task_token": "turn_0123456789abcdef",
        "source_record_line_sha256": "1" * 64,
        "expression_turn_record_sha256": "2" * 64,
        "source_interval": {"start_frame": 100, "end_frame_exclusive": 104},
        "trajectory_path": str(trajectory),
        "trajectory_sha256": digest,
        "trajectory_reference_policy": (
            "reference_existing_physical_qc_pass_only_no_copy_no_rewindow"
        ),
        "trajectory_copied": False,
        "fps": 30.0,
        "duration_sec": 4 / 30,
        "duration_bin": "short_under_3s",
        "gesture_category_balance_key": "controller-only-category",
        "source_official_emotion_exposed_to_reviewers": False,
        "official_emotion_field_present": False,
        "official_emotion_is_trusted_supervision": False,
        "automated_emotion_label_assigned": False,
        "review_protocol": (
            "two_independent_blind_robot_affect_reviews_then_adjudication_v1"
        ),
        "allowed_observability": list(OBSERVABILITY),
        "allowed_observed_emotions": list(EMOTIONS),
        "primary_reviews": [
            {
                "slot": "reviewer_1",
                "reviewer_id": None,
                "observability": None,
                "observed_emotion": None,
                "confidence": None,
                "submitted_at_utc": None,
                "status": "pending_independent_review",
            },
            {
                "slot": "reviewer_2",
                "reviewer_id": None,
                "observability": None,
                "observed_emotion": None,
                "confidence": None,
                "submitted_at_utc": None,
                "status": "pending_independent_review",
            },
        ],
        "primary_reviewer_ids_must_be_distinct": True,
        "primary_agreement_required": True,
        "third_adjudication_required_on_any_primary_disagreement": True,
        "third_adjudication": {
            "reviewer_id": None,
            "must_differ_from_primary_reviewers": True,
            "observability": None,
            "observed_emotion": None,
            "confidence": None,
            "submitted_at_utc": None,
            "status": "not_requested_pending_primary_reviews",
        },
        "review_state": "pending_two_independent_reviews",
        "supervision_gate_status": (
            "closed_pending_primary_agreement_or_distinct_third_adjudication"
        ),
        "emotion_supervision_mask": False,
        "affect_observable_supervision_mask": False,
        "official_emotion_conditioning_enabled": False,
        "accepted_for_training": False,
    }


def _fixture(tmp_path: Path) -> tuple[Path, dict]:
    trajectory = tmp_path / "beat2" / "motion.csv"
    _trajectory(trajectory)
    source = _source_row(trajectory)
    queue = tmp_path / "review_queue.jsonl"
    _write_jsonl(queue, [source])
    return queue, source


def _render_pass(
    *,
    render_record: dict,
    video: Path,
) -> dict:
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"silent-rendered-video")
    return {
        "schema_version": "1.0.0",
        "task_id": render_record["task_id"],
        "status": "passed",
        "input_fingerprint": value_sha256(render_record),
        "trajectory_sha256": render_record["trajectory_sha256"],
        "video_path": str(video),
        "video_sha256": sha256_file(video),
        "video_check": {"passed": True, "audio_streams": 0},
        "emotion_supervision_mask": False,
        "official_emotion_conditioning_enabled": False,
        "affect_observable_supervision_mask": False,
        "manual_review_required": True,
        "render_pass_grants_training_admission": False,
        "accepted_for_training": False,
    }


def test_prepare_writes_a_renderer_compatible_controller_queue_only(
    tmp_path: Path,
) -> None:
    queue, source = _fixture(tmp_path)
    output = tmp_path / "bundle"
    result = build_bundle(
        controller_queue=queue,
        output_root=output,
        shard_count=2,
        secret_hex="11" * 32,
    )

    render_queue = Path(result["controller_only"]["render_queue"])
    render_rows = [row for row, _binding in read_jsonl_bound(render_queue)]
    renderer.validate_queue_structure(render_rows)
    assert len(render_rows) == 1
    render = render_rows[0]
    assert render["task_id"] == source["sample_id"]
    assert render["canonical_action"] == "blind_robot_affect_observation"
    assert render["canonical_prompt"]["en"].startswith("Observe this silent")
    assert "emotion_id" not in render
    assert "gesture_category_balance_key" not in render
    assert result["reviewer_visible"]["rendered_records"] == 0
    assert result["reviewer_visible"]["primary_review_tasks"] == {
        role: 0 for role in PRIMARY_ROLES
    }
    assert result["controller_only"]["accepted_for_training"] is False


def test_render_pass_builds_two_anonymous_hash_bound_primary_shard_sets(
    tmp_path: Path,
) -> None:
    queue, source = _fixture(tmp_path)
    output = tmp_path / "bundle"
    first = build_bundle(
        controller_queue=queue,
        output_root=output,
        shard_count=2,
        secret_hex="22" * 32,
    )
    render_record = read_jsonl_bound(
        Path(first["controller_only"]["render_queue"])
    )[0][0]
    passed = _render_pass(
        render_record=render_record,
        video=tmp_path / "rendered" / "source_happy_named_video.mp4",
    )
    passed_manifest = tmp_path / "passed.jsonl"
    _write_jsonl(passed_manifest, [passed])

    result = build_bundle(
        controller_queue=queue,
        output_root=output,
        render_passed_manifests=[passed_manifest],
        shard_count=2,
        secret_hex="22" * 32,
    )

    public_rows = {}
    for role in PRIMARY_ROLES:
        role_summary = result["reviewer_visible"]["primary_roles"][role]
        rows = [
            row
            for row, _binding in read_jsonl_bound(
                Path(role_summary["full_review_queue"])
            )
        ]
        assert len(rows) == 1
        public_rows[role] = rows[0]
        assert set(rows[0]) == PUBLIC_TASK_KEYS
        assert rows[0]["assignment_role"] == role
        assert rows[0]["source_label_visible"] is False
        assert rows[0]["observability"] is None
        assert rows[0]["observed_emotion"] is None
        assert rows[0]["accepted_for_training"] is False
        assert role_summary["coverage_complete_without_overlap"] is True
        assert role_summary["maximum_shard_records"] == 1
        assert role_summary["minimum_shard_records"] == 0

    assert public_rows["primary_1"]["sample_id"] == public_rows["primary_2"][
        "sample_id"
    ]
    assert public_rows["primary_1"]["assignment_id"] != public_rows["primary_2"][
        "assignment_id"
    ]
    anonymous_video = Path(public_rows["primary_1"]["video_path"])
    assert anonymous_video.name == f'{public_rows["primary_1"]["sample_id"]}.mp4'
    assert sha256_file(anonymous_video) == passed["video_sha256"]
    serialized = json.dumps(public_rows, ensure_ascii=False)
    assert source["sample_id"] not in serialized
    assert source["speaker_group_token"] not in serialized
    assert source["gesture_category_balance_key"] not in serialized
    assert source["trajectory_path"] not in serialized

    hidden = read_jsonl_bound(
        Path(result["controller_only"]["mapping"])
    )[0][0]
    assert hidden["controller_sample_id"] == source["sample_id"]
    assert hidden["trajectory_sha256"] == source["trajectory_sha256"]
    assert set(hidden["primary_assignments"]) == set(PRIMARY_ROLES)
    assert result["reviewer_visible"][
        "every_rendered_sample_assigned_to_two_primary_reviews"
    ] is True
    assert result["reviewer_visible"][
        "third_adjudication_queue_state"
    ] == "not_materialized_until_primary_disagreement"
    assert (
        output
        / "reviewer_visible/adjudication/pending_queue.jsonl"
    ).read_text(encoding="utf-8") == ""


def test_source_label_field_is_rejected_even_if_absence_flag_lies(
    tmp_path: Path,
) -> None:
    queue, source = _fixture(tmp_path)
    source["hidden_metadata"] = {"official_emotion": "happy"}
    _write_jsonl(queue, [source])
    with pytest.raises(ValueError, match="contains source emotion fields"):
        build_bundle(
            controller_queue=queue,
            output_root=tmp_path / "bundle",
            secret_hex="33" * 32,
        )


def test_kimodo_trajectory_is_rejected(tmp_path: Path) -> None:
    trajectory = tmp_path / "Kimodo" / "motion.csv"
    _trajectory(trajectory)
    queue = tmp_path / "review_queue.jsonl"
    _write_jsonl(queue, [_source_row(trajectory)])
    with pytest.raises(ValueError, match="forbidden dataset marker"):
        build_bundle(
            controller_queue=queue,
            output_root=tmp_path / "bundle",
            secret_hex="44" * 32,
        )


def test_tampered_render_fingerprint_is_rejected(tmp_path: Path) -> None:
    queue, _source = _fixture(tmp_path)
    output = tmp_path / "bundle"
    first = build_bundle(
        controller_queue=queue,
        output_root=output,
        secret_hex="55" * 32,
    )
    render_record = read_jsonl_bound(
        Path(first["controller_only"]["render_queue"])
    )[0][0]
    passed = _render_pass(
        render_record=render_record,
        video=tmp_path / "rendered.mp4",
    )
    passed["input_fingerprint"] = hashlib.sha256(b"tampered").hexdigest()
    manifest = tmp_path / "passed.jsonl"
    _write_jsonl(manifest, [passed])
    with pytest.raises(ValueError, match="input_fingerprint"):
        build_bundle(
            controller_queue=queue,
            output_root=output,
            render_passed_manifests=[manifest],
            secret_hex="55" * 32,
        )
