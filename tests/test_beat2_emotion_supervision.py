import hashlib
import json
from pathlib import Path

import pytest

from upper_body_skeleton.beat2_emotion_supervision import (
    ADJUDICATION_KIND,
    AUDIT_KIND,
    CONFIG_KIND,
    CONTROLLER_KIND,
    EMOTIONS,
    MAPPING_KIND,
    OBSERVABILITY,
    PRIMARY_KIND,
    PROTOCOL,
    RECORD_KIND,
    STRONG_TIER,
    WEAK_TIER,
    build_emotion_supervision_from_config,
    load_emotion_training_rows,
    load_emotion_supervision_manifest,
    sha256_file,
    stable_json,
)


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    path.write_text(
        "".join(stable_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return sha256_file(path)


def _group_token(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:16]}"


def _weak_row(
    tmp_path: Path,
    *,
    sample_id: str,
    emotion: str,
    split: str = "train",
    speaker: str = "speaker-a",
    source_group: str = "source-a",
) -> dict:
    trajectory = tmp_path / f"{sample_id}.csv"
    trajectory.write_text(f"trajectory,{sample_id}\n", encoding="utf-8")
    digest = sha256_file(trajectory)
    return {
        "artifact_kind": "ula_v2_18d_expression_turn_retarget_v1",
        "status": "passed",
        "task_id": sample_id,
        "dataset": "BEAT2",
        "fixed_split_assignment": split,
        "speaker_key": speaker,
        "source_group_key": source_group,
        "emotion_id": emotion,
        "emotion_label_source": "official_beat2_filename_protocol",
        "source_emotion_label_verified": True,
        "official_emotion_conditioning_enabled": False,
        "emotion_supervision_mask": False,
        "quality_gate": {"passed": True, "velocity_pass": True},
        "safe_csv": str(trajectory),
        "safe_csv_sha256": digest,
    }


def _controller_row(weak: dict) -> dict:
    return {
        "schema_version": "1.0.0",
        "artifact_kind": CONTROLLER_KIND,
        "sample_id": "beat2_affect_controller_a",
        "dataset": "BEAT2",
        "controller_only_render_queue": True,
        "reviewers_must_not_receive_controller_queue": True,
        "reviewer_visible_projection_required": True,
        "fixed_split_assignment": weak["fixed_split_assignment"],
        "speaker_group_token": _group_token("speaker", weak["speaker_key"]),
        "source_group_token": _group_token("source", weak["source_group_key"]),
        "trajectory_path": weak["safe_csv"],
        "trajectory_sha256": weak["safe_csv_sha256"],
        "source_official_emotion_exposed_to_reviewers": False,
        "official_emotion_field_present": False,
        "official_emotion_is_trusted_supervision": False,
        "automated_emotion_label_assigned": False,
        "review_protocol": PROTOCOL,
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
        "emotion_supervision_mask": False,
        "affect_observable_supervision_mask": False,
        "official_emotion_conditioning_enabled": False,
        "accepted_for_training": False,
    }


def _mapping_row(
    tmp_path: Path,
    *,
    controller: dict,
    controller_line_sha256: str,
) -> dict:
    video = tmp_path / "anonymous.mp4"
    video.write_bytes(b"reviewed silent robot video")
    video_sha256 = sha256_file(video)
    binding = hashlib.sha256(b"sample binding").hexdigest()
    return {
        "schema_version": "1.0.0",
        "artifact_kind": MAPPING_KIND,
        "controller_sample_id": controller["sample_id"],
        "controller_record_line_sha256": controller_line_sha256,
        "public_sample_id": "motion_anonymous_a",
        "anonymous_video_path": str(video),
        "source_video_path": str(video),
        "trajectory_sha256": controller["trajectory_sha256"],
        "video_sha256": video_sha256,
        "render_input_record_sha256": "1" * 64,
        "render_pass_record_sha256": "2" * 64,
        "primary_assignments": {
            "primary_1": {
                "assignment_id": "assignment_primary_1",
                "sample_binding_sha256": binding,
                "shard_index": 0,
            },
            "primary_2": {
                "assignment_id": "assignment_primary_2",
                "sample_binding_sha256": binding,
                "shard_index": 0,
            },
        },
        "primary_reviewer_ids_must_be_distinct": True,
        "third_adjudication_required_on_any_primary_disagreement": True,
        "third_adjudicator_must_differ_from_primary_reviewers": True,
        "accepted_for_training": False,
    }


def _primary_submission(
    mapping: dict,
    *,
    role: str,
    reviewer: str,
    observability: str,
    emotion: str | None,
) -> dict:
    assignment = mapping["primary_assignments"][role]
    return {
        "schema_version": "1.0.0",
        "artifact_kind": PRIMARY_KIND,
        "sample_id": mapping["public_sample_id"],
        "assignment_id": assignment["assignment_id"],
        "assignment_role": role,
        "video_path": mapping["anonymous_video_path"],
        "video_sha256": mapping["video_sha256"],
        "sample_binding_sha256": assignment["sample_binding_sha256"],
        "review_protocol": PROTOCOL,
        "allowed_observability": list(OBSERVABILITY),
        "allowed_observed_emotions": list(EMOTIONS),
        "observability": observability,
        "observed_emotion": emotion,
        "confidence": 0.8,
        "reviewer_id": reviewer,
        "submitted_at_utc": "2026-07-27T08:00:00+00:00",
        "paired_primary_reviewer_must_be_distinct": True,
        "third_adjudication_required_on_any_disagreement": True,
        "source_label_visible": False,
        "accepted_for_training": False,
    }


def _adjudication_submission(
    mapping: dict,
    *,
    reviewer: str = "reviewer-c",
    observability: str = "observable",
    emotion: str | None = "fear",
) -> dict:
    return {
        "schema_version": "1.0.0",
        "artifact_kind": ADJUDICATION_KIND,
        "sample_id": mapping["public_sample_id"],
        "assignment_id": "adjudication_assignment_a",
        "assignment_role": "adjudication",
        "video_path": mapping["anonymous_video_path"],
        "video_sha256": mapping["video_sha256"],
        "sample_binding_sha256": mapping["primary_assignments"]["primary_1"][
            "sample_binding_sha256"
        ],
        "review_protocol": PROTOCOL,
        "allowed_observability": list(OBSERVABILITY),
        "allowed_observed_emotions": list(EMOTIONS),
        "observability": observability,
        "observed_emotion": emotion,
        "confidence": 0.9,
        "reviewer_id": reviewer,
        "submitted_at_utc": "2026-07-27T09:00:00+00:00",
        "source_label_visible": False,
        "primary_decisions_visible": False,
        "adjudicator_must_differ_from_primary_reviewers": True,
        "accepted_for_training": False,
    }


def _fixture(
    tmp_path: Path,
    *,
    primary_1: dict | None = None,
    primary_2: dict | None = None,
    adjudication: dict | None = None,
    weak_rows: list[dict] | None = None,
    controller_mutation=None,
) -> tuple[Path, dict]:
    reviewed = _weak_row(
        tmp_path,
        sample_id="reviewed-turn",
        emotion="sad",
        speaker="speaker-a",
        source_group="source-a",
    )
    unrelated = _weak_row(
        tmp_path,
        sample_id="unreviewed-turn",
        emotion="happy",
        speaker="speaker-b",
        source_group="source-b",
    )
    rows = weak_rows if weak_rows is not None else [reviewed, unrelated]
    weak_path = tmp_path / "weak.jsonl"
    weak_sha = _write_jsonl(weak_path, rows)

    controller = _controller_row(reviewed)
    if controller_mutation is not None:
        controller_mutation(controller)
    controller_path = tmp_path / "controller.jsonl"
    controller_sha = _write_jsonl(controller_path, [controller])
    controller_line_sha = hashlib.sha256(
        stable_json(controller).encode("utf-8")
    ).hexdigest()

    mapping = _mapping_row(
        tmp_path,
        controller=controller,
        controller_line_sha256=controller_line_sha,
    )
    mapping_path = tmp_path / "mapping.jsonl"
    mapping_sha = _write_jsonl(mapping_path, [mapping])

    if primary_1 == {}:
        primary_1 = _primary_submission(
            mapping,
            role="primary_1",
            reviewer="reviewer-a",
            observability="observable",
            emotion="happy",
        )
    if primary_2 == {}:
        primary_2 = _primary_submission(
            mapping,
            role="primary_2",
            reviewer="reviewer-b",
            observability="observable",
            emotion="happy",
        )
    if adjudication == {}:
        adjudication = _adjudication_submission(mapping)

    primary_paths: dict[str, list[dict[str, str]]] = {}
    for role, submission in (
        ("primary_1", primary_1),
        ("primary_2", primary_2),
    ):
        path = tmp_path / f"{role}.jsonl"
        inputs: list[dict[str, str]] = []
        if submission is not None:
            digest = _write_jsonl(path, [submission])
            inputs.append({"path": str(path), "sha256": digest})
        primary_paths[role] = inputs
    adjudication_inputs: list[dict[str, str]] = []
    if adjudication is not None:
        path = tmp_path / "adjudication.jsonl"
        digest = _write_jsonl(path, [adjudication])
        adjudication_inputs.append({"path": str(path), "sha256": digest})

    output = tmp_path / "emotion_supervision.jsonl"
    audit = tmp_path / "emotion_supervision_audit.json"
    pending = tmp_path / "pending_adjudication.jsonl"
    config = {
        "schema_version": "1.0.0",
        "artifact_kind": CONFIG_KIND,
        "allowed_dataset": "BEAT2",
        "weights": {
            "human_confirmed_observable": 1.0,
            "intended_metadata_weak": 0.1,
        },
        "weak_source_manifests": [
            {"path": str(weak_path), "sha256": weak_sha}
        ],
        "controller_queue": {
            "path": str(controller_path),
            "sha256": controller_sha,
        },
        "hidden_mapping": {
            "path": str(mapping_path),
            "sha256": mapping_sha,
        },
        "primary_submissions": primary_paths,
        "adjudication_submissions": adjudication_inputs,
        "output_manifest": str(output),
        "output_audit": str(audit),
        "output_adjudication_queue": str(pending),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    return config_path, {
        "reviewed": reviewed,
        "unrelated": unrelated,
        "controller": controller,
        "mapping": mapping,
        "output": output,
        "audit": audit,
        "pending": pending,
        "config": config,
    }


def test_primary_agreement_promotes_only_exact_trajectory(tmp_path: Path):
    config_path, paths = _fixture(
        tmp_path,
        primary_1={},
        primary_2={},
    )
    audit = build_emotion_supervision_from_config(config_path)
    rows = load_emotion_supervision_manifest(paths["output"])
    training_rows = load_emotion_training_rows(paths["output"])

    assert audit["artifact_kind"] == AUDIT_KIND
    assert len(rows) == 2
    assert len(training_rows) == 2
    by_tier = {row["supervision_tier"]: row for row in rows}
    strong = by_tier[STRONG_TIER]
    weak = by_tier[WEAK_TIER]
    assert strong["artifact_kind"] == RECORD_KIND
    assert strong["emotion_id"] == "happy"
    assert strong["trajectory_sha256"] == paths["reviewed"]["safe_csv_sha256"]
    assert strong["human_confirmed_observable"] is True
    assert strong["emotion_supervision_mask"] is True
    assert strong["emotion_loss_weight"] == 1.0
    assert strong["review_resolution"].startswith("two_independent")
    assert weak["sample_id"] == "unreviewed-turn"
    assert weak["emotion_id"] == "happy"
    assert weak["human_confirmed_observable"] is False
    assert weak["emotion_supervision_mask"] is False
    assert weak["weak_emotion_training_mask"] is True
    assert weak["emotion_loss_weight"] == 0.1
    assert audit["integrity"]["human_review_not_propagated_across_windows"] is True
    assert audit["counts"]["human_confirmed_observable_records"] == 1


def test_disagreement_suppresses_weak_and_materializes_third_review(
    tmp_path: Path,
):
    config_path, paths = _fixture(tmp_path)
    mapping = paths["mapping"]
    p1 = _primary_submission(
        mapping,
        role="primary_1",
        reviewer="reviewer-a",
        observability="observable",
        emotion="happy",
    )
    p2 = _primary_submission(
        mapping,
        role="primary_2",
        reviewer="reviewer-b",
        observability="observable",
        emotion="sad",
    )
    config_path, paths = _fixture(tmp_path, primary_1=p1, primary_2=p2)
    audit = build_emotion_supervision_from_config(config_path)
    rows = load_emotion_supervision_manifest(paths["output"])
    pending = [
        json.loads(line)
        for line in paths["pending"].read_text(encoding="utf-8").splitlines()
        if line
    ]

    assert [row["sample_id"] for row in rows] == ["unreviewed-turn"]
    assert audit["counts"]["pending_adjudications"] == 1
    assert (
        audit["counts"]["review_resolutions"][
            "primary_disagreement_pending_adjudication"
        ]
        == 1
    )
    assert len(pending) == 1
    assert pending[0]["artifact_kind"] == ADJUDICATION_KIND
    assert pending[0]["primary_decisions_visible"] is False
    assert pending[0]["source_label_visible"] is False
    assert pending[0]["observed_emotion"] is None


def test_distinct_third_reviewer_resolves_disagreement(tmp_path: Path):
    config_path, paths = _fixture(tmp_path)
    mapping = paths["mapping"]
    p1 = _primary_submission(
        mapping,
        role="primary_1",
        reviewer="reviewer-a",
        observability="observable",
        emotion="happy",
    )
    p2 = _primary_submission(
        mapping,
        role="primary_2",
        reviewer="reviewer-b",
        observability="observable",
        emotion="sad",
    )
    third = _adjudication_submission(mapping, emotion="fear")
    config_path, paths = _fixture(
        tmp_path,
        primary_1=p1,
        primary_2=p2,
        adjudication=third,
    )
    audit = build_emotion_supervision_from_config(config_path)
    rows = load_emotion_supervision_manifest(paths["output"])
    strong = next(row for row in rows if row["supervision_tier"] == STRONG_TIER)

    assert strong["emotion_id"] == "fear"
    assert strong["review_resolution"].startswith("distinct_third")
    assert strong["review_receipt"]["adjudication"]["reviewer_id"] == "reviewer-c"
    assert audit["counts"]["pending_adjudications"] == 0


def test_adjudicator_must_differ_from_both_primary_reviewers(tmp_path: Path):
    config_path, paths = _fixture(tmp_path)
    mapping = paths["mapping"]
    p1 = _primary_submission(
        mapping,
        role="primary_1",
        reviewer="reviewer-a",
        observability="observable",
        emotion="happy",
    )
    p2 = _primary_submission(
        mapping,
        role="primary_2",
        reviewer="reviewer-b",
        observability="observable",
        emotion="sad",
    )
    third = _adjudication_submission(
        mapping,
        reviewer="reviewer-a",
        emotion="fear",
    )
    config_path, _ = _fixture(
        tmp_path,
        primary_1=p1,
        primary_2=p2,
        adjudication=third,
    )
    with pytest.raises(ValueError, match="must differ"):
        build_emotion_supervision_from_config(config_path)


def test_non_observable_agreement_never_becomes_an_emotion_target(
    tmp_path: Path,
):
    config_path, paths = _fixture(tmp_path)
    mapping = paths["mapping"]
    p1 = _primary_submission(
        mapping,
        role="primary_1",
        reviewer="reviewer-a",
        observability="not_observable",
        emotion=None,
    )
    p2 = _primary_submission(
        mapping,
        role="primary_2",
        reviewer="reviewer-b",
        observability="not_observable",
        emotion=None,
    )
    config_path, paths = _fixture(tmp_path, primary_1=p1, primary_2=p2)
    audit = build_emotion_supervision_from_config(config_path)
    rows = load_emotion_supervision_manifest(paths["output"])

    assert [row["sample_id"] for row in rows] == ["unreviewed-turn"]
    assert (
        audit["counts"]["review_resolutions"][
            "primary_agreement_not_observable_excluded"
        ]
        == 1
    )
    assert audit["counts"]["human_confirmed_observable_records"] == 0


def test_controller_cannot_smuggle_the_official_emotion_into_blind_review(
    tmp_path: Path,
):
    def expose_label(row: dict) -> None:
        row["emotion_id"] = "happy"

    config_path, _ = _fixture(
        tmp_path,
        controller_mutation=expose_label,
    )
    with pytest.raises(ValueError, match="exposes source emotion"):
        build_emotion_supervision_from_config(config_path)


def test_cross_split_speaker_or_source_leakage_is_a_hard_error(tmp_path: Path):
    first = _weak_row(
        tmp_path,
        sample_id="train-row",
        emotion="happy",
        split="train",
        speaker="shared-speaker",
        source_group="train-source",
    )
    second = _weak_row(
        tmp_path,
        sample_id="test-row",
        emotion="sad",
        split="test",
        speaker="shared-speaker",
        source_group="test-source",
    )
    config_path, _ = _fixture(tmp_path, weak_rows=[first, second])
    with pytest.raises(ValueError, match="cross split/source leakage"):
        build_emotion_supervision_from_config(config_path)


def test_training_loader_never_returns_validation_or_test_rows(tmp_path: Path):
    train_row = _weak_row(
        tmp_path,
        sample_id="train-row",
        emotion="happy",
        split="train",
        speaker="train-speaker",
        source_group="train-source",
    )
    test_row = _weak_row(
        tmp_path,
        sample_id="test-row",
        emotion="sad",
        split="test",
        speaker="test-speaker",
        source_group="test-source",
    )
    config_path, paths = _fixture(
        tmp_path,
        weak_rows=[train_row, test_row],
    )
    build_emotion_supervision_from_config(config_path)
    rows = load_emotion_training_rows(paths["output"])

    assert [row["sample_id"] for row in rows] == ["train-row"]
    assert all(row["fixed_split_assignment"] == "train" for row in rows)


def test_forbidden_external_dataset_marker_is_rejected_recursively(
    tmp_path: Path,
):
    weak = _weak_row(
        tmp_path,
        sample_id="row-a",
        emotion="happy",
    )
    weak["provenance_note"] = "derived from KIMODO"
    config_path, _ = _fixture(tmp_path, weak_rows=[weak])
    with pytest.raises(ValueError, match="forbidden dataset marker"):
        build_emotion_supervision_from_config(config_path)


def test_tampered_assignment_binding_is_rejected(tmp_path: Path):
    config_path, paths = _fixture(tmp_path)
    mapping = paths["mapping"]
    p1 = _primary_submission(
        mapping,
        role="primary_1",
        reviewer="reviewer-a",
        observability="observable",
        emotion="happy",
    )
    p1["sample_binding_sha256"] = "0" * 64
    p2 = _primary_submission(
        mapping,
        role="primary_2",
        reviewer="reviewer-b",
        observability="observable",
        emotion="happy",
    )
    config_path, _ = _fixture(tmp_path, primary_1=p1, primary_2=p2)
    with pytest.raises(ValueError, match="submission binding failed"):
        build_emotion_supervision_from_config(config_path)
