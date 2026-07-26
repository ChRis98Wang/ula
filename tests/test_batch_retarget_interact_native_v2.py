import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.gmr_v2 import batch_retarget_interact_native_v2 as BATCH


def _stable(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def test_atomic_json_retries_transient_cifs_permission_error(tmp_path, monkeypatch):
    destination = tmp_path / "state.json"
    real_replace = BATCH.os.replace
    attempts = []

    def flaky_replace(source, target):
        attempts.append((source, target))
        if len(attempts) < 3:
            raise PermissionError("transient CIFS sharing violation")
        real_replace(source, target)

    monkeypatch.setattr(BATCH.os, "replace", flaky_replace)
    monkeypatch.setattr(BATCH.time, "sleep", lambda _seconds: None)

    BATCH.atomic_json(destination, {"status": "running"})

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "status": "running"
    }
    assert len(attempts) == 3


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(_stable(row) + "\n" for row in rows), encoding="utf-8")


def _catalog(tmp_path: Path):
    raw_root = tmp_path / "raw"
    source = raw_root / "bvhs/20240101_001_001.bvh"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"native bvh fixture")
    row = {
        "schema_version": "1.1.0",
        "artifact_kind": BATCH.TASK_ARTIFACT_KIND,
        "episode_task_id": "natural-turn-001",
        "source_interval": {
            "start_frame": 17,
            "end_frame_exclusive": 244,
            "frame_count": 227,
        },
        "training_source_interval": None,
        "context_plan": {
            "duration_policy": BATCH.NATURAL_DURATION_POLICY,
            "duration_gate_used": False,
            "selected_training_interval": None,
        },
        "interaction_partner_lineage": {"actor_id": "002"},
        "retarget_task": {
            "source_bvh": "bvhs/20240101_001_001.bvh",
            "source_bvh_sha256": _sha(source),
            "source_frame_interval": [17, 244],
            "partner_motion_mixed_into_target": False,
        },
        "semantic_supervision_masks": {
            "communicative_intent": False,
            "prompt_text": False,
        },
        "emotion_supervision_mask": False,
        "admission_mask": False,
        "accepted_for_training": False,
    }
    row["episode_task_record_sha256"] = hashlib.sha256(
        _stable(row).encode("utf-8")
    ).hexdigest()
    manifest = tmp_path / "catalog/tasks.jsonl"
    _jsonl(manifest, [row])
    summary = {
        "artifact_kind": BATCH.CATALOG_ARTIFACT_KIND,
        "duration_policy": BATCH.NATURAL_DURATION_POLICY,
        "duration_contract_audit": {"forbidden_constraint_key_paths": []},
        "selection": {
            "scope": "full_pool_all_locally_complete_paired_performances",
            "accepted_for_training_count": 0,
            "actor_specific_robot_episode_task_count": 1,
        },
        "license_gate": {"training_authorized_by_this_receipt": False},
        "artifacts": {
            "actor_robot_episode_tasks": {
                "path": str(manifest.resolve()),
                "sha256": _sha(manifest),
            }
        },
    }
    summary_path = tmp_path / "catalog/summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return raw_root, manifest, summary_path, row


def _axis_review(tmp_path: Path, *, overall="pass"):
    video = tmp_path / "review/video.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    queue = tmp_path / "review/axis_queue.jsonl"
    queue_row = {
        "sample_id": "axis-001",
        "video_sha256": _sha(video),
        "video_path": str(video),
    }
    _jsonl(queue, [queue_row])
    public = {
        "artifact_kind": "interact_native_bvh_separate_anonymous_blind_review_bundle_v2",
        "fixed_duration_window_used": False,
        "identity_scenario_official_text_or_emotion_exposed": False,
        "axis_queue": str(queue.resolve()),
        "axis_queue_sha256": _sha(queue),
        "axis_records": 1,
    }
    public_path = tmp_path / "review/summary.json"
    public_path.write_text(json.dumps(public), encoding="utf-8")
    review = tmp_path / "review/axis_reviewer.jsonl"
    _jsonl(
        review,
        [
            {
                "sample_id": "axis-001",
                "protocol_version": BATCH.AXIS_PROTOCOL,
                "axis_queue_sha256": _sha(queue),
                "public_summary_sha256": _sha(public_path),
                "video_sha256": _sha(video),
                "video_sha256_verified": True,
                "decode_complete": True,
                "declared_frame_count": 1,
                "decoded_frame_count": 1,
                "native_duration_preserved": True,
                "fixed_duration_window_used": False,
                "label_metadata_exposed": False,
                "overall_result": overall,
                "accepted_for_training": False,
                "review_id": "axis-review-001",
                "reviewer_id": "independent-axis-reviewer",
            }
        ],
    )
    return public_path, queue, review


def test_catalog_binding_preserves_native_interval_and_closed_masks(tmp_path):
    raw_root, manifest, summary, _row = _catalog(tmp_path)
    jobs, binding = BATCH.load_catalog(manifest, summary, raw_root)
    assert len(jobs) == 1
    assert jobs[0]["start_frame"] == 17
    assert jobs[0]["end_frame"] == 244
    assert binding["duration_policy"] == BATCH.NATURAL_DURATION_POLICY
    assert binding["training_authorized_by_catalog"] is False


def test_catalog_rejects_a_fixed_duration_gate(tmp_path):
    raw_root, manifest, summary, row = _catalog(tmp_path)
    row["context_plan"]["duration_gate_used"] = True
    row.pop("episode_task_record_sha256")
    row["episode_task_record_sha256"] = hashlib.sha256(
        _stable(row).encode("utf-8")
    ).hexdigest()
    _jsonl(manifest, [row])
    value = json.loads(summary.read_text())
    value["artifacts"]["actor_robot_episode_tasks"]["sha256"] = _sha(manifest)
    summary.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="elapsed duration"):
        BATCH.load_catalog(manifest, summary, raw_root)


def test_axis_review_gate_requires_every_anonymous_review_to_pass(tmp_path):
    public, queue, review = _axis_review(tmp_path)
    binding = BATCH.validate_axis_review(public, queue, review)
    assert binding["all_axis_reviews_passed"] is True
    assert binding["training_admission_granted"] is False

    public, queue, review = _axis_review(tmp_path / "failed", overall="fail")
    with pytest.raises(ValueError, match="did not pass"):
        BATCH.validate_axis_review(public, queue, review)


def test_output_validation_keeps_quality_pass_distinct_from_training_admission(
    tmp_path,
):
    job = {
        "episode_task_id": "natural-turn-001",
        "episode_task_record_sha256": "a" * 64,
        "source_bvh_sha256": "b" * 64,
        "start_frame": 17,
        "end_frame": 244,
    }
    paths = BATCH._artifact_paths(job["episode_task_id"], tmp_path)
    paths["raw_csv"].write_text("x\n", encoding="utf-8")
    paths["safe_csv"].write_text("x\n", encoding="utf-8")
    quality = {
        "artifact_kind": BATCH.RETARGET_ARTIFACT_KIND,
        "episode_task_id": job["episode_task_id"],
        "episode_task_record_sha256": job["episode_task_record_sha256"],
        "source_sha256": job["source_bvh_sha256"],
        "source_interval": {
            "start_frame": 17,
            "end_frame_exclusive": 244,
            "sample_span_sec": 7.533,
        },
        "temporal_selection": {"elapsed_time_cut_used": False},
        "license_gate": {"training_authorized": False},
        "quality_gate": {"passed": True, "joint_limits_pass": True},
        "accepted_for_training": False,
        "frames": 281,
        "retimed": True,
        "output_sample_span_sec": 9.333,
    }
    paths["quality_json"].write_text(json.dumps(quality), encoding="utf-8")
    result = BATCH.validate_output(job, tmp_path)
    assert result["status"] == "passed"
    assert result["source_frames"] == 227
    assert result["output_frames"] == 281
    assert result["safety_retimed"] is True
    assert result["accepted_for_training"] is False


def test_resume_recovers_hash_valid_existing_output_without_retargeting(
    tmp_path, monkeypatch
):
    job = {
        "episode_task_id": "natural-turn-001",
        "episode_task_record_sha256": "a" * 64,
    }
    catalog_binding = {"catalog": "bound"}
    axis_binding = {"axis": "passed"}
    output_root = tmp_path / "output"
    output_root.mkdir()
    state_path = output_root / "interact_native_bvh_full_retarget_v2.run_state.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "artifact_kind": "interact_native_bvh_full_physical_retarget_v2_run_state",
                "status": "running",
                "catalog_binding": catalog_binding,
                "axis_review_binding": axis_binding,
                "execution_task_count": 1,
                "accepted_for_training": False,
                "results": {},
            }
        ),
        encoding="utf-8",
    )
    recovered = {
        "episode_task_id": "natural-turn-001",
        "status": "passed",
        "accepted_for_training": False,
    }
    monkeypatch.setattr(
        BATCH, "load_catalog", lambda *_args: ([job], catalog_binding)
    )
    monkeypatch.setattr(
        BATCH, "validate_axis_review", lambda *_args: axis_binding
    )
    monkeypatch.setattr(BATCH, "validate_output", lambda *_args: recovered)

    class UnexpectedExecutor:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("valid existing output must not be retargeted")

    monkeypatch.setattr(BATCH, "ProcessPoolExecutor", UnexpectedExecutor)
    args = SimpleNamespace(
        workers=1,
        limit=None,
        warmup_source_frames=0,
        task_manifest=tmp_path / "tasks.jsonl",
        catalog_summary=tmp_path / "summary.json",
        raw_root=tmp_path / "raw",
        public_summary=tmp_path / "public.json",
        axis_queue=tmp_path / "axis.jsonl",
        axis_review=tmp_path / "axis_review.jsonl",
        output_root=output_root,
        resume=True,
        retry_failed=False,
        max_velocity=4.0,
        smoothing_window=5,
        posture_cost=0.1,
        solver="SLSQP",
        elbow_branch="auto",
    )

    state = BATCH.run(args)

    assert state["status"] == "complete"
    assert state["results"] == {"natural-turn-001": recovered}
    assert state["resume_recovered_existing_output_count"] == 1
