#!/usr/bin/env python3
"""Project the BEAT2 emotion controller queue into a blind visual-review bundle.

The source queue is controller-only even though it does not carry per-record
official emotion labels: trajectory paths and source-balancing metadata can
still reveal identity.  This tool therefore creates two deliberately separated
artifacts:

* a controller-only queue compatible with ``render_beat2_annotation_review``;
* reviewer-visible, anonymously named videos and two independent hash-bound
  primary-review shard sets, but only for supplied render-passed records.

No source label is copied into the reviewer-visible projection.  Third-reviewer
adjudication is not pre-populated: its queue must be created only for primary
disagreements, with an adjudicator distinct from both primary reviewers.
Nothing produced here enables training admission.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import secrets
import shutil
from typing import Any, Iterable, Mapping

from tools.human_motion_review.build_blind_review_shards_v1 import assign_shards
from tools.human_motion_review.render_beat2_annotation_review import (
    ROBOT_CONTRACT,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTROLLER_QUEUE = (
    PROJECT_ROOT
    / "deliverables/beat2_emotion_human_review_queue_v1/review_queue.jsonl"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "deliverables/beat2_emotion_human_review_queue_v1/blind_visual_review_v1"
)

SCHEMA_VERSION = "1.0.0"
SOURCE_KIND = "beat2_robot_observable_emotion_review_queue_record_v1"
RENDER_KIND = "beat2_robot_observable_emotion_render_projection_v1"
PUBLIC_TASK_KIND = "beat2_blind_robot_affect_primary_review_task_v1"
PROTOCOL = "two_independent_blind_robot_affect_reviews_then_adjudication_v1"
OBSERVABILITY = ("observable", "not_observable", "ambiguous")
EMOTIONS = ("neutral", "sad", "happy", "angry", "surprise", "fear")
PRIMARY_ROLES = ("primary_1", "primary_2")
FORBIDDEN_PATH_MARKERS = ("kimodo",)
FORBIDDEN_SOURCE_LABEL_KEYS = (
    "emotion_id",
    "source_emotion_label",
    "official_emotion",
    "official_emotion_id",
)
SHA256_LENGTH = 64

GENERIC_RENDER_PROMPT = {
    "en": "Observe this silent robot motion for a later blind affect review.",
    "zh": "观察这段无声机器人动作，随后进行盲法情绪判断。",
}

PUBLIC_TASK_KEYS = {
    "schema_version",
    "artifact_kind",
    "sample_id",
    "assignment_id",
    "assignment_role",
    "video_path",
    "video_sha256",
    "sample_binding_sha256",
    "review_protocol",
    "allowed_observability",
    "allowed_observed_emotions",
    "observability",
    "observed_emotion",
    "confidence",
    "reviewer_id",
    "submitted_at_utc",
    "paired_primary_reviewer_must_be_distinct",
    "third_adjudication_required_on_any_disagreement",
    "source_label_visible",
    "accepted_for_training",
}


def stable_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def value_sha256(value: object) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def atomic_text(path: Path, payload: str, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    if mode is not None:
        os.chmod(temporary, mode)
    os.replace(temporary, path)


def atomic_json(path: Path, value: object, *, mode: int | None = None) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        mode=mode,
    )


def atomic_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    mode: int | None = None,
) -> None:
    atomic_text(
        path,
        "".join(stable_json(dict(row)) + "\n" for row in rows),
        mode=mode,
    )


def read_jsonl_bound(path: Path) -> list[tuple[dict[str, Any], str]]:
    rows: list[tuple[dict[str, Any], str]] = []
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            payload = raw[:-1] if raw.endswith(b"\n") else raw
            if payload.endswith(b"\r"):
                raise ValueError(f"CRLF JSONL is not accepted: {path}:{line_number}")
            if not payload.strip():
                continue
            try:
                row = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid JSONL: {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"expected object: {path}:{line_number}")
            rows.append((row, hashlib.sha256(payload).hexdigest()))
    return rows


def _reject_forbidden_path(path: Path, *, field: str) -> None:
    folded = str(path).casefold()
    if any(marker in folded for marker in FORBIDDEN_PATH_MARKERS):
        raise ValueError(f"{field} contains a forbidden dataset marker")


def _source_label_paths(
    value: object,
    *,
    prefix: str = "$",
) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{prefix}.{key}"
            if str(key).casefold() in FORBIDDEN_SOURCE_LABEL_KEYS:
                found.append(child_path)
            found.extend(_source_label_paths(child, prefix=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_source_label_paths(child, prefix=f"{prefix}[{index}]"))
    return found


def _require_source_record(
    row: dict[str, Any],
    *,
    controller_queue: Path,
    line_sha256: str,
) -> tuple[str, Path]:
    sample_id = row.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id.startswith("beat2_affect_"):
        raise ValueError("controller row has an invalid sample_id")
    exact = {
        "artifact_kind": row.get("artifact_kind") == SOURCE_KIND,
        "dataset": row.get("dataset") == "BEAT2",
        "controller_only": row.get("controller_only_render_queue") is True,
        "reviewer_projection": row.get("reviewer_visible_projection_required") is True,
        "reviewer_separation": row.get("reviewers_must_not_receive_controller_queue")
        is True,
        "official_field_absent": row.get("official_emotion_field_present") is False,
        "official_not_exposed": row.get(
            "source_official_emotion_exposed_to_reviewers"
        )
        is False,
        "automated_label_absent": row.get("automated_emotion_label_assigned") is False,
        "emotion_mask": row.get("emotion_supervision_mask") is False,
        "affect_mask": row.get("affect_observable_supervision_mask") is False,
        "official_conditioning": row.get("official_emotion_conditioning_enabled")
        is False,
        "training_closed": row.get("accepted_for_training") is False,
        "protocol": row.get("review_protocol") == PROTOCOL,
        "primary_distinct": row.get("primary_reviewer_ids_must_be_distinct") is True,
        "primary_agreement": row.get("primary_agreement_required") is True,
        "third_required": row.get(
            "third_adjudication_required_on_any_primary_disagreement"
        )
        is True,
    }
    failed = sorted(key for key, passed in exact.items() if not passed)
    if failed:
        raise ValueError(f"{sample_id}: controller queue is not fail-closed: {failed}")
    present_source_labels = sorted(_source_label_paths(row))
    if present_source_labels:
        raise ValueError(
            f"{sample_id}: controller row contains source emotion fields: "
            f"{present_source_labels}"
        )
    if row.get("allowed_observability") != list(OBSERVABILITY):
        raise ValueError(f"{sample_id}: observability vocabulary changed")
    if row.get("allowed_observed_emotions") != list(EMOTIONS):
        raise ValueError(f"{sample_id}: observed-emotion vocabulary changed")

    primary = row.get("primary_reviews")
    if not isinstance(primary, list) or len(primary) != 2:
        raise ValueError(f"{sample_id}: exactly two primary review slots are required")
    expected_slots = ("reviewer_1", "reviewer_2")
    for slot, expected in zip(primary, expected_slots, strict=True):
        if (
            not isinstance(slot, dict)
            or slot.get("slot") != expected
            or slot.get("status") != "pending_independent_review"
            or any(
                slot.get(field) is not None
                for field in (
                    "reviewer_id",
                    "observability",
                    "observed_emotion",
                    "confidence",
                    "submitted_at_utc",
                )
            )
        ):
            raise ValueError(f"{sample_id}: primary review slot is not pristine")
    adjudication = row.get("third_adjudication")
    if (
        not isinstance(adjudication, dict)
        or adjudication.get("must_differ_from_primary_reviewers") is not True
        or adjudication.get("status")
        != "not_requested_pending_primary_reviews"
    ):
        raise ValueError(f"{sample_id}: third-adjudication contract changed")

    if not _is_sha256(line_sha256):
        raise AssertionError("raw controller-line binding is not SHA256")
    trajectory_value = row.get("trajectory_path")
    trajectory_sha256 = row.get("trajectory_sha256")
    if not isinstance(trajectory_value, str) or not _is_sha256(trajectory_sha256):
        raise ValueError(f"{sample_id}: invalid trajectory binding")
    trajectory = Path(trajectory_value)
    if not trajectory.is_absolute():
        trajectory = controller_queue.parent / trajectory
    trajectory = trajectory.resolve()
    _reject_forbidden_path(trajectory, field=f"{sample_id}.trajectory_path")
    if not trajectory.is_file() or sha256_file(trajectory) != trajectory_sha256:
        raise ValueError(f"{sample_id}: trajectory hash mismatch")

    fps = row.get("fps")
    duration = row.get("duration_sec")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isclose(float(fps), 30.0, rel_tol=0.0, abs_tol=1e-9)
        or isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or float(duration) <= 0.0
    ):
        raise ValueError(f"{sample_id}: invalid native duration")
    return sample_id, trajectory


def _render_record(
    source: dict[str, Any],
    *,
    trajectory: Path,
    line_sha256: str,
) -> dict[str, Any]:
    sample_id = str(source["sample_id"])
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": RENDER_KIND,
        "task_id": sample_id,
        "source_clip_id": sample_id,
        "speaker_key": source["speaker_group_token"],
        "official_split": source["fixed_split_assignment"],
        "robot_contract": ROBOT_CONTRACT,
        "canonical_action": "blind_robot_affect_observation",
        "canonical_prompt": dict(GENERIC_RENDER_PROMPT),
        "trajectory_path": str(trajectory),
        "trajectory_sha256": source["trajectory_sha256"],
        "retarget_segment": {
            "output_sample_span_sec": float(source["duration_sec"]),
            "duration_policy": "native_expression_turn_no_fixed_window",
        },
        "emotion_supervision_mask": False,
        "official_emotion_conditioning_enabled": False,
        "affect_observable_review_status": "candidate_unreviewed",
        "affect_observable_supervision_mask": False,
        "review_state": "pending_silent_video_render_for_blind_affect_review",
        "manual_review_required": True,
        "render_pass_grants_training_admission": False,
        "accepted_for_training": False,
        "controller_record_line_sha256": line_sha256,
    }


def _secret(hidden_root: Path, provided_hex: str | None) -> bytes:
    path = hidden_root / "bundle_secret.json"
    if path.is_file():
        record = json.loads(path.read_text(encoding="utf-8"))
        secret = bytes.fromhex(str(record.get("secret_hex") or ""))
        if provided_hex is not None and secret != bytes.fromhex(provided_hex):
            raise ValueError("provided secret does not match existing bundle secret")
        if len(secret) < 16:
            raise ValueError("stored bundle secret is invalid")
        os.chmod(path, 0o600)
        return secret
    secret = bytes.fromhex(provided_hex) if provided_hex is not None else secrets.token_bytes(32)
    if len(secret) < 16:
        raise ValueError("bundle secret must contain at least 16 bytes")
    atomic_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "beat2_emotion_blind_review_bundle_secret_v1",
            "secret_hex": secret.hex(),
            "reviewer_distribution_forbidden": True,
        },
        mode=0o600,
    )
    return secret


def _opaque_id(
    secret: bytes,
    *,
    prefix: str,
    parts: Iterable[str],
    length: int = 24,
) -> str:
    message = "\0".join(parts).encode("utf-8")
    return prefix + hmac.new(secret, message, hashlib.sha256).hexdigest()[:length]


def _materialize_video(source: Path, target: Path, expected_sha256: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or sha256_file(target) != expected_sha256:
            raise ValueError(f"existing anonymous video does not match: {target}")
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    if sha256_file(target) != expected_sha256:
        raise ValueError(f"anonymous video copy failed integrity check: {target}")


def _validate_render_pass(
    record: dict[str, Any],
    *,
    expected_render: dict[str, Any],
) -> tuple[Path, str]:
    sample_id = str(expected_render["task_id"])
    exact = {
        "task_id": record.get("task_id") == sample_id,
        "status": record.get("status") == "passed",
        "input_fingerprint": record.get("input_fingerprint")
        == value_sha256(expected_render),
        "training_closed": record.get("accepted_for_training") is False,
        "manual_review": record.get("manual_review_required") is True,
        "render_not_admission": record.get("render_pass_grants_training_admission")
        is False,
        "trajectory_sha256": record.get("trajectory_sha256")
        == expected_render["trajectory_sha256"],
        "emotion_mask": record.get("emotion_supervision_mask") is False,
        "official_conditioning": record.get("official_emotion_conditioning_enabled")
        is False,
        "affect_mask": record.get("affect_observable_supervision_mask") is False,
    }
    failed = sorted(key for key, passed in exact.items() if not passed)
    if failed:
        raise ValueError(f"{sample_id}: invalid render pass: {failed}")
    video_check = record.get("video_check")
    if (
        not isinstance(video_check, dict)
        or video_check.get("passed") is not True
        or video_check.get("audio_streams") != 0
    ):
        raise ValueError(f"{sample_id}: silent video verification did not pass")
    video_value = record.get("video_path")
    video_sha256 = record.get("video_sha256")
    if not isinstance(video_value, str) or not _is_sha256(video_sha256):
        raise ValueError(f"{sample_id}: invalid video binding")
    video = Path(video_value).resolve()
    _reject_forbidden_path(video, field=f"{sample_id}.video_path")
    if not video.is_file() or sha256_file(video) != video_sha256:
        raise ValueError(f"{sample_id}: rendered video hash mismatch")
    return video, str(video_sha256)


def _public_task(
    *,
    secret: bytes,
    sample_id: str,
    role: str,
    video_path: Path,
    video_sha256: str,
    controller_queue_sha256: str,
    controller_line_sha256: str,
    trajectory_sha256: str,
) -> dict[str, Any]:
    public_sample_id = _opaque_id(
        secret,
        prefix="motion_",
        parts=(
            "sample",
            controller_queue_sha256,
            sample_id,
            trajectory_sha256,
        ),
    )
    assignment_id = _opaque_id(
        secret,
        prefix="assignment_",
        parts=("assignment", role, public_sample_id, video_sha256),
    )
    binding = hmac.new(
        secret,
        (
            "binding\0"
            + controller_queue_sha256
            + "\0"
            + controller_line_sha256
            + "\0"
            + trajectory_sha256
            + "\0"
            + video_sha256
            + "\0"
            + public_sample_id
        ).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    record = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": PUBLIC_TASK_KIND,
        "sample_id": public_sample_id,
        "assignment_id": assignment_id,
        "assignment_role": role,
        "video_path": str(video_path),
        "video_sha256": video_sha256,
        "sample_binding_sha256": binding,
        "review_protocol": PROTOCOL,
        "allowed_observability": list(OBSERVABILITY),
        "allowed_observed_emotions": list(EMOTIONS),
        "observability": None,
        "observed_emotion": None,
        "confidence": None,
        "reviewer_id": None,
        "submitted_at_utc": None,
        "paired_primary_reviewer_must_be_distinct": True,
        "third_adjudication_required_on_any_disagreement": True,
        "source_label_visible": False,
        "accepted_for_training": False,
    }
    if set(record) != PUBLIC_TASK_KEYS:
        raise AssertionError("reviewer-visible task allowlist changed")
    return record


def _write_role_shards(
    *,
    role: str,
    rows: list[dict[str, Any]],
    reviewer_root: Path,
    shard_count: int,
) -> tuple[dict[str, Any], dict[str, int]]:
    role_root = reviewer_root / role
    full_queue = role_root / "full_review_queue.jsonl"
    atomic_jsonl(full_queue, rows)
    queue_sha256 = sha256_file(full_queue)
    shards = assign_shards(
        rows,
        queue_sha256=queue_sha256,
        shard_count=shard_count,
    )
    sample_shards: dict[str, int] = {}
    summaries: list[dict[str, Any]] = []
    for index, shard_rows in enumerate(shards):
        shard_root = role_root / f"shard_{index:03d}"
        queue = shard_root / "review_queue.jsonl"
        atomic_jsonl(queue, shard_rows)
        for row in shard_rows:
            sample_shards[str(row["sample_id"])] = index
        summary = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "beat2_blind_robot_affect_primary_review_shard_v1",
            "assignment_role": role,
            "shard_index": index,
            "shard_count": shard_count,
            "records": len(shard_rows),
            "review_queue": str(queue),
            "review_queue_sha256": sha256_file(queue),
            "one_reviewer_per_shard": True,
            "paired_primary_reviewer_must_be_distinct": True,
            "source_labels_included": False,
            "accepted_for_training": False,
        }
        atomic_json(shard_root / "summary.json", summary)
        summaries.append(summary)
    role_summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "beat2_blind_robot_affect_primary_review_shards_v1",
        "assignment_role": role,
        "records": len(rows),
        "shard_count": shard_count,
        "minimum_shard_records": min((len(shard) for shard in shards), default=0),
        "maximum_shard_records": max((len(shard) for shard in shards), default=0),
        "coverage_complete_without_overlap": (
            len(sample_shards) == len(rows)
            and sum(len(shard) for shard in shards) == len(rows)
        ),
        "full_review_queue": str(full_queue),
        "full_review_queue_sha256": queue_sha256,
        "shards": summaries,
        "source_labels_included": False,
        "accepted_for_training": False,
    }
    atomic_json(role_root / "summary.json", role_summary)
    return role_summary, sample_shards


def _write_reviewer_schema(reviewer_root: Path) -> None:
    atomic_json(
        reviewer_root / "review_schema.json",
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "beat2_blind_robot_affect_review_schema_v1",
            "review_protocol": PROTOCOL,
            "allowed_observability": list(OBSERVABILITY),
            "allowed_observed_emotions": list(EMOTIONS),
            "required_submission_fields": [
                "sample_id",
                "assignment_id",
                "video_sha256",
                "sample_binding_sha256",
                "observability",
                "observed_emotion",
                "confidence",
                "reviewer_id",
                "submitted_at_utc",
            ],
            "observed_emotion_policy": (
                "required_only_when_observability_is_observable_otherwise_null"
            ),
            "confidence_policy": "number_in_closed_interval_0_to_1",
            "primary_review_count_per_sample": 2,
            "primary_reviewer_ids_must_be_distinct": True,
            "third_adjudication_policy": (
                "controller_creates_a_label_blind_queue_only_for_any_primary_"
                "disagreement_and_requires_a_reviewer_distinct_from_both_primaries"
            ),
            "primary_decisions_visible_to_third_adjudicator": False,
            "source_labels_visible": False,
            "source_identity_visible": False,
            "accepted_for_training": False,
        },
    )
    pending = reviewer_root / "adjudication/pending_queue.jsonl"
    atomic_jsonl(pending, [])
    atomic_json(
        pending.parent / "summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "beat2_blind_robot_affect_adjudication_pending_v1",
            "records": 0,
            "state": "not_materialized_until_primary_disagreement",
            "adjudicator_must_differ_from_both_primary_reviewers": True,
            "source_labels_included": False,
            "primary_decisions_included": False,
            "accepted_for_training": False,
        },
    )


def build_bundle(
    *,
    controller_queue: Path,
    output_root: Path,
    render_passed_manifests: list[Path] | None = None,
    hidden_root: Path | None = None,
    shard_count: int = 6,
    secret_hex: str | None = None,
) -> dict[str, Any]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    controller_queue = controller_queue.resolve()
    output_root = output_root.resolve()
    hidden_root = (
        hidden_root.resolve()
        if hidden_root is not None
        else output_root / "controller_only"
    )
    reviewer_root = output_root / "reviewer_visible"
    _reject_forbidden_path(controller_queue, field="controller_queue")
    _reject_forbidden_path(output_root, field="output_root")
    _reject_forbidden_path(hidden_root, field="hidden_root")
    if not controller_queue.is_file():
        raise FileNotFoundError(controller_queue)
    if hidden_root == reviewer_root or reviewer_root in hidden_root.parents:
        raise ValueError("controller-only root must not be inside reviewer-visible root")
    hidden_root.mkdir(parents=True, exist_ok=True)
    os.chmod(hidden_root, 0o700)
    reviewer_root.mkdir(parents=True, exist_ok=True)
    secret = _secret(hidden_root, secret_hex)

    controller_queue_sha256 = sha256_file(controller_queue)
    source_rows = read_jsonl_bound(controller_queue)
    if not source_rows:
        raise ValueError("controller queue is empty")
    render_rows: list[dict[str, Any]] = []
    source_by_id: dict[str, tuple[dict[str, Any], str]] = {}
    trajectory_hashes: set[str] = set()
    for source, line_sha256 in source_rows:
        sample_id, trajectory = _require_source_record(
            source,
            controller_queue=controller_queue,
            line_sha256=line_sha256,
        )
        if sample_id in source_by_id:
            raise ValueError(f"duplicate controller sample_id: {sample_id}")
        trajectory_sha256 = str(source["trajectory_sha256"])
        if trajectory_sha256 in trajectory_hashes:
            raise ValueError("controller queue contains duplicate trajectory content")
        trajectory_hashes.add(trajectory_sha256)
        source_by_id[sample_id] = (source, line_sha256)
        render_rows.append(
            _render_record(
                source,
                trajectory=trajectory,
                line_sha256=line_sha256,
            )
        )
    render_rows.sort(key=lambda row: str(row["task_id"]))
    render_queue = hidden_root / "render_queue.jsonl"
    atomic_jsonl(render_queue, render_rows, mode=0o600)
    render_by_id = {str(row["task_id"]): row for row in render_rows}

    passed_by_id: dict[str, dict[str, Any]] = {}
    render_passed_manifests = render_passed_manifests or []
    for manifest in render_passed_manifests:
        resolved = manifest.resolve()
        _reject_forbidden_path(resolved, field="render_passed_manifest")
        for record, _line_sha256 in read_jsonl_bound(resolved):
            sample_id = str(record.get("task_id") or "")
            if sample_id not in render_by_id:
                raise ValueError(f"render pass is not in controller queue: {sample_id}")
            if sample_id in passed_by_id:
                raise ValueError(f"duplicate render pass: {sample_id}")
            passed_by_id[sample_id] = record

    public_rows_by_role: dict[str, list[dict[str, Any]]] = {
        role: [] for role in PRIMARY_ROLES
    }
    mapping_rows: list[dict[str, Any]] = []
    rendered_bindings: dict[str, dict[str, Any]] = {}
    for sample_id in sorted(passed_by_id):
        source, line_sha256 = source_by_id[sample_id]
        render_record = render_by_id[sample_id]
        passed = passed_by_id[sample_id]
        video, video_sha256 = _validate_render_pass(
            passed,
            expected_render=render_record,
        )
        public_sample_id = _opaque_id(
            secret,
            prefix="motion_",
            parts=(
                "sample",
                controller_queue_sha256,
                sample_id,
                str(source["trajectory_sha256"]),
            ),
        )
        anonymous_video = reviewer_root / "videos" / f"{public_sample_id}.mp4"
        _materialize_video(video, anonymous_video, video_sha256)
        for role in PRIMARY_ROLES:
            public_rows_by_role[role].append(
                _public_task(
                    secret=secret,
                    sample_id=sample_id,
                    role=role,
                    video_path=anonymous_video,
                    video_sha256=video_sha256,
                    controller_queue_sha256=controller_queue_sha256,
                    controller_line_sha256=line_sha256,
                    trajectory_sha256=str(source["trajectory_sha256"]),
                )
            )
        rendered_bindings[public_sample_id] = {
            "controller_sample_id": sample_id,
            "controller_record_line_sha256": line_sha256,
            "trajectory_sha256": source["trajectory_sha256"],
            "render_input_record_sha256": value_sha256(render_record),
            "render_pass_record_sha256": value_sha256(passed),
            "source_video_path": str(video),
            "anonymous_video_path": str(anonymous_video),
            "video_sha256": video_sha256,
        }
    expected_public_videos = {
        (reviewer_root / "videos" / f"{sample_id}.mp4").resolve()
        for sample_id in rendered_bindings
    }
    existing_public_videos = {
        path.resolve()
        for path in (reviewer_root / "videos").glob("*.mp4")
        if path.is_file()
    }
    stale_public_videos = sorted(existing_public_videos - expected_public_videos)
    if stale_public_videos:
        raise ValueError(
            "reviewer-visible root contains stale videos; use a fresh output root "
            "or provide a cumulative render-passed manifest"
        )

    _write_reviewer_schema(reviewer_root)
    role_summaries: dict[str, dict[str, Any]] = {}
    role_shards: dict[str, dict[str, int]] = {}
    for role in PRIMARY_ROLES:
        rows = sorted(
            public_rows_by_role[role],
            key=lambda row: str(row["sample_id"]),
        )
        summary, shards = _write_role_shards(
            role=role,
            rows=rows,
            reviewer_root=reviewer_root,
            shard_count=shard_count,
        )
        role_summaries[role] = summary
        role_shards[role] = shards

    for public_sample_id in sorted(rendered_bindings):
        binding = rendered_bindings[public_sample_id]
        role_rows = {
            role: next(
                row
                for row in public_rows_by_role[role]
                if row["sample_id"] == public_sample_id
            )
            for role in PRIMARY_ROLES
        }
        mapping_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_kind": "beat2_emotion_blind_review_hidden_mapping_v1",
                "public_sample_id": public_sample_id,
                **binding,
                "primary_assignments": {
                    role: {
                        "assignment_id": role_rows[role]["assignment_id"],
                        "sample_binding_sha256": role_rows[role][
                            "sample_binding_sha256"
                        ],
                        "shard_index": role_shards[role][public_sample_id],
                    }
                    for role in PRIMARY_ROLES
                },
                "primary_reviewer_ids_must_be_distinct": True,
                "third_adjudication_required_on_any_primary_disagreement": True,
                "third_adjudicator_must_differ_from_primary_reviewers": True,
                "accepted_for_training": False,
            }
        )
    mapping = hidden_root / "sample_mapping.jsonl"
    atomic_jsonl(mapping, mapping_rows, mode=0o600)

    hidden_summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "beat2_emotion_blind_review_controller_bundle_v1",
        "controller_queue": str(controller_queue),
        "controller_queue_sha256": controller_queue_sha256,
        "controller_records": len(source_rows),
        "render_queue": str(render_queue),
        "render_queue_sha256": sha256_file(render_queue),
        "render_passed_manifests": [
            str(path.resolve()) for path in render_passed_manifests
        ],
        "render_passed_manifest_sha256": [
            sha256_file(path.resolve()) for path in render_passed_manifests
        ],
        "rendered_records": len(passed_by_id),
        "mapping": str(mapping),
        "mapping_sha256": sha256_file(mapping),
        "reviewer_distribution_forbidden": True,
        "source_official_emotion_fields_available": False,
        "accepted_for_training": False,
    }
    atomic_json(hidden_root / "summary.json", hidden_summary, mode=0o600)

    public_summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "beat2_emotion_blind_visual_review_bundle_v1",
        "rendered_records": len(passed_by_id),
        "primary_review_tasks": {
            role: len(public_rows_by_role[role]) for role in PRIMARY_ROLES
        },
        "primary_roles": role_summaries,
        "every_rendered_sample_assigned_to_two_primary_reviews": all(
            len(public_rows_by_role[role]) == len(passed_by_id)
            for role in PRIMARY_ROLES
        ),
        "primary_reviewer_ids_must_be_distinct": True,
        "third_adjudication_required_on_any_primary_disagreement": True,
        "third_adjudicator_must_differ_from_primary_reviewers": True,
        "third_adjudication_queue_state": (
            "not_materialized_until_primary_disagreement"
        ),
        "source_labels_included": False,
        "source_identity_included": False,
        "controller_queue_included": False,
        "review_schema": str(reviewer_root / "review_schema.json"),
        "review_schema_sha256": sha256_file(reviewer_root / "review_schema.json"),
        "accepted_for_training": False,
    }
    atomic_json(reviewer_root / "summary.json", public_summary)
    return {
        "controller_only": hidden_summary,
        "reviewer_visible": public_summary,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller-queue",
        type=Path,
        default=DEFAULT_CONTROLLER_QUEUE,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--hidden-root",
        type=Path,
        help="Store controller files outside the reviewer-visible bundle root.",
    )
    parser.add_argument(
        "--render-passed-manifest",
        action="append",
        type=Path,
        default=[],
        help="Verified output from render_beat2_annotation_review.py; repeatable.",
    )
    parser.add_argument("--shard-count", type=int, default=6)
    parser.add_argument("--secret-hex")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_bundle(
        controller_queue=args.controller_queue,
        output_root=args.output_root,
        render_passed_manifests=args.render_passed_manifest,
        hidden_root=args.hidden_root,
        shard_count=args.shard_count,
        secret_hex=args.secret_hex,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
