#!/usr/bin/env python3
"""Build fail-closed round-N InterAct natural-boundary continuation plans."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from tools.human_motion_review.build_interact_arc_expansion_plan_v2 import (
        atomic_json,
        atomic_jsonl,
        read_json,
        sha256_file,
        value_sha256,
    )
except ModuleNotFoundError:  # pragma: no cover - direct invocation by path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.human_motion_review.build_interact_arc_expansion_plan_v2 import (
        atomic_json,
        atomic_jsonl,
        read_json,
        sha256_file,
        value_sha256,
    )


SCHEMA_VERSION = "2.0.0"
ARC_PROTOCOL = "interact_dyadic_arc_action_blind_video_native_bvh_v2"
PLAN_KIND = "interact_dyadic_arc_action_one_level_expansion_plan_v2"
REQUEST_KIND = "interact_dyadic_one_level_natural_context_expansion_request_v2"
COMPLETE_KIND = "interact_dyadic_current_context_arc_action_complete_v2"
PUBLIC_KIND = "interact_dyadic_natural_context_expansion_anonymous_bundle_v2"
EXPANSION_HIDDEN_KIND = "interact_dyadic_natural_context_expansion_hidden_mapping_v2"
REVIEW_SUMMARY_KIND = "interact_dyadic_arc_action_blind_review_submission_v2"
COMPLETE = "complete"
INCOMPLETE = "incomplete_requires_expansion"
REQUESTED_BOUNDARY = "next_predeclared_natural_boundary"
EXPANSION_UNIT = "exactly_one_next_predeclared_shared_rest_boundary_level"
SELECTION_POLICY = "exactly_one_next_predeclared_shared_rest_boundary_level"
PUBLIC_DURATION_POLICY = "one_predeclared_shared_rest_boundary_level_no_fixed_window"
ORIGINAL_CONTEXT_DURATION_POLICY = (
    "semantic_affect_complete_at_predeclared_shared_rest_boundaries;"
    "no_fixed_target_minimum_or_maximum_duration"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-summary", type=Path, required=True)
    parser.add_argument("--arc-review-submission", type=Path, required=True)
    parser.add_argument("--arc-review-summary", type=Path, required=True)
    parser.add_argument("--expansion-hidden-mapping", type=Path, required=True)
    parser.add_argument("--expansion-hidden-summary", type=Path, required=True)
    parser.add_argument("--original-dyad-mapping", type=Path, required=True)
    parser.add_argument("--previous-plan-summary", type=Path, required=True)
    parser.add_argument("--previous-expansion-requests", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def _read_bound_jsonl(path: Path) -> list[tuple[dict[str, Any], str]]:
    rows: list[tuple[dict[str, Any], str]] = []
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            raw_record = raw_line[:-1] if raw_line.endswith(b"\n") else raw_line
            if raw_record.endswith(b"\r"):
                raise ValueError(f"CRLF JSONL is not accepted: {path}:{line_number}")
            if not raw_record.strip():
                continue
            try:
                value = json.loads(raw_record.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid UTF-8 JSON at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append((value, hashlib.sha256(raw_record).hexdigest()))
    return rows


def _index_bound_rows(
    rows: list[tuple[dict[str, Any], str]],
    key: str,
    *,
    context: str,
) -> dict[str, tuple[dict[str, Any], str]]:
    result: dict[str, tuple[dict[str, Any], str]] = {}
    for row, line_sha256 in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value or value in result:
            raise ValueError(f"{context} contains invalid or duplicate {key}")
        result[value] = (row, line_sha256)
    return result


def _index_rows(
    rows: list[tuple[dict[str, Any], str]],
    key: str,
    *,
    context: str,
) -> dict[str, dict[str, Any]]:
    return {
        value: row
        for value, (row, _line_sha256) in _index_bound_rows(
            rows, key, context=context
        ).items()
    }


def _declared_path(owner: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing declared path: {field}")
    path = Path(value)
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve()


def _require_declared_file(
    owner: Path,
    metadata: dict[str, Any],
    *,
    path_field: str,
    sha_field: str,
    expected: Path,
) -> None:
    declared = _declared_path(owner, metadata.get(path_field), field=path_field)
    if declared != expected.resolve():
        raise ValueError(f"{path_field} does not resolve to the supplied input")
    if metadata.get(sha_field) != sha256_file(expected):
        raise ValueError(f"{sha_field} does not bind the supplied input")


def _interval(value: Any, *, context: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} is not an interval object")
    start = value.get("start_frame")
    end = value.get("end_frame_exclusive")
    inferred_frames = (
        end - start if isinstance(start, int) and isinstance(end, int) else None
    )
    frames = value.get("frame_count", inferred_frames)
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or not isinstance(frames, int)
        or isinstance(frames, bool)
        or start < 0
        or end <= start
        or frames != end - start
    ):
        raise ValueError(f"{context} is invalid")
    return {"start_frame": start, "end_frame_exclusive": end, "frame_count": frames}


def _strictly_expands(current: dict[str, int], requested: dict[str, int]) -> bool:
    return (
        requested["start_frame"] <= current["start_frame"]
        and requested["end_frame_exclusive"] >= current["end_frame_exclusive"]
        and requested != current
    )


def _context_levels(mapping: dict[str, Any]) -> dict[int, dict[str, int]]:
    sample_id = mapping.get("sample_id")
    context = mapping.get("context_plan")
    if not isinstance(context, dict):
        raise ValueError(f"Original dyad mapping lacks context plan: {sample_id}")
    if context.get("duration_gate_used") is not False:
        raise ValueError(f"Original context plan used a duration gate: {sample_id}")
    duration_policy = context.get("duration_policy")
    if duration_policy not in (None, ORIGINAL_CONTEXT_DURATION_POLICY):
        raise ValueError(f"Original context plan changed duration policy: {sample_id}")
    completeness = context.get("completeness_review") or {}
    if completeness.get("elapsed_seconds_may_influence_decision") not in (None, False):
        raise ValueError(f"Original context plan permits elapsed-time decisions: {sample_id}")
    raw_levels = context.get("levels")
    if not isinstance(raw_levels, list) or not raw_levels:
        raise ValueError(f"Original context plan has no levels: {sample_id}")
    levels: dict[int, dict[str, int]] = {}
    previous: dict[str, int] | None = None
    for expected_level, raw_level in enumerate(raw_levels):
        if not isinstance(raw_level, dict) or raw_level.get("level") != expected_level:
            raise ValueError(f"Context levels are not contiguous from zero: {sample_id}")
        interval = _interval(raw_level, context=f"context level {expected_level} for {sample_id}")
        if previous is not None and not _strictly_expands(previous, interval):
            raise ValueError(f"Context level does not strictly expand its predecessor: {sample_id}")
        levels[expected_level] = interval
        previous = interval
    return levels


def _actor_mapping(mapping: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sample_id = mapping.get("sample_id")
    actors = mapping.get("actor_mapping")
    if not isinstance(actors, dict) or set(actors) != {"A", "B"}:
        raise ValueError(f"Original dyad mapping has incomplete actor roles: {sample_id}")
    result: dict[str, dict[str, Any]] = {}
    for role in ("A", "B"):
        actor = actors[role]
        if not isinstance(actor, dict):
            raise ValueError(f"Actor {role} mapping is invalid: {sample_id}")
        for field in ("episode_task_id", "actor_id", "partner_actor_id"):
            if not isinstance(actor.get(field), str) or not actor[field]:
                raise ValueError(f"Actor {role} mapping lacks {field}: {sample_id}")
        result[role] = deepcopy(actor)
    if (
        result["A"]["actor_id"] == result["B"]["actor_id"]
        or result["A"]["partner_actor_id"] != result["B"]["actor_id"]
        or result["B"]["partner_actor_id"] != result["A"]["actor_id"]
    ):
        raise ValueError(f"Actor partner mapping is inconsistent: {sample_id}")
    return result


def _verify_previous_request(row: dict[str, Any]) -> None:
    sample_id = row.get("sample_id")
    expected = row.get("plan_record_sha256")
    payload = dict(row)
    payload.pop("plan_record_sha256", None)
    if expected != value_sha256(payload):
        raise ValueError(f"Previous plan record SHA mismatch: {sample_id}")
    if row.get("artifact_kind") != REQUEST_KIND:
        raise ValueError(f"Unexpected previous request kind: {sample_id}")
    if (
        row.get("accepted_for_training") is not False
        or row.get("semantic_supervision_mask") is not False
        or row.get("emotion_supervision_mask") is not False
        or row.get("elapsed_duration_used_as_gate") is not False
        or row.get("expansion_unit") != EXPANSION_UNIT
    ):
        raise ValueError(f"Previous request violates fail-closed policy: {sample_id}")
    current_level = row.get("reviewed_context_level")
    next_level = row.get("requested_context_level")
    if (
        not isinstance(current_level, int)
        or isinstance(current_level, bool)
        or next_level != current_level + 1
    ):
        raise ValueError(f"Previous request skips a context level: {sample_id}")
    current = _interval(
        row.get("reviewed_interval"), context=f"previous reviewed interval {sample_id}"
    )
    requested = _interval(
        row.get("requested_interval"), context=f"previous requested interval {sample_id}"
    )
    if not _strictly_expands(current, requested):
        raise ValueError(f"Previous request is not a strict natural expansion: {sample_id}")


def _verify_review_summary(
    review_summary_path: Path,
    summary: dict[str, Any],
    *,
    review_submission: Path,
    public_summary_sha256: str,
    queue_sha256: str,
    review_count: int,
) -> None:
    if summary.get("artifact_kind") != REVIEW_SUMMARY_KIND:
        raise ValueError("Unexpected arc/action review summary kind")
    if (
        summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("accepted_for_training") is not False
        or summary.get("fixed_duration_window_used") is not False
        or summary.get("native_duration_preserved") is not True
        or summary.get("record_count") != review_count
    ):
        raise ValueError("Arc/action review summary violates the fail-closed contract")
    actual_submission_sha256 = sha256_file(review_submission)
    declared_hashes = {
        value
        for field in ("output_jsonl_sha256", "submission_jsonl_sha256")
        if isinstance((value := summary.get(field)), str)
    }
    if declared_hashes != {actual_submission_sha256}:
        raise ValueError("Arc/action review summary submission SHA mismatch")
    bindings = summary.get("input_bindings")
    if not isinstance(bindings, dict):
        raise ValueError("Arc/action review summary lacks input bindings")
    if (
        bindings.get("public_summary_sha256") != public_summary_sha256
        or bindings.get("arc_action_queue_sha256") != queue_sha256
    ):
        raise ValueError("Arc/action review summary input binding mismatch")
    if review_summary_path.resolve() == review_submission.resolve():
        raise ValueError("Review summary and submission must be separate artifacts")


def _verify_review(
    review: dict[str, Any],
    *,
    queued: dict[str, Any],
    queue_record_sha256: str,
    public_summary_sha256: str,
    queue_sha256: str,
    mapping: dict[str, Any],
    displayed_interval: dict[str, int],
) -> dict[str, dict[str, Any]]:
    sample_id = review.get("sample_id")
    if (
        review.get("schema_version") != SCHEMA_VERSION
        or review.get("protocol_version") != ARC_PROTOCOL
        or review.get("temporal_unit") != "complete_natural_interaction_arc"
        or review.get("accepted_for_training") is not False
        or review.get("fixed_duration_window_used") is not False
        or review.get("native_duration_preserved") is not True
    ):
        raise ValueError(f"Review violates the arc/action protocol: {sample_id}")
    if (
        review.get("video_sha256") != queued.get("video_sha256")
        or review.get("video_path") != queued.get("video_path")
        or review.get("video_sha256") != mapping.get("video_sha256")
    ):
        raise ValueError(f"Review video binding mismatch: {sample_id}")
    current_level = mapping.get("displayed_context_level")
    if review.get("context_level") != current_level or queued.get("context_level") != current_level:
        raise ValueError(f"Reviewed context level mismatch: {sample_id}")
    provenance = review.get("blind_review_provenance")
    if not isinstance(provenance, dict) or (
        provenance.get("public_summary_sha256") != public_summary_sha256
        or provenance.get("arc_action_queue_sha256") != queue_sha256
        or provenance.get("queue_record_hash_method") != "sha256_utf8_line_without_lf"
        or provenance.get("queue_record_sha256") != queue_record_sha256
    ):
        raise ValueError(f"Review provenance binding mismatch: {sample_id}")
    decode = review.get("decode_validation")
    if not isinstance(decode, dict) or (
        decode.get("complete") is not True
        or decode.get("video_sha256_verified") is not True
        or not isinstance(decode.get("decoded_frame_count"), int)
        or isinstance(decode.get("decoded_frame_count"), bool)
        or decode.get("decoded_frame_count") <= 0
        or decode.get("decoded_frame_count") != decode.get("reported_frame_count")
        or decode.get("decoded_frame_count") != displayed_interval["frame_count"]
    ):
        raise ValueError(f"Review decode/frame binding mismatch: {sample_id}")
    evidence: dict[str, dict[str, Any]] = {}
    for phase in ("onset", "apex", "offset"):
        local_frame = review.get(f"{phase}_evidence_frame")
        status = review.get(f"{phase}_status")
        if (
            not isinstance(local_frame, int)
            or isinstance(local_frame, bool)
            or not 0 <= local_frame < decode["decoded_frame_count"]
            or status not in {"complete", "incomplete"}
        ):
            raise ValueError(f"Invalid {phase} evidence for {sample_id}")
        evidence[phase] = {
            "status": status,
            "local_frame": local_frame,
            "source_frame": displayed_interval["start_frame"] + local_frame,
        }
    return evidence


def _verify_public_queue_record(
    queued: dict[str, Any],
    *,
    public_root: Path,
) -> None:
    sample_id = queued.get("sample_id")
    if (
        queued.get("schema_version") != SCHEMA_VERSION
        or queued.get("protocol_version") != ARC_PROTOCOL
        or queued.get("temporal_unit") != "complete_natural_interaction_arc"
        or queued.get("accepted_for_training") is not False
        or queued.get("fixed_duration_window_used") is not False
        or queued.get("native_duration_preserved") is not True
        or queued.get("label_metadata_exposed") is not False
    ):
        raise ValueError(f"Public queue record violates protocol: {sample_id}")
    video_path = _declared_path(
        public_root / "summary.json", queued.get("video_path"), field="video_path"
    )
    try:
        video_path.relative_to(public_root.resolve())
    except ValueError as error:
        raise ValueError(f"Public video escapes the public bundle: {sample_id}") from error
    video_sha256 = queued.get("video_sha256")
    if not isinstance(video_sha256, str) or sha256_file(video_path) != video_sha256:
        raise ValueError(f"Public video SHA mismatch: {sample_id}")


def build_plan(
    *,
    public_summary: Path,
    arc_review_submission: Path,
    arc_review_summary: Path,
    expansion_hidden_mapping: Path,
    expansion_hidden_summary: Path,
    original_dyad_mapping: Path,
    previous_plan_summary: Path,
    previous_expansion_requests: Path,
    output_root: Path,
) -> dict[str, Any]:
    public_summary = public_summary.resolve()
    arc_review_submission = arc_review_submission.resolve()
    arc_review_summary = arc_review_summary.resolve()
    expansion_hidden_mapping = expansion_hidden_mapping.resolve()
    expansion_hidden_summary = expansion_hidden_summary.resolve()
    original_dyad_mapping = original_dyad_mapping.resolve()
    previous_plan_summary = previous_plan_summary.resolve()
    previous_expansion_requests = previous_expansion_requests.resolve()

    input_sha256 = {
        "public_summary_sha256": sha256_file(public_summary),
        "review_submission_sha256": sha256_file(arc_review_submission),
        "review_summary_sha256": sha256_file(arc_review_summary),
        "expansion_hidden_mapping_sha256": sha256_file(expansion_hidden_mapping),
        "expansion_hidden_summary_sha256": sha256_file(expansion_hidden_summary),
        "hidden_mapping_sha256": sha256_file(original_dyad_mapping),
        "original_dyad_mapping_sha256": sha256_file(original_dyad_mapping),
        "previous_plan_summary_sha256": sha256_file(previous_plan_summary),
        "previous_expansion_requests_sha256": sha256_file(previous_expansion_requests),
    }
    public = read_json(public_summary)
    if (
        public.get("schema_version") != SCHEMA_VERSION
        or public.get("artifact_kind") != PUBLIC_KIND
        or public.get("duration_policy") != PUBLIC_DURATION_POLICY
        or public.get("fixed_duration_window_used") is not False
        or public.get("accepted_for_training") is not False
        or public.get("identity_scenario_official_text_or_emotion_exposed") is not False
        or public.get("plan_summary_sha256") != input_sha256["previous_plan_summary_sha256"]
    ):
        raise ValueError("Unexpected or unbound expansion public summary")
    queue_path = _declared_path(
        public_summary, public.get("arc_action_queue"), field="arc_action_queue"
    )
    if queue_path.parent != public_summary.parent.resolve():
        raise ValueError("Arc/action queue is outside the public bundle")
    queue_sha256 = sha256_file(queue_path)
    if public.get("arc_action_queue_sha256") != queue_sha256:
        raise ValueError("Expansion public arc/action queue SHA mismatch")
    input_sha256["arc_action_queue_sha256"] = queue_sha256

    previous_plan = read_json(previous_plan_summary)
    if (
        previous_plan.get("schema_version") != SCHEMA_VERSION
        or previous_plan.get("artifact_kind") != PLAN_KIND
        or previous_plan.get("accepted_for_training_count") != 0
        or previous_plan.get("selection_policy") != SELECTION_POLICY
        or previous_plan.get("fixed_minimum_maximum_or_target_duration_used") is not False
        or (previous_plan.get("inputs") or {}).get("hidden_mapping_sha256")
        != input_sha256["original_dyad_mapping_sha256"]
    ):
        raise ValueError("Unexpected or unbound previous InterAct expansion plan")
    previous_declared = (previous_plan.get("outputs") or {}).get("expansion_requests")
    if not isinstance(previous_declared, dict):
        raise ValueError("Previous plan does not declare expansion requests")
    if (
        previous_declared.get("sha256")
        != input_sha256["previous_expansion_requests_sha256"]
        or _declared_path(
            previous_plan_summary,
            previous_declared.get("path"),
            field="outputs.expansion_requests.path",
        )
        != previous_expansion_requests
    ):
        raise ValueError("Previous plan expansion request binding mismatch")
    previous_requests = _index_rows(
        _read_bound_jsonl(previous_expansion_requests),
        "sample_id",
        context="previous expansion requests",
    )
    if previous_declared.get("records") != len(previous_requests):
        raise ValueError("Previous expansion request count mismatch")
    for request in previous_requests.values():
        _verify_previous_request(request)

    hidden_summary = read_json(expansion_hidden_summary)
    if (
        hidden_summary.get("schema_version") != SCHEMA_VERSION
        or hidden_summary.get("artifact_kind") != EXPANSION_HIDDEN_KIND
        or hidden_summary.get("accepted_for_training") is not False
    ):
        raise ValueError("Unexpected expansion hidden summary")
    if _declared_path(
        expansion_hidden_summary,
        hidden_summary.get("public_summary"),
        field="public_summary",
    ) != public_summary:
        raise ValueError("Expansion hidden summary public path mismatch")
    _require_declared_file(
        expansion_hidden_summary,
        hidden_summary,
        path_field="plan_summary",
        sha_field="plan_summary_sha256",
        expected=previous_plan_summary,
    )
    _require_declared_file(
        expansion_hidden_summary,
        hidden_summary,
        path_field="expansion_requests",
        sha_field="expansion_requests_sha256",
        expected=previous_expansion_requests,
    )
    _require_declared_file(
        expansion_hidden_summary,
        hidden_summary,
        path_field="sample_mapping",
        sha_field="sample_mapping_sha256",
        expected=expansion_hidden_mapping,
    )
    if hidden_summary.get("run_state_sha256") != public.get("run_state_sha256"):
        raise ValueError("Expansion public and hidden run-state bindings differ")

    queue_bound = _index_bound_rows(
        _read_bound_jsonl(queue_path), "sample_id", context="public arc/action queue"
    )
    reviews = _index_rows(
        _read_bound_jsonl(arc_review_submission),
        "sample_id",
        context="arc/action review submission",
    )
    expansion_hidden = _index_rows(
        _read_bound_jsonl(expansion_hidden_mapping),
        "sample_id",
        context="expansion hidden mapping",
    )
    original_mapping = _index_rows(
        _read_bound_jsonl(original_dyad_mapping),
        "sample_id",
        context="original dyad mapping",
    )
    queue_ids = set(queue_bound)
    if queue_ids != set(reviews) or queue_ids != set(expansion_hidden):
        raise ValueError("Public queue, reviews, and expansion hidden mapping sample sets differ")
    if public.get("arc_action_records") != len(queue_ids):
        raise ValueError("Expansion public arc/action record count mismatch")
    _verify_review_summary(
        arc_review_summary,
        read_json(arc_review_summary),
        review_submission=arc_review_submission,
        public_summary_sha256=input_sha256["public_summary_sha256"],
        queue_sha256=queue_sha256,
        review_count=len(reviews),
    )

    base_ids: list[str] = []
    for mapping in expansion_hidden.values():
        base_sample_id = mapping.get("base_sample_id")
        if not isinstance(base_sample_id, str) or not base_sample_id:
            raise ValueError("Expansion hidden mapping lacks base_sample_id")
        base_ids.append(base_sample_id)
    if len(set(base_ids)) != len(base_ids) or set(base_ids) != set(previous_requests):
        raise ValueError("Expansion anonymous samples do not map one-to-one to previous requests")

    expansion_requests: list[dict[str, Any]] = []
    complete_current_context: list[dict[str, Any]] = []
    for anonymous_sample_id in sorted(queue_ids):
        queued, queue_record_sha256 = queue_bound[anonymous_sample_id]
        review = reviews[anonymous_sample_id]
        expansion_mapping = expansion_hidden[anonymous_sample_id]
        base_sample_id = expansion_mapping["base_sample_id"]
        previous_request = previous_requests[base_sample_id]
        original = original_mapping.get(base_sample_id)
        if original is None:
            raise ValueError(f"Base sample is absent from original dyad mapping: {base_sample_id}")
        if (
            original.get("accepted_for_training") is not False
            or original.get("native_duration_preserved") is not True
            or original.get("official_scenario_or_emotion_exposed") is not False
        ):
            raise ValueError(f"Original dyad mapping violates policy: {base_sample_id}")
        for value, context in (
            (queued, "public queue"),
            (review, "review"),
            (expansion_mapping, "expansion mapping"),
        ):
            if value.get("sample_id") != anonymous_sample_id:
                raise ValueError(f"Anonymous sample mismatch in {context}: {anonymous_sample_id}")
        if (
            expansion_mapping.get("accepted_for_training") is not False
            or expansion_mapping.get("native_duration_preserved") is not True
            or expansion_mapping.get("official_scenario_or_emotion_exposed") is not False
        ):
            raise ValueError(f"Expansion hidden mapping violates policy: {anonymous_sample_id}")
        if (
            expansion_mapping.get("turn_id") != previous_request.get("turn_id")
            or expansion_mapping.get("turn_id") != original.get("turn_id")
            or expansion_mapping.get("reviewed_context_level")
            != previous_request.get("reviewed_context_level")
            or expansion_mapping.get("displayed_context_level")
            != previous_request.get("requested_context_level")
            or expansion_mapping.get("plan_record_sha256")
            != previous_request.get("plan_record_sha256")
            or expansion_mapping.get("video_sha256") != queued.get("video_sha256")
        ):
            raise ValueError(
                f"Expansion mapping does not bind the previous plan: {anonymous_sample_id}"
            )
        current_level = expansion_mapping.get("displayed_context_level")
        if not isinstance(current_level, int) or isinstance(current_level, bool):
            raise ValueError(f"Invalid displayed context level: {anonymous_sample_id}")
        levels = _context_levels(original)
        current_interval = levels.get(current_level)
        if current_interval is None:
            raise ValueError(
                f"Displayed level is absent from original context plan: {anonymous_sample_id}"
            )
        displayed_interval = _interval(
            expansion_mapping.get("displayed_interval"),
            context=f"displayed interval {anonymous_sample_id}",
        )
        previous_interval = _interval(
            previous_request.get("requested_interval"),
            context=f"previous plan requested interval {base_sample_id}",
        )
        if displayed_interval != current_interval or displayed_interval != previous_interval:
            raise ValueError(
                f"Displayed interval changed since the previous plan: {anonymous_sample_id}"
            )
        actors = _actor_mapping(original)
        _verify_public_queue_record(queued, public_root=public_summary.parent)
        evidence = _verify_review(
            review,
            queued=queued,
            queue_record_sha256=queue_record_sha256,
            public_summary_sha256=input_sha256["public_summary_sha256"],
            queue_sha256=queue_sha256,
            mapping=expansion_mapping,
            displayed_interval=displayed_interval,
        )
        common = {
            "schema_version": SCHEMA_VERSION,
            "sample_id": base_sample_id,
            "reviewed_anonymous_sample_id": anonymous_sample_id,
            "turn_id": original["turn_id"],
            "actor_mapping": actors,
            "actor_mapping_sha256": value_sha256(actors),
            "base_video_sha256": review["video_sha256"],
            "reviewed_video_sha256": review["video_sha256"],
            "reviewed_video_frame_count": displayed_interval["frame_count"],
            "reviewed_context_level": current_level,
            "reviewed_interval": displayed_interval,
            "review_evidence_frames": evidence,
            "review_expression_completeness_result": review.get(
                "expression_completeness_result"
            ),
            "review_record_sha256": value_sha256(review),
            "review_queue_record_sha256": queue_record_sha256,
            "review_submission_sha256": input_sha256["review_submission_sha256"],
            "review_summary_sha256": input_sha256["review_summary_sha256"],
            "source_public_summary_sha256": input_sha256["public_summary_sha256"],
            "source_arc_action_queue_sha256": queue_sha256,
            "expansion_hidden_mapping_sha256": input_sha256[
                "expansion_hidden_mapping_sha256"
            ],
            "expansion_hidden_summary_sha256": input_sha256[
                "expansion_hidden_summary_sha256"
            ],
            "original_dyad_mapping_sha256": input_sha256[
                "original_dyad_mapping_sha256"
            ],
            "previous_plan_summary_sha256": input_sha256[
                "previous_plan_summary_sha256"
            ],
            "previous_expansion_requests_sha256": input_sha256[
                "previous_expansion_requests_sha256"
            ],
            "previous_plan_record_sha256": previous_request["plan_record_sha256"],
            "temporal_unit": "complete_natural_interaction_arc",
            "elapsed_duration_used_as_gate": False,
            "fixed_duration_window_used": False,
            "native_duration_preserved": True,
            "semantic_supervision_mask": False,
            "emotion_supervision_mask": False,
            "accepted_for_training": False,
        }
        status = review.get("expression_completeness_result")
        if status == COMPLETE:
            if review.get("expansion_request") is not None or any(
                review.get(f"{phase}_status") != "complete"
                for phase in ("onset", "apex", "offset")
            ):
                raise ValueError(
                    f"Complete review has incomplete boundaries: {anonymous_sample_id}"
                )
            record = {
                **common,
                "artifact_kind": COMPLETE_KIND,
                "next_action": "archive_current_context_pending_other_independent_admission_gates",
            }
            record["plan_record_sha256"] = value_sha256(record)
            complete_current_context.append(record)
            continue
        if status != INCOMPLETE:
            raise ValueError(f"Unknown review completeness result: {anonymous_sample_id}")
        request = review.get("expansion_request")
        if not isinstance(request, dict) or set(request) != {
            "next_context_level",
            "requested_boundary",
        }:
            raise ValueError(
                "Expansion request contains non-natural or extra fields: "
                f"{anonymous_sample_id}"
            )
        next_level = request.get("next_context_level")
        if (
            next_level != current_level + 1
            or request.get("requested_boundary") != REQUESTED_BOUNDARY
            or not any(
                review.get(f"{phase}_status") == "incomplete"
                for phase in ("onset", "apex", "offset")
            )
        ):
            raise ValueError(f"Invalid next natural-boundary request: {anonymous_sample_id}")
        requested_interval = levels.get(next_level)
        if requested_interval is None:
            raise ValueError(
                f"Requested next context level is not predeclared: {anonymous_sample_id}"
            )
        if not _strictly_expands(current_interval, requested_interval):
            raise ValueError(
                f"Requested level is not a strict natural expansion: {anonymous_sample_id}"
            )
        directions = {
            "extend_before": requested_interval["start_frame"]
            < current_interval["start_frame"],
            "extend_after": requested_interval["end_frame_exclusive"]
            > current_interval["end_frame_exclusive"],
        }
        if not any(directions.values()):
            raise ValueError(
                f"Requested level does not extend either boundary: {anonymous_sample_id}"
            )
        record = {
            **common,
            "artifact_kind": REQUEST_KIND,
            "review_requested_boundary": REQUESTED_BOUNDARY,
            "requested_context_level": next_level,
            "requested_interval": requested_interval,
            "review_requested_directions": directions,
            "actual_one_level_expansion": {
                "extended_before": directions["extend_before"],
                "extended_after": directions["extend_after"],
            },
            "expansion_unit": EXPANSION_UNIT,
            "next_action": "render_and_repeat_independent_blind_arc_action_review",
        }
        record["plan_record_sha256"] = value_sha256(record)
        expansion_requests.append(record)

    output_root = output_root.resolve()
    outputs: dict[str, dict[str, Any]] = {}
    for name, rows in (
        ("expansion_requests", expansion_requests),
        ("complete_current_context", complete_current_context),
    ):
        path = output_root / f"{name}.jsonl"
        atomic_jsonl(path, rows)
        outputs[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "records": len(rows),
        }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": PLAN_KIND,
        "plan_role": "round_n_natural_boundary_continuation",
        "inputs": input_sha256,
        "input_record_counts": {
            "previous_expansion_requests": len(previous_requests),
            "reviewed_anonymous_samples": len(reviews),
        },
        "expression_completeness_distribution": dict(
            sorted(
                Counter(
                    row["expression_completeness_result"] for row in reviews.values()
                ).items()
            )
        ),
        "reviewed_context_level_distribution": dict(
            sorted(Counter(str(row["context_level"]) for row in reviews.values()).items())
        ),
        "requested_context_level_distribution": dict(
            sorted(
                Counter(
                    str(row["requested_context_level"])
                    for row in expansion_requests
                ).items()
            )
        ),
        "selection_policy": SELECTION_POLICY,
        "continuation_requires_exact_requested_boundary": REQUESTED_BOUNDARY,
        "anonymous_samples_mapped_one_to_one_to_previous_base_samples": True,
        "previous_plan_records_bound_by_sha256": True,
        "fixed_minimum_maximum_or_target_duration_used": False,
        "outputs": outputs,
        "accepted_for_training_count": 0,
    }
    atomic_json(output_root / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_plan(
        public_summary=args.public_summary,
        arc_review_submission=args.arc_review_submission,
        arc_review_summary=args.arc_review_summary,
        expansion_hidden_mapping=args.expansion_hidden_mapping,
        expansion_hidden_summary=args.expansion_hidden_summary,
        original_dyad_mapping=args.original_dyad_mapping,
        previous_plan_summary=args.previous_plan_summary,
        previous_expansion_requests=args.previous_expansion_requests,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
