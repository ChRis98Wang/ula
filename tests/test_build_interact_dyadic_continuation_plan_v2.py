import hashlib
import json
from pathlib import Path

import pytest

from tools.human_motion_review.build_interact_arc_expansion_plan_v2 import (
    sha256_file,
    value_sha256,
)
from tools.human_motion_review.build_interact_dyadic_continuation_plan_v2 import (
    ARC_PROTOCOL,
    EXPANSION_HIDDEN_KIND,
    EXPANSION_UNIT,
    INCOMPLETE,
    PLAN_KIND,
    PUBLIC_DURATION_POLICY,
    PUBLIC_KIND,
    REQUESTED_BOUNDARY,
    REQUEST_KIND,
    REVIEW_SUMMARY_KIND,
    SCHEMA_VERSION,
    SELECTION_POLICY,
    build_plan,
)
from tools.human_motion_review.run_interact_dyadic_expansion_review_v2 import (
    load_groups,
)


def _json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _interval(start: int, end: int):
    return {
        "start_frame": start,
        "end_frame_exclusive": end,
        "frame_count": end - start,
    }


def _previous_request(sample_id, turn_id, current, requested):
    row = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": REQUEST_KIND,
        "sample_id": sample_id,
        "turn_id": turn_id,
        "reviewed_context_level": 0,
        "requested_context_level": 1,
        "reviewed_interval": current,
        "requested_interval": requested,
        "expansion_unit": EXPANSION_UNIT,
        "elapsed_duration_used_as_gate": False,
        "semantic_supervision_mask": False,
        "emotion_supervision_mask": False,
        "accepted_for_training": False,
    }
    row["plan_record_sha256"] = value_sha256(row)
    return row


def _actor_mapping(prefix):
    return {
        "A": {
            "episode_task_id": f"task-{prefix}-a",
            "actor_id": f"actor-{prefix}-a",
            "partner_actor_id": f"actor-{prefix}-b",
        },
        "B": {
            "episode_task_id": f"task-{prefix}-b",
            "actor_id": f"actor-{prefix}-b",
            "partner_actor_id": f"actor-{prefix}-a",
        },
    }


def _original_mapping(sample_id, turn_id, prefix, levels):
    return {
        "sample_id": sample_id,
        "turn_id": turn_id,
        "actor_mapping": _actor_mapping(prefix),
        "displayed_context_level": 0,
        "context_plan": {
            "duration_gate_used": False,
            "duration_policy": (
                "semantic_affect_complete_at_predeclared_shared_rest_boundaries;"
                "no_fixed_target_minimum_or_maximum_duration"
            ),
            "completeness_review": {
                "elapsed_seconds_may_influence_decision": False,
            },
            "levels": [
                {
                    "level": index,
                    "start_frame": interval["start_frame"],
                    "end_frame_exclusive": interval["end_frame_exclusive"],
                }
                for index, interval in enumerate(levels)
            ],
        },
        "native_duration_preserved": True,
        "official_scenario_or_emotion_exposed": False,
        "accepted_for_training": False,
    }


def _queue_record(sample_id, video, context_level):
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": ARC_PROTOCOL,
        "sample_id": sample_id,
        "temporal_unit": "complete_natural_interaction_arc",
        "video_path": str(video.resolve()),
        "video_sha256": sha256_file(video),
        "context_level": context_level,
        "label_metadata_exposed": False,
        "native_duration_preserved": True,
        "fixed_duration_window_used": False,
        "accepted_for_training": False,
    }


def _review_record(
    queued,
    *,
    public_summary_sha256,
    queue_sha256,
    queue_record_sha256,
    frame_count,
    complete,
):
    row = {
        **queued,
        "blind_review_provenance": {
            "public_summary_sha256": public_summary_sha256,
            "arc_action_queue_sha256": queue_sha256,
            "queue_record_hash_method": "sha256_utf8_line_without_lf",
            "queue_record_sha256": queue_record_sha256,
        },
        "decode_validation": {
            "complete": True,
            "decoded_frame_count": frame_count,
            "reported_frame_count": frame_count,
            "video_sha256_verified": True,
        },
        "onset_evidence_frame": 0,
        "apex_evidence_frame": min(10, frame_count - 1),
        "offset_evidence_frame": frame_count - 1,
        "onset_status": "complete" if complete else "incomplete",
        "apex_status": "complete",
        "offset_status": "complete" if complete else "incomplete",
        "expression_completeness_result": "complete" if complete else INCOMPLETE,
        "expansion_request": (
            None
            if complete
            else {
                "next_context_level": queued["context_level"] + 1,
                "requested_boundary": REQUESTED_BOUNDARY,
            }
        ),
    }
    return row


def _case(tmp_path: Path):
    intervals = {
        "complete": [_interval(100, 200), _interval(50, 200), _interval(50, 260)],
        "expand": [_interval(300, 400), _interval(250, 400), _interval(250, 480)],
    }
    bases = {
        "base-complete": ("turn-complete", "complete"),
        "base-expand": ("turn-expand", "expand"),
    }
    original_path = tmp_path / "original/dyad_mapping.jsonl"
    original_rows = [
        _original_mapping(base, turn, prefix, intervals[prefix])
        for base, (turn, prefix) in bases.items()
    ]
    _jsonl(original_path, original_rows)

    previous_requests_path = tmp_path / "previous/expansion_requests.jsonl"
    previous_requests = [
        _previous_request(base, turn, intervals[prefix][0], intervals[prefix][1])
        for base, (turn, prefix) in bases.items()
    ]
    _jsonl(previous_requests_path, previous_requests)
    previous_plan_path = tmp_path / "previous/summary.json"
    _json(
        previous_plan_path,
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": PLAN_KIND,
            "inputs": {"hidden_mapping_sha256": sha256_file(original_path)},
            "selection_policy": SELECTION_POLICY,
            "fixed_minimum_maximum_or_target_duration_used": False,
            "outputs": {
                "expansion_requests": {
                    "path": str(previous_requests_path.resolve()),
                    "sha256": sha256_file(previous_requests_path),
                    "records": len(previous_requests),
                }
            },
            "accepted_for_training_count": 0,
        },
    )

    public_root = tmp_path / "public"
    videos = {}
    for anonymous in ("anon-complete", "anon-expand"):
        video = public_root / "videos" / f"{anonymous}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes((anonymous + "-native-video").encode("utf-8"))
        videos[anonymous] = video
    queue_path = public_root / "arc_action_review_queue.jsonl"
    queue_rows = [
        _queue_record("anon-complete", videos["anon-complete"], 1),
        _queue_record("anon-expand", videos["anon-expand"], 1),
    ]
    _jsonl(queue_path, queue_rows)
    public_summary_path = public_root / "summary.json"
    run_state_sha256 = "9" * 64
    _json(
        public_summary_path,
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": PUBLIC_KIND,
            "arc_action_records": len(queue_rows),
            "arc_action_queue": str(queue_path.resolve()),
            "arc_action_queue_sha256": sha256_file(queue_path),
            "plan_summary_sha256": sha256_file(previous_plan_path),
            "run_state_sha256": run_state_sha256,
            "duration_policy": PUBLIC_DURATION_POLICY,
            "fixed_duration_window_used": False,
            "identity_scenario_official_text_or_emotion_exposed": False,
            "accepted_for_training": False,
        },
    )

    previous_by_base = {row["sample_id"]: row for row in previous_requests}
    queue_by_sample = {row["sample_id"]: row for row in queue_rows}
    expansion_mapping_path = tmp_path / "hidden/sample_mapping.jsonl"
    expansion_mapping = [
        {
            "sample_id": anonymous,
            "base_sample_id": base,
            "turn_id": turn,
            "reviewed_context_level": 0,
            "displayed_context_level": 1,
            "displayed_interval": intervals[prefix][1],
            "plan_record_sha256": previous_by_base[base]["plan_record_sha256"],
            "video_sha256": queue_by_sample[anonymous]["video_sha256"],
            "native_duration_preserved": True,
            "official_scenario_or_emotion_exposed": False,
            "accepted_for_training": False,
        }
        for anonymous, base, turn, prefix in (
            ("anon-complete", "base-complete", "turn-complete", "complete"),
            ("anon-expand", "base-expand", "turn-expand", "expand"),
        )
    ]
    _jsonl(expansion_mapping_path, expansion_mapping)
    expansion_hidden_summary_path = tmp_path / "hidden/summary.json"
    _json(
        expansion_hidden_summary_path,
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": EXPANSION_HIDDEN_KIND,
            "public_summary": str(public_summary_path.resolve()),
            "plan_summary": str(previous_plan_path.resolve()),
            "plan_summary_sha256": sha256_file(previous_plan_path),
            "expansion_requests": str(previous_requests_path.resolve()),
            "expansion_requests_sha256": sha256_file(previous_requests_path),
            "run_state": str((tmp_path / "run_state.json").resolve()),
            "run_state_sha256": run_state_sha256,
            "sample_mapping": str(expansion_mapping_path.resolve()),
            "sample_mapping_sha256": sha256_file(expansion_mapping_path),
            "accepted_for_training": False,
        },
    )

    queue_record_hashes = [
        hashlib.sha256(line).hexdigest() for line in queue_path.read_bytes().splitlines()
    ]
    public_summary_sha256 = sha256_file(public_summary_path)
    queue_sha256 = sha256_file(queue_path)
    review_rows = [
        _review_record(
            queue_rows[0],
            public_summary_sha256=public_summary_sha256,
            queue_sha256=queue_sha256,
            queue_record_sha256=queue_record_hashes[0],
            frame_count=intervals["complete"][1]["frame_count"],
            complete=True,
        ),
        _review_record(
            queue_rows[1],
            public_summary_sha256=public_summary_sha256,
            queue_sha256=queue_sha256,
            queue_record_sha256=queue_record_hashes[1],
            frame_count=intervals["expand"][1]["frame_count"],
            complete=False,
        ),
    ]
    review_path = tmp_path / "reviews/arc_action_reviewer.jsonl"
    _jsonl(review_path, review_rows)
    review_summary_path = tmp_path / "reviews/arc_action_reviewer.summary.json"
    _json(
        review_summary_path,
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": REVIEW_SUMMARY_KIND,
            "record_count": len(review_rows),
            "output_jsonl_sha256": sha256_file(review_path),
            "input_bindings": {
                "public_summary_sha256": public_summary_sha256,
                "arc_action_queue_sha256": queue_sha256,
            },
            "fixed_duration_window_used": False,
            "native_duration_preserved": True,
            "accepted_for_training": False,
        },
    )
    return {
        "public_summary": public_summary_path,
        "arc_review_submission": review_path,
        "arc_review_summary": review_summary_path,
        "expansion_hidden_mapping": expansion_mapping_path,
        "expansion_hidden_summary": expansion_hidden_summary_path,
        "original_dyad_mapping": original_path,
        "previous_plan_summary": previous_plan_path,
        "previous_expansion_requests": previous_requests_path,
        "intervals": intervals,
    }


def _build(case, output_root):
    return build_plan(
        public_summary=case["public_summary"],
        arc_review_submission=case["arc_review_submission"],
        arc_review_summary=case["arc_review_summary"],
        expansion_hidden_mapping=case["expansion_hidden_mapping"],
        expansion_hidden_summary=case["expansion_hidden_summary"],
        original_dyad_mapping=case["original_dyad_mapping"],
        previous_plan_summary=case["previous_plan_summary"],
        previous_expansion_requests=case["previous_expansion_requests"],
        output_root=output_root,
    )


def _rewrite_review(case, mutate):
    rows = _read_jsonl(case["arc_review_submission"])
    mutate(rows)
    _jsonl(case["arc_review_submission"], rows)
    summary = json.loads(case["arc_review_summary"].read_text(encoding="utf-8"))
    summary["output_jsonl_sha256"] = sha256_file(case["arc_review_submission"])
    _json(case["arc_review_summary"], summary)


def test_round_n_plan_maps_back_to_base_and_is_directly_runner_compatible(tmp_path):
    case = _case(tmp_path)
    output_root = tmp_path / "continuation"
    summary = _build(case, output_root)
    assert summary["outputs"]["complete_current_context"]["records"] == 1
    assert summary["outputs"]["expansion_requests"]["records"] == 1
    complete = _read_jsonl(output_root / "complete_current_context.jsonl")[0]
    request = _read_jsonl(output_root / "expansion_requests.jsonl")[0]
    assert complete["sample_id"] == "base-complete"
    assert complete["reviewed_anonymous_sample_id"] == "anon-complete"
    assert request["sample_id"] == "base-expand"
    assert request["reviewed_anonymous_sample_id"] == "anon-expand"
    assert request["turn_id"] == "turn-expand"
    assert request["actor_mapping"] == _actor_mapping("expand")
    assert request["reviewed_context_level"] == 1
    assert request["requested_context_level"] == 2
    assert request["requested_interval"] == case["intervals"]["expand"][2]
    assert request["previous_plan_record_sha256"]
    assert request["review_evidence_frames"]["offset"] == {
        "status": "incomplete",
        "local_frame": 149,
        "source_frame": 399,
    }
    assert summary["inputs"]["hidden_mapping_sha256"] == sha256_file(
        case["original_dyad_mapping"]
    )

    source_root = tmp_path / "sources"
    receipt_rows = []
    for role in ("a", "b"):
        source = source_root / f"{role}.bvh"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"source-{role}".encode("utf-8"))
        receipt_rows.append(
            {
                "episode_task_id": f"task-expand-{role}",
                "source_bvh": str(source.resolve()),
                "source_bvh_sha256": sha256_file(source),
            }
        )
    receipt = tmp_path / "receipt.json"
    _json(receipt, {"selected": receipt_rows})
    groups, _binding = load_groups(
        plan_summary=output_root / "summary.json",
        expansion_requests=output_root / "expansion_requests.jsonl",
        hidden_mapping=case["original_dyad_mapping"],
        receipt=receipt,
    )
    assert len(groups) == 1
    assert groups[0]["sample_id"] == "base-expand"
    assert groups[0]["requested_context_level"] == 2
    assert groups[0]["source_interval"] == [250, 480]
    assert groups[0]["expected_frame_count"] == 230


def test_round_n_plan_rejects_fixed_window_or_extra_expansion_fields(tmp_path):
    case = _case(tmp_path)

    def mutate(rows):
        rows[1]["expansion_request"]["fixed_window_seconds"] = 6

    _rewrite_review(case, mutate)
    with pytest.raises(ValueError, match="non-natural or extra fields"):
        _build(case, tmp_path / "out")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda rows: rows[1]["expansion_request"].update(
                {"next_context_level": 3}
            ),
            "Invalid next natural-boundary request",
        ),
        (
            lambda rows: rows[1]["decode_validation"].update(
                {"decoded_frame_count": 149, "reported_frame_count": 149}
            ),
            "decode/frame binding mismatch",
        ),
        (
            lambda rows: rows[1].update({"context_level": 2}),
            "Reviewed context level mismatch",
        ),
    ],
)
def test_round_n_plan_rejects_level_and_frame_binding_changes(
    tmp_path, mutate, message
):
    case = _case(tmp_path)
    _rewrite_review(case, mutate)
    with pytest.raises(ValueError, match=message):
        _build(case, tmp_path / "out")


def test_round_n_plan_rejects_changed_previous_plan_record_binding(tmp_path):
    case = _case(tmp_path)
    mappings = _read_jsonl(case["expansion_hidden_mapping"])
    mappings[1]["plan_record_sha256"] = "0" * 64
    _jsonl(case["expansion_hidden_mapping"], mappings)
    hidden_summary = json.loads(
        case["expansion_hidden_summary"].read_text(encoding="utf-8")
    )
    hidden_summary["sample_mapping_sha256"] = sha256_file(
        case["expansion_hidden_mapping"]
    )
    _json(case["expansion_hidden_summary"], hidden_summary)
    with pytest.raises(ValueError, match="does not bind the previous plan"):
        _build(case, tmp_path / "out")
