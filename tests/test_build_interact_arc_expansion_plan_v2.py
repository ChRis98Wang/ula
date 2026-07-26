import hashlib
import json
from pathlib import Path

from tools.human_motion_review.build_interact_arc_expansion_plan_v2 import (
    ARC_PROTOCOL,
    build_plan,
    value_sha256,
)


def _json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_plan_advances_exactly_one_predeclared_level_without_duration_gate(tmp_path):
    public = tmp_path / "public/summary.json"
    queue = public.parent / "arc.jsonl"
    queued = {"sample_id": "dyad-1", "video_sha256": "a" * 64}
    _jsonl(queue, [queued])
    _json(
        public,
        {
            "arc_action_queue": str(queue),
            "arc_action_queue_sha256": _sha(queue),
        },
    )
    review_path = tmp_path / "review.jsonl"
    review = {
        **queued,
        "protocol_version": ARC_PROTOCOL,
        "accepted_for_training": False,
        "fixed_duration_window_used": False,
        "native_duration_preserved": True,
        "context_level": 0,
        "expression_completeness_result": "incomplete_requires_natural_boundary_expansion",
        "expansion_request": {
            "required": True,
            "boundary_policy": "next_predeclared_natural_context_level",
            "next_context_level": 1,
            "extend_before": True,
            "extend_after": True,
        },
    }
    _jsonl(review_path, [review])
    review_summary = tmp_path / "review.summary.json"
    _json(
        review_summary,
        {"accepted_for_training": False, "submission_jsonl_sha256": _sha(review_path)},
    )
    hidden = tmp_path / "hidden.jsonl"
    _jsonl(
        hidden,
        [
            {
                "sample_id": "dyad-1",
                "turn_id": "turn-1",
                "displayed_context_level": 0,
                "context_plan": {
                    "duration_gate_used": False,
                    "levels": [
                        {"level": 0, "start_frame": 100, "end_frame_exclusive": 200},
                        {"level": 1, "start_frame": 50, "end_frame_exclusive": 200},
                        {"level": 2, "start_frame": 50, "end_frame_exclusive": 260},
                    ],
                },
            }
        ],
    )
    migration = tmp_path / "migration.json"
    migration_value = {
        "artifact_kind": "interact_blind_review_axis_only_bundle_migration_evidence_v2",
        "new_public_summary": {"sha256": _sha(public)},
        "carried_forward_review": {
            "submission_sha256": _sha(review_path),
            "summary_sha256": _sha(review_summary),
        },
        "review_axes_allowed_to_carry_forward": ["arc_action"],
    }
    migration_value["evidence_record_sha256"] = value_sha256(migration_value)
    _json(migration, migration_value)
    result = build_plan(
        public_summary=public,
        review_submission=review_path,
        review_summary=review_summary,
        migration_evidence=migration,
        hidden_mapping=hidden,
        output_root=tmp_path / "out",
    )
    assert result["outputs"]["expansion_requests"]["records"] == 1
    output = json.loads((tmp_path / "out/expansion_requests.jsonl").read_text())
    assert output["requested_context_level"] == 1
    assert output["requested_interval"] == {
        "start_frame": 50,
        "end_frame_exclusive": 200,
        "frame_count": 150,
    }
    assert output["actual_one_level_expansion"] == {
        "extended_before": True,
        "extended_after": False,
    }
    assert output["elapsed_duration_used_as_gate"] is False
    assert output["accepted_for_training"] is False
