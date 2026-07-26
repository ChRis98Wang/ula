import json
from pathlib import Path

from tools.human_motion_review.build_interact_arc_expansion_plan_v2 import (
    sha256_file,
    value_sha256,
)
from tools.human_motion_review.run_interact_dyadic_expansion_review_v2 import load_groups


def _json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_load_groups_binds_one_level_interval_and_partner_sources(tmp_path):
    sources = []
    for name in ("a.bvh", "b.bvh"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        sources.append(path)
    request = {
        "schema_version": "2.0.0",
        "artifact_kind": "interact_dyadic_one_level_natural_context_expansion_request_v2",
        "sample_id": "dyad-1",
        "turn_id": "turn-1",
        "reviewed_context_level": 0,
        "requested_context_level": 1,
        "reviewed_interval": {"start_frame": 100, "end_frame_exclusive": 200, "frame_count": 100},
        "requested_interval": {"start_frame": 50, "end_frame_exclusive": 200, "frame_count": 150},
        "expansion_unit": "exactly_one_next_predeclared_shared_rest_boundary_level",
        "elapsed_duration_used_as_gate": False,
        "semantic_supervision_mask": False,
        "emotion_supervision_mask": False,
        "accepted_for_training": False,
    }
    request["plan_record_sha256"] = value_sha256(request)
    requests = tmp_path / "requests.jsonl"
    _jsonl(requests, [request])
    hidden = tmp_path / "hidden.jsonl"
    _jsonl(
        hidden,
        [
            {
                "sample_id": "dyad-1",
                "turn_id": "turn-1",
                "actor_mapping": {
                    "A": {"episode_task_id": "task-a"},
                    "B": {"episode_task_id": "task-b"},
                },
            }
        ],
    )
    receipt = tmp_path / "receipt.json"
    _json(
        receipt,
        {
            "selected": [
                {
                    "episode_task_id": "task-a",
                    "source_bvh": str(sources[0]),
                    "source_bvh_sha256": sha256_file(sources[0]),
                },
                {
                    "episode_task_id": "task-b",
                    "source_bvh": str(sources[1]),
                    "source_bvh_sha256": sha256_file(sources[1]),
                },
            ]
        },
    )
    plan = tmp_path / "plan.json"
    _json(
        plan,
        {
            "artifact_kind": "interact_dyadic_arc_action_one_level_expansion_plan_v2",
            "accepted_for_training_count": 0,
            "inputs": {"hidden_mapping_sha256": sha256_file(hidden)},
            "outputs": {
                "expansion_requests": {"sha256": sha256_file(requests), "records": 1}
            },
        },
    )
    groups, binding = load_groups(
        plan_summary=plan,
        expansion_requests=requests,
        hidden_mapping=hidden,
        receipt=receipt,
    )
    assert len(groups) == 1
    assert groups[0]["source_interval"] == [50, 200]
    assert groups[0]["expected_frame_count"] == 150
    assert groups[0]["requested_context_level"] == 1
    assert groups[0]["accepted_for_training"] is False
    assert binding["expansion_requests_sha256"] == sha256_file(requests)
