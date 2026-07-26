import hashlib
import json
from pathlib import Path

from tools.human_motion_review.build_expression_turn_expansion_plan_v8 import (
    build_plan,
    value_sha256,
)


def _stable(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(_stable(row) + "\n" for row in rows), encoding="utf-8")


def test_expansion_advances_one_natural_level_and_never_uses_duration(tmp_path):
    task_id = "turn-001"
    expression_hash = "a" * 64
    comparison = {
        "sample_id": "expr-001",
        "video_sha256": "b" * 64,
        "context_level": 0,
        "qualification_status": "natural_context_expansion_required",
        "elapsed_duration_used_as_gate": False,
        "accepted_for_training": False,
    }
    comparison["comparison_record_sha256"] = value_sha256(comparison)
    hidden = {
        "sample_id": "expr-001",
        "task_id": task_id,
        "source_clip_id": "source-001",
        "fixed_split_assignment": "train",
        "video_sha256": "b" * 64,
        "expression_turn_record_sha256": expression_hash,
        "official_emotion": "must-not-leak",
    }
    candidate = {
        "task_id": task_id,
        "expression_turn_record_sha256": expression_hash,
        "context_plan": {
            "selected_level": 0,
            "context_exhausted_at_level": 2,
            "levels": [
                {"level": 0, "start_frame": 20, "end_frame_exclusive": 50},
                {"level": 1, "start_frame": 10, "end_frame_exclusive": 70},
                {"level": 2, "start_frame": 0, "end_frame_exclusive": 90},
            ],
        },
    }
    comparison_path = tmp_path / "comparison.jsonl"
    hidden_path = tmp_path / "hidden.jsonl"
    catalog_path = tmp_path / "catalog.jsonl"
    _jsonl(comparison_path, [comparison])
    _jsonl(hidden_path, [hidden])
    _jsonl(catalog_path, [candidate])
    summary = build_plan(
        comparison_records=comparison_path,
        hidden_mapping=hidden_path,
        candidate_catalog=catalog_path,
        output_root=tmp_path / "out",
    )
    assert summary["outputs"]["expansion_requests"]["records"] == 1
    output = json.loads(
        (tmp_path / "out/expansion_requests.jsonl").read_text().strip()
    )
    assert output["requested_context_level"] == 1
    assert output["requested_interval"]["frame_count"] == 60
    assert output["elapsed_duration_used_as_gate"] is False
    assert "official_emotion" not in _stable(output)
    assert output["accepted_for_training"] is False
