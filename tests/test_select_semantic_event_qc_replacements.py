import json
from pathlib import Path

import pytest

from tools.human_motion_collection import select_semantic_event_qc_replacements as repl


def _write(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _candidate(clip_id: str, split: str, emotion: str, rank: int, band="low"):
    return {
        "clip_id": clip_id,
        "task_id": clip_id,
        "source_clip_id": f"source_{clip_id}",
        "speaker_key": f"speaker_{split}",
        "fixed_split_assignment": split,
        "pilot_split": split,
        "emotion_id": emotion,
        "semantic_event": {"category": "deictic", "intensity": "low"},
        "pilot_dynamic_band": band,
        "pilot_candidate_rank_within_split_stratum": rank,
        "training_segment": {
            "representation": "native_variable_length_semantic_clip_v1",
            "fixed_window_sec": None,
        },
        "accepted_for_training": False,
    }


def _result(record: dict, status: str) -> dict:
    return {**record, "status": status}


def test_selects_next_rank_in_same_split_stratum_and_hash_binds(tmp_path):
    selected = [
        _candidate("train_original", "train", "happy", 1),
        _candidate("test_original", "test", "sad", 1),
    ]
    train_next = _candidate("train_next", "train", "happy", 2, "medium")
    wrong_split = _candidate("test_happy", "test", "happy", 2)
    high_fallback = _candidate("train_high", "train", "happy", 3, "high_fallback")
    candidates = selected + [train_next, wrong_split, high_fallback]
    paths = {name: tmp_path / f"{name}.jsonl" for name in ("candidates", "selected", "passed", "failed")}
    _write(paths["candidates"], candidates)
    _write(paths["selected"], selected)
    _write(paths["passed"], [_result(selected[1], "passed")])
    _write(paths["failed"], [_result(selected[0], "quality_failed")])
    output = tmp_path / "replacement.jsonl"

    summary = repl.build_replacements(
        paths["candidates"], paths["selected"], [paths["passed"]], [paths["failed"]], output
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]

    assert summary["coverage_fillable"] is True
    assert summary["replacement_count"] == 1
    assert [row["clip_id"] for row in rows] == ["train_next"]
    assert rows[0]["fixed_split_assignment"] == "train"
    assert rows[0]["qc_replacement_for_stratum"] == "train|happy|deictic|low"
    assert rows[0]["qc_replacement_contract_sha256"] == summary[
        "replacement_contract_sha256"
    ]
    assert rows[0]["accepted_for_training"] is False


def test_refuses_replacement_while_original_selected_clip_is_pending(tmp_path):
    selected = [_candidate("original", "train", "happy", 1)]
    candidates = selected + [_candidate("next", "train", "happy", 2)]
    candidates_path = tmp_path / "candidates.jsonl"
    selected_path = tmp_path / "selected.jsonl"
    passed_path = tmp_path / "passed.jsonl"
    failed_path = tmp_path / "failed.jsonl"
    _write(candidates_path, candidates)
    _write(selected_path, selected)
    _write(passed_path, [])
    _write(failed_path, [])

    with pytest.raises(ValueError, match="pending clips"):
        repl.build_replacements(
            candidates_path,
            selected_path,
            [passed_path],
            [failed_path],
            tmp_path / "replacement.jsonl",
        )


def test_reports_unfillable_when_only_high_fallback_remains(tmp_path):
    selected = [_candidate("original", "validation", "fear", 1)]
    candidates = selected + [
        _candidate("high", "validation", "fear", 2, "high_fallback")
    ]
    paths = {name: tmp_path / f"{name}.jsonl" for name in ("candidates", "selected", "passed", "failed")}
    _write(paths["candidates"], candidates)
    _write(paths["selected"], selected)
    _write(paths["passed"], [])
    _write(paths["failed"], [_result(selected[0], "quality_failed")])

    summary = repl.build_replacements(
        paths["candidates"],
        paths["selected"],
        [paths["passed"]],
        [paths["failed"]],
        tmp_path / "replacement.jsonl",
    )

    assert summary["coverage_fillable"] is False
    assert summary["replacement_count"] == 0
    assert summary["unfillable_deficits"] == {"validation|fear|deictic|low": 1}
