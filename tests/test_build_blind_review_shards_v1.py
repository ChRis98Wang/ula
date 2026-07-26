from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.human_motion_review.build_blind_review_shards_v1 import (
    assign_shards,
    build_shards,
    read_jsonl,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, records: int = 10) -> Path:
    public = tmp_path / "public"
    videos = public / "videos"
    videos.mkdir(parents=True)
    rows = []
    for index in range(records):
        video = videos / f"sample_{index}.mp4"
        video.write_bytes(f"video-{index}".encode())
        rows.append(
            {
                "sample_id": f"sample_{index}",
                "video_path": str(video),
                "video_sha256": _sha(video),
                "accepted_for_training": False,
            }
        )
    queue = public / "arc.jsonl"
    queue.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    summary = public / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "accepted_for_training": False,
                "fixed_duration_window_used": False,
                "arc_action_queue": str(queue),
                "arc_action_queue_sha256": _sha(queue),
            }
        ),
        encoding="utf-8",
    )
    return summary


def test_assign_shards_is_balanced_complete_and_deterministic() -> None:
    rows = [{"sample_id": f"sample_{index}"} for index in range(10)]
    first = assign_shards(rows, queue_sha256="a" * 64, shard_count=3)
    second = assign_shards(list(reversed(rows)), queue_sha256="a" * 64, shard_count=3)
    assert first == second
    assert sorted(row["sample_id"] for shard in first for row in shard) == sorted(
        row["sample_id"] for row in rows
    )
    assert [len(shard) for shard in first] == [4, 3, 3]


def test_build_shards_writes_exact_nonoverlapping_coverage(tmp_path: Path) -> None:
    public_summary = _fixture(tmp_path)
    output = tmp_path / "shards"
    result = build_shards(
        public_summary_path=public_summary,
        queue_kind="arc_action",
        shard_count=4,
        output_root=output,
    )
    assert result["records"] == 10
    assert result["coverage_complete_without_overlap"] is True
    sample_ids = []
    for shard in result["shards"]:
        rows = read_jsonl(Path(shard["summary"]).parent / "review_queue.jsonl")
        sample_ids.extend(row["sample_id"] for row in rows)
    assert len(sample_ids) == len(set(sample_ids)) == 10
    assert result["maximum_shard_records"] - result["minimum_shard_records"] <= 1


def test_build_shards_rejects_tampered_source_queue(tmp_path: Path) -> None:
    public_summary = _fixture(tmp_path)
    public = json.loads(public_summary.read_text(encoding="utf-8"))
    Path(public["arc_action_queue"]).write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA mismatch"):
        build_shards(
            public_summary_path=public_summary,
            queue_kind="arc_action",
            shard_count=2,
            output_root=tmp_path / "shards",
        )
