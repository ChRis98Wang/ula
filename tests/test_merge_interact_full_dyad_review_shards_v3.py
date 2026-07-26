import json
from pathlib import Path

import pytest

from tools.human_motion_review import build_blind_review_shards_v1 as sharder
from tools.human_motion_review import finalize_interact_full_dyad_review_shard_v3 as finalizer
from tools.human_motion_review import merge_interact_full_dyad_review_shards_v3 as merger


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def arc_decision(offset: int = 18) -> dict:
    return {
        "full_video_reviewed": True,
        "onset_status": "complete",
        "onset_evidence_frame": 2,
        "apex_status": "complete",
        "apex_evidence_frame": 10,
        "offset_status": "complete",
        "offset_evidence_frame": offset,
        "interaction_observable_result": "observable",
        "interaction_description_en": "The pair coordinate an arm gesture.",
        "robot_a_observable_motion_en": "Robot A raises and lowers one arm.",
        "robot_b_observable_motion_en": "Robot B turns and responds.",
        "expression_completeness_result": "complete",
        "expansion_request": None,
        "review_notes": "Reviewed from frame zero through EOF.",
    }


def source_bundle(tmp_path: Path) -> Path:
    videos = tmp_path / "public/videos"
    videos.mkdir(parents=True)
    rows = []
    for index in range(2):
        video = videos / f"sample-{index}.mp4"
        video.write_bytes(f"video-{index}".encode())
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "video_path": str(video),
                "video_sha256": finalizer.sha256_file(video),
                "native_variable_length": True,
                "fixed_duration_window_used": False,
                "accepted_for_training": False,
            }
        )
    queue = tmp_path / "public/arc.jsonl"
    queue.write_text("".join(finalizer.stable_json(row) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "arc_action_queue": str(queue),
        "arc_action_queue_sha256": finalizer.sha256_file(queue),
        "fixed_duration_window_used": False,
        "accepted_for_training": False,
    }
    summary_path = tmp_path / "public/summary.json"
    write_json(summary_path, summary)
    return summary_path


def build_reviewed_shards(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    public = source_bundle(tmp_path)
    shards_root = tmp_path / "shards"
    sharder.build_shards(
        public_summary_path=public,
        queue_kind="arc_action",
        shard_count=2,
        output_root=shards_root,
    )
    monkeypatch.setattr(
        finalizer,
        "probe_video",
        lambda _: {"frames": 20, "fps": 30.0, "width": 1280, "height": 720},
    )
    submissions = tmp_path / "submissions"
    for index in range(2):
        shard_summary = shards_root / f"shard_{index:03d}/summary.json"
        queue = finalizer.read_jsonl(shards_root / f"shard_{index:03d}/review_queue.jsonl")
        decisions = tmp_path / f"decisions-{index}.json"
        write_json(decisions, {row["sample_id"]: arc_decision() for row in queue})
        finalizer.finalize(
            shard_summary_path=shard_summary,
            decisions_path=decisions,
            reviewer_id=f"reviewer-{index}",
            output_submission_path=submissions / f"shard_{index:03d}/submission.jsonl",
            output_summary_path=submissions / f"shard_{index:03d}/summary.json",
        )
    return shards_root / "summary.json", submissions


def test_merge_requires_and_covers_every_shard(tmp_path, monkeypatch):
    shards, submissions = build_reviewed_shards(tmp_path, monkeypatch)
    result = merger.merge(
        shards_summary_path=shards,
        submissions_root=submissions,
        output_submission_path=tmp_path / "merged/submission.jsonl",
        output_summary_path=tmp_path / "merged/summary.json",
    )
    assert result["records"] == 2
    assert result["coverage_complete_without_overlap"] is True
    assert len(finalizer.read_jsonl(tmp_path / "merged/submission.jsonl")) == 2


def test_merge_rejects_missing_shard(tmp_path, monkeypatch):
    shards, submissions = build_reviewed_shards(tmp_path, monkeypatch)
    (submissions / "shard_001/summary.json").unlink()
    with pytest.raises(ValueError, match="Missing reviewed shard"):
        merger.merge(
            shards_summary_path=shards,
            submissions_root=submissions,
            output_submission_path=tmp_path / "merged/submission.jsonl",
            output_summary_path=tmp_path / "merged/summary.json",
        )
