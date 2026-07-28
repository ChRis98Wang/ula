import json

from tools import audit_beat2_motion_foundation_ingest_gap as audit


def _event(source, start, end, *, task_id="event"):
    return {
        "clip_id": task_id,
        "task_id": task_id,
        "source_clip_id": source,
        "training_segment": {
            "start_frame": start,
            "end_frame_exclusive": end,
            "frame_count": end - start,
        },
        "semantic_event": {"source_duration_sec": (end - start) / 30.0},
    }


def test_chunk_bounds_absorbs_only_short_tail():
    assert audit.chunk_bounds(620, max_frames=300, min_frames=30) == [
        (0, 300),
        (300, 620),
    ]
    assert audit.chunk_bounds(610, max_frames=300, min_frames=30) == [
        (0, 300),
        (300, 610),
    ]
    assert audit.chunk_bounds(620, max_frames=200, min_frames=30) == [
        (0, 200),
        (200, 400),
        (400, 620),
    ]
    assert audit.chunk_bounds(625, max_frames=200, min_frames=30) == [
        (0, 200),
        (200, 400),
        (400, 625),
    ]


def test_chunk_inventory_exactly_covers_sources_without_overlap(tmp_path):
    motion_root = tmp_path / "smplxflame_30"
    source_frames = {"1_wayne_0_1_1": 620, "2_scott_0_1_1": 305}
    splits = {"1_wayne": "train", "2_scott": "test"}

    records = audit.build_chunk_inventory(
        motion_root=motion_root,
        source_frames=source_frames,
        splits=splits,
        max_frames=300,
        min_frames=30,
    )
    scale = audit.validate_chunk_inventory(
        records, source_frames=source_frames, splits=splits
    )

    assert scale["records"] == 3
    assert scale["frames"] == 925
    assert scale["source_coverage_complete"] is True
    assert scale["source_overlap_frames"] == 0
    assert scale["by_fixed_split"]["train"]["frames"] == 620
    assert scale["by_fixed_split"]["test"]["frames"] == 305
    assert all(record["accepted_for_training"] is False for record in records)
    assert all(
        record["semantic_supervision_masks"] == audit.SEMANTIC_MASKS
        for record in records
    )
    forbidden = {
        "audio_relpath",
        "canonical_prompt",
        "prompt",
        "source_text",
        "window_transcript_context",
    }
    assert all(not forbidden.intersection(record) for record in records)


def test_event_summary_reconciles_declared_and_source_bounded_frames(tmp_path):
    inventory = tmp_path / "events.jsonl"
    records = [
        _event("1_wayne_0_1_1", 0, 40, task_id="valid"),
        _event("1_wayne_0_1_1", 80, 120, task_id="overrun"),
    ]
    inventory.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    summary = audit.event_inventory_summary(
        inventory, {"1_wayne_0_1_1": 105}
    )

    assert summary["summed_frames"] == 80
    assert summary["source_bounded_summed_frames"] == 65
    assert summary["union_frames"] == 65
    assert summary["overlap_hours"] == 0
    assert summary["out_of_bounds_record_count"] == 1
    assert summary["declared_excess_frames"] == 15
    assert summary["out_of_bounds_records"][0]["task_id"] == "overrun"
