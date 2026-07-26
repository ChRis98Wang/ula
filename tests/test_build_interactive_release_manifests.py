import json
from pathlib import Path

from tools.human_motion_collection.build_interactive_release_manifests import (
    build_release,
)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def beat_record(clip_id: str, issues: list[str]) -> dict:
    return {
        "clip_id": clip_id,
        "fps": 30.0,
        "issues": issues,
        "motion_relpath": f"smplxflame_30/{clip_id}.npz",
        "audio_relpath": f"wave16k/{clip_id}.wav",
        "textgrid_relpath": f"textgrid/{clip_id}.TextGrid",
        "transcript_relpath": f"text/{clip_id}.txt",
        "official_split": "train",
        "speaker_key": "12_zhao",
        "window_transcript_context": "测试语境",
        "window": {
            "selection_status": "selected_nonstatic_low_dynamic",
            "start_frame": 30,
            "end_frame_exclusive": 210,
            "duration_sec": 6.0,
        },
    }


def haa_pending(clip_id: str, state: str) -> dict:
    return {
        "clip_id": clip_id,
        "canonical_prompt_en": "Bow toward the other person.",
        "context_dependency": "none",
        "communicative_intent": "respectful_bow",
        "candidate_state": state,
        "trajectory": {
            "path": f"/processed/{clip_id}.csv",
            "fps": 30.0,
            "frames": 45,
            "duration_sec": 1.5,
        },
    }


def test_build_release_denies_all_unreviewed_records_and_separates_blockers(tmp_path):
    beat = tmp_path / "beat.jsonl"
    primitive = tmp_path / "primitive.jsonl"
    offer = tmp_path / "offer.jsonl"
    excluded = tmp_path / "excluded.jsonl"
    write_jsonl(
        beat,
        [beat_record("clean", []), beat_record("bad", ["missing_textgrid_alignment"])],
    )
    write_jsonl(
        primitive,
        [haa_pending("bow", "communicative_primitive_pending_video_review")],
    )
    write_jsonl(
        offer,
        [haa_pending("high_five", "partner_conditioned_probe_pending_review")],
    )
    write_jsonl(
        excluded,
        [
            {
                "clip_id": "boxing",
                "source_action": "boxing",
                "exclusion_reasons": ["not_in_interactive_ontology"],
            }
        ],
    )

    output = tmp_path / "out"
    summary = build_release(
        beat2_inventory=beat,
        beat2_root=tmp_path / "beat2",
        haa_primitives=primitive,
        haa_partner_offers=offer,
        haa_excluded=excluded,
        output_dir=output,
    )

    assert read_jsonl(output / "train_ready.jsonl") == []
    pending = read_jsonl(output / "pending_review.jsonl")
    rejected = read_jsonl(output / "rejected.jsonl")
    assert {record["record_id"] for record in pending} == {
        "beat2:clean",
        "haa500:bow",
        "haa500:high_five",
    }
    assert {record["record_id"] for record in rejected} == {
        "beat2:bad",
        "haa500:boxing",
    }
    assert all(record["accepted_for_training"] is False for record in pending + rejected)
    clean = next(record for record in pending if record["record_id"] == "beat2:clean")
    assert clean["conditioning_text_status"].endswith("not_approved_action_prompt")
    assert clean["motion_path"].endswith("smplxflame_30/clean.npz")
    assert summary["high_dynamic_actions_included"] is False


def test_fallback_dynamic_beat_window_is_rejected(tmp_path):
    record = beat_record("fallback", [])
    record["window"]["selection_status"] = "fallback_no_non_high_dynamic_window"
    beat = tmp_path / "beat.jsonl"
    empty = tmp_path / "empty.jsonl"
    write_jsonl(beat, [record])
    write_jsonl(empty, [])

    build_release(
        beat2_inventory=beat,
        beat2_root=tmp_path,
        haa_primitives=empty,
        haa_partner_offers=empty,
        haa_excluded=empty,
        output_dir=tmp_path / "out",
    )

    rejected = read_jsonl(tmp_path / "out/rejected.jsonl")
    assert rejected[0]["decision_reasons"] == ["not_selected_nonstatic_low_dynamic"]


def test_low_dynamic_window_with_aligned_speech_is_pending(tmp_path):
    record = beat_record("aligned", [])
    record["window"]["selection_status"] = (
        "selected_nonstatic_low_dynamic_with_aligned_speech"
    )
    beat = tmp_path / "beat.jsonl"
    empty = tmp_path / "empty.jsonl"
    write_jsonl(beat, [record])
    write_jsonl(empty, [])

    build_release(
        beat2_inventory=beat,
        beat2_root=tmp_path,
        haa_primitives=empty,
        haa_partner_offers=empty,
        haa_excluded=empty,
        output_dir=tmp_path / "out",
    )

    pending = read_jsonl(tmp_path / "out/pending_review.jsonl")
    assert [record["record_id"] for record in pending] == ["beat2:aligned"]


def test_clip_level_alignment_warnings_do_not_reject_a_valid_window(tmp_path):
    record = beat_record(
        "warning",
        [
            "motion_audio_duration_mismatch_gt_0_3s",
            "textgrid_transcript_mismatch",
        ],
    )
    beat = tmp_path / "beat.jsonl"
    empty = tmp_path / "empty.jsonl"
    write_jsonl(beat, [record])
    write_jsonl(empty, [])

    build_release(
        beat2_inventory=beat,
        beat2_root=tmp_path,
        haa_primitives=empty,
        haa_partner_offers=empty,
        haa_excluded=empty,
        output_dir=tmp_path / "out",
    )

    pending = read_jsonl(tmp_path / "out/pending_review.jsonl")
    assert pending[0]["record_id"] == "beat2:warning"
    assert pending[0]["source_warnings"] == [
        "motion_audio_duration_mismatch_gt_0_3s",
        "textgrid_transcript_mismatch",
    ]
