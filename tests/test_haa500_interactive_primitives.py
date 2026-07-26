import csv
import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tools/human_motion_collection/build_haa500_interactive_primitives.py"
)
SPEC = importlib.util.spec_from_file_location("build_haa500_interactive_primitives", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _inventory(path):
    rows = [
        ("applauding_1_clip1", "applauding", ""),
        ("hailing_taxi_1_clip1", "hailing_taxi", ""),
        ("salute_1_clip1", "salute", "cross_action_duplicate"),
        ("fist_bump_1_clip1", "fist_bump", ""),
        ("high_five_1_clip1", "high_five", ""),
        ("arm_wave_1_clip1", "arm_wave", ""),
        ("backflip_1_clip1", "backflip", ""),
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["clip_id", "action", "source_text_flags", "dataset_revision"],
        )
        writer.writeheader()
        for clip_id, action, flags in rows:
            writer.writerow(
                {
                    "clip_id": clip_id,
                    "action": action,
                    "source_text_flags": flags,
                    "dataset_revision": "fixture-revision",
                }
            )
    return rows


def _physical(clip_id):
    return {
        "clip_id": clip_id,
        "retarget_status": "retarget_qc_passed",
        "source": {"path": f"/raw/{clip_id}.npy", "sha256": "1" * 64},
        "output_18d": {
            "frames": 30,
            "fps": 30,
            "duration_sec": 1.0,
            "quality_json": f"/processed/{clip_id}/quality.json",
            "quality_sha256": "2" * 64,
            "safe_csv": {
                "path": f"/processed/{clip_id}/{clip_id}_gmr_safe_18d.csv",
                "sha256": "3" * 64,
            },
        },
    }


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_conservative_partition_never_grants_training_admission(tmp_path):
    inventory = tmp_path / "inventory.csv"
    rows = _inventory(inventory)
    passed = tmp_path / "passed.jsonl"
    passing_ids = [row[0] for row in rows if not row[0].startswith("high_five")]
    passed.write_text(
        "".join(json.dumps(_physical(clip_id)) + "\n" for clip_id in passing_ids),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    summary = MODULE.build_subset(inventory, passed, output)

    assert summary["counts"] == {
        "inventory": 7,
        "physical_pass_pool": 6,
        "communicative_primitives_pending_review": 2,
        "partner_offers_needs_review": 1,
        "train_ready": 0,
        "excluded": 4,
    }
    primitives = _read_jsonl(output / "communicative_primitives_pending_review.jsonl")
    assert {record["communicative_intent"] for record in primitives} == {
        "positive_feedback_applause",
        "raise_hand_to_get_attention",
    }
    attention = next(
        record for record in primitives if record["source_action"] == "hailing_taxi"
    )
    assert "taxi" not in attention["canonical_prompt_en"].lower()
    partner = _read_jsonl(output / "partner_offers_needs_review.jsonl")
    assert partner[0]["context_dependency"] == "partner_target_required"
    assert all(
        record["semantic_review"]["accepted_for_training"] is False
        for record in primitives + partner
    )
    assert (output / "train_ready.jsonl").read_bytes() == b""


def test_exclusions_record_physical_text_and_ontology_failures(tmp_path):
    inventory = tmp_path / "inventory.csv"
    _inventory(inventory)
    passed = tmp_path / "passed.jsonl"
    passed.write_text(
        json.dumps(_physical("arm_wave_1_clip1")) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"
    MODULE.build_subset(inventory, passed, output)
    excluded = {record["clip_id"]: record for record in _read_jsonl(output / "excluded.jsonl")}

    assert any(
        reason.startswith("known_hard_exclusion:")
        for reason in excluded["arm_wave_1_clip1"]["exclusion_reasons"]
    )
    assert "retarget_qc_not_passed" in excluded["high_five_1_clip1"]["exclusion_reasons"]
    assert (
        "source_text_risk:cross_action_duplicate"
        in excluded["salute_1_clip1"]["exclusion_reasons"]
    )
    assert all(record["accepted_for_training"] is False for record in excluded.values())
