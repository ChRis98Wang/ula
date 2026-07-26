import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools/human_motion_collection/build_haa500_full_inventory.py"
)
SPEC = importlib.util.spec_from_file_location("build_haa500_full_inventory", SCRIPT_PATH)
assert SPEC and SPEC.loader
INVENTORY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INVENTORY)

action_from_stem = INVENTORY.action_from_stem
observable_risks = INVENTORY.observable_risks
readable_action = INVENTORY.readable_action


def write_fixture(root, matching_labels=True):
    motion_root = root / "raw/Motion-Xplusplus/extracted/haa500/smplx322"
    label_root = root / "raw/Motion-Xplusplus/extracted/haa500/semantic_label"
    motion_root.mkdir(parents=True)
    label_root.mkdir(parents=True)
    for index in (1, 2):
        clip_id = f"arm_wave_{index}_clip1"
        np.save(motion_root / f"{clip_id}.npy", np.zeros((1, 322), dtype=np.float32))
        if matching_labels:
            (label_root / f"{clip_id}.txt").write_text(
                "A person moves both arms in a wave.", encoding="utf-8"
            )


def test_action_from_stem_handles_numbered_and_unnumbered_clips():
    assert action_from_stem("salute_11_clip1") == "salute"
    assert action_from_stem("add_new_car_tire_clip1") == "add_new_car_tire"
    assert action_from_stem("add_new_car_tire_15_clip2") == "add_new_car_tire"


def test_observability_flags_missing_robot_channels():
    assert "fixed_base_discards_lower_body_or_root_motion" in observable_risks("backflip")
    assert "face_finger_or_fine_hand_cues_unavailable" in observable_risks("blowing_kisses")
    assert "interaction_partner_or_contact_unavailable" in observable_risks("hugging_human")
    assert "scene_object_unavailable" in observable_risks("air_guitar")


def test_readable_action_is_deterministic():
    assert readable_action("ALS_IceBucket_Challenge") == "ALS icebucket challenge"


def test_vague_text_detects_static_frame_wording():
    assert INVENTORY.VAGUE.search(
        "The frames are identical and show no visible movement."
    )


def test_inventory_marks_machine_prompt_and_uses_exact_total_duration(tmp_path):
    write_fixture(tmp_path)

    summary = INVENTORY.build_inventory(tmp_path, include_motion_hash=False)
    catalog = tmp_path / "catalog"
    with (catalog / f"{INVENTORY.OUTPUT_STEM}.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        csv_rows = list(csv.DictReader(handle))
    records = [
        json.loads(line)
        for line in (catalog / f"{INVENTORY.OUTPUT_STEM}.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert summary["frame_count"] == 2
    assert summary["duration_sec"] == round(2 / 30.0, 6)
    assert summary["canonical_prompt_policy"] == "deterministic_filename_action_template"
    assert all(
        row["canonical_prompt_source"] == "deterministic_filename_action_template"
        for row in csv_rows
    )
    assert all(record["primary_label_source"] == "filename_action" for record in records)
    assert all(record["source_text_role"] == "gpt4v_auxiliary_only" for record in records)
    assert all(record["accepted_for_training"] is False for record in records)
    assert "human_verified" not in json.dumps(records)


def test_inventory_rejects_motion_label_stem_mismatch(tmp_path):
    write_fixture(tmp_path, matching_labels=False)

    with pytest.raises(ValueError, match="motion/label stem mismatch"):
        INVENTORY.build_inventory(tmp_path, include_motion_hash=False)


def test_inventory_reuses_only_fully_validated_hashes(tmp_path, monkeypatch):
    write_fixture(tmp_path)
    INVENTORY.build_inventory(tmp_path)
    prior_jsonl = tmp_path / "catalog" / f"{INVENTORY.OUTPUT_STEM}.jsonl"
    prior_records = [
        json.loads(line) for line in prior_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    prior_hashes = {
        record["clip_id"]: (record["motion_sha256"], record["label_sha256"])
        for record in prior_records
    }

    def unexpected_hash(_path):
        raise AssertionError("source file hashing must not run during validated reuse")

    monkeypatch.setattr(INVENTORY, "sha256_file", unexpected_hash)
    summary = INVENTORY.build_inventory(
        tmp_path,
        reuse_hashes_from=prior_jsonl,
    )
    rebuilt_records = [
        json.loads(line) for line in prior_jsonl.read_text(encoding="utf-8").splitlines()
    ]

    assert summary["hash_policy"] == "reused_after_exact_clip_path_revision_validation"
    assert summary["reused_motion_hash_count"] == 2
    assert summary["reused_label_hash_count"] == 2
    assert all(
        (record["motion_sha256"], record["label_sha256"])
        == prior_hashes[record["clip_id"]]
        for record in rebuilt_records
    )


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("motion_relpath", "raw/wrong.npy", "motion path mismatch"),
        ("label_relpath", "raw/wrong.txt", "label path mismatch"),
        ("motion_sha256", "not-a-sha", "Invalid reusable motion SHA256"),
        ("label_sha256", "not-a-sha", "Invalid reusable label SHA256"),
        ("dataset_revision", "wrong-revision", "dataset revision mismatch"),
    ],
)
def test_hash_reuse_rejects_unsafe_prior_records(
    tmp_path, field, bad_value, message
):
    write_fixture(tmp_path)
    INVENTORY.build_inventory(tmp_path)
    prior_jsonl = tmp_path / "catalog" / f"{INVENTORY.OUTPUT_STEM}.jsonl"
    records = [
        json.loads(line) for line in prior_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    records[0][field] = bad_value
    prior_jsonl.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=message):
        INVENTORY.build_inventory(tmp_path, reuse_hashes_from=prior_jsonl)


def test_hash_reuse_rejects_incomplete_clip_set(tmp_path):
    write_fixture(tmp_path)
    INVENTORY.build_inventory(tmp_path)
    prior_jsonl = tmp_path / "catalog" / f"{INVENTORY.OUTPUT_STEM}.jsonl"
    records = prior_jsonl.read_text(encoding="utf-8").splitlines()
    prior_jsonl.write_text(records[0] + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Hash reuse clip set mismatch"):
        INVENTORY.build_inventory(tmp_path, reuse_hashes_from=prior_jsonl)
