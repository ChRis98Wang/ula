import csv
import importlib.util
import json
from pathlib import Path

from jsonschema import Draft202012Validator


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools/human_motion_collection/build_expression_semantics_v1.py"
)
SPEC = importlib.util.spec_from_file_location("build_expression_semantics_v1", SCRIPT_PATH)
assert SPEC and SPEC.loader
SEMANTICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SEMANTICS)

OUTPUT_STEM = SEMANTICS.OUTPUT_STEM
action_from_stem = SEMANTICS.action_from_stem
build_and_write = SEMANTICS.build_and_write
semantics_schema = SEMANTICS.semantics_schema


def write_fixture(root):
    motion_root = root / "raw/Motion-Xplusplus/extracted/haa500/smplx322"
    label_root = root / "raw/Motion-Xplusplus/extracted/haa500/semantic_label"
    catalog = root / "catalog"
    motion_root.mkdir(parents=True)
    label_root.mkdir(parents=True)
    catalog.mkdir(parents=True)
    rows = [
        ("applauding_1_clip1", "applauding", "Someone rides the scooter."),
        ("hailing_taxi_1_clip1", "hailing_taxi", "Sorry, I cannot provide that information."),
        (
            "arm_wave_1_clip1",
            "arm_wave",
            "The frames are identical and show no visible movement.",
        ),
        (
            "salute_1_clip1",
            "salute",
            "The person raises the right hand to the forehead in a salute.",
        ),
    ]
    manifest_rows = []
    for clip_id, action, text in rows:
        motion = motion_root / f"{clip_id}.npy"
        label = label_root / f"{clip_id}.txt"
        motion.write_bytes(f"motion:{clip_id}".encode())
        label.write_text(text, encoding="utf-8")
        manifest_rows.append(
            {
                "clip_id": clip_id,
                "action": action,
                "motion_relpath": str(motion.relative_to(root)),
                "label_relpath": str(label.relative_to(root)),
                "semantic_label": text,
            }
        )
    # This full-dataset label makes the applauding text a cross-action duplicate.
    (label_root / "ride_scooter_2_clip1.txt").write_text(
        "Someone rides the scooter.", encoding="utf-8"
    )
    manifest = catalog / "expression_candidates_haa500.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    return manifest


def test_action_from_filename_is_primary_label():
    assert action_from_stem("bowing_waist_12_clip1") == "bowing_waist"
    assert action_from_stem("arm_wave_clip1") == "arm_wave"


def test_semantics_flags_generated_text_and_never_claims_manual_verification(tmp_path):
    manifest = write_fixture(tmp_path)

    summary, records = build_and_write(tmp_path, manifest)
    by_clip = {record["clip_id"]: record for record in records}

    applause = by_clip["applauding_1_clip1"]
    assert applause["canonical_action"] == "applauding"
    assert applause["provenance"]["primary_label_source"] == "filename_action"
    assert "cross_action_duplicate" in applause["source_text_quality"]["flags"]
    assert "canonical_action_text_conflict" in applause["source_text_quality"]["flags"]
    assert applause["source_text_quality"]["conditioning_eligible"] is False

    refusal = by_clip["hailing_taxi_1_clip1"]
    assert "refusal" in refusal["source_text_quality"]["flags"]
    assert refusal["confidence"]["source_text_alignment"] == 0.0

    vague = by_clip["arm_wave_1_clip1"]
    assert "vague_or_non_observable" in vague["source_text_quality"]["flags"]

    clean = by_clip["salute_1_clip1"]
    assert clean["source_text_quality"]["flags"] == []
    assert clean["review_status"]["state"] == "machine_labeled_pending_review"
    assert clean["review_status"]["accepted_for_training"] is False

    serialized = json.dumps(records, ensure_ascii=False)
    assert "human_verified" not in serialized
    assert summary["conditioning_eligible_count"] == 0
    assert summary["accepted_for_training_count"] == 0


def test_records_validate_and_outputs_are_byte_reproducible(tmp_path):
    manifest = write_fixture(tmp_path)
    _, records = build_and_write(tmp_path, manifest)
    validator = Draft202012Validator(semantics_schema())
    for record in records:
        validator.validate(record)

    catalog = tmp_path / "catalog"
    paths = [
        catalog / f"{OUTPUT_STEM}.schema.json",
        catalog / f"{OUTPUT_STEM}.jsonl",
        catalog / f"{OUTPUT_STEM}.csv",
        catalog / f"{OUTPUT_STEM}.summary.json",
    ]
    first = {path.name: path.read_bytes() for path in paths}
    build_and_write(tmp_path, manifest)
    second = {path.name: path.read_bytes() for path in paths}

    assert first == second
    assert len((catalog / f"{OUTPUT_STEM}.jsonl").read_text().splitlines()) == 4
