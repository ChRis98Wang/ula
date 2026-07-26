import csv
import hashlib
import json
import os
from pathlib import Path

import pytest

from tools.human_motion_review.adjudicate_training_dataset import (
    EXPECTED_18D_JOINT_ORDER,
    REQUIRED_18D_GATES,
)
from tools.human_motion_review.summarize_full_18d_retarget import (
    PASSED_MANIFEST,
    REJECTED_MANIFEST,
    discover_quality_candidates,
    select_latest_matching_candidate,
    summarize_full_retarget,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_inventory(root: Path, clip_ids):
    catalog = root / "catalog"
    raw = root / "raw"
    catalog.mkdir(parents=True)
    raw.mkdir(parents=True)
    rows = []
    for clip_id in clip_ids:
        source = raw / f"{clip_id}.npy"
        source.write_bytes(f"motion:{clip_id}".encode())
        action = clip_id.split("_")[0]
        rows.append(
            {
                "clip_id": clip_id,
                "action": action,
                "canonical_prompt_en": f"Perform {action}.",
                "robot_contract": "ula_v2_18d_head_v1",
                "motion_relpath": str(source.relative_to(root)),
                "frame_count": "3",
                "nominal_fps": "30.0",
                "motion_sha256": _sha256(source),
                "review_state": "machine_labeled_pending_review",
                "manual_review_required": "True",
                "accepted_for_training": "False",
            }
        )
    path = catalog / "inventory.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path, {row["clip_id"]: row for row in rows}


def _motion_rows():
    rows = []
    for index in range(3):
        row = [0.01 * index] * 15
        row.extend([0.1 * index, -0.1 * index, 0.05 * index])
        rows.append(row)
    return rows


def _write_quality(
    root: Path,
    dataset_root: Path,
    inventory_row: dict,
    *,
    passed=True,
    mtime_ns=1_000_000_000,
    malformed_header=False,
    source_override=None,
):
    clip_id = inventory_row["clip_id"]
    sample_dir = root / clip_id
    sample_dir.mkdir(parents=True)
    csv_path = sample_dir / f"{clip_id}_gmr_safe_18d.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = EXPECTED_18D_JOINT_ORDER[:-1] if malformed_header else EXPECTED_18D_JOINT_ORDER
        writer.writerow(header)
        for row in _motion_rows():
            writer.writerow(row[: len(header)])
    gates = {gate: True for gate in REQUIRED_18D_GATES}
    if not passed:
        gates["target_fit_pass"] = False
        gates["passed"] = False
    source = dataset_root / inventory_row["motion_relpath"]
    quality = {
        "source_motionx": str(source_override or source),
        "source_sha256": inventory_row["motion_sha256"],
        "source_frames": 3,
        "source_fps": 30,
        "frames": 3,
        "fps": 30,
        "duration_sec": 0.1,
        "retime_factor": 1.0,
        "output_contract": "ula_v2_18d_head_v1",
        "action_dim": 18,
        "joint_order": EXPECTED_18D_JOINT_ORDER,
        "head_joint_order": EXPECTED_18D_JOINT_ORDER[-3:],
        "head_safe_max_velocity_rad_s": 3.0,
        "joint_ranges": {
            "head_roll_joint": {"min_rad": 0.0, "max_rad": 0.2, "range_deg": 11.4591559026},
            "head_pitch_joint": {"min_rad": -0.2, "max_rad": 0.0, "range_deg": 11.4591559026},
            "head_yaw_joint": {"min_rad": 0.0, "max_rad": 0.1, "range_deg": 5.7295779513},
        },
        "quality_gate": gates,
    }
    quality_path = sample_dir / "quality.json"
    quality_path.write_text(json.dumps(quality, sort_keys=True), encoding="utf-8")
    os.utime(quality_path, ns=(mtime_ns, mtime_ns))
    return quality_path


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _fixture(tmp_path):
    clip_ids = ["wave_1_clip1", "wave_2_clip1", "wave_3_clip1", "wave_4_clip1"]
    inventory_path, rows = _write_inventory(tmp_path, clip_ids)
    passed_root = tmp_path / "passed"
    rejected_root = tmp_path / "rejected"
    _write_quality(passed_root, tmp_path, rows["wave_1_clip1"], mtime_ns=200)
    _write_quality(
        rejected_root / "20260101_000000",
        tmp_path,
        rows["wave_2_clip1"],
        passed=False,
        mtime_ns=200,
    )
    _write_quality(
        rejected_root / "20260101_000000",
        tmp_path,
        rows["wave_3_clip1"],
        passed=False,
        mtime_ns=100,
    )
    _write_quality(passed_root, tmp_path, rows["wave_3_clip1"], mtime_ns=300)
    return inventory_path, rows, passed_root, rejected_root


def test_full_summary_selects_latest_and_never_grants_training_admission(tmp_path):
    inventory, _, passed_root, rejected_root = _fixture(tmp_path)
    output = tmp_path / "output"
    report = summarize_full_retarget(
        inventory,
        passed_root,
        rejected_root,
        output,
        dataset_root=tmp_path,
        expected_count=4,
    )

    assert report["counts"] == {"inventory": 4, "passed": 2, "rejected": 1, "missing": 1}
    assert report["completeness"]["equation_holds"] is True
    assert report["completeness"]["passed"] is False
    passed = _read_jsonl(output / PASSED_MANIFEST)
    rejected = _read_jsonl(output / REJECTED_MANIFEST)
    assert [record["clip_id"] for record in passed] == ["wave_1_clip1", "wave_3_clip1"]
    assert [record["clip_id"] for record in rejected] == ["wave_2_clip1"]
    assert all(
        record["training_admission"] == {
            "accepted_for_training": False,
            "inventory_review_state": "machine_labeled_pending_review",
            "manual_review_required": True,
            "rule": "retarget QC never grants semantic training admission",
            "state": "pending_semantic_review",
        }
        for record in passed + rejected
    )
    retry = next(record for record in passed if record["clip_id"] == "wave_3_clip1")
    assert retry["quality"]["selection"]["matching_candidate_count"] == 2
    assert len(retry["quality"]["selection"]["superseded_quality_paths"]) == 1
    assert report["failed_gate_counts"] == {"passed": 1, "target_fit_pass": 1}
    assert report["head_distributions"]["all_selected"]["head_roll_joint.max_abs_rad"][
        "count"
    ] == 3


def test_bad_csv_and_source_link_are_hard_rejections(tmp_path):
    clip_ids = ["wave_1_clip1", "wave_2_clip1"]
    inventory, rows = _write_inventory(tmp_path, clip_ids)
    passed_root = tmp_path / "passed"
    rejected_root = tmp_path / "rejected"
    _write_quality(
        passed_root,
        tmp_path,
        rows["wave_1_clip1"],
        malformed_header=True,
    )
    _write_quality(
        passed_root,
        tmp_path,
        rows["wave_2_clip1"],
        source_override=tmp_path / "raw/other.npy",
    )
    output = tmp_path / "output"
    report = summarize_full_retarget(
        inventory,
        passed_root,
        rejected_root,
        output,
        dataset_root=tmp_path,
        expected_count=2,
    )
    assert report["counts"] == {"inventory": 2, "passed": 0, "rejected": 2, "missing": 0}
    by_id = {record["clip_id"]: record for record in _read_jsonl(output / REJECTED_MANIFEST)}
    assert "safe_csv_header_mismatch" in by_id["wave_1_clip1"]["quality"]["reasons"]
    assert "safe_csv_row_width_mismatch" in by_id["wave_1_clip1"]["quality"]["reasons"]
    assert "source_path_mismatch" in by_id["wave_2_clip1"]["quality"]["reasons"]


def test_identical_latest_mtime_is_rejected_as_ambiguous(tmp_path):
    inventory, rows = _write_inventory(tmp_path, ["wave_1_clip1"])
    passed_root = tmp_path / "passed"
    rejected_root = tmp_path / "rejected"
    _write_quality(passed_root, tmp_path, rows["wave_1_clip1"], mtime_ns=123)
    _write_quality(
        rejected_root / "20260101_000000",
        tmp_path,
        rows["wave_1_clip1"],
        passed=False,
        mtime_ns=123,
    )
    candidates, _ = discover_quality_candidates(
        passed_root,
        rejected_root,
        {"wave_1_clip1"},
    )
    with pytest.raises(ValueError, match="ambiguous latest quality evidence"):
        select_latest_matching_candidate(
            candidates["wave_1_clip1"],
            expected_contract="ula_v2_18d_head_v1",
        )
    assert inventory.is_file()


def test_complete_finished_batch_and_byte_reproducibility(tmp_path):
    inventory, rows, passed_root, rejected_root = _fixture(tmp_path)
    _write_quality(passed_root, tmp_path, rows["wave_4_clip1"], mtime_ns=400)
    model = tmp_path / "smplx_model.npz"
    model.write_bytes(b"fixture-smplx-model")
    model_sha256 = _sha256(model)
    selected_dirs = {
        "wave_1_clip1": passed_root / "wave_1_clip1",
        "wave_2_clip1": rejected_root / "20260101_000000" / "wave_2_clip1",
        "wave_3_clip1": passed_root / "wave_3_clip1",
        "wave_4_clip1": passed_root / "wave_4_clip1",
    }
    summary = {
        "batch_id": "20260101_000001",
        "output_contract": "ula_v2_18d_head_v1",
        "total_tasks": 4,
        "completed_tasks": 4,
        "finished": True,
        "counts": {"passed": 3, "quality_failed": 1},
        "manifest": str(inventory),
        "output_root": str(passed_root),
        "rejected_root": str(rejected_root),
        "smplx_model": str(model),
        "smplx_model_sha_check": "skipped",
        "results": [
            {
                "clip_id": clip_id,
                "action": row["action"],
                "source": str(tmp_path / row["motion_relpath"]),
                "status": "quality_failed" if clip_id == "wave_2_clip1" else "passed",
                "output_dir": str(selected_dirs[clip_id]),
            }
            for clip_id, row in sorted(rows.items())
        ],
    }
    (passed_root / "_batch_18d_head_v1_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    output = tmp_path / "output"
    first_report = summarize_full_retarget(
        inventory,
        passed_root,
        rejected_root,
        output,
        dataset_root=tmp_path,
        smplx_model_path=model,
        expected_smplx_sha256=model_sha256,
        expected_count=4,
    )
    assert first_report["completeness"]["passed"] is True
    provenance = first_report["inputs"]["smplx_model_provenance"]
    assert provenance["verification_scope"] == "verified_once_out_of_band_for_batch"
    assert provenance["quality_reports_per_clip_sha256_verified"] is False
    assert provenance["sha256_matches_expected"] is True
    names = [PASSED_MANIFEST, REJECTED_MANIFEST, "full_retarget_scale_report.json"]
    first = {name: (output / name).read_bytes() for name in names}
    second_report = summarize_full_retarget(
        inventory,
        passed_root,
        rejected_root,
        output,
        dataset_root=tmp_path,
        smplx_model_path=model,
        expected_smplx_sha256=model_sha256,
        expected_count=4,
    )
    second = {name: (output / name).read_bytes() for name in names}
    assert second_report["completeness"]["passed"] is True
    assert first == second

    original_output_dir = summary["results"][0]["output_dir"]
    summary["results"][0]["output_dir"] = str(tmp_path / "stale_output")
    (passed_root / "_batch_18d_head_v1_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    stale_output = summarize_full_retarget(
        inventory,
        passed_root,
        rejected_root,
        output,
        dataset_root=tmp_path,
        smplx_model_path=model,
        expected_smplx_sha256=model_sha256,
        expected_count=4,
    )
    assert stale_output["completeness"]["passed"] is False
    assert stale_output["completeness"][
        "batch_result_output_dirs_match_selected_quality"
    ] is False
    assert stale_output["association_checks"][
        "batch_result_output_dir_mismatch_count"
    ] == 1
    summary["results"][0]["output_dir"] = original_output_dir

    summary["counts"] = {"passed": 2, "quality_failed": 2}
    (passed_root / "_batch_18d_head_v1_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    wrong_counts = summarize_full_retarget(
        inventory,
        passed_root,
        rejected_root,
        output,
        dataset_root=tmp_path,
        smplx_model_path=model,
        expected_smplx_sha256=model_sha256,
        expected_count=4,
    )
    assert wrong_counts["completeness"]["passed"] is False
    assert wrong_counts["completeness"]["batch_qc_counts_match_manifests"] is False
    summary["counts"] = {"passed": 3, "quality_failed": 1}
    (passed_root / "_batch_18d_head_v1_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )

    wrong_model_hash = summarize_full_retarget(
        inventory,
        passed_root,
        rejected_root,
        output,
        dataset_root=tmp_path,
        smplx_model_path=model,
        expected_smplx_sha256="0" * 64,
        expected_count=4,
    )
    assert wrong_model_hash["completeness"]["passed"] is False
    assert wrong_model_hash["completeness"]["model_provenance_pass"] is False

    summary["results"][0]["action"] = "wrong_action"
    (passed_root / "_batch_18d_head_v1_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    mismatched = summarize_full_retarget(
        inventory,
        passed_root,
        rejected_root,
        output,
        dataset_root=tmp_path,
        smplx_model_path=model,
        expected_smplx_sha256=model_sha256,
        expected_count=4,
    )
    assert mismatched["completeness"]["passed"] is False
    assert mismatched["completeness"]["batch_result_associations_pass"] is False
    assert mismatched["association_checks"]["batch_result_action_mismatch_count"] == 1
