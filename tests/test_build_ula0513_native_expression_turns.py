import csv
import hashlib
from pathlib import Path

from tools.human_motion_collection import build_ula0513_native_expression_turns as BUILD
from upper_body_skeleton.retarget_v2_18d import JOINT_ORDER_18D


def _write_source(path: Path, frame_count: int = 7) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_from_start", *JOINT_ORDER_18D, "ignored_finger"])
        for frame in range(frame_count):
            values = [0.0] * len(JOINT_ORDER_18D)
            values[3] = 0.02 * frame
            values[15] = 0.01 * frame
            writer.writerow([frame / 30.0, *values, 123.0])


def test_native_builder_preserves_complete_variable_length_and_native_head(tmp_path):
    archive = tmp_path / "0513csv.zip"
    archive.write_bytes(b"user archive fixture")
    source_root = tmp_path / "motion_viewer"
    _write_source(source_root / "Robot_Model0530_V2_GreetingGuest.csv")
    output = tmp_path / "catalog"
    processed = tmp_path / "processed"
    summary = BUILD.build_catalog(
        archive,
        source_root,
        output,
        processed,
        expected_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        expected_source_count=1,
    )
    assert summary["record_count"] == 1
    assert summary["frame_count_min"] == 7
    assert summary["legacy_fixed_150_frame_export_used"] is False
    record = BUILD.json.loads(
        (output / f"{BUILD.OUTPUT_STEM}.jsonl").read_text(encoding="utf-8")
    )
    assert record["training_segment"]["frame_count"] == 7
    assert record["training_segment"]["fixed_window_sec"] is None
    assert record["training_segment"]["cropped"] is False
    assert record["training_segment"]["resampled"] is False
    assert record["time_axes"]["output"]["sample_span_sec"] == 6 / 30
    assert record["time_axes"]["output"]["frame_coverage_sec"] == 7 / 30
    assert record["motion_18d"]["native_head_3dof_present"] is True
    assert record["motion_18d"]["head_mapping_or_synthesis_used"] is False
    with Path(record["motion_18d"]["safe_csv_path"]).open(
        newline="", encoding="utf-8"
    ) as handle:
        reader = csv.reader(handle)
        assert next(reader) == JOINT_ORDER_18D
        assert sum(1 for _row in reader) == 7


def test_source_asset_name_and_affect_remain_fail_closed(tmp_path):
    source = tmp_path / "Robot_Model0530_V2_Joy01.csv"
    _write_source(source)
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"fixture")
    record = BUILD.build_record(
        source,
        source_root=tmp_path,
        processed_root=tmp_path / "processed",
        archive_path=archive,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    assert record["source_behavior_label"] == "Joy01"
    assert record["canonical_action"] is None
    assert record["canonical_prompt"] is None
    assert not any(record["semantic_supervision_masks"].values())
    assert record["emotion_supervision_mask"] is False
    assert record["base_motion_eligible"] is False
    assert record["accepted_for_training"] is False


def test_timing_errors_fail_physical_qc_without_duration_cropping(tmp_path):
    source = tmp_path / "Robot_Model0530_V2_Hesitate.csv"
    _write_source(source, frame_count=5)
    rows = source.read_text(encoding="utf-8").splitlines()
    fields = rows[3].split(",")
    fields[0] = "0.4"
    rows[3] = ",".join(fields)
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"fixture")
    record = BUILD.build_record(
        source,
        source_root=tmp_path,
        processed_root=tmp_path / "processed",
        archive_path=archive,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    assert record["physical_qc"]["timing"]["native_30hz"] is False
    assert record["physical_qc"]["passed"] is False
    assert record["training_segment"]["frame_count"] == 5
    assert record["training_segment"]["fixed_window_sec"] is None


def test_small_limit_noise_is_projected_but_large_hardware_range_is_quarantined(
    tmp_path,
):
    source = tmp_path / "Robot_Model0530_V2_Search.csv"
    _write_source(source, frame_count=5)
    rows = list(csv.reader(source.open(newline="", encoding="utf-8")))
    pelvis_pitch = rows[0].index("joint_pelvisPitch")
    rows[2][pelvis_pitch] = "-0.005"
    with source.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"fixture")
    record = BUILD.build_record(
        source,
        source_root=tmp_path,
        processed_root=tmp_path / "processed",
        archive_path=archive,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    assert record["physical_qc"]["joint_limits_pass"] is False
    assert record["physical_qc"]["safe_projection_pass"] is True
    assert record["physical_qc"]["passed"] is True
    assert record["motion_18d"]["safety_projection_applied"] is True

    rows[2][pelvis_pitch] = "-0.5"
    with source.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    quarantined = BUILD.build_record(
        source,
        source_root=tmp_path,
        processed_root=tmp_path / "processed_large",
        archive_path=archive,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    assert quarantined["physical_qc"]["safe_projection_pass"] is False
    assert quarantined["physical_qc"]["passed"] is False
