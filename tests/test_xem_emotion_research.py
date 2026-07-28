import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
from scipy.io import savemat

import upper_body_skeleton.xem_emotion_research as xem


def _synthetic_cells(*, invalid_columns: bool = False) -> np.ndarray:
    cells = np.empty((5, 10, 20), dtype=object)
    for action in range(5):
        for participant in range(10):
            for example in range(20):
                frames = 7 + (action + participant + example) % 3
                time = np.arange(frames, dtype=np.float64) * 16.0
                matrix = np.zeros((frames, 277), dtype=np.float64)
                matrix[:, 0] = time
                phase = np.linspace(0.0, 1.0, frames)
                emotion = example // 5
                repetition = example % 5
                emotion_amplitude = float(emotion + 1)
                action_amplitude = float(action + 1) * 0.15
                participant_amplitude = float(participant + 1) * 0.01
                unique = (
                    action * 10_000
                    + participant * 1_000
                    + example * 10
                    + repetition
                )

                position = matrix[:, 1:70].reshape(frames, 23, 3)
                position[:, :, 0] = (
                    emotion_amplitude * phase[:, None]
                    + action_amplitude * np.sin(np.pi * phase)[:, None]
                )
                position[:, :, 1] = participant_amplitude * phase[:, None]
                position[0, 0, 2] = unique

                linear = matrix[:, 70:139].reshape(frames, 23, 3)
                linear[:, :, 0] = emotion_amplitude
                linear[:, :, 1] = action_amplitude
                linear[:, :, 2] = repetition * 0.02

                angular = matrix[:, 139:208].reshape(frames, 23, 3)
                angular[:, :, 0] = emotion_amplitude * 2.0
                angular[:, :, 1] = action_amplitude

                euler = matrix[:, 208:274].reshape(frames, 22, 3)
                euler[:, :, 0] = emotion_amplitude * phase[:, None] * 10.0
                euler[:, :, 1] = action_amplitude * phase[:, None] * 10.0

                center = matrix[:, 274:277]
                center[:, 0] = emotion_amplitude * phase
                center[:, 1] = action_amplitude * phase
                cells[action, participant, example] = matrix
    if invalid_columns:
        cells[0, 0, 0] = cells[0, 0, 0][:, :-1]
    return cells


def _write_archive(
    tmp_path: Path,
    *,
    invalid_columns: bool = False,
    extra_member: str | None = None,
) -> Path:
    mat_path = tmp_path / "Data.mat"
    savemat(
        mat_path,
        {"VelocityData": _synthetic_cells(invalid_columns=invalid_columns)},
        do_compression=True,
    )
    archive = tmp_path / "xem-dataset.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        output.writestr("matlab/", b"")
        output.write(mat_path, "matlab/Data.mat")
        output.writestr("matlab/README", b"synthetic XEM schema fixture")
        if extra_member is not None:
            output.writestr(extra_member, b"not allowed")
    return archive


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _metadata(path: Path, *, archive_bytes: int, archive_md5: str) -> Path:
    payload = {
        "status": "OK",
        "data": {
            "persistentUrl": "https://doi.org/10.57745/GZQCOY",
            "latestVersion": {
                "versionNumber": 1,
                "versionMinorNumber": 0,
                "versionState": "RELEASED",
                "datasetPersistentId": xem.OFFICIAL_DOI,
                "license": {"name": xem.OFFICIAL_LICENSE},
                "files": [
                    {
                        "dataFile": {
                            "filename": "xem-dataset.zip",
                            "persistentId": xem.OFFICIAL_FILE_DOI,
                            "filesize": archive_bytes,
                            "md5": archive_md5,
                            "checksum": {
                                "type": "MD5",
                                "value": archive_md5,
                            },
                        }
                    }
                ],
            },
        },
    }
    metadata = path / "dataverse_metadata.json"
    metadata.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return metadata


def _config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    invalid_columns: bool = False,
    extra_member: str | None = None,
) -> tuple[Path, dict]:
    archive = _write_archive(
        tmp_path,
        invalid_columns=invalid_columns,
        extra_member=extra_member,
    )
    archive_bytes = archive.stat().st_size
    archive_md5 = _md5(archive)
    monkeypatch.setattr(xem, "OFFICIAL_ARCHIVE_BYTES", archive_bytes)
    monkeypatch.setattr(xem, "OFFICIAL_ARCHIVE_MD5", archive_md5)
    metadata = _metadata(
        tmp_path,
        archive_bytes=archive_bytes,
        archive_md5=archive_md5,
    )
    output_root = tmp_path / "external_emotion_research" / "xem_baseline"
    payload = {
        "schema_version": xem.SCHEMA_VERSION,
        "artifact_kind": xem.CONFIG_KIND,
        "dataset": xem.DATASET,
        "research_scope": xem.RESEARCH_SCOPE,
        "archive_path": str(archive),
        "expected_archive_bytes": archive_bytes,
        "expected_archive_md5": archive_md5,
        "metadata_path": str(metadata),
        "expected_metadata_sha256": xem.sha256_file(metadata),
        "mat_member": "matlab/Data.mat",
        "mat_variable": "VelocityData",
        "participant_split": {
            split: list(xem.DEFAULT_PARTICIPANT_SPLIT[split])
            for split in xem.SPLITS
        },
        "ridge_alphas": [0.1, 1.0, 10.0],
        "seed": 17,
        "output_root": str(output_root),
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return config, {
        "archive": archive,
        "metadata": metadata,
        "output_root": output_root,
        "payload": payload,
    }


def test_full_pipeline_is_participant_disjoint_and_research_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config, paths = _config(tmp_path, monkeypatch)
    audit = xem.run_xem_research_pipeline(config)
    root = paths["output_root"]
    rows = xem.load_xem_inventory(root / "inventory.jsonl")
    report = json.loads((root / "baseline_report.json").read_text())

    assert len(rows) == 1_000
    assert audit["artifact_kind"] == xem.AUDIT_KIND
    assert audit["inventory"]["counts_by_split"] == {
        "test": 200,
        "train": 600,
        "validation": 200,
    }
    assert audit["inventory"]["counts_by_emotion"] == {
        "angry": 250,
        "happy": 250,
        "neutral": 250,
        "sad": 250,
    }
    assert audit["inventory"]["frame_count_summary"]["sum"] > 2_000
    assert audit["inventory"]["time_unit_policy"].startswith(
        "preserve_official_raw_values"
    )
    participant_splits: dict[str, set[str]] = {}
    for row in rows:
        participant_splits.setdefault(row["participant_id"], set()).add(
            row["fixed_split_assignment"]
        )
        assert row["production_training_eligible"] is False
        assert row["foundation_ingest_allowed"] is False
        assert row["generator_ingest_allowed"] is False
        assert row["human_confirmed_robot_observable"] is False
    assert all(len(splits) == 1 for splits in participant_splits.values())
    assert report["test_metrics"]["samples"] == 200
    assert report["validation_metrics"]["samples"] == 200
    assert report["preprocessing"]["standardization_statistics_source"] == (
        "train_participants_only"
    )
    assert report["label_limitations"][
        "labels_are_intended_performance_protocol_not_human_perception"
    ]
    assert report["generator_checkpoint_emitted"] is False
    assert audit["isolation"]["external_inputs_only"] is True
    assert audit["isolation"]["kimodo_input_count"] == 0
    assert audit["isolation"]["beat2_input_count"] == 0
    assert audit["source"]["sequence_column_contract_zero_based_half_open"][
        "joint_angular_velocity_23x3"
    ] == [139, 208]
    assert audit["source"]["config"]["sha256"] == xem.sha256_file(config)
    assert audit["software"]["classifier_dependency"] == "numpy_only"
    assert len(audit["implementation"]["module_sha256"]) == 64
    assert not (root / "generator_checkpoint.pt").exists()

    with np.load(root / "baseline_model.npz", allow_pickle=False) as model:
        assert not bool(model["generator_compatible"])
        assert str(model["research_scope"]) == xem.RESEARCH_SCOPE
    with np.load(root / "features.npz", allow_pickle=False) as features:
        assert features["features"].shape == (1_000, 464)
        assert features["feature_names"].shape == (464,)
        assert not bool(features["generator_compatible"])


def test_inspection_is_read_only_and_finds_official_cell_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config, paths = _config(tmp_path, monkeypatch)
    loaded = xem.load_config(config)
    receipt = xem.inspect_xem_archive(
        loaded["archive_path"],
        mat_member=loaded["mat_member"],
        mat_variable=loaded["mat_variable"],
    )

    assert receipt["archive"]["bytes"] == paths["archive"].stat().st_size
    assert receipt["selected_mat_variable"] == {
        "name": "VelocityData",
        "shape": [5, 10, 20],
        "class": "cell",
    }
    assert receipt["mat_member"]["name"] == "matlab/Data.mat"
    assert "no_persistent_mat_extraction" in receipt["read_policy"]
    assert not paths["output_root"].exists()


def test_participant_overlap_is_rejected_before_data_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config, paths = _config(tmp_path, monkeypatch)
    payload = paths["payload"]
    payload["participant_split"]["validation"] = [6, 7]
    payload["participant_split"]["train"] = [1, 2, 3, 4, 5, 6]
    config.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="crosses"):
        xem.load_config(config)


def test_invalid_sequence_column_count_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config, _ = _config(tmp_path, monkeypatch, invalid_columns=True)
    with pytest.raises(ValueError, match="expected 277 columns"):
        xem.run_xem_research_pipeline(config)


def test_archive_hash_mismatch_fails_before_zip_or_mat_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config, paths = _config(tmp_path, monkeypatch)
    archive = paths["archive"]
    with archive.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="incomplete XEM archive"):
        xem.run_xem_research_pipeline(config)


def test_unapproved_dataset_marker_in_archive_member_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    marker = "ki" + "modo"
    config, paths = _config(
        tmp_path,
        monkeypatch,
        extra_member=f"external/{marker}/cache.bin",
    )
    # Avoid putting the forbidden token in the config path itself; it exists
    # only inside the ZIP central directory.
    assert marker not in str(paths["archive"]).casefold()
    with pytest.raises(ValueError, match="forbidden dataset marker"):
        xem.run_xem_research_pipeline(config)


def test_research_output_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config, paths = _config(tmp_path, monkeypatch)
    paths["output_root"].mkdir(parents=True)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        xem.load_config(config)
