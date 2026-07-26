import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.human_motion_collection import acquire_interact_motion_only as ACQUIRE


def _git_blob_sha1(payload: bytes) -> str:
    return hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()


def _row(path: str, payload: bytes) -> dict:
    return {
        "path": path,
        "size": len(payload),
        "remote_git_blob_oid_sha1": _git_blob_sha1(payload),
        "remote_lfs_content_sha256": None,
    }


def test_selection_excludes_audio_face_and_renders():
    assert ACQUIRE.selected_remote_path("bvhs/20231119_001_051.bvh")
    assert ACQUIRE.selected_remote_path("actors.db")
    assert not ACQUIRE.selected_remote_path("wav/20231119_001_051.wav")
    assert not ACQUIRE.selected_remote_path("face_arkit/20231119_001_051.npy")
    assert not ACQUIRE.selected_remote_path("body_renders/20231119_051.mp4")


def test_remote_inventory_is_revision_and_count_pinned():
    siblings = []
    for index in range(ACQUIRE.EXPECTED_BVH_COUNT):
        date = f"{20230000 + index // 1000 + 10101:08d}"[-8:]
        path = f"bvhs/{date}_{index % 1000:03d}_{index % 1000:03d}.bvh"
        siblings.append(
            SimpleNamespace(rfilename=path, size=10, blob_id=f"{index:040x}", lfs=None)
        )
    siblings.extend(
        SimpleNamespace(rfilename=name, size=1, blob_id="a" * 40, lfs=None)
        for name in ACQUIRE.EXPECTED_METADATA_FILES
    )
    api = SimpleNamespace(
        dataset_info=lambda *_args, **_kwargs: SimpleNamespace(
            sha=ACQUIRE.REVISION, siblings=siblings
        )
    )
    rows = ACQUIRE.remote_inventory(api)
    assert len(rows) == ACQUIRE.EXPECTED_BVH_COUNT + 3

    api.dataset_info = lambda *_args, **_kwargs: SimpleNamespace(
        sha="0" * 40, siblings=siblings
    )
    with pytest.raises(RuntimeError, match="revision drift"):
        ACQUIRE.remote_inventory(api)


def test_receipt_records_remote_and_local_hashes_but_blocks_without_license(tmp_path):
    payloads = {
        "README.md": b"card",
        "actors.db": b"actors",
        "scenarios.db": b"scenarios",
        "bvhs/20231119_001_051.bvh": b"motion",
    }
    rows = []
    for relative, payload in payloads.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        rows.append(_row(relative, payload))
    receipt = ACQUIRE.build_receipt(tmp_path, rows)
    assert receipt["verification"]["download_complete"] is True
    assert receipt["verification"]["verified_local_file_count"] == 4
    assert all(row["local_sha256"] for row in receipt["files"])
    assert receipt["license_gate"]["training_authorized_by_this_receipt"] is False
    assert receipt["license_gate"]["formal_training_blocked"] is True
    assert receipt["accepted_for_training"] is False


def test_valid_noncommercial_confirmation_only_opens_license_gate(tmp_path):
    payload = b"motion"
    relative = "bvhs/20231119_001_051.bvh"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    confirmation = tmp_path / "confirmation.json"
    confirmation.write_text(
        json.dumps(
            {
                "artifact_kind": "interact_noncommercial_use_confirmation",
                "repo_id": ACQUIRE.REPO_ID,
                "revision": ACQUIRE.REVISION,
                "license_id": ACQUIRE.LICENSE_ID,
                "approved": True,
                "use_scope": "noncommercial_research_training",
                "approver": "authorized-reviewer",
                "approved_at_utc": "2026-07-24T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    receipt = ACQUIRE.build_receipt(
        tmp_path, [_row(relative, payload)], confirmation_receipt=confirmation
    )
    assert receipt["license_gate"]["training_authorized_by_this_receipt"] is True
    # Dataset admission remains a separate physical/semantic review question.
    assert receipt["accepted_for_training"] is False


def test_missing_or_forbidden_data_remains_fail_closed(tmp_path):
    payload = b"motion"
    receipt = ACQUIRE.build_receipt(
        tmp_path, [_row("bvhs/20231119_001_051.bvh", payload)]
    )
    assert receipt["verification"]["missing_file_count"] == 1
    assert receipt["verification"]["download_complete"] is False

    path = tmp_path / "bvhs/20231119_001_051.bvh"
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    (tmp_path / "wav").mkdir()
    receipt = ACQUIRE.build_receipt(tmp_path, [_row(path.relative_to(tmp_path).as_posix(), payload)])
    assert receipt["verification"]["download_complete"] is False
    assert receipt["verification"]["forbidden_component_directories_present"] == ["wav"]
