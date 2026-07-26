import importlib.util
import hashlib
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools/human_motion_collection/download_beat2_motion_only.py"
)
SPEC = importlib.util.spec_from_file_location("download_beat2_motion_only", SCRIPT_PATH)
assert SPEC and SPEC.loader
DOWNLOAD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOWNLOAD)


def test_default_selection_never_includes_audio_or_weights():
    patterns = DOWNLOAD.allowed_patterns(
        DOWNLOAD.LANGUAGE_LAYOUT,
        ("smplxflame_30", "labels", "textgrid", "metadata"),
    )
    assert patterns
    assert all("wave16k" not in pattern for pattern in patterns)
    assert all("weights" not in pattern for pattern in patterns)


def test_remote_component_match_is_fail_closed():
    languages = {"english"}
    components = {"smplxflame_30", "labels", "textgrid", "metadata"}
    assert DOWNLOAD._matches_component(
        "beat_english_v2.0.0/smplxflame_30/1_wayne_0_1_1.npz",
        languages,
        components,
    )
    assert DOWNLOAD._matches_component(
        "beat_english_v2.0.0/sem/1_wayne_0_1_1.txt",
        languages,
        components,
    )
    assert not DOWNLOAD._matches_component(
        "beat_english_v2.0.0/wave16k/1_wayne_0_1_1.wav",
        languages,
        components,
    )
    assert not DOWNLOAD._matches_component(
        "beat_chinese_v2.0.0/smplxflame_30/12_zhao_2_1_1.npz",
        languages,
        components,
    )


def test_build_manifest_verifies_selected_files_without_audio(tmp_path):
    relative = "beat_english_v2.0.0/smplxflame_30/1_wayne_0_1_1.npz"
    motion = tmp_path / relative
    motion.parent.mkdir(parents=True)
    motion.write_bytes(b"motion")
    manifest = DOWNLOAD.build_manifest(
        tmp_path,
        ["english"],
        ["smplxflame_30"],
        [{"path": relative, "size": len(b"motion")}],
    )
    assert manifest["audio_policy"] == "excluded_not_downloaded"
    assert manifest["source"]["revision"] == DOWNLOAD.REVISION
    assert manifest["per_language"]["english"]["motion_npz_count"] == 1
    assert manifest["verification"]["forbidden_audio_selected"] is False
    assert manifest["license_review"]["training_authorized_by_this_receipt"] is False


def test_git_lfs_pointer_receipt_uses_content_oid_and_size():
    sha256 = "a" * 64
    payload = (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{sha256}\n"
        "size 1234\n"
    ).encode("ascii")
    row = DOWNLOAD.git_blob_receipt("motion.npz", payload)
    assert row == {
        "path": "motion.npz",
        "size": 1234,
        "sha256": sha256,
        "integrity_source": "git_lfs_oid_sha256",
    }


def test_plain_git_blob_receipt_hashes_blob_content():
    payload = b"official metadata\n"
    row = DOWNLOAD.git_blob_receipt("readme.md", payload)
    assert row["size"] == len(payload)
    assert row["sha256"] == hashlib.sha256(payload).hexdigest()
    assert row["integrity_source"] == "git_blob_content_sha256"


def test_build_manifest_verifies_sha256_and_rejects_mismatch(tmp_path):
    relative = "beat_english_v2.0.0/smplxflame_30/1_wayne_0_1_1.npz"
    motion = tmp_path / relative
    motion.parent.mkdir(parents=True)
    motion.write_bytes(b"motion")
    row = {
        "path": relative,
        "size": len(b"motion"),
        "sha256": hashlib.sha256(b"motion").hexdigest(),
        "integrity_source": "git_lfs_oid_sha256",
    }
    manifest = DOWNLOAD.build_manifest(
        tmp_path,
        ["english"],
        ["smplxflame_30"],
        [row],
        inventory_source="pinned_git_tree_and_lfs_oid",
    )
    assert manifest["verification"]["all_selected_sha256_match"] is True
    assert manifest["verification"]["verified_sha256_file_count"] == 1
    assert manifest["files"] == [row]
    console = DOWNLOAD.manifest_console_summary(manifest)
    assert "files" not in console
    assert console["file_receipt_count"] == 1

    row["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="sha256_mismatch=1"):
        DOWNLOAD.build_manifest(
            tmp_path,
            ["english"],
            ["smplxflame_30"],
            [row],
        )
