import json
from pathlib import Path

import pytest

from tools.gmr_v2 import batch_retarget_beat2_v2 as ordinary
from tools.gmr_v2 import run_beat2_v7_cifs_resume as recovery


def locked_run_contract(root: Path) -> None:
    contract = {
        "artifacts": {
            "ordinary_batch_contract_helpers": {
                "sha256": recovery.EXPECTED_ORDINARY_SHA256
            },
            "grouped_batch_runner": {"sha256": recovery.EXPECTED_GROUPED_SHA256},
        }
    }
    document = {
        "run_contract": contract,
        "run_contract_sha256": ordinary.json_sha256(contract),
    }
    (root / ordinary.RUN_CONTRACT_FILENAME).write_text(
        json.dumps(document), encoding="utf-8"
    )


def test_validate_locked_run_preserves_original_contract(tmp_path):
    locked_run_contract(tmp_path)
    output, document = recovery.validate_locked_run(
        ["--output-root", str(tmp_path), "--resume"]
    )
    assert output == tmp_path
    assert document["run_contract_sha256"] == ordinary.json_sha256(
        document["run_contract"]
    )


def test_recovery_launcher_is_resume_only(tmp_path):
    locked_run_contract(tmp_path)
    with pytest.raises(ValueError, match="resume-only"):
        recovery.validate_locked_run(["--output-root", str(tmp_path)])


def test_resilient_atomic_text_retries_permission_error(tmp_path, monkeypatch):
    target = tmp_path / "passed_manifest.jsonl"
    target.write_text("old\n", encoding="utf-8")
    real_replace = recovery.os.replace
    attempts = 0

    def flaky_replace(source, destination):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("transient CIFS close race")
        real_replace(source, destination)

    monkeypatch.setattr(recovery.os, "replace", flaky_replace)
    monkeypatch.setattr(recovery.time, "sleep", lambda _: None)
    recovery.resilient_atomic_text(target, "new\n")
    assert attempts == 3
    assert target.read_text(encoding="utf-8") == "new\n"
    assert not list(tmp_path.glob("*.cifs-recovery.tmp"))
