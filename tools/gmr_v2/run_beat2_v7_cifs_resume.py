#!/usr/bin/env python3
"""Resume the locked BEAT2 v7 run with a CIFS-safe manifest publisher."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Sequence

from tools.gmr_v2 import batch_retarget_beat2_semantic_events_v2 as grouped
from tools.gmr_v2 import batch_retarget_beat2_v2 as ordinary


EXPECTED_ORDINARY_SHA256 = "cd3dd79c4420d7eeea3ce11a96a85b9453907976cae8e9f3049e078170c0f652"
EXPECTED_GROUPED_SHA256 = "38f2614e6d13b2ddb68ca8ce65314a8d14dfda2640d50cccfc0455f2c5f7e086"
AUDIT_NAME = "cifs_manifest_recovery_audit.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resilient_atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.cifs-recovery.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(0.25 * (2**attempt), 2.0))


def _argument_value(argv: Sequence[str], name: str) -> str:
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"Recovery launcher requires {name}") from error


def validate_locked_run(argv: Sequence[str]) -> tuple[Path, dict]:
    if "--resume" not in argv:
        raise ValueError("CIFS recovery launcher is resume-only")
    output_root = Path(_argument_value(argv, "--output-root")).resolve()
    run_contract_path = output_root / ordinary.RUN_CONTRACT_FILENAME
    document = json.loads(run_contract_path.read_text(encoding="utf-8"))
    run_contract = document.get("run_contract") or {}
    bindings = run_contract.get("artifacts") or {}
    if document.get("run_contract_sha256") != ordinary.json_sha256(run_contract):
        raise RuntimeError("Stored run-contract record SHA mismatch")
    expected = {
        "ordinary_batch_contract_helpers": EXPECTED_ORDINARY_SHA256,
        "grouped_batch_runner": EXPECTED_GROUPED_SHA256,
    }
    current = {
        "ordinary_batch_contract_helpers": sha256_file(Path(ordinary.__file__).resolve()),
        "grouped_batch_runner": sha256_file(Path(grouped.__file__).resolve()),
    }
    for name, expected_sha in expected.items():
        declared = (bindings.get(name) or {}).get("sha256")
        if declared != expected_sha or current[name] != expected_sha:
            raise RuntimeError(f"Locked implementation binding changed: {name}")
    return output_root, document


def run(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    output_root, run_contract_document = validate_locked_run(arguments)
    wrapper = Path(__file__).resolve()
    audit = {
        "artifact_kind": "beat2_v7_cifs_manifest_recovery_audit_v1",
        "schema_version": "1.0.0",
        "scope": "atomic_manifest_publication_only_no_retarget_or_contract_change",
        "resume_only": True,
        "run_contract": str(output_root / ordinary.RUN_CONTRACT_FILENAME),
        "run_contract_sha256": sha256_file(output_root / ordinary.RUN_CONTRACT_FILENAME),
        "locked_run_contract_record_sha256": run_contract_document[
            "run_contract_sha256"
        ],
        "ordinary_batch_contract_helpers_sha256": EXPECTED_ORDINARY_SHA256,
        "grouped_batch_runner_sha256": EXPECTED_GROUPED_SHA256,
        "recovery_launcher": str(wrapper),
        "recovery_launcher_sha256": sha256_file(wrapper),
        "replace_retry_attempts": 8,
        "temporary_file_fsynced": True,
        "accepted_for_training": False,
    }
    resilient_atomic_text(
        output_root / AUDIT_NAME,
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
    )
    ordinary.atomic_text = resilient_atomic_text
    return grouped.main(arguments)


if __name__ == "__main__":
    raise SystemExit(run())
