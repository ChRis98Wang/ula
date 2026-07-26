#!/usr/bin/env python3
"""Audit a pinned, motion-only InterAct acquisition.

The source dataset is CC-BY-NC-SA-4.0.  This receipt is deliberately
fail-closed: complete byte verification is not sufficient to authorize
training.  A separate, explicit non-commercial-use confirmation receipt is
required before ``training_authorized_by_this_receipt`` can become true.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable

from huggingface_hub import HfApi


REPO_ID = "leohocs/interact"
REPO_TYPE = "dataset"
REVISION = "152ba832f379c465f5b1e10c67166d646014d675"
LICENSE_ID = "CC-BY-NC-SA-4.0"
EXPECTED_BVH_COUNT = 482
EXPECTED_METADATA_FILES = ("README.md", "actors.db", "scenarios.db")
DEFAULT_ROOT = Path("/home/gez/nas/cloud/gez/human_motion/raw/InterAct")
DEFAULT_RECEIPT = Path(
    "/home/gez/nas/cloud/gez/human_motion/catalog/"
    "interact_motion_only_acquisition.json"
)
BVH_PATH = re.compile(r"bvhs/\d{8}_\d{3}_\d{3}\.bvh\Z")
FORBIDDEN_TOP_LEVEL = {
    "body_renders",
    "body_renders_noaudio",
    "bvhs_retarget",
    "face_arkit",
    "face_ict",
    "face_ict_templates",
    "face_renders_noaudio",
    "lip_acc",
    "wav",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument(
        "--noncommercial-confirmation-receipt",
        type=Path,
        help="Separately approved, pinned confirmation; absent means blocked",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Return an error after writing the receipt if any selected file is absent",
    )
    return parser.parse_args(argv)


def selected_remote_path(path: str) -> bool:
    return path in EXPECTED_METADATA_FILES or bool(BVH_PATH.fullmatch(path))


def remote_inventory(api: Any | None = None) -> list[dict[str, Any]]:
    api = api or HfApi()
    info = api.dataset_info(REPO_ID, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise RuntimeError(f"InterAct revision drift: {info.sha} != {REVISION}")
    rows: list[dict[str, Any]] = []
    for sibling in info.siblings:
        path = sibling.rfilename
        if not selected_remote_path(path):
            continue
        lfs = getattr(sibling, "lfs", None)
        lfs_sha256 = None
        if isinstance(lfs, dict):
            lfs_sha256 = lfs.get("sha256") or lfs.get("oid")
        elif lfs is not None:
            lfs_sha256 = getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)
        rows.append(
            {
                "path": path,
                "size": int(sibling.size or 0),
                "remote_git_blob_oid_sha1": str(sibling.blob_id),
                "remote_lfs_content_sha256": lfs_sha256,
            }
        )
    rows.sort(key=lambda row: row["path"])
    bvh_count = sum(bool(BVH_PATH.fullmatch(row["path"])) for row in rows)
    metadata = {row["path"] for row in rows if row["path"] in EXPECTED_METADATA_FILES}
    if bvh_count != EXPECTED_BVH_COUNT or metadata != set(EXPECTED_METADATA_FILES):
        raise RuntimeError(
            "Pinned InterAct selection changed: "
            f"bvh={bvh_count}/{EXPECTED_BVH_COUNT}, metadata={sorted(metadata)}"
        )
    return rows


def _hashes_for_stable_file(path: Path) -> tuple[int, str, str] | None:
    before = path.stat()
    sha256 = hashlib.sha256()
    git_blob_sha1 = hashlib.sha1()
    git_blob_sha1.update(f"blob {before.st_size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha256.update(chunk)
            git_blob_sha1.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        return None
    return before.st_size, sha256.hexdigest(), git_blob_sha1.hexdigest()


def _load_noncommercial_confirmation(path: Path | None) -> tuple[bool, dict[str, Any]]:
    if path is None:
        return False, {
            "status": "pending_explicit_noncommercial_use_confirmation",
            "receipt_path": None,
            "receipt_sha256": None,
            "validation_errors": ["confirmation_receipt_absent"],
        }
    raw = path.read_bytes()
    value = json.loads(raw)
    errors = []
    expected = {
        "artifact_kind": "interact_noncommercial_use_confirmation",
        "repo_id": REPO_ID,
        "revision": REVISION,
        "license_id": LICENSE_ID,
        "approved": True,
        "use_scope": "noncommercial_research_training",
    }
    for key, required in expected.items():
        if value.get(key) != required:
            errors.append(f"{key}_invalid")
    for key in ("approver", "approved_at_utc"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            errors.append(f"{key}_missing")
    return not errors, {
        "status": "confirmed" if not errors else "invalid_confirmation_receipt",
        "receipt_path": str(path.resolve()),
        "receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "validation_errors": errors,
    }


def build_receipt(
    root: Path,
    rows: Iterable[dict[str, Any]],
    *,
    confirmation_receipt: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    file_receipts = []
    missing = []
    unstable = []
    size_mismatch = []
    content_mismatch = []
    for remote in rows:
        row = dict(remote)
        local = root / row["path"]
        row["local_path"] = str(local)
        if not local.is_file():
            row["local_status"] = "missing"
            row["local_sha256"] = None
            missing.append(row["path"])
            file_receipts.append(row)
            continue
        hashes = _hashes_for_stable_file(local)
        if hashes is None:
            row["local_status"] = "unstable_during_hash"
            row["local_sha256"] = None
            unstable.append(row["path"])
            file_receipts.append(row)
            continue
        observed_size, local_sha256, local_git_blob_sha1 = hashes
        row.update(
            {
                "observed_size": observed_size,
                "local_sha256": local_sha256,
                "local_git_blob_oid_sha1": local_git_blob_sha1,
            }
        )
        matches_size = observed_size == row["size"]
        if row.get("remote_lfs_content_sha256"):
            matches_content = local_sha256 == row["remote_lfs_content_sha256"]
            identity_method = "lfs_content_sha256"
        else:
            matches_content = local_git_blob_sha1 == row["remote_git_blob_oid_sha1"]
            identity_method = "git_blob_oid_sha1"
        row["remote_local_identity_method"] = identity_method
        row["local_status"] = "verified" if matches_size and matches_content else "mismatch"
        if not matches_size:
            size_mismatch.append(row["path"])
        if not matches_content:
            content_mismatch.append(row["path"])
        file_receipts.append(row)

    forbidden_present = sorted(
        name for name in FORBIDDEN_TOP_LEVEL if (root / name).exists()
    )
    confirmation_valid, confirmation = _load_noncommercial_confirmation(
        confirmation_receipt
    )
    complete = not (
        missing
        or unstable
        or size_mismatch
        or content_mismatch
        or forbidden_present
    )
    selection_payload = json.dumps(
        [
            {
                "path": row["path"],
                "size": row["size"],
                "remote_git_blob_oid_sha1": row["remote_git_blob_oid_sha1"],
                "remote_lfs_content_sha256": row.get("remote_lfs_content_sha256"),
            }
            for row in file_receipts
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": "1.0.0",
        "artifact_kind": "interact_motion_only_acquisition",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repo_id": REPO_ID,
            "repo_type": REPO_TYPE,
            "revision": REVISION,
            "url": f"https://huggingface.co/datasets/{REPO_ID}",
            "dataset_card_license": LICENSE_ID,
        },
        "root": str(root),
        "selection_policy": {
            "included": ["bvhs/*.bvh", *EXPECTED_METADATA_FILES],
            "excluded": ["audio", "face", "finger-specific targets", "renders"],
            "bvh_count_expected": EXPECTED_BVH_COUNT,
            "metadata_file_count_expected": len(EXPECTED_METADATA_FILES),
            "remote_file_count": len(file_receipts),
            "remote_bytes": sum(row["size"] for row in file_receipts),
            "selection_sha256": hashlib.sha256(selection_payload).hexdigest(),
        },
        "files": file_receipts,
        "verification": {
            "all_selected_files_present": not missing,
            "all_selected_sizes_match": not size_mismatch and not missing and not unstable,
            "all_remote_local_content_identities_match": (
                not content_mismatch and not missing and not unstable
            ),
            "verified_local_file_count": sum(
                row["local_status"] == "verified" for row in file_receipts
            ),
            "missing_file_count": len(missing),
            "unstable_file_count": len(unstable),
            "size_mismatch_count": len(size_mismatch),
            "content_mismatch_count": len(content_mismatch),
            "forbidden_component_directories_present": forbidden_present,
            "download_complete": complete,
        },
        "license_gate": {
            "license_id": LICENSE_ID,
            "commercial_use_allowed": False,
            "noncommercial_confirmation": confirmation,
            "training_authorized_by_this_receipt": complete and confirmation_valid,
            "formal_training_blocked": not (complete and confirmation_valid),
        },
        "accepted_for_training": False,
        "admission_note": (
            "Acquisition integrity never admits motion, semantic, or emotion supervision; "
            "independent retarget and blind review gates remain required."
        ),
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = remote_inventory()
    receipt = build_receipt(
        args.root,
        rows,
        confirmation_receipt=args.noncommercial_confirmation_receipt,
    )
    atomic_json(args.receipt.resolve(), receipt)
    summary = {
        "receipt": str(args.receipt.resolve()),
        "remote_file_count": len(rows),
        "remote_bytes": receipt["selection_policy"]["remote_bytes"],
        "verification": receipt["verification"],
        "license_gate": receipt["license_gate"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.require_complete and not receipt["verification"]["download_complete"]:
        raise RuntimeError("InterAct acquisition is incomplete; receipt remains fail-closed")


if __name__ == "__main__":
    main()
