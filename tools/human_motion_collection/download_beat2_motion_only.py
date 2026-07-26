#!/usr/bin/env python3
"""Download pinned BEAT2 motion and text metadata without audio."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Iterable

from huggingface_hub import HfApi, snapshot_download


REPO_ID = "H-Liu1997/BEAT2"
REPO_TYPE = "dataset"
REVISION = "8689ecb43513ba31964fd60e0ca69be02d3b0872"
DEFAULT_ROOT = Path("/home/gez/nas/cloud/gez/human_motion/raw/BEAT2")
DEFAULT_MANIFEST = Path(
    "/home/gez/nas/cloud/gez/human_motion/catalog/beat2_motion_only_acquisition.json"
)
LANGUAGE_LAYOUT = {
    "english": {
        "root": "beat_english_v2.0.0",
        "label_dir": "sem",
    },
    "japanese": {
        "root": "beat_japanese_v2.0.0",
        "label_dir": "text",
    },
    "spanish": {
        "root": "beat_spanish_v2.0.0",
        "label_dir": "text",
    },
}
ALLOWED_COMPONENTS = (
    "smplxflame_30",
    "labels",
    "textgrid",
    "metadata",
)
GIT_LFS_POINTER = re.compile(
    rb"\Aversion https://git-lfs\.github\.com/spec/v1\n"
    rb"oid sha256:([0-9a-f]{64})\n"
    rb"size ([0-9]+)\n?\Z"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=tuple(LANGUAGE_LAYOUT),
        default=tuple(LANGUAGE_LAYOUT),
    )
    parser.add_argument(
        "--components",
        nargs="+",
        choices=ALLOWED_COMPONENTS,
        default=ALLOWED_COMPONENTS,
    )
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--metadata-git-repo",
        type=Path,
        help=(
            "Read the pinned file inventory and Git LFS OIDs from an existing "
            "metadata-only clone instead of the Hugging Face API"
        ),
    )
    parser.add_argument(
        "--verify-existing-only",
        action="store_true",
        help="Verify local files and write the receipt without downloading",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def allowed_patterns(languages: Iterable[str], components: Iterable[str]) -> list[str]:
    selected = set(components)
    patterns: list[str] = []
    for language in languages:
        layout = LANGUAGE_LAYOUT[language]
        root = layout["root"]
        if "smplxflame_30" in selected:
            patterns.append(f"{root}/smplxflame_30/*.npz")
        if "labels" in selected:
            patterns.append(f"{root}/{layout['label_dir']}/*")
        if "textgrid" in selected:
            patterns.append(f"{root}/textgrid/*")
        if "metadata" in selected:
            patterns.extend((f"{root}/readme.md", f"{root}/train_test_split.csv"))
    return patterns


def _matches_component(path: str, languages: set[str], components: set[str]) -> bool:
    for language in languages:
        layout = LANGUAGE_LAYOUT[language]
        root = layout["root"]
        prefix = f"{root}/"
        if not path.startswith(prefix):
            continue
        relative = path[len(prefix) :]
        if "smplxflame_30" in components and relative.startswith("smplxflame_30/"):
            return relative.endswith(".npz")
        if "labels" in components and relative.startswith(f"{layout['label_dir']}/"):
            return True
        if "textgrid" in components and relative.startswith("textgrid/"):
            return True
        if "metadata" in components and relative in {"readme.md", "train_test_split.csv"}:
            return True
    return False


def remote_inventory(languages: Iterable[str], components: Iterable[str]) -> list[dict]:
    language_set = set(languages)
    component_set = set(components)
    info = HfApi().dataset_info(REPO_ID, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise RuntimeError(f"Resolved BEAT2 revision changed: {info.sha}")
    rows = []
    for sibling in info.siblings:
        path = sibling.rfilename
        if _matches_component(path, language_set, component_set):
            if "/wave16k/" in path or "/weights/" in path:
                raise RuntimeError(f"Forbidden BEAT2 component selected: {path}")
            rows.append({"path": path, "size": int(sibling.size or 0)})
    if not rows:
        raise RuntimeError("BEAT2 selection resolved to zero remote files")
    return sorted(rows, key=lambda row: row["path"])


def git_blob_receipt(path: str, payload: bytes) -> dict:
    """Return the expected local content hash for a Git or Git LFS blob."""
    match = GIT_LFS_POINTER.fullmatch(payload)
    if match:
        return {
            "path": path,
            "size": int(match.group(2)),
            "sha256": match.group(1).decode("ascii"),
            "integrity_source": "git_lfs_oid_sha256",
        }
    return {
        "path": path,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "integrity_source": "git_blob_content_sha256",
    }


def _git_bytes(repo: Path, *args: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", b"")
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        raise RuntimeError(f"Git metadata query failed: {detail}") from error
    return completed.stdout


def git_metadata_inventory(
    repo: Path,
    languages: Iterable[str],
    components: Iterable[str],
    *,
    revision: str = REVISION,
) -> list[dict]:
    """Build an offline content receipt from a pinned metadata-only clone."""
    repo = repo.resolve()
    resolved = _git_bytes(repo, "rev-parse", revision).decode("ascii").strip()
    if resolved != revision:
        raise RuntimeError(
            f"Resolved BEAT2 metadata revision changed: {resolved} != {revision}"
        )
    language_set = set(languages)
    component_set = set(components)
    tree = _git_bytes(repo, "ls-tree", "-r", "-z", revision)
    rows = []
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        _mode, object_type, object_id = metadata.decode("ascii").split()
        path = raw_path.decode("utf-8")
        if object_type != "blob" or not _matches_component(
            path, language_set, component_set
        ):
            continue
        if "/wave16k/" in path or "/weights/" in path:
            raise RuntimeError(f"Forbidden BEAT2 component selected: {path}")
        payload = _git_bytes(repo, "cat-file", "blob", object_id)
        rows.append(git_blob_receipt(path, payload))
    if not rows:
        raise RuntimeError("BEAT2 Git metadata selection resolved to zero files")
    return sorted(rows, key=lambda row: row["path"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def manifest_console_summary(manifest: dict) -> dict:
    """Keep per-file receipts on disk without flooding terminal logs."""
    return {
        key: value
        for key, value in manifest.items()
        if key != "files"
    } | {"file_receipt_count": len(manifest.get("files") or [])}


def build_manifest(
    root: Path,
    languages: list[str],
    components: list[str],
    rows: list[dict],
    *,
    inventory_source: str = "huggingface_api_file_metadata",
) -> dict:
    missing = []
    mismatched_size = []
    mismatched_sha256 = []
    verified_sha256_count = 0
    per_language: dict[str, dict] = {}
    for language in languages:
        language_root = LANGUAGE_LAYOUT[language]["root"]
        selected = [row for row in rows if row["path"].startswith(f"{language_root}/")]
        present = []
        language_verified_sha256_count = 0
        for row in selected:
            local_path = root / row["path"]
            if not local_path.is_file():
                missing.append(row["path"])
                continue
            observed_size = local_path.stat().st_size
            if observed_size != row["size"]:
                mismatched_size.append(
                    {
                        "path": row["path"],
                        "expected": row["size"],
                        "observed": observed_size,
                    }
                )
            expected_sha256 = row.get("sha256")
            if expected_sha256:
                observed_sha256 = sha256_file(local_path)
                if observed_sha256 != expected_sha256:
                    mismatched_sha256.append(
                        {
                            "path": row["path"],
                            "expected": expected_sha256,
                            "observed": observed_sha256,
                        }
                    )
                else:
                    verified_sha256_count += 1
                    language_verified_sha256_count += 1
            present.append(local_path)
        per_language[language] = {
            "remote_file_count": len(selected),
            "remote_bytes": sum(row["size"] for row in selected),
            "verified_local_file_count": len(present),
            "verified_sha256_file_count": language_verified_sha256_count,
            "motion_npz_count": sum(path.suffix == ".npz" for path in present),
        }
    if missing or mismatched_size or mismatched_sha256:
        raise RuntimeError(
            f"BEAT2 download verification failed: missing={len(missing)}, "
            f"size_mismatch={len(mismatched_size)}, "
            f"sha256_mismatch={len(mismatched_sha256)}"
        )
    all_rows_have_sha256 = all(bool(row.get("sha256")) for row in rows)
    return {
        "schema_version": 1,
        "artifact_kind": "beat2_motion_only_acquisition",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repo_id": REPO_ID,
            "repo_type": REPO_TYPE,
            "revision": REVISION,
            "url": f"https://huggingface.co/datasets/{REPO_ID}",
            "dataset_card_license": "apache-2.0",
            "body_model_terms": "SMPL-X model files remain separately licensed",
            "inventory_source": inventory_source,
        },
        "root": str(root),
        "languages": languages,
        "components": components,
        "audio_policy": "excluded_not_downloaded",
        "face_and_finger_policy": "present_in_source_npz_but_ignored_by_18d_retarget_adapter",
        "remote_file_count": len(rows),
        "remote_bytes": sum(row["size"] for row in rows),
        "files": rows,
        "per_language": per_language,
        "selection_sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "verification": {
            "all_selected_files_present": True,
            "all_selected_sizes_match": True,
            "all_selected_sha256_available": all_rows_have_sha256,
            "all_selected_sha256_match": (
                all_rows_have_sha256 and verified_sha256_count == len(rows)
            ),
            "verified_sha256_file_count": verified_sha256_count,
            "forbidden_audio_selected": False,
        },
        "license_review": {
            "dataset_card_declared_license": "apache-2.0",
            "dataset_training_use_review_status": "pending_human_review",
            "smplx_terms_review_status": "pending_human_review",
            "training_authorized_by_this_receipt": False,
        },
    }


def main() -> None:
    args = parse_args()
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be positive")
    languages = list(dict.fromkeys(args.languages))
    components = list(dict.fromkeys(args.components))
    if args.verify_existing_only and args.metadata_git_repo is None:
        raise ValueError("--verify-existing-only requires --metadata-git-repo")
    if args.metadata_git_repo is not None:
        rows = git_metadata_inventory(
            args.metadata_git_repo, languages, components
        )
        inventory_source = "pinned_git_tree_and_lfs_oid"
    else:
        rows = remote_inventory(languages, components)
        inventory_source = "huggingface_api_file_metadata"
    plan = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "root": str(args.root.resolve()),
        "languages": languages,
        "components": components,
        "audio_policy": "excluded_not_downloaded",
        "file_count": len(rows),
        "bytes": sum(row["size"] for row in rows),
        "allow_patterns": allowed_patterns(languages, components),
        "inventory_source": inventory_source,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    root = args.root.resolve()
    if not args.verify_existing_only:
        root.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            revision=REVISION,
            local_dir=root,
            allow_patterns=plan["allow_patterns"],
            max_workers=args.max_workers,
        )
    manifest = build_manifest(
        root,
        languages,
        components,
        rows,
        inventory_source=inventory_source,
    )
    atomic_json(args.manifest.resolve(), manifest)
    print(json.dumps(manifest_console_summary(manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
