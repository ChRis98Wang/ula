#!/usr/bin/env python3
"""Initialize and inventory the NAS human-motion collection."""

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_ROOT = Path("/home/gez/nas/cloud/gez/human_motion")
DEFAULT_GMR_SAMPLE = Path(
    "/home/gez/shuaiwang/GMR/assets/xsens_bvh_test/"
    "251021_04_boxing_120Hz_cm_3DsMax.bvh"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--gmr-sample", type=Path, default=DEFAULT_GMR_SAMPLE)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def count_files(path, suffix):
    if not path.exists():
        return 0
    return sum(1 for item in path.glob(f"*{suffix}") if item.is_file())


def copy_validation_sample(source, destination):
    if not source.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(source)
    if destination.exists():
        if sha256(destination) != source_hash:
            raise RuntimeError(f"Existing validation sample differs: {destination}")
    else:
        shutil.copy2(source, destination)
    return source_hash


def inventory(path):
    extensions = Counter()
    file_count = 0
    total_bytes = 0
    if path.exists():
        for item in path.rglob("*"):
            if not item.is_file() or ".cache" in item.parts:
                continue
            file_count += 1
            total_bytes += item.stat().st_size
            extensions[item.suffix.lower() or "<none>"] += 1
    return {
        "relative_path": str(path),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "extensions": dict(sorted(extensions.items())),
    }


def main():
    args = parse_args()
    root = args.root.resolve()
    directories = [
        "catalog",
        "licenses",
        "models/smplx2020",
        "processed/canonical_smplx30/v1",
        "processed/ula_v2_15d/v1",
        "processed/qc/v1",
        "processed/mujoco_previews/v1",
        "splits/official",
        "splits/subject_holdout",
        "splits/action_holdout",
        "staging",
    ]
    for relative in directories:
        (root / relative).mkdir(parents=True, exist_ok=True)

    validation_path = root / "raw/xsens_bvh/gmr_boxing/251021_04_boxing_120Hz_cm_3DsMax.bvh"
    validation_hash = copy_validation_sample(args.gmr_sample, validation_path)
    beat2_root = root / "raw/BEAT2/beat_chinese_v2.0.0"
    beat2_counts = {
        "motion_npz": count_files(beat2_root / "smplxflame_30", ".npz"),
        "audio_wav": count_files(beat2_root / "wave16k", ".wav"),
        "text_txt": count_files(beat2_root / "text", ".txt"),
        "textgrid": count_files(beat2_root / "textgrid", ".TextGrid"),
    }
    beat2_complete = beat2_counts == {
        "motion_npz": 310,
        "audio_wav": 317,
        "text_txt": 317,
        "textgrid": 287,
    }
    motionx_root = root / "raw/Motion-Xplusplus"
    motionx_counts = {
        "motion_npy": count_files(
            motionx_root / "extracted/haa500/smplx322", ".npy"
        ),
        "semantic_label_txt": count_files(
            motionx_root / "extracted/haa500/semantic_label", ".txt"
        ),
    }
    motionx_complete = motionx_counts == {
        "motion_npy": 6944,
        "semantic_label_txt": 6944,
    }
    revision_beat2 = "8689ecb43513ba31964fd60e0ca69be02d3b0872"
    revision_motionx = "c74fa62247289ed31e407b6133d954d3c171db43"
    sources = {
        "schema_version": 1,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "collection_root": str(root),
        "policy": {
            "raw_is_immutable": True,
            "head_face_fingers_mapped_to_v2": False,
            "training_requires_qc_pass": True,
            "training_requires_license_review": True,
            "required_generalization_splits": ["subject_holdout", "action_holdout"],
        },
        "datasets": [
            {
                "id": "beat2_chinese_v2",
                "role": "co_speech_expression",
                "source_url": "https://huggingface.co/datasets/H-Liu1997/BEAT2",
                "revision": revision_beat2,
                "path": "raw/BEAT2/beat_chinese_v2.0.0",
                "source_format": "SMPL-X/FLAME NPZ at 30 Hz plus audio and text",
                "expected_motion_clips": 310,
                "observed_counts": beat2_counts,
                "license": "Apache-2.0 dataset card; SMPL-X model has separate terms",
                "download_status": "complete" if beat2_complete else "partial",
                "training_status": "raw_only_requires_smplx2020_adapter_and_qc",
            },
            {
                "id": "motion_xplusplus_haa500",
                "role": "single_person_semantic_action",
                "source_url": "https://huggingface.co/datasets/YuhongZhang/Motion-Xplusplus",
                "revision": revision_motionx,
                "path": "raw/Motion-Xplusplus",
                "source_format": "SMPL-X 322D archives plus semantic labels",
                "expected_motion_clips": 6944,
                "observed_counts": motionx_counts,
                "license": "Research/non-commercial and source-specific terms; review before training",
                "download_status": "complete" if motionx_complete else "partial",
                "training_status": "raw_only_requires_smplx_adapter_and_qc",
            },
            {
                "id": "gmr_xsens_boxing_validation",
                "role": "high_speed_coordination_validation",
                "source_url": "https://github.com/YanjieZe/GMR/blob/master/assets/xsens_bvh_test/251021_04_boxing_120Hz_cm_3DsMax.bvh",
                "revision": "bb1bbe40774794fceb2a7c579a3464a28e68c844",
                "path": str(validation_path.relative_to(root)),
                "source_format": "Xsens 3DSM BVH at 120 Hz",
                "sha256": validation_hash,
                "license": "GMR repository sample; validate asset provenance before training",
                "download_status": "complete" if validation_hash else "missing",
                "training_status": "validation_only",
            },
        ],
    }
    schema = {
        "schema_version": 1,
        "required_fields": [
            "clip_id",
            "dataset",
            "dataset_revision",
            "source_relpath",
            "source_sha256",
            "source_format",
            "license_status",
            "subject_id",
            "language",
            "source_fps",
            "frame_count",
            "duration_sec",
            "action",
            "communicative_intent",
            "label_source",
            "v2_csv_relpath",
            "preview_relpath",
            "quality_relpath",
            "joint_limit_violations",
            "max_velocity_rad_s",
            "collision_frames",
            "qc_pass",
            "manual_reviewed",
            "accepted_for_training",
            "generalization_split",
        ],
    }
    scan_paths = {
        "beat2": root / "raw/BEAT2",
        "motion_xplusplus": root / "raw/Motion-Xplusplus",
        "xsens_validation": root / "raw/xsens_bvh",
        "smplx_models": root / "models/smplx2020",
        "processed_v2": root / "processed/ula_v2_15d/v1",
        "mujoco_previews": root / "processed/mujoco_previews/v1",
    }
    inventories = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "entries": {name: inventory(path) for name, path in scan_paths.items()},
    }
    atomic_json(root / "catalog/sources.json", sources)
    atomic_json(root / "catalog/manifest_schema.json", schema)
    atomic_json(root / "catalog/inventory.json", inventories)
    atomic_text(
        root / "README.md",
        "# Human motion collection\n\n"
        "This directory is the controlled source collection for ULA motion training.\n\n"
        "- `raw/`: immutable source data at pinned dataset revisions.\n"
        "- `catalog/`: source inventory, schemas, and expression candidates.\n"
        "- `processed/`: canonical SMPL-X, ULA V2, QC, and preview outputs.\n"
        "- `splits/`: official, subject-holdout, and action-holdout splits.\n"
        "- `models/`: separately licensed body-model files; never redistribute.\n\n"
        "Raw data is not training-ready. A clip must be retargeted without adding "
        "head/face/finger outputs, pass physical QC, receive manual video review, "
        "and be assigned to both subject and action generalization splits.\n",
    )
    atomic_text(
        root / "licenses/README.md",
        "# License records\n\n"
        "Keep a copy or link to each source's terms here before using processed "
        "clips for training. Motion-X++ and its source datasets require a "
        "research/non-commercial and source-specific review. The SMPL-X body "
        "model has separate terms and must not be redistributed.\n",
    )
    print(json.dumps(inventories, indent=2))


if __name__ == "__main__":
    main()
