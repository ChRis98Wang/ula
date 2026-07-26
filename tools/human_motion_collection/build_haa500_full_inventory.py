#!/usr/bin/env python3
"""Build a deterministic inventory for every local Motion-X++ HAA500 clip.

The filename action is the primary label.  The bundled GPT-4V sentence remains
auxiliary and never grants training admission by itself.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path("/home/gez/nas/cloud/gez/human_motion")
DATASET_REVISION = "c74fa62247289ed31e407b6133d954d3c171db43"
OUTPUT_STEM = "motionx_haa500_full_inventory_v1"
ACTION_SUFFIX = re.compile(r"(?:_\d+)?_clip\d+$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REFUSAL = re.compile(
    r"\b(?:sorry|cannot (?:assist|provide|describe)|unable to (?:assist|provide|describe))\b",
    re.IGNORECASE,
)
VAGUE = re.compile(
    r"(?:images?|frames?) (?:appear|are|seem(?: to be)?) (?:identical|repetitive)|"
    r"no (?:visible|observable|significant|discernible) "
    r"(?:movement|motion|action|difference|change)|"
    r"do not show any (?:movement|motion|action|change)|"
    r"no clear temporal sequence|motion (?:is )?minimal to non-existent|"
    r"cannot (?:give|provide) (?:a )?detailed description|no content visible",
    re.IGNORECASE,
)

LOWER_BODY_TOKENS = {
    "backflip", "flip", "foot", "jump", "kick", "layup", "leg", "run",
    "skate", "ski", "spin", "squat", "step", "vault", "walk",
}
FINE_DETAIL_TOKENS = {
    "blow", "blowing", "cream", "eye", "face", "finger", "glasses", "gum",
    "hair", "kiss", "kisses", "makeup", "nose", "teeth", "whistle",
}
INTERACTION_TOKENS = {
    "catch", "fight", "handshake", "hug", "hugging", "human", "person", "wrestling",
}
OBJECT_TOKENS = {
    "apple", "ball", "balloon", "bat", "bike", "billiard", "bottle", "bow",
    "drum", "floor", "glass", "guitar", "leaf", "rope", "snowman", "tire",
    "viola",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-motion-sha256", action="store_true")
    parser.add_argument(
        "--reuse-hashes-from",
        type=Path,
        help=(
            "Reuse per-file hashes from a prior inventory JSONL after exact "
            "clip/path/revision validation"
        ),
    )
    args = parser.parse_args()
    if args.skip_motion_sha256 and args.reuse_hashes_from:
        parser.error("--skip-motion-sha256 cannot be combined with --reuse-hashes-from")
    return args


def action_from_stem(stem):
    return ACTION_SUFFIX.sub("", stem)


def normalize_text(value):
    return " ".join((value or "").strip().split())


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def display_path(path, root):
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def load_reusable_hashes(path, root, motion_root, label_root, expected_stems):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Hash reuse inventory does not exist: {path}")
    prior_records = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank line in hash reuse inventory at line {line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in hash reuse inventory at line {line_number}: {exc}"
                ) from exc
            clip_id = record.get("clip_id")
            if not isinstance(clip_id, str) or not clip_id:
                raise ValueError(
                    f"Missing clip_id in hash reuse inventory at line {line_number}"
                )
            if clip_id in prior_records:
                raise ValueError(f"Duplicate clip_id in hash reuse inventory: {clip_id}")
            prior_records[clip_id] = record

    prior_stems = set(prior_records)
    if prior_stems != expected_stems:
        missing = sorted(expected_stems - prior_stems)
        extra = sorted(prior_stems - expected_stems)
        raise ValueError(
            "Hash reuse clip set mismatch: "
            f"missing={len(missing)} {missing[:5]}, extra={len(extra)} {extra[:5]}"
        )

    reusable = {}
    for clip_id in sorted(expected_stems):
        record = prior_records[clip_id]
        expected_motion = str((motion_root / f"{clip_id}.npy").relative_to(root))
        expected_label = str((label_root / f"{clip_id}.txt").relative_to(root))
        if record.get("motion_relpath") != expected_motion:
            raise ValueError(f"Hash reuse motion path mismatch for {clip_id}")
        if record.get("label_relpath") != expected_label:
            raise ValueError(f"Hash reuse label path mismatch for {clip_id}")
        if Path(record["motion_relpath"]).stem != clip_id:
            raise ValueError(f"Hash reuse motion stem mismatch for {clip_id}")
        if Path(record["label_relpath"]).stem != clip_id:
            raise ValueError(f"Hash reuse label stem mismatch for {clip_id}")
        if record.get("dataset_revision") != DATASET_REVISION:
            raise ValueError(f"Hash reuse dataset revision mismatch for {clip_id}")
        motion_sha256 = record.get("motion_sha256")
        label_sha256 = record.get("label_sha256")
        if not isinstance(motion_sha256, str) or not SHA256_PATTERN.fullmatch(
            motion_sha256
        ):
            raise ValueError(f"Invalid reusable motion SHA256 for {clip_id}")
        if not isinstance(label_sha256, str) or not SHA256_PATTERN.fullmatch(
            label_sha256
        ):
            raise ValueError(f"Invalid reusable label SHA256 for {clip_id}")
        reusable[clip_id] = {
            "motion_sha256": motion_sha256,
            "label_sha256": label_sha256,
        }
    return reusable, path


def action_tokens(action):
    return {token.lower() for token in re.split(r"[^A-Za-z0-9]+", action) if token}


def readable_action(action):
    words = action.replace("_", " ").split()
    return " ".join("ALS" if word.lower() == "als" else word.lower() for word in words)


def observable_risks(action):
    tokens = action_tokens(action)
    risks = []
    if tokens & LOWER_BODY_TOKENS:
        risks.append("fixed_base_discards_lower_body_or_root_motion")
    if tokens & FINE_DETAIL_TOKENS:
        risks.append("face_finger_or_fine_hand_cues_unavailable")
    if tokens & INTERACTION_TOKENS:
        risks.append("interaction_partner_or_contact_unavailable")
    if tokens & OBJECT_TOKENS:
        risks.append("scene_object_unavailable")
    return risks


def build_inventory(
    root,
    output_dir=None,
    include_motion_hash=True,
    reuse_hashes_from=None,
):
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve() if output_dir else root / "catalog"
    motion_root = root / "raw/Motion-Xplusplus/extracted/haa500/smplx322"
    label_root = root / "raw/Motion-Xplusplus/extracted/haa500/semantic_label"
    motion_paths = sorted(motion_root.glob("*.npy"))
    label_paths = {path.stem: path for path in sorted(label_root.glob("*.txt"))}
    if not motion_paths:
        raise FileNotFoundError(f"No HAA500 motion files under {motion_root}")
    motion_stems = {path.stem for path in motion_paths}
    label_stems = set(label_paths)
    if motion_stems != label_stems:
        missing_labels = sorted(motion_stems - label_stems)
        orphan_labels = sorted(label_stems - motion_stems)
        raise ValueError(
            "HAA500 motion/label stem mismatch: "
            f"missing_labels={len(missing_labels)} {missing_labels[:5]}, "
            f"orphan_labels={len(orphan_labels)} {orphan_labels[:5]}"
        )
    if reuse_hashes_from and not include_motion_hash:
        raise ValueError("Hash reuse requires include_motion_hash=True")
    reusable_hashes = None
    reuse_path = None
    if reuse_hashes_from:
        reusable_hashes, reuse_path = load_reusable_hashes(
            reuse_hashes_from,
            root,
            motion_root,
            label_root,
            motion_stems,
        )

    texts = {}
    duplicate_groups = defaultdict(list)
    for clip_id, path in label_paths.items():
        text = normalize_text(path.read_text(encoding="utf-8", errors="replace"))
        texts[clip_id] = text
        duplicate_groups[text].append((clip_id, action_from_stem(clip_id)))

    records = []
    for motion_path in motion_paths:
        clip_id = motion_path.stem
        action = action_from_stem(clip_id)
        label_path = label_paths.get(clip_id)
        source_text = texts.get(clip_id, "")
        group = duplicate_groups.get(source_text, []) if source_text else []
        group_actions = sorted({item[1] for item in group})
        text_flags = []
        if not source_text:
            text_flags.append("missing_or_empty")
        if len(group) > 1:
            text_flags.append(
                "cross_action_duplicate" if len(group_actions) > 1 else "same_action_duplicate"
            )
        if REFUSAL.search(source_text):
            text_flags.append("refusal")
        if VAGUE.search(source_text):
            text_flags.append("vague_or_non_observable")

        motion = np.load(motion_path, mmap_mode="r", allow_pickle=False)
        if motion.ndim != 2 or motion.shape[1] != 322:
            raise ValueError(f"Unexpected Motion-X shape {motion.shape}: {motion_path}")
        risks = observable_risks(action)
        reused = reusable_hashes.get(clip_id) if reusable_hashes else None
        critical_text = bool(
            {"missing_or_empty", "cross_action_duplicate", "refusal", "vague_or_non_observable"}
            & set(text_flags)
        )
        records.append(
            {
                "schema_version": "1.0.0",
                "clip_id": clip_id,
                "action": action,
                "canonical_prompt_en": (
                    f'Perform "{readable_action(action)}" while making the upper-body motion clear.'
                ),
                "canonical_prompt_source": "deterministic_filename_action_template",
                "primary_label_source": "filename_action",
                "source_text": source_text,
                "source_text_role": "gpt4v_auxiliary_only",
                "source_text_flags": text_flags,
                "duplicate_group_size": len(group),
                "duplicate_actions": group_actions,
                "observable_risks": risks,
                "robot_contract": "ula_v2_18d_head_v1",
                "motion_relpath": str(motion_path.relative_to(root)),
                "label_relpath": str(label_path.relative_to(root)) if label_path else "",
                "frame_count": int(motion.shape[0]),
                "feature_dim": int(motion.shape[1]),
                "nominal_fps": 30.0,
                "nominal_duration_sec": round(float(motion.shape[0]) / 30.0, 6),
                "motion_sha256": (
                    reused["motion_sha256"]
                    if reused
                    else sha256_file(motion_path) if include_motion_hash else None
                ),
                "label_sha256": (
                    reused["label_sha256"] if reused else sha256_file(label_path)
                ),
                "review_state": (
                    "machine_flagged_pending_review"
                    if critical_text or risks
                    else "machine_labeled_pending_review"
                ),
                "manual_review_required": True,
                "accepted_for_training": False,
                "dataset_revision": DATASET_REVISION,
            }
        )

    if len({record["clip_id"] for record in records}) != len(records):
        raise ValueError("Duplicate clip_id in HAA500 inventory")

    jsonl_path = output_dir / f"{OUTPUT_STEM}.jsonl"
    csv_path = output_dir / f"{OUTPUT_STEM}.csv"
    summary_path = output_dir / f"{OUTPUT_STEM}.summary.json"
    jsonl_bytes = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ).encode("utf-8")
    fields = [
        "clip_id", "action", "canonical_prompt_en", "canonical_prompt_source",
        "primary_label_source", "source_text_role", "source_text_flags",
        "observable_risks", "robot_contract", "motion_relpath", "label_relpath",
        "frame_count", "feature_dim", "nominal_fps", "nominal_duration_sec",
        "motion_sha256", "label_sha256", "review_state", "manual_review_required",
        "accepted_for_training", "dataset_revision",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for record in records:
        row = {key: record[key] for key in fields}
        row["source_text_flags"] = "|".join(record["source_text_flags"])
        row["observable_risks"] = "|".join(record["observable_risks"])
        writer.writerow(row)
    csv_bytes = buffer.getvalue().encode("utf-8")

    flag_counts = Counter(flag for record in records for flag in record["source_text_flags"])
    risk_counts = Counter(risk for record in records for risk in record["observable_risks"])
    total_frames = sum(record["frame_count"] for record in records)
    summary = {
        "schema_version": "1.0.0",
        "dataset": "Motion-X++/HAA500",
        "dataset_revision": DATASET_REVISION,
        "record_count": len(records),
        "action_count": len({record["action"] for record in records}),
        "frame_count": total_frames,
        "duration_sec": round(total_frames / 30.0, 6),
        "motion_hashes_included": include_motion_hash,
        "hash_policy": (
            "reused_after_exact_clip_path_revision_validation"
            if reusable_hashes
            else "calculated_from_source"
        ),
        "hash_reuse_source": display_path(reuse_path, root) if reuse_path else None,
        "reused_motion_hash_count": len(reusable_hashes) if reusable_hashes else 0,
        "reused_label_hash_count": len(reusable_hashes) if reusable_hashes else 0,
        "primary_label_policy": "filename_action",
        "canonical_prompt_policy": "deterministic_filename_action_template",
        "source_text_policy": "GPT-4V text is auxiliary and cannot grant training admission",
        "conditioning_policy": "deny_until_review",
        "accepted_for_training_count": 0,
        "counts_by_review_state": dict(sorted(Counter(r["review_state"] for r in records).items())),
        "counts_by_source_text_flag": dict(sorted(flag_counts.items())),
        "counts_by_observable_risk": dict(sorted(risk_counts.items())),
        "output_sha256": {
            "jsonl": hashlib.sha256(jsonl_bytes).hexdigest(),
            "csv": hashlib.sha256(csv_bytes).hexdigest(),
        },
    }
    summary_bytes = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_write(jsonl_path, jsonl_bytes)
    atomic_write(csv_path, csv_bytes)
    atomic_write(summary_path, summary_bytes)
    return summary


def main():
    args = parse_args()
    summary = build_inventory(
        args.root,
        args.output_dir,
        include_motion_hash=not args.skip_motion_sha256,
        reuse_hashes_from=args.reuse_hashes_from,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
