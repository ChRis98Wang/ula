#!/usr/bin/env python3
"""Build a review-only HAA500 subset for readable upper-body expression."""

import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path("/home/gez/nas/cloud/gez/human_motion")
ACTION_RULES = {
    "applauding": {
        "intent_zh": "鼓掌/赞同",
        "tier": "core",
        "readability": "high",
        "context": "none",
    },
    "arm_wave": {
        "intent_zh": "手臂波浪动作（不能直接当作打招呼）",
        "tier": "generalization",
        "readability": "medium",
        "context": "none",
    },
    "blowing_kisses": {
        "intent_zh": "飞吻/友好",
        "tier": "auxiliary",
        "readability": "medium",
        "context": "face_and_hand_detail_removed",
    },
    "bowing_fullbody": {
        "intent_zh": "鞠躬/致意",
        "tier": "core",
        "readability": "high",
        "context": "none",
    },
    "bowing_waist": {
        "intent_zh": "弯腰鞠躬/致意",
        "tier": "core",
        "readability": "high",
        "context": "none",
    },
    "hailing_taxi": {
        "intent_zh": "招手引起注意",
        "tier": "core",
        "readability": "high",
        "context": "none",
    },
    "hugging_human": {
        "intent_zh": "拥抱/亲近",
        "tier": "interaction_probe",
        "readability": "medium",
        "context": "missing_second_person",
    },
    "salute": {
        "intent_zh": "敬礼/致意",
        "tier": "core",
        "readability": "high",
        "context": "none",
    },
}
EXPLICIT_EXCLUSIONS = {
    "boxing": "Fast-dynamics benchmark only; weak communicative semantics.",
    "dabbing": "Only one clip and its generated semantic label is invalid.",
    "fist_bump": "Needs a second person and sampled semantic labels are noisy.",
    "high_five": "Needs a second person and sampled semantic labels are noisy.",
    "rock_paper_scissors": "Meaning depends on fingers, which V2 intentionally omits.",
    "shaking_head": "Meaning depends on the head channel, which V2 intentionally omits.",
}
ACTION_PATTERN = re.compile(r"(?:_\d+)?_clip\d+$")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--nominal-fps", type=float, default=30.0)
    return parser.parse_args()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def action_from_stem(stem):
    return ACTION_PATTERN.sub("", stem)


def main():
    args = parse_args()
    root = args.root.resolve()
    motion_root = root / "raw/Motion-Xplusplus/extracted/haa500/smplx322"
    label_root = root / "raw/Motion-Xplusplus/extracted/haa500/semantic_label"
    output_root = root / "catalog"
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    missing_labels = []
    invalid_shapes = []
    for motion_path in sorted(motion_root.glob("*.npy")):
        action = action_from_stem(motion_path.stem)
        rule = ACTION_RULES.get(action)
        if rule is None:
            continue
        array = np.load(motion_path, mmap_mode="r")
        if array.ndim != 2 or array.shape[1] != 322:
            invalid_shapes.append(
                {"clip_id": motion_path.stem, "shape": list(array.shape)}
            )
            continue
        label_path = label_root / f"{motion_path.stem}.txt"
        if not label_path.exists():
            missing_labels.append(motion_path.stem)
            semantic_label = ""
        else:
            semantic_label = label_path.read_text(
                encoding="utf-8", errors="replace"
            ).strip()
        quality_path = root / f"processed/ula_v2_15d/v1/{motion_path.stem}/quality.json"
        preview_path = root / (
            f"processed/mujoco_previews/v1/{motion_path.stem}/"
            f"{motion_path.stem}_mujoco.mp4"
        )
        quality_pass = False
        if quality_path.exists():
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
            quality_pass = bool(quality.get("quality_gate", {}).get("passed", False))
        rows.append(
            {
                "clip_id": motion_path.stem,
                "action": action,
                **rule,
                "motion_relpath": str(motion_path.relative_to(root)),
                "label_relpath": str(label_path.relative_to(root)),
                "frame_count": int(array.shape[0]),
                "feature_dim": int(array.shape[1]),
                "nominal_fps": args.nominal_fps,
                "nominal_duration_sec": round(array.shape[0] / args.nominal_fps, 3),
                "semantic_label": semantic_label,
                "quality_relpath": (
                    str(quality_path.relative_to(root)) if quality_path.exists() else ""
                ),
                "preview_relpath": (
                    str(preview_path.relative_to(root)) if preview_path.exists() else ""
                ),
                "preview_available": preview_path.exists(),
                "manual_video_reviewed": False,
                "retarget_qc_pass": quality_pass,
                "accepted_for_training": False,
            }
        )

    csv_path = output_root / "expression_candidates_haa500.csv"
    fieldnames = list(rows[0]) if rows else []
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_csv, csv_path)

    summary = {
        "schema_version": 1,
        "source": "Motion-X++/HAA500 SMPL-X 322D",
        "purpose": "Review candidates for readable upper-body expression",
        "selection_policy": {
            "boxing_role": "high_speed_validation_only",
            "head_face_fingers_mapped": False,
            "raw_labels_are_generated_and_require_manual_video_review": True,
            "training_default": "deny_until_retarget_qc_and_manual_review",
        },
        "candidate_count": len(rows),
        "counts_by_action": dict(sorted(Counter(row["action"] for row in rows).items())),
        "counts_by_tier": dict(sorted(Counter(row["tier"] for row in rows).items())),
        "retarget_qc_pass_count": sum(row["retarget_qc_pass"] for row in rows),
        "preview_available_count": sum(row["preview_available"] for row in rows),
        "missing_labels": missing_labels,
        "invalid_shapes": invalid_shapes,
        "explicit_exclusions": EXPLICIT_EXCLUSIONS,
        "manifest": str(csv_path.relative_to(root)),
    }
    atomic_json(output_root / "expression_subset_haa500.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
