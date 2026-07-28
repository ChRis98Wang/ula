#!/usr/bin/env python3
"""Join verified BEAT2 18D motion with ordinary-speaking realization labels."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from upper_body_skeleton.robot_observable_motion_realizations import (
    REALIZATION_ID,
    validate_conversational_realization_annotation,
)
from upper_body_skeleton.ula_v2_conversational_realization_episode import (
    FORMAL_ELIGIBILITY_MODE,
    FORMAL_EPISODE_CONTRACT,
    PROMPT_TEXT_PROVENANCE,
    TRAINING_SEGMENT_REPRESENTATION,
)


DEFAULT_MOTION_MANIFEST = Path(
    "/home/gez/nas/cloud/gez/human_motion/processed/"
    "beat2_semantic_event_training_pool_18d_v8/expansion/release/"
    "adjudication_min30f/train_ready.jsonl"
)
DEFAULT_REALIZATION_MANIFEST = (
    PROJECT_ROOT
    / "deliverables/expressive_human_motion_v2/robot_observable_intents_v1/"
    "beat2_full12148_intent_candidates_v9_tiera/motion_realization_train_ready.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "deliverables/expressive_human_motion_v2/robot_observable_intents_v1/"
    "beat2_conversational_realization_release_v9_formal"
)
DATASET_SOURCE = "beat2_official_semantic_event_training_pool_v8_expanded"
PROVENANCE_LOCK_KIND = "ula_v2_conversational_realization_v9_provenance_lock_v1"


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            records.append(value)
    if not records:
        raise ValueError(f"manifest is empty: {path}")
    return records


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, records: list[Mapping[str, Any]]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
    os.replace(temporary, path)


def build_release(
    motion_manifest: str | Path,
    realization_manifest: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    motion_manifest = Path(motion_manifest).resolve()
    realization_manifest = Path(realization_manifest).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    motion_rows = _read_jsonl(motion_manifest)
    realization_rows = _read_jsonl(realization_manifest)
    motion_sha256 = _sha256_file(motion_manifest)
    realization_sha256 = _sha256_file(realization_manifest)
    motion_by_id = {str(row.get("clip_id") or row.get("task_id") or ""): row for row in motion_rows}
    realization_by_id = {
        str(row.get("task_id") or row.get("clip_id") or ""): row
        for row in realization_rows
    }
    if "" in motion_by_id or len(motion_by_id) != len(motion_rows):
        raise ValueError("motion manifest has missing or duplicate clip ids")
    if "" in realization_by_id or len(realization_by_id) != len(realization_rows):
        raise ValueError("realization manifest has missing or duplicate clip ids")
    if set(motion_by_id) != set(realization_by_id):
        missing = sorted(set(motion_by_id) - set(realization_by_id))
        extra = sorted(set(realization_by_id) - set(motion_by_id))
        raise ValueError(f"motion/realization membership differs: missing={missing[:5]}, extra={extra[:5]}")

    formal: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    frame_counts: list[int] = []
    prompt_counts: dict[str, int] = {}
    for clip_id, source in motion_by_id.items():
        realization_row = realization_by_id[clip_id]
        realization = deepcopy(realization_row.get("motion_realization"))
        validate_conversational_realization_annotation(realization)
        if realization_row.get("accepted_for_motion_realization_training") is not True:
            raise ValueError(f"{clip_id}: realization record is not train-ready")
        if realization_row.get("accepted_for_intent_training") is not False:
            raise ValueError(f"{clip_id}: realization record improperly grants intent admission")
        if realization_row.get("trajectory_sha256") != source.get("safe_csv_sha256"):
            raise ValueError(f"{clip_id}: trajectory binding differs between manifests")
        if realization_row.get("fixed_split_assignment") != source.get("fixed_split_assignment"):
            raise ValueError(f"{clip_id}: fixed split changed")
        motion = deepcopy(source.get("motion_18d"))
        if not isinstance(motion, dict) or motion.get("state") != "passed":
            raise ValueError(f"{clip_id}: source lacks passed motion_18d")
        if motion.get("safe_csv_sha256") != source.get("safe_csv_sha256"):
            raise ValueError(f"{clip_id}: nested and top-level trajectory hashes differ")
        frames = int(source.get("frames") or 0)
        start = int(source.get("start_frame") or 0)
        end = int(source.get("end_frame_exclusive") or 0)
        source_frames = end - start
        retarget_segment = deepcopy(source.get("retarget_segment"))
        if (
            frames < 3
            or source_frames < 2
            or not isinstance(retarget_segment, dict)
            or retarget_segment.get("source_start_frame") != start
            or retarget_segment.get("source_end_frame_exclusive") != end
            or retarget_segment.get("source_frame_count") != source_frames
            or retarget_segment.get("output_frame_count") != frames
        ):
            raise ValueError(f"{clip_id}: native source interval is invalid")
        prompt = str(realization["motion_realization_prompt"]["en"])
        source_record_sha256 = _canonical_sha256(source)
        realization_record_sha256 = _canonical_sha256(realization_row)
        training_admission = {
            "contract": FORMAL_EPISODE_CONTRACT,
            "trajectory_sha256": source["safe_csv_sha256"],
            "source_record_sha256": source_record_sha256,
            "realization_record_sha256": realization_record_sha256,
            "motion_realization_ontology_sha256": realization[
                "motion_realization_ontology_sha256"
            ],
            "motion_realization_id": REALIZATION_ID,
            "training_channel_masks": {
                "motion": True,
                "motion_realization": True,
                "primary_intent": False,
                "emotion": False,
                "audio": False,
            },
        }
        formal.append(
            {
                "schema_version": "1.0.0",
                "artifact_kind": "ula_v2_conversational_realization_v9_train_episode",
                "formal_episode_contract": FORMAL_EPISODE_CONTRACT,
                "eligibility_mode": FORMAL_ELIGIBILITY_MODE,
                "accepted_for_training": True,
                "accepted_for_motion_realization_training": True,
                "native_variable_length": True,
                "clip_id": clip_id,
                "task_id": clip_id,
                "dataset_source": DATASET_SOURCE,
                "source_clip_id": source["source_clip_id"],
                "speaker_key": source["speaker_key"],
                "source_group_key": source["source_group_key"],
                "fixed_split_assignment": source["fixed_split_assignment"],
                "official_split": source.get("official_split"),
                "fps": 30.0,
                "frames": frames,
                "duration_sec": (frames - 1) / 30.0,
                "motion_18d": motion,
                "trajectory_path": source["safe_csv"],
                "trajectory_sha256": source["safe_csv_sha256"],
                "quality_gate": deepcopy(source["quality_gate"]),
                "retarget_segment": retarget_segment,
                "retarget_qc_passed": True,
                "training_segment": {
                    "representation": TRAINING_SEGMENT_REPRESENTATION,
                    "start_frame": start,
                    "end_frame_exclusive": end,
                    "frame_count": source_frames,
                    "output_frame_count": frames,
                    "fixed_window_sec": None,
                    "cropped": False,
                },
                "motion_realization": realization,
                "motion_realization_supervision_mask": True,
                "motion_style": deepcopy(realization_row["motion_style"]),
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt_text_provenance": PROMPT_TEXT_PROVENANCE,
                "source_transcript_semantics_used": False,
                "source_filename_semantics_used": False,
                "observable_intent_id": None,
                "intent_supervision_mask": False,
                "intent_conditioning_mask": False,
                "emotion_id": None,
                "emotion_supervision_mask": False,
                "emotion_conditioning_mask": False,
                "behavior_supervision_mask": False,
                "audio_conditioning_enabled": False,
                "source_emotion_metadata_only": source.get("emotion_id"),
                "source_manifest": str(motion_manifest),
                "source_manifest_sha256": motion_sha256,
                "source_record_sha256": source_record_sha256,
                "realization_manifest": str(realization_manifest),
                "realization_manifest_sha256": realization_sha256,
                "realization_record_sha256": realization_record_sha256,
                "training_admission": training_admission,
            }
        )
        split = str(source["fixed_split_assignment"])
        split_counts[split] = split_counts.get(split, 0) + 1
        frame_counts.append(frames)
        prompt_counts[prompt] = prompt_counts.get(prompt, 0) + 1

    output_manifest = output_dir / "train_ready.jsonl"
    _atomic_jsonl(output_manifest, formal)
    manifest_sha256 = _sha256_file(output_manifest)
    dataset_scale = {
        "episode_count": len(formal),
        "frame_count": sum(frame_counts),
        "sample_span_sec": sum((frames - 1) / 30.0 for frames in frame_counts),
        "source_group_count": len({row["source_group_key"] for row in formal}),
        "speaker_count": len({row["speaker_key"] for row in formal}),
        "distinct_frame_counts": len(set(frame_counts)),
        "minimum_frames": min(frame_counts),
        "maximum_frames": max(frame_counts),
        "distinct_prompts": len(prompt_counts),
        "fixed_split_counts": dict(sorted(split_counts.items())),
    }
    lock = {
        "schema_version": 1,
        "artifact_kind": PROVENANCE_LOCK_KIND,
        "accepted_for_training": True,
        "formal_release_allowed": True,
        "experimental_local_processing_allowed": True,
        "formal_episode_contract": FORMAL_EPISODE_CONTRACT,
        "duration_policy": "native_variable_length_conversational_event_no_fixed_duration_v1",
        "source_count": 1,
        "source_manifest": str(motion_manifest),
        "source_manifest_sha256": motion_sha256,
        "realization_manifest": str(realization_manifest),
        "realization_manifest_sha256": realization_sha256,
        "train_ready_manifest": str(output_manifest),
        "train_ready_manifest_sha256": manifest_sha256,
        "motion_realization_ontology_sha256": formal[0]["motion_realization"][
            "motion_realization_ontology_sha256"
        ],
        "dataset_scale": dataset_scale,
        "minimum_training_scale": {"episode_count": 10000, "speaker_count": 20},
        "scale_gate_passed": len(formal) >= 10000 and dataset_scale["speaker_count"] >= 20,
        "audio_conditioning_enabled": False,
        "primary_intent_conditioning_enabled": False,
        "emotion_conditioning_enabled": False,
        "license_gate": {
            "allowed_scope": "non-commercial_research_only",
            "authorized_via": "noncommercial_research_user_confirmation_v1",
            "dataset_card_declared_license": "apache-2.0",
            "dataset_training_use_review_status": (
                "confirmed_noncommercial_research_by_user"
            ),
            "formal_release_blocked": False,
            "official_project_statement": "Non-commercial",
            "smplx_terms_review_status": "pending_human_review",
            "terms_status": "conflicting_upstream_statements",
            "training_authorized_by_this_lock": True,
        },
        "locked_artifacts": {
            "acquisition_receipt": {
                "path": (
                    "/home/gez/nas/cloud/gez/human_motion/catalog/"
                    "beat2_motion_only_acquisition.json"
                ),
                "sha256": "551350dec66d9d050c11ea1e036a47ef7f7c18d03f811bc8fc44683bfbbf3ddd",
            },
            "source_motion_manifest": {
                "path": str(motion_manifest),
                "sha256": motion_sha256,
            },
            "motion_realization_manifest": {
                "path": str(realization_manifest),
                "sha256": realization_sha256,
            },
            "train_ready_manifest": {
                "path": str(output_manifest),
                "sha256": manifest_sha256,
            },
            "user_confirmation_receipt": {
                "path": (
                    "/home/gez/nas/cloud/gez/human_motion/processed/"
                    "beat2_semantic_event_training_pool_18d_v7_full/adjudication/"
                    "user_confirmation_receipt.json"
                ),
                "sha256": "038a47604cc3294e5473160976712f10a288a68303347d4058a5fc92eba70509",
            },
        },
    }
    lock_path = output_dir / "provenance_lock.json"
    _atomic_json(lock_path, lock)
    summary = {
        "artifact_kind": "beat2_conversational_realization_release_v9_summary",
        "formal_episode_contract": FORMAL_EPISODE_CONTRACT,
        "train_ready_manifest": str(output_manifest),
        "train_ready_manifest_sha256": manifest_sha256,
        "provenance_lock": str(lock_path),
        "provenance_lock_sha256": _sha256_file(lock_path),
        "dataset_scale": dataset_scale,
        "scale_gate_passed": lock["scale_gate_passed"],
        "specific_intent_train_ready_count": 0,
        "ordinary_speaking_realization_train_ready_count": len(formal),
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-manifest", type=Path, default=DEFAULT_MOTION_MANIFEST)
    parser.add_argument("--realization-manifest", type=Path, default=DEFAULT_REALIZATION_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(
        json.dumps(
            build_release(args.motion_manifest, args.realization_manifest, args.output_dir),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
