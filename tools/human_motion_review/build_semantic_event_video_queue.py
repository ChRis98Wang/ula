#!/usr/bin/env python3
"""Build a fail-closed MuJoCo review queue from semantic-event QC passes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROBOT_CONTRACT = "ula_v2_18d_head_v1"
EXPECTED_MASKS = {
    "official_category": False,
    "robot_observable_motion_form": False,
    "communicative_intent": False,
    "prompt_text": False,
    "legacy_gesture": False,
}
CATEGORY_ZH = {
    "deictic": "指示性上半身动作",
    "iconic": "图示性上半身动作",
    "metaphoric": "隐喻性上半身动作",
}
INTENSITY_ZH = {"low": "低强度", "medium": "中等强度", "high": "高强度"}
EMOTION_ZH = {
    "neutral": "中性",
    "sad": "悲伤",
    "happy": "愉快",
    "angry": "愤怒",
    "surprise": "惊讶",
    "fear": "恐惧",
}


def stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    atomic_text(path, "".join(stable_json(record) + "\n" for record in records))


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            records.append(value)
    return records


def _verified_file(record: dict[str, Any], path_key: str, hash_key: str) -> Path:
    task_id = record.get("task_id")
    value = record.get(path_key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{task_id}: missing {path_key}")
    path = Path(value).resolve()
    if not path.is_file() or sha256(path) != record.get(hash_key):
        raise ValueError(f"{task_id}: {path_key}/{hash_key} evidence mismatch")
    return path


def queue_record(record: dict[str, Any]) -> dict[str, Any]:
    task_id = str(record.get("task_id") or "")
    if record.get("status") != "passed":
        raise ValueError(f"{task_id}: only physical-QC passed records may be rendered")
    if record.get("accepted_for_training") is not False:
        raise ValueError(f"{task_id}: accepted_for_training must remain false")
    if record.get("official_category_verified") is not True:
        raise ValueError(f"{task_id}: official category is not verified")
    if record.get("robot_observable_motion_form") != "candidate_unreviewed":
        raise ValueError(f"{task_id}: robot motion form must remain candidate_unreviewed")
    if record.get("communicative_intent") != "candidate_unreviewed":
        raise ValueError(f"{task_id}: communicative intent must remain candidate_unreviewed")
    if record.get("canonical_prompt_role") != "coarse_category_only":
        raise ValueError(f"{task_id}: canonical prompt role is not coarse_category_only")
    if record.get("semantic_supervision_masks") != EXPECTED_MASKS:
        raise ValueError(f"{task_id}: semantic supervision masks are not fail-closed")

    event = record.get("semantic_event")
    if not isinstance(event, dict):
        raise ValueError(f"{task_id}: semantic_event is missing")
    category = str(event.get("category") or "")
    intensity = str(event.get("intensity") or "")
    emotion = str(record.get("emotion_id") or "")
    if category not in CATEGORY_ZH or intensity not in INTENSITY_ZH or emotion not in EMOTION_ZH:
        raise ValueError(f"{task_id}: unsupported official semantic stratum")
    if record.get("canonical_action") != f"official_gesture_category:{category}":
        raise ValueError(f"{task_id}: canonical action is not the official coarse category")
    if record.get("canonical_action_role") != "official_category_metadata_split_key_only":
        raise ValueError(f"{task_id}: canonical action role is not metadata split-key only")
    if record.get("semantic_mapping_status") != "official_category_verified_metadata_only":
        raise ValueError(f"{task_id}: semantic mapping status is not metadata-only verified")
    if record.get("official_category_role") != "verified_metadata_split_and_evaluation_only":
        raise ValueError(f"{task_id}: official category role is not metadata-only")
    if (
        "official_category_condition_channel" not in record
        or record["official_category_condition_channel"] is not None
    ):
        raise ValueError(f"{task_id}: official category condition channel must be null")
    if "official_category_loss" not in record or record["official_category_loss"] is not None:
        raise ValueError(f"{task_id}: official category loss must be null")
    if record.get("official_category_conditioning_enabled") is not False:
        raise ValueError(f"{task_id}: official category conditioning must be disabled")
    if record.get("emotion_supervision_mask") is not False:
        raise ValueError(f"{task_id}: source emotion must not directly supervise motion")
    if record.get("source_emotion_label_verified") is not True:
        raise ValueError(f"{task_id}: source emotion label is not verified")
    if record.get("emotion_supervision_role") != "disabled_pending_robot_affect_review":
        raise ValueError(f"{task_id}: source emotion supervision role is invalid")
    if record.get("official_emotion_conditioning_enabled") is not False:
        raise ValueError(f"{task_id}: official emotion conditioning must be disabled")
    if (
        "official_emotion_condition_channel" not in record
        or record["official_emotion_condition_channel"] is not None
    ):
        raise ValueError(f"{task_id}: official emotion condition channel must be null")
    if "official_emotion_loss" not in record or record["official_emotion_loss"] is not None:
        raise ValueError(f"{task_id}: official emotion loss must be null")
    if record.get("affect_observable_review_status") != "candidate_unreviewed":
        raise ValueError(f"{task_id}: affect observability must remain candidate_unreviewed")
    if record.get("affect_observable_supervision_mask") is not False:
        raise ValueError(f"{task_id}: affect observability supervision must remain disabled")
    quality_gate = record.get("quality_gate")
    if not isinstance(quality_gate, dict) or quality_gate.get("passed") is not True:
        raise ValueError(f"{task_id}: quality_gate.passed must be true")
    strict_quality_gates = {
        key: value for key, value in quality_gate.items() if key != "passed"
    }
    if not strict_quality_gates or not all(
        value is True for value in strict_quality_gates.values()
    ):
        raise ValueError(f"{task_id}: every strict quality gate must be boolean true")
    fps = record.get("fps")
    segment = record.get("retarget_segment")
    segment_fps = segment.get("fps") if isinstance(segment, dict) else None
    if fps != 30 or segment_fps != 30:
        raise ValueError(f"{task_id}: formal review trajectory must be 30 Hz")
    prompt = record.get("canonical_prompt")
    prompt_en = prompt.get("en") if isinstance(prompt, dict) else None
    if not isinstance(prompt_en, str) or not prompt_en.strip():
        raise ValueError(f"{task_id}: canonical_prompt.en is missing")

    trajectory = _verified_file(record, "safe_csv", "safe_csv_sha256")
    quality = _verified_file(record, "quality_json", "quality_json_sha256")
    return {
        "schema_version": "1.0.0",
        "task_id": task_id,
        "status": "passed",
        "source_clip_id": record.get("source_clip_id"),
        "speaker_key": record.get("speaker_key"),
        "official_split": record.get("official_split"),
        "fixed_split_assignment": record.get("fixed_split_assignment"),
        "robot_contract": ROBOT_CONTRACT,
        "fps": 30,
        "canonical_action": record.get("canonical_action"),
        "canonical_action_role": record.get("canonical_action_role"),
        "canonical_prompt": {
            "en": prompt_en,
            "zh": (
                f"执行一个{INTENSITY_ZH[intensity]}的{CATEGORY_ZH[category]}，"
                f"情绪类别为{EMOTION_ZH[emotion]}；具体意图待视频复核。"
            ),
        },
        "canonical_prompt_role": "coarse_category_only",
        "official_category_verified": True,
        "official_category_role": record.get("official_category_role"),
        "official_category_condition_channel": record.get(
            "official_category_condition_channel"
        ),
        "official_category_loss": record.get("official_category_loss"),
        "official_category_conditioning_enabled": False,
        "robot_observable_motion_form": "candidate_unreviewed",
        "communicative_intent": "candidate_unreviewed",
        "semantic_supervision_masks": dict(EXPECTED_MASKS),
        "semantic_event": event,
        "semantic_mapping_status": record.get("semantic_mapping_status"),
        "emotion_id": emotion,
        "emotion_supervision_mask": False,
        "source_emotion_label_verified": True,
        "emotion_supervision_role": "disabled_pending_robot_affect_review",
        "official_emotion_conditioning_enabled": False,
        "official_emotion_condition_channel": None,
        "official_emotion_loss": None,
        "affect_observable_review_status": record[
            "affect_observable_review_status"
        ],
        "affect_observable_supervision_mask": record[
            "affect_observable_supervision_mask"
        ],
        "trajectory_path": str(trajectory),
        "trajectory_sha256": record["safe_csv_sha256"],
        "safe_csv": str(trajectory),
        "safe_csv_sha256": record["safe_csv_sha256"],
        "quality_json": str(quality),
        "quality_json_sha256": record["quality_json_sha256"],
        "quality_gate": dict(quality_gate),
        "retarget_segment": segment,
        "upstream_inventory_record_sha256": record.get(
            "upstream_inventory_record_sha256"
        ),
        "selected_record_sha256": record.get("selected_record_sha256"),
        "retarget_input_manifest_sha256": record.get(
            "retarget_input_manifest_sha256"
        ),
        "review_state": "pending_independent_motion_text_video_review",
        "manual_review_required": True,
        "semantic_action_completeness_review_required": True,
        "affect_observable_review_required": True,
        "render_pass_grants_training_admission": False,
        "accepted_for_training": False,
    }


def build_queue(passed_paths: list[Path], output: Path) -> dict[str, Any]:
    records = []
    seen = set()
    for path in passed_paths:
        for source in read_jsonl(path.resolve()):
            queued = queue_record(source)
            if queued["task_id"] in seen:
                raise ValueError(f"duplicate task_id across passed manifests: {queued['task_id']}")
            seen.add(queued["task_id"])
            records.append(queued)
    records.sort(key=lambda item: item["task_id"])
    atomic_jsonl(output.resolve(), records)
    summary = {
        "artifact_kind": "semantic_event_physical_qc_pass_video_review_queue",
        "records": len(records),
        "counts_by_category": dict(
            sorted(Counter(item["semantic_event"]["category"] for item in records).items())
        ),
        "counts_by_emotion": dict(
            sorted(Counter(item["emotion_id"] for item in records).items())
        ),
        "passed_manifests": [str(path.resolve()) for path in passed_paths],
        "passed_manifest_sha256": [sha256(path.resolve()) for path in passed_paths],
        "output": str(output.resolve()),
        "output_sha256": sha256(output.resolve()),
        "accepted_for_training": 0,
        "manual_review_required": True,
    }
    atomic_json(output.with_suffix(".summary.json").resolve(), summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passed-manifest", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_queue(args.passed_manifest, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
