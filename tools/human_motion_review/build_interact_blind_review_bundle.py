#!/usr/bin/env python3
"""Build anonymous InterAct axis, dyadic semantics, and affect review queues."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import shutil
import stat
import tempfile
from typing import Any, Iterable

try:
    from tools.human_motion_review.render_beat2_annotation_review import validate_video
except ModuleNotFoundError:  # pragma: no cover - direct invocation by path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.human_motion_review.render_beat2_annotation_review import validate_video


SCHEMA_VERSION = "1.0.0"
AXIS_PROTOCOL = "interact_robot_axis_blind_video_v1"
ARC_ACTION_PROTOCOL = "interact_dyadic_arc_action_blind_video_v1"
AFFECT_PROTOCOL = "interact_dyadic_affect_blind_video_v1"
PUBLIC_VIDEO_MODE = 0o644
DEFAULT_RECEIPT = Path(
    "/home/gez/nas/cloud/gez/human_motion/catalog/"
    "interact_axis_smoke_four_performance_v1_receipt.json"
)
DEFAULT_STAGING = Path(
    "/home/gez/nas/cloud/gez/human_motion/review/"
    "interact_blind_expression_v1/staging"
)
DEFAULT_OUTPUT = Path(
    "/home/gez/nas/cloud/gez/human_motion/review/interact_blind_expression_v1"
)
DEFAULT_HIDDEN = Path(
    "/home/gez/shuaiwang/.private_human_motion/interact_blind_expression_v1"
)
FORBIDDEN_PUBLIC_KEYS = {
    "actor_id",
    "canonical_action",
    "date",
    "emotion",
    "emotion_id",
    "emotion_label",
    "official_emotion",
    "performance_id",
    "scenario",
    "scenario_id",
    "source",
    "source_bvh",
    "source_clip_id",
    "source_text",
    "task_id",
    "transcript",
    "turn_id",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--dyad-staging-dir", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hidden-root", type=Path, default=DEFAULT_HIDDEN)
    parser.add_argument("--secret-hex")
    return parser.parse_args()


def stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, payload: str, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    if mode is not None:
        os.chmod(temporary, mode)
    os.replace(temporary, path)


def atomic_json(path: Path, value: object, *, mode: int | None = None) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        mode=mode,
    )


def atomic_jsonl(
    path: Path, rows: Iterable[dict[str, Any]], *, mode: int | None = None
) -> None:
    atomic_text(path, "".join(stable_json(row) + "\n" for row in rows), mode=mode)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_tasks(path: Path) -> dict[str, dict[str, Any]]:
    tasks = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                tasks[value["episode_task_id"]] = value
    return tasks


def walk_keys(value: object):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_keys(child)


def assert_public_privacy(record: dict[str, Any]) -> None:
    for key in walk_keys(record):
        lowered = key.lower()
        if (
            lowered in FORBIDDEN_PUBLIC_KEYS
            or lowered.startswith("official_")
            or lowered.startswith("source_")
            or "transcript" in lowered
        ):
            raise ValueError(f"Public InterAct blind record leaks forbidden key: {key}")


def bundle_secret(hidden_root: Path, provided_hex: str | None) -> bytes:
    path = hidden_root / "bundle_secret.json"
    if path.is_file():
        value = load_json(path)
        secret = bytes.fromhex(value["secret_hex"])
        if provided_hex is not None and bytes.fromhex(provided_hex) != secret:
            raise ValueError("Provided secret does not match existing InterAct bundle")
        os.chmod(path, 0o600)
        return secret
    secret = bytes.fromhex(provided_hex) if provided_hex else secrets.token_bytes(32)
    if len(secret) < 16:
        raise ValueError("InterAct blind secret must contain at least 16 bytes")
    atomic_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "interact_blind_bundle_secret",
            "secret_hex": secret.hex(),
            "public_distribution_forbidden": True,
        },
        mode=0o600,
    )
    return secret


def enforce_hidden_permissions(hidden_root: Path) -> None:
    os.chmod(hidden_root, 0o700)
    for path in hidden_root.iterdir():
        if path.is_file():
            os.chmod(path, 0o600)
    if (hidden_root.stat().st_mode & 0o777) != 0o700:
        raise PermissionError("InterAct hidden root does not enforce mode 0700")
    unsafe = [
        str(path)
        for path in hidden_root.iterdir()
        if path.is_file() and (path.stat().st_mode & 0o777) != 0o600
    ]
    if unsafe:
        raise PermissionError("InterAct hidden files do not enforce mode 0600: " + ", ".join(unsafe))


def anonymous_id(secret: bytes, namespace: str, identifier: str, video_hash: str) -> str:
    payload = f"{namespace}\0{identifier}\0{video_hash}".encode("utf-8")
    return namespace + "_" + hmac.new(secret, payload, hashlib.sha256).hexdigest()[:24]


def materialize_video(source: Path, target: Path, expected_hash: str) -> None:
    source = source.resolve()
    target = Path(os.path.abspath(target))
    if source == target:
        raise ValueError("Anonymous video target must differ from its source")
    if not source.is_file() or sha256_file(source) != expected_hash:
        raise ValueError(f"Anonymous video source integrity failure: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.is_file() or sha256_file(target) != expected_hash:
            raise ValueError(f"Anonymous video mismatch: {target}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        os.chmod(temporary, PUBLIC_VIDEO_MODE)
        if sha256_file(temporary) != expected_hash:
            raise ValueError(f"Anonymous video temporary copy mismatch: {target}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)

    target_stat = target.lstat()
    if (
        not stat.S_ISREG(target_stat.st_mode)
        or target_stat.st_nlink != 1
        or os.path.samefile(source, target)
        or sha256_file(target) != expected_hash
    ):
        raise ValueError(f"Anonymous video integrity failure: {target}")


def axis_record(sample_id: str, video: Path, video_hash: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "video_path": str(video.resolve()),
        "video_sha256": video_hash,
        "audio_available": False,
        "label_metadata_exposed": False,
        "protocol_version": AXIS_PROTOCOL,
        "review_id": None,
        "reviewer_id": None,
        "left_right_identity_result": None,
        "raise_lower_direction_result": None,
        "forward_backward_direction_result": None,
        "elbow_branch_result": None,
        "head_neck_direction_result": None,
        "overall_result": None,
        "review_notes": None,
    }


def arc_action_record(sample_id: str, video: Path, video_hash: str) -> dict[str, Any]:
    actor_fields = {
        "observable_motion_description_en": None,
        "communicative_intent_en": None,
        "robot_prompt_en": None,
        "robot_prompt_sha256": None,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "video_path": str(video.resolve()),
        "video_sha256": video_hash,
        "context_level": 0,
        "audio_available": False,
        "label_metadata_exposed": False,
        "protocol_version": ARC_ACTION_PROTOCOL,
        "arc_review_id": None,
        "arc_reviewer_id": None,
        "onset_status": None,
        "onset_evidence_frame": None,
        "apex_status": None,
        "apex_evidence_frame": None,
        "offset_status": None,
        "offset_evidence_frame": None,
        "expression_completeness_result": None,
        "expansion_request": None,
        "action_review_id": None,
        "action_reviewer_id": None,
        "interaction_observable_result": None,
        "interaction_description_en": None,
        "actor_a": dict(actor_fields),
        "actor_b": dict(actor_fields),
        "review_notes": None,
    }


def affect_record(sample_id: str, video: Path, video_hash: str) -> dict[str, Any]:
    actor_fields = {
        "result": None,
        "predicted_class": None,
        "confidence": None,
        "intensity": None,
        "evidence": None,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "video_path": str(video.resolve()),
        "video_sha256": video_hash,
        "context_level": 0,
        "audio_available": False,
        "label_metadata_exposed": False,
        "protocol_version": AFFECT_PROTOCOL,
        "review_id": None,
        "reviewer_id": None,
        "allowed_classes": ["neutral", "sad", "happy", "angry", "surprise", "fear"],
        "actor_a": dict(actor_fields),
        "actor_b": dict(actor_fields),
        "interaction_affect_relation_en": None,
        "review_notes": None,
    }


def main() -> None:
    args = parse_args()
    receipt_path = args.receipt.resolve()
    receipt = load_json(receipt_path)
    selected = receipt.get("selected") or []
    if len(selected) != 8 or receipt.get("accepted_for_training") is not False:
        raise ValueError("InterAct blind bundle requires the fail-closed eight-task receipt")
    if receipt.get("admission_gate", {}).get("axis_visual_blind_review_passed") is not False:
        raise ValueError("Axis visual review must be pending before bundle creation")

    output_root = args.output_root.resolve()
    public_root = output_root / "public"
    hidden_root = args.hidden_root.resolve()
    hidden_root.mkdir(parents=True, exist_ok=True)
    os.chmod(hidden_root, 0o700)
    secret = bundle_secret(hidden_root, args.secret_hex)
    task_manifest = Path(receipt["catalog_task_manifest"]).resolve()
    tasks = load_tasks(task_manifest)

    axis_queue = []
    axis_hidden = []
    for row in sorted(selected, key=lambda item: item["episode_task_id"]):
        task_id = row["episode_task_id"]
        source_video = Path(row["artifacts"]["source_vs_robot_mp4"]).resolve()
        video_hash = row["artifacts"]["source_vs_robot_mp4_sha256"]
        if sha256_file(source_video) != video_hash:
            raise ValueError(f"Axis video SHA mismatch: {task_id}")
        sample_id = anonymous_id(secret, "axis", task_id, video_hash)
        anonymous_video = public_root / "videos" / f"{sample_id}.mp4"
        materialize_video(source_video, anonymous_video, video_hash)
        public = axis_record(sample_id, anonymous_video, video_hash)
        assert_public_privacy(public)
        axis_queue.append(public)
        axis_hidden.append(
            {
                "sample_id": sample_id,
                "episode_task_id": task_id,
                "performance_id": row["performance_id"],
                "turn_id": row["turn_id"],
                "target_actor_id": row["target_actor_id"],
                "partner_actor_id": row["partner_actor_id"],
                "video_sha256": video_hash,
                "accepted_for_training": False,
            }
        )

    by_turn: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        by_turn.setdefault(row["turn_id"], []).append(row)
    if len(by_turn) != 4 or any(len(rows) != 2 for rows in by_turn.values()):
        raise ValueError("Dyadic bundle requires four complete partner turns")
    arc_queue = []
    affect_queue = []
    dyad_hidden = []
    staging = args.dyad_staging_dir.resolve()
    for turn_id, rows in sorted(by_turn.items()):
        summary_path = staging / f"{turn_id}_dyad_summary.json"
        summary = load_json(summary_path)
        source_video = Path(summary["output_mp4"]).resolve()
        video_hash = summary["output_mp4_sha256"]
        if sha256_file(source_video) != video_hash:
            raise ValueError(f"Dyadic video SHA mismatch: {turn_id}")
        check = validate_video(
            source_video,
            expected_frames=int(summary["frames"]),
            expected_width=1280,
            expected_height=720,
            expected_fps=30.0,
        )
        sample_id = anonymous_id(secret, "dyad", turn_id, video_hash)
        anonymous_video = public_root / "videos" / f"{sample_id}.mp4"
        materialize_video(source_video, anonymous_video, video_hash)
        arc_public = arc_action_record(sample_id, anonymous_video, video_hash)
        affect_public = affect_record(sample_id, anonymous_video, video_hash)
        assert_public_privacy(arc_public)
        assert_public_privacy(affect_public)
        arc_queue.append(arc_public)
        affect_queue.append(affect_public)

        hashes_to_role = {
            summary["actor_a_bvh_sha256"]: "A",
            summary["actor_b_bvh_sha256"]: "B",
        }
        actor_mapping = {}
        for row in rows:
            role = hashes_to_role.get(row["source_bvh_sha256"])
            if role is None or role in actor_mapping:
                raise ValueError(f"Cannot bind anonymous dyad roles: {turn_id}")
            actor_mapping[role] = {
                "episode_task_id": row["episode_task_id"],
                "actor_id": row["target_actor_id"],
                "partner_actor_id": row["partner_actor_id"],
            }
        context_plan = tasks[rows[0]["episode_task_id"]]["context_plan"]
        dyad_hidden.append(
            {
                "sample_id": sample_id,
                "performance_id": rows[0]["performance_id"],
                "turn_id": turn_id,
                "actor_mapping": actor_mapping,
                "displayed_context_level": 0,
                "context_plan": context_plan,
                "dyad_summary": str(summary_path),
                "dyad_summary_sha256": sha256_file(summary_path),
                "video_validation": check,
                "official_scenario_or_emotion_exposed": False,
                "accepted_for_training": False,
            }
        )

    axis_queue.sort(key=lambda item: item["sample_id"])
    arc_queue.sort(key=lambda item: item["sample_id"])
    affect_queue.sort(key=lambda item: item["sample_id"])
    axis_hidden.sort(key=lambda item: item["sample_id"])
    dyad_hidden.sort(key=lambda item: item["sample_id"])
    paths = {
        "axis": public_root / "axis_review_queue.jsonl",
        "arc_action": public_root / "arc_action_review_queue.jsonl",
        "affect": public_root / "affect_review_queue.jsonl",
        "axis_hidden": hidden_root / "axis_mapping.jsonl",
        "dyad_hidden": hidden_root / "dyad_mapping.jsonl",
    }
    atomic_jsonl(paths["axis"], axis_queue)
    atomic_jsonl(paths["arc_action"], arc_queue)
    atomic_jsonl(paths["affect"], affect_queue)
    atomic_jsonl(paths["axis_hidden"], axis_hidden, mode=0o600)
    atomic_jsonl(paths["dyad_hidden"], dyad_hidden, mode=0o600)
    public_summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "interact_separate_anonymous_blind_review_bundle",
        "axis_records": len(axis_queue),
        "dyadic_arc_action_records": len(arc_queue),
        "dyadic_affect_records": len(affect_queue),
        "axis_queue": str(paths["axis"]),
        "axis_queue_sha256": sha256_file(paths["axis"]),
        "arc_action_queue": str(paths["arc_action"]),
        "arc_action_queue_sha256": sha256_file(paths["arc_action"]),
        "affect_queue": str(paths["affect"]),
        "affect_queue_sha256": sha256_file(paths["affect"]),
        "identity_scenario_official_text_or_emotion_exposed": False,
        "arc_action_and_affect_reviewers_must_be_independent": True,
        "duration_is_not_a_review_or_admission_gate": True,
        "incomplete_arc_action_requires_next_predeclared_natural_context_level": True,
        "accepted_for_training": False,
    }
    hidden_summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "interact_hidden_blind_mapping",
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "task_manifest": str(task_manifest),
        "task_manifest_sha256": sha256_file(task_manifest),
        "axis_mapping": str(paths["axis_hidden"]),
        "axis_mapping_sha256": sha256_file(paths["axis_hidden"]),
        "dyad_mapping": str(paths["dyad_hidden"]),
        "dyad_mapping_sha256": sha256_file(paths["dyad_hidden"]),
        "public_distribution_forbidden": True,
        "accepted_for_training": False,
    }
    atomic_json(public_root / "summary.json", public_summary)
    atomic_json(hidden_root / "summary.json", hidden_summary, mode=0o600)
    enforce_hidden_permissions(hidden_root)
    print(json.dumps({"public": public_summary, "hidden": hidden_summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
