#!/usr/bin/env python3
"""Render all conservative HAA500 interaction candidates for blind review."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.human_motion_collection.package_haa500_interaction_samples import (  # noqa: E402
    _codec_check,
    _peak_frame_index,
    _video_check,
)
from upper_body_skeleton.mujoco_playback import render_motion  # noqa: E402


SCHEMA_VERSION = "1.0.0"
RENDER_CONTRACT = "haa500_observable_intent_render_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_sha256(record: Any) -> str:
    encoded = json.dumps(
        record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            records.append(record)
    return records


def load_candidates(manifests: list[Path]) -> list[dict[str, Any]]:
    by_clip: dict[str, dict[str, Any]] = {}
    for manifest in manifests:
        for record in _read_jsonl(manifest):
            clip_id = record.get("clip_id")
            if not isinstance(clip_id, str) or not clip_id.strip():
                raise ValueError(f"{manifest}: candidate lacks clip_id")
            if clip_id in by_clip:
                raise ValueError(f"duplicate candidate clip_id: {clip_id}")
            by_clip[clip_id] = record
    return [by_clip[clip_id] for clip_id in sorted(by_clip)]


def _validate_candidate(candidate: dict[str, Any]) -> None:
    clip_id = candidate["clip_id"]
    if candidate.get("physical_qc", {}).get("passed") is not True:
        raise ValueError(f"{clip_id}: physical QC has not passed")
    semantic_review = candidate.get("semantic_review")
    if not isinstance(semantic_review, dict):
        raise ValueError(f"{clip_id}: semantic_review is missing")
    if semantic_review.get("accepted_for_training") is not False:
        raise ValueError(f"{clip_id}: candidate must remain excluded before review")
    if semantic_review.get("manual_video_review_required") is not True:
        raise ValueError(f"{clip_id}: manual video review is not required")
    trajectory = candidate.get("trajectory")
    if not isinstance(trajectory, dict):
        raise ValueError(f"{clip_id}: trajectory metadata is missing")
    path = Path(str(trajectory.get("path") or ""))
    if not path.is_file():
        raise FileNotFoundError(f"{clip_id}: missing trajectory {path}")
    if _sha256(path) != trajectory.get("sha256"):
        raise ValueError(f"{clip_id}: trajectory SHA256 mismatch")
    if trajectory.get("fps") != 30.0:
        raise ValueError(f"{clip_id}: trajectory must be 30 Hz")


def _validated_existing_render(
    candidate: dict[str, Any], video_path: Path, receipt_path: Path
) -> dict[str, Any] | None:
    if not video_path.is_file() or not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("render_contract") != RENDER_CONTRACT
        or receipt.get("source_candidate_sha256") != _record_sha256(candidate)
        or receipt.get("input_csv") != candidate["trajectory"]["path"]
        or receipt.get("video_sha256") != _sha256(video_path)
    ):
        return None
    checks = receipt.get("checks")
    if not isinstance(checks, dict) or any(
        checks.get(field) is not True
        for field in ("nonblank", "has_motion", "full_frame_uncropped")
    ):
        return None
    return receipt


def render_candidate(
    candidate: dict[str, Any], output_dir: Path, *, resume: bool
) -> dict[str, Any]:
    _validate_candidate(candidate)
    clip_id = candidate["clip_id"]
    video_path = output_dir / "videos" / f"{clip_id}.mp4"
    receipt_path = output_dir / "receipts" / f"{clip_id}.json"
    if resume:
        existing = _validated_existing_render(candidate, video_path, receipt_path)
        if existing is not None:
            return existing

    video_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory = candidate["trajectory"]
    render = render_motion(
        trajectory["path"],
        video_path,
        fps=30.0,
        width=1280,
        height=720,
        camera_margin=1.25,
        camera_lookat_z_offset=-0.08,
    )
    if render.get("output_contract") != "ula_v2_18d_head_v1":
        raise ValueError(f"{clip_id}: renderer did not use the 18D head contract")
    if render.get("frames") != trajectory.get("frames"):
        raise ValueError(f"{clip_id}: rendered frame count mismatch")
    checks, _ = _video_check(video_path, int(trajectory["frames"]))
    checks.update(_codec_check(video_path))
    receipt = {
        **render,
        "schema_version": SCHEMA_VERSION,
        "render_contract": RENDER_CONTRACT,
        "source_candidate_sha256": _record_sha256(candidate),
        "video_sha256": _sha256(video_path),
        "checks": checks,
        "representative_frame": _peak_frame_index(Path(trajectory["path"])),
    }
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def build_package(
    *,
    candidate_manifests: list[Path],
    output_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    candidates = load_candidates(candidate_manifests)
    package_records: list[dict[str, Any]] = []
    resumed = 0
    for index, candidate in enumerate(candidates, start=1):
        video_path = output_dir / "videos" / f"{candidate['clip_id']}.mp4"
        receipt_path = output_dir / "receipts" / f"{candidate['clip_id']}.json"
        existing = (
            _validated_existing_render(candidate, video_path, receipt_path)
            if resume
            else None
        )
        if existing is not None:
            receipt = existing
            resumed += 1
        else:
            print(f"[{index}/{len(candidates)}] render {candidate['clip_id']}", flush=True)
            receipt = render_candidate(candidate, output_dir, resume=False)
        partner_conditioning = str(candidate.get("context_dependency") or "").startswith(
            "partner_"
        )
        package_records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "clip_id": candidate["clip_id"],
                "communicative_intent": candidate["communicative_intent"],
                "canonical_prompt_en": candidate["canonical_prompt_en"],
                "source_action": candidate["source_action"],
                "sample_role": (
                    "partner_offer_probe"
                    if partner_conditioning
                    else "single_actor_communicative_primitive"
                ),
                "context_dependency": candidate["context_dependency"],
                "partner_conditioning_required": partner_conditioning,
                "accepted_for_training": False,
                "manual_video_review_required": True,
                "training_exclusion_reason": "observable intent blind review is incomplete",
                "high_dynamic": False,
                "robot_contract": candidate["robot_contract"],
                "source": candidate["source"],
                "trajectory": candidate["trajectory"],
                "physical_qc": candidate["physical_qc"],
                "render": {
                    "video": f"{candidate['clip_id']}.mp4",
                    "video_sha256": receipt["video_sha256"],
                    "summary": f"../receipts/{candidate['clip_id']}.json",
                    "summary_sha256": _sha256(
                        output_dir / "receipts" / f"{candidate['clip_id']}.json"
                    ),
                    "model_source": receipt["model_source"],
                    "camera_framing": receipt["camera_framing"],
                    "representative_frame": receipt["representative_frame"],
                    "checks": receipt["checks"],
                },
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "sample_manifest.jsonl"
    manifest_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for record in package_records
        ),
        encoding="utf-8",
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "haa500_observable_intent_render_package",
        "render_contract": RENDER_CONTRACT,
        "candidate_count": len(candidates),
        "rendered_or_verified_count": len(package_records),
        "resume_reused_count": resumed,
        "accepted_for_training_count": 0,
        "semantic_review_required_count": len(package_records),
        "sample_manifest": str(manifest_path.resolve()),
        "sample_manifest_sha256": _sha256(manifest_path),
        "candidate_manifests": [str(path.resolve()) for path in candidate_manifests],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-manifest", type=Path, action="append", required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_package(
        candidate_manifests=args.candidate_manifest,
        output_dir=args.output_dir,
        resume=not args.no_resume,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
