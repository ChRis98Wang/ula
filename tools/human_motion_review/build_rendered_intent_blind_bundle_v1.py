#!/usr/bin/env python3
"""Anonymize existing rendered robot samples for observable-intent review."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.human_motion_review.build_observable_intent_review_v1 import (  # noqa: E402
    _canonical_json_bytes,
    _file_sha256,
    _read_jsonl,
    _record_sha256,
    _write_jsonl,
    build_outputs,
)
from upper_body_skeleton.robot_observable_intents import (  # noqa: E402
    DEFAULT_ONTOLOGY_PATH,
    load_observable_intent_ontology,
    ontology_sha256,
)


SCHEMA_VERSION = "1.0.0"


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _opaque_sample_id(record: Mapping[str, Any], ontology_digest: str) -> str:
    digest = hashlib.sha256()
    digest.update(ontology_digest.encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json_bytes(record))
    return f"intent_{digest.hexdigest()[:24]}"


def build_anonymous_bundle(
    *,
    input_manifest: Path,
    source_video_dir: Path,
    ontology_path: Path,
    output_dir: Path,
    private_mapping_path: Path,
) -> dict[str, Any]:
    ontology = load_observable_intent_ontology(ontology_path)
    ontology_digest = ontology_sha256(ontology_path)
    source_records = _read_jsonl(input_manifest)
    public_video_dir = output_dir / "public" / "videos"
    public_video_dir.mkdir(parents=True, exist_ok=True)
    private_mapping_path.parent.mkdir(parents=True, exist_ok=True)

    public_source_records: list[dict[str, Any]] = []
    private_mapping: list[dict[str, Any]] = []
    anonymous_ids: set[str] = set()
    for source_record in source_records:
        clip_id = source_record.get("clip_id")
        if not isinstance(clip_id, str) or not clip_id.strip():
            raise ValueError("rendered source record lacks clip_id")
        if source_record.get("accepted_for_training") is not False:
            raise ValueError(f"{clip_id}: source must still be excluded from training")
        if source_record.get("manual_video_review_required") is not True:
            raise ValueError(f"{clip_id}: source does not require manual video review")
        physical_qc = _require_mapping(source_record.get("physical_qc"), "physical_qc")
        if physical_qc.get("passed") is not True:
            raise ValueError(f"{clip_id}: physical QC did not pass")
        render = _require_mapping(source_record.get("render"), "render")
        checks = _require_mapping(render.get("checks"), "render.checks")
        required_checks = ("nonblank", "has_motion", "full_frame_uncropped")
        if any(checks.get(name) is not True for name in required_checks):
            raise ValueError(f"{clip_id}: render checks are incomplete")
        source_video_name = render.get("video")
        expected_video_digest = render.get("video_sha256")
        if not isinstance(source_video_name, str) or not source_video_name:
            raise ValueError(f"{clip_id}: render video is missing")
        if not isinstance(expected_video_digest, str) or len(expected_video_digest) != 64:
            raise ValueError(f"{clip_id}: render video SHA256 is missing")
        source_video = source_video_dir / source_video_name
        if not source_video.is_file():
            raise FileNotFoundError(f"{clip_id}: missing rendered video {source_video}")
        if _file_sha256(source_video) != expected_video_digest:
            raise ValueError(f"{clip_id}: rendered video SHA256 mismatch")

        anonymous_id = _opaque_sample_id(source_record, ontology_digest)
        if anonymous_id in anonymous_ids:
            raise ValueError(f"duplicate anonymous sample id: {anonymous_id}")
        anonymous_ids.add(anonymous_id)
        public_video = public_video_dir / f"{anonymous_id}.mp4"
        shutil.copyfile(source_video, public_video)
        if _file_sha256(public_video) != expected_video_digest:
            raise ValueError(f"{clip_id}: anonymous video copy SHA256 mismatch")

        public_source_records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_kind": "anonymous_rendered_robot_intent_source",
                "sample_id": anonymous_id,
                "clip_id": anonymous_id,
                "prompt": None,
                "prompt_text_provenance": None,
                "expression_turn_review_record": {
                    "action_semantic_review": {
                        "anonymous_video_sha256": expected_video_digest,
                    }
                },
                "physical_qc_passed": True,
                "source_label_exposed": False,
                "audio_available": False,
            }
        )
        private_mapping.append(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_kind": "private_rendered_intent_source_binding",
                "anonymous_sample_id": anonymous_id,
                "anonymous_video_sha256": expected_video_digest,
                "source_clip_id": clip_id,
                "source_record_sha256": _record_sha256(source_record),
                "source_manifest_path": str(input_manifest.resolve()),
                "source_manifest_sha256": _file_sha256(input_manifest),
                "source_video_path": str(source_video.resolve()),
                "ontology_id": ontology["ontology_id"],
                "ontology_sha256": ontology_digest,
                "reviewer_visible": False,
            }
        )

    public_source_manifest = output_dir / "public" / "anonymous_source_manifest.jsonl"
    _write_jsonl(public_source_manifest, public_source_records)
    _write_jsonl(private_mapping_path, private_mapping)
    review_summary = build_outputs(
        input_manifest=public_source_manifest,
        video_dir=public_video_dir,
        ontology_path=ontology_path,
        output_dir=output_dir / "review",
        decision_paths=[],
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "rendered_robot_observable_intent_blind_bundle",
        "source_record_count": len(source_records),
        "anonymous_video_count": len(public_source_records),
        "source_labels_exposed": False,
        "automatic_intent_labels_emitted": 0,
        "public_source_manifest": str(public_source_manifest.resolve()),
        "public_source_manifest_sha256": _file_sha256(public_source_manifest),
        "public_video_dir": str(public_video_dir.resolve()),
        "private_mapping_path": str(private_mapping_path.resolve()),
        "private_mapping_sha256": _file_sha256(private_mapping_path),
        "ontology_id": ontology["ontology_id"],
        "ontology_sha256": ontology_digest,
        "review_summary": review_summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--source-video-dir", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-mapping", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_anonymous_bundle(
        input_manifest=args.input_manifest,
        source_video_dir=args.source_video_dir,
        ontology_path=args.ontology,
        output_dir=args.output_dir,
        private_mapping_path=args.private_mapping,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
