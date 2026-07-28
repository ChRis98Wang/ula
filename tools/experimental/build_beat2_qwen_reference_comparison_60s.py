#!/usr/bin/env python3
"""Compare sealed Qwen-generated motion with held-out BEAT2 18D examples.

This tool never generates A/B motion.  It first validates and seals the
previously generated B artifact, then loads one pre-declared held-out BEAT2
18D retargeted example per prompt.  Reference actions therefore cannot affect
the generator seed, duration, condition, checkpoint, or output.

Each published safe-CSV reference is played at 30 Hz from the start of its
six-second slot.  The final pose is held for the remainder.  This comparison
adds no loop, temporal crop, time warp, interpolation, or smoothing.  Upstream
GMR retargeting may itself contain explicitly receipted retiming.  The pane is
not raw human SMPL-X: it is the exact physical-QC-passed 18D representation
published by the same data pipeline and held out from generator training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from typing import Any, Mapping, Sequence

import imageio_ffmpeg
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.experimental import build_beat2_clean_abc_video as abc_video
from tools.experimental import build_beat2_qwen_text_interaction_60s as long_video
from upper_body_skeleton.mujoco_playback import render_trajectory_comparison
from upper_body_skeleton.ula_v2_18d_head import read_joint_csv


SCHEMA_VERSION = 1
CONFIG_ARTIFACT_KIND = "beat2_qwen_reference_comparison_60s_config_v1"
SUMMARY_ARTIFACT_KIND = "beat2_qwen_reference_comparison_60s_v1"
INTERACTION_ARTIFACT_KIND = "beat2_qwen_text_interaction_60s_v1"
FPS = 30.0
ACTION_DIM = 18
FRAMES_PER_SLOT = 180
SLOT_COUNT = 10


class ReferenceComparisonError(RuntimeError):
    """Raised when the comparison cannot prove its provenance or output."""


def _resolve_path(value: Any, *, config_dir: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ReferenceComparisonError(f"{field} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def _require_file(path: Path, field: str) -> Path:
    if not path.is_file():
        raise ReferenceComparisonError(f"{field} does not exist: {path}")
    return path


def _require_sha(value: Any, field: str) -> str:
    value = str(value)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ReferenceComparisonError(f"{field} must be a lowercase SHA256")
    return value


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def pad_reference_with_endpoint_hold(
    trajectory: np.ndarray, *, slot_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    """Preserve published safe-CSV samples and hold the final pose."""
    values = np.asarray(trajectory, dtype=np.float32)
    if (
        values.ndim != 2
        or values.shape[1] != ACTION_DIM
        or len(values) < 2
        or len(values) > slot_frames
        or not np.isfinite(values).all()
    ):
        raise ReferenceComparisonError(
            "reference must be finite [2..slot_frames,18]"
        )
    padded = np.repeat(values[-1:], slot_frames, axis=0)
    padded[: len(values)] = values
    valid = np.zeros(slot_frames, dtype=np.bool_)
    valid[: len(values)] = True
    if not np.array_equal(padded[: len(values)], values):
        raise ReferenceComparisonError("published reference values were modified")
    return padded, valid


def select_sealed_reference(
    *,
    prompt: str,
    representative_clip_id: str,
    expected_candidate_count: int,
    manifest_records: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any], list[str]]:
    """Reproduce the already-sealed lexicographic test-reference receipt."""
    candidates = sorted(
        clip_id
        for clip_id, record in manifest_records.items()
        if record.get("fixed_split_assignment") == "test"
        and str(record.get("prompt", "")).strip() == prompt
    )
    if len(candidates) != int(expected_candidate_count):
        raise ReferenceComparisonError(
            f"held-out candidate count changed for prompt: {prompt}"
        )
    if not candidates or candidates[0] != representative_clip_id:
        raise ReferenceComparisonError(
            f"sealed representative is not the lexicographic first candidate: {prompt}"
        )
    record = manifest_records[representative_clip_id]
    motion = record.get("motion_18d")
    quality = motion.get("quality_gate") if isinstance(motion, Mapping) else None
    if (
        record.get("dataset") != "BEAT2"
        or record.get("accepted_for_training") is not True
        or (record.get("adjudication") or {}).get("status")
        != "motion_only_train_ready"
        or not isinstance(motion, Mapping)
        or motion.get("state") != "passed"
        or motion.get("action_dim") != ACTION_DIM
        or motion.get("fps") != int(FPS)
        or not isinstance(quality, Mapping)
        or quality.get("passed") is not True
        or record.get("training_admission_status")
        != "motion_only_physical_qc_train_ready"
    ):
        raise ReferenceComparisonError(
            f"sealed reference is not a BEAT2 physical-QC test record: "
            f"{representative_clip_id}"
        )
    return record, candidates


def _ass_time(seconds: float) -> str:
    centiseconds = int(round(float(seconds) * 100.0))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, centis = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _ass_escape(value: str) -> str:
    return (
        str(value)
        .replace("\\", "／")
        .replace("{", "(")
        .replace("}", ")")
        .replace("\n", " ")
    )


def build_reference_ass_document(
    timeline: Sequence[Mapping[str, Any]],
    *,
    duration_sec: float,
    width: int,
    height: int,
    robot_width: int,
) -> str:
    if not timeline or duration_sec <= 0:
        raise ReferenceComparisonError("ASS timeline must be non-empty")
    panel_left = robot_width + 34
    panel_width = width - robot_width
    if panel_width < 480:
        raise ReferenceComparisonError("reference text panel is too narrow")
    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        (
            "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,"
            "OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,"
            "ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,"
            "MarginR,MarginV,Encoding"
        ),
        (
            f"Style:PanelHeader,DejaVu Sans,23,&H0050CD89,&H0050CD89,"
            f"&H000F172A,&H000F172A,-1,0,0,0,100,100,0,0,1,0,0,7,"
            f"{panel_left},30,34,1"
        ),
        (
            f"Style:Prompt,DejaVu Sans,25,&H00FFFFFF,&H00FFFFFF,&H000F172A,"
            f"&H000F172A,0,0,0,0,100,100,0,0,1,0,0,7,"
            f"{panel_left},34,100,1"
        ),
        (
            f"Style:Receipt,DejaVu Sans,17,&H00C8D2DC,&H00C8D2DC,&H000F172A,"
            f"&H000F172A,0,0,0,0,100,100,0,0,1,0,0,1,"
            f"{panel_left},34,34,1"
        ),
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]
    start = _ass_time(0.0)
    end = _ass_time(duration_sec)
    events = [
        (
            f"Dialogue: 0,{start},{end},PanelHeader,,0,0,0,,"
            "GENERATED B  ↔  PROJECT FIXED-TEST BEAT2 18D"
        ),
        (
            f"Dialogue: 0,{start},{end},Receipt,,0,0,0,,"
            "R = BEAT2 → GMR RETARGET → QC-PASSED 18D\\N"
            "R IS NOT RAW HUMAN SMPL-X OR PAIRED GROUND TRUTH\\N"
            "REFERENCE LOADED AFTER SEALED A/B · NEVER GENERATION INPUT\\N"
            "PUBLISHED SAFE-CSV 30 FPS · ENDPOINT HOLD ONLY\\N"
            "THIS COMPARISON ADDS NO LOOP / TIME-WARP / CROP / SMOOTHING\\N"
            "PUBLISHED 18D MAY INCLUDE DISCLOSED UPSTREAM GMR RETIMING\\N"
            "PRE-SEALED LEXICOGRAPHIC FIXED-TEST EXAMPLE · NO VISUAL PICKING\\N"
            "METADATA-MATCHED QUALITATIVE VIEW · NO FRAMEWISE SCORE\\N"
            "BEAT2 ONLY · KIMODO FORBIDDEN AND UNUSED"
        ),
    ]
    wrap_width = max(28, min(40, panel_width // 15))
    total = len(timeline)
    for item in timeline:
        prompt = _ass_escape(str(item["prompt"]))
        wrapped = "\\N".join(textwrap.wrap(prompt, width=wrap_width))
        segment_start = _ass_time(float(item["start_sec"]))
        segment_end = _ass_time(float(item["end_sec"]))
        source_clip = _ass_escape(str(item["reference_source_clip_id"]))
        label = (
            f"TEXT GROUP  {int(item['index']):02d}/{total:02d}\\N"
            f"[{float(item['start_sec']):05.1f}s – "
            f"{float(item['end_sec']):05.1f}s]\\N\\N"
            f"{wrapped}\\N\\N"
            f"R source: {source_clip}\\N"
            f"published {int(item['reference_published_frames'])} frames / "
            f"{float(item['reference_published_coverage_sec']):.2f}s"
            f"  · hold {int(item['reference_hold_frames'])} frames\\N"
            f"upstream GMR retimed: "
            f"{str(bool(item['upstream_retarget_retimed'])).lower()}"
            f"  · source {int(item['upstream_source_frame_count'])} frames\\N"
            f"fixed test candidates {int(item['reference_candidate_count'])}"
            f"  · csv {str(item['reference_csv_sha256'])[:12]}"
        )
        events.append(
            f"Dialogue: 0,{segment_start},{segment_end},Prompt,,0,0,0,,{label}"
        )
    return "\n".join(header + events) + "\n"


def _compose_final_video(
    *,
    comparison_video: Path,
    ass_path: Path,
    output_path: Path,
    robot_width: int,
    panel_width: int,
    height: int,
    fps: float,
) -> dict[str, Any]:
    ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    output_width = robot_width + panel_width
    escaped_ass = (
        str(ass_path.resolve())
        .replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
    )
    temporary = output_path.with_name(f".{output_path.stem}.{os.getpid()}.tmp.mp4")
    command = [
        str(ffmpeg),
        "-y",
        "-i",
        str(comparison_video),
        "-filter_complex",
        (
            f"[0:v]scale={robot_width}:{height}[r];"
            f"[r]pad={output_width}:{height}:0:0:color=0x0f172a[p];"
            f"[p]ass='{escaped_ass}'[out]"
        ),
        "-map",
        "[out]",
        "-an",
        "-r",
        f"{fps:g}",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise ReferenceComparisonError(
            "ffmpeg reference composition failed: " + completed.stderr[-2000:]
        )
    decode = subprocess.run(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-i",
            str(temporary),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    if decode.returncode != 0:
        raise ReferenceComparisonError(
            "reference MP4 full decode failed: " + decode.stderr[-2000:]
        )
    frame_count, duration = imageio_ffmpeg.count_frames_and_secs(str(temporary))
    reader = imageio_ffmpeg.read_frames(str(temporary), pix_fmt="rgb24")
    metadata = next(reader)
    reader.close()
    expected_frames = int(round(duration * fps))
    if (
        tuple(metadata.get("size") or ()) != (output_width, height)
        or not math.isclose(float(metadata.get("fps", 0.0)), fps, abs_tol=1e-6)
        or abs(frame_count - expected_frames) > 1
    ):
        raise ReferenceComparisonError(
            f"final reference MP4 metadata mismatch: {metadata}"
        )
    os.replace(temporary, output_path)
    return {
        "path": str(output_path),
        "sha256": abc_video.sha256_file(output_path),
        "bytes": output_path.stat().st_size,
        "width": output_width,
        "height": height,
        "fps": float(metadata["fps"]),
        "decoded_frames": int(frame_count),
        "duration_sec": float(duration),
        "codec": "H.264/yuv420p",
        "full_decode_passed": True,
        "ffmpeg": str(ffmpeg),
    }


def build_reference_comparison(
    config_path: str | Path, *, overwrite: bool = False
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = abc_video.load_json(config_path)
    if (
        config.get("schema_version") != SCHEMA_VERSION
        or config.get("artifact_kind") != CONFIG_ARTIFACT_KIND
    ):
        raise ReferenceComparisonError("reference comparison config contract mismatch")
    config_dir = config_path.parent
    output_dir = _resolve_path(
        config.get("output_dir"), config_dir=config_dir, field="output_dir"
    )
    final_video = output_dir / "B_vs_heldout_BEAT2_18d_reference_60s.mp4"
    summary_path = output_dir / "summary.json"
    if not overwrite and (final_video.exists() or summary_path.exists()):
        raise ReferenceComparisonError(
            f"output exists; pass --overwrite to rebuild: {output_dir}"
        )

    # Stage 1: validate and seal the pre-existing generated artifact before any
    # reference action values are loaded.
    interaction_summary_path = _resolve_path(
        config.get("sealed_interaction_summary"),
        config_dir=config_dir,
        field="sealed_interaction_summary",
    )
    _require_file(interaction_summary_path, "sealed interaction summary")
    expected_summary_sha = _require_sha(
        config.get("sealed_interaction_summary_sha256"),
        "sealed_interaction_summary_sha256",
    )
    actual_summary_sha = abc_video.sha256_file(interaction_summary_path)
    if actual_summary_sha != expected_summary_sha:
        raise ReferenceComparisonError(
            "sealed interaction summary SHA changed before reference loading"
        )
    interaction = abc_video.load_json(interaction_summary_path)
    if (
        interaction.get("artifact_kind") != INTERACTION_ARTIFACT_KIND
        or interaction.get("status") != "complete"
        or (interaction.get("contracts") or {}).get("dataset") != "BEAT2-only"
        or (interaction.get("contracts") or {}).get("kimodo")
        != "forbidden and unused"
        or interaction.get("open_text_unvalidated") is not True
        or interaction.get("stitched_long_form") is not True
    ):
        raise ReferenceComparisonError("sealed interaction summary is ineligible")
    interaction_artifacts = interaction.get("artifacts")
    if not isinstance(interaction_artifacts, Mapping):
        raise ReferenceComparisonError("sealed interaction artifacts are missing")
    source_npz_receipt = interaction_artifacts.get("trajectory_npz")
    if not isinstance(source_npz_receipt, Mapping):
        raise ReferenceComparisonError("sealed trajectory receipt is missing")
    source_npz = _require_file(
        Path(str(source_npz_receipt.get("path"))).resolve(),
        "sealed interaction NPZ",
    )
    source_npz_sha = abc_video.sha256_file(source_npz)
    if source_npz_sha != source_npz_receipt.get("sha256"):
        raise ReferenceComparisonError("sealed interaction NPZ SHA mismatch")
    with np.load(source_npz, allow_pickle=False) as payload:
        generated_b = np.asarray(payload["playback_B"], dtype=np.float32)
        prompts = payload["prompts"].astype(str).tolist()
    if (
        generated_b.shape != (SLOT_COUNT * FRAMES_PER_SLOT, ACTION_DIM)
        or len(prompts) != SLOT_COUNT
        or not np.isfinite(generated_b).all()
    ):
        raise ReferenceComparisonError("sealed B trajectory contract mismatch")
    sealed_b_value_sha = abc_video.value_sha256(generated_b.tolist())

    timeline = interaction.get("timeline")
    if not isinstance(timeline, list) or len(timeline) != SLOT_COUNT:
        raise ReferenceComparisonError("sealed timeline contract mismatch")
    if [str(item.get("prompt")) for item in timeline] != prompts:
        raise ReferenceComparisonError("sealed prompts differ from trajectory NPZ")
    if any(
        item.get("reference_trajectory_loaded_for_generation") is not False
        for item in timeline
    ):
        raise ReferenceComparisonError(
            "sealed generation lacks a no-reference-action receipt"
        )

    # Stage 2: only after B is sealed in memory and by SHA do we open the
    # BEAT2 manifest and reference action CSVs.
    manifest_path = _require_file(
        Path(str(interaction.get("source_manifest"))).resolve(),
        "BEAT2 source manifest",
    )
    manifest_sha = _require_sha(
        interaction.get("source_manifest_sha256"), "source_manifest_sha256"
    )
    manifest_records, actual_manifest_sha = abc_video.load_manifest(
        manifest_path, expected_sha256=manifest_sha
    )
    if actual_manifest_sha != manifest_sha:
        raise ReferenceComparisonError("BEAT2 manifest SHA mismatch")

    reference_segments: list[np.ndarray] = []
    reference_published: list[np.ndarray] = []
    reference_valid_masks: list[np.ndarray] = []
    reference_timeline: list[dict[str, Any]] = []
    for index, (prompt, item) in enumerate(zip(prompts, timeline, strict=True)):
        representative = str(item.get("representative_clip_id"))
        candidate_count = int(item.get("canonical_test_clip_count", 0))
        record, candidates = select_sealed_reference(
            prompt=prompt,
            representative_clip_id=representative,
            expected_candidate_count=candidate_count,
            manifest_records=manifest_records,
        )
        motion = record["motion_18d"]
        csv_path = _require_file(
            Path(str(motion.get("safe_csv"))).resolve(),
            f"reference safe CSV {representative}",
        )
        csv_sha = abc_video.sha256_file(csv_path)
        if csv_sha != motion.get("safe_csv_sha256"):
            raise ReferenceComparisonError(
                f"reference CSV SHA mismatch: {representative}"
            )
        values = read_joint_csv(csv_path)
        frames = int(record.get("frames"))
        if (
            values.shape != (frames, ACTION_DIM)
            or motion.get("frames") != frames
            or motion.get("csv_rows") != frames
            or frames > FRAMES_PER_SLOT
        ):
            raise ReferenceComparisonError(
                f"reference row contract mismatch: {representative}"
            )
        padded, valid = pad_reference_with_endpoint_hold(
            values, slot_frames=FRAMES_PER_SLOT
        )
        reference_published.append(values.copy())
        reference_segments.append(padded)
        reference_valid_masks.append(valid)
        retarget_segment = motion.get("retarget_segment")
        if not isinstance(retarget_segment, Mapping):
            raise ReferenceComparisonError(
                f"reference retarget receipt is missing: {representative}"
            )
        if int(retarget_segment.get("output_frame_count", -1)) != frames:
            raise ReferenceComparisonError(
                f"reference retarget output count mismatch: {representative}"
            )
        source_frame_count = int(
            retarget_segment.get("source_frame_count", -1)
        )
        if source_frame_count < 2:
            raise ReferenceComparisonError(
                f"reference source frame count is invalid: {representative}"
            )
        reference_timeline.append(
            {
                "index": index + 1,
                "prompt": prompt,
                "start_sec": index * FRAMES_PER_SLOT / FPS,
                "end_sec": (index + 1) * FRAMES_PER_SLOT / FPS,
                "reference_clip_id": representative,
                "reference_source_clip_id": record.get("source_clip_id"),
                "reference_speaker_key": record.get("speaker_key"),
                "reference_fixed_split": record.get("fixed_split_assignment"),
                "reference_official_split": record.get("official_split"),
                "reference_candidate_count": len(candidates),
                "reference_published_frames": frames,
                "reference_published_coverage_sec": frames / FPS,
                "reference_published_sample_span_sec": (frames - 1) / FPS,
                "upstream_retarget_retimed": bool(
                    retarget_segment.get("retimed")
                ),
                "upstream_source_frame_count": source_frame_count,
                "upstream_published_output_frame_count": frames,
                "reference_hold_frames": FRAMES_PER_SLOT - frames,
                "reference_safe_csv": str(csv_path),
                "reference_csv_sha256": csv_sha,
                "reference_quality_json": motion.get("quality_json"),
                "reference_quality_sha256": motion.get("quality_sha256"),
                "reference_upstream_selected_record_sha256": record.get(
                    "selected_record_sha256"
                ),
                "reference_upstream_selected_record_sha256_role": record.get(
                    "selected_record_sha256_role"
                ),
                "reference_loaded_record_value_sha256": abc_video.value_sha256(
                    record
                ),
                "reference_metrics_published_safe_csv": abc_video.trajectory_metrics(
                    values, fps=FPS
                ),
                "reference_value_preservation_max_abs_error": float(
                    np.max(np.abs(padded[:frames] - values))
                ),
            }
        )

    padded_reference = np.stack(reference_segments)
    valid_mask = np.stack(reference_valid_masks)
    reference_playback = padded_reference.reshape(
        SLOT_COUNT * FRAMES_PER_SLOT, ACTION_DIM
    )
    if not np.array_equal(
        reference_playback.reshape(SLOT_COUNT, FRAMES_PER_SLOT, ACTION_DIM)[
            valid_mask
        ],
        np.concatenate(reference_published),
    ):
        raise ReferenceComparisonError("reference valid-mask preservation failed")
    published_offsets = np.zeros(SLOT_COUNT + 1, dtype=np.int64)
    published_offsets[1:] = np.cumsum(
        [len(values) for values in reference_published]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "comparison_trajectories.npz"
    _atomic_npz(
        trajectory_path,
        generated_B_playback=generated_b,
        reference_published_safe_csv_concat=np.concatenate(reference_published),
        reference_published_offsets=published_offsets,
        reference_padded_segments=padded_reference,
        reference_playback=reference_playback,
        reference_valid_mask=valid_mask,
        prompts=np.asarray(prompts),
        reference_clip_ids=np.asarray(
            [item["reference_clip_id"] for item in reference_timeline]
        ),
        joint_order=np.asarray(abc_video.JOINT_ORDER_18D),
        fps=np.asarray(FPS, dtype=np.float32),
    )

    render = config.get("render") or {}
    pane_width = int(render.get("pane_width", 640))
    pane_height = int(render.get("pane_height", 680))
    title_height = int(render.get("title_height", 40))
    panel_width = int(render.get("panel_width", 640))
    height = pane_height + title_height
    if min(pane_width, pane_height, title_height, panel_width) <= 0:
        raise ReferenceComparisonError("render dimensions must be positive")
    comparison_video = output_dir / "B_vs_reference_fixed_camera.mp4"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    render_receipt = render_trajectory_comparison(
        generated_b,
        reference_playback,
        comparison_video,
        fps=FPS,
        pane_width=pane_width,
        pane_height=pane_height,
        title_height=title_height,
        simplified=bool(render.get("simplified", False)),
        camera_view=str(render.get("camera_view", "upper")),
        network_label="B QWEN GENERATED (DISPLAY)",
        reference_label="R FIXED-TEST BEAT2 18D (PUBLISHED + HOLD)",
    )
    ass_path = output_dir / "reference_timeline.ass"
    ass_document = build_reference_ass_document(
        reference_timeline,
        duration_sec=SLOT_COUNT * FRAMES_PER_SLOT / FPS,
        width=pane_width * 2 + panel_width,
        height=height,
        robot_width=pane_width * 2,
    )
    ass_path.write_text(ass_document, encoding="utf-8")
    final_receipt = _compose_final_video(
        comparison_video=comparison_video,
        ass_path=ass_path,
        output_path=final_video,
        robot_width=pane_width * 2,
        panel_width=panel_width,
        height=height,
        fps=FPS,
    )
    expected_frames = SLOT_COUNT * FRAMES_PER_SLOT
    if (
        final_receipt["decoded_frames"] != expected_frames
        or not math.isclose(
            float(final_receipt["duration_sec"]),
            expected_frames / FPS,
            abs_tol=1e-6,
        )
    ):
        raise ReferenceComparisonError("final reference video is not exact 60s")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": SUMMARY_ARTIFACT_KIND,
        "status": "complete",
        "experimental_only": True,
        "formal_release_eligible": False,
        "config": str(config_path),
        "config_sha256": abc_video.sha256_file(config_path),
        "contracts": {
            "dataset": "BEAT2-only",
            "kimodo": "forbidden and unused",
            "reference_kind": (
                "project-fixed-test BEAT2-derived GMR-retargeted "
                "physical-QC-passed 18D robot motion"
            ),
            "fixed_test_scope": (
                "project speaker-held-out split; not the upstream official "
                "BEAT2 split field"
            ),
            "reference_is_raw_human_smplx": False,
            "reference_is_paired_ground_truth": False,
            "reference_role": "metadata-matched held-out qualitative example",
            "selection_policy": config.get("selection_policy"),
            "selection_uses_generated_motion": False,
            "reference_slot_policy": config.get("reference_slot_policy"),
            "comparison_loop_applied_to_reference": False,
            "comparison_additional_time_warp_applied_to_reference": False,
            "comparison_temporal_crop_applied_to_reference": False,
            "comparison_smoothing_applied_to_reference": False,
            "upstream_retarget_may_include_disclosed_retiming": True,
            "reference_actions_loaded_after_sealed_B": True,
            "reference_trajectory_generation_input": False,
            "reference_duration_generation_input": False,
            "reference_style_generation_input": False,
            "framewise_generated_reference_metric_valid": False,
            "semantic_scope": (
                "coarse canonical metadata prompt; open-text and robot affect "
                "semantics remain unvalidated"
            ),
        },
        "sealed_generation": {
            "summary": str(interaction_summary_path),
            "summary_sha256": actual_summary_sha,
            "trajectory_npz": str(source_npz),
            "trajectory_npz_sha256": source_npz_sha,
            "B_playback_value_sha256": sealed_b_value_sha,
            "loaded_before_reference_actions": True,
            "generation_rerun_by_this_tool": False,
        },
        "source_manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha,
            "record_count": len(manifest_records),
        },
        "sampling": {
            "fps": FPS,
            "slots": SLOT_COUNT,
            "frames_per_slot": FRAMES_PER_SLOT,
            "frames_total": expected_frames,
            "duration_sec": expected_frames / FPS,
        },
        "timeline": reference_timeline,
        "metrics": {
            "interpretation": {
                "generated_B": (
                    "sealed display playback from the prior A/B diagnostic"
                ),
                "reference": (
                    "published safe-CSV per-clip metrics are stored in timeline; "
                    "no global reference derivatives are reported across "
                    "independent clip boundaries or endpoint holds"
                ),
                "comparison": (
                    "qualitative metadata-group comparison only; trajectories "
                    "are not phase-aligned or paired"
                ),
            },
            "generated_B_playback": abc_video.trajectory_metrics(
                generated_b, fps=FPS
            ),
        },
        "render": {
            **render_receipt,
            "comparison_video_sha256": abc_video.sha256_file(comparison_video),
            "same_fixed_camera_B_R": True,
            "reference_video_trajectory_transform": (
                "published safe-CSV values followed by exact endpoint hold only; "
                "no additional temporal transform"
            ),
        },
        "artifacts": {
            "trajectory_npz": {
                "path": str(trajectory_path),
                "sha256": abc_video.sha256_file(trajectory_path),
                "bytes": trajectory_path.stat().st_size,
            },
            "comparison_video": {
                "path": str(comparison_video),
                "sha256": abc_video.sha256_file(comparison_video),
                "bytes": comparison_video.stat().st_size,
            },
            "text_timeline_ass": {
                "path": str(ass_path),
                "sha256": abc_video.sha256_file(ass_path),
                "bytes": ass_path.stat().st_size,
            },
            "final_video": final_receipt,
        },
    }
    _atomic_json(summary_path, summary)
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = build_reference_comparison(
        args.config, overwrite=bool(args.overwrite)
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "artifact_kind": summary["artifact_kind"],
                "video": summary["artifacts"]["final_video"]["path"],
                "duration_sec": summary["artifacts"]["final_video"][
                    "duration_sec"
                ],
                "decoded_frames": summary["artifacts"]["final_video"][
                    "decoded_frames"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
