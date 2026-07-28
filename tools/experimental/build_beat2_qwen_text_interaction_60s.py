#!/usr/bin/env python3
"""Build a 60-second, text-labelled BEAT2 Qwen interaction diagnostic.

This is deliberately an experimental stitched long-form video, not a formal
semantic evaluation and not a claim that the generator natively models a
60-second sequence.  Each short segment is sampled from the audited generator.
The displayed playback adds a short boundary blend, clamps poses to the robot
bounds, and applies a centered low-pass display filter.  Exact raw network
segments remain in the accompanying NPZ.

The two robot panes isolate the text path:

* A: clean foundation, default style [0, 0, 0], zero text latent.
* B: paired frozen-Qwen generator, the same style/seed, direct 128D text latent.

No reference trajectory, reference-derived style, or reference duration is
used as a generation input.
"""

from __future__ import annotations

import argparse
import gc
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
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import train_beat2_qwen_motion_alignment as qwen_alignment
from tools.experimental import build_beat2_clean_abc_video as abc_video


SCHEMA_VERSION = 1
CONFIG_ARTIFACT_KIND = "beat2_qwen_text_interaction_60s_config_v1"
SUMMARY_ARTIFACT_KIND = "beat2_qwen_text_interaction_60s_v1"
EXPECTED_QWEN_CHECKPOINT_KIND = "beat2_qwen_frozen_base_alignment_v1"
FPS = 30.0
ACTION_DIM = 18
CONDITION_DIM = 264
STYLE_SLICE = slice(133, 136)
LATENT_SLICE = slice(136, 264)


class InteractionVideoError(RuntimeError):
    """Raised when the long-form diagnostic cannot prove its inputs/output."""


def _resolve_path(value: Any, *, config_dir: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise InteractionVideoError(f"{field} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def _require_file(path: Path, field: str) -> Path:
    if not path.is_file():
        raise InteractionVideoError(f"{field} does not exist: {path}")
    return path


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def build_timeline(
    prompts: Sequence[str],
    *,
    frames_per_segment: int,
    fps: float,
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    if not prompts or len(prompts) != len(seeds):
        raise InteractionVideoError("prompts and seeds must be non-empty and paired")
    if frames_per_segment < 4 or not math.isclose(fps, FPS, abs_tol=1e-9):
        raise InteractionVideoError("timeline requires at least four frames at 30 Hz")
    timeline = []
    for index, (prompt, seed) in enumerate(zip(prompts, seeds, strict=True)):
        prompt = str(prompt).strip()
        if not prompt:
            raise InteractionVideoError(f"prompt {index} is empty")
        start_frame = index * frames_per_segment
        end_frame = start_frame + frames_per_segment
        timeline.append(
            {
                "index": index + 1,
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "seed": int(seed),
                "start_frame": start_frame,
                "end_frame_exclusive": end_frame,
                "start_sec": start_frame / fps,
                "end_sec": end_frame / fps,
            }
        )
    return timeline


def stitch_network_segments(
    segments: Sequence[np.ndarray], *, transition_frames: int
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Join exact network segments with a disclosed decaying pose-offset blend."""
    if not segments:
        raise InteractionVideoError("at least one network segment is required")
    if transition_frames < 2:
        raise InteractionVideoError("transition_frames must be at least two")
    stitched: list[np.ndarray] = []
    receipts: list[dict[str, Any]] = []
    expected_shape = None
    previous_last = None
    for index, segment in enumerate(segments):
        raw = np.asarray(segment, dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != ACTION_DIM or len(raw) < transition_frames:
            raise InteractionVideoError(f"segment {index} has an invalid shape")
        if not np.isfinite(raw).all():
            raise InteractionVideoError(f"segment {index} contains non-finite values")
        if expected_shape is None:
            expected_shape = raw.shape
        elif raw.shape != expected_shape:
            raise InteractionVideoError("all interaction segments must have equal shape")
        adjusted = raw.copy()
        boundary_delta_rms = 0.0
        if previous_last is not None:
            delta = previous_last - adjusted[0]
            boundary_delta_rms = float(np.sqrt(np.mean(np.square(delta))))
            phase = np.linspace(0.0, 1.0, transition_frames, dtype=np.float32)
            weights = 0.5 * (1.0 + np.cos(np.pi * phase))
            adjusted[:transition_frames] += weights[:, None] * delta[None, :]
            adjusted[0] = previous_last
            if not np.array_equal(adjusted[0], previous_last):
                raise InteractionVideoError("boundary blend did not preserve C0 continuity")
        previous_last = adjusted[-1].copy()
        stitched.append(adjusted)
        receipts.append(
            {
                "segment_index": index + 1,
                "network_frames": len(raw),
                "transition_frames": 0 if index == 0 else transition_frames,
                "preblend_boundary_delta_rms_rad": boundary_delta_rms,
                "first_frame_matches_previous": index == 0
                or bool(np.array_equal(adjusted[0], stitched[index - 1][-1])),
            }
        )
    return np.concatenate(stitched, axis=0), receipts


def build_safety_playback(
    trajectory: np.ndarray,
    *,
    fps: float,
    max_velocity_rad_s: float,
    smooth_window: int,
    smooth_passes: int,
) -> np.ndarray:
    """Build bounded, zero-phase display motion without derivative clipping."""
    values = np.asarray(trajectory, dtype=np.float32)
    if (
        values.ndim != 2
        or values.shape[1] != ACTION_DIM
        or not np.isfinite(values).all()
    ):
        raise InteractionVideoError("safety playback input must be finite [frames,18]")
    if fps <= 0 or max_velocity_rad_s <= 0:
        raise InteractionVideoError("safety playback rates must be positive")
    smooth_window = int(smooth_window)
    smooth_passes = int(smooth_passes)
    if (
        smooth_window < 3
        or smooth_window % 2 == 0
        or smooth_window > len(values)
        or smooth_passes < 1
    ):
        raise InteractionVideoError(
            "safety playback requires an odd in-range window and positive passes"
        )
    from upper_body_skeleton.long_emotion_infer import (
        clamp_to_generation_pose_bounds,
    )

    # Clamp first.  Every later centered moving average is a convex combination,
    # so it remains inside the same pose bounds without a sharp final clip.
    filtered = np.asarray(
        clamp_to_generation_pose_bounds(values), dtype=np.float64
    )
    radius = smooth_window // 2
    for _ in range(smooth_passes):
        padded = np.pad(filtered, ((radius, radius), (0, 0)), mode="edge")
        cumulative = np.cumsum(
            np.vstack([np.zeros((1, ACTION_DIM), dtype=np.float64), padded]),
            axis=0,
        )
        filtered = (
            cumulative[smooth_window:] - cumulative[:-smooth_window]
        ) / float(smooth_window)
    result = filtered.astype(np.float32)
    maximum_velocity = float(
        np.max(np.abs(np.diff(result, axis=0))) * fps
        if len(result) > 1
        else 0.0
    )
    if maximum_velocity > max_velocity_rad_s + 1e-5:
        raise InteractionVideoError(
            "smoothed playback exceeds the configured velocity gate: "
            f"{maximum_velocity:.6f} > {max_velocity_rad_s:.6f} rad/s"
        )
    return result


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


def build_ass_document(
    timeline: Sequence[Mapping[str, Any]],
    *,
    duration_sec: float,
    width: int,
    height: int,
    pane_width: int,
) -> str:
    if not timeline or duration_sec <= 0:
        raise InteractionVideoError("ASS timeline must be non-empty and positive")
    panel_left = pane_width * 2 + 34
    b_label_left = pane_width + 20
    panel_width = width - pane_width * 2
    if panel_width < 480:
        raise InteractionVideoError("text panel is too narrow")
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
            "Style:ALabel,DejaVu Sans,22,&H00FFFFFF,&H00FFFFFF,&H80000000,"
            "&H80000000,-1,0,0,0,100,100,0,0,3,1,0,7,20,20,18,1"
        ),
        (
            f"Style:BLabel,DejaVu Sans,22,&H0000E5FF,&H0000E5FF,&H80000000,"
            f"&H80000000,-1,0,0,0,100,100,0,0,3,1,0,7,{b_label_left},20,18,1"
        ),
        (
            f"Style:PanelHeader,DejaVu Sans,24,&H0000E5FF,&H0000E5FF,"
            f"&H000F172A,&H000F172A,-1,0,0,0,100,100,0,0,1,0,0,7,"
            f"{panel_left},30,34,1"
        ),
        (
            f"Style:Prompt,DejaVu Sans,29,&H00FFFFFF,&H00FFFFFF,&H000F172A,"
            f"&H000F172A,0,0,0,0,100,100,0,0,1,0,0,7,{panel_left},34,118,1"
        ),
        (
            f"Style:Receipt,DejaVu Sans,18,&H00C8D2DC,&H00C8D2DC,&H000F172A,"
            f"&H000F172A,0,0,0,0,100,100,0,0,1,0,0,1,{panel_left},34,36,1"
        ),
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]
    start = _ass_time(0.0)
    end = _ass_time(duration_sec)
    events = [
        f"Dialogue: 0,{start},{end},ALabel,,0,0,0,,A  DEFAULT / NO TEXT LATENT",
        f"Dialogue: 0,{start},{end},BLabel,,0,0,0,,B  FROZEN QWEN TEXT LATENT",
        (
            f"Dialogue: 0,{start},{end},PanelHeader,,0,0,0,,"
            "TEXT → QWEN 128D → AdaLN"
        ),
        (
            f"Dialogue: 0,{start},{end},Receipt,,0,0,0,,"
            "NETWORK-GENERATED SHORT SEGMENTS\\N"
            "STYLE = [0, 0, 0] (NON-ORACLE DEFAULT)\\N"
            "32 FLOW STEPS · SAME SEED A/B · 30 FPS\\N"
            "0.5 s BOUNDARY BLEND + SAFETY PLAYBACK\\N"
            "DISPLAY POSTPROCESS ≠ EXACT RAW · RAW SAVED IN NPZ\\N"
            "NO REFERENCE TRAJECTORY USED FOR GENERATION\\N"
            "CANONICAL 54-GROUP DEMO · OPEN TEXT UNVALIDATED\\N"
            "Δ = PATH SENSITIVITY ONLY · CROSS-PROMPT SEEDS DIFFER"
        ),
    ]
    wrap_width = max(26, min(38, panel_width // 16))
    total = len(timeline)
    for item in timeline:
        prompt = _ass_escape(str(item["prompt"]))
        wrapped = "\\N".join(textwrap.wrap(prompt, width=wrap_width))
        segment_start = _ass_time(float(item["start_sec"]))
        segment_end = _ass_time(float(item["end_sec"]))
        label = (
            f"CURRENT TEXT COMMAND  {int(item['index']):02d}/{total:02d}\\N"
            f"[{float(item['start_sec']):05.1f}s – "
            f"{float(item['end_sec']):05.1f}s]\\N\\N"
            f"{wrapped}\\N\\N"
            f"seed {int(item['seed'])}  ·  "
            f"prompt {str(item['prompt_sha256'])[:12]}"
        )
        text_delta = item.get("text_path_delta_rms_rad")
        if isinstance(text_delta, (int, float)):
            label += f"\\Nmeasured text-path Δ A→B: {float(text_delta):.6f} rad RMS"
        events.append(
            f"Dialogue: 0,{segment_start},{segment_end},Prompt,,0,0,0,,{label}"
        )
    return "\n".join(header + events) + "\n"


def encode_prompts_with_frozen_qwen(
    prompts: Sequence[str],
    *,
    resolved_config_path: Path,
    checkpoint_path: Path,
    expected_manifest_sha256: str,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    resolved = abc_video.load_json(resolved_config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise InteractionVideoError("Qwen checkpoint is not a mapping")
    qwen_receipt = checkpoint.get("qwen")
    if (
        checkpoint.get("artifact_kind") != EXPECTED_QWEN_CHECKPOINT_KIND
        or checkpoint.get("variant") != "frozen_base"
        or checkpoint.get("no_kimodo") is not True
        or checkpoint.get("data_policy")
        != "beat2_only_no_external_motion_dataset_v1"
        or checkpoint.get("sources", {}).get("manifest_sha256")
        != expected_manifest_sha256
        or not isinstance(qwen_receipt, Mapping)
        or qwen_receipt.get("model_name") != resolved.get("model_name")
        or qwen_receipt.get("revision") != resolved.get("revision")
        or checkpoint.get("instruction") != resolved.get("instruction")
        or checkpoint.get("qwen_lora_state_dict") is not None
    ):
        raise InteractionVideoError("frozen Qwen checkpoint provenance is invalid")
    qwen, tokenizer, qwen_metadata = qwen_alignment._load_official_qwen_base(
        resolved, device=device
    )
    qwen.requires_grad_(False).eval()
    state = checkpoint.get("text_head_state_dict")
    if not isinstance(state, Mapping):
        raise InteractionVideoError("frozen Qwen text head is missing")
    label_sizes = {
        name: int(state[f"{name}_head.weight"].shape[0])
        for name in ("category", "intensity", "emotion")
    }
    head = qwen_alignment.TextAlignmentHead(
        int(checkpoint["qwen_component_dim"]),
        int(checkpoint["alignment_config"]["hidden_dim"]),
        int(checkpoint["latent_dim"]),
        label_sizes,
        dropout=float(checkpoint["alignment_config"]["dropout"]),
    ).to(device)
    head.load_state_dict(state, strict=True)
    head.requires_grad_(False).eval()
    tokens = qwen_alignment.tokenize_prompts(
        tokenizer,
        prompts,
        instruction=str(checkpoint["instruction"]),
        max_length=int(resolved["max_length"]),
        device=device,
    )
    with torch.no_grad():
        components = qwen_alignment._qwen_components(
            qwen,
            tokens,
            component_dim=int(checkpoint["qwen_component_dim"]),
        )
        latents = head(components)["embedding"].float().cpu().numpy()
    if (
        latents.shape != (len(prompts), 128)
        or not np.isfinite(latents).all()
        or not np.allclose(np.linalg.norm(latents, axis=1), 1.0, atol=1e-5)
    ):
        raise InteractionVideoError("Qwen text encoder returned invalid 128D latents")
    receipt = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": abc_video.sha256_file(checkpoint_path),
        "artifact_kind": checkpoint["artifact_kind"],
        "variant": checkpoint["variant"],
        "step": int(checkpoint["step"]),
        "instruction": checkpoint["instruction"],
        "qwen": qwen_metadata,
        "text_head_initial_state_sha256": checkpoint[
            "text_head_initial_state_sha256"
        ],
        "semantic_scope": checkpoint["semantic_scope"],
        "open_text_unvalidated": True,
        "no_kimodo": True,
    }
    del tokens, components, head, qwen, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return latents.astype(np.float32), receipt


def _run_ffmpeg_overlay(
    *,
    foundation_video: Path,
    text_video: Path,
    ass_path: Path,
    output_path: Path,
    pane_width: int,
    height: int,
    panel_width: int,
    fps: float,
) -> dict[str, Any]:
    ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    output_width = pane_width * 2 + panel_width
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
        str(foundation_video),
        "-i",
        str(text_video),
        "-filter_complex",
        (
            f"[0:v]scale={pane_width}:{height}[a];"
            f"[1:v]scale={pane_width}:{height}[b];"
            f"[a][b]hstack=inputs=2[robots];"
            f"[robots]pad={output_width}:{height}:0:0:color=0x0f172a[p];"
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
        raise InteractionVideoError(
            "ffmpeg text-panel composition failed: " + completed.stderr[-2000:]
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
        raise InteractionVideoError(
            "final MP4 full decode failed: " + decode.stderr[-2000:]
        )
    frame_count, duration_sec = imageio_ffmpeg.count_frames_and_secs(
        str(temporary)
    )
    reader = imageio_ffmpeg.read_frames(str(temporary), pix_fmt="rgb24")
    metadata = next(reader)
    reader.close()
    expected_frames = int(round(duration_sec * fps))
    if (
        tuple(metadata.get("size") or ()) != (output_width, height)
        or not math.isclose(float(metadata.get("fps", 0.0)), fps, abs_tol=1e-6)
        or abs(frame_count - expected_frames) > 1
    ):
        raise InteractionVideoError(
            f"final MP4 metadata mismatch: frames={frame_count}, meta={metadata}"
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
        "duration_sec": float(duration_sec),
        "codec": "H.264/yuv420p",
        "full_decode_passed": True,
        "ffmpeg": str(ffmpeg),
    }


def _validate_prompt_cache_binding(
    *,
    prompts: Sequence[str],
    direct_latents: np.ndarray,
    manifest_records: Mapping[str, Mapping[str, Any]],
    cache_latents: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    receipts = []
    for index, prompt in enumerate(prompts):
        candidates = sorted(
            clip_id
            for clip_id, row in manifest_records.items()
            if row.get("fixed_split_assignment") == "test"
            and str(row.get("prompt", "")).strip() == prompt
            and clip_id in cache_latents
        )
        if not candidates:
            raise InteractionVideoError(
                f"prompt is not a held-out canonical BEAT2 prompt: {prompt}"
            )
        cached = np.asarray(cache_latents[candidates[0]], dtype=np.float32)
        maximum_candidate_delta = max(
            float(
                np.max(
                    np.abs(
                        np.asarray(cache_latents[clip_id], dtype=np.float32)
                        - cached
                    )
                )
            )
            for clip_id in candidates
        )
        direct_delta = float(np.max(np.abs(direct_latents[index] - cached)))
        if maximum_candidate_delta != 0.0 or direct_delta > 1e-5:
            raise InteractionVideoError(
                f"direct Qwen latent does not reproduce the canonical cache: {prompt}"
            )
        receipts.append(
            {
                "prompt": prompt,
                "canonical_test_clip_count": len(candidates),
                "representative_clip_id": candidates[0],
                "cache_candidates_exactly_equal": True,
                "direct_vs_cache_max_abs_error": direct_delta,
                "latent_sha256": abc_video.value_sha256(
                    direct_latents[index].tolist()
                ),
                "latent_l2": float(np.linalg.norm(direct_latents[index])),
                "reference_trajectory_loaded_for_generation": False,
            }
        )
    return receipts


def canonical_prompt_order_from_cache(path: str | Path) -> list[str]:
    """Recover the exact 54-prompt batch order used to export the Qwen cache."""
    with np.load(path, allow_pickle=False) as payload:
        if "prompts" not in payload.files or "semantic_group_indices" not in payload.files:
            raise InteractionVideoError(
                "Qwen cache lacks prompts/semantic_group_indices"
            )
        prompts = payload["prompts"].astype(str)
        group_indices = payload["semantic_group_indices"].astype(np.int64)
    if (
        prompts.ndim != 1
        or group_indices.shape != prompts.shape
        or len(prompts) == 0
        or int(group_indices.min()) != 0
    ):
        raise InteractionVideoError("Qwen cache group arrays are invalid")
    group_count = int(group_indices.max()) + 1
    ordered = []
    for group_index in range(group_count):
        values = sorted(set(prompts[group_indices == group_index].tolist()))
        if len(values) != 1:
            raise InteractionVideoError(
                f"Qwen semantic group {group_index} has {len(values)} prompts"
            )
        ordered.append(values[0])
    if len(set(ordered)) != len(ordered):
        raise InteractionVideoError("Qwen canonical prompt groups are not unique")
    return ordered


def build_interaction_video(
    config_path: str | Path, *, overwrite: bool = False
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = abc_video.load_json(config_path)
    if config.get("artifact_kind") != CONFIG_ARTIFACT_KIND:
        raise InteractionVideoError(
            f"config artifact_kind must be {CONFIG_ARTIFACT_KIND}"
        )
    config_dir = config_path.parent
    output_dir = _resolve_path(
        config.get("output_dir"), config_dir=config_dir, field="output_dir"
    )
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not overwrite:
        raise InteractionVideoError(
            f"completed output already exists; pass --overwrite: {summary_path}"
        )
    paths = {
        name: _require_file(
            _resolve_path(config.get(name), config_dir=config_dir, field=name),
            name,
        )
        for name in (
            "source_manifest",
            "style_condition_cache",
            "qwen_condition_cache",
            "qwen_resolved_config",
            "qwen_checkpoint",
            "foundation_checkpoint",
            "foundation_completion_summary",
            "text_generator_checkpoint",
            "text_generator_completion_summary",
        )
    }
    manifest_records, manifest_sha256 = abc_video.load_manifest(
        paths["source_manifest"],
        expected_sha256=config.get("source_manifest_sha256"),
    )
    style_cache = abc_video.load_style_cache(
        paths["style_condition_cache"], manifest_records=manifest_records
    )
    cache_latents, cache_metadata = abc_video.load_qwen_cache(
        paths["qwen_condition_cache"],
        variant="frozen_base",
        manifest_records=manifest_records,
        manifest_sha256=manifest_sha256,
    )
    segment_config = config.get("segments")
    if not isinstance(segment_config, Mapping):
        raise InteractionVideoError("segments config is missing")
    prompts = segment_config.get("prompts")
    seeds = segment_config.get("seeds")
    duration_sec = segment_config.get("duration_sec")
    if (
        not isinstance(prompts, list)
        or not isinstance(seeds, list)
        or not isinstance(duration_sec, (int, float))
        or float(duration_sec) <= 0
    ):
        raise InteractionVideoError("segments prompts/seeds/duration are invalid")
    frames_per_segment = int(round(float(duration_sec) * FPS))
    if not math.isclose(
        frames_per_segment / FPS, float(duration_sec), abs_tol=1e-9
    ):
        raise InteractionVideoError("segment duration must map exactly to 30 Hz")
    timeline = build_timeline(
        prompts,
        frames_per_segment=frames_per_segment,
        fps=FPS,
        seeds=seeds,
    )
    expected_duration_sec = len(timeline) * frames_per_segment / FPS
    target_duration_sec = float(config.get("target_duration_sec", 60.0))
    if not math.isclose(expected_duration_sec, target_duration_sec, abs_tol=1e-9):
        raise InteractionVideoError(
            f"timeline duration is {expected_duration_sec}, expected {target_duration_sec}"
        )
    device_name = str((config.get("sampling") or {}).get("device", "cuda"))
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise InteractionVideoError("CUDA sampling was requested but is unavailable")
    canonical_prompts = canonical_prompt_order_from_cache(
        paths["qwen_condition_cache"]
    )
    canonical_latents, qwen_receipt = encode_prompts_with_frozen_qwen(
        canonical_prompts,
        resolved_config_path=paths["qwen_resolved_config"],
        checkpoint_path=paths["qwen_checkpoint"],
        expected_manifest_sha256=manifest_sha256,
        device=device,
    )
    canonical_index = {
        prompt: index for index, prompt in enumerate(canonical_prompts)
    }
    if any(prompt not in canonical_index for prompt in prompts):
        raise InteractionVideoError(
            "this reproducible video requires canonical held-out prompts"
        )
    direct_latents = np.stack(
        [canonical_latents[canonical_index[prompt]] for prompt in prompts]
    )
    qwen_receipt["encoder_batch_policy"] = (
        "exact_original_54_group_canonical_batch_order"
    )
    qwen_receipt["encoder_batch_prompt_count"] = len(canonical_prompts)
    prompt_receipts = _validate_prompt_cache_binding(
        prompts=prompts,
        direct_latents=direct_latents,
        manifest_records=manifest_records,
        cache_latents=cache_latents,
    )
    foundation, foundation_receipt = abc_video.validate_checkpoint(
        paths["foundation_checkpoint"],
        expected_manifest_sha256=manifest_sha256,
        branch="A",
        device=device_name,
    )
    foundation_receipt["completion_summary"] = abc_video.validate_completion_summary(
        paths["foundation_completion_summary"],
        checkpoint_path=paths["foundation_checkpoint"],
        branch="A",
    )
    expected_bridge_conditions = {
        clip_id: abc_video.compose_condition(
            style_cache[clip_id], cache_latents[clip_id]
        )
        for clip_id in manifest_records
    }
    text_generator, text_generator_receipt = (
        abc_video.validate_experimental_checkpoint(
            paths["text_generator_checkpoint"],
            branch="B",
            variant="frozen_base",
            expected_manifest_sha256=manifest_sha256,
            foundation_checkpoint_sha256=foundation_receipt["sha256"],
            source_128d_cache_sha256=abc_video.sha256_file(
                paths["qwen_condition_cache"]
            ),
            style_cache_sha256=abc_video.sha256_file(
                paths["style_condition_cache"]
            ),
            manifest_records=manifest_records,
            expected_conditions=expected_bridge_conditions,
            device=device_name,
        )
    )
    text_generator_receipt[
        "completion_summary"
    ] = abc_video.validate_completion_summary(
        paths["text_generator_completion_summary"],
        checkpoint_path=paths["text_generator_checkpoint"],
        branch="B",
        variant="frozen_base",
        checkpoint_sha256=text_generator_receipt["sha256"],
        foundation_checkpoint_sha256=foundation_receipt["sha256"],
        condition_cache_sha256=text_generator_receipt["condition_cache_sha256"],
        pair_contract_sha256=text_generator_receipt["pair_contract_sha256"],
    )
    zero_style = np.zeros(CONDITION_DIM, dtype=np.float32)
    abc_video.validate_zero_latent_equivalence(
        foundation,
        text_generator,
        style_conditions=[zero_style],
        device=device_name,
    )
    sampling = config.get("sampling") or {}
    steps = int(sampling.get("steps", 32))
    if steps <= 0:
        raise InteractionVideoError("sampling.steps must be positive")
    raw_a_segments = []
    raw_b_segments = []
    segment_summaries = []
    for index, item in enumerate(timeline):
        condition_a = zero_style.copy()
        condition_b = abc_video.compose_condition(
            zero_style, direct_latents[index]
        )
        if (
            np.any(condition_a)
            or np.any(condition_b[: STYLE_SLICE.start])
            or np.any(condition_b[STYLE_SLICE])
            or not np.array_equal(condition_b[LATENT_SLICE], direct_latents[index])
        ):
            raise InteractionVideoError("text-only condition layout is invalid")
        raw_a = abc_video._sample(
            foundation,
            condition_a,
            frames=frames_per_segment,
            steps=steps,
            seed=int(item["seed"]),
            device=device_name,
        )
        raw_b = abc_video._sample(
            text_generator,
            condition_b,
            frames=frames_per_segment,
            steps=steps,
            seed=int(item["seed"]),
            device=device_name,
        )
        raw_a_segments.append(raw_a.astype(np.float32))
        raw_b_segments.append(raw_b.astype(np.float32))
        raw_text_delta = abc_video.trajectory_delta_metrics(raw_b, raw_a)
        segment_summaries.append(
            {
                **item,
                **prompt_receipts[index],
                "condition_A_sha256": abc_video.value_sha256(
                    condition_a.tolist()
                ),
                "condition_B_sha256": abc_video.value_sha256(
                    condition_b.tolist()
                ),
                "condition_layout": (
                    "zero[0:136] + direct frozen-Qwen text latent[136:264]"
                ),
                "network_raw_A": abc_video.trajectory_metrics(raw_a, fps=FPS),
                "network_raw_B": abc_video.trajectory_metrics(raw_b, fps=FPS),
                "network_raw_B_minus_A": raw_text_delta,
                "text_path_delta_rms_rad": raw_text_delta["rms"],
            }
        )
    transition_frames = int(
        round(float((config.get("playback") or {}).get("transition_sec", 0.5)) * FPS)
    )
    stitched_a, transition_a = stitch_network_segments(
        raw_a_segments, transition_frames=transition_frames
    )
    stitched_b, transition_b = stitch_network_segments(
        raw_b_segments, transition_frames=transition_frames
    )
    playback_config = config.get("playback") or {}
    smooth_window = int(playback_config.get("smooth_window", 11))
    smooth_passes = int(playback_config.get("smooth_passes", 2))
    max_velocity_rad_s = float(
        playback_config.get("max_velocity_rad_s", 3.0)
    )
    playback_a = build_safety_playback(
        stitched_a,
        fps=FPS,
        max_velocity_rad_s=max_velocity_rad_s,
        smooth_window=smooth_window,
        smooth_passes=smooth_passes,
    )
    playback_b = build_safety_playback(
        stitched_b,
        fps=FPS,
        max_velocity_rad_s=max_velocity_rad_s,
        smooth_window=smooth_window,
        smooth_passes=smooth_passes,
    )
    expected_frames = int(round(target_duration_sec * FPS))
    if (
        stitched_a.shape != (expected_frames, ACTION_DIM)
        or stitched_b.shape != (expected_frames, ACTION_DIM)
        or playback_a.shape != stitched_a.shape
        or playback_b.shape != stitched_b.shape
    ):
        raise InteractionVideoError("60-second trajectory shape is invalid")
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "long_trajectories.npz"
    conditions_b = np.stack(
        [
            abc_video.compose_condition(zero_style, latent)
            for latent in direct_latents
        ]
    )
    _atomic_npz(
        trajectory_path,
        network_raw_A=np.stack(raw_a_segments),
        network_raw_B=np.stack(raw_b_segments),
        stitched_A=stitched_a,
        stitched_B=stitched_b,
        playback_A=playback_a,
        playback_B=playback_b,
        conditions_A=np.zeros((len(timeline), CONDITION_DIM), dtype=np.float32),
        conditions_B=conditions_b.astype(np.float32),
        qwen_latents=direct_latents,
        prompts=np.asarray(prompts),
        seeds=np.asarray(seeds, dtype=np.int64),
        joint_order=np.asarray(abc_video.JOINT_ORDER_18D),
        fps=np.asarray(FPS, dtype=np.float32),
    )
    render = config.get("render") or {}
    pane_width = int(render.get("pane_width", 640))
    panel_width = int(render.get("panel_width", 640))
    height = int(render.get("height", 720))
    if min(pane_width, panel_width, height) <= 0:
        raise InteractionVideoError("render dimensions must be positive")
    foundation_csv = output_dir / "A_default_no_text.csv"
    foundation_video = output_dir / "A_default_no_text.mp4"
    text_csv = output_dir / "B_frozen_qwen_text.csv"
    text_video = output_dir / "B_frozen_qwen_text.mp4"
    render_a = abc_video._render_single(
        playback_a,
        csv_path=foundation_csv,
        mp4_path=foundation_video,
        fps=FPS,
        width=pane_width,
        height=height,
        simplified=bool(render.get("simplified", False)),
    )
    render_b = abc_video._render_single(
        playback_b,
        csv_path=text_csv,
        mp4_path=text_video,
        fps=FPS,
        width=pane_width,
        height=height,
        simplified=bool(render.get("simplified", False)),
    )
    ass_path = output_dir / "text_timeline.ass"
    ass_document = build_ass_document(
        segment_summaries,
        duration_sec=target_duration_sec,
        width=pane_width * 2 + panel_width,
        height=height,
        pane_width=pane_width,
    )
    ass_path.write_text(ass_document, encoding="utf-8")
    final_video = output_dir / "beat2_qwen_text_interaction_60s.mp4"
    final_receipt = _run_ffmpeg_overlay(
        foundation_video=foundation_video,
        text_video=text_video,
        ass_path=ass_path,
        output_path=final_video,
        pane_width=pane_width,
        height=height,
        panel_width=panel_width,
        fps=FPS,
    )
    if abs(final_receipt["decoded_frames"] - expected_frames) > 1:
        raise InteractionVideoError(
            f"decoded frame count is not 60 seconds: {final_receipt}"
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": SUMMARY_ARTIFACT_KIND,
        "status": "complete",
        "experimental_only": True,
        "formal_release_eligible": False,
        "open_text_unvalidated": True,
        "stitched_long_form": True,
        "config": str(config_path),
        "config_sha256": abc_video.sha256_file(config_path),
        "source_manifest": str(paths["source_manifest"]),
        "source_manifest_sha256": manifest_sha256,
        "contracts": {
            "dataset": "BEAT2-only",
            "kimodo": "forbidden and unused",
            "reference_trajectory_generation_input": False,
            "reference_derived_style_generation_input": False,
            "reference_duration_generation_input": False,
            "style_cache_role": (
                "checkpoint lineage validation only; no cache row is used by "
                "the generated A/B conditions"
            ),
            "style": [0.0, 0.0, 0.0],
            "style_policy": "non-oracle training-mean default",
            "text": "direct pinned frozen-Qwen checkpoint encoding",
            "semantic_scope": (
                "experimental 54-group metadata alignment; not validated open text"
            ),
            "long_form": (
                "independent short network samples stitched with disclosed "
                "boundary blend and safety playback"
            ),
            "displayed_motion": (
                "network samples after disclosed stitching, pose-bound clamp, "
                "and centered zero-phase display filtering"
            ),
            "exact_network_motion": (
                "unmodified per-segment raw_A/raw_B arrays in trajectory NPZ"
            ),
            "text_discrimination_scope": (
                "within each segment A/B uses the same seed and isolates the text "
                "path; different prompts use different seeds, so cross-segment "
                "motion differences are not a prompt-only causal comparison"
            ),
            "expression": "18D upper body and 3-DoF head; no facial blendshapes",
        },
        "qwen": qwen_receipt,
        "qwen_cache": {
            "path": str(paths["qwen_condition_cache"]),
            "sha256": abc_video.sha256_file(paths["qwen_condition_cache"]),
            "artifact_kind": cache_metadata["artifact_kind"],
            "role": "canonical direct-encoder reproducibility check only",
        },
        "checkpoints": {
            "A": foundation_receipt,
            "B": text_generator_receipt,
        },
        "sampling": {
            "steps": steps,
            "fps": FPS,
            "frames_per_segment": frames_per_segment,
            "segment_count": len(timeline),
            "network_frames_total_per_arm": expected_frames,
            "same_seed_A_B": True,
            "device": device_name,
        },
        "timeline": segment_summaries,
        "transitions": {
            "policy": "cosine-decayed pose offset for C0 boundary continuity",
            "transition_frames": transition_frames,
            "transition_sec": transition_frames / FPS,
            "A": transition_a,
            "B": transition_b,
        },
        "playback_transform": {
            "pose_bounds_clamped_before_smoothing": True,
            "filter": "centered moving average (zero phase)",
            "smooth_window": smooth_window,
            "smooth_passes": smooth_passes,
            "max_velocity_rad_s_gate": max_velocity_rad_s,
            "velocity_policy": (
                "validation gate after smoothing; no derivative clipping"
            ),
            "applied_after_stitching": True,
        },
        "metrics": {
            "interpretation": {
                "network_raw_segments": (
                    "exact per-segment metrics are stored in timeline; no aggregate "
                    "raw velocity/acceleration/jerk is reported because derivatives "
                    "across independent segment boundaries are meaningless"
                ),
                "stitched_A_B": (
                    "network segments after disclosed 0.5-second boundary alignment"
                ),
                "playback_A_B": (
                    "stitched trajectories after pose-bound clamp and centered "
                    "low-pass display filtering"
                ),
            },
            "segment_network_raw_jerk_rms_mean": {
                "A": float(
                    np.mean(
                        [
                            item["network_raw_A"]["jerk_rad_s3"]["rms"]
                            for item in segment_summaries
                        ]
                    )
                ),
                "B": float(
                    np.mean(
                        [
                            item["network_raw_B"]["jerk_rad_s3"]["rms"]
                            for item in segment_summaries
                        ]
                    )
                ),
            },
            "stitched_A": abc_video.trajectory_metrics(stitched_a, fps=FPS),
            "stitched_B": abc_video.trajectory_metrics(stitched_b, fps=FPS),
            "playback_A": abc_video.trajectory_metrics(playback_a, fps=FPS),
            "playback_B": abc_video.trajectory_metrics(playback_b, fps=FPS),
            "network_raw_B_minus_A": abc_video.trajectory_delta_metrics(
                np.concatenate(raw_b_segments), np.concatenate(raw_a_segments)
            ),
            "playback_B_minus_A": abc_video.trajectory_delta_metrics(
                playback_b, playback_a
            ),
        },
        "artifacts": {
            "trajectory_npz": {
                "path": str(trajectory_path),
                "sha256": abc_video.sha256_file(trajectory_path),
                "bytes": trajectory_path.stat().st_size,
            },
            "A_csv": {
                "path": str(foundation_csv),
                "sha256": abc_video.sha256_file(foundation_csv),
            },
            "B_csv": {
                "path": str(text_csv),
                "sha256": abc_video.sha256_file(text_csv),
            },
            "A_robot_video": {
                "path": str(foundation_video),
                "sha256": abc_video.sha256_file(foundation_video),
                "render": render_a,
            },
            "B_robot_video": {
                "path": str(text_video),
                "sha256": abc_video.sha256_file(text_video),
                "render": render_b,
            },
            "text_timeline_ass": {
                "path": str(ass_path),
                "sha256": abc_video.sha256_file(ass_path),
            },
            "final_video": final_receipt,
        },
    }
    abc_video.atomic_json(summary_path, summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = build_interaction_video(args.config, overwrite=args.overwrite)
    except (
        InteractionVideoError,
        abc_video.EvaluationContractError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": summary["status"],
                "artifact_kind": summary["artifact_kind"],
                "video": summary["artifacts"]["final_video"]["path"],
                "duration_sec": summary["artifacts"]["final_video"]["duration_sec"],
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
