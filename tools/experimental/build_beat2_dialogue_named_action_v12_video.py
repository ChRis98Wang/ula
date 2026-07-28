#!/usr/bin/env python3
"""Build an honest same-seed V12 action-directive comparison video."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import textwrap
from typing import Any, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.experimental import build_beat2_clean_abc_video as abc_video  # noqa: E402
from tools.experimental import build_beat2_qwen_text_interaction_60s as interaction  # noqa: E402
from upper_body_skeleton.ula_training import (  # noqa: E402
    ULA_MMDIT_V4_DUAL_TEXT_ADALN_ARCHITECTURE,
    sample_trajectory,
)
from upper_body_skeleton.ula_v2_18d_head import (  # noqa: E402
    load_contract_checkpoint,
)
from upper_body_skeleton.ula_v2_dialogue_action_episode import (  # noqa: E402
    DIALOGUE_LATENT_SLICE,
    DIRECTIVE_LATENT_SLICE,
    sha256_file,
    validate_dialogue_action_v11_episode,
)


CONFIG_KIND = "beat2_dialogue_named_action_v12_video_config_v1"
SUMMARY_KIND = "beat2_dialogue_named_action_v12_real_video_summary_v1"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/beat2_dialogue_named_action_v12_video.json"
STYLE_SLICE = slice(133, 136)
CONDITION_DIM = 264
ACTION_DIM = 18


class V12VideoError(RuntimeError):
    """Raised when the video cannot prove its inputs or outputs."""


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False
    ) as stream:
        stream.write(payload)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise V12VideoError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise V12VideoError(f"{path}:{line_number} must contain an object")
            rows.append(value)
    if not rows:
        raise V12VideoError(f"manifest is empty: {path}")
    return rows


def read_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    path = Path(path).resolve()
    config = _read_json(path)
    if config.get("schema_version") != 1 or config.get("artifact_kind") != CONFIG_KIND:
        raise V12VideoError("unexpected V12 video config contract")
    resolved = deepcopy(config)
    for field in (
        "training_summary", "training_progress", "manifest", "condition_cache",
        "output_dir",
    ):
        raw = resolved.get(field)
        if not isinstance(raw, str) or not raw:
            raise V12VideoError(f"config.{field} is required")
        resolved[f"_{field}"] = Path(raw).resolve()
    pairs = resolved.get("action_pairs")
    if (
        not isinstance(pairs, list)
        or len(pairs) < 2
        or any(not isinstance(pair, list) or len(pair) != 2 for pair in pairs)
    ):
        raise V12VideoError("action_pairs must contain at least two pairs")
    sampling = resolved.get("sampling") or {}
    if (
        float(sampling.get("fps", 0)) != 30.0
        or int(sampling.get("frames_per_segment", 0)) < 30
        or int(sampling.get("steps", 0)) <= 0
        or int(sampling.get("transition_frames", 0)) < 2
    ):
        raise V12VideoError("sampling config is invalid")
    duration = (
        len(pairs)
        * int(sampling["frames_per_segment"])
        / float(sampling["fps"])
    )
    if duration < 60.0:
        raise V12VideoError("final comparison must be at least 60 seconds")
    resolved["_config_path"] = path
    resolved["_config_sha256"] = sha256_file(path)
    resolved["_duration_sec"] = duration
    return resolved


def training_completion_state(config: Mapping[str, Any]) -> dict[str, Any]:
    summary_path = Path(config["_training_summary"])
    if not summary_path.is_file():
        return {"ready": False, "status": "waiting_for_training_summary"}
    summary = _read_json(summary_path)
    target = int(config["expected_training_target_steps"])
    completed = int(summary.get("completed_steps", -1))
    if int(summary.get("target_steps", -1)) != target:
        raise V12VideoError("training summary target step changed")
    stopped_early = summary.get("stopped_early") is True
    if completed != target and not stopped_early:
        raise V12VideoError("training summary is neither complete nor early-stopped")
    checkpoint = Path(str(summary.get("checkpoint") or "")).resolve()
    if not checkpoint.is_file():
        raise V12VideoError("selected best checkpoint is missing")
    return {
        "ready": True,
        "status": "training_complete",
        "summary": str(summary_path),
        "completed_steps": completed,
        "target_steps": target,
        "stopped_early": stopped_early,
        "best_step": int(summary["best_step"]),
        "best_validation_loss": float(summary["best_validation_loss"]),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
    }


def load_condition_sources(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    manifest = Path(config["_manifest"])
    cache = Path(config["_condition_cache"])
    if sha256_file(manifest) != config["manifest_sha256"]:
        raise V12VideoError("V12 manifest SHA changed")
    if sha256_file(cache) != config["condition_cache_sha256"]:
        raise V12VideoError("V12 condition cache SHA changed")
    rows = _read_jsonl(manifest)
    for row in rows:
        validate_dialogue_action_v11_episode(row)
    with np.load(cache, allow_pickle=False) as payload:
        clip_ids = payload["clip_ids"].astype(str).tolist()
        conditions = np.asarray(payload["conditions"], dtype=np.float32)
    if (
        clip_ids != [str(row["clip_id"]) for row in rows]
        or conditions.shape != (len(rows), CONDITION_DIM)
        or not np.isfinite(conditions).all()
    ):
        raise V12VideoError("condition cache shape/order changed")
    metadata = _read_json(cache.with_suffix(cache.suffix + ".json"))
    if (
        metadata.get("cache_sha256") != config["condition_cache_sha256"]
        or metadata.get("count") != len(rows)
        or metadata.get("mapping_head") != "absent"
    ):
        raise V12VideoError("condition cache metadata changed")
    return rows, conditions, metadata


def build_pair_conditions(
    rows: Sequence[Mapping[str, Any]],
    conditions: np.ndarray,
    *,
    action_pairs: Sequence[Sequence[str]],
    fixed_dialogue_clip_id: str,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, Any]]:
    by_id = {str(row["clip_id"]): index for index, row in enumerate(rows)}
    dialogue_index = by_id.get(str(fixed_dialogue_clip_id))
    if dialogue_index is None:
        raise V12VideoError("fixed dialogue clip is absent from the manifest")
    dialogue_row = rows[dialogue_index]
    if dialogue_row.get("fixed_split_assignment") != "test":
        raise V12VideoError("fixed dialogue must come from the held-out test split")
    action_sources: dict[str, int] = {}
    for index, row in enumerate(rows):
        action_id = str((row.get("action_summary") or {}).get("action_id") or "")
        if row.get("fixed_split_assignment") == "train" and action_id not in action_sources:
            action_sources[action_id] = index
    requested = {str(action_id) for pair in action_pairs for action_id in pair}
    missing = sorted(requested - set(action_sources))
    if missing:
        raise V12VideoError(f"action pairs lack train examples: {missing}")

    left, right, receipts = [], [], []
    dialogue_latent = conditions[dialogue_index, DIALOGUE_LATENT_SLICE]
    for pair_index, pair in enumerate(action_pairs, 1):
        pair_conditions = []
        pair_rows = []
        for action_id in pair:
            source_index = action_sources[str(action_id)]
            source = rows[source_index]
            condition = np.zeros(CONDITION_DIM, dtype=np.float32)
            condition[STYLE_SLICE] = 0.0
            condition[DIRECTIVE_LATENT_SLICE] = conditions[
                source_index, DIRECTIVE_LATENT_SLICE
            ]
            condition[DIALOGUE_LATENT_SLICE] = dialogue_latent
            if (
                not math.isclose(
                    float(np.linalg.norm(condition[DIRECTIVE_LATENT_SLICE])),
                    1.0,
                    abs_tol=1e-4,
                )
                or not math.isclose(
                    float(np.linalg.norm(condition[DIALOGUE_LATENT_SLICE])),
                    1.0,
                    abs_tol=1e-4,
                )
                or np.any(condition[: STYLE_SLICE.start])
            ):
                raise V12VideoError(f"{action_id}: constructed condition is invalid")
            pair_conditions.append(condition)
            pair_rows.append(source)
        left.append(pair_conditions[0])
        right.append(pair_conditions[1])
        receipts.append(
            {
                "index": pair_index,
                "left": dict(pair_rows[0]["action_summary"]),
                "right": dict(pair_rows[1]["action_summary"]),
                "left_source_clip_id": pair_rows[0]["clip_id"],
                "right_source_clip_id": pair_rows[1]["clip_id"],
            }
        )
    dialogue = {
        "clip_id": dialogue_row["clip_id"],
        "split": dialogue_row["fixed_split_assignment"],
        "text": dialogue_row["dialogue_text"],
        "text_sha256": dialogue_row["dialogue_text_sha256"],
    }
    return receipts, np.stack(left), np.stack(right), dialogue


def validate_final_checkpoint(
    path: str | Path, cache_metadata: Mapping[str, Any], *, device: str
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    model, checkpoint = load_contract_checkpoint(
        path, expected_action_dim=ACTION_DIM, device=device
    )
    if (
        checkpoint.get("architecture") != ULA_MMDIT_V4_DUAL_TEXT_ADALN_ARCHITECTURE
        or checkpoint.get("condition_dim") != CONDITION_DIM
        or (checkpoint.get("dual_text_conditioning_contract") or {}).get("sha256")
        != cache_metadata.get("dual_text_conditioning_contract_sha256")
    ):
        raise V12VideoError("best checkpoint is not bound to the V12 dual-text cache")
    receipt = {
        "path": str(Path(path).resolve()),
        "sha256": sha256_file(path),
        "architecture": checkpoint["architecture"],
        "posttrain_step": int(checkpoint["posttrain_step"]),
        "best_step": int(checkpoint["best_step"]),
        "best_validation_loss": float(checkpoint["best_validation_loss"]),
        "condition_dim": int(checkpoint["condition_dim"]),
        "action_dim": int(checkpoint["action_dim"]),
    }
    return model, checkpoint, receipt


def generate_action_pairs(
    model: torch.nn.Module,
    checkpoint: Mapping[str, Any],
    left_conditions: np.ndarray,
    right_conditions: np.ndarray,
    *,
    frames: int,
    steps: int,
    base_seed: int,
    device: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
    left_segments, right_segments, metrics = [], [], []
    for index, (left_condition, right_condition) in enumerate(
        zip(left_conditions, right_conditions, strict=True)
    ):
        seed = int(base_seed) + index
        left = sample_trajectory(
            model, left_condition, frames=frames, action_dim=ACTION_DIM,
            steps=steps, device=device, seed=seed,
            action_stats=checkpoint["action_stats"],
        ).astype(np.float32)
        right = sample_trajectory(
            model, right_condition, frames=frames, action_dim=ACTION_DIM,
            steps=steps, device=device, seed=seed,
            action_stats=checkpoint["action_stats"],
        ).astype(np.float32)
        left_segments.append(left)
        right_segments.append(right)
        metrics.append(
            {
                "seed": seed,
                "raw_network_delta": abc_video.trajectory_delta_metrics(left, right),
                "left_raw": abc_video.trajectory_metrics(left, fps=30.0),
                "right_raw": abc_video.trajectory_metrics(right, fps=30.0),
            }
        )
    return left_segments, right_segments, metrics


def _ass_time(seconds: float) -> str:
    centiseconds = int(round(float(seconds) * 100.0))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, centis = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _ass_escape(value: str) -> str:
    return str(value).replace("\\", "／").replace("{", "(").replace("}", ")").replace("\n", " ")


def build_ass_document(
    timeline: Sequence[Mapping[str, Any]], *, width: int, height: int,
    pane_width: int, fixed_dialogue: str
) -> str:
    if not timeline:
        raise V12VideoError("video timeline is empty")
    panel_left = pane_width * 2 + 30
    right_left = pane_width + 18
    duration = float(timeline[-1]["end_sec"])
    header = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {width}",
        f"PlayResY: {height}", "WrapStyle: 2", "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
        "Style:Left,Noto Sans CJK SC,22,&H00FFFFFF,&H00FFFFFF,&H80000000,&H80000000,-1,0,0,0,100,100,0,0,3,1,0,7,18,18,18,1",
        f"Style:Right,Noto Sans CJK SC,22,&H0000E5FF,&H0000E5FF,&H80000000,&H80000000,-1,0,0,0,100,100,0,0,3,1,0,7,{right_left},18,18,1",
        f"Style:Panel,Noto Sans CJK SC,24,&H0000E5FF,&H0000E5FF,&H000F172A,&H000F172A,-1,0,0,0,100,100,0,0,1,0,0,7,{panel_left},30,28,1",
        f"Style:Prompt,Noto Sans CJK SC,25,&H00FFFFFF,&H00FFFFFF,&H000F172A,&H000F172A,0,0,0,0,100,100,0,0,1,0,0,7,{panel_left},30,92,1",
        f"Style:Receipt,Noto Sans CJK SC,17,&H00C8D2DC,&H00C8D2DC,&H000F172A,&H000F172A,0,0,0,0,100,100,0,0,1,0,0,1,{panel_left},30,30,1",
        "", "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]
    start, end = _ass_time(0), _ass_time(duration)
    events = [
        f"Dialogue: 0,{start},{end},Left,,0,0,0,,LEFT  ACTION DESCRIPTION A",
        f"Dialogue: 0,{start},{end},Right,,0,0,0,,RIGHT  ACTION DESCRIPTION B",
        f"Dialogue: 0,{start},{end},Panel,,0,0,0,,V12 REAL GENERATED MOTION",
        (
            f"Dialogue: 0,{start},{end},Receipt,,0,0,0,,"
            "SAME CHECKPOINT · SAME SEED · SAME DIALOGUE · SAME STYLE\\N"
            "ONLY ACTION DESCRIPTION CHANGES\\N"
            "NO GT / NO REFERENCE MOTION USED FOR GENERATION\\N"
            "4 s INDEPENDENT NETWORK SAMPLES STITCHED FOR VIEWING\\N"
            "DISPLAY IS SAFETY-SMOOTHED · EXACT RAW MOTION SAVED IN NPZ"
        ),
    ]
    for item in timeline:
        left = item["left"]
        right = item["right"]
        left_en = "\\N".join(textwrap.wrap(_ass_escape(left["prompt_en"]), width=32))
        right_en = "\\N".join(textwrap.wrap(_ass_escape(right["prompt_en"]), width=32))
        dialogue = "\\N".join(textwrap.wrap(_ass_escape(fixed_dialogue), width=38))
        delta = float(item["raw_network_delta_rms_rad"])
        label = (
            f"PAIR {item['index']:02d}/{len(timeline):02d}\\N\\N"
            f"A  {_ass_escape(left['prompt_zh'])}\\N{left_en}\\N\\N"
            f"B  {_ass_escape(right['prompt_zh'])}\\N{right_en}\\N\\N"
            f"固定对话 / FIXED DIALOGUE\\N{dialogue}\\N\\N"
            f"same seed {item['seed']}\\Nraw A/B delta RMS: {delta:.6f} rad"
        )
        events.append(
            f"Dialogue: 0,{_ass_time(item['start_sec'])},{_ass_time(item['end_sec'])},Prompt,,0,0,0,,{label}"
        )
    return "\n".join(header + events) + "\n"


def prepare_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    rows, conditions, metadata = load_condition_sources(config)
    pairs, left, right, dialogue = build_pair_conditions(
        rows, conditions, action_pairs=config["action_pairs"],
        fixed_dialogue_clip_id=config["fixed_dialogue_clip_id"],
    )
    if left.shape != right.shape or left.shape != (len(pairs), CONDITION_DIM):
        raise V12VideoError("prepared action-pair condition shape changed")
    return {
        "status": "ready_to_wait",
        "pair_count": len(pairs),
        "duration_sec": config["_duration_sec"],
        "fixed_dialogue": dialogue,
        "condition_cache_sha256": metadata["cache_sha256"],
        "same_seed_within_pair": True,
        "only_action_directive_changes_within_pair": True,
        "reference_motion_generation_input": False,
    }


def build_video(config: Mapping[str, Any], *, overwrite: bool = False) -> dict[str, Any]:
    completion = training_completion_state(config)
    if completion.get("ready") is not True:
        raise V12VideoError(f"training is not complete: {completion['status']}")
    output_dir = Path(config["_output_dir"])
    summary_path = output_dir / "summary.json"
    if summary_path.is_file() and not overwrite:
        summary = _read_json(summary_path)
        if summary.get("status") != "complete":
            raise V12VideoError("existing video summary is incomplete")
        return summary
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, conditions, cache_metadata = load_condition_sources(config)
    pair_receipts, left_conditions, right_conditions, dialogue = build_pair_conditions(
        rows, conditions, action_pairs=config["action_pairs"],
        fixed_dialogue_clip_id=config["fixed_dialogue_clip_id"],
    )
    sampling = config["sampling"]
    device = str(sampling["device"])
    model, checkpoint, checkpoint_receipt = validate_final_checkpoint(
        completion["checkpoint"], cache_metadata, device=device
    )
    left_raw, right_raw, pair_metrics = generate_action_pairs(
        model, checkpoint, left_conditions, right_conditions,
        frames=int(sampling["frames_per_segment"]), steps=int(sampling["steps"]),
        base_seed=int(sampling["base_seed"]), device=device,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    transition_frames = int(sampling["transition_frames"])
    stitched_left, left_transitions = interaction.stitch_network_segments(
        left_raw, transition_frames=transition_frames
    )
    stitched_right, right_transitions = interaction.stitch_network_segments(
        right_raw, transition_frames=transition_frames
    )
    playback = config["playback"]
    playback_left = interaction.build_safety_playback(
        stitched_left, fps=30.0,
        max_velocity_rad_s=float(playback["max_velocity_rad_s"]),
        smooth_window=int(playback["smooth_window"]),
        smooth_passes=int(playback["smooth_passes"]),
    )
    playback_right = interaction.build_safety_playback(
        stitched_right, fps=30.0,
        max_velocity_rad_s=float(playback["max_velocity_rad_s"]),
        smooth_window=int(playback["smooth_window"]),
        smooth_passes=int(playback["smooth_passes"]),
    )
    frames_per_segment = int(sampling["frames_per_segment"])
    timeline = []
    for index, (receipt, metrics) in enumerate(
        zip(pair_receipts, pair_metrics, strict=True)
    ):
        start = index * frames_per_segment
        stop = start + frames_per_segment
        playback_delta = abc_video.trajectory_delta_metrics(
            playback_left[start:stop], playback_right[start:stop]
        )
        timeline.append(
            receipt
            | metrics
            | {
                "start_frame": start,
                "end_frame_exclusive": stop,
                "start_sec": start / 30.0,
                "end_sec": stop / 30.0,
                "raw_network_delta_rms_rad": metrics["raw_network_delta"]["rms"],
                "playback_delta": playback_delta,
            }
        )
    interaction._atomic_npz(
        output_dir / "exact_network_and_playback_trajectories.npz",
        raw_left=np.stack(left_raw), raw_right=np.stack(right_raw),
        stitched_left=stitched_left, stitched_right=stitched_right,
        playback_left=playback_left, playback_right=playback_right,
        left_conditions=left_conditions, right_conditions=right_conditions,
        seeds=np.asarray([item["seed"] for item in pair_metrics], dtype=np.int64),
        action_pairs=np.asarray(config["action_pairs"]),
        fps=np.asarray(30.0, dtype=np.float32),
    )
    render = config["render"]
    pane_width = int(render["pane_width"])
    panel_width = int(render["panel_width"])
    height = int(render["height"])
    left_video = output_dir / "left_action_A.mp4"
    right_video = output_dir / "right_action_B.mp4"
    left_render = abc_video._render_single(
        playback_left, csv_path=output_dir / "left_action_A.csv",
        mp4_path=left_video, fps=30.0, width=pane_width, height=height,
        simplified=bool(render["simplified"]),
    )
    right_render = abc_video._render_single(
        playback_right, csv_path=output_dir / "right_action_B.csv",
        mp4_path=right_video, fps=30.0, width=pane_width, height=height,
        simplified=bool(render["simplified"]),
    )
    ass_path = output_dir / "action_timeline.ass"
    ass_path.write_text(
        build_ass_document(
            timeline, width=pane_width * 2 + panel_width, height=height,
            pane_width=pane_width, fixed_dialogue=dialogue["text"],
        ),
        encoding="utf-8",
    )
    final_video = output_dir / str(config["video_filename"])
    final_render = interaction._run_ffmpeg_overlay(
        foundation_video=left_video, text_video=right_video,
        ass_path=ass_path, output_path=final_video, pane_width=pane_width,
        height=height, panel_width=panel_width, fps=30.0,
    )
    expected_frames = len(config["action_pairs"]) * frames_per_segment
    if abs(int(final_render["decoded_frames"]) - expected_frames) > 1:
        raise V12VideoError("final video frame count changed")
    raw_deltas = [float(item["raw_network_delta"]["rms"]) for item in pair_metrics]
    summary = {
        "schema_version": 1,
        "artifact_kind": SUMMARY_KIND,
        "status": "complete",
        "experimental_evaluation": True,
        "formal_release_eligible": False,
        "config": str(config["_config_path"]),
        "config_sha256": config["_config_sha256"],
        "training_completion": completion,
        "checkpoint": checkpoint_receipt,
        "condition_cache": {
            "path": str(config["_condition_cache"]),
            "sha256": config["condition_cache_sha256"],
            "mapping_head": "absent",
        },
        "comparison_contract": {
            "same_checkpoint": True,
            "same_seed_within_pair": True,
            "same_dialogue_within_pair": True,
            "same_zero_style_within_pair": True,
            "only_action_directive_changes_within_pair": True,
            "ground_truth_or_reference_motion_used_for_generation": False,
            "stitched_long_form": True,
            "segment_duration_sec": frames_per_segment / 30.0,
            "exact_raw_network_motion_saved": True,
            "display_motion_safety_smoothed": True,
        },
        "fixed_dialogue": dialogue,
        "timeline": timeline,
        "condition_sensitivity": {
            "raw_pair_delta_rms_rad_min": min(raw_deltas),
            "raw_pair_delta_rms_rad_mean": sum(raw_deltas) / len(raw_deltas),
            "raw_pair_delta_rms_rad_max": max(raw_deltas),
            "zero_visible_difference_is_reported_not_rejected": True,
        },
        "transitions": {"left": left_transitions, "right": right_transitions},
        "renders": {"left": left_render, "right": right_render, "final": final_render},
        "trajectory_npz": {
            "path": str(output_dir / "exact_network_and_playback_trajectories.npz"),
            "sha256": sha256_file(output_dir / "exact_network_and_playback_trajectories.npz"),
        },
    }
    _atomic_json(summary_path, summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = read_config(args.config)
        result = prepare_plan(config) if args.dry_run else build_video(
            config, overwrite=bool(args.overwrite)
        )
    except (OSError, RuntimeError, ValueError, V12VideoError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
