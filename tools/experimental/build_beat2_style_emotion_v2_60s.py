#!/usr/bin/env python3
"""Build a 60-second BEAT2 V2 text-style/emotion qualitative comparison.

Each row compares a neutral prompt with an emotion prompt using the same
initial flow noise.  The generator receives only a frozen-Qwen 128D text
latent plus the three style controls predicted by the checkpoint's
``QwenStyleHead``.  No trajectory style value is accepted at inference.

The planner determines each network sample's native duration.  A fixed
montage slot is filled by repeating the generated endpoint after that native
sample ends.  There is deliberately no smoothing, time warp, pose blend,
boundary blend, or forced replacement of a generated last frame.
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
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.experimental import build_beat2_clean_abc_video as abc_video
from upper_body_skeleton.beat2_condition_control import (
    CONDITION_DIM,
    QwenStyleHead,
    STYLE_CONTROL_SLICE,
    TEXT_LATENT_DIM,
    TEXT_LATENT_SLICE,
    assemble_text_style_conditions,
)
from upper_body_skeleton.retarget_v2_18d import JOINT_ORDER_18D
from upper_body_skeleton.ula_training import (
    ULA_MMDIT_V3_ADALN_ARCHITECTURE,
    create_ula_model,
    denormalize_action_tensor,
    joint_limit_tensors,
    normalize_action_tensor,
)


SCHEMA_VERSION = 2
CONFIG_ARTIFACT_KIND = "beat2_style_emotion_qualitative_60s_config_v2"
SUMMARY_ARTIFACT_KIND = "beat2_style_emotion_qualitative_60s_v2"
CHECKPOINT_ARTIFACT_KIND = "beat2_text_style_emotion_generator_v2"
CHECKPOINT_SUMMARY_ARTIFACT_KIND = (
    "beat2_text_style_emotion_training_summary_v2"
)
CONDITION_POLICY = (
    "zero_base_0_133_qwen_predicted_style_133_136_"
    "frozen_qwen_text_136_264_no_trajectory_oracle_v2"
)
EXPECTED_TRAINING_POLICY = (
    "adaln_condition_path_plus_qwen_style_head_per_episode_rank_"
    "balanced_group_gate_global54_semantic_perceptual_v6"
)
DATA_POLICY = "beat2_only_no_external_motion_dataset_v1"
FORBIDDEN_EXTERNAL_TOKEN = "kimodo"
ACTION_DIM = 18
FPS = 30.0
ALLOWED_GUIDANCE_SCALES = (1.0, 1.5)
EXPECTED_HIDDEN_DIM = 384
EXPECTED_LAYERS = 6
EXPECTED_SEMANTIC_TOKENS = 7
OUTPUT_SUMMARY_FILENAME = "summary_v2.json"
OUTPUT_TRAJECTORY_FILENAME = "trajectories_v2.npz"
OUTPUT_NEUTRAL_CSV_FILENAME = "neutral_endpoint_hold_only.csv"
OUTPUT_NEUTRAL_VIDEO_FILENAME = "neutral_endpoint_hold_only.mp4"
OUTPUT_EMOTION_CSV_FILENAME = "emotion_endpoint_hold_only.csv"
OUTPUT_EMOTION_VIDEO_FILENAME = "emotion_endpoint_hold_only.mp4"
OUTPUT_ASS_FILENAME = "prompt_timeline_v2.ass"
OUTPUT_FINAL_VIDEO_FILENAME = "beat2_style_emotion_v2_60s.mp4"


class StyleEmotionVideoError(RuntimeError):
    """Raised when V2 generation or its provenance cannot be proven."""


def _resolve_path(value: Any, *, config_dir: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise StyleEmotionVideoError(f"{field} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    path = path.resolve()
    if FORBIDDEN_EXTERNAL_TOKEN in str(path).lower():
        raise StyleEmotionVideoError(
            f"{field} contains a forbidden external-data path token"
        )
    return path


def _require_file(path: Path, field: str) -> Path:
    if not path.is_file():
        raise StyleEmotionVideoError(f"{field} does not exist: {path}")
    return path


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def validate_guidance_scale(value: Any) -> float:
    scale = float(value)
    if not math.isfinite(scale) or not any(
        math.isclose(scale, allowed, abs_tol=1e-12)
        for allowed in ALLOWED_GUIDANCE_SCALES
    ):
        raise StyleEmotionVideoError(
            "sampling.guidance_scale must be exactly 1.0 or 1.5"
        )
    return scale


def classifier_free_guidance_velocity(
    unconditional: torch.Tensor,
    conditioned: torch.Tensor,
    *,
    guidance_scale: float,
) -> torch.Tensor:
    """Apply the explicit CFG velocity equation used by the sampler."""
    scale = validate_guidance_scale(guidance_scale)
    if (
        not isinstance(unconditional, torch.Tensor)
        or not isinstance(conditioned, torch.Tensor)
        or unconditional.shape != conditioned.shape
        or unconditional.device != conditioned.device
        or unconditional.dtype != conditioned.dtype
        or not torch.is_floating_point(unconditional)
    ):
        raise StyleEmotionVideoError(
            "CFG velocities must be matching floating-point tensors"
        )
    return unconditional + scale * (conditioned - unconditional)


def endpoint_hold(
    network_segment: np.ndarray, *, slot_frames: int
) -> np.ndarray:
    """Preserve every network frame and fill only with endpoint repetition."""
    raw = np.asarray(network_segment, dtype=np.float32)
    slot_frames = int(slot_frames)
    if (
        raw.ndim != 2
        or raw.shape[1] != ACTION_DIM
        or len(raw) < 4
        or not np.isfinite(raw).all()
    ):
        raise StyleEmotionVideoError(
            "network segment must be finite [at least 4 frames,18]"
        )
    if slot_frames < len(raw):
        raise StyleEmotionVideoError(
            "planner duration exceeds the montage slot; cropping is forbidden"
        )
    playback = np.repeat(raw[-1:, :], slot_frames, axis=0)
    playback[: len(raw)] = raw
    if (
        not np.array_equal(playback[: len(raw)], raw)
        or (
            slot_frames > len(raw)
            and not np.array_equal(
                playback[len(raw) :],
                np.repeat(
                    raw[-1:, :], slot_frames - len(raw), axis=0
                ),
            )
        )
    ):
        raise StyleEmotionVideoError("endpoint-only hold invariant failed")
    return playback


def build_pair_timeline(
    pairs: Sequence[Mapping[str, Any]],
    *,
    slot_frames: int,
    fps: float,
    target_duration_sec: float,
) -> list[dict[str, Any]]:
    if not pairs or slot_frames < 4 or not math.isclose(
        float(fps), FPS, abs_tol=1e-9
    ):
        raise StyleEmotionVideoError(
            "timeline needs comparison pairs and at least four frames per slot"
        )
    total_frames = len(pairs) * int(slot_frames)
    expected_frames = int(round(float(target_duration_sec) * fps))
    if total_frames != expected_frames:
        raise StyleEmotionVideoError(
            f"timeline has {total_frames} frames, expected {expected_frames}"
        )
    seen_seeds: set[int] = set()
    timeline: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, Mapping):
            raise StyleEmotionVideoError(f"comparison_pairs[{index}] is invalid")
        neutral = str(pair.get("neutral_prompt", "")).strip()
        emotion = str(pair.get("emotion_prompt", "")).strip()
        label = str(pair.get("label", "")).strip()
        seed = pair.get("seed")
        if (
            not neutral
            or not emotion
            or neutral == emotion
            or "neutral affect" not in neutral.lower()
            or "neutral affect" in emotion.lower()
            or not label
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
        ):
            raise StyleEmotionVideoError(
                f"comparison_pairs[{index}] must be a neutral/emotion pair "
                "with a nonnegative integer seed"
            )
        if seed in seen_seeds:
            raise StyleEmotionVideoError("comparison pair seeds must be unique")
        seen_seeds.add(seed)
        start = index * slot_frames
        end = start + slot_frames
        timeline.append(
            {
                "index": index + 1,
                "label": label,
                "neutral_prompt": neutral,
                "emotion_prompt": emotion,
                "neutral_prompt_sha256": hashlib.sha256(
                    neutral.encode("utf-8")
                ).hexdigest(),
                "emotion_prompt_sha256": hashlib.sha256(
                    emotion.encode("utf-8")
                ).hexdigest(),
                "seed": seed,
                "start_frame": start,
                "end_frame_exclusive": end,
                "start_sec": start / fps,
                "end_sec": end / fps,
            }
        )
    return timeline


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
    guidance_scale: float,
) -> str:
    if not timeline or duration_sec <= 0:
        raise StyleEmotionVideoError("ASS timeline must be non-empty")
    scale = validate_guidance_scale(guidance_scale)
    panel_left = pane_width * 2 + 28
    emotion_label_left = pane_width + 18
    panel_width = width - pane_width * 2
    if panel_width < 560:
        raise StyleEmotionVideoError("text panel is too narrow")
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
            "Style:Neutral,DejaVu Sans,22,&H00FFFFFF,&H00FFFFFF,&H80000000,"
            "&H80000000,-1,0,0,0,100,100,0,0,3,1,0,7,18,18,16,1"
        ),
        (
            f"Style:Emotion,DejaVu Sans,22,&H0000E5FF,&H0000E5FF,&H80000000,"
            f"&H80000000,-1,0,0,0,100,100,0,0,3,1,0,7,"
            f"{emotion_label_left},18,16,1"
        ),
        (
            f"Style:Header,DejaVu Sans,23,&H0000E5FF,&H0000E5FF,&H000F172A,"
            f"&H000F172A,-1,0,0,0,100,100,0,0,1,0,0,7,"
            f"{panel_left},24,28,1"
        ),
        (
            f"Style:Prompt,DejaVu Sans,24,&H00FFFFFF,&H00FFFFFF,&H000F172A,"
            f"&H000F172A,0,0,0,0,100,100,0,0,1,0,0,7,"
            f"{panel_left},24,92,1"
        ),
        (
            f"Style:Receipt,DejaVu Sans,17,&H00C8D2DC,&H00C8D2DC,&H000F172A,"
            f"&H000F172A,0,0,0,0,100,100,0,0,1,0,0,1,"
            f"{panel_left},24,24,1"
        ),
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]
    start = _ass_time(0.0)
    end = _ass_time(duration_sec)
    events = [
        f"Dialogue: 0,{start},{end},Neutral,,0,0,0,,NEUTRAL TEXT",
        f"Dialogue: 0,{start},{end},Emotion,,0,0,0,,EMOTION TEXT",
        (
            f"Dialogue: 0,{start},{end},Header,,0,0,0,,"
            "BEAT2 V2 · TEXT → QWEN 128D → PREDICTED STYLE → AdaLN"
        ),
        (
            f"Dialogue: 0,{start},{end},Receipt,,0,0,0,,"
            f"CFG v = uncond + {scale:g} × (cond − uncond)\\N"
            "SAME INITIAL NOISE WITHIN EACH NEUTRAL/EMOTION PAIR\\N"
            "PLANNER-PREDICTED NATIVE DURATION · ENDPOINT HOLD ONLY\\N"
            "NO SMOOTHING · NO TIMEWARP · NO BOUNDARY/LAST-FRAME BLEND\\N"
            "DISPLAY ARRAYS = RAW NETWORK FRAMES + ENDPOINT HOLD\\N"
            "CANONICAL 54-GROUP BEAT2 METADATA · OPEN TEXT UNVALIDATED\\N"
            "18D UPPER BODY + HEAD ORIENTATION · NO FACE BLENDSHAPES"
        ),
    ]
    wrap_width = max(30, min(44, panel_width // 14))
    for item in timeline:
        neutral = "\\N".join(
            textwrap.wrap(
                _ass_escape(str(item["neutral_prompt"])), width=wrap_width
            )
        )
        emotion = "\\N".join(
            textwrap.wrap(
                _ass_escape(str(item["emotion_prompt"])), width=wrap_width
            )
        )
        neutral_style = ", ".join(
            f"{float(value):+.2f}"
            for value in item.get("neutral_predicted_style", ())
        )
        emotion_style = ", ".join(
            f"{float(value):+.2f}"
            for value in item.get("emotion_predicted_style", ())
        )
        label = (
            f"PAIR {int(item['index']):02d}/{len(timeline):02d} · "
            f"{_ass_escape(str(item['label']))}\\N"
            f"seed {int(item['seed'])} · shared initial noise\\N\\N"
            f"NEUTRAL  ({float(item.get('neutral_predicted_duration_sec', 0.0)):.2f}s, "
            f"style [{neutral_style}])\\N{neutral}\\N\\N"
            f"EMOTION  ({float(item.get('emotion_predicted_duration_sec', 0.0)):.2f}s, "
            f"style [{emotion_style}])\\N{emotion}\\N\\N"
            "After each native sample: exact endpoint repetition only."
        )
        events.append(
            "Dialogue: 0,"
            f"{_ass_time(float(item['start_sec']))},"
            f"{_ass_time(float(item['end_sec']))},"
            f"Prompt,,0,0,0,,{label}"
        )
    return "\n".join(header + events) + "\n"


def _load_prompt_latents(
    cache_path: Path,
    *,
    prompts: Sequence[str],
    manifest_records: Mapping[str, Mapping[str, Any]],
    manifest_sha256: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cache_latents, metadata = abc_video.load_qwen_cache(
        cache_path,
        variant="frozen_base",
        manifest_records=manifest_records,
        manifest_sha256=manifest_sha256,
    )
    try:
        with np.load(cache_path, allow_pickle=False) as payload:
            clip_ids = payload["clip_ids"].astype(str)
            cache_prompts = payload["prompts"].astype(str)
            splits = payload["fixed_split_assignments"].astype(str)
    except (OSError, ValueError, KeyError) as error:
        raise StyleEmotionVideoError(
            f"cannot read prompt identities from Qwen cache: {error}"
        ) from error
    if (
        clip_ids.shape != cache_prompts.shape
        or clip_ids.shape != splits.shape
        or len(set(clip_ids.tolist())) != len(clip_ids)
    ):
        raise StyleEmotionVideoError("Qwen prompt identity arrays are invalid")
    result: dict[str, np.ndarray] = {}
    prompt_receipts: list[dict[str, Any]] = []
    for prompt in sorted(set(str(value) for value in prompts)):
        indices = np.flatnonzero(cache_prompts == prompt)
        test_indices = [
            int(index) for index in indices if splits[int(index)] == "test"
        ]
        if not test_indices:
            raise StyleEmotionVideoError(
                f"prompt is not a held-out canonical BEAT2 prompt: {prompt}"
            )
        representative = test_indices[0]
        latent = np.asarray(
            cache_latents[str(clip_ids[representative])], dtype=np.float32
        )
        if latent.shape != (TEXT_LATENT_DIM,) or not np.isfinite(latent).all():
            raise StyleEmotionVideoError("canonical Qwen latent is invalid")
        for index in indices:
            candidate = np.asarray(
                cache_latents[str(clip_ids[int(index)])], dtype=np.float32
            )
            if not np.array_equal(candidate, latent):
                raise StyleEmotionVideoError(
                    f"same canonical prompt has multiple Qwen latents: {prompt}"
                )
        result[prompt] = latent.copy()
        prompt_receipts.append(
            {
                "prompt": prompt,
                "cache_row_count": int(len(indices)),
                "held_out_test_row_count": len(test_indices),
                "representative_test_clip_id": str(
                    clip_ids[representative]
                ),
                "latent_sha256": hashlib.sha256(
                    latent.tobytes()
                ).hexdigest(),
                "latent_l2": float(np.linalg.norm(latent)),
                "all_prompt_rows_exactly_equal": True,
            }
        )
    return result, {
        "path": str(cache_path),
        "sha256": abc_video.sha256_file(cache_path),
        "artifact_kind": metadata["artifact_kind"],
        "data_policy": metadata["data_policy"],
        "source_manifest_sha256": metadata["source_manifest_sha256"],
        "prompt_receipts": prompt_receipts,
        "offline_cache_lookup_only": True,
        "network_or_model_download": False,
    }


def _load_v2_checkpoint(
    checkpoint_path: Path,
    training_summary_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_qwen_cache_sha256: str,
    minimum_training_steps: int,
    device: torch.device,
) -> tuple[torch.nn.Module, QwenStyleHead, dict[str, Any]]:
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise StyleEmotionVideoError(
            f"cannot load V2 generator checkpoint: {error}"
        ) from error
    if not isinstance(checkpoint, Mapping):
        raise StyleEmotionVideoError("V2 checkpoint is not a mapping")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": CHECKPOINT_ARTIFACT_KIND,
        "experimental_only": True,
        "formal_release_eligible": False,
        "condition_policy": CONDITION_POLICY,
        "architecture": ULA_MMDIT_V3_ADALN_ARCHITECTURE,
        "action_dim": ACTION_DIM,
        "condition_dim": CONDITION_DIM,
        "joint_order": list(JOINT_ORDER_18D),
        "no_external_data": True,
        "no_kimodo": True,
    }
    changed = [
        field
        for field, expected_value in expected.items()
        if checkpoint.get(field) != expected_value
    ]
    if changed:
        raise StyleEmotionVideoError(
            f"checkpoint is not the exact isolated V2 contract: {changed}"
        )
    data_receipt = checkpoint.get("data_receipt")
    training_contract = checkpoint.get("training_contract")
    gate = checkpoint.get("anti_collapse_gate")
    semantic_receipt = (
        training_contract.get("semantic_perceptual")
        if isinstance(training_contract, Mapping)
        else None
    )
    if (
        not isinstance(data_receipt, Mapping)
        or not isinstance(training_contract, Mapping)
        or not isinstance(gate, Mapping)
        or gate.get("passed") is not True
        or data_receipt.get("no_external_data") is not True
        or data_receipt.get("no_kimodo") is not True
        or (data_receipt.get("manifest") or {}).get("sha256")
        != expected_manifest_sha256
        or (data_receipt.get("frozen_qwen_cache") or {}).get("sha256")
        != expected_qwen_cache_sha256
        or training_contract.get("policy") != EXPECTED_TRAINING_POLICY
        or not isinstance(semantic_receipt, Mapping)
        or semantic_receipt.get("enabled") is not True
        or semantic_receipt.get("no_external_data") is not True
        or semantic_receipt.get("no_kimodo") is not True
        or semantic_receipt.get("use_global_train_prototype_bank")
        is not True
        or semantic_receipt.get("use_in_batch_contrastive") is not False
        or float(semantic_receipt.get("contrastive_weight", -1.0)) != 0.0
        or float(
            semantic_receipt.get("global_contrastive_weight", 0.0)
        )
        <= 0.0
        or semantic_receipt.get("prototype_fit_split") != "train"
        or semantic_receipt.get("prototype_aggregation")
        != "require_identical"
        or int(
            semantic_receipt.get(
                "validation_or_test_rows_used_for_prototypes", -1
            )
        )
        != 0
        or (
            semantic_receipt.get("global_prototype_source_qwen_cache")
            or {}
        ).get("sha256")
        != expected_qwen_cache_sha256
        or int(training_contract.get("steps", 0))
        < int(minimum_training_steps)
    ):
        raise StyleEmotionVideoError(
            "checkpoint has not passed the production data/training/"
            "anti-collapse contract"
        )
    summary = abc_video.load_json(training_summary_path)
    checkpoint_sha256 = abc_video.sha256_file(checkpoint_path)
    try:
        summary_checkpoint = Path(str(summary.get("checkpoint"))).resolve()
    except (TypeError, ValueError):
        summary_checkpoint = Path()
    if (
        summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("artifact_kind")
        != CHECKPOINT_SUMMARY_ARTIFACT_KIND
        or summary.get("status") != "experimental_candidate"
        or summary.get("checkpoint_sha256") != checkpoint_sha256
        or summary_checkpoint != checkpoint_path.resolve()
        or int(summary.get("completed_steps", -1))
        != int(summary.get("target_steps", -2))
        or int(summary.get("completed_steps", 0))
        < int(minimum_training_steps)
        or (summary.get("anti_collapse_gate") or {}).get("passed") is not True
        or summary.get("no_external_data") is not True
        or summary.get("no_kimodo") is not True
    ):
        raise StyleEmotionVideoError(
            "production V2 training summary is incomplete or rejected"
        )
    model_config = checkpoint.get("config")
    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_config, Mapping) or not isinstance(
        model_state, Mapping
    ):
        raise StyleEmotionVideoError("checkpoint model payload is missing")
    model_shape = {
        "hidden_dim": int(model_config.get("hidden_dim", -1)),
        "layers": int(model_config.get("layers", -1)),
        "semantic_tokens": int(model_config.get("semantic_tokens", -1)),
    }
    if model_shape != {
        "hidden_dim": EXPECTED_HIDDEN_DIM,
        "layers": EXPECTED_LAYERS,
        "semantic_tokens": EXPECTED_SEMANTIC_TOKENS,
    }:
        raise StyleEmotionVideoError(
            f"unexpected V2 model shape: {model_shape}"
        )
    model = create_ula_model(
        checkpoint["architecture"],
        action_dim=ACTION_DIM,
        condition_dim=CONDITION_DIM,
        **model_shape,
    )
    model.load_state_dict(model_state, strict=True)
    action_stats = checkpoint.get("action_stats")
    if not isinstance(action_stats, Mapping):
        raise StyleEmotionVideoError("checkpoint action_stats are missing")
    model.action_stats = {
        name: torch.as_tensor(
            action_stats.get(name), dtype=torch.float32
        ).clone()
        for name in ("mean", "std")
    }
    if (
        any(value.shape != (ACTION_DIM,) for value in model.action_stats.values())
        or any(
            not torch.isfinite(value).all()
            for value in model.action_stats.values()
        )
        or torch.any(model.action_stats["std"] <= 0)
    ):
        raise StyleEmotionVideoError("checkpoint action_stats are invalid")
    try:
        style_head = QwenStyleHead.from_config(
            checkpoint["qwen_style_head_config"]
        )
        style_head.load_state_dict(
            checkpoint["qwen_style_head_state_dict"], strict=True
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise StyleEmotionVideoError(
            f"checkpoint QwenStyleHead is invalid: {error}"
        ) from error
    model = model.to(device).eval()
    style_head = style_head.to(device).eval()
    model.requires_grad_(False)
    style_head.requires_grad_(False)
    return model, style_head, {
        "path": str(checkpoint_path),
        "sha256": checkpoint_sha256,
        "artifact_kind": checkpoint["artifact_kind"],
        "architecture": checkpoint["architecture"],
        "global_step": int(checkpoint.get("global_step", 0)),
        "posttrain_steps": int(training_contract["steps"]),
        "condition_policy": checkpoint["condition_policy"],
        "anti_collapse_gate": dict(gate),
        "training_summary": {
            "path": str(training_summary_path),
            "sha256": abc_video.sha256_file(training_summary_path),
            "status": summary["status"],
            "completed_steps": int(summary["completed_steps"]),
        },
        "qwen_style_head_config": dict(
            checkpoint["qwen_style_head_config"]
        ),
        "no_external_data": True,
        "no_kimodo": True,
    }


@torch.no_grad()
def compose_text_style_condition(
    style_head: QwenStyleHead,
    latent: np.ndarray,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    latent = np.asarray(latent, dtype=np.float32)
    if latent.shape != (TEXT_LATENT_DIM,) or not np.isfinite(latent).all():
        raise StyleEmotionVideoError("text latent must be finite [128]")
    tensor = torch.as_tensor(latent[None, :], device=device)
    keep = torch.ones(1, dtype=torch.bool, device=device)
    predicted_style = style_head(tensor, keep)
    condition = assemble_text_style_conditions(
        tensor, predicted_style, keep
    )
    condition_np = condition[0].detach().cpu().numpy().astype(np.float32)
    style_np = (
        predicted_style[0].detach().cpu().numpy().astype(np.float32)
    )
    if (
        condition_np.shape != (CONDITION_DIM,)
        or style_np.shape != (3,)
        or not np.isfinite(condition_np).all()
        or not np.array_equal(
            condition_np[STYLE_CONTROL_SLICE], style_np
        )
        or not np.array_equal(
            condition_np[TEXT_LATENT_SLICE], latent
        )
        or np.any(condition_np[: STYLE_CONTROL_SLICE.start])
    ):
        raise StyleEmotionVideoError("runtime V2 condition layout is invalid")
    return condition_np, style_np


@torch.no_grad()
def predict_duration_sec(
    model: torch.nn.Module,
    condition: np.ndarray,
    *,
    device: torch.device,
) -> float:
    tensor = torch.as_tensor(
        np.asarray(condition, dtype=np.float32)[None, :], device=device
    )
    duration = float(
        model.plan_condition(tensor)["duration_sec"][0].detach().cpu()
    )
    if not math.isfinite(duration) or duration <= 0.0:
        raise StyleEmotionVideoError(
            f"planner returned an invalid duration: {duration}"
        )
    return duration


def shared_initial_noise(
    *, seed: int, frames: int, action_dim: int = ACTION_DIM
) -> np.ndarray:
    if int(frames) < 4 or int(action_dim) != ACTION_DIM:
        raise StyleEmotionVideoError("shared initial-noise shape is invalid")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return (
        torch.stack(
            [
                torch.randn(
                    (int(action_dim),),
                    generator=generator,
                    dtype=torch.float32,
                )
                for _ in range(int(frames))
            ],
            dim=0,
        )
        .numpy()
        .copy()
    )


@torch.no_grad()
def sample_trajectory_cfg(
    model: torch.nn.Module,
    condition: np.ndarray,
    *,
    initial_noise: np.ndarray,
    steps: int,
    guidance_scale: float,
    device: torch.device,
) -> np.ndarray:
    """Integrate flow velocity with CFG from an explicit shared noise array."""
    scale = validate_guidance_scale(guidance_scale)
    condition = np.asarray(condition, dtype=np.float32)
    noise = np.asarray(initial_noise, dtype=np.float32)
    steps = int(steps)
    if (
        condition.shape != (CONDITION_DIM,)
        or noise.ndim != 2
        or noise.shape[1] != ACTION_DIM
        or len(noise) < 4
        or not np.isfinite(condition).all()
        or not np.isfinite(noise).all()
        or steps <= 0
    ):
        raise StyleEmotionVideoError("CFG sampling inputs are invalid")
    conditioned = torch.as_tensor(condition[None, :], device=device)
    unconditional = torch.zeros_like(conditioned)
    x = torch.as_tensor(noise[None, :, :], device=device)
    lower, upper = joint_limit_tensors(device, ACTION_DIM)
    action_stats = getattr(model, "action_stats", None)
    if action_stats is None:
        raise StyleEmotionVideoError("V2 model has no action statistics")
    lower = normalize_action_tensor(lower, action_stats)
    upper = normalize_action_tensor(upper, action_stats)
    lower, upper = torch.minimum(lower, upper), torch.maximum(lower, upper)
    x = torch.clamp(x, lower, upper)
    dt = 1.0 / float(steps)
    for index in range(steps):
        t = torch.full(
            (1,), index * dt, dtype=torch.float32, device=device
        )
        unconditioned_velocity = model(x, t, unconditional)
        conditioned_velocity = model(x, t, conditioned)
        velocity = classifier_free_guidance_velocity(
            unconditioned_velocity,
            conditioned_velocity,
            guidance_scale=scale,
        )
        x = torch.clamp(x + dt * velocity, lower, upper)
    result = (
        denormalize_action_tensor(x, action_stats)[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    if result.shape != noise.shape or not np.isfinite(result).all():
        raise StyleEmotionVideoError("CFG sampler produced invalid motion")
    return result


def _pack_segments(
    segments: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if not segments:
        raise StyleEmotionVideoError("cannot pack an empty segment list")
    offsets = [0]
    values = []
    for segment in segments:
        array = np.asarray(segment, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != ACTION_DIM:
            raise StyleEmotionVideoError("packed segment has an invalid shape")
        values.append(array)
        offsets.append(offsets[-1] + len(array))
    return np.concatenate(values, axis=0), np.asarray(offsets, dtype=np.int64)


def _run_ffmpeg_overlay(
    *,
    neutral_video: Path,
    emotion_video: Path,
    ass_path: Path,
    output_path: Path,
    pane_width: int,
    panel_width: int,
    height: int,
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
    temporary = output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.tmp.mp4"
    )
    command = [
        str(ffmpeg),
        "-y",
        "-i",
        str(neutral_video),
        "-i",
        str(emotion_video),
        "-filter_complex",
        (
            f"[0:v]scale={pane_width}:{height}[n];"
            f"[1:v]scale={pane_width}:{height}[e];"
            "[n][e]hstack=inputs=2[robots];"
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
        raise StyleEmotionVideoError(
            "ffmpeg montage composition failed: " + completed.stderr[-2000:]
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
        raise StyleEmotionVideoError(
            "final montage full decode failed: " + decode.stderr[-2000:]
        )
    frame_count, duration_sec = imageio_ffmpeg.count_frames_and_secs(
        str(temporary)
    )
    reader = imageio_ffmpeg.read_frames(str(temporary), pix_fmt="rgb24")
    metadata = next(reader)
    reader.close()
    if (
        tuple(metadata.get("size") or ()) != (output_width, height)
        or not math.isclose(
            float(metadata.get("fps", 0.0)), fps, abs_tol=1e-6
        )
        or abs(frame_count - int(round(duration_sec * fps))) > 1
    ):
        raise StyleEmotionVideoError(
            f"final montage metadata mismatch: {metadata}, {frame_count}"
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


def build_video(
    config_path: str | Path, *, overwrite: bool = False
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = abc_video.load_json(config_path)
    if (
        config.get("schema_version") != SCHEMA_VERSION
        or config.get("artifact_kind") != CONFIG_ARTIFACT_KIND
        or config.get("data_policy") != DATA_POLICY
        or config.get("no_external_data") is not True
        or config.get("no_kimodo") is not True
    ):
        raise StyleEmotionVideoError("V2 video config contract is invalid")
    config_dir = config_path.parent
    paths = {
        name: _require_file(
            _resolve_path(
                config.get(name), config_dir=config_dir, field=name
            ),
            name,
        )
        for name in (
            "source_manifest",
            "qwen_condition_cache",
            "generator_checkpoint",
            "generator_training_summary",
        )
    }
    output_dir = _resolve_path(
        config.get("output_dir"),
        config_dir=config_dir,
        field="output_dir",
    )
    summary_path = output_dir / OUTPUT_SUMMARY_FILENAME
    if summary_path.exists() and not overwrite:
        raise StyleEmotionVideoError(
            f"completed output exists; pass --overwrite: {summary_path}"
        )
    manifest_records, manifest_sha256 = abc_video.load_manifest(
        paths["source_manifest"],
        expected_sha256=config.get("source_manifest_sha256"),
    )
    cache_sha256 = abc_video.sha256_file(paths["qwen_condition_cache"])
    sampling = config.get("sampling")
    if not isinstance(sampling, Mapping):
        raise StyleEmotionVideoError("sampling config is missing")
    guidance_scale = validate_guidance_scale(
        sampling.get("guidance_scale")
    )
    steps = int(sampling.get("steps", 0))
    if steps <= 0:
        raise StyleEmotionVideoError("sampling.steps must be positive")
    device = torch.device(str(sampling.get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise StyleEmotionVideoError(
            "CUDA sampling was requested but is unavailable"
        )
    model, style_head, checkpoint_receipt = _load_v2_checkpoint(
        paths["generator_checkpoint"],
        paths["generator_training_summary"],
        expected_manifest_sha256=manifest_sha256,
        expected_qwen_cache_sha256=cache_sha256,
        minimum_training_steps=int(config.get("minimum_training_steps", 0)),
        device=device,
    )
    target_duration_sec = float(config.get("target_duration_sec", 60.0))
    slot_duration_sec = float(config.get("slot_duration_sec", 0.0))
    if (
        not math.isclose(target_duration_sec, 60.0, abs_tol=1e-9)
        or not math.isfinite(slot_duration_sec)
        or slot_duration_sec <= 0
    ):
        raise StyleEmotionVideoError(
            "target duration must be 60 seconds and slot duration positive"
        )
    slot_frames = int(round(slot_duration_sec * FPS))
    if not math.isclose(
        slot_frames / FPS, slot_duration_sec, abs_tol=1e-9
    ):
        raise StyleEmotionVideoError(
            "slot duration must map exactly to 30 Hz"
        )
    pairs = config.get("comparison_pairs")
    if not isinstance(pairs, list):
        raise StyleEmotionVideoError("comparison_pairs must be a list")
    timeline = build_pair_timeline(
        pairs,
        slot_frames=slot_frames,
        fps=FPS,
        target_duration_sec=target_duration_sec,
    )
    all_prompts = [
        prompt
        for item in timeline
        for prompt in (item["neutral_prompt"], item["emotion_prompt"])
    ]
    prompt_latents, cache_receipt = _load_prompt_latents(
        paths["qwen_condition_cache"],
        prompts=all_prompts,
        manifest_records=manifest_records,
        manifest_sha256=manifest_sha256,
    )

    neutral_raw_segments: list[np.ndarray] = []
    emotion_raw_segments: list[np.ndarray] = []
    neutral_slots: list[np.ndarray] = []
    emotion_slots: list[np.ndarray] = []
    common_noise_segments: list[np.ndarray] = []
    neutral_conditions: list[np.ndarray] = []
    emotion_conditions: list[np.ndarray] = []
    neutral_latents: list[np.ndarray] = []
    emotion_latents: list[np.ndarray] = []
    neutral_styles: list[np.ndarray] = []
    emotion_styles: list[np.ndarray] = []
    for item in timeline:
        neutral_latent = prompt_latents[item["neutral_prompt"]]
        emotion_latent = prompt_latents[item["emotion_prompt"]]
        neutral_condition, neutral_style = compose_text_style_condition(
            style_head, neutral_latent, device=device
        )
        emotion_condition, emotion_style = compose_text_style_condition(
            style_head, emotion_latent, device=device
        )
        neutral_duration = predict_duration_sec(
            model, neutral_condition, device=device
        )
        emotion_duration = predict_duration_sec(
            model, emotion_condition, device=device
        )
        neutral_frames = max(4, int(round(neutral_duration * FPS)))
        emotion_frames = max(4, int(round(emotion_duration * FPS)))
        if max(neutral_frames, emotion_frames) > slot_frames:
            raise StyleEmotionVideoError(
                "planner duration exceeds its slot; increase slot_duration_sec "
                "instead of cropping or time-warping"
            )
        common_noise = shared_initial_noise(
            seed=int(item["seed"]),
            frames=max(neutral_frames, emotion_frames),
        )
        neutral_raw = sample_trajectory_cfg(
            model,
            neutral_condition,
            initial_noise=common_noise[:neutral_frames],
            steps=steps,
            guidance_scale=guidance_scale,
            device=device,
        )
        emotion_raw = sample_trajectory_cfg(
            model,
            emotion_condition,
            initial_noise=common_noise[:emotion_frames],
            steps=steps,
            guidance_scale=guidance_scale,
            device=device,
        )
        neutral_slot = endpoint_hold(
            neutral_raw, slot_frames=slot_frames
        )
        emotion_slot = endpoint_hold(
            emotion_raw, slot_frames=slot_frames
        )
        neutral_raw_segments.append(neutral_raw)
        emotion_raw_segments.append(emotion_raw)
        neutral_slots.append(neutral_slot)
        emotion_slots.append(emotion_slot)
        common_noise_segments.append(common_noise)
        neutral_conditions.append(neutral_condition)
        emotion_conditions.append(emotion_condition)
        neutral_latents.append(neutral_latent)
        emotion_latents.append(emotion_latent)
        neutral_styles.append(neutral_style)
        emotion_styles.append(emotion_style)
        item.update(
            {
                "guidance_scale": guidance_scale,
                "same_seed_neutral_emotion": True,
                "shared_initial_noise_prefix_frames": min(
                    neutral_frames, emotion_frames
                ),
                "shared_initial_noise_sha256": hashlib.sha256(
                    common_noise.tobytes()
                ).hexdigest(),
                "neutral_predicted_duration_sec": neutral_duration,
                "emotion_predicted_duration_sec": emotion_duration,
                "neutral_network_frames": neutral_frames,
                "emotion_network_frames": emotion_frames,
                "neutral_quantized_duration_sec": neutral_frames / FPS,
                "emotion_quantized_duration_sec": emotion_frames / FPS,
                "neutral_endpoint_hold_frames": slot_frames - neutral_frames,
                "emotion_endpoint_hold_frames": slot_frames - emotion_frames,
                "neutral_predicted_style": neutral_style.tolist(),
                "emotion_predicted_style": emotion_style.tolist(),
                "neutral_condition_sha256": hashlib.sha256(
                    neutral_condition.tobytes()
                ).hexdigest(),
                "emotion_condition_sha256": hashlib.sha256(
                    emotion_condition.tobytes()
                ).hexdigest(),
                "neutral_network_raw": abc_video.trajectory_metrics(
                    neutral_raw, fps=FPS
                ),
                "emotion_network_raw": abc_video.trajectory_metrics(
                    emotion_raw, fps=FPS
                ),
                "emotion_minus_neutral_raw_overlap": (
                    abc_video.trajectory_delta_metrics(
                        emotion_raw[: min(neutral_frames, emotion_frames)],
                        neutral_raw[: min(neutral_frames, emotion_frames)],
                    )
                ),
                "display_transform": "endpoint repetition only",
            }
        )

    display_neutral = np.concatenate(neutral_slots, axis=0)
    display_emotion = np.concatenate(emotion_slots, axis=0)
    expected_frames = int(round(target_duration_sec * FPS))
    if (
        display_neutral.shape != (expected_frames, ACTION_DIM)
        or display_emotion.shape != (expected_frames, ACTION_DIM)
        or not np.isfinite(display_neutral).all()
        or not np.isfinite(display_emotion).all()
    ):
        raise StyleEmotionVideoError("60-second display arrays are invalid")
    packed_neutral, neutral_offsets = _pack_segments(
        neutral_raw_segments
    )
    packed_emotion, emotion_offsets = _pack_segments(
        emotion_raw_segments
    )
    packed_noise, noise_offsets = _pack_segments(common_noise_segments)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / OUTPUT_TRAJECTORY_FILENAME
    _atomic_npz(
        trajectory_path,
        network_raw_neutral=packed_neutral,
        network_raw_neutral_offsets=neutral_offsets,
        network_raw_emotion=packed_emotion,
        network_raw_emotion_offsets=emotion_offsets,
        display_neutral_endpoint_hold_only=display_neutral,
        display_emotion_endpoint_hold_only=display_emotion,
        shared_initial_noise=packed_noise,
        shared_initial_noise_offsets=noise_offsets,
        neutral_conditions=np.stack(neutral_conditions).astype(np.float32),
        emotion_conditions=np.stack(emotion_conditions).astype(np.float32),
        neutral_qwen_latents=np.stack(neutral_latents).astype(np.float32),
        emotion_qwen_latents=np.stack(emotion_latents).astype(np.float32),
        neutral_predicted_styles=np.stack(neutral_styles).astype(np.float32),
        emotion_predicted_styles=np.stack(emotion_styles).astype(np.float32),
        neutral_prompts=np.asarray(
            [item["neutral_prompt"] for item in timeline]
        ),
        emotion_prompts=np.asarray(
            [item["emotion_prompt"] for item in timeline]
        ),
        seeds=np.asarray(
            [item["seed"] for item in timeline], dtype=np.int64
        ),
        slot_frames=np.asarray(slot_frames, dtype=np.int64),
        fps=np.asarray(FPS, dtype=np.float32),
        joint_order=np.asarray(JOINT_ORDER_18D),
        guidance_scale=np.asarray(guidance_scale, dtype=np.float32),
    )

    render = config.get("render")
    if not isinstance(render, Mapping):
        raise StyleEmotionVideoError("render config is missing")
    pane_width = int(render.get("pane_width", 600))
    panel_width = int(render.get("panel_width", 720))
    height = int(render.get("height", 720))
    if min(pane_width, panel_width, height) <= 0:
        raise StyleEmotionVideoError("render dimensions must be positive")
    neutral_csv = output_dir / OUTPUT_NEUTRAL_CSV_FILENAME
    neutral_video = output_dir / OUTPUT_NEUTRAL_VIDEO_FILENAME
    emotion_csv = output_dir / OUTPUT_EMOTION_CSV_FILENAME
    emotion_video = output_dir / OUTPUT_EMOTION_VIDEO_FILENAME
    render_neutral = abc_video._render_single(
        display_neutral,
        csv_path=neutral_csv,
        mp4_path=neutral_video,
        fps=FPS,
        width=pane_width,
        height=height,
        simplified=bool(render.get("simplified", False)),
    )
    render_emotion = abc_video._render_single(
        display_emotion,
        csv_path=emotion_csv,
        mp4_path=emotion_video,
        fps=FPS,
        width=pane_width,
        height=height,
        simplified=bool(render.get("simplified", False)),
    )
    ass_path = output_dir / OUTPUT_ASS_FILENAME
    ass_document = build_ass_document(
        timeline,
        duration_sec=target_duration_sec,
        width=pane_width * 2 + panel_width,
        height=height,
        pane_width=pane_width,
        guidance_scale=guidance_scale,
    )
    ass_path.write_text(ass_document, encoding="utf-8")
    final_video = output_dir / OUTPUT_FINAL_VIDEO_FILENAME
    final_receipt = _run_ffmpeg_overlay(
        neutral_video=neutral_video,
        emotion_video=emotion_video,
        ass_path=ass_path,
        output_path=final_video,
        pane_width=pane_width,
        panel_width=panel_width,
        height=height,
        fps=FPS,
    )
    if abs(final_receipt["decoded_frames"] - expected_frames) > 1:
        raise StyleEmotionVideoError(
            "final MP4 does not decode to exactly 60 seconds"
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": SUMMARY_ARTIFACT_KIND,
        "status": "complete",
        "experimental_only": True,
        "formal_release_eligible": False,
        "config": str(config_path),
        "config_sha256": abc_video.sha256_file(config_path),
        "source_manifest": str(paths["source_manifest"]),
        "source_manifest_sha256": manifest_sha256,
        "data_policy": DATA_POLICY,
        "no_external_data": True,
        "no_kimodo": True,
        "checkpoint": checkpoint_receipt,
        "qwen_cache": cache_receipt,
        "sampling": {
            "steps": steps,
            "guidance_scale": guidance_scale,
            "cfg_equation": "uncond + scale * (conditioned - uncond)",
            "device": str(device),
            "fps": FPS,
            "same_initial_noise_within_each_pair": True,
            "planner_duration_used": True,
            "forced_six_second_network_sample": False,
        },
        "timeline": timeline,
        "display_contract": {
            "target_duration_sec": target_duration_sec,
            "decoded_frames_expected": expected_frames,
            "slot_duration_sec": slot_duration_sec,
            "transform": "native network frames then exact endpoint repetition",
            "smoothing": False,
            "time_warp": False,
            "network_frame_crop": False,
            "boundary_blend": False,
            "forced_last_frame_blend": False,
            "pose_postprocess": False,
            "raw_and_display_arrays_saved": True,
        },
        "semantic_scope": (
            "experimental canonical 54-group BEAT2 metadata control; "
            "not validated open-text or formal robot affect truth"
        ),
        "metrics": {
            "network_raw_segment_jerk_rms_mean": {
                "neutral": float(
                    np.mean(
                        [
                            item["neutral_network_raw"]["jerk_rad_s3"]["rms"]
                            for item in timeline
                        ]
                    )
                ),
                "emotion": float(
                    np.mean(
                        [
                            item["emotion_network_raw"]["jerk_rad_s3"]["rms"]
                            for item in timeline
                        ]
                    )
                ),
            },
            "display_neutral": abc_video.trajectory_metrics(
                display_neutral, fps=FPS
            ),
            "display_emotion": abc_video.trajectory_metrics(
                display_emotion, fps=FPS
            ),
            "display_metric_warning": (
                "full-display derivatives include unblended pair boundaries"
            ),
        },
        "artifacts": {
            "trajectory_npz": {
                "path": str(trajectory_path),
                "sha256": abc_video.sha256_file(trajectory_path),
                "bytes": trajectory_path.stat().st_size,
            },
            "neutral_csv": {
                "path": str(neutral_csv),
                "sha256": abc_video.sha256_file(neutral_csv),
            },
            "emotion_csv": {
                "path": str(emotion_csv),
                "sha256": abc_video.sha256_file(emotion_csv),
            },
            "neutral_robot_video": {
                "path": str(neutral_video),
                "sha256": abc_video.sha256_file(neutral_video),
                "render": render_neutral,
            },
            "emotion_robot_video": {
                "path": str(emotion_video),
                "sha256": abc_video.sha256_file(emotion_video),
                "render": render_emotion,
            },
            "prompt_timeline_ass": {
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
        summary = build_video(args.config, overwrite=args.overwrite)
    except (
        StyleEmotionVideoError,
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
