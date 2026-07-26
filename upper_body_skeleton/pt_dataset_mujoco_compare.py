#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re


os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq
import torch

from upper_body_skeleton.long_emotion_infer import trajectory_quality
from upper_body_skeleton.motion_latent import load_motion_latent_episodes, stratified_episode_split
from upper_body_skeleton.mujoco_playback import render_trajectory_comparison
from upper_body_skeleton.pt_mujoco_infer import (
    DEFAULT_SEMANTIC_ADAPTER_CHECKPOINT,
    EXPERIMENTAL_KIMODO_CHECKPOINT,
    PtMotionGenerator,
    load_motion_latent_lora_condition_builder,
    validate_generator_condition_source,
)
from upper_body_skeleton.retarget_v2 import JOINT_ORDER
from upper_body_skeleton.semantic_adapter import AdapterConditionBuilder, load_semantic_adapter
from upper_body_skeleton.ula_training import KIMODO_CONDITION_DIM


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = REPO_ROOT / "datasets" / "kimodo_lerobot_mmdit_lite"
DEFAULT_MOTION_SPLIT_CHECKPOINT = (
    REPO_ROOT / "training" / "runs" / "kimodo_motion_latent_v1" / "motion_latent_checkpoint.pt"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "training"
    / "runs"
    / "kimodo_mmdit_lite_qwen_compatible_5k_math_sdp"
    / "mujoco_dataset_comparison"
)
COMPARISON_TITLE_HEIGHT = 40


def _atomic_json_write(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _safe_name(value):
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value)).strip("_")
    return value.lower() or "sample"


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _action_data_manifest(dataset_dir):
    dataset_dir = Path(dataset_dir)
    files = sorted((dataset_dir / "data").rglob("*.parquet"))
    if not files:
        raise ValueError("comparison dataset contains no action parquet files")
    records = []
    manifest_digest = hashlib.sha256()
    for path in files:
        relative_path = path.relative_to(dataset_dir).as_posix()
        file_hash = _file_sha256(path)
        records.append(
            {
                "path": relative_path,
                "bytes": path.stat().st_size,
                "sha256": file_hash,
            }
        )
        manifest_digest.update(relative_path.encode("utf-8"))
        manifest_digest.update(b"\0")
        manifest_digest.update(file_hash.encode("ascii"))
        manifest_digest.update(b"\n")
    return {"sha256": manifest_digest.hexdigest(), "files": records}


def validate_dataset_contract(dataset_dir):
    dataset_dir = Path(dataset_dir)
    info_path = dataset_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    names = info.get("features", {}).get("observation.state", {}).get("names")
    if names != JOINT_ORDER:
        raise ValueError("dataset observation.state joint order does not match the MuJoCo V2 joint order")
    fps = float(info.get("fps", 0))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("dataset fps must be finite and positive")
    return {"info_path": str(info_path), "fps": fps, "joint_order": list(names)}


def _prioritize_diverse_rows(rows):
    by_behavior = {}
    for row in rows:
        by_behavior.setdefault(str(row["behavior_id"]), []).append(row)
    selected = []
    selected_episode_ids = set()
    for behavior_index, behavior_id in enumerate(sorted(by_behavior)):
        options = sorted(
            by_behavior[behavior_id],
            key=lambda row: (str(row["emotion_id"]), int(row["episode_index"])),
        )
        row = options[behavior_index % len(options)]
        selected.append(row)
        selected_episode_ids.add(int(row["episode_index"]))
    selected.extend(
        row for row in rows if int(row["episode_index"]) not in selected_episode_ids
    )
    return selected


def _normalized_behavior_ids(value):
    if value is None:
        return None
    values = [value] if isinstance(value, str) else list(value)
    normalized = []
    for item in values:
        behavior_id = str(item).strip()
        if not behavior_id:
            raise ValueError("behavior_id cannot be empty")
        if behavior_id not in normalized:
            normalized.append(behavior_id)
    if not normalized:
        raise ValueError("at least one behavior_id is required")
    return normalized


def select_dataset_reference_rows(
    dataset_dir,
    split_checkpoint_path,
    *,
    motion_latent_split="test",
    episode_index=None,
    behavior_id=None,
    emotion_id=None,
    count=1,
):
    behavior_ids = _normalized_behavior_ids(behavior_id)
    split_names = ("train", "validation", "test")
    if motion_latent_split not in set(split_names):
        raise ValueError("motion_latent_split must be train, validation, or test")
    if episode_index is not None and (behavior_ids is not None or emotion_id is not None):
        raise ValueError("select by either episode_index or behavior/emotion labels, not both")
    if emotion_id is not None and behavior_ids is None:
        raise ValueError("emotion_id requires at least one behavior_id")
    count = int(count)
    if count <= 0:
        raise ValueError("count must be positive")
    if episode_index is not None and count != 1:
        raise ValueError("episode_index selects exactly one comparison")
    checkpoint = torch.load(split_checkpoint_path, map_location="cpu", weights_only=True)
    split_manifest = checkpoint.get("split_episode_indices")
    if not isinstance(split_manifest, dict) or set(split_manifest) != set(split_names):
        raise ValueError("motion checkpoint must contain train/validation/test episode partitions")
    columns = ["episode_index", "sample_id", "language_instruction", "behavior_id", "emotion_id"]
    rows = pq.read_table(Path(dataset_dir) / "meta" / "semantic_index.parquet", columns=columns).to_pylist()
    dataset_ids = [int(row["episode_index"]) for row in rows]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("dataset semantic index contains duplicate episode indices")
    normalized_manifest = {
        name: [int(value) for value in split_manifest[name]]
        for name in split_names
    }
    partition_ids = [value for name in split_names for value in normalized_manifest[name]]
    if len(partition_ids) != len(set(partition_ids)):
        raise ValueError("motion checkpoint episode partitions overlap or contain duplicates")
    if set(partition_ids) != set(dataset_ids):
        raise ValueError("motion checkpoint episode partitions do not cover this dataset exactly")

    seed = (checkpoint.get("config") or {}).get("seed")
    if seed is None:
        raise ValueError("motion checkpoint does not record the stratified partition seed")
    semantic_episodes = [
        {
            "episode_index": int(row["episode_index"]),
            "meta": {
                "behavior_id": str(row["behavior_id"]),
                "emotion_id": str(row["emotion_id"]),
            },
        }
        for row in rows
    ]
    expected_partitions = stratified_episode_split(semantic_episodes, seed=int(seed))
    for name, expected_rows in zip(split_names, expected_partitions):
        expected_ids = {int(row["episode_index"]) for row in expected_rows}
        if set(normalized_manifest[name]) != expected_ids:
            raise ValueError(
                "motion checkpoint partition does not match this dataset's labels and recorded seed"
            )
    for checkpoint_field, row_field in (
        ("behavior_ids", "behavior_id"),
        ("emotion_ids", "emotion_id"),
    ):
        checkpoint_labels = checkpoint.get(checkpoint_field)
        dataset_labels = {str(row[row_field]) for row in rows}
        if checkpoint_labels is not None and set(map(str, checkpoint_labels)) != dataset_labels:
            raise ValueError(f"motion checkpoint {checkpoint_field} do not match this dataset")

    allowed_ids = set(normalized_manifest[motion_latent_split])
    rows = [row for row in rows if int(row["episode_index"]) in allowed_ids]
    if episode_index is not None:
        rows = [row for row in rows if int(row["episode_index"]) == int(episode_index)]
    if behavior_ids is not None:
        behavior_id_set = set(behavior_ids)
        rows = [
            row
            for row in rows
            if row["behavior_id"] in behavior_id_set
            and (emotion_id is None or row["emotion_id"] == emotion_id)
        ]
    rows.sort(key=lambda row: int(row["episode_index"]))
    if not rows:
        raise ValueError("no dataset reference matches the requested Motion Metric partition and labels")
    if episode_index is None and (behavior_ids is None or len(behavior_ids) > 1):
        rows = _prioritize_diverse_rows(rows)
    if len(rows) < count:
        raise ValueError(f"requested {count} comparisons but only {len(rows)} references match")
    selected = rows[:count]
    for row in selected:
        text = str(row.get("language_instruction", "")).strip()
        if not text:
            raise ValueError(f"episode {row['episode_index']} has no language_instruction")
        row["language_instruction"] = text
    return selected


def load_reference_trajectories(dataset_dir, selected_rows, *, generator_checkpoint=None):
    requested = {int(row["episode_index"]) for row in selected_rows}
    episodes = load_motion_latent_episodes(dataset_dir)
    stats_episodes = episodes
    preprocessing = {"mode": "raw_legacy"}
    if generator_checkpoint is not None and generator_checkpoint.get("v2_contracts") is not None:
        from upper_body_skeleton.ula_v2_conditioning import (
            clean_joint_trajectory,
            extract_style_features,
            normalize_style_features,
            trim_episode,
        )

        contracts = generator_checkpoint["v2_contracts"]
        processed = []
        for episode in episodes:
            item = dict(episode)
            item["meta"] = dict(episode.get("meta") or {})
            item["fps"] = float(item.get("fps") or item["meta"].get("fps") or 30.0)
            item["actions"] = clean_joint_trajectory(
                item["actions"],
                fps=item["fps"],
                preprocess_contract=contracts["preprocess"],
            )
            item = trim_episode(item, active_window_contract=contracts["active_window"])
            item["style_features"] = extract_style_features(item["actions"], fps=item["fps"])
            item["style_controls"] = normalize_style_features(
                item["style_features"], contracts["style"]
            )
            processed.append(item)
        episodes = processed
        training_ids = {
            int(value)
            for value in (
                generator_checkpoint.get("training_episode_indices")
                or generator_checkpoint.get("split_episode_indices", {}).get("train")
                or []
            )
        }
        if not training_ids:
            raise ValueError("V2 generator checkpoint does not record its training episode IDs")
        stats_episodes = [episode for episode in episodes if int(episode["episode_index"]) in training_ids]
        if len(stats_episodes) != len(training_ids):
            raise ValueError("V2 generator training episode IDs do not match the comparison dataset")
        preprocessing = {
            "mode": "v2_contract",
            "contracts_sha256": contracts["sha256"],
            "preprocess_sha256": contracts["preprocess"]["sha256"],
            "active_window_sha256": contracts["active_window"]["sha256"],
        }
    by_index = {int(episode["episode_index"]): episode for episode in episodes if int(episode["episode_index"]) in requested}
    if set(by_index) != requested:
        raise ValueError("not all selected reference episodes were found in the LeRobot parquet data")
    for row in selected_rows:
        episode = by_index[int(row["episode_index"])]
        if episode["meta"]["behavior_id"] != row["behavior_id"]:
            raise ValueError("reference behavior metadata mismatch")
        if episode["meta"]["emotion_id"] != row["emotion_id"]:
            raise ValueError("reference emotion metadata mismatch")
    actions = np.concatenate([episode["actions"] for episode in stats_episodes], axis=0).astype(np.float32)
    return by_index, {
        "dataset_episode_count": len(episodes),
        "episode_count": len(stats_episodes),
        "frame_count": int(actions.shape[0]),
        "mean": actions.mean(axis=0),
        "std": actions.std(axis=0),
        "preprocessing": preprocessing,
    }


def trajectory_comparison_metrics(network, reference, *, fps):
    network = np.asarray(network, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if network.shape != reference.shape:
        raise ValueError("trajectory comparison requires identical network/reference shapes")
    if network.ndim != 2 or network.shape[1] != len(JOINT_ORDER) or network.shape[0] < 1:
        raise ValueError(f"trajectories must have shape [frames, {len(JOINT_ORDER)}]")
    if not np.isfinite(network).all() or not np.isfinite(reference).all():
        raise ValueError("trajectories must contain only finite values")
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    difference = network - reference
    return {
        "mae_rad": float(np.abs(difference).mean()),
        "rmse_rad": float(np.sqrt(np.square(difference).mean())),
        "per_joint_mae_rad": {
            joint: float(np.abs(difference[:, index]).mean())
            for index, joint in enumerate(JOINT_ORDER)
        },
        "network_quality": trajectory_quality(network, fps=fps),
        "reference_quality": trajectory_quality(reference, fps=fps),
    }


def inspect_comparison_video(path, *, expected_frames, title_height=COMPARISON_TITLE_HEIGHT):
    expected_frames = int(expected_frames)
    if expected_frames <= 0:
        raise ValueError("expected_frames must be positive")
    title_height = int(title_height)
    if title_height < 0:
        raise ValueError("title_height cannot be negative")
    sampled_indices = sorted({0, expected_frames // 2, expected_frames - 1})
    reader = imageio.get_reader(path)
    frames = []
    try:
        decoded_frames = int(reader.count_frames())
        if decoded_frames != expected_frames:
            raise ValueError(
                f"comparison video has {decoded_frames} frames; expected {expected_frames}"
            )
        for index in sampled_indices:
            frames.append(reader.get_data(index)[:, :, :3])
    finally:
        reader.close()
    if not frames:
        raise ValueError("comparison video contains no inspectable frames")
    height, width, _ = frames[0].shape
    if any(frame.shape != frames[0].shape for frame in frames):
        raise ValueError("comparison video frames do not have a stable shape")
    if width % 2:
        raise ValueError("comparison video width is not divisible into left/right panes")
    divider_margin = 4
    content_top = title_height + 4
    half_width = width // 2
    if content_top >= height or half_width <= divider_margin:
        raise ValueError("comparison video is too small for content-only inspection")
    left_variance = float(
        np.mean([frame[content_top:, : half_width - divider_margin].var() for frame in frames])
    )
    right_variance = float(
        np.mean([frame[content_top:, half_width + divider_margin :].var() for frame in frames])
    )
    if min(left_variance, right_variance) <= 1.0:
        raise ValueError("comparison video contains a blank MuJoCo pane")
    return {
        "decoded_frames": decoded_frames,
        "sampled_frame_indices": sampled_indices,
        "decoded_shape": [height, width, 3],
        "content_crop": {
            "top": content_top,
            "divider_margin": divider_margin,
        },
        "left_pixel_variance": left_variance,
        "right_pixel_variance": right_variance,
        "nonblank": True,
    }


def run_dataset_mujoco_comparison(
    *,
    dataset_dir=DEFAULT_DATASET_DIR,
    split_checkpoint_path=DEFAULT_MOTION_SPLIT_CHECKPOINT,
    generator_checkpoint=EXPERIMENTAL_KIMODO_CHECKPOINT,
    semantic_adapter_checkpoint=DEFAULT_SEMANTIC_ADAPTER_CHECKPOINT,
    motion_latent_lora_checkpoint=None,
    output_dir=DEFAULT_OUTPUT_DIR,
    motion_latent_split="test",
    episode_index=None,
    behavior_id=None,
    emotion_id=None,
    count=1,
    device="auto",
    semantic_device=None,
    semantic_local_files_only=True,
    motion_latent_device=None,
    motion_latent_local_files_only=True,
    sampling_steps=32,
    seed=7,
    max_velocity_rad_s=3.0,
    smooth_window=None,
    pane_width=640,
    pane_height=640,
    camera_view="upper",
    simplified=False,
    require_label_match=True,
):
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    behavior_ids = _normalized_behavior_ids(behavior_id)
    dataset_contract = validate_dataset_contract(dataset_dir)
    selected_rows = select_dataset_reference_rows(
        dataset_dir,
        split_checkpoint_path,
        motion_latent_split=motion_latent_split,
        episode_index=episode_index,
        behavior_id=behavior_id,
        emotion_id=emotion_id,
        count=count,
    )
    generator = PtMotionGenerator.from_checkpoint(generator_checkpoint, device=device)
    references, dataset_action_stats = load_reference_trajectories(
        dataset_dir,
        selected_rows,
        generator_checkpoint=generator.checkpoint,
    )
    text_motion_contract = (generator.checkpoint.get("v2_contracts") or {}).get(
        "text_motion_latent"
    )
    motion_latent_lora_provenance = None
    if text_motion_contract is not None:
        if motion_latent_lora_checkpoint is None:
            raise ValueError(
                "this V2 comparison requires the Qwen Motion LoRA checkpoint recorded by the generator"
            )
        condition_builder, lora_checkpoint, condition_source, lora_hash = (
            load_motion_latent_lora_condition_builder(
                generator.checkpoint,
                motion_latent_lora_checkpoint,
                dataset_dir=dataset_dir,
                device=motion_latent_device or device,
                local_files_only=motion_latent_local_files_only,
            )
        )
        motion_latent_lora_provenance = {
            "checkpoint": str(Path(motion_latent_lora_checkpoint).resolve()),
            "checkpoint_sha256": lora_hash,
            "qwen": lora_checkpoint["qwen"],
            "best_step": int(lora_checkpoint["best_step"]),
        }
    else:
        if motion_latent_lora_checkpoint is not None:
            raise ValueError("generator was not trained with Qwen LoRA text-motion latents")
        semantic_adapter, semantic_checkpoint = load_semantic_adapter(
            semantic_adapter_checkpoint,
            device=semantic_device or device,
            local_files_only=semantic_local_files_only,
        )
        condition_bank = semantic_checkpoint.get("condition_bank")
        condition_source = validate_generator_condition_source(generator.checkpoint, condition_bank)
        condition_builder = AdapterConditionBuilder(semantic_adapter, condition_bank=condition_bank)
    dataset_semantic_index = dataset_dir / "meta" / "semantic_index.parquet"
    dataset_semantic_hash = _file_sha256(dataset_semantic_index)
    if dataset_semantic_hash != condition_source["semantic_index_sha256"]:
        raise ValueError(
            "comparison dataset does not match the semantic index recorded by the generator"
        )
    dataset_episode_count = pq.read_table(dataset_semantic_index, columns=["episode_index"]).num_rows
    if dataset_action_stats["dataset_episode_count"] != dataset_episode_count:
        raise ValueError("action parquet episode count does not match the semantic index")
    checkpoint_action_stats = generator.checkpoint.get("action_stats")
    if not isinstance(checkpoint_action_stats, dict):
        raise ValueError("generator checkpoint does not contain action normalization provenance")
    action_stat_errors = {}
    for name in ("mean", "std"):
        recorded = torch.as_tensor(checkpoint_action_stats.get(name), dtype=torch.float32).cpu().numpy()
        observed = np.asarray(dataset_action_stats[name], dtype=np.float32)
        if recorded.shape != (len(JOINT_ORDER),) or observed.shape != recorded.shape:
            raise ValueError(f"generator/dataset action {name} has an invalid shape")
        if not np.isfinite(recorded).all() or not np.isfinite(observed).all():
            raise ValueError(f"generator/dataset action {name} contains non-finite values")
        action_stat_errors[name] = float(np.max(np.abs(recorded - observed)))
        if not np.allclose(recorded, observed, rtol=1e-6, atol=1e-7):
            raise ValueError(
                f"comparison action parquet does not match the generator's recorded {name} statistics"
            )
    action_data_validation = {
        "checkpoint_stats_match": True,
        "episode_count": int(dataset_action_stats["episode_count"]),
        "dataset_episode_count": int(dataset_action_stats["dataset_episode_count"]),
        "frame_count": int(dataset_action_stats["frame_count"]),
        "mean_max_abs_error": action_stat_errors["mean"],
        "std_max_abs_error": action_stat_errors["std"],
        "parquet_manifest": _action_data_manifest(dataset_dir),
        "preprocessing": dataset_action_stats["preprocessing"],
        "provenance_limit": (
            "The generator checkpoint records action statistics but not the original parquet hash; "
            "the manifest is recorded for reproducibility, while the statistics provide the available match check."
        ),
    }
    generator_config = generator.checkpoint.get("config") or {}
    generator_episodes_loaded = generator_config.get("episodes_loaded")
    generator_used_all_episodes = (
        generator_episodes_loaded is not None
        and int(generator_episodes_loaded) == int(dataset_episode_count)
    )
    training_episode_ids = {
        int(value)
        for value in (
            generator.checkpoint.get("training_episode_indices")
            or generator.checkpoint.get("split_episode_indices", {}).get("train")
            or []
        )
    }
    selected_episode_ids = {int(row["episode_index"]) for row in selected_rows}
    overlap_ids = sorted(training_episode_ids & selected_episode_ids)
    held_out = bool(training_episode_ids) and not overlap_ids
    generator_training_coverage = {
        "dataset_episode_count": int(dataset_episode_count),
        "generator_episodes_loaded": (
            None if generator_episodes_loaded is None else int(generator_episodes_loaded)
        ),
        "all_dataset_episodes_used": generator_used_all_episodes,
        "training_episode_indices_recorded": bool(training_episode_ids),
        "reference_training_overlap": overlap_ids,
        "reference_is_generator_held_out": held_out if training_episode_ids else (False if generator_used_all_episodes else None),
        "note": (
            "The current generator trained on every dataset episode; the Motion Metric partition below is "
            "reference selection only, not a held-out MMDiT generalization test."
            if generator_used_all_episodes
            else "Generator episode membership is recorded; held-out status is computed by exact episode ID."
            if training_episode_ids
            else "Generator episode membership is not recorded, so held-out status is unknown."
        ),
    }
    motion_partition = {
        "name": motion_latent_split,
        "source": "motion_metric_encoder_checkpoint",
        "checkpoint": str(Path(split_checkpoint_path).resolve()),
        "checkpoint_sha256": _file_sha256(split_checkpoint_path),
        "purpose": "reference_selection_only",
        "selection_strategy": (
            "exact_episode"
            if episode_index is not None
            else "exact_behavior_emotion"
            if behavior_ids is not None and emotion_id is not None
            else "behavior_set_all_emotions"
            if behavior_ids is not None
            else "diverse_behavior_round_robin"
        ),
    }
    selection_request = {
        "motion_latent_split": motion_latent_split,
        "episode_index": episode_index,
        "behavior_ids": behavior_ids,
        "emotion_id": emotion_id,
        "count": int(count),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_selection_path = output_dir / "dataset_selection.json"
    dataset_selection = {
        "schema_version": 1,
        "source_dataset": str(dataset_dir.resolve()),
        "source_semantic_index_sha256": dataset_semantic_hash,
        "motion_latent_partition": motion_partition,
        "selection_request": selection_request,
        "episodes": [
            {
                "episode_index": int(row["episode_index"]),
                "sample_id": row["sample_id"],
                "behavior_id": row["behavior_id"],
                "emotion_id": row["emotion_id"],
                "language_instruction": row["language_instruction"],
                "frames": int(references[int(row["episode_index"])]["actions"].shape[0]),
            }
            for row in selected_rows
        ],
    }
    _atomic_json_write(dataset_selection, dataset_selection_path)
    results = []
    for row in selected_rows:
        episode_index_value = int(row["episode_index"])
        reference = np.asarray(references[episode_index_value]["actions"], dtype=np.float32)
        reference_style_controls = references[episode_index_value].get("style_controls")
        text = row["language_instruction"]
        motion = generator.infer(
            text,
            frames=reference.shape[0],
            fps=dataset_contract["fps"],
            sampling_steps=int(sampling_steps),
            seed=int(seed) + episode_index_value,
            max_velocity_rad_s=max_velocity_rad_s,
            smooth_window=smooth_window,
            condition_builder=condition_builder,
            style_controls=reference_style_controls,
        )
        behavior_match = motion.behavior_id == row["behavior_id"]
        emotion_match = motion.emotion_id == row["emotion_id"]
        if require_label_match and not (behavior_match and emotion_match):
            raise ValueError(
                f"semantic prediction for episode {episode_index_value} does not match its dataset labels: "
                f"predicted={motion.behavior_id}/{motion.emotion_id} "
                f"expected={row['behavior_id']}/{row['emotion_id']}"
            )

        stem = (
            f"episode_{episode_index_value:04d}__"
            f"{_safe_name(row['behavior_id'].removeprefix('Behavior.'))}__{_safe_name(row['emotion_id'])}"
        )
        video_path = output_dir / f"{stem}__network_left_reference_right.mp4"
        render_summary = render_trajectory_comparison(
            motion.trajectory,
            reference,
            video_path,
            fps=dataset_contract["fps"],
            pane_width=pane_width,
            pane_height=pane_height,
            title_height=COMPARISON_TITLE_HEIGHT,
            simplified=simplified,
            camera_view=camera_view,
        )
        result = {
            "episode_index": episode_index_value,
            "sample_id": row["sample_id"],
            "motion_latent_partition": motion_partition,
            "generator_training_coverage": generator_training_coverage,
            "action_data_validation": action_data_validation,
            "text_source": "dataset.meta.semantic_index.language_instruction",
            "text": text,
            "expected": {
                "behavior_id": row["behavior_id"],
                "emotion_id": row["emotion_id"],
            },
            "predicted": {
                "behavior_id": motion.behavior_id,
                "emotion_id": motion.emotion_id,
                "behavior_confidence": motion.behavior_confidence,
                "emotion_confidence": motion.emotion_confidence,
                "behavior_match": behavior_match,
                "emotion_match": emotion_match,
                "style_controls": None
                if motion.style_controls is None
                else list(motion.style_controls),
            },
            "generator": {
                "checkpoint": str(generator_checkpoint),
                "architecture": generator.info.architecture,
                "configured_steps": generator.info.configured_steps,
                "condition_dim": generator.info.condition_dim,
                "sampling_steps": int(sampling_steps),
                "seed": int(seed) + episode_index_value,
                "uses_cross_modal_lora_checkpoint": motion_latent_lora_provenance is not None,
            },
            "semantic_adapter": None
            if motion_latent_lora_provenance is not None
            else str(semantic_adapter_checkpoint),
            "motion_latent_lora": motion_latent_lora_provenance,
            "condition_source": condition_source,
            "dataset_contract": dataset_contract,
            "trajectory": trajectory_comparison_metrics(
                motion.trajectory,
                reference,
                fps=dataset_contract["fps"],
            ),
            "render": render_summary,
        }
        result["video_check"] = inspect_comparison_video(
            video_path,
            expected_frames=reference.shape[0],
            title_height=COMPARISON_TITLE_HEIGHT,
        )
        _atomic_json_write(result, output_dir / f"{stem}.json")
        results.append(result)
        print(json.dumps({"episode_index": episode_index_value, "video": str(video_path)}, ensure_ascii=False), flush=True)

    index = {
        "schema_version": 1,
        "layout": {"left": "network_output", "right": "dataset_reference"},
        "motion_latent_partition": motion_partition,
        "selection_request": selection_request,
        "dataset_selection_manifest": str(dataset_selection_path),
        "generator_training_coverage": generator_training_coverage,
        "action_data_validation": action_data_validation,
        "generator_scope": (
            "ULA MMDiT V2 with held-out generator splits, variable duration, motion prototypes, and explicit style controls."
            if generator.info.condition_dim != KIMODO_CONDITION_DIM
            else "Legacy 5k 136-dimensional Kimodo MMDiT generator."
        ),
        "results": results,
    }
    _atomic_json_write(index, output_dir / "comparison_index.json")
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Render network output on the left and its exact dataset-label reference on the right in MuJoCo"
    )
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--split-checkpoint", default=str(DEFAULT_MOTION_SPLIT_CHECKPOINT))
    parser.add_argument("--generator-checkpoint", default=str(EXPERIMENTAL_KIMODO_CHECKPOINT))
    parser.add_argument("--semantic-adapter-checkpoint", default=str(DEFAULT_SEMANTIC_ADAPTER_CHECKPOINT))
    parser.add_argument("--motion-latent-lora-checkpoint")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--motion-latent-split",
        choices=("train", "validation", "test"),
        default="test",
        help="Motion Metric Encoder partition used only to select dataset references",
    )
    parser.add_argument("--episode-index", type=int)
    parser.add_argument(
        "--behavior-id",
        action="append",
        help="Dataset behavior filter; repeat to select several behaviors",
    )
    parser.add_argument("--emotion-id")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--semantic-device")
    parser.add_argument("--allow-semantic-download", action="store_true")
    parser.add_argument("--motion-latent-device")
    parser.add_argument("--allow-motion-latent-download", action="store_true")
    parser.add_argument("--sampling-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-velocity-rad-s", type=float, default=3.0)
    parser.add_argument(
        "--smooth-window",
        type=int,
        help="Defaults to the generator checkpoint preprocessing contract",
    )
    parser.add_argument("--pane-width", type=int, default=640)
    parser.add_argument("--pane-height", type=int, default=640)
    parser.add_argument("--camera-view", choices=("front", "upper"), default="upper")
    parser.add_argument("--simplified", action="store_true")
    parser.add_argument("--allow-label-mismatch", action="store_true")
    args = parser.parse_args(argv)
    result = run_dataset_mujoco_comparison(
        dataset_dir=args.dataset_dir,
        split_checkpoint_path=args.split_checkpoint,
        generator_checkpoint=args.generator_checkpoint,
        semantic_adapter_checkpoint=args.semantic_adapter_checkpoint,
        motion_latent_lora_checkpoint=args.motion_latent_lora_checkpoint,
        output_dir=args.output_dir,
        motion_latent_split=args.motion_latent_split,
        episode_index=args.episode_index,
        behavior_id=args.behavior_id,
        emotion_id=args.emotion_id,
        count=args.count,
        device=args.device,
        semantic_device=args.semantic_device,
        semantic_local_files_only=not args.allow_semantic_download,
        motion_latent_device=args.motion_latent_device,
        motion_latent_local_files_only=not args.allow_motion_latent_download,
        sampling_steps=args.sampling_steps,
        seed=args.seed,
        max_velocity_rad_s=args.max_velocity_rad_s,
        smooth_window=args.smooth_window,
        pane_width=args.pane_width,
        pane_height=args.pane_height,
        camera_view=args.camera_view,
        simplified=args.simplified,
        require_label_match=not args.allow_label_mismatch,
    )
    print(
        json.dumps(
            {"output_dir": args.output_dir, "comparisons": len(result["results"])},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
