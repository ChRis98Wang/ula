#!/usr/bin/env python3
"""Leakage-safe, variable-duration training for the Kimodo ULA MMDiT V2 generator."""

from contextlib import contextmanager
from copy import deepcopy
import gc
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from upper_body_skeleton.kimodo_semantics import KIMODO_BEHAVIOR_IDS, KIMODO_EMOTION_IDS
from upper_body_skeleton.motion_latent import build_motion_features, load_motion_metric_checkpoint
from upper_body_skeleton.retarget_v2 import JOINT_ORDER
from upper_body_skeleton.ula_training import (
    KIMODO_V2_CONDITION_DIM,
    ULA_MMDIT_V2_ARCHITECTURE,
    build_kimodo_condition_contract,
    choose_device,
    clip_grad_norm_float64,
    compute_action_normalization_stats,
    condition_vector,
    create_ula_model,
    denormalize_action_tensor,
    normalize_episode_actions,
)
from upper_body_skeleton.ula_v2_conditioning import prepare_v2_episode_splits


V2_TRAINING_SCHEMA_VERSION = 1
KNOWN_OUTPUT_NAMES = {
    "last.pt",
    "last.pt.tmp",
    "progress.jsonl",
    "training_summary.json",
    "ula_fm_checkpoint.pt",
    "ula_fm_checkpoint.pt.tmp",
}
DEFAULT_CONFIG = {
    "steps": 50_000,
    "batch_size": 64,
    "lr": 1e-4,
    "minimum_lr_ratio": 0.1,
    "weight_decay": 1e-4,
    "adam_eps": 1e-6,
    "warmup_steps": 1_000,
    "max_grad_norm": 1.0,
    "architecture": ULA_MMDIT_V2_ARCHITECTURE,
    "hidden_dim": 384,
    "layers": 6,
    "semantic_tokens": 7,
    "device": "auto",
    "seed": 7,
    "ema_decay": 0.999,
    "log_interval": 25,
    "validation_interval": 500,
    "validation_batch_size": 16,
    "checkpoint_interval": 500,
    "phase_frame_choices": [64, 96, 128],
    "mode_balanced_fraction": 0.5,
    "max_velocity_rad_s": 3.0,
    "smooth_window": 1,
    "style_clip": 5.0,
    "overwrite": False,
    "resume_from": None,
    "text_motion_checkpoint": None,
    "text_motion_batch_size": 16,
    "text_motion_local_files_only": True,
    "loss": {
        "flow": 1.0,
        "position": 0.25,
        "velocity": 0.01,
        "acceleration": 0.0005,
        "descriptor": 0.001,
        "motion_latent": 0.1,
        "duration": 0.1,
    },
}


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_state_dict(state):
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def _atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_json_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def resolve_v2_config(config):
    config = dict(config)
    missing = [name for name in ("dataset_dir", "split_checkpoint", "output_dir") if config.get(name) in (None, "")]
    if missing:
        raise ValueError(f"ULA V2 training config requires: {missing}")
    resolved = deepcopy(DEFAULT_CONFIG)
    resolved.update({key: value for key, value in config.items() if key != "loss"})
    resolved["loss"] = dict(DEFAULT_CONFIG["loss"] | dict(config.get("loss") or {}))
    if resolved["architecture"] != ULA_MMDIT_V2_ARCHITECTURE:
        raise ValueError(f"ULA V2 training requires architecture={ULA_MMDIT_V2_ARCHITECTURE}")
    for name in (
        "steps",
        "batch_size",
        "hidden_dim",
        "layers",
        "semantic_tokens",
        "validation_batch_size",
        "text_motion_batch_size",
    ):
        resolved[name] = int(resolved[name])
        if resolved[name] <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("warmup_steps", "log_interval", "validation_interval", "checkpoint_interval"):
        resolved[name] = int(resolved[name])
        if resolved[name] < 0:
            raise ValueError(f"{name} must be non-negative")
    for name in ("lr", "adam_eps", "max_grad_norm", "ema_decay", "minimum_lr_ratio"):
        resolved[name] = float(resolved[name])
        if not math.isfinite(resolved[name]) or resolved[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    resolved["weight_decay"] = float(resolved["weight_decay"])
    if not math.isfinite(resolved["weight_decay"]) or resolved["weight_decay"] < 0:
        raise ValueError("weight_decay must be finite and non-negative")
    if not 0 < resolved["ema_decay"] < 1:
        raise ValueError("ema_decay must be between zero and one")
    if not 0 < resolved["minimum_lr_ratio"] <= 1:
        raise ValueError("minimum_lr_ratio must be in (0, 1]")
    resolved["mode_balanced_fraction"] = float(resolved["mode_balanced_fraction"])
    if not 0 <= resolved["mode_balanced_fraction"] <= 1:
        raise ValueError("mode_balanced_fraction must be in [0, 1]")
    frame_choices = sorted({int(value) for value in resolved["phase_frame_choices"]})
    if not frame_choices or frame_choices[0] < 8:
        raise ValueError("phase_frame_choices must contain frame counts of at least 8")
    resolved["phase_frame_choices"] = frame_choices
    for name, value in resolved["loss"].items():
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"loss.{name} must be finite and non-negative")
        resolved["loss"][name] = value
    resolved["text_motion_checkpoint"] = (
        None
        if resolved["text_motion_checkpoint"] in (None, "")
        else str(resolved["text_motion_checkpoint"])
    )
    resolved["text_motion_local_files_only"] = bool(resolved["text_motion_local_files_only"])
    return resolved


def resample_motion_phase(actions, frame_count):
    values = np.asarray(actions, dtype=np.float32)
    frame_count = int(frame_count)
    if values.ndim != 2 or values.shape[1] != len(JOINT_ORDER) or values.shape[0] < 1:
        raise ValueError(f"actions must have shape [frames, {len(JOINT_ORDER)}]")
    if frame_count < 2:
        raise ValueError("frame_count must be at least 2")
    if values.shape[0] == frame_count:
        return np.ascontiguousarray(values)
    source = np.linspace(0.0, 1.0, values.shape[0], dtype=np.float64)
    target = np.linspace(0.0, 1.0, frame_count, dtype=np.float64)
    output = np.empty((frame_count, values.shape[1]), dtype=np.float32)
    for joint_index in range(values.shape[1]):
        output[:, joint_index] = np.interp(target, source, values[:, joint_index]).astype(np.float32)
    return output


class SemanticModeSampler:
    """Samples semantic groups uniformly and preserves rare within-label motion modes."""

    def __init__(self, episodes, *, seed=7, mode_balanced_fraction=0.5):
        self.rng = np.random.default_rng(int(seed))
        self.mode_balanced_fraction = float(mode_balanced_fraction)
        self.groups = {}
        for episode in episodes:
            meta = episode.get("meta") or {}
            key = (str(meta.get("behavior_id")), str(meta.get("emotion_id")))
            self.groups.setdefault(key, []).append(episode)
        expected = {(behavior, emotion) for behavior in KIMODO_BEHAVIOR_IDS for emotion in KIMODO_EMOTION_IDS}
        if set(self.groups) != expected:
            missing = sorted(expected - set(self.groups))
            raise ValueError(f"train split does not cover every Kimodo semantic group: {missing[:8]}")
        self.keys = sorted(self.groups)
        self.order = self.rng.permutation(len(self.keys)).tolist()
        self.cursor = 0
        self.mode_buckets = {}
        for key, rows in self.groups.items():
            buckets = {}
            amplitudes = np.asarray([float(row["style_controls"][1]) for row in rows], dtype=np.float32)
            low, high = np.quantile(amplitudes, [1.0 / 3.0, 2.0 / 3.0])
            for row in rows:
                balance = float(row["style_controls"][0])
                amplitude = float(row["style_controls"][1])
                side = -1 if balance < -0.25 else (1 if balance > 0.25 else 0)
                scale = -1 if amplitude < low else (1 if amplitude > high else 0)
                buckets.setdefault((side, scale), []).append(row)
            self.mode_buckets[key] = list(buckets.values())

    def _next_key(self):
        if self.cursor >= len(self.order):
            self.order = self.rng.permutation(len(self.keys)).tolist()
            self.cursor = 0
        key = self.keys[self.order[self.cursor]]
        self.cursor += 1
        return key

    def sample(self, batch_size):
        selected = []
        for _ in range(int(batch_size)):
            key = self._next_key()
            if self.rng.random() < self.mode_balanced_fraction:
                buckets = self.mode_buckets[key]
                bucket = buckets[int(self.rng.integers(len(buckets)))]
                rows = bucket
            else:
                rows = self.groups[key]
            selected.append(rows[int(self.rng.integers(len(rows)))])
        return selected

    def state_dict(self):
        return {
            "bit_generator_state": self.rng.bit_generator.state,
            "order": list(self.order),
            "cursor": int(self.cursor),
        }

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state["bit_generator_state"]
        self.order = [int(value) for value in state["order"]]
        self.cursor = int(state["cursor"])


class ModelEMA:
    def __init__(self, model, decay):
        self.decay = float(decay)
        self.shadow = {name: value.detach().clone() for name, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for name, value in model.state_dict().items():
            current = value.detach()
            if torch.is_floating_point(current):
                self.shadow[name].mul_(self.decay).add_(current, alpha=1.0 - self.decay)
            else:
                self.shadow[name].copy_(current)

    @contextmanager
    def apply(self, model):
        backup = {name: value.detach().clone() for name, value in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)
        try:
            yield model
        finally:
            model.load_state_dict(backup, strict=True)


def _episode_sample_span_sec(episode):
    fps = float(episode.get("fps") or 30.0)
    source_frames = int(np.asarray(episode["actions"]).shape[0])
    if not math.isfinite(fps) or fps <= 0.0 or source_frames < 2:
        raise ValueError("planner duration requires at least two frames and positive fps")
    sample_span = float((source_frames - 1) / fps)
    declared = episode.get("duration_sec")
    if declared is not None:
        declared = float(declared)
        if not math.isfinite(declared) or not math.isclose(
            declared, sample_span, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError(
                "duration_sec must equal (frame_count-1)/fps; N/fps frame coverage "
                "cannot be a planner target"
            )
    return sample_span


def _batch_tensors(episodes, *, frame_count, device):
    actions = np.stack([resample_motion_phase(episode["actions"], frame_count) for episode in episodes])
    conditions = np.stack([episode["condition"] for episode in episodes])
    durations = [_episode_sample_span_sec(episode) for episode in episodes]
    duration_mask = [float(episode.get("duration_supervision_valid", True)) for episode in episodes]
    return (
        torch.as_tensor(actions, dtype=torch.float32, device=device),
        torch.as_tensor(conditions, dtype=torch.float32, device=device),
        torch.as_tensor(durations, dtype=torch.float32, device=device),
        torch.as_tensor(duration_mask, dtype=torch.float32, device=device),
    )


def _physical_derivatives(actions, durations):
    frame_count = actions.shape[1]
    dt = (durations / float(max(1, frame_count - 1))).clamp_min(1e-4)[:, None, None]
    velocity = (actions[:, 1:] - actions[:, :-1]) / dt
    acceleration = (velocity[:, 1:] - velocity[:, :-1]) / dt if frame_count > 2 else velocity[:, :0]
    return velocity, acceleration


def motion_descriptor_tensor(actions, durations):
    centered = actions - actions.mean(dim=1, keepdim=True)
    amplitude = torch.sqrt(centered.square().mean(dim=1) + 1e-8)
    velocity, acceleration = _physical_derivatives(actions, durations)
    velocity_rms = torch.sqrt(velocity.square().mean(dim=1) + 1e-8)
    if acceleration.shape[1]:
        acceleration_rms = torch.sqrt(acceleration.square().mean(dim=1) + 1e-8)
    else:
        acceleration_rms = torch.zeros_like(amplitude)
    left = torch.sqrt(velocity[:, :, 3:9].square().mean(dim=(1, 2)) + 1e-8)
    right = torch.sqrt(velocity[:, :, 9:15].square().mean(dim=(1, 2)) + 1e-8)
    balance = ((right - left) / (right + left + 1e-6))[:, None]
    return torch.cat([amplitude, velocity_rms, acceleration_rms, left[:, None], right[:, None], balance], dim=-1)


def _perceptual_embedding(model, actions, normalization, *, reference_frames=150):
    if actions.shape[1] != int(reference_frames):
        actions = F.interpolate(
            actions.transpose(1, 2),
            size=int(reference_frames),
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)
    return model(build_motion_features(actions, normalization))["embedding"]


def flow_matching_v2_objective(
    model,
    actions,
    condition,
    durations,
    *,
    duration_mask=None,
    loss_weights,
    action_stats,
    motion_metric_model=None,
    motion_metric_stats=None,
    generator=None,
):
    if generator is None:
        noise = torch.randn_like(actions)
        t = torch.rand(actions.shape[0], device=actions.device)
    else:
        noise = torch.randn(actions.shape, dtype=actions.dtype, device=actions.device, generator=generator)
        t = torch.rand(actions.shape[0], dtype=actions.dtype, device=actions.device, generator=generator)
    x_t = (1.0 - t[:, None, None]) * noise + t[:, None, None] * actions
    target_velocity_field = actions - noise
    predicted_velocity_field = model(x_t, t, condition)
    reconstructed = x_t + (1.0 - t[:, None, None]) * predicted_velocity_field

    losses = {
        "flow": F.mse_loss(predicted_velocity_field, target_velocity_field),
        "position": F.smooth_l1_loss(reconstructed, actions),
    }
    reconstructed_velocity, reconstructed_acceleration = _physical_derivatives(reconstructed, durations)
    target_velocity, target_acceleration = _physical_derivatives(actions, durations)
    losses["velocity"] = F.smooth_l1_loss(reconstructed_velocity, target_velocity)
    losses["acceleration"] = F.smooth_l1_loss(reconstructed_acceleration, target_acceleration)
    losses["descriptor"] = F.smooth_l1_loss(
        motion_descriptor_tensor(reconstructed, durations),
        motion_descriptor_tensor(actions, durations),
    )
    if motion_metric_model is not None and float(loss_weights.get("motion_latent", 0.0)) > 0:
        reconstructed_physical = denormalize_action_tensor(reconstructed, action_stats)
        actions_physical = denormalize_action_tensor(actions, action_stats)
        predicted_embedding = _perceptual_embedding(
            motion_metric_model,
            reconstructed_physical,
            motion_metric_stats,
        )
        with torch.no_grad():
            target_embedding = _perceptual_embedding(
                motion_metric_model,
                actions_physical,
                motion_metric_stats,
            )
        losses["motion_latent"] = (1.0 - F.cosine_similarity(predicted_embedding, target_embedding)).mean()
    else:
        losses["motion_latent"] = actions.new_zeros(())
    predicted_duration = model.plan_condition(condition)["duration_sec"]
    duration_values = F.smooth_l1_loss(
        torch.log1p(predicted_duration),
        torch.log1p(durations),
        reduction="none",
    )
    if duration_mask is None:
        losses["duration"] = duration_values.mean()
    else:
        duration_mask = duration_mask.to(duration_values.device, duration_values.dtype)
        losses["duration"] = (duration_values * duration_mask).sum() / duration_mask.sum().clamp_min(1.0)
    losses["total"] = sum(float(loss_weights[name]) * losses[name] for name in loss_weights)
    return losses


def _lr_scale(step, *, total_steps, warmup_steps, minimum_ratio):
    if warmup_steps and step <= warmup_steps:
        return max(1e-8, float(step) / float(warmup_steps))
    decay_steps = max(1, int(total_steps) - int(warmup_steps))
    progress = min(1.0, max(0.0, (float(step) - float(warmup_steps)) / float(decay_steps)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(minimum_ratio) + (1.0 - float(minimum_ratio)) * cosine


def evaluate_v2_objective(
    model,
    episodes,
    *,
    batch_size,
    frame_count,
    device,
    loss_weights,
    action_stats,
    motion_metric_model,
    motion_metric_stats,
    seed,
):
    was_training = model.training
    model.eval()
    totals = {name: 0.0 for name in (*loss_weights, "total")}
    count = 0
    try:
        with torch.no_grad():
            for offset in range(0, len(episodes), int(batch_size)):
                rows = episodes[offset : offset + int(batch_size)]
                actions, condition, durations, duration_mask = _batch_tensors(
                    rows, frame_count=frame_count, device=device
                )
                generator = torch.Generator(device=torch.device(device).type).manual_seed(int(seed) + offset)
                losses = flow_matching_v2_objective(
                    model,
                    actions,
                    condition,
                    durations,
                    duration_mask=duration_mask,
                    loss_weights=loss_weights,
                    action_stats=action_stats,
                    motion_metric_model=motion_metric_model,
                    motion_metric_stats=motion_metric_stats,
                    generator=generator,
                )
                for name in totals:
                    totals[name] += float(losses[name].detach().cpu()) * len(rows)
                count += len(rows)
    finally:
        model.train(was_training)
    return {name: value / float(max(1, count)) for name, value in totals.items()}


def _base_condition_contract(episodes, dataset_dir):
    canonical = []
    for episode in episodes:
        item = dict(episode)
        item["condition"] = condition_vector(item.get("meta") or {})
        canonical.append(item)
    contract = build_kimodo_condition_contract(canonical, dataset_dir)
    if contract is None:
        raise ValueError("train split does not provide the complete Kimodo base condition contract")
    return contract


def _checkpoint_payload(
    model_state_dict,
    *,
    model,
    config,
    action_stats,
    contracts,
    base_condition_contract,
    sources,
    global_step,
    best_step,
    best_validation_loss,
    validation_metrics,
    raw_model_state_dict=None,
    optimizer=None,
    sampler=None,
):
    split_ids = contracts["split"]["episode_indices"]
    payload = {
        "schema_version": V2_TRAINING_SCHEMA_VERSION,
        "artifact_kind": "ula_mmdit_v2_generator",
        "model_state_dict": _cpu_state_dict(model_state_dict),
        "joint_order": list(JOINT_ORDER),
        "condition_dim": KIMODO_V2_CONDITION_DIM,
        "base_condition_dim": int(base_condition_contract["condition_dim"]),
        "action_dim": len(JOINT_ORDER),
        "architecture": model.architecture,
        "action_stats": {name: value.detach().cpu().clone() for name, value in action_stats.items()},
        "condition_contract": base_condition_contract,
        "v2_contracts": contracts,
        "split_episode_indices": split_ids,
        "training_episode_indices": list(split_ids["train"]),
        "sources": sources,
        "config": dict(config)
        | {
            "checkpoint_step": int(global_step),
            "checkpoint_loss": float(validation_metrics.get("total", best_validation_loss)),
            "episodes_loaded": len(split_ids["train"]),
        },
        "global_step": int(global_step),
        "best_step": int(best_step),
        "best_validation_loss": float(best_validation_loss),
        "validation_metrics": dict(validation_metrics),
    }
    if raw_model_state_dict is not None and optimizer is not None and sampler is not None:
        payload["training_state"] = {
            "raw_model_state_dict": _cpu_state_dict(raw_model_state_dict),
            "optimizer_state_dict": optimizer.state_dict(),
            "sampler_state_dict": sampler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
    return payload


def _load_resume(path, *, model, ema, optimizer, sampler, config, contracts, sources, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("artifact_kind") != "ula_mmdit_v2_generator":
        raise ValueError("resume checkpoint is not a ULA MMDiT V2 generator")
    if checkpoint.get("v2_contracts", {}).get("sha256") != contracts.get("sha256"):
        raise ValueError("resume checkpoint V2 data/conditioning contracts do not match")
    if checkpoint.get("sources") != sources:
        raise ValueError("resume checkpoint source hashes do not match")
    for field in ("architecture", "hidden_dim", "layers", "semantic_tokens", "batch_size"):
        if checkpoint.get("config", {}).get(field) != config.get(field):
            raise ValueError(f"resume config mismatch for {field}")
    state = checkpoint.get("training_state")
    if not isinstance(state, dict):
        raise ValueError("resume checkpoint does not contain exact training state")
    model.load_state_dict(state["raw_model_state_dict"], strict=True)
    current_state = model.state_dict()
    ema.shadow = {
        name: value.detach().to(current_state[name].device).clone()
        for name, value in checkpoint["model_state_dict"].items()
    }
    optimizer.load_state_dict(state["optimizer_state_dict"])
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if torch.is_tensor(value):
                optimizer_state[key] = value.to(device)
    sampler.load_state_dict(state["sampler_state_dict"])
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state.get("cuda_rng_state_all"):
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    return checkpoint


def train_ula_v2(config):
    config = resolve_v2_config(config)
    output_dir = Path(config["output_dir"])
    resume_from = config.get("resume_from")
    if output_dir.exists() and any(output_dir.iterdir()) and not resume_from:
        if not config.get("overwrite"):
            raise FileExistsError(f"ULA V2 output directory is not empty: {output_dir}")
        for name in KNOWN_OUTPUT_NAMES:
            path = output_dir / name
            if path.is_file():
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(choose_device(str(config["device"])))
    text_motion_encoder = None
    text_motion_source = None
    loaded_text_encoder = None
    if config["text_motion_checkpoint"] is not None:
        from upper_body_skeleton.cross_modal_latent import load_qwen_motion_text_encoder

        loaded_text_encoder, text_motion_checkpoint = load_qwen_motion_text_encoder(
            config["text_motion_checkpoint"],
            device=device,
            local_files_only=config["text_motion_local_files_only"],
        )
        text_motion_encoder = lambda texts: loaded_text_encoder.encode(
            texts, batch_size=config["text_motion_batch_size"]
        )
        text_motion_source = {
            "checkpoint_sha256": _sha256_file(config["text_motion_checkpoint"]),
            "artifact_kind": text_motion_checkpoint["artifact_kind"],
            "global_step": int(text_motion_checkpoint["global_step"]),
            "best_step": int(text_motion_checkpoint["best_step"]),
            "model_name": str(text_motion_checkpoint["qwen"]["model_name"]),
            "revision": str(text_motion_checkpoint["qwen"]["revision"]),
            "latent_dim": int(text_motion_checkpoint["config"]["latent_dim"]),
        }

    # Optional Qwen loading must not change the baseline data/model RNG sequence.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    train_raw, validation_raw, test_raw, contracts = prepare_v2_episode_splits(
        config["dataset_dir"],
        config["split_checkpoint"],
        device=device,
        max_velocity_rad_s=config["max_velocity_rad_s"],
        smooth_window=config["smooth_window"],
        style_clip=config["style_clip"],
        text_motion_encoder=text_motion_encoder,
        text_motion_source=text_motion_source,
    )
    if loaded_text_encoder is not None:
        del text_motion_encoder
        del loaded_text_encoder
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    action_stats = compute_action_normalization_stats(train_raw)
    train_episodes = normalize_episode_actions(train_raw, action_stats)
    validation_episodes = normalize_episode_actions(validation_raw, action_stats)
    test_episodes = normalize_episode_actions(test_raw, action_stats)
    model = create_ula_model(
        config["architecture"],
        action_dim=len(JOINT_ORDER),
        condition_dim=KIMODO_V2_CONDITION_DIM,
        hidden_dim=config["hidden_dim"],
        layers=config["layers"],
        semantic_tokens=config["semantic_tokens"],
    ).to(device)
    model.action_stats = action_stats
    motion_metric_model, motion_metric_stats, _ = load_motion_metric_checkpoint(
        config["split_checkpoint"], device=device
    )
    motion_metric_model.requires_grad_(False).eval()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
        eps=float(config["adam_eps"]),
    )
    sampler = SemanticModeSampler(
        train_episodes,
        seed=seed,
        mode_balanced_fraction=config["mode_balanced_fraction"],
    )
    ema = ModelEMA(model, config["ema_decay"])
    dataset_semantic_path = Path(config["dataset_dir"]) / "meta" / "semantic_index.parquet"
    sources = {
        "semantic_index_sha256": _sha256_file(dataset_semantic_path),
        "split_checkpoint_sha256": _sha256_file(config["split_checkpoint"]),
    }
    if config["text_motion_checkpoint"] is not None:
        sources["text_motion_checkpoint_sha256"] = _sha256_file(config["text_motion_checkpoint"])
    base_contract = _base_condition_contract(train_raw, config["dataset_dir"])
    global_step = 0
    best_step = 0
    best_validation_loss = float("inf")
    validation_metrics = {}
    if resume_from:
        checkpoint = _load_resume(
            resume_from,
            model=model,
            ema=ema,
            optimizer=optimizer,
            sampler=sampler,
            config=config,
            contracts=contracts,
            sources=sources,
            device=device,
        )
        global_step = int(checkpoint["global_step"])
        best_step = int(checkpoint["best_step"])
        best_validation_loss = float(checkpoint["best_validation_loss"])
        validation_metrics = dict(checkpoint.get("validation_metrics") or {})
    else:
        (output_dir / "progress.jsonl").write_text("", encoding="utf-8")
    if global_step >= int(config["steps"]):
        raise ValueError("target steps must be greater than the resumed global step")

    print(
        json.dumps(
            {
                "device": str(device),
                "architecture": model.architecture,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "start_step": global_step,
                "target_steps": config["steps"],
                "split_counts": {
                    "train": len(train_episodes),
                    "validation": len(validation_episodes),
                    "test": len(test_episodes),
                },
                "phase_frame_choices": config["phase_frame_choices"],
                "v2_contract_sha256": contracts["sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    progress_path = output_dir / "progress.jsonl"
    frame_rng = np.random.default_rng(seed + 104729)
    if resume_from:
        phase_rng_state = checkpoint.get("training_state", {}).get("phase_rng_state")
        if phase_rng_state is None:
            raise ValueError("resume checkpoint does not contain phase-frame RNG state")
        frame_rng.bit_generator.state = phase_rng_state
    model.train()
    last_train_losses = {}
    for step in range(global_step + 1, int(config["steps"]) + 1):
        scale = _lr_scale(
            step,
            total_steps=config["steps"],
            warmup_steps=config["warmup_steps"],
            minimum_ratio=config["minimum_lr_ratio"],
        )
        current_lr = float(config["lr"]) * scale
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        selected = sampler.sample(config["batch_size"])
        frame_count = int(frame_rng.choice(config["phase_frame_choices"]))
        actions, condition, durations, duration_mask = _batch_tensors(
            selected, frame_count=frame_count, device=device
        )
        losses = flow_matching_v2_objective(
            model,
            actions,
            condition,
            durations,
            duration_mask=duration_mask,
            loss_weights=config["loss"],
            action_stats=action_stats,
            motion_metric_model=motion_metric_model,
            motion_metric_stats=motion_metric_stats,
        )
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(f"non-finite ULA V2 loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        grad_norm = clip_grad_norm_float64(model.parameters(), config["max_grad_norm"])
        optimizer.step()
        ema.update(model)
        global_step = step
        last_train_losses = {name: float(value.detach().cpu()) for name, value in losses.items()}

        should_validate = step == 1 or step % int(config["validation_interval"]) == 0 or step == int(config["steps"])
        should_checkpoint = step == 1 or step % int(config["checkpoint_interval"]) == 0 or step == int(config["steps"])
        is_best = False
        if should_validate:
            with ema.apply(model):
                validation_metrics = evaluate_v2_objective(
                    model,
                    validation_episodes,
                    batch_size=config["validation_batch_size"],
                    frame_count=max(config["phase_frame_choices"]),
                    device=device,
                    loss_weights=config["loss"],
                    action_stats=action_stats,
                    motion_metric_model=motion_metric_model,
                    motion_metric_stats=motion_metric_stats,
                    seed=seed + 1_000_003,
                )
            if validation_metrics["total"] < best_validation_loss:
                best_validation_loss = float(validation_metrics["total"])
                best_step = step
                is_best = True

        event = {
            "step": step,
            "steps": config["steps"],
            "lr": current_lr,
            "phase_frames": frame_count,
            "grad_norm": grad_norm,
            "train": last_train_losses,
        }
        if should_validate:
            event["validation"] = validation_metrics
            event["is_best"] = is_best
        if step == 1 or step % int(config["log_interval"]) == 0 or should_validate:
            line = json.dumps(event, sort_keys=True)
            print(line, flush=True)
            with progress_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")

        if is_best:
            payload = _checkpoint_payload(
                ema.shadow,
                model=model,
                config=config,
                action_stats=action_stats,
                contracts=contracts,
                base_condition_contract=base_contract,
                sources=sources,
                global_step=step,
                best_step=best_step,
                best_validation_loss=best_validation_loss,
                validation_metrics=validation_metrics,
            )
            _atomic_torch_save(payload, output_dir / "ula_fm_checkpoint.pt")
        if should_checkpoint:
            payload = _checkpoint_payload(
                ema.shadow,
                model=model,
                config=config,
                action_stats=action_stats,
                contracts=contracts,
                base_condition_contract=base_contract,
                sources=sources,
                global_step=step,
                best_step=best_step,
                best_validation_loss=best_validation_loss,
                validation_metrics=validation_metrics,
                raw_model_state_dict=model.state_dict(),
                optimizer=optimizer,
                sampler=sampler,
            )
            payload["training_state"]["phase_rng_state"] = frame_rng.bit_generator.state
            _atomic_torch_save(payload, output_dir / "last.pt")

    best_checkpoint = torch.load(output_dir / "ula_fm_checkpoint.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    test_metrics = evaluate_v2_objective(
        model,
        test_episodes,
        batch_size=config["validation_batch_size"],
        frame_count=max(config["phase_frame_choices"]),
        device=device,
        loss_weights=config["loss"],
        action_stats=action_stats,
        motion_metric_model=motion_metric_model,
        motion_metric_stats=motion_metric_stats,
        seed=seed + 2_000_003,
    )
    summary = {
        "output_dir": str(output_dir),
        "steps": int(config["steps"]),
        "best_step": int(best_step),
        "best_validation_loss": float(best_validation_loss),
        "best_validation": dict(best_checkpoint.get("validation_metrics") or {}),
        "final_validation": validation_metrics,
        "test": test_metrics,
        "last_train": last_train_losses,
        "sources": sources,
        "v2_contract_sha256": contracts["sha256"],
    }
    _atomic_json_save(summary, output_dir / "training_summary.json")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


__all__ = [
    "DEFAULT_CONFIG",
    "ModelEMA",
    "SemanticModeSampler",
    "evaluate_v2_objective",
    "flow_matching_v2_objective",
    "motion_descriptor_tensor",
    "resample_motion_phase",
    "resolve_v2_config",
    "train_ula_v2",
]
