#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from upper_body_skeleton.kimodo_semantics import (
    KIMODO_BEHAVIOR_IDS,
    KIMODO_CONDITION_CONTRACT_VERSION,
    KIMODO_CONDITION_SCHEMA_VERSION,
    KIMODO_EMOTION_IDS,
    kimodo_condition_metadata,
)
from upper_body_skeleton.long_emotion_infer import postprocess_trajectory, trajectory_quality
from upper_body_skeleton.mujoco_playback import MujocoMotionPlayer
from upper_body_skeleton.retarget_v2 import JOINT_ORDER
from upper_body_skeleton.ula_infer import model_from_checkpoint
from upper_body_skeleton.ula_training import (
    KIMODO_CONDITION_DIM,
    KIMODO_V2_CONDITION_DIM,
    ULA_ADALN_LITE_ARCHITECTURE,
    ULA_FM_LEGACY_ARCHITECTURE,
    ULA_MMDIT_LITE_ARCHITECTURE,
    ULA_MMDIT_V2_ARCHITECTURE,
    ULA_MMDIT_V3_ADALN_ARCHITECTURE,
    build_condition_from_text,
    choose_device,
    frame_count_to_coverage,
    frame_count_to_sample_span,
    sample_trajectory,
    sample_span_to_frame_count,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTAL_KIMODO_CHECKPOINT = (
    REPO_ROOT
    / "training"
    / "runs"
    / "kimodo_mmdit_lite_qwen_compatible_5k_math_sdp"
    / "ula_fm_checkpoint.pt"
)
DEFAULT_CHECKPOINT = EXPERIMENTAL_KIMODO_CHECKPOINT
DEFAULT_SEMANTIC_ADAPTER_CHECKPOINT = (
    REPO_ROOT
    / "training"
    / "runs"
    / "kimodo_qwen3_semantic_adapter_deploy_v1"
    / "semantic_adapter_checkpoint.pt"
)
GENERATOR_ARCHITECTURES = {
    ULA_FM_LEGACY_ARCHITECTURE,
    ULA_MMDIT_LITE_ARCHITECTURE,
    ULA_ADALN_LITE_ARCHITECTURE,
    ULA_MMDIT_V2_ARCHITECTURE,
    ULA_MMDIT_V3_ADALN_ARCHITECTURE,
}
LEGACY_CONDITION_DIM = 92
SUPPORTED_CONDITION_DIMS = {LEGACY_CONDITION_DIM, KIMODO_CONDITION_DIM, KIMODO_V2_CONDITION_DIM}


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class GeneratorCheckpointInfo:
    path: str
    architecture: str
    action_dim: int
    condition_dim: int
    hidden_dim: int
    layers: int
    configured_steps: int | None
    checkpoint_step: int | None
    episodes_loaded: int | None
    parameter_count: int
    has_action_stats: bool


@dataclass(frozen=True)
class GeneratedMotion:
    text: str
    behavior_id: str | None
    emotion_id: str | None
    trajectory: np.ndarray
    raw_trajectory: np.ndarray
    fps: float
    sampling_steps: int
    seed: int | None
    behavior_confidence: float | None = None
    emotion_confidence: float | None = None
    predicted_duration_sec: float | None = None
    style_controls: tuple[float, float, float] | None = None

    def summary(self):
        frame_count = int(self.trajectory.shape[0])
        sample_span = frame_count_to_sample_span(frame_count, self.fps)
        return {
            "text": self.text,
            "behavior_id": self.behavior_id,
            "emotion_id": self.emotion_id,
            "behavior_confidence": self.behavior_confidence,
            "emotion_confidence": self.emotion_confidence,
            "frames": frame_count,
            "action_dim": int(self.trajectory.shape[1]),
            "fps": float(self.fps),
            "duration_sec": sample_span,
            "sample_span_sec": sample_span,
            "frame_coverage_sec": frame_count_to_coverage(frame_count, self.fps),
            "duration_contract": "sample_span=(frames-1)/fps",
            "duration_quantization_error_sec": (
                None
                if self.predicted_duration_sec is None
                else sample_span - float(self.predicted_duration_sec)
            ),
            "sampling_steps": int(self.sampling_steps),
            "seed": self.seed,
            "predicted_duration_sec": self.predicted_duration_sec,
            "style_controls": None if self.style_controls is None else list(self.style_controls),
            "trajectory_quality": {
                "raw": trajectory_quality(self.raw_trajectory, fps=self.fps),
                "processed": trajectory_quality(self.trajectory, fps=self.fps),
            },
        }


def _checkpoint_architecture(checkpoint):
    return str(checkpoint.get("architecture") or ULA_FM_LEGACY_ARCHITECTURE)


def validate_generator_condition_contract(contract, *, path="<memory>"):
    if not isinstance(contract, dict):
        raise ValueError(f"136-dimensional generator checkpoint has no condition contract: {path}")
    expected = {
        "contract_version": KIMODO_CONDITION_CONTRACT_VERSION,
        "condition_schema_version": KIMODO_CONDITION_SCHEMA_VERSION,
        "condition_dim": KIMODO_CONDITION_DIM,
        "behavior_ids": list(KIMODO_BEHAVIOR_IDS),
        "emotion_ids": list(KIMODO_EMOTION_IDS),
    }
    for field, value in expected.items():
        if contract.get(field) != value:
            raise ValueError(f"generator condition contract {field} mismatch: {path}")
    for field in ("source_semantic_index_sha256", "canonical_vectors_sha256"):
        digest = contract.get(field)
        if not isinstance(digest, str) or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest.lower()
        ):
            raise ValueError(f"generator condition contract {field} is invalid: {path}")
    return contract


def validate_v2_generator_contracts(checkpoint, *, path="<memory>"):
    contracts = checkpoint.get("v2_contracts")
    required = {
        "split",
        "preprocess",
        "active_window",
        "duration",
        "style",
        "style_bank",
        "motion_prototypes",
        "condition",
        "sha256",
    }
    if not isinstance(contracts, dict) or not required.issubset(contracts):
        raise ValueError(f"ULA MMDiT V2 checkpoint has incomplete conditioning contracts: {path}")
    condition = contracts["condition"]
    if int(condition.get("condition_dim", -1)) != KIMODO_V2_CONDITION_DIM:
        raise ValueError(f"ULA MMDiT V2 condition contract dimension mismatch: {path}")
    if int(condition.get("base_condition_dim", -1)) != KIMODO_CONDITION_DIM:
        raise ValueError(f"ULA MMDiT V2 base condition dimension mismatch: {path}")
    prototypes = contracts["motion_prototypes"]
    if int(prototypes.get("latent_dim", -1)) != KIMODO_V2_CONDITION_DIM - KIMODO_CONDITION_DIM:
        raise ValueError(f"ULA MMDiT V2 motion prototype dimension mismatch: {path}")
    expected_groups = len(KIMODO_BEHAVIOR_IDS) * len(KIMODO_EMOTION_IDS)
    if len(prototypes.get("groups") or []) != expected_groups:
        raise ValueError(f"ULA MMDiT V2 prototype bank must cover all semantic groups: {path}")
    if len(contracts["style_bank"].get("groups") or []) != expected_groups:
        raise ValueError(f"ULA MMDiT V2 style bank must cover all semantic groups: {path}")
    for name in required - {"sha256"}:
        digest = contracts[name].get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"ULA MMDiT V2 {name} contract hash is invalid: {path}")
    text_motion = contracts.get("text_motion_latent")
    if text_motion is not None:
        if int(text_motion.get("latent_dim", -1)) != KIMODO_V2_CONDITION_DIM - KIMODO_CONDITION_DIM:
            raise ValueError(f"ULA MMDiT V2 text-motion latent dimension mismatch: {path}")
        if text_motion.get("contract_type") != "ula_v2_qwen_lora_text_motion_latent":
            raise ValueError(f"ULA MMDiT V2 text-motion latent contract type mismatch: {path}")
        for field in ("sha256",):
            digest = text_motion.get(field)
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"ULA MMDiT V2 text-motion latent {field} is invalid: {path}")
        source_digest = (text_motion.get("source") or {}).get("checkpoint_sha256")
        if not isinstance(source_digest, str) or len(source_digest) != 64:
            raise ValueError(f"ULA MMDiT V2 text-motion checkpoint hash is invalid: {path}")
    return contracts


def validate_generator_checkpoint(checkpoint, *, path="<memory>"):
    if not isinstance(checkpoint, dict):
        raise ValueError(f"generator checkpoint must contain a dictionary: {path}")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"generator checkpoint has no model_state_dict: {path}")

    state_roots = {str(name).split(".", 1)[0] for name in state}
    latent_encoder_roots = {"backbone", "projection", "behavior_head", "emotion_head", "descriptor_head"}
    if {"backbone", "projection"}.issubset(state_roots) and state_roots <= latent_encoder_roots:
        raise ValueError(
            "motion-latent checkpoints only encode existing trajectories and cannot generate MuJoCo motion; "
            "use a ula_fm/ula_mmdit/ula_adaln generator checkpoint"
        )

    architecture = _checkpoint_architecture(checkpoint)
    if architecture not in GENERATOR_ARCHITECTURES:
        raise ValueError(f"unsupported ULA generator architecture {architecture!r}: {path}")
    action_dim = int(checkpoint.get("action_dim", len(JOINT_ORDER)))
    if action_dim != len(JOINT_ORDER):
        raise ValueError(
            f"legacy pt_mujoco_infer is a strict {len(JOINT_ORDER)}D path and will not truncate "
            f"a {action_dim}D checkpoint; use tools/train_ula_v2_18d_head.py infer for the "
            f"versioned 18D contract: {path}"
        )
    condition_dim = int(checkpoint.get("condition_dim", LEGACY_CONDITION_DIM))
    if condition_dim not in SUPPORTED_CONDITION_DIMS:
        raise ValueError(
            f"checkpoint condition_dim must be one of {sorted(SUPPORTED_CONDITION_DIMS)}, got {condition_dim}: {path}"
        )
    if condition_dim in {KIMODO_CONDITION_DIM, KIMODO_V2_CONDITION_DIM}:
        validate_generator_condition_contract(checkpoint.get("condition_contract"), path=path)
    if condition_dim == KIMODO_V2_CONDITION_DIM:
        validate_v2_generator_contracts(checkpoint, path=path)
    checkpoint_joint_order = checkpoint.get("joint_order")
    if checkpoint_joint_order is None:
        raise ValueError(f"generator checkpoint does not declare joint_order: {path}")
    if list(checkpoint_joint_order) != JOINT_ORDER:
        raise ValueError(f"checkpoint joint_order does not match the MuJoCo V2 joint order: {path}")

    parameter_count = 0
    for name, value in state.items():
        if not torch.is_tensor(value):
            continue
        parameter_count += int(value.numel())
        if not torch.isfinite(value).all():
            raise ValueError(f"checkpoint contains non-finite parameter values at {name}: {path}")
    action_stats = checkpoint.get("action_stats")
    if action_stats is not None:
        if not isinstance(action_stats, dict) or set(action_stats) != {"mean", "std"}:
            raise ValueError(f"checkpoint action_stats must contain exactly mean and std: {path}")
        for name, value in action_stats.items():
            try:
                tensor = torch.as_tensor(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"checkpoint action_stats.{name} is not numeric: {path}") from exc
            if tensor.shape != (action_dim,):
                raise ValueError(
                    f"checkpoint action_stats.{name} must have shape ({action_dim},), got {tuple(tensor.shape)}: {path}"
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(f"checkpoint contains non-finite action_stats values at {name}: {path}")
            if name == "std" and not (tensor > 0).all():
                raise ValueError(f"checkpoint action_stats.std must be strictly positive: {path}")

    config = checkpoint.get("config", {}) or {}
    return GeneratorCheckpointInfo(
        path=str(path),
        architecture=architecture,
        action_dim=action_dim,
        condition_dim=condition_dim,
        hidden_dim=int(config.get("hidden_dim", 256)),
        layers=int(config.get("layers", 4)),
        configured_steps=None if config.get("steps") is None else int(config["steps"]),
        checkpoint_step=None if config.get("checkpoint_step") is None else int(config["checkpoint_step"]),
        episodes_loaded=None if config.get("episodes_loaded") is None else int(config["episodes_loaded"]),
        parameter_count=parameter_count,
        has_action_stats=checkpoint.get("action_stats") is not None,
    )


def inspect_generator_checkpoint(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"generator checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return validate_generator_checkpoint(checkpoint, path=path), checkpoint


def validate_generator_condition_source(checkpoint, condition_bank, *, repo_root=REPO_ROOT):
    from upper_body_skeleton.semantic_adapter import validate_condition_bank

    validate_condition_bank(condition_bank)
    contract = validate_generator_condition_contract(checkpoint.get("condition_contract"))
    for field in (
        "contract_version",
        "condition_schema_version",
        "condition_dim",
        "behavior_ids",
        "emotion_ids",
        "source_semantic_index_sha256",
        "canonical_vectors_sha256",
    ):
        if condition_bank.get(field) != contract.get(field):
            raise ValueError(f"semantic adapter condition bank does not match generator {field}")
    expected_hash = str(contract["source_semantic_index_sha256"])
    dataset_dir = (checkpoint.get("config") or {}).get("dataset_dir")
    if not dataset_dir:
        raise ValueError("generator checkpoint does not record the dataset used for its 136-dimensional condition")
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = Path(repo_root) / dataset_dir
    semantic_path = dataset_dir / "meta" / "semantic_index.parquet"
    if not semantic_path.is_file():
        raise ValueError(f"generator condition semantic index is unavailable: {semantic_path}")
    actual_hash = hashlib.sha256(semantic_path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError("semantic adapter condition bank does not match the generator training dataset")
    return {
        "dataset_dir": str(dataset_dir),
        "semantic_index": str(semantic_path),
        "semantic_index_sha256": actual_hash,
        "canonical_vectors_sha256": contract["canonical_vectors_sha256"],
    }


def load_motion_latent_lora_condition_builder(
    generator_checkpoint,
    lora_checkpoint,
    *,
    dataset_dir=None,
    device="auto",
    local_files_only=True,
):
    contracts = validate_v2_generator_contracts(generator_checkpoint)
    text_motion_contract = contracts.get("text_motion_latent")
    if text_motion_contract is None:
        raise ValueError("generator was not trained with Qwen LoRA text-motion latents")
    expected_hash = text_motion_contract["source"]["checkpoint_sha256"]
    actual_hash = _sha256_file(lora_checkpoint)
    if actual_hash != expected_hash:
        raise ValueError("Qwen Motion LoRA checkpoint hash does not match the generator training contract")

    from upper_body_skeleton.cross_modal_latent import (
        LoRAMotionConditionBuilder,
        load_qwen_motion_text_encoder,
    )
    from upper_body_skeleton.semantic_adapter import load_kimodo_condition_bank

    if dataset_dir is None:
        dataset_dir = Path(generator_checkpoint.get("config", {}).get("dataset_dir") or "")
        if not dataset_dir.is_absolute():
            dataset_dir = REPO_ROOT / dataset_dir
    condition_bank = load_kimodo_condition_bank(dataset_dir)
    condition_source = validate_generator_condition_source(generator_checkpoint, condition_bank)
    text_encoder, text_checkpoint = load_qwen_motion_text_encoder(
        lora_checkpoint,
        device=device,
        local_files_only=local_files_only,
    )
    builder = LoRAMotionConditionBuilder(text_encoder, condition_bank=condition_bank)
    return builder, text_checkpoint, condition_source, actual_hash


def _text_style_overrides(text, controls):
    controls = np.asarray(controls, dtype=np.float32).copy()
    lowered = str(text).lower()
    if any(token in lowered for token in ("左手", "left hand", "left arm")):
        controls[0] = min(float(controls[0]), -1.0)
    elif any(token in lowered for token in ("右手", "right hand", "right arm")):
        controls[0] = max(float(controls[0]), 1.0)
    if any(token in lowered for token in ("大幅", "large", "big motion", "wide")):
        controls[1] = max(float(controls[1]), 1.0)
    elif any(token in lowered for token in ("小幅", "small", "subtle", "轻微")):
        controls[1] = min(float(controls[1]), -1.0)
    if any(token in lowered for token in ("快速", "快地", "fast", "quickly")):
        controls[2] = max(float(controls[2]), 1.0)
    elif any(token in lowered for token in ("缓慢", "慢慢", "slow", "slowly")):
        controls[2] = min(float(controls[2]), -1.0)
    return controls


def _assemble_checkpoint_v2_condition(
    checkpoint,
    base_condition,
    *,
    behavior_id,
    emotion_id,
    text,
    seed,
    style_controls=None,
    style_policy="sample",
    motion_latent=None,
):
    from upper_body_skeleton.ula_v2_conditioning import (
        assemble_v2_condition,
        style_controls_for_semantic,
    )

    contracts = validate_v2_generator_contracts(checkpoint)
    if contracts.get("text_motion_latent") is not None and motion_latent is None:
        raise ValueError(
            "this ULA V2 generator requires a Qwen LoRA text-motion latent; "
            "load its recorded motion-latent LoRA checkpoint"
        )
    if style_controls is None:
        if style_policy not in {"sample", "mean"}:
            raise ValueError("style_policy must be sample or mean")
        controls = style_controls_for_semantic(
            contracts["style_bank"],
            behavior_id,
            emotion_id,
            seed=seed,
            mean=style_policy == "mean",
        )
        controls = _text_style_overrides(text, controls)
    else:
        controls = np.asarray(style_controls, dtype=np.float32)
    condition = assemble_v2_condition(
        base_condition,
        behavior_id=behavior_id,
        emotion_id=emotion_id,
        prototype_contract=contracts["motion_prototypes"],
        style_controls=controls,
        motion_latent=motion_latent,
    )
    return condition, tuple(float(value) for value in controls)


def _predict_v2_duration(model, checkpoint, condition, *, device):
    tensor = torch.as_tensor(condition, dtype=torch.float32, device=device)[None, :]
    with torch.no_grad():
        predicted = float(model.plan_condition(tensor)["duration_sec"][0].detach().cpu())
    duration_contract = (checkpoint.get("v2_contracts") or {}).get("duration")
    if not isinstance(duration_contract, dict):
        raise ValueError(
            "V2 checkpoint has no learned variable-duration contract; "
            "implicit fixed-duration generation is disabled"
        )
    if duration_contract.get("fixed_frame_count") is not None or duration_contract.get(
        "fixed_duration_sec"
    ) is not None:
        raise ValueError("V2 inference rejects fixed-frame or fixed-duration contracts")
    representation = str(duration_contract.get("trajectory_representation") or "")
    if "fixed" in representation.lower():
        raise ValueError("V2 inference rejects a fixed-length trajectory representation")
    bounds = duration_contract.get("duration_supervision_sec") or duration_contract.get(
        "train_duration_sec"
    )
    if not isinstance(bounds, dict):
        raise ValueError("V2 variable-duration contract has no train-derived bounds")
    lower = float(bounds["min"])
    upper = float(bounds["max"])
    if not math.isfinite(lower) or not math.isfinite(upper) or lower <= 0.0 or upper < lower:
        raise ValueError("V2 variable-duration contract has invalid train-derived bounds")
    if not math.isfinite(predicted):
        raise FloatingPointError("duration head returned a non-finite value")
    return max(lower, min(upper, predicted))


def infer_motion(
    model,
    checkpoint,
    *,
    text,
    behavior_id=None,
    emotion_id=None,
    frames=None,
    fps=30.0,
    sampling_steps=32,
    device="cpu",
    seed=7,
    max_velocity_rad_s=3.0,
    smooth_window=None,
    style_controls=None,
    style_policy="sample",
    motion_latent=None,
    condition_builder=build_condition_from_text,
    sampler=sample_trajectory,
    postprocessor=postprocess_trajectory,
):
    text = str(text).strip()
    if not text:
        raise ValueError("inference text must not be empty")
    if frames is not None:
        frames = int(frames)
        if frames < 2:
            raise ValueError("frames must be at least 2")
    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    sampling_steps = int(sampling_steps)
    if sampling_steps <= 0:
        raise ValueError("sampling_steps must be positive")
    max_velocity_rad_s = float(max_velocity_rad_s)
    if not math.isfinite(max_velocity_rad_s) or max_velocity_rad_s <= 0:
        raise ValueError("max_velocity_rad_s must be finite and positive")
    condition_dim = int(checkpoint.get("condition_dim", LEGACY_CONDITION_DIM))
    if smooth_window is None:
        smooth_window = (
            int(checkpoint.get("v2_contracts", {}).get("preprocess", {}).get("smooth_window", 1))
            if condition_dim == KIMODO_V2_CONDITION_DIM
            else 5
        )
    smooth_window = int(smooth_window)
    if smooth_window < 1:
        raise ValueError("smooth_window must be at least 1")

    if condition_dim == LEGACY_CONDITION_DIM and (behavior_id is not None or emotion_id is not None):
        raise ValueError(
            "legacy 92-dimensional ULA checkpoints do not support structured behavior_id/emotion_id; "
            "put the requested behavior and emotion in --text or use a Kimodo generator checkpoint"
        )
    builder_condition_dim = int(checkpoint.get("base_condition_dim", condition_dim))
    condition = condition_builder(
        text,
        behavior_id=behavior_id,
        emotion_id=emotion_id,
        condition_dim=builder_condition_dim,
    )
    semantic_prediction = getattr(condition_builder, "last_prediction", None)
    if motion_latent is None:
        motion_latent = getattr(condition_builder, "last_motion_latent", None)
    resolved_behavior_id = getattr(semantic_prediction, "behavior_id", behavior_id)
    resolved_emotion_id = getattr(semantic_prediction, "emotion_id", emotion_id)
    behavior_confidence = getattr(semantic_prediction, "behavior_confidence", None)
    emotion_confidence = getattr(semantic_prediction, "emotion_confidence", None)
    if condition_dim == KIMODO_V2_CONDITION_DIM and (resolved_behavior_id is None or resolved_emotion_id is None):
        inferred = kimodo_condition_metadata(behavior_id=behavior_id, emotion_id=emotion_id, text=text)
        resolved_behavior_id = inferred["behavior_id"]
        resolved_emotion_id = inferred["emotion_id"]
    condition = np.asarray(condition, dtype=np.float32)
    resolved_style_controls = None
    if condition_dim == KIMODO_V2_CONDITION_DIM:
        if condition.shape != (builder_condition_dim,):
            raise ValueError(
                f"base condition shape mismatch: built {condition.shape}, checkpoint expects ({builder_condition_dim},)"
            )
        condition, resolved_style_controls = _assemble_checkpoint_v2_condition(
            checkpoint,
            condition,
            behavior_id=resolved_behavior_id,
            emotion_id=resolved_emotion_id,
            text=text,
            seed=seed,
            style_controls=style_controls,
            style_policy=style_policy,
            motion_latent=motion_latent,
        )
    elif condition.shape != (condition_dim,):
        raise ValueError(f"condition shape mismatch: built {condition.shape}, checkpoint expects ({condition_dim},)")

    predicted_duration_sec = None
    if condition_dim == KIMODO_V2_CONDITION_DIM:
        predicted_duration_sec = _predict_v2_duration(model, checkpoint, condition, device=device)
    if frames is None:
        if predicted_duration_sec is None:
            raise ValueError(
                "frames must be explicit for checkpoints without a trained duration "
                "head; implicit fixed-duration generation is disabled"
            )
        frames = sample_span_to_frame_count(predicted_duration_sec, fps)

    raw = sampler(
        model,
        condition=condition,
        frames=frames,
        action_dim=int(checkpoint.get("action_dim", len(JOINT_ORDER))),
        steps=sampling_steps,
        device=device,
        seed=seed,
        action_stats=getattr(model, "action_stats", None),
    )
    raw = np.asarray(raw, dtype=np.float32)
    expected_shape = (frames, int(checkpoint.get("action_dim", len(JOINT_ORDER))))
    if raw.shape != expected_shape:
        raise ValueError(f"generator returned trajectory shape {raw.shape}, expected {expected_shape}")
    if not np.isfinite(raw).all():
        raise FloatingPointError("generator returned a non-finite trajectory")

    trajectory = postprocessor(
        raw,
        fps=fps,
        max_velocity_rad_s=max_velocity_rad_s,
        smooth_window=smooth_window,
    )
    trajectory = np.asarray(trajectory, dtype=np.float32)
    if trajectory.shape != expected_shape or not np.isfinite(trajectory).all():
        raise FloatingPointError("postprocessing returned an invalid trajectory")
    return GeneratedMotion(
        text=text,
        behavior_id=resolved_behavior_id,
        emotion_id=resolved_emotion_id,
        trajectory=trajectory,
        raw_trajectory=raw,
        fps=fps,
        sampling_steps=sampling_steps,
        seed=seed,
        behavior_confidence=behavior_confidence,
        emotion_confidence=emotion_confidence,
        predicted_duration_sec=predicted_duration_sec,
        style_controls=resolved_style_controls,
    )


class PtMotionGenerator:
    def __init__(self, model, checkpoint, info, *, device):
        self.model = model
        self.checkpoint = checkpoint
        self.info = info
        self.device = str(device)

    @classmethod
    def from_checkpoint(cls, checkpoint_path, *, device="auto"):
        info, checkpoint = inspect_generator_checkpoint(checkpoint_path)
        resolved_device = choose_device(device)
        model, _ = model_from_checkpoint(checkpoint, resolved_device, strict=True)
        return cls(model, checkpoint, info, device=resolved_device)

    def infer(self, text, **kwargs):
        return infer_motion(
            self.model,
            self.checkpoint,
            text=text,
            device=self.device,
            **kwargs,
        )


def _is_quit_text(text):
    return str(text).strip().lower() in {":q", ":quit", "quit", "exit"}


def run_direct_pt_session(
    generator,
    player,
    input_lines,
    *,
    behavior_id=None,
    emotion_id=None,
    frames=None,
    fps=30.0,
    sampling_steps=32,
    seed=7,
    max_velocity_rad_s=3.0,
    smooth_window=None,
    style_controls=None,
    style_policy="sample",
    loops=1,
    realtime=True,
    condition_builder=None,
    writer=print,
):
    runs = 0
    with player as active_player:
        for raw_line in input_lines:
            text = str(raw_line).strip()
            if not text:
                continue
            if _is_quit_text(text):
                break
            inference_kwargs = {}
            if condition_builder is not None:
                inference_kwargs["condition_builder"] = condition_builder
            motion = generator.infer(
                text,
                behavior_id=behavior_id,
                emotion_id=emotion_id,
                frames=frames,
                fps=fps,
                sampling_steps=sampling_steps,
                seed=None if seed is None else int(seed) + runs,
                max_velocity_rad_s=max_velocity_rad_s,
                smooth_window=smooth_window,
                style_controls=style_controls,
                style_policy=style_policy,
                **inference_kwargs,
            )
            viewer_summary = active_player.play_trajectory(
                motion.trajectory,
                loops=loops,
                realtime=realtime,
            )
            runs += 1
            writer(json.dumps(motion.summary() | {"viewer": viewer_summary}, ensure_ascii=False))
            if active_player.viewer is not None and not active_player.viewer.is_running():
                break
    return {"runs": runs, "checkpoint": generator.info.path, "device": generator.device}


def _stdin_lines(prompt="> "):
    while True:
        try:
            yield input(prompt)
        except EOFError:
            return


def require_graphical_display():
    if not sys.platform.startswith("linux"):
        return
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return
    raise RuntimeError(
        "MuJoCo interactive viewer needs a graphical display. Connect with X11 forwarding "
        "(ssh -Y gez@172.16.60.184) or use a desktop/VNC session; --no-viewer can verify PT inference only."
    )


def _checkpoint_info_dict(info):
    return {
        "path": info.path,
        "architecture": info.architecture,
        "action_dim": info.action_dim,
        "condition_dim": info.condition_dim,
        "hidden_dim": info.hidden_dim,
        "layers": info.layers,
        "configured_steps": info.configured_steps,
        "checkpoint_step": info.checkpoint_step,
        "episodes_loaded": info.episodes_loaded,
        "parameter_count": info.parameter_count,
        "has_action_stats": info.has_action_stats,
    }


def resolve_runtime_paths(
    *,
    checkpoint=None,
    kimodo_experimental=False,
    kimodo_qwen=False,
    semantic_adapter_checkpoint=None,
):
    if kimodo_qwen and kimodo_experimental:
        raise ValueError("--kimodo-qwen already selects --kimodo-experimental")
    if (kimodo_qwen or kimodo_experimental) and checkpoint:
        raise ValueError("--checkpoint cannot be combined with a Kimodo checkpoint shortcut")
    generator_path = (
        EXPERIMENTAL_KIMODO_CHECKPOINT
        if kimodo_experimental or kimodo_qwen
        else Path(checkpoint or DEFAULT_CHECKPOINT)
    )
    semantic_path = None if semantic_adapter_checkpoint in (None, "") else Path(semantic_adapter_checkpoint)
    if (kimodo_qwen or (not checkpoint and not kimodo_experimental)) and semantic_path is None:
        semantic_path = DEFAULT_SEMANTIC_ADAPTER_CHECKPOINT
    return generator_path, semantic_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a ULA generator PT directly into an in-memory MuJoCo viewer")
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--kimodo-experimental",
        action="store_true",
        help="Use the contract-bound 5,000-step Kimodo MMDiT checkpoint",
    )
    parser.add_argument(
        "--kimodo-qwen",
        action="store_true",
        help="Use the 5,000-step Kimodo generator with the trained default Qwen semantic adapter",
    )
    parser.add_argument("--text", help="Run one prompt; omit for an interactive terminal prompt")
    parser.add_argument("--behavior-id")
    parser.add_argument("--emotion-id")
    parser.add_argument("--semantic-adapter-checkpoint")
    parser.add_argument(
        "--motion-latent-lora-checkpoint",
        help="Qwen Motion LoRA checkpoint recorded by a text-latent ULA V2 generator",
    )
    parser.add_argument("--motion-latent-device", help="Device for Qwen Motion LoRA; defaults to --device")
    parser.add_argument("--motion-latent-local-files-only", action="store_true")
    parser.add_argument("--semantic-model", help="Override the frozen Qwen model stored in the adapter checkpoint")
    parser.add_argument("--semantic-revision", help="Override the pinned Qwen model revision")
    parser.add_argument(
        "--allow-incompatible-semantic-encoder",
        action="store_true",
        help="Allow a Qwen model/revision that differs from the semantic adapter training metadata",
    )
    parser.add_argument("--semantic-device", help="Device for Qwen and the semantic adapter; defaults to --device")
    parser.add_argument(
        "--semantic-local-files-only",
        action="store_true",
        help="Load Qwen only from the local Hugging Face cache",
    )
    parser.add_argument(
        "--frames",
        type=int,
        help="Override output frames; V2 defaults to predicted duration × FPS",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--sampling-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-velocity-rad-s", type=float, default=3.0)
    parser.add_argument("--smooth-window", type=int, help="Defaults to the checkpoint preprocessing contract")
    parser.add_argument("--style-policy", choices=("sample", "mean"), default="sample")
    parser.add_argument(
        "--style-controls",
        nargs=3,
        type=float,
        metavar=("SIDE", "AMPLITUDE", "SPEED"),
        help="Override normalized V2 side/amplitude/speed controls",
    )
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--simplified", action="store_true")
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Run PT inference in memory without opening MuJoCo; requires --text",
    )
    args = parser.parse_args(argv)

    if args.no_viewer and not args.text:
        parser.error("--no-viewer requires --text")
    if args.frames is not None and args.frames < 2:
        parser.error("--frames must be at least 2")
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be finite and positive")
    if args.sampling_steps <= 0:
        parser.error("--sampling-steps must be positive")
    if args.loops < 0:
        parser.error("--loops must be zero or positive")
    if not math.isfinite(args.max_velocity_rad_s) or args.max_velocity_rad_s <= 0:
        parser.error("--max-velocity-rad-s must be finite and positive")
    if args.smooth_window is not None and args.smooth_window < 1:
        parser.error("--smooth-window must be at least 1")
    if args.motion_latent_lora_checkpoint and args.semantic_adapter_checkpoint:
        parser.error("Qwen Motion LoRA and the classification-only semantic adapter are mutually exclusive")
    try:
        checkpoint_path, semantic_checkpoint_path = resolve_runtime_paths(
            checkpoint=args.checkpoint,
            kimodo_experimental=args.kimodo_experimental,
            kimodo_qwen=args.kimodo_qwen,
            semantic_adapter_checkpoint=args.semantic_adapter_checkpoint,
        )
    except ValueError as exc:
        parser.error(str(exc))
    semantic_options = (
        args.semantic_model
        or args.semantic_revision
        or args.semantic_local_files_only
        or args.allow_incompatible_semantic_encoder
        or args.semantic_device
    )
    if semantic_options and not semantic_checkpoint_path:
        parser.error("semantic model options require --semantic-adapter-checkpoint")
    if not args.no_viewer:
        require_graphical_display()
    generator = PtMotionGenerator.from_checkpoint(checkpoint_path, device=args.device)
    print(json.dumps({"checkpoint": _checkpoint_info_dict(generator.info), "device": generator.device}, indent=2))

    condition_builder = None
    if (
        generator.info.condition_dim in {KIMODO_CONDITION_DIM, KIMODO_V2_CONDITION_DIM}
        and not semantic_checkpoint_path
        and not args.motion_latent_lora_checkpoint
    ):
        parser.error(
            "Kimodo generators require a semantic adapter condition bank; "
            "pass --kimodo-qwen or --semantic-adapter-checkpoint"
        )
    if args.motion_latent_lora_checkpoint:
        if generator.info.condition_dim != KIMODO_V2_CONDITION_DIM:
            parser.error("Qwen Motion LoRA requires a 264-dimensional ULA V2 generator")
        try:
            condition_builder, motion_text_checkpoint, condition_source, actual_hash = (
                load_motion_latent_lora_condition_builder(
                    generator.checkpoint,
                    args.motion_latent_lora_checkpoint,
                    device=args.motion_latent_device or args.device,
                    local_files_only=args.motion_latent_local_files_only,
                )
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "motion_latent_lora": str(args.motion_latent_lora_checkpoint),
                    "qwen": motion_text_checkpoint["qwen"],
                    "best_step": motion_text_checkpoint.get("best_step"),
                    "validation_metrics": motion_text_checkpoint.get("validation_metrics"),
                    "condition_source": condition_source,
                    "checkpoint_sha256": actual_hash,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif semantic_checkpoint_path:
        if generator.info.condition_dim not in {KIMODO_CONDITION_DIM, KIMODO_V2_CONDITION_DIM}:
            parser.error(
                f"Qwen semantic adapters require a {KIMODO_CONDITION_DIM}-dimensional Kimodo generator checkpoint"
            )
        from upper_body_skeleton.semantic_adapter import AdapterConditionBuilder, load_semantic_adapter

        semantic_adapter, semantic_checkpoint = load_semantic_adapter(
            semantic_checkpoint_path,
            model_name=args.semantic_model,
            revision=args.semantic_revision,
            device=args.semantic_device or args.device,
            local_files_only=args.semantic_local_files_only,
            allow_incompatible_encoder=args.allow_incompatible_semantic_encoder,
        )
        condition_bank = semantic_checkpoint.get("condition_bank")
        if condition_bank is None:
            parser.error(
                "semantic adapter checkpoint has no canonical 136-dimensional condition bank; retrain it with "
                "condition_dataset_dir"
            )
        try:
            condition_source = validate_generator_condition_source(generator.checkpoint, condition_bank)
        except ValueError as exc:
            parser.error(str(exc))
        condition_builder = AdapterConditionBuilder(semantic_adapter, condition_bank=condition_bank)
        text_encoder = semantic_adapter.text_encoder
        print(
            json.dumps(
                {
                    "semantic_adapter": str(semantic_checkpoint_path),
                    "qwen": semantic_checkpoint["qwen"],
                    "effective_encoder": {
                        "model_name": getattr(text_encoder, "model_name", None),
                        "revision": getattr(text_encoder, "revision", None),
                    },
                    "best_step": semantic_checkpoint.get("best_step"),
                    "validation_metrics": semantic_checkpoint.get("validation_metrics"),
                    "test_metrics": semantic_checkpoint.get("test_metrics"),
                    "condition_source": condition_source,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    infer_kwargs = {
        "behavior_id": args.behavior_id,
        "emotion_id": args.emotion_id,
        "frames": args.frames,
        "fps": args.fps,
        "sampling_steps": args.sampling_steps,
        "seed": args.seed,
        "max_velocity_rad_s": args.max_velocity_rad_s,
        "smooth_window": args.smooth_window,
        "style_controls": args.style_controls,
        "style_policy": args.style_policy,
    }
    if args.no_viewer:
        if condition_builder is not None:
            infer_kwargs["condition_builder"] = condition_builder
        motion = generator.infer(args.text, **infer_kwargs)
        print(json.dumps(motion.summary(), ensure_ascii=False, indent=2))
        return 0

    player = MujocoMotionPlayer(fps=args.fps, simplified=args.simplified)
    input_lines = [args.text] if args.text else _stdin_lines()
    if not args.text:
        print("输入文本后回车，PT 将直接在内存中推理并送入 MuJoCo；输入 :q 退出。")
    session_loops = args.loops
    if not args.text and session_loops == 0:
        session_loops = 1
        print("交互模式将每条动作播放一次；--loops 0 仅用于单条 --text 持续循环。")
    result = run_direct_pt_session(
        generator,
        player,
        input_lines,
        behavior_id=args.behavior_id,
        emotion_id=args.emotion_id,
        frames=args.frames,
        fps=args.fps,
        sampling_steps=args.sampling_steps,
        seed=args.seed,
        max_velocity_rad_s=args.max_velocity_rad_s,
        smooth_window=args.smooth_window,
        style_controls=args.style_controls,
        style_policy=args.style_policy,
        loops=session_loops,
        realtime=not args.no_realtime,
        condition_builder=condition_builder,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
