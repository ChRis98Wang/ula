#!/usr/bin/env python3
"""Build a read-only, restart-safe A/B morning report for clean ULA training.

The report has three explicit experiment arms:

* A: motion-only generator foundation
* B: the same foundation with an official frozen Qwen encoder
* C: the same foundation with a BEAT2-only Qwen LoRA

This tool never starts, resumes, or stops training.  Missing and partially
written artifacts are represented as ``unavailable`` instead of being treated
as zeros or successful checks.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
ARTIFACT_KIND = "ula_clean_training_three_arm_report_v1"
RETRIEVAL_KEYS = (
    "retrieval_loss",
    "text_to_motion_recall_at_1",
    "text_to_motion_recall_at_5",
    "motion_to_text_recall_at_1",
    "motion_to_text_recall_at_5",
    "text_to_motion_median_rank",
    "motion_to_text_median_rank",
    "positive_cosine",
    "negative_cosine",
    "cosine_gap",
)
LOWER_IS_BETTER = {
    "retrieval_loss",
    "text_to_motion_median_rank",
    "motion_to_text_median_rank",
}
RETRIEVAL_FIELD_ALIASES = {
    # The clean Qwen alignment pipeline names the group-centroid and episode
    # metrics explicitly.  Keep the report's compact legacy schema while
    # preferring the group-centroid retrieval metrics used for model selection.
    "retrieval_loss": ("retrieval_loss", "group_retrieval_loss"),
    "text_to_motion_recall_at_1": (
        "text_to_motion_recall_at_1",
        "group_text_to_motion_recall_at_1",
        "episode_text_to_motion_recall_at_1",
    ),
    "text_to_motion_recall_at_5": (
        "text_to_motion_recall_at_5",
        "group_text_to_motion_recall_at_5",
        "episode_text_to_motion_recall_at_5",
    ),
    "motion_to_text_recall_at_1": (
        "motion_to_text_recall_at_1",
        "group_motion_to_text_recall_at_1",
        "episode_motion_to_text_recall_at_1",
    ),
    "motion_to_text_recall_at_5": (
        "motion_to_text_recall_at_5",
        "group_motion_to_text_recall_at_5",
        "episode_motion_to_text_recall_at_5",
    ),
    "text_to_motion_median_rank": (
        "text_to_motion_median_rank",
        "group_text_to_motion_median_rank",
    ),
    "motion_to_text_median_rank": (
        "motion_to_text_median_rank",
        "group_motion_to_text_median_rank",
        "episode_motion_to_text_median_rank",
    ),
    "positive_cosine": (
        "positive_cosine",
        "positive_episode_cosine",
        "group_centroid_positive_cosine",
    ),
    "negative_cosine": ("negative_cosine", "negative_episode_cosine"),
    "cosine_gap": ("cosine_gap", "episode_cosine_gap"),
    # Group recall is evaluated over semantic-group queries.  Preserve the
    # episode count below as a native field, but use group count in the compact
    # table so its denominator matches the displayed recall metrics.
    "count": ("count", "semantic_group_count", "episode_count"),
}
NATIVE_RETRIEVAL_KEYS = tuple(
    dict.fromkeys(
        alias
        for aliases in RETRIEVAL_FIELD_ALIASES.values()
        for alias in aliases
        if alias not in RETRIEVAL_KEYS and alias != "count"
    )
)
CONDITION_SLICES = {
    "legacy_intent": (0, 6),
    "legacy_affect": (6, 14),
    "legacy_style": (14, 17),
    "legacy_gesture": (17, 23),
    "legacy_scalars": (23, 28),
    "legacy_text": (28, 92),
    "behavior_one_hot": (92, 119),
    "emotion_one_hot": (119, 125),
    "behavior_family": (125, 133),
    "style_controls": (133, 136),
    "qwen_motion_latent": (136, 264),
}
MOTION_ONLY_ZERO_SLICES = {
    "legacy_intent",
    "legacy_affect",
    "legacy_gesture",
    "behavior_one_hot",
    "emotion_one_hot",
    "behavior_family",
    "qwen_motion_latent",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _json_safe(value),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name.strip(), Path(raw_path).expanduser().resolve()


def parse_progress(path: str | Path) -> dict:
    """Stream a JSONL progress file, tolerating a concurrently written tail."""
    path = Path(path)
    result = {
        "status": "unavailable",
        "path": str(path),
        "records": 0,
        "malformed_records": 0,
        "current_step": None,
        "target_steps": None,
        "latest_train_total": None,
        "latest_grad_norm": None,
        "latest_validation": None,
        "best_validation": None,
        "best_validation_step": None,
        "age_seconds": None,
    }
    if not path.is_file():
        result["reason"] = "progress file does not exist"
        return result
    latest = None
    best = None
    best_value = None
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                result["malformed_records"] += 1
                continue
            if not isinstance(event, dict):
                result["malformed_records"] += 1
                continue
            result["records"] += 1
            latest = event
            step = event.get("step")
            if isinstance(step, int) and not isinstance(step, bool):
                result["current_step"] = step
            target = event.get("steps") or event.get("target_steps")
            if isinstance(target, int) and not isinstance(target, bool):
                result["target_steps"] = target
            validation = event.get("validation")
            if isinstance(validation, Mapping):
                value = _finite_float(
                    validation.get("total", validation.get("retrieval_loss"))
                )
                result["latest_validation"] = _json_safe(dict(validation))
                if value is not None and (best_value is None or value < best_value):
                    best_value = value
                    best = (result["current_step"], dict(validation))
    if latest is not None:
        train = latest.get("train")
        if isinstance(train, Mapping):
            result["latest_train_total"] = _finite_float(train.get("total"))
        else:
            result["latest_train_total"] = _finite_float(latest.get("loss"))
        result["latest_grad_norm"] = _finite_float(latest.get("grad_norm"))
    if best is not None:
        result["best_validation_step"] = best[0]
        result["best_validation"] = _json_safe(best[1])
    result["age_seconds"] = max(0.0, time.time() - path.stat().st_mtime)
    result["status"] = "available" if result["records"] else "unavailable"
    if not result["records"]:
        result["reason"] = "progress file contains no complete JSON records"
    return result


def _discover_progress(run_dir: Path) -> Path:
    candidates = (
        run_dir / "training" / "progress.jsonl",
        run_dir / "progress.jsonl",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def _discover_generator_checkpoint(run_dir: Path) -> Path | None:
    candidates = (
        run_dir / "training" / "ula_fm_checkpoint.pt",
        run_dir / "ula_fm_checkpoint.pt",
        run_dir / "training" / "best.pt",
        run_dir / "best.pt",
        # B/C are deliberately outside the formal checkpoint contract.  Merely
        # discovering this name does not authorize the formal loader below;
        # it is accepted only through the strict experimental ABC adapter.
        run_dir / "generator_experimental.pt",
    )
    return next((path for path in candidates if path.is_file()), None)


def _discover_condition_cache(run_dir: Path) -> Path | None:
    variant = run_dir.name
    candidates = (
        run_dir / "conditioning" / "conditions.npz",
        run_dir / "conditions.npz",
        run_dir
        / "prepared"
        / f"conditions_264d_{variant}.experimental.npz",
        run_dir.parent
        / "prepared"
        / f"conditions_264d_{variant}.experimental.npz",
    )
    return next((path for path in candidates if path.is_file()), None)


def _discover_completion_summary(run_dir: Path) -> Path | None:
    candidates = (
        run_dir / "training" / "training_summary.json",
        run_dir / "training_summary.json",
    )
    return next((path for path in candidates if path.is_file()), None)


def _discover_config(run_dir: Path) -> tuple[dict | None, Path | None]:
    local = (run_dir / "resolved_config.json", run_dir / "config.json")
    for path in local:
        if path.is_file():
            try:
                return _read_json(path), path
            except (OSError, ValueError, json.JSONDecodeError):
                pass
    config_dir = PROJECT_ROOT / "configs"
    if config_dir.is_dir():
        for path in sorted(config_dir.glob("*.json")):
            try:
                config = _read_json(path)
                output = config.get("output_dir")
                if output and Path(output).expanduser().resolve() == run_dir.resolve():
                    return config, path
            except (OSError, ValueError, json.JSONDecodeError):
                continue
    return None, None


def _processes_for_run(run_dir: Path) -> list[dict]:
    """Find Python training processes without depending on psutil or shelling out."""
    matches = []
    proc = Path("/proc")
    if not proc.is_dir():
        return matches
    try:
        uptime = float((proc / "uptime").read_text().split()[0])
        ticks = float(os.sysconf(os.sysconf_names["SC_CLK_TCK"]))
    except (OSError, ValueError, KeyError):
        uptime = ticks = None
    needle = run_dir.name.lower()
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            tokens = [
                token.decode("utf-8", errors="replace")
                for token in (entry / "cmdline").read_bytes().split(b"\0")
                if token
            ]
        except (OSError, PermissionError):
            continue
        if not tokens:
            continue
        command = " ".join(tokens).lower()
        executable = Path(tokens[0]).name.lower()
        if needle not in command or "train" not in command:
            continue
        if "python" not in executable and "torchrun" not in executable:
            continue
        elapsed = None
        state = None
        try:
            stat = (entry / "stat").read_text().split()
            state = stat[2]
            if uptime is not None and ticks:
                elapsed = max(0.0, uptime - float(stat[21]) / ticks)
        except (OSError, ValueError, IndexError):
            pass
        script = next((Path(token).name for token in tokens[1:] if "train" in token), None)
        matches.append(
            {
                "pid": int(entry.name),
                "state": state,
                "elapsed_seconds": elapsed,
                "executable": Path(tokens[0]).name,
                "script": script,
            }
        )
    return sorted(matches, key=lambda item: item["pid"])


def _target_steps(config: Mapping | None, progress: Mapping) -> int | None:
    if progress.get("target_steps") is not None:
        return int(progress["target_steps"])
    training = (config or {}).get("training")
    if isinstance(training, Mapping) and isinstance(training.get("steps"), int):
        return int(training["steps"])
    if isinstance((config or {}).get("steps"), int):
        return int(config["steps"])
    return None


def _lineage_strings(value: Any, *, prefix: str = "") -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            output.extend(_lineage_strings(item, prefix=child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            output.extend(_lineage_strings(item, prefix=f"{prefix}[{index}]"))
    elif isinstance(value, str):
        output.append((prefix, value))
    return output


def _forbidden_hits(value: Any, forbidden_tokens: Sequence[str]) -> list[dict]:
    hits = []
    lowered_tokens = [token.lower() for token in forbidden_tokens if token.strip()]
    for field, text in _lineage_strings(value):
        lowered = text.lower()
        matched = sorted({token for token in lowered_tokens if token in lowered})
        if matched:
            hits.append(
                {"field": field, "tokens": matched, "value": text[:300]}
            )
    return hits


def audit_condition_cache(
    path: str | Path | None,
    *,
    formal_episode_contract: str | None = None,
    experimental_variant: str | None = None,
) -> dict:
    if path is None:
        return {"status": "unavailable", "reason": "condition cache was not found"}
    path = Path(path)
    if not path.is_file():
        return {
            "status": "unavailable",
            "path": str(path),
            "reason": "condition cache does not exist",
        }
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "conditions" not in archive:
                raise ValueError("cache has no conditions array")
            conditions = np.asarray(archive["conditions"])
            if conditions.ndim != 2:
                raise ValueError("conditions must be a two-dimensional array")
            clip_ids = (
                archive["clip_ids"].astype(str)
                if "clip_ids" in archive
                else None
            )
            fixed_splits = (
                archive["fixed_split_assignments"].astype(str)
                if "fixed_split_assignments" in archive
                else None
            )
            slices = {}
            for name, (start, end) in CONDITION_SLICES.items():
                if end > conditions.shape[1]:
                    continue
                values = conditions[:, start:end]
                slices[name] = {
                    "start": start,
                    "end": end,
                    "nonzero_count": int(np.count_nonzero(values)),
                    "max_abs": (
                        float(np.max(np.abs(values))) if values.size else 0.0
                    ),
                    "expected_zero": bool(
                        (
                            formal_episode_contract
                            == "ula_v2_18d_motion_only_physical_qc_v1"
                            and name in MOTION_ONLY_ZERO_SLICES
                        )
                        or (
                            experimental_variant is not None
                            and end <= 133
                        )
                    ),
                }
            finite = bool(np.isfinite(conditions).all())
            style_matches = None
            if "style_controls" in archive and conditions.shape[1] >= 136:
                controls = np.asarray(archive["style_controls"])
                style_matches = bool(
                    controls.shape == conditions[:, 133:136].shape
                    and np.array_equal(controls, conditions[:, 133:136])
                )
            mask_counts = {}
            for key in archive.files:
                if key.endswith("_supervision_mask") or key.endswith(
                    "_conditioning_mask"
                ):
                    mask_counts[key] = int(np.count_nonzero(archive[key]))
            experimental_layout_valid = None
            if experimental_variant is not None:
                experimental_layout_valid = bool(
                    clip_ids is not None
                    and fixed_splits is not None
                    and len(clip_ids) == len(conditions)
                    and len(fixed_splits) == len(conditions)
                    and len(set(clip_ids.tolist())) == len(clip_ids)
                    and set(fixed_splits.tolist())
                    == {"train", "validation", "test"}
                )
    except Exception as exc:  # Artifact may be mid-write; report, do not crash.
        return {
            "status": "unavailable",
            "path": str(path),
            "reason": f"{type(exc).__name__}: {exc}",
        }
    zero_violations = [
        name
        for name, item in slices.items()
        if item["expected_zero"] and item["nonzero_count"]
    ]
    passed = bool(
        finite
        and conditions.shape[1] == 264
        and not zero_violations
        and style_matches is not False
        and experimental_layout_valid is not False
    )
    return {
        "status": "completed",
        "path": str(path),
        "count": int(conditions.shape[0]),
        "condition_dim": int(conditions.shape[1]),
        "finite": finite,
        "style_controls_match_cache_slice": style_matches,
        "supervision_nonzero_counts": mask_counts,
        "experimental_variant": experimental_variant,
        "experimental_metadata_only": experimental_variant is not None,
        "experimental_layout_valid": experimental_layout_valid,
        "slices": slices,
        "zero_required_violations": zero_violations,
        "passed": passed,
    }


def _trajectory_metrics(trajectory: np.ndarray, *, fps: float = 30.0) -> dict:
    values = np.asarray(trajectory, dtype=np.float64)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("trajectory must be a finite [frames, joints] array")
    centered = values - values.mean(axis=0, keepdims=True)
    velocity = np.diff(values, axis=0) * float(fps)
    ranges = np.ptp(values, axis=0)
    return {
        "temporal_rms_rad": float(np.sqrt(np.mean(np.square(centered)))),
        "mean_joint_range_rad": float(ranges.mean()),
        "max_joint_range_rad": float(ranges.max()),
        "velocity_rms_rad_per_sec": (
            float(np.sqrt(np.mean(np.square(velocity)))) if len(velocity) else 0.0
        ),
    }


def _load_strict_experimental_abc(
    config_path: str | Path, *, device: str
) -> dict:
    """Load A/B/C using the independent video tool's strict contracts.

    A continues through ``validate_checkpoint`` (the clean formal loader).
    B/C continue through ``validate_experimental_checkpoint`` only.  This
    adapter intentionally does not catch or reinterpret a formal validation
    failure and never sends an experimental artifact to the formal validator.
    """
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from tools.experimental import build_beat2_clean_abc_video as abc

    plan = abc.validate_inputs(config_path)
    records = plan["manifest_records"]
    manifest_sha = plan["manifest_sha256"]
    style = abc.load_style_cache(
        plan["paths"]["style_cache"], manifest_records=records
    )
    frozen, _ = abc.load_qwen_cache(
        plan["paths"]["frozen_cache"],
        variant="frozen_base",
        manifest_records=records,
        manifest_sha256=manifest_sha,
    )
    lora, _ = abc.load_qwen_cache(
        plan["paths"]["lora_cache"],
        variant="lora_finetuned",
        manifest_records=records,
        manifest_sha256=manifest_sha,
    )
    foundation_model, foundation_metadata = abc.validate_checkpoint(
        plan["checkpoints"]["A"],
        expected_manifest_sha256=manifest_sha,
        branch="A",
        device=device,
    )
    foundation_metadata["completion_summary"] = abc.validate_completion_summary(
        plan["completion_summaries"]["A"],
        checkpoint_path=plan["checkpoints"]["A"],
        branch="A",
    )
    models = {"A": foundation_model}
    metadata = {"A": foundation_metadata}
    latent_maps = {"B": frozen, "C": lora}
    variants = {"B": "frozen_base", "C": "lora_finetuned"}
    source_hashes = {
        "B": abc.sha256_file(plan["paths"]["frozen_cache"]),
        "C": abc.sha256_file(plan["paths"]["lora_cache"]),
    }
    style_hash = abc.sha256_file(plan["paths"]["style_cache"])
    for branch in ("B", "C"):
        expected_conditions = {
            clip_id: abc.compose_condition(
                style[clip_id], latent_maps[branch][clip_id]
            )
            for clip_id in records
        }
        model, branch_metadata = abc.validate_experimental_checkpoint(
            plan["checkpoints"][branch],
            branch=branch,
            variant=variants[branch],
            expected_manifest_sha256=manifest_sha,
            foundation_checkpoint_sha256=foundation_metadata["sha256"],
            source_128d_cache_sha256=source_hashes[branch],
            style_cache_sha256=style_hash,
            manifest_records=records,
            expected_conditions=expected_conditions,
            device=device,
        )
        branch_metadata["completion_summary"] = abc.validate_completion_summary(
            plan["completion_summaries"][branch],
            checkpoint_path=plan["checkpoints"][branch],
            branch=branch,
            variant=variants[branch],
            checkpoint_sha256=branch_metadata["sha256"],
            foundation_checkpoint_sha256=foundation_metadata["sha256"],
            condition_cache_sha256=branch_metadata["condition_cache_sha256"],
            pair_contract_sha256=branch_metadata["pair_contract_sha256"],
        )
        abc.validate_zero_latent_equivalence(
            foundation_model,
            model,
            style_conditions=[
                abc.compose_condition(style[case["clip_id"]], None)
                for case in plan["cases"]
            ],
            device=device,
        )
        # The checkpoint has already passed the experimental-only loader.  Read
        # scalar/report payloads for the morning report; model_state_dict is
        # deliberately excluded.
        import torch

        payload = torch.load(
            plan["checkpoints"][branch], map_location="cpu", weights_only=True
        )
        branch_metadata.update(
            {
                "semantic_scope": payload.get("semantic_scope"),
                "semantic_supervision_status": payload.get(
                    "semantic_supervision_status"
                ),
                "held_out_posttrain_metrics": _json_safe(
                    payload.get("metrics") or {}
                ),
                "preservation": _json_safe(payload.get("preservation") or {}),
                "split_contract": _json_safe(
                    payload.get("split_contract") or {}
                ),
                "training_contract": _json_safe(
                    payload.get("training_contract") or {}
                ),
            }
        )
        models[branch] = model
        metadata[branch] = branch_metadata
    if len(
        {
            item["initialization_state_sha256"]
            for item in metadata.values()
        }
    ) != 1:
        raise abc.EvaluationContractError(
            "A/B/C do not share the same clean random initialization"
        )
    by_checkpoint = {
        str(plan["checkpoints"][branch].resolve()): {
            "branch": branch,
            "model": models[branch],
            "metadata": metadata[branch],
        }
        for branch in ("A", "B", "C")
    }
    return {
        "config_path": str(Path(config_path).resolve()),
        "manifest_sha256": manifest_sha,
        "by_checkpoint": by_checkpoint,
        "branches": metadata,
    }


def _model_audit(
    checkpoint_path: Path,
    *,
    device: str,
    frames: int,
    sampling_steps: int,
    seeds: Sequence[int],
    style_delta: float,
    padding_tolerance: float,
    min_generation_rms: float,
    min_style_response: float,
    validated_entry: Mapping[str, Any] | None = None,
) -> tuple[dict, dict]:
    """Load one exported checkpoint and run deterministic, dataset-free audits."""
    try:
        import torch

        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from upper_body_skeleton.ula_training import sample_trajectory
        from upper_body_skeleton.ula_v2_18d_head import (
            instantiate_checkpoint_model,
            validate_checkpoint_contract,
        )
        from upper_body_skeleton.ula_v2_18d_random_init import (
            forward_with_frame_mask,
        )

        if validated_entry is None:
            if checkpoint_path.name == "generator_experimental.pt":
                raise ValueError(
                    "experimental checkpoint requires --abc-video-config and "
                    "the strict experimental-only loader"
                )
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            validate_checkpoint_contract(checkpoint, expected_action_dim=18)
            model = instantiate_checkpoint_model(checkpoint, device=device).eval()
            validated_metadata: Mapping[str, Any] = {}
        else:
            model = validated_entry["model"]
            validated_metadata = validated_entry.get("metadata") or {}
            checkpoint = {
                "condition_dim": 264,
                "action_dim": 18,
                **dict(validated_metadata),
            }
    except Exception as exc:
        return (
            {
                "status": "unavailable",
                "reason": f"{type(exc).__name__}: {exc}",
            },
            {},
        )

    condition_dim = int(checkpoint["condition_dim"])
    action_dim = int(checkpoint["action_dim"])
    zero = np.zeros(condition_dim, dtype=np.float32)
    style_conditions = {}
    for label, value in (("low", -style_delta), ("neutral", 0.0), ("high", style_delta)):
        condition = zero.copy()
        condition[134] = value
        style_conditions[label] = condition
    generated: dict[str, list[dict]] = {name: [] for name in style_conditions}
    trajectories: dict[str, list[np.ndarray]] = {name: [] for name in style_conditions}
    try:
        for seed in seeds:
            for label, condition in style_conditions.items():
                trajectory = sample_trajectory(
                    model,
                    condition,
                    frames=frames,
                    action_dim=action_dim,
                    steps=sampling_steps,
                    device=device,
                    seed=int(seed),
                    action_stats=getattr(model, "action_stats", None),
                )
                trajectories[label].append(trajectory)
                generated[label].append(_trajectory_metrics(trajectory))

        def metric_mean(label: str, key: str) -> float:
            return float(np.mean([row[key] for row in generated[label]]))

        amplitude = {
            label: {
                key: metric_mean(label, key)
                for key in generated[label][0]
            }
            for label in generated
        }
        neutral_rms = amplitude["neutral"]["temporal_rms_rad"]
        generated_response = (
            amplitude["high"]["temporal_rms_rad"]
            - amplitude["low"]["temporal_rms_rad"]
        )

        torch.manual_seed(913)
        valid_lengths = (max(3, frames // 2), max(4, frames - 2))
        width = max(valid_lengths) + 3
        x = torch.randn((2, width, action_dim), dtype=torch.float32, device=device)
        mask = torch.zeros((2, width), dtype=torch.bool, device=device)
        for row, count in enumerate(valid_lengths):
            mask[row, :count] = True
        t = torch.tensor([0.25, 0.75], dtype=torch.float32, device=device)
        conditions = torch.zeros(
            (2, condition_dim), dtype=torch.float32, device=device
        )
        changed = x.clone()
        changed[~mask] = 1000.0
        with torch.no_grad():
            first = forward_with_frame_mask(model, x, t, conditions, mask)
            second = forward_with_frame_mask(model, changed, t, conditions, mask)
        valid_difference = (first - second).abs()[mask]
        padding_error = (
            float(valid_difference.max().detach().cpu())
            if valid_difference.numel()
            else 0.0
        )

        torch.manual_seed(121)
        probe_x = torch.randn(
            (1, max(4, min(frames, 24)), action_dim),
            dtype=torch.float32,
            device=device,
        )
        probe_t = torch.tensor([0.5], dtype=torch.float32, device=device)
        probe_conditions = [zero]
        response_names = []
        for name in (
            "legacy_intent",
            "behavior_one_hot",
            "emotion_one_hot",
            "behavior_family",
            "style_controls",
            "qwen_motion_latent",
        ):
            condition = zero.copy()
            start, end = CONDITION_SLICES[name]
            index = start + (1 if name == "style_controls" else 0)
            condition[min(index, end - 1)] = style_delta
            probe_conditions.append(condition)
            response_names.append(name)
        probe_batch = torch.as_tensor(
            np.stack(probe_conditions), dtype=torch.float32, device=device
        )
        with torch.no_grad():
            output = model(
                probe_x.expand(len(probe_conditions), -1, -1),
                probe_t.expand(len(probe_conditions)),
                probe_batch,
            )
        baseline = output[0]
        responses = {
            name: float(
                torch.sqrt(torch.mean(torch.square(output[index + 1] - baseline)))
                .detach()
                .cpu()
            )
            for index, name in enumerate(response_names)
        }
        style_direct = responses["style_controls"]
        result = {
            "status": "completed",
            "settings": {
                "device": device,
                "frames": frames,
                "sampling_steps": sampling_steps,
                "seeds": list(seeds),
                "style_amplitude_delta": style_delta,
            },
            "generation_amplitude": {
                "by_style": amplitude,
                "neutral_temporal_rms_rad": neutral_rms,
                "minimum_required_rms_rad": min_generation_rms,
                "noncollapsed": bool(neutral_rms >= min_generation_rms),
            },
            "style_condition_response": {
                "direct_flow_rmse": style_direct,
                "generated_high_minus_low_temporal_rms_rad": generated_response,
                "monotonic_amplitude": bool(generated_response > 0),
                "minimum_direct_response_rmse": min_style_response,
                "passed": bool(
                    style_direct >= min_style_response and generated_response > 0
                ),
            },
            "condition_response_by_slice_rmse": responses,
            "padding_audit": {
                "valid_output_max_abs_difference": padding_error,
                "tolerance": padding_tolerance,
                "passed": bool(padding_error <= padding_tolerance),
            },
        }
        result["passed"] = bool(
            result["generation_amplitude"]["noncollapsed"]
            and result["style_condition_response"]["passed"]
            and result["padding_audit"]["passed"]
        )
        metadata = {
            **_json_safe(dict(validated_metadata)),
            "artifact_kind": checkpoint.get("artifact_kind"),
            "architecture": checkpoint.get("architecture"),
            "global_step": checkpoint.get("global_step"),
            "best_step": checkpoint.get("best_step"),
            "best_validation_loss": checkpoint.get("best_validation_loss"),
            "formal_episode_contract": checkpoint.get("formal_episode_contract"),
            "condition_dim": condition_dim,
            "action_dim": action_dim,
            "config": _json_safe(checkpoint.get("config") or {}),
            "sources": _json_safe(checkpoint.get("sources") or {}),
            "data_provenance": _json_safe(checkpoint.get("data_provenance") or {}),
        }
        return result, metadata
    except Exception as exc:
        return (
            {
                "status": "unavailable",
                "reason": f"{type(exc).__name__}: {exc}",
            },
            {
                "architecture": checkpoint.get("architecture"),
                "formal_episode_contract": checkpoint.get(
                    "formal_episode_contract"
                ),
            },
        )
    finally:
        try:
            del model
            if "torch" in locals() and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _load_artifact(
    path: Path, *, role: str | None = None
) -> tuple[dict | None, Path | None, str | None]:
    if not path.exists():
        return None, None, "artifact path does not exist"
    if path.is_dir():
        role_candidates: tuple[Path, ...] = ()
        normalized_role = str(role or "").lower()
        if "frozen" in normalized_role:
            role_candidates = (path / "qwen_frozen_base_summary.json",)
        elif "lora" in normalized_role or "finetuned" in normalized_role:
            role_candidates = (path / "qwen_lora_finetuned_summary.json",)
        candidates = role_candidates + (
            path / "training_summary.json",
            path / "evaluation.json",
            path / "report.json",
            path / "comparison.json",
            path / "best.pt",
            path / "last.pt",
        )
        selected = next((item for item in candidates if item.is_file()), None)
        if selected is None:
            return None, None, "artifact directory contains no supported file"
        path = selected
    try:
        if path.suffix.lower() in {".json", ".jsonl"}:
            if path.suffix.lower() == ".jsonl":
                progress = parse_progress(path)
                payload = {
                    "validation_metrics": progress.get("best_validation") or {},
                    "global_step": progress.get("current_step"),
                    "best_step": progress.get("best_validation_step"),
                }
            else:
                payload = _read_json(path)
        else:
            import torch

            payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            return None, path, "artifact payload is not a mapping"
        return payload, path, None
    except Exception as exc:
        return None, path, f"{type(exc).__name__}: {exc}"


def _metrics_from_payload(payload: Mapping) -> dict:
    validation = (
        payload.get("final_validation")
        or payload.get("validation_metrics")
        or payload.get("best_validation")
        or payload.get("validation")
    )
    test = payload.get("final_test") or payload.get("test_metrics") or payload.get(
        "test"
    )
    metric_input_keys = {
        alias
        for aliases in RETRIEVAL_FIELD_ALIASES.values()
        for alias in aliases
    }
    if validation is None and any(key in payload for key in metric_input_keys):
        validation = payload

    def clean(value: Any) -> dict | None:
        if not isinstance(value, Mapping):
            return None
        result = {}
        for output_key, aliases in RETRIEVAL_FIELD_ALIASES.items():
            for input_key in aliases:
                number = _finite_float(value.get(input_key))
                if number is not None:
                    result[output_key] = number
                    break
        # Retain the pipeline-native fields in JSON for unambiguous downstream
        # inspection while the Markdown table consumes the normalized aliases.
        result.update(
            {
                key: number
                for key in NATIVE_RETRIEVAL_KEYS
                if (number := _finite_float(value.get(key))) is not None
            }
        )
        return result or None

    return {"validation": clean(validation), "test": clean(test)}


def _select_qwen_branch(payload: Mapping, role: str) -> tuple[Mapping | None, str | None]:
    """Select one arm when the supplied artifact is the A/B comparison."""
    has_comparison_arms = isinstance(payload.get("baseline"), Mapping) or isinstance(
        payload.get("lora_finetuned"), Mapping
    )
    if not has_comparison_arms:
        return payload, None
    normalized_role = role.lower()
    branch_name = (
        "baseline"
        if "frozen" in normalized_role
        else "lora_finetuned"
        if "lora" in normalized_role or "finetuned" in normalized_role
        else None
    )
    if branch_name is None:
        return None, "cannot select a comparison branch for the requested role"
    branch = payload.get(branch_name)
    if not isinstance(branch, Mapping):
        return None, f"comparison artifact does not contain {branch_name}"
    return branch, None


def summarize_qwen_artifact(
    role: str, path: str | Path | None, forbidden_tokens: Sequence[str]
) -> dict:
    if path is None:
        return {
            "role": role,
            "status": "unavailable",
            "reason": f"{role} artifact was not provided",
        }
    requested = Path(path).expanduser().resolve()
    payload, selected, error = _load_artifact(requested, role=role)
    if payload is None:
        return {
            "role": role,
            "status": "unavailable",
            "path": str(requested),
            "reason": error,
        }
    container_artifact_kind = payload.get("artifact_kind")
    payload, branch_error = _select_qwen_branch(payload, role)
    if payload is None:
        return {
            "role": role,
            "status": "unavailable",
            "path": str(requested),
            "selected_artifact": str(selected),
            "reason": branch_error,
        }
    metrics = _metrics_from_payload(payload)
    status = (
        "available"
        if metrics["validation"] is not None or metrics["test"] is not None
        else "unavailable"
    )
    config = payload.get("config") if isinstance(payload.get("config"), Mapping) else {}
    sources = payload.get("sources") if isinstance(payload.get("sources"), Mapping) else {}
    qwen = payload.get("qwen") if isinstance(payload.get("qwen"), Mapping) else {}
    lora_present = bool(
        payload.get("qwen_lora_state_dict")
        or config.get("lora_rank")
        or qwen.get("lora")
        or "lora" in str(payload.get("artifact_kind", "")).lower()
    )
    declared_mode = (
        payload.get("qwen_mode")
        or payload.get("encoder_policy")
        or config.get("qwen_policy")
        or payload.get("variant")
    )
    foundation_sha = next(
        (
            value
            for container in (payload, sources, config)
            for key, value in container.items()
            if key
            in {
                "foundation_checkpoint_sha256",
                "generator_checkpoint_sha256",
                "source_generator_checkpoint_sha256",
            }
            and isinstance(value, str)
        ),
        None,
    )
    result = {
        "role": role,
        "status": status,
        "path": str(requested),
        "selected_artifact": str(selected),
        "container_artifact_kind": container_artifact_kind,
        "artifact_kind": payload.get("artifact_kind"),
        "global_step": payload.get("global_step", payload.get("steps")),
        "best_step": payload.get("best_step"),
        "declared_qwen_mode": declared_mode,
        "lora_state_present": lora_present,
        "qwen": _json_safe(qwen),
        "metrics": metrics,
        "sources": _json_safe(sources),
        "foundation_checkpoint_sha256": foundation_sha,
        "provenance_audit": {
            "forbidden_tokens": list(forbidden_tokens),
            "hits": _forbidden_hits(
                {"config": config, "sources": sources}, forbidden_tokens
            ),
        },
    }
    result["provenance_audit"]["passed"] = not result["provenance_audit"]["hits"]
    if status == "unavailable":
        result["reason"] = "artifact contains no held-out retrieval metrics"
    return result


def compare_qwen_artifacts(frozen: Mapping, finetuned: Mapping) -> dict:
    output = {
        "status": "unavailable",
        "comparison": "finetuned_minus_frozen",
        "splits": {},
    }
    for split in ("validation", "test"):
        baseline = (frozen.get("metrics") or {}).get(split)
        candidate = (finetuned.get("metrics") or {}).get(split)
        if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
            output["splits"][split] = {
                "status": "unavailable",
                "reason": "both artifacts must provide this held-out split",
            }
            continue
        deltas = {}
        for key in RETRIEVAL_KEYS:
            left = _finite_float(baseline.get(key))
            right = _finite_float(candidate.get(key))
            if left is None or right is None:
                continue
            raw_delta = right - left
            deltas[key] = {
                "frozen": left,
                "finetuned": right,
                "delta": raw_delta,
                "improvement": -raw_delta if key in LOWER_IS_BETTER else raw_delta,
            }
        count_match = baseline.get("count") == candidate.get("count")
        output["splits"][split] = {
            "status": "available",
            "sample_count_match": count_match,
            "deltas": deltas,
        }
        output["status"] = "available"

    frozen_sources = frozen.get("sources") or {}
    tuned_sources = finetuned.get("sources") or {}
    shared_hash_fields = sorted(
        key
        for key in set(frozen_sources).intersection(tuned_sources)
        if key.endswith("_sha256")
    )
    mismatches = [
        key
        for key in shared_hash_fields
        if frozen_sources.get(key) != tuned_sources.get(key)
    ]
    frozen_foundation = frozen.get("foundation_checkpoint_sha256")
    tuned_foundation = finetuned.get("foundation_checkpoint_sha256")
    if frozen_foundation and tuned_foundation:
        foundation_status = (
            "verified_same" if frozen_foundation == tuned_foundation else "mismatch"
        )
    else:
        foundation_status = "unavailable"
    output["comparability"] = {
        "same_generator_by_report_design": True,
        "declared_foundation_lineage": foundation_status,
        "shared_source_hash_fields": shared_hash_fields,
        "source_hash_mismatches": mismatches,
        "held_out_counts_match": all(
            split.get("status") != "available" or split.get("sample_count_match")
            for split in output["splits"].values()
        ),
        "passed": bool(
            not mismatches
            and foundation_status != "mismatch"
            and all(
                split.get("status") != "available"
                or split.get("sample_count_match")
                for split in output["splits"].values()
            )
        ),
    }
    return output


def summarize_video_artifact(path: str | Path | None) -> dict:
    if path is None:
        return {
            "status": "unavailable",
            "reason": "video summary was not provided",
        }
    requested = Path(path).expanduser().resolve()
    if requested.is_dir():
        requested = requested / "summary.json"
    if not requested.is_file():
        return {
            "status": "unavailable",
            "path": str(requested),
            "reason": "video summary does not exist",
        }
    try:
        payload = _read_json(requested)
        if (
            payload.get("artifact_kind")
            != "beat2_clean_adaln_abc_experimental_video_v1"
            or payload.get("status") != "complete"
        ):
            raise ValueError("video artifact is not a completed ABC evaluation")
        cases = payload.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError("video artifact has no held-out cases")
        for case in cases:
            if not isinstance(case, Mapping) or set(case.get("branches") or {}) != {
                "A",
                "B",
                "C",
            }:
                raise ValueError("video case does not contain A/B/C branches")
            side = case.get("side_by_side")
            if not isinstance(side, Mapping):
                raise ValueError("video case has no side-by-side receipt")
            side_path = Path(str(side.get("output_mp4", ""))).expanduser()
            expected_sha = side.get("sha256")
            if (
                not side_path.is_file()
                or not isinstance(expected_sha, str)
                or _sha256_file(side_path) != expected_sha
            ):
                raise ValueError("side-by-side video receipt/hash is invalid")
    except Exception as exc:
        return {
            "status": "unavailable",
            "path": str(requested),
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return {
        "status": "complete",
        "path": str(requested),
        "case_count": len(cases),
        "source_manifest_sha256": payload.get("source_manifest_sha256"),
        "checkpoints": _json_safe(payload.get("checkpoints") or {}),
        "contracts": _json_safe(payload.get("contracts") or {}),
        "cases": _json_safe(cases),
    }


def summarize_generator(
    name: str,
    run_dir: Path,
    *,
    condition_cache: Path | None,
    forbidden_tokens: Sequence[str],
    model_audits: bool,
    audit_options: Mapping[str, Any],
    previous: Mapping | None,
    strict_abc: Mapping[str, Any] | None = None,
) -> dict:
    progress_path = _discover_progress(run_dir)
    progress = parse_progress(progress_path)
    config, config_path = _discover_config(run_dir)
    target_steps = _target_steps(config, progress)
    processes = _processes_for_run(run_dir)
    current_step = progress.get("current_step")
    lifecycle = "unavailable"
    if processes:
        lifecycle = "running"
    elif current_step is not None and target_steps is not None and current_step >= target_steps:
        lifecycle = "completed"
    elif current_step is not None:
        lifecycle = "not_running_incomplete"
    elif run_dir.exists():
        lifecycle = "initializing_or_incomplete"
    completion = (
        min(1.0, current_step / target_steps)
        if current_step is not None and target_steps
        else None
    )
    rate = eta = None
    if previous and current_step is not None:
        old_step = ((previous.get("training") or {}).get("current_step"))
        old_time = previous.get("_generated_epoch")
        if isinstance(old_step, int) and isinstance(old_time, (int, float)):
            elapsed = time.time() - float(old_time)
            if elapsed > 0 and current_step > old_step:
                rate = (current_step - old_step) / elapsed
                if target_steps and current_step < target_steps:
                    eta = (target_steps - current_step) / rate

    checkpoint_path = _discover_generator_checkpoint(run_dir)
    strict_entry = None
    if checkpoint_path is not None and strict_abc is not None:
        strict_entry = (strict_abc.get("by_checkpoint") or {}).get(
            str(checkpoint_path.resolve())
        )
    formal_contract = None
    metadata: dict = {}
    audit = {"status": "skipped", "reason": "model audits disabled"}
    if model_audits:
        if checkpoint_path is None:
            audit = {
                "status": "unavailable",
                "reason": "exported generator checkpoint was not found",
            }
        else:
            audit, metadata = _model_audit(
                checkpoint_path,
                **dict(audit_options),
                validated_entry=strict_entry,
            )
            formal_contract = metadata.get("formal_episode_contract")
    if formal_contract is None and config:
        formal_contract = config.get("formal_episode_contract")
    cache_path = condition_cache or _discover_condition_cache(run_dir)
    experimental_variant = metadata.get("variant")
    condition_audit = audit_condition_cache(
        cache_path,
        formal_episode_contract=formal_contract,
        experimental_variant=(
            str(experimental_variant) if experimental_variant else None
        ),
    )

    completion_summary = {}
    completion_summary_path = _discover_completion_summary(run_dir)
    if completion_summary_path is not None:
        try:
            completion_summary = _read_json(completion_summary_path)
        except Exception:
            completion_summary = {}
    if strict_entry is not None:
        receipt = (strict_entry.get("metadata") or {}).get(
            "completion_summary"
        ) or {}
        completed = receipt.get("completed_steps")
        target = receipt.get("target_steps")
        if isinstance(completed, int):
            current_step = completed
        if isinstance(target, int):
            target_steps = target
        if (
            isinstance(completed, int)
            and isinstance(target, int)
            and (
                completed >= target
                or receipt.get("stopped_early") is True
            )
        ):
            lifecycle = "completed"
            completion = min(1.0, completed / target)

    initialization_report = None
    init_path = run_dir / "initialization" / "initialization_report.json"
    if init_path.is_file():
        try:
            initialization_report = _read_json(init_path)
        except Exception:
            pass
    motion_lineage = {
        "config_motion_sources": (config or {}).get("motion_sources"),
        "checkpoint_motion_sources": (metadata.get("sources") or {}).get(
            "motion_manifests"
        ),
        "initialization_sources": (initialization_report or {}).get("sources"),
    }
    conditioning_lineage = {
        "config_qwen_checkpoint": (config or {}).get("qwen_checkpoint"),
    }
    cache_meta_path = (
        cache_path.with_suffix(cache_path.suffix + ".json") if cache_path else None
    )
    if cache_meta_path and cache_meta_path.is_file():
        try:
            cache_meta = _read_json(cache_meta_path)
            conditioning_lineage["cache_qwen_checkpoint"] = cache_meta.get(
                "qwen_checkpoint"
            )
        except Exception:
            pass
    motion_hits = _forbidden_hits(motion_lineage, forbidden_tokens)
    condition_hits = _forbidden_hits(conditioning_lineage, forbidden_tokens)
    best = progress.get("best_validation") or {}
    best_total = _finite_float(
        best.get("total", best.get("retrieval_loss"))
        if isinstance(best, Mapping)
        else None
    )
    return {
        "name": name,
        "path": str(run_dir),
        "availability": "available" if run_dir.exists() else "unavailable",
        "lifecycle": lifecycle,
        "processes": processes,
        "config_path": str(config_path) if config_path else None,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_contract": (
            "experimental_only"
            if metadata.get("experimental_only") is True
            else "formal"
        ),
        "architecture": metadata.get("architecture")
        or ((config or {}).get("model") or {}).get("architecture"),
        "training": {
            "current_step": current_step,
            "target_steps": target_steps,
            "completion_fraction": completion,
            "best_validation_total": best_total,
            "best_validation_step": progress.get("best_validation_step"),
            "latest_train_total": progress.get("latest_train_total"),
            "latest_grad_norm": progress.get("latest_grad_norm"),
            "progress_age_seconds": progress.get("age_seconds"),
            "steps_per_second_since_previous_report": rate,
            "eta_seconds": eta,
            "progress_parse": progress,
        },
        "generator_quality": audit,
        "held_out_posttrain_metrics": _json_safe(
            metadata.get("held_out_posttrain_metrics")
            or {
                key: completion_summary.get(key)
                for key in (
                    "initial_validation",
                    "final_validation",
                    "test",
                    "initial_condition_response",
                    "final_condition_response",
                    "metrics_delta",
                )
                if completion_summary.get(key) is not None
            }
        ),
        "lineage_hashes": {
            key: metadata.get(key)
            for key in (
                "sha256",
                "foundation_checkpoint_sha256",
                "condition_cache_sha256",
                "pair_contract_sha256",
                "manifest_sha256",
                "initialization_state_sha256",
            )
            if metadata.get(key) is not None
        },
        "semantic_scope": metadata.get("semantic_scope"),
        "semantic_supervision_status": metadata.get(
            "semantic_supervision_status"
        ),
        "preservation": _json_safe(metadata.get("preservation") or {}),
        "condition_slice_audit": condition_audit,
        "provenance_audit": {
            "forbidden_tokens": list(forbidden_tokens),
            "strict_checkpoint_contract": {
                "loader": (
                    "experimental_only"
                    if metadata.get("experimental_only") is True
                    else "formal"
                ),
                "passed": bool(
                    checkpoint_path is not None
                    and audit.get("status") == "completed"
                    and (
                        metadata.get("experimental_only") is not True
                        or metadata.get("no_kimodo") is True
                    )
                ),
            },
            "motion_training_sources": {
                "hits": motion_hits,
                "passed": not motion_hits,
            },
            "conditioning_artifacts": {
                "hits": condition_hits,
                "passed": not condition_hits,
            },
        },
        "_generated_epoch": time.time(),
    }


def _fmt(value: Any, digits: int = 4) -> str:
    number = _finite_float(value)
    return "—" if number is None else f"{number:.{digits}f}"


def _pct(value: Any) -> str:
    number = _finite_float(value)
    return "—" if number is None else f"{100.0 * number:.1f}%"


def render_markdown(report: Mapping) -> str:
    lines = [
        "# Clean ULA three-arm morning report",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Experiment arms",
        "",
        "| Arm | Generator | Qwen state | Artifact status |",
        "|---|---|---|---|",
    ]
    for key in ("A", "B", "C"):
        arm = report["experiment_arms"][key]
        lines.append(
            f"| {key} | {arm['generator']} | {arm['qwen_state']} | {arm['status']} |"
        )
    lines.extend(
        [
            "",
            "B and C reference the exact same foundation entry in this report. "
            "Artifact-declared checkpoint lineage is checked separately below.",
            "",
            "## Generator training and quality",
            "",
            "| Run | State | PID | Step | Progress | Best val | Best step | Amplitude RMS | Style response | Padding |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for name, generator in report["generators"].items():
        training = generator["training"]
        quality = generator["generator_quality"]
        amplitude = (quality.get("generation_amplitude") or {}).get(
            "neutral_temporal_rms_rad"
        )
        style = (quality.get("style_condition_response") or {}).get("passed")
        padding = (quality.get("padding_audit") or {}).get("passed")
        pid = ",".join(str(item["pid"]) for item in generator["processes"]) or "—"
        step = training.get("current_step")
        target = training.get("target_steps")
        step_text = (
            f"{step}/{target}"
            if step is not None and target is not None
            else str(step if step is not None else "—")
        )
        lines.append(
            f"| {name} | {generator['lifecycle']} | {pid} | {step_text} | "
            f"{_pct(training.get('completion_fraction'))} | "
            f"{_fmt(training.get('best_validation_total'))} | "
            f"{training.get('best_validation_step') or '—'} | {_fmt(amplitude, 6)} | "
            f"{'PASS' if style is True else ('FAIL' if style is False else quality.get('status'))} | "
            f"{'PASS' if padding is True else ('FAIL' if padding is False else quality.get('status'))} |"
        )
    lines.extend(
        [
            "",
            "Generator quality is dataset-free and deterministic: temporal amplitude, "
            "style-conditioned generation, and padding invariance are measured from "
            "the exported generator checkpoint. It is not a text-retrieval score.",
            "",
            "### Experimental generator held-out posttrain",
            "",
            "| Run | Contract | Final val total | Test total | Style response RMS | Latent response RMS | Foundation SHA | Pair SHA |",
            "|---|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for name, generator in report["generators"].items():
        metrics = generator.get("held_out_posttrain_metrics") or {}
        final_validation = metrics.get("final_validation") or {}
        test = metrics.get("test") or {}
        response = metrics.get("final_condition_response") or {}
        quality_style = (
            (generator.get("generator_quality") or {})
            .get("style_condition_response", {})
            .get("direct_flow_rmse")
        )
        hashes = generator.get("lineage_hashes") or {}
        lines.append(
            f"| {name} | {generator.get('checkpoint_contract', '—')} | "
            f"{_fmt(final_validation.get('total'))} | {_fmt(test.get('total'))} | "
            f"{_fmt(quality_style, 6)} | "
            f"{_fmt(response.get('aligned_vs_zero_prediction_rms'), 6)} | "
            f"{str(hashes.get('foundation_checkpoint_sha256') or hashes.get('sha256') or '—')[:12]} | "
            f"{str(hashes.get('pair_contract_sha256') or '—')[:12]} |"
        )
    lines.extend(
        [
            "",
            "B/C metrics use experimental official-metadata text latents and "
            "oracle trajectory-derived style. They are not formal semantic "
            "supervision or a formal-release claim.",
            "",
            "## Held-out text-motion alignment",
            "",
            "| Qwen arm | Split | Groups | Group T→M R@1 | Group T→M R@5 | Group M→T R@1 | Group M→T R@5 | Group retrieval loss | Episode cosine gap |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for role in ("frozen", "finetuned"):
        artifact = report["qwen"][role]
        for split in ("validation", "test"):
            metrics = (artifact.get("metrics") or {}).get(split)
            if not metrics:
                lines.append(
                    f"| {role} | {split} | — | — | — | — | — | — | unavailable |"
                )
                continue
            lines.append(
                f"| {role} | {split} | {int(metrics.get('count', 0)) or '—'} | "
                f"{_fmt(metrics.get('text_to_motion_recall_at_1'))} | "
                f"{_fmt(metrics.get('text_to_motion_recall_at_5'))} | "
                f"{_fmt(metrics.get('motion_to_text_recall_at_1'))} | "
                f"{_fmt(metrics.get('motion_to_text_recall_at_5'))} | "
                f"{_fmt(metrics.get('retrieval_loss'))} | {_fmt(metrics.get('cosine_gap'))} |"
            )
    comparison = report["qwen"]["comparison"]
    lines.extend(
        [
            "",
            "### Frozen → fine-tuned deltas",
            "",
            f"Comparability: **{'PASS' if comparison.get('comparability', {}).get('passed') else comparison.get('status', 'unavailable').upper()}**; "
            f"declared foundation lineage: `{comparison.get('comparability', {}).get('declared_foundation_lineage', 'unavailable')}`.",
            "",
            "| Split | Metric | Frozen | Fine-tuned | Raw delta | Improvement |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    any_delta = False
    for split, split_result in comparison.get("splits", {}).items():
        for key, item in (split_result.get("deltas") or {}).items():
            any_delta = True
            lines.append(
                f"| {split} | {key} | {_fmt(item['frozen'])} | "
                f"{_fmt(item['finetuned'])} | {_fmt(item['delta'])} | "
                f"{_fmt(item['improvement'])} |"
            )
    if not any_delta:
        lines.append("| — | unavailable | — | — | — | — |")
    lines.extend(["", "## Data and condition audits", ""])
    for name, generator in report["generators"].items():
        provenance = generator["provenance_audit"]
        condition = generator["condition_slice_audit"]
        lines.append(f"### {name}")
        lines.append("")
        lines.append(
            f"- Motion-source forbidden-token audit: "
            f"{'PASS' if provenance['motion_training_sources']['passed'] else 'FAIL'}"
        )
        lines.append(
            f"- Conditioning-artifact forbidden-token audit: "
            f"{'PASS' if provenance['conditioning_artifacts']['passed'] else 'FAIL'}"
        )
        lines.append(
            f"- Condition slice audit: "
            f"{'PASS' if condition.get('passed') is True else condition.get('status', 'unavailable').upper()}"
        )
        if condition.get("zero_required_violations"):
            lines.append(
                "- Non-zero forbidden slices: "
                + ", ".join(condition["zero_required_violations"])
            )
        if condition.get("reason"):
            lines.append(f"- Condition audit note: {condition['reason']}")
        hashes = generator.get("lineage_hashes") or {}
        if hashes:
            lines.append(
                "- Lineage hashes: "
                + ", ".join(
                    f"`{key}={value}`" for key, value in hashes.items()
                )
            )
        lines.append("")
    strict_abc = report.get("strict_abc_checkpoint_validation") or {}
    video = report.get("video") or {}
    lines.extend(
        [
            "## Experimental scope and deliverables",
            "",
            f"- Strict A/B/C checkpoint validation: `{strict_abc.get('status', 'unavailable')}`.",
            f"- Held-out side-by-side video: `{video.get('status', 'unavailable')}`"
            + (
                f" ({video.get('case_count')} cases)."
                if video.get("case_count") is not None
                else "."
            ),
            "- B/C use 54 canonical official-metadata prompt groups; this is "
            "experimental metadata alignment, not formal semantic supervision.",
            "- Style controls are oracle-derived from each trajectory and are "
            "held identical for paired B/C comparison.",
            "- “Expression” here means upper-body motion and 3-DoF head "
            "orientation; the 18D representation has no facial blendshapes.",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_report(args: argparse.Namespace) -> dict:
    forbidden_tokens = args.forbidden_source_token or ["kimodo"]
    generators: list[tuple[str, Path]] = []
    if args.foundation_run is not None:
        generators.append(("A_motion_only_foundation", args.foundation_run.resolve()))
    generators.extend(args.generator or [])
    if not generators:
        raise ValueError("provide --foundation-run or at least one --generator NAME=PATH")
    seen = set()
    for name, _ in generators:
        if name in seen:
            raise ValueError(f"duplicate generator name: {name}")
        seen.add(name)
    condition_caches = dict(args.condition_cache or [])
    previous_report = {}
    if args.output_json.is_file():
        try:
            previous_report = _read_json(args.output_json)
        except Exception:
            previous_report = {}
    prior_generators = previous_report.get("generators") or {}
    audit_options = {
        "device": args.audit_device,
        "frames": args.audit_frames,
        "sampling_steps": args.audit_sampling_steps,
        "seeds": args.audit_seeds,
        "style_delta": args.style_delta,
        "padding_tolerance": args.padding_tolerance,
        "min_generation_rms": args.min_generation_rms,
        "min_style_response": args.min_style_response,
    }
    strict_abc = None
    strict_abc_status = {
        "status": "unavailable",
        "reason": "ABC video config was not provided",
    }
    if args.abc_video_config is not None and args.model_audits:
        try:
            strict_abc = _load_strict_experimental_abc(
                args.abc_video_config, device=args.audit_device
            )
            strict_abc_status = {
                "status": "validated",
                "config_path": strict_abc["config_path"],
                "manifest_sha256": strict_abc["manifest_sha256"],
                "branches": {
                    branch: {
                        key: value
                        for key, value in metadata.items()
                        if key
                        in {
                            "path",
                            "sha256",
                            "artifact_kind",
                            "experimental_only",
                            "foundation_checkpoint_sha256",
                            "condition_cache_sha256",
                            "pair_contract_sha256",
                            "initialization_state_sha256",
                            "manifest_sha256",
                        }
                    }
                    for branch, metadata in strict_abc["branches"].items()
                },
            }
        except Exception as exc:
            strict_abc_status = {
                "status": "unavailable",
                "config_path": str(args.abc_video_config),
                "reason": f"{type(exc).__name__}: {exc}",
            }
    generator_results = {
        name: summarize_generator(
            name,
            path,
            condition_cache=condition_caches.get(name),
            forbidden_tokens=forbidden_tokens,
            model_audits=args.model_audits,
            audit_options=audit_options,
            previous=prior_generators.get(name),
            strict_abc=strict_abc,
        )
        for name, path in generators
    }
    foundation_name = generators[0][0]
    checkpoint_to_name = {
        str(Path(generator["checkpoint_path"]).resolve()): name
        for name, generator in generator_results.items()
        if generator.get("checkpoint_path")
    }
    branch_generator = {"A": foundation_name}
    if strict_abc is not None:
        for branch in ("A", "B", "C"):
            checkpoint = strict_abc["branches"][branch]["path"]
            if str(Path(checkpoint).resolve()) in checkpoint_to_name:
                branch_generator[branch] = checkpoint_to_name[
                    str(Path(checkpoint).resolve())
                ]
    frozen = summarize_qwen_artifact(
        "official_frozen", args.qwen_frozen, forbidden_tokens
    )
    finetuned = summarize_qwen_artifact(
        "beat2_only_lora", args.qwen_finetuned, forbidden_tokens
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "generated_at": _utc_now(),
        "experiment_arms": {
            "A": {
                "generator": foundation_name,
                "qwen_state": "none_motion_only",
                "status": generator_results[foundation_name]["availability"],
            },
            "B": {
                "generator": branch_generator.get("B", foundation_name),
                "qwen_state": "official_frozen",
                "status": (
                    generator_results[
                        branch_generator.get("B", foundation_name)
                    ]["availability"]
                    if branch_generator.get("B")
                    else "unavailable"
                ),
            },
            "C": {
                "generator": branch_generator.get("C", foundation_name),
                "qwen_state": "beat2_only_lora",
                "status": (
                    generator_results[
                        branch_generator.get("C", foundation_name)
                    ]["availability"]
                    if branch_generator.get("C")
                    else "unavailable"
                ),
            },
        },
        "generators": generator_results,
        "qwen": {
            "frozen": frozen,
            "finetuned": finetuned,
            "comparison": compare_qwen_artifacts(frozen, finetuned),
        },
        "strict_abc_checkpoint_validation": strict_abc_status,
        "video": summarize_video_artifact(args.video_artifact),
        "report_contract": {
            "generator_quality_separate_from_text_motion_alignment": True,
            "missing_artifacts_fail_closed_as_unavailable": True,
            "read_only_training_monitor": True,
            "formal_validator_not_extended_for_experimental_artifacts": True,
            "experimental_metadata_is_not_formal_semantic_supervision": True,
            "trajectory_style_is_oracle_derived": True,
            "expression_scope": (
                "18D upper-body plus 3-DoF head orientation; no facial "
                "blendshape channels"
            ),
            "forbidden_source_tokens": list(forbidden_tokens),
        },
    }
    return _json_safe(report)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--foundation-run",
        type=Path,
        help="Arm A motion-only foundation run directory",
    )
    parser.add_argument(
        "--generator",
        action="append",
        type=_parse_named_path,
        metavar="NAME=PATH",
        help="Additional generator run to monitor/compare (repeatable)",
    )
    parser.add_argument("--qwen-frozen", type=Path)
    parser.add_argument("--qwen-finetuned", type=Path)
    parser.add_argument(
        "--abc-video-config",
        type=Path,
        help=(
            "Strict A/B/C evaluation config used to validate A formally and "
            "B/C through the independent experimental-only loader"
        ),
    )
    parser.add_argument(
        "--video-artifact",
        type=Path,
        help="Completed ABC video summary.json or its output directory",
    )
    parser.add_argument(
        "--condition-cache",
        action="append",
        type=_parse_named_path,
        metavar="NAME=PATH",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument(
        "--model-audits",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--audit-device", default="cpu")
    parser.add_argument("--audit-frames", type=int, default=48)
    parser.add_argument("--audit-sampling-steps", type=int, default=4)
    parser.add_argument(
        "--audit-seeds",
        type=lambda value: tuple(int(item) for item in value.split(",") if item),
        default=(7, 17),
    )
    parser.add_argument("--style-delta", type=float, default=2.0)
    parser.add_argument("--padding-tolerance", type=float, default=1e-6)
    parser.add_argument("--min-generation-rms", type=float, default=0.01)
    parser.add_argument("--min-style-response", type=float, default=1e-5)
    parser.add_argument(
        "--forbidden-source-token",
        action="append",
        help="Case-insensitive provenance substring; defaults to kimodo",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when a completed audit fails (unavailable is reported, not failed)",
    )
    parser.add_argument(
        "--wait-for-artifacts",
        action="store_true",
        help=(
            "Wait for terminal A/B/C, Qwen comparison, and video artifacts, "
            "then generate this report exactly once"
        ),
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-hours", type=float, default=18.0)
    return parser


def wait_for_terminal_artifacts(
    args: argparse.Namespace, *, poll_seconds: float, timeout_hours: float
) -> None:
    """Wait for immutable terminal receipts, never for live checkpoints alone."""
    if args.abc_video_config is None or args.video_artifact is None:
        raise ValueError(
            "--wait-for-artifacts requires --abc-video-config and --video-artifact"
        )
    if not math.isfinite(poll_seconds) or poll_seconds < 1.0:
        raise ValueError("--poll-seconds must be finite and at least 1")
    if not math.isfinite(timeout_hours) or timeout_hours <= 0:
        raise ValueError("--timeout-hours must be finite and positive")
    config_path = args.abc_video_config.expanduser().resolve()
    config = _read_json(config_path)
    branches = config.get("branches")
    if not isinstance(branches, Mapping) or set(branches) != {"A", "B", "C"}:
        raise ValueError("ABC video config must define exactly A/B/C branches")
    required: list[tuple[str, Path]] = []
    for branch in ("A", "B", "C"):
        record = branches[branch]
        if not isinstance(record, Mapping):
            raise ValueError(f"ABC video branch {branch} is invalid")
        for field in ("checkpoint", "completion_summary"):
            raw = record.get(field)
            if not isinstance(raw, str) or not raw:
                raise ValueError(f"ABC video branches.{branch}.{field} is missing")
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = config_path.parent / path
            required.append((f"{branch}.{field}", path.resolve()))
    for label, raw in (
        ("qwen_frozen", args.qwen_frozen),
        ("qwen_finetuned", args.qwen_finetuned),
    ):
        if raw is None:
            raise ValueError(f"--wait-for-artifacts requires --{label.replace('_', '-')}")
        path = raw.expanduser().resolve()
        if path.is_dir():
            path = path / "comparison.json"
        required.append((label, path))
    video = args.video_artifact.expanduser().resolve()
    if video.is_dir() or video.suffix.lower() != ".json":
        video = video / "summary.json"
    required.append(("video_summary", video))

    deadline = time.monotonic() + timeout_hours * 3600.0
    while True:
        missing = [(label, path) for label, path in required if not path.is_file()]
        if not missing:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "timed out waiting for terminal artifacts: "
                + ", ".join(f"{label}={path}" for label, path in missing)
            )
        print(
            json.dumps(
                {
                    "status": "waiting_for_terminal_morning_report_artifacts",
                    "missing": [label for label, _ in missing],
                    "checked_at": _utc_now(),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        time.sleep(poll_seconds)


def _terminal_report_complete(report: Mapping[str, Any]) -> tuple[bool, list[str]]:
    missing = []
    if (
        (report.get("strict_abc_checkpoint_validation") or {}).get("status")
        != "validated"
    ):
        missing.append("strict_abc_checkpoint_validation")
    for role in ("frozen", "finetuned"):
        if ((report.get("qwen") or {}).get(role) or {}).get("status") != "available":
            missing.append(f"qwen.{role}")
    if (report.get("video") or {}).get("status") != "complete":
        missing.append("video")
    for branch in ("A", "B", "C"):
        arm = (report.get("experiment_arms") or {}).get(branch) or {}
        generator = (report.get("generators") or {}).get(arm.get("generator")) or {}
        if (
            arm.get("status") != "available"
            or generator.get("lifecycle") != "completed"
            or (generator.get("generator_quality") or {}).get("status")
            != "completed"
        ):
            missing.append(f"generator.{branch}")
    return not missing, missing


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    args.output_json = args.output_json.expanduser().resolve()
    args.output_md = (
        args.output_md.expanduser().resolve()
        if args.output_md
        else args.output_json.with_suffix(".md")
    )
    if args.audit_frames < 4 or args.audit_sampling_steps < 1 or not args.audit_seeds:
        parser.error("model audit requires frames>=4, sampling steps>=1, and seeds")
    if args.wait_for_artifacts:
        try:
            wait_for_terminal_artifacts(
                args,
                poll_seconds=args.poll_seconds,
                timeout_hours=args.timeout_hours,
            )
        except (OSError, ValueError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"morning-report wait failed closed: {exc}", file=sys.stderr)
            return 3
    try:
        report = build_report(args)
    except ValueError as exc:
        parser.error(str(exc))
    _atomic_write(
        args.output_json,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
    )
    _atomic_write(args.output_md, render_markdown(report))
    complete, incomplete_fields = _terminal_report_complete(report)
    print(
        json.dumps(
            {
                "artifact_kind": ARTIFACT_KIND,
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
                "foundation_state": next(iter(report["generators"].values()))[
                    "lifecycle"
                ],
                "qwen_comparison": report["qwen"]["comparison"]["status"],
                "terminal_report_complete": complete,
                "incomplete_fields": incomplete_fields,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.strict:
        failures = []
        for name, generator in report["generators"].items():
            for field in ("generator_quality", "condition_slice_audit"):
                check = generator[field]
                if check.get("status") == "completed" and check.get("passed") is False:
                    failures.append(f"{name}.{field}")
            if not generator["provenance_audit"]["motion_training_sources"]["passed"]:
                failures.append(f"{name}.motion_provenance")
            if not generator["provenance_audit"]["conditioning_artifacts"]["passed"]:
                failures.append(f"{name}.conditioning_provenance")
            if (
                generator.get("checkpoint_path")
                and not generator["provenance_audit"][
                    "strict_checkpoint_contract"
                ]["passed"]
            ):
                failures.append(f"{name}.checkpoint_contract")
        if failures:
            print("failed checks: " + ", ".join(failures), file=sys.stderr)
            return 2
    if args.wait_for_artifacts and not complete:
        print(
            "terminal artifacts failed strict report validation: "
            + ", ".join(incomplete_fields),
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
