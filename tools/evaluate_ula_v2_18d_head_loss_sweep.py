#!/usr/bin/env python3
"""Select a head-loss smoke candidate using physical and spectral metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from upper_body_skeleton.ula_v2_18d_head import (
    ACTION_DIM,
    LEGACY_ACTION_DIM,
    STYLE_CONTROL_SLICE,
    assess_body_compatibility,
    body_sampling_drift_metrics,
    load_condition_cache,
    load_contract_checkpoint,
    nonzero_head_forward_drift_metrics,
    read_joint_csv,
    sample_contract_trajectory,
    sha256_file,
    validate_condition_cache_for_generator,
    write_contract_csv,
)


DEFAULT_CLIP_IDS = (
    "24_kexin_2_1_1_f000720-000900",
    "24_kexin_3_1_1_f017460-017640",
    "24_kexin_2_11_11_f000720-000900",
)
NONORACLE_MODE = "text_default_style_nonoracle"
ORACLE_MODE = "oracle_trajectory_style_controls"


def resample(values: np.ndarray, frames: int) -> np.ndarray:
    source = np.linspace(0.0, 1.0, values.shape[0])
    target = np.linspace(0.0, 1.0, int(frames))
    return np.stack(
        [np.interp(target, source, values[:, index]) for index in range(values.shape[1])],
        axis=-1,
    ).astype(np.float32)


def head_motion_metrics(values: np.ndarray, *, fps: float) -> dict:
    head = np.asarray(values[:, LEGACY_ACTION_DIM:], dtype=np.float64)
    centered = head - head.mean(axis=0, keepdims=True)
    spectrum = np.fft.rfft(centered, axis=0)
    frequencies = np.fft.rfftfreq(head.shape[0], d=1.0 / float(fps))
    power = np.square(np.abs(spectrum))
    non_dc = frequencies > 0
    high = frequencies > 5.0
    denominator = float(power[non_dc].sum())
    high_fraction = float(power[high].sum() / denominator) if denominator > 0 else 0.0
    high_spectrum = spectrum.copy()
    high_spectrum[~high] = 0
    high_signal = np.fft.irfft(high_spectrum, n=head.shape[0], axis=0)
    dt = 1.0 / float(fps)
    jerk = np.diff(head, n=3, axis=0) / (dt**3)
    velocity = np.diff(head, axis=0) / dt
    acceleration = np.diff(head, n=2, axis=0) / (dt**2)
    ranges = np.ptp(head, axis=0)
    return {
        "high_frequency_energy_fraction_gt_5hz": high_fraction,
        "high_frequency_rms_rad_gt_5hz": float(np.sqrt(np.mean(np.square(high_signal)))),
        "jerk_rms_rad_s3": float(np.sqrt(np.mean(np.square(jerk)))) if len(jerk) else 0.0,
        "velocity_rms_rad_s": float(np.sqrt(np.mean(np.square(velocity)))),
        "acceleration_rms_rad_s2": float(
            np.sqrt(np.mean(np.square(acceleration)))
        ),
        "peak_abs_acceleration_rad_s2": float(np.max(np.abs(acceleration))),
        "range_norm_rad": float(np.linalg.norm(ranges)),
        "axis_range_rad": [float(value) for value in ranges],
    }


def _candidate_lineage(checkpoint: dict, baseline: Path, baseline_sha256: str) -> None:
    source = checkpoint.get("posttrain_source") or {}
    contract = checkpoint.get("training_contract") or {}
    if source.get("checkpoint_sha256") != baseline_sha256:
        raise ValueError("head smoke candidate does not descend from the sweep baseline")
    if Path(source.get("checkpoint", "")).resolve() != baseline.resolve():
        raise ValueError("head smoke candidate source checkpoint path changed")
    if contract.get("training_policy") != "head_projection_only":
        raise ValueError("head smoke candidate is not projection-only")
    if contract.get("only_new_projection_slices_trainable") is not True:
        raise ValueError("head smoke candidate lacks its frozen-prefix contract")
    if (
        checkpoint.get("training_scope") != "head_mechanism_experiment_only"
        or checkpoint.get("formal_training_enabled") is not False
        or checkpoint.get("temporal_unit_policy") != "fixed_window_experimental"
        or checkpoint.get("formal_release_eligible") is not False
    ):
        raise ValueError("head smoke candidate is not isolated as experimental")


def _read_manifest(path: Path) -> dict[str, dict]:
    records = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                records[str(record["clip_id"])] = record
    return records


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--clip-id", action="append", dest="clip_ids")
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--sampling-steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("audio_policy") != "disabled_not_loaded":
        raise ValueError("evaluation requires audio_policy=disabled_not_loaded")
    clip_ids = tuple(args.clip_ids or DEFAULT_CLIP_IDS)
    baseline_path = Path(config["initial_checkpoint"]).resolve()
    baseline_sha256 = sha256_file(baseline_path)
    cache_path = Path(config["condition_cache"]).resolve()
    ids, prompts, conditions, provenance = load_condition_cache(cache_path)
    baseline_model, baseline_checkpoint = load_contract_checkpoint(
        baseline_path, expected_action_dim=ACTION_DIM, device=args.device
    )
    validate_condition_cache_for_generator(
        baseline_checkpoint,
        provenance,
        generator_checkpoint_path=baseline_path,
    )
    id_to_index = {clip_id: index for index, clip_id in enumerate(ids)}
    missing = sorted(set(clip_ids) - set(id_to_index))
    if missing:
        raise ValueError(f"held-out clips missing from condition cache: {missing}")
    manifest = _read_manifest(Path(config["beat_manifest"]))
    output_root = Path(config["output_root"])
    evaluation_root = output_root / "sampling_evaluation"
    evaluation_root.mkdir(parents=True, exist_ok=True)

    checkpoints = {"untrained_migrated": baseline_path}
    checkpoints.update(
        {
            name: output_root / name / "ula_fm_checkpoint.pt"
            for name in config["candidates"]
        }
    )
    reference_metrics = {}
    references = {}
    for clip_id in clip_ids:
        values = read_joint_csv(Path(manifest[clip_id]["trajectory_path"]))
        values = resample(values, args.frames)
        references[clip_id] = values
        reference_metrics[clip_id] = head_motion_metrics(values, fps=args.fps)
        write_contract_csv(
            evaluation_root / f"{clip_id}_reference.csv", values, fps=args.fps
        )

    staged_config = json.loads(
        Path(config["staged_config"]).read_text(encoding="utf-8")
    )
    challenge_path = (
        Path(staged_config["output_root"]) / "temporal_quarantine_eval_challenges.jsonl"
    )
    challenge_ids = (
        set(_read_manifest(challenge_path)) if challenge_path.is_file() else set()
    )
    challenged_selection = sorted(set(clip_ids) & challenge_ids)
    if challenged_selection:
        raise ValueError(
            f"head selection clips include temporal challenges: {challenged_selection}"
        )

    condition_modes = {ORACLE_MODE: {}, NONORACLE_MODE: {}}
    for clip_id in clip_ids:
        condition = np.asarray(conditions[id_to_index[clip_id]], dtype=np.float32)
        condition_modes[ORACLE_MODE][clip_id] = condition.copy()
        nonoracle = condition.copy()
        nonoracle[STYLE_CONTROL_SLICE] = 0.0
        condition_modes[NONORACLE_MODE][clip_id] = nonoracle

    results = {}
    candidate_models = {}
    split_sha256 = None
    aggregate_keys = (
        "high_frequency_energy_fraction_gt_5hz",
        "high_frequency_rms_rad_gt_5hz",
        "velocity_rms_rad_s",
        "acceleration_rms_rad_s2",
        "peak_abs_acceleration_rad_s2",
        "jerk_rms_rad_s3",
        "range_ratio_to_reference",
        "range_relative_error",
    )
    for name, checkpoint_path in checkpoints.items():
        model, checkpoint = load_contract_checkpoint(
            checkpoint_path, expected_action_dim=ACTION_DIM, device=args.device
        )
        if name != "untrained_migrated":
            _candidate_lineage(checkpoint, baseline_path, baseline_sha256)
            split_contract = checkpoint.get("posttrain_split_contract") or {}
            current_split = split_contract.get("sha256")
            split_sha256 = split_sha256 or current_split
            if not current_split or current_split != split_sha256:
                raise ValueError("head smoke candidates do not share an identical split")
            assignment = {
                row["clip_id"]: row["split"]
                for row in split_contract.get("episodes") or []
            }
            if any(assignment.get(clip_id) != "test" for clip_id in clip_ids):
                raise ValueError("head selection clips are not all in the held-out test split")
            candidate_models[name] = (model, checkpoint)
        mode_results = {}
        for mode, mode_conditions in condition_modes.items():
            per_clip = {}
            for clip_id in clip_ids:
                index = id_to_index[clip_id]
                trajectory = sample_contract_trajectory(
                    model,
                    mode_conditions[clip_id],
                    frames=args.frames,
                    steps=args.sampling_steps,
                    seed=args.seed,
                    device=args.device,
                )
                metrics = head_motion_metrics(trajectory, fps=args.fps)
                reference = reference_metrics[clip_id]
                metrics["range_ratio_to_reference"] = metrics["range_norm_rad"] / max(
                    reference["range_norm_rad"], 1e-8
                )
                metrics["range_relative_error"] = abs(
                    metrics["range_ratio_to_reference"] - 1.0
                )
                per_clip[clip_id] = {
                    "prompt": prompts[index],
                    "seed": args.seed,
                    "sampling_steps": args.sampling_steps,
                    "condition_policy": mode,
                    "metrics": metrics,
                }
                write_contract_csv(
                    evaluation_root / f"{clip_id}_{name}_{mode}.csv",
                    trajectory,
                    fps=args.fps,
                )
            mode_results[mode] = {
                "per_clip": per_clip,
                "aggregate": {
                    key: float(
                        np.mean([row["metrics"][key] for row in per_clip.values()])
                    )
                    for key in aggregate_keys
                },
            }
        results[name] = {
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "modes": mode_results,
        }

    base_15d_path = Path(baseline_checkpoint["migration_source"]["path"])
    base_15d_model, _ = load_contract_checkpoint(
        base_15d_path, expected_action_dim=LEGACY_ACTION_DIM, device=args.device
    )
    for name, (model, checkpoint) in candidate_models.items():
        forward = nonzero_head_forward_drift_metrics(
            base_15d_model,
            model,
            seed=args.seed + 2003,
            device=args.device,
        )
        for mode, mode_conditions in condition_modes.items():
            drift = body_sampling_drift_metrics(
                base_15d_model,
                model,
                [mode_conditions[clip_id] for clip_id in clip_ids],
                action_stats=checkpoint["action_stats"],
                frames=args.frames,
                steps=args.sampling_steps,
                seeds=(args.seed, args.seed + 1009),
                device=args.device,
            )
            results[name]["modes"][mode]["body_drift"] = drift
            results[name]["modes"][mode]["body_compatibility"] = (
                assess_body_compatibility(forward, drift)
            )

    reference_aggregate = {
        key: float(np.mean([row[key] for row in reference_metrics.values()]))
        for key in (
            "high_frequency_energy_fraction_gt_5hz",
            "high_frequency_rms_rad_gt_5hz",
            "velocity_rms_rad_s",
            "acceleration_rms_rad_s2",
            "peak_abs_acceleration_rad_s2",
            "jerk_rms_rad_s3",
            "range_norm_rad",
        )
    }
    hard_gate = config["selection_hard_gates"]
    relative_limit = float(hard_gate["maximum_ratio_to_reference"])
    absolute_limits = hard_gate["absolute_maximums"]
    metric_limits = {
        "high_frequency_rms_rad_gt_5hz": max(
            relative_limit * reference_aggregate["high_frequency_rms_rad_gt_5hz"],
            math.radians(float(absolute_limits["high_frequency_rms_degrees"])),
        ),
        "velocity_rms_rad_s": max(
            relative_limit * reference_aggregate["velocity_rms_rad_s"],
            float(absolute_limits["velocity_rms_rad_s"]),
        ),
        "acceleration_rms_rad_s2": max(
            relative_limit * reference_aggregate["acceleration_rms_rad_s2"],
            float(absolute_limits["acceleration_rms_rad_s2"]),
        ),
        "jerk_rms_rad_s3": max(
            relative_limit * reference_aggregate["jerk_rms_rad_s3"],
            float(absolute_limits["jerk_rms_rad_s3"]),
        ),
    }
    ranked = []
    for name in config["candidates"]:
        mode_result = results[name]["modes"][NONORACLE_MODE]
        aggregate = mode_result["aggregate"]
        compatibility = mode_result["body_compatibility"]
        checks = {
            f"maximum_{metric}": all(
                row["metrics"][metric] <= metric_limits[metric]
                for row in mode_result["per_clip"].values()
            )
            for metric in metric_limits
        }
        checks["peak_abs_acceleration_rad_s2"] = all(
            row["metrics"]["peak_abs_acceleration_rad_s2"]
            <= float(absolute_limits["peak_abs_acceleration_rad_s2"])
            for row in mode_result["per_clip"].values()
        )
        checks["minimum_range_ratio"] = all(
            row["metrics"]["range_ratio_to_reference"]
            >= float(hard_gate["minimum_range_ratio_to_reference"])
            for row in mode_result["per_clip"].values()
        )
        checks["maximum_range_ratio"] = all(
            row["metrics"]["range_ratio_to_reference"]
            <= float(hard_gate["maximum_range_ratio_to_reference"])
            for row in mode_result["per_clip"].values()
        )
        checks["body_compatibility"] = compatibility["passed"]
        spectral_match = abs(
            math.log(
                (aggregate["high_frequency_rms_rad_gt_5hz"] + 1e-8)
                / (reference_aggregate["high_frequency_rms_rad_gt_5hz"] + 1e-8)
            )
        )
        jerk_match = abs(
            math.log(
                (aggregate["jerk_rms_rad_s3"] + 1e-8)
                / (reference_aggregate["jerk_rms_rad_s3"] + 1e-8)
            )
        )
        score = spectral_match + jerk_match + aggregate["range_relative_error"]
        ranked.append(
            {
                "name": name,
                "accepted": all(checks.values()),
                "hard_gate_checks": checks,
                "body_compatibility_passed": compatibility["passed"],
                "selection_score": float(score),
                "spectral_log_error": float(spectral_match),
                "jerk_log_error": float(jerk_match),
                "range_relative_error": aggregate["range_relative_error"],
            }
        )
    ranked.sort(key=lambda row: (not row["accepted"], row["selection_score"]))
    accepted = [row for row in ranked if row["accepted"]]
    selected = accepted[0]["name"] if accepted else None
    report = {
        "schema_version": 1,
        "audio_policy": "disabled_not_loaded",
        "held_out_clip_ids": list(clip_ids),
        "held_out_split": "test_speaker_24_kexin",
        "seed": args.seed,
        "frames": args.frames,
        "fps": args.fps,
        "sampling_steps": args.sampling_steps,
        "split_contract_sha256": split_sha256,
        "training_scope": "head_mechanism_experiment_only",
        "formal_training_enabled": False,
        "selection_condition_mode": NONORACLE_MODE,
        "oracle_condition_mode_role": "diagnostic_only_never_used_for_selection",
        "selection_hard_gate_limits": {
            "per_clip_metric_limits": metric_limits,
            **hard_gate,
        },
        "reference_aggregate": reference_aggregate,
        "results": results,
        "ranking": ranked,
        "selected_candidate": selected,
        "selection_policy": (
            "accept only non-oracle 24-step samples passing head velocity, acceleration, "
            "jerk, high-frequency, amplitude, and body-drift hard gates; then minimize "
            "spectral log error + physical jerk log error + head range relative error"
        ),
    }
    report_path = output_root / "head_loss_selection_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown = [
        "# ULA V2 18D head loss sweep",
        "",
        "Audio is disabled and not loaded. This is a fixed-window head-mechanism experiment, not formal variable-length training. Selection uses text/default-style non-oracle conditions; target-derived style controls are diagnostic only.",
        "",
        "| candidate | >5 Hz RMS rad | jerk RMS rad/s^3 | range error | body mean drift rad | body gate | score |",
        "|---|---:|---:|---:|---:|:---:|---:|",
    ]
    for row in ranked:
        result = results[row["name"]]["modes"][NONORACLE_MODE]
        aggregate = result["aggregate"]
        markdown.append(
            f"| {row['name']} | {aggregate['high_frequency_rms_rad_gt_5hz']:.6f} | "
            f"{aggregate['jerk_rms_rad_s3']:.3f} | "
            f"{aggregate['range_relative_error']:.3f} | "
            f"{result['body_drift']['body_mean_abs_rad']:.6f} | "
            f"{'pass' if row['body_compatibility_passed'] else 'fail'} | "
            f"{row['selection_score']:.3f} |"
        )
    markdown.extend(
        [
            "",
            f"Selected: `{selected}`." if selected else "Selected: none (hard gate failed).",
            "",
            f"Reference >5 Hz RMS: `{reference_aggregate['high_frequency_rms_rad_gt_5hz']:.6f}` rad.",
            f"Reference jerk RMS: `{reference_aggregate['jerk_rms_rad_s3']:.3f}` rad/s^3.",
        ]
    )
    (output_root / "head_loss_selection_report.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    print(json.dumps({"selected_candidate": selected, "ranking": ranked}, indent=2))
    if selected is None:
        raise RuntimeError("no head-loss candidate passed the non-oracle physical hard gate")


if __name__ == "__main__":
    main()
