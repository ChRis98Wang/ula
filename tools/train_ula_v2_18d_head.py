#!/usr/bin/env python3
"""Migrate, condition, train, and sample the ULA V2 18D head adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from upper_body_skeleton.ula_v2_18d_head import (
    DEFAULT_BASE_CHECKPOINT,
    DEFAULT_QWEN_CHECKPOINT,
    attach_condition_cache,
    benchmark_contract_inference,
    benchmark_text_to_trajectory_inference,
    build_condition_cache,
    compute_18d_action_stats,
    load_18d_episodes,
    load_condition_cache,
    load_contract_checkpoint,
    migrate_15d_checkpoint,
    predict_contract_frame_count,
    sample_contract_trajectory,
    train_head_adapter,
    validate_condition_cache_for_generator,
    validate_qwen_checkpoint_for_generator,
    write_contract_csv,
    write_contract_npz,
)


def add_data_arguments(parser):
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--motion-root", type=Path)
    parser.add_argument("--semantics", type=Path)
    parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="Allow machine-labeled pending-review clips for an explicit smoke test.",
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache = subparsers.add_parser("cache-conditions")
    add_data_arguments(cache)
    cache.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    cache.add_argument("--qwen-checkpoint", type=Path, default=DEFAULT_QWEN_CHECKPOINT)
    cache.add_argument("--output", type=Path, required=True)
    cache.add_argument("--device", default="auto")
    cache.add_argument("--batch-size", type=int, default=16)
    cache.add_argument("--allow-download", action="store_true")

    migrate = subparsers.add_parser("migrate")
    add_data_arguments(migrate)
    migrate.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    migrate.add_argument("--output", type=Path, required=True)

    train = subparsers.add_parser("train")
    add_data_arguments(train)
    train.add_argument("--condition-cache", type=Path, required=True)
    train.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--steps", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--frames", type=int, default=48)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--body-distillation-weight", type=float, default=1.0)
    train.add_argument("--device", default="auto")
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--log-interval", type=int, default=5)
    train.add_argument(
        "--allow-unsafe-condition-cache",
        action="store_true",
        help="Permit a cache without its versioned metadata only for an explicit smoke test.",
    )

    infer = subparsers.add_parser("infer")
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument(
        "--expected-action-dim",
        type=int,
        choices=(15, 18),
        help="Optional deployment guard; auto-detects a valid 15D/18D contract when omitted.",
    )
    infer.add_argument("--condition-cache", type=Path, required=True)
    infer.add_argument("--clip-id", required=True)
    infer.add_argument("--prompt")
    infer.add_argument("--output-dir", type=Path, required=True)
    infer.add_argument(
        "--frames",
        type=int,
        help="Diagnostic override; default uses the checkpoint's learned native duration.",
    )
    infer.add_argument("--sampling-steps", type=int, default=24)
    infer.add_argument("--fps", type=float, default=30.0)
    infer.add_argument("--seed", type=int, default=7)
    infer.add_argument("--device", default="auto")
    infer.add_argument("--allow-unsafe-condition-cache", action="store_true")

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--checkpoint", type=Path, required=True)
    benchmark.add_argument("--condition-cache", type=Path, required=True)
    benchmark.add_argument("--clip-id", required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--frames", type=int, default=90)
    benchmark.add_argument("--sampling-steps", type=int, default=24)
    benchmark.add_argument("--warmup", type=int, default=2)
    benchmark.add_argument("--repeats", type=int, default=5)
    benchmark.add_argument("--seed", type=int, default=7)
    benchmark.add_argument("--device", default="auto")
    benchmark.add_argument("--allow-unsafe-condition-cache", action="store_true")

    benchmark_e2e = subparsers.add_parser("benchmark-e2e")
    benchmark_e2e.add_argument("--checkpoint", type=Path, required=True)
    benchmark_e2e.add_argument("--qwen-checkpoint", type=Path, default=DEFAULT_QWEN_CHECKPOINT)
    benchmark_e2e.add_argument("--prompt", required=True)
    benchmark_e2e.add_argument("--output", type=Path, required=True)
    benchmark_e2e.add_argument("--frames", type=int, default=90)
    benchmark_e2e.add_argument("--sampling-steps", type=int, default=24)
    benchmark_e2e.add_argument("--warmup", type=int, default=2)
    benchmark_e2e.add_argument("--repeats", type=int, default=5)
    benchmark_e2e.add_argument("--seed", type=int, default=7)
    benchmark_e2e.add_argument("--device", default="auto")
    benchmark_e2e.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def load_data(args):
    return load_18d_episodes(
        manifest=args.manifest,
        motion_root=args.motion_root,
        semantics=args.semantics,
        allow_unreviewed=args.allow_unreviewed,
    )


def main():
    args = parse_args()
    if args.command == "cache-conditions":
        episodes = load_data(args)
        result = build_condition_cache(
            episodes,
            args.qwen_checkpoint,
            args.output,
            base_checkpoint=args.base_checkpoint,
            device=args.device,
            local_files_only=not args.allow_download,
            batch_size=args.batch_size,
        )
    elif args.command == "migrate":
        episodes = load_data(args)
        _, base = load_contract_checkpoint(args.base_checkpoint, expected_action_dim=15)
        action_stats = compute_18d_action_stats(
            [item["actions"] for item in episodes], base["action_stats"]
        )
        _, result = migrate_15d_checkpoint(
            args.base_checkpoint, args.output, action_stats=action_stats
        )
        result["checkpoint"] = str(args.output.resolve())
    elif args.command == "train":
        episodes = attach_condition_cache(
            load_data(args),
            args.condition_cache,
            allow_unsafe_metadata=args.allow_unsafe_condition_cache,
        )
        result = train_head_adapter(
            base_checkpoint_path=args.base_checkpoint,
            episodes=episodes,
            output_dir=args.output_dir,
            steps=args.steps,
            batch_size=args.batch_size,
            frames=args.frames,
            lr=args.lr,
            body_distillation_weight=args.body_distillation_weight,
            device=args.device,
            seed=args.seed,
            log_interval=args.log_interval,
            require_train_ready=not args.allow_unreviewed,
            allow_unsafe_condition_cache=args.allow_unsafe_condition_cache,
        )
    elif args.command == "benchmark-e2e":
        import torch

        resolved_device = args.device
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        model, checkpoint = load_contract_checkpoint(
            args.checkpoint, expected_action_dim=18, device=resolved_device
        )
        validate_qwen_checkpoint_for_generator(checkpoint, args.qwen_checkpoint)
        result, _ = benchmark_text_to_trajectory_inference(
            model,
            args.prompt,
            args.qwen_checkpoint,
            frames=args.frames,
            steps=args.sampling_steps,
            warmup=args.warmup,
            repeats=args.repeats,
            seed=args.seed,
            device=resolved_device,
            local_files_only=not args.allow_download,
        )
        result["checkpoint"] = str(args.checkpoint.resolve())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif args.command in {"infer", "benchmark"}:
        import numpy as np
        import torch

        resolved_device = args.device
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        model, checkpoint = load_contract_checkpoint(
            args.checkpoint,
            expected_action_dim=getattr(args, "expected_action_dim", None),
            device=resolved_device,
        )
        ids, prompts, conditions, cache_provenance = load_condition_cache(
            args.condition_cache,
            allow_unsafe_metadata=args.allow_unsafe_condition_cache,
        )
        validate_condition_cache_for_generator(
            checkpoint,
            cache_provenance,
            generator_checkpoint_path=args.checkpoint,
            allow_unsafe=args.allow_unsafe_condition_cache,
        )
        if args.clip_id not in ids:
            raise ValueError(f"condition cache does not contain {args.clip_id!r}")
        index = ids.index(args.clip_id)
        condition = conditions[index].astype(np.float32)
        prompt = prompts[index]
        requested_prompt = getattr(args, "prompt", None)
        if requested_prompt is not None and requested_prompt != prompt:
            raise ValueError(
                "--prompt cannot relabel a cached condition; rebuild the cache or use "
                "benchmark-e2e for a new prompt"
            )
        if args.command == "benchmark":
            result, _ = benchmark_contract_inference(
                model,
                condition,
                frames=args.frames,
                steps=args.sampling_steps,
                warmup=args.warmup,
                repeats=args.repeats,
                seed=args.seed,
                device=resolved_device,
            )
            result.update(
                {
                    "checkpoint": str(args.checkpoint.resolve()),
                    "clip_id": args.clip_id,
                    "prompt": prompt,
                }
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            predicted_duration_sec = None
            generation_length_policy = "explicit_diagnostic_frame_override"
            frames = args.frames
            if frames is None:
                frames, predicted_duration_sec = predict_contract_frame_count(
                    model,
                    checkpoint,
                    condition,
                    fps=args.fps,
                    device=resolved_device,
                )
                generation_length_policy = (
                    "duration_head_complete_expression_arc_no_reference_frames"
                )
            trajectory = sample_contract_trajectory(
                model,
                condition,
                frames=frames,
                steps=args.sampling_steps,
                seed=args.seed,
                device=resolved_device,
            )
            args.output_dir.mkdir(parents=True, exist_ok=True)
            action_dim = int(checkpoint["action_dim"])
            csv_path = args.output_dir / f"{args.clip_id}_generated_{action_dim}d.csv"
            npz_path = args.output_dir / f"{args.clip_id}_generated_{action_dim}d.npz"
            write_contract_csv(csv_path, trajectory, fps=args.fps)
            write_contract_npz(
                npz_path,
                trajectory,
                fps=args.fps,
                prompt=prompt,
                checkpoint_path=args.checkpoint,
            )
            result = {
                "checkpoint": str(args.checkpoint.resolve()),
                "action_contract": checkpoint.get("action_contract"),
                "clip_id": args.clip_id,
                "prompt": prompt,
                "frames": int(trajectory.shape[0]),
                "sample_span_sec": float((trajectory.shape[0] - 1) / args.fps),
                "predicted_duration_sec": predicted_duration_sec,
                "generation_length_policy": generation_length_policy,
                "action_dim": action_dim,
                "csv": str(csv_path.resolve()),
                "npz": str(npz_path.resolve()),
            }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
