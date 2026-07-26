#!/usr/bin/env python3
"""Measure one full 18D ULA V2 native-length training microbatch on CUDA."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from upper_body_skeleton.ula_training import (  # noqa: E402
    KIMODO_V2_CONDITION_DIM,
    ULA_MMDIT_V2_ARCHITECTURE,
    create_ula_model,
)
from upper_body_skeleton.ula_v2_18d_head import ACTION_DIM  # noqa: E402
from upper_body_skeleton.ula_v2_18d_posttrain import (  # noqa: E402
    masked_18d_objective,
    native_length_microbatch_capacity,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--semantic-tokens", type=int, default=7)
    parser.add_argument("--max-motion-tokens", type=int, default=4096)
    parser.add_argument("--max-attention-elements", type=int, default=8_000_000)
    return parser.parse_args(argv)


def mib(value: int) -> float:
    return round(int(value) / (1024 * 1024), 3)


def memory(device) -> dict:
    return {
        "allocated_mib": mib(torch.cuda.memory_allocated(device)),
        "reserved_mib": mib(torch.cuda.memory_reserved(device)),
        "peak_allocated_mib": mib(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_mib": mib(torch.cuda.max_memory_reserved(device)),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.frames < 3:
        raise ValueError("frames must be at least three")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(20260724)
    torch.cuda.manual_seed_all(20260724)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    plan = native_length_microbatch_capacity(
        args.frames,
        semantic_tokens=args.semantic_tokens,
        max_batch_size=16,
        max_motion_tokens=args.max_motion_tokens,
        max_attention_elements=args.max_attention_elements,
    )
    if args.batch_size > plan["capacity"]:
        raise ValueError(
            f"requested batch-size {args.batch_size} exceeds planned capacity "
            f"{plan['capacity']}"
        )
    report = {
        "artifact_kind": "ula_v2_native_length_cuda_memory_smoke",
        "frames": args.frames,
        "batch_size": args.batch_size,
        "dtype": "float32",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "semantic_tokens": args.semantic_tokens,
        "budget_plan": plan,
        "status": "running",
    }
    model = optimizer = losses = None
    actions = condition = dim_mask = frame_mask = duration = None
    try:
        model = create_ula_model(
            ULA_MMDIT_V2_ARCHITECTURE,
            action_dim=ACTION_DIM,
            condition_dim=KIMODO_V2_CONDITION_DIM,
            hidden_dim=args.hidden_dim,
            layers=args.layers,
            semantic_tokens=args.semantic_tokens,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        torch.cuda.synchronize(device)
        report["after_model"] = memory(device)

        actions = torch.zeros(
            (args.batch_size, args.frames, ACTION_DIM),
            dtype=torch.float32,
            device=device,
        )
        condition = torch.zeros(
            (args.batch_size, KIMODO_V2_CONDITION_DIM),
            dtype=torch.float32,
            device=device,
        )
        dim_mask = torch.ones(
            (args.batch_size, ACTION_DIM), dtype=torch.bool, device=device
        )
        frame_mask = torch.ones(
            (args.batch_size, args.frames), dtype=torch.bool, device=device
        )
        duration = torch.full(
            (args.batch_size,),
            (args.frames - 1) / 30.0,
            dtype=torch.float32,
            device=device,
        )
        losses = masked_18d_objective(
            model,
            actions,
            condition,
            dim_mask,
            duration,
            frame_valid_mask=frame_mask,
            loss_weights={
                "flow": 1.0,
                "position": 0.25,
                "body": 0.0,
                "velocity": 0.01,
                "acceleration": 0.0005,
                "head_flow": 1.0,
                "head_position": 0.5,
                "head_velocity": 0.05,
                "head_acceleration": 0.005,
                "head_jerk": 0.0005,
                "planner_duration": 0.1,
                "planner_transition": 0.0,
            },
            generator=torch.Generator(device=device).manual_seed(17),
        )
        torch.cuda.synchronize(device)
        report["after_forward"] = memory(device)
        losses["total"].backward()
        torch.cuda.synchronize(device)
        report["after_backward"] = memory(device)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize(device)
        report["after_optimizer_step"] = memory(device)
        report["loss"] = float(losses["total"].detach().cpu())
        report["status"] = "passed"
        report["full_episode_preserved"] = actions.shape[1] == args.frames
    except torch.OutOfMemoryError as error:
        report["status"] = "cuda_oom"
        report["error"] = str(error)
        report["at_failure"] = memory(device)
    finally:
        losses = optimizer = model = None
        actions = condition = dim_mask = frame_mask = duration = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        report["after_cleanup"] = memory(device)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
