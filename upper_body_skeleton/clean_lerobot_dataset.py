#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from upper_body_skeleton.lerobot_export import finish_stats, initial_stats, update_stats


DEFAULT_MAX_VELOCITY_RAD_S = 6.0
DEFAULT_SPIKE_DELTA_RAD = 0.35
DEFAULT_SPIKE_NEIGHBOR_RATIO = 2.5
DEFAULT_SMOOTHING_WINDOW = 3
DEFAULT_MIN_ENERGY_RATIO = 0.85


def trajectory_quality(trajectory, fps=30.0):
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.shape[0] < 2:
        return {
            "frames": int(trajectory.shape[0]),
            "max_delta_rad": 0.0,
            "p99_delta_rad": 0.0,
            "max_velocity_rad_s": 0.0,
            "p99_velocity_rad_s": 0.0,
            "mean_velocity_rad_s": 0.0,
            "motion_energy": 0.0,
        }
    delta = np.abs(np.diff(trajectory, axis=0))
    velocity = delta * float(fps)
    return {
        "frames": int(trajectory.shape[0]),
        "max_delta_rad": float(delta.max()),
        "p99_delta_rad": float(np.percentile(delta, 99)),
        "max_velocity_rad_s": float(velocity.max()),
        "p99_velocity_rad_s": float(np.percentile(velocity, 99)),
        "mean_velocity_rad_s": float(velocity.mean()),
        "motion_energy": float(delta.mean()),
    }


def _odd_window(value):
    value = int(value)
    if value <= 1:
        return 1
    return value if value % 2 else value + 1


def replace_single_frame_spikes(trajectory, spike_delta_rad, spike_neighbor_ratio):
    cleaned = np.asarray(trajectory, dtype=np.float64).copy()
    replacements = 0
    if cleaned.shape[0] < 3:
        return cleaned, replacements
    for frame_index in range(1, cleaned.shape[0] - 1):
        previous = cleaned[frame_index - 1]
        current = cleaned[frame_index]
        following = cleaned[frame_index + 1]
        prev_delta = np.abs(current - previous)
        next_delta = np.abs(following - current)
        bridge_delta = np.abs(following - previous)
        spike_mask = (
            (prev_delta > spike_delta_rad)
            & (next_delta > spike_delta_rad)
            & (prev_delta > np.maximum(bridge_delta, 1e-6) * spike_neighbor_ratio)
            & (next_delta > np.maximum(bridge_delta, 1e-6) * spike_neighbor_ratio)
        )
        if np.any(spike_mask):
            cleaned[frame_index, spike_mask] = (previous[spike_mask] + following[spike_mask]) * 0.5
            replacements += int(spike_mask.sum())
    return cleaned, replacements


def limit_velocity(trajectory, fps, max_velocity_rad_s):
    limited = np.asarray(trajectory, dtype=np.float64).copy()
    max_delta = float(max_velocity_rad_s) / float(fps)
    limited_steps = 0
    for frame_index in range(1, limited.shape[0]):
        delta = limited[frame_index] - limited[frame_index - 1]
        clipped = np.clip(delta, -max_delta, max_delta)
        if np.any(np.abs(clipped - delta) > 1e-9):
            limited_steps += int(np.count_nonzero(np.abs(clipped - delta) > 1e-9))
        limited[frame_index] = limited[frame_index - 1] + clipped
    return limited, limited_steps


def smooth_preserving_energy(trajectory, original_energy, window, min_energy_ratio):
    window = _odd_window(window)
    if window <= 1 or trajectory.shape[0] < window:
        return trajectory.copy(), 1.0
    padded = np.pad(trajectory, ((window // 2, window // 2), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    smoothed = np.vstack(
        [np.convolve(padded[:, joint_index], kernel, mode="valid") for joint_index in range(trajectory.shape[1])]
    ).T
    smoothed_energy = trajectory_quality(smoothed)["motion_energy"]
    if original_energy <= 1e-9:
        return smoothed, 1.0
    energy_ratio = smoothed_energy / original_energy
    if energy_ratio >= min_energy_ratio:
        return smoothed, float(energy_ratio)
    blend = np.clip((1.0 - min_energy_ratio) / max(1.0 - energy_ratio, 1e-6), 0.0, 0.95)
    blended = trajectory + blend * (smoothed - trajectory)
    while blend > 1e-4 and trajectory_quality(blended)["motion_energy"] / original_energy < min_energy_ratio:
        blend *= 0.5
        blended = trajectory + blend * (smoothed - trajectory)
    return blended, float(trajectory_quality(blended)["motion_energy"] / original_energy)


def clean_episode_trajectory(
    trajectory,
    *,
    fps=30.0,
    max_velocity_rad_s=DEFAULT_MAX_VELOCITY_RAD_S,
    spike_delta_rad=DEFAULT_SPIKE_DELTA_RAD,
    spike_neighbor_ratio=DEFAULT_SPIKE_NEIGHBOR_RATIO,
    smoothing_window=DEFAULT_SMOOTHING_WINDOW,
    min_energy_ratio=DEFAULT_MIN_ENERGY_RATIO,
):
    original = np.asarray(trajectory, dtype=np.float64)
    before = trajectory_quality(original, fps=fps)
    cleaned, spike_replacements = replace_single_frame_spikes(
        original,
        spike_delta_rad=spike_delta_rad,
        spike_neighbor_ratio=spike_neighbor_ratio,
    )
    cleaned, velocity_limited_steps = limit_velocity(
        cleaned,
        fps=fps,
        max_velocity_rad_s=max_velocity_rad_s,
    )
    valid_before_energy = trajectory_quality(cleaned, fps=fps)["motion_energy"]
    limited_energy = trajectory_quality(cleaned, fps=fps)["motion_energy"]
    cleaned, smooth_energy_ratio = smooth_preserving_energy(
        cleaned,
        original_energy=limited_energy,
        window=smoothing_window,
        min_energy_ratio=min_energy_ratio,
    )
    after = trajectory_quality(cleaned, fps=fps)
    raw_energy_ratio = after["motion_energy"] / before["motion_energy"] if before["motion_energy"] > 1e-9 else 1.0
    valid_energy_ratio = after["motion_energy"] / valid_before_energy if valid_before_energy > 1e-9 else 1.0
    report = {
        "before": before,
        "after": after,
        "spike_replacements": int(spike_replacements),
        "velocity_limited_steps": int(velocity_limited_steps),
        "smooth_energy_ratio": float(smooth_energy_ratio),
        "raw_energy_ratio": float(raw_energy_ratio),
        "valid_energy_ratio": float(valid_energy_ratio),
    }
    return cleaned.astype(np.float32), report


def _copy_static_files(input_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in input_dir.iterdir():
        if child.name in {"data", "meta"}:
            continue
        target = output_dir / child.name
        if child.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)
    meta_in = input_dir / "meta"
    meta_out = output_dir / "meta"
    if meta_out.exists():
        shutil.rmtree(meta_out)
    shutil.copytree(meta_in, meta_out)
    data_out = output_dir / "data"
    if data_out.exists():
        shutil.rmtree(data_out)
    data_out.mkdir(parents=True, exist_ok=True)


def clean_lerobot_dataset(
    input_dir,
    output_dir,
    *,
    max_velocity_rad_s=DEFAULT_MAX_VELOCITY_RAD_S,
    spike_delta_rad=DEFAULT_SPIKE_DELTA_RAD,
    spike_neighbor_ratio=DEFAULT_SPIKE_NEIGHBOR_RATIO,
    smoothing_window=DEFAULT_SMOOTHING_WINDOW,
    min_energy_ratio=DEFAULT_MIN_ENERGY_RATIO,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    _copy_static_files(input_dir, output_dir)
    stats = initial_stats()
    episode_reports = []
    global_before_delta = []
    global_after_delta = []
    global_before_vel = []
    global_after_vel = []

    source_tables = []
    episode_index_rows = {}
    for file_index, source_path in enumerate(sorted((input_dir / "data").glob("chunk-*/*.parquet"))):
        table = pq.read_table(source_path)
        rows = table.to_pylist()
        source_tables.append({"path": source_path, "schema": table.schema, "rows": rows, "cleaned_rows": list(rows)})
        for row_index, row in enumerate(rows):
            episode_index_rows.setdefault(int(row["episode_index"]), []).append((file_index, row_index, row))

    for episode_index, indexed_rows in sorted(episode_index_rows.items()):
        indexed_rows = sorted(indexed_rows, key=lambda item: int(item[2]["frame_index"]))
        original = np.asarray([row["observation.state"] for _, _, row in indexed_rows], dtype=np.float32)
        if original.shape[0] == 0:
            continue
        fps = 30.0
        if original.shape[0] > 1:
            timestamps = [float(row["timestamp"]) for _, _, row in indexed_rows]
            dt = np.diff(timestamps)
            positive = dt[dt > 1e-6]
            if positive.size:
                fps = float(1.0 / np.median(positive))
        cleaned, report = clean_episode_trajectory(
            original,
            fps=fps,
            max_velocity_rad_s=max_velocity_rad_s,
            spike_delta_rad=spike_delta_rad,
            spike_neighbor_ratio=spike_neighbor_ratio,
            smoothing_window=smoothing_window,
            min_energy_ratio=min_energy_ratio,
        )
        if original.shape[0] > 1:
            before_delta = np.abs(np.diff(original.astype(np.float64), axis=0)).reshape(-1)
            after_delta = np.abs(np.diff(cleaned.astype(np.float64), axis=0)).reshape(-1)
            global_before_delta.append(before_delta)
            global_after_delta.append(after_delta)
            global_before_vel.append(before_delta * fps)
            global_after_vel.append(after_delta * fps)
        source_files = set()
        for local_index, (file_index, row_index, row) in enumerate(indexed_rows):
            next_action = cleaned[local_index + 1] if local_index + 1 < cleaned.shape[0] else cleaned[local_index]
            new_row = dict(row)
            new_row["observation.state"] = [float(value) for value in cleaned[local_index]]
            new_row["action"] = [float(value) for value in next_action]
            source_tables[file_index]["cleaned_rows"][row_index] = new_row
            source_files.add(str(source_tables[file_index]["path"]))
            update_stats(stats, new_row["observation.state"], new_row["action"])
        episode_reports.append(
            {
                "episode_index": int(episode_index),
                "source_files": sorted(source_files),
                "spike_replacements": report["spike_replacements"],
                "velocity_limited_steps": report["velocity_limited_steps"],
                "raw_energy_ratio": report["raw_energy_ratio"],
                "valid_energy_ratio": report["valid_energy_ratio"],
                "before_max_delta_rad": report["before"]["max_delta_rad"],
                "after_max_delta_rad": report["after"]["max_delta_rad"],
                "before_motion_energy": report["before"]["motion_energy"],
                "after_motion_energy": report["after"]["motion_energy"],
            }
        )

    for table_info in source_tables:
        relative = table_info["path"].relative_to(input_dir)
        target_path = output_dir / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(table_info["cleaned_rows"], schema=table_info["schema"]),
            target_path,
            compression="zstd",
        )

    before_delta = np.concatenate(global_before_delta) if global_before_delta else np.asarray([], dtype=np.float64)
    after_delta = np.concatenate(global_after_delta) if global_after_delta else np.asarray([], dtype=np.float64)
    before_vel = np.concatenate(global_before_vel) if global_before_vel else np.asarray([], dtype=np.float64)
    after_vel = np.concatenate(global_after_vel) if global_after_vel else np.asarray([], dtype=np.float64)

    def summarize(values):
        if values.size == 0:
            return {"max": 0.0, "p99": 0.0, "p999": 0.0, "mean": 0.0}
        return {
            "max": float(values.max()),
            "p99": float(np.percentile(values, 99)),
            "p999": float(np.percentile(values, 99.9)),
            "mean": float(values.mean()),
        }

    total_spikes = sum(row["spike_replacements"] for row in episode_reports)
    total_limited = sum(row["velocity_limited_steps"] for row in episode_reports)
    raw_energy_ratios = np.asarray([row["raw_energy_ratio"] for row in episode_reports], dtype=np.float64)
    valid_energy_ratios = np.asarray([row["valid_energy_ratio"] for row in episode_reports], dtype=np.float64)
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "episodes": len(episode_reports),
        "max_velocity_rad_s": float(max_velocity_rad_s),
        "spike_delta_rad": float(spike_delta_rad),
        "spike_neighbor_ratio": float(spike_neighbor_ratio),
        "smoothing_window": int(smoothing_window),
        "min_energy_ratio": float(min_energy_ratio),
        "spike_replacements": int(total_spikes),
        "velocity_limited_steps": int(total_limited),
        "raw_energy_ratio_mean": float(raw_energy_ratios.mean()) if raw_energy_ratios.size else 1.0,
        "raw_energy_ratio_p01": float(np.percentile(raw_energy_ratios, 1)) if raw_energy_ratios.size else 1.0,
        "raw_energy_ratio_min": float(raw_energy_ratios.min()) if raw_energy_ratios.size else 1.0,
        "valid_energy_ratio_mean": float(valid_energy_ratios.mean()) if valid_energy_ratios.size else 1.0,
        "valid_energy_ratio_p01": float(np.percentile(valid_energy_ratios, 1)) if valid_energy_ratios.size else 1.0,
        "valid_energy_ratio_min": float(valid_energy_ratios.min()) if valid_energy_ratios.size else 1.0,
        "delta_before": summarize(before_delta),
        "delta_after": summarize(after_delta),
        "velocity_before": summarize(before_vel),
        "velocity_after": summarize(after_vel),
        "episodes_with_velocity_limit": int(sum(row["velocity_limited_steps"] > 0 for row in episode_reports)),
        "episodes_with_spike_replacement": int(sum(row["spike_replacements"] > 0 for row in episode_reports)),
    }

    (output_dir / "meta" / "stats.json").write_text(
        json.dumps(finish_stats(stats), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "cleaning_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "cleaning_episode_report.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in episode_reports),
        encoding="utf-8",
    )
    readme = output_dir / "README.md"
    existing = readme.read_text(encoding="utf-8") if readme.exists() else ""
    readme.write_text(
        existing
        + "\n## Clean v1\n\n"
        + "This copy removes implausible one-frame upper-body joint jumps while preserving most motion energy. "
        + "The original dataset is left unchanged; cleaning statistics are in `cleaning_summary.json` and per-episode details are in `cleaning_episode_report.jsonl`.\n",
        encoding="utf-8",
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Clean LeRobot parquet trajectories without overwriting the source dataset")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-velocity-rad-s", type=float, default=DEFAULT_MAX_VELOCITY_RAD_S)
    parser.add_argument("--spike-delta-rad", type=float, default=DEFAULT_SPIKE_DELTA_RAD)
    parser.add_argument("--spike-neighbor-ratio", type=float, default=DEFAULT_SPIKE_NEIGHBOR_RATIO)
    parser.add_argument("--smoothing-window", type=int, default=DEFAULT_SMOOTHING_WINDOW)
    parser.add_argument("--min-energy-ratio", type=float, default=DEFAULT_MIN_ENERGY_RATIO)
    args = parser.parse_args()
    summary = clean_lerobot_dataset(
        args.input_dir,
        args.output_dir,
        max_velocity_rad_s=args.max_velocity_rad_s,
        spike_delta_rad=args.spike_delta_rad,
        spike_neighbor_ratio=args.spike_neighbor_ratio,
        smoothing_window=args.smoothing_window,
        min_energy_ratio=args.min_energy_ratio,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
