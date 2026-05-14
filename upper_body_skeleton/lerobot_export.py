#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from upper_body_skeleton.retarget_v2 import JOINT_ORDER


DEFAULT_ROWS_PER_FILE = 250_000
LEROBOT_VERSION = "v3.0"
ROBOT_TYPE = "v2_upper_body_15d"


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_joint_window(csv_path, start_row, end_row):
    rows = []
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for index, row in enumerate(reader):
            if index < start_row:
                continue
            if index >= end_row:
                break
            rows.append([float(row[joint]) for joint in JOINT_ORDER])
    return rows


def task_text(record):
    variants = record.get("language_condition", {}).get("instruction_variants", [])
    if variants:
        return str(variants[0])
    language = record.get("language_condition", {})
    labels = record.get("labels", {})
    return (
        f"{language.get('intent_text', 'conversational expression')}; "
        f"gesture: {record.get('meta_semantics', {}).get('semantic_gesture', 'null')}; "
        f"style: {labels.get('motion_style', 'restrained')}; "
        f"affect: {labels.get('observed_affect', 'neutral')}"
    )


def optional_float(value):
    if value is None:
        return None
    return float(value)


def optional_int(value):
    if value is None:
        return None
    return int(value)


def json_dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def parquet_path(root, kind, file_index):
    chunk_index = file_index // 1000
    return root / kind / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.parquet"


def write_table(root, kind, rows, schema, file_index):
    path = parquet_path(root, kind, file_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")
    return path


def data_schema():
    vector = pa.list_(pa.float32(), list_size=len(JOINT_ORDER))
    return pa.schema(
        [
            pa.field("index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("frame_index", pa.int64()),
            pa.field("timestamp", pa.float32()),
            pa.field("task_index", pa.int64()),
            pa.field("observation.state", vector),
            pa.field("action", vector),
            pa.field("next.done", pa.bool_()),
        ]
    )


def episode_schema():
    return pa.schema(
        [
            pa.field("episode_index", pa.int64()),
            pa.field("task_index", pa.int64()),
            pa.field("length", pa.int64()),
            pa.field("sample_id", pa.string()),
            pa.field("source_dataset", pa.string()),
            pa.field("source_video_path", pa.string()),
            pa.field("source_json_path", pa.string()),
            pa.field("source_npz_path", pa.string()),
            pa.field("source_retarget_csv_path", pa.string()),
            pa.field("start_row", pa.int64()),
            pa.field("end_row", pa.int64()),
            pa.field("start_sec", pa.float32()),
            pa.field("end_sec", pa.float32()),
            pa.field("fps", pa.float32()),
            pa.field("language_instruction", pa.string()),
            pa.field("raw_transcript", pa.string()),
            pa.field("action_description", pa.string()),
            pa.field("intent", pa.string()),
            pa.field("observed_affect", pa.string()),
            pa.field("motion_style", pa.string()),
            pa.field("semantic_gesture", pa.string()),
            pa.field("arousal", pa.float32()),
            pa.field("valence", pa.float32()),
            pa.field("arousal_token", pa.int64()),
            pa.field("valence_token", pa.int64()),
            pa.field("motion_energy", pa.float32()),
            pa.field("accepted_for_training", pa.bool_()),
            pa.field("quality_frame_count", pa.int64()),
            pa.field("quality_flagged_frame_count", pa.int64()),
            pa.field("quality_max_elbow_overfold", pa.float32()),
            pa.field("quality_max_yaw_under_response", pa.float32()),
        ]
    )


def semantic_schema():
    return pa.schema(
        [
            pa.field("episode_index", pa.int64()),
            pa.field("sample_id", pa.string()),
            pa.field("language_instruction", pa.string()),
            pa.field("raw_transcript", pa.string()),
            pa.field("scenario_description", pa.string()),
            pa.field("action_description", pa.string()),
            pa.field("intent_text", pa.string()),
            pa.field("mood_text", pa.string()),
            pa.field("rationale_text", pa.string()),
            pa.field("intent", pa.string()),
            pa.field("observed_affect", pa.string()),
            pa.field("motion_style", pa.string()),
            pa.field("semantic_gesture", pa.string()),
            pa.field("arousal", pa.float32()),
            pa.field("valence", pa.float32()),
            pa.field("arousal_token", pa.int64()),
            pa.field("valence_token", pa.int64()),
            pa.field("motion_energy", pa.float32()),
            pa.field("instruction_variants_json", pa.string()),
            pa.field("annotation_views_json", pa.string()),
            pa.field("label_sources_json", pa.string()),
            pa.field("source_video_path", pa.string()),
            pa.field("source_retarget_csv_path", pa.string()),
            pa.field("start_row", pa.int64()),
            pa.field("end_row", pa.int64()),
        ]
    )


def manifest_schema():
    return pa.schema(
        [
            pa.field("episode_index", pa.int64()),
            pa.field("sample_id", pa.string()),
            pa.field("valid", pa.bool_()),
            pa.field("skip_reason", pa.string()),
            pa.field("length", pa.int64()),
            pa.field("source_retarget_csv_path", pa.string()),
        ]
    )


def info_json(total_episodes, total_frames, total_tasks, rows_per_file):
    scalar = {"dtype": "int64", "shape": [], "names": None}
    return {
        "codebase_version": LEROBOT_VERSION,
        "robot_type": ROBOT_TYPE,
        "fps": 30,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_chunks": max(1, math.ceil(total_frames / rows_per_file)) if total_frames else 0,
        "chunks_size": rows_per_file,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": None,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [len(JOINT_ORDER)], "names": JOINT_ORDER},
            "action": {"dtype": "float32", "shape": [len(JOINT_ORDER)], "names": JOINT_ORDER},
            "timestamp": {"dtype": "float32", "shape": [], "names": None},
            "frame_index": scalar,
            "episode_index": scalar,
            "index": scalar,
            "task_index": scalar,
            "next.done": {"dtype": "bool", "shape": [], "names": None},
        },
    }


def initial_stats():
    return {
        "count": 0,
        "obs_sum": np.zeros(len(JOINT_ORDER), dtype=np.float64),
        "obs_sq": np.zeros(len(JOINT_ORDER), dtype=np.float64),
        "obs_min": np.full(len(JOINT_ORDER), np.inf, dtype=np.float64),
        "obs_max": np.full(len(JOINT_ORDER), -np.inf, dtype=np.float64),
        "act_sum": np.zeros(len(JOINT_ORDER), dtype=np.float64),
        "act_sq": np.zeros(len(JOINT_ORDER), dtype=np.float64),
        "act_min": np.full(len(JOINT_ORDER), np.inf, dtype=np.float64),
        "act_max": np.full(len(JOINT_ORDER), -np.inf, dtype=np.float64),
    }


def update_stats(stats, obs, action):
    obs_arr = np.asarray(obs, dtype=np.float64)
    action_arr = np.asarray(action, dtype=np.float64)
    stats["count"] += 1
    stats["obs_sum"] += obs_arr
    stats["obs_sq"] += obs_arr * obs_arr
    stats["obs_min"] = np.minimum(stats["obs_min"], obs_arr)
    stats["obs_max"] = np.maximum(stats["obs_max"], obs_arr)
    stats["act_sum"] += action_arr
    stats["act_sq"] += action_arr * action_arr
    stats["act_min"] = np.minimum(stats["act_min"], action_arr)
    stats["act_max"] = np.maximum(stats["act_max"], action_arr)


def finish_stats(stats):
    count = max(1, stats["count"])
    obs_mean = stats["obs_sum"] / count
    act_mean = stats["act_sum"] / count
    obs_std = np.sqrt(np.maximum(stats["obs_sq"] / count - obs_mean * obs_mean, 0.0))
    act_std = np.sqrt(np.maximum(stats["act_sq"] / count - act_mean * act_mean, 0.0))
    return {
        "observation.state": {
            "mean": obs_mean.tolist(),
            "std": obs_std.tolist(),
            "min": stats["obs_min"].tolist(),
            "max": stats["obs_max"].tolist(),
        },
        "action": {
            "mean": act_mean.tolist(),
            "std": act_std.tolist(),
            "min": stats["act_min"].tolist(),
            "max": stats["act_max"].tolist(),
        },
    }


def flatten_episode_record(record, episode_index, task_index, length):
    source = record.get("source", {})
    time_window = record.get("time_window", {})
    language = record.get("language_condition", {})
    labels = record.get("labels", {})
    quality = record.get("quality", {})
    action = record.get("action", {})
    meta = record.get("meta_semantics", {})
    return {
        "episode_index": episode_index,
        "task_index": task_index,
        "length": length,
        "sample_id": record.get("sample_id", ""),
        "source_dataset": source.get("dataset", ""),
        "source_video_path": source.get("video_path", ""),
        "source_json_path": source.get("json_path", ""),
        "source_npz_path": source.get("npz_path", ""),
        "source_retarget_csv_path": action.get("retarget_csv_path") or source.get("retarget_csv_path", ""),
        "start_row": int(action.get("start_row", 0)),
        "end_row": int(action.get("end_row", 0)),
        "start_sec": float(time_window.get("start_sec", 0.0)),
        "end_sec": float(time_window.get("end_sec", 0.0)),
        "fps": float(time_window.get("fps", action.get("fps", 30.0))),
        "language_instruction": task_text(record),
        "raw_transcript": language.get("raw_transcript", ""),
        "action_description": language.get("action_description", ""),
        "intent": labels.get("intent", ""),
        "observed_affect": labels.get("observed_affect", ""),
        "motion_style": labels.get("motion_style", ""),
        "semantic_gesture": meta.get("semantic_gesture", ""),
        "arousal": optional_float(labels.get("arousal")),
        "valence": optional_float(labels.get("valence")),
        "arousal_token": optional_int(labels.get("arousal_token")),
        "valence_token": optional_int(labels.get("valence_token")),
        "motion_energy": optional_float(labels.get("motion_energy")),
        "accepted_for_training": bool(quality.get("accepted_for_training", False)),
        "quality_frame_count": int(quality.get("frame_count", 0)),
        "quality_flagged_frame_count": int(quality.get("flagged_frame_count", 0)),
        "quality_max_elbow_overfold": float(quality.get("max_elbow_overfold", 0.0)),
        "quality_max_yaw_under_response": float(quality.get("max_yaw_under_response", 0.0)),
    }


def flatten_semantic_record(record, episode_index):
    language = record.get("language_condition", {})
    labels = record.get("labels", {})
    meta = record.get("meta_semantics", {})
    source = record.get("source", {})
    action = record.get("action", {})
    return {
        "episode_index": episode_index,
        "sample_id": record.get("sample_id", ""),
        "language_instruction": task_text(record),
        "raw_transcript": language.get("raw_transcript", ""),
        "scenario_description": language.get("scenario_description", ""),
        "action_description": language.get("action_description", ""),
        "intent_text": language.get("intent_text", ""),
        "mood_text": language.get("mood_text", ""),
        "rationale_text": language.get("rationale_text", ""),
        "intent": labels.get("intent", ""),
        "observed_affect": labels.get("observed_affect", ""),
        "motion_style": labels.get("motion_style", ""),
        "semantic_gesture": meta.get("semantic_gesture", ""),
        "arousal": optional_float(labels.get("arousal")),
        "valence": optional_float(labels.get("valence")),
        "arousal_token": optional_int(labels.get("arousal_token")),
        "valence_token": optional_int(labels.get("valence_token")),
        "motion_energy": optional_float(labels.get("motion_energy")),
        "instruction_variants_json": json_dumps(language.get("instruction_variants", [])),
        "annotation_views_json": json_dumps(meta.get("annotation_views", {})),
        "label_sources_json": json_dumps(labels.get("label_sources", [])),
        "source_video_path": source.get("video_path", ""),
        "source_retarget_csv_path": action.get("retarget_csv_path") or source.get("retarget_csv_path", ""),
        "start_row": int(action.get("start_row", 0)),
        "end_row": int(action.get("end_row", 0)),
    }


def export_lerobot_dataset(jsonl_path, output_dir, rows_per_file=DEFAULT_ROWS_PER_FILE, max_episodes=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_rows = []
    data_file_index = 0
    episode_rows = []
    semantic_rows = []
    manifest_rows = []
    tasks = OrderedDict()
    stats = initial_stats()
    global_index = 0
    valid_episodes = 0
    skipped = 0

    for record in load_jsonl(jsonl_path):
        if max_episodes is not None and valid_episodes >= max_episodes:
            break
        action_meta = record.get("action", {})
        csv_path = action_meta.get("retarget_csv_path") or record.get("source", {}).get("retarget_csv_path")
        start_row = int(action_meta.get("start_row", 0))
        end_row = int(action_meta.get("end_row", 0))
        joints = read_joint_window(csv_path, start_row, end_row) if csv_path else []
        expected_len = end_row - start_row
        if not joints or len(joints) != expected_len:
            manifest_rows.append(
                {
                    "episode_index": -1,
                    "sample_id": record.get("sample_id", ""),
                    "valid": False,
                    "skip_reason": f"joint_window_length={len(joints)} expected={expected_len}",
                    "length": len(joints),
                    "source_retarget_csv_path": csv_path or "",
                }
            )
            skipped += 1
            continue

        episode_index = valid_episodes
        text = task_text(record)
        if text not in tasks:
            tasks[text] = len(tasks)
        task_index = tasks[text]
        length = len(joints)

        for frame_index, obs in enumerate(joints):
            action = joints[frame_index + 1] if frame_index + 1 < length else obs
            row = {
                "index": global_index,
                "episode_index": episode_index,
                "frame_index": frame_index,
                "timestamp": float(frame_index / float(action_meta.get("fps", 30.0))),
                "task_index": task_index,
                "observation.state": [float(x) for x in obs],
                "action": [float(x) for x in action],
                "next.done": frame_index == length - 1,
            }
            data_rows.append(row)
            update_stats(stats, row["observation.state"], row["action"])
            global_index += 1
            if len(data_rows) >= rows_per_file:
                write_table(output_dir, "data", data_rows, data_schema(), data_file_index)
                data_file_index += 1
                data_rows = []

        episode_rows.append(flatten_episode_record(record, episode_index, task_index, length))
        semantic_rows.append(flatten_semantic_record(record, episode_index))
        manifest_rows.append(
            {
                "episode_index": episode_index,
                "sample_id": record.get("sample_id", ""),
                "valid": True,
                "skip_reason": "",
                "length": length,
                "source_retarget_csv_path": csv_path,
            }
        )
        valid_episodes += 1

    if data_rows:
        write_table(output_dir, "data", data_rows, data_schema(), data_file_index)

    write_table(output_dir / "meta", "episodes", episode_rows, episode_schema(), 0)
    pq.write_table(pa.Table.from_pylist(semantic_rows, schema=semantic_schema()), output_dir / "meta" / "semantic_index.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(manifest_rows, schema=manifest_schema()), output_dir / "meta" / "export_manifest.parquet", compression="zstd")
    tasks_rows = [{"task_index": index, "task": text} for text, index in tasks.items()]
    write_jsonl(output_dir / "meta" / "tasks.jsonl", tasks_rows)
    write_json(output_dir / "meta" / "info.json", info_json(valid_episodes, global_index, len(tasks), rows_per_file))
    write_json(output_dir / "meta" / "stats.json", finish_stats(stats))
    readme = (
        "# V2 Upper-body LeRobot Dataset\n\n"
        "Body-only LeRobot-style export generated from retargeted V2 upper-body CSV windows and semantic JSONL.\n\n"
        "- `data/chunk-*/file-*.parquet`: per-frame rows with `observation.state`, `action`, indices, timestamps, and done flags.\n"
        "- `meta/episodes/chunk-000/file-000.parquet`: one row per 120-frame window with source paths, language labels, affect codes, and quality fields.\n"
        "- `meta/semantic_index.parquet`: flattened language/semantic table keyed by `episode_index`.\n"
        "- `meta/export_manifest.parquet`: valid/skipped export rows.\n"
        "- No face, gaze, or head features are included.\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    summary = {
        "output_dir": str(output_dir),
        "total_episodes": valid_episodes,
        "total_frames": global_index,
        "total_tasks": len(tasks),
        "skipped": skipped,
        "data_files": data_file_index + (1 if data_rows else 0),
    }
    write_json(output_dir / "export_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Export body-only language-action JSONL to LeRobot-style parquet dataset")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rows-per-file", type=int, default=DEFAULT_ROWS_PER_FILE)
    parser.add_argument("--max-episodes", type=int)
    args = parser.parse_args()
    summary = export_lerobot_dataset(args.jsonl, args.output_dir, rows_per_file=args.rows_per_file, max_episodes=args.max_episodes)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
