#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


ROOT = Path("/Users/demo/Desktop/upper_body_motion_roadmap")
PYTHON = Path("/Users/demo/Desktop/mjlab/.venv/bin/python")
EXTRACTED_ROOT = ROOT / "video/seamless_interaction_50g/extracted"
RUN_ROOT = ROOT / "data_build/new_urdf_0514_close_v2"
SHARDS_ROOT = RUN_ROOT / "full_retarget_shards"
LOG_DIR = RUN_ROOT / "logs"
FINAL_RETARGET = RUN_ROOT / "full_retarget"
FINAL_MANIFEST = FINAL_RETARGET / "manifest.csv"
LANGUAGE_INDEX = FINAL_RETARGET / "language_action_index.body_0514_close_v2.jsonl"
DATASET_RAW = ROOT / "datasets/lerobot_v2_upper_body_0514_close_v2_raw"
DATASET_CLEAN = ROOT / "datasets/lerobot_v2_upper_body_0514_close_v2_clean"
EXPECTED_COUNT = 487


def log(message):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    with (LOG_DIR / "pipeline.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd, *, log_path=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    log("running: " + " ".join(map(str, cmd)))
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(log_path).open("w", encoding="utf-8") as f:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    else:
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def shard_command(shard_index, shard_count):
    out_root = SHARDS_ROOT / f"shard_{shard_index}"
    return [
        str(PYTHON),
        "-m",
        "upper_body_skeleton.batch_retarget",
        "--extracted-root",
        str(EXTRACTED_ROOT),
        "--output-root",
        str(out_root),
        "--manifest",
        str(out_root / "manifest.csv"),
        "--fps",
        "30",
        "--full-video",
        "--stride",
        "1",
        "--output-hz",
        "30",
        "--retarget-mode",
        "ik",
        "--smooth-window-frames",
        "11",
        "--image-width",
        "1080",
        "--image-height",
        "1920",
        "--progress-interval",
        "5",
        "--overwrite",
        "--shard-index",
        str(shard_index),
        "--shard-count",
        str(shard_count),
    ]


def run_shard(shard_index, shard_count):
    log_path = LOG_DIR / f"shard_{shard_index}.log"
    run(shard_command(shard_index, shard_count), log_path=log_path)
    return shard_index


def manifest_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_manifest(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample",
        "npz_path",
        "video_path",
        "status",
        "frame_count",
        "flagged_frame_count",
        "max_cross_body_intent",
        "max_yaw_under_response",
        "max_elbow_overfold",
        "skeleton_json",
        "joint_csv",
        "monitor_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def merge_manifests(shard_count):
    rows = []
    for shard_index in range(shard_count):
        path = SHARDS_ROOT / f"shard_{shard_index}/manifest.csv"
        if not path.exists():
            raise RuntimeError(f"missing shard manifest: {path}")
        rows.extend(manifest_rows(path))
    rows.sort(key=lambda row: row["sample"])
    write_manifest(rows, FINAL_MANIFEST)
    statuses = {}
    for row in rows:
        status = row.get("status", "")
        statuses[status] = statuses.get(status, 0) + 1
    summary = {
        "manifest": str(FINAL_MANIFEST),
        "rows": len(rows),
        "statuses": statuses,
        "errors": [row for row in rows if row.get("status", "").startswith("error")],
    }
    (FINAL_RETARGET / "merge_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log(json.dumps(summary, ensure_ascii=False))
    if summary["errors"]:
        raise RuntimeError("retarget manifest contains errors")
    if len(rows) != EXPECTED_COUNT:
        raise RuntimeError(f"expected {EXPECTED_COUNT} rows, got {len(rows)}")
    return summary


def remove_dir(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)


def export_dataset():
    run(
        [
            str(PYTHON),
            "-m",
            "upper_body_skeleton.language_action_index",
            "--manifest",
            str(FINAL_MANIFEST),
            "--output-jsonl",
            str(LANGUAGE_INDEX),
            "--window-sec",
            "4",
            "--stride-sec",
            "2",
            "--fps",
            "30",
        ],
        log_path=LOG_DIR / "language_action_index.log",
    )
    remove_dir(DATASET_RAW)
    run(
        [
            str(PYTHON),
            "-m",
            "upper_body_skeleton.lerobot_export",
            "--jsonl",
            str(LANGUAGE_INDEX),
            "--output-dir",
            str(DATASET_RAW),
            "--rows-per-file",
            "250000",
        ],
        log_path=LOG_DIR / "lerobot_export.log",
    )
    remove_dir(DATASET_CLEAN)
    run(
        [
            str(PYTHON),
            "-m",
            "upper_body_skeleton.clean_lerobot_dataset",
            "--input-dir",
            str(DATASET_RAW),
            "--output-dir",
            str(DATASET_CLEAN),
        ],
        log_path=LOG_DIR / "clean_lerobot_dataset.log",
    )


def main():
    parser = argparse.ArgumentParser(description="Run 0514 close-hand V2 retarget + LeRobot export pipeline")
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--retarget-only", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not args.export_only:
        remove_dir(SHARDS_ROOT)
        remove_dir(FINAL_RETARGET)
        SHARDS_ROOT.mkdir(parents=True, exist_ok=True)
        FINAL_RETARGET.mkdir(parents=True, exist_ok=True)
        log(f"starting retarget shards={args.shards}")
        with ThreadPoolExecutor(max_workers=args.shards) as executor:
            futures = [executor.submit(run_shard, index, args.shards) for index in range(args.shards)]
            for future in as_completed(futures):
                log(f"finished shard {future.result()}")
        merge_manifests(args.shards)

    if args.retarget_only:
        log("retarget-only complete")
        return

    export_dataset()
    summary = {
        "retarget_manifest": str(FINAL_MANIFEST),
        "language_index": str(LANGUAGE_INDEX),
        "dataset_raw": str(DATASET_RAW),
        "dataset_clean": str(DATASET_CLEAN),
    }
    (RUN_ROOT / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log("pipeline complete: " + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
