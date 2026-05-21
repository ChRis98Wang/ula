#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


ROOT = Path("/Users/demo/Desktop/upper_body_motion_roadmap")
PYTHON = Path("/Users/demo/Desktop/mjlab/.venv/bin/python")
EXTRACTED_ROOT = ROOT / "video/seamless_interaction_50g/extracted"
RUN_ROOT = ROOT / "data_build/new_urdf_0514_close_v2"
SHARDS_ROOT = RUN_ROOT / "full_retarget_shards"
FAST_EXTRACTED_ROOT = RUN_ROOT / "remaining_fast_extracted"
LOG_DIR = RUN_ROOT / "logs"
FINAL_RETARGET = RUN_ROOT / "full_retarget"
FINAL_MANIFEST = FINAL_RETARGET / "manifest.csv"
LANGUAGE_INDEX = FINAL_RETARGET / "language_action_index.body_0514_close_v2.jsonl"
DATASET_RAW = ROOT / "datasets/lerobot_v2_upper_body_0514_close_v2_raw"
DATASET_CLEAN = ROOT / "datasets/lerobot_v2_upper_body_0514_close_v2_clean"
EXPECTED_COUNT = 487


FIELDNAMES = [
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


def manifest_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_manifest(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def sample_key(npz_path):
    return Path(npz_path).relative_to(EXTRACTED_ROOT).with_suffix("").as_posix()


def collect_completed_rows():
    completed = {}
    errors = []
    for manifest in sorted(SHARDS_ROOT.glob("shard_*/manifest.csv")):
        for row in manifest_rows(manifest):
            status = row.get("status", "")
            if status.startswith("error"):
                errors.append({"manifest": str(manifest), "sample": row.get("sample", ""), "status": status})
                continue
            if status in {"processed", "skipped_existing"}:
                completed[row["sample"]] = row
    if errors:
        raise RuntimeError("existing shard manifests contain errors: " + json.dumps(errors[:3], ensure_ascii=False))
    return completed


def clear_fast_shards():
    for path in sorted(SHARDS_ROOT.glob("shard_fast_*")):
        if path.is_dir():
            shutil.rmtree(path)


def build_remaining_extracted_root(remaining_npz):
    if FAST_EXTRACTED_ROOT.exists():
        shutil.rmtree(FAST_EXTRACTED_ROOT)
    for src_npz in remaining_npz:
        rel = Path(src_npz).relative_to(EXTRACTED_ROOT)
        dst_npz = FAST_EXTRACTED_ROOT / rel
        dst_npz.parent.mkdir(parents=True, exist_ok=True)
        dst_npz.symlink_to(src_npz)
        src_mp4 = Path(src_npz).with_suffix(".mp4")
        if src_mp4.exists():
            dst_mp4 = dst_npz.with_suffix(".mp4")
            dst_mp4.symlink_to(src_mp4)


def shard_command(shard_index, shard_count):
    out_root = SHARDS_ROOT / f"shard_fast_{shard_index:02d}"
    return [
        str(PYTHON),
        "-m",
        "upper_body_skeleton.batch_retarget",
        "--extracted-root",
        str(FAST_EXTRACTED_ROOT),
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


def run_fast_shard(shard_index, shard_count):
    run(shard_command(shard_index, shard_count), log_path=LOG_DIR / f"shard_fast_{shard_index:02d}.log")
    return shard_index


def merge_all_manifests():
    merged = {}
    errors = []
    for manifest in sorted(SHARDS_ROOT.glob("shard_*/manifest.csv")):
        for row in manifest_rows(manifest):
            status = row.get("status", "")
            if status.startswith("error"):
                errors.append({"manifest": str(manifest), "sample": row.get("sample", ""), "status": status})
                continue
            if status in {"processed", "skipped_existing"}:
                merged[row["sample"]] = row
    rows = [merged[key] for key in sorted(merged)]
    write_manifest(rows, FINAL_MANIFEST)
    summary = {
        "manifest": str(FINAL_MANIFEST),
        "rows": len(rows),
        "statuses": {},
        "errors": errors[:10],
    }
    for row in rows:
        status = row.get("status", "")
        summary["statuses"][status] = summary["statuses"].get(status, 0) + 1
    (FINAL_RETARGET / "merge_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log(json.dumps(summary, ensure_ascii=False))
    if errors:
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
    parser = argparse.ArgumentParser(description="Resume 0514 close-hand V2 retarget using extra workers on remaining samples")
    parser.add_argument("--shards", type=int, default=12)
    parser.add_argument("--retarget-only", action="store_true")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SHARDS_ROOT.mkdir(parents=True, exist_ok=True)
    FINAL_RETARGET.mkdir(parents=True, exist_ok=True)

    completed = collect_completed_rows()
    all_npz = sorted(EXTRACTED_ROOT.rglob("*.npz"))
    remaining_npz = [path for path in all_npz if sample_key(path) not in completed]
    log(f"fast resume: completed={len(completed)} remaining={len(remaining_npz)} shards={args.shards}")

    clear_fast_shards()
    build_remaining_extracted_root(remaining_npz)

    started = time.time()
    with ThreadPoolExecutor(max_workers=args.shards) as executor:
        futures = [executor.submit(run_fast_shard, index, args.shards) for index in range(args.shards)]
        for future in as_completed(futures):
            log(f"finished fast shard {future.result()}")
    log(f"fast retarget finished in {time.time() - started:.1f}s")

    merge_all_manifests()
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
