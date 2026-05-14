#!/usr/bin/env python3
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path("/Users/demo/Desktop/upper_body_motion_roadmap")
MANIFEST = ROOT / "video/seamless_interaction_50g/batch_v2_retarget_ik_contact_safe/full_single_progress/manifest.csv"
JSONL = ROOT / "video/seamless_interaction_50g/batch_v2_retarget_ik_contact_safe/full_single_progress/language_action_index.body_final.jsonl"
OUT = ROOT / "video/seamless_interaction_50g/lerobot_v2_upper_body_body_only_final"
TRAIN_OUT = ROOT / "video/seamless_interaction_50g/ula_fm_runs/final_auto"
LOG = OUT / "export.log"
RETARGET_PID = 53733
EXPECTED_COUNT = 487
CHECK_INTERVAL_SEC = 60


def log(message):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {message}\n")


def pid_running(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def batch_process_running():
    if pid_running(RETARGET_PID):
        return True
    result = subprocess.run(
        ["pgrep", "-f", "upper_body_skeleton.batch_retarget"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip())


def manifest_counts():
    total = 0
    ok = 0
    errors = []
    if not MANIFEST.exists():
        return {"total": 0, "ok": 0, "errors": ["manifest missing"]}
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            total += 1
            status = row.get("status", "")
            if status in {"processed", "skipped_existing"}:
                ok += 1
            if status.startswith("error"):
                errors.append(json.dumps(row, ensure_ascii=False))
    return {"total": total, "ok": ok, "errors": errors}


def run_module(module, *args):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    cmd = [sys.executable, "-m", module, *map(str, args)]
    log("running: " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def main():
    log("watcher started; waiting for retarget batch to finish")
    while batch_process_running():
        counts = manifest_counts()
        log(f"retarget still running; manifest ok={counts['ok']} total={counts['total']} errors={len(counts['errors'])}")
        time.sleep(CHECK_INTERVAL_SEC)

    counts = manifest_counts()
    log(f"retarget stopped; manifest ok={counts['ok']} total={counts['total']} errors={len(counts['errors'])}")
    if counts["errors"]:
        log("not exporting final parquet because manifest has errors: " + " | ".join(counts["errors"][:5]))
        raise SystemExit(2)
    if counts["ok"] < EXPECTED_COUNT:
        log(f"not exporting final parquet because only {counts['ok']}/{EXPECTED_COUNT} rows are processed")
        raise SystemExit(3)

    run_module(
        "upper_body_skeleton.language_action_index",
        "--manifest",
        MANIFEST,
        "--output-jsonl",
        JSONL,
    )
    if OUT.exists():
        for path in sorted(OUT.rglob("*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    OUT.mkdir(parents=True, exist_ok=True)
    run_module(
        "upper_body_skeleton.lerobot_export",
        "--jsonl",
        JSONL,
        "--output-dir",
        OUT,
        "--rows-per-file",
        250000,
    )
    log("final LeRobot parquet export complete")
    run_module(
        "upper_body_skeleton.ula_training",
        "--dataset-dir",
        OUT,
        "--output-dir",
        TRAIN_OUT,
        "--steps",
        2000,
        "--batch-size",
        32,
        "--max-episodes",
        12000,
        "--hidden-dim",
        256,
        "--layers",
        4,
        "--device",
        "auto",
    )
    log(f"final ULA-FM training complete: {TRAIN_OUT}")


if __name__ == "__main__":
    main()
