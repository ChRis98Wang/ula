#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/demo/Desktop/upper_body_motion_roadmap"
PY="/Users/demo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
MANIFEST="$ROOT/video/seamless_interaction_50g/batch_v2_retarget_ik_contact_safe/full_single_progress/manifest.csv"
JSONL="$ROOT/video/seamless_interaction_50g/batch_v2_retarget_ik_contact_safe/full_single_progress/language_action_index.body_final.jsonl"
OUT="$ROOT/video/seamless_interaction_50g/lerobot_v2_upper_body_body_only_final"
LOG="$ROOT/video/seamless_interaction_50g/lerobot_v2_upper_body_body_only_final/export.log"

mkdir -p "$(dirname "$LOG")"
{
  echo "[lerobot-export] waiting for retarget batch to finish: $(date)"
  while pgrep -f "upper_body_skeleton.batch_retarget" >/dev/null 2>&1; do
    sleep 300
  done

  echo "[lerobot-export] retarget batch stopped: $(date)"
  awk -F, 'NR>1 {total++; if ($4=="processed" || $4=="skipped_existing") ok++; if ($4 ~ /^error/) err++} END {printf "[lerobot-export] manifest total=%d ok=%d errors=%d\n", total, ok, err; if (err>0) exit 2; if (ok<487) exit 3}' "$MANIFEST"

  echo "[lerobot-export] rebuilding body-only JSONL: $(date)"
  PYTHONPATH="$ROOT" "$PY" -m upper_body_skeleton.language_action_index \
    --manifest "$MANIFEST" \
    --output-jsonl "$JSONL"

  echo "[lerobot-export] exporting LeRobot parquet: $(date)"
  rm -rf "$OUT"
  PYTHONPATH="$ROOT" "$PY" -m upper_body_skeleton.lerobot_export \
    --jsonl "$JSONL" \
    --output-dir "$OUT" \
    --rows-per-file 250000

  echo "[lerobot-export] done: $(date)"
} >> "$LOG" 2>&1
