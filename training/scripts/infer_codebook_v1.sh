#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/demo/Desktop/upper_body_motion_roadmap"
PY="/Users/demo/Desktop/mjlab/.venv/bin/python"
TEXT="${1:-紧张地解释，同时双手做克制的上肢手势}"
NAME="${2:-generated_custom}"

PYTHONPATH="$ROOT" "$PY" -m upper_body_skeleton.ula_infer \
  --checkpoint "$ROOT/weights/codebook_v1_checkpoint.pt" \
  --text "$TEXT" \
  --output-csv "$ROOT/training/runs/codebook_v1/${NAME}.csv" \
  --output-npz "$ROOT/training/runs/codebook_v1/${NAME}.npz" \
  --summary-json "$ROOT/training/runs/codebook_v1/${NAME}.json" \
  --frames 120 \
  --sampling-steps 32 \
  --device auto
