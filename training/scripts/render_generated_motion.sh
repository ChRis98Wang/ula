#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/demo/Desktop/upper_body_motion_roadmap"
PY="/Users/demo/Desktop/mjlab/.venv/bin/python"
CSV="${1:-$ROOT/training/runs/codebook_v1/generated_nervous_explain.csv}"
NAME="${2:-generated_motion}"

PYTHONPATH="$ROOT" "$PY" -m upper_body_skeleton.mujoco_playback \
  --joint-csv "$CSV" \
  --output-mp4 "$ROOT/deliverables/generated_motion_previews/${NAME}_mujoco.mp4" \
  --summary-json "$ROOT/deliverables/generated_motion_previews/${NAME}_mujoco.json" \
  --fps 30 \
  --width 1280 \
  --height 720
