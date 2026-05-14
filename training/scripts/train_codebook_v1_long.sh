#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/demo/Desktop/upper_body_motion_roadmap"
PY="/Users/demo/Desktop/mjlab/.venv/bin/python"

PYTHONPATH="$ROOT" "$PY" -m upper_body_skeleton.ula_training \
  --dataset-dir "$ROOT/datasets/lerobot_v2_upper_body_body_only_final" \
  --output-dir "$ROOT/training/runs/codebook_v1_long" \
  --steps 20000 \
  --batch-size 32 \
  --max-episodes 51101 \
  --hidden-dim 384 \
  --layers 6 \
  --device auto \
  --log-interval 200
