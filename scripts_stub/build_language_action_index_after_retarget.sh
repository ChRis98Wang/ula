#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/demo/Desktop/upper_body_motion_roadmap"
PY="/Users/demo/Desktop/mjlab/.venv/bin/python"
MANIFEST="$ROOT/video/seamless_interaction_50g/batch_v2_retarget_ik_contact_safe/full_single_progress/manifest.csv"
OUT="$ROOT/video/seamless_interaction_50g/batch_v2_retarget_ik_contact_safe/full_single_progress/language_action_index.jsonl"

cd "$ROOT"
PYTHONPATH="$ROOT" "$PY" -m upper_body_skeleton.language_action_index \
  --manifest "$MANIFEST" \
  --output-jsonl "$OUT"
