#!/usr/bin/env bash
set -euo pipefail

# Read-only terminal-artifact watcher.  It never starts or mutates training.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PYTHON="${PROJECT_DIR}/../.venvs/gmr/bin/python"
if [[ ! -x "${DEFAULT_PYTHON}" ]]; then
  DEFAULT_PYTHON="python3"
fi
PYTHON_BIN="${ULA_REPORT_PYTHON:-${DEFAULT_PYTHON}}"
RUNS_DIR="${PROJECT_DIR}/training/runs"
FOUNDATION_RUN="${RUNS_DIR}/beat2_18d_from_scratch_formal_v7_clean_adaln"
POSTTRAIN_RUN="${RUNS_DIR}/beat2_experimental_metadata_posttrain_ab_v1"
QWEN_RUN="${RUNS_DIR}/beat2_qwen_motion_alignment_ab_v1"
VIDEO_RUN="${RUNS_DIR}/beat2_clean_adaln_abc_video_v1"
REPORT_DIR="${RUNS_DIR}/beat2_clean_morning_report_v1"

exec "${PYTHON_BIN}" "${PROJECT_DIR}/tools/build_clean_training_ab_report.py" \
  --foundation-run "${FOUNDATION_RUN}" \
  --generator "B_frozen_qwen=${POSTTRAIN_RUN}/frozen_base" \
  --generator "C_beat2_lora=${POSTTRAIN_RUN}/lora_finetuned" \
  --condition-cache "B_frozen_qwen=${POSTTRAIN_RUN}/prepared/conditions_264d_frozen_base.experimental.npz" \
  --condition-cache "C_beat2_lora=${POSTTRAIN_RUN}/prepared/conditions_264d_lora_finetuned.experimental.npz" \
  --qwen-frozen "${QWEN_RUN}" \
  --qwen-finetuned "${QWEN_RUN}" \
  --abc-video-config "${PROJECT_DIR}/configs/beat2_clean_adaln_abc_video_v1.json" \
  --video-artifact "${VIDEO_RUN}" \
  --output-json "${REPORT_DIR}/report.json" \
  --output-md "${REPORT_DIR}/report.md" \
  --wait-for-artifacts \
  --strict
