#!/usr/bin/env bash
set -euo pipefail

project_root="/home/gez/shuaiwang/ula-motion-generate"
python_bin="/home/gez/shuaiwang/.venvs/gmr/bin/python"
config_path="${BEAT2_POSTTRAIN_CONFIG:-${project_root}/configs/beat2_experimental_metadata_posttrain_ab_v1.json}"
output_dir="${BEAT2_POSTTRAIN_OUTPUT:-${project_root}/training/runs/beat2_experimental_metadata_posttrain_ab_v1}"
run_mode="${BEAT2_POSTTRAIN_RUN_MODE:-parallel}"

foundation_summary="${project_root}/training/runs/beat2_18d_from_scratch_formal_v7_clean_adaln/training/training_summary.json"
foundation_checkpoint="${project_root}/training/runs/beat2_18d_from_scratch_formal_v7_clean_adaln/training/ula_fm_checkpoint.pt"
qwen_root="${project_root}/training/runs/beat2_qwen_motion_alignment_ab_v1"
frozen_cache="${qwen_root}/conditions_128d_frozen_base.npz"
lora_cache="${qwen_root}/conditions_128d_lora_finetuned.npz"

mkdir -p "${output_dir}/logs"
exec 9>"${output_dir}/nightly.lock"
if ! flock -n 9; then
  echo "another BEAT2 experimental metadata posttrain launcher holds the lock" >&2
  exit 1
fi

required_inputs=(
  "${foundation_summary}"
  "${foundation_checkpoint}"
  "${frozen_cache}"
  "${frozen_cache}.json"
  "${lora_cache}"
  "${lora_cache}.json"
)
while true; do
  missing=0
  for path in "${required_inputs[@]}"; do
    if [[ ! -s "${path}" ]]; then
      missing=1
    fi
  done
  if [[ "${missing}" -eq 0 ]]; then
    break
  fi
  date --iso-8601=seconds
  echo "waiting for completed clean foundation and full Qwen B/C caches"
  sleep 30
done

cd "${project_root}"
"${python_bin}" tools/train_beat2_experimental_metadata_posttrain_ab.py \
  --config "${config_path}" \
  --output-dir "${output_dir}" \
  --stage prepare \
  --resume

run_branch() {
  local stage="$1"
  "${python_bin}" tools/train_beat2_experimental_metadata_posttrain_ab.py \
    --config "${config_path}" \
    --output-dir "${output_dir}" \
    --stage "${stage}" \
    --resume \
    >>"${output_dir}/logs/${stage}.log" 2>&1
}

if [[ "${run_mode}" == "parallel" ]]; then
  run_branch frozen &
  frozen_pid=$!
  run_branch lora &
  lora_pid=$!
  frozen_status=0
  lora_status=0
  wait "${frozen_pid}" || frozen_status=$?
  wait "${lora_pid}" || lora_status=$?
  if [[ "${frozen_status}" -ne 0 || "${lora_status}" -ne 0 ]]; then
    echo "paired branches failed: frozen=${frozen_status} lora=${lora_status}" >&2
    exit 1
  fi
elif [[ "${run_mode}" == "sequential" ]]; then
  run_branch frozen
  run_branch lora
else
  echo "BEAT2_POSTTRAIN_RUN_MODE must be parallel or sequential" >&2
  exit 2
fi

"${python_bin}" tools/train_beat2_experimental_metadata_posttrain_ab.py \
  --config "${config_path}" \
  --output-dir "${output_dir}" \
  --stage compare \
  --resume \
  >>"${output_dir}/logs/compare.log" 2>&1

echo "paired experimental metadata posttrain completed: ${output_dir}"
