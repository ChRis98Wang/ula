#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/gez/shuaiwang/ula-motion-generate"
conda_bin="/home/gez/miniconda3/condabin/conda"
lora_service="ula-qwen-motion-latent-v2-train.service"
lora_dir="training/runs/kimodo_qwen_motion_latent_lora_v2"
smoke_dir="training/runs/kimodo_ula_v2_lora_smoke"

cd "$repo_root"

while systemctl --user is-active --quiet "$lora_service"; do
  sleep 30
done

"$conda_bin" run --no-capture-output -n env_isaaclab python - <<'PY'
import json
from pathlib import Path

summary_path = Path("training/runs/kimodo_qwen_motion_latent_lora_v2/training_summary.json")
best_path = summary_path.parent / "best.pt"
if not summary_path.is_file() or not best_path.is_file():
    raise SystemExit("Qwen Motion LoRA did not produce training_summary.json and best.pt")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if int(summary.get("steps", -1)) != 100000:
    raise SystemExit(f"Qwen Motion LoRA stopped at unexpected step: {summary.get('steps')}")
print(json.dumps({"stage": "lora_complete", "summary": summary}, sort_keys=True), flush=True)
PY

"$conda_bin" run --no-capture-output -n env_isaaclab \
  python -u training/scripts/train_ula_v2.py \
  --config configs/train_kimodo_ula_v2_lora_smoke.yaml

"$conda_bin" run --no-capture-output -n env_isaaclab python - <<'PY'
import hashlib
from pathlib import Path
import torch

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

lora_path = Path("training/runs/kimodo_qwen_motion_latent_lora_v2/best.pt")
generator_path = Path("training/runs/kimodo_ula_v2_lora_smoke/ula_fm_checkpoint.pt")
checkpoint = torch.load(generator_path, map_location="cpu", weights_only=True)
contract = checkpoint["v2_contracts"]["text_motion_latent"]
if contract["source"]["checkpoint_sha256"] != sha256_file(lora_path):
    raise SystemExit("smoke generator did not bind the exact Qwen Motion LoRA checkpoint")
print({"stage": "smoke_contract_verified", "generator": str(generator_path)}, flush=True)
PY

"$conda_bin" run --no-capture-output -n env_isaaclab \
  python -u -m upper_body_skeleton.pt_mujoco_infer \
  --checkpoint "$smoke_dir/ula_fm_checkpoint.pt" \
  --motion-latent-lora-checkpoint "$lora_dir/best.pt" \
  --motion-latent-device cuda \
  --motion-latent-local-files-only \
  --device cuda \
  --text "开心地向主人挥手" \
  --no-viewer

"$conda_bin" run --no-capture-output -n env_isaaclab \
  python -u training/scripts/train_ula_v2.py \
  --config configs/train_kimodo_ula_v2_lora.yaml
