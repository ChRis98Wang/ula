# Project Structure

The workspace is organized by role. Large video/data folders are linked instead
of copied so the 122G source dataset is not duplicated and existing absolute
paths in metadata remain valid.

## Top-Level Folders

- `datasets/`
  Final model-ready dataset entry points.
- `data_build/`
  Raw downloads, extracted videos, retarget outputs, manifests, and data-making
  scripts.
- `training/`
  Training scripts and training run folders.
- `weights/`
  Checkpoint-only links for quick loading.
- `deliverables/`
  Clean inspection entry point for demos, generated videos, data, and weights.
- `upper_body_skeleton/`
  Source code package.
- `tests/`
  Automated tests.
- `configs/`
  Robot, binding, and label configuration.
- `docs/`
  Roadmap docs, references, and HTML architecture demos.
- `schemas/`
  JSON schemas for data records.

## Dataset Folder

Use this for training:

```text
/Users/demo/Desktop/upper_body_motion_roadmap/datasets/lerobot_v2_upper_body_body_only_final
```

Useful dataset files:

- `datasets/semantic_index.parquet`
- `datasets/language_action_index.body_final.jsonl`
- `datasets/retarget_manifest.csv`

## Data-Making Folder

Use `data_build/` when rebuilding or debugging the dataset:

- `data_build/raw_downloads`
- `data_build/extracted_videos`
- `data_build/retargeted_full_progress`
- `data_build/final_lerobot_export`
- `data_build/scripts/`

## Training Folder

Current recommended run:

```text
/Users/demo/Desktop/upper_body_motion_roadmap/training/runs/codebook_v1
```

Manual scripts:

- `training/scripts/train_codebook_v1_long.sh`
- `training/scripts/infer_codebook_v1.sh`
- `training/scripts/render_generated_motion.sh`

## Training With Automatic MuJoCo Previews

The training module can generate and record a MuJoCo preview every N steps.
For example, every 1,000 steps:

```bash
cd /Users/demo/Desktop/upper_body_motion_roadmap

PYTHONPATH=/Users/demo/Desktop/upper_body_motion_roadmap \
/Users/demo/Desktop/mjlab/.venv/bin/python -m upper_body_skeleton.ula_training \
  --dataset-dir /Users/demo/Desktop/upper_body_motion_roadmap/datasets/lerobot_v2_upper_body_body_only_final \
  --output-dir /Users/demo/Desktop/upper_body_motion_roadmap/training/runs/codebook_v1_long \
  --steps 20000 \
  --batch-size 32 \
  --max-episodes 51101 \
  --hidden-dim 384 \
  --layers 6 \
  --device auto \
  --log-interval 200 \
  --preview-every-steps 1000 \
  --preview-dir /Users/demo/Desktop/upper_body_motion_roadmap/training/runs/codebook_v1_long/previews \
  --preview-text "紧张地解释，同时双手做克制的上肢手势" \
  --preview-frames 120 \
  --preview-sampling-steps 32 \
  --preview-width 1280 \
  --preview-height 720
```

Each preview is written to:

```text
training/runs/codebook_v1_long/previews/step_001000/
  generated.csv
  generated.npz
  preview.mp4
  summary.json
```

## Weights Folder

Recommended checkpoint:

```text
/Users/demo/Desktop/upper_body_motion_roadmap/weights/codebook_v1_checkpoint.pt
```

Baseline checkpoint:

```text
/Users/demo/Desktop/upper_body_motion_roadmap/weights/baseline_final_auto_checkpoint.pt
```

## Generated Motion Preview

```text
/Users/demo/Desktop/upper_body_motion_roadmap/deliverables/generated_motion_previews/nervous_explain_mujoco.mp4
```
