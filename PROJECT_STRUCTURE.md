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
/Users/demo/Desktop/upper_body_motion_roadmap/datasets/lerobot_v2_upper_body_body_only_clean_v1
```

Useful dataset files:

- `datasets/lerobot_v2_upper_body_body_only_clean_v1/meta/semantic_index.parquet`
- `datasets/lerobot_v2_upper_body_body_only_clean_v1/data/chunk-000/*.parquet`
- `datasets/lerobot_v2_upper_body_body_only_clean_v1/cleaning_summary.json`

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

- `training/scripts/train_ula_fm.py`
- `training/scripts/train_codebook_v1_long.sh`
- `training/scripts/infer_codebook_v1.sh`
- `training/scripts/render_generated_motion.sh`

Server/S3 training:

- `configs/train_v04_s3_example.yaml`
- `docs/roadmap/12_server_s3_training.md`

## Training With Automatic MuJoCo Previews

The training module can generate and record a MuJoCo preview every N steps.
For example, every 1,000 steps:

```bash
cd /Users/demo/Desktop/upper_body_motion_roadmap

PYTHONPATH=/Users/demo/Desktop/upper_body_motion_roadmap \
/Users/demo/Desktop/mjlab/.venv/bin/python -u -m training.scripts.train_ula_fm \
  --config configs/train_v04_s3_example.yaml
```

Each preview is written to:

```text
training/runs/ula_fm_v04_clean_v1_pos_long_1m/previews/step_001000/
  long_motion.csv
  long_motion.npz
  long_motion_original_v2.mp4
  plan.json
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
