# Server And S3 Training Layout

Date: 2026-05-15

Goal: keep this repository as code/config/docs, put datasets and training artifacts in S3, and run ULA-FM training on a server without local absolute Mac paths.

## Repository Boundary

Commit to git:

- `upper_body_skeleton/`
- `training/scripts/`
- `configs/`
- `schemas/`
- `tests/`
- `docs/`
- `README.md`
- `PROJECT_STRUCTURE.md`

Do not commit:

- `datasets/`
- `data_build/`
- `video/`
- `training/runs/`
- `weights/`
- `models/smplh/`
- `deliverables/`
- `final_dataset_package/`

These are already covered by `.gitignore`.

## Recommended Server Layout

```text
/workspace/upper_body_motion_roadmap/          # git repo
/mnt/datasets/lerobot_v2_upper_body_body_only_clean_v1/
  data/chunk-000/file-000.parquet
  ...
  meta/info.json
  meta/stats.json
  meta/tasks.jsonl
  meta/semantic_index.parquet
  cleaning_summary.json
/mnt/artifacts/ula_fm/
  runs/
  checkpoints/
  previews/
```

## Recommended S3 Layout

Replace `s3://YOUR_BUCKET/upper_body_motion_roadmap` with the real bucket prefix.

```text
s3://YOUR_BUCKET/upper_body_motion_roadmap/
  datasets/
    lerobot_v2_upper_body_body_only_clean_v1/
      data/
      meta/
      cleaning_summary.json
      cleaning_episode_report.jsonl
      export_summary.json
  training/
    runs/
      ula_fm_v04_clean_v1_pos_long_1m/
        progress.jsonl
        train_log.json
        ula_fm_checkpoint.pt
        previews/
  docs/
    dataset_cards/
```

## Upload Dataset From Local Machine

Use `aws s3 sync` so interrupted uploads can resume.

```bash
aws s3 sync \
  /Users/demo/Desktop/upper_body_motion_roadmap/datasets/lerobot_v2_upper_body_body_only_clean_v1 \
  s3://YOUR_BUCKET/upper_body_motion_roadmap/datasets/lerobot_v2_upper_body_body_only_clean_v1 \
  --exclude ".DS_Store"
```

Check size:

```bash
aws s3 ls \
  s3://YOUR_BUCKET/upper_body_motion_roadmap/datasets/lerobot_v2_upper_body_body_only_clean_v1/ \
  --recursive --summarize
```

## Download Dataset On Server

```bash
mkdir -p /mnt/datasets/lerobot_v2_upper_body_body_only_clean_v1

aws s3 sync \
  s3://YOUR_BUCKET/upper_body_motion_roadmap/datasets/lerobot_v2_upper_body_body_only_clean_v1 \
  /mnt/datasets/lerobot_v2_upper_body_body_only_clean_v1
```

## Server Environment

From a fresh server:

```bash
cd /workspace
git clone <YOUR_GIT_REMOTE> upper_body_motion_roadmap
cd /workspace/upper_body_motion_roadmap

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch numpy pyarrow pyyaml imageio-ffmpeg mediapy mujoco pytest
```

If the server has CUDA, install the CUDA-matched PyTorch build instead of generic `torch`.

## Smoke Test

```bash
cd /workspace/upper_body_motion_roadmap
source .venv/bin/activate

PYTHONPATH=/workspace/upper_body_motion_roadmap \
python -m pytest -q
```

## Print Training Command From Config

```bash
cd /workspace/upper_body_motion_roadmap
source .venv/bin/activate

PYTHONPATH=/workspace/upper_body_motion_roadmap \
python -m training.scripts.train_ula_fm \
  --config configs/train_v04_s3_example.yaml \
  --print-command
```

## Start Training

Foreground, transparent:

```bash
cd /workspace/upper_body_motion_roadmap
source .venv/bin/activate

PYTHONPATH=/workspace/upper_body_motion_roadmap \
python -u -m training.scripts.train_ula_fm \
  --config configs/train_v04_s3_example.yaml
```

Background with log:

```bash
cd /workspace/upper_body_motion_roadmap
source .venv/bin/activate

mkdir -p /workspace/upper_body_motion_roadmap/training/runs/ula_fm_v04_clean_v1_pos_long_1m

PYTHONPATH=/workspace/upper_body_motion_roadmap \
nohup python -u -m training.scripts.train_ula_fm \
  --config configs/train_v04_s3_example.yaml \
  > training/runs/ula_fm_v04_clean_v1_pos_long_1m/train_stdout.log \
  2> training/runs/ula_fm_v04_clean_v1_pos_long_1m/train_stderr.log &

echo $! > training/runs/ula_fm_v04_clean_v1_pos_long_1m/train.pid
```

## Monitor Training

```bash
tail -f /workspace/upper_body_motion_roadmap/training/runs/ula_fm_v04_clean_v1_pos_long_1m/train_stdout.log
```

Latest preview videos:

```bash
find /workspace/upper_body_motion_roadmap/training/runs/ula_fm_v04_clean_v1_pos_long_1m/previews \
  -name "long_motion_original_v2.mp4" | sort | tail
```

## Upload Training Artifacts Back To S3

```bash
aws s3 sync \
  /workspace/upper_body_motion_roadmap/training/runs/ula_fm_v04_clean_v1_pos_long_1m \
  s3://YOUR_BUCKET/upper_body_motion_roadmap/training/runs/ula_fm_v04_clean_v1_pos_long_1m
```

For frequent checkpoint sync during long runs:

```bash
aws s3 sync \
  /workspace/upper_body_motion_roadmap/training/runs/ula_fm_v04_clean_v1_pos_long_1m \
  s3://YOUR_BUCKET/upper_body_motion_roadmap/training/runs/ula_fm_v04_clean_v1_pos_long_1m \
  --exclude "previews/*" \
  --include "progress.jsonl" \
  --include "train_log.json" \
  --include "ula_fm_checkpoint.pt"
```

## Notes

- The current clean dataset is about 110 MB locally, so S3 transfer is small.
- MuJoCo preview rendering on a headless server may require EGL/OSMesa setup depending on the machine image.
- If preview rendering fails on the server, set `preview.every_steps: 0` for pure training, then run inference/render on a machine with MuJoCo rendering support.
