# 0514 Close V2 Clone Demo

This repo includes the 0514 close-hand V2 upper-body inference code, MuJoCo playback code, robot URDF assets, and the trained checkpoint needed for a local demo. The training dataset is not committed.

## Included

- Checkpoint: `training/runs/ula_fm_0514_close_v2_1m/ula_fm_checkpoint.pt`
- Training entrypoint: `training/scripts/train_ula_fm.py`
- Inference entrypoint: `upper_body_skeleton.long_emotion_infer`
- MuJoCo playback entrypoint: `upper_body_skeleton.mujoco_playback`
- Robot assets: `urdf_V2_20260514/`

## Not Included

- `datasets/`
- `data_build/`
- Training run logs and preview outputs

To retrain, place the clean LeRobot dataset at:

```text
datasets/lerobot_v2_upper_body_0514_close_v2_clean
```

## Python Environment

Use Python 3.13 with these packages available:

```zsh
python -m pip install -r requirements-0514-demo.txt
```

On macOS, use `mjpython` for commands that open the MuJoCo viewer.

## Generate Motion And MP4

Run from the repo root:

```zsh
env PYTHONPATH="$PWD" python -m upper_body_skeleton.long_emotion_infer \
  --checkpoint "$PWD/training/runs/ula_fm_0514_close_v2_1m/ula_fm_checkpoint.pt" \
  --text "开心地挥手" \
  --output-dir "$PWD/deliverables/long_emotion_previews/clone_demo" \
  --min-segment-sec 3 \
  --max-segment-sec 3 \
  --min-segments 4 \
  --max-segments 4 \
  --max-duration-sec 12 \
  --sampling-steps 16 \
  --width 960 \
  --height 540
```

Outputs:

```text
deliverables/long_emotion_previews/clone_demo/long_motion.csv
deliverables/long_emotion_previews/clone_demo/long_motion.npz
deliverables/long_emotion_previews/clone_demo/plan.json
deliverables/long_emotion_previews/clone_demo/summary.json
deliverables/long_emotion_previews/clone_demo/long_motion_original_v2.mp4
```

## Open MuJoCo Viewer

Use `mjpython` on macOS:

```zsh
env PYTHONPATH="$PWD" mjpython -m upper_body_skeleton.long_emotion_infer \
  --checkpoint "$PWD/training/runs/ula_fm_0514_close_v2_1m/ula_fm_checkpoint.pt" \
  --text "开心地挥手" \
  --output-dir "$PWD/deliverables/long_emotion_previews/clone_viewer" \
  --min-segment-sec 3 \
  --max-segment-sec 3 \
  --min-segments 4 \
  --max-segments 4 \
  --max-duration-sec 12 \
  --sampling-steps 16 \
  --no-render \
  --viewer \
  --viewer-loops 0
```

To play an existing generated CSV:

```zsh
env PYTHONPATH="$PWD" mjpython -m upper_body_skeleton.mujoco_playback \
  --joint-csv "$PWD/deliverables/long_emotion_previews/clone_demo/long_motion.csv" \
  --viewer \
  --loops 0
```

## Retrain

After placing the clean dataset under `datasets/`, run:

```zsh
env PYTHONPATH="$PWD" python -m training.scripts.train_ula_fm \
  --config "$PWD/configs/train_0514_close_v2.yaml"
```

For a two-step smoke test:

```zsh
env PYTHONPATH="$PWD" python -m training.scripts.train_ula_fm \
  --config "$PWD/configs/train_0514_close_v2_smoke.yaml"
```
