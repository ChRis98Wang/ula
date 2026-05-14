# Upper-Body Language-Action Deliverables

This is the clean entry point for checking the current dataset, model training,
generated motion, and MuJoCo playback.

## Main Folders

- `01_datasets/`
  Final LeRobot-style Parquet dataset and body-only semantic JSONL.
- `02_codebook_v1_train_run/`
  Current recommended trained model run.
- `03_weights/`
  Checkpoint-only entry point.
- `04_network_demo.html`
  HTML architecture and data-flow demo.
- `generated_motion_previews/`
  Rendered MuJoCo videos for generated motions.

## Recommended Dataset For Training

```text
/Users/demo/Desktop/upper_body_motion_roadmap/datasets/lerobot_v2_upper_body_body_only_final
```

Shape:

- Episodes: 51,101
- Frames: 6,132,120
- Tasks: 14
- Action/state dimension: 15 V2 upper-body joints
- Data parquet files: 25

## Recommended Checkpoint

```text
/Users/demo/Desktop/upper_body_motion_roadmap/deliverables/02_codebook_v1_train_run/ula_fm_checkpoint.pt
```

The cleaner checkpoint path is:

```text
/Users/demo/Desktop/upper_body_motion_roadmap/weights/codebook_v1_checkpoint.pt
```

Training summary:

- Episodes loaded: 24,000
- Steps: 3,000
- Device: MPS
- Final loss: 0.14353804290294647

## Generated Motion Preview

```text
/Users/demo/Desktop/upper_body_motion_roadmap/deliverables/generated_motion_previews/nervous_explain_mujoco.mp4
```

The source generated joint trajectory is:

```text
/Users/demo/Desktop/upper_body_motion_roadmap/deliverables/02_codebook_v1_train_run/generated_nervous_explain.csv
```

## Train More

For a stronger run, start from the final dataset and increase training steps:

```bash
/Users/demo/Desktop/upper_body_motion_roadmap/training/scripts/train_codebook_v1_long.sh
```

## Generate A New Motion

```bash
/Users/demo/Desktop/upper_body_motion_roadmap/training/scripts/infer_codebook_v1.sh \
  "紧张地解释，同时双手做克制的上肢手势" \
  generated_custom
```

## Render A Generated Motion In MuJoCo

```bash
/Users/demo/Desktop/upper_body_motion_roadmap/training/scripts/render_generated_motion.sh \
  /Users/demo/Desktop/upper_body_motion_roadmap/training/runs/codebook_v1/generated_custom.csv \
  generated_custom
```
