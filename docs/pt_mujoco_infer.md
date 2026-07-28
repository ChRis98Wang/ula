# PT Direct MuJoCo Inference

`upper_body_skeleton.pt_mujoco_infer` loads a ULA trajectory-generator checkpoint, runs flow inference in memory,
postprocesses the resulting `[frames, 15 joints]` array, and passes that array directly to
`MujocoMotionPlayer.play_trajectory()`. It does not write an intermediate CSV or NPZ.

## Checkpoint Types

- Default: `training/runs/kimodo_mmdit_lite_qwen_compatible_5k_math_sdp/ula_fm_checkpoint.pt` together with the
  deployment Qwen semantic adapter. This checkpoint completed 5,000 normalized-action steps with the math SDPA
  backend and records the exact semantic condition contract.
- Unsupported: `training/runs/kimodo_motion_latent_v1/motion_latent_checkpoint.pt`. That model encodes an existing
  trajectory into a latent vector and has no trajectory decoder.

The earlier 1M-step attempt used the default memory-efficient SDPA backend and stopped after a CUDA backward
instability. The 5,000-step replacement forces math SDPA and completed with finite loss and gradients.

## Interactive Viewer

The viewer needs a graphical session. From a client with an X server, connect with X11 forwarding:

```bash
ssh -Y gez@172.16.60.184
cd /home/gez/shuaiwang/ula-motion-generate
echo "$DISPLAY"

conda run --no-capture-output -n env_isaaclab \
  python -u -m upper_body_skeleton.pt_mujoco_infer \
  --kimodo-qwen \
  --device cuda \
  --semantic-device cuda \
  --semantic-local-files-only \
  --loops 1
```

Enter a text prompt and press Enter. The PT generates a five-second motion in memory and the same MuJoCo window
plays it. Enter `:q` to exit. The camera remains interactive through the standard MuJoCo mouse controls.

Run one detailed prompt instead of opening the terminal prompt loop:

```bash
conda run --no-capture-output -n env_isaaclab \
  python -u -m upper_body_skeleton.pt_mujoco_infer \
  --kimodo-qwen \
  --text "开心地跟主人点头并在胸前挥手打招呼" \
  --device cuda \
  --semantic-device cuda \
  --semantic-local-files-only \
  --loops 0
```

Use the 5,000-step Kimodo checkpoint explicitly:

```bash
conda run --no-capture-output -n env_isaaclab \
  python -u -m upper_body_skeleton.pt_mujoco_infer \
  --kimodo-experimental \
  --semantic-adapter-checkpoint training/runs/kimodo_qwen3_semantic_adapter_deploy_v1/semantic_adapter_checkpoint.pt \
  --behavior-id Behavior.GreetingOwner01 \
  --emotion-id happy \
  --device cuda
```

The no-flag default and `--kimodo-qwen` shortcut both select the current Kimodo generator and deployment adapter.
The explicit equivalent is:

```bash
conda run --no-capture-output -n env_isaaclab \
  python -u -m upper_body_skeleton.pt_mujoco_infer \
  --kimodo-experimental \
  --semantic-adapter-checkpoint training/runs/kimodo_qwen3_semantic_adapter_deploy_v1/semantic_adapter_checkpoint.pt \
  --device cuda \
  --semantic-device cuda \
  --semantic-local-files-only \
  --loops 1
```

This interactive form accepts one command per line and prints the resolved behavior, emotion, and confidence after playback.
The selected generator is the completed 5,000-step model.

## Headless Check

An ordinary SSH session has no GUI. `--no-viewer` verifies checkpoint loading and CUDA inference without writing
trajectory artifacts:

```bash
conda run --no-capture-output -n env_isaaclab \
  python -u -m upper_body_skeleton.pt_mujoco_infer \
  --kimodo-qwen \
  --no-viewer \
  --text "开心地跟主人点头并在胸前挥手打招呼" \
  --frames 150 \
  --device cuda \
  --semantic-device cuda \
  --semantic-local-files-only
```

`--frames` is required here: the 5,000-step Kimodo checkpoint's `condition_dim` (136) is below the
264-dimensional V2 contract that triggers automatic duration prediction, so `infer_motion` raises
`frames must be explicit for checkpoints without a trained duration head` if it is omitted. 150 frames
at the default 30 fps matches the five-second motion described above. Verified working 2026-07-26 against
`training/runs/kimodo_mmdit_lite_qwen_compatible_5k_math_sdp/ula_fm_checkpoint.pt` under `env_isaaclab`
with no other environment variables required (no `MUJOCO_GL` needed since `--no-viewer` never opens a
renderer).

## Dataset-Label MuJoCo Comparison

This command uses the Motion Metric Encoder checkpoint's `test` partition to select a reference episode, passes that
episode's exact `meta/semantic_index.parquet:language_instruction` text to the PT generator, and renders both
trajectories directly from memory. The network output is on the left and the matching dataset `observation.state`
trajectory is on the right:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
conda run --no-capture-output -n env_isaaclab \
  python -u -m upper_body_skeleton.pt_dataset_mujoco_compare \
  --behavior-id Behavior.GreetingOwner01 \
  --emotion-id happy \
  --motion-latent-split test \
  --device cuda \
  --semantic-device cuda \
  --sampling-steps 32
```

The command requires the text adapter's predicted behavior and emotion to match the dataset labels, verifies all
encoded frames and both rendered panes, and writes the MP4 plus JSON provenance under
`training/runs/kimodo_mmdit_lite_qwen_compatible_5k_math_sdp/mujoco_dataset_comparison/`.

This comparison uses the current 5,000-step, 136-dimensional Kimodo MMDiT generator. The separate 128-dimensional
Qwen LoRA alignment checkpoint is not condition-compatible with that generator and is therefore not silently used.
The current MMDiT trained on all 1,620 dataset episodes, so the Motion Metric `test` partition is only a deterministic
way to select references. It is not a held-out MMDiT generalization score; this limitation is also recorded in every
output JSON file.

To render ten different label/episode references from the Motion Metric `test` partition in one model-loading pass,
omit the label filter and set the count:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
conda run --no-capture-output -n env_isaaclab \
  python -u -m upper_body_skeleton.pt_dataset_mujoco_compare \
  --motion-latent-split test \
  --count 10 \
  --device cuda \
  --semantic-device cuda
```

For the concrete core-wave subset, select both greeting wave behaviors across all six emotions. This produces twelve
test episodes and a `dataset_selection.json` manifest alongside the videos:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
conda run --no-capture-output -n env_isaaclab \
  python -u -m upper_body_skeleton.pt_dataset_mujoco_compare \
  --behavior-id Behavior.GreetingOwner01 \
  --behavior-id Behavior.GreetingOwner04 \
  --motion-latent-split test \
  --count 12 \
  --output-dir training/runs/kimodo_mmdit_lite_qwen_compatible_5k_math_sdp/mujoco_wave_core_test \
  --device cuda \
  --semantic-device cuda
```
