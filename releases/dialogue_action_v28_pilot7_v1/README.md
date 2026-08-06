# Dialogue Action V28 Pilot7 v1

This directory freezes the latest validated V28 Pilot7 dialogue-action network and
its network-versus-GT demonstration video.

## Status and scope

- Status: pilot checkpoint, not a formal general-purpose release.
- Supported intent IDs: `idle_attentive`, `search_scan`.
- Training GT: 7 double-reviewed trajectories (3 idle, 4 search).
- Output length: variable; any requested length of at least 2 frames is accepted.
- Runtime generation does not read the training dataset or source CSV files.
- The unfinished V29/Qwen text-generalization work is intentionally excluded.

The checkpoint is intended as a reproducible quality baseline. It reconstructs
the reviewed trajectories closely, but it does not yet understand arbitrary text
or provide broad gesture coverage.

## Files

- `model/ula_fm_checkpoint.pt`: complete V28 Pilot7 checkpoint, stored with Git LFS.
- `video/interactive_command_demo.mp4`: 29.8-second H.264 comparison demo.
- `training_summary.json`: frozen training and provenance metrics.
- `evaluation_report.json`: per-realization video evaluation receipt.
- `SHA256SUMS`: artifact integrity hashes.

In the video, the left robot is the V28 network output and the right robot is the
reviewed GT. It contains four representative cases at their native, variable
trajectory lengths.

## Metrics

- Best training step: 6800.
- Maximum network-versus-formal-GT RMS during training: `0.0005454293` rad.
- Maximum network-versus-GT RMS in the rendered evaluation: `0.0005381698` rad.
- Mean network-versus-GT RMS in the rendered evaluation: `0.0001912162` rad.
- Minimum diversity retention: `0.9994489`.
- Minimum cross-intent MSE: `0.03669786` rad^2.

## Download

Git LFS is required for the model file:

```bash
git lfs install
git clone git@github.com:ChRis98Wang/ula.git
cd ula
git checkout codex/ula-mmdit-lite
git lfs pull
sha256sum -c releases/dialogue_action_v28_pilot7_v1/SHA256SUMS
```

## Python runtime

Run from the repository root with PyTorch and NumPy installed:

```python
from upper_body_skeleton.dialogue_action_runtime_v28 import DialogueActionV28Runtime

checkpoint = (
    "releases/dialogue_action_v28_pilot7_v1/model/ula_fm_checkpoint.pt"
)
runtime = DialogueActionV28Runtime(checkpoint, device="cpu")

# Variable-length [frames, 18] joint-angle trajectory.
motion = runtime.generate("search_scan", frames=180, seed=0)
print(motion.shape)
```

`seed` selects one of the reviewed realizations for the requested intent. Generated
joint angles are clipped to the repository's 18-DoF joint limits.
