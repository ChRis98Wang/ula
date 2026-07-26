# GMR to ULA V2

This adapter retargets an Xsens BVH clip with the official GMR/Mink solver and
exports the ULA V2 15-joint CSV contract at 30 Hz. The robot has a fixed base,
so targets are anchored at the torso and joint values are extracted by name.

Run retargeting with the isolated environment:

```bash
/home/gez/shuaiwang/.venvs/gmr/bin/python \
  tools/gmr_v2/retarget_xsens_v2.py
```

Render the safe trajectory with the project's real V2 URDF:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH="$PWD" \
conda run --no-capture-output -n env_isaaclab \
python -m upper_body_skeleton.mujoco_playback \
  --joint-csv deliverables/gmr_v2_quality/boxing_xsens/boxing_gmr_safe_15d.csv \
  --output-mp4 deliverables/gmr_v2_quality/boxing_xsens/boxing_gmr_mujoco.mp4 \
  --summary-json deliverables/gmr_v2_quality/boxing_xsens/render_summary.json \
  --fps 30 --width 1280 --height 720
```

The MP4 is a kinematic pose preview. It checks retargeting direction, motion
smoothness, joint limits, and visible intersections; it does not validate
torque limits, balance, or closed-loop controller tracking.

For Motion-X++ HAA500 SMPL-X 322D clips, use
`retarget_motionx322_v2.py`. It derives a constant anatomical coordinate
alignment per clip and intentionally ignores all head, face, and finger
channels. The SMPL-X body model is separately licensed and is therefore kept
outside the repository under the NAS collection's `models/` directory.

## BEAT2 interaction windows

The BEAT2 batch runner selects only aligned, non-static, low-dynamic windows
from the interaction inventory and keeps physical retarget quality separate
from semantic training admission:

```bash
/home/gez/shuaiwang/.venvs/gmr/bin/python \
  tools/gmr_v2/batch_retarget_beat2_v2.py --workers 2
```

Resume an interrupted batch without recomputing hash-validated passes:

```bash
/home/gez/shuaiwang/.venvs/gmr/bin/python \
  tools/gmr_v2/batch_retarget_beat2_v2.py --workers 2 --resume
```

Add `--retry-failed` only when failed tasks should be attempted again. The
runner atomically maintains `status.json` plus mutually exclusive
`passed_manifest.jsonl`, `failed_manifest.jsonl`, and
`pending_manifest.jsonl`; source records outside the selection contract are
written to `excluded_manifest.jsonl`. A physical pass always remains
`accepted_for_training=false` until the separate semantic review admits it.

## BEAT2 variable-length semantic events

The English semantic-event inventory is not a six-second window dataset. Run
the grouped adapter so one worker loads each source NPZ once and processes all
of that source's native-length events. SMPL-X and GMR instances are cached per
worker, while IK reset/warmup, smoothing, retiming, quality control, and output
publication remain independent for every event:

```bash
/home/gez/shuaiwang/.venvs/gmr/bin/python \
  tools/gmr_v2/batch_retarget_beat2_semantic_events_v2.py \
  --inventory /home/gez/nas/cloud/gez/human_motion/catalog/beat2_semantic_event_pilot_v4/beat2_semantic_event_pilot_v4.selected.jsonl \
  --beat2-root /home/gez/nas/cloud/gez/human_motion/raw/BEAT2 \
  --output-root /home/gez/nas/cloud/gez/human_motion/processed/beat2_semantic_event_pilot_18d_v4 \
  --workers 4
```

Use `--limit-events` for a global event cap. Resume with `--resume`; add
`--retry-failed` only to retry terminal failures. Fixed-duration semantic
records are denied by this runner even if they appear in another inventory.
The original `training_segment` is never overwritten when velocity-safe
retiming changes frame count. Result and quality records add a self-hashed
`retarget_segment` containing source bounds/count, output count/duration,
30 Hz, `retimed`, and `cropped=false`. It distinguishes the planner target
`output_sample_span_sec=(N-1)/fps` from half-open frame coverage
`output_frame_coverage_sec=N/fps`; legacy quality `duration_sec` is
compatibility-only.

The current v4 artifact is a schema/physical-QC pilot selected from the
partial inventory with source-manifest SHA
`21d185bde9b1d3628f061f93a1aa6c0666965331508565ecdc2c234c9ae64395`.
Its `PARTIAL_INVENTORY_SCHEMA_PILOT_ONLY.json` marker forbids treating it as a
final training pool. Regenerate selection and all downstream evidence after
the complete inventory is published.
