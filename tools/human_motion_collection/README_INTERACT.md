# InterAct motion-only pipeline

This pipeline is intentionally separate from BEAT2. It consumes only the 482
paired BVH files plus `README.md`, `actors.db`, and `scenarios.db` from pinned
Hugging Face revision
`152ba832f379c465f5b1e10c67166d646014d675`.

Audio, face parameters, finger targets, retargeted BVHs, and source renders are
not selected. The source license is CC-BY-NC-SA-4.0. Complete hashes do not
authorize training: the acquisition tool requires a separately approved
`interact_noncommercial_use_confirmation` receipt and otherwise writes
`formal_training_blocked: true`.

## Reproducible audit

```bash
python tools/human_motion_collection/acquire_interact_motion_only.py \
  --root /home/gez/nas/cloud/gez/human_motion/raw/InterAct \
  --receipt /home/gez/nas/cloud/gez/human_motion/catalog/interact_motion_only_acquisition.json \
  --require-complete

python tools/human_motion_collection/build_interact_dyadic_turns.py \
  --root /home/gez/nas/cloud/gez/human_motion/raw/InterAct \
  --acquisition-receipt /home/gez/nas/cloud/gez/human_motion/catalog/interact_motion_only_acquisition.json \
  --output-dir /home/gez/nas/cloud/gez/human_motion/catalog/interact_dyadic_turn_v1 \
  --pilot-performance-count 4
```

The inventory maps `<date>_<actor>_<scenario>.bvh` through both SQLite
databases and requires exactly two session-matched actors, readable 30 Hz
headers, and equal partner frame counts. It never truncates one partner to fit
the other.

Pilot boundaries use the maximum of the two partners' RMS angular speeds over
the torso, head/neck, arms, forearms, and hands. A cut is proposed only inside
a shared joint-rest basin. A recording with no internal shared-rest basin is
kept whole, regardless of elapsed time. There is no target, minimum, maximum,
or fixed candidate duration.

Scenario, relationship, and primary emotion remain source provenance. All
semantic, emotion, physical-QC, and admission masks are false. Independent
silent robot-video reviews must later establish a complete onset/apex/offset,
observable action semantics, and observable affect.

## 18D adapter boundary

`tools/gmr_v2/interact_bvh_adapter.py` maps:

| InterAct | Existing ULA retarget source name |
|---|---|
| `Spine3` | `Chest4` |
| `Left/RightArm` | `Left/RightShoulder` |
| `Left/RightForeArm` | `Left/RightElbow` |
| `Left/RightHand` | `Left/RightWrist` |

It also preserves `Spine3 -> Neck -> Neck1 -> Head` and derives an unaligned
native three-DoF head trajectory. Parser/geometry smoke success does not admit
retargeting: InterAct-to-robot axis direction must first pass a rendered
MuJoCo review.
