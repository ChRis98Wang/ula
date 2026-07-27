# BEAT2 emotion blind visual review

`build_beat2_emotion_review_queue_v1.py` produces a **controller-only**
candidate queue. Its 598 current records do not contain a per-record official
emotion field, but reviewers still must not receive it: absolute trajectory
paths, source-group tokens, split assignments, category balance keys, and
duration strata are controller metadata.

`build_beat2_emotion_blind_review_bundle_v1.py` is the projection boundary. It
reuses:

- `render_beat2_annotation_review.py` for silent, verified EGL/MuJoCo video;
- `build_blind_review_shards_v1.assign_shards` for deterministic balanced
  shards.

It explicitly rejects any path containing `Kimodo` (case-insensitive). It also
rejects a controller record if a source/official emotion field is present, even
when the record falsely claims that the field is absent.

## 1. Prepare the controller-only render queue

This validates and SHA-binds all source rows and trajectories, but does **not**
render videos:

```bash
PYTHONPATH=. /home/gez/miniconda3/envs/env_isaaclab/bin/python \
  tools/human_motion_review/build_beat2_emotion_blind_review_bundle_v1.py \
  --controller-queue deliverables/beat2_emotion_human_review_queue_v1/review_queue.jsonl \
  --output-root deliverables/beat2_emotion_human_review_queue_v1/blind_visual_review_v1 \
  --shard-count 6
```

The renderer input is:

```text
deliverables/beat2_emotion_human_review_queue_v1/
  blind_visual_review_v1/controller_only/render_queue.jsonl
```

It uses only opaque BEAT2 sample/group tokens and a generic observation prompt.
It carries no source emotion, emotion prompt, category prompt, speech, or
training-admission flag.

## 2. Render a smoke sample

Do not accidentally launch all 598 records while checking the environment.
This command renders exactly one deterministic queue entry:

```bash
PYTHONPATH=. /home/gez/miniconda3/envs/env_isaaclab/bin/python \
  tools/human_motion_review/render_beat2_annotation_review.py \
  --review-queue deliverables/beat2_emotion_human_review_queue_v1/blind_visual_review_v1/controller_only/render_queue.jsonl \
  --output-root deliverables/beat2_emotion_human_review_queue_v1/smoke_render_v1 \
  --limit 1 \
  --sampling sequential \
  --workers 1 \
  --width 640 \
  --height 360
```

Rendering uses EGL and writes silent H.264/yuv420p videos. A render pass remains
`accepted_for_training=false`.

## 3. Materialize reviewer-visible primary shards

Pass the renderer's verified `passed_manifest.jsonl` back through the same
projection tool:

```bash
PYTHONPATH=. /home/gez/miniconda3/envs/env_isaaclab/bin/python \
  tools/human_motion_review/build_beat2_emotion_blind_review_bundle_v1.py \
  --controller-queue deliverables/beat2_emotion_human_review_queue_v1/review_queue.jsonl \
  --output-root deliverables/beat2_emotion_human_review_queue_v1/blind_visual_review_v1 \
  --render-passed-manifest deliverables/beat2_emotion_human_review_queue_v1/smoke_render_v1/passed_manifest.jsonl \
  --shard-count 6
```

Only distribute the `reviewer_visible/primary_1/shard_*` and
`reviewer_visible/primary_2/shard_*` directories. Never distribute
`controller_only/`, the source queue, renderer logs, or the renderer's passed
manifest. Anonymous videos are hard-linked when possible and otherwise copied;
their SHA256 binding is identical to the verified render.

Each public task contains only:

- opaque sample and assignment IDs;
- an anonymously named video plus video and sample-binding hashes;
- fixed observability/emotion vocabularies;
- blank human-judgment fields;
- fail-closed protocol flags.

It contains no source label, source identity, split, category, duration,
trajectory path, or controller sample ID.

## Reviewer assignment protocol

Use separate reviewer pools for `primary_1` and `primary_2`. This guarantees
that the two reviewer IDs for any sample are distinct. A reviewer fills:

- `observability`: `observable`, `not_observable`, or `ambiguous`;
- `observed_emotion`: one of the six fixed classes only when observable,
  otherwise `null`;
- `confidence`: a number from 0 through 1;
- `reviewer_id` and `submitted_at_utc`.

Any disagreement in observability **or** observed emotion requires a third,
label-blind adjudicator whose reviewer ID differs from both primary reviewers.
The third reviewer must not see either primary decision. Therefore
`reviewer_visible/adjudication/pending_queue.jsonl` intentionally stays empty
until primary submissions are compared by the controller.

No review, agreement, adjudication, render, or bundle-building step in this
workflow directly enables training. A separately validated merge/admission
stage is still required.
