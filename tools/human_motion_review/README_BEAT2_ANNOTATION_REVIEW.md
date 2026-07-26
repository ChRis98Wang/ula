# BEAT2 annotation review videos

`render_beat2_annotation_review.py` turns a validated BEAT2 annotation review
queue into silent MuJoCo videos for independent motion/text review. Rendering is
an inspection aid only. A passed video never changes `accepted_for_training`
from `false`.

## Input

Generate `review_queue.jsonl` with
`validate_beat2_annotation_batch.py`. Each record must retain the ULA V2 18D
trajectory and quality-evidence hashes, an English/Chinese robot-observable
prompt, speaker/split provenance, and the pending-review admission flags.
Speech context is neither copied into the sampled review manifest nor used as an
action prompt.

For the native-length semantic-event pipeline, build the queue directly from
one or more physical-QC passed manifests. This adapter verifies trajectory and
quality hashes and rejects any row that enables unreviewed prompt or intent
supervision:

```bash
PYTHONPATH=. /home/gez/shuaiwang/.venvs/gmr/bin/python \
  tools/human_motion_review/build_semantic_event_video_queue.py \
  --passed-manifest /path/to/retarget/passed_manifest.jsonl \
  --output /path/to/review/review_queue.jsonl
```

## Stratified sample

```bash
PYTHONPATH=. /home/gez/shuaiwang/.venvs/gmr/bin/python \
  tools/human_motion_review/render_beat2_annotation_review.py \
  --review-queue deliverables/interactive_human_motion_v1/batch_annotation_v1/review/review_queue.jsonl \
  --output-root deliverables/interactive_human_motion_v1/batch_annotation_v1/review/videos_v1 \
  --limit 24 \
  --sampling stratified \
  --workers 1 \
  --resume
```

Native semantic-event strata combine speaker, fixed split, official emotion,
official category, and official intensity. Other queues combine
`speaker_key`, `official_split`, and available observable-motion features; if
feature fields are absent, `canonical_action` is used. `--seed` controls
deterministic ordering. Use `--sampling sequential` for a sorted task-id order.

Use `--sampling duration_quantiles --limit 3` to render the shortest, median,
and longest native semantic events. This explicitly verifies that the pipeline
preserves event-defined duration instead of imposing a fixed window.

## Full queue

Omit `--limit` to render every queue record. Resume revalidates existing passed
videos before reuse. Existing failures remain terminal unless
`--retry-failed` is supplied.

The default renderer is
`/home/gez/miniconda3/envs/env_isaaclab/bin/python`; override it with
`--renderer-python` when needed. Each renderer subprocess uses EGL, the real
ULA V2 URDF, 30 Hz, 1280x720 output, automatic full-body framing, a 1.12 camera
margin, and a -0.06 m camera look-at Z offset.

## Outputs

- `sampled_manifest.jsonl`: deterministic, speech-free review selection.
- `videos/<task_id>.mp4`: silent H.264/yuv420p video.
- `render_summaries/<task_id>.json`: MuJoCo model and framing evidence.
- `logs/<task_id>/<run_id>.log`: per-item renderer log.
- `results/<task_id>.json`: resumable per-item result.
- `passed_manifest.jsonl` and `failed_manifest.jsonl`: closed render result sets.
- `status.json` and `summary.json`: progress and final counts.

Every passed video is fully decoded and checked for exact CSV frame count,
30 Hz, configured dimensions, a nonblank image, one H.264/yuv420p video stream,
no audio stream, and a faststart MP4 layout. Manual motion/text review remains
required afterward.

## Blind affect review

The action/text queue above is not affect-blind: it contains the prompt,
official category, and source emotion. Never use it to enable affect
conditioning. Build a separate A/B bundle from render-passed records:

```bash
PYTHONPATH=. /home/gez/shuaiwang/.venvs/gmr/bin/python \
  tools/human_motion_review/build_blind_affect_review_bundle.py \
  --render-passed-manifest /path/to/render/passed_manifest.jsonl \
  --output-root /path/to/reviewer_visible_bundle \
  --hidden-root /path/to/controller_only_bundle_B
```

Bundle A contains only opaque `sample_id` values, anonymously named videos,
video hashes, and blank review fields. Reviewers submit `observed_affect`
(`neutral`, `sad`, `happy`, `angry`, `surprise`, `fear`, `not_observable`, or
`ambiguous`), confidence, reviewer ID, and protocol version. Bundle B alone
maps anonymous samples to task/source/category/official-emotion metadata and
must not be distributed to reviewers. Match official emotion only after blind
submission; render or physical-QC success never upgrades affect supervision.
