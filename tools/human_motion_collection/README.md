# Human Motion Collection

Initialize and inventory the NAS collection:

```bash
python tools/human_motion_collection/build_catalog.py \
  --root /home/gez/nas/cloud/gez/human_motion
```

The `raw` layer is immutable. A clip is eligible for training only after it has
been converted to the V2 15-joint contract, passed limit/velocity/collision QC,
been manually reviewed, and assigned to subject and action holdout splits.

The legacy V2 15D adapter ignores head, face, and finger channels. The
\`ula_v2_18d_head_v1\` contract adds three head joints while still ignoring
face and finger channels.

## BEAT2 motion-only acquisition

Download the pinned English, Japanese, and Spanish BEAT2 motion subsets plus
their text/semantic alignment metadata:

```bash
/home/gez/miniconda3/envs/env_isaaclab/bin/python \
  tools/human_motion_collection/download_beat2_motion_only.py \
  --max-workers 12
```

This acquisition is fail-closed and excludes `wave16k` audio and pretrained
weights. The SMPL-X/FLAME source arrays still contain face and finger channels,
but the 18D retarget adapter reads only root/body pose and maps only the robot's
15 body plus 3 head joints. The resulting acquisition manifest is written to
`/home/gez/nas/cloud/gez/human_motion/catalog/beat2_motion_only_acquisition.json`.

When the files already exist locally, build the same receipt without a network
request from a metadata-only clone at the pinned revision. This verifies every
selected local file against either its Git LFS SHA-256 OID or its ordinary Git
blob content hash:

```bash
/home/gez/miniconda3/envs/env_isaaclab/bin/python \
  tools/human_motion_collection/download_beat2_motion_only.py \
  --languages english \
  --components smplxflame_30 labels textgrid metadata \
  --metadata-git-repo /tmp/beat2-meta-full \
  --verify-existing-only --max-workers 1
```

The receipt records declared license metadata but deliberately leaves dataset
training-use and SMPL-X terms review pending; a content hash receipt is not a
license approval.

The following six-second multilingual inventory is a legacy weak-label path and
must not be used for formal interaction training. It remains available only for
historical comparison and motion-only diagnostics:

```bash
/home/gez/miniconda3/envs/env_isaaclab/bin/python \
  tools/human_motion_collection/build_beat2_multilingual_motion_inventory.py \
  --require-acquisition-manifest --workers 8
```

The output directory is
`/home/gez/nas/cloud/gez/human_motion/catalog/beat2_multilingual_motion_only_v1`.
The full JSONL keeps every non-overlapping motion/TextGrid-valid window. The
`priority_manifest.jsonl` keeps at most one non-static, low-dynamic window per
source clip for the first retarget pass. English priority prefers windows with
an official `sem` span other than `01_beat_align`; Japanese and Spanish text
remains speech context only. Both manifests keep emotion supervision disabled.
Per-source state under `state/` is reused only while file size, modification
time, official split, and window contract still match.

Retarget the priority manifest with the existing motion-only-compatible batch
runner:

```bash
/home/gez/shuaiwang/.venvs/gmr/bin/python \
  tools/gmr_v2/batch_retarget_beat2_v2.py \
  --inventory /home/gez/nas/cloud/gez/human_motion/catalog/beat2_multilingual_motion_only_v1/priority_manifest.jsonl \
  --beat2-root /home/gez/nas/cloud/gez/human_motion/raw/BEAT2 \
  --output-root /home/gez/nas/cloud/gez/human_motion/processed/beat2_multilingual_18d_priority_v1 \
  --workers 4 --resume
```

## BEAT2 native-length interaction events

The six-second multilingual inventory above is not the formal interaction
training source. For English BEAT2, build clips around complete official
deictic, iconic, and metaphoric events. Event boundaries determine duration;
there is no fixed duration and no maximum-duration crop:

```bash
/home/gez/miniconda3/envs/env_isaaclab/bin/python \
  tools/human_motion_collection/build_beat2_semantic_event_inventory.py \
  --workers 4
```

Create the reproducible pilot before retargeting. It assigns every source from
one speaker to exactly one of `train`, `validation`, or `test`, balances all six
network emotions over gesture category and intensity, and prefers low/medium
dynamic interaction events:

```bash
python tools/human_motion_collection/select_beat2_semantic_event_pilot.py
```

The selector writes candidate, selected, rejected, per-split, assignment, and
summary artifacts under
`/home/gez/nas/cloud/gez/human_motion/catalog/beat2_semantic_event_pilot_v7_full`.
Every selected row remains `accepted_for_training=false` pending independent
18D retarget QC and video/text review. Official emotion provenance is hashed;
the dataset-scope interaction behavior is explicitly weak and its behavior
condition channels must remain masked. The deterministic English Qwen prompt
contains only official gesture category, intensity, and filename-protocol
emotion. Any official lexical anchor is retained separately as speech context
and cannot override the emotion label or enter the canonical prompt.
`official_category_verified=true` means verified metadata for fixed splitting
and evaluation only; it does not enable conditioning or loss. All five
`semantic_supervision_masks` (`official_category`,
`robot_observable_motion_form`, `communicative_intent`, `prompt_text`, and
`legacy_gesture`) remain false. Source emotion is also metadata-only:
`emotion_supervision_mask=false`, official-emotion conditioning is disabled,
and `affect_observable_supervision_mask=false`. Robot-observable motion form,
communicative intent, and affect observability remain `candidate_unreviewed`.
Physical QC cannot enable any semantic, intent, prompt, or affect supervision.

The locked v7 inventory contains 15,054 network-emotion-supported native
events. The interaction-energy-filtered low/medium training pool contains
14,973 events (902,758 frames, about 8.36 hours at 30 Hz) with 202 distinct
frame counts from 10 to 465. `provenance_lock.json` binds the acquisition
receipt, source inventory, 162-event balanced pilot, and full pool hashes.
Local experimental processing is allowed, but formal release and training
admission remain blocked while dataset/SMPL-X terms review is pending.

Retarget a first source-grouped pilot without converting events to six-second
windows:

```bash
/home/gez/shuaiwang/.venvs/gmr/bin/python \
  tools/gmr_v2/batch_retarget_beat2_semantic_events_v2.py \
  --inventory /home/gez/nas/cloud/gez/human_motion/catalog/beat2_semantic_event_pilot_v7_full/beat2_semantic_event_pilot_v7_full.selected.jsonl \
  --beat2-root /home/gez/nas/cloud/gez/human_motion/raw/BEAT2 \
  --output-root /home/gez/nas/cloud/gez/human_motion/processed/beat2_semantic_event_pilot_18d_v7_full \
  --workers 4
```

After pilot physical/video QC, retarget the full native-length pool with the
same fail-closed contract:

```bash
/home/gez/shuaiwang/.venvs/gmr/bin/python \
  tools/gmr_v2/batch_retarget_beat2_semantic_events_v2.py \
  --inventory /home/gez/nas/cloud/gez/human_motion/catalog/beat2_semantic_event_pilot_v7_full/training_pool_low_medium.jsonl \
  --beat2-root /home/gez/nas/cloud/gez/human_motion/raw/BEAT2 \
  --output-root /home/gez/nas/cloud/gez/human_motion/processed/beat2_semantic_event_training_pool_18d_v7_full \
  --workers 8
```

Build the review-only HAA500 expression shortlist:

```bash
python tools/human_motion_collection/build_expression_subset.py \
  --root /home/gez/nas/cloud/gez/human_motion
```

The shortlist is deny-by-default: every candidate starts with
`accepted_for_training=false` until retarget QC and manual video review pass.

Build deterministic robot-observable semantics for the 100 expression
candidates:

```bash
python tools/human_motion_collection/build_expression_semantics_v1.py \
  --root /home/gez/nas/cloud/gez/human_motion
```

The v1 JSONL/CSV records use the filename action as the primary label. The
GPT-4V source caption is auxiliary-only and is automatically flagged for
cross-action duplication, refusal, vague content, and action-text conflict.
Every record remains pending manual review and ineligible for training.

## ULA V2 18D observable-motion draft labels

Run the complete resumable BEAT2 batch pipeline with the isolated GMR Python:

```bash
/home/gez/shuaiwang/.venvs/gmr/bin/python \
  tools/human_motion_collection/run_beat2_annotation_pipeline.py \
  --workers 4
```

The command runs strict 18D retargeting, trajectory-only bilingual draft
labeling, batch-closure validation, and review-queue construction. Re-running
the same command resumes valid per-task outputs and verifies their hashes before
skipping them. Use `--stage retarget`, `--stage label`, or `--stage validate`
to run selected stages, and `--retry-failed` only when failed retargets should
be attempted again.

After validation, render a deterministic 24-record stratified review sample:

```bash
/home/gez/shuaiwang/.venvs/gmr/bin/python \
  tools/human_motion_collection/run_beat2_annotation_pipeline.py \
  --stage render --render-limit 24 --render-workers 1
```

This optional stage uses the real ULA V2 URDF, renders silent 30 Hz videos with
full-body framing, decodes every output frame, and still leaves every sample
pending independent motion/text review.

Batch outputs are written below
`deliverables/interactive_human_motion_v1/batch_annotation_v1/`. Physical QC
and generated text never grant training admission: the generated
`manifests/train_ready.jsonl` remains empty until independent motion/text video
review is adjudicated.

Generate conservative bilingual label drafts from quality-gated safe CSV files:

    python tools/human_motion_collection/label_ula_v2_18d_motion.py \
      --input-manifest path/to/retarget_passed.jsonl \
      --output-dir deliverables/interactive_human_motion_v1/annotations/ula_v2_18d_drafts_v1 \
      --resume

Each input JSONL record must have a unique task ID in \`task_id\`, a separate
source ID in \`source_clip_id\`, an FPS in \`fps\`, \`source_motion.fps\`, or
\`trajectory.fps\`, and an 18D CSV path in \`retarget.safe_csv\`,
\`safe_csv_path\`, \`trajectory_path\`, \`trajectory.path\`, \`safe_csv\`, or
\`motion_path\`. Admission is fail-closed: \`status\` must equal \`passed\`;
\`quality_gate.passed\` and every other embedded quality gate must be the
boolean \`true\`; and \`safe_csv_sha256\`, \`quality_json\`, and
\`quality_json_sha256\` must match the evidence files. Missing evidence is
rejected.

The tool derives only robot-observable kinematics: arm amplitude/laterality,
bilateral temporal coordination, regular repetition/continuity, and head/torso
motion. \`speech_context\` is copied for traceability with role
\`speech_context_only_not_action_label\`; it cannot affect the prompt. Outputs
are \`draft_prompts.jsonl\`, \`needs_human_review.jsonl\`, \`rejected.jsonl\`,
\`summary.json\`, and one English and Chinese TXT per task under \`text/en\` and
\`text/zh\`. All drafts remain \`accepted_for_training=false\`.
