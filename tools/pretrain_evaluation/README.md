# ULA V2 18D motion-only pretrain long-video acceptance

`build_18d_pretrain_long_video.py` creates the post-pretrain video requested for
visual acceptance. It does not run training or inference, and it cannot create
placeholder motion. It consumes completed inference outputs and binds every
frame to the formal checkpoint and immutable held-out split.

The default publication directory is:

```text
/home/gez/nas/cloud/gez/human_motion/deliverables/ula_v2_18d_pretrain_v1/long_video
```

The finished directory contains a 1920x1080, 30 Hz H.264 MP4, Chinese and
English SRT files, embedded selectable subtitle tracks, chapter text/metadata,
the individual MuJoCo renders, and `index.json` with hashes and exact frame
timings. The final directory is published only after a complete decode check.

## Required inputs

- The formal 18D checkpoint. Its checkpoint contract, release eligibility,
  training scope, native batching scope, and unsafe-data flag are checked.
- The completed formal `training_summary.json`. The best checkpoint path and
  completion/early-stop state must match.
- The immutable `split_manifest.json`. Only `validation` and `test` records are
  accepted.
- A JSONL generation manifest. Each row describes one real model sample and
  binds its CSV hash to the checkpoint hash and split-manifest hash.

Every generated CSV is the ordered ULA V2 18D joint contract, with an optional
leading `time_sec` column. If present, time must lie exactly on the 30 Hz grid.
The CSV frame count must equal the held-out episode's native requested frame
count. Cropping, padding, fixed-window keys, and fixed six-second units fail
validation. A long-video selection must include at least two distinct native
frame counts and defaults to at least 60 seconds total.

The declared native frame count is not trusted on its own: it must equal the
held-out record in the checkpoint's hash-validated `posttrain_data_contract`.
The supplied split manifest must also exactly equal the split contract embedded
in that checkpoint.

## Caption policy

This pretrain does not consume text. The video therefore never calls a caption
a prompt or model condition.

Two caption kinds are accepted:

1. `reviewed_robot_observable_text`: an independently authored/reviewed
   description based on a complete blind decode of this generated trajectory.
   The review JSONL record must bind the generated CSV hash, checkpoint hash,
   frame count, and protocol.
2. `motion_only_metadata`: explicitly unverified metadata. The subtitle is
   automatically prefixed with `不是文本条件；语义未经验证` (and the equivalent
   English warning). This is appropriate immediately after inference, before a
   generated-video review has been completed.

Dataset text, ASR text, source emotion labels, and descriptions reviewed only
against a source/reference motion are not valid reviewed captions for a newly
generated trajectory.

## Generation manifest row

Paths may be absolute or relative to the JSONL file.

```json
{
  "schema_version": "1.0.0",
  "artifact_kind": "ula_v2_18d_motion_only_pretrain_generation_v1",
  "segment_id": "heldout_001_seed7",
  "held_out_clip_id": "the-immutable-split-clip-id",
  "split": "test",
  "generated_csv": "generated/heldout_001_seed7.csv",
  "generated_csv_sha256": "<sha256>",
  "generated_from_checkpoint_sha256": "<sha256>",
  "split_manifest_sha256": "<sha256>",
  "output_kind": "model_generated_motion",
  "robot_contract": "ula_v2_18d_head_v1",
  "action_dim": 18,
  "fps": 30.0,
  "conditioning": {
    "mode": "motion_only",
    "text_conditioning_used": false,
    "emotion_conditioning_used": false,
    "audio_conditioning_used": false
  },
  "native_duration": {
    "policy": "held_out_native_variable_length_no_crop_no_pad",
    "held_out_episode_frame_count": 247,
    "requested_generation_frame_count": 247,
    "cropped": false,
    "padded": false
  },
  "caption": {
    "kind": "motion_only_metadata",
    "semantic_role": "metadata_not_model_condition_not_verified_semantics",
    "text_zh": "测试集样本 heldout_001，原生时长 8.20 秒",
    "text_en": "Held-out sample heldout_001, native duration 8.20 seconds"
  }
}
```

For a reviewed caption, replace `caption` with:

```json
{
  "kind": "reviewed_robot_observable_text",
  "review_artifact": "reviews/generated_caption_review.jsonl",
  "review_artifact_sha256": "<sha256>",
  "review_record_id": "review-heldout-001-seed7",
  "review_record_sha256": "<stable-json-record-sha256>"
}
```

The referenced record has this contract:

```json
{
  "artifact_kind": "robot_observable_generated_motion_caption_review_v1",
  "protocol_version": "independent_blind_generated_motion_caption_v1",
  "review_id": "review-heldout-001-seed7",
  "review_status": "accepted_robot_observable_text",
  "full_decode_to_eof": true,
  "label_metadata_exposed": false,
  "emotion_inference_performed": false,
  "text_conditioning_claimed": false,
  "generated_csv_sha256": "<sha256>",
  "checkpoint_sha256": "<sha256>",
  "decoded_frame_count": 247,
  "fps": 30.0,
  "native_duration_preserved": true,
  "fixed_duration_window_used": false,
  "robot_observable_text": {
    "zh": "机器人抬起左前臂，然后将双臂向外展开。",
    "en": "The robot raises its left forearm, then opens both arms outward."
  }
}
```

`review_record_sha256` is SHA-256 over compact JSON with sorted keys and UTF-8
text (`ensure_ascii=false`, separators `(',', ':')`).

## Commands

Validate all real inputs without writing a deliverable:

```bash
python -m tools.pretrain_evaluation.build_18d_pretrain_long_video \
  --checkpoint /path/to/training/ula_fm_checkpoint.pt \
  --training-summary /path/to/training/training_summary.json \
  --split-manifest /path/to/training/split_manifest.json \
  --generation-manifest /path/to/evaluation/generated_segments.jsonl \
  --validate-only
```

After validation succeeds, render and publish:

```bash
python -m tools.pretrain_evaluation.build_18d_pretrain_long_video \
  --checkpoint /path/to/training/ula_fm_checkpoint.pt \
  --training-summary /path/to/training/training_summary.json \
  --split-manifest /path/to/training/split_manifest.json \
  --generation-manifest /path/to/evaluation/generated_segments.jsonl
```

The output directory must not already exist. This prevents an older or partial
video from being silently mixed with the current checkpoint.
