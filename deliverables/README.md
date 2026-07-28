# Upper-Body Language-Action Deliverables

This is the clean entry point for checking the current dataset, model training,
generated motion, and MuJoCo playback.

> Rewritten 2026-07-26. The previous version of this file pointed at
> `01_datasets/`, `02_codebook_v1_train_run/`, `03_weights/`, and
> `04_network_demo.html` under `/Users/demo/Desktop/upper_body_motion_roadmap/...`.
> Those were dead symlinks to a retired macOS machine and none of that content
> exists here — they belonged to the superseded 11/15-joint route (see the
> banner in the root `README.md`). The folders below are what actually exists
> in this directory today.

## Main Folders

- `expressive_human_motion_v2/`
  BEAT2 "expression turn" (single-speaker affect/gesture) review videos and
  blind-review bundles — e.g. `salute_v2_review.mp4`, `hailing_taxi_v2_review.mp4`,
  plus `*.render.json` provenance for each rendered clip.
- `interactive_human_motion_v1/`, `v2/`, `v3/`
  InterAct dyadic-motion review pipeline outputs (catalogs, manifests, blind
  review bundles, batch annotation, reports) — v1/v2/v3 track successive
  expansion rounds of the interaction dataset.
- `gmr_v2_quality/`
  Retargeting quality spot-checks per source (e.g. `boxing_xsens`).
- `interactive_text_tests/`
  Timestamped ad hoc interactive-viewer test sessions (text prompt -> generated
  motion), one folder per run.
- `long_emotion_previews/`
  Long-horizon emotion-conditioned inference previews and smoke tests.
- `parquet_dataset_reference/`
  Example decoded episodes from the packaged Parquet dataset, by episode id
  and label (e.g. `episode_33565_sad_like`).
- `retarget_preview_0514/`
  Camera-angle preview renders from the 2026-05-14 retargeting pass.
- `wave_videos/`, `wave_videos_v2/`
  Side-by-side "network output vs. dataset reference" comparison videos per
  episode/behavior/emotion, plus a contact sheet and `dataset_selection.json`
  documenting which episodes were sampled. v2 supersedes v1.
- `generated_motion_previews/`
  Standalone generated-motion JSON/video previews (e.g. `nervous_explain_mujoco.json`).

## Current Dataset

```text
datasets/kimodo_lerobot_mmdit_lite/
```

LeRobot-style Parquet dataset (`meta/semantic_index.parquet`, 1,620 rows across
37 columns; `data/chunk-000/*.parquet`) plus a body-only semantic JSONL
(`kimodo_language_action_index.jsonl`). 15 V2 upper-body joints, no head.

## Current Checkpoints

See `weights/README.md` at the repo root for the maintained pointer. Two lines
exist:

- **18D BEAT2 formal (current recommended for new work):**
  `training/runs/beat2_18d_from_scratch_formal_v3/training/ula_fm_checkpoint.pt`
  — 60,000 steps, `checkpoint_loss` 49.96, provenance-locked and license-gated
  against the BEAT2 official semantic-event pool v7. No ad hoc single-sample
  generation CLI exists for this line yet; `tools/pretrain_evaluation/build_18d_pretrain_long_video.py`
  only assembles an acceptance video from an already-completed inference run.
- **Kimodo/Qwen 15D (deployed, interactively runnable today):**
  `training/runs/kimodo_mmdit_lite_qwen_compatible_5k_math_sdp/ula_fm_checkpoint.pt`
  with the semantic adapter at
  `training/runs/kimodo_qwen3_semantic_adapter_deploy_v1/semantic_adapter_checkpoint.pt`.
  See `docs/pt_mujoco_infer.md` for exact commands — verified working
  end-to-end (text -> Qwen embedding -> behavior/emotion -> 15-DOF trajectory)
  on 2026-07-26 under the `env_isaaclab` conda environment.

## Generate A New Motion (Kimodo/Qwen line)

```bash
conda run --no-capture-output -n env_isaaclab \
  python -u -m upper_body_skeleton.pt_mujoco_infer \
  --kimodo-qwen \
  --no-viewer \
  --text "紧张地解释，同时双手做克制的上肢手势" \
  --frames 150 \
  --device cuda \
  --semantic-device cuda \
  --semantic-local-files-only
```

Drop `--no-viewer` (and add `--frames` only if you want a fixed duration) to
open the interactive MuJoCo viewer over an X11-forwarded SSH session — see
`docs/pt_mujoco_infer.md`.
