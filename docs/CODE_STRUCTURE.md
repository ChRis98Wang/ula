# ULA Project — Code Structure

ULA (Upper-body Language-Action) is a pipeline that turns human motion capture
into robot upper-body motions:

```text
motion capture (BEAT2 speech-gesture, InterAct dyadic, MotionX, Xsens, ...)
    -> 3D human skeleton
    -> retarget to robot URDF upper-body joints (15D or 18D contract)
    -> QC + human review + provenance lock (hash-bound, license-gated)
    -> Flow Matching transformer (MMDiT) trained on the locked data
    -> conditioned motion generation, MuJoCo preview / ROS2 execution
```

The current generation of the model is the **"V2 18D"** contract (15 upper-body
joints + 3 head joints). Everything under `ula_v2_18d_*` is the active line;
older `ula_v2` (15D, no head) and `ula_training`/`ula_infer` (V1) modules are
still present but superseded.

There is no single up-to-date architecture README — this file is that map.
`README.md` at the repo root is a 2026-05 Chinese-language design roadmap
(useful for *why*, written before the 18D head contract existed).
`PROJECT_STRUCTURE.md` describes an older macOS layout and is stale — ignore
its paths.

## Top-level layout

| Path | What it is |
| --- | --- |
| `upper_body_skeleton/` | The core Python package: retargeting, model, training, inference, eval. |
| `tools/` | CLI scripts: data acquisition, retargeting batch jobs, human review, training entry points, release/provenance builders. |
| `configs/` | JSON/YAML training and binding configs. One file = one reproducible run spec. |
| `tests/` | ~105 pytest files, roughly one per module/tool. |
| `docs/` | Roadmap (`roadmap/`), reference literature (`references/`), HTML architecture demos (`demos/`), design plans (`plans/`). |
| `schemas/` | JSON Schemas for pose/annotation/action records. |
| `training/runs/` | Output directory per training run: checkpoints, `progress.jsonl`, previews, split manifests. |
| `deliverables/` | Clean, reviewed outputs meant for external viewing (demos, packaged datasets, videos). |
| `.venvs/gmr` | The Python venv used for retargeting and the 18D formal training pipeline (`train_ula_v2_18d_formal_from_scratch.py` runs here). |

Large raw datasets live outside the repo, under `/home/gez/nas/cloud/gez/human_motion/`
(`raw/`, `processed/`, `catalog/`, `review/`) — the repo holds code and small
metadata/lock files, and configs point at NAS paths by absolute path + sha256.

## Core package: `upper_body_skeleton/`

Retargeting contracts (human skeleton → robot joints):

| File | Purpose |
| --- | --- |
| `retarget_v2.py` | Base V2 15-joint contract (no head). |
| `retarget_v2_18d.py` | Append-only V2 contract adding 3 head-orientation joints. **Current contract.** |
| `retarget_v2_ik.py` | IK solving for the V2 contracts. |
| `v2_axis_calibration.py` | Axis/sign calibration for retargeting. |

Model, conditioning, and training (the 18D generator):

| File | Purpose |
| --- | --- |
| `ula_v2_18d_head.py` | Versioned 18D head-adapter contract for the MMDiT generator — the schema everything else validates against. |
| `ula_v2_18d_random_init.py` | Strict full-random (no warm-start) initialization contract for the 18D model. |
| `ula_v2_18d_posttrain.py` | Auditable post-training loop (the actual optimizer/training step logic used by the formal-from-scratch run). |
| `ula_v2_conditioning.py` | Leakage-safe conditioning/trajectory preprocessing (text, style, duration channels). |
| `ula_v2_expression_turn_episode.py` | Formal training contract for reviewed BEAT2 "expression turn" (dyadic/affect) episodes — the other formal contract besides motion-only. |
| `ula_training_v2.py` | Leakage-safe, variable-duration training loop for the general Kimodo ULA MMDiT V2 generator. |
| `ula_training.py` / `ula_infer.py` | Older V1 training/inference (legacy). |

Semantic/language conditioning and cross-modal work:

- `semantic_adapter.py`, `kimodo_semantics.py`, `language_action_index.py` — text/behavior label handling.
- `cross_modal_latent.py`, `motion_latent.py` — Qwen-conditioned motion latent alignment (see `docs/qwen_motion_latent_lora.md`).
- `long_emotion_infer.py` — long-horizon emotion-conditioned inference.

Evaluation, playback, data export:

- `pt_v2_evaluate.py` — reproducible held-out evaluation for V2 generators.
- `pt_mujoco_infer.py`, `pt_dataset_mujoco_compare.py`, `mujoco_playback.py`, `preview.py`, `side_by_side_preview.py` — MuJoCo rendering/preview.
- `lerobot_export.py`, `clean_lerobot_dataset.py`, `kimodo_parquet_export.py` — dataset export formats.
- `batch_retarget.py`, `batch_progress.py`, `extract.py`, `smooth.py`, `retarget_monitor.py` — retargeting batch utilities.

## `tools/` — organized by pipeline stage

**Data acquisition** (`tools/human_motion_collection/`): download/inventory raw
sources (BEAT2, InterAct, HAA500), build catalogs, select QC pilots, run the
annotation pipeline. See its `README.md` / `README_INTERACT.md`.

**Retargeting** (`tools/gmr_v2/`): batch-retarget each source (BEAT2, InterAct,
MotionX, Xsens BVH) into the V2/18D joint contract using the GMR/Mink IK
solver. Runs in `.venvs/gmr`. See `tools/gmr_v2/README.md`.

**Human review / QC** (`tools/human_motion_review/`): build blind-review
bundles, render review videos, normalize/merge/finalize reviewer submissions,
adjudicate the training dataset. This is where "candidate" clips become
"train-ready" ones.

**Release & provenance locking** (`tools/*.py`, top level):
- `build_beat2_v7_motion_only_release.py` — turns a QC-passed manifest into an
  immutable `train_ready.jsonl` + release report (hash-pinned).
- `build_expression_turn_v8_provenance_lock.py` — same idea for the
  multi-source expression-turn contract.
- A **provenance lock** is a hash-bound JSON file (`provenance_lock.json` /
  `motion_only_pretrain_provenance_lock.json` on the NAS) that pins exact
  artifact paths + sha256 for every input, plus a `license_gate` recording
  dataset license terms and an explicit human `user_confirmation` (identity +
  timestamp) when terms are ambiguous (e.g. BEAT2's HuggingFace metadata says
  Apache-2.0 but the official project page says non-commercial). Formal
  training refuses to start unless every referenced file's hash still matches
  and the confirmation is present — see `audit_formal_readiness()` in
  `tools/train_ula_v2_18d_formal_from_scratch.py`.

**Training entry points** (`tools/*.py`, top level):
- `init_ula_v2_18d_random.py` — create the untrained full-random checkpoint.
- `train_ula_v2_18d_formal_from_scratch.py` — **the formal, audited pipeline.**
  Staged as `--stage audit|initialize|cache|train`; refuses warm-starts,
  fixed/cropped durations, and unreviewed/unsafe data by construction. This is
  what the current `beat2_18d_from_scratch_formal_v1` run used.
- `train_ula_v2_18d_head.py`, `train_ula_v2_18d_posttrain.py`,
  `train_ula_v2_18d_staged.py` — earlier/alternate training paths (15D→18D
  migration, interaction-domain post-training with Kimodo replay, staged
  resumable pipeline).
- `sweep_ula_v2_18d_head_losses.py` / `evaluate_ula_v2_18d_head_loss_sweep.py`
  — loss-weight search.
- `smoke_ula_v2_native_length_memory.py` — one-microbatch CUDA memory smoke test.

**Post-pretrain acceptance** (`tools/pretrain_evaluation/`):
`build_18d_pretrain_long_video.py` renders the formal acceptance video from a
finished checkpoint's inference output — it never fabricates motion, only
assembles already-generated frames.

## `configs/`

Each file is a complete, reproducible run spec. Naming tells you the lineage:

- `beat2_18d_from_scratch_formal_v1.json` — current formal motion-only run (no warm start, BEAT2 only).
- `beat2_18d_from_15d_staged_v1.json` — staged migration from the 15D contract.
- `beat2_18d_posttrain_full_v1.json` — interaction-domain post-training config.
- `beat2_18d_head_loss_sweep_v1.json` — loss-sweep config.
- `train_kimodo_*.yaml` — the general Kimodo MMDiT/adapter/LoRA training family (older, non-BEAT2-formal line).
- `human_robot_binding.yaml`, `robot_upper_body_0421.yaml`, `emotion_label_schema.yaml` — static binding/schema config, not training runs.

## Tests

`tests/` mirrors the module/tool names 1:1 (`test_ula_v2_18d_head.py`,
`test_formal_from_scratch_provenance.py`, `test_build_beat2_v7_motion_only_release.py`,
etc.). Run the relevant subset with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/gez/shuaiwang/.venvs/gmr/bin/python -m pytest -q tests/
```

## Where to start reading

For the current 18D formal pipeline, read in this order:
1. `upper_body_skeleton/retarget_v2_18d.py` — the joint contract itself.
2. `upper_body_skeleton/ula_v2_18d_head.py` — the model/episode contract.
3. `upper_body_skeleton/ula_v2_18d_random_init.py` — how a checkpoint is born.
4. `tools/train_ula_v2_18d_formal_from_scratch.py` — how config, provenance
   lock, license gate, and the above contracts are all tied together and
   audited before any training step runs.
5. `upper_body_skeleton/ula_v2_18d_posttrain.py` — the actual training loop.
