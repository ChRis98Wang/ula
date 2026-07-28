# Project Structure

> This file previously described a macOS layout
> (`/Users/demo/Desktop/upper_body_motion_roadmap/...`) that never existed on this
> host. It has been rewritten to match the actual repo at
> `/home/gez/shuaiwang/ula-motion-generate`. For the full pipeline map (modules,
> tools, current model line) read **`docs/CODE_STRUCTURE.md`** — this file only
> covers where things live on disk.

## Top-Level Folders

- `datasets/`
  Model-ready dataset entry points. Currently `datasets/kimodo_lerobot_mmdit_lite/`
  (LeRobot-style Parquet + semantic index, 15-joint Kimodo/Qwen line).
- `tools/`
  Data-making scripts, organized by pipeline stage: `human_motion_collection/`
  (acquisition), `gmr_v2/` (retargeting), `human_motion_review/` (QC/adjudication),
  `pretrain_evaluation/` (formal acceptance video), plus top-level training and
  release/provenance-lock scripts. Large raw sources live outside the repo on the
  NAS mount at `/home/gez/nas/cloud/gez/human_motion/` (`raw/`, `processed/`,
  `catalog/`, `review/`); configs and provenance locks reference that NAS by
  absolute path + sha256, not by a repo-relative symlink.
- `training/`
  Training scripts (`training/scripts/`, plus top-level `tools/train_*.py`
  entry points) and training run folders (`training/runs/<run_name>/`, each with
  checkpoints, `progress.jsonl`, and previews).
- `weights/`
  Local, gitignored symlink index into the current best checkpoints under
  `training/runs/`. See `weights/README.md` for what each link points at and why.
  Nothing is duplicated here — delete and recreate freely.
- `deliverables/`
  Clean inspection entry point for demos, generated videos, and dataset/quality
  references. See `deliverables/README.md`.
- `upper_body_skeleton/`
  Core Python package: retargeting contracts, model, training, inference, eval.
- `tests/`
  ~105+ pytest files, one per module/tool. Run with
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n env_isaaclab python -m pytest -q`
  (or the `.venvs/gmr` interpreter used for retargeting/formal-training scripts —
  see `docs/CODE_STRUCTURE.md`). Headless MuJoCo rendering tests need
  `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`.
- `configs/`
  Robot/binding/label configs and one JSON/YAML per reproducible training run.
- `docs/`
  Roadmap docs (`roadmap/`, mostly the 2026-05 pre-18D design rationale),
  references, HTML architecture demos (`demos/`), and `CODE_STRUCTURE.md` (the
  current, accurate pipeline map — read that first).
- `schemas/`
  JSON schemas for data records.

## Current Model Line

The active contract is **V2 18D** (15 upper-body joints + 3 head joints),
implemented in `upper_body_skeleton/retarget_v2_18d.py`. The formal, audited
training pipeline is `tools/train_ula_v2_18d_formal_from_scratch.py`; its most
recent complete candidate is `training/runs/beat2_18d_from_scratch_formal_v3/`
(60,000 steps, `checkpoint_loss` 49.96, BEAT2-only, provenance-locked). See
`weights/README.md` for the current pointer and `docs/CODE_STRUCTURE.md` for
how config, provenance lock, license gate, and training loop tie together.

A separate, already-deployed line is the Kimodo/Qwen generator (15 joints, no
head): `training/runs/kimodo_mmdit_lite_qwen_compatible_5k_math_sdp/` with the
semantic adapter at `training/runs/kimodo_qwen3_semantic_adapter_deploy_v1/`.
This is what `docs/pt_mujoco_infer.md`'s `--kimodo-qwen` shortcut runs — verified
working end-to-end (text -> Qwen embedding -> behavior/emotion -> MuJoCo-ready
trajectory) as of 2026-07-26.

## Superseded

The README.md roadmap's own banner line says the 11/15-joint interface it
describes in most of the document is superseded. `codebook_v1`,
`baseline_final_auto`, and any `weights/*_checkpoint.pt` path under the old
macOS root belong to that superseded route and no longer exist on this host —
do not try to recreate them.
