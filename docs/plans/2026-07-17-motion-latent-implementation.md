# Motion Latent And Prototype Retrieval Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Train, evaluate, and export a compact Kimodo motion latent space and representative prototype motions without changing the current ULA Flow generator.

**Architecture:** Add an independent PyTorch temporal encoder over normalized joint position and velocity. Train it with semantic classification, supervised contrastive, and kinematic descriptor losses; then export held-out diagnostics and train-only class medoids for later Flow conditioning.

**Tech Stack:** Python, PyTorch, NumPy, PyArrow, pytest.

---

### Task 1: Dataset Split And Motion Features

**Files:**
- Create: `upper_body_skeleton/motion_latent.py`
- Create: `tests/test_motion_latent.py`

1. Write failing tests for deterministic per-semantic-group `8/1/1` splitting, disjoint episode IDs, and train-only normalization.
2. Run `python -m pytest tests/test_motion_latent.py -q` and confirm imports fail because the module does not exist.
3. Implement `stratified_episode_split`, `compute_motion_normalization`, and normalized position/velocity feature construction.
4. Re-run the focused tests and confirm they pass.

### Task 2: Metric Encoder And Losses

**Files:**
- Modify: `upper_body_skeleton/motion_latent.py`
- Modify: `tests/test_motion_latent.py`

1. Write failing shape and normalization tests for `MotionMetricEncoder`.
2. Write failing numerical tests for `supervised_contrastive_loss`, including batches with no positive pair.
3. Implement the temporal Conv1d encoder, behavior/emotion/descriptor heads, descriptor extraction, and weighted metric loss.
4. Add paired semantic batch sampling so each anchor has a positive example.
5. Run the focused tests after each red-green cycle.

### Task 3: Diagnostics And Prototype Selection

**Files:**
- Modify: `upper_body_skeleton/motion_latent.py`
- Modify: `tests/test_motion_latent.py`

1. Write failing tests for nearest-neighbor metrics, split-specific collapse/effective-rank reporting, raw-feature comparison, and train-only medoid selection.
2. Implement batched episode encoding, cosine-distance diagnostics, a downsampled position/velocity baseline, and deterministic greedy k-medoids.
3. Verify every requested semantic group receives a prototype and test IDs cannot be selected.
4. Run `python -m pytest tests/test_motion_latent.py -q`.

### Task 4: Training And Artifact CLI

**Files:**
- Create: `training/scripts/train_motion_latent.py`
- Create: `configs/train_kimodo_motion_latent.yaml`
- Modify: `tests/test_motion_latent.py`

1. Write a failing end-to-end test using a tiny in-memory episode fixture and a short CPU training run.
2. Implement deterministic training, train-reference validation retrieval, atomic best-checkpoint export, JSONL progress, `embeddings.npz`, `diagnostics.json`, and `prototypes.json`.
3. Support direct CLI options plus strict checked-in YAML defaults and a streaming Parquet loader without adding dependencies.
4. Run the focused test and inspect every artifact.

### Task 5: Verification

**Files:**
- Modify only files required by failures attributable to this feature.

1. Run `python -m pytest tests/test_motion_latent.py -q`.
2. Run existing ULA regression tests with `python -m pytest tests/test_ula_training.py tests/test_train_config.py -q`.
3. Run a short CPU or CUDA smoke training against `datasets/kimodo_lerobot_mmdit_lite` and inspect diagnostics for finite values and complete prototype coverage.
4. Review the final diff for accidental changes to the user's existing training work.
