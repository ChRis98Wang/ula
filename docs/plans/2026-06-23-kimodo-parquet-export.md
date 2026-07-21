# Kimodo Parquet Export Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Kimodo CSV plus prompt-table exporter that creates the existing LeRobot-style parquet dataset used by ULA MMDiT-lite training.

**Architecture:** Add a small Kimodo-specific JSONL builder that scans `Kimodo_CSV/csv/<behavior>/<emotion>/*.csv`, matches each file to `kimodo_action_emotion_prompts.csv`, emits language-action records, then reuses `upper_body_skeleton.lerobot_export.export_lerobot_dataset` for parquet output. Keep matching strict by default so missing prompt/data pairs are visible.

**Tech Stack:** Python stdlib CSV/pathlib/argparse, existing `upper_body_skeleton.kimodo_semantics`, existing `upper_body_skeleton.lerobot_export`, pytest, pyarrow.

---

### Task 1: Kimodo Record Builder

**Files:**
- Create: `upper_body_skeleton/kimodo_parquet_export.py`
- Test: `tests/test_kimodo_parquet_export.py`

**Steps:**
1. Write failing tests for parsing sample filenames and matching prompt records by `output_name` stem.
2. Run `pytest tests/test_kimodo_parquet_export.py -v` and confirm import/function failures.
3. Implement parsing, prompt matching, metadata defaults, JSONL writing, and parquet export CLI.
4. Run the focused tests again.

### Task 2: Real Data Smoke

**Files:**
- Read: `Kimodo_CSV/kimodo_action_emotion_prompts.csv`
- Read: `Kimodo_CSV/csv/**/**/*.csv`

**Steps:**
1. Run a two-episode export to a temporary dataset directory.
2. Load it with `upper_body_skeleton.ula_training.load_lerobot_episodes`.
3. Verify total episodes, prompt text, behavior/emotion IDs, and condition dimensions.

### Task 3: Final Verification

**Commands:**
- `pytest tests/test_kimodo_parquet_export.py tests/test_lerobot_export.py tests/test_ula_training.py -v`
- `python -m upper_body_skeleton.kimodo_parquet_export --kimodo-root Kimodo_CSV --output-dir datasets/kimodo_lerobot_mmdit_lite --max-episodes 8`
