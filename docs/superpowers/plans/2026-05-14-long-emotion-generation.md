# Long Emotion Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add long-horizon emotion-aware V2 upper-body generation that can predict segment duration and transition/end intent, then record generated motion with the original V2 MuJoCo model.

**Architecture:** Extend the existing ULA-FM model with a small planner head that predicts `duration_sec` plus transition logits (`continue`, `emotion_change`, `action_change`, `end`) from the text/emotion condition. Add an inference loop that repeatedly samples motion segments, feeds the previous segment end pose as continuity context, stops on the learned end decision or a max duration, and renders the combined CSV with the original V2 URDF. This is forward-compatible with future variable-duration data; current fixed-window data provides weak duration labels from episode length.

**Tech Stack:** Python, PyTorch, NumPy, MuJoCo, pyarrow/LeRobot parquet, pytest.

---

### Files
- Modify: `/Users/demo/Desktop/upper_body_motion_roadmap/upper_body_skeleton/ula_training.py`
- Modify: `/Users/demo/Desktop/upper_body_motion_roadmap/upper_body_skeleton/ula_infer.py`
- Create: `/Users/demo/Desktop/upper_body_motion_roadmap/upper_body_skeleton/long_emotion_infer.py`
- Modify: `/Users/demo/Desktop/upper_body_motion_roadmap/tests/test_ula_training.py`
- Create: `/Users/demo/Desktop/upper_body_motion_roadmap/tests/test_long_emotion_infer.py`

### Tasks
- [ ] Add tests that `UlaFmModel.plan_condition()` returns positive duration and 4 transition logits.
- [ ] Add tests that long-horizon inference generates multiple segments, concatenates CSV rows, writes `plan.json`, and records an MP4 path under `deliverables/long_emotion_previews`.
- [ ] Implement planner head in `UlaFmModel` without breaking existing checkpoints: tolerate missing planner weights by using initialized defaults when loading old checkpoints.
- [ ] Add a planner loss using weak labels: duration from episode length/fps and transition class default `end` for standalone dataset episodes.
- [ ] Add long-emotion CLI for manual use: checkpoint, text, max duration, max segments, output dir, fps, segment frame limits, render width/height.
- [ ] Verify with unit tests and a short smoke generation using the existing checkpoint.
