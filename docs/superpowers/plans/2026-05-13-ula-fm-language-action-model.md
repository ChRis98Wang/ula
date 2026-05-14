# ULA-FM Language Action Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a no-vision-head Conditional Flow Matching model that maps language, affect/style labels, and current V2 robot upper-body state to safe 15-joint action chunks.

**Architecture:** Freeze the first text encoder in v0.1 and train only the language adapter, structured condition encoders, robot-state encoder, condition fusion, FiLM/AdaLN modulation, and action expert. The action expert learns a conditional flow field over normalized `[120, 15]` V2 upper-body joint chunks, then a safety/export layer produces CSV and MuJoCo previews.

**Tech Stack:** Python 3.13, PyTorch 2.9, NumPy, MuJoCo 3.7, pytest. `transformers` and `sentence_transformers` are not currently installed, so v0.1 must include a deterministic local text encoder fallback and keep the text encoder interface swappable.

---

## Current Context

The existing project already has:

- Contact-safe IK retargeting output under `/Users/demo/Desktop/upper_body_motion_roadmap/video/seamless_interaction_50g/batch_v2_retarget_ik_contact_safe/full_single_progress`.
- A language-action index builder at `/Users/demo/Desktop/upper_body_motion_roadmap/upper_body_skeleton/language_action_index.py`.
- Current language-action schema at `/Users/demo/Desktop/upper_body_motion_roadmap/schemas/language_action_sample.schema.json`.
- Old flow sample schema at `/Users/demo/Desktop/upper_body_motion_roadmap/schemas/flow_matching_sample.schema.json`, which still says 11 joints and must be upgraded to the current 15 V2 joints.
- PyTorch and MuJoCo are installed in `/Users/demo/Desktop/mjlab/.venv/bin/python`; `transformers` and `sentence_transformers` are missing.

The implementation must not depend on downloading a language model for the first runnable baseline.

## Trainable vs Frozen Parameters

v0.1 training policy:

| Module | Status | Reason |
| --- | --- | --- |
| Text encoder | Frozen | Dataset is small; avoid damaging semantic space. Use deterministic fallback if no sentence model is installed. |
| Language adapter | Trainable | Maps text embedding into robot action condition space. |
| Intent/affect/style embeddings | Trainable | Learns discrete control semantics. |
| Numeric condition MLP | Trainable | Encodes arousal, valence, motion energy, duration, confidence. |
| Robot state MLP | Trainable | Encodes q0, dq0, joint margin for continuity. |
| Condition fusion MLP | Trainable | Produces `cond_emb` for action expert. |
| FiLM/AdaLN modulation | Trainable | Injects language and state into every Transformer layer. |
| Action expert Transformer | Trainable | Learns conditional flow field over action chunks. |
| Safety decoder | Not learned in v0.1 | Deterministic clamp/filter/export layer. |
| Text encoder LoRA | Off in v0.1, optional v0.2 | Only after enough reviewed text-action labels exist. |

---

## File Structure

Create a new focused package:

- Create `/Users/demo/Desktop/upper_body_motion_roadmap/upper_body_lam/__init__.py`
  - Package exports and version.
- Create `/Users/demo/Desktop/upper_body_motion_roadmap/upper_body_lam/constants.py`
  - 15-joint V2 order, chunk sizes, default fps, default hidden sizes.
- Create `/Users/demo/Desktop/upper_body_motion_roadmap/upper_body_lam/text_encoder.py`
  - `FrozenTextEncoder` protocol and `HashingTextEncoder` fallback.
- Create `/Users/demo/Desktop/upper_body_motion_roadmap/upper_body_lam/dataset.py`
  - JSONL + CSV window loader, normalization, masks, condition parsing.
- Create `/Users/demo/Desktop/upper_body_motion_roadmap/upper_body_lam/model.py`
  - Condition encoders, language adapter, FiLM/AdaLN Transformer blocks, action expert.
- Create `/Users/demo/Desktop/upper_body_motion_roadmap/upper_body_lam/flow.py`
  - Flow matching corruption, target velocity, sampling integrators.
- Create `/Users/demo/Desktop/upper_body_motion_roadmap/upper_body_lam/losses.py`
  - Flow, start-pose, smoothness, joint-limit margin losses.
- Create `/Users/demo/Desktop/upper_body_motion_roadmap/upper_body_lam/train.py`
  - CLI train loop, checkpoint, metrics JSONL.
- Create `/Users/demo/Desktop/upper_body_motion_roadmap/upper_body_lam/sample.py`
  - CLI prompt-to-action sampling.
- Create `/Users/demo/Desktop/upper_body_motion_roadmap/upper_body_lam/export.py`
  - Denormalize and export generated chunks to CSV.
- Create `/Users/demo/Desktop/upper_body_motion_roadmap/configs/ula_fm_v0.yaml`
  - Model and training defaults.
- Modify `/Users/demo/Desktop/upper_body_motion_roadmap/schemas/flow_matching_sample.schema.json`
  - Upgrade from 11 joints to 15 V2 joints.

Tests:

- Create `/Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_text_encoder.py`
- Create `/Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_dataset.py`
- Create `/Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_model.py`
- Create `/Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_flow.py`
- Create `/Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_export.py`

---

## Task 1: Constants and 15-Joint Schema

**Files:**
- Create: `/Users/demo/Desktop/upper_body_motion_roadmap/upper_body_lam/__init__.py`
- Create: `/Users/demo/Desktop/upper_body_motion_roadmap/upper_body_lam/constants.py`
- Modify: `/Users/demo/Desktop/upper_body_motion_roadmap/schemas/flow_matching_sample.schema.json`
- Test: `/Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_constants.py`

- [ ] **Step 1: Write failing tests for constants**

Create `/Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_constants.py`:

```python
from upper_body_lam.constants import ACTION_DIM, DEFAULT_FPS, DEFAULT_HORIZON, V2_JOINT_ORDER


def test_v2_joint_order_is_15_dimensional():
    assert ACTION_DIM == 15
    assert len(V2_JOINT_ORDER) == 15
    assert V2_JOINT_ORDER[:3] == [
        "joint_pelvisYaw",
        "joint_pelvisPitch",
        "joint_pelvisRoll",
    ]
    assert "joint_lElbow" in V2_JOINT_ORDER
    assert "joint_rElbow" in V2_JOINT_ORDER


def test_default_action_chunk_is_two_seconds_at_60hz():
    assert DEFAULT_FPS == 60
    assert DEFAULT_HORIZON == 120
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=/Users/demo/Desktop/upper_body_motion_roadmap /Users/demo/Desktop/mjlab/.venv/bin/python -m pytest /Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_constants.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'upper_body_lam'`.

- [ ] **Step 3: Add package and constants**

Create `/Users/demo/Desktop/upper_body_motion_roadmap/upper_body_lam/__init__.py`:

```python
"""Language-action model package for V2 upper-body motion generation."""

__version__ = "0.1.0"
```

Create `/Users/demo/Desktop/upper_body_motion_roadmap/upper_body_lam/constants.py`:

```python
V2_JOINT_ORDER = [
    "joint_pelvisYaw",
    "joint_pelvisPitch",
    "joint_pelvisRoll",
    "joint_lShoulderPitch",
    "joint_lShoulderRoll",
    "joint_lShoulderYaw",
    "joint_lElbow",
    "joint_lWristRoll",
    "joint_lWristPitch",
    "joint_rShoulderPitch",
    "joint_rShoulderRoll",
    "joint_rShoulderYaw",
    "joint_rElbow",
    "joint_rWristRoll",
    "joint_rWristPitch",
]

ACTION_DIM = len(V2_JOINT_ORDER)
DEFAULT_FPS = 60
DEFAULT_HORIZON = 120
DEFAULT_STRIDE = 30
TEXT_EMBED_DIM = 384
MODEL_DIM = 384

JOINT_LIMIT_LOW = [-1.5] * ACTION_DIM
JOINT_LIMIT_HIGH = [1.5] * ACTION_DIM
```

- [ ] **Step 4: Upgrade flow schema to 15 joints**

Replace `/Users/demo/Desktop/upper_body_motion_roadmap/schemas/flow_matching_sample.schema.json` with:

```json
{
  "schema_name": "flow_matching_motion_sample",
  "version": "0.2",
  "required_top_level_fields": [
    "motion_id",
    "fps",
    "joint_order",
    "q",
    "dq",
    "condition"
  ],
  "joint_count": 15,
  "fps": 60,
  "horizon_frames": 120,
  "joint_order": [
    "joint_pelvisYaw",
    "joint_pelvisPitch",
    "joint_pelvisRoll",
    "joint_lShoulderPitch",
    "joint_lShoulderRoll",
    "joint_lShoulderYaw",
    "joint_lElbow",
    "joint_lWristRoll",
    "joint_lWristPitch",
    "joint_rShoulderPitch",
    "joint_rShoulderRoll",
    "joint_rShoulderYaw",
    "joint_rElbow",
    "joint_rWristRoll",
    "joint_rWristPitch"
  ],
  "condition_required_fields": [
    "scenario_description",
    "action_description",
    "intent_text",
    "mood_text",
    "intent",
    "observed_affect",
    "motion_style",
    "arousal",
    "valence",
    "motion_energy",
    "human_state_confidence",
    "robot_energy_level",
    "robot_safety_mode",
    "target_direction",
    "duration_sec"
  ],
  "notes": [
    "Generated trajectories must pass safety filtering before simulation or robot playback.",
    "Emotion fields are observable affect estimates, not true internal state.",
    "The v0.1 model freezes the text encoder and trains the adapter, condition fusion, and action expert."
  ]
}
```

- [ ] **Step 5: Run constants test**

Run:

```bash
PYTHONPATH=/Users/demo/Desktop/upper_body_motion_roadmap /Users/demo/Desktop/mjlab/.venv/bin/python -m pytest /Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_constants.py -v
```

Expected: PASS.

---

## Task 2: Frozen Text Encoder Interface and Hashing Fallback

**Files:**
- Create: `/Users/demo/Desktop/upper_body_motion_roadmap/upper_body_lam/text_encoder.py`
- Test: `/Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_text_encoder.py`

- [ ] **Step 1: Write failing text encoder tests**

Create `/Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_text_encoder.py`:

```python
import torch

from upper_body_lam.text_encoder import HashingTextEncoder


def test_hashing_text_encoder_is_deterministic_and_frozen():
    encoder = HashingTextEncoder(embed_dim=384)
    first = encoder.encode(["friendly explaining gesture"])
    second = encoder.encode(["friendly explaining gesture"])
    assert first.shape == (1, 384)
    assert torch.allclose(first, second)
    assert first.requires_grad is False


def test_hashing_text_encoder_separates_different_text():
    encoder = HashingTextEncoder(embed_dim=384)
    first = encoder.encode(["friendly explaining gesture"])
    second = encoder.encode(["restrained refusal gesture"])
    assert not torch.allclose(first, second)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=/Users/demo/Desktop/upper_body_motion_roadmap /Users/demo/Desktop/mjlab/.venv/bin/python -m pytest /Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_text_encoder.py -v
```

Expected: FAIL with missing `upper_body_lam.text_encoder`.

- [ ] **Step 3: Implement deterministic frozen fallback**

Create `/Users/demo/Desktop/upper_body_motion_roadmap/upper_body_lam/text_encoder.py`:

```python
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import torch


TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


class FrozenTextEncoder:
    embed_dim: int

    def encode(self, texts: list[str]) -> torch.Tensor:
        raise NotImplementedError


@dataclass
class HashingTextEncoder(FrozenTextEncoder):
    embed_dim: int = 384

    def encode(self, texts: list[str]) -> torch.Tensor:
        rows = []
        for text in texts:
            vec = torch.zeros(self.embed_dim, dtype=torch.float32)
            tokens = TOKEN_RE.findall(text.lower())
            if not tokens:
                tokens = ["<empty>"]
            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % self.embed_dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[index] += sign
            vec = torch.nn.functional.normalize(vec, dim=0)
            rows.append(vec)
        return torch.stack(rows, dim=0).detach()
```

- [ ] **Step 4: Run text encoder tests**

Run:

```bash
PYTHONPATH=/Users/demo/Desktop/upper_body_motion_roadmap /Users/demo/Desktop/mjlab/.venv/bin/python -m pytest /Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_text_encoder.py -v
```

Expected: PASS.

---

## Task 3: Dataset Loader and Normalization

**Files:**
- Create: `/Users/demo/Desktop/upper_body_motion_roadmap/upper_body_lam/dataset.py`
- Test: `/Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_dataset.py`

- [ ] **Step 1: Write failing dataset test with tiny fixture**

Create `/Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_dataset.py`:

```python
import csv
import json

import torch

from upper_body_lam.constants import V2_JOINT_ORDER
from upper_body_lam.dataset import LanguageActionDataset
from upper_body_lam.text_encoder import HashingTextEncoder


def write_csv(path, rows=140):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["time_sec", *V2_JOINT_ORDER])
        writer.writeheader()
        for i in range(rows):
            row = {"time_sec": i / 60.0}
            for j, name in enumerate(V2_JOINT_ORDER):
                row[name] = 0.01 * j + 0.001 * i
            writer.writerow(row)


def test_language_action_dataset_loads_condition_and_chunk(tmp_path):
    csv_path = tmp_path / "motion.csv"
    write_csv(csv_path)
    jsonl_path = tmp_path / "index.jsonl"
    record = {
        "sample_id": "sample_001",
        "language_condition": {
            "raw_transcript": "hello",
            "scenario_description": "a person explains calmly",
            "action_description": "hands move near the chest",
            "intent_text": "explain",
            "mood_text": "neutral",
            "instruction_variants": ["explain with restrained upper body motion"]
        },
        "labels": {
            "intent": "explaining",
            "observed_affect": "neutral",
            "motion_style": "restrained",
            "arousal": 0.3,
            "valence": 0.1,
            "motion_energy": 0.2
        },
        "action": {
            "retarget_csv_path": str(csv_path),
            "start_row": 0,
            "end_row": 120,
            "fps": 60,
            "joint_order": V2_JOINT_ORDER
        },
        "quality": {"accepted_for_training": True}
    }
    jsonl_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    dataset = LanguageActionDataset(jsonl_path, text_encoder=HashingTextEncoder())
    item = dataset[0]
    assert item["action"].shape == (120, 15)
    assert item["text_emb"].shape == (384,)
    assert item["robot_state"].shape == (30,)
    assert item["action"].dtype == torch.float32
    assert item["sample_id"] == "sample_001"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=/Users/demo/Desktop/upper_body_motion_roadmap /Users/demo/Desktop/mjlab/.venv/bin/python -m pytest /Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_dataset.py -v
```

Expected: FAIL with missing `upper_body_lam.dataset`.

- [ ] **Step 3: Implement dataset loader**

Create `/Users/demo/Desktop/upper_body_motion_roadmap/upper_body_lam/dataset.py`:

```python
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from upper_body_lam.constants import ACTION_DIM, DEFAULT_HORIZON, V2_JOINT_ORDER
from upper_body_lam.text_encoder import FrozenTextEncoder, HashingTextEncoder


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def condition_text(record: dict[str, Any]) -> str:
    condition = record.get("language_condition", {})
    fields = [
        condition.get("scenario_description", ""),
        condition.get("action_description", ""),
        condition.get("intent_text", ""),
        condition.get("mood_text", ""),
        " ".join(condition.get("instruction_variants", [])),
    ]
    return " | ".join(str(value).strip() for value in fields if str(value).strip())


def load_action_window(csv_path: str | Path, start_row: int, horizon: int, joint_order: list[str]) -> torch.Tensor:
    rows = []
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        for index, row in enumerate(csv.DictReader(f)):
            if index < start_row:
                continue
            if len(rows) >= horizon:
                break
            rows.append([float(row[name]) for name in joint_order])
    if not rows:
        raise ValueError(f"No action rows found in {csv_path} from row {start_row}")
    while len(rows) < horizon:
        rows.append(rows[-1])
    return torch.tensor(rows, dtype=torch.float32)


def normalize_action(action: torch.Tensor) -> torch.Tensor:
    return torch.clamp(action / 1.5, -1.0, 1.0)


class LanguageActionDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        text_encoder: FrozenTextEncoder | None = None,
        horizon: int = DEFAULT_HORIZON,
    ) -> None:
        self.records = [
            record
            for record in read_jsonl(jsonl_path)
            if record.get("quality", {}).get("accepted_for_training", True)
        ]
        self.text_encoder = text_encoder or HashingTextEncoder()
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        action_meta = record["action"]
        joint_order = action_meta.get("joint_order", V2_JOINT_ORDER)
        if joint_order != V2_JOINT_ORDER:
            raise ValueError("LanguageActionDataset requires current 15-joint V2 order")
        action = load_action_window(
            action_meta["retarget_csv_path"],
            int(action_meta["start_row"]),
            self.horizon,
            joint_order,
        )
        action = normalize_action(action)
        text_emb = self.text_encoder.encode([condition_text(record)])[0]
        q0 = action[0]
        dq0 = torch.zeros(ACTION_DIM, dtype=torch.float32)
        robot_state = torch.cat([q0, dq0], dim=0)
        labels = record.get("labels", {})
        numeric = torch.tensor(
            [
                float(labels.get("arousal") or 0.0),
                float(labels.get("valence") or 0.0),
                float(labels.get("motion_energy") or 0.0),
                float(record.get("time_window", {}).get("end_sec", 2.0) - record.get("time_window", {}).get("start_sec", 0.0)),
            ],
            dtype=torch.float32,
        )
        return {
            "sample_id": record["sample_id"],
            "action": action,
            "text_emb": text_emb,
            "robot_state": robot_state,
            "numeric": numeric,
            "labels": labels,
        }
```

- [ ] **Step 4: Run dataset test**

Run:

```bash
PYTHONPATH=/Users/demo/Desktop/upper_body_motion_roadmap /Users/demo/Desktop/mjlab/.venv/bin/python -m pytest /Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_dataset.py -v
```

Expected: PASS.

---

## Task 4: Flow Matching Utilities

**Files:**
- Create: `/Users/demo/Desktop/upper_body_motion_roadmap/upper_body_lam/flow.py`
- Test: `/Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_flow.py`

- [ ] **Step 1: Write failing tests for corruption and sampling shapes**

Create `/Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_flow.py`:

```python
import torch

from upper_body_lam.flow import flow_match_batch, euler_sample


def test_flow_match_batch_shapes():
    x_data = torch.zeros(4, 120, 15)
    x_t, t, target_v = flow_match_batch(x_data)
    assert x_t.shape == x_data.shape
    assert t.shape == (4,)
    assert target_v.shape == x_data.shape


def test_euler_sample_keeps_shape():
    class ZeroModel(torch.nn.Module):
        def forward(self, x_t, t, condition):
            return torch.zeros_like(x_t)

    noise = torch.randn(2, 120, 15)
    out = euler_sample(ZeroModel(), noise, condition={"dummy": torch.zeros(2, 1)}, steps=4)
    assert out.shape == noise.shape
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=/Users/demo/Desktop/upper_body_motion_roadmap /Users/demo/Desktop/mjlab/.venv/bin/python -m pytest /Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_flow.py -v
```

Expected: FAIL with missing `upper_body_lam.flow`.

- [ ] **Step 3: Implement flow utilities**

Create `/Users/demo/Desktop/upper_body_motion_roadmap/upper_body_lam/flow.py`:

```python
from __future__ import annotations

import torch


def flow_match_batch(x_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = x_data.shape[0]
    noise = torch.randn_like(x_data)
    t = torch.rand(batch, device=x_data.device, dtype=x_data.dtype)
    view_shape = (batch,) + (1,) * (x_data.ndim - 1)
    t_view = t.view(view_shape)
    x_t = (1.0 - t_view) * noise + t_view * x_data
    target_v = x_data - noise
    return x_t, t, target_v


@torch.no_grad()
def euler_sample(model: torch.nn.Module, noise: torch.Tensor, condition: dict, steps: int = 12) -> torch.Tensor:
    x = noise
    dt = 1.0 / float(steps)
    for step in range(steps):
        t_value = torch.full((x.shape[0],), step / float(steps), device=x.device, dtype=x.dtype)
        velocity = model(x, t_value, condition)
        x = x + dt * velocity
    return x
```

- [ ] **Step 4: Run flow tests**

Run:

```bash
PYTHONPATH=/Users/demo/Desktop/upper_body_motion_roadmap /Users/demo/Desktop/mjlab/.venv/bin/python -m pytest /Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_flow.py -v
```

Expected: PASS.

---

## Task 5: Condition Encoder and Action Expert Model

**Files:**
- Create: `/Users/demo/Desktop/upper_body_motion_roadmap/upper_body_lam/model.py`
- Test: `/Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_model.py`

- [ ] **Step 1: Write failing model tests**

Create `/Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_model.py`:

```python
import torch

from upper_body_lam.model import UlaFmModel


def test_ula_fm_model_forward_shape_and_trainable_parts():
    model = UlaFmModel(model_dim=128, layers=2, heads=4)
    x_t = torch.randn(3, 120, 15)
    t = torch.rand(3)
    condition = {
        "text_emb": torch.randn(3, 384),
        "robot_state": torch.randn(3, 30),
        "numeric": torch.randn(3, 4),
    }
    out = model(x_t, t, condition)
    assert out.shape == x_t.shape

    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    assert any(name.startswith("language_adapter") for name in trainable_names)
    assert any(name.startswith("condition_fusion") for name in trainable_names)
    assert any(name.startswith("blocks") for name in trainable_names)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=/Users/demo/Desktop/upper_body_motion_roadmap /Users/demo/Desktop/mjlab/.venv/bin/python -m pytest /Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_model.py -v
```

Expected: FAIL with missing `upper_body_lam.model`.

- [ ] **Step 3: Implement compact Transformer action expert**

Create `/Users/demo/Desktop/upper_body_motion_roadmap/upper_body_lam/model.py`:

```python
from __future__ import annotations

import math

import torch
from torch import nn

from upper_body_lam.constants import ACTION_DIM, MODEL_DIM, TEXT_EMBED_DIM


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=t.device, dtype=t.dtype) * (-math.log(10000.0) / max(half - 1, 1))
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = torch.nn.functional.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class AdaLayerNormBlock(nn.Module):
    def __init__(self, model_dim: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.attn = nn.MultiheadAttention(model_dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(model_dim)
        self.ff = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Linear(model_dim * 4, model_dim),
        )
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(model_dim, model_dim * 4))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale1, shift1, scale2, shift2 = self.mod(cond).chunk(4, dim=-1)
        h = self.norm1(x) * (1 + scale1[:, None, :]) + shift1[:, None, :]
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        h = self.norm2(x) * (1 + scale2[:, None, :]) + shift2[:, None, :]
        x = x + self.ff(h)
        return x


class UlaFmModel(nn.Module):
    def __init__(
        self,
        model_dim: int = MODEL_DIM,
        layers: int = 8,
        heads: int = 6,
        text_dim: int = TEXT_EMBED_DIM,
        numeric_dim: int = 4,
        robot_state_dim: int = ACTION_DIM * 2,
    ) -> None:
        super().__init__()
        self.language_adapter = nn.Sequential(
            nn.Linear(text_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.numeric_encoder = nn.Sequential(
            nn.Linear(numeric_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.robot_state_encoder = nn.Sequential(
            nn.Linear(robot_state_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.condition_fusion = nn.Sequential(
            nn.Linear(model_dim * 4, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )
        self.input_proj = nn.Linear(ACTION_DIM, model_dim)
        self.blocks = nn.ModuleList([AdaLayerNormBlock(model_dim, heads) for _ in range(layers)])
        self.output_proj = nn.Linear(model_dim, ACTION_DIM)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, condition: dict[str, torch.Tensor]) -> torch.Tensor:
        text = self.language_adapter(condition["text_emb"].to(x_t.device, x_t.dtype))
        numeric = self.numeric_encoder(condition["numeric"].to(x_t.device, x_t.dtype))
        robot = self.robot_state_encoder(condition["robot_state"].to(x_t.device, x_t.dtype))
        time = self.time_encoder(sinusoidal_embedding(t.to(x_t.device, x_t.dtype), text.shape[-1]))
        cond = self.condition_fusion(torch.cat([text, numeric, robot, time], dim=-1))
        x = self.input_proj(x_t)
        for block in self.blocks:
            x = block(x, cond)
        return self.output_proj(x)
```

- [ ] **Step 4: Run model tests**

Run:

```bash
PYTHONPATH=/Users/demo/Desktop/upper_body_motion_roadmap /Users/demo/Desktop/mjlab/.venv/bin/python -m pytest /Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_model.py -v
```

Expected: PASS.

---

## Task 6: Losses and One-Step Training Smoke Test

**Files:**
- Create: `/Users/demo/Desktop/upper_body_motion_roadmap/upper_body_lam/losses.py`
- Test: `/Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_training_step.py`

- [ ] **Step 1: Write failing smoke test**

Create `/Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_training_step.py`:

```python
import torch

from upper_body_lam.flow import flow_match_batch
from upper_body_lam.losses import total_training_loss
from upper_body_lam.model import UlaFmModel


def test_one_training_step_updates_trainable_parameters():
    model = UlaFmModel(model_dim=64, layers=1, heads=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x_data = torch.randn(2, 120, 15).clamp(-1, 1)
    condition = {
        "text_emb": torch.randn(2, 384),
        "robot_state": torch.randn(2, 30),
        "numeric": torch.randn(2, 4),
    }
    x_t, t, target_v = flow_match_batch(x_data)
    pred_v = model(x_t, t, condition)
    loss, parts = total_training_loss(pred_v, target_v, x_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert parts["flow"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=/Users/demo/Desktop/upper_body_motion_roadmap /Users/demo/Desktop/mjlab/.venv/bin/python -m pytest /Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_training_step.py -v
```

Expected: FAIL with missing `upper_body_lam.losses`.

- [ ] **Step 3: Implement losses**

Create `/Users/demo/Desktop/upper_body_motion_roadmap/upper_body_lam/losses.py`:

```python
from __future__ import annotations

import torch
import torch.nn.functional as F


def velocity_smoothness(action: torch.Tensor) -> torch.Tensor:
    if action.shape[1] < 2:
        return action.new_tensor(0.0)
    return torch.mean((action[:, 1:] - action[:, :-1]) ** 2)


def acceleration_smoothness(action: torch.Tensor) -> torch.Tensor:
    if action.shape[1] < 3:
        return action.new_tensor(0.0)
    vel = action[:, 1:] - action[:, :-1]
    return torch.mean((vel[:, 1:] - vel[:, :-1]) ** 2)


def joint_limit_margin(action: torch.Tensor, margin: float = 0.95) -> torch.Tensor:
    over = torch.relu(torch.abs(action) - margin)
    return torch.mean(over ** 2)


def total_training_loss(
    pred_v: torch.Tensor,
    target_v: torch.Tensor,
    x_data: torch.Tensor,
    flow_weight: float = 1.0,
    velocity_weight: float = 0.1,
    accel_weight: float = 0.05,
    limit_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    flow = F.mse_loss(pred_v, target_v)
    velocity = velocity_smoothness(x_data)
    accel = acceleration_smoothness(x_data)
    limit = joint_limit_margin(x_data)
    total = flow_weight * flow + velocity_weight * velocity + accel_weight * accel + limit_weight * limit
    return total, {
        "flow": float(flow.detach().cpu()),
        "velocity": float(velocity.detach().cpu()),
        "acceleration": float(accel.detach().cpu()),
        "limit": float(limit.detach().cpu()),
    }
```

- [ ] **Step 4: Run smoke test**

Run:

```bash
PYTHONPATH=/Users/demo/Desktop/upper_body_motion_roadmap /Users/demo/Desktop/mjlab/.venv/bin/python -m pytest /Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_training_step.py -v
```

Expected: PASS.

---

## Task 7: CSV Export and Prompt Sampling Skeleton

**Files:**
- Create: `/Users/demo/Desktop/upper_body_motion_roadmap/upper_body_lam/export.py`
- Create: `/Users/demo/Desktop/upper_body_motion_roadmap/upper_body_lam/sample.py`
- Test: `/Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_export.py`

- [ ] **Step 1: Write failing export test**

Create `/Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_export.py`:

```python
import csv

import torch

from upper_body_lam.constants import V2_JOINT_ORDER
from upper_body_lam.export import denormalize_action, write_action_csv


def test_write_action_csv_exports_15_joint_order(tmp_path):
    action = torch.zeros(120, 15)
    csv_path = tmp_path / "generated.csv"
    write_action_csv(csv_path, action, fps=60)
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 120
    assert list(rows[0].keys()) == ["time_sec", *V2_JOINT_ORDER]


def test_denormalize_action_scales_back_to_joint_units():
    action = torch.ones(2, 15)
    denorm = denormalize_action(action)
    assert torch.allclose(denorm, torch.full((2, 15), 1.5))
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=/Users/demo/Desktop/upper_body_motion_roadmap /Users/demo/Desktop/mjlab/.venv/bin/python -m pytest /Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_export.py -v
```

Expected: FAIL with missing `upper_body_lam.export`.

- [ ] **Step 3: Implement CSV export**

Create `/Users/demo/Desktop/upper_body_motion_roadmap/upper_body_lam/export.py`:

```python
from __future__ import annotations

import csv
from pathlib import Path

import torch

from upper_body_lam.constants import V2_JOINT_ORDER


def denormalize_action(action: torch.Tensor) -> torch.Tensor:
    return torch.clamp(action, -1.0, 1.0) * 1.5


def write_action_csv(path: str | Path, action: torch.Tensor, fps: float = 60.0) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    denorm = denormalize_action(action.detach().cpu()).numpy()
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["time_sec", *V2_JOINT_ORDER])
        writer.writeheader()
        for index, values in enumerate(denorm):
            row = {"time_sec": f"{index / fps:.6f}"}
            row.update({name: f"{float(value):.8f}" for name, value in zip(V2_JOINT_ORDER, values)})
            writer.writerow(row)
```

- [ ] **Step 4: Create sampling CLI skeleton**

Create `/Users/demo/Desktop/upper_body_motion_roadmap/upper_body_lam/sample.py`:

```python
from __future__ import annotations

import argparse

import torch

from upper_body_lam.constants import ACTION_DIM, DEFAULT_HORIZON
from upper_body_lam.export import write_action_csv
from upper_body_lam.flow import euler_sample
from upper_body_lam.model import UlaFmModel
from upper_body_lam.text_encoder import HashingTextEncoder


def sample_to_csv(prompt: str, output_csv: str, steps: int = 12) -> None:
    model = UlaFmModel()
    model.eval()
    text_encoder = HashingTextEncoder()
    noise = torch.randn(1, DEFAULT_HORIZON, ACTION_DIM)
    condition = {
        "text_emb": text_encoder.encode([prompt]),
        "robot_state": torch.zeros(1, ACTION_DIM * 2),
        "numeric": torch.zeros(1, 4),
    }
    action = euler_sample(model, noise, condition, steps=steps)[0]
    write_action_csv(output_csv, action)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample ULA-FM action chunk to CSV")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()
    sample_to_csv(args.prompt, args.output_csv, steps=args.steps)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run export tests**

Run:

```bash
PYTHONPATH=/Users/demo/Desktop/upper_body_motion_roadmap /Users/demo/Desktop/mjlab/.venv/bin/python -m pytest /Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_export.py -v
```

Expected: PASS.

---

## Task 8: Minimal Train CLI

**Files:**
- Create: `/Users/demo/Desktop/upper_body_motion_roadmap/upper_body_lam/train.py`
- Create: `/Users/demo/Desktop/upper_body_motion_roadmap/configs/ula_fm_v0.yaml`

- [ ] **Step 1: Implement CLI train loop with JSONL metrics**

Create `/Users/demo/Desktop/upper_body_motion_roadmap/upper_body_lam/train.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from upper_body_lam.dataset import LanguageActionDataset
from upper_body_lam.flow import flow_match_batch
from upper_body_lam.losses import total_training_loss
from upper_body_lam.model import UlaFmModel
from upper_body_lam.text_encoder import HashingTextEncoder


def collate(batch):
    return {
        "sample_id": [item["sample_id"] for item in batch],
        "action": torch.stack([item["action"] for item in batch]),
        "text_emb": torch.stack([item["text_emb"] for item in batch]),
        "robot_state": torch.stack([item["robot_state"] for item in batch]),
        "numeric": torch.stack([item["numeric"] for item in batch]),
    }


def train(jsonl_path: str, output_dir: str, steps: int, batch_size: int, lr: float) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = LanguageActionDataset(jsonl_path, text_encoder=HashingTextEncoder())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    model = UlaFmModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    metrics_path = output / "metrics.jsonl"
    iterator = iter(loader)
    with metrics_path.open("w", encoding="utf-8") as metrics:
        for step in range(1, steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            x_data = batch["action"]
            condition = {
                "text_emb": batch["text_emb"],
                "robot_state": batch["robot_state"],
                "numeric": batch["numeric"],
            }
            x_t, t, target_v = flow_match_batch(x_data)
            pred_v = model(x_t, t, condition)
            loss, parts = total_training_loss(pred_v, target_v, x_data)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            record = {"step": step, "loss": float(loss.detach().cpu()), **parts}
            metrics.write(json.dumps(record) + "\n")
            metrics.flush()
    torch.save({"model": model.state_dict()}, output / "checkpoint.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ULA-FM v0.1")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    train(args.jsonl, args.output_dir, args.steps, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add config**

Create `/Users/demo/Desktop/upper_body_motion_roadmap/configs/ula_fm_v0.yaml`:

```yaml
model:
  action_dim: 15
  horizon: 120
  fps: 60
  model_dim: 384
  layers: 8
  heads: 6
training:
  text_encoder: frozen_hashing_v0
  train_language_adapter: true
  train_text_encoder: false
  train_text_lora: false
  batch_size: 4
  learning_rate: 0.0001
  gradient_clip_norm: 1.0
sampling:
  steps: 12
  candidates: 4
  execute_prefix_sec: 0.5
```

- [ ] **Step 3: Smoke-train after JSONL exists**

After the retarget batch produces the full language-action JSONL, run:

```bash
PYTHONPATH=/Users/demo/Desktop/upper_body_motion_roadmap /Users/demo/Desktop/mjlab/.venv/bin/python -m upper_body_lam.train \
  --jsonl /Users/demo/Desktop/upper_body_motion_roadmap/video/seamless_interaction_50g/batch_v2_retarget_ik_contact_safe/full_single_progress/language_action_index.jsonl \
  --output-dir /Users/demo/Desktop/upper_body_motion_roadmap/video/seamless_interaction_50g/ula_fm_runs/debug_overfit \
  --steps 20 \
  --batch-size 2
```

Expected:

- `checkpoint.pt` exists.
- `metrics.jsonl` exists.
- Loss is finite every step.

---

## Task 9: Update HTML Demo With Final Module Boundaries

**Files:**
- Modify: `/Users/demo/Desktop/upper_body_motion_roadmap/ula_fm_network_demo.html`

- [ ] **Step 1: Verify HTML contains language-head training status**

Run:

```bash
rg -n "Language Head Detail|FROZEN|TRAINABLE|OPTIONAL v0.2|Language Adapter|FiLM" /Users/demo/Desktop/upper_body_motion_roadmap/ula_fm_network_demo.html
```

Expected: all key strings are present.

- [ ] **Step 2: Keep page aligned with implementation names**

After creating the package, make sure the HTML names match:

- `HashingTextEncoder`
- `Language Adapter`
- `Condition Fusion`
- `AdaLayerNormBlock`
- `UlaFmModel`
- `flow_match_batch`
- `euler_sample`

---

## Verification Commands

Run all LAM tests:

```bash
PYTHONPATH=/Users/demo/Desktop/upper_body_motion_roadmap /Users/demo/Desktop/mjlab/.venv/bin/python -m pytest \
  /Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_constants.py \
  /Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_text_encoder.py \
  /Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_dataset.py \
  /Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_flow.py \
  /Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_model.py \
  /Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_training_step.py \
  /Users/demo/Desktop/upper_body_motion_roadmap/tests/test_lam_export.py \
  -v
```

Run existing project tests after integration:

```bash
PYTHONPATH=/Users/demo/Desktop/upper_body_motion_roadmap /Users/demo/Desktop/mjlab/.venv/bin/python -m pytest /Users/demo/Desktop/upper_body_motion_roadmap/tests -v
```

## First Success Criteria

The first useful milestone is not a good-looking generated motion. The first milestone is:

- The dataset loader reads real `language_action_index.jsonl` records.
- A 2-layer tiny model can overfit 10 windows.
- Sampling exports a 15-joint CSV with the correct V2 joint order.
- The CSV can be passed into the existing MuJoCo preview/safety pipeline.
- The HTML clearly marks trainable vs frozen modules.

Only after this should v0.2 consider LoRA or a real sentence-transformer dependency.

