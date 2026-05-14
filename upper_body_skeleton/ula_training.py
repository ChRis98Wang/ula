#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn

from upper_body_skeleton.retarget_v2 import JOINT_LIMITS, JOINT_ORDER


INTENT_IDS = {"waiting": 0, "explaining": 1, "refusing": 2, "requesting_help": 3, "greeting": 4, "warning": 5}
AFFECT_IDS = {"low_confidence_unknown": 0, "neutral": 1, "sad_like": 2, "nervous": 3, "friendly": 4, "uncertain": 5, "angry_like": 6, "excited": 7}
STYLE_IDS = {"restrained": 0, "relaxed": 1, "energetic": 2}
GESTURE_IDS = {"null": 0, "upper_body_gesture": 1, "pointing": 2, "crossed_arms": 3, "shrugging": 4, "waving": 5}
BASE_CONDITION_DIM = len(INTENT_IDS) + len(AFFECT_IDS) + len(STYLE_IDS) + len(GESTURE_IDS) + 5
TEXT_EMBED_DIM = 64


INTENT_KEYWORDS = [
    ("warning", ("warning", "warn", "stop", "danger", "careful", "警告", "小心", "停止", "危险")),
    ("requesting_help", ("help", "assist", "please", "request", "求助", "帮忙", "请求", "请")),
    ("greeting", ("hello", "hi", "greet", "wave", "你好", "打招呼", "挥手")),
    ("refusing", ("refuse", "reject", "no ", "don't", "cannot", "拒绝", "不要", "不行")),
    ("explaining", ("explain", "tell", "describe", "conversational", "解释", "说明", "表达", "讲")),
]
AFFECT_KEYWORDS = [
    ("excited", ("excited", "energetic", "happy", "eager", "兴奋", "激动", "高兴")),
    ("angry_like", ("angry", "frustrated", "annoyed", "生气", "愤怒", "不满")),
    ("nervous", ("nervous", "tense", "anxious", "紧张", "焦虑")),
    ("uncertain", ("uncertain", "unsure", "hesitant", "不确定", "犹豫")),
    ("sad_like", ("sad", "down", "upset", "难过", "低落")),
    ("friendly", ("friendly", "warm", "kind", "友好", "亲切")),
]
STYLE_KEYWORDS = [
    ("energetic", ("energetic", "large", "fast", "big", "active", "用力", "大幅度", "快速")),
    ("relaxed", ("relaxed", "soft", "calm", "loose", "放松", "柔和", "平静")),
    ("restrained", ("restrained", "small", "subtle", "reserved", "克制", "小幅度", "保守")),
]
GESTURE_KEYWORDS = [
    ("crossed_arms", ("cross", "fold arms", "arms crossed", "交叉", "抱臂")),
    ("pointing", ("point", "pointing", "指", "指向")),
    ("shrugging", ("shrug", "shrugging", "耸肩")),
    ("waving", ("wave", "waving", "挥手", "摆手")),
    ("upper_body_gesture", ("gesture", "body", "upper body", "手势", "肢体", "上肢")),
]
AFFECT_DEFAULTS = {
    "low_confidence_unknown": (0.0, 0.0, 0, 0),
    "neutral": (0.0, 0.0, 0, 0),
    "sad_like": (-0.25, -0.45, 0, 0),
    "nervous": (0.45, -0.2, 2, 0),
    "friendly": (0.25, 0.45, 1, 2),
    "uncertain": (0.2, -0.15, 1, 0),
    "angry_like": (0.65, -0.55, 3, 0),
    "excited": (0.75, 0.45, 3, 2),
}


def one_hot(index, size):
    vec = np.zeros(size, dtype=np.float32)
    vec[max(0, min(size - 1, int(index)))] = 1.0
    return vec


def _first_keyword_label(text, choices, default):
    lowered = f" {text.lower()} "
    for label, keywords in choices:
        if any(keyword in lowered for keyword in keywords):
            return label
    return default


def infer_codes_from_text(text):
    return {
        "intent": _first_keyword_label(text, INTENT_KEYWORDS, "explaining"),
        "observed_affect": _first_keyword_label(text, AFFECT_KEYWORDS, "neutral"),
        "motion_style": _first_keyword_label(text, STYLE_KEYWORDS, "restrained"),
        "semantic_gesture": _first_keyword_label(text, GESTURE_KEYWORDS, "upper_body_gesture"),
    }


def frozen_text_embedding(text, dim=TEXT_EMBED_DIM):
    """Small deterministic frozen text encoder used until a larger LM is wired in."""
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
    if not tokens:
        return vec
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
        for offset in range(0, len(digest), 2):
            index = digest[offset] % dim
            sign = 1.0 if digest[offset + 1] % 2 == 0 else -1.0
            vec[index] += sign
    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        vec /= norm
    return vec


def _semantic_text(meta_row):
    parts = [
        meta_row.get("language_instruction", ""),
        meta_row.get("raw_transcript", ""),
        meta_row.get("scenario_description", ""),
        meta_row.get("action_description", ""),
        meta_row.get("intent_text", ""),
        meta_row.get("mood_text", ""),
        meta_row.get("rationale_text", ""),
    ]
    return " ".join(str(part) for part in parts if part)


def condition_vector(meta_row):
    text = _semantic_text(meta_row)
    labels = []
    labels.append(one_hot(INTENT_IDS.get(meta_row.get("intent", ""), 0), len(INTENT_IDS)))
    labels.append(one_hot(AFFECT_IDS.get(meta_row.get("observed_affect", ""), 0), len(AFFECT_IDS)))
    labels.append(one_hot(STYLE_IDS.get(meta_row.get("motion_style", ""), 0), len(STYLE_IDS)))
    labels.append(one_hot(GESTURE_IDS.get(meta_row.get("semantic_gesture", ""), 0), len(GESTURE_IDS)))
    scalars = np.asarray(
        [
            meta_row.get("arousal") if meta_row.get("arousal") is not None else 0.0,
            meta_row.get("valence") if meta_row.get("valence") is not None else 0.0,
            meta_row.get("motion_energy") if meta_row.get("motion_energy") is not None else 0.0,
            meta_row.get("arousal_token") if meta_row.get("arousal_token") is not None else 0.0,
            meta_row.get("valence_token") if meta_row.get("valence_token") is not None else 0.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([*labels, scalars, frozen_text_embedding(text)], axis=0)


def build_condition_from_text(
    text,
    *,
    intent=None,
    affect=None,
    style=None,
    gesture=None,
    arousal=None,
    valence=None,
    arousal_token=None,
    valence_token=None,
    motion_energy=0.05,
    text_dim=TEXT_EMBED_DIM,
):
    codes = infer_codes_from_text(text)
    if intent is not None:
        codes["intent"] = intent
    if affect is not None:
        codes["observed_affect"] = affect
    if style is not None:
        codes["motion_style"] = style
    if gesture is not None:
        codes["semantic_gesture"] = gesture
    default_arousal, default_valence, default_arousal_token, default_valence_token = AFFECT_DEFAULTS.get(
        codes["observed_affect"], AFFECT_DEFAULTS["neutral"]
    )
    meta_row = {
        "language_instruction": text,
        "intent": codes["intent"],
        "observed_affect": codes["observed_affect"],
        "motion_style": codes["motion_style"],
        "semantic_gesture": codes["semantic_gesture"],
        "arousal": default_arousal if arousal is None else arousal,
        "valence": default_valence if valence is None else valence,
        "motion_energy": motion_energy,
        "arousal_token": default_arousal_token if arousal_token is None else arousal_token,
        "valence_token": default_valence_token if valence_token is None else valence_token,
    }
    base = condition_vector(meta_row)
    if text_dim == TEXT_EMBED_DIM:
        return base
    return np.concatenate([base[:BASE_CONDITION_DIM], frozen_text_embedding(text, dim=text_dim)], axis=0)


def load_lerobot_episodes(dataset_dir, max_episodes=None):
    dataset_dir = Path(dataset_dir)
    semantic_path = dataset_dir / "meta" / "semantic_index.parquet"
    episode_meta = {row["episode_index"]: row for row in pq.read_table(semantic_path).to_pylist()}
    grouped = {}
    episodes = []

    def flush_complete(episode_index):
        frame_rows = sorted(grouped.pop(episode_index), key=lambda row: row["frame_index"])
        actions = np.asarray([row["observation.state"] for row in frame_rows], dtype=np.float32)
        meta = episode_meta.get(episode_index, {})
        episodes.append(
            {
                "episode_index": episode_index,
                "actions": actions,
                "condition": condition_vector(meta),
                "task_index": frame_rows[0]["task_index"] if frame_rows else 0,
            }
        )

    for path in sorted((dataset_dir / "data").glob("chunk-*/*.parquet")):
        for row in pq.read_table(path).to_pylist():
            episode_index = row["episode_index"]
            grouped.setdefault(episode_index, []).append(row)
            if row.get("next.done"):
                flush_complete(episode_index)
                if max_episodes is not None and len(episodes) >= max_episodes:
                    return episodes
    for episode_index in sorted(grouped):
        if max_episodes is not None and len(episodes) >= max_episodes:
            break
        flush_complete(episode_index)
    return episodes


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half, device=t.device))
        angles = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class UlaFmModel(nn.Module):
    def __init__(self, action_dim=15, condition_dim=BASE_CONDITION_DIM + TEXT_EMBED_DIM, hidden_dim=256, layers=4):
        super().__init__()
        self.action_dim = action_dim
        self.condition_dim = condition_dim
        self.input = nn.Linear(action_dim, hidden_dim)
        self.time = SinusoidalTimeEmbedding(hidden_dim)
        self.cond = nn.Sequential(nn.Linear(condition_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x_t, t, condition):
        h = self.input(x_t)
        cond = self.cond(condition)[:, None, :]
        time = self.time(t)[:, None, :]
        h = h + cond + time
        return self.output(self.blocks(h))


def joint_limit_tensors(device, action_dim):
    lowers = [JOINT_LIMITS[joint][0] for joint in JOINT_ORDER[:action_dim]]
    uppers = [JOINT_LIMITS[joint][1] for joint in JOINT_ORDER[:action_dim]]
    return (
        torch.tensor(lowers, dtype=torch.float32, device=device)[None, None, :],
        torch.tensor(uppers, dtype=torch.float32, device=device)[None, None, :],
    )


def sample_trajectory(model, condition, frames=120, action_dim=15, steps=24, device="cpu", seed=None):
    if seed is not None:
        torch.manual_seed(int(seed))
    model.to(device)
    model.eval()
    condition_tensor = torch.as_tensor(condition, dtype=torch.float32, device=device)
    if condition_tensor.ndim == 1:
        condition_tensor = condition_tensor[None, :]
    x = torch.randn((condition_tensor.shape[0], frames, action_dim), dtype=torch.float32, device=device)
    dt = 1.0 / float(max(1, steps))
    lower, upper = joint_limit_tensors(device, action_dim)
    with torch.no_grad():
        for index in range(steps):
            t = torch.full((condition_tensor.shape[0],), index * dt, dtype=torch.float32, device=device)
            velocity = model(x, t, condition_tensor)
            x = torch.clamp(x + dt * velocity, lower, upper)
    return x[0].detach().cpu().numpy()


def write_generated_csv(path, trajectory, fps=30.0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["time_sec"] + JOINT_ORDER[: trajectory.shape[1]])
        writer.writeheader()
        for frame_index, values in enumerate(trajectory):
            row = {"time_sec": f"{frame_index / float(fps):.6f}"}
            row.update({joint: f"{float(values[i]):.6f}" for i, joint in enumerate(JOINT_ORDER[: trajectory.shape[1]])})
            writer.writerow(row)


def write_training_preview(
    model,
    *,
    preview_root,
    step,
    text,
    frames,
    sampling_steps,
    fps,
    device,
    seed,
    width=1280,
    height=720,
):
    # Import lazily so training-only tests and environments do not need a renderer
    # until preview generation is explicitly requested.
    from upper_body_skeleton.mujoco_playback import render_motion

    preview_dir = Path(preview_root) / f"step_{int(step):06d}"
    preview_dir.mkdir(parents=True, exist_ok=True)
    condition = build_condition_from_text(text)
    trajectory = sample_trajectory(
        model,
        condition=condition,
        frames=frames,
        action_dim=len(JOINT_ORDER),
        steps=sampling_steps,
        device=device,
        seed=seed,
    )
    csv_path = preview_dir / "generated.csv"
    npz_path = preview_dir / "generated.npz"
    mp4_path = preview_dir / "preview.mp4"
    summary_path = preview_dir / "summary.json"
    write_generated_csv(csv_path, trajectory, fps=fps)
    np.savez_compressed(
        npz_path,
        trajectory=trajectory.astype(np.float32),
        joint_order=np.asarray(JOINT_ORDER, dtype=object),
        text=text,
        fps=np.asarray(fps, dtype=np.float32),
        step=np.asarray(step, dtype=np.int64),
    )
    render_summary = render_motion(csv_path, mp4_path, fps=fps, width=width, height=height)
    summary = {
        "step": int(step),
        "text": text,
        "generated_csv": str(csv_path),
        "generated_npz": str(npz_path),
        "preview_mp4": str(mp4_path),
        "frames": int(trajectory.shape[0]),
        "sampling_steps": int(sampling_steps),
        "fps": float(fps),
        "render": render_summary,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def sample_batch(episodes, batch_size, device):
    selected = random.choices(episodes, k=batch_size)
    actions = torch.tensor(np.stack([episode["actions"] for episode in selected]), dtype=torch.float32, device=device)
    condition = torch.tensor(np.stack([episode["condition"] for episode in selected]), dtype=torch.float32, device=device)
    return actions, condition


def flow_matching_loss(model, actions, condition):
    noise = torch.randn_like(actions)
    t = torch.rand(actions.shape[0], device=actions.device)
    x_t = (1.0 - t[:, None, None]) * noise + t[:, None, None] * actions
    target_v = actions - noise
    pred_v = model(x_t, t, condition)
    return torch.mean((pred_v - target_v) ** 2)


def train_steps(
    model,
    episodes,
    steps,
    batch_size,
    lr,
    device,
    log_interval=0,
    progress_path=None,
    preview_interval=0,
    preview_root=None,
    preview_text=None,
    preview_frames=120,
    preview_sampling_steps=32,
    preview_fps=30.0,
    preview_seed=7,
    preview_width=1280,
    preview_height=720,
):
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    progress_path = Path(progress_path) if progress_path else None
    if progress_path:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
    for step_index in range(steps):
        actions, condition = sample_batch(episodes, batch_size, device)
        loss = flow_matching_loss(model, actions, condition)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))
        step = step_index + 1
        if log_interval and (step == 1 or step % log_interval == 0 or step == steps):
            event = {"step": step, "steps": steps, "loss": losses[-1]}
            print(json.dumps(event), flush=True)
            if progress_path:
                with progress_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(event) + "\n")
        if preview_interval and preview_root and preview_text and (step % preview_interval == 0 or step == steps):
            summary = write_training_preview(
                model,
                preview_root=preview_root,
                step=step,
                text=preview_text,
                frames=preview_frames,
                sampling_steps=preview_sampling_steps,
                fps=preview_fps,
                device=device,
                seed=preview_seed + step,
                width=preview_width,
                height=preview_height,
            )
            print(json.dumps({"preview": summary["preview_mp4"], "step": step}), flush=True)
    return losses


def choose_device(requested):
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def write_training_log(path, config, losses):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": config, "losses": losses, "final_loss": losses[-1] if losses else None}, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Train a minimal body-only ULA-FM model from LeRobot parquet")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--preview-every-steps", type=int, default=0)
    parser.add_argument("--preview-text", default="紧张地解释，同时双手做克制的上肢手势")
    parser.add_argument("--preview-dir")
    parser.add_argument("--preview-frames", type=int, default=120)
    parser.add_argument("--preview-sampling-steps", type=int, default=32)
    parser.add_argument("--preview-fps", type=float, default=30.0)
    parser.add_argument("--preview-seed", type=int, default=7)
    parser.add_argument("--preview-width", type=int, default=1280)
    parser.add_argument("--preview-height", type=int, default=720)
    args = parser.parse_args()
    device = choose_device(args.device)
    episodes = load_lerobot_episodes(args.dataset_dir, max_episodes=args.max_episodes)
    if not episodes:
        raise SystemExit("no episodes loaded")
    model = UlaFmModel(
        action_dim=len(JOINT_ORDER),
        condition_dim=episodes[0]["condition"].shape[0],
        hidden_dim=args.hidden_dim,
        layers=args.layers,
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    losses = train_steps(
        model,
        episodes,
        args.steps,
        args.batch_size,
        args.lr,
        device,
        log_interval=args.log_interval,
        progress_path=out / "progress.jsonl",
        preview_interval=args.preview_every_steps,
        preview_root=args.preview_dir or out / "previews",
        preview_text=args.preview_text,
        preview_frames=args.preview_frames,
        preview_sampling_steps=args.preview_sampling_steps,
        preview_fps=args.preview_fps,
        preview_seed=args.preview_seed,
        preview_width=args.preview_width,
        preview_height=args.preview_height,
    )
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "joint_order": JOINT_ORDER,
            "condition_dim": episodes[0]["condition"].shape[0],
            "action_dim": len(JOINT_ORDER),
            "config": vars(args) | {"device": device, "episodes_loaded": len(episodes)},
        },
        out / "ula_fm_checkpoint.pt",
    )
    write_training_log(out / "train_log.json", vars(args) | {"device": device, "episodes_loaded": len(episodes)}, losses)
    print(json.dumps({"output_dir": str(out), "steps": args.steps, "episodes_loaded": len(episodes), "final_loss": losses[-1]}, indent=2))


if __name__ == "__main__":
    main()
