#!/usr/bin/env python3
from collections import defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn
from torch.nn import functional as F

from upper_body_skeleton.kimodo_semantics import KIMODO_BEHAVIOR_IDS, KIMODO_EMOTION_IDS


BEHAVIOR_TO_INDEX = {label: index for index, label in enumerate(KIMODO_BEHAVIOR_IDS)}
EMOTION_TO_INDEX = {label: index for index, label in enumerate(KIMODO_EMOTION_IDS)}


def _semantic_key(episode):
    meta = episode.get("meta", {})
    behavior_id = meta.get("behavior_id")
    emotion_id = meta.get("emotion_id")
    if not behavior_id or not emotion_id:
        raise ValueError("each episode must contain meta.behavior_id and meta.emotion_id")
    return str(behavior_id), str(emotion_id)


def _split_counts(size, fractions):
    if size < 3:
        raise ValueError("each semantic group needs at least three episodes")
    train_fraction, validation_fraction, test_fraction = (float(value) for value in fractions)
    if min(train_fraction, validation_fraction, test_fraction) <= 0:
        raise ValueError("split fractions must be positive")
    total = train_fraction + validation_fraction + test_fraction
    train_count = max(1, int(size * train_fraction / total))
    validation_count = max(1, int(size * validation_fraction / total))
    test_count = size - train_count - validation_count
    while test_count < 1 and train_count > 1:
        train_count -= 1
        test_count += 1
    if test_count < 1:
        raise ValueError("split fractions leave no test episodes")
    return train_count, validation_count, test_count


def stratified_episode_split(episodes, *, seed=7, fractions=(0.8, 0.1, 0.1)):
    groups = defaultdict(list)
    for episode in episodes:
        groups[_semantic_key(episode)].append(episode)
    if not groups:
        raise ValueError("cannot split an empty episode collection")

    rng = np.random.default_rng(int(seed))
    train = []
    validation = []
    test = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda episode: int(episode["episode_index"]))
        rng.shuffle(group)
        train_count, validation_count, _ = _split_counts(len(group), fractions)
        train.extend(group[:train_count])
        validation.extend(group[train_count : train_count + validation_count])
        test.extend(group[train_count + validation_count :])
    return train, validation, test


def load_motion_latent_episodes(dataset_dir, max_episodes=None, *, batch_rows=65_536):
    dataset_dir = Path(dataset_dir)
    semantic_path = dataset_dir / "meta" / "semantic_index.parquet"
    semantic_rows = pq.read_table(
        semantic_path,
        columns=["episode_index", "behavior_id", "emotion_id"],
    ).to_pylist()
    semantic_index = {int(row["episode_index"]): row for row in semantic_rows}
    if max_episodes is not None and int(max_episodes) <= 0:
        return []

    episodes = []
    completed = set()
    current_index = None
    current_frames = []
    current_actions = []

    def flush_current():
        nonlocal current_index, current_frames, current_actions
        if current_index is None:
            return False
        if current_index in completed:
            raise ValueError(f"episode {current_index} is not contiguous in the parquet data")
        meta = semantic_index.get(current_index)
        if meta is None:
            raise ValueError(f"episode {current_index} has no semantic metadata")
        order = np.argsort(np.asarray(current_frames, dtype=np.int64))
        actions = np.asarray(current_actions, dtype=np.float32)[order]
        episodes.append(
            {
                "episode_index": current_index,
                "actions": actions,
                "meta": {
                    "behavior_id": str(meta["behavior_id"]),
                    "emotion_id": str(meta["emotion_id"]),
                },
            }
        )
        completed.add(current_index)
        current_index = None
        current_frames = []
        current_actions = []
        return max_episodes is not None and len(episodes) >= int(max_episodes)

    columns = ["episode_index", "frame_index", "observation.state", "next.done"]
    for path in sorted((dataset_dir / "data").glob("chunk-*/*.parquet")):
        parquet = pq.ParquetFile(path)
        for record_batch in parquet.iter_batches(batch_size=int(batch_rows), columns=columns):
            values = record_batch.to_pydict()
            for episode_index, frame_index, action, done in zip(
                values["episode_index"],
                values["frame_index"],
                values["observation.state"],
                values["next.done"],
            ):
                episode_index = int(episode_index)
                if current_index is not None and episode_index != current_index:
                    if flush_current():
                        return episodes
                if current_index is None:
                    current_index = episode_index
                current_frames.append(int(frame_index))
                current_actions.append(action)
                if bool(done) and flush_current():
                    return episodes
    flush_current()
    return episodes


def compute_motion_normalization(episodes, *, eps=1e-4):
    if not episodes:
        raise ValueError("cannot compute normalization from no episodes")
    actions = np.concatenate([np.asarray(episode["actions"], dtype=np.float32) for episode in episodes], axis=0)
    mean = torch.as_tensor(actions.mean(axis=0), dtype=torch.float32).reshape(1, 1, -1)
    std = torch.as_tensor(actions.std(axis=0), dtype=torch.float32).clamp_min(float(eps)).reshape(1, 1, -1)
    return {"mean": mean, "std": std}


def build_motion_features(actions, stats):
    actions = torch.as_tensor(actions, dtype=torch.float32)
    if actions.ndim == 2:
        actions = actions.unsqueeze(0)
    if actions.ndim != 3:
        raise ValueError("actions must have shape [batch, frames, joints] or [frames, joints]")
    mean = torch.as_tensor(stats["mean"], dtype=actions.dtype, device=actions.device)
    std = torch.as_tensor(stats["std"], dtype=actions.dtype, device=actions.device)
    normalized = (actions - mean) / std
    velocity = torch.zeros_like(normalized)
    velocity[:, 1:] = normalized[:, 1:] - normalized[:, :-1]
    return torch.cat([normalized, velocity], dim=-1).transpose(1, 2)


def _group_norm_groups(channels):
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class MotionMetricEncoder(nn.Module):
    def __init__(
        self,
        *,
        action_dim=15,
        latent_dim=128,
        hidden_dim=128,
        behavior_classes=len(KIMODO_BEHAVIOR_IDS),
        emotion_classes=len(KIMODO_EMOTION_IDS),
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        output_dim = self.hidden_dim * 2
        self.backbone = nn.Sequential(
            nn.Conv1d(self.action_dim * 2, self.hidden_dim, kernel_size=7, padding=3),
            nn.GroupNorm(_group_norm_groups(self.hidden_dim), self.hidden_dim),
            nn.SiLU(),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(_group_norm_groups(self.hidden_dim), self.hidden_dim),
            nn.SiLU(),
            nn.Conv1d(self.hidden_dim, output_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_group_norm_groups(output_dim), output_dim),
            nn.SiLU(),
        )
        self.projection = nn.Sequential(
            nn.Linear(output_dim * 2, self.hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 2, self.latent_dim),
        )
        self.behavior_head = nn.Linear(self.latent_dim, int(behavior_classes))
        self.emotion_head = nn.Linear(self.latent_dim, int(emotion_classes))
        self.descriptor_head = nn.Linear(self.latent_dim, 6)

    def forward(self, features):
        if features.ndim != 3 or features.shape[1] != self.action_dim * 2:
            raise ValueError(f"features must have shape [batch, {self.action_dim * 2}, frames]")
        encoded = self.backbone(features)
        pooled = torch.cat([encoded.mean(dim=-1), encoded.amax(dim=-1)], dim=-1)
        latent = self.projection(pooled)
        embedding = F.normalize(latent, dim=-1)
        return {
            "embedding": embedding,
            "behavior_logits": self.behavior_head(latent),
            "emotion_logits": self.emotion_head(latent),
            "descriptors": self.descriptor_head(latent),
        }


def compute_motion_descriptors(actions, stats=None, *, eps=1e-8):
    actions = torch.as_tensor(actions, dtype=torch.float32)
    if actions.ndim == 2:
        actions = actions.unsqueeze(0)
    if actions.ndim != 3:
        raise ValueError("actions must have shape [batch, frames, joints] or [frames, joints]")
    if stats is not None:
        mean = torch.as_tensor(stats["mean"], dtype=actions.dtype, device=actions.device)
        std = torch.as_tensor(stats["std"], dtype=actions.dtype, device=actions.device)
        actions = (actions - mean) / std

    velocity = actions[:, 1:] - actions[:, :-1]
    acceleration = velocity[:, 1:] - velocity[:, :-1]
    pose_amplitude = actions.std(dim=1, unbiased=False).mean(dim=-1)
    velocity_rms = torch.sqrt(velocity.square().mean(dim=(1, 2)) + eps)
    acceleration_rms = torch.sqrt(acceleration.square().mean(dim=(1, 2)) + eps)

    if actions.shape[-1] >= 15:
        left_velocity = velocity[:, :, 3:9]
        right_velocity = velocity[:, :, 9:15]
    else:
        midpoint = actions.shape[-1] // 2
        left_velocity = velocity[:, :, :midpoint]
        right_velocity = velocity[:, :, midpoint:]
    left_activity = torch.sqrt(left_velocity.square().mean(dim=(1, 2)) + eps)
    right_activity = torch.sqrt(right_velocity.square().mean(dim=(1, 2)) + eps)
    asymmetry = (left_activity - right_activity).abs() / (left_activity + right_activity + eps)
    return torch.stack(
        [pose_amplitude, velocity_rms, acceleration_rms, left_activity, right_activity, asymmetry], dim=-1
    )


def supervised_contrastive_loss(embeddings, labels, *, temperature=0.1):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    embeddings = F.normalize(embeddings, dim=-1)
    labels = torch.as_tensor(labels, device=embeddings.device).reshape(-1)
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError("embeddings and labels must have the same batch size")

    count = embeddings.shape[0]
    self_mask = torch.eye(count, dtype=torch.bool, device=embeddings.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    valid = positive_mask.any(dim=1)
    if not valid.any():
        return embeddings.sum() * 0.0

    logits = embeddings @ embeddings.T / float(temperature)
    logits = logits.masked_fill(self_mask, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_log_prob = torch.where(positive_mask, log_prob, torch.zeros_like(log_prob)).sum(dim=1)
    positive_count = positive_mask.sum(dim=1).clamp_min(1)
    return -(positive_log_prob[valid] / positive_count[valid]).mean()


def motion_metric_loss(
    output,
    behavior_target,
    emotion_target,
    descriptor_target,
    *,
    behavior_weight=1.0,
    emotion_weight=0.5,
    contrastive_weight=0.25,
    descriptor_weight=0.2,
):
    behavior_target = torch.as_tensor(behavior_target, dtype=torch.long, device=output["embedding"].device)
    emotion_target = torch.as_tensor(emotion_target, dtype=torch.long, device=output["embedding"].device)
    descriptor_target = torch.as_tensor(
        descriptor_target, dtype=output["descriptors"].dtype, device=output["descriptors"].device
    )
    behavior = F.cross_entropy(output["behavior_logits"], behavior_target)
    emotion = F.cross_entropy(output["emotion_logits"], emotion_target)
    joint_label = behavior_target * output["emotion_logits"].shape[-1] + emotion_target
    contrastive = supervised_contrastive_loss(output["embedding"], joint_label)
    descriptor = F.smooth_l1_loss(output["descriptors"], descriptor_target)
    total = (
        float(behavior_weight) * behavior
        + float(emotion_weight) * emotion
        + float(contrastive_weight) * contrastive
        + float(descriptor_weight) * descriptor
    )
    return {
        "total": total,
        "behavior": behavior,
        "emotion": emotion,
        "contrastive": contrastive,
        "descriptor": descriptor,
    }


def sample_semantic_pair_batch(episodes, *, batch_size, seed=None, rng=None):
    if batch_size <= 0 or batch_size % 2:
        raise ValueError("batch_size must be a positive even number")
    groups = defaultdict(list)
    for episode in episodes:
        groups[_semantic_key(episode)].append(episode)
    groups = {key: values for key, values in groups.items() if len(values) >= 2}
    if not groups:
        raise ValueError("paired batches require at least one semantic group with two episodes")
    if rng is None:
        rng = np.random.default_rng(seed)

    keys = sorted(groups)
    batch = []
    for _ in range(batch_size // 2):
        key = keys[int(rng.integers(0, len(keys)))]
        pair_indices = rng.choice(len(groups[key]), size=2, replace=False)
        batch.extend([groups[key][int(index)] for index in pair_indices])
    return batch


@dataclass(frozen=True)
class MotionLatentIndex:
    embeddings: np.ndarray
    episode_indices: np.ndarray
    behavior_ids: np.ndarray
    emotion_ids: np.ndarray

    def __post_init__(self):
        embeddings = np.asarray(self.embeddings, dtype=np.float32)
        episode_indices = np.asarray(self.episode_indices, dtype=np.int64)
        behavior_ids = np.asarray(self.behavior_ids, dtype=str)
        emotion_ids = np.asarray(self.emotion_ids, dtype=str)
        if embeddings.ndim != 2:
            raise ValueError("embeddings must have shape [episodes, latent_dim]")
        if not np.isfinite(embeddings).all():
            raise ValueError("embeddings must be finite")
        size = embeddings.shape[0]
        if any(values.shape != (size,) for values in (episode_indices, behavior_ids, emotion_ids)):
            raise ValueError("latent index fields must contain the same number of episodes")
        object.__setattr__(self, "embeddings", embeddings)
        object.__setattr__(self, "episode_indices", episode_indices)
        object.__setattr__(self, "behavior_ids", behavior_ids)
        object.__setattr__(self, "emotion_ids", emotion_ids)

    def __len__(self):
        return self.embeddings.shape[0]


def _normalized_embeddings(embeddings, eps=1e-12):
    embeddings = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, eps)


def _embedding_effective_rank(embeddings, eps=1e-12):
    if len(embeddings) == 0:
        return 0.0
    singular_values = np.linalg.svd(_normalized_embeddings(embeddings), compute_uv=False)
    energy = singular_values**2
    total = float(energy.sum())
    if total <= eps:
        return 0.0
    probabilities = energy / total
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, eps)))
    return float(np.exp(entropy))


def compute_latent_diagnostics(reference, query):
    if not isinstance(reference, MotionLatentIndex) or not isinstance(query, MotionLatentIndex):
        raise TypeError("reference and query must be MotionLatentIndex instances")
    if len(reference) == 0 or len(query) == 0:
        raise ValueError("reference and query indices must not be empty")

    reference_embeddings = _normalized_embeddings(reference.embeddings)
    query_embeddings = _normalized_embeddings(query.embeddings)
    distances = 1.0 - query_embeddings @ reference_embeddings.T
    nearest = distances.argmin(axis=1)
    predicted_behavior = reference.behavior_ids[nearest]
    predicted_emotion = reference.emotion_ids[nearest]
    behavior_correct = predicted_behavior == query.behavior_ids
    emotion_correct = predicted_emotion == query.emotion_ids

    same_joint = (query.behavior_ids[:, None] == reference.behavior_ids[None, :]) & (
        query.emotion_ids[:, None] == reference.emotion_ids[None, :]
    )
    intra_values = distances[same_joint]
    inter_values = distances[~same_joint]
    mean_intra = float(intra_values.mean()) if intra_values.size else 0.0
    mean_inter = float(inter_values.mean()) if inter_values.size else 0.0
    separation_ratio = mean_inter / max(mean_intra, 1e-12) if mean_inter > 0 else 0.0
    reference_effective_rank = _embedding_effective_rank(reference.embeddings)
    query_effective_rank = _embedding_effective_rank(query.embeddings)
    reference_collapsed = reference_effective_rank < 1.5
    query_collapsed = query_effective_rank < 1.5
    joint_group_count = len(set(zip(reference.behavior_ids.tolist(), reference.emotion_ids.tolist())))
    return {
        "reference_count": int(len(reference)),
        "query_count": int(len(query)),
        "knn_behavior_accuracy": float(behavior_correct.mean()),
        "knn_emotion_accuracy": float(emotion_correct.mean()),
        "knn_joint_accuracy": float((behavior_correct & emotion_correct).mean()),
        "random_joint_accuracy": float(1.0 / max(1, joint_group_count)),
        "mean_intra_class_distance": mean_intra,
        "mean_inter_class_distance": mean_inter,
        "separation_ratio": float(separation_ratio),
        "reference_effective_rank": reference_effective_rank,
        "query_effective_rank": query_effective_rank,
        "reference_collapsed": bool(reference_collapsed),
        "query_collapsed": bool(query_collapsed),
        "effective_rank": query_effective_rank,
        "collapsed": bool(query_collapsed),
    }


def cross_set_retrieval_loss(reference, query, *, temperature=0.1):
    if not isinstance(reference, MotionLatentIndex) or not isinstance(query, MotionLatentIndex):
        raise TypeError("reference and query must be MotionLatentIndex instances")
    if len(reference) == 0 or len(query) == 0:
        raise ValueError("reference and query indices must not be empty")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    reference_embeddings = _normalized_embeddings(reference.embeddings)
    query_embeddings = _normalized_embeddings(query.embeddings)
    logits = query_embeddings @ reference_embeddings.T / float(temperature)
    positive_mask = (query.behavior_ids[:, None] == reference.behavior_ids[None, :]) & (
        query.emotion_ids[:, None] == reference.emotion_ids[None, :]
    )
    if not positive_mask.any(axis=1).all():
        missing = np.flatnonzero(~positive_mask.any(axis=1)).tolist()
        raise ValueError(f"query rows have no matching reference semantic label: {missing[:5]}")

    row_max = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - row_max)
    numerator = (exp_logits * positive_mask).sum(axis=1)
    denominator = exp_logits.sum(axis=1)
    return float(-np.log(np.maximum(numerator, 1e-300) / np.maximum(denominator, 1e-300)).mean())


def select_prototype_medoids(index, *, per_group=1):
    if not isinstance(index, MotionLatentIndex):
        raise TypeError("index must be a MotionLatentIndex")
    if per_group <= 0:
        raise ValueError("per_group must be positive")

    groups = defaultdict(list)
    for row, key in enumerate(zip(index.behavior_ids.tolist(), index.emotion_ids.tolist())):
        groups[key].append(row)
    normalized = _normalized_embeddings(index.embeddings)
    result = []
    for behavior_id, emotion_id in sorted(groups):
        rows = sorted(groups[(behavior_id, emotion_id)], key=lambda row: int(index.episode_indices[row]))
        group_embeddings = normalized[rows]
        distances = 1.0 - group_embeddings @ group_embeddings.T
        selected = [int(np.argmin(distances.mean(axis=1)))]
        while len(selected) < min(int(per_group), len(rows)):
            remaining = [candidate for candidate in range(len(rows)) if candidate not in selected]
            next_row = min(
                remaining,
                key=lambda candidate: (
                    float(distances[:, selected + [candidate]].min(axis=1).sum()),
                    int(index.episode_indices[rows[candidate]]),
                ),
            )
            selected.append(next_row)
        result.append(
            {
                "behavior_id": behavior_id,
                "emotion_id": emotion_id,
                "episode_indices": [int(index.episode_indices[rows[position]]) for position in selected],
            }
        )
    return result


def _validation_reference_bank(episodes, *, per_group=2):
    groups = defaultdict(list)
    for episode in episodes:
        groups[_semantic_key(episode)].append(episode)
    bank = []
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda episode: int(episode["episode_index"]))
        bank.extend(rows[: min(int(per_group), len(rows))])
    return bank


def _batch_tensors(episodes, stats, device):
    try:
        actions = np.stack([np.asarray(episode["actions"], dtype=np.float32) for episode in episodes])
    except ValueError as exc:
        raise ValueError("all episodes in a training batch must have the same action shape") from exc
    actions = torch.as_tensor(actions, dtype=torch.float32, device=device)
    behavior = []
    emotion = []
    for episode in episodes:
        behavior_id, emotion_id = _semantic_key(episode)
        if behavior_id not in BEHAVIOR_TO_INDEX:
            raise ValueError(f"unknown Kimodo behavior_id: {behavior_id}")
        if emotion_id not in EMOTION_TO_INDEX:
            raise ValueError(f"unknown Kimodo emotion_id: {emotion_id}")
        behavior.append(BEHAVIOR_TO_INDEX[behavior_id])
        emotion.append(EMOTION_TO_INDEX[emotion_id])
    features = build_motion_features(actions, stats)
    descriptors = compute_motion_descriptors(actions, stats)
    return (
        features,
        torch.as_tensor(behavior, dtype=torch.long, device=device),
        torch.as_tensor(emotion, dtype=torch.long, device=device),
        descriptors,
    )


def _evaluate_metric_model(model, episodes, stats, *, batch_size, device):
    was_training = model.training
    model.eval()
    totals = defaultdict(float)
    count = 0
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            batch = episodes[start : start + batch_size]
            features, behavior, emotion, descriptors = _batch_tensors(batch, stats, device)
            losses = motion_metric_loss(model(features), behavior, emotion, descriptors)
            for name, value in losses.items():
                totals[name] += float(value.detach().cpu()) * len(batch)
            count += len(batch)
    if was_training:
        model.train()
    return {name: value / max(1, count) for name, value in totals.items()}


def encode_motion_episodes(model, episodes, stats, *, batch_size=128, device="cpu"):
    if not episodes:
        raise ValueError("cannot encode an empty episode collection")
    device = torch.device(device)
    model.to(device)
    was_training = model.training
    model.eval()
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(episodes), int(batch_size)):
            batch = episodes[start : start + int(batch_size)]
            features, _, _, _ = _batch_tensors(batch, stats, device)
            embeddings.append(model(features)["embedding"].detach().cpu().numpy())
    if was_training:
        model.train()
    return MotionLatentIndex(
        embeddings=np.concatenate(embeddings, axis=0),
        episode_indices=np.asarray([episode["episode_index"] for episode in episodes], dtype=np.int64),
        behavior_ids=np.asarray([_semantic_key(episode)[0] for episode in episodes]),
        emotion_ids=np.asarray([_semantic_key(episode)[1] for episode in episodes]),
    )


def build_raw_motion_index(episodes, stats, *, stride=5):
    if not episodes:
        raise ValueError("cannot index an empty episode collection")
    if stride <= 0:
        raise ValueError("stride must be positive")
    try:
        actions = np.stack([np.asarray(episode["actions"], dtype=np.float32) for episode in episodes])
    except ValueError as exc:
        raise ValueError("all episodes in a raw motion index must have the same action shape") from exc
    features = build_motion_features(torch.from_numpy(actions), stats)
    embeddings = features[:, :, :: int(stride)].reshape(len(episodes), -1).numpy()
    return MotionLatentIndex(
        embeddings=embeddings,
        episode_indices=np.asarray([episode["episode_index"] for episode in episodes], dtype=np.int64),
        behavior_ids=np.asarray([_semantic_key(episode)[0] for episode in episodes]),
        emotion_ids=np.asarray([_semantic_key(episode)[1] for episode in episodes]),
    )


def _save_combined_latent_indices(path, indices):
    split_names = []
    embeddings = []
    episode_indices = []
    behavior_ids = []
    emotion_ids = []
    for split_name, index in indices.items():
        embeddings.append(index.embeddings)
        episode_indices.append(index.episode_indices)
        behavior_ids.append(index.behavior_ids)
        emotion_ids.append(index.emotion_ids)
        split_names.extend([split_name] * len(index))
    np.savez_compressed(
        path,
        embeddings=np.concatenate(embeddings, axis=0),
        episode_indices=np.concatenate(episode_indices, axis=0),
        behavior_ids=np.concatenate(behavior_ids, axis=0),
        emotion_ids=np.concatenate(emotion_ids, axis=0),
        split=np.asarray(split_names),
    )


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


RUN_ARTIFACT_NAMES = {
    "motion_latent_checkpoint.pt",
    "motion_latent_checkpoint.pt.tmp",
    "embeddings.npz",
    "diagnostics.json",
    "prototypes.json",
    "progress.jsonl",
    "training_summary.json",
}


def _prepare_output_dir(output_dir, *, overwrite):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"motion latent output directory is not empty: {output_dir}")
        for name in RUN_ARTIFACT_NAMES:
            path = output_dir / name
            if path.is_file():
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _atomic_torch_save(payload, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _cpu_state_dict(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _motion_checkpoint_payload(
    *,
    model_state_dict,
    optimizer,
    stats,
    train_episodes,
    validation_episodes,
    test_episodes,
    action_dim,
    latent_dim,
    hidden_dim,
    steps,
    batch_size,
    lr,
    weight_decay,
    seed,
    deterministic,
    best_step,
    best_validation_loss,
    best_validation_retrieval_loss,
):
    return {
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "normalization": {name: tensor.detach().cpu() for name, tensor in stats.items()},
        "behavior_ids": list(KIMODO_BEHAVIOR_IDS),
        "emotion_ids": list(KIMODO_EMOTION_IDS),
        "config": {
            "action_dim": int(action_dim),
            "latent_dim": int(latent_dim),
            "hidden_dim": int(hidden_dim),
            "steps": int(steps),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "seed": int(seed),
            "deterministic": bool(deterministic),
        },
        "split_episode_indices": {
            "train": [int(episode["episode_index"]) for episode in train_episodes],
            "validation": [int(episode["episode_index"]) for episode in validation_episodes],
            "test": [int(episode["episode_index"]) for episode in test_episodes],
        },
        "best_step": int(best_step),
        "best_validation_loss": float(best_validation_loss),
        "best_validation_retrieval_loss": float(best_validation_retrieval_loss),
    }


def _resolved_device(requested):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_motion_metric_checkpoint(path, *, device="cpu"):
    device = torch.device(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    required = {"action_dim", "latent_dim", "hidden_dim"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"motion latent checkpoint is missing config fields: {missing}")
    model = MotionMetricEncoder(
        action_dim=int(config["action_dim"]),
        latent_dim=int(config["latent_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        behavior_classes=len(checkpoint.get("behavior_ids", KIMODO_BEHAVIOR_IDS)),
        emotion_classes=len(checkpoint.get("emotion_ids", KIMODO_EMOTION_IDS)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    stats = {
        name: torch.as_tensor(value, dtype=torch.float32, device=device)
        for name, value in checkpoint["normalization"].items()
    }
    return model, stats, checkpoint


def train_motion_latent(
    episodes,
    *,
    output_dir,
    steps=20_000,
    batch_size=64,
    lr=3e-4,
    latent_dim=128,
    hidden_dim=128,
    device="auto",
    seed=7,
    log_interval=100,
    prototypes_per_group=1,
    weight_decay=1e-4,
    deterministic=True,
    overwrite=False,
):
    if steps <= 0:
        raise ValueError("steps must be positive")
    if batch_size <= 0 or batch_size % 2:
        raise ValueError("batch_size must be a positive even number")
    if log_interval <= 0:
        raise ValueError("log_interval must be positive")

    output_dir = _prepare_output_dir(output_dir, overwrite=bool(overwrite))
    progress_path = output_dir / "progress.jsonl"
    progress_path.write_text("", encoding="utf-8")
    train_episodes, validation_episodes, test_episodes = stratified_episode_split(episodes, seed=seed)
    validation_reference_episodes = _validation_reference_bank(train_episodes, per_group=2)
    stats = compute_motion_normalization(train_episodes)
    resolved_device = _resolved_device(device)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    torch.manual_seed(int(seed))
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    rng = np.random.default_rng(int(seed))

    model = MotionMetricEncoder(action_dim=train_episodes[0]["actions"].shape[-1], latent_dim=latent_dim, hidden_dim=hidden_dim)
    model.to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    best_validation_loss = float("inf")
    best_validation_retrieval_loss = float("inf")
    best_score = (float("inf"), float("inf"))
    best_step = 0
    best_state = None
    checkpoint_path = output_dir / "motion_latent_checkpoint.pt"

    for step in range(1, int(steps) + 1):
        model.train()
        batch = sample_semantic_pair_batch(train_episodes, batch_size=int(batch_size), rng=rng)
        features, behavior, emotion, descriptors = _batch_tensors(batch, stats, resolved_device)
        losses = motion_metric_loss(model(features), behavior, emotion, descriptors)
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(f"non-finite motion latent loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        should_log = step == 1 or step % int(log_interval) == 0 or step == int(steps)
        if should_log:
            validation_losses = _evaluate_metric_model(
                model,
                validation_episodes,
                stats,
                batch_size=max(int(batch_size), len(validation_episodes)),
                device=resolved_device,
            )
            validation_reference_index = encode_motion_episodes(
                model,
                validation_reference_episodes,
                stats,
                device=resolved_device,
            )
            validation_index = encode_motion_episodes(
                model,
                validation_episodes,
                stats,
                device=resolved_device,
            )
            validation_retrieval_loss = cross_set_retrieval_loss(
                validation_reference_index,
                validation_index,
            )
            score = (float(validation_retrieval_loss), float(validation_losses["total"]))
            is_best = score < best_score
            if is_best:
                best_score = score
                best_validation_retrieval_loss = score[0]
                best_validation_loss = score[1]
                best_step = step
                best_state = _cpu_state_dict(model)
                checkpoint = _motion_checkpoint_payload(
                    model_state_dict=best_state,
                    optimizer=optimizer,
                    stats=stats,
                    train_episodes=train_episodes,
                    validation_episodes=validation_episodes,
                    test_episodes=test_episodes,
                    action_dim=train_episodes[0]["actions"].shape[-1],
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    steps=steps,
                    batch_size=batch_size,
                    lr=lr,
                    weight_decay=weight_decay,
                    seed=seed,
                    deterministic=deterministic,
                    best_step=best_step,
                    best_validation_loss=best_validation_loss,
                    best_validation_retrieval_loss=best_validation_retrieval_loss,
                )
                _atomic_torch_save(checkpoint, checkpoint_path)
            event = {
                "step": step,
                "steps": int(steps),
                "train_loss": float(losses["total"].detach().cpu()),
                "validation_loss": float(validation_losses["total"]),
                "validation_retrieval_loss": float(validation_retrieval_loss),
                "behavior_loss": float(losses["behavior"].detach().cpu()),
                "emotion_loss": float(losses["emotion"].detach().cpu()),
                "contrastive_loss": float(losses["contrastive"].detach().cpu()),
                "descriptor_loss": float(losses["descriptor"].detach().cpu()),
                "is_best": bool(is_best),
            }
            with progress_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, sort_keys=True) + "\n")

    if best_state is None:
        raise RuntimeError("training completed without a finite validation checkpoint")
    model.load_state_dict(best_state)
    model.to(resolved_device)
    indices = {
        "train": encode_motion_episodes(model, train_episodes, stats, device=resolved_device),
        "validation": encode_motion_episodes(model, validation_episodes, stats, device=resolved_device),
        "test": encode_motion_episodes(model, test_episodes, stats, device=resolved_device),
    }
    learned_diagnostics = {
        "validation": compute_latent_diagnostics(indices["train"], indices["validation"]),
        "test": compute_latent_diagnostics(indices["train"], indices["test"]),
    }
    raw_indices = {
        "train": build_raw_motion_index(train_episodes, stats),
        "validation": build_raw_motion_index(validation_episodes, stats),
        "test": build_raw_motion_index(test_episodes, stats),
    }
    raw_diagnostics = {
        "validation": compute_latent_diagnostics(raw_indices["train"], raw_indices["validation"]),
        "test": compute_latent_diagnostics(raw_indices["train"], raw_indices["test"]),
    }
    comparison = {
        "latent_dim": int(indices["train"].embeddings.shape[1]),
        "raw_feature_dim": int(raw_indices["train"].embeddings.shape[1]),
        "compression_ratio": float(raw_indices["train"].embeddings.shape[1] / indices["train"].embeddings.shape[1]),
        "validation_joint_accuracy_retention": float(
            learned_diagnostics["validation"]["knn_joint_accuracy"]
            / max(raw_diagnostics["validation"]["knn_joint_accuracy"], 1e-12)
        ),
        "test_joint_accuracy_retention": float(
            learned_diagnostics["test"]["knn_joint_accuracy"]
            / max(raw_diagnostics["test"]["knn_joint_accuracy"], 1e-12)
        ),
        "test_joint_lift_over_random": float(
            learned_diagnostics["test"]["knn_joint_accuracy"]
            / max(learned_diagnostics["test"]["random_joint_accuracy"], 1e-12)
        ),
    }
    diagnostics = {
        "learned": learned_diagnostics,
        "raw_feature_baseline": raw_diagnostics,
        "comparison": comparison,
    }
    prototypes = select_prototype_medoids(indices["train"], per_group=int(prototypes_per_group))
    _save_combined_latent_indices(output_dir / "embeddings.npz", indices)
    _write_json(output_dir / "diagnostics.json", diagnostics)
    _write_json(output_dir / "prototypes.json", prototypes)

    summary = {
        "output_dir": str(output_dir),
        "episodes": int(len(episodes)),
        "train_episodes": int(len(train_episodes)),
        "validation_episodes": int(len(validation_episodes)),
        "test_episodes": int(len(test_episodes)),
        "semantic_groups": int(len(prototypes)),
        "best_step": int(best_step),
        "best_validation_loss": float(best_validation_loss),
        "best_validation_retrieval_loss": float(best_validation_retrieval_loss),
        "validation_joint_accuracy": learned_diagnostics["validation"]["knn_joint_accuracy"],
        "test_joint_accuracy": learned_diagnostics["test"]["knn_joint_accuracy"],
        "raw_test_joint_accuracy": raw_diagnostics["test"]["knn_joint_accuracy"],
        "test_joint_accuracy_retention": comparison["test_joint_accuracy_retention"],
    }
    _write_json(output_dir / "training_summary.json", summary)
    return summary
