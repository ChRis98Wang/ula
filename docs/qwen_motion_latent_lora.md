# Qwen Motion Latent LoRA Training

This experiment trains a differentiable text-to-motion latent alignment model. It does not use the frozen embedding
cache from the deployment semantic classifier.

## Architecture

```text
Chinese or English motion text
  -> pinned Qwen3-Embedding-0.6B
  -> top-four-layer q/k/v/o LoRA (rank 8)
  -> dual action/emotion pooling
  -> text projection
  -> normalized 128-dimensional text latent

150 x 15 joint trajectory
  -> train-only position/velocity normalization
  -> MotionMetricEncoder final block and projection
  -> motion projection
  -> normalized 128-dimensional motion latent
```

The frozen initial MotionMetricEncoder is retained as an anchor teacher. Training combines bidirectional
text-motion retrieval, text behavior/emotion classification, the original motion metric losses, teacher anchoring,
and VICReg variance/covariance regularization.

The split is inherited from the original MotionMetric checkpoint:

- Train: 1,296 motion episodes and 810 independent text prompts.
- Validation: 162 motion episodes and 162 unseen Chinese paraphrases.
- Test: 162 motion episodes and 162 separate unseen Chinese paraphrases.

Episode IDs and normalized text hashes are checked for pairwise disjointness before training. Test data is not used
for checkpoint selection.

## Active Run

Configuration: `configs/train_kimodo_qwen_motion_latent_lora.yaml`

Output: `training/runs/kimodo_qwen_motion_latent_lora_v1`

Log: `/home/gez/shuaiwang/qwen_motion_latent_lora_v1.log`

The long run is managed by the user systemd service `ula-qwen-motion-latent-v1.service` so it survives SSH session
termination.

```bash
systemctl --user status ula-qwen-motion-latent-v1.service
tail -f /home/gez/shuaiwang/qwen_motion_latent_lora_v1.log
nvidia-smi
```

Artifacts:

- `last.pt`: current optimizer, LoRA, motion encoder, projections, sampler, and RNG state for resuming.
- `best.pt`: best bidirectional validation Recall@1/5, with cosine gap and retrieval loss as tie-breakers, after the
  effective-rank collapse gate passes.
- `split_manifest.json`: immutable episode IDs, text hashes, counts, and source hashes.
- `progress.jsonl`: training and validation metrics.

## Resume

The target `steps` value is the total global step, not an additional step count. Resume from `last.pt` using the same
model, optimizer, data, and split configuration:

```bash
conda run --no-capture-output -n env_isaaclab \
  python -u -m training.scripts.train_qwen_motion_latent \
  --config configs/train_kimodo_qwen_motion_latent_lora.yaml \
  --resume-from training/runs/kimodo_qwen_motion_latent_lora_v1/last.pt
```

The loader rejects mismatched Qwen revisions, LoRA structure, source hashes, latent dimensions, optimizer settings,
or batch size. The 0.6B frozen base model is referenced by its pinned revision and is not duplicated in each
checkpoint.

## Scope

This run improves and measures the shared text-motion representation. It does not automatically replace the current
136-dimensional condition contract used by `pt_mujoco_infer`. The aligned latent must first pass held-out retrieval
and collapse diagnostics, then a generator must be trained or adapted to consume that latent before deployment.
