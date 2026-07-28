# BEAT2-only Qwen text-to-motion-latent A/B

This experiment starts from two allowed sources only:

1. the adjudicated BEAT2 min-30-frame manifest and its 18-DoF CSVs;
2. the official, commit-pinned `Qwen/Qwen3-Embedding-0.6B` base.

It never accepts an existing motion encoder, text adapter, prompt catalog, or
generator checkpoint as an alignment-training input.

## Controlled branches

- **B / `frozen_base`**: official Qwen is frozen; a newly initialized projector
  learns text to the 128D BEAT2 motion space.
- **C / `lora_finetuned`**: the same official base gets a BEAT2-only LoRA; an
  identically initialized projector uses the same data, loss, learning rate,
  and number of steps.

The motion space is learned first by a randomly initialized BEAT2-only encoder.
Its primary continuous objective reconstructs train-normalized temporal
position/velocity/acceleration descriptors. Category, intensity, emotion, and
joint-group heads are auxiliary; this is not merely a 54-way classifier.

Each branch exports a direct `[episode, 128]` condition cache:

- `conditions_128d_frozen_base.npz`
- `conditions_128d_lora_finetuned.npz`

Their clip, prompt, split, speaker, semantic-group, and trajectory-hash arrays
must be byte-identical in ordering. The only changed values are the predicted
128D text-to-motion latents. These are deliberately not legacy 264D condition
caches: a clean latent-to-motion prior must consume the 128D vector directly or
place it in a dedicated 128D input slice.

## Split and interpretation limits

The script fails unless the locked full release retains exactly 17 train,
4 validation, and 4 test speakers with pairwise-disjoint speakers and source
groups. Current episode counts are 7,522 / 1,629 / 2,988.

The official metadata yields only 54 deterministic canonical prompts—one for
each category/intensity/emotion combination. Canonical retrieval therefore
does not establish open-text generalization. Each branch additionally reports
a deterministic, unseen-wording template probe, clearly labeled diagnostic and
non-formal. Speech-context fields are not loaded as motion instructions.

The release masks prompt supervision. Outputs therefore carry the explicit
scope `experimental_official_metadata_alignment_only_not_formal_generator_supervision`;
they must not be presented as formally reviewed robot semantics.

## Commands

Fast end-to-end smoke test:

```bash
/home/gez/shuaiwang/.venvs/gmr/bin/python \
  tools/train_beat2_qwen_motion_alignment.py \
  --config configs/beat2_qwen_motion_alignment_ab_v1.yaml \
  --stage all --smoke-test --device cuda --overwrite
```

Full B/C alignment:

```bash
/home/gez/shuaiwang/.venvs/gmr/bin/python \
  tools/train_beat2_qwen_motion_alignment.py \
  --config configs/beat2_qwen_motion_alignment_ab_v1.yaml \
  --stage all --device cuda
```

The run is staged (`audit`, `cache`, `motion`, `frozen`, `lora`, `compare`) so a
completed artifact is reused after validation. `comparison.json` contains
held-out motion/text alignment deltas and the paired-generator contract.

After a clean BEAT2 generator foundation exists, set
`generator_pairing.foundation_checkpoint` to that checkpoint and rerun
`--stage compare`. Both generator post-trains must then use that exact hash,
the recorded common seed, and the same step/optimizer budget; only the B or C
condition cache may differ.
