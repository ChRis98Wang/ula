# Motion Latent And Prototype Retrieval Design

## Goal

Build a small-data motion representation that can be measured before it is used to condition ULA Flow. The first delivery trains a compact motion encoder, reports whether its latent space separates Kimodo semantics, and exports representative real-motion prototypes. It does not fine-tune Qwen/GLM and does not replace joint-space Flow.

## Data Evidence

Kimodo contains 1,620 unique 150-frame episodes across 162 `behavior_id + emotion_id` groups, with exactly 10 episodes per group. Pairwise trajectory RMS distances overlap substantially: the within-group median is `0.339` rad and the between-group median is `0.490` rad. Labels alone therefore cannot be assumed to define a clean motion space.

Use a deterministic per-group `8/1/1` train/validation/test split. Normalization statistics must be computed from the train split only. This evaluates generalization to held-out trajectories for every known semantic combination. Free-text generalization is a later, separate split over prompt paraphrases.

## Architecture

The encoder consumes normalized position and first-difference velocity channels. A small temporal Conv1d network pools variable-length motion into an L2-normalized 128-dimensional embedding. Auxiliary heads predict 27 behaviors, 6 emotions, and continuous kinematic descriptors so the embedding preserves both semantic identity and style variation.

Training uses paired semantic batches and four losses:

- behavior cross entropy;
- emotion cross entropy;
- supervised contrastive loss using the joint behavior/emotion label;
- Smooth L1 descriptor regression for amplitude, velocity, acceleration, left/right activity, and asymmetry.

Because the `8/1/1` split leaves only one validation trajectory per joint semantic class, ordinary within-validation supervised contrastive loss has no positive pairs. Checkpoint selection therefore uses a fixed train reference bank with two examples per semantic group and a cross-set retrieval loss for each validation query. Auxiliary validation loss is only the tie-breaker. Every improved checkpoint is written atomically before diagnostics, and a non-empty output directory is rejected unless overwrite is explicit.

The default generator remains the existing AdaLN Flow over `[T, 15]` joint trajectories. The exported latent index selects class medoids as real-motion prototypes. Prototype conditioning will only be added to Flow after held-out diagnostics show useful separation.

## Diagnostics And Acceptance

The artifact report includes held-out nearest-neighbor accuracy for behavior, emotion, and their joint label; intra/inter-class cosine distances; separate reference/query effective ranks; and selected prototype episode IDs. It also computes the same metrics from downsampled normalized position/velocity features. This raw-feature baseline prevents a compressed latent from being accepted merely because it beats random guessing. Artifacts contain the encoder checkpoint, optimizer state, normalization statistics, embeddings with labels, and JSON diagnostics. The training CLI streams only required Parquet columns and enables deterministic PyTorch algorithms by default.

The first-stage acceptance gate is deliberately empirical:

- joint-label nearest-neighbor accuracy must beat random by a large margin;
- the 128-dimensional latent should retain at least 80% of the raw-feature nearest-neighbor accuracy before Flow integration;
- behavior accuracy must exceed emotion accuracy if motion identity dominates style;
- inter-class distance must exceed intra-class distance;
- both reference and held-out query effective ranks must remain well above one, ruling out split-specific collapse;
- every semantic group must have a prototype and no test episode may enter the prototype pool.

If this gate fails, improve labels or motion augmentation before integrating the latent into Flow. If it passes, stage two appends a projected prototype latent to the AdaLN condition and trains with a different same-class prototype for each target episode. Multiple prototypes are selected with greedy k-medoids so they minimize group assignment distance instead of simply choosing farthest outliers. Stage three freezes Qwen/GLM and trains a semantic adapter to predict the same structured condition.
