# Qwen Semantic Adapter

The semantic adapter maps free-form motion instructions to the structured Kimodo labels used by the 136-dimensional ULA condition vector. It is separate from the motion generator: Qwen classifies the instruction, while the ULA checkpoint generates the joint trajectory.

## Scope

The current adapter predicts only:

- `behavior_id`: one of the 27 Kimodo behavior labels;
- `emotion_id`: one of `neutral`, `sad`, `happy`, `angry`, `surprise`, or `fear`.

After classification, inference looks up the exact canonical 136-dimensional condition used by the Kimodo generator training dataset for the predicted `(behavior_id, emotion_id)` pair. It does not hash the raw Chinese text again. The adapter does not yet predict independent continuous style controls; those values remain the canonical values for that label pair.

The adapter is supported only with a Kimodo generator whose `condition_dim` is `136`. It cannot be used with the legacy 92-dimensional ULA checkpoint. Explicit `--behavior-id` or `--emotion-id` values override the corresponding adapter prediction.

## Model

The default frozen encoder is `Qwen/Qwen3-Embedding-0.6B`, pinned in the checked-in config to a specific model revision. Training performs the following steps:

1. Format each prompt as an embedding instruction and query.
2. Run Qwen in evaluation and inference mode with all Qwen parameters frozen.
3. Encode the text twice with one shared Qwen instance: one instruction isolates the physical action and one isolates the explicitly requested emotion.
4. Last-token pool each result, keep 128 dimensions from each, L2-normalize, and concatenate them into 256 dimensions.
5. Train independent action and emotion adapter branches with 27-class behavior and 6-class emotion heads. Qwen remains frozen.

The adapter checkpoint contains the small trainable network, pinned Qwen metadata, and a `27 x 6 x 136` canonical condition bank with the source dataset hash. It does not contain the 0.6B Qwen weights. Loading it therefore needs either network access to Hugging Face or an already populated local model cache.

Install the optional dependencies in the isolated environment:

```bash
cd /home/gez/shuaiwang/ula-motion-generate

conda run --no-capture-output -n env_isaaclab \
  python -m pip install -r requirements-semantic-adapter.txt
```

## Data Splits

The prompt catalog must contain the complete `27 behaviors x 6 emotions = 162` Kimodo grid with exactly one prompt per label pair. It is not split by the ten repeated motion episodes.

For behavior index `i` and fold `f`:

- test emotion: `(i + f) % 6`;
- validation emotion: `(i + f + 1) % 6`;
- the other four emotions: training.

With the Chinese paraphrase asset, each Latin fold contains 540 training texts, 54 validation texts, and 54 test texts. Changing `fold` from 0 through 5 rotates every behavior/emotion pair through the test partition. This mode measures held-out behavior/emotion composition and unseen Chinese wording.

The default config uses deployment mode. It trains all 162 legal label pairs using 810 English and Chinese texts, then evaluates 162 validation and 162 test prompts built from separate Chinese wording pools. This measures seen-combination paraphrase robustness and produces the checkpoint intended for interactive playback.

## Training

The default config trains the deployment adapter with frozen Qwen, dual task instructions, and early stopping:

```bash
cd /home/gez/shuaiwang/ula-motion-generate

conda run --no-capture-output -n env_isaaclab \
  python -u -m training.scripts.train_semantic_adapter \
  --config configs/train_kimodo_semantic_adapter.yaml \
  --overwrite
```

The first run may download `Qwen3-Embedding-0.6B`. To require an existing local cache, add `--local-files-only`.

The default output directory is:

```text
training/runs/kimodo_qwen3_semantic_adapter_deploy_v1/
```

It contains:

- `semantic_adapter_checkpoint.pt`: best adapter weights, pinned Qwen metadata, and canonical condition bank;
- `metrics.json`: best validation and held-out test metrics;
- `progress.jsonl`: training and validation history;
- `split_manifest.json`: exact prompt assignments for the configured split mode.

Run the stricter Latin composition holdout separately:

```bash
conda run --no-capture-output -n env_isaaclab \
  python -u -m training.scripts.train_semantic_adapter \
  --config configs/train_kimodo_semantic_adapter.yaml \
  --split-mode latin \
  --fold 1 \
  --output-dir training/runs/kimodo_qwen3_semantic_adapter_eval_fold1
```

The completed fold-0 evaluation reached 88.89% joint accuracy on its 27 held-out Chinese composition examples (92.59% behavior and 96.30% emotion); including the 27 canonical English examples, joint accuracy was 94.44%. The deployment checkpoint reached 91.98% joint accuracy on 162 separate Chinese test prompts (93.21% behavior and 98.77% emotion). These are semantic classification scores, not motion-quality scores.

## Headless Prediction

Inspect behavior/emotion predictions and confidence scores without loading a motion generator or MuJoCo:

```bash
conda run --no-capture-output -n env_isaaclab \
  python -u -m upper_body_skeleton.semantic_adapter \
  --checkpoint training/runs/kimodo_qwen3_semantic_adapter_deploy_v1/semantic_adapter_checkpoint.pt \
  --text "开心地跟主人点头并在胸前挥手打招呼" \
  --device cuda \
  --local-files-only
```

`--text` may be repeated to classify several prompts in one Qwen load. Add `--local-files-only` after the pinned model revision has been cached.

To run the complete semantic-adapter-to-generator path without opening a viewer:

```bash
conda run --no-capture-output -n env_isaaclab \
  python -u -m upper_body_skeleton.pt_mujoco_infer \
  --kimodo-qwen \
  --text "开心地跟主人点头并在胸前挥手打招呼" \
  --device cuda \
  --semantic-device cuda \
  --semantic-local-files-only \
  --no-viewer
```

This command keeps the generated `[frames, 15]` trajectory in memory and does not export CSV or NPZ files.

## MuJoCo Interaction

Run from a graphical desktop, VNC session, or an SSH session with trusted X11 forwarding:

```bash
ssh -Y gez@172.16.60.184
cd /home/gez/shuaiwang/ula-motion-generate
echo "$DISPLAY"

conda run --no-capture-output -n env_isaaclab \
  python -u -m upper_body_skeleton.pt_mujoco_infer \
  --kimodo-qwen \
  --device cuda \
  --semantic-device cuda \
  --semantic-local-files-only \
  --loops 1
```

Enter one motion instruction per line. The adapter predicts behavior and emotion, the generator produces an in-memory trajectory, and the existing MuJoCo player plays it immediately. Enter `:q` to exit.

Use enough motion detail to distinguish nearby labels. For example, `开心地挥手` is compatible with several owner-greeting variants and therefore has low behavior confidence; `开心地跟主人点头并在胸前挥手打招呼` identifies `Behavior.GreetingOwner01` clearly. Each playback summary includes both predicted IDs and confidence values.

## Generator Limitation

The repository's default legacy generator is a 92-dimensional checkpoint and is incompatible with this adapter. The
`--kimodo-qwen` shortcut selects both the deployment semantic adapter and the bundled compatible Kimodo MMDiT
checkpoint. That generator uses action normalization, an exact condition contract, and the math SDPA backend. It
completed 5,000 steps with finite loss and gradients. On a fixed 162-episode training-reference evaluation, loss
improved from `0.58138` at step 1,000 to `0.40417` at step 5,000.

To evaluate another 136-dimensional Kimodo generator, replace the shortcut with its checkpoint and the adapter path:

```bash
conda run --no-capture-output -n env_isaaclab \
  python -u -m upper_body_skeleton.pt_mujoco_infer \
  --checkpoint /path/to/completed_kimodo_generator.pt \
  --semantic-adapter-checkpoint training/runs/kimodo_qwen3_semantic_adapter_deploy_v1/semantic_adapter_checkpoint.pt \
  --device cuda \
  --semantic-device cuda \
  --loops 1
```

Do not treat semantic classification accuracy or the training-reference generator loss as independent test-set motion
quality. The adapter and generator still need a held-out generation evaluation, followed by qualitative testing in
MuJoCo.

At startup, the runtime verifies that the condition bank label order and source `semantic_index.parquet` hash match the generator training dataset. A custom generator checkpoint must therefore record an accessible `config.dataset_dir`. A different Qwen model or revision is also rejected unless the explicit compatibility-experiment flag is provided.
