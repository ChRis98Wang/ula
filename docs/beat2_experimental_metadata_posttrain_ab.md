# BEAT2 experimental metadata-conditioned generator B/C

This experiment leaves the clean motion-only A foundation and its style cache
unchanged. It creates two new, explicitly experimental branches:

- `frozen_base`: frozen official Qwen text-alignment cache;
- `lora_finetuned`: BEAT2-only Qwen LoRA text-alignment cache.

The artifacts are not formal semantic supervision and are permanently marked
`formal_release_eligible: false`.

## Bridge layout and isolation

The V3 generator already accepts 264 condition dimensions. Preparation copies
the validated clean style-only cache and creates a separate cache per branch:

- `0:133`: exact zero;
- `133:136`: the original trajectory-derived style, byte-identical in B/C;
- `136:264`: the branch's 128D experimental metadata text latent.

Every clip must match on clip ID, prompt, fixed split, speaker, trajectory hash,
and order. The completed clean foundation hash, style-cache hash, split/style
contract hashes, both source-cache hashes, and optimizer contract are frozen in
`prepared/pair_contract.json`.

The clean A checkpoint/cache are read-only inputs. The bridge never overwrites
them and never changes the formal motion-only validators.

## Trainable-parameter policy

This is not full-network training. Both branches use the exact policy
`zero_latent_preserving_condition_path_only_v1`:

- train `motion_latent_condition.0.weight`;
- train only columns `136:264` of `plan.0.weight`;
- freeze every bias and downstream/backbone parameter;
- mask planner gradients outside `136:264`;
- disable weight decay for the partially masked planner tensor.

The effective trainable count for V3 384×6 is 147,456. This construction makes
the experimental model exactly reproduce A when the 128D latent is zero.
Checkpoint creation fails unless zero-latent output error, frozen-parameter
error, and non-latent planner-column error are all exactly zero.

Both branches run 50,000 steps with the same foundation, seed, sampler schedule,
loss, optimizer, native-length batching, and fixed validation/test IDs.
Early stopping is disabled so the budgets cannot diverge.

## Outputs

```text
training/runs/beat2_experimental_metadata_posttrain_ab_v1/
├── prepared/
│   ├── pair_contract.json
│   ├── conditions_264d_frozen_base.experimental.npz
│   └── conditions_264d_lora_finetuned.experimental.npz
├── frozen_base/
│   ├── generator_experimental.pt
│   ├── last_state.pt
│   ├── progress.jsonl
│   └── training_summary.json
├── lora_finetuned/
│   └── ...
└── comparison.json
```

`last_state.pt` contains model, EMA, optimizer, native sampler, and RNG state.
`--resume` is enabled by default. A restarted branch exact-resumes; a completed
branch validates and returns without retraining. Preparation is idempotent.

The summaries report held-out flow/physical objectives, aligned-versus-zero and
aligned-versus-shuffled condition response, preservation receipts, and parameter
deltas. `comparison.json` additionally reports B/C cache cross-swap response.
These response checks prove that the mechanism consumes the latent; they do not
establish formal semantic quality.

## Manual commands

Preparation waits logically on two completed upstream products:

1. the clean v7 `training_summary.json` and final best checkpoint;
2. the full frozen/LoRA 128D Qwen condition caches.

Then run:

```bash
python_bin=/home/gez/shuaiwang/.venvs/gmr/bin/python
config=configs/beat2_experimental_metadata_posttrain_ab_v1.json

"${python_bin}" tools/train_beat2_experimental_metadata_posttrain_ab.py \
  --config "${config}" --stage prepare

mkdir -p training/runs/beat2_experimental_metadata_posttrain_ab_v1/logs

"${python_bin}" tools/train_beat2_experimental_metadata_posttrain_ab.py \
  --config "${config}" --stage frozen --resume \
  > training/runs/beat2_experimental_metadata_posttrain_ab_v1/logs/frozen.log 2>&1 &
frozen_pid=$!

"${python_bin}" tools/train_beat2_experimental_metadata_posttrain_ab.py \
  --config "${config}" --stage lora --resume \
  > training/runs/beat2_experimental_metadata_posttrain_ab_v1/logs/lora.log 2>&1 &
lora_pid=$!

wait "${frozen_pid}"
wait "${lora_pid}"

"${python_bin}" tools/train_beat2_experimental_metadata_posttrain_ab.py \
  --config "${config}" --stage compare
```

For one-GPU sequential execution, run `frozen` and then `lora`.

## Recoverable nightly launcher

`tools/run_beat2_experimental_metadata_posttrain_ab_nightly.sh` waits until the
completed A summary and both full Qwen caches exist, prepares once, launches B/C
in parallel, and compares them. Set
`BEAT2_POSTTRAIN_RUN_MODE=sequential` if desired.

The supplied systemd service uses `Restart=on-failure`; restart is safe because
the preparation and branch stages are idempotent and branch state is exact:

```bash
mkdir -p "${HOME}/.config/systemd/user"
ln -sfn \
  /home/gez/shuaiwang/ula-motion-generate/deploy/systemd/beat2-experimental-metadata-posttrain-ab.service \
  "${HOME}/.config/systemd/user/beat2-experimental-metadata-posttrain-ab.service"
systemctl --user daemon-reload
systemctl --user enable --now beat2-experimental-metadata-posttrain-ab.service
```

Do not use `--overwrite` for recovery. It is only for an explicit clean restart.

## Smoke test

Use a stable trained-A snapshot and the Qwen smoke caches:

```bash
/home/gez/shuaiwang/.venvs/gmr/bin/python \
  tools/train_beat2_experimental_metadata_posttrain_ab.py \
  --config configs/beat2_experimental_metadata_posttrain_ab_v1.json \
  --stage all --smoke-test --device cuda \
  --output-dir /tmp/beat2_experimental_metadata_posttrain_smoke \
  --foundation-checkpoint /path/to/stable/clean-A-snapshot.pt \
  --foundation-training-summary /tmp/not-required-for-smoke.json \
  --frozen-condition-cache /path/to/conditions_128d_frozen_base.npz \
  --lora-condition-cache /path/to/conditions_128d_lora_finetuned.npz
```

The smoke-only summary waiver permits a clean trained snapshot without its
completion summary. Production preparation always requires and verifies the
completed summary.
