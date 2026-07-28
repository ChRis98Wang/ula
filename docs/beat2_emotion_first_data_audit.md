# BEAT2 emotion-first data audit

Audit date: 2026-07-27 (Asia/Shanghai)

## Decision

BEAT2 can support an **experimental emotion-first curriculum**, but its
filename emotion is an **intended performance label**, not a perceived
robot-emotion truth label.  The current manifest correctly keeps all emotion
supervision disabled.  It is therefore safe to use the label for a weak,
clearly named experiment, but unsafe to claim that a generated robot motion is
perceived as that emotion until blind human review has passed.

The best immediate curriculum is hierarchical:

1. learn neutral versus non-neutral from source-group-balanced BEAT2;
2. learn six emotion prototypes while treating gesture category and gesture
   intensity as nuisance factors;
3. only then add the 54
   emotion × category × intensity prompt groups;
4. calibrate/fine-tune against the independent blind robot-observability
   review set when its reviews are complete.

Starting with a uniformly sampled 54-way objective is not recommended.  Five
training cells contain fewer than five events and three cells are singletons.

## Audited artifacts and label semantics

- Manifest:
  `/home/gez/nas/cloud/gez/human_motion/processed/beat2_semantic_event_training_pool_18d_v7_full/adjudication_min30f/train_ready.jsonl`
- Manifest SHA-256:
  `2b3692c4f0a9ea8e10f3bde74fa178800556ed9bb79d0918a770b963a7f7c7fd`
- Rows: 12,139, all `dataset == "BEAT2"`.
- The official emotion source is
  `official_beat2_filename_protocol` on all 12,139 rows.  It describes the
  actor's intended clip-level performance condition.
- `source_emotion_label_verified == true` on all rows means that the filename
  was decoded according to the official protocol.  It does **not** mean that a
  viewer perceived the retargeted robot motion as that emotion.
- All 12,139 rows have each of the following disabled:
  `emotion_supervision_mask`, `emotion_conditioning_mask`,
  `affect_observable_supervision_mask`, and
  `official_emotion_conditioning_enabled`.
- The deterministic prompt contains the official intended emotion and the
  official gesture category/intensity, but
  `prompt_text_supervision_mask == false`.  There are 54 such prompt groups.
- Gesture category and intensity come from official semantic-event
  annotations.  They are useful metadata, but they are not perceived-emotion
  annotations or emotion intensity.
- Recursive inspection found zero `kimodo` strings in the manifest.  All
  source paths are under BEAT2.  The associated style cache declares
  `kimodo_policy =
  forbidden_dataset_checkpoint_replay_and_condition_channels_v1`.

## Scale and imbalance

Event duration is the retargeted 30 Hz output duration.  Source duration is
the sum of the original source intervals.  The 12,139 event windows do not
overlap one another within a source recording.

| Intended emotion | Events | Share | Output hours | Source hours | Independent source recordings |
|---|---:|---:|---:|---:|---:|
| neutral | 9,615 | 79.21% | 5.968 | 5.461 | 871 |
| sad | 326 | 2.69% | 0.192 | 0.178 | 78 |
| happy | 633 | 5.21% | 0.383 | 0.345 | 101 |
| angry | 516 | 4.25% | 0.310 | 0.276 | 100 |
| surprise | 620 | 5.11% | 0.364 | 0.335 | 103 |
| fear | 429 | 3.53% | 0.249 | 0.232 | 98 |
| **total** | **12,139** | **100%** | **7.466** | **6.829** | **1,351** |

The effective non-neutral scale is therefore only 480 independent recordings
in the full data and 292 in training, not 2,524 independent examples.

### Fixed split distribution

The split is speaker-disjoint.

| Split | Speakers | Events | Sources | Output h | Source h | neutral | sad | happy | angry | surprise | fear |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 17 | 7,522 | 865 | 4.606 | 4.236 | 6,055 | 174 | 398 | 305 | 384 | 206 |
| validation | 4 | 1,629 | 174 | 1.019 | 0.892 | 1,227 | 74 | 98 | 79 | 72 | 79 |
| test | 4 | 2,988 | 312 | 1.841 | 1.700 | 2,333 | 78 | 137 | 132 | 164 | 144 |

Independent source counts by intended emotion:

| Split | neutral | sad | happy | angry | surprise | fear |
|---|---:|---:|---:|---:|---:|---:|
| train | 573 | 47 | 61 | 61 | 64 | 59 |
| validation | 107 | 12 | 14 | 13 | 14 | 14 |
| test | 191 | 19 | 26 | 26 | 25 | 25 |

### Duration

Most semantic-event rows are short.  Counts use output duration:

| Intended emotion | under 3 s | 3–6 s | over 6 s | median s | p10–p90 s |
|---|---:|---:|---:|---:|---:|
| neutral | 8,403 | 1,146 | 66 | 2.10 | 1.40–3.13 |
| sad | 301 | 25 | 0 | 2.07 | 1.38–2.87 |
| happy | 557 | 75 | 1 | 2.07 | 1.34–3.13 |
| angry | 463 | 53 | 0 | 2.08 | 1.42–3.00 |
| surprise | 563 | 57 | 0 | 2.00 | 1.40–2.97 |
| fear | 397 | 32 | 0 | 2.03 | 1.43–2.80 |

The main event manifest is unsuitable as direct 60-second emotion
supervision.  Long interaction must be composed from native clips, or use
separately reviewed natural expression turns; it must not be fabricated by
time-stretching these two-second events.

## Kinematic emotion signal

The audited style cache computes:

- amplitude: RMS temporal arm deviation in radians;
- speed: RMS arm velocity in radians/second;
- balance: signed right-versus-left arm activity.

The values below are medians with p10–p90 in parentheses.

| Intended emotion | Arm amplitude rad | Arm speed rad/s | Signed balance |
|---|---:|---:|---:|
| neutral | 0.202 (0.097–0.328) | 0.793 (0.466–1.086) | 0.050 (-0.313–0.594) |
| sad | 0.181 (0.087–0.322) | 0.694 (0.427–0.995) | 0.034 (-0.415–0.613) |
| happy | 0.217 (0.122–0.348) | 0.835 (0.543–1.146) | 0.020 (-0.250–0.508) |
| angry | 0.203 (0.104–0.339) | 0.805 (0.484–1.099) | 0.037 (-0.580–0.663) |
| surprise | 0.202 (0.100–0.325) | 0.795 (0.487–1.116) | 0.011 (-0.347–0.444) |
| fear | 0.187 (0.085–0.311) | 0.730 (0.382–0.978) | 0.026 (-0.466–0.482) |

After accounting for speaker, gesture category, and gesture intensity with an
additive least-squares audit, intended emotion explains only:

- 0.25% of residual arm-balance variance;
- 0.57% of residual log-amplitude variance;
- 1.40% of residual log-speed variance;
- 0.26% of residual duration variance.

Thus amplitude and speed are useful style targets, but they are not adequate
emotion truth.  A train-prototype nearest-centroid diagnostic on the complete
742D motion descriptor obtained 17.9% balanced accuracy on validation and
27.9% on test for the six intended labels (chance is 16.7%).  Results vary
strongly by held-out speaker, so the present motion signal is weak and not yet
a reliable general emotion classifier.

## Speaker distribution and leakage audit

Every speaker belongs to exactly one split.  Validation and test contain all
six intended emotions for every speaker.  Training has 99 of 102 possible
speaker × emotion cells; the only missing cells are:

- `23_hailing / angry`;
- `5_stewart / sad`;
- `5_stewart / happy`.

Per-speaker event counts are ordered
`neutral, sad, happy, angry, surprise, fear`:

| Speaker | Split | Events | Source h | Counts |
|---|---|---:|---:|---|
| 10_kieks | test | 80 | 0.051 | 49, 5, 7, 10, 4, 5 |
| 11_nidal | train | 870 | 0.487 | 701, 14, 47, 38, 55, 15 |
| 12_zhao | train | 204 | 0.114 | 171, 9, 3, 7, 11, 3 |
| 13_lu | train | 208 | 0.107 | 170, 8, 9, 6, 4, 11 |
| 15_carlos | train | 293 | 0.160 | 221, 16, 18, 16, 13, 9 |
| 16_jorge | train | 187 | 0.096 | 144, 6, 23, 2, 11, 1 |
| 17_itoi | train | 179 | 0.100 | 141, 8, 2, 3, 14, 11 |
| 18_daiki | validation | 47 | 0.025 | 31, 4, 5, 1, 3, 3 |
| 1_wayne | test | 904 | 0.526 | 761, 26, 39, 19, 19, 40 |
| 20_li | train | 225 | 0.130 | 151, 19, 17, 10, 21, 7 |
| 21_ayana | train | 849 | 0.549 | 724, 20, 25, 37, 20, 23 |
| 22_luqi | validation | 226 | 0.122 | 111, 29, 31, 11, 24, 20 |
| 23_hailing | train | 272 | 0.137 | 211, 21, 11, 0, 22, 7 |
| 24_kexin | train | 287 | 0.157 | 248, 4, 12, 12, 9, 2 |
| 25_goto | train | 146 | 0.079 | 104, 4, 11, 11, 6, 10 |
| 27_yingqing | validation | 384 | 0.211 | 289, 16, 11, 30, 13, 25 |
| 28_tiffnay | train | 154 | 0.084 | 124, 1, 6, 7, 2, 14 |
| 2_scott | train | 1,353 | 0.785 | 1,110, 12, 81, 60, 73, 17 |
| 30_katya | train | 212 | 0.135 | 151, 14, 25, 14, 5, 3 |
| 3_solomon | train | 1,225 | 0.671 | 999, 11, 57, 52, 58, 48 |
| 4_lawrence | test | 1,429 | 0.765 | 1,144, 10, 40, 56, 107, 72 |
| 5_stewart | train | 186 | 0.093 | 179, 0, 0, 2, 4, 1 |
| 6_carla | test | 575 | 0.358 | 379, 37, 51, 47, 34, 27 |
| 7_sophie | train | 672 | 0.353 | 506, 7, 51, 28, 56, 24 |
| 9_miranda | validation | 972 | 0.534 | 796, 25, 51, 37, 32, 31 |

Cross-split conflicts are zero for each of:

- speaker key;
- source group/source clip;
- source trajectory SHA-256;
- retargeted trajectory SHA-256;
- clip ID and task ID;
- source time interval overlap.

All 12,139 retargeted trajectory hashes are unique.  Source recordings contain
multiple semantic events, so batches and weights must be selected by source
group first; treating all 12,139 rows as independent would overstate the
effective data size.

## 54-way cross sparsity

The three official gesture categories are `deictic`, `iconic`, and
`metaphoric`; intensities are `low`, `medium`, and `high`.  All 54
emotion × category × intensity cells are present in train and validation, but
test is missing one cell.  Sparsity:

| Split | Present | <5 events | <10 events | <20 events | Min / median / max |
|---|---:|---:|---:|---:|---:|
| train | 54/54 | 5 | 8 | 16 | 1 / 37.5 / 1,154 |
| validation | 54/54 | 11 | 31 | 43 | 1 / 9 / 274 |
| test | 53/54 | 6 | 15 | 33 | 1 / 18 / 508 |

Training counts are listed as
`deictic-low/medium/high; iconic-low/medium/high;
metaphoric-low/medium/high`:

- neutral: `187,494,651; 428,1154,1088; 454,1048,551`
- sad: `1,11,24; 5,39,37; 8,39,10`
- happy: `13,36,38; 40,85,40; 52,76,18`
- angry: `1,18,40; 22,60,38; 21,74,31`
- surprise: `2,16,32; 37,87,69; 43,65,33`
- fear: `2,10,25; 5,50,70; 1,25,18`

Uniformly oversampling these cells would repeatedly present single source
motions as if they were a population.  In train, 14 cells have fewer than ten
independent source recordings.

## Immediately usable BEAT2-only curriculum

This curriculum must remain explicitly experimental until human review
converts observable clips into accepted supervision.

### Stage 0 — immutable data and sampling contract

- Lock the manifest SHA above and reject every non-BEAT2 record, path, cache,
  checkpoint, statistic, or replay source.
- Preserve the fixed 17/4/4 speaker split.
- Sample `emotion -> speaker -> source_group -> one event`, not rows directly.
- Downsample neutral at the source-group level; do not inflate rare cells with
  unbounded inverse-frequency weights.
- Keep the current clean AdaLN BEAT2 foundation.  Data scarcity is not a reason
  to revert AdaLN.

### Stage 1 — neutral versus non-neutral weak curriculum

- Use official filename emotion only as
  `experimental_intended_emotion_metadata`; do not flip the formal manifest
  supervision masks.
- Balance 573 neutral versus 292 non-neutral train source recordings through
  source-group sampling.
- Match hard pairs by speaker, gesture category, gesture intensity, and
  duration band so that the model cannot solve the task from speaker or gesture
  metadata alone.
- Train the emotion adapter/critic and AdaLN condition gates first while
  preserving the foundation flow and jerk losses.
- Require stable speaker-disjoint validation and test gains.  The current
  descriptor diagnostic is inconsistent across the two held-out sets, so
  training loss alone is not an acceptance gate.

### Stage 2 — six intended-emotion prototypes

- Average the nine Qwen
  category × intensity prompts into six separate intended-emotion prototypes
  for the primary contrastive loss.
- Treat different categories/intensities of the same emotion as positives.
- Treat the same category/intensity with different emotions as hard negatives.
- Use high and medium strata first.  Add low-intensity cells only after the
  six-way held-out metric improves; the rare low cells must never control the
  sampler.
- Use same-noise counterfactual generation for each emotion and explicitly
  preserve motion validity, response magnitude, and jerk.

This is preferable to making all 54 Qwen prompt groups equally important at
step zero: the latter mostly learns prompt-group identity and amplifies sparse
cells rather than isolating emotion.

### Stage 3 — full prompt hierarchy and Qwen A/B

- Add the 54-way semantic group objective as a secondary loss after Stage 2.
- Run frozen-Qwen and Qwen-LoRA variants from the same Stage-2 checkpoint,
  split, sampler order, noise seeds, and training budget.
- A Qwen fine-tune is accepted only if it improves speaker-disjoint
  motion-based emotion metrics and blind perceived-emotion ratings, without
  worsening jerk or motion amplitude.  Text-side separation by itself is not
  evidence that the motion learned emotion.

### Stage 4 — perceived robot emotion

The existing blind-review queue is the correct path to actual emotion
supervision:

- 598 source-group-unique natural expression turns;
- train/validation/test = 338/92/168;
- two independent blinded reviewers, with a distinct third adjudicator on any
  disagreement;
- source emotion is hidden from reviewers;
- observable clips require agreement on the perceived emotion;
- `ambiguous` and `not_observable` clips stay out of emotion loss.

At audit time the queue has zero accepted training labels and only a smoke
video has been rendered.  Therefore it is a ready review protocol, not yet a
supervised emotion dataset.  Once reviewed, use it to calibrate the weak
filename-label critic and fine-tune only the condition/emotion path before
considering a wider generator update.

## Acceptance criteria

An emotion-first run should not be described as successful unless all of these
hold:

1. no non-BEAT2 or Kimodo lineage anywhere in data, statistics, cache,
   checkpoint, or replay;
2. speaker/source-disjoint validation and test both improve, rather than only
   one held-out speaker set;
3. emotion accuracy is computed from generated motion, not Qwen text
   embeddings;
4. response magnitude does not collapse and jerk does not regress;
5. blind reviewers perceive the requested emotion above a neutral/chance
   baseline;
6. intended-label metrics and perceived-label metrics are reported
   separately.
