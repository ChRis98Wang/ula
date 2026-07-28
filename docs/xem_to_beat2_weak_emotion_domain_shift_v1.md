# XEM → BEAT2 weak-emotion domain-shift diagnostic v1

## Decision

**Fail closed.** The XEM classifier is not suitable as a BEAT2/robot emotion
training gate, pseudolabel source, or automatic screening model.

It remains isolated research material. It was not connected to the motion
foundation, generator, Qwen conditioning path, or any production trainer.
No Kimodo input, cache, latent, split, checkpoint, statistic, or weight was
read.

## Why this experiment is intentionally narrow

There is no validated XEM-segment/Euler-to-robot mapping in this repository.
The diagnostic therefore does not invent one:

- XEM input: norms of the official 23 joint angular-velocity vectors.
- BEAT2 input: absolute finite-difference velocities of the 18 controller
  angles.
- Common representation: 41 anonymous statistics of angular activity,
  including amplitude distribution, entity participation, normalized temporal
  burst shape, normalized spectrum, and anonymous coordination.
- Excluded information: joint names/order, skeleton topology, absolute pose,
  segment Euler angles, joint correspondence, retargeting, duration, sample
  rate, action labels, text, and audio.

The four shared labels are `angry`, `neutral`, `happy`, and `sad`. Both
datasets' labels are intended-performance protocol labels, not
human-confirmed robot-observable emotion labels.

## Leakage controls

- XEM model selection uses participants 1–6 for training, 7–8 for validation,
  and 9–10 for final test.
- BEAT2 contains 17 train, 4 validation, and 4 test speakers with no speaker
  overlap.
- BEAT2 contains 742 train, 146 validation, and 262 test source groups with no
  source-group overlap.
- XEM ridge alpha is selected on XEM validation only.
- BEAT2 labels are never used to fit or select the classifier.
- Cross-domain neutral calibration uses only weak-neutral samples from the
  BEAT2 train split. BEAT2 validation/test features and labels are not used for
  calibration.
- Metrics are reported at both clip and source-group level. Speaker-cluster
  bootstrap intervals avoid treating thousands of clips as thousands of
  independent people.

## Data actually evaluated

| Dataset | Rows | Split |
|---|---:|---|
| XEM | 1,000 | 600 train / 200 validation / 200 test |
| BEAT2 overlapping four classes | 11,090 | 6,932 train / 1,478 validation / 2,680 test |

BEAT2 class counts are 9,615 neutral, 633 happy, 516 angry, and 326 sad.
The 429 fear and 620 surprise rows have no XEM counterpart and are excluded.
All 11,090 referenced controller CSVs were hash-verified before use.

## Primary results

The primary variant uses per-domain train-neutral robust centering/scaling and
all 41 common features.

| Evaluation | Balanced accuracy | Macro F1 | Minimum class recall |
|---|---:|---:|---:|
| XEM validation | 63.0% | 61.9% | 44.0% |
| XEM test | 75.0% | 74.4% | 46.0% |
| BEAT2 validation, clip | 27.8% | 21.6% | 0.0% |
| BEAT2 validation, source group | **21.2%** | **18.4%** | **0.0%** |
| BEAT2 test, clip | 28.4% | 18.0% | 0.0% |
| BEAT2 test, source group | **31.4%** | **19.0%** | **0.0%** |

Four-way chance balanced accuracy is 25%.

At BEAT2 source-group level:

- validation recalls: angry 0%, happy 0%, neutral 59.8%, sad 25.0%;
- test recalls: angry 0%, happy 0%, neutral 46.6%, sad 78.9%.

The model therefore does not identify the four emotions. It collapses the
target domain mostly into neutral/sad and misses angry/happy entirely.

The BEAT2 speaker-cluster bootstrap 95% interval is 20.5–32.1% on validation
and 25.3–33.1% on test. Validation is not above chance. The validation/test
source-group balanced-accuracy gap is 10.18 percentage points, just beyond the
predeclared stability limit.

## Ablations

| Variant | BEAT2 val source-group BA | BEAT2 test source-group BA |
|---|---:|---:|
| No BEAT2 neutral calibration | 24.0% | 25.3% |
| Train-neutral calibrated, all features | 21.2% | 31.4% |
| Train-neutral calibrated, shape only | 18.6% | 22.2% |

Target neutral calibration changes which low-activity class wins, but does not
recover emotion discrimination. Removing amplitude makes transfer worse,
which also shows that much of the XEM success comes from dataset-specific
activity magnitude rather than portable expressive dynamics.

Leaving each of the 17 BEAT2 train-neutral speakers out of calibration changes
test source-group balanced accuracy from 25.6% to 33.4%. This range passes the
calibration-sensitivity bound, but all values remain below the required 40%.

## Predeclared gate outcome

Only 4 of 12 checks pass. Critical failures include:

- BEAT2 validation and test source-group balanced accuracy below 40%;
- BEAT2 validation and test source-group macro F1 below 35%;
- zero recall for at least one class on both target splits;
- BEAT2 validation speaker-bootstrap lower bound below chance;
- validation/test source-group stability gap above 10%.

The saved research model and feature cache explicitly carry
`generator_compatible=false`. Even a future passing result would only make
this a candidate for prioritizing human review; robot-observable human labels
would still be required before any training use.

## Artifacts

- Config:
  `configs/xem_to_beat2_weak_emotion_domain_shift_v1.json`
- Tool:
  `tools/external_emotion/evaluate_xem_on_beat2_weak_emotion_v1.py`
- Implementation:
  `upper_body_skeleton/xem_beat2_domain_bridge.py`
- Machine-readable report:
  `deliverables/external_emotion_research/xem_to_beat2_weak_emotion_domain_shift_v1/domain_shift_report.json`
- Audit:
  `deliverables/external_emotion_research/xem_to_beat2_weak_emotion_domain_shift_v1/audit.json`
- Tests:
  `tests/test_xem_beat2_domain_bridge.py`

The focused XEM and bridge test suite passes 12/12 tests.
