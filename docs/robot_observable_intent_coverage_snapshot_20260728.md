# Observable interaction intent coverage snapshot

Snapshot time: 2026-07-28, before the new two-reviewer intent merge.

## What is actually available

| Pool | State | Clips | Explicit intent train-ready |
| --- | --- | ---: | ---: |
| ULA0513 native 18D semantic-motion records | Anonymous intent review queued | 23 | 0 |
| HAA500 full physically-passed interaction candidates | All robot videos rendered; anonymous intent review queued | 43 | 0 |
| BEAT2 v8 expanded release | Motion-only; semantic and affect blocks masked | 12,148 | 0 |

The 66 queued videos are not 66 proven interaction labels. They are 66 videos
that can now be judged under one versioned ontology without source-category
leakage.

## HAA500 candidate coverage

The 43 conservative candidates are source-filtered and physically passed, but
source classes do not grant semantic admission:

| Candidate meaning | Clips | Target observable intent |
| --- | ---: | --- |
| Salute | 4 | `salute` |
| Applause | 7 | `applaud` |
| Raise hand for attention | 11 | `raise_hand_get_attention` |
| Bow | 7 | `bow` |
| Deep bow | 8 | `bow` |
| Fist-bump offer | 2 | `offer_fist_bump`, dyadic evidence required |
| High-five offer | 1 | `offer_high_five`, dyadic evidence required |
| Hug offer | 3 | `offer_hug`, partner geometry required |

All 43 now have dedicated 1280x720, 30 Hz, full-robot review videos: 2,514
frames and 83.8 seconds in total. Every video passed frame-count, nonblank,
measurable-motion, full-body framing, H.264/yuv420p, and SHA256 checks. The
public anonymous bundle contains no source action names. Independent intent
review is still required.

## Critical gaps

- Verified `wave_to_person`: 0 clips.
- Known HAA500 greeting-wave candidates: 0 clips.
- HAA500 `arm_wave`: 20 clips, explicitly excluded because it is a dance-style
  arm wave and is not evidence for greeting or farewell.
- `greeting` versus `farewell`: no motion-only labels; these are context roles
  that require later dialogue or scene evidence.
- BEAT2 conversational motion: useful for motion and head priors, but current
  general motion descriptions do not establish these explicit interaction
  intents.

Therefore the current database is not yet sufficient for a credible
wave/greeting-conditioned training run. The ontology and network path are now
defined, but the next data collection priority is repeated, varied
`wave_to_person`, `beckon_come_here`, `raise_hand_get_attention`, `stop_warning`,
`agree_nod`, and `disagree_head_shake` actions with full onset-apex-offset arcs.
