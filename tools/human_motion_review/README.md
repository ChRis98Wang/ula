# Human Motion Review

Scripts for **data quality review and training-admission adjudication** — the
gate between "physically retargeted / draft-labeled" (see
`tools/human_motion_collection/`) and "admitted for training." Nothing
upstream of this directory can set `accepted_for_training=true` by itself;
every clip stays fail-closed until it has been through blind human review and
an explicit adjudication step here.

Two things run in parallel and are kept deliberately separate:

- **Physical QC** (retargeting quality: joint limits, velocity, collision,
  smoothness) — produced by `tools/gmr_v2/`.
- **Semantic/affect review** (does the motion actually look like what the
  label claims; is the emotion legible) — produced by this directory, via
  *blind* review bundles so reviewers cannot see the source label while
  judging the motion.

A render passing QC, or a label being auto-generated, never by itself grants
training admission — see the `never_grants_training_admission` assertions in
`tests/test_summarize_full_18d_retarget.py` for the enforced invariant.

## Pipeline shape

```text
build_*_blind_review_bundle*.py   -> anonymous, hash-bound review queue
render_*.py                       -> MuJoCo video evidence for each queue item
(human reviewer fills in judgments, out of repo)
normalize_*_submission*.py        -> canonicalize reviewer input without changing judgments
merge_*_reviews*.py               -> combine independent reviewers into a fail-closed tier
finalize_*_decisions*.py          -> bind decisions back to the original anonymous queue
adjudicate_training_dataset.py    -> final training-admission decision
validate_*.py / audit_*.py        -> integrity checks at any stage
```

## Script index

### Build review bundles (anonymous, hash-bound queues)

| Script | Purpose |
| --- | --- |
| `build_blind_affect_review_bundle.py` | Build separated public/hidden bundles for blind robot-affect review. |
| `build_blind_review_shards_v1.py` | Split an anonymous blind-review queue into balanced hash-bound shards. |
| `build_semantic_event_video_queue.py` | Build a fail-closed MuJoCo review queue from semantic-event physical-QC passes. |
| `build_robot_observable_semantic_review_v1.py` | Build and merge label-blind robot-observable semantic review bundles. |
| `build_expression_turn_blind_review_bundle_v8.py` | Build separate anonymous arc/action and affect queues from v8 render passes. |
| `build_expression_turn_expansion_blind_review_bundle_v8.py` | Build anonymous blind-review queues for BEAT2 v8 context expansions. |
| `build_expression_turn_video_queue_v8.py` | Build internal MuJoCo queues for v8 expression-turn physical QC passes. |
| `build_beat2_emotion_review_queue_v1.py` | Select a source-group-unique, split/emotion-balanced BEAT2 controller queue with two-reviewer/third-adjudicator fail-closed fields. |
| `build_beat2_emotion_blind_review_bundle_v1.py` | Project the controller-only BEAT2 emotion queue into a generic render queue and two label-blind, hash-bound primary-review shard sets (see `README_BEAT2_EMOTION_BLIND_REVIEW.md`). |
| `build_ula0513_native_blind_review_bundle.py` | Publish label-free blind review queues for rendered ULA0513 motions. |
| `build_ula0513_native_video_queue.py` | Build a private, anonymous render queue for native-length ULA0513 motions. |
| `build_interact_blind_review_bundle.py` | Build anonymous InterAct axis, dyadic semantics, and affect review queues. |
| `build_interact_blind_review_bundle_v2.py` | Build the native-BVH InterAct v2 anonymous blind-review bundle. |
| `build_interact_dyadic_expansion_blind_bundle_v2.py` | Build anonymous arc/action and affect queues for InterAct context expansions. |
| `build_interact_full_dyad_blind_bundle_v3.py` | Build an anonymous fail-closed bundle from full-dyad 2x2 evidence. |
| `build_interact_full_dyad_review_manifest_v3.py` | Build the fail-closed initial blind-review manifest for the full InterAct pool. |
| `build_interact_review_evidence_migration_v2.py` | Prove which InterAct blind-review evidence survives an axis-only bundle rebuild. |

### Expansion planning (ask for more context only where reviewers flagged ambiguity)

| Script | Purpose |
| --- | --- |
| `build_expression_turn_expansion_plan_v8.py` | Build one-level-at-a-time natural-context expansion requests from blind review. |
| `build_expression_turn_expansion_continuation_plan_v8.py` | Build fail-closed round-N BEAT2 v8 natural-boundary continuation plans. |
| `build_interact_arc_expansion_plan_v2.py` | Build fail-closed one-level InterAct dyadic natural-context expansions. |
| `build_interact_dyadic_continuation_plan_v2.py` | Build fail-closed round-N InterAct natural-boundary continuation plans. |

### Render video evidence

| Script | Purpose |
| --- | --- |
| `render_beat2_annotation_review.py` | Render and verify silent MuJoCo videos for a BEAT2 annotation review queue (see `README_BEAT2_ANNOTATION_REVIEW.md`). |
| `render_interact_axis_review.py` | Compose InterAct source-stick and real-URDF MuJoCo retarget evidence. |
| `render_interact_dyadic_review.py` | Render an anonymous two-view InterAct dyad for blind expression review. |
| `render_interact_full_dyad_evidence_v3.py` | Render full-span 2x2 source-dyad and real-URDF robot review evidence. |
| `run_interact_dyadic_review_v2.py` | Render the four native-duration InterAct dyads with the native BVH parser. |
| `run_interact_dyadic_expansion_review_v2.py` | Render one-level InterAct dyadic natural-context expansions for blind review. |
| `run_interact_full_dyad_review_v3.py` | Resumably render full-span InterAct 2x2 blind-review evidence. |

### Normalize, merge, finalize reviewer submissions

| Script | Purpose |
| --- | --- |
| `normalize_beat_arc_submission_v8.py` | Normalize BEAT2 arc-review enums and bindings without changing judgments. |
| `normalize_expression_turn_affect_submission_v8.py` | Normalize a blind affect submission without changing its judgments. |
| `normalize_interact_arc_submission_v2.py` | Normalize InterAct arc-review bindings without changing judgments. |
| `merge_expression_turn_blind_reviews.py` | Merge independent expression-turn reviews into fail-closed training tiers. |
| `merge_expression_turn_expansion_qualification_v8.py` | Merge one BEAT2 v8 expansion round into fail-closed candidate tiers. |
| `merge_interact_full_dyad_review_shards_v3.py` | Merge a complete set of validated InterAct v3 blind-review shards. |
| `finalize_beat_affect_decisions_v8.py` | Bind independent BEAT2 v8 affect decisions to an anonymous public queue. |
| `finalize_beat_arc_decisions_v8.py` | Finalize independently authored BEAT2 arc decisions against a public queue. |
| `finalize_interact_full_dyad_review_shard_v3.py` | Bind independent InterAct v3 shard decisions to their anonymous queue. |
| `compare_expression_turn_arc_reviews.py` | Fail-closed comparison of two anonymous expression arc/action reviews. |

### Adjudicate, validate, audit

| Script | Purpose |
| --- | --- |
| `adjudicate_training_dataset.py` | Deterministically adjudicate semantic, 18D QC, and independent review data into the final training-admission decision. |
| `validate_beat2_annotation_batch.py` | Fail-closed validation for a BEAT2 18D annotation batch (task-id closure, hash tamper detection, split/admission metadata). |
| `validate_review_bundle.py` | Validate an independent agent's motion-review bundle and conflict queue. |
| `summarize_full_18d_retarget.py` | Summarize and strictly validate a full ULA V2 18D retarget batch; never grants training admission on its own. |
| `audit_expression_turn_catalog_v8.py` | Fail-closed audit for a BEAT2 variable-length expression-turn catalog. |
| `audit_expression_turn_v8_catalog.py` | Independently audit a rebuilt BEAT2 expression-turn v8 catalog. |

### Contracts (imported by the scripts above, not run directly)

| Module | Purpose |
| --- | --- |
| `expression_turn_contract.py` | Fail-closed acceptance contract for variable-length expression turns. |
| `expression_turn_retarget_contract.py` | Fail-closed output contract for v8 expression-turn 18D retargeting. |

## Detailed workflow example

See `README_BEAT2_ANNOTATION_REVIEW.md` for a full, runnable example: building a
review queue, rendering a stratified sample of MuJoCo videos, and building a
blind affect A/B bundle from the render-passed set.
