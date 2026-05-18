# Emotional Gesture Literature Notes

Date: 2026-05-15

Scope: upper-body V2 robot language-action model. Current priority is text-conditioned emotional body expression; audio-conditioned synchronization is the next stage.

## Practical Takeaway

The next useful step is not to train a full audio model immediately. First, strengthen the text-to-emotion-to-motion layer:

- Add intention labels, not only emotion labels.
- Separate emotion state from gesture function.
- Add duration and transition labels for long-form expression.
- Train smoothness, energy, and semantic consistency losses.
- Reweight rare emotional / expressive samples so neutral restrained motion does not dominate.

Audio should be introduced later as a rhythm/prosody stream that modulates timing, beat, and intensity. It should not replace the text/intent/emotion control path.

## Papers And What To Use

### LiveGesture

Source: https://m-usamasaleem.github.io/publication/LiveGesture/LiveGesture.html and https://arxiv.org/abs/2604.10927

Core idea:

- Real-time speech-driven full-body gesture generation.
- Zero look-ahead and arbitrary sequence length.
- Streamable Vector-Quantized Motion Tokenizer converts body regions into causal discrete motion tokens.
- Hierarchical Autoregressive Transformer has region experts for body parts, then fuses cross-region dynamics.
- Conditions on live causal audio tokens.
- Uses autoregressive masking and random region masking for robustness.

Use later:

- For the audio stage, split robot motion into regions: torso, left arm, right arm, wrists.
- Use causal audio features and token history when online generation matters.
- Use masking training so the model survives imperfect generated history.

Do not copy directly now:

- It is full-body SMPL-X and live audio first. Our current problem is robot upper-body expression controlled by text/emotion.

### MIBURI

Source: https://arxiv.org/abs/2603.03282

Core idea:

- Online causal gesture synthesis for real-time spoken dialogue.
- Uses body-part aware gesture codecs with hierarchical multi-level discrete tokens.
- Autoregressive generation is conditioned on LLM-based speech-text embeddings.
- Adds auxiliary objectives to increase expressiveness and diversity while avoiding static poses.

Use now:

- Add anti-static / expressiveness objectives to avoid low-energy average motion.
- Treat body parts hierarchically, even for the robot: torso sets posture, shoulders/elbows express, wrists add detail.
- Use text embeddings for intent and emotional state before adding raw audio.

Use later:

- For speech, use the speech transcript plus audio embeddings, not audio alone.

### Intentional Gesture

Source: https://arxiv.org/abs/2505.15197

Core idea:

- Gestures should carry communicative intention, not only speech rhythm or text surface form.
- The InG dataset augments BEAT2 with intention annotations.
- A motion tokenizer injects high-level intention annotations into motion representations.

Use now:

- Add a field `communicative_intent_text`.
- Add discrete intent classes such as explain, emphasize, refuse, reassure, request_help, warn, invite, yield_turn.
- Generate / curate intention sentences for every motion window.
- Train the language head to predict intention before action generation.

Why important:

- Our current labels are too weak. “nervous” alone does not specify whether the robot is explaining, apologizing, refusing, or asking for help.

### BEAT / BEAT2 / EMAGE

Sources:

- BEAT: https://arxiv.org/abs/2203.05297 and https://pantomatrix.github.io/BEAT/
- EMAGE / BEAT2: https://arxiv.org/abs/2401.00374

Core idea:

- BEAT provides Body-Expression-Audio-Text data with emotion and semantic annotations.
- BEAT2 upgrades to SMPL-X / FLAME holistic mesh-level data.
- EMAGE uses masked audio gesture modeling and VQ-VAEs for face, local body, hands, and global movement.

Use now:

- Follow BEAT-style labels: text, emotion, semantic relevance, motion, speaker/context.
- Use masked motion reconstruction as an auxiliary task: mask some frames or joints and reconstruct them.
- Keep face/head out of our target, but preserve the idea of region-specific motion modules.

Use later:

- Use BEAT/BEAT2 as an external reference dataset for audio-text-motion alignment.

### EmotionGesture

Source: https://arxiv.org/abs/2305.18891

Core idea:

- Audio-driven emotional co-speech gesture generation.
- Emotion and beat are entangled, so it mines both emotion and audio beat features.
- Uses transcript-based visual-rhythm alignment.
- Adds Motion-Smooth Loss to reduce unstable jitter.

Use now:

- Add motion smoothness loss to the robot model, not only post-processing.
- Add energy consistency loss: generated motion energy should match emotional intensity.
- For text-only stage, derive pseudo-beats from punctuation, phrase boundaries, and emphasis words.

Use later:

- Replace pseudo-beats with audio beat/onset/prosody features.

### EMoG

Source: https://arxiv.org/abs/2306.11496

Core idea:

- Emotion helps resolve the one-to-many mapping from speech to gesture.
- Separates joint correlation modeling and temporal dynamics modeling.

Use now:

- Keep emotion as an explicit condition, not hidden only in text embedding.
- Add joint-correlation-aware constraints for shoulder/elbow/wrist groups.
- Consider separate torso/arm temporal modules if the current transformer remains unstable.

### ConvoFusion

Sources:

- https://arxiv.org/abs/2403.17936
- https://vcai.mpi-inf.mpg.de/projects/ConvoFusion/
- https://github.com/m-hamza-mughal/convofusion

Core idea:

- Diffusion-based multimodal conversational gesture synthesis.
- Supports controllability and conversational / multi-person settings.

Use now:

- Keep the architecture controllable: text, emotion, gesture type, duration, transition should stay editable.
- For future dialogue, add role/turn-taking labels: speaking, listening, interrupting, yielding, responding.

### Text-To-Affective-Gesture / Text2Gestures

Sources:

- Toward Automated Generation of Affective Gestures from Text: https://arxiv.org/abs/2103.03079
- Text2Gestures: https://arxiv.org/abs/2101.11101

Core idea:

- Text contains both semantic and affective information.
- Gesture can be parameterized by shape, intensity, and speed.
- Emotive gestures can be generated directly from text and task context.

Use now:

- Add explicit control dimensions:
  - shape: open, close, cross, point, hold, shrug.
  - intensity: small, medium, large.
  - speed: slow, normal, fast.
  - tension: relaxed, tense.
  - openness: closed, neutral, open.

These are easier to map onto robot joints than vague emotion names.

## Proposed Label Schema For Current Dataset

Add one semantic record per episode/window:

```json
{
  "language_instruction": "紧张地解释一件困难的事情，然后逐渐缓和下来",
  "communicative_intent": "explaining",
  "intent_detail": "justify_or_clarify",
  "emotion": "nervous",
  "emotion_trajectory": "nervous_to_calm",
  "arousal": 0.65,
  "valence": -0.15,
  "gesture_function": "emphasis",
  "body_shape": "semi_closed",
  "motion_style": "tense_restrained",
  "intensity": 0.55,
  "speed": 0.45,
  "openness": 0.25,
  "tension": 0.75,
  "duration_sec": 12.0,
  "transition": "emotion_change",
  "phase": "onset_hold_decay",
  "quality_weight": 1.0
}
```

## Model Changes To Prioritize Before Audio

1. Add a language-to-code head:
   - text embedding -> intent, emotion, gesture function, intensity, speed, openness, tension, duration, transition.

2. Train with balanced sampling:
   - oversample nervous, uncertain, angry_like, friendly, excited.
   - downsample neutral/restrained/null.

3. Add losses:
   - flow matching loss.
   - planner duration/transition loss.
   - smoothness loss on velocity and acceleration.
   - energy matching loss against intensity/arousal.
   - code classification/regression loss.

4. Add long-form labels:
   - not every sample should be 4 seconds.
   - construct 8-30 second episodes by stitching or resegmenting original long retarget CSVs.

## Audio Stage Later

When text emotion works:

1. Add audio features:
   - RMS energy.
   - pitch / F0.
   - onset / beat.
   - speech rate.
   - pause boundaries.
   - wav2vec/HuBERT-style embeddings.

2. Add time alignment:
   - transcript word timestamps.
   - phrase boundaries.
   - beat-aligned motion peaks.

3. Architecture:
   - text/emotion branch keeps semantic intent.
   - audio branch controls timing, beat, speed, and intensity modulation.
   - action expert receives fused condition.

4. Evaluation:
   - rhythm/beat consistency.
   - semantic gesture relevance.
   - emotion recognizability.
   - jerk/velocity safety.
   - MuJoCo collision and joint-limit safety.

## Immediate Recommendation

Do the next iteration as `ULA-FM v0.5 text-emotion`:

- Keep no visual head.
- Keep no audio head yet.
- Add richer labels and code head.
- Train balanced emotional batches.
- Add smoothness and energy losses.
- Generate 20-30 second previews for nervous_to_calm, angry_to_controlled, uncertain_to_confident, friendly_explaining, refusing_politely.

Only after this is visually coherent should audio be added.
