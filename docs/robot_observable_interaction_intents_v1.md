# Robot-observable interaction semantics v1

This contract gives the current 18D database Kimodo-like, action-level semantics
without importing opaque Kimodo names or inferring labels from filenames.

## Label layers

| Layer | Example | Network role | Admission evidence |
| --- | --- | --- | --- |
| Observable intent | `wave_to_person` | Primary 27D one-hot | Anonymous full-robot video consensus |
| Semantic text | `Raise one hand and wave it side to side toward the person.` | Auxiliary Qwen/text latent | Canonical intent plus independently reviewed motion realization |
| Pragmatic role | `greeting`, `farewell` | Context-only, currently masked | Later dialogue or scene-context review |
| Emotion | `happy`, `sad`, etc. | Independent emotion channel | Separate affect review |
| Motion attributes | laterality, speed, amplitude | Realization/style attributes | Robot trajectory or blind motion description |
| Ordinary speaking motion | `conversational_gesturing` | Secondary motion realization + auxiliary text | Official BEAT2 co-speech event plus verified 18D trajectory |
| Source metadata | `deictic`, action filename, transcript | Metadata only | Never enables semantic supervision |

`greeting` and `farewell` are deliberately not separate motion intents. A wave
can perform either role, but robot motion alone normally cannot prove which role
was intended. Both therefore share `wave_to_person`; the role remains masked
until dialogue or scene context exists.

## 27 condition slots

| Slot | Intent id | Chinese meaning | Evidence mode |
| ---: | --- | --- | --- |
| 0 | `idle_attentive` | 专注待机 | visual |
| 1 | `listen_attentively` | 积极倾听 | visual + dyadic context |
| 2 | `wave_to_person` | 向人挥手 | visual |
| 3 | `beckon_come_here` | 招手示意过来 | visual |
| 4 | `raise_hand_get_attention` | 举手引起注意 | visual |
| 5 | `salute` | 敬礼 | visual |
| 6 | `bow` | 向人鞠躬 | visual |
| 7 | `stop_warning` | 停止或警告 | visual |
| 8 | `point_left` | 指向左侧 | visual |
| 9 | `point_right` | 指向右侧 | visual |
| 10 | `point_forward` | 指向前方或对方 | visual |
| 11 | `agree_nod` | 点头同意 | visual |
| 12 | `disagree_head_shake` | 摇头否定 | visual |
| 13 | `applaud` | 鼓掌 | visual |
| 14 | `celebrate` | 庆祝成功 | visual |
| 15 | `comfort_reach` | 伸手安慰 | visual + dyadic context |
| 16 | `offer_hug` | 张臂拥抱 | visual + dyadic context |
| 17 | `offer_high_five` | 举手击掌 | visual + dyadic context |
| 18 | `refuse_push_away` | 拒绝或推开 | visual |
| 19 | `shrug_uncertain` | 耸肩表示不确定 | visual |
| 20 | `search_scan` | 转头搜寻 | visual |
| 21 | `curious_lean_look` | 好奇探看 | visual + target context |
| 22 | `withdraw_turn_away` | 退缩转开 | visual |
| 23 | `disappointment_slump` | 失望低落 | visual |
| 24 | `hesitate_retract` | 犹豫后收回 | visual |
| 25 | `explain_present` | 解释或展示 | visual + speech context |
| 26 | `offer_fist_bump` | 伸拳邀请碰拳 | visual + dyadic context |

Dance-style arm waving is explicitly outside this interaction ontology and may
not be relabeled as `wave_to_person`.

## Ordinary speaking is not `explain_present`

BEAT2's ordinary co-speech gestures use the independent secondary realization
label `conversational_gesturing`.  This label says that the robot makes natural
upper-body gestures while speaking; it does not claim a greeting, explanation,
emotion, or any other primary intent.  Its bilingual prompt is composed with
trajectory-derived amplitude, pace, laterality, energy, and head engagement.

`explain_present` remains a primary interaction intent and therefore requires
anonymous robot-video evidence plus separately verified speech-turn context.
The source transcript, legacy prompt, and filename cannot upgrade an ordinary
BEAT2 gesture into this intent.

The machine-readable secondary contract is
`training/contracts/robot_observable_motion_realizations_v1.json`.

The machine-readable ontology also defines the motion signature and hard
negatives for every slot. For example, a wave requires repeated lateral cycles;
a raised-hand request has a hold and no repeated cycle; beckoning repeatedly
pulls inward toward the robot.

## Fail-closed review

Two independent reviewers must watch the anonymous robot video with audio,
filename, source action, transcript, and prior prompt hidden. A label becomes
train-ready only when both reviewers select the same intent, both report at
least 0.8 confidence, and both explicitly check the intent's hard negatives.
Disagreement remains `pending_adjudication`; it is not converted into a coarse
label.

The initial ULA0513 migration correctly emits no automatic labels:

- source records: 23
- pending intent review: 23
- intent train-ready: 0
- rejected: 0

This does not remove the 23 records from motion or motion-text training. It only
keeps the new explicit intent block at zero until the additional intent review
has established what is visibly present.

## Network compatibility

The v9 adapter preserves the existing 264D tensor width so the architecture does
not grow solely for this ontology. It reuses the physical 27D behavior region
under an explicit v9 checkpoint contract and clears incompatible legacy Kimodo
family, coarse legacy intent, and coarse gesture values. Old v8 checkpoints keep
their historical interpretation and cannot be loaded as v9 without compatible
ontology metadata.

Machine-readable artifacts:

- `training/contracts/robot_observable_interaction_intents_v1.json`
- `upper_body_skeleton/robot_observable_intents.py`
- `upper_body_skeleton/ula_v2_observable_intent_v9.py`
- `tools/human_motion_review/build_observable_intent_review_v1.py`
