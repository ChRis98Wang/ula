# 05 情绪、意图与状态标签设计

日期：2026-05-09

## 1. 核心原则

情绪标签可以做，但不能当成真实心理读数。

系统不能直接知道人的真实内心状态，只能根据以下可观察信号估计：

- 姿态
- 动作速度
- 动作幅度
- 面部表情
- 视线
- 语音
- 对话文本
- 场景
- 人工复核

因此标签必须带：

```text
label + confidence + source
```

## 2. 推荐标签层级

### 2.1 动作意图 intent

```text
greeting
pointing
refusing
thinking
comforting
warning
explaining
requesting_help
waiting
attention_shift
```

### 2.2 可观察情绪表现 observed_affect

```text
friendly
neutral
uncertain
nervous
angry_like
sad_like
excited
low_confidence_unknown
```

避免使用：

```text
true_emotion
```

### 2.3 运动风格 motion_style

```text
slow_safe
energetic
hesitant
sharp
relaxed
restrained
```

### 2.4 交互状态 interaction_state

```text
looking_at_robot
looking_away
approaching
backing_away
waiting
requesting_help
refusing
```

### 2.5 连续值

```text
arousal: 0.0 到 1.0
valence: -1.0 到 1.0
confidence: 0.0 到 1.0
motion_energy: 0.0 到 1.0
```

## 3. 信息来源

### 3.1 3D 骨架

能提供：

- 动作幅度。
- 动作速度。
- 身体是否靠近或远离。
- 躯干是否转开。
- 头是否看向机器人。
- 手臂是否收缩、防御或开放。

不能可靠提供：

- 真实情绪。
- 复杂心理状态。
- 讽刺、反话等语义。

### 3.2 脸部

能提供：

- 表情。
- 视线方向。
- 头部方向。

风险：

- 遮挡。
- 光照。
- 表演化表情。
- 文化差异。

### 3.3 语音

能提供：

- 语速。
- 音量。
- 语调。
- 停顿。

### 3.4 对话文本

能提供：

- 请求。
- 拒绝。
- 感谢。
- 命令。
- 疑问。
- 求助。

### 3.5 场景

能提供：

- 人与机器人距离。
- 当前任务状态。
- 是否有危险。
- 是否有其他人或物体介入。

## 4. 标签 JSON

```json
{
  "motion_id": "wave_right_001",
  "source": {
    "video_id": "2026-05-09_subject001_wave_right_front",
    "source_type": "self_collected"
  },
  "intent": {
    "label": "greeting",
    "confidence": 0.95
  },
  "observed_affect": {
    "label": "friendly",
    "arousal": 0.42,
    "valence": 0.68,
    "confidence": 0.74,
    "source": ["pose", "face"]
  },
  "motion_style": {
    "label": "relaxed",
    "motion_energy": 0.52,
    "speed_level": "medium"
  },
  "interaction_state": {
    "attention": "looking_at_robot",
    "distance_level": "near",
    "confidence": 0.81
  },
  "human_review": {
    "reviewed": true,
    "accepted_for_training": true,
    "notes": "动作友好，右手方向正确"
  }
}
```

## 5. 自动标注策略

第一阶段：

- 自动生成动作候选标签。
- 自动生成运动能量。
- 自动生成基础风格标签。
- 情绪只给候选，不直接用于真机动作。
- 高价值样本人工复核。

第二阶段：

- 使用人工复核数据训练轻量分类器。
- 输出情绪表现和意图置信度。
- 低置信度输出 `low_confidence_unknown`。

第三阶段：

- 标签进入 Flow Matching 条件输入。
- 生成动作后仍然经过安全过滤。

## 6. 机器人内部状态

机器人内部状态不是猜测，是从机器人系统读到的真实状态。

字段：

```json
{
  "robot_state": {
    "battery_level": 0.82,
    "motor_temperature": "normal",
    "joint_risk": "low",
    "current_task_priority": "normal",
    "perception_confidence": 0.76,
    "energy_mode": "normal",
    "safety_mode": "interactive_safe"
  }
}
```

对动作的影响：

| 状态 | 行为 |
| --- | --- |
| 电量低 | 降低动作幅度和速度 |
| 电机温度高 | 禁止大幅动作 |
| 关节风险高 | 保守动作或保持 |
| 任务优先级高 | 动作更短、更明确 |
| 感知置信度低 | 使用中性动作 |
| 人距离太近 | 减小手臂外展 |

## 7. 标签验收标准

标签系统完成标准：

- 每条动作都有 intent。
- 每条动作都有 motion_style。
- observed_affect 可以是 unknown，但必须有 confidence。
- 自动标签和人工复核字段分开。
- 训练集只接受人工复核通过的样本。
