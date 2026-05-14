# 09 Flow Matching 动作生成计划

日期：2026-05-09

## 1. 训练时机

不要在第一阶段一开始就训练 Flow Matching。

训练前必须满足：

- 视频到 3D 骨架稳定。
- 3D 骨架到机器人关节稳定。
- 至少 5 类动作，每类 3 条以上可用样本。
- 动作 CSV 已经过仿真或回放验证。
- 标签体系稳定。

## 2. 模型目标

输入：

```text
动作意图
情绪表现
运动风格
机器人内部状态
目标方向
目标时长
```

输出：

```text
机器人 11 维上半身关节轨迹
```

不输出：

- 电机电流。
- 电机力矩。
- 底层控制命令。

## 3. 训练样本格式

```json
{
  "motion_id": "wave_right_001",
  "fps": 60,
  "joint_order": [
    "waist_yaw_joint",
    "waist_roll_joint",
    "right_arm_shoulder_pitch_joint",
    "right_arm_shoulder_roll_joint",
    "right_arm_shoulder_yaw_joint",
    "right_arm_elbow_roll_joint",
    "left_arm_shoulder_pitch_joint",
    "left_arm_shoulder_roll_joint",
    "left_arm_shoulder_yaw_joint",
    "left_arm_elbow_roll_joint",
    "head_yaw_joint"
  ],
  "q": [
    [0.0, 0.0, 0.1, 0.2, 1.2, 0.5, 0.0, 0.1, 1.1, 0.4, 0.0]
  ],
  "dq": [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  ],
  "condition": {
    "intent": "greeting",
    "observed_affect": "friendly",
    "motion_style": "relaxed",
    "arousal": 0.42,
    "valence": 0.68,
    "human_state_confidence": 0.74,
    "robot_energy_level": 0.82,
    "robot_safety_mode": "interactive_safe",
    "target_direction": "front",
    "duration_sec": 3.0
  }
}
```

## 4. 条件编码

离散标签：

- intent
- observed_affect
- motion_style
- robot_safety_mode
- target_direction

连续值：

- arousal
- valence
- human_state_confidence
- robot_energy_level
- duration_sec

## 5. 轨迹标准化

训练前处理：

- q 按关节 limit 归一化到 -1 到 1。
- dq 按速度限制归一化。
- 所有序列重采样到 60 Hz。
- 可按固定长度切片，例如 2 秒、4 秒、6 秒。
- 或使用 mask 支持变长。

## 6. 生成后处理

模型输出后必须：

```text
反归一化
  -> joint limit clamp
  -> velocity limit
  -> acceleration limit
  -> smoothing
  -> collision / self-interference check
  -> export CSV
```

低置信度状态：

- 输出中性动作。
- 或保持当前姿态。
- 不生成大幅动作。

## 7. MVP 模型建议

第一版模型可以简单：

```text
Conditional Flow Matching over joint trajectory
```

输入：

- condition embedding
- time embedding
- noisy trajectory

输出：

- velocity field / denoising direction

训练集小的时候，先不要追求大型模型。先确认：

- 能复现训练动作。
- 同一 intent 不同 style 有差异。
- 生成动作满足限位。

## 8. 验收标准

Flow Matching 第一版完成标准：

- 可以生成 5 类基础动作。
- 生成结果经过安全过滤后可导出 CSV。
- 同一动作在 friendly / hesitant / sharp 风格下有合理差异。
- 生成动作不超限。
- 生成动作无明显抖动。
- 仿真预览通过后才允许真机低刚度验证。
