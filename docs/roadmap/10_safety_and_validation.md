# 10 安全与验证

日期：2026-05-09

## 1. 安全原则

任何来自视频、自动标注或生成模型的动作，都不能直接上真机。

必须经过：

```text
格式检查
  -> 关节限位检查
  -> 速度检查
  -> 加速度检查
  -> 平滑
  -> 预览
  -> 仿真
  -> 低刚度真机
  -> 正常真机
```

## 2. CSV 格式检查

检查项：

- 表头完全匹配。
- 时间单调递增。
- 采样频率约等于 60 Hz。
- 所有值是有限数。
- 没有 NaN。
- 没有空字段。

## 3. 关节限位检查

每个关节必须满足：

```text
lower <= q <= upper
```

如果超限：

- 小幅超限可以 clamp 并记录。
- 大幅超限应拒绝动作。

## 4. 速度限制

每个关节：

```text
dq = (q_t - q_(t-1)) / dt
```

必须小于安全速度：

```text
safe_velocity = urdf_velocity_limit * velocity_scale
```

MVP 推荐：

```text
velocity_scale = 0.5
```

## 5. 加速度限制

```text
ddq = (dq_t - dq_(t-1)) / dt
```

如果加速度突然过大：

- 说明 pose 抖动。
- 或插值有跳变。
- 或模型生成不稳定。

处理：

- 增加平滑。
- 降低速度限制。
- 拒绝该片段。

## 6. 置信度策略

如果人体关键点置信度低：

- 小于 0.3 秒：插值。
- 大于 0.3 秒：保持上一姿态或切断。
- 大于 1.0 秒：动作作废。

如果情绪标签置信度低：

- 使用 neutral。
- 不使用 sharp / energetic 动作。
- 不做大幅动作。

## 7. 真机测试流程

第一次测试：

```text
安全缩放 0.2
低刚度
慢速 blend
人员远离运动范围
急停准备
```

通过后：

```text
安全缩放 0.5
正常低速
观察温度和电流
```

最后：

```text
接近目标幅度
仍保留速度限制
```

## 8. 验证报告

每次动作验证保存：

```json
{
  "motion_id": "wave_right_001",
  "csv_valid": true,
  "joint_limit_violations": 0,
  "velocity_violations": 0,
  "acceleration_warnings": 1,
  "preview_passed": true,
  "simulation_passed": true,
  "robot_tested": false,
  "human_score": 4,
  "accepted_for_library": true,
  "notes": "右手挥手方向正确，幅度略小"
}
```

## 9. 进入动作库的条件

必须同时满足：

- 格式检查通过。
- 限位检查通过。
- 速度检查通过。
- 预览通过。
- 人工评分 4 分以上。
- 无明显左右错乱。
- 无明显关节反向。

如果用于训练，还必须：

- 标签完整。
- 3D pose 保存完整。
- 来源和授权清晰。
