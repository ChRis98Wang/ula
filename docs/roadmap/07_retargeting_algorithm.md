# 07 重定向算法开发说明

日期：2026-05-09

## 1. 输入与输出

输入：

```text
pose_clean JSON
robot_upper_body config
human_robot_binding config
URDF
```

输出：

```text
60 Hz robot joint CSV
retargeting report
robot preview
```

## 2. 处理流程

```text
读取 clean pose
  -> 检查关键点
  -> 建立 torso local frame
  -> 重采样到 60 Hz
  -> 计算人体骨段方向
  -> 几何解算机器人关节初值
  -> clamp 到 URDF limit
  -> 速度限制
  -> 平滑
  -> 导出 CSV
  -> 生成 report
```

## 3. 伪代码

```python
pose = load_pose_clean(path)
robot = load_robot_config(robot_config)
binding = load_binding(binding_config)

frames_60hz = resample_pose_by_time(pose.frames, output_hz=60)
q_prev = robot.neutral_pose
rows = []

for frame in frames_60hz:
    if not upper_body_confident(frame):
        q = hold_or_interpolate(q_prev)
    else:
        torso_frame = compute_torso_frame(frame.landmarks_3d)
        human_features = compute_human_features(frame.landmarks_3d, torso_frame)
        q_raw = solve_geometric_initial_q(human_features, binding)
        q_limited = apply_joint_limits(q_raw, robot.joint_limits)
        q_velocity_safe = apply_velocity_limits(q_limited, q_prev, dt=1/60)
        q = smooth(q_velocity_safe, q_prev)

    rows.append([frame.time_sec] + q_in_joint_order(q))
    q_prev = q

write_csv(rows)
write_report()
```

## 4. 关键模块

### 4.1 `compute_torso_frame`

输入：

- left_hip
- right_hip
- left_shoulder
- right_shoulder

输出：

- torso origin
- torso_right
- torso_up
- torso_forward

### 4.2 `compute_human_features`

输出：

```text
left_upper_arm_dir
left_forearm_dir
right_upper_arm_dir
right_forearm_dir
left_elbow_flexion
right_elbow_flexion
torso_yaw
torso_roll
head_yaw
motion_energy
```

### 4.3 `solve_geometric_initial_q`

作用：

- 根据绑定表把人体特征映射到机器人关节角。
- 使用 `scale` 和 `offset` 做标定。
- 左右臂分开处理，不共享符号。

### 4.4 `apply_joint_limits`

作用：

- 从 URDF 或配置读取 joint lower / upper。
- 超限直接 clamp。
- report 记录超限次数和幅度。

### 4.5 `apply_velocity_limits`

作用：

- 限制相邻帧关节变化。
- 防止真机抖动和突然动作。

公式：

```text
delta_max = max_velocity * dt * velocity_scale
q_t = q_prev + clamp(q_target - q_prev, -delta_max, delta_max)
```

### 4.6 `smooth`

推荐：

- 一阶低通。
- 或 Savitzky-Golay。
- 或 One Euro Filter。

MVP 用一阶低通即可。

```text
q_smooth = alpha * q_target + (1 - alpha) * q_prev
```

## 5. 安全默认值

```yaml
output_hz: 60
safety_scale: 0.5
max_velocity_scale: 0.5
low_confidence_policy: hold_last
max_missing_duration_sec: 0.3
filter_alpha: 0.35
```

## 6. CSV 导出

表头必须固定：

```csv
time_sec,waist_yaw_joint,waist_roll_joint,right_arm_shoulder_pitch_joint,right_arm_shoulder_roll_joint,right_arm_shoulder_yaw_joint,right_arm_elbow_roll_joint,left_arm_shoulder_pitch_joint,left_arm_shoulder_roll_joint,left_arm_shoulder_yaw_joint,left_arm_elbow_roll_joint,head_yaw_joint
```

时间列：

```text
0.0000
0.0167
0.0333
```

精度：

```text
小数 6 位足够
```

## 7. 验收测试

每个测试输入一个简单动作：

- 右手向前。
- 右手向右。
- 右肘弯曲。
- 左手向前。
- 左手向左。
- 左肘弯曲。
- 头向左。
- 头向右。

每个测试检查：

- 对应关节变化方向正确。
- 非相关关节不大幅变化。
- 所有关节未超限。
- 输出 CSV 行数正确。
- 没有 NaN。

## 8. 失败处理

如果关键点缺失：

- 小于 0.3 秒：插值。
- 大于 0.3 秒：保持上一帧或切片。

如果出现 NaN：

- 当前片段作废。
- report 记录失败原因。

如果关节超限严重：

- 自动缩放动作幅度。
- 仍超限则作废。
