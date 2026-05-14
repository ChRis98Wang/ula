# 08 ROS2 回放与机器人接入

日期：2026-05-09

## 1. 目标

把重定向生成的 60 Hz CSV 接入上半身回放节点，完成仿真或真机动作验证。

## 2. CSV 格式

```csv
time_sec,waist_yaw_joint,waist_roll_joint,right_arm_shoulder_pitch_joint,right_arm_shoulder_roll_joint,right_arm_shoulder_yaw_joint,right_arm_elbow_roll_joint,left_arm_shoulder_pitch_joint,left_arm_shoulder_roll_joint,left_arm_shoulder_yaw_joint,left_arm_elbow_roll_joint,head_yaw_joint
0.0000,...
0.0167,...
0.0333,...
```

要求：

- 60 Hz。
- 时间单调递增。
- 没有 NaN。
- 关节名完全匹配。
- 单位为 rad。

## 3. 动作文件目录

默认目录：

```text
~/.ros/kh_upper_body_teach/motions/
```

复制：

```bash
cp upper_body_motion_data/retargeted_csv/wave_right.csv ~/.ros/kh_upper_body_teach/motions/wave_right.csv
```

## 4. 回放命令

```bash
ros2 topic pub --once /upper_body_teach_node/command std_msgs/msg/String "{data: 'play wave_right 0.4'}"
```

停止：

```bash
ros2 topic pub --once /upper_body_teach_node/command std_msgs/msg/String "{data: 'stop'}"
```

列出动作：

```bash
ros2 topic pub --once /upper_body_teach_node/command std_msgs/msg/String "{data: 'list'}"
```

## 5. 首次真机执行策略

首次播放必须保守：

- 使用低幅度 CSV。
- 使用较长 blend。
- 使用低刚度或安全模式。
- 人手远离运动范围。
- 急停可用。
- 先播放 0.2 到 0.5 倍幅度。

建议流程：

```text
离线 CSV 检查
  -> 机器人预览
  -> 仿真播放
  -> 低刚度真机
  -> 正常参数真机
```

## 6. 安全检查清单

播放前：

- CSV 表头正确。
- 行数符合时长。
- 没有 NaN 或空值。
- 关节角在 limit 内。
- 速度在限制内。
- 动作没有突变。
- 左右臂方向已经预览确认。

播放中：

- 观察电机温度。
- 观察异常噪声。
- 观察电流或力矩异常。
- 观察机械干涉。
- 保持急停可达。

播放后：

- 记录动作评分。
- 记录是否可进入动作库。
- 记录是否需要 scale / offset 调整。

## 7. 动作评分

| 分数 | 标准 |
| --- | --- |
| 5 | 方向、节奏、幅度都对应，可进入训练库 |
| 4 | 基本对应，小误差可接受 |
| 3 | 能看出意图，但关节局部明显不准 |
| 2 | 左右、前后或肘部有明显错误 |
| 1 | 不可用 |

只把 4 到 5 分动作放入训练库。

## 8. 失败动作处理

如果左右错：

- 检查输入视频是否镜像。
- 检查 landmark 左右。
- 检查 binding 的左右臂配置。

如果幅度过大：

- 降低 `safety_scale`。
- 降低对应关节 scale。

如果抖动：

- 增大滤波。
- 降低速度上限。
- 检查 pose confidence。

如果动作方向反：

- 调整对应关节 scale 符号。
- 检查 URDF axis。
