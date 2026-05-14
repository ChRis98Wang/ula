# 06 人体 3D 骨架到机器人 URDF 绑定

日期：2026-05-09

## 1. 目标

把人体上半身 3D 骨架转换成机器人上半身 11 个关节角。

输入：

```text
人体 3D 关键点序列
```

输出：

```text
机器人上半身 q(t)
```

目标不是人体和机器人几何完全相同，而是让机器人骨段方向、动作节奏和表达意图与人体相对应。

## 2. 机器人关节

输出顺序：

```text
waist_yaw_joint
waist_roll_joint
right_arm_shoulder_pitch_joint
right_arm_shoulder_roll_joint
right_arm_shoulder_yaw_joint
right_arm_elbow_roll_joint
left_arm_shoulder_pitch_joint
left_arm_shoulder_roll_joint
left_arm_shoulder_yaw_joint
left_arm_elbow_roll_joint
head_yaw_joint
```

## 3. 关键人体骨段

```text
pelvis_center = (left_hip + right_hip) / 2
shoulder_center = (left_shoulder + right_shoulder) / 2

torso = pelvis_center -> shoulder_center
head = shoulder_center -> nose / ear_center
left_upper_arm = left_shoulder -> left_elbow
left_forearm = left_elbow -> left_wrist
right_upper_arm = right_shoulder -> right_elbow
right_forearm = right_elbow -> right_wrist
```

## 4. 绑定表

| 机器人关节 | 人体输入 | 说明 |
| --- | --- | --- |
| `waist_yaw_joint` | torso_forward | 躯干左右转向 |
| `waist_roll_joint` | torso_up / shoulder line | 躯干左右倾斜 |
| `right_arm_shoulder_pitch_joint` | right_upper_arm | 右上臂前后摆 |
| `right_arm_shoulder_roll_joint` | right_upper_arm | 右上臂外展/内收 |
| `right_arm_shoulder_yaw_joint` | right_upper_arm + right_forearm | 右上臂旋转 |
| `right_arm_elbow_roll_joint` | right_upper_arm 与 right_forearm 夹角 | 右肘弯曲 |
| `left_arm_shoulder_pitch_joint` | left_upper_arm | 左上臂前后摆 |
| `left_arm_shoulder_roll_joint` | left_upper_arm | 左上臂外展/内收 |
| `left_arm_shoulder_yaw_joint` | left_upper_arm + left_forearm | 左上臂旋转 |
| `left_arm_elbow_roll_joint` | left_upper_arm 与 left_forearm 夹角 | 左肘弯曲 |
| `head_yaw_joint` | nose、left_ear、right_ear | 头部左右转向 |

## 5. 解算方法

### 5.1 几何初值

根据人体骨段方向估计：

- 肩 pitch。
- 肩 roll。
- 肩 yaw。
- 肘 flexion。
- 腰 yaw。
- 腰 roll。
- 头 yaw。

几何初值用于快速 MVP。

### 5.2 IK 微调

第二阶段加入 IK：

```text
目标：机器人上臂方向接近人体上臂方向
目标：机器人前臂方向接近人体前臂方向
目标：头部方向接近人体头部方向
约束：URDF joint limit
约束：速度限制
约束：动作平滑
```

优化目标：

```text
minimize:
  w_upper * angle_error(robot_upper_arm_dir, human_upper_arm_dir)
+ w_fore  * angle_error(robot_forearm_dir, human_forearm_dir)
+ w_head  * angle_error(robot_head_dir, human_head_dir)
+ w_smooth * ||q_t - q_(t-1)||^2
+ w_neutral * ||q_t - q_neutral||^2

subject to:
  q_lower <= q_t <= q_upper
  abs(q_t - q_(t-1)) / dt <= velocity_limit
```

## 6. 标定动作

必须准备几个标定视频，用于检查符号：

- 双臂自然下垂。
- 右手向前。
- 右手向右侧。
- 右肘弯曲。
- 左手向前。
- 左手向左侧。
- 左肘弯曲。
- 头向左。
- 头向右。
- 躯干向左转。
- 躯干向右转。

如果标定动作都正确，再处理复杂动作。

## 7. 常见错误

| 错误 | 表现 | 修正 |
| --- | --- | --- |
| 左右镜像 | 左手视频变成右手动 | 检查 landmark 名称和相机镜像 |
| 肩 pitch 反向 | 向前伸变向后摆 | 反转 scale 或 axis |
| 肩 roll 反向 | 外展变内收 | 检查左右臂轴符号 |
| 肘角反向 | 弯曲变伸直 | 调整 elbow flexion 映射 |
| 头 yaw 反向 | 看左变看右 | 检查 head yaw 符号 |
| 腰过大 | 躯干动作夸张 | 降低 waist scale |

## 8. 输出报告

每条动作生成：

```json
{
  "motion_id": "wave_right_001",
  "joint_limit_violations": 0,
  "velocity_violations": 0,
  "mean_upper_arm_direction_error_deg": {
    "left": 8.2,
    "right": 6.7
  },
  "mean_forearm_direction_error_deg": {
    "left": 10.1,
    "right": 9.5
  },
  "smoothing_applied": true,
  "accepted_for_preview": true
}
```
