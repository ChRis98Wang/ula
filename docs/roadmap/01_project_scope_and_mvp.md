# 01 项目范围与 MVP 定义

日期：2026-05-09

## 1. 项目目标

本项目要建立一条机器人上半身动作数据链路：

```text
人体视频 / 相机 / 示教
        -> 人体 3D 上半身骨架
        -> 情绪、意图、状态标签
        -> 机器人 URDF 上半身关节轨迹
        -> 动作库
        -> 条件式动作生成
        -> 仿真和真机执行
```

最终机器人应该能够根据：

- 外部情景
- 人的可观察状态
- 机器人内部状态
- 任务意图

生成合适的上半身动作。

## 2. MVP 只解决什么

MVP 只解决第一段：

```text
单人视频 -> 3D 上半身骨架 -> 机器人上半身 CSV
```

MVP 不训练 Flow Matching，不做复杂情绪识别，不做下半身，不做手指和抓取。

## 3. MVP 输入

输入视频要求：

- 单人
- 上半身完整可见
- 手臂尽量无遮挡
- 固定相机
- 30 fps 以上
- 1080p 或更高
- 每条 5 到 15 秒

第一批动作：

- 右手挥手
- 左手挥手
- 右手指向
- 左手指向
- 双臂欢迎
- 双手拒绝
- 头部转向
- 思考姿态

## 4. MVP 输出

每条视频输出：

```text
pose_raw/<video_id>.pose.json
pose_clean/<video_id>.pose.clean.json
annotations/<video_id>.annotation.json
retargeted_csv/<motion_id>.csv
reports/<motion_id>.retargeting_report.json
retargeted_preview/<motion_id>.overlay.mp4
robot_preview/<motion_id>.robot_preview.mp4
```

CSV 必须采用 60 Hz：

```csv
time_sec,waist_yaw_joint,waist_roll_joint,right_arm_shoulder_pitch_joint,right_arm_shoulder_roll_joint,right_arm_shoulder_yaw_joint,right_arm_elbow_roll_joint,left_arm_shoulder_pitch_joint,left_arm_shoulder_roll_joint,left_arm_shoulder_yaw_joint,left_arm_elbow_roll_joint,head_yaw_joint
```

## 5. MVP 验收标准

MVP 完成必须满足：

- 至少 5 类动作可自动转换为机器人 CSV。
- 每类动作至少 3 条可用样本。
- 每条样本有 3D pose JSON、annotation JSON、CSV、report。
- 95% 以上帧满足关节限位和速度限制。
- 左右臂没有镜像错乱。
- 肘部方向正确。
- 头部 yaw 方向正确。
- 动作节奏和视频一致。
- CSV 可以接入 ROS2 上半身回放节点。

## 6. 非目标

MVP 不做：

- 下半身和平衡。
- 手指、手腕精细动作。
- 真实物体抓取和接触。
- 大规模电视剧数据训练。
- 自动判断人的真实心理状态。
- 直接从模型输出电机命令。

## 7. 核心工程原则

先把数据链路做硬：

```text
3D 骨架稳定
左右绑定正确
URDF 限位正确
CSV 可回放
动作可预览
低置信度可拒绝
```

只有这条链路稳定，后续 Flow Matching 才值得训练。
