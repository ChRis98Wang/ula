# 机器人上半身 3D 骨架重定向、情绪标签与动作生成总路线

日期：2026-05-09

本文档是一个独立开发路线文件，用于在工作站上规划和实现：

```text
视频 / 相机 / 示教数据
        -> 人体 3D 上半身骨架
        -> 情绪、意图、场景、内部状态标签
        -> 人体骨架到机器人 URDF 上半身关节重定向
        -> 机器人上半身动作库
        -> Flow Matching 条件式动作生成
        -> 仿真安全过滤
        -> ROS2 / 真机执行
```

核心目标：

让机器人根据外部情景、人的可观察状态、机器人内部状态和任务意图，生成合适的上半身动作。第一阶段只要求视频中的人体上半身 3D 骨架能够稳定映射到机器人上半身 URDF 关节，并输出可回放的关节轨迹。

## 1. 总体原则

### 1.1 只做上半身

第一阶段只处理：

- 腰部 yaw / roll
- 左右肩 pitch / roll / yaw
- 左右肘
- 头部 yaw

不处理：

- 下半身
- 手指
- 手腕复杂姿态
- 真实抓取接触
- 全身平衡控制

原因：

上半身表达动作最适合从视频学习，也最适合先接到现有机器人上半身回放系统。下半身和平衡会显著增加安全和动力学复杂度，应放到后续阶段。

### 1.2 骨架必须是 3D

只用 2D 骨架不够。2D 关键点只有屏幕 x/y，很难判断手是在向前伸、向侧面伸，还是只是透视变化。

第一阶段最低要求：

```json
{
  "time_sec": 0.0167,
  "landmarks_3d": {
    "left_shoulder": [x, y, z],
    "left_elbow": [x, y, z],
    "left_wrist": [x, y, z],
    "right_shoulder": [x, y, z],
    "right_elbow": [x, y, z],
    "right_wrist": [x, y, z],
    "left_hip": [x, y, z],
    "right_hip": [x, y, z],
    "nose": [x, y, z],
    "left_ear": [x, y, z],
    "right_ear": [x, y, z]
  },
  "confidence": {
    "left_shoulder": 0.98,
    "left_elbow": 0.96
  }
}
```

3D 坐标可以分三类：

| 类型 | 来源 | 优点 | 缺点 | 适用阶段 |
| --- | --- | --- | --- | --- |
| 单目相对 3D | MediaPipe / MMPose 单目 3D | 部署简单，能快速打通 | 深度和尺度不稳定 | 第一阶段 MVP |
| 多相机三角化 3D | 2 台以上同步相机 + 标定 | 真实 3D 更可靠 | 需要相机标定和同步 | 第二阶段高质量数据 |
| RGB-D 3D | 深度相机 | 有真实深度，采集标准动作方便 | 硬件依赖 | 第二阶段动作库 |

推荐路线：

```text
第一阶段：单目 3D 骨架，快速实现视频到机器人动作
第二阶段：双目 / 多相机 / RGB-D，制作高质量训练数据
第三阶段：融合情绪、意图、内部状态，训练动作生成模型
```

### 1.3 情绪标签可以做，但不能当成真实心理读数

系统不能直接读取人的真实内心状态，只能根据可观察信号进行估计。因此情绪标签必须设计成：

```text
可观察情绪表现 + 意图 + 运动风格 + 置信度
```

错误做法：

```json
{
  "emotion": "angry"
}
```

正确做法：

```json
{
  "observed_affect": "angry_like",
  "intent": "refuse",
  "arousal": 0.72,
  "valence": -0.45,
  "confidence": 0.63,
  "source": ["face", "voice", "dialogue", "pose"]
}
```

低置信度时，机器人应使用中性和安全动作，不应做夸张反应。

## 2. 第一阶段目标

第一阶段名称：

```text
Upper Body 3D Skeleton Retargeting MVP
```

输入：

- 单人上半身视频，最好是自采视频
- 或相机实时视频

中间数据：

- 3D 人体上半身关键点
- 帧级置信度
- 自动质量标注
- 人体局部坐标系

输出：

- 机器人上半身 60 Hz 关节 CSV
- 可视化预览视频
- 重定向质量报告

验收目标：

- 输入一段挥手、指向、欢迎、拒绝或头部转向视频。
- 系统输出机器人上半身 CSV。
- CSV 在仿真中能正常播放。
- CSV 可以接入 ROS2 上半身回放节点。
- 机器人动作方向、节奏、幅度与视频主体动作大体一致。
- 所有关节满足 URDF 限位和速度限制。

## 3. 当前机器人上半身关节接口

第一版目标 CSV 使用 11 个上半身关节：

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

CSV 格式：

```csv
time_sec,waist_yaw_joint,waist_roll_joint,right_arm_shoulder_pitch_joint,right_arm_shoulder_roll_joint,right_arm_shoulder_yaw_joint,right_arm_elbow_roll_joint,left_arm_shoulder_pitch_joint,left_arm_shoulder_roll_joint,left_arm_shoulder_yaw_joint,left_arm_elbow_roll_joint,head_yaw_joint
0.0000,...
0.0167,...
0.0333,...
```

输出频率：

```text
60 Hz
```

机器人动作播放接口建议沿用：

```bash
ros2 topic pub --once /upper_body_teach_node/command std_msgs/msg/String "{data: 'play wave_right 0.4'}"
```

## 4. 视频数据获取

### 4.1 自采视频

第一阶段最推荐自采视频。

拍摄规范：

- 单人出镜。
- 上半身、手臂、头部完整可见。
- 摄像机固定。
- 光照稳定。
- 背景简单。
- 1080p 或更高。
- 30 fps 或更高。
- 每条动作 5 到 15 秒。
- 动作幅度先保守，不要超过机器人肩肘范围。

推荐第一批动作：

| 动作名 | 说明 |
| --- | --- |
| `wave_right` | 右手挥手 |
| `wave_left` | 左手挥手 |
| `point_right` | 右手指向 |
| `point_left` | 左手指向 |
| `arms_open` | 双臂打开欢迎 |
| `deny` | 双手轻微拒绝 |
| `thinking` | 头转向 + 手靠近脸 |
| `attention_shift` | 头和躯干转向 |

建议每个动作采集：

- 正面
- 左前 45 度
- 右前 45 度

命名：

```text
2026-05-09_subject001_wave_right_front.mp4
2026-05-09_subject001_wave_right_left45.mp4
2026-05-09_subject001_wave_right_right45.mp4
```

### 4.2 公开视频

公开视频适合增加多样性，但要检查授权。

筛选条件：

- 授权允许研究或内部开发。
- 单人主体清晰。
- 上半身无遮挡。
- 没有频繁剪辑。
- 动作适合机器人安全执行。

### 4.3 电视剧或电影片段

电视剧片段可以用于观察人类动作和设计动作标签，但不建议作为第一阶段主数据源。

原因：

- 版权不清晰。
- 镜头切换频繁。
- 遮挡和多人同框多。
- 演员动作表演化，可能不适合机器人。
- 动作没有连续完整的 3D 信息。

如果使用，只建议作为内部研究样本，并保留：

```json
{
  "source_type": "tv_clip",
  "source_name": "example_show",
  "episode": "S01E01",
  "start_time_sec": 123.4,
  "end_time_sec": 130.2,
  "license_status": "internal_review_only",
  "allowed_use": ["annotation_schema_design", "offline_retargeting_test"]
}
```

## 5. 工作站数据目录

建议在桌面或大容量磁盘建立独立数据目录：

```text
upper_body_motion_data/
  raw_videos/
    self_collected/
    public/
    restricted_tv_clips/
  pose_raw/
    mediapipe/
    mmpose/
  pose_clean/
  annotations/
  retargeted_csv/
  retargeted_preview/
  robot_preview/
  reports/
  motion_library/
```

代码目录建议：

```text
upper_body_motion_workspace/
  configs/
    robot_upper_body_0421.yaml
    human_robot_binding.yaml
    emotion_label_schema.yaml
  scripts/
    video_pose_extract.py
    pose_cleaning.py
    auto_annotate_motion.py
    retarget_upper_body.py
    preview_retargeting.py
    export_flow_matching_dataset.py
  docs/
    README.md
```

## 6. 3D 骨架提取工具

### 6.1 MediaPipe Pose Landmarker

推荐作为第一阶段默认工具。

用途：

- 从图片、视频帧或实时视频提取人体 pose。
- 输出 33 个 pose landmarks。
- 输出 normalized image landmarks 和 world landmarks。
- 适合快速打通单目 3D 骨架链路。

优点：

- 安装和使用简单。
- 速度快。
- 对第一阶段足够。

限制：

- 单目 3D 的深度和真实尺度不稳定。
- 遮挡和侧身时精度会下降。
- 多人场景需要额外主体选择逻辑。

### 6.2 MMPose / RTMPose

推荐作为第二阶段高质量离线处理工具。

用途：

- 2D / 3D 人体姿态估计。
- WholeBody 关键点。
- 多人和复杂场景。

优点：

- 模型选择多。
- 更适合离线批处理。
- 可扩展到手部、脸部等更丰富标注。

限制：

- 安装和模型管理更复杂。
- 工作站最好有 GPU。

### 6.3 OpenPose

适合作为备选工具，尤其是多相机 3D 重建实验。

用途：

- 身体、手、脸关键点。
- 视频、图片、相机输入。
- 多视角 3D reconstruction demo。

限制：

- 部署较重。
- 对新项目不一定是最快 MVP 路线。

## 7. 3D 骨架数据格式

每个视频输出一个 pose JSON：

```json
{
  "video_id": "2026-05-09_subject001_wave_right_front",
  "pose_tool": "mediapipe_pose_landmarker",
  "coordinate_type": "single_view_world_landmarks",
  "fps": 30.0,
  "frames": [
    {
      "frame_index": 0,
      "time_sec": 0.0,
      "landmarks_3d": {
        "left_shoulder": [-0.18, 0.42, -0.03],
        "right_shoulder": [0.18, 0.42, -0.02],
        "left_elbow": [-0.32, 0.21, -0.05],
        "right_elbow": [0.36, 0.22, -0.04],
        "left_wrist": [-0.41, 0.04, -0.08],
        "right_wrist": [0.49, 0.02, -0.07],
        "left_hip": [-0.11, 0.0, 0.0],
        "right_hip": [0.11, 0.0, 0.0],
        "nose": [0.01, 0.64, -0.12],
        "left_ear": [-0.07, 0.62, -0.08],
        "right_ear": [0.08, 0.62, -0.08]
      },
      "confidence": {
        "left_shoulder": 0.99,
        "right_shoulder": 0.99,
        "left_elbow": 0.95,
        "right_elbow": 0.94,
        "left_wrist": 0.91,
        "right_wrist": 0.92
      }
    }
  ]
}
```

必须保存原始 3D 骨架，不要只保存最终机器人关节角。后续换模型或改重定向算法时，原始骨架可以复用。

## 8. 人体局部坐标系

重定向不要直接使用相机坐标。要先建立人体 torso local frame。

定义：

```text
pelvis_center = (left_hip + right_hip) / 2
shoulder_center = (left_shoulder + right_shoulder) / 2
torso_up = normalize(shoulder_center - pelvis_center)
torso_right = normalize(right_shoulder - left_shoulder)
torso_forward = normalize(cross(torso_right, torso_up))
```

所有手臂向量转换到 torso local frame：

```text
right_upper_arm = right_elbow - right_shoulder
right_forearm = right_wrist - right_elbow
left_upper_arm = left_elbow - left_shoulder
left_forearm = left_wrist - left_elbow
```

这样可以减少相机角度、人体位置和距离变化对重定向的影响。

## 9. 人体骨架到机器人 URDF 绑定

### 9.1 绑定原则

机器人和人体不是同一个结构，因此不要追求逐点完全相等。工程目标是：

- 上臂方向对应。
- 前臂方向对应。
- 肘部弯曲程度对应。
- 腰和头部朝向对应。
- 节奏对应。
- 幅度在机器人安全范围内。

### 9.2 绑定表

| 机器人关节 | 人体 3D 输入 | 第一阶段估计方式 |
| --- | --- | --- |
| `waist_yaw_joint` | torso_forward | 躯干相对初始方向的左右转角 |
| `waist_roll_joint` | shoulder_center 与 pelvis_center | 躯干左右倾斜 |
| `right_arm_shoulder_pitch_joint` | right_upper_arm | 上臂前后方向 |
| `right_arm_shoulder_roll_joint` | right_upper_arm | 上臂外展/内收方向 |
| `right_arm_shoulder_yaw_joint` | right_upper_arm + right_forearm | 手臂平面旋转 |
| `right_arm_elbow_roll_joint` | right_upper_arm 与 right_forearm 夹角 | 肘部弯曲角 |
| `left_arm_shoulder_pitch_joint` | left_upper_arm | 上臂前后方向 |
| `left_arm_shoulder_roll_joint` | left_upper_arm | 上臂外展/内收方向 |
| `left_arm_shoulder_yaw_joint` | left_upper_arm + left_forearm | 手臂平面旋转 |
| `left_arm_elbow_roll_joint` | left_upper_arm 与 left_forearm 夹角 | 肘部弯曲角 |
| `head_yaw_joint` | nose、left_ear、right_ear | 头部相对 torso 的左右转角 |

### 9.3 几何解算 + IK 微调

第一版可以先做几何解算：

```text
3D 骨段方向 -> 肩 pitch/roll/yaw、肘角、腰 yaw/roll、头 yaw
```

第二版加入 IK 微调：

```text
目标：机器人上臂方向接近人体上臂方向
目标：机器人前臂方向接近人体前臂方向
约束：URDF joint limit
约束：速度限制
约束：平滑项
```

优化目标：

```text
minimize:
  w1 * angle_error(robot_upper_arm, human_upper_arm)
+ w2 * angle_error(robot_forearm, human_forearm)
+ w3 * ||q_t - q_(t-1)||^2
+ w4 * ||q_t - q_neutral||^2

subject to:
  q_lower <= q_t <= q_upper
  abs(q_t - q_(t-1)) / dt <= velocity_limit
```

## 10. 情绪、意图和内部状态标签

### 10.1 标签不是读心

情绪标签不能表示人的真实心理，只表示系统根据可观察信号估计到的表现。

建议命名：

```text
observed_affect
```

而不是：

```text
true_emotion
```

### 10.2 推荐标签层级

动作意图：

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

情绪表现：

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

运动风格：

```text
slow_safe
energetic
hesitant
sharp
relaxed
restrained
```

交互状态：

```text
looking_at_robot
looking_away
approaching
backing_away
waiting
requesting_help
refusing
```

连续数值：

```text
arousal: 0.0 到 1.0
valence: -1.0 到 1.0
confidence: 0.0 到 1.0
motion_energy: 0.0 到 1.0
```

### 10.3 情绪标签来源

骨架能提供：

- 动作速度
- 动作幅度
- 是否靠近或后退
- 躯干是否转开
- 手臂是否防御性收缩
- 头部是否看向机器人

脸部能提供：

- 表情
- 视线
- 眼睛方向
- 头部朝向

语音能提供：

- 音量
- 语速
- 音调变化
- 停顿
- 文本内容

字幕/对话能提供：

- 是否拒绝
- 是否感谢
- 是否命令
- 是否求助
- 是否困惑

场景能提供：

- 人和机器人距离
- 当前任务是否失败
- 是否存在危险
- 是否有物体或其他人介入

人工复核能提供：

- 高质量标签
- 对自动标签的修正
- 用于后续训练分类器或标注模型

### 10.4 标签 JSON 示例

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

### 10.5 自动标注策略

第一阶段：

- 自动生成候选标签。
- 人工复核高价值样本。
- 不让自动情绪标签直接控制真机。

第二阶段：

- 用人工复核数据训练轻量分类器。
- 标签输出必须带置信度。
- 低置信度输出 `unknown`。

第三阶段：

- 标签作为 Flow Matching 条件输入。
- 生成动作后仍然要经过安全过滤。

## 11. 机器人内部状态

机器人内部状态是机器人自身可以真实读取的数据，不是推断。

建议字段：

```json
{
  "robot_state": {
    "battery_level": 0.82,
    "motor_temperature": "normal",
    "joint_risk": "low",
    "current_task_priority": "normal",
    "confidence": 0.76,
    "energy_mode": "normal",
    "safety_mode": "interactive_safe"
  }
}
```

内部状态对动作的影响：

| 内部状态 | 动作策略 |
| --- | --- |
| 电量低 | 降低动作幅度和速度 |
| 电机温度高 | 禁止大幅快速动作 |
| 关节风险高 | 使用保守动作或不动 |
| 任务优先级高 | 动作更明确、更短 |
| 识别置信度低 | 使用中性动作 |
| 人距离太近 | 减小手臂外展和速度 |

## 12. 自动标注与质量筛选

每个视频生成质量报告：

```json
{
  "video_id": "2026-05-09_subject001_wave_right_front",
  "valid_frame_ratio": 0.96,
  "min_upper_body_confidence": 0.78,
  "occlusion_events": [
    {
      "start_sec": 2.1,
      "end_sec": 2.4,
      "reason": "right_wrist_low_confidence"
    }
  ],
  "motion_energy": {
    "left_arm": 0.23,
    "right_arm": 0.81,
    "torso": 0.12,
    "head": 0.18
  },
  "recommended_use": "retargeting",
  "reject_reason": ""
}
```

筛选规则：

- 关键点连续丢失超过 0.3 秒，需要人工复核。
- 上半身有效帧比例低于 90%，默认不进入训练集。
- 手臂动作过快或跳变，默认只做离线分析。
- 低质量电视剧片段不进入真机动作库。

## 13. 重定向输出与安全过滤

每条动作输出：

```text
retargeted_csv/wave_right_001.csv
reports/wave_right_001_retargeting_report.json
retargeted_preview/wave_right_001_overlay.mp4
robot_preview/wave_right_001_robot_preview.mp4
```

安全过滤：

- 关节限位 clamp。
- 速度限制。
- 加速度限制。
- 低通滤波。
- 突变检测。
- 首次真机播放使用低刚度或低幅度。

默认安全参数：

```yaml
output_hz: 60
safety_scale: 0.5
max_velocity_scale: 0.5
waist_roll_limit_scale: 0.6
low_confidence_action: hold_or_neutral
```

## 14. Flow Matching 数据格式预留

后续训练数据格式：

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

训练时条件输入：

```text
动作意图 + 情绪表现 + 运动风格 + 人状态置信度 + 机器人内部状态 + 目标方向 + 时长
```

模型输出：

```text
机器人 11 维上半身关节轨迹
```

推理后必须经过：

```text
关节限位 -> 速度限制 -> 平滑 -> 仿真检查 -> ROS2 执行
```

## 15. 阶段划分

### 阶段 0：环境准备

目标：

- 工作站建立数据目录。
- 安装 Python 环境。
- 准备 URDF。
- 准备 5 条自采视频。

依赖：

```text
Python 3.10+
opencv-python
numpy
scipy
pandas
pyyaml
mediapipe
matplotlib
pinocchio 或 mujoco python
```

### 阶段 1：视频到 3D 骨架

目标：

- 输入 mp4。
- 输出 3D pose JSON。
- 输出骨架叠加视频。

命令示例：

```bash
python scripts/video_pose_extract.py \
  --input upper_body_motion_data/raw_videos/self_collected/wave_right.mp4 \
  --output upper_body_motion_data/pose_raw/mediapipe/wave_right.pose.json \
  --preview upper_body_motion_data/reports/wave_right_pose_preview.mp4
```

### 阶段 2：骨架清洗和质量标注

目标：

- 插值短时间缺失。
- 滤除抖动。
- 标记低质量帧。
- 输出 clean pose。

命令示例：

```bash
python scripts/pose_cleaning.py \
  --input upper_body_motion_data/pose_raw/mediapipe/wave_right.pose.json \
  --output upper_body_motion_data/pose_clean/wave_right.pose.clean.json
```

### 阶段 3：情绪、意图、风格自动标注

目标：

- 生成候选动作标签。
- 生成运动风格。
- 生成情绪表现候选。
- 人工复核高价值样本。

命令示例：

```bash
python scripts/auto_annotate_motion.py \
  --pose upper_body_motion_data/pose_clean/wave_right.pose.clean.json \
  --output upper_body_motion_data/annotations/wave_right.annotation.json
```

### 阶段 4：3D 骨架到机器人关节重定向

目标：

- 输入 clean pose。
- 输出 60 Hz 机器人 CSV。
- 输出重定向报告。
- 输出机器人预览。

命令示例：

```bash
python scripts/retarget_upper_body.py \
  --urdf robot/urdf/0421.urdf \
  --binding configs/human_robot_binding.yaml \
  --pose upper_body_motion_data/pose_clean/wave_right.pose.clean.json \
  --output upper_body_motion_data/retargeted_csv/wave_right.csv \
  --report upper_body_motion_data/reports/wave_right_retargeting_report.json
```

### 阶段 5：ROS2 回放验证

目标：

- 把 CSV 放入上半身回放目录。
- 用低刚度或仿真方式验证。
- 人工确认方向、节奏、幅度。

命令示例：

```bash
cp upper_body_motion_data/retargeted_csv/wave_right.csv ~/.ros/kh_upper_body_teach/motions/wave_right.csv
ros2 topic pub --once /upper_body_teach_node/command std_msgs/msg/String "{data: 'play wave_right 0.4'}"
```

### 阶段 6：动作库固化

目标：

- 只收录评分 4 到 5 的动作。
- 每条动作保留原视频、3D pose、annotation、CSV、report。

动作库结构：

```text
motion_library/
  wave_right/
    wave_right_001.csv
    wave_right_001.pose.clean.json
    wave_right_001.annotation.json
    wave_right_001.report.json
```

### 阶段 7：Flow Matching 动作生成

目标：

- 使用动作库训练条件式生成模型。
- 输入情景、人的可观察状态、机器人内部状态。
- 输出上半身关节轨迹。

训练重点：

- 不直接生成电机命令。
- 只生成目标关节轨迹。
- 轨迹必须经过安全过滤。

## 16. 验收标准

第一阶段完成标准：

- 至少 5 类动作可从视频自动转成机器人 CSV。
- 每类至少 3 条可用样本。
- 每条样本有 3D pose JSON。
- 每条样本有 annotation JSON。
- 每条样本有 retargeting report。
- CSV 能在仿真或 ROS2 回放系统中播放。
- 95% 以上帧满足关节限位和速度限制。
- 左右手没有镜像错乱。
- 肘部弯曲方向正确。
- 头部 yaw 方向正确。
- 动作无明显抖动和突变。

人工评分：

| 分数 | 含义 |
| --- | --- |
| 5 | 动作方向、节奏、幅度都对应，可进入训练库 |
| 4 | 基本对应，小误差可接受 |
| 3 | 能看出动作意图，但局部明显不准 |
| 2 | 左右、前后或关节方向有明显错误 |
| 1 | 不可用 |

只有 4 到 5 分进入训练库。

## 17. 近期开发任务清单

第一周：

- 建立 `upper_body_motion_data`。
- 拍摄 5 条自采视频。
- 跑通 MediaPipe 3D pose。
- 保存原始 3D pose JSON。
- 输出骨架叠加视频。

第二周：

- 实现骨架清洗。
- 建立人体 torso local frame。
- 实现动作质量报告。
- 实现基础情绪/意图/风格候选标签。

第三周：

- 编写人体到机器人绑定配置。
- 实现几何重定向。
- 输出 60 Hz CSV。
- 检查左右臂符号和肘部方向。

第四周：

- 接入 URDF limit。
- 接入速度和加速度限制。
- 输出机器人预览。
- 用 ROS2 回放验证 5 类动作。

第五周以后：

- 加入 IK 微调。
- 加入 MMPose 或多相机 3D。
- 扩展动作库。
- 导出 Flow Matching 训练数据。

## 18. 最重要的技术风险

### 18.1 单目 3D 深度不稳定

处理：

- 第一阶段只做表达动作。
- 使用相对 torso 坐标。
- 控制动作幅度。
- 后续用 RGB-D 或多相机提高质量。

### 18.2 人体与机器人自由度不一致

处理：

- 不追求手腕和手指。
- 用骨段方向误差做优化目标。
- 输出前必须经过 URDF 限位。

### 18.3 情绪标签误判

处理：

- 标签带置信度。
- 低置信度输出 unknown。
- 高价值样本人工复核。
- 真机动作不直接依赖低置信度情绪。

### 18.4 视频版权

处理：

- 优先自采。
- 公开视频检查授权。
- 电视剧片段只做内部研究，不进入商业训练库。

### 18.5 真机动作安全

处理：

- 所有动作先仿真。
- 速度、加速度、幅度限制。
- 低刚度首次播放。
- 保留急停接口。

## 19. 推荐工作站最小实现

最小 MVP 只需要 5 个脚本：

```text
video_pose_extract.py
pose_cleaning.py
auto_annotate_motion.py
retarget_upper_body.py
preview_retargeting.py
```

最小配置只需要 3 个文件：

```text
robot_upper_body_0421.yaml
human_robot_binding.yaml
emotion_label_schema.yaml
```

最小数据只需要：

```text
5 类动作 x 每类 3 条自采视频 = 15 条视频
```

先把 15 条视频稳定转成机器人 CSV，再考虑扩大数据集和训练 Flow Matching。

## 20. 结论

这条路线的关键不是先训练大模型，而是先把数据链路做硬：

```text
3D 骨架稳定
左右臂绑定正确
机器人关节限位正确
CSV 可回放
情绪标签带置信度
内部状态可控
动作先仿真再真机
```

只要第一阶段的视频到 3D 骨架再到机器人 CSV 稳定，后续 Flow Matching 才有可靠训练数据。否则生成模型会放大前面的错误，学到方向错、幅度错或不安全的动作。
