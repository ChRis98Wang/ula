# 04 3D 骨架提取与清洗管线

日期：2026-05-09

## 1. 为什么必须是 3D

2D 骨架只提供图像平面坐标：

```text
x, y
```

机器人重定向需要知道：

- 手臂是向前还是向侧面。
- 肘部平面如何旋转。
- 躯干是否转向。
- 头部相对身体朝向。

因此最低要求是：

```text
x, y, z + confidence
```

## 2. 3D 骨架来源

### 2.1 单目 3D

工具：

- MediaPipe Pose Landmarker
- MMPose 3D 模型

优点：

- 部署快。
- 适合 MVP。
- 能从普通视频启动。

缺点：

- 深度不稳定。
- 绝对尺度不可靠。
- 遮挡时容易跳变。

### 2.2 多相机 3D

流程：

```text
多相机同步视频
        -> 每个视角 2D 关键点
        -> 相机标定参数
        -> 三角化
        -> 真实 3D 骨架
```

优点：

- 几何上更可靠。
- 适合训练集。

缺点：

- 需要相机标定。
- 需要时间同步。
- 部署复杂。

### 2.3 RGB-D

流程：

```text
RGB 图像 + 深度图
        -> 2D 关键点
        -> 深度采样
        -> 米制 3D 坐标
```

优点：

- 对自采动作库很实用。
- 有真实深度。

缺点：

- 依赖深度相机。
- 深度空洞和反光会影响质量。

## 3. 推荐 MVP 流程

```text
mp4
  -> OpenCV 读帧
  -> MediaPipe Pose Landmarker
  -> 33 landmarks + world landmarks
  -> 提取上半身关键点
  -> 保存 pose_raw JSON
  -> 清洗、插值、滤波
  -> 保存 pose_clean JSON
```

## 4. 关键点集合

MVP 使用：

| 名称 | 用途 |
| --- | --- |
| `nose` | 头部 yaw 辅助 |
| `left_ear` | 头部 yaw |
| `right_ear` | 头部 yaw |
| `left_shoulder` | 左肩、躯干 |
| `right_shoulder` | 右肩、躯干 |
| `left_elbow` | 左上臂 |
| `right_elbow` | 右上臂 |
| `left_wrist` | 左前臂 |
| `right_wrist` | 右前臂 |
| `left_hip` | 骨盆/躯干 |
| `right_hip` | 骨盆/躯干 |

## 5. 原始 pose JSON

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

## 6. 清洗规则

### 6.1 低置信度帧

如果关键点 confidence 低于阈值：

```text
confidence < 0.5
```

则标记为低质量。

如果低质量连续时间小于 0.3 秒：

- 用前后帧插值。

如果超过 0.3 秒：

- 片段切断或标记人工复核。

### 6.2 滤波

推荐：

- 关键点坐标：中值滤波 + One Euro Filter 或低通滤波。
- 关节角：低通滤波。
- 速度：限幅。
- 加速度：限幅。

### 6.3 重采样

输入视频可能是 30 fps，输出 CSV 是 60 Hz。

流程：

```text
按 timestamp 插值 3D 骨架
        -> 60 Hz pose sequence
        -> retarget
        -> 60 Hz joint CSV
```

不要按帧号硬插值，必须使用 `time_sec`。

## 7. 人体局部坐标系

定义：

```text
pelvis_center = (left_hip + right_hip) / 2
shoulder_center = (left_shoulder + right_shoulder) / 2
torso_up = normalize(shoulder_center - pelvis_center)
torso_right = normalize(right_shoulder - left_shoulder)
torso_forward = normalize(cross(torso_right, torso_up))
```

所有手臂向量先转换到 torso local frame：

```text
right_upper_arm = right_elbow - right_shoulder
right_forearm = right_wrist - right_elbow
left_upper_arm = left_elbow - left_shoulder
left_forearm = left_wrist - left_elbow
```

## 8. 质量报告

```json
{
  "video_id": "2026-05-09_subject001_wave_right_front",
  "valid_frame_ratio": 0.96,
  "min_upper_body_confidence": 0.78,
  "mean_upper_body_confidence": 0.91,
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

## 9. 验收

3D 骨架管线验收：

- 原视频能生成 pose JSON。
- 能输出骨架叠加预览。
- 关键点没有明显左右错乱。
- 手腕短暂遮挡不会导致关节突变。
- 每帧都有 `time_sec`。
- 每个关键点都有置信度。
