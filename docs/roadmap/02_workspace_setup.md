# 02 工作站环境与目录结构

日期：2026-05-09

## 1. 推荐目录结构

建议不要把大视频直接放进代码仓库。工作站上分成数据目录和代码目录。

数据目录：

```text
upper_body_motion_data/
  raw_videos/
    self_collected/
    public/
    restricted_tv_clips/
  extracted_frames/
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

代码目录：

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
```

## 2. Python 环境

建议：

```text
Python 3.10 或 3.11
```

基础依赖：

```text
opencv-python
numpy
scipy
pandas
pyyaml
mediapipe
matplotlib
tqdm
```

机器人和仿真依赖可选：

```text
pin
mujoco
meshcat
```

如果使用 MMPose，需要单独按 OpenMMLab 文档配置 PyTorch、MMCV、MMPose。

## 3. 推荐安装命令

MVP 最小环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install opencv-python numpy scipy pandas pyyaml mediapipe matplotlib tqdm
```

如果工作站有 CUDA，并且准备做 MMPose，先确认：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

## 4. URDF 准备

需要把机器人 URDF 和 mesh 放到工作站可访问路径。

建议：

```text
robot/
  urdf/
    0421.urdf
  meshes/
    ...
```

配置里使用相对路径或绝对路径均可，但要保持一致。

## 5. ROS2 回放准备

如果工作站或机器人端有 ROS2 上半身回放节点，动作 CSV 放到：

```text
~/.ros/kh_upper_body_teach/motions/
```

回放命令：

```bash
ros2 topic pub --once /upper_body_teach_node/command std_msgs/msg/String "{data: 'play wave_right 0.4'}"
```

第一版建议先在仿真或低刚度模式下回放，不要直接高刚度真机执行。

## 6. 数据命名规范

视频命名：

```text
YYYY-MM-DD_subjectXXX_action_view.mp4
```

示例：

```text
2026-05-09_subject001_wave_right_front.mp4
2026-05-09_subject001_wave_right_left45.mp4
2026-05-09_subject001_wave_right_right45.mp4
```

动作 ID：

```text
action_subject_view_index
```

示例：

```text
wave_right_subject001_front_001
```

## 7. 最小启动流程

```bash
mkdir -p upper_body_motion_data/raw_videos/self_collected
mkdir -p upper_body_motion_data/pose_raw/mediapipe
mkdir -p upper_body_motion_data/pose_clean
mkdir -p upper_body_motion_data/annotations
mkdir -p upper_body_motion_data/retargeted_csv
mkdir -p upper_body_motion_data/reports
mkdir -p upper_body_motion_data/retargeted_preview
mkdir -p upper_body_motion_data/robot_preview
```

然后按顺序实现：

```text
video_pose_extract.py
pose_cleaning.py
auto_annotate_motion.py
retarget_upper_body.py
preview_retargeting.py
```
