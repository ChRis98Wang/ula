# 11 实施清单

日期：2026-05-09

## 1. 第一周：数据和 3D 骨架

- [ ] 建立 `upper_body_motion_data` 目录。
- [ ] 建立 `upper_body_motion_workspace` 目录。
- [ ] 安装 Python 环境。
- [ ] 准备机器人 URDF 和 mesh。
- [ ] 自采 5 条基础动作视频。
- [ ] 实现 `video_pose_extract.py`。
- [ ] 输出 raw pose JSON。
- [ ] 输出骨架叠加预览视频。
- [ ] 检查 3D 关键点方向。

## 2. 第二周：清洗和标注

- [ ] 实现 `pose_cleaning.py`。
- [ ] 实现短缺失插值。
- [ ] 实现低通滤波。
- [ ] 实现 60 Hz 重采样。
- [ ] 实现质量报告。
- [ ] 实现 `auto_annotate_motion.py`。
- [ ] 输出 intent 候选。
- [ ] 输出 motion_style 候选。
- [ ] 输出 observed_affect 候选和 confidence。

## 3. 第三周：绑定和重定向

- [ ] 编写 `robot_upper_body_0421.yaml`。
- [ ] 编写 `human_robot_binding.yaml`。
- [ ] 实现 torso local frame。
- [ ] 实现人体骨段方向计算。
- [ ] 实现几何重定向。
- [ ] 输出 60 Hz CSV。
- [ ] 检查右臂标定动作。
- [ ] 检查左臂标定动作。
- [ ] 检查头部 yaw。
- [ ] 检查腰 yaw / roll。

## 4. 第四周：安全和回放

- [ ] 从 URDF 或配置读取 joint limits。
- [ ] 实现 joint limit clamp。
- [ ] 实现速度限制。
- [ ] 实现加速度警告。
- [ ] 实现 retargeting report。
- [ ] 实现 robot preview。
- [ ] 把 CSV 放到 ROS2 motions 目录。
- [ ] 低刚度回放。
- [ ] 人工评分。

## 5. 第五周：动作库

- [ ] 每类动作至少 3 条可用样本。
- [ ] 每条样本有 pose clean JSON。
- [ ] 每条样本有 annotation JSON。
- [ ] 每条样本有 CSV。
- [ ] 每条样本有 report。
- [ ] 只收录 4 分以上动作。
- [ ] 固化 motion_library 目录结构。

## 6. 后续：Flow Matching

- [ ] 实现 `export_flow_matching_dataset.py`。
- [ ] 将 CSV 转为 q / dq。
- [ ] 合并 condition 标签。
- [ ] 做 joint limit 归一化。
- [ ] 训练小型条件式生成模型。
- [ ] 生成动作后走安全过滤。
- [ ] 仿真验证。
- [ ] 低刚度真机验证。

## 7. 每条动作的完成标准

一条动作进入动作库前必须有：

- [ ] 原始视频。
- [ ] 3D raw pose。
- [ ] 3D clean pose。
- [ ] annotation。
- [ ] retargeted CSV。
- [ ] retargeting report。
- [ ] 预览视频。
- [ ] 人工评分。
- [ ] 来源和授权记录。

## 8. 不要提前做的事

- [ ] 不要先大规模爬电视剧。
- [ ] 不要在重定向不稳定时训练 Flow Matching。
- [ ] 不要把低置信度情绪直接用于真机动作。
- [ ] 不要跳过仿真。
- [ ] 不要直接生成电机命令。
