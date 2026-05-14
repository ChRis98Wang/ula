# 03 视频数据获取规范

日期：2026-05-09

## 1. 数据来源优先级

推荐优先级：

1. 自采视频
2. 公开授权视频
3. 电视剧/电影片段，仅限内部研究和标签体系设计

第一阶段强烈建议自采，因为动作可控、授权清晰、画面稳定。

## 2. 自采视频规范

拍摄条件：

- 单人主体。
- 上半身、双肩、双肘、双腕、头部、髋部上缘尽量可见。
- 背景简单。
- 光照稳定。
- 摄像机固定。
- 不要快速变焦。
- 不要多人遮挡。

参数：

- 分辨率：1080p 或更高。
- 帧率：30 fps 以上。
- 时长：每条 5 到 15 秒。
- 动作幅度：先限制在机器人安全范围内。

视角：

- 正面。
- 左前 45 度。
- 右前 45 度。

第一批动作：

| 动作 | 说明 | 风险 |
| --- | --- | --- |
| `wave_right` | 右手挥手 | 腕部缺失不影响 MVP |
| `wave_left` | 左手挥手 | 检查左右绑定 |
| `point_right` | 右手指向 | 单目深度可能不稳 |
| `point_left` | 左手指向 | 检查左臂符号 |
| `arms_open` | 双臂欢迎 | 肩 roll 限位 |
| `deny` | 双手拒绝 | 节奏和幅度 |
| `thinking` | 思考姿态 | 手靠脸可能遮挡 |
| `attention_shift` | 头和躯干转向 | 头 yaw 校准 |

## 3. 公开视频规范

公开数据进入项目前必须记录：

```json
{
  "source_type": "public_video",
  "url": "",
  "license": "",
  "download_date": "2026-05-09",
  "allowed_use": ["internal_research"],
  "notes": ""
}
```

筛选条件：

- 授权清晰。
- 单人动作为主。
- 上半身清楚。
- 镜头连续。
- 动作能被机器人安全模仿。

## 4. 电视剧/电影片段规范

电视剧片段不建议作为训练主数据。

允许用途：

- 观察动作类别。
- 设计情绪/意图标签。
- 离线测试骨架提取和重定向鲁棒性。

不建议用途：

- 直接进入训练集。
- 分发给第三方。
- 商业模型训练。
- 直接真机执行。

必须记录：

```json
{
  "source_type": "tv_clip",
  "source_name": "example_show",
  "episode": "S01E01",
  "start_time_sec": 123.4,
  "end_time_sec": 130.2,
  "license_status": "internal_review_only",
  "allowed_use": ["offline_test", "label_schema_design"]
}
```

## 5. 采集元数据

每条视频一个 metadata JSON：

```json
{
  "video_id": "2026-05-09_subject001_wave_right_front",
  "file_path": "raw_videos/self_collected/2026-05-09_subject001_wave_right_front.mp4",
  "source_type": "self_collected",
  "subject_id": "subject001",
  "action_label": "wave_right",
  "camera_view": "front",
  "fps": 30.0,
  "width": 1920,
  "height": 1080,
  "duration_sec": 8.4,
  "license_status": "owned",
  "notes": ""
}
```

## 6. 第一批数据量

最小可用：

```text
5 类动作 x 每类 3 条视频 = 15 条视频
```

推荐第一批：

```text
8 类动作 x 每类 3 个视角 x 每个视角 3 条 = 72 条视频
```

不要一开始采太大规模。先用 15 条视频把流程打通，再扩大。
