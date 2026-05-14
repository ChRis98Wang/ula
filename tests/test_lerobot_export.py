import csv
import json

import pyarrow.parquet as pq

from upper_body_skeleton.lerobot_export import export_lerobot_dataset
from upper_body_skeleton.retarget_v2 import JOINT_ORDER


def write_joint_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["time_sec"] + JOINT_ORDER)
        writer.writeheader()
        for frame_index in range(rows):
            row = {"time_sec": frame_index / 30.0}
            row.update({joint: float(frame_index + joint_index) for joint_index, joint in enumerate(JOINT_ORDER)})
            writer.writerow(row)


def test_export_lerobot_dataset_writes_v3_parquet_layout(tmp_path):
    csv_path = tmp_path / "sample.v2_upper_body_joints.csv"
    jsonl_path = tmp_path / "language_action_index.body_partial.jsonl"
    out_dir = tmp_path / "lerobot"
    write_joint_csv(csv_path, rows=4)
    record = {
        "sample_id": "sample__0_4",
        "source": {
            "dataset": "seamless_interaction_50g",
            "video_path": "/video.mp4",
            "json_path": "/sample.json",
            "npz_path": "/sample.npz",
            "retarget_csv_path": str(csv_path),
            "monitor_report_path": "/monitor.json",
        },
        "time_window": {"start_sec": 0.0, "end_sec": 4.0 / 30.0, "fps": 30.0},
        "language_condition": {
            "raw_transcript": "hello",
            "scenario_description": "hello",
            "action_description": "waves both hands",
            "intent_text": "greeting",
            "mood_text": "friendly",
            "rationale_text": "greeting",
            "instruction_variants": ["greeting; gesture style: relaxed; affect: friendly"],
        },
        "labels": {
            "intent": "greeting",
            "observed_affect": "friendly",
            "motion_style": "relaxed",
            "arousal": 0.2,
            "valence": 0.5,
            "arousal_token": 1,
            "valence_token": 2,
            "motion_energy": 0.03,
            "label_sources": ["annotations:3P-V"],
        },
        "meta_semantics": {
            "semantic_gesture": "waving",
            "body_motion": {
                "joint_space": "v2_upper_body_15d",
                "action_shape": [4, 15],
                "joint_order": JOINT_ORDER,
                "motion_style": "relaxed",
                "motion_energy": 0.03,
                "semantic_gesture": "waving",
            },
        },
        "action": {
            "retarget_csv_path": str(csv_path),
            "start_row": 0,
            "end_row": 4,
            "fps": 30.0,
            "joint_order": JOINT_ORDER,
        },
        "quality": {
            "accepted_for_training": True,
            "frame_count": 4,
            "flagged_frame_count": 0,
            "max_elbow_overfold": 0.0,
            "max_yaw_under_response": 0.0,
        },
    }
    jsonl_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    summary = export_lerobot_dataset(jsonl_path, out_dir, rows_per_file=10)

    assert summary["total_episodes"] == 1
    assert summary["total_frames"] == 4
    assert (out_dir / "data" / "chunk-000" / "file-000.parquet").exists()
    assert (out_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet").exists()
    assert (out_dir / "meta" / "semantic_index.parquet").exists()
    info = json.loads((out_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["codebase_version"] == "v3.0"
    assert info["robot_type"] == "v2_upper_body_15d"
    assert info["features"]["observation.state"]["shape"] == [15]
    table = pq.read_table(out_dir / "data" / "chunk-000" / "file-000.parquet")
    assert table.num_rows == 4
    assert table.column_names[:7] == [
        "index",
        "episode_index",
        "frame_index",
        "timestamp",
        "task_index",
        "observation.state",
        "action",
    ]
    rows = table.to_pylist()
    assert rows[0]["observation.state"] == [float(i) for i in range(15)]
    assert rows[0]["action"] == [float(i + 1) for i in range(15)]
    assert rows[-1]["next.done"] is True
    episode_rows = pq.read_table(out_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pylist()
    assert episode_rows[0]["sample_id"] == "sample__0_4"
    assert episode_rows[0]["length"] == 4
