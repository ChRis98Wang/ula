import csv
import json
import math

import pyarrow.parquet as pq

from upper_body_skeleton.kimodo_parquet_export import (
    build_kimodo_jsonl,
    export_kimodo_parquet_dataset,
    parse_kimodo_csv_path,
)
from upper_body_skeleton.kimodo_semantics import KIMODO_BEHAVIOR_IDS, KIMODO_EMOTION_IDS
from upper_body_skeleton.retarget_v2 import JOINT_ORDER


def write_prompt_csv(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "behavior_id",
                "emotion_id",
                "emotion_zh_label",
                "prompt",
                "negative_prompt",
                "output_name",
                "output_format",
                "requires_bvh_without_t_pose",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "behavior_id": "Behavior.GreetingOwner03",
                "emotion_id": "happy",
                "emotion_zh_label": "开心",
                "prompt": "A human performer greets the owner with a happy wave.",
                "negative_prompt": "text, watermark",
                "output_name": "greetingowner03__happy.bvh",
                "output_format": "bvh_without_t_pose",
                "requires_bvh_without_t_pose": "True",
            }
        )


def write_kimodo_joint_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "Frame",
            "root_translateX",
            "root_translateY",
            "root_translateZ",
            "root_rotateX",
            "root_rotateY",
            "root_rotateZ",
        ] + [f"{joint}_dof" for joint in JOINT_ORDER]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for frame_index in range(rows):
            row = {
                "Frame": frame_index,
                "root_translateX": 0.0,
                "root_translateY": 0.0,
                "root_translateZ": 0.0,
                "root_rotateX": 0.0,
                "root_rotateY": 0.0,
                "root_rotateZ": 0.0,
            }
            row.update({f"{joint}_dof": frame_index + joint_index for joint_index, joint in enumerate(JOINT_ORDER)})
            writer.writerow(row)


def test_parse_kimodo_csv_path_extracts_behavior_emotion_and_sample_index(tmp_path):
    csv_path = tmp_path / "csv" / "greetingowner03" / "happy" / "greetingowner03__happy__sample_07.csv"

    parsed = parse_kimodo_csv_path(csv_path, tmp_path / "csv")

    assert parsed.behavior_slug == "greetingowner03"
    assert parsed.emotion_id == "happy"
    assert parsed.prompt_stem == "greetingowner03__happy"
    assert parsed.sample_index == 7


def test_build_kimodo_jsonl_matches_prompt_rows_by_output_stem(tmp_path):
    kimodo_root = tmp_path / "Kimodo_CSV"
    prompt_csv = kimodo_root / "kimodo_action_emotion_prompts.csv"
    motion_csv = kimodo_root / "csv" / "greetingowner03" / "happy" / "greetingowner03__happy__sample_00.csv"
    out_jsonl = tmp_path / "index.jsonl"
    write_prompt_csv(prompt_csv)
    write_kimodo_joint_csv(motion_csv, rows=4)

    summary = build_kimodo_jsonl(kimodo_root, out_jsonl)

    assert summary["records"] == 1
    record = json.loads(out_jsonl.read_text(encoding="utf-8").strip())
    assert record["sample_id"] == "greetingowner03__happy__sample_00"
    assert record["labels"]["behavior_id"] == "Behavior.GreetingOwner03"
    assert record["labels"]["emotion_id"] == "happy"
    assert record["language_condition"]["instruction_variants"] == [
        "A human performer greets the owner with a happy wave."
    ]
    assert record["action"]["end_row"] == 4
    assert record["action"]["joint_order"] == JOINT_ORDER


def test_export_kimodo_parquet_dataset_writes_training_compatible_parquet(tmp_path):
    kimodo_root = tmp_path / "Kimodo_CSV"
    prompt_csv = kimodo_root / "kimodo_action_emotion_prompts.csv"
    motion_csv = kimodo_root / "csv" / "greetingowner03" / "happy" / "greetingowner03__happy__sample_00.csv"
    output_dir = tmp_path / "lerobot_kimodo"
    write_prompt_csv(prompt_csv)
    write_kimodo_joint_csv(motion_csv, rows=4)

    summary = export_kimodo_parquet_dataset(kimodo_root, output_dir)

    assert summary["jsonl"]["records"] == 1
    assert summary["parquet"]["total_episodes"] == 1
    semantic_rows = pq.read_table(output_dir / "meta" / "semantic_index.parquet").to_pylist()
    assert semantic_rows[0]["language_instruction"] == "A human performer greets the owner with a happy wave."
    assert semantic_rows[0]["behavior_id"] == "Behavior.GreetingOwner03"
    assert semantic_rows[0]["emotion_id"] == "happy"
    assert "Behavior.GreetingOwner03" in KIMODO_BEHAVIOR_IDS
    assert "happy" in KIMODO_EMOTION_IDS
    data_rows = pq.read_table(output_dir / "data" / "chunk-000" / "file-000.parquet").to_pylist()
    assert len(data_rows) == 4
    assert len(data_rows[0]["observation.state"]) == len(JOINT_ORDER)
    assert math.isclose(data_rows[0]["observation.state"][1], math.radians(1.0), rel_tol=1e-6)
    assert data_rows[-1]["next.done"] is True
