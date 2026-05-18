import csv
import json

import numpy as np

from upper_body_skeleton.language_action_index import (
    annotations_for_window,
    build_records,
    choose_intent_and_descriptions,
    npz_affect_stats,
    transcript_for_window,
)
from upper_body_skeleton.retarget_v2 import JOINT_ORDER


def test_transcript_and_annotations_are_selected_by_overlap():
    metadata = {
        "metadata:transcript": [
            {"start": 0.0, "end": 1.0, "transcript": "too early"},
            {"start": 1.5, "end": 3.0, "transcript": "in window"},
        ],
        "annotations:3P-V": [
            {"start_ts": 2.0, "end_ts": 2.5, "annotation": "crosses arms"},
            {"start_ts": 6.0, "end_ts": 7.0, "annotation": "late"},
        ],
    }

    assert transcript_for_window(metadata, 1.2, 3.5) == "in window"
    assert annotations_for_window(metadata, 1.2, 3.5) == {"annotations:3P-V": ["crosses arms"]}


def test_build_records_writes_jsonl_referencing_action_csv(tmp_path):
    extracted = tmp_path / "extracted" / "session"
    extracted.mkdir(parents=True)
    npz_path = extracted / "sample.npz"
    json_path = extracted / "sample.json"
    csv_path = tmp_path / "sample.v2_upper_body_joints.csv"
    monitor_path = tmp_path / "monitor.json"
    manifest_path = tmp_path / "manifest.csv"
    output_jsonl = tmp_path / "index.jsonl"

    np.savez(
        npz_path,
        **{
            "movement:emotion_arousal": np.ones((120, 1), dtype=np.float32) * 0.25,
            "movement:emotion_valence": np.ones((120, 1), dtype=np.float32) * 0.5,
        },
    )
    json_path.write_text(
        json.dumps(
            {
                "metadata:transcript": [{"start": 0.0, "end": 4.0, "transcript": "I am explaining."}],
                "annotations:3P-V": [{"start_ts": 0.0, "end_ts": 4.0, "annotation": "moves both arms"}],
            }
        ),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["time_sec"] + JOINT_ORDER)
        writer.writeheader()
        for index in range(120):
            row = {"time_sec": index / 30.0}
            row.update({joint: 0.0 for joint in JOINT_ORDER})
            writer.writerow(row)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample",
            "npz_path",
            "video_path",
            "status",
            "frame_count",
            "flagged_frame_count",
            "max_cross_body_intent",
            "max_yaw_under_response",
            "max_elbow_overfold",
            "skeleton_json",
            "joint_csv",
            "monitor_json",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "sample": "session/sample",
                "npz_path": str(npz_path),
                "video_path": str(extracted / "sample.mp4"),
                "status": "processed",
                "frame_count": 120,
                "flagged_frame_count": 0,
                "max_cross_body_intent": 0.0,
                "max_yaw_under_response": 0.0,
                "max_elbow_overfold": 0.0,
                "skeleton_json": "",
                "joint_csv": str(csv_path),
                "monitor_json": str(monitor_path),
            }
        )

    count = build_records(manifest_path, output_jsonl, window_sec=4.0, stride_sec=2.0)

    record = json.loads(output_jsonl.read_text(encoding="utf-8").strip())
    assert count == 1
    assert record["language_condition"]["raw_transcript"] == "I am explaining."
    assert record["language_condition"]["action_description"] == "moves both arms"
    assert record["labels"]["arousal"] == 0.25
    assert record["action"]["retarget_csv_path"] == str(csv_path)
    assert record["action"]["end_row"] == 120
    assert record["meta_semantics"]["body_motion"]["action_shape"] == [120, len(JOINT_ORDER)]
    assert record["meta_semantics"]["body_motion"]["joint_space"] == "v2_upper_body_15d"
    assert record["labels"]["communicative_intent"] == "explaining"
    assert record["labels"]["gesture_function"] == "representational"
    assert record["labels"]["emotion_trajectory"] == "friendly_sustained"
    assert record["labels"]["intensity"] == "low"
    assert record["labels"]["openness"] == "neutral"
    assert record["labels"]["tension"] == "low"
    assert record["labels"]["duration_sec"] == 4.0
    assert record["labels"]["transition"] == "end"
    assert record["meta_semantics"]["body_expression"]["gesture_function"] == "representational"
    assert record["meta_semantics"]["body_expression"]["openness"] == "neutral"
    assert "facial_action_units" not in record["meta_semantics"]
    assert "head_gaze" not in record["meta_semantics"]


def test_meta_semantics_prioritizes_visual_internal_state_and_rationale_annotations():
    annotations = {
        "annotations:3P-V": ["person folds both arms tightly across the chest"],
        "annotations:1P-IS": ["I feel guarded and unsure"],
        "annotations:3P-IS": ["the person appears hesitant"],
        "annotations:1P-R": ["because the question is uncomfortable"],
        "annotations:3P-R": ["the person is responding to a difficult topic"],
    }

    text = choose_intent_and_descriptions("I do not want to talk about that.", annotations)

    assert text["action_description"] == "person folds both arms tightly across the chest"
    assert "guarded" in text["mood_text"]
    assert "uncomfortable" in text["rationale_text"]
    assert text["meta_semantics"]["visual_element"] == "person folds both arms tightly across the chest"
    assert text["meta_semantics"]["internal_state"].startswith("I feel guarded")
    assert text["meta_semantics"]["rationale"].startswith("because the question")
    assert text["intent_label"] == "refusing"
    assert text["observed_affect"] == "uncertain"


def test_unknown_rationale_does_not_trigger_refusing_intent():
    text = choose_intent_and_descriptions(
        "The person is explaining the process.",
        {},
    )

    assert text["rationale_text"] == "rationale unknown"
    assert text["intent_label"] == "explaining"


def test_npz_affect_stats_ignores_face_and_head_features_for_body_only_dataset(tmp_path):
    npz_path = tmp_path / "sample.npz"
    scores = np.zeros((4, 8), dtype=np.float32)
    scores[:, 3] = 0.8
    np.savez(
        npz_path,
        **{
            "movement:emotion_arousal": np.ones((4, 1), dtype=np.float32) * 0.7,
            "movement:emotion_valence": np.ones((4, 1), dtype=np.float32) * -0.2,
            "movement:EmotionArousalToken": np.ones((4, 1), dtype=np.float32) * 5,
            "movement:EmotionValenceToken": np.ones((4, 1), dtype=np.float32) * 2,
            "movement:emotion_scores": scores,
            "movement:FAUToken": np.ones((4, 1), dtype=np.float32) * 9,
            "movement:FAUValue": np.ones((4, 24), dtype=np.float32) * 0.1,
            "movement:gaze_encodings": np.ones((4, 2), dtype=np.float32) * 0.2,
            "movement:head_encodings": np.ones((4, 3), dtype=np.float32) * -0.3,
        },
    )

    stats = npz_affect_stats(npz_path, 0.0, 4.0, fps=1.0)

    assert stats["arousal"] == 0.699999988079071
    assert stats["valence"] == -0.20000000298023224
    assert stats["arousal_token"] == 5
    assert stats["valence_token"] == 2
    assert "emotion_scores_mean" not in stats
    assert "dominant_emotion_index" not in stats
    assert "fau_token" not in stats
    assert "fau_mean" not in stats
    assert "gaze_mean" not in stats
    assert "head_mean" not in stats
    assert "movement:emotion_scores" not in stats["label_sources"]
    assert "movement:FAUToken" not in stats["label_sources"]
    assert "movement:FAUValue" not in stats["label_sources"]
    assert "movement:gaze_encodings" not in stats["label_sources"]
    assert "movement:head_encodings" not in stats["label_sources"]
