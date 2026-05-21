import csv

import numpy as np


def write_kimodo_prompt_csv(path):
    rows = [
        {
            "behavior_id": "Behavior.GreetingOwner01",
            "emotion_id": "happy",
            "emotion_zh_label": "开心",
            "prompt": "A human performer greets a familiar person with a small nod and wave, performed with happy emotion.",
            "negative_prompt": "text, robot",
            "output_name": "greetingowner01__happy.bvh",
            "output_format": "bvh_without_t_pose",
            "requires_bvh_without_t_pose": "True",
        },
        {
            "behavior_id": "Behavior.Alert",
            "emotion_id": "fear",
            "emotion_zh_label": "恐惧",
            "prompt": "A human performer snaps into an alert warning pose, performed with fear emotion.",
            "negative_prompt": "text, robot",
            "output_name": "alert__fear.bvh",
            "output_format": "bvh_without_t_pose",
            "requires_bvh_without_t_pose": "True",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_kimodo_prompt_csv_parser_indexes_behavior_and_emotion(tmp_path):
    from upper_body_skeleton.kimodo_semantics import (
        KIMODO_BEHAVIOR_IDS,
        KIMODO_EMOTION_IDS,
        load_kimodo_prompt_index,
    )

    csv_path = tmp_path / "kimodo_prompts.csv"
    write_kimodo_prompt_csv(csv_path)

    index = load_kimodo_prompt_index(csv_path)

    assert len(KIMODO_BEHAVIOR_IDS) == 27
    assert len(KIMODO_EMOTION_IDS) == 6
    assert index[("Behavior.GreetingOwner01", "happy")].emotion_zh_label == "开心"
    assert index[("Behavior.Alert", "fear")].output_name == "alert__fear.bvh"


def test_kimodo_condition_vector_exposes_behavior_emotion_family_and_robot_constraints():
    from upper_body_skeleton.kimodo_semantics import (
        KIMODO_CONDITION_EXTRA_DIM,
        KIMODO_CONDITION_SCHEMA_VERSION,
        KIMODO_EMOTION_IDS,
        build_kimodo_condition_extra,
    )

    extra = build_kimodo_condition_extra(
        behavior_id="Behavior.FingerHeart",
        emotion_id="happy",
        prompt="A performer forms a finger heart near the chest.",
    )

    assert extra.shape == (KIMODO_CONDITION_EXTRA_DIM,)
    assert extra.sum() >= 4.0
    assert extra[KIMODO_EMOTION_IDS.index("happy") + 27] == 1.0
    assert extra[-3] == KIMODO_CONDITION_SCHEMA_VERSION
    assert extra[-2] == 1.0
    assert extra[-1] == 0.0


def test_kimodo_condition_vector_falls_back_from_free_text():
    from upper_body_skeleton.kimodo_semantics import infer_kimodo_ids_from_text

    assert infer_kimodo_ids_from_text("开心地向主人挥手") == ("Behavior.GreetingOwner01", "happy")
    assert infer_kimodo_ids_from_text("恐惧地做停止警告动作") == ("Behavior.Alert", "fear")
    assert infer_kimodo_ids_from_text("平静地站着") == ("Behavior.IdleQuiet", "neutral")


def test_kimodo_extra_differs_between_specific_behaviors():
    from upper_body_skeleton.kimodo_semantics import build_kimodo_condition_extra

    greeting = build_kimodo_condition_extra("Behavior.GreetingOwner01", "happy")
    alert = build_kimodo_condition_extra("Behavior.Alert", "happy")

    assert not np.array_equal(greeting, alert)
