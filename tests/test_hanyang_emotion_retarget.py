from __future__ import annotations

import numpy as np
import pytest

from upper_body_skeleton.data_source_registry import (
    HANYANG_EMOTIONAL_BODY_SOURCE_ID,
)
from upper_body_skeleton.hanyang_emotion_retarget import (
    DATASET_ID,
    DUPLICATE_EXCLUDED_STEMS,
    EVALUATION_PROTOCOL_ANOMALY_STEMS,
    EXACT_DUPLICATE_STEM_GROUPS,
    SOURCE_JOINTS,
    parse_clip_name,
    source_geometry_quality,
)


def _valid_source_positions(frames: int = 150) -> np.ndarray:
    positions = np.zeros((frames, len(SOURCE_JOINTS), 3), dtype=np.float64)
    index = {joint: offset for offset, joint in enumerate(SOURCE_JOINTS)}
    points = {
        "Hips": (1.0, 1.0, 1.0),
        "Spine": (1.0, 1.2, 1.0),
        "Spine1": (1.0, 1.5, 1.0),
        "Neck": (1.0, 1.7, 1.0),
        "Head": (1.0, 1.9, 1.0),
        "LeftShoulder": (0.85, 1.55, 1.0),
        "LeftArm": (0.75, 1.5, 1.0),
        "LeftForeArm": (0.55, 1.35, 1.0),
        "LeftHand": (0.35, 1.2, 1.0),
        "RightShoulder": (1.15, 1.55, 1.0),
        "RightArm": (1.25, 1.5, 1.0),
        "RightForeArm": (1.45, 1.35, 1.0),
        "RightHand": (1.65, 1.2, 1.0),
    }
    for joint, point in points.items():
        positions[:, index[joint], :] = point
    return positions


def test_dataset_id_is_the_closed_registry_id() -> None:
    assert DATASET_ID == HANYANG_EMOTIONAL_BODY_SOURCE_ID


def test_parse_clip_name_accepts_only_official_ranges() -> None:
    parsed = parse_clip_name("29_4_5_7.csv")
    assert parsed["source_stem"] == "29_4_5_7"
    assert parsed["participant_id"] == 29
    assert parsed["block_id"] == 4
    assert parsed["trial_id"] == 5
    assert parsed["emotion_index"] == 7
    assert parsed["fixed_split_assignment"] == "test"


@pytest.mark.parametrize(
    "name",
    (
        "0_1_1_1.csv",
        "30_1_1_1.csv",
        "1_0_1_1.csv",
        "1_5_1_1.csv",
        "1_1_0_1.csv",
        "1_1_6_1.csv",
        "1_1_1_0.csv",
        "1_1_1_8.csv",
        "1_1_1_1.txt",
        "1_1_1_1_extra.csv",
    ),
)
def test_parse_clip_name_rejects_out_of_contract_names(name: str) -> None:
    with pytest.raises(ValueError, match="Hanyang|invalid"):
        parse_clip_name(name)


def test_protocol_anomalies_and_duplicate_exclusions_are_frozen() -> None:
    assert EVALUATION_PROTOCOL_ANOMALY_STEMS == {
        "15_2_3_1",
        "20_2_2_2",
    }
    assert EXACT_DUPLICATE_STEM_GROUPS == (
        ("15_2_3_1", "15_2_4_1"),
        ("15_3_3_2", "15_3_4_2"),
        ("29_2_2_7", "29_2_3_7"),
        ("9_1_4_1", "9_3_4_1"),
    )
    assert DUPLICATE_EXCLUDED_STEMS == {
        "15_3_4_2",
        "29_2_3_7",
        "9_3_4_1",
    }
    assert "15_2_4_1" not in DUPLICATE_EXCLUDED_STEMS


def test_source_geometry_rejects_any_exact_zero_upper_body_point() -> None:
    positions = _valid_source_positions()
    valid_report = source_geometry_quality(positions)
    assert valid_report["upper_body_exact_zero_point_count"] == 0
    assert valid_report["quality_gate"][
        "no_exact_zero_upper_body_points_pass"
    ]
    assert valid_report["quality_gate"]["passed"]

    left_hand = SOURCE_JOINTS.index("LeftHand")
    positions[16, left_hand, :] = 0.0
    report = source_geometry_quality(positions)
    assert report["upper_body_exact_zero_point_count"] == 1
    assert report["upper_body_exact_zero_frame_count"] == 1
    assert report["upper_body_exact_zero_source_frames"] == [17]
    assert report["upper_body_exact_zero_counts_by_joint"] == {"LeftHand": 1}
    assert not report["quality_gate"][
        "no_exact_zero_upper_body_points_pass"
    ]
    assert not report["quality_gate"]["passed"]


def test_source_geometry_does_not_treat_lower_body_origin_as_upper_body_zero() -> None:
    report = source_geometry_quality(_valid_source_positions())
    assert report["upper_body_exact_zero_point_count"] == 0
    assert report["quality_gate"]["no_exact_zero_upper_body_points_pass"]
