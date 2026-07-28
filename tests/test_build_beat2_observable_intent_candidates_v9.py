from __future__ import annotations

from tools.human_motion_collection.build_beat2_observable_intent_candidates_v9 import (
    build_style_descriptor,
    route_candidate_intents,
)


def _features() -> dict:
    axis = {
        "roll": {"excursion_deg": 3.0, "mean_abs_speed_rad_s": 0.1, "full_band_sweep_count": 0},
        "pitch": {"excursion_deg": 4.0, "mean_abs_speed_rad_s": 0.1, "full_band_sweep_count": 0},
        "yaw": {"excursion_deg": 5.0, "mean_abs_speed_rad_s": 0.1, "full_band_sweep_count": 0},
    }
    return {
        "sample_span_sec": 2.5,
        "groups": {
            "left_arm": {"excursion_deg": 40.0, "mean_speed_rad_s": 0.8, "p95_speed_rad_s": 1.5},
            "right_arm": {"excursion_deg": 20.0, "mean_speed_rad_s": 0.3, "p95_speed_rad_s": 0.8},
            "head": {"excursion_deg": 7.0, "mean_speed_rad_s": 0.2, "p95_speed_rad_s": 0.5},
            "torso": {"excursion_deg": 6.0, "mean_speed_rad_s": 0.15, "p95_speed_rad_s": 0.4},
        },
        "arm": {
            "amplitude": "moderate",
            "laterality": "left",
            "energy_dominance": "left",
            "bilateral_temporally_coordinated": False,
            "continuity": "continuous",
            "regularly_repeated": False,
            "estimated_period_sec": None,
            "bilateral_speed_correlation": 0.2,
            "bilateral_energy_balance": 0.375,
        },
        "overall_motion": {"pace": "quick", "normalized_change_rate_hz": 0.55},
        "head_motion": "subtle",
        "head": {
            "axis_motion": {"dominant_axis": "yaw", "per_axis": axis},
            "repeated_pattern": {"pattern": "none"},
        },
        "torso_motion": "subtle",
        "torso": {
            "axis_motion": {"dominant_axis": "yaw", "per_axis": axis},
        },
    }


def test_style_descriptor_keeps_style_separate_from_intent() -> None:
    style = build_style_descriptor(_features())
    assert style["laterality"] == "left"
    assert style["head_engagement"] == "engaged"
    assert set(style["style_controls"]) == {"amplitude", "tempo", "energy"}
    assert all(-1.0 <= value <= 1.0 for value in style["style_controls"].values())
    assert "intent" not in style


def test_repeated_pitch_motion_routes_to_nod_candidate_only() -> None:
    features = _features()
    features["head"]["axis_motion"]["per_axis"]["pitch"].update(
        {"excursion_deg": 22.0, "full_band_sweep_count": 5}
    )
    routes = route_candidate_intents(
        {"semantic_event": {"category": "metaphoric"}, "prompt": "wave hello"},
        features,
        {"final_to_peak_ratio": 0.2, "peak_hold_fraction": 0.05},
    )
    nod = next(item for item in routes if item["candidate_intent_id"] == "agree_nod")
    assert nod["candidate_only"] is True
    assert nod["grants_training_admission"] is False
    assert all(item["candidate_intent_id"] != "wave_to_person" for item in routes)


def test_source_prompt_cannot_create_wave_candidate() -> None:
    routes = route_candidate_intents(
        {"semantic_event": {"category": "iconic"}, "prompt": "wave hello repeatedly"},
        _features(),
        {"final_to_peak_ratio": 0.7, "peak_hold_fraction": 0.02},
    )
    assert all(item["candidate_intent_id"] != "wave_to_person" for item in routes)
    assert all(item["candidate_intent_id"] != "explain_present" for item in routes)


def test_deictic_direction_stays_ambiguous_until_video_review() -> None:
    features = _features()
    features["arm"]["amplitude"] = "large"
    routes = route_candidate_intents(
        {"semantic_event": {"category": "deictic"}},
        features,
        {"final_to_peak_ratio": 0.3, "peak_hold_fraction": 0.4},
    )
    points = [item for item in routes if item["review_family"] == "deictic_direction_unknown"]
    assert {item["candidate_intent_id"] for item in points} == {
        "point_left",
        "point_right",
        "point_forward",
    }
    assert all(item["candidate_only"] for item in points)
