import math

import numpy as np

from tools.gmr_v2.safety_monotonic_retime_v1 import (
    ALGORITHM_CONTRACT_SHA256,
    MAX_SLOWDOWN_RATIO,
    minimum_velocity_safety_retime,
    trajectory_sha256,
)


JOINTS = ("joint_a", "joint_b")
LIMITS = {name: (-2.0, 2.0) for name in JOINTS}


def _run(raw):
    return minimum_velocity_safety_retime(
        raw,
        fps=30.0,
        max_velocity_rad_s=3.0,
        smoothing_window=3,
        joint_order=JOINTS,
        joint_limits=LIMITS,
    )


def test_static_input_preserves_native_grid_and_endpoints():
    raw = np.zeros((6, 2), dtype=np.float64)

    safe, key_times, output_times, audit = _run(raw)

    assert safe.shape == raw.shape
    assert np.array_equal(safe[0], raw[0])
    assert np.array_equal(safe[-1], raw[-1])
    assert np.all(np.diff(key_times) > 0)
    assert np.allclose(output_times, np.arange(6) / 30)
    assert audit["retime_ratio"] == 1.0
    assert audit["triggering_joints"] == []
    assert audit["algorithm_contract_sha256"] == ALGORITHM_CONTRACT_SHA256
    assert audit["output_trajectory_sha256"] == trajectory_sha256(safe)


def test_velocity_violation_uses_minimum_uniform_slowdown_without_crop():
    raw = np.zeros((6, 2), dtype=np.float64)
    raw[3:, 0] = 0.11

    safe, key_times, output_times, audit = _run(raw)

    required_frames = (
        math.ceil(audit["required_continuous_sample_span_sec"] * 30 - 1e-12)
        + 1
    )
    assert len(safe) == required_frames == 7
    assert len(key_times) == len(raw)
    assert len(output_times) == len(safe)
    assert np.array_equal(safe[0], raw[0])
    assert np.array_equal(safe[-1], raw[-1])
    assert audit["triggering_joints"] == ["joint_a"]
    assert audit["post_velocity_pass"] is True
    assert max(audit["post_retime_max_velocity_rad_s_by_joint"].values()) <= 3.0 + 1e-6
    assert audit["cropped"] is False
    assert audit["tiled"] is False
    assert audit["target_duration_sec"] is None


def test_slowdown_over_bound_is_generated_but_fail_closed_for_quarantine():
    raw = np.zeros((6, 2), dtype=np.float64)
    raw[3:, 0] = 0.5

    safe, _key_times, _output_times, audit = _run(raw)

    assert len(safe) / len(raw) > MAX_SLOWDOWN_RATIO
    assert audit["slowdown_ratio_pass"] is False
    assert audit["blind_review_must_use_retimed_output"] is True
