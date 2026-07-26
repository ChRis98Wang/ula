from collections import Counter

import numpy as np

from tools.human_motion_review.audit_expression_turn_catalog_v8 import (
    joint_counts,
    npz_array_shape,
    proportional_quotas,
)


def test_proportional_quotas_uses_deterministic_largest_remainder():
    counts = Counter({("short", "single"): 7, ("long", "multi"): 2, ("long", "pair"): 1})

    assert proportional_quotas(counts, 6) == {
        "long|multi": 1,
        "long|pair": 1,
        "short|single": 4,
    }


def test_joint_counts_preserves_duration_event_pairing():
    records = [
        {"duration_band": "short", "event_count_band": "single"},
        {"duration_band": "short", "event_count_band": "single"},
        {"duration_band": "long", "event_count_band": "multi"},
    ]

    assert joint_counts(records) == {"long|multi": 1, "short|single": 2}


def test_npz_array_shape_reads_header_without_loading_array(tmp_path):
    path = tmp_path / "motion.npz"
    np.savez_compressed(path, poses=np.zeros((37, 165), dtype=np.float32))

    assert npz_array_shape(path, "poses") == (37, 165)
