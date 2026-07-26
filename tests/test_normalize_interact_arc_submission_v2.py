from __future__ import annotations

import pytest

from tools.human_motion_review.normalize_interact_arc_submission_v2 import (
    normalize_provenance,
)


def test_normalize_provenance_adds_planner_alias_without_changing_judgment() -> None:
    row = {
        "sample_id": "sample",
        "expression_completeness_result": "incomplete",
        "offset_status": "incomplete",
        "expansion_request": {"next_context_level": 6},
        "blind_review_provenance": {
            "arc_action_review_queue_sha256": "abc",
            "queue_record_sha256": "def",
        },
    }
    normalized = normalize_provenance(row, "abc")
    assert normalized["expression_completeness_result"] == "incomplete_requires_expansion"
    assert normalized["blind_review_provenance"]["arc_action_queue_sha256"] == "abc"
    assert "arc_action_queue_sha256" not in row["blind_review_provenance"]


def test_normalize_provenance_rejects_conflicting_queue_hashes() -> None:
    row = {
        "blind_review_provenance": {
            "arc_action_review_queue_sha256": "abc",
            "arc_action_queue_sha256": "changed",
        }
    }
    with pytest.raises(ValueError, match="queue hash binding mismatch"):
        normalize_provenance(row, "abc")
