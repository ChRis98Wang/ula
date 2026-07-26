from __future__ import annotations

import pytest

from tools.human_motion_review.finalize_beat_arc_decisions_v8 import _validate_decision


def _decision() -> dict:
    return {
        "observable_description": "The robot raises an arm and returns it to rest.",
        "onset_status": "complete",
        "onset_evidence_frame": 1,
        "onset_basis": "The arm begins rising.",
        "apex_status": "complete",
        "apex_evidence_frame": 5,
        "apex_basis": "The arm reaches its highest point.",
        "offset_status": "complete",
        "offset_evidence_frame": 9,
        "offset_basis": "The arm returns to rest.",
    }


def test_validate_decision_accepts_complete_native_arc() -> None:
    _validate_decision("sample", _decision(), 10)


def test_validate_decision_rejects_out_of_range_evidence() -> None:
    decision = _decision()
    decision["offset_evidence_frame"] = 10
    with pytest.raises(ValueError, match="invalid offset"):
        _validate_decision("sample", decision, 10)


def test_validate_decision_rejects_unreviewed_extra_fields() -> None:
    decision = _decision()
    decision["emotion"] = "happy"
    with pytest.raises(ValueError, match="unexpected fields"):
        _validate_decision("sample", decision, 10)
