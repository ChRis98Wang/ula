import pytest

from tools.human_motion_review.normalize_beat_arc_submission_v8 import normalize_status


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("observed", "complete"),
        ("not_observed_needs_extension", "incomplete"),
        ("ambiguous", "ambiguous"),
    ],
)
def test_normalize_status_maps_only_declared_equivalents(source, expected):
    assert normalize_status(source) == expected


def test_normalize_status_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unknown arc phase status"):
        normalize_status("guessed")
