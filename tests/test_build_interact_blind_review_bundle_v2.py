import os
from pathlib import Path
import stat

import pytest

from tools.gmr_v2.interact_bvh_adapter import INTERACT_NATIVE_AXIS_POLICY
from tools.human_motion_review.build_interact_blind_review_bundle_v2 import (
    AFFECT_PROTOCOL,
    ARC_ACTION_PROTOCOL,
    AXIS_PROTOCOL,
    _affect_record,
    _arc_action_record,
    _assert_complete_state,
    _axis_record,
    _same_interval,
)
from tools.human_motion_review.build_interact_blind_review_bundle import (
    materialize_video,
    sha256_file,
)


def test_public_records_bind_native_duration_and_remain_fail_closed(tmp_path):
    video = tmp_path / "anonymous.mp4"
    digest = "a" * 64
    records = (
        (_axis_record("axisv2_sample", video, digest), AXIS_PROTOCOL),
        (_arc_action_record("dyadv2_sample", video, digest), ARC_ACTION_PROTOCOL),
        (_affect_record("dyadv2_sample", video, digest), AFFECT_PROTOCOL),
    )

    for record, protocol in records:
        assert record["protocol_version"] == protocol
        assert record["temporal_unit"] == "complete_natural_interaction_arc"
        assert record["fixed_duration_window_used"] is False
        assert record["native_duration_preserved"] is True
        assert record["accepted_for_training"] is False
        assert "actor_id" not in record
        assert "source_bvh" not in record


def test_complete_state_requires_native_policy_and_matching_receipt():
    receipt_hash = "b" * 64
    state = {
        "artifact_kind": "interact_native_bvh_axis_smoke_v2_run_state",
        "status": "complete_pending_blind_review",
        "failure_count": 0,
        "axis_policy": INTERACT_NATIVE_AXIS_POLICY,
        "input_receipt_sha256": receipt_hash,
    }
    _assert_complete_state(
        state,
        artifact_kind="interact_native_bvh_axis_smoke_v2_run_state",
        receipt_hash=receipt_hash,
    )

    with pytest.raises(ValueError, match="native InterAct BVH policy"):
        _assert_complete_state(
            state | {"axis_policy": "legacy_wrong_parser"},
            artifact_kind="interact_native_bvh_axis_smoke_v2_run_state",
            receipt_hash=receipt_hash,
        )
    with pytest.raises(ValueError, match="catalog receipt hashes differ"):
        _assert_complete_state(
            state,
            artifact_kind="interact_native_bvh_axis_smoke_v2_run_state",
            receipt_hash="c" * 64,
        )


def test_axis_interval_comparison_ignores_diagnostic_duration_rounding():
    receipt = {
        "start_frame": 10,
        "end_frame_exclusive": 73,
        "frame_count": 63,
        "sample_span_sec": 2.066666666666667,
    }
    result = receipt | {"sample_span_sec": 2.0666666666666664}
    assert _same_interval(result, receipt) is True
    assert _same_interval(result | {"frame_count": 64}, receipt) is False


@pytest.mark.parametrize("alias_kind", ("hardlink", "symlink"))
def test_shared_materialize_video_detaches_existing_alias(tmp_path, alias_kind):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"independent InterAct video bytes")
    source_inode = source.stat().st_ino
    digest = sha256_file(source)
    target = tmp_path / "public" / "anonymous.mp4"
    target.parent.mkdir()
    if alias_kind == "hardlink":
        os.link(source, target)
    else:
        external = tmp_path / "external.mp4"
        external.write_bytes(source.read_bytes())
        target.symlink_to(external)

    materialize_video(source, target, digest)

    assert source.stat().st_ino == source_inode
    assert sha256_file(source) == digest
    assert not target.is_symlink()
    assert stat.S_ISREG(target.stat().st_mode)
    assert target.stat().st_nlink == 1
    assert not os.path.samefile(source, target)
    assert sha256_file(target) == digest
    assert target.stat().st_mode & 0o777 == 0o644
