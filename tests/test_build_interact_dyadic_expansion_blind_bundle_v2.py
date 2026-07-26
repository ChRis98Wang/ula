import json

import pytest

from tools.human_motion_review.build_interact_dyadic_expansion_blind_bundle_v2 import (
    PLAN_KIND,
    RUN_KIND,
    build_bundle,
    validate_state,
)
from tools.human_motion_review.build_interact_blind_review_bundle_v2 import (
    _affect_record,
    _arc_action_record,
)
from tools.human_motion_review.build_interact_arc_expansion_plan_v2 import sha256_file


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _bundle_fixture(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"round-N InterAct expansion video")
    video_hash = sha256_file(source)
    request = {
        "sample_id": "dyadv2_base_sample",
        "turn_id": "private_turn_id",
        "reviewed_context_level": 1,
        "requested_context_level": 2,
        "requested_interval": {
            "start_frame": 0,
            "end_frame_exclusive": 3,
            "frame_count": 3,
        },
        "plan_record_sha256": "a" * 64,
    }
    requests = tmp_path / "expansion_requests.jsonl"
    _write_json(requests, request)
    plan = tmp_path / "plan_summary.json"
    _write_json(
        plan,
        {
            "artifact_kind": PLAN_KIND,
            "accepted_for_training_count": 0,
            "outputs": {
                "expansion_requests": {
                    "records": 1,
                    "sha256": sha256_file(requests),
                }
            },
        },
    )
    run_state = tmp_path / "run_state.json"
    _write_json(
        run_state,
        {
            "artifact_kind": RUN_KIND,
            "status": "complete_pending_repeat_blind_review",
            "failure_count": 0,
            "accepted_for_training": False,
            "inputs": {
                "plan_summary_sha256": sha256_file(plan),
                "expansion_requests_sha256": sha256_file(requests),
            },
            "results": {
                "expansion": {
                    "sample_id": request["sample_id"],
                    "status": "rendered_pending_repeat_blind_review",
                    "source_interval": [0, 3],
                    "requested_context_level": 2,
                    "fixed_duration_window_used": False,
                    "accepted_for_training": False,
                    "video": str(source),
                    "video_sha256": video_hash,
                    "video_validation": {"passed": True, "decoded_frames": 3},
                }
            },
        },
    )
    secret = tmp_path / "bundle_secret.json"
    _write_json(secret, {"secret_hex": "55" * 32})
    return {
        "plan_summary": plan,
        "expansion_requests": requests,
        "run_state": run_state,
        "output_root": tmp_path / "public_bundle",
        "hidden_root": tmp_path / "private_bundle",
        "secret_file": secret,
    }, source


def test_expansion_public_records_are_native_variable_length_and_fail_closed(tmp_path):
    video = tmp_path / "video.mp4"
    for record in (
        _arc_action_record("sample", video, "a" * 64),
        _affect_record("sample", video, "a" * 64),
    ):
        assert record["native_duration_preserved"] is True
        assert record["fixed_duration_window_used"] is False
        assert record["accepted_for_training"] is False


def test_expansion_state_must_bind_plan_and_requests(tmp_path):
    plan = tmp_path / "plan.json"
    requests = tmp_path / "requests.jsonl"
    plan.write_text("{}\n")
    requests.write_text("{}\n")
    state = {
        "artifact_kind": RUN_KIND,
        "status": "complete_pending_repeat_blind_review",
        "failure_count": 0,
        "accepted_for_training": False,
        "inputs": {
            "plan_summary_sha256": sha256_file(plan),
            "expansion_requests_sha256": sha256_file(requests),
        },
    }
    validate_state(state, plan_summary=plan, expansion_requests=requests)


def test_expansion_bundle_rejects_stale_sensitive_public_file(tmp_path):
    arguments, source = _bundle_fixture(tmp_path)
    source_inode = source.stat().st_ino
    build_bundle(**arguments)
    public_root = arguments["output_root"] / "public"
    public_video = next((public_root / "videos").glob("*.mp4"))
    assert public_video.stat().st_nlink == 1
    assert not public_video.is_symlink()
    assert source.stat().st_ino == source_inode

    stale = public_root / "sample_mapping.jsonl"
    stale.write_text(
        '{"official_emotion":"fear","turn_id":"private"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="stale unexpected public bundle entries"):
        build_bundle(**arguments)
