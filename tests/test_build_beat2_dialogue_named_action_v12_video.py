from __future__ import annotations

import numpy as np

from tools.experimental import build_beat2_dialogue_named_action_v12_video as video


def _summary(action_id: str) -> dict:
    return {
        "action_id": action_id,
        "prompt_en": f"English {action_id}",
        "prompt_zh": f"中文 {action_id}",
    }


def test_pair_conditions_change_only_directive_role() -> None:
    rows = [
        {
            "clip_id": "dialogue",
            "fixed_split_assignment": "test",
            "dialogue_text": "fixed held-out dialogue",
            "dialogue_text_sha256": "d" * 64,
            "action_summary": _summary("unused"),
        },
        {
            "clip_id": "left",
            "fixed_split_assignment": "train",
            "action_summary": _summary("wave"),
        },
        {
            "clip_id": "right",
            "fixed_split_assignment": "train",
            "action_summary": _summary("turn"),
        },
    ]
    conditions = np.zeros((3, video.CONDITION_DIM), dtype=np.float32)
    conditions[0, video.DIALOGUE_LATENT_SLICE] = 1.0 / 8.0
    conditions[1, video.DIRECTIVE_LATENT_SLICE] = 1.0 / 8.0
    conditions[2, video.DIRECTIVE_LATENT_SLICE] = -1.0 / 8.0
    receipts, left, right, dialogue = video.build_pair_conditions(
        rows,
        conditions,
        action_pairs=[["wave", "turn"]],
        fixed_dialogue_clip_id="dialogue",
    )
    assert receipts[0]["left"]["action_id"] == "wave"
    assert dialogue["split"] == "test"
    assert np.array_equal(
        left[0, video.DIALOGUE_LATENT_SLICE],
        right[0, video.DIALOGUE_LATENT_SLICE],
    )
    assert np.array_equal(left[0, video.STYLE_SLICE], right[0, video.STYLE_SLICE])
    assert not np.array_equal(
        left[0, video.DIRECTIVE_LATENT_SLICE],
        right[0, video.DIRECTIVE_LATENT_SLICE],
    )


def test_ass_document_discloses_real_comparison_contract() -> None:
    timeline = [
        {
            "index": 1,
            "start_sec": 0.0,
            "end_sec": 4.0,
            "seed": 7,
            "raw_network_delta_rms_rad": 0.0,
            "left": _summary("wave"),
            "right": _summary("turn"),
        }
    ]
    document = video.build_ass_document(
        timeline,
        width=2000,
        height=720,
        pane_width=640,
        fixed_dialogue="the same dialogue",
    )
    assert "ONLY ACTION DESCRIPTION CHANGES" in document
    assert "NO GT / NO REFERENCE MOTION" in document
    assert "raw A/B delta RMS: 0.000000 rad" in document
    assert "中文 wave" in document
