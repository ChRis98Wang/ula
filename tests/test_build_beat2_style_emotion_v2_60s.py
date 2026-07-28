import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.experimental import build_beat2_style_emotion_v2_60s as video
from upper_body_skeleton.beat2_condition_control import QwenStyleHead


def test_checked_in_config_targets_global54_v6_candidate_only():
    config = json.loads(
        (
            Path(__file__).parents[1]
            / "configs"
            / "beat2_style_emotion_v2_60s.json"
        ).read_text()
    )

    assert "global54_semantic_v6" in config["generator_checkpoint"]
    assert "global54_semantic_v6" in config["generator_training_summary"]
    assert "global54_semantic_v6" in config["output_dir"]
    assert "global54_semantic_perceptual_v6" in video.EXPECTED_TRAINING_POLICY


def _pair(index: int) -> dict:
    return {
        "label": f"pair {index}",
        "neutral_prompt": (
            f"Perform a medium-intensity gesture {index} with neutral affect."
        ),
        "emotion_prompt": (
            f"Perform a medium-intensity gesture {index} with a happy affect."
        ),
        "seed": 1000 + index,
    }


def test_pair_timeline_is_exactly_sixty_seconds():
    timeline = video.build_pair_timeline(
        [_pair(index) for index in range(6)],
        slot_frames=300,
        fps=30.0,
        target_duration_sec=60.0,
    )

    assert timeline[0]["start_frame"] == 0
    assert timeline[-1]["end_frame_exclusive"] == 1800
    assert timeline[-1]["end_sec"] == 60.0
    for left, right in zip(timeline[:-1], timeline[1:], strict=True):
        assert left["end_frame_exclusive"] == right["start_frame"]


def test_pair_timeline_rejects_non_emotion_comparison():
    pair = _pair(0)
    pair["emotion_prompt"] = pair["neutral_prompt"]

    with pytest.raises(video.StyleEmotionVideoError, match="neutral/emotion"):
        video.build_pair_timeline(
            [pair],
            slot_frames=1800,
            fps=30.0,
            target_duration_sec=60.0,
        )


@pytest.mark.parametrize("scale", [1.0, 1.5])
def test_cfg_velocity_uses_explicit_equation(scale):
    unconditional = torch.tensor([[[-2.0, 4.0]]])
    conditioned = torch.tensor([[[3.0, -6.0]]])

    result = video.classifier_free_guidance_velocity(
        unconditional, conditioned, guidance_scale=scale
    )

    expected = unconditional + scale * (conditioned - unconditional)
    assert torch.equal(result, expected)


def test_cfg_rejects_unapproved_scale():
    with pytest.raises(video.StyleEmotionVideoError, match="1.0 or 1.5"):
        video.classifier_free_guidance_velocity(
            torch.zeros(1), torch.ones(1), guidance_scale=2.0
        )


def test_endpoint_hold_preserves_every_raw_frame_without_blending():
    raw = np.arange(6 * 18, dtype=np.float32).reshape(6, 18)
    original = raw.copy()

    playback = video.endpoint_hold(raw, slot_frames=10)

    assert np.array_equal(raw, original)
    assert np.array_equal(playback[:6], raw)
    assert np.array_equal(playback[6:], np.repeat(raw[-1:], 4, axis=0))
    assert np.array_equal(playback[5], raw[-1])


def test_endpoint_hold_refuses_crop_or_timewarp():
    raw = np.zeros((12, 18), dtype=np.float32)

    with pytest.raises(video.StyleEmotionVideoError, match="cropping"):
        video.endpoint_hold(raw, slot_frames=11)


def test_shared_initial_noise_is_exactly_reproducible_and_prefix_stable():
    full = video.shared_initial_noise(seed=73, frames=20)
    repeated = video.shared_initial_noise(seed=73, frames=20)
    shorter = video.shared_initial_noise(seed=73, frames=12)

    assert np.array_equal(full, repeated)
    assert np.array_equal(full[:12], shorter)


def test_qwen_style_head_condition_has_no_oracle_channels():
    head = QwenStyleHead(hidden_dim=8, zero_initialize_output=False)
    with torch.no_grad():
        head.output_projection.bias.copy_(torch.tensor([0.25, -0.5, 0.75]))
    latent = np.linspace(-1.0, 1.0, 128, dtype=np.float32)

    condition, style = video.compose_text_style_condition(
        head, latent, device=torch.device("cpu")
    )

    assert condition.shape == (264,)
    assert np.count_nonzero(condition[:133]) == 0
    assert np.array_equal(condition[133:136], style)
    assert np.array_equal(condition[136:264], latent)


def test_ass_discloses_native_duration_cfg_and_endpoint_only_display():
    timeline = video.build_pair_timeline(
        [_pair(0)],
        slot_frames=1800,
        fps=30.0,
        target_duration_sec=60.0,
    )
    timeline[0].update(
        {
            "neutral_predicted_duration_sec": 2.1,
            "emotion_predicted_duration_sec": 2.4,
            "neutral_predicted_style": [0.0, -0.1, 0.2],
            "emotion_predicted_style": [0.1, 0.3, 0.5],
        }
    )

    document = video.build_ass_document(
        timeline,
        duration_sec=60.0,
        width=1920,
        height=720,
        pane_width=600,
        guidance_scale=1.5,
    )

    assert "CFG v = uncond + 1.5" in document
    assert "SAME INITIAL NOISE" in document
    assert "PLANNER-PREDICTED NATIVE DURATION" in document
    assert "ENDPOINT HOLD ONLY" in document
    assert "NO SMOOTHING" in document
    assert "NO TIMEWARP" in document
    assert "NO BOUNDARY/LAST-FRAME BLEND" in document
    assert "neutral affect." in document
    assert "happy affect." in document


def test_forbidden_external_dataset_path_is_rejected(tmp_path):
    forbidden = tmp_path / "forbidden_kimodo_cache.npz"

    with pytest.raises(
        video.StyleEmotionVideoError, match="forbidden external-data"
    ):
        video._resolve_path(
            str(forbidden), config_dir=tmp_path, field="cache"
        )
