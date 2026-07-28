from __future__ import annotations

import json

import numpy as np
import pytest

from upper_body_skeleton.ula_v2_dialogue_action_episode import (
    DIALOGUE_LATENT_SLICE,
    DIRECTIVE_LATENT_SLICE,
    action_directive_from_motion_style,
    validate_dialogue_action_v11_episode,
)


def test_action_directive_describes_complete_motion_not_gesture() -> None:
    directive = action_directive_from_motion_style(
        {
            "motion_style": {
                "arm_amplitude": "broad",
                "laterality": "both",
                "pace": "quick",
                "head_engagement": "engaged",
            }
        }
    )
    assert "18-DoF upper-body action" in directive
    assert "gesture" not in directive.casefold()


def test_role_latent_slices_are_disjoint_and_cover_128d() -> None:
    assert DIRECTIVE_LATENT_SLICE == slice(136, 200)
    assert DIALOGUE_LATENT_SLICE == slice(200, 264)


def test_validator_rejects_gesture_narrowing() -> None:
    with pytest.raises(ValueError, match="narrow motion to gesture"):
        validate_dialogue_action_v11_episode(
            {
                "clip_id": "x",
                "artifact_kind": "ula_v2_dialogue_action_v11_train_episode",
                "formal_episode_contract": "ula_v2_18d_dialogue_action_v11_episode_v1",
                "eligibility_mode": "dialogue_action_v11_train_ready",
                "accepted_for_training": True,
                "native_variable_length": True,
                "action_directive_text": "Make a gesture.",
                "dialogue_text": "hello",
            }
        )
