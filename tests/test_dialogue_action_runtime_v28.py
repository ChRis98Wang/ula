from pathlib import Path

import numpy as np
import pytest

from upper_body_skeleton.dialogue_action_runtime_v28 import DialogueActionV28Runtime


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = (
    ROOT
    / "releases/dialogue_action_v28_pilot7_v1/model/ula_fm_checkpoint.pt"
)


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="V28 runtime checkpoint not built")
def test_v28_runtime_uses_embedded_adapter_with_dynamic_lengths_and_realizations():
    runtime = DialogueActionV28Runtime(CHECKPOINT, device="cpu")
    assert runtime.supported_intent_ids == ("idle_attentive", "search_scan")
    first = runtime.generate("search_scan", frames=73, seed=0)
    second = runtime.generate("search_scan", frames=101, seed=1)
    assert first.shape == (73, 18)
    assert second.shape == (101, 18)
    assert np.isfinite(first).all() and np.isfinite(second).all()
    assert float(np.mean(np.square(first[:50] - second[:50]))) > 1e-5
    with pytest.raises(ValueError, match="does not support"):
        runtime.generate("wave_to_person", frames=73)
