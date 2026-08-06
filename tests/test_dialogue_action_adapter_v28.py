import pytest
import torch

from upper_body_skeleton.dialogue_action_adapter_v28 import (
    DialogueActionFormalAdapterV28,
)


def test_v28_adapter_supports_mixed_native_lengths_without_layout_shift():
    model = DialogueActionFormalAdapterV28(
        {
            "realization_count": 3,
            "realization_embedding_dim": 8,
            "condition_dim": 16,
            "hidden_dim": 32,
            "time_frequencies": 4,
        }
    )
    intents = torch.zeros(3, 27)
    intents[0, 1] = 1.0
    intents[1, 2] = 1.0
    intents[2, 1] = 1.0
    output, mask = model.forward_padded(
        intents,
        torch.tensor([0, 1, 2]),
        torch.tensor([5, 8, 3]),
    )
    assert output.shape == (3, 8, 18)
    assert mask.sum(dim=1).tolist() == [5, 8, 3]
    assert torch.isfinite(output).all()
    assert torch.allclose(output[0, 4], model(intents[:1], torch.tensor([0]), frames=5)[0, 4])


def test_v28_adapter_rejects_invalid_lengths_and_realizations():
    model = DialogueActionFormalAdapterV28(
        {
            "realization_count": 2,
            "realization_embedding_dim": 8,
            "condition_dim": 16,
            "hidden_dim": 32,
            "time_frequencies": 4,
        }
    )
    intents = torch.zeros(1, 27)
    with pytest.raises(ValueError, match="at least two"):
        model.forward_padded(intents, torch.tensor([0]), torch.tensor([1]))
    with pytest.raises(ValueError, match="out of range"):
        model.forward_padded(intents, torch.tensor([2]), torch.tensor([5]))
