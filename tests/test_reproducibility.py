"""Tests that the same seed produces the same results."""

import random

import numpy as np
import torch

from dequbit.models.shared_scorer import SharedScorer
from dequbit.training import train
from dequbit.utils import set_seed
from tests.test_training import NUM_QUBITS, RADIUS, tiny_dataset, training_config

SEED = 0


def trained_weights(seed: int) -> dict[str, torch.Tensor]:
    """Seed everything, build a shared scorer and train it briefly.

    Args:
        seed: Seed of the initial weights and of the batch order.

    Returns:
        Weights of the trained model.
    """
    set_seed(seed)
    model = SharedScorer(NUM_QUBITS, RADIUS, hidden_sizes=[16])
    config = training_config(max_steps=100)

    train(
        model, tiny_dataset(64, SEED), tiny_dataset(16, SEED + 1), NUM_QUBITS, RADIUS, config, seed
    )

    return model.state_dict()


def test_set_seed_repeats_python_numpy_and_torch_draws() -> None:
    """After set_seed with the same seed, all three generators draw the same numbers."""
    set_seed(SEED)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    set_seed(SEED)
    second = (random.random(), np.random.rand(), torch.rand(1).item())

    assert first == second


def test_same_seed_gives_same_trained_weights() -> None:
    """Two trainings with the same seed end with identical weights."""
    first = trained_weights(SEED)
    second = trained_weights(SEED)

    for name in first:
        assert torch.equal(first[name], second[name])


def test_other_seed_gives_other_trained_weights() -> None:
    """A different seed changes the initial weights and batches, so the result differs."""
    first = trained_weights(SEED)
    second = trained_weights(SEED + 1)

    assert any(not torch.equal(first[name], second[name]) for name in first)
