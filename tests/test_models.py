"""Tests for dequbit.models."""

import pytest
import torch

from dequbit.actions import initial_logits
from dequbit.models.mlp import MLPCorrector
from dequbit.models.per_qubit import PerQubitFlipModel

NUM_QUBITS = 3
RADIUS = 2
NUM_ACTIONS = 7
SEED = 0


def random_counts(num_circuits: int) -> torch.Tensor:
    """Draw random shot counts of every circuit.

    Args:
        num_circuits: Number of circuits.

    Returns:
        Counts from 0 to 99, shape (num_circuits, 2**NUM_QUBITS).
    """
    generator = torch.Generator().manual_seed(SEED)

    return torch.randint(0, 100, (num_circuits, 2**NUM_QUBITS), generator=generator).float()


def test_per_qubit_model_gives_same_logits_for_every_bitstring() -> None:
    """B4 outputs initial_logits for every bitstring of every circuit, whatever the counts."""
    model = PerQubitFlipModel(NUM_QUBITS, RADIUS)

    logits = model(random_counts(4))

    assert logits.shape == (4, 2**NUM_QUBITS, NUM_ACTIONS)
    torch.testing.assert_close(logits, initial_logits(NUM_ACTIONS).expand_as(logits))


def test_mlp_starts_with_initial_logits() -> None:
    """Before training the MLP outputs initial_logits for every bitstring of every circuit."""
    model = MLPCorrector(NUM_QUBITS, RADIUS, hidden_sizes=[16, 16])

    logits = model(random_counts(4))

    assert logits.shape == (4, 2**NUM_QUBITS, NUM_ACTIONS)
    torch.testing.assert_close(logits, initial_logits(NUM_ACTIONS).expand_as(logits))


def test_mlp_output_depends_on_counts_once_head_is_trained() -> None:
    """With nonzero head weights, different counts give different logits."""
    torch.manual_seed(SEED)
    model = MLPCorrector(NUM_QUBITS, RADIUS, hidden_sizes=[16])
    torch.nn.init.normal_(model.head.weight)
    counts = random_counts(2)

    first = model(counts[:1])
    second = model(counts[1:])

    assert not torch.allclose(first, second)


def test_mlp_rejects_counts_of_wrong_length() -> None:
    """Counts over 16 bitstrings do not fit a 3-qubit model and raise ValueError."""
    model = MLPCorrector(NUM_QUBITS, RADIUS, hidden_sizes=[16])

    with pytest.raises(ValueError):
        model(torch.ones(1, 16))
