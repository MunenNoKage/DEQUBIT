"""Tests for dequbit.features."""

import math

import pytest
import torch

from dequbit.actions import IDENTITY_ACTION, build_masks, destinations
from dequbit.dataset import GATE_INFIDELITY, LOG_READOUT_P01, LOG_READOUT_P10, index_to_bits
from dequbit.features import (
    NUM_ACTION_CIRCUIT_FEATURES,
    NUM_CIRCUIT_FEATURES,
    NUM_GLOBAL_FEATURES,
    NUM_QUBIT_FEATURES,
    action_circuit_features,
    circuit_features,
    global_context,
    local_counts,
    neighbourhood_counts,
    own_count,
    signed_bits,
)

NUM_QUBITS = 2


def test_signed_bits_are_plus_minus_one_bits_of_every_bitstring() -> None:
    """Bit 0 becomes -1 and bit 1 becomes +1, for every bitstring."""
    bits = signed_bits(NUM_QUBITS)

    expected = index_to_bits(torch.arange(2**NUM_QUBITS), NUM_QUBITS)
    assert bits.shape == (2**NUM_QUBITS, NUM_QUBITS)
    assert torch.equal((bits + 1) / 2, expected.float())


def test_local_counts_compare_destination_with_own_count() -> None:
    """Action feature is log(destination count + 1) - log(own count + 1), 0 for the identity."""
    masks = build_masks(NUM_QUBITS, radius=2)
    counts = torch.tensor([[3.0, 0.0, 1.0, 7.0]])

    features = local_counts(counts, destinations(masks, NUM_QUBITS))

    assert features.shape == (1, 2**NUM_QUBITS, len(masks))
    assert torch.all(features[..., IDENTITY_ACTION] == 0)
    # Bitstring 1 flipped by mask 2 becomes bitstring 3.
    action = masks.tolist().index(2)
    torch.testing.assert_close(features[0, 1, action], torch.tensor(math.log(8) - math.log(1)))


def test_own_count_is_log_count_relative_to_shots() -> None:
    """Own count of a bitstring is log((count + 1) / number of shots)."""
    counts = torch.tensor([[3.0, 0.0, 1.0, 6.0]])

    features = own_count(counts)

    expected = torch.log(torch.tensor([[4.0, 1.0, 2.0, 7.0]]) / 10)
    torch.testing.assert_close(features, expected)


def test_global_context_of_uniform_shots() -> None:
    """Shots spread evenly over 4 bitstrings give the largest entropy and the smallest collision
    probability."""
    counts = torch.full((1, 2**NUM_QUBITS), 5.0)

    context = global_context(counts, NUM_QUBITS)

    assert context.shape == (1, NUM_GLOBAL_FEATURES)
    expected = torch.tensor([[math.log(2), -math.log(2), math.log(4 / 20) / NUM_QUBITS]])
    torch.testing.assert_close(context, expected)


def test_global_context_of_one_repeated_bitstring() -> None:
    """Shots all on one bitstring give entropy 0 and collision probability 1."""
    counts = torch.tensor([[0.0, 0.0, 20.0, 0.0]])

    context = global_context(counts, NUM_QUBITS)

    expected = torch.tensor([[0.0, 0.0, math.log(1 / 20) / NUM_QUBITS]])
    torch.testing.assert_close(context, expected)


def test_global_context_rejects_circuit_without_shots() -> None:
    """A circuit with no shots raises ValueError."""
    counts = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])

    with pytest.raises(ValueError):
        global_context(counts, NUM_QUBITS)


def test_action_circuit_features_describe_the_flipped_bits() -> None:
    """Flipping both bits of bitstring 01 flips one 1, adds both infidelities and both
    readout errors in the direction the flip undoes: p01 of bit 0, p10 of bit 1."""
    masks = build_masks(NUM_QUBITS, radius=2)
    qubit_features = torch.zeros(1, NUM_QUBITS, NUM_QUBIT_FEATURES)
    qubit_features[0, :, GATE_INFIDELITY] = torch.tensor([0.1, 0.3])
    qubit_features[0, :, LOG_READOUT_P01] = torch.tensor([-1.0, -2.0])
    qubit_features[0, :, LOG_READOUT_P10] = torch.tensor([-3.0, -4.0])

    features = action_circuit_features(qubit_features, masks)

    assert features.shape == (1, 2**NUM_QUBITS, len(masks), NUM_ACTION_CIRCUIT_FEATURES)
    assert torch.all(features[:, :, IDENTITY_ACTION] == 0)
    both = masks.tolist().index(3)
    torch.testing.assert_close(features[0, 1, both], torch.tensor([1.0, 0.4, -1.0 - 4.0]))


def test_circuit_features_are_mean_and_sum_of_gate_infidelity() -> None:
    """Infidelities 0.1 and 0.3 give mean 0.2 and sum 0.4."""
    qubit_features = torch.zeros(1, NUM_QUBITS, NUM_QUBIT_FEATURES)
    qubit_features[0, :, GATE_INFIDELITY] = torch.tensor([0.1, 0.3])

    features = circuit_features(qubit_features)

    assert features.shape == (1, NUM_CIRCUIT_FEATURES)
    torch.testing.assert_close(features, torch.tensor([[0.2, 0.4]]))


def test_neighbourhood_counts_average_the_counts_at_each_distance() -> None:
    """Of 3 qubits with counts 0..7, bitstring 0 has neighbours 1, 2, 4 and 3, 5, 6."""
    counts = torch.arange(8.0).unsqueeze(0)

    features = neighbourhood_counts(counts, 3)

    assert features.shape == (1, 8, 2)
    torch.testing.assert_close(features[0, 0], torch.log1p(torch.tensor([7.0 / 3, 14.0 / 3])))
