"""Tests for dequbit.training."""

import math

import pytest
import torch
from torch.utils.data import TensorDataset

from dequbit.actions import build_masks, destinations, hamming_weights
from dequbit.config import Config
from dequbit.models.mlp import MLPCorrector
from dequbit.models.shared_scorer import SharedScorer
from dequbit.training import (
    circuit_kl,
    expected_flipped_bits,
    predict_distributions,
    train,
    training_loss,
)

NUM_QUBITS = 3
RADIUS = 1
FLIP_PROBABILITY = 0.1
SHOTS = 1000
SEED = 0


def training_config(**changes: float) -> Config:
    """Build a training config for fast runs on tiny datasets.

    Args:
        **changes: Values that replace the defaults of this function.

    Returns:
        Training section of the config.
    """
    config = {
        "max_steps": 300,
        "circuits_per_batch": 16,
        "learning_rate": 1e-2,
        "weight_decay": 0.0,
        "validation_interval": 50,
        "early_stopping_patience": 100,
        "flip_penalty": 0.0,
        "relative_loss": False,
    }

    return {**config, **changes}


def tiny_dataset(num_circuits: int, seed: int) -> TensorDataset:
    """Build circuits with a one-peak target and readout flips on every qubit.

    Args:
        num_circuits: Number of circuits.
        seed: Seed of the peak positions.

    Returns:
        Pairs (shot counts, P_target), both of shape (num_circuits, 2**NUM_QUBITS).
    """
    generator = torch.Generator().manual_seed(seed)
    peaks = torch.randint(0, 2**NUM_QUBITS, (num_circuits,), generator=generator)
    targets = torch.nn.functional.one_hot(peaks, 2**NUM_QUBITS).float()

    noisy = targets
    bitstrings = torch.arange(2**NUM_QUBITS)
    for qubit in range(NUM_QUBITS):
        flipped = noisy[:, bitstrings ^ (1 << qubit)]
        noisy = (1 - FLIP_PROBABILITY) * noisy + FLIP_PROBABILITY * flipped

    return TensorDataset((noisy * SHOTS).round(), targets)


def action_tables() -> tuple[torch.Tensor, torch.Tensor]:
    """Build the destination table and the flipped bits of the actions.

    Returns:
        Destination table, shape (2**NUM_QUBITS, actions), and the number of bits every
        action flips, shape (actions,).
    """
    masks = build_masks(NUM_QUBITS, RADIUS)

    return destinations(masks, NUM_QUBITS), hamming_weights(masks, NUM_QUBITS).float()


def test_circuit_kl_is_zero_only_for_equal_distributions() -> None:
    """KL is 0 when the prediction equals the target and positive otherwise."""
    target = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    predicted = torch.tensor([[0.5, 0.5], [0.9, 0.1]])

    kl = circuit_kl(target, predicted)

    assert kl[0] == 0
    assert kl[1] > 0


def test_circuit_kl_skips_bitstrings_with_zero_target() -> None:
    """A bitstring with target 0 adds nothing: KL([1, 0] || [0.5, 0.5]) = log 2."""
    kl = circuit_kl(torch.tensor([[1.0, 0.0]]), torch.tensor([[0.5, 0.5]]))

    torch.testing.assert_close(kl, torch.tensor([math.log(2)]))


def test_expected_flipped_bits_weights_actions_by_mass() -> None:
    """Moving half of the mass with a pair flip gives 0.5 * 2 = 1 flipped bit per shot."""
    masks = build_masks(2, radius=2)
    flipped_bits = hamming_weights(masks, 2).float()
    noisy = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
    action_probs = torch.zeros(1, 4, len(masks))
    action_probs[0, 0, masks.tolist().index(3)] = 1.0
    action_probs[0, 1, masks.tolist().index(0)] = 1.0

    flips = expected_flipped_bits(noisy, action_probs, flipped_bits)

    torch.testing.assert_close(flips, torch.tensor([1.0]))


def test_predict_distributions_gives_probabilities() -> None:
    """Action probabilities sum to 1 for every bitstring, corrected distributions too."""
    destination_table, _ = action_tables()
    model = MLPCorrector(NUM_QUBITS, RADIUS, hidden_sizes=[8])
    counts, _ = tiny_dataset(4, SEED).tensors

    action_probs, corrected = predict_distributions(model, counts, destination_table)

    assert action_probs.shape == (4, 2**NUM_QUBITS, NUM_QUBITS + 1)
    assert corrected.shape == (4, 2**NUM_QUBITS)
    torch.testing.assert_close(action_probs.sum(dim=-1), torch.ones(4, 2**NUM_QUBITS))
    torch.testing.assert_close(corrected.sum(dim=-1), torch.ones(4))


def test_flip_penalty_adds_expected_flipped_bits() -> None:
    """With flip_penalty 1 the loss grows by exactly the mean expected flipped bits."""
    destination_table, flipped_bits = action_tables()
    model = MLPCorrector(NUM_QUBITS, RADIUS, hidden_sizes=[8])
    counts, targets = tiny_dataset(4, SEED).tensors
    noisy = counts / counts.sum(dim=-1, keepdim=True)
    action_probs, _ = predict_distributions(model, counts, destination_table)

    plain = training_loss(
        model, counts, targets, destination_table, flipped_bits, training_config()
    )
    penalised = training_loss(
        model, counts, targets, destination_table, flipped_bits, training_config(flip_penalty=1.0)
    )

    flips = expected_flipped_bits(noisy, action_probs, flipped_bits).mean()
    torch.testing.assert_close(penalised - plain, flips)


def test_relative_loss_divides_by_kl_of_noisy_distribution() -> None:
    """The relative loss is the mean of KL(target || corrected) / KL(target || noisy)."""
    destination_table, flipped_bits = action_tables()
    model = MLPCorrector(NUM_QUBITS, RADIUS, hidden_sizes=[8])
    counts, targets = tiny_dataset(4, SEED).tensors
    noisy = counts / counts.sum(dim=-1, keepdim=True)
    _, corrected = predict_distributions(model, counts, destination_table)

    loss = training_loss(
        model, counts, targets, destination_table, flipped_bits, training_config(relative_loss=True)
    )

    expected = (circuit_kl(targets, corrected) / circuit_kl(targets, noisy)).mean()
    torch.testing.assert_close(loss, expected)


@pytest.mark.parametrize("model_class", [MLPCorrector, SharedScorer])
def test_train_lowers_validation_loss_on_tiny_dataset(model_class: type) -> None:
    """Training moves the flipped shots back to the peak: loss halves, L1RC is negative."""
    torch.manual_seed(SEED)
    model = model_class(NUM_QUBITS, RADIUS, hidden_sizes=[16])

    history = train(
        model,
        tiny_dataset(256, SEED),
        tiny_dataset(64, SEED + 1),
        NUM_QUBITS,
        RADIUS,
        training_config(),
        SEED,
    )

    best_loss = min(record["validation_loss"] for record in history)
    assert best_loss < 0.5 * history[0]["validation_loss"]
    assert min(record["validation_l1rc"] for record in history) < 0


def test_train_keeps_weights_of_best_validation() -> None:
    """After training, the model has the lowest validation loss seen in the history."""
    torch.manual_seed(SEED)
    model = SharedScorer(NUM_QUBITS, RADIUS, hidden_sizes=[16])
    validation_set = tiny_dataset(64, SEED + 1)
    config = training_config(learning_rate=0.1)

    history = train(
        model, tiny_dataset(256, SEED), validation_set, NUM_QUBITS, RADIUS, config, SEED
    )

    destination_table, flipped_bits = action_tables()
    with torch.no_grad():
        final_loss = training_loss(
            model, *validation_set.tensors, destination_table, flipped_bits, config
        )
    best_loss = min(record["validation_loss"] for record in history)
    assert final_loss.item() == pytest.approx(best_loss, rel=1e-5)


def test_train_stops_after_patience_without_improvement() -> None:
    """With learning rate 0 the loss never improves, so training stops after 3 validations."""
    model = SharedScorer(NUM_QUBITS, RADIUS, hidden_sizes=[16])
    config = training_config(learning_rate=0.0, early_stopping_patience=3)

    history = train(
        model, tiny_dataset(32, SEED), tiny_dataset(16, SEED + 1), NUM_QUBITS, RADIUS, config, SEED
    )

    assert [record["step"] for record in history] == [0, 50, 100, 150]
