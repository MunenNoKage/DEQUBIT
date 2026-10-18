"""Tests for dequbit.actions."""

import math

import pytest
import torch

from dequbit.actions import (
    IDENTITY_ACTION,
    apply_actions,
    build_masks,
    destinations,
    empirical_distribution,
    expected_distribution,
    hamming_weights,
    initial_logits,
    sample_actions,
)

NUM_QUBITS = 5
SEED = 0


def random_distribution(num_circuits: int) -> torch.Tensor:
    """Draw a random distribution over the bitstrings of every circuit.

    Args:
        num_circuits: Number of circuits.

    Returns:
        Distributions, shape (num_circuits, 2**NUM_QUBITS); rows sum to 1.
    """
    generator = torch.Generator().manual_seed(SEED)

    return torch.rand(num_circuits, 2**NUM_QUBITS, generator=generator).softmax(dim=-1)


def random_action_probs(num_circuits: int, num_actions: int) -> torch.Tensor:
    """Draw random P(a | x) for every bitstring of every circuit.

    Args:
        num_circuits: Number of circuits.
        num_actions: Number of actions.

    Returns:
        Probabilities, shape (num_circuits, 2**NUM_QUBITS, num_actions); rows sum to 1.
    """
    generator = torch.Generator().manual_seed(SEED)
    logits = torch.randn(num_circuits, 2**NUM_QUBITS, num_actions, generator=generator)

    return logits.softmax(dim=-1)


@pytest.mark.parametrize("radius, num_actions", [(1, 6), (2, 16), (5, 32)])
def test_build_masks_gives_all_flips_up_to_radius(radius: int, num_actions: int) -> None:
    """Radius r gives the identity first and every mask with 1 to r flipped bits, each once.

    Args:
        radius: Largest number of flipped bits.
        num_actions: Expected number of masks for 5 qubits.
    """
    masks = build_masks(NUM_QUBITS, radius)

    assert len(masks) == num_actions
    assert len(masks.unique()) == num_actions
    assert masks[IDENTITY_ACTION] == 0
    assert hamming_weights(masks, NUM_QUBITS).max() == radius


@pytest.mark.parametrize("radius", [0, NUM_QUBITS + 1])
def test_build_masks_rejects_radius_out_of_range(radius: int) -> None:
    """A radius below 1 or above the number of qubits raises ValueError.

    Args:
        radius: Invalid radius.
    """
    with pytest.raises(ValueError):
        build_masks(NUM_QUBITS, radius)


def test_hamming_weights_are_sorted_by_number_of_flips() -> None:
    """Masks come in order: the identity, then 5 single flips, then 10 pair flips."""
    masks = build_masks(NUM_QUBITS, radius=2)

    weights = hamming_weights(masks, NUM_QUBITS)

    assert weights.tolist() == [0] + [1] * 5 + [2] * 10


def test_destinations_are_xor_and_undo_themselves() -> None:
    """Bitstring x under mask m goes to x XOR m, and the same mask applied twice returns x."""
    masks = build_masks(NUM_QUBITS, radius=2)

    table = destinations(masks, NUM_QUBITS)

    bitstrings = torch.arange(2**NUM_QUBITS).unsqueeze(-1)
    assert torch.equal(table, bitstrings ^ masks)
    assert torch.equal(table.gather(0, table), bitstrings.expand_as(table))


def test_empirical_distribution_counts_shares() -> None:
    """Shots 0, 3, 3, 31 give 0.25 to bitstrings 0 and 31 and 0.5 to bitstring 3."""
    shots = torch.tensor([[0, 3, 3, 31]])

    distribution = empirical_distribution(shots, NUM_QUBITS)

    expected = torch.zeros(1, 2**NUM_QUBITS)
    expected[0, [0, 3, 31]] = torch.tensor([0.25, 0.5, 0.25])
    assert torch.equal(distribution, expected)


def test_expected_distribution_sums_to_one() -> None:
    """Moving probability with random action probabilities keeps the total at 1."""
    masks = build_masks(NUM_QUBITS, radius=2)
    noisy = random_distribution(3)
    action_probs = random_action_probs(3, len(masks))

    corrected = expected_distribution(noisy, action_probs, destinations(masks, NUM_QUBITS))

    torch.testing.assert_close(corrected.sum(dim=-1), torch.ones(3))


def test_expected_distribution_with_identity_equals_input() -> None:
    """If every bitstring always takes the identity action, the distribution stays the same."""
    masks = build_masks(NUM_QUBITS, radius=2)
    noisy = random_distribution(3)
    action_probs = torch.zeros(3, 2**NUM_QUBITS, len(masks))
    action_probs[..., IDENTITY_ACTION] = 1.0

    corrected = expected_distribution(noisy, action_probs, destinations(masks, NUM_QUBITS))

    torch.testing.assert_close(corrected, noisy)


def test_expected_distribution_passes_gradient_to_action_probs() -> None:
    """The corrected distribution has a gradient in the action probabilities, so a model can be
    trained through it."""
    masks = build_masks(NUM_QUBITS, radius=1)
    noisy = random_distribution(2)
    action_probs = random_action_probs(2, len(masks)).requires_grad_()

    corrected = expected_distribution(noisy, action_probs, destinations(masks, NUM_QUBITS))
    corrected[:, 0].sum().backward()

    assert action_probs.grad is not None
    assert action_probs.grad.abs().sum() > 0


def test_expected_distribution_rejects_wrong_shape() -> None:
    """Action probabilities with one action too many raise ValueError."""
    masks = build_masks(NUM_QUBITS, radius=1)
    noisy = random_distribution(2)
    action_probs = random_action_probs(2, len(masks) + 1)

    with pytest.raises(ValueError):
        expected_distribution(noisy, action_probs, destinations(masks, NUM_QUBITS))


def test_apply_actions_flips_only_the_chosen_bits() -> None:
    """The identity keeps 00000, a single flip clears bit 0 of 10101, a pair flip clears bits 0 and
    1 of 11111."""
    masks = build_masks(NUM_QUBITS, radius=2)
    shots = torch.tensor([0b00000, 0b10101, 0b11111])
    actions = torch.tensor([IDENTITY_ACTION, 1, 6])

    corrected = apply_actions(shots, actions, masks)

    assert corrected.tolist() == [0b00000, 0b10100, 0b11100]


def test_sample_actions_follows_one_hot_probs() -> None:
    """If action 3 has probability 1 for every bitstring, every shot gets action 3."""
    masks = build_masks(NUM_QUBITS, radius=1)
    generator = torch.Generator().manual_seed(SEED)
    shots = torch.randint(0, 2**NUM_QUBITS, (2, 50), generator=generator)
    action_probs = torch.zeros(2, 2**NUM_QUBITS, len(masks))
    action_probs[..., 3] = 1.0

    actions = sample_actions(shots, action_probs, generator)

    assert torch.equal(actions, torch.full_like(shots, 3))


def test_sampled_corrections_match_expected_distribution_on_average() -> None:
    """With many shots, sampled corrected shots are within L1 0.02 of the expected distribution."""
    masks = build_masks(NUM_QUBITS, radius=2)
    generator = torch.Generator().manual_seed(SEED)
    shots = torch.randint(0, 2**NUM_QUBITS, (2, 200_000), generator=generator)
    action_probs = random_action_probs(2, len(masks))

    actions = sample_actions(shots, action_probs, generator)
    sampled = empirical_distribution(apply_actions(shots, actions, masks), NUM_QUBITS)

    noisy = empirical_distribution(shots, NUM_QUBITS)
    expected = expected_distribution(noisy, action_probs, destinations(masks, NUM_QUBITS))
    assert (sampled - expected).abs().sum(dim=-1).max() < 0.02


def test_initial_logits_give_identity_about_three_quarters() -> None:
    """With 16 actions the identity starts with probability 16e / (16e + 15), the rest equal."""
    probs = initial_logits(16).softmax(dim=-1)

    expected_identity = 16 * math.e / (16 * math.e + 15)
    torch.testing.assert_close(probs[IDENTITY_ACTION], torch.tensor(expected_identity))
    torch.testing.assert_close(probs[1:], torch.full((15,), (1 - expected_identity) / 15))
