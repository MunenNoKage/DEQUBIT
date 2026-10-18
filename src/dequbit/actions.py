"""Action space and the distributions built from corrected shots.

Shots are integers whose bit j is qubit j (see dequbit.dataset). An action is an XOR
mask: applying action a to shot x gives x ^ masks[a]. Action 0 is the identity, then
come the flips of one bit, then the flips of two bits, up to radius flipped bits.
"""

import math
from itertools import combinations

import torch

from dequbit.dataset import index_to_bits

IDENTITY_ACTION = 0


def build_masks(num_qubits: int, radius: int) -> torch.Tensor:
    """Build the XOR mask of every action.

    Args:
        num_qubits: Number of measured qubits.
        radius: Largest number of bits one action flips.

    Returns:
        Masks as integers, shape (actions,): the identity 0, then all masks with one
        set bit, then all masks with two set bits, and so on up to radius.

    Raises:
        ValueError: If radius is not between 1 and num_qubits.
    """
    if not 1 <= radius <= num_qubits:
        raise ValueError(f"radius must be between 1 and {num_qubits}, got {radius}")

    masks = [0]
    for num_flips in range(1, radius + 1):
        for qubits in combinations(range(num_qubits), num_flips):
            masks.append(sum(1 << qubit for qubit in qubits))

    return torch.tensor(masks, dtype=torch.int64)


def hamming_weights(masks: torch.Tensor, num_qubits: int) -> torch.Tensor:
    """Count the bits each action flips.

    Args:
        masks: Masks of the actions, shape (actions,).
        num_qubits: Number of measured qubits.

    Returns:
        Number of flipped bits of every action, shape (actions,).
    """
    return index_to_bits(masks, num_qubits).sum(dim=-1)


def destinations(masks: torch.Tensor, num_qubits: int) -> torch.Tensor:
    """Find where every action moves every bitstring.

    Args:
        masks: Masks of the actions, shape (actions,).
        num_qubits: Number of measured qubits.

    Returns:
        Table of x ^ masks[a] for every bitstring x and action a, shape
        (2**num_qubits, actions).
    """
    bitstrings = torch.arange(2**num_qubits, device=masks.device)

    return bitstrings.unsqueeze(-1) ^ masks


def initial_logits(num_actions: int) -> torch.Tensor:
    """Give the action logits a model outputs before training.

    The identity gets log(num_actions) + 1 and every other action 0, so the identity has
    probability e * A / (e * A + A - 1) for A actions, about 0.75, and the flips share
    the rest. With equal logits most of every bitstring would move to random
    neighbours. With the identity near probability 1 the softmax saturates and the
    flips get almost no gradient, so training stays at changing nothing.

    Args:
        num_actions: Number of actions, the identity included.

    Returns:
        Logits, shape (num_actions,).
    """
    logits = torch.zeros(num_actions)
    logits[IDENTITY_ACTION] = math.log(num_actions) + 1

    return logits


def empirical_distribution(shots: torch.Tensor, num_qubits: int) -> torch.Tensor:
    """Compute the share of shots equal to each bitstring.

    Args:
        shots: Shots as integers, shape (circuits, shots).
        num_qubits: Number of measured qubits.

    Returns:
        Distribution of every circuit, shape (circuits, 2**num_qubits).
    """
    counts = torch.zeros(len(shots), 2**num_qubits, device=shots.device)
    counts.scatter_add_(1, shots, torch.ones(shots.shape, device=shots.device))

    return counts / shots.shape[1]


def expected_distribution(
    noisy: torch.Tensor, action_probs: torch.Tensor, destination_table: torch.Tensor
) -> torch.Tensor:
    """Compute the corrected distribution averaged over the action probabilities.

    The mass noisy[x] is split between the destinations x ^ masks[a] in proportion to
    P(a | x). The result is differentiable in action_probs, so a loss against P_target
    trains the model.

    Args:
        noisy: Noisy distribution of every circuit, shape (circuits, 2**num_qubits).
        action_probs: P(a | x) for every bitstring x, shape
            (circuits, 2**num_qubits, actions); rows sum to 1.
        destination_table: Output of destinations, shape (2**num_qubits, actions).

    Returns:
        Corrected distribution of every circuit, shape (circuits, 2**num_qubits).

    Raises:
        ValueError: If action_probs does not match noisy and destination_table.
    """
    if action_probs.shape != (*noisy.shape, destination_table.shape[1]):
        raise ValueError(f"action_probs has shape {tuple(action_probs.shape)}")

    # The (bitstring, action) pairs of a circuit are flattened, so one scatter_add moves
    # every mass to its destination.
    moved_mass = (noisy.unsqueeze(-1) * action_probs).flatten(start_dim=1)
    index = destination_table.expand_as(action_probs).flatten(start_dim=1)
    corrected = torch.zeros_like(noisy).scatter_add(1, index, moved_mass)

    return corrected


def sample_actions(
    shots: torch.Tensor, action_probs: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """Draw one action for every shot from P(a | x) of its bitstring.

    Args:
        shots: Shots as integers, shape (circuits, shots).
        action_probs: P(a | x) for every bitstring x, shape
            (circuits, 2**num_qubits, actions).
        generator: Random generator of the draw.

    Returns:
        Action index of every shot, shape (circuits, shots).
    """
    circuit_index = torch.arange(len(shots), device=shots.device).unsqueeze(-1)
    shot_probs = action_probs[circuit_index, shots]
    actions = torch.multinomial(shot_probs.flatten(end_dim=1), 1, generator=generator)

    return actions.view(shots.shape)


def apply_actions(shots: torch.Tensor, actions: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Apply one action to every shot.

    Args:
        shots: Shots as integers, any shape (...).
        actions: Action index of every shot, same shape as shots.
        masks: Masks of the actions, shape (actions,).

    Returns:
        Corrected shots, same shape as shots.
    """
    return shots ^ masks[actions]
