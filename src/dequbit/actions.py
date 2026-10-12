"""Action space and the distributions built from corrected shots.

Action 0 keeps a bitstring, action i + 1 flips bit i, so there are num_qubits + 1
actions.
"""

import torch

IDENTITY_ACTION = 0


def num_actions(num_qubits: int) -> int:
    """Identity plus one flip per bit."""
    raise NotImplementedError  # TODO


def candidate_bitstrings(bits: torch.Tensor) -> torch.Tensor:
    """Result of every action on every shot.

    (shots, num_qubits) -> (shots, num_actions, num_qubits).
    """
    raise NotImplementedError  # TODO


def apply_actions(bits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Apply one action per shot.

    Args:
        bits: shape (shots, num_qubits).
        actions: action indices, shape (shots,).

    Returns:
        Corrected bits, shape (shots, num_qubits).
    """
    raise NotImplementedError  # TODO


def empirical_distribution(bits: torch.Tensor, num_qubits: int) -> torch.Tensor:
    """Share of shots equal to each bitstring: (shots, num_qubits) -> (2**num_qubits,)."""
    raise NotImplementedError  # TODO


def expected_distribution(bits: torch.Tensor, action_probs: torch.Tensor) -> torch.Tensor:
    """Corrected distribution averaged over the action probabilities of each shot.

    Differentiable in ``action_probs``, so a loss against P_target trains the model.

    Args:
        bits: noisy shots, shape (shots, num_qubits).
        action_probs: shape (shots, num_actions), rows sum to 1.

    Returns:
        Distribution of shape (2**num_qubits,).
    """
    raise NotImplementedError  # TODO
