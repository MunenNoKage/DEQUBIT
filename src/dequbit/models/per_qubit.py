"""Learned baseline B4: the same action probabilities for every shot."""

import torch
from torch import nn

from dequbit.actions import build_masks, initial_logits


class PerQubitFlipModel(nn.Module):
    """Learned action logits that do not depend on the shot or the circuit.

    With radius 1 this is one flip probability per qubit. It shows what a correction
    can do without looking at the shots.
    """

    def __init__(self, num_qubits: int, radius: int) -> None:
        """Create the logits.

        Args:
            num_qubits: Number of measured qubits.
            radius: Largest number of bits one action flips.
        """
        super().__init__()

        num_actions = len(build_masks(num_qubits, radius))
        self.logits = nn.Parameter(initial_logits(num_actions))

    def forward(self, counts: torch.Tensor) -> torch.Tensor:
        """Repeat the logits for every bitstring of every circuit.

        Args:
            counts: Shot counts of every circuit, shape (circuits, 2**num_qubits).

        Returns:
            Action logits, shape (circuits, 2**num_qubits, actions).
        """
        return self.logits.expand(*counts.shape, -1)
