"""Learned baseline: one flip probability per qubit, independent of the input bitstring."""

import torch
from torch import nn


class PerQubitFlipModel(nn.Module):
    """Same action logits for every shot."""

    def __init__(self, num_qubits: int) -> None:
        super().__init__()
        raise NotImplementedError  # TODO

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        """(batch, num_qubits) -> (batch, num_actions)."""
        raise NotImplementedError  # TODO
