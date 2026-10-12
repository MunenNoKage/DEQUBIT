"""MLP-correction model."""

import torch
from torch import nn


class MLPCorrector(nn.Module):
    """Fully connected network from the bits of one shot to action logits."""

    def __init__(self, num_qubits: int, hidden_sizes: list[int], dropout: float) -> None:
        super().__init__()
        raise NotImplementedError  # TODO

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        """(batch, num_qubits) -> (batch, num_actions)."""
        raise NotImplementedError  # TODO
