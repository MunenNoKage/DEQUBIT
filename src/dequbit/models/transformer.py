"""Transformer correction model."""

import torch
from torch import nn


class TransformerCorrector(nn.Module):
    """Transformer encoder over the bits of one shot, one token per qubit."""

    def __init__(
        self,
        num_qubits: int,
        embedding_size: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        raise NotImplementedError  # TODO

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        """(batch, num_qubits) -> (batch, num_actions)."""
        raise NotImplementedError  # TODO
