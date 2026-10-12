"""Perceiver correction model."""

import torch
from torch import nn


class PerceiverCorrector(nn.Module):
    """Learned latents that cross-attend to the bit tokens of one shot."""

    def __init__(
        self,
        num_qubits: int,
        latent_size: int,
        num_latents: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        raise NotImplementedError  # TODO

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        """(batch, num_qubits) -> (batch, num_actions)."""
        raise NotImplementedError  # TODO
