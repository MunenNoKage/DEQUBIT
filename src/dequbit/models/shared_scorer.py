"""Shared action scorer: one small network scores every action of every bitstring."""

import torch
from torch import nn

from dequbit.actions import (
    IDENTITY_ACTION,
    build_masks,
    destinations,
    hamming_weights,
    initial_logits,
)
from dequbit.features import NUM_GLOBAL_FEATURES, global_context, local_counts, own_count

# Local count, number of flipped bits and own count of the bitstring.
NUM_ACTION_FEATURES = 3


class SharedScorer(nn.Module):
    """Network that scores one action of one bitstring from features of that pair.

    The features are the local count of the action, the number of bits it flips, the
    own count of the bitstring and the global context of the circuit. None of them
    names a qubit, so the same weights fit every number of qubits, and two actions with
    the same features get the same logit.
    """

    def __init__(self, num_qubits: int, radius: int, hidden_sizes: list[int]) -> None:
        """Build the layers.

        Args:
            num_qubits: Number of measured qubits.
            radius: Largest number of bits one action flips.
            hidden_sizes: Width of every hidden layer.
        """
        super().__init__()

        masks = build_masks(num_qubits, radius)
        self.num_qubits = num_qubits
        # Buffers move with the model to the GPU but are not trained or saved.
        self.register_buffer("destination_table", destinations(masks, num_qubits), persistent=False)
        self.register_buffer(
            "flipped_bits", hamming_weights(masks, num_qubits).long(), persistent=False
        )

        layers = []
        input_size = NUM_ACTION_FEATURES + NUM_GLOBAL_FEATURES
        for hidden_size in hidden_sizes:
            layers += [nn.Linear(input_size, hidden_size), nn.GELU()]
            input_size = hidden_size
        self.hidden = nn.Sequential(*layers)
        self.head = nn.Linear(input_size, 1)

        # The head has one output for all actions, so its bias cannot favour the
        # identity. A learned bias per number of flipped bits does, and with a zero head
        # the first output is initial_logits for every input.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        flip_bias = torch.zeros(radius + 1)
        flip_bias[0] = initial_logits(len(masks))[IDENTITY_ACTION]
        self.flip_bias = nn.Parameter(flip_bias)

    def forward(self, counts: torch.Tensor) -> torch.Tensor:
        """Compute the action logits of every bitstring of every circuit.

        Args:
            counts: Shot counts of every circuit, shape (circuits, 2**num_qubits).

        Returns:
            Action logits, shape (circuits, 2**num_qubits, actions).

        Raises:
            ValueError: If counts does not have 2**num_qubits columns.
        """
        num_bitstrings = counts.shape[-1]
        if num_bitstrings != 2**self.num_qubits:
            raise ValueError(f"counts has {num_bitstrings} columns, not {2**self.num_qubits}")

        features = self._pair_features(counts)
        scores = self.head(self.hidden(features)).squeeze(-1)

        return scores + self.flip_bias[self.flipped_bits]

    def _pair_features(self, counts: torch.Tensor) -> torch.Tensor:
        """Build the features of every pair of a bitstring and an action.

        Args:
            counts: Shot counts of every circuit, shape (circuits, 2**num_qubits).

        Returns:
            Features, shape (circuits, 2**num_qubits, actions,
            NUM_ACTION_FEATURES + NUM_GLOBAL_FEATURES).
        """
        local = local_counts(counts, self.destination_table)
        pair_shape = local.shape

        flipped = self.flipped_bits.to(counts.dtype).expand(pair_shape)
        own = own_count(counts).unsqueeze(-1).expand(pair_shape)
        action_features = torch.stack([local, flipped, own], dim=-1)

        # The context is the same for every bitstring and action of a circuit.
        context = global_context(counts, self.num_qubits)
        context = context[:, None, None, :].expand(*pair_shape, -1)

        return torch.cat([action_features, context], dim=-1)
