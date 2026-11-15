"""Shared action scorer: one small network scores every action of every bitstring."""

from functools import lru_cache

import torch
from torch import nn

from dequbit.actions import build_masks, destinations, hamming_weights, initial_logits
from dequbit.features import NUM_GLOBAL_FEATURES, global_context, local_counts, own_count

# Local count, number of flipped bits and own count of the bitstring.
NUM_ACTION_FEATURES = 3


class SharedScorer(nn.Module):
    """Network that scores one action of one bitstring from features of that pair.

    The features are the local count of the action, the number of bits it flips, the
    own count of the bitstring and the global context of the circuit. None of them
    names a qubit, so one model corrects circuits of every number of qubits, and two
    actions with the same features get the same logit.
    """

    def __init__(self, radius: int, hidden_sizes: list[int]) -> None:
        """Build the layers.

        Args:
            radius: Largest number of bits one action flips.
            hidden_sizes: Width of every hidden layer.
        """
        super().__init__()

        self.radius = radius

        layers = []
        input_size = NUM_ACTION_FEATURES + NUM_GLOBAL_FEATURES
        for hidden_size in hidden_sizes:
            layers += [nn.Linear(input_size, hidden_size), nn.GELU()]
            input_size = hidden_size
        self.hidden = nn.Sequential(*layers)
        self.head = nn.Linear(input_size, 1)

        # The head has one output for all actions, so its bias cannot tell the identity
        # from a flip. A learned bias per number of flipped bits can. With a zero head
        # and zero biases the first output is initial_logits for every input.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.flip_bias = nn.Parameter(torch.zeros(radius + 1))

    def forward(self, counts: torch.Tensor) -> torch.Tensor:
        """Compute the action logits of every bitstring of every circuit.

        The number of qubits is read from the number of bitstrings, so circuits of any
        size are accepted, one size per call.

        Args:
            counts: Shot counts of every circuit, shape (circuits, 2**num_qubits).

        Returns:
            Action logits, shape (circuits, 2**num_qubits, actions).

        Raises:
            ValueError: If the number of columns of counts is not a power of two.
        """
        num_bitstrings = counts.shape[-1]
        num_qubits = num_bitstrings.bit_length() - 1
        if num_bitstrings != 2**num_qubits:
            raise ValueError(f"counts has {num_bitstrings} columns, not a power of two")

        destination_table, flipped_bits = _action_tables(num_qubits, self.radius)
        destination_table = destination_table.to(counts.device)
        flipped_bits = flipped_bits.to(counts.device)

        features = self._pair_features(counts, destination_table, flipped_bits)
        scores = self.head(self.hidden(features)).squeeze(-1)
        # The identity logit of initial_logits grows with the number of actions, which
        # depends on the number of qubits, so it is added to the output and not learned.
        start = initial_logits(len(flipped_bits)).to(counts.device)

        return scores + self.flip_bias[flipped_bits] + start

    def _pair_features(
        self, counts: torch.Tensor, destination_table: torch.Tensor, flipped_bits: torch.Tensor
    ) -> torch.Tensor:
        """Build the features of every pair of a bitstring and an action.

        Args:
            counts: Shot counts of every circuit, shape (circuits, 2**num_qubits).
            destination_table: Output of dequbit.actions.destinations, shape
                (2**num_qubits, actions).
            flipped_bits: Number of bits every action flips, shape (actions,).

        Returns:
            Features, shape (circuits, 2**num_qubits, actions,
            NUM_ACTION_FEATURES + NUM_GLOBAL_FEATURES).
        """
        num_qubits = counts.shape[-1].bit_length() - 1
        local = local_counts(counts, destination_table)
        pair_shape = local.shape

        flipped = flipped_bits.to(counts.dtype).expand(pair_shape)
        own = own_count(counts).unsqueeze(-1).expand(pair_shape)
        action_features = torch.stack([local, flipped, own], dim=-1)

        # The context is the same for every bitstring and action of a circuit.
        context = global_context(counts, num_qubits)
        context = context[:, None, None, :].expand(*pair_shape, -1)

        return torch.cat([action_features, context], dim=-1)


@lru_cache
def _action_tables(num_qubits: int, radius: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the action tables of one number of qubits, once per size.

    Args:
        num_qubits: Number of measured qubits.
        radius: Largest number of bits one action flips.

    Returns:
        Destination of every action for every bitstring, shape (2**num_qubits, actions),
        and the number of bits every action flips, shape (actions,).
    """
    masks = build_masks(num_qubits, radius)

    return destinations(masks, num_qubits), hamming_weights(masks, num_qubits)
