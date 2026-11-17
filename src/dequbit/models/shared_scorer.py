"""Shared action scorer: one small network scores every action of every bitstring."""

from functools import lru_cache

import torch
from torch import nn

from dequbit.actions import build_masks, destinations, hamming_weights, initial_logits
from dequbit.features import (
    NEIGHBOURHOOD_DISTANCES,
    NUM_ACTION_CIRCUIT_FEATURES,
    NUM_CIRCUIT_FEATURES,
    NUM_GLOBAL_FEATURES,
    action_circuit_features,
    circuit_features,
    global_context,
    local_counts,
    neighbourhood_counts,
    own_count,
)

# Local count, number of flipped bits and own count of the bitstring.
NUM_ACTION_FEATURES = 3


class SharedScorer(nn.Module):
    """Network that scores one action of one bitstring from features of that pair.

    The features are the local count of the action, the number of bits it flips, the
    own count of the bitstring and the global context of the circuit. With circuit
    context they also describe the flipped bits in the decoded circuit (gate
    infidelity, readout error, flipped ones) and the gate noise of the whole circuit.
    With neighbourhood counts they also compare how densely the neighbourhoods of the
    destination and of the bitstring were measured. None of them names a qubit, so one
    model corrects circuits of every number of qubits, and two actions with the same
    features get the same logit.
    """

    def __init__(
        self,
        radius: int,
        hidden_sizes: list[int],
        use_circuit_context: bool = True,
        use_neighbourhood: bool = False,
    ) -> None:
        """Build the layers.

        Args:
            radius: Largest number of bits one action flips.
            hidden_sizes: Width of every hidden layer.
            use_circuit_context: Whether features of the decoded circuit are inputs.
            use_neighbourhood: Whether the neighbourhood counts of the destination,
                relative to those of the bitstring, are inputs.
        """
        super().__init__()

        self.radius = radius
        self.use_circuit_context = use_circuit_context
        self.use_neighbourhood = use_neighbourhood

        layers = []
        input_size = NUM_ACTION_FEATURES + NUM_GLOBAL_FEATURES
        if use_circuit_context:
            input_size += NUM_ACTION_CIRCUIT_FEATURES + NUM_CIRCUIT_FEATURES
        if use_neighbourhood:
            input_size += len(NEIGHBOURHOOD_DISTANCES)
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

    def forward(self, counts: torch.Tensor, qubit_features: torch.Tensor) -> torch.Tensor:
        """Compute the action logits of every bitstring of every circuit.

        The number of qubits is read from the number of bitstrings, so circuits of any
        size are accepted, one size per call.

        Args:
            counts: Shot counts of every circuit, shape (circuits, 2**num_qubits).
            qubit_features: Description of every measured bit, shape
                (circuits, num_qubits, NUM_QUBIT_FEATURES).

        Returns:
            Action logits, shape (circuits, 2**num_qubits, actions).

        Raises:
            ValueError: If the number of columns of counts is not a power of two.
        """
        num_bitstrings = counts.shape[-1]
        num_qubits = num_bitstrings.bit_length() - 1
        if num_bitstrings != 2**num_qubits:
            raise ValueError(f"counts has {num_bitstrings} columns, not a power of two")

        masks, destination_table, flipped_bits = (
            table.to(counts.device) for table in _action_tables(num_qubits, self.radius)
        )

        features = self._pair_features(counts, destination_table, flipped_bits)
        if self.use_circuit_context:
            features = torch.cat(
                [features, self._circuit_pair_features(qubit_features, masks)], dim=-1
            )
        if self.use_neighbourhood:
            neighbourhood = neighbourhood_counts(counts, num_qubits)
            # Destination minus bitstring, like the local counts: shape (circuits,
            # 2**num_qubits, actions, distances).
            relative = neighbourhood[:, destination_table] - neighbourhood.unsqueeze(2)
            features = torch.cat([features, relative], dim=-1)

        return self.score(features, flipped_bits)

    def score(self, features: torch.Tensor, flipped_bits: torch.Tensor) -> torch.Tensor:
        """Compute the action logits from the features of every pair.

        Args:
            features: Features of every pair of a bitstring and an action, in the order
                of forward, shape (..., actions, input features).
            flipped_bits: Number of bits every action flips, shape (actions,).

        Returns:
            Action logits, shape (..., actions).
        """
        scores = self.head(self.hidden(features)).squeeze(-1)
        # The identity logit of initial_logits grows with the number of actions, which
        # depends on the number of qubits, so it is added to the output and not learned.
        start = initial_logits(len(flipped_bits)).to(features.device)

        return scores + self.flip_bias[flipped_bits] + start

    def _pair_features(
        self, counts: torch.Tensor, destination_table: torch.Tensor, flipped_bits: torch.Tensor
    ) -> torch.Tensor:
        """Build the features of every pair of a bitstring and an action from the counts.

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

    def _circuit_pair_features(
        self, qubit_features: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        """Build the features of every pair of a bitstring and an action from the circuit.

        Args:
            qubit_features: Description of every measured bit, shape
                (circuits, num_qubits, NUM_QUBIT_FEATURES).
            masks: Masks of the actions, shape (actions,).

        Returns:
            Features, shape (circuits, 2**num_qubits, actions,
            NUM_ACTION_CIRCUIT_FEATURES + NUM_CIRCUIT_FEATURES).
        """
        action_features = action_circuit_features(qubit_features, masks)
        pair_shape = action_features.shape[:-1]

        # The gate noise of the circuit is the same for every bitstring and action.
        summary = circuit_features(qubit_features)
        summary = summary[:, None, None, :].expand(*pair_shape, -1)

        return torch.cat([action_features, summary], dim=-1)


@lru_cache
def _action_tables(num_qubits: int, radius: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the action tables of one number of qubits, once per size.

    Args:
        num_qubits: Number of measured qubits.
        radius: Largest number of bits one action flips.

    Returns:
        Masks of the actions, shape (actions,); destination of every action for every
        bitstring, shape (2**num_qubits, actions); and the number of bits every action
        flips, shape (actions,).
    """
    masks = build_masks(num_qubits, radius)

    return masks, destinations(masks, num_qubits), hamming_weights(masks, num_qubits)
