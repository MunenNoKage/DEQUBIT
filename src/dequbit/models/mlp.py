"""MLP correction model."""

import torch
from torch import nn

from dequbit.actions import build_masks, destinations, initial_logits
from dequbit.features import NUM_GLOBAL_FEATURES, global_context, local_counts, signed_bits


class MLPCorrector(nn.Module):
    """Fully connected network from the inputs of one bitstring to its action logits.

    The inputs of bitstring x are its bits as -1 and +1, the local counts of every
    action and the global context of the circuit. The same network is applied to every
    bitstring of every circuit.
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
        self.register_buffer("bits", signed_bits(num_qubits), persistent=False)
        self.register_buffer("destination_table", destinations(masks, num_qubits), persistent=False)

        layers = []
        input_size = num_qubits + len(masks) + NUM_GLOBAL_FEATURES
        for hidden_size in hidden_sizes:
            layers += [nn.Linear(input_size, hidden_size), nn.GELU()]
            input_size = hidden_size
        self.hidden = nn.Sequential(*layers)
        self.head = nn.Linear(input_size, len(masks))

        # Zero weights make the first output initial_logits for every input.
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.copy_(initial_logits(len(masks)))

    def forward(self, counts: torch.Tensor) -> torch.Tensor:
        """Compute the action logits of every bitstring of every circuit.

        Args:
            counts: Shot counts of every circuit, shape (circuits, 2**num_qubits).

        Returns:
            Action logits, shape (circuits, 2**num_qubits, actions).

        Raises:
            ValueError: If counts does not have 2**num_qubits columns.
        """
        num_circuits, num_bitstrings = counts.shape
        if num_bitstrings != 2**self.num_qubits:
            raise ValueError(f"counts has {num_bitstrings} columns, not {2**self.num_qubits}")

        bits = self.bits.expand(num_circuits, -1, -1)
        local = local_counts(counts, self.destination_table)
        context = global_context(counts, self.num_qubits)
        context = context.unsqueeze(1).expand(-1, num_bitstrings, -1)
        inputs = torch.cat([bits, local, context], dim=-1)

        return self.head(self.hidden(inputs))
