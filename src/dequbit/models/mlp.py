"""MLP correction model."""

import torch
from torch import nn

from dequbit.actions import build_masks, destinations, initial_logits
from dequbit.features import NUM_GLOBAL_FEATURES, global_context, local_counts, signed_bits


class MLPCorrector(nn.Module):
    """Fully connected network from the inputs of one bitstring to its action logits.

    The inputs of bitstring x are its bits as -1 and +1, the local counts of every
    action and the global context of the circuit. The same network is applied to every
    bitstring of every circuit. Local counts and context can be left out to measure what
    they add.
    """

    def __init__(
        self,
        num_qubits: int,
        radius: int,
        hidden_sizes: list[int],
        use_local_counts: bool = True,
        use_context: bool = True,
    ) -> None:
        """Build the layers.

        Args:
            num_qubits: Number of measured qubits.
            radius: Largest number of bits one action flips.
            hidden_sizes: Width of every hidden layer.
            use_local_counts: Whether the local counts are inputs.
            use_context: Whether the global context of the circuit is an input.
        """
        super().__init__()

        masks = build_masks(num_qubits, radius)
        self.num_qubits = num_qubits
        self.use_local_counts = use_local_counts
        self.use_context = use_context
        # Buffers move with the model to the GPU but are not trained or saved.
        self.register_buffer("bits", signed_bits(num_qubits), persistent=False)
        self.register_buffer("destination_table", destinations(masks, num_qubits), persistent=False)

        layers = []
        input_size = num_qubits
        if use_local_counts:
            input_size += len(masks)
        if use_context:
            input_size += NUM_GLOBAL_FEATURES
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

        inputs = [self.bits.expand(num_circuits, -1, -1)]
        if self.use_local_counts:
            inputs.append(local_counts(counts, self.destination_table))
        if self.use_context:
            context = global_context(counts, self.num_qubits)
            inputs.append(context.unsqueeze(1).expand(-1, num_bitstrings, -1))

        return self.head(self.hidden(torch.cat(inputs, dim=-1)))
