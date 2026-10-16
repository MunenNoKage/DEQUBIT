"""Model inputs computed from the shot counts of a circuit.

Every input is about one bitstring x or about the whole circuit, so its size grows with
the number of actions or qubits, never with 2**num_qubits per bitstring.
"""

import torch

from dequbit.dataset import index_to_bits

NUM_GLOBAL_FEATURES = 3


def signed_bits(num_qubits: int) -> torch.Tensor:
    """Write the bits of every bitstring as -1 and +1.

    Args:
        num_qubits: Number of measured qubits.

    Returns:
        Bits of every bitstring x, shape (2**num_qubits, num_qubits), where 0 becomes -1
        and 1 becomes +1.
    """
    bits = index_to_bits(torch.arange(2**num_qubits), num_qubits)

    return 2.0 * bits - 1.0


def local_counts(counts: torch.Tensor, destination_table: torch.Tensor) -> torch.Tensor:
    """Compare the count of every action destination with the count of the bitstring.

    The feature is log(count(x ^ m) + 1) - log(count(x) + 1). A noisy shot whose
    neighbour x ^ m was measured much more often is likely a corrupted copy of it. The
    difference does not depend on the number of shots, and it is 0 for the identity.

    Args:
        counts: Shot counts of every circuit, shape (circuits, 2**num_qubits).
        destination_table: Output of dequbit.actions.destinations, shape
            (2**num_qubits, actions).

    Returns:
        Local counts, shape (circuits, 2**num_qubits, actions).
    """
    log_counts = torch.log1p(counts)

    return log_counts[:, destination_table] - log_counts.unsqueeze(-1)


def own_count(counts: torch.Tensor) -> torch.Tensor:
    """Compute how often every bitstring was measured, on a log scale.

    Args:
        counts: Shot counts of every circuit, shape (circuits, 2**num_qubits).

    Returns:
        log(count(x) + 1) - log(shots) of every bitstring, shape
        (circuits, 2**num_qubits).
    """
    num_shots = counts.sum(dim=-1, keepdim=True)

    return torch.log1p(counts) - num_shots.log()


def global_context(counts: torch.Tensor, num_qubits: int) -> torch.Tensor:
    """Describe how spread out the shots of every circuit are.

    The three features are the entropy of the shot distribution, the log of its collision
    probability (sum of squared shares) and the log share of distinct bitstrings among
    the shots. Stronger noise spreads the shots, so these tell the model how much to
    correct. They are divided by num_qubits to keep the same scale for every number of
    qubits.

    Args:
        counts: Shot counts of every circuit, shape (circuits, 2**num_qubits).
        num_qubits: Number of measured qubits.

    Returns:
        Context of every circuit, shape (circuits, NUM_GLOBAL_FEATURES).

    Raises:
        ValueError: If a circuit has no shots.
    """
    num_shots = counts.sum(dim=-1)
    if (num_shots == 0).any():
        raise ValueError("every circuit needs at least one shot")

    shares = counts / num_shots.unsqueeze(-1)
    entropy = -torch.special.xlogy(shares, shares).sum(dim=-1)
    log_collision = shares.square().sum(dim=-1).log()
    num_distinct = (counts > 0).sum(dim=-1).to(counts.dtype)
    log_distinct_share = num_distinct.log() - num_shots.log()
    context = torch.stack([entropy, log_collision, log_distinct_share], dim=-1)

    return context / num_qubits
