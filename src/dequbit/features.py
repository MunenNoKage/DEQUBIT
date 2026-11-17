"""Model inputs computed from the shot counts of a circuit.

Every input is about one bitstring x or about the whole circuit, so its size grows with
the number of actions or qubits, never with 2**num_qubits per bitstring.
"""

import torch

from dequbit.dataset import (
    GATE_INFIDELITY,
    LOG_READOUT_P01,
    LOG_READOUT_P10,
    QUBIT_FEATURE_NAMES,
    index_to_bits,
)

NUM_GLOBAL_FEATURES = 3
# Flipped ones, gate infidelity and log readout error of the flipped bits.
NUM_ACTION_CIRCUIT_FEATURES = 3
# Mean and sum of the gate infidelity of the bits.
NUM_CIRCUIT_FEATURES = 2
NUM_QUBIT_FEATURES = len(QUBIT_FEATURE_NAMES)
# Hamming distances of the neighbourhood counts.
NEIGHBOURHOOD_DISTANCES = (1, 2)


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


def neighbourhood_counts(counts: torch.Tensor, num_qubits: int) -> torch.Tensor:
    """Measure how densely the neighbourhood of every bitstring was measured.

    For every distance d of NEIGHBOURHOOD_DISTANCES the feature is log(mean count + 1)
    of the bitstrings d bit flips away. A correct outcome sits in a region that noise
    filled with many shots, while an outcome made by noise usually does not; the same
    observation is behind HAMMER. The mean, not the sum, keeps the scale the same for
    every number of qubits.

    Args:
        counts: Shot counts of every circuit, shape (circuits, 2**num_qubits).
        num_qubits: Number of measured qubits.

    Returns:
        Neighbourhood counts, shape (circuits, 2**num_qubits, len(NEIGHBOURHOOD_DISTANCES)).
    """
    bitstrings = torch.arange(2**num_qubits, device=counts.device)
    bits = index_to_bits(bitstrings, num_qubits)

    features = []
    for distance in NEIGHBOURHOOD_DISTANCES:
        masks = bitstrings[bits.sum(dim=-1) == distance]
        neighbours = bitstrings.unsqueeze(-1) ^ masks
        features.append(torch.log1p(counts[:, neighbours].mean(dim=-1)))

    return torch.stack(features, dim=-1)


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


def action_circuit_features(qubit_features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Describe the bits every action flips in every bitstring, from the decoded circuit.

    The three features of bitstring x and action a are the number of flipped bits that
    are 1 in x, the sum of the gate infidelity of the flipped bits, and the sum of the
    log readout error that the flip would undo: reading 1 for a prepared 0 where the
    bit of x is 1, reading 0 for a prepared 1 where it is 0. All three are 0 for the
    identity. None of them names a qubit, so they have the same meaning for every
    number of qubits.

    Args:
        qubit_features: Output of Circuits.qubit_features, shape
            (circuits, num_qubits, NUM_QUBIT_FEATURES).
        masks: Masks of the actions, shape (actions,).

    Returns:
        Features, shape (circuits, 2**num_qubits, actions, NUM_ACTION_CIRCUIT_FEATURES).
    """
    num_circuits, num_qubits, _ = qubit_features.shape
    mask_bits = index_to_bits(masks, num_qubits).to(qubit_features.dtype)
    bits = index_to_bits(torch.arange(2**num_qubits, device=masks.device), num_qubits)
    bits = bits.to(qubit_features.dtype)

    flipped_ones = (bits @ mask_bits.T).expand(num_circuits, -1, -1)
    infidelity = qubit_features[..., GATE_INFIDELITY] @ mask_bits.T
    infidelity = infidelity.unsqueeze(1).expand_as(flipped_ones)
    # The readout error a flip undoes depends on the value of the bit in x.
    undone = torch.where(
        bits.bool(),
        qubit_features[:, None, :, LOG_READOUT_P01],
        qubit_features[:, None, :, LOG_READOUT_P10],
    )
    readout = undone @ mask_bits.T

    return torch.stack([flipped_ones, infidelity, readout], dim=-1)


def circuit_features(qubit_features: torch.Tensor) -> torch.Tensor:
    """Summarize how much gate noise the decoded circuit carries.

    Args:
        qubit_features: Output of Circuits.qubit_features, shape
            (circuits, num_qubits, NUM_QUBIT_FEATURES).

    Returns:
        Mean and sum over the bits of the gate infidelity, shape
        (circuits, NUM_CIRCUIT_FEATURES).
    """
    infidelity = qubit_features[..., GATE_INFIDELITY]

    return torch.stack([infidelity.mean(dim=-1), infidelity.sum(dim=-1)], dim=-1)
