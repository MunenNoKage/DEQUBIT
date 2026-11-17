"""Corrections of circuits with too many qubits for vectors of 2**num_qubits entries.

A circuit is described by its distinct measured bitstrings with their counts, and
P_target by its support, as written by data2/clifford_generation.py. The shared scorer
reads the same pair features as on dense counts, found by looking up x ^ m among the
measured bitstrings. Every corrected distribution sums to 1, so its L1 distance to
P_target only needs the probability it puts on the support of P_target:
L1 = sum over the support of |P(y) - P_target(y)| + 1 - P(support).
"""

from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from dequbit.actions import build_masks, hamming_weights
from dequbit.dataset import (
    CALIBRATION_NAMES,
    GATE_INFIDELITY,
    LOG_READOUT_P01,
    LOG_READOUT_P10,
    METADATA_GROUP,
    NOISE_MODEL_GROUP,
    compute_qubit_features,
    index_to_bits,
)

NOISY_BITSTRINGS_GROUP = "noisy_bitstrings"
NOISY_COUNTS_GROUP = "noisy_counts"
TARGET_BITSTRINGS_GROUP = "target_bitstrings"
TARGET_PROBS_GROUP = "target_probs"
# Bitstrings are int64, and bit 63 would make them negative.
MAX_SPARSE_QUBITS = 63
# (bitstring, action) pairs scored at once, which bounds the memory.
PAIRS_PER_CHUNK = 1_000_000
# Rows of the distance matrix between measured bitstrings computed at once in HAMMER.
HAMMER_ROWS_PER_CHUNK = 256
SPARSE_INFERENCE_MODES = ("expected", "sample")
BITS_PER_BYTE = 8
BYTE_VALUES = 256


@dataclass
class SparseCircuit:
    """One circuit stored by its measured bitstrings.

    Attributes:
        bitstrings: Distinct measured bitstrings, sorted, int64 of shape (distinct,).
        counts: Shots of every distinct bitstring, shape (distinct,).
        target_bitstrings: Support of P_target, shape (support,).
        target_probs: P_target on its support, shape (support,).
        qubit_features: Output of dequbit.dataset.compute_qubit_features for this
            circuit, shape (num_qubits, NUM_QUBIT_FEATURES).
        circuit_type: Name of the circuit family.
        num_qubits: Number of measured qubits.
    """

    bitstrings: np.ndarray
    counts: np.ndarray
    target_bitstrings: np.ndarray
    target_probs: np.ndarray
    qubit_features: np.ndarray
    circuit_type: str
    num_qubits: int

    @property
    def num_shots(self) -> int:
        """Number of shots of the circuit."""
        return int(self.counts.sum())


def load_sparse_circuits(path: Path) -> list[SparseCircuit]:
    """Read the circuits of one file of data2/clifford_generation.py.

    Args:
        path: File, for example data_clifford/clifford_val_N20.h5.

    Returns:
        Circuits of the file in the order of their keys.

    Raises:
        ValueError: If the file was not written completely or has more than
            MAX_SPARSE_QUBITS qubits.
    """
    with h5py.File(path, "r") as file:
        if not file.attrs["complete"]:
            raise ValueError(f"{path} was not written completely")
        num_qubits = int(file.attrs["num_qubits"])
        if num_qubits > MAX_SPARSE_QUBITS:
            raise ValueError(f"{path} has {num_qubits} qubits, at most {MAX_SPARSE_QUBITS}")

        keys = sorted(file[NOISY_BITSTRINGS_GROUP])
        metadata = file[METADATA_GROUP]
        calibration = {name: file[NOISE_MODEL_GROUP][name][:] for name in CALIBRATION_NAMES}
        qubit_features = compute_qubit_features(
            metadata["gates_1q_per_qubit"][:],
            metadata["gates_2q_per_pair"][:],
            metadata["measured_qubit"][:],
            calibration,
        )
        circuit_types = metadata["circuit_type"].asstr()[:]

        return [
            SparseCircuit(
                bitstrings=file[NOISY_BITSTRINGS_GROUP][key][:].astype(np.int64),
                counts=file[NOISY_COUNTS_GROUP][key][:].astype(np.int64),
                target_bitstrings=file[TARGET_BITSTRINGS_GROUP][key][:].astype(np.int64),
                target_probs=file[TARGET_PROBS_GROUP][key][:].astype(np.float64),
                qubit_features=qubit_features[index],
                circuit_type=str(circuit_types[index]),
                num_qubits=num_qubits,
            )
            for index, key in enumerate(keys)
        ]


def lookup_counts(
    bitstrings: torch.Tensor, counts: torch.Tensor, queries: torch.Tensor
) -> torch.Tensor:
    """Find the count of every queried bitstring, 0 for one that was not measured.

    Args:
        bitstrings: Distinct measured bitstrings, sorted, shape (distinct,).
        counts: Counts of the bitstrings, shape (distinct,).
        queries: Bitstrings to look up, any shape (...).

    Returns:
        Counts of the queries, shape (...).
    """
    positions = torch.searchsorted(bitstrings, queries).clamp(max=len(bitstrings) - 1)
    found = bitstrings[positions] == queries

    return torch.where(found, counts[positions], torch.zeros_like(counts[positions]))


def sparse_global_context(counts: torch.Tensor, num_qubits: int) -> torch.Tensor:
    """Compute dequbit.features.global_context of one circuit from its distinct counts.

    Bitstrings that were never measured add nothing to the entropy, the collision
    probability or the number of distinct bitstrings, so the values are the same.

    Args:
        counts: Counts of the distinct measured bitstrings, shape (distinct,).
        num_qubits: Number of measured qubits.

    Returns:
        Context of the circuit, shape (NUM_GLOBAL_FEATURES,).
    """
    num_shots = counts.sum()
    shares = counts / num_shots
    entropy = -torch.special.xlogy(shares, shares).sum()
    log_collision = shares.square().sum().log()
    log_distinct_share = counts.new_tensor(float(len(counts))).log() - num_shots.log()

    return torch.stack([entropy, log_collision, log_distinct_share]) / num_qubits


def sparse_pair_features(
    bitstrings: torch.Tensor,
    counts: torch.Tensor,
    rows: slice,
    masks: torch.Tensor,
    flipped_bits: torch.Tensor,
    qubit_features: torch.Tensor | None,
    num_qubits: int,
) -> torch.Tensor:
    """Build the shared scorer's features of some measured bitstrings and every action.

    The features and their order are those of SharedScorer.forward on dense counts.

    Args:
        bitstrings: Distinct measured bitstrings, sorted, shape (distinct,).
        counts: Their counts as floats, shape (distinct,).
        rows: Positions of the bitstrings whose pairs are built.
        masks: Masks of the actions, shape (actions,).
        flipped_bits: Number of bits every action flips, shape (actions,).
        qubit_features: Qubit features of the circuit, shape
            (num_qubits, NUM_QUBIT_FEATURES), or None without circuit context.
        num_qubits: Number of measured qubits.

    Returns:
        Features, shape (rows, actions, input features of the scorer).
    """
    shots = bitstrings[rows]
    log_own = torch.log1p(counts[rows])
    destinations = shots.unsqueeze(-1) ^ masks
    local = torch.log1p(lookup_counts(bitstrings, counts, destinations)) - log_own.unsqueeze(-1)
    pair_shape = local.shape

    flipped = flipped_bits.to(counts.dtype).expand(pair_shape)
    own = (log_own - counts.sum().log()).unsqueeze(-1).expand(pair_shape)
    context = sparse_global_context(counts, num_qubits).expand(*pair_shape, -1)
    features = [torch.stack([local, flipped, own], dim=-1), context]

    if qubit_features is not None:
        bits = index_to_bits(shots, num_qubits).to(counts.dtype)
        mask_bits = index_to_bits(masks, num_qubits).to(counts.dtype)
        flipped_ones = bits @ mask_bits.T
        infidelity = (qubit_features[:, GATE_INFIDELITY] @ mask_bits.T).expand(pair_shape)
        # The readout error a flip undoes depends on the value of the bit in x.
        undone = torch.where(
            bits.bool(), qubit_features[:, LOG_READOUT_P01], qubit_features[:, LOG_READOUT_P10]
        )
        readout = undone @ mask_bits.T
        gate_noise = qubit_features[:, GATE_INFIDELITY]
        summary = torch.stack([gate_noise.mean(), gate_noise.sum()]).expand(*pair_shape, -1)
        features += [torch.stack([flipped_ones, infidelity, readout], dim=-1), summary]

    return torch.cat(features, dim=-1)


def corrected_support_mass(
    model: nn.Module,
    circuit: SparseCircuit,
    radius: int,
    mode: str,
    generator: torch.Generator,
    device: torch.device,
) -> np.ndarray:
    """Correct one circuit with the shared scorer and return its probability on the support.

    Args:
        model: Trained SharedScorer without neighbourhood counts, on device.
        circuit: Circuit to correct.
        radius: Largest number of bits one action flips.
        mode: "expected" moves P(a | x) of every bitstring, "sample" draws one action
            per shot.
        generator: Random generator on device, used in sample mode.
        device: Device of the computation.

    Returns:
        Probability of the corrected distribution on every bitstring of the support of
        P_target, shape (support,).

    Raises:
        ValueError: If mode is unknown.
    """
    if mode not in SPARSE_INFERENCE_MODES:
        raise ValueError(f"unknown mode {mode!r}, expected one of {SPARSE_INFERENCE_MODES}")

    targets = torch.as_tensor(circuit.target_bitstrings, device=device)
    mass = torch.zeros(len(targets), device=device, dtype=torch.float64)
    for shots, counts, masks, action_probs in _action_probabilities(model, circuit, radius, device):
        sorted_masks, mask_order = masks.sort()
        # Probability that bitstring x moves to every target bitstring, shape
        # (rows, support); 0 where the target is more than radius flips away.
        needed = shots.unsqueeze(-1) ^ targets
        positions = torch.searchsorted(sorted_masks, needed).clamp(max=len(masks) - 1)
        reachable = sorted_masks[positions] == needed
        to_target = torch.gather(action_probs, 1, mask_order[positions])
        to_target = torch.where(reachable, to_target, torch.zeros_like(to_target))
        mass += _moved_mass(counts, to_target, mode, generator)

    return (mass / circuit.num_shots).cpu().numpy()


def corrected_shots(
    model: nn.Module,
    circuit: SparseCircuit,
    radius: int,
    generator: torch.Generator,
    device: torch.device,
) -> SparseCircuit:
    """Draw one action for every shot and return the corrected shots as a circuit.

    Args:
        model: Trained SharedScorer without neighbourhood counts, on device.
        circuit: Circuit to correct.
        radius: Largest number of bits one action flips.
        generator: Random generator on device.
        device: Device of the computation.

    Returns:
        The circuit with the distinct corrected bitstrings and their counts in place of
        the measured ones.
    """
    corrected = []
    for shots, counts, masks, action_probs in _action_probabilities(model, circuit, radius, device):
        # Row of every shot of these bitstrings, drawn a block of shots at a time.
        shot_rows = torch.repeat_interleave(torch.arange(len(shots), device=device), counts.long())
        shots_per_block = max(1, PAIRS_PER_CHUNK // len(masks))
        for start in range(0, len(shot_rows), shots_per_block):
            block = shot_rows[start : start + shots_per_block]
            actions = torch.multinomial(action_probs[block], 1, generator=generator).squeeze(-1)
            corrected.append(shots[block] ^ masks[actions])

    bitstrings, counts = torch.cat(corrected).unique(return_counts=True)

    return replace(circuit, bitstrings=bitstrings.cpu().numpy(), counts=counts.cpu().numpy())


def _action_probabilities(
    model: nn.Module, circuit: SparseCircuit, radius: int, device: torch.device
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield P(a | x) of the measured bitstrings of a circuit, a block of bitstrings at a time.

    Args:
        model: Trained SharedScorer without neighbourhood counts, on device.
        circuit: Circuit to correct.
        radius: Largest number of bits one action flips.
        device: Device of the computation.

    Yields:
        The bitstrings of the block, shape (rows,), their counts, shape (rows,), the
        masks of the actions, shape (actions,), and P(a | x), shape (rows, actions).

    Raises:
        ValueError: If the model reads neighbourhood counts, which need the counts of
            every neighbour of every destination.
    """
    if getattr(model, "use_neighbourhood", False):
        raise ValueError("neighbourhood counts are not available for sparse circuits")

    bitstrings = torch.as_tensor(circuit.bitstrings, device=device)
    counts = torch.as_tensor(circuit.counts, device=device, dtype=torch.float32)
    qubit_features = None
    if model.use_circuit_context:
        qubit_features = torch.as_tensor(circuit.qubit_features, device=device)
    masks = build_masks(circuit.num_qubits, radius).to(device)
    flipped_bits = hamming_weights(masks, circuit.num_qubits)

    rows_per_chunk = max(1, PAIRS_PER_CHUNK // len(masks))
    with torch.no_grad():
        for start in range(0, len(bitstrings), rows_per_chunk):
            rows = slice(start, start + rows_per_chunk)
            features = sparse_pair_features(
                bitstrings, counts, rows, masks, flipped_bits, qubit_features, circuit.num_qubits
            )
            action_probs = model.score(features, flipped_bits).softmax(dim=-1)
            yield bitstrings[rows], counts[rows], masks, action_probs


def _moved_mass(
    counts: torch.Tensor, to_target: torch.Tensor, mode: str, generator: torch.Generator
) -> torch.Tensor:
    """Count the shots of some bitstrings that the correction moves to every target.

    Args:
        counts: Shots of the bitstrings, shape (rows,).
        to_target: Probability that a shot of every bitstring moves to every target,
            shape (rows, support).
        mode: "expected" or "sample".
        generator: Random generator on the device of counts.

    Returns:
        Shots moved to every target, summed over the bitstrings, shape (support,).
    """
    if mode == "expected":
        return (counts.unsqueeze(-1) * to_target).sum(dim=0).double()

    # One multinomial draw per bitstring over the targets and the rest, as a chain of
    # binomials: each target takes its share of the shots the earlier ones left.
    remaining = counts.clone()
    left = torch.ones_like(counts)
    moved = []
    for column in to_target.T:
        share = (column / left.clamp_min(1e-12)).clamp(0, 1)
        drawn = torch.binomial(remaining, share, generator=generator)
        moved.append(drawn.sum())
        remaining = remaining - drawn
        left = left - column

    return torch.stack(moved).double()


def noisy_support_mass(circuit: SparseCircuit) -> np.ndarray:
    """Probability of the noisy distribution on the support of P_target, shape (support,)."""
    counts = lookup_counts(
        torch.as_tensor(circuit.bitstrings),
        torch.as_tensor(circuit.counts),
        torch.as_tensor(circuit.target_bitstrings),
    )

    return counts.numpy() / circuit.num_shots


def oracle_support_mass(circuit: SparseCircuit, rng: np.random.Generator) -> np.ndarray:
    """Probability on the support of as many shots as the circuit has, drawn from P_target.

    Args:
        circuit: Circuit whose P_target is sampled.
        rng: Random generator.

    Returns:
        Probability of the drawn shots on every bitstring of the support, shape (support,).
    """
    return rng.multinomial(circuit.num_shots, circuit.target_probs) / circuit.num_shots


def hammer_support_mass(circuit: SparseCircuit, device: torch.device) -> np.ndarray:
    """Apply HAMMER to one circuit, as dequbit.baselines.hammer_correction.

    Args:
        circuit: Circuit to correct.
        device: Device of the computation.

    Returns:
        Probability of the corrected distribution on every bitstring of the support,
        shape (support,).
    """
    num_qubits = circuit.num_qubits
    num_counted = (num_qubits + 1) // 2
    bitstrings = torch.as_tensor(circuit.bitstrings, device=device)
    probabilities = torch.as_tensor(circuit.counts, device=device, dtype=torch.float64)
    probabilities = probabilities / probabilities.sum()
    table = _byte_popcounts(device)

    strength = torch.zeros(num_qubits + 1, device=device, dtype=torch.float64)
    for rows in _row_chunks(len(bitstrings)):
        distances = _hamming_distances(bitstrings[rows], bitstrings, table)
        weights = probabilities.expand_as(distances)
        strength += torch.bincount(
            distances.flatten(), weights=weights.flatten(), minlength=num_qubits + 1
        )
    # A distance at which no probability was found gets weight 0, as in the paper.
    distance_weights = torch.zeros(num_qubits + 1, device=device, dtype=torch.float64)
    counted = strength[:num_counted]
    distance_weights[:num_counted] = torch.where(
        counted > 0, 1 / counted.clamp_min(1e-300), torch.zeros_like(counted)
    )

    scores = probabilities.clone()
    for rows in _row_chunks(len(bitstrings)):
        distances = _hamming_distances(bitstrings[rows], bitstrings, table)
        # Only neighbours less likely than x add to its score; x itself is not one.
        less_likely = probabilities.unsqueeze(0) < probabilities[rows].unsqueeze(-1)
        scores[rows] += (distance_weights[distances] * probabilities * less_likely).sum(dim=1)

    corrected = scores * probabilities
    corrected = corrected / corrected.sum()
    targets = torch.as_tensor(circuit.target_bitstrings, device=device)

    return lookup_counts(bitstrings, corrected, targets).cpu().numpy()


def l1_from_support_mass(mass: np.ndarray, target_probs: np.ndarray) -> float:
    """L1 distance of a distribution that sums to 1 to P_target, from its mass on the support.

    Args:
        mass: Probability of the distribution on the support of P_target, shape (support,).
        target_probs: P_target on its support, shape (support,).

    Returns:
        The L1 distance.
    """
    return float(np.abs(mass - target_probs).sum() + 1 - mass.sum())


def _row_chunks(num_rows: int) -> list[slice]:
    """Split num_rows rows into slices of HAMMER_ROWS_PER_CHUNK rows."""
    return [
        slice(start, start + HAMMER_ROWS_PER_CHUNK)
        for start in range(0, num_rows, HAMMER_ROWS_PER_CHUNK)
    ]


def _byte_popcounts(device: torch.device) -> torch.Tensor:
    """Number of set bits of every byte value, shape (BYTE_VALUES,)."""
    values = torch.arange(BYTE_VALUES, device=device)

    return sum((values >> bit) & 1 for bit in range(BITS_PER_BYTE))


def _hamming_distances(
    rows: torch.Tensor, columns: torch.Tensor, table: torch.Tensor
) -> torch.Tensor:
    """Hamming distances between two sets of bitstrings, shape (rows, columns).

    Torch has no popcount, so the set bits of the XOR are counted one byte at a time.
    """
    xor = rows.unsqueeze(-1) ^ columns
    distances = torch.zeros_like(xor)
    for byte in range(torch.iinfo(torch.int64).bits // BITS_PER_BYTE):
        distances += table[(xor >> (BITS_PER_BYTE * byte)) & (BYTE_VALUES - 1)]

    return distances
