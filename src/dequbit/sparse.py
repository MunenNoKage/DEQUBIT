"""Corrections of circuits with too many qubits for vectors of 2**num_qubits entries.

A circuit is described by its distinct measured bitstrings with their counts, and
P_target by its support, as written by data2/clifford_generation.py. A bitstring is a row
of 63-bit words, so any number of qubits fits: bit j of the bitstring is bit j % 63 of
word j // 63. The shared scorer reads the same pair features as on dense counts, found
by looking up x ^ m among the measured bitstrings. Every corrected distribution sums to
1, so its L1 distance to P_target only needs the probability it puts on the support of
P_target: L1 = sum over the support of |P(y) - P_target(y)| + 1 - P(support).
"""

from collections.abc import Iterator
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from dequbit.dataset import (
    CALIBRATION_NAMES,
    CLIFFORD_BITSTRINGS_GROUP,
    CLIFFORD_COUNTS_GROUP,
    CLIFFORD_TARGET_BITSTRINGS_GROUP,
    CLIFFORD_TARGET_PROBS_GROUP,
    GATE_INFIDELITY,
    LOG_READOUT_P01,
    LOG_READOUT_P10,
    METADATA_GROUP,
    NOISE_MODEL_GROUP,
    compute_qubit_features,
    index_to_bits,
)

# Bits of one int64 word; the sign bit stays unused, so every word is non-negative.
BITS_PER_WORD = 63
# Odd 64-bit constant (the golden ratio) that mixes the words of a row into one key.
HASH_MULTIPLIER = -7046029254386353131
# (bitstring, action) pairs scored at once, which bounds the memory.
PAIRS_PER_CHUNK = 1_000_000
# Rows of the distance matrix between measured bitstrings computed at once in HAMMER.
HAMMER_ROWS_PER_CHUNK = 256
SPARSE_INFERENCE_MODES = ("expected", "sample")
BITS_PER_BYTE = 8
BYTE_VALUES = 256
BYTES_PER_WORD = 8


@dataclass
class SparseCircuit:
    """One circuit stored by its measured bitstrings.

    Attributes:
        bitstrings: Distinct measured bitstrings as words, int64 of shape
            (distinct, num_words(num_qubits)).
        counts: Shots of every distinct bitstring, shape (distinct,).
        target_bitstrings: Support of P_target as words, shape (support, words).
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


class RowIndex:
    """Find bitstrings in a table of distinct bitstrings.

    The words of every row are mixed into one int64 key, the keys are sorted, and a
    query is found by binary search; its words are then compared, so two rows with the
    same key can only make a query look unmeasured, never match a wrong row.
    """

    def __init__(self, rows: torch.Tensor) -> None:
        """Sort the keys of the table.

        Args:
            rows: Distinct bitstrings as words, shape (rows, words).
        """
        self.rows = rows
        self.keys, self.order = _row_keys(rows).sort()

    def find(self, queries: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Find the row of every queried bitstring.

        Args:
            queries: Bitstrings as words, shape (..., words).

        Returns:
            Row of every query in the table, shape (...), and whether it is there,
            shape (...); the row is meaningless where it is not.
        """
        keys = _row_keys(queries)
        positions = torch.searchsorted(self.keys, keys).clamp(max=len(self.keys) - 1)
        rows = self.order[positions]
        found = (self.keys[positions] == keys) & (self.rows[rows] == queries).all(dim=-1)

        return rows, found


def num_words(num_qubits: int) -> int:
    """Number of 63-bit words of a bitstring of num_qubits bits."""
    return -(-num_qubits // BITS_PER_WORD)


def int_to_words(value: int, num_qubits: int) -> list[int]:
    """Split a bitstring given as a Python integer into its words.

    Args:
        value: Bitstring, bit j = qubit j.
        num_qubits: Number of qubits.

    Returns:
        num_words(num_qubits) words, the lowest bits first.
    """
    mask = (1 << BITS_PER_WORD) - 1

    return [(value >> (BITS_PER_WORD * word)) & mask for word in range(num_words(num_qubits))]


def words_to_bits(words: torch.Tensor, num_qubits: int) -> torch.Tensor:
    """Convert bitstrings given as words to their bits.

    Args:
        words: Bitstrings, shape (..., words).
        num_qubits: Number of qubits.

    Returns:
        Bits, shape (..., num_qubits), where bit j is qubit j.
    """
    parts = []
    for word in range(words.shape[-1]):
        width = min(BITS_PER_WORD, num_qubits - BITS_PER_WORD * word)
        parts.append(index_to_bits(words[..., word], width))

    return torch.cat(parts, dim=-1)


def action_masks(num_qubits: int, radius: int) -> torch.Tensor:
    """Build the masks of the actions as words, in the order of dequbit.actions.build_masks.

    Args:
        num_qubits: Number of qubits.
        radius: Largest number of bits one action flips.

    Returns:
        Masks, shape (actions, num_words(num_qubits)): the identity, then all masks
        with one set bit, then all with two, up to radius.

    Raises:
        ValueError: If radius is not between 1 and num_qubits.
    """
    if not 1 <= radius <= num_qubits:
        raise ValueError(f"radius must be between 1 and {num_qubits}, got {radius}")

    masks = [0]
    for num_flips in range(1, radius + 1):
        masks += [
            sum(1 << qubit for qubit in qubits)
            for qubits in combinations(range(num_qubits), num_flips)
        ]

    return torch.tensor([int_to_words(mask, num_qubits) for mask in masks], dtype=torch.int64)


def load_sparse_circuits(path: Path) -> list[SparseCircuit]:
    """Read the circuits of one file of data2/clifford_generation.py.

    Files with one word per bitstring may store the bitstrings as a vector.

    Args:
        path: File, for example data_clifford/clifford_val_N20.h5.

    Returns:
        Circuits of the file in the order of their keys.

    Raises:
        ValueError: If the file was not written completely.
    """
    with h5py.File(path, "r") as file:
        if not file.attrs["complete"]:
            raise ValueError(f"{path} was not written completely")
        num_qubits = int(file.attrs["num_qubits"])
        words = num_words(num_qubits)

        keys = sorted(file[CLIFFORD_BITSTRINGS_GROUP])
        metadata = file[METADATA_GROUP]
        calibration = {name: file[NOISE_MODEL_GROUP][name][:] for name in CALIBRATION_NAMES}
        qubit_features = compute_qubit_features(
            metadata["gates_1q_per_qubit"][:],
            metadata["gates_2q_per_pair"][:],
            metadata["measured_qubit"][:],
            calibration,
        )
        circuit_types = metadata["circuit_type"].asstr()[:]

        def read_words(group: str, key: str) -> np.ndarray:
            return file[group][key][:].astype(np.int64).reshape(-1, words)

        return [
            SparseCircuit(
                bitstrings=read_words(CLIFFORD_BITSTRINGS_GROUP, key),
                counts=file[CLIFFORD_COUNTS_GROUP][key][:].astype(np.int64),
                target_bitstrings=read_words(CLIFFORD_TARGET_BITSTRINGS_GROUP, key),
                target_probs=file[CLIFFORD_TARGET_PROBS_GROUP][key][:].astype(np.float64),
                qubit_features=qubit_features[index],
                circuit_type=str(circuit_types[index]),
                num_qubits=num_qubits,
            )
            for index, key in enumerate(keys)
        ]


def lookup_counts(index: RowIndex, counts: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
    """Find the count of every queried bitstring, 0 for one that was not measured.

    Args:
        index: Index of the measured bitstrings.
        counts: Counts of the rows of the index, shape (distinct,).
        queries: Bitstrings to look up as words, shape (..., words).

    Returns:
        Counts of the queries, shape (...).
    """
    rows, found = index.find(queries)

    return torch.where(found, counts[rows], torch.zeros_like(counts[rows]))


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
    index: RowIndex,
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
        index: Index of the distinct measured bitstrings.
        counts: Their counts as floats, shape (distinct,).
        rows: Positions of the bitstrings whose pairs are built.
        masks: Masks of the actions as words, shape (actions, words).
        flipped_bits: Number of bits every action flips, shape (actions,).
        qubit_features: Qubit features of the circuit, shape
            (num_qubits, NUM_QUBIT_FEATURES), or None without circuit context.
        num_qubits: Number of measured qubits.

    Returns:
        Features, shape (rows, actions, input features of the scorer).
    """
    shots = index.rows[rows]
    log_own = torch.log1p(counts[rows])
    destinations = shots.unsqueeze(1) ^ masks
    local = torch.log1p(lookup_counts(index, counts, destinations)) - log_own.unsqueeze(-1)
    pair_shape = local.shape

    flipped = flipped_bits.to(counts.dtype).expand(pair_shape)
    own = (log_own - counts.sum().log()).unsqueeze(-1).expand(pair_shape)
    context = sparse_global_context(counts, num_qubits).expand(*pair_shape, -1)
    features = [torch.stack([local, flipped, own], dim=-1), context]

    if qubit_features is not None:
        bits = words_to_bits(shots, num_qubits).to(counts.dtype)
        mask_bits = words_to_bits(masks, num_qubits).to(counts.dtype)
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
    masks = action_masks(circuit.num_qubits, radius).to(device)
    mask_index = RowIndex(masks)
    mass = torch.zeros(len(targets), device=device, dtype=torch.float64)
    for shots, counts, action_probs in _action_probabilities(model, circuit, masks, device):
        # Probability that bitstring x moves to every target bitstring, shape
        # (rows, support); 0 where the target is more than radius flips away.
        actions, reachable = mask_index.find(shots.unsqueeze(1) ^ targets)
        to_target = torch.gather(action_probs, 1, actions)
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
    masks = action_masks(circuit.num_qubits, radius).to(device)
    corrected = []
    for shots, counts, action_probs in _action_probabilities(model, circuit, masks, device):
        # Row of every shot of these bitstrings, drawn a block of shots at a time.
        shot_rows = torch.repeat_interleave(torch.arange(len(shots), device=device), counts.long())
        shots_per_block = max(1, PAIRS_PER_CHUNK // len(masks))
        for start in range(0, len(shot_rows), shots_per_block):
            block = shot_rows[start : start + shots_per_block]
            actions = torch.multinomial(action_probs[block], 1, generator=generator).squeeze(-1)
            corrected.append(shots[block] ^ masks[actions])

    bitstrings, counts = torch.cat(corrected).unique(dim=0, return_counts=True)

    return replace(circuit, bitstrings=bitstrings.cpu().numpy(), counts=counts.cpu().numpy())


def _action_probabilities(
    model: nn.Module, circuit: SparseCircuit, masks: torch.Tensor, device: torch.device
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield P(a | x) of the measured bitstrings of a circuit, a block of bitstrings at a time.

    Args:
        model: Trained SharedScorer without neighbourhood counts, on device.
        circuit: Circuit to correct.
        masks: Masks of the actions from action_masks, on device, shape (actions, words).
        device: Device of the computation.

    Yields:
        The bitstrings of the block, shape (rows, words), their counts, shape (rows,),
        and P(a | x), shape (rows, actions).

    Raises:
        ValueError: If the model reads neighbourhood counts, which need the counts of
            every neighbour of every destination.
    """
    if getattr(model, "use_neighbourhood", False):
        raise ValueError("neighbourhood counts are not available for sparse circuits")

    index = RowIndex(torch.as_tensor(circuit.bitstrings, device=device))
    counts = torch.as_tensor(circuit.counts, device=device, dtype=torch.float32)
    qubit_features = None
    if model.use_circuit_context:
        qubit_features = torch.as_tensor(circuit.qubit_features, device=device)
    flipped_bits = words_to_bits(masks, circuit.num_qubits).sum(dim=-1)

    rows_per_chunk = max(1, PAIRS_PER_CHUNK // len(masks))
    with torch.no_grad():
        for start in range(0, len(counts), rows_per_chunk):
            rows = slice(start, start + rows_per_chunk)
            features = sparse_pair_features(
                index, counts, rows, masks, flipped_bits, qubit_features, circuit.num_qubits
            )
            action_probs = model.score(features, flipped_bits).softmax(dim=-1)
            yield index.rows[rows], counts[rows], action_probs


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
        RowIndex(torch.as_tensor(circuit.bitstrings)),
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

    return lookup_counts(RowIndex(bitstrings), corrected, targets).cpu().numpy()


def l1_from_support_mass(mass: np.ndarray, target_probs: np.ndarray) -> float:
    """L1 distance of a distribution that sums to 1 to P_target, from its mass on the support.

    Args:
        mass: Probability of the distribution on the support of P_target, shape (support,).
        target_probs: P_target on its support, shape (support,).

    Returns:
        The L1 distance.
    """
    return float(np.abs(mass - target_probs).sum() + 1 - mass.sum())


def _row_keys(words: torch.Tensor) -> torch.Tensor:
    """Mix the words of every bitstring into one int64 key, shape (...).

    With one word the key is the word itself, so different bitstrings never share it.
    """
    keys = words[..., 0]
    for word in range(1, words.shape[-1]):
        keys = keys * HASH_MULTIPLIER + words[..., word]

    return keys


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
    """Hamming distances between two sets of bitstrings given as words, (rows, columns).

    Torch has no popcount, so the set bits of the XOR are counted one byte at a time.
    """
    distances = torch.zeros(len(rows), len(columns), dtype=torch.int64, device=rows.device)
    for word in range(rows.shape[-1]):
        xor = rows[:, word].unsqueeze(-1) ^ columns[:, word]
        for byte in range(BYTES_PER_WORD):
            distances += table[(xor >> (BITS_PER_BYTE * byte)) & (BYTE_VALUES - 1)]

    return distances
