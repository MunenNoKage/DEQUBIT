"""Tests for dequbit.sparse: on small circuits it must agree with the dense code."""

from pathlib import Path

import numpy as np
import pytest
import torch

from dequbit.actions import build_masks, destinations, hamming_weights
from dequbit.baselines import hammer_correction
from dequbit.features import NUM_QUBIT_FEATURES
from dequbit.metrics import l1_distance
from dequbit.models.shared_scorer import SharedScorer
from dequbit.sparse import (
    BITS_PER_WORD,
    RowIndex,
    SparseCircuit,
    action_masks,
    corrected_shots,
    corrected_support_mass,
    hammer_support_mass,
    int_to_words,
    l1_from_support_mass,
    load_sparse_circuits,
    lookup_counts,
    noisy_support_mass,
    num_words,
    sparse_pair_features,
    words_to_bits,
)
from dequbit.training import predict_distributions
from tests.conftest import FAKE_CIRCUIT_TYPES, FAKE_SHOTS, write_fake_clifford_file

NUM_QUBITS = 4
RADIUS = 2
SEED = 0
CPU = torch.device("cpu")
# More qubits than one word holds, and the qubits that carry the shots there.
WIDE_QUBITS = 70
LOW_QUBITS = list(range(6))
HIGH_QUBITS = list(range(64, 70))


def random_circuit(seed: int = SEED) -> SparseCircuit:
    """Draw a circuit of NUM_QUBITS qubits with some bitstrings never measured.

    Args:
        seed: Seed of the counts and features.

    Returns:
        Circuit whose P_target is supported on bitstrings 3 and 12.
    """
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, 50, size=2**NUM_QUBITS)
    counts[[1, 6]] = 0
    measured = np.flatnonzero(counts)

    return SparseCircuit(
        bitstrings=measured.astype(np.int64)[:, np.newaxis],
        counts=counts[measured].astype(np.int64),
        target_bitstrings=np.array([[3], [12]], dtype=np.int64),
        target_probs=np.array([0.4, 0.6]),
        qubit_features=rng.random((NUM_QUBITS, NUM_QUBIT_FEATURES)).astype(np.float32),
        circuit_type="bv",
        num_qubits=NUM_QUBITS,
    )


def wide_circuit(qubits: list[int], seed: int = SEED) -> SparseCircuit:
    """Draw a WIDE_QUBITS-qubit circuit whose shots differ only on the given qubits.

    The same seed gives the same circuit up to which qubits carry the shots: the
    features of the active qubits follow them, and all other qubits are alike.

    Args:
        qubits: The six qubits that carry the shots.
        seed: Seed of the counts and features.

    Returns:
        The circuit, with P_target on two bitstrings of the active qubits.
    """
    rng = np.random.default_rng(seed)
    patterns = np.flatnonzero(rng.integers(0, 2, size=2 ** len(qubits)))
    counts = rng.integers(1, 40, size=len(patterns))
    active_features = rng.random((len(qubits), NUM_QUBIT_FEATURES)).astype(np.float32)
    qubit_features = np.full((WIDE_QUBITS, NUM_QUBIT_FEATURES), 0.1, dtype=np.float32)
    qubit_features[qubits] = active_features

    def place(pattern: int) -> list[int]:
        value = sum(1 << qubit for bit, qubit in enumerate(qubits) if (pattern >> bit) & 1)
        return int_to_words(value, WIDE_QUBITS)

    return SparseCircuit(
        bitstrings=np.array([place(int(p)) for p in patterns], dtype=np.int64),
        counts=counts.astype(np.int64),
        target_bitstrings=np.array([place(int(patterns[0])), place(int(patterns[-1]))]),
        target_probs=np.array([0.5, 0.5]),
        qubit_features=qubit_features,
        circuit_type="ghz",
        num_qubits=WIDE_QUBITS,
    )


def dense_counts(circuit: SparseCircuit) -> torch.Tensor:
    """Counts of every bitstring of the circuit, shape (1, 2**NUM_QUBITS)."""
    counts = torch.zeros(1, 2**NUM_QUBITS)
    counts[0, torch.as_tensor(circuit.bitstrings[:, 0])] = torch.as_tensor(circuit.counts).float()

    return counts


def trained_like_scorer() -> SharedScorer:
    """Shared scorer with random head weights, as after some training."""
    torch.manual_seed(SEED)
    model = SharedScorer(RADIUS, hidden_sizes=[8])
    torch.nn.init.normal_(model.head.weight)

    return model.eval()


def hammer_reference(circuit: SparseCircuit) -> np.ndarray:
    """HAMMER by Algorithm 1 of Tannu et al. with Python integers, on the support."""
    values = [
        sum(int(w) << (BITS_PER_WORD * i) for i, w in enumerate(row)) for row in circuit.bitstrings
    ]
    probs = circuit.counts / circuit.counts.sum()
    distances = [[bin(a ^ b).count("1") for b in values] for a in values]
    strength = np.zeros(circuit.num_qubits + 1)
    for row in distances:
        for column, distance in enumerate(row):
            strength[distance] += probs[column]
    counted = (circuit.num_qubits + 1) // 2
    weights = [
        1 / strength[d] if d < counted and strength[d] > 0 else 0 for d in range(len(strength))
    ]
    scores = [
        probs[i]
        + sum(
            weights[distances[i][j]] * probs[j] for j in range(len(values)) if probs[j] < probs[i]
        )
        for i in range(len(values))
    ]
    corrected = np.array(scores) * probs
    corrected = corrected / corrected.sum()
    position = {value: i for i, value in enumerate(values)}
    targets = [
        sum(int(w) << (BITS_PER_WORD * i) for i, w in enumerate(row))
        for row in circuit.target_bitstrings
    ]

    return np.array([corrected[position[t]] if t in position else 0.0 for t in targets])


def test_words_round_trip_beyond_one_word() -> None:
    """A 70-qubit bitstring splits into two words and back into the same bits."""
    value = (1 << 69) | (1 << 63) | (1 << 62) | 5

    words = torch.tensor([int_to_words(value, WIDE_QUBITS)])
    bits = words_to_bits(words, WIDE_QUBITS)[0]

    assert num_words(WIDE_QUBITS) == 2
    assert torch.nonzero(bits).flatten().tolist() == [0, 2, 62, 63, 69]


def test_action_masks_match_build_masks_in_one_word() -> None:
    """With one word the masks are those of dequbit.actions, and their count grows as C(n, k)."""
    masks = action_masks(NUM_QUBITS, RADIUS)

    assert masks[:, 0].tolist() == build_masks(NUM_QUBITS, RADIUS).tolist()
    assert len(action_masks(WIDE_QUBITS, RADIUS)) == 1 + WIDE_QUBITS + WIDE_QUBITS * 69 // 2


def test_row_index_finds_rows_of_two_words() -> None:
    """Rows are found by all their words, and absent rows are reported as absent."""
    rows = torch.tensor([[5, 1], [5, 2], [7, 0]])
    index = RowIndex(rows)

    found_rows, found = index.find(torch.tensor([[5, 2], [7, 0], [5, 3]]))

    assert found.tolist() == [True, True, False]
    assert found_rows[:2].tolist() == [1, 2]
    assert lookup_counts(index, torch.tensor([10, 20, 30]), torch.tensor([[7, 0]])).tolist() == [30]


def test_sparse_features_give_the_dense_logits() -> None:
    """The scorer gives the same logits from sparse features as from dense counts."""
    circuit = random_circuit()
    model = trained_like_scorer()
    masks = action_masks(NUM_QUBITS, RADIUS)
    flipped_bits = hamming_weights(masks[:, 0], NUM_QUBITS)
    counts = torch.as_tensor(circuit.counts).float()
    qubit_features = torch.as_tensor(circuit.qubit_features)

    with torch.no_grad():
        dense = model(dense_counts(circuit), qubit_features.unsqueeze(0))[0]
        features = sparse_pair_features(
            RowIndex(torch.as_tensor(circuit.bitstrings)),
            counts,
            slice(0, None),
            masks,
            flipped_bits,
            qubit_features,
            NUM_QUBITS,
        )
        sparse = model.score(features, flipped_bits)

    torch.testing.assert_close(sparse, dense[torch.as_tensor(circuit.bitstrings[:, 0])])


def test_expected_correction_matches_the_dense_one_on_the_support() -> None:
    """Expected mode puts the same probability on the target bitstrings as the dense code."""
    circuit = random_circuit()
    model = trained_like_scorer()
    destination_table = destinations(build_masks(NUM_QUBITS, RADIUS), NUM_QUBITS)
    qubit_features = torch.as_tensor(circuit.qubit_features).unsqueeze(0)

    with torch.no_grad():
        _, dense = predict_distributions(
            model, dense_counts(circuit), qubit_features, destination_table
        )
    mass = corrected_support_mass(
        model, circuit, RADIUS, "expected", torch.Generator().manual_seed(SEED), CPU
    )

    np.testing.assert_allclose(mass, dense[0, circuit.target_bitstrings[:, 0]].numpy(), rtol=1e-5)


def test_correction_does_not_depend_on_which_word_holds_the_qubits() -> None:
    """Moving the active qubits from the first word to the second changes nothing."""
    model = trained_like_scorer()
    generator = torch.Generator().manual_seed(SEED)

    low = corrected_support_mass(
        model, wide_circuit(LOW_QUBITS), RADIUS, "expected", generator, CPU
    )
    high = corrected_support_mass(
        model, wide_circuit(HIGH_QUBITS), RADIUS, "expected", generator, CPU
    )

    np.testing.assert_allclose(high, low, rtol=1e-5)
    assert low.sum() > 0


def test_sampled_correction_is_close_to_the_expected_one() -> None:
    """With many shots, sampled actions put about the expected probability on the support."""
    circuit = random_circuit()
    circuit.counts = circuit.counts * 1000
    model = trained_like_scorer()
    generator = torch.Generator().manual_seed(SEED)

    expected = corrected_support_mass(model, circuit, RADIUS, "expected", generator, CPU)
    sampled = corrected_support_mass(model, circuit, RADIUS, "sample", generator, CPU)

    np.testing.assert_allclose(sampled, expected, atol=0.01)
    assert np.allclose(sampled * circuit.num_shots, np.round(sampled * circuit.num_shots))


def test_sparse_hammer_matches_the_dense_one() -> None:
    """HAMMER on measured bitstrings gives the dense result on the support."""
    circuit = random_circuit()
    dense = hammer_correction((dense_counts(circuit) / circuit.num_shots).numpy())

    mass = hammer_support_mass(circuit, CPU)

    np.testing.assert_allclose(mass, dense[0, circuit.target_bitstrings[:, 0]], rtol=1e-6)


def test_sparse_hammer_beyond_one_word_matches_algorithm_1() -> None:
    """On 70 qubits HAMMER agrees with a direct implementation on Python integers."""
    circuit = wide_circuit([1, 30, 62, 63, 66, 69])

    mass = hammer_support_mass(circuit, CPU)

    np.testing.assert_allclose(mass, hammer_reference(circuit), rtol=1e-10)


def test_l1_from_support_mass_matches_the_dense_distance() -> None:
    """The support formula gives the full L1 distance of a normalised distribution."""
    circuit = random_circuit()
    noisy = (dense_counts(circuit) / circuit.num_shots).numpy()
    target = np.zeros_like(noisy)
    target[0, circuit.target_bitstrings[:, 0]] = circuit.target_probs

    distance = l1_from_support_mass(noisy_support_mass(circuit), circuit.target_probs)

    assert distance == pytest.approx(l1_distance(noisy, target)[0])


def test_neighbourhood_scorer_is_refused() -> None:
    """Neighbourhood counts would need every neighbour of every destination."""
    model = SharedScorer(RADIUS, hidden_sizes=[8], use_neighbourhood=True)

    with pytest.raises(ValueError, match="neighbourhood"):
        corrected_support_mass(model, random_circuit(), RADIUS, "expected", torch.Generator(), CPU)


def test_corrected_shots_keep_the_shots_and_follow_the_expected_correction() -> None:
    """Corrected shots keep their number and put about the expected mass on the support."""
    circuit = random_circuit()
    circuit.counts = circuit.counts * 200
    model = trained_like_scorer()
    generator = torch.Generator().manual_seed(SEED)

    corrected = corrected_shots(model, circuit, RADIUS, generator, CPU)
    expected = corrected_support_mass(model, circuit, RADIUS, "expected", generator, CPU)

    assert corrected.num_shots == circuit.num_shots
    assert np.all(np.diff(corrected.bitstrings[:, 0]) > 0)
    np.testing.assert_allclose(noisy_support_mass(corrected), expected, atol=0.01)


def test_load_sparse_circuits_reads_counts_targets_and_features(tmp_path: Path) -> None:
    """A file in the format of data2/clifford_generation.py is read back."""
    path = tmp_path / "clifford_val_N4.h5"
    write_fake_clifford_file(path, num_circuits=3, seed=SEED, num_qubits=NUM_QUBITS)

    circuits = load_sparse_circuits(path)

    assert len(circuits) == 3
    assert circuits[1].circuit_type == FAKE_CIRCUIT_TYPES[1]
    assert circuits[0].num_shots == FAKE_SHOTS
    assert circuits[0].bitstrings.shape[1] == 1
    assert circuits[0].target_probs.tolist() == [1.0]
    assert circuits[0].qubit_features.shape == (NUM_QUBITS, NUM_QUBIT_FEATURES)
    assert circuits[0].qubit_features[0, 0] == pytest.approx(-np.log(0.99))
