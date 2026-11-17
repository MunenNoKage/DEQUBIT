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
    SparseCircuit,
    corrected_shots,
    corrected_support_mass,
    hammer_support_mass,
    l1_from_support_mass,
    load_sparse_circuits,
    lookup_counts,
    noisy_support_mass,
    sparse_pair_features,
)
from dequbit.training import predict_distributions
from tests.conftest import FAKE_CIRCUIT_TYPES, FAKE_SHOTS, write_fake_clifford_file

NUM_QUBITS = 4
RADIUS = 2
SEED = 0
CPU = torch.device("cpu")


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
        bitstrings=measured.astype(np.int64),
        counts=counts[measured].astype(np.int64),
        target_bitstrings=np.array([3, 12], dtype=np.int64),
        target_probs=np.array([0.4, 0.6]),
        qubit_features=rng.random((NUM_QUBITS, NUM_QUBIT_FEATURES)).astype(np.float32),
        circuit_type="bv",
        num_qubits=NUM_QUBITS,
    )


def dense_counts(circuit: SparseCircuit) -> torch.Tensor:
    """Counts of every bitstring of the circuit, shape (1, 2**NUM_QUBITS)."""
    counts = torch.zeros(1, 2**NUM_QUBITS)
    counts[0, torch.as_tensor(circuit.bitstrings)] = torch.as_tensor(circuit.counts).float()

    return counts


def trained_like_scorer() -> SharedScorer:
    """Shared scorer with random head weights, as after some training."""
    torch.manual_seed(SEED)
    model = SharedScorer(RADIUS, hidden_sizes=[8])
    torch.nn.init.normal_(model.head.weight)

    return model.eval()


def test_lookup_counts_gives_zero_for_unmeasured_bitstrings() -> None:
    """Measured bitstrings get their count, all others 0."""
    bitstrings = torch.tensor([2, 5, 9])
    counts = torch.tensor([7, 1, 4])

    found = lookup_counts(bitstrings, counts, torch.tensor([[9, 3], [2, 10]]))

    assert found.tolist() == [[4, 0], [7, 0]]


def test_sparse_features_give_the_dense_logits() -> None:
    """The scorer gives the same logits from sparse features as from dense counts."""
    circuit = random_circuit()
    model = trained_like_scorer()
    masks = build_masks(NUM_QUBITS, RADIUS)
    flipped_bits = hamming_weights(masks, NUM_QUBITS)
    counts = torch.as_tensor(circuit.counts).float()
    qubit_features = torch.as_tensor(circuit.qubit_features)

    with torch.no_grad():
        dense = model(dense_counts(circuit), qubit_features.unsqueeze(0))[0]
        features = sparse_pair_features(
            torch.as_tensor(circuit.bitstrings),
            counts,
            slice(0, None),
            masks,
            flipped_bits,
            qubit_features,
            NUM_QUBITS,
        )
        sparse = model.score(features, flipped_bits)

    torch.testing.assert_close(sparse, dense[torch.as_tensor(circuit.bitstrings)])


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

    np.testing.assert_allclose(mass, dense[0, circuit.target_bitstrings].numpy(), rtol=1e-5)


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

    np.testing.assert_allclose(mass, dense[0, circuit.target_bitstrings], rtol=1e-6)


def test_l1_from_support_mass_matches_the_dense_distance() -> None:
    """The support formula gives the full L1 distance of a normalised distribution."""
    circuit = random_circuit()
    noisy = (dense_counts(circuit) / circuit.num_shots).numpy()
    target = np.zeros_like(noisy)
    target[0, circuit.target_bitstrings] = circuit.target_probs

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
    assert np.all(np.diff(corrected.bitstrings) > 0)
    np.testing.assert_allclose(noisy_support_mass(corrected), expected, atol=0.01)


def test_load_sparse_circuits_reads_counts_targets_and_features(tmp_path: Path) -> None:
    """A file in the format of data2/clifford_generation.py is read back."""
    path = tmp_path / "clifford_val_N4.h5"
    write_fake_clifford_file(path, num_circuits=3, seed=SEED, num_qubits=NUM_QUBITS)

    circuits = load_sparse_circuits(path)

    assert len(circuits) == 3
    assert circuits[1].circuit_type == FAKE_CIRCUIT_TYPES[1]
    assert circuits[0].num_shots == FAKE_SHOTS
    assert np.all(np.diff(circuits[0].bitstrings) > 0)
    assert circuits[0].target_probs.tolist() == [1.0]
    assert circuits[0].qubit_features.shape == (NUM_QUBITS, NUM_QUBIT_FEATURES)
    assert circuits[0].qubit_features[0, 0] == pytest.approx(-np.log(0.99))
