"""Tests for dequbit.dataset."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from dequbit.dataset import (
    index_to_bits,
    load_circuits,
    make_dataset,
    shot_counts,
    split_train_validation,
    unpack_shots,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CLEAN_TEST = DATA_DIR / "2_clean_test.h5"
NOISY_TEST = DATA_DIR / "4_noisy_test.h5"
NUM_QUBITS = 5

requires_data = pytest.mark.skipif(
    not (CLEAN_TEST.exists() and NOISY_TEST.exists()), reason="datasets not generated"
)


@pytest.mark.parametrize("num_qubits", [5, 10])
def test_unpack_shots_gives_bit_j_weight_two_to_the_j(num_qubits: int) -> None:
    bits = np.random.default_rng(0).integers(0, 2, size=(100, num_qubits), dtype=np.uint8)
    expected = sum(bits[:, j].astype(np.int64) << j for j in range(num_qubits))

    unpacked = unpack_shots(np.packbits(bits, axis=1), num_qubits)

    np.testing.assert_array_equal(unpacked, expected)


def test_index_to_bits_inverts_integer_encoding() -> None:
    shots = torch.arange(2**NUM_QUBITS)
    weights = 1 << torch.arange(NUM_QUBITS)

    bits = index_to_bits(shots, NUM_QUBITS)

    assert torch.equal((bits * weights).sum(dim=-1), shots)


@requires_data
def test_clean_shots_match_target_in_integer_order() -> None:
    with h5py.File(CLEAN_TEST, "r") as file:
        target = file["probs/circuit_0000"][:]
        packed = file["clean_bitstrings/circuit_0000"][:]

    shots = unpack_shots(packed, NUM_QUBITS)
    histogram = np.bincount(shots, minlength=2**NUM_QUBITS) / len(shots)

    assert np.abs(histogram - target).sum() < 0.1


def test_split_train_validation_is_disjoint_and_covers_all_circuits() -> None:
    train, validation = split_train_validation(100, 0.2, seed=0)

    assert len(validation) == 20
    assert set(train).isdisjoint(validation)
    assert sorted([*train, *validation]) == list(range(100))


def test_split_train_validation_depends_only_on_seed() -> None:
    first = split_train_validation(100, 0.2, seed=0)
    second = split_train_validation(100, 0.2, seed=0)

    np.testing.assert_array_equal(first[1], second[1])


@requires_data
def test_load_circuits_shapes_and_normalised_targets() -> None:
    shots, targets = load_circuits(CLEAN_TEST, NOISY_TEST, NUM_QUBITS)

    assert shots.shape == (200, 10000)
    assert targets.shape == (200, 2**NUM_QUBITS)
    assert shots.min() >= 0 and shots.max() < 2**NUM_QUBITS
    np.testing.assert_allclose(targets.sum(axis=1), 1.0, atol=1e-5)


@requires_data
def test_load_circuits_rejects_wrong_number_of_qubits() -> None:
    with pytest.raises(ValueError):
        load_circuits(CLEAN_TEST, NOISY_TEST, NUM_QUBITS + 1)


def test_shot_counts_counts_every_circuit_separately() -> None:
    shots = np.array([[0, 3, 3], [31, 31, 0]])

    counts = shot_counts(shots, NUM_QUBITS)

    expected = np.zeros((2, 2**NUM_QUBITS), dtype=np.int64)
    expected[0, [0, 3]] = [1, 2]
    expected[1, [0, 31]] = [1, 2]
    np.testing.assert_array_equal(counts, expected)


def test_make_dataset_item_is_counts_of_first_shots() -> None:
    shots = np.full((3, 50), 7, dtype=np.int64)
    shots[:, :10] = 5
    targets = np.full((3, 2**NUM_QUBITS), 1 / 2**NUM_QUBITS, dtype=np.float32)

    counts, target = make_dataset(shots, targets, shots_per_circuit=10, num_qubits=NUM_QUBITS)[1]

    assert counts.shape == target.shape == (2**NUM_QUBITS,)
    assert counts.dtype == target.dtype == torch.float32
    assert counts[5] == 10 and counts.sum() == 10


def test_make_dataset_rejects_too_few_shots() -> None:
    shots = np.zeros((3, 50), dtype=np.int64)
    targets = np.full((3, 2**NUM_QUBITS), 1 / 2**NUM_QUBITS, dtype=np.float32)

    with pytest.raises(ValueError):
        make_dataset(shots, targets, shots_per_circuit=51, num_qubits=NUM_QUBITS)
