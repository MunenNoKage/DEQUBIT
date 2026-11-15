"""Tests for dequbit.dataset."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from dequbit.dataset import (
    CALIBRATION_NAMES,
    Circuits,
    index_to_bits,
    load_circuits,
    make_dataset,
    unpack_shots,
)
from tests.conftest import (
    FAKE_CIRCUIT_TYPES,
    FAKE_NUM_QUBITS,
    FAKE_READOUT_P01,
    FAKE_SHOTS,
    write_fake_file,
)

NUM_QUBITS = 5
NUM_CIRCUITS = 12


@pytest.mark.parametrize("num_qubits", [5, 10])
def test_unpack_shots_gives_bit_j_weight_two_to_the_j(num_qubits: int) -> None:
    """Packed bits unpack to the integer sum(b_j * 2**j), also for shots longer than a byte.

    Args:
        num_qubits: Number of bits of a shot.
    """
    bits = np.random.default_rng(0).integers(0, 2, size=(100, num_qubits), dtype=np.uint8)
    expected = sum(bits[:, j].astype(np.int64) << j for j in range(num_qubits))

    unpacked = unpack_shots(np.packbits(bits, axis=1), num_qubits)

    np.testing.assert_array_equal(unpacked, expected)


def test_index_to_bits_inverts_integer_encoding() -> None:
    """Bits of every integer 0..31, weighted by 2**j, add up to the integer again."""
    shots = torch.arange(2**NUM_QUBITS)
    weights = 1 << torch.arange(NUM_QUBITS)

    bits = index_to_bits(shots, NUM_QUBITS)

    assert torch.equal((bits * weights).sum(dim=-1), shots)


def test_load_circuits_reads_counts_targets_and_metadata(fake_circuits: Circuits) -> None:
    """Counts add up to the shots, targets to 1, and metadata has one row per circuit."""
    assert fake_circuits.counts.shape == (NUM_CIRCUITS, 2**FAKE_NUM_QUBITS)
    np.testing.assert_array_equal(fake_circuits.counts.sum(axis=1), FAKE_SHOTS)
    np.testing.assert_allclose(fake_circuits.targets.sum(axis=1), 1.0, atol=1e-5)
    assert list(fake_circuits.circuit_types[:2]) == list(FAKE_CIRCUIT_TYPES)
    assert fake_circuits.one_qubit_gates.shape == (NUM_CIRCUITS, FAKE_NUM_QUBITS)
    assert fake_circuits.two_qubit_gates.shape == (NUM_CIRCUITS, FAKE_NUM_QUBITS, FAKE_NUM_QUBITS)
    assert fake_circuits.measured_qubits.shape == (NUM_CIRCUITS, FAKE_NUM_QUBITS)


def test_load_circuits_reads_calibration(fake_circuits: Circuits) -> None:
    """Every calibration array is read; the readout errors are those of the file."""
    assert set(fake_circuits.calibration) == set(CALIBRATION_NAMES)
    np.testing.assert_allclose(fake_circuits.calibration["readout_p01"], FAKE_READOUT_P01)


def test_load_circuits_counts_only_first_shots(tmp_path: Path) -> None:
    """With shots_per_circuit 50 every circuit has the counts of its first 50 shots."""
    path = tmp_path / "noisy.h5"
    write_fake_file(path, num_circuits=3, seed=0)
    with h5py.File(path, "r") as file:
        packed = file["noisy_bitstrings/circuit_00001"][:50]

    circuits = load_circuits(path, shots_per_circuit=50)

    expected = np.bincount(unpack_shots(packed, FAKE_NUM_QUBITS), minlength=2**FAKE_NUM_QUBITS)
    np.testing.assert_array_equal(circuits.counts[1], expected)


def test_load_circuits_rejects_too_few_shots(tmp_path: Path) -> None:
    """Asking for more shots than the file has raises ValueError."""
    path = tmp_path / "noisy.h5"
    write_fake_file(path, num_circuits=3, seed=0)

    with pytest.raises(ValueError):
        load_circuits(path, FAKE_SHOTS + 1)


def test_load_circuits_rejects_incomplete_file(tmp_path: Path) -> None:
    """A file that the generator did not finish raises ValueError."""
    path = tmp_path / "noisy.h5"
    write_fake_file(path, num_circuits=3, seed=0)
    with h5py.File(path, "a") as file:
        file.attrs["complete"] = False

    with pytest.raises(ValueError):
        load_circuits(path, FAKE_SHOTS)


def test_circuits_properties_describe_the_shots(fake_circuits: Circuits) -> None:
    """The number of qubits and shots are those of the file; noisy is counts / shots."""
    assert fake_circuits.num_qubits == FAKE_NUM_QUBITS
    assert fake_circuits.num_shots == FAKE_SHOTS
    np.testing.assert_allclose(fake_circuits.noisy, fake_circuits.counts / FAKE_SHOTS)


def test_make_dataset_item_is_float_counts_and_target(fake_circuits: Circuits) -> None:
    """An item is the float32 counts and P_target of one circuit."""
    counts, target = make_dataset(fake_circuits)[1]

    assert counts.shape == target.shape == (2**FAKE_NUM_QUBITS,)
    assert counts.dtype == target.dtype == torch.float32
    np.testing.assert_array_equal(counts.numpy(), fake_circuits.counts[1])
