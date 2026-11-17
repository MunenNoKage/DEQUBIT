"""Tests for dequbit.dataset."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from dequbit.dataset import (
    CALIBRATION_NAMES,
    GATE_INFIDELITY,
    LOG_READOUT_P01,
    MAX_PERTURBED_RATE,
    QUBIT_FEATURE_NAMES,
    READOUT_REFERENCE_RATE,
    Circuits,
    index_to_bits,
    load_circuits,
    make_dataset,
    perturb_calibration,
    select_circuits,
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


def test_make_dataset_item_is_counts_qubit_features_and_target(fake_circuits: Circuits) -> None:
    """An item is the float32 counts, qubit features and P_target of one circuit."""
    counts, qubit_features, target = make_dataset(fake_circuits)[1]

    assert counts.shape == target.shape == (2**FAKE_NUM_QUBITS,)
    assert qubit_features.shape == (FAKE_NUM_QUBITS, len(QUBIT_FEATURE_NAMES))
    assert counts.dtype == qubit_features.dtype == target.dtype == torch.float32
    np.testing.assert_array_equal(counts.numpy(), fake_circuits.counts[1])


def test_qubit_features_sum_gate_errors_and_read_calibration(tmp_path: Path) -> None:
    """Gate infidelity adds every gate on the measured qubit; readout is log(rate / 0.02).

    The fake files have two one-qubit gates with error 1e-3 on every qubit and one
    two-qubit gate with error 0.01 between qubits 0 and 1, so qubit 2 has only the
    one-qubit gates.
    """
    path = tmp_path / "noisy.h5"
    write_fake_file(path, num_circuits=2, seed=0, num_qubits=3)

    features = load_circuits(path, FAKE_SHOTS).qubit_features

    one_qubit = -2 * np.log1p(-1e-3)
    two_qubit = -np.log1p(-0.01)
    np.testing.assert_allclose(
        features[0, :, GATE_INFIDELITY], [one_qubit + two_qubit] * 2 + [one_qubit], rtol=1e-6
    )
    expected_p01 = np.log(np.resize(FAKE_READOUT_P01, 3) / READOUT_REFERENCE_RATE)
    np.testing.assert_allclose(features[1, :, LOG_READOUT_P01], expected_p01, rtol=1e-6)


def test_select_circuits_keeps_rows_of_every_array(fake_circuits: Circuits) -> None:
    """Keeping the bv circuits keeps their counts, types and gates, and the calibration."""
    keep = fake_circuits.circuit_types == "bv"

    selected = select_circuits(fake_circuits, keep)

    assert list(selected.circuit_types) == ["bv"] * int(keep.sum())
    np.testing.assert_array_equal(selected.counts, fake_circuits.counts[keep])
    assert selected.two_qubit_gates.shape[0] == selected.measured_qubits.shape[0] == keep.sum()
    assert selected.calibration is fake_circuits.calibration


def test_perturb_calibration_without_noise_changes_nothing(fake_circuits: Circuits) -> None:
    """Relative noise 0 keeps every rate and the qubit features."""
    perturbed = perturb_calibration(fake_circuits, 0.0, np.random.default_rng(0))

    np.testing.assert_allclose(perturbed.qubit_features, fake_circuits.qubit_features)


def test_perturb_calibration_drifts_rates_but_not_shots(fake_circuits: Circuits) -> None:
    """Rates change, two-qubit errors stay symmetric and bounded, counts stay the same."""
    perturbed = perturb_calibration(fake_circuits, 0.5, np.random.default_rng(0))

    error_2q = perturbed.calibration["error_2q"]
    assert not np.allclose(perturbed.calibration["readout_p01"], FAKE_READOUT_P01)
    np.testing.assert_allclose(error_2q, error_2q.T)
    assert (error_2q > 0).sum() == (fake_circuits.calibration["error_2q"] > 0).sum()
    assert perturbed.calibration["readout_p10"].max() <= MAX_PERTURBED_RATE
    np.testing.assert_array_equal(perturbed.counts, fake_circuits.counts)
