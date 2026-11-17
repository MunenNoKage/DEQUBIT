"""Reading the .h5 files written by data2/generation.py and serving circuits for training.

A shot is the integer x = sum(b_j * 2**j), where b_j is the measured bit of qubit j.
This is the index order of P_target. Only the shot counts of a circuit are kept: every
model input and every correction is computed from them.

Arrays about qubits come in two index orders. Calibration and gate counts are indexed by
device qubit: position i is qubit i of the subset of the device the circuits ran on.
Shots and P_target are indexed by bit. Routing can move a logical qubit on the device,
so bit j of circuit c is measured from device qubit measured_qubits[c, j].
"""

from dataclasses import dataclass, replace
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset

TARGETS_GROUP = "probs"
NOISY_SHOTS_GROUP = "noisy_bitstrings"
NOISE_MODEL_GROUP = "noise_model"
METADATA_GROUP = "circuit_meta"
# Groups of the files of data2/clifford_generation.py.
CLIFFORD_BITSTRINGS_GROUP = "noisy_bitstrings"
CLIFFORD_COUNTS_GROUP = "noisy_counts"
CLIFFORD_TARGET_BITSTRINGS_GROUP = "target_bitstrings"
CLIFFORD_TARGET_PROBS_GROUP = "target_probs"
CALIBRATION_NAMES = ("readout_p01", "readout_p10", "t1", "t2", "error_1q", "error_2q")
# Description of every measured bit, in this order along the last axis of
# Circuits.qubit_features.
QUBIT_FEATURE_NAMES = ("gate_infidelity", "log_readout_p01", "log_readout_p10")
GATE_INFIDELITY, LOG_READOUT_P01, LOG_READOUT_P10 = range(len(QUBIT_FEATURE_NAMES))
# Readout errors are given as log(rate / READOUT_REFERENCE_RATE), so a typical rate is
# near 0; rates below READOUT_RATE_FLOOR are raised to it to keep the log finite.
READOUT_REFERENCE_RATE = 0.02
READOUT_RATE_FLOOR = 1e-4
# Error rates that perturb_calibration changes, and the largest rate it gives: a readout
# error of 0.5 or more could not be inverted.
PERTURBED_RATE_NAMES = ("readout_p01", "readout_p10", "error_1q", "error_2q")
MAX_PERTURBED_RATE = 0.45


@dataclass
class Circuits:
    """Circuits of one split and one number of qubits.

    Attributes:
        counts: Noisy shot counts, shape (circuits, 2**num_qubits).
        targets: P_target of every circuit, shape (circuits, 2**num_qubits).
        circuit_types: Name of the circuit family of every circuit, shape (circuits,).
        one_qubit_gates: One-qubit gates of the transpiled circuit on every device
            qubit, shape (circuits, num_qubits).
        two_qubit_gates: Two-qubit gates of the transpiled circuit between every pair of
            device qubits, symmetric, shape (circuits, num_qubits, num_qubits).
        measured_qubits: Device qubit measured into every bit, shape
            (circuits, num_qubits).
        calibration: Error rates of the device qubits by name: readout_p01 (reading 1
            after preparing 0), readout_p10, t1, t2 and error_1q of shape (num_qubits,),
            error_2q of shape (num_qubits, num_qubits) with 0 for unconnected pairs.
    """

    counts: np.ndarray
    targets: np.ndarray
    circuit_types: np.ndarray
    one_qubit_gates: np.ndarray
    two_qubit_gates: np.ndarray
    measured_qubits: np.ndarray
    calibration: dict[str, np.ndarray]

    @property
    def num_qubits(self) -> int:
        """Number of measured qubits."""
        return self.measured_qubits.shape[1]

    @property
    def num_shots(self) -> int:
        """Number of shots of every circuit."""
        return int(self.counts[0].sum())

    @property
    def noisy(self) -> np.ndarray:
        """Noisy distribution of every circuit, shape (circuits, 2**num_qubits)."""
        return self.counts / self.counts.sum(axis=-1, keepdims=True)

    @property
    def qubit_features(self) -> np.ndarray:
        """Describe the errors that can hit every measured bit, from the decoded circuit.

        Returns:
            Output of compute_qubit_features for these circuits, shape
            (circuits, num_qubits, len(QUBIT_FEATURE_NAMES)).
        """
        return compute_qubit_features(
            self.one_qubit_gates, self.two_qubit_gates, self.measured_qubits, self.calibration
        )


def compute_qubit_features(
    one_qubit_gates: np.ndarray,
    two_qubit_gates: np.ndarray,
    measured_qubits: np.ndarray,
    calibration: dict[str, np.ndarray],
) -> np.ndarray:
    """Describe the errors that can hit every measured bit, from the decoded circuit.

    The gate infidelity of a bit is -sum(log(1 - error)) over every gate of the
    transpiled circuit on the device qubit measured into that bit, with two-qubit gates
    counted on both of their qubits. A qubit that no gate touched has 0. The readout
    errors are those of the same device qubit.

    Args:
        one_qubit_gates: One-qubit gates on every device qubit, shape
            (circuits, num_qubits).
        two_qubit_gates: Two-qubit gates between every pair of device qubits, shape
            (circuits, num_qubits, num_qubits).
        measured_qubits: Device qubit measured into every bit, shape
            (circuits, num_qubits).
        calibration: Error rates of the device qubits, as in Circuits.calibration.

    Returns:
        Features in the order of QUBIT_FEATURE_NAMES, float32 of shape
        (circuits, num_qubits, len(QUBIT_FEATURE_NAMES)).
    """
    one_qubit = one_qubit_gates * -np.log1p(-calibration["error_1q"])
    two_qubit = (two_qubit_gates * -np.log1p(-calibration["error_2q"])).sum(axis=-1)
    infidelity = np.take_along_axis(one_qubit + two_qubit, measured_qubits, axis=1)

    p01 = calibration["readout_p01"][measured_qubits]
    p10 = calibration["readout_p10"][measured_qubits]
    log_p01 = np.log(np.maximum(p01, READOUT_RATE_FLOOR) / READOUT_REFERENCE_RATE)
    log_p10 = np.log(np.maximum(p10, READOUT_RATE_FLOOR) / READOUT_REFERENCE_RATE)

    return np.stack([infidelity, log_p01, log_p10], axis=-1).astype(np.float32)


def unpack_shots(packed: np.ndarray, num_qubits: int) -> np.ndarray:
    """Convert shots packed with np.packbits to integers.

    Args:
        packed: Packed shots, shape (shots, bytes).
        num_qubits: Number of measured qubits.

    Returns:
        Shots as integers, shape (shots,).
    """
    bits = np.unpackbits(packed, axis=1)[:, :num_qubits].astype(np.int64)
    weights = 1 << np.arange(num_qubits)

    return bits @ weights


def index_to_bits(shots: torch.Tensor, num_qubits: int) -> torch.Tensor:
    """Convert integer shots to their bits.

    Args:
        shots: Shots as integers, any shape (...).
        num_qubits: Number of measured qubits.

    Returns:
        Bits of the shots, shape (..., num_qubits), where bit j is qubit j.
    """
    positions = torch.arange(num_qubits, device=shots.device)

    return (shots.unsqueeze(-1) >> positions) & 1


def load_circuits(path: Path, shots_per_circuit: int) -> Circuits:
    """Read the circuits of one noisy file.

    The shots of a circuit are stored in random order, so the first shots_per_circuit
    shots are a random sample of them.

    Args:
        path: File with the noisy shots, for example data_v4/noisy_train_N5.h5.
        shots_per_circuit: Number of shots counted for each circuit.

    Returns:
        Circuits of the file.

    Raises:
        ValueError: If the file was not written completely or its circuits have fewer
            than shots_per_circuit shots.
    """
    with h5py.File(path, "r") as file:
        if not file.attrs["complete"]:
            raise ValueError(f"{path} was not written completely")
        if file.attrs["num_shots"] < shots_per_circuit:
            raise ValueError(
                f"{path} has {file.attrs['num_shots']} shots per circuit, "
                f"not {shots_per_circuit}"
            )

        num_qubits = int(file.attrs["num_qubits"])
        keys = sorted(file[TARGETS_GROUP])
        counts = [
            _count_shots(file[NOISY_SHOTS_GROUP][key][:shots_per_circuit], num_qubits)
            for key in keys
        ]
        targets = [file[TARGETS_GROUP][key][:] for key in keys]
        metadata = file[METADATA_GROUP]

        return Circuits(
            counts=np.stack(counts),
            targets=np.stack(targets),
            circuit_types=metadata["circuit_type"].asstr()[:],
            one_qubit_gates=metadata["gates_1q_per_qubit"][:],
            two_qubit_gates=metadata["gates_2q_per_pair"][:],
            measured_qubits=metadata["measured_qubit"][:],
            calibration={name: file[NOISE_MODEL_GROUP][name][:] for name in CALIBRATION_NAMES},
        )


def load_clifford_circuits(path: Path, shots_per_circuit: int) -> Circuits:
    """Read a file of data2/clifford_generation.py into dense counts and targets.

    The file stores the distinct measured bitstrings and the support of P_target; here
    both become vectors of 2**num_qubits entries, so this is for small circuits only.

    Args:
        path: File, for example data_clifford/clifford_train_N5.h5.
        shots_per_circuit: Number of shots every circuit must have; the file stores
            counts, not the shots in order, so fewer shots cannot be taken.

    Returns:
        Circuits of the file.

    Raises:
        ValueError: If the file was not written completely or its circuits do not have
            shots_per_circuit shots.
    """
    with h5py.File(path, "r") as file:
        if not file.attrs["complete"]:
            raise ValueError(f"{path} was not written completely")
        if file.attrs["num_shots"] != shots_per_circuit:
            raise ValueError(
                f"{path} has {file.attrs['num_shots']} shots per circuit, "
                f"not {shots_per_circuit}"
            )

        num_qubits = int(file.attrs["num_qubits"])
        keys = sorted(file[CLIFFORD_BITSTRINGS_GROUP])
        counts = np.zeros((len(keys), 2**num_qubits), dtype=np.int64)
        targets = np.zeros((len(keys), 2**num_qubits), dtype=np.float32)
        for row, key in enumerate(keys):
            counts[row, file[CLIFFORD_BITSTRINGS_GROUP][key][:]] = file[CLIFFORD_COUNTS_GROUP][key][
                :
            ]
            support = file[CLIFFORD_TARGET_BITSTRINGS_GROUP][key][:]
            targets[row, support] = file[CLIFFORD_TARGET_PROBS_GROUP][key][:]
        metadata = file[METADATA_GROUP]

        return Circuits(
            counts=counts,
            targets=targets,
            circuit_types=metadata["circuit_type"].asstr()[:],
            one_qubit_gates=metadata["gates_1q_per_qubit"][:],
            two_qubit_gates=metadata["gates_2q_per_pair"][:],
            measured_qubits=metadata["measured_qubit"][:],
            calibration={name: file[NOISE_MODEL_GROUP][name][:] for name in CALIBRATION_NAMES},
        )


def concatenate_circuits(first: Circuits, second: Circuits) -> Circuits:
    """Put the circuits of two sets of the same device qubits into one.

    Args:
        first: Circuits of one number of qubits.
        second: Circuits of the same number of qubits and the same calibration.

    Returns:
        The circuits of first, then those of second.

    Raises:
        ValueError: If the two sets have different calibrations.
    """
    for name in CALIBRATION_NAMES:
        if not np.allclose(first.calibration[name], second.calibration[name]):
            raise ValueError(f"the circuits have different calibrations ({name})")

    return replace(
        first,
        counts=np.concatenate([first.counts, second.counts]),
        targets=np.concatenate([first.targets, second.targets]),
        circuit_types=np.concatenate([first.circuit_types, second.circuit_types]),
        one_qubit_gates=np.concatenate([first.one_qubit_gates, second.one_qubit_gates]),
        two_qubit_gates=np.concatenate([first.two_qubit_gates, second.two_qubit_gates]),
        measured_qubits=np.concatenate([first.measured_qubits, second.measured_qubits]),
    )


def select_circuits(circuits: Circuits, keep: np.ndarray) -> Circuits:
    """Keep some of the circuits, with the same calibration.

    Args:
        circuits: Circuits of one split.
        keep: True for every circuit to keep, shape (circuits,).

    Returns:
        The kept circuits.
    """
    return replace(
        circuits,
        counts=circuits.counts[keep],
        targets=circuits.targets[keep],
        circuit_types=circuits.circuit_types[keep],
        one_qubit_gates=circuits.one_qubit_gates[keep],
        two_qubit_gates=circuits.two_qubit_gates[keep],
        measured_qubits=circuits.measured_qubits[keep],
    )


def perturb_calibration(
    circuits: Circuits, relative_noise: float, rng: np.random.Generator
) -> Circuits:
    """Replace the calibrated error rates by drifted ones, as on a device recalibrated later.

    Every rate of PERTURBED_RATE_NAMES is multiplied by its own factor
    exp(relative_noise * z) with z drawn from N(0, 1), so relative_noise 0.2 changes a
    typical rate by about 20 percent. The two-qubit errors stay symmetric, and every
    rate stays at most MAX_PERTURBED_RATE. The shots are not changed: only what the
    corrections are told about the device.

    Args:
        circuits: Circuits of one split.
        relative_noise: Standard deviation of the log of the factors; 0 changes nothing.
        rng: Random generator of the factors.

    Returns:
        The same circuits with the drifted calibration.
    """
    calibration = dict(circuits.calibration)
    for name in PERTURBED_RATE_NAMES:
        rates = calibration[name]
        factors = np.exp(relative_noise * rng.standard_normal(rates.shape))
        if rates.ndim == 2:
            # One factor per pair of qubits, the same in both directions.
            factors = np.triu(factors, 1) + np.triu(factors, 1).T
        calibration[name] = np.minimum(rates * factors, MAX_PERTURBED_RATE)

    return replace(circuits, calibration=calibration)


def make_dataset(circuits: Circuits) -> TensorDataset:
    """Build a dataset whose item is the shot counts and qubit features of one circuit.

    Args:
        circuits: Circuits of one split.

    Returns:
        Dataset of triples (shot counts of one circuit, shape (2**num_qubits,); its
        qubit features, shape (num_qubits, len(QUBIT_FEATURE_NAMES)); its P_target,
        shape (2**num_qubits,)), all float32.
    """
    return TensorDataset(
        torch.from_numpy(circuits.counts).float(),
        torch.from_numpy(circuits.qubit_features),
        torch.from_numpy(circuits.targets).float(),
    )


def _count_shots(packed: np.ndarray, num_qubits: int) -> np.ndarray:
    """Count how many shots of one circuit fall on each bitstring.

    Args:
        packed: Packed shots of the circuit, shape (shots, bytes).
        num_qubits: Number of measured qubits.

    Returns:
        Counts, shape (2**num_qubits,).
    """
    return np.bincount(unpack_shots(packed, num_qubits), minlength=2**num_qubits)
