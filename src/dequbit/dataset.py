"""Reading the .h5 files written by data2/generation.py and serving circuits for training.

A shot is the integer x = sum(b_j * 2**j), where b_j is the measured bit of qubit j.
This is the index order of P_target. Only the shot counts of a circuit are kept: every
model input and every correction is computed from them.

Arrays about qubits come in two index orders. Calibration and gate counts are indexed by
device qubit: position i is qubit i of the subset of the device the circuits ran on.
Shots and P_target are indexed by bit. Routing can move a logical qubit on the device,
so bit j of circuit c is measured from device qubit measured_qubits[c, j].
"""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset

TARGETS_GROUP = "probs"
NOISY_SHOTS_GROUP = "noisy_bitstrings"
NOISE_MODEL_GROUP = "noise_model"
METADATA_GROUP = "circuit_meta"
CALIBRATION_NAMES = ("readout_p01", "readout_p10", "t1", "t2", "error_1q", "error_2q")


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


def make_dataset(circuits: Circuits) -> TensorDataset:
    """Build a dataset whose item is the shot counts of one circuit.

    Args:
        circuits: Circuits of one split.

    Returns:
        Dataset of pairs (shot counts of one circuit, its P_target), both float32 of
        shape (2**num_qubits,).
    """
    return TensorDataset(
        torch.from_numpy(circuits.counts).float(), torch.from_numpy(circuits.targets).float()
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
