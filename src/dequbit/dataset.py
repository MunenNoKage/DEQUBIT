"""Reading the .h5 files written by data/generate_data.py and serving circuits for training.

A shot is stored as the integer x = sum(b_j * 2**j), where b_j is the measured bit of
qubit j. This is the index order of P_target, the order of the Qiskit statevector. The
all_bitstrings table in the files lists bitstrings in the reverse bit order, so it is
not used.
"""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset

TARGETS_GROUP = "probs"
NOISY_SHOTS_GROUP = "noisy_bitstrings"


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


def load_circuits(
    clean_path: Path, noisy_path: Path, num_qubits: int
) -> tuple[np.ndarray, np.ndarray]:
    """Read the noisy shots and P_target of every circuit of one split.

    Args:
        clean_path: File with the ideal distributions, for example data/1_clean_train.h5.
        noisy_path: File with the noisy shots of the same circuits.
        num_qubits: Number of qubits the files must have.

    Returns:
        Shots as integers, shape (circuits, shots), and P_target, shape
        (circuits, 2**num_qubits).

    Raises:
        ValueError: If a file has another number of qubits or the files have different
            numbers of circuits.
    """
    targets = np.stack(_read_circuits(clean_path, TARGETS_GROUP, num_qubits))
    packed = _read_circuits(noisy_path, NOISY_SHOTS_GROUP, num_qubits)
    shots = np.stack([unpack_shots(circuit, num_qubits) for circuit in packed])

    if len(shots) != len(targets):
        raise ValueError(f"{noisy_path} has {len(shots)} circuits, {clean_path} has {len(targets)}")

    return shots, targets


def split_train_validation(
    num_circuits: int, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split circuit indices into training and validation circuits.

    Args:
        num_circuits: Number of circuits in the training file.
        validation_fraction: Share of the circuits held out for validation.
        seed: Seed of the random split.

    Returns:
        Sorted indices of the training circuits and of the validation circuits.
    """
    order = np.random.default_rng(seed).permutation(num_circuits)
    num_validation = round(num_circuits * validation_fraction)

    return np.sort(order[num_validation:]), np.sort(order[:num_validation])


def make_dataset(shots: np.ndarray, targets: np.ndarray, shots_per_circuit: int) -> TensorDataset:
    """Build a dataset whose item is one circuit.

    The shots of a circuit are stored in random order, so the first shots_per_circuit
    shots are a random sample of them.

    Args:
        shots: Shots as integers, shape (circuits, shots).
        targets: P_target of every circuit, shape (circuits, 2**num_qubits).
        shots_per_circuit: Number of shots kept for each circuit.

    Returns:
        Dataset of pairs (shots of one circuit, its P_target).
    """
    circuit_shots = torch.from_numpy(shots[:, :shots_per_circuit])

    return TensorDataset(circuit_shots, torch.from_numpy(targets))


def _read_circuits(path: Path, group: str, num_qubits: int) -> list[np.ndarray]:
    """Read the array of every circuit in one group of a file.

    Args:
        path: Dataset file.
        group: Group holding one array per circuit, for example "probs".
        num_qubits: Number of qubits the file must have.

    Returns:
        Arrays of the circuits, in circuit order.

    Raises:
        ValueError: If the file has another number of qubits.
    """
    with h5py.File(path, "r") as file:
        if file.attrs["num_qubits"] != num_qubits:
            raise ValueError(f"{path} has {file.attrs['num_qubits']} qubits, not {num_qubits}")

        circuits = file[group]

        return [circuits[f"circuit_{index:04d}"][:] for index in range(len(circuits))]
