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
NOISE_MODEL_GROUP = "noise_model"
QASM_GROUP = "qasm"
TWO_QUBIT_GATES = ("cx", "cz")


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


def load_noise_parameters(noisy_path: Path) -> dict[str, float]:
    """Read the parameters of the noise model that produced the noisy shots.

    Args:
        noisy_path: File with the noisy shots, for example data/3_noisy_train.h5.

    Returns:
        Parameter name to value, for example "p_ro" (readout flip probability of one
        qubit) and "p2" (depolarising probability of a two-qubit gate).
    """
    with h5py.File(noisy_path, "r") as file:
        attributes = file[NOISE_MODEL_GROUP].attrs

        return {name: float(value) for name, value in attributes.items()}


def load_two_qubit_gate_counts(clean_path: Path) -> np.ndarray:
    """Count the two-qubit gates of every circuit in its stored OpenQASM code.

    Args:
        clean_path: File with the circuits, for example data/1_clean_train.h5.

    Returns:
        Number of two-qubit gates of every circuit, shape (circuits,).
    """
    with h5py.File(clean_path, "r") as file:
        circuits = file[QASM_GROUP]
        codes = [circuits[f"circuit_{index:04d}"][()].decode() for index in range(len(circuits))]

    return np.array([_count_two_qubit_gates(code) for code in codes])


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


def shot_counts(shots: np.ndarray, num_qubits: int) -> np.ndarray:
    """Count how many shots of every circuit fall on each bitstring.

    Args:
        shots: Shots as integers, shape (circuits, shots).
        num_qubits: Number of measured qubits.

    Returns:
        Counts, shape (circuits, 2**num_qubits).
    """
    num_bitstrings = 2**num_qubits

    # Shifting the shots of circuit c by c * 2**num_qubits gives every circuit its own
    # range of bins, so one bincount counts all circuits.
    offsets = np.arange(len(shots))[:, np.newaxis] * num_bitstrings
    counts = np.bincount((shots + offsets).ravel(), minlength=len(shots) * num_bitstrings)

    return counts.reshape(len(shots), num_bitstrings)


def make_dataset(
    shots: np.ndarray, targets: np.ndarray, shots_per_circuit: int, num_qubits: int
) -> TensorDataset:
    """Build a dataset whose item is the shot counts of one circuit.

    The shots of a circuit are stored in random order, so the first shots_per_circuit
    shots are a random sample of them. Every input of the model is computed from the
    counts, so the order of the shots is not kept.

    Args:
        shots: Shots as integers, shape (circuits, shots).
        targets: P_target of every circuit, shape (circuits, 2**num_qubits).
        shots_per_circuit: Number of shots kept for each circuit.
        num_qubits: Number of measured qubits.

    Returns:
        Dataset of pairs (shot counts of one circuit, its P_target), both float32 of
        shape (2**num_qubits,).

    Raises:
        ValueError: If the circuits have fewer than shots_per_circuit shots.
    """
    if shots.shape[1] < shots_per_circuit:
        raise ValueError(f"circuits have {shots.shape[1]} shots, not {shots_per_circuit}")

    counts = shot_counts(shots[:, :shots_per_circuit], num_qubits)

    return TensorDataset(torch.from_numpy(counts).float(), torch.from_numpy(targets).float())


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


def _count_two_qubit_gates(qasm: str) -> int:
    """Count the two-qubit gate instructions of one OpenQASM program.

    Args:
        qasm: OpenQASM 2 code with one instruction per line.

    Returns:
        Number of lines whose instruction is one of TWO_QUBIT_GATES.
    """
    instructions = [line.split(maxsplit=1)[0] for line in qasm.splitlines() if line.strip()]

    return sum(instruction in TWO_QUBIT_GATES for instruction in instructions)
