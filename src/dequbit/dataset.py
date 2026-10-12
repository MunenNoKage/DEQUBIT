"""Reading the .h5 files written by data/generate_data.py and serving circuits for training.

Circuit i is the same circuit in a clean file and in the matching noisy file. Bit j of a
shot is qubit j and has weight 2**j in the index of P_target, the order of the Qiskit
statevector. The ``all_bitstrings`` table in the files lists bitstrings in the reverse
order, so it is not used for indexing.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def unpack_shots(packed: np.ndarray, num_qubits: int) -> np.ndarray:
    """Unpack ``np.packbits`` shots: (shots, bytes) -> (shots, num_qubits)."""
    raise NotImplementedError  # TODO


def load_targets(path: Path) -> np.ndarray:
    """P_target of every circuit in a clean file, shape (circuits, 2**num_qubits)."""
    raise NotImplementedError  # TODO


def load_noisy_shots(path: Path, num_qubits: int) -> np.ndarray:
    """Shots of every circuit in a noisy file, shape (circuits, shots, num_qubits)."""
    raise NotImplementedError  # TODO


def bits_to_index(bits: torch.Tensor) -> torch.Tensor:
    """Index of each bitstring in P_target: (..., num_qubits) -> (...)."""
    raise NotImplementedError  # TODO


def split_train_validation(
    num_circuits: int, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Shuffled circuit indices for training and for validation."""
    raise NotImplementedError  # TODO


class CircuitDataset(Dataset):
    """One item is one circuit: a sample of its noisy shots and its P_target."""

    def __init__(
        self,
        noisy_shots: np.ndarray,
        targets: np.ndarray,
        shots_per_circuit: int,
        seed: int,
    ) -> None:
        """
        Args:
            noisy_shots: shape (circuits, shots, num_qubits).
            targets: shape (circuits, 2**num_qubits).
            shots_per_circuit: number of shots sampled for one item.
            seed: seed of the shot sampling.
        """
        raise NotImplementedError  # TODO

    def __len__(self) -> int:
        raise NotImplementedError  # TODO

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Float shots (shots_per_circuit, num_qubits) and target (2**num_qubits,)."""
        raise NotImplementedError  # TODO
