"""Shared test fixtures: tiny data files in the format of data2/generation.py."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from dequbit.config import load_config
from dequbit.dataset import Circuits, load_circuits

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
FAKE_NUM_QUBITS = 2
FAKE_QUBIT_COUNTS = (2, 3)
# A number of qubits that only the evaluation sees.
FAKE_UNTRAINED_QUBITS = 4
FAKE_SHOTS = 200
FAKE_READOUT_P01 = np.array([0.03, 0.05])
FAKE_READOUT_P10 = np.array([0.06, 0.08])
FAKE_SPLIT_CIRCUITS = {"train": 40, "val": 12, "test": 10}
FAKE_CIRCUIT_TYPES = ("bv", "qft")


def write_fake_file(
    path: Path, num_circuits: int, seed: int, num_qubits: int = FAKE_NUM_QUBITS
) -> None:
    """Write a noisy file in the format of data2/generation.py.

    Shots are drawn from a random target and every bit is flipped with the readout
    errors FAKE_READOUT_P01 and FAKE_READOUT_P10, repeated over the qubits. Every circuit
    has two one-qubit gates on each qubit and one two-qubit gate between qubits 0 and 1,
    and bit j is measured from qubit j.

    Args:
        path: File to write.
        num_circuits: Number of circuits.
        seed: Seed of the targets and shots.
        num_qubits: Number of measured qubits.
    """
    rng = np.random.default_rng(seed)
    targets = rng.dirichlet(np.ones(2**num_qubits), size=num_circuits)
    readout_p01 = np.resize(FAKE_READOUT_P01, num_qubits)
    readout_p10 = np.resize(FAKE_READOUT_P10, num_qubits)
    first_pair = np.zeros((num_qubits, num_qubits))
    first_pair[0, 1] = first_pair[1, 0] = 1

    with h5py.File(path, "w") as file:
        file.attrs.update(num_qubits=num_qubits, num_shots=FAKE_SHOTS, complete=True)
        for index, target in enumerate(targets):
            shots = rng.choice(2**num_qubits, size=FAKE_SHOTS, p=target)
            bits = (shots[:, np.newaxis] >> np.arange(num_qubits)) & 1
            flip_probability = np.where(bits == 1, readout_p10, readout_p01)
            bits ^= rng.random(bits.shape) < flip_probability
            file[f"probs/circuit_{index:05d}"] = target.astype(np.float32)
            file[f"noisy_bitstrings/circuit_{index:05d}"] = np.packbits(
                bits.astype(np.uint8), axis=1
            )

        metadata = file.create_group("circuit_meta")
        types = [FAKE_CIRCUIT_TYPES[index % 2] for index in range(num_circuits)]
        metadata.create_dataset("circuit_type", data=types, dtype=h5py.string_dtype())
        metadata["gates_1q_per_qubit"] = np.full((num_circuits, num_qubits), 2, np.int32)
        metadata["gates_2q_per_pair"] = np.tile(first_pair.astype(np.int32), (num_circuits, 1, 1))
        metadata["measured_qubit"] = np.tile(np.arange(num_qubits), (num_circuits, 1))

        noise_model = file.create_group("noise_model")
        noise_model["readout_p01"] = readout_p01
        noise_model["readout_p10"] = readout_p10
        noise_model["t1"] = np.full(num_qubits, 2e-4)
        noise_model["t2"] = np.full(num_qubits, 1e-4)
        noise_model["error_1q"] = np.full(num_qubits, 1e-3)
        noise_model["error_2q"] = 0.01 * first_pair


@pytest.fixture
def fake_circuits(tmp_path: Path) -> Circuits:
    """Write one fake file and read it.

    Args:
        tmp_path: Temporary folder of the test, from pytest.

    Returns:
        The 12 circuits of the file with FAKE_SHOTS shots each.
    """
    path = tmp_path / "noisy_fake.h5"
    write_fake_file(path, num_circuits=12, seed=0)

    return load_circuits(path, FAKE_SHOTS)


@pytest.fixture
def fake_config(tmp_path: Path) -> Path:
    """Write fake files of every split and number of qubits and a config for short runs.

    Args:
        tmp_path: Temporary folder of the test, from pytest.

    Returns:
        Path of the config; models and results go into tmp_path.
    """
    for seed, (split, num_circuits) in enumerate(FAKE_SPLIT_CIRCUITS.items()):
        for num_qubits in (*FAKE_QUBIT_COUNTS, FAKE_UNTRAINED_QUBITS):
            path = tmp_path / f"noisy_{split}_N{num_qubits}.h5"
            write_fake_file(path, num_circuits, seed, num_qubits)

    config = load_config(CONFIG_PATH)
    config["paths"].update(
        {
            "data": str(tmp_path / "noisy_{split}_N{num_qubits}.h5"),
            "model": str(tmp_path / "models"),
            "metrics": str(tmp_path / "results" / "metrics_{split}.json"),
            "figures_dir": str(tmp_path / "results" / "figures"),
        }
    )
    config["data"].update(
        {"qubit_counts": list(FAKE_QUBIT_COUNTS), "shots_per_circuit": FAKE_SHOTS}
    )
    config["training"].update({"max_steps": 20, "validation_interval": 10})
    config["evaluation"].update(
        {
            "models": ["shared_scorer"],
            "seeds": [0, 1],
            "qubit_counts": [*FAKE_QUBIT_COUNTS, FAKE_UNTRAINED_QUBITS],
        }
    )
    config["logging"]["level"] = "WARNING"

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    return path
