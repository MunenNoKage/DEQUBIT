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
FAKE_SHOTS = 200
FAKE_READOUT_P01 = np.array([0.03, 0.05])
FAKE_READOUT_P10 = np.array([0.06, 0.08])
FAKE_SPLIT_CIRCUITS = {"train": 40, "val": 12, "test": 10}
FAKE_CIRCUIT_TYPES = ("bv", "qft")


def write_fake_file(path: Path, num_circuits: int, seed: int) -> None:
    """Write a noisy file in the format of data2/generation.py.

    Shots are drawn from a random target and every bit is flipped with the readout
    errors FAKE_READOUT_P01 and FAKE_READOUT_P10. Every circuit has two one-qubit gates
    on each qubit and one two-qubit gate, and bit j is measured from qubit j.

    Args:
        path: File to write.
        num_circuits: Number of circuits.
        seed: Seed of the targets and shots.
    """
    rng = np.random.default_rng(seed)
    targets = rng.dirichlet(np.ones(2**FAKE_NUM_QUBITS), size=num_circuits)
    two_qubit_gates = np.array([[0, 1], [1, 0]], dtype=np.int32)

    with h5py.File(path, "w") as file:
        file.attrs.update(num_qubits=FAKE_NUM_QUBITS, num_shots=FAKE_SHOTS, complete=True)
        for index, target in enumerate(targets):
            shots = rng.choice(2**FAKE_NUM_QUBITS, size=FAKE_SHOTS, p=target)
            bits = (shots[:, np.newaxis] >> np.arange(FAKE_NUM_QUBITS)) & 1
            flip_probability = np.where(bits == 1, FAKE_READOUT_P10, FAKE_READOUT_P01)
            bits ^= rng.random(bits.shape) < flip_probability
            file[f"probs/circuit_{index:05d}"] = target.astype(np.float32)
            file[f"noisy_bitstrings/circuit_{index:05d}"] = np.packbits(
                bits.astype(np.uint8), axis=1
            )

        metadata = file.create_group("circuit_meta")
        types = [FAKE_CIRCUIT_TYPES[index % 2] for index in range(num_circuits)]
        metadata.create_dataset("circuit_type", data=types, dtype=h5py.string_dtype())
        metadata["gates_1q_per_qubit"] = np.full((num_circuits, FAKE_NUM_QUBITS), 2, np.int32)
        metadata["gates_2q_per_pair"] = np.tile(two_qubit_gates, (num_circuits, 1, 1))
        metadata["measured_qubit"] = np.tile(np.arange(FAKE_NUM_QUBITS), (num_circuits, 1))

        noise_model = file.create_group("noise_model")
        noise_model["readout_p01"] = FAKE_READOUT_P01
        noise_model["readout_p10"] = FAKE_READOUT_P10
        noise_model["t1"] = np.full(FAKE_NUM_QUBITS, 2e-4)
        noise_model["t2"] = np.full(FAKE_NUM_QUBITS, 1e-4)
        noise_model["error_1q"] = np.full(FAKE_NUM_QUBITS, 1e-3)
        noise_model["error_2q"] = np.array([[0.0, 0.01], [0.01, 0.0]])


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
    """Write fake files of every split and a config that trains briefly on them.

    Args:
        tmp_path: Temporary folder of the test, from pytest.

    Returns:
        Path of the config; models and results go into tmp_path.
    """
    for seed, (split, num_circuits) in enumerate(FAKE_SPLIT_CIRCUITS.items()):
        write_fake_file(tmp_path / f"noisy_{split}_N{FAKE_NUM_QUBITS}.h5", num_circuits, seed)

    config = load_config(CONFIG_PATH)
    config["paths"].update(
        {
            "data": str(tmp_path / "noisy_{split}_N{num_qubits}.h5"),
            "model": str(tmp_path / "models"),
            "metrics": str(tmp_path / "results" / "metrics_{split}.json"),
            "figures_dir": str(tmp_path / "results" / "figures"),
        }
    )
    config["data"].update({"num_qubits": FAKE_NUM_QUBITS, "shots_per_circuit": FAKE_SHOTS})
    config["training"].update({"max_steps": 20, "validation_interval": 10})
    config["evaluation"].update({"models": ["shared_scorer"], "seeds": [0, 1]})
    config["logging"]["level"] = "WARNING"

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    return path
