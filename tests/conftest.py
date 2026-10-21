"""Shared test fixtures: tiny data files in the format of data/generate_data.py."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from dequbit.config import load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
FAKE_NUM_QUBITS = 2
FAKE_SHOTS = 200
FAKE_FLIP_PROBABILITY = 0.05
FAKE_TRAIN_CIRCUITS = 40
FAKE_TEST_CIRCUITS = 10
FAKE_QASM = b'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nrx(0.5) q[0];\ncx q[0],q[1];\n'


def write_fake_split(directory: Path, name: str, num_circuits: int, seed: int) -> None:
    """Write a clean and a noisy file in the format of data/generate_data.py.

    Shots are drawn from a random target and every bit is flipped with probability
    FAKE_FLIP_PROBABILITY. Every circuit has one cx gate.

    Args:
        directory: Folder of the files.
        name: Suffix of the file names, "train" or "test".
        num_circuits: Number of circuits.
        seed: Seed of the targets and shots.
    """
    rng = np.random.default_rng(seed)
    targets = rng.dirichlet(np.ones(2**FAKE_NUM_QUBITS), size=num_circuits)

    with h5py.File(directory / f"clean_{name}.h5", "w") as clean:
        clean.attrs["num_qubits"] = FAKE_NUM_QUBITS
        for index, target in enumerate(targets):
            clean[f"probs/circuit_{index:04d}"] = target
            clean[f"qasm/circuit_{index:04d}"] = FAKE_QASM

    with h5py.File(directory / f"noisy_{name}.h5", "w") as noisy:
        noisy.attrs["num_qubits"] = FAKE_NUM_QUBITS
        noisy.create_group("noise_model").attrs.update({"p_ro": FAKE_FLIP_PROBABILITY, "p2": 0.02})
        for index, target in enumerate(targets):
            shots = rng.choice(2**FAKE_NUM_QUBITS, size=FAKE_SHOTS, p=target)
            bits = (shots[:, np.newaxis] >> np.arange(FAKE_NUM_QUBITS)) & 1
            bits ^= rng.random(bits.shape) < FAKE_FLIP_PROBABILITY
            noisy[f"noisy_bitstrings/circuit_{index:04d}"] = np.packbits(
                bits.astype(np.uint8), axis=1
            )


@pytest.fixture
def fake_config(tmp_path: Path) -> Path:
    """Write fake training and test files and a config that trains briefly on them.

    Args:
        tmp_path: Temporary folder of the test, from pytest.

    Returns:
        Path of the config; models and results go into tmp_path.
    """
    write_fake_split(tmp_path, "train", FAKE_TRAIN_CIRCUITS, seed=0)
    write_fake_split(tmp_path, "test", FAKE_TEST_CIRCUITS, seed=1)

    config = load_config(CONFIG_PATH)
    config["paths"].update(
        {
            "clean_train": str(tmp_path / "clean_train.h5"),
            "noisy_train": str(tmp_path / "noisy_train.h5"),
            "clean_test": str(tmp_path / "clean_test.h5"),
            "noisy_test": str(tmp_path / "noisy_test.h5"),
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
