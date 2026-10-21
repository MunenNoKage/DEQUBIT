"""Tests for dequbit.cli: argument parsing and a full train and evaluate run."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from dequbit.cli import build_parser, main, model_path
from dequbit.config import load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
NUM_QUBITS = 2
NUM_SHOTS = 200
GATE_QASM = b'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nrx(0.5) q[0];\ncx q[0],q[1];\n'


def write_fake_split(directory: Path, name: str, num_circuits: int, seed: int) -> None:
    """Write a clean and a noisy file in the format of data/generate_data.py.

    Shots are drawn from the target and every bit is flipped with probability 0.05.

    Args:
        directory: Folder of the files.
        name: Prefix of the file names, for example "train".
        num_circuits: Number of circuits.
        seed: Seed of the targets and shots.
    """
    rng = np.random.default_rng(seed)
    targets = rng.dirichlet(np.ones(2**NUM_QUBITS), size=num_circuits)

    with h5py.File(directory / f"clean_{name}.h5", "w") as clean:
        clean.attrs["num_qubits"] = NUM_QUBITS
        for index, target in enumerate(targets):
            clean[f"probs/circuit_{index:04d}"] = target
            clean[f"qasm/circuit_{index:04d}"] = GATE_QASM

    with h5py.File(directory / f"noisy_{name}.h5", "w") as noisy:
        noisy.attrs["num_qubits"] = NUM_QUBITS
        noisy.create_group("noise_model").attrs.update({"p_ro": 0.05, "p2": 0.02})
        for index, target in enumerate(targets):
            shots = rng.choice(2**NUM_QUBITS, size=NUM_SHOTS, p=target)
            bits = (shots[:, np.newaxis] >> np.arange(NUM_QUBITS)) & 1
            bits ^= rng.random(bits.shape) < 0.05
            noisy[f"noisy_bitstrings/circuit_{index:04d}"] = np.packbits(
                bits.astype(np.uint8), axis=1
            )


def write_fake_config(directory: Path) -> Path:
    """Write a config that trains briefly on fake files inside directory.

    Args:
        directory: Folder of the fake data, models and results.

    Returns:
        Path of the written config.
    """
    config = load_config(CONFIG_PATH)
    config["paths"].update(
        {
            "clean_train": str(directory / "clean_train.h5"),
            "noisy_train": str(directory / "noisy_train.h5"),
            "clean_test": str(directory / "clean_test.h5"),
            "noisy_test": str(directory / "noisy_test.h5"),
            "model": str(directory / "models"),
            "metrics": str(directory / "results" / "metrics_{split}.json"),
            "figures_dir": str(directory / "results" / "figures"),
        }
    )
    config["data"].update({"num_qubits": NUM_QUBITS, "shots_per_circuit": NUM_SHOTS})
    config["training"].update({"max_steps": 20, "validation_interval": 10})
    config["evaluation"].update({"models": ["shared_scorer"], "seeds": [0, 1]})
    config["logging"]["level"] = "WARNING"

    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    return path


def test_parser_reads_train_options() -> None:
    """The train subcommand takes a model name and a seed."""
    args = build_parser().parse_args(["train", "--model", "mlp", "--seed", "3"])

    assert (args.command, args.model, args.seed) == ("train", "mlp", 3)


def test_parser_rejects_unknown_model() -> None:
    """A model name outside MODEL_NAMES stops the parser."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", "--model", "transformer"])


def test_model_path_names_model_and_seed() -> None:
    """Every model and seed gets its own file in paths.model."""
    config = {"paths": {"model": "models"}}

    assert model_path(config, "mlp", 2) == Path("models") / "mlp_seed2.pt"


def test_train_and_evaluate_write_models_metrics_and_figure(tmp_path: Path) -> None:
    """Training two seeds and evaluating on the test split writes every expected output."""
    write_fake_split(tmp_path, "train", num_circuits=40, seed=0)
    write_fake_split(tmp_path, "test", num_circuits=10, seed=1)
    config_path = write_fake_config(tmp_path)

    for seed in ("0", "1"):
        code = main(
            ["--config", str(config_path), "train", "--model", "shared_scorer", "--seed", seed]
        )
        assert code == 0
    code = main(["--config", str(config_path), "evaluate", "--split", "test"])

    metrics = json.loads((tmp_path / "results" / "metrics_test.json").read_text(encoding="utf-8"))
    assert code == 0
    assert (tmp_path / "models" / "shared_scorer_seed1_history.json").exists()
    assert metrics["num_circuits"] == 10
    assert {"identity", "oracle", "mix", "linear_map"} <= set(metrics["baselines"])
    assert set(metrics["models"]["shared_scorer"]["seeds"]) == {"0", "1"}
    assert (tmp_path / "results" / "figures" / "l1_relative_change_test.png").exists()


def test_evaluate_reports_missing_model(tmp_path: Path) -> None:
    """Evaluating a model that was never trained raises FileNotFoundError."""
    write_fake_split(tmp_path, "train", num_circuits=40, seed=0)
    config_path = write_fake_config(tmp_path)

    with pytest.raises(FileNotFoundError):
        main(["--config", str(config_path), "evaluate", "--split", "validation"])
