"""Tests for dequbit.cli: argument parsing and full train and evaluate runs."""

import json
from pathlib import Path

import pytest
import torch

from dequbit.cli import build_parser, main
from dequbit.dataset import QUBIT_FEATURE_NAMES
from dequbit.models import load_model


def test_parser_reads_train_options() -> None:
    """The train subcommand takes a model name, a seed and config overrides."""
    args = build_parser().parse_args(
        ["--set", "actions.radius=1", "train", "--model", "mlp", "--seed", "3"]
    )

    assert (args.command, args.model, args.seed) == ("train", "mlp", 3)
    assert args.overrides == ["actions.radius=1"]


def test_parser_rejects_unknown_model() -> None:
    """A model name outside MODEL_NAMES stops the parser."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", "--model", "transformer"])


def test_parser_rejects_model_together_with_all() -> None:
    """--all trains every model, so it cannot be combined with --model."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", "--all", "--model", "mlp"])


def test_train_all_and_evaluate_write_models_metrics_and_figures(fake_config: Path) -> None:
    """train --all and evaluate on the test split write every model, metric and figure."""
    directory = fake_config.parent
    figures = directory / "results" / "figures"

    assert main(["--config", str(fake_config), "train", "--all"]) == 0
    assert main(["--config", str(fake_config), "evaluate", "--split", "test"]) == 0

    metrics = json.loads((directory / "results" / "metrics_test.json").read_text(encoding="utf-8"))
    assert (directory / "models" / "shared_scorer_seed1_history.json").exists()
    assert set(metrics["qubits"]) == {"2", "3", "4"}
    for size_metrics in metrics["qubits"].values():
        assert size_metrics["num_circuits"] == 10
        assert {"identity", "oracle", "mix", "linear_map"} <= set(size_metrics["baselines"])
        assert set(size_metrics["models"]["shared_scorer"]["seeds"]) == {"0", "1"}
        assert "shared_scorer" in size_metrics["circuit_types"]["qft"]
    assert (figures / "l1_relative_change_test_N2.png").exists()
    assert (figures / "training_curves_N3.png").exists()
    assert (figures / "l1_relative_change_by_qubits_test.png").exists()
    assert (figures / "l1_relative_change_test_N4.png").exists()
    assert not (figures / "training_curves_N4.png").exists()


def test_train_saves_one_model_per_number_of_qubits(fake_config: Path) -> None:
    """The MLP is saved once per number of qubits, with the radius 1 given by --set.

    With radius 1 the 2-qubit model has 3 actions and the 3-qubit model 4.
    """
    directory = fake_config.parent

    main(["--config", str(fake_config), "--set", "actions.radius=1", "train", "--model", "mlp"])

    model = load_model(directory / "models" / "mlp_N2_seed42.pt")
    larger_model = load_model(directory / "models" / "mlp_N3_seed42.pt")
    features = len(QUBIT_FEATURE_NAMES)
    assert model(torch.ones(1, 4), torch.zeros(1, 2, features)).shape == (1, 4, 3)
    assert larger_model(torch.ones(1, 8), torch.zeros(1, 3, features)).shape == (1, 8, 4)


def test_evaluate_reports_missing_model(fake_config: Path) -> None:
    """Evaluating a model that was never trained raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        main(["--config", str(fake_config), "evaluate", "--split", "validation"])
