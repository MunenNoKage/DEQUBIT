"""Tests for dequbit.cli: argument parsing and full train and evaluate runs."""

import json
from pathlib import Path

import pytest
import torch

from dequbit.cli import build_parser, main
from dequbit.config import load_config
from dequbit.dataset import QUBIT_FEATURE_NAMES
from dequbit.models import build_model, load_model, save_model
from dequbit.pipeline import model_path


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
        low, high = size_metrics["median_intervals"]["shared_scorer"]
        assert low <= size_metrics["models"]["shared_scorer"]["median_mean"] <= high
        comparison = size_metrics["comparisons"]["shared_scorer"]
        assert comparison["baseline"] not in ("identity", "oracle", "shared_scorer")
        assert 0 <= comparison["wilcoxon_p"] <= 1
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


def test_ablate_trains_variant_models_and_writes_their_metrics(fake_config: Path) -> None:
    """A training variant saves its models apart and writes metrics under its name."""
    directory = fake_config.parent

    assert main(["--config", str(fake_config), "ablate", "no_circuit_context"]) == 0

    models = directory / "models" / "ablations" / "no_circuit_context"
    metrics_path = Path(load_config(fake_config)["paths"]["ablations"])
    assert (models / "shared_scorer_seed2.pt").exists()
    assert not (directory / "models" / "shared_scorer_seed0.pt").exists()
    assert (metrics_path / "no_circuit_context" / "metrics_validation.json").exists()


def test_evaluate_reports_missing_model(fake_config: Path) -> None:
    """Evaluating a model that was never trained raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        main(["--config", str(fake_config), "evaluate", "--split", "validation"])


def test_evaluate_clifford_writes_metrics_and_both_figures(fake_config: Path) -> None:
    """evaluate-clifford writes its metrics, the medians and the probabilities by size."""
    directory = fake_config.parent
    config = load_config(fake_config)
    radius = config["actions"]["radius"]
    model_config = {**config["model"], "name": "shared_scorer"}
    for seed in config["evaluation"]["seeds"]:
        path = model_path(config, "shared_scorer", seed, None)
        save_model(build_model(model_config, None, radius), model_config, None, radius, path)

    assert main(["--config", str(fake_config), "evaluate-clifford"]) == 0

    metrics_path = directory / "results" / "metrics_clifford_validation.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    figures = directory / "results" / "figures"
    assert set(next(iter(metrics["qubits"].values()))["models"]) == {
        "shared_scorer",
        "shared_scorer_hammer",
    }
    assert (figures / "l1_relative_change_clifford_validation.png").exists()
    assert (figures / "support_probability_clifford_validation.png").exists()
