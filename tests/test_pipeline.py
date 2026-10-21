"""Tests for dequbit.pipeline on tiny fake data files."""

from pathlib import Path

import numpy as np
import pytest

from dequbit.config import load_config
from dequbit.pipeline import (
    evaluate_baselines,
    evaluate_split,
    load_split,
    model_path,
    train_model,
)
from tests.conftest import FAKE_NUM_QUBITS, FAKE_SHOTS, FAKE_TEST_CIRCUITS, FAKE_TRAIN_CIRCUITS


def test_load_split_divides_training_file_and_reads_test_file(fake_config: Path) -> None:
    """Training and validation circuits add up to the training file; test is its own file."""
    config = load_config(fake_config)

    train, validation, test = (
        load_split(config, split) for split in ("train", "validation", "test")
    )

    assert len(train.shots) + len(validation.shots) == FAKE_TRAIN_CIRCUITS
    assert len(validation.shots) == round(
        FAKE_TRAIN_CIRCUITS * config["data"]["validation_fraction"]
    )
    assert test.shots.shape == (FAKE_TEST_CIRCUITS, FAKE_SHOTS)
    assert test.targets.shape == (FAKE_TEST_CIRCUITS, 2**FAKE_NUM_QUBITS)
    np.testing.assert_array_equal(test.gate_counts, 1)
    assert test.noise["p_ro"] == pytest.approx(0.05)


def test_circuits_noisy_is_distribution_of_shots(fake_config: Path) -> None:
    """The noisy distribution of every circuit sums to 1 over its bitstrings."""
    circuits = load_split(load_config(fake_config), "validation")

    np.testing.assert_allclose(circuits.noisy.sum(axis=1), 1.0)


def test_load_split_rejects_unknown_split(fake_config: Path) -> None:
    """A split outside SPLITS raises ValueError."""
    with pytest.raises(ValueError):
        load_split(load_config(fake_config), "holdout")


def test_model_path_names_model_and_seed() -> None:
    """Every model and seed gets its own file in paths.model."""
    config = {"paths": {"model": "models"}}

    assert model_path(config, "mlp", 2) == Path("models") / "mlp_seed2.pt"


def test_train_model_returns_history_from_step_zero(fake_config: Path) -> None:
    """train_model returns the model and a history that starts with the untrained model."""
    model, history = train_model(load_config(fake_config), "mlp", seed=0)

    assert history[0]["step"] == 0
    assert history[-1]["step"] == 20
    assert sum(parameter.numel() for parameter in model.parameters()) > 0


def test_evaluate_baselines_corrects_every_circuit(fake_config: Path) -> None:
    """Every baseline returns one distribution per evaluated circuit."""
    config = load_config(fake_config)

    baselines, _ = evaluate_baselines(
        config, load_split(config, "train"), load_split(config, "test")
    )

    for corrected in baselines.values():
        assert corrected.shape == (FAKE_TEST_CIRCUITS, 2**FAKE_NUM_QUBITS)


def test_evaluate_split_without_models_has_baseline_metrics(fake_config: Path) -> None:
    """With no models in the config, evaluate_split still returns every baseline."""
    config = load_config(fake_config)
    config["evaluation"]["models"] = []

    metrics, changes = evaluate_split(config, "validation")

    assert metrics["models"] == {}
    assert set(changes) == set(metrics["baselines"])
    assert metrics["baselines"]["identity"]["median"] == 0.0
