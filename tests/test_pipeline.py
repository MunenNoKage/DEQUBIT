"""Tests for dequbit.pipeline on tiny fake data files."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from dequbit.config import load_config
from dequbit.models import build_model, save_model
from dequbit.pipeline import (
    THEN_HAMMER_SUFFIX,
    _training_circuits,
    ablation_config,
    evaluate_baselines,
    evaluate_clifford_split,
    evaluate_split,
    evaluated_models,
    load_split,
    model_path,
    model_sizes,
    train_model,
)
from tests.conftest import (
    FAKE_CIRCUIT_TYPES,
    FAKE_CLIFFORD_QUBITS,
    FAKE_NUM_QUBITS,
    FAKE_QUBIT_COUNTS,
    FAKE_SHOTS,
    FAKE_SPLIT_CIRCUITS,
    FAKE_UNTRAINED_QUBITS,
    write_fake_clifford_file,
)


def test_load_split_reads_the_file_of_every_split(fake_config: Path) -> None:
    """Training, validation and test circuits come from their own files."""
    config = load_config(fake_config)

    train, validation, test = (
        load_split(config, split, FAKE_NUM_QUBITS) for split in ("train", "validation", "test")
    )

    assert len(train.counts) == FAKE_SPLIT_CIRCUITS["train"]
    assert len(validation.counts) == FAKE_SPLIT_CIRCUITS["val"]
    assert test.counts.shape == (FAKE_SPLIT_CIRCUITS["test"], 2**FAKE_NUM_QUBITS)
    assert test.num_shots == FAKE_SHOTS


def test_circuits_noisy_is_distribution_of_shots(fake_config: Path) -> None:
    """The noisy distribution of every circuit sums to 1 over its bitstrings."""
    circuits = load_split(load_config(fake_config), "validation", FAKE_NUM_QUBITS)

    np.testing.assert_allclose(circuits.noisy.sum(axis=1), 1.0)


def test_load_split_rejects_unknown_split(fake_config: Path) -> None:
    """A split outside SPLITS raises ValueError."""
    with pytest.raises(ValueError):
        load_split(load_config(fake_config), "holdout", FAKE_NUM_QUBITS)


def test_model_path_names_model_size_and_seed() -> None:
    """Every model, seed and trained number of qubits gets its own file in paths.model."""
    config = {"paths": {"model": "models"}}

    assert model_path(config, "mlp", 2, 5) == Path("models") / "mlp_N5_seed2.pt"
    assert model_path(config, "shared_scorer", 2, None) == Path("models") / "shared_scorer_seed2.pt"


def test_model_sizes_are_one_per_number_of_qubits_except_for_the_scorer(
    fake_config: Path,
) -> None:
    """The MLP is trained per number of qubits, the shared scorer once for all."""
    config = load_config(fake_config)

    assert model_sizes(config, "mlp") == list(FAKE_QUBIT_COUNTS)
    assert model_sizes(config, "shared_scorer") == [None]


def test_train_model_returns_history_from_step_zero(fake_config: Path) -> None:
    """train_model returns the model and a history that starts with the untrained model."""
    model, history = train_model(load_config(fake_config), "mlp", seed=0, num_qubits=2)

    assert history[0]["step"] == 0
    assert history[-1]["step"] == 20
    assert set(history[0]["validation_l1rc_by_qubits"]) == {"2"}
    assert sum(parameter.numel() for parameter in model.parameters()) > 0


def test_train_model_without_size_validates_every_number_of_qubits(fake_config: Path) -> None:
    """The shared scorer trained on all sizes reports the validation of each of them."""
    _, history = train_model(load_config(fake_config), "shared_scorer", seed=0, num_qubits=None)

    assert set(history[-1]["validation_l1rc_by_qubits"]) == {"2", "3"}


def test_evaluate_baselines_corrects_every_circuit(fake_config: Path) -> None:
    """Every baseline returns one distribution per evaluated circuit."""
    config = load_config(fake_config)

    baselines, _ = evaluate_baselines(
        config,
        load_split(config, "train", FAKE_NUM_QUBITS),
        load_split(config, "test", FAKE_NUM_QUBITS),
    )

    for corrected in baselines.values():
        assert corrected.shape == (FAKE_SPLIT_CIRCUITS["test"], 2**FAKE_NUM_QUBITS)


def test_evaluate_split_without_models_has_baseline_metrics(fake_config: Path) -> None:
    """With no models in the config, evaluate_split still returns every baseline per size."""
    config = load_config(fake_config)
    config["evaluation"]["models"] = []

    metrics, changes = evaluate_split(config, "validation")

    assert set(metrics["qubits"]) == {"2", "3", "4"}
    assert set(changes) == {*FAKE_QUBIT_COUNTS, FAKE_UNTRAINED_QUBITS}
    for num_qubits in FAKE_QUBIT_COUNTS:
        size_metrics = metrics["qubits"][str(num_qubits)]
        assert size_metrics["models"] == {}
        assert set(changes[num_qubits]) == set(size_metrics["baselines"])
        assert size_metrics["baselines"]["identity"]["median"] == 0.0


def test_evaluate_split_summarizes_every_circuit_type(fake_config: Path) -> None:
    """Every circuit family of the files has the metrics of every method."""
    config = load_config(fake_config)
    config["evaluation"]["models"] = []

    metrics, _ = evaluate_split(config, "validation")

    by_type = metrics["qubits"]["2"]["circuit_types"]
    assert set(by_type) == set(FAKE_CIRCUIT_TYPES)
    assert set(by_type["bv"]) == set(metrics["qubits"]["2"]["baselines"])
    assert by_type["bv"]["identity"]["median"] == 0.0


def test_evaluated_models_leave_out_per_size_models_on_untrained_sizes(
    fake_config: Path,
) -> None:
    """Where no MLP was trained only the shared scorer is evaluated."""
    config = load_config(fake_config)
    config["evaluation"]["models"] = ["mlp", "shared_scorer"]

    assert evaluated_models(config, FAKE_NUM_QUBITS) == ["mlp", "shared_scorer"]
    assert evaluated_models(config, FAKE_UNTRAINED_QUBITS) == ["shared_scorer"]


def test_ablation_config_redirects_outputs_and_applies_overrides(fake_config: Path) -> None:
    """A training variant gets its own models and seeds; an evaluation variant only outputs."""
    config = load_config(fake_config)

    trained = ablation_config(config, "no_circuit_context")
    evaluated = ablation_config(config, "calibration_noise_20")

    assert Path(trained["paths"]["model"]).parts[-2:] == ("ablations", "no_circuit_context")
    assert trained["evaluation"]["seeds"] == config["ablation"]["seeds"]
    assert trained["model"]["shared_scorer"]["use_circuit_context"] is False
    assert evaluated["paths"]["model"] == config["paths"]["model"]
    assert evaluated["evaluation"]["calibration_noise"] == 0.2
    assert "calibration_noise_20" in evaluated["paths"]["metrics"]
    assert config["model"]["shared_scorer"]["use_circuit_context"] is True


def test_ablation_config_rejects_unknown_variant(fake_config: Path) -> None:
    """A name outside ablation.variants raises ValueError."""
    with pytest.raises(ValueError):
        ablation_config(load_config(fake_config), "no_such_variant")


def test_train_model_leaves_out_excluded_types(fake_config: Path) -> None:
    """Without the bv circuits training still runs and validates on the rest."""
    config = load_config(fake_config)
    config["data"]["excluded_types"] = ["bv"]

    _, history = train_model(config, "shared_scorer", seed=0, num_qubits=None)

    assert history[-1]["step"] == 20


def test_training_circuits_include_clifford_circuits_of_listed_sizes(fake_config: Path) -> None:
    """With data.clifford_qubit_counts the Clifford circuits join the v4 circuits."""
    config = load_config(fake_config)
    num_qubits = FAKE_QUBIT_COUNTS[0]
    config["data"]["clifford_qubit_counts"] = [num_qubits]
    for split, file_split in (("train", "train"), ("validation", "val")):
        path = Path(
            config["paths"]["clifford_data"].format(split=file_split, num_qubits=num_qubits)
        )
        write_fake_clifford_file(path, 6, seed=0, num_qubits=num_qubits)
        source = config["paths"]["data"].format(split=file_split, num_qubits=num_qubits)
        with h5py.File(source, "r") as v4_file, h5py.File(path, "a") as clifford_file:
            del clifford_file["noise_model"]
            v4_file.copy("noise_model", clifford_file)

    circuits = _training_circuits(config, "train", num_qubits)
    _, history = train_model(config, "shared_scorer", seed=0, num_qubits=None)

    assert len(circuits.counts) == FAKE_SPLIT_CIRCUITS["train"] + 6
    assert history[-1]["step"] == 20


def test_evaluate_split_with_drifted_calibration_changes_calibrated_baselines(
    fake_config: Path,
) -> None:
    """Drifted calibration changes readout inversion but not the calibration-free threshold."""
    config = load_config(fake_config)
    config["evaluation"]["models"] = []
    drifted = load_config(fake_config)
    drifted["evaluation"]["models"] = []
    drifted["evaluation"]["calibration_noise"] = 0.5

    true_metrics, _ = evaluate_split(config, "validation")
    drifted_metrics, _ = evaluate_split(drifted, "validation")

    true_size, drifted_size = true_metrics["qubits"]["2"], drifted_metrics["qubits"]["2"]
    assert true_size["baselines"]["readout"] != drifted_size["baselines"]["readout"]
    assert true_size["baselines"]["threshold"] == drifted_size["baselines"]["threshold"]


def test_evaluate_clifford_split_compares_scorer_and_hammer(fake_config: Path) -> None:
    """The Clifford evaluation scores HAMMER, the scorer and the scorer followed by HAMMER."""
    config = load_config(fake_config)
    radius = config["actions"]["radius"]
    model_config = {**config["model"], "name": "shared_scorer"}
    for seed in config["evaluation"]["seeds"]:
        model = build_model(model_config, None, radius)
        save_model(
            model, model_config, None, radius, model_path(config, "shared_scorer", seed, None)
        )

    metrics, changes = evaluate_clifford_split(config, "validation")

    size_metrics = metrics["qubits"][str(FAKE_CLIFFORD_QUBITS)]
    combined = f"shared_scorer{THEN_HAMMER_SUFFIX}"
    assert set(size_metrics["baselines"]) == {"identity", "oracle", "hammer"}
    assert set(size_metrics["models"]) == {"shared_scorer", combined}
    assert size_metrics["baselines"]["identity"]["median"] == 0.0
    assert size_metrics["baselines"]["oracle"]["median"] == pytest.approx(-1.0)
    assert size_metrics["comparisons"][combined]["baseline"] == "hammer"
    assert len(changes[FAKE_CLIFFORD_QUBITS]["shared_scorer"]) == 2 * size_metrics["num_circuits"]
