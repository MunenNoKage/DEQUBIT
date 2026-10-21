"""Tests for dequbit.evaluate."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dequbit.actions import build_masks
from dequbit.dataset import shot_counts
from dequbit.evaluate import (
    INFERENCE_MODES,
    baseline_distributions,
    correct_circuits,
    method_metrics,
    plot_l1_relative_change,
    write_json,
)
from dequbit.models.per_qubit import PerQubitFlipModel

NUM_QUBITS = 2
RADIUS = 1
NUM_SHOTS = 500
SEED = 0
# A logit this large makes its action certain after the softmax.
CERTAIN_LOGIT = 100.0
BASELINES_CONFIG = {"threshold_max": 0.5, "threshold_steps": 6, "rate_max": 0.9, "rate_steps": 10}


def random_shots(num_circuits: int) -> np.ndarray:
    """Draw random shots of every circuit.

    Args:
        num_circuits: Number of circuits.

    Returns:
        Shots as integers, shape (num_circuits, NUM_SHOTS).
    """
    return np.random.default_rng(SEED).integers(0, 2**NUM_QUBITS, (num_circuits, NUM_SHOTS))


def certain_model(action_mask: int) -> PerQubitFlipModel:
    """Build a model that applies one action to every shot.

    Args:
        action_mask: Mask of the action the model always chooses.

    Returns:
        B4 model whose logit of that action is CERTAIN_LOGIT and of the others 0.
    """
    model = PerQubitFlipModel(NUM_QUBITS, RADIUS)
    action = build_masks(NUM_QUBITS, RADIUS).tolist().index(action_mask)
    with torch.no_grad():
        model.logits.zero_()
        model.logits[action] = CERTAIN_LOGIT

    return model


def noisy_distributions(shots: np.ndarray) -> np.ndarray:
    """Turn shots into the noisy distribution of every circuit.

    Args:
        shots: Shots as integers, shape (circuits, NUM_SHOTS).

    Returns:
        Distributions, shape (circuits, 2**NUM_QUBITS).
    """
    return shot_counts(shots, NUM_QUBITS) / NUM_SHOTS


@pytest.mark.parametrize("mode", INFERENCE_MODES)
def test_correct_circuits_with_identity_changes_nothing(mode: str) -> None:
    """A model that always keeps the shot returns the noisy distribution and flips 0 bits."""
    shots = random_shots(3)

    corrected, flips = correct_circuits(
        certain_model(0), shots, NUM_QUBITS, RADIUS, mode, torch.Generator().manual_seed(SEED)
    )

    np.testing.assert_allclose(corrected, noisy_distributions(shots), atol=1e-6)
    np.testing.assert_allclose(flips, 0.0, atol=1e-6)


@pytest.mark.parametrize("mode", INFERENCE_MODES)
def test_correct_circuits_moves_every_shot_with_certain_flip(mode: str) -> None:
    """A model that always flips bit 0 moves the mass of x to x ^ 1 and flips 1 bit per shot."""
    shots = random_shots(3)
    noisy = noisy_distributions(shots)

    corrected, flips = correct_circuits(
        certain_model(1), shots, NUM_QUBITS, RADIUS, mode, torch.Generator().manual_seed(SEED)
    )

    flipped_index = np.arange(2**NUM_QUBITS) ^ 1
    np.testing.assert_allclose(corrected, noisy[:, flipped_index], atol=1e-6)
    np.testing.assert_allclose(flips, 1.0, atol=1e-6)


def test_correct_circuits_sample_repeats_with_same_generator_seed() -> None:
    """Sampled corrections with equally seeded generators are identical and sum to 1."""
    model = PerQubitFlipModel(NUM_QUBITS, RADIUS)
    shots = random_shots(3)

    first, _ = correct_circuits(
        model, shots, NUM_QUBITS, RADIUS, "sample", torch.Generator().manual_seed(SEED)
    )
    second, _ = correct_circuits(
        model, shots, NUM_QUBITS, RADIUS, "sample", torch.Generator().manual_seed(SEED)
    )

    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first.sum(axis=1), 1.0, atol=1e-6)


def test_correct_circuits_rejects_unknown_mode() -> None:
    """A mode outside INFERENCE_MODES raises ValueError."""
    with pytest.raises(ValueError):
        correct_circuits(
            certain_model(0), random_shots(1), NUM_QUBITS, RADIUS, "mean", torch.Generator()
        )


def test_baseline_distributions_gives_every_baseline() -> None:
    """Every baseline returns distributions; identity is the noisy one, oracle sums to 1."""
    rng = np.random.default_rng(SEED)
    targets = rng.dirichlet(np.ones(2**NUM_QUBITS), size=12)
    noisy = 0.8 * targets + 0.2 / 2**NUM_QUBITS

    distributions, parameters = baseline_distributions(
        noisy[:8],
        targets[:8],
        noisy[8:],
        targets[8:],
        NUM_SHOTS,
        {"p_ro": 0.01, "p2": 0.02},
        np.array([1, 1, 2, 2]),
        BASELINES_CONFIG,
        SEED,
    )

    expected_names = {"identity", "oracle", "threshold", "repolarizer", "readout", "mix"}
    assert expected_names | {"mix_physical", "linear_map"} == set(distributions)
    assert set(parameters) == {"threshold", "repolarizer_rate", "mix_rate"}
    np.testing.assert_array_equal(distributions["identity"], noisy[8:])
    for corrected in distributions.values():
        np.testing.assert_allclose(corrected.sum(axis=1), 1.0, atol=1e-6)


def test_method_metrics_of_perfect_correction() -> None:
    """Correcting to the target gives L1 relative change -1, all improved, zero distances."""
    targets = np.array([[0.5, 0.5, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]])
    noisy = np.array([[0.4, 0.4, 0.1, 0.1], [0.4, 0.2, 0.2, 0.2]])

    metrics = method_metrics(noisy, targets, targets)

    assert metrics["median"] == pytest.approx(-1.0)
    assert metrics["improved_fraction"] == 1.0
    assert metrics["mean_l1"] == pytest.approx(0.0)
    assert metrics["mean_kl"] == pytest.approx(0.0, abs=1e-9)


def test_write_json_creates_folders_and_round_trips(tmp_path: Path) -> None:
    """Written JSON reads back as the same dictionary, in a folder created on the way."""
    path = tmp_path / "results" / "metrics.json"
    content = {"mlp": {"median": -0.7, "seeds": [0, 1]}}

    write_json(content, path)

    assert json.loads(path.read_text(encoding="utf-8")) == content


def test_plot_l1_relative_change_writes_image(tmp_path: Path) -> None:
    """The figure with a highlighted model is written as a non-empty PNG file."""
    path = tmp_path / "figures" / "l1.png"
    changes = {"mlp": np.array([-0.7, -0.6]), "identity": np.zeros(2)}

    plot_l1_relative_change(changes, path, models=["mlp"])

    assert path.read_bytes().startswith(b"\x89PNG")
