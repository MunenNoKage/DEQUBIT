"""Tests for dequbit.evaluate."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dequbit.actions import build_masks
from dequbit.dataset import QUBIT_FEATURE_NAMES, Circuits
from dequbit.evaluate import (
    INFERENCE_MODES,
    baseline_distributions,
    correct_circuits,
    method_metrics,
    plot_l1_relative_change,
    plot_medians_by_qubits,
    plot_training_histories,
    write_json,
)
from dequbit.models.per_qubit import PerQubitFlipModel

NUM_QUBITS = 2
RADIUS = 1
NUM_SHOTS = 500
SEED = 0
# A logit this large makes its action certain after the softmax.
CERTAIN_LOGIT = 100.0
BASELINES_CONFIG = {
    "threshold_max": 0.5,
    "threshold_steps": 6,
    "rate_max": 0.9,
    "rate_steps": 10,
    "scale_max": 2.0,
    "scale_steps": 5,
}


def random_counts(num_circuits: int) -> np.ndarray:
    """Draw random shot counts of every circuit.

    Args:
        num_circuits: Number of circuits.

    Returns:
        Counts of NUM_SHOTS shots, shape (num_circuits, 2**NUM_QUBITS).
    """
    rng = np.random.default_rng(SEED)
    uniform = np.full(2**NUM_QUBITS, 1 / 2**NUM_QUBITS)

    return rng.multinomial(NUM_SHOTS, uniform, size=num_circuits)


def no_qubit_features(counts: np.ndarray) -> np.ndarray:
    """Give qubit features of zeros for every circuit of the counts.

    Args:
        counts: Shot counts, shape (circuits, 2**NUM_QUBITS).

    Returns:
        Zeros, float32 of shape (circuits, NUM_QUBITS, len(QUBIT_FEATURE_NAMES)).
    """
    return np.zeros((len(counts), NUM_QUBITS, len(QUBIT_FEATURE_NAMES)), dtype=np.float32)


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


@pytest.mark.parametrize("mode", INFERENCE_MODES)
def test_correct_circuits_with_identity_changes_nothing(mode: str) -> None:
    """A model that always keeps the shot returns the noisy distribution and flips 0 bits."""
    counts = random_counts(3)

    corrected, flips = correct_circuits(
        certain_model(0),
        counts,
        no_qubit_features(counts),
        NUM_QUBITS,
        RADIUS,
        mode,
        np.random.default_rng(SEED),
    )

    np.testing.assert_allclose(corrected, counts / NUM_SHOTS, atol=1e-6)
    np.testing.assert_allclose(flips, 0.0, atol=1e-6)


@pytest.mark.parametrize("mode", INFERENCE_MODES)
def test_correct_circuits_moves_every_shot_with_certain_flip(mode: str) -> None:
    """A model that always flips bit 0 moves the mass of x to x ^ 1 and flips 1 bit per shot."""
    counts = random_counts(3)

    corrected, flips = correct_circuits(
        certain_model(1),
        counts,
        no_qubit_features(counts),
        NUM_QUBITS,
        RADIUS,
        mode,
        np.random.default_rng(SEED),
    )

    flipped_index = np.arange(2**NUM_QUBITS) ^ 1
    np.testing.assert_allclose(corrected, counts[:, flipped_index] / NUM_SHOTS, atol=1e-6)
    np.testing.assert_allclose(flips, 1.0, atol=1e-6)


def test_correct_circuits_sample_repeats_with_same_generator_seed() -> None:
    """Sampled corrections with equally seeded generators are identical and sum to 1."""
    model = PerQubitFlipModel(NUM_QUBITS, RADIUS)
    counts = random_counts(3)

    first, _ = correct_circuits(
        model,
        counts,
        no_qubit_features(counts),
        NUM_QUBITS,
        RADIUS,
        "sample",
        np.random.default_rng(SEED),
    )
    second, _ = correct_circuits(
        model,
        counts,
        no_qubit_features(counts),
        NUM_QUBITS,
        RADIUS,
        "sample",
        np.random.default_rng(SEED),
    )

    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first.sum(axis=1), 1.0, atol=1e-6)


def test_correct_circuits_sample_moves_whole_shots() -> None:
    """Sampled corrections are counts of whole shots, close to the expected correction."""
    model = PerQubitFlipModel(NUM_QUBITS, RADIUS)
    counts = random_counts(3)

    sampled, _ = correct_circuits(
        model,
        counts,
        no_qubit_features(counts),
        NUM_QUBITS,
        RADIUS,
        "sample",
        np.random.default_rng(SEED),
    )
    expected, _ = correct_circuits(
        model,
        counts,
        no_qubit_features(counts),
        NUM_QUBITS,
        RADIUS,
        "expected",
        np.random.default_rng(SEED),
    )

    np.testing.assert_allclose(sampled * NUM_SHOTS, np.round(sampled * NUM_SHOTS), atol=1e-3)
    assert np.abs(sampled - expected).max() < 0.1


def test_correct_circuits_handles_bitstrings_without_shots() -> None:
    """A circuit whose shots all fall on one bitstring is corrected without NaN."""
    counts = np.zeros((1, 2**NUM_QUBITS), dtype=np.int64)
    counts[0, 2] = NUM_SHOTS

    corrected, _ = correct_circuits(
        PerQubitFlipModel(NUM_QUBITS, RADIUS),
        counts,
        no_qubit_features(counts),
        NUM_QUBITS,
        RADIUS,
        "sample",
        np.random.default_rng(SEED),
    )

    np.testing.assert_allclose(corrected.sum(axis=1), 1.0, atol=1e-6)


def test_correct_circuits_rejects_unknown_mode() -> None:
    """A mode outside INFERENCE_MODES raises ValueError."""
    with pytest.raises(ValueError):
        counts = random_counts(1)
        correct_circuits(
            certain_model(0),
            counts,
            no_qubit_features(counts),
            NUM_QUBITS,
            RADIUS,
            "mean",
            np.random.default_rng(),
        )


def test_baseline_distributions_gives_every_baseline(fake_circuits: Circuits) -> None:
    """Every baseline returns distributions; identity is the noisy one and all sum to 1."""
    distributions, parameters = baseline_distributions(
        fake_circuits, fake_circuits, BASELINES_CONFIG, SEED
    )

    expected_names = {"identity", "oracle", "threshold", "repolarizer", "readout", "mix", "hammer"}
    assert expected_names | {"mix_physical", "mix_physical_fitted", "linear_map"} == set(
        distributions
    )
    assert set(parameters) == {"threshold", "repolarizer_rate", "mix_rate", "mix_physical_scale"}
    np.testing.assert_array_equal(distributions["identity"], fake_circuits.noisy)
    for corrected in distributions.values():
        assert corrected.shape == fake_circuits.targets.shape
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


def test_plot_training_histories_writes_image(tmp_path: Path) -> None:
    """Curves of two seeds of one model are written as a non-empty PNG file."""
    path = tmp_path / "figures" / "curves.png"
    history = [
        {"step": 0, "validation_loss": 0.3, "validation_l1rc_by_qubits": {"5": 0.7}},
        {"step": 200, "validation_loss": 0.1, "validation_l1rc_by_qubits": {"5": -0.6}},
    ]

    plot_training_histories({"mlp": [history, history]}, path, num_qubits=5)

    assert path.read_bytes().startswith(b"\x89PNG")


def test_plot_medians_by_qubits_writes_image(tmp_path: Path) -> None:
    """Lines of a model and of a baseline without the largest size are written as PNG."""
    path = tmp_path / "figures" / "scaling.png"
    medians = {"mlp": {3: -0.7, 5: -0.6, 8: -0.5}, "linear_map": {3: -0.6, 5: -0.5}}

    plot_medians_by_qubits(medians, path, models=["mlp"])

    assert path.read_bytes().startswith(b"\x89PNG")
