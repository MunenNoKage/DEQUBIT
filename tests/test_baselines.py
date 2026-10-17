"""Tests for dequbit.baselines."""

import numpy as np
import pytest

from dequbit.baselines import (
    best_parameter,
    depolarising_correction,
    oracle_distribution,
    threshold_correction,
)

SEED = 0


def test_oracle_distribution_is_close_to_target_and_reproducible() -> None:
    targets = np.array([[0.5, 0.0, 0.0, 0.5], [0.1, 0.2, 0.3, 0.4]], dtype=np.float32)

    first = oracle_distribution(targets, 100_000, np.random.default_rng(SEED))
    second = oracle_distribution(targets, 100_000, np.random.default_rng(SEED))

    np.testing.assert_allclose(first.sum(axis=-1), 1.0)
    np.testing.assert_allclose(first, targets, atol=0.01)
    np.testing.assert_array_equal(first, second)


def test_threshold_correction_removes_small_probabilities_and_sums_to_one() -> None:
    distributions = np.array([[0.6, 0.05, 0.1, 0.25]])

    corrected = threshold_correction(distributions, threshold=0.1)

    np.testing.assert_allclose(corrected, [[0.6 / 0.95, 0.0, 0.1 / 0.95, 0.25 / 0.95]])


def test_threshold_correction_with_zero_threshold_changes_nothing() -> None:
    distributions = np.array([[0.6, 0.05, 0.1, 0.25]])

    corrected = threshold_correction(distributions, threshold=0.0)

    np.testing.assert_allclose(corrected, distributions)


def test_threshold_correction_keeps_largest_probability() -> None:
    distributions = np.array([[0.3, 0.2, 0.2, 0.3], [0.1, 0.6, 0.2, 0.1]])

    corrected = threshold_correction(distributions, threshold=0.9)

    np.testing.assert_allclose(corrected, [[0.5, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.0]])


def test_best_parameter_picks_threshold_that_removes_noise() -> None:
    targets = np.array([[1.0, 0.0, 0.0, 0.0]])
    noisy = np.array([[0.7, 0.1, 0.1, 0.1]])

    best = best_parameter(threshold_correction, [0.0, 0.05, 0.2], noisy, targets)

    assert best == 0.2


def test_best_parameter_rejects_empty_grid() -> None:
    targets = np.array([[1.0, 0.0]])

    with pytest.raises(ValueError):
        best_parameter(threshold_correction, [], targets, targets)


def depolarise(targets: np.ndarray, rate: float) -> np.ndarray:
    """Apply global depolarising noise to ideal distributions.

    Args:
        targets: Ideal distributions, shape (circuits, 2**num_qubits).
        rate: Share of shots replaced by uniform noise.

    Returns:
        Noisy distributions, same shape.
    """
    return (1 - rate) * targets + rate / targets.shape[-1]


def test_depolarising_correction_inverts_depolarising_noise() -> None:
    targets = np.array([[0.4, 0.3, 0.2, 0.1]])

    corrected = depolarising_correction(depolarise(targets, 0.2), rate=0.2)

    np.testing.assert_allclose(corrected, targets)


def test_depolarising_correction_with_zero_rate_changes_nothing() -> None:
    distributions = np.array([[0.6, 0.05, 0.1, 0.25]])

    corrected = depolarising_correction(distributions, rate=0.0)

    np.testing.assert_allclose(corrected, distributions)


def test_depolarising_correction_clips_negative_probabilities() -> None:
    distributions = np.array([[0.7, 0.1, 0.1, 0.1]])

    corrected = depolarising_correction(distributions, rate=0.5)

    np.testing.assert_allclose(corrected, [[1.0, 0.0, 0.0, 0.0]])


@pytest.mark.parametrize("rate", [-0.1, 1.0])
def test_depolarising_correction_rejects_rate_outside_unit_interval(rate: float) -> None:
    distributions = np.array([[0.5, 0.5]])

    with pytest.raises(ValueError):
        depolarising_correction(distributions, rate)


def test_best_parameter_recovers_known_depolarising_rate() -> None:
    targets = np.random.default_rng(SEED).dirichlet(np.ones(8), size=20)
    noisy = depolarise(targets, 0.3)

    best = best_parameter(depolarising_correction, [0.0, 0.1, 0.2, 0.3, 0.4], noisy, targets)

    assert best == 0.3
