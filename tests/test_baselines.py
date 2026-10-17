"""Tests for dequbit.baselines."""

import numpy as np
import pytest

from dequbit.baselines import (
    best_parameter,
    depolarising_correction,
    mix_correction,
    oracle_distribution,
    physical_depolarising_rate,
    readout_correction,
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


def add_readout_noise(targets: np.ndarray, flip_probability: float) -> np.ndarray:
    """Flip the readout of every qubit independently with one probability.

    Args:
        targets: Ideal distributions, shape (circuits, 2**num_qubits).
        flip_probability: Probability that the readout of one qubit is wrong.

    Returns:
        Noisy distributions, same shape.
    """
    num_bitstrings = targets.shape[-1]
    bitstrings = np.arange(num_bitstrings)

    noisy = targets
    for qubit in range(num_bitstrings.bit_length() - 1):
        flipped = noisy[..., bitstrings ^ (1 << qubit)]
        noisy = (1 - flip_probability) * noisy + flip_probability * flipped

    return noisy


def test_add_readout_noise_flips_one_qubit_with_given_probability() -> None:
    targets = np.array([[1.0, 0.0, 0.0, 0.0]])

    noisy = add_readout_noise(targets, flip_probability=0.1)

    np.testing.assert_allclose(noisy, [[0.81, 0.09, 0.09, 0.01]])


def test_readout_correction_inverts_readout_noise() -> None:
    targets = np.random.default_rng(SEED).dirichlet(np.ones(8), size=3)

    corrected = readout_correction(add_readout_noise(targets, 0.05), flip_probability=0.05)

    np.testing.assert_allclose(corrected, targets)


def test_readout_correction_with_zero_probability_changes_nothing() -> None:
    distributions = np.array([[0.6, 0.05, 0.1, 0.25]])

    corrected = readout_correction(distributions, flip_probability=0.0)

    np.testing.assert_allclose(corrected, distributions)


def test_readout_correction_clips_negative_probabilities() -> None:
    distributions = np.array([[0.9, 0.1]])

    corrected = readout_correction(distributions, flip_probability=0.2)

    np.testing.assert_allclose(corrected, [[1.0, 0.0]])


@pytest.mark.parametrize("flip_probability", [-0.1, 0.5])
def test_readout_correction_rejects_probability_outside_range(flip_probability: float) -> None:
    distributions = np.array([[0.5, 0.5]])

    with pytest.raises(ValueError):
        readout_correction(distributions, flip_probability)


def test_readout_correction_rejects_length_that_is_not_power_of_two() -> None:
    distributions = np.array([[0.5, 0.25, 0.25]])

    with pytest.raises(ValueError):
        readout_correction(distributions, 0.1)


def test_mix_correction_inverts_depolarising_then_readout_noise() -> None:
    targets = np.random.default_rng(SEED).dirichlet(np.ones(8), size=3)
    noisy = add_readout_noise(depolarise(targets, 0.2), 0.05)

    corrected = mix_correction(noisy, rate=0.2, flip_probability=0.05)

    np.testing.assert_allclose(corrected, targets)


def test_mix_correction_without_readout_errors_is_depolarising_correction() -> None:
    distributions = np.array([[0.6, 0.05, 0.1, 0.25]])

    corrected = mix_correction(distributions, rate=0.1, flip_probability=0.0)

    np.testing.assert_allclose(corrected, depolarising_correction(distributions, 0.1))


def test_mix_correction_without_depolarising_is_readout_correction() -> None:
    distributions = np.array([[0.6, 0.05, 0.1, 0.25]])

    corrected = mix_correction(distributions, rate=0.0, flip_probability=0.05)

    np.testing.assert_allclose(corrected, readout_correction(distributions, 0.05))


@pytest.mark.parametrize(
    ("num_two_qubit_gates", "expected"), [(0, 0.0), (1, 0.02), (12, 1 - 0.98**12)]
)
def test_physical_depolarising_rate_counts_shots_hit_by_any_gate(
    num_two_qubit_gates: int, expected: float
) -> None:
    rate = physical_depolarising_rate(0.02, num_two_qubit_gates)

    assert rate == pytest.approx(expected)


@pytest.mark.parametrize(
    ("two_qubit_error", "num_two_qubit_gates"), [(-0.1, 12), (1.0, 12), (0.02, -1)]
)
def test_physical_depolarising_rate_rejects_invalid_arguments(
    two_qubit_error: float, num_two_qubit_gates: int
) -> None:
    with pytest.raises(ValueError):
        physical_depolarising_rate(two_qubit_error, num_two_qubit_gates)
