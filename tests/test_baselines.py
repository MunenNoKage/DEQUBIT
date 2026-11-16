"""Tests for dequbit.baselines."""

import numpy as np
import pytest

from dequbit.baselines import (
    MAX_PHYSICAL_RATE,
    best_parameter,
    depolarising_correction,
    fit_linear_map,
    linear_map_correction,
    mix_correction,
    oracle_distribution,
    physical_depolarising_rate,
    readout_correction,
    scaled_mix_correction,
    scaled_physical_rate,
    threshold_correction,
)

SEED = 0


def test_oracle_distribution_is_close_to_target_and_reproducible() -> None:
    """Oracle samples of 100000 shots sum to 1, are within 0.01 of the target and repeat with the
    same seed."""
    targets = np.array([[0.5, 0.0, 0.0, 0.5], [0.1, 0.2, 0.3, 0.4]], dtype=np.float32)

    first = oracle_distribution(targets, 100_000, np.random.default_rng(SEED))
    second = oracle_distribution(targets, 100_000, np.random.default_rng(SEED))

    np.testing.assert_allclose(first.sum(axis=-1), 1.0)
    np.testing.assert_allclose(first, targets, atol=0.01)
    np.testing.assert_array_equal(first, second)


def test_threshold_correction_removes_small_probabilities_and_sums_to_one() -> None:
    """Threshold 0.1 removes the entry 0.05 and rescales the rest to sum 1."""
    distributions = np.array([[0.6, 0.05, 0.1, 0.25]])

    corrected = threshold_correction(distributions, threshold=0.1)

    np.testing.assert_allclose(corrected, [[0.6 / 0.95, 0.0, 0.1 / 0.95, 0.25 / 0.95]])


def test_threshold_correction_with_zero_threshold_changes_nothing() -> None:
    """Threshold 0 keeps every probability."""
    distributions = np.array([[0.6, 0.05, 0.1, 0.25]])

    corrected = threshold_correction(distributions, threshold=0.0)

    np.testing.assert_allclose(corrected, distributions)


def test_threshold_correction_keeps_largest_probability() -> None:
    """A threshold above every probability still keeps the largest ones."""
    distributions = np.array([[0.3, 0.2, 0.2, 0.3], [0.1, 0.6, 0.2, 0.1]])

    corrected = threshold_correction(distributions, threshold=0.9)

    np.testing.assert_allclose(corrected, [[0.5, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.0]])


def test_best_parameter_picks_threshold_that_removes_noise() -> None:
    """Of thresholds 0, 0.05 and 0.2 only 0.2 removes the three noise entries 0.1."""
    targets = np.array([[1.0, 0.0, 0.0, 0.0]])
    noisy = np.array([[0.7, 0.1, 0.1, 0.1]])

    best = best_parameter(threshold_correction, [0.0, 0.05, 0.2], noisy, targets)

    assert best == 0.2


def test_best_parameter_rejects_empty_grid() -> None:
    """An empty grid raises ValueError."""
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
    """Correcting with the rate that made the noise returns the ideal distribution."""
    targets = np.array([[0.4, 0.3, 0.2, 0.1]])

    corrected = depolarising_correction(depolarise(targets, 0.2), rate=0.2)

    np.testing.assert_allclose(corrected, targets)


def test_depolarising_correction_with_zero_rate_changes_nothing() -> None:
    """Rate 0 keeps every probability."""
    distributions = np.array([[0.6, 0.05, 0.1, 0.25]])

    corrected = depolarising_correction(distributions, rate=0.0)

    np.testing.assert_allclose(corrected, distributions)


def test_depolarising_correction_clips_negative_probabilities() -> None:
    """Rate 0.5 makes the entries 0.1 negative; they become 0 and the result is [1, 0, 0, 0]."""
    distributions = np.array([[0.7, 0.1, 0.1, 0.1]])

    corrected = depolarising_correction(distributions, rate=0.5)

    np.testing.assert_allclose(corrected, [[1.0, 0.0, 0.0, 0.0]])


@pytest.mark.parametrize("rate", [-0.1, 1.0])
def test_depolarising_correction_rejects_rate_outside_unit_interval(rate: float) -> None:
    """A rate below 0 or of at least 1 raises ValueError.

    Args:
        rate: Invalid rate.
    """
    distributions = np.array([[0.5, 0.5]])

    with pytest.raises(ValueError):
        depolarising_correction(distributions, rate)


def test_best_parameter_recovers_known_depolarising_rate() -> None:
    """On 20 random distributions with noise of rate 0.3, the grid search finds 0.3."""
    targets = np.random.default_rng(SEED).dirichlet(np.ones(8), size=20)
    noisy = depolarise(targets, 0.3)

    best = best_parameter(depolarising_correction, [0.0, 0.1, 0.2, 0.3, 0.4], noisy, targets)

    assert best == 0.3


def add_readout_noise(targets: np.ndarray, p01: np.ndarray, p10: np.ndarray) -> np.ndarray:
    """Misread every bit independently with its own two error rates.

    Args:
        targets: Ideal distributions, shape (circuits, 2**num_qubits).
        p01: Probability of reading 1 for state 0 of every bit, shape (num_qubits,) or
            (circuits, num_qubits).
        p10: Probability of reading 0 for state 1 of every bit, same shape as p01.

    Returns:
        Noisy distributions, same shape as targets.
    """
    num_bitstrings = targets.shape[-1]
    bitstrings = np.arange(num_bitstrings)

    noisy = targets
    for bit in range(num_bitstrings.bit_length() - 1):
        bit_is_one = (bitstrings >> bit) & 1 == 1
        leave = np.where(bit_is_one, p10[..., bit, np.newaxis], p01[..., bit, np.newaxis])
        arrive = np.where(bit_is_one, p01[..., bit, np.newaxis], p10[..., bit, np.newaxis])
        noisy = (1 - leave) * noisy + arrive * noisy[..., bitstrings ^ (1 << bit)]

    return noisy


def test_add_readout_noise_misreads_each_state_with_its_own_rate() -> None:
    """Helper check on one qubit: state 0 is read as 1 with p01, state 1 as 0 with p10."""
    targets = np.array([[1.0, 0.0], [0.0, 1.0]])

    noisy = add_readout_noise(targets, p01=np.array([0.1]), p10=np.array([0.3]))

    np.testing.assert_allclose(noisy, [[0.9, 0.1], [0.3, 0.7]])


def test_readout_correction_inverts_asymmetric_readout_noise() -> None:
    """Correcting with the rates that made the noise returns the ideal 3-qubit distributions."""
    targets = np.random.default_rng(SEED).dirichlet(np.ones(8), size=3)
    p01, p10 = np.array([0.01, 0.03, 0.05]), np.array([0.02, 0.08, 0.04])

    corrected = readout_correction(add_readout_noise(targets, p01, p10), p01, p10)

    np.testing.assert_allclose(corrected, targets)


def test_readout_correction_takes_rates_of_every_circuit() -> None:
    """Rates of shape (circuits, num_qubits) correct every circuit with its own row."""
    rng = np.random.default_rng(SEED)
    targets = rng.dirichlet(np.ones(8), size=4)
    p01, p10 = rng.uniform(0, 0.1, size=(4, 3)), rng.uniform(0, 0.1, size=(4, 3))

    corrected = readout_correction(add_readout_noise(targets, p01, p10), p01, p10)

    np.testing.assert_allclose(corrected, targets)


def test_readout_correction_with_zero_rates_changes_nothing() -> None:
    """Rates 0 keep every probability."""
    distributions = np.array([[0.6, 0.05, 0.1, 0.25]])

    corrected = readout_correction(distributions, np.zeros(2), np.zeros(2))

    np.testing.assert_allclose(corrected, distributions)


def test_readout_correction_clips_negative_probabilities() -> None:
    """Rates 0.2 make the entry of [0.9, 0.1] at bitstring 1 negative; it becomes 0 and the
    result is [1, 0]."""
    distributions = np.array([[0.9, 0.1]])

    corrected = readout_correction(distributions, np.array([0.2]), np.array([0.2]))

    np.testing.assert_allclose(corrected, [[1.0, 0.0]])


@pytest.mark.parametrize(("p01", "p10"), [(-0.1, 0.1), (0.5, 0.5)])
def test_readout_correction_rejects_rates_outside_range(p01: float, p10: float) -> None:
    """A negative rate or rates that add up to at least 1 raise ValueError.

    Args:
        p01: Probability of reading 1 for state 0.
        p10: Probability of reading 0 for state 1.
    """
    distributions = np.array([[0.5, 0.5]])

    with pytest.raises(ValueError):
        readout_correction(distributions, np.array([p01]), np.array([p10]))


def test_readout_correction_rejects_rates_of_other_number_of_qubits() -> None:
    """Rates of 3 bits for a distribution over 4 bitstrings raise ValueError."""
    distributions = np.array([[0.25, 0.25, 0.25, 0.25]])

    with pytest.raises(ValueError):
        readout_correction(distributions, np.zeros(3), np.zeros(3))


def test_mix_correction_inverts_depolarising_then_readout_noise() -> None:
    """Depolarising noise followed by readout noise is undone by MIX with the same parameters."""
    targets = np.random.default_rng(SEED).dirichlet(np.ones(8), size=3)
    p01, p10 = np.array([0.01, 0.03, 0.05]), np.array([0.02, 0.08, 0.04])
    noisy = add_readout_noise(depolarise(targets, 0.2), p01, p10)

    corrected = mix_correction(noisy, 0.2, p01, p10)

    np.testing.assert_allclose(corrected, targets)


def test_mix_correction_without_readout_errors_is_depolarising_correction() -> None:
    """With readout rates 0, MIX is the repolarizer alone."""
    distributions = np.array([[0.6, 0.05, 0.1, 0.25]])

    corrected = mix_correction(distributions, 0.1, np.zeros(2), np.zeros(2))

    np.testing.assert_allclose(corrected, depolarising_correction(distributions, 0.1))


def test_mix_correction_without_depolarising_is_readout_correction() -> None:
    """With rate 0, MIX is the readout correction alone."""
    distributions = np.array([[0.6, 0.05, 0.1, 0.25]])
    p01, p10 = np.array([0.05, 0.02]), np.array([0.03, 0.04])

    corrected = mix_correction(distributions, 0.0, p01, p10)

    np.testing.assert_allclose(corrected, readout_correction(distributions, p01, p10))


def test_depolarising_correction_takes_rate_of_every_circuit() -> None:
    """A rate per circuit inverts the depolarising noise of every circuit separately."""
    targets = np.random.default_rng(SEED).dirichlet(np.ones(8), size=2)
    noisy = np.stack([depolarise(targets[0], 0.1), depolarise(targets[1], 0.4)])

    corrected = depolarising_correction(noisy, np.array([0.1, 0.4]))

    np.testing.assert_allclose(corrected, targets)


def test_physical_depolarising_rate_counts_shots_hit_by_any_gate() -> None:
    """Rate is the share of shots hit by at least one gate: 0 without gates, and 1 minus the
    product of (1 - error) over all gates otherwise."""
    one_qubit_gates = np.array([[0, 0], [3, 1]])
    two_qubit_gates = np.array([[[0, 0], [0, 0]], [[0, 2], [2, 0]]])
    error_1q = np.array([0.001, 0.002])
    error_2q = np.array([[0.0, 0.01], [0.01, 0.0]])

    rates = physical_depolarising_rate(one_qubit_gates, two_qubit_gates, error_1q, error_2q)

    expected = 1 - 0.999**3 * 0.998 * 0.99**2
    np.testing.assert_allclose(rates, [0.0, expected])


def test_physical_depolarising_rate_stays_below_one() -> None:
    """A circuit with very many gates gets MAX_PHYSICAL_RATE, which can still be inverted."""
    one_qubit_gates = np.array([[10**6, 10**6]])
    two_qubit_gates = np.zeros((1, 2, 2))

    rates = physical_depolarising_rate(
        one_qubit_gates, two_qubit_gates, np.array([0.01, 0.01]), np.zeros((2, 2))
    )

    np.testing.assert_allclose(rates, [MAX_PHYSICAL_RATE])


def test_fit_linear_map_recovers_inverse_of_readout_noise() -> None:
    """A map fitted on 50 circuits with readout noise undoes the noise exactly."""
    targets = np.random.default_rng(SEED).dirichlet(np.ones(4), size=50)
    noisy = add_readout_noise(targets, np.full(2, 0.1), np.full(2, 0.1))

    weights = fit_linear_map(noisy, targets)

    np.testing.assert_allclose(linear_map_correction(noisy, weights), targets, atol=1e-10)


def test_fit_linear_map_without_noise_is_identity() -> None:
    """Fitting on noiseless data gives the identity matrix."""
    targets = np.random.default_rng(SEED).dirichlet(np.ones(4), size=10)

    weights = fit_linear_map(targets, targets)

    np.testing.assert_allclose(weights, np.eye(4), atol=1e-10)


def test_fit_linear_map_rejects_fewer_circuits_than_bitstrings() -> None:
    """3 circuits cannot determine the weights of 4 bitstrings, so it raises ValueError."""
    targets = np.random.default_rng(SEED).dirichlet(np.ones(4), size=3)

    with pytest.raises(ValueError):
        fit_linear_map(targets, targets)


def test_linear_map_correction_clips_negative_probabilities() -> None:
    """Weights that make the entry at bitstring 1 negative give [1, 0] after clipping."""
    distributions = np.array([[0.8, 0.2]])
    weights = np.array([[1.0, -0.5], [0.0, 1.5]])

    corrected = linear_map_correction(distributions, weights)

    np.testing.assert_allclose(corrected, [[1.0, 0.0]])


def test_scaled_physical_rate_scales_the_log_survival() -> None:
    """Scale 1 keeps the rate, 0 removes it and 2 squares the survival probability."""
    rates = np.array([0.0, 0.2, 0.5])

    np.testing.assert_allclose(scaled_physical_rate(rates, 1.0), rates)
    np.testing.assert_allclose(scaled_physical_rate(rates, 0.0), 0.0)
    np.testing.assert_allclose(scaled_physical_rate(rates, 2.0), [0.0, 0.36, 0.75])


def test_scaled_mix_correction_with_scale_one_is_mix_with_the_physical_rate() -> None:
    """Scale 1 gives MIX with the physical rate of every circuit unchanged."""
    distributions = np.random.default_rng(0).dirichlet(np.ones(4), size=3)
    physical_rate = np.array([0.1, 0.3, 0.5])
    p01, p10 = np.array([0.01, 0.02]), np.array([0.03, 0.04])

    scaled = scaled_mix_correction(distributions, 1.0, physical_rate, p01, p10)

    np.testing.assert_allclose(scaled, mix_correction(distributions, physical_rate, p01, p10))
