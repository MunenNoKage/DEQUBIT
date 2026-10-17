"""Baselines without a model: corrections of a whole noisy distribution.

Distributions have shape (circuits, 2**num_qubits). A correction maps the noisy
distributions to corrected ones of the same shape.
"""

from collections.abc import Callable, Sequence

import numpy as np

from dequbit.metrics import l1_relative_change

Correction = Callable[[np.ndarray, float], np.ndarray]


def oracle_distribution(
    targets: np.ndarray, num_shots: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample shots from the ideal distribution, as a noiseless device would.

    Even a perfect correction ends with a finite number of shots, so the L1 distance
    of these samples to the target is the floor that no correction can beat.

    Args:
        targets: P_target of every circuit, shape (circuits, 2**num_qubits).
        num_shots: Number of shots per circuit, the same as of the noisy shots.
        rng: Random generator of the shots.

    Returns:
        Distribution of the sampled shots, shape (circuits, 2**num_qubits).
    """
    # Float32 targets can sum to slightly more than 1, which multinomial rejects.
    probabilities = targets.astype(np.float64)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    counts = rng.multinomial(num_shots, probabilities)

    return counts / num_shots


def threshold_correction(distributions: np.ndarray, threshold: float) -> np.ndarray:
    """Zero the probabilities below threshold and renormalise.

    The largest probability of a circuit is always kept, so no distribution becomes
    empty.

    Args:
        distributions: Noisy distributions, shape (circuits, 2**num_qubits).
        threshold: Probabilities below this value are removed.

    Returns:
        Corrected distributions, same shape.
    """
    largest = distributions.max(axis=-1, keepdims=True)
    kept = np.where((distributions >= threshold) | (distributions == largest), distributions, 0)

    return kept / kept.sum(axis=-1, keepdims=True)


def best_parameter(
    correction: Correction, grid: Sequence[float], noisy: np.ndarray, targets: np.ndarray
) -> float:
    """Find the parameter of a correction with the best median L1 relative change.

    Args:
        correction: Function of (distributions, parameter) returning corrected
            distributions, for example threshold_correction.
        grid: Parameter values to try.
        noisy: Noisy distributions of the training circuits, shape
            (circuits, 2**num_qubits).
        targets: P_target of the same circuits, same shape.

    Returns:
        Value from grid with the lowest median L1 relative change.

    Raises:
        ValueError: If grid is empty.
    """
    if len(grid) == 0:
        raise ValueError("grid is empty")

    medians = [
        np.median(l1_relative_change(noisy, correction(noisy, value), targets)) for value in grid
    ]

    return float(grid[int(np.argmin(medians))])


def depolarising_correction(distributions: np.ndarray, rate: float) -> np.ndarray:
    """Invert global depolarising noise, the Repolarizer of Placidi et al.

    The noise model replaces a share rate of the shots by uniformly random bitstrings:
    p_noisy = (1 - rate) * p + rate / 2**num_qubits. Solving for p gives the
    correction. Probabilities that become negative are clipped to zero, then the result
    is renormalised.

    Args:
        distributions: Noisy distributions, shape (circuits, 2**num_qubits).
        rate: Share of shots replaced by uniform noise, at least 0 and below 1.

    Returns:
        Corrected distributions, same shape.

    Raises:
        ValueError: If rate is outside [0, 1).
    """
    if not 0 <= rate < 1:
        raise ValueError(f"rate must be in [0, 1), got {rate}")

    num_bitstrings = distributions.shape[-1]

    # At least one probability stays positive: the mean of a distribution is
    # 1 / num_bitstrings, more than rate / num_bitstrings.
    inverted = (distributions - rate / num_bitstrings) / (1 - rate)

    return _clip_and_normalise(inverted)


def readout_correction(distributions: np.ndarray, flip_probability: float) -> np.ndarray:
    """Invert independent readout errors, the SPAM baseline of Placidi et al.

    Readout of every qubit reports the wrong bit with probability flip_probability,
    independently of the other qubits. For one qubit j the noise mixes the probability
    of x with the probability of x with bit j flipped:
    p_noisy(x) = (1 - f) * p(x) + f * p(x ^ 2**j), where f is flip_probability.
    Solving for p gives p(x) = ((1 - f) * p_noisy(x) - f * p_noisy(x ^ 2**j)) / (1 - 2f),
    applied once for every qubit. Probabilities that become negative are clipped to
    zero, then the result is renormalised.

    Args:
        distributions: Noisy distributions, shape (circuits, 2**num_qubits).
        flip_probability: Probability that the readout of one qubit is wrong, at least 0
            and below 0.5.

    Returns:
        Corrected distributions, same shape.

    Raises:
        ValueError: If flip_probability is outside [0, 0.5) or the number of
            bitstrings is not a power of two.
    """
    if not 0 <= flip_probability < 0.5:
        raise ValueError(f"flip_probability must be in [0, 0.5), got {flip_probability}")

    num_bitstrings = distributions.shape[-1]
    num_qubits = num_bitstrings.bit_length() - 1
    if num_bitstrings != 2**num_qubits:
        raise ValueError(f"{num_bitstrings} bitstrings is not a power of two")

    bitstrings = np.arange(num_bitstrings)
    inverted = distributions
    for qubit in range(num_qubits):
        flipped = inverted[..., bitstrings ^ (1 << qubit)]
        inverted = ((1 - flip_probability) * inverted - flip_probability * flipped) / (
            1 - 2 * flip_probability
        )

    # The inversion keeps the sum at 1, so at least one probability stays positive.
    return _clip_and_normalise(inverted)


def mix_correction(distributions: np.ndarray, rate: float, flip_probability: float) -> np.ndarray:
    """Invert readout errors, then global depolarising noise, the MIX of Placidi et al.

    Gate noise acts during the circuit and readout errors at the end, so the readout
    errors are inverted first.

    Args:
        distributions: Noisy distributions, shape (circuits, 2**num_qubits).
        rate: Share of shots replaced by uniform noise, at least 0 and below 1.
        flip_probability: Probability that the readout of one qubit is wrong, at least 0
            and below 0.5.

    Returns:
        Corrected distributions, same shape.
    """
    readout_corrected = readout_correction(distributions, flip_probability)

    return depolarising_correction(readout_corrected, rate)


def physical_depolarising_rate(two_qubit_error: float, num_two_qubit_gates: int) -> float:
    """Estimate the depolarising rate of a circuit from its two-qubit gates.

    A shot is free of gate noise only if no two-qubit gate depolarised it, which
    happens with probability (1 - two_qubit_error)**num_two_qubit_gates.

    Args:
        two_qubit_error: Depolarising probability of one two-qubit gate, at least 0 and
            below 1.
        num_two_qubit_gates: Number of two-qubit gates in the circuit.

    Returns:
        Share of shots replaced by uniform noise.

    Raises:
        ValueError: If two_qubit_error is outside [0, 1) or num_two_qubit_gates is
            negative.
    """
    if not 0 <= two_qubit_error < 1:
        raise ValueError(f"two_qubit_error must be in [0, 1), got {two_qubit_error}")
    if num_two_qubit_gates < 0:
        raise ValueError(f"num_two_qubit_gates must be non-negative, got {num_two_qubit_gates}")

    return 1 - (1 - two_qubit_error) ** num_two_qubit_gates


def fit_linear_map(noisy: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Fit one matrix that maps noisy distributions to ideal ones, by least squares.

    The map has 4**num_qubits weights, so it is only practical for few qubits. A
    constant term is not needed: every distribution sums to 1, so the sum of the
    columns of noisy is already a constant input.

    Args:
        noisy: Noisy distributions of the training circuits, shape
            (circuits, 2**num_qubits).
        targets: P_target of the same circuits, same shape.

    Returns:
        Weights, shape (2**num_qubits, 2**num_qubits); the corrected distribution of
        a circuit is its noisy distribution times the weights.

    Raises:
        ValueError: If there are fewer circuits than bitstrings, so the weights are not
            determined.
    """
    num_circuits, num_bitstrings = noisy.shape
    if num_circuits < num_bitstrings:
        raise ValueError(f"{num_circuits} circuits cannot determine {num_bitstrings} weights")

    weights, *_ = np.linalg.lstsq(noisy, targets, rcond=None)

    return weights


def linear_map_correction(distributions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Apply a fitted linear map, then clip negative probabilities and renormalise.

    Args:
        distributions: Noisy distributions, shape (circuits, 2**num_qubits).
        weights: Matrix from fit_linear_map, shape (2**num_qubits, 2**num_qubits).

    Returns:
        Corrected distributions, same shape as distributions.
    """
    # The least-squares map keeps the sum near 1, so at least one probability stays
    # positive.
    return _clip_and_normalise(distributions @ weights)


def _clip_and_normalise(distributions: np.ndarray) -> np.ndarray:
    """Set negative probabilities to zero and rescale every distribution to sum 1.

    Args:
        distributions: Corrected distributions that may have negative entries, shape
            (circuits, 2**num_qubits), with a positive entry in every row.

    Returns:
        Valid distributions, same shape.
    """
    clipped = np.maximum(distributions, 0)

    return clipped / clipped.sum(axis=-1, keepdims=True)
