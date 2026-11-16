"""Baselines without a model: corrections of a whole noisy distribution.

Distributions have shape (circuits, 2**num_qubits). A correction maps the noisy
distributions to corrected ones of the same shape.
"""

from collections.abc import Callable, Sequence

import numpy as np

from dequbit.metrics import l1_relative_change

Correction = Callable[[np.ndarray, float], np.ndarray]

# A rate of 1 means no shot survived the gates, which cannot be inverted.
MAX_PHYSICAL_RATE = 0.999


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


def depolarising_correction(distributions: np.ndarray, rate: float | np.ndarray) -> np.ndarray:
    """Invert global depolarising noise, the Repolarizer of Placidi et al.

    The noise model replaces a share rate of the shots by uniformly random bitstrings:
    p_noisy = (1 - rate) * p + rate / 2**num_qubits. Solving for p gives the
    correction. Probabilities that become negative are clipped to zero, then the result
    is renormalised.

    Args:
        distributions: Noisy distributions, shape (circuits, 2**num_qubits).
        rate: Share of shots replaced by uniform noise, at least 0 and below 1: one
            value for all circuits or one per circuit, shape (circuits,).

    Returns:
        Corrected distributions, same shape.

    Raises:
        ValueError: If a rate is outside [0, 1).
    """
    rates = np.asarray(rate, dtype=np.float64)
    if (rates < 0).any() or (rates >= 1).any():
        raise ValueError(f"rate must be in [0, 1), got {rate}")

    num_bitstrings = distributions.shape[-1]

    # At least one probability stays positive: the mean of a distribution is
    # 1 / num_bitstrings, more than rate / num_bitstrings.
    rates = rates[..., np.newaxis]
    inverted = (distributions - rates / num_bitstrings) / (1 - rates)

    return _clip_and_normalise(inverted)


def readout_correction(distributions: np.ndarray, p01: np.ndarray, p10: np.ndarray) -> np.ndarray:
    """Invert independent readout errors, the SPAM baseline of Placidi et al.

    Readout of bit j reports 1 for a qubit in state 0 with probability p01[j] and 0 for
    a qubit in state 1 with probability p10[j], independently of the other bits. For one
    bit the noise mixes the probability a of a bitstring with this bit 0 and the
    probability b of the same bitstring with this bit 1:
    a_noisy = (1 - p01) * a + p10 * b and b_noisy = p01 * a + (1 - p10) * b.
    Solving for a and b gives a = ((1 - p10) * a_noisy - p10 * b_noisy) / (1 - p01 - p10)
    and b = ((1 - p01) * b_noisy - p01 * a_noisy) / (1 - p01 - p10), applied once for
    every bit. Probabilities that become negative are clipped to zero, then the result
    is renormalised.

    Args:
        distributions: Noisy distributions, shape (circuits, 2**num_qubits).
        p01: Probability of reading 1 for state 0 of every bit, shape (num_qubits,) or
            (circuits, num_qubits).
        p10: Probability of reading 0 for state 1 of every bit, same shape as p01.

    Returns:
        Corrected distributions, same shape.

    Raises:
        ValueError: If a probability is negative, p01 + p10 is not below 1 for some
            bit, or the rates do not have one column per bit.
    """
    num_bitstrings = distributions.shape[-1]
    num_qubits = num_bitstrings.bit_length() - 1
    p01, p10 = np.asarray(p01, dtype=np.float64), np.asarray(p10, dtype=np.float64)
    if p01.shape[-1] != num_qubits or p10.shape != p01.shape or num_bitstrings != 2**num_qubits:
        raise ValueError(
            f"rates of shape {p01.shape} and {p10.shape} do not describe the bits of "
            f"{num_bitstrings} bitstrings"
        )
    if (p01 < 0).any() or (p10 < 0).any() or (p01 + p10 >= 1).any():
        raise ValueError("readout rates must be non-negative with p01 + p10 below 1")

    bitstrings = np.arange(num_bitstrings)
    inverted = distributions
    for bit in range(num_qubits):
        # Shape (2**num_qubits,) or (circuits, 2**num_qubits): for every bitstring the
        # rate of leaving its own value of this bit and of arriving from the other one.
        bit_is_one = (bitstrings >> bit) & 1 == 1
        leave = np.where(bit_is_one, p10[..., bit, np.newaxis], p01[..., bit, np.newaxis])
        arrive = np.where(bit_is_one, p01[..., bit, np.newaxis], p10[..., bit, np.newaxis])
        flipped = inverted[..., bitstrings ^ (1 << bit)]
        inverted = ((1 - arrive) * inverted - arrive * flipped) / (1 - leave - arrive)

    # The inversion keeps the sum at 1, so at least one probability stays positive.
    return _clip_and_normalise(inverted)


def mix_correction(
    distributions: np.ndarray, rate: float | np.ndarray, p01: np.ndarray, p10: np.ndarray
) -> np.ndarray:
    """Invert readout errors, then global depolarising noise, the MIX of Placidi et al.

    Gate noise acts during the circuit and readout errors at the end, so the readout
    errors are inverted first.

    Args:
        distributions: Noisy distributions, shape (circuits, 2**num_qubits).
        rate: Share of shots replaced by uniform noise, at least 0 and below 1: one
            value for all circuits or one per circuit, shape (circuits,).
        p01: Probability of reading 1 for state 0 of every bit, shape (num_qubits,) or
            (circuits, num_qubits).
        p10: Probability of reading 0 for state 1 of every bit, same shape as p01.

    Returns:
        Corrected distributions, same shape.
    """
    readout_corrected = readout_correction(distributions, p01, p10)

    return depolarising_correction(readout_corrected, rate)


def physical_depolarising_rate(
    one_qubit_gates: np.ndarray,
    two_qubit_gates: np.ndarray,
    error_1q: np.ndarray,
    error_2q: np.ndarray,
) -> np.ndarray:
    """Estimate the depolarising rate of every circuit from its gates and their errors.

    A shot is free of gate noise only if no gate failed, which happens with probability
    prod((1 - error of the gate) for every gate of the circuit).

    Args:
        one_qubit_gates: One-qubit gates on every device qubit, shape
            (circuits, num_qubits).
        two_qubit_gates: Two-qubit gates between every pair of device qubits, symmetric,
            shape (circuits, num_qubits, num_qubits).
        error_1q: Error of a one-qubit gate on every device qubit, shape (num_qubits,).
        error_2q: Error of the two-qubit gate between every pair of device qubits,
            symmetric, shape (num_qubits, num_qubits).

    Returns:
        Share of shots replaced by uniform noise in every circuit, shape (circuits,),
        at most MAX_PHYSICAL_RATE.
    """
    log_survival = one_qubit_gates @ np.log1p(-error_1q)
    # Both matrices are symmetric, so the sum counts every pair twice.
    log_survival = log_survival + (two_qubit_gates * np.log1p(-error_2q)).sum(axis=(1, 2)) / 2

    return np.minimum(1 - np.exp(log_survival), MAX_PHYSICAL_RATE)


def scaled_physical_rate(physical_rate: np.ndarray, scale: float) -> np.ndarray:
    """Scale the gate noise behind a physical depolarising rate.

    The rate 1 - prod(1 - error) becomes 1 - prod(1 - error)**scale, as if every gate
    error were scale times larger in the log. Scale 1 keeps the rate and 0 removes it.

    Args:
        physical_rate: Output of physical_depolarising_rate, shape (circuits,).
        scale: Factor of the log survival probability, at least 0.

    Returns:
        Scaled rate of every circuit, shape (circuits,), at most MAX_PHYSICAL_RATE.
    """
    return np.minimum(1 - (1 - physical_rate) ** scale, MAX_PHYSICAL_RATE)


def scaled_mix_correction(
    distributions: np.ndarray,
    scale: float,
    physical_rate: np.ndarray,
    p01: np.ndarray,
    p10: np.ndarray,
) -> np.ndarray:
    """Apply MIX with the physical depolarising rate of every circuit scaled by scale.

    Args:
        distributions: Noisy distributions, shape (circuits, 2**num_qubits).
        scale: Factor of the log survival probability, see scaled_physical_rate.
        physical_rate: Output of physical_depolarising_rate, shape (circuits,).
        p01: Probability of reading 1 for state 0 of every bit, shape (num_qubits,) or
            (circuits, num_qubits).
        p10: Probability of reading 0 for state 1, same shape as p01.

    Returns:
        Corrected distributions, same shape as distributions.
    """
    return mix_correction(distributions, scaled_physical_rate(physical_rate, scale), p01, p10)


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
