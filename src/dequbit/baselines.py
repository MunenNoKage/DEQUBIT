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


def fit_depolarising_rate(noisy: np.ndarray, targets: np.ndarray) -> float:
    """Depolarising rate that best explains the noisy distributions.

    Args:
        noisy: noisy distributions of the training circuits, shape (circuits, 2**num_qubits).
        targets: their ideal distributions, same shape.
    """
    raise NotImplementedError  # TODO


def depolarising_correction(distribution: np.ndarray, rate: float) -> np.ndarray:
    """Invert global depolarising noise.

    The noise model is p_noisy = (1 - rate) * p + rate / 2**num_qubits. Probabilities
    that become negative after the inversion are clipped to zero, then the result is
    renormalised.
    """
    raise NotImplementedError  # TODO
