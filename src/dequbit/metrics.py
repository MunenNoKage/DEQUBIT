"""Distances between distributions and the summary metrics of the evaluation.

Distances take arrays of shape (circuits, 2**num_qubits) and return one value per
circuit, shape (circuits,).
"""

import math

import numpy as np

# Probabilities are clipped to this value before a logarithm, so a bitstring that the
# corrected distribution misses gives a large but finite KL divergence.
PROBABILITY_FLOOR = 1e-12


def l1_distance(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compute the sum of absolute differences between two distributions.

    Args:
        p: Distributions, shape (circuits, 2**num_qubits).
        q: Distributions, same shape as p.

    Returns:
        L1 distance of every circuit, shape (circuits,), between 0 and 2.
    """
    return np.abs(p - q).sum(axis=-1)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compute KL(p || q), the extra log loss of predicting q when the truth is p.

    Args:
        p: True distributions, shape (circuits, 2**num_qubits).
        q: Predicted distributions, same shape as p.

    Returns:
        KL divergence of every circuit, shape (circuits,), in nats.
    """
    # Bitstrings with p = 0 add nothing: their term is multiplied by p.
    log_ratio = np.log(np.maximum(p, PROBABILITY_FLOOR)) - np.log(np.maximum(q, PROBABILITY_FLOOR))

    return (p * log_ratio).sum(axis=-1)


def hellinger_distance(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compute the Hellinger distance between two distributions.

    Args:
        p: Distributions, shape (circuits, 2**num_qubits).
        q: Distributions, same shape as p.

    Returns:
        Hellinger distance of every circuit, shape (circuits,), between 0 and 1.
    """
    squared_differences = (np.sqrt(p) - np.sqrt(q)) ** 2

    return np.sqrt(squared_differences.sum(axis=-1) / 2)


def l1_relative_change(noisy: np.ndarray, corrected: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute the change of the L1 distance to the target as a share of the noisy one.

    (L1(corrected, target) - L1(noisy, target)) / L1(noisy, target). It is -1 for a
    perfect correction, 0 when nothing changed and positive when the correction hurt.

    Args:
        noisy: Noisy distributions, shape (circuits, 2**num_qubits).
        corrected: Corrected distributions, same shape.
        target: P_target, same shape.

    Returns:
        L1 relative change of every circuit, shape (circuits,).

    Raises:
        ValueError: If a noisy distribution equals its target.
    """
    noisy_distance = l1_distance(noisy, target)
    if (noisy_distance == 0).any():
        raise ValueError("a noisy distribution equals its target")

    corrected_distance = l1_distance(corrected, target)

    return (corrected_distance - noisy_distance) / noisy_distance


def improved_fraction(relative_changes: np.ndarray) -> float:
    """Compute the share of circuits that the correction brought closer to the target.

    Args:
        relative_changes: L1 relative change of every circuit, shape (circuits,).

    Returns:
        Share of negative changes, between 0 and 1.
    """
    return float(np.mean(relative_changes < 0))


def summarize_relative_changes(relative_changes: np.ndarray) -> dict[str, float]:
    """Summarize the L1 relative changes of all circuits.

    The median and the quartiles are used instead of the mean, because a few circuits
    with a tiny noisy distance can have very large relative changes.

    Args:
        relative_changes: L1 relative change of every circuit, shape (circuits,).

    Returns:
        Median, first and third quartile, and the improved share.
    """
    first_quartile, median, third_quartile = np.percentile(relative_changes, [25, 50, 75])

    return {
        "median": float(median),
        "first_quartile": float(first_quartile),
        "third_quartile": float(third_quartile),
        "improved_fraction": improved_fraction(relative_changes),
    }


def median_confidence_interval(
    relative_changes: np.ndarray, rng: np.random.Generator, resamples: int, level: float
) -> tuple[float, float]:
    """Estimate a bootstrap confidence interval of the median L1 relative change.

    Circuits are drawn with replacement. For a learned model every row holds the
    circuits of one training seed, and the statistic is the mean over the seeds of the
    median over the circuits, the number reported for the model.

    Args:
        relative_changes: L1 relative change of every circuit, shape (circuits,), or of
            every seed and circuit, shape (seeds, circuits).
        rng: Random generator of the resampled circuits.
        resamples: Number of bootstrap samples.
        level: Probability that the interval covers the median, for example 0.95.

    Returns:
        Lower and upper end of the interval.
    """
    changes = np.atleast_2d(relative_changes)
    num_circuits = changes.shape[1]

    picks = rng.integers(0, num_circuits, size=(resamples, num_circuits))
    statistics = np.median(changes[:, picks], axis=-1).mean(axis=0)
    tail = 100 * (1 - level) / 2
    low, high = np.percentile(statistics, [tail, 100 - tail])

    return float(low), float(high)


def paired_comparison(
    first: np.ndarray,
    second: np.ndarray,
    rng: np.random.Generator,
    resamples: int,
    level: float,
) -> dict[str, float]:
    """Compare two corrections of the same circuits, circuit by circuit.

    Args:
        first: L1 relative change of every circuit after the first correction, shape
            (circuits,).
        second: The same after the second correction, same shape.
        rng: Random generator of the bootstrap.
        resamples: Number of bootstrap samples.
        level: Coverage of the confidence interval, for example 0.95.

    Returns:
        Median of first - second (negative when the first correction is better) with
        its bootstrap confidence interval, the share of circuits where the first is
        better, and the two-sided p-value of the Wilcoxon signed-rank test.
    """
    differences = first - second

    low, high = median_confidence_interval(differences, rng, resamples, level)

    return {
        "median_difference": float(np.median(differences)),
        "difference_low": low,
        "difference_high": high,
        "first_better_fraction": float(np.mean(differences < 0)),
        "wilcoxon_p": wilcoxon_p_value(differences),
    }


def wilcoxon_p_value(differences: np.ndarray) -> float:
    """Test whether paired differences are centred at zero (Wilcoxon signed-rank test).

    Zero differences are dropped. The rank sum of the positive differences is compared
    with its distribution under the null hypothesis through the normal approximation,
    with the variance corrected for tied ranks; it is accurate from about 20 nonzero
    differences.

    Args:
        differences: Paired differences, shape (pairs,).

    Returns:
        Two-sided p-value; 1 when every difference is zero.
    """
    nonzero = differences[differences != 0]
    count = len(nonzero)
    if count == 0:
        return 1.0

    ranks = _average_ranks(np.abs(nonzero))
    positive_rank_sum = ranks[nonzero > 0].sum()
    _, tie_sizes = np.unique(np.abs(nonzero), return_counts=True)
    mean = count * (count + 1) / 4
    variance = count * (count + 1) * (2 * count + 1) / 24
    variance -= (tie_sizes**3 - tie_sizes).sum() / 48
    z_score = (positive_rank_sum - mean) / math.sqrt(variance)

    return math.erfc(abs(z_score) / math.sqrt(2))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Rank values from 1 upwards, giving tied values the mean of their ranks.

    Args:
        values: Values to rank, shape (values,).

    Returns:
        Rank of every value, shape (values,).
    """
    order = np.argsort(values, kind="stable")
    _, group, sizes = np.unique(values[order], return_inverse=True, return_counts=True)
    last_rank = np.cumsum(sizes)
    group_rank = last_rank - (sizes - 1) / 2

    ranks = np.empty(len(values))
    ranks[order] = group_rank[group]

    return ranks


def mean_flipped_bits(noisy_shots: np.ndarray, corrected_shots: np.ndarray) -> float:
    """Compute the average number of bits the correction changed in a shot.

    Args:
        noisy_shots: Shots as integers, any shape (...).
        corrected_shots: Corrected shots, same shape.

    Returns:
        Mean number of changed bits per shot.
    """
    changed_bits = np.bitwise_count(noisy_shots ^ corrected_shots)

    return float(changed_bits.mean())


def unreachable_mass(
    counts: np.ndarray, targets: np.ndarray, destination_table: np.ndarray
) -> np.ndarray:
    """Compute the target mass that no action can move a measured shot to.

    A bitstring is reachable if some action moves a measured shot onto it. Actions are
    XOR masks, so this holds when some action moves the bitstring onto a measured shot.
    The unreachable mass is a lower bound on what the L1 distance of any correction
    with these actions misses.

    Args:
        counts: Shot counts of every circuit, shape (circuits, 2**num_qubits).
        targets: P_target of every circuit, same shape.
        destination_table: Output of dequbit.actions.destinations as an array, shape
            (2**num_qubits, actions).

    Returns:
        Unreachable target mass of every circuit, shape (circuits,).
    """
    reachable = (counts[:, destination_table] > 0).any(axis=-1)

    return (targets * ~reachable).sum(axis=-1)
