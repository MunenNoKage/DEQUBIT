"""Distances between distributions and the summary metrics of the evaluation."""

import numpy as np


def l1_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Sum of absolute differences between two distributions."""
    raise NotImplementedError  # TODO


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) between two distributions."""
    raise NotImplementedError  # TODO: zero probabilities in q


def l1_relative_change(noisy: np.ndarray, corrected: np.ndarray, target: np.ndarray) -> float:
    """Change of the L1 distance to target as a fraction of the noisy L1 distance.

    (L1(corrected, target) - L1(noisy, target)) / L1(noisy, target). Negative means the
    correction helped.
    """
    raise NotImplementedError  # TODO


def improved_fraction(relative_changes: np.ndarray) -> float:
    """Share of circuits with a negative L1 relative change."""
    raise NotImplementedError  # TODO


def mean_flipped_bits(noisy_bits: np.ndarray, corrected_bits: np.ndarray) -> float:
    """Average number of changed bits per shot; both inputs have shape (shots, num_qubits)."""
    raise NotImplementedError  # TODO
