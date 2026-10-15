"""Baselines without a model: corrections of a whole noisy distribution."""

import numpy as np


def threshold_correction(distribution: np.ndarray, threshold: float) -> np.ndarray:
    """Zero the probabilities below threshold and renormalise."""
    raise NotImplementedError  # TODO


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
