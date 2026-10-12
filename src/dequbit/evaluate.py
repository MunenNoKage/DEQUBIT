"""Evaluation of a trained model and the baselines on the test circuits."""

from pathlib import Path

import numpy as np
import torch
from torch import nn


def correct_shots(model: nn.Module, bits: torch.Tensor) -> torch.Tensor:
    """Apply one predicted action to every shot: (shots, num_qubits) -> (shots, num_qubits)."""
    raise NotImplementedError  # TODO: most likely or sampled action


def method_metrics(
    noisy: np.ndarray, corrected: np.ndarray, targets: np.ndarray
) -> dict[str, float]:
    """L1, KL, mean L1 relative change and improved share of one method.

    Args:
        noisy: noisy distributions of the test circuits, shape (circuits, 2**num_qubits).
        corrected: the same distributions after the correction, same shape.
        targets: P_target of the test circuits, same shape.
    """
    raise NotImplementedError  # TODO


def write_metrics(metrics: dict[str, dict[str, float]], path: Path) -> None:
    """Write metrics of every method as JSON."""
    raise NotImplementedError  # TODO


def plot_l1_relative_change(changes: dict[str, np.ndarray], path: Path) -> None:
    """Histogram of the L1 relative change over test circuits, one series per method."""
    raise NotImplementedError  # TODO
