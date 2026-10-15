"""Loss and training loop with validation and early stopping."""

import torch
from torch import nn
from torch.utils.data import DataLoader

from dequbit.config import Config


def distribution_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean loss over circuits; both inputs have shape (circuits, 2**num_qubits)."""
    raise NotImplementedError  # TODO: cross-entropy or KL divergence


def predict_distributions(model: nn.Module, shots: torch.Tensor) -> torch.Tensor:
    """Expected corrected distribution of every circuit in a batch.

    (circuits, shots_per_circuit, num_qubits) -> (circuits, 2**num_qubits).
    """
    raise NotImplementedError  # TODO


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    """One pass over loader; returns the mean training loss."""
    raise NotImplementedError  # TODO


def validation_loss(model: nn.Module, loader: DataLoader) -> float:
    """Mean loss on loader without updating the model."""
    raise NotImplementedError  # TODO


def train(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    training_config: Config,
) -> nn.Module:
    """Train until early stopping; returns the model with the best validation loss."""
    raise NotImplementedError  # TODO
