"""Correction models: one noisy bitstring in, logits over the actions out."""

from pathlib import Path

from torch import nn

from dequbit.config import Config


def build_model(model_config: Config, num_qubits: int) -> nn.Module:
    """Model named in model_config["name"] with its hyperparameters."""
    raise NotImplementedError  # TODO


def save_model(model: nn.Module, model_config: Config, num_qubits: int, path: Path) -> None:
    """Save the weights with everything needed to rebuild the model."""
    raise NotImplementedError  # TODO


def load_model(path: Path) -> nn.Module:
    """Rebuild a model saved with save_model."""
    raise NotImplementedError  # TODO
