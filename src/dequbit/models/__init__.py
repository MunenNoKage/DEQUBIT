"""Correction models: shot counts of a circuit in, action logits of every bitstring out."""

from pathlib import Path

import torch
from torch import nn

from dequbit.config import Config
from dequbit.models.mlp import MLPCorrector
from dequbit.models.per_qubit import PerQubitFlipModel
from dequbit.models.shared_scorer import SharedScorer

MODEL_NAMES = ("mlp", "shared_scorer", "per_qubit")


def build_model(model_config: Config, num_qubits: int, radius: int) -> nn.Module:
    """Create the model named in model_config["name"] with its hyperparameters.

    Args:
        model_config: Model section of the config: the name and one section of
            hyperparameters per model, for example {"name": "mlp", "mlp":
            {"hidden_sizes": [128, 128]}}.
        num_qubits: Number of measured qubits.
        radius: Largest number of bits one action flips.

    Returns:
        Untrained model that maps counts (circuits, 2**num_qubits) to action logits
        (circuits, 2**num_qubits, actions).

    Raises:
        ValueError: If the name is not one of MODEL_NAMES.
    """
    name = model_config["name"]
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model {name!r}, expected one of {MODEL_NAMES}")

    if name == "mlp":
        return MLPCorrector(num_qubits, radius, model_config["mlp"]["hidden_sizes"])
    if name == "shared_scorer":
        return SharedScorer(num_qubits, radius, model_config["shared_scorer"]["hidden_sizes"])

    return PerQubitFlipModel(num_qubits, radius)


def save_model(
    model: nn.Module, model_config: Config, num_qubits: int, radius: int, path: Path
) -> None:
    """Save the weights with everything needed to rebuild the model.

    Args:
        model: Model created by build_model.
        model_config: Model section of the config the model was built from.
        num_qubits: Number of measured qubits of the model.
        radius: Largest number of bits one action of the model flips.
        path: File to write; missing parent folders are created.
    """
    checkpoint = {
        "model_config": model_config,
        "num_qubits": num_qubits,
        "radius": radius,
        "state_dict": model.state_dict(),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_model(path: Path) -> nn.Module:
    """Rebuild a model saved with save_model.

    Args:
        path: File written by save_model.

    Returns:
        The model with the saved weights, on the CPU.
    """
    # weights_only reads tensors and plain Python values but never runs code from the file.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    model = build_model(checkpoint["model_config"], checkpoint["num_qubits"], checkpoint["radius"])
    model.load_state_dict(checkpoint["state_dict"])

    return model
