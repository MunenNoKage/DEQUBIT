"""Steps of an experiment: load the circuits, train a model, evaluate every method.

The command line and the notebooks call the same functions, so a notebook shows exactly
what python -m dequbit computes.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from dequbit.config import Config
from dequbit.dataset import (
    load_circuits,
    load_noise_parameters,
    load_two_qubit_gate_counts,
    make_dataset,
    shot_counts,
    split_train_validation,
)
from dequbit.evaluate import (
    INFERENCE_MODES,
    baseline_distributions,
    correct_circuits,
    method_metrics,
)
from dequbit.metrics import l1_relative_change
from dequbit.models import build_model, load_model
from dequbit.training import train
from dequbit.utils import collect_environment, set_seed

SPLITS = ("train", "validation", "test")

logger = logging.getLogger(__name__)


@dataclass
class Circuits:
    """Circuits of one split with everything the corrections and the metrics need.

    Attributes:
        shots: Noisy shots as integers, shape (circuits, shots_per_circuit).
        targets: P_target of every circuit, shape (circuits, 2**num_qubits).
        gate_counts: Two-qubit gates of every circuit, shape (circuits,).
        noise: Parameters of the noise model of the data file, with "p_ro" and "p2".
    """

    shots: np.ndarray
    targets: np.ndarray
    gate_counts: np.ndarray
    noise: dict[str, float]

    @property
    def noisy(self) -> np.ndarray:
        """Noisy distribution of every circuit, shape (circuits, 2**num_qubits)."""
        num_qubits = int(np.log2(self.targets.shape[1]))

        return shot_counts(self.shots, num_qubits) / self.shots.shape[1]


def load_split(config: Config, split: str) -> Circuits:
    """Read the circuits of one split.

    Training and validation circuits are a split of the training file by the config
    seed, so every model seed sees the same circuits. The test circuits are the test
    file.

    Args:
        config: Full config.
        split: One of SPLITS.

    Returns:
        Circuits of the split with shots_per_circuit shots each.

    Raises:
        ValueError: If split is not one of SPLITS.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")

    paths, data = config["paths"], config["data"]
    prefix = "test" if split == "test" else "train"
    clean_path, noisy_path = Path(paths[f"clean_{prefix}"]), Path(paths[f"noisy_{prefix}"])
    shots, targets = load_circuits(clean_path, noisy_path, data["num_qubits"])
    gate_counts = load_two_qubit_gate_counts(clean_path)

    if split != "test":
        train_index, validation_index = split_train_validation(
            len(shots), data["validation_fraction"], config["seed"]
        )
        index = train_index if split == "train" else validation_index
        shots, targets, gate_counts = shots[index], targets[index], gate_counts[index]

    return Circuits(
        shots=shots[:, : data["shots_per_circuit"]],
        targets=targets,
        gate_counts=gate_counts,
        noise=load_noise_parameters(noisy_path),
    )


def model_path(config: Config, model_name: str, seed: int) -> Path:
    """Find the file of the model trained with one seed.

    Args:
        config: Full config.
        model_name: Model name from dequbit.models.MODEL_NAMES.
        seed: Training seed of the model.

    Returns:
        Path of the saved model; its history is next to it with the suffix _history.json.
    """
    return Path(config["paths"]["model"]) / f"{model_name}_seed{seed}.pt"


def train_model(config: Config, model_name: str, seed: int) -> tuple[nn.Module, list[dict]]:
    """Train one model on the training circuits with early stopping on validation.

    Args:
        config: Full config.
        model_name: Model name from dequbit.models.MODEL_NAMES.
        seed: Seed of the initial weights and of the batch order.

    Returns:
        The model with the weights of its best validation, and the training history.
    """
    num_qubits, radius = config["data"]["num_qubits"], config["actions"]["radius"]
    shots_per_circuit = config["data"]["shots_per_circuit"]
    train_circuits = load_split(config, "train")
    validation_circuits = load_split(config, "validation")

    set_seed(seed)
    model = build_model({**config["model"], "name": model_name}, num_qubits, radius)
    history = train(
        model,
        make_dataset(train_circuits.shots, train_circuits.targets, shots_per_circuit, num_qubits),
        make_dataset(
            validation_circuits.shots,
            validation_circuits.targets,
            shots_per_circuit,
            num_qubits,
        ),
        num_qubits,
        radius,
        config["training"],
        seed,
    )

    return model, history


def load_history(config: Config, model_name: str, seed: int) -> list[dict[str, float]]:
    """Read the training history saved next to a model.

    Args:
        config: Full config.
        model_name: Model name from dequbit.models.MODEL_NAMES.
        seed: Training seed of the model.

    Returns:
        One record per validation: step, train_loss, validation_loss, validation_l1rc.
    """
    path = model_path(config, model_name, seed)
    history_path = path.with_name(f"{path.stem}_history.json")

    return json.loads(history_path.read_text(encoding="utf-8"))


def evaluate_baselines(
    config: Config, train_circuits: Circuits, circuits: Circuits
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Correct the circuits with every baseline fitted on the training circuits.

    Args:
        config: Full config.
        train_circuits: Circuits the baseline parameters are fitted on.
        circuits: Circuits to correct.

    Returns:
        Corrected distributions by baseline name, each of shape (circuits, 2**num_qubits),
        and the fitted parameters.
    """
    return baseline_distributions(
        train_circuits.noisy,
        train_circuits.targets,
        circuits.noisy,
        circuits.targets,
        circuits.shots.shape[1],
        circuits.noise,
        circuits.gate_counts,
        config["baselines"],
        config["seed"],
    )


def evaluate_model(
    config: Config, model_name: str, circuits: Circuits
) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluate the saved models of one name for every seed and inference mode.

    Args:
        config: Full config.
        model_name: Model name from dequbit.models.MODEL_NAMES.
        circuits: Circuits to correct.

    Returns:
        Metrics per seed and mode with the mean and standard deviation of the median L1
        relative change over the seeds in the configured mode, and the L1 relative
        changes of all seeds in that mode together, shape (seeds * circuits,).

    Raises:
        FileNotFoundError: If a model of some seed has not been trained.
    """
    num_qubits, radius = config["data"]["num_qubits"], config["actions"]["radius"]
    mode = config["evaluation"]["inference"]
    noisy = circuits.noisy
    generator = torch.Generator().manual_seed(config["seed"])

    per_seed: dict[str, dict[str, Any]] = {}
    mode_changes = []
    for seed in config["evaluation"]["seeds"]:
        path = model_path(config, model_name, seed)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing: python -m dequbit train --model {model_name} --seed {seed}"
            )

        model = load_model(path)
        per_seed[str(seed)] = {}
        for inference in INFERENCE_MODES:
            corrected, flips = correct_circuits(
                model, circuits.shots, num_qubits, radius, inference, generator
            )
            per_seed[str(seed)][inference] = {
                **method_metrics(noisy, corrected, circuits.targets),
                "mean_flipped_bits": float(flips.mean()),
            }
            if inference == mode:
                mode_changes.append(l1_relative_change(noisy, corrected, circuits.targets))

    medians = [per_seed[seed][mode]["median"] for seed in per_seed]
    summary = {
        "inference": mode,
        "median_mean": float(np.mean(medians)),
        "median_std": float(np.std(medians)),
        "seeds": per_seed,
    }

    return summary, np.concatenate(mode_changes)


def evaluate_split(config: Config, split: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate the baselines and every model of evaluation.models on one split.

    Args:
        config: Full config.
        split: "validation" or "test".

    Returns:
        All metrics, as written to paths.metrics, and the L1 relative changes of every
        method by name, for figures.
    """
    train_circuits = load_split(config, "train")
    circuits = load_split(config, split)
    noisy = circuits.noisy

    baselines, parameters = evaluate_baselines(config, train_circuits, circuits)
    metrics: dict[str, Any] = {
        "split": split,
        "num_circuits": len(circuits.shots),
        "environment": collect_environment(),
        "baseline_parameters": parameters,
        "baselines": {
            name: method_metrics(noisy, corrected, circuits.targets)
            for name, corrected in baselines.items()
        },
        "models": {},
    }
    changes = {
        name: l1_relative_change(noisy, corrected, circuits.targets)
        for name, corrected in baselines.items()
    }

    for model_name in config["evaluation"]["models"]:
        logger.info("evaluating %s", model_name)
        metrics["models"][model_name], changes[model_name] = evaluate_model(
            config, model_name, circuits
        )

    return metrics, changes
