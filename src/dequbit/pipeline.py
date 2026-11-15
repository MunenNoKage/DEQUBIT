"""Steps of an experiment: load the circuits, train a model, evaluate every method.

The command line and the notebooks call the same functions, so a notebook shows exactly
what python -m dequbit computes.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from torch import nn

from dequbit.config import Config
from dequbit.dataset import Circuits, load_circuits, make_dataset
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
# Names of the splits in the file names of data2/generation.py.
SPLIT_FILE_NAMES = {"train": "train", "validation": "val", "test": "test"}

logger = logging.getLogger(__name__)


def load_split(config: Config, split: str) -> Circuits:
    """Read the circuits of one split with data.num_qubits qubits.

    Args:
        config: Full config.
        split: One of SPLITS.

    Returns:
        Circuits of the split with the counts of shots_per_circuit shots each.

    Raises:
        ValueError: If split is not one of SPLITS.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")

    data = config["data"]
    path = config["paths"]["data"].format(
        split=SPLIT_FILE_NAMES[split], num_qubits=data["num_qubits"]
    )

    return load_circuits(Path(path), data["shots_per_circuit"])


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
    train_circuits = load_split(config, "train")
    validation_circuits = load_split(config, "validation")

    set_seed(seed)
    model = build_model({**config["model"], "name": model_name}, num_qubits, radius)
    history = train(
        model,
        make_dataset(train_circuits),
        make_dataset(validation_circuits),
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
    return baseline_distributions(train_circuits, circuits, config["baselines"], config["seed"])


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
    rng = np.random.default_rng(config["seed"])

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
                model, circuits.counts, num_qubits, radius, inference, rng
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
        "num_circuits": len(circuits.counts),
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
