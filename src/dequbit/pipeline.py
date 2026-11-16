"""Steps of an experiment: load the circuits, train a model, evaluate every method.

The command line and the notebooks call the same functions, so a notebook shows exactly
what python -m dequbit computes.

The data has circuits of several numbers of qubits, data.qubit_counts. A model of
dequbit.models.SIZE_INDEPENDENT_MODELS is trained once on all of them; every other model
is trained once per number of qubits. Every method is evaluated per number of qubits of
evaluation.qubit_counts, which may hold numbers no model was trained on: there only the
baselines and the models of SIZE_INDEPENDENT_MODELS are evaluated.
"""

import json
import logging
from functools import lru_cache
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
from dequbit.metrics import l1_relative_change, summarize_relative_changes
from dequbit.models import SIZE_INDEPENDENT_MODELS, build_model, load_model
from dequbit.training import train
from dequbit.utils import collect_environment, resolve_device, set_seed

SPLITS = ("train", "validation", "test")
# Names of the splits in the file names of data2/generation.py.
SPLIT_FILE_NAMES = {"train": "train", "validation": "val", "test": "test"}
# Files kept in memory after the first read: every split of every number of qubits, so
# training several models or seeds in one run reads each file once.
CACHED_FILES = 32

# Counting the shots of a file takes up to half a minute, and the circuits are only read.
_load_circuits = lru_cache(maxsize=CACHED_FILES)(load_circuits)

logger = logging.getLogger(__name__)


def load_split(config: Config, split: str, num_qubits: int) -> Circuits:
    """Read the circuits of one split and one number of qubits.

    Args:
        config: Full config.
        split: One of SPLITS.
        num_qubits: One of data.qubit_counts.

    Returns:
        Circuits of the split with the counts of shots_per_circuit shots each. A file
        read before in the same process is returned from memory, so the result must
        not be changed.

    Raises:
        ValueError: If split is not one of SPLITS.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")

    path = config["paths"]["data"].format(split=SPLIT_FILE_NAMES[split], num_qubits=num_qubits)

    return _load_circuits(Path(path), config["data"]["shots_per_circuit"])


def model_sizes(config: Config, model_name: str) -> list[int | None]:
    """List the numbers of qubits a model is trained for separately.

    Args:
        config: Full config.
        model_name: Model name from dequbit.models.MODEL_NAMES.

    Returns:
        [None] for a model trained once on every number of qubits together, otherwise
        data.qubit_counts.
    """
    if model_name in SIZE_INDEPENDENT_MODELS:
        return [None]

    return list(config["data"]["qubit_counts"])


def model_path(config: Config, model_name: str, seed: int, num_qubits: int | None) -> Path:
    """Find the file of the model trained with one seed.

    Args:
        config: Full config.
        model_name: Model name from dequbit.models.MODEL_NAMES.
        seed: Training seed of the model.
        num_qubits: Number of qubits the model was trained for; None for a model
            trained on every number of qubits together.

    Returns:
        Path of the saved model; its history is next to it with the suffix _history.json.
    """
    size = "" if num_qubits is None else f"_N{num_qubits}"

    return Path(config["paths"]["model"]) / f"{model_name}{size}_seed{seed}.pt"


def train_model(
    config: Config, model_name: str, seed: int, num_qubits: int | None
) -> tuple[nn.Module, list[dict]]:
    """Train one model on the training circuits with early stopping on validation.

    Args:
        config: Full config.
        model_name: Model name from dequbit.models.MODEL_NAMES.
        seed: Seed of the initial weights and of the batch order.
        num_qubits: Number of qubits of the circuits; None trains on the circuits of
            every number of data.qubit_counts together.

    Returns:
        The model on the CPU with the weights of its best validation, and the training
        history.
    """
    radius = config["actions"]["radius"]
    qubit_counts = config["data"]["qubit_counts"] if num_qubits is None else [num_qubits]
    train_sets = {size: make_dataset(load_split(config, "train", size)) for size in qubit_counts}
    validation_sets = {
        size: make_dataset(load_split(config, "validation", size)) for size in qubit_counts
    }

    set_seed(seed)
    model = build_model({**config["model"], "name": model_name}, num_qubits, radius)
    model.to(resolve_device(config["training"]["device"]))
    history = train(model, train_sets, validation_sets, radius, config["training"], seed)

    return model.cpu(), history


def load_history(
    config: Config, model_name: str, seed: int, num_qubits: int | None
) -> list[dict[str, Any]]:
    """Read the training history saved next to a model.

    Args:
        config: Full config.
        model_name: Model name from dequbit.models.MODEL_NAMES.
        seed: Training seed of the model.
        num_qubits: Number of qubits the model was trained for; None for a model
            trained on every number of qubits together.

    Returns:
        One record per validation: step, train_loss, validation_loss, validation_l1rc
        and validation_l1rc_by_qubits.
    """
    path = model_path(config, model_name, seed, num_qubits)
    history_path = path.with_name(f"{path.stem}_history.json")

    return json.loads(history_path.read_text(encoding="utf-8"))


def evaluate_baselines(
    config: Config, train_circuits: Circuits, circuits: Circuits
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Correct the circuits with every baseline fitted on the training circuits.

    Args:
        config: Full config.
        train_circuits: Circuits the baseline parameters are fitted on.
        circuits: Circuits to correct, with the same number of qubits.

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
        circuits: Circuits of one number of qubits to correct.

    Returns:
        Metrics per seed and mode with the mean and standard deviation of the median L1
        relative change over the seeds in the configured mode, and the L1 relative
        changes of all seeds in that mode one after another, shape (seeds * circuits,).

    Raises:
        FileNotFoundError: If a model of some seed has not been trained.
    """
    radius = config["actions"]["radius"]
    mode = config["evaluation"]["inference"]
    trained_size = None if model_name in SIZE_INDEPENDENT_MODELS else circuits.num_qubits
    noisy = circuits.noisy
    qubit_features = circuits.qubit_features
    rng = np.random.default_rng(config["seed"])

    per_seed: dict[str, dict[str, Any]] = {}
    mode_changes = []
    for seed in config["evaluation"]["seeds"]:
        path = model_path(config, model_name, seed, trained_size)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing: python -m dequbit train --model {model_name} --seed {seed}"
            )

        model = load_model(path)
        per_seed[str(seed)] = {}
        for inference in INFERENCE_MODES:
            corrected, flips = correct_circuits(
                model,
                circuits.counts,
                qubit_features,
                circuits.num_qubits,
                radius,
                inference,
                rng,
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


def evaluated_models(config: Config, num_qubits: int) -> list[str]:
    """List the models of evaluation.models that exist for one number of qubits.

    Args:
        config: Full config.
        num_qubits: Number of qubits of the evaluated circuits.

    Returns:
        Every model of SIZE_INDEPENDENT_MODELS, and the other models only if num_qubits
        is one of data.qubit_counts.
    """
    trained = num_qubits in config["data"]["qubit_counts"]

    return [
        name
        for name in config["evaluation"]["models"]
        if trained or name in SIZE_INDEPENDENT_MODELS
    ]


def evaluate_size(
    config: Config, split: str, num_qubits: int
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate the baselines and every model of evaluation.models on one number of qubits.

    A model trained per number of qubits is left out where num_qubits is not one of
    data.qubit_counts, since no such model exists.

    Args:
        config: Full config.
        split: "validation" or "test".
        num_qubits: One of evaluation.qubit_counts.

    Returns:
        Metrics of the circuits with num_qubits qubits, and the L1 relative changes of
        every method by name, for figures.
    """
    train_circuits = load_split(config, "train", num_qubits)
    circuits = load_split(config, split, num_qubits)
    noisy = circuits.noisy

    baselines, parameters = evaluate_baselines(config, train_circuits, circuits)
    metrics: dict[str, Any] = {
        "num_circuits": len(circuits.counts),
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

    for model_name in evaluated_models(config, num_qubits):
        logger.info("evaluating %s on %d qubits", model_name, num_qubits)
        metrics["models"][model_name], changes[model_name] = evaluate_model(
            config, model_name, circuits
        )
    metrics["circuit_types"] = _metrics_by_type(changes, circuits.circuit_types)

    return metrics, changes


def evaluate_split(
    config: Config, split: str
) -> tuple[dict[str, Any], dict[int, dict[str, np.ndarray]]]:
    """Evaluate the baselines and the models on every number of qubits of one split.

    Args:
        config: Full config.
        split: "validation" or "test".

    Returns:
        All metrics, as written to paths.metrics, and the L1 relative changes of every
        method by number of qubits and method name, for figures.
    """
    metrics: dict[str, Any] = {
        "split": split,
        "environment": collect_environment(),
        "qubits": {},
    }
    changes = {}
    for num_qubits in config["evaluation"]["qubit_counts"]:
        metrics["qubits"][str(num_qubits)], changes[num_qubits] = evaluate_size(
            config, split, num_qubits
        )

    return metrics, changes


def _metrics_by_type(
    changes: dict[str, np.ndarray], circuit_types: np.ndarray
) -> dict[str, dict[str, dict[str, float]]]:
    """Summarize the L1 relative change of every method for every circuit family.

    Args:
        changes: L1 relative changes of every method by name. A baseline has shape
            (circuits,); a model has its seeds one after another, shape
            (seeds * circuits,).
        circuit_types: Name of the circuit family of every circuit, shape (circuits,).

    Returns:
        Median, quartiles and improved share by circuit family and method name.
    """
    by_type: dict[str, dict[str, dict[str, float]]] = {}
    for circuit_type in sorted(set(circuit_types)):
        by_type[circuit_type] = {}
        for name, method_changes in changes.items():
            repeats = len(method_changes) // len(circuit_types)
            of_type = np.tile(circuit_types == circuit_type, repeats)
            by_type[circuit_type][name] = summarize_relative_changes(method_changes[of_type])

    return by_type
