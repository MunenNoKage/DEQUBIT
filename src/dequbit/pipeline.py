"""Steps of an experiment: load the circuits, train a model, evaluate every method.

The command line and the notebooks call the same functions, so a notebook shows exactly
what python -m dequbit computes.

The data has circuits of several numbers of qubits, data.qubit_counts. A model of
dequbit.models.SIZE_INDEPENDENT_MODELS is trained once on all of them; every other model
is trained once per number of qubits. Every method is evaluated per number of qubits of
evaluation.qubit_counts, which may hold numbers no model was trained on: there only the
baselines and the models of SIZE_INDEPENDENT_MODELS are evaluated.

Clifford circuits of data2/clifford_generation.py, up to 63 qubits, are evaluated
without 2**num_qubits vectors by dequbit.sparse, with the size-independent models only.
"""

import copy
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from dequbit.config import Config, apply_overrides
from dequbit.dataset import (
    Circuits,
    load_circuits,
    make_dataset,
    perturb_calibration,
    select_circuits,
)
from dequbit.evaluate import (
    INFERENCE_MODES,
    REFERENCE_METHODS,
    baseline_distributions,
    correct_circuits,
    method_metrics,
)
from dequbit.metrics import (
    l1_relative_change,
    median_confidence_interval,
    paired_comparison,
    summarize_relative_changes,
)
from dequbit.models import SIZE_INDEPENDENT_MODELS, build_model, load_model
from dequbit.sparse import (
    SPARSE_INFERENCE_MODES,
    SparseCircuit,
    corrected_shots,
    corrected_support_mass,
    hammer_support_mass,
    l1_from_support_mass,
    load_sparse_circuits,
    noisy_support_mass,
    oracle_support_mass,
)
from dequbit.training import train
from dequbit.utils import collect_environment, resolve_device, set_seed

SPLITS = ("train", "validation", "test")
# Names of the splits in the file names of data2/generation.py.
SPLIT_FILE_NAMES = {"train": "train", "validation": "val", "test": "test"}
# Files kept in memory after the first read: every split of every number of qubits, so
# training several models or seeds in one run reads each file once.
CACHED_FILES = 32

# Name of a model followed by HAMMER on the Clifford circuits: the model name plus this.
THEN_HAMMER_SUFFIX = "_hammer"

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

    The circuit families of data.excluded_types are left out of the training and the
    validation circuits.

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
    train_sets = {
        size: make_dataset(_training_circuits(config, "train", size)) for size in qubit_counts
    }
    validation_sets = {
        size: make_dataset(_training_circuits(config, "validation", size)) for size in qubit_counts
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
    data.qubit_counts, since no such model exists. With evaluation.calibration_noise
    above 0 the evaluated circuits get a drifted calibration (see
    dequbit.dataset.perturb_calibration), seen by the models and the baselines alike;
    the baseline parameters are still fitted with the true calibration.

    Args:
        config: Full config.
        split: "validation" or "test".
        num_qubits: One of evaluation.qubit_counts.

    Returns:
        Metrics of the circuits with num_qubits qubits, with confidence intervals of the
        medians and the paired comparison of every model with the best baseline, and
        the L1 relative changes of every method by name, for figures.
    """
    train_circuits = load_split(config, "train", num_qubits)
    circuits = load_split(config, split, num_qubits)
    calibration_noise = config["evaluation"]["calibration_noise"]
    if calibration_noise > 0:
        rng = np.random.default_rng(config["seed"])
        circuits = perturb_calibration(circuits, calibration_noise, rng)
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

    model_names = evaluated_models(config, num_qubits)
    for model_name in model_names:
        logger.info("evaluating %s on %d qubits", model_name, num_qubits)
        metrics["models"][model_name], changes[model_name] = evaluate_model(
            config, model_name, circuits
        )
    metrics["circuit_types"] = _metrics_by_type(changes, circuits.circuit_types)
    metrics.update(_significance(config, changes, model_names))

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


def evaluate_clifford_size(
    config: Config, split: str, num_qubits: int
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate HAMMER and the size-independent models on Clifford circuits of one size.

    Only baselines that work without 2**num_qubits vectors take part: no correction, the
    oracle and HAMMER. Repolarizer and MIX subtract rate / 2**num_qubits from every
    bitstring, which changes nothing at these sizes, and readout inversion spreads every
    shot over 2**num_qubits bitstrings. Every model is also evaluated followed by HAMMER,
    as NAME_hammer: HAMMER applied to the shots the model corrected.

    Args:
        config: Full config.
        split: "validation" or "test".
        num_qubits: One of evaluation.clifford_qubit_counts.

    Returns:
        Metrics of the circuits, with the mean probability every method puts on the
        support of P_target, confidence intervals and paired comparisons, and the L1
        relative changes of every method by name.
    """
    path = Path(
        config["paths"]["clifford_data"].format(
            split=SPLIT_FILE_NAMES[split], num_qubits=num_qubits
        )
    )
    circuits = load_sparse_circuits(path)
    device = resolve_device(config["training"]["device"])
    rng = np.random.default_rng(config["seed"])

    baseline_masses = {
        "identity": [noisy_support_mass(circuit) for circuit in circuits],
        "oracle": [oracle_support_mass(circuit, rng) for circuit in circuits],
        "hammer": [hammer_support_mass(circuit, device) for circuit in circuits],
    }
    noisy_distances = _support_distances(baseline_masses["identity"], circuits)
    metrics: dict[str, Any] = {"num_circuits": len(circuits), "baselines": {}, "models": {}}
    changes = {}
    for name, masses in baseline_masses.items():
        changes[name] = _support_changes(masses, circuits, noisy_distances)
        metrics["baselines"][name] = _sparse_summary(changes[name], masses)

    model_names = [
        name for name in config["evaluation"]["models"] if name in SIZE_INDEPENDENT_MODELS
    ]
    for model_name in model_names:
        logger.info("evaluating %s on %d-qubit Clifford circuits", model_name, num_qubits)
        metrics["models"][model_name], changes[model_name] = _evaluate_sparse_model(
            config, model_name, circuits, noisy_distances, device
        )
        combined = f"{model_name}{THEN_HAMMER_SUFFIX}"
        metrics["models"][combined], changes[combined] = _evaluate_sparse_model_then_hammer(
            config, model_name, circuits, noisy_distances, device
        )
    types = np.array([circuit.circuit_type for circuit in circuits])
    metrics["circuit_types"] = _metrics_by_type(changes, types)
    metrics.update(_significance(config, changes, list(metrics["models"])))

    return metrics, changes


def evaluate_clifford_split(
    config: Config, split: str
) -> tuple[dict[str, Any], dict[int, dict[str, np.ndarray]]]:
    """Evaluate every size of evaluation.clifford_qubit_counts of one split.

    Args:
        config: Full config.
        split: "validation" or "test".

    Returns:
        All metrics, as written to paths.clifford_metrics, and the L1 relative changes
        of every method by number of qubits and method name.
    """
    metrics: dict[str, Any] = {
        "split": split,
        "environment": collect_environment(),
        "qubits": {},
    }
    changes = {}
    for num_qubits in config["evaluation"]["clifford_qubit_counts"]:
        metrics["qubits"][str(num_qubits)], changes[num_qubits] = evaluate_clifford_size(
            config, split, num_qubits
        )

    return metrics, changes


def _evaluate_sparse_model(
    config: Config,
    model_name: str,
    circuits: list[SparseCircuit],
    noisy_distances: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluate the saved models of one name on sparse circuits, as evaluate_model.

    Args:
        config: Full config.
        model_name: One of SIZE_INDEPENDENT_MODELS.
        circuits: Circuits to correct.
        noisy_distances: L1 distance of every noisy distribution, shape (circuits,).
        device: Device of the computation.

    Returns:
        Metrics per seed and mode with the mean and standard deviation of the median
        over the seeds in the configured mode, and the L1 relative changes of all seeds
        in that mode one after another, shape (seeds * circuits,).

    Raises:
        ValueError: If evaluation.inference is not a mode of SPARSE_INFERENCE_MODES.
        FileNotFoundError: If a model of some seed has not been trained.
    """
    mode = config["evaluation"]["inference"]
    if mode not in SPARSE_INFERENCE_MODES:
        raise ValueError(f"inference {mode!r} is not one of {SPARSE_INFERENCE_MODES}")
    radius = config["actions"]["radius"]

    per_seed: dict[str, dict[str, Any]] = {}
    mode_changes = []
    for seed in config["evaluation"]["seeds"]:
        path = model_path(config, model_name, seed, None)
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing")
        model = load_model(path).to(device)
        generator = torch.Generator(device=device).manual_seed(config["seed"])

        per_seed[str(seed)] = {}
        for inference in SPARSE_INFERENCE_MODES:
            masses = [
                corrected_support_mass(model, circuit, radius, inference, generator, device)
                for circuit in circuits
            ]
            seed_changes = _support_changes(masses, circuits, noisy_distances)
            per_seed[str(seed)][inference] = _sparse_summary(seed_changes, masses)
            if inference == mode:
                mode_changes.append(seed_changes)

    medians = [per_seed[seed][mode]["median"] for seed in per_seed]
    summary = {
        "inference": mode,
        "median_mean": float(np.mean(medians)),
        "median_std": float(np.std(medians)),
        "seeds": per_seed,
    }

    return summary, np.concatenate(mode_changes)


def _evaluate_sparse_model_then_hammer(
    config: Config,
    model_name: str,
    circuits: list[SparseCircuit],
    noisy_distances: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluate HAMMER applied to the shots corrected by the saved models of one name.

    Args:
        config: Full config.
        model_name: One of SIZE_INDEPENDENT_MODELS.
        circuits: Circuits to correct.
        noisy_distances: L1 distance of every noisy distribution, shape (circuits,).
        device: Device of the computation.

    Returns:
        Metrics per seed, with the mean and standard deviation of the median over the
        seeds, and the L1 relative changes of all seeds one after another, shape
        (seeds * circuits,).
    """
    radius = config["actions"]["radius"]

    per_seed: dict[str, dict[str, Any]] = {}
    seed_changes = []
    for seed in config["evaluation"]["seeds"]:
        model = load_model(model_path(config, model_name, seed, None)).to(device)
        generator = torch.Generator(device=device).manual_seed(config["seed"])
        masses = [
            hammer_support_mass(corrected_shots(model, circuit, radius, generator, device), device)
            for circuit in circuits
        ]
        seed_changes.append(_support_changes(masses, circuits, noisy_distances))
        per_seed[str(seed)] = {"sample": _sparse_summary(seed_changes[-1], masses)}

    medians = [per_seed[seed]["sample"]["median"] for seed in per_seed]
    summary = {
        "inference": "sample",
        "median_mean": float(np.mean(medians)),
        "median_std": float(np.std(medians)),
        "seeds": per_seed,
    }

    return summary, np.concatenate(seed_changes)


def _support_distances(masses: list[np.ndarray], circuits: list[SparseCircuit]) -> np.ndarray:
    """Compute the L1 distance to P_target of every circuit from its mass on the support.

    Args:
        masses: Probability on the support of P_target of every circuit.
        circuits: The circuits, in the same order.

    Returns:
        L1 distances, shape (circuits,).
    """
    return np.array(
        [
            l1_from_support_mass(mass, circuit.target_probs)
            for mass, circuit in zip(masses, circuits)
        ]
    )


def _support_changes(
    masses: list[np.ndarray], circuits: list[SparseCircuit], noisy_distances: np.ndarray
) -> np.ndarray:
    """Compute the L1 relative change of every circuit from its mass on the support.

    Args:
        masses: Probability of the corrected distribution on the support of every circuit.
        circuits: The circuits, in the same order.
        noisy_distances: L1 distance of every noisy distribution, shape (circuits,).

    Returns:
        L1 relative changes, shape (circuits,).
    """
    return (_support_distances(masses, circuits) - noisy_distances) / noisy_distances


def _sparse_summary(changes: np.ndarray, masses: list[np.ndarray]) -> dict[str, float]:
    """Summarize the relative changes and the mean probability on the support.

    Args:
        changes: L1 relative change of every circuit, shape (circuits,).
        masses: Probability on the support of P_target of every circuit.

    Returns:
        Median, quartiles, improved share and mean_support_probability.
    """
    return {
        **summarize_relative_changes(changes),
        "mean_support_probability": float(np.mean([mass.sum() for mass in masses])),
    }


def ablation_config(config: Config, name: str) -> Config:
    """Build the config of one variant of ablation.variants.

    Metrics and figures of the variant go to paths.ablations/NAME. A variant that trains
    its own models keeps them in paths.model/ablations/NAME and uses the seeds of
    ablation.seeds; any other variant evaluates the main models. The overrides of the
    variant are applied last.

    Args:
        config: Full config.
        name: Name of a variant in ablation.variants.

    Returns:
        Config of the variant; config itself is not changed.

    Raises:
        ValueError: If name is not a variant of ablation.variants.
    """
    variants = config["ablation"]["variants"]
    if name not in variants:
        raise ValueError(f"unknown ablation {name!r}, expected one of {sorted(variants)}")

    variant = variants[name]
    config = copy.deepcopy(config)
    directory = Path(config["paths"]["ablations"]) / name
    config["paths"]["metrics"] = str(directory / "metrics_{split}.json")
    config["paths"]["figures_dir"] = str(directory / "figures")
    if variant["train"]:
        config["paths"]["model"] = str(Path(config["paths"]["model"]) / "ablations" / name)
        config["evaluation"]["seeds"] = config["ablation"]["seeds"]

    return apply_overrides(config, variant["overrides"])


def _training_circuits(config: Config, split: str, num_qubits: int) -> Circuits:
    """Read the circuits a model is trained or validated on.

    Args:
        config: Full config.
        split: "train" or "validation".
        num_qubits: Number of qubits of the circuits.

    Returns:
        The circuits of the split without the families of data.excluded_types.
    """
    circuits = load_split(config, split, num_qubits)
    excluded = config["data"]["excluded_types"]

    return select_circuits(circuits, ~np.isin(circuits.circuit_types, excluded))


def _significance(
    config: Config, changes: dict[str, np.ndarray], model_names: list[str]
) -> dict[str, dict[str, Any]]:
    """Attach confidence intervals and paired tests to the medians of one number of qubits.

    Every model is compared with the baseline of the lowest median, other than the
    reference methods, on the same circuits; the model change of a circuit is its mean
    over the training seeds.

    Args:
        config: Full config.
        changes: L1 relative changes by method name: shape (circuits,) for a baseline,
            the seeds one after another, shape (seeds * circuits,), for a model.
        model_names: Names of the learned models among the methods.

    Returns:
        median_intervals, the confidence interval of the median of every method, and
        comparisons, the output of dequbit.metrics.paired_comparison for every model
        together with the name of the baseline it was compared with.
    """
    evaluation = config["evaluation"]
    resamples, level = evaluation["bootstrap_resamples"], evaluation["confidence_level"]
    rng = np.random.default_rng(config["seed"])
    num_circuits = len(changes[REFERENCE_METHODS[0]])
    by_seed = {name: values.reshape(-1, num_circuits) for name, values in changes.items()}

    intervals = {
        name: median_confidence_interval(values, rng, resamples, level)
        for name, values in by_seed.items()
    }
    baselines = [name for name in changes if name not in (*model_names, *REFERENCE_METHODS)]
    best = min(baselines, key=lambda name: np.median(changes[name]))
    comparisons = {
        name: {
            "baseline": best,
            **paired_comparison(by_seed[name].mean(axis=0), changes[best], rng, resamples, level),
        }
        for name in model_names
    }

    return {"median_intervals": intervals, "comparisons": comparisons}


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
