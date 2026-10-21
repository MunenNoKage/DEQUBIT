"""Command-line entry point: subcommands and argument parsing."""

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dequbit.config import DEFAULT_CONFIG_PATH, Config, load_config
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
    plot_l1_relative_change,
    write_json,
)
from dequbit.metrics import l1_relative_change
from dequbit.models import MODEL_NAMES, build_model, load_model, save_model
from dequbit.training import train
from dequbit.utils import collect_environment, set_seed, setup_logging

SPLITS = ("validation", "test")

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the parser with the train and evaluate subcommands and a --config option.

    Returns:
        Parser of the command-line arguments.
    """
    parser = argparse.ArgumentParser(
        prog="dequbit", description="Per-shot quantum error mitigation."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="YAML config file")
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="train a model and save it to paths.model")
    train_parser.add_argument(
        "--model", choices=MODEL_NAMES, help="model name (default: model.name)"
    )
    train_parser.add_argument(
        "--seed", type=int, help="seed of the weights and batches (default: seed)"
    )

    evaluate_parser = commands.add_parser("evaluate", help="evaluate models and baselines")
    evaluate_parser.add_argument(
        "--split", choices=SPLITS, help="circuits (default: evaluation.split)"
    )

    return parser


def model_path(config: Config, model_name: str, seed: int) -> Path:
    """Find the file of the model trained with one seed.

    Args:
        config: Full config.
        model_name: One of MODEL_NAMES.
        seed: Training seed of the model.

    Returns:
        Path of the saved model.
    """
    return Path(config["paths"]["model"]) / f"{model_name}_seed{seed}.pt"


def run_train(config: Config, model_name: str, seed: int) -> None:
    """Train one model on the training circuits and save it with its history.

    Args:
        config: Full config.
        model_name: One of MODEL_NAMES.
        seed: Seed of the initial weights and of the batch order.
    """
    num_qubits = config["data"]["num_qubits"]
    radius = config["actions"]["radius"]
    shots_per_circuit = config["data"]["shots_per_circuit"]
    shots, targets, train_index, validation_index = _training_file(config)

    set_seed(seed)
    model_config = {**config["model"], "name": model_name}
    model = build_model(model_config, num_qubits, radius)
    history = train(
        model,
        make_dataset(shots[train_index], targets[train_index], shots_per_circuit, num_qubits),
        make_dataset(
            shots[validation_index], targets[validation_index], shots_per_circuit, num_qubits
        ),
        num_qubits,
        radius,
        config["training"],
        seed,
    )

    path = model_path(config, model_name, seed)
    save_model(model, model_config, num_qubits, radius, path)
    write_json(history, path.with_name(f"{path.stem}_history.json"))
    logger.info("saved %s", path)


def run_evaluate(config: Config, split: str) -> None:
    """Evaluate the baselines and the trained models and write the metrics and a figure.

    The baselines are fitted on the training circuits. Every model is evaluated for every
    seed in evaluation.seeds and every inference mode.

    Args:
        config: Full config.
        split: "validation" for the held-out training circuits, "test" for the test file.
    """
    num_qubits = config["data"]["num_qubits"]
    shots_per_circuit = config["data"]["shots_per_circuit"]
    train_shots, train_targets, shots, targets, gate_counts, noise = _evaluation_data(config, split)
    train_noisy = shot_counts(train_shots, num_qubits) / shots_per_circuit
    noisy = shot_counts(shots, num_qubits) / shots_per_circuit

    baselines, parameters = baseline_distributions(
        train_noisy,
        train_targets,
        noisy,
        targets,
        shots_per_circuit,
        noise,
        gate_counts,
        config["baselines"],
        config["seed"],
    )
    metrics: dict[str, Any] = {
        "split": split,
        "num_circuits": len(shots),
        "environment": collect_environment(),
        "baseline_parameters": parameters,
        "baselines": {name: method_metrics(noisy, d, targets) for name, d in baselines.items()},
        "models": {},
    }
    changes = {name: l1_relative_change(noisy, d, targets) for name, d in baselines.items()}

    for model_name in config["evaluation"]["models"]:
        model_metrics, model_changes = _evaluate_model(config, model_name, shots, noisy, targets)
        metrics["models"][model_name] = model_metrics
        changes[model_name] = model_changes

    write_json(metrics, Path(config["paths"]["metrics"].format(split=split)))
    figure_path = Path(config["paths"]["figures_dir"]) / f"l1_relative_change_{split}.png"
    plot_l1_relative_change(changes, figure_path, config["evaluation"]["models"])
    logger.info("wrote metrics and %s", figure_path)


def main(argv: list[str] | None = None) -> int:
    """Run the subcommand given in argv.

    Args:
        argv: Command-line arguments without the program name; None reads sys.argv.

    Returns:
        Exit code, 0 on success.
    """
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    setup_logging(config["logging"]["level"])

    if args.command == "train":
        model_name = args.model or config["model"]["name"]
        seed = config["seed"] if args.seed is None else args.seed
        run_train(config, model_name, seed)
    else:
        run_evaluate(config, args.split or config["evaluation"]["split"])

    return 0


def _training_file(config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read the training file and split its circuits into training and validation.

    The split always uses the config seed, so every model seed sees the same circuits.

    Args:
        config: Full config.

    Returns:
        Shots, shape (circuits, shots_per_circuit), P_target, shape
        (circuits, 2**num_qubits), and the indices of the training and of the validation
        circuits.
    """
    paths = config["paths"]
    shots, targets = load_circuits(
        Path(paths["clean_train"]), Path(paths["noisy_train"]), config["data"]["num_qubits"]
    )
    train_index, validation_index = split_train_validation(
        len(shots), config["data"]["validation_fraction"], config["seed"]
    )

    return shots[:, : config["data"]["shots_per_circuit"]], targets, train_index, validation_index


def _evaluation_data(config: Config, split: str) -> tuple[Any, ...]:
    """Read the training circuits and the circuits to evaluate.

    Args:
        config: Full config.
        split: "validation" or "test".

    Returns:
        Shots and P_target of the training circuits; shots, P_target and two-qubit gate
        counts of the evaluated circuits; and the noise parameters of their file. Shots
        have shape (circuits, shots_per_circuit).
    """
    paths = config["paths"]
    shots, targets, train_index, validation_index = _training_file(config)
    if split == "validation":
        noisy_path = Path(paths["noisy_train"])
        gate_counts = load_two_qubit_gate_counts(Path(paths["clean_train"]))[validation_index]
        evaluated_shots, evaluated_targets = shots[validation_index], targets[validation_index]
    else:
        clean_path, noisy_path = Path(paths["clean_test"]), Path(paths["noisy_test"])
        evaluated_shots, evaluated_targets = load_circuits(
            clean_path, noisy_path, config["data"]["num_qubits"]
        )
        evaluated_shots = evaluated_shots[:, : config["data"]["shots_per_circuit"]]
        gate_counts = load_two_qubit_gate_counts(clean_path)

    noise = load_noise_parameters(noisy_path)

    return (
        shots[train_index],
        targets[train_index],
        evaluated_shots,
        evaluated_targets,
        gate_counts,
        noise,
    )


def _evaluate_model(
    config: Config, model_name: str, shots: np.ndarray, noisy: np.ndarray, targets: np.ndarray
) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluate the saved models of one name for every seed and inference mode.

    Args:
        config: Full config.
        model_name: One of MODEL_NAMES.
        shots: Noisy shots of the evaluated circuits, shape (circuits, shots).
        noisy: Noisy distributions of the same circuits, shape (circuits, 2**num_qubits).
        targets: P_target of the same circuits, same shape.

    Returns:
        Metrics per seed and mode with the mean and standard deviation of the median L1
        relative change over the seeds in the configured mode, and the L1 relative
        changes of all seeds in that mode together, shape (seeds * circuits,).

    Raises:
        FileNotFoundError: If a model of some seed has not been trained.
    """
    mode = config["evaluation"]["inference"]
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
                model,
                shots,
                config["data"]["num_qubits"],
                config["actions"]["radius"],
                inference,
                generator,
            )
            per_seed[str(seed)][inference] = {
                **method_metrics(noisy, corrected, targets),
                "mean_flipped_bits": float(flips.mean()),
            }
            if inference == mode:
                mode_changes.append(l1_relative_change(noisy, corrected, targets))

    medians = [per_seed[seed][mode]["median"] for seed in per_seed]
    summary = {
        "inference": mode,
        "median_mean": float(np.mean(medians)),
        "median_std": float(np.std(medians)),
        "seeds": per_seed,
    }

    return summary, np.concatenate(mode_changes)
