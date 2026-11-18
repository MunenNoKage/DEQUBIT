"""Command-line entry point: subcommands and argument parsing."""

import argparse
import logging
from pathlib import Path

import numpy as np

from dequbit.config import DEFAULT_CONFIG_PATH, Config, apply_overrides, load_config
from dequbit.evaluate import (
    plot_l1_relative_change,
    plot_medians_by_qubits,
    plot_support_probability_by_qubits,
    plot_training_histories,
    write_json,
)
from dequbit.models import MODEL_NAMES, save_model
from dequbit.pipeline import (
    ablation_config,
    evaluate_clifford_split,
    evaluate_split,
    evaluated_models,
    load_history,
    model_path,
    model_sizes,
    train_model,
)
from dequbit.utils import setup_logging

EVALUATION_SPLITS = ("validation", "test")

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the parser with the train, evaluate, evaluate-clifford and ablate subcommands.

    Returns:
        Parser of the command-line arguments.
    """
    parser = argparse.ArgumentParser(
        prog="dequbit", description="Per-shot quantum error mitigation."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="YAML config file")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="replace one config value, for example --set actions.radius=1",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="train models and save them to paths.model")
    which = train_parser.add_mutually_exclusive_group()
    which.add_argument("--model", choices=MODEL_NAMES, help="model name (default: model.name)")
    which.add_argument(
        "--all",
        action="store_true",
        help="train every model of evaluation.models with every seed of evaluation.seeds",
    )
    train_parser.add_argument(
        "--seed", type=int, help="seed of the weights and batches (default: seed)"
    )

    evaluate_parser = commands.add_parser("evaluate", help="evaluate models and baselines")
    evaluate_parser.add_argument(
        "--split", choices=EVALUATION_SPLITS, help="circuits (default: evaluation.split)"
    )

    clifford_parser = commands.add_parser(
        "evaluate-clifford",
        help="evaluate the size-independent models and HAMMER on large Clifford circuits",
    )
    clifford_parser.add_argument(
        "--split", choices=EVALUATION_SPLITS, help="circuits (default: evaluation.split)"
    )

    ablate_parser = commands.add_parser(
        "ablate", help="train if needed and evaluate one variant of ablation.variants"
    )
    ablate_parser.add_argument("variant", help="name of a variant in ablation.variants")
    ablate_parser.add_argument(
        "--split", choices=EVALUATION_SPLITS, help="circuits (default: evaluation.split)"
    )

    return parser


def run_train(config: Config, model_name: str, seed: int) -> None:
    """Train one model and save it with its training history.

    The shared scorer is trained once on the circuits of every number of qubits; any
    other model is trained and saved once per number of data.qubit_counts.

    Args:
        config: Full config.
        model_name: One of MODEL_NAMES.
        seed: Seed of the initial weights and of the batch order.
    """
    model_config = {**config["model"], "name": model_name}

    for num_qubits in model_sizes(config, model_name):
        model, history = train_model(config, model_name, seed, num_qubits)

        path = model_path(config, model_name, seed, num_qubits)
        save_model(model, model_config, num_qubits, config["actions"]["radius"], path)
        write_json(history, path.with_name(f"{path.stem}_history.json"))
        logger.info("saved %s", path)


def run_evaluate(config: Config, split: str) -> None:
    """Evaluate the baselines and the trained models and write the metrics and figures.

    Args:
        config: Full config.
        split: "validation" for the validation files, "test" for the test files.
    """
    metrics, changes = evaluate_split(config, split)
    write_json(metrics, Path(config["paths"]["metrics"].format(split=split)))

    figures = Path(config["paths"]["figures_dir"])
    models = config["evaluation"]["models"]
    for num_qubits, size_changes in changes.items():
        plot_l1_relative_change(
            size_changes,
            figures / f"l1_relative_change_{split}_N{num_qubits}.png",
            evaluated_models(config, num_qubits),
        )
        # Training curves exist only for the numbers of qubits the models validated on.
        if models and num_qubits in config["data"]["qubit_counts"]:
            histories = {name: _histories_with_size(config, name, num_qubits) for name in models}
            plot_training_histories(
                histories, figures / f"training_curves_N{num_qubits}.png", num_qubits
            )

    medians = _medians_by_qubits(changes)
    plot_medians_by_qubits(medians, figures / f"l1_relative_change_by_qubits_{split}.png", models)
    logger.info("wrote metrics and figures to %s", figures)


def run_evaluate_clifford(config: Config, split: str) -> None:
    """Evaluate on the Clifford circuits and write the metrics and two figures over sizes.

    Args:
        config: Full config.
        split: "validation" or "test".
    """
    metrics, changes = evaluate_clifford_split(config, split)
    write_json(metrics, Path(config["paths"]["clifford_metrics"].format(split=split)))

    figures = Path(config["paths"]["figures_dir"])
    models = [name for size in metrics["qubits"].values() for name in size["models"]]
    plot_medians_by_qubits(
        _medians_by_qubits(changes), figures / f"l1_relative_change_clifford_{split}.png", models
    )
    plot_support_probability_by_qubits(
        support_probabilities_by_qubits(metrics),
        figures / f"support_probability_clifford_{split}.png",
        models,
    )
    logger.info("wrote Clifford metrics and figures to %s", figures)


def support_probabilities_by_qubits(metrics: dict) -> dict[str, dict[int, float]]:
    """Take the mean probability on the ideal outputs of every method from Clifford metrics.

    The oracle is left out, since its probability is 1. For a model it is the mean over
    the seeds in the mode the model was evaluated in.

    Args:
        metrics: Output of dequbit.pipeline.evaluate_clifford_split.

    Returns:
        Probabilities by method name and number of qubits.
    """
    probabilities: dict[str, dict[int, float]] = {}
    for size, size_metrics in metrics["qubits"].items():
        for name, values in size_metrics["baselines"].items():
            if name != "oracle":
                probabilities.setdefault(name, {})[int(size)] = values["mean_support_probability"]
        for name, values in size_metrics["models"].items():
            seeds = values["seeds"].values()
            mean = np.mean(
                [seed[values["inference"]]["mean_support_probability"] for seed in seeds]
            )
            probabilities.setdefault(name, {})[int(size)] = float(mean)

    return probabilities


def run_ablation(config: Config, name: str, split: str | None) -> None:
    """Train the models of one ablation variant if it has its own, then evaluate it.

    Args:
        config: Full config.
        name: Name of a variant in ablation.variants.
        split: "validation" or "test"; None takes evaluation.split.
    """
    variant_config = ablation_config(config, name)

    if config["ablation"]["variants"][name]["train"]:
        for model_name in variant_config["evaluation"]["models"]:
            for seed in variant_config["evaluation"]["seeds"]:
                run_train(variant_config, model_name, seed)
    run_evaluate(variant_config, split or variant_config["evaluation"]["split"])


def _histories_with_size(config: Config, model_name: str, num_qubits: int) -> list[list[dict]]:
    """Read the training histories of the models that validated one number of qubits.

    Args:
        config: Full config.
        model_name: One of MODEL_NAMES.
        num_qubits: One of data.qubit_counts.

    Returns:
        One history per seed of evaluation.seeds.
    """
    trained_size = None if None in model_sizes(config, model_name) else num_qubits

    return [
        load_history(config, model_name, seed, trained_size)
        for seed in config["evaluation"]["seeds"]
    ]


def _medians_by_qubits(changes: dict[int, dict[str, np.ndarray]]) -> dict[str, dict[int, float]]:
    """Take the median L1 relative change of every method for every number of qubits.

    Args:
        changes: L1 relative changes of the circuits by number of qubits and method name.

    Returns:
        Medians by method name and number of qubits.
    """
    medians: dict[str, dict[int, float]] = {}
    for num_qubits, size_changes in changes.items():
        for name, method_changes in size_changes.items():
            medians.setdefault(name, {})[num_qubits] = float(np.median(method_changes))

    return medians


def main(argv: list[str] | None = None) -> int:
    """Run the subcommand given in argv.

    Args:
        argv: Command-line arguments without the program name; None reads sys.argv.

    Returns:
        Exit code, 0 on success.
    """
    args = build_parser().parse_args(argv)
    config = apply_overrides(load_config(args.config), args.overrides)
    setup_logging(config["logging"]["level"])

    if args.command == "evaluate":
        run_evaluate(config, args.split or config["evaluation"]["split"])
    elif args.command == "evaluate-clifford":
        run_evaluate_clifford(config, args.split or config["evaluation"]["split"])
    elif args.command == "ablate":
        run_ablation(config, args.variant, args.split)
    elif args.all:
        for model_name in config["evaluation"]["models"]:
            for seed in config["evaluation"]["seeds"]:
                run_train(config, model_name, seed)
    else:
        model_name = args.model or config["model"]["name"]
        seed = config["seed"] if args.seed is None else args.seed
        run_train(config, model_name, seed)

    return 0
