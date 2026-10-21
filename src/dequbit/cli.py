"""Command-line entry point: subcommands and argument parsing."""

import argparse
import logging
from pathlib import Path

from dequbit.config import DEFAULT_CONFIG_PATH, Config, apply_overrides, load_config
from dequbit.evaluate import plot_l1_relative_change, plot_training_histories, write_json
from dequbit.models import MODEL_NAMES, save_model
from dequbit.pipeline import evaluate_split, load_history, model_path, train_model
from dequbit.utils import setup_logging

EVALUATION_SPLITS = ("validation", "test")

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the parser with the train and evaluate subcommands.

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

    return parser


def run_train(config: Config, model_name: str, seed: int) -> None:
    """Train one model and save it with its training history.

    Args:
        config: Full config.
        model_name: One of MODEL_NAMES.
        seed: Seed of the initial weights and of the batch order.
    """
    model, history = train_model(config, model_name, seed)

    path = model_path(config, model_name, seed)
    model_config = {**config["model"], "name": model_name}
    save_model(model, model_config, config["data"]["num_qubits"], config["actions"]["radius"], path)
    write_json(history, path.with_name(f"{path.stem}_history.json"))
    logger.info("saved %s", path)


def run_evaluate(config: Config, split: str) -> None:
    """Evaluate the baselines and the trained models and write the metrics and figures.

    Args:
        config: Full config.
        split: "validation" for the held-out training circuits, "test" for the test file.
    """
    metrics, changes = evaluate_split(config, split)
    write_json(metrics, Path(config["paths"]["metrics"].format(split=split)))

    figures = Path(config["paths"]["figures_dir"])
    models = config["evaluation"]["models"]
    plot_l1_relative_change(changes, figures / f"l1_relative_change_{split}.png", models)
    histories = {
        name: [load_history(config, name, seed) for seed in config["evaluation"]["seeds"]]
        for name in models
    }
    plot_training_histories(histories, figures / "training_curves.png")
    logger.info("wrote metrics and figures to %s", figures)


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
    elif args.all:
        for model_name in config["evaluation"]["models"]:
            for seed in config["evaluation"]["seeds"]:
                run_train(config, model_name, seed)
    else:
        model_name = args.model or config["model"]["name"]
        seed = config["seed"] if args.seed is None else args.seed
        run_train(config, model_name, seed)

    return 0
