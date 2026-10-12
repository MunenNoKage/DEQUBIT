"""Command-line entry point: subcommands and argument parsing."""

import argparse

from dequbit.config import Config


def build_parser() -> argparse.ArgumentParser:
    """Parser with the ``train`` and ``evaluate`` subcommands and a ``--config`` option."""
    raise NotImplementedError  # TODO


def run_train(config: Config) -> None:
    """Train the model named in the config and save it to ``paths.model``."""
    raise NotImplementedError  # TODO


def run_evaluate(config: Config) -> None:
    """Evaluate the saved model and the baselines on the test circuits and write results/."""
    raise NotImplementedError  # TODO


def main(argv: list[str] | None = None) -> int:
    """Run the subcommand given in ``argv`` and return the exit code."""
    raise NotImplementedError  # TODO
