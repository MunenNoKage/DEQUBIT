"""Seeding, logging setup and environment capture."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed the Python, NumPy and PyTorch random generators.

    Args:
        seed: Seed of all three generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_logging(level: str) -> None:
    """Configure the root logger with the level from the config."""
    raise NotImplementedError  # TODO


def collect_environment() -> dict[str, str]:
    """Python and library versions, OS and device used for the run."""
    raise NotImplementedError  # TODO
