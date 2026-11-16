"""Seeding, logging setup and environment capture."""

import logging
import platform
import random

import numpy as np
import torch

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
DEVICE_NAMES = ("auto", "cpu", "cuda")


def set_seed(seed: int) -> None:
    """Seed the Python, NumPy and PyTorch random generators.

    Args:
        seed: Seed of all three generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str) -> torch.device:
    """Choose the device to train on.

    Args:
        name: One of DEVICE_NAMES; "auto" is the GPU when one is available and the CPU
            otherwise.

    Returns:
        The device.

    Raises:
        ValueError: If name is not one of DEVICE_NAMES.
    """
    if name not in DEVICE_NAMES:
        raise ValueError(f"unknown device {name!r}, expected one of {DEVICE_NAMES}")

    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"

    return torch.device(name)


def setup_logging(level: str) -> None:
    """Configure the root logger with the level from the config.

    Args:
        level: Name of the logging level, for example "INFO".
    """
    logging.basicConfig(level=level, format=LOG_FORMAT, force=True)


def collect_environment() -> dict[str, str]:
    """Describe the software and hardware of the run.

    Returns:
        Python, NumPy and PyTorch versions, the operating system and the device.
    """
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name()
    else:
        device = "cpu"

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": device,
    }
