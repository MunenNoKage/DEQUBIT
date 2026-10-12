"""Loading config/default.yaml."""

from pathlib import Path
from typing import Any

Config = dict[str, Any]

DEFAULT_CONFIG_PATH = Path("config/default.yaml")


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    """Read a YAML config file into a nested dictionary."""
    raise NotImplementedError  # TODO
