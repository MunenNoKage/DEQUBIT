"""Loading config/default.yaml."""

from pathlib import Path
from typing import Any

import yaml

Config = dict[str, Any]

DEFAULT_CONFIG_PATH = Path("config/default.yaml")


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    """Read a YAML config file into a nested dictionary.

    Args:
        path: Config file.

    Returns:
        Sections of the config as nested dictionaries.
    """
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)
