"""Loading config/default.yaml and replacing single values in it."""

import copy
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


def apply_overrides(config: Config, overrides: list[str]) -> Config:
    """Replace config values given as KEY=VALUE, where KEY is a dotted path.

    The value is read as YAML, so "actions.radius=1" gives the integer 1 and
    "evaluation.seeds=[0, 1]" a list. Only existing keys can be replaced, so a typo
    raises instead of adding an unused key.

    Args:
        config: Config to start from; it is not changed.
        overrides: Replacements such as "training.learning_rate=3.0e-3".

    Returns:
        A copy of the config with the replaced values.

    Raises:
        ValueError: If an override has no "=" or its key is not in the config.
    """
    config = copy.deepcopy(config)

    for override in overrides:
        key, separator, value = override.partition("=")
        if not separator:
            raise ValueError(f"override {override!r} is not KEY=VALUE")

        *parents, name = key.split(".")
        section = config
        for parent in parents:
            section = section.get(parent) if isinstance(section, dict) else None
        if not isinstance(section, dict) or name not in section:
            raise ValueError(f"config has no key {key!r}")

        section[name] = yaml.safe_load(value)

    return config
