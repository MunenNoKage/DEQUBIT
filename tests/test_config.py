"""Tests for dequbit.config."""

from pathlib import Path

from dequbit.config import load_config
from dequbit.evaluate import INFERENCE_MODES
from dequbit.models import MODEL_NAMES

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def test_default_config_loads_with_every_section() -> None:
    """The default config has the sections that train and evaluate read."""
    config = load_config(CONFIG_PATH)

    sections = {"paths", "data", "actions", "model", "training", "baselines", "evaluation"}
    assert sections <= set(config)
    assert isinstance(config["seed"], int)


def test_default_config_numbers_have_numeric_types() -> None:
    """Learning rate and weight decay are floats: YAML reads 1e-3 without a dot as text."""
    training = load_config(CONFIG_PATH)["training"]

    assert isinstance(training["learning_rate"], float)
    assert isinstance(training["weight_decay"], float)


def test_default_config_names_existing_models_and_modes() -> None:
    """Every model name and the inference mode in the config are ones the code knows."""
    config = load_config(CONFIG_PATH)

    assert config["model"]["name"] in MODEL_NAMES
    assert set(config["evaluation"]["models"]) <= set(MODEL_NAMES)
    assert config["evaluation"]["inference"] in INFERENCE_MODES
