"""Tests for dequbit.models."""

import pytest

pytestmark = pytest.mark.skip(reason="TODO")


@pytest.mark.parametrize("name", ["mlp", "transformer", "perceiver", "per_qubit"])
def test_model_returns_logits_for_every_action(name: str) -> None:
    raise NotImplementedError


def test_per_qubit_model_ignores_input() -> None:
    raise NotImplementedError


def test_loaded_model_gives_same_output() -> None:
    raise NotImplementedError
