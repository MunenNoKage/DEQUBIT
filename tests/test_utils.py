"""Tests for dequbit.utils."""

import pytest
import torch

from dequbit.utils import resolve_device


def test_resolve_device_gives_named_device() -> None:
    """cpu gives the CPU; auto gives the GPU exactly when one is available."""
    assert resolve_device("cpu") == torch.device("cpu")
    assert (resolve_device("auto").type == "cuda") == torch.cuda.is_available()


def test_resolve_device_rejects_unknown_name() -> None:
    """A name outside DEVICE_NAMES raises ValueError."""
    with pytest.raises(ValueError):
        resolve_device("tpu")
