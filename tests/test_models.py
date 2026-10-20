"""Tests for dequbit.models."""

from pathlib import Path

import pytest
import torch

from dequbit.actions import build_masks, hamming_weights, initial_logits
from dequbit.config import Config
from dequbit.dataset import index_to_bits
from dequbit.models import MODEL_NAMES, build_model, load_model, save_model
from dequbit.models.mlp import MLPCorrector
from dequbit.models.per_qubit import PerQubitFlipModel
from dequbit.models.shared_scorer import SharedScorer

NUM_QUBITS = 3
RADIUS = 2
NUM_ACTIONS = 7
SEED = 0


def random_counts(num_circuits: int) -> torch.Tensor:
    """Draw random shot counts of every circuit.

    Args:
        num_circuits: Number of circuits.

    Returns:
        Counts from 0 to 99, shape (num_circuits, 2**NUM_QUBITS).
    """
    generator = torch.Generator().manual_seed(SEED)

    return torch.randint(0, 100, (num_circuits, 2**NUM_QUBITS), generator=generator).float()


def randomise_head(model: torch.nn.Module) -> None:
    """Replace the zero head weights by random ones, as after some training.

    Args:
        model: Model with a linear layer named head.
    """
    torch.manual_seed(SEED)
    torch.nn.init.normal_(model.head.weight)


def perturb_parameters(model: torch.nn.Module) -> None:
    """Add random noise to every parameter, as after some training.

    Args:
        model: Any model.
    """
    torch.manual_seed(SEED)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter))


def model_config(name: str) -> Config:
    """Build a small model section of the config.

    Args:
        name: Model name.

    Returns:
        Model config with the name and small hidden layers for every model.
    """
    return {"name": name, "mlp": {"hidden_sizes": [16]}, "shared_scorer": {"hidden_sizes": [16]}}


def test_per_qubit_model_gives_same_logits_for_every_bitstring() -> None:
    """B4 outputs initial_logits for every bitstring of every circuit, whatever the counts."""
    model = PerQubitFlipModel(NUM_QUBITS, RADIUS)

    logits = model(random_counts(4))

    assert logits.shape == (4, 2**NUM_QUBITS, NUM_ACTIONS)
    torch.testing.assert_close(logits, initial_logits(NUM_ACTIONS).expand_as(logits))


def test_mlp_starts_with_initial_logits() -> None:
    """Before training the MLP outputs initial_logits for every bitstring of every circuit."""
    model = MLPCorrector(NUM_QUBITS, RADIUS, hidden_sizes=[16, 16])

    logits = model(random_counts(4))

    assert logits.shape == (4, 2**NUM_QUBITS, NUM_ACTIONS)
    torch.testing.assert_close(logits, initial_logits(NUM_ACTIONS).expand_as(logits))


def test_mlp_output_depends_on_counts_once_head_is_trained() -> None:
    """With nonzero head weights, different counts give different logits."""
    model = MLPCorrector(NUM_QUBITS, RADIUS, hidden_sizes=[16])
    randomise_head(model)
    counts = random_counts(2)

    first = model(counts[:1])
    second = model(counts[1:])

    assert not torch.allclose(first, second)


def test_mlp_rejects_counts_of_wrong_length() -> None:
    """Counts over 16 bitstrings do not fit a 3-qubit model and raise ValueError."""
    model = MLPCorrector(NUM_QUBITS, RADIUS, hidden_sizes=[16])

    with pytest.raises(ValueError):
        model(torch.ones(1, 16))


def test_shared_scorer_starts_with_initial_logits() -> None:
    """Before training the shared scorer outputs initial_logits for every bitstring."""
    model = SharedScorer(NUM_QUBITS, RADIUS, hidden_sizes=[16, 16])

    logits = model(random_counts(4))

    assert logits.shape == (4, 2**NUM_QUBITS, NUM_ACTIONS)
    torch.testing.assert_close(logits, initial_logits(NUM_ACTIONS).expand_as(logits))


def test_shared_scorer_gives_same_logit_to_actions_with_same_features() -> None:
    """The scorer does not know qubit indices: symmetric flips get equal logits.

    The count of every bitstring depends only on its number of ones, so flipping
    qubit 0, 1 or 2 of bitstring 000 leads to bitstrings with the same count. All
    single flips then have the same features, and so have all pair flips.
    """
    model = SharedScorer(NUM_QUBITS, RADIUS, hidden_sizes=[16])
    randomise_head(model)
    ones = index_to_bits(torch.arange(2**NUM_QUBITS), NUM_QUBITS).sum(dim=-1)
    counts = (10.0 * 2.0**ones).unsqueeze(0)
    flipped_bits = hamming_weights(build_masks(NUM_QUBITS, RADIUS), NUM_QUBITS)

    logits = model(counts)[0, 0]

    for num_flipped in (1, 2):
        same_kind = logits[flipped_bits == num_flipped]
        torch.testing.assert_close(same_kind, same_kind[0].expand_as(same_kind))
    assert not torch.isclose(logits[flipped_bits == 1][0], logits[flipped_bits == 2][0])


def test_shared_scorer_weights_fit_another_number_of_qubits() -> None:
    """Weights trained for 3 qubits load into a 4-qubit scorer, which gives 11 logits."""
    model = SharedScorer(NUM_QUBITS, RADIUS, hidden_sizes=[16])
    randomise_head(model)
    larger_model = SharedScorer(NUM_QUBITS + 1, RADIUS, hidden_sizes=[16])
    larger_model.load_state_dict(model.state_dict())

    logits = larger_model(torch.ones(2, 2 ** (NUM_QUBITS + 1)))

    assert logits.shape == (2, 2 ** (NUM_QUBITS + 1), 11)


def test_shared_scorer_output_depends_on_counts_once_head_is_trained() -> None:
    """With nonzero head weights, different counts give different logits."""
    model = SharedScorer(NUM_QUBITS, RADIUS, hidden_sizes=[16])
    randomise_head(model)
    counts = random_counts(2)

    first = model(counts[:1])
    second = model(counts[1:])

    assert not torch.allclose(first, second)


def test_shared_scorer_rejects_counts_of_wrong_length() -> None:
    """Counts over 16 bitstrings do not fit a 3-qubit scorer and raise ValueError."""
    model = SharedScorer(NUM_QUBITS, RADIUS, hidden_sizes=[16])

    with pytest.raises(ValueError):
        model(torch.ones(1, 16))


@pytest.mark.parametrize(
    ("name", "model_class"),
    [("mlp", MLPCorrector), ("shared_scorer", SharedScorer), ("per_qubit", PerQubitFlipModel)],
)
def test_build_model_creates_named_model(name: str, model_class: type) -> None:
    """build_model creates the model named in the config, ready for counts of 3 qubits."""
    model = build_model(model_config(name), NUM_QUBITS, RADIUS)

    logits = model(random_counts(2))

    assert isinstance(model, model_class)
    assert logits.shape == (2, 2**NUM_QUBITS, NUM_ACTIONS)


def test_build_model_rejects_unknown_name() -> None:
    """A model name that is not in MODEL_NAMES raises ValueError."""
    with pytest.raises(ValueError):
        build_model(model_config("transformer"), NUM_QUBITS, RADIUS)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_saved_model_loads_with_same_logits(name: str, tmp_path: Path) -> None:
    """A model saved and loaded again gives the same logits as before saving.

    The weights are perturbed first, so the test fails if the trained weights are lost
    and the loaded model falls back to its initial logits.
    """
    config = model_config(name)
    model = build_model(config, NUM_QUBITS, RADIUS)
    perturb_parameters(model)
    path = tmp_path / "models" / "model.pt"

    save_model(model, config, NUM_QUBITS, RADIUS, path)
    loaded = load_model(path)

    counts = random_counts(2)
    assert type(loaded) is type(model)
    assert not torch.allclose(
        model(counts), initial_logits(NUM_ACTIONS).expand(2, 2**NUM_QUBITS, -1)
    )
    torch.testing.assert_close(loaded(counts), model(counts))
