"""Tests for dequbit.metrics."""

import math

import numpy as np
import pytest

from dequbit.actions import build_masks, destinations
from dequbit.metrics import (
    hellinger_distance,
    improved_fraction,
    kl_divergence,
    l1_distance,
    l1_relative_change,
    mean_flipped_bits,
    summarize_relative_changes,
    unreachable_mass,
)

NUM_QUBITS = 2


def test_distances_are_zero_for_equal_distributions() -> None:
    p = np.array([[0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.25, 0.25]])

    for distance in (l1_distance, kl_divergence, hellinger_distance):
        np.testing.assert_allclose(distance(p, p), [0.0, 0.0], atol=1e-12)


def test_distances_of_disjoint_distributions_are_maximal() -> None:
    p = np.array([[1.0, 0.0]])
    q = np.array([[0.0, 1.0]])

    assert l1_distance(p, q) == pytest.approx([2.0])
    assert hellinger_distance(p, q) == pytest.approx([1.0])
    assert np.isfinite(kl_divergence(p, q)).all()


def test_kl_divergence_skips_bitstrings_with_zero_probability() -> None:
    p = np.array([[0.5, 0.5, 0.0]])
    q = np.array([[0.25, 0.25, 0.5]])

    assert kl_divergence(p, q) == pytest.approx([math.log(2)])


def test_l1_relative_change_of_half_correction_identity_and_perfect() -> None:
    target = np.array([[1.0, 0.0]] * 3)
    noisy = np.array([[0.5, 0.5]] * 3)
    corrected = np.array([[0.75, 0.25], [0.5, 0.5], [1.0, 0.0]])

    changes = l1_relative_change(noisy, corrected, target)

    np.testing.assert_allclose(changes, [-0.5, 0.0, -1.0])


def test_l1_relative_change_rejects_noisy_equal_to_target() -> None:
    target = np.array([[1.0, 0.0]])

    with pytest.raises(ValueError):
        l1_relative_change(target, target, target)


def test_improved_fraction_counts_negative_changes() -> None:
    changes = np.array([-0.5, 0.1, -0.2, 0.0])

    assert improved_fraction(changes) == 0.5


def test_summarize_relative_changes_gives_median_quartiles_and_share() -> None:
    changes = np.array([-0.8, -0.6, -0.4, -0.2, 0.2])

    summary = summarize_relative_changes(changes)

    assert summary == pytest.approx(
        {
            "median": -0.4,
            "first_quartile": -0.6,
            "third_quartile": -0.2,
            "improved_fraction": 0.8,
        }
    )


def test_mean_flipped_bits_counts_changed_bits() -> None:
    noisy = np.array([0b000, 0b101, 0b111])
    corrected = np.array([0b011, 0b101, 0b110])

    assert mean_flipped_bits(noisy, corrected) == pytest.approx(1.0)


def test_unreachable_mass_is_target_mass_beyond_radius() -> None:
    table = destinations(build_masks(NUM_QUBITS, radius=1), NUM_QUBITS).numpy()
    counts = np.array([[10, 0, 0, 0], [10, 0, 0, 0]])
    targets = np.array([[0.0, 0.0, 0.0, 1.0], [0.5, 0.25, 0.0, 0.25]])

    mass = unreachable_mass(counts, targets, table)

    np.testing.assert_allclose(mass, [1.0, 0.25])
