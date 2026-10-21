"""Evaluation of the trained models and the baselines on held-out circuits."""

import json
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from torch import nn

from dequbit.actions import (
    apply_actions,
    build_masks,
    destinations,
    empirical_distribution,
    hamming_weights,
    sample_actions,
)
from dequbit.baselines import (
    best_parameter,
    depolarising_correction,
    fit_linear_map,
    linear_map_correction,
    mix_correction,
    oracle_distribution,
    physical_depolarising_rate,
    readout_correction,
    threshold_correction,
)
from dequbit.config import Config
from dequbit.dataset import shot_counts
from dequbit.metrics import (
    hellinger_distance,
    kl_divergence,
    l1_distance,
    l1_relative_change,
    summarize_relative_changes,
)
from dequbit.training import expected_flipped_bits, predict_distributions

INFERENCE_MODES = ("expected", "sample", "argmax")

# Circuits corrected at once; bounds the memory of the model inputs for many qubits.
EVALUATION_BATCH_SIZE = 64

# Names of the methods in figures.
METHOD_LABELS = {
    "identity": "No correction",
    "oracle": "Oracle (best possible)",
    "threshold": "Thresholding",
    "repolarizer": "Repolarizer",
    "readout": "Readout inversion",
    "mix": "MIX, fitted rate",
    "mix_physical": "MIX, physical rate",
    "linear_map": "Linear map",
    "mlp": "MLP",
    "shared_scorer": "Shared scorer",
    "per_qubit": "Per-qubit flip",
}
# Methods that only mark the scale: nothing corrected and the best possible correction.
REFERENCE_METHODS = ("identity", "oracle")
MODEL_COLOUR = "#90CAF9"
BASELINE_COLOUR = "#E0E0E0"
REFERENCE_COLOUR = "#FFFFFF"


def correct_circuits(
    model: nn.Module,
    shots: np.ndarray,
    num_qubits: int,
    radius: int,
    mode: str,
    generator: torch.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Correct the shots of every circuit with the model.

    In mode "expected" every shot is split over the actions in proportion to P(a | x).
    In mode "sample" every shot gets one action drawn from P(a | x), and in mode
    "argmax" the most probable action of its bitstring.

    Args:
        model: Trained model that maps counts to action logits.
        shots: Noisy shots as integers, shape (circuits, shots).
        num_qubits: Number of measured qubits.
        radius: Largest number of bits one action flips.
        mode: One of INFERENCE_MODES.
        generator: Random generator of the drawn actions.

    Returns:
        Corrected distribution of every circuit, shape (circuits, 2**num_qubits), and the
        mean number of flipped bits per shot of every circuit, shape (circuits,).

    Raises:
        ValueError: If mode is not one of INFERENCE_MODES.
    """
    if mode not in INFERENCE_MODES:
        raise ValueError(f"unknown inference mode {mode!r}, expected one of {INFERENCE_MODES}")

    masks = build_masks(num_qubits, radius)
    destination_table = destinations(masks, num_qubits)
    flipped_bits = hamming_weights(masks, num_qubits).float()

    model.eval()
    corrected, flips = [], []
    with torch.no_grad():
        for start in range(0, len(shots), EVALUATION_BATCH_SIZE):
            batch = torch.from_numpy(shots[start : start + EVALUATION_BATCH_SIZE])
            counts = torch.from_numpy(shot_counts(batch.numpy(), num_qubits)).float()
            action_probs, expected = predict_distributions(model, counts, destination_table)

            if mode == "expected":
                noisy = counts / counts.sum(dim=-1, keepdim=True)
                corrected.append(expected)
                flips.append(expected_flipped_bits(noisy, action_probs, flipped_bits))
                continue

            actions = _choose_actions(batch, action_probs, mode, generator)
            corrected.append(
                empirical_distribution(apply_actions(batch, actions, masks), num_qubits)
            )
            flips.append(flipped_bits[actions].mean(dim=-1))

    return torch.cat(corrected).numpy(), torch.cat(flips).numpy()


def baseline_distributions(
    train_noisy: np.ndarray,
    train_targets: np.ndarray,
    noisy: np.ndarray,
    targets: np.ndarray,
    num_shots: int,
    noise_parameters: dict[str, float],
    gate_counts: np.ndarray,
    baselines_config: Config,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Correct the evaluated circuits with every baseline.

    Parameters of the threshold, the repolarizer and MIX are fitted on the training
    circuits. The readout correction and MIX use the known readout flip probability
    p_ro; physical MIX also takes its rate from p2 and the gate count of every circuit.

    Args:
        train_noisy: Noisy distributions of the training circuits, shape
            (training circuits, 2**num_qubits).
        train_targets: P_target of the training circuits, same shape.
        noisy: Noisy distributions of the evaluated circuits, shape
            (circuits, 2**num_qubits).
        targets: P_target of the evaluated circuits, same shape; used only by the oracle.
        num_shots: Shots per circuit, also drawn by the oracle.
        noise_parameters: Noise model of the data, with "p_ro" and "p2".
        gate_counts: Two-qubit gates of every evaluated circuit, shape (circuits,).
        baselines_config: Baselines section of the config.
        seed: Seed of the oracle shots.

    Returns:
        Corrected distributions of every baseline by name, and the fitted parameters.
    """
    flip_probability = noise_parameters["p_ro"]
    threshold_grid = np.linspace(
        0, baselines_config["threshold_max"], baselines_config["threshold_steps"]
    )
    rate_grid = np.linspace(0, baselines_config["rate_max"], baselines_config["rate_steps"])
    mix_known_readout = partial(mix_correction, flip_probability=flip_probability)

    parameters = {
        "threshold": best_parameter(
            threshold_correction, threshold_grid, train_noisy, train_targets
        ),
        "repolarizer_rate": best_parameter(
            depolarising_correction, rate_grid, train_noisy, train_targets
        ),
        "mix_rate": best_parameter(mix_known_readout, rate_grid, train_noisy, train_targets),
    }

    distributions = {
        "identity": noisy,
        "oracle": oracle_distribution(targets, num_shots, np.random.default_rng(seed)),
        "threshold": threshold_correction(noisy, parameters["threshold"]),
        "repolarizer": depolarising_correction(noisy, parameters["repolarizer_rate"]),
        "readout": readout_correction(noisy, flip_probability),
        "mix": mix_known_readout(noisy, parameters["mix_rate"]),
        "mix_physical": _physical_mix(noisy, gate_counts, noise_parameters),
    }
    if len(train_noisy) >= noisy.shape[1]:
        weights = fit_linear_map(train_noisy, train_targets)
        distributions["linear_map"] = linear_map_correction(noisy, weights)

    return distributions, parameters


def method_metrics(
    noisy: np.ndarray, corrected: np.ndarray, targets: np.ndarray
) -> dict[str, float]:
    """Summarize how close the corrected distributions of one method are to the targets.

    Args:
        noisy: Noisy distributions of the evaluated circuits, shape
            (circuits, 2**num_qubits).
        corrected: The same distributions after the correction, same shape.
        targets: P_target of the evaluated circuits, same shape.

    Returns:
        Median and quartiles of the L1 relative change, the improved share, and the mean
        L1 distance, KL divergence and Hellinger distance to the targets.
    """
    relative_changes = l1_relative_change(noisy, corrected, targets)

    return {
        **summarize_relative_changes(relative_changes),
        "mean_l1": float(l1_distance(corrected, targets).mean()),
        "mean_kl": float(kl_divergence(targets, corrected).mean()),
        "mean_hellinger": float(hellinger_distance(corrected, targets).mean()),
    }


def write_json(content: dict[str, Any] | list[Any], path: Path) -> None:
    """Write metrics or a training history as indented JSON.

    Args:
        content: Data made of dictionaries, lists, strings and numbers.
        path: File to write; missing parent folders are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(content, file, indent=2)


def plot_l1_relative_change(
    changes: dict[str, np.ndarray], path: Path, models: Sequence[str] = ()
) -> None:
    """Draw the L1 relative change of every method with its median and quartiles written.

    Methods are sorted from the best median at the top to the worst at the bottom. A box
    spans the first to the third quartile, the whiskers 1.5 times that range, and the
    numbers on the right are the median and the quartiles.

    Args:
        changes: L1 relative change of every circuit, shape (circuits,), by method name.
        path: Image file to write; missing parent folders are created.
        models: Names of the learned models among the methods, drawn in another colour.
    """
    names = sorted(changes, key=lambda name: np.median(changes[name]), reverse=True)
    rows = range(1, len(names) + 1)

    figure = Figure(figsize=(10, 0.55 * len(names) + 1.8))
    axes = figure.add_subplot()
    boxes = axes.boxplot(
        [changes[name] for name in names],
        orientation="horizontal",
        showfliers=False,
        patch_artist=True,
        widths=0.6,
        medianprops={"color": "black", "linewidth": 2},
    )
    for name, box in zip(names, boxes["boxes"]):
        box.set_facecolor(_method_colour(name, models))
    axes.set_yticks(rows, [METHOD_LABELS.get(name, name) for name in names])

    # The numbers stand right of the plot, in data rows, so every method has its own line.
    axes.annotate("median  [Q1, Q3]", (1.02, len(names) + 0.75), xycoords=("axes fraction", "data"))
    for row, name in zip(rows, names):
        first_quartile, median, third_quartile = np.percentile(changes[name], [25, 50, 75])
        axes.annotate(
            f"{median:+.3f}  [{first_quartile:+.3f}, {third_quartile:+.3f}]",
            (1.02, row),
            xycoords=("axes fraction", "data"),
            va="center",
            family="monospace",
            fontweight="bold" if name in models else "normal",
        )

    axes.axvline(0, color="grey", linestyle="--", linewidth=1)
    axes.grid(axis="x", alpha=0.3)
    axes.set_ylim(0.4, len(names) + 0.6)
    axes.set_xlabel("L1 relative change: -1 perfect, 0 unchanged, above 0 worse")
    axes.set_title("Change of the L1 distance to P_target (lower is better)")
    axes.legend(
        handles=[
            Patch(facecolor=MODEL_COLOUR, edgecolor="black", label="learned model"),
            Patch(facecolor=BASELINE_COLOUR, edgecolor="black", label="baseline"),
            Patch(facecolor=REFERENCE_COLOUR, edgecolor="black", label="reference"),
        ],
        loc="lower left",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")


def _choose_actions(
    shots: torch.Tensor, action_probs: torch.Tensor, mode: str, generator: torch.Generator
) -> torch.Tensor:
    """Choose one action for every shot.

    Args:
        shots: Shots as integers, shape (circuits, shots).
        action_probs: P(a | x) for every bitstring x, shape
            (circuits, 2**num_qubits, actions).
        mode: "sample" to draw from P(a | x), "argmax" for the most probable action.
        generator: Random generator of the draw.

    Returns:
        Action index of every shot, shape (circuits, shots).
    """
    if mode == "sample":
        return sample_actions(shots, action_probs, generator)

    circuit_index = torch.arange(len(shots)).unsqueeze(-1)

    return action_probs.argmax(dim=-1)[circuit_index, shots]


def _physical_mix(
    noisy: np.ndarray, gate_counts: np.ndarray, noise_parameters: dict[str, float]
) -> np.ndarray:
    """Apply MIX with the depolarising rate given by the gate count of every circuit.

    Args:
        noisy: Noisy distributions, shape (circuits, 2**num_qubits).
        gate_counts: Two-qubit gates of every circuit, shape (circuits,).
        noise_parameters: Noise model of the data, with "p_ro" and "p2".

    Returns:
        Corrected distributions, same shape as noisy.
    """
    corrected = np.empty_like(noisy)
    for gate_count in np.unique(gate_counts):
        circuits = gate_counts == gate_count
        rate = physical_depolarising_rate(noise_parameters["p2"], int(gate_count))
        corrected[circuits] = mix_correction(noisy[circuits], rate, noise_parameters["p_ro"])

    return corrected


def _method_colour(name: str, models: Sequence[str]) -> str:
    """Pick the box colour of a method.

    Args:
        name: Method name.
        models: Names of the learned models.

    Returns:
        MODEL_COLOUR, REFERENCE_COLOUR or BASELINE_COLOUR.
    """
    if name in models:
        return MODEL_COLOUR
    if name in REFERENCE_METHODS:
        return REFERENCE_COLOUR

    return BASELINE_COLOUR
