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

from dequbit.actions import build_masks, destinations, expected_distribution, hamming_weights
from dequbit.baselines import (
    best_parameter,
    depolarising_correction,
    fit_linear_map,
    hammer_correction,
    linear_map_correction,
    mix_correction,
    oracle_distribution,
    physical_depolarising_rate,
    readout_correction,
    scaled_mix_correction,
    threshold_correction,
)
from dequbit.config import Config
from dequbit.dataset import Circuits
from dequbit.metrics import (
    hellinger_distance,
    kl_divergence,
    l1_distance,
    l1_relative_change,
    summarize_relative_changes,
)
from dequbit.training import expected_flipped_bits, predict_distributions

INFERENCE_MODES = ("expected", "sample", "argmax")

# Pairs of a bitstring and an action corrected at once; bounds the memory of the model
# for many qubits.
EVALUATION_PAIRS_PER_BATCH = 1_000_000

# Names of the methods in figures.
METHOD_LABELS = {
    "identity": "No correction",
    "oracle": "Oracle (best possible)",
    "threshold": "Thresholding",
    "repolarizer": "Repolarizer",
    "readout": "Readout inversion",
    "mix": "MIX, fitted rate",
    "mix_physical": "MIX, physical rate",
    "mix_physical_fitted": "MIX, fitted physical rate",
    "linear_map": "Linear map",
    "hammer": "HAMMER",
    "mlp": "MLP",
    "shared_scorer": "Shared scorer",
    "per_qubit": "Per-qubit flip",
}
# Methods that only mark the scale: nothing corrected and the best possible correction.
REFERENCE_METHODS = ("identity", "oracle")
MODEL_COLOUR = "#90CAF9"
BASELINE_COLOUR = "#E0E0E0"
REFERENCE_COLOUR = "#FFFFFF"
# Line colours of the models in the training curves, in matplotlib's default cycle.
HISTORY_COLOURS = ("C0", "C1", "C2", "C3", "C4")
HISTORY_LOWEST_CHANGE = -0.8
HISTORY_HIGHEST_CHANGE = 0.1


def correct_circuits(
    model: nn.Module,
    counts: np.ndarray,
    qubit_features: np.ndarray,
    num_qubits: int,
    radius: int,
    mode: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Correct the shots of every circuit with the model.

    In mode "expected" every shot is split over the actions in proportion to P(a | x).
    In mode "sample" every shot gets one action drawn from P(a | x), and in mode
    "argmax" the most probable action of its bitstring.

    Args:
        model: Trained model that maps counts and qubit features to action logits.
        counts: Noisy shot counts, shape (circuits, 2**num_qubits).
        qubit_features: Description of every measured bit, from
            dequbit.dataset.Circuits.qubit_features, shape
            (circuits, num_qubits, qubit features).
        num_qubits: Number of measured qubits.
        radius: Largest number of bits one action flips.
        mode: One of INFERENCE_MODES.
        rng: Random generator of the drawn actions.

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

    batch_size = max(1, EVALUATION_PAIRS_PER_BATCH // destination_table.numel())

    model.eval()
    corrected, flips = [], []
    with torch.no_grad():
        for start in range(0, len(counts), batch_size):
            batch = torch.from_numpy(counts[start : start + batch_size]).float()
            batch_features = torch.from_numpy(qubit_features[start : start + batch_size])
            action_probs, _ = predict_distributions(model, batch, batch_features, destination_table)
            if mode == "sample":
                action_probs = _sampled_action_shares(batch, action_probs, rng)
            elif mode == "argmax":
                action_probs = _most_probable_action(action_probs)

            noisy = batch / batch.sum(dim=-1, keepdim=True)
            corrected.append(expected_distribution(noisy, action_probs, destination_table))
            flips.append(expected_flipped_bits(noisy, action_probs, flipped_bits))

    return torch.cat(corrected).numpy(), torch.cat(flips).numpy()


def baseline_distributions(
    train_circuits: Circuits, circuits: Circuits, baselines_config: Config, seed: int
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Correct the evaluated circuits with every baseline.

    Parameters of the threshold, the repolarizer and MIX are fitted on the training
    circuits. The readout correction and MIX use the calibrated readout errors of the
    device; physical MIX also takes its rate from the gates of every circuit and their
    calibrated errors, and fitted physical MIX scales that rate by a factor fitted on
    the training circuits.

    Args:
        train_circuits: Circuits the parameters are fitted on.
        circuits: Circuits to correct, with the same number of qubits.
        baselines_config: Baselines section of the config.
        seed: Seed of the oracle shots.

    Returns:
        Corrected distributions of every baseline by name, each of shape
        (circuits, 2**num_qubits), and the fitted parameters.
    """
    train_noisy, noisy = train_circuits.noisy, circuits.noisy
    threshold_grid = np.linspace(
        0, baselines_config["threshold_max"], baselines_config["threshold_steps"]
    )
    rate_grid = np.linspace(0, baselines_config["rate_max"], baselines_config["rate_steps"])
    scale_grid = np.linspace(0, baselines_config["scale_max"], baselines_config["scale_steps"])
    p01, p10 = _readout_rates(circuits)
    train_p01, train_p10 = _readout_rates(train_circuits)
    train_mix = partial(mix_correction, p01=train_p01, p10=train_p10)
    train_scaled_mix = partial(
        scaled_mix_correction,
        physical_rate=_physical_rate(train_circuits),
        p01=train_p01,
        p10=train_p10,
    )

    parameters = {
        "threshold": best_parameter(
            threshold_correction, threshold_grid, train_noisy, train_circuits.targets
        ),
        "repolarizer_rate": best_parameter(
            depolarising_correction, rate_grid, train_noisy, train_circuits.targets
        ),
        "mix_rate": best_parameter(train_mix, rate_grid, train_noisy, train_circuits.targets),
        "mix_physical_scale": best_parameter(
            train_scaled_mix, scale_grid, train_noisy, train_circuits.targets
        ),
    }
    physical_rate = _physical_rate(circuits)

    distributions = {
        "identity": noisy,
        "oracle": oracle_distribution(
            circuits.targets, circuits.num_shots, np.random.default_rng(seed)
        ),
        "threshold": threshold_correction(noisy, parameters["threshold"]),
        "repolarizer": depolarising_correction(noisy, parameters["repolarizer_rate"]),
        "readout": readout_correction(noisy, p01, p10),
        "hammer": hammer_correction(noisy),
        "mix": mix_correction(noisy, parameters["mix_rate"], p01, p10),
        "mix_physical": mix_correction(noisy, physical_rate, p01, p10),
        "mix_physical_fitted": scaled_mix_correction(
            noisy, parameters["mix_physical_scale"], physical_rate, p01, p10
        ),
    }
    if len(train_noisy) >= noisy.shape[1]:
        weights = fit_linear_map(train_noisy, train_circuits.targets)
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
        file.write("\n")


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


def plot_training_histories(
    histories: dict[str, list[list[dict[str, Any]]]], path: Path, num_qubits: int
) -> None:
    """Draw the validation median L1 relative change of every model during training.

    Every model gets one colour and one line per seed. A dot marks the validation with
    the lowest loss, whose weights the model keeps.

    Args:
        histories: Training histories of every seed, by model name, as returned by
            dequbit.training.train; every history has validated num_qubits qubits.
        path: Image file to write; missing parent folders are created.
        num_qubits: Number of qubits of the validation circuits drawn.
    """
    size = str(num_qubits)

    figure = Figure(figsize=(8, 4.5))
    axes = figure.add_subplot()
    for colour, (name, seeds) in zip(HISTORY_COLOURS, histories.items()):
        for seed_index, history in enumerate(seeds):
            steps = [record["step"] for record in history]
            changes = [record["validation_l1rc_by_qubits"][size] for record in history]
            best = min(history, key=lambda record: record["validation_loss"])
            label = METHOD_LABELS.get(name, name) if seed_index == 0 else None
            axes.plot(steps, changes, color=colour, linewidth=1, alpha=0.8, label=label)
            axes.plot(
                best["step"],
                best["validation_l1rc_by_qubits"][size],
                "o",
                color=colour,
                markersize=4,
            )

    # Every model starts from initial_logits far above zero; the interesting range is below.
    start = next(iter(histories.values()))[0][0]["validation_l1rc_by_qubits"][size]
    axes.set_ylim(HISTORY_LOWEST_CHANGE, HISTORY_HIGHEST_CHANGE)
    axes.axhline(0, color="grey", linestyle="--", linewidth=1)
    axes.set_xlabel("training step")
    axes.set_ylabel("median L1 relative change (validation)")
    axes.set_title(
        f"Training curves, {num_qubits} qubits; every model starts at {start:+.2f} (step 0)"
    )
    axes.grid(alpha=0.3)
    axes.legend()

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")


def plot_medians_by_qubits(
    medians: dict[str, dict[int, float]], path: Path, models: Sequence[str] = ()
) -> None:
    """Draw the median L1 relative change of every method against the number of qubits.

    Args:
        medians: Median L1 relative change by method name and number of qubits. A
            method may lack some numbers of qubits.
        path: Image file to write; missing parent folders are created.
        models: Names of the learned models among the methods, drawn with thick lines.
    """
    figure = Figure(figsize=(8, 4.5))
    axes = figure.add_subplot()
    for name, by_qubits in medians.items():
        qubit_counts = sorted(by_qubits)
        axes.plot(
            qubit_counts,
            [by_qubits[num_qubits] for num_qubits in qubit_counts],
            marker="o",
            markersize=4,
            linewidth=2.5 if name in models else 1,
            linestyle="--" if name in REFERENCE_METHODS else "-",
            label=METHOD_LABELS.get(name, name),
        )

    all_qubit_counts = sorted({size for by_qubits in medians.values() for size in by_qubits})
    axes.set_xticks(all_qubit_counts)
    axes.set_xlabel("number of qubits")
    axes.set_ylabel("median L1 relative change")
    axes.set_title("Median change of the L1 distance to P_target (lower is better)")
    axes.grid(alpha=0.3)
    axes.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")


def _sampled_action_shares(
    counts: torch.Tensor, action_probs: torch.Tensor, rng: np.random.Generator
) -> torch.Tensor:
    """Draw one action for every shot and return the share of shots that got each action.

    The shots of one bitstring are interchangeable, so their actions are drawn together
    from a multinomial distribution with P(a | x).

    Args:
        counts: Shot counts of every circuit, shape (circuits, 2**num_qubits).
        action_probs: P(a | x) for every bitstring x, shape
            (circuits, 2**num_qubits, actions).
        rng: Random generator of the draw.

    Returns:
        Share of the shots of every bitstring that got each action, same shape as
        action_probs; all zero for bitstrings without shots.
    """
    # The multinomial draw needs probabilities that sum to 1 in float64.
    probabilities = action_probs.double().numpy()
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    action_counts = rng.multinomial(counts.long().numpy(), probabilities)

    return torch.from_numpy(action_counts).float() / counts.clamp_min(1).unsqueeze(-1)


def _most_probable_action(action_probs: torch.Tensor) -> torch.Tensor:
    """Put all probability of every bitstring on its most probable action.

    Args:
        action_probs: P(a | x) for every bitstring x, shape
            (circuits, 2**num_qubits, actions).

    Returns:
        One-hot action probabilities, same shape.
    """
    best_action = action_probs.argmax(dim=-1, keepdim=True)

    return torch.zeros_like(action_probs).scatter_(-1, best_action, 1.0)


def _readout_rates(circuits: Circuits) -> tuple[np.ndarray, np.ndarray]:
    """Look up the readout errors of every bit of every circuit.

    Args:
        circuits: Circuits with the calibration of the device qubits.

    Returns:
        Probability of reading 1 for state 0 and of reading 0 for state 1 of the device
        qubit measured into every bit, both of shape (circuits, num_qubits).
    """
    p01 = circuits.calibration["readout_p01"][circuits.measured_qubits]
    p10 = circuits.calibration["readout_p10"][circuits.measured_qubits]

    return p01, p10


def _physical_rate(circuits: Circuits) -> np.ndarray:
    """Compute the physical depolarising rate of every circuit from its gates.

    Args:
        circuits: Circuits with gate counts and the calibration of the device qubits.

    Returns:
        Output of dequbit.baselines.physical_depolarising_rate, shape (circuits,).
    """
    return physical_depolarising_rate(
        circuits.one_qubit_gates,
        circuits.two_qubit_gates,
        circuits.calibration["error_1q"],
        circuits.calibration["error_2q"],
    )


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
