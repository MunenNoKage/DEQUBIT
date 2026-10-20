"""Loss and training loop with validation and early stopping."""

import copy
import logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

from dequbit.actions import build_masks, destinations, expected_distribution, hamming_weights
from dequbit.config import Config
from dequbit.metrics import PROBABILITY_FLOOR, l1_relative_change

logger = logging.getLogger(__name__)


def circuit_kl(target: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    """Compute KL(target || predicted) of every circuit.

    Args:
        target: P_target of every circuit, shape (circuits, 2**num_qubits).
        predicted: Predicted distribution of every circuit, same shape.

    Returns:
        KL divergence of every circuit in nats, shape (circuits,).
    """
    # xlogy gives 0 where the target is 0, so bitstrings that never occur add nothing.
    log_loss = torch.special.xlogy(target, predicted.clamp_min(PROBABILITY_FLOOR))

    return (torch.special.xlogy(target, target) - log_loss).sum(dim=-1)


def expected_flipped_bits(
    noisy: torch.Tensor, action_probs: torch.Tensor, flipped_bits: torch.Tensor
) -> torch.Tensor:
    """Compute the mean number of bits the correction flips in one shot.

    Args:
        noisy: Noisy distribution of every circuit, shape (circuits, 2**num_qubits).
        action_probs: P(a | x) for every bitstring x, shape
            (circuits, 2**num_qubits, actions).
        flipped_bits: Number of bits every action flips, shape (actions,).

    Returns:
        Expected flipped bits per shot of every circuit, shape (circuits,).
    """
    flips_per_bitstring = (action_probs * flipped_bits).sum(dim=-1)

    return (noisy * flips_per_bitstring).sum(dim=-1)


def predict_distributions(
    model: nn.Module, counts: torch.Tensor, destination_table: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Correct every circuit with the expected action of every bitstring.

    Args:
        model: Model that maps counts to action logits.
        counts: Shot counts of every circuit, shape (circuits, 2**num_qubits).
        destination_table: Output of dequbit.actions.destinations, shape
            (2**num_qubits, actions).

    Returns:
        P(a | x) of every bitstring, shape (circuits, 2**num_qubits, actions), and the
        corrected distribution of every circuit, shape (circuits, 2**num_qubits).
    """
    noisy = counts / counts.sum(dim=-1, keepdim=True)
    action_probs = model(counts).softmax(dim=-1)

    return action_probs, expected_distribution(noisy, action_probs, destination_table)


def training_loss(
    model: nn.Module,
    counts: torch.Tensor,
    targets: torch.Tensor,
    destination_table: torch.Tensor,
    flipped_bits: torch.Tensor,
    training_config: Config,
) -> torch.Tensor:
    """Compute the loss of one batch of circuits.

    The loss of a circuit is KL(P_target || P_corrected), optionally divided by
    KL(P_target || P_noisy) so that every circuit weighs the same, plus flip_penalty
    times the expected flipped bits per shot.

    Args:
        model: Model that maps counts to action logits.
        counts: Shot counts of every circuit, shape (circuits, 2**num_qubits).
        targets: P_target of every circuit, same shape.
        destination_table: Output of dequbit.actions.destinations, shape
            (2**num_qubits, actions).
        flipped_bits: Number of bits every action flips, shape (actions,).
        training_config: Training section of the config.

    Returns:
        Mean loss over the circuits, a scalar tensor.
    """
    noisy = counts / counts.sum(dim=-1, keepdim=True)
    action_probs, corrected = predict_distributions(model, counts, destination_table)

    loss = circuit_kl(targets, corrected)
    if training_config["relative_loss"]:
        loss = loss / circuit_kl(targets, noisy).clamp_min(PROBABILITY_FLOOR)
    flips = expected_flipped_bits(noisy, action_probs, flipped_bits)

    return (loss + training_config["flip_penalty"] * flips).mean()


def train(
    model: nn.Module,
    train_set: TensorDataset,
    validation_set: TensorDataset,
    num_qubits: int,
    radius: int,
    training_config: Config,
    seed: int,
) -> list[dict[str, float]]:
    """Train the model with AdamW until the validation loss stops improving.

    Every step uses a random batch of training circuits. Every validation_interval steps
    the loss and the median L1 relative change on the validation circuits are recorded.
    Training stops after early_stopping_patience validations without a lower loss or
    after max_steps steps, and the model keeps the weights of the best validation.

    Args:
        model: Model built for num_qubits and radius; trained in place.
        train_set: Pairs (shot counts, P_target) of the training circuits.
        validation_set: Pairs (shot counts, P_target) of the validation circuits.
        num_qubits: Number of measured qubits.
        radius: Largest number of bits one action flips.
        training_config: Training section of the config.
        seed: Seed of the batch order.

    Returns:
        One record per validation with the step, the mean training loss since the last
        validation, the validation loss and the validation median L1 relative change.
    """
    device = next(model.parameters()).device
    masks = build_masks(num_qubits, radius)
    destination_table = destinations(masks, num_qubits).to(device)
    flipped_bits = hamming_weights(masks, num_qubits).float().to(device)
    train_counts, train_targets = (tensor.to(device) for tensor in train_set.tensors)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    generator = torch.Generator().manual_seed(seed)
    batch_size = min(training_config["circuits_per_batch"], len(train_counts))

    history = []
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale_validations = 0
    step = 0
    recent_losses = []
    while True:
        record = _validate(model, validation_set, destination_table, flipped_bits, training_config)
        history.append({"step": step, "train_loss": _mean_or_nan(recent_losses), **record})
        logger.info(
            "step %d: validation loss %.4f, median L1RC %.3f",
            step,
            record["validation_loss"],
            record["validation_l1rc"],
        )

        if record["validation_loss"] < best_loss:
            best_loss = record["validation_loss"]
            best_state = copy.deepcopy(model.state_dict())
            stale_validations = 0
        else:
            stale_validations += 1
        if stale_validations >= training_config["early_stopping_patience"]:
            break
        if step >= training_config["max_steps"]:
            break

        model.train()
        recent_losses = []
        num_steps = min(training_config["validation_interval"], training_config["max_steps"] - step)
        for _ in range(num_steps):
            batch = torch.randperm(len(train_counts), generator=generator)[:batch_size].to(device)
            loss = training_loss(
                model,
                train_counts[batch],
                train_targets[batch],
                destination_table,
                flipped_bits,
                training_config,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            recent_losses.append(loss.item())
        step += num_steps

    model.load_state_dict(best_state)

    return history


def _validate(
    model: nn.Module,
    validation_set: TensorDataset,
    destination_table: torch.Tensor,
    flipped_bits: torch.Tensor,
    training_config: Config,
) -> dict[str, float]:
    """Compute the loss and the median L1 relative change on the validation circuits.

    The circuits are processed in batches of circuits_per_batch, so memory does not grow
    with the number of validation circuits.

    Args:
        model: Model that maps counts to action logits.
        validation_set: Pairs (shot counts, P_target) of the validation circuits.
        destination_table: Output of dequbit.actions.destinations, shape
            (2**num_qubits, actions).
        flipped_bits: Number of bits every action flips, shape (actions,).
        training_config: Training section of the config.

    Returns:
        Validation loss (the training loss averaged over all validation circuits) and
        the median L1 relative change of the expected corrected distributions.
    """
    device = destination_table.device
    all_counts, all_targets = validation_set.tensors
    batch_size = training_config["circuits_per_batch"]

    model.eval()
    loss_sum = 0.0
    relative_changes = []
    with torch.no_grad():
        for start in range(0, len(all_counts), batch_size):
            counts = all_counts[start : start + batch_size].to(device)
            targets = all_targets[start : start + batch_size].to(device)
            loss = training_loss(
                model, counts, targets, destination_table, flipped_bits, training_config
            )
            loss_sum += loss.item() * len(counts)

            _, corrected = predict_distributions(model, counts, destination_table)
            noisy = counts / counts.sum(dim=-1, keepdim=True)
            relative_changes.append(
                l1_relative_change(
                    noisy.cpu().numpy(), corrected.cpu().numpy(), targets.cpu().numpy()
                )
            )

    return {
        "validation_loss": loss_sum / len(all_counts),
        "validation_l1rc": float(np.median(np.concatenate(relative_changes))),
    }


def _mean_or_nan(values: list[float]) -> float:
    """Average a list of numbers.

    Args:
        values: Numbers to average.

    Returns:
        Their mean, or NaN for an empty list.
    """
    if not values:
        return float("nan")

    return float(np.mean(values))
