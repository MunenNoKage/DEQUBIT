# DEQUBIT

Sequence-to-sequence denoising of quantum measurement bitstrings for scalable error
mitigation.

![lint](../../actions/workflows/lint.yml/badge.svg)
![license](https://img.shields.io/badge/license-MIT-blue.svg)

Status: models, baselines, training and evaluation implemented; results on the
validation circuits of 3 to 13 qubits. The Streamlit demo is not implemented yet.

## Problem

Quantum computers return their results as measured bitstrings, and hardware noise
corrupts many of these measurements. Deep learning can turn a noisy output distribution
into the ideal one [1], but a distribution over n qubits has 2^n entries, so this does
not scale.

DEQUBIT corrects single bitstrings instead. The model reads one noisy measured
bitstring and chooses a small correction for it: keep it, or flip one or two bits. The
corrected shots of a circuit form the mitigated distribution. The number of possible
corrections grows with the square of the number of qubits instead of exponentially.

## Team

Mentor: Yuriy Pryyma.

| Name | GitHub | Role |
|---|---|---|
| Kateryna Suchok | [fro-og](https://github.com/fro-og) | Circuit and data generation, noise simulation, preprocessing, EDA |
| Roman Prokhorov | [MunenNoKage](https://github.com/MunenNoKage) | Models, action space, training, baselines, evaluation |

## Approach

```text
shots of a circuit -> shot counts -> model -> P(action | bitstring) for every measured bitstring
every shot moved by its action -> corrected distribution -> KL loss against P_target
```

- **Actions.** An action XORs a mask into the shot: keep it, flip one bit or flip two
  bits (radius 2: 16 actions for 5 qubits, 67 for 11).
- **Inputs.** A single shot cannot tell whether it is noise. The model therefore sees,
  for every bitstring, how often its neighbours x ^ m were measured compared with x
  itself (local counts), and three numbers on how spread the shots of the circuit are.
- **Circuit context.** The transpiled circuit is decoded into the errors that can hit
  every measured bit: the gate infidelity of its qubit (the sum of -log(1 - error)
  over every gate on it, so a qubit that few gates touched is unlikely to be wrong) and
  the calibrated readout errors of that qubit in both directions. All inputs grow with
  the number of qubits n, never with 2^n.
- **Models.** An MLP on the bits, the local counts, the context and the circuit
  context, trained for one number of qubits; and a shared scorer, one small network
  that scores every action from features that do not name a qubit (for example the
  gate infidelity of the flipped bits and the readout error the flip would undo), so
  one model is trained on circuits of every size and also corrects larger ones.
- **Training.** The action probabilities move the probability of every bitstring to
  its neighbours, which gives a predicted distribution. The loss compares it with the
  ideal distribution P_target of the circuit. Matching pairs of clean and noisy shots are
  not needed, because a single noisy shot has no single correct answer.

## Data

Simulated with Qiskit Aer by [`data2/generation.py`](data2/generation.py); the full
description, file layout and generation commands are in
[`data2/description.md`](data2/description.md).

- Noise of a real IBM device: `AerSimulator.from_backend(FakeSherbrooke)` with the
  density matrix method, from a calibration snapshot with per-qubit T1 and T2, gate
  errors and asymmetric readout errors.
- Circuits on nested connected subsets of the device. Five families: QFT, phase
  estimation (up to 6 qubits), hidden linear function, Bernstein-Vazirani and IQP, each
  followed by 1 to 4 layers of random rotations and CX chains, and transpiled to the
  device.
- 92 160 shots per circuit; P_target from the exact statevector.
- Stored with every circuit: the gates of the transpiled circuit on every qubit and
  pair of qubits, the device qubit measured into every bit, and the calibration.

| Qubits | Training | Validation | Test |
|---|---|---|---|
| 3, 5, 6 | 2500 | 200 | 200 |
| 8, 9, 11 | 2000 | 160 | 160 |
| 12 | 160 | 80 | 80 |
| 13 | 60 | 40 | 40 |

The models are trained on 3 to 11 qubits. The 12 and 13 qubit circuits are only
evaluated; their training circuits are used to fit the baseline parameters. The files
take 3.8 GB and are not in git. The 5-qubit dataset of the midterm is in
[`data/`](data/README.md).

## Evaluation

| Method | Kind |
|---|---|
| No correction | The noisy distribution itself |
| Oracle | Shots drawn from P_target: the best any correction with this many shots can do |
| Thresholding | Removes very small probabilities [1] |
| Repolarizer | Inverts uniform depolarising noise, rate fitted on the training circuits [1] |
| Readout inversion | Inverts the calibrated readout errors of every measured qubit, in both directions [1] |
| MIX | Readout inversion, then the repolarizer [1]. The rate is fitted on the training circuits, computed from the gates of every circuit and their calibrated errors (physical rate), or that physical rate scaled by a factor fitted on the training circuits |
| Linear map | Least-squares 2^n x 2^n inverse of the noise, learned from the training circuits |
| HAMMER | Raises the probability of measured bitstrings whose Hamming neighbours were also measured often; no fitting, no calibration [4] |
| MLP, shared scorer | Learned, per-bitstring corrections |
| Per-qubit flip model | Learned, the same action probabilities for every shot; a control of the midterm (0.001), not in the current evaluation |

Metrics, computed against P_target:

- L1 distance and KL divergence;
- **L1 relative change** (main metric): change of the L1 distance as a fraction of the
  noisy L1 distance; negative means the correction helped. Reported as the median and
  the interquartile range over circuits;
- share of circuits the correction improved;
- mean number of flipped bits per shot.

Uncertainty: a 95% bootstrap interval of every median (circuits resampled; for the
learned models the median is averaged over the seeds), and a paired comparison of
every learned model with the best baseline of each size on the same circuits: the
median of the per-circuit differences with its bootstrap interval, the share of circuits
where the model is better, and the p-value of the Wilcoxon signed-rank test.

## Results

Validation circuits (see Data). The learned models are the mean over 5 training seeds,
with one action sampled for every shot. Both models are trained on 3 to 11 qubits: the
MLP once per number of qubits, the shared scorer once for all of them. At 12 and 13
qubits the shared scorer corrects circuits larger than any it was trained on. The test
circuits are kept for the final results.

Median L1 relative change (lower is better; 0 is no change, -1 a perfect correction):

| Method | 3 | 5 | 6 | 8 | 9 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|
| Thresholding | -0.161 | -0.186 | +0.037 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Readout inversion | -0.157 | -0.076 | -0.055 | -0.072 | -0.053 | -0.047 | -0.030 | -0.022 |
| Repolarizer | -0.668 | -0.625 | -0.558 | -0.471 | -0.405 | -0.267 | -0.216 | -0.118 |
| MIX, fitted rate | -0.682 | -0.627 | -0.564 | -0.507 | -0.433 | -0.318 | -0.251 | -0.124 |
| MIX, physical rate | -0.636 | -0.623 | -0.599 | -0.484 | -0.430 | -0.284 | -0.231 | -0.089 |
| MIX, fitted physical rate | **-0.735** | -0.660 | -0.594 | -0.481 | -0.423 | -0.289 | -0.235 | -0.118 |
| Linear map | -0.608 | -0.483 | -0.388 | -0.326 | -0.224 | | | |
| HAMMER | +1.966 | -0.281 | -0.487 | -0.389 | -0.446 | -0.345 | -0.295 | **-0.192** |
| MLP | -0.725 | -0.654 | -0.622 | -0.544 | -0.501 | -0.382 | | |
| Shared scorer | -0.716 | **-0.668** | **-0.644** | **-0.596** | **-0.535** | **-0.411** | **-0.302** | -0.185 |
| Oracle | -0.952 | -0.965 | -0.961 | -0.930 | -0.920 | -0.867 | -0.833 | -0.779 |

Standard deviation of the median over seeds: at most 0.006, and 0.011 for the shared
scorer at 12 qubits. The physical MIX rates come from the gate counts of every circuit
and the calibrated gate errors; the fitted physical rate scales them by one factor
fitted on the training circuits. The linear map needs at least 2^n training circuits.
All numbers, per circuit family too:
[`results/metrics_validation.json`](results/metrics_validation.json), figures:
[`results/figures/`](results/figures/).

![Median L1 relative change by number of qubits](results/figures/l1_relative_change_by_qubits_validation.png)

- With a known noise model, physics is hard to beat on small circuits. At 3 qubits the
  shared scorer is level with MIX with the fitted physical rate (median paired
  difference +0.007, 95% interval [-0.004, 0.024], p = 0.13); at 5 qubits it is ahead by
  a small margin (-0.007, p = 0.002, better on 55% of the circuits).
- From 6 qubits on the shared scorer is ahead of every baseline: the median paired
  difference with the best baseline is -0.05 to -0.12 at 6 to 11 qubits (p < 1e-13), and
  the scorer is better on 74 to 88% of the circuits. The gain is largest on
  Bernstein-Vazirani circuits, whose ideal output is one bitstring (-0.716 against
  -0.422 for the best baseline at 11 qubits), and disappears on hidden linear function
  circuits, where HAMMER is level with it (-0.258 against -0.259 at 11 qubits).
- HAMMER, which needs neither calibration nor training, is the strongest baseline from
  9 qubits on. On small circuits it does harm (+1.97 at 3 qubits): multiplying every
  probability by its score sharpens the distribution, which suits a single correct
  answer but not a spread ideal distribution.
- Trained only on 3 to 11 qubits, the shared scorer corrects 12 qubit circuits better
  than every baseline (median paired difference with HAMMER -0.026, p < 1e-7; MIX
  reaches -0.251). At 13 qubits it is level with HAMMER (-0.185 against -0.192,
  p = 0.05); with expected instead of sampled corrections the scorer reaches -0.220.
- One shared scorer for every size beats an MLP trained for each size from 5 qubits on.
- It improves 97 to 99% of the circuits up to 12 qubits and 86% at 13; MIX with the
  fitted rate improves 82 to 96%.
- Expected corrections instead of sampled ones are equal up to 6 qubits and 0.035
  better at 13 qubits, where most bitstrings have only a few shots; with them the
  scorer is ahead of HAMMER at 13 qubits too (median paired difference -0.033,
  p < 1e-4; `expected_inference`). The most probable
  action for every shot is much worse (-0.03 to -0.27).

### Ablations

Every ablation is a variant of `ablation.variants` in the config, run with
`python -m dequbit ablate NAME`; metrics and figures are in
[`results/ablations/`](results/ablations/). Validation circuits; variants that train
their own models use seeds 0 to 2 and the numbers are sampled corrections, unless a
paragraph says otherwise.

**Circuit context** (`no_circuit_context`):

| Variant | 3 | 5 | 6 | 8 | 9 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|
| Shared scorer without circuit context | -0.667 | -0.629 | -0.559 | -0.567 | -0.502 | -0.355 | -0.291 | -0.184 |
| Shared scorer | -0.716 | -0.668 | -0.644 | -0.596 | -0.535 | -0.411 | -0.302 | -0.185 |
| MLP without circuit context | -0.688 | -0.625 | -0.561 | -0.522 | -0.482 | -0.368 | | |
| MLP | -0.725 | -0.654 | -0.622 | -0.544 | -0.501 | -0.382 | | |

The decoded circuit improves both models at every trained size, the shared scorer by
0.03 to 0.09 and the MLP by 0.01 to 0.06. Without it both models lose to MIX at 3 and 5
qubits: the calibration is what lets a learned model match the physics there.

**Sizes seen in training** (`trained_on_3_to_8_qubits`):

| Shared scorer trained on | 3 | 5 | 6 | 8 | 9 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|
| 3 to 8 qubits | -0.724 | -0.684 | -0.650 | -0.616 | -0.505 | -0.314 | -0.146 | +0.006 |
| 3 to 11 qubits | -0.716 | -0.668 | -0.644 | -0.596 | -0.535 | -0.411 | -0.302 | -0.185 |

Trained on fewer sizes, the scorer is better on those sizes, still beats every baseline
one step beyond them (9 qubits: -0.505 against -0.446 for HAMMER), and falls behind
further out (11 qubits: -0.314 against -0.345). Training on the largest affordable
circuits is what carries it to 12 and 13 qubits.

**Neighbourhood counts** (`with_neighbourhood`, 5 seeds): `use_neighbourhood` adds two
inputs to every pair of a bitstring and an action: how densely the bitstrings 1 and 2
flips away from the destination were measured, relative to those of the bitstring
itself. This is the observation behind HAMMER, as an input the model can weigh.

| Shared scorer | 3 | 5 | 6 | 8 | 9 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|
| Without neighbourhood counts | -0.716 | -0.668 | -0.644 | -0.596 | -0.535 | -0.411 | -0.302 | -0.185 |
| With neighbourhood counts | -0.720 | -0.683 | -0.649 | -0.611 | -0.544 | -0.435 | -0.324 | -0.229 |
| Best baseline | -0.735 | -0.660 | -0.599 | -0.507 | -0.446 | -0.345 | -0.295 | -0.192 |

The neighbourhood counts add 0.004 to 0.044, most at 11 to 13 qubits, and the scorer
then beats every baseline from 5 to 13 qubits (paired p < 0.01 at every size; at 13
qubits -0.021 against HAMMER). At 3 qubits MIX with the fitted physical rate stays ahead
(+0.011, p = 0.014).

**Unseen circuit families** (`without_qft`, `without_hlf`, `without_bv`, `without_iqp`,
seeds 0 and 1): the shared scorer trained without one family, evaluated on the circuits of that family
only, against the scorer trained on all families and the best baseline on that family.

| Family | Shared scorer | 3 | 5 | 6 | 8 | 9 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|
| QFT | without QFT | -0.692 | -0.664 | -0.628 | -0.395 | -0.538 | -0.432 | -0.351 | -0.269 |
| | all families | -0.685 | -0.672 | -0.708 | -0.688 | -0.692 | -0.579 | -0.480 | -0.377 |
| | best baseline | -0.643 | -0.680 | -0.626 | -0.450 | -0.533 | -0.455 | -0.412 | -0.340 |
| HLF | without HLF | -0.729 | -0.670 | -0.630 | -0.543 | -0.393 | -0.241 | -0.093 | -0.031 |
| | all families | -0.709 | -0.685 | -0.659 | -0.590 | -0.492 | -0.258 | -0.096 | -0.013 |
| | best baseline | -0.729 | -0.737 | -0.706 | -0.601 | -0.475 | -0.259 | -0.097 | -0.032 |
| BV | without BV | -0.589 | -0.790 | -0.705 | -0.662 | -0.635 | -0.549 | -0.516 | -0.493 |
| | all families | -0.765 | -0.806 | -0.755 | -0.751 | -0.754 | -0.716 | -0.680 | -0.628 |
| | best baseline | -0.778 | -0.809 | -0.759 | -0.674 | -0.529 | -0.422 | -0.419 | -0.523 |
| IQP | without IQP | -0.680 | -0.630 | -0.536 | -0.439 | -0.417 | -0.271 | -0.160 | -0.094 |
| | all families | -0.687 | -0.639 | -0.586 | -0.488 | -0.468 | -0.344 | -0.218 | -0.135 |
| | best baseline | -0.694 | -0.632 | -0.601 | -0.379 | -0.419 | -0.317 | -0.218 | -0.153 |

A family missing from training costs the scorer up to 0.29 on that family, and it falls
to about the level of the best baseline there; on Bernstein-Vazirani circuits it stays
ahead at 9 to 12 qubits. Part of what the scorer learns is specific to the circuit
families it was trained on, so the training set should cover the families it will
correct.

**Calibration drift** (`calibration_noise_20`, `calibration_noise_50`): every error rate
of the evaluated circuits is multiplied by a random factor exp(sigma * N(0, 1)) before
the models and the physical baselines read it, as if the calibration were taken hours
before the circuits ran. The models are not retrained.

| Method | sigma | 3 | 5 | 6 | 8 | 9 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|
| Shared scorer | 0 | -0.716 | -0.668 | -0.644 | -0.596 | -0.535 | -0.411 | -0.302 | -0.185 |
| | 0.2 | -0.702 | -0.672 | -0.643 | -0.593 | -0.540 | -0.412 | -0.301 | -0.186 |
| | 0.5 | -0.545 | -0.676 | -0.643 | -0.565 | -0.531 | -0.404 | -0.299 | -0.141 |
| MIX, fitted physical rate | 0 | -0.735 | -0.660 | -0.594 | -0.481 | -0.423 | -0.289 | -0.235 | -0.118 |
| | 0.2 | -0.698 | -0.665 | -0.600 | -0.488 | -0.431 | -0.286 | -0.234 | -0.117 |
| | 0.5 | -0.533 | -0.658 | -0.595 | -0.485 | -0.417 | -0.291 | -0.235 | -0.113 |

A drift of about 20% changes nothing beyond the seed spread from 5 qubits on. At 3
qubits, where readout errors are a large part of the noise, a drift of 50% costs both
the scorer and MIX about 0.2; from 5 qubits on the scorer loses at most 0.04.

**Number of shots** (`shots_1024`, `shots_8192`, `trained_on_1024_shots`,
`trained_on_thinned_shots`): circuits with fewer shots, from the first shots of every
circuit. The baselines are refitted on training circuits with the same number of shots.
`training.min_shots` thins every training batch to a random number of shots between
1024 and all of them, so one model learns every shot count in between. Expected
corrections, mean over seeds:

| Shots | Method | 3 | 5 | 6 | 8 | 9 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | Best baseline | -0.409 | -0.384 | -0.260 | -0.033 | -0.010 | 0.000 | 0.000 | -0.002 |
| | Shared scorer, trained on 92 160 shots | -0.245 | -0.243 | -0.210 | -0.041 | -0.021 | -0.126 | -0.254 | -0.364 |
| | Shared scorer, trained on 1024 shots | -0.404 | -0.397 | -0.278 | -0.172 | -0.135 | -0.319 | -0.380 | -0.378 |
| | Shared scorer, thinned shots | -0.401 | -0.380 | -0.284 | -0.155 | -0.115 | -0.312 | -0.371 | -0.397 |
| | Shared scorer, thinned shots and neighbourhood counts | -0.407 | -0.395 | -0.294 | -0.224 | -0.216 | -0.365 | -0.396 | -0.420 |
| | Oracle | -0.542 | -0.696 | -0.651 | -0.461 | -0.434 | -0.314 | -0.243 | -0.200 |
| 8192 | Best baseline | -0.658 | -0.599 | -0.533 | -0.330 | -0.238 | -0.075 | -0.012 | -0.005 |
| | Shared scorer, trained on 92 160 shots | -0.587 | -0.509 | -0.498 | -0.360 | -0.294 | -0.103 | -0.060 | -0.112 |
| | Shared scorer, thinned shots | -0.641 | -0.616 | -0.563 | -0.387 | -0.324 | -0.144 | -0.117 | -0.151 |
| | Shared scorer, thinned shots and neighbourhood counts | -0.643 | -0.623 | -0.570 | -0.415 | -0.337 | -0.235 | -0.235 | -0.292 |
| 92 160 | Best baseline | -0.735 | -0.660 | -0.599 | -0.507 | -0.446 | -0.345 | -0.295 | -0.192 |
| | Shared scorer | -0.719 | -0.668 | -0.646 | -0.601 | -0.542 | -0.428 | -0.317 | -0.220 |
| | Shared scorer, thinned shots | -0.710 | -0.673 | -0.638 | -0.587 | -0.532 | -0.413 | -0.303 | -0.210 |
| | Shared scorer, thinned shots and neighbourhood counts | -0.707 | -0.687 | -0.641 | -0.603 | -0.545 | -0.446 | -0.351 | -0.255 |

- A model trained on one number of shots does not transfer to far fewer: the local
  counts it reads change their scale.
- Trained on thinned shots, one model matches a model trained on 1024 shots and gives
  up 0.00 to 0.015 at 92 160 shots. It beats every baseline from 5 qubits on at 8192
  shots and from 6 qubits on at 1024 shots. With the neighbourhood counts as well
  (`with_neighbourhood_and_thinned_shots`, 5 seeds) one model is ahead of the best
  baseline from 5 qubits on at every number of shots; at 3 qubits it is behind by
  0.002 to 0.03.
- With few shots and many bitstrings no baseline helps at all, while the expected
  correction still reaches -0.3 to -0.4 at 11 to 13 qubits. It is better than the oracle
  there because the oracle is itself only 1024 shots of P_target, while the expected
  correction is a distribution, like the output of every baseline.
- Sampled corrections add the noise of drawing one action per shot, which dominates
  with few shots: at 1024 shots they are -0.37 to -0.22 at 3 to 6 qubits and +0.03 to
  +0.08 from 8 qubits on. With 92 160 shots the two modes differ by at most 0.04.

A learning rate of 3e-3, cosine decay, batches of 256 circuits, radius 3, the relative
loss, a wider network and a wider MLP changed the mean over sizes by less than the seed
spread (0.008) or made it worse in the hyperparameter search; training for 30 000
instead of 10 000 steps was the only clear gain. Applying the trained scorer twice
over-corrects (-0.44 becomes -0.01 at 11 qubits).

The 5-qubit results of the midterm, with the ablations of the local counts and the
radius, are in [`results/experiments.md`](results/experiments.md).

## Plan

1. **Done.** Fixed-depth Pauli gadget circuits on 5 qubits: baselines, the learned
   models, ablations and error analysis on the validation circuits.
2. **Done.** Circuits of five families transpiled to a simulated IBM device, with the
   decoded circuit and the calibration as model inputs.
3. **Done.** 3 to 11 qubits for training, 12 and 13 for the transfer of the shared
   scorer. 14 qubits need a 4.3 GB density matrix per simulation and were left out.
4. **Done.** HAMMER as a baseline, confidence intervals and paired tests, ablations of
   the inputs, the training sizes and the circuit families, robustness to calibration
   drift and to the number of shots.

The test circuits are evaluated once, for the final results.

## Quick start

```bash
git clone https://github.com/MunenNoKage/DEQUBIT.git && cd DEQUBIT
python -m venv .venv
.venv/Scripts/activate              # Windows; source .venv/bin/activate elsewhere
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
pre-commit install
```

`make setup` does the same in one step, if you have `make`.

## Usage

```bash
python -m dequbit train --all       # every model of evaluation.models, every seed
python -m dequbit train --model shared_scorer --seed 1
python -m dequbit evaluate          # models and baselines on the validation circuits
python -m dequbit ablate no_circuit_context   # one variant of ablation.variants
python -m dequbit evaluate --split test
python -m dequbit --set actions.radius=1 train --model mlp   # change one config value
make data                           # regenerate data_v4/: several hours, see data2/description.md
streamlit run app/streamlit_app.py  # interactive demo
```

All paths, hyperparameters and the seed are in [`config/default.yaml`](config/default.yaml).
`make reproduce` trains every model with every seed and evaluates them: about three
hours on a laptop GPU (RTX 4060), much longer on a CPU. `training.device: auto` uses a
GPU when PyTorch finds one; `requirements.txt` installs the CPU build.

In a notebook or a script the same steps are plain functions:

```python
from dequbit.config import load_config
from dequbit.pipeline import evaluate_split, load_split, train_model

config = load_config()
validation = load_split(config, "validation", num_qubits=5)  # counts, targets, metadata
model, history = train_model(config, "shared_scorer", seed=0, num_qubits=None)  # all sizes
metrics, changes = evaluate_split(config, "validation")
```

## Repository structure

```text
src/dequbit/
    config.py           loads config/default.yaml
    dataset.py          reads the .h5 files into shot counts, circuit metadata, qubit features
    features.py         model inputs: local counts, context, circuit context
    actions.py          action space, distributions built from corrected shots
    models/             mlp, shared_scorer, per_qubit; build, save, load
    baselines.py        oracle, thresholding, repolarizer, readout inversion, MIX, linear map, HAMMER
    training.py         loss, training loop, early stopping
    metrics.py          L1, KL, L1 relative change, improved share, flipped bits,
                        bootstrap intervals, paired comparison, Wilcoxon test
    evaluate.py         corrected circuits, metrics of every method, figures
    utils.py            seeding, logging, environment
    pipeline.py         load circuits, train a model, evaluate every method
    cli.py              python -m dequbit train | evaluate | ablate
data2/                  generation.py of the current dataset, description.md
data_v4/                generated .h5 files, not committed
data/                   the 5-qubit dataset of the midterm, data card
config/default.yaml     paths, hyperparameters, seed
tests/                  one test file per module
notebooks/              01-04, see notebooks/README.md
app/                    Streamlit demo
models/                 trained weights, not committed
results/                metrics_validation.json, metrics_test.json, figures/, experiments.md
docs/                   CONTRIBUTING.md, proposal/, poster/, slides/
```

## Reproducibility

- Seed 42 for the oracle and the sampled corrections; the learned models use the
  training seeds 0 to 4 (`evaluation.seeds`). The splits are separate files with their
  own generation seeds.
- `make reproduce` (or `python -m dequbit train --all` and `python -m dequbit evaluate`)
  rebuilds [`results/metrics_validation.json`](results/metrics_validation.json); running
  the evaluation twice gives identical numbers.
- The models above were trained with PyTorch 2.14.0 (CUDA 13.0) on an RTX 4060 laptop
  GPU and evaluated with Python 3.13.7, NumPy 2.5.3 and PyTorch 2.14.0 (CPU) on Windows
  11; the environment of the evaluation is also stored in every metrics file. Training
  on a CPU gives results that differ in the last digits.

## Development

```bash
flake8 src/ tests/ app/ && black --check src/ tests/ app/ && isort --check src/ tests/ app/
black src/ tests/ app/ && isort src/ tests/ app/       # auto-fix
pytest
```

Pre-commit hooks run the same checks before each commit, see
`.pre-commit-config.yaml`. Commit conventions are in
[`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md).

## AI usage

| Tool | Used for | How the result was checked |
|---|---|---|
| Claude Code | Writing code, tests, experiments and documentation; explaining methods | Every change reviewed step by step, unit tests, rerunning the evaluation, comparing results with the paper and with ablations |

## Contribution statement

| Member | Responsible for |
|---|---|
| Kateryna Suchok | Circuit generation, data generation, noise simulation, preprocessing, EDA and analysis notebooks, documentation |
| Roman Prokhorov | Model architecture design, action-space formulation, training pipeline, loss functions and regularisation, hyperparameter tuning, baselines, evaluation and metrics, documentation |

Both members agree with this division of work.

## References

1. L. Placidi et al. Deep learning approaches to quantum error mitigation.
   arXiv:2601.14226, 2026. <https://arxiv.org/abs/2601.14226>
2. H. Patil, D. Baron, H. Zhou. Q-Cluster: quantum error mitigation through
   noise-aware unsupervised learning. arXiv:2504.10801, 2025.
   <https://arxiv.org/abs/2504.10801>
3. H. Liao et al. Machine learning for practical quantum error mitigation. Nature
   Machine Intelligence, 2024. <https://arxiv.org/abs/2309.17368>
4. S. Tannu, P. Das, R. Ayanzadeh, M. Qureshi. HAMMER: boosting fidelity of noisy
   quantum circuits by exploiting Hamming behavior of erroneous outcomes. ASPLOS 2022.
   <https://arxiv.org/abs/2208.09371>

## License and citation

| | License | Notes |
|---|---|---|
| Code | MIT, see [`LICENSE`](LICENSE) | |
| Data | See [`data/README.md`](data/README.md) | Not covered by the code license |
| Trained weights | The data's terms and the code license | See [`models/README.md`](models/README.md) |
| Text, poster, slides | CC BY 4.0 | |

Citation metadata: [`CITATION.cff`](CITATION.cff).
