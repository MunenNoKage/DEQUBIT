# DEQUBIT

Sequence-to-sequence denoising of quantum measurement bitstrings for scalable error
mitigation.

![lint](../../actions/workflows/lint.yml/badge.svg)
![license](https://img.shields.io/badge/license-MIT-blue.svg)

Status: initial structure. Functions in `src/dequbit/` have their signatures and
docstrings and raise `NotImplementedError` until they are implemented.

## Problem

Quantum computers return their results as measured bitstrings, and hardware noise
corrupts many of these measurements. Deep learning can turn a noisy output distribution
into the ideal one [1], but a distribution over n qubits has 2^n entries, so this does
not scale.

DEQUBIT corrects single bitstrings instead. The model reads one noisy measured
bitstring and chooses a small correction for it: keep it or flip one bit. The corrected
shots of a circuit form the mitigated distribution. The number of possible corrections
grows linearly with the number of qubits instead of exponentially.

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
  bits (radius 2, 16 actions for 5 qubits).
- **Inputs.** A single shot cannot tell whether it is noise. The model therefore sees,
  for every bitstring, how often its neighbours x ^ m were measured compared with x
  itself (local counts), and three numbers on how spread the shots of the circuit are.
  All inputs grow with the number of qubits n, never with 2^n.
- **Models.** An MLP on the bits, the local counts and the context; a shared scorer, one
  small network that scores every action from features that do not depend on n, so the
  same weights work for any number of qubits; and a per-qubit flip model with the same
  action probabilities for every shot, which shows what is possible without the inputs.
- **Training.** The action probabilities move the probability of every bitstring to
  its neighbours, which gives a predicted distribution. The loss compares it with the
  ideal distribution P_target of the circuit. Matching pairs of clean and noisy shots are
  not needed, because a single noisy shot has no single correct answer.

## Data

Simulated with Qiskit Aer by [`data/generate_data.py`](data/generate_data.py):

- 5 qubits, circuits of 3 random Pauli gadget bricks;
- 2000 training and 200 test circuits, 10 000 shots per circuit;
- P_target from the exact statevector, noisy shots from a noise model with
  depolarising errors (p1 = 0.005, p2 = 0.02), thermal relaxation (T1 = 80 us,
  T2 = 60 us) and readout errors (p = 0.02).

File layout: [`data/data_description.md`](data/data_description.md). Data card:
[`data/README.md`](data/README.md).

## Evaluation

| Method | Kind |
|---|---|
| No correction | The noisy distribution itself |
| Oracle | Shots drawn from P_target: the best any correction with this many shots can do |
| Thresholding | Removes very small probabilities [1] |
| Repolarizer | Inverts uniform depolarising noise, rate fitted on the training circuits [1] |
| Readout inversion | Inverts known per-qubit readout errors [1] |
| MIX | Readout inversion, then the repolarizer; rate fitted, or from p2 and the CX count [1] |
| Linear map | Least-squares 2^n x 2^n inverse of the noise, learned from the training circuits |
| Per-qubit flip model | Learned, the same action probabilities for every shot |
| MLP, shared scorer | Learned, per-bitstring corrections |

Metrics, computed against P_target:

- L1 distance and KL divergence;
- **L1 relative change** (main metric): change of the L1 distance as a fraction of the
  noisy L1 distance; negative means the correction helped. Reported as the median and
  the interquartile range over circuits;
- share of circuits the correction improved;
- mean number of flipped bits per shot.

## Results

Validation circuits (400 held-out training circuits, 10 000 shots each). The learned
models are the mean over 5 training seeds, with one action sampled for every shot. The
test circuits are kept for the final results.

| Method | L1 | KL | L1 relative change, median [IQR] | Improved circuits | Flipped bits |
|---|---|---|---|---|---|
| No correction | 0.321 | 0.120 | 0.000 | 0% | 0 |
| Thresholding | 0.205 | 0.880 | -0.362 [-0.487, -0.196] | 93% | |
| Readout inversion | 0.238 | 0.078 | -0.256 [-0.283, -0.227] | 100% | |
| Repolarizer | 0.123 | 0.345 | -0.638 [-0.713, -0.529] | 99% | |
| MIX, physical rate | 0.102 | 0.133 | -0.681 [-0.734, -0.625] | 100% | |
| MIX, fitted rate | 0.100 | 0.249 | -0.693 [-0.772, -0.598] | 100% | |
| Linear map | 0.098 | 0.100 | -0.693 [-0.746, -0.641] | 100% | |
| Per-qubit flip | 0.321 | 0.120 | 0.001 [0.001, 0.002] | 12% | 0.001 |
| MLP | 0.096 | 0.026 | -0.694 [-0.757, -0.624] | 100% | 0.32 |
| Shared scorer | **0.088** | **0.024** | **-0.719** [-0.778, -0.651] | 100% | 0.28 |
| Oracle | 0.029 | 0.003 | -0.912 [-0.937, -0.872] | 100% | |

Standard deviation of the median over seeds: 0.003 (MLP), 0.0015 (shared scorer).
Expected instead of sampled corrections: -0.699 (MLP), -0.723 (shared scorer); the most
probable action for every shot: -0.478 and -0.458. All numbers:
[`results/metrics_validation.json`](results/metrics_validation.json), figures:
[`results/figures/`](results/figures/).

- The shared scorer improves every validation circuit (worst circuit -0.215), while MIX
  makes some circuits worse (up to +0.54). Its gain over MIX is largest on circuits with
  a spread P_target: -0.641 against -0.483 on the quarter with the highest entropy
  ([`notebooks/04-error-analysis.ipynb`](notebooks/04-error-analysis.ipynb)).
- The MLP stops early after about 2000 steps; the shared scorer still improves slowly at
  the limit of 10 000 steps ([`results/figures/training_curves.png`](results/figures/training_curves.png)).

**Ablations** (3 seeds, median L1 relative change, sampled corrections,
[`results/ablations/`](results/ablations/)):

| Variant | MLP | Shared scorer |
|---|---|---|
| Full model (radius 2, local counts, context) | -0.694 | -0.719 |
| Without local counts (bits and context only) | 0.000 | |
| Without context | -0.692 | |
| Radius 1 (single bit flips only) | -0.662 | -0.669 |

The local counts carry the result: from the bits of a single shot alone nothing is
learned. The context adds nothing at a single noise level, and pair flips add 0.03 to
0.05.

## Plan

1. **Done.** Fixed-depth Pauli gadget circuits on 5 qubits: baselines, the learned
   models, ablations and error analysis on the validation circuits.
2. Bricks stacked across 6 depth levels and several noise strengths: robustness to noise
   strength, with depth and noise strength as model inputs.
3. Different numbers of qubits (5 to 14): whether correction quality holds as bitstrings
   grow, and whether the shared scorer trained on one number of qubits works on another.

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
python -m dequbit evaluate --split test
python -m dequbit --set actions.radius=1 train --model mlp   # change one config value
python data/generate_data.py        # regenerate the datasets in data/
streamlit run app/streamlit_app.py  # interactive demo
```

All paths, hyperparameters and the seed are in [`config/default.yaml`](config/default.yaml).
`make reproduce` trains every model with every seed and evaluates them: about two hours
on a laptop CPU.

In a notebook or a script the same steps are plain functions:

```python
from dequbit.config import load_config
from dequbit.pipeline import evaluate_split, load_split, train_model

config = load_config()
validation = load_split(config, "validation")  # shots, targets, noisy, gate counts
model, history = train_model(config, "shared_scorer", seed=0)
metrics, changes = evaluate_split(config, "validation")
```

## Repository structure

```text
src/dequbit/
    config.py           loads config/default.yaml
    dataset.py          reads the .h5 files, train/validation split, CircuitDataset
    actions.py          action space, distributions built from corrected shots
    models/             mlp, shared_scorer, per_qubit; build, save, load
    baselines.py        oracle, thresholding, repolarizer, readout inversion, MIX, linear map
    training.py         loss, training loop, early stopping
    metrics.py          L1, KL, L1 relative change, improved share, flipped bits
    evaluate.py         corrected circuits, metrics of every method, figures
    utils.py            seeding, logging, environment
    pipeline.py         load circuits, train a model, evaluate every method
    cli.py              python -m dequbit train | evaluate
data/                   generate_data.py, the .h5 datasets, data card
config/default.yaml     paths, hyperparameters, seed
tests/                  one test file per module
notebooks/              01-04, see notebooks/README.md
app/                    Streamlit demo
models/                 trained weights, not committed
results/                metrics_validation.json, metrics_test.json, figures/, experiments.md
docs/                   CONTRIBUTING.md, proposal/, poster/, slides/
```

## Reproducibility

- Seed 42 for the train/validation split, the oracle and the sampled corrections; the
  learned models use the training seeds 0 to 4 (`evaluation.seeds`).
- `make reproduce` (or `python -m dequbit train --all` and `python -m dequbit evaluate`)
  rebuilds [`results/metrics_validation.json`](results/metrics_validation.json); running
  the evaluation twice gives identical numbers.
- The validation numbers above were produced with Python 3.13.7, NumPy 2.5.3 and PyTorch
  2.14.0 (CPU) on Windows 11; the environment is also stored in every metrics file.

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

**TODO**

| Tool | Used for | How the result was checked |
|---|---|---|
| | | |

## Contribution statement

| Member | Responsible for |
|---|---|
| Kateryna Suchok | Circuit generation, data generation, noise simulation, preprocessing, EDA, documentation |
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

## License and citation

| | License | Notes |
|---|---|---|
| Code | MIT, see [`LICENSE`](LICENSE) | |
| Data | See [`data/README.md`](data/README.md) | Not covered by the code license |
| Trained weights | The data's terms and the code license | See [`models/README.md`](models/README.md) |
| Text, poster, slides | CC BY 4.0 | |

Citation metadata: [`CITATION.cff`](CITATION.cff).
