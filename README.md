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
noisy shot (n bits) -> model -> probabilities of n + 1 actions
all shots of a circuit, corrected -> distribution over 2^n bitstrings -> loss against P_target
```

- **Actions.** Action 0 keeps the bitstring, action i + 1 flips bit i.
- **Models.** A model maps one shot to logits over the actions. Three architectures,
  adapted from [1]: an MLP-correction model, a Transformer and a Perceiver.
- **Training.** The corrected shots of a circuit, weighted by the action
  probabilities, give a predicted distribution, which the loss compares with the ideal
  distribution P_target of that circuit. Matching pairs of clean and noisy shots are not
  needed, because a single noisy shot has no single correct answer.

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
| Thresholding | Removes very small probabilities [1] |
| Global depolarising correction | Inverts uniform depolarising noise [1] |
| Per-qubit flip model | Learned, one flip probability per qubit, ignores the input bitstring |
| MLP, Transformer, Perceiver | Learned, per-bitstring corrections |

Metrics, computed against P_target on the test circuits:

- L1 distance and KL divergence;
- **L1 relative change** (main metric): change of the L1 distance as a fraction of the
  noisy L1 distance; negative means the correction helped;
- share of circuits the correction improved;
- mean number of flipped bits per shot.

## Results

**TODO**

| Method | L1 | KL | L1 relative change | Improved circuits | Flipped bits |
|---|---|---|---|---|---|
| No correction | | | | | |
| Thresholding | | | | | |
| Global depolarising | | | | | |
| Per-qubit flip | | | | | |
| MLP | | | | | |
| Transformer | | | | | |
| Perceiver | | | | | |

## Plan

1. Fixed-depth Pauli gadget circuits on 5 qubits: baselines and the three models.
2. Bricks stacked across 6 depth levels: robustness to noise strength.
3. Different numbers of qubits (5 to 12): whether correction quality holds as
   bitstrings grow.

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
python -m dequbit train             # train the model named in config/default.yaml
python -m dequbit evaluate          # models and baselines on the test circuits -> results/
python data/generate_data.py        # regenerate the datasets in data/
streamlit run app/streamlit_app.py  # interactive demo
```

All paths, hyperparameters and the seed are in [`config/default.yaml`](config/default.yaml).

## Repository structure

```text
src/dequbit/
    config.py           loads config/default.yaml
    dataset.py          reads the .h5 files, train/validation split, CircuitDataset
    actions.py          action space, distributions built from corrected shots
    models/             mlp, transformer, perceiver, per_qubit; build, save, load
    baselines.py        thresholding, global depolarising correction
    training.py         loss, training loop, early stopping
    metrics.py          L1, KL, L1 relative change, improved share, flipped bits
    evaluate.py         correction of test circuits, metrics.json, figures
    utils.py            seeding, logging, environment
    cli.py              python -m dequbit train | evaluate
data/                   generate_data.py, the .h5 datasets, data card
config/default.yaml     paths, hyperparameters, seed
tests/                  one test file per module
notebooks/              01-04, see notebooks/README.md
app/                    Streamlit demo
models/                 trained weights, not committed
results/                metrics.json, figures/, experiments.md
docs/                   CONTRIBUTING.md, proposal/, poster/, slides/
```

## Reproducibility

**TODO**: the seed, the library versions, the OS and the hardware the final numbers
were produced on.

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
