# DEQUBIT

Bitstring-to-bitstring denoising with machine learning.

![lint](../../actions/workflows/lint.yml/badge.svg)
![license](https://img.shields.io/badge/license-MIT-blue.svg)

> [!note]
> Starting structure only — nothing is implemented yet. Every Python file contains a
> one-line docstring naming what belongs in it. Sections marked **TODO** are to be
> filled in as the work happens; delete this note once the project is underway.

## Problem

A supervised learning problem where both the input and the output are binary sequences
of the same length. Given a noisy bitstring such as `010101010`, the model should
predict the corresponding clean bitstring. A dataset consists of pairs of noisy and
clean binary sequences, and different ML architectures can be compared on how
accurately they reconstruct the original sequences. The main challenge is to design a
model that works efficiently for long bitstrings and generalises across different noise
levels and noise patterns.

The application comes from quantum error mitigation. Quantum computers produce their
results as many measured bitstrings, but hardware noise can corrupt these measurements.
Recent work uses deep learning to correct the entire probability distribution of
possible outcomes, but the size of this distribution grows exponentially with the
number of bits/qubits.

This project explores a potentially more scalable alternative: correct each measured
bitstring individually and reconstruct the improved distribution from the corrected
samples. Training data can initially be generated using noisy quantum simulators,
making it possible to create large synthetic datasets without access to real quantum
hardware.

Reference: <https://arxiv.org/abs/2601.14226>

## Team

**TODO**

| Name | GitHub | Role |
|---|---|---|
| | | |

## Approach

**TODO**

## Data

**TODO** — see [`data/README.md`](data/README.md) for the data card to fill in.

## Results

**TODO**

## Quick start

```bash
git clone <repo-url> && cd DEQUBIT
python -m venv .venv
.venv/Scripts/activate              # Windows; source .venv/bin/activate elsewhere
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
pre-commit install
```

`make setup` does the same in one step, if you have `make`.

## Repository structure

```
.
├── src/dequbit/          # the package
│   ├── cli.py              # command-line entry point
│   ├── config.py           # loads config/default.yaml
│   ├── data.py             # loading, splitting, persisting the dataset
│   ├── features.py         # feature preparation
│   ├── models.py           # model definitions and persistence
│   ├── evaluate.py         # metrics and figures
│   └── utils.py            # seeding, environment capture, logging
├── scripts/                # prepare_data.py, run_experiment.py
├── app/                    # streamlit_app.py — the interactive demo
├── notebooks/              # 01-04 (see notebooks/README.md)
├── tests/                  # test_data.py, test_models.py, test_reproducibility.py
├── config/default.yaml     # every hyperparameter, path, and the seed
├── examples/               # small sample inputs
├── data/                   # datasets NOT committed; data/README.md (data card) IS
├── models/                 # weights NOT committed; models/README.md IS
├── results/                # metrics + figures — committed
├── docs/                   # CONTRIBUTING.md, proposal/, poster/, slides/
└── .github/workflows/      # lint.yml
```

## Reproducibility

**TODO** — record the seed, the library versions, the OS, and the hardware the final
numbers were produced on.

## Development

```bash
flake8 src/ scripts/ tests/ app/ && black --check src/ scripts/ tests/ app/ && isort --check src/ scripts/ tests/ app/
black src/ scripts/ tests/ app/ && isort src/ scripts/ tests/ app/       # auto-fix
pytest
```

Pre-commit hooks catch most of this before a PR exists — see
`.pre-commit-config.yaml`. Branch policy, commit conventions, and working agreements
are in [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md).

This repo uses `pip` + `flake8` + `black` + `isort` as the lowest common denominator. A
`uv` + `ruff` setup is a fine substitute; what matters is that the environment is
reproducible and the formatting is enforced by CI rather than by agreement.

## AI usage

**TODO** — fill in as you go, not at the end.

| Tool | Used for | How the result was checked |
|---|---|---|
| | | |

## Contribution statement

**TODO**

| Member | Contribution | Responsible for |
|---|---|---|
| | | |

## License and citation

| | License | Notes |
|---|---|---|
| Code (this repo) | MIT — see [`LICENSE`](LICENSE) | Free to reuse, including commercially |
| Data | See [`data/README.md`](data/README.md) | **Not** covered by the code license |
| Trained weights | Inherit the data's terms + this repo's code license | See [`models/README.md`](models/README.md) |
| Text / poster / slides | CC BY 4.0 (recommended) | Attribute this repo if reused |

> [!important]
> An MIT `LICENSE` at the repo root covers the **code**, not data you collect or
> redistribute. Code and data are licensed independently — check both.

Citation metadata: [`CITATION.cff`](CITATION.cff) — fill in the author names and
repository URL, and GitHub renders a "Cite this repository" button from it.
