# Shortcuts for the commands that get run repeatedly. Anything that is a single
# command is left to be typed directly, so this file stays readable rather than
# becoming a second, undocumented CLI.
#
# `reproduce` trains every model with every seed, then evaluates them.

PYTHON ?= python3.13
VENV   := .venv

# venv puts executables in Scripts/ on Windows and bin/ everywhere else.
ifeq ($(OS),Windows_NT)
BIN := $(VENV)/Scripts
else
BIN := $(VENV)/bin
endif

.PHONY: setup data reproduce test lint format

setup: ## Create the venv, install runtime + dev deps and this package, install git hooks
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -r requirements.txt -r requirements-dev.txt
	$(BIN)/python -m pip install -e .
	$(BIN)/pre-commit install

data: ## Regenerate the dataset in data_v4/ (several hours, see data2/description.md)
	$(BIN)/python data2/generation.py --N 3 --splits val test train --val-count 40 --test-count 40 --train-scale 0.5813953488372093
	$(BIN)/python data2/generation.py --N 5 --splits val test train --val-count 40 --test-count 40 --train-scale 0.34965034965034963
	$(BIN)/python data2/generation.py --N 6 --splits val test train --val-count 40 --test-count 40 --train-scale 0.2915451895043732
	$(BIN)/python data2/generation.py --N 8 --splits val test train --val-count 40 --test-count 40 --train-scale 0.2188183807439825 --types qft hlf bv iqp
	$(BIN)/python data2/generation.py --N 9 --splits val test train --val-count 40 --test-count 40 --train-scale 0.19455252918287938 --types qft hlf bv iqp
	$(BIN)/python data2/generation.py --N 11 --splits val test train --val-count 40 --test-count 40 --train-scale 0.1592356687898089 --types qft hlf bv iqp
	$(BIN)/python data2/generation.py --N 12 --splits val train test --val-count 20 --test-count 20 --train-scale 0.0127389 --types qft hlf bv iqp
	$(BIN)/python data2/generation.py --N 13 --splits val train test --val-count 10 --test-count 10 --train-scale 0.00477707 --types qft hlf bv iqp

reproduce: ## Train and evaluate on the dataset in data_v4/
	$(BIN)/python -m dequbit train --all
	$(BIN)/python -m dequbit evaluate

test: ## Run the test suite
	$(BIN)/python -m pytest

lint: ## Exactly what CI checks
	$(BIN)/python -m flake8 src/ tests/ app/
	$(BIN)/python -m black --check src/ tests/ app/
	$(BIN)/python -m isort --check src/ tests/ app/

format: ## Fix formatting in place
	$(BIN)/python -m black src/ tests/ app/
	$(BIN)/python -m isort src/ tests/ app/
