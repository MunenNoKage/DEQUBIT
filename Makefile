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

data: ## Regenerate the datasets in data/
	$(BIN)/python data/generate_data.py

reproduce: ## Train and evaluate on the datasets in data/
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
