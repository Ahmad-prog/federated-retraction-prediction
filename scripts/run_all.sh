#!/usr/bin/env bash
# Regenerate every result in the paper from scratch.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.venvs/fedretract/bin/python}

$PY -m src.data.build_corpus
$PY -m src.features.build_features
$PY scripts/run_experiment.py experiments/main_publisher.yaml
$PY scripts/run_experiment.py experiments/main_field.yaml
$PY scripts/run_dp.py experiments/dp_publisher.yaml
$PY scripts/make_figures.py
