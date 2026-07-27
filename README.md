# Federated Retraction Prediction — Experiments

Cross-silo federated learning experiments for predicting scientific article
retractions across publisher/discipline silos that cannot pool their data.
Publishers train a shared model on their own private archives and exchange only
model updates — no article leaves its silo.

## Layout

```
src/
  data/        Retraction Watch loading + OpenAlex enrichment
  features/    metadata / citation-dynamics / readability / certainty features
  partition/   non-IID silo partitioners (by publisher, by discipline)
  models/      LogReg / XGBoost / MLP baselines + evaluation
  fl/          federated engine (FedAvg, FedProx, imbalance-aware aggregation)
  dp/          differentially private aggregation
scripts/       experiment entry points; run_all.sh runs everything
experiments/   YAML configs for the main runs and sweeps
data/          sample / model-ready data (see data/README.md)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU torch
```

## Run

```bash
# Main federated benchmark from the shipped feature matrix (no corpus rebuild)
python scripts/run_experiment.py --config experiments/main_publisher.yaml

# Ablations, DP sweep, heterogeneity, reproduction
python scripts/run_ablations.py
python scripts/run_dp.py
bash scripts/run_all.sh
```

## Data

The model-ready feature matrix (`data/processed/features.parquet`) is included
so experiments run without a corpus rebuild. The full corpus (Retraction Watch
via Crossref + OpenAlex) is rebuilt from source per [`data/README.md`](data/README.md).

## License

MIT — see [`LICENSE`](LICENSE).
