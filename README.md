# Federated Retraction Prediction

**Cross-silo federated learning for scientific-integrity screening at the scale
of the scientific record.**

Retractions of scientific publications are growing rapidly, yet automated
screening is limited by two structural constraints: prior work relies on small
hand-curated corpora, and the strongest predictive signals (full text, citation
contexts, editorial metadata) are fragmented across publishers that cannot
legally pool their data. This project casts large-scale retraction prediction as
a **cross-silo federated learning** problem: publishers train a shared model on
their own private archives and exchange only model updates — no article ever
leaves its silo.

It extends the retraction-prediction line of Usman & Balke (TPDL'23/'24,
DASFAA'25, WebSci'25) to the federated, privacy-preserving, large-scale setting.

> Accompanying paper: *Federated Retraction Prediction: Scaling
> Scientific-Integrity Screening Across Publisher Silos* (under submission,
> IP&MC 2026). Manuscript in [`paper/ipm/`](paper/ipm/).

## Headline results

| Finding | Number |
|---|---|
| FedAvg vs. (unattainable) centralized model | **0.558 / 0.609 AUPRC — 92% recovered** |
| FedAvg vs. silo-local status quo | **0.558 / 0.315 — +77%** |
| Value of paywalled full text (abstract-only vs. full-text) | **0.88 → chance** |
| Corpus scale vs. prior retraction-prediction corpora | **138,537 articles — 180× larger** |
| Robustness to consortium size (K = 5→20 silos) | stable within noise |

Full experiment log: [`results/SUMMARY.md`](results/SUMMARY.md).

## Repository layout

```
src/
  data/        Retraction Watch loading + OpenAlex enrichment (openalex.py, build_corpus.py)
  features/    metadata / citation-dynamics / readability / certainty features
  partition/   non-IID silo partitioners (by publisher, by discipline)
  models/      LogReg / XGBoost / MLP baselines + evaluation
  fl/          federated engine (FedAvg, FedProx, imbalance-aware aggregation)
  dp/          differentially private aggregation
scripts/       entry points; run_all.sh regenerates every table and figure
experiments/   YAML configs for the main runs and sweeps
results/       all experiment outputs (CSV/JSON) + SUMMARY.md
paper/         the IP&M manuscript (paper/ipm/) and figures
data/          see data/README.md (raw corpus is rebuilt, not shipped)
```

## Quickstart

```bash
# 1. Environment (CPU-only research stack)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU torch

# 2. Reproduce the main federated benchmark from the shipped feature matrix
#    (data/processed/features.parquet — no corpus rebuild needed)
python scripts/run_experiment.py --config experiments/main_publisher.yaml

# 3. Regenerate all tables and figures
bash scripts/run_all.sh
```

To rebuild the full 138K-article corpus from source (Retraction Watch +
OpenAlex), see [`data/README.md`](data/README.md).

## Data

The model-ready feature matrix (`data/processed/features.parquet`) is included
so experiments reproduce without a corpus rebuild. The raw corpus (~1.8 GB of
Retraction Watch records, OpenAlex metadata caches, and full-text) is **not**
committed; [`data/README.md`](data/README.md) documents exactly how to
regenerate it from public sources.

## Citation

If you use this code or the benchmark, please cite the accompanying paper (see
[`CITATION.cff`](CITATION.cff)).

## License

Code released under the MIT License ([`LICENSE`](LICENSE)). Data are derived
from Retraction Watch (via Crossref), OpenAlex, and Europe PMC under their
respective terms.
