# Federated Retraction Prediction

Code for *"Publishers can screen for retractions together without sharing manuscripts: federated fine-tuning of
language models on FedRetract"* (M. Ahmad, M.T. Afzal; Information Processing & Management Conference 2026).

Publishers hold their articles (and, in deployment, their confidential submissions) in separate silos. We test
whether they can train a text-based retraction screener jointly with federated learning, without exchanging any
article, and which privacy guarantee such a consortium can afford.

Headline results (pooled test AUPRC, base rate 0.29; 3 seeds):

| Regime | ModernBERT-base | ModernBERT-large | Qwen3-1.7B |
|---|---|---|---|
| Central (data pooled) | 0.752 | 0.766 | 0.784 |
| FedAvg | 0.720 | 0.729 | 0.744 |
| FedAvg + local fine-tuning (each publisher scores its own articles) | 0.783 | 0.790 | 0.806 |
| ROC-AUC on 213 articles retracted after the snapshot (central) | 0.883 | 0.886 | 0.898 |

Metadata-only XGBoost: 0.512; topic-only model: 0.455. Every number in the paper is regenerated from the result files
by `scripts/v2/p1_paper_tables.py` (tables + `numbers.json`) and `scripts/v2/p2_paper_figures.py` (figures).

## Layout

```
src/
  data/ features/ models/ fl/ dp/ partition/   shared modules (corpus building, features, models, FL, DP)
  v2/            leakage cleaning rules (cleaning.py), Europe PMC client (epmc.py), FL utilities (fl.py)
scripts/v2/      the pipeline of the paper (one script per step, see below)
experiments/v2/  run scripts used on the GPU server and the exact command of every GPU job (jobs.txt)
scripts/, experiments/*.yaml   earlier metadata-only experiments (kept for reference)
```

## Setup

Python 3.12. The versions used for the paper are pinned in `requirements-v2.txt`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-v2.txt          # torch 2.11 + CUDA 12.8 for the GPU steps
export OPENALEX_MAILTO=you@example.org      # OpenAlex polite pool (optional but recommended)
```

## Reproducing the paper

All paths are relative to the repository root; data go to `data_v2/`, results to `results_v2/`.

| Step | Script(s) | Output | Hardware |
|---|---|---|---|
| 0. Base corpus: Retraction Watch positives, OpenAlex enrichment, publisher–year matched controls | `python -m src.data.build_corpus --stage positives / enrich / controls` (see `data/README.md`) | `data/processed/corpus*.parquet` | CPU, network |
| 1. Retraction Watch refresh, label sets, prospective positives | `a1_rw_refresh.py` | `data_v2/` | CPU, network |
| 2. Full-text subset: PMCID mapping, matched OA controls, XML download, parsing and cleaning | `a2_map_pmcid.py`, `a3_pmc_controls.py`, `a4_fetch_fulltext.py`, `a5_parse_clean.py`, `a6_ft_dataset.py` | `data_v2/ft_dataset.parquet` | CPU, network |
| 3. Prospective abstracts | `a7_prospective_abstracts.py` | `data_v2/prospective_abstracts.parquet` | CPU |
| 4. Leakage-clean tabular features | `b1_tabular_features.py` | `data_v2/tabular_v2.parquet` | CPU |
| 5. Leakage audit | `d1_leak_audit.py` | `results_v2/d1_leak_audit_*.txt` | CPU |
| 6. Tabular benchmark, ablations, consortium size and client DP | `b2_main_benchmark.py`, `b3_ablations.py`, `b4_scale_dp.py` | `results_v2/b2_main`, `b3_ablations`, `b4_scale_dp` | CPU (40 cores) |
| 7. Deployment (prevalence, country, reasons, prospective), cross-corpus, certainty | `b5_deployment.py`, `b6_cross_corpus.py`, `c5_certainty.py`, `b7_certainty_eval.py` | `results_v2/b5_*`, `b6_*`, `b7_*` | CPU / 1 GPU |
| 8. Frozen embeddings and heads | `c1_embed.py`, `c2_heads.py` | `results_v2/c2_heads` | 1 GPU |
| 9. LoRA fine-tuning (central, local, FedAvg, FedPer, Ditto, client DP, DP-SGD) | `c3_lora_fl.py` (commands in `experiments/v2/jobs.txt`) | `results_v2/c3_lora*` | 1 GPU per job, ≈63 GPU-h in total |
| 10. Text-model deployment and topic confound | `b10_text_deployment.py`, `b8_topic_confound.py` | `results_v2/b10_*`, `b8_*` | CPU |
| 11. Record-level DP heads, journal-level client DP, membership inference | `b11_dp_heads.py` (`--only record/journal/mia`) | `results_v2/b11_dp_heads` | CPU |
| 12. Robustness checks from the saved test scores: matched-input prospective test, within-journal ROC, publisher prior, paired bootstraps, formatting-only classifier, near-duplicates | `p3_review_checks.py` | `results_v2/p3_review` | CPU |
| 13. Second-round checks: paired bootstrap on the own-publisher metric, other-publisher reference rows, prospective AUPRC with year-matched controls, results without Hindawi, new-member (leave-one-publisher-out), temporal and journal-grouped splits on frozen embeddings | `p4_review_checks.py` | `results_v2/p4_review` | CPU |
| 14. Tables, numbers and figures of the paper | `p1_paper_tables.py`, `p2_paper_figures.py` | `paper/final/tables`, `numbers.json`, `figures` | CPU |

`experiments/v2/` contains the shell scripts that ran these steps (`run_phaseA.sh`, `data_chain.sh`,
`cpu_chain*.sh`) and the two GPU job queues (`gpu_queue.sh`, `gpu_queue0.sh`): each job in `jobs.txt` is one line
`<job id>: <command>`, run in id order. Seeds: 42–44 for language models, 42–51 for CPU experiments.

Reported numbers exclude smoke-test runs (`*_smoke`) and anything under `results_v2/_stale`.

**Results.** All result files of the paper (CSV/JSON/TXT, 4.4 MB) are in `results_v2/`, so
`p1_paper_tables.py` and `p2_paper_figures.py` run without re-training. Per-article test scores (`*.npz`) and
LoRA adapters (`*.pt`) are not included (size); the deployment, topic and membership-inference analyses and steps
12–13 need them (step 13 also needs the frozen abstract embeddings of step 8). Their outputs are included, so the
tables regenerate without them.

## Data

The FedRetract dataset (article identifiers, labels, cleaned text where licences allow, splits, embeddings and a
datasheet) will be released on Zenodo under the reserved DOI
[10.5281/zenodo.23227728](https://doi.org/10.5281/zenodo.23227728) (data CC BY 4.0, code MIT). The link resolves once
the record is published. Until then, every step above rebuilds it from the public
sources: the Retraction Watch database (distributed by Crossref), OpenAlex and the Europe PMC / PMC open-access subset.
`scripts/v2/r1_build_release.py` packages a rebuilt `data_v2/` into the release layout, and
`scripts/v2/r3_build_scidata_release.py` turns it into the licence-aware release: abstracts are shared only where the
article's licence allows it, with a SHA-256 checksum for the others. `scripts/v2/rebuild_abstracts.py` rebuilds those
abstracts from OpenAlex and checks them against the checksums, and `scripts/v2/rebuild_fulltext.py` re-downloads full
text whose licence does not allow redistribution.

The earlier metadata-only feature matrix in `data/processed/` belongs to the reference experiments in `scripts/`
and is not used by the pipeline above.

## Responsible use

The models rank articles for human review. Their false alarms are not evenly distributed (they are higher for
first authors from some countries and for paper-mill-prone topics; see the paper's appendix). They must not be used to
judge individual authors or articles without expert review.

## License

Code: MIT — see [`LICENSE`](LICENSE).
