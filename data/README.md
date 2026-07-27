# Data

## What is shipped

- `processed/features.parquet` — the model-ready feature matrix (metadata +
  abstract-linguistic features, one row per article, with silo and label
  columns). All experiments in `experiments/` run directly from this file.
- `processed/usman_clean_subset.csv` — the cleaned reproduction subset used for
  the cross-corpus transfer experiments against the Usman & Balke WebSci'25
  benchmark.

## What is NOT shipped (and why)

The full raw corpus (~1.8 GB) is not committed to keep the repository light and
to respect the terms of the upstream sources. This includes the Retraction
Watch dump, the per-DOI OpenAlex metadata cache, matched-control caches, and
retrieved full text. It is fully reproducible from the pipeline below.

## Rebuilding the corpus from source

All steps use public data and the polite-pool APIs (set your contact email in
`src/data/openalex.py`).

```bash
# 1. Retraction Watch positives (research articles with resolvable DOIs)
python -m src.data.build_corpus --stage positives

# 2. OpenAlex metadata enrichment (authorship, venue, citations, abstract)
python -m src.data.build_corpus --stage enrich

# 3. Publisher- and year-matched controls (3:1)
python -m src.data.build_corpus --stage controls

# 4. Feature extraction -> data/processed/features.parquet
python -m src.features.build_features
```

Sources:
- **Retraction Watch Database** — Center for Scientific Integrity, distributed
  openly via Crossref: http://retractiondatabase.org
- **OpenAlex** — https://openalex.org (API)
- **Europe PMC** — open-access full text (JATS XML) for the ~50K OA subset
