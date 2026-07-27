"""Step 2: time-aware bibliographic-coupling features + full benchmark rerun.

Features per paper (all computed only from events BEFORE its publication year):
  n_refs_retracted_prior / frac_refs_retracted_prior:
      how many of its references were already-retracted papers
  n_refs_coupled_prior / frac_refs_coupled_prior:
      how many of its references are shared with the reference lists of
      already-retracted papers (bibliographic coupling with retracted work)

Feature sets compared: base -> +authorhist -> +coupling (centralized XGB and
federated FedAvg, 3 seeds).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.author_history import author_history_features
from src.features.build_features import NUMERIC_FEATURES
from src.fl.engine import Client, run_fl
from src.models.baselines import preprocessor, train_xgb
from src.partition.partitioner import partition
from scripts.run_author_history import author_lists_from_cache

SEEDS = [42, 43, 44]
REF_CACHE = ROOT / "data/cache/references"


def load_reference_lists() -> dict[str, list[str]]:
    refs = {}
    for f in REF_CACHE.glob("*.json"):
        for w in json.loads(f.read_text()):
            doi = (w.get("doi") or "").replace("https://doi.org/", "").lower()
            if doi:
                refs[doi] = w.get("referenced_works") or []
    return refs


def build_features(df: pd.DataFrame, refs: dict[str, list[str]]) -> pd.DataFrame:
    rw = pd.read_csv(ROOT / "data/raw/retraction_watch.csv", low_memory=False,
                     encoding_errors="replace")
    rw = rw[rw["RetractionNature"].str.strip().str.lower() == "retraction"]
    rw["rdoi"] = rw["OriginalPaperDOI"].astype(str).str.strip().str.lower()
    rw["ryear"] = pd.to_datetime(rw["RetractionDate"], errors="coerce").dt.year
    rw = rw.dropna(subset=["ryear"])
    doi2ryear = dict(zip(rw["rdoi"], rw["ryear"].astype(int)))

    # openalex id of each corpus paper (positives carry retraction year via doi)
    corpus = pd.read_parquet(ROOT / "data/processed/corpus.parquet")[
        ["doi", "openalex_id"]].dropna()
    doi2oa = dict(zip(corpus["doi"], corpus["openalex_id"]))
    oa_ryear = {doi2oa[d]: y for d, y in doi2ryear.items() if d in doi2oa}

    # first year each reference id becomes "coupled" (cited by a retracted paper)
    coupled_first_year: dict[str, int] = {}
    for d, y in doi2ryear.items():
        for r in refs.get(d, []):
            if r not in coupled_first_year or y < coupled_first_year[r]:
                coupled_first_year[r] = y

    n_ret, f_ret, n_cpl, f_cpl = [], [], [], []
    for doi, year in zip(df["doi"], df["year"]):
        rl = refs.get(doi, [])
        year = int(year) if pd.notna(year) else 0
        if not rl or not year:
            n_ret.append(0); f_ret.append(0.0); n_cpl.append(0); f_cpl.append(0.0)
            continue
        ret = sum(1 for r in rl if oa_ryear.get(r, 9999) < year)
        cpl = sum(1 for r in rl if coupled_first_year.get(r, 9999) < year)
        n_ret.append(ret); f_ret.append(ret / len(rl))
        n_cpl.append(cpl); f_cpl.append(cpl / len(rl))
    return pd.DataFrame({
        "n_refs_retracted_prior": np.array(n_ret, dtype=np.float32),
        "frac_refs_retracted_prior": np.array(f_ret, dtype=np.float32),
        "n_refs_coupled_prior": np.array(n_cpl, dtype=np.float32),
        "frac_refs_coupled_prior": np.array(f_cpl, dtype=np.float32),
    })


def main() -> None:
    feats = pd.read_parquet(ROOT / "data/processed/features.parquet")
    refs = load_reference_lists()
    print(f"reference lists loaded: {len(refs)} papers, "
          f"coverage {np.mean([d in refs for d in feats['doi']]):.3f}")

    au = author_lists_from_cache()
    df = feats.merge(au, on="doi", how="left")
    hist = author_history_features(df, df["authors"].fillna(""))
    cpl = build_features(df, refs)
    df = pd.concat([df.reset_index(drop=True), hist, cpl], axis=1)
    print(df.groupby("retracted")[["any_prior_author_retraction",
                                   "frac_refs_retracted_prior",
                                   "frac_refs_coupled_prior"]].mean().round(4).to_string())

    AH = ["n_prior_author_retractions", "any_prior_author_retraction"]
    CP = ["n_refs_retracted_prior", "frac_refs_retracted_prior",
          "n_refs_coupled_prior", "frac_refs_coupled_prior"]
    sets = [("base", NUMERIC_FEATURES),
            ("base+authorhist", NUMERIC_FEATURES + AH),
            ("base+authorhist+coupling", NUMERIC_FEATURES + AH + CP)]
    df = partition(df, by="publisher", k=10)
    df.to_parquet(ROOT / "data/processed/features_extended.parquet", index=False)

    results = []
    for seed in SEEDS:
        tr, te = train_test_split(df, test_size=0.2, stratify=df["retracted"],
                                  random_state=seed)
        for fname, fset in sets:
            Xtr = tr[fset].to_numpy(dtype=np.float32)
            ytr = tr["retracted"].to_numpy(dtype=np.float32)
            Xte = te[fset].to_numpy(dtype=np.float32)
            yte = te["retracted"].to_numpy(dtype=np.float32)
            m = train_xgb(Xtr, ytr, Xte, yte, seed)
            results.append({"seed": seed, "features": fname, "model": "central_xgb", **m})
            pre = preprocessor()
            Xtr_s = pre.fit_transform(Xtr).astype(np.float32)
            Xte_s = pre.transform(Xte).astype(np.float32)
            clients = [Client(str(s), Xtr_s[tr.index.get_indexer(p.index)],
                              ytr[tr.index.get_indexer(p.index)])
                       for s, p in tr.groupby("silo")
                       if p["retracted"].nunique() == 2]
            out = run_fl(clients, Xte_s, yte, strategy="fedavg", rounds=50,
                         seed=seed, eval_every=50)
            results.append({"seed": seed, "features": fname, "model": "fl_fedavg",
                            **out["final"]})
            print(results[-2]["features"], results[-2]["auprc"], "| fl:",
                  results[-1]["auprc"], flush=True)
    res = pd.DataFrame(results)
    res.to_csv(ROOT / "results/bib_coupling.csv", index=False)
    print(res.groupby(["features", "model"])[["auprc", "roc_auc", "recall_at_5fpr"]]
          .agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
