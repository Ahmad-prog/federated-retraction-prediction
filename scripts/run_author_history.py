"""Step 1 of the 0.9 plan: add time-aware author retraction-history features
and measure the lift (centralized XGB + federated FedAvg)."""
from __future__ import annotations

import json
import sys
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
from src.models.evaluate import evaluate
from src.partition.partitioner import partition

SEEDS = [42, 43, 44]


def author_lists_from_cache() -> pd.DataFrame:
    rows = []
    for sub in ["positives", "controls_pub"]:
        for f in (ROOT / "data/cache" / sub).glob("*.json"):
            for w in json.loads(f.read_text()):
                doi = (w.get("doi") or "").replace("https://doi.org/", "").lower()
                if not doi:
                    continue
                names = "; ".join(
                    ((a.get("author") or {}).get("display_name") or "")
                    for a in (w.get("authorships") or []))
                rows.append({"doi": doi, "authors": names,
                             "pub_date": w.get("publication_date")})
    return pd.DataFrame(rows).drop_duplicates("doi")


def main() -> None:
    feats = pd.read_parquet(ROOT / "data/processed/features.parquet")
    au = author_lists_from_cache()
    df = feats.merge(au, on="doi", how="left")
    print(f"author lists matched: {df['authors'].notna().mean():.3f}")

    hist = author_history_features(df, df["authors"].fillna(""))
    df = pd.concat([df.reset_index(drop=True), hist], axis=1)
    print("any_prior_author_retraction rate by class:")
    print(df.groupby("retracted")["any_prior_author_retraction"].mean().round(4).to_string())

    FEATS = NUMERIC_FEATURES + ["n_prior_author_retractions", "any_prior_author_retraction"]
    df = partition(df, by="publisher", k=10)
    results = []
    for seed in SEEDS:
        tr, te = train_test_split(df, test_size=0.2, stratify=df["retracted"],
                                  random_state=seed)
        for fset_name, fset in [("base", NUMERIC_FEATURES), ("base+authorhist", FEATS)]:
            Xtr = tr[fset].to_numpy(dtype=np.float32)
            ytr = tr["retracted"].to_numpy(dtype=np.float32)
            Xte = te[fset].to_numpy(dtype=np.float32)
            yte = te["retracted"].to_numpy(dtype=np.float32)
            m = train_xgb(Xtr, ytr, Xte, yte, seed)
            results.append({"seed": seed, "features": fset_name, "model": "central_xgb", **m})
            pre = preprocessor()
            Xtr_s = pre.fit_transform(Xtr).astype(np.float32)
            Xte_s = pre.transform(Xte).astype(np.float32)
            clients = []
            for silo, p in tr.groupby("silo"):
                if p["retracted"].nunique() < 2:
                    continue
                rowsel = tr.index.get_indexer(p.index)
                clients.append(Client(str(silo), Xtr_s[rowsel], ytr[rowsel]))
            out = run_fl(clients, Xte_s, yte, strategy="fedavg", rounds=50,
                         seed=seed, eval_every=50)
            results.append({"seed": seed, "features": fset_name, "model": "fl_fedavg",
                            **out["final"]})
    res = pd.DataFrame(results)
    res.to_csv(ROOT / "results/author_history.csv", index=False)
    print(res.groupby(["features", "model"])[["auprc", "roc_auc", "recall_at_5fpr"]]
          .mean().round(4).to_string())


if __name__ == "__main__":
    main()
