"""Heterogeneous-modality federated learning.

Silos contribute whatever features they have: all silos share 15 metadata +
abstract features; the curated biomedical silo (Usman WebSci'25 corpus)
additionally has 8 full-text section features (readability + certainty of
Abstract/Methods/Results/Conclusion). Absent features are zero with a
has_fulltext indicator = 0.

Experiments:
  H1  centralized upper bounds on the curated corpus WITH full text (5-fold CV)
  H2  federated: 10 publisher silos + curated silo, optional-modality input
  H3  ablation: same federation without the full-text block
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.build_features import NUMERIC_FEATURES, text_features
from src.fl.engine import Client, run_fl
from src.models.evaluate import evaluate
from src.models.mlp import MLP, set_weights
from src.partition.partitioner import partition

FULLTEXT = ["abstract_Readability", "Method_Readability", "Results_Readability",
            "Conclusion_Readability", "Abstract_Certainty", "Method_Certainty",
            "Result_Certainty", "Conclusion_Certainty"]
ALL_FEATURES = NUMERIC_FEATURES + FULLTEXT + ["has_fulltext"]
SEEDS = [42, 43, 44]


def load_usman_silo() -> pd.DataFrame:
    """Curated corpus with metadata (Europe PMC cache), our abstract features,
    and his precomputed full-text section features."""
    import json
    raw = pd.read_csv(
        ROOT / "data/raw/Scientific-Accountability-main/Retracted_NonRetracted Articles.csv",
        sep=";", engine="python", quotechar='"', on_bad_lines="skip")
    raw["retracted"] = (raw["Label"].str.strip() == "R").astype(float)
    cache = ROOT / "data/cache/usman_epmc"
    rows = []
    for _, rec in raw.iterrows():
        meta = {}
        f = cache / f"PMC{int(rec['pmcid'])}.json"
        if f.exists():
            meta = json.loads(f.read_text())
        tf = text_features(str(rec["abstract"]) if pd.notna(rec["abstract"]) else "")
        row = {
            "n_authors": meta.get("n_authors"), "n_countries": np.nan,
            "n_references": meta.get("n_references"),
            "cited_by_count": meta.get("cited_by_count"),
            "early_citations_2y": np.nan, "year": meta.get("year"),
            "is_oa": meta.get("is_oa"), "is_international": np.nan, **tf,
            "retracted": rec["retracted"], "has_fulltext": 1.0,
        }
        for c in FULLTEXT:
            # values use decimal commas (German locale export)
            row[c] = pd.to_numeric(str(rec.get(c)).replace(",", "."), errors="coerce")
        rows.append(row)
    return pd.DataFrame(rows)


def xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df.reindex(columns=ALL_FEATURES).to_numpy(dtype=np.float32)
    y = df["retracted"].to_numpy(dtype=np.float32)
    return X, y


def h1_centralized_fulltext(us: pd.DataFrame) -> pd.DataFrame:
    """Upper bounds on the curated corpus with full-text features, 5-fold CV."""
    feats = FULLTEXT + [f for f in NUMERIC_FEATURES]
    X = us.reindex(columns=feats).to_numpy(dtype=np.float32)
    Xft = us.reindex(columns=FULLTEXT).to_numpy(dtype=np.float32)
    y = us["retracted"].to_numpy(dtype=np.float32)
    out = []
    for name, Xc in [("fulltext+meta", X), ("fulltext_only", Xft)]:
        for mname, mk in [("rf", lambda s: RandomForestClassifier(
                              n_estimators=300, class_weight="balanced",
                              n_jobs=4, random_state=s)),
                          ("xgb", lambda s: XGBClassifier(
                              n_estimators=300, max_depth=4, learning_rate=0.08,
                              random_state=s, eval_metric="aucpr", n_jobs=4))]:
            for seed in SEEDS:
                skf = StratifiedKFold(5, shuffle=True, random_state=seed)
                scores = np.zeros(len(y))
                imp = SimpleImputer(strategy="median")
                for tr_i, te_i in skf.split(Xc, y):
                    clf = mk(seed)
                    clf.fit(imp.fit_transform(Xc[tr_i]), y[tr_i])
                    scores[te_i] = clf.predict_proba(imp.transform(Xc[te_i]))[:, 1]
                m = evaluate(y, scores)
                m["balanced_accuracy"] = float(((scores > 0.5) == (y > 0.5)).mean())
                out.append({"exp": "H1", "features": name, "model": mname,
                            "seed": seed, **m})
    return pd.DataFrame(out)


def build_clients(tr: pd.DataFrame, pre: tuple) -> list[Client]:
    imp, sc = pre
    clients = []
    for silo, p in tr.groupby("silo"):
        if p["retracted"].nunique() < 2:
            continue
        X, y = xy(p)
        Xs = sc.transform(imp.transform(X)).astype(np.float32)
        clients.append(Client(str(silo), Xs, y))
    return clients


def h2_h3_federated(big: pd.DataFrame, us: pd.DataFrame, use_fulltext: bool,
                    boost: int, seed: int) -> dict:
    tag = ("H2" if use_fulltext else "H3") + (f"_boost{boost}" if boost > 1 else "")
    big = partition(big, by="publisher", k=10)
    us = us.copy()
    us["silo"] = "CuratedBio"
    if not use_fulltext:
        for c in FULLTEXT + ["has_fulltext"]:
            us[c] = 0.0
    big_tr, big_te = train_test_split(big, test_size=0.2,
                                      stratify=big["retracted"], random_state=seed)
    us_tr, us_te = train_test_split(us, test_size=0.2,
                                    stratify=us["retracted"], random_state=seed)
    if boost > 1:
        us_tr = pd.concat([us_tr] * boost, ignore_index=True)
    tr = pd.concat([big_tr, us_tr], ignore_index=True)

    Xtr_all, _ = xy(tr)
    imp = SimpleImputer(strategy="median").fit(Xtr_all)
    sc = StandardScaler().fit(imp.transform(Xtr_all))
    clients = build_clients(tr, (imp, sc))

    Xge, yge = xy(big_te)
    Xue, yue = xy(us_te)
    Xge_s = sc.transform(imp.transform(Xge)).astype(np.float32)
    Xue_s = sc.transform(imp.transform(Xue)).astype(np.float32)

    out = run_fl(clients, Xge_s, yge, strategy="fedavg", rounds=50,
                 seed=seed, eval_every=50)
    model = MLP(Xge_s.shape[1])
    set_weights(model, out["weights"])
    model.eval()
    with torch.no_grad():
        sc_g = torch.sigmoid(model(torch.tensor(Xge_s))).numpy()
        sc_u = torch.sigmoid(model(torch.tensor(Xue_s))).numpy()
    mg = evaluate(yge, sc_g)
    mu = evaluate(yue, sc_u)
    mu["balanced_accuracy"] = float(((sc_u > 0.5) == (yue > 0.5)).mean())
    return {"exp": tag, "seed": seed,
            "global_auprc": mg["auprc"], "global_roc": mg["roc_auc"],
            "usman_auprc": mu["auprc"], "usman_roc": mu["roc_auc"],
            "usman_accuracy": mu["balanced_accuracy"]}


def main() -> None:
    us = load_usman_silo()
    print(f"curated silo: {len(us)} rows, fulltext features present: "
          f"{us[FULLTEXT].notna().all(axis=1).sum()}")
    big = pd.read_parquet(ROOT / "data/processed/features.parquet")
    big["has_fulltext"] = 0.0
    for c in FULLTEXT:
        big[c] = 0.0

    res1 = h1_centralized_fulltext(us)
    print(res1.groupby(["features", "model"])[["auprc", "roc_auc", "balanced_accuracy"]]
          .mean().round(3).to_string())

    rows = []
    for seed in SEEDS:
        for use_ft, boost in [(True, 1), (True, 10), (False, 1)]:
            r = h2_h3_federated(big, us, use_ft, boost, seed)
            print(r)
            rows.append(r)
    res2 = pd.DataFrame(rows)

    res1.to_csv(ROOT / "results/hetero_h1.csv", index=False)
    res2.to_csv(ROOT / "results/hetero_fl.csv", index=False)
    print(res2.groupby("exp").mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
