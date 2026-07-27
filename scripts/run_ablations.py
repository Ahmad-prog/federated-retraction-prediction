"""Extended experiment suite — dimensions that strengthen the paper vs prior work.

A) Scale curve:      AUPRC vs training-set size (the "why 138k beats 225" proof)
B) Feature ablation: metadata-only vs text-only vs all features
C) Temporal split:   train on <=2018 papers, test on >=2019 (deployment realism)
D) Per-silo gains:   local vs federated AUPRC for every publisher (fairness)
E) K sweep:          FL performance as the consortium grows (5..20 silos)
F) Feature importance (XGBoost gain) — interpretability a la Usman

Each block is independent and saves its own CSV; failures don't kill the rest.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.build_features import NUMERIC_FEATURES
from src.fl.engine import Client, run_fl
from src.models.baselines import make_xy, preprocessor, train_mlp, train_rf, train_xgb
from src.models.evaluate import evaluate
from src.models.mlp import MLP, set_weights
from src.partition.partitioner import partition
from sklearn.model_selection import train_test_split
import torch

RES = ROOT / "results"
SEEDS = [42, 43, 44]

TEXT_FEATURES = ["flesch_reading_ease", "flesch_kincaid", "gunning_fog", "smog",
                 "abstract_len_words", "hedge_ratio", "certainty_ratio"]
META_FEATURES = [f for f in NUMERIC_FEATURES if f not in TEXT_FEATURES]


def make_clients(tr, Xtr_s, ytr):
    clients = []
    for silo, p in tr.groupby("silo"):
        rows = tr.index.get_indexer(p.index)
        if p["retracted"].nunique() < 2:
            continue
        clients.append(Client(str(silo), Xtr_s[rows], ytr[rows]))
    return clients


def fl_scores(clients, Xte_s, seed):
    out = run_fl(clients, Xte_s, np.zeros(len(Xte_s), dtype=np.float32),
                 strategy="fedavg", rounds=50, seed=seed, eval_every=50)
    model = MLP(Xte_s.shape[1])
    set_weights(model, out["weights"])
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(Xte_s))).numpy()


def a_scale_curve(df):
    rows = []
    for seed in SEEDS:
        tr, te = train_test_split(df, test_size=0.2, stratify=df["retracted"], random_state=seed)
        Xte, yte = make_xy(te)
        for n in [500, 2000, 8000, 32000, len(tr)]:
            sub = tr.sample(n=min(n, len(tr)), random_state=seed)
            Xs, ys = make_xy(sub)
            m = train_xgb(Xs, ys, Xte, yte, seed)
            rows.append({"seed": seed, "n_train": len(sub), **m})
    pd.DataFrame(rows).to_csv(RES / "abl_scale_curve.csv", index=False)


def b_feature_ablation(df):
    rows = []
    for seed in SEEDS:
        tr, te = train_test_split(df, test_size=0.2, stratify=df["retracted"], random_state=seed)
        for fset_name, fset in [("all", NUMERIC_FEATURES), ("metadata_only", META_FEATURES),
                                ("text_only", TEXT_FEATURES)]:
            Xtr = tr[fset].to_numpy(dtype=np.float32)
            ytr = tr["retracted"].to_numpy(dtype=np.float32)
            Xte = te[fset].to_numpy(dtype=np.float32)
            yte = te["retracted"].to_numpy(dtype=np.float32)
            m = train_xgb(Xtr, ytr, Xte, yte, seed)
            rows.append({"seed": seed, "features": fset_name, "model": "central_xgb", **m})
            # federated with same feature set
            part = partition(tr.assign(), by="publisher", k=10)
            pre = preprocessor()
            Xtr_s = pre.fit_transform(Xtr).astype(np.float32)
            Xte_s = pre.transform(Xte).astype(np.float32)
            clients = make_clients(part, Xtr_s, ytr)
            scores = fl_scores(clients, Xte_s, seed)
            rows.append({"seed": seed, "features": fset_name, "model": "fl_fedavg",
                         **evaluate(yte, scores)})
    pd.DataFrame(rows).to_csv(RES / "abl_features.csv", index=False)


def c_temporal(df):
    rows = []
    tr = df[df["year"] <= 2018]
    te = df[df["year"] >= 2019]
    for seed in SEEDS:
        Xtr, ytr = make_xy(tr)
        Xte, yte = make_xy(te)
        for name, fn in [("central_rf", train_rf), ("central_xgb", train_xgb),
                         ("central_mlp", train_mlp)]:
            rows.append({"seed": seed, "model": name, **fn(Xtr, ytr, Xte, yte, seed)})
        part = partition(tr, by="publisher", k=10)
        pre = preprocessor()
        Xtr_s = pre.fit_transform(Xtr).astype(np.float32)
        Xte_s = pre.transform(Xte).astype(np.float32)
        clients = make_clients(part, Xtr_s, ytr)
        scores = fl_scores(clients, Xte_s, seed)
        rows.append({"seed": seed, "model": "fl_fedavg", **evaluate(yte, scores)})
    out = pd.DataFrame(rows)
    out["n_train"], out["n_test"] = len(tr), len(te)
    out.to_csv(RES / "abl_temporal.csv", index=False)


def d_per_silo(df):
    rows = []
    seed = 42
    dfp = partition(df, by="publisher", k=10)
    tr, te = train_test_split(dfp, test_size=0.2, stratify=dfp["retracted"], random_state=seed)
    Xtr, ytr = make_xy(tr)
    pre = preprocessor()
    Xtr_s = pre.fit_transform(Xtr).astype(np.float32)
    clients = make_clients(tr, Xtr_s, ytr)
    Xte_all = pre.transform(make_xy(te)[0]).astype(np.float32)
    scores_fed = fl_scores(clients, Xte_all, seed)
    te = te.reset_index(drop=True)
    for silo, g in te.groupby("silo"):
        idx = g.index.to_numpy()
        yte_s = g["retracted"].to_numpy(dtype=np.float32)
        # local model of this silo
        loc = tr[tr["silo"] == silo]
        m_local = {"auprc": np.nan}
        if loc["retracted"].nunique() == 2:
            Xl, yl = make_xy(loc)
            Xte_s_raw = g[NUMERIC_FEATURES].to_numpy(dtype=np.float32)
            m_local = train_xgb(Xl, yl, Xte_s_raw, yte_s, seed)
        m_fed = evaluate(yte_s, scores_fed[idx])
        rows.append({"silo": silo, "n_test": len(g),
                     "auprc_local": m_local["auprc"], "auprc_fed": m_fed["auprc"],
                     "pos_rate": float(yte_s.mean())})
    pd.DataFrame(rows).to_csv(RES / "abl_per_silo.csv", index=False)


def e_k_sweep(df):
    rows = []
    for seed in SEEDS:
        tr, te = train_test_split(df, test_size=0.2, stratify=df["retracted"], random_state=seed)
        Xte, yte = make_xy(te)
        for k in [5, 10, 15, 20]:
            part = partition(tr, by="publisher", k=k)
            Xtr, ytr = make_xy(part)
            pre = preprocessor()
            Xtr_s = pre.fit_transform(Xtr).astype(np.float32)
            Xte_s = pre.transform(Xte).astype(np.float32)
            clients = make_clients(part, Xtr_s, ytr)
            scores = fl_scores(clients, Xte_s, seed)
            rows.append({"seed": seed, "k": k, "n_clients": len(clients),
                         **evaluate(yte, scores)})
    pd.DataFrame(rows).to_csv(RES / "abl_k_sweep.csv", index=False)


def f_feature_importance(df):
    from xgboost import XGBClassifier
    from sklearn.impute import SimpleImputer
    tr, _ = train_test_split(df, test_size=0.2, stratify=df["retracted"], random_state=42)
    X, y = make_xy(tr)
    clf = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
                        random_state=42, eval_metric="aucpr", n_jobs=4)
    clf.fit(SimpleImputer(strategy="median").fit_transform(X), y)
    imp = pd.DataFrame({"feature": NUMERIC_FEATURES,
                        "gain": clf.feature_importances_}).sort_values("gain", ascending=False)
    imp.to_csv(RES / "abl_feature_importance.csv", index=False)


def main() -> None:
    df = pd.read_parquet(ROOT / "data/processed/features.parquet")
    df = partition(df, by="publisher", k=10)
    for name, fn in [("scale_curve", a_scale_curve), ("feature_ablation", b_feature_ablation),
                     ("temporal", c_temporal), ("per_silo", d_per_silo),
                     ("k_sweep", e_k_sweep), ("feature_importance", f_feature_importance)]:
        try:
            print(f"=== {name} ===", flush=True)
            fn(df.copy())
            print(f"{name} done", flush=True)
        except Exception:
            print(f"{name} FAILED:\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
