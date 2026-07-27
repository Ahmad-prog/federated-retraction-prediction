"""Run one experiment config: baselines + FL strategies on a partitioned corpus.

Usage:
    python scripts/run_experiment.py experiments/main_publisher.yaml [--smoke]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.build_features import NUMERIC_FEATURES
from src.fl.engine import Client, run_fl
from src.models.baselines import (make_xy, preprocessor, train_logreg, train_mlp,
                                  train_rf, train_xgb)
from src.partition.partitioner import partition, silo_stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.smoke:
        cfg["seeds"] = cfg["seeds"][:1]
        cfg["fl"]["rounds"] = 2
        cfg["partition"]["k"] = 3
        cfg["sample_rows"] = 1000

    df = pd.read_parquet(ROOT / cfg["features"])
    if cfg.get("sample_rows"):
        df = df.sample(n=min(cfg["sample_rows"], len(df)), random_state=42)
    df = partition(df, by=cfg["partition"]["by"], k=cfg["partition"]["k"])
    stats = silo_stats(df)
    print(stats.to_string(index=False))

    results = []
    for seed in cfg["seeds"]:
        tr, te = train_test_split(df, test_size=0.2, stratify=df["retracted"],
                                  random_state=seed)
        Xtr, ytr = make_xy(tr)
        Xte, yte = make_xy(te)
        pre = preprocessor()
        Xtr_s = pre.fit_transform(Xtr).astype(np.float32)
        Xte_s = pre.transform(Xte).astype(np.float32)

        # centralized upper bounds
        for name, fn in [("central_logreg", train_logreg), ("central_rf", train_rf),
                         ("central_xgb", train_xgb), ("central_mlp", train_mlp)]:
            m = fn(Xtr, ytr, Xte, yte, seed)
            results.append({"seed": seed, "model": name, **m})

        # local-only: each silo trains alone, evaluated on the global test set
        local = []
        for silo, part in tr.groupby("silo"):
            if part["retracted"].nunique() < 2:
                continue
            Xl, yl = make_xy(part)
            local.append(train_logreg(Xl, yl, Xte, yte, seed))
        if local:
            avg = {k: float(np.nanmean([r[k] for r in local])) for k in local[0]}
            results.append({"seed": seed, "model": "local_only_avg", **avg})

        # federated strategies
        clients = []
        for silo, part in tr.groupby("silo"):
            idx = part.index
            rows = tr.index.get_indexer(idx)
            clients.append(Client(str(silo), Xtr_s[rows], ytr[rows]))
        for strat in cfg["fl"]["strategies"]:
            out = run_fl(clients, Xte_s, yte, strategy=strat,
                         rounds=cfg["fl"]["rounds"],
                         local_epochs=cfg["fl"]["local_epochs"],
                         lr=cfg["fl"]["lr"], mu=cfg["fl"].get("mu", 0.01),
                         seed=seed)
            results.append({"seed": seed, "model": f"fl_{strat}", **out["final"]})
            hist_f = ROOT / "results" / f"{Path(args.config).stem}_{strat}_seed{seed}_history.json"
            hist_f.parent.mkdir(exist_ok=True)
            hist_f.write_text(json.dumps(out["history"]))

    res = pd.DataFrame(results)
    out_f = ROOT / "results" / f"{Path(args.config).stem}{'_smoke' if args.smoke else ''}.csv"
    res.to_csv(out_f, index=False)
    print(res.drop(columns=["round"], errors="ignore")
          .groupby("model").mean(numeric_only=True).round(4).to_string())
    print(f"\nSaved -> {out_f}")
    stats.to_csv(ROOT / "results" / f"{Path(args.config).stem}_silostats.csv", index=False)


if __name__ == "__main__":
    main()
