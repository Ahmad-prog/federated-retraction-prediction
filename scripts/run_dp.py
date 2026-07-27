"""DP-FedAvg privacy-utility sweep.

Usage: python scripts/run_dp.py experiments/dp_publisher.yaml [--smoke]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dp.dp_engine import epsilon_for, run_dp_fl
from src.fl.engine import Client
from src.models.baselines import make_xy, preprocessor
from src.partition.partitioner import partition


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.smoke:
        cfg["seeds"] = cfg["seeds"][:1]
        cfg["rounds"] = 2
        cfg["sigmas"] = cfg["sigmas"][:2]
        cfg["sample_rows"] = 1000

    df = pd.read_parquet(ROOT / cfg["features"])
    if cfg.get("sample_rows"):
        df = df.sample(n=min(cfg["sample_rows"], len(df)), random_state=42)
    df = partition(df, by=cfg["partition"]["by"], k=cfg["partition"]["k"])

    results = []
    for seed in cfg["seeds"]:
        tr, te = train_test_split(df, test_size=0.2, stratify=df["retracted"],
                                  random_state=seed)
        Xtr, ytr = make_xy(tr)
        Xte, yte = make_xy(te)
        pre = preprocessor()
        Xtr_s = pre.fit_transform(Xtr).astype(np.float32)
        Xte_s = pre.transform(Xte).astype(np.float32)
        clients = []
        for silo, part in tr.groupby("silo"):
            rows = tr.index.get_indexer(part.index)
            clients.append(Client(str(silo), Xtr_s[rows], ytr[rows]))
        for sigma in cfg["sigmas"]:
            out = run_dp_fl(clients, Xte_s, yte, sigma=sigma, clip=cfg["clip"],
                            rounds=cfg["rounds"], seed=seed)
            eps = epsilon_for(sigma, cfg["rounds"])
            results.append({"seed": seed, "sigma": sigma, "epsilon": eps, **out["final"]})
            print(results[-1])

    res = pd.DataFrame(results)
    out_f = ROOT / "results" / f"{Path(args.config).stem}{'_smoke' if args.smoke else ''}.csv"
    res.to_csv(out_f, index=False)
    print(res.groupby("sigma").mean(numeric_only=True).round(4).to_string())


if __name__ == "__main__":
    main()
