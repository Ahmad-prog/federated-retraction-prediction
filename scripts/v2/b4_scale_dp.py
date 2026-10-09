"""B4: consortium size, journal-level silos and corrected client-level DP (R1.5, R1.6; C9, M4).

Setting pub (publication-time features), 10 seeds, same splits as B2.
  ksweep    publisher silos K = 5, 10, 15, 20 (FedAvg)
  journal   one silo per journal (venue) with >= MIN_JOURNAL training articles,
            the rest pooled into per-publisher remainder silos; FedAvg with full and
            with Poisson client sampling (q = 0.1)
  dp        DP-FedAvg (clip C, uniform fixed-denominator weights) for
              publisher K=10, full participation   sigma in {0, 0.3, 0.5, 1, 2}
              journal-level, q = 0.1, 200 rounds   sigma in {0, 0.5, 1, 2, 4}
            epsilon reported at delta = 1e-3 (publisher) and 1e-5 (journal);
            with q < 1 the subsampled-Gaussian RDP accountant (Opacus) is used.
Output: results_v2/b4_scale_dp/results.csv
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.v2.b2_main_benchmark import META, TEXT, partition, split  # noqa: E402
from src.models.evaluate import evaluate  # noqa: E402
from src.v2 import DATA2, RES2  # noqa: E402

OUT = RES2 / "b4_scale_dp"
MIN_JOURNAL = 60


def run_seed(seed: int, smoke: bool) -> list[dict]:
    import torch

    from src.v2.fl import FedStandardizer, Silo, epsilon_gaussian, run_fl, scores
    torch.set_num_threads(1)
    df = pd.read_parquet(DATA2 / "tabular_v2.parquet").reset_index(drop=True)
    if smoke:
        df = df.sample(n=8000, random_state=seed).reset_index(drop=True)
    df["silo"] = partition(df, "publisher")
    df["part"] = split(df, seed)        # identical split to B2 for this seed
    feats = META + TEXT
    X = df[feats].to_numpy(np.float64)
    y = df["retracted"].to_numpy(np.float32)
    m = {p: (df["part"] == p).to_numpy() for p in ["train", "val", "test"]}
    yte = y[m["test"]]
    rows = []
    R = 3 if smoke else 50

    def make_silos(labels: pd.Series):
        names = sorted(labels.unique())
        sm = {s: (labels == s).to_numpy() for s in names}
        std = FedStandardizer.fit([X[m["train"] & sm[s]] for s in names])
        Z = std.transform(X)
        objs = [Silo(str(s), Z[m["train"] & sm[s]], y[m["train"] & sm[s]],
                     Z[m["val"] & sm[s]], y[m["val"] & sm[s]]) for s in names]
        # single-class silos are legitimate clients (journals without retractions)
        objs = [o for o in objs if o.n >= 10]
        for o in objs:
            if len(o.yva) == 0:  # no validation rows: reuse a train slice for val-loss only
                o.Xva, o.yva = o.Xtr[:5], o.ytr[:5]
                o.__post_init__()
        return objs, Z

    def rec(exp, model, sc, **kw):
        rows.append({"seed": seed, "exp": exp, "model": model, **kw, **evaluate(yte, sc)})

    # ---- consortium size
    for k in [5, 10, 15, 20]:
        objs, Z = make_silos(partition(df, "publisher", k))
        out = run_fl(objs, Z.shape[1], "fedavg", rounds=R, seed=seed)
        rec("ksweep", "fl_fedavg", scores(out["model"], Z[m["test"]]), k=k, n_clients=len(objs))

    # ---- journal-level silos
    tr_counts = df.loc[m["train"]].groupby("venue_id").size()
    big = set(tr_counts[tr_counts >= MIN_JOURNAL].index)
    jl = df["venue_id"].where(df["venue_id"].isin(big),
                              "rest::" + df["publisher"].fillna("Unknown").astype(str))
    objs, Z = make_silos(jl)
    n_j = len(objs)
    for q, rounds in [(1.0, R), (0.1, 4 * R)]:
        out = run_fl(objs, Z.shape[1], "fedavg", rounds=rounds, seed=seed, client_frac=q)
        rec("journal", "fl_fedavg", scores(out["model"], Z[m["test"]]), q=q, n_clients=n_j,
            rounds=rounds)

    # ---- DP
    pub_objs, Zp = make_silos(df["silo"])
    for sigma in [0.0, 0.3, 0.5, 1.0, 2.0]:
        out = run_fl(pub_objs, Zp.shape[1], "fedavg", rounds=R, seed=seed,
                     dp_sigma=sigma, dp_clip=1.0)
        rec("dp_publisher", "dp_fedavg", scores(out["model"], Zp[m["test"]]), sigma=sigma,
            epsilon=epsilon_gaussian(sigma, R, 1e-3), delta=1e-3, n_clients=len(pub_objs),
            q=1.0, rounds=R)
    for sigma in [0.0, 0.5, 1.0, 2.0, 4.0]:
        out = run_fl(objs, Z.shape[1], "fedavg", rounds=4 * R, seed=seed, client_frac=0.1,
                     dp_sigma=sigma, dp_clip=1.0)
        rec("dp_journal", "dp_fedavg", scores(out["model"], Z[m["test"]]), sigma=sigma,
            epsilon=epsilon_gaussian(sigma, 4 * R, 1e-5, q=0.1), delta=1e-5, n_clients=n_j,
            q=0.1, rounds=4 * R)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(run_seed, 42 + i, a.smoke): 42 + i for i in range(a.seeds)}
        for f in as_completed(futs):
            try:
                rows.extend(f.result())
                print(f"seed {futs[f]} done", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"seed {futs[f]} FAILED: {e!r}", flush=True)
            pd.DataFrame(rows).to_csv(OUT / ("results_smoke.csv" if a.smoke else "results.csv"),
                                      index=False)
    res = pd.DataFrame(rows)
    keys = [c for c in ["exp", "k", "q", "sigma", "epsilon"] if c in res]
    print(res.groupby(keys, dropna=False)[["auprc", "roc_auc"]].agg(["mean", "std"])
          .round(4).to_string())


if __name__ == "__main__":
    main()
