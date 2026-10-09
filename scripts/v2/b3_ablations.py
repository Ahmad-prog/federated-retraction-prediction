"""B3: attribution + ablations + personalisation (R1.2, R1.3; C1, C2, C4, M2).

Per seed (publisher partition, same splits as B2):
  attribution  v1exact : the exact v1 feature matrix (2026 citations, v1 early-citation
                         window, missing abstract coded as length 0) under the v2 protocol,
                         incl. the v1-style local baseline (per-silo logistic regression)
  ablations    (setting pub) meta_only, text_only, no_year, has_abstract subset
  silo-aware   central XGB/MLP with a one-hot silo feature (is local's own-silo edge
               just silo identity?)
  personalise  FedAvg + fine-tuning for 3/10/30 epochs (own-silo view)
Output: results_v2/b3_ablations/results.csv
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.v2.b2_main_benchmark import META, TEXT, partition, split, xgb_model  # noqa: E402
from src.features.build_features import NUMERIC_FEATURES as V1_FEATURES  # noqa: E402
from src.models.evaluate import evaluate  # noqa: E402
from src.v2 import DATA2, RES2, ROOT  # noqa: E402

OUT = RES2 / "b3_ablations"


def run_seed(seed: int, smoke: bool) -> list[dict]:
    import torch
    from sklearn.linear_model import LogisticRegression

    from src.v2.fl import FedStandardizer, Silo, personalize, run_fl, scores, train_central_mlp
    torch.set_num_threads(1)
    base = pd.read_parquet(DATA2 / "tabular_v2.parquet").reset_index(drop=True)
    v1 = pd.read_parquet(ROOT / "data/processed/features.parquet")[["doi"] + V1_FEATURES]
    base = base.merge(v1.add_prefix("v1_").rename(columns={"v1_doi": "doi"}), on="doi", how="left")
    if smoke:
        base = base.sample(n=6000, random_state=seed).reset_index(drop=True)
    base["silo"] = partition(base, "publisher")
    base["part"] = split(base, seed)
    rows = []
    E = 8 if smoke else 100
    R_ = 3 if smoke else 50

    def fit_eval(tag, df, feats, models=("central_xgb", "central_mlp", "fl_fedavg"),
                 silo_onehot=False, ft_epochs=()):
        X = df[feats].to_numpy(np.float64)
        if silo_onehot:
            X = np.hstack([X, pd.get_dummies(df["silo"]).to_numpy(np.float64)])
        y = df["retracted"].to_numpy(np.float32)
        silos = sorted(df["silo"].unique())
        m = {p: (df["part"] == p).to_numpy() for p in ["train", "val", "test"]}
        sm = {s: (df["silo"] == s).to_numpy() for s in silos}
        std = FedStandardizer.fit([X[m["train"] & sm[s]] for s in silos])
        Z = std.transform(X)
        d = Z.shape[1]
        yte, ste = y[m["test"]], df.loc[m["test"], "silo"].to_numpy()
        Ztr, ytr, Zte = Z[m["train"]], y[m["train"]], Z[m["test"]]

        def rec(model, sc, view="pooled", silo="ALL"):
            msk = np.ones(len(yte), bool) if view == "pooled" else (
                ste == silo if view == "own_silo" else ste != silo)
            rows.append({"seed": seed, "exp": tag, "model": model, "view": view, "silo": silo,
                         **evaluate(yte[msk], sc[msk])})

        def rec_all(model, sc):
            rec(model, sc)
            for s in silos:
                rec(model, sc, "own_silo", s)

        spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
        if "central_xgb" in models:
            rec_all("central_xgb", xgb_model(seed, spw).fit(Ztr, ytr).predict_proba(Zte)[:, 1])
        if "central_mlp" in models:
            rec_all("central_mlp", scores(train_central_mlp(Ztr, ytr, Z[m["val"]], y[m["val"]], d,
                                                            epochs=E, seed=seed), Zte))
        silo_objs = [Silo(s, Z[m["train"] & sm[s]], y[m["train"] & sm[s]],
                          Z[m["val"] & sm[s]], y[m["val"] & sm[s]]) for s in silos]
        silo_objs = [s for s in silo_objs if s.n and 0 < s.prev < 1 and len(s.yva)]
        if "local_logreg" in models:  # the v1 local baseline, evaluated as in v1
            for s in silo_objs:
                lr = LogisticRegression(max_iter=3000, class_weight="balanced").fit(s.Xtr, s.ytr)
                sc = lr.predict_proba(Zte)[:, 1]
                rec("local_logreg", sc, "pooled", s.name)
                rec("local_logreg", sc, "own_silo", s.name)
        if "fl_fedavg" in models:
            out = run_fl(silo_objs, d, "fedavg", rounds=R_, seed=seed)
            rec_all("fl_fedavg", scores(out["model"], Zte))
            for ep in ft_epochs:
                for s in silo_objs:
                    pm = personalize(out["model"], s, d, epochs=ep if not smoke else 1, seed=seed)
                    rec(f"fl_fedavg_ft{ep}", scores(pm, Zte), "own_silo", s.name)

    v1cols = [f"v1_{c}" for c in V1_FEATURES]
    fit_eval("v1exact", base, v1cols, models=("central_xgb", "central_mlp", "fl_fedavg",
                                               "local_logreg"))
    feats_pub = META + TEXT
    fit_eval("pub_all", base, feats_pub, ft_epochs=(10, 30))
    fit_eval("pub_meta_only", base, META)
    fit_eval("pub_text_only", base, TEXT)
    fit_eval("pub_no_year", base, [f for f in feats_pub if f != "year"])
    fit_eval("pub_has_abstract", base[base["has_abstract"] == 1].reset_index(drop=True), feats_pub)
    fit_eval("pub_silo_onehot", base, feats_pub, models=("central_xgb", "central_mlp"),
             silo_onehot=True)
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
    res = pd.DataFrame(rows)
    res.to_csv(OUT / ("results_smoke.csv" if a.smoke else "results.csv"), index=False)
    for view in ["pooled", "own_silo"]:
        r = res[res.view == view]
        if view == "pooled":
            r = r.groupby(["exp", "model", "seed"])["auprc"].mean().reset_index()
        else:
            r = r.groupby(["exp", "model", "seed"])["auprc"].mean().reset_index()
        print(f"\n== {view} ==")
        print(r.groupby(["exp", "model"])["auprc"].agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
