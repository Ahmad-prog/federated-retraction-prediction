"""B2: corrected main benchmark (R1.1, R1.3, R1.4, R1.8) on the leakage-clean tabular corpus.

Per seed (10 seeds), per setting, per partition:
  split 70/10/20 (train/val/test), stratified by label x silo
  federated preprocessing (FedStandardizer)
  central : LogReg, RF, XGBoost, MLP (same SGD optimizer as FL; epoch picked on val loss)
  local   : per-silo MLP and XGBoost
  FL      : FedAvg, FedProx, SCAFFOLD, FedBal(fixed), FedAvg + per-silo fine-tuning,
            cyclic federated XGBoost
Evaluation views:
  pooled    : the shared 20% test set (v1 view)
  own_silo  : each silo's model on its own test slice (deployment view, C2)
  out_silo  : each silo's local model on the other silos' test slices
All test scores are saved (npz) for bootstrap CIs / DeLong / prevalence re-weighting.

Settings:
  pub    screening at publication: no citation features (main, leak-free)
  pub2   screening 2 years after publication: + citations in years y..y+2 (subset)
  v1leak v1 features incl. 2026 cited_by_count (only to quantify the v1 leakage)

Usage: python scripts/v2/b2_main_benchmark.py [--seeds 10] [--jobs 40] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.evaluate import evaluate  # noqa: E402
from src.v2 import DATA2, RES2  # noqa: E402

META = ["n_authors", "n_countries", "n_references", "is_oa", "year"]
TEXT = ["flesch_reading_ease", "flesch_kincaid", "gunning_fog", "smog",
        "abstract_len_words", "hedge_ratio", "certainty_ratio", "has_abstract"]
SETTINGS = {
    "pub": META + TEXT,
    "pub2": META + TEXT + ["cites_pub2"],
    "v1leak": META + TEXT + ["cited_by_count_2026"],
}
OUT = RES2 / "b2_main"


def partition(df: pd.DataFrame, by: str, k: int = 10) -> pd.Series:
    col = df[by].fillna("Unknown")
    top = col.value_counts().index[: k - 1]
    return col.where(col.isin(top), "Other")


def split(df: pd.DataFrame, seed: int):
    rng = np.random.default_rng(seed)
    strata = df["silo"].astype(str) + "|" + df["retracted"].astype(str)
    part = np.empty(len(df), dtype=object)
    for _, idx in df.groupby(strata).indices.items():
        idx = rng.permutation(idx)
        n = len(idx)
        n_te, n_va = int(round(0.2 * n)), int(round(0.1 * n))
        part[idx[:n_te]] = "test"
        part[idx[n_te:n_te + n_va]] = "val"
        part[idx[n_te + n_va:]] = "train"
    return part


def xgb_model(seed, spw, n_estimators=300):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=n_estimators, max_depth=5, learning_rate=0.08,
                         subsample=0.9, colsample_bytree=0.9, n_jobs=1, tree_method="hist",
                         scale_pos_weight=spw, random_state=seed, eval_metric="aucpr")


def run_job(setting: str, by: str, seed: int, smoke: bool) -> list[dict]:
    import torch
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    from src.v2.fl import (FedStandardizer, Silo, personalize, run_fl, scores,
                           train_central_mlp)
    torch.set_num_threads(1)
    df = pd.read_parquet(DATA2 / "tabular_v2.parquet")
    if setting == "pub2":
        df = df[df["in_setting_pub2"]]
    df = df.reset_index(drop=True)
    if smoke:
        df = df.sample(n=6000, random_state=seed).reset_index(drop=True)
    df["silo"] = partition(df, by)
    df["part"] = split(df, seed)
    feats = SETTINGS[setting]
    X = df[feats].to_numpy(dtype=np.float64)
    y = df["retracted"].to_numpy(dtype=np.float32)
    silos = sorted(df["silo"].unique())
    m = {p: (df["part"] == p).to_numpy() for p in ["train", "val", "test"]}
    sm = {s: (df["silo"] == s).to_numpy() for s in silos}

    std = FedStandardizer.fit([X[m["train"] & sm[s]] for s in silos])
    Z = std.transform(X)
    d = Z.shape[1]
    yte = y[m["test"]]
    te_silo = df.loc[m["test"], "silo"].to_numpy()
    rows, saved = [], {"y_test": yte, "silo_test": te_silo}
    tag = dict(setting=setting, partition=by, seed=seed)

    def record(model_name, score_vec, view="pooled", silo="ALL", extra=None):
        """`silo` names the model's silo (local/personalised models) or the evaluated
        slice (global models in the own_silo view)."""
        if view == "pooled":
            mask = np.ones(len(yte), bool)
        elif view == "own_silo":
            mask = te_silo == silo
        else:  # out_silo
            mask = te_silo != silo
        r = {**tag, "model": model_name, "view": view, "silo": silo,
             "n_test": int(mask.sum()), "pos_rate": float(yte[mask].mean()),
             **evaluate(yte[mask], score_vec[mask]), **(extra or {})}
        rows.append(r)

    spw = float((y[m["train"]] == 0).sum() / max((y[m["train"]] == 1).sum(), 1))
    Ztr, ytr, Zva, yva, Zte = Z[m["train"]], y[m["train"]], Z[m["val"]], y[m["val"]], Z[m["test"]]

    # ---- centralized (pooled data; the legally unattainable reference)
    lr = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed).fit(Ztr, ytr)
    s_lr = lr.predict_proba(Zte)[:, 1]
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=1,
                                min_samples_leaf=2, random_state=seed).fit(Ztr, ytr)
    s_rf = rf.predict_proba(Zte)[:, 1]
    xg = xgb_model(seed, spw).fit(Ztr, ytr)
    s_xg = xg.predict_proba(Zte)[:, 1]
    mlp = train_central_mlp(Ztr, ytr, Zva, yva, d, epochs=8 if smoke else 100, seed=seed)
    s_mlp = scores(mlp, Zte)
    for name, s in [("central_logreg", s_lr), ("central_rf", s_rf), ("central_xgb", s_xg),
                    ("central_mlp", s_mlp)]:
        record(name, s)
        saved[name] = s
        for si in silos:
            record(name, s, "own_silo", si)

    # ---- federated silos
    silo_objs = [Silo(si, Z[m["train"] & sm[si]], y[m["train"] & sm[si]],
                      Z[m["val"] & sm[si]], y[m["val"] & sm[si]]) for si in silos]
    silo_objs = [s for s in silo_objs if s.n > 0 and 0 < s.prev < 1 and len(s.yva) > 0]

    # ---- local-only (same MLP and same XGBoost as central)
    for s in silo_objs:
        lm = train_central_mlp(s.Xtr, s.ytr, s.Xva, s.yva, d, epochs=8 if smoke else 100,
                               seed=seed)
        sc_mlp = scores(lm, Zte)
        lspw = float((s.ytr == 0).sum() / max((s.ytr == 1).sum(), 1))
        sc_xgb = xgb_model(seed, lspw).fit(s.Xtr, s.ytr).predict_proba(Zte)[:, 1]
        for name, sc in [("local_mlp", sc_mlp), ("local_xgb", sc_xgb)]:
            record(name, sc, "pooled", s.name)        # this silo's model on everyone's articles
            record(name, sc, "own_silo", s.name)      # on its own articles
            record(name, sc, "out_silo", s.name)      # on other publishers' articles
            saved[f"{name}__{s.name}"] = sc

    # ---- FL strategies
    rounds = 3 if smoke else 50
    for strat, kw in [("fedavg", {}), ("fedprox", {"mu": 0.01}), ("scaffold", {"lr": 0.05}),
                      ("fedbal", {})]:
        t0 = time.time()
        out = run_fl(silo_objs, d, strategy=strat, rounds=rounds, seed=seed, **kw)
        sc = scores(out["model"], Zte)
        name = f"fl_{strat}"
        extra = {"best_round": out["best_round"], "secs": round(time.time() - t0, 1)}
        record(name, sc, extra=extra)
        saved[name] = sc
        for s in silo_objs:
            record(name, sc, "own_silo", s.name, extra)
        if strat == "fedavg":
            json.dump(out["history"], open(OUT / f"hist_{setting}_{by}_s{seed}_fedavg.json", "w"))
            # personalised: fine-tune the global model on each silo, evaluate on own slice
            for s in silo_objs:
                pm = personalize(out["model"], s, d, epochs=1 if smoke else 3, seed=seed)
                psc = scores(pm, Zte)
                record("fl_fedavg_ft", psc, "own_silo", s.name)
                record("fl_fedavg_ft", psc, "out_silo", s.name)

    # ---- cyclic federated XGBoost: the booster visits silos in turn, adding trees
    booster, per_visit = None, 10 if not smoke else 3
    cyc_rounds = 3
    for r in range(cyc_rounds):
        for s in silo_objs:
            lspw = float((s.ytr == 0).sum() / max((s.ytr == 1).sum(), 1))
            mdl = xgb_model(seed + r, lspw, n_estimators=per_visit)
            mdl.fit(s.Xtr, s.ytr, xgb_model=booster)
            booster = mdl.get_booster()
    import xgboost as xgb
    sc = booster.predict(xgb.DMatrix(Zte))
    record("fl_xgb_cyclic", sc)
    saved["fl_xgb_cyclic"] = sc
    for s in silo_objs:
        record("fl_xgb_cyclic", sc, "own_silo", s.name)

    np.savez_compressed(OUT / f"scores_{setting}_{by}_s{seed}.npz",
                        **{k: np.asarray(v) for k, v in saved.items()})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=40)
    ap.add_argument("--settings", default="pub,pub2,v1leak")
    ap.add_argument("--partitions", default="publisher,field")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(st, by, 42 + i, a.smoke) for st in a.settings.split(",")
            for by in a.partitions.split(",") for i in range(a.seeds)]
    print(f"{len(jobs)} jobs", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(run_job, *j): j for j in jobs}
        for f in as_completed(futs):
            try:
                rows.extend(f.result())
                print(f"done {futs[f]}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"FAILED {futs[f]}: {e!r}", flush=True)
            pd.DataFrame(rows).to_csv(OUT / ("results_smoke.csv" if a.smoke else "results.csv"),
                                      index=False)
    res = pd.DataFrame(rows)
    pooled = res[(res.view == "pooled") & (res.silo == "ALL")]
    print(pooled.groupby(["setting", "partition", "model"])["auprc"]
          .agg(["mean", "std", "count"]).round(4).to_string())


if __name__ == "__main__":
    main()
