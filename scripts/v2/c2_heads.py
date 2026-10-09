"""C2: classifiers on frozen embeddings (R2.4): central / local / FedAvg / personalised.

--source abstracts : 138K corpus, B2 split (setting pub), features = embedding
                     [+ tabular publication-time features with --with-tab]
--source fulltext  : full-text corpus, the C3 split for the same seed (so heads and
                     LoRA fine-tuning are directly comparable), incl. prospective set
Runs on CPU (one process per seed). Output: results_v2/c2_heads/<tag>.csv + scores npz
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

OUT = RES2 / "c2_heads"
TAB = META + TEXT


def load(source: str, emb: str, seed: int, with_tab: bool, k: int):
    E = np.load(DATA2 / "emb" / f"{emb}.npy").astype(np.float32)
    ids = (DATA2 / "emb" / f"{emb}.ids.txt").read_text().split("\n")
    if source == "abstracts":
        df = pd.read_parquet(DATA2 / "tabular_v2.parquet").reset_index(drop=True)
        assert df["doi"].tolist() == ids, "embedding order must match tabular_v2"
        df["silo"] = partition(df, "publisher")
        df["part"] = split(df, seed)
    else:
        from scripts.v2.c3_lora_fl import load_dataset
        df = load_dataset(k, seed)
        pos = {p: i for i, p in enumerate(ids)}
        keep = df["pmcid"].map(pos).notna().to_numpy()
        df = df[keep].reset_index(drop=True)
        E = E[df["pmcid"].map(pos).astype(int).to_numpy()]
        for c in TAB:
            if c not in df:
                df[c] = np.nan
    X = E.astype(np.float64)
    if with_tab:
        X = np.hstack([X, df[TAB].to_numpy(np.float64)])
    return df, X


def run(source, emb, seed, with_tab, k, smoke):
    import torch
    from sklearn.linear_model import LogisticRegression

    from src.v2.fl import FedStandardizer, Silo, personalize, run_fl, scores, train_central_mlp
    torch.set_num_threads(2)
    df, X = load(source, emb, seed, with_tab, k)
    if smoke:
        keep = np.random.default_rng(seed).random(len(df)) < 0.05
        df, X = df[keep].reset_index(drop=True), X[keep]
    y = df["retracted"].to_numpy(np.float32)
    silos = sorted(df["silo"].unique())
    m = {p: (df["part"] == p).to_numpy() for p in ["train", "val", "test", "prospective"]}
    sm = {s: (df["silo"] == s).to_numpy() for s in silos}
    Z = FedStandardizer.fit([X[m["train"] & sm[s]] for s in silos]).transform(X)
    d = Z.shape[1]
    yte, ste = y[m["test"]], df.loc[m["test"], "silo"].to_numpy()
    tag = dict(source=source, emb=emb, with_tab=with_tab, seed=seed)
    rows, saved = [], {"y_test": yte, "silo_test": ste}

    def rec(model, sc, view="pooled", silo="ALL", sc_pros=None):
        msk = np.ones(len(yte), bool) if view == "pooled" else (
            ste == silo if view == "own_silo" else ste != silo)
        rows.append({**tag, "model": model, "view": view, "silo": silo,
                     **evaluate(yte[msk], sc[msk])})
        if sc_pros is not None and m["prospective"].sum() and len(np.unique(y[m["prospective"]])) > 1:
            rows.append({**tag, "model": model, "view": "prospective", "silo": silo,
                         **evaluate(y[m["prospective"]], sc_pros)})

    def rec_all(model, sc, sc_pros=None):
        rec(model, sc, sc_pros=sc_pros)
        for s in silos:
            rec(model, sc, "own_silo", s)
        saved[model] = sc

    E_, R_ = (4, 3) if smoke else (40, 20)
    Ztr, ytr, Zte, Zpr = Z[m["train"]], y[m["train"]], Z[m["test"]], Z[m["prospective"]]
    lr = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced").fit(Ztr, ytr)
    rec_all("central_logreg", lr.predict_proba(Zte)[:, 1],
            lr.predict_proba(Zpr)[:, 1] if len(Zpr) else None)
    cm = train_central_mlp(Ztr, ytr, Z[m["val"]], y[m["val"]], d, epochs=E_, seed=seed)
    rec_all("central_mlp", scores(cm, Zte), scores(cm, Zpr) if len(Zpr) else None)
    objs = [Silo(s, Z[m["train"] & sm[s]], y[m["train"] & sm[s]], Z[m["val"] & sm[s]],
                 y[m["val"] & sm[s]]) for s in silos]
    objs = [o for o in objs if o.n and 0 < o.prev < 1 and len(o.yva)]
    for o in objs:
        lm = train_central_mlp(o.Xtr, o.ytr, o.Xva, o.yva, d, epochs=E_, seed=seed)
        sc = scores(lm, Zte)
        rec("local_mlp", sc, "pooled", o.name)
        rec("local_mlp", sc, "own_silo", o.name)
        rec("local_mlp", sc, "out_silo", o.name)
    out = run_fl(objs, d, "fedavg", rounds=R_, seed=seed)
    rec_all("fl_fedavg", scores(out["model"], Zte), scores(out["model"], Zpr) if len(Zpr) else None)
    for o in objs:
        pm = personalize(out["model"], o, d, epochs=10 if not smoke else 1, seed=seed)
        rec("fl_fedavg_ft10", scores(pm, Zte), "own_silo", o.name)
    name = f"{source}_{emb}_{'tab' if with_tab else 'emb'}_s{seed}{'_smoke' if smoke else ''}"
    np.savez_compressed(OUT / f"scores_{name}.npz", **saved)
    pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["abstracts", "fulltext"], required=True)
    ap.add_argument("--emb", required=True, help="stem in data_v2/emb, e.g. abstracts_modernbert")
    ap.add_argument("--with-tab", action="store_true")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(run, a.source, a.emb, 42 + i, a.with_tab, a.k, a.smoke): i
                for i in range(a.seeds)}
        for f in as_completed(futs):
            try:
                rows.extend(f.result())
            except Exception as e:  # noqa: BLE001
                print(f"seed {42 + futs[f]} FAILED: {e!r}", flush=True)
    r = pd.DataFrame(rows)
    print(r.groupby(["view", "model"])["auprc"].agg(["mean", "std", "count"]).round(4).to_string())


if __name__ == "__main__":
    main()
