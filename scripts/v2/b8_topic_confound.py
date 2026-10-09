"""B8: is the text signal more than topic? (paper-mill topic confound)

1. Topic-only baselines: one-hot OpenAlex field + subfield (243 subfields), central
   logistic regression and XGBoost, same B2 split -> how far does topic alone go?
2. Within-topic discrimination for every scored model (B2 tabular, C3 LoRA text):
   ROC-AUC and AUPRC lift computed inside each subfield (>= 20 positives and 20 negatives
   in the test set) and averaged with positive-count weights. A model that only knew the
   topic would score 0.5 ROC-AUC inside every subfield.
3. Flag concentration: share of articles flagged at 5% FPR that fall in the 5 subfields
   with most positives, vs their share of positives (does the model flag whole topics?).
Output: results_v2/b8_topic_*.csv
"""
from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.v2.b2_main_benchmark import partition, split, xgb_model  # noqa: E402
from scripts.v2.b5_deployment import thr_at_fpr  # noqa: E402
from src.models.evaluate import evaluate  # noqa: E402
from src.v2 import RES2, DATA2  # noqa: E402

MIN_N = 20


def frame(seed):
    df = pd.read_parquet(DATA2 / "tabular_v2.parquet").reset_index(drop=True)
    df["silo"] = partition(df, "publisher")
    df["part"] = split(df, seed)
    df["subfield"] = df["subfield"].fillna("unknown")
    df["field"] = df["field"].fillna("unknown")
    return df


def within(te, y, s):
    rows = []
    for sf, g in te.groupby("subfield"):
        i = g.index.to_numpy()
        npos, nneg = int(y[i].sum()), int((y[i] == 0).sum())
        if npos >= MIN_N and nneg >= MIN_N:
            rows.append({"subfield": sf, "n_pos": npos, "n_neg": nneg,
                         "auc": roc_auc_score(y[i], s[i]),
                         "lift": average_precision_score(y[i], s[i]) / y[i].mean()})
    w = pd.DataFrame(rows)
    return w, {"within_auc": float(np.average(w.auc, weights=w.n_pos)),
               "within_lift": float(np.average(w.lift, weights=w.n_pos)),
               "n_subfields": len(w), "pos_covered": float(w.n_pos.sum() / y.sum())}


def scored_models(seed):
    """(name, scores) on the B2 test order for this seed."""
    out = []
    f = RES2 / "b2_main" / f"scores_pub_publisher_s{seed}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        for k in ["central_xgb", "central_mlp", "fl_fedavg"]:
            if k in z:
                out.append((f"tabular/{k}", z[k]))
    for f in sorted(glob.glob(str(RES2 / "c3_lora_abstracts*" / f"scores_*_s{seed}.npz"))):
        z = np.load(f, allow_pickle=True)
        for k in z.files:
            if k.endswith("__ALL__test") or k.endswith("__personal__test"):
                out.append((f"{Path(f).parent.name}/{k.replace('__ALL__test', '').replace('__test', '')}", z[k]))
    return out


def main() -> None:
    base_rows, within_rows, conc_rows, per_sf = [], [], [], []
    for seed in range(42, 52):
        df = frame(seed)
        tr, te = df[df.part == "train"], df[df.part == "test"].reset_index(drop=True)
        y = te["retracted"].to_numpy(np.float32)
        oh = pd.get_dummies(df[["field", "subfield"]].astype(str), sparse=False).to_numpy(np.float32)
        Xtr, Xte = oh[(df.part == "train").to_numpy()], oh[(df.part == "test").to_numpy()]
        ytr = tr["retracted"].to_numpy(np.float32)
        models = [("topic_only/logreg", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
                   .fit(Xtr, ytr).predict_proba(Xte)[:, 1]),
                  ("topic_only/xgb", xgb_model(seed, float((ytr == 0).sum() / (ytr == 1).sum()))
                   .fit(Xtr, ytr).predict_proba(Xte)[:, 1])]
        top5 = te[y == 1]["subfield"].value_counts().index[:5]
        in_top = te["subfield"].isin(top5).to_numpy()
        for name, s in models + scored_models(seed):
            if len(s) != len(y):
                continue
            w, agg = within(te, y, s)
            base_rows.append({"seed": seed, "model": name, **evaluate(y, s), **agg})
            w["seed"], w["model"] = seed, name
            per_sf.append(w)
            flag = s > thr_at_fpr(y, s)
            conc_rows.append({"seed": seed, "model": name,
                              "top5_share_of_positives": float(in_top[y == 1].mean()),
                              "top5_share_of_flagged": float(in_top[flag].mean()),
                              "top5_share_of_false_alarms": float(in_top[flag & (y == 0)].mean()),
                              "top5_share_of_negatives": float(in_top[y == 0].mean())})
        print(f"seed {seed} done", flush=True)
    res = pd.DataFrame(base_rows)
    res.to_csv(RES2 / "b8_topic_summary.csv", index=False)
    pd.concat(per_sf).to_csv(RES2 / "b8_topic_per_subfield.csv", index=False)
    conc = pd.DataFrame(conc_rows)
    conc.to_csv(RES2 / "b8_topic_concentration.csv", index=False)
    pd.set_option("display.width", 220)
    print(res.groupby("model")[["auprc", "roc_auc", "within_auc", "within_lift", "n_subfields",
                                "pos_covered"]].agg(["mean", "count"]).round(3).to_string())
    print(conc.groupby("model").mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
