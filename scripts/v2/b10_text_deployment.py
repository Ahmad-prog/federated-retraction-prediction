"""B10: deployment-facing evaluation of the TEXT models (same protocol as B5 for tabular):
realistic prevalence, false-positive rate by first-author country at 5% overall FPR,
recall by retraction reason. Prospective AUPRC comes from the c3 result files.

Input: results_v2/c3_lora_abstracts*/scores_<regime>_s<seed>.npz (test order = B2 split)
Output: results_v2/b10_{prevalence,fairness_country,reasons,prospective}.csv
"""
from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.v2.b5_deployment import prevalence_table, reason_cat, test_frame, thr_at_fpr  # noqa: E402
from src.models.evaluate import evaluate  # noqa: E402
from src.v2 import RES2  # noqa: E402


def main() -> None:
    prev_rows, fair_rows, reason_rows, pros_rows = [], [], [], []
    frames = {}
    for f in sorted(glob.glob(str(RES2 / "c3_lora_abstracts*" / "scores_*_s[0-9][0-9].npz"))):
        run = Path(f).parent.name
        seed = int(re.search(r"_s(\d+)\.npz$", f).group(1))
        z = np.load(f, allow_pickle=True)
        if seed not in frames:
            df = test_frame(seed)
            frames[seed] = df[df["part"] == "test"].reset_index(drop=True)
        te = frames[seed]
        y = z["y_test"]
        if len(te) != len(y) or not np.array_equal(te["retracted"].to_numpy(np.float32), y):
            print(f"skip {f}: test order mismatch")
            continue
        country = te["first_author_country"].fillna("").str.split(";").str[0].replace("", "unknown")
        top = country.value_counts().index[:10]
        country = country.where(country.isin(top), "other")
        reason = te["reason"].fillna("").map(reason_cat)
        for k in z.files:
            if not (k.endswith("__ALL__test") or k.endswith("__personal__test")):
                continue
            mdl = k.replace("__ALL__test", "").replace("__test", "")
            s = z[k]
            key = {"run": run, "seed": seed, "model": mdl}
            prev_rows.append({**key, **evaluate(y, s), **prevalence_table(y, s)})
            t = thr_at_fpr(y, s)
            flag = s > t
            for c in sorted(country.unique()):
                m = (country == c).to_numpy()
                fair_rows.append({**key, "country": c, "n_neg": int((y[m] == 0).sum()),
                                  "n_pos": int((y[m] == 1).sum()),
                                  "fpr": float(flag[m & (y == 0)].mean()) if (m & (y == 0)).any() else np.nan,
                                  "tpr": float(flag[m & (y == 1)].mean()) if (m & (y == 1)).any() else np.nan})
            for rc in sorted(reason[y == 1].unique()):
                m = (reason == rc).to_numpy() & (y == 1)
                reason_rows.append({**key, "reason": rc, "n": int(m.sum()),
                                    "recall@5%fpr": float(flag[m].mean())})
    for f in sorted(glob.glob(str(RES2 / "c3_lora_abstracts*" / "results_*_s[0-9][0-9].csv"))):
        r = pd.read_csv(f)
        r = r[r["view"] == "prospective"]
        if len(r):
            pros_rows.append(r.assign(run=Path(f).parent.name))
    out = {"prevalence": pd.DataFrame(prev_rows), "fairness_country": pd.DataFrame(fair_rows),
           "reasons": pd.DataFrame(reason_rows),
           "prospective": pd.concat(pros_rows) if pros_rows else pd.DataFrame()}
    for k, v in out.items():
        v.to_csv(RES2 / f"b10_{k}.csv", index=False)
    pd.set_option("display.width", 250)
    prev = out["prevalence"]
    if len(prev):
        cols = [c for c in prev.columns if c.startswith("prec@rec0.5") or c.startswith("prec@top1%")]
        print(prev.groupby(["run", "model"])[["auprc"] + cols].mean().round(3).to_string())
        fair = out["fairness_country"]
        print(fair.groupby(["run", "model", "country"])["fpr"].mean().unstack().round(3).to_string())
        rs = out["reasons"]
        print(rs.groupby(["run", "model", "reason"])["recall@5%fpr"].mean().unstack().round(3).to_string())
    if len(out["prospective"]):
        print(out["prospective"].groupby(["run", "model"])[["auprc", "roc_auc", "recall_at_5fpr"]]
              .agg(["mean", "count"]).round(3).to_string())


if __name__ == "__main__":
    main()
