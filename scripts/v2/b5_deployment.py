"""B5: deployment-facing evaluation (R0.7, R1.7; M11, fairness, per-reason, prospective).

Uses the saved B2 test scores (setting pub, publisher partition, 10 seeds) plus a
prospective test on articles retracted after the v1 snapshot (2026-07-19).
  prevalence  precision at fixed recall / top-k under realistic retraction rates:
              precision(pi) = pi*TPR / (pi*TPR + (1-pi)*FPR), FPR from the matched controls
  fairness    FPR and TPR by first-author country at the threshold giving 5% overall FPR
  reasons     recall at 5% FPR by Retraction Watch reason category
  prospective models trained on the B2 train split, scored on the 2026-07..10 retractions
              (OpenAlex features, same cleaning) vs held-out test controls of the same publishers
Output: results_v2/b5_*.csv
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.v2.b1_tabular_features import row as oa_row  # noqa: E402
from scripts.v2.b2_main_benchmark import META, TEXT, partition, split, xgb_model  # noqa: E402
from src.features.build_features import text_features  # noqa: E402
from src.models.evaluate import evaluate  # noqa: E402
from src.v2 import DATA2, RES2  # noqa: E402
from src.v2.cleaning import clean_abstract, clean_title, is_english, is_nonresearch  # noqa: E402

B2 = RES2 / "b2_main"
MODELS = ["central_xgb", "central_mlp", "fl_fedavg", "fl_xgb_cyclic", "central_logreg"]
REASONS = [("paper mill", r"paper mill"), ("fake peer review", r"fake peer review|peer review"),
           ("image issues", r"image"), ("fabrication/falsification", r"fabricat|falsif"),
           ("plagiarism/duplication", r"plagiar|duplicat"), ("data/results errors", r"error|unreliable"),
           ("authorship/ethics", r"author|ethic|consent|approval")]


def test_frame(seed: int) -> pd.DataFrame:
    df = pd.read_parquet(DATA2 / "tabular_v2.parquet").reset_index(drop=True)
    df["silo"] = partition(df, "publisher")
    df["part"] = split(df, seed)
    return df


def thr_at_fpr(y, s, fpr_target=0.05):
    neg = np.sort(s[y == 0])[::-1]
    return neg[int(np.floor(fpr_target * len(neg)))]


def prevalence_table(y, s, pis=(None, 0.05, 0.01, 0.002)):
    fpr, tpr, thr = roc_curve(y, s)
    out = {}
    for pi in pis:
        p = y.mean() if pi is None else pi
        prec = p * tpr / np.maximum(p * tpr + (1 - p) * fpr, 1e-12)
        key = "matched" if pi is None else f"{pi:g}"
        for rec in (0.1, 0.25, 0.5):
            i = np.searchsorted(tpr, rec)
            out[f"prec@rec{rec}_pi{key}"] = float(prec[min(i, len(prec) - 1)])
        # top 1% of articles flagged: share that are retracted
        k_rate = 0.01
        flag = p * tpr + (1 - p) * fpr
        j = np.searchsorted(flag, k_rate)
        out[f"prec@top1%_pi{key}"] = float(prec[min(j, len(prec) - 1)])
    return out


def reason_cat(r: str) -> str:
    r = (r or "").lower()
    for name, pat in REASONS:
        if re.search(pat, r):
            return name
    return "other/unspecified"


def main() -> None:
    prev_rows, fair_rows, reason_rows = [], [], []
    for seed in range(42, 52):
        f = B2 / f"scores_pub_publisher_s{seed}.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        df = test_frame(seed)
        te = df[df["part"] == "test"].reset_index(drop=True)
        y = z["y_test"]
        assert len(te) == len(y) and np.array_equal(te["retracted"].to_numpy(np.float32), y)
        country = te["first_author_country"].fillna("").str.split(";").str[0].replace("", "unknown")
        top = country.value_counts().index[:10]
        country = country.where(country.isin(top), "other")
        reason = te["reason"].fillna("").map(reason_cat)
        for mdl in MODELS:
            if mdl not in z:
                continue
            s = z[mdl]
            prev_rows.append({"seed": seed, "model": mdl, **evaluate(y, s), **prevalence_table(y, s)})
            t = thr_at_fpr(y, s)
            flag = s > t
            for c in sorted(country.unique()):
                m = (country == c).to_numpy()
                fair_rows.append({"seed": seed, "model": mdl, "country": c,
                                  "n_neg": int((y[m] == 0).sum()), "n_pos": int((y[m] == 1).sum()),
                                  "fpr": float(flag[m & (y == 0)].mean()) if (m & (y == 0)).any() else np.nan,
                                  "tpr": float(flag[m & (y == 1)].mean()) if (m & (y == 1)).any() else np.nan})
            for rc in sorted(reason[y == 1].unique()):
                m = (reason == rc).to_numpy() & (y == 1)
                reason_rows.append({"seed": seed, "model": mdl, "reason": rc, "n": int(m.sum()),
                                    "recall@5%fpr": float(flag[m].mean())})
    prev = pd.DataFrame(prev_rows)
    prev.to_csv(RES2 / "b5_prevalence.csv", index=False)
    fair = pd.DataFrame(fair_rows)
    fair.to_csv(RES2 / "b5_fairness_country.csv", index=False)
    rs = pd.DataFrame(reason_rows)
    rs.to_csv(RES2 / "b5_reasons.csv", index=False)
    pd.set_option("display.width", 220)
    cols = [c for c in prev.columns if c.startswith("prec@rec0.5") or c.startswith("prec@top1%")]
    print(prev.groupby("model")[["auprc"] + cols].mean().round(3).to_string())
    print(fair[fair.model == "fl_fedavg"].groupby("country")[["n_neg", "n_pos", "fpr", "tpr"]]
          .mean().round(3).sort_values("fpr").to_string())
    print(rs[rs.model == "fl_fedavg"].groupby("reason")[["n", "recall@5%fpr"]].mean().round(3).to_string())

    prospective()


def prospective() -> None:
    """Retractions added after 2026-07-19 vs held-out controls of the same publishers."""
    from src.v2.fl import FedStandardizer, Silo, run_fl, scores
    pros = pd.read_parquet(DATA2 / "prospective_positives.parquet")
    cache = DATA2 / "cache" / "openalex_prospective.json"
    if cache.exists():
        works = json.loads(cache.read_text())
    else:
        works, dois = [], sorted(set(pros["doi"].dropna()))
        for i in range(0, len(dois), 50):
            r = requests.get("https://api.openalex.org/works",
                             params={"filter": "doi:" + "|".join(dois[i:i + 50]), "per-page": 50},
                             timeout=90)
            r.raise_for_status()
            works += r.json().get("results", [])
            time.sleep(0.5)
        cache.write_text(json.dumps(works))
    p = pd.DataFrame([oa_row(w) for w in works]).drop_duplicates("doi")
    p = p[[not is_nonresearch(t, a) and is_english(f"{t} {a}")
           for t, a in zip(p["raw_title"], p["raw_abstract"])]]
    p["title"] = p["raw_title"].map(clean_title)
    ab = [clean_abstract(a) for a in p["raw_abstract"]]
    p["abstract"] = [x[0] for x in ab]
    p["has_abstract"] = (p["abstract"].str.len() >= 40).astype(float)
    tf = pd.DataFrame([text_features(a) if len(a) >= 40 else {} for a in p["abstract"]], index=p.index)
    p = pd.concat([p, tf], axis=1)
    p.loc[p["has_abstract"] == 0, "abstract_len_words"] = np.nan
    p["retracted"] = 1
    print(f"\nprospective positives with OpenAlex records (research, English): {len(p)}")
    rows = []
    for seed in range(42, 47):
        df = test_frame(seed)
        tr, va, te = (df[df.part == x] for x in ("train", "val", "test"))
        ctl = te[(te.retracted == 0) & te["publisher"].isin(set(p["publisher"].dropna()))]
        test = pd.concat([p.assign(silo="prospective"), ctl], ignore_index=True)
        feats = META + TEXT
        silos = sorted(tr["silo"].unique())
        std = FedStandardizer.fit([tr.loc[tr.silo == s, feats].to_numpy(np.float64) for s in silos])
        Ztr, Zte = std.transform(tr[feats].to_numpy(np.float64)), std.transform(test[feats].to_numpy(np.float64))
        Zva = std.transform(va[feats].to_numpy(np.float64))
        ytr, yte = tr["retracted"].to_numpy(np.float32), test["retracted"].to_numpy(np.float32)
        spw = float((ytr == 0).sum() / (ytr == 1).sum())
        rows.append({"seed": seed, "model": "central_xgb",
                     **evaluate(yte, xgb_model(seed, spw).fit(Ztr, ytr).predict_proba(Zte)[:, 1])})
        objs = [Silo(s, Ztr[(tr.silo == s).to_numpy()], ytr[(tr.silo == s).to_numpy()],
                     Zva[(va.silo == s).to_numpy()], va.loc[va.silo == s, "retracted"].to_numpy(np.float32))
                for s in silos]
        out = run_fl([o for o in objs if o.n and len(o.yva)], Ztr.shape[1], "fedavg", rounds=50, seed=seed)
        rows.append({"seed": seed, "model": "fl_fedavg", **evaluate(yte, scores(out["model"], Zte))})
        rows[-1]["n_pos"], rows[-1]["n_neg"] = int(yte.sum()), int((yte == 0).sum())
    r = pd.DataFrame(rows)
    r.to_csv(RES2 / "b5_prospective.csv", index=False)
    print(r.groupby("model")[["auprc", "roc_auc", "recall_at_5fpr"]].agg(["mean", "std"]).round(3).to_string())
    print(f"positive rate in prospective test: {r['n_pos'].dropna().iloc[0]} / "
          f"{r['n_pos'].dropna().iloc[0] + r['n_neg'].dropna().iloc[0]}")


if __name__ == "__main__":
    main()
