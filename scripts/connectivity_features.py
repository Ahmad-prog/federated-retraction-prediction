"""Contribution 2 candidate: connectivity features on the Usman corpus.

Per paper (time-aware, only events before its publication year):
  - n_prior_author_retractions / any_prior_author_retraction
    (author appears in Retraction Watch with an earlier retraction)
  - n_refs_retracted_prior / frac_refs_retracted_prior
    (references already retracted at publication time; PMID matching)

Data: Europe PMC (authors, pubYear, reference lists) — cached per paper.
Then the ladder: baseline / +profiles / +connectivity / +both, HP + CV arenas.
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.gapfix_train import HIS_COLS, load_arenas, persist  # noqa: E402

EPMC = "https://www.ebi.ac.uk/europepmc/webservice/rest"
CACHE = ROOT / "data/cache/usman_connectivity"
CACHE.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 43, 44]


def epmc(pmcid: str) -> dict:
    f = CACHE / f"{pmcid}.json"
    if f.exists():
        return json.loads(f.read_text())
    out = {"authors": [], "year": None, "ref_pmids": []}
    try:
        r = requests.get(f"{EPMC}/search", params={
            "query": f"PMCID:{pmcid}", "format": "json", "resultType": "core"}, timeout=30)
        res = (r.json().get("resultList") or {}).get("result") or []
        if res:
            w = res[0]
            out["year"] = int(w["pubYear"]) if w.get("pubYear") else None
            out["authors"] = [a.get("fullName", "") for a in
                              ((w.get("authorList") or {}).get("author")) or []]
        r2 = requests.get(f"{EPMC}/PMC/{pmcid.replace('PMC','')}/references",
                          params={"format": "json", "pageSize": 500}, timeout=30)
        refs = (r2.json().get("referenceList") or {}).get("reference") or []
        out["ref_pmids"] = [str(x["id"]) for x in refs if x.get("id") and x.get("source") == "MED"]
    except Exception:
        pass
    f.write_text(json.dumps(out))
    time.sleep(0.1)
    return out


def norm_name(n: str) -> str:
    n = re.sub(r"[^a-z ]", "", str(n).lower().strip())
    p = [x for x in n.split() if x]
    return f"{p[-1]} {p[0][0]}" if len(p) > 1 else (p[0] if p else "")


def build_rw_indexes():
    rw = pd.read_csv(ROOT / "data/raw/retraction_watch.csv", low_memory=False,
                     encoding_errors="replace")
    rw = rw[rw["RetractionNature"].str.strip().str.lower() == "retraction"]
    rw["ryear"] = pd.to_datetime(rw["RetractionDate"], errors="coerce").dt.year
    rw = rw.dropna(subset=["ryear"])
    author_idx = defaultdict(list)
    for a, y in zip(rw["Author"], rw["ryear"]):
        if pd.isna(a):
            continue
        for name in str(a).split(";"):
            k = norm_name(name)
            if k:
                author_idx[k].append(int(y))
    pmid_ryear = {}
    for pmid, y in zip(rw["OriginalPaperPubMedID"], rw["ryear"]):
        try:
            pmid_ryear[str(int(float(pmid)))] = int(y)
        except (ValueError, TypeError):
            pass
    return author_idx, pmid_ryear


def main() -> None:
    df_all, clean180, audited172 = load_arenas()
    author_idx, pmid_ryear = build_rw_indexes()

    feats = {}
    for i, row in df_all.iterrows():
        meta = epmc(f"PMC{int(row['pmcid'])}")
        year = meta["year"] or 0
        n_auth = sum(1 for a in meta["authors"]
                     for _ in [1] if any(y < year for y in author_idx.get(norm_name(a), [])))
        refs = meta["ref_pmids"]
        n_ret = sum(1 for p in refs if pmid_ryear.get(p, 9999) < year)
        feats[i] = {"n_prior_auth_ret": float(n_auth),
                    "any_prior_auth_ret": 1.0 if n_auth else 0.0,
                    "n_refs_ret_prior": float(n_ret),
                    "frac_refs_ret_prior": n_ret / len(refs) if refs else 0.0,
                    "n_refs_found": float(len(refs))}
        if i % 100 == 0:
            print(f"  connectivity {i}/{len(df_all)}", flush=True)
    conn = pd.DataFrame.from_dict(feats, orient="index").sort_index()
    conn.to_parquet(ROOT / "data/processed/usman_connectivity.parquet")
    CONN = list(conn.columns)

    gf = pd.read_parquet(ROOT / "data/processed/gapfix_features.parquet")
    PROF = [c for c in gf.columns if "_cert_" in c and not c.endswith("mean")] + \
           ["traj_abs_minus_methods", "traj_abs_minus_results"]

    def matrix(rows, cols):
        parts = pd.concat([rows.reset_index(drop=True)[HIS_COLS],
                           gf.iloc[rows["row_id"]].reset_index(drop=True),
                           conn.iloc[rows["row_id"]].reset_index(drop=True)], axis=1)
        return parts.reindex(columns=cols).to_numpy(np.float32)

    print("class means (audited):", flush=True)
    a = pd.concat([audited172.reset_index(drop=True)[["Label"]],
                   conn.iloc[audited172["row_id"]].reset_index(drop=True)], axis=1)
    print(a.groupby("Label")[CONN].mean().round(3).to_string(), flush=True)

    ladders = [("L_baseline", HIS_COLS),
               ("L_profiles", HIS_COLS + PROF),
               ("L_connectivity", HIS_COLS + CONN),
               ("L_full_fusion", HIS_COLS + PROF + CONN)]
    xgb = lambda s: XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.06,
                                  subsample=0.9, colsample_bytree=0.9,
                                  random_state=s, eval_metric="logloss", n_jobs=12)
    rf = lambda s: RandomForestClassifier(random_state=s)
    for name, cols in ladders:
        for mtag, mk in [("RF", rf), ("XGB", xgb)]:
            out = []
            X = matrix(clean180, cols)
            y = clean180["Label"].to_numpy()
            tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42)
            imp, sc = SimpleImputer(strategy="median"), StandardScaler()
            clf = mk(42)
            clf.fit(sc.fit_transform(imp.fit_transform(X[tr])), y[tr])
            s = clf.predict_proba(sc.transform(imp.transform(X[te])))[:, 1]
            out.append({"model": f"{name}_{mtag}", "arena": "HP",
                        "acc": accuracy_score(y[te], s > 0.5),
                        "f1": f1_score(y[te], s > 0.5), "auc": roc_auc_score(y[te], s)})
            X = matrix(audited172, cols)
            y = audited172["Label"].to_numpy()
            accs = []
            for seed in SEEDS:
                scores = np.zeros(len(y))
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
                    imp, sc = SimpleImputer(strategy="median"), StandardScaler()
                    clf = mk(seed)
                    clf.fit(sc.fit_transform(imp.fit_transform(X[tr])), y[tr])
                    scores[te] = clf.predict_proba(sc.transform(imp.transform(X[te])))[:, 1]
                accs.append(accuracy_score(y, scores > 0.5))
            out.append({"model": f"{name}_{mtag}", "arena": "CV",
                        "acc": float(np.mean(accs)), "acc_std": float(np.std(accs))})
            persist(out)


if __name__ == "__main__":
    main()
