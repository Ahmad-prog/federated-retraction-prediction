"""Zero-shot transfer: train on our 138k corpus, test on Usman & Balke's
WebSci'25 corpus (232 retracted / 232 non-retracted, methodology-error cases).

Metadata for his papers comes from Europe PMC (OpenAlex budget-free).
Features missing there (early citations, countries) are NaN -> median-imputed
by the pipeline fitted on our training corpus.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.build_features import NUMERIC_FEATURES, text_features
from src.fl.engine import Client, run_fl
from src.models.baselines import make_xy, preprocessor, train_mlp, train_rf, train_xgb
from src.models.evaluate import evaluate
from src.models.mlp import MLP, set_weights
from src.partition.partitioner import partition
import torch

EPMC = "https://www.ebi.ac.uk/europepmc/webservice/rest"
CACHE = ROOT / "data" / "cache" / "usman_epmc"
CACHE.mkdir(parents=True, exist_ok=True)


def epmc_meta(pmcid: str) -> dict:
    cfile = CACHE / f"{pmcid}.json"
    if cfile.exists():
        return json.loads(cfile.read_text())
    out = {}
    try:
        r = requests.get(f"{EPMC}/search", params={
            "query": f"PMCID:{pmcid}", "format": "json", "resultType": "core"},
            timeout=30)
        res = (r.json().get("resultList") or {}).get("result") or []
        if res:
            w = res[0]
            out = {
                "n_authors": len(((w.get("authorList") or {}).get("author")) or []),
                "year": int(w.get("pubYear")) if w.get("pubYear") else None,
                "cited_by_count": w.get("citedByCount"),
                "is_oa": 1.0 if w.get("isOpenAccess") == "Y" else 0.0,
            }
        r2 = requests.get(f"{EPMC}/PMC/{pmcid.replace('PMC','')}/references",
                          params={"format": "json", "pageSize": 1}, timeout=30)
        out["n_references"] = r2.json().get("hitCount")
    except Exception:
        pass
    cfile.write_text(json.dumps(out))
    time.sleep(0.1)
    return out


def main() -> None:
    raw = pd.read_csv(
        ROOT / "data/raw/Scientific-Accountability-main/Retracted_NonRetracted Articles.csv",
        sep=";", engine="python", quotechar='"', on_bad_lines="skip")
    raw["retracted"] = (raw["Label"].str.strip() == "R").astype(float)
    print(f"Usman corpus rows: {len(raw)}, positives: {int(raw.retracted.sum())}")

    rows = []
    for i, rec in raw.iterrows():
        meta = epmc_meta(f"PMC{int(rec['pmcid'])}")
        tf = text_features(str(rec["abstract"]) if pd.notna(rec["abstract"]) else "")
        rows.append({
            "n_authors": meta.get("n_authors"), "n_countries": np.nan,
            "n_references": meta.get("n_references"),
            "cited_by_count": meta.get("cited_by_count"),
            "early_citations_2y": np.nan, "year": meta.get("year"),
            "is_oa": meta.get("is_oa"), "is_international": np.nan,
            **tf, "retracted": rec["retracted"],
        })
        if i % 100 == 0:
            print(f"  enriched {i}/{len(raw)}", flush=True)
    test = pd.DataFrame(rows)
    got = test["n_authors"].notna().sum()
    print(f"Europe PMC metadata found for {got}/{len(test)}")

    train = pd.read_parquet(ROOT / "data/processed/features.parquet")
    Xtr, ytr = make_xy(train)
    Xte = test[NUMERIC_FEATURES].to_numpy(dtype=np.float32)
    yte = test["retracted"].to_numpy(dtype=np.float32)

    results = {}
    for name, fn in [("central_rf", train_rf), ("central_xgb", train_xgb),
                     ("central_mlp", train_mlp)]:
        results[name] = fn(Xtr, ytr, Xte, yte, seed=42)

    # federated model trained across publisher silos, tested on Usman corpus
    part = partition(train, by="publisher", k=10)
    pre = preprocessor()
    Xtr_s = pre.fit_transform(Xtr).astype(np.float32)
    Xte_s = pre.transform(Xte).astype(np.float32)
    clients = []
    for silo, p in part.groupby("silo"):
        rowsel = part.index.get_indexer(p.index)
        clients.append(Client(str(silo), Xtr_s[rowsel], ytr[rowsel]))
    out = run_fl(clients, Xte_s, yte, strategy="fedavg", rounds=50, seed=42)
    results["fl_fedavg"] = out["final"]

    # balanced-set extras: accuracy at 0.5 for comparability with their 0.88
    model = MLP(Xte_s.shape[1])
    set_weights(model, out["weights"])
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(torch.tensor(Xte_s))).numpy()
    acc = float(((scores > 0.5) == (yte > 0.5)).mean())
    results["fl_fedavg"]["balanced_accuracy_at_0.5"] = acc

    res = pd.DataFrame(results).T
    res.to_csv(ROOT / "results" / "usman_transfer.csv")
    print(res.round(3).to_string())


if __name__ == "__main__":
    main()
