"""B6: cross-corpus experiments against Usman & Balke's curated corpus, redone (R3.1-R3.4;
C5, C6, C7, C8).

Arenas (from the gap-fix work): clean180 = rows whose section features are not
placeholder values repeated > 5 times (on all 464 rows those placeholders leak the
label); audited172 = clean180 rows that also passed the manual audit.

  T4'  his RF on HIS OWN feature columns: all 10 vs abstract-only (abstract_Readability,
       Abstract_Certainty), his protocol (80/20, seed 42, fillna 0, scaler) -> ACCURACY,
       plus 5-fold CV x 3 seeds on audited172 (accuracy + AUPRC)
  T1'  curated -> ours: RF on curated abstracts (our text features) -> our test set;
       size control: RF on n random articles of OUR corpus (n = len(curated)) -> same test
  T2'/T3'  ours -> curated with the SAME model in every arm (XGBoost and FedAvg MLP),
       feature sets text-only / metadata-only / all, metadata from Europe PMC (fixed URL)
Output: results_v2/b6_cross_corpus.csv
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.v2.b2_main_benchmark import META, TEXT, partition, split, xgb_model  # noqa: E402
from src.features.build_features import text_features  # noqa: E402
from src.models.evaluate import evaluate  # noqa: E402
from src.v2 import DATA2, RES2, ROOT  # noqa: E402
from src.v2.cleaning import clean_abstract  # noqa: E402
from src.v2.epmc import cached_json, get, search  # noqa: E402

HIS = ["abstract_Readability", "Method_Readability", "Results_Readability",
       "Conclusion_Readability", "Avg_Readability", "Avg_Certainty", "Abstract_Certainty",
       "Method_Certainty", "Result_Certainty", "Conclusion_Certainty"]
HIS_ABS = ["abstract_Readability", "Abstract_Certainty"]
CACHE = DATA2 / "cache" / "usman_epmc_v2"
CACHE.mkdir(parents=True, exist_ok=True)


def load_arenas():
    df = pd.read_csv(ROOT / "data/raw/Scientific-Accountability-main/Retracted_NonRetracted Articles.csv",
                     sep=";", engine="python", quotechar='"', on_bad_lines="skip")
    df["Label"] = df["Label"].str.strip().map({"R": 1, "NR": 0})
    for c in HIS:
        df[c] = df[c].astype(str).str.replace(",", ".").astype(float)
    df = df.reset_index(drop=True)
    mask = pd.Series(False, index=df.index)
    for c in HIS:
        vc = df[c].value_counts()
        mask |= df[c].isin(vc[vc > 5].index)
    clean180 = df[~mask].reset_index(drop=True)
    audited = set(pd.read_csv(ROOT / "data/processed/usman_clean_subset.csv")["pmcid"])
    audited172 = clean180[clean180["pmcid"].isin(audited)].reset_index(drop=True)
    return df, clean180, audited172


def epmc_meta(pmcid: int) -> dict:
    def fetch():
        r = search(f"PMCID:PMC{pmcid}", page_size=1, result_type="core")
        res = r["resultList"].get("result", [])
        if not res:
            return {"found": False}
        w = res[0]
        auths = (w.get("authorList") or {}).get("author") or []
        refs = get(f"PMC/PMC{pmcid}/references", {"format": "json", "pageSize": 1}, allow_404=True)
        return {"found": True, "n_authors": len(auths),
                "year": int(w["pubYear"]) if w.get("pubYear") else None,
                "is_oa": 1.0 if w.get("isOpenAccess") == "Y" else 0.0,
                "n_references": (refs.json().get("hitCount") if refs is not None else None),
                "n_countries": len({(a.get("authorAffiliationDetailsList") or {}).get(
                    "authorAffiliation", [{}])[0].get("affiliation", "").split(",")[-1].strip()
                    for a in auths if a.get("authorAffiliationDetailsList")}) or None}
    return cached_json(CACHE / f"PMC{pmcid}.json", fetch)


def his_protocol(df, cols):
    X, y = df[cols].fillna(0).to_numpy(), df["Label"].to_numpy()
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42)
    sc = StandardScaler().fit(X[tr])
    clf = RandomForestClassifier(random_state=42).fit(sc.transform(X[tr]), y[tr])
    s = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return {"accuracy": accuracy_score(y[te], s > 0.5), **evaluate(y[te], s)}


def cv(df, cols, seeds=(0, 1, 2)):
    X, y = df[cols].fillna(0).to_numpy(), df["Label"].to_numpy()
    accs, aps = [], []
    for sd in seeds:
        s = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(X, y):
            sc = StandardScaler().fit(X[tr])
            s[te] = RandomForestClassifier(n_estimators=300, random_state=sd).fit(
                sc.transform(X[tr]), y[tr]).predict_proba(sc.transform(X[te]))[:, 1]
        accs.append(accuracy_score(y, s > 0.5))
        aps.append(evaluate(y, s)["auprc"])
    return {"accuracy": float(np.mean(accs)), "accuracy_sd": float(np.std(accs)),
            "auprc": float(np.mean(aps))}


def our_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a in df["abstract"].fillna(""):
        txt, _, _ = clean_abstract(a)
        tf = text_features(txt) if len(txt) >= 40 else {}
        tf["has_abstract"] = float(len(txt) >= 40)
        rows.append(tf)
    out = pd.DataFrame(rows, index=df.index)
    if "abstract_len_words" in out:
        out.loc[out["has_abstract"] == 0, "abstract_len_words"] = np.nan
    return out


def main() -> None:
    from src.v2.fl import FedStandardizer, Silo, run_fl, scores
    df_all, clean180, audited172 = load_arenas()
    rows = []
    print(f"curated corpus: all {len(df_all)}, clean180 {len(clean180)}, audited172 {len(audited172)}")

    # ---- T4': his own features, all vs abstract-only
    for arena, d in [("all464", df_all), ("clean180", clean180)]:
        for name, cols in [("his_all_10", HIS), ("his_abstract_only", HIS_ABS)]:
            rows.append({"exp": "T4_his_protocol", "arena": arena, "features": name,
                         **his_protocol(d, cols)})
    for name, cols in [("his_all_10", HIS), ("his_abstract_only", HIS_ABS)]:
        rows.append({"exp": "T4_cv5x3", "arena": "audited172", "features": name, **cv(audited172, cols)})

    # ---- curated corpus in our feature space (metadata from Europe PMC, fixed URL)
    with ThreadPoolExecutor(max_workers=6) as ex:
        meta = list(ex.map(epmc_meta, clean180["pmcid"].astype(int)))
    found = sum(m.get("found", False) for m in meta)
    print(f"Europe PMC metadata found for {found}/{len(clean180)} curated articles")
    cur = pd.concat([our_features(clean180),
                     pd.DataFrame([{k: m.get(k) for k in ["n_authors", "n_countries", "n_references",
                                                          "is_oa", "year"]} for m in meta])], axis=1)
    cur["retracted"] = clean180["Label"].to_numpy()

    ours = pd.read_parquet(DATA2 / "tabular_v2.parquet").reset_index(drop=True)
    ours["silo"] = partition(ours, "publisher")
    for seed in range(42, 47):
        ours["part"] = split(ours, seed)
        tr, te = ours[ours.part == "train"], ours[ours.part == "test"]
        # T1': curated -> ours (text features only: computable identically on both sides)
        for name, Xtr, ytr in [("curated_180", cur[TEXT], cur["retracted"])] + [
                (f"ours_random_{len(cur)}", s[TEXT], s["retracted"]) for s in
                [tr.sample(n=len(cur), random_state=seed)]] + [
                ("ours_random_464", tr.sample(n=464, random_state=seed)[TEXT],
                 tr.sample(n=464, random_state=seed)["retracted"])]:
            clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed)
            clf.fit(Xtr.fillna(Xtr.median()).fillna(0), ytr)
            s = clf.predict_proba(te[TEXT].fillna(Xtr.median()).fillna(0))[:, 1]
            rows.append({"exp": "T1_to_ours", "train": name, "seed": seed, **evaluate(
                te["retracted"].to_numpy(), s)})
        # T2'/T3': ours -> curated, same model in every arm
        for fname, feats in [("text_only", TEXT), ("meta_only", META), ("all", META + TEXT)]:
            Xtr = tr[feats].to_numpy(np.float64)
            spw = float((tr.retracted == 0).sum() / (tr.retracted == 1).sum())
            silos = sorted(tr["silo"].unique())
            std = FedStandardizer.fit([Xtr[(tr.silo == s).to_numpy()] for s in silos])
            Ztr, Zcur = std.transform(Xtr), std.transform(cur[feats].to_numpy(np.float64))
            s_x = xgb_model(seed, spw).fit(Ztr, tr["retracted"]).predict_proba(Zcur)[:, 1]
            va = ours[ours.part == "val"]
            Zva = std.transform(va[feats].to_numpy(np.float64))
            objs = [Silo(s, Ztr[(tr.silo == s).to_numpy()], tr.loc[tr.silo == s, "retracted"].to_numpy(np.float32),
                         Zva[(va.silo == s).to_numpy()], va.loc[va.silo == s, "retracted"].to_numpy(np.float32))
                    for s in silos]
            out = run_fl([o for o in objs if o.n and len(o.yva)], Ztr.shape[1], "fedavg",
                         rounds=50, seed=seed)
            s_f = scores(out["model"], Zcur.astype(np.float32))
            for model, s in [("central_xgb", s_x), ("fl_fedavg", s_f)]:
                rows.append({"exp": "T23_ours_to_curated", "features": fname, "model": model,
                             "seed": seed, "accuracy_at_0.5": accuracy_score(cur["retracted"], s > 0.5),
                             **evaluate(cur["retracted"].to_numpy(), s)})
        print(f"seed {seed} done", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(RES2 / "b6_cross_corpus.csv", index=False)
    pd.set_option("display.width", 200)
    for exp, g in res.groupby("exp"):
        keys = [c for c in ["arena", "features", "train", "model"] if c in g and g[c].notna().any()]
        cols = [c for c in ["accuracy", "accuracy_sd", "accuracy_at_0.5", "auprc", "roc_auc"] if c in g]
        print(f"\n== {exp} ==")
        print(g.groupby(keys)[cols].mean().round(3).to_string())
    json.dump({"epmc_found": found, "n_clean180": len(clean180)},
              open(RES2 / "b6_meta_coverage.json", "w"))


if __name__ == "__main__":
    main()
