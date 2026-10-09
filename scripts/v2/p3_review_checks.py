"""P3: robustness checks requested in review, computed from saved test scores (no re-training).

  prospective   future-retraction test with matched inputs: positives and controls that both have an
                abstract, and both title-only (152 of the 213 prospective positives have no abstract)
  journal       within-journal ROC-AUC (venues with >= MIN_V test articles of each class)
  prior         central model + per-publisher prior (log-odds shift by the publisher's training
                prevalence), compared with the "each publisher scores its own articles" views
  paired        paired bootstrap of AUPRC differences between regimes on the same test articles
  format        formatting-only classifier (lengths, casing, punctuation, markup; no words) and
                missing-abstract rates by label
  neardup       MinHash near-duplicates between training and test abstracts; AUPRC without them

Writes results_v2/p3_review/*.csv and summary.json.
Usage: python scripts/v2/p3_review_checks.py [--boot 1000]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.v2.b2_main_benchmark import partition, split  # noqa: E402

RES = ROOT / "results_v2"
OUT = RES / "p3_review"
FAMS = {"c3_lora_abstracts": "ModernBERT-base", "c3_lora_abstracts_modernbert-large": "ModernBERT-large",
        "c3_lora_abstracts_qwen3-1.7b": "Qwen3-1.7B"}
SEEDS = [42, 43, 44]
MIN_V = 5
RNG = np.random.default_rng(0)


def corpus() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data_v2/tabular_v2.parquet").reset_index(drop=True)
    df["silo"] = partition(df, "publisher")
    return df


def scores(fam: str, run: str, seed: int):
    f = RES / fam / f"scores_{run}_s{seed}.npz"
    return np.load(f, allow_pickle=True) if f.exists() else None


def boot_ci(y, s, fn, n):
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(n):
        i = np.concatenate([RNG.choice(pos, len(pos)), RNG.choice(neg, len(neg))])
        vals.append(fn(y[i], s[i]))
    return np.percentile(vals, [2.5, 97.5])


# ------------------------------------------------------------------ prospective
def prospective(df, nboot):
    pro = pd.read_parquet(ROOT / "data_v2/prospective_abstracts.parquet")
    pro = pro[~pro["doi"].isin(set(df["doi"]))].reset_index(drop=True)
    has_p = (pro["has_abstract"] == 1).to_numpy()
    rows = []
    for fam, name in FAMS.items():
        for run, key in [("central", "central_lora"), ("fedavg", "fl_fedavg_lora")]:
            for seed in SEEDS:
                d = scores(fam, run, seed)
                if d is None or f"{key}__ALL__prospective" not in d.files:
                    continue
                sp, st = d[f"{key}__ALL__prospective"], d[f"{key}__ALL__test"]
                ctl = d["pros_ctl"] & (d["y_test"] == 0)
                has_c = (df.loc[d["test_idx"], "has_abstract"].to_numpy() == 1)
                for sub, mp, mc in [("all inputs (as reported)", np.ones_like(has_p), np.ones_like(has_c)),
                                    ("both with abstract", has_p, has_c),
                                    ("both title only", ~has_p, ~has_c)]:
                    s = np.concatenate([sp[mp], st[ctl & mc]])
                    y = np.concatenate([np.ones(mp.sum()), np.zeros((ctl & mc).sum())])
                    lo, hi = boot_ci(y, s, auc, nboot) if seed == 42 else (np.nan, np.nan)
                    rows.append({"model": name, "regime": run, "seed": seed, "subset": sub, "n_pos": int(mp.sum()),
                                 "n_neg": int((ctl & mc).sum()), "base_rate": y.mean(), "roc_auc": auc(y, s),
                                 "auprc": ap(y, s), "roc_lo": lo, "roc_hi": hi})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "prospective_matched.csv", index=False)
    comp = pro.groupby("publisher").size().sort_values(ascending=False)
    comp.to_frame("n").assign(with_abstract=pro.groupby("publisher").has_abstract.sum()).to_csv(OUT / "prospective_composition.csv")
    return out, comp


# ------------------------------------------------------------------ within journal
def within_journal(df):
    rows = []
    for fam, name in FAMS.items():
        for run, key in [("central", "central_lora__ALL__test"), ("fedavg", "fl_fedavg_lora__ALL__test"),
                         ("fedavg", "fl_fedavg_lora_ft__personal__test"), ("local", "local_lora__personal__test")]:
            for seed in SEEDS:
                d = scores(fam, run, seed)
                if d is None or key not in d.files:
                    continue
                t = pd.DataFrame({"y": d["y_test"], "s": d[key], "v": df.loc[d["test_idx"], "venue_id"].to_numpy(),
                                  "sub": df.loc[d["test_idx"], "subfield"].to_numpy()})
                for grp in ["v", "sub"]:
                    a, w, npos = [], [], 0
                    for _, g in t.groupby(grp):
                        k1, k0 = int(g.y.sum()), int((g.y == 0).sum())
                        if k1 >= MIN_V and k0 >= MIN_V:
                            a.append(auc(g.y, g.s)); w.append(len(g)); npos += k1
                    rows.append({"model": name, "scores": key.split("__")[0] + ("__personal" if "personal" in key else ""),
                                 "seed": seed, "group": {"v": "journal", "sub": "subfield"}[grp],
                                 "n_groups": len(a), "within_auc": float(np.average(a, weights=w)) if a else np.nan,
                                 "pos_covered": npos / t.y.sum(), "pooled_auc": auc(t.y, t.s)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "within_journal.csv", index=False)
    return out


# ------------------------------------------------------------------ publisher prior + paired bootstrap
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def prior_and_paired(df, nboot):
    prior_rows, pair_rows = [], []
    for fam, name in FAMS.items():
        for seed in SEEDS:
            c, f, lo = scores(fam, "central", seed), scores(fam, "fedavg", seed), scores(fam, "local", seed)
            if c is None:
                continue
            part = split(df, seed)
            tr = df[part == "train"]
            prev = tr.groupby("silo").retracted.mean()
            p_all = tr.retracted.mean()
            y, silo = c["y_test"], c["silo_test"]
            sc = c["central_lora__ALL__test"]
            sc = sc if (sc.min() >= 0 and sc.max() <= 1) else 1 / (1 + np.exp(-sc))
            shift = np.array([logit(prev[s]) - logit(p_all) for s in silo])
            prior = logit(sc) + shift
            cand = {"central": sc, "central + publisher prior": prior}
            if f is not None:
                cand["FedAvg"] = f["fl_fedavg_lora__ALL__test"]
                if "fl_fedavg_lora_ft__personal__test" in f.files:
                    cand["FedAvg + local FT (personal)"] = f["fl_fedavg_lora_ft__personal__test"]
            if lo is not None and "local_lora__personal__test" in lo.files:
                cand["local only (personal)"] = lo["local_lora__personal__test"]
            if fam == "c3_lora_abstracts":
                for run, key in [("ditto", "fl_ditto_lora__personal__test"), ("fedper", "fl_fedper_lora__personal__test")]:
                    d = scores(fam, run, seed)
                    if d is not None and key in d.files:
                        cand[{"ditto": "Ditto (personal)", "fedper": "FedPer (personal)"}[run]] = d[key]
            for k, s in cand.items():
                own = np.mean([ap(y[silo == q], s[silo == q]) for q in np.unique(silo)])
                prior_rows.append({"model": name, "seed": seed, "scores": k, "pooled_auprc": ap(y, s),
                                   "pooled_auc": auc(y, s), "own_silo_macro_auprc": own})
            ref = "FedAvg + local FT (personal)"
            if ref in cand:
                for k in cand:
                    if k == ref:
                        continue
                    a, b = cand[ref], cand[k]
                    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
                    ds = []
                    for _ in range(nboot):
                        i = np.concatenate([RNG.choice(pos, len(pos)), RNG.choice(neg, len(neg))])
                        ds.append(ap(y[i], a[i]) - ap(y[i], b[i]))
                    pair_rows.append({"model": name, "seed": seed, "vs": k, "delta_auprc": ap(y, a) - ap(y, b),
                                      "lo": np.percentile(ds, 2.5), "hi": np.percentile(ds, 97.5)})
    pr, pa = pd.DataFrame(prior_rows), pd.DataFrame(pair_rows)
    pr.to_csv(OUT / "publisher_prior.csv", index=False)
    pa.to_csv(OUT / "paired_bootstrap.csv", index=False)
    return pr, pa


# ------------------------------------------------------------------ formatting-only classifier
def format_check(df):
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold

    def feats(s: pd.Series, p: str):
        s = s.fillna("")
        n = s.str.len().clip(lower=1)
        return pd.DataFrame({
            f"{p}_chars": s.str.len(), f"{p}_words": s.str.split().str.len().fillna(0),
            f"{p}_upper": s.str.count(r"[A-Z]") / n, f"{p}_digit": s.str.count(r"[0-9]") / n,
            f"{p}_punct": s.str.count(r"[^\w\s]") / n, f"{p}_nonascii": s.str.count(r"[^\x00-\x7f]") / n,
            f"{p}_markup": s.str.count(r"<[^>]+>|&[a-z]+;"), f"{p}_sent": s.str.count(r"[.!?](\s|$)"),
            f"{p}_dblspace": s.str.count(r"\s{2,}"), f"{p}_colon": s.str.count(":"),
            f"{p}_endsdot": s.str.strip().str.endswith(".").astype(int),
            f"{p}_lowerstart": s.str.match(r"^\s*[a-z]").astype(int)})
    X = pd.concat([feats(df["title"], "title"), feats(df["abstract"], "abs"), df[["has_abstract"]]], axis=1).fillna(0)
    y = df["retracted"].to_numpy()
    oof = np.zeros(len(y))
    for trn, tst in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=16, tree_method="hist")
        m.fit(X.iloc[trn], y[trn])
        oof[tst] = m.predict_proba(X.iloc[tst])[:, 1]
    m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=16, tree_method="hist").fit(X, y)
    imp = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
    # within-subfield AUC of the format-only scores (same rule as b8: >= 20 of each class)
    t = pd.DataFrame({"y": y, "s": oof, "sub": df["subfield"].to_numpy()})
    wa, ww = [], []
    for _, g_ in t.groupby("sub"):
        if g_.y.sum() >= 20 and (g_.y == 0).sum() >= 20:
            wa.append(auc(g_.y, g_.s)); ww.append(len(g_))
    # does the language model separate articles that look alike on formatting?
    lm = []
    for fam, name in FAMS.items():
        for seed in SEEDS:
            d = scores(fam, "central", seed)
            if d is None:
                continue
            ti = d["test_idx"]
            tt = pd.DataFrame({"y": d["y_test"], "lm": d["central_lora__ALL__test"], "fmt": oof[ti]})
            tt["dec"] = pd.qcut(tt.fmt.rank(method="first"), 10, labels=False)
            aa = [(auc(g_.y, g_.lm), len(g_)) for _, g_ in tt.groupby("dec") if 0 < g_.y.mean() < 1]
            lm.append({"model": name, "seed": seed, "lm_auc_within_format_deciles": np.average([x for x, _ in aa], weights=[w for _, w in aa]),
                       "lm_auc": auc(tt.y, tt.lm), "format_auc_on_test": auc(tt.y, tt.fmt)})
    pd.DataFrame(lm).to_csv(OUT / "format_vs_lm.csv", index=False)
    g = df.groupby("retracted")
    res = {"format_only_auprc": ap(y, oof), "format_only_auc": auc(y, oof), "base_rate": y.mean(),
           "format_only_within_subfield_auc": float(np.average(wa, weights=ww)),
           "top_features": imp.head(8).round(3).to_dict(),
           "lm_within_format_deciles": pd.DataFrame(lm).groupby("model")[["lm_auc", "lm_auc_within_format_deciles"]].mean().round(3).to_dict("index"),
           "has_abstract_by_label": g.has_abstract.mean().to_dict(),
           "abstract_words_median_by_label": g.abstract_len_words.median().to_dict(),
           "title_chars_median_by_label": g.title.apply(lambda s: s.fillna("").str.len().median()).to_dict()}
    return res


# ------------------------------------------------------------------ near-duplicates
P = (1 << 61) - 1


def minhash(texts, k=5, nperm=32, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.integers(1, P - 1, nperm, dtype=np.uint64)
    b = rng.integers(0, P - 1, nperm, dtype=np.uint64)
    sig = np.full((len(texts), nperm), np.iinfo(np.uint64).max, dtype=np.uint64)
    for i, t in enumerate(texts):
        w = re.findall(r"[a-z0-9]+", t.lower())
        if len(w) < k:
            continue
        x = np.array([zlib.crc32(" ".join(w[j:j + k]).encode()) for j in range(len(w) - k + 1)], dtype=np.uint64)
        sig[i] = ((a[:, None] * x[None, :] + b[:, None]) % np.uint64(P)).min(axis=1)
    return sig


def near_dups(df, bands=8, thr=0.5):
    out = []
    for fam, name in list(FAMS.items())[:1]:
        for seed in SEEDS:
            c = scores(fam, "central", seed)
            part = split(df, seed)
            m = (df["has_abstract"] == 1).to_numpy()
            tr = np.where((part == "train") & m)[0]
            te_all = c["test_idx"]
            te = te_all[m[te_all]]
            sig = minhash(df["abstract"].fillna("").to_numpy()[np.concatenate([tr, te])].tolist())
            st, ss = sig[: len(tr)], sig[len(tr):]
            r = sig.shape[1] // bands
            buckets: dict = {}
            for bi in range(bands):
                for i, row in enumerate(st[:, bi * r:(bi + 1) * r]):
                    buckets.setdefault((bi, row.tobytes()), []).append(i)
            dup = np.zeros(len(te), bool)
            for j, row in enumerate(ss):
                if row[0] == np.iinfo(np.uint64).max:
                    continue
                cands = set()
                for bi in range(bands):
                    cands.update(buckets.get((bi, row[bi * r:(bi + 1) * r].tobytes()), []))
                for i in cands:
                    if (st[i] == row).mean() >= thr:
                        dup[j] = True
                        break
            yte = df["retracted"].to_numpy()[te]
            pos_in = {v: k for k, v in enumerate(te_all)}
            keep = np.ones(len(te_all), bool)
            for v in te[dup]:
                keep[pos_in[v]] = False
            y, s = c["y_test"], c["central_lora__ALL__test"]
            out.append({"model": name, "seed": seed, "test_with_abstract": len(te),
                        "neardup_share_retracted": dup[yte == 1].mean(), "neardup_share_control": dup[yte == 0].mean(),
                        "auprc_all": ap(y, s), "auprc_without_neardups": ap(y[keep], s[keep]),
                        "base_rate_without": y[keep].mean()})
    res = pd.DataFrame(out)
    res.to_csv(OUT / "near_duplicates.csv", index=False)
    return res


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--boot", type=int, default=1000)
    ap_.add_argument("--only", default="all")
    a = ap_.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df = corpus()
    summ = {}
    todo = a.only.split(",") if a.only != "all" else ["prospective", "journal", "prior", "format", "neardup"]
    if "prospective" in todo:
        p, comp = prospective(df, a.boot)
        print(p.groupby(["model", "regime", "subset"])[["n_pos", "n_neg", "roc_auc", "auprc"]].mean().round(3).to_string())
        print(comp.head(10).to_string())
    if "journal" in todo:
        j = within_journal(df)
        print(j.groupby(["model", "scores", "group"])[["n_groups", "within_auc", "pos_covered", "pooled_auc"]].mean().round(3).to_string())
    if "prior" in todo:
        pr, pa = prior_and_paired(df, a.boot)
        print(pr.groupby(["model", "scores"])[["pooled_auprc", "own_silo_macro_auprc"]].agg(["mean", "std"]).round(3).to_string())
        print(pa.groupby(["model", "vs"])[["delta_auprc", "lo", "hi"]].mean().round(4).to_string())
    if "format" in todo:
        summ["format"] = format_check(df)
        print(json.dumps(summ["format"], indent=1, default=float))
    if "neardup" in todo:
        nd = near_dups(df)
        print(nd.round(4).to_string())
    old = json.loads((OUT / "summary.json").read_text()) if (OUT / "summary.json").exists() else {}
    old.update(summ)
    (OUT / "summary.json").write_text(json.dumps(old, indent=1, default=float))
