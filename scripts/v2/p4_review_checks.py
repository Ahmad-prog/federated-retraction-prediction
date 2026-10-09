"""P4: second round of review checks, computed on CPU from saved test scores and frozen embeddings
(no language model is re-trained).

  own       paired bootstrap on the metric of Table 1(b): mean over publishers of the AUPRC on their own
            test articles (resampling within each publisher), plus the number of publishers that gain
  others    Table 1(c) reference rows: Central and FedAvg on the other publishers' articles
  prosp     future-retraction test with an abstract: AUPRC and ROC with 95% CIs, controls re-weighted
            to the publication years of the positives, and the title-only diagnostics
  nohindawi pooled and own-publisher AUPRC without the Hindawi silo (templated special-issue batches)
  emb       logistic-regression heads on frozen Qwen3-Embedding-8B abstract embeddings:
            leave-one-publisher-out (a new member), a temporal split and a journal-grouped split

Writes results_v2/p4_review/*.csv.
Usage: python scripts/v2/p4_review_checks.py [--only own,others,prosp,nohindawi,emb] [--boot 1000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.v2.b2_main_benchmark import partition, split  # noqa: E402

RES = ROOT / "results_v2"
OUT = RES / "p4_review"
FAMS = {"c3_lora_abstracts": "ModernBERT-base", "c3_lora_abstracts_modernbert-large": "ModernBERT-large",
        "c3_lora_abstracts_qwen3-1.7b": "Qwen3-1.7B"}
SEEDS = [42, 43, 44]
RNG = np.random.default_rng(0)
HINDAWI = "Hindawi Publishing Corporation"


def corpus() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data_v2/tabular_v2.parquet").reset_index(drop=True)
    df["silo"] = partition(df, "publisher")
    return df


def scores(fam: str, run: str, seed: int):
    f = RES / fam / f"scores_{run}_s{seed}.npz"
    return np.load(f, allow_pickle=True) if f.exists() else None


def candidates(fam: str, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Test labels, publisher of each test article and every saved score vector of this seed."""
    c = scores(fam, "central", seed)
    y, silo = c["y_test"], c["silo_test"].astype(str)
    cand = {"Central": c["central_lora__ALL__test"]}
    f = scores(fam, "fedavg", seed)
    if f is not None:
        cand["FedAvg"] = f["fl_fedavg_lora__ALL__test"]
        if "fl_fedavg_lora_ft__personal__test" in f.files:
            cand["FedAvg + local FT"] = f["fl_fedavg_lora_ft__personal__test"]
    lo = scores(fam, "local", seed)
    if lo is not None and "local_lora__personal__test" in lo.files:
        cand["Local only"] = lo["local_lora__personal__test"]
    for run, key, lab in [("ditto", "fl_ditto_lora__personal__test", "Ditto"),
                          ("fedper", "fl_fedper_lora__personal__test", "FedPer")]:
        d = scores(fam, run, seed)
        if d is not None and key in d.files:
            cand[lab] = d[key]
    for d in (f, lo):
        if d is not None:
            assert np.array_equal(d["y_test"], y), "runs of one seed must share the test split"
    return y, silo, cand


# ------------------------------------------------------------------ own-publisher paired bootstrap
def own(nboot: int) -> pd.DataFrame:
    rows = []
    for fam, name in FAMS.items():
        for seed in SEEDS:
            if scores(fam, "central", seed) is None:
                continue
            y, silo, cand = candidates(fam, seed)
            ref = "FedAvg + local FT"
            if ref not in cand:
                continue
            groups = [(np.where((silo == q) & (y == 1))[0], np.where((silo == q) & (y == 0))[0]) for q in np.unique(silo)]
            per = {k: np.array([ap(y[np.r_[p, n]], s[np.r_[p, n]]) for p, n in groups]) for k, s in cand.items()}
            for k in cand:
                if k == ref:
                    continue
                ds = []
                for _ in range(nboot):
                    d = 0.0
                    for p, n in groups:
                        i = np.r_[RNG.choice(p, len(p)), RNG.choice(n, len(n))]
                        d += ap(y[i], cand[ref][i]) - ap(y[i], cand[k][i])
                    ds.append(d / len(groups))
                rows.append({"model": name, "seed": seed, "vs": k, "delta_own_auprc": per[ref].mean() - per[k].mean(),
                             "lo": np.percentile(ds, 2.5), "hi": np.percentile(ds, 97.5),
                             "publishers_gaining": int((per[ref] > per[k]).sum()), "publishers": len(groups)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "own_paired_bootstrap.csv", index=False)
    return out


# ------------------------------------------------------------------ other publishers' articles
def others() -> pd.DataFrame:
    rows = []
    for fam, name in FAMS.items():
        for seed in SEEDS:
            for run, key, lab in [("central", "central_lora", "Central"), ("fedavg", "fl_fedavg_lora", "FedAvg"),
                                  ("local", "local_lora", "Local only"), ("fedavg", "fl_fedavg_lora_ft", "FedAvg + local FT")]:
                d = scores(fam, run, seed)
                if d is None:
                    continue
                y, silo = d["y_test"], d["silo_test"].astype(str)
                vals = []
                for q in np.unique(silo):
                    k = f"{key}__ALL__test" if lab in ("Central", "FedAvg") else f"{key}__{q}__test"
                    if k in d.files:
                        m = silo != q
                        vals.append(ap(y[m], d[k][m]))
                if len(vals) == len(np.unique(silo)):
                    rows.append({"model": name, "seed": seed, "regime": lab, "other_auprc": float(np.mean(vals))})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "other_publishers.csv", index=False)
    return out


# ------------------------------------------------------------------ prospective
def boot2(y, s, w, nboot):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    a, r = [], []
    for _ in range(nboot):
        i = np.r_[RNG.choice(pos, len(pos)), RNG.choice(neg, len(neg))]
        a.append(ap(y[i], s[i], sample_weight=w[i]))
        r.append(auc(y[i], s[i], sample_weight=w[i]))
    return np.percentile(a, [2.5, 97.5]), np.percentile(r, [2.5, 97.5])


def prosp(df: pd.DataFrame, nboot: int) -> pd.DataFrame:
    pro = pd.read_parquet(ROOT / "data_v2/prospective_abstracts.parquet")
    pro = pro[~pro["doi"].isin(set(df["doi"]))].reset_index(drop=True)
    has_p = (pro["has_abstract"] == 1).to_numpy()
    rows, diag = [], []
    for fam, name in FAMS.items():
        for run, key in [("central", "central_lora"), ("fedavg", "fl_fedavg_lora")]:
            for seed in SEEDS:
                d = scores(fam, run, seed)
                if d is None or f"{key}__ALL__prospective" not in d.files:
                    continue
                sp, st = d[f"{key}__ALL__prospective"], d[f"{key}__ALL__test"]
                ti = d["test_idx"]
                ctl = d["pros_ctl"] & (d["y_test"] == 0)
                has_c = df.loc[ti, "has_abstract"].to_numpy() == 1
                yr_c = df.loc[ti, "year"].to_numpy()
                pub_c = df.loc[ti, "publisher"].to_numpy().astype(str)
                for sub, mp, mc in [("with abstract", has_p, has_c), ("title only", ~has_p, ~has_c)]:
                    c = ctl & mc
                    s = np.r_[sp[mp], st[c]]
                    y = np.r_[np.ones(mp.sum()), np.zeros(c.sum())]
                    yr = np.r_[pro.loc[mp, "year"].to_numpy(), yr_c[c]]
                    pub = np.r_[pro.loc[mp, "publisher"].to_numpy().astype(str), pub_c[c]]
                    # controls re-weighted so that their publication years match the positives'
                    py = pd.Series(yr[y == 1]).value_counts(normalize=True)
                    cy = pd.Series(yr[y == 0]).value_counts(normalize=True)
                    w_year = np.where(y == 1, 1.0, [py.get(v, 0.0) / cy.get(v, np.inf) for v in yr])
                    for wlab, w in [("unweighted", np.ones_like(y)), ("year-matched", w_year)]:
                        r = {"model": name, "regime": run, "seed": seed, "subset": sub, "controls": wlab,
                             "n_pos": int(mp.sum()), "n_neg": int(c.sum()),
                             "base_rate": float(np.average(y, weights=w)),
                             "auprc": ap(y, s, sample_weight=w), "roc_auc": auc(y, s, sample_weight=w)}
                        if seed == 42:
                            (r["auprc_lo"], r["auprc_hi"]), (r["roc_lo"], r["roc_hi"]) = boot2(y, s, w, nboot)
                        rows.append(r)
                    if run == "central":
                        e = pub == "Elsevier BV"
                        diag.append({"model": name, "seed": seed, "subset": sub,
                                     "pos_median_year": float(np.median(yr[y == 1])),
                                     "ctl_median_year": float(np.median(yr[y == 0])),
                                     "pos_share_elsevier": float(e[y == 1].mean()),
                                     "ctl_share_elsevier": float(e[y == 0].mean()),
                                     "roc_elsevier_only": auc(y[e], s[e]) if 0 < y[e].mean() < 1 else np.nan,
                                     "roc_non_elsevier": auc(y[~e], s[~e]) if 0 < y[~e].mean() < 1 else np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "prospective_auprc.csv", index=False)
    pd.DataFrame(diag).to_csv(OUT / "prospective_title_only_diagnostics.csv", index=False)
    return out


# ------------------------------------------------------------------ without Hindawi
def nohindawi() -> pd.DataFrame:
    rows = []
    for fam, name in FAMS.items():
        for seed in SEEDS:
            if scores(fam, "central", seed) is None:
                continue
            y, silo, cand = candidates(fam, seed)
            keep = silo != HINDAWI
            for k, s in cand.items():
                own_q = [ap(y[silo == q], s[silo == q]) for q in np.unique(silo[keep])]
                rows.append({"model": name, "seed": seed, "scores": k, "pooled_auprc_all": ap(y, s),
                             "pooled_auprc_no_hindawi": ap(y[keep], s[keep]), "roc_no_hindawi": auc(y[keep], s[keep]),
                             "base_rate_no_hindawi": y[keep].mean(), "own_macro_no_hindawi": float(np.mean(own_q))})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "without_hindawi.csv", index=False)
    return out


# ------------------------------------------------------------------ frozen-embedding heads
def emb(df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler

    name = "abstracts_qwen3emb8b"
    E = np.load(ROOT / "data_v2/emb" / f"{name}.npy").astype(np.float32)
    ids = (ROOT / "data_v2/emb" / f"{name}.ids.txt").read_text().split("\n")
    assert df["doi"].tolist() == ids, "embedding order must match tabular_v2"
    y = df["retracted"].to_numpy()
    silo = df["silo"].to_numpy().astype(str)

    def fit_eval(tr, te):
        sc = StandardScaler().fit(E[tr])
        m = LogisticRegression(C=0.05, max_iter=300).fit(sc.transform(E[tr]), y[tr])
        s = m.decision_function(sc.transform(E[te]))
        return ap(y[te], s), auc(y[te], s), y[te].mean()

    rows = []
    for seed in SEEDS:
        part = split(df, seed)
        tr_all, te_all = np.where(part == "train")[0], np.where(part == "test")[0]
        a, r, b = fit_eval(tr_all, te_all)
        rows.append({"exp": "random split (reference)", "seed": seed, "silo": "ALL", "auprc": a, "roc_auc": r, "base": b})
        # leave one publisher out: a new member gets the model of the other nine
        for q in np.unique(silo):
            te = te_all[silo[te_all] == q]
            for lab, tr in [("other nine publishers", tr_all[silo[tr_all] != q]),
                            ("own data only", tr_all[silo[tr_all] == q]),
                            ("all ten publishers", tr_all)]:
                a, r, b = fit_eval(tr, te)
                rows.append({"exp": f"held-out publisher: {lab}", "seed": seed, "silo": q, "auprc": a, "roc_auc": r,
                             "base": b})
        # temporal: train on articles published up to 2020, test on 2022-2025 (2021 left out as a gap)
        rng = np.random.default_rng(seed)
        yr = df["year"].to_numpy()
        late = np.where(yr >= 2022)[0]
        late = rng.permutation(late)
        te, late_rest = late[: len(late) // 2], late[len(late) // 2:]
        tr_t = np.where(yr <= 2020)[0]
        pool = np.r_[tr_t, np.where(yr == 2021)[0], late_rest]
        tr_r = rng.choice(pool, len(tr_t), replace=False)
        for lab, tr in [("temporal: train <= 2020", tr_t), ("temporal reference: random train, same size", tr_r)]:
            a, r, b = fit_eval(tr, te)
            rows.append({"exp": lab, "seed": seed, "silo": "ALL", "auprc": a, "roc_auc": r, "base": b})
        # journal-grouped: test journals never seen in training
        g = df["venue_id"].fillna("none").to_numpy()
        tr_g, te_g = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(E, y, g))
        tr_g2 = rng.choice(np.setdiff1d(np.arange(len(y)), te_g), len(tr_g), replace=False)
        a, r, b = fit_eval(tr_g, te_g)
        rows.append({"exp": "journal-grouped: unseen journals", "seed": seed, "silo": "ALL", "auprc": a, "roc_auc": r, "base": b})
        print(pd.DataFrame(rows).tail(5).to_string(), flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "embedding_splits.csv", index=False)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--only", default="own,others,prosp,nohindawi,emb")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df = corpus()
    todo = a.only.split(",")
    pd.set_option("display.width", 200)
    if "own" in todo:
        o = own(a.boot)
        print(o.groupby(["model", "vs"])[["delta_own_auprc", "lo", "hi", "publishers_gaining"]].mean().round(4).to_string())
    if "others" in todo:
        o = others()
        print(o.groupby(["model", "regime"]).other_auprc.agg(["mean", "std", "count"]).round(3).to_string())
    if "prosp" in todo:
        o = prosp(df, a.boot)
        print(o.groupby(["model", "regime", "subset", "controls"])[["auprc", "roc_auc", "base_rate"]].mean().round(3).to_string())
    if "nohindawi" in todo:
        o = nohindawi()
        print(o.groupby(["model", "scores"]).mean(numeric_only=True).drop(columns="seed").round(3).to_string())
    if "emb" in todo:
        o = emb(df)
        print(o.groupby("exp")[["auprc", "roc_auc", "base"]].mean().round(3).to_string())
