"""Overnight gap-fix ladder on the Usman WebSci'25 corpus.

Arenas:
  HP  "his protocol": clean-180 subset, single 80/20 split seed42
      (baseline anchors to locked 0.889)
  CV  "armor": audited-172 subset, 5-fold stratified CV x 3 seeds

Ladder:
  A    baseline: his exported features, RF                     (locked 0.889)
  B    A + Gap3/4 certainty profiles (our recomputation)
  C    B + Gap7 tortured phrases + GPT-2 perplexity
  D0   Gap1 control: abstract-embedding-only + MLP head
  D    Gap1: section SciBERT embeddings + attention pooling
  E    unified: D + (B,C feature block)
  LOO  leave-one-gap-out from E (impact attribution)
Every stage appends to results/gapfix_results.csv immediately.
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

torch.set_num_threads(12)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HIS_COLS = ["abstract_Readability", "Method_Readability", "Results_Readability",
            "Conclusion_Readability", "Avg_Readability", "Avg_Certainty",
            "Abstract_Certainty", "Method_Certainty", "Result_Certainty",
            "Conclusion_Certainty"]
RESULTS = ROOT / "results/gapfix_results.csv"
SEEDS = [42, 43, 44]


def load_arenas():
    df = pd.read_csv(
        ROOT / "data/raw/Scientific-Accountability-main/Retracted_NonRetracted Articles.csv",
        sep=";", engine="python", quotechar='"', on_bad_lines="skip")
    df["Label"] = df["Label"].str.strip().map({"R": 1, "NR": 0})
    for c in HIS_COLS:
        df[c] = df[c].astype(str).str.replace(",", ".").astype(float)
    df = df.reset_index(drop=True)
    df["row_id"] = df.index
    mask = pd.Series(False, index=df.index)
    for c in HIS_COLS:
        vc = df[c].value_counts()
        mask |= df[c].isin(vc[vc > 5].index)
    clean180 = df[~mask]
    audited = pd.read_csv(ROOT / "data/processed/usman_clean_subset.csv")
    audited_ids = set(audited["pmcid"])
    audited172 = clean180[clean180["pmcid"].isin(audited_ids)]
    return df, clean180, audited172


def feature_matrix(df_all: pd.DataFrame, rows: pd.DataFrame, cols: list[str]) -> np.ndarray:
    gf = pd.read_parquet(ROOT / "data/processed/gapfix_features.parquet")
    gf = gf.iloc[rows["row_id"].to_numpy()]
    joined = pd.concat([rows.reset_index(drop=True)[HIS_COLS],
                        gf.reset_index(drop=True)], axis=1)
    return joined.reindex(columns=cols).to_numpy(dtype=np.float32)


def eval_tabular(name, cols, clean180, audited172, model_fn, df_all):
    out = []
    # HP arena
    X = feature_matrix(df_all, clean180, cols)
    y = clean180["Label"].to_numpy()
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=42)
    imp, sc = SimpleImputer(strategy="median"), StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(X[tr]))
    Xte = sc.transform(imp.transform(X[te]))
    clf = model_fn(42)
    clf.fit(Xtr, y[tr])
    s = clf.predict_proba(Xte)[:, 1]
    out.append({"model": name, "arena": "HP", "acc": accuracy_score(y[te], s > 0.5),
                "f1": f1_score(y[te], s > 0.5), "auc": roc_auc_score(y[te], s)})
    # CV arena
    X = feature_matrix(df_all, audited172, cols)
    y = audited172["Label"].to_numpy()
    accs, f1s, aucs = [], [], []
    for seed in SEEDS:
        scores = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            imp, sc = SimpleImputer(strategy="median"), StandardScaler()
            Xtr = sc.fit_transform(imp.fit_transform(X[tr]))
            Xte = sc.transform(imp.transform(X[te]))
            clf = model_fn(seed)
            clf.fit(Xtr, y[tr])
            scores[te] = clf.predict_proba(Xte)[:, 1]
        accs.append(accuracy_score(y, scores > 0.5))
        f1s.append(f1_score(y, scores > 0.5))
        aucs.append(roc_auc_score(y, scores))
    out.append({"model": name, "arena": "CV", "acc": np.mean(accs), "acc_std": np.std(accs),
                "f1": np.mean(f1s), "auc": np.mean(aucs)})
    persist(out)
    return out


class AttnPool(nn.Module):
    def __init__(self, d=768, extra=0):
        super().__init__()
        self.score = nn.Linear(d, 1)
        self.head = nn.Sequential(nn.Linear(d + extra, 64), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, embs, mask, extra=None):
        a = self.score(embs).squeeze(-1).masked_fill(mask < 0.5, -1e9)
        w = torch.softmax(a, dim=1).unsqueeze(-1)
        doc = (embs * w).sum(1)
        if extra is not None:
            doc = torch.cat([doc, extra], dim=1)
        return self.head(doc).squeeze(-1)


def train_head(embs, mask, y, tr, te, seed, extra=None, epochs=80):
    torch.manual_seed(seed)
    model = AttnPool(extra=0 if extra is None else extra.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    E, M = torch.tensor(embs[tr]), torch.tensor(mask[tr])
    Y = torch.tensor(y[tr], dtype=torch.float32)
    X = None if extra is None else torch.tensor(extra[tr])
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        lossf(model(E, M, X), Y).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        s = torch.sigmoid(model(torch.tensor(embs[te]), torch.tensor(mask[te]),
                                None if extra is None else torch.tensor(extra[te])))
    return s.numpy()


def eval_attn(name, sec_slice, clean180, audited172, df_all, extra_cols=None):
    npz = np.load(ROOT / "data/processed/gapfix_scibert.npz")
    embs_all = npz["embs"].astype(np.float32)[:, sec_slice, :]
    mask_all = npz["mask"].astype(np.float32)[:, sec_slice]
    out = []
    for arena, rows in [("HP", clean180), ("CV", audited172)]:
        ids = rows["row_id"].to_numpy()
        embs, mask = embs_all[ids], mask_all[ids]
        y = rows["Label"].to_numpy()
        extra = None
        if extra_cols:
            extra = feature_matrix(df_all, rows, extra_cols)
        if arena == "HP":
            idx = np.arange(len(y))
            tr, te = train_test_split(idx, test_size=0.2, random_state=42)
            ex = prep_extra(extra, tr, te) if extra is not None else None
            s = train_head(embs, mask, y, tr, te, 42, ex)
            out.append({"model": name, "arena": "HP", "acc": accuracy_score(y[te], s > 0.5),
                        "f1": f1_score(y[te], s > 0.5), "auc": roc_auc_score(y[te], s)})
        else:
            accs, f1s, aucs = [], [], []
            for seed in SEEDS:
                scores = np.zeros(len(y))
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(embs[:, 0, :], y):
                    ex = prep_extra(extra, tr, te) if extra is not None else None
                    scores[te] = train_head(embs, mask, y, tr, te, seed, ex)
                accs.append(accuracy_score(y, scores > 0.5))
                f1s.append(f1_score(y, scores > 0.5))
                aucs.append(roc_auc_score(y, scores))
            out.append({"model": name, "arena": "CV", "acc": np.mean(accs),
                        "acc_std": np.std(accs), "f1": np.mean(f1s), "auc": np.mean(aucs)})
    persist(out)
    return out


def prep_extra(extra, tr, te):
    imp, sc = SimpleImputer(strategy="median"), StandardScaler()
    ex = np.zeros_like(extra, dtype=np.float32)
    ex[tr] = sc.fit_transform(imp.fit_transform(extra[tr]))
    ex[te] = sc.transform(imp.transform(extra[te]))
    return ex


def persist(rows):
    df = pd.DataFrame(rows)
    header = not RESULTS.exists()
    df.to_csv(RESULTS, mode="a", header=header, index=False)
    print(df.round(4).to_string(index=False), flush=True)


def main() -> None:
    t0 = time.time()
    df_all, clean180, audited172 = load_arenas()
    print(f"arenas: HP={len(clean180)}, CV={len(audited172)}", flush=True)
    gf = pd.read_parquet(ROOT / "data/processed/gapfix_features.parquet")
    PROF = [c for c in gf.columns if "_cert_" in c] + \
           ["traj_abs_minus_methods", "traj_abs_minus_results"]
    G7 = [c for c in gf.columns if c.endswith("_tortured") or c.endswith("_ppl")]

    rf = lambda s: RandomForestClassifier(300, class_weight="balanced", n_jobs=12, random_state=s)
    xgb = lambda s: XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.06,
                                  subsample=0.9, colsample_bytree=0.9,
                                  random_state=s, eval_metric="logloss", n_jobs=12)

    eval_tabular("A_baseline_hisfeat_RF", HIS_COLS, clean180, audited172, rf, df_all)
    eval_tabular("B_gap34_profiles_XGB", HIS_COLS + PROF, clean180, audited172, xgb, df_all)
    eval_tabular("C_gap347_XGB", HIS_COLS + PROF + G7, clean180, audited172, xgb, df_all)
    eval_attn("D0_abstractonly_head", slice(0, 1), clean180, audited172, df_all)
    eval_attn("D_gap1_section_attn", slice(0, 4), clean180, audited172, df_all)
    eval_attn("E_unified", slice(0, 4), clean180, audited172, df_all,
              extra_cols=HIS_COLS + PROF + G7)
    # impact attribution: leave-one-gap-out from E
    eval_attn("LOO_minus_gap34", slice(0, 4), clean180, audited172, df_all,
              extra_cols=HIS_COLS + G7)
    eval_attn("LOO_minus_gap7", slice(0, 4), clean180, audited172, df_all,
              extra_cols=HIS_COLS + PROF)
    eval_tabular("LOO_minus_gap1_XGB", HIS_COLS + PROF + G7, clean180, audited172,
                 xgb, df_all)
    print(f"LADDER_DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
