"""Night iteration 2: fix baseline fidelity + repair the Gap-1 neural arm.

N1  faithful baseline: his RF config exactly (fillna 0, default params) -> HP
N2  mean-pooled SciBERT re-extraction (CLS -> token mean, the standard fix)
N3  head v2 on mean-pooled embeddings: standardized, 800 steps, D0/D/E rows
N4  fine-tuning: SciBERT flat-128 vs section-512 (2 seeds, HP arena)
Appends to results/gapfix_results.csv (models prefixed N_).
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(12)

from scripts.gapfix_train import (HIS_COLS, RESULTS, SEEDS, AttnPool, load_arenas,
                                  feature_matrix, persist, prep_extra)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

SECTIONS = {"abstract": "abstract", "methods": "Materials_and_Methods",
            "results": "Results", "conclusion": "Conclusion"}
MEANPOOL = ROOT / "data/processed/gapfix_scibert_meanpool.npz"


def n1_faithful_baseline(clean180):
    X = clean180[HIS_COLS].fillna(0).to_numpy(np.float32)
    y = clean180["Label"].to_numpy()
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=42)
    sc = StandardScaler()
    Xtr, Xte = sc.fit_transform(X[tr]), sc.transform(X[te])
    clf = RandomForestClassifier(random_state=42)
    clf.fit(Xtr, y[tr])
    s = clf.predict_proba(Xte)[:, 1]
    persist([{"model": "N_A_faithful_baseline_RF", "arena": "HP",
              "acc": accuracy_score(y[te], s > 0.5), "f1": f1_score(y[te], s > 0.5),
              "auc": roc_auc_score(y[te], s)}])


def n2_meanpool_extract(df_all):
    if MEANPOOL.exists():
        print("meanpool cached", flush=True)
        return
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    enc = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
    enc.eval()
    embs = np.zeros((len(df_all), 4, 768), dtype=np.float32)
    mask = np.zeros((len(df_all), 4), dtype=np.float32)
    with torch.no_grad():
        for i, row in df_all.iterrows():
            for si, (tag, col) in enumerate(SECTIONS.items()):
                txt = (str(row[col]) if pd.notna(row[col]) else "").strip()
                if len(txt) < 50:
                    continue
                ids = tok(txt, return_tensors="pt", truncation=False).input_ids[0]
                chunks = [ids[j:j + 510] for j in range(0, min(len(ids), 2040), 510)]
                vecs = []
                for ch in chunks:
                    ch = torch.cat([torch.tensor([tok.cls_token_id]), ch,
                                    torch.tensor([tok.sep_token_id])]).unsqueeze(0)
                    h = enc(ch).last_hidden_state[0]
                    vecs.append(h.mean(0).numpy())  # token mean, not CLS
                embs[i, si] = np.mean(vecs, axis=0)
                mask[i, si] = 1.0
            if i % 50 == 0:
                print(f"  meanpool {i}/{len(df_all)}", flush=True)
    np.savez_compressed(MEANPOOL, embs=embs, mask=mask)
    del enc
    gc.collect()
    print("meanpool done", flush=True)


def head_v2(embs, mask, y, tr, te, seed, extra=None, steps=800):
    torch.manual_seed(seed)
    mu, sd = embs[tr].mean((0, 1), keepdims=True), embs[tr].std((0, 1), keepdims=True) + 1e-6
    E_all = (embs - mu) / sd
    model = AttnPool(extra=0 if extra is None else extra.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    lossf = nn.BCEWithLogitsLoss()
    E, M = torch.tensor(E_all[tr]), torch.tensor(mask[tr])
    Y = torch.tensor(y[tr], dtype=torch.float32)
    X = None if extra is None else torch.tensor(extra[tr])
    model.train()
    for _ in range(steps):
        opt.zero_grad()
        lossf(model(E, M, X), Y).backward()
        opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        s = torch.sigmoid(model(torch.tensor(E_all[te]), torch.tensor(mask[te]),
                                None if extra is None else torch.tensor(extra[te])))
    return s.numpy()


def n3_head_rows(df_all, clean180, audited172):
    npz = np.load(MEANPOOL)
    embs_all, mask_all = npz["embs"], npz["mask"]
    gf = pd.read_parquet(ROOT / "data/processed/gapfix_features.parquet")
    PROF = [c for c in gf.columns if "_cert_" in c] + \
           ["traj_abs_minus_methods", "traj_abs_minus_results"]
    G7 = [c for c in gf.columns if c.endswith("_tortured") or c.endswith("_ppl")]
    variants = [("N_D0_abs_meanpool", slice(0, 1), None),
                ("N_D_section_meanpool", slice(0, 4), None),
                ("N_E_unified_meanpool", slice(0, 4), HIS_COLS + PROF + G7)]
    for name, sl, extra_cols in variants:
        out = []
        for arena, rows in [("HP", clean180), ("CV", audited172)]:
            ids = rows["row_id"].to_numpy()
            embs, mask = embs_all[ids][:, sl, :], mask_all[ids][:, sl]
            y = rows["Label"].to_numpy()
            extra = feature_matrix(df_all, rows, extra_cols) if extra_cols else None
            if arena == "HP":
                idx = np.arange(len(y))
                tr, te = train_test_split(idx, test_size=0.2, random_state=42)
                ex = prep_extra(extra, tr, te) if extra is not None else None
                s = head_v2(embs, mask, y, tr, te, 42, ex)
                out.append({"model": name, "arena": arena,
                            "acc": accuracy_score(y[te], s > 0.5),
                            "f1": f1_score(y[te], s > 0.5),
                            "auc": roc_auc_score(y[te], s)})
            else:
                accs, f1s, aucs = [], [], []
                for seed in SEEDS:
                    scores = np.zeros(len(y))
                    for tr, te in StratifiedKFold(5, shuffle=True,
                                                  random_state=seed).split(embs[:, 0, :], y):
                        ex = prep_extra(extra, tr, te) if extra is not None else None
                        scores[te] = head_v2(embs, mask, y, tr, te, seed, ex)
                    accs.append(accuracy_score(y, scores > 0.5))
                    f1s.append(f1_score(y, scores > 0.5))
                    aucs.append(roc_auc_score(y, scores))
                out.append({"model": name, "arena": arena, "acc": np.mean(accs),
                            "acc_std": np.std(accs), "f1": np.mean(f1s),
                            "auc": np.mean(aucs)})
        persist(out)


def n4_finetune(df_all, clean180):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    y = clean180["Label"].to_numpy()
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=42)

    def run(name, texts, max_len, seed, epochs=4):
        torch.manual_seed(seed)
        tok = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        model = AutoModelForSequenceClassification.from_pretrained(
            "allenai/scibert_scivocab_uncased", num_labels=2)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
        enc = tok(list(texts), max_length=max_len, truncation=True,
                  padding="max_length", return_tensors="pt")
        ids, am = enc.input_ids, enc.attention_mask
        Y = torch.tensor(y)
        model.train()
        for ep in range(epochs):
            perm = torch.randperm(len(tr))
            for b in range(0, len(tr), 4):
                sel = torch.tensor(tr)[perm[b:b + 4]]
                opt.zero_grad()
                out = model(input_ids=ids[sel], attention_mask=am[sel], labels=Y[sel])
                out.loss.backward()
                opt.step()
            print(f"  {name} seed{seed} epoch{ep} done", flush=True)
        model.eval()
        preds, scores = [], []
        with torch.no_grad():
            for b in range(0, len(te), 8):
                sel = torch.tensor(te)[b:b + 8]
                logits = model(input_ids=ids[sel], attention_mask=am[sel]).logits
                sm = torch.softmax(logits, -1)[:, 1]
                scores.extend(sm.tolist())
        s = np.array(scores)
        persist([{"model": name, "arena": "HP", "acc": accuracy_score(y[te], s > 0.5),
                  "f1": f1_score(y[te], s > 0.5), "auc": roc_auc_score(y[te], s)}])
        del model
        gc.collect()

    cols = ["abstract", "Materials_and_Methods", "Conclusion", "Results"]
    flat = clean180[cols].fillna("").agg(" ".join, axis=1)
    for seed in [42, 43]:
        run(f"N_FT_flat128_s{seed}", flat, 128, seed)
        run(f"N_FT_full512_s{seed}", flat, 512, seed)


def main():
    t0 = time.time()
    df_all, clean180, audited172 = load_arenas()
    n1_faithful_baseline(clean180)
    n2_meanpool_extract(df_all)
    n3_head_rows(df_all, clean180, audited172)
    n4_finetune(df_all, clean180)
    print(f"NIGHT2_DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
