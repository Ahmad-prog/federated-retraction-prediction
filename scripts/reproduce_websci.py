"""Faithful reproduction of the WebSci'25 pipelines on our laptop.

Track 1: ConventionalModelClassification_Ranking.py logic — his released
features, his preprocessing (comma->dot, fillna(0)), his single 80/20 split
with random_state=42, his model zoo.
Track 2: BertBasedClassification.py logic — flat section concatenation,
max_len=128, 1 epoch, batch 8, same split. (BERT-base + SciBERT + BioBERT.)

Targets (his slides): RF/DT 0.88, XGB 0.87, GB 0.86, KNN 0.71, SVM 0.67,
LR 0.52, GNB 0.42 | BioBERT 0.55, SciBERT 0.51, BERT-base 0.42.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data/raw/Scientific-Accountability-main/Retracted_NonRetracted Articles.csv"

FEATURE_COLS = [
    "abstract_Readability", "Method_Readability", "Results_Readability",
    "Conclusion_Readability", "Avg_Readability", "Avg_Certainty",
    "Abstract_Certainty", "Method_Certainty", "Result_Certainty",
    "Conclusion_Certainty",
]


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV, sep=";", engine="python", quotechar='"', on_bad_lines="skip")
    df["Label"] = df["Label"].str.strip().map({"R": 1, "NR": 0})
    return df


def track1() -> None:
    df = load()
    for c in FEATURE_COLS:  # his exact preprocessing
        df[c] = df[c].astype(str).str.replace(",", ".").astype(float)
    df = df.fillna(0)
    X, y = df[FEATURE_COLS], df["Label"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    Xtr_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xte)
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(),
        "Naive Bayes": GaussianNB(),
    }
    his = {"Logistic Regression": 0.52, "Random Forest": 0.88, "Decision Tree": 0.88,
           "XGBoost": 0.87, "Gradient Boosting": 0.86, "KNN": 0.71, "SVM": 0.67,
           "Naive Bayes": 0.42}
    print(f"{'model':22s} {'ours':>6s} {'his':>6s}   f1")
    rows = []
    for name, m in models.items():
        m.fit(Xtr_s, ytr)
        p = m.predict(Xte_s)
        acc, f1 = accuracy_score(yte, p), f1_score(yte, p)
        rows.append({"model": name, "acc_ours": acc, "acc_his": his.get(name), "f1": f1})
        print(f"{name:22s} {acc:6.3f} {his.get(name, float('nan')):6.2f}  {f1:5.3f}")
    pd.DataFrame(rows).to_csv(ROOT / "results/reproduce_track1.csv", index=False)


def track2() -> None:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    df = load()
    cols = ["abstract", "Materials_and_Methods", "Conclusion", "Results"]
    df[cols] = df[cols].fillna("")
    texts = df[cols].agg(" ".join, axis=1)
    Xtr, Xte, ytr, yte = train_test_split(texts, df["Label"], test_size=0.2,
                                          random_state=42)

    class DS(Dataset):
        def __init__(self, t, y, tok):
            self.t, self.y, self.tok = t.reset_index(drop=True), y.reset_index(drop=True), tok

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            e = self.tok(str(self.t[i]), max_length=128, padding="max_length",
                         truncation=True, return_tensors="pt")
            return {"input_ids": e.input_ids[0], "attention_mask": e.attention_mask[0],
                    "label": torch.tensor(int(self.y[i]))}

    MODELS = {"bert-base": "bert-base-uncased",
              "scibert": "allenai/scibert_scivocab_uncased",
              "biobert": "dmis-lab/biobert-base-cased-v1.1"}
    his = {"bert-base": 0.42, "scibert": 0.51, "biobert": 0.55}
    rows = []
    for name, ckpt in MODELS.items():
        tok = AutoTokenizer.from_pretrained(ckpt)
        model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
        dl = DataLoader(DS(Xtr, ytr, tok), batch_size=8, shuffle=True)
        model.train()
        for batch in dl:  # 1 epoch, faithful
            opt.zero_grad()
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                        labels=batch["label"])
            out.loss.backward()
            opt.step()
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in DataLoader(DS(Xte, yte, tok), batch_size=16):
                logits = model(input_ids=batch["input_ids"],
                               attention_mask=batch["attention_mask"]).logits
                preds.extend(logits.argmax(-1).tolist())
        acc = accuracy_score(yte, preds)
        f1 = f1_score(yte, preds)
        rows.append({"model": name, "acc_ours": acc, "acc_his": his[name], "f1": f1})
        print(f"{name:10s} ours {acc:.3f} | his {his[name]:.2f} | f1 {f1:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results/reproduce_track2.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["1", "2", "both"], default="1")
    a = ap.parse_args()
    if a.track in ("1", "both"):
        track1()
    if a.track in ("2", "both"):
        track2()
