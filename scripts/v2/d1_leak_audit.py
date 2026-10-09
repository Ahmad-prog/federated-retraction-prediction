"""D1: leakage audit of the cleaned text (R0.4, C3).

For --source abstracts (138K) or fulltext:
  1. keyword-only classifier on notice vocabulary -> must be ~chance after cleaning
  2. TF-IDF + logistic regression (5-fold CV) on title / abstract / both (or full text)
  3. strongest positive / negative n-grams (are any of them notice artefacts?)
  4. sample of the highest-scored texts for manual inspection
Output: results_v2/d1_leak_audit_<source>.txt
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.evaluate import evaluate  # noqa: E402
from src.v2 import DATA2, RES2  # noqa: E402

NOTICE_VOCAB = ["retract", "retracted", "retraction", "withdrawn", "withdraw", "erratum",
                "corrigendum", "correction", "concern", "notice", "editor", "editors",
                "publisher", "misconduct", "integrity", "investigation", "compromised",
                "manipulated", "unreliable", "duplicated", "plagiarism", "fabricated"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["abstracts", "fulltext"], default="abstracts")
    a = ap.parse_args()
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(str(s))

    if a.source == "abstracts":
        df = pd.read_parquet(DATA2 / "tabular_v2.parquet",
                             columns=["doi", "retracted", "title", "abstract", "publisher"])
        views = {"title": df["title"], "abstract": df["abstract"],
                 "title+abstract": df["title"] + ". " + df["abstract"]}
    else:
        df = pd.read_parquet(DATA2 / "ft_dataset.parquet")
        df = df[df["split"] == "main"]
        views = {"title+abstract": df["title"].fillna("") + ". " + df["abstract"].fillna(""),
                 "methods": df["methods"].fillna(""), "body": df["body"].fillna("").str[:20000]}
    y = df["retracted"].to_numpy()
    out(f"source={a.source} n={len(df)} positive rate={y.mean():.3f}")
    skf = StratifiedKFold(5, shuffle=True, random_state=0)

    for name, text in views.items():
        text = text.fillna("").str.lower()
        kw = CountVectorizer(vocabulary=NOTICE_VOCAB, token_pattern=r"[a-z]+").fit_transform(text)
        out(f"\n[{name}] docs containing any notice-vocabulary word: "
            f"pos {np.mean(kw[y == 1].sum(1) > 0):.4f} | neg {np.mean(kw[y == 0].sum(1) > 0):.4f}")
        if kw.sum() > 0:
            s = cross_val_predict(LogisticRegression(max_iter=2000), kw, y, cv=skf,
                                  method="predict_proba")[:, 1]
            out(f"[{name}] keyword-only classifier: {evaluate(y, s)}")
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=20, max_features=200000,
                              sublinear_tf=True)
        Xt = vec.fit_transform(text)
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
        s = cross_val_predict(clf, Xt, y, cv=skf, method="predict_proba")[:, 1]
        out(f"[{name}] TF-IDF logistic regression (5-fold): {evaluate(y, s)}")
        clf.fit(Xt, y)
        vocab = np.array(vec.get_feature_names_out())
        order = np.argsort(clf.coef_[0])
        out(f"[{name}] top +retracted n-grams: {', '.join(vocab[order[-60:]][::-1])}")
        out(f"[{name}] top -retracted n-grams: {', '.join(vocab[order[:40]])}")
        flagged = [v for v in vocab[order[-300:]] if re.search(r"retract|withdr|errat|corrig|notice|concern", v)]
        out(f"[{name}] notice-like tokens among top-300 positive n-grams: {flagged or 'none'}")
        if name in ("title+abstract", "body"):
            top = np.argsort(-s)[:12]
            out(f"[{name}] highest-scored documents (label, score, publisher, text[:240]):")
            for i in top:
                out(f"   y={y[i]} s={s[i]:.3f} {df['publisher'].iloc[i] if 'publisher' in df else ''} | "
                    f"{text.iloc[i][:240]}")
    (RES2 / f"d1_leak_audit_{a.source}.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
