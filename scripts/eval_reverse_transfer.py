"""Reverse transfer: train Usman-style RF on his 464-paper corpus, test on our
138k corpus. Uses only features computable identically on both sides — the
abstract text features (readability, hedging, certainty, length).

Expected outcome: poor generalization from a few hundred hand-curated
biomedical papers to the full multi-domain corpus — the scale argument.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.build_features import text_features
from src.models.evaluate import evaluate

TEXT_FEATURES = ["flesch_reading_ease", "flesch_kincaid", "gunning_fog", "smog",
                 "abstract_len_words", "hedge_ratio", "certainty_ratio"]


def main() -> None:
    raw = pd.read_csv(
        ROOT / "data/raw/Scientific-Accountability-main/Retracted_NonRetracted Articles.csv",
        sep=";", engine="python", quotechar='"', on_bad_lines="skip")
    raw["retracted"] = (raw["Label"].str.strip() == "R").astype(float)
    tf = pd.DataFrame([text_features(str(a) if pd.notna(a) else "") for a in raw["abstract"]])
    Xtr = tf[TEXT_FEATURES].to_numpy(dtype=np.float32)
    ytr = raw["retracted"].to_numpy(dtype=np.float32)

    ours = pd.read_parquet(ROOT / "data/processed/features.parquet")
    Xte = ours[TEXT_FEATURES].to_numpy(dtype=np.float32)
    yte = ours["retracted"].to_numpy(dtype=np.float32)

    imp = SimpleImputer(strategy="median")
    Xtr_i = imp.fit_transform(Xtr)
    Xte_i = imp.transform(Xte)

    results = {}
    # his direction: small hand-curated corpus -> full corpus
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                n_jobs=4, random_state=42)
    rf.fit(Xtr_i, ytr)
    results["usman_rf_on_138k"] = evaluate(yte, rf.predict_proba(Xte_i)[:, 1])

    # symmetric check: same text-only features, our corpus -> his corpus
    rf2 = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 n_jobs=4, random_state=42)
    rf2.fit(Xte_i, yte)
    results["ours_textonly_on_usman"] = evaluate(ytr, rf2.predict_proba(Xtr_i)[:, 1])

    # sanity: his corpus, within-corpus 5-fold CV (does his signal replicate?)
    from sklearn.model_selection import cross_val_predict
    rf3 = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 n_jobs=4, random_state=42)
    scores = cross_val_predict(rf3, Xtr_i, ytr, cv=5, method="predict_proba")[:, 1]
    m = evaluate(ytr, scores)
    m["balanced_accuracy_at_0.5"] = float(((scores > 0.5) == (ytr > 0.5)).mean())
    results["usman_within_corpus_cv"] = m

    res = pd.DataFrame(results).T
    res.to_csv(ROOT / "results" / "reverse_transfer.csv")
    print(res.round(3).to_string())
    print(f"\nbaselines: our corpus positive rate = {yte.mean():.3f}, "
          f"his corpus positive rate = {ytr.mean():.3f}")


if __name__ == "__main__":
    main()
