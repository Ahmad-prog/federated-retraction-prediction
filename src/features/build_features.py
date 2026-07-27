"""Feature engineering.

Feature families mirror Usman & Balke (WebSci'25: readability/certainty of text;
TPDL'24: metadata + citation dynamics), computed here on title+abstract since
full text is the very thing silos cannot share.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import textstat

ROOT = Path(__file__).resolve().parents[2]

HEDGE_WORDS = {
    "may", "might", "could", "possibly", "perhaps", "suggest", "suggests",
    "appear", "appears", "likely", "unlikely", "probable", "presumably",
    "putative", "potential", "potentially", "seems", "seem", "tend", "tends",
}
CERTAINTY_WORDS = {
    "demonstrate", "demonstrates", "prove", "proves", "proven", "confirm",
    "confirms", "establish", "establishes", "clearly", "definitely",
    "significantly", "robust", "conclusive", "undoubtedly", "show", "shows",
}


def text_features(text: str) -> dict:
    text = (text or "").strip()
    if len(text) < 40:
        return {
            "flesch_reading_ease": np.nan, "flesch_kincaid": np.nan,
            "gunning_fog": np.nan, "smog": np.nan, "abstract_len_words": 0,
            "hedge_ratio": np.nan, "certainty_ratio": np.nan,
        }
    words = text.lower().split()
    n = len(words)
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid": textstat.flesch_kincaid_grade(text),
        "gunning_fog": textstat.gunning_fog(text),
        "smog": textstat.smog_index(text),
        "abstract_len_words": n,
        "hedge_ratio": sum(w in HEDGE_WORDS for w in words) / n,
        "certainty_ratio": sum(w in CERTAINTY_WORDS for w in words) / n,
    }


NUMERIC_FEATURES = [
    "n_authors", "n_countries", "n_references", "cited_by_count",
    "early_citations_2y", "year", "is_oa", "is_international",
    "flesch_reading_ease", "flesch_kincaid", "gunning_fog", "smog",
    "abstract_len_words", "hedge_ratio", "certainty_ratio",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "data/processed/corpus.parquet"))
    ap.add_argument("--out", default=str(ROOT / "data/processed/features.parquet"))
    args = ap.parse_args()

    df = pd.read_parquet(args.corpus)
    tf = pd.DataFrame([text_features(t) for t in df["abstract"]])
    feats = pd.concat([df.reset_index(drop=True), tf], axis=1)
    feats["is_oa"] = feats["is_oa"].astype(float)
    feats["is_international"] = feats["is_international"].astype(float)
    keep = ["doi", "retracted", "publisher", "venue", "field", "subfield"] + NUMERIC_FEATURES
    feats[keep].to_parquet(args.out, index=False)
    print(f"Saved {len(feats)} rows, {len(NUMERIC_FEATURES)} numeric features -> {args.out}")


if __name__ == "__main__":
    main()
