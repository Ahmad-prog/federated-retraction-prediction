"""Time-aware author retraction-history features from the Retraction Watch CSV.

For each corpus paper: did any of its authors appear on a retraction whose
retraction date precedes this paper's publication date? Only past events are
used, so the feature is deployable and leak-free.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _norm_name(n: str) -> str:
    n = re.sub(r"[^a-z ]", "", n.lower().strip())
    parts = [p for p in n.split() if p]
    if not parts:
        return ""
    # surname + first initial: robust to middle-name variation
    return f"{parts[-1]} {parts[0][0]}" if len(parts) > 1 else parts[0]


def build_author_index() -> dict[str, list[pd.Timestamp]]:
    rw = pd.read_csv(ROOT / "data/raw/retraction_watch.csv", low_memory=False,
                     encoding_errors="replace")
    rw = rw[rw["RetractionNature"].str.strip().str.lower() == "retraction"]
    rw["rdate"] = pd.to_datetime(rw["RetractionDate"], errors="coerce")
    idx: dict[str, list] = defaultdict(list)
    for authors, rdate in zip(rw["Author"], rw["rdate"]):
        if pd.isna(authors) or pd.isna(rdate):
            continue
        for a in str(authors).split(";"):
            key = _norm_name(a)
            if key:
                idx[key].append(rdate)
    return {k: sorted(v) for k, v in idx.items()}


def author_history_features(corpus: pd.DataFrame, author_lists: pd.Series) -> pd.DataFrame:
    """corpus needs 'pub_date' (or 'year'); author_lists = ';'-joined author names."""
    idx = build_author_index()
    pub = pd.to_datetime(corpus.get("pub_date"), errors="coerce")
    pub = pub.fillna(pd.to_datetime(corpus["year"].astype("Int64").astype(str),
                                    format="%Y", errors="coerce"))
    n_prior, any_prior = [], []
    for authors, pdate in zip(author_lists, pub):
        count = 0
        if pd.notna(pdate) and isinstance(authors, str) and authors:
            for a in authors.split(";"):
                key = _norm_name(a)
                events = idx.get(key)
                if events:
                    count += sum(1 for e in events if e < pdate)
        n_prior.append(count)
        any_prior.append(1.0 if count > 0 else 0.0)
    return pd.DataFrame({"n_prior_author_retractions": np.array(n_prior, dtype=np.float32),
                         "any_prior_author_retraction": np.array(any_prior, dtype=np.float32)})
