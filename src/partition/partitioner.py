"""Non-IID silo partitioners.

Silos model organizations that cannot pool data:
- by_publisher: top-(K-1) publishers + "Other" bucket
- by_field:     top-(K-1) OpenAlex fields + "Other" bucket

Both are naturally non-IID: silo size, feature distributions, and retraction
rates all differ across publishers/fields.
"""
from __future__ import annotations

import pandas as pd


def partition(df: pd.DataFrame, by: str, k: int = 10, min_rows: int = 200) -> pd.DataFrame:
    """Add a 'silo' column. `by` is 'publisher' or 'field'."""
    col = {"publisher": "publisher", "field": "field"}[by]
    counts = df[col].fillna("Unknown").value_counts()
    top = [p for p in counts.index[: k - 1] if counts[p] >= min_rows]
    out = df.copy()
    out["silo"] = out[col].fillna("Unknown").where(out[col].isin(top), "Other")
    return out


def silo_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-silo statistics table (goes straight into the paper)."""
    g = df.groupby("silo")
    stats = pd.DataFrame({
        "n": g.size(),
        "n_retracted": g["retracted"].sum(),
        "retraction_rate": g["retracted"].mean().round(4),
        "median_year": g["year"].median(),
    }).sort_values("n", ascending=False)
    return stats.reset_index()
