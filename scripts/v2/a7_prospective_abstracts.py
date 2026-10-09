"""A7: title + abstract of the prospective positives (retracted after 2026-07-19), cleaned
exactly like the main corpus, for scoring text models on the prospective test (B5 uses the
same OpenAlex records for the tabular models).

Output: data_v2/prospective_abstracts.parquet
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.v2.b1_tabular_features import row as oa_row  # noqa: E402
from src.v2 import DATA2  # noqa: E402
from src.v2.cleaning import clean_abstract, clean_title, is_english, is_nonresearch  # noqa: E402


def main() -> None:
    works = json.loads((DATA2 / "cache" / "openalex_prospective.json").read_text())
    p = pd.DataFrame([oa_row(w) for w in works]).drop_duplicates("doi")
    p = p[[not is_nonresearch(t, a) and is_english(f"{t} {a}")
           for t, a in zip(p["raw_title"], p["raw_abstract"])]].copy()
    p["title"] = p["raw_title"].map(clean_title)
    p["abstract"] = [clean_abstract(a)[0] for a in p["raw_abstract"]]
    p["has_abstract"] = (p["abstract"].str.len() >= 40).astype(float)
    p["retracted"] = 1
    pros = pd.read_parquet(DATA2 / "prospective_positives.parquet")
    if "reason" in pros:
        p = p.merge(pros[["doi", "reason"]].drop_duplicates("doi"), on="doi", how="left",
                    suffixes=("_oa", ""))
    keep = ["doi", "title", "abstract", "has_abstract", "publisher", "first_author_country",
            "year", "retracted"] + (["reason"] if "reason" in p else [])
    p = p[keep].reset_index(drop=True)
    p.to_parquet(DATA2 / "prospective_abstracts.parquet", index=False)
    print(f"prospective abstracts: {len(p)} (with abstract: {int(p.has_abstract.sum())})")


if __name__ == "__main__":
    main()
