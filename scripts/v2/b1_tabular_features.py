"""B1: leakage-clean, time-aware tabular features for the 138K corpus (C3, C4, M12).

Changes vs v1 (src/features/build_features.py):
  * controls whose DOI appears anywhere in Retraction Watch are removed (M12)
  * abstracts: retraction/withdrawal sentences removed (same rule as full text);
    records whose abstract IS a notice are dropped; a missing abstract gives NaN
    text features plus has_abstract = 0 (v1 coded it as length 0)
  * citations are computed for an explicit screening time:
      - setting "pub":  no citation features at all (screening at publication)
      - setting "pub2": citations received in publication years y..y+2, only for
        articles whose window is fully observed in OpenAlex counts_by_year and
        (for retracted articles) that were not already retracted at y+2
    v1 used cited_by_count measured in 2026 (includes post-retraction citations)
  * certainty/hedging lexicon features are kept but labelled as lexicon proxies;
    the model-based certainty features come from the GPU track (R2.8)

Output: data_v2/tabular_v2.parquet (one row per article; setting flags as columns)
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.build_features import text_features  # noqa: E402
from src.v2 import DATA2, ROOT  # noqa: E402
from src.v2.cleaning import clean_abstract as _clean_abs, clean_title, is_english, is_nonresearch  # noqa: E402

LEAK = re.compile(r"retract|withdraw|expression of concern|erratum|corrigend", re.I)
NOTICE = re.compile(r"^\s*(this (article|paper|manuscript) (has been|was|is) "
                    r"(retracted|withdrawn)|retraction|notice of retraction|retracted)", re.I)
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")
CBY_MIN_YEAR = 2016      # first year reliably covered by OpenAlex counts_by_year (fetched 2026)
SNAPSHOT_YEAR = 2026


def abstract_from_inv(inv):
    if not inv:
        return ""
    pos = {}
    for w, idx in inv.items():
        for i in idx:
            pos[i] = w
    return " ".join(pos[i] for i in sorted(pos))


def clean_abstract(text: str) -> tuple[str, bool, int]:
    return _clean_abs(text)


def load_works(sub: str) -> list[dict]:
    rows = []
    for f in sorted(glob.glob(str(ROOT / "data" / "cache" / sub / "*.json"))):
        rows.extend(json.loads(Path(f).read_text()))
    return rows


def row(w: dict) -> dict:
    loc = w.get("primary_location") or {}
    src = loc.get("source") or {}
    topic = w.get("primary_topic") or {}
    auths = w.get("authorships") or []
    countries = {c for a in auths for c in (a.get("countries") or [])}
    cby = {c["year"]: c["cited_by_count"] for c in (w.get("counts_by_year") or [])}
    return {
        "doi": (w.get("doi") or "").replace("https://doi.org/", "").lower(),
        "openalex_id": w.get("id"), "year": w.get("publication_year"),
        "pub_date": w.get("publication_date"),
        "publisher": src.get("host_organization_name"), "venue": src.get("display_name"),
        "venue_id": src.get("id"), "issn_l": src.get("issn_l"),
        "field": (topic.get("field") or {}).get("display_name"),
        "subfield": (topic.get("subfield") or {}).get("display_name"),
        "n_authors": len(auths), "n_countries": len(countries),
        "countries": ";".join(sorted(countries)),
        "first_author_country": ";".join((auths[0].get("countries") or [])) if auths else "",
        "is_oa": float(bool((w.get("open_access") or {}).get("is_oa"))),
        "n_references": w.get("referenced_works_count", 0),
        "cby": cby, "cited_by_count_2026": w.get("cited_by_count", 0),
        "raw_title": w.get("title") or "",
        "raw_abstract": abstract_from_inv(w.get("abstract_inverted_index")),
    }


def main() -> None:
    corpus = pd.read_parquet(ROOT / "data/processed/corpus.parquet", columns=["doi", "retracted"])
    label = dict(zip(corpus["doi"], corpus["retracted"]))
    works = [row(w) for w in load_works("positives") + load_works("controls_pub")]
    df = pd.DataFrame(works).drop_duplicates("doi")
    df = df[df["doi"].isin(label)].copy()
    df["retracted"] = df["doi"].map(label).astype(int)
    print(f"v1 corpus rows matched to cached OpenAlex records: {len(df)} / {len(label)}")

    rw_all = set((DATA2 / "rw_all_dois.txt").read_text().split("\n"))
    contaminated = (df["retracted"] == 0) & df["doi"].isin(rw_all)
    print(f"controls that appear in Retraction Watch (removed, M12): {int(contaminated.sum())}")
    df = df[~contaminated]

    rw = pd.read_csv(DATA2 / "raw" / "retraction_watch_2026-10-07.csv", low_memory=False,
                     encoding_errors="replace")
    rw["doi"] = rw["OriginalPaperDOI"].astype(str).str.strip().str.lower()
    rw["rdate"] = pd.to_datetime(rw["RetractionDate"], errors="coerce", format="mixed")
    rw = rw[rw["RetractionNature"].str.strip().str.lower() == "retraction"]
    rdate = rw.groupby("doi")["rdate"].min()
    df["retraction_date"] = df["doi"].map(rdate)
    df["reason"] = df["doi"].map(rw.drop_duplicates("doi").set_index("doi")["Reason"])

    df["title"] = df["raw_title"].map(clean_title)
    nonres = [is_nonresearch(t, a) for t, a in zip(df["raw_title"], df["raw_abstract"])]
    df["nonresearch"] = nonres
    print("non-research items (reviews, case reports, letters, ...) by class -> removed:")
    print(df.groupby("retracted")["nonresearch"].agg(["sum", "mean"]).round(4).to_string())
    df = df[~df["nonresearch"]].copy()
    df["english"] = [is_english(f"{t} {a}") for t, a in zip(df["raw_title"], df["raw_abstract"])]
    print("non-English items by class -> removed:")
    print(df.groupby("retracted")["english"].apply(lambda s: (~s).agg(["sum", "mean"])).round(4).to_string())
    df = df[df["english"]].copy()
    cleaned = [clean_abstract(a) for a in df["raw_abstract"]]
    df["abstract"] = [c[0] for c in cleaned]
    df["abstract_is_notice"] = [c[1] for c in cleaned]
    df["abstract_leak_sentences"] = [c[2] for c in cleaned]
    print("abstract notice-contaminated (sentences removed or abstract dropped), by class:")
    print(df.groupby("retracted")["abstract_is_notice"].mean().round(4).to_string())
    df["has_abstract"] = (df["abstract"].str.len() >= 40).astype(float)
    print("has_abstract by class:")
    print(df.groupby("retracted")["has_abstract"].mean().round(4).to_string())

    tf = pd.DataFrame([text_features(a) if len(a) >= 40 else {} for a in df["abstract"]],
                      index=df.index)
    if "abstract_len_words" in tf:
        tf.loc[df["has_abstract"] == 0, "abstract_len_words"] = np.nan
    df = pd.concat([df, tf], axis=1)

    y = df["year"].astype("Int64")
    early = []
    for cby, yr in zip(df["cby"], y):
        early.append(np.nan if pd.isna(yr) else float(sum(v for k, v in cby.items()
                                                            if yr <= k <= yr + 2)))
    df["cites_pub2"] = early
    ryear = df["retraction_date"].dt.year
    df["in_setting_pub2"] = (
        y.notna() & (y >= CBY_MIN_YEAR) & (y + 2 <= SNAPSHOT_YEAR - 1)
        & ~((df["retracted"] == 1) & ryear.notna() & (ryear <= y + 2))
    ).astype(bool)
    df.loc[~df["in_setting_pub2"], "cites_pub2"] = np.nan
    df["in_setting_pub"] = True

    df["retracted_title_flag"] = df["raw_title"].str.contains("retract", case=False)
    out = df.drop(columns=["cby", "raw_abstract"])
    out.to_parquet(DATA2 / "tabular_v2.parquet", index=False)
    print(f"\nrows: {len(out)}; positives {int(out.retracted.sum())}")
    print(f"setting pub2 rows: {int(out.in_setting_pub2.sum())} "
          f"(pos rate {out.loc[out.in_setting_pub2, 'retracted'].mean():.3f})")


if __name__ == "__main__":
    main()
