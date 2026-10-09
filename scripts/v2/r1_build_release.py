"""R1: build the FedRetract release package (Zenodo).

release/FedRetract/
  benchmark/articles.parquet     121,594 leakage-audited research articles (features, labels, text)
  benchmark/splits.parquet       publisher (K=10) and field silos, train/val/test for seeds 42-51
  fulltext/index.parquet         all 14,172 full-text articles: ids, labels, licence, splits, flags
  fulltext/text_cc_by.parquet    cleaned sections, CC BY / CC0 articles
  fulltext/text_cc_by_nc.parquet cleaned sections, CC BY-NC / CC BY-NC-SA articles (non-commercial)
  fulltext/matching_cells.parquet journal x year cells used to match controls
  prospective/articles.parquet   retractions added after the 2026-07-19 snapshot
  audit/                         leakage-audit reports
  embeddings/                    frozen-encoder embeddings (abstracts; full text for redistributable articles)
  code/                          cleaning + benchmark code needed to rebuild everything
  stats.json                     numbers quoted in the datasheet
Articles under CC BY-NC-ND or without a reuse licence are released as identifiers + labels
only; code/rebuild_fulltext.py re-downloads and re-cleans them.
"""
from __future__ import annotations

import gzip
import json
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.v2.b2_main_benchmark import partition, split  # noqa: E402
from src.v2 import DATA2, RES2, ROOT  # noqa: E402

OUT = ROOT / "release" / "FedRetract"
SEEDS = range(42, 52)
SECTIONS = ["title", "abstract", "introduction", "methods", "results", "discussion", "conclusion", "body"]
BENCH_DROP = ["raw_title", "retracted_title_flag", "cited_by_count_2026", "nonresearch", "english",
              "abstract_is_notice"]  # leaky (raw title, post-hoc citations) or constant after filtering


def licence(pmcid: str) -> str:
    try:
        x = gzip.open(DATA2 / "fulltext_xml" / f"{pmcid}.xml.gz", "rb").read()
    except OSError:
        return "unknown"
    m = re.search(rb"<license[^>]*>(.*?)</license>", x, re.S)
    href = re.search(rb"<license[^>]*(?:xlink:)?href=\"([^\"]+)\"", x) or \
        re.search(rb"<ali:license_ref[^>]*>([^<]+)<", x)
    s = ((href.group(1) if href else b"") + b" " + (m.group(1)[:400] if m else b"")).decode("utf8", "ignore").lower()
    if "by-nc-nd" in s or "noncommercial-noderiv" in s:
        return "CC BY-NC-ND"
    if "by-nc-sa" in s:
        return "CC BY-NC-SA"
    if "by-nc" in s or "noncommercial" in s or "non-commercial" in s:
        return "CC BY-NC"
    if "by-nd" in s or "noderiv" in s:
        return "CC BY-ND"
    if "by-sa" in s:
        return "CC BY-SA"
    if "creativecommons.org/publicdomain" in s or "cc0" in s or "public domain" in s:
        return "CC0"
    if "creativecommons.org/licenses/by" in s or "cc by" in s or "cc-by" in s or \
            "creative commons attribution" in s:
        return "CC BY"
    return "other" if s.strip() else "none stated"


def ft_splits(df: pd.DataFrame, k: int = 8) -> pd.DataFrame:
    """Same silos and splits as scripts/v2/c3_lora_fl.py:load_dataset."""
    out = pd.DataFrame({"pmcid": df["pmcid"]})
    pub = df["publisher"].fillna("Unknown")
    top = pub[df.split == "main"].value_counts().index[: k - 1]
    out["silo_publisher"] = pub.where(pub.isin(top), "Other")
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        part = pd.Series("prospective", index=df.index)
        main = df[df.split == "main"].assign(silo=out["silo_publisher"])
        for _, idx in main.groupby(["silo", "retracted"]).groups.items():
            idx = rng.permutation(np.array(idx))
            n_te, n_va = int(round(0.2 * len(idx))), int(round(0.1 * len(idx)))
            part.loc[idx[:n_te]] = "test"
            part.loc[idx[n_te:n_te + n_va]] = "val"
            part.loc[idx[n_te + n_va:]] = "train"
        out[f"part_seed{seed}"] = part.to_numpy()
    return out


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for d in ["benchmark", "fulltext", "prospective", "audit", "embeddings", "code"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)
    stats = {}

    # ---------------------------------------------------------------- benchmark
    df = pd.read_parquet(DATA2 / "tabular_v2.parquet").reset_index(drop=True)
    sp = pd.DataFrame({"doi": df["doi"], "silo_publisher": partition(df, "publisher"),
                       "silo_field": partition(df, "field")})
    for by in ["publisher", "field"]:  # B2 stratifies the split by silo x label for each partition
        d = df.assign(silo=sp[f"silo_{by}"])
        for seed in SEEDS:
            sp[f"part_{by}_seed{seed}"] = split(d, seed)
    sp.to_parquet(OUT / "benchmark" / "splits.parquet", index=False)
    bench = df.drop(columns=[c for c in BENCH_DROP if c in df])
    bench.to_parquet(OUT / "benchmark" / "articles.parquet", index=False)
    y = df["retracted"]
    stats["benchmark"] = {
        "n": len(df), "n_retracted": int(y.sum()), "n_control": int((y == 0).sum()),
        "has_abstract_share": {"retracted": float(df.loc[y == 1, "has_abstract"].mean()),
                               "control": float(df.loc[y == 0, "has_abstract"].mean())},
        "year_min": int(df["year"].min()), "year_max": int(df["year"].max()),
        "n_publishers": int(df["publisher"].nunique()), "n_venues": int(df["venue_id"].nunique()),
        "n_fields": int(df["field"].nunique()), "n_subfields": int(df["subfield"].nunique()),
        "silo_publisher_sizes": sp.groupby("silo_publisher").size().to_dict(),
        "silo_publisher_retracted_rate": df.groupby(sp["silo_publisher"])["retracted"].mean().round(3).to_dict(),
        "top_fields": df["field"].value_counts().head(10).to_dict(),
        "top_first_author_countries": df["first_author_country"].fillna("unknown").str.split(";").str[0]
        .value_counts().head(10).to_dict(),
        "year_counts": df.groupby(["year", "retracted"]).size().unstack(fill_value=0).to_dict(),
        "columns": list(bench.columns),
    }

    # ---------------------------------------------------------------- full text
    ft = pd.read_parquet(DATA2 / "ft_dataset.parquet").reset_index(drop=True)
    with ProcessPoolExecutor(32) as ex:
        ft["licence"] = list(ex.map(licence, ft["pmcid"], chunksize=200))
    redistributable = {"CC BY": "cc_by", "CC0": "cc_by", "CC BY-SA": "cc_by",
                       "CC BY-NC": "cc_by_nc", "CC BY-NC-SA": "cc_by_nc"}
    ft["text_file"] = ft["licence"].map(redistributable).fillna("none (rebuild script)")
    reason = pd.concat([df[["doi", "reason", "retraction_date"]],
                        pd.read_parquet(DATA2 / "prospective_positives.parquet")
                        .reindex(columns=["doi", "reason", "retraction_date"])]).drop_duplicates("doi")
    ft = ft.merge(reason, on="doi", how="left")
    meta_cols = ["pmcid", "doi", "openalex_id", "retracted", "retraction_date", "reason", "split",
                 "issn", "pubYear", "year", "publisher", "venue", "venue_id", "issn_l", "field", "subfield",
                 "n_authors", "n_countries", "countries", "first_author_country", "is_oa", "n_references",
                 "article_type", "n_body_words", "n_sections_found", "n_leak_sentences_removed",
                 "licence", "text_file"]
    idx = ft[[c for c in meta_cols if c in ft]].merge(ft_splits(ft), on="pmcid")
    idx.to_parquet(OUT / "fulltext" / "index.parquet", index=False)
    for key in ["cc_by", "cc_by_nc"]:
        sub = ft[ft["text_file"] == key]
        sub[["pmcid", "doi", "retracted", "licence"] + SECTIONS].to_parquet(
            OUT / "fulltext" / f"text_{key}.parquet", index=False)
    shutil.copy(DATA2 / "ft_cells.parquet", OUT / "fulltext" / "matching_cells.parquet")
    stats["fulltext"] = {
        "n": len(ft), "n_retracted": int(ft["retracted"].sum()), "n_control": int((ft["retracted"] == 0).sum()),
        "by_split": ft.groupby(["split", "retracted"]).size().unstack(fill_value=0).to_dict(),
        "licence_by_class": ft.groupby(["licence", "retracted"]).size().unstack(fill_value=0).to_dict(),
        "text_file_counts": ft["text_file"].value_counts().to_dict(),
        "median_body_words": float(ft["n_body_words"].median()),
        "share_ge3_sections": float((ft["n_sections_found"] >= 3).mean()),
        "share_docs_with_leak_sentences_removed": float((ft["n_leak_sentences_removed"] > 0).mean()),
        "silo_sizes": idx.groupby("silo_publisher").size().to_dict(),
    }

    # ---------------------------------------------------------------- prospective
    pr = pd.read_parquet(DATA2 / "prospective_abstracts.parquet")
    pr.to_parquet(OUT / "prospective" / "articles.parquet", index=False)
    stats["prospective"] = {"n": len(pr), "with_abstract": int(pr["has_abstract"].sum()),
                            "snapshot_date": "2026-07-19"}
    rw = json.loads((DATA2 / "rw_stats.json").read_text()) if (DATA2 / "rw_stats.json").exists() else {}
    stats["retraction_watch_snapshot"] = rw

    # ---------------------------------------------------------------- audit + embeddings
    for f in RES2.glob("d1_leak_audit_*.txt"):
        shutil.copy(f, OUT / "audit" / f.name.replace("d1_", ""))
    emb = DATA2 / "emb"
    keep_ft = set(ft.loc[ft["text_file"] != "none (rebuild script)", "pmcid"])
    for f in sorted(emb.glob("*.npy")):
        ids = (emb / f"{f.stem}.ids.txt").read_text().split("\n")
        E = np.load(f)
        if f.stem.startswith("fulltext"):
            m = np.array([i in keep_ft for i in ids])
            E, ids = E[m], [i for i, k in zip(ids, m) if k]
        np.save(OUT / "embeddings" / f"{f.stem}.npy", E.astype(np.float16))
        (OUT / "embeddings" / f"{f.stem}.ids.txt").write_text("\n".join(ids))
    stats["embeddings"] = {f.stem: list(np.load(f, mmap_mode="r").shape)
                           for f in sorted((OUT / "embeddings").glob("*.npy"))}

    # ---------------------------------------------------------------- code
    code = OUT / "code"
    for rel in ["src/__init__.py", "src/v2", "src/features/build_features.py", "src/features/__init__.py",
                "src/models/evaluate.py", "src/models/__init__.py", "scripts/v2"]:
        p = ROOT / rel
        if not p.exists():
            continue
        dst = code / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if p.is_dir():
            shutil.copytree(p, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy(p, dst)
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1, default=str))
    print(json.dumps({k: {kk: v for kk, v in s.items() if not isinstance(v, (dict, list))}
                      for k, s in stats.items() if isinstance(s, dict)}, indent=1))
    print(json.dumps(stats["fulltext"]["text_file_counts"]))


if __name__ == "__main__":
    main()
