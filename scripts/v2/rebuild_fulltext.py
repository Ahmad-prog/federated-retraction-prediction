"""Rebuild the cleaned full text of FedRetract articles whose licence does not allow us to
redistribute text (CC BY-NC-ND, other, none stated), or of all articles.

Downloads the JATS XML from Europe PMC (NCBI PMC OAI-PMH fallback) and applies exactly the
parsing and cleaning used to build the dataset (scripts/v2/a5_parse_clean.py).

usage: python scripts/v2/rebuild_fulltext.py --index fulltext/index.parquet --out rebuilt.parquet [--all]
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.v2.a5_parse_clean import parse  # noqa: E402
from src.v2.epmc import fetch_fulltext_xml  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--xml-dir", default="fulltext_xml")
    ap.add_argument("--all", action="store_true", help="rebuild every article, not only the missing ones")
    a = ap.parse_args()
    idx = pd.read_parquet(a.index)
    if not a.all:
        idx = idx[idx["text_file"].str.startswith("none")]
    xml = Path(a.xml_dir)
    xml.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(8) as ex:
        status = list(ex.map(lambda p: fetch_fulltext_xml(p, xml), idx["pmcid"]))
    print(pd.Series(status).value_counts().to_string())
    rows = [parse(xml / f"{p}.xml.gz") for p in idx["pmcid"] if (xml / f"{p}.xml.gz").exists()]
    out = pd.DataFrame(rows).merge(idx[["pmcid", "doi", "retracted"]], on="pmcid")
    out.to_parquet(a.out, index=False)
    print(f"rebuilt {len(out)} of {len(idx)} articles -> {a.out}")


if __name__ == "__main__":
    main()
