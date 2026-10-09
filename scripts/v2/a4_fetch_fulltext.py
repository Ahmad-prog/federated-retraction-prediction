"""A4: download Europe PMC fullTextXML for every full-text positive and control.

Resumable; a 404 is recorded as <PMCID>.missing, any other failure raises and is
retried on the next run (never cached). Output: data_v2/fulltext_xml/<PMCID>.xml.gz
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.v2 import DATA2  # noqa: E402
from src.v2.epmc import fetch_fulltext_xml  # noqa: E402

OUT = DATA2 / "fulltext_xml"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ids = sorted(set(pd.read_parquet(DATA2 / "ft_positives.parquet")["pmcid"].dropna())
                 | set(pd.read_parquet(DATA2 / "ft_controls.parquet")["pmcid"].dropna()))
    print(f"PMCIDs to fetch: {len(ids)}", flush=True)
    status, t0 = Counter(), time.time()

    def one(pmcid):
        try:
            return fetch_fulltext_xml(pmcid, OUT)
        except Exception:  # noqa: BLE001
            return "error"

    with ThreadPoolExecutor(max_workers=16) as ex:
        for i, s in enumerate(ex.map(one, ids)):
            status[s] += 1
            if i % 2000 == 0:
                rate = (i + 1) / max(time.time() - t0, 1)
                print(f"  {i}/{len(ids)} {dict(status)} {rate:.1f}/s", flush=True)
    print(f"done: {dict(status)}")
    # a handful of persistent server errors must not block the pipeline forever:
    # retry (via the caller's loop) only while errors exceed 0.5% of the documents
    if status["error"] > 0.005 * len(ids):
        sys.exit(f"{status['error']} errors; rerun to resume")
    if status["error"]:
        print(f"WARNING: {status['error']} documents failed and are skipped")


if __name__ == "__main__":
    main()
