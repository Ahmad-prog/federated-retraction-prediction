"""Fetch referenced_works (reference lists) for every corpus DOI from OpenAlex.

~138k DOIs -> ~2,800 batched requests; the free daily budget allows ~1k-1.5k,
so this script is designed to be rerun daily: cached batches are skipped, and
on sustained 429s it exits cleanly (rerun after midnight UTC resumes).
"""
from __future__ import annotations

import os
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE = "https://api.openalex.org/works"
MAILTO = os.environ.get("OPENALEX_MAILTO", "")  # your e-mail for the OpenAlex polite pool
CACHE = ROOT / "data/cache/references"
CACHE.mkdir(parents=True, exist_ok=True)

budget_exhausted = False


def fetch_batch(job):
    global budget_exhausted
    bi, batch = job
    bfile = CACHE / f"batch_{bi:06d}.json"
    if bfile.exists() or budget_exhausted:
        return 0
    flt = "|".join(f"https://doi.org/{d}" for d in batch)
    params = {"filter": f"doi:{flt}", "per-page": 50, "mailto": MAILTO,
              "select": "id,doi,referenced_works"}
    for attempt in range(4):
        try:
            r = requests.get(BASE, params=params, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            bfile.write_text(json.dumps(r.json().get("results", [])))
            time.sleep(0.12)
            return 1
        if r.status_code == 429:
            body = r.text[:200]
            if "budget" in body.lower() or "Insufficient" in body:
                budget_exhausted = True
                return 0
            time.sleep(10 * (attempt + 1))
        else:
            time.sleep(2 ** attempt)
    return 0


def main() -> None:
    df = pd.read_parquet(ROOT / "data/processed/corpus.parquet")
    dois = sorted({d for d in df["doi"] if isinstance(d, str) and d})
    batches = [dois[i:i + 50] for i in range(0, len(dois), 50)]
    done_before = sum(1 for i in range(len(batches))
                      if (CACHE / f"batch_{i:06d}.json").exists())
    print(f"{len(dois)} DOIs, {len(batches)} batches, {done_before} already cached")
    fetched = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        for n in ex.map(fetch_batch, enumerate(batches)):
            fetched += n
            if fetched and fetched % 200 == 0:
                print(f"  fetched {fetched} new batches", flush=True)
    done_after = sum(1 for i in range(len(batches))
                     if (CACHE / f"batch_{i:06d}.json").exists())
    print(f"cached {done_after}/{len(batches)} batches"
          + (" (budget exhausted, rerun after midnight UTC)" if budget_exhausted else ""))
    if done_after == len(batches):
        print("ALL_REFERENCES_FETCHED")


if __name__ == "__main__":
    main()
