"""OpenAlex API client with local parquet caching.

Rate limits: polite pool (mailto) allows 10 req/s, 100k/day.
We batch-fetch works by DOI using the filter endpoint (50 DOIs per request).
"""
from __future__ import annotations

import os
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.openalex.org/works"
MAILTO = os.environ.get("OPENALEX_MAILTO", "")  # your e-mail for the OpenAlex polite pool
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"


def _batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def fetch_works_by_doi(dois: list[str], cache_name: str, sleep: float = 0.15) -> pd.DataFrame:
    """Fetch OpenAlex work records for a list of DOIs, caching per-batch JSON.

    Returns a flat DataFrame of the fields we need for features.
    """
    cache = CACHE_DIR / cache_name
    cache.mkdir(parents=True, exist_ok=True)
    dois = [d.strip().lower() for d in dois if isinstance(d, str) and d.strip()]
    batches = list(_batched(sorted(set(dois)), 50))

    def fetch_batch(job):
        bi, batch = job
        bfile = cache / f"batch_{bi:06d}.json"
        if bfile.exists():
            return json.loads(bfile.read_text())
        flt = "|".join(f"https://doi.org/{d}" for d in batch)
        params = {
            "filter": f"doi:{flt}",
            "per-page": 50,
            "mailto": MAILTO,
            "select": ",".join(
                [
                    "id", "doi", "title", "publication_year", "publication_date",
                    "type", "cited_by_count", "counts_by_year", "authorships",
                    "primary_location", "open_access", "referenced_works_count",
                    "primary_topic", "abstract_inverted_index",
                ]
            ),
        }
        for attempt in range(6):
            try:
                r = requests.get(BASE, params=params, timeout=60)
                if r.status_code == 200:
                    break
                if r.status_code == 429:
                    time.sleep(5 * (attempt + 1))
                    continue
            except requests.RequestException:
                pass
            time.sleep(2**attempt)
        r.raise_for_status()
        results = r.json().get("results", [])
        bfile.write_text(json.dumps(results))
        time.sleep(sleep)
        return results

    rows = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for bi, results in enumerate(ex.map(fetch_batch, enumerate(batches))):
            if bi % 100 == 0:
                print(f"  positives batch {bi}/{len(batches)}", flush=True)
            rows.extend(flatten(w) for w in results)
    return pd.DataFrame(rows)


def _abstract_from_inverted(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def flatten(w: dict) -> dict:
    loc = w.get("primary_location") or {}
    src = loc.get("source") or {}
    topic = w.get("primary_topic") or {}
    auths = w.get("authorships") or []
    countries = {c for a in auths for c in (a.get("countries") or [])}
    counts = {c["year"]: c["cited_by_count"] for c in (w.get("counts_by_year") or [])}
    year = w.get("publication_year")
    early = sum(v for y, v in counts.items() if year and y <= year + 2)
    return {
        "openalex_id": w.get("id"),
        "doi": (w.get("doi") or "").replace("https://doi.org/", "").lower(),
        "title": w.get("title") or "",
        "year": year,
        "pub_date": w.get("publication_date"),
        "type": w.get("type"),
        "cited_by_count": w.get("cited_by_count", 0),
        "early_citations_2y": early,
        "n_authors": len(auths),
        "n_countries": len(countries),
        "is_international": len(countries) > 1,
        "venue": src.get("display_name"),
        "venue_id": src.get("id"),
        "publisher": src.get("host_organization_name"),
        "publisher_id": src.get("host_organization"),
        "is_oa": (w.get("open_access") or {}).get("is_oa", False),
        "n_references": w.get("referenced_works_count", 0),
        "field": (topic.get("field") or {}).get("display_name"),
        "subfield": (topic.get("subfield") or {}).get("display_name"),
        "topic": topic.get("display_name"),
        "abstract": _abstract_from_inverted(w.get("abstract_inverted_index")),
    }
