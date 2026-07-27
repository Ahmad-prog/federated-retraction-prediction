"""Build the labeled corpus: retracted positives + matched controls.

Steps:
1. Load Retraction Watch CSV (data/raw/retraction_watch.csv).
2. Keep retractions of journal articles with a valid original-paper DOI.
3. Enrich positives via OpenAlex.
4. Sample matched controls (same venue, same year, not retracted) via the
   OpenAlex works endpoint, one control query per (venue_id, year) cell.
5. Save processed parquet: data/processed/corpus.parquet
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

from src.data.openalex import BASE, CACHE_DIR, MAILTO, fetch_works_by_doi, flatten

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "retraction_watch.csv"
OUT = ROOT / "data" / "processed"


def load_retraction_watch(limit: int | None = None) -> pd.DataFrame:
    rw = pd.read_csv(RAW, low_memory=False, encoding="utf-8", encoding_errors="replace")
    rw.columns = [c.strip() for c in rw.columns]
    # Journal-article retractions with an original DOI
    df = rw[rw["RetractionNature"].str.strip().str.lower() == "retraction"]
    df = df[df["ArticleType"].str.contains("Research Article", case=False, na=False)]
    df = df[df["OriginalPaperDOI"].notna() & (df["OriginalPaperDOI"].str.strip() != "") &
            (df["OriginalPaperDOI"].str.lower() != "unavailable")]
    df = df.drop_duplicates(subset="OriginalPaperDOI")
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=42)
    return df


SELECT_FIELDS = ",".join(
    [
        "id", "doi", "title", "publication_year", "publication_date", "type",
        "cited_by_count", "counts_by_year", "authorships", "primary_location",
        "open_access", "referenced_works_count", "primary_topic",
        "abstract_inverted_index",
    ]
)


def fetch_controls(cells: pd.DataFrame, per_cell: int, exclude_dois: set[str],
                   sleep: float = 0.15) -> pd.DataFrame:
    """For each (publisher_id, year) cell, random-sample non-retracted articles.

    Publisher-year matching (rather than venue-year) keeps the total request
    count within the OpenAlex free daily budget: ~n_cells * pages requests.
    """
    from concurrent.futures import ThreadPoolExecutor

    cache = CACHE_DIR / "controls_pub"
    cache.mkdir(parents=True, exist_ok=True)

    def get(params):
        for attempt in range(6):
            try:
                r = requests.get(BASE, params=params, timeout=60)
                if r.status_code == 200:
                    return r
                if r.status_code == 429:
                    time.sleep(10 * (attempt + 1))
                    continue
            except requests.RequestException:
                pass
            time.sleep(2**attempt)
        return None

    def fetch_cell(cell):
        pid = cell["publisher_id"].rsplit("/", 1)[-1]
        year = int(cell["year"])
        n_needed = int(cell["n"]) * per_cell
        sample = min(10000, max(25, int(n_needed * 1.1)))
        cfile = cache / f"{pid}_{year}.json"
        if cfile.exists():
            results = json.loads(cfile.read_text())
        else:
            results = []
            n_pages = (sample + 199) // 200
            for page in range(1, n_pages + 1):
                r = get({
                    "filter": (
                        f"primary_location.source.host_organization:{pid},"
                        f"publication_year:{year},type:article,is_retracted:false"
                    ),
                    "per-page": 200,
                    "sample": sample,
                    "seed": 42,
                    "page": page,
                    "select": SELECT_FIELDS,
                    "mailto": MAILTO,
                })
                if r is None:
                    break
                results.extend(r.json().get("results", []))
                time.sleep(sleep)
            if results:
                cfile.write_text(json.dumps(results))
        flat = [flatten(w) for w in results]
        return [f for f in flat if f["doi"] and f["doi"] not in exclude_dois][:n_needed]

    rows = []
    cell_dicts = cells.to_dict("records")
    with ThreadPoolExecutor(max_workers=4) as ex:
        for ci, flat in enumerate(ex.map(fetch_cell, cell_dicts)):
            if ci % 50 == 0:
                print(f"  control cell {ci}/{len(cell_dicts)}", flush=True)
            rows.extend(flat)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap number of positives (smoke)")
    ap.add_argument("--controls-per-positive", type=int, default=3)
    ap.add_argument("--min-cell", type=int, default=5,
                    help="min positives per (publisher, year) cell")
    args = ap.parse_args()

    rw = load_retraction_watch(args.limit)
    print(f"Retraction Watch positives with DOI: {len(rw)}")

    pos = fetch_works_by_doi(rw["OriginalPaperDOI"].tolist(), cache_name="positives")
    pos["retracted"] = 1
    print(f"Enriched positives via OpenAlex: {len(pos)}")

    cells = (
        pos.dropna(subset=["publisher_id", "year"])
        .groupby(["publisher_id", "year"]).size().rename("n").reset_index()
    )
    # drop sparse cells to stay within the OpenAlex free daily budget;
    # positives in dropped cells are excluded to preserve the matched design
    cells = cells[cells["n"] >= args.min_cell]
    kept = set(zip(cells["publisher_id"], cells["year"]))
    pos = pos[[(p, y) in kept for p, y in zip(pos["publisher_id"], pos["year"])]]
    print(f"positives after min-cell filter: {len(pos)}")
    est_requests = int(((cells["n"] * args.controls_per_positive * 1.1).clip(25, 10000) / 200)
                       .apply(lambda x: max(1, round(x + 0.5))).sum())
    print(f"control cells: {len(cells)}, estimated requests: {est_requests}")
    neg = fetch_controls(cells, args.controls_per_positive, set(pos["doi"]))
    neg["retracted"] = 0
    print(f"Matched controls: {len(neg)}")

    corpus = pd.concat([pos, neg], ignore_index=True)
    corpus = corpus.drop_duplicates(subset="doi")
    OUT.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(OUT / "corpus.parquet", index=False)
    print(f"Saved {len(corpus)} rows -> {OUT / 'corpus.parquet'}")
    print(corpus["retracted"].value_counts())


if __name__ == "__main__":
    main()
