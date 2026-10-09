"""A6: assemble the full-text dataset (labels, silos, metadata) for the GPU track.

Fetches OpenAlex records for DOIs not already cached (full-text controls and
prospective positives; ~1 request per 50 DOIs), then joins:
  ft_positives / ft_controls  (Europe PMC: pmcid, issn, year)
  fulltext_clean              (parsed, leakage-cleaned text; notices dropped)
  OpenAlex                    (publisher, field, authors, countries, references)
Output: data_v2/ft_dataset.parquet
"""
from __future__ import annotations

import glob
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.v2 import DATA2, ROOT  # noqa: E402
from scripts.v2.b1_tabular_features import row as oa_row  # noqa: E402

OA = "https://api.openalex.org/works"
CACHE = DATA2 / "cache" / "openalex_ft"
MIN_BODY_WORDS = 200
CACHE.mkdir(parents=True, exist_ok=True)
SELECT = ("id,doi,title,publication_year,publication_date,cited_by_count,counts_by_year,"
          "authorships,primary_location,open_access,referenced_works_count,primary_topic,"
          "abstract_inverted_index")


def fetch_batch(job):
    bi, dois = job
    f = CACHE / f"b{bi:05d}.json"
    if f.exists():
        return json.loads(f.read_text())
    params = {"filter": "doi:" + "|".join(dois), "per-page": 50, "select": SELECT}
    for attempt in range(6):
        try:
            r = requests.get(OA, params=params, timeout=90)
            if r.status_code == 200:
                res = r.json().get("results", [])
                f.write_text(json.dumps(res))
                return res
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError(f"OpenAlex batch {bi} failed")


def main() -> None:
    pos = pd.read_parquet(DATA2 / "ft_positives.parquet")
    ctl = pd.read_parquet(DATA2 / "ft_controls.parquet")
    pos = pos.assign(retracted=1)[["doi", "pmcid", "issn", "pubYear", "split", "retracted"]]
    ctl = ctl.assign(retracted=0, split="main")
    ctl = ctl.rename(columns={"cell_year": "pubYear_cell"})[
        ["doi", "pmcid", "issn", "pubYear", "split", "retracted"]]
    # controls matched to prospective positives are part of the prospective test set
    pros_cells = set(zip(pos.loc[pos.split == "prospective", "issn"],
                         pos.loc[pos.split == "prospective", "pubYear"].astype(int)))
    ctl["pubYear"] = pd.to_numeric(ctl["pubYear"], errors="coerce")
    main_cells = set(zip(pos.loc[pos.split == "main", "issn"],
                         pos.loc[pos.split == "main", "pubYear"].astype(int)))
    only_pros = [(i, int(y)) in pros_cells and (i, int(y)) not in main_cells
                 if pd.notna(y) else False for i, y in zip(ctl["issn"], ctl["pubYear"])]
    ctl.loc[only_pros, "split"] = "prospective"
    ds = pd.concat([pos, ctl], ignore_index=True).drop_duplicates("doi")

    wanted, cached = set(ds["doi"]), {}
    for sub in ["positives", "controls_pub"]:
        for f in glob.glob(str(ROOT / "data" / "cache" / sub / "*.json")):
            for w in json.loads(Path(f).read_text()):
                d = (w.get("doi") or "").replace("https://doi.org/", "").lower()
                if d in wanted:
                    cached[d] = w
    need = sorted(set(ds["doi"]) - set(cached))
    print(f"full-text docs: {len(ds)}; OpenAlex records to fetch: {len(need)}", flush=True)
    batches = list(enumerate([need[i:i + 50] for i in range(0, len(need), 50)]))
    with ThreadPoolExecutor(max_workers=4) as ex:
        for res in ex.map(fetch_batch, batches):
            for w in res:
                cached[(w.get("doi") or "").replace("https://doi.org/", "").lower()] = w

    meta = pd.DataFrame([oa_row(w) for w in cached.values()]).drop_duplicates("doi")
    meta = meta.drop(columns=["cby", "raw_abstract"], errors="ignore")
    ft = pd.read_parquet(DATA2 / "fulltext_clean.parquet")
    # stub bodies (< MIN_BODY_WORDS after cleaning) are notice/footnote shells, 95% retracted
    ft = ft[ft["parse_ok"] & ~ft["is_notice"] & (ft["n_body_words"] >= MIN_BODY_WORDS)]
    out = ds.merge(ft, on="pmcid", how="inner").merge(meta, on="doi", how="left",
                                                      suffixes=("", "_oa"))
    out.to_parquet(DATA2 / "ft_dataset.parquet", index=False)
    print(f"ft_dataset: {len(out)} rows; by split x label:")
    print(out.groupby(["split", "retracted"]).size().to_string())
    print(f"publisher coverage: {out['publisher'].notna().mean():.3f}")
    print(out["publisher"].value_counts().head(12).to_string())


if __name__ == "__main__":
    main()
