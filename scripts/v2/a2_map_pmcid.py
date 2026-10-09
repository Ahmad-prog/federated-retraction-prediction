"""A2: map every corpus DOI (v1 corpus + prospective positives) to Europe PMC.

For each DOI we record PMCID, open-access flag, full-text availability, journal ISSN,
publication year and publication types. Output: data_v2/epmc_map.parquet.
Cached per batch; failed batches raise and are never cached (bug B2).
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.v2 import DATA2, ROOT  # noqa: E402
from src.v2.epmc import cached_json, search  # noqa: E402

CACHE = DATA2 / "cache" / "epmc_map"
CACHE.mkdir(parents=True, exist_ok=True)
BATCH = 20
KEEP = ["doi", "pmid", "pmcid", "source", "journalTitle", "journalIssn", "pubYear",
        "pubType", "isOpenAccess", "inEPMC", "inPMC", "firstPublicationDate"]


def map_batch(job):
    bi, dois = job

    def fetch():
        q = " OR ".join(f'DOI:"{d}"' for d in dois)
        res = search(q, page_size=100)["resultList"].get("result", [])
        wanted = set(dois)
        return [{k: r.get(k) for k in KEEP} for r in res
                if (r.get("doi") or "").lower() in wanted]

    return cached_json(CACHE / f"b{bi:06d}.json", fetch)


def main() -> None:
    corpus = pd.read_parquet(ROOT / "data/processed/corpus.parquet", columns=["doi", "retracted"])
    prosp = pd.read_parquet(DATA2 / "prospective_positives.parquet", columns=["doi"])
    dois = sorted(set(corpus["doi"].dropna()) | set(prosp["doi"].dropna()))
    dois = [d for d in dois if d and '"' not in d]
    batches = [dois[i:i + BATCH] for i in range(0, len(dois), BATCH)]
    print(f"mapping {len(dois)} DOIs in {len(batches)} batches", flush=True)

    rows, failed = [], 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(map_batch, j) for j in enumerate(batches)]
        for i, f in enumerate(futs):
            try:
                rows.extend(f.result())
            except Exception as e:  # noqa: BLE001 - report and continue; rerun resumes
                failed += 1
                print(f"  batch {i} FAILED: {e}", flush=True)
            if i % 500 == 0:
                print(f"  {i}/{len(batches)} batches, {len(rows)} hits", flush=True)

    m = pd.DataFrame(rows)
    m["doi"] = m["doi"].str.lower()
    # prefer the record that carries a PMCID (MED/PMC) over preprint duplicates
    m["has_pmc"] = m["pmcid"].notna()
    m = m.sort_values("has_pmc", ascending=False).drop_duplicates("doi")
    m["oa_fulltext"] = (m["isOpenAccess"] == "Y") & (m["inEPMC"] == "Y") & m["has_pmc"]
    m.to_parquet(DATA2 / "epmc_map.parquet", index=False)

    lab = corpus.drop_duplicates("doi").merge(m[["doi", "oa_fulltext"]], on="doi", how="left")
    lab["oa_fulltext"] = lab["oa_fulltext"].fillna(False).astype(bool)
    print(f"\nfound in Europe PMC: {len(m)} / {len(dois)}; failed batches: {failed}")
    print("OA full text available, by class:")
    print(lab.groupby("retracted")["oa_fulltext"].agg(["sum", "mean"]).to_string())
    if failed:
        sys.exit(f"{failed} batches failed; rerun to resume")


if __name__ == "__main__":
    main()
