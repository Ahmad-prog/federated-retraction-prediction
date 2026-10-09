"""A3: journal- and year-matched open-access controls for the full-text corpus.

Why: in v1, ~59% of retracted articles but only ~7% of controls have open-access
full text in Europe PMC, so a full-text subset would let a model learn "has OA full
text" instead of retraction. Here every retracted article with OA full text gets
CONTROLS_PER_POS controls from the SAME journal (ISSN) and publication year, drawn
from Europe PMC's OA full-text set, excluding every DOI that appears anywhere in
Retraction Watch (any notice nature; M12).

Sampling: if a journal-year cell has <= CAP hits we fetch all and sample uniformly;
otherwise we fetch per publication month (up to CAP/12 each) to avoid the ordering
bias of taking the first pages of a large result set.

Outputs: data_v2/ft_controls.parquet, data_v2/ft_positives.parquet,
         data_v2/ft_cells.parquet (per-cell counts and shortfalls).
"""
from __future__ import annotations

import calendar
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.v2 import DATA2, ROOT  # noqa: E402
from src.v2.epmc import cached_json, search  # noqa: E402

CONTROLS_PER_POS = 2
CAP = 5000
CACHE = DATA2 / "cache" / "pmc_controls"
CACHE.mkdir(parents=True, exist_ok=True)
FIELDS = ["doi", "pmid", "pmcid", "journalTitle", "journalIssn", "pubYear", "pubType",
          "firstPublicationDate", "isOpenAccess", "inEPMC"]
EXCLUDE_TYPES = ("retract", "erratum", "correction", "expression of concern", "review",
                 "editorial", "comment", "letter", "news", "published erratum")


def eligible(pub_type: str | None) -> bool:
    t = (pub_type or "").lower()
    return ("research-article" in t or "journal article" in t) and not any(
        x in t for x in EXCLUDE_TYPES)


def fetch_all(query: str, limit: int) -> tuple[int, list[dict]]:
    out, cursor, hits = [], "*", None
    while True:
        d = search(query, page_size=1000, cursor=cursor)
        hits = d["hitCount"] if hits is None else hits
        res = d["resultList"].get("result", [])
        out.extend({k: r.get(k) for k in FIELDS} for r in res)
        nxt = d.get("nextCursorMark")
        if not res or not nxt or nxt == cursor or len(out) >= limit:
            return hits, out[:limit]
        cursor = nxt


def cell_candidates(issn: str, year: int) -> list[dict]:
    base = f'ISSN:"{issn}" AND PUB_YEAR:{year} AND OPEN_ACCESS:Y AND IN_EPMC:Y'

    def fetch():
        hits, rows = fetch_all(base, CAP)
        if hits <= CAP:
            return rows
        rows, per_month = [], math.ceil(CAP / 12)
        for m in range(1, 13):
            last = calendar.monthrange(year, m)[1]
            q = (f"{base} AND FIRST_PDATE:[{year}-{m:02d}-01 TO {year}-{m:02d}-{last}]")
            rows.extend(fetch_all(q, per_month)[1])
        return rows

    return cached_json(CACHE / f"{issn}_{year}.json", fetch)


def main() -> None:
    m = pd.read_parquet(DATA2 / "epmc_map.parquet")
    corpus = pd.read_parquet(ROOT / "data/processed/corpus.parquet", columns=["doi", "retracted"])
    prosp = set(pd.read_parquet(DATA2 / "prospective_positives.parquet")["doi"])
    rw_all = set((DATA2 / "rw_all_dois.txt").read_text().split("\n"))

    pos_dois = set(corpus.loc[corpus["retracted"] == 1, "doi"]) | prosp
    pos = m[m["doi"].isin(pos_dois) & m["oa_fulltext"]].copy()
    pos["pubYear"] = pd.to_numeric(pos["pubYear"], errors="coerce")
    pos = pos.dropna(subset=["journalIssn", "pubYear"])
    pos["issn"] = pos["journalIssn"].str.split(";").str[0].str.strip()
    pos["split"] = np.where(pos["doi"].isin(prosp), "prospective", "main")
    pos.to_parquet(DATA2 / "ft_positives.parquet", index=False)

    cells = pos.groupby(["issn", "pubYear"]).size().rename("n_pos").reset_index()
    print(f"OA full-text positives: {len(pos)} "
          f"(main {int((pos.split == 'main').sum())}, prospective {int((pos.split == 'prospective').sum())}); "
          f"journal-year cells: {len(cells)}", flush=True)

    def do_cell(rec):
        issn, year, n_pos = rec["issn"], int(rec["pubYear"]), int(rec["n_pos"])
        try:
            cands = cell_candidates(issn, year)
        except Exception as e:  # noqa: BLE001
            return issn, year, n_pos, None, str(e)
        cands = [c for c in cands if c.get("doi") and eligible(c.get("pubType"))
                 and c["doi"].lower() not in rw_all and c["doi"].lower() not in pos_dois]
        return issn, year, n_pos, cands, None

    rng = np.random.default_rng(42)
    ctrl_rows, cell_rows, failed = [], [], 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        for i, (issn, year, n_pos, cands, err) in enumerate(
                ex.map(do_cell, cells.to_dict("records"))):
            if err:
                failed += 1
                print(f"  cell {issn} {year} FAILED: {err[:150]}", flush=True)
                continue
            need = CONTROLS_PER_POS * n_pos
            take = rng.choice(len(cands), size=min(need, len(cands)), replace=False) if cands else []
            for j in take:
                c = dict(cands[j])
                c.update(issn=issn, cell_year=year)
                ctrl_rows.append(c)
            cell_rows.append({"issn": issn, "year": year, "n_pos": n_pos,
                              "n_candidates": len(cands), "n_ctrl": len(take)})
            if i % 250 == 0:
                print(f"  {i}/{len(cells)} cells, {len(ctrl_rows)} controls", flush=True)

    ctrl = pd.DataFrame(ctrl_rows)
    ctrl["doi"] = ctrl["doi"].str.lower()
    ctrl = ctrl.drop_duplicates("doi")
    ctrl.to_parquet(DATA2 / "ft_controls.parquet", index=False)
    cdf = pd.DataFrame(cell_rows)
    cdf.to_parquet(DATA2 / "ft_cells.parquet", index=False)
    short = cdf[cdf["n_ctrl"] < CONTROLS_PER_POS * cdf["n_pos"]]
    print(f"\ncontrols: {len(ctrl)}; cells with shortfall: {len(short)} "
          f"(missing {int((CONTROLS_PER_POS * short.n_pos - short.n_ctrl).sum())} controls); "
          f"failed cells: {failed}")
    if failed:
        sys.exit(f"{failed} cells failed; rerun to resume")


if __name__ == "__main__":
    main()
